"""Repository-research task executor owned by the independent Harness Worker."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.harness.context import RuntimeContext
from nanobot.harness.models import AgentSpec, ReviewResult
from nanobot.repo_agent.executor import RepositoryAgentRun
from nanobot.security.logical_vfs import VFSIdentity
from nanobot.utils.llm_runtime import LLMRuntime

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.harness.worker import TaskExecutor


class RepositoryResearchPoolExecutor:
    """Route repository research to a Worker process, retaining normal tasks.

    A Worker owns the complete persisted research state and never receives the
    parent conversation transcript. The Gateway only creates the POOL task.
    Existing non-repository tasks continue through ``fallback`` (normally the
    bwrap child executor).
    """

    isolation_mode = "worker_process_repository_research"

    def __init__(self, loop: AgentLoop, fallback: TaskExecutor) -> None:
        self.loop = loop
        self.fallback = fallback

    async def __call__(
        self,
        context: RuntimeContext,
        spec: AgentSpec,
    ) -> str | ReviewResult:
        if context.task_metadata.get("domain") != "repository_research":
            return await self.fallback(context, spec)
        if context.task_metadata.get("operation") != "run":
            raise RuntimeError("unsupported repository research pool operation")
        return await self._run_research(context, spec)

    async def _run_research(self, context: RuntimeContext, spec: AgentSpec) -> str:
        orchestrator = self.loop.repository_orchestrator
        if orchestrator is None:
            raise RuntimeError("repository agent is disabled in Harness Worker")
        research_id = context.task_metadata.get("research_id")
        if not isinstance(research_id, str) or not research_id:
            raise RuntimeError("repository research task is missing research_id")
        task = orchestrator.store.load_research(research_id)
        if task is None:
            raise RuntimeError(f"repository research state {research_id!r} was not found")
        attempt = 1
        try:
            harness_task = await self.loop.harness.store.get_task(context.task_id)
            attempt = max(1, int(harness_task.attempt))
        except Exception:
            # The durable research record remains the source of truth if a
            # custom HarnessStore does not expose task lookup.
            pass
        actor = str(context.task_metadata.get("actor") or task.actor)
        session_key = str(context.task_metadata.get("session_key") or task.session_key)
        tenant_id = str(context.task_metadata.get("tenant_id") or context.tenant_id)

        async def on_update(content: str, **detail: Any) -> None:
            await self.loop.harness.store.record_audit_event(
                "repository_research.progress",
                component="repository_research_worker",
                tenant_id=tenant_id,
                workflow_id=context.workflow_id,
                task_id=context.task_id,
                agent_id=context.agent_id,
                session_key=session_key,
                trace_id=task.trace_id,
                payload={
                    "research_id": task.research_id,
                    "status": task.status.value,
                    "content": content,
                    "detail": detail,
                    "attempt": attempt,
                },
            )

        async def on_report_stream(content: str) -> None:
            """Persist synthesis deltas so Gateway can relay them immediately."""
            if not content:
                return
            await self.loop.harness.store.record_audit_event(
                "repository_research.report_delta",
                component="repository_research_worker",
                tenant_id=tenant_id,
                workflow_id=context.workflow_id,
                task_id=context.task_id,
                agent_id=context.agent_id,
                session_key=session_key,
                trace_id=task.trace_id,
                payload={
                    "research_id": task.research_id,
                    "status": task.status.value,
                    "content": content,
                    "attempt": attempt,
                },
            )

        identity = VFSIdentity(
            tenant_id=tenant_id,
            user_id=actor or session_key,
            session_id=session_key,
            task_id=context.task_id,
            agent_id=context.agent_id,
        )
        manager = orchestrator.manager
        vfs_scope = manager.vfs.bind(identity) if manager.vfs and manager.vfs.enabled else nullcontext()
        with vfs_scope, manager.bind_vfs_identity(identity):
            if attempt > 1:
                await on_update(
                    f"上一轮研究尝试未完成；Worker 正在进行第 {attempt} 次尝试，"
                    "将重新检索并分析候选仓库。"
                )
            result = await orchestrator.research.run(
                direction=task.direction,
                actor=actor,
                session_key=session_key,
                runtime=self.loop.llm_runtime(),
                invoke=lambda *args, **kwargs: self._invoke_repository_agent(
                    *args,
                    worker_context=context,
                    worker_spec=spec,
                    **kwargs,
                ),
                on_update=on_update,
                on_report_stream=on_report_stream,
                existing_task=task,
                use_harness=False,
            )
        report = result.report or "研究流程已完成，但没有生成报告。"
        await self.loop.harness.store.record_audit_event(
            "repository_research.completed",
            component="repository_research_worker",
            tenant_id=tenant_id,
            workflow_id=context.workflow_id,
            task_id=context.task_id,
            agent_id=context.agent_id,
            session_key=session_key,
            trace_id=result.trace_id,
            payload={"research_id": result.research_id, "status": result.status.value, "report": report},
        )
        return report

    async def _invoke_repository_agent(
        self,
        messages: list[dict[str, object]],
        *,
        runtime: LLMRuntime,
        tools: ToolRegistry,
        max_iterations: int | None,
        ephemeral: bool,
        stream: bool,
        workspace: Path,
        repository_full_name: str,
        worker_context: RuntimeContext,
        worker_spec: AgentSpec,
    ) -> RepositoryAgentRun:
        del stream
        metadata: dict[str, Any] = {
            "repository_full_name": repository_full_name,
            "repository_workspace": str(workspace),
            "domain": "repository_research",
            "task_id": worker_context.task_id,
        }
        request_context = RequestContext(
            channel="harness",
            chat_id=f"research:{repository_full_name}",
            session_key=f"harness:research:{repository_full_name}",
            original_user_text="Repository research source analysis",
            runtime=runtime,
            metadata=metadata,
            sender_id="harness-worker",
            workspace=workspace,
            tenant_id=worker_context.tenant_id,
            user_id=str(worker_context.task_metadata.get("user_id") or "harness-worker"),
        )
        # The durable Worker process owns the research state. Source analysis
        # itself is additionally run in the configured bwrap child when that
        # backend is enabled, so repository code and tool execution do not run
        # in the Worker control-plane process.
        if getattr(self.fallback, "isolation_mode", "") == "bwrap_process":
            projected = RuntimeContext(
                workflow_id=worker_context.workflow_id,
                task_id=worker_context.task_id,
                tenant_id=worker_context.tenant_id,
                agent_id=worker_context.agent_id,
                role=worker_spec.role.value,
                objective=worker_spec.objective,
                task_input="\n\n".join(
                    str(message.get("content", "")) for message in messages
                ),
                task_metadata={
                    **worker_context.task_metadata,
                    **metadata,
                    "repository_workspace": str(workspace),
                },
                checkpoint=None,
                dependency_results=[],
                messages=[],
            )
            isolated = await self.fallback(projected, worker_spec)
            if isinstance(isolated, ReviewResult):
                raise RuntimeError("repository analysis returned an unexpected review result")
            return RepositoryAgentRun(
                final_content=isolated,
                tools_used=list(worker_spec.allowed_tools),
                messages=messages,
                stop_reason="completed",
            )
        content, used, all_messages, stop_reason, had_injections = await self.loop._run_agent_loop(
            messages,
            runtime=runtime,
            session=None,
            channel="harness",
            chat_id=request_context.chat_id,
            metadata=metadata,
            session_key=request_context.session_key,
            original_user_text=request_context.original_user_text,
            ephemeral=ephemeral,
            tools=tools,
            request_context=request_context,
            max_iterations=max_iterations,
            workspace_override=workspace,
        )
        return RepositoryAgentRun(
            final_content=content or "",
            tools_used=used,
            messages=all_messages,
            stop_reason=stop_reason,
            had_injections=had_injections,
        )
