"""LLM/tool executor used by independently deployed Harness workers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.runner import AgentRunSpec
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.harness.context import RuntimeContext
from nanobot.harness.models import AgentSpec, ReviewResult
from nanobot.security.workspace_access import (
    WorkspaceScope,
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


class _ToolCallBudget(AgentHook):
    def __init__(self, maximum: int | None) -> None:
        super().__init__()
        self.maximum = maximum
        self.used = 0

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        self.used += len(context.tool_calls)
        if self.maximum is not None and self.used > self.maximum:
            raise RuntimeError(f"Harness tool-call budget exceeded ({self.used}/{self.maximum})")


class HarnessAgentExecutor:
    """Execute a leased task using an AgentLoop's provider and tool catalog."""

    def __init__(self, loop: AgentLoop) -> None:
        self.loop = loop

    def _workspace(self, context: RuntimeContext) -> tuple[Path, WorkspaceScope]:
        default_scope = self.loop.workspace_scopes.default()
        raw = context.task_metadata.get("repository_workspace")
        workspace = Path(raw).expanduser().resolve() if isinstance(raw, str) else default_scope.project_path
        root = default_scope.project_path.resolve()
        if default_scope.restrict_to_workspace and workspace != root and root not in workspace.parents:
            raise PermissionError("Harness task workspace is outside the configured workspace")
        scope = build_workspace_scope(workspace, default_scope.access_mode, source_channel="harness")
        return workspace, scope

    async def __call__(self, context: RuntimeContext, spec: AgentSpec) -> str | ReviewResult:
        workspace, scope = self._workspace(context)
        runtime = self.loop.llm_runtime()
        tools = self.loop.tools.subset(tuple(spec.allowed_tools))
        system = render_template(
            "agent/subagent_system.md",
            role=spec.role.value,
            objective=spec.objective,
            review_gate=context.review_gate,
            workspace=str(workspace),
            agent_workspace=str(Path(self.loop.workspace).resolve()),
            skills_summary="",
        )
        request_token = bind_request_context(
            RequestContext(
                channel="harness",
                chat_id=context.agent_id,
                session_key=f"harness:{context.agent_id}",
                runtime=runtime,
                metadata=dict(context.task_metadata),
                workspace=workspace,
                tenant_id=context.tenant_id,
                user_id=str(context.task_metadata.get("user_id") or context.agent_id),
            )
        )
        workspace_token = bind_workspace_scope(scope)
        try:
            result = await self.loop.subagents.runner.run(
                AgentRunSpec(
                    initial_messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": context.render()},
                    ],
                    tools=tools,
                    runtime=runtime,
                    max_iterations=spec.resource_budget.max_iterations,
                    max_tool_result_chars=self.loop.max_tool_result_chars,
                    hook=_ToolCallBudget(spec.resource_budget.max_tool_calls),
                    max_iterations_message="Task ended before a final response was generated.",
                    finalize_on_max_iterations=False,
                    fail_on_tool_error=True,
                    workspace=workspace,
                    session_key=f"harness:{context.agent_id}",
                    provider_retry_mode=self.loop.provider_retry_mode,
                )
            )
        finally:
            reset_workspace_scope(workspace_token)
            reset_request_context(request_token)
        if result.stop_reason in {"error", "tool_error", "max_iterations"}:
            raise RuntimeError(result.error or f"Harness agent stopped: {result.stop_reason}")
        content = result.final_content or ""
        if context.review_gate:
            return ReviewResult.model_validate_json(content)
        return content
