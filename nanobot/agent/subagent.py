"""Subagent manager for background task execution."""

import asyncio
import hashlib
import json
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypedDict

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec
from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import (
    RequestContext,
    ToolContext,
    bind_request_context,
    reset_request_context,
)
from nanobot.agent.tools.exec_session import ExecSessionManager
from nanobot.agent.tools.file_state import FileStates
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults, ToolsConfig
from nanobot.harness import (
    AgentInstance,
    AgentRole,
    HarnessConflictError,
    HarnessService,
    ReviewResult,
    RuntimeContext,
    Task,
    TaskExecutionMode,
    TaskState,
    WorkerLease,
)
from nanobot.providers.base import LLMProvider
from nanobot.security.workspace_access import (
    WorkspaceScope,
    bind_workspace_scope,
    reset_workspace_scope,
    workspace_sandbox_status,
)
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template


class _SubagentOrigin(TypedDict):
    channel: str
    chat_id: str
    session_key: str | None
    tenant_id: str
    user_id: str


@dataclass(slots=True)
class SubagentStatus:
    """Real-time status of a running subagent."""

    task_id: str
    label: str
    task_description: str
    started_at: float  # time.monotonic()
    phase: str = "initializing"  # initializing | awaiting_tools | tools_completed | final_response | done | error
    iteration: int = 0
    tool_events: list[dict[str, str]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    error: str | None = None
    harness_agent_id: str | None = None
    harness_task_id: str | None = None
    role: str = AgentRole.WORKER.value


@dataclass(slots=True)
class _HarnessRun:
    task: Task
    supervisor: AgentInstance
    worker: AgentInstance
    lease: WorkerLease
    lease_ttl_seconds: int
    context: RuntimeContext | None = None


class _SubagentHook(AgentHook):
    """Hook for subagent execution — logs tool calls and updates status."""

    def __init__(
        self,
        task_id: str,
        status: SubagentStatus | None = None,
        *,
        max_tool_calls: int | None = None,
    ) -> None:
        super().__init__()
        self._task_id = task_id
        self._status = status
        self._max_tool_calls = max_tool_calls
        self._tool_calls = 0

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        self._tool_calls += len(context.tool_calls)
        if self._max_tool_calls is not None and self._tool_calls > self._max_tool_calls:
            raise RuntimeError(
                f"subagent tool-call budget exceeded ({self._tool_calls}/{self._max_tool_calls})"
            )
        for tool_call in context.tool_calls:
            args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
            logger.debug(
                "Subagent [{}] executing: {} with arguments: {}",
                self._task_id,
                tool_call.name,
                args_str,
            )

    async def after_iteration(self, context: AgentHookContext) -> None:
        if self._status is None:
            return
        self._status.iteration = context.iteration
        self._status.tool_events = list(context.tool_events)
        self._status.usage = dict(context.usage)
        if context.error:
            self._status.error = str(context.error)


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        workspace: Path | None = None,
        bus: MessageBus | None = None,
        max_tool_result_chars: int | None = None,
        model: str | None = None,
        tools_config: ToolsConfig | None = None,
        restrict_to_workspace: bool = False,
        disabled_skills: list[str] | None = None,
        max_iterations: int | None = None,
        max_concurrent_subagents: int | None = None,
        fail_on_tool_error: bool | None = None,
        llm_wall_timeout_for_session: Callable[[str | None], float | None] | None = None,
        harness: HarnessService | None = None,
    ):
        if workspace is None:
            raise TypeError("SubagentManager.__init__() missing required argument: 'workspace'")
        if bus is None:
            raise TypeError("SubagentManager.__init__() missing required argument: 'bus'")
        if max_tool_result_chars is None:
            raise TypeError(
                "SubagentManager.__init__() missing required argument: 'max_tool_result_chars'"
            )
        if model is not None and provider is None:
            raise TypeError("SubagentManager model compatibility argument requires provider")

        defaults = AgentDefaults()
        self._compat_runtime: LLMRuntime | None = None
        if provider is not None:
            warnings.warn(
                "SubagentManager provider/model constructor arguments are deprecated; "
                "pass runtime=... to spawn() instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self._compat_runtime = LLMRuntime.capture(
                provider,
                model or provider.get_default_model(),
                context_window_tokens=defaults.context_window_tokens,
            )
        self.workspace = workspace
        self.bus = bus
        self.tools_config = tools_config or ToolsConfig()
        self.max_tool_result_chars = max_tool_result_chars
        self.restrict_to_workspace = restrict_to_workspace
        self.disabled_skills = set(disabled_skills or [])
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.max_concurrent_subagents = (
            max_concurrent_subagents
            if max_concurrent_subagents is not None
            else defaults.max_concurrent_subagents
        )
        self.fail_on_tool_error = (
            fail_on_tool_error if fail_on_tool_error is not None else defaults.fail_on_tool_error
        )
        self.runner = AgentRunner()
        self._exec_session_manager = ExecSessionManager()
        self._llm_wall_timeout_for_session = llm_wall_timeout_for_session
        self._running_tasks: dict[str, asyncio.Task[str]] = {}
        self._task_statuses: dict[str, SubagentStatus] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}
        self.harness = harness or HarnessService()
        self._harness_runs: dict[str, _HarnessRun] = {}

    async def _admit_harness_run(
        self,
        status: SubagentStatus,
        task: str,
        *,
        scope_key: str,
        role: AgentRole,
        tenant_id: str = "default",
        workspace_root: Path | None = None,
        exclusive_workspace: bool = False,
        origin_message_id: str | None = None,
    ) -> tuple[bool, str | None]:
        """Declare the supervisor/worker boundary before executing a subagent."""
        await self.harness.flush_outbox()
        workflow, supervisor, _ = await self.harness.get_or_create_workflow(
            scope_key=scope_key,
            goal=task,
            tenant_id=tenant_id,
        )
        idempotency_key = None
        if origin_message_id:
            digest = hashlib.sha256(
                f"{origin_message_id}\0{role.value}\0{task}".encode()
            ).hexdigest()
            idempotency_key = f"spawn:{digest}"
        harness_task, created = await self.harness.ensure_task(
            workflow,
            title=status.label,
            task_input=task,
            execution_mode=TaskExecutionMode.INLINE,
            required_role=role,
            resource_keys=(
                frozenset({f"workspace:{workspace_root.resolve()}"})
                if role == AgentRole.WORKER
                and workspace_root is not None
                and exclusive_workspace
                else frozenset()
            ),
            idempotency_key=idempotency_key,
        )
        status.harness_task_id = harness_task.task_id
        status.role = role.value
        if not created:
            if harness_task.state in {TaskState.LEASED, TaskState.RUNNING}:
                return False, f"Subagent [{status.label}] is already running."
            if harness_task.state == TaskState.SUCCEEDED:
                return False, harness_task.result or "Task already completed."
            if harness_task.state in {TaskState.FAILED, TaskState.CANCELLED}:
                return False, harness_task.error or f"Task is {harness_task.state.value}."
        worker = await self.harness.spawn_agent(workflow, supervisor, role)
        await self.harness.scheduler.ready_tasks(workflow.workflow_id)
        spec = await self.harness.store.get_spec(worker.spec_id)
        lease_ttl = max(60, spec.resource_budget.max_runtime_seconds)
        lease = await self.harness.scheduler.assign(
            task=harness_task,
            agent=worker,
            spec=spec,
            worker_id=f"asyncio:{status.task_id}",
            ttl_seconds=lease_ttl,
        )
        status.harness_agent_id = worker.instance_id
        self._harness_runs[status.task_id] = _HarnessRun(
            task=harness_task,
            supervisor=supervisor,
            worker=worker,
            lease=lease,
            lease_ttl_seconds=lease_ttl,
        )
        return True, None

    def set_provider(self, provider: LLMProvider, model: str) -> None:
        """Update the deprecated runtime source used by legacy ``spawn`` calls."""
        warnings.warn(
            "SubagentManager.set_provider() is deprecated; pass runtime=... to spawn() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        context_window_tokens = (
            self._compat_runtime.context_window_tokens
            if self._compat_runtime is not None
            else AgentDefaults().context_window_tokens
        )
        self._compat_runtime = LLMRuntime.capture(
            provider,
            model,
            context_window_tokens=context_window_tokens,
        )

    def _compat_spawn_runtime(self) -> LLMRuntime:
        runtime = self._compat_runtime
        if runtime is None:
            raise TypeError(
                "SubagentManager.spawn() missing required keyword-only argument: 'runtime'"
            )
        warnings.warn(
            "SubagentManager.spawn() without runtime is deprecated; pass runtime=... explicitly",
            DeprecationWarning,
            stacklevel=3,
        )
        return LLMRuntime.capture(
            runtime.provider,
            runtime.model,
            context_window_tokens=runtime.context_window_tokens,
        )

    def _subagent_tools_config(self) -> ToolsConfig:
        """Build a ToolsConfig scoped for subagent use."""
        return ToolsConfig(
            exec=self.tools_config.exec,
            web=self.tools_config.web,
            file=self.tools_config.file,
            # Subagents are always capability-confined even when the main agent
            # is intentionally running without a workspace restriction.
            restrict_to_workspace=True,
        )

    def _build_tools(
        self,
        workspace: Path | None = None,
        tools_config: ToolsConfig | None = None,
    ) -> ToolRegistry:
        """Build an isolated subagent tool registry via ToolLoader."""
        root = self.workspace if workspace is None else workspace
        registry = ToolRegistry()
        cfg = tools_config if tools_config is not None else self._subagent_tools_config()
        ctx = ToolContext(
            config=cfg,
            workspace=str(root.resolve()),
            exec_session_manager=self._exec_session_manager,
            file_state_store=FileStates(),
            workspace_sandbox=workspace_sandbox_status(
                restrict_to_workspace=cfg.restrict_to_workspace,
                workspace=root,
            ),
            allow_history_read=False,
        )
        ToolLoader().load(ctx, registry, scope="subagent")
        return registry

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        origin_message_id: str | None = None,
        temperature: float | None = None,
        role: str = AgentRole.WORKER.value,
        tenant_id: str = "default",
        user_id: str = "default",
        exclusive_workspace: bool = False,
        workspace_scope: WorkspaceScope | None = None,
        *,
        runtime: LLMRuntime | None = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        if runtime is None:
            runtime = self._compat_spawn_runtime()
        if temperature is not None:
            runtime = runtime.with_generation_overrides(temperature=temperature)
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin: _SubagentOrigin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
            "session_key": session_key,
            "tenant_id": tenant_id,
            "user_id": user_id,
        }

        agent_role = AgentRole(role)
        status = SubagentStatus(
            task_id=task_id,
            label=display_label,
            task_description=task,
            started_at=time.monotonic(),
        )
        admitted, replay = await self._admit_harness_run(
            status,
            task,
            scope_key=session_key or f"{origin_channel}:{origin_chat_id}",
            role=agent_role,
            tenant_id=tenant_id,
            workspace_root=(
                workspace_scope.project_path if workspace_scope is not None else self.workspace
            ),
            exclusive_workspace=exclusive_workspace,
            origin_message_id=origin_message_id,
        )
        if not admitted:
            return replay or f"Subagent [{display_label}] request was already accepted."
        self._task_statuses[task_id] = status

        bg_task = asyncio.create_task(
            self._run_subagent(
                task_id,
                task,
                display_label,
                origin,
                status,
                runtime,
                origin_message_id,
                tenant_id,
                exclusive_workspace,
                workspace_scope,
            )
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task[str]) -> None:
            self._running_tasks.pop(task_id, None)
            self._task_statuses.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def run_inline(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        origin_message_id: str | None = None,
        temperature: float | None = None,
        role: str = AgentRole.WORKER.value,
        tenant_id: str = "default",
        user_id: str = "default",
        exclusive_workspace: bool = False,
        workspace_scope: WorkspaceScope | None = None,
        *,
        runtime: LLMRuntime | None = None,
    ) -> str:
        """Run a subagent synchronously and return its result to the caller."""
        if runtime is None:
            runtime = self._compat_spawn_runtime()
        if temperature is not None:
            runtime = runtime.with_generation_overrides(temperature=temperature)
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin: _SubagentOrigin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
            "session_key": session_key,
            "tenant_id": tenant_id,
            "user_id": user_id,
        }
        agent_role = AgentRole(role)
        status = SubagentStatus(
            task_id=task_id,
            label=display_label,
            task_description=task,
            started_at=time.monotonic(),
        )
        admitted, replay = await self._admit_harness_run(
            status,
            task,
            scope_key=session_key or f"{origin_channel}:{origin_chat_id}",
            role=agent_role,
            tenant_id=tenant_id,
            workspace_root=(
                workspace_scope.project_path if workspace_scope is not None else self.workspace
            ),
            exclusive_workspace=exclusive_workspace,
            origin_message_id=origin_message_id,
        )
        if not admitted:
            return replay or "Task request was already accepted."
        self._task_statuses[task_id] = status
        logger.info("Running inline subagent [{}]: {}", task_id, display_label)
        inline_task = asyncio.create_task(
            self._run_subagent(
                task_id,
                task,
                display_label,
                origin,
                status,
                runtime,
                origin_message_id,
                tenant_id,
                exclusive_workspace,
                workspace_scope,
                announce=False,
            )
        )
        self._running_tasks[task_id] = inline_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)
        try:
            result = await inline_task
            if status.phase == "error" or status.stop_reason in {"error", "tool_error"}:
                return ToolResult.error(result)
            return result
        finally:
            self._running_tasks.pop(task_id, None)
            self._task_statuses.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: _SubagentOrigin,
        status: SubagentStatus,
        runtime: LLMRuntime,
        origin_message_id: str | None = None,
        tenant_id: str = "default",
        exclusive_workspace: bool = False,
        workspace_scope: WorkspaceScope | None = None,
        *,
        announce: bool = True,
    ) -> str:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)
        harness_run = self._harness_runs.get(task_id)
        if harness_run is None:
            admitted, replay = await self._admit_harness_run(
                status,
                task,
                scope_key=origin.get("session_key") or f"{origin['channel']}:{origin['chat_id']}",
                role=AgentRole(status.role),
                tenant_id=tenant_id,
                workspace_root=(
                    workspace_scope.project_path
                    if workspace_scope is not None
                    else self.workspace
                ),
                exclusive_workspace=exclusive_workspace,
                origin_message_id=origin_message_id,
            )
            if not admitted:
                return replay or "Task request was already accepted."
            harness_run = self._harness_runs[task_id]
        heartbeat_stop = asyncio.Event()
        heartbeat_task: asyncio.Task[None] | None = None
        terminal_recorded = False

        async def _on_checkpoint(payload: dict[str, Any]) -> None:
            status.phase = payload.get("phase", status.phase)
            status.iteration = payload.get("iteration", status.iteration)
            if run := self._harness_runs.get(task_id):
                previous = await self.harness.store.latest_checkpoint(run.worker.instance_id)
                sequence = 0 if previous is None else previous.sequence + 1
                await self.harness.scheduler.checkpoint(run.worker, payload, sequence)

        async def _heartbeat(run: _HarnessRun) -> None:
            interval = max(5.0, min(30.0, run.lease_ttl_seconds / 3))
            while True:
                try:
                    await asyncio.wait_for(heartbeat_stop.wait(), timeout=interval)
                    return
                except TimeoutError:
                    run.lease = await self.harness.scheduler.heartbeat(
                        run.lease,
                        ttl_seconds=run.lease_ttl_seconds,
                    )

        try:
            heartbeat_task = asyncio.create_task(_heartbeat(harness_run))
            root = workspace_scope.project_path if workspace_scope is not None else self.workspace
            cfg = None
            if workspace_scope is not None:
                cfg = self._subagent_tools_config()
            # Construct from the agent workspace; the bound scope below supplies the project cwd.
            spec = await self.harness.store.get_spec(harness_run.worker.spec_id)
            tools = self._build_tools(tools_config=cfg).subset(tuple(spec.allowed_tools))
            runtime_context = await self.harness.runtime_context(
                harness_run.worker.instance_id,
                harness_run.task.task_id,
                consumer_id=f"asyncio:{task_id}",
            )
            harness_run.context = runtime_context
            system_prompt = self._build_subagent_prompt(
                workspace=root,
                role=spec.role,
                objective=spec.objective,
                review_gate=harness_run.task.review_target_task_id is not None,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": runtime_context.render()},
            ]

            sess_key = origin.get("session_key")
            llm_timeout = (
                self._llm_wall_timeout_for_session(sess_key)
                if self._llm_wall_timeout_for_session
                else None
            )
            request_token = bind_request_context(
                RequestContext(
                    channel=origin["channel"],
                    chat_id=origin["chat_id"],
                    message_id=origin_message_id,
                    session_key=sess_key,
                    runtime=runtime,
                    tenant_id=str(origin.get("tenant_id", "default")),
                    user_id=str(
                        origin.get("user_id")
                        or origin.get("sender_id")
                        or origin.get("chat_id")
                        or "default"
                    ),
                )
            )
            token = bind_workspace_scope(workspace_scope) if workspace_scope is not None else None
            try:
                async with asyncio.timeout(spec.resource_budget.max_runtime_seconds):
                    result = await self.runner.run(
                        AgentRunSpec(
                            initial_messages=messages,
                            tools=tools,
                            runtime=runtime,
                            max_iterations=min(
                                self.max_iterations, spec.resource_budget.max_iterations
                            ),
                            max_tool_result_chars=self.max_tool_result_chars,
                            hook=_SubagentHook(
                                task_id,
                                status,
                                max_tool_calls=spec.resource_budget.max_tool_calls,
                            ),
                            max_iterations_message="Task completed but no final response was generated.",
                            finalize_on_max_iterations=False,
                            error_message=None,
                            fail_on_tool_error=self.fail_on_tool_error,
                            checkpoint_callback=_on_checkpoint,
                            session_key=sess_key,
                            workspace=root,
                            llm_timeout_s=llm_timeout,
                        )
                    )
            finally:
                if token is not None:
                    reset_workspace_scope(token)
                reset_request_context(request_token)
            status.phase = "done"
            status.stop_reason = result.stop_reason

            if result.stop_reason == "tool_error":
                status.tool_events = list(result.tool_events)
                final_result = self._format_partial_progress(result)
                final_status = "error"
            elif result.stop_reason == "error":
                final_result = result.error or "Error: subagent execution failed."
                final_status = "error"
            else:
                final_result = (
                    result.final_content or "Task completed but no final response was generated."
                )
                final_status = "ok"
                logger.info("Subagent [{}] completed successfully", task_id)
                bounded_result = self._bounded_handoff(final_result)
                if heartbeat_task.done() and (heartbeat_error := heartbeat_task.exception()):
                    raise heartbeat_error
                if harness_run.task.review_target_task_id is not None:
                    decision = ReviewResult.model_validate_json(final_result)
                    await self.harness.complete_review(
                        task=harness_run.task,
                        agent=harness_run.worker,
                        lease=harness_run.lease,
                        approved=decision.approved,
                        notes=decision.notes,
                    )
                else:
                    await self.harness.complete_task(
                        task=harness_run.task,
                        agent=harness_run.worker,
                        lease=harness_run.lease,
                        result=bounded_result,
                    )
                terminal_recorded = True
            if final_status == "error":
                await self.harness.scheduler.fail(
                    harness_run.task,
                    harness_run.worker,
                    harness_run.lease,
                    final_result,
                )
            if announce:
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    final_result,
                    origin,
                    final_status,
                    origin_message_id,
                )
            return final_result

        except asyncio.CancelledError:
            status.phase = "error"
            status.stop_reason = "cancelled"
            try:
                await self.harness.scheduler.cancel(
                    harness_run.task,
                    harness_run.worker,
                    harness_run.lease,
                )
            except HarnessConflictError:
                logger.warning("Subagent [{}] cancellation ignored after lease loss", task_id)
            raise
        except Exception as e:
            status.phase = "error"
            status.error = str(e)
            logger.exception("Subagent [{}] failed", task_id)
            final_result = f"Error: {e}"
            if not terminal_recorded:
                try:
                    await self.harness.scheduler.fail(
                        harness_run.task,
                        harness_run.worker,
                        harness_run.lease,
                        final_result,
                    )
                except HarnessConflictError:
                    logger.warning("Subagent [{}] failure ignored after lease loss", task_id)
            if announce:
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    final_result,
                    origin,
                    "error",
                    origin_message_id,
                )
            return final_result
        finally:
            heartbeat_stop.set()
            if heartbeat_task is not None:
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            if harness_run.context is not None:
                await self.harness.ack_context(harness_run.context)
            self._harness_runs.pop(task_id, None)

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: _SubagentOrigin,
        status: str,
        origin_message_id: str | None = None,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = render_template(
            "agent/subagent_announce.md",
            label=label,
            status_text=status_text,
            task=self._bounded_handoff(task),
            result=self._bounded_handoff(result),
        )

        # Inject as system message to trigger main agent.
        # Use session_key_override to align with the main agent's effective
        # session key (which accounts for unified sessions) so the result is
        # routed to the correct pending queue (mid-turn injection) instead of
        # being dispatched as a competing independent task.
        override = origin.get("session_key") or f"{origin['channel']}:{origin['chat_id']}"
        metadata: dict[str, Any] = {
            "injected_event": "subagent_result",
            "subagent_task_id": task_id,
        }
        if origin_message_id:
            metadata["origin_message_id"] = origin_message_id
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
            session_key_override=override,
            metadata=metadata,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(
            "Subagent [{}] announced result to {}:{}", task_id, origin["channel"], origin["chat_id"]
        )

    def _bounded_handoff(self, content: str) -> str:
        """Bound cross-agent payloads independently from either agent's transcript."""
        if len(content) <= self.max_tool_result_chars:
            return content
        omitted = len(content) - self.max_tool_result_chars
        return f"{content[: self.max_tool_result_chars]}\n\n[handoff truncated: {omitted} chars omitted]"

    @staticmethod
    def _format_partial_progress(result: AgentRunResult) -> str:
        completed = [e for e in result.tool_events if e["status"] == "ok"]
        failure = next((e for e in reversed(result.tool_events) if e["status"] == "error"), None)
        lines: list[str] = []
        if completed:
            lines.append("Completed steps:")
            for event in completed[-3:]:
                lines.append(f"- {event['name']}: {event['detail']}")
        if failure:
            if lines:
                lines.append("")
            lines.append("Failure:")
            lines.append(f"- {failure['name']}: {failure['detail']}")
        if result.error and not failure:
            if lines:
                lines.append("")
            lines.append("Failure:")
            lines.append(f"- {result.error}")
        return "\n".join(lines) or (result.error or "Error: subagent execution failed.")

    def _build_subagent_prompt(
        self,
        workspace: Path | None = None,
        *,
        role: AgentRole = AgentRole.WORKER,
        objective: str = "Complete the assigned task.",
        review_gate: bool = False,
    ) -> str:
        """Build a focused system prompt for the subagent."""
        from nanobot.agent.skills import SkillsLoader

        agent_workspace = self.workspace.expanduser().resolve()
        project_workspace = workspace.expanduser().resolve() if workspace else agent_workspace
        skills_summary = SkillsLoader(
            self.workspace,
            disabled_skills=self.disabled_skills,
        ).build_skills_summary()
        return render_template(
            "agent/subagent_system.md",
            role=role.value,
            objective=objective,
            review_gate=review_gate,
            workspace=str(project_workspace),
            agent_workspace=str(agent_workspace),
            skills_summary=skills_summary or "",
        )

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [
            self._running_tasks[tid]
            for tid in self._session_tasks.get(session_key, [])
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        ]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._exec_session_manager.terminate_by_owner(session_key)
        return len(tasks)

    async def close(self) -> None:
        """Cancel running subagents and close their shared exec sessions."""
        tasks = [task for task in self._running_tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._exec_session_manager.close_all()
        await self.harness.store.close()

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    def get_running_count_by_session(self, session_key: str) -> int:
        """Return the number of currently running subagents for a session."""
        tids = self._session_tasks.get(session_key, set())
        return sum(
            1 for tid in tids if tid in self._running_tasks and not self._running_tasks[tid].done()
        )
