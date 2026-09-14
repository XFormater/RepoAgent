"""Repository-domain adapter for the shared multi-agent Harness control plane."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from nanobot.harness import (
    AgentInstance,
    AgentRole,
    AgentSpec,
    ContextPolicy,
    HarnessService,
    ResourceBudget,
    RuntimeContext,
    Task,
    TaskExecutionMode,
    TaskState,
    WorkerLease,
    Workflow,
)
from nanobot.repo_agent.models import RepositoryTask
from nanobot.repo_agent.registry import RepositoryExpert


@dataclass(slots=True)
class RepositoryHarnessRun:
    """One leased repository task and its recipient-owned context projection."""

    workflow: Workflow
    task: Task
    agent: AgentInstance
    lease: WorkerLease
    context: RuntimeContext
    lease_ttl_seconds: int
    heartbeat_stop: asyncio.Event
    heartbeat_task: asyncio.Task[None] | None = None


class RepositoryHarnessCoordinator:
    """Make Harness state authoritative for repository-agent execution.

    Repository JSON records remain useful user-facing domain projections and
    approval/audit documents. Scheduling, leases, role admission, dependency
    results, retries and runtime context are owned by the shared Harness.
    """

    def __init__(self, service: HarnessService, *, tenant_id: str = "default") -> None:
        self.service = service
        self.tenant_id = tenant_id
        self._control_lock = asyncio.Lock()

    @staticmethod
    def code_scope(task_id: str) -> str:
        return f"repository-task:{task_id}"

    @staticmethod
    def research_scope(research_id: str) -> str:
        return f"repository-research:{research_id}"

    async def workflow(
        self,
        *,
        scope_key: str,
        goal: str,
        tenant_id: str | None = None,
    ) -> tuple[Workflow, AgentInstance]:
        workflow, owner, _ = await self.service.get_or_create_workflow(
            scope_key=scope_key,
            goal=goal,
            tenant_id=tenant_id or self.tenant_id,
            sealed=False,
        )
        return workflow, owner

    async def seal(self, workflow: Workflow) -> Workflow:
        return await self.service.seal_workflow(workflow.workflow_id)

    @staticmethod
    def _spec_id(name: str, role: AgentRole, tools: tuple[str, ...]) -> str:
        digest = hashlib.sha256("\0".join(sorted(tools)).encode()).hexdigest()[:12]
        return f"repository-{role.value}-{name}-{digest}"

    @classmethod
    def expert_spec(cls, expert: RepositoryExpert) -> AgentSpec:
        role = AgentRole.REVIEWER if expert.name == "review" else AgentRole.WORKER
        capabilities = (
            frozenset({"review", "verify"})
            if role == AgentRole.REVIEWER
            else frozenset({"repository", "research", "analyze", "code"})
        )
        return AgentSpec(
            spec_id=cls._spec_id(expert.name, role, expert.tools),
            role=role,
            objective=expert.description,
            capabilities=capabilities,
            allowed_tools=frozenset(expert.tools),
            context_policy=ContextPolicy(
                max_messages=16,
                max_chars=32_000,
                max_tokens=8_000,
                max_message_chars=8_000,
                inherit_parent_messages=0,
            ),
            resource_budget=ResourceBudget(
                max_iterations=expert.max_iterations,
                max_runtime_seconds=900,
                max_tool_calls=max(12, expert.max_iterations * 3),
            ),
        )

    @classmethod
    def stage_spec(
        cls,
        *,
        name: str,
        role: AgentRole,
        objective: str,
        tools: tuple[str, ...] = (),
        max_iterations: int | None = 5,
    ) -> AgentSpec:
        capabilities = {
            AgentRole.PLANNER: frozenset({"plan"}),
            AgentRole.REVIEWER: frozenset({"review", "verify"}),
            AgentRole.WORKER: frozenset({"repository", "research", "analyze", "code"}),
        }[role]
        return AgentSpec(
            spec_id=cls._spec_id(name, role, tools),
            role=role,
            objective=objective,
            capabilities=capabilities,
            allowed_tools=frozenset(tools),
            context_policy=ContextPolicy(
                max_messages=24,
                max_chars=48_000,
                max_tokens=12_000,
                max_message_chars=12_000,
                inherit_parent_messages=0,
            ),
            resource_budget=ResourceBudget(
                max_iterations=max_iterations,
                max_runtime_seconds=900,
                # An unlimited iteration stage is also allowed an unlimited
                # tool-call budget; finite stages retain a proportional cap.
                max_tool_calls=(None if max_iterations is None else max(8, max_iterations * 4)),
            ),
        )

    async def ensure_task(
        self,
        workflow: Workflow,
        *,
        title: str,
        task_input: str,
        spec: AgentSpec,
        idempotency_key: str,
        metadata: dict[str, JsonValue] | None = None,
        dependencies: frozenset[str] = frozenset(),
        resource_keys: frozenset[str] = frozenset(),
        review_target_task_id: str | None = None,
        priority: int = 0,
        execution_mode: TaskExecutionMode = TaskExecutionMode.INLINE,
    ) -> Task:
        # A POOL worker can be a separate process and therefore cannot rely on
        # this coordinator's in-memory ``spec`` object.  Persist the selected
        # domain spec id with the Task so the worker recreates an AgentInstance
        # with the same capability boundary when it claims the work.
        task_metadata = dict(metadata or {})
        task_metadata.setdefault("agent_spec_id", spec.spec_id)
        await self.service.register(spec)
        required_capabilities = {
            AgentRole.PLANNER: frozenset({"plan"}),
            AgentRole.REVIEWER: frozenset({"review", "verify"}),
            AgentRole.WORKER: frozenset({"repository"}),
        }[spec.role]
        task, _ = await self.service.ensure_task(
            workflow,
            title=title,
            task_input=task_input,
            metadata=task_metadata,
            execution_mode=execution_mode,
            required_role=spec.role,
            required_capabilities=required_capabilities,
            dependencies=dependencies,
            resource_keys=resource_keys,
            review_target_task_id=review_target_task_id,
            idempotency_key=idempotency_key,
            priority=priority,
        )
        return task

    async def start(
        self,
        workflow: Workflow,
        owner: AgentInstance,
        task: Task,
        spec: AgentSpec,
        *,
        worker_id: str,
    ) -> RepositoryHarnessRun:
        async with self._control_lock:
            persisted = await self.service.store.get_task(task.task_id)
            if persisted.state == TaskState.SUCCEEDED:
                raise RuntimeError(f"Harness task {task.task_id} is already complete")
            await self.service.scheduler.recover_abandoned_tasks(workflow.workflow_id)
            await self.service.scheduler.ready_tasks(workflow.workflow_id)
            task = await self.service.store.get_task(task.task_id)
            agent = await self.service.spawn_agent_from_spec(workflow, owner, spec)
            lease_ttl_seconds = max(60, spec.resource_budget.max_runtime_seconds)
            lease = await self.service.scheduler.assign(
                task=task,
                agent=agent,
                spec=spec,
                worker_id=worker_id,
                ttl_seconds=lease_ttl_seconds,
            )
            await self.service.scheduler.start(task, agent, lease)
        context = await self.service.runtime_context(
            agent.instance_id,
            task.task_id,
            consumer_id=worker_id,
        )
        run = RepositoryHarnessRun(
            workflow=workflow,
            task=task,
            agent=agent,
            lease=lease,
            context=context,
            lease_ttl_seconds=lease_ttl_seconds,
            heartbeat_stop=asyncio.Event(),
        )
        run.heartbeat_task = asyncio.create_task(self._heartbeat(run))
        return run

    async def _heartbeat(self, run: RepositoryHarnessRun) -> None:
        interval = max(5.0, min(30.0, run.lease_ttl_seconds / 3))
        while True:
            try:
                await asyncio.wait_for(run.heartbeat_stop.wait(), timeout=interval)
                return
            except TimeoutError:
                run.lease = await self.service.scheduler.heartbeat(
                    run.lease,
                    ttl_seconds=run.lease_ttl_seconds,
                )

    @staticmethod
    async def _stop_heartbeat(run: RepositoryHarnessRun) -> None:
        run.heartbeat_stop.set()
        if run.heartbeat_task is None:
            return
        outcomes = await asyncio.gather(run.heartbeat_task, return_exceptions=True)
        if outcomes and isinstance(outcomes[0], BaseException):
            raise outcomes[0]

    async def complete(self, run: RepositoryHarnessRun, result: str) -> None:
        try:
            await self._stop_heartbeat(run)
            async with self._control_lock:
                await self.service.complete_task(
                    task=run.task,
                    agent=run.agent,
                    lease=run.lease,
                    result=result[:262_144],
                    notify_supervisor=False,
                )
        finally:
            await self.service.ack_context(run.context)

    async def fail(self, run: RepositoryHarnessRun, error: str) -> None:
        try:
            await self._stop_heartbeat(run)
            async with self._control_lock:
                await self.service.scheduler.fail(
                    run.task,
                    run.agent,
                    run.lease,
                    error[:262_144],
                    retry=False,
                )
        finally:
            await self.service.ack_context(run.context)

    async def cancel(self, run: RepositoryHarnessRun) -> None:
        try:
            await self._stop_heartbeat(run)
            async with self._control_lock:
                await self.service.scheduler.cancel(
                    run.task,
                    run.agent,
                    run.lease,
                )
        finally:
            await self.service.ack_context(run.context)

    async def planner_task(
        self,
        workflow: Workflow,
        *,
        task_input: str,
        key: str,
        title: str = "Plan repository workflow",
    ) -> tuple[Task, AgentSpec]:
        spec = self.stage_spec(
            name="planner",
            role=AgentRole.PLANNER,
            objective="Create a bounded repository workflow with explicit dependencies.",
        )
        task = await self.ensure_task(
            workflow,
            title=title,
            task_input=task_input,
            spec=spec,
            idempotency_key=key,
            priority=100,
        )
        return task, spec

    async def materialize_plan(
        self,
        workflow: Workflow,
        task: RepositoryTask,
        *,
        planner_task_id: str,
        experts: dict[str, RepositoryExpert],
    ) -> dict[str, tuple[Task, AgentSpec]]:
        """Materialize a repository PlanStep DAG after the Planner commits."""
        pending = {step.step_id: step for step in task.steps}
        materialized: dict[str, tuple[Task, AgentSpec]] = {}
        while pending:
            progressed = False
            for step_id, step in list(pending.items()):
                if not set(step.depends_on) <= set(materialized):
                    continue
                expert = experts[step.assigned_agent]
                spec = self.expert_spec(expert)
                dependencies = frozenset(
                    {
                        planner_task_id,
                        *(materialized[value][0].task_id for value in step.depends_on),
                    }
                )
                review_target = None
                if spec.role == AgentRole.REVIEWER and step.depends_on:
                    review_target = materialized[step.depends_on[-1]][0].task_id
                resources: frozenset[str] = (
                    frozenset({f"workspace:{Path(task.workspace_path).resolve()}"})
                    if "write_file" in spec.allowed_tools or "edit_file" in spec.allowed_tools
                    else frozenset()
                )
                harness_task = await self.ensure_task(
                    workflow,
                    title=step.name,
                    task_input=step.instruction,
                    metadata={
                        "domain": "repository",
                        "repository_full_name": task.repository_full_name,
                        "repository_workspace": task.workspace_path,
                        "repository_task_id": task.task_id,
                        "repository_step_id": step.step_id,
                    },
                    spec=spec,
                    dependencies=dependencies,
                    resource_keys=resources,
                    review_target_task_id=review_target,
                    idempotency_key=f"repository-step:{task.task_id}:{step.step_id}",
                )
                materialized[step_id] = (harness_task, spec)
                del pending[step_id]
                progressed = True
            if not progressed:
                raise ValueError("repository plan could not be materialized as a Harness DAG")
        return materialized
