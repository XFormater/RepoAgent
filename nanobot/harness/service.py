"""Unified orchestration interface for the six harness abstractions."""

from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger
from pydantic import JsonValue

from nanobot.harness.context import ContextProjector, RuntimeContext
from nanobot.harness.metrics import HarnessMetrics
from nanobot.harness.models import (
    AgentInstance,
    AgentMessage,
    AgentRole,
    AgentSpec,
    ContextPolicy,
    MessageKind,
    PlannedWorkItem,
    ResourceBudget,
    Task,
    TaskExecutionMode,
    TaskState,
    WorkerLease,
    Workflow,
    WorkflowState,
    new_id,
    utcnow,
)
from nanobot.harness.scheduler import HarnessScheduler
from nanobot.harness.store import (
    HarnessCapacityError,
    HarnessConflictError,
    HarnessStore,
    InMemoryHarnessStore,
)


def main_agent_spec() -> AgentSpec:
    return AgentSpec(
        spec_id="main-supervisor",
        role=AgentRole.SUPERVISOR,
        objective="Own the user goal, decompose work, delegate bounded tasks, and synthesize results.",
        capabilities=frozenset({"plan", "delegate", "synthesize", "review"}),
        allowed_tools=frozenset({"spawn"}),
        context_policy=ContextPolicy(max_messages=48, max_chars=96_000),
        resource_budget=ResourceBudget(max_iterations=40, max_tool_calls=100),
    )


def worker_agent_spec() -> AgentSpec:
    return AgentSpec(
        spec_id="bounded-worker",
        role=AgentRole.WORKER,
        objective="Complete one assigned task and return a concise evidence-backed result.",
        capabilities=frozenset({"research", "code", "analyze"}),
        allowed_tools=frozenset(
            {
                "apply_patch",
                "edit_file",
                "exec",
                "find_files",
                "grep",
                "list_dir",
                "read_file",
                "web_fetch",
                "web_search",
                "write_file",
            }
        ),
        context_policy=ContextPolicy(
            max_messages=16,
            max_chars=32_000,
            max_message_chars=8_000,
            inherit_parent_messages=0,
        ),
    )


def planner_agent_spec() -> AgentSpec:
    return AgentSpec(
        spec_id="research-planner",
        role=AgentRole.PLANNER,
        objective="Decompose the workflow goal into bounded tasks and explicit dependencies.",
        capabilities=frozenset({"plan", "research_design"}),
        allowed_tools=frozenset({"read_file", "list_dir", "grep", "web_search"}),
        context_policy=ContextPolicy(max_messages=16, max_chars=32_000),
    )


def reviewer_agent_spec() -> AgentSpec:
    return AgentSpec(
        spec_id="evidence-reviewer",
        role=AgentRole.REVIEWER,
        objective="Verify evidence, detect unsupported claims, and accept or reject results.",
        capabilities=frozenset({"review", "verify"}),
        allowed_tools=frozenset({"read_file", "list_dir", "grep", "web_search"}),
        context_policy=ContextPolicy(max_messages=24, max_chars=48_000),
    )


def spec_for_role(role: AgentRole) -> AgentSpec:
    if role == AgentRole.PLANNER:
        return planner_agent_spec()
    if role == AgentRole.REVIEWER:
        return reviewer_agent_spec()
    if role == AgentRole.WORKER:
        return worker_agent_spec()
    return main_agent_spec()


class HarnessService:
    """Single admission point for roles, workflows, messages, context, and recovery."""

    def __init__(
        self,
        store: HarnessStore | None = None,
        metrics: HarnessMetrics | None = None,
        max_queued_tasks_per_tenant: int = 1_000,
        checkpoint_max_bytes: int = 262_144,
        retention_days: int = 30,
    ) -> None:
        self.store = store or InMemoryHarnessStore()
        self.metrics = metrics or HarnessMetrics()
        self.scheduler = HarnessScheduler(
            self.store,
            self.metrics,
            checkpoint_max_bytes=checkpoint_max_bytes,
        )
        self.projector = ContextProjector()
        self.max_queued_tasks_per_tenant = max_queued_tasks_per_tenant
        self.retention_days = retention_days

    async def register(self, spec: AgentSpec) -> None:
        await self.store.put_spec(spec)

    async def create_workflow(
        self,
        goal: str,
        *,
        tenant_id: str = "default",
        sealed: bool = True,
    ) -> tuple[Workflow, AgentInstance]:
        spec = main_agent_spec()
        await self.register(spec)
        owner = AgentInstance(
            spec_id=spec.spec_id,
            workflow_id="pending",
            tenant_id=tenant_id,
        )
        workflow = Workflow(
            goal=goal,
            owner_agent_id=owner.instance_id,
            tenant_id=tenant_id,
            sealed=sealed,
        )
        owner.workflow_id = workflow.workflow_id
        persisted, persisted_owner, _ = await self.store.get_or_create_workflow(
            workflow, owner
        )
        return persisted, persisted_owner

    async def get_or_create_workflow(
        self,
        *,
        scope_key: str,
        goal: str,
        tenant_id: str = "default",
        sealed: bool = True,
    ) -> tuple[Workflow, AgentInstance, bool]:
        spec = main_agent_spec()
        await self.register(spec)
        owner = AgentInstance(
            spec_id=spec.spec_id,
            workflow_id="pending",
            tenant_id=tenant_id,
        )
        workflow = Workflow(
            goal=goal,
            owner_agent_id=owner.instance_id,
            tenant_id=tenant_id,
            scope_key=scope_key,
            sealed=sealed,
        )
        owner.workflow_id = workflow.workflow_id
        result = await self.store.get_or_create_workflow(workflow, owner)
        self.metrics.event("workflow.created" if result[2] else "workflow.reused")
        return result

    async def seal_workflow(self, workflow_id: str) -> Workflow:
        """Close a dynamic DAG so all-successful Tasks may finish the Workflow."""
        for _ in range(3):
            workflow = await self.store.get_workflow(workflow_id)
            tasks = await self.store.list_tasks(workflow_id)
            workflow.sealed = True
            if workflow.state not in {WorkflowState.FAILED, WorkflowState.CANCELLED}:
                workflow.state = (
                    WorkflowState.SUCCEEDED
                    if tasks and all(task.state == TaskState.SUCCEEDED for task in tasks)
                    else WorkflowState.ACTIVE
                )
            try:
                await self.store.put_workflow(workflow)
                self.metrics.event("workflow.sealed")
                return workflow
            except HarnessConflictError:
                continue
        raise HarnessConflictError("could not seal workflow after CAS retries")

    async def add_task(
        self,
        workflow: Workflow,
        *,
        title: str,
        task_input: str,
        metadata: dict[str, JsonValue] | None = None,
        execution_mode: TaskExecutionMode = TaskExecutionMode.POOL,
        required_role: AgentRole = AgentRole.WORKER,
        required_capabilities: frozenset[str] = frozenset(),
        resource_keys: frozenset[str] = frozenset(),
        dependencies: frozenset[str] = frozenset(),
        review_target_task_id: str | None = None,
        idempotency_key: str | None = None,
        priority: int = 0,
        available_at: datetime | None = None,
        deadline_at: datetime | None = None,
        max_attempts: int = 3,
        retry_backoff_seconds: int = 5,
    ) -> Task:
        task, _ = await self.ensure_task(
            workflow,
            title=title,
            task_input=task_input,
            metadata=metadata,
            execution_mode=execution_mode,
            required_role=required_role,
            required_capabilities=required_capabilities,
            resource_keys=resource_keys,
            dependencies=dependencies,
            review_target_task_id=review_target_task_id,
            idempotency_key=idempotency_key,
            priority=priority,
            available_at=available_at,
            deadline_at=deadline_at,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        return task

    async def ensure_task(
        self,
        workflow: Workflow,
        *,
        title: str,
        task_input: str,
        metadata: dict[str, JsonValue] | None = None,
        execution_mode: TaskExecutionMode = TaskExecutionMode.POOL,
        required_role: AgentRole = AgentRole.WORKER,
        required_capabilities: frozenset[str] = frozenset(),
        resource_keys: frozenset[str] = frozenset(),
        dependencies: frozenset[str] = frozenset(),
        review_target_task_id: str | None = None,
        idempotency_key: str | None = None,
        priority: int = 0,
        available_at: datetime | None = None,
        deadline_at: datetime | None = None,
        max_attempts: int = 3,
        retry_backoff_seconds: int = 5,
    ) -> tuple[Task, bool]:
        """Create a task exactly once and report whether this call inserted it."""
        existing = (
            await self.store.get_task_by_idempotency(workflow.workflow_id, idempotency_key)
            if idempotency_key is not None
            else None
        )
        active_states = frozenset(
            {
                TaskState.PENDING.value,
                TaskState.READY.value,
                TaskState.LEASED.value,
                TaskState.RUNNING.value,
            }
        )
        if existing is None and await self.store.count_tasks(
            workflow.tenant_id, active_states
        ) >= self.max_queued_tasks_per_tenant:
            self.metrics.event("task.admission", outcome="capacity_rejected")
            raise HarnessCapacityError(
                f"tenant {workflow.tenant_id!r} reached queued task limit "
                f"{self.max_queued_tasks_per_tenant}"
            )
        task_data: dict[str, object] = {
            "workflow_id": workflow.workflow_id,
            "tenant_id": workflow.tenant_id,
            "title": title,
            "input": task_input,
            "metadata": metadata or {},
            "execution_mode": execution_mode,
            "required_role": required_role,
            "required_capabilities": required_capabilities,
            "resource_keys": resource_keys,
            "dependencies": dependencies,
            "review_target_task_id": review_target_task_id,
            "idempotency_key": idempotency_key,
            "priority": priority,
            "deadline_at": deadline_at,
            "max_attempts": max_attempts,
            "retry_backoff_seconds": retry_backoff_seconds,
            "state": TaskState.PENDING if dependencies else TaskState.READY,
        }
        if available_at is not None:
            task_data["available_at"] = available_at
        task = Task.model_validate(task_data)
        persisted, created = (
            (existing, False)
            if existing is not None
            else await self.store.get_or_create_task(task)
        )
        # Keep the workflow snapshot repairable even when a process died after
        # the atomic task insert but before updating the aggregate payload.
        for _ in range(3):
            current = await self.store.get_workflow(workflow.workflow_id)
            if persisted.task_id in {item.task_id for item in current.tasks}:
                break
            current.state = WorkflowState.ACTIVE
            current.tasks.append(persisted)
            Workflow.model_validate(current.model_dump())
            try:
                await self.store.put_workflow(current)
                workflow.tasks = current.tasks
                workflow.state = current.state
                workflow.version = current.version
                break
            except HarnessConflictError:
                continue
        else:
            raise HarnessConflictError("could not attach task to workflow after CAS retries")
        self.metrics.event("task.created" if created else "task.reused", role=required_role.value)
        return persisted, created

    async def add_review_gate(
        self,
        workflow: Workflow,
        target: Task,
        *,
        instructions: str = "Verify the evidence and reject unsupported conclusions.",
        idempotency_key: str | None = None,
    ) -> Task:
        """Create a Reviewer task that cannot run before its target succeeds."""
        if target.workflow_id != workflow.workflow_id or target.tenant_id != workflow.tenant_id:
            raise PermissionError("review target belongs to another workflow or tenant")
        return await self.add_task(
            workflow,
            title=f"Review: {target.title}",
            task_input=instructions,
            required_role=AgentRole.REVIEWER,
            required_capabilities=frozenset({"review", "verify"}),
            dependencies=frozenset({target.task_id}),
            review_target_task_id=target.task_id,
            idempotency_key=idempotency_key or f"review:{target.task_id}",
        )

    async def create_role_dag(
        self,
        *,
        goal: str,
        work_items: list[PlannedWorkItem],
        tenant_id: str = "default",
        scope_key: str | None = None,
    ) -> tuple[Workflow, AgentInstance, Task, list[Task], list[Task]]:
        """Materialize the production role topology for a bounded plan.

        The Planner must finish before any Worker becomes ready. Each reviewed
        Worker has a dedicated Reviewer gate, so workflow success means every
        configured quality gate passed rather than merely that workers returned.
        """
        if not work_items:
            raise ValueError("a role DAG requires at least one work item")
        if scope_key is None:
            workflow, owner = await self.create_workflow(goal, tenant_id=tenant_id)
        else:
            workflow, owner, _ = await self.get_or_create_workflow(
                scope_key=scope_key,
                goal=goal,
                tenant_id=tenant_id,
            )
        planner = await self.add_task(
            workflow,
            title="Plan workflow",
            task_input=goal,
            required_role=AgentRole.PLANNER,
            required_capabilities=frozenset({"plan"}),
            priority=100,
            idempotency_key="role-dag:planner",
        )
        workers: list[Task] = []
        reviews: list[Task] = []
        for index, item in enumerate(work_items):
            worker = await self.add_task(
                workflow,
                title=item.title,
                task_input=item.task_input,
                required_role=AgentRole.WORKER,
                required_capabilities=item.required_capabilities,
                resource_keys=item.resource_keys,
                dependencies=frozenset({planner.task_id}),
                priority=item.priority,
                idempotency_key=f"role-dag:worker:{index}",
            )
            workers.append(worker)
            if item.require_review:
                reviews.append(
                    await self.add_review_gate(
                        workflow,
                        worker,
                        instructions=item.review_instructions,
                        idempotency_key=f"role-dag:review:{index}",
                    )
                )
        return workflow, owner, planner, workers, reviews

    async def spawn_worker(self, workflow: Workflow, parent: AgentInstance) -> AgentInstance:
        return await self.spawn_agent(workflow, parent, AgentRole.WORKER)

    async def spawn_agent(
        self,
        workflow: Workflow,
        parent: AgentInstance,
        role: AgentRole,
    ) -> AgentInstance:
        if role == AgentRole.SUPERVISOR:
            raise ValueError("a workflow can have only its owning supervisor")
        return await self.spawn_agent_from_spec(
            workflow,
            parent,
            spec_for_role(role),
        )

    async def spawn_agent_from_spec(
        self,
        workflow: Workflow,
        parent: AgentInstance,
        spec: AgentSpec,
    ) -> AgentInstance:
        """Spawn a child from a versioned domain-specific AgentSpec.

        Domain workflows may narrow the generic role policy (for example a
        read-only repository analyst).  The supervisor boundary remains the
        same and the supplied spec is persisted before the instance is made
        runnable.
        """
        if spec.role == AgentRole.SUPERVISOR:
            raise ValueError("a workflow can have only its owning supervisor")
        persisted_parent = await self.store.get_agent(parent.instance_id)
        parent_spec = await self.store.get_spec(persisted_parent.spec_id)
        if (
            persisted_parent.workflow_id != workflow.workflow_id
            or persisted_parent.tenant_id != workflow.tenant_id
            or persisted_parent.instance_id != workflow.owner_agent_id
            or parent_spec.role != AgentRole.SUPERVISOR
        ):
            raise PermissionError("only the workflow supervisor may spawn child agents")
        await self.register(spec)
        worker = AgentInstance(
            spec_id=spec.spec_id,
            workflow_id=workflow.workflow_id,
            tenant_id=workflow.tenant_id,
            parent_agent_id=parent.instance_id,
        )
        await self.store.put_agent(worker)
        return worker

    async def send_message(
        self,
        *,
        workflow_id: str,
        tenant_id: str = "default",
        sender_agent_id: str,
        recipient_agent_id: str,
        kind: MessageKind,
        content: str,
        task_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> bool:
        sender = await self.store.get_agent(sender_agent_id)
        recipient = await self.store.get_agent(recipient_agent_id)
        if (
            sender.workflow_id != workflow_id
            or recipient.workflow_id != workflow_id
            or sender.tenant_id != tenant_id
            or recipient.tenant_id != tenant_id
        ):
            raise PermissionError("cross-workflow or cross-tenant agent message rejected")
        message = AgentMessage(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            task_id=task_id,
            sender_agent_id=sender_agent_id,
            recipient_agent_id=recipient_agent_id,
            kind=kind,
            content=content,
            idempotency_key=idempotency_key or new_id("idem"),
        )
        inserted = await self.store.send(message)
        self.metrics.event("message.sent", outcome="ok" if inserted else "duplicate")
        return inserted

    async def complete_task(
        self,
        *,
        task: Task,
        agent: AgentInstance,
        lease: WorkerLease,
        result: str,
        notify_supervisor: bool = True,
    ) -> None:
        """Commit a result and its Supervisor handoff as one durable unit."""
        workflow = await self.store.get_workflow(task.workflow_id)
        if (
            workflow.tenant_id != task.tenant_id
            or agent.workflow_id != workflow.workflow_id
            or agent.tenant_id != workflow.tenant_id
        ):
            raise PermissionError("task completion crossed a tenant or workflow boundary")
        message = (
            AgentMessage(
                workflow_id=task.workflow_id,
                tenant_id=task.tenant_id,
                task_id=task.task_id,
                sender_agent_id=agent.instance_id,
                recipient_agent_id=workflow.owner_agent_id,
                kind=MessageKind.RESULT,
                content=result,
                idempotency_key=f"result:{task.task_id}:{task.attempt}",
            )
            if notify_supervisor
            else None
        )
        await self.scheduler.complete(
            task,
            agent,
            lease,
            result,
            result_message=message,
        )
        if not notify_supervisor:
            return
        self.metrics.event("message.outbox_committed", role="worker")
        try:
            await self.flush_outbox()
        except Exception as exc:
            # Task and Outbox are already durable in PostgreSQL. Redis delivery
            # is retried by the next admission/worker maintenance cycle.
            logger.warning("Harness result delivery deferred for task {}: {}", task.task_id, exc)
            self.metrics.event("outbox.flush", outcome="deferred")

    async def complete_review(
        self,
        *,
        task: Task,
        agent: AgentInstance,
        lease: WorkerLease,
        approved: bool,
        notes: str,
    ) -> None:
        """Apply an explicit Reviewer quality decision under the active fence."""
        spec = await self.store.get_spec(agent.spec_id)
        if (
            spec.role != AgentRole.REVIEWER
            or task.required_role != AgentRole.REVIEWER
            or task.review_target_task_id is None
        ):
            raise HarnessConflictError("review decisions require a Reviewer gate task")
        if approved:
            await self.complete_task(
                task=task,
                agent=agent,
                lease=lease,
                result=f"APPROVED: {notes}",
            )
            self.metrics.event("review.completed", outcome="approved")
            return
        workflow = await self.store.get_workflow(task.workflow_id)
        message = AgentMessage(
            workflow_id=task.workflow_id,
            tenant_id=task.tenant_id,
            task_id=task.task_id,
            sender_agent_id=agent.instance_id,
            recipient_agent_id=workflow.owner_agent_id,
            kind=MessageKind.ERROR,
            content=f"REJECTED: {notes}",
            idempotency_key=f"review-rejected:{task.task_id}:{task.attempt}",
        )
        await self.scheduler.fail(
            task,
            agent,
            lease,
            f"review rejected: {notes}",
            retry=False,
            error_message=message,
        )
        self.metrics.event("review.completed", outcome="rejected")
        try:
            await self.flush_outbox()
        except Exception as exc:
            logger.warning("Review rejection delivery deferred for task {}: {}", task.task_id, exc)

    async def requeue_dead_letter(
        self,
        task_id: str,
        *,
        tenant_id: str,
        reset_attempts: bool = True,
    ) -> Task:
        """Explicit operator action to replay one tenant-owned DLQ Task."""
        task = await self.store.get_task(task_id)
        if task.tenant_id != tenant_id:
            raise PermissionError("dead-letter task belongs to another tenant")
        if task.state != TaskState.DEAD_LETTER:
            raise HarnessConflictError("only dead-letter tasks can be requeued")
        workflow_tasks = await self.store.list_tasks(task.workflow_id)
        succeeded = {
            item.task_id for item in workflow_tasks if item.state == TaskState.SUCCEEDED
        }
        task.state = (
            TaskState.READY if task.dependencies <= succeeded else TaskState.PENDING
        )
        task.attempt = 0 if reset_attempts else task.attempt
        task.error = None
        task.dead_lettered_at = None
        task.available_at = utcnow()
        task.updated_at = utcnow()
        await self.store.put_task(task)
        workflow = await self.store.get_workflow(task.workflow_id)
        workflow.state = WorkflowState.ACTIVE
        await self.store.put_workflow(workflow)
        self.metrics.event("task.dead_letter", outcome="requeued")
        return task

    async def runtime_context(
        self,
        agent_id: str,
        task_id: str,
        *,
        consumer_id: str | None = None,
    ) -> RuntimeContext:
        agent = await self.store.get_agent(agent_id)
        spec = await self.store.get_spec(agent.spec_id)
        task = await self.store.get_task(task_id)
        if task.workflow_id != agent.workflow_id or task.tenant_id != agent.tenant_id:
            raise PermissionError("agent cannot read context for another tenant or workflow")
        if consumer_id:
            inbox = await self.store.claim_messages(
                agent_id,
                consumer_id=consumer_id,
                now=utcnow(),
                limit=spec.context_policy.max_messages + 1,
            )
        else:
            inbox = await self.store.inbox(
                agent_id,
                now=utcnow(),
                limit=spec.context_policy.max_messages + 1,
            )
        checkpoint = await self.store.latest_checkpoint(agent_id)
        tasks = await self.store.list_tasks(task.workflow_id)
        dependencies = [item for item in tasks if item.task_id in task.dependencies]
        context = self.projector.project(
            agent=agent,
            spec=spec,
            task=task,
            dependencies=dependencies,
            inbox=inbox,
            checkpoint=checkpoint,
        )
        self.metrics.context(
            role=context.role,
            chars=len(context.render()),
            truncated=context.truncated,
        )
        return context

    async def ack_context(self, context: RuntimeContext) -> None:
        await self.store.ack_messages(context.agent_id, context.messages)

    async def flush_outbox(self, *, limit: int = 100) -> int:
        delivered = await self.store.flush_outbox(limit=limit)
        if delivered:
            self.metrics.event("outbox.flushed", outcome="ok")
        return delivered

    async def health(self) -> dict[str, bool]:
        return await self.store.health()

    async def diagnostics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, object]:
        storage = await self.store.diagnostics(tenant_id)
        for key, value in storage.items():
            self.metrics.gauge(key, value)
        return {
            "health": await self.health(),
            "storage": storage,
            "metrics": self.metrics.snapshot(),
        }

    async def maintenance(self, *, tenant_id: str) -> dict[str, object]:
        """Bounded, idempotent maintenance suitable for a periodic worker tick."""
        recovered = 0
        for workflow_id in await self.store.list_active_workflow_ids(tenant_id):
            recovered += await self.scheduler.recover_abandoned_tasks(workflow_id)
        expired = await self.scheduler.expire_deadlines(tenant_id)
        flushed = await self.flush_outbox()
        pruned = await self.store.prune(
            before=utcnow() - timedelta(days=self.retention_days),
            tenant_id=tenant_id,
        )
        self.metrics.event("maintenance.completed")
        return {
            "recovered": recovered,
            "expired": expired,
            "outbox_flushed": flushed,
            "pruned": pruned,
        }
