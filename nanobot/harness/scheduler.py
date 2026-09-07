"""DAG scheduling, role/capability admission, checkpoints, and worker leases."""

from __future__ import annotations

from datetime import timedelta

from nanobot.harness.checkpoint import sanitize_checkpoint_state
from nanobot.harness.metrics import HarnessMetrics
from nanobot.harness.models import (
    AgentInstance,
    AgentMessage,
    AgentSpec,
    AgentState,
    Checkpoint,
    Task,
    TaskState,
    WorkerLease,
    WorkflowState,
    utcnow,
)
from nanobot.harness.store import HarnessConflictError, HarnessStore


class HarnessScheduler:
    def __init__(
        self,
        store: HarnessStore,
        metrics: HarnessMetrics | None = None,
        *,
        checkpoint_max_bytes: int = 262_144,
    ) -> None:
        self.store = store
        self.metrics = metrics or HarnessMetrics()
        self.checkpoint_max_bytes = checkpoint_max_bytes

    async def _audit_task(
        self,
        event_type: str,
        task: Task,
        *,
        agent: AgentInstance | None = None,
        lease: WorkerLease | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        metadata = task.metadata
        trace_id = metadata.get("trace_id")
        session_key = metadata.get("session_key")
        await self.store.record_audit_event(
            event_type,
            component="harness_scheduler",
            tenant_id=task.tenant_id,
            workflow_id=task.workflow_id,
            task_id=task.task_id,
            agent_id=agent.instance_id if agent else task.assigned_agent_id,
            session_key=session_key if isinstance(session_key, str) else None,
            trace_id=trace_id if isinstance(trace_id, str) else None,
            payload={
                "state": task.state.value,
                "attempt": task.attempt,
                "title": task.title,
                "metadata": metadata,
                "lease_id": lease.lease_id if lease else task.lease_id,
                "fencing_token": lease.fencing_token if lease else task.fencing_token,
                **(payload or {}),
            },
        )

    async def ready_tasks(self, workflow_id: str) -> list[Task]:
        tasks = await self.store.list_tasks(workflow_id)
        succeeded = {task.task_id for task in tasks if task.state == TaskState.SUCCEEDED}
        now = utcnow()
        ready: list[Task] = []
        for task in tasks:
            if (
                task.state in {TaskState.PENDING, TaskState.READY}
                and task.dependencies <= succeeded
                and task.available_at <= now
                and (task.deadline_at is None or task.deadline_at > now)
            ):
                if task.state != TaskState.READY:
                    task.state = TaskState.READY
                    task.updated_at = utcnow()
                    await self.store.put_task(task)
                    await self._audit_task("harness.task.ready", task)
                ready.append(task)
        return ready

    async def recover_abandoned_tasks(self, workflow_id: str) -> int:
        """Return tasks whose worker lease expired to the ready queue.

        PostgreSQL task state is durable while Redis leases are intentionally
        ephemeral. This reconciliation is therefore required after worker or
        Redis restarts. Versioned writes fence a concurrent completion.
        """
        recovered = 0
        now = utcnow()
        for task in await self.store.list_tasks(workflow_id):
            if task.state not in {TaskState.LEASED, TaskState.RUNNING}:
                continue
            if await self.store.lease_active(task.task_id, now=now):
                continue
            assigned_agent_id = task.assigned_agent_id
            task.state = TaskState.READY
            task.assigned_agent_id = None
            task.lease_id = None
            task.error = "worker lease expired; task recovered"
            task.updated_at = now
            try:
                await self.store.put_task(task)
            except HarnessConflictError:
                continue
            if assigned_agent_id:
                try:
                    agent = await self.store.get_agent(assigned_agent_id)
                    if agent.current_task_id == task.task_id:
                        agent.state = AgentState.IDLE
                        agent.current_task_id = None
                        agent.worker_id = None
                        agent.updated_at = now
                        await self.store.put_agent(agent)
                except (KeyError, HarnessConflictError):
                    pass
            recovered += 1
            self.metrics.event("task.recovered")
            await self._audit_task("harness.task.recovered", task, payload={"reason": task.error})
        return recovered

    @staticmethod
    def admits(spec: AgentSpec, task: Task) -> bool:
        return spec.role == task.required_role and task.required_capabilities <= spec.capabilities

    async def assign(
        self,
        *,
        task: Task,
        agent: AgentInstance,
        spec: AgentSpec,
        worker_id: str,
        ttl_seconds: int = 60,
    ) -> WorkerLease:
        persisted = await self.store.get_task(task.task_id)
        task.state = persisted.state
        task.updated_at = persisted.updated_at
        task.version = persisted.version
        if task.state != TaskState.READY:
            raise HarnessConflictError("only ready tasks can be assigned")
        now = utcnow()
        if task.available_at > now:
            raise HarnessConflictError("task is not available yet")
        if task.deadline_at is not None and task.deadline_at <= now:
            raise HarnessConflictError("task deadline has expired")
        if agent.workflow_id != task.workflow_id:
            raise HarnessConflictError("agent and task belong to different workflows")
        if not self.admits(spec, task):
            raise HarnessConflictError("agent role or capabilities do not satisfy task")
        lease = WorkerLease(
            task_id=task.task_id,
            agent_id=agent.instance_id,
            worker_id=worker_id,
            fencing_token=task.fencing_token + 1,
            resource_keys=task.resource_keys,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        if not await self.store.acquire_lease(lease, now=now):
            raise HarnessConflictError("task already has an active worker lease")
        if not await self.store.acquire_resources(lease, now=now):
            await self.store.update_lease(lease.model_copy(update={"released_at": now}))
            raise HarnessConflictError("a required resource is owned by another task")
        task.state = TaskState.LEASED
        task.assigned_agent_id = agent.instance_id
        task.lease_id = lease.lease_id
        task.fencing_token = lease.fencing_token
        task.attempt += 1
        task.updated_at = now
        agent.state = AgentState.RUNNING
        agent.current_task_id = task.task_id
        agent.worker_id = worker_id
        agent.updated_at = now
        try:
            await self.store.commit_task_agent(task, agent)
        except Exception:
            await self.store.release_resources(lease)
            try:
                await self.store.update_lease(lease.model_copy(update={"released_at": utcnow()}))
            except HarnessConflictError:
                pass
            raise
        self.metrics.event("task.assigned", role=spec.role.value)
        await self._audit_task(
            "harness.task.assigned",
            task,
            agent=agent,
            lease=lease,
            payload={"worker_id": worker_id, "role": spec.role.value},
        )
        return lease

    async def heartbeat(self, lease: WorkerLease, ttl_seconds: int = 60) -> WorkerLease:
        now = utcnow()
        if not lease.active_at(now):
            raise HarnessConflictError("cannot heartbeat an expired or released lease")
        renewed = lease.model_copy(
            update={
                "heartbeat_at": now,
                "expires_at": now + timedelta(seconds=ttl_seconds),
            }
        )
        if not await self.store.renew_resources(renewed):
            raise HarnessConflictError("worker lost an exclusive task resource")
        await self.store.update_lease(renewed)
        self.metrics.event("lease.heartbeat")
        task = await self.store.get_task(lease.task_id)
        await self._audit_task(
            "harness.lease.heartbeat", task, lease=renewed, payload={"worker_id": renewed.worker_id}
        )
        return renewed

    async def start(
        self,
        task: Task,
        agent: AgentInstance,
        lease: WorkerLease,
    ) -> None:
        """Persist the leased-to-running transition under the current fence."""
        self._assert_fence(task, agent, lease)
        task.state = TaskState.RUNNING
        task.updated_at = utcnow()
        await self.store.commit_task_agent(task, agent)
        self.metrics.event("task.started")
        await self._audit_task("harness.task.started", task, agent=agent, lease=lease)

    async def checkpoint(
        self,
        agent: AgentInstance,
        state: dict[str, object],
        sequence: int,
    ) -> Checkpoint:
        safe_state, truncated = sanitize_checkpoint_state(
            state,
            max_bytes=self.checkpoint_max_bytes,
        )
        checkpoint = Checkpoint(
            workflow_id=agent.workflow_id,
            agent_id=agent.instance_id,
            task_id=agent.current_task_id,
            sequence=sequence,
            state=safe_state,
        )
        await self.store.save_checkpoint(checkpoint)
        agent.checkpoint_id = checkpoint.checkpoint_id
        agent.updated_at = utcnow()
        await self.store.put_agent(agent)
        self.metrics.event(
            "checkpoint.saved",
            outcome="truncated" if truncated else "complete",
        )
        task = await self.store.get_task(agent.current_task_id) if agent.current_task_id else None
        if task is not None:
            await self._audit_task(
                "harness.checkpoint.saved",
                task,
                agent=agent,
                payload={"checkpoint_id": checkpoint.checkpoint_id, "sequence": sequence},
            )
        return checkpoint

    async def complete(
        self,
        task: Task,
        agent: AgentInstance,
        lease: WorkerLease,
        result: str,
        *,
        result_message: AgentMessage | None = None,
    ) -> None:
        now = utcnow()
        if not lease.active_at(now):
            raise HarnessConflictError("cannot complete task without an active lease")
        self._assert_fence(task, agent, lease)
        released = lease.model_copy(update={"released_at": now})
        await self.store.update_lease(released)
        await self.store.release_resources(lease)
        task.state = TaskState.SUCCEEDED
        task.result = result
        task.error = None
        task.assigned_agent_id = None
        task.lease_id = None
        task.updated_at = now
        agent.state = AgentState.IDLE
        agent.current_task_id = None
        agent.worker_id = None
        agent.updated_at = now
        await self.store.commit_task_agent(task, agent, outbox_message=result_message)
        tasks = await self.store.list_tasks(task.workflow_id)
        workflow = await self.store.get_workflow(task.workflow_id)
        if workflow.sealed and tasks and all(item.state == TaskState.SUCCEEDED for item in tasks):
            workflow.state = WorkflowState.SUCCEEDED
            await self.store.put_workflow(workflow)
        else:
            await self.ready_tasks(task.workflow_id)
        self.metrics.event("task.completed")
        await self._audit_task(
            "harness.task.completed",
            task,
            agent=agent,
            lease=lease,
            payload={"result": result, "outbox_message": result_message.model_dump(mode="json") if result_message else None},
        )

    async def fail(
        self,
        task: Task,
        agent: AgentInstance,
        lease: WorkerLease,
        error: str,
        *,
        retry: bool = True,
        error_message: AgentMessage | None = None,
    ) -> TaskState:
        now = utcnow()
        self._assert_fence(task, agent, lease)
        released = lease.model_copy(update={"released_at": now})
        # Losing the lease removes this worker's authority to mutate durable
        # task state. The reconciler or the new lease owner decides the outcome.
        await self.store.update_lease(released)
        await self.store.release_resources(lease)
        task.error = error
        retrying = retry and task.attempt < task.max_attempts
        task.state = TaskState.READY if retrying else TaskState.DEAD_LETTER
        task.assigned_agent_id = None
        task.lease_id = None
        if retrying:
            delay = min(
                task.retry_backoff_seconds * (2 ** max(0, task.attempt - 1)),
                3_600,
            )
            task.available_at = now + timedelta(seconds=delay)
        else:
            task.dead_lettered_at = now
        task.updated_at = now
        agent.state = AgentState.IDLE if retrying else AgentState.FAILED
        agent.current_task_id = None
        agent.worker_id = None
        agent.updated_at = now
        await self.store.commit_task_agent(
            task,
            agent,
            outbox_message=error_message,
        )
        if task.state == TaskState.DEAD_LETTER:
            workflow = await self.store.get_workflow(task.workflow_id)
            workflow.state = WorkflowState.FAILED
            await self.store.put_workflow(workflow)
        self.metrics.event(
            "task.failed",
            outcome="retry" if retrying else "dead_letter",
        )
        await self._audit_task(
            "harness.task.failed",
            task,
            agent=agent,
            lease=lease,
            payload={"error": error, "retrying": retrying},
        )
        return task.state

    async def expire_deadlines(self, tenant_id: str, *, limit: int = 100) -> int:
        """Move never-started Tasks past their deadline to the dead-letter queue."""
        expired = await self.store.list_expired_tasks(tenant_id, limit=limit)
        now = utcnow()
        moved = 0
        for task in expired:
            task.state = TaskState.DEAD_LETTER
            task.error = "task deadline expired before execution"
            task.dead_lettered_at = now
            task.updated_at = now
            try:
                await self.store.put_task(task)
            except HarnessConflictError:
                continue
            workflow = await self.store.get_workflow(task.workflow_id)
            workflow.state = WorkflowState.FAILED
            try:
                await self.store.put_workflow(workflow)
            except HarnessConflictError:
                pass
            moved += 1
            self.metrics.event("task.dead_letter", outcome="deadline")
            await self._audit_task("harness.task.dead_letter", task, payload={"reason": task.error})
        return moved

    async def cancel(
        self,
        task: Task,
        agent: AgentInstance,
        lease: WorkerLease,
    ) -> None:
        now = utcnow()
        self._assert_fence(task, agent, lease)
        await self.store.update_lease(lease.model_copy(update={"released_at": now}))
        await self.store.release_resources(lease)
        task.state = TaskState.CANCELLED
        task.assigned_agent_id = None
        task.lease_id = None
        task.updated_at = now
        agent.state = AgentState.STOPPED
        agent.current_task_id = None
        agent.worker_id = None
        agent.updated_at = now
        await self.store.commit_task_agent(task, agent)
        workflow = await self.store.get_workflow(task.workflow_id)
        workflow.state = WorkflowState.CANCELLED
        await self.store.put_workflow(workflow)
        self.metrics.event("task.cancelled")
        await self._audit_task("harness.task.cancelled", task, agent=agent, lease=lease)

    @staticmethod
    def _assert_fence(task: Task, agent: AgentInstance, lease: WorkerLease) -> None:
        if (
            task.state not in {TaskState.LEASED, TaskState.RUNNING}
            or task.assigned_agent_id != agent.instance_id
            or agent.current_task_id != task.task_id
            or task.lease_id != lease.lease_id
            or task.fencing_token != lease.fencing_token
        ):
            raise HarnessConflictError("stale worker fencing token")
