"""Concurrency-safe in-process reference store for harness contracts."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Protocol

from nanobot.harness.models import (
    AgentInstance,
    AgentMessage,
    AgentSpec,
    Checkpoint,
    Task,
    TaskState,
    WorkerLease,
    WorkerRegistration,
    Workflow,
    WorkflowState,
)


class HarnessConflictError(RuntimeError):
    pass


class HarnessCapacityError(RuntimeError):
    pass


class HarnessStore(Protocol):
    async def put_spec(self, spec: AgentSpec) -> None: ...
    async def put_agent(self, agent: AgentInstance) -> None: ...
    async def put_workflow(self, workflow: Workflow) -> None: ...
    async def get_or_create_workflow(
        self, workflow: Workflow, owner: AgentInstance
    ) -> tuple[Workflow, AgentInstance, bool]: ...
    async def get_workflow(self, workflow_id: str) -> Workflow: ...
    async def get_workflow_by_scope(self, tenant_id: str, scope_key: str) -> Workflow | None: ...
    async def list_active_workflow_ids(self, tenant_id: str, *, limit: int = 1_000) -> list[str]: ...
    async def put_task(self, task: Task) -> None: ...
    async def commit_task_agent(
        self,
        task: Task,
        agent: AgentInstance,
        *,
        outbox_message: AgentMessage | None = None,
    ) -> None: ...
    async def get_or_create_task(self, task: Task) -> tuple[Task, bool]: ...
    async def get_task_by_idempotency(
        self, workflow_id: str, idempotency_key: str
    ) -> Task | None: ...
    async def get_spec(self, spec_id: str) -> AgentSpec: ...
    async def get_agent(self, agent_id: str) -> AgentInstance: ...
    async def get_task(self, task_id: str) -> Task: ...
    async def list_tasks(self, workflow_id: str) -> list[Task]: ...
    async def count_tasks(self, tenant_id: str, states: frozenset[str]) -> int: ...
    async def list_ready_tasks(self, tenant_id: str, *, limit: int) -> list[Task]: ...
    async def list_expired_tasks(
        self, tenant_id: str, *, limit: int = 100
    ) -> list[Task]: ...
    async def list_dead_letters(
        self, tenant_id: str, *, limit: int = 100
    ) -> list[Task]: ...
    async def send(self, message: AgentMessage) -> bool: ...
    async def inbox(self, agent_id: str, *, now: datetime, limit: int) -> list[AgentMessage]: ...
    async def claim_messages(
        self, agent_id: str, *, consumer_id: str, now: datetime, limit: int
    ) -> list[AgentMessage]: ...
    async def ack_messages(self, agent_id: str, messages: list[AgentMessage]) -> None: ...
    async def flush_outbox(self, *, limit: int = 100) -> int: ...
    async def health(self) -> dict[str, bool]: ...
    async def diagnostics(self, tenant_id: str | None = None) -> dict[str, int | float]: ...
    async def prune(self, *, before: datetime, tenant_id: str | None = None) -> dict[str, int]: ...
    async def save_checkpoint(self, checkpoint: Checkpoint) -> None: ...
    async def latest_checkpoint(self, agent_id: str) -> Checkpoint | None: ...
    async def acquire_lease(self, lease: WorkerLease, *, now: datetime) -> bool: ...
    async def get_lease(self, task_id: str) -> WorkerLease | None: ...
    async def lease_active(self, task_id: str, *, now: datetime) -> bool: ...
    async def update_lease(self, lease: WorkerLease) -> None: ...
    async def acquire_resources(self, lease: WorkerLease, *, now: datetime) -> bool: ...
    async def renew_resources(self, lease: WorkerLease) -> bool: ...
    async def release_resources(self, lease: WorkerLease) -> None: ...
    async def register_worker(self, worker: WorkerRegistration, *, ttl_seconds: int) -> None: ...
    async def list_workers(self, tenant_id: str, *, now: datetime) -> list[WorkerRegistration]: ...
    async def record_audit_event(
        self,
        event_type: str,
        *,
        component: str,
        tenant_id: str | None = None,
        workflow_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        session_key: str | None = None,
        trace_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None: ...
    async def latest_audit_event_id(self) -> int: ...
    async def list_audit_events_after(
        self, event_id: int, *, event_types: frozenset[str], limit: int = 100
    ) -> list[dict[str, Any]]: ...
    async def close(self) -> None: ...


class InMemoryHarnessStore:
    """Executable reference implementation; production adapters can preserve this API."""

    def __init__(self) -> None:
        self.specs: dict[str, AgentSpec] = {}
        self.agents: dict[str, AgentInstance] = {}
        self.workflows: dict[str, Workflow] = {}
        self.tasks: dict[str, Task] = {}
        self.messages: dict[str, AgentMessage] = {}
        self.mailboxes: dict[str, list[str]] = defaultdict(list)
        self.idempotency_keys: set[str] = set()
        self.acked_messages: set[str] = set()
        self.checkpoints: dict[str, list[Checkpoint]] = defaultdict(list)
        self.leases: dict[str, WorkerLease] = {}
        self.resource_leases: dict[str, tuple[str, datetime]] = {}
        self.workers: dict[str, WorkerRegistration] = {}
        self.audit_events: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def put_spec(self, spec: AgentSpec) -> None:
        async with self._lock:
            old = self.specs.get(spec.spec_id)
            if old is not None and old.version > spec.version:
                raise HarnessConflictError("cannot replace an agent spec with an older version")
            self.specs[spec.spec_id] = spec

    async def record_audit_event(
        self,
        event_type: str,
        *,
        component: str,
        tenant_id: str | None = None,
        workflow_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        session_key: str | None = None,
        trace_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        from nanobot.harness.audit import payload_digest, redact_audit_value

        value = redact_audit_value(payload or {})
        assert isinstance(value, dict)
        async with self._lock:
            self.audit_events.append(
                {
                    "event_id": len(self.audit_events) + 1,
                    "event_type": event_type,
                    "component": component,
                    "tenant_id": tenant_id,
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "session_key": session_key,
                    "trace_id": trace_id,
                    "payload": value,
                    "payload_sha256": payload_digest(value),
                }
            )

    async def latest_audit_event_id(self) -> int:
        async with self._lock:
            return len(self.audit_events)

    async def list_audit_events_after(
        self, event_id: int, *, event_types: frozenset[str], limit: int = 100
    ) -> list[dict[str, Any]]:
        async with self._lock:
            return [
                dict(event)
                for event in self.audit_events
                if int(event.get("event_id", 0)) > event_id
                and event.get("event_type") in event_types
            ][:limit]

    async def put_agent(self, agent: AgentInstance) -> None:
        async with self._lock:
            self.agents[agent.instance_id] = agent

    async def put_workflow(self, workflow: Workflow) -> None:
        async with self._lock:
            self.workflows[workflow.workflow_id] = workflow
            for task in workflow.tasks:
                self.tasks[task.task_id] = task

    async def get_or_create_workflow(
        self, workflow: Workflow, owner: AgentInstance
    ) -> tuple[Workflow, AgentInstance, bool]:
        async with self._lock:
            for existing in self.workflows.values():
                if (
                    workflow.scope_key is not None
                    and existing.scope_key == workflow.scope_key
                    and existing.tenant_id == workflow.tenant_id
                ):
                    return existing, self.agents[existing.owner_agent_id], False
            self.workflows[workflow.workflow_id] = workflow
            self.agents[owner.instance_id] = owner
            return workflow, owner, True

    async def get_workflow(self, workflow_id: str) -> Workflow:
        return self.workflows[workflow_id]

    async def get_workflow_by_scope(self, tenant_id: str, scope_key: str) -> Workflow | None:
        return next(
            (
                workflow
                for workflow in self.workflows.values()
                if workflow.tenant_id == tenant_id and workflow.scope_key == scope_key
            ),
            None,
        )

    async def list_active_workflow_ids(
        self, tenant_id: str, *, limit: int = 1_000
    ) -> list[str]:
        return [
            workflow.workflow_id
            for workflow in self.workflows.values()
            if workflow.tenant_id == tenant_id and workflow.state == WorkflowState.ACTIVE
        ][:limit]

    async def put_task(self, task: Task) -> None:
        async with self._lock:
            self.tasks[task.task_id] = task

    async def commit_task_agent(
        self,
        task: Task,
        agent: AgentInstance,
        *,
        outbox_message: AgentMessage | None = None,
    ) -> None:
        """Commit a lifecycle transition and optional result handoff atomically."""
        async with self._lock:
            self.tasks[task.task_id] = task
            self.agents[agent.instance_id] = agent
            if outbox_message is not None:
                if outbox_message.idempotency_key in self.idempotency_keys:
                    return
                self.idempotency_keys.add(outbox_message.idempotency_key)
                self.messages[outbox_message.message_id] = outbox_message
                self.mailboxes[outbox_message.recipient_agent_id].append(
                    outbox_message.message_id
                )

    async def get_or_create_task(self, task: Task) -> tuple[Task, bool]:
        async with self._lock:
            if task.idempotency_key is not None:
                for existing in self.tasks.values():
                    if (
                        existing.workflow_id == task.workflow_id
                        and existing.idempotency_key == task.idempotency_key
                    ):
                        return existing, False
            self.tasks[task.task_id] = task
            return task, True

    async def get_task_by_idempotency(
        self, workflow_id: str, idempotency_key: str
    ) -> Task | None:
        async with self._lock:
            return next(
                (
                    task
                    for task in self.tasks.values()
                    if task.workflow_id == workflow_id
                    and task.idempotency_key == idempotency_key
                ),
                None,
            )

    async def get_spec(self, spec_id: str) -> AgentSpec:
        return self.specs[spec_id]

    async def get_agent(self, agent_id: str) -> AgentInstance:
        return self.agents[agent_id]

    async def get_task(self, task_id: str) -> Task:
        return self.tasks[task_id]

    async def list_tasks(self, workflow_id: str) -> list[Task]:
        return [task for task in self.tasks.values() if task.workflow_id == workflow_id]

    async def count_tasks(self, tenant_id: str, states: frozenset[str]) -> int:
        return sum(
            task.tenant_id == tenant_id and task.state.value in states
            for task in self.tasks.values()
        )

    async def list_ready_tasks(self, tenant_id: str, *, limit: int) -> list[Task]:
        now = datetime.now().astimezone()
        tasks = [
            task
            for task in self.tasks.values()
            if task.tenant_id == tenant_id
            and task.state.value == "ready"
            and task.available_at <= now
            and (task.deadline_at is None or task.deadline_at > now)
        ]
        return sorted(tasks, key=lambda item: (-item.priority, item.created_at))[:limit]

    async def list_expired_tasks(
        self, tenant_id: str, *, limit: int = 100
    ) -> list[Task]:
        now = datetime.now().astimezone()
        return [
            task
            for task in self.tasks.values()
            if task.tenant_id == tenant_id
            and task.state.value in {"pending", "ready"}
            and task.deadline_at is not None
            and task.deadline_at <= now
        ][:limit]

    async def list_dead_letters(
        self, tenant_id: str, *, limit: int = 100
    ) -> list[Task]:
        tasks = [
            task
            for task in self.tasks.values()
            if task.tenant_id == tenant_id and task.state.value == "dead_letter"
        ]
        return sorted(tasks, key=lambda item: item.updated_at, reverse=True)[:limit]

    async def send(self, message: AgentMessage) -> bool:
        async with self._lock:
            if message.idempotency_key in self.idempotency_keys:
                return False
            self.idempotency_keys.add(message.idempotency_key)
            self.messages[message.message_id] = message
            self.mailboxes[message.recipient_agent_id].append(message.message_id)
            return True

    async def inbox(
        self,
        agent_id: str,
        *,
        now: datetime,
        limit: int,
    ) -> list[AgentMessage]:
        async with self._lock:
            result: list[AgentMessage] = []
            for message_id in self.mailboxes.get(agent_id, []):
                message = self.messages[message_id]
                expires_at = message.expires_at
                if expires_at is None or expires_at > now:
                    result.append(message)
            return result[-limit:]

    async def claim_messages(
        self,
        agent_id: str,
        *,
        consumer_id: str,
        now: datetime,
        limit: int,
    ) -> list[AgentMessage]:
        del consumer_id
        messages = await self.inbox(agent_id, now=now, limit=limit)
        return [msg for msg in messages if msg.message_id not in self.acked_messages]

    async def ack_messages(self, agent_id: str, messages: list[AgentMessage]) -> None:
        del agent_id
        async with self._lock:
            self.acked_messages.update(message.message_id for message in messages)

    async def flush_outbox(self, *, limit: int = 100) -> int:
        del limit
        return 0

    async def health(self) -> dict[str, bool]:
        return {"store": True, "postgres": True, "redis": True}

    async def diagnostics(self, tenant_id: str | None = None) -> dict[str, int | float]:
        async with self._lock:
            tasks = [
                task
                for task in self.tasks.values()
                if tenant_id is None or task.tenant_id == tenant_id
            ]
            values: dict[str, int | float] = {
                "workflows": sum(
                    1
                    for workflow in self.workflows.values()
                    if tenant_id is None or workflow.tenant_id == tenant_id
                ),
                "agents": sum(
                    1
                    for agent in self.agents.values()
                    if tenant_id is None or agent.tenant_id == tenant_id
                ),
                "checkpoints": sum(len(items) for items in self.checkpoints.values()),
                "outbox_pending": 0,
            }
            for state in TaskState:
                values[f"tasks.{state.value}"] = sum(
                    1 for task in tasks if task.state == state
                )
            return values

    async def prune(
        self,
        *,
        before: datetime,
        tenant_id: str | None = None,
    ) -> dict[str, int]:
        async with self._lock:
            terminal = {WorkflowState.SUCCEEDED, WorkflowState.FAILED, WorkflowState.CANCELLED}
            workflow_ids = {
                workflow_id
                for workflow_id, workflow in self.workflows.items()
                if workflow.state in terminal
                and workflow.created_at < before
                and (tenant_id is None or workflow.tenant_id == tenant_id)
            }
            task_ids = {
                task_id
                for task_id, task in self.tasks.items()
                if task.workflow_id in workflow_ids
            }
            agent_ids = {
                agent_id
                for agent_id, agent in self.agents.items()
                if agent.workflow_id in workflow_ids
            }
            for task_id in task_ids:
                self.tasks.pop(task_id, None)
                self.leases.pop(task_id, None)
            for agent_id in agent_ids:
                self.agents.pop(agent_id, None)
                self.checkpoints.pop(agent_id, None)
                self.mailboxes.pop(agent_id, None)
            for workflow_id in workflow_ids:
                self.workflows.pop(workflow_id, None)
            return {
                "workflows": len(workflow_ids),
                "tasks": len(task_ids),
                "agents": len(agent_ids),
                "outbox": 0,
            }

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        async with self._lock:
            items = self.checkpoints[checkpoint.agent_id]
            if items and checkpoint.sequence <= items[-1].sequence:
                raise HarnessConflictError("checkpoint sequence must increase")
            items.append(checkpoint)

    async def latest_checkpoint(self, agent_id: str) -> Checkpoint | None:
        async with self._lock:
            items = self.checkpoints.get(agent_id, [])
            return items[-1] if items else None

    async def acquire_lease(self, lease: WorkerLease, *, now: datetime) -> bool:
        async with self._lock:
            current = self.leases.get(lease.task_id)
            if current is not None and current.active_at(now):
                return False
            self.leases[lease.task_id] = lease
            return True

    async def get_lease(self, task_id: str) -> WorkerLease | None:
        async with self._lock:
            return self.leases.get(task_id)

    async def update_lease(self, lease: WorkerLease) -> None:
        async with self._lock:
            current = self.leases.get(lease.task_id)
            if current is None or current.lease_id != lease.lease_id:
                raise HarnessConflictError("worker no longer owns this lease")
            self.leases[lease.task_id] = lease

    async def acquire_resources(self, lease: WorkerLease, *, now: datetime) -> bool:
        async with self._lock:
            for resource in lease.resource_keys:
                current = self.resource_leases.get(resource)
                if current is not None and current[0] != lease.lease_id and current[1] > now:
                    return False
            for resource in lease.resource_keys:
                self.resource_leases[resource] = (lease.lease_id, lease.expires_at)
            return True

    async def renew_resources(self, lease: WorkerLease) -> bool:
        async with self._lock:
            if any(
                self.resource_leases.get(resource, (None, None))[0] != lease.lease_id
                for resource in lease.resource_keys
            ):
                return False
            for resource in lease.resource_keys:
                self.resource_leases[resource] = (lease.lease_id, lease.expires_at)
            return True

    async def release_resources(self, lease: WorkerLease) -> None:
        async with self._lock:
            for resource in lease.resource_keys:
                current = self.resource_leases.get(resource)
                if current is not None and current[0] == lease.lease_id:
                    del self.resource_leases[resource]

    async def lease_active(self, task_id: str, *, now: datetime) -> bool:
        async with self._lock:
            lease = self.leases.get(task_id)
            return lease is not None and lease.active_at(now)

    async def register_worker(
        self, worker: WorkerRegistration, *, ttl_seconds: int
    ) -> None:
        del ttl_seconds
        async with self._lock:
            self.workers[worker.worker_id] = worker

    async def list_workers(
        self, tenant_id: str, *, now: datetime
    ) -> list[WorkerRegistration]:
        async with self._lock:
            return [
                worker
                for worker in self.workers.values()
                if worker.tenant_id == tenant_id and worker.expires_at > now
            ]

    async def close(self) -> None:
        return None
