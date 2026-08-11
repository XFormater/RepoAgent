"""Distributed worker runtime that executes leased Harness tasks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta

from loguru import logger

from nanobot.harness.context import RuntimeContext
from nanobot.harness.models import (
    AgentInstance,
    AgentRole,
    AgentSpec,
    AgentState,
    ReviewResult,
    Task,
    TaskExecutionMode,
    TaskState,
    WorkerRegistration,
    WorkerState,
    utcnow,
)
from nanobot.harness.service import HarnessService
from nanobot.harness.store import HarnessConflictError

TaskExecutor = Callable[[RuntimeContext, AgentSpec], Awaitable[str | ReviewResult]]


class HarnessWorker:
    """Pull-ready, lease, heartbeat, execute, retry, and report runtime."""

    def __init__(
        self,
        *,
        service: HarnessService,
        agent: AgentInstance,
        worker_id: str,
        executor: TaskExecutor,
        lease_ttl_seconds: int = 60,
    ) -> None:
        self.service = service
        self.agent = agent
        self.worker_id = worker_id
        self.executor = executor
        self.lease_ttl_seconds = lease_ttl_seconds

    async def run_once(self) -> bool:
        await self.service.flush_outbox()
        # Rehydrate the versioned instance before every admission. A previous
        # process or the abandoned-task reconciler may have advanced its CAS
        # version while this worker was idle.
        self.agent = await self.service.store.get_agent(self.agent.instance_id)
        spec = await self.service.store.get_spec(self.agent.spec_id)
        await self.service.scheduler.recover_abandoned_tasks(self.agent.workflow_id)
        self.agent = await self.service.store.get_agent(self.agent.instance_id)
        candidates = await self.service.scheduler.ready_tasks(self.agent.workflow_id)
        task = next(
            (item for item in candidates if self.service.scheduler.admits(spec, item)),
            None,
        )
        if task is None:
            return False
        return await self.run_task(task)

    async def run_task(self, task: Task) -> bool:
        """Attempt one explicitly selected Task; return false if another Worker won."""
        started_at = time.perf_counter()
        outcome = "succeeded"
        self.agent = await self.service.store.get_agent(self.agent.instance_id)
        spec = await self.service.store.get_spec(self.agent.spec_id)
        try:
            lease = await self.service.scheduler.assign(
                task=task,
                agent=self.agent,
                spec=spec,
                worker_id=self.worker_id,
                ttl_seconds=self.lease_ttl_seconds,
            )
        except HarnessConflictError:
            return False

        stop = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal lease
            interval = max(1.0, self.lease_ttl_seconds / 3)
            while True:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                    return
                except TimeoutError:
                    lease = await self.service.scheduler.heartbeat(
                        lease, ttl_seconds=self.lease_ttl_seconds
                    )

        heartbeat_task = asyncio.create_task(heartbeat())
        context: RuntimeContext | None = None
        try:
            await self.service.scheduler.start(task, self.agent, lease)
            context = await self.service.runtime_context(
                self.agent.instance_id,
                task.task_id,
                consumer_id=self.worker_id,
            )
            isolation_mode = str(getattr(self.executor, "isolation_mode", "in_process"))
            self.service.metrics.event(
                "task.execution_isolation",
                role=spec.role.value,
                outcome=isolation_mode,
            )
            async with asyncio.timeout(spec.resource_budget.max_runtime_seconds):
                result = await self.executor(context, spec)
            if heartbeat_task.done() and (heartbeat_error := heartbeat_task.exception()):
                raise heartbeat_error
            if task.review_target_task_id is not None:
                if not isinstance(result, ReviewResult):
                    raise TypeError("Reviewer gate executor must return ReviewResult")
                await self.service.complete_review(
                    task=task,
                    agent=self.agent,
                    lease=lease,
                    approved=result.approved,
                    notes=result.notes,
                )
            else:
                content = result.notes if isinstance(result, ReviewResult) else result
                bounded = content[: spec.context_policy.max_message_chars]
                await self.service.complete_task(
                    task=task,
                    agent=self.agent,
                    lease=lease,
                    result=bounded,
                )
        except asyncio.CancelledError:
            outcome = "cancelled"
            await self.service.scheduler.cancel(task, self.agent, lease)
            raise
        except Exception as exc:
            outcome = "failed"
            logger.exception("Harness worker {} failed task {}", self.worker_id, task.task_id)
            try:
                await self.service.scheduler.fail(task, self.agent, lease, str(exc))
            except HarnessConflictError:
                logger.warning(
                    "Harness worker {} lost its lease for task {}; state left to recovery",
                    self.worker_id,
                    task.task_id,
                )
        finally:
            stop.set()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            if context is not None:
                await self.service.ack_context(context)
            self.service.metrics.duration(
                "task",
                time.perf_counter() - started_at,
                role=spec.role.value,
                outcome=outcome,
            )
            self.service.metrics.duration(
                "queue_delay",
                max(0.0, (lease.acquired_at - task.created_at).total_seconds()),
                role=spec.role.value,
            )
        return True

    async def run_until_idle(self, *, max_tasks: int = 100) -> int:
        processed = 0
        while processed < max_tasks and await self.run_once():
            processed += 1
        return processed

    async def run_forever(self, *, poll_interval: float = 1.0) -> None:
        while True:
            if not await self.run_once():
                await asyncio.sleep(poll_interval)


def retryable(task: Task) -> bool:
    now = utcnow()
    return (
        task.state == TaskState.READY
        and task.attempt < task.max_attempts
        and task.available_at <= now
        and (task.deadline_at is None or task.deadline_at > now)
    )


class HarnessPoolWorker:
    """Tenant-scoped, capability-aware worker pool with registration and draining."""

    def __init__(
        self,
        *,
        service: HarnessService,
        worker_id: str,
        tenant_id: str,
        roles: frozenset[AgentRole],
        capabilities: frozenset[str],
        executor: TaskExecutor,
        capacity: int = 4,
        heartbeat_ttl_seconds: int = 90,
        lease_ttl_seconds: int = 60,
    ) -> None:
        self.service = service
        self.worker_id = worker_id
        self.tenant_id = tenant_id
        self.roles = roles
        self.capabilities = capabilities
        self.executor = executor
        self.capacity = capacity
        self.heartbeat_ttl_seconds = heartbeat_ttl_seconds
        self.lease_ttl_seconds = lease_ttl_seconds
        self._draining = False
        self._active_tasks: set[str] = set()
        self._started_at = utcnow()

    async def heartbeat(self) -> WorkerRegistration:
        now = utcnow()
        registration = WorkerRegistration(
            worker_id=self.worker_id,
            tenant_id=self.tenant_id,
            roles=self.roles,
            capabilities=self.capabilities,
            capacity=self.capacity,
            active_tasks=frozenset(self._active_tasks),
            state=WorkerState.DRAINING if self._draining else WorkerState.ONLINE,
            started_at=self._started_at,
            heartbeat_at=now,
            expires_at=now + timedelta(seconds=self.heartbeat_ttl_seconds),
        )
        await self.service.store.register_worker(
            registration,
            ttl_seconds=self.heartbeat_ttl_seconds,
        )
        self.service.metrics.event(
            "worker.heartbeat",
            outcome="draining" if self._draining else "online",
        )
        return registration

    async def drain(self) -> None:
        """Stop claiming work; current Tasks retain heartbeat until they finish."""
        self._draining = True
        await self.heartbeat()

    def _admits(self, task: Task) -> bool:
        return (
            task.execution_mode == TaskExecutionMode.POOL
            and task.required_role in self.roles
            and task.required_capabilities <= self.capabilities
        )

    async def _execute_candidate(self, task: Task) -> bool:
        workflow = await self.service.store.get_workflow(task.workflow_id)
        supervisor = await self.service.store.get_agent(workflow.owner_agent_id)
        agent = await self.service.spawn_agent(
            workflow,
            supervisor,
            task.required_role,
        )
        self._active_tasks.add(task.task_id)
        await self.heartbeat()
        try:
            worker = HarnessWorker(
                service=self.service,
                agent=agent,
                worker_id=self.worker_id,
                executor=self.executor,
                lease_ttl_seconds=self.lease_ttl_seconds,
            )
            claimed = await worker.run_task(task)
            if not claimed:
                agent.state = AgentState.STOPPED
                agent.updated_at = utcnow()
                await self.service.store.put_agent(agent)
            return claimed
        finally:
            self._active_tasks.discard(task.task_id)
            await self.heartbeat()

    async def run_batch(self) -> int:
        await self.heartbeat()
        if self._draining:
            return 0
        await self.service.flush_outbox()
        await self.service.scheduler.expire_deadlines(self.tenant_id)
        for workflow_id in await self.service.store.list_active_workflow_ids(
            self.tenant_id
        ):
            await self.service.scheduler.recover_abandoned_tasks(workflow_id)
        available = max(0, self.capacity - len(self._active_tasks))
        if not available:
            return 0
        candidates = await self.service.store.list_ready_tasks(
            self.tenant_id,
            limit=max(available * 4, available),
        )
        selected = [task for task in candidates if self._admits(task)][:available]
        if not selected:
            return 0
        outcomes = await asyncio.gather(
            *(self._execute_candidate(task) for task in selected),
            return_exceptions=True,
        )
        completed = 0
        for task, outcome in zip(selected, outcomes):
            if isinstance(outcome, BaseException):
                logger.exception(
                    "Harness pool worker {} failed candidate {}",
                    self.worker_id,
                    task.task_id,
                    exception=outcome,
                )
            elif outcome:
                completed += 1
        return completed

    async def run_forever(self, *, poll_interval: float = 1.0) -> None:
        while not self._draining:
            if not await self.run_batch():
                await asyncio.sleep(poll_interval)
        while self._active_tasks:
            await self.heartbeat()
            await asyncio.sleep(min(1.0, self.heartbeat_ttl_seconds / 3))
