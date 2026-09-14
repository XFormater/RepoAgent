import asyncio
from datetime import timedelta

import pytest
from pydantic import ValidationError

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.harness import (
    AgentMessage,
    AgentRole,
    AgentSpec,
    ContextPolicy,
    HarnessCapacityError,
    HarnessConflictError,
    HarnessPoolWorker,
    HarnessService,
    HarnessWorker,
    MessageKind,
    PlannedWorkItem,
    ReviewResult,
    Task,
    TaskExecutionMode,
    TaskState,
    Workflow,
    WorkflowState,
    main_agent_spec,
    planner_agent_spec,
    reviewer_agent_spec,
    worker_agent_spec,
)
from nanobot.harness.models import utcnow


def test_main_and_worker_roles_are_explicit_and_distinct():
    main = main_agent_spec()
    worker = worker_agent_spec()

    assert main.role is AgentRole.SUPERVISOR
    assert "delegate" in main.capabilities
    assert worker.role is AgentRole.WORKER
    assert "delegate" not in worker.capabilities
    assert "spawn" not in worker.allowed_tools
    assert worker.context_policy.inherit_parent_messages == 0


@pytest.mark.asyncio
async def test_message_bus_audit_assigns_trace_and_records_lifecycle():
    bus = MessageBus()
    observed: list[tuple[str, dict[str, object]]] = []

    async def audit(event_type: str, payload: dict[str, object]) -> None:
        observed.append((event_type, payload))

    bus.set_audit_observer(audit)
    message = InboundMessage(channel="cli", sender_id="user", chat_id="chat", content="hello")
    await bus.publish_inbound(message)
    await bus.consume_inbound()

    assert isinstance(message.metadata.get("trace_id"), str)
    assert [item[0] for item in observed] == [
        "bus.inbound.published",
        "bus.inbound.consumed",
    ]
    assert observed[0][1]["content"] == "hello"


@pytest.mark.asyncio
async def test_durable_audit_redacts_credential_shaped_payloads():
    service = HarnessService()
    await service.store.record_audit_event(
        "test.audit",
        component="test",
        payload={"api_key": "secret-value", "content": "Bearer abcdefghijklmnopqrst"},
    )

    event = service.store.audit_events[-1]
    assert event["payload"]["api_key"] == "[REDACTED]"
    assert "abcdefghijklmnopqrst" not in event["payload"]["content"]
    assert len(event["payload_sha256"]) == 64


def test_workflow_rejects_cycles():
    workflow_id = "wf"
    first = Task(task_id="a", workflow_id=workflow_id, title="a", input="a", dependencies={"b"})
    second = Task(task_id="b", workflow_id=workflow_id, title="b", input="b", dependencies={"a"})

    with pytest.raises(ValidationError, match="dependency cycle"):
        Workflow(
            workflow_id=workflow_id, goal="goal", owner_agent_id="owner", tasks=[first, second]
        )


def test_agent_message_disallows_self_route():
    with pytest.raises(ValidationError, match="cross an agent boundary"):
        AgentMessage(
            workflow_id="wf",
            sender_agent_id="same",
            recipient_agent_id="same",
            kind=MessageKind.CONTROL,
            content="x",
            idempotency_key="one",
        )


@pytest.mark.asyncio
async def test_scheduler_enforces_role_capability_and_lease_ownership():
    service = HarnessService()
    workflow, main = await service.create_workflow("goal")
    task = await service.add_task(
        workflow,
        title="research",
        task_input="inspect",
        required_capabilities=frozenset({"research"}),
    )
    worker = await service.spawn_worker(workflow, main)
    await service.scheduler.ready_tasks(workflow.workflow_id)
    spec = await service.store.get_spec(worker.spec_id)
    lease = await service.scheduler.assign(task=task, agent=worker, spec=spec, worker_id="worker-1")

    assert task.state is TaskState.LEASED
    with pytest.raises(HarnessConflictError, match="only ready"):
        await service.scheduler.assign(task=task, agent=worker, spec=spec, worker_id="worker-2")
    renewed = await service.scheduler.heartbeat(lease)
    assert renewed.expires_at > lease.expires_at


@pytest.mark.asyncio
async def test_mailboxes_are_recipient_isolated_and_idempotent():
    service = HarnessService()
    workflow, main = await service.create_workflow("goal")
    left = await service.spawn_worker(workflow, main)
    right = await service.spawn_worker(workflow, main)

    inserted = await service.send_message(
        workflow_id=workflow.workflow_id,
        sender_agent_id=main.instance_id,
        recipient_agent_id=left.instance_id,
        kind=MessageKind.TASK,
        content="left-only",
        idempotency_key="stable-key",
    )
    duplicate = await service.send_message(
        workflow_id=workflow.workflow_id,
        sender_agent_id=main.instance_id,
        recipient_agent_id=left.instance_id,
        kind=MessageKind.TASK,
        content="duplicate",
        idempotency_key="stable-key",
    )

    assert inserted is True
    assert duplicate is False
    assert [
        m.content for m in await service.store.inbox(left.instance_id, now=utcnow(), limit=10)
    ] == ["left-only"]
    assert await service.store.inbox(right.instance_id, now=utcnow(), limit=10) == []


@pytest.mark.asyncio
async def test_runtime_context_never_inherits_parent_transcript_and_is_bounded():
    service = HarnessService()
    workflow, main = await service.create_workflow("private parent goal")
    worker = await service.spawn_worker(workflow, main)
    bounded = AgentSpec(
        spec_id="tiny-worker",
        role=AgentRole.WORKER,
        objective="bounded",
        capabilities=frozenset({"analyze"}),
        context_policy=ContextPolicy(max_messages=2, max_chars=1_000, max_message_chars=256),
    )
    await service.register(bounded)
    worker.spec_id = bounded.spec_id
    task = await service.add_task(
        workflow,
        title="task",
        task_input="explicit handoff" + "x" * 300,
    )
    for index, content in enumerate(["first-parent-secret", "second", "third"]):
        await service.send_message(
            workflow_id=workflow.workflow_id,
            sender_agent_id=main.instance_id,
            recipient_agent_id=worker.instance_id,
            kind=MessageKind.TASK,
            content=content,
            idempotency_key=f"m-{index}",
        )

    context = await service.runtime_context(worker.instance_id, task.task_id)

    assert context.workflow_id == workflow.workflow_id
    assert context.task_id == task.task_id
    assert context.tenant_id == "default"
    assert context.task_input == ("explicit handoff" + "x" * 300)[:256]
    assert [message.content for message in context.messages] == ["second", "third"]
    assert "private parent goal" not in context.model_dump_json()
    assert "first-parent-secret" not in context.model_dump_json()
    assert context.truncated is True


@pytest.mark.asyncio
async def test_checkpoint_is_agent_owned_and_monotonic():
    service = HarnessService()
    workflow, main = await service.create_workflow("goal")
    worker = await service.spawn_worker(workflow, main)

    saved = await service.scheduler.checkpoint(worker, {"cursor": 3}, sequence=1)
    assert (await service.store.latest_checkpoint(worker.instance_id)) == saved
    assert await service.store.latest_checkpoint(main.instance_id) is None
    with pytest.raises(HarnessConflictError, match="sequence"):
        await service.scheduler.checkpoint(worker, {"cursor": 2}, sequence=1)


@pytest.mark.asyncio
async def test_expired_messages_and_leases_are_not_active():
    message = AgentMessage(
        workflow_id="wf",
        sender_agent_id="a",
        recipient_agent_id="b",
        kind=MessageKind.RESULT,
        content="old",
        idempotency_key="old",
        expires_at=utcnow() - timedelta(seconds=1),
    )
    service = HarnessService()
    await service.store.send(message)
    assert await service.store.inbox("b", now=utcnow(), limit=10) == []


@pytest.mark.asyncio
async def test_workflow_and_supervisor_are_stable_per_tenant_scope():
    service = HarnessService()
    first, first_owner, created = await service.get_or_create_workflow(
        scope_key="session-1", goal="first", tenant_id="tenant-a"
    )
    second, second_owner, created_again = await service.get_or_create_workflow(
        scope_key="session-1", goal="second", tenant_id="tenant-a"
    )
    other, _, _ = await service.get_or_create_workflow(
        scope_key="session-1", goal="other", tenant_id="tenant-b"
    )

    assert created is True and created_again is False
    assert first.workflow_id == second.workflow_id
    assert first_owner.instance_id == second_owner.instance_id
    assert other.workflow_id != first.workflow_id


@pytest.mark.asyncio
async def test_dynamic_workflow_cannot_finish_before_it_is_sealed():
    service = HarnessService()
    workflow, owner = await service.create_workflow("dynamic", sealed=False)
    task = await service.add_task(workflow, title="plan", task_input="plan")
    worker = await service.spawn_worker(workflow, owner)
    spec = await service.store.get_spec(worker.spec_id)
    lease = await service.scheduler.assign(
        task=task,
        agent=worker,
        spec=spec,
        worker_id="worker",
    )
    await service.complete_task(
        task=task,
        agent=worker,
        lease=lease,
        result="done",
        notify_supervisor=False,
    )

    assert (await service.store.get_workflow(workflow.workflow_id)).state is WorkflowState.ACTIVE
    sealed = await service.seal_workflow(workflow.workflow_id)
    assert sealed.sealed is True
    assert sealed.state is WorkflowState.SUCCEEDED


@pytest.mark.asyncio
async def test_task_creation_is_idempotent_within_a_workflow():
    service = HarnessService()
    workflow, _ = await service.create_workflow("goal")

    first, created = await service.ensure_task(
        workflow,
        title="research",
        task_input="inspect",
        idempotency_key="message-1:call-1",
    )
    second, created_again = await service.ensure_task(
        workflow,
        title="research retry",
        task_input="inspect",
        idempotency_key="message-1:call-1",
    )

    assert created is True and created_again is False
    assert first.task_id == second.task_id
    assert len(await service.store.list_tasks(workflow.workflow_id)) == 1
    assert service.metrics.snapshot()["task.reused.ok"] == 1


def test_planner_and_reviewer_have_non_mutating_role_policies():
    planner = planner_agent_spec()
    reviewer = reviewer_agent_spec()

    assert planner.role is AgentRole.PLANNER
    assert reviewer.role is AgentRole.REVIEWER
    assert "write_file" not in planner.allowed_tools
    assert "apply_patch" not in reviewer.allowed_tools
    assert "spawn" not in planner.allowed_tools | reviewer.allowed_tools


@pytest.mark.asyncio
async def test_failure_releases_lease_and_transitions_to_retry():
    service = HarnessService()
    workflow, main = await service.create_workflow("goal")
    task = await service.add_task(
        workflow,
        title="task",
        task_input="input",
        retry_backoff_seconds=0,
    )
    worker = await service.spawn_worker(workflow, main)
    await service.scheduler.ready_tasks(workflow.workflow_id)
    spec = await service.store.get_spec(worker.spec_id)
    lease = await service.scheduler.assign(task=task, agent=worker, spec=spec, worker_id="worker")

    state = await service.scheduler.fail(task, worker, lease, "temporary")

    assert state is TaskState.READY
    assert task.assigned_agent_id is None
    assert service.metrics.snapshot()["task.failed.retry"] == 1


@pytest.mark.asyncio
async def test_expired_lease_is_recovered_and_agent_is_unblocked():
    service = HarnessService()
    workflow, main = await service.create_workflow("goal")
    task = await service.add_task(workflow, title="task", task_input="input")
    worker = await service.spawn_worker(workflow, main)
    await service.scheduler.ready_tasks(workflow.workflow_id)
    spec = await service.store.get_spec(worker.spec_id)
    await service.scheduler.assign(
        task=task,
        agent=worker,
        spec=spec,
        worker_id="crashed-worker",
        ttl_seconds=1,
    )
    service.store.leases.pop(task.task_id)

    recovered = await service.scheduler.recover_abandoned_tasks(workflow.workflow_id)

    assert recovered == 1
    assert task.state is TaskState.READY
    assert task.assigned_agent_id is None
    assert worker.current_task_id is None
    assert service.metrics.snapshot()["task.recovered.ok"] == 1


@pytest.mark.asyncio
async def test_worker_that_lost_lease_is_fenced_from_failure_transition():
    service = HarnessService()
    workflow, main = await service.create_workflow("goal")
    task = await service.add_task(workflow, title="task", task_input="input")
    worker = await service.spawn_worker(workflow, main)
    await service.scheduler.ready_tasks(workflow.workflow_id)
    spec = await service.store.get_spec(worker.spec_id)
    lease = await service.scheduler.assign(
        task=task,
        agent=worker,
        spec=spec,
        worker_id="old-worker",
    )
    service.store.leases.pop(task.task_id)

    with pytest.raises(HarnessConflictError, match="no longer owns"):
        await service.scheduler.fail(task, worker, lease, "late failure")

    assert task.state is TaskState.LEASED
    assert task.error is None


@pytest.mark.asyncio
async def test_worker_runtime_retries_then_completes_and_reports_result():
    service = HarnessService()
    workflow, main = await service.create_workflow("goal")
    task = await service.add_task(
        workflow,
        title="task",
        task_input="input",
        retry_backoff_seconds=0,
    )
    agent = await service.spawn_worker(workflow, main)
    calls = 0

    async def executor(context, spec):
        nonlocal calls
        calls += 1
        assert context.agent_id == agent.instance_id
        assert spec.role is AgentRole.WORKER
        if calls == 1:
            raise RuntimeError("temporary")
        return "verified"

    worker = HarnessWorker(
        service=service,
        agent=agent,
        worker_id="worker-1",
        executor=executor,
        lease_ttl_seconds=3,
    )
    processed = await worker.run_until_idle()

    assert processed == 2
    assert task.state is TaskState.SUCCEEDED
    assert task.result == "verified"
    assert task.attempt == 2
    messages = await service.store.inbox(main.instance_id, now=utcnow(), limit=10)
    assert [message.content for message in messages] == ["verified"]


@pytest.mark.asyncio
async def test_cancel_task_interrupts_a_running_worker_executor():
    service = HarnessService()
    workflow, main = await service.create_workflow("cancel running task")
    task = await service.add_task(workflow, title="task", task_input="input")
    agent = await service.spawn_worker(workflow, main)
    started = asyncio.Event()

    async def executor(context, spec):
        del context, spec
        started.set()
        await asyncio.Event().wait()
        return "must not complete"

    worker = HarnessWorker(
        service=service,
        agent=agent,
        worker_id="worker-cancel",
        executor=executor,
    )
    run = asyncio.create_task(worker.run_once())
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await service.cancel_task(task.task_id) is True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, timeout=2)

    assert task.state is TaskState.CANCELLED
    assert (await service.store.get_workflow(workflow.workflow_id)).state is WorkflowState.CANCELLED
    assert any(event["event_type"] == "harness.task.cancelled" for event in service.store.audit_events)


@pytest.mark.asyncio
async def test_cancel_task_cancels_a_queued_task_without_a_lease():
    service = HarnessService()
    workflow, _ = await service.create_workflow("cancel queued task")
    task = await service.add_task(workflow, title="task", task_input="input")

    assert await service.cancel_task(task.task_id) is True
    assert task.state is TaskState.CANCELLED
    assert (await service.store.get_workflow(workflow.workflow_id)).state is WorkflowState.CANCELLED


@pytest.mark.asyncio
async def test_cross_tenant_message_and_context_are_rejected():
    service = HarnessService()
    left, left_main, _ = await service.get_or_create_workflow(
        scope_key="left", goal="left", tenant_id="tenant-left"
    )
    right, right_main, _ = await service.get_or_create_workflow(
        scope_key="right", goal="right", tenant_id="tenant-right"
    )
    right_task = await service.add_task(right, title="right", task_input="secret")

    with pytest.raises(PermissionError, match="cross-workflow"):
        await service.send_message(
            workflow_id=left.workflow_id,
            tenant_id=left.tenant_id,
            sender_agent_id=left_main.instance_id,
            recipient_agent_id=right_main.instance_id,
            kind=MessageKind.CONTROL,
            content="leak",
        )
    with pytest.raises(PermissionError, match="another tenant"):
        await service.runtime_context(left_main.instance_id, right_task.task_id)


@pytest.mark.asyncio
async def test_worker_result_must_pass_reviewer_gate_with_isolated_dependency_context():
    service = HarnessService()
    workflow, supervisor = await service.create_workflow("private supervisor goal")
    work = await service.add_task(
        workflow,
        title="Analyze evidence",
        task_input="inspect the source",
        required_capabilities=frozenset({"analyze"}),
    )
    review = await service.add_review_gate(workflow, work)
    worker = await service.spawn_agent(workflow, supervisor, AgentRole.WORKER)
    reviewer = await service.spawn_agent(workflow, supervisor, AgentRole.REVIEWER)

    ready = await service.scheduler.ready_tasks(workflow.workflow_id)
    assert [task.task_id for task in ready] == [work.task_id]
    worker_spec = await service.store.get_spec(worker.spec_id)
    worker_lease = await service.scheduler.assign(
        task=work,
        agent=worker,
        spec=worker_spec,
        worker_id="worker",
    )
    await service.complete_task(
        task=work,
        agent=worker,
        lease=worker_lease,
        result="source-backed finding",
    )

    assert workflow.state is not WorkflowState.SUCCEEDED
    ready = await service.scheduler.ready_tasks(workflow.workflow_id)
    assert [task.task_id for task in ready] == [review.task_id]
    reviewer_context = await service.runtime_context(reviewer.instance_id, review.task_id)
    assert [item.result for item in reviewer_context.dependency_results] == [
        "source-backed finding"
    ]
    assert "private supervisor goal" not in reviewer_context.render()

    reviewer_spec = await service.store.get_spec(reviewer.spec_id)
    reviewer_lease = await service.scheduler.assign(
        task=review,
        agent=reviewer,
        spec=reviewer_spec,
        worker_id="reviewer",
    )
    await service.complete_task(
        task=review,
        agent=reviewer,
        lease=reviewer_lease,
        result="approved",
    )
    assert workflow.state is WorkflowState.SUCCEEDED


@pytest.mark.asyncio
async def test_fencing_token_blocks_forged_terminal_transition():
    service = HarnessService()
    workflow, supervisor = await service.create_workflow("goal")
    task = await service.add_task(workflow, title="task", task_input="input")
    worker = await service.spawn_worker(workflow, supervisor)
    await service.scheduler.ready_tasks(workflow.workflow_id)
    spec = await service.store.get_spec(worker.spec_id)
    lease = await service.scheduler.assign(
        task=task,
        agent=worker,
        spec=spec,
        worker_id="worker",
    )
    task.fencing_token += 1

    with pytest.raises(HarnessConflictError, match="fencing token"):
        await service.scheduler.complete(task, worker, lease, "late result")

    assert task.state is TaskState.LEASED


@pytest.mark.asyncio
async def test_task_admission_applies_tenant_backpressure_but_allows_replay():
    service = HarnessService(max_queued_tasks_per_tenant=1)
    workflow, _ = await service.create_workflow("goal")
    first, created = await service.ensure_task(
        workflow,
        title="one",
        task_input="one",
        idempotency_key="stable",
    )
    replay, replay_created = await service.ensure_task(
        workflow,
        title="one replay",
        task_input="one",
        idempotency_key="stable",
    )

    assert created is True and replay_created is False
    assert replay.task_id == first.task_id
    with pytest.raises(HarnessCapacityError, match="queued task limit"):
        await service.add_task(workflow, title="two", task_input="two")


@pytest.mark.asyncio
async def test_pool_worker_registers_runs_by_priority_and_drains():
    service = HarnessService()
    workflow, _ = await service.create_workflow("goal")
    low = await service.add_task(
        workflow,
        title="low",
        task_input="low",
        priority=1,
    )
    high = await service.add_task(
        workflow,
        title="high",
        task_input="high",
        priority=10,
    )
    execution_order: list[str] = []

    async def execute(context, spec):
        assert spec.role is AgentRole.WORKER
        execution_order.append(context.task_input)
        return f"done:{context.task_input}"

    pool = HarnessPoolWorker(
        service=service,
        worker_id="pool-1",
        tenant_id="default",
        roles=frozenset({AgentRole.WORKER}),
        capabilities=frozenset({"research", "code", "analyze"}),
        executor=execute,
        capacity=1,
    )

    assert await pool.run_batch() == 1
    assert execution_order == ["high"]
    assert high.state is TaskState.SUCCEEDED
    assert low.state is TaskState.READY
    workers = await service.store.list_workers("default", now=utcnow())
    assert [worker.worker_id for worker in workers] == ["pool-1"]

    await pool.drain()
    assert await pool.run_batch() == 0
    assert low.state is TaskState.READY


@pytest.mark.asyncio
async def test_pool_worker_does_not_steal_inline_task():
    service = HarnessService()
    workflow, _ = await service.create_workflow("goal")
    inline = await service.add_task(
        workflow,
        title="request-local",
        task_input="inline",
        execution_mode=TaskExecutionMode.INLINE,
        priority=100,
    )
    pooled = await service.add_task(
        workflow,
        title="distributed",
        task_input="pool",
        execution_mode=TaskExecutionMode.POOL,
    )
    executions: list[str] = []

    async def execute(context, _spec):
        executions.append(context.task_input)
        return "done"

    pool = HarnessPoolWorker(
        service=service,
        worker_id="pool-isolation",
        tenant_id="default",
        roles=frozenset({AgentRole.WORKER}),
        capabilities=frozenset({"research", "code", "analyze"}),
        executor=execute,
        capacity=2,
    )

    assert await pool.run_batch() == 1
    assert executions == ["pool"]
    assert inline.state is TaskState.READY
    assert pooled.state is TaskState.SUCCEEDED


@pytest.mark.asyncio
async def test_exhausted_task_enters_tenant_scoped_dlq_and_can_be_requeued():
    service = HarnessService()
    workflow, main = await service.create_workflow("goal", tenant_id="tenant-a")
    task = await service.add_task(
        workflow,
        title="fragile",
        task_input="input",
        max_attempts=1,
        retry_backoff_seconds=0,
    )
    worker = await service.spawn_worker(workflow, main)
    spec = await service.store.get_spec(worker.spec_id)
    lease = await service.scheduler.assign(
        task=task,
        agent=worker,
        spec=spec,
        worker_id="worker",
    )

    state = await service.scheduler.fail(task, worker, lease, "permanent")

    assert state is TaskState.DEAD_LETTER
    assert workflow.state is WorkflowState.FAILED
    assert [item.task_id for item in await service.store.list_dead_letters("tenant-a")] == [
        task.task_id
    ]
    with pytest.raises(PermissionError, match="another tenant"):
        await service.requeue_dead_letter(task.task_id, tenant_id="tenant-b")

    replay = await service.requeue_dead_letter(task.task_id, tenant_id="tenant-a")
    assert replay.state is TaskState.READY
    assert replay.attempt == 0
    assert workflow.state is WorkflowState.ACTIVE


@pytest.mark.asyncio
async def test_scheduler_moves_expired_ready_tasks_to_dlq():
    service = HarnessService()
    workflow, _ = await service.create_workflow("goal", tenant_id="tenant-a")
    now = utcnow()
    task = await service.add_task(
        workflow,
        title="expired",
        task_input="input",
        available_at=now - timedelta(seconds=2),
        deadline_at=now - timedelta(seconds=1),
    )

    assert await service.scheduler.expire_deadlines("tenant-a") == 1
    assert task.state is TaskState.DEAD_LETTER
    assert task.error == "task deadline expired before execution"


@pytest.mark.asyncio
async def test_role_dag_enforces_planner_worker_reviewer_quality_gate():
    service = HarnessService()
    workflow, owner, planner, workers, reviews = await service.create_role_dag(
        goal="research",
        tenant_id="tenant-a",
        work_items=[
            PlannedWorkItem(
                title="collect evidence",
                task_input="find sources",
                required_capabilities=frozenset({"research"}),
            )
        ],
    )
    worker_task = workers[0]
    review_task = reviews[0]
    assert planner.state is TaskState.READY
    assert worker_task.state is TaskState.PENDING
    assert review_task.state is TaskState.PENDING

    planner_agent = await service.spawn_agent(workflow, owner, AgentRole.PLANNER)
    planner_spec = await service.store.get_spec(planner_agent.spec_id)
    planner_lease = await service.scheduler.assign(
        task=planner,
        agent=planner_agent,
        spec=planner_spec,
        worker_id="planner-pool",
    )
    await service.complete_task(
        task=planner,
        agent=planner_agent,
        lease=planner_lease,
        result="plan",
    )
    assert worker_task.state is TaskState.READY

    worker_agent = await service.spawn_agent(workflow, owner, AgentRole.WORKER)
    worker_spec = await service.store.get_spec(worker_agent.spec_id)
    worker_lease = await service.scheduler.assign(
        task=worker_task,
        agent=worker_agent,
        spec=worker_spec,
        worker_id="worker-pool",
    )
    await service.complete_task(
        task=worker_task,
        agent=worker_agent,
        lease=worker_lease,
        result="unsupported claim",
    )
    assert review_task.state is TaskState.READY
    assert workflow.state is WorkflowState.ACTIVE

    reviewer_agent = await service.spawn_agent(workflow, owner, AgentRole.REVIEWER)
    reviewer_spec = await service.store.get_spec(reviewer_agent.spec_id)
    reviewer_lease = await service.scheduler.assign(
        task=review_task,
        agent=reviewer_agent,
        spec=reviewer_spec,
        worker_id="review-pool",
    )
    await service.complete_review(
        task=review_task,
        agent=reviewer_agent,
        lease=reviewer_lease,
        approved=False,
        notes="missing evidence",
    )

    assert review_task.state is TaskState.DEAD_LETTER
    assert workflow.state is WorkflowState.FAILED
    messages = await service.store.inbox(owner.instance_id, now=utcnow(), limit=10)
    assert any(message.content == "REJECTED: missing evidence" for message in messages)


@pytest.mark.asyncio
async def test_child_agent_cannot_spawn_sibling():
    service = HarnessService()
    workflow, owner = await service.create_workflow("goal")
    child = await service.spawn_agent(workflow, owner, AgentRole.WORKER)

    with pytest.raises(PermissionError, match="workflow supervisor"):
        await service.spawn_agent(workflow, child, AgentRole.WORKER)


@pytest.mark.asyncio
async def test_checkpoint_is_secret_safe_and_size_bounded():
    service = HarnessService(checkpoint_max_bytes=1_500)
    workflow, owner = await service.create_workflow("goal")
    child = await service.spawn_agent(workflow, owner, AgentRole.WORKER)

    checkpoint = await service.scheduler.checkpoint(
        child,
        {
            "phase": "tool",
            "iteration": 7,
            "api_key": "must-not-persist",
            "provider_state": {"encrypted_content": "must-not-persist"},
            "messages": ["very large " * 2_000],
        },
        1,
    )

    encoded = checkpoint.model_dump_json()
    assert "must-not-persist" not in encoded
    assert len(str(checkpoint.state).encode()) <= 1_500
    assert checkpoint.state["_checkpoint_truncated"] is True


@pytest.mark.asyncio
async def test_exclusive_resource_fences_parallel_workspace_writers():
    service = HarnessService()
    workflow, owner = await service.create_workflow("goal")
    first = await service.add_task(
        workflow,
        title="first",
        task_input="write",
        resource_keys=frozenset({"workspace:/project"}),
    )
    second = await service.add_task(
        workflow,
        title="second",
        task_input="write",
        resource_keys=frozenset({"workspace:/project"}),
    )
    first_agent = await service.spawn_agent(workflow, owner, AgentRole.WORKER)
    second_agent = await service.spawn_agent(workflow, owner, AgentRole.WORKER)
    spec = await service.store.get_spec(first_agent.spec_id)
    first_lease = await service.scheduler.assign(
        task=first,
        agent=first_agent,
        spec=spec,
        worker_id="one",
    )

    with pytest.raises(HarnessConflictError, match="required resource"):
        await service.scheduler.assign(
            task=second,
            agent=second_agent,
            spec=spec,
            worker_id="two",
        )

    await service.complete_task(
        task=first,
        agent=first_agent,
        lease=first_lease,
        result="done",
    )
    second_lease = await service.scheduler.assign(
        task=second,
        agent=second_agent,
        spec=spec,
        worker_id="two",
    )
    assert second_lease.resource_keys == frozenset({"workspace:/project"})


@pytest.mark.asyncio
async def test_diagnostics_and_tenant_scoped_retention():
    service = HarnessService(retention_days=7)
    old_workflow, old_owner = await service.create_workflow("old", tenant_id="tenant-a")
    old_workflow.created_at = utcnow() - timedelta(days=8)
    old_workflow.state = WorkflowState.SUCCEEDED
    await service.store.put_workflow(old_workflow)
    other_workflow, _ = await service.create_workflow("other", tenant_id="tenant-b")

    before = await service.diagnostics(tenant_id="tenant-a")
    outcome = await service.maintenance(tenant_id="tenant-a")

    assert before["health"]["store"] is True
    assert before["storage"]["workflows"] == 1
    assert outcome["pruned"]["workflows"] == 1
    with pytest.raises(KeyError):
        await service.store.get_agent(old_owner.instance_id)
    assert await service.store.get_workflow(other_workflow.workflow_id) == other_workflow


@pytest.mark.asyncio
async def test_pool_requires_machine_readable_reviewer_gate_decision():
    service = HarnessService()
    workflow, owner = await service.create_workflow("goal")
    target = await service.add_task(workflow, title="work", task_input="work")
    target_agent = await service.spawn_agent(workflow, owner, AgentRole.WORKER)
    target_spec = await service.store.get_spec(target_agent.spec_id)
    target_lease = await service.scheduler.assign(
        task=target,
        agent=target_agent,
        spec=target_spec,
        worker_id="worker",
    )
    await service.complete_task(
        task=target,
        agent=target_agent,
        lease=target_lease,
        result="claim",
    )
    gate = await service.add_review_gate(workflow, target)
    await service.scheduler.ready_tasks(workflow.workflow_id)
    reviewer = await service.spawn_agent(workflow, owner, AgentRole.REVIEWER)

    async def invalid_review(context, spec):
        return "looks good"

    worker = HarnessWorker(
        service=service,
        agent=reviewer,
        worker_id="reviewer",
        executor=invalid_review,
    )
    assert await worker.run_task(gate) is True
    assert gate.state is TaskState.READY
    assert gate.error == "Reviewer gate executor must return ReviewResult"
    gate.available_at = utcnow()
    await service.store.put_task(gate)

    async def valid_review(context, spec):
        return ReviewResult(approved=True, notes="evidence verified")

    worker.executor = valid_review
    assert await worker.run_task(gate) is True
    assert gate.state is TaskState.SUCCEEDED
    assert gate.result == "APPROVED: evidence verified"
