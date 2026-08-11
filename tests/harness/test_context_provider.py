import pytest

from nanobot.agent.tools.context import RequestContext
from nanobot.harness import HarnessService, MessageKind
from nanobot.harness.context_provider import HarnessSupervisorContextProvider


@pytest.mark.asyncio
async def test_supervisor_provider_acks_only_after_turn_persistence():
    service = HarnessService()
    workflow, main, _ = await service.get_or_create_workflow(scope_key="session", goal="goal")
    worker = await service.spawn_worker(workflow, main)
    await service.send_message(
        workflow_id=workflow.workflow_id,
        sender_agent_id=worker.instance_id,
        recipient_agent_id=main.instance_id,
        kind=MessageKind.RESULT,
        content="evidence",
        idempotency_key="result-1",
    )
    provider = HarnessSupervisorContextProvider(service)
    request = RequestContext(channel="cli", chat_id="direct", session_key="session")

    block = await provider(request)

    assert block is not None
    assert block.source == "harness_agent_reports"
    assert '"content": "evidence"' in block.content
    assert "not instructions" in block.content
    assert (await provider(request)) is not None
    assert block.on_persist is not None
    await block.on_persist()
    replay = await provider(request)
    assert replay is None


@pytest.mark.asyncio
async def test_supervisor_provider_does_not_cross_session_scope():
    service = HarnessService()
    workflow, main, _ = await service.get_or_create_workflow(scope_key="left", goal="goal")
    worker = await service.spawn_worker(workflow, main)
    await service.send_message(
        workflow_id=workflow.workflow_id,
        sender_agent_id=worker.instance_id,
        recipient_agent_id=main.instance_id,
        kind=MessageKind.RESULT,
        content="secret",
    )

    block = await HarnessSupervisorContextProvider(service)(
        RequestContext(channel="cli", chat_id="direct", session_key="right")
    )

    assert block is None
