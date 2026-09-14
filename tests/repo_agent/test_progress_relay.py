from __future__ import annotations

from typing import Any

import pytest

from nanobot.bus.outbound_events import ProgressEvent, StreamDeltaEvent, StreamEndEvent
from nanobot.bus.queue import MessageBus
from nanobot.repo_agent.progress_relay import RepositoryResearchProgressRelay


class _AuditStore:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events

    async def latest_audit_event_id(self) -> int:
        return 0

    async def list_audit_events_after(
        self, event_id: int, *, event_types: frozenset[str], limit: int = 100
    ) -> list[dict[str, Any]]:
        return [
            event for event in self.events
            if int(event["event_id"]) > event_id and event["event_type"] in event_types
        ][:limit]


@pytest.mark.asyncio
async def test_progress_relay_emits_trace_then_streamed_report() -> None:
    bus = MessageBus()
    relay = RepositoryResearchProgressRelay(store=_AuditStore([]), bus=bus)  # type: ignore[arg-type]

    await relay._publish(
        {
            "event_id": 1,
            "event_type": "repository_research.progress",
            "session_key": "websocket:chat-1",
            "payload": {"research_id": "research-1", "content": "正在下载仓库…"},
        }
    )
    await relay._publish(
        {
            "event_id": 2,
            "event_type": "repository_research.completed",
            "session_key": "websocket:chat-1",
            "payload": {"research_id": "research-1", "report": "# 报告\n\n完成"},
        }
    )

    messages = [await bus.consume_outbound() for _ in range(4)]
    assert messages[0].chat_id == "chat-1"
    assert isinstance(messages[0].event, ProgressEvent)
    assert messages[0].content == "正在下载仓库…"
    assert isinstance(messages[1].event, ProgressEvent)
    assert isinstance(messages[2].event, StreamDeltaEvent)
    assert messages[2].content == "# 报告\n\n完成"
    assert isinstance(messages[3].event, StreamEndEvent)


@pytest.mark.asyncio
async def test_progress_relay_forwards_live_report_deltas_without_waiting_for_completion() -> None:
    bus = MessageBus()
    relay = RepositoryResearchProgressRelay(store=_AuditStore([]), bus=bus)  # type: ignore[arg-type]

    await relay._publish(
        {
            "event_id": 1,
            "event_type": "repository_research.report_delta",
            "session_key": "websocket:chat-1",
            "payload": {"research_id": "research-1", "content": "第一个"},
        }
    )
    delta = await bus.consume_outbound()
    assert isinstance(delta.event, StreamDeltaEvent)
    assert delta.content == "第一个"
    assert delta.metadata["stream_live"] is True

    await relay._publish(
        {
            "event_id": 2,
            "event_type": "repository_research.completed",
            "session_key": "websocket:chat-1",
            "payload": {"research_id": "research-1", "report": "第一个完整报告"},
        }
    )
    end = await bus.consume_outbound()
    assert isinstance(end.event, StreamEndEvent)


@pytest.mark.asyncio
async def test_progress_relay_ignores_non_websocket_sessions() -> None:
    bus = MessageBus()
    relay = RepositoryResearchProgressRelay(store=_AuditStore([]), bus=bus)  # type: ignore[arg-type]

    await relay._publish(
        {
            "event_id": 1,
            "event_type": "repository_research.progress",
            "session_key": "telegram:chat-1",
            "payload": {"content": "不会进入 WebUI"},
        }
    )

    assert bus.outbound_size == 0


@pytest.mark.asyncio
async def test_progress_relay_decodes_asyncpg_json_payload() -> None:
    bus = MessageBus()
    relay = RepositoryResearchProgressRelay(store=_AuditStore([]), bus=bus)  # type: ignore[arg-type]

    await relay._publish(
        {
            "event_id": 1,
            "event_type": "repository_research.progress",
            "session_key": "websocket:chat-1",
            "payload": '{"research_id":"research-1","content":"正在分析源码…"}',
        }
    )

    message = await bus.consume_outbound()
    assert isinstance(message.event, ProgressEvent)
    assert message.content == "正在分析源码…"


@pytest.mark.asyncio
async def test_progress_relay_suppresses_reasoning_token_events() -> None:
    bus = MessageBus()
    relay = RepositoryResearchProgressRelay(store=_AuditStore([]), bus=bus)  # type: ignore[arg-type]

    await relay._publish(
        {
            "event_id": 1,
            "event_type": "repository_research.progress",
            "session_key": "websocket:chat-1",
            "payload": {
                "content": " Completed step open",
                "detail": {"reasoning": True},
            },
        }
    )

    assert bus.outbound_size == 0


@pytest.mark.asyncio
async def test_progress_relay_explains_retry_for_repository_task() -> None:
    bus = MessageBus()
    relay = RepositoryResearchProgressRelay(store=_AuditStore([]), bus=bus)  # type: ignore[arg-type]

    await relay._publish(
        {
            "event_id": 1,
            "event_type": "harness.task.failed",
            "session_key": "websocket:chat-1",
            "payload": {
                "state": "ready",
                "error": "analysis budget exceeded",
                "metadata": {"domain": "repository_research"},
            },
        }
    )

    message = await bus.consume_outbound()
    assert isinstance(message.event, ProgressEvent)
    assert "自动重试" in message.content
    assert "analysis budget exceeded" in message.content
