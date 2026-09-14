"""Bridge durable POOL research events to live WebSocket clients."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import ProgressEvent, StreamDeltaEvent, StreamEndEvent
from nanobot.bus.queue import MessageBus
from nanobot.harness.store import HarnessStore


class RepositoryResearchProgressRelay:
    """Publish Worker audit events without sharing memory with the Worker.

    PostgreSQL is the process boundary: the Worker appends redacted audit
    events, and this Gateway-owned relay polls those durable rows before
    emitting normal WebSocket progress/stream frames.
    """

    _EVENT_TYPES = frozenset(
        {
            "repository_research.queued",
            "repository_research.progress",
            "repository_research.report_delta",
            "repository_research.completed",
            "harness.task.failed",
            "harness.task.cancelled",
        }
    )

    # A Gateway restart must not make a completed report disappear.  Replay a
    # bounded tail of durable events; the relay filters model-token events
    # below, so this remains cheap even for a long research trace.
    _REPLAY_EVENT_WINDOW = 20_000

    def __init__(self, *, store: HarnessStore, bus: MessageBus, poll_interval: float = 0.35) -> None:
        self.store = store
        self.bus = bus
        self.poll_interval = poll_interval
        self._cursor = 0
        self._streamed_research_ids: set[str] = set()

    async def run(self) -> None:
        # Let the WebSocket channel finish subscribing before replaying the
        # durable tail; otherwise a fast Gateway boot can publish the catch-up
        # report while there are no browser subscribers yet.
        await asyncio.sleep(max(2.0, self.poll_interval * 4))
        latest = await self.store.latest_audit_event_id()
        self._cursor = max(0, latest - self._REPLAY_EVENT_WINDOW)
        while True:
            events = await self.store.list_audit_events_after(
                self._cursor,
                event_types=self._EVENT_TYPES,
                limit=100,
            )
            if not events:
                await asyncio.sleep(self.poll_interval)
                continue
            for event in events:
                event_id = event.get("event_id")
                if isinstance(event_id, int):
                    self._cursor = max(self._cursor, event_id)
                try:
                    await self._publish(event)
                except Exception:
                    logger.exception("Could not relay repository research event {}", event_id)

    @staticmethod
    def _websocket_chat(session_key: object) -> str | None:
        if not isinstance(session_key, str) or not session_key.startswith("websocket:"):
            return None
        chat_id = session_key.removeprefix("websocket:")
        return chat_id or None

    async def _publish(self, event: dict[str, Any]) -> None:
        chat_id = self._websocket_chat(event.get("session_key"))
        if chat_id is None:
            return
        payload = event.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        payload = payload if isinstance(payload, dict) else {}
        research_id = payload.get("research_id")
        research_id = research_id if isinstance(research_id, str) else "research"
        event_type = event.get("event_type")
        if event_type in {"harness.task.failed", "harness.task.cancelled"}:
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("domain") != "repository_research":
                return
            if event_type == "harness.task.cancelled":
                content = "仓库研究任务已停止，Worker 和隔离分析进程已中断。"
            else:
                error = payload.get("error")
                error_text = str(error).strip() if error else "未知错误"
                if payload.get("state") == "ready":
                    content = f"本轮仓库分析失败：{error_text}；Worker 将自动重试（下一次尝试）。"
                else:
                    content = f"仓库分析失败：{error_text}。已停止自动重试。"
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel="websocket",
                    chat_id=chat_id,
                    content=content,
                    event=ProgressEvent(content=content),
                )
            )
            return
        if event_type == "repository_research.report_delta":
            content = payload.get("content")
            if not isinstance(content, str) or not content:
                return
            self._streamed_research_ids.add(research_id)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel="websocket",
                    chat_id=chat_id,
                    content=content,
                    metadata={"stream_live": True},
                    event=StreamDeltaEvent(
                        content=content,
                        stream_id=f"repository-research:{research_id}",
                    ),
                )
            )
            return
        if event_type == "repository_research.completed":
            report = payload.get("report")
            report = report if isinstance(report, str) and report.strip() else "研究流程已完成，但没有生成报告。"
            stream_id = f"repository-research:{research_id}"
            if research_id in self._streamed_research_ids:
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel="websocket",
                        chat_id=chat_id,
                        content="",
                        metadata={"stream_live": True},
                        event=StreamEndEvent(stream_id=stream_id),
                    )
                )
                self._streamed_research_ids.discard(research_id)
                return
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel="websocket",
                    chat_id=chat_id,
                    content="研究完成，正在流式输出报告…",
                    event=ProgressEvent(content="研究完成，正在流式输出报告…"),
                )
            )
            for start in range(0, len(report), 800):
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel="websocket",
                        chat_id=chat_id,
                        content=report[start : start + 800],
                        event=StreamDeltaEvent(
                            content=report[start : start + 800], stream_id=stream_id
                        ),
                    )
                )
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel="websocket",
                    chat_id=chat_id,
                    content="",
                    event=StreamEndEvent(stream_id=stream_id),
                )
            )
            return
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        detail = payload.get("detail")
        # Repository control agents use the same callback for their private
        # chain-of-thought stream.  It remains durably audited, but must never
        # be rendered as user-facing progress (one token otherwise becomes one
        # WebSocket frame and can starve the final report).
        if isinstance(detail, dict) and (
            detail.get("reasoning") is True
            or detail.get("reasoning_end") is True
            or detail.get("tool_hint") is True
        ):
            return
        if content.startswith("Completed step") or content.startswith("Completed steps:"):
            return
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="websocket",
                chat_id=chat_id,
                content=content,
                event=ProgressEvent(content=content),
            )
        )
