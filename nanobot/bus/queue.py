"""Async message queue for decoupled channel-agent communication."""

import asyncio
import dataclasses
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from nanobot.bus.events import InboundMessage, OutboundMessage

AuditObserver = Callable[[str, dict[str, Any]], Awaitable[None]]


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._audit_observer: AuditObserver | None = None

    def set_audit_observer(self, observer: AuditObserver | None) -> None:
        """Install a durable observer at the bus boundary.

        The observer is awaited before a message becomes visible to a consumer,
        making the audit record the write-ahead record for channel delivery.
        """
        self._audit_observer = observer

    async def _audit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._audit_observer is not None:
            await self._audit_observer(event_type, payload)

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        msg.metadata.setdefault("trace_id", uuid.uuid4().hex)
        await self._audit(
            "bus.inbound.published",
            {
                "channel": msg.channel,
                "chat_id": msg.chat_id,
                "sender_id": msg.sender_id,
                "session_key": msg.session_key,
                "trace_id": msg.metadata.get("trace_id"),
                "content": msg.content,
                "media": msg.media,
                "metadata": msg.metadata,
            },
        )
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        msg = await self.inbound.get()
        await self._audit(
            "bus.inbound.consumed",
            {
                "channel": msg.channel,
                "chat_id": msg.chat_id,
                "sender_id": msg.sender_id,
                "session_key": msg.session_key,
                "trace_id": msg.metadata.get("trace_id"),
                "content": msg.content,
            },
        )
        return msg

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        event = msg.event
        await self._audit(
            "bus.outbound.published",
            {
                "channel": msg.channel,
                "chat_id": msg.chat_id,
                "trace_id": msg.metadata.get("trace_id"),
                "content": msg.content,
                "media": msg.media,
                "metadata": msg.metadata,
                "event_type": type(event).__name__ if event is not None else None,
                "event": dataclasses.asdict(event) if dataclasses.is_dataclass(event) else None,
            },
        )
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        msg = await self.outbound.get()
        event = msg.event
        await self._audit(
            "bus.outbound.consumed",
            {
                "channel": msg.channel,
                "chat_id": msg.chat_id,
                "trace_id": msg.metadata.get("trace_id"),
                "content": msg.content,
                "event_type": type(event).__name__ if event is not None else None,
            },
        )
        return msg

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
