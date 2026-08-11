"""Warm cache for high-frequency user preferences and static profile facts."""

# Optional Redis is imported lazily and its async client typing varies by version.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from nanobot.memory_system.models import MemoryRecord


class WarmCache(ABC):
    @abstractmethod
    async def get(self, user_id: str) -> list[MemoryRecord]: ...

    @abstractmethod
    async def put(
        self,
        user_id: str,
        records: list[MemoryRecord],
        ttl_seconds: int,
    ) -> None: ...

    @abstractmethod
    async def invalidate(self, user_id: str) -> None: ...


class InMemoryWarmCache(WarmCache):
    def __init__(self) -> None:
        self._records: dict[str, tuple[list[MemoryRecord], float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: str) -> list[MemoryRecord]:
        async with self._lock:
            entry = self._records.get(user_id)
            if entry is None:
                return []
            records, expires_at = entry
            if expires_at <= time.monotonic():
                self._records.pop(user_id, None)
                return []
            return [
                record.model_copy(deep=True)
                for record in records
            ]

    async def put(
        self,
        user_id: str,
        records: list[MemoryRecord],
        ttl_seconds: int,
    ) -> None:
        async with self._lock:
            self._records[user_id] = (
                [record.model_copy(deep=True) for record in records],
                time.monotonic() + ttl_seconds,
            )

    async def invalidate(self, user_id: str) -> None:
        async with self._lock:
            self._records.pop(user_id, None)


class RedisWarmCache(WarmCache):
    def __init__(
        self,
        url: str,
        *,
        socket_timeout: float = 2.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from redis.asyncio import Redis
            except ImportError as exc:
                raise RuntimeError(
                    "Redis memory requires the 'memory' extra: pip install nanobot-ai[memory]"
                ) from exc
            client = Redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=socket_timeout,
            )
        self._client = client
        self._prefix = "nanobot:memory:warm:"

    async def get(self, user_id: str) -> list[MemoryRecord]:
        raw = await self._client.get(self._prefix + user_id)
        if not raw:
            return []
        try:
            values = json.loads(raw)
            return [MemoryRecord.model_validate(value) for value in values]
        except (ValueError, TypeError):
            await self.invalidate(user_id)
            return []

    async def put(
        self,
        user_id: str,
        records: list[MemoryRecord],
        ttl_seconds: int,
    ) -> None:
        payload = json.dumps(
            [record.model_dump(mode="json") for record in records],
            ensure_ascii=False,
        )
        await self._client.set(
            self._prefix + user_id,
            payload,
            ex=ttl_seconds,
        )

    async def invalidate(self, user_id: str) -> None:
        await self._client.delete(self._prefix + user_id)
