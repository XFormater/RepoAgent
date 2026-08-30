"""Batch idempotency keys and stores."""

# Optional Redis/asyncpg clients are imported lazily and have version-dependent types.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any


def make_idempotency_key(source_message_ids: list[str], batch_id: str) -> str:
    """Hash sorted source IDs and the scheduler-owned batch UUID.

    Retries must preserve ``batch_id``.  A newly triggered batch deliberately
    receives a new key, even if it references the same source messages.
    """
    canonical = "\x1f".join([*sorted(set(source_message_ids)), batch_id])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IdempotencyStore(ABC):
    @abstractmethod
    async def claim(self, key: str, ttl_seconds: int) -> bool: ...

    @abstractmethod
    async def complete(self, key: str, ttl_seconds: int) -> None: ...

    @abstractmethod
    async def release(self, key: str) -> None: ...


class InMemoryIdempotencyStore(IdempotencyStore):
    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def claim(self, key: str, ttl_seconds: int) -> bool:
        now = time.monotonic()
        async with self._lock:
            self._entries = {
                item_key: value
                for item_key, value in self._entries.items()
                if value[1] > now
            }
            if key in self._entries:
                return False
            self._entries[key] = ("processing", now + ttl_seconds)
            return True

    async def complete(self, key: str, ttl_seconds: int) -> None:
        async with self._lock:
            self._entries[key] = ("completed", time.monotonic() + ttl_seconds)

    async def release(self, key: str) -> None:
        async with self._lock:
            self._entries.pop(key, None)


class RedisIdempotencyStore(IdempotencyStore):
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
        self._prefix = "nanobot:memory:idempotency:"

    async def claim(self, key: str, ttl_seconds: int) -> bool:
        result = await self._client.set(
            self._prefix + key,
            "processing",
            ex=ttl_seconds,
            nx=True,
        )
        return bool(result)

    async def complete(self, key: str, ttl_seconds: int) -> None:
        await self._client.set(self._prefix + key, "completed", ex=ttl_seconds)

    async def release(self, key: str) -> None:
        await self._client.delete(self._prefix + key)


class PostgresIdempotencyStore(IdempotencyStore):
    def __init__(self, dsn: str, *, pool: Any | None = None) -> None:
        self._dsn = dsn
        self._pool = pool
        self._ready = False
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Any:
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL memory requires the 'memory' extra: "
                    "pip install nanobot-ai[memory]"
                ) from exc
            self._pool = await asyncpg.create_pool(self._dsn)
        if not self._ready:
            async with self._lock:
                if not self._ready:
                    async with self._pool.acquire() as connection:
                        await connection.execute(
                            """
                            CREATE TABLE IF NOT EXISTS agent_memory_idempotency (
                                key TEXT PRIMARY KEY,
                                state TEXT NOT NULL,
                                expires_at TIMESTAMPTZ NOT NULL
                            )
                            """
                        )
                    self._ready = True
        return self._pool

    async def claim(self, key: str, ttl_seconds: int) -> bool:
        pool = await self._ensure()
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "DELETE FROM agent_memory_idempotency WHERE expires_at <= now()"
            )
            result = await connection.execute(
                """
                INSERT INTO agent_memory_idempotency(key, state, expires_at)
                VALUES ($1, 'processing', now() + ($2 * interval '1 second'))
                ON CONFLICT DO NOTHING
                """,
                key,
                ttl_seconds,
            )
        return result.endswith("1")

    async def complete(self, key: str, ttl_seconds: int) -> None:
        pool = await self._ensure()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                UPDATE agent_memory_idempotency
                   SET state='completed',
                       expires_at=now() + ($2 * interval '1 second')
                 WHERE key=$1
                """,
                key,
                ttl_seconds,
            )

    async def release(self, key: str) -> None:
        pool = await self._ensure()
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM agent_memory_idempotency WHERE key=$1 AND state='processing'",
                key,
            )
