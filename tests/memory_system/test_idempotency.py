from __future__ import annotations

import asyncio

from nanobot.memory_system.record.idempotency import (
    InMemoryIdempotencyStore,
    make_idempotency_key,
)


def test_idempotency_key_sorts_source_ids_and_includes_batch() -> None:
    first = make_idempotency_key(["message-b", "message-a"], "batch-1")
    reordered = make_idempotency_key(["message-a", "message-b"], "batch-1")
    other_batch = make_idempotency_key(["message-a", "message-b"], "batch-2")

    assert first == reordered
    assert first != other_batch
    assert len(first) == 64


async def test_only_one_concurrent_claim_wins() -> None:
    store = InMemoryIdempotencyStore()

    results = await asyncio.gather(
        *(store.claim("same-key", 60) for _ in range(10))
    )

    assert results.count(True) == 1
    assert results.count(False) == 9


async def test_release_allows_retry_but_complete_does_not() -> None:
    store = InMemoryIdempotencyStore()

    assert await store.claim("key", 60)
    await store.release("key")
    assert await store.claim("key", 60)
    await store.complete("key", 60)
    assert not await store.claim("key", 60)
