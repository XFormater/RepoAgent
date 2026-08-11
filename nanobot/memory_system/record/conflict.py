"""Preference conflict strategies and optimistic-lock handling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum

from nanobot.memory_system.models import MemoryRecord, MemoryStatus
from nanobot.memory_system.storage.metadata_store import MetadataStore


class ConflictResolutionStrategy(StrEnum):
    LATEST_WINS = "latest_wins"
    MERGE_BY_LLM = "merge_by_llm"
    MANUAL_REVIEW = "manual_review"


MergeCallable = Callable[[MemoryRecord, MemoryRecord], Awaitable[str]]


class ConflictResolver(ABC):
    @abstractmethod
    async def resolve(
        self,
        existing: MemoryRecord,
        incoming: MemoryRecord,
        metadata: MetadataStore,
    ) -> MemoryRecord: ...


class LatestWinsResolver(ConflictResolver):
    async def resolve(
        self,
        existing: MemoryRecord,
        incoming: MemoryRecord,
        metadata: MetadataStore,
    ) -> MemoryRecord:
        await metadata.update(
            existing.model_copy(update={"status": MemoryStatus.SUPERSEDED}),
            read_version=existing.version,
        )
        return incoming


class LLMMergeResolver(ConflictResolver):
    def __init__(self, merge: MergeCallable) -> None:
        self._merge = merge

    async def resolve(
        self,
        existing: MemoryRecord,
        incoming: MemoryRecord,
        metadata: MetadataStore,
    ) -> MemoryRecord:
        merged_text = await self._merge(existing, incoming)
        await metadata.update(
            existing.model_copy(update={"status": MemoryStatus.SUPERSEDED}),
            read_version=existing.version,
        )
        return incoming.model_copy(
            update={
                "fact_text": merged_text,
                "source_message_ids": list(
                    dict.fromkeys(
                        [*existing.source_message_ids, *incoming.source_message_ids]
                    )
                ),
            }
        )


class ManualReviewResolver(ConflictResolver):
    async def resolve(
        self,
        existing: MemoryRecord,
        incoming: MemoryRecord,
        metadata: MetadataStore,
    ) -> MemoryRecord:
        del existing, metadata
        return incoming.model_copy(update={"status": MemoryStatus.PENDING_REVIEW})


def build_conflict_resolver(
    strategy: ConflictResolutionStrategy,
    *,
    merge: MergeCallable | None = None,
) -> ConflictResolver:
    if strategy == ConflictResolutionStrategy.LATEST_WINS:
        return LatestWinsResolver()
    if strategy == ConflictResolutionStrategy.MANUAL_REVIEW:
        return ManualReviewResolver()
    if merge is None:
        raise ValueError("MERGE_BY_LLM requires a merge callback")
    return LLMMergeResolver(merge)
