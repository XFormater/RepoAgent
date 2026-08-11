from __future__ import annotations

import pytest

from nanobot.memory_system.config import RecordConfig
from nanobot.memory_system.models import (
    ExtractedFact,
    FactType,
    MemoryRecord,
    MemoryStatus,
)
from nanobot.memory_system.record.conflict import LatestWinsResolver
from nanobot.memory_system.record.extractor import LLMMemoryExtractor, StaticMemoryExtractor
from nanobot.memory_system.record.idempotency import InMemoryIdempotencyStore
from nanobot.memory_system.record.pipeline import RecordPipeline
from nanobot.memory_system.storage.embedding import HashEmbeddingProvider
from nanobot.memory_system.storage.graph_store import InMemoryGraphStore
from nanobot.memory_system.storage.metadata_store import (
    InMemoryMetadataStore,
    OptimisticLockError,
)
from nanobot.memory_system.storage.vector_store import InMemoryVectorStore


def _pipeline(facts: list[ExtractedFact]) -> tuple[
    RecordPipeline,
    InMemoryVectorStore,
    InMemoryGraphStore,
    InMemoryMetadataStore,
]:
    vector = InMemoryVectorStore()
    graph = InMemoryGraphStore()
    metadata = InMemoryMetadataStore()
    pipeline = RecordPipeline(
        config=RecordConfig(max_retries=0),
        extractor=StaticMemoryExtractor(facts),
        embedding=HashEmbeddingProvider(32),
        vector_store=vector,
        graph_store=graph,
        metadata_store=metadata,
        idempotency_store=InMemoryIdempotencyStore(),
        conflict_resolver=LatestWinsResolver(),
    )
    return pipeline, vector, graph, metadata


async def test_record_writes_active_facts_but_holds_hypotheses() -> None:
    facts = [
        ExtractedFact(
            fact_text="The user prefers dark mode.",
            fact_type=FactType.PREFERENCE,
            entities=["dark mode"],
            confidence=0.95,
            source_message_ids=["m1"],
        ),
        ExtractedFact(
            fact_text="The user may move to Berlin.",
            fact_type=FactType.HYPOTHESIS,
            entities=["Berlin"],
            confidence=0.8,
            source_message_ids=["m1"],
        ),
    ]
    pipeline, vector, graph, metadata = _pipeline(facts)

    result = await pipeline.record(
        user_id="user-1",
        messages=[{"id": "m1", "role": "user", "content": "I may move."}],
        batch_id="batch-1",
    )

    assert result.inserted == 1
    assert result.pending_review == 1
    assert len(vector.records) == 1
    assert len(metadata.records) == 2
    assert {record.status for record in metadata.records.values()} == {
        MemoryStatus.ACTIVE,
        MemoryStatus.PENDING_REVIEW,
    }
    assert graph.entities


async def test_record_batch_is_idempotent() -> None:
    fact = ExtractedFact(
        fact_text="The user uses Python.",
        fact_type=FactType.FACT,
        entities=["Python"],
        confidence=1.0,
        source_message_ids=["m1"],
    )
    pipeline, vector, _, metadata = _pipeline([fact])
    kwargs = {
        "user_id": "user-1",
        "messages": [{"id": "m1", "role": "user", "content": "I use Python."}],
        "batch_id": "batch-1",
    }

    first = await pipeline.record(**kwargs)
    second = await pipeline.record(**kwargs)

    assert first.inserted == 1
    assert second.skipped
    assert len(vector.records) == 1
    assert len(metadata.records) == 1


async def test_optimistic_lock_rejects_stale_update() -> None:
    metadata = InMemoryMetadataStore()
    record = MemoryRecord(
        id="memory-1",
        user_id="user-1",
        fact_text="Initial fact",
        fact_type=FactType.FACT,
        confidence=1.0,
        source_message_ids=["m1"],
        batch_id="batch-1",
        idempotency_key="key",
    )
    await metadata.insert_many([record])
    first_reader = await metadata.get(record.id)
    stale_reader = await metadata.get(record.id)
    assert first_reader is not None
    assert stale_reader is not None

    updated = await metadata.update(
        first_reader.model_copy(update={"fact_text": "Updated fact"}),
        read_version=first_reader.version,
    )

    assert updated.version == 2
    with pytest.raises(OptimisticLockError):
        await metadata.update(
            stale_reader.model_copy(update={"fact_text": "Stale update"}),
            read_version=stale_reader.version,
        )


async def test_latest_preference_supersedes_old_record() -> None:
    old_fact = ExtractedFact(
        fact_text="The user prefers light mode.",
        fact_type=FactType.PREFERENCE,
        entities=["display theme"],
        confidence=0.9,
        source_message_ids=["m1"],
    )
    pipeline, vector, _, metadata = _pipeline([old_fact])
    await pipeline.record(
        user_id="user-1",
        messages=[{"id": "m1", "role": "user", "content": "Light mode."}],
        batch_id="batch-1",
    )
    pipeline.extractor = StaticMemoryExtractor(
        [
            ExtractedFact(
                fact_text="The user prefers dark mode.",
                fact_type=FactType.PREFERENCE,
                entities=["display theme"],
                confidence=0.99,
                source_message_ids=["m2"],
            )
        ]
    )

    result = await pipeline.record(
        user_id="user-1",
        messages=[{"id": "m2", "role": "user", "content": "Dark mode now."}],
        batch_id="batch-2",
    )

    assert result.inserted == 1
    statuses = {record.fact_text: record.status for record in metadata.records.values()}
    assert statuses["The user prefers light mode."] == MemoryStatus.SUPERSEDED
    assert statuses["The user prefers dark mode."] == MemoryStatus.ACTIVE
    active_vectors = [
        record for record in vector.records.values() if record.status == MemoryStatus.ACTIVE
    ]
    assert [record.fact_text for record in active_vectors] == [
        "The user prefers dark mode."
    ]


async def test_extractor_rejects_fact_citing_only_assistant_text() -> None:
    async def invoke(_messages):
        return """[{
          "fact_text": "The user lives on Mars.",
          "fact_type": "fact",
          "entities": ["Mars"],
          "confidence": 0.99,
          "source_message_ids": ["assistant-1"]
        }]"""

    extractor = LLMMemoryExtractor(invoke)

    facts = await extractor.extract(
        [
            {"id": "user-1", "role": "user", "content": "Hello"},
            {
                "id": "assistant-1",
                "role": "assistant",
                "content": "You live on Mars.",
            },
        ]
    )

    assert facts == []


class _FailingGraphStore(InMemoryGraphStore):
    def __init__(self) -> None:
        super().__init__()
        self.failures = 1

    async def upsert_entity(self, entity, *, read_version=None):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary graph failure")
        await super().upsert_entity(entity, read_version=read_version)


class _AlwaysFailingGraphStore(InMemoryGraphStore):
    async def upsert_entity(self, entity, *, read_version=None):
        del entity, read_version
        raise RuntimeError("graph unavailable")


async def test_partial_write_retry_resumes_with_stable_record() -> None:
    fact = ExtractedFact(
        fact_text="The user uses Python.",
        fact_type=FactType.FACT,
        entities=["Python"],
        confidence=1.0,
        source_message_ids=["m1"],
    )
    vector = InMemoryVectorStore()
    graph = _FailingGraphStore()
    metadata = InMemoryMetadataStore()
    pipeline = RecordPipeline(
        config=RecordConfig(max_retries=1),
        extractor=StaticMemoryExtractor([fact]),
        embedding=HashEmbeddingProvider(16),
        vector_store=vector,
        graph_store=graph,
        metadata_store=metadata,
        idempotency_store=InMemoryIdempotencyStore(),
        conflict_resolver=LatestWinsResolver(),
    )
    pipeline._retry_delay = lambda _attempt: 0

    result = await pipeline.record(
        user_id="user-1",
        messages=[{"id": "m1", "role": "user", "content": "I use Python."}],
        batch_id="stable-batch",
    )

    assert result.inserted == 1
    assert len(metadata.records) == 1
    assert len(vector.records) == 1
    assert len(graph.entities) == 2
    assert {entity.entity_type for entity in graph.entities.values()} == {
        "User",
        "Technology",
    }


async def test_best_effort_keeps_metadata_when_graph_is_unavailable() -> None:
    fact = ExtractedFact(
        fact_text="The user works on Project Atlas with Python.",
        fact_type=FactType.FACT,
        entities=["Project Atlas", "Python"],
        confidence=0.95,
        source_message_ids=["m1"],
    )
    vector = InMemoryVectorStore()
    metadata = InMemoryMetadataStore()
    pipeline = RecordPipeline(
        config=RecordConfig(max_retries=0, mode="best_effort"),
        extractor=StaticMemoryExtractor([fact]),
        embedding=HashEmbeddingProvider(16),
        vector_store=vector,
        graph_store=_AlwaysFailingGraphStore(),
        metadata_store=metadata,
        idempotency_store=InMemoryIdempotencyStore(),
        conflict_resolver=LatestWinsResolver(),
    )

    result = await pipeline.record(
        user_id="user-1",
        messages=[{"id": "m1", "role": "user", "content": "Atlas uses Python"}],
        batch_id="batch-degraded",
    )

    assert result.inserted == 1
    assert result.error is None
    assert result.failed_backends == ("neo4j",)
    assert len(metadata.records) == 1
    assert len(vector.records) == 1
