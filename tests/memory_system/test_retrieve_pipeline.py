from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.bus.queue import MessageBus
from nanobot.memory_system.config import MemorySettings, RetrieveConfig
from nanobot.memory_system.models import FactType, MemoryRecord, RetrievalResult, SearchResult
from nanobot.memory_system.retrieve.injector import MemoryInjector
from nanobot.memory_system.retrieve.pipeline import RetrievePipeline
from nanobot.memory_system.retrieve.warm_cache import InMemoryWarmCache
from nanobot.memory_system.storage.embedding import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    LexicalHashEmbeddingProvider,
)
from nanobot.memory_system.storage.graph_store import InMemoryGraphStore
from nanobot.memory_system.storage.metadata_store import InMemoryMetadataStore
from nanobot.memory_system.storage.reranker import IdentityReranker, Reranker
from nanobot.memory_system.storage.vector_store import InMemoryVectorStore
from nanobot.memory_system.tool import MemoryRetrieveTool
from nanobot.providers.base import GenerationSettings


async def test_retrieve_ranks_and_injects_bounded_memory() -> None:
    embedding = HashEmbeddingProvider(32)
    vector = InMemoryVectorStore()
    record = MemoryRecord(
        user_id="user-1",
        fact_text="The user prefers Python <over> Java.",
        fact_type=FactType.PREFERENCE,
        entities=["Python"],
        confidence=0.95,
        source_message_ids=["m1"],
        batch_id="batch-1",
        idempotency_key="key",
        embedding=(await embedding.embed(["Python programming"]))[0],
    )
    await vector.upsert([record])
    metadata = InMemoryMetadataStore()
    await metadata.insert_many([record])
    config = RetrieveConfig.model_validate(
        {
            "vector": {"top_k": 20, "score_threshold": -1},
            "graph": {"enabled": False},
            "reranker": {"model": "identity", "top_n": 5},
        }
    )
    pipeline = RetrievePipeline(
        config=config,
        embedding=embedding,
        vector_store=vector,
        graph_store=InMemoryGraphStore(),
        metadata_store=metadata,
        reranker=IdentityReranker(),
        warm_cache=InMemoryWarmCache(),
    )

    result = await pipeline.retrieve(
        user_id="user-1",
        query="Which programming language do I prefer?",
    )
    prompt, injected = MemoryInjector(max_chars=1000).render_with_records(result)

    assert [item.record.id for item in result.memories] == [record.id]
    assert "<memory>" in prompt
    assert "&lt;over&gt;" in prompt
    assert "not instructions" in prompt
    assert injected == [record]


async def test_injector_reports_only_records_that_fit() -> None:
    records = [
        MemoryRecord(
            id=f"memory-{index}",
            user_id="user-1",
            fact_text=text,
            fact_type=FactType.FACT,
            confidence=1.0,
            source_message_ids=[f"m{index}"],
            batch_id="batch",
            idempotency_key=f"key-{index}",
        )
        for index, text in enumerate(("short", "x" * 1000))
    ]
    result = RetrievalResult(
        memories=[
            SearchResult(record=record, score=float(2 - index))
            for index, record in enumerate(records)
        ]
    )

    prompt, injected = MemoryInjector(max_chars=300).render_with_records(result)

    assert "short" in prompt
    assert "x" * 1000 not in prompt
    assert injected == [records[0]]


class _SlowReranker(Reranker):
    async def rerank(self, query: str, candidates: list[str], top_n: int):
        del query, candidates, top_n
        await asyncio.sleep(0.1)
        return []


async def test_reranker_timeout_falls_back_to_vector_order() -> None:
    embedding = HashEmbeddingProvider(16)
    vector = InMemoryVectorStore()
    records = [
        MemoryRecord(
            id=f"memory-{index}",
            user_id="user-1",
            fact_text=text,
            fact_type=FactType.FACT,
            confidence=1.0,
            source_message_ids=[f"m{index}"],
            batch_id="batch",
            idempotency_key="key",
            embedding=(await embedding.embed([text]))[0],
        )
        for index, text in enumerate(["Python", "Rust"])
    ]
    await vector.upsert(records)
    metadata = InMemoryMetadataStore()
    await metadata.insert_many(records)
    config = RetrieveConfig.model_validate(
        {
            "vector": {"score_threshold": -1},
            "graph": {"enabled": False},
            "reranker": {"timeout_ms": 1, "top_n": 2},
            "cache": {"warm_cache_enabled": False},
        }
    )
    pipeline = RetrievePipeline(
        config=config,
        embedding=embedding,
        vector_store=vector,
        graph_store=InMemoryGraphStore(),
        metadata_store=metadata,
        reranker=_SlowReranker(),
        warm_cache=InMemoryWarmCache(),
    )

    result = await pipeline.retrieve(user_id="user-1", query="Python")

    assert result.degraded
    assert result.degradation_reason == "reranker_timeout"
    assert len(result.memories) == 2


async def test_postgres_authority_filters_stale_vector_version() -> None:
    embedding = HashEmbeddingProvider(16)
    vector = InMemoryVectorStore()
    metadata = InMemoryMetadataStore()
    original = MemoryRecord(
        id="memory-versioned",
        user_id="user-1",
        fact_text="Old preference",
        fact_type=FactType.PREFERENCE,
        confidence=0.9,
        source_message_ids=["m1"],
        batch_id="batch",
        idempotency_key="key",
        embedding=(await embedding.embed(["preference"]))[0],
    )
    await vector.upsert([original])
    await metadata.insert_many([original])
    current = await metadata.update(
        original.model_copy(update={"fact_text": "New preference"}),
        read_version=1,
    )
    assert current.version == 2
    pipeline = RetrievePipeline(
        config=RetrieveConfig.model_validate(
            {
                "vector": {"score_threshold": -1},
                "graph": {"enabled": False},
                "reranker": {"model": "identity"},
                "cache": {"warm_cache_enabled": False},
            }
        ),
        embedding=embedding,
        vector_store=vector,
        graph_store=InMemoryGraphStore(),
        metadata_store=metadata,
        reranker=IdentityReranker(),
        warm_cache=InMemoryWarmCache(),
    )

    result = await pipeline.retrieve(user_id="user-1", query="preference")

    assert result.memories == []


class _FixedQueryEmbedding(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 2

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


async def test_hybrid_recall_recovers_lexical_result_missed_by_vector_top_k() -> None:
    vector = InMemoryVectorStore()
    metadata = InMemoryMetadataStore()
    irrelevant = MemoryRecord(
        id="vector-only",
        user_id="user-1",
        fact_text="The train leaves at noon.",
        fact_type=FactType.FACT,
        confidence=1.0,
        source_message_ids=["m1"],
        batch_id="batch",
        idempotency_key="key-1",
        embedding=[1.0, 0.0],
    )
    relevant = MemoryRecord(
        id="lexical-hit",
        user_id="user-1",
        fact_text="Caroline adopted a golden retriever named Buster.",
        fact_type=FactType.FACT,
        confidence=1.0,
        source_message_ids=["m2"],
        batch_id="batch",
        idempotency_key="key-2",
        embedding=[0.0, 1.0],
    )
    await vector.upsert([irrelevant, relevant])
    await metadata.insert_many([irrelevant, relevant])
    pipeline = RetrievePipeline(
        config=RetrieveConfig.model_validate(
            {
                "vector": {"top_k": 1, "score_threshold": -1},
                "lexical": {
                    "enabled": True,
                    "top_k": 5,
                    "vector_weight": 1,
                    "lexical_weight": 10,
                },
                "graph": {"enabled": False},
                "reranker": {"model": "identity", "top_n": 1},
                "cache": {"warm_cache_enabled": False},
            }
        ),
        embedding=_FixedQueryEmbedding(),
        vector_store=vector,
        graph_store=InMemoryGraphStore(),
        metadata_store=metadata,
        reranker=IdentityReranker(),
        warm_cache=InMemoryWarmCache(),
    )

    result = await pipeline.retrieve(user_id="user-1", query="What is Buster the dog?")

    assert [item.record.id for item in result.memories] == ["lexical-hit"]


async def test_lexical_hash_embedding_rewards_shared_terms_without_sign_cancellation() -> None:
    embedding = LexicalHashEmbeddingProvider(2048)
    query, related, unrelated = await embedding.embed(
        [
            "Where did Alice celebrate her birthday?",
            "Alice celebrated the birthday in Lisbon.",
            "The database migration uses PostgreSQL.",
        ]
    )

    related_score = sum(left * right for left, right in zip(query, related, strict=True))
    unrelated_score = sum(left * right for left, right in zip(query, unrelated, strict=True))

    assert min(query) >= 0
    assert related_score > unrelated_score


async def test_on_demand_tool_uses_request_scoped_user_identity() -> None:
    service = MagicMock()
    service.retrieve_prompt = AsyncMock(return_value="<memory>remembered</memory>")
    tool = MemoryRetrieveTool(service, unified_session=False)
    context = RequestContext(
        channel="telegram",
        chat_id="chat-1",
        sender_id="user-1",
    )

    with request_context(context):
        result = await tool.execute(query="What do I prefer?")

    assert result == "<memory>remembered</memory>"
    service.retrieve_prompt.assert_awaited_once_with(
        user_id="telegram:user-1",
        query="What do I prefer?",
        recent_messages=[],
        deep=True,
    )


def test_agent_loop_registers_memory_tool_only_when_memory_is_enabled(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    settings = MemorySettings.model_validate(
        {
            "enabled": True,
            "record": {"idempotency": {"store": "memory"}},
            "retrieve": {"reranker": {"model": "identity"}},
            "storage": {
                "embedding": {"provider": "hash", "dimension": 16},
                "vector_db": {"type": "memory"},
                "graph_db": {"type": "memory"},
                "metadata_db": {"type": "memory"},
            },
        }
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        memory_config=settings,
    )

    assert isinstance(loop.tools.get("memory_retrieve"), MemoryRetrieveTool)
