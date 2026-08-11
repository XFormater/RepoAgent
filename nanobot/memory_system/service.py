"""Facade and dependency factory for nanobot's semantic memory pipelines."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nanobot.memory_system.config import MemorySettings
from nanobot.memory_system.file_sync import FileSyncOutcome, MemoryFileSynchronizer
from nanobot.memory_system.models import MemoryRecord
from nanobot.memory_system.observability import logger
from nanobot.memory_system.record.conflict import (
    ConflictResolutionStrategy,
    build_conflict_resolver,
)
from nanobot.memory_system.record.extractor import LLMMemoryExtractor
from nanobot.memory_system.record.idempotency import (
    InMemoryIdempotencyStore,
    PostgresIdempotencyStore,
    RedisIdempotencyStore,
)
from nanobot.memory_system.record.pipeline import RecordOutcome, RecordPipeline
from nanobot.memory_system.retrieve.injector import MemoryInjector
from nanobot.memory_system.retrieve.pipeline import RetrievePipeline
from nanobot.memory_system.retrieve.warm_cache import InMemoryWarmCache, RedisWarmCache
from nanobot.memory_system.short_term.redis_memory import RedisShortTermMemory
from nanobot.memory_system.storage.embedding import (
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from nanobot.memory_system.storage.graph_store import InMemoryGraphStore, Neo4jGraphStore
from nanobot.memory_system.storage.metadata_store import (
    InMemoryMetadataStore,
    PostgresMetadataStore,
)
from nanobot.memory_system.storage.reranker import CrossEncoderReranker, IdentityReranker
from nanobot.memory_system.storage.vector_store import InMemoryVectorStore, QdrantVectorStore

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class LongTermMemoryService:
    def __init__(
        self,
        *,
        settings: MemorySettings,
        record_pipeline: RecordPipeline,
        retrieve_pipeline: RetrievePipeline,
        injector: MemoryInjector,
        short_term: RedisShortTermMemory | None = None,
    ) -> None:
        self.settings = settings
        self.record_pipeline = record_pipeline
        self.retrieve_pipeline = retrieve_pipeline
        self.injector = injector
        self.short_term = short_term
        self.file_synchronizer = MemoryFileSynchronizer(
            record_pipeline.metadata_store,
            settings.file_sync,
        )

    async def sync_core_memory(self, store: Any) -> FileSyncOutcome:
        """Refresh the managed MEMORY.md projection without touching manual content."""
        return await self.file_synchronizer.sync(store)

    def should_retrieve(self, *, first_turn: bool, deep: bool = False) -> bool:
        if not self.settings.retrieve.enabled:
            return False
        trigger = self.settings.retrieve.trigger
        return deep or trigger == "every_turn" or (trigger == "first_turn" and first_turn)

    async def retrieve_prompt(
        self,
        *,
        user_id: str,
        query: str,
        recent_messages: list[str],
        deep: bool = False,
    ) -> str:
        try:
            result = await self.retrieve_pipeline.retrieve(
                user_id=user_id,
                query=query,
                recent_messages=recent_messages,
                deep=deep,
            )
            return self.injector.render(result)
        except Exception as exc:
            logger.warning(
                "memory_retrieve_failed",
                user_id=user_id,
                error=str(exc),
            )
            return ""

    async def record_turn(
        self,
        *,
        user_id: str,
        messages: list[dict[str, str]],
        batch_id: str,
    ) -> RecordOutcome:
        outcome = await self.record_pipeline.record(
            user_id=user_id,
            messages=messages,
            batch_id=batch_id,
        )
        if outcome.error:
            logger.warning(
                "memory_record_failed",
                user_id=user_id,
                batch_id=batch_id,
                error=outcome.error,
            )
        elif outcome.inserted:
            await self.retrieve_pipeline.warm_cache.invalidate(user_id)
            if outcome.failed_backends:
                logger.warning(
                    "memory_record_projection_degraded",
                    user_id=user_id,
                    batch_id=batch_id,
                    failed_backends=outcome.failed_backends,
                )
        return outcome

    async def close(self) -> None:
        """Close optional backend clients without assuming one client API."""
        seen: set[int] = set()
        objects = (
            self.record_pipeline.vector_store,
            self.record_pipeline.graph_store,
            self.record_pipeline.metadata_store,
            self.record_pipeline.idempotency_store,
            self.retrieve_pipeline.warm_cache,
            self.short_term,
        )
        for value in objects:
            client = (
                getattr(value, "_client", None)
                or getattr(value, "_driver", None)
                or getattr(value, "_pool", None)
            )
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            closer = getattr(client, "aclose", None) or getattr(client, "close", None)
            if closer is None:
                continue
            result = closer()
            if hasattr(result, "__await__"):
                await result


def build_memory_service(
    settings: MemorySettings,
    *,
    provider: LLMProvider,
    default_model: str,
) -> LongTermMemoryService:
    """Build production adapters lazily; network connections occur on first use."""
    storage = settings.storage
    embedding = (
        HashEmbeddingProvider(storage.embedding.dimension)
        if storage.embedding.provider == "hash"
        else OpenAIEmbeddingProvider(
            model=storage.embedding.model,
            dimension=storage.embedding.dimension,
            api_key=storage.embedding.api_key,
            base_url=storage.embedding.base_url,
        )
    )
    vector_store = (
        InMemoryVectorStore()
        if storage.vector_db.type == "memory"
        else QdrantVectorStore(
            host=storage.vector_db.host,
            port=storage.vector_db.port,
            rest_port=storage.vector_db.rest_port,
            api_key=storage.vector_db.api_key,
            https=storage.vector_db.https,
            collection=storage.vector_db.collection_prefix,
            dimension=embedding.dimension,
            prefer_grpc=storage.vector_db.prefer_grpc,
            hnsw_m=storage.vector_db.hnsw_m,
            hnsw_ef_construct=storage.vector_db.hnsw_ef_construct,
            search_ef=settings.retrieve.vector.hnsw_ef,
        )
    )
    graph_store = (
        InMemoryGraphStore()
        if storage.graph_db.type == "memory"
        else Neo4jGraphStore(
            uri=storage.graph_db.uri,
            username=storage.graph_db.username,
            password=storage.graph_db.password,
            database=storage.graph_db.database,
            initialize_schema=storage.graph_db.initialize_schema,
        )
    )
    metadata_store = (
        InMemoryMetadataStore()
        if storage.metadata_db.type == "memory"
        else PostgresMetadataStore(
            storage.metadata_db.dsn,
            min_pool_size=storage.metadata_db.min_pool_size,
            max_pool_size=storage.metadata_db.max_pool_size,
        )
    )
    if settings.record.idempotency.store == "memory":
        idempotency_store = InMemoryIdempotencyStore()
    elif settings.record.idempotency.store == "postgres":
        idempotency_store = PostgresIdempotencyStore(storage.metadata_db.dsn)
    else:
        idempotency_store = RedisIdempotencyStore(
            storage.redis.url,
            socket_timeout=storage.redis.socket_timeout,
        )
    warm_cache = (
        InMemoryWarmCache()
        if storage.vector_db.type == "memory"
        else RedisWarmCache(
            storage.redis.url,
            socket_timeout=storage.redis.socket_timeout,
        )
    )
    reranker = (
        IdentityReranker()
        if settings.retrieve.reranker.model == "identity"
        else CrossEncoderReranker(
            settings.retrieve.reranker.model,
            max_batch_size=settings.retrieve.reranker.max_batch_size,
        )
    )
    extraction_model = settings.record.llm.model or default_model

    async def invoke_extractor(messages: list[dict[str, str]]) -> str:
        response = await provider.chat_with_retry(
            messages=messages,
            tools=None,
            model=extraction_model,
            max_tokens=settings.record.llm.max_tokens,
            temperature=settings.record.llm.temperature,
            retry_mode="standard",
        )
        if response.finish_reason == "error":
            raise RuntimeError(response.content or "memory extraction LLM failed")
        return response.content or "[]"

    async def merge_preferences(existing: MemoryRecord, incoming: MemoryRecord) -> str:
        response = await provider.chat_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Merge two user-preference memories. Return only one concise "
                        "fact. Preserve compatible details; on contradiction prefer "
                        "the newer second fact."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "existing": existing.fact_text,
                            "incoming": incoming.fact_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools=None,
            model=extraction_model,
            max_tokens=512,
            temperature=0.0,
            retry_mode="standard",
        )
        if response.finish_reason == "error" or not response.content:
            raise RuntimeError(response.content or "preference merge failed")
        return response.content.strip()

    conflict_resolver = build_conflict_resolver(
        ConflictResolutionStrategy(settings.record.conflict_strategy),
        merge=merge_preferences,
    )
    record_pipeline = RecordPipeline(
        config=settings.record,
        extractor=LLMMemoryExtractor(invoke_extractor),
        embedding=embedding,
        vector_store=vector_store,
        graph_store=graph_store,
        metadata_store=metadata_store,
        idempotency_store=idempotency_store,
        conflict_resolver=conflict_resolver,
    )
    retrieve_pipeline = RetrievePipeline(
        config=settings.retrieve,
        embedding=embedding,
        vector_store=vector_store,
        graph_store=graph_store,
        metadata_store=metadata_store,
        reranker=reranker,
        warm_cache=warm_cache,
    )
    return LongTermMemoryService(
        settings=settings,
        record_pipeline=record_pipeline,
        retrieve_pipeline=retrieve_pipeline,
        injector=MemoryInjector(settings.retrieve.max_injected_chars),
        short_term=(
            RedisShortTermMemory(
                storage.redis.url,
                settings.short_term,
                socket_timeout=storage.redis.socket_timeout,
            )
            if settings.short_term.enabled
            else None
        ),
    )
