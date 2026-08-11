"""First-packet semantic retrieval with graph and reranker degradation."""

from __future__ import annotations

import asyncio
import hashlib

from nanobot.memory_system.config import RetrieveConfig
from nanobot.memory_system.models import MemoryRecord, RetrievalResult, SearchResult
from nanobot.memory_system.observability import logger, tracer
from nanobot.memory_system.retrieve.warm_cache import WarmCache
from nanobot.memory_system.storage.embedding import EmbeddingProvider
from nanobot.memory_system.storage.graph_store import GraphStore
from nanobot.memory_system.storage.metadata_store import MetadataStore
from nanobot.memory_system.storage.reranker import Reranker
from nanobot.memory_system.storage.vector_store import VectorStore


class RetrievePipeline:
    def __init__(
        self,
        *,
        config: RetrieveConfig,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        graph_store: GraphStore,
        metadata_store: MetadataStore,
        reranker: Reranker,
        warm_cache: WarmCache,
    ) -> None:
        self.config = config
        self.embedding = embedding
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.metadata_store = metadata_store
        self.reranker = reranker
        self.warm_cache = warm_cache

    async def retrieve(
        self,
        *,
        user_id: str,
        query: str,
        recent_messages: list[str] | None = None,
        deep: bool = False,
    ) -> RetrievalResult:
        context = [*(recent_messages or [])[-self.config.recent_turns :], query]
        query_text = "\n".join(item for item in context if item.strip())
        if not query_text:
            return RetrievalResult()

        with tracer.start_as_current_span("memory.retrieve.pipeline") as span:
            if hasattr(span, "set_attribute"):
                span.set_attribute("memory.user_id", user_id)
                span.set_attribute("memory.deep", deep)
            cache_task = asyncio.create_task(self._get_warm_cache(user_id))
            query_vector = (await self.embedding.embed([query_text]))[0]
            recalled = await self.vector_store.search(
                query_vector,
                self.config.vector.top_k,
                filters={"user_id": user_id},
            )
            recalled = [
                item
                for item in recalled
                if item.score >= self.config.vector.score_threshold
            ]
            cache_records = await cache_task
            graph_records: list[MemoryRecord] = []
            degraded = False
            degradation_reason: str | None = None

            if self.config.graph.enabled and recalled:
                try:
                    graph_records = await asyncio.wait_for(
                        self._graph_expand(user_id, recalled),
                        timeout=self.config.graph.timeout_ms / 1000,
                    )
                except TimeoutError:
                    degraded = True
                    degradation_reason = "graph_timeout"
                    logger.warning("memory_graph_timeout", user_id=user_id)
                except Exception as exc:
                    degraded = True
                    degradation_reason = "graph_error"
                    logger.warning(
                        "memory_graph_error",
                        user_id=user_id,
                        error=str(exc),
                    )

            candidates = self._deduplicate(
                [
                    *(item.record for item in recalled),
                    *graph_records,
                    *cache_records,
                ]
            )
            candidates = await self.metadata_store.filter_active(
                candidates,
                min_confidence=self.config.min_confidence,
            )
            if not candidates:
                return RetrievalResult(
                    degraded=degraded,
                    degradation_reason=degradation_reason,
                )

            try:
                ranked = await asyncio.wait_for(
                    self.reranker.rerank(
                        query_text,
                        [record.fact_text for record in candidates],
                        self.config.reranker.top_n,
                    ),
                    timeout=self.config.reranker.timeout_ms / 1000,
                )
                selected = [
                    SearchResult(
                        record=candidates[item.source_index],
                        score=item.score,
                    )
                    for item in ranked
                    if 0 <= item.source_index < len(candidates)
                ]
            except TimeoutError:
                selected = [
                    SearchResult(record=record, score=float(len(candidates) - index))
                    for index, record in enumerate(
                        candidates[: self.config.reranker.top_n]
                    )
                ]
                degraded = True
                degradation_reason = "reranker_timeout"
                logger.warning("memory_reranker_timeout", user_id=user_id)
            except Exception as exc:
                selected = [
                    SearchResult(record=record, score=float(len(candidates) - index))
                    for index, record in enumerate(
                        candidates[: self.config.reranker.top_n]
                    )
                ]
                degraded = True
                degradation_reason = "reranker_error"
                logger.warning(
                    "memory_reranker_error",
                    user_id=user_id,
                    error=str(exc),
                )
            return RetrievalResult(
                memories=selected,
                degraded=degraded,
                degradation_reason=degradation_reason,
            )

    async def _get_warm_cache(self, user_id: str) -> list[MemoryRecord]:
        if not self.config.cache.warm_cache_enabled:
            return []
        cached = await self.warm_cache.get(user_id)
        if cached:
            return cached
        warm_records = await self.metadata_store.find_warm_records(
            user_id,
            self.config.cache.preload_fields,
        )
        await self.warm_cache.put(
            user_id,
            warm_records,
            self.config.cache.redis_ttl,
        )
        return warm_records

    async def _graph_expand(
        self,
        user_id: str,
        recalled: list[SearchResult],
    ) -> list[MemoryRecord]:
        names = {
            name
            for result in recalled
            for name in result.record.entities
        }
        entity_ids = [
            hashlib.sha256(
                f"{user_id}\x1f{name.casefold()}".encode("utf-8")
            ).hexdigest()
            for name in names
        ]
        subgraphs = await self.graph_store.multi_hop_query(
            entity_ids,
            self.config.graph.max_depth,
        )
        direct = [
            memory
            for subgraph in subgraphs
            for memory in subgraph.supporting_memories
            if memory.user_id == user_id
        ]
        memory_ids = list(
            dict.fromkeys(
                relation.source_memory_id
                for subgraph in subgraphs
                for relation in subgraph.relations
                if relation.source_memory_id
            )
        )
        loaded = await self.metadata_store.get_by_ids(memory_ids)
        return self._deduplicate([*direct, *loaded])

    @staticmethod
    def _deduplicate(records: list[MemoryRecord]) -> list[MemoryRecord]:
        seen: set[str] = set()
        output: list[MemoryRecord] = []
        for record in records:
            if record.id not in seen:
                seen.add(record.id)
                output.append(record)
        return output
