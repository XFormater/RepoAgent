"""VectorStore interface, in-memory backend, and Qdrant adapter."""

# Qdrant is an optional lazy import and its generated models vary across releases.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum
from typing import Any, cast

from nanobot.memory_system.models import MemoryRecord, MemoryStatus, SearchResult


class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, records: list[MemoryRecord]) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def delete_by_idempotency_key(self, key: str) -> None: ...


class InMemoryVectorStore(VectorStore):
    """Concurrency-safe reference backend suitable for tests and local demos."""

    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, records: list[MemoryRecord]) -> None:
        async with self._lock:
            for record in records:
                self.records[record.id] = record.model_copy(deep=True)

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        filters = filters or {}
        async with self._lock:
            candidates = [
                record.model_copy(deep=True)
                for record in self.records.values()
                if _matches(record, filters) and record.status == MemoryStatus.ACTIVE
            ]
        scored = [
            SearchResult(record=record, score=_cosine(query_vector, record.embedding))
            for record in candidates
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    async def delete_by_idempotency_key(self, key: str) -> None:
        async with self._lock:
            self.records = {
                record_id: record
                for record_id, record in self.records.items()
                if record.idempotency_key != key
            }


def _matches(record: MemoryRecord, filters: Mapping[str, Any]) -> bool:
    for key, value in filters.items():
        actual = getattr(record, key, None)
        if isinstance(actual, StrEnum):
            actual = actual.value
        if actual != value:
            return False
    return True


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class QdrantVectorStore(VectorStore):
    """Qdrant gRPC adapter using user_id as a payload partition."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        rest_port: int = 6333,
        api_key: str | None = None,
        https: bool = False,
        collection: str,
        dimension: int,
        prefer_grpc: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 128,
        search_ef: int = 128,
        client: Any | None = None,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant memory requires the 'memory' extra: pip install nanobot-ai[memory]"
            ) from exc
        self._client = client or AsyncQdrantClient(
            host=host,
            port=rest_port,
            grpc_port=port,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
            https=https,
        )
        self._collection = collection
        self._dimension = dimension
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        self._search_ef = search_ef
        self._ready = False
        self._ready_lock = asyncio.Lock()

    async def _ensure_collection(self) -> None:
        if self._ready:
            return
        async with self._ready_lock:
            if self._ready:
                return
            from qdrant_client import models

            if not await self._client.collection_exists(self._collection):
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=self._dimension,
                        distance=models.Distance.COSINE,
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=self._hnsw_m,
                        ef_construct=self._hnsw_ef_construct,
                    ),
                )
            for field_name in ("user_id", "id", "fact_type", "status", "batch_id"):
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            self._ready = True

    async def upsert(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        await self._ensure_collection()
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=record.id,
                vector=record.embedding,
                payload=record.model_dump(mode="json", exclude={"embedding"}),
            )
            for record in records
        ]
        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        await self._ensure_collection()
        from qdrant_client import models

        conditions = [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in (filters or {}).items()
        ]
        if not filters or "status" not in filters:
            conditions.append(
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value="active"),
                )
            )
        query_filter = models.Filter(must=conditions) if conditions else None
        search_params = models.SearchParams(hnsw_ef=self._search_ef)
        legacy_search = cast(
            Callable[..., Awaitable[Any]] | None,
            getattr(self._client, "search", None),
        )
        if callable(legacy_search):
            # qdrant-client <=1.13 exposes ``search`` and supports Qdrant 1.9.
            # Newer clients removed it in favour of the universal query API.
            points = await legacy_search(
                collection_name=self._collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                search_params=search_params,
                with_payload=True,
            )
        else:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                search_params=search_params,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        results: list[SearchResult] = []
        for point in points:
            payload = cast(dict[str, Any], point.payload or {})
            results.append(
                SearchResult(
                    record=MemoryRecord.model_validate({**payload, "embedding": []}),
                    score=float(point.score),
                )
            )
        return results

    async def delete_by_idempotency_key(self, key: str) -> None:
        await self._ensure_collection()
        from qdrant_client import models

        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="idempotency_key",
                            match=models.MatchValue(value=key),
                        )
                    ]
                )
            ),
            wait=True,
        )
