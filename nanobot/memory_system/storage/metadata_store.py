"""Relational metadata store with optimistic locking."""

# asyncpg is optional and imported only when the PostgreSQL backend is selected.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from nanobot.memory_system.models import MemoryRecord, MemoryStatus, SearchResult
from nanobot.memory_system.storage.lexical import bm25_scores


class OptimisticLockError(RuntimeError):
    def __init__(self, record_id: str, expected: int, actual: int) -> None:
        super().__init__(
            f"optimistic lock failed for {record_id}: expected version {expected}, got {actual}"
        )
        self.record_id = record_id
        self.expected = expected
        self.actual = actual


class MetadataStore(ABC):
    @abstractmethod
    async def insert_many(self, records: list[MemoryRecord]) -> None: ...

    @abstractmethod
    async def get(self, record_id: str) -> MemoryRecord | None: ...

    @abstractmethod
    async def update(
        self,
        record: MemoryRecord,
        *,
        read_version: int,
    ) -> MemoryRecord: ...

    @abstractmethod
    async def find_preferences(
        self,
        user_id: str,
        entities: list[str],
    ) -> list[MemoryRecord]: ...

    @abstractmethod
    async def find_warm_records(
        self,
        user_id: str,
        fields: list[str],
    ) -> list[MemoryRecord]: ...

    @abstractmethod
    async def get_by_ids(self, record_ids: list[str]) -> list[MemoryRecord]: ...

    @abstractmethod
    async def search_text(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int,
        min_confidence: float,
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def filter_active(
        self,
        records: list[MemoryRecord],
        *,
        min_confidence: float,
    ) -> list[MemoryRecord]: ...

    @abstractmethod
    async def list_active_for_file_sync(
        self,
        *,
        min_confidence: float,
        limit: int,
    ) -> list[MemoryRecord]: ...


class InMemoryMetadataStore(MetadataStore):
    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}
        self._lock = asyncio.Lock()

    async def insert_many(self, records: list[MemoryRecord]) -> None:
        async with self._lock:
            for record in records:
                self.records.setdefault(record.id, record.model_copy(deep=True))

    async def get(self, record_id: str) -> MemoryRecord | None:
        async with self._lock:
            record = self.records.get(record_id)
            return record.model_copy(deep=True) if record else None

    async def update(
        self,
        record: MemoryRecord,
        *,
        read_version: int,
    ) -> MemoryRecord:
        async with self._lock:
            current = self.records.get(record.id)
            actual = current.version if current else 0
            if current is None or actual != read_version:
                raise OptimisticLockError(record.id, read_version, actual)
            updated = record.model_copy(
                update={
                    "version": read_version + 1,
                    "updated_at": datetime.now(UTC),
                },
                deep=True,
            )
            self.records[record.id] = updated
            return updated.model_copy(deep=True)

    async def find_preferences(
        self,
        user_id: str,
        entities: list[str],
    ) -> list[MemoryRecord]:
        wanted = {entity.casefold() for entity in entities}
        async with self._lock:
            return [
                record.model_copy(deep=True)
                for record in self.records.values()
                if record.user_id == user_id
                and record.fact_type.value == "preference"
                and record.status == MemoryStatus.ACTIVE
                and (
                    not wanted
                    or wanted.intersection(entity.casefold() for entity in record.entities)
                )
            ]

    async def get_by_ids(self, record_ids: list[str]) -> list[MemoryRecord]:
        async with self._lock:
            return [
                self.records[record_id].model_copy(deep=True)
                for record_id in record_ids
                if record_id in self.records
            ]

    async def search_text(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int,
        min_confidence: float,
    ) -> list[SearchResult]:
        async with self._lock:
            records = [
                record.model_copy(deep=True)
                for record in self.records.values()
                if record.user_id == user_id
                and record.status == MemoryStatus.ACTIVE
                and record.confidence >= min_confidence
            ]
        scores = bm25_scores(query, [record.fact_text for record in records])
        ranked = sorted(
            (
                SearchResult(record=record, score=score)
                for record, score in zip(records, scores, strict=True)
                if score > 0
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:top_k]

    async def find_warm_records(
        self,
        user_id: str,
        fields: list[str],
    ) -> list[MemoryRecord]:
        fact_types = _warm_fact_types(fields)
        async with self._lock:
            return [
                record.model_copy(deep=True)
                for record in self.records.values()
                if record.user_id == user_id
                and record.fact_type.value in fact_types
                and record.status == MemoryStatus.ACTIVE
            ]

    async def filter_active(
        self,
        records: list[MemoryRecord],
        *,
        min_confidence: float,
    ) -> list[MemoryRecord]:
        async with self._lock:
            output: list[MemoryRecord] = []
            for candidate in records:
                current = self.records.get(candidate.id)
                if (
                    current is not None
                    and current.status == MemoryStatus.ACTIVE
                    and current.confidence >= min_confidence
                    and current.version == candidate.version
                ):
                    output.append(current.model_copy(deep=True))
            return output

    async def list_active_for_file_sync(
        self,
        *,
        min_confidence: float,
        limit: int,
    ) -> list[MemoryRecord]:
        async with self._lock:
            records = [
                record.model_copy(deep=True)
                for record in self.records.values()
                if record.status == MemoryStatus.ACTIVE
                and record.confidence >= min_confidence
            ]
        records.sort(key=lambda item: (item.confidence, item.updated_at), reverse=True)
        return records[:limit]


class PostgresMetadataStore(MetadataStore):
    """asyncpg metadata adapter.

    ``initialize`` is idempotent and runs lazily before the first operation.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        pool: Any | None = None,
    ) -> None:
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool = pool
        self._ready = False
        self._lock = asyncio.Lock()

    async def _ensure_pool(self) -> Any:
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL memory requires the 'memory' extra: "
                    "pip install nanobot-ai[memory]"
                ) from exc
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
            )
        if not self._ready:
            async with self._lock:
                if not self._ready:
                    async with self._pool.acquire() as connection:
                        await connection.execute(_SCHEMA)
                    self._ready = True
        return self._pool

    async def insert_many(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        pool = await self._ensure_pool()
        async with pool.acquire() as connection, connection.transaction():
            await connection.executemany(
                """
                INSERT INTO agent_memory_records
                  (id, user_id, payload, fact_type, content, source_message_id,
                   confidence, status, entities, version, batch_id,
                   idempotency_key, content_hash, created_at, updated_at)
                VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                ON CONFLICT DO NOTHING
                """,
                [_row(record) for record in records],
            )

    async def get(self, record_id: str) -> MemoryRecord | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT payload FROM agent_memory_records WHERE id=$1",
                record_id,
            )
        return _record_from_payload(row["payload"]) if row else None

    async def update(
        self,
        record: MemoryRecord,
        *,
        read_version: int,
    ) -> MemoryRecord:
        pool = await self._ensure_pool()
        updated = record.model_copy(
            update={"version": read_version + 1, "updated_at": datetime.now(UTC)}
        )
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE agent_memory_records
                   SET payload=$1::jsonb, content=$2, source_message_id=$3,
                       confidence=$4, status=$5, entities=$6,
                       version=version+1, updated_at=$7
                 WHERE id=$8 AND version=$9
             RETURNING version
                """,
                updated.model_dump_json(),
                updated.fact_text,
                updated.source_message_id,
                updated.confidence,
                updated.status.value,
                updated.entities,
                updated.updated_at,
                updated.id,
                read_version,
            )
            if row is None:
                actual = await connection.fetchval(
                    "SELECT version FROM agent_memory_records WHERE id=$1",
                    record.id,
                )
                raise OptimisticLockError(record.id, read_version, int(actual or 0))
        return updated

    async def find_preferences(
        self,
        user_id: str,
        entities: list[str],
    ) -> list[MemoryRecord]:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT payload FROM agent_memory_records
                 WHERE user_id=$1 AND fact_type='preference' AND status='active'
                   AND (cardinality($2::text[]) = 0 OR entities && $2::text[])
                """,
                user_id,
                entities,
            )
        return [_record_from_payload(row["payload"]) for row in rows]

    async def find_warm_records(
        self,
        user_id: str,
        fields: list[str],
    ) -> list[MemoryRecord]:
        fact_types = sorted(_warm_fact_types(fields))
        if not fact_types:
            return []
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT payload FROM agent_memory_records
                 WHERE user_id=$1 AND fact_type = ANY($2::text[]) AND status='active'
                 ORDER BY updated_at DESC
                 LIMIT 100
                """,
                user_id,
                fact_types,
            )
        return [_record_from_payload(row["payload"]) for row in rows]

    async def get_by_ids(self, record_ids: list[str]) -> list[MemoryRecord]:
        if not record_ids:
            return []
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT payload FROM agent_memory_records WHERE id = ANY($1::text[])",
                record_ids,
            )
        return [_record_from_payload(row["payload"]) for row in rows]

    async def search_text(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int,
        min_confidence: float,
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                WITH query AS (
                    SELECT to_tsquery(
                        'english',
                        array_to_string(
                            tsvector_to_array(to_tsvector('english', $2)),
                            ' | '
                        )
                    ) AS terms
                )
                SELECT payload,
                       ts_rank_cd(
                           to_tsvector('english', content),
                           query.terms,
                           32
                       ) AS score
                  FROM agent_memory_records, query
                 WHERE user_id=$1
                   AND status='active'
                   AND confidence >= $3
                   AND to_tsvector('english', content) @@ query.terms
                 ORDER BY score DESC, updated_at DESC
                 LIMIT $4
                """,
                user_id,
                query,
                min_confidence,
                top_k,
            )
        return [
            SearchResult(
                record=_record_from_payload(row["payload"]),
                score=float(row["score"]),
            )
            for row in rows
        ]

    async def filter_active(
        self,
        records: list[MemoryRecord],
        *,
        min_confidence: float,
    ) -> list[MemoryRecord]:
        if not records:
            return []
        versions = {record.id: record.version for record in records}
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT payload, version FROM agent_memory_records
                 WHERE id = ANY($1::text[])
                   AND status='active'
                   AND confidence >= $2
                """,
                list(versions),
                min_confidence,
            )
        return [
            _record_from_payload(row["payload"])
            for row in rows
            if versions.get(_payload_id(row["payload"])) == int(row["version"])
        ]

    async def list_active_for_file_sync(
        self,
        *,
        min_confidence: float,
        limit: int,
    ) -> list[MemoryRecord]:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT payload FROM agent_memory_records
                 WHERE status='active' AND confidence >= $1
                 ORDER BY confidence DESC, updated_at DESC
                 LIMIT $2
                """,
                min_confidence,
                limit,
            )
        return [_record_from_payload(row["payload"]) for row in rows]


def _row(record: MemoryRecord) -> tuple[Any, ...]:
    return (
        record.id,
        record.user_id,
        record.model_dump_json(),
        record.fact_type.value,
        record.fact_text,
        record.source_message_id,
        record.confidence,
        record.status.value,
        record.entities,
        record.version,
        record.batch_id,
        record.idempotency_key,
        hashlib.sha256(record.fact_text.casefold().encode()).hexdigest(),
        record.created_at,
        record.updated_at,
    )


def _record_from_payload(payload: Any) -> MemoryRecord:
    return (
        MemoryRecord.model_validate_json(payload)
        if isinstance(payload, str)
        else MemoryRecord.model_validate(payload)
    )


def _payload_id(payload: Any) -> str:
    return _record_from_payload(payload).id


def _warm_fact_types(fields: list[str]) -> set[str]:
    fact_types: set[str] = set()
    if "preferences" in fields:
        fact_types.add("preference")
    if "static_profile" in fields:
        fact_types.add("fact")
    return fact_types


_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_memory_records (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    fact_type TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    source_message_id TEXT NOT NULL DEFAULT '',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    entities TEXT[] NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL DEFAULT 1,
    batch_id TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
ALTER TABLE agent_memory_records ADD COLUMN IF NOT EXISTS content TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_memory_records ADD COLUMN IF NOT EXISTS source_message_id TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_memory_records ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE agent_memory_records ADD COLUMN IF NOT EXISTS batch_id TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_memory_records ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT '';
UPDATE agent_memory_records
   SET content = COALESCE(NULLIF(content, ''), payload->>'fact_text', ''),
       source_message_id = COALESCE(
           NULLIF(source_message_id, ''),
           payload->'source_message_ids'->>0,
           ''
       ),
       confidence = CASE
           WHEN confidence = 0 AND payload ? 'confidence'
           THEN (payload->>'confidence')::DOUBLE PRECISION
           ELSE confidence
       END,
       batch_id = COALESCE(NULLIF(batch_id, ''), payload->>'batch_id', ''),
       content_hash = COALESCE(
           NULLIF(content_hash, ''),
           md5(lower(COALESCE(payload->>'fact_text', '')))
       )
 WHERE content = '' OR source_message_id = '' OR confidence = 0
    OR batch_id = '' OR content_hash = '';
CREATE INDEX IF NOT EXISTS idx_agent_memory_user_type
ON agent_memory_records(user_id, fact_type, status);
CREATE INDEX IF NOT EXISTS idx_agent_memory_user_active_confidence
ON agent_memory_records(user_id, status, confidence DESC, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_memory_source_message
ON agent_memory_records(user_id, source_message_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_batch_content
ON agent_memory_records(user_id, batch_id, content_hash)
WHERE batch_id <> '' AND content_hash <> '';
CREATE INDEX IF NOT EXISTS idx_agent_memory_entities
ON agent_memory_records USING GIN(entities);
CREATE INDEX IF NOT EXISTS idx_agent_memory_content_fts
ON agent_memory_records USING GIN(to_tsvector('english', content));
"""
