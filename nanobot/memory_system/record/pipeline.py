"""Best-effort asynchronous memory record pipeline."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from nanobot.memory_system.config import RecordConfig
from nanobot.memory_system.models import (
    Entity,
    ExtractedFact,
    FactType,
    MemoryRecord,
    MemoryStatus,
    Relation,
    uuid7,
)
from nanobot.memory_system.observability import logger, tracer
from nanobot.memory_system.record.conflict import ConflictResolver
from nanobot.memory_system.record.extractor import MemoryExtractor
from nanobot.memory_system.record.idempotency import (
    IdempotencyStore,
    make_idempotency_key,
)
from nanobot.memory_system.storage.embedding import EmbeddingProvider, embed_in_batches
from nanobot.memory_system.storage.graph_store import GraphStore
from nanobot.memory_system.storage.metadata_store import MetadataStore
from nanobot.memory_system.storage.vector_store import VectorStore


@dataclass(frozen=True)
class RecordOutcome:
    batch_id: str
    inserted: int = 0
    pending_review: int = 0
    skipped: bool = False
    error: str | None = None
    failed_backends: tuple[str, ...] = ()


class RecordPipeline:
    """Extract once, then retry stable records with the same batch ID and IDs."""

    def __init__(
        self,
        *,
        config: RecordConfig,
        extractor: MemoryExtractor,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        graph_store: GraphStore,
        metadata_store: MetadataStore,
        idempotency_store: IdempotencyStore,
        conflict_resolver: ConflictResolver,
    ) -> None:
        self.config = config
        self.extractor = extractor
        self.embedding = embedding
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.metadata_store = metadata_store
        self.idempotency_store = idempotency_store
        self.conflict_resolver = conflict_resolver

    async def record(
        self,
        *,
        user_id: str,
        messages: list[dict[str, str]],
        batch_id: str | None = None,
    ) -> RecordOutcome:
        batch_id = batch_id or uuid7()
        source_ids = [
            message["id"]
            for message in messages
            if message.get("id")
        ]
        if not source_ids:
            return RecordOutcome(batch_id=batch_id, skipped=True)
        idempotency_key: str | None = None
        facts = None
        records: list[MemoryRecord] | None = None
        claimed = False
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                with tracer.start_as_current_span("memory.record.pipeline") as span:
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("memory.batch_id", batch_id)
                        span.set_attribute("memory.attempt", attempt)
                    if facts is None:
                        facts = [
                            fact
                            for fact in await self.extractor.extract(messages)
                            if fact.confidence >= self.config.min_confidence
                        ]
                    if not facts:
                        return RecordOutcome(batch_id=batch_id, skipped=True)
                    if records is None:
                        fact_source_ids = [
                            source_id
                            for fact in facts
                            for source_id in fact.source_message_ids
                        ]
                        idempotency_key = make_idempotency_key(
                            fact_source_ids,
                            batch_id,
                        )
                        records = await self._build_records(
                            user_id=user_id,
                            batch_id=batch_id,
                            idempotency_key=idempotency_key,
                            facts=facts,
                        )
                    if not claimed:
                        assert idempotency_key is not None
                        claimed = await self.idempotency_store.claim(
                            idempotency_key,
                            self.config.idempotency.ttl,
                        )
                        if not claimed:
                            logger.info(
                                "memory_record_duplicate",
                                batch_id=batch_id,
                                idempotency_key=idempotency_key,
                            )
                            return RecordOutcome(
                                batch_id=batch_id,
                                skipped=True,
                            )
                    prepared = await self._resolve_preference_conflicts(records)
                    entities, relations = self._build_graph(prepared)
                    failed_backends = await self._write(prepared, entities, relations)
                    assert idempotency_key is not None
                    await self.idempotency_store.complete(
                        idempotency_key,
                        self.config.idempotency.ttl,
                    )
                    active_count = sum(
                        record.status == MemoryStatus.ACTIVE for record in prepared
                    )
                    pending_count = len(prepared) - active_count
                    logger.info(
                        "memory_record_completed",
                        batch_id=batch_id,
                        active=active_count,
                        pending_review=pending_count,
                        failed_backends=failed_backends,
                    )
                    return RecordOutcome(
                        batch_id=batch_id,
                        inserted=active_count,
                        pending_review=pending_count,
                        failed_backends=failed_backends,
                    )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "memory_record_retry",
                    batch_id=batch_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if claimed:
                    assert idempotency_key is not None
                    await self.idempotency_store.release(idempotency_key)
                    claimed = False
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self._retry_delay(attempt))

        return RecordOutcome(
            batch_id=batch_id,
            error=str(last_error) if last_error else "record failed",
        )

    async def _build_records(
        self,
        *,
        user_id: str,
        batch_id: str,
        idempotency_key: str,
        facts: list[ExtractedFact],
    ) -> list[MemoryRecord]:
        embeddings: list[list[float]]
        try:
            embeddings = await embed_in_batches(
                self.embedding,
                [fact.fact_text for fact in facts],
            )
        except Exception as exc:
            logger.warning("memory_embedding_failed", error=str(exc))
            embeddings = [[] for _ in facts]
        records: list[MemoryRecord] = []
        for fact, vector in zip(facts, embeddings, strict=True):
            status = (
                MemoryStatus.PENDING_REVIEW
                if fact.fact_type == FactType.HYPOTHESIS
                else MemoryStatus.ACTIVE
            )
            records.append(
                MemoryRecord(
                    user_id=user_id,
                    fact_text=fact.fact_text,
                    fact_type=fact.fact_type,
                    entities=fact.entities,
                    relations=fact.relations,
                    confidence=fact.confidence,
                    source_message_ids=fact.source_message_ids,
                    batch_id=batch_id,
                    idempotency_key=idempotency_key,
                    embedding=vector,
                    status=status,
                )
            )
        return records

    async def _resolve_preference_conflicts(
        self,
        records: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        prepared: list[MemoryRecord] = []
        for record in records:
            if record.fact_type != FactType.PREFERENCE:
                prepared.append(record)
                continue
            existing_records = await self.metadata_store.find_preferences(
                record.user_id,
                record.entities,
            )
            duplicate = next(
                (
                    existing
                    for existing in existing_records
                    if existing.id != record.id
                    if existing.fact_text.casefold() == record.fact_text.casefold()
                ),
                None,
            )
            if duplicate is not None:
                continue
            resolved = record
            for existing in existing_records:
                if existing.id == record.id:
                    continue
                resolved = await self.conflict_resolver.resolve(
                    existing,
                    resolved,
                    self.metadata_store,
                )
                if resolved.status == MemoryStatus.ACTIVE:
                    await self.vector_store.upsert(
                        [
                            existing.model_copy(
                                update={
                                    "status": MemoryStatus.SUPERSEDED,
                                    "version": existing.version + 1,
                                }
                            )
                        ]
                    )
            prepared.append(resolved)
        return prepared

    async def _write(
        self,
        records: list[MemoryRecord],
        entities: list[Entity],
        relations: list[Relation],
    ) -> tuple[str, ...]:
        failed: list[str] = []
        # PostgreSQL metadata is the source of truth. If it fails, let the
        # outer retry loop retry the stable record IDs and batch.
        await self.metadata_store.insert_many(records)
        active = [record for record in records if record.status == MemoryStatus.ACTIVE]
        vector_ready = [record for record in active if record.embedding]
        if len(vector_ready) != len(active):
            failed.append("embedding")
        if vector_ready:
            try:
                await self._retry_backend(
                    "qdrant",
                    lambda: self.vector_store.upsert(vector_ready),
                )
            except Exception as exc:
                failed.append("qdrant")
                logger.warning("memory_vector_write_failed", error=str(exc))
                if self.config.mode == "synchronous":
                    raise
        for entity in entities:
            try:
                await self._retry_backend(
                    "neo4j",
                    lambda entity=entity: self.graph_store.upsert_entity(entity),
                )
            except Exception as exc:
                if "neo4j" not in failed:
                    failed.append("neo4j")
                logger.warning(
                    "memory_graph_entity_write_failed",
                    entity_id=entity.id,
                    error=str(exc),
                )
                if self.config.mode == "synchronous":
                    raise
        for relation in relations:
            try:
                await self._retry_backend(
                    "neo4j",
                    lambda relation=relation: self.graph_store.upsert_relation(relation),
                )
            except Exception as exc:
                if "neo4j" not in failed:
                    failed.append("neo4j")
                logger.warning(
                    "memory_graph_relation_write_failed",
                    relation_id=relation.id,
                    error=str(exc),
                )
                if self.config.mode == "synchronous":
                    raise
        return tuple(failed)

    async def _retry_backend(
        self,
        name: str,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                await operation()
                return
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    logger.warning(
                        "memory_backend_retry",
                        backend=name,
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    await asyncio.sleep(self._retry_delay(attempt))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _build_graph(records: list[MemoryRecord]) -> tuple[list[Entity], list[Relation]]:
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []
        for record in records:
            if record.status != MemoryStatus.ACTIVE:
                continue
            user_entity_id = hashlib.sha256(
                f"{record.user_id}\x1f__user__".encode()
            ).hexdigest()
            entities.setdefault(
                user_entity_id,
                Entity(
                    id=user_entity_id,
                    name=record.user_id,
                    entity_type="User",
                    attributes={"user_id": record.user_id},
                ),
            )
            ids: list[str] = []
            name_to_id: dict[str, str] = {}
            for name in record.entities:
                entity_id = hashlib.sha256(
                    f"{record.user_id}\x1f{name.casefold()}".encode("utf-8")
                ).hexdigest()
                entity_type = RecordPipeline._entity_type(name, record.fact_type)
                entities.setdefault(
                    entity_id,
                    Entity(id=entity_id, name=name, entity_type=entity_type),
                )
                ids.append(entity_id)
                name_to_id[name.casefold()] = entity_id
                user_relation_type = {
                    FactType.PREFERENCE: "PREFERS",
                    FactType.GOAL: "INTERESTED_IN",
                }.get(record.fact_type, "RELATES_TO")
                relations.append(
                    Relation(
                        id=hashlib.sha256(
                            f"{record.id}\x1f{user_entity_id}\x1f{entity_id}".encode()
                        ).hexdigest(),
                        source_entity_id=user_entity_id,
                        target_entity_id=entity_id,
                        relation_type=user_relation_type,
                        source_memory_id=record.id,
                    )
                )
            for hint in record.relations:
                source_id = name_to_id.get(hint.source.casefold())
                target_id = name_to_id.get(hint.target.casefold())
                if source_id and target_id:
                    relations.append(
                        Relation(
                            id=hashlib.sha256(
                                (
                                    f"{record.id}\x1f{source_id}\x1f"
                                    f"{hint.relation_type}\x1f{target_id}"
                                ).encode()
                            ).hexdigest(),
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            relation_type=hint.relation_type,
                            source_memory_id=record.id,
                        )
                    )
            for index, source_id in enumerate(ids):
                for target_id in ids[index + 1 :]:
                    relation_id = hashlib.sha256(
                        f"{record.id}\x1f{source_id}\x1f{target_id}".encode("utf-8")
                    ).hexdigest()
                    relations.append(
                        Relation(
                            id=relation_id,
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            source_memory_id=record.id,
                        )
                    )
        return list(entities.values()), relations

    @staticmethod
    def _entity_type(name: str, fact_type: FactType) -> str:
        lowered = name.casefold()
        if fact_type == FactType.PREFERENCE:
            return "Preference"
        if fact_type == FactType.GOAL:
            return "Domain"
        if any(marker in lowered for marker in ("project", "项目", "系统", "平台")):
            return "Project"
        if any(
            marker in lowered
            for marker in (
                "python",
                "java",
                "redis",
                "postgres",
                "qdrant",
                "neo4j",
                "framework",
                "model",
                "diffusion",
            )
        ):
            return "Technology"
        return "Domain"

    def _retry_delay(self, attempt: int) -> float:
        if self.config.retry_backoff == "fixed":
            return 0.25
        return min(0.25 * (2**attempt), 4.0)
