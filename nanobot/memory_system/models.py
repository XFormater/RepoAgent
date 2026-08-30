"""Validated data models used by memory pipelines and storage adapters."""

from __future__ import annotations

import secrets
import time
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FactType(StrEnum):
    PREFERENCE = "preference"
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    GOAL = "goal"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    PENDING_REVIEW = "pending_review"


def uuid7() -> str:
    """Generate an RFC 9562 UUIDv7 on Python versions without ``uuid.uuid7``."""
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    random_bits = secrets.randbits(74)
    value = timestamp_ms << 80
    value |= 0x7 << 76
    value |= ((random_bits >> 62) & 0xFFF) << 64
    value |= 0b10 << 62
    value |= random_bits & ((1 << 62) - 1)
    return str(uuid.UUID(int=value))


class ExtractedFact(BaseModel):
    fact_text: str = Field(min_length=1, max_length=8000)
    fact_type: FactType
    entities: list[str] = Field(default_factory=list, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    source_message_ids: list[str] = Field(min_length=1, max_length=100)
    relations: list["ExtractedRelation"] = Field(default_factory=list, max_length=64)

    @field_validator("fact_text")
    @classmethod
    def clean_fact_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("entities", "source_message_ids")
    @classmethod
    def deduplicate_strings(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in values if item.strip()))


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=uuid7)
    user_id: str = Field(min_length=1)
    fact_text: str = Field(min_length=1)
    fact_type: FactType
    entities: list[str] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_message_ids: list[str] = Field(min_length=1)
    batch_id: str
    idempotency_key: str
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = Field(default=1, ge=1)
    status: MemoryStatus = MemoryStatus.ACTIVE

    @property
    def source_message_id(self) -> str:
        return self.source_message_ids[0]


class ExtractedRelation(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    target: str = Field(min_length=1, max_length=500)
    relation_type: Literal[
        "INTERESTED_IN",
        "WORKS_ON",
        "USES",
        "PREFERS",
        "RELATES_TO",
    ] = "RELATES_TO"


class ConversationMessage(BaseModel):
    message_id: str = Field(min_length=1)
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskState(BaseModel):
    task_id: str = Field(min_length=1)
    goal: str
    current_step: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    pending_steps: list[str] = Field(default_factory=list)
    intermediate_results: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReasoningState(BaseModel):
    constraints: list[str] = Field(default_factory=list)
    current_requirements: list[str] = Field(default_factory=list)
    confirmed_information: dict[str, Any] = Field(default_factory=dict)
    temporary_decisions: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolExecutionState(BaseModel):
    execution_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class SessionContextState(BaseModel):
    """Bounded, explicitly stored context needed to resume one agent session."""

    current_goal: str | None = None
    active_constraints: list[str] = Field(default_factory=list)
    confirmed_context: dict[str, Any] = Field(default_factory=dict)
    active_files: list[str] = Field(default_factory=list)
    active_skills: list[str] = Field(default_factory=list)
    last_summary: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Entity(BaseModel):
    id: str
    name: str
    entity_type: str = "Concept"
    attributes: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)


class Relation(BaseModel):
    id: str = Field(default_factory=uuid7)
    source_entity_id: str
    target_entity_id: str
    relation_type: str = "RELATES_TO"
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    source_memory_id: str


class SearchResult(BaseModel):
    record: MemoryRecord
    score: float


class SubGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    supporting_memories: list[MemoryRecord] = Field(default_factory=list)


class RankedResult(BaseModel):
    text: str
    score: float
    source_index: int


class RetrievalResult(BaseModel):
    memories: list[SearchResult] = Field(default_factory=list)
    graph_memories: list[MemoryRecord] = Field(default_factory=list)
    cache_memories: list[MemoryRecord] = Field(default_factory=list)
    degraded: bool = False
    degradation_reason: str | None = None
    vector_candidate_count: int = Field(default=0, ge=0)
    lexical_candidate_count: int = Field(default=0, ge=0)
    graph_candidate_count: int = Field(default=0, ge=0)
    cache_candidate_count: int = Field(default=0, ge=0)

    @property
    def records(self) -> list[MemoryRecord]:
        ordered = [
            *(item.record for item in self.memories),
            *self.graph_memories,
            *self.cache_memories,
        ]
        seen: set[str] = set()
        return [
            record
            for record in ordered
            if not (record.id in seen or seen.add(record.id))
        ]
