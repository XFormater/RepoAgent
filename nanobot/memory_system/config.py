"""Pydantic settings for semantic long-term memory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from nanobot.config_base import Base


class MemoryLLMConfig(Base):
    model: str | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=128)


class IdempotencyConfig(Base):
    key_strategy: Literal["source_ids_and_batch"] = "source_ids_and_batch"
    store: Literal["memory", "redis", "postgres"] = "redis"
    ttl: int = Field(default=86_400, ge=60)


class RecordConfig(Base):
    enabled: bool = True
    mode: Literal["best_effort", "synchronous"] = "best_effort"
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff: Literal["exponential", "fixed"] = "exponential"
    conflict_strategy: Literal["latest_wins", "merge_by_llm", "manual_review"] = "latest_wins"
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    recent_turns: int = Field(default=8, ge=1, le=50)
    llm: MemoryLLMConfig = Field(default_factory=MemoryLLMConfig)
    idempotency: IdempotencyConfig = Field(default_factory=IdempotencyConfig)


class VectorRetrieveConfig(Base):
    top_k: int = Field(default=20, ge=1, le=200)
    score_threshold: float = Field(default=0.7, ge=-1.0, le=1.0)
    hnsw_ef: int = Field(default=128, ge=8)


class LexicalRetrieveConfig(Base):
    enabled: bool = False
    top_k: int = Field(default=30, ge=1, le=200)
    min_score: float = Field(default=0.0, ge=0.0)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    vector_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    lexical_weight: float = Field(default=1.5, ge=0.0, le=10.0)


class GraphRetrieveConfig(Base):
    enabled: bool = True
    max_depth: int = Field(default=2, ge=1, le=4)
    max_paths: int = Field(default=100, ge=1, le=2000)
    timeout_ms: int = Field(default=100, ge=1)


class RerankerConfig(Base):
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = Field(default=5, ge=1, le=64)
    timeout_ms: int = Field(default=200, ge=1)
    fallback: Literal["original_order"] = "original_order"
    max_batch_size: int = Field(default=32, ge=1, le=128)


class CacheConfig(Base):
    warm_cache_enabled: bool = True
    redis_ttl: int = Field(default=3600, ge=1)
    preload_fields: list[str] = Field(
        default_factory=lambda: ["preferences", "static_profile"]
    )


class RetrieveConfig(Base):
    enabled: bool = True
    trigger: Literal["first_turn", "every_turn", "on_demand"] = "first_turn"
    recent_turns: int = Field(default=3, ge=0, le=20)
    max_injected_chars: int = Field(default=6000, ge=256, le=50_000)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    vector: VectorRetrieveConfig = Field(default_factory=VectorRetrieveConfig)
    lexical: LexicalRetrieveConfig = Field(default_factory=LexicalRetrieveConfig)
    graph: GraphRetrieveConfig = Field(default_factory=GraphRetrieveConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


class VectorDBConfig(Base):
    type: Literal["qdrant", "memory"] = "qdrant"
    host: str = "localhost"
    rest_port: int = Field(default=6333, ge=1, le=65_535)
    port: int = Field(default=6334, ge=1, le=65_535)
    api_key: str | None = None
    https: bool = False
    collection_prefix: str = "agent_memory"
    prefer_grpc: bool = True
    hnsw_m: int = Field(default=16, ge=4)
    hnsw_ef_construct: int = Field(default=128, ge=8)


class GraphDBConfig(Base):
    type: Literal["neo4j", "memory"] = "neo4j"
    uri: str = "bolt://localhost:7687"
    database: str = "neo4j"
    username: str = "neo4j"
    password: str = "nanobot-memory"
    initialize_schema: bool = True


class MetadataDBConfig(Base):
    type: Literal["postgresql", "memory"] = "postgresql"
    dsn: str = "postgresql://nanobot:nanobot-memory@localhost:5432/nanobot"
    min_pool_size: int = Field(default=1, ge=1, le=100)
    max_pool_size: int = Field(default=10, ge=1, le=200)


class RedisConfig(Base):
    url: str = "redis://localhost:6379/0"
    socket_timeout: float = Field(default=2.0, gt=0.0, le=60.0)


class ShortTermConfig(Base):
    enabled: bool = False
    key_prefix: str = "nanobot:stm"
    conversation_ttl: int = Field(default=86_400, ge=60)
    task_ttl: int = Field(default=86_400, ge=60)
    reasoning_ttl: int = Field(default=21_600, ge=60)
    tool_ttl: int = Field(default=86_400, ge=60)
    run_state_ttl: int = Field(default=3_600, ge=60)
    context_ttl: int = Field(default=21_600, ge=60)
    max_messages: int = Field(default=200, ge=1, le=10_000)
    max_tool_result_chars: int = Field(default=20_000, ge=256, le=1_000_000)
    runtime_context_messages: int = Field(default=12, ge=0, le=100)
    runtime_context_tasks: int = Field(default=8, ge=0, le=100)
    runtime_context_tools: int = Field(default=12, ge=0, le=100)
    runtime_context_max_chars: int = Field(default=12_000, ge=512, le=100_000)


class FileSyncConfig(Base):
    """Low-frequency projection from semantic memory into core MEMORY.md."""

    enabled: bool = True
    cron: str = "0 3 * * 0"
    min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    max_records: int = Field(default=100, ge=1, le=1000)
    max_chars: int = Field(default=20_000, ge=1000, le=100_000)


class EmbeddingConfig(Base):
    provider: Literal["openai", "hash", "lexical_hash"] = "openai"
    model: str = "text-embedding-3-large"
    dimension: int = Field(default=3072, ge=8)
    api_key: str | None = None
    base_url: str | None = None


class StorageConfig(Base):
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    graph_db: GraphDBConfig = Field(default_factory=GraphDBConfig)
    metadata_db: MetadataDBConfig = Field(default_factory=MetadataDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class MemorySettings(BaseSettings):
    """Root semantic-memory settings.

    ``enabled`` defaults to false so existing installations keep the current
    file-based Dream memory until external services and credentials are ready.
    """

    model_config = SettingsConfigDict(
        alias_generator=lambda value: "".join(
            word.capitalize() if index else word
            for index, word in enumerate(value.split("_"))
        ),
        populate_by_name=True,
        env_prefix="NANOBOT_MEMORY__",
        env_nested_delimiter="__",
        extra="ignore",
    )

    enabled: bool = False
    short_term: ShortTermConfig = Field(default_factory=ShortTermConfig)
    record: RecordConfig = Field(default_factory=RecordConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    file_sync: FileSyncConfig = Field(default_factory=FileSyncConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def validate_memory_backends(self) -> "MemorySettings":
        if not self.enabled:
            return self
        if (
            self.record.idempotency.store == "postgres"
            and self.storage.metadata_db.type != "postgresql"
        ):
            raise ValueError("postgres idempotency requires a postgresql metadata_db")
        return self


def load_memory_settings(path: str | Path) -> MemorySettings:
    """Load a standalone YAML settings file for embedding in other agents."""
    raw = Path(path).read_text(encoding="utf-8")
    loaded: object = yaml.safe_load(os.path.expandvars(raw))
    value: object = loaded or {}
    if not isinstance(value, dict):
        raise ValueError("memory settings YAML must contain an object")
    data = cast(dict[str, Any], value)
    nested = data.get("memory", data)
    if not isinstance(nested, dict):
        raise ValueError("the memory key must contain an object")
    return MemorySettings.model_validate(nested)
