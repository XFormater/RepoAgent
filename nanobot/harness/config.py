"""Explicit configuration for the multi-agent Harness runtime."""

from typing import Literal

from pydantic import Field

from nanobot.config_base import Base


class HarnessConfig(Base):
    enabled: bool = True
    backend: Literal["auto", "memory", "postgres_redis"] = "auto"
    key_prefix: str = "nanobot:harness"
    mailbox_ttl: int = Field(default=86_400, ge=60)
    idempotency_ttl: int = Field(default=86_400, ge=60)
    message_claim_idle_ms: int = Field(default=60_000, ge=1_000)
    max_queued_tasks_per_tenant: int = Field(default=1_000, ge=1)
    worker_capacity: int = Field(default=4, ge=1, le=128)
    worker_heartbeat_ttl: int = Field(default=90, ge=10)
    worker_execution_backend: Literal["in_process", "bwrap"] = "in_process"
    sandbox_network: Literal["host", "none"] = "host"
    sandbox_memory_mb: int = Field(default=1536, ge=256, le=32768)
    sandbox_process_limit: int = Field(default=128, ge=16, le=4096)
    sandbox_open_files_limit: int = Field(default=256, ge=64, le=65536)
    sandbox_file_size_mb: int = Field(default=512, ge=16, le=32768)
    checkpoint_max_bytes: int = Field(default=262_144, ge=4_096)
    retention_days: int = Field(default=30, ge=1)
