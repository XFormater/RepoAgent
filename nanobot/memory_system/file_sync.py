"""Low-frequency projection of semantic records into human-readable MEMORY.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nanobot.memory_system.config import FileSyncConfig
from nanobot.memory_system.storage.metadata_store import MetadataStore

if TYPE_CHECKING:
    from nanobot.agent.memory import MemoryStore

_START = "<!-- BEGIN DATABASE MEMORY PROJECTION -->"
_END = "<!-- END DATABASE MEMORY PROJECTION -->"


@dataclass(frozen=True)
class FileSyncOutcome:
    records: int
    changed: bool


class MemoryFileSynchronizer:
    """Own only a marked section; preserve manually curated core memory."""

    def __init__(self, metadata: MetadataStore, config: FileSyncConfig) -> None:
        self.metadata = metadata
        self.config = config

    async def sync(self, store: MemoryStore) -> FileSyncOutcome:
        records = await self.metadata.list_active_for_file_sync(
            min_confidence=self.config.min_confidence,
            limit=self.config.max_records,
        )
        seen: set[tuple[str, str]] = set()
        lines = [
            _START,
            "## Database Memory Projection (managed)",
            "",
            "High-confidence semantic memories synchronized on the configured schedule.",
        ]
        for record in records:
            key = (record.user_id, record.fact_text.casefold())
            if key in seen:
                continue
            seen.add(key)
            line = (
                f"- [{record.user_id}] ({record.fact_type.value}, "
                f"confidence={record.confidence:.2f}) {record.fact_text}"
            )
            candidate = "\n".join([*lines, line, _END])
            if len(candidate) > self.config.max_chars:
                break
            lines.append(line)
        lines.append(_END)
        managed = "\n".join(lines)
        current = store.read_memory().strip()
        updated = self._replace_managed_section(current, managed).strip() + "\n"
        old = store.read_memory()
        changed = updated != old
        if changed:
            store.write_memory(updated)
        return FileSyncOutcome(records=max(0, len(lines) - 5), changed=changed)

    @staticmethod
    def _replace_managed_section(current: str, managed: str) -> str:
        start = current.find(_START)
        end = current.find(_END)
        if start >= 0 and end >= start:
            end += len(_END)
            return f"{current[:start].rstrip()}\n\n{managed}{current[end:]}"
        return f"{current}\n\n{managed}" if current else managed
