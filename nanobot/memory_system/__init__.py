"""Optional semantic long-term memory for nanobot.

The legacy ``MemoryStore``/Dream files remain the durable human-readable
memory.  This package adds an opt-in semantic record/retrieve layer.
"""

from nanobot.memory_system.config import MemorySettings
from nanobot.memory_system.models import (
    Entity,
    ExtractedFact,
    FactType,
    MemoryRecord,
    MemoryStatus,
    Relation,
    RetrievalResult,
    SearchResult,
)
from nanobot.memory_system.service import LongTermMemoryService

__all__ = [
    "Entity",
    "ExtractedFact",
    "FactType",
    "LongTermMemoryService",
    "MemoryRecord",
    "MemorySettings",
    "MemoryStatus",
    "Relation",
    "RetrievalResult",
    "SearchResult",
]
