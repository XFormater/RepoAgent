from nanobot.agent.memory import MemoryStore
from nanobot.memory_system.config import FileSyncConfig
from nanobot.memory_system.file_sync import MemoryFileSynchronizer
from nanobot.memory_system.models import FactType, MemoryRecord
from nanobot.memory_system.storage.metadata_store import InMemoryMetadataStore


async def test_file_sync_preserves_manual_core_and_replaces_managed_section(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_memory("# Core Memory\n\nManually curated fact.\n")
    metadata = InMemoryMetadataStore()
    await metadata.insert_many([
        MemoryRecord(
            user_id="user-1",
            fact_text="User works on ultrasound generation.",
            fact_type=FactType.FACT,
            confidence=0.95,
            source_message_ids=["m1"],
            batch_id="b1",
            idempotency_key="k1",
        ),
        MemoryRecord(
            user_id="user-1",
            fact_text="Transient low confidence fact.",
            fact_type=FactType.HYPOTHESIS,
            confidence=0.4,
            source_message_ids=["m2"],
            batch_id="b2",
            idempotency_key="k2",
        ),
    ])
    synchronizer = MemoryFileSynchronizer(
        metadata,
        FileSyncConfig(min_confidence=0.85),
    )

    first = await synchronizer.sync(store)
    second = await synchronizer.sync(store)
    content = store.read_memory()

    assert first.changed is True
    assert first.records == 1
    assert second.changed is False
    assert "Manually curated fact." in content
    assert "User works on ultrasound generation." in content
    assert "Transient low confidence fact." not in content
    assert content.count("BEGIN DATABASE MEMORY PROJECTION") == 1
