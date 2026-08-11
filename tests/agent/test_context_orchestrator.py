from unittest.mock import MagicMock

from nanobot.agent.context import ContextBuilder
from nanobot.agent.context_orchestrator import ContextMetrics, ContextOrchestrator
from nanobot.runtime_context import RuntimeContextBlock


def test_core_memory_is_runtime_context_not_duplicated_in_system(tmp_path):
    builder = ContextBuilder(tmp_path)
    builder.memory.write_memory("Core durable fact")
    orchestrator = ContextOrchestrator(
        builder=builder,
        consolidator=MagicMock(),
        auto_compact=MagicMock(),
    )
    block = orchestrator.core_memory_block()

    assert block is not None
    messages = orchestrator.build_messages(
        history=[],
        current_message="hello",
        runtime_context_blocks=[block],
        session_key="cli:test",
    )

    assert "Core durable fact" not in messages[0]["content"]
    assert "<core_long_term_memory>" in messages[-1]["content"]


def test_context_metrics_exposes_diagnostic_snapshot():
    metrics = ContextMetrics()
    metrics.blocks([
        RuntimeContextBlock(source="redis", content="one"),
        RuntimeContextBlock(source="database", content="two"),
    ])
    metrics.compacted(3, reason="tool_result")
    metrics.observe_tokens(42, model="test")

    snapshot = metrics.snapshot()

    assert snapshot["block.redis"] == 1
    assert snapshot["block.database"] == 1
    assert snapshot["compaction.tool_result"] == 3
    assert snapshot["input_tokens"] == 42
