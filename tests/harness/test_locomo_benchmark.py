from __future__ import annotations

import importlib.util
from pathlib import Path

from nanobot.memory_system.models import FactType, MemoryRecord, RetrievalResult, SearchResult

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_locomo_memory_eval", ROOT / "scripts" / "run_locomo_memory_eval.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

HYBRID_SPEC = importlib.util.spec_from_file_location(
    "run_locomo_hybrid_eval", ROOT / "scripts" / "run_locomo_hybrid_eval.py"
)
assert HYBRID_SPEC and HYBRID_SPEC.loader
HYBRID_MODULE = importlib.util.module_from_spec(HYBRID_SPEC)
HYBRID_SPEC.loader.exec_module(HYBRID_MODULE)


def test_official_locomo_scoring_rules() -> None:
    assert MODULE.locomo_score("Adoption agency", "Adoption agencies", 4) == 1.0
    assert MODULE.locomo_score("May 7 2023", "7 May 2023", 2) == 1.0
    assert MODULE.locomo_score("psychology, counseling", "Psychology, counseling", 1) == 1.0
    assert MODULE.locomo_score("No information available", "Rome", 5) == 1.0
    assert MODULE.locomo_score("Rome", "Rome", 5) == 0.0


def test_fixed_subset_is_deterministic_and_balanced_by_conversation() -> None:
    import json

    data = json.loads((ROOT / "evals/vendor/locomo/locomo10.json").read_text())
    first = MODULE.select_questions(data, per_conversation=10)
    second = MODULE.select_questions(data, per_conversation=10)
    assert [MODULE.question_id(s["sample_id"], i) for s, i, _ in first] == [
        MODULE.question_id(s["sample_id"], i) for s, i, _ in second
    ]
    assert len(first) == 100
    counts: dict[str, int] = {}
    for sample, _, _ in first:
        counts[sample["sample_id"]] = counts.get(sample["sample_id"], 0) + 1
    assert set(counts.values()) == {10}
    categories = {category: 0 for category in MODULE.CATEGORY_NAMES}
    for _, _, qa in first:
        categories[int(qa["category"])] += 1
    assert categories == {1: 14, 2: 16, 3: 5, 4: 42, 5: 23}
    assert all(
        ("adversarial_answer" in qa and "answer" not in qa)
        for _, _, qa in first
        if int(qa["category"]) == 5
    )


def test_hybrid_records_are_hierarchical_and_exclude_qa_annotations() -> None:
    sample = {
        "sample_id": "conv-test",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1 pm on 1 May, 2023",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted Buster."}
            ],
        },
        "observation": {
            "session_1_observation": {"Alice": [["Alice adopted Buster.", "D1:1"]]}
        },
        "session_summary": {"session_1_summary": "Alice told Bob about Buster."},
        "event_summary": "GOLD EVENT SUMMARY MUST NOT BE INDEXED",
        "qa": [{"question": "gold question", "answer": "gold answer", "evidence": ["D1:1"]}],
    }

    records = HYBRID_MODULE.build_records(sample)

    assert len(records) == 4
    assert {record.batch_id.rsplit(":", 1)[-1] for record in records} == {
        "dialogue",
        "observation",
        "observation_bundle",
        "summary",
    }
    indexed = "\n".join(record.fact_text for record in records)
    assert "GOLD EVENT SUMMARY" not in indexed
    assert "gold question" not in indexed
    assert "gold answer" not in indexed


def test_relative_date_normalization_is_derived_only_from_turn_and_session_date() -> None:
    normalized = HYBRID_MODULE.resolve_relative_dates(
        "Two days ago I wanted to be alone with nature.",
        "2:00 pm on 12 April, 2022",
    )

    assert normalized == '"two days ago" resolves to 10 April 2022'


def test_hybrid_context_policy_preserves_atomic_facts_before_aggregates() -> None:
    def result_item(record_id: str, kind: str, score: float) -> SearchResult:
        return SearchResult(
            record=MemoryRecord(
                id=record_id,
                user_id="user-1",
                fact_text=f"{kind} text",
                fact_type=FactType.FACT,
                confidence=1.0,
                source_message_ids=[record_id],
                batch_id=f"namespace:sample:{kind}",
                idempotency_key=record_id,
            ),
            score=score,
        )

    result = RetrievalResult(
        memories=[
            result_item("bundle", "observation_bundle", 3.0),
            result_item("dialogue", "dialogue", 2.0),
            result_item("observation", "observation", 1.0),
        ]
    )

    atomic_first = HYBRID_MODULE.context_order(result, "atomic-first")
    atomic_only = HYBRID_MODULE.context_order(result, "atomic-only")

    assert [item.record.id for item in atomic_first.memories] == [
        "dialogue",
        "observation",
        "bundle",
    ]
    assert [item.record.id for item in atomic_only.memories] == [
        "dialogue",
        "observation",
    ]
