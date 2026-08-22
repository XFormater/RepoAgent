from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_locomo_memory_eval", ROOT / "scripts" / "run_locomo_memory_eval.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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
