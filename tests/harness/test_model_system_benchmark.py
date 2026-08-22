from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_agent_system_benchmark import (
    LAYERS,
    aggregate,
    answer_matches,
    context_history,
    load_suite,
    normalize_answer,
)


def test_committed_suite_covers_all_four_layers() -> None:
    root = Path(__file__).resolve().parents[2]
    suite = load_suite(root / "evals/suites/agent_system_model_v1.json")
    cases = suite["cases"]

    assert len(cases) == 12
    assert {case["layer"] for case in cases} == set(LAYERS)
    assert all(sum(case["layer"] == layer for case in cases) == 3 for layer in LAYERS)


def test_memory_v2_suite_has_benefit_and_safety_slices() -> None:
    root = Path(__file__).resolve().parents[2]
    suite = load_suite(root / "evals/suites/memory_benefit_v2.json")
    cases = suite["cases"]

    assert len(cases) == 20
    assert sum(case["memoryCaseType"] == "benefit_required" for case in cases) == 12
    assert sum(case["memoryCaseType"] == "conflict_override" for case in cases) == 4
    assert sum(case["memoryCaseType"] == "irrelevant_non_regression" for case in cases) == 4
    composition = next(case for case in cases if case["caseId"] == "memory-v2-compose-aurora")
    assert composition["retrievalTerms"] == ["EAST-4", "WAVE-9"]


def test_context_v2_suite_has_information_depth_and_tool_slices() -> None:
    root = Path(__file__).resolve().parents[2]
    suite = load_suite(root / "evals/suites/context_governance_v2.json")
    cases = suite["cases"]
    types = {case["contextCaseType"] for case in cases}

    assert len(cases) == 16
    assert {"latest_authority", "recent_dependency", "early_dependency"} <= types
    assert {"system_anchor", "split_dependency", "recent_tool_result"} <= types
    assert "early_tool_result" in types
    assert context_history(cases[0]) == context_history(cases[0])


def test_suite_rejects_missing_layer(tmp_path: Path) -> None:
    suite = {
        "cases": [
            {
                "caseId": "only-one",
                "layer": "harness_regression",
                "scenario": "lookup",
                "prompt": "p",
                "expected": "e",
            }
        ]
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")

    with pytest.raises(ValueError, match="layer coverage mismatch"):
        load_suite(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("VALUE-1", "VALUE-1"),
        ("`VALUE-1`", "VALUE-1"),
        ("答案：VALUE-1", "VALUE-1"),
        ("```text\nVALUE-1\n```", "VALUE-1"),
    ],
)
def test_answer_normalization_accepts_format_only_variance(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected
    assert answer_matches(raw, expected)


def test_aggregate_keeps_fixed_ablation_and_artifact_paths(tmp_path: Path) -> None:
    suite = {
        "benchmark": "test",
        "version": "1",
        "thresholds": {
            "minLayerPassRate": 1.0,
            "minContextTokenReduction": 0.1,
            "minMemoryAccuracyGain": 0.5,
        },
    }
    results = [
        {"caseId": "h", "layer": "harness_regression", "passed": True},
        {
            "caseId": "c",
            "layer": "context_governance",
            "passed": True,
            "estimatedTokenReduction": 0.5,
            "control": {"correct": True},
            "treatment": {"correct": True},
        },
        {
            "caseId": "m",
            "layer": "memory_benefit",
            "passed": True,
            "memoryCaseType": "benefit_required",
            "retrievalHit": True,
            "control": {"correct": False, "usage": {"prompt_tokens": 50}},
            "treatment": {
                "correct": True,
                "usage": {"prompt_tokens": 125},
                "injectedChars": 300,
            },
        },
        {"caseId": "r", "layer": "recovery_correctness", "passed": True},
    ]

    report = aggregate(
        suite,
        results,
        model="real-model",
        run_id="run-1",
        artifact_path=tmp_path,
        event_count=10,
    )

    assert report["passed"] is True
    assert report["executionMode"] == "online-model-in-the-loop"
    assert set(report["evaluationPaths"]) == {
        "fixedBenchmark",
        "controlledExperiments",
        "artifactAggregation",
    }
    assert report["evaluationPaths"]["controlledExperiments"]["memoryBenefit"][
        "accuracyGain"
    ] == 1.0
    assert report["evaluationPaths"]["controlledExperiments"]["memoryBenefit"][
        "harmfulRegressionRate"
    ] == 0.0
    assert report["evaluationPaths"]["controlledExperiments"]["memoryBenefit"][
        "meanPromptTokenOverhead"
    ] == 75.0


def test_partial_run_only_gates_selected_layers(tmp_path: Path) -> None:
    suite = {
        "benchmark": "test",
        "version": "1",
        "thresholds": {
            "minLayerPassRate": 1.0,
            "minContextTokenReduction": 0.1,
            "minMemoryAccuracyGain": 0.5,
        },
    }
    report = aggregate(
        suite,
        [{"caseId": "h", "layer": "harness_regression", "passed": True}],
        model="real-model",
        run_id="partial",
        artifact_path=tmp_path,
        event_count=2,
    )

    assert report["passed"] is True
    assert report["layerScores"]["context_governance"]["passRate"] is None
