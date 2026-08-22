from pathlib import Path

import pytest

from scripts.run_agent_offline_benchmark import (
    REQUIRED_CATEGORIES,
    build_report,
    load_suite,
    sanitized_environment,
)


def test_committed_suite_is_balanced_and_has_runnable_cases() -> None:
    suite = load_suite(Path("evals/suites/agent_self_capability_v1.json"))
    categories = [case["category"] for case in suite["cases"]]
    assert set(categories) == REQUIRED_CATEGORIES
    assert {category: categories.count(category) for category in REQUIRED_CATEGORIES} == {
        category: 6 for category in REQUIRED_CATEGORIES
    }
    assert all(Path(case["nodeId"].split("::", 1)[0]).exists() for case in suite["cases"])


def test_report_applies_overall_and_category_gates() -> None:
    suite = {
        "benchmark": "test",
        "version": "1",
        "thresholds": {"minPassRate": 0.5, "minCategoryPassRate": 1.0},
    }
    results = [
        {"category": "a", "passed": True, "latencyMs": 1},
        {"category": "b", "passed": False, "latencyMs": 3},
    ]
    report = build_report(suite, results)
    assert report["passRate"] == pytest.approx(0.5)
    assert report["passed"] is False
    assert report["categoryScores"]["b"]["passRate"] == 0.0


def test_offline_environment_removes_credentials_and_proxies(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy")
    env = sanitized_environment()
    assert "DEEPSEEK_API_KEY" not in env
    assert env["HTTPS_PROXY"] == ""
    assert env["NO_PROXY"] == "*"
    assert env["NANOBOT_OFFLINE_BENCHMARK"] == "1"
