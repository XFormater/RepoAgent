import pytest

from nanobot.harness import (
    EvaluationCase,
    EvaluationObservation,
    EvaluationThresholds,
    evaluate_case,
    evaluate_suite,
)


def test_case_evaluation_explains_content_and_efficiency_failures():
    case = EvaluationCase(
        case_id="repo-evidence",
        tags={"repository", "grounding"},
        required_terms=("src/main.py",),
        forbidden_terms=("I guess",),
        required_patterns=(r"line\s+\d+",),
        min_output_chars=20,
        max_latency_ms=1_000,
        max_tool_calls=3,
    )
    observation = EvaluationObservation(
        case_id="repo-evidence",
        output="I guess src/main.py contains the implementation on line 12.",
        latency_ms=1_500,
        tool_calls=2,
    )

    result = evaluate_case(case, observation)

    assert result.passed is False
    assert result.score == pytest.approx(5 / 7)
    failed = {criterion.name for criterion in result.criteria if not criterion.passed}
    assert failed == {"forbidden_term:I guess", "max_latency_ms"}


def test_suite_reports_tag_slices_and_blocks_baseline_regression():
    cases = [
        EvaluationCase(case_id="a", tags={"research"}, required_terms=("source",)),
        EvaluationCase(case_id="b", tags={"research", "safety"}, forbidden_terms=("token",)),
    ]
    observations = [
        EvaluationObservation(case_id="a", output="no citation"),
        EvaluationObservation(case_id="b", output="safe response"),
    ]

    report = evaluate_suite(
        cases,
        observations,
        thresholds=EvaluationThresholds(
            min_case_score=1.0,
            min_pass_rate=0.5,
            min_mean_score=0.7,
            max_mean_score_regression=0.05,
        ),
        baseline_mean_score=1.0,
    )

    assert report.pass_rate == 0.5
    assert report.mean_score == pytest.approx(0.75)
    assert report.tag_scores == {"research": pytest.approx(0.75), "safety": 1.0}
    assert report.passed is False
    assert any("regression" in failure for failure in report.failures)


def test_suite_rejects_incomplete_case_coverage():
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_suite(
            [EvaluationCase(case_id="expected")],
            [EvaluationObservation(case_id="unexpected")],
        )
