"""Deterministic offline quality evaluation for Harness runs.

The evaluator deliberately starts with reproducible assertions rather than an
LLM judge.  A recorded output can therefore be promoted to a CI quality gate
without model variance, credentials, or network access.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class EvaluationCase(_EvaluationModel):
    """One golden case and its observable quality contract."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)

    case_id: str = Field(min_length=1, max_length=200)
    task_input: str = Field(default="", max_length=1_000_000)
    tags: frozenset[str] = Field(default_factory=frozenset)
    expected_success: bool = True
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    required_patterns: tuple[str, ...] = ()
    min_output_chars: int = Field(default=0, ge=0)
    max_output_chars: int | None = Field(default=None, ge=1)
    max_latency_ms: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_length_bounds(self) -> "EvaluationCase":
        if self.max_output_chars is not None and self.max_output_chars < self.min_output_chars:
            raise ValueError("max_output_chars must be >= min_output_chars")
        for pattern in self.required_patterns:
            re.compile(pattern)
        return self


class EvaluationObservation(_EvaluationModel):
    """Provider-independent observations captured from one Harness run."""

    case_id: str = Field(min_length=1, max_length=200)
    output: str = ""
    succeeded: bool = True
    latency_ms: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)


class CriterionResult(_EvaluationModel):
    name: str
    passed: bool
    detail: str = ""


class CaseEvaluation(_EvaluationModel):
    case_id: str
    tags: frozenset[str]
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    criteria: tuple[CriterionResult, ...]


class EvaluationThresholds(_EvaluationModel):
    min_case_score: float = Field(default=1.0, ge=0.0, le=1.0)
    min_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_mean_score: float = Field(default=1.0, ge=0.0, le=1.0)
    max_mean_score_regression: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationReport(_EvaluationModel):
    passed: bool
    case_count: int
    pass_count: int
    pass_rate: float
    mean_score: float
    baseline_mean_score: float | None = None
    mean_score_delta: float | None = None
    tag_scores: dict[str, float] = Field(default_factory=dict)
    cases: tuple[CaseEvaluation, ...]
    failures: tuple[str, ...] = ()


def _criterion(name: str, passed: bool, detail: str = "") -> CriterionResult:
    return CriterionResult(name=name, passed=passed, detail=detail)


def evaluate_case(case: EvaluationCase, observation: EvaluationObservation) -> CaseEvaluation:
    """Score a recorded run against one golden contract.

    Every configured criterion has equal weight. This keeps the first quality
    gate explainable: a failed score always maps to a concrete assertion.
    """
    if case.case_id != observation.case_id:
        raise ValueError("case and observation IDs differ")
    folded = observation.output.casefold()
    criteria = [
        _criterion(
            "success",
            observation.succeeded == case.expected_success,
            f"expected={case.expected_success} actual={observation.succeeded}",
        )
    ]
    for term in case.required_terms:
        criteria.append(
            _criterion(f"required_term:{term}", term.casefold() in folded, "term must appear")
        )
    for term in case.forbidden_terms:
        criteria.append(
            _criterion(f"forbidden_term:{term}", term.casefold() not in folded, "term must not appear")
        )
    for pattern in case.required_patterns:
        criteria.append(
            _criterion(
                f"required_pattern:{pattern}",
                re.search(pattern, observation.output, flags=re.IGNORECASE | re.MULTILINE) is not None,
                "pattern must match",
            )
        )
    if case.min_output_chars > 0:
        criteria.append(
            _criterion(
                "min_output_chars",
                len(observation.output) >= case.min_output_chars,
                f"limit={case.min_output_chars} actual={len(observation.output)}",
            )
        )
    if case.max_output_chars is not None:
        criteria.append(
            _criterion(
                "max_output_chars",
                len(observation.output) <= case.max_output_chars,
                f"limit={case.max_output_chars} actual={len(observation.output)}",
            )
        )
    if case.max_latency_ms is not None:
        criteria.append(
            _criterion(
                "max_latency_ms",
                observation.latency_ms <= case.max_latency_ms,
                f"limit={case.max_latency_ms} actual={observation.latency_ms}",
            )
        )
    if case.max_tool_calls is not None:
        criteria.append(
            _criterion(
                "max_tool_calls",
                observation.tool_calls <= case.max_tool_calls,
                f"limit={case.max_tool_calls} actual={observation.tool_calls}",
            )
        )
    score = sum(item.passed for item in criteria) / len(criteria)
    return CaseEvaluation(
        case_id=case.case_id,
        tags=case.tags,
        passed=all(item.passed for item in criteria),
        score=score,
        criteria=tuple(criteria),
    )


def evaluate_suite(
    cases: Iterable[EvaluationCase],
    observations: Iterable[EvaluationObservation],
    *,
    thresholds: EvaluationThresholds | None = None,
    baseline_mean_score: float | None = None,
) -> EvaluationReport:
    """Evaluate a complete dataset and apply aggregate/regression gates."""
    policy = thresholds or EvaluationThresholds()
    case_list = list(cases)
    observation_list = list(observations)
    observation_map = {item.case_id: item for item in observation_list}
    if len(observation_map) != len(observation_list):
        raise ValueError("observation case_id values must be unique")
    expected_ids = {item.case_id for item in case_list}
    if len(expected_ids) != len(case_list):
        raise ValueError("evaluation case_id values must be unique")
    missing = expected_ids - observation_map.keys()
    extra = observation_map.keys() - expected_ids
    if missing or extra:
        raise ValueError(f"case coverage mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    results = tuple(evaluate_case(case, observation_map[case.case_id]) for case in case_list)
    count = len(results)
    if count == 0:
        raise ValueError("evaluation suite must contain at least one case")
    pass_count = sum(result.score >= policy.min_case_score for result in results)
    pass_rate = pass_count / count
    mean_score = sum(result.score for result in results) / count
    tag_values: defaultdict[str, list[float]] = defaultdict(list)
    for result in results:
        for tag in result.tags:
            tag_values[tag].append(result.score)
    tag_scores = {
        tag: sum(scores) / len(scores) for tag, scores in sorted(tag_values.items())
    }
    failures: list[str] = []
    if pass_rate < policy.min_pass_rate:
        failures.append(f"pass_rate {pass_rate:.4f} < {policy.min_pass_rate:.4f}")
    if mean_score < policy.min_mean_score:
        failures.append(f"mean_score {mean_score:.4f} < {policy.min_mean_score:.4f}")
    delta = None if baseline_mean_score is None else mean_score - baseline_mean_score
    if delta is not None and delta < -policy.max_mean_score_regression:
        failures.append(
            f"mean_score regression {delta:.4f} < -{policy.max_mean_score_regression:.4f}"
        )
    return EvaluationReport(
        passed=not failures,
        case_count=count,
        pass_count=pass_count,
        pass_rate=pass_rate,
        mean_score=mean_score,
        baseline_mean_score=baseline_mean_score,
        mean_score_delta=delta,
        tag_scores=tag_scores,
        cases=results,
        failures=tuple(failures),
    )
