"""Validate downloaded agent benchmarks and build a small Harness eval suite."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any, Protocol, cast

from nanobot.harness import EvaluationCase, EvaluationThresholds


class _ArrowTable(Protocol):
    num_rows: int


class _ParquetModule(Protocol):
    def read_table(self, source: Path) -> _ArrowTable: ...


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _bfcl_suite(root: Path, limit: int, offset: int = 0) -> dict[str, object]:
    questions = _jsonl(root / "bfcl_v4" / "simple_python.json")
    answers = _jsonl(root / "bfcl_v4" / "simple_python_answers.json")
    answer_by_id = {item["id"]: item for item in answers}
    if len(answer_by_id) != len(answers):
        raise ValueError("BFCL answer IDs are not unique")
    cases: list[EvaluationCase] = []
    for question in questions[offset : offset + limit]:
        case_id = str(question["id"])
        answer = answer_by_id.get(case_id)
        if answer is None:
            raise ValueError(f"BFCL question {case_id} has no answer")
        function = question["function"][0]
        user_text = question["question"][0][0]["content"]
        function_name = str(function["name"])
        ground_truth = answer["ground_truth"][0][function_name]
        patterns = [rf'"name"\s*:\s*"{re.escape(function_name)}"']
        for argument, accepted in ground_truth.items():
            if accepted and accepted[0] != "":
                encoded = re.escape(json.dumps(accepted[0], ensure_ascii=False))
                patterns.append(rf'"{re.escape(argument)}"\s*:\s*{encoded}')
        prompt = (
            "Function-calling evaluation. Do not execute the function. Return only one JSON "
            "object with keys name and arguments.\n"
            f"User request: {user_text}\n"
            f"Available function: {json.dumps(function, ensure_ascii=False)}"
        )
        cases.append(
            EvaluationCase(
                case_id=f"bfcl:{case_id}",
                task_input=prompt,
                tags=frozenset({"bfcl", "tool-selection", "argument-grounding"}),
                required_patterns=tuple(patterns),
                forbidden_terms=("```",),
                max_output_chars=2_000,
                max_latency_ms=60_000,
                max_tool_calls=0,
            )
        )
    return {
        "thresholds": EvaluationThresholds(
            min_case_score=1.0,
            min_pass_rate=0.9,
            min_mean_score=0.98,
            max_mean_score_regression=0.01,
        ).model_dump(mode="json", by_alias=True),
        "cases": [case.model_dump(mode="json", by_alias=True) for case in cases],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, default=Path("evals/vendor"))
    parser.add_argument("--output", type=Path, default=Path("evals/suites/bfcl_smoke.json"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    if args.offset < 0:
        raise ValueError("offset must be non-negative")

    bfcl_questions = _jsonl(args.vendor / "bfcl_v4" / "simple_python.json")
    bfcl_answers = _jsonl(args.vendor / "bfcl_v4" / "simple_python_answers.json")
    agentbench_value: object = json.loads(
        (args.vendor / "agentbench_os" / "dev.json").read_text()
    )
    agentbench = cast(list[object], agentbench_value)
    parquet = cast(_ParquetModule, importlib.import_module("pyarrow.parquet"))
    swe_table = parquet.read_table(args.vendor / "swe_bench_lite" / "dev.parquet")
    if len(bfcl_questions) != len(bfcl_answers):
        raise ValueError("BFCL question/answer counts differ")
    if not isinstance(agentbench_value, list):
        raise ValueError("AgentBench OS dev must be a JSON list")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            _bfcl_suite(args.vendor, args.limit, args.offset), indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "bfcl_simple_python": len(bfcl_questions),
                "agentbench_os_dev": len(agentbench),
                "swe_bench_lite_dev": swe_table.num_rows,
                "generated_cases": min(args.limit, max(0, len(bfcl_questions) - args.offset)),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
