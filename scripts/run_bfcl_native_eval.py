"""Evaluate the pinned BFCL simple_python split through native tool calling.

The runner intentionally does not execute calls.  It checks the selected function and
arguments against every accepted BFCL ground-truth alternative and writes an
incremental, resumable JSON report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from nanobot.providers.factory import load_provider_snapshot


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "__", name)


def normalize_schema(value: Any) -> Any:
    """Translate BFCL's Python-ish annotations into JSON Schema types."""
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: normalize_schema(item) for key, item in value.items()}
    if "type" not in normalized:
        return normalized
    schema_type = normalized.get("type")
    if isinstance(schema_type, dict) and len(normalized) == 1:
        nested = schema_type
        normalized.update(nested)
        schema_type = nested.get("type")
    elif isinstance(schema_type, dict):
        return normalized
    normalized["type"] = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}.get(schema_type, schema_type)
    return normalized


def equivalent(actual: Any, accepted: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(accepted, (int, float)):
        return math.isclose(float(actual), float(accepted), rel_tol=1e-9, abs_tol=1e-9)
    return actual == accepted


def matches(actual_name: str, actual_args: dict[str, Any], alternatives: list[dict[str, Any]]) -> bool:
    for alternative in alternatives:
        if actual_name not in alternative:
            continue
        expected = alternative[actual_name]
        okay = True
        for key, candidates in expected.items():
            values = candidates if isinstance(candidates, list) else [candidates]
            if key not in actual_args:
                if "" not in values:
                    okay = False
                    break
            elif not any(equivalent(actual_args[key], value) for value in values if value != ""):
                okay = False
                break
        if okay and not (set(actual_args) - set(expected)):
            return True
    return False


async def main_async(args: argparse.Namespace) -> None:
    cases = read_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    answers = {row["id"]: row["ground_truth"] for row in read_jsonl(args.answers)}
    snapshot = load_provider_snapshot(args.config)
    prior: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        prior = {row["case_id"]: row for row in report.get("results", [])}
        if args.retry_errors:
            prior = {case_id: row for case_id, row in prior.items() if row.get("finish_reason") != "error"}
    semaphore = asyncio.Semaphore(args.concurrency)

    async def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        if case["id"] in prior:
            return prior[case["id"]]
        async with semaphore:
            started = time.monotonic()
            name_map: dict[str, str] = {}
            tools = []
            for function in case["function"]:
                original = function["name"]
                sanitized = safe_name(original)
                name_map[sanitized] = original
                spec = normalize_schema(json.loads(json.dumps(function)))
                spec["name"] = sanitized
                tools.append({"type": "function", "function": spec})
            messages = [
                {"role": "system", "content": "Call the function that exactly satisfies the user request. Do not answer in prose."},
                *case["question"][0],
            ]
            response = await snapshot.provider.chat(
                messages=messages,
                tools=tools,
                model=snapshot.model,
                max_tokens=512,
                temperature=0.0,
                reasoning_effort="none",
                tool_choice="auto",
            )
            calls = response.tool_calls or []
            parsed = []
            for call in calls:
                original_name = name_map.get(call.name, call.name)
                call_args = call.arguments if isinstance(call.arguments, dict) else {}
                parsed.append({"name": original_name, "arguments": call_args})
            passed = len(parsed) == 1 and matches(
                parsed[0]["name"], parsed[0]["arguments"], answers[case["id"]]
            )
            return {
                "case_id": case["id"],
                "passed": passed,
                "calls": parsed,
                "finish_reason": response.finish_reason,
                "error": response.content if response.finish_reason == "error" else None,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }

    tasks = [asyncio.create_task(evaluate(case)) for case in cases]
    results: list[dict[str, Any]] = []
    for completed, task in enumerate(asyncio.as_completed(tasks), 1):
        row = await task
        results.append(row)
        results.sort(key=lambda item: int(item["case_id"].rsplit("_", 1)[-1]))
        passed = sum(bool(item["passed"]) for item in results)
        payload = {
            "benchmark": "BFCL V4 simple_python",
            "model": snapshot.model,
            "total": len(cases),
            "completed": len(results),
            "passed": passed,
            "accuracy": passed / len(results),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{completed}/{len(cases)}] {row['case_id']} {'PASS' if row['passed'] else 'FAIL'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("evals/vendor/bfcl_v4/simple_python.json"))
    parser.add_argument("--answers", type=Path, default=Path("evals/vendor/bfcl_v4/simple_python_answers.json"))
    parser.add_argument("--config", type=Path, default=Path("/root/.nanobot/config.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/results/bfcl_v4_simple_python_deepseek.json"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
