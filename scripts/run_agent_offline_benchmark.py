"""Run the versioned, deterministic self-capability benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_CATEGORIES = {
    "planning_state_machine",
    "vfs_sandbox",
    "delegation_subagents",
    "memory_management",
    "human_intervention",
}


def load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark cases must be a non-empty list")
    ids: set[str] = set()
    categories: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        for field in ("caseId", "category", "capability", "nodeId"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"case {index} has invalid {field}")
        if case["caseId"] in ids:
            raise ValueError(f"duplicate caseId: {case['caseId']}")
        ids.add(case["caseId"])
        categories.add(case["category"])
        timeout = case.get("timeoutSeconds", 30)
        if not isinstance(timeout, int) or timeout < 1:
            raise ValueError(f"case {case['caseId']} has invalid timeoutSeconds")
    missing = REQUIRED_CATEGORIES - categories
    extra = categories - REQUIRED_CATEGORIES
    if missing or extra:
        raise ValueError(f"category coverage mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return payload


def sanitized_environment() -> dict[str, str]:
    env = os.environ.copy()
    secret_tokens = (
        "API_KEY", "APIKEY", "TOKEN", "PASSWORD", "SECRET",
        "OPENAI", "ANTHROPIC", "DEEPSEEK", "GEMINI", "GROQ",
    )
    for key in list(env):
        if any(token in key.upper() for token in secret_tokens):
            env.pop(key, None)
    env.update({
        "NANOBOT_OFFLINE_BENCHMARK": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "PYTHONHASHSEED": "0",
    })
    return env


def run_case(case: dict[str, Any], *, pytest: str, root: Path) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [pytest, "-q", "--disable-warnings", str(case["nodeId"])],
            cwd=root,
            env=sanitized_environment(),
            text=True,
            capture_output=True,
            timeout=int(case.get("timeoutSeconds", 30)),
            check=False,
        )
        status = "passed" if completed.returncode == 0 else "failed"
        return_code = completed.returncode
        diagnostic = (completed.stdout + "\n" + completed.stderr).strip()[-4000:]
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        return_code = None
        diagnostic = f"timeout after {case.get('timeoutSeconds', 30)}s: {exc}"
    return {
        "caseId": case["caseId"],
        "category": case["category"],
        "capability": case["capability"],
        "nodeId": case["nodeId"],
        "status": status,
        "passed": status == "passed",
        "score": 1.0 if status == "passed" else 0.0,
        "latencyMs": round((time.monotonic() - started) * 1000),
        "returnCode": return_code,
        "diagnostic": diagnostic,
    }


def build_report(suite: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        buckets[result["category"]].append(result)
    category_scores = {
        category: {
            "passed": sum(bool(item["passed"]) for item in items),
            "total": len(items),
            "passRate": sum(bool(item["passed"]) for item in items) / len(items),
            "meanLatencyMs": round(sum(item["latencyMs"] for item in items) / len(items)),
        }
        for category, items in sorted(buckets.items())
    }
    passed = sum(bool(item["passed"]) for item in results)
    pass_rate = passed / len(results)
    thresholds = suite.get("thresholds", {})
    min_pass_rate = float(thresholds.get("minPassRate", 1.0))
    min_category = float(thresholds.get("minCategoryPassRate", 1.0))
    failures = []
    if pass_rate < min_pass_rate:
        failures.append(f"passRate {pass_rate:.4f} < {min_pass_rate:.4f}")
    for category, score in category_scores.items():
        if score["passRate"] < min_category:
            failures.append(f"{category}.passRate {score['passRate']:.4f} < {min_category:.4f}")
    return {
        "benchmark": suite["benchmark"],
        "version": suite["version"],
        "executionMode": "offline-deterministic-isolated-process",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkCredentialsRemoved": True,
        },
        "passed": not failures,
        "caseCount": len(results),
        "passCount": passed,
        "passRate": pass_rate,
        "totalLatencyMs": sum(item["latencyMs"] for item in results),
        "categoryScores": category_scores,
        "failures": failures,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=Path("evals/suites/agent_self_capability_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/results/agent_self_capability_v1_report.json"))
    parser.add_argument("--pytest", default=".venv/bin/pytest")
    parser.add_argument("--category", choices=sorted(REQUIRED_CATEGORIES))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite_path = root / args.suite if not args.suite.is_absolute() else args.suite
    suite = load_suite(suite_path)
    cases = suite["cases"]
    if args.category:
        cases = [case for case in cases if case["category"] == args.category]
    results = []
    for index, case in enumerate(cases, 1):
        result = run_case(case, pytest=args.pytest, root=root)
        results.append(result)
        print(f"[{index}/{len(cases)}] {case['caseId']}: {result['status'].upper()} ({result['latencyMs']}ms)", flush=True)
    report = build_report(suite, results)
    output = root / args.output if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pass rate: {report['passCount']}/{report['caseCount']} ({report['passRate']:.2%})")
    print(f"report: {output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
