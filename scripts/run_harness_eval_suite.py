"""Run taskInput prompts through the nanobot CLI and record eval observations."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def _final_json(text: str) -> str:
    candidates = re.findall(r"\{[^\n]*\}", text)
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "name" in value and "arguments" in value:
            return json.dumps(value, ensure_ascii=False)
    return text.strip()[-4_000:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nanobot", default=".venv/bin/nanobot")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    payload: dict[str, Any] = json.loads(args.suite.read_text(encoding="utf-8"))
    observations: list[dict[str, object]] = []
    for index, case in enumerate(payload["cases"], start=1):
        started = time.monotonic()
        try:
            process = subprocess.run(
                [args.nanobot, "agent", "-m", case["taskInput"]],
                cwd=Path.cwd(),
                env=os.environ.copy(),
                text=True,
                capture_output=True,
                timeout=args.timeout,
                check=False,
            )
            combined = process.stdout + "\n" + process.stderr
            output = _final_json(combined)
            succeeded = process.returncode == 0
        except subprocess.TimeoutExpired as exc:
            output = f"timeout after {args.timeout}s: {exc}"
            succeeded = False
        latency_ms = int((time.monotonic() - started) * 1_000)
        observations.append(
            {
                "caseId": case["caseId"],
                "output": output,
                "succeeded": succeeded,
                "latencyMs": latency_ms,
                "toolCalls": 0,
            }
        )
        print(
            f"[{index}/{len(payload['cases'])}] {case['caseId']} "
            f"{'ok' if succeeded else 'failed'} {latency_ms}ms",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(observations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
