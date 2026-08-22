"""Generate and officially grade patches for the pinned SWE-bench Lite split."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.factory import load_provider_snapshot
from nanobot.utils.llm_runtime import runtime_from_provider_snapshot


async def command(*parts: str, stdin: str | None = None, timeout: float = 1800) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *parts,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(process.communicate(stdin.encode() if stdin is not None else None), timeout)
    return process.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def ensure_image(image: str, mode: str) -> None:
    exists, _, _ = await command("docker", "image", "inspect", image, timeout=30)
    if exists == 0:
        return
    if mode == "docker" or (mode == "auto" and shutil.which("skopeo") is None):
        code, _, err = await command("docker", "pull", image, timeout=3600)
        if code:
            raise RuntimeError(f"image pull failed: {err[-1000:]}")
        return
    archive = Path("/tmp") / f"swebench-{uuid.uuid4().hex}.tar"
    try:
        source = image if image.startswith("docker.io/") else "docker.io/" + image
        code, _, err = await command(
            "skopeo", "copy", "--override-arch", "amd64",
            f"docker://{source}", f"docker-archive:{archive}:{image}", timeout=3600,
        )
        if code:
            raise RuntimeError(f"skopeo image copy failed: {err[-1000:]}")
        code, _, err = await command("docker", "load", "-i", str(archive), timeout=1800)
        if code:
            raise RuntimeError(f"docker load failed: {err[-1000:]}")
    finally:
        archive.unlink(missing_ok=True)


class DockerExecTool(Tool):
    def __init__(self, container: str):
        self.container = container

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a non-interactive shell command in the isolated repository container at /testbed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to execute."}}, "required": ["command"], "additionalProperties": False}

    async def execute(self, **kwargs: Any) -> str:
        code, out, err = await command("docker", "exec", "-w", "/testbed", self.container, "bash", "-lc", str(kwargs["command"]), timeout=600)
        text = out + err
        if code:
            text += f"\n(exit code {code})"
        return text or "Command completed with no output."


async def main_async(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = pq.read_table(args.dataset).to_pylist()
    indexed_rows = list(enumerate(rows))
    indexed_rows = [(index, row) for index, row in indexed_rows if index >= args.start_index]
    if args.shard_count > 1:
        indexed_rows = [item for offset, item in enumerate(indexed_rows) if offset % args.shard_count == args.shard_index]
    rows = [row for _, row in indexed_rows]
    if args.limit is not None:
        rows = rows[: args.limit]
    selected = set(args.instance or [])
    if selected:
        rows = [row for row in rows if row["instance_id"] in selected]
    snapshot = load_provider_snapshot(args.config)
    runtime = runtime_from_provider_snapshot(snapshot).with_generation_overrides(temperature=0.0, max_tokens=2048, reasoning_effort="none")
    prior: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        prior = {row["instance_id"]: row for row in old.get("results", [])}
    results: list[dict[str, Any]] = list(prior.values())

    for position, case in enumerate(rows, 1):
        instance_id = case["instance_id"]
        if instance_id in prior:
            print(f"[{position}/{len(rows)}] {instance_id} cached", flush=True)
            continue
        started = time.monotonic()
        container = "swe-agent-" + uuid.uuid4().hex[:12]
        patch = ""
        report: dict[str, Any] = {}
        error = None
        image = case["image"]
        try:
            await ensure_image(image, args.image_transfer)
            run_code, _, run_err = await command("docker", "run", "-d", "--rm", "--name", container, "--user", "root", image, "tail", "-f", "/dev/null", timeout=180)
            if run_code:
                raise RuntimeError(run_err)
            registry = ToolRegistry()
            registry.register(DockerExecTool(container))
            prompt = (
                "You are fixing a real repository checked out at /testbed. Analyze the issue, inspect the code, edit the working tree, and run focused tests. "
                "Do not merely explain a patch: make the changes with shell commands. Do not use git reset, git checkout, git clean, or commit. "
                "When the fix is complete, respond concisely without calling another tool.\n\nISSUE:\n" + case["problem_statement"]
            )
            agent_result = await AgentRunner().run(AgentRunSpec(
                initial_messages=[{"role": "system", "content": "You are an autonomous software-engineering agent with an isolated shell."}, {"role": "user", "content": prompt}],
                tools=registry,
                runtime=runtime,
                max_iterations=args.max_iterations,
                max_tool_result_chars=12_000,
                provider_retry_mode="standard",
                finalize_on_max_iterations=True,
                llm_timeout_s=300,
            ))
            _, patch, patch_err = await command("docker", "exec", "-w", "/testbed", container, "git", "diff", "--binary", timeout=60)
            if patch_err and not patch:
                raise RuntimeError(f"could not capture patch: {patch_err}")
            eval_code, eval_out, eval_err = await command("docker", "exec", "-i", "-w", "/testbed", container, "bash", "-s", stdin=case["eval_script"], timeout=args.test_timeout)
            log_path = args.logs / f"{instance_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(eval_out + eval_err, encoding="utf-8")
            from swebench.harness.grading import get_eval_report
            from swebench.harness.utils import make_test_spec
            prediction = {"instance_id": instance_id, "model_name_or_path": snapshot.model, "model_patch": patch}
            report = get_eval_report(make_test_spec(case), prediction, str(log_path), include_tests_status=True)
            resolved = bool(report.get(instance_id, {}).get("resolved"))
            stop_reason = agent_result.stop_reason
        except Exception as exc:
            resolved = False
            stop_reason = "error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            await command("docker", "rm", "-f", container, timeout=30)
            if args.remove_images:
                await command("docker", "image", "rm", image, timeout=120)
        row = {"instance_id": instance_id, "resolved": resolved, "model_patch": patch, "report": report, "stop_reason": stop_reason, "error": error, "latency_ms": round((time.monotonic() - started) * 1000)}
        results.append(row)
        results.sort(key=lambda item: item["instance_id"])
        resolved_count = sum(bool(item["resolved"]) for item in results)
        payload = {"benchmark": "SWE-bench Lite pinned dev subset", "model": snapshot.model, "total": len(rows), "completed": len(results), "resolved": resolved_count, "resolve_rate": resolved_count / len(results), "results": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{position}/{len(rows)}] {instance_id} {'RESOLVED' if resolved else 'UNRESOLVED'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evals/vendor/swe_bench_lite/dev.parquet"))
    parser.add_argument("--config", type=Path, default=Path("/root/.nanobot/config.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/results/swebench_lite_deepseek.json"))
    parser.add_argument("--logs", type=Path, default=Path("evals/results/swebench_logs"))
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--test-timeout", type=int, default=1800)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--instance", action="append")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--remove-images", action="store_true")
    parser.add_argument("--image-transfer", choices=("auto", "docker", "skopeo"), default="auto")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
