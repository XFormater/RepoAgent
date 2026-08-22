"""Run the pinned AgentBench OS dev split in disposable Docker containers."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from nanobot.providers.factory import load_provider_snapshot

SYSTEM = """You are operating a fresh Ubuntu system. Solve the user's OS task by calling exactly one tool per turn. Use bash_action to inspect or modify the system. Use answer_action once you have an exact answer (usually only a number or short string). Use finish_action only for implementation tasks after the requested change is complete. Never answer only in prose."""

TOOLS = [
    {"type": "function", "function": {"name": "bash_action", "description": "Execute bash code in the isolated Linux environment.", "parameters": {"type": "object", "properties": {"script": {"type": "string"}}, "required": ["script"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "finish_action", "description": "Indicate that an implementation task is finished.", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "answer_action", "description": "Submit the exact answer to the question.", "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}}},
]


async def command(*parts: str, stdin: str | None = None, timeout: float = 120) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *parts,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(process.communicate(stdin.encode() if stdin is not None else None), timeout)
    return process.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


def load_script(obj: Any, script_root: Path) -> tuple[str, str] | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return "bash", obj
    language = obj.get("language", "bash")
    if "file" in obj:
        return language, (script_root / obj["file"]).read_text(encoding="utf-8")
    return language, obj["code"]


async def run_script(container: str, script: tuple[str, str], params: list[str] | None = None) -> tuple[int, str, str]:
    language, code = script
    argv = params or []
    if language == "python":
        return await command("docker", "exec", "-w", "/root", container, "python3", "-c", code, *argv)
    return await command("docker", "exec", "-w", "/root", container, "bash", "-c", code, "--", *argv)


async def evaluate_answer(case: dict[str, Any], container: str, answer: str, script_root: Path) -> bool:
    evaluation = case["evaluation"]
    if "match" in evaluation:
        spec = evaluation["match"]
        if isinstance(spec, str):
            return answer.strip() == spec
        if "answer" in spec:
            return answer.strip() == spec["answer"] if spec.get("strip") else answer == spec["answer"]
        import re
        return re.search(spec["regex"], answer) is not None
    checks = evaluation["check"] if isinstance(evaluation["check"], list) else [evaluation["check"]]
    params = [str(answer)]
    for check in checks:
        script = load_script(check, script_root) if check is not None else load_script(evaluation["example"], script_root)
        assert script is not None
        code, out, _ = await run_script(container, script, params)
        if code != 0:
            return False
        params.append(out)
    return True


async def main_async(args: argparse.Namespace) -> None:
    cases: list[dict[str, Any]] = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.limit is not None:
        cases = cases[: args.limit]
    snapshot = load_provider_snapshot(args.config)
    prior: dict[int, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        prior = {row["case_index"]: row for row in old.get("results", [])}
        if args.retry_failures:
            prior = {index: row for index, row in prior.items() if row.get("passed")}
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_case(index: int, case: dict[str, Any]) -> dict[str, Any]:
        if index in prior:
            return prior[index]
        async with semaphore:
            started = time.monotonic()
            container = f"ab-os-{index}-{uuid.uuid4().hex[:8]}"
            transcript: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Now solve this problem in the new OS:\n\n" + case["description"]},
            ]
            answer = ""
            error = None
            try:
                code, _, err = await command("docker", "run", "-d", "--rm", "--name", container, "--memory", "1g", "--cpus", "2", args.image, "sleep", "infinity", timeout=180)
                if code:
                    raise RuntimeError(err)
                create = case.get("create")
                if isinstance(create, dict) and create.get("init") is not None:
                    init_items = create["init"] if isinstance(create["init"], list) else [create["init"]]
                    for item in init_items:
                        script = load_script(item, args.script_root)
                        if script is not None:
                            result, _, stderr = await run_script(container, script)
                            if result:
                                raise RuntimeError(f"init failed: {stderr}")
                if case.get("start") is not None:
                    script = load_script(case["start"], args.script_root)
                    assert script is not None
                    result, _, stderr = await command("docker", "exec", "-d", "-w", "/root", container, "bash", "-lc", script[1])
                    if result:
                        raise RuntimeError(f"start failed: {stderr}")
                for _round in range(args.round_limit):
                    response = await snapshot.provider.chat(
                        messages=transcript,
                        tools=TOOLS,
                        model=snapshot.model,
                        max_tokens=1024,
                        temperature=0.0,
                        reasoning_effort="none",
                        tool_choice="auto",
                    )
                    if response.finish_reason == "error":
                        raise RuntimeError(response.content or "provider error")
                    calls = response.tool_calls or []
                    if not calls:
                        transcript.append({"role": "assistant", "content": response.content or ""})
                        transcript.append({"role": "user", "content": "Call exactly one of the provided tools."})
                        continue
                    call = calls[0]
                    arguments = call.arguments if isinstance(call.arguments, dict) else {}
                    transcript.append({"role": "assistant", "content": response.content, "tool_calls": [call.to_openai_tool_call()]})
                    if call.name == "bash_action":
                        code, out, err = await command("docker", "exec", "-w", "/root", container, "bash", "-lc", str(arguments.get("script", "")), timeout=120)
                        content = (out + err)[:800] or "The output of the OS is empty."
                        if code:
                            content += f"\n(exit code {code})"
                        transcript.append({"role": "tool", "tool_call_id": call.id, "content": content})
                    elif call.name == "answer_action":
                        answer = str(arguments.get("answer", ""))
                        break
                    elif call.name == "finish_action":
                        answer = str(arguments.get("thought", ""))
                        break
                    else:
                        transcript.append({"role": "tool", "tool_call_id": call.id, "content": "Unknown tool."})
                passed = await evaluate_answer(case, container, answer, args.script_root)
            except Exception as exc:
                passed = False
                error = f"{type(exc).__name__}: {exc}"
            finally:
                await command("docker", "rm", "-f", container, timeout=30)
            return {"case_index": index, "passed": passed, "answer": answer, "error": error, "rounds": sum(1 for m in transcript if m.get("role") == "assistant"), "latency_ms": round((time.monotonic() - started) * 1000)}

    tasks = [asyncio.create_task(run_case(i, case)) for i, case in enumerate(cases)]
    results: list[dict[str, Any]] = []
    for completed, task in enumerate(asyncio.as_completed(tasks), 1):
        row = await task
        results.append(row)
        results.sort(key=lambda item: item["case_index"])
        passed = sum(bool(item["passed"]) for item in results)
        payload = {"benchmark": "AgentBench OS dev", "model": snapshot.model, "total": len(cases), "completed": len(results), "passed": passed, "accuracy": passed / len(results), "results": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{completed}/{len(cases)}] case {row['case_index']} {'PASS' if row['passed'] else 'FAIL'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("evals/vendor/agentbench_os/dev.json"))
    parser.add_argument("--script-root", type=Path, default=Path("/tmp/agentbench-src-WVP4sj/data/os_interaction/scripts/dev"))
    parser.add_argument("--config", type=Path, default=Path("/root/.nanobot/config.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/results/agentbench_os_deepseek.json"))
    parser.add_argument("--image", default="agentbench-os-default:local")
    parser.add_argument("--round-limit", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
