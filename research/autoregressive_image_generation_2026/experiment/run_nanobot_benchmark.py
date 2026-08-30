"""Run a fixed-corpus nanobot token benchmark in orchestrator or plain-agent mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from nanobot.agent.hook import AgentHook, AgentRunHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.loader import load_config, resolve_config_env_vars
from nanobot.nanobot import Nanobot
from nanobot.providers.image_generation import image_gen_provider_configs
from nanobot.repo_agent.agents import RepositoryControlClient
from nanobot.utils.llm_runtime import LLMRuntime

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROMPT_PATH = EXPERIMENT_DIR / "benchmark_prompt.md"


def _add_usage(target: dict[str, int], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


class UsageHook(AgentHook):
    def __init__(self) -> None:
        super().__init__()
        self.usage: dict[str, int] = {}
        self.runs = 0
        self.tools: list[str] = []

    async def after_run(self, context: AgentRunHookContext) -> None:
        self.runs += 1
        _add_usage(self.usage, context.usage)
        self.tools.extend(context.tools_used)


class MeasuredControlClient(RepositoryControlClient):
    def __init__(self) -> None:
        super().__init__()
        self.usage: dict[str, int] = {}
        self.calls = 0

    async def complete(
        self,
        prompt: str,
        *,
        runtime: LLMRuntime,
        max_tokens: int,
    ) -> str:
        result = await self.runner.run(
            AgentRunSpec(
                initial_messages=[{"role": "system", "content": prompt}],
                tools=ToolRegistry(),
                runtime=runtime.with_generation_overrides(
                    temperature=0.0,
                    max_tokens=max_tokens,
                ),
                max_iterations=1,
                max_tool_result_chars=4_000,
                fail_on_tool_error=True,
                finalize_on_max_iterations=False,
            )
        )
        self.calls += 1
        _add_usage(self.usage, result.usage)
        if result.stop_reason == "error":
            raise RuntimeError(result.error or "repository control agent failed")
        return result.final_content or ""


def _build_bot(mode: str) -> tuple[Nanobot, MeasuredControlClient | None]:
    config = resolve_config_env_vars(load_config())
    config.agents.defaults.workspace = str(ROOT)
    config.memory.enabled = False
    config.harness.enabled = False
    config.vfs.enabled = False
    config.repository_agent.research.max_repositories = 3
    config.repository_agent.research.max_concurrent_analyses = 1
    config.repository_agent.cache_dir = (
        "research/autoregressive_image_generation_2026/experiment/orchestrator-cache"
    )
    if mode == "plain":
        config.repository_agent.enabled = False

    loop = AgentLoop.from_config(
        config,
        image_generation_provider_configs=image_gen_provider_configs(config),
    )
    bot = Nanobot(loop, config=config)
    if mode != "orchestrator":
        return bot, None

    orchestrator = loop.repository_orchestrator
    if orchestrator is None:
        raise RuntimeError("repository orchestrator is not configured")
    measured = MeasuredControlClient()
    orchestrator.client = measured
    orchestrator.research.client = measured
    orchestrator.intent_agent.client = measured
    orchestrator.planner_agent.client = measured
    return bot, measured


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("orchestrator", "plain"))
    args = parser.parse_args()

    base_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    if args.mode == "orchestrator":
        prompt = (
            "研究并比较以下三个指定的 GitHub 开源项目："
            "ShareLab-SII/UniAR、amazon-far/BAR、stepfun-ai/NextStep-1。"
            "只选择这三个项目，不要替换候选。下载源码并进行实现级分析。\n\n"
            + base_prompt
        )
    else:
        prompt = (
            "这是普通 Agent 横向基准。不要调用 Repository Research Orchestrator，"
            "不要重新下载仓库；使用下面给出的三个本地源码目录完成同一份实现级分析。\n\n"
            + base_prompt
        )

    bot, control = _build_bot(args.mode)
    hook = UsageHook()
    started = time.monotonic()
    result = await bot.run(
        prompt,
        session_key=f"benchmark:{args.mode}:{int(time.time())}",
        ephemeral=True,
        hooks=[hook],
    )
    elapsed = time.monotonic() - started

    total_usage = dict(hook.usage)
    control_usage: dict[str, int] = {}
    control_calls = 0
    if control is not None:
        control_usage = dict(control.usage)
        control_calls = control.calls
        _add_usage(total_usage, control_usage)

    report_path = EXPERIMENT_DIR / f"{args.mode}_report.md"
    metrics_path = EXPERIMENT_DIR / f"{args.mode}_metrics.json"
    report_path.write_text(result.content, encoding="utf-8")
    metrics = {
        "mode": args.mode,
        "elapsed_seconds": round(elapsed, 3),
        "usage": total_usage,
        "agent_runner_usage": hook.usage,
        "control_usage": control_usage,
        "agent_runs": hook.runs,
        "control_calls": control_calls,
        "tool_calls": len(hook.tools),
        "tools": hook.tools,
        "report_chars": len(result.content),
        "stop_reason": result.stop_reason,
        "report_path": str(report_path.relative_to(ROOT)),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
