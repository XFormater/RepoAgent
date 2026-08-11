"""Entrypoint for one process-isolated Harness task."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config, resolve_config_env_vars
from nanobot.harness.agent_executor import HarnessAgentExecutor
from nanobot.harness.context import RuntimeContext
from nanobot.harness.models import AgentSpec, ReviewResult
from nanobot.harness.sandbox_executor import _RESULT_PREFIX


async def _run(config_path: Path) -> None:
    request = json.loads(sys.stdin.buffer.read())
    context = RuntimeContext.model_validate(request["context"])
    spec = AgentSpec.model_validate(request["spec"])
    config = resolve_config_env_vars(load_config(config_path), config_path=config_path)
    loop = AgentLoop.from_config(config, MessageBus())
    try:
        await loop._connect_mcp()
        result = await HarnessAgentExecutor(loop)(context, spec)
    finally:
        await loop.close_mcp()
    envelope = (
        {"kind": "review", "value": result.model_dump(mode="json")}
        if isinstance(result, ReviewResult)
        else {"kind": "text", "value": result}
    )
    print(_RESULT_PREFIX + json.dumps(envelope, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()
