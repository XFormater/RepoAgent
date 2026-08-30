"""Run BFCL V4 through GitHubResearch_Agent's real AgentRunner.

BFCL remains the source of datasets, stateful execution backends, and scoring.
Every model decision is admitted through the project's provider, context
governor, tool registry, and shared agent state machine.  Results are therefore
an agent-in-the-loop system benchmark and are not directly leaderboard-comparable
with a model-only BFCL submission.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.loader import load_config, resolve_config_env_vars
from nanobot.config.schema import ModelPresetConfig
from nanobot.providers.factory import build_provider_snapshot
from nanobot.utils.llm_runtime import runtime_from_provider_snapshot

MODEL_REGISTRY_NAME = "GitHubResearch-Agent-DeepSeek-V4-Flash-FC"
MODEL_API_NAME = "deepseek-v4-flash"
AGENT_ERROR_PLACEHOLDER = "[Assistant reply unavailable due to model error.]"
OFFICIAL_SOURCE = Path("evals/vendor/bfcl_official_source")
DEFAULT_RUN_DIR = Path("evals/runs/bfcl-v4-agent-full-flash-20260821")
DEFAULT_RESULT_DIR = DEFAULT_RUN_DIR / "results"
DEFAULT_SCORE_DIR = DEFAULT_RUN_DIR / "scores"
DEFAULT_AUDIT = DEFAULT_RUN_DIR / "audit.json"
DEFAULT_CONFIG = Path("/root/.nanobot/config.json")


class AsyncBridge:
    """Keep one event loop/provider client behind BFCL's worker threads."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coroutine: Any, *, timeout: float = 240.0) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)


class CaptureTool(Tool):
    """Exercise ToolRegistry validation while BFCL retains authoritative state."""

    def __init__(self, definition: dict[str, Any]) -> None:
        self._name = str(definition["name"])
        self._description = str(definition.get("description", ""))
        self._parameters = deepcopy(definition.get("parameters", {"type": "object"}))

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._parameters)

    async def execute(self, **arguments: Any) -> str:
        return json.dumps(
            {"status": "captured_for_bfcl_execution", "arguments": arguments},
            ensure_ascii=False,
        )


@dataclass(slots=True)
class AgentStepResponse:
    model_responses: Any
    assistant_message: dict[str, Any]
    tool_call_ids: list[str]
    reasoning_content: str
    input_tokens: int
    output_tokens: int


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    value = message.get("tool_calls")
    return value if isinstance(value, list) else []


def _assistant_from_delta(
    initial_count: int,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    additions = messages[initial_count:]
    tool_assistants = [
        message
        for message in additions
        if message.get("role") == "assistant" and _tool_calls(message)
    ]
    if tool_assistants:
        return deepcopy(tool_assistants[0])
    assistants = [message for message in additions if message.get("role") == "assistant"]
    if assistants:
        return deepcopy(assistants[0])
    return {"role": "assistant", "content": ""}


def _register_agent_model(config_path: Path) -> None:
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        os.environ.setdefault("OPENAI_API_KEY", deepseek_key)

    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_response import (
        OpenAIResponsesHandler,
    )
    from bfcl_eval.model_handler.base_handler import BaseHandler

    class GitHubResearchAgentHandler(OpenAIResponsesHandler):
        """Route BFCL action selection through the project's shared agent runtime."""

        _write_lock = threading.Lock()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Avoid constructing an unused vendor client; inherited methods still
            # provide BFCL schema compilation and official call decoding.
            BaseHandler.__init__(self, *args, **kwargs)
            self.model_style = ModelStyle.OPENAI_RESPONSES
            config = resolve_config_env_vars(
                load_config(config_path),
                config_path=config_path,
            )
            configured = config.resolve_preset()
            flash_preset = ModelPresetConfig(
                label="BFCL Agent DeepSeek V4 Flash",
                provider="deepseek",
                model=MODEL_API_NAME,
                max_tokens=min(int(configured.max_tokens), 2048),
                context_window_tokens=int(configured.context_window_tokens),
                temperature=float(self.temperature),
                reasoning_effort=None,
            )
            snapshot = build_provider_snapshot(config, preset=flash_preset)
            self.runtime = runtime_from_provider_snapshot(snapshot)
            self.runner = AgentRunner()
            self.bridge = AsyncBridge()

        def write(self, result: Any, result_dir: Path, update_mode: bool = False) -> None:
            """Serialize BFCL read-modify-write updates across worker threads."""
            with self._write_lock:
                BaseHandler.write(self, result, result_dir, update_mode)

        async def _agent_step(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
        ) -> AgentStepResponse:
            registry = ToolRegistry()
            for definition in tools:
                registry.register(CaptureTool(definition))
            initial = deepcopy(messages)
            result = await self.runner.run(
                AgentRunSpec(
                    initial_messages=initial,
                    tools=registry,
                    runtime=self.runtime,
                    max_iterations=1,
                    max_tool_result_chars=8_000,
                    concurrent_tools=True,
                    provider_retry_mode="persistent",
                    finalize_on_max_iterations=False,
                    llm_timeout_s=180,
                )
            )
            if result.error:
                raise RuntimeError(f"AgentRunner failed: {result.error}")
            assistant = _assistant_from_delta(len(initial), result.messages)
            calls = _tool_calls(assistant)
            parsed: list[dict[str, str]] = []
            call_ids: list[str] = []
            for call in calls:
                function = call.get("function", {})
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name", ""))
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                parsed.append({name: arguments})
                call_ids.append(str(call.get("id", "")))
            model_responses: Any = parsed
            if not parsed:
                model_responses = str(assistant.get("content") or result.final_content or "")
            return AgentStepResponse(
                model_responses=model_responses,
                assistant_message=assistant,
                tool_call_ids=call_ids,
                reasoning_content=str(assistant.get("reasoning_content") or ""),
                input_tokens=int(result.usage.get("prompt_tokens", 0)),
                output_tokens=int(result.usage.get("completion_tokens", 0)),
            )

        @override
        def _query_FC(self, inference_data: dict[str, Any]) -> tuple[AgentStepResponse, float]:
            messages = inference_data["message"]
            tools = inference_data["tools"]
            inference_data["inference_input_log"] = {
                "message": repr(messages),
                "tools": tools,
                "agentRuntime": "AgentRunner/ContextGovernor/ToolRegistry",
            }
            started = time.monotonic()
            response = self.bridge.call(self._agent_step(messages, tools))
            return response, time.monotonic() - started

        @override
        def _parse_query_response_FC(self, api_response: AgentStepResponse) -> dict[str, Any]:
            return {
                "model_responses": api_response.model_responses,
                "model_responses_message_for_chat_history": api_response.assistant_message,
                "tool_call_ids": api_response.tool_call_ids,
                "reasoning_content": api_response.reasoning_content,
                "input_token": api_response.input_tokens,
                "output_token": api_response.output_tokens,
            }

        @override
        def _add_assistant_message_FC(
            self,
            inference_data: dict[str, Any],
            model_response_data: dict[str, Any],
        ) -> dict[str, Any]:
            inference_data["message"].append(
                deepcopy(model_response_data["model_responses_message_for_chat_history"])
            )
            return inference_data

        @override
        def _add_execution_results_FC(
            self,
            inference_data: dict[str, Any],
            execution_results: list[str],
            model_response_data: dict[str, Any],
        ) -> dict[str, Any]:
            for execution_result, tool_call_id in zip(
                execution_results,
                model_response_data["tool_call_ids"],
                strict=False,
            ):
                inference_data["message"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": execution_result,
                    }
                )
            return inference_data

    MODEL_CONFIG_MAPPING[MODEL_REGISTRY_NAME] = ModelConfig(
        model_name=MODEL_API_NAME,
        display_name="GitHubResearch_Agent + DeepSeek V4 Flash (agent-in-loop)",
        url="local://github-research-agent",
        org="Local system evaluation",
        license="Mixed",
        model_handler=GitHubResearchAgentHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )

    if not os.environ.get("SERPAPI_API_KEY"):
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import (
            WebSearchAPI,
        )
        from ddgs import DDGS

        def ddgs_search_engine_query(
            self: WebSearchAPI,
            keywords: str,
            max_results: int | None = 10,
            region: str | None = "wt-wt",
        ) -> list[dict[str, str]] | dict[str, str]:
            try:
                rows = DDGS(proxy=os.environ.get("BFCL_WEB_PROXY")).text(
                    keywords,
                    region=region or "wt-wt",
                    max_results=max_results or 10,
                )
            except Exception as exc:
                return {"error": f"DuckDuckGo search failed: {exc}"}
            results: list[dict[str, str]] = []
            for row in rows:
                item = {
                    "title": str(row.get("title", "")),
                    "href": str(row.get("href", "")),
                }
                if self.show_snippet:
                    item["body"] = str(row.get("body", ""))
                results.append(item)
            return results

        WebSearchAPI.search_engine_query = ddgs_search_engine_query


def _official_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(OFFICIAL_SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inventory(output: Path | None) -> dict[str, Any]:
    from bfcl_eval.constants.category_mapping import ALL_SCORING_CATEGORIES
    from bfcl_eval.utils import is_memory_prereq, load_dataset_entry

    categories: dict[str, dict[str, int]] = {}
    scoring_total = 0
    generation_total = 0
    for category in ALL_SCORING_CATEGORIES:
        entries = load_dataset_entry(category)
        prereqs = sum(is_memory_prereq(str(entry["id"]).rsplit("_", 1)[0]) for entry in entries)
        categories[category] = {
            "scoringEntries": len(entries) - prereqs,
            "prerequisiteEntries": prereqs,
            "generationEntries": len(entries),
        }
        scoring_total += len(entries) - prereqs
        generation_total += len(entries)
    payload = {
        "benchmark": "BFCL V4 agent-in-loop",
        "officialCommit": _official_commit(),
        "modelRegistryName": MODEL_REGISTRY_NAME,
        "modelApiName": MODEL_API_NAME,
        "agentRuntime": ["AgentRunner", "ContextGovernor", "ToolRegistry"],
        "resultIsOfficialLeaderboardComparable": False,
        "scoringCategories": list(ALL_SCORING_CATEGORIES),
        "categoryCount": len(ALL_SCORING_CATEGORIES),
        "scoringEntryCount": scoring_total,
        "generationEntryCountIncludingPrerequisites": generation_total,
        "categories": categories,
    }
    if output is not None:
        _write_json(output, payload)
    return payload


def generate(args: argparse.Namespace) -> None:
    from bfcl_eval._llm_response_generation import main as generation_main

    generation_main(
        SimpleNamespace(
            model=[MODEL_REGISTRY_NAME],
            test_category=args.categories,
            temperature=args.temperature,
            include_input_log=args.include_input_log,
            exclude_state_log=args.exclude_state_log,
            num_gpus=1,
            num_threads=args.threads,
            gpu_memory_utilization=0.0,
            backend="vllm",
            skip_server_setup=True,
            local_model_path=None,
            result_dir=str(args.result_dir.resolve()),
            allow_overwrite=args.overwrite,
            run_ids=args.run_ids,
            enable_lora=False,
            max_lora_rank=None,
            lora_modules=None,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    evaluation_main(
        [MODEL_REGISTRY_NAME],
        args.categories,
        str(args.result_dir.resolve()),
        str(args.score_dir.resolve()),
        args.partial,
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from bfcl_eval.constants.category_mapping import ALL_SCORING_CATEGORIES
    from bfcl_eval.constants.eval_config import RESULT_FILE_PATTERN
    from bfcl_eval.utils import extract_test_category, is_memory_prereq, load_dataset_entry

    model_root = args.result_dir.resolve() / MODEL_REGISTRY_NAME
    result_files = sorted(model_root.rglob(RESULT_FILE_PATTERN)) if model_root.exists() else []
    actual_by_category: dict[str, list[dict[str, Any]]] = {}
    duplicate_ids: list[str] = []
    error_ids: list[str] = []
    error_ids_by_category: dict[str, list[str]] = {}
    seen_global: set[str] = set()
    for result_file in result_files:
        category = extract_test_category(result_file)
        rows = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines() if line]
        actual_by_category.setdefault(category, []).extend(rows)
        for row in rows:
            row_id = str(row["id"])
            if row_id in seen_global:
                duplicate_ids.append(row_id)
            seen_global.add(row_id)
            rendered_result = json.dumps(row.get("result"), ensure_ascii=False)
            if (
                "Error during inference:" in rendered_result
                or AGENT_ERROR_PLACEHOLDER in rendered_result
            ):
                error_ids.append(row_id)
                rerun_category = category.removesuffix("_prereq")
                error_ids_by_category.setdefault(rerun_category, []).append(row_id)
    categories: dict[str, dict[str, int]] = {}
    missing_total = 0
    extra_total = 0
    for category in ALL_SCORING_CATEGORIES:
        expected = {
            str(entry["id"])
            for entry in load_dataset_entry(category)
            if not is_memory_prereq(str(entry["id"]).rsplit("_", 1)[0])
        }
        actual = {
            str(row["id"])
            for row in actual_by_category.get(category, [])
            if not is_memory_prereq(str(row["id"]).rsplit("_", 1)[0])
        }
        missing = expected - actual
        extra = actual - expected
        missing_total += len(missing)
        extra_total += len(extra)
        categories[category] = {
            "expected": len(expected),
            "actual": len(actual),
            "missingCount": len(missing),
            "extraCount": len(extra),
        }
    payload = {
        "benchmark": "BFCL V4 agent-in-loop",
        "officialCommit": _official_commit(),
        "model": MODEL_REGISTRY_NAME,
        "complete": missing_total == 0 and extra_total == 0 and not duplicate_ids and not error_ids,
        "missingCount": missing_total,
        "extraCount": extra_total,
        "duplicateCount": len(duplicate_ids),
        "inferenceErrorCount": len(error_ids),
        "duplicateIds": sorted(set(duplicate_ids)),
        "inferenceErrorIds": sorted(set(error_ids)),
        "inferenceErrorIdsByCategory": {
            category: sorted(set(ids))
            for category, ids in sorted(error_ids_by_category.items())
        },
        "categories": categories,
    }
    _write_json(args.output, payload)
    if args.rerun_ids_output is not None:
        _write_json(args.rerun_ids_output, payload["inferenceErrorIdsByCategory"])
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--output", type=Path)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--categories", nargs="+", default=["all_scoring"])
    generate_parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    generate_parser.add_argument("--threads", type=int, default=4)
    generate_parser.add_argument("--temperature", type=float, default=0.001)
    generate_parser.add_argument("--include-input-log", action="store_true")
    generate_parser.add_argument("--exclude-state-log", action="store_true")
    generate_parser.add_argument("--overwrite", action="store_true")
    generate_parser.add_argument("--run-ids", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--categories", nargs="+", default=["all_scoring"])
    evaluate_parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    evaluate_parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    evaluate_parser.add_argument("--partial", action="store_true")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT)
    audit_parser.add_argument("--rerun-ids-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    _register_agent_model(args.config)
    if args.command == "inventory":
        print(json.dumps(inventory(args.output), ensure_ascii=False, indent=2))
    elif args.command == "generate":
        generate(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "audit":
        print(json.dumps(audit(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
