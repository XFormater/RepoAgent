"""Run the four-layer model-in-the-loop system benchmark.

Unlike the deterministic contract suite, this benchmark always sends the fixed
tasks through the configured real model.  Context and memory cases are matched
on/off experiments; recovery cases combine a real model with deterministic fault
injection.  Every run writes a manifest, JSONL event stream, per-case artifacts,
paired comparisons, and an aggregate report under one run directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import re
import subprocess
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nanobot.agent.context_governance import ContextGovernor
from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec
from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.memory_system.config import MemorySettings
from nanobot.memory_system.models import FactType, MemoryRecord
from nanobot.memory_system.service import build_memory_service
from nanobot.providers.factory import load_provider_snapshot
from nanobot.utils.helpers import estimate_message_tokens
from nanobot.utils.llm_runtime import LLMRuntime, runtime_from_provider_snapshot

LAYERS = (
    "harness_regression",
    "context_governance",
    "memory_benefit",
    "recovery_correctness",
)


def load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite cases must be a non-empty list")
    seen: set[str] = set()
    covered: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        for field in ("caseId", "layer", "scenario", "prompt", "expected"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"case {index} has invalid {field}")
        if case["caseId"] in seen:
            raise ValueError(f"duplicate caseId: {case['caseId']}")
        if case["layer"] not in LAYERS:
            raise ValueError(f"unknown layer: {case['layer']}")
        seen.add(case["caseId"])
        covered.add(case["layer"])
    required_layers = set(payload.get("requiredLayers", LAYERS))
    unknown_required = required_layers - set(LAYERS)
    if unknown_required:
        raise ValueError(f"unknown required layers: {sorted(unknown_required)}")
    missing = required_layers - covered
    if missing:
        raise ValueError(f"layer coverage mismatch: missing={sorted(missing)}")
    return payload


def normalize_answer(value: str | None) -> str:
    text = (value or "").strip()
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:答案|answer)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"`":
        text = text[1:-1].strip()
    return text


def answer_matches(actual: str | None, expected: str) -> bool:
    return normalize_answer(actual).casefold() == expected.strip().casefold()


def token_total(result: AgentRunResult, key: str) -> int:
    value = result.usage.get(key, 0)
    return int(value) if isinstance(value, int) else 0


def git_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


class ArtifactWriter:
    def __init__(self, root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.path = root / run_id
        self.case_path = self.path / "cases"
        self.case_path.mkdir(parents=True, exist_ok=False)
        self.events_path = self.path / "events.jsonl"
        self.event_count = 0

    def write(self, name: str, payload: Any) -> None:
        (self.path / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def event(self, event: str, **values: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "runId": self.run_id,
            "event": event,
            **values,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.event_count += 1

    def case(self, result: dict[str, Any]) -> None:
        repetition = int(result.get("repetition", 1))
        name = f"{result['caseId']}__r{repetition}.json"
        (self.case_path / name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


@tool_parameters(
    {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    }
)
class LookupTool(Tool):
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.keys: list[str] = []

    @property
    def name(self) -> str:
        return "lookup_value"

    @property
    def description(self) -> str:
        return "Look up one opaque benchmark value by its exact key."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, key: str) -> str:
        self.keys.append(key)
        return self.values.get(key, "NOT_FOUND")


@tool_parameters(
    {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["key", "value"],
        "additionalProperties": False,
    }
)
class WriteStateTool(Tool):
    def __init__(self, state: dict[str, str], calls: list[str]) -> None:
        self.state = state
        self.calls = calls

    @property
    def name(self) -> str:
        return "write_state"

    @property
    def description(self) -> str:
        return "Write an exact string value to temporary benchmark state."

    async def execute(self, key: str, value: str) -> str:
        self.calls.append(self.name)
        self.state[key] = value
        return "WROTE"


@tool_parameters(
    {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    }
)
class ReadStateTool(Tool):
    def __init__(self, state: dict[str, str], calls: list[str]) -> None:
        self.state = state
        self.calls = calls

    @property
    def name(self) -> str:
        return "read_state"

    @property
    def description(self) -> str:
        return "Read an exact string value from temporary benchmark state."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, key: str) -> str:
        self.calls.append(self.name)
        return self.state.get(key, "NOT_FOUND")


@tool_parameters(
    {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    }
)
class FlakyLookupTool(Tool):
    def __init__(self, expected: str) -> None:
        self.expected = expected
        self.attempts = 0

    @property
    def name(self) -> str:
        return "flaky_lookup"

    @property
    def description(self) -> str:
        return "A lookup that may explicitly report a transient failure; retry only when asked."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, key: str) -> str:
        _ = key
        self.attempts += 1
        if self.attempts == 1:
            return ToolResult.error("TRANSIENT_FAILURE: retry this exact lookup once")
        return self.expected


@dataclass
class CommitState:
    attempts: int = 0
    side_effects: int = 0
    committed: bool = False


@tool_parameters(
    {
        "type": "object",
        "properties": {"operation_id": {"type": "string"}},
        "required": ["operation_id"],
        "additionalProperties": False,
    }
)
class CommitOnceTool(Tool):
    def __init__(self, state: CommitState) -> None:
        self.state = state

    @property
    def name(self) -> str:
        return "commit_once"

    @property
    def description(self) -> str:
        return "Commit an operation once. ACK_LOST means inspect status instead of resubmitting."

    async def execute(self, operation_id: str) -> str:
        _ = operation_id
        self.state.attempts += 1
        if not self.state.committed:
            self.state.committed = True
            self.state.side_effects += 1
            return ToolResult.error("ACK_LOST: effect may be committed; call inspect_commit")
        return "ALREADY_COMMITTED"


@tool_parameters(
    {
        "type": "object",
        "properties": {"operation_id": {"type": "string"}},
        "required": ["operation_id"],
        "additionalProperties": False,
    }
)
class InspectCommitTool(Tool):
    def __init__(self, state: CommitState) -> None:
        self.state = state
        self.attempts = 0

    @property
    def name(self) -> str:
        return "inspect_commit"

    @property
    def description(self) -> str:
        return "Inspect whether an operation is already committed without changing it."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, operation_id: str) -> str:
        _ = operation_id
        self.attempts += 1
        return "COMMITTED-77" if self.state.committed else "NOT_COMMITTED"


@tool_parameters(
    {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    }
)
class CheckpointLookupTool(Tool):
    def __init__(self, expected: str) -> None:
        self.expected = expected
        self.calls = 0

    @property
    def name(self) -> str:
        return "checkpoint_lookup"

    @property
    def description(self) -> str:
        return "Read an opaque value whose completed result is checkpointed before continuation."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, key: str) -> str:
        _ = key
        self.calls += 1
        return self.expected


class CapturingGovernor:
    def __init__(self, delegate: ContextGovernor | None) -> None:
        self.delegate = delegate
        self.last_messages: list[dict[str, Any]] = []

    def prepare_for_model(
        self,
        config: Any,
        messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str],
    ) -> list[dict[str, Any]]:
        prepared = (
            messages
            if self.delegate is None
            else self.delegate.prepare_for_model(config, messages, compacted_tool_call_ids)
        )
        self.last_messages = [dict(message) for message in prepared]
        return prepared


def registry_with(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def base_messages(prompt: str, *, extra_system: str = "") -> list[dict[str, Any]]:
    system = (
        "You are executing a deterministic system benchmark. Follow the requested tool "
        "sequence exactly. Never invent opaque values. Your final response must contain "
        "only the requested value, with no prose, Markdown, or punctuation."
    )
    if extra_system:
        system += "\n\n" + extra_system
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


async def agent_run(
    runtime: LLMRuntime,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    *,
    runner: AgentRunner | None = None,
    max_iterations: int = 6,
    context_block_limit: int | None = None,
    checkpoint_callback: Any = None,
) -> AgentRunResult:
    return await (runner or AgentRunner()).run(
        AgentRunSpec(
            initial_messages=messages,
            tools=tools,
            runtime=runtime,
            max_iterations=max_iterations,
            max_tool_result_chars=8_000,
            context_block_limit=context_block_limit,
            provider_retry_mode="standard",
            finalize_on_max_iterations=True,
            llm_timeout_s=180,
            checkpoint_callback=checkpoint_callback,
        )
    )


async def run_harness_case(case: dict[str, Any], runtime: LLMRuntime) -> dict[str, Any]:
    scenario = case["scenario"]
    if scenario == "lookup":
        tool = LookupTool(dict(case["values"]))
        result = await agent_run(runtime, base_messages(case["prompt"]), registry_with(tool))
        actual_calls = [event["name"] for event in result.tool_events]
        required_keys = case.get("requiredLookupKeys")
        keys_ok = required_keys is None or tool.keys == required_keys
    elif scenario == "state_roundtrip":
        state: dict[str, str] = {}
        calls: list[str] = []
        result = await agent_run(
            runtime,
            base_messages(case["prompt"]),
            registry_with(WriteStateTool(state, calls), ReadStateTool(state, calls)),
        )
        actual_calls = calls
        keys_ok = state.get("deployment_token") == case["expected"]
    else:
        raise ValueError(f"unsupported harness scenario: {scenario}")
    required_calls = list(case.get("requiredCalls", []))
    passed = (
        answer_matches(result.final_content, case["expected"])
        and actual_calls == required_calls
        and keys_ok
        and result.stop_reason == "completed"
    )
    return {
        "passed": passed,
        "answer": result.final_content,
        "normalizedAnswer": normalize_answer(result.final_content),
        "expected": case["expected"],
        "stopReason": result.stop_reason,
        "toolCalls": actual_calls,
        "requiredCalls": required_calls,
        "lookupKeysValid": keys_ok,
        "usage": result.usage,
    }


def _context_noise(count: int, *, offset: int = 0) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    filler = "Archived diagnostics without authority for the current question. " * 10
    for index in range(offset, offset + count):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Archived turn {index}: OLD-{index:03d}. {filler}",
                },
                {
                    "role": "assistant",
                    "content": f"Archived acknowledgement {index}. {filler}",
                },
            ]
        )
    return messages


def context_history(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one identical, information-graded history for both A/B variants."""
    case_type = str(case.get("contextCaseType", "latest_authority"))
    system = (
        "Use only information present in this conversation. The latest explicit user "
        "statement overrides older conflicts. If any requested value is unavailable, "
        "output only UNKNOWN. Otherwise output only the requested value."
    )
    if case_type == "system_anchor":
        system += "\n\nDurable session summary: " + str(case["fact"])
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    noise_turns = int(case.get("noiseTurns", 0))

    if case_type in {"latest_authority", "conflict_override"}:
        if case_type == "conflict_override":
            messages.append({"role": "user", "content": str(case["oldFact"])})
            messages.append({"role": "assistant", "content": "Recorded the earlier value."})
        messages.extend(_context_noise(noise_turns))
    elif case_type == "recent_dependency":
        messages.extend(_context_noise(noise_turns))
        messages.append({"role": "user", "content": str(case["fact"])})
        messages.append({"role": "assistant", "content": "Recorded for the next question."})
        messages.extend(_context_noise(int(case.get("trailingNoiseTurns", 1)), offset=900))
    elif case_type == "early_dependency":
        messages.append({"role": "user", "content": str(case["fact"])})
        messages.append({"role": "assistant", "content": "Recorded for later."})
        messages.extend(_context_noise(noise_turns))
    elif case_type == "system_anchor":
        messages.extend(_context_noise(noise_turns))
    elif case_type == "split_dependency":
        messages.append({"role": "user", "content": str(case["earlyFact"])})
        messages.append({"role": "assistant", "content": "Recorded the first component."})
        messages.extend(_context_noise(noise_turns))
        messages.append({"role": "user", "content": str(case["recentFact"])})
        messages.append({"role": "assistant", "content": "Recorded the second component."})
    elif case_type in {"recent_tool_result", "early_tool_result"}:
        tool_exchange = [
            {"role": "user", "content": "Look up the opaque value for later use."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{case['caseId']}",
                        "type": "function",
                        "function": {
                            "name": "historical_lookup",
                            "arguments": json.dumps({"key": "opaque"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": f"call-{case['caseId']}",
                "name": "historical_lookup",
                "content": str(case["toolResult"]),
            },
            {"role": "assistant", "content": "The lookup result is available."},
        ]
        if case_type == "early_tool_result":
            messages.extend(tool_exchange)
            messages.extend(_context_noise(noise_turns))
        else:
            messages.extend(_context_noise(noise_turns))
            messages.extend(tool_exchange)
    else:
        raise ValueError(f"unsupported contextCaseType: {case_type}")
    messages.append({"role": "user", "content": case["prompt"]})
    return messages


async def context_variant(
    case: dict[str, Any],
    runtime: LLMRuntime,
    *,
    enabled: bool,
    source_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    governor = CapturingGovernor(ContextGovernor() if enabled else None)
    runner = AgentRunner()
    runner.context_governor = governor  # type: ignore[assignment]
    result = await agent_run(
        runtime,
        deepcopy(source_messages),
        ToolRegistry(),
        runner=runner,
        max_iterations=1,
        context_block_limit=int(case["contextBlockLimit"]),
    )
    estimated_tokens = sum(estimate_message_tokens(message) for message in governor.last_messages)
    provider_prompt_tokens = token_total(result, "prompt_tokens")
    return {
        "variant": "governance_on" if enabled else "governance_off",
        "correct": answer_matches(result.final_content, case["expected"]),
        "answer": result.final_content,
        "normalizedAnswer": normalize_answer(result.final_content),
        "stopReason": result.stop_reason,
        "usage": result.usage,
        "modelInputMessages": len(governor.last_messages),
        "estimatedInputTokens": estimated_tokens,
        "providerPromptTokens": provider_prompt_tokens,
    }


async def run_context_case(case: dict[str, Any], runtime: LLMRuntime) -> dict[str, Any]:
    source_messages = context_history(case)
    source_hash = hashlib.sha256(
        json.dumps(source_messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    treatment_first = int(hashlib.sha256(case["caseId"].encode()).hexdigest()[-1], 16) % 2 == 0
    variant_order = (
        ["governance_on", "governance_off"]
        if treatment_first
        else ["governance_off", "governance_on"]
    )
    variants: dict[str, dict[str, Any]] = {}
    for name in variant_order:
        enabled = name == "governance_on"
        variants[name] = await context_variant(
            case,
            runtime,
            enabled=enabled,
            source_messages=source_messages,
        )
    control = variants["governance_off"]
    treatment = variants["governance_on"]
    control_tokens = control["estimatedInputTokens"]
    treatment_tokens = treatment["estimatedInputTokens"]
    reduction = 1.0 - (treatment_tokens / control_tokens) if control_tokens else 0.0
    control_provider_tokens = control["providerPromptTokens"]
    treatment_provider_tokens = treatment["providerPromptTokens"]
    provider_reduction = (
        1.0 - (treatment_provider_tokens / control_provider_tokens)
        if control_provider_tokens
        else 0.0
    )
    passed = bool(
        treatment["correct"]
        and int(treatment["correct"]) >= int(control["correct"])
        and reduction > 0
    )
    return {
        "passed": passed,
        "expected": case["expected"],
        "contextCaseType": str(case.get("contextCaseType", "latest_authority")),
        "contextBlockLimit": int(case["contextBlockLimit"]),
        "sourceInputSha256": source_hash,
        "sourceInputMessages": len(source_messages),
        "variantOrder": variant_order,
        "control": control,
        "treatment": treatment,
        "qualityDelta": int(treatment["correct"]) - int(control["correct"]),
        "estimatedTokenReduction": reduction,
        "providerPromptTokenReduction": provider_reduction,
    }


async def memory_prompt(case: dict[str, Any], runtime: LLMRuntime) -> tuple[str, bool]:
    settings = MemorySettings.model_validate(
        {
            "enabled": True,
            "record": {"idempotency": {"store": "memory"}},
            "retrieve": {
                "minConfidence": 0.0,
                "maxInjectedChars": 4000,
                "vector": {"topK": 20, "scoreThreshold": -1.0},
                "graph": {"enabled": False},
                "reranker": {"model": "identity", "topN": 10},
                "cache": {"warmCacheEnabled": False},
            },
            "storage": {
                "vectorDb": {"type": "memory"},
                "graphDb": {"type": "memory"},
                "metadataDb": {"type": "memory"},
                "embedding": {"provider": "hash", "dimension": 64},
            },
        }
    )
    service = build_memory_service(
        settings,
        provider=runtime.provider,
        default_model=runtime.model,
    )
    try:
        texts = list(case["memories"])
        embeddings = await service.retrieve_pipeline.embedding.embed(texts)
        records = [
            MemoryRecord(
                user_id=case["caseId"],
                fact_text=text,
                fact_type=FactType.FACT,
                entities=[],
                confidence=1.0,
                source_message_ids=[f"source-{index}"],
                batch_id=case["caseId"],
                idempotency_key=f"{case['caseId']}-{index}",
                embedding=embedding,
            )
            for index, (text, embedding) in enumerate(zip(texts, embeddings, strict=True))
        ]
        await service.record_pipeline.metadata_store.insert_many(records)
        await service.record_pipeline.vector_store.upsert(records)
        prompt = await service.retrieve_prompt(
            user_id=case["caseId"],
            query=case["memoryQuery"],
            recent_messages=[],
            deep=False,
        )
        raw_terms = case.get("retrievalTerms")
        retrieval_terms = (
            [str(term) for term in raw_terms]
            if isinstance(raw_terms, list)
            else [str(case.get("retrievalTerm", case["expected"]))]
        )
        return prompt, bool(prompt) and all(term in prompt for term in retrieval_terms)
    finally:
        await service.close()


async def memory_variant(
    case: dict[str, Any], runtime: LLMRuntime, *, injected: str
) -> dict[str, Any]:
    result = await agent_run(
        runtime,
        base_messages(case["prompt"], extra_system=injected),
        ToolRegistry(),
        max_iterations=1,
    )
    return {
        "variant": "memory_on" if injected else "memory_off",
        "correct": answer_matches(result.final_content, case["expected"]),
        "answer": result.final_content,
        "normalizedAnswer": normalize_answer(result.final_content),
        "stopReason": result.stop_reason,
        "usage": result.usage,
        "injectedChars": len(injected),
    }


async def run_memory_case(case: dict[str, Any], runtime: LLMRuntime) -> dict[str, Any]:
    injected, retrieval_hit = await memory_prompt(case, runtime)
    control = await memory_variant(case, runtime, injected="")
    treatment = await memory_variant(case, runtime, injected=injected)
    quality_delta = int(treatment["correct"]) - int(control["correct"])
    case_type = str(case.get("memoryCaseType", "benefit_required"))
    if case_type == "benefit_required":
        passed = retrieval_hit and treatment["correct"] and quality_delta > 0
    elif case_type in {"conflict_override", "irrelevant_non_regression"}:
        passed = retrieval_hit and treatment["correct"] and quality_delta >= 0
    else:
        raise ValueError(f"unsupported memoryCaseType: {case_type}")
    return {
        "passed": bool(passed),
        "expected": case["expected"],
        "memoryCaseType": case_type,
        "retrievalHit": retrieval_hit,
        "control": control,
        "treatment": treatment,
        "qualityDelta": quality_delta,
        "injectedMemory": injected,
    }


class InjectedCrashError(RuntimeError):
    pass


async def run_recovery_case(case: dict[str, Any], runtime: LLMRuntime) -> dict[str, Any]:
    scenario = case["scenario"]
    if scenario == "transient_tool":
        tool = FlakyLookupTool(case["expected"])
        result = await agent_run(
            runtime,
            base_messages(case["prompt"]),
            registry_with(tool),
            max_iterations=5,
        )
        passed = answer_matches(result.final_content, case["expected"]) and tool.attempts == 2
        details = {"attempts": tool.attempts, "duplicateSideEffects": 0}
    elif scenario == "ack_loss":
        state = CommitState()
        inspect = InspectCommitTool(state)
        result = await agent_run(
            runtime,
            base_messages(case["prompt"]),
            registry_with(CommitOnceTool(state), inspect),
            max_iterations=5,
        )
        passed = bool(
            answer_matches(result.final_content, case["expected"])
            and state.side_effects == 1
            and state.attempts == 1
            and inspect.attempts >= 1
        )
        details = {
            "commitAttempts": state.attempts,
            "inspectAttempts": inspect.attempts,
            "sideEffects": state.side_effects,
            "duplicateSideEffects": max(0, state.side_effects - 1),
        }
    elif scenario == "checkpoint_resume":
        tool = CheckpointLookupTool(case["expected"])
        registry = registry_with(tool)
        initial = base_messages(case["prompt"])
        checkpoint: dict[str, Any] = {}
        crash_observed = False

        async def crash_after_tools(payload: dict[str, Any]) -> None:
            if payload.get("phase") != "tools_completed":
                return
            checkpoint.update(payload)
            raise InjectedCrashError("controlled crash after durable tool-result boundary")

        try:
            await agent_run(
                runtime,
                initial,
                registry,
                max_iterations=4,
                checkpoint_callback=crash_after_tools,
            )
        except InjectedCrashError:
            crash_observed = True
        resume_messages = [
            *initial,
            dict(checkpoint["assistant_message"]),
            *[dict(item) for item in checkpoint["completed_tool_results"]],
        ]
        result = await agent_run(runtime, resume_messages, registry, max_iterations=3)
        passed = bool(
            crash_observed
            and answer_matches(result.final_content, case["expected"])
            and tool.calls == 1
        )
        details = {
            "crashObserved": crash_observed,
            "checkpointPhase": checkpoint.get("phase"),
            "toolExecutions": tool.calls,
            "duplicateSideEffects": max(0, tool.calls - 1),
        }
    else:
        raise ValueError(f"unsupported recovery scenario: {scenario}")
    return {
        "passed": passed,
        "answer": result.final_content,
        "normalizedAnswer": normalize_answer(result.final_content),
        "expected": case["expected"],
        "stopReason": result.stop_reason,
        "usage": result.usage,
        **details,
    }


async def run_case(case: dict[str, Any], runtime: LLMRuntime) -> dict[str, Any]:
    layer = case["layer"]
    if layer == "harness_regression":
        return await run_harness_case(case, runtime)
    if layer == "context_governance":
        return await run_context_case(case, runtime)
    if layer == "memory_benefit":
        return await run_memory_case(case, runtime)
    if layer == "recovery_correctness":
        return await run_recovery_case(case, runtime)
    raise ValueError(f"unsupported layer: {layer}")


def aggregate(
    suite: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    model: str,
    run_id: str,
    artifact_path: Path,
    event_count: int,
) -> dict[str, Any]:
    thresholds = suite.get("thresholds", {})
    min_layer = float(thresholds.get("minLayerPassRate", 1.0))
    layer_scores: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for layer in LAYERS:
        rows = [row for row in results if row["layer"] == layer]
        passed = sum(bool(row.get("passed")) for row in rows)
        rate = passed / len(rows) if rows else None
        layer_scores[layer] = {"passed": passed, "total": len(rows), "passRate": rate}
        if rate is not None and rate < min_layer:
            failures.append(f"{layer}.passRate {rate:.4f} < {min_layer:.4f}")

    context_rows = [row for row in results if row["layer"] == "context_governance"]
    context_estimated_reduction = (
        sum(float(row.get("estimatedTokenReduction", 0.0)) for row in context_rows)
        / len(context_rows)
        if context_rows
        else 0.0
    )
    context_reduction = (
        sum(
            float(
                row.get(
                    "providerPromptTokenReduction",
                    row.get("estimatedTokenReduction", 0.0),
                )
            )
            for row in context_rows
        )
        / len(context_rows)
        if context_rows
        else 0.0
    )
    context_control_tokens = sum(
        int(row.get("control", {}).get("providerPromptTokens", 0) or 0)
        for row in context_rows
    )
    context_treatment_tokens = sum(
        int(row.get("treatment", {}).get("providerPromptTokens", 0) or 0)
        for row in context_rows
    )
    context_weighted_reduction = (
        1.0 - (context_treatment_tokens / context_control_tokens)
        if context_control_tokens
        else 0.0
    )
    min_context = float(thresholds.get("minContextTokenReduction", 0.0))
    if context_rows and context_reduction < min_context:
        failures.append(
            f"context.tokenReduction {context_reduction:.4f} < {min_context:.4f}"
        )

    context_slices: dict[str, dict[str, Any]] = {}
    for case_type in sorted(
        {str(row.get("contextCaseType", "unspecified")) for row in context_rows}
    ):
        rows = [
            row
            for row in context_rows
            if row.get("contextCaseType", "unspecified") == case_type
        ]
        control_correct = sum(bool(row.get("control", {}).get("correct")) for row in rows)
        treatment_correct = sum(
            bool(row.get("treatment", {}).get("correct")) for row in rows
        )
        harmful = sum(
            bool(row.get("control", {}).get("correct"))
            and not bool(row.get("treatment", {}).get("correct"))
            for row in rows
        )
        context_slices[case_type] = {
            "cases": len(rows),
            "controlAccuracy": control_correct / len(rows),
            "treatmentAccuracy": treatment_correct / len(rows),
            "accuracyDelta": (treatment_correct - control_correct) / len(rows),
            "harmfulRegressionRate": harmful / len(rows),
            "meanProviderPromptTokenReduction": sum(
                float(
                    row.get(
                        "providerPromptTokenReduction",
                        row.get("estimatedTokenReduction", 0.0),
                    )
                )
                for row in rows
            )
            / len(rows),
        }

    memory_rows = [row for row in results if row["layer"] == "memory_benefit"]
    control_correct = sum(bool(row.get("control", {}).get("correct")) for row in memory_rows)
    treatment_correct = sum(bool(row.get("treatment", {}).get("correct")) for row in memory_rows)
    memory_gain = (
        (treatment_correct - control_correct) / len(memory_rows) if memory_rows else 0.0
    )
    control_prompt_tokens = sum(
        int(row.get("control", {}).get("usage", {}).get("prompt_tokens", 0) or 0)
        for row in memory_rows
    )
    treatment_prompt_tokens = sum(
        int(row.get("treatment", {}).get("usage", {}).get("prompt_tokens", 0) or 0)
        for row in memory_rows
    )
    mean_control_prompt_tokens = (
        control_prompt_tokens / len(memory_rows) if memory_rows else 0.0
    )
    mean_treatment_prompt_tokens = (
        treatment_prompt_tokens / len(memory_rows) if memory_rows else 0.0
    )
    prompt_token_overhead = mean_treatment_prompt_tokens - mean_control_prompt_tokens
    min_memory = float(thresholds.get("minMemoryAccuracyGain", 0.0))
    if memory_rows and memory_gain < min_memory:
        failures.append(f"memory.accuracyGain {memory_gain:.4f} < {min_memory:.4f}")

    memory_slices: dict[str, dict[str, Any]] = {}
    for case_type in sorted({str(row.get("memoryCaseType", "unspecified")) for row in memory_rows}):
        rows = [row for row in memory_rows if row.get("memoryCaseType", "unspecified") == case_type]
        slice_control = sum(bool(row.get("control", {}).get("correct")) for row in rows)
        slice_treatment = sum(bool(row.get("treatment", {}).get("correct")) for row in rows)
        harmful = sum(
            bool(row.get("control", {}).get("correct"))
            and not bool(row.get("treatment", {}).get("correct"))
            for row in rows
        )
        memory_slices[case_type] = {
            "cases": len(rows),
            "controlAccuracy": slice_control / len(rows),
            "treatmentAccuracy": slice_treatment / len(rows),
            "accuracyGain": (slice_treatment - slice_control) / len(rows),
            "harmfulRegressionRate": harmful / len(rows),
        }

    comparisons = {
        "contextGovernance": {
            "pairedCases": len(context_rows),
            "controlAccuracy": (
                sum(bool(row["control"]["correct"]) for row in context_rows)
                / len(context_rows)
                if context_rows
                else 0.0
            ),
            "treatmentAccuracy": (
                sum(bool(row["treatment"]["correct"]) for row in context_rows)
                / len(context_rows)
                if context_rows
                else 0.0
            ),
            "meanProviderPromptTokenReduction": context_reduction,
            "meanEstimatedTokenReduction": context_estimated_reduction,
            "totalControlPromptTokens": context_control_tokens,
            "totalTreatmentPromptTokens": context_treatment_tokens,
            "meanControlPromptTokens": (
                context_control_tokens / len(context_rows) if context_rows else 0.0
            ),
            "meanTreatmentPromptTokens": (
                context_treatment_tokens / len(context_rows) if context_rows else 0.0
            ),
            "weightedProviderPromptTokenReduction": context_weighted_reduction,
            "harmfulRegressionRate": (
                sum(
                    bool(row.get("control", {}).get("correct"))
                    and not bool(row.get("treatment", {}).get("correct"))
                    for row in context_rows
                )
                / len(context_rows)
                if context_rows
                else 0.0
            ),
            "slices": context_slices,
        },
        "memoryBenefit": {
            "pairedCases": len(memory_rows),
            "controlAccuracy": control_correct / len(memory_rows) if memory_rows else 0.0,
            "treatmentAccuracy": treatment_correct / len(memory_rows) if memory_rows else 0.0,
            "accuracyGain": memory_gain,
            "meanControlPromptTokens": mean_control_prompt_tokens,
            "meanTreatmentPromptTokens": mean_treatment_prompt_tokens,
            "meanPromptTokenOverhead": prompt_token_overhead,
            "relativePromptTokenOverhead": (
                prompt_token_overhead / mean_control_prompt_tokens
                if mean_control_prompt_tokens
                else 0.0
            ),
            "meanInjectedChars": (
                sum(int(row.get("treatment", {}).get("injectedChars", 0)) for row in memory_rows)
                / len(memory_rows)
                if memory_rows
                else 0.0
            ),
            "retrievalHitRate": (
                sum(bool(row.get("retrievalHit")) for row in memory_rows) / len(memory_rows)
                if memory_rows
                else 0.0
            ),
            "harmfulRegressionRate": (
                sum(
                    bool(row.get("control", {}).get("correct"))
                    and not bool(row.get("treatment", {}).get("correct"))
                    for row in memory_rows
                )
                / len(memory_rows)
                if memory_rows
                else 0.0
            ),
            "slices": memory_slices,
        },
    }
    total_passed = sum(bool(row.get("passed")) for row in results)
    return {
        "benchmark": suite["benchmark"],
        "version": suite["version"],
        "runId": run_id,
        "model": model,
        "executionMode": "online-model-in-the-loop",
        "passed": not failures,
        "caseRuns": len(results),
        "passedCaseRuns": total_passed,
        "passRate": total_passed / len(results) if results else 0.0,
        "layerScores": layer_scores,
        "evaluationPaths": {
            "fixedBenchmark": {"caseRuns": len(results), "layerScores": layer_scores},
            "controlledExperiments": comparisons,
            "artifactAggregation": {
                "runDirectory": str(artifact_path),
                "eventCount": event_count,
                "perCaseArtifacts": len(results),
            },
        },
        "failures": failures,
        "results": results,
    }


async def main_async(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    suite_path = args.suite if args.suite.is_absolute() else root / args.suite
    suite = load_suite(suite_path)
    selected = [
        case
        for case in suite["cases"]
        if (args.layer is None or case["layer"] == args.layer)
        and (args.case is None or case["caseId"] == args.case)
    ]
    if not selected:
        raise ValueError("no benchmark cases selected")

    snapshot = load_provider_snapshot(args.config)
    runtime = runtime_from_provider_snapshot(snapshot).with_generation_overrides(
        temperature=0.0,
        max_tokens=256,
        reasoning_effort="none",
    )
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    writer = ArtifactWriter(output_root, run_id)
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    manifest = {
        "benchmark": suite["benchmark"],
        "version": suite["version"],
        "runId": run_id,
        "startedAt": datetime.now(UTC).isoformat(),
        "executionMode": "online-model-in-the-loop",
        "model": snapshot.model,
        "providerType": type(snapshot.provider).__name__,
        "contextWindowTokens": snapshot.context_window_tokens,
        "generation": {
            "temperature": runtime.generation.temperature,
            "maxTokens": runtime.generation.max_tokens,
            "reasoningEffort": runtime.generation.reasoning_effort,
        },
        "suite": str(suite_path),
        "suiteSha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "configSha256": config_hash,
        "gitCommit": git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repetitions": args.repetitions,
        "selectedLayers": sorted({case["layer"] for case in selected}),
        "selectedCases": [case["caseId"] for case in selected],
    }
    writer.write("manifest.json", manifest)
    writer.event("run_started", model=snapshot.model, cases=len(selected))

    results: list[dict[str, Any]] = []
    total = len(selected) * args.repetitions
    position = 0
    for repetition in range(1, args.repetitions + 1):
        for case in selected:
            position += 1
            started = time.monotonic()
            writer.event(
                "case_started",
                caseId=case["caseId"],
                layer=case["layer"],
                repetition=repetition,
            )
            error = None
            try:
                outcome = await run_case(case, runtime)
            except Exception as exc:
                outcome = {"passed": False}
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "caseId": case["caseId"],
                "layer": case["layer"],
                "scenario": case["scenario"],
                "repetition": repetition,
                "latencyMs": round((time.monotonic() - started) * 1000),
                "error": error,
                **outcome,
            }
            results.append(record)
            writer.case(record)
            writer.event(
                "case_completed",
                caseId=case["caseId"],
                layer=case["layer"],
                repetition=repetition,
                passed=bool(record["passed"]),
                latencyMs=record["latencyMs"],
                error=error,
            )
            print(
                f"[{position}/{total}] {case['caseId']} "
                f"{'PASS' if record['passed'] else 'FAIL'} ({record['latencyMs']}ms)",
                flush=True,
            )

    report = aggregate(
        suite,
        results,
        model=snapshot.model,
        run_id=run_id,
        artifact_path=writer.path,
        event_count=writer.event_count + 1,
    )
    writer.write("comparisons.json", report["evaluationPaths"]["controlledExperiments"])
    writer.event("run_completed", passed=report["passed"], passRate=report["passRate"])
    report["evaluationPaths"]["artifactAggregation"]["eventCount"] = writer.event_count
    writer.write("report.json", report)
    print(f"pass rate: {report['passedCaseRuns']}/{report['caseRuns']} ({report['passRate']:.2%})")
    print(f"report: {writer.path / 'report.json'}")
    return 0 if report["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("evals/suites/agent_system_model_v1.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("/root/.nanobot/config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("evals/runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--layer", choices=LAYERS)
    parser.add_argument("--case")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
