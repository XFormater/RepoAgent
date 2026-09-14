"""Intent, dispatch, and planning agents for repository workflows."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, cast

from loguru import logger

from nanobot.agent.progress_hook import AgentProgressHook
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.repo_agent.config import RepositoryAgentConfig
from nanobot.repo_agent.models import (
    PlanStep,
    RepositoryIntent,
    RepositoryIntentKind,
    RepositoryRecord,
    RepositoryTask,
    TaskComplexity,
)
from nanobot.repo_agent.registry import RepositoryExpert, RepositoryExpertRegistry
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template


def extract_json_object(text: str | None) -> dict[str, Any]:
    if not text:
        raise ValueError("model returned no JSON")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise ValueError("model response does not contain a JSON object") from None
        depth = 0
        quoted = False
        escaped = False
        end = -1
        for index, char in enumerate(stripped[start:], start=start):
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end < 0:
            raise ValueError("model response contains incomplete JSON") from None
        value = json.loads(stripped[start:end])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return cast(dict[str, Any], value)


class RepositoryControlClient:
    # These calls expose no tools. Four iterations allow all automatic
    # length-recovery continuations plus a terminal response.
    # Control prompts have no tools and should terminate on their structured
    # response; the unbounded setting applies to repository source analysis.
    _MAX_ITERATIONS = 4

    def __init__(self) -> None:
        self.runner = AgentRunner()
        self.no_tools = ToolRegistry()

    async def complete(
        self,
        prompt: str,
        *,
        runtime: LLMRuntime,
        max_tokens: int | None = None,
        progress_callback: Callable[..., Awaitable[None]] | None = None,
        stream_callback: Callable[[str], Awaitable[None]] | None = None,
        stream_end_callback: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
        async def reasoning_progress(
            content: str,
            *,
            reasoning: bool = False,
            reasoning_end: bool = False,
            **_: object,
        ) -> None:
            # Machine-readable control output should not leak into the trace;
            # forward only the model's reasoning stream.
            if progress_callback is not None and (reasoning or reasoning_end):
                await progress_callback(
                    content,
                    reasoning=reasoning,
                    reasoning_end=reasoning_end,
                )

        async def discard_control_output(_: str) -> None:
            # Force the provider's streaming path so reasoning arrives live;
            # the complete machine-readable output is returned by the runner.
            return None

        visible_progress = reasoning_progress if progress_callback is not None else None
        progress_hook = AgentProgressHook(
            on_progress=visible_progress,
            on_stream=stream_callback
            if stream_callback is not None
            else discard_control_output
            if progress_callback is not None
            else None,
            on_stream_end=stream_end_callback if stream_callback is not None else None,
        )
        result = await self.runner.run(
            AgentRunSpec(
                initial_messages=[{"role": "system", "content": prompt}],
                tools=self.no_tools,
                runtime=runtime.with_generation_overrides(
                    temperature=0.0,
                    max_tokens=max_tokens,
                ),
                max_iterations=self._MAX_ITERATIONS,
                max_tool_result_chars=4_000,
                fail_on_tool_error=True,
                finalize_on_max_iterations=False,
                progress_callback=visible_progress,
                hook=progress_hook,
            )
        )
        if result.stop_reason == "error":
            raise RuntimeError(result.error or "repository control agent failed")
        return result.final_content or ""


class RepositoryIntentAgent:
    def __init__(
        self,
        config: RepositoryAgentConfig,
        registry: RepositoryExpertRegistry,
        client: RepositoryControlClient,
    ):
        self.config = config
        self.registry = registry
        self.client = client

    @staticmethod
    def rules(request: str) -> RepositoryIntent:
        lowered = request.casefold()
        issue_match = re.search(r"(?:issue\s*#?|#)\s*(\d+)", lowered)
        change_markers = (
            "修复",
            "修改",
            "实现",
            "增加",
            "添加",
            "重构",
            "生成测试",
            "补充测试",
            "fix ",
            "implement ",
            "change ",
            "refactor",
            "add test",
            "write test",
        )
        review_markers = ("审查", "review", "检查这个 diff", "检查修改")
        bug_markers = ("bug", "报错", "错误", "异常", "根因", "为什么失败", "定位问题")
        architecture_markers = (
            "架构",
            "主流程",
            "调用链",
            "入口",
            "architecture",
            "data flow",
            "main loop",
        )
        dependency_markers = ("依赖", "dependency", "dependencies", "requirements")
        symbol_markers = ("在哪里", "哪个文件", "定义", "引用", "symbol", "defined")

        if any(marker in lowered for marker in change_markers):
            kind = RepositoryIntentKind.CODE_CHANGE
            preferred = "code"
            complexity = TaskComplexity.COMPLEX
        elif any(marker in lowered for marker in review_markers):
            kind = RepositoryIntentKind.CODE_REVIEW
            preferred = "review"
            complexity = TaskComplexity.SIMPLE
        elif issue_match and any(marker in lowered for marker in bug_markers):
            kind = RepositoryIntentKind.BUG_DIAGNOSIS
            preferred = "issue_analyst"
            complexity = TaskComplexity.SIMPLE
        elif issue_match:
            kind = RepositoryIntentKind.ISSUE_TRIAGE
            preferred = "issue_analyst"
            complexity = TaskComplexity.SIMPLE
        elif any(marker in lowered for marker in architecture_markers):
            kind = RepositoryIntentKind.ARCHITECTURE_ANALYSIS
            preferred = "repo_knowledge"
            complexity = TaskComplexity.SIMPLE
        elif any(marker in lowered for marker in dependency_markers):
            kind = RepositoryIntentKind.DEPENDENCY_ANALYSIS
            preferred = "repo_knowledge"
            complexity = TaskComplexity.SIMPLE
        elif any(marker in lowered for marker in symbol_markers):
            kind = RepositoryIntentKind.SYMBOL_LOCATION
            preferred = "repo_knowledge"
            complexity = TaskComplexity.SIMPLE
        elif any(marker in lowered for marker in ("项目", "仓库", "overview", "介绍")):
            kind = RepositoryIntentKind.REPOSITORY_OVERVIEW
            preferred = "repo_knowledge"
            complexity = TaskComplexity.SIMPLE
        else:
            kind = RepositoryIntentKind.CODE_QUESTION
            preferred = "repo_knowledge"
            complexity = TaskComplexity.SIMPLE
        return RepositoryIntent(
            intent=kind,
            complexity=complexity,
            preferred_agent=preferred,
            issue_number=int(issue_match.group(1)) if issue_match else None,
            confidence=0.6,
            reason="Deterministic repository routing fallback.",
        )

    async def classify(
        self,
        request: str,
        record: RepositoryRecord,
        runtime: LLMRuntime,
    ) -> RepositoryIntent:
        fallback = self.rules(request)
        if self.config.intent_strategy == "rules":
            return fallback
        catalog = [
            {
                "name": name,
                "description": expert.description,
                "intents": sorted(expert.intents),
            }
            for name in self.registry.names
            if (expert := self.registry.get(name)) is not None
        ]
        prompt = render_template(
            "repo_agent/intent.md",
            experts_json=json.dumps(catalog, ensure_ascii=False),
            repository=record.full_name,
            request=request,
        )
        try:
            raw = await self.client.complete(prompt, runtime=runtime, max_tokens=900)
            result = RepositoryIntent.model_validate(extract_json_object(raw))
        except Exception as exc:
            if self.config.intent_strategy == "llm":
                raise
            logger.warning("Repository intent agent failed; using rules: {}", exc)
            return fallback
        if result.intent in {
            RepositoryIntentKind.CODE_CHANGE,
            RepositoryIntentKind.TEST_GENERATION,
        }:
            result.complexity = TaskComplexity.COMPLEX
        if result.preferred_agent not in self.registry.names:
            result.preferred_agent = fallback.preferred_agent
        if result.issue_number is None:
            result.issue_number = fallback.issue_number
        return result


class RepositoryDispatcherAgent:
    def __init__(self, registry: RepositoryExpertRegistry):
        self.registry = registry

    def dispatch(self, intent: RepositoryIntent) -> RepositoryExpert | None:
        return self.registry.dispatch(intent)


class RepositoryPlannerAgent:
    def __init__(
        self,
        config: RepositoryAgentConfig,
        registry: RepositoryExpertRegistry,
        client: RepositoryControlClient,
    ):
        self.config = config
        self.registry = registry
        self.client = client

    def fallback_steps(self, intent: RepositoryIntent) -> list[PlanStep]:
        steps: list[PlanStep] = []
        if intent.issue_number is not None:
            steps.append(
                PlanStep(
                    step_id="analyze_issue",
                    name="Analyze issue and affected code",
                    instruction=(
                        f"Analyze GitHub Issue #{intent.issue_number}, find source evidence, "
                        "and define the narrow implementation scope."
                    ),
                    assigned_agent="issue_analyst",
                )
            )
        dependencies = [steps[-1].step_id] if steps else []
        steps.extend(
            [
                PlanStep(
                    step_id="implement",
                    name="Implement the bounded change",
                    instruction=(
                        "Implement the requested change in the isolated task workspace. "
                        "Read repository instructions first and modify only necessary files."
                    ),
                    assigned_agent="code",
                    depends_on=dependencies,
                ),
                PlanStep(
                    step_id="test",
                    name="Validate the change",
                    instruction=(
                        "Inspect the relevant test configuration and validate the change. "
                        "Run commands only if an execution tool is available; otherwise "
                        "state clearly that tests were not executed."
                    ),
                    assigned_agent="test",
                    depends_on=["implement"],
                ),
                PlanStep(
                    step_id="review",
                    name="Independently review the diff",
                    instruction=(
                        "Review the complete diff for correctness, scope, security, "
                        "compatibility, and test coverage. Report concrete evidence."
                    ),
                    assigned_agent="review",
                    depends_on=["test"],
                ),
            ]
        )
        return steps

    async def create_task(
        self,
        request: str,
        intent: RepositoryIntent,
        record: RepositoryRecord,
        task: RepositoryTask,
        runtime: LLMRuntime,
    ) -> RepositoryTask:
        catalog = [
            {
                "name": name,
                "description": expert.description,
                "tools": list(expert.tools),
            }
            for name in self.registry.names
            if (expert := self.registry.get(name)) is not None
            and name in {"issue_analyst", "code", "test", "review"}
        ]
        prompt = render_template(
            "repo_agent/planner.md",
            max_steps=self.config.max_plan_steps,
            repository=record.full_name,
            base_commit=record.commit_sha,
            experts_json=json.dumps(catalog, ensure_ascii=False),
            intent_json=intent.model_dump_json(),
            request=request,
        )
        try:
            raw = await self.client.complete(prompt, runtime=runtime, max_tokens=2_500)
            payload = extract_json_object(raw)
            steps_value = payload.get("steps")
            if not isinstance(steps_value, list) or not steps_value:
                raise ValueError("planner returned no steps")
            raw_steps = cast(list[object], steps_value)
            if len(raw_steps) > self.config.max_plan_steps:
                raise ValueError("planner exceeded maximum steps")
            steps = [PlanStep.model_validate(item) for item in raw_steps]
            task.objective = str(payload.get("objective") or request)
        except Exception as exc:
            logger.warning("Repository planner failed; using safe fallback: {}", exc)
            steps = self.fallback_steps(intent)
        for step in steps:
            if self.registry.get(step.assigned_agent) is None:
                raise ValueError(f"planner selected unknown expert {step.assigned_agent!r}")
        if not any(step.assigned_agent == "code" for step in steps):
            raise ValueError("complex repository plan must contain a code step")
        if not any(step.assigned_agent == "review" for step in steps):
            raise ValueError("complex repository plan must contain an independent review step")
        task.steps = steps
        return RepositoryTask.model_validate(task.model_dump())
