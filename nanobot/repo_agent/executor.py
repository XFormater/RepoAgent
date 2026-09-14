"""State-driven execution of repository task plans."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Protocol

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.repo_agent.models import (
    PlanStep,
    RepositoryRecord,
    RepositoryTask,
    StepStatus,
    TaskStatus,
)
from nanobot.repo_agent.registry import RepositoryExpertRegistry
from nanobot.repo_agent.store import RepositoryStateStore
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.repo_agent.harness import RepositoryHarnessCoordinator, RepositoryHarnessRun


@dataclass(slots=True)
class RepositoryAgentRun:
    final_content: str
    tools_used: list[str]
    messages: list[dict[str, object]]
    stop_reason: str
    had_injections: bool = False


class RepositoryAgentInvoker(Protocol):
    def __call__(
        self,
        messages: list[dict[str, object]],
        *,
        runtime: LLMRuntime,
        tools: ToolRegistry,
        max_iterations: int | None,
        ephemeral: bool,
        stream: bool,
        workspace: Path,
        repository_full_name: str,
    ) -> Awaitable[RepositoryAgentRun]: ...


@dataclass(slots=True)
class RepositoryExecutionRecord:
    agent: str
    step_id: str
    content: str
    tools_used: list[str]
    succeeded: bool


class RepositoryPlanExecutor:
    def __init__(
        self,
        registry: RepositoryExpertRegistry,
        tools: ToolRegistry,
        store: RepositoryStateStore,
        harness: RepositoryHarnessCoordinator | None = None,
    ):
        self.registry = registry
        self.tools = tools
        self.store = store
        self.harness = harness

    @staticmethod
    def _ready(task: RepositoryTask, step: PlanStep) -> bool:
        statuses = {item.step_id: item.status for item in task.steps}
        return all(statuses.get(value) == StepStatus.SUCCEEDED for value in step.depends_on)

    async def execute(
        self,
        task: RepositoryTask,
        record: RepositoryRecord,
        *,
        runtime: LLMRuntime,
        invoke: RepositoryAgentInvoker,
    ) -> list[RepositoryExecutionRecord]:
        task.status = TaskStatus.RUNNING
        self.store.save_task(task, event="repository_task.running")
        records: list[RepositoryExecutionRecord] = []
        harness = self.harness
        harness_tasks = None
        harness_workflow = None
        harness_owner = None
        if harness is not None:
            harness_workflow, harness_owner = await harness.workflow(
                scope_key=harness.code_scope(task.task_id),
                goal=task.objective,
            )
            planner_task = await harness.service.store.get_task_by_idempotency(
                harness_workflow.workflow_id,
                f"repository-plan:{task.task_id}",
            )
            if planner_task is None or planner_task.state.value != "succeeded":
                raise RuntimeError("repository Planner must complete before Worker DAG execution")
            experts = {
                name: expert
                for name in self.registry.names
                if (expert := self.registry.get(name)) is not None
            }
            harness_tasks = await harness.materialize_plan(
                harness_workflow,
                task,
                planner_task_id=planner_task.task_id,
                experts=experts,
            )
            harness_workflow = await harness.seal(harness_workflow)
        while True:
            pending = [step for step in task.steps if step.status == StepStatus.PENDING]
            if not pending:
                break
            ready = [step for step in pending if self._ready(task, step)]
            if not ready:
                for step in pending:
                    step.status = StepStatus.SKIPPED
                    step.error = "Dependencies did not complete successfully."
                task.status = TaskStatus.FAILED
                self.store.save_task(task, event="repository_task.failed")
                return records
            step = ready[0]
            expert = self.registry.get(step.assigned_agent)
            if expert is None:
                step.status = StepStatus.FAILED
                step.error = f"Unknown expert {step.assigned_agent!r}."
                task.status = TaskStatus.FAILED
                self.store.save_task(task, event="repository_task.failed")
                return records
            expert_tools = self.tools.subset(expert.tools)
            step.status = StepStatus.RUNNING
            self.store.save_task(task, event="repository_step.running")
            previous = [
                {
                    "step_id": item.step_id,
                    "agent": item.assigned_agent,
                    "status": item.status.value,
                    "result": (item.result or "")[:8_000],
                }
                for item in task.steps
                if item.step_id in step.depends_on and item.result
            ]
            context = {
                "repository": record.full_name,
                "base_branch": record.default_branch,
                "base_commit": task.base_commit,
                "task_id": task.task_id,
                "task_workspace": str(task.workspace),
                "objective": task.objective,
                "previous_steps": previous,
            }
            harness_run: RepositoryHarnessRun | None = None
            if harness_tasks is not None:
                assert harness_workflow is not None and harness_owner is not None
                assert harness is not None
                harness_task, harness_spec = harness_tasks[step.step_id]
                persisted = await harness.service.store.get_task(harness_task.task_id)
                if persisted.state.value == "succeeded":
                    replayed = persisted.result or "Harness task completed without a result."
                    step.result = replayed
                    step.status = StepStatus.SUCCEEDED
                    records.append(
                        RepositoryExecutionRecord(
                            agent=expert.name,
                            step_id=step.step_id,
                            content=replayed,
                            tools_used=[],
                            succeeded=True,
                        )
                    )
                    self.store.save_task(task, event="repository_step.replayed")
                    continue
                harness_run = await harness.start(
                    harness_workflow,
                    harness_owner,
                    harness_task,
                    harness_spec,
                    worker_id=f"repository-inline:{task.task_id}:{step.step_id}",
                )
                context["harness_runtime_context"] = harness_run.context.render()
            system = render_template(
                "repo_agent/expert.md",
                agent_name=expert.name,
                description=expert.description,
                repository_context=json.dumps(context, ensure_ascii=False),
                instruction=step.instruction,
            )
            try:
                result = await invoke(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": step.instruction},
                    ],
                    runtime=runtime,
                    tools=expert_tools,
                    max_iterations=expert.max_iterations,
                    ephemeral=True,
                    stream=False,
                    workspace=task.workspace,
                    repository_full_name=record.full_name,
                )
                succeeded = result.stop_reason not in {
                    "error",
                    "tool_error",
                    "max_iterations",
                }
                if (
                    expert.name == "review"
                    and "REVIEW_STATUS: BLOCKED" in result.final_content.upper()
                ):
                    succeeded = False
                if harness_run is not None:
                    if succeeded:
                        assert harness is not None
                        await harness.complete(harness_run, result.final_content)
                    else:
                        assert harness is not None
                        await harness.fail(harness_run, result.final_content)
            except asyncio.CancelledError:
                if harness_run is not None:
                    assert harness is not None
                    await harness.cancel(harness_run)
                raise
            except Exception as exc:
                if harness_run is not None:
                    try:
                        assert harness is not None
                        await harness.fail(harness_run, str(exc))
                    except Exception:
                        pass
                result = RepositoryAgentRun(
                    final_content=f"Agent execution failed: {exc}",
                    tools_used=[],
                    messages=[],
                    stop_reason="error",
                )
                succeeded = False
            step.result = result.final_content
            step.status = StepStatus.SUCCEEDED if succeeded else StepStatus.FAILED
            if not succeeded:
                step.error = result.final_content
            records.append(
                RepositoryExecutionRecord(
                    agent=expert.name,
                    step_id=step.step_id,
                    content=result.final_content,
                    tools_used=result.tools_used,
                    succeeded=succeeded,
                )
            )
            self.store.save_task(
                task,
                event=(
                    "repository_step.succeeded"
                    if succeeded
                    else "repository_step.failed"
                ),
            )
            if not succeeded:
                task.status = TaskStatus.FAILED
                self.store.save_task(task, event="repository_task.failed")
                return records
        task.status = TaskStatus.READY_TO_SAVE
        self.store.save_task(task, event="repository_task.ready_to_save")
        return records
