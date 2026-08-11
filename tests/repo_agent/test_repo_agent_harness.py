from pathlib import Path

import pytest

from nanobot.harness import (
    HarnessService,
    TaskExecutionMode,
    TaskState,
    WorkflowState,
)
from nanobot.repo_agent.harness import RepositoryHarnessCoordinator
from nanobot.repo_agent.models import PlanStep, RepositoryTask
from nanobot.repo_agent.registry import RepositoryExpert


@pytest.mark.asyncio
async def test_repository_plan_uses_shared_harness_dag_and_isolated_context(
    tmp_path: Path,
) -> None:
    service = HarnessService()
    coordinator = RepositoryHarnessCoordinator(service)
    domain_task = RepositoryTask(
        session_key="cli:test",
        repository_id="repo",
        repository_full_name="owner/repo",
        source_url="https://github.com/owner/repo",
        base_branch="main",
        base_commit="a" * 40,
        objective="implement and review",
        workspace_path=str(tmp_path),
        steps=[
            PlanStep(
                step_id="implement",
                name="Implement",
                instruction="change one file",
                assigned_agent="code",
            ),
            PlanStep(
                step_id="review",
                name="Review",
                instruction="review the diff",
                assigned_agent="review",
                depends_on=["implement"],
            ),
        ],
    )
    experts = {
        "code": RepositoryExpert(
            name="code",
            description="Implement a bounded change.",
            intents=frozenset({"code_change"}),
            tools=("read_file", "edit_file"),
            max_iterations=5,
        ),
        "review": RepositoryExpert(
            name="review",
            description="Review evidence independently.",
            intents=frozenset({"code_review"}),
            tools=("read_file", "repo_diff"),
            max_iterations=3,
        ),
    }

    workflow, owner = await coordinator.workflow(
        scope_key=coordinator.code_scope(domain_task.task_id),
        goal=domain_task.objective,
    )
    planner_task, planner_spec = await coordinator.planner_task(
        workflow,
        task_input=domain_task.objective,
        key=f"repository-plan:{domain_task.task_id}",
    )
    planner_run = await coordinator.start(
        workflow,
        owner,
        planner_task,
        planner_spec,
        worker_id="planner",
    )
    await coordinator.complete(planner_run, "bounded plan")

    tasks = await coordinator.materialize_plan(
        workflow,
        domain_task,
        planner_task_id=planner_task.task_id,
        experts=experts,
    )
    workflow = await coordinator.seal(workflow)
    implement_task, implement_spec = tasks["implement"]
    assert implement_task.execution_mode is TaskExecutionMode.INLINE
    assert implement_spec.allowed_tools == frozenset({"read_file", "edit_file"})
    implement_run = await coordinator.start(
        workflow,
        owner,
        implement_task,
        implement_spec,
        worker_id="code",
    )
    assert implement_run.context.task_metadata["repository_full_name"] == "owner/repo"
    assert "private parent transcript" not in implement_run.context.render()
    await coordinator.complete(implement_run, "changed app.py")

    review_task, review_spec = tasks["review"]
    review_run = await coordinator.start(
        workflow,
        owner,
        review_task,
        review_spec,
        worker_id="review",
    )
    assert review_run.context.review_gate is True
    assert any(
        dependency.task_id == implement_task.task_id
        and dependency.result == "changed app.py"
        for dependency in review_run.context.dependency_results
    )
    await coordinator.complete(review_run, "APPROVED: evidence verified")

    assert (await service.store.get_task(review_task.task_id)).state is TaskState.SUCCEEDED
    assert (await service.store.get_workflow(workflow.workflow_id)).state is WorkflowState.SUCCEEDED
