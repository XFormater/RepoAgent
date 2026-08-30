from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.repo_agent.config import RepositoryAgentConfig
from nanobot.repo_agent.executor import RepositoryAgentRun
from nanobot.repo_agent.github import PublishResult
from nanobot.repo_agent.models import (
    ApprovalStatus,
    PlanStep,
    RepositoryRecord,
    RepositoryTask,
    StepStatus,
    TaskStatus,
)
from nanobot.repo_agent.orchestrator import RepositoryOrchestrator
from nanobot.utils.llm_runtime import LLMRuntime


def test_academic_project_search_routes_without_github_url(
    tmp_path: Path,
) -> None:
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(enabled=True),
        workspace=tmp_path,
        tools=ToolRegistry(),
    )

    assert orchestrator.can_handle(
        "寻找图神经网络用于药物发现的 GitHub 开源项目",
        "cli:test",
    )
    assert orchestrator.is_research_request(
        "寻找图神经网络用于药物发现的 GitHub 开源项目"
    )
    assert not orchestrator.is_research_request(
        "https://github.com/owner/repo 分析项目架构"
    )


@pytest.mark.parametrize(
    "user_text",
    [
        "下载并总结 5 个图像生成开源项目",
        "整理五个图像分割 GitHub 项目并比较实现",
        "盘点并评估几个开源代码仓库",
        "clone and summarize 5 open-source image generation projects",
        "curate a shortlist of GitHub repositories for image segmentation",
    ],
)
def test_repository_collection_requests_route_to_research_orchestrator(
    tmp_path: Path,
    user_text: str,
) -> None:
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(enabled=True),
        workspace=tmp_path,
        tools=ToolRegistry(),
    )

    assert orchestrator.is_research_request(user_text)
    assert orchestrator.can_handle(user_text, "websocket:test")


def test_explicit_research_command_always_routes_to_repository_orchestrator(
    tmp_path: Path,
) -> None:
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(enabled=True),
        workspace=tmp_path,
        tools=ToolRegistry(),
    )
    orchestrator.config.research.enabled = False

    command = "/rere 分析三个指定实现"
    assert not orchestrator.is_research_request(command)
    assert orchestrator.is_command_request(command)
    assert orchestrator.can_handle(command, "websocket:test")


@pytest.mark.asyncio
async def test_explicit_research_command_forces_workflow_execution(tmp_path: Path) -> None:
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(enabled=True),
        workspace=tmp_path,
        tools=ToolRegistry(),
    )
    task = MagicMock(report="forced report", research_id="research-forced")
    orchestrator.research.run = AsyncMock(return_value=task)  # type: ignore[method-assign]

    result = await orchestrator._run_scoped(
        request="/rere 分析三个指定候选",
        actor="user",
        session_key="websocket:test",
        initial_messages=[],
        runtime=MagicMock(),
        invoke=AsyncMock(),
    )

    assert "forced report" in result.final_content
    assert orchestrator.research.run.await_args.kwargs["direction"] == "分析三个指定候选"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _ready_task(tmp_path: Path, orchestrator: RepositoryOrchestrator) -> RepositoryTask:
    source = tmp_path / "task"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "test")
    _git(source, "config", "user.email", "test@example.com")
    (source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", "app.py")
    _git(source, "commit", "-m", "initial")
    commit = _git(source, "rev-parse", "HEAD")
    (source / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    task = RepositoryTask(
        session_key="cli:test",
        repository_id="repo",
        repository_full_name="owner/repo",
        source_url="https://github.com/owner/repo",
        base_branch="main",
        base_commit=commit,
        objective="fix value",
        workspace_path=str(source),
        status=TaskStatus.READY_TO_SAVE,
        steps=[
            PlanStep(
                step_id="implement",
                name="Implement",
                instruction="change value",
                assigned_agent="code",
                status=StepStatus.SUCCEEDED,
                result="changed app.py",
            ),
            PlanStep(
                step_id="review",
                name="Review",
                instruction="review",
                assigned_agent="review",
                depends_on=["implement"],
                status=StepStatus.SUCCEEDED,
                result="REVIEW_STATUS: PASS",
            ),
        ],
    )
    orchestrator.store.save_task(task)
    record = RepositoryRecord(
        repository_id="repo",
        url=task.source_url,
        owner="owner",
        name="repo",
        default_branch="main",
        commit_sha=commit,
        local_path=str(source),
        file_count=1,
        size_bytes=10,
    )
    orchestrator.store.set_active_repository(task.session_key, record)
    return task


async def _cached_record(
    orchestrator: RepositoryOrchestrator,
    owner: str = "owner",
    name: str = "repo",
) -> RepositoryRecord:
    source = (
        orchestrator.manager.cache_root
        / owner.casefold()
        / name.casefold()
        / "source"
    )
    source.mkdir(parents=True)
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "test")
    _git(source, "config", "user.email", "test@example.com")
    (source / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-m", "initial")
    _git(
        source,
        "remote",
        "add",
        "origin",
        f"https://github.com/{owner}/{name}.git",
    )
    return await orchestrator.manager.load(f"https://github.com/{owner}/{name}")


@pytest.mark.asyncio
async def test_state_driven_executor_runs_code_test_and_review(tmp_path: Path) -> None:
    tools = ToolRegistry()
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(
            enabled=True,
            intent_strategy="rules",
            test_execution="disabled",
        ),
        workspace=tmp_path,
        tools=tools,
    )
    task = _ready_task(tmp_path, orchestrator)
    task.status = TaskStatus.PLANNING
    task.steps = [
        PlanStep(
            step_id="implement",
            name="Implement",
            instruction="implement",
            assigned_agent="code",
        ),
        PlanStep(
            step_id="test",
            name="Test",
            instruction="test",
            assigned_agent="test",
            depends_on=["implement"],
        ),
        PlanStep(
            step_id="review",
            name="Review",
            instruction="review",
            assigned_agent="review",
            depends_on=["test"],
        ),
    ]
    orchestrator.store.save_task(task)
    record = orchestrator.store.active_repository(task.session_key)
    assert record is not None
    calls: list[str] = []

    async def invoke(
        messages,
        *,
        runtime,
        tools,
        max_iterations,
        ephemeral,
        stream,
        workspace,
        repository_full_name,
    ) -> RepositoryAgentRun:
        del runtime, tools, max_iterations, ephemeral, stream, repository_full_name
        instruction = str(messages[-1]["content"])
        calls.append(instruction)
        content = (
            "REVIEW_STATUS: PASS"
            if instruction == "review"
            else f"completed {instruction} in {workspace}"
        )
        return RepositoryAgentRun(
            final_content=content,
            tools_used=[],
            messages=messages,
            stop_reason="completed",
        )

    records = await orchestrator.executor.execute(
        task,
        record,
        runtime=cast(LLMRuntime, object()),
        invoke=invoke,
    )

    assert calls == ["implement", "test", "review"]
    assert len(records) == 3
    stored = orchestrator.store.load_task(task.task_id)
    assert stored is not None
    assert stored.status == TaskStatus.READY_TO_SAVE
    assert all(step.status == StepStatus.SUCCEEDED for step in stored.steps)


@pytest.mark.asyncio
async def test_draft_pr_requires_approval_then_publishes(tmp_path: Path) -> None:
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(enabled=True, intent_strategy="rules"),
        workspace=tmp_path,
        tools=ToolRegistry(),
    )
    task = _ready_task(tmp_path, orchestrator)
    orchestrator.publisher.current_user = AsyncMock(return_value="alice")
    orchestrator.publisher.publish = AsyncMock(
        return_value=PublishResult(
            fork_url="https://github.com/alice/repo",
            branch="nanobot/fix",
            commit_sha="b" * 40,
            pull_request_url="https://github.com/owner/repo/pull/7",
        )
    )
    runtime = cast(LLMRuntime, object())

    first = await orchestrator.run(
        request="/repo pr",
        actor="user",
        session_key=task.session_key,
        initial_messages=[{"role": "user", "content": "/repo pr"}],
        runtime=runtime,
        invoke=cast(object, None),
    )
    assert "/approve" in first.final_content
    stored_task = orchestrator.store.load_task(task.task_id)
    assert stored_task is not None
    assert stored_task.approval_id is not None

    second = await orchestrator.run(
        request=f"/approve {stored_task.approval_id}",
        actor="user",
        session_key=task.session_key,
        initial_messages=[{"role": "user", "content": "approve"}],
        runtime=runtime,
        invoke=cast(object, None),
    )

    assert "pull/7" in second.final_content
    published = orchestrator.store.load_task(task.task_id)
    approval = orchestrator.store.load_approval(stored_task.approval_id)
    assert published is not None
    assert published.status == TaskStatus.PUBLISHED
    assert approval is not None
    assert approval.status == ApprovalStatus.EXECUTED


@pytest.mark.asyncio
async def test_changed_diff_invalidates_publish_approval(tmp_path: Path) -> None:
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(enabled=True, intent_strategy="rules"),
        workspace=tmp_path,
        tools=ToolRegistry(),
    )
    task = _ready_task(tmp_path, orchestrator)
    orchestrator.publisher.current_user = AsyncMock(return_value="alice")
    runtime = cast(LLMRuntime, object())
    first = await orchestrator.run(
        request="/repo fork",
        actor="user",
        session_key=task.session_key,
        initial_messages=[],
        runtime=runtime,
        invoke=cast(object, None),
    )
    assert "/approve" in first.final_content
    stored = orchestrator.store.load_task(task.task_id)
    assert stored is not None and stored.approval_id is not None
    (task.workspace / "app.py").write_text("VALUE = 3\n", encoding="utf-8")

    result = await orchestrator.run(
        request=f"/approve {stored.approval_id}",
        actor="user",
        session_key=task.session_key,
        initial_messages=[],
        runtime=runtime,
        invoke=cast(object, None),
    )

    assert "失效" in result.final_content
    assert result.stop_reason == "policy_denied"


@pytest.mark.asyncio
async def test_natural_language_delete_requires_approval_then_deletes_target(
    tmp_path: Path,
) -> None:
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(enabled=True, intent_strategy="rules"),
        workspace=tmp_path,
        tools=ToolRegistry(),
    )
    record = await _cached_record(orchestrator, "owner", "repo")
    other = await _cached_record(orchestrator, "other", "keep")
    orchestrator.store.set_active_repository("cli:test", record)
    runtime = cast(LLMRuntime, object())

    prepared = await orchestrator.run(
        request="删除本地下载的 owner/repo 项目",
        actor="user",
        session_key="cli:test",
        initial_messages=[],
        runtime=runtime,
        invoke=cast(object, None),
    )

    match = re.search(r"/approve-delete\s+([A-Za-z0-9_-]+)", prepared.final_content)
    assert match is not None
    agent_ui = prepared.agent_ui
    assert agent_ui is not None
    assert agent_ui == {
        "kind": "repository_delete_approval",
        "data": {
            "approval_id": match.group(1),
            "targets": [
                {
                    "repository_full_name": record.full_name,
                    "commit_sha": record.commit_sha,
                    "source_path": str(record.path.resolve()),
                    "cache_path": str(record.path.resolve().parent),
                }
            ],
            "expires_at": agent_ui["data"]["expires_at"],
        },
    }
    assert record.path.exists()
    assert other.path.exists()

    completed = await orchestrator.run(
        request=f"/approve-delete {match.group(1)}",
        actor="user",
        session_key="cli:test",
        initial_messages=[],
        runtime=runtime,
        invoke=cast(object, None),
    )

    assert "本地项目已删除" in completed.final_content
    assert not record.path.exists()
    assert other.path.exists()
    assert orchestrator.store.active_repository("cli:test") is None


@pytest.mark.asyncio
async def test_delete_approval_is_invalidated_when_commit_changes(tmp_path: Path) -> None:
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(enabled=True, intent_strategy="rules"),
        workspace=tmp_path,
        tools=ToolRegistry(),
    )
    record = await _cached_record(orchestrator)
    orchestrator.store.set_active_repository("cli:test", record)
    runtime = cast(LLMRuntime, object())
    prepared = await orchestrator.run(
        request="删除当前项目",
        actor="user",
        session_key="cli:test",
        initial_messages=[],
        runtime=runtime,
        invoke=cast(object, None),
    )
    match = re.search(r"/approve-delete\s+([A-Za-z0-9_-]+)", prepared.final_content)
    assert match is not None
    (record.path / "new.txt").write_text("changed\n", encoding="utf-8")
    _git(record.path, "add", "new.txt")
    _git(record.path, "commit", "-m", "changed after approval")

    result = await orchestrator.run(
        request=f"/approve-delete {match.group(1)}",
        actor="user",
        session_key="cli:test",
        initial_messages=[],
        runtime=runtime,
        invoke=cast(object, None),
    )

    assert result.stop_reason == "policy_denied"
    assert "发生变化" in result.final_content
    assert record.path.exists()


@pytest.mark.asyncio
async def test_delete_approval_rejects_different_actor(tmp_path: Path) -> None:
    orchestrator = RepositoryOrchestrator(
        config=RepositoryAgentConfig(enabled=True, intent_strategy="rules"),
        workspace=tmp_path,
        tools=ToolRegistry(),
    )
    record = await _cached_record(orchestrator)
    orchestrator.store.set_active_repository("cli:test", record)
    runtime = cast(LLMRuntime, object())
    prepared = await orchestrator.run(
        request="删除当前项目",
        actor="owner",
        session_key="cli:test",
        initial_messages=[],
        runtime=runtime,
        invoke=cast(object, None),
    )
    match = re.search(r"/approve-delete\s+([A-Za-z0-9_-]+)", prepared.final_content)
    assert match is not None

    result = await orchestrator.run(
        request=f"/approve-delete {match.group(1)}",
        actor="different-user",
        session_key="cli:test",
        initial_messages=[],
        runtime=runtime,
        invoke=cast(object, None),
    )

    assert result.stop_reason == "policy_denied"
    assert "无权" in result.final_content
    assert record.path.exists()
