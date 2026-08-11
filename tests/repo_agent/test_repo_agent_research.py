from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.harness import HarnessService, WorkflowState
from nanobot.repo_agent.agents import RepositoryControlClient
from nanobot.repo_agent.config import RepositoryAgentConfig, RepositoryResearchConfig
from nanobot.repo_agent.executor import RepositoryAgentRun
from nanobot.repo_agent.harness import RepositoryHarnessCoordinator
from nanobot.repo_agent.models import (
    GitHubRepositoryCandidate,
    RepositoryRecord,
    ResearchStatus,
)
from nanobot.repo_agent.repository import RepositoryManager
from nanobot.repo_agent.research import RepositoryResearchOrchestrator
from nanobot.repo_agent.store import RepositoryStateStore


def _candidate(
    name: str,
    *,
    description: str = "graph neural network for drug discovery",
    stars: int = 100,
) -> GitHubRepositoryCandidate:
    return GitHubRepositoryCandidate(
        full_name=f"research/{name}",
        url=f"https://github.com/research/{name}",
        description=description,
        stars=stars,
        forks=20,
        language="Python",
        license="MIT",
        topics=["graph-neural-network", "drug-discovery"],
        updated_at=datetime.now(UTC),
    )


def _record(tmp_path: Path, candidate: GitHubRepositoryCandidate) -> RepositoryRecord:
    name = candidate.full_name.split("/", 1)[1]
    source = tmp_path / name
    source.mkdir()
    (source / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    return RepositoryRecord(
        repository_id=name,
        url=candidate.url,
        owner="research",
        name=name,
        default_branch="main",
        commit_sha="a" * 40,
        local_path=str(source),
        file_count=1,
        size_bytes=10,
    )


def _research(tmp_path: Path) -> RepositoryResearchOrchestrator:
    config = RepositoryAgentConfig(
        enabled=True,
        research=RepositoryResearchConfig(
            max_repositories=2,
            max_concurrent_downloads=2,
            max_concurrent_analyses=2,
        ),
    )
    manager = RepositoryManager(tmp_path, config)
    store = RepositoryStateStore(tmp_path, config.state_dir)
    return RepositoryResearchOrchestrator(
        config=config,
        manager=manager,
        store=store,
        tools=ToolRegistry(),
        client=RepositoryControlClient(),
    )


def test_candidate_ranking_deduplicates_and_prefers_relevance(tmp_path: Path) -> None:
    research = _research(tmp_path)
    relevant = _candidate("gnn-drug", stars=50)
    duplicate = _candidate("gnn-drug", stars=10)
    popular_but_unrelated = _candidate(
        "web-framework",
        description="general purpose web framework",
        stars=100_000,
    )
    popular_but_unrelated.topics = ["web", "framework"]

    ranked = research.rank_candidates(
        "graph neural network drug discovery",
        ["graph neural network drug discovery"],
        [popular_but_unrelated, duplicate, relevant],
    )

    assert [item.full_name for item in ranked].count("research/gnn-drug") == 1
    assert ranked[0].full_name == "research/gnn-drug"
    assert ranked[0].relevance_score > ranked[1].relevance_score


@pytest.mark.asyncio
async def test_research_flow_downloads_analyzes_synthesizes_and_persists(
    tmp_path: Path,
) -> None:
    research = _research(tmp_path)
    candidates = [_candidate("one", stars=200), _candidate("two", stars=100)]
    records = {
        candidate.url: _record(tmp_path, candidate)
        for candidate in candidates
    }
    research.discover = AsyncMock(  # type: ignore[method-assign]
        return_value=(["graph neural network drug discovery"], candidates)
    )
    research.manager.load = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda url: records[url]
    )
    research.synthesize = AsyncMock(return_value="# Comparative report")  # type: ignore[method-assign]
    invoke = AsyncMock(
        return_value=RepositoryAgentRun(
            final_content="Evidence from README.md:1.",
            tools_used=["repo_read"],
            messages=[],
            stop_reason="completed",
            had_injections=False,
        )
    )

    task = await research.run(
        direction="寻找图神经网络用于药物发现的开源项目",
        actor="user",
        session_key="cli:test",
        runtime=MagicMock(),
        invoke=invoke,
    )

    assert task.status == ResearchStatus.COMPLETED
    assert len(task.selected) == 2
    assert len(task.analyses) == 2
    assert task.report == "# Comparative report"
    assert all(item.repository is not None for item in task.analyses)
    assert research.store.load_research(task.research_id) == task
    assert invoke.await_count == 2


@pytest.mark.asyncio
async def test_research_flow_materializes_shared_harness_workflow(tmp_path: Path) -> None:
    research = _research(tmp_path)
    service = HarnessService()
    research.harness = RepositoryHarnessCoordinator(service)
    candidate = _candidate("harnessed")
    record = _record(tmp_path, candidate)
    research.discover = AsyncMock(return_value=(["topic"], [candidate]))  # type: ignore[method-assign]
    research.manager.load = AsyncMock(return_value=record)  # type: ignore[method-assign]
    research.synthesize = AsyncMock(return_value="shared report")  # type: ignore[method-assign]
    invoke = AsyncMock(
        return_value=RepositoryAgentRun(
            final_content="README.md:1 evidence",
            tools_used=["repo_read"],
            messages=[],
            stop_reason="completed",
        )
    )

    result = await research.run(
        direction="topic",
        actor="user",
        session_key="cli:test",
        runtime=MagicMock(),
        invoke=invoke,
    )

    workflow = await service.store.get_workflow_by_scope(
        "default",
        research.harness.research_scope(result.research_id),
    )
    assert workflow is not None
    assert workflow.state is WorkflowState.SUCCEEDED
    tasks = await service.store.list_tasks(workflow.workflow_id)
    assert {task.title for task in tasks} == {
        "Discover and rank GitHub repositories",
        "Download research/harnessed",
        "Analyze research/harnessed",
        "Synthesize cross-project research report",
    }
    analysis_call = invoke.await_args
    assert "<agent_runtime_context>" in str(analysis_call.args[0])


@pytest.mark.asyncio
async def test_research_flow_keeps_successful_results_when_one_download_fails(
    tmp_path: Path,
) -> None:
    research = _research(tmp_path)
    good = _candidate("good")
    bad = _candidate("bad")
    record = _record(tmp_path, good)
    research.discover = AsyncMock(return_value=(["topic"], [good, bad]))  # type: ignore[method-assign]

    async def load(url: str) -> RepositoryRecord:
        if url == bad.url:
            from nanobot.repo_agent.repository import RepositoryError

            raise RepositoryError("clone failed")
        return record

    research.manager.load = load  # type: ignore[method-assign]
    research.synthesize = AsyncMock(return_value="partial report")  # type: ignore[method-assign]
    invoke = AsyncMock(
        return_value=RepositoryAgentRun(
            final_content="source evidence",
            tools_used=[],
            messages=[],
            stop_reason="completed",
        )
    )

    task = await research.run(
        direction="search open source topic projects",
        actor="user",
        session_key="cli:test",
        runtime=MagicMock(),
        invoke=invoke,
    )

    assert task.status == ResearchStatus.PARTIAL
    assert len(task.analyses) == 2
    assert sum(item.error is not None for item in task.analyses) == 1
    assert sum(bool(item.summary) for item in task.analyses) == 1
