from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.harness import HarnessService, TaskExecutionMode, WorkflowState
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.repo_agent.agents import RepositoryControlClient
from nanobot.repo_agent.config import RepositoryAgentConfig, RepositoryResearchConfig
from nanobot.repo_agent.executor import RepositoryAgentRun
from nanobot.repo_agent.harness import RepositoryHarnessCoordinator
from nanobot.repo_agent.models import (
    GitHubRepositoryCandidate,
    RepositoryRecord,
    RepositoryResearchTask,
    ResearchStatus,
)
from nanobot.repo_agent.repository import RepositoryManager
from nanobot.repo_agent.research import RepositoryResearchOrchestrator
from nanobot.repo_agent.store import RepositoryStateStore
from nanobot.security.logical_vfs import LogicalVFS, LogicalVFSConfig, VFSIdentity
from nanobot.utils.llm_runtime import LLMRuntime


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


@pytest.mark.asyncio
async def test_research_enqueue_uses_durable_pool_task(tmp_path: Path) -> None:
    research = _research(tmp_path)
    service = HarnessService()
    research.harness = RepositoryHarnessCoordinator(service)

    queued = await research.enqueue_pool(
        direction="分析两个 RAG 开源项目",
        actor="user-a",
        session_key="websocket:session-a",
        tenant_id="tenant-a",
        trace_id="trace-a",
    )

    assert queued.workflow_id is not None
    tasks = await service.store.list_tasks(queued.workflow_id)
    assert len(tasks) == 1
    assert tasks[0].execution_mode is TaskExecutionMode.POOL
    assert tasks[0].metadata["research_id"] == queued.research_id
    assert tasks[0].metadata["trace_id"] == "trace-a"
    assert research.store.load_research(queued.research_id) == queued


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


def test_fallback_queries_use_explicit_repository_slugs(tmp_path: Path) -> None:
    research = _research(tmp_path)

    queries = research._fallback_queries(
        "只分析 ShareLab-SII/UniAR、amazon-far/BAR 和 stepfun-ai/NextStep-1"
    )

    assert queries == [
        "repo:ShareLab-SII/UniAR",
        "repo:amazon-far/BAR",
        "repo:stepfun-ai/NextStep-1",
    ]


def test_fallback_query_stays_within_github_search_limit(tmp_path: Path) -> None:
    research = _research(tmp_path)

    queries = research._fallback_queries("自回归图像生成 " * 100)

    assert len(queries) == 1
    assert len(queries[0]) <= 240


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("下载并分析2026年的3个高星RAG检索项目", 3),
        ("下载并分析三个高星 RAG 项目", 3),
        ("整理五個圖像生成開源專案", 5),
        ("analyze three high-star RAG repositories", 3),
        ("download 4 open-source image generation projects", 4),
        ("分析 2026 年的高星 RAG 项目", None),
    ],
)
def test_requested_repository_count_uses_explicit_project_quantity(
    direction: str,
    expected: int | None,
) -> None:
    assert RepositoryResearchOrchestrator.requested_repository_count(direction) == expected


@pytest.mark.parametrize(
    ("direction", "query", "expected"),
    [
        (
            "下载并分析2026年的3个高星RAG检索项目",
            "three high star RAG retrieval projects",
            "high star RAG retrieval projects",
        ),
        (
            "download 4 open-source image generation projects",
            "top 4 image generation repositories",
            "top image generation repositories",
        ),
        (
            "分析 2026 年的高星 RAG 项目",
            "2026 high star RAG retrieval projects",
            "2026 high star RAG retrieval projects",
        ),
    ],
)
def test_query_sanitizer_treats_requested_count_as_selection_limit(
    direction: str,
    query: str,
    expected: str,
) -> None:
    assert (
        RepositoryResearchOrchestrator._strip_requested_count_from_query(
            direction,
            query,
        )
        == expected
    )


def test_native_search_queries_reuse_only_llm_planned_corpus() -> None:
    queries = RepositoryResearchOrchestrator._native_search_queries(
        "下载并分析2026年的3个高星RAG检索项目",
        ["high star RAG retrieval projects", "RAG implementation Python"],
    )

    assert queries == [
        "high star RAG retrieval projects",
        "RAG implementation Python",
    ]
    assert RepositoryResearchOrchestrator._github_repository_urls_from_text(
        "1. RAGFlow https://github.com/infiniflow/ragflow\n"
        "2. topic https://github.com/topics/retrieval-augmented-generation\n"
        "3. LightRAG https://github.com/HKUDS/LightRAG/blob/main/README.md"
    ) == [
        "https://github.com/infiniflow/ragflow",
        "https://github.com/HKUDS/LightRAG",
    ]


@pytest.mark.asyncio
async def test_query_expansion_prompt_requires_english_github_queries(
    tmp_path: Path,
) -> None:
    research = _research(tmp_path)
    captured_prompt = ""

    async def complete(prompt: str, **_: object) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return '{"queries": ["three high star RAG retrieval systems"]}'

    research.client.complete = complete  # type: ignore[method-assign]

    queries = await research.expand_queries(
        "下载并分析2026年的3个高星RAG检索项目",
        MagicMock(),
    )

    assert queries == ["high star RAG retrieval systems"]
    assert "Return English-language GitHub search phrases only" in captured_prompt
    assert "Treat requested project counts as selection limits" in captured_prompt
    assert "Requested repository count" in captured_prompt


@pytest.mark.asyncio
async def test_discover_merges_native_web_search_repository_candidates(
    tmp_path: Path,
) -> None:
    research = _research(tmp_path)
    web_search = MagicMock()
    web_search.name = "web_search"
    web_search.execute = AsyncMock(
        return_value=(
            "Results for: RAG\n"
            "1. RAGFlow\n"
            "   https://github.com/infiniflow/ragflow\n"
            "   Retrieval-Augmented Generation engine"
        )
    )
    research.tools.register(web_search)
    research.expand_queries = AsyncMock(return_value=["RAG retrieval framework"])  # type: ignore[method-assign]
    research.github.search = AsyncMock(return_value=[])  # type: ignore[method-assign]
    research.github.repository = AsyncMock(  # type: ignore[method-assign]
        return_value=GitHubRepositoryCandidate(
            full_name="infiniflow/ragflow",
            url="https://github.com/infiniflow/ragflow",
            description="Retrieval-Augmented Generation engine",
            stars=60_000,
            forks=5_000,
            language="Python",
            license="Apache-2.0",
            topics=["rag", "retrieval-augmented-generation"],
            updated_at=datetime.now(UTC),
        )
    )
    updates: list[str] = []

    async def on_progress(content: str, **_: object) -> None:
        updates.append(content)

    queries, candidates = await research.discover(
        "下载并分析2026年的3个高星RAG检索项目",
        MagicMock(),
        on_progress,
    )

    assert queries == ["RAG retrieval framework"]
    assert [candidate.full_name for candidate in candidates] == ["infiniflow/ragflow"]
    assert any("nanobot 原生 web_search" in update for update in updates)
    web_search.execute.assert_awaited_once_with(query="RAG retrieval framework", count=10)


@pytest.mark.asyncio
async def test_control_client_continues_length_truncated_output() -> None:
    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(content='{"queries": ["RAG ', finish_reason="length"),
            LLMResponse(content='systems"]}', finish_reason="stop"),
        ]
    )
    runtime = LLMRuntime(
        provider=provider,
        model="test-model",
        generation=GenerationSettings(),
        context_window_tokens=32_000,
    )

    result = await RepositoryControlClient().complete(
        "return JSON",
        runtime=runtime,
        max_tokens=8,
    )

    assert result == '{"queries": ["RAG systems"]}'
    assert provider.chat_with_retry.await_count == 2


@pytest.mark.asyncio
async def test_control_client_forwards_reasoning_progress_without_control_output() -> None:
    client = RepositoryControlClient()
    observed: list[tuple[str, bool, bool]] = []

    async def progress(
        content: str,
        *,
        reasoning: bool = False,
        reasoning_end: bool = False,
        **_: object,
    ) -> None:
        observed.append((content, reasoning, reasoning_end))

    async def run(spec):
        assert spec.max_iterations == 4
        assert spec.progress_callback is not None
        assert spec.hook is not None
        assert spec.hook.wants_streaming()
        await spec.progress_callback("machine output")
        await spec.hook.emit_reasoning("planning repositories")
        await spec.hook.emit_reasoning_end()
        return MagicMock(stop_reason="completed", final_content="result")

    client.runner.run = run  # type: ignore[method-assign]

    result = await client.complete(
        "prompt",
        runtime=MagicMock(),
        max_tokens=100,
        progress_callback=progress,
    )

    assert result == "result"
    assert observed == [
        ("planning repositories", True, False),
        ("", False, True),
    ]


@pytest.mark.asyncio
async def test_research_synthesis_streams_final_report(tmp_path: Path) -> None:
    research = _research(tmp_path)
    streamed: list[str] = []
    ended: list[bool] = []
    task = RepositoryResearchTask(
        session_key="cli:test",
        actor="user",
        direction="RAG retrieval",
    )

    async def complete(prompt: str, **kwargs: object) -> str:
        del prompt
        stream_callback = kwargs["stream_callback"]
        stream_end_callback = kwargs["stream_end_callback"]
        assert callable(stream_callback)
        assert callable(stream_end_callback)
        await stream_callback("# Report")
        await stream_end_callback(resuming=False)
        return "# Report"

    research.client.complete = complete  # type: ignore[method-assign]

    async def on_stream(delta: str) -> None:
        streamed.append(delta)

    async def on_stream_end(**_: object) -> None:
        ended.append(True)

    result = await research.synthesize(
        task,
        MagicMock(),
        on_stream=on_stream,
        on_stream_end=on_stream_end,
    )

    assert result == "# Report"
    assert streamed == ["# Report"]
    assert ended == [True]


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
async def test_research_flow_honors_requested_count_below_configured_maximum(
    tmp_path: Path,
) -> None:
    research = _research(tmp_path)
    research.research_config.max_repositories = 5
    candidates = [_candidate(f"project-{index}") for index in range(5)]
    records = {candidate.url: _record(tmp_path, candidate) for candidate in candidates}
    research.discover = AsyncMock(return_value=(["RAG"], candidates))  # type: ignore[method-assign]
    research.manager.load = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda url: records[url]
    )
    research.synthesize = AsyncMock(return_value="three-project report")  # type: ignore[method-assign]
    invoke = AsyncMock(
        return_value=RepositoryAgentRun(
            final_content="source evidence",
            tools_used=["repo_read"],
            messages=[],
            stop_reason="completed",
        )
    )

    task = await research.run(
        direction="下载并分析2026年的3个高星RAG检索项目",
        actor="user",
        session_key="cli:count",
        runtime=MagicMock(),
        invoke=invoke,
    )

    assert len(task.selected) == 3
    assert len(task.analyses) == 3
    assert invoke.await_count == 3


@pytest.mark.asyncio
async def test_research_flow_reports_download_progress_under_default_cache(
    tmp_path: Path,
) -> None:
    research = _research(tmp_path)
    candidate = _candidate("progress")
    record = _record(tmp_path, candidate)
    research.discover = AsyncMock(return_value=(["RAG"], [candidate]))  # type: ignore[method-assign]
    research.manager.load = AsyncMock(return_value=record)  # type: ignore[method-assign]
    research.synthesize = AsyncMock(return_value="report")  # type: ignore[method-assign]
    invoke = AsyncMock(
        return_value=RepositoryAgentRun(
            final_content="source evidence",
            tools_used=["repo_read"],
            messages=[],
            stop_reason="completed",
        )
    )
    updates: list[str] = []

    async def on_update(content: str, **_: object) -> None:
        updates.append(content)

    await research.run(
        direction="下载并分析1个RAG项目",
        actor="user",
        session_key="cli:progress",
        runtime=MagicMock(),
        invoke=invoke,
        on_update=on_update,
    )

    assert any(".nanobot/repositories" in update for update in updates)
    assert any("下载进度：开始 `research/progress`" in update for update in updates)
    assert any("下载进度：`research/progress` 完成" in update for update in updates)


@pytest.mark.asyncio
async def test_research_flow_uses_vfs_cache_when_identity_is_bound(
    tmp_path: Path,
) -> None:
    config = RepositoryAgentConfig(
        enabled=True,
        research=RepositoryResearchConfig(max_repositories=1),
    )
    vfs = LogicalVFS(
        tmp_path,
        LogicalVFSConfig(root=".nanobot/vfs", max_user_bytes=None, max_user_files=None),
    )
    manager = RepositoryManager(tmp_path, config, vfs=vfs)
    store = RepositoryStateStore(tmp_path, config.state_dir)
    research = RepositoryResearchOrchestrator(
        config=config,
        manager=manager,
        store=store,
        tools=ToolRegistry(),
        client=RepositoryControlClient(),
    )
    candidate = _candidate("vfs-progress")
    research.discover = AsyncMock(return_value=(["RAG"], [candidate]))  # type: ignore[method-assign]

    async def load(url: str) -> RepositoryRecord:
        assert url == candidate.url
        source = manager.cache_root / "research" / "vfs-progress" / "source"
        source.mkdir(parents=True)
        (source / "README.md").write_text("# vfs-progress\n", encoding="utf-8")
        return RepositoryRecord(
            repository_id="vfs-progress",
            url=candidate.url,
            owner="research",
            name="vfs-progress",
            default_branch="main",
            commit_sha="a" * 40,
            local_path=str(source),
            file_count=1,
            size_bytes=10,
        )

    research.manager.load = AsyncMock(side_effect=load)  # type: ignore[method-assign]
    research.synthesize = AsyncMock(return_value="report")  # type: ignore[method-assign]
    invoke = AsyncMock(
        return_value=RepositoryAgentRun(
            final_content="source evidence",
            tools_used=["repo_read"],
            messages=[],
            stop_reason="completed",
        )
    )
    updates: list[str] = []

    async def on_update(content: str, **_: object) -> None:
        updates.append(content)

    with manager.bind_vfs_identity(VFSIdentity(user_id="alice")):
        await research.run(
            direction="下载并分析1个RAG项目",
            actor="user",
            session_key="cli:vfs",
            runtime=MagicMock(),
            invoke=invoke,
            on_update=on_update,
        )

    assert any(".nanobot/vfs" in update and "/projects/" in update for update in updates)


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

    updates: list[str] = []

    async def on_update(content: str, **_: object) -> None:
        updates.append(content)

    result = await research.run(
        direction="topic",
        actor="user",
        session_key="cli:test",
        runtime=MagicMock(),
        invoke=invoke,
        on_update=on_update,
    )

    workflow = await service.store.get_workflow_by_scope(
        "default",
        research.harness.research_scope(result.research_id),
    )
    assert workflow is not None
    assert updates[0] == "正在初始化 Harness 研究工作流…"
    assert "正在扩展检索词并搜索 GitHub 候选项目…" in updates
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
