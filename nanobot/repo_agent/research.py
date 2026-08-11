"""GitHub discovery and comparative source analysis for academic directions."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import httpx
from loguru import logger

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.repo_agent.agents import RepositoryControlClient, extract_json_object
from nanobot.repo_agent.config import RepositoryAgentConfig
from nanobot.repo_agent.executor import RepositoryAgentInvoker
from nanobot.repo_agent.models import (
    GitHubRepositoryCandidate,
    RepositoryResearchAnalysis,
    RepositoryResearchTask,
    ResearchStatus,
)
from nanobot.repo_agent.repository import RepositoryError, RepositoryManager
from nanobot.repo_agent.store import RepositoryStateStore
from nanobot.security.network import PinnedDNSAsyncTransport
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.repo_agent.harness import (
        RepositoryHarnessCoordinator,
        RepositoryHarnessRun,
    )


class GitHubSearchError(RuntimeError):
    pass


class GitHubRepositorySearch:
    """Small, read-only client for GitHub's repository Search API."""

    api_url = "https://api.github.com/search/repositories"

    def __init__(self, config: RepositoryAgentConfig):
        self.config = config.research

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "nanobot-repository-research",
        }
        token = os.environ.get(self.config.github_token_env, "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def search(self, query: str) -> list[GitHubRepositoryCandidate]:
        qualifiers = ["archived:false"]
        if self.config.minimum_stars:
            qualifiers.append(f"stars:>={self.config.minimum_stars}")
        search_query = f"{query.strip()} in:name,description,readme {' '.join(qualifiers)}"
        try:
            async with httpx.AsyncClient(
                timeout=self.config.request_timeout_seconds,
                follow_redirects=False,
                transport=PinnedDNSAsyncTransport(),
            ) as client:
                response = await client.get(
                    self.api_url,
                    headers=self._headers(),
                    params={
                        "q": search_query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": self.config.search_results_per_query,
                    },
                )
        except httpx.HTTPError as exc:
            raise GitHubSearchError(f"GitHub 搜索请求失败：{exc}") from exc
        if response.status_code != 200:
            detail = response.text[:500]
            raise GitHubSearchError(
                f"GitHub 搜索失败（HTTP {response.status_code}）：{detail}"
            )
        payload_value: object = response.json()
        if not isinstance(payload_value, dict):
            raise GitHubSearchError("GitHub 搜索响应不是 JSON 对象。")
        payload = cast(dict[str, Any], payload_value)
        items = payload.get("items")
        if not isinstance(items, list):
            raise GitHubSearchError("GitHub 搜索响应缺少 items。")
        candidates: list[GitHubRepositoryCandidate] = []
        for raw_value in cast(list[object], items):
            if not isinstance(raw_value, dict):
                continue
            raw = cast(dict[str, Any], raw_value)
            if not raw.get("full_name") or not raw.get("html_url"):
                continue
            license_value = raw.get("license")
            license_data = (
                cast(dict[str, Any], license_value)
                if isinstance(license_value, dict)
                else None
            )
            license_name = (
                str(license_data.get("spdx_id") or license_data.get("name"))
                if license_data is not None
                else None
            )
            topics_value = raw.get("topics")
            topics = (
                [str(value) for value in cast(list[object], topics_value)]
                if isinstance(topics_value, list)
                else []
            )
            updated: datetime | None = None
            if raw.get("updated_at"):
                try:
                    updated = datetime.fromisoformat(str(raw["updated_at"]).replace("Z", "+00:00"))
                except ValueError:
                    pass
            candidates.append(
                GitHubRepositoryCandidate(
                    full_name=str(raw["full_name"]),
                    url=str(raw["html_url"]),
                    description=str(raw.get("description") or ""),
                    stars=int(raw.get("stargazers_count") or 0),
                    forks=int(raw.get("forks_count") or 0),
                    open_issues=int(raw.get("open_issues_count") or 0),
                    language=str(raw["language"]) if raw.get("language") else None,
                    license=license_name,
                    topics=topics,
                    default_branch=str(raw.get("default_branch") or "main"),
                    updated_at=updated,
                    archived=bool(raw.get("archived")),
                )
            )
        return candidates


class RepositoryResearchOrchestrator:
    def __init__(
        self,
        *,
        config: RepositoryAgentConfig,
        manager: RepositoryManager,
        store: RepositoryStateStore,
        tools: ToolRegistry,
        client: RepositoryControlClient,
        harness: RepositoryHarnessCoordinator | None = None,
    ):
        self.config = config
        self.research_config = config.research
        self.manager = manager
        self.store = store
        self.tools = tools
        self.client = client
        self.github = GitHubRepositorySearch(config)
        self.harness = harness

    @staticmethod
    def _fallback_queries(direction: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", direction).strip()
        return [cleaned]

    async def expand_queries(self, direction: str, runtime: LLMRuntime) -> list[str]:
        try:
            raw = await self.client.complete(
                render_template(
                    "repo_agent/research_queries.md",
                    direction=direction,
                ),
                runtime=runtime,
                max_tokens=800,
            )
            payload = extract_json_object(raw)
            values = payload.get("queries")
            if not isinstance(values, list):
                raise ValueError("query planner returned no queries")
            queries = [
                re.sub(r"\s+", " ", str(value)).strip()
                for value in cast(list[object], values)
                if str(value).strip()
            ][:4]
            return list(dict.fromkeys(queries)) or self._fallback_queries(direction)
        except Exception as exc:
            logger.warning("Research query expansion failed; using direction verbatim: {}", exc)
            return self._fallback_queries(direction)

    @staticmethod
    def _terms(direction: str, queries: list[str]) -> set[str]:
        text = " ".join([direction, *queries]).casefold()
        return {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9+_.-]{1,}|[\u4e00-\u9fff]{2,}", text)
            if token not in {"github", "project", "projects", "项目", "开源", "相关"}
        }

    def rank_candidates(
        self,
        direction: str,
        queries: list[str],
        candidates: list[GitHubRepositoryCandidate],
    ) -> list[GitHubRepositoryCandidate]:
        terms = self._terms(direction, queries)
        unique: dict[str, GitHubRepositoryCandidate] = {}
        for candidate in candidates:
            key = candidate.full_name.casefold()
            current = unique.get(key)
            if current is None or candidate.stars > current.stars:
                unique[key] = candidate
        ranked: list[GitHubRepositoryCandidate] = []
        for candidate in unique.values():
            if candidate.archived:
                continue
            if self.research_config.require_license and not candidate.license:
                continue
            haystack = " ".join(
                [
                    candidate.full_name,
                    candidate.description,
                    candidate.language or "",
                    " ".join(candidate.topics),
                ]
            ).casefold()
            matched = sorted(term for term in terms if term in haystack)
            relevance = len(matched) / max(1, len(terms))
            popularity = min(1.0, math.log10(candidate.stars + 1) / 5)
            freshness = 0.0
            if candidate.updated_at is not None:
                age_days = max(
                    0.0,
                    (datetime.now(candidate.updated_at.tzinfo) - candidate.updated_at).days,
                )
                freshness = max(0.0, 1.0 - age_days / 1825)
            license_bonus = 0.05 if candidate.license else 0.0
            candidate.relevance_score = round(
                0.65 * relevance + 0.2 * popularity + 0.1 * freshness + license_bonus,
                4,
            )
            candidate.selection_reason = (
                f"匹配关键词：{', '.join(matched[:8]) or '由 GitHub 搜索召回'}；"
                f"stars={candidate.stars}；"
                f"license={candidate.license or '未声明'}"
            )
            ranked.append(candidate)
        ranked.sort(
            key=lambda value: (value.relevance_score, value.stars, value.forks),
            reverse=True,
        )
        return ranked[: self.research_config.max_candidates]

    async def discover(
        self,
        direction: str,
        runtime: LLMRuntime,
    ) -> tuple[list[str], list[GitHubRepositoryCandidate]]:
        queries = await self.expand_queries(direction, runtime)
        results = await asyncio.gather(
            *(self.github.search(query) for query in queries),
            return_exceptions=True,
        )
        candidates: list[GitHubRepositoryCandidate] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                errors.append(str(result))
            else:
                candidates.extend(result)
        if not candidates:
            raise GitHubSearchError("；".join(errors) or "GitHub 搜索没有返回候选项目。")
        return queries, self.rank_candidates(direction, queries, candidates)

    async def _download(
        self,
        candidate: GitHubRepositoryCandidate,
        semaphore: asyncio.Semaphore,
    ) -> tuple[GitHubRepositoryCandidate, Any]:
        async with semaphore:
            try:
                return candidate, await self.manager.load(candidate.url)
            except RepositoryError as exc:
                return candidate, exc

    async def analyze_repository(
        self,
        *,
        direction: str,
        candidate: GitHubRepositoryCandidate,
        record: Any,
        runtime: LLMRuntime,
        invoke: RepositoryAgentInvoker,
        semaphore: asyncio.Semaphore,
        harness_context: str | None = None,
    ) -> RepositoryResearchAnalysis:
        async with semaphore:
            prompt = render_template(
                "repo_agent/research_repository.md",
                direction=direction,
                candidate_json=candidate.model_dump_json(),
                repository=record.full_name,
            )
            messages: list[dict[str, object]] = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "分析该仓库与研究方向的相关性、方法、代码架构、运行入口、"
                        "数据与依赖、成熟度、局限性，并引用仓库相对路径和行号。"
                    ),
                },
            ]
            if harness_context:
                messages.insert(
                    1,
                    {
                        "role": "user",
                        "content": harness_context,
                    },
                )
            try:
                result = await invoke(
                    messages,
                    runtime=runtime,
                    tools=self.tools.subset(
                        ["repo_tree", "repo_search", "repo_read", "repo_git_history"]
                    ),
                    max_iterations=self.research_config.analysis_iterations,
                    ephemeral=True,
                    stream=False,
                    workspace=record.path,
                    repository_full_name=record.full_name,
                )
                return RepositoryResearchAnalysis(
                    candidate=candidate,
                    repository=record,
                    summary=result.final_content,
                    tools_used=result.tools_used,
                )
            except Exception as exc:
                logger.warning("Repository research analysis failed for {}: {}", record.full_name, exc)
                return RepositoryResearchAnalysis(
                    candidate=candidate,
                    repository=record,
                    error=str(exc),
                )

    @staticmethod
    def _fallback_report(task: RepositoryResearchTask) -> str:
        lines = [
            f"# GitHub 开源项目研究：{task.direction}",
            "",
            f"研究任务：`{task.research_id}`",
            "",
            "## 项目结论",
            "",
        ]
        for analysis in task.analyses:
            candidate = analysis.candidate
            lines.extend(
                [
                    f"### [{candidate.full_name}]({candidate.url})",
                    "",
                    (
                        f"- Stars：{candidate.stars}；语言：{candidate.language or '未知'}；"
                        f"许可证：{candidate.license or '未声明'}"
                    ),
                    f"- 入选依据：{candidate.selection_reason}",
                    f"- 本地路径：`{analysis.repository.local_path}`"
                    if analysis.repository
                    else "- 下载失败",
                    "",
                    analysis.summary or f"分析失败：{analysis.error}",
                    "",
                ]
            )
        return "\n".join(lines)

    async def synthesize(self, task: RepositoryResearchTask, runtime: LLMRuntime) -> str:
        payload = task.model_dump(mode="json", exclude={"report"})
        for analysis in payload["analyses"]:
            summary = analysis.get("summary")
            if isinstance(summary, str) and len(summary) > 12_000:
                analysis["summary"] = summary[:12_000] + "\n... analysis truncated"
        try:
            return await self.client.complete(
                render_template(
                    "repo_agent/research_synthesizer.md",
                    research_json=json.dumps(payload, ensure_ascii=False, default=str),
                ),
                runtime=runtime,
                max_tokens=4_000,
            )
        except Exception as exc:
            logger.warning("Research synthesis failed; using deterministic report: {}", exc)
            return self._fallback_report(task)

    async def run(
        self,
        *,
        direction: str,
        actor: str,
        session_key: str,
        runtime: LLMRuntime,
        invoke: RepositoryAgentInvoker,
        on_update: Callable[[str], Awaitable[None]] | None = None,
    ) -> RepositoryResearchTask:
        task = RepositoryResearchTask(
            session_key=session_key,
            actor=actor,
            direction=direction.strip(),
        )
        self.store.save_research(task, event="repository_research.created")
        harness = self.harness
        harness_workflow = None
        harness_owner = None
        discovery_run: RepositoryHarnessRun | None = None
        try:
            if harness is not None:
                from nanobot.harness import AgentRole

                harness_workflow, harness_owner = await harness.workflow(
                    scope_key=harness.research_scope(task.research_id),
                    goal=task.direction,
                )
                discovery_spec = harness.stage_spec(
                    name="research-discovery",
                    role=AgentRole.PLANNER,
                    objective=(
                        "Expand the academic direction, search GitHub, rank candidates, "
                        "and produce a bounded project shortlist."
                    ),
                    max_iterations=4,
                )
                discovery_task = await harness.ensure_task(
                    harness_workflow,
                    title="Discover and rank GitHub repositories",
                    task_input=task.direction,
                    spec=discovery_spec,
                    idempotency_key=f"research-discovery:{task.research_id}",
                    priority=100,
                )
                discovery_run = await harness.start(
                    harness_workflow,
                    harness_owner,
                    discovery_task,
                    discovery_spec,
                    worker_id=f"repository-research:{task.research_id}:discovery",
                )
            task.status = ResearchStatus.SEARCHING
            self.store.save_research(task, event="repository_research.searching")
            if on_update:
                await on_update("正在扩展检索词并搜索 GitHub 候选项目…")
            task.queries, task.candidates = await self.discover(direction, runtime)
            task.selected = task.candidates[: self.research_config.max_repositories]
            if not task.selected:
                raise GitHubSearchError("候选筛选后没有可分析项目。")
            if discovery_run is not None:
                assert harness is not None
                await harness.complete(
                    discovery_run,
                    json.dumps(
                        {
                            "queries": task.queries,
                            "selected": [
                                candidate.model_dump(mode="json")
                                for candidate in task.selected
                            ],
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
                discovery_run = None
            self.store.save_research(task, event="repository_research.candidates_selected")

            task.status = ResearchStatus.DOWNLOADING
            self.store.save_research(task, event="repository_research.downloading")
            if on_update:
                await on_update(f"正在下载 {len(task.selected)} 个候选项目…")
            download_sem = asyncio.Semaphore(self.research_config.max_concurrent_downloads)
            download_tasks: dict[str, tuple[Any, Any]] = {}
            if harness is not None:
                assert harness_workflow is not None
                from nanobot.harness import AgentRole

                download_spec = harness.stage_spec(
                    name="repository-download",
                    role=AgentRole.WORKER,
                    objective="Download one selected repository into the bounded cache.",
                    max_iterations=1,
                )
                discovery_task = discovery_run.task if discovery_run is not None else await (
                    harness.service.store.get_task_by_idempotency(
                        harness_workflow.workflow_id,
                        f"research-discovery:{task.research_id}",
                    )
                )
                assert discovery_task is not None
                for candidate in task.selected:
                    download_task = await harness.ensure_task(
                        harness_workflow,
                        title=f"Download {candidate.full_name}",
                        task_input=candidate.url,
                        metadata={
                            "domain": "repository_research",
                            "research_id": task.research_id,
                            "repository_full_name": candidate.full_name,
                            "repository_url": candidate.url,
                        },
                        spec=download_spec,
                        dependencies=frozenset({discovery_task.task_id}),
                        resource_keys=frozenset({f"repository-cache:{candidate.full_name.casefold()}"}),
                        idempotency_key=(
                            f"research-download:{task.research_id}:{candidate.full_name.casefold()}"
                        ),
                    )
                    download_tasks[candidate.full_name] = (download_task, download_spec)

            async def download_one(candidate: GitHubRepositoryCandidate) -> tuple[GitHubRepositoryCandidate, Any]:
                run: RepositoryHarnessRun | None = None
                if harness is not None:
                    assert harness_workflow is not None and harness_owner is not None
                    harness_task, harness_spec = download_tasks[candidate.full_name]
                    run = await harness.start(
                        harness_workflow,
                        harness_owner,
                        harness_task,
                        harness_spec,
                        worker_id=(
                            f"repository-research:{task.research_id}:download:"
                            f"{candidate.full_name}"
                        ),
                    )
                try:
                    candidate_result, value = await self._download(candidate, download_sem)
                except asyncio.CancelledError:
                    if run is not None:
                        assert harness is not None
                        await harness.cancel(run)
                    raise
                except Exception as exc:
                    candidate_result, value = candidate, exc
                if run is not None:
                    content = (
                        json.dumps(value.model_dump(mode="json"), ensure_ascii=False, default=str)
                        if not isinstance(value, Exception)
                        else f"DOWNLOAD_ERROR: {value}"
                    )
                    assert harness is not None
                    await harness.complete(run, content)
                return candidate_result, value

            downloads = await asyncio.gather(*(download_one(candidate) for candidate in task.selected))

            task.status = ResearchStatus.ANALYZING
            self.store.save_research(task, event="repository_research.analyzing")
            valid = [(candidate, value) for candidate, value in downloads if not isinstance(value, Exception)]
            failures = [
                RepositoryResearchAnalysis(candidate=candidate, error=str(value))
                for candidate, value in downloads
                if isinstance(value, Exception)
            ]
            if on_update:
                await on_update(f"正在并行分析 {len(valid)} 个已下载项目的代码…")
            analysis_sem = asyncio.Semaphore(self.research_config.max_concurrent_analyses)
            analysis_tasks: dict[str, tuple[Any, Any]] = {}
            if harness is not None:
                assert harness_workflow is not None
                from nanobot.harness import AgentRole

                analysis_spec = harness.stage_spec(
                    name="repository-source-analysis",
                    role=AgentRole.WORKER,
                    objective=(
                        "Analyze one repository against the research direction using "
                        "direct source evidence."
                    ),
                    tools=("repo_tree", "repo_search", "repo_read", "repo_git_history"),
                    max_iterations=self.research_config.analysis_iterations,
                )
                for candidate, _record in valid:
                    download_task, _ = download_tasks[candidate.full_name]
                    analysis_task = await harness.ensure_task(
                        harness_workflow,
                        title=f"Analyze {candidate.full_name}",
                        task_input=(
                            f"Research direction: {direction}\nRepository: {candidate.full_name}"
                        ),
                        metadata={
                            "domain": "repository_research",
                            "research_id": task.research_id,
                            "repository_full_name": candidate.full_name,
                            "repository_workspace": _record.local_path,
                            "repository_url": candidate.url,
                        },
                        spec=analysis_spec,
                        dependencies=frozenset({download_task.task_id}),
                        idempotency_key=(
                            f"research-analysis:{task.research_id}:{candidate.full_name.casefold()}"
                        ),
                    )
                    analysis_tasks[candidate.full_name] = (analysis_task, analysis_spec)

            async def analyze_one(candidate: GitHubRepositoryCandidate, record: Any) -> RepositoryResearchAnalysis:
                run: RepositoryHarnessRun | None = None
                if harness is not None:
                    assert harness_workflow is not None and harness_owner is not None
                    harness_task, harness_spec = analysis_tasks[candidate.full_name]
                    run = await harness.start(
                        harness_workflow,
                        harness_owner,
                        harness_task,
                        harness_spec,
                        worker_id=(
                            f"repository-research:{task.research_id}:analysis:"
                            f"{candidate.full_name}"
                        ),
                    )
                try:
                    analysis = await self.analyze_repository(
                        direction=direction,
                        candidate=candidate,
                        record=record,
                        runtime=runtime,
                        invoke=invoke,
                        semaphore=analysis_sem,
                        harness_context=run.context.render() if run is not None else None,
                    )
                except asyncio.CancelledError:
                    if run is not None:
                        assert harness is not None
                        await harness.cancel(run)
                    raise
                if run is not None:
                    assert harness is not None
                    await harness.complete(
                        run,
                        analysis.summary or f"ANALYSIS_ERROR: {analysis.error}",
                    )
                return analysis

            analyses = await asyncio.gather(
                *(analyze_one(candidate, record) for candidate, record in valid)
            )
            task.analyses = [*analyses, *failures]
            if not any(item.summary for item in task.analyses):
                raise RuntimeError("所有候选项目均下载或分析失败。")

            task.status = ResearchStatus.SYNTHESIZING
            self.store.save_research(task, event="repository_research.synthesizing")
            if on_update:
                await on_update("正在生成跨项目比较与最终研究报告…")
            synthesis_run: RepositoryHarnessRun | None = None
            if harness is not None:
                assert harness_workflow is not None and harness_owner is not None
                from nanobot.harness import AgentRole

                synthesis_spec = harness.stage_spec(
                    name="research-synthesizer",
                    role=AgentRole.WORKER,
                    objective="Synthesize a concise cross-project evidence-backed report.",
                    max_iterations=3,
                )
                synthesis_task = await harness.ensure_task(
                    harness_workflow,
                    title="Synthesize cross-project research report",
                    task_input=task.direction,
                    spec=synthesis_spec,
                    dependencies=frozenset(
                        harness_task.task_id
                        for harness_task, _ in analysis_tasks.values()
                    ),
                    idempotency_key=f"research-synthesis:{task.research_id}",
                )
                harness_workflow = await harness.seal(harness_workflow)
                synthesis_run = await harness.start(
                    harness_workflow,
                    harness_owner,
                    synthesis_task,
                    synthesis_spec,
                    worker_id=f"repository-research:{task.research_id}:synthesis",
                )
            try:
                task.report = await self.synthesize(task, runtime)
            except asyncio.CancelledError:
                if synthesis_run is not None:
                    assert harness is not None
                    await harness.cancel(synthesis_run)
                raise
            if synthesis_run is not None:
                assert harness is not None
                await harness.complete(synthesis_run, task.report)
            task.status = (
                ResearchStatus.PARTIAL
                if any(item.error for item in task.analyses)
                else ResearchStatus.COMPLETED
            )
            self.store.save_research(task, event="repository_research.completed")
            return task
        except asyncio.CancelledError:
            task.status = ResearchStatus.CANCELLED
            self.store.save_research(task, event="repository_research.cancelled")
            raise
        except Exception as exc:
            if discovery_run is not None and harness is not None:
                try:
                    await harness.fail(discovery_run, str(exc))
                except Exception:
                    pass
            task.status = ResearchStatus.FAILED
            self.store.save_research(task, event="repository_research.failed")
            raise
