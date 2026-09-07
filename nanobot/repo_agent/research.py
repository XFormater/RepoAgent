"""GitHub discovery and comparative source analysis for academic directions."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

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
    repository_api_url = "https://api.github.com/repos/{owner}/{name}"

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
            candidate = self._candidate_from_raw(raw)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _candidate_from_raw(
        self,
        raw: dict[str, Any],
    ) -> GitHubRepositoryCandidate | None:
        if not raw.get("full_name") or not raw.get("html_url"):
            return None
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
        return GitHubRepositoryCandidate(
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

    async def repository(self, repository_url: str) -> GitHubRepositoryCandidate:
        parsed = urlsplit(repository_url)
        parts = [part for part in parsed.path.split("/") if part]
        if (parsed.hostname or "").casefold() != "github.com" or len(parts) < 2:
            raise GitHubSearchError(f"不是 GitHub 仓库 URL：{repository_url}")
        owner, name = parts[0], parts[1].removesuffix(".git")
        try:
            async with httpx.AsyncClient(
                timeout=self.config.request_timeout_seconds,
                follow_redirects=False,
                transport=PinnedDNSAsyncTransport(),
            ) as client:
                response = await client.get(
                    self.repository_api_url.format(owner=owner, name=name),
                    headers=self._headers(),
                )
        except httpx.HTTPError as exc:
            raise GitHubSearchError(f"GitHub 仓库元数据请求失败：{exc}") from exc
        if response.status_code != 200:
            detail = response.text[:500]
            raise GitHubSearchError(
                f"GitHub 仓库元数据失败（HTTP {response.status_code}）：{detail}"
            )
        payload_value: object = response.json()
        if not isinstance(payload_value, dict):
            raise GitHubSearchError("GitHub 仓库元数据响应不是 JSON 对象。")
        candidate = self._candidate_from_raw(cast(dict[str, Any], payload_value))
        if candidate is None:
            raise GitHubSearchError("GitHub 仓库元数据缺少必要字段。")
        return candidate


class RepositoryResearchOrchestrator:
    _GITHUB_REPOSITORY_URL_RE = re.compile(
        r"https?://github\.com/([A-Za-z0-9_.-]{1,100})/"
        r"([A-Za-z0-9_.-]{1,100})(?:[/?#][^\s)]*)?",
        re.IGNORECASE,
    )
    _GITHUB_NON_REPOSITORY_OWNERS = {
        "apps",
        "collections",
        "events",
        "explore",
        "features",
        "marketplace",
        "new",
        "notifications",
        "orgs",
        "pricing",
        "pulls",
        "resources",
        "search",
        "settings",
        "sponsors",
        "topics",
        "trending",
    }
    _ENGLISH_COUNT_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    _CHINESE_DIGITS = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "兩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }

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
        repository_slugs = re.findall(
            r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100})",
            cleaned,
        )
        if repository_slugs:
            return [f"repo:{slug}" for slug in dict.fromkeys(repository_slugs)][:4]
        # GitHub's search API rejects queries longer than 256 characters.
        return [cleaned[:240]]

    @staticmethod
    def _native_search_queries(direction: str, queries: list[str]) -> list[str]:
        """Reuse only LLM-planned search corpus for native web discovery.

        Static domain anchors contaminate an unrelated request whenever a
        procedural word such as ``检索`` appears in it.  The query planner is
        the sole component that translates the user's research direction into
        searchable terminology; both GitHub Search and native web search must
        consume that same bounded corpus.
        """
        del direction
        return list(dict.fromkeys(query.strip() for query in queries if query.strip()))[:4]

    @classmethod
    def _github_repository_urls_from_text(cls, text: str) -> list[str]:
        urls: list[str] = []
        for match in cls._GITHUB_REPOSITORY_URL_RE.finditer(text):
            owner = match.group(1).removesuffix(".git")
            name = match.group(2).removesuffix(".git")
            if owner.casefold() in cls._GITHUB_NON_REPOSITORY_OWNERS:
                continue
            urls.append(f"https://github.com/{owner}/{name}")
        return list(dict.fromkeys(urls))

    @classmethod
    def _parse_chinese_count(cls, value: str) -> int | None:
        if value == "十":
            return 10
        if "十" in value:
            tens_text, ones_text = value.split("十", 1)
            tens = cls._CHINESE_DIGITS.get(tens_text, 1) if tens_text else 1
            ones = cls._CHINESE_DIGITS.get(ones_text, 0) if ones_text else 0
            return tens * 10 + ones
        result = 0
        for char in value:
            digit = cls._CHINESE_DIGITS.get(char)
            if digit is None:
                return None
            result = result * 10 + digit
        return result or None

    @classmethod
    def requested_repository_count(cls, direction: str) -> int | None:
        """Extract an explicit project count without mistaking years for counts."""
        numeric = re.search(r"(?<!\d)(\d{1,2})\s*(?:个|個|项|項)", direction)
        if numeric:
            return int(numeric.group(1)) or None

        chinese = re.search(
            r"([零〇一二两兩三四五六七八九十]{1,3})\s*(?:个|個|项|項)",
            direction,
        )
        if chinese:
            return cls._parse_chinese_count(chinese.group(1))

        english = re.search(
            r"\b(\d{1,2}|" + "|".join(cls._ENGLISH_COUNT_WORDS) + r")\b"
            r"(?:\s+[\w+-]+){0,6}\s+"
            r"(?:repositories|repository|repos|repo|projects|project)\b",
            direction,
            re.IGNORECASE,
        )
        if not english:
            return None
        raw = english.group(1).casefold()
        return (int(raw) if raw.isdigit() else cls._ENGLISH_COUNT_WORDS[raw]) or None

    @classmethod
    def _requested_count_search_tokens(cls, count: int) -> set[str]:
        tokens = {str(count)}
        for word, value in cls._ENGLISH_COUNT_WORDS.items():
            if value == count:
                tokens.add(word)
        return tokens

    @classmethod
    def _strip_requested_count_from_query(cls, direction: str, query: str) -> str:
        count = cls.requested_repository_count(direction)
        if count is None:
            return query
        tokens = cls._requested_count_search_tokens(count)
        pattern = r"\b(?:" + "|".join(re.escape(token) for token in sorted(tokens)) + r")\b"
        stripped = re.sub(pattern, " ", query, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", stripped).strip(" -_,")

    async def expand_queries(
        self,
        direction: str,
        runtime: LLMRuntime,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> list[str]:
        try:
            raw = await self.client.complete(
                render_template(
                    "repo_agent/research_queries.md",
                    direction=direction,
                    requested_count=self.requested_repository_count(direction),
                ),
                runtime=runtime,
                max_tokens=800,
                progress_callback=on_progress,
            )
            payload = extract_json_object(raw)
            values = payload.get("queries")
            if not isinstance(values, list):
                raise ValueError("query planner returned no queries")
            queries = [
                self._strip_requested_count_from_query(
                    direction,
                    re.sub(r"\s+", " ", str(value)).strip(),
                )
                for value in cast(list[object], values)
                if str(value).strip()
            ][:4]
            return list(dict.fromkeys(queries)) or self._fallback_queries(direction)
        except Exception as exc:
            logger.warning("Research query expansion failed; using direction verbatim: {}", exc)
            return self._fallback_queries(direction)

    async def native_web_discover(
        self,
        direction: str,
        queries: list[str],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[list[str], list[GitHubRepositoryCandidate]]:
        web_search = self.tools.get("web_search")
        if web_search is None:
            return [], []
        native_queries = self._native_search_queries(direction, queries)
        if not native_queries:
            return [], []
        if on_progress:
            await on_progress("正在用 nanobot 原生 web_search 检索 GitHub 仓库候选…")
        candidates: list[GitHubRepositoryCandidate] = []
        seen_urls: set[str] = set()
        for query in native_queries:
            try:
                result = await web_search.execute(query=query, count=10)
            except Exception as exc:
                logger.warning("Native web_search failed for {!r}: {}", query, exc)
                continue
            if bool(getattr(result, "is_error", False)):
                logger.warning("Native web_search returned an error for {!r}: {}", query, result)
                continue
            for repository_url in self._github_repository_urls_from_text(str(result)):
                if repository_url.casefold() in seen_urls:
                    continue
                seen_urls.add(repository_url.casefold())
                try:
                    candidates.append(await self.github.repository(repository_url))
                except Exception as exc:
                    logger.warning(
                        "Native web_search GitHub candidate metadata failed for {}: {}",
                        repository_url,
                        exc,
                    )
        return native_queries, candidates

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
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[list[str], list[GitHubRepositoryCandidate]]:
        queries = await self.expand_queries(direction, runtime, on_progress)
        native_queries, native_candidates = await self.native_web_discover(
            direction,
            queries,
            on_progress,
        )
        results = await asyncio.gather(
            *(self.github.search(query) for query in queries),
            return_exceptions=True,
        )
        candidates: list[GitHubRepositoryCandidate] = [*native_candidates]
        errors: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                errors.append(str(result))
            else:
                candidates.extend(result)
        if not candidates:
            raise GitHubSearchError("；".join(errors) or "GitHub 搜索没有返回候选项目。")
        # Native results are an additional candidate source, not additional
        # relevance evidence. Ranking only against the LLM-planned corpus
        # keeps source-specific wording from changing the user's intent.
        return queries, self.rank_candidates(direction, queries, candidates)

    async def _dedup_and_fill(
        self,
        direction: str,
        candidates: list[GitHubRepositoryCandidate],
        runtime: LLMRuntime,
        target: int,
        on_progress: Callable[..., Awaitable[None]] | None,
    ) -> list[GitHubRepositoryCandidate]:
        """Dedup against already-downloaded VFS repos, then fill up to ``target``."""
        try:
            downloaded = {
                record.full_name.casefold()
                for record in await self.manager.list_vfs_repositories()
            }
        except Exception:
            downloaded = set()
        candidates = [
            candidate
            for candidate in candidates
            if candidate.full_name.casefold() not in downloaded
        ]
        seen = {candidate.full_name.casefold() for candidate in candidates} | downloaded
        rounds = 0
        while len(candidates) < target and rounds < 3:
            extra_queries = self._fallback_queries(direction)
            if not extra_queries:
                break
            if on_progress:
                await on_progress(
                    f"去重后候选不足（{len(candidates)}/{target}），补充检索…"
                )
            results = await asyncio.gather(
                *(self.github.search(query) for query in extra_queries),
                return_exceptions=True,
            )
            _, native_more = await self.native_web_discover(
                direction, extra_queries, on_progress
            )
            added: list[GitHubRepositoryCandidate] = list(native_more)
            for result in results:
                if not isinstance(result, BaseException):
                    added.extend(result)
            fresh = [
                candidate
                for candidate in added
                if candidate.full_name.casefold() not in seen
            ]
            if not fresh:
                break
            for candidate in fresh:
                seen.add(candidate.full_name.casefold())
            candidates.extend(fresh)
            candidates = self.rank_candidates(direction, [], candidates)
            rounds += 1
        return candidates

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

    async def synthesize(
        self,
        task: RepositoryResearchTask,
        runtime: LLMRuntime,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
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
                progress_callback=on_progress,
                stream_callback=on_stream,
                stream_end_callback=on_stream_end,
            )
        except Exception as exc:
            logger.warning("Research synthesis failed; using deterministic report: {}", exc)
            return self._fallback_report(task)

    async def enqueue_pool(
        self,
        *,
        direction: str,
        actor: str,
        session_key: str,
        tenant_id: str = "default",
        trace_id: str | None = None,
    ) -> RepositoryResearchTask:
        """Create a durable POOL task; an independent Worker owns execution.

        The Gateway intentionally does not execute discovery, cloning or source
        analysis after this point.  The persisted research record is the
        hand-off contract between the request process and the Worker process.
        """
        if self.harness is None:
            raise RuntimeError("/rere requires the durable Harness backend")
        from nanobot.harness import AgentRole, TaskExecutionMode

        task = RepositoryResearchTask(
            session_key=session_key,
            actor=actor,
            direction=direction.strip(),
            trace_id=trace_id,
        )
        workflow, _owner = await self.harness.workflow(
            scope_key=self.harness.research_scope(task.research_id),
            goal=task.direction,
            tenant_id=tenant_id,
        )
        task.workflow_id = workflow.workflow_id
        self.store.save_research(task, event="repository_research.queued")
        spec = self.harness.stage_spec(
            name="research-pool-runner",
            role=AgentRole.WORKER,
            objective=(
                "Run a bounded GitHub repository research workflow in an isolated "
                "Worker process and return an evidence-backed report."
            ),
            tools=("repo_tree", "repo_search", "repo_read", "repo_git_history"),
            max_iterations=max(4, self.research_config.analysis_iterations),
        )
        await self.harness.ensure_task(
            workflow,
            title="Run GitHub repository research",
            task_input=task.direction,
            metadata={
                "domain": "repository_research",
                "operation": "run",
                "research_id": task.research_id,
                "session_key": session_key,
                "actor": actor,
                "user_id": actor,
                "tenant_id": tenant_id,
                "trace_id": trace_id or "",
            },
            spec=spec,
            idempotency_key=f"repository-research:run:{task.research_id}",
            priority=100,
            execution_mode=TaskExecutionMode.POOL,
        )
        return task

    async def run(
        self,
        *,
        direction: str,
        actor: str,
        session_key: str,
        runtime: LLMRuntime,
        invoke: RepositoryAgentInvoker,
        on_update: Callable[[str], Awaitable[None]] | None = None,
        on_report_stream: Callable[[str], Awaitable[None]] | None = None,
        on_report_stream_end: Callable[..., Awaitable[None]] | None = None,
        existing_task: RepositoryResearchTask | None = None,
        use_harness: bool = True,
    ) -> RepositoryResearchTask:
        task = existing_task or RepositoryResearchTask(
            session_key=session_key,
            actor=actor,
            direction=direction.strip(),
        )
        if existing_task is None:
            self.store.save_research(task, event="repository_research.created")
        try:
            cache_root = self.manager.research_cache_root(session_key)
            cache_scope = (
                self.manager.bind_cache_root(cache_root)
                if cache_root is not None
                else nullcontext()
            )
            with cache_scope:
                return await self._run_with_cache_root(
                    task=task,
                    direction=direction,
                    actor=actor,
                    session_key=session_key,
                    runtime=runtime,
                    invoke=invoke,
                    on_update=on_update,
                    on_report_stream=on_report_stream,
                    on_report_stream_end=on_report_stream_end,
                    use_harness=use_harness,
                )
        except asyncio.CancelledError:
            task.status = ResearchStatus.CANCELLED
            self.store.save_research(task, event="repository_research.cancelled")
            raise
        except Exception:
            task.status = ResearchStatus.FAILED
            self.store.save_research(task, event="repository_research.failed")
            raise

    async def _run_with_cache_root(
        self,
        *,
        task: RepositoryResearchTask,
        direction: str,
        actor: str,
        session_key: str,
        runtime: LLMRuntime,
        invoke: RepositoryAgentInvoker,
        on_update: Callable[..., Awaitable[None]] | None,
        on_report_stream: Callable[[str], Awaitable[None]] | None,
        on_report_stream_end: Callable[..., Awaitable[None]] | None,
        use_harness: bool,
    ) -> RepositoryResearchTask:
        harness = self.harness if use_harness else None
        harness_workflow = None
        harness_owner = None
        discovery_run: RepositoryHarnessRun | None = None
        try:
            if on_update:
                await on_update(
                    "正在初始化 Harness 研究工作流…"
                    if harness is not None
                    else "正在初始化仓库研究工作流…"
                )
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
            task.queries, task.candidates = await self.discover(
                direction,
                runtime,
                on_progress=on_update,
            )
            requested_count = self.requested_repository_count(direction)
            selection_limit = min(
                requested_count or self.research_config.max_repositories,
                self.research_config.max_repositories,
            )
            fill_target = selection_limit if requested_count is not None else 0
            task.candidates = await self._dedup_and_fill(
                direction,
                task.candidates,
                runtime,
                fill_target,
                on_update,
            )
            task.selected = task.candidates[:selection_limit]
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
                await on_update(
                    f"准备下载 {len(task.selected)} 个候选项目到 "
                    f"`{self.manager.cache_root}`…"
                )
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
                if on_update:
                    await on_update(
                        f"下载进度：开始 `{candidate.full_name}` "
                        f"({candidate.stars} stars)"
                    )
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
                if on_update:
                    if isinstance(value, Exception):
                        await on_update(
                            f"下载进度：`{candidate_result.full_name}` 失败：{value}"
                        )
                    else:
                        await on_update(
                            f"下载进度：`{candidate_result.full_name}` 完成，源码在 "
                            f"`{value.local_path}`"
                        )
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
                task.report = await self.synthesize(
                    task,
                    runtime,
                    on_progress=on_update,
                    on_stream=on_report_stream,
                    on_stream_end=on_report_stream_end,
                )
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
        except Exception as exc:
            if discovery_run is not None and harness is not None:
                try:
                    await harness.fail(discovery_run, str(exc))
                except Exception:
                    pass
            raise
