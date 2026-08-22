"""Read-only source and GitHub issue tools bound to a repository workspace."""

# pyright: reportAbstractUsage=false, reportIncompatibleMethodOverride=false

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, cast

import httpx

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.repo_agent.repository import RepositoryError, read_worktree_diff


class _RepositoryTool(Tool):
    _plugin_discoverable = False

    def __init__(self, cache_root: Path | Iterable[Path]):
        roots = [cache_root] if isinstance(cache_root, Path) else list(cache_root)
        self.cache_roots = tuple(path.resolve() for path in roots)
        if not self.cache_roots:
            raise ValueError("repository tools require at least one allowed root")

    @property
    def read_only(self) -> bool:
        return True

    def _workspace(self) -> Path:
        context = current_request_context()
        if context is None or context.workspace is None:
            raise ValueError("repository tool requires an active repository workspace")
        workspace = context.workspace.resolve()
        if not any(workspace == root or root in workspace.parents for root in self.cache_roots):
            raise ValueError("active repository workspace is outside the repository cache")
        return workspace

    def _path(self, relative: str) -> Path:
        if "\0" in relative:
            raise ValueError("path contains invalid characters")
        workspace = self._workspace()
        path = (workspace / relative).resolve()
        if path != workspace and workspace not in path.parents:
            raise ValueError("path escapes the repository workspace")
        return path


@tool_parameters(
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": ""},
            "max_depth": {"type": "integer", "minimum": 1, "maximum": 8, "default": 3},
            "max_entries": {
                "type": "integer",
                "minimum": 10,
                "maximum": 5000,
                "default": 500,
            },
        },
    }
)
class RepositoryTreeTool(_RepositoryTool):
    @property
    def name(self) -> str:
        return "repo_tree"

    @property
    def description(self) -> str:
        return "List the active repository tree without leaving its workspace."

    async def execute(
        self,
        path: str = "",
        max_depth: int = 3,
        max_entries: int = 500,
    ) -> str:
        root = self._path(path)
        if not root.is_dir():
            return ToolResult.error(f"Not a directory: {path}")
        base_depth = len(root.parts)
        entries: list[str] = []
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.parts) - base_depth
            directories[:] = sorted(
                name for name in directories if name not in {".git", "node_modules", "__pycache__"}
            )
            if depth >= max_depth:
                directories[:] = []
            relative_root = current_path.relative_to(self._workspace())
            for name in [*directories, *sorted(files)]:
                candidate = relative_root / name
                suffix = "/" if name in directories else ""
                entries.append(candidate.as_posix() + suffix)
                if len(entries) >= max_entries:
                    return "\n".join(entries) + "\n... output truncated"
        return "\n".join(entries) or "(empty directory)"


@tool_parameters(
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1, "default": 1},
            "end_line": {"type": "integer", "minimum": 1, "default": 400},
        },
        "required": ["path"],
    }
)
class RepositoryReadTool(_RepositoryTool):
    @property
    def name(self) -> str:
        return "repo_read"

    @property
    def description(self) -> str:
        return "Read a UTF-8 source file with stable line numbers from the active repository."

    async def execute(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 400,
    ) -> str:
        target = self._path(path)
        if not target.is_file():
            return ToolResult.error(f"File not found: {path}")
        if target.stat().st_size > 2_000_000:
            return ToolResult.error("File is larger than the 2 MB repository-read limit.")
        if end_line < start_line:
            return ToolResult.error("end_line must be greater than or equal to start_line")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start_line - 1 : min(end_line, start_line + 1999)]
        return "\n".join(
            f"{number}|{content}"
            for number, content in enumerate(selected, start=start_line)
        )


@tool_parameters(
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 300},
            "path": {"type": "string", "default": ""},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "default": 50,
            },
        },
        "required": ["query"],
    }
)
class RepositorySearchTool(_RepositoryTool):
    @property
    def name(self) -> str:
        return "repo_search"

    @property
    def description(self) -> str:
        return "Search source text and return repository-relative file paths and line numbers."

    async def execute(
        self,
        query: str,
        path: str = "",
        max_results: int = 50,
    ) -> str:
        search_root = self._path(path)
        if not search_root.exists():
            return ToolResult.error(f"Path not found: {path}")
        query_folded = query.casefold()

        def search() -> list[str]:
            matches: list[str] = []
            candidates = [search_root] if search_root.is_file() else search_root.rglob("*")
            for candidate in candidates:
                if not candidate.is_file() or any(
                    part in {".git", "node_modules", "__pycache__", "dist", "build"}
                    for part in candidate.parts
                ):
                    continue
                try:
                    if candidate.stat().st_size > 1_000_000:
                        continue
                    content = candidate.read_text(encoding="utf-8", errors="strict")
                except (OSError, UnicodeError):
                    continue
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if query_folded in line.casefold():
                        relative = candidate.relative_to(self._workspace()).as_posix()
                        excerpt = re.sub(r"\s+", " ", line.strip())[:300]
                        matches.append(f"{relative}:{line_number}: {excerpt}")
                        if len(matches) >= max_results:
                            return matches
            return matches

        matches = await asyncio.to_thread(search)
        return "\n".join(matches) if matches else "No matches found."


@tool_parameters(
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": ""},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
        },
    }
)
class RepositoryGitHistoryTool(_RepositoryTool):
    @property
    def name(self) -> str:
        return "repo_git_history"

    @property
    def description(self) -> str:
        return "Read recent Git commit history for the active repository or one path."

    async def execute(self, path: str = "", limit: int = 20) -> str:
        workspace = self._workspace()
        args = [
            "git",
            "log",
            f"-n{limit}",
            "--date=short",
            "--pretty=format:%h%x09%ad%x09%an%x09%s",
        ]
        if path:
            target = self._path(path)
            args.extend(["--", str(target.relative_to(workspace))])
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(workspace),
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            return ToolResult.error(stderr.decode("utf-8", errors="replace")[-1000:])
        return stdout.decode("utf-8", errors="replace") or "No Git history available."


@tool_parameters({"type": "object", "properties": {}})
class RepositoryDiffTool(_RepositoryTool):
    @property
    def name(self) -> str:
        return "repo_diff"

    @property
    def description(self) -> str:
        return "Read the current uncommitted task diff, including newly added files."

    async def execute(self) -> str:
        workspace = self._workspace()
        try:
            output = await read_worktree_diff(workspace, max_chars=120_000)
        except RepositoryError as exc:
            return ToolResult.error(str(exc))
        return output or "No uncommitted changes."


@tool_parameters(
    {
        "type": "object",
        "properties": {
            "issue_number": {"type": "integer", "minimum": 1},
        },
        "required": ["issue_number"],
    }
)
class GitHubIssueTool(_RepositoryTool):
    @property
    def name(self) -> str:
        return "github_get_issue"

    @property
    def description(self) -> str:
        return "Fetch one public GitHub issue and its comments for the active repository."

    async def execute(self, issue_number: int) -> str:
        context = current_request_context()
        metadata = context.metadata if context is not None else {}
        full_name = metadata.get("repository_full_name")
        if not isinstance(full_name, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", full_name
        ):
            return ToolResult.error("Active repository metadata is missing.")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "nanobot-repository-agent",
        }
        token = os.environ.get("NANOBOT_GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.get(
                f"https://api.github.com/repos/{full_name}/issues/{issue_number}",
                headers=headers,
            )
            if response.status_code != 200:
                return ToolResult.error(
                    f"GitHub returned HTTP {response.status_code}: {response.text[:500]}"
                )
            issue_value: object = response.json()
            if not isinstance(issue_value, dict):
                return ToolResult.error("GitHub issue response is not an object.")
            issue = cast(dict[str, Any], issue_value)
            comments_response = await client.get(
                f"https://api.github.com/repos/{full_name}/issues/{issue_number}/comments",
                headers=headers,
                params={"per_page": 30},
            )
        comments_value: object = (
            comments_response.json() if comments_response.status_code == 200 else []
        )
        comments = (
            cast(list[object], comments_value)
            if isinstance(comments_value, list)
            else []
        )
        user_value: object = issue.get("user")
        user = cast(dict[str, Any], user_value) if isinstance(user_value, dict) else {}
        labels_value: object = issue.get("labels")
        labels = (
            cast(list[object], labels_value)
            if isinstance(labels_value, list)
            else []
        )
        comment_rows: list[dict[str, Any]] = []
        for item_value in comments:
            if not isinstance(item_value, dict):
                continue
            item = cast(dict[str, Any], item_value)
            item_user_value: object = item.get("user")
            item_user = (
                cast(dict[str, Any], item_user_value)
                if isinstance(item_user_value, dict)
                else {}
            )
            comment_rows.append(
                {
                    "author": item_user.get("login"),
                    "body": item.get("body"),
                    "created_at": item.get("created_at"),
                }
            )
        label_names: list[object] = []
        for item_value in labels:
            if isinstance(item_value, dict):
                label_names.append(cast(dict[str, Any], item_value).get("name"))
        payload: dict[str, Any] = {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "state": issue.get("state"),
            "author": user.get("login"),
            "body": issue.get("body"),
            "labels": label_names,
            "comments": comment_rows,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def repository_tools(cache_root: Path | Iterable[Path]) -> list[Tool]:
    return [
        RepositoryTreeTool(cache_root),
        RepositoryReadTool(cache_root),
        RepositorySearchTool(cache_root),
        RepositoryGitHistoryTool(cache_root),
        RepositoryDiffTool(cache_root),
        GitHubIssueTool(cache_root),
    ]
