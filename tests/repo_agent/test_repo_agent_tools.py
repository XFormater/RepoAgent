from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.repo_agent.tools import (
    RepositoryDiffTool,
    RepositoryReadTool,
    RepositorySearchTool,
    RepositoryTreeTool,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.mark.asyncio
async def test_repository_tools_are_bound_to_active_workspace(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    repo = cache / "owner" / "repo" / "source"
    repo.mkdir(parents=True)
    (repo / "app.py").write_text(
        "def answer():\n    return 42\n",
        encoding="utf-8",
    )
    context = RequestContext(
        channel="cli",
        chat_id="chat",
        workspace=repo,
    )

    with request_context(context):
        tree = await RepositoryTreeTool(cache).execute()
        search = await RepositorySearchTool(cache).execute(query="answer")
        content = await RepositoryReadTool(cache).execute(path="app.py")

    assert "app.py" in tree
    assert "app.py:1" in search
    assert "1|def answer" in content


@pytest.mark.asyncio
async def test_repository_read_rejects_path_escape(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    repo = cache / "owner" / "repo"
    repo.mkdir(parents=True)
    context = RequestContext(channel="cli", chat_id="chat", workspace=repo)

    with request_context(context), pytest.raises(ValueError, match="escapes"):
        await RepositoryReadTool(cache).execute(path="../../outside.txt")


@pytest.mark.asyncio
async def test_repository_diff_includes_untracked_files_without_mutating_index(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    repo = cache / "owner" / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    (repo / "tracked.txt").write_text("after\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    status_before = _git(repo, "status", "--porcelain=v1")
    context = RequestContext(channel="cli", chat_id="chat", workspace=repo)

    with request_context(context):
        output = await RepositoryDiffTool(cache).execute()

    assert "tracked.txt" in output
    assert "untracked.txt" in output
    assert _git(repo, "status", "--porcelain=v1") == status_before
