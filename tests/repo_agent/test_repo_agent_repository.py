from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nanobot.repo_agent.config import RepositoryAgentConfig
from nanobot.repo_agent.models import RepositoryRecord
from nanobot.repo_agent.repository import RepositoryError, RepositoryManager


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_repo(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "test")
    _git(source, "config", "user.email", "test@example.com")
    (source / "README.md").write_text("# Real repository\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-m", "initial")
    return source, _git(source, "rev-parse", "HEAD")


def test_parse_github_repository_url(tmp_path: Path) -> None:
    manager = RepositoryManager(tmp_path, RepositoryAgentConfig())

    parsed = manager.parse_url("https://github.com/OpenAI/openai-python.git")

    assert parsed.owner == "OpenAI"
    assert parsed.name == "openai-python"
    assert manager.extract_url(
        "分析 https://github.com/OpenAI/openai-python 的架构"
    ) == "https://github.com/OpenAI/openai-python"


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/owner/repo",
        "https://evil.example/owner/repo",
        "https://github.com/owner/repo/issues/1",
        "https://user:token@github.com/owner/repo",
        "https://github.com/owner/repo?token=secret",
    ],
)
def test_parse_rejects_unsafe_or_non_root_url(tmp_path: Path, url: str) -> None:
    manager = RepositoryManager(tmp_path, RepositoryAgentConfig())

    with pytest.raises(RepositoryError):
        manager.parse_url(url)


@pytest.mark.asyncio
async def test_task_workspace_and_diff_hash_survive_local_commit(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    manager = RepositoryManager(
        tmp_path,
        RepositoryAgentConfig(cache_dir=".cache/repos"),
    )
    record = RepositoryRecord(
        repository_id="repo123",
        url="https://github.com/example/project",
        owner="example",
        name="project",
        default_branch="main",
        commit_sha=commit,
        local_path=str(source),
        file_count=1,
        size_bytes=20,
    )
    task = await manager.create_task_workspace(
        record,
        session_key="cli:test",
        objective="update readme",
    )
    (task.workspace / "README.md").write_text(
        "# Real repository\n\nUpdated.\n",
        encoding="utf-8",
    )
    (task.workspace / "new.txt").write_text("New file.\n", encoding="utf-8")

    status_before = _git(task.workspace, "status", "--porcelain=v1")
    diff_before = await manager.diff(task)
    status_after = _git(task.workspace, "status", "--porcelain=v1")
    before = await manager.diff_hash(task)

    assert status_before == status_after
    assert "?? new.txt" in status_after
    assert "new.txt" in diff_before
    assert "New file." in diff_before

    _git(task.workspace, "config", "user.name", "test")
    _git(task.workspace, "config", "user.email", "test@example.com")
    _git(task.workspace, "add", "--all")
    _git(task.workspace, "commit", "-m", "update")
    after = await manager.diff_hash(task)

    assert before == after
    assert "Updated." in await manager.diff(task)


@pytest.mark.asyncio
async def test_resolve_and_delete_one_cached_repository(tmp_path: Path) -> None:
    manager = RepositoryManager(tmp_path, RepositoryAgentConfig())
    source = manager.cache_root / "owner" / "project" / "source"
    source.mkdir(parents=True)
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "test")
    _git(source, "config", "user.email", "test@example.com")
    (source / "README.md").write_text("# Project\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-m", "initial")
    _git(
        source,
        "remote",
        "add",
        "origin",
        "https://github.com/Owner/Project.git",
    )

    record = await manager.resolve_cached_repository("删除下载的 Owner/Project 项目")
    deleted = await manager.delete_cached_repository(record)

    assert record.full_name == "Owner/Project"
    assert deleted == manager.cache_root / "owner" / "project"
    assert not deleted.exists()


@pytest.mark.asyncio
async def test_cached_repository_name_must_be_unambiguous(tmp_path: Path) -> None:
    manager = RepositoryManager(tmp_path, RepositoryAgentConfig())
    for owner in ("one", "two"):
        source = manager.cache_root / owner / "project" / "source"
        source.mkdir(parents=True)
        _git(source, "init", "-b", "main")
        _git(source, "config", "user.name", "test")
        _git(source, "config", "user.email", "test@example.com")
        (source / "README.md").write_text(f"# {owner}\n", encoding="utf-8")
        _git(source, "add", "README.md")
        _git(source, "commit", "-m", "initial")
        _git(
            source,
            "remote",
            "add",
            "origin",
            f"https://github.com/{owner}/project.git",
        )

    with pytest.raises(RepositoryError, match="名称不唯一"):
        await manager.resolve_cached_repository("删除 project")
