from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.shell import ExecTool
from nanobot.security.logical_vfs import (
    LogicalVFS,
    LogicalVFSConfig,
    VFSIdentity,
    bind_vfs,
    reset_vfs,
)


@pytest.mark.asyncio
async def test_exec_checks_existing_vfs_quota_before_spawn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "existing.txt").write_text("already over quota", encoding="utf-8")
    vfs = LogicalVFS(
        workspace,
        LogicalVFSConfig(
            root=".vfs",
            max_user_bytes=1,
            max_user_files=100,
            max_task_bytes=1,
            max_task_files=100,
        ),
    )
    tool = ExecTool(working_dir=str(workspace))
    tool._spawn = AsyncMock()  # type: ignore[method-assign]
    tokens = bind_vfs(vfs, VFSIdentity(user_id="quota-test"))
    try:
        result = await tool.execute(command="echo should-not-run")
    finally:
        reset_vfs(tokens)

    assert "command was not executed" in result
    assert "hard safety boundary" in result
    tool._spawn.assert_not_awaited()


def test_post_execution_quota_error_discloses_side_effects(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "new.txt").write_text("over quota", encoding="utf-8")
    vfs = LogicalVFS(
        workspace,
        LogicalVFSConfig(
            root=".vfs",
            max_user_bytes=1,
            max_user_files=100,
            max_task_bytes=1,
            max_task_files=100,
        ),
    )
    tokens = bind_vfs(vfs, VFSIdentity(user_id="quota-test"))
    try:
        result = ExecTool._vfs_quota_error(str(workspace), before_execution=False)
    finally:
        reset_vfs(tokens)

    assert result is not None
    assert "command executed" in result
    assert "Side effects were not rolled back" in result


@pytest.mark.asyncio
async def test_exec_uses_configured_quota_root_instead_of_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    research = workspace / "research"
    research.mkdir(parents=True)
    (workspace / "large-runtime-file").write_text("outside quota scope", encoding="utf-8")
    vfs = LogicalVFS(
        workspace,
        LogicalVFSConfig(
            root=".vfs",
            quota_root="research",
            max_user_bytes=1,
            max_user_files=100,
            max_task_bytes=1,
            max_task_files=100,
        ),
    )
    tool = ExecTool(working_dir=str(workspace))
    tool._spawn = AsyncMock()  # type: ignore[method-assign]
    tool._spawn.return_value = AsyncMock(pid=123)
    tool._spawn.return_value.communicate.return_value = (b"ok", b"")
    tokens = bind_vfs(vfs, VFSIdentity(user_id="quota-test"))
    try:
        result = await tool.execute(command="echo allowed")
    finally:
        reset_vfs(tokens)

    assert "command was not executed" not in result
    tool._spawn.assert_awaited_once()


def test_exec_blocks_when_configured_quota_root_is_over_limit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    research = workspace / "research"
    research.mkdir(parents=True)
    (research / "result.txt").write_text("over quota", encoding="utf-8")
    vfs = LogicalVFS(
        workspace,
        LogicalVFSConfig(
            quota_root="research",
            max_user_bytes=1,
            max_user_files=100,
        ),
    )
    tokens = bind_vfs(vfs, VFSIdentity(user_id="quota-test"))
    try:
        result = ExecTool._vfs_quota_error(str(workspace), before_execution=True)
    finally:
        reset_vfs(tokens)

    assert result is not None
    assert "command was not executed" in result


def test_exec_skips_quota_when_vfs_is_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.txt").write_text("over quota", encoding="utf-8")
    vfs = LogicalVFS(
        workspace,
        LogicalVFSConfig(enabled=False, max_user_bytes=1, max_user_files=1),
    )
    tokens = bind_vfs(vfs, VFSIdentity(user_id="quota-test"))
    try:
        result = ExecTool._vfs_quota_error(str(workspace), before_execution=True)
    finally:
        reset_vfs(tokens)

    assert result is None


def test_exec_skips_quota_when_limits_are_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.txt").write_text("over old quota", encoding="utf-8")
    vfs = LogicalVFS(
        workspace,
        LogicalVFSConfig(
            max_user_bytes=None,
            max_user_files=None,
            max_task_bytes=None,
            max_task_files=None,
        ),
    )
    tokens = bind_vfs(vfs, VFSIdentity(user_id="quota-test"))
    try:
        result = ExecTool._vfs_quota_error(str(workspace), before_execution=True)
    finally:
        reset_vfs(tokens)

    assert result is None
