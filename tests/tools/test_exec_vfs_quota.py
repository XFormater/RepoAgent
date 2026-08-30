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
