from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from nanobot.security.logical_vfs import (
    LogicalVFS,
    LogicalVFSConfig,
    VFSIdentity,
    VFSPathError,
    VFSQuotaError,
)


def test_user_and_task_namespaces_are_stable_and_isolated(tmp_path: Path) -> None:
    vfs = LogicalVFS(tmp_path, LogicalVFSConfig(root="vfs"))
    alice = VFSIdentity(tenant_id="tenant-a", user_id="alice")
    bob = VFSIdentity(tenant_id="tenant-a", user_id="bob")
    alice_task = VFSIdentity(tenant_id="tenant-a", user_id="alice", task_id="task-1")

    alice_project = vfs.workspace(alice, "project", "research")
    bob_project = vfs.workspace(bob, "project", "research")
    task = vfs.workspace(alice_task, "task", "workspace")

    assert alice_project.root != bob_project.root
    assert alice_project.root != task.root
    assert alice_project.resolve("notes.md").is_relative_to(alice_project.root)
    with pytest.raises(VFSPathError):
        alice_project.resolve("../bob/secret.txt")
    assert alice_project.virtual_path.startswith("/projects/")


def test_same_repository_is_scoped_per_tenant_and_user(tmp_path: Path) -> None:
    vfs = LogicalVFS(tmp_path)
    first = VFSIdentity(tenant_id="tenant-a", user_id="alice")
    other_user = VFSIdentity(tenant_id="tenant-a", user_id="bob")
    other_tenant = VFSIdentity(tenant_id="tenant-b", user_id="alice")

    roots = {
        vfs.repository_root(first),
        vfs.repository_root(other_user),
        vfs.repository_root(other_tenant),
    }
    assert len(roots) == 3
    assert vfs.snapshot() == {}


def test_shared_skills_are_read_only_and_user_skills_are_writable(tmp_path: Path) -> None:
    vfs = LogicalVFS(tmp_path)
    identity = VFSIdentity(user_id="alice")
    shared = vfs.workspace(identity, "shared_skill", "research")
    private = vfs.workspace(identity, "skill", "research")

    with pytest.raises(VFSPathError):
        shared.resolve("SKILL.md", access="write")
    assert private.resolve("SKILL.md", access="write").parent == private.root
    (tmp_path / "skills").mkdir()
    assert vfs.is_read_only_path(tmp_path / "skills" / "shared" / "SKILL.md")


def test_quota_is_enforced_for_user_and_task_writes(tmp_path: Path) -> None:
    vfs = LogicalVFS(
        tmp_path,
        LogicalVFSConfig(max_user_bytes=10, max_user_files=1, max_task_bytes=5, max_task_files=1),
    )
    identity = VFSIdentity(user_id="alice")
    workspace = vfs.workspace(identity, "project", "p")
    target = workspace.resolve("a.txt", access="write")
    target.write_text("1234567890", encoding="utf-8")
    with pytest.raises(VFSQuotaError):
        vfs.enforce_write(identity, target, 11)

    task_identity = VFSIdentity(user_id="alice", task_id="t1")
    task = vfs.workspace(task_identity, "task", "work")
    task_target = task.resolve("a.txt", access="write")
    with pytest.raises(VFSQuotaError):
        vfs.enforce_write(task_identity, task_target, 6)


def test_cleanup_expired_task_namespace(tmp_path: Path) -> None:
    vfs = LogicalVFS(tmp_path, LogicalVFSConfig(task_ttl_seconds=60))
    identity = VFSIdentity(user_id="alice", task_id="old-task")
    task = vfs.workspace(identity, "task", "work")
    result = task.resolve("result.txt", access="write")
    result.write_text("done", encoding="utf-8")
    old = time.time() - 120
    os.utime(task.root, (old, old))
    os.utime(result, (old, old))
    assert vfs.cleanup_expired(now=time.time()) == 1
    assert not task.root.exists()
