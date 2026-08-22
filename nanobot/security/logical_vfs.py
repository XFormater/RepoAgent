"""Logical, tenant-scoped filesystem namespaces for agent workspaces.

This module deliberately does not try to replace the host filesystem.  It
provides one canonical path/policy layer above it: every user/task workspace is
derived from an immutable identity, paths are resolved below that workspace,
and quota/retention checks are performed before data is exposed to an agent.
Bubblewrap remains the process-level isolation mechanism when configured.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from pydantic import Field

from nanobot.config_base import Base


class LogicalVFSConfig(Base):
    """Configuration for the logical VFS layer."""

    enabled: bool = True
    root: str = ".nanobot/vfs"
    max_user_bytes: int = Field(default=5 * 1024**3, ge=1)
    max_user_files: int = Field(default=100_000, ge=1)
    max_task_bytes: int = Field(default=1 * 1024**3, ge=1)
    max_task_files: int = Field(default=50_000, ge=1)
    task_ttl_seconds: int = Field(default=7 * 24 * 3600, ge=60)
    project_ttl_seconds: int = Field(default=30 * 24 * 3600, ge=60)


class VFSPathError(PermissionError):
    """A path is outside the current logical namespace."""


class VFSQuotaError(RuntimeError):
    """A write would exceed a logical VFS quota."""


class VFSIdentityError(ValueError):
    """An identity component is empty or unsafe."""


VFSAccess = Literal["read", "write"]
VFSWorkspaceKind = Literal["project", "task", "skill", "shared_skill"]


def _component(value: str, label: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise VFSIdentityError(f"{label} must not be empty")
    # Do not expose raw channel/user identifiers in directory names.  This also
    # makes separators, dot segments and Unicode normalization irrelevant to
    # the physical path boundary.
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class VFSIdentity:
    tenant_id: str = "default"
    user_id: str = "default"
    session_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None

    @property
    def tenant_key(self) -> str:
        return _component(self.tenant_id, "tenant_id")

    @property
    def user_key(self) -> str:
        return _component(self.user_id, "user_id")

    @property
    def task_key(self) -> str | None:
        return _component(self.task_id, "task_id") if self.task_id else None


@dataclass(frozen=True, slots=True)
class VFSWorkspace:
    """A physical directory plus its logical namespace and permissions."""

    root: Path
    virtual_path: str
    kind: VFSWorkspaceKind
    read_only: bool
    identity: VFSIdentity

    def resolve(self, path: str | Path = ".", *, access: VFSAccess = "read") -> Path:
        if access == "write" and self.read_only:
            raise VFSPathError(f"{self.kind} workspace is read-only")
        raw = Path(path)
        if raw.is_absolute():
            # VFS callers use paths relative to the returned namespace.  An
            # absolute host path would bypass that contract.
            raise VFSPathError("absolute host paths are not valid VFS paths")
        candidate = (self.root / raw).resolve(strict=False)
        try:
            candidate.relative_to(self.root.resolve(strict=False))
        except ValueError as exc:
            raise VFSPathError("path escapes the logical VFS workspace") from exc
        return candidate


@dataclass(frozen=True, slots=True)
class VFSUsage:
    bytes: int
    files: int


_CURRENT_VFS: ContextVar["LogicalVFS | None"] = ContextVar(
    "nanobot_logical_vfs", default=None
)
_CURRENT_IDENTITY: ContextVar[VFSIdentity | None] = ContextVar(
    "nanobot_logical_vfs_identity", default=None
)


class LogicalVFS:
    """Resolve and govern user/task namespaces below one configured root."""

    def __init__(self, workspace: Path, config: LogicalVFSConfig | None = None):
        self.host_workspace = Path(workspace).expanduser().resolve(strict=False)
        self.config = config or LogicalVFSConfig()
        configured = Path(self.config.root).expanduser()
        self.root = (configured if configured.is_absolute() else self.host_workspace / configured).resolve(
            strict=False
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.counters: Counter[str] = Counter()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def user_root(self, identity: VFSIdentity) -> Path:
        return self.root / "tenants" / identity.tenant_key / "users" / identity.user_key

    def repository_root(self, identity: VFSIdentity) -> Path:
        return self.user_root(identity) / "projects"

    def user_skills_root(self, identity: VFSIdentity) -> Path:
        root = self.user_root(identity) / "skills"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def task_root(self, identity: VFSIdentity) -> Path:
        if not identity.task_key:
            raise VFSIdentityError("task_id is required for a task workspace")
        return self.user_root(identity) / "tasks" / identity.task_key

    def workspace(
        self,
        identity: VFSIdentity,
        kind: VFSWorkspaceKind,
        name: str = "default",
        *,
        create: bool = True,
    ) -> VFSWorkspace:
        name_key = _component(name, "workspace name")
        if kind == "project":
            root = self.repository_root(identity) / name_key
            virtual = f"/projects/{name_key}"
            read_only = False
        elif kind == "task":
            root = self.task_root(identity) / name_key
            virtual = f"/tasks/{identity.task_key}/{name_key}"
            read_only = False
        elif kind == "skill":
            root = self.user_skills_root(identity) / name_key
            virtual = f"/skills/{name_key}"
            read_only = False
        elif kind == "shared_skill":
            root = self.root / "shared" / "skills" / name_key
            virtual = f"/shared/skills/{name_key}"
            read_only = True
        else:  # pragma: no cover - Literal protects normal callers
            raise VFSIdentityError(f"unsupported workspace kind: {kind}")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return VFSWorkspace(root, virtual, kind, read_only, identity)

    def usage(self, root: Path) -> VFSUsage:
        root = root.resolve(strict=False)
        if not root.exists():
            return VFSUsage(0, 0)
        total = 0
        files = 0
        for current, directories, names in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if not (Path(current) / name).is_symlink()]
            for name in names:
                path = Path(current) / name
                try:
                    total += path.lstat().st_size
                    files += 1
                except OSError:
                    continue
        return VFSUsage(total, files)

    def metric(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)

    def check_quota(
        self,
        identity: VFSIdentity,
        *,
        root: Path | None = None,
        additional_bytes: int = 0,
        additional_files: int = 0,
    ) -> VFSUsage:
        usage = self.usage(root or self.user_root(identity))
        user_root = self.user_root(identity).resolve(strict=False)
        task_limit = identity.task_id is not None
        if root is not None:
            try:
                root.resolve(strict=False).relative_to(user_root / "tasks")
            except ValueError:
                pass
            else:
                task_limit = True
        max_bytes = self.config.max_task_bytes if task_limit else self.config.max_user_bytes
        max_files = self.config.max_task_files if task_limit else self.config.max_user_files
        if usage.bytes + max(0, additional_bytes) > max_bytes:
            self.metric("quota_exceeded")
            raise VFSQuotaError(f"VFS byte quota exceeded ({max_bytes} bytes)")
        if usage.files + max(0, additional_files) > max_files:
            self.metric("quota_exceeded")
            raise VFSQuotaError(f"VFS file quota exceeded ({max_files} files)")
        return usage

    def enforce_write(self, identity: VFSIdentity, path: Path, new_size: int) -> None:
        """Check a prospective write when *path* belongs to this identity."""
        path = path.resolve(strict=False)
        user_root = self.user_root(identity).resolve(strict=False)
        quota_root = user_root
        task_root: Path | None = None
        if identity.task_id:
            task_root = self.task_root(identity).resolve(strict=False)
        else:
            # Repository turns may only know their physical task workspace.
            # Recover the task namespace from the canonical ``tasks/<id>``
            # segment so task quotas still apply without trusting path input.
            tasks_root = (user_root / "tasks").resolve(strict=False)
            try:
                relative = path.relative_to(tasks_root)
            except ValueError:
                pass
            else:
                if relative.parts:
                    task_root = tasks_root / relative.parts[0]
        if task_root is not None:
            try:
                path.relative_to(task_root)
            except ValueError:
                pass
            else:
                quota_root = task_root
        try:
            path.relative_to(quota_root)
        except ValueError:
            # Legacy workspaces remain governed by WorkspaceScope.  The VFS
            # quota applies only after a path has entered a VFS namespace.
            return
        old_size = 0
        existed = path.exists()
        if existed:
            try:
                old_size = path.stat().st_size
            except OSError:
                old_size = 0
        self.check_quota(
            identity,
            root=quota_root,
            additional_bytes=max(0, new_size - old_size),
            additional_files=0 if existed else 1,
        )

    def is_read_only_path(self, path: Path) -> bool:
        """Return whether *path* is in a shared, read-only skill namespace."""
        resolved = path.resolve(strict=False)
        shared_roots = (
            (self.root / "shared").resolve(strict=False),
            (self.host_workspace / "skills").resolve(strict=False),
        )
        return any(
            resolved == shared_root or shared_root in resolved.parents
            for shared_root in shared_roots
        )

    def cleanup_task(self, identity: VFSIdentity) -> bool:
        root = self.task_root(identity)
        if not root.exists():
            return False
        self._safe_remove(root)
        self.metric("task_cleanup")
        return True

    def cleanup_expired(self, *, now: float | None = None) -> int:
        """Remove stale task/project namespaces and return removal count."""
        now = time.time() if now is None else now
        removed = 0
        for kind, ttl in (("tasks", self.config.task_ttl_seconds), ("projects", self.config.project_ttl_seconds)):
            base = self.root / "tenants"
            if not base.exists():
                continue
            for path in base.glob(f"*/users/*/{kind}/*"):
                try:
                    latest = 0.0
                    if path.is_dir() and not path.is_symlink():
                        for current, directories, names in os.walk(path, followlinks=False):
                            directories[:] = [
                                name for name in directories
                                if not (Path(current) / name).is_symlink()
                            ]
                            entries = [*directories, *names]
                            if not entries:
                                try:
                                    latest = max(latest, Path(current).stat().st_mtime)
                                except OSError:
                                    pass
                            for name in entries:
                                try:
                                    latest = max(latest, (Path(current) / name).stat().st_mtime)
                                except OSError:
                                    continue
                    if latest and now - latest > ttl:
                        self._safe_remove(path)
                        self.metric(f"cleanup_{kind}")
                        removed += 1
                except FileNotFoundError:
                    continue
        return removed

    def _safe_remove(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise VFSPathError("refusing to remove a path outside the VFS root") from exc
        if resolved.is_dir() and not resolved.is_symlink():
            shutil.rmtree(resolved)
        elif resolved.exists() or resolved.is_symlink():
            resolved.unlink()

    @contextmanager
    def bind(self, identity: VFSIdentity) -> Iterator[None]:
        vfs_token = _CURRENT_VFS.set(self)
        identity_token = _CURRENT_IDENTITY.set(identity)
        try:
            yield
        finally:
            _CURRENT_IDENTITY.reset(identity_token)
            _CURRENT_VFS.reset(vfs_token)


def bind_vfs(vfs: LogicalVFS, identity: VFSIdentity) -> tuple[Token[LogicalVFS | None], Token[VFSIdentity | None]]:
    return _CURRENT_VFS.set(vfs), _CURRENT_IDENTITY.set(identity)


def reset_vfs(tokens: tuple[Token[LogicalVFS | None], Token[VFSIdentity | None]]) -> None:
    vfs_token, identity_token = tokens
    _CURRENT_IDENTITY.reset(identity_token)
    _CURRENT_VFS.reset(vfs_token)


def current_vfs() -> LogicalVFS | None:
    return _CURRENT_VFS.get()


def current_vfs_identity() -> VFSIdentity | None:
    return _CURRENT_IDENTITY.get()
