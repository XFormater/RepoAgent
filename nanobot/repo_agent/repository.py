"""Safe GitHub repository intake and isolated task workspaces."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from nanobot.repo_agent.config import RepositoryAgentConfig
from nanobot.repo_agent.models import RepositoryRecord, RepositoryTask
from nanobot.security.logical_vfs import LogicalVFS, VFSIdentity

_GITHUB_URL_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}(?:\.git)?",
    re.IGNORECASE,
)
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


class RepositoryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedRepositoryURL:
    url: str
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class RepositoryManager:
    def __init__(
        self,
        workspace: Path,
        config: RepositoryAgentConfig,
        *,
        vfs: LogicalVFS | None = None,
    ):
        self.workspace = workspace.resolve()
        self.config = config
        self._base_cache_root = (self.workspace / config.cache_dir).resolve()
        if self._base_cache_root != self.workspace and self.workspace not in self._base_cache_root.parents:
            raise ValueError("repository cache directory must remain inside the workspace")
        self._base_cache_root.mkdir(parents=True, exist_ok=True)
        self.vfs = vfs
        self._scope: ContextVar[VFSIdentity | None] = ContextVar(
            "repository_manager_vfs_identity", default=None
        )
        self._cache_root_override: ContextVar[Path | None] = ContextVar(
            "repository_manager_cache_root_override", default=None
        )

    def resolve_cache_dir(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        resolved = (path if path.is_absolute() else self.workspace / path).resolve(
            strict=False
        )
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise ValueError("repository cache directory must remain inside the workspace")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def session_cache_root(self, session_key: str) -> Path:
        """Return the per-session repository cache root below the base cache root.

        The session key is hashed so channel/user identifiers never leak into
        directory names and path separators or dot segments cannot escape the
        base cache root. The directory is created lazily on first use.
        """
        key = hashlib.sha256((session_key or "").encode("utf-8")).hexdigest()[:24]
        root = (self._base_cache_root / key).resolve()
        if self._base_cache_root not in root.parents:
            raise RepositoryError("仓库缓存路径越界。")
        return root

    def research_cache_root(self, session_key: str) -> Path | None:
        """Resolve the cache root for a research run.

        Precedence: configured ``repository_cache_dir`` > active VFS identity >
        per-session folder below the base cache root. Returns ``None`` when an
        active VFS identity should own the cache (the caller then leaves the
        cache root unbound so the VFS scope applies).
        """
        if self.config.research.repository_cache_dir:
            return self.resolve_cache_dir(self.config.research.repository_cache_dir)
        identity = self._scope.get()
        if self.vfs is not None and self.vfs.enabled and identity is not None:
            return None
        return self.session_cache_root(session_key)

    @property
    def cache_root(self) -> Path:
        """Return the active user-scoped cache, or the legacy root."""
        override = self._cache_root_override.get()
        if override is not None:
            override.mkdir(parents=True, exist_ok=True)
            return override
        identity = self._scope.get()
        if self.vfs is not None and self.vfs.enabled and identity is not None:
            root = self.vfs.repository_root(identity)
            root.mkdir(parents=True, exist_ok=True)
            return root
        return self._base_cache_root

    @property
    def allowed_roots(self) -> list[Path]:
        """Roots that repository read tools may safely resolve below."""
        roots = [self._base_cache_root]
        if self.config.research.repository_cache_dir:
            roots.append(self.resolve_cache_dir(self.config.research.repository_cache_dir))
        override = self._cache_root_override.get()
        if override is not None:
            roots.append(override)
        if self.vfs is not None and self.vfs.enabled:
            roots.append(self.vfs.root)
        return roots

    @contextmanager
    def bind_vfs_identity(self, identity: VFSIdentity) -> Any:
        token = self._scope.set(identity)
        try:
            yield
        finally:
            self._scope.reset(token)

    @contextmanager
    def bind_cache_root(self, cache_root: Path) -> Any:
        resolved = cache_root.resolve(strict=False)
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise RepositoryError("仓库缓存路径越界。")
        token = self._cache_root_override.set(resolved)
        try:
            yield
        finally:
            self._cache_root_override.reset(token)

    @staticmethod
    def extract_url(text: str) -> str | None:
        match = _GITHUB_URL_RE.search(text)
        return match.group(0).rstrip(".,;:!?)]}") if match else None

    def parse_url(self, raw_url: str) -> ParsedRepositoryURL:
        parsed = urlsplit(raw_url.strip())
        if parsed.scheme != "https":
            raise RepositoryError("只允许通过 HTTPS 加载 GitHub 仓库。")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RepositoryError("GitHub URL 不能包含凭证、查询参数或片段。")
        host = (parsed.hostname or "").casefold()
        allowed = {value.casefold() for value in self.config.allowed_hosts}
        if host not in allowed:
            raise RepositoryError(f"仓库主机 `{host or '(empty)'}` 不在允许列表中。")
        segments = [segment for segment in parsed.path.split("/") if segment]
        if len(segments) != 2:
            raise RepositoryError("请输入仓库根地址，例如 https://github.com/owner/repo。")
        owner, name = segments
        if name.endswith(".git"):
            name = name[:-4]
        if (
            not _SEGMENT_RE.fullmatch(owner)
            or not _SEGMENT_RE.fullmatch(name)
            or owner in {".", ".."}
            or name in {".", ".."}
        ):
            raise RepositoryError("GitHub owner 或仓库名称不合法。")
        return ParsedRepositoryURL(
            url=f"https://{host}/{owner}/{name}",
            owner=owner,
            name=name,
        )

    def _repo_root(self, parsed: ParsedRepositoryURL) -> Path:
        root = (self.cache_root / parsed.owner.casefold() / parsed.name.casefold()).resolve()
        if self.cache_root not in root.parents:
            raise RepositoryError("仓库缓存路径越界。")
        return root

    def cache_path(self, record: RepositoryRecord) -> Path:
        """Return the validated cache root that owns a repository source."""
        parsed = self.parse_url(record.url)
        root = self._repo_root(parsed)
        expected_source = (root / "source").resolve()
        if record.path.resolve() != expected_source:
            raise RepositoryError("仓库记录与缓存路径不一致。")
        return root

    @staticmethod
    def _contains_selector(text: str, selector: str) -> bool:
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(selector)}(?![A-Za-z0-9_.-])",
                text,
                re.IGNORECASE,
            )
        )

    def has_cached_reference(self, text: str) -> bool:
        """Cheap synchronous hint used only for top-level request routing."""
        lowered = text.casefold()
        if not self.cache_root.is_dir():
            return False
        try:
            owner_roots = list(self.cache_root.iterdir())
        except OSError:
            return False
        for owner_root in owner_roots:
            try:
                repository_roots = list(owner_root.iterdir())
            except OSError:
                continue
            for repository_root in repository_roots:
                if (
                    not owner_root.is_dir()
                    or not (repository_root / "source" / ".git").is_dir()
                ):
                    continue
                selectors = (
                    repository_root.name.casefold(),
                    f"{owner_root.name}/{repository_root.name}".casefold(),
                )
                if any(self._contains_selector(lowered, value) for value in selectors):
                    return True
        return False

    @staticmethod
    async def _run_git(
        args: list[str],
        *,
        cwd: Path,
        timeout: int = 120,
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            **(env_overrides or {}),
        }
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RepositoryError("未找到 Git，请先安装 Git 并加入 PATH。") from exc
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise RepositoryError("Git 操作超时。") from None
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RepositoryError(f"Git 操作失败：{detail[-1500:]}")
        return stdout.decode("utf-8", errors="replace").strip()

    def _measure(self, source: Path) -> tuple[int, int]:
        size_bytes = 0
        for root, _directories, files in os.walk(source, followlinks=False):
            root_path = Path(root)
            for name in files:
                try:
                    size_bytes += (root_path / name).lstat().st_size
                except OSError:
                    continue
                if size_bytes > self.config.max_repository_size_mb * 1024 * 1024:
                    raise RepositoryError(
                        f"仓库大小超过限制 {self.config.max_repository_size_mb} MB。"
                    )
        file_count = 0
        for root, directories, files in os.walk(source, followlinks=False):
            directories[:] = [name for name in directories if name != ".git"]
            for name in files:
                file_count += 1
                if file_count > self.config.max_files:
                    raise RepositoryError(
                        f"仓库文件数超过限制 {self.config.max_files:,}。"
                    )
        return file_count, size_bytes

    async def _record_from_source(
        self,
        parsed: ParsedRepositoryURL,
        source: Path,
        *,
        measure: bool,
    ) -> RepositoryRecord:
        origin = await self._run_git(["remote", "get-url", "origin"], cwd=source)
        normalized_origin = origin.removesuffix(".git").rstrip("/").casefold()
        if normalized_origin != parsed.url.casefold():
            raise RepositoryError("缓存仓库的 origin 与请求 URL 不一致，已停止加载。")
        commit = await self._run_git(["rev-parse", "HEAD"], cwd=source)
        branch = await self._run_git(
            ["symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=source,
        )
        file_count, size_bytes = self._measure(source) if measure else (0, 0)
        repository_id = hashlib.sha256(
            parsed.full_name.casefold().encode()
        ).hexdigest()[:20]
        return RepositoryRecord(
            repository_id=repository_id,
            url=parsed.url,
            owner=parsed.owner,
            name=parsed.name,
            default_branch=branch,
            commit_sha=commit,
            local_path=str(source.resolve()),
            file_count=file_count,
            size_bytes=size_bytes,
        )

    async def list_cached_repositories(self) -> list[RepositoryRecord]:
        """Discover valid downloaded repositories without cloning anything."""
        records: list[RepositoryRecord] = []
        if not self.cache_root.is_dir():
            return records
        for source in sorted(self.cache_root.glob("*/*/source")):
            if not (source / ".git").is_dir():
                continue
            try:
                origin = await self._run_git(["remote", "get-url", "origin"], cwd=source)
                parsed = self.parse_url(origin)
                expected = (self._repo_root(parsed) / "source").resolve()
                if source.resolve() != expected:
                    continue
                records.append(
                    await self._record_from_source(parsed, source, measure=False)
                )
            except RepositoryError:
                continue
        return records

    async def list_all_cached_repositories(self) -> list[RepositoryRecord]:
        """Read-only listing of downloaded repos, including per-session research downloads."""
        records: list[RepositoryRecord] = []
        if not self.cache_root.is_dir():
            return records
        sources = sorted(
            {*self.cache_root.glob("*/*/source"), *self.cache_root.glob("*/*/*/source")}
        )
        for source in sources:
            if not (source / ".git").is_dir():
                continue
            try:
                origin = await self._run_git(["remote", "get-url", "origin"], cwd=source)
                parsed = self.parse_url(origin)
                if (
                    source.parent.name.casefold() != parsed.name.casefold()
                    or source.parent.parent.name.casefold() != parsed.owner.casefold()
                ):
                    continue
                records.append(
                    await self._record_from_source(parsed, source, measure=False)
                )
            except RepositoryError:
                continue
        return records

    async def list_vfs_repositories(self) -> list[RepositoryRecord]:
        """List every repository across all tenants/users under the VFS root."""
        records: list[RepositoryRecord] = []
        if self.vfs is None or not self.vfs.enabled:
            return records
        root = self.vfs.root
        if not root.is_dir():
            return records
        sources = sorted(root.glob("tenants/*/users/*/projects/*/*/source"))
        for source in sources:
            if not (source / ".git").is_dir():
                continue
            try:
                origin = await self._run_git(["remote", "get-url", "origin"], cwd=source)
                parsed = self.parse_url(origin)
                if (
                    source.parent.name.casefold() != parsed.name.casefold()
                    or source.parent.parent.name.casefold() != parsed.owner.casefold()
                ):
                    continue
                records.append(
                    await self._record_from_source(parsed, source, measure=False)
                )
            except RepositoryError:
                continue
        return records

    async def commit_sha_at(self, source_path: str | Path) -> str:
        """Return the current HEAD commit of a cloned source directory."""
        output = await self._run_git(["rev-parse", "HEAD"], cwd=Path(source_path))
        return output.strip()

    def delete_repository_source(self, source_path: str | Path) -> Path:
        """Validate and remove the cache root owning a cloned source directory."""
        source = Path(source_path).resolve()
        if source.name != "source":
            raise RepositoryError("仓库缓存路径异常。")
        root = source.parent
        allowed = [self._base_cache_root]
        if self.vfs is not None and self.vfs.enabled:
            allowed.append(self.vfs.root)
        if not any(base == root or base in root.parents for base in allowed):
            raise RepositoryError("仓库缓存路径越界。")
        self._remove_tree(root)
        return root

    async def resolve_cached_repository(
        self,
        request: str,
        *,
        active: RepositoryRecord | None = None,
    ) -> RepositoryRecord:
        """Resolve one explicit natural-language target from downloaded caches."""
        records = await self.list_cached_repositories()
        if not records:
            raise RepositoryError("当前没有已下载的项目。")

        if url := self.extract_url(request):
            parsed = self.parse_url(url)
            matches = [
                record
                for record in records
                if record.full_name.casefold() == parsed.full_name.casefold()
            ]
        else:
            full_matches = [
                record
                for record in records
                if self._contains_selector(request, record.full_name)
            ]
            name_matches = [
                record
                for record in records
                if self._contains_selector(request, record.name)
            ]
            matches = full_matches or name_matches

        if not matches and active is not None and re.search(
            r"(?:当前|这个|该|本次|current|this)\s*(?:项目|仓库|repo(?:sitory)?)?",
            request,
            re.IGNORECASE,
        ):
            matches = [
                record
                for record in records
                if record.repository_id == active.repository_id
            ]

        unique = {record.repository_id: record for record in matches}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if len(unique) > 1:
            choices = "、".join(sorted(record.full_name for record in unique.values()))
            raise RepositoryError(f"项目名称不唯一，请使用 owner/repo 指定：{choices}")
        available = "、".join(sorted(record.full_name for record in records))
        raise RepositoryError(
            f"没有定位到要删除的已下载项目。请指定 owner/repo；当前项目：{available}"
        )

    async def delete_cached_repository(
        self,
        approved: RepositoryRecord,
    ) -> Path:
        """Delete exactly one revalidated repository cache root."""
        current = await self.resolve_cached_repository(approved.url)
        if (
            current.repository_id != approved.repository_id
            or current.full_name.casefold() != approved.full_name.casefold()
            or current.commit_sha != approved.commit_sha
            or current.path.resolve() != approved.path.resolve()
        ):
            raise RepositoryError("仓库缓存自审批后发生变化，旧删除审批已失效。")
        root = self.cache_path(current)
        if not root.is_dir():
            raise RepositoryError("目标仓库缓存已不存在。")
        self._remove_tree(root)
        return root

    def _remove_created_cache(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved == self.cache_root or self.cache_root not in resolved.parents:
            raise RepositoryError("拒绝清理未验证的缓存路径。")
        if resolved.exists():
            self._remove_tree(resolved)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        def make_writable_and_retry(
            function: Any,
            target: str,
            _error: Any,
        ) -> None:
            if not os.path.islink(target):
                os.chmod(target, os.lstat(target).st_mode | stat.S_IWRITE)
            function(target)

        shutil.rmtree(path, onerror=make_writable_and_retry)

    async def load(self, raw_url: str) -> RepositoryRecord:
        parsed = self.parse_url(raw_url)
        repo_root = self._repo_root(parsed)
        source = repo_root / "source"
        created = False
        if not (source / ".git").is_dir():
            repo_root.mkdir(parents=True, exist_ok=True)
            try:
                await self._run_git(
                    [
                        "-c",
                        "protocol.file.allow=never",
                        "clone",
                        "--depth",
                        str(self.config.clone_depth),
                        "--single-branch",
                        "--no-tags",
                        "--",
                        f"{parsed.url}.git",
                        str(source),
                    ],
                    cwd=repo_root,
                )
                created = True
            except Exception:
                self._remove_created_cache(source)
                raise
        try:
            return await self._record_from_source(parsed, source, measure=True)
        except RepositoryError:
            if created:
                self._remove_created_cache(source)
            raise

    async def create_task_workspace(
        self,
        record: RepositoryRecord,
        *,
        session_key: str,
        objective: str,
    ) -> RepositoryTask:
        task = RepositoryTask(
            session_key=session_key,
            repository_id=record.repository_id,
            repository_full_name=record.full_name,
            source_url=record.url,
            base_branch=record.default_branch,
            base_commit=record.commit_sha,
            objective=objective,
            workspace_path="",
        )
        repo_root = self._repo_root(self.parse_url(record.url))
        task_root = (repo_root / "tasks" / task.task_id).resolve()
        if self.cache_root not in task_root.parents:
            raise RepositoryError("任务工作区路径越界。")
        task_root.parent.mkdir(parents=True, exist_ok=True)
        await self._run_git(
            [
                "-c",
                "protocol.file.allow=always",
                "clone",
                "--local",
                "--no-hardlinks",
                "--",
                str(record.path),
                str(task_root),
            ],
            cwd=task_root.parent,
        )
        await self._run_git(["remote", "set-url", "origin", f"{record.url}.git"], cwd=task_root)
        task.workspace_path = str(task_root)
        return task

    async def diff(self, task: RepositoryTask) -> str:
        return await read_worktree_diff(
            task.workspace,
            base_commit=task.base_commit,
        )

    async def diff_hash(self, task: RepositoryTask) -> str:
        diff = await self.diff(task)
        # Git's ``index old..new`` line may use the worktree blob before a
        # commit and the normalized index blob after it (notably with
        # core.autocrlf). The patch content is unchanged, so exclude only that
        # transport metadata while retaining paths, modes, hunks, and binary
        # patch data in the approval hash.
        canonical = "\n".join(
            line for line in diff.splitlines() if not line.startswith("index ")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def write_patch(self, task: RepositoryTask) -> Path:
        diff = await self.diff(task)
        patch = (task.workspace / f"{task.task_id}.patch").resolve()
        if task.workspace.resolve() not in patch.parents:
            raise RepositoryError("补丁输出路径越界。")
        patch.write_text(diff, encoding="utf-8", newline="\n")
        return patch


async def read_worktree_diff(
    workspace: Path,
    *,
    base_commit: str = "HEAD",
    max_chars: int | None = None,
) -> str:
    """Return a worktree diff, including untracked files, without changing the real index."""
    workspace = workspace.resolve()
    with tempfile.TemporaryDirectory(prefix="nanobot-repo-diff-") as temp_dir:
        temporary_index = Path(temp_dir) / "index"
        isolated_index = {"GIT_INDEX_FILE": str(temporary_index)}
        await RepositoryManager._run_git(
            ["read-tree", base_commit],
            cwd=workspace,
            env_overrides=isolated_index,
        )
        await RepositoryManager._run_git(
            ["add", "--intent-to-add", "--all"],
            cwd=workspace,
            env_overrides=isolated_index,
        )
        output = await RepositoryManager._run_git(
            [
                "diff",
                "--binary",
                "--no-ext-diff",
                base_commit,
                "--",
            ],
            cwd=workspace,
            env_overrides=isolated_index,
        )
    if max_chars is not None and len(output) > max_chars:
        return output[:max_chars]
    return output
