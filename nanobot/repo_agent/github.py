"""Deterministic GitHub Fork, Push, and Draft PR publisher."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from nanobot.repo_agent.config import GitHubWorkflowConfig
from nanobot.repo_agent.models import PublishAction, PublishApproval, RepositoryTask
from nanobot.repo_agent.repository import RepositoryManager


class GitHubPublishError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PublishResult:
    fork_url: str
    branch: str
    commit_sha: str
    pull_request_url: str | None = None


class GitHubPublisher:
    def __init__(
        self,
        config: GitHubWorkflowConfig,
        repository_manager: RepositoryManager,
    ):
        self.config = config
        self.repository_manager = repository_manager

    @staticmethod
    def branch_name(prefix: str, task: RepositoryTask) -> str:
        words = re.findall(r"[A-Za-z0-9]+", task.objective.casefold())
        slug = "-".join(words[:7])[:50] or task.task_id
        branch = f"{prefix.strip('/')}/{slug}-{task.task_id[-6:]}"
        if not re.fullmatch(r"[A-Za-z0-9._/-]{1,100}", branch):
            raise GitHubPublishError("generated branch name is invalid")
        return branch

    @staticmethod
    async def _run(
        executable: str,
        args: list[str],
        *,
        cwd: Path,
        stdin: str | None = None,
        timeout: int = 120,
    ) -> str:
        env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GH_PROMPT_DISABLED": "1",
        }
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *args,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise GitHubPublishError(f"未找到 `{executable}` 命令。") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin.encode("utf-8") if stdin is not None else None),
                timeout=timeout,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise GitHubPublishError(f"{executable} 操作超时。") from None
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise GitHubPublishError(f"{executable} 操作失败：{detail[-2000:]}")
        return stdout.decode("utf-8", errors="replace").strip()

    async def current_user(self, cwd: Path) -> str:
        if shutil.which("gh") is None:
            raise GitHubPublishError("未安装 GitHub CLI，请先安装并执行 `gh auth login`。")
        value = await self._run("gh", ["api", "user", "--jq", ".login"], cwd=cwd)
        if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", value):
            raise GitHubPublishError("无法确定当前 GitHub 用户。")
        return value

    async def _ensure_fork(
        self,
        task: RepositoryTask,
        owner: str,
        current_user: str,
    ) -> str:
        endpoint = f"repos/{owner}/{task.repository_full_name.split('/', 1)[1]}"
        try:
            raw = await self._run("gh", ["api", endpoint], cwd=task.workspace)
            existing_value: object = json.loads(raw)
            if not isinstance(existing_value, dict):
                raise GitHubPublishError("GitHub 仓库响应不是对象。")
            existing = cast(dict[str, Any], existing_value)
            parent_value: object = existing.get("parent")
            parent = (
                cast(dict[str, Any], parent_value)
                if isinstance(parent_value, dict)
                else {}
            )
            if (
                str(parent.get("full_name", "")).casefold()
                != task.repository_full_name.casefold()
            ):
                raise GitHubPublishError(
                    f"`{endpoint.removeprefix('repos/')}` 已存在但不是目标仓库的 Fork。"
                )
            return str(existing["html_url"])
        except GitHubPublishError as exc:
            if "HTTP 404" not in str(exc):
                raise
        payload: dict[str, Any] = {
            "default_branch_only": self.config.default_branch_only,
        }
        if owner.casefold() != current_user.casefold():
            payload["organization"] = owner
        await self._run(
            "gh",
            [
                "api",
                "--method",
                "POST",
                f"repos/{task.repository_full_name}/forks",
                "--input",
                "-",
            ],
            cwd=task.workspace,
            stdin=json.dumps(payload),
        )
        for _ in range(45):
            await asyncio.sleep(1)
            try:
                raw = await self._run("gh", ["api", endpoint], cwd=task.workspace, timeout=20)
                created = json.loads(raw)
                return str(created["html_url"])
            except GitHubPublishError:
                continue
        raise GitHubPublishError("Fork 已提交，但等待 GitHub 初始化超时。")

    async def _prepare_commit(
        self,
        task: RepositoryTask,
        approval: PublishApproval,
    ) -> str:
        current_hash = await self.repository_manager.diff_hash(task)
        if current_hash != approval.diff_hash:
            raise GitHubPublishError("代码 Diff 在审批后发生变化，旧审批已失效。")
        status = await self._run(
            "git",
            ["status", "--porcelain=v1"],
            cwd=task.workspace,
        )
        current_branch = await self._run(
            "git",
            ["branch", "--show-current"],
            cwd=task.workspace,
        )
        if current_branch != approval.head_branch:
            branches = await self._run(
                "git",
                ["branch", "--list", approval.head_branch],
                cwd=task.workspace,
            )
            if branches.strip():
                await self._run("git", ["switch", approval.head_branch], cwd=task.workspace)
            else:
                await self._run(
                    "git",
                    ["switch", "-c", approval.head_branch],
                    cwd=task.workspace,
                )
        if status:
            await self._run(
                "git",
                ["config", "user.name", self.config.commit_author_name],
                cwd=task.workspace,
            )
            await self._run(
                "git",
                ["config", "user.email", self.config.commit_author_email],
                cwd=task.workspace,
            )
            await self._run("git", ["add", "--all"], cwd=task.workspace)
            await self._run(
                "git",
                ["commit", "-m", approval.title],
                cwd=task.workspace,
            )
        return await self._run("git", ["rev-parse", "HEAD"], cwd=task.workspace)

    async def publish(
        self,
        task: RepositoryTask,
        approval: PublishApproval,
    ) -> PublishResult:
        if not approval.draft and self.config.draft_pr_only:
            raise GitHubPublishError("当前策略只允许创建 Draft PR。")
        if (
            approval.task_id != task.task_id
            or approval.upstream.casefold() != task.repository_full_name.casefold()
            or approval.base_branch != task.base_branch
            or approval.base_commit != task.base_commit
        ):
            raise GitHubPublishError("审批目标与当前代码任务不一致。")
        current_hash = await self.repository_manager.diff_hash(task)
        if current_hash != approval.diff_hash:
            raise GitHubPublishError("代码 Diff 在审批后发生变化，旧审批已失效。")
        current_user = await self.current_user(task.workspace)
        owner = approval.fork_owner or current_user
        fork_url = await self._ensure_fork(task, owner, current_user)
        commit_sha = await self._prepare_commit(task, approval)
        fork_repo = f"{owner}/{task.repository_full_name.split('/', 1)[1]}"
        origin = f"https://github.com/{fork_repo}.git"
        await self._run("git", ["remote", "set-url", "origin", origin], cwd=task.workspace)
        await self._run(
            "git",
            [
                "-c",
                "credential.helper=!gh auth git-credential",
                "push",
                "--set-upstream",
                "origin",
                approval.head_branch,
            ],
            cwd=task.workspace,
        )
        pr_url: str | None = None
        if approval.action == PublishAction.DRAFT_PR:
            existing_raw = await self._run(
                "gh",
                [
                    "api",
                    "--method",
                    "GET",
                    f"repos/{task.repository_full_name}/pulls",
                    "-f",
                    "state=open",
                    "-f",
                    f"head={owner}:{approval.head_branch}",
                    "-f",
                    f"base={approval.base_branch}",
                ],
                cwd=task.workspace,
            )
            existing = json.loads(existing_raw)
            if existing:
                pr_url = str(existing[0]["html_url"])
            else:
                payload = {
                    "title": approval.title,
                    "head": f"{owner}:{approval.head_branch}",
                    "base": approval.base_branch,
                    "body": approval.body,
                    "draft": True,
                    "maintainer_can_modify": True,
                }
                raw = await self._run(
                    "gh",
                    [
                        "api",
                        "--method",
                        "POST",
                        f"repos/{task.repository_full_name}/pulls",
                        "--input",
                        "-",
                    ],
                    cwd=task.workspace,
                    stdin=json.dumps(payload, ensure_ascii=False),
                )
                pr_url = str(json.loads(raw)["html_url"])
        return PublishResult(
            fork_url=fork_url,
            branch=approval.head_branch,
            commit_sha=commit_sha,
            pull_request_url=pr_url,
        )
