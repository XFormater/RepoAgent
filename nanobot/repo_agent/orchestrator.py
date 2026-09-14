"""Top-level repository analysis, code-task, and publishing orchestration."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.repo_agent.agents import (
    RepositoryControlClient,
    RepositoryDispatcherAgent,
    RepositoryIntentAgent,
    RepositoryPlannerAgent,
)
from nanobot.repo_agent.config import RepositoryAgentConfig
from nanobot.repo_agent.executor import (
    RepositoryAgentInvoker,
    RepositoryExecutionRecord,
    RepositoryPlanExecutor,
)
from nanobot.repo_agent.github import GitHubPublisher, GitHubPublishError
from nanobot.repo_agent.models import (
    ApprovalStatus,
    PublishAction,
    PublishApproval,
    RepositoryDeletionApproval,
    RepositoryRecord,
    RepositoryTask,
    ResearchStatus,
    TaskComplexity,
    TaskStatus,
    utc_now,
)
from nanobot.repo_agent.policy import RepositoryRiskPolicy
from nanobot.repo_agent.registry import RepositoryExpertRegistry
from nanobot.repo_agent.repository import RepositoryError, RepositoryManager
from nanobot.repo_agent.research import RepositoryResearchOrchestrator
from nanobot.repo_agent.store import RepositoryStateStore
from nanobot.security.logical_vfs import LogicalVFS, VFSIdentity
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.harness import HarnessService


@dataclass(slots=True)
class RepositoryTurnResult:
    final_content: str
    tools_used: list[str]
    messages: list[dict[str, Any]]
    stop_reason: str = "completed"
    had_injections: bool = False
    reset_provider_state: bool = False
    agent_ui: dict[str, Any] | None = None
    deferred_to_worker: bool = False


class RepositoryOrchestrator:
    _SECRET_RE = re.compile(
        r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
        r"(?i:bearer)\s+[A-Za-z0-9._~+/=-]{16,})"
    )
    _COMMAND_RE = re.compile(
        r"^/repo(?:\s+(?P<action>close|save|patch|fork|pr|status))?(?:\s+(?P<arg>\S+))?\s*$",
        re.IGNORECASE,
    )
    _RESEARCH_COMMAND_RE = re.compile(
        r"^/rere(?:\s+(?P<query>[\s\S]+?))?\s*$",
        re.IGNORECASE,
    )
    _REGET_COMMAND_RE = re.compile(
        r"^/reget\s*$",
        re.IGNORECASE,
    )
    _APPROVE_RE = re.compile(
        r"^(?:/approve|/repo\s+approve|确认提交|确认创建)\s+([A-Za-z0-9_-]{1,100})\s*$",
        re.IGNORECASE,
    )
    _REJECT_RE = re.compile(
        r"^(?:/reject|/repo\s+reject|拒绝提交|取消创建)\s+([A-Za-z0-9_-]{1,100})\s*$",
        re.IGNORECASE,
    )
    _DELETE_APPROVE_RE = re.compile(
        r"^(?:/approve-delete|/repo\s+approve-delete|确认删除)\s+"
        r"([A-Za-z0-9_-]{1,100})\s*$",
        re.IGNORECASE,
    )
    _DELETE_REJECT_RE = re.compile(
        r"^(?:/reject-delete|/repo\s+reject-delete|取消删除|拒绝删除)\s+"
        r"([A-Za-z0-9_-]{1,100})\s*$",
        re.IGNORECASE,
    )
    _REDEL_COMMAND_RE = re.compile(
        r"^/redel(?:\s+(?P<target>\S+))?\s*$",
        re.IGNORECASE,
    )
    _DELETE_VERB_RE = re.compile(
        r"(?:删除|刪除|删掉|刪掉|移除|清除|delete|remove)",
        re.IGNORECASE,
    )
    _DELETE_OBJECT_RE = re.compile(
        r"(?:项目|項目|仓库|倉庫|代码库|代碼庫|repo(?:sitory)?|project)",
        re.IGNORECASE,
    )
    _LOCAL_DELETE_HINT_RE = re.compile(
        r"(?:下载|下載|本地|缓存|快取|cache|downloaded|local)",
        re.IGNORECASE,
    )
    _REMOTE_DELETE_HINT_RE = re.compile(
        r"(?:github\s*(?:上|远程|遠端)|远程仓库|遠端倉庫|remote\s+repo)",
        re.IGNORECASE,
    )
    _LOCAL_CACHE_HINT_RE = re.compile(
        r"(?:已下载|已下載|下载过|下载的|本地|缓存|快取|现有|已有|"
        r"downloaded|cached|local|existing)",
        re.IGNORECASE,
    )
    _LOCAL_LIST_ACTION_RE = re.compile(
        r"(?:整理|列出|列举|清单|汇总|总结|盘点|概览|有哪些|"
        r"list|show|summari[sz]e|organize|inventory|enumerate)",
        re.IGNORECASE,
    )
    _RESEARCH_ACTION_RE = re.compile(
        r"(?:寻找|查找|搜索|检索|推荐|调研|研究|比较|对比|筛选|下载|克隆|整理|"
        r"总结|汇总|盘点|评估|find|search|discover|recommend|survey|research|"
        r"compare|download|clone|summari[sz]e|curate|shortlist|evaluate)",
        re.IGNORECASE,
    )
    _RESEARCH_TARGET_RE = re.compile(
        r"(?:github|开源|开放源代码|项目|仓库|代码库|实现|"
        r"open[\s-]*source|repositories|repository|repos?|projects?|implementations?)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        config: RepositoryAgentConfig,
        workspace: Path,
        tools: ToolRegistry,
        harness: HarnessService | None = None,
        vfs: LogicalVFS | None = None,
    ):
        self.config = config
        self.tools = tools
        self.manager = RepositoryManager(workspace, config, vfs=vfs)
        self.store = RepositoryStateStore(workspace, config.state_dir)
        self.registry = RepositoryExpertRegistry(config, tools)
        self.client = RepositoryControlClient()
        self.intent_agent = RepositoryIntentAgent(config, self.registry, self.client)
        self.dispatcher_agent = RepositoryDispatcherAgent(self.registry)
        self.planner_agent = RepositoryPlannerAgent(
            config,
            self.registry,
            self.client,
        )
        self.harness = None
        if harness is not None:
            from nanobot.repo_agent.harness import RepositoryHarnessCoordinator

            self.harness = RepositoryHarnessCoordinator(harness)
        self.executor = RepositoryPlanExecutor(
            self.registry,
            tools,
            self.store,
            harness=self.harness,
        )
        self.policy = RepositoryRiskPolicy()
        self.publisher = GitHubPublisher(config.github, self.manager)
        self.research = RepositoryResearchOrchestrator(
            config=config,
            manager=self.manager,
            store=self.store,
            tools=tools,
            client=self.client,
            harness=self.harness,
        )

    def is_research_request(self, request: str) -> bool:
        # Organizing/listing repositories already downloaded to the local cache
        # is not GitHub discovery/download research.
        if self.is_local_cache_list_request(request):
            return False
        return bool(
            self.config.research.enabled
            and not self.manager.extract_url(request)
            and self._RESEARCH_ACTION_RE.search(request)
            and self._RESEARCH_TARGET_RE.search(request)
        )

    def is_local_cache_list_request(self, request: str) -> bool:
        """Whether the request asks to list/summarize already-downloaded repos."""
        return bool(
            self._LOCAL_CACHE_HINT_RE.search(request)
            and self._LOCAL_LIST_ACTION_RE.search(request)
            and not self.manager.extract_url(request)
        )

    def can_handle(self, request: str, session_key: str) -> bool:
        stripped = request.strip()
        return bool(
            self.manager.extract_url(request)
            or self.is_research_request(request)
            or self._RESEARCH_COMMAND_RE.fullmatch(stripped)
            or self._REGET_COMMAND_RE.fullmatch(stripped)
            or self._COMMAND_RE.fullmatch(stripped)
            or self._APPROVE_RE.fullmatch(stripped)
            or self._REJECT_RE.fullmatch(stripped)
            or self._DELETE_APPROVE_RE.fullmatch(stripped)
            or self._DELETE_REJECT_RE.fullmatch(stripped)
            or self._REDEL_COMMAND_RE.fullmatch(stripped)
            or self._is_local_delete_request(request, session_key)
            or (
                self._DELETE_VERB_RE.search(request)
                and self._DELETE_OBJECT_RE.search(request)
            )
            or self.store.active_repository(session_key)
        )

    def is_command_request(self, request: str) -> bool:
        """Return whether a slash command belongs to the repository workflow."""
        stripped = request.strip()
        return bool(
            self._RESEARCH_COMMAND_RE.fullmatch(stripped)
            or self._REGET_COMMAND_RE.fullmatch(stripped)
            or self._COMMAND_RE.fullmatch(stripped)
            or self._APPROVE_RE.fullmatch(stripped)
            or self._REJECT_RE.fullmatch(stripped)
            or self._DELETE_APPROVE_RE.fullmatch(stripped)
            or self._DELETE_REJECT_RE.fullmatch(stripped)
            or self._REDEL_COMMAND_RE.fullmatch(stripped)
        )

    def _is_local_delete_request(self, request: str, session_key: str) -> bool:
        if not self._DELETE_VERB_RE.search(request):
            return False
        if self._REMOTE_DELETE_HINT_RE.search(request):
            return False
        if self.manager.extract_url(request):
            return bool(self._LOCAL_DELETE_HINT_RE.search(request))
        if self._LOCAL_DELETE_HINT_RE.search(request):
            return True
        if self.manager.has_cached_reference(request):
            return True
        return bool(
            self._DELETE_OBJECT_RE.search(request)
            and self.store.active_repository(session_key)
        )

    @staticmethod
    def _synthetic(
        initial_messages: list[dict[str, Any]],
        content: str,
        *,
        stop_reason: str = "completed",
        agent_ui: dict[str, Any] | None = None,
        deferred_to_worker: bool = False,
    ) -> RepositoryTurnResult:
        return RepositoryTurnResult(
            final_content=content,
            tools_used=[],
            messages=(
                list(initial_messages)
                if deferred_to_worker
                else [*initial_messages, {"role": "assistant", "content": content}]
            ),
            stop_reason=stop_reason,
            reset_provider_state=True,
            agent_ui=agent_ui,
            deferred_to_worker=deferred_to_worker,
        )

    @staticmethod
    def _merge_system(
        messages: list[dict[str, Any]],
        instruction: str,
    ) -> list[dict[str, object]]:
        merged: list[dict[str, object]] = [dict(message) for message in messages]
        for message in merged:
            if message.get("role") == "system":
                existing = message.get("content")
                message["content"] = (
                    f"{existing}\n\n{instruction}" if existing else instruction
                )
                return merged
        return [{"role": "system", "content": instruction}, *merged]

    def close_session(self, session_key: str) -> None:
        self.store.clear_active_repository(session_key)

    async def cancel_research(self, session_key: str, *, tenant_id: str = "default") -> int:
        """Cancel active POOL research tasks owned by a WebSocket session."""
        if self.harness is None:
            return 0
        cancelled = 0
        for research in self.store.active_research(session_key):
            harness_task_id = research.harness_task_id
            # Recover the root task for records written by an older Gateway
            # just before the explicit harness_task_id projection was added.
            if not harness_task_id and research.workflow_id:
                try:
                    root = await self.harness.service.store.get_task_by_idempotency(
                        research.workflow_id,
                        f"repository-research:run:{research.research_id}",
                    )
                    harness_task_id = root.task_id if root is not None else None
                except Exception:
                    logger.exception(
                        "Could not resolve Harness task for research {}",
                        research.research_id,
                    )
            if not harness_task_id:
                continue
            try:
                changed = await self.harness.service.cancel_task(
                    harness_task_id,
                    tenant_id=tenant_id,
                )
            except Exception:
                logger.exception("Could not cancel research task {}", research.research_id)
                continue
            if changed:
                research.status = ResearchStatus.CANCELLED
                self.store.save_research(research, event="repository_research.cancelled")
                cancelled += 1
        return cancelled

    @staticmethod
    def _repository_summary(record: RepositoryRecord) -> str:
        return (
            f"已加载 `{record.full_name}`。\n\n"
            f"- 分支：`{record.default_branch}`\n"
            f"- Commit：`{record.commit_sha}`\n"
            f"- 文件数：{record.file_count:,}\n"
            f"- 本地只读缓存：`{record.local_path}`"
        )

    @staticmethod
    def _with_repository_path(
        result: RepositoryTurnResult,
        record: RepositoryRecord,
    ) -> RepositoryTurnResult:
        footer = f"项目本地保存路径：`{record.local_path}`"
        if footer in result.final_content:
            return result
        result.final_content = f"{result.final_content.rstrip()}\n\n---\n{footer}"
        for message in reversed(result.messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and footer not in content:
                message["content"] = f"{content.rstrip()}\n\n---\n{footer}"
            break
        return result

    @staticmethod
    def _save_choices(task: RepositoryTask) -> str:
        return (
            "\n\n代码当前只保存在隔离工作区，尚未写入 GitHub：\n"
            f"- 任务：`{task.task_id}`\n"
            f"- 工作区：`{task.workspace_path}`\n\n"
            "可选操作：\n"
            "- `/repo save`：保留本地工作区\n"
            "- `/repo patch`：生成 Patch\n"
            "- `/repo fork`：Fork 并 Push 任务分支\n"
            "- `/repo pr`：Fork、Push 并创建 Draft PR"
        )

    async def _run_simple(
        self,
        *,
        request: str,
        record: RepositoryRecord,
        intent: Any,
        initial_messages: list[dict[str, Any]],
        runtime: LLMRuntime,
        invoke: RepositoryAgentInvoker,
    ) -> RepositoryTurnResult:
        expert = self.dispatcher_agent.dispatch(intent)
        if expert is None:
            return self._synthetic(initial_messages, "没有可处理该仓库任务的专家。")
        expert_tools = self.tools.subset(expert.tools)
        context = {
            "repository": record.full_name,
            "url": record.url,
            "branch": record.default_branch,
            "commit": record.commit_sha,
            "workspace": record.local_path,
            "mode": "simple_react",
            "issue_number": intent.issue_number,
        }
        system = render_template(
            "repo_agent/expert.md",
            agent_name=expert.name,
            description=expert.description,
            repository_context=json.dumps(context, ensure_ascii=False),
            instruction=request,
        )
        messages = self._merge_system(initial_messages, system)
        result = await invoke(
            messages,
            runtime=runtime,
            tools=expert_tools,
            max_iterations=min(
                expert.max_iterations,
                self.config.simple_max_iterations,
            ),
            ephemeral=False,
            stream=True,
            workspace=record.path,
            repository_full_name=record.full_name,
        )
        return RepositoryTurnResult(
            final_content=result.final_content,
            tools_used=result.tools_used,
            messages=[dict(value) for value in result.messages],
            stop_reason=result.stop_reason,
            had_injections=result.had_injections,
        )

    async def _synthesize_task(
        self,
        request: str,
        task: RepositoryTask,
        records: list[RepositoryExecutionRecord],
        runtime: LLMRuntime,
    ) -> str:
        payload = {
            "task_id": task.task_id,
            "status": task.status.value,
            "repository": task.repository_full_name,
            "base_commit": task.base_commit,
            "workspace": task.workspace_path,
            "steps": [
                {
                    "step_id": step.step_id,
                    "agent": step.assigned_agent,
                    "status": step.status.value,
                    "result": step.result,
                    "error": step.error,
                }
                for step in task.steps
            ],
            "records": [
                {
                    "agent": item.agent,
                    "step_id": item.step_id,
                    "tools_used": item.tools_used,
                    "succeeded": item.succeeded,
                    "content": item.content,
                }
                for item in records
            ],
        }
        try:
            return await self.client.complete(
                render_template(
                    "repo_agent/synthesizer.md",
                    request=request,
                    execution_json=json.dumps(payload, ensure_ascii=False),
                ),
                runtime=runtime,
                max_tokens=2_000,
            )
        except Exception as exc:
            logger.warning("Repository task synthesizer failed: {}", exc)
            lines = [f"代码任务 `{task.task_id}` 状态：{task.status.value}", ""]
            for step in task.steps:
                lines.append(f"- {step.name}: {step.status.value}")
            return "\n".join(lines)

    async def _run_complex(
        self,
        *,
        request: str,
        record: RepositoryRecord,
        intent: Any,
        actor: str,
        session_key: str,
        initial_messages: list[dict[str, Any]],
        runtime: LLMRuntime,
        invoke: RepositoryAgentInvoker,
    ) -> RepositoryTurnResult:
        task = await self.manager.create_task_workspace(
            record,
            session_key=session_key,
            objective=request,
        )
        harness = self.harness
        planner_run = None
        if harness is not None:
            workflow, owner = await harness.workflow(
                scope_key=harness.code_scope(task.task_id),
                goal=request,
            )
            planner_task, planner_spec = await harness.planner_task(
                workflow,
                task_input=request,
                key=f"repository-plan:{task.task_id}",
            )
            planner_run = await harness.start(
                workflow,
                owner,
                planner_task,
                planner_spec,
                worker_id=f"repository-inline:{task.task_id}:planner",
            )
        try:
            task = await self.planner_agent.create_task(
                request,
                intent,
                record,
                task,
                runtime,
            )
            if planner_run is not None:
                assert harness is not None
                await harness.complete(
                    planner_run,
                    json.dumps(
                        {
                            "objective": task.objective,
                            "steps": [step.model_dump(mode="json") for step in task.steps],
                        },
                        ensure_ascii=False,
                    ),
                )
        except asyncio.CancelledError:
            if planner_run is not None:
                assert harness is not None
                await harness.cancel(planner_run)
            raise
        except Exception as exc:
            if planner_run is not None:
                assert harness is not None
                await harness.fail(planner_run, str(exc))
            raise
        self.store.save_task(task, event="repository_task.planned")
        self.store.append_audit(
            "repository_task.created",
            task_id=task.task_id,
            actor=actor,
            repository=record.full_name,
            base_commit=record.commit_sha,
        )
        records = await self.executor.execute(
            task,
            record,
            runtime=runtime,
            invoke=invoke,
        )
        if task.status == TaskStatus.READY_TO_SAVE:
            task.diff_hash = await self.manager.diff_hash(task)
            self.store.save_task(task, event="repository_task.diff_ready")
        final = await self._synthesize_task(request, task, records, runtime)
        if task.status == TaskStatus.READY_TO_SAVE:
            final += self._save_choices(task)
        return self._synthetic(initial_messages, final)

    def _approval_body(self, task: RepositoryTask) -> tuple[str, str]:
        safe_objective = self._SECRET_RE.sub("[REDACTED]", task.objective)
        title = f"nanobot: {safe_objective.strip()}"[:72].rstrip()
        step_lines = "\n".join(
            f"- {step.name}: {step.status.value}" for step in task.steps
        )
        body = (
            "## Summary\n\n"
            f"{safe_objective}\n\n"
            "## Workflow\n\n"
            f"{step_lines}\n\n"
            "## Validation\n\n"
            "See the Test Agent and Review Agent records in the nanobot task audit. "
            "This PR does not claim checks that were not actually executed.\n\n"
            "## Agent trace\n\n"
            f"- Task: `{task.task_id}`\n"
            f"- Base commit: `{task.base_commit}`\n"
            "- Generated by: nanobot repository multi-agent workflow\n"
        )
        return title, body

    async def _prepare_publish(
        self,
        *,
        task: RepositoryTask,
        action: PublishAction,
        actor: str,
        initial_messages: list[dict[str, Any]],
    ) -> RepositoryTurnResult:
        if task.status not in {TaskStatus.READY_TO_SAVE, TaskStatus.WAITING_APPROVAL}:
            return self._synthetic(
                initial_messages,
                f"任务 `{task.task_id}` 状态为 `{task.status.value}`，不能发布。",
            )
        diff = await self.manager.diff(task)
        if not diff.strip():
            return self._synthetic(initial_messages, "当前任务没有可保存的代码变更。")
        diff_hash = await self.manager.diff_hash(task)
        try:
            fork_owner = (
                self.config.github.fork_owner
                or await self.publisher.current_user(task.workspace)
            )
        except GitHubPublishError as exc:
            return self._synthetic(
                initial_messages,
                f"无法准备 GitHub 发布：{exc}",
                stop_reason="error",
            )
        if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", fork_owner):
            return self._synthetic(
                initial_messages,
                "配置的 GitHub Fork owner 不合法。",
                stop_reason="error",
            )
        branch = self.publisher.branch_name(
            self.config.github.branch_prefix,
            task,
        )
        title, body = self._approval_body(task)
        approval = PublishApproval(
            task_id=task.task_id,
            requester_id=actor,
            action=action,
            upstream=task.repository_full_name,
            fork_owner=fork_owner,
            base_branch=task.base_branch,
            base_commit=task.base_commit,
            head_branch=branch,
            diff_hash=diff_hash,
            title=title,
            body=body,
            draft=True,
            expires_at=utc_now()
            + timedelta(seconds=self.config.github.approval_ttl_seconds),
        )
        task.status = TaskStatus.WAITING_APPROVAL
        task.diff_hash = diff_hash
        task.approval_id = approval.approval_id
        task.approval_hash = approval.approval_hash()
        self.store.save_task(task, event="repository_task.waiting_approval")
        self.store.save_approval(approval, event="publish.approval_requested")
        operation = (
            "创建或复用 Fork，并 Push 新任务分支"
            if action == PublishAction.FORK
            else "创建或复用 Fork、Push 新任务分支，并创建 Draft PR"
        )
        return self._synthetic(
            initial_messages,
            "准备执行 GitHub 外部写操作（L3）：\n\n"
            f"- 操作：{operation}\n"
            f"- 上游：`{approval.upstream}`\n"
            f"- Fork：`{approval.fork_owner}/{task.repository_full_name.split('/', 1)[1]}`\n"
            f"- Base：`{approval.base_branch}` @ `{task.base_commit}`\n"
            f"- Head：`{approval.head_branch}`\n"
            f"- Diff SHA-256：`{approval.diff_hash}`\n"
            f"- 标题：{approval.title}\n"
            f"- Draft PR：`{approval.draft}`\n"
            "- 不会合并、发布或强推默认分支\n\n"
            f"确认：`/approve {approval.approval_id}`\n"
            f"拒绝：`/reject {approval.approval_id}`\n"
            f"审批有效期至：{approval.expires_at.isoformat()}",
        )

    async def _resume_publish(
        self,
        *,
        approval_id: str,
        approve: bool,
        actor: str,
        initial_messages: list[dict[str, Any]],
    ) -> RepositoryTurnResult:
        approval = self.store.load_approval(approval_id)
        if approval is None:
            return self._synthetic(initial_messages, "发布审批不存在或已清理。")
        task = self.store.load_task(approval.task_id)
        if task is None:
            return self._synthetic(initial_messages, "审批关联的代码任务不存在。")
        if actor != approval.requester_id:
            self.store.append_audit(
                "publish.approval_unauthorized",
                approval_id=approval_id,
                task_id=task.task_id,
                actor=actor,
            )
            return self._synthetic(initial_messages, "当前用户无权确认该发布请求。")
        if (
            task.approval_id != approval.approval_id
            or task.approval_hash != approval.approval_hash()
            or approval.task_id != task.task_id
            or approval.upstream.casefold() != task.repository_full_name.casefold()
            or approval.base_branch != task.base_branch
            or approval.base_commit != task.base_commit
        ):
            approval.status = ApprovalStatus.REJECTED
            approval.decided_at = utc_now()
            approval.decided_by = "policy"
            task.status = TaskStatus.READY_TO_SAVE
            self.store.save_approval(
                approval,
                event="publish.approval_binding_invalid",
            )
            self.store.save_task(
                task,
                event="repository_task.approval_binding_invalid",
            )
            return self._synthetic(
                initial_messages,
                "审批内容与当前任务不一致，已拒绝执行。请重新检查并发起。",
                stop_reason="policy_denied",
            )
        if approval.status == ApprovalStatus.EXPIRED:
            task.status = TaskStatus.READY_TO_SAVE
            self.store.save_task(task, event="repository_task.approval_expired")
            return self._synthetic(initial_messages, "审批已过期，请重新发起发布。")
        if not approve:
            approval.status = ApprovalStatus.REJECTED
            approval.decided_at = utc_now()
            approval.decided_by = actor
            task.status = TaskStatus.READY_TO_SAVE
            self.store.save_approval(approval, event="publish.approval_rejected")
            self.store.save_task(task, event="repository_task.publish_rejected")
            return self._synthetic(
                initial_messages,
                "已取消 GitHub 写操作，代码仍保留在本地任务工作区。",
            )
        if approval.status not in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}:
            return self._synthetic(
                initial_messages,
                f"审批状态为 `{approval.status.value}`，不能再次执行。",
            )
        current_hash = await self.manager.diff_hash(task)
        if current_hash != approval.diff_hash:
            approval.status = ApprovalStatus.REJECTED
            approval.decided_at = utc_now()
            approval.decided_by = "policy"
            task.status = TaskStatus.READY_TO_SAVE
            self.store.save_approval(approval, event="publish.approval_invalidated")
            self.store.save_task(task, event="repository_task.diff_changed")
            return self._synthetic(
                initial_messages,
                "代码在审批后发生变化，旧审批已失效。请检查新 Diff 后重新发起。",
                stop_reason="policy_denied",
            )
        approval.status = ApprovalStatus.APPROVED
        approval.decided_at = approval.decided_at or utc_now()
        approval.decided_by = actor
        self.store.save_approval(approval, event="publish.approval_approved")
        try:
            result = await self.publisher.publish(task, approval)
        except (GitHubPublishError, RepositoryError) as exc:
            self.store.append_audit(
                "publish.failed",
                approval_id=approval.approval_id,
                task_id=task.task_id,
                error_type=type(exc).__name__,
            )
            return self._synthetic(
                initial_messages,
                f"GitHub 发布失败：{exc}\n\n"
                f"审批仍绑定原 Diff，可修复认证或网络后重试 `/approve {approval.approval_id}`。",
                stop_reason="error",
            )
        approval.status = ApprovalStatus.EXECUTED
        task.status = TaskStatus.PUBLISHED
        task.published_url = result.pull_request_url or result.fork_url
        self.store.save_approval(approval, event="publish.completed")
        self.store.save_task(task, event="repository_task.published")
        content = (
            "GitHub 操作完成：\n\n"
            f"- Fork：{result.fork_url}\n"
            f"- 分支：`{result.branch}`\n"
            f"- Commit：`{result.commit_sha}`"
        )
        if result.pull_request_url:
            content += f"\n- Draft PR：{result.pull_request_url}"
        content += "\n\n系统未执行合并、Release 或强推。"
        return self._synthetic(initial_messages, content)

    async def _prepare_repository_deletion(
        self,
        *,
        request: str,
        actor: str,
        session_key: str,
        initial_messages: list[dict[str, Any]],
    ) -> RepositoryTurnResult:
        target_name = request.strip()
        try:
            records = await self.manager.list_vfs_repositories()
        except RepositoryError as exc:
            return self._synthetic(
                initial_messages,
                f"无法准备删除：{exc}",
                stop_reason="error",
            )
        matched = [
            record
            for record in records
            if record.full_name.casefold() == target_name.casefold()
        ]
        if not matched:
            active = self.store.active_repository(session_key)
            try:
                record = await self.manager.resolve_cached_repository(
                    target_name,
                    active=active,
                )
            except RepositoryError as exc:
                return self._synthetic(
                    initial_messages,
                    f"无法准备删除：{exc}",
                    stop_reason="error",
                )
            matched = [record]
        targets: list[dict[str, str]] = []
        for record in matched:
            targets.append(
                {
                    "repository_id": record.repository_id,
                    "full_name": record.full_name,
                    "url": record.url,
                    "source_path": str(record.path.resolve()),
                    "cache_path": str(record.path.resolve().parent),
                    "commit_sha": record.commit_sha,
                }
            )
        approval = RepositoryDeletionApproval(
            requester_id=actor,
            targets=targets,
            expires_at=utc_now()
            + timedelta(seconds=self.config.deletion_approval_ttl_seconds),
        )
        approval.binding_hash = approval.approval_hash()
        self.store.save_deletion_approval(
            approval,
            event="repository.delete.approval_requested",
        )
        listing = "\n".join(
            f"- `{target['full_name']}` → `{target['source_path']}`"
            for target in targets
        )
        return self._synthetic(
            initial_messages,
            "准备删除本地下载项目（L3，必须人工确认）：\n\n"
            f"- 命中 {len(targets)} 个项目\n"
            f"{listing}\n\n"
            "- 删除范围包含只读源码缓存、隔离任务工作区和该项目的本地 Patch\n"
            "- 不会删除 GitHub 上的仓库或 Fork\n"
            "- 删除后系统无法恢复，只能重新 Clone\n\n"
            f"确认删除：`/approve-delete {approval.approval_id}`\n"
            f"取消删除：`/reject-delete {approval.approval_id}`\n"
            f"审批有效期至：{approval.expires_at.isoformat()}",
            agent_ui={
                "kind": "repository_delete_approval",
                "data": {
                    "approval_id": approval.approval_id,
                    "targets": [
                        {
                            "repository_full_name": target["full_name"],
                            "commit_sha": target["commit_sha"],
                            "source_path": target["source_path"],
                            "cache_path": target["cache_path"],
                        }
                        for target in targets
                    ],
                    "expires_at": approval.expires_at.isoformat(),
                },
            },
        )

    async def _resume_repository_deletion(
        self,
        *,
        approval_id: str,
        approve: bool,
        actor: str,
        initial_messages: list[dict[str, Any]],
    ) -> RepositoryTurnResult:
        approval = self.store.load_deletion_approval(approval_id)
        if approval is None:
            return self._synthetic(initial_messages, "删除审批不存在或已清理。")
        if actor != approval.requester_id:
            self.store.append_audit(
                "repository.delete.approval_unauthorized",
                approval_id=approval.approval_id,
                repository=",".join(target["full_name"] for target in approval.targets),
                actor=actor,
            )
            return self._synthetic(
                initial_messages,
                "当前用户无权确认该删除请求。",
                stop_reason="policy_denied",
            )
        if approval.binding_hash != approval.approval_hash():
            approval.status = ApprovalStatus.REJECTED
            approval.decided_at = utc_now()
            approval.decided_by = "policy"
            self.store.save_deletion_approval(
                approval,
                event="repository.delete.approval_binding_invalid",
            )
            return self._synthetic(
                initial_messages,
                "删除目标与原审批内容不一致，已拒绝执行。",
                stop_reason="policy_denied",
            )
        if approval.status == ApprovalStatus.EXPIRED:
            return self._synthetic(initial_messages, "删除审批已过期，请重新发起。")
        if not approve:
            if approval.status in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}:
                approval.status = ApprovalStatus.REJECTED
                approval.decided_at = utc_now()
                approval.decided_by = actor
                self.store.save_deletion_approval(
                    approval,
                    event="repository.delete.approval_rejected",
                )
            return self._synthetic(
                initial_messages,
                "已取消删除，项目仍保留。",
            )
        if approval.status not in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}:
            return self._synthetic(
                initial_messages,
                f"删除审批状态为 `{approval.status.value}`，不能再次执行。",
            )

        deleted: list[str] = []
        try:
            for target in approval.targets:
                source = Path(target["source_path"]).resolve()
                if not source.is_dir() or not (source / ".git").is_dir():
                    raise RepositoryError("删除目标已不存在。")
                commit = await self.manager.commit_sha_at(source)
                if commit != target["commit_sha"]:
                    raise RepositoryError(
                        "仓库缓存自审批后发生变化，旧删除审批已失效。"
                    )
                root = self.manager.delete_repository_source(source)
                deleted.append(str(root))
        except RepositoryError as exc:
            approval.status = ApprovalStatus.REJECTED
            approval.decided_at = utc_now()
            approval.decided_by = "policy"
            self.store.save_deletion_approval(
                approval,
                event="repository.delete.target_changed",
            )
            return self._synthetic(
                initial_messages,
                f"删除未执行：{exc}",
                stop_reason="policy_denied",
            )

        for target in approval.targets:
            self.store.invalidate_repository(target["repository_id"])
        approval.status = ApprovalStatus.APPROVED
        approval.decided_at = approval.decided_at or utc_now()
        approval.decided_by = actor
        self.store.save_deletion_approval(
            approval,
            event="repository.delete.approval_approved",
        )
        approval.status = ApprovalStatus.EXECUTED
        self.store.save_deletion_approval(
            approval,
            event="repository.delete.completed",
        )
        return self._synthetic(
            initial_messages,
            "本地项目已删除：\n\n"
            + "\n".join(f"- `{path}`" for path in deleted)
            + "\n- GitHub 远程仓库未受影响\n"
            "- 如需再次分析，请重新发送仓库 URL",
        )

    async def _handle_repository_deletion(
        self,
        *,
        request: str,
        actor: str,
        session_key: str,
        initial_messages: list[dict[str, Any]],
    ) -> RepositoryTurnResult | None:
        stripped = request.strip()
        if match := self._DELETE_APPROVE_RE.fullmatch(stripped):
            return await self._resume_repository_deletion(
                approval_id=match.group(1),
                approve=True,
                actor=actor,
                initial_messages=initial_messages,
            )
        if match := self._DELETE_REJECT_RE.fullmatch(stripped):
            return await self._resume_repository_deletion(
                approval_id=match.group(1),
                approve=False,
                actor=actor,
                initial_messages=initial_messages,
            )
        command = self._REDEL_COMMAND_RE.fullmatch(stripped)
        if command is not None:
            target = (command.group("target") or "当前项目").strip()
            return await self._prepare_repository_deletion(
                request=target,
                actor=actor,
                session_key=session_key,
                initial_messages=initial_messages,
            )
        if self._is_local_delete_request(request, session_key):
            return await self._prepare_repository_deletion(
                request=request,
                actor=actor,
                session_key=session_key,
                initial_messages=initial_messages,
            )
        return None

    async def _handle_command(
        self,
        *,
        request: str,
        actor: str,
        session_key: str,
        initial_messages: list[dict[str, Any]],
    ) -> RepositoryTurnResult | None:
        deletion_result = await self._handle_repository_deletion(
            request=request,
            actor=actor,
            session_key=session_key,
            initial_messages=initial_messages,
        )
        if deletion_result is not None:
            return deletion_result
        if match := self._APPROVE_RE.fullmatch(request.strip()):
            return await self._resume_publish(
                approval_id=match.group(1),
                approve=True,
                actor=actor,
                initial_messages=initial_messages,
            )
        if match := self._REJECT_RE.fullmatch(request.strip()):
            return await self._resume_publish(
                approval_id=match.group(1),
                approve=False,
                actor=actor,
                initial_messages=initial_messages,
            )
        normalized = re.sub(r"\s+", "", request.strip().casefold())
        natural_action: str | None = None
        if normalized in {"保存代码", "保存修改", "保存当前修改"}:
            natural_action = "choices"
        elif normalized in {"fork代码", "fork修改", "fork到github", "保存到fork"}:
            natural_action = "fork"
        elif normalized in {
            "创建pr",
            "创建pullrequest",
            "提交pr",
            "fork并创建pr",
        }:
            natural_action = "pr"
        command = self._COMMAND_RE.fullmatch(request.strip())
        if command is None and natural_action is None:
            return None
        if natural_action is not None:
            action = natural_action
        elif command is not None:
            action = (command.group("action") or "status").casefold()
        else:
            return None
        if action == "close":
            self.close_session(session_key)
            return self._synthetic(
                initial_messages,
                "已退出当前仓库工作区，后续消息将恢复普通 Agent 模式。",
            )
        record = self.store.active_repository(session_key)
        if record is None:
            return self._synthetic(
                initial_messages,
                "当前会话没有仓库，请先发送 GitHub 仓库 URL。",
            )
        if action == "status":
            content = self._repository_summary(record)
            task = self.store.latest_task(session_key)
            if task is not None:
                content += (
                    f"\n\n最近任务：`{task.task_id}`\n"
                    f"- 状态：`{task.status.value}`\n"
                    f"- 工作区：`{task.workspace_path}`"
                )
            return self._synthetic(initial_messages, content)
        task = self.store.latest_task(session_key)
        if task is None:
            return self._synthetic(
                initial_messages,
                "当前会话没有可保存的代码任务。",
            )
        if action == "choices":
            return self._synthetic(
                initial_messages,
                "代码修改已完成，请选择保存方式：" + self._save_choices(task),
            )
        if action == "save":
            return self._synthetic(
                initial_messages,
                f"代码已保留在本地工作区：`{task.workspace_path}`\n"
                "尚未 Fork、Push 或创建 PR。",
            )
        if action == "patch":
            patch = await self.manager.write_patch(task)
            self.store.append_audit(
                "repository_task.patch_created",
                task_id=task.task_id,
                path=str(patch),
                actor=actor,
            )
            return self._synthetic(
                initial_messages,
                f"Patch 已生成：`{patch}`\n尚未写入 GitHub。",
            )
        publish_action = (
            PublishAction.FORK if action == "fork" else PublishAction.DRAFT_PR
        )
        return await self._prepare_publish(
            task=task,
            action=publish_action,
            actor=actor,
            initial_messages=initial_messages,
        )

    async def run(
        self,
        *,
        request: str,
        actor: str,
        session_key: str,
        initial_messages: list[dict[str, Any]],
        runtime: LLMRuntime,
        invoke: RepositoryAgentInvoker,
        on_update: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        tenant_id: str = "default",
        trace_id: str | None = None,
    ) -> RepositoryTurnResult:
        """Run one repository turn inside its user-scoped logical VFS."""
        if self.manager.vfs is None or not self.manager.vfs.enabled:
            return await self._run_scoped(
                request=request,
                actor=actor,
                session_key=session_key,
                initial_messages=initial_messages,
                runtime=runtime,
                invoke=invoke,
                on_update=on_update,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                tenant_id=tenant_id,
                trace_id=trace_id,
            )
        identity = VFSIdentity(
            tenant_id=tenant_id,
            user_id=actor or session_key,
            session_id=session_key,
        )
        with self.manager.vfs.bind(identity), self.manager.bind_vfs_identity(identity):
            return await self._run_scoped(
                request=request,
                actor=actor,
                session_key=session_key,
                initial_messages=initial_messages,
                runtime=runtime,
                invoke=invoke,
                on_update=on_update,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                tenant_id=tenant_id,
                trace_id=trace_id,
            )

    async def _list_downloaded_repositories(
        self,
        initial_messages: list[dict[str, Any]],
    ) -> RepositoryTurnResult:
        try:
            records = await self.manager.list_all_cached_repositories()
        except RepositoryError as exc:
            return self._synthetic(
                initial_messages,
                f"列出已下载仓库失败：{exc}",
                stop_reason="error",
            )
        if not records:
            return self._synthetic(initial_messages, "当前没有已下载的仓库。")
        lines = [f"- `{record.full_name}` → `{record.local_path}`" for record in records]
        return self._synthetic(
            initial_messages,
            f"当前已下载 {len(records)} 个仓库：\n\n" + "\n".join(lines),
        )

    async def _list_all_vfs_repositories(
        self,
        initial_messages: list[dict[str, Any]],
        runtime: LLMRuntime,
    ) -> RepositoryTurnResult:
        try:
            records = await self.manager.list_vfs_repositories()
        except RepositoryError as exc:
            return self._synthetic(
                initial_messages,
                f"检索 VFS 项目失败：{exc}",
                stop_reason="error",
            )
        if not records:
            return self._synthetic(initial_messages, "VFS 文件夹下没有找到任何项目。")
        unique: dict[str, str] = {}
        for record in records:
            unique.setdefault(record.full_name.casefold(), record.full_name)
        names = sorted(unique.values(), key=str.casefold)
        lines = [f"- `{name}`" for name in names]
        return self._synthetic(
            initial_messages,
            f"VFS 下共找到 {len(names)} 个项目：\n\n" + "\n".join(lines),
        )

    async def _run_scoped(
        self,
        *,
        request: str,
        actor: str,
        session_key: str,
        initial_messages: list[dict[str, Any]],
        runtime: LLMRuntime,
        invoke: RepositoryAgentInvoker,
        on_update: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        tenant_id: str = "default",
        trace_id: str | None = None,
    ) -> RepositoryTurnResult:
        if self._REGET_COMMAND_RE.fullmatch(request.strip()):
            return await self._list_all_vfs_repositories(initial_messages, runtime)
        research_command = self._RESEARCH_COMMAND_RE.fullmatch(request.strip())
        force_research = research_command is not None
        if research_command is not None:
            request = (research_command.group("query") or "").strip()
            if not request:
                return self._synthetic(
                    initial_messages,
                    "请在命令后提供研究需求，例如："
                    "`/rere 下载并分析 3 个自回归图像生成开源项目`。",
                )
            status_match = re.fullmatch(r"status\s+([A-Za-z0-9_-]{1,100})", request, re.IGNORECASE)
            if status_match is not None:
                task = self.store.load_research(status_match.group(1))
                if task is None:
                    return self._synthetic(initial_messages, "没有找到该研究任务。", stop_reason="error")
                report = f"\n\n---\n\n{task.report.rstrip()}" if task.report else ""
                return self._synthetic(
                    initial_messages,
                    f"研究任务：`{task.research_id}`\n\n"
                    f"- 工作流：`{task.workflow_id or '未关联'}`\n"
                    f"- 状态：`{task.status.value}`\n"
                    f"- 已选仓库：{len(task.selected)}\n"
                    f"- 已完成分析：{sum(1 for item in task.analyses if item.summary)}"
                    f"{report}",
                )
        command_result = await self._handle_command(
            request=request,
            actor=actor,
            session_key=session_key,
            initial_messages=initial_messages,
        )
        if command_result is not None:
            return command_result
        risk = self.policy.assess_request(request)
        if not risk.allowed:
            self.store.append_audit(
                "repository_request.denied",
                actor=actor,
                session_key=session_key,
                reasons=risk.reasons,
            )
            return self._synthetic(
                initial_messages,
                "该请求命中 L4 GitHub 禁止策略，已直接拒绝："
                + "；".join(risk.reasons),
                stop_reason="policy_denied",
            )
        if self.is_local_cache_list_request(request):
            return await self._list_downloaded_repositories(initial_messages)
        if force_research or self.is_research_request(request):
            try:
                task = await self.research.enqueue_pool(
                    direction=request,
                    actor=actor,
                    session_key=session_key,
                    tenant_id=tenant_id,
                    trace_id=trace_id,
                )
            except Exception as exc:
                return self._synthetic(
                    initial_messages,
                    f"GitHub 项目研究失败：{exc}",
                    stop_reason="error",
                )
            if on_update:
                await on_update("研究任务已进入独立 Worker 队列，Gateway 不再执行下载和分析。")
            return self._synthetic(
                initial_messages,
                "",
                deferred_to_worker=True,
            )
        url = self.manager.extract_url(request)
        record: RepositoryRecord | None
        if url:
            try:
                record = await self.manager.load(url)
            except RepositoryError as exc:
                return self._synthetic(
                    initial_messages,
                    f"仓库加载失败：{exc}",
                    stop_reason="error",
                )
            self.store.set_active_repository(session_key, record)
            self.store.append_audit(
                "repository.loaded",
                actor=actor,
                session_key=session_key,
                repository=record.full_name,
                commit=record.commit_sha,
            )
        else:
            record = self.store.active_repository(session_key)
        if record is None:
            return self._synthetic(
                initial_messages,
                "请先发送 GitHub 仓库地址，例如 `https://github.com/owner/repo`。",
            )
        question_without_url = request.replace(url, "").strip() if url else request.strip()
        if not question_without_url:
            question_without_url = "请概览这个项目的用途、技术栈、入口和核心架构。"
        intent = await self.intent_agent.classify(
            question_without_url,
            record,
            runtime,
        )
        if intent.complexity == TaskComplexity.COMPLEX:
            result = await self._run_complex(
                request=question_without_url,
                record=record,
                intent=intent,
                actor=actor,
                session_key=session_key,
                initial_messages=initial_messages,
                runtime=runtime,
                invoke=invoke,
            )
        else:
            result = await self._run_simple(
                request=question_without_url,
                record=record,
                intent=intent,
                initial_messages=initial_messages,
                runtime=runtime,
                invoke=invoke,
            )
        return self._with_repository_path(result, record)
