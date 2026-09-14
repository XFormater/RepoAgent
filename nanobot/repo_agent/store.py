"""Atomic local state for repository sessions, tasks, approvals, and audit."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from nanobot.repo_agent.models import (
    ApprovalStatus,
    PublishApproval,
    RepositoryDeletionApproval,
    RepositoryRecord,
    RepositoryResearchTask,
    RepositoryTask,
    ResearchStatus,
    TaskStatus,
    utc_now,
)


class RepositoryStateStore:
    def __init__(self, workspace: Path, state_dir: str):
        root = (workspace / state_dir).resolve()
        workspace_root = workspace.resolve()
        if root != workspace_root and workspace_root not in root.parents:
            raise ValueError("repository state directory must remain inside the workspace")
        self.root = root
        self.session_dir = root / "sessions"
        self.task_dir = root / "tasks"
        self.approval_dir = root / "approvals"
        self.deletion_approval_dir = root / "deletion-approvals"
        self.research_dir = root / "research"
        self.audit_path = root / "audit.jsonl"
        for path in (
            self.session_dir,
            self.task_dir,
            self.approval_dir,
            self.deletion_approval_dir,
            self.research_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _key(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_id(value: str) -> str:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
            raise ValueError("invalid persistent object id")
        return value

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def set_active_repository(self, session_key: str, record: RepositoryRecord) -> None:
        payload = {
            "session_key": session_key,
            "repository": record.model_dump(mode="json"),
            "updated_at": utc_now().isoformat(),
        }
        with self._lock:
            self._atomic_json(self.session_dir / f"{self._key(session_key)}.json", payload)

    def active_repository(self, session_key: str) -> RepositoryRecord | None:
        path = self.session_dir / f"{self._key(session_key)}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return RepositoryRecord.model_validate(payload["repository"])
        except Exception:
            return None

    def clear_active_repository(self, session_key: str) -> None:
        path = self.session_dir / f"{self._key(session_key)}.json"
        with self._lock:
            if path.exists():
                path.unlink()
            self.append_audit(
                "repository.session_closed",
                session_key=session_key,
            )

    def save_task(self, task: RepositoryTask, *, event: str | None = None) -> None:
        task.updated_at = utc_now()
        path = self.task_dir / f"{self._safe_id(task.task_id)}.json"
        with self._lock:
            self._atomic_json(path, task.model_dump(mode="json"))
            if event:
                self.append_audit(
                    event,
                    task_id=task.task_id,
                    session_key=task.session_key,
                    status=task.status.value,
                )

    def save_research(
        self,
        task: RepositoryResearchTask,
        *,
        event: str | None = None,
    ) -> None:
        task.updated_at = utc_now()
        path = self.research_dir / f"{self._safe_id(task.research_id)}.json"
        with self._lock:
            self._atomic_json(path, task.model_dump(mode="json"))
            if event:
                self.append_audit(
                    event,
                    research_id=task.research_id,
                    session_key=task.session_key,
                    status=task.status.value,
                    candidate_count=len(task.candidates),
                    selected_count=len(task.selected),
                )

    def load_research(self, research_id: str) -> RepositoryResearchTask | None:
        path = self.research_dir / f"{self._safe_id(research_id)}.json"
        if not path.exists():
            return None
        return RepositoryResearchTask.model_validate_json(path.read_text(encoding="utf-8"))

    def active_research(self, session_key: str) -> list[RepositoryResearchTask]:
        """Return non-terminal research tasks for a session, newest first."""
        active: list[RepositoryResearchTask] = []
        for path in self.research_dir.glob("*.json"):
            try:
                task = RepositoryResearchTask.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if task.session_key == session_key and task.status not in {
                ResearchStatus.COMPLETED,
                ResearchStatus.PARTIAL,
                ResearchStatus.FAILED,
                ResearchStatus.CANCELLED,
            }:
                active.append(task)
        return sorted(active, key=lambda item: item.updated_at, reverse=True)

    def load_task(self, task_id: str) -> RepositoryTask | None:
        path = self.task_dir / f"{self._safe_id(task_id)}.json"
        if not path.exists():
            return None
        return RepositoryTask.model_validate_json(path.read_text(encoding="utf-8"))

    def latest_task(self, session_key: str) -> RepositoryTask | None:
        tasks: list[RepositoryTask] = []
        for path in self.task_dir.glob("*.json"):
            try:
                task = RepositoryTask.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if task.session_key == session_key and task.status in {
                TaskStatus.READY_TO_SAVE,
                TaskStatus.WAITING_APPROVAL,
                TaskStatus.PUBLISHED,
            }:
                tasks.append(task)
        return max(tasks, key=lambda item: item.updated_at) if tasks else None

    def save_approval(
        self,
        approval: PublishApproval,
        *,
        event: str | None = None,
    ) -> None:
        path = self.approval_dir / f"{self._safe_id(approval.approval_id)}.json"
        with self._lock:
            self._atomic_json(path, approval.model_dump(mode="json"))
            if event:
                self.append_audit(
                    event,
                    approval_id=approval.approval_id,
                    task_id=approval.task_id,
                    action=approval.action.value,
                    status=approval.status.value,
                    actor=approval.decided_by or approval.requester_id,
                )

    def load_approval(self, approval_id: str) -> PublishApproval | None:
        path = self.approval_dir / f"{self._safe_id(approval_id)}.json"
        if not path.exists():
            return None
        approval = PublishApproval.model_validate_json(path.read_text(encoding="utf-8"))
        if approval.status == ApprovalStatus.PENDING and approval.expires_at <= utc_now():
            approval.status = ApprovalStatus.EXPIRED
            self.save_approval(approval, event="publish.approval_expired")
        return approval

    def save_deletion_approval(
        self,
        approval: RepositoryDeletionApproval,
        *,
        event: str | None = None,
    ) -> None:
        path = self.deletion_approval_dir / f"{self._safe_id(approval.approval_id)}.json"
        with self._lock:
            self._atomic_json(path, approval.model_dump(mode="json"))
            if event:
                self.append_audit(
                    event,
                    approval_id=approval.approval_id,
                    repository=",".join(target["full_name"] for target in approval.targets),
                    repository_id=",".join(target["repository_id"] for target in approval.targets),
                    cache_path=",".join(target["cache_path"] for target in approval.targets),
                    status=approval.status.value,
                    actor=approval.decided_by or approval.requester_id,
                )

    def load_deletion_approval(
        self,
        approval_id: str,
    ) -> RepositoryDeletionApproval | None:
        path = self.deletion_approval_dir / f"{self._safe_id(approval_id)}.json"
        if not path.exists():
            return None
        approval = RepositoryDeletionApproval.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if approval.status == ApprovalStatus.PENDING and approval.expires_at <= utc_now():
            approval.status = ApprovalStatus.EXPIRED
            self.save_deletion_approval(
                approval,
                event="repository.delete.approval_expired",
            )
        return approval

    def invalidate_repository(self, repository_id: str) -> None:
        """Clear live bindings while retaining cancelled task records for audit."""
        affected_task_ids: set[str] = set()
        with self._lock:
            for path in self.session_dir.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    record = RepositoryRecord.model_validate(payload["repository"])
                except Exception:
                    continue
                if record.repository_id == repository_id:
                    path.unlink()

            for path in self.task_dir.glob("*.json"):
                try:
                    task = RepositoryTask.model_validate_json(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if task.repository_id != repository_id:
                    continue
                affected_task_ids.add(task.task_id)
                if task.status not in {TaskStatus.PUBLISHED, TaskStatus.CANCELLED}:
                    task.status = TaskStatus.CANCELLED
                    self.save_task(task, event="repository_task.cache_deleted")

            for path in self.approval_dir.glob("*.json"):
                try:
                    approval = PublishApproval.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                except Exception:
                    continue
                if (
                    approval.task_id in affected_task_ids
                    and approval.status in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}
                ):
                    approval.status = ApprovalStatus.REJECTED
                    approval.decided_at = utc_now()
                    approval.decided_by = "policy"
                    self.save_approval(
                        approval,
                        event="publish.approval_repository_deleted",
                    )

    def append_audit(self, event: str, **fields: Any) -> None:
        row = {"timestamp": utc_now().isoformat(), "event": event, **fields}
        with self._lock, self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
