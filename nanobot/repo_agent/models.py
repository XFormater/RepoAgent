"""Persistent domain models for repository analysis and code workflows."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(9)}"


class RepositoryIntentKind(StrEnum):
    REPOSITORY_OVERVIEW = "repository_overview"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    CODE_QUESTION = "code_question"
    SYMBOL_LOCATION = "symbol_location"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    ISSUE_TRIAGE = "issue_triage"
    BUG_DIAGNOSIS = "bug_diagnosis"
    CODE_CHANGE = "code_change"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    PULL_REQUEST = "pull_request"


class TaskComplexity(StrEnum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class RepositoryIntent(BaseModel):
    intent: RepositoryIntentKind
    complexity: TaskComplexity
    preferred_agent: str | None = None
    issue_number: int | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    reason: str = ""


class RepositoryRecord(BaseModel):
    repository_id: str
    url: str
    owner: str
    name: str
    default_branch: str
    commit_sha: str
    local_path: str
    file_count: int
    size_bytes: int
    loaded_at: datetime = Field(default_factory=utc_now)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def path(self) -> Path:
        return Path(self.local_path)


class ResearchStatus(StrEnum):
    PLANNING = "planning"
    SEARCHING = "searching"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GitHubRepositoryCandidate(BaseModel):
    full_name: str
    url: str
    description: str = ""
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    language: str | None = None
    license: str | None = None
    topics: list[str] = Field(default_factory=list)
    default_branch: str = "main"
    updated_at: datetime | None = None
    archived: bool = False
    relevance_score: float = 0.0
    selection_reason: str = ""


class RepositoryResearchAnalysis(BaseModel):
    candidate: GitHubRepositoryCandidate
    repository: RepositoryRecord | None = None
    summary: str = ""
    tools_used: list[str] = Field(default_factory=list)
    error: str | None = None


class RepositoryResearchTask(BaseModel):
    research_id: str = Field(default_factory=lambda: new_id("research"))
    session_key: str
    actor: str
    direction: str
    workflow_id: str | None = None
    harness_task_id: str | None = None
    trace_id: str | None = None
    queries: list[str] = Field(default_factory=list)
    candidates: list[GitHubRepositoryCandidate] = Field(default_factory=list)
    selected: list[GitHubRepositoryCandidate] = Field(default_factory=list)
    analyses: list[RepositoryResearchAnalysis] = Field(default_factory=list)
    status: ResearchStatus = ResearchStatus.PLANNING
    report: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskStatus(StrEnum):
    PLANNING = "planning"
    RUNNING = "running"
    READY_TO_SAVE = "ready_to_save"
    WAITING_APPROVAL = "waiting_approval"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanStep(BaseModel):
    step_id: str
    name: str
    instruction: str
    assigned_agent: str
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None


class RepositoryTask(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    session_key: str
    repository_id: str
    repository_full_name: str
    source_url: str
    base_branch: str
    base_commit: str
    objective: str
    workspace_path: str
    status: TaskStatus = TaskStatus.PLANNING
    steps: list[PlanStep] = Field(default_factory=list)
    diff_hash: str | None = None
    approval_id: str | None = None
    approval_hash: str | None = None
    published_url: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_dag(self) -> "RepositoryTask":
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        known = set(ids)
        for step in self.steps:
            missing = set(step.depends_on) - known
            if missing:
                raise ValueError(
                    f"step {step.step_id!r} has unknown dependencies: {sorted(missing)}"
                )
        visiting: set[str] = set()
        visited: set[str] = set()
        dependencies = {step.step_id: step.depends_on for step in self.steps}

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("plan dependencies contain a cycle")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in dependencies[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in ids:
            visit(step_id)
        return self

    @property
    def workspace(self) -> Path:
        return Path(self.workspace_path)


class PublishAction(StrEnum):
    FORK = "fork"
    DRAFT_PR = "draft_pr"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"


class PublishApproval(BaseModel):
    approval_id: str = Field(default_factory=lambda: new_id("approval"))
    task_id: str
    requester_id: str
    action: PublishAction
    upstream: str
    fork_owner: str
    base_branch: str
    base_commit: str
    head_branch: str
    diff_hash: str
    title: str
    body: str
    draft: bool = True
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    decided_at: datetime | None = None
    decided_by: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "action": self.action.value,
            "upstream": self.upstream,
            "fork_owner": self.fork_owner,
            "base_branch": self.base_branch,
            "base_commit": self.base_commit,
            "head_branch": self.head_branch,
            "diff_hash": self.diff_hash,
            "title": self.title,
            "body": self.body,
            "draft": self.draft,
        }

    def approval_hash(self) -> str:
        data = json.dumps(
            self.payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


class RepositoryDeletionApproval(BaseModel):
    """Exact, expiring binding for deleting one or more local repository caches."""

    approval_id: str = Field(default_factory=lambda: new_id("delete"))
    requester_id: str
    targets: list[dict[str, str]]
    binding_hash: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    decided_at: datetime | None = None
    decided_by: str | None = None

    def payload(self) -> dict[str, Any]:
        return {"targets": self.targets}

    def approval_hash(self) -> str:
        data = json.dumps(
            self.payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
