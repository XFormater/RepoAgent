"""Stable contracts for multi-agent harness orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class AgentRole(StrEnum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    WORKER = "worker"
    REVIEWER = "reviewer"


class AgentState(StrEnum):
    CREATED = "created"
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    STOPPED = "stopped"
    FAILED = "failed"


class TaskState(StrEnum):
    PENDING = "pending"
    READY = "ready"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


class TaskExecutionMode(StrEnum):
    """Selects the runtime allowed to claim a task.

    INLINE tasks are owned by the request-local coordinator that created them.
    POOL tasks may be claimed by an independent HarnessPoolWorker.
    """

    INLINE = "inline"
    POOL = "pool"


class WorkflowState(StrEnum):
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerState(StrEnum):
    ONLINE = "online"
    DRAINING = "draining"


class MessageKind(StrEnum):
    TASK = "task"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    CONTROL = "control"


class ContextPolicy(BaseModel):
    """Hard context boundary. Parent transcript inheritance is disabled by default."""

    max_messages: int = Field(default=24, ge=1, le=200)
    max_chars: int = Field(default=48_000, ge=1_000)
    max_tokens: int = Field(default=12_000, ge=256)
    max_message_chars: int = Field(default=12_000, ge=256)
    inherit_parent_messages: int = Field(default=0, ge=0, le=20)
    accepted_message_kinds: frozenset[MessageKind] = Field(
        default_factory=lambda: frozenset(MessageKind)
    )


class ResourceBudget(BaseModel):
    max_iterations: int | None = Field(default=20, ge=1)
    max_runtime_seconds: int = Field(default=900, ge=1)
    max_tool_calls: int | None = Field(default=60, ge=0)


class PlannedWorkItem(BaseModel):
    """A bounded unit emitted by planning and materialized as a role DAG."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    task_input: str = Field(min_length=1)
    required_capabilities: frozenset[str] = Field(default_factory=frozenset)
    resource_keys: frozenset[str] = Field(default_factory=frozenset)
    priority: int = Field(default=0, ge=-100, le=100)
    require_review: bool = True
    review_instructions: str = "Verify evidence and reject unsupported conclusions."


class ReviewResult(BaseModel):
    """Machine-readable decision required for a configured Reviewer gate."""

    approved: bool
    notes: str = Field(min_length=1, max_length=32_000)


class AgentSpec(BaseModel):
    """Versioned role and capability declaration; contains no runtime state."""

    model_config = ConfigDict(frozen=True)

    spec_id: str
    version: int = Field(default=1, ge=1)
    role: AgentRole
    objective: str = Field(min_length=1)
    capabilities: frozenset[str] = Field(default_factory=frozenset)
    allowed_tools: frozenset[str] = Field(default_factory=frozenset)
    model_policy: str = "inherit"
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)
    resource_budget: ResourceBudget = Field(default_factory=ResourceBudget)


class AgentInstance(BaseModel):
    instance_id: str = Field(default_factory=lambda: new_id("agent"))
    spec_id: str
    workflow_id: str
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    parent_agent_id: str | None = None
    state: AgentState = AgentState.CREATED
    current_task_id: str | None = None
    worker_id: str | None = None
    checkpoint_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    version: int = Field(default=1, ge=1)


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    workflow_id: str
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    title: str = Field(min_length=1, max_length=500)
    input: str = Field(min_length=1, max_length=1_000_000)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    execution_mode: TaskExecutionMode = TaskExecutionMode.POOL
    required_role: AgentRole = AgentRole.WORKER
    required_capabilities: frozenset[str] = Field(default_factory=frozenset)
    resource_keys: frozenset[str] = Field(default_factory=frozenset)
    dependencies: frozenset[str] = Field(default_factory=frozenset)
    review_target_task_id: str | None = None
    idempotency_key: str | None = None
    priority: int = Field(default=0, ge=-100, le=100)
    available_at: datetime = Field(default_factory=utcnow)
    deadline_at: datetime | None = None
    state: TaskState = TaskState.PENDING
    assigned_agent_id: str | None = None
    lease_id: str | None = None
    fencing_token: int = Field(default=0, ge=0)
    attempt: int = 0
    max_attempts: int = Field(default=3, ge=1)
    retry_backoff_seconds: int = Field(default=5, ge=0, le=3_600)
    result: str | None = None
    error: str | None = None
    dead_lettered_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_schedule(self) -> "Task":
        if self.deadline_at is not None and self.deadline_at <= self.available_at:
            raise ValueError("task deadline must be after available_at")
        return self


class Workflow(BaseModel):
    workflow_id: str = Field(default_factory=lambda: new_id("workflow"))
    goal: str = Field(min_length=1, max_length=1_000_000)
    owner_agent_id: str
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    scope_key: str | None = Field(default=None, max_length=512)
    sealed: bool = True
    state: WorkflowState = WorkflowState.ACTIVE
    version: int = Field(default=1, ge=1)
    tasks: list[Task] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_task_graph(self) -> "Workflow":
        ids = {task.task_id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("workflow task_id values must be unique")
        for task in self.tasks:
            if task.workflow_id != self.workflow_id:
                raise ValueError(f"task {task.task_id} belongs to another workflow")
            unknown = task.dependencies - ids
            if unknown:
                raise ValueError(f"task {task.task_id} has unknown dependencies: {sorted(unknown)}")
            if task.task_id in task.dependencies:
                raise ValueError(f"task {task.task_id} cannot depend on itself")
            if task.review_target_task_id is not None:
                if task.required_role != AgentRole.REVIEWER:
                    raise ValueError("review gates must use the reviewer role")
                if task.review_target_task_id not in task.dependencies:
                    raise ValueError("review gate must depend on its target task")
        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {task.task_id: task.dependencies for task in self.tasks}

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("workflow contains a dependency cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in graph[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in graph:
            visit(task_id)
        return self


class AgentMessage(BaseModel):
    """Point-to-point envelope. A broadcast recipient is intentionally unsupported."""

    message_id: str = Field(default_factory=lambda: new_id("msg"))
    workflow_id: str
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    task_id: str | None = None
    sender_agent_id: str
    recipient_agent_id: str
    kind: MessageKind
    content: str = Field(max_length=262_144)
    correlation_id: str | None = None
    idempotency_key: str = Field(min_length=1, max_length=512)
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    delivery_id: str | None = None

    @model_validator(mode="after")
    def validate_route(self) -> "AgentMessage":
        if self.sender_agent_id == self.recipient_agent_id:
            raise ValueError("agent messages must cross an agent boundary")
        return self


class Checkpoint(BaseModel):
    checkpoint_id: str = Field(default_factory=lambda: new_id("checkpoint"))
    workflow_id: str
    agent_id: str
    task_id: str | None = None
    sequence: int = Field(ge=0)
    state: dict[str, Any]
    created_at: datetime = Field(default_factory=utcnow)


class WorkerLease(BaseModel):
    lease_id: str = Field(default_factory=lambda: new_id("lease"))
    task_id: str
    agent_id: str
    worker_id: str
    fencing_token: int = Field(ge=1)
    resource_keys: frozenset[str] = Field(default_factory=frozenset)
    acquired_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    heartbeat_at: datetime = Field(default_factory=utcnow)
    released_at: datetime | None = None

    def active_at(self, instant: datetime) -> bool:
        return self.released_at is None and self.expires_at > instant


class WorkerRegistration(BaseModel):
    worker_id: str
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    roles: frozenset[AgentRole]
    capabilities: frozenset[str] = Field(default_factory=frozenset)
    capacity: int = Field(default=1, ge=1, le=128)
    active_tasks: frozenset[str] = Field(default_factory=frozenset)
    state: WorkerState = WorkerState.ONLINE
    started_at: datetime = Field(default_factory=utcnow)
    heartbeat_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime

    def active_at(self, instant: datetime) -> bool:
        return self.state == WorkerState.ONLINE and self.expires_at > instant
