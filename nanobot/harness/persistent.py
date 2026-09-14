"""PostgreSQL state plus Redis coordination store for multi-process harnesses."""

# Optional clients have version-dependent typing and are imported lazily.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel

from nanobot.harness.audit import payload_digest, redact_audit_value
from nanobot.harness.models import (
    AgentInstance,
    AgentMessage,
    AgentSpec,
    Checkpoint,
    Task,
    WorkerLease,
    WorkerRegistration,
    Workflow,
)
from nanobot.harness.store import HarnessConflictError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS harness_agent_specs (
  spec_id text PRIMARY KEY, version integer NOT NULL, payload jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS harness_agent_instances (
  instance_id text PRIMARY KEY, workflow_id text NOT NULL, spec_id text NOT NULL, version integer NOT NULL DEFAULT 1, payload jsonb NOT NULL, updated_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS harness_agents_workflow_idx ON harness_agent_instances(workflow_id);
CREATE TABLE IF NOT EXISTS harness_workflows (
  workflow_id text PRIMARY KEY, owner_agent_id text NOT NULL, tenant_id text NOT NULL DEFAULT 'default', scope_key text, state text NOT NULL DEFAULT 'active', version integer NOT NULL DEFAULT 1, payload jsonb NOT NULL, created_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS harness_tasks (
  task_id text PRIMARY KEY, workflow_id text NOT NULL, tenant_id text NOT NULL DEFAULT 'default', idempotency_key text, state text NOT NULL, priority integer NOT NULL DEFAULT 0, available_at timestamptz NOT NULL DEFAULT now(), deadline_at timestamptz, assigned_agent_id text, lease_id text, fencing_token bigint NOT NULL DEFAULT 0, version integer NOT NULL DEFAULT 1, payload jsonb NOT NULL, updated_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS harness_tasks_ready_idx ON harness_tasks(workflow_id, state);
CREATE TABLE IF NOT EXISTS harness_checkpoints (
  checkpoint_id text PRIMARY KEY, agent_id text NOT NULL, sequence bigint NOT NULL, payload jsonb NOT NULL, created_at timestamptz NOT NULL,
  UNIQUE(agent_id, sequence)
);
CREATE INDEX IF NOT EXISTS harness_checkpoint_latest_idx ON harness_checkpoints(agent_id, sequence DESC);
CREATE TABLE IF NOT EXISTS harness_outbox (
  idempotency_key text PRIMARY KEY,
  recipient_agent_id text NOT NULL,
  payload jsonb NOT NULL,
  attempts integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  delivered_at timestamptz
);
CREATE INDEX IF NOT EXISTS harness_outbox_pending_idx ON harness_outbox(created_at) WHERE delivered_at IS NULL;
CREATE TABLE IF NOT EXISTS harness_audit_events (
  event_id bigserial PRIMARY KEY,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  component text NOT NULL,
  event_type text NOT NULL,
  tenant_id text,
  workflow_id text,
  task_id text,
  agent_id text,
  session_key text,
  trace_id text,
  payload jsonb NOT NULL,
  payload_sha256 text NOT NULL
);
CREATE INDEX IF NOT EXISTS harness_audit_trace_idx ON harness_audit_events(trace_id, event_id);
CREATE INDEX IF NOT EXISTS harness_audit_workflow_idx ON harness_audit_events(workflow_id, event_id);
CREATE INDEX IF NOT EXISTS harness_audit_session_idx ON harness_audit_events(session_key, event_id);
ALTER TABLE harness_workflows ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
ALTER TABLE harness_workflows ADD COLUMN IF NOT EXISTS scope_key text;
ALTER TABLE harness_workflows ADD COLUMN IF NOT EXISTS version integer NOT NULL DEFAULT 1;
ALTER TABLE harness_agent_instances ADD COLUMN IF NOT EXISTS version integer NOT NULL DEFAULT 1;
ALTER TABLE harness_tasks ADD COLUMN IF NOT EXISTS version integer NOT NULL DEFAULT 1;
ALTER TABLE harness_tasks ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
ALTER TABLE harness_tasks ADD COLUMN IF NOT EXISTS idempotency_key text;
ALTER TABLE harness_tasks ADD COLUMN IF NOT EXISTS lease_id text;
ALTER TABLE harness_tasks ADD COLUMN IF NOT EXISTS fencing_token bigint NOT NULL DEFAULT 0;
CREATE UNIQUE INDEX IF NOT EXISTS harness_workflow_scope_idx ON harness_workflows(tenant_id, scope_key) WHERE scope_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS harness_task_idempotency_idx ON harness_tasks(workflow_id,idempotency_key) WHERE idempotency_key IS NOT NULL;
"""

_MIGRATION_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS harness_schema_migrations (
  version integer PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);
"""

_MIGRATION_CONSTRAINTS = """
CREATE INDEX IF NOT EXISTS harness_tasks_tenant_state_idx
  ON harness_tasks(tenant_id,state,updated_at);
CREATE INDEX IF NOT EXISTS harness_agents_workflow_state_idx
  ON harness_agent_instances(workflow_id,updated_at);
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='harness_agents_workflow_fk') THEN
    ALTER TABLE harness_agent_instances ADD CONSTRAINT harness_agents_workflow_fk
      FOREIGN KEY(workflow_id) REFERENCES harness_workflows(workflow_id)
      ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED NOT VALID;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='harness_tasks_workflow_fk') THEN
    ALTER TABLE harness_tasks ADD CONSTRAINT harness_tasks_workflow_fk
      FOREIGN KEY(workflow_id) REFERENCES harness_workflows(workflow_id)
      ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED NOT VALID;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='harness_checkpoints_agent_fk') THEN
    ALTER TABLE harness_checkpoints ADD CONSTRAINT harness_checkpoints_agent_fk
      FOREIGN KEY(agent_id) REFERENCES harness_agent_instances(instance_id)
      ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED NOT VALID;
  END IF;
END $$;
"""

_MIGRATION_SCHEDULING = """
ALTER TABLE harness_tasks ADD COLUMN IF NOT EXISTS priority integer NOT NULL DEFAULT 0;
ALTER TABLE harness_tasks ADD COLUMN IF NOT EXISTS available_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE harness_tasks ADD COLUMN IF NOT EXISTS deadline_at timestamptz;
CREATE INDEX IF NOT EXISTS harness_tasks_dispatch_idx
  ON harness_tasks(tenant_id,state,priority DESC,available_at,updated_at);
"""

_MIGRATION_WORKFLOW_STATE = """
ALTER TABLE harness_workflows ADD COLUMN IF NOT EXISTS state text NOT NULL DEFAULT 'active';
UPDATE harness_workflows SET state=COALESCE(payload->>'state','active');
CREATE INDEX IF NOT EXISTS harness_workflows_retention_idx
  ON harness_workflows(tenant_id,state,created_at);
"""

_MIGRATION_AUDIT_EVENTS = """
CREATE TABLE IF NOT EXISTS harness_audit_events (
  event_id bigserial PRIMARY KEY,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  component text NOT NULL,
  event_type text NOT NULL,
  tenant_id text,
  workflow_id text,
  task_id text,
  agent_id text,
  session_key text,
  trace_id text,
  payload jsonb NOT NULL,
  payload_sha256 text NOT NULL
);
CREATE INDEX IF NOT EXISTS harness_audit_trace_idx ON harness_audit_events(trace_id, event_id);
CREATE INDEX IF NOT EXISTS harness_audit_workflow_idx ON harness_audit_events(workflow_id, event_id);
CREATE INDEX IF NOT EXISTS harness_audit_session_idx ON harness_audit_events(session_key, event_id);
"""

_MIGRATIONS = (
    (1, _SCHEMA),
    (2, _MIGRATION_CONSTRAINTS),
    (3, _MIGRATION_SCHEDULING),
    (4, _MIGRATION_WORKFLOW_STATE),
    (5, _MIGRATION_AUDIT_EVENTS),
)


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _model_from_json(model: type[_ModelT], payload: Any) -> _ModelT:
    if isinstance(payload, (str, bytes, bytearray)):
        return model.model_validate_json(payload)
    return model.model_validate(payload)


def _stream_entries(value: object) -> list[tuple[str, dict[str, str]]]:
    if not isinstance(value, (list, tuple)):
        return []
    entries: list[tuple[str, dict[str, str]]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        delivery_id, raw_fields = item
        if not isinstance(delivery_id, str) or not isinstance(raw_fields, dict):
            continue
        message = raw_fields.get("message")
        if isinstance(message, str):
            entries.append((delivery_id, {"message": message}))
    return entries


class PostgresRedisHarnessStore:
    """Durable state in PostgreSQL; ephemeral mailbox, dedupe and leases in Redis."""

    _RENEW_LEASE = """
local current = redis.call('GET', KEYS[1])
if not current then return 0 end
if cjson.decode(current)['lease_id'] ~= ARGV[1] then return -1 end
if ARGV[4] == '1' then redis.call('DEL', KEYS[1]); return 1 end
redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3])
return 1
"""
    _ACQUIRE_RESOURCES = """
for _, key in ipairs(KEYS) do
  local owner = redis.call('GET', key)
  if owner and owner ~= ARGV[1] then return 0 end
end
for _, key in ipairs(KEYS) do
  redis.call('SET', key, ARGV[1], 'PX', ARGV[2])
end
return 1
"""
    _RENEW_RESOURCES = """
for _, key in ipairs(KEYS) do
  if redis.call('GET', key) ~= ARGV[1] then return 0 end
end
for _, key in ipairs(KEYS) do redis.call('PEXPIRE', key, ARGV[2]) end
return 1
"""
    _RELEASE_RESOURCES = """
for _, key in ipairs(KEYS) do
  if redis.call('GET', key) == ARGV[1] then redis.call('DEL', key) end
end
return 1
"""

    def __init__(
        self,
        postgres_dsn: str,
        redis_url: str,
        *,
        key_prefix: str = "nanobot:harness",
        mailbox_ttl: int = 86_400,
        idempotency_ttl: int = 86_400,
        message_claim_idle_ms: int = 60_000,
        pool: Any | None = None,
        redis: Any | None = None,
    ) -> None:
        self._dsn = postgres_dsn
        self._redis_url = redis_url
        self._prefix = key_prefix.rstrip(":")
        self._mailbox_ttl = mailbox_ttl
        self._idempotency_ttl = idempotency_ttl
        self._message_claim_idle_ms = message_claim_idle_ms
        self._pool = pool
        self._redis = redis
        self._ready = False
        self._lock = asyncio.Lock()

    async def _clients(self) -> tuple[Any, Any]:
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as exc:
                raise RuntimeError("persistent harness requires nanobot-ai[memory]") from exc
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)
        if self._redis is None:
            try:
                from redis.asyncio import Redis
            except ImportError as exc:
                raise RuntimeError("persistent harness requires nanobot-ai[memory]") from exc
            self._redis = Redis.from_url(self._redis_url, decode_responses=True)
        if not self._ready:
            async with self._lock:
                if not self._ready:
                    async with self._pool.acquire() as connection:
                        await self._apply_migrations(connection)
                    self._ready = True
        return self._pool, self._redis

    @staticmethod
    async def _apply_migrations(connection: Any) -> None:
        """Apply ordered migrations once, serialized across all gateway processes."""
        async with connection.transaction():
            await connection.execute(_MIGRATION_BOOTSTRAP)
            await connection.execute("SELECT pg_advisory_xact_lock(727638817201)")
            current = int(
                await connection.fetchval(
                    "SELECT COALESCE(MAX(version),0) FROM harness_schema_migrations"
                )
                or 0
            )
            for version, sql in _MIGRATIONS:
                if version <= current:
                    continue
                await connection.execute(sql)
                await connection.execute(
                    "INSERT INTO harness_schema_migrations(version) VALUES($1)",
                    version,
                )

    def _mailbox_key(self, agent_id: str) -> str:
        return f"{self._prefix}:mailbox:{agent_id}"

    async def record_audit_event(
        self,
        event_type: str,
        *,
        component: str,
        tenant_id: str | None = None,
        workflow_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        session_key: str | None = None,
        trace_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append a redacted, durable audit event without mutating runtime state."""
        value = redact_audit_value(payload or {})
        if not isinstance(value, dict):  # defensive; redact_audit_value preserves mappings
            raise TypeError("audit payload must be a mapping")
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            await connection.execute(
                """INSERT INTO harness_audit_events
                     (component,event_type,tenant_id,workflow_id,task_id,agent_id,
                      session_key,trace_id,payload,payload_sha256)
                   VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10)""",
                component,
                event_type,
                tenant_id,
                workflow_id,
                task_id,
                agent_id,
                session_key,
                trace_id,
                json.dumps(value, ensure_ascii=False, default=str),
                payload_digest(value),
            )

    async def latest_audit_event_id(self) -> int:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            return int(
                await connection.fetchval("SELECT COALESCE(max(event_id), 0) FROM harness_audit_events")
                or 0
            )

    async def list_audit_events_after(
        self, event_id: int, *, event_types: frozenset[str], limit: int = 100
    ) -> list[dict[str, Any]]:
        if not event_types:
            return []
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT event_id,event_type,tenant_id,workflow_id,task_id,agent_id,
                          session_key,trace_id,payload
                     FROM harness_audit_events
                    WHERE event_id > $1 AND event_type = ANY($2::text[])
                    ORDER BY event_id ASC LIMIT $3""",
                event_id,
                list(event_types),
                limit,
            )
        return [dict(row) for row in rows]

    @staticmethod
    def _mailbox_group() -> str:
        return "agent-runtime"

    def _dedupe_key(self, key: str) -> str:
        return f"{self._prefix}:idempotency:{key}"

    def _lease_key(self, task_id: str) -> str:
        return f"{self._prefix}:lease:{task_id}"

    def _resource_key(self, resource: str) -> str:
        digest = hashlib.sha256(resource.encode("utf-8")).hexdigest()
        return f"{self._prefix}:resource:{digest}"

    def _worker_key(self, worker_id: str) -> str:
        return f"{self._prefix}:worker:{worker_id}"

    def _worker_index_key(self, tenant_id: str) -> str:
        return f"{self._prefix}:workers:{tenant_id}"

    async def put_spec(self, spec: AgentSpec) -> None:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            result = await connection.execute(
                """INSERT INTO harness_agent_specs(spec_id,version,payload) VALUES($1,$2,$3::jsonb)
                   ON CONFLICT(spec_id) DO UPDATE SET version=EXCLUDED.version,payload=EXCLUDED.payload,updated_at=now()
                   WHERE harness_agent_specs.version <= EXCLUDED.version""",
                spec.spec_id,
                spec.version,
                spec.model_dump_json(),
            )
        if result.endswith(" 0"):
            raise HarnessConflictError("cannot replace an agent spec with an older version")

    async def put_agent(self, agent: AgentInstance) -> None:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            version = await self._put_agent(connection, agent)
        if version is None:
            raise HarnessConflictError(f"stale agent version for {agent.instance_id}")
        agent.version = int(version)

    async def _put_agent(self, connection: Any, agent: AgentInstance) -> int | None:
        next_version = agent.version + 1
        updated = agent.model_copy(update={"version": next_version})
        version = await connection.fetchval(
            """INSERT INTO harness_agent_instances
                 (instance_id,workflow_id,spec_id,version,payload,updated_at)
               VALUES($1,$2,$3,$4,$5::jsonb,$6) ON CONFLICT(instance_id) DO UPDATE SET
                 workflow_id=EXCLUDED.workflow_id,spec_id=EXCLUDED.spec_id,
                 version=harness_agent_instances.version+1,payload=$7::jsonb,
                 updated_at=EXCLUDED.updated_at
               WHERE harness_agent_instances.version=$8 RETURNING version""",
            agent.instance_id,
            agent.workflow_id,
            agent.spec_id,
            agent.version,
            agent.model_dump_json(),
            agent.updated_at,
            updated.model_dump_json(),
            agent.version,
        )
        return version

    async def put_workflow(self, workflow: Workflow) -> None:
        pool, _ = await self._clients()
        async with pool.acquire() as connection, connection.transaction():
            next_version = workflow.version + 1
            updated = workflow.model_copy(update={"version": next_version})
            version = await connection.fetchval(
                """INSERT INTO harness_workflows
                     (workflow_id,owner_agent_id,tenant_id,scope_key,state,version,payload,created_at)
                   VALUES($1,$2,$3,$4,$5,$6,$7::jsonb,$8) ON CONFLICT(workflow_id) DO UPDATE SET
                   owner_agent_id=EXCLUDED.owner_agent_id,tenant_id=EXCLUDED.tenant_id,
                   scope_key=EXCLUDED.scope_key,state=EXCLUDED.state,
                   version=harness_workflows.version+1,
                   payload=$9::jsonb WHERE harness_workflows.version=$10 RETURNING version""",
                workflow.workflow_id,
                workflow.owner_agent_id,
                workflow.tenant_id,
                workflow.scope_key,
                workflow.state.value,
                workflow.version,
                workflow.model_dump_json(),
                workflow.created_at,
                updated.model_dump_json(),
                workflow.version,
            )
            if version is None:
                raise HarnessConflictError(f"stale workflow version for {workflow.workflow_id}")
            workflow.version = int(version)

    async def get_or_create_workflow(
        self,
        workflow: Workflow,
        owner: AgentInstance,
    ) -> tuple[Workflow, AgentInstance, bool]:
        pool, _ = await self._clients()
        async with pool.acquire() as connection, connection.transaction():
            inserted = await connection.fetchval(
                """INSERT INTO harness_workflows
                     (workflow_id,owner_agent_id,tenant_id,scope_key,state,version,payload,created_at)
                   VALUES($1,$2,$3,$4,$5,$6,$7::jsonb,$8)
                   ON CONFLICT(tenant_id,scope_key) WHERE scope_key IS NOT NULL DO NOTHING
                   RETURNING workflow_id""",
                workflow.workflow_id,
                workflow.owner_agent_id,
                workflow.tenant_id,
                workflow.scope_key,
                workflow.state.value,
                workflow.version,
                workflow.model_dump_json(),
                workflow.created_at,
            )
            if inserted:
                await connection.execute(
                    """INSERT INTO harness_agent_instances
                         (instance_id,workflow_id,spec_id,version,payload,updated_at)
                       VALUES($1,$2,$3,$4,$5::jsonb,$6)""",
                    owner.instance_id,
                    owner.workflow_id,
                    owner.spec_id,
                    owner.version,
                    owner.model_dump_json(),
                    owner.updated_at,
                )
                return workflow, owner, True
            row = await connection.fetchrow(
                """SELECT w.payload AS workflow, a.payload AS owner
                     FROM harness_workflows w JOIN harness_agent_instances a
                       ON a.instance_id=w.owner_agent_id
                    WHERE w.tenant_id=$1 AND w.scope_key=$2""",
                workflow.tenant_id,
                workflow.scope_key,
            )
        if row is None:
            raise HarnessConflictError("workflow scope exists without an owner")
        existing = Workflow.model_validate(_model_from_json(Workflow, row["workflow"]))
        existing_owner = AgentInstance.model_validate(_model_from_json(AgentInstance, row["owner"]))
        return existing, existing_owner, False

    async def get_workflow(self, workflow_id: str) -> Workflow:
        return Workflow.model_validate(
            _model_from_json(
                Workflow,
                await self._payload(
                    "SELECT payload FROM harness_workflows WHERE workflow_id=$1", workflow_id
                ),
            )
        )

    async def get_workflow_by_scope(self, tenant_id: str, scope_key: str) -> Workflow | None:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            payload = await connection.fetchval(
                "SELECT payload FROM harness_workflows WHERE tenant_id=$1 AND scope_key=$2",
                tenant_id,
                scope_key,
            )
        return Workflow.model_validate(_model_from_json(Workflow, payload)) if payload else None

    async def list_active_workflow_ids(
        self, tenant_id: str, *, limit: int = 1_000
    ) -> list[str]:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT workflow_id FROM harness_workflows
                    WHERE tenant_id=$1 AND state='active'
                    ORDER BY created_at LIMIT $2""",
                tenant_id,
                limit,
            )
        return [str(row["workflow_id"]) for row in rows]

    async def _put_task(self, connection: Any, task: Task) -> int:
        next_version = task.version + 1
        updated = task.model_copy(update={"version": next_version})
        version = await connection.fetchval(
            """INSERT INTO harness_tasks
                 (task_id,workflow_id,tenant_id,idempotency_key,state,priority,available_at,deadline_at,assigned_agent_id,lease_id,fencing_token,version,payload,updated_at)
               VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb,$14) ON CONFLICT(task_id) DO UPDATE SET
                 state=EXCLUDED.state,assigned_agent_id=EXCLUDED.assigned_agent_id,
                 priority=EXCLUDED.priority,available_at=EXCLUDED.available_at,
                 deadline_at=EXCLUDED.deadline_at,
                 lease_id=EXCLUDED.lease_id,fencing_token=EXCLUDED.fencing_token,
                 tenant_id=EXCLUDED.tenant_id,idempotency_key=EXCLUDED.idempotency_key,
                 version=harness_tasks.version+1,payload=$15::jsonb,updated_at=EXCLUDED.updated_at
               WHERE harness_tasks.version=$16 RETURNING version""",
            task.task_id,
            task.workflow_id,
            task.tenant_id,
            task.idempotency_key,
            task.state.value,
            task.priority,
            task.available_at,
            task.deadline_at,
            task.assigned_agent_id,
            task.lease_id,
            task.fencing_token,
            task.version,
            task.model_dump_json(),
            task.updated_at,
            updated.model_dump_json(),
            task.version,
        )
        if version is None:
            raise HarnessConflictError(f"stale task version for {task.task_id}")
        return int(version)

    async def put_task(self, task: Task) -> None:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            version = await self._put_task(connection, task)
        task.version = version

    async def commit_task_agent(
        self,
        task: Task,
        agent: AgentInstance,
        *,
        outbox_message: AgentMessage | None = None,
    ) -> None:
        """Atomically commit Task/Agent state and the durable result Outbox row."""
        pool, _ = await self._clients()
        async with pool.acquire() as connection, connection.transaction():
            task_version = await self._put_task(connection, task)
            agent_version = await self._put_agent(connection, agent)
            if agent_version is None:
                raise HarnessConflictError(f"stale agent version for {agent.instance_id}")
            if outbox_message is not None:
                await connection.execute(
                    """INSERT INTO harness_outbox
                         (idempotency_key,recipient_agent_id,payload)
                       VALUES($1,$2,$3::jsonb)
                       ON CONFLICT(idempotency_key) DO NOTHING""",
                    outbox_message.idempotency_key,
                    outbox_message.recipient_agent_id,
                    outbox_message.model_dump_json(),
                )
        task.version = task_version
        agent.version = int(agent_version)

    async def get_or_create_task(self, task: Task) -> tuple[Task, bool]:
        if task.idempotency_key is None:
            await self.put_task(task)
            return task, True
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            payload = await connection.fetchval(
                """INSERT INTO harness_tasks
                     (task_id,workflow_id,tenant_id,idempotency_key,state,priority,
                      available_at,deadline_at,assigned_agent_id,lease_id,fencing_token,
                      version,payload,updated_at)
                   VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb,$14)
                   ON CONFLICT(workflow_id,idempotency_key)
                     WHERE idempotency_key IS NOT NULL DO NOTHING
                   RETURNING payload""",
                task.task_id,
                task.workflow_id,
                task.tenant_id,
                task.idempotency_key,
                task.state.value,
                task.priority,
                task.available_at,
                task.deadline_at,
                task.assigned_agent_id,
                task.lease_id,
                task.fencing_token,
                task.version,
                task.model_dump_json(),
                task.updated_at,
            )
            if payload is not None:
                return task, True
            payload = await connection.fetchval(
                """SELECT payload FROM harness_tasks
                    WHERE workflow_id=$1 AND idempotency_key=$2""",
                task.workflow_id,
                task.idempotency_key,
            )
        if payload is None:
            raise HarnessConflictError("idempotent task insert could not be resolved")
        return Task.model_validate(_model_from_json(Task, payload)), False

    async def get_task_by_idempotency(
        self, workflow_id: str, idempotency_key: str
    ) -> Task | None:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            payload = await connection.fetchval(
                """SELECT payload FROM harness_tasks
                    WHERE workflow_id=$1 AND idempotency_key=$2""",
                workflow_id,
                idempotency_key,
            )
        return Task.model_validate(_model_from_json(Task, payload)) if payload else None

    async def get_spec(self, spec_id: str) -> AgentSpec:
        return AgentSpec.model_validate(
            _model_from_json(
                AgentSpec,
                await self._payload(
                    "SELECT payload FROM harness_agent_specs WHERE spec_id=$1", spec_id
                ),
            )
        )

    async def get_agent(self, agent_id: str) -> AgentInstance:
        return AgentInstance.model_validate(
            _model_from_json(
                AgentInstance,
                await self._payload(
                    "SELECT payload FROM harness_agent_instances WHERE instance_id=$1", agent_id
                ),
            )
        )

    async def get_task(self, task_id: str) -> Task:
        return Task.model_validate(
            _model_from_json(
                Task,
                await self._payload("SELECT payload FROM harness_tasks WHERE task_id=$1", task_id),
            )
        )

    async def _payload(self, query: str, value: str) -> Any:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            payload = await connection.fetchval(query, value)
        if payload is None:
            raise KeyError(value)
        return payload

    async def list_tasks(self, workflow_id: str) -> list[Task]:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT payload FROM harness_tasks WHERE workflow_id=$1", workflow_id
            )
        return [Task.model_validate(_model_from_json(Task, row["payload"])) for row in rows]

    async def count_tasks(self, tenant_id: str, states: frozenset[str]) -> int:
        if not states:
            return 0
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            return int(
                await connection.fetchval(
                    """SELECT COUNT(*) FROM harness_tasks
                        WHERE tenant_id=$1 AND state=ANY($2::text[])""",
                    tenant_id,
                    list(states),
                )
                or 0
            )

    async def list_ready_tasks(self, tenant_id: str, *, limit: int) -> list[Task]:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT payload FROM harness_tasks
                    WHERE tenant_id=$1 AND state='ready' AND available_at <= now()
                      AND (deadline_at IS NULL OR deadline_at > now())
                    ORDER BY priority DESC,available_at,updated_at
                    LIMIT $2""",
                tenant_id,
                limit,
            )
        return [Task.model_validate(_model_from_json(Task, row["payload"])) for row in rows]

    async def list_expired_tasks(
        self, tenant_id: str, *, limit: int = 100
    ) -> list[Task]:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT payload FROM harness_tasks
                    WHERE tenant_id=$1 AND state=ANY($2::text[])
                      AND deadline_at IS NOT NULL AND deadline_at <= now()
                    ORDER BY deadline_at LIMIT $3""",
                tenant_id,
                ["pending", "ready"],
                limit,
            )
        return [Task.model_validate(_model_from_json(Task, row["payload"])) for row in rows]

    async def list_dead_letters(
        self, tenant_id: str, *, limit: int = 100
    ) -> list[Task]:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT payload FROM harness_tasks
                    WHERE tenant_id=$1 AND state='dead_letter'
                    ORDER BY updated_at DESC LIMIT $2""",
                tenant_id,
                limit,
            )
        return [Task.model_validate(_model_from_json(Task, row["payload"])) for row in rows]

    async def send(self, message: AgentMessage) -> bool:
        pool, redis = await self._clients()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """INSERT INTO harness_outbox(idempotency_key,recipient_agent_id,payload)
                   VALUES($1,$2,$3::jsonb) ON CONFLICT(idempotency_key) DO UPDATE SET
                   idempotency_key=EXCLUDED.idempotency_key
                   RETURNING delivered_at""",
                message.idempotency_key,
                message.recipient_agent_id,
                message.model_dump_json(),
            )
        if row is not None and row["delivered_at"] is not None:
            await self.record_audit_event(
                "harness.mailbox.duplicate",
                component="harness_mailbox",
                tenant_id=message.tenant_id,
                workflow_id=message.workflow_id,
                task_id=message.task_id,
                agent_id=message.sender_agent_id,
                payload=message.model_dump(mode="json"),
            )
            return False
        try:
            inserted = await self._deliver_message(redis, message)
        except Exception as exc:
            async with pool.acquire() as connection:
                await connection.execute(
                    "UPDATE harness_outbox SET attempts=attempts+1 WHERE idempotency_key=$1",
                    message.idempotency_key,
                )
            logger.warning(
                "Harness outbox delivery deferred for {}: {}",
                message.idempotency_key,
                exc,
            )
            await self.record_audit_event(
                "harness.outbox.deferred",
                component="harness_mailbox",
                tenant_id=message.tenant_id,
                workflow_id=message.workflow_id,
                task_id=message.task_id,
                agent_id=message.sender_agent_id,
                payload={"message": message.model_dump(mode="json"), "error": str(exc)},
            )
            return True
        async with pool.acquire() as connection:
            await connection.execute(
                """UPDATE harness_outbox SET delivered_at=now(),attempts=attempts+1
                    WHERE idempotency_key=$1""",
                message.idempotency_key,
            )
        await self.record_audit_event(
            "harness.outbox.delivered",
            component="harness_mailbox",
            tenant_id=message.tenant_id,
            workflow_id=message.workflow_id,
            task_id=message.task_id,
            agent_id=message.sender_agent_id,
            payload={"inserted": inserted, "message": message.model_dump(mode="json")},
        )
        return inserted

    async def _deliver_message(self, redis: Any, message: AgentMessage) -> bool:
        inserted = await redis.set(
            self._dedupe_key(message.idempotency_key), "1", ex=self._idempotency_ttl, nx=True
        )
        if not inserted:
            return False
        key = self._mailbox_key(message.recipient_agent_id)
        try:
            await redis.xadd(
                key,
                {"message": message.model_dump_json()},
                maxlen=1000,
                approximate=True,
            )
            await redis.expire(key, self._mailbox_ttl)
        except Exception:
            await redis.delete(self._dedupe_key(message.idempotency_key))
            raise
        return True

    async def flush_outbox(self, *, limit: int = 100) -> int:
        pool, redis = await self._clients()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT idempotency_key,payload FROM harness_outbox
                    WHERE delivered_at IS NULL ORDER BY created_at LIMIT $1""",
                limit,
            )
        delivered = 0
        for row in rows:
            message = AgentMessage.model_validate(_model_from_json(AgentMessage, row["payload"]))
            try:
                await self._deliver_message(redis, message)
            except Exception as exc:
                logger.warning(
                    "Harness outbox retry failed for {}: {}",
                    row["idempotency_key"],
                    exc,
                )
                continue
            async with pool.acquire() as connection:
                await connection.execute(
                    """UPDATE harness_outbox SET delivered_at=now(),attempts=attempts+1
                        WHERE idempotency_key=$1""",
                    row["idempotency_key"],
                )
            await self.record_audit_event(
                "harness.outbox.flushed",
                component="harness_mailbox",
                tenant_id=message.tenant_id,
                workflow_id=message.workflow_id,
                task_id=message.task_id,
                agent_id=message.sender_agent_id,
                payload={"message": message.model_dump(mode="json")},
            )
            delivered += 1
        return delivered

    async def inbox(self, agent_id: str, *, now: datetime, limit: int) -> list[AgentMessage]:
        _, redis = await self._clients()
        values = await redis.xrevrange(self._mailbox_key(agent_id), count=limit)
        messages = [
            AgentMessage.model_validate_json(fields["message"]).model_copy(
                update={"delivery_id": delivery_id}
            )
            for delivery_id, fields in reversed(values)
        ]
        return [msg for msg in messages if msg.expires_at is None or msg.expires_at > now]

    async def claim_messages(
        self,
        agent_id: str,
        *,
        consumer_id: str,
        now: datetime,
        limit: int,
    ) -> list[AgentMessage]:
        _, redis = await self._clients()
        key = self._mailbox_key(agent_id)
        group = self._mailbox_group()
        try:
            await redis.xgroup_create(key, group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        claimed = await redis.xautoclaim(
            key,
            group,
            consumer_id,
            min_idle_time=self._message_claim_idle_ms,
            start_id="0-0",
            count=limit,
        )
        entries = _stream_entries(
            claimed[1] if isinstance(claimed, (list, tuple)) and len(claimed) > 1 else []
        )
        if len(entries) < limit:
            rows = await redis.xreadgroup(
                group,
                consumer_id,
                {key: ">"},
                count=limit - len(entries),
            )
            if (
                isinstance(rows, (list, tuple))
                and rows
                and isinstance(rows[0], (list, tuple))
                and len(rows[0]) > 1
            ):
                entries.extend(_stream_entries(rows[0][1]))
        messages = [
            AgentMessage.model_validate_json(fields["message"]).model_copy(
                update={"delivery_id": delivery_id}
            )
            for delivery_id, fields in entries
        ]
        valid = [msg for msg in messages if msg.expires_at is None or msg.expires_at > now]
        expired = [msg for msg in messages if msg not in valid]
        if expired:
            await self.ack_messages(agent_id, expired)
        for message in valid:
            await self.record_audit_event(
                "harness.mailbox.claimed",
                component="harness_mailbox",
                tenant_id=message.tenant_id,
                workflow_id=message.workflow_id,
                task_id=message.task_id,
                agent_id=agent_id,
                payload={"consumer_id": consumer_id, "message": message.model_dump(mode="json")},
            )
        return valid

    async def ack_messages(self, agent_id: str, messages: list[AgentMessage]) -> None:
        ids = [message.delivery_id for message in messages if message.delivery_id]
        if not ids:
            return
        _, redis = await self._clients()
        await redis.xack(self._mailbox_key(agent_id), self._mailbox_group(), *ids)
        for message in messages:
            await self.record_audit_event(
                "harness.mailbox.acknowledged",
                component="harness_mailbox",
                tenant_id=message.tenant_id,
                workflow_id=message.workflow_id,
                task_id=message.task_id,
                agent_id=agent_id,
                payload={"delivery_id": message.delivery_id, "message_id": message.message_id},
            )

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            try:
                await connection.execute(
                    """INSERT INTO harness_checkpoints(checkpoint_id,agent_id,sequence,payload,created_at)
                       VALUES($1,$2,$3,$4::jsonb,$5)""",
                    checkpoint.checkpoint_id,
                    checkpoint.agent_id,
                    checkpoint.sequence,
                    checkpoint.model_dump_json(),
                    checkpoint.created_at,
                )
            except Exception as exc:
                if getattr(exc, "sqlstate", None) == "23505":
                    raise HarnessConflictError("checkpoint sequence must increase") from exc
                raise

    async def latest_checkpoint(self, agent_id: str) -> Checkpoint | None:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            payload = await connection.fetchval(
                "SELECT payload FROM harness_checkpoints WHERE agent_id=$1 ORDER BY sequence DESC LIMIT 1",
                agent_id,
            )
        return Checkpoint.model_validate(_model_from_json(Checkpoint, payload)) if payload else None

    async def acquire_lease(self, lease: WorkerLease, *, now: datetime) -> bool:
        del now
        _, redis = await self._clients()
        ttl_ms = max(1, int((lease.expires_at - lease.acquired_at).total_seconds() * 1000))
        return bool(
            await redis.set(
                self._lease_key(lease.task_id), lease.model_dump_json(), px=ttl_ms, nx=True
            )
        )

    async def get_lease(self, task_id: str) -> WorkerLease | None:
        _, redis = await self._clients()
        payload = await redis.get(self._lease_key(task_id))
        if not payload:
            return None
        try:
            return WorkerLease.model_validate_json(payload)
        except Exception:
            logger.warning("Discarding malformed Harness lease for task {}", task_id)
            return None

    async def lease_active(self, task_id: str, *, now: datetime) -> bool:
        _, redis = await self._clients()
        payload = await redis.get(self._lease_key(task_id))
        if not payload:
            return False
        try:
            return WorkerLease.model_validate_json(payload).active_at(now)
        except Exception:
            logger.warning("Discarding malformed Harness lease for task {}", task_id)
            return False

    async def update_lease(self, lease: WorkerLease) -> None:
        _, redis = await self._clients()
        ttl_ms = max(1, int((lease.expires_at - lease.heartbeat_at).total_seconds() * 1000))
        result = int(
            await redis.eval(
                self._RENEW_LEASE,
                1,
                self._lease_key(lease.task_id),
                lease.lease_id,
                lease.model_dump_json(),
                ttl_ms,
                "1" if lease.released_at else "0",
            )
        )
        if result != 1:
            raise HarnessConflictError("worker no longer owns this lease")

    async def acquire_resources(self, lease: WorkerLease, *, now: datetime) -> bool:
        if not lease.resource_keys:
            return True
        _, redis = await self._clients()
        keys = [self._resource_key(resource) for resource in sorted(lease.resource_keys)]
        ttl_ms = max(1, int((lease.expires_at - now).total_seconds() * 1000))
        return bool(
            await redis.eval(
                self._ACQUIRE_RESOURCES,
                len(keys),
                *keys,
                lease.lease_id,
                ttl_ms,
            )
        )

    async def renew_resources(self, lease: WorkerLease) -> bool:
        if not lease.resource_keys:
            return True
        _, redis = await self._clients()
        keys = [self._resource_key(resource) for resource in sorted(lease.resource_keys)]
        ttl_ms = max(1, int((lease.expires_at - lease.heartbeat_at).total_seconds() * 1000))
        return bool(
            await redis.eval(
                self._RENEW_RESOURCES,
                len(keys),
                *keys,
                lease.lease_id,
                ttl_ms,
            )
        )

    async def release_resources(self, lease: WorkerLease) -> None:
        if not lease.resource_keys:
            return
        _, redis = await self._clients()
        keys = [self._resource_key(resource) for resource in sorted(lease.resource_keys)]
        await redis.eval(
            self._RELEASE_RESOURCES,
            len(keys),
            *keys,
            lease.lease_id,
        )

    async def register_worker(
        self, worker: WorkerRegistration, *, ttl_seconds: int
    ) -> None:
        _, redis = await self._clients()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.set(
                self._worker_key(worker.worker_id),
                worker.model_dump_json(),
                ex=ttl_seconds,
            )
            pipe.sadd(self._worker_index_key(worker.tenant_id), worker.worker_id)
            pipe.expire(self._worker_index_key(worker.tenant_id), ttl_seconds * 2)
            await pipe.execute()

    async def list_workers(
        self, tenant_id: str, *, now: datetime
    ) -> list[WorkerRegistration]:
        _, redis = await self._clients()
        index = self._worker_index_key(tenant_id)
        worker_ids = list(await redis.smembers(index))
        if not worker_ids:
            return []
        payloads = await redis.mget([self._worker_key(worker_id) for worker_id in worker_ids])
        workers: list[WorkerRegistration] = []
        stale: list[str] = []
        for worker_id, payload in zip(worker_ids, payloads):
            if not payload:
                stale.append(worker_id)
                continue
            try:
                worker = WorkerRegistration.model_validate_json(payload)
            except Exception:
                stale.append(worker_id)
                continue
            if worker.expires_at <= now:
                stale.append(worker_id)
                continue
            workers.append(worker)
        if stale:
            await redis.srem(index, *stale)
        return workers

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
        if self._pool is not None:
            await self._pool.close()

    async def health(self) -> dict[str, bool]:
        postgres = False
        redis_ok = False
        try:
            pool, redis = await self._clients()
            async with pool.acquire() as connection:
                postgres = bool(await connection.fetchval("SELECT 1"))
            redis_ok = bool(await redis.ping())
        except Exception as exc:
            logger.warning("Harness health check failed: {}", exc)
        return {
            "store": postgres and redis_ok,
            "postgres": postgres,
            "redis": redis_ok,
        }

    async def diagnostics(self, tenant_id: str | None = None) -> dict[str, int | float]:
        pool, _ = await self._clients()
        async with pool.acquire() as connection:
            task_rows = await connection.fetch(
                """SELECT state,count(*) AS count FROM harness_tasks
                   WHERE ($1::text IS NULL OR tenant_id=$1) GROUP BY state""",
                tenant_id,
            )
            workflows = int(
                await connection.fetchval(
                    """SELECT count(*) FROM harness_workflows
                       WHERE ($1::text IS NULL OR tenant_id=$1)""",
                    tenant_id,
                )
                or 0
            )
            agents = int(
                await connection.fetchval(
                    """SELECT count(*) FROM harness_agent_instances a
                       JOIN harness_workflows w ON w.workflow_id=a.workflow_id
                       WHERE ($1::text IS NULL OR w.tenant_id=$1)""",
                    tenant_id,
                )
                or 0
            )
            checkpoints = int(
                await connection.fetchval(
                    """SELECT count(*) FROM harness_checkpoints c
                       JOIN harness_agent_instances a ON a.instance_id=c.agent_id
                       JOIN harness_workflows w ON w.workflow_id=a.workflow_id
                       WHERE ($1::text IS NULL OR w.tenant_id=$1)""",
                    tenant_id,
                )
                or 0
            )
            outbox_pending = int(
                await connection.fetchval(
                    """SELECT count(*) FROM harness_outbox
                       WHERE delivered_at IS NULL
                         AND ($1::text IS NULL OR payload->>'tenant_id'=$1)""",
                    tenant_id,
                )
                or 0
            )
            oldest = float(
                await connection.fetchval(
                    """SELECT COALESCE(EXTRACT(EPOCH FROM (now()-min(created_at))),0)
                       FROM harness_outbox WHERE delivered_at IS NULL
                         AND ($1::text IS NULL OR payload->>'tenant_id'=$1)""",
                    tenant_id,
                )
                or 0
            )
        values: dict[str, int | float] = {
            "workflows": workflows,
            "agents": agents,
            "checkpoints": checkpoints,
            "outbox_pending": outbox_pending,
            "outbox_oldest_seconds": max(0.0, oldest),
        }
        values.update({f"tasks.{row['state']}": int(row["count"]) for row in task_rows})
        return values

    async def prune(
        self,
        *,
        before: datetime,
        tenant_id: str | None = None,
    ) -> dict[str, int]:
        """Delete old terminal workflow aggregates and delivered Outbox rows."""
        pool, _ = await self._clients()
        async with pool.acquire() as connection, connection.transaction():
            counts = await connection.fetchrow(
                """SELECT
                     count(DISTINCT w.workflow_id) AS workflows,
                     count(DISTINCT t.task_id) AS tasks,
                     count(DISTINCT a.instance_id) AS agents
                   FROM harness_workflows w
                   LEFT JOIN harness_tasks t ON t.workflow_id=w.workflow_id
                   LEFT JOIN harness_agent_instances a ON a.workflow_id=w.workflow_id
                   WHERE w.state=ANY($1::text[]) AND w.created_at < $2
                     AND ($3::text IS NULL OR w.tenant_id=$3)""",
                ["succeeded", "failed", "cancelled"],
                before,
                tenant_id,
            )
            outbox_rows = await connection.fetch(
                """DELETE FROM harness_outbox
                   WHERE delivered_at IS NOT NULL AND delivered_at < $1
                     AND ($2::text IS NULL OR payload->>'tenant_id'=$2)
                   RETURNING idempotency_key""",
                before,
                tenant_id,
            )
            workflow_rows = await connection.fetch(
                """DELETE FROM harness_workflows
                   WHERE state=ANY($1::text[]) AND created_at < $2
                     AND ($3::text IS NULL OR tenant_id=$3)
                   RETURNING workflow_id""",
                ["succeeded", "failed", "cancelled"],
                before,
                tenant_id,
            )
        return {
            "workflows": len(workflow_rows),
            "tasks": int(counts["tasks"] or 0),
            "agents": int(counts["agents"] or 0),
            "outbox": len(outbox_rows),
        }
