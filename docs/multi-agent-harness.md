# Production Multi-Agent Harness

GitHubResearchAgent's Harness is the control plane for supervised multi-agent execution.
GitHub discovery is one Worker capability; it is not the system's orchestration model.

## Runtime topology

```text
User session
  -> stable Workflow + Supervisor AgentInstance
      -> Planner / Worker / Reviewer Task DAG and quality gates
          -> tenant priority queue + admission backpressure
          -> registered Worker pools + heartbeat/drain
          -> role and capability admission
          -> Redis worker lease + heartbeat
          -> isolated RuntimeContext + tool allowlist
          -> AgentRunner or distributed HarnessWorker
      -> Redis Streams result mailbox
      -> Supervisor RuntimeContext
      -> final user response
```

PostgreSQL is authoritative for AgentSpec, AgentInstance, Workflow, Task,
Checkpoint, the durable message Outbox, and the append-only audit trail. Redis owns ephemeral coordination:
Streams mailboxes, idempotency keys and expiring Worker leases.

## Logical VFS

The runtime also exposes a logical VFS namespace above the host filesystem.
`LogicalVFS` derives physical roots from `(tenant_id, user_id, task_id)` and
never uses raw user or channel identifiers as directory names. Repository
downloads and task workspaces created during a repository turn are placed under
the active user's project namespace; a different user receives a different
cache root even when the GitHub repository is identical. The context builder
adds the active user's private skill root to the request-local skill search
path, while built-in and workspace-level skills remain shared and read-only.
User/task workspaces are read-write.

File tools enforce the VFS byte/file quota before writes that target a managed
namespace. Expired task and project namespaces can be removed by
`cleanup_expired()`; cleanup re-validates that every target is below the VFS
root. This layer is a policy and namespace abstraction, not a FUSE filesystem:
Bubblewrap remains responsible for process/mount isolation when enabled.

The six control-plane abstractions are deliberately separate:

| Abstraction | Responsibility | Authoritative storage |
|---|---|---|
| `AgentSpec` | immutable role, capabilities, tools and budgets | PostgreSQL |
| `AgentInstance` | one runtime identity and current lifecycle state | PostgreSQL |
| `Workflow` | tenant-scoped goal and validated Task DAG | PostgreSQL |
| `Task` | idempotent unit of work, dependency and result | PostgreSQL |
| `AgentMessage` | point-to-point handoff envelope | PostgreSQL Outbox + Redis Stream |
| `Checkpoint` | monotonic recovery state owned by one AgentInstance | PostgreSQL |

## Roles

- `supervisor`: owns the user goal, Workflow and synthesis.
- `planner`: decomposes research work; read-only tools.
- `worker`: performs bounded research, repository analysis or code work.
- `reviewer`: verifies evidence and rejects unsupported conclusions; read-only tools.

The Scheduler admits a Task only when both `required_role` and
`required_capabilities` match the immutable AgentSpec. The runtime then filters the
actual ToolRegistry through `allowed_tools`; the allowlist is enforced, not advisory.
Repository-domain experts are registered as versioned, narrowed `AgentSpec` values;
their configured repository tools become the enforced Harness allowlist rather than a
second advisory role definition.

## Repository workflow integration

`repo_agent` retains domain records for repository identity, approval binding and the
user-facing audit, but it no longer owns a separate scheduling control plane. Complex
code work materializes its Planner and every `PlanStep` as one Harness DAG. Academic
project research similarly materializes discovery, per-repository download, parallel
source analysis and synthesis Tasks. Repository path/name data travels as bounded Task
metadata, so an independently deployed Worker can reconstruct the same isolated tool
scope without receiving the parent transcript.

Dynamic workflows are created unsealed. Completing the Planner or discovery stage does
not make the Workflow terminal while later Tasks are still being added. The orchestrator
seals the DAG only after all expected Tasks exist; only a sealed all-successful Workflow
may transition to `succeeded`.

`/rere` is admitted as a `POOL` task. The Gateway persists the research request and
returns its research/workflow IDs; the independently deployed `harness-worker` owns
discovery, cloning, source analysis and synthesis. Repository source-analysis calls use
the configured `bwrap` child backend when enabled. Query `/rere status <research-id>`
to retrieve the persisted status or final report.

## Context isolation

Each child LLM call is constructed from only:

1. its role prompt;
2. its explicit Task input;
3. successful direct-dependency results;
4. messages addressed to its AgentInstance;
5. its latest sanitized Checkpoint.

Parent and sibling transcripts are never copied. Subagents cannot read
`memory/history.jsonl`, are always workspace-restricted, and do not receive `spawn`
or channel messaging tools. Context has message, character, token and per-message
budgets. Direct messages and dependency results are JSON-encoded as untrusted data.
Credential-like Checkpoint fields are removed and serialized size is bounded. Child
results are injected through the Supervisor's bounded runtime-context provider rather
than copied into notification text.

Planner and Reviewer roles have read-only tool surfaces. Reviewer gates require a
machine-readable `ReviewResult`; an arbitrary success string cannot approve a gate.
Worker Tasks can declare
`resource_keys`; a Redis-backed multi-resource lease permits only one active writer for
the same workspace while unrelated work remains parallel. Resource ownership is
renewed with the Task lease and checked under the same fencing identity.
The `spawn` tool exposes `exclusive_workspace=true` for file-mutating work; declarative
role DAGs set the equivalent `PlannedWorkItem.resource_keys` explicitly.

## Delivery and recovery

Redis Streams uses one consumer group per Agent mailbox. Delivery supports ACK and
idle-message recovery through `XAUTOCLAIM`. PostgreSQL Outbox records are written
before Redis delivery; a later Worker admission or explicit `flush_outbox()` replays
deferred events after Redis recovery.

Supervisor report messages remain pending while their RuntimeContext is only being
assembled. The provider supplies a post-persistence callback, and the AgentLoop ACKs
the claimed Stream entries only after the corresponding conversation turn is durably
saved. A crash during prompt construction or model execution therefore causes safe
at-least-once redelivery instead of losing the child result. The lightweight system
notification merely wakes the session; the result body has one authoritative path—the
Harness report envelope.

Tasks and AgentInstances use optimistic versions. A Redis `SET NX PX` lease prevents
two processes from executing one Task. Active runtimes renew leases automatically.
Failures release the lease and return the Task to `READY` after exponential backoff
until `max_attempts` is exhausted; terminal failure enters the tenant-scoped DLQ and
marks the Workflow failed. Deadline-expired Tasks enter the same DLQ. Requeue requires
an explicit tenant-authorized operator action. Checkpoints are monotonic per
Agent and enter the next RuntimeContext after process recovery.

Before polling, each distributed Worker reconciles durable `leased`/`running` Tasks
against Redis. A Task without a live lease returns to `ready`, and its old AgentInstance
is unblocked. A Worker that loses its lease is fenced from all terminal writes. Task
creation accepts an idempotency key and is guarded by a PostgreSQL partial unique index,
so concurrent replay of one tool call creates exactly one Task.

## Configuration

```json
{
  "harness": {
    "enabled": true,
    "backend": "postgres_redis",
    "keyPrefix": "nanobot:harness",
    "mailboxTtl": 86400,
    "idempotencyTtl": 86400,
    "messageClaimIdleMs": 60000,
    "maxQueuedTasksPerTenant": 1000,
    "workerCapacity": 4,
    "workerHeartbeatTtl": 90,
    "workerExecutionBackend": "bwrap",
    "sandboxNetwork": "host",
    "sandboxMemoryMb": 1536,
    "sandboxProcessLimit": 128,
    "sandboxOpenFilesLimit": 256,
    "sandboxFileSizeMb": 512,
    "checkpointMaxBytes": 262144,
    "retentionDays": 30
  }
}
```

`postgres_redis` reuses `memory.storage.metadataDb.dsn` and
`memory.storage.redis.url`. `auto` selects that backend when semantic memory and
PostgreSQL are enabled, otherwise the in-process store is used for development.

The corresponding environment override names are
`NANOBOT_HARNESS__ENABLED`, `NANOBOT_HARNESS__BACKEND`,
`NANOBOT_HARNESS__KEY_PREFIX`, `NANOBOT_HARNESS__MAILBOX_TTL`,
`NANOBOT_HARNESS__IDEMPOTENCY_TTL`, and
`NANOBOT_HARNESS__MESSAGE_CLAIM_IDLE_MS`. Capacity/lifecycle overrides are
`NANOBOT_HARNESS__MAX_QUEUED_TASKS_PER_TENANT`,
`NANOBOT_HARNESS__WORKER_CAPACITY`,
`NANOBOT_HARNESS__WORKER_HEARTBEAT_TTL`,
`NANOBOT_HARNESS__WORKER_EXECUTION_BACKEND`,
`NANOBOT_HARNESS__SANDBOX_NETWORK`,
`NANOBOT_HARNESS__SANDBOX_MEMORY_MB`,
`NANOBOT_HARNESS__SANDBOX_PROCESS_LIMIT`,
`NANOBOT_HARNESS__SANDBOX_OPEN_FILES_LIMIT`,
`NANOBOT_HARNESS__SANDBOX_FILE_SIZE_MB`,
`NANOBOT_HARNESS__CHECKPOINT_MAX_BYTES`, and
`NANOBOT_HARNESS__RETENTION_DAYS`. Keep PostgreSQL, Redis, Neo4j and
Qdrant credentials in Docker secrets or an untracked `.env`; never put them in
`MEMORY.md`, prompts, Task input or result messages.

## State machine

```text
pending -> ready -> leased -> succeeded
                    |  |
                    |  +-> lease lost -> recovered -> ready
                    +----> execution error -> ready at available_at
                    +----> exhausted/deadline/review rejection -> dead_letter
                    +----> cancelled
```

Dependency successors become `ready` only after every predecessor is `succeeded`.
A Workflow becomes terminal when all Tasks succeed, any Task exhausts retries, or the
Supervisor cancels it. State writes use optimistic versions; callers must reload after
a conflict rather than overwrite the newer state.

For a dynamic DAG, `sealed=false` suppresses the all-successful transition while the
Planner is expanding the graph. `seal_workflow()` marks the planned graph complete and
performs the terminal-state check.

## Storage objects

The adapter bootstraps these tables idempotently on first connection:
`harness_agent_specs`, `harness_agent_instances`, `harness_workflows`,
`harness_tasks`, `harness_checkpoints`, `harness_outbox`, and `harness_audit_events`.
The audit table is append-only during normal maintenance and records redacted inbound/
outbound bus payloads, runtime events, task lifecycle transitions, context projections,
mailbox operations and repository-research progress. Use `trace_id`, `session_key`,
`workflow_id`, and `task_id` together to reconstruct a full trace. Credential-shaped
fields are redacted before storage; do not put secrets in prompts or task payloads.

The important constraints
are unique `(tenant_id, scope_key)` for a stable Supervisor, unique
`(workflow_id, idempotency_key)` for Task replay, unique `(agent_id, sequence)` for
checkpoints, and a primary-key idempotency key for Outbox delivery.
Ordered migrations are serialized with a PostgreSQL advisory transaction lock.

Redis keys are namespaced by `harness.keyPrefix`:

```text
<prefix>:mailbox:<agent_id>          Redis Stream, consumer group agent-runtime
<prefix>:idempotency:<key>           expiring delivery dedupe key
<prefix>:lease:<task_id>             expiring, ownership-fenced WorkerLease JSON
<prefix>:resource:<sha256>           expiring exclusive resource owner
<prefix>:worker:<worker_id>          expiring WorkerRegistration JSON
<prefix>:workers:<tenant_id>         Worker registry index
```

## Operations

`GET /health` is process liveness. `GET /ready` checks Harness PostgreSQL and Redis and
returns 503 when durable delegation is unavailable. `HarnessService.diagnostics()`
reports tenant task-state counts, workflow/agent/checkpoint counts, pending Outbox count
and age, plus metric snapshots. The readiness payload also exposes logical VFS status
and process-local counters such as quota violations and namespace cleanup. `maintenance()` expires deadlines, flushes Outbox and
prunes old terminal aggregates according to `retentionDays`. It also scans active
workflows and returns lease-orphaned Tasks to `ready`.

Run an independent worker with:

```bash
nanobot harness-worker --tenant default --capacity 4
```

For production, set `workerExecutionBackend` to `bwrap`. The durable Worker
remains the Harness control plane, while each claimed Task runs in a separate
process with PID/IPC/UTS namespaces, all Linux capabilities dropped, a task
workspace mount, bounded resource limits, and a minimal environment. The child
receives only the projected RuntimeContext and AgentSpec; it does not receive
the parent transcript, database configuration, Redis state, or `MEMORY.md`.
The active LLM credential is passed through a short-lived, mode-0600 child
config and is never mounted from the host `.env` file.

`sandboxNetwork: "host"` is intentional when the deployment uses a host VPN
proxy such as `127.0.0.1:7890`; it preserves access to that proxy while the
filesystem and process namespaces remain isolated. Use `"none"` only when the
provider is reachable through an external network sidecar or another explicit
egress design.

The Worker loads the configured provider and ToolRegistry, advertises its roles and
capabilities, projects Task metadata/dependencies/checkpoints into the isolated runtime,
enforces the AgentSpec tool/iteration/tool-call budgets, heartbeats leases, and runs
maintenance every 30 seconds. Tasks declare an execution mode: `inline` tasks are owned
by the request-local coordinator and are never claimed by the pool, while `pool` tasks
are available to independent workers. This prevents a background worker from stealing
a repository or subagent task whose caller is already executing it. Docker deployment
is available through the `harness`
profile together with the `memory` profile:

```bash
docker compose --profile memory --profile harness up -d
```

OpenTelemetry emits low-cardinality `nanobot.harness.events`,
`nanobot.harness.context.chars`, and operation-duration instruments. At minimum,
alert on terminal task failures, repeated lease loss, `task.recovered`, pending Outbox
age/count, Stream pending-entry age/count and sustained context truncation.

Recommended production checks:

1. Require `health() == {store: true, postgres: true, redis: true}` before accepting
   delegated work. Liveness of the main conversation can remain independent.
2. Run at least two Worker processes and terminate one during a long Task. The Task
   must be recovered after its lease TTL and complete once.
3. Stop Redis while sending a result. The PostgreSQL Outbox must remain pending, then
   deliver exactly once after Redis returns and `flush_outbox()` runs.
4. Replay the same Task idempotency key concurrently. Both calls must resolve to the
   same `task_id`.
5. Verify a Worker cannot read a sibling mailbox, parent transcript,
   `memory/history.jsonl`, or tools outside its AgentSpec allowlist.

Back up PostgreSQL as the recovery source. Redis persistence is useful but not required
for correctness: expired/lost leases are reconciled and undelivered messages remain in
the Outbox. Set retention and backup policy for Task results and checkpoints according
to the data classification of the deployment.

Shutdown must call `SubagentManager.close()` so local tasks are cancelled, execution
sessions close, and database clients drain cleanly.

## Memory boundary

Harness state is not long-term user memory. Redis Conversation Memory carries readable
session messages, tool executions, Task State and Agent State. `MEMORY.md` initializes
the core curated profile, while Qdrant/Neo4j/PostgreSQL provide deeper asynchronous
long-term retrieval. Harness RuntimeContext may consume bounded retrieved memory, but
child-agent results and checkpoints are never promoted to user memory merely because
they exist; the post-conversation Memory Writer still performs extraction, filtering,
confidence/version checks and best-effort storage.
