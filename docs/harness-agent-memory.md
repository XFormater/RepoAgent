# Harness Agent Memory System

This document describes nanobot's production-oriented short-term and semantic
long-term memory design. The file-based Dream memory remains available; this
subsystem adds Redis, PostgreSQL, Qdrant, and Neo4j for low-latency session state
and source-traceable semantic recall.

## Architecture

```text
                              ┌───────────────────────────────┐
User message ────────────────▶│ Agent turn / tool execution   │
                              └──────┬──────────────┬─────────┘
                                     │              │
                         active turn │              │ turn completed
                                     ▼              ▼
                           ┌───────────────┐  ┌────────────────────┐
                           │ Redis STM     │  │ Async Memory Writer │
                           │ conversation  │  └─────────┬──────────┘
                           │ task/reasoning│            │
                           │ tool/run state│      LLM extraction
                           └───────────────┘            │
                                                       ▼
                                               filter + normalize
                                                       │
                               ┌───────────────────────┼──────────────────────┐
                               ▼                       ▼                      ▼
                         PostgreSQL                 Qdrant                 Neo4j
                         source of truth            embeddings             graph
                         version/status             semantic recall        multi-hop
                               │                       │                      │
                               └───────────────┬───────┴──────────────┬───────┘
                                               ▼                      ▼
                                         Memory Fusion          Redis warm cache
                                               │
                                               ▼
                                      bounded prompt injection
```

PostgreSQL is authoritative for lifecycle state and versions. Qdrant and Neo4j
are rebuildable projections. Redis is ephemeral.

## Redis short-term memory

Default prefix: `nanobot:stm`. IDs must be opaque, URL-safe identifiers; never
put raw user text or secrets in a key.

| Key | Redis type | Value | Default TTL |
|---|---|---|---:|
| `nanobot:stm:session:{sid}:messages` | List | `ConversationMessage` JSON | 24 h |
| `nanobot:stm:session:{sid}:task:{tid}` | String | `TaskState` JSON | 24 h |
| `nanobot:stm:session:{sid}:reasoning` | String | `ReasoningState` JSON | 6 h |
| `nanobot:stm:session:{sid}:tool:{eid}` | String | `ToolExecutionState` JSON | 24 h |
| `nanobot:stm:session:{sid}:tool-dedup:{sha256}` | String | execution ID | 24 h |
| `nanobot:stm:session:{sid}:run` | String | agent run-state JSON | 1 h |
| `nanobot:memory:warm:{uid}` | String | hot `MemoryRecord[]` JSON | 1 h |
| `nanobot:memory:idempotency:{sha256}` | String | processing/completed | 24 h |

Conversation lists are trimmed to `maxMessages` and their TTL is refreshed on
append. Reasoning state has a shorter TTL because it contains temporary
decisions. Tool dedup hashes canonical JSON arguments and points to the stored
execution, preventing repeated calls within the TTL window.

The validated JSON models live in
`nanobot/memory_system/models.py`; the asynchronous interface is
`RedisShortTermMemory` in
`nanobot/memory_system/short_term/redis_memory.py`.

```python
await memory.append_message(session_id, message)
messages = await memory.get_messages(session_id, limit=20)
task = await memory.put_task(session_id, task, expected_version=3)
await memory.put_reasoning(session_id, reasoning)
duplicate = await memory.find_tool_execution(session_id, "web_search", args)
```

`TaskState.version` provides optimistic concurrency within one session. Updates
use a Redis Lua compare-and-set, so concurrent writers cannot both commit the
same expected version.

## Extracted memory object

The LLM extractor emits only JSON validated by Pydantic:

```json
{
  "fact_text": "The user researches ultrasound generation.",
  "fact_type": "fact",
  "entities": ["ultrasound generation"],
  "relations": [
    {
      "source": "ultrasound generation",
      "target": "stable diffusion",
      "relation_type": "USES"
    }
  ],
  "confidence": 0.94,
  "source_message_ids": ["message-123"]
}
```

Allowed fact types are `preference`, `fact`, `goal`, and `hypothesis`.
Hypotheses enter `pending_review`. The extractor rejects facts without at least
one valid user-authored source ID and removes greetings, small talk, transient
questions, one-time requests, assistant claims, secrets, and low-information
content.

## PostgreSQL metadata

The canonical migration is
[`sql/agent-memory.sql`](./sql/agent-memory.sql). Important normalized columns
are:

- `id`, `user_id`, `fact_type`, `content`;
- `source_message_id`, `confidence`, `status`;
- `version`, `batch_id`, `idempotency_key`, `content_hash`;
- `created_at`, `updated_at`;
- full validated record in `payload JSONB`.

Indexes support active user recall, type/profile loading, source audit, entity
overlap, and `(user_id, batch_id, content_hash)` idempotency. Updates use:

```sql
UPDATE agent_memory_records
   SET payload=$1, version=version+1, updated_at=now()
 WHERE id=$2 AND version=$3
RETURNING version;
```

No returned row means a stale writer and raises `OptimisticLockError`.

## Qdrant store

Collection name defaults to `agent_memory`; use a versioned name such as
`agent_memory_v1` when changing embeddings.

- vector size: `storage.embedding.dimension`;
- distance: cosine;
- default embedding: `text-embedding-3-large`, 3072 dimensions;
- HNSW: `m=16`, `ef_construct=128`, query `hnsw_ef=128`;
- payload indexes: `user_id`, `id`, `fact_type`, `status`, `batch_id`.

Point ID equals `memory_id`. Payload contains the complete `MemoryRecord`
except the vector. Queries always partition by `user_id` and active status,
then apply the configured similarity threshold. PostgreSQL subsequently
revalidates confidence, status, and exact version, so stale vector projections
cannot enter the prompt.

Use a new collection and re-embed all active PostgreSQL records when changing
the embedding model or dimension.

Keep Qdrant server and `qdrant-client` within Qdrant's supported compatibility
window. The bundled deployment uses server 1.15.4 and constrains the client to
`>=1.15,<1.17`; a much older server may create collections but reject or misread
modern gRPC and universal-query requests.

## Neo4j knowledge graph

Supported node labels:

- `User`
- `Project`
- `Technology`
- `Preference`
- `Domain`

Compatibility labels `Person`, `Concept`, `Goal`, and `Event` are also allowed.
Every label has a unique `id` constraint initialized lazily.

Supported relationships include:

- `INTERESTED_IN`
- `WORKS_ON`
- `USES`
- `PREFERS`
- `RELATES_TO`

Every edge stores `id`, `weight`, and `source_memory_id`. Writes use `MERGE`:

```cypher
MERGE (u:User {id: $user_id})
MERGE (p:Project {id: $project_id})
MERGE (u)-[r:WORKS_ON {id: $relation_id}]->(p)
SET r.weight=$weight, r.source_memory_id=$memory_id
```

Multi-hop retrieval is bounded to depth 1–4:

```cypher
MATCH p=(start)-[rels*1..3]-(neighbor)
WHERE start.id IN $entity_ids
RETURN nodes(p), relationships(p)
LIMIT 100
```

The relation `source_memory_id` is resolved back through PostgreSQL before
injection. This supports `User → Project → Technology → Domain/Method` while
preserving provenance.

## Memory Writer

At the end of an agent turn:

1. derive stable source message IDs;
2. derive a deterministic extraction batch ID from session and turn ID;
3. asynchronously call `LongTermMemoryService.record_turn`;
4. extract and validate durable facts once;
5. filter by minimum confidence;
6. generate stable UUIDv7 memory IDs and embeddings;
7. claim `sha256(sorted(source_ids) + extraction_batch_id)` in Redis or
   PostgreSQL;
8. write PostgreSQL metadata as the source of truth;
9. independently retry Qdrant and Neo4j projections;
10. complete the idempotency key;
11. invalidate the Redis user-profile cache.

## Layered context and core-memory projection

`ContextOrchestrator` is the single runtime facade for session preparation,
history consolidation, dynamic block resolution, prompt construction, and the
model-facing governance copy. It emits OpenTelemetry histograms/counters under
`nanobot.context.*` and exposes an in-process diagnostic snapshot.

```text
Agent startup
  → MEMORY.md core memory → Runtime Context
  → Redis conversation/task/tool/reasoning/agent state → Runtime Context
  → Qdrant + Neo4j recall, PostgreSQL authority filter → Runtime Context
  → model
  → completed turn → asynchronous database writer
  → weekly memory-file-sync → managed MEMORY.md section
```

Dream can read `MEMORY.md` but cannot edit it. The scheduled projection owns
only the section between `BEGIN/END DATABASE MEMORY PROJECTION`; manually
curated content outside the markers is preserved. The default schedule is
Sunday 03:00 in the configured agent timezone.

`best_effort` mode runs through the AgentLoop background-task supervisor, so
extraction or storage failure never fails the user conversation. PostgreSQL
failures retry the stable batch. Embedding, Qdrant, and Neo4j failures are
logged in `RecordOutcome.failed_backends`; successful PostgreSQL records remain
available for repair/reprojection.

For restart-safe delivery at high volume, replace the in-process background
task with a PostgreSQL transactional outbox consumed by Celery, Dramatiq, or
RQ. Preserve the same batch ID and idempotency key across retries.

## Retrieval and ranking

For a query, the retriever:

1. joins the last configured turns with the current query;
2. starts Redis warm-profile loading;
3. embeds the query and searches Qdrant by `user_id`, active status, threshold,
   and `top_k`;
4. expands recalled entities through Neo4j with a strict timeout;
5. loads graph `source_memory_id` records from PostgreSQL;
6. deduplicates by memory ID;
7. asks PostgreSQL to retain only active records with current version and
   sufficient confidence;
8. cross-encoder reranks the fused candidates;
9. falls back to vector/fusion order on reranker timeout;
10. renders a bounded, XML-escaped `<memory>` block.

The effective ordering is therefore:

```text
semantic threshold
→ graph/profile union
→ authoritative lifecycle filtering
→ cross-encoder relevance
→ top_n and maxInjectedChars
```

The injection explicitly marks memories as recalled user data rather than
instructions and tells the model to prefer the current user message on
conflict.

## Failure and repair policy

- LLM extraction failure: log and return no memories.
- PostgreSQL failure: retry the stable batch; do not mark idempotency complete.
- embedding/Qdrant failure: keep metadata, report degraded projection.
- Neo4j failure: keep metadata/vector, report degraded graph.
- Redis failure: conversation continues; warm cache becomes a miss.
- graph/reranker timeout: return vector-backed results with degradation reason.

A repair worker can scan active PostgreSQL records and idempotently rebuild
Qdrant points and Neo4j nodes/edges.

## Docker environment variables

Nested settings use `NANOBOT_MEMORY__` and `__` delimiters:

```bash
NANOBOT_MEMORY__ENABLED=true
NANOBOT_MEMORY__SHORT_TERM__ENABLED=true
NANOBOT_MEMORY__STORAGE__REDIS__URL=redis://redis:6379/0
NANOBOT_MEMORY__STORAGE__METADATA_DB__DSN=postgresql://user:pass@postgres:5432/nanobot
NANOBOT_MEMORY__STORAGE__VECTOR_DB__HOST=qdrant
NANOBOT_MEMORY__STORAGE__VECTOR_DB__REST_PORT=6333
NANOBOT_MEMORY__STORAGE__VECTOR_DB__PORT=6334
NANOBOT_MEMORY__STORAGE__GRAPH_DB__URI=bolt://neo4j:7687
NANOBOT_MEMORY__STORAGE__GRAPH_DB__USERNAME=neo4j
NANOBOT_MEMORY__STORAGE__GRAPH_DB__PASSWORD=change-me
NANOBOT_MEMORY__STORAGE__EMBEDDING__API_KEY=...
```

Do not commit credentials to YAML.

## Extension roadmap

1. PostgreSQL transactional outbox and dead-letter queue.
2. Projection repair/reconciliation worker with lag metrics.
3. Redis Lua CAS for highly concurrent task state.
4. Memory deletion/privacy API that removes all projections.
5. Temporal facts with `valid_from`/`valid_to`.
6. Learned fusion weights and retrieval evaluation datasets.
7. Collection aliases for zero-downtime embedding migrations.
8. Graph entity normalization and ontology-specific extractors.
9. Human review UI for hypotheses and conflicting preferences.
10. OpenTelemetry dashboards for extraction yield, recall latency, projection
    lag, cache hit rate, and degraded writes.
