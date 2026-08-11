CREATE TABLE IF NOT EXISTS agent_memory_records (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    fact_type TEXT NOT NULL,
    content TEXT NOT NULL,
    source_message_id TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    status TEXT NOT NULL CHECK (
        status IN ('active', 'superseded', 'pending_review')
    ),
    entities TEXT[] NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    batch_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_memory_user_type
    ON agent_memory_records(user_id, fact_type, status);
CREATE INDEX IF NOT EXISTS idx_agent_memory_user_active_confidence
    ON agent_memory_records(user_id, status, confidence DESC, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_memory_source_message
    ON agent_memory_records(user_id, source_message_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_entities
    ON agent_memory_records USING GIN(entities);
CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_memory_batch_content
    ON agent_memory_records(user_id, batch_id, content_hash);

CREATE TABLE IF NOT EXISTS agent_memory_idempotency (
    key TEXT PRIMARY KEY,
    state TEXT NOT NULL CHECK (state IN ('processing', 'completed')),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Optimistic update example:
-- UPDATE agent_memory_records
--    SET payload=$1, content=$2, confidence=$3, version=version+1,
--        updated_at=now()
--  WHERE id=$4 AND version=$5
-- RETURNING version;
