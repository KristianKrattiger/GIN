CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    source_uri TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT 'synthetic',
    outlet TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    head_sentence TEXT NOT NULL DEFAULT '',
    eval_layer TEXT NOT NULL DEFAULT 'realism',
    eval_tag TEXT,
    content_hash TEXT NOT NULL,
    embedding vector(384),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_eval_layer ON chunks(eval_layer);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN(tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS edges (
    edge_id UUID PRIMARY KEY,
    src_chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    dst_chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    note TEXT,
    proposer TEXT,
    confidence REAL,
    content_hash TEXT,
    src_anchor INT[],
    dst_anchor INT[],
    admitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (src_chunk_id, dst_chunk_id, edge_type)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id UUID PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    stats_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS peer_anchors (
    peer_node_id  TEXT NOT NULL,
    chunk_id      TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    outlet        TEXT NOT NULL,
    title         TEXT NOT NULL,
    bucket_index  SMALLINT NOT NULL,
    synced_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (peer_node_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_peer_anchors_bucket
    ON peer_anchors(peer_node_id, bucket_index);
