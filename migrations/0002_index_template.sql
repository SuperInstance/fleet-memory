-- ============================================================================
-- GENERATED FILE — DO NOT EDIT.
-- Produced by scripts/gen-migrations.sh from the canonical schema:
--   /home/eileen/.openclaw/workspace/memory/fleet-memory-schema-kimi.sql
-- Canonical sha256: c0669e7016c9831f9eb6e08fcda16f83c7e25d5d21db756e84461e5ae5eaa729
-- Sections: §1 pragmas + §3 index template (@DIMS@ substituted at build time)
-- Any manual change here will be overwritten on regeneration.
-- ============================================================================

-- ============================================================================
-- §1. PRAGMAS — apply to BOTH databases
-- ============================================================================
PRAGMA journal_mode = WAL;        -- crash-safe, readers never block the writer
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;       -- wait out a checkpoint instead of erroring
PRAGMA synchronous = NORMAL;      -- WAL-safe; FULL buys nothing here


-- ============================================================================
-- §3. INDEX TEMPLATE — index.<provider>.<model>.<dims>.db
--     Applied by memory-indexer to each NEW index file at build time.
--     The filename carries provenance; index_meta carries it again inside,
--     because a file can be renamed by accident and a header cannot.
--     On open, the query layer MUST verify header == filename == the
--     registry row. Any mismatch is a hard error, never a silent query.
-- ============================================================================

-- 3.1 index_meta — single-row checked header (id = 1 enforced).
CREATE TABLE IF NOT EXISTS index_meta (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    provider_id     TEXT NOT NULL,
    model           TEXT NOT NULL,
    dims            INTEGER NOT NULL CHECK (dims > 0),
    index_version   INTEGER NOT NULL,             -- matches registry row
    chunker_version TEXT NOT NULL,                -- chunking is part of identity
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- 3.2 documents — the source files, in snapshot order.
CREATE TABLE IF NOT EXISTS documents (
    doc_id      INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,             -- relative to corpus root
    sha256      TEXT NOT NULL,
    mtime_ns    INTEGER NOT NULL,
    size_bytes  INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active','stale','gone')),
    indexed_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- 3.3 chunks — text with stable identity. rowid == chunk_id is ALSO the
--     rowid in vec_chunks: that 1:1 rowid alignment is the join, so chunks
--     and their vectors are inserted in one transaction, always.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      INTEGER PRIMARY KEY,
    doc_id        INTEGER NOT NULL
                  REFERENCES documents(doc_id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,               -- order within the document
    start_offset  INTEGER NOT NULL,               -- char offsets into source
    end_offset    INTEGER NOT NULL,
    text          TEXT NOT NULL,
    content_hash  TEXT NOT NULL,                  -- skip unchanged chunks on reindex
    token_count   INTEGER,
    embedded_at   TEXT,
    UNIQUE (doc_id, seq),
    CHECK (end_offset > start_offset)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- 3.4 vec_chunks — sqlite-vec KNN store. @DIMS@ is substituted at build time
--     and MUST equal index_meta.dims; the indexer asserts this before the
--     first insert. A 1024-dim vector physically cannot land in a 768-dim
--     hold — the extension rejects it, which is the point.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    embedding float[@DIMS@]
);


