-- ============================================================================
-- GENERATED FILE — DO NOT EDIT.
-- Produced by scripts/gen-migrations.sh from the canonical schema:
--   /home/eileen/.openclaw/workspace/memory/fleet-memory-schema-kimi.sql
-- Canonical sha256: c0669e7016c9831f9eb6e08fcda16f83c7e25d5d21db756e84461e5ae5eaa729
-- Sections: §1 pragmas + §2 registry (fleet-memory.db)
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
-- §2. REGISTRY — fleet-memory.db  (apply this section to the registry file)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 2.1 embedding_providers — who is allowed to make vectors, and at what tier.
--     The gateway's health probes update `status`/`last_health_at`; the
--     indexer reads this table and REFUSES any provider not marked active
--     whose (model, dims) matches the target index header. This is where the
--     "embeddings never fall back" rule lives as data, not convention.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embedding_providers (
    provider_id       TEXT PRIMARY KEY,              -- 'ollama/nomic-embed-text'
    kind              TEXT NOT NULL
                      CHECK (kind IN ('local-ollama','local-onnx','api')),
    model             TEXT NOT NULL,
    dims              INTEGER NOT NULL CHECK (dims > 0),
    endpoint          TEXT,                          -- NULL for local-onnx
    quality_tier      INTEGER NOT NULL DEFAULT 1,    -- 1 = preferred floor
    fallback_allowed  INTEGER NOT NULL DEFAULT 0
                      CHECK (fallback_allowed IN (0,1)),  -- almost always 0
    status            TEXT NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active','degraded','retired')),
    last_health_at    TEXT,                          -- UTC ISO-8601
    notes             TEXT,
    UNIQUE (kind, model, dims)
);

-- ----------------------------------------------------------------------------
-- 2.2 index_registry — every cargo hold we have ever built, and which one is
--     currently being served. The partial unique index makes "exactly one
--     current index" a database invariant, not a symlink convention. Cutover
--     is one transaction: is_current flips here AND the symlink flips on disk;
--     a crash between the two is detected at startup (symlink disagrees with
--     this table → alarm, keep serving the DB file that passes its header).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS index_registry (
    index_name     TEXT PRIMARY KEY,        -- 'index.ollama.nomic-embed-text.768'
    provider_id    TEXT NOT NULL
                   REFERENCES embedding_providers(provider_id),
    db_path        TEXT NOT NULL,           -- ext4 path; CHECK below bars 9P
    index_version  INTEGER NOT NULL,        -- schema+chunker version; bump = rebuild
    is_current     INTEGER NOT NULL DEFAULT 0 CHECK (is_current IN (0,1)),
    doc_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    retired_at     TEXT,
    CHECK (db_path NOT LIKE '/mnt/%')
);
-- Exactly one serving index, fleet-wide:
CREATE UNIQUE INDEX IF NOT EXISTS one_current_index
    ON index_registry(is_current) WHERE is_current = 1;
CREATE INDEX IF NOT EXISTS idx_registry_provider ON index_registry(provider_id);

-- ----------------------------------------------------------------------------
-- 2.3 reindex_runs + reindex_checkpoints — crash recovery as data.
--     A run FREEZES its input set at start (snapshot_manifest = a file
--     listing path+mtime+size, written before the first batch; fixes
--     "index changed while building"). The checkpoint row is updated in the
--     SAME transaction as each batch insert, so a kill -9 at any point
--     leaves a resumable cursor, never a half-batch.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reindex_runs (
    run_id             TEXT PRIMARY KEY,          -- ulid
    index_name         TEXT NOT NULL
                       REFERENCES index_registry(index_name),
    trigger_kind       TEXT NOT NULL
                       CHECK (trigger_kind IN ('manual','provider-change',
                                               'scheduled','file-watch')),
    snapshot_manifest  TEXT NOT NULL,             -- ext4 path to frozen input list
    snapshot_hash      TEXT NOT NULL,             -- sha256 of that manifest
    status             TEXT NOT NULL DEFAULT 'running'
                       CHECK (status IN ('running','completed','failed',
                                         'superseded')),
    started_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    finished_at        TEXT,
    error              TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_index ON reindex_runs(index_name, started_at);

CREATE TABLE IF NOT EXISTS reindex_checkpoints (
    run_id           TEXT PRIMARY KEY             -- 1:1 with the run
                     REFERENCES reindex_runs(run_id) ON DELETE CASCADE,
    last_doc_path    TEXT NOT NULL DEFAULT '',    -- resume cursor (path order)
    docs_total       INTEGER NOT NULL,            -- from the snapshot manifest
    docs_done        INTEGER NOT NULL DEFAULT 0,
    chunks_written   INTEGER NOT NULL DEFAULT 0,
    batches_done     INTEGER NOT NULL DEFAULT 0,
    peak_rss_bytes   INTEGER,                     -- filled at finish; proves O(batch)
    updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- ----------------------------------------------------------------------------
-- 2.4 creative_works — the registry of everything the fleet makes.
--     One row per WORK (the idea); renders of it live in work_renders.
--     "PFD speech" is one row here; its outline, three text drafts, the TTS
--     take, and the TapScript score are five rows downstream.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS creative_works (
    work_id        TEXT PRIMARY KEY,              -- ulid
    slug           TEXT NOT NULL UNIQUE,          -- 'pfd-speech'
    title          TEXT NOT NULL,
    kind           TEXT NOT NULL
                   CHECK (kind IN ('essay','poem','story','speech','radio',
                                   'letter','script','song','lore','other')),
    status         TEXT NOT NULL DEFAULT 'outline'
                   CHECK (status IN ('outline','draft','rendered','voiced',
                                     'published','archived')),
    created_by     TEXT NOT NULL,                 -- agent callsign
    synopsis       TEXT,
    source_session TEXT,                          -- which conversation produced it
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_works_created ON creative_works(created_at);
CREATE INDEX IF NOT EXISTS idx_works_kind    ON creative_works(kind, status);

-- Subjects are FIRST-CLASS, weighted, and lowercase-collated: this is the
-- fast lane for "find pieces about silence" — no vector round-trip needed
-- when a human already tagged the theme.
CREATE TABLE IF NOT EXISTS work_subjects (
    work_id  TEXT NOT NULL
             REFERENCES creative_works(work_id) ON DELETE CASCADE,
    subject  TEXT NOT NULL COLLATE NOCASE,
    weight   REAL NOT NULL DEFAULT 1.0            -- 1.0 central … 0.3 passing
             CHECK (weight BETWEEN 0.0 AND 1.0),
    PRIMARY KEY (work_id, subject)
);
CREATE INDEX IF NOT EXISTS idx_subjects_subject ON work_subjects(subject);

-- ----------------------------------------------------------------------------
-- 2.5 work_renders — every materialization of a work.
--     F6 media policy is enforced by the CHECK: anything large lives behind
--     an r2:// key; git only ever sees this manifest row. `spec_json` is the
--     reproducibility contract — a render with no recorded spec is treated
--     as unreproducible and flagged by the nightly audit.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS work_renders (
    render_id     TEXT PRIMARY KEY,               -- ulid
    work_id       TEXT NOT NULL
                  REFERENCES creative_works(work_id) ON DELETE CASCADE,
    render_kind   TEXT NOT NULL
                  CHECK (render_kind IN ('outline','text','tapscript',
                                         'tts-audio','music','image',
                                         'video','pdf')),
    seq           INTEGER NOT NULL DEFAULT 1,     -- v1, v2, … per kind
    location_kind TEXT NOT NULL CHECK (location_kind IN ('ext4','r2')),
    location      TEXT NOT NULL,                  -- ext4 path or r2://key
    mime          TEXT,
    sha256        TEXT,
    size_bytes    INTEGER CHECK (size_bytes >= 0),
    duration_ms   INTEGER CHECK (duration_ms >= 0),   -- audio/video only
    renderer      TEXT,                           -- 'sag', 'fleet-audio 0.3.0', …
    spec_json     TEXT,                           -- the spec that produced it
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (work_id, render_kind, seq),
    CHECK (location_kind <> 'r2' OR location LIKE 'r2://%'),
    CHECK (location_kind <> 'ext4' OR location NOT LIKE '/mnt/%'),
    CHECK (render_kind NOT IN ('tts-audio','music','image','video')
           OR location_kind = 'r2' OR size_bytes <= 1048576)  -- F6 fence
);
CREATE INDEX IF NOT EXISTS idx_renders_work ON work_renders(work_id, render_kind);
CREATE INDEX IF NOT EXISTS idx_renders_created ON work_renders(created_at);

-- Full-text over text-class renders. Populated by a trigger-free indexer pass
-- (renders arrive via the fleet's own writers; a small sync job keeps this
-- current — deliberately NOT a trigger, so bulk backfill stays O(batch)).
CREATE VIRTUAL TABLE IF NOT EXISTS work_text_fts USING fts5(
    title,
    body,
    work_id   UNINDEXED,
    render_id UNINDEXED,
    tokenize = 'porter unicode61'
);

-- ----------------------------------------------------------------------------
-- 2.6 agent_decisions — what was decided, by whom, when, and whether it
--     still stands. Append-only by convention: corrections are new rows
--     with supersedes set, never UPDATEs of the record itself (status is
--     the only mutable column, and only toward 'superseded'/'reverted').
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id    TEXT PRIMARY KEY,              -- ulid
    decided_at     TEXT NOT NULL,                 -- UTC ISO-8601, explicit
    agent          TEXT NOT NULL,                 -- 'navigation/kimi', 'ops/opus', …
    domain         TEXT NOT NULL
                   CHECK (domain IN ('infra','creative','comms',
                                     'fleet-policy','memory','media')),
    summary        TEXT NOT NULL,                 -- one line, loggable
    rationale      TEXT,
    reversibility  TEXT CHECK (reversibility IN ('trivial','moderate','hard')),
    status         TEXT NOT NULL DEFAULT 'active'
                   CHECK (status IN ('proposed','active','superseded','reverted')),
    supersedes     TEXT REFERENCES agent_decisions(decision_id),
    session_ref    TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_time  ON agent_decisions(decided_at);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON agent_decisions(agent, decided_at);
CREATE INDEX IF NOT EXISTS idx_decisions_domain ON agent_decisions(domain, decided_at);

CREATE TABLE IF NOT EXISTS decision_links (
    decision_id TEXT NOT NULL
                REFERENCES agent_decisions(decision_id) ON DELETE CASCADE,
    link_kind   TEXT NOT NULL
                CHECK (link_kind IN ('file','work','render','run','index','url')),
    target      TEXT NOT NULL,                    -- path, work_id, run_id, …
    PRIMARY KEY (decision_id, link_kind, target)
);


