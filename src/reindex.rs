//! The streaming reindex pipeline — Phase 4 data flow, end to end.
//!
//! ```text
//! snapshot manifest (frozen at start)
//!        → reader thread  (verifies each file against the manifest,
//!                           chunks it with O(chunk) memory)
//!        → bounded channel (cap 64 — backpressure, never buffering the corpus)
//!        → embedder       (batches of 32, current provider ONLY)
//!        → writer         (ONE transaction per batch: documents + chunks +
//!                           vec rows + reindex_checkpoints advance together)
//! ```
//!
//! Concurrency is guarded by `flock(2)` on a guard file (see `lock.rs`) —
//! no PID files, no staleness heuristics; the kernel is the staleness
//! detector. Crash recovery is data: the checkpoint row lives in the same
//! transaction as the batch it describes, so a kill -9 at any point leaves a
//! resumable cursor and never a half-batch. Zero duplicate chunks are
//! guaranteed by `UNIQUE(doc_id, seq)` plus per-doc cleanup on first touch.
//!
//! Two databases (per the canonical schema):
//!   * `fleet-memory.db` — the registry (permanent): providers, index
//!     registry, reindex runs/checkpoints, creative works, decisions.
//!   * `index.<provider>.<model>.<dims>.db` — the cargo hold (disposable):
//!     index_meta + documents + chunks + vec_chunks, built from
//!     `migrations/0002_index_template.sql` with @DIMS@ substituted.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::chunker::{ChunkSpec, HashingReader, LineChunker, CHUNKER_VERSION};
use crate::index::{IndexIdentity, IndexManager};
use crate::lock::IndexLock;
use crate::snapshot::{self, SnapshotEntry};

/// Migration SQL, generated from the canonical schema. Single source of
/// truth: memory/fleet-memory-schema-kimi.sql → scripts/gen-migrations.sh.
const REGISTRY_MIGRATION: &str = include_str!("../migrations/0001_registry.sql");
const INDEX_TEMPLATE_MIGRATION: &str = include_str!("../migrations/0002_index_template.sql");

/// Schema+chunker version recorded in index_registry/index_meta. A bump
/// means "build a new index", never an in-place migration.
pub const INDEX_VERSION: i64 = 1;

/// Channel capacity between reader and embedder/writer (Phase 4 spec).
pub const CHANNEL_CAP: usize = 64;
/// Default embedding batch size (Phase 4 spec).
pub const BATCH_SIZE: usize = 32;

#[derive(Error, Debug)]
pub enum ReindexError {
    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("snapshot error: {0}")]
    Snapshot(#[from] snapshot::SnapshotError),
    #[error("lock error: {0}")]
    Lock(#[from] crate::lock::LockError),
    #[error("index manager error: {0}")]
    Index(#[from] crate::index::IndexError),
    #[error("embedding failed: {0}")]
    Embed(String),
    #[error("provider {0} is not active — refusing to index with it (embeddings never fall back)")]
    ProviderNotActive(String),
    #[error("provenance mismatch: index {index} holds {held}, refusing {offered}")]
    Provenance {
        index: String,
        held: String,
        offered: String,
    },
    #[error("index {0} was built with schema version {1}; this indexer speaks version {2} — build a new index")]
    VersionBump(String, i64, i64),
    #[error("chunker changed: index {0} was chunked with {1}, this indexer chunks with {2} — build a new index")]
    ChunkerChanged(String, String, String),
    #[error("embedding dimension mismatch: index holds {index_dims}, provider returned {got_dims}")]
    EmbedDims { index_dims: usize, got_dims: usize },
}

/// The async embedding function the pipeline calls once per batch.
/// Real runs wrap `EmbeddingClient::embed_batch`; tests return fakes.
pub type EmbedFuture = Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>, String>> + Send>>;
pub type EmbedFn<'a> = dyn Fn(Vec<String>) -> EmbedFuture + Send + Sync + 'a;

// ---------------------------------------------------------------------------
// Registry database (fleet-memory.db)
// ---------------------------------------------------------------------------

/// Handle on the registry database. Applies migrations/0001 on open.
pub struct Registry {
    conn: Connection,
    path: PathBuf,
}

impl Registry {
    /// Open (creating if needed) the registry at `path`.
    pub fn open(path: &Path) -> Result<Self, ReindexError> {
        crate::db::ensure_vec_registered();
        let conn = Connection::open(path)?;
        conn.execute_batch(REGISTRY_MIGRATION)?;
        Ok(Self {
            conn,
            path: path.to_path_buf(),
        })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Ensure the provider row exists and is usable. Refuses retired or
    /// (model, dims)-mismatched providers — the "no fallback" rule as data.
    pub fn ensure_provider(&self, identity: &IndexIdentity) -> Result<String, ReindexError> {
        let provider_id = format!("{}/{}", identity.provider, identity.model);
        let existing: Option<(String, i64, usize)> = self
            .conn
            .query_row(
                "SELECT status, dims, dims FROM embedding_providers WHERE provider_id = ?1",
                params![provider_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;

        if let Some((status, dims, _)) = existing {
            if dims as usize != identity.dims {
                return Err(ReindexError::Provenance {
                    index: provider_id.clone(),
                    held: format!("{}d", dims),
                    offered: format!("{}d", identity.dims),
                });
            }
            if status == "retired" {
                return Err(ReindexError::ProviderNotActive(provider_id));
            }
            return Ok(provider_id);
        }

        self.conn.execute(
            "INSERT INTO embedding_providers (provider_id, kind, model, dims, status)
             VALUES (?1, 'local-ollama', ?2, ?3, 'active')",
            params![provider_id, identity.model, identity.dims as i64],
        )?;
        Ok(provider_id)
    }

    /// Find the latest still-running (crashed) run for an index, for resume.
    pub fn load_resume(&self, index_name: &str) -> Result<Option<(String, String, i64, i64, i64)>, ReindexError> {
        let row = self
            .conn
            .query_row(
                "SELECT r.run_id, c.last_doc_path, c.docs_done, c.chunks_written, c.batches_done
                 FROM reindex_runs r
                 JOIN reindex_checkpoints c ON c.run_id = r.run_id
                 WHERE r.index_name = ?1 AND r.status IN ('running', 'failed')
                 ORDER BY r.started_at DESC LIMIT 1",
                params![index_name],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, i64>(4)?,
                    ))
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Mark any 'running' runs for this index as superseded (new run started).
    pub fn supersede_running(&self, index_name: &str) -> Result<(), ReindexError> {
        self.conn.execute(
            "UPDATE reindex_runs SET status = 'superseded',
                finished_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE index_name = ?1 AND status = 'running'",
            params![index_name],
        )?;
        Ok(())
    }

    /// Insert a fresh run + checkpoint row. Returns the run_id.
    pub fn begin_run(
        &self,
        index_name: &str,
        manifest_path: &str,
        manifest_hash: &str,
        docs_total: i64,
        trigger: &str,
    ) -> Result<String, ReindexError> {
        let run_id = new_run_id();
        self.conn.execute(
            "INSERT INTO reindex_runs (run_id, index_name, trigger_kind, snapshot_manifest, snapshot_hash, status)
             VALUES (?1, ?2, ?3, ?4, ?5, 'running')",
            params![run_id, index_name, trigger, manifest_path, manifest_hash],
        )?;
        self.conn.execute(
            "INSERT INTO reindex_checkpoints (run_id, last_doc_path, docs_total)
             VALUES (?1, '', ?2)",
            params![run_id, docs_total],
        )?;
        Ok(run_id)
    }

    pub fn fail_run(&self, run_id: &str, error: &str) {
        let _ = self.conn.execute(
            "UPDATE reindex_runs SET status = 'failed',
                finished_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'), error = ?2
             WHERE run_id = ?1",
            params![run_id, error],
        );
    }

    pub fn complete_run(&self, run_id: &str, peak_rss_bytes: Option<i64>) -> Result<(), ReindexError> {
        self.conn.execute(
            "UPDATE reindex_runs SET status = 'completed',
                finished_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE run_id = ?1",
            params![run_id],
        )?;
        if let Some(rss) = peak_rss_bytes {
            self.conn.execute(
                "UPDATE reindex_checkpoints SET peak_rss_bytes = ?2 WHERE run_id = ?1",
                params![run_id, rss],
            )?;
        }
        Ok(())
    }

    /// Flip is_current to this index in ONE transaction (the partial unique
    /// index `one_current_index` makes "exactly one current" an invariant).
    pub fn set_current(
        &mut self,
        index_name: &str,
        provider_id: &str,
        db_path: &str,
        doc_count: i64,
        chunk_count: i64,
    ) -> Result<(), ReindexError> {
        let tx = self.conn.transaction()?;
        tx.execute("UPDATE index_registry SET is_current = 0 WHERE is_current = 1", [])?;
        let n = tx.execute(
            "UPDATE index_registry
             SET is_current = 1, doc_count = ?3, chunk_count = ?4, retired_at = NULL
             WHERE index_name = ?1",
            params![index_name, provider_id, doc_count, chunk_count],
        )?;
        if n == 0 {
            tx.execute(
                "INSERT INTO index_registry
                 (index_name, provider_id, db_path, index_version, is_current, doc_count, chunk_count)
                 VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6)",
                params![index_name, provider_id, db_path, INDEX_VERSION, doc_count, chunk_count],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// The registry row for the current index, if any.
    pub fn current_row(&self) -> Result<Option<(String, String)>, ReindexError> {
        let row = self
            .conn
            .query_row(
                "SELECT index_name, db_path FROM index_registry WHERE is_current = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        Ok(row)
    }

    /// Ensure a (non-current) registry row exists for an index before a run
    /// references it — reindex_runs.index_name has an FK here. Also verifies
    /// the db_path stays on ext4 via the table's CHECK.
    pub fn upsert_index_row(
        &self,
        index_name: &str,
        provider_id: &str,
        db_path: &str,
    ) -> Result<(), ReindexError> {
        self.conn.execute(
            "INSERT INTO index_registry (index_name, provider_id, db_path, index_version, is_current)
             VALUES (?1, ?2, ?3, ?4, 0)
             ON CONFLICT(index_name) DO UPDATE SET
               provider_id = excluded.provider_id,
               db_path = excluded.db_path,
               index_version = excluded.index_version",
            params![index_name, provider_id, db_path, INDEX_VERSION],
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Index database (index.<provider>.<model>.<dims>.db)
// ---------------------------------------------------------------------------

/// Handle on a cargo-hold index file. Applies migrations/0002 with @DIMS@
/// substituted, then verifies/stamps index_meta.
pub struct IndexFile {
    conn: Connection,
    path: PathBuf,
    identity: IndexIdentity,
    provider_id: String,
}

impl std::fmt::Debug for IndexFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexFile")
            .field("path", &self.path)
            .field("identity", &self.identity)
            .finish()
    }
}

impl IndexFile {
    /// Open or create the index file for `identity`. On mismatch of any
    /// identity field (provider/model/dims/version/chunker) this is a HARD
    /// error before the first insert — provenance refusal.
    pub fn open(path: &Path, identity: &IndexIdentity) -> Result<Self, ReindexError> {
        crate::db::ensure_vec_registered();
        let provider_id = format!("{}/{}", identity.provider, identity.model);

        // Create path's parent if needed.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;
        conn.execute_batch(REGISTRY_PRAGMAS)?;

        let template = INDEX_TEMPLATE_MIGRATION.replace("@DIMS@", &identity.dims.to_string());
        conn.execute_batch(&template)?;

        // Verify or stamp index_meta (the checked header).
        let held: Option<(String, String, i64, i64, String)> = conn
            .query_row(
                "SELECT provider_id, model, dims, index_version, chunker_version
                 FROM index_meta WHERE id = 1",
                [],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                    ))
                },
            )
            .optional()?;

        if let Some((h_provider, h_model, h_dims, h_version, h_chunker)) = held {
            let held_s = format!("{h_provider} {h_model} {h_dims}d v{h_version} chunker={h_chunker}");
            let off_s = format!("{provider_id} {} {}d v{INDEX_VERSION} chunker={CHUNKER_VERSION}", identity.model, identity.dims);
            if h_provider != provider_id || h_model != identity.model || h_dims as usize != identity.dims {
                return Err(ReindexError::Provenance {
                    index: path.display().to_string(),
                    held: held_s,
                    offered: off_s,
                });
            }
            if h_version != INDEX_VERSION {
                return Err(ReindexError::VersionBump(
                    path.display().to_string(),
                    h_version,
                    INDEX_VERSION,
                ));
            }
            if h_chunker != CHUNKER_VERSION {
                return Err(ReindexError::ChunkerChanged(
                    path.display().to_string(),
                    h_chunker,
                    CHUNKER_VERSION.to_string(),
                ));
            }
        } else {
            conn.execute(
                "INSERT INTO index_meta (id, provider_id, model, dims, index_version, chunker_version)
                 VALUES (1, ?1, ?2, ?3, ?4, ?5)",
                params![provider_id, identity.model, identity.dims as i64, INDEX_VERSION, CHUNKER_VERSION],
            )?;
        }

        Ok(Self {
            conn,
            path: path.to_path_buf(),
            identity: identity.clone(),
            provider_id,
        })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn identity(&self) -> &IndexIdentity {
        &self.identity
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn index_name(&self) -> String {
        // index.<provider>.<model>.<dims> (registry key, without .db)
        self.path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| self.identity.filename())
    }

    /// Upsert a documents row; returns doc_id.
    pub(crate) fn upsert_doc(
        tx: &rusqlite::Transaction,
        rel_path: &str,
        sha256: &str,
        mtime_ns: i64,
        size: u64,
    ) -> Result<i64, ReindexError> {
        tx.execute(
            "INSERT INTO documents (path, sha256, mtime_ns, size_bytes, status)
             VALUES (?1, ?2, ?3, ?4, 'active')
             ON CONFLICT(path) DO UPDATE
               SET sha256 = excluded.sha256, mtime_ns = excluded.mtime_ns,
                   size_bytes = excluded.size_bytes, status = 'active',
                   indexed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            params![rel_path, sha256, mtime_ns, size as i64],
        )?;
        let doc_id: i64 = tx.query_row(
            "SELECT doc_id FROM documents WHERE path = ?1",
            params![rel_path],
            |r| r.get(0),
        )?;
        Ok(doc_id)
    }

    /// Delete any prior chunks+vectors of a doc (first touch in this run —
    /// makes resume idempotent; UNIQUE(doc_id, seq) is the backstop).
    pub(crate) fn cleanup_doc(tx: &rusqlite::Transaction, doc_id: i64) -> Result<(), ReindexError> {
        tx.execute(
            "DELETE FROM vec_chunks WHERE rowid IN
               (SELECT chunk_id FROM chunks WHERE doc_id = ?1)",
            params![doc_id],
        )?;
        tx.execute("DELETE FROM chunks WHERE doc_id = ?1", params![doc_id])?;
        Ok(())
    }

    /// Insert one chunk + its aligned vec row. chunk_id (rowid) == vec rowid.
    pub(crate) fn insert_chunk(
        tx: &rusqlite::Transaction,
        doc_id: i64,
        spec: &ChunkSpec,
        embedding: &[f32],
    ) -> Result<(), ReindexError> {
        let content_hash = {
            use sha2::Digest;
            let mut h = sha2::Sha256::new();
            h.update(spec.text.as_bytes());
            format!("{:x}", h.finalize())
        };
        tx.execute(
            "INSERT INTO chunks (doc_id, seq, start_offset, end_offset, text, content_hash, token_count, embedded_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
            params![
                doc_id,
                spec.seq as i64,
                spec.start_offset as i64,
                spec.end_offset as i64,
                spec.text,
                content_hash,
                spec.token_count() as i64
            ],
        )?;
        let chunk_id = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?1, ?2)",
            params![chunk_id, crate::db::embedding_to_bytes(embedding)],
        )?;
        Ok(())
    }

    pub fn counts(&self) -> Result<(i64, i64), ReindexError> {
        let docs: i64 = self.conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
        let chunks: i64 = self.conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok((docs, chunks))
    }

    /// Consume the handle, returning the raw connection (tests, tooling).
    pub fn into_conn(self) -> Connection {
        self.conn
    }
}

/// §1 pragmas applied to the index file (extracted from the generated
/// migration so behavior stays identical).
const REGISTRY_PRAGMAS: &str = "PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000; PRAGMA synchronous = NORMAL;";

// ---------------------------------------------------------------------------
// Pipeline messages
// ---------------------------------------------------------------------------

/// Frozen identity of a document flowing through the pipe.
#[derive(Debug, Clone)]
struct DocInfo {
    rel_path: String,
    mtime_ns: i64,
    size: u64,
    sha256: String,
}

#[derive(Debug)]
enum PipeMsg {
    Chunk {
        doc: DocInfo,
        spec: ChunkSpec,
    },
    /// Reader finished a doc cleanly (hash verified).
    DocDone {
        rel_path: String,
    },
    /// Doc failed the cheap stat check before reading (defer to next run).
    Skipped {
        rel_path: String,
    },
    /// Doc changed while being read — chunks already sent must be dropped.
    Invalidate {
        rel_path: String,
    },
    /// Reader thread finished the manifest.
    Finished,
}

/// Outcome of a completed pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReindexOutcome {
    pub run_id: String,
    pub docs_total: usize,
    pub docs_done: usize,
    pub chunks_written: usize,
    pub batches: usize,
    /// Files whose stat no longer matched the manifest (deferred to next run).
    pub skipped_midrun: usize,
    /// Files whose content changed during the read (invalidated this run).
    pub invalidated_midrun: usize,
    pub errors: usize,
    pub peak_rss_bytes: Option<i64>,
}

/// Options for the pipeline.
pub struct PipelineOpts<'a> {
    pub root: &'a Path,
    pub index_dir: &'a Path,
    pub identity: &'a IndexIdentity,
    /// Target chunk size in characters.
    pub chunk_chars: usize,
    pub include: Option<regex::Regex>,
    pub force: bool,
    pub trigger: &'a str,
}

/// Run the streaming pipeline. Memory is O(channel + batch), never O(corpus).
pub async fn run_pipeline(
    opts: PipelineOpts<'_>,
    embed: &(impl Fn(Vec<String>) -> EmbedFuture + Send + Sync),
) -> Result<ReindexOutcome, ReindexError> {
    let manager = IndexManager::new(&opts.index_dir.to_string_lossy());
    manager.ensure_dir()?;
    let index_path = manager.index_path(opts.identity);

    // flock(2) guard — one writer per index file, kernel-released.
    let _guard = IndexLock::acquire(&index_path)?;
    tracing::debug!("pipeline: flock held on {}", index_path.display());

    let registry_path = opts.index_dir.join("fleet-memory.db");
    let mut registry = Registry::open(&registry_path)?;
    let provider_id = registry.ensure_provider(opts.identity)?;

    let index = IndexFile::open(&index_path, opts.identity)?;
    let index_name = index.index_name();

    // The run row references the registry row (FK) — make sure it exists
    // before begin_run. (Also enforces the ext4 CHECK on db_path.)
    registry.upsert_index_row(
        &index_name,
        &provider_id,
        &index_path.to_string_lossy(),
    )?;

    // ---- Freeze the input set BEFORE the first batch. ----
    let manifests_dir = opts.index_dir.join("manifests");
    let probe_run_id = new_run_id();
    let snap = snapshot::freeze(opts.root, opts.include.as_ref(), &manifests_dir, &probe_run_id)?;
    let docs_total = snap.entries.len();
    tracing::info!(
        "pipeline: frozen {} files under {} (manifest {})",
        docs_total,
        opts.root.display(),
        snap.manifest_path.display()
    );

    // ---- Run bookkeeping: fresh run, or resume a crashed one. ----
    let (run_id, cursor, mut docs_done, mut chunks_written, mut batches_done) =
        if opts.force {
            registry.supersede_running(&index_name)?;
            (
                registry.begin_run(
                    &index_name,
                    &snap.manifest_path.to_string_lossy(),
                    &snap.hash,
                    docs_total as i64,
                    opts.trigger,
                )?,
                String::new(),
                0i64,
                0i64,
                0i64,
            )
        } else if let Some((rid, cur, dd, cw, bd)) = registry.load_resume(&index_name)? {
            tracing::info!("pipeline: resuming run {rid} from cursor {cur:?}");
            (rid, cur, dd, cw, bd)
        } else {
            (
                registry.begin_run(
                    &index_name,
                    &snap.manifest_path.to_string_lossy(),
                    &snap.hash,
                    docs_total as i64,
                    opts.trigger,
                )?,
                String::new(),
                0,
                0,
                0,
            )
        };

    // The pipe. The registry is ATTACHed to the index connection so each
    // batch transaction spans chunks+vectors (index db) AND the checkpoint
    // advance (registry db) — one transaction per batch, per the schema.
    // (SQLite cannot make cross-FILE WAL commits fully atomic; the worst
    // crash window leaves the cursor behind the data, and resume is
    // idempotent by design: per-doc cleanup + UNIQUE(doc_id, seq).)
    let (tx, mut rx) = mpsc::channel::<PipeMsg>(CHANNEL_CAP);

    let reader_handle = spawn_reader(opts.root.to_path_buf(), snap.entries.clone(), cursor.clone(), opts.chunk_chars, tx);

    let mut pending: Vec<(DocInfo, ChunkSpec)> = Vec::with_capacity(BATCH_SIZE);
    let mut touched: HashSet<String> = HashSet::new();
    let mut last_done = cursor.clone();
    let mut outcome_errors = 0usize;
    let mut skipped_midrun = 0usize;
    let mut invalidated_midrun = 0usize;

    let mut conn = index.conn; // owned connection for transactions

    // The registry is ATTACHed to the index connection so each batch
    // transaction spans chunks+vectors (index db) AND the checkpoint
    // advance (registry db) — one transaction per batch, per the schema.
    // (SQLite cannot make cross-FILE WAL commits fully atomic; the worst
    // crash window leaves the cursor behind the data, and resume is
    // idempotent by design: per-doc cleanup + UNIQUE(doc_id, seq).)
    conn.execute(
        "ATTACH DATABASE ? AS reg",
        params![registry_path.to_string_lossy()],
    )?;
    let result: Result<(), ReindexError> = async {
        while let Some(msg) = rx.recv().await {
            match msg {
                PipeMsg::Chunk { doc, spec } => {
                    pending.push((doc, spec));
                    if pending.len() >= BATCH_SIZE {
                        flush_batch(
                            &mut conn,
                            &run_id,
                            &mut pending,
                            &mut touched,
                            &last_done,
                            docs_done,
                            &mut chunks_written,
                            &mut batches_done,
                            embed,
                        )
                        .await?;
                    }
                }
                PipeMsg::DocDone { rel_path } => {
                    if !touched.contains(&rel_path) {
                        // Doc with zero chunks (e.g. empty file): still record it.
                        let tx = conn.transaction()?;
                        let entry = snap.entries.iter().find(|e| e.rel_path == rel_path);
                        if let Some(e) = entry {
                            IndexFile::upsert_doc(&tx, &rel_path, &e.sha256, e.mtime_ns, e.size)?;
                        }
                        tx.commit()?;
                    }
                    docs_done += 1;
                    last_done = rel_path;
                }
                PipeMsg::Skipped { rel_path } => {
                    skipped_midrun += 1;
                    docs_done += 1;
                    tracing::warn!("pipeline: {} changed on disk mid-run — deferred to next run", rel_path);
                    last_done = rel_path;
                }
                PipeMsg::Invalidate { rel_path } => {
                    invalidated_midrun += 1;
                    docs_done += 1;
                    IndexFile::invalidate_doc_conn(&mut conn, &rel_path)?;
                    tracing::warn!("pipeline: {} modified while being read — invalidated this run", rel_path);
                    last_done = rel_path;
                }
                PipeMsg::Finished => {
                    if !pending.is_empty() {
                        flush_batch(
                            &mut conn,
                            &run_id,
                            &mut pending,
                            &mut touched,
                            &last_done,
                            docs_done,
                            &mut chunks_written,
                            &mut batches_done,
                            embed,
                        )
                        .await?;
                    }
                    break;
                }
            }
        }
        Ok(())
    }
    .await;

    // Join the reader thread regardless of pipeline outcome.
    match reader_handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            outcome_errors += 1;
            tracing::error!("pipeline: reader thread failed: {e}");
            if result.is_ok() {
                registry.fail_run(&run_id, &format!("reader failed: {e}"));
                return Err(ReindexError::Io(e));
            }
        }
        Err(_) => {
            outcome_errors += 1;
            tracing::error!("pipeline: reader thread panicked");
        }
    }

    if let Err(e) = result {
        registry.fail_run(&run_id, &e.to_string());
        return Err(e);
    }

    // ---- Completion: counts, checkpoint, cutover. ----
    let (doc_count, chunk_count) = (
        conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get::<_, i64>(0))?,
        conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get::<_, i64>(0))?,
    );
    let peak = peak_rss_bytes();
    registry.complete_run(&run_id, peak)?;
    registry.checkpoint_final(&run_id, &last_done, docs_done, chunks_written, batches_done)?;

    // Registry cutover + atomic symlink swap. Registry first; a crash in
    // between is detected at startup (symlink vs table disagreement).
    registry.set_current(
        &index_name,
        &provider_id,
        &index_path.to_string_lossy(),
        doc_count,
        chunk_count,
    )?;
    manager.swap_current(&index_path)?;

    tracing::info!(
        "pipeline: run {run_id} complete — {} docs, {} chunks, {} batches",
        doc_count,
        chunk_count,
        batches_done
    );

    Ok(ReindexOutcome {
        run_id,
        docs_total,
        docs_done: docs_done as usize,
        chunks_written: chunks_written as usize,
        batches: batches_done as usize,
        skipped_midrun,
        invalidated_midrun,
        errors: outcome_errors,
        peak_rss_bytes: peak,
    })
}

/// Reader thread: streams manifest entries (from the cursor forward),
/// verifies each against the frozen record, chunks with O(chunk) memory.
fn spawn_reader(
    root: PathBuf,
    entries: Vec<SnapshotEntry>,
    cursor: String,
    chunk_chars: usize,
    tx: mpsc::Sender<PipeMsg>,
) -> std::thread::JoinHandle<std::io::Result<()>> {
    std::thread::spawn(move || -> std::io::Result<()> {
        for entry in entries {
            // Resume: skip docs at or before the cursor (path order).
            if entry.rel_path <= cursor {
                continue;
            }

            let abs = root.join(&entry.rel_path);

            // Cheap pre-read verification against the frozen record.
            if !snapshot::stat_matches(&root, &entry) {
                if tx.blocking_send(PipeMsg::Skipped { rel_path: entry.rel_path }).is_err() {
                    return Ok(()); // writer gone — pipeline failed upstream
                }
                continue;
            }

            // Stream the file: hash raw bytes while chunking decoded text.
            let file = std::fs::File::open(&abs)?;
            let hashing = HashingReader::new(std::io::BufReader::with_capacity(64 * 1024, file));
            let mut chunker = LineChunker::new(hashing, chunk_chars);
            let mut send_err = false;
            for spec in chunker.by_ref() {
                let spec = match spec {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("reader: {} chunk error: {e}", entry.rel_path);
                        break;
                    }
                };
                let msg = PipeMsg::Chunk {
                    doc: DocInfo {
                        rel_path: entry.rel_path.clone(),
                        mtime_ns: entry.mtime_ns,
                        size: entry.size,
                        sha256: entry.sha256.clone(),
                    },
                    spec,
                };
                if tx.blocking_send(msg).is_err() {
                    send_err = true; // writer gone
                    break;
                }
            }
            if send_err {
                return Ok(());
            }

            // Strong post-read verification: content must still hash to the
            // frozen sha256, else the run must not keep what it read.
            let digest = chunker.into_inner().sha256_hex();
            if digest != entry.sha256 {
                if tx
                    .blocking_send(PipeMsg::Invalidate { rel_path: entry.rel_path })
                    .is_err()
                {
                    return Ok(());
                }
                continue;
            }

            if tx.blocking_send(PipeMsg::DocDone { rel_path: entry.rel_path }).is_err() {
                return Ok(());
            }
        }
        let _ = tx.blocking_send(PipeMsg::Finished);
        Ok(())
    })
}

/// Embed + write one batch in a single transaction, checkpoint included.
/// `conn` must have the registry ATTACHed as `reg`.
#[allow(clippy::too_many_arguments)]
async fn flush_batch(
    conn: &mut Connection,
    run_id: &str,
    pending: &mut Vec<(DocInfo, ChunkSpec)>,
    touched: &mut HashSet<String>,
    last_done: &str,
    docs_done: i64,
    chunks_written: &mut i64,
    batches_done: &mut i64,
    embed: &(impl Fn(Vec<String>) -> EmbedFuture + Send + Sync),
) -> Result<(), ReindexError> {
    let texts: Vec<String> = pending.iter().map(|(_, s)| s.text.clone()).collect();
    let embeddings = embed(texts)
        .await
        .map_err(ReindexError::Embed)?;

    let dims_hint = embeddings.first().map(|v| v.len());
    if let Some(d) = dims_hint {
        // Provenance refusal extends to runtime: wrong-dim vectors never land.
        // (vec0 also rejects them physically; this gives the better error.)
        let expected = conn
            .query_row("SELECT dims FROM index_meta WHERE id = 1", [], |r| r.get::<_, i64>(0))? as usize;
        if d != expected {
            return Err(ReindexError::EmbedDims { index_dims: expected, got_dims: d });
        }
    }

    let tx = conn.transaction()?;

    // Docs appear in order (the reader is sequential); clean up each doc's
    // prior rows on first touch, then reuse its doc_id for the batch.
    let mut last_doc: Option<(String, i64)> = None;
    for ((doc, spec), emb) in pending.iter().zip(embeddings.iter()) {
        let doc_id = match &last_doc {
            Some((rel, id)) if rel == &doc.rel_path => *id,
            _ => {
                let id = IndexFile::upsert_doc(&tx, &doc.rel_path, &doc.sha256, doc.mtime_ns, doc.size)?;
                if !touched.contains(&doc.rel_path) {
                    IndexFile::cleanup_doc(&tx, id)?;
                    touched.insert(doc.rel_path.clone());
                }
                last_doc = Some((doc.rel_path.clone(), id));
                id
            }
        };
        IndexFile::insert_chunk(&tx, doc_id, spec, emb)?;
    }

    *chunks_written += pending.len() as i64;
    *batches_done += 1;
    tx.execute(
        "UPDATE reg.reindex_checkpoints
         SET last_doc_path = ?2, docs_done = ?3, chunks_written = ?4,
             batches_done = ?5,
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
         WHERE run_id = ?1",
        params![run_id, last_done, docs_done, *chunks_written, *batches_done],
    )?;

    tx.commit()?;
    tracing::debug!(
        "pipeline: batch #{} committed — {} chunks, cursor {:?}",
        batches_done,
        pending.len(),
        last_done
    );
    pending.clear();
    Ok(())
}

impl IndexFile {
    /// Invalidate using a raw connection (mid-run modification path).
    fn invalidate_doc_conn(conn: &mut Connection, rel_path: &str) -> Result<(), ReindexError> {
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM vec_chunks WHERE rowid IN
               (SELECT chunk_id FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE d.path = ?1)",
            params![rel_path],
        )?;
        tx.execute(
            "DELETE FROM chunks WHERE doc_id IN
               (SELECT doc_id FROM documents WHERE path = ?1)",
            params![rel_path],
        )?;
        tx.execute(
            "UPDATE documents SET status = 'stale' WHERE path = ?1",
            params![rel_path],
        )?;
        tx.commit()?;
        Ok(())
    }
}

impl Registry {
    /// Final checkpoint write after completion.
    fn checkpoint_final(
        &self,
        run_id: &str,
        last_doc_path: &str,
        docs_done: i64,
        chunks_written: i64,
        batches_done: i64,
    ) -> Result<(), ReindexError> {
        self.conn.execute(
            "UPDATE reindex_checkpoints
             SET last_doc_path = ?2, docs_done = ?3, chunks_written = ?4,
                 batches_done = ?5,
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE run_id = ?1",
            params![run_id, last_doc_path, docs_done, chunks_written, batches_done],
        )?;
        Ok(())
    }
}

/// Monotonic-ish unique run id: hex nanos + pid (ulid-shaped, no dep).
fn new_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:016x}{:08x}", nanos, std::process::id())
}

/// Peak RSS from /proc (Linux), for the O(batch) proof in the checkpoint.
pub fn peak_rss_bytes() -> Option<i64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: i64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::chunk_text;
    use std::fs;
    use tempfile::tempdir;

    fn identity() -> IndexIdentity {
        IndexIdentity {
            provider: "test".into(),
            model: "fake-embed".into(),
            dims: 4,
        }
    }

    /// Deterministic fake embeddings: unit-ish vector derived from seq.
    fn fake_embed(_texts: Vec<String>) -> EmbedFuture {
        Box::pin(async move { Ok(_texts.iter().map(|_| vec![0.5f32, 0.5, 0.5, 0.5]).collect()) })
    }

    fn make_corpus(dir: &Path, files: &[(&str, &str)]) {
        for (name, content) in files {
            let path = dir.join(name);
            if let Some(p) = path.parent() {
                fs::create_dir_all(p).unwrap();
            }
            fs::write(path, content).unwrap();
        }
    }

    fn opts<'a>(root: &'a Path, index_dir: &'a Path, identity: &'a IndexIdentity) -> PipelineOpts<'a> {
        PipelineOpts {
            root,
            index_dir,
            identity,
            chunk_chars: 200,
            include: None,
            force: true,
            trigger: "manual",
        }
    }

    #[tokio::test]
    async fn test_pipeline_end_to_end() {
        let root = tempdir().unwrap();
        let idx = tempdir().unwrap();
        make_corpus(
            root.path(),
            &[
                ("a.md", "alpha\n".repeat(50).as_str()),
                ("b.md", "beta\n".repeat(30).as_str()),
                ("empty.md", ""),
            ],
        );
        let id = identity();
        let out = run_pipeline(opts(root.path(), idx.path(), &id), &fake_embed).await.unwrap();

        assert_eq!(out.docs_total, 3);
        assert!(out.chunks_written > 0);
        assert_eq!(out.skipped_midrun, 0);
        assert_eq!(out.invalidated_midrun, 0);

        // Index file: header + data + aligned vec rows.
        let index = IndexFile::open(&idx.path().join(id.filename()), &id).unwrap();
        let (docs, chunks) = index.counts().unwrap();
        assert_eq!(docs, 3); // empty.md recorded via DocDone
        assert!(chunks > 0);
        let (provider, model, dims, ver, chunker): (String, String, i64, i64, String) = index
            .conn()
            .query_row(
                "SELECT provider_id, model, dims, index_version, chunker_version FROM index_meta WHERE id = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .unwrap();
        assert_eq!(provider, "test/fake-embed");
        assert_eq!(model, "fake-embed");
        assert_eq!(dims, 4);
        assert_eq!(ver, INDEX_VERSION);
        assert_eq!(chunker, CHUNKER_VERSION);

        // rowid alignment: every chunk has its vector.
        let misaligned: i64 = index
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks c
                 WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.rowid = c.chunk_id)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(misaligned, 0);

        // Registry: run completed, index current.
        let reg = Registry::open(&idx.path().join("fleet-memory.db")).unwrap();
        let status: String = reg
            .conn()
            .query_row("SELECT status FROM reindex_runs WHERE run_id = ?1", params![out.run_id], |r| r.get(0))
            .unwrap();
        assert_eq!(status, "completed");
        let (cur_name, _): (String, String) = reg.current_row().unwrap().unwrap();
        assert_eq!(cur_name, "index.test.fake-embed.4");
        // Symlink flipped.
        assert!(idx.path().join("current").exists());
        // Checkpoint proves O(batch): peak_rss recorded.
        let rss: Option<i64> = reg
            .conn()
            .query_row("SELECT peak_rss_bytes FROM reindex_checkpoints WHERE run_id = ?1", params![out.run_id], |r| r.get(0))
            .unwrap();
        assert!(rss.unwrap_or(0) > 0);
    }

    #[tokio::test]
    async fn test_resume_after_failure_zero_duplicates() {
        let root = tempdir().unwrap();
        let idx = tempdir().unwrap();
        // Enough content for multiple batches (32 per batch).
        let mut files = Vec::new();
        for i in 0..12 {
            files.push((
                format!("f{i:02}.md"),
                format!("file {i}\n{}", "content line\n".repeat(60)),
            ));
        }
        let files_ref: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        make_corpus(root.path(), &files_ref);
        let id = identity();

        // Control: uninterrupted run chunk count.
        let ctrl_dir = tempdir().unwrap();
        make_corpus(ctrl_dir.path(), &files_ref);
        let ctrl = run_pipeline(opts(ctrl_dir.path(), idx.path().join("ctrl").as_path(), &id), &fake_embed)
            .await
            .unwrap();

        // Expected chunk count straight from the chunker.
        let expected: usize = files
            .iter()
            .map(|(_, c)| chunk_text(c, 200).unwrap().len())
            .sum();

        // Failing embedder: succeeds on the first batch, then dies
        // (simulates kill -9 mid-run; the checkpoint survives).
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let failing = move |texts: Vec<String>| -> EmbedFuture {
            let n = calls2.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if n >= 1 {
                    Err("simulated crash".into())
                } else {
                    Ok(texts.iter().map(|_| vec![0.5f32, 0.5, 0.5, 0.5]).collect())
                }
            })
        };
        let run_dir = idx.path().join("run");
        let first = run_pipeline(
            PipelineOpts {
                root: root.path(),
                index_dir: &run_dir,
                identity: &id,
                chunk_chars: 200,
                include: None,
                force: true,
                trigger: "manual",
            },
            &failing,
        )
        .await;
        assert!(first.is_err(), "first run must fail");

        // The failed run left a resumable cursor.
        let reg = Registry::open(&run_dir.join("fleet-memory.db")).unwrap();
        let resume = reg.load_resume("index.test.fake-embed.4").unwrap();
        assert!(resume.is_some(), "failed run must be resumable");

        // Resume with a healthy embedder (force=false).
        let healed = run_pipeline(
            PipelineOpts {
                root: root.path(),
                index_dir: &run_dir,
                identity: &id,
                chunk_chars: 200,
                include: None,
                force: false,
                trigger: "manual",
            },
            &fake_embed,
        )
        .await
        .unwrap();

        let index = IndexFile::open(&run_dir.join(id.filename()), &id).unwrap();
        let (_, chunks) = index.counts().unwrap();
        assert_eq!(
            chunks as usize, expected,
            "resumed run must equal the chunker's expected output"
        );
        assert_eq!(chunks as usize, ctrl.chunks_written, "must match control run");

        // Zero duplicate chunks: UNIQUE(doc_id, seq) holds with no IGNORE.
        let dups: i64 = index
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM (SELECT doc_id, seq FROM chunks GROUP BY doc_id, seq HAVING COUNT(*) > 1)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(dups, 0);

        // seq dense per doc.
        let gaps: i64 = index
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks c WHERE c.seq <> (
                    SELECT COUNT(*) FROM chunks c2 WHERE c2.doc_id = c.doc_id AND c2.seq < c.seq)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(gaps, 0);
        assert!(healed.chunks_written >= ctrl.chunks_written);
    }

    #[tokio::test]
    async fn test_provenance_refusal() {
        let idx = tempdir().unwrap();
        let id768 = IndexIdentity { provider: "p".into(), model: "m".into(), dims: 768 };
        let id1024 = IndexIdentity { provider: "p".into(), model: "m".into(), dims: 1024 };
        let path = idx.path().join(id768.filename());

        IndexFile::open(&path, &id768).unwrap();
        let err = IndexFile::open(&path, &id1024).unwrap_err();
        match err {
            ReindexError::Provenance { .. } => {}
            other => panic!("expected Provenance, got {other:?}"),
        }

        // Chunker version refusal.
        let conn = Connection::open(&path).unwrap();
        conn.execute("UPDATE index_meta SET chunker_version = 'ancient-v0'", []).unwrap();
        drop(conn);
        let err = IndexFile::open(&path, &id768).unwrap_err();
        assert!(matches!(err, ReindexError::ChunkerChanged { .. }));
    }

    #[tokio::test]
    async fn test_one_current_invariant() {
        let idx = tempdir().unwrap();
        let mut reg = Registry::open(&idx.path().join("fleet-memory.db")).unwrap();
        reg.ensure_provider(&identity()).unwrap();
        reg.set_current("index.a", "test/fake-embed", "/tmp/a.db", 1, 2).unwrap();
        reg.set_current("index.b", "test/fake-embed", "/tmp/b.db", 3, 4).unwrap();

        let (name, _): (String, String) = reg.current_row().unwrap().unwrap();
        assert_eq!(name, "index.b");

        // Direct violation of the partial unique index is rejected by SQLite.
        let result = reg.conn().execute(
            "UPDATE index_registry SET is_current = 1 WHERE index_name = 'index.a'",
            [],
        );
        assert!(result.is_err(), "partial unique index must reject a second current");
    }

    #[tokio::test]
    async fn test_midrun_modification_deferred_by_reader() {
        // Freeze, modify, then run the READER directly against the frozen
        // manifest: it must send Skipped (defer to next run), not chunks.
        let root = tempdir().unwrap();
        let out = tempdir().unwrap();
        fs::write(root.path().join("x.md"), "original\n").unwrap();
        fs::write(root.path().join("y.md"), "untouched\n").unwrap();
        let snap = snapshot::freeze(root.path(), None, out.path(), "r1").unwrap();

        // x.md modified AFTER the freeze (mid-run, from the run's viewpoint).
        fs::write(root.path().join("x.md"), "changed mid-run, different length\n").unwrap();

        let (tx, mut rx) = mpsc::channel::<PipeMsg>(CHANNEL_CAP);
        let handle = spawn_reader(
            root.path().to_path_buf(),
            snap.entries.clone(),
            String::new(),
            200,
            tx,
        );
        handle.join().unwrap().unwrap();

        let mut msgs = Vec::new();
        while let Some(m) = rx.try_recv().ok() {
            msgs.push(m);
        }
        // x.md skipped, y.md done, reader finished — and NO chunk for x.md.
        assert!(matches!(
            msgs.iter().find(|m| matches!(m, PipeMsg::Skipped { rel_path } if rel_path == "x.md")),
            Some(_)
        ));
        assert!(matches!(
            msgs.iter().find(|m| matches!(m, PipeMsg::DocDone { rel_path } if rel_path == "y.md")),
            Some(_)
        ));
        assert!(matches!(msgs.last(), Some(PipeMsg::Finished)));
        assert!(!msgs.iter().any(|m| matches!(m, PipeMsg::Chunk { doc, .. } if doc.rel_path == "x.md")));
        assert!(msgs.iter().any(|m| matches!(m, PipeMsg::Chunk { doc, .. } if doc.rel_path == "y.md")));
    }

    #[tokio::test]
    async fn test_invalidate_doc_drops_chunks_and_vectors() {
        let idx = tempdir().unwrap();
        let id = identity();
        fs::write(idx.path().join("corpus.md"), "some content\n").unwrap();
        let index = IndexFile::open(&idx.path().join(id.filename()), &id).unwrap();
        let mut conn = index.conn;

        // Insert a doc + chunk + vector by hand through the same helpers.
        let tx = conn.transaction().unwrap();
        let doc_id = IndexFile::upsert_doc(&tx, "corpus.md", "deadbeef", 1, 13).unwrap();
        let spec = crate::chunker::chunk_text("some content\n", 200).unwrap().remove(0);
        IndexFile::insert_chunk(&tx, doc_id, &spec, &[0.5, 0.5, 0.5, 0.5]).unwrap();
        tx.commit().unwrap();
        let (docs, chunks) = {
            let c = &conn;
            (
                c.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0)).unwrap(),
                c.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0)).unwrap(),
            )
        };
        assert_eq!((docs, chunks), (1, 1));

        // Invalidate: chunks + vectors gone, documents row marked stale.
        IndexFile::invalidate_doc_conn(&mut conn, "corpus.md").unwrap();
        let remaining_chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0)).unwrap();
        let remaining_vecs: i64 = conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0)).unwrap();
        let status: String = conn
            .query_row("SELECT status FROM documents WHERE path = 'corpus.md'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining_chunks, 0);
        assert_eq!(remaining_vecs, 0);
        assert_eq!(status, "stale");
    }
}
