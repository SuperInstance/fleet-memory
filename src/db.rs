//! SQLite database wrapper with sqlite-vec for vector search.
//!
//! Schema:
//! - `chunks` table with content, path, line numbers, embeddings
//! - `vec_chunks` virtual table (sqlite-vec) for vector similarity
//! - `meta` table for provider/model/dims/version metadata
//! - WAL mode for crash-safe concurrent reads

use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OptionalExtension};
use thiserror::Error;

use crate::index::IndexIdentity;

#[derive(Error, Debug)]
pub enum DbError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("sqlite-vec load failed: {0}")]
    VecLoad(String),
    #[error("dimension mismatch: index has {index_dims}, got {request_dims}")]
    DimMismatch { index_dims: usize, request_dims: usize },
    #[error("meta key not found: {0}")]
    MissingMeta(String),
    #[error("malformed embedding blob: expected {expected} bytes, got {got}")]
    BadEmbedding { expected: usize, got: usize },
}

/// Managed SQLite connection with WAL mode and sqlite-vec loaded.
pub struct Database {
    conn: Connection,
    path: PathBuf,
    identity: IndexIdentity,
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("path", &self.path)
            .field("identity", &self.identity)
            .finish()
    }
}

impl Database {
    /// Open or create an index database at the given path.
    /// The identity is stored in the meta table and checked on subsequent opens.
    pub fn open(path: &Path, identity: &IndexIdentity) -> Result<Self, DbError> {
        // Register sqlite-vec BEFORE opening the connection so vec0 is
        // available on this and all future connections.
        ensure_vec_registered();

        let conn = Connection::open(path)?;

        // Enable WAL mode
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;

        // Run schema migrations
        Self::migrate(&conn)?;

        // Check or write identity
        let stored_identity = Self::read_identity(&conn)?;
        if let Some(ref stored) = stored_identity {
            if stored != identity {
                return Err(DbError::DimMismatch {
                    index_dims: stored.dims,
                    request_dims: identity.dims,
                });
            }
        } else {
            Self::write_identity(&conn, identity)?;
        }

        let identity = stored_identity.unwrap_or_else(|| identity.clone());

        Ok(Self {
            conn,
            path: path.to_path_buf(),
            identity,
        })
    }

    /// Read-only open for queries (still WAL, no writes).
    pub fn open_readonly(path: &Path) -> Result<(Self, IndexIdentity), DbError> {
        ensure_vec_registered();

        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "query_only", true)?;
        conn.pragma_update(None, "busy_timeout", 5000)?;

        let identity = Self::read_identity(&conn)?
            .ok_or(DbError::MissingMeta("identity".into()))?;

        Ok((
            Self {
                conn,
                path: path.to_path_buf(),
                identity: identity.clone(),
            },
            identity,
        ))
    }

    fn migrate(conn: &Connection) -> Result<(), DbError> {
        // Main chunks table
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding BLOB,
                indexed_at REAL NOT NULL,
                reindex_offset INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reindex_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                offset INTEGER NOT NULL DEFAULT 0,
                total_files INTEGER,
                processed_files INTEGER NOT NULL DEFAULT 0,
                started_at REAL,
                updated_at REAL
            );

            INSERT OR IGNORE INTO reindex_state (id, offset) VALUES (1, 0);
            "#,
        )?;

        // Create the vec0 virtual table for vector search
        // dims are read from meta at query time
        let dims: Option<usize> = conn
            .query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'dims'",
                [],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(dims) = dims {
            let sql = format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding float[{}] distance_metric=cosine
                )",
                dims
            );
            conn.execute_batch(&sql)?;
        }

        Ok(())
    }

    fn read_identity(conn: &Connection) -> Result<Option<IndexIdentity>, DbError> {
        let get = |key: &str| -> Result<Option<String>, DbError> {
            Ok(conn
                .query_row(
                    "SELECT value FROM meta WHERE key = ?1",
                    params![key],
                    |row| row.get(0),
                )
                .optional()?)
        };

        let provider = match get("provider")? {
            Some(p) => p,
            None => return Ok(None),
        };
        let model = get("model")?.unwrap_or_default();
        let dims: usize = get("dims")?
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        Ok(Some(IndexIdentity {
            provider,
            model,
            dims,
        }))
    }

    fn write_identity(conn: &Connection, identity: &IndexIdentity) -> Result<(), DbError> {
        let now = chrono::Utc::now().timestamp() as f64;
        let kv = [
            ("provider", identity.provider.clone()),
            ("model", identity.model.clone()),
            ("dims", identity.dims.to_string()),
            ("version", "1".to_string()),
            ("created_at", now.to_string()),
            ("total_chunks", "0".to_string()),
        ];

        for (k, v) in kv {
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
                params![k, v],
            )?;
        }
        Ok(())
    }

    /// Create the vec0 virtual table (called after identity is written for new DBs).
    pub fn ensure_vec_table(&self) -> Result<(), DbError> {
        let sql = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{}] distance_metric=cosine
            )",
            self.identity.dims
        );
        self.conn.execute_batch(&sql)?;
        Ok(())
    }

    /// Insert a chunk with its embedding.
    pub fn insert_chunk(
        &self,
        path: &str,
        start_line: Option<i64>,
        end_line: Option<i64>,
        content: &str,
        content_hash: &str,
        embedding: &[f32],
        reindex_offset: i64,
    ) -> Result<i64, DbError> {
        let now = chrono::Utc::now().timestamp() as f64;
        let emb_bytes = embedding_to_bytes(embedding);

        self.conn.execute(
            "INSERT INTO chunks (path, start_line, end_line, content, content_hash, embedding, indexed_at, reindex_offset)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![path, start_line, end_line, content, content_hash, emb_bytes, now, reindex_offset],
        )?;

        let chunk_id = self.conn.last_insert_rowid();

        // Insert into vec table
        self.conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
            params![chunk_id, emb_bytes],
        )?;

        Ok(chunk_id)
    }

    /// Batch insert chunks in a single transaction.
    pub fn insert_chunks_batch(
        &mut self,
        chunks: &[(String, Option<i64>, Option<i64>, String, String, Vec<f32>)],
        reindex_offset: i64,
    ) -> Result<usize, DbError> {
        let now = chrono::Utc::now().timestamp() as f64;
        let tx = self.conn.transaction()?;

        let mut count = 0;
        for (path, start_line, end_line, content, content_hash, embedding) in chunks {
            let emb_bytes = embedding_to_bytes(embedding);

            tx.execute(
                "INSERT INTO chunks (path, start_line, end_line, content, content_hash, embedding, indexed_at, reindex_offset)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![path, start_line, end_line, content, content_hash, emb_bytes, now, reindex_offset],
            )?;

            let chunk_id = tx.last_insert_rowid();

            tx.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
                params![chunk_id, emb_bytes],
            )?;

            count += 1;
        }

        // Update reindex offset
        tx.execute(
            "UPDATE reindex_state SET offset = ?1, updated_at = ?2 WHERE id = 1",
            params![reindex_offset, now],
        )?;

        // Update total_chunks
        tx.execute(
            "UPDATE meta SET value = (SELECT COUNT(*) FROM chunks) WHERE key = 'total_chunks'",
            [],
        )?;

        tx.commit()?;
        Ok(count)
    }

    /// Get the current reindex checkpoint offset.
    pub fn get_checkpoint(&self) -> Result<i64, DbError> {
        let offset: i64 = self
            .conn
            .query_row("SELECT offset FROM reindex_state WHERE id = 1", [], |row| {
                row.get(0)
            })?;
        Ok(offset)
    }

    /// Set the reindex checkpoint offset.
    pub fn set_checkpoint(&self, offset: i64) -> Result<(), DbError> {
        let now = chrono::Utc::now().timestamp() as f64;
        self.conn.execute(
            "UPDATE reindex_state SET offset = ?1, updated_at = ?2 WHERE id = 1",
            params![offset, now],
        )?;
        Ok(())
    }

    /// Reset the reindex checkpoint to zero.
    pub fn reset_checkpoint(&self) -> Result<(), DbError> {
        let now = chrono::Utc::now().timestamp() as f64;
        self.conn.execute(
            "UPDATE reindex_state SET offset = 0, processed_files = 0, started_at = ?1, updated_at = ?1 WHERE id = 1",
            params![now],
        )?;
        Ok(())
    }

    /// Check if a content hash already exists (skip unchanged files).
    pub fn has_hash(&self, hash: &str) -> Result<bool, DbError> {
        let exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM chunks WHERE content_hash = ?1 LIMIT 1)",
            params![hash],
            |row| row.get(0),
        )?;
        Ok(exists)
    }

    /// Get total chunk count.
    pub fn chunk_count(&self) -> Result<i64, DbError> {
        let count: i64 = self.conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Get a meta value.
    pub fn get_meta(&self, key: &str) -> Result<Option<String>, DbError> {
        Ok(self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?)
    }

    /// Set a meta value.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<(), DbError> {
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    /// Get the index identity.
    pub fn identity(&self) -> &IndexIdentity {
        &self.identity
    }

    /// Get the database path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get a raw connection reference (for search queries).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

/// Convert a float32 slice to little-endian bytes for sqlite-vec.
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &v in embedding {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Register sqlite-vec's vec0 module globally via sqlite3_auto_extension.
/// This must be called BEFORE any Connection::open() so that every new
/// connection automatically has the vec0 module available. We use a Once
/// guard so the registration happens exactly once per process.
static VEC_INIT: std::sync::Once = std::sync::Once::new();

fn ensure_vec_registered() {
    VEC_INIT.call_once(|| {
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
        tracing::debug!("sqlite-vec auto-extension registered");
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_identity() -> IndexIdentity {
        IndexIdentity {
            provider: "test".into(),
            model: "test-model".into(),
            dims: 4, // small for tests
        }
    }

    #[test]
    fn test_open_and_identity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let id = make_identity();

        let db = Database::open(&path, &id).unwrap();
        db.ensure_vec_table().unwrap();

        // Reopen — should read the same identity
        let db2 = Database::open(&path, &id).unwrap();
        assert_eq!(db2.identity(), &id);
    }

    #[test]
    fn test_dim_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mismatch.db");
        let id1 = IndexIdentity {
            provider: "test".into(),
            model: "m".into(),
            dims: 128,
        };
        let id2 = IndexIdentity {
            provider: "test".into(),
            model: "m".into(),
            dims: 256,
        };

        Database::open(&path, &id1).unwrap();
        let result = Database::open(&path, &id2);
        assert!(result.is_err());
        match result.unwrap_err() {
            DbError::DimMismatch {
                index_dims: 128,
                request_dims: 256,
            } => {}
            other => panic!("expected DimMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_insert_and_count() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("insert.db");
        let id = make_identity();

        let db = Database::open(&path, &id).unwrap();
        db.ensure_vec_table().unwrap();

        let emb = vec![0.1f32, 0.2, 0.3, 0.4];
        db.insert_chunk(
            "/test/file.txt",
            Some(1),
            Some(10),
            "hello world",
            "abc123",
            &emb,
            0,
        )
        .unwrap();

        assert_eq!(db.chunk_count().unwrap(), 1);
    }

    #[test]
    fn test_batch_insert_and_checkpoint() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("batch.db");
        let id = make_identity();

        let mut db = Database::open(&path, &id).unwrap();
        db.ensure_vec_table().unwrap();

        let emb = vec![0.0f32, 1.0, 0.0, 1.0];
        let chunks = vec![
            ("/a.txt".into(), Some(1i64), Some(5i64), "content a".into(), "hash1".into(), emb.clone()),
            ("/b.txt".into(), Some(1i64), Some(3i64), "content b".into(), "hash2".into(), emb.clone()),
            ("/c.txt".into(), Some(1i64), Some(8i64), "content c".into(), "hash3".into(), emb),
        ];

        let count = db.insert_chunks_batch(&chunks, 42).unwrap();
        assert_eq!(count, 3);
        assert_eq!(db.chunk_count().unwrap(), 3);
        assert_eq!(db.get_checkpoint().unwrap(), 42);
    }

    #[test]
    fn test_checkpoint_reset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.db");
        let id = make_identity();

        let db = Database::open(&path, &id).unwrap();
        db.set_checkpoint(99).unwrap();
        assert_eq!(db.get_checkpoint().unwrap(), 99);

        db.reset_checkpoint().unwrap();
        assert_eq!(db.get_checkpoint().unwrap(), 0);
    }

    #[test]
    fn test_meta_get_set() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.db");
        let id = make_identity();

        let db = Database::open(&path, &id).unwrap();

        // Should have been written during open
        assert_eq!(db.get_meta("provider").unwrap().unwrap(), "test");
        assert_eq!(db.get_meta("dims").unwrap().unwrap(), "4");

        db.set_meta("custom", "value123").unwrap();
        assert_eq!(db.get_meta("custom").unwrap().unwrap(), "value123");
    }
}
