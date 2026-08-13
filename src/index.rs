//! Index manager — handles provider-tagged index files, symlink swapping,
//! and the streaming reindex pipeline.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use regex::Regex;
use sha2::{Digest, Sha256};
use thiserror::Error;
use walkdir::WalkDir;

use crate::db::Database;
use crate::embed::EmbeddingClient;
use crate::lock::IndexLock;

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("database error: {0}")]
    Db(#[from] crate::db::DbError),
    #[error("embed error: {0}")]
    Embed(#[from] crate::embed::EmbedError),
    #[error("lock error: {0}")]
    Lock(#[from] crate::lock::LockError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("index directory does not exist: {0}")]
    NoIndexDir(PathBuf),
    #[error("no current index symlink found in {0}")]
    NoCurrentIndex(PathBuf),
    #[error("failed to resolve symlink: {0}")]
    BadSymlink(PathBuf),
    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),
}

/// Provider+model+dims identity for an index file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexIdentity {
    pub provider: String,
    pub model: String,
    pub dims: usize,
}

impl IndexIdentity {
    /// Generate the index filename: `index.<provider>.<model>.<dims>.db`
    pub fn filename(&self) -> String {
        // Sanitize model name for filesystem
        let safe_model = self.model.replace('/', "-");
        format!(
            "index.{}.{}.{}.db",
            self.provider, safe_model, self.dims
        )
    }
}

impl std::fmt::Display for IndexIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{} ({}d)", self.provider, self.model, self.dims)
    }
}

/// Manages index files in a directory, including the `current` symlink.
pub struct IndexManager {
    pub index_dir: PathBuf,
}

impl IndexManager {
    pub fn new(index_dir: &str) -> Self {
        let dir = shellexpand(index_dir);
        Self {
            index_dir: PathBuf::from(dir),
        }
    }

    /// Ensure the index directory exists.
    pub fn ensure_dir(&self) -> Result<(), IndexError> {
        if !self.index_dir.exists() {
            fs::create_dir_all(&self.index_dir)?;
        }
        Ok(())
    }

    /// Path to the index file for the given identity.
    pub fn index_path(&self, identity: &IndexIdentity) -> PathBuf {
        self.index_dir.join(identity.filename())
    }

    /// Path to the `current` symlink.
    pub fn current_link(&self) -> PathBuf {
        self.index_dir.join("current")
    }

    /// Resolve the `current` symlink to the actual index file.
    pub fn resolve_current(&self) -> Result<PathBuf, IndexError> {
        let link = self.current_link();
        if !link.exists() {
            return Err(IndexError::NoCurrentIndex(self.index_dir.clone()));
        }
        let target = fs::canonicalize(&link).map_err(|_| IndexError::BadSymlink(link.clone()))?;
        Ok(target)
    }

    /// Atomically swap the `current` symlink to point at a new index.
    /// Uses rename(2) for atomicity: create a temp symlink, rename over `current`.
    pub fn swap_current(&self, target: &Path) -> Result<(), IndexError> {
        let current = self.current_link();
        let tmp = current.with_extension("db.symlink.tmp");

        // Get the target filename (relative to index_dir)
        let target_name = target
            .file_name()
            .ok_or_else(|| IndexError::BadSymlink(target.to_path_buf()))?;

        // Create temp symlink
        if tmp.exists() {
            fs::remove_file(&tmp)?;
        }
        std::os::unix::fs::symlink(target_name, &tmp)?;

        // Atomic rename
        fs::rename(&tmp, &current)?;

        tracing::info!(
            "swapped current -> {}",
            target_name.to_string_lossy()
        );
        Ok(())
    }

    /// List all index files in the directory.
    pub fn list_indexes(&self) -> Result<Vec<(PathBuf, IndexIdentity)>, IndexError> {
        let mut indexes = Vec::new();
        if !self.index_dir.exists() {
            return Ok(indexes);
        }

        for entry in fs::read_dir(&self.index_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Parse: index.<provider>.<model>.<dims>.db
            if let Some(parsed) = parse_index_filename(&name_str) {
                indexes.push((entry.path(), parsed));
            }
        }

        Ok(indexes)
    }

    /// Run a streaming reindex of the given root directory.
    ///
    /// Memory is O(batch_size * chunk_size) — never O(corpus).
    /// Progress is checkpointed after every batch for crash recovery.
    pub async fn reindex(
        &self,
        root: &Path,
        identity: &IndexIdentity,
        embedder: &EmbeddingClient,
        batch_size: usize,
        chunk_size: usize,
        include_filter: Option<&Regex>,
        force: bool,
    ) -> Result<ReindexStats, IndexError> {
        self.ensure_dir()?;

        let index_path = self.index_path(identity);
        let lock = IndexLock::acquire(&index_path)?;
        tracing::debug!("acquired index lock: {:?}", lock);

        let mut db = Database::open(&index_path, identity)?;
        db.ensure_vec_table()?;

        // Reset checkpoint if forced
        if force {
            tracing::info!("forced reindex — resetting checkpoint");
            db.reset_checkpoint()?;
        }

        // Snapshot the input file set at start (fixes "index changed while building")
        let files = snapshot_files(root, include_filter)?;
        let total_files = files.len();
        tracing::info!(
            "reindex: {} files under {}, batch_size={}, chunk_size={}",
            total_files,
            root.display(),
            batch_size,
            chunk_size
        );

        let checkpoint = db.get_checkpoint()?;
        let start_offset = if force { 0 } else { checkpoint };
        tracing::info!("resuming from offset {}", start_offset);

        let mut stats = ReindexStats {
            total_files,
            processed_files: 0,
            chunks_indexed: 0,
            skipped_unchanged: 0,
            errors: 0,
        };

        // Stream: read files in batches → embed → insert → checkpoint
        let mut offset = start_offset;
        let mut batch = Vec::with_capacity(batch_size);
        let mut batch_file_indices = Vec::with_capacity(batch_size);

        for (file_idx, file_path) in files.iter().enumerate() {
            if (file_idx as i64) < offset {
                continue;
            }

            // Read and chunk this file
            match read_and_chunk(file_path, chunk_size) {
                Ok(chunks) => {
                    for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
                        batch.push(chunk);
                        batch_file_indices.push((file_idx, chunk_idx));

                        if batch.len() >= batch_size {
                            let inserted = self
                                .process_batch(
                                    &mut db,
                                    embedder,
                                    &batch,
                                    &batch_file_indices,
                                    (file_idx + 1) as i64,
                                )
                                .await?;

                            stats.chunks_indexed += inserted;
                            stats.processed_files =
                                stats.processed_files.max(file_idx + 1);
                            batch.clear();
                            batch_file_indices.clear();
                        }
                    }
                    // After all chunks of this file, checkpoint the file offset
                    db.set_checkpoint((file_idx + 1) as i64)?;
                    offset = (file_idx + 1) as i64;
                }
                Err(e) => {
                    tracing::warn!("skipping {}: {}", file_path.display(), e);
                    stats.errors += 1;
                    db.set_checkpoint((file_idx + 1) as i64)?;
                }
            }
        }

        // Process remaining batch
        if !batch.is_empty() {
            let inserted = self
                .process_batch(
                    &mut db,
                    embedder,
                    &batch,
                    &batch_file_indices,
                    files.len() as i64,
                )
                .await?;
            stats.chunks_indexed += inserted;
        }

        // Update symlink if this is a new or updated index
        self.swap_current(&index_path)?;

        tracing::info!(
            "reindex complete: {} files, {} chunks, {} errors",
            stats.processed_files,
            stats.chunks_indexed,
            stats.errors
        );

        Ok(stats)
    }

    /// Embed and insert a batch of chunks.
    async fn process_batch(
        &self,
        db: &mut Database,
        embedder: &EmbeddingClient,
        batch: &[TextChunk],
        _indices: &[(usize, usize)],
        reindex_offset: i64,
    ) -> Result<usize, IndexError> {
        let texts: Vec<String> = batch.iter().map(|c| c.content.clone()).collect();

        let embeddings = embedder.embed_batch(&texts).await?;

        let chunks_data: Vec<(
            String,
            Option<i64>,
            Option<i64>,
            String,
            String,
            Vec<f32>,
        )> = batch
            .iter()
            .zip(embeddings.iter())
            .map(|(chunk, emb)| {
                let hash = hash_content(&chunk.content);
                (
                    chunk.path.clone(),
                    Some(chunk.start_line as i64),
                    Some(chunk.end_line as i64),
                    chunk.content.clone(),
                    hash,
                    emb.clone(),
                )
            })
            .collect();

        let count = db.insert_chunks_batch(&chunks_data, reindex_offset)?;

        tracing::debug!(
            "inserted {} chunks at offset {}",
            count,
            reindex_offset
        );

        Ok(count)
    }
}

/// A text chunk from a file.
#[derive(Debug, Clone)]
struct TextChunk {
    path: String,
    start_line: usize,
    end_line: usize,
    content: String,
}

/// Statistics from a reindex run.
#[derive(Debug, Clone)]
pub struct ReindexStats {
    pub total_files: usize,
    pub processed_files: usize,
    pub chunks_indexed: usize,
    pub skipped_unchanged: usize,
    pub errors: usize,
}

/// Recursively collect text files under root, filtered by optional regex.
fn snapshot_files(root: &Path, include: Option<&Regex>) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut files = Vec::new();

    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let path_str = path.to_string_lossy();

        // Skip hidden files/dirs
        if path_str.contains("/.") {
            continue;
        }

        // Skip binary-looking extensions
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if matches!(
                ext.as_str(),
                "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "ico" | "mp3" | "wav"
                    | "mp4" | "avi" | "mov" | "webm" | "pdf" | "zip" | "gz" | "tar"
                    | "bz2" | "7z" | "exe" | "dll" | "so" | "dylib" | "o" | "a"
                    | "class" | "jar" | "wasm" | "db" | "sqlite" | "bin"
            ) {
                continue;
            }
        }

        // Apply include filter
        if let Some(ref re) = include {
            if !re.is_match(&path_str) {
                continue;
            }
        }

        files.push(path.to_path_buf());
    }

    // Sort for deterministic ordering (checkpoint consistency)
    files.sort();
    Ok(files)
}

/// Read a file and split into line-based chunks of approximately `max_chars` characters.
fn read_and_chunk(path: &Path, max_chars: usize) -> Result<Vec<TextChunk>, std::io::Error> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;

    if lines.is_empty() {
        return Ok(Vec::new());
    }

    let path_str = path.to_string_lossy().to_string();
    let mut chunks = Vec::new();
    let mut current_lines = Vec::new();
    let mut current_len = 0;
    let mut start_line = 1;

    for (idx, line) in lines.iter().enumerate() {
        let line_len = line.len() + 1; // +1 for newline
        if current_len + line_len > max_chars && !current_lines.is_empty() {
            // Flush current chunk
            let end_line = idx;
            chunks.push(TextChunk {
                path: path_str.clone(),
                start_line,
                end_line,
                content: current_lines.join("\n"),
            });
            current_lines.clear();
            current_len = 0;
            start_line = idx + 1;
        }
        current_lines.push(line.clone());
        current_len += line_len;
    }

    // Flush remaining
    if !current_lines.is_empty() {
        chunks.push(TextChunk {
            path: path_str.clone(),
            start_line,
            end_line: lines.len(),
            content: current_lines.join("\n"),
        });
    }

    Ok(chunks)
}

/// SHA-256 hash of content for change detection.
fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Parse an index filename: `index.<provider>.<model>.<dims>.db`
fn parse_index_filename(name: &str) -> Option<IndexIdentity> {
    if !name.starts_with("index.") || !name.ends_with(".db") {
        return None;
    }

    // Strip prefix and suffix
    let inner = &name["index.".len()..name.len() - ".db".len()];

    // Split into provider.model.dims — dims is last, model is everything between
    let parts: Vec<&str> = inner.rsplitn(3, '.').collect();
    if parts.len() != 3 {
        return None;
    }

    // parts are [dims, model, provider] due to rsplitn
    let dims: usize = parts[0].parse().ok()?;
    let model = parts[1].to_string();
    let provider = parts[2].to_string();

    Some(IndexIdentity {
        provider,
        model,
        dims,
    })
}

/// Expand ~ in paths.
fn shellexpand(path: &str) -> String {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{stripped}");
        }
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_index_filename() {
        let id = parse_index_filename("index.ollama.nomic-embed-text.768.db").unwrap();
        assert_eq!(id.provider, "ollama");
        assert_eq!(id.model, "nomic-embed-text");
        assert_eq!(id.dims, 768);
    }

    #[test]
    fn test_parse_index_filename_simple() {
        let id = parse_index_filename("index.openai.text-embedding-3-small.1536.db").unwrap();
        assert_eq!(id.provider, "openai");
        assert_eq!(id.dims, 1536);
    }

    #[test]
    fn test_parse_index_filename_invalid() {
        assert!(parse_index_filename("not-an-index.db").is_none());
        assert!(parse_index_filename("index.foo.db").is_none());
    }

    #[test]
    fn test_identity_filename() {
        let id = IndexIdentity {
            provider: "ollama".into(),
            model: "nomic-embed-text".into(),
            dims: 768,
        };
        assert_eq!(id.filename(), "index.ollama.nomic-embed-text.768.db");
    }

    #[test]
    fn test_hash_content() {
        let h1 = hash_content("hello world");
        let h2 = hash_content("hello world");
        let h3 = hash_content("goodbye world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_shellexpand() {
        let expanded = shellexpand("~/foo/bar");
        assert!(expanded.contains("foo/bar"));
        assert!(!expanded.starts_with("~"));

        let unchanged = shellexpand("/absolute/path");
        assert_eq!(unchanged, "/absolute/path");
    }

    use std::io::Write;

    #[test]
    fn test_read_and_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let mut f = fs::File::create(&path).unwrap();
        for i in 0..100 {
            writeln!(f, "Line number {}", i).unwrap();
        }

        let chunks = read_and_chunk(&path, 100).unwrap();
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].start_line, 1);
        // Each chunk should not exceed ~100 chars
        for chunk in &chunks {
            assert!(chunk.content.len() < 200); // some slack
        }
    }
}
