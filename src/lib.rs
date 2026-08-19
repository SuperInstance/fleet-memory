//! fleet-memory — Streaming memory index with sqlite-vec
//!
//! Provider-tagged index files, crash-recoverable streaming reindex,
//! atomic symlink swap, flock-based locking. Memory is O(batch).
//!
//! Phase 4 adds the two-database model from the canonical schema:
//! a permanent registry (`fleet-memory.db`) and disposable, provenance-
//! tagged index holds (`index.<provider>.<model>.<dims>.db`).

pub mod chunker;
pub mod cli;
pub mod db;
pub mod embed;
pub mod index;
pub mod lock;
pub mod query;
pub mod reindex;
pub mod search;
pub mod snapshot;

pub use chunker::{ChunkSpec, CHUNKER_VERSION};
pub use cli::{Cli, Commands};
pub use db::Database;
pub use embed::EmbeddingClient;
pub use index::{IndexManager, IndexIdentity};
pub use lock::IndexLock;
pub use query::{DecisionRow, FindReport, RenderRow};
pub use reindex::{Registry, ReindexOutcome, BATCH_SIZE, CHANNEL_CAP, INDEX_VERSION};
pub use search::{SearchResult, Searcher};
pub use snapshot::{Snapshot, SnapshotEntry};
