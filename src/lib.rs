//! fleet-memory — Streaming memory index with sqlite-vec
//!
//! Provider-tagged index files, crash-recoverable streaming reindex,
//! atomic symlink swap, flock-based locking. Memory is O(batch).

pub mod cli;
pub mod db;
pub mod embed;
pub mod index;
pub mod lock;
pub mod search;

pub use cli::{Cli, Commands};
pub use db::Database;
pub use embed::EmbeddingClient;
pub use index::{IndexManager, IndexIdentity};
pub use lock::IndexLock;
pub use search::{SearchResult, Searcher};
