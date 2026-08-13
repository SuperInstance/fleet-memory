//! CLI definitions using clap.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "fleet-memory")]
#[command(version, about = "Streaming memory index with sqlite-vec")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Base directory for index files
    #[arg(long, global = true, default_value = "~/.openclaw/agents/main/agent")]
    pub index_dir: String,

    /// Embedding API endpoint (fleet-gateway)
    #[arg(long, global = true, default_value = "http://127.0.0.1:8787/v1")]
    pub gateway: String,

    /// Embedding provider name
    #[arg(long, global = true, default_value = "ollama")]
    pub provider: String,

    /// Embedding model name
    #[arg(long, global = true, default_value = "nomic-embed-text")]
    pub model: String,

    /// Embedding dimensions
    #[arg(long, global = true, default_value = "768")]
    pub dims: usize,

    /// Batch size for streaming reindex
    #[arg(long, global = true, default_value = "32")]
    pub batch_size: usize,

    /// Maximum chunk size in characters
    #[arg(long, global = true, default_value = "2000")]
    pub chunk_size: usize,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Build or rebuild the index
    Index {
        /// Root directory to scan for text files
        #[arg(long)]
        root: PathBuf,

        /// Force full reindex (ignore checkpoint)
        #[arg(long)]
        force: bool,

        /// File patterns to include (regex), e.g. "\\.md$|\\.txt$|\\.rs$"
        #[arg(long)]
        include: Option<String>,
    },

    /// Search the current index
    Search {
        /// Query text to embed and search for
        #[arg(long)]
        query: String,

        /// Maximum number of results
        #[arg(long, default_value = "10")]
        limit: usize,

        /// Minimum cosine similarity threshold (0.0 - 1.0)
        #[arg(long, default_value = "0.0")]
        threshold: f64,
    },

    /// Show index status
    Status,

    /// List all index files
    List,

    /// Switch the current symlink to a different index
    Switch {
        /// Index file to point 'current' at
        #[arg(long)]
        target: String,
    },
}
