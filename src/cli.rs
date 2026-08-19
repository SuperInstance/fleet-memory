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

    /// Registry database path (defaults to <index_dir>/fleet-memory.db)
    #[arg(long, global = true)]
    pub registry: Option<PathBuf>,
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

    /// Phase 4 streaming pipeline: snapshot manifest → bounded channel →
    /// embedder (batches of 32) → one transaction per batch, with
    /// checkpoint/resume and the registry cutover
    Reindex {
        /// Root directory to scan for text files
        #[arg(long)]
        root: PathBuf,

        /// Force a fresh run (ignore any crashed run's checkpoint)
        #[arg(long)]
        force: bool,

        /// File patterns to include (regex), e.g. "\.md$|\.txt$|\.rs$"
        #[arg(long)]
        include: Option<String>,

        /// Trigger kind recorded in reindex_runs
        #[arg(long, default_value = "manual")]
        trigger: String,
    },

    /// Q1: find pieces about a phrase (tagged + full-text + semantic lanes)
    Find {
        /// The phrase to look for
        phrase: String,

        /// K for the semantic lane's KNN query
        #[arg(long, default_value = "20")]
        k: usize,
    },

    /// Q2: show all renders for a work, by slug
    Renders {
        /// Work slug, e.g. 'pfd-speech'
        slug: String,
    },

    /// Q3: what was decided on a given date (YYYY-MM-DD)
    Decided {
        /// UTC day, e.g. 2026-08-13
        date: String,
    },
}
