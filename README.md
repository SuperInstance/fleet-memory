# Fleet Memory

**A streaming memory index with [sqlite-vec](https://github.com/asg0171/sqlite-vec) vector search, provider-tagged schemas, and crash recovery — the fleet's semantic memory.**

> *The fleet reads code by lines and remembers it by meaning. This is how it remembers.*

---

## Table of Contents

- [Vision](#vision)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Key Concepts](#key-concepts)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Further Reading](#further-reading)
- [Relation to the Fleet](#relation-to-the-fleet)

---

## Vision

The fleet has hundreds of thousands of lines of code, thousands of documents, and a growing corpus of creative writing. Finding the right piece of information — by *meaning*, not by keyword — requires a [vector search](https://en.wikipedia.org/wiki/Vector_database) engine.

Fleet Memory is that engine. It walks a directory tree, chunks files into readable passages, generates [embeddings](https://en.wikipedia.org/wiki/Word_embedding) via the [fleet-gateway](https://github.com/SuperInstance/fleet-gateway), stores them in [SQLite](https://www.sqlite.org/) with the [sqlite-vec](https://github.com/asg0171/sqlite-vec) extension, and answers [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) queries in milliseconds.

### Why Not Use Pinecone/Weaviate/qdrant?

| Managed Vector DB | Fleet Memory |
|---|---|
| Network round-trip per query | Local disk, zero-latency reads |
| $X/month per GB stored | Free — it's a file |
| Ops burden (auth, scaling, monitoring) | One binary, one `.db` file |
| Vendor lock-in | Open format ([SQLite](https://www.sqlite.org/fileformat.html)) |
| Fixed schema | Provider-tagged: swap embedding models without rebuilding |

Fleet Memory is a local-first, single-file, crash-resistant vector database. It runs in [WAL mode](https://www.sqlite.org/wal.html) for concurrent readers, checkpoints progress after every batch, and uses [flock](https://en.wikipedia.org/wiki/File_locking) for exclusive write access. If the process dies — even by `kill -9` — the index is never corrupted.

### Design Principles

1. **[Memory is O(chunk), never O(corpus)](https://en.wikipedia.org/wiki/Big_O_notation)** — files are streamed in batches. A 10GB corpus uses the same constant memory as a 10MB one.
2. **[Crash-safe by construction](https://www.sqlite.org/atomiccommit.html)** — SQLite WAL + flock + checkpointing means never losing progress.
3. **[Provider-tagged indexes](#index-identity)** — switch embedding models without rebuilding. Each model gets its own index file.
4. **[Atomic symlink swapping](https://en.wikipedia.org/wiki/Atomic_operation)** — the `current` pointer is updated via `rename(2)`, so readers never see a partial index.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        fleet-memory                                  │
│                                                                     │
│  CLI (clap)                                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────┐  ┌─────────┐    │
│  │  index   │  │  search  │  │ status │  │ list │  │ switch  │    │
│  └────┬─────┘  └────┬─────┘  └────────┘  └──────┘  └─────────┘    │
│       │              │                                              │
│       ▼              ▼                                              │
│  ┌─────────┐   ┌──────────┐                                        │
│  │ Index   │   │ Searcher │                                        │
│  │ Manager │   │          │                                        │
│  └────┬────┘   └────┬─────┘                                        │
│       │              │                                              │
│       ▼              ▼                                              │
│  ┌──────────────────────────────┐                                   │
│  │       Database (SQLite)       │                                   │
│  │  ┌────────┐  ┌─────────────┐ │                                   │
│  │  │ chunks │  │ vec_chunks  │ │                                   │
│  │  │ table  │  │ (vec0 vtab) │ │                                   │
│  │  └────────┘  └─────────────┘ │                                   │
│  │  ┌────────┐  ┌─────────────┐ │                                   │
│  │  │  meta  │  │reindex_state│ │                                   │
│  │  └────────┘  └─────────────┘ │                                   │
│  └──────────────────────────────┘                                   │
│       ▲                                                             │
│       │ embeddings                                                  │
│  ┌────┴─────────────┐                                              │
│  │ Embedding Client  │──► fleet-gateway ──► Ollama / OpenAI        │
│  └──────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Map

| Module | Lines | Responsibility |
|--------|-------|---------------|
| [`main.rs`](src/main.rs) | 188 | Entry point: parse CLI, dispatch to subcommands, print results |
| [`index.rs`](src/index.rs) | 560 | Index manager: file walking, chunking, batch embedding, checkpointing, symlink swap |
| [`db.rs`](src/db.rs) | 543 | SQLite wrapper: schema migrations, WAL mode, batch inserts, vec0 virtual table |
| [`search.rs`](src/search.rs) | 178 | Vector similarity search via sqlite-vec cosine distance |
| [`embed.rs`](src/embed.rs) | 105 | OpenAI-compatible embedding client (talks to fleet-gateway) |
| [`lock.rs`](src/lock.rs) | 131 | flock-based exclusive lock (kernel-released on process death) |
| [`cli.rs`](src/cli.rs) | 86 | CLI argument definitions via [clap](https://docs.rs/clap) |

### Data Flow: Indexing

```
1. Walk directory tree (walkdir), filtering by extension and include regex
2. Sort files deterministically (for checkpoint consistency)
3. For each file:
   a. Read and chunk into ~2000-char passages (line-based boundaries)
   b. SHA-256 hash each chunk for change detection
   c. Accumulate chunks into batches of batch_size
   d. When batch is full:
      - Send texts to fleet-gateway /v1/embeddings
      - Receive embedding vectors
      - INSERT batch in a single SQLite transaction
      - UPDATE checkpoint offset
4. Atomically swap `current` symlink → new index file
5. Print stats
```

### Data Flow: Searching

```
1. Resolve `current` symlink → index file path
2. Open SQLite in read-only mode
3. Embed the query text via fleet-gateway
4. SELECT from vec_chunks WHERE embedding MATCH ? ORDER BY distance
5. JOIN with chunks table for content + metadata
6. Filter by similarity threshold
7. Return ranked results with file paths and line numbers
```

---

## Quick Start

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.75+ (2021 edition)
- [fleet-gateway](https://github.com/SuperInstance/fleet-gateway) running on `http://127.0.0.1:8787`
- An embedding model available through the gateway (default: `nomic-embed-text` via [Ollama](https://ollama.ai))

### Build

```bash
git clone https://github.com/SuperInstance/fleet-memory.git
cd fleet-memory
cargo build --release
```

### Index a Directory

```bash
# Index your projects directory
./target/release/fleet-memory index \
  --root ~/projects \
  --include '\.rs$|\.ts$|\.py$|\.md$'

# Force a full reindex (ignore checkpoint)
./target/release/fleet-memory index --root ~/projects --force
```

### Search

```bash
# Semantic search
./target/release/fleet-memory search \
  --query "circuit breaker pattern implementation" \
  --limit 5 \
  --threshold 0.7
```

### Check Status

```bash
./target/release/fleet-memory status
```

Output:
```
📂 Index directory:  ~/.openclaw/agents/main/agent/current
🆔 Provider:          ollama
   Model:             nomic-embed-text
   Dimensions:        768
📊 Total chunks:      15432
📋 Version:           1
🕐 Created:           2026-08-13 14:23:00 UTC
💾 Index file:        ~/.openclaw/agents/main/agent/index.ollama.nomic-embed-text.768.db
```

---

## Key Concepts

### Vector Embeddings

An [embedding](https://en.wikipedia.org/wiki/Word_embedding) is a vector (a list of numbers) that captures the *meaning* of a text passage. Texts with similar meanings have vectors that point in similar directions. This lets you search by *concept* instead of by keyword.

The canonical example: "The cat sat on the mat" and "A feline rested on the rug" share no keywords, but their embeddings are nearly identical because they mean the same thing.

Fleet Memory uses the [OpenAI-compatible embeddings API](https://platform.openai.com/docs/guides/embeddings) — any model that speaks that protocol works. The default is [`nomic-embed-text`](https://ollama.ai/library/nomic-embed-text) (768 dimensions, runs locally via Ollama), but you can use [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) (1024 dims via DeepInfra) or OpenAI's `text-embedding-3-small` (1536 dims).

#### Further Reading on Embeddings

- [Vector Embeddings (Wikipedia)](https://en.wikipedia.org/wiki/Word_embedding) — the foundational concept
- [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) by Jay Alammar — visual explanation
- [Sentence Embeddings (SBERT)](https://www.sbert.net/) — modern sentence-level embeddings
- [Deep Learning for Vector Search](https://engineer.evri.com/blog/vector-search/) — production perspective

### Cosine Similarity

[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) measures the angle between two vectors, regardless of their magnitude. For normalized vectors (unit length), it ranges from -1 (opposite) through 0 (orthogonal) to 1 (identical).

$$\text{similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \cos(\theta)$$

sqlite-vec uses **cosine distance** = 1 - cosine_similarity. So lower distance = more similar. Fleet Memory converts back to similarity scores (0.0 to 1.0) for user-facing results.

#### Further Reading on Similarity Metrics

- [Cosine Similarity (Wikipedia)](https://en.wikipedia.org/wiki/Cosine_similarity) — mathematical definition
- [Euclidean vs Cosine Distance](https://www.machinelearningplus.com/statistics/cosine-similarity/) — when to use which
- [Hypersphere Geometry](https://en.wikipedia.org/wiki/Hypersphere) — why high-dimensional vectors live on a sphere

### Index Identity

Each index file is tagged with its embedding provider, model, and dimensionality:

```
index.ollama.nomic-embed-text.768.db
       ──┬──  ──────┬───────  ─┬─
       provider    model      dims
```

This means you can have **multiple indexes simultaneously** — one per embedding model. Switch between them with `fleet-memory switch`. The `current` symlink points at the active one.

### Streaming Pipeline

The reindex pipeline is **streaming** — it processes files in batches and never loads the entire corpus into memory:

```
File system ──► [walker] ──► [chunker] ──► [batch buffer] ──► [embed API] ──► [SQLite INSERT]
                    │              │              │                   │                │
                    │              │              │                   │                ▼
                    │              │              │                   │         [checkpoint]
                    │              │              │                   │           offset += N
                    ▼              ▼              ▼
                 (filtered)    (text chunks)  (O(batch_size * chunk_size) memory)
```

Memory usage is `O(batch_size × chunk_size)` — typically 32 × 2000 chars = ~64KB. Whether your corpus is 10 files or 10 million, the memory footprint stays the same.

### Crash Recovery

Three mechanisms ensure crash safety:

1. **[SQLite WAL Mode](https://www.sqlite.org/wal.html)** — Write-Ahead Logging means readers never block writers. If the process crashes, the WAL is replayed on next open, restoring consistency.

2. **[flock Index Lock](https://man7.org/linux/man-pages/man2/flock.2.html)** — An exclusive advisory lock (`LOCK_EX | LOCK_NB`) on `<index>.lock`. If the process dies (even `kill -9`), the kernel releases the lock automatically. No stale locks, no manual cleanup.

3. **Checkpoint Offsets** — After each batch, the file offset is written to `reindex_state`. If indexing is interrupted, it resumes from the last checkpoint. The `--force` flag resets the checkpoint to zero for a full rebuild.

### Atomic Symlink Swap

When indexing completes, the `current` symlink is updated to point at the new index. This is done atomically:

```
1. Create temp symlink: tmp → index.ollama.nomic-embed-text.768.db
2. rename(tmp, current)    ← atomic on the same filesystem (POSIX guarantee)
```

Readers currently using the old index continue working (they have an open fd). New readers get the new index. There is never a moment where `current` points at a partial or missing file.

See: [`rename(2)`](https://man7.org/linux/man-pages/man2/rename.2.html) — "If newpath already exists, it will be atomically replaced."

---

## CLI Reference

### `fleet-memory index`

Build or rebuild the index.

| Flag | Default | Description |
|------|---------|-------------|
| `--root` | (required) | Root directory to scan for text files |
| `--force` | false | Reset checkpoint and do a full reindex |
| `--include` | (none) | Regex pattern for file inclusion (e.g., `\.rs$\|\.md$`) |
| `--index-dir` | `~/.openclaw/agents/main/agent` | Directory for index files |
| `--gateway` | `http://127.0.0.1:8787/v1` | Fleet gateway URL |
| `--provider` | `ollama` | Embedding provider name (for tagging) |
| `--model` | `nomic-embed-text` | Embedding model name |
| `--dims` | `768` | Embedding dimensions |
| `--batch-size` | `32` | Files per embedding batch |
| `--chunk-size` | `2000` | Max characters per chunk |

### `fleet-memory search`

Search the current index.

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | (required) | Text to search for |
| `--limit` | `10` | Max results |
| `--threshold` | `0.0` | Minimum cosine similarity (0.0–1.0) |

### `fleet-memory status`

Show current index info: provider, model, dimensions, chunk count, creation time.

### `fleet-memory list`

List all index files in the index directory, marking the current one.

### `fleet-memory switch --target <file>`

Point the `current` symlink at a different index file.

---

## Configuration

Fleet Memory has no config file — all configuration is via CLI flags and environment variables.

### Key Defaults

| Setting | Default | How to Override |
|---------|---------|-----------------|
| Gateway URL | `http://127.0.0.1:8787/v1` | `--gateway` |
| Index directory | `~/.openclaw/agents/main/agent` | `--index-dir` |
| Embedding model | `nomic-embed-text` (Ollama) | `--model` |
| Dimensions | `768` | `--dims` |
| Batch size | `32` | `--batch-size` |
| Chunk size | `2000` chars | `--chunk-size` |

### Dependencies

| Crate | Version | Purpose |
|---|---|---|
| [rusqlite](https://docs.rs/rusqlite) | 0.32 | SQLite bindings (bundled) |
| [sqlite-vec](https://github.com/asg0171/sqlite-vec) | 0.1.6-alpha | Loadable vector search extension |
| [tokio](https://docs.rs/tokio) | 1 | Async runtime |
| [reqwest](https://docs.rs/reqwest) | 0.12 | HTTP client for embeddings API |
| [clap](https://docs.rs/clap) | 4 | CLI parsing |
| [walkdir](https://docs.rs/walkdir) | 2 | Recursive directory traversal |
| [sha2](https://docs.rs/sha2) | 0.10 | Content hashing for change detection |
| [regex](https://docs.rs/regex) | 1 | File inclusion filter |
| [serde](https://docs.rs/serde) / [serde_json](https://docs.rs/serde_json) | 1 | JSON for API communication |
| [tracing](https://docs.rs/tracing) | 0.1 | Structured logging |
| [thiserror](https://docs.rs/thiserror) | 2 | Error derive macros |

---

## Testing

```bash
cargo test        # 20+ tests across all modules
cargo clippy      # zero warnings
```

Tests cover:
- Database open/close, identity checking, dimension mismatch detection
- Batch insert and checkpoint persistence
- flock contention and release
- Vector search with known embeddings (cosine similarity verification)
- Threshold filtering
- File chunking (line-based boundary detection)
- Content hashing
- Index filename parsing
- Shell expansion (`~/` → `$HOME`)

All tests use [`tempfile::tempdir()`](https://docs.rs/tempfile) for isolation — no test pollution, no leftover files.

---

## Deployment

### As a CLI Tool

```bash
# Build and install to ~/.cargo/bin
cargo install --path .

# Use from anywhere
fleet-memory index --root ~/projects
fleet-memory search --query "how does the circuit breaker work?"
```

### With fleet-gateway

Fleet Memory is designed to work with [fleet-gateway](https://github.com/SuperInstance/fleet-gateway) as its embedding backend:

```bash
# 1. Start the gateway (which routes to Ollama for embeddings)
systemctl --user start fleet-gateway

# 2. Index
fleet-memory index --root ~/projects --gateway http://127.0.0.1:8787/v1

# 3. Search
fleet-memory search --query "error handling pattern" --gateway http://127.0.0.1:8787/v1
```

### Scheduled Reindexing (Cron)

```bash
# Reindex nightly at 3 AM
0 3 * * * /home/eileen/.cargo/bin/fleet-memory index --root /home/eileen/projects --force >> /tmp/fleet-memory-reindex.log 2>&1
```

---

## Further Reading

### For Developers

- [sqlite-vec Documentation](https://github.com/asg0171/sqlite-vec) — the vector search extension used
- [SQLite WAL Mode](https://www.sqlite.org/wal.html) — how concurrent reads work
- [SQLite File Format](https://www.sqlite.org/fileformat.html) — what's actually in a `.db` file
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) — the API standard followed
- [Ollama Embedding Models](https://ollama.ai/blog/embedding-models) — local embedding models

### For Engineers

- [Vector Database Benchmarks](https://ann-benchmarks.com/) —ANN algorithm comparisons
- [FAISS Paper](https://arxiv.org/abs/2401.08281) — Facebook's similarity search (the algorithmic foundation)
- [HNSW: Hierarchical Navigable Small World](https://arxiv.org/abs/1603.09320) — the most popular ANN algorithm
- [Product Quantization (PQ)](https://lear.inrialpes.fr/pubs/2011/JDSG11/jegou_searchingwithquantization.pdf) — compressing vectors for memory efficiency
- [POSIX flock(2)](https://man7.org/linux/man-pages/man2/flock.2.html) — the locking primitive used

### For Mathematicians

- [Cosine Similarity (Wikipedia)](https://en.wikipedia.org/wiki/Cosine_similarity) — formal definition
- [Inner Product Space (Wikipedia)](https://en.wikipedia.org/wiki/Inner_product_space) — the vector space where embeddings live
- [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) — why high-dimensional search is hard
- [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) — dimension reduction guarantees
- [k-Nearest Neighbors (Wikipedia)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) — the search problem

### For Students

- [What is a Vector Database?](https://www.pinecone.io/learn/vector-database/) — beginner-friendly intro (Pinecone)
- [Embeddings: Text as Numbers](https://www.tensorflow.org/text/guide/word2vec) — TensorFlow tutorial
- [SQLite Tutorial](https://www.sqlitetutorial.net/) — learn the database engine
- [Big O Notation (Khan Academy)](https://www.khanacademy.org/computing/ap-csp/algorithms/big-o-notation/a/analyzing-the-efficiency-of-algorithms) — why O(chunk) matters

---

## Relation to the Fleet

| Component | Relationship |
|---|---|
| **[fleet-gateway](https://github.com/SuperInstance/fleet-gateway)** | Provides the embeddings API endpoint. Fleet Memory is the gateway's biggest customer for `/v1/embeddings`. |
| **[fleet-jepa-midi](https://github.com/SuperInstance/fleet-jepa-midi)** | Uses Fleet Memory to store and retrieve MIDI corpus embeddings for JEPA training |
| **[fleet-wiki](https://github.com/SuperInstance/fleet-wiki)** | Complementary: wiki uses Cloudflare Vectorize for cloud search; fleet-memory is the local equivalent |
| **[OpenClaw](https://github.com/SuperInstance/openclaw)** | Uses fleet-memory's index for memory recall in agent sessions |

---

## License

MIT — part of the [SuperInstance](https://github.com/SuperInstance) fleet.
