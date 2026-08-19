//! fleet-memory — entry point

use clap::Parser;
use fleet_memory::{cli::{Cli, Commands}, db::Database, embed::EmbeddingClient, index::{IndexIdentity, IndexManager}, search::Searcher};
use regex::Regex;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let manager = IndexManager::new(&cli.index_dir);

    match cli.command {
        Commands::Index {
            root,
            force,
            include,
        } => {
            let identity = IndexIdentity {
                provider: cli.provider.clone(),
                model: cli.model.clone(),
                dims: cli.dims,
            };

            tracing::info!("indexing {} as {}", root.display(), identity);

            let embedder = EmbeddingClient::new(&cli.gateway, &cli.model, cli.dims)?;

            let include_re = if let Some(ref pattern) = include {
                Some(Regex::new(pattern)?)
            } else {
                None
            };

            let stats = manager
                .reindex(
                    &root,
                    &identity,
                    &embedder,
                    cli.batch_size,
                    cli.chunk_size,
                    include_re.as_ref(),
                    force,
                )
                .await?;

            println!("\n✅ Reindex complete");
            println!("   Files scanned:    {}", stats.total_files);
            println!("   Files processed:  {}", stats.processed_files);
            println!("   Chunks indexed:   {}", stats.chunks_indexed);
            println!("   Errors:           {}", stats.errors);
            println!("   Index:            {}", manager.index_path(&identity).display());

            // Verify the swap
            let current = manager.resolve_current()?;
            println!("   Current →          {}", current.display());
        }

        Commands::Search {
            query,
            limit,
            threshold,
        } => {
            let current = manager.resolve_current()?;
            tracing::info!("searching against {}", current.display());

            let (db, identity) = Database::open_readonly(&current)?;

            let embedder = EmbeddingClient::new(&cli.gateway, &cli.model, identity.dims)?;
            let query_emb = embedder.embed_one(&query).await?;

            let results = Searcher::search_by_embedding(&db, &query_emb, limit, threshold)?;

            if results.is_empty() {
                println!("No results above threshold {:.2}", threshold);
            } else {
                println!("Found {} results:\n", results.len());
                for (i, result) in results.iter().enumerate() {
                    println!(
                        "─── #{} (score: {:.4}) ───",
                        i + 1,
                        result.score
                    );
                    println!(
                        "📁 {}:{}-{}",
                        result.path,
                        result.start_line.unwrap_or(0),
                        result.end_line.unwrap_or(0)
                    );
                    // Show a snippet
                    let snippet = if result.content.len() > 500 {
                        format!("{}...", &result.content[..500])
                    } else {
                        result.content.clone()
                    };
                    println!("📝 {}", snippet);
                    println!();
                }
            }
        }

        Commands::Status => {
            match manager.resolve_current() {
                Ok(current) => {
                    println!("📂 Index directory:  {}", manager.current_link().display());
                    match Database::open_readonly(&current) {
                        Ok((db, identity)) => {
                            let chunks = db.chunk_count().unwrap_or(0);
                            let created = db.get_meta("created_at").unwrap_or(None);
                            let version = db.get_meta("version").unwrap_or(None);

                            println!("🆔 Provider:         {}", identity.provider);
                            println!("   Model:            {}", identity.model);
                            println!("   Dimensions:       {}", identity.dims);
                            println!("📊 Total chunks:     {}", chunks);
                            println!("📋 Version:          {}", version.unwrap_or("?".into()));
                            if let Some(ts_str) = created {
                                if let Ok(ts) = ts_str.parse::<f64>() {
                                    let dt = chrono::DateTime::from_timestamp(ts as i64, 0);
                                    if let Some(dt) = dt {
                                        println!("🕐 Created:          {}", dt.format("%Y-%m-%d %H:%M:%S UTC"));
                                    }
                                }
                            }
                            println!("💾 Index file:       {}", current.display());
                        }
                        Err(_) => {
                            // New-format hold (index_meta header, §3):
                            // report from the checked header directly.
                            match fleet_memory::query::verify_index_header(&current, None) {
                                Ok(id) => {
                                    let conn = rusqlite::Connection::open(&current).ok();
                                    let (docs, chunks, chunker) = conn
                                        .and_then(|c| {
                                            let d: i64 = c
                                                .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
                                                .unwrap_or(0);
                                            let k: i64 = c
                                                .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
                                                .unwrap_or(0);
                                            let v: String = c
                                                .query_row("SELECT chunker_version FROM index_meta WHERE id = 1", [], |r| r.get(0))
                                                .unwrap_or_else(|_| "?".into());
                                            Some((d, k, v))
                                        })
                                        .unwrap_or((0, 0, "?".to_string()));
                                    println!("🆔 Provider:         {}", id.provider);
                                    println!("   Model:            {}", id.model);
                                    println!("   Dimensions:       {}", id.dims);
                                    println!("📊 Documents:        {}", docs);
                                    println!("   Chunks:           {}", chunks);
                                    println!("🔧 Chunker:          {}", chunker);
                                    println!("💾 Index file:       {}", current.display());
                                }
                                Err(e) => println!("⚠️  Failed to open index: {e}"),
                            }
                        }
                    }
                }
                Err(_) => {
                    println!("📭 No current index found in {}", manager.current_link().display());
                    println!("   Run `fleet-memory index --root <dir>` to create one.");
                }
            }
        }

        Commands::List => {
            let indexes = manager.list_indexes()?;
            if indexes.is_empty() {
                println!("No index files found in index directory.");
            } else {
                println!("Index files:");
                for (path, identity) in &indexes {
                    let is_current = manager
                        .resolve_current()
                        .map(|c| c == *path)
                        .unwrap_or(false);
                    let marker = if is_current { " ← current" } else { "" };
                    println!(
                        "  {} [{} {}]{}",
                        path.file_name().unwrap().to_string_lossy(),
                        identity.provider,
                        identity.dims,
                        marker
                    );
                }
            }
        }

        Commands::Switch { target } => {
            let target_path = PathBuf::from(&target);
            let path = if target_path.is_absolute() {
                target_path
            } else {
                manager.index_dir.join(&target)
            };

            if !path.exists() {
                anyhow::bail!("index file does not exist: {}", path.display());
            }

            manager.swap_current(&path)?;
            println!("✅ Switched current → {}", path.display());
        }

        Commands::Reindex {
            root,
            force,
            include,
            trigger,
        } => {
            let identity = IndexIdentity {
                provider: cli.provider.clone(),
                model: cli.model.clone(),
                dims: cli.dims,
            };

            let embedder = EmbeddingClient::new(&cli.gateway, &cli.model, cli.dims)?;
            let include_re = match include {
                Some(ref pattern) => Some(Regex::new(pattern)?),
                None => None,
            };

            let embed = move |texts: Vec<String>| -> fleet_memory::reindex::EmbedFuture {
                let client = embedder.clone();
                Box::pin(async move {
                    client
                        .embed_batch(&texts)
                        .await
                        .map_err(|e| e.to_string())
                })
            };

            let opts = fleet_memory::reindex::PipelineOpts {
                root: &root,
                index_dir: &manager.index_dir,
                identity: &identity,
                chunk_chars: cli.chunk_size,
                include: include_re,
                force,
                trigger: &trigger,
            };

            let out = fleet_memory::reindex::run_pipeline(opts, &embed).await?;

            println!("\n✅ Pipeline run {} complete", out.run_id);
            println!("   Docs in snapshot:  {}", out.docs_total);
            println!("   Docs processed:    {}", out.docs_done);
            println!("   Chunks written:   {}", out.chunks_written);
            println!("   Batches:           {}", out.batches);
            println!("   Deferred mid-run:  {} (next run picks them up)", out.skipped_midrun + out.invalidated_midrun);
            println!("   Peak RSS:          {}", out
                .peak_rss_bytes
                .map(|b| format!("{b} bytes (recorded in checkpoint — the O(batch) proof)"))
                .unwrap_or_else(|| "n/a".into()));
            let current = manager.resolve_current()?;
            println!("   Current →          {}", current.display());
        }

        Commands::Find { phrase, k } => {
            let registry_path = cli
                .registry
                .clone()
                .unwrap_or_else(|| manager.index_dir.join("fleet-memory.db"));

            let mut report = fleet_memory::query::FindReport {
                tagged: Vec::new(),
                fts: Vec::new(),
                semantic: Vec::new(),
                semantic_skipped: None,
            };

            let reg = if registry_path.exists() {
                match fleet_memory::reindex::Registry::open(&registry_path) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        report.semantic_skipped =
                            Some(format!("registry unreadable ({e})"));
                        None
                    }
                }
            } else {
                report.semantic_skipped =
                    Some(format!("no registry at {}", registry_path.display()));
                None
            };

            if let Some(r) = &reg {
                match fleet_memory::query::find_tagged(r.conn(), &phrase) {
                    Ok(hits) => report.tagged = hits,
                    Err(e) => tracing::warn!("tagged lane failed: {e}"),
                }
                match fleet_memory::query::find_fts(r.conn(), &phrase) {
                    Ok(hits) => report.fts = hits,
                    Err(e) => tracing::warn!("full-text lane failed: {e}"),
                }
            }

            // Semantic lane: current index only, same provider, no fallback.
            match manager.resolve_current() {
                Ok(current) => {
                    match fleet_memory::query::verify_index_header(
                        &current,
                        reg.as_ref().map(|r| r.conn()),
                    ) {
                        Ok(identity) => {
                            let embedder =
                                EmbeddingClient::new(&cli.gateway, &identity.model, identity.dims)?;
                            match embedder.embed_one(&phrase).await {
                                Ok(qv) => {
                                    report.semantic =
                                        fleet_memory::query::find_semantic(&current, &qv, k)?;
                                }
                                Err(e) => {
                                    report.semantic_skipped =
                                        Some(format!("embedding unavailable ({e}) — lanes above are still valid"));
                                }
                            }
                        }
                        Err(e) => {
                            report.semantic_skipped = Some(e.to_string());
                        }
                    }
                }
                Err(_) => {
                    if report.semantic_skipped.is_none() {
                        report.semantic_skipped =
                            Some("no current index — run `reindex` first".into());
                    }
                }
            }

            println!("find {:?}", phrase);
            println!("── tagged lane (work_subjects) ──");
            if report.tagged.is_empty() {
                println!("  (none)");
            }
            for h in &report.tagged {
                println!("  [{:.2}] {} — {} ({})", h.weight, h.slug, h.title, h.kind);
            }
            println!("── full-text lane (work_text_fts) ──");
            if report.fts.is_empty() {
                println!("  (none)");
            }
            for h in &report.fts {
                println!("  {} — {}", h.slug, h.snippet);
            }
            println!("── semantic lane (current index KNN) ──");
            if let Some(reason) = &report.semantic_skipped {
                println!("  skipped: {reason}");
            }
            if report.semantic.is_empty() {
                println!("  (none)");
            }
            for h in &report.semantic {
                println!("  [d={:.4}] {} — {}", h.distance, h.path, excerpt(&h.text, 120));
            }
        }

        Commands::Renders { slug } => {
            let registry_path = cli
                .registry
                .clone()
                .unwrap_or_else(|| manager.index_dir.join("fleet-memory.db"));
            if !registry_path.exists() {
                println!("No registry at {} — nothing rendered yet.", registry_path.display());
                return Ok(());
            }
            let reg = fleet_memory::reindex::Registry::open(&registry_path)?;
            let rows = fleet_memory::query::renders(reg.conn(), &slug)?;
            if rows.is_empty() {
                println!("No renders found for {:?}.", slug);
            } else {
                println!("renders for {:?}:", slug);
                for r in &rows {
                    println!(
                        "  {:<10} v{}  {:<5} {}  {}{}",
                        r.render_kind,
                        r.seq,
                        r.location_kind,
                        r.location,
                        r.renderer.as_deref().unwrap_or("-"),
                        r.duration_ms
                            .map(|d| format!("  ({} ms)", d))
                            .unwrap_or_default()
                    );
                }
            }
        }

        Commands::Decided { date } => {
            let registry_path = cli
                .registry
                .clone()
                .unwrap_or_else(|| manager.index_dir.join("fleet-memory.db"));
            if !registry_path.exists() {
                println!("No registry at {} — no decisions logged yet.", registry_path.display());
                return Ok(());
            }
            let reg = fleet_memory::reindex::Registry::open(&registry_path)?;
            let rows = fleet_memory::query::decided(reg.conn(), &date)?;
            if rows.is_empty() {
                println!("No decisions on {}.", date);
            } else {
                println!("decided on {}:", date);
                for d in &rows {
                    println!("  {}  [{}/{}] {} ({})", d.decided_at, d.agent, d.domain, d.summary, d.status);
                }
            }
        }
    }

    Ok(())
}

/// First line / bounded excerpt of a chunk for terminal display.
fn excerpt(text: &str, max: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max {
        collapsed
    } else {
        let cut: String = collapsed.chars().take(max).collect();
        format!("{cut}…")
    }
}
