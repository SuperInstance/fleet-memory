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
                        Err(e) => {
                            println!("⚠️  Failed to open index: {}", e);
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
    }

    Ok(())
}
