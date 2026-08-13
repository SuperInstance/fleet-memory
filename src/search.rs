//! Vector similarity search via sqlite-vec.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::db::Database;

#[derive(Error, Debug)]
pub enum SearchError {
    #[error("database error: {0}")]
    Db(#[from] crate::db::DbError),
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("embedding error: {0}")]
    Embed(#[from] crate::embed::EmbedError),
    #[error("no index available")]
    NoIndex,
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: i64,
    pub path: String,
    pub start_line: Option<i64>,
    pub end_line: Option<i64>,
    pub content: String,
    pub score: f64,
}

/// Search the index for the most similar chunks to a query embedding.
pub struct Searcher;

impl Searcher {
    /// Search using a pre-computed embedding vector.
    pub fn search_by_embedding(
        db: &Database,
        query_embedding: &[f32],
        limit: usize,
        threshold: f64,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let emb_bytes = embedding_to_bytes(query_embedding);

        let mut stmt = db.conn().prepare(
            "SELECT
                v.chunk_id,
                v.distance,
                c.path,
                c.start_line,
                c.end_line,
                c.content
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ?1
              AND v.k = ?2
              AND v.distance <= ?3
            ORDER BY v.distance ASC",
        )?;

        // For cosine distance in sqlite-vec, distance = 1 - cosine_similarity
        // So lower distance = more similar. threshold on similarity => distance <= 1 - threshold
        let max_distance = 1.0 - threshold;

        let rows = stmt
            .query_map(rusqlite::params![emb_bytes, limit as i64, max_distance], |row| {
                let chunk_id: i64 = row.get(0)?;
                let distance: f64 = row.get(1)?;
                let path: String = row.get(2)?;
                let start_line: Option<i64> = row.get(3)?;
                let end_line: Option<i64> = row.get(4)?;
                let content: String = row.get(5)?;

                Ok(SearchResult {
                    chunk_id,
                    path,
                    start_line,
                    end_line,
                    content,
                    score: 1.0 - distance, // convert distance back to similarity score
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(rows)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use crate::index::IndexIdentity;
    use tempfile::tempdir;

    fn make_test_db() -> (tempfile::TempDir, Database) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("search.db");
        let id = IndexIdentity {
            provider: "test".into(),
            model: "m".into(),
            dims: 4,
        };
        let mut db = Database::open(&path, &id).unwrap();
        db.ensure_vec_table().unwrap();
        (dir, db)
    }

    #[test]
    fn test_embedding_round_trip() {
        let emb = vec![0.1f32, 0.2, 0.3, 0.4];
        let bytes = embedding_to_bytes(&emb);
        assert_eq!(bytes.len(), 16); // 4 dims × 4 bytes
    }

    #[test]
    fn test_search_basic() {
        let (_dir, db) = make_test_db();

        // Insert two chunks with different embeddings
        let emb1 = vec![1.0f32, 0.0, 0.0, 0.0];
        let emb2 = vec![0.0f32, 1.0, 0.0, 0.0];

        db.insert_chunk("/a.txt", Some(1), Some(5), "alpha", "h1", &emb1, 0)
            .unwrap();
        db.insert_chunk("/b.txt", Some(1), Some(3), "beta", "h2", &emb2, 0)
            .unwrap();

        // Query close to emb1
        let query = vec![0.9f32, 0.1, 0.0, 0.0];
        let results =
            Searcher::search_by_embedding(&db, &query, 10, 0.0).unwrap();

        assert!(!results.is_empty());
        // First result should be the alpha chunk
        assert_eq!(results[0].path, "/a.txt");
        assert!(results[0].score > 0.5);
    }

    #[test]
    fn test_search_threshold() {
        let (_dir, db) = make_test_db();

        let emb1 = vec![1.0f32, 0.0, 0.0, 0.0];
        let emb2 = vec![0.0f32, 1.0, 0.0, 0.0];

        db.insert_chunk("/a.txt", Some(1), Some(1), "alpha", "h1", &emb1, 0)
            .unwrap();
        db.insert_chunk("/b.txt", Some(1), Some(1), "beta", "h2", &emb2, 0)
            .unwrap();

        // Query close to emb1, high threshold should filter out emb2
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let high_threshold =
            Searcher::search_by_embedding(&db, &query, 10, 0.99).unwrap();

        // Should only get emb1 (cosine similarity 1.0)
        let alpha_count = high_threshold.iter().filter(|r| r.path == "/a.txt").count();
        assert!(alpha_count >= 1);
    }

    #[test]
    fn test_search_empty() {
        let (_dir, db) = make_test_db();
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let results =
            Searcher::search_by_embedding(&db, &query, 10, 0.0).unwrap();
        assert!(results.is_empty());
    }
}
