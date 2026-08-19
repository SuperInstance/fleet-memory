//! The three reference queries (schema §4) as library calls + CLI wiring.
//!
//! Q1 `find <phrase>`  — cheap bearings before expensive ones: the tagged
//!                      lane (work_subjects) and the full-text lane
//!                      (work_text_fts) run in the registry; the semantic
//!                      lane (KNN in the current index) runs only if a
//!                      current index exists and the phrase can be embedded
//!                      with the SAME provider recorded in index_meta —
//!                      never a fallback.
//! Q2 `renders <slug>` — every materialization of a work.
//! Q3 `decided <date>` — the decision log for one UTC day.
//!
//! On open, the index header is verified: index_meta == filename == the
//! registry's current row. Any mismatch is a hard error, never a silent
//! query (§3).

use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use rusqlite::{params, Connection};
use thiserror::Error;

use crate::index::IndexIdentity;

#[derive(Error, Debug)]
pub enum QueryError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("registry unavailable: {0}")]
    NoRegistry(PathBuf),
    #[error("index header mismatch: {0}")]
    HeaderMismatch(String),
    #[error("bad date (want YYYY-MM-DD): {0}")]
    BadDate(String),
}

// ---------------------------------------------------------------------------
// Q1 — find
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct TaggedHit {
    pub slug: String,
    pub title: String,
    pub kind: String,
    pub weight: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FtsHit {
    pub slug: String,
    pub title: String,
    pub snippet: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SemanticHit {
    pub path: String,
    pub text: String,
    pub distance: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FindReport {
    pub tagged: Vec<TaggedHit>,
    pub fts: Vec<FtsHit>,
    pub semantic: Vec<SemanticHit>,
    /// Set when the semantic lane could not run (no index / no embedding).
    pub semantic_skipped: Option<String>,
}

/// Q1a — tagged lane (verbatim structure from §4).
pub fn find_tagged(registry: &Connection, phrase: &str) -> Result<Vec<TaggedHit>, QueryError> {
    let mut stmt = registry.prepare(
        "SELECT DISTINCT w.slug, w.title, w.kind, s.weight
         FROM work_subjects s
         JOIN creative_works w USING (work_id)
         WHERE s.subject = ?1
         ORDER BY s.weight DESC",
    )?;
    let rows = stmt
        .query_map(params![phrase], |row| {
            Ok(TaggedHit {
                slug: row.get(0)?,
                title: row.get(1)?,
                kind: row.get(2)?,
                weight: row.get(3)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Q1b — full-text lane (verbatim structure; phrase is quoted so user text
/// is never parsed as FTS query syntax).
pub fn find_fts(registry: &Connection, phrase: &str) -> Result<Vec<FtsHit>, QueryError> {
    let quoted = format!("\"{}\"", phrase.replace('"', "\"\""));
    let mut stmt = registry.prepare(
        "SELECT w.slug, w.title, snippet(work_text_fts, 1, '«', '»', '…', 12)
         FROM work_text_fts
         JOIN creative_works w ON w.work_id = work_text_fts.work_id
         WHERE work_text_fts MATCH ?1
         ORDER BY rank",
    )?;
    let rows = stmt
        .query_map(params![quoted], |row| {
            Ok(FtsHit {
                slug: row.get(0)?,
                title: row.get(1)?,
                snippet: row.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Q1c — semantic lane. Same query shape as §4; the ATTACH is replaced by a
/// direct read-only open of the current index (all joined tables live in the
/// index file, so the result is identical with fewer moving parts).
/// `query_vector` must have been embedded by the provider in index_meta.
pub fn find_semantic(
    index_path: &Path,
    query_vector: &[f32],
    k: usize,
) -> Result<Vec<SemanticHit>, QueryError> {
    crate::db::ensure_vec_registered();
    let conn = Connection::open(index_path)?;
    conn.pragma_update(None, "query_only", true)?;
    conn.pragma_update(None, "busy_timeout", 5000)?;

    let emb = crate::db::embedding_to_bytes(query_vector);
    let mut stmt = conn.prepare(
        "SELECT d.path, c.text, v.distance
         FROM vec_chunks v
         JOIN chunks c   ON c.chunk_id = v.rowid
         JOIN documents d ON d.doc_id  = c.doc_id
         WHERE v.embedding MATCH ?1 AND k = ?2
         ORDER BY v.distance",
    )?;
    let rows = stmt
        .query_map(params![emb, k as i64], |row| {
            Ok(SemanticHit {
                path: row.get(0)?,
                text: row.get(1)?,
                distance: row.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// §3 header verification: index_meta must agree with the filename, and —
/// when the registry knows a current index — with the registry row too.
/// Returns the identity from the header.
pub fn verify_index_header(
    index_path: &Path,
    registry: Option<&Connection>,
) -> Result<IndexIdentity, QueryError> {
    crate::db::ensure_vec_registered();
    let conn = Connection::open(index_path)?;

    let (provider_id, model, dims): (String, String, i64) = conn
        .query_row(
            "SELECT provider_id, model, dims FROM index_meta WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .map_err(|_| QueryError::HeaderMismatch(format!("{}: no index_meta header", index_path.display())))?;

    // Filename carries provenance: index.<provider>.<model>.<dims>.db.
    // (provider_id in the header is the composite "<provider>/<model>";
    // the filename uses just the provider part — slashes are not fs-safe.)
    let provider_part = provider_id.split('/').next().unwrap_or(&provider_id);
    let fname = index_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let expect = format!("index.{provider_part}.{model}.{dims}.db");
    if fname != expect {
        return Err(QueryError::HeaderMismatch(format!(
            "{}: filename says {fname}, header says {expect}",
            index_path.display()
        )));
    }

    let identity = IndexIdentity {
        // provider_id in the header is "<provider>/<model>"; split it back.
        provider: provider_id
            .split('/')
            .next()
            .unwrap_or(&provider_id)
            .to_string(),
        model,
        dims: dims as usize,
    };

    // Cross-check the registry's current row when available.
    if let Some(reg) = registry {
        let current: Option<String> = reg
            .query_row(
                "SELECT index_name FROM index_registry WHERE is_current = 1",
                [],
                |r| r.get(0),
            )
            .ok();
        if let Some(name) = current {
            let stem = fname.trim_end_matches(".db").to_string();
            if name != stem {
                return Err(QueryError::HeaderMismatch(format!(
                    "registry says current is {name}, but serving {}",
                    index_path.display()
                )));
            }
        }
    }

    Ok(identity)
}

// ---------------------------------------------------------------------------
// Q2 — renders
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct RenderRow {
    pub render_kind: String,
    pub seq: i64,
    pub location_kind: String,
    pub location: String,
    pub duration_ms: Option<i64>,
    pub size_bytes: Option<i64>,
    pub renderer: Option<String>,
    pub created_at: String,
}

/// Q2 — all renders for a work, by slug or fuzzy title (verbatim structure;
/// the slug parameter also feeds the title LIKE pattern, generalizing the
/// reference's literal '%pfd%speech%').
pub fn renders(registry: &Connection, slug: &str) -> Result<Vec<RenderRow>, QueryError> {
    let pattern = format!("%{}%", slug.to_lowercase().replace(['-', '_'], "%"));
    let mut stmt = registry.prepare(
        "SELECT r.render_kind, r.seq, r.location_kind, r.location,
                r.duration_ms, r.size_bytes, r.renderer, r.created_at
         FROM work_renders r
         JOIN creative_works w USING (work_id)
         WHERE w.slug = ?1
            OR lower(w.title) LIKE ?2
         ORDER BY r.render_kind, r.seq",
    )?;
    let rows = stmt
        .query_map(params![slug, pattern], |row| {
            Ok(RenderRow {
                render_kind: row.get(0)?,
                seq: row.get(1)?,
                location_kind: row.get(2)?,
                location: row.get(3)?,
                duration_ms: row.get(4)?,
                size_bytes: row.get(5)?,
                renderer: row.get(6)?,
                created_at: row.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Q3 — decided
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct DecisionRow {
    pub decided_at: String,
    pub agent: String,
    pub domain: String,
    pub summary: String,
    pub status: String,
}

/// Q3 — decisions for one UTC day (verbatim structure; the day boundary is
/// computed in Rust so the parameter stays a plain date).
pub fn decided(registry: &Connection, date: &str) -> Result<Vec<DecisionRow>, QueryError> {
    let day = NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|_| QueryError::BadDate(date.to_string()))?;
    let next = day + chrono::Duration::days(1);
    let (from, to) = (day.format("%Y-%m-%d").to_string(), next.format("%Y-%m-%d").to_string());

    let mut stmt = registry.prepare(
        "SELECT decided_at, agent, domain, summary, status
         FROM agent_decisions
         WHERE decided_at >= ?1 AND decided_at < ?2
         ORDER BY decided_at",
    )?;
    let rows = stmt
        .query_map(params![from, to], |row| {
            Ok(DecisionRow {
                decided_at: row.get(0)?,
                agent: row.get(1)?,
                domain: row.get(2)?,
                summary: row.get(3)?,
                status: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reindex::Registry;
    use tempfile::tempdir;

    /// Fixture registry with the schema's own examples: a PFD speech with
    /// renders and a 'silence' subject, plus decisions on Aug 13/14 2026.
    fn fixture() -> (tempfile::TempDir, Registry) {
        let dir = tempdir().unwrap();
        let reg = Registry::open(&dir.path().join("fleet-memory.db")).unwrap();
        let c = reg.conn();
        c.execute(
            "INSERT INTO creative_works (work_id, slug, title, kind, status, created_by, synopsis)
             VALUES ('w1', 'pfd-speech', 'PFD Speech', 'speech', 'rendered', 'navigation/kimi',
                     'a speech about silence at sea')",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO creative_works (work_id, slug, title, kind, status, created_by)
             VALUES ('w2', 'harbor-nights', 'Harbor Nights', 'poem', 'draft', 'ops/opus')",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO work_subjects (work_id, subject, weight) VALUES ('w1', 'silence', 1.0)",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO work_subjects (work_id, subject, weight) VALUES ('w2', 'silence', 0.4)",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO work_renders (render_id, work_id, render_kind, seq, location_kind, location, renderer)
             VALUES ('r1', 'w1', 'outline', 1, 'ext4', '/works/pfd/outline.md', 'kimi')",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO work_renders (render_id, work_id, render_kind, seq, location_kind, location, renderer, duration_ms)
             VALUES ('r2', 'w1', 'tts-audio', 1, 'r2', 'r2://works/pfd/v1.wav', 'sag', 91000)",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO work_renders (render_id, work_id, render_kind, seq, location_kind, location, renderer)
             VALUES ('r3', 'w1', 'text', 2, 'ext4', '/works/pfd/draft2.md', 'kimi')",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO work_text_fts (title, body, work_id, render_id)
             VALUES ('PFD Speech', 'the engine falls to silence and the fog settles over the fishing fleet', 'w1', 'r1')",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO agent_decisions (decision_id, decided_at, agent, domain, summary)
             VALUES ('d1', '2026-08-13T10:15:00Z', 'navigation/kimi', 'memory', 'fleet-memory schema adopted verbatim')",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO agent_decisions (decision_id, decided_at, agent, domain, summary)
             VALUES ('d2', '2026-08-13T18:40:00Z', 'ops/opus', 'infra', 'flock over PID files')",
            [],
        )
        .unwrap();
        c.execute(
            "INSERT INTO agent_decisions (decision_id, decided_at, agent, domain, summary)
             VALUES ('d3', '2026-08-14T09:00:00Z', 'ops/opus', 'infra', 'next-day decision must not leak in')",
            [],
        )
        .unwrap();
        (dir, reg)
    }

    #[test]
    fn test_q1a_tagged_lane() {
        let (_d, reg) = fixture();
        let hits = find_tagged(reg.conn(), "silence").unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].slug, "pfd-speech"); // weight 1.0 first
        assert_eq!(hits[1].slug, "harbor-nights");
        // Unknown subject: empty, not an error.
        assert!(find_tagged(reg.conn(), "fishing").unwrap().is_empty());
    }

    #[test]
    fn test_q1b_fts_lane() {
        let (_d, reg) = fixture();
        let hits = find_fts(reg.conn(), "silence").unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].slug, "pfd-speech");
        assert!(hits[0].snippet.contains('«'), "snippet markers present");

        // User punctuation must not break FTS syntax.
        assert!(find_fts(reg.conn(), "a*b \"quoted\"").is_ok());
        // A word absent from the corpus: empty, not an error.
        assert!(find_fts(reg.conn(), "zeppelin").unwrap().is_empty());
    }

    #[test]
    fn test_q1c_semantic_lane() {
        // Build a real index file through the pipeline helpers.
        let dir = tempdir().unwrap();
        let id = IndexIdentity { provider: "test".into(), model: "fake".into(), dims: 4 };
        let index = crate::reindex::IndexFile::open(&dir.path().join(id.filename()), &id).unwrap();
        let mut conn = index.into_conn();
        {
            let tx = conn.transaction().unwrap();
            let doc = crate::reindex::IndexFile::upsert_doc(&tx, "notes/fishing.md", "aa", 1, 2).unwrap();
            for (i, text) in ["gone fishing at dawn", "silence over the water"].iter().enumerate() {
                let mut spec = crate::chunker::chunk_text(text, 200).unwrap().remove(0);
                spec.seq = i as u32; // two chunks of one doc: 0, 1
                let emb = if i == 0 { vec![1.0f32, 0.0, 0.0, 0.0] } else { vec![0.0f32, 1.0, 0.0, 0.0] };
                crate::reindex::IndexFile::insert_chunk(&tx, doc, &spec, &emb).unwrap();
            }
            tx.commit().unwrap();
        }

        // Header verifies and round-trips identity.
        let ident = verify_index_header(&dir.path().join(id.filename()), None).unwrap();
        assert_eq!(ident.provider, "test");
        assert_eq!(ident.model, "fake");
        assert_eq!(ident.dims, 4);

        // Query near the fishing chunk returns it first.
        let hits = find_semantic(&dir.path().join(id.filename()), &[0.9, 0.1, 0.0, 0.0], 10).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].path.contains("fishing"));
        assert!(hits[0].text.contains("fishing"));
    }

    #[test]
    fn test_header_mismatch_is_hard_error() {
        let dir = tempdir().unwrap();
        let id = IndexIdentity { provider: "test".into(), model: "fake".into(), dims: 4 };
        let path = dir.path().join(id.filename());
        crate::reindex::IndexFile::open(&path, &id).unwrap();
        // Rename the file: filename no longer matches the header.
        let renamed = dir.path().join("index.someone.else.999.db");
        std::fs::rename(&path, &renamed).unwrap();
        assert!(matches!(
            verify_index_header(&renamed, None),
            Err(QueryError::HeaderMismatch(_))
        ));
    }

    #[test]
    fn test_q2_renders() {
        let (_d, reg) = fixture();
        let rows = renders(reg.conn(), "pfd-speech").unwrap();
        assert_eq!(rows.len(), 3);
        // Ordered by (render_kind, seq).
        assert_eq!(rows[0].render_kind, "outline");
        assert_eq!(rows[1].render_kind, "text");
        assert_eq!(rows[2].render_kind, "tts-audio");
        assert_eq!(rows[2].location, "r2://works/pfd/v1.wav");
        assert_eq!(rows[2].duration_ms, Some(91000));

        // Fuzzy title match (schema's LIKE lane).
        let by_title = renders(reg.conn(), "pfd-speech-aka").unwrap();
        // slug exact fails, but title LIKE '%pfd%speech%aka%' fails too — 0.
        assert!(by_title.is_empty());
        let fuzzy = renders(reg.conn(), "pfd speech").unwrap();
        assert_eq!(fuzzy.len(), 3, "hyphen/space slug should still match via title lane");
    }

    #[test]
    fn test_q3_decided() {
        let (_d, reg) = fixture();
        let rows = decided(reg.conn(), "2026-08-13").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].agent, "navigation/kimi");
        assert_eq!(rows[1].agent, "ops/opus");
        // Day boundary: the 14th does not leak in.
        let next = decided(reg.conn(), "2026-08-14").unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].agent, "ops/opus");

        assert!(matches!(
            decided(reg.conn(), "not-a-date"),
            Err(QueryError::BadDate(_))
        ));
    }
}
