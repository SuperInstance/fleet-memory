//! Snapshot manifest — freeze the input set before the first batch.
//!
//! A reindex run FREEZES its input set at start: every candidate file's
//! (path, mtime_ns, size, sha256) is recorded into a manifest file that is
//! written BEFORE the first batch is embedded. This fixes "index changed
//! while building": a file modified mid-run cannot affect the current run —
//! the reader verifies each file against the frozen record before and while
//! reading, and any mismatch defers the file to the NEXT run (it is skipped
//! or invalidated in this one).
//!
//! Memory is O(entry + file being hashed): files are hashed streaming with
//! a fixed 64 KiB buffer; only the (small) entry list is held.

use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Error, Debug)]
pub enum SnapshotError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corpus root does not exist: {0}")]
    NoRoot(PathBuf),
    #[error("corpus root is on /mnt (9P) — all state must live on ext4: {0}")]
    NineP(PathBuf),
    #[error("manifest path is on /mnt (9P) — all state must live on ext4: {0}")]
    ManifestOnNineP(PathBuf),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// One frozen input file. Matches the manifest JSONL schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotEntry {
    /// Path relative to the corpus root, '/'-separated, sorted key.
    pub rel_path: String,
    /// Modification time in nanoseconds since the Unix epoch.
    pub mtime_ns: i64,
    /// File size in bytes at freeze time.
    pub size: u64,
    /// sha256 of the file content at freeze time (hex).
    pub sha256: String,
}

/// A frozen input set: the manifest entries plus bookkeeping.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub root: PathBuf,
    /// Where the manifest was written (ext4 path).
    pub manifest_path: PathBuf,
    /// sha256 of the manifest file itself — recorded in reindex_runs.
    pub hash: String,
    /// Entries sorted by rel_path (the deterministic run/cursor order).
    pub entries: Vec<SnapshotEntry>,
}

/// Extensions that are never treated as text (mirrors the legacy walker).
const BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "mp3", "wav", "mp4", "avi", "mov",
    "webm", "pdf", "zip", "gz", "tar", "bz2", "7z", "exe", "dll", "so", "dylib", "o", "a",
    "class", "jar", "wasm", "db", "sqlite", "bin",
];

/// Hard rule: state never lives under /mnt (9P locking is unreliable).
fn assert_ext4(path: &Path, err_on_mount: fn(PathBuf) -> SnapshotError) -> Result<(), SnapshotError> {
    // Use the literal path (canonicalize needs existence and resolves symlinks
    // that may legitimately cross into the mount we are guarding against).
    let s = path.to_string_lossy();
    if s.starts_with("/mnt/") || s == "/mnt" {
        return Err(err_on_mount(path.to_path_buf()));
    }
    Ok(())
}

/// Streaming sha256 of a file, O(64 KiB) memory.
pub fn sha256_file(path: &Path) -> Result<String, SnapshotError> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Freeze the input set under `root` into a manifest file under
/// `manifest_dir`, named `<run_id>.manifest.jsonl`. Returns the snapshot.
///
/// The manifest is sorted by relative path so the resume cursor ("path
/// order") is deterministic across runs.
pub fn freeze(
    root: &Path,
    include: Option<&Regex>,
    manifest_dir: &Path,
    run_id: &str,
) -> Result<Snapshot, SnapshotError> {
    if !root.is_dir() {
        return Err(SnapshotError::NoRoot(root.to_path_buf()));
    }
    assert_ext4(root, SnapshotError::NineP)?;
    assert_ext4(manifest_dir, SnapshotError::ManifestOnNineP)?;

    let mut entries: Vec<SnapshotEntry> = Vec::new();

    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();

        // Skip hidden files/directories (any '.'-prefixed component).
        let rel = path.strip_prefix(root).unwrap_or(path);
        if rel
            .components()
            .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
        {
            continue;
        }

        // Skip binary-looking extensions.
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if BINARY_EXTENSIONS.contains(&ext.as_str()) {
                continue;
            }
        }

        // Apply the caller's include filter on the relative path.
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if let Some(re) = include {
            if !re.is_match(&rel_str) {
                continue;
            }
        }

        let meta = fs::metadata(path)?;
        let mtime_ns = meta
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        let sha = sha256_file(path)?;

        entries.push(SnapshotEntry {
            rel_path: rel_str,
            mtime_ns,
            size: meta.len(),
            sha256: sha,
        });
    }

    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    fs::create_dir_all(manifest_dir)?;
    let manifest_path = manifest_dir.join(format!("{run_id}.manifest.jsonl"));
    let mut out = File::create(&manifest_path)?;
    for e in &entries {
        serde_json::to_writer(&mut out, e)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;

    let hash = sha256_file(&manifest_path)?;

    Ok(Snapshot {
        root: root.to_path_buf(),
        manifest_path,
        hash,
        entries,
    })
}

/// Load a previously frozen manifest (resume path).
pub fn load_manifest(path: &Path) -> Result<Snapshot, SnapshotError> {
    use std::io::BufRead;
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        entries.push(serde_json::from_str(&line)?);
    }
    let hash = sha256_file(path)?;
    Ok(Snapshot {
        root: PathBuf::new(),
        manifest_path: path.to_path_buf(),
        hash,
        entries,
    })
}

/// Cheap pre-read check: does the file on disk still match the frozen
/// (mtime_ns, size)? A mismatch means "modified mid-run".
pub fn stat_matches(root: &Path, entry: &SnapshotEntry) -> bool {
    let path = root.join(&entry.rel_path);
    match fs::metadata(&path) {
        Ok(meta) => {
            let mtime_ns = meta
                .modified()
                .ok()
                .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            mtime_ns == entry.mtime_ns && meta.len() == entry.size
        }
        Err(_) => false,
    }
}

/// Strong post-read check: does the content still hash to the frozen sha256?
pub fn content_hash(path: &Path, expected: &str) -> bool {
    match sha256_file(path) {
        Ok(actual) => actual == expected,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write(path: &Path, content: &str) {
        use std::io::Write;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    fn find_entry<'a>(snap: &'a Snapshot, rel: &str) -> &'a SnapshotEntry {
        snap.entries.iter().find(|e| e.rel_path == rel).unwrap()
    }

    #[test]
    fn test_freeze_records_identity() {
        let root = tempdir().unwrap();
        let out = tempdir().unwrap();
        write(&root.path().join("a.md"), "alpha content\n");
        write(&root.path().join("b.md"), "beta content\n");

        let snap = freeze(root.path(), None, out.path(), "run01").unwrap();

        assert_eq!(snap.entries.len(), 2);
        // Sorted by rel_path.
        assert_eq!(snap.entries[0].rel_path, "a.md");
        let a = find_entry(&snap, "a.md");
        assert_eq!(a.size, "alpha content\n".len() as u64);
        assert_eq!(a.sha256, sha256_file(&root.path().join("a.md")).unwrap());
        assert!(a.mtime_ns > 0);
        // Manifest written before returning, hash matches its bytes.
        assert!(snap.manifest_path.exists());
        assert_eq!(
            snap.hash,
            sha256_file(&snap.manifest_path).unwrap()
        );
        // Round-trip via load_manifest.
        let loaded = load_manifest(&snap.manifest_path).unwrap();
        assert_eq!(loaded.entries, snap.entries);
        assert_eq!(loaded.hash, snap.hash);
    }

    #[test]
    fn test_midrun_modification_detected() {
        let root = tempdir().unwrap();
        let out = tempdir().unwrap();
        let target = root.path().join("doc.md");
        write(&target, "original content\n");

        let snap = freeze(root.path(), None, out.path(), "run02").unwrap();
        let entry = find_entry(&snap, "doc.md");

        // Unmodified: both checks pass.
        assert!(stat_matches(root.path(), entry));
        assert!(content_hash(&target, &entry.sha256));

        // Modified mid-run: stat and content diverge from the frozen record.
        std::thread::sleep(std::time::Duration::from_millis(10));
        write(&target, "modified content — much longer now\n");
        assert!(!stat_matches(root.path(), entry));
        assert!(!content_hash(&target, &entry.sha256));

        // Deleted mid-run: also a mismatch, never a panic.
        fs::remove_file(&target).unwrap();
        assert!(!stat_matches(root.path(), entry));
        assert!(!content_hash(&target, &entry.sha256));
    }

    #[test]
    fn test_filters_hidden_binary_include() {
        let root = tempdir().unwrap();
        let out = tempdir().unwrap();
        write(&root.path().join("keep.md"), "keep me\n");
        write(&root.path().join(".hidden.md"), "skip me\n");
        write(&root.path().join("sub").join(".hiddendir").join("x.md"), "skip\n");
        write(&root.path().join("photo.png"), "not text\n");

        let snap = freeze(root.path(), None, out.path(), "run03").unwrap();
        let names: Vec<&str> = snap.entries.iter().map(|e| e.rel_path.as_str()).collect();
        assert_eq!(names, vec!["keep.md"]);

        // Include filter keeps only matching files.
        let re = Regex::new(r"\.md$").unwrap();
        write(&root.path().join("notes.txt"), "text but filtered out\n");
        let snap2 = freeze(root.path(), Some(&re), out.path(), "run04").unwrap();
        assert_eq!(snap2.entries.len(), 1);
        assert_eq!(snap2.entries[0].rel_path, "keep.md");
    }

    #[test]
    fn test_unicode_paths() {
        let root = tempdir().unwrap();
        let out = tempdir().unwrap();
        write(&root.path().join("héllo-🌊.md"), "unicode name\n");

        let snap = freeze(root.path(), None, out.path(), "run05").unwrap();
        assert_eq!(snap.entries.len(), 1);
        assert_eq!(snap.entries[0].rel_path, "héllo-🌊.md");
    }

    #[test]
    fn test_missing_root_errors() {
        let out = tempdir().unwrap();
        let err = freeze(Path::new("/nonexistent-root-xyz"), None, out.path(), "r").unwrap_err();
        assert!(matches!(err, SnapshotError::NoRoot(_)));
    }
}
