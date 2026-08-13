//! flock-based index locking — kernel-released on process death.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LockError {
    #[error("index is locked by another process: {0}")]
    Locked(PathBuf),
    #[error("failed to open lock file: {0}")]
    Io(#[from] io::Error),
}

/// RAII guard for an flock on the index's guard file.
/// The lock is automatically released when the guard is dropped
/// (process exit, panic, SIGKILL — the kernel handles it).
pub struct IndexLock {
    _file: File,
    path: PathBuf,
}

impl IndexLock {
    /// Acquire an exclusive flock on `<index_path>.lock`.
    /// Returns immediately if the lock is free; fails if another
    /// live process holds it.
    pub fn acquire(index_path: &Path) -> Result<Self, LockError> {
        let lock_path = index_path.with_extension("db.lock");

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)?;

        // Try non-blocking exclusive lock
        let fd = std::os::unix::io::AsRawFd::as_raw_fd(&file);
        let rc = unsafe { flock(fd, LOCK_EX | LOCK_NB) };

        if rc < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::WouldBlock {
                return Err(LockError::Locked(lock_path));
            }
            return Err(LockError::Io(err));
        }

        // Write our PID for diagnostics (not used for lock logic — flock is the authority)
        let pid = std::process::id();
        let _ = std::fs::write(&lock_path, format!("{pid}\n"));

        Ok(Self {
            _file: file, // held for the lifetime of the guard
            path: lock_path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for IndexLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexLock")
            .field("path", &self.path)
            .finish()
    }
}

impl Drop for IndexLock {
    fn drop(&mut self) {
        // flock is released automatically when the fd closes.
        tracing::debug!("released lock: {}", self.path.display());
    }
}

// libc flock constants
const LOCK_EX: std::os::raw::c_int = 2;
const LOCK_NB: std::os::raw::c_int = 4;

extern "C" {
    fn flock(fd: std::os::raw::c_int, operation: std::os::raw::c_int) -> std::os::raw::c_int;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_acquire_and_release() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        // Acquire
        let lock = IndexLock::acquire(&path).unwrap();
        assert!(lock.path().exists());

        // Release (drop)
        drop(lock);

        // Should be able to acquire again
        let lock2 = IndexLock::acquire(&path).unwrap();
        drop(lock2);
    }

    #[test]
    fn test_contention() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("contention.db");

        let lock1 = IndexLock::acquire(&path).unwrap();

        // Second lock should fail while first is held
        let result = IndexLock::acquire(&path);
        assert!(result.is_err());
        match result.unwrap_err() {
            LockError::Locked(_) => {}
            other => panic!("expected Locked, got {other:?}"),
        }

        drop(lock1);

        // Now it should work
        let lock2 = IndexLock::acquire(&path).unwrap();
        drop(lock2);
    }
}
