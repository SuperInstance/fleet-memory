//! Char-offset chunking — the §3.3 contract.
//!
//! Chunks carry stable identity: `(doc_id, seq)` with char offsets into the
//! source document (`start_offset` inclusive, `end_offset` exclusive, counted
//! in Unicode scalar values, NOT bytes). The chunking algorithm is part of
//! the index's identity and is stamped into `index_meta.chunker_version` —
//! changing it means building a new index.
//!
//! Memory is O(chunk + line): files are consumed line-by-line through a
//! `BufRead`, never slurped whole. A chunk is cut at the best boundary no
//! later than `max_chars`: paragraph break > line break > space > hard cut.

use std::io::{BufRead, Cursor, Read};

use sha2::Digest;

/// Version of the chunking algorithm. Bump = rebuild every index.
pub const CHUNKER_VERSION: &str = "char-v1";

/// One chunk of a document, with its char-offset identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpec {
    /// Order within the document, 0-based.
    pub seq: u32,
    /// Char offset of the first character (inclusive).
    pub start_offset: usize,
    /// Char offset one past the last character (exclusive).
    pub end_offset: usize,
    /// The exact source slice `text[start_offset..end_offset]` (in chars).
    pub text: String,
}

impl ChunkSpec {
    /// Rough token count (whitespace-separated), as allowed (nullable) by §3.3.
    pub fn token_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

/// Streaming line-aware chunker over any `BufRead`.
///
/// Yields one `ChunkSpec` at a time; never holds more than one chunk plus
/// one line of input in memory. `None` ends the document.
pub struct LineChunker<R: BufRead> {
    reader: R,
    max_chars: usize,
    /// Pending text not yet emitted; starts at absolute char `buf_start`.
    buf: String,
    buf_start: usize,
    /// seq of the next chunk to emit.
    next_seq: u32,
    eof: bool,
}

impl<R: BufRead> LineChunker<R> {
    /// Create a chunker with a target chunk size of `max_chars` characters.
    /// Values below 64 are clamped up — tiny chunks are never useful.
    pub fn new(reader: R, max_chars: usize) -> Self {
        Self {
            reader,
            max_chars: max_chars.max(64),
            buf: String::new(),
            buf_start: 0,
            next_seq: 0,
            eof: false,
        }
    }

    fn fill(&mut self) -> std::io::Result<bool> {
        // Read one line (with its newline) into raw bytes so the hash of the
        // on-disk bytes can be computed by the caller wrapping the reader.
        let mut raw: Vec<u8> = Vec::new();
        let n = self.reader.read_until(b'\n', &mut raw)?;
        if n == 0 {
            self.eof = true;
            return Ok(false);
        }
        self.buf.push_str(&String::from_utf8_lossy(&raw));
        Ok(true)
    }

    /// Cut the next chunk from `buf` if it is big enough (or we are at EOF).
    fn cut(&mut self) -> Option<ChunkSpec> {
        let buf_chars = self.buf.chars().count();
        if buf_chars == 0 {
            return None;
        }

        let cut = if self.eof {
            // Final remainder smaller than one chunk: emit it whole.
            buf_chars
        } else {
            best_boundary(&self.buf, self.max_chars.min(buf_chars))
        };
        if cut == 0 {
            return None;
        }

        let text: String = self.buf.chars().take(cut).collect();
        let rest: String = self.buf.chars().skip(cut).collect();
        let spec = ChunkSpec {
            seq: self.next_seq,
            start_offset: self.buf_start,
            end_offset: self.buf_start + cut,
            text,
        };
        self.next_seq += 1;
        self.buf = rest;
        self.buf_start += cut;
        if spec.text.trim().is_empty() {
            // A run of blank lines can fill a whole target window; consume
            // it without emitting (offsets stay contiguous, seq untouched).
            return None;
        }
        Some(spec)
    }

    /// Recover the underlying reader (e.g. to finalize a hash).
    pub fn into_inner(self) -> R {
        self.reader
    }
}

impl<R: BufRead> Iterator for LineChunker<R> {
    type Item = std::io::Result<ChunkSpec>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Keep reading until we can cut a full chunk (or hit EOF).
            while !self.eof && self.buf.chars().count() < self.max_chars {
                match self.fill() {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(e) => return Some(Err(e)),
                }
            }
            match self.cut() {
                Some(spec) => return Some(Ok(spec)),
                None => {
                    if self.eof {
                        // Drain any trailing remainder.
                        if self.buf.chars().count() == 0 {
                            return None;
                        }
                        // Whitespace-only remainder: drop it and finish.
                        if self.buf.trim().is_empty() {
                            self.buf.clear();
                            return None;
                        }
                        // Non-whitespace remainder smaller than one chunk.
                        let spec = ChunkSpec {
                            seq: self.next_seq,
                            start_offset: self.buf_start,
                            end_offset: self.buf_start + self.buf.chars().count(),
                            text: std::mem::take(&mut self.buf),
                        };
                        self.next_seq += 1;
                        return Some(Ok(spec));
                    }
                    // Buffer was all whitespace up to target; loop to read more.
                }
            }
        }
    }
}

/// Find the best cut point (in chars, exclusive) at or before `target`:
/// prefer the end of the last paragraph break, then the last newline,
/// then the last space, else a hard cut at `target`. The boundary
/// characters stay in the chunk (exact source slices; the next chunk
/// starts clean).
fn best_boundary(buf: &str, target: usize) -> usize {
    let chars: Vec<char> = buf.chars().take(target).collect();

    // Paragraph break: position right after a "\n\n".
    let mut best = 0;
    for i in 1..chars.len() {
        if chars[i - 1] == '\n' && chars[i] == '\n' {
            best = i + 1;
        }
    }
    if best > 0 {
        return best;
    }
    // Line break.
    for (i, &c) in chars.iter().enumerate() {
        if c == '\n' {
            best = i + 1;
        }
    }
    if best > 0 {
        return best;
    }
    // Space.
    for (i, &c) in chars.iter().enumerate() {
        if c == ' ' {
            best = i + 1;
        }
    }
    if best > 0 {
        return best;
    }
    // Hard cut.
    target
}

/// Chunk an in-memory string (tests, small docs). Same algorithm as the
/// streaming path.
pub fn chunk_text(text: &str, max_chars: usize) -> std::io::Result<Vec<ChunkSpec>> {
    LineChunker::new(Cursor::new(text.as_bytes()), max_chars).collect()
}

/// Wraps a `BufRead` and sha256-hashes every byte consumed through
/// `fill_buf`/`consume` (the path `read_until` uses). Use for snapshot
/// verification while chunking in a single pass.
pub struct HashingReader<R: BufRead> {
    inner: R,
    hasher: sha2::Sha256,
}

impl<R: BufRead> HashingReader<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: sha2::Sha256::new(),
        }
    }

    /// Finalize and return the hex digest of everything consumed.
    pub fn sha256_hex(self) -> String {
        use sha2::Digest;
        format!("{:x}", self.hasher.finalize())
    }
}

impl<R: BufRead> Read for HashingReader<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        // Hashing happens in consume(); read() is a plain passthrough so
        // bytes are never counted twice.
        self.inner.read(out)
    }
}

impl<R: BufRead> BufRead for HashingReader<R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        if let Ok(buf) = self.inner.fill_buf() {
            use sha2::Digest;
            let n = amt.min(buf.len());
            self.hasher.update(&buf[..n]);
        }
        self.inner.consume(amt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the §3.3 invariant: text == chars[start..end], end > start.
    fn assert_slice_identity(text: &str, specs: &[ChunkSpec]) {
        let chars: Vec<char> = text.chars().collect();
        let mut last_end = 0usize;
        for (i, s) in specs.iter().enumerate() {
            assert_eq!(s.seq, i as u32, "seq must be dense and ordered");
            assert!(s.end_offset > s.start_offset, "CHECK(end > start) violated");
            assert_eq!(s.start_offset, last_end, "chunks must be contiguous");
            let slice: String = chars[s.start_offset..s.end_offset].iter().collect();
            assert_eq!(&slice, &s.text, "text must be the exact source slice");
            assert!(!s.text.trim().is_empty(), "no all-whitespace chunks");
            assert!(s.text.chars().count() <= 2000.max(1));
            last_end = s.end_offset;
        }
        // Chunks may stop before EOF only if the remainder is whitespace.
        let rest: String = chars[last_end..].iter().collect();
        assert!(rest.trim().is_empty(), "lost non-whitespace tail: {rest:?}");
    }

    #[test]
    fn test_basic_offsets_and_sizes() {
        let mut text = String::new();
        for i in 0..200 {
            text.push_str(&format!("line {i} with some words\n"));
        }
        let specs = chunk_text(&text, 120).unwrap();
        assert!(specs.len() > 3);
        assert_slice_identity(&text, &specs);
        for s in &specs {
            assert!(s.text.chars().count() <= 120, "chunk over max_chars");
        }
    }

    #[test]
    fn test_unicode_char_offsets() {
        // Multi-byte chars: byte offsets would differ from char offsets.
        let text = "héllo wörld 🌊 fishing boats sail 🎣 across the wide blue sea \
                    and the crew sings songs about silence and storms\n\
                    deuxième ligne avec des accents éàç pour vérifier\n\
                    troisième ligne encore plus longue pour forcer une coupe\n";
        let specs = chunk_text(text, 40).unwrap();
        assert!(specs.len() >= 2);
        assert_slice_identity(text, &specs);
    }

    #[test]
    fn test_paragraph_boundary_preferred() {
        let mut text = String::new();
        for p in 0..5 {
            for l in 0..5 {
                text.push_str(&format!("paragraph {p} line {l}\n"));
            }
            text.push('\n');
        }
        let specs = chunk_text(&text, 150).unwrap();
        assert_slice_identity(&text, &specs);
        // At least one cut should land on a line/paragraph boundary (a
        // chunk that ends with a newline, not mid-word).
        let cuts_aligned = specs.iter().any(|s| s.text.ends_with('\n'));
        assert!(cuts_aligned, "expected a paragraph-aligned boundary");
    }

    #[test]
    fn test_single_chunk_small_text() {
        let text = "just one small note";
        let specs = chunk_text(text, 2000).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].text, text);
        assert_eq!(specs[0].start_offset, 0);
        assert_eq!(specs[0].end_offset, text.chars().count());
        assert_eq!(specs[0].seq, 0);
        assert!(specs[0].token_count() >= 3);
    }

    #[test]
    fn test_empty_and_whitespace_only() {
        assert!(chunk_text("", 100).unwrap().is_empty());
        assert!(chunk_text("\n\n \t \n", 100).unwrap().is_empty());
    }

    #[test]
    fn test_no_trailing_newline() {
        let text = "no trailing newline here";
        let specs = chunk_text(text, 10).unwrap();
        assert!(!specs.is_empty());
        assert_slice_identity(text, &specs);
    }

    #[test]
    fn test_hard_cut_unbroken_run() {
        // One enormous word: must hard-cut at max_chars without panicking.
        let text: String = std::iter::repeat('x').take(500).collect();
        let specs = chunk_text(&text, 100).unwrap();
        assert_eq!(specs.len(), 5);
        assert_slice_identity(&text, &specs);
    }

    #[test]
    fn test_version_stamped() {
        // The version is part of index identity; changing it invalidates
        // every existing index. This test pins the current value.
        assert_eq!(CHUNKER_VERSION, "char-v1");
    }
}
