//! The canonical byte encoder (spec 5.3, Appendix B.6).
//!
//! [`CanonWriter`] is the only way a value becomes hash input: little-endian integers,
//! length-prefixed strings and byte strings, `f64` as IEEE bits with `-0.0` normalized and
//! `NaN` rejected. Maps must be iterated in sorted key order — use `BTreeMap`, which is sorted
//! by construction (`HashMap` is banned by spec 3.4 and by clippy).

use crate::codes;
use crate::diag::Diagnostic;

/// Canonical byte encoder. Writes never fail; an unencodable value latches an error that
/// [`CanonWriter::finish`] returns, so node encoders stay infallible.
#[derive(Debug, Default)]
pub struct CanonWriter {
    buf: Vec<u8>,
    err: Option<Diagnostic>,
}

impl CanonWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Length-prefixed UTF-8.
    pub fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }

    /// Length-prefixed bytes.
    pub fn bytes(&mut self, b: &[u8]) {
        self.seq(b.len());
        self.buf.extend_from_slice(b);
    }

    /// A raw 32-byte digest: fixed width, so no length prefix.
    pub fn digest(&mut self, d: &[u8; 32]) {
        self.buf.extend_from_slice(d);
    }

    /// Element count of a sequence or map. Every variable-length item is prefixed with one.
    pub fn seq(&mut self, len: usize) {
        self.u32(u32::try_from(len).unwrap_or(u32::MAX));
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }

    /// `f32` is widened: the two are then interchangeable in a hash, which is what a spec
    /// field switching precision should mean.
    pub fn f32(&mut self, v: f32) {
        self.f64(f64::from(v));
    }

    /// IEEE bits, with `-0.0` normalized to `0.0` (they compare equal, so they must hash
    /// equal) and `NaN` rejected (it compares unequal to itself, so no hash of it is sound).
    pub fn f64(&mut self, v: f64) {
        if v.is_nan() {
            self.fail(Diagnostic::new(
                codes::HASH_001,
                "NaN cannot be encoded canonically",
            ));
            return;
        }
        let v = if v == 0.0 { 0.0 } else { v };
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    /// Latches a diagnostic; the first one wins.
    pub fn fail(&mut self, diag: Diagnostic) {
        if self.err.is_none() {
            self.err = Some(diag);
        }
    }

    pub fn finish(self) -> Result<Vec<u8>, Diagnostic> {
        match self.err {
            Some(e) => Err(e),
            None => Ok(self.buf),
        }
    }

    pub fn hash(self) -> Result<[u8; 32], Diagnostic> {
        Ok(*blake3::hash(&self.finish()?).as_bytes())
    }
}
