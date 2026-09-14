//! Hand-rolled little-endian CDR (OMG DDS-XTypes 1.3 clause 7.6.3.1.2, `PLAIN_CDR`) writer and
//! reader for the fixed message subset in [`crate::msg`]. No `rosidl`, no Fast-CDR binding:
//! rules and golden bytes are pinned in `docs/api-notes/ros2-cdr.md` (spec 24.1, 1.4).
//!
//! Bytes read here cross a trust boundary (spec 25.1): [`CdrReader`] is total (never panics),
//! bounds-checks a sequence's byte span against what remains in the buffer *before* allocating
//! for it, and every message is capped at [`MAX_MESSAGE_BYTES`].

use thiserror::Error;

/// The telemetry frame cap (spec 25.1), reused here as the CDR message cap: nothing in this
/// crate's fixed message subset needs more, and a smaller cap bounds the cost of a hostile
/// sequence count before a single byte of it has been validated.
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// Why a CDR payload failed to decode. Every network byte is untrusted (spec 25.1): no variant
/// here is reachable only by a `panic!`/`unwrap` — [`CdrReader`] returns one of these instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CdrError {
    /// The 4-byte encapsulation header's representation id was not `00 00` or `00 01`.
    #[error("bad CDR encapsulation header")]
    BadHeader,
    /// A field needed more bytes than the buffer had left.
    #[error("CDR payload truncated")]
    Truncated,
    /// More than 3 bytes remained after the last field was read (spec: rosbags tolerates up to
    /// 3 trailing bytes past the last field; a 4th is an error).
    #[error("{0} trailing byte(s) after the last field")]
    TrailingBytes(usize),
    /// A `string` field's bytes were not valid UTF-8.
    #[error("invalid UTF-8 in a CDR string")]
    InvalidUtf8,
    /// A `string` field's last byte was not the required NUL terminator.
    #[error("CDR string is missing its NUL terminator")]
    MissingNul,
    /// The message, or a single length-prefixed field inside it, exceeds
    /// [`MAX_MESSAGE_BYTES`].
    #[error("CDR payload exceeds MAX_MESSAGE_BYTES")]
    TooLarge,
    /// A length or count prefix was structurally invalid (e.g. a zero-length CDR string, which
    /// must carry at least the NUL).
    #[error("bad CDR length prefix")]
    BadLength,
}

/// `CDR_LE` (`XTypes` Table 60; Fast DDS `#define CDR_LE 0x0001`): little-endian plain CDR.
const CDR_LE: [u8; 2] = [0x00, 0x01];

/// Little-endian CDR writer. `new()` writes the 4-byte encapsulation header
/// (`00 01 00 00`, matching rosbags: no trailing padding, options bytes zero); every write
/// helper aligns relative to the byte *after* that header, per `docs/api-notes/ros2-cdr.md`.
#[derive(Debug, Clone)]
pub struct CdrWriter {
    buf: Vec<u8>,
}

impl Default for CdrWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl CdrWriter {
    #[must_use]
    pub fn new() -> Self {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&CDR_LE);
        buf.extend_from_slice(&[0x00, 0x00]); // options bytes, always zero on encode.
        Self { buf }
    }

    /// Pad with zero bytes until the position (measured from the byte after the 4-byte header)
    /// is a multiple of `n`. Called before writing a primitive of size `n`; a sequence of `n`
    /// bytes writes zero padding, not garbage — real encoders leave this uninitialized (see
    /// `docs/api-notes/ros2-cdr.md`'s live-capture row), but our own writer's output must be
    /// deterministic to match the rosbags goldens byte-for-byte.
    fn align(&mut self, n: usize) {
        let rel = self.buf.len() - 4;
        let rem = rel % n;
        if rem != 0 {
            self.buf.resize(self.buf.len() + (n - rem), 0);
        }
    }

    pub fn write_u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }

    pub fn write_bool(&mut self, v: bool) -> &mut Self {
        self.write_u8(u8::from(v))
    }

    pub fn write_u32(&mut self, v: u32) -> &mut Self {
        self.align(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn write_i32(&mut self, v: i32) -> &mut Self {
        self.align(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn write_f64(&mut self, v: f64) -> &mut Self {
        self.align(8);
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// `uint32` length *including* the NUL, then the UTF-8 bytes, then `0x00`.
    pub fn write_string(&mut self, s: &str) -> &mut Self {
        let len = u32::try_from(s.len() + 1).expect("a message-subset string is under 4 GiB");
        self.write_u32(len);
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
        self
    }

    /// `uint32` count, then the raw bytes (no per-byte alignment: `uint8` has none).
    pub fn write_u8_seq(&mut self, v: &[u8]) -> &mut Self {
        let count = u32::try_from(v.len()).expect("a message-subset sequence is under 2^32");
        self.write_u32(count);
        self.buf.extend_from_slice(v);
        self
    }

    /// `uint32` count, then each `float64`, 8-byte aligned. Rosbags (and this writer) align to
    /// the element size only when `count > 0`: an empty sequence never touches the alignment
    /// padding, because [`Self::write_f64`] is simply never called.
    pub fn write_f64_seq(&mut self, v: &[f64]) -> &mut Self {
        let count = u32::try_from(v.len()).expect("a message-subset sequence is under 2^32");
        self.write_u32(count);
        for x in v {
            self.write_f64(*x);
        }
        self
    }

    /// `uint32` count, then each string (length-prefixed, NUL-terminated, 4-byte aligned).
    pub fn write_string_seq<S: AsRef<str>>(&mut self, v: &[S]) -> &mut Self {
        let count = u32::try_from(v.len()).expect("a message-subset sequence is under 2^32");
        self.write_u32(count);
        for s in v {
            self.write_string(s.as_ref());
        }
        self
    }

    /// Current length in bytes, including the 4-byte header. Used by nested-message encoders
    /// (none in this fixed subset need it directly, but tests use it to check golden lengths).
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        false // the 4-byte header is always present.
    }

    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

/// Little-endian (or big-endian, decode-only) CDR reader over a borrowed byte slice. Total:
/// every method returns [`CdrError`] instead of panicking, and a sequence's declared byte span
/// is checked against what remains in `buf` *before* an output `Vec`/`String` is allocated for
/// it (spec 25.1).
#[derive(Debug, Clone)]
pub struct CdrReader<'a> {
    buf: &'a [u8],
    pos: usize,
    big_endian: bool,
}

impl<'a> CdrReader<'a> {
    /// Validates the 4-byte encapsulation header and positions the reader at byte 4. Accepts
    /// `00 00` (big-endian) and `00 01` (little-endian) representation ids — and decodes
    /// multi-byte primitives accordingly, so a `string_hello_be`-style golden decodes to the
    /// same values a little-endian one would — the options bytes (2-3) are never inspected.
    pub fn new(buf: &'a [u8]) -> Result<Self, CdrError> {
        if buf.len() > MAX_MESSAGE_BYTES {
            return Err(CdrError::TooLarge);
        }
        if buf.len() < 4 {
            return Err(CdrError::Truncated);
        }
        match buf[0..2] {
            [0x00, 0x00] => Ok(Self {
                buf,
                pos: 4,
                big_endian: true,
            }),
            [0x00, 0x01] => Ok(Self {
                buf,
                pos: 4,
                big_endian: false,
            }),
            _ => Err(CdrError::BadHeader),
        }
    }

    fn align(&mut self, n: usize) {
        let rel = self.pos - 4;
        let rem = rel % n;
        if rem != 0 {
            self.pos += n - rem;
        }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn need(&self, n: usize) -> Result<(), CdrError> {
        if n > self.remaining() {
            Err(CdrError::Truncated)
        } else {
            Ok(())
        }
    }

    pub fn read_u8(&mut self) -> Result<u8, CdrError> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    pub fn read_bool(&mut self) -> Result<bool, CdrError> {
        Ok(self.read_u8()? != 0)
    }

    pub fn read_u32(&mut self) -> Result<u32, CdrError> {
        self.align(4);
        self.need(4)?;
        let bytes: [u8; 4] = self.buf[self.pos..self.pos + 4].try_into().unwrap();
        let v = if self.big_endian {
            u32::from_be_bytes(bytes)
        } else {
            u32::from_le_bytes(bytes)
        };
        self.pos += 4;
        Ok(v)
    }

    pub fn read_i32(&mut self) -> Result<i32, CdrError> {
        self.align(4);
        self.need(4)?;
        let bytes: [u8; 4] = self.buf[self.pos..self.pos + 4].try_into().unwrap();
        let v = if self.big_endian {
            i32::from_be_bytes(bytes)
        } else {
            i32::from_le_bytes(bytes)
        };
        self.pos += 4;
        Ok(v)
    }

    pub fn read_f64(&mut self) -> Result<f64, CdrError> {
        self.align(8);
        self.need(8)?;
        let bytes: [u8; 8] = self.buf[self.pos..self.pos + 8].try_into().unwrap();
        let v = if self.big_endian {
            f64::from_be_bytes(bytes)
        } else {
            f64::from_le_bytes(bytes)
        };
        self.pos += 8;
        Ok(v)
    }

    /// `uint32` length (including the NUL), then that many bytes, the last of which must be
    /// `0x00`. The byte span is checked against `remaining()` before the `String` is allocated.
    pub fn read_string(&mut self) -> Result<String, CdrError> {
        let len = self.read_u32()? as usize;
        if len == 0 {
            return Err(CdrError::BadLength);
        }
        self.need(len)?;
        let bytes = &self.buf[self.pos..self.pos + len];
        if bytes[len - 1] != 0 {
            return Err(CdrError::MissingNul);
        }
        let s = std::str::from_utf8(&bytes[..len - 1])
            .map_err(|_| CdrError::InvalidUtf8)?
            .to_owned();
        self.pos += len;
        Ok(s)
    }

    /// `uint32` count, then that many raw bytes. The byte span is checked before allocating.
    pub fn read_u8_seq(&mut self) -> Result<Vec<u8>, CdrError> {
        let n = self.read_u32()? as usize;
        self.need(n)?;
        let v = self.buf[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(v)
    }

    /// `uint32` count, then that many 8-byte-aligned `float64`s. The full span — the alignment
    /// pad plus `n * 8` bytes — is checked against what remains before `Vec::with_capacity(n)`
    /// runs, so a hostile huge `n` in a small buffer returns [`CdrError::Truncated`] having
    /// allocated nothing.
    pub fn read_f64_seq(&mut self) -> Result<Vec<f64>, CdrError> {
        let n = self.read_u32()? as usize;
        if n == 0 {
            return Ok(Vec::new());
        }
        let rel = self.pos - 4;
        let pad = (8 - rel % 8) % 8;
        let bytes = n
            .checked_mul(8)
            .and_then(|b| b.checked_add(pad))
            .ok_or(CdrError::TooLarge)?;
        if bytes > self.remaining() {
            return Err(CdrError::Truncated);
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.read_f64()?);
        }
        Ok(v)
    }

    /// `uint32` count, then that many length-prefixed strings. Element size is variable, so —
    /// unlike [`Self::read_f64_seq`] — this grows the output incrementally instead of
    /// pre-reserving `n` slots: each [`Self::read_string`] call already bounds-checks its own
    /// span, so the loop can allocate at most as many strings as the buffer actually holds.
    pub fn read_string_seq(&mut self) -> Result<Vec<String>, CdrError> {
        let n = self.read_u32()? as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(self.read_string()?);
        }
        Ok(v)
    }

    /// Consumes the reader, requiring at most 3 unread bytes remain (rosbags' own tolerance:
    /// `assert pos + 4 + 3 >= len(rawdata)`).
    pub fn finish(self) -> Result<(), CdrError> {
        let remaining = self.remaining();
        if remaining > 3 {
            Err(CdrError::TrailingBytes(remaining))
        } else {
            Ok(())
        }
    }
}
