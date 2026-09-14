//! The `.eshil` v1 input log (`docs/design/ros2-boundary.md` section 7.4, spec 25.3).
//!
//! ```text
//! header   "ESHIL\0\0\x01" | u32 nj | u32 h | u64 rate_num | u64 rate_den
//!          | [u8;32] deployment_hash | u32 ir_len | serde_json bytes of the DeploymentIr
//! records  u8 tag | u32 len | body
//! trailer  0xFF | u32 40 | [u8;32] blake3(concatenated Decision record bytes) | u64 steps
//! ```
//!
//! The log records the Safety Plane's inputs *after* tick binning (design note section 7.3),
//! as integers and bit patterns, which is why a replay of a non-deterministic live run is
//! byte-identical. Nothing here reads a clock.

use std::io::Write;

use es_core::PhysTick;
use es_safety::{ActionChunk, ActionSource, FallbackKind, SafeAction};
use thiserror::Error;

/// `"ESHIL"` plus two reserved bytes plus the format version.
pub const LOG_MAGIC: [u8; 8] = *b"ESHIL\0\0\x01";

pub(crate) const TAG_OBSERVE: u8 = 0x01;
pub(crate) const TAG_HEARTBEAT: u8 = 0x02;
pub(crate) const TAG_STEP: u8 = 0x03;
pub(crate) const TAG_DECISION: u8 = 0x10;
/// Non-normative, ignored by replay.
pub(crate) const TAG_STATS: u8 = 0x20;
pub(crate) const TAG_TRAILER: u8 = 0xFF;
pub(crate) const TRAILER_LEN: u32 = 40;
/// Bytes before `ir_len`'s payload: magic, `nj`, `h`, `rate_num`, `rate_den`, hash, `ir_len`.
pub(crate) const HEADER_FIXED: usize = 8 + 4 + 4 + 8 + 8 + 32 + 4;
/// One record's `tag` plus its `u32 len`.
pub(crate) const REC_PREFIX: usize = 5;

/// Why a `.eshil` log could not be replayed. A *truncated* log is not an error: it replays its
/// complete prefix and reports [`super::ReplayReport::truncated`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HilLogError {
    #[error("not an .eshil v1 log, or the header is incomplete")]
    Header,
    #[error("log is nj={nj} h={h}, replayed as nj={want_nj} h={want_h}")]
    Shape {
        nj: u32,
        h: u32,
        want_nj: usize,
        want_h: usize,
    },
    #[error("the embedded Deployment IR does not hash to the header's deployment_hash")]
    DeploymentHash,
    #[error("embedded Deployment IR: {0}")]
    Ir(String),
    #[error("malformed record: {0}")]
    Record(&'static str),
}

/// The `.eshil` header, decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogHeader {
    pub nj: u32,
    pub h: u32,
    pub rate_num: u64,
    pub rate_den: u64,
    pub deployment_hash: [u8; 32],
    /// Offset of the first record.
    pub end: usize,
}

pub(crate) fn encode_header(
    nj: usize,
    h: usize,
    rate_num: u64,
    rate_den: u64,
    deployment_hash: &[u8; 32],
    ir_json: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_FIXED + ir_json.len());
    out.extend_from_slice(&LOG_MAGIC);
    out.extend_from_slice(&(nj as u32).to_le_bytes());
    out.extend_from_slice(&(h as u32).to_le_bytes());
    out.extend_from_slice(&rate_num.to_le_bytes());
    out.extend_from_slice(&rate_den.to_le_bytes());
    out.extend_from_slice(deployment_hash);
    out.extend_from_slice(&(ir_json.len() as u32).to_le_bytes());
    out.extend_from_slice(ir_json);
    out
}

/// Decodes the header and returns it with the embedded IR's JSON bytes.
pub(crate) fn parse_header(log: &[u8]) -> Result<(LogHeader, &[u8]), HilLogError> {
    let fixed = log.get(..HEADER_FIXED).ok_or(HilLogError::Header)?;
    if fixed[..8] != LOG_MAGIC {
        return Err(HilLogError::Header);
    }
    let u32_at = |o: usize| u32::from_le_bytes(fixed[o..o + 4].try_into().expect("4 bytes"));
    let u64_at = |o: usize| u64::from_le_bytes(fixed[o..o + 8].try_into().expect("8 bytes"));
    let ir_len = u32_at(64) as usize;
    let end = HEADER_FIXED
        .checked_add(ir_len)
        .ok_or(HilLogError::Header)?;
    let json = log.get(HEADER_FIXED..end).ok_or(HilLogError::Header)?;
    Ok((
        LogHeader {
            nj: u32_at(8),
            h: u32_at(12),
            rate_num: u64_at(16),
            rate_den: u64_at(24),
            deployment_hash: fixed[32..64].try_into().expect("32 bytes"),
            end,
        },
        json,
    ))
}

/// One record at `i`: `(tag, body, raw record bytes, next offset)`. `None` means the log ends
/// inside this record, which is truncation, not corruption.
pub(crate) fn read_record(log: &[u8], i: usize) -> Option<(u8, &[u8], &[u8], usize)> {
    let prefix = log.get(i..i.checked_add(REC_PREFIX)?)?;
    let len = u32::from_le_bytes(prefix[1..5].try_into().expect("4 bytes")) as usize;
    let end = i.checked_add(REC_PREFIX)?.checked_add(len)?;
    let body = log.get(i + REC_PREFIX..end)?;
    Some((prefix[0], body, &log[i..end], end))
}

pub(crate) fn record(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(REC_PREFIX + body.len());
    out.push(tag);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    out
}

// --- record bodies ------------------------------------------------------------------------

pub(crate) fn observe_body<const NJ: usize>(
    tick: PhysTick,
    q: &[f64; NJ],
    qd: &[f64; NJ],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + 16 * NJ);
    out.extend_from_slice(&tick.0.to_le_bytes());
    put_f64s(&mut out, q);
    put_f64s(&mut out, qd);
    out
}

pub(crate) fn heartbeat_body(tick: PhysTick) -> Vec<u8> {
    tick.0.to_le_bytes().to_vec()
}

pub(crate) fn step_body<const NJ: usize, const H: usize>(
    tick: PhysTick,
    obs_age_us: u64,
    chunk: Option<&ActionChunk<NJ, H>>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(17 + 2 + 8 * NJ * H);
    out.extend_from_slice(&tick.0.to_le_bytes());
    out.extend_from_slice(&obs_age_us.to_le_bytes());
    match chunk {
        None => out.push(0),
        Some(c) => {
            out.push(1);
            let rows = c.valid.min(H);
            out.extend_from_slice(&(rows as u16).to_le_bytes());
            for row in c.actions.iter().take(rows) {
                put_f64s(&mut out, row);
            }
        }
    }
    out
}

pub(crate) fn decision_body<const NJ: usize>(tick: PhysTick, a: &SafeAction<NJ>) -> Vec<u8> {
    let (source, fallback) = match a.source {
        ActionSource::Policy => (0u8, 0xFFu8),
        ActionSource::Clamped => (1, 0xFF),
        ActionSource::Fallback(k) => (2, fallback_index(k)),
    };
    let mut out = Vec::with_capacity(8 + 8 * NJ + 6);
    out.extend_from_slice(&tick.0.to_le_bytes());
    for v in &a.q {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out.push(source);
    out.push(fallback);
    out.extend_from_slice(&a.events.bits().to_le_bytes());
    out
}

/// `FallbackKind`'s declaration index (design note section 7.4).
fn fallback_index(k: FallbackKind) -> u8 {
    match k {
        FallbackKind::HoldPosition => 0,
        FallbackKind::ZeroVelocity => 1,
        FallbackKind::RetractToHome => 2,
        FallbackKind::HandoffController => 3,
        FallbackKind::EmergencyStop => 4,
    }
}

fn put_f64s(out: &mut Vec<u8>, v: &[f64]) {
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
}

// --- record body decoding (replay) --------------------------------------------------------

/// A bounds-checked little-endian cursor. `None` from any method means the body is malformed.
pub(crate) struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cur<'a> {
    pub(crate) fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.i.checked_add(n)?;
        let out = self.b.get(self.i..end)?;
        self.i = end;
        Some(out)
    }

    pub(crate) fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    pub(crate) fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    pub(crate) fn f64s<const NJ: usize>(&mut self) -> Option<[f64; NJ]> {
        let mut out = [0.0f64; NJ];
        for v in &mut out {
            *v = f64::from_bits(self.u64()?);
        }
        Some(out)
    }

    pub(crate) fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    pub(crate) fn at_end(&self) -> bool {
        self.i == self.b.len()
    }
}

// --- writer -------------------------------------------------------------------------------

/// Appends records to the caller's sink and keeps the running decisions hash.
///
/// `Box<dyn Write + Send>` is `std`'s own trait object, not a new extension point (INV-17).
/// An I/O error is latched rather than returned, because the methods that write are the ones
/// whose signatures must not grow a `Result` ([`super::HilCore::observe_state`],
/// [`super::HilCore::tick`]); [`super::HilCore::finish`] reports it.
pub(crate) struct LogWriter {
    sink: Box<dyn Write + Send>,
    hasher: blake3::Hasher,
    steps: u64,
    err: Option<std::io::Error>,
}

impl std::fmt::Debug for LogWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogWriter")
            .field("steps", &self.steps)
            .field("err", &self.err)
            .finish_non_exhaustive()
    }
}

impl LogWriter {
    pub(crate) fn new(sink: Box<dyn Write + Send>) -> Self {
        Self {
            sink,
            hasher: blake3::Hasher::new(),
            steps: 0,
            err: None,
        }
    }

    pub(crate) fn raw(&mut self, bytes: &[u8]) {
        if self.err.is_some() {
            return;
        }
        if let Err(e) = self.sink.write_all(bytes) {
            self.err = Some(e);
        }
    }

    /// Appends one record. A `Decision` record also feeds the decisions hash, which is exactly
    /// the sequence replay re-derives.
    pub(crate) fn write(&mut self, tag: u8, body: &[u8]) {
        let bytes = record(tag, body);
        if tag == TAG_DECISION {
            self.hasher.update(&bytes);
            self.steps += 1;
        }
        self.raw(&bytes);
    }

    /// Writes the trailer and returns the decisions hash.
    pub(crate) fn finish(mut self) -> Result<[u8; 32], std::io::Error> {
        let hash = *self.hasher.finalize().as_bytes();
        let mut body = Vec::with_capacity(TRAILER_LEN as usize);
        body.extend_from_slice(&hash);
        body.extend_from_slice(&self.steps.to_le_bytes());
        self.raw(&record(TAG_TRAILER, &body));
        if self.err.is_none() {
            if let Err(e) = self.sink.flush() {
                self.err = Some(e);
            }
        }
        match self.err {
            Some(e) => Err(e),
            None => Ok(hash),
        }
    }
}
