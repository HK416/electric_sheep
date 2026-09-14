//! The HIL UDP wire protocol (`docs/design/ros2-boundary.md` section 7.2), little-endian.
//!
//! ```text
//! 0   4  magic "ESH1"      | 16  8  seq
//! 4   1  version = 1       | 24  8  sender_mono_ns
//! 5   1  kind              | 32  4  body_len
//! 6   2  reserved = 0      | 36  n  body
//! 8   8  session_id        | 36+n 16 tag = keyed_blake3(key, bytes[..36+n])[..16]
//! ```
//!
//! Every datagram that crosses this boundary is authenticated and bounded before anything
//! reaches the Safety Plane (spec 25.1): the decoder is total, allocates nothing beyond the
//! returned message, and has no `unwrap` on attacker-controlled input. The tag authenticates,
//! it does not encrypt (design note section 7.7).
//!
//! `f64` travel as [`f64::to_bits`], so a `NaN` arrives as the same `NaN` and meets the
//! plane's non-finite rule rather than being normalised away.

use es_core::PhysTick;
use thiserror::Error;

pub const MAGIC: [u8; 4] = *b"ESH1";
pub const VERSION: u8 = 1;
/// Bytes before the body.
pub const HEADER_LEN: usize = 36;
/// Bytes of keyed-blake3 tag after the body.
pub const TAG_LEN: usize = 16;
/// Largest UDP payload on IPv4. Above 1,472 this fragments on a 1,500 byte MTU link: that is
/// documented (design note section 7.2), not forbidden.
pub const MAX_DATAGRAM: usize = 65_507;

/// `Bye` reasons (design note section 7.2).
pub const BYE_HASH_MISMATCH: u16 = 1;
pub const BYE_SHAPE_MISMATCH: u16 = 2;
pub const BYE_VERSION: u16 = 3;
pub const BYE_SHUTDOWN: u16 = 4;

/// Why a datagram was refused. Every variant is counted as `rx_invalid` by the link and the
/// datagram is dropped: none of them reaches the Safety Plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum WireError {
    #[error("datagram is {0} bytes, over the {MAX_DATAGRAM} byte limit")]
    TooLarge(usize),
    #[error("datagram is {0} bytes, shorter than a header plus a tag")]
    TooShort(usize),
    #[error("bad magic")]
    BadMagic,
    #[error("unsupported wire version {0}")]
    BadVersion(u8),
    #[error("unknown message kind {0}")]
    BadKind(u8),
    #[error("body_len disagrees with the datagram length")]
    BadLength,
    #[error("authentication tag mismatch")]
    BadTag,
    #[error("command declares {rows} rows, over the horizon {h}")]
    BadRows { rows: u16, h: usize },
}

/// Message kinds, numbered as in the design note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgKind {
    Hello = 1,
    HelloAck = 2,
    State = 3,
    Command = 4,
    Heartbeat = 5,
    Bye = 6,
}

impl MsgKind {
    fn from_u8(v: u8) -> Result<Self, WireError> {
        Ok(match v {
            1 => Self::Hello,
            2 => Self::HelloAck,
            3 => Self::State,
            4 => Self::Command,
            5 => Self::Heartbeat,
            6 => Self::Bye,
            other => return Err(WireError::BadKind(other)),
        })
    }
}

/// The fields every datagram carries before its body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub kind: MsgKind,
    /// `0` in `Hello`; the link's assigned id afterwards.
    pub session_id: u64,
    /// Per sender and session, from 1, strictly increasing.
    pub seq: u64,
    /// The sender's monotonic clock. Statistics only: never a Safety Plane input (spec 3.4).
    pub sender_mono_ns: u64,
}

/// One action chunk from the external controller, as it arrives on the wire.
///
/// There is no accessor anywhere that hands this to the plant: it only ever enters
/// [`super::HilCore::on_command`], and what comes back out is a
/// [`es_safety::SafeAction`] (INV-12, INV-13).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Command<const NJ: usize, const H: usize> {
    /// The `State.tick` this command answers.
    pub obs_tick: PhysTick,
    /// Filled rows, `<= H`.
    pub rows: u16,
    pub actions: [[f64; NJ]; H],
}

/// A decoded datagram body (design note section 7.2's table).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HilMsg<const NJ: usize, const H: usize> {
    Hello {
        nj: u32,
        h: u32,
        deployment_hash: [u8; 32],
    },
    HelloAck {
        rate_num: u64,
        rate_den: u64,
        start_tick: u64,
    },
    State {
        tick: PhysTick,
        q: [f64; NJ],
        qd: [f64; NJ],
    },
    Command(Command<NJ, H>),
    Heartbeat,
    Bye {
        reason: u16,
    },
}

impl<const NJ: usize, const H: usize> HilMsg<NJ, H> {
    pub fn kind(&self) -> MsgKind {
        match self {
            Self::Hello { .. } => MsgKind::Hello,
            Self::HelloAck { .. } => MsgKind::HelloAck,
            Self::State { .. } => MsgKind::State,
            Self::Command(_) => MsgKind::Command,
            Self::Heartbeat => MsgKind::Heartbeat,
            Self::Bye { .. } => MsgKind::Bye,
        }
    }

    fn write_body(&self, out: &mut Vec<u8>) {
        match self {
            Self::Hello {
                nj,
                h,
                deployment_hash,
            } => {
                out.extend_from_slice(&nj.to_le_bytes());
                out.extend_from_slice(&h.to_le_bytes());
                out.extend_from_slice(deployment_hash);
            }
            Self::HelloAck {
                rate_num,
                rate_den,
                start_tick,
            } => {
                out.extend_from_slice(&rate_num.to_le_bytes());
                out.extend_from_slice(&rate_den.to_le_bytes());
                out.extend_from_slice(&start_tick.to_le_bytes());
            }
            Self::State { tick, q, qd } => {
                out.extend_from_slice(&tick.0.to_le_bytes());
                put_joints(out, q);
                put_joints(out, qd);
            }
            Self::Command(cmd) => {
                out.extend_from_slice(&cmd.obs_tick.0.to_le_bytes());
                out.extend_from_slice(&cmd.rows.to_le_bytes());
                out.extend_from_slice(&0u16.to_le_bytes());
                for row in cmd.actions.iter().take(cmd.rows as usize) {
                    put_joints(out, row);
                }
            }
            Self::Heartbeat => {}
            Self::Bye { reason } => out.extend_from_slice(&reason.to_le_bytes()),
        }
    }
}

fn put_joints(out: &mut Vec<u8>, v: &[f64]) {
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
}

/// Encodes one authenticated datagram into `out`, which is cleared first.
pub fn encode<const NJ: usize, const H: usize>(
    key: &[u8; 32],
    session_id: u64,
    seq: u64,
    sender_mono_ns: u64,
    msg: &HilMsg<NJ, H>,
    out: &mut Vec<u8>,
) {
    out.clear();
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(msg.kind() as u8);
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&session_id.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&sender_mono_ns.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    msg.write_body(out);
    let body_len = (out.len() - HEADER_LEN) as u32;
    out[32..36].copy_from_slice(&body_len.to_le_bytes());
    let tag = blake3::keyed_hash(key, out);
    out.extend_from_slice(&tag.as_bytes()[..TAG_LEN]);
}

/// Decodes one datagram. Total and non-panicking on any input (spec 25.1).
pub fn decode<const NJ: usize, const H: usize>(
    key: &[u8; 32],
    bytes: &[u8],
) -> Result<(Header, HilMsg<NJ, H>), WireError> {
    if bytes.len() > MAX_DATAGRAM {
        return Err(WireError::TooLarge(bytes.len()));
    }
    if bytes.len() < HEADER_LEN + TAG_LEN {
        return Err(WireError::TooShort(bytes.len()));
    }
    if bytes[0..4] != MAGIC {
        return Err(WireError::BadMagic);
    }
    if bytes[4] != VERSION {
        return Err(WireError::BadVersion(bytes[4]));
    }
    let kind = MsgKind::from_u8(bytes[5])?;
    let body_len = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]) as usize;
    let signed = HEADER_LEN
        .checked_add(body_len)
        .ok_or(WireError::BadLength)?;
    if signed.checked_add(TAG_LEN) != Some(bytes.len()) {
        return Err(WireError::BadLength);
    }
    let expect = blake3::keyed_hash(key, &bytes[..signed]);
    if !ct_eq(&expect.as_bytes()[..TAG_LEN], &bytes[signed..]) {
        return Err(WireError::BadTag);
    }
    let header = Header {
        kind,
        session_id: u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")),
        seq: u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes")),
        sender_mono_ns: u64::from_le_bytes(bytes[24..32].try_into().expect("8 bytes")),
    };
    let mut rd = Rd {
        b: &bytes[HEADER_LEN..signed],
        i: 0,
    };
    let msg = match kind {
        MsgKind::Hello => HilMsg::Hello {
            nj: rd.u32()?,
            h: rd.u32()?,
            deployment_hash: rd.hash()?,
        },
        MsgKind::HelloAck => HilMsg::HelloAck {
            rate_num: rd.u64()?,
            rate_den: rd.u64()?,
            start_tick: rd.u64()?,
        },
        MsgKind::State => HilMsg::State {
            tick: PhysTick(rd.u64()?),
            q: rd.joints()?,
            qd: rd.joints()?,
        },
        MsgKind::Command => {
            let obs_tick = PhysTick(rd.u64()?);
            let rows = rd.u16()?;
            let _reserved = rd.u16()?;
            if rows as usize > H {
                return Err(WireError::BadRows { rows, h: H });
            }
            let mut actions = [[0.0f64; NJ]; H];
            for row in actions.iter_mut().take(rows as usize) {
                *row = rd.joints()?;
            }
            HilMsg::Command(Command {
                obs_tick,
                rows,
                actions,
            })
        }
        MsgKind::Heartbeat => HilMsg::Heartbeat,
        MsgKind::Bye => HilMsg::Bye { reason: rd.u16()? },
    };
    if rd.i != rd.b.len() {
        return Err(WireError::BadLength);
    }
    Ok((header, msg))
}

/// Constant-time byte comparison: a tag check must not leak how many bytes matched.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = u8::from(a.len() != b.len());
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

/// A bounds-checked little-endian body reader. Every method returns [`WireError::BadLength`]
/// rather than panicking on a short body.
struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl Rd<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], WireError> {
        let end = self.i.checked_add(n).ok_or(WireError::BadLength)?;
        let out = self.b.get(self.i..end).ok_or(WireError::BadLength)?;
        self.i = end;
        Ok(out)
    }

    fn u16(&mut self) -> Result<u16, WireError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("2 bytes"),
        ))
    }

    fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn hash(&mut self) -> Result<[u8; 32], WireError> {
        Ok(self.take(32)?.try_into().expect("32 bytes"))
    }

    fn joints<const NJ: usize>(&mut self) -> Result<[f64; NJ], WireError> {
        let mut out = [0.0f64; NJ];
        for v in &mut out {
            *v = f64::from_bits(self.u64()?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const KEY: [u8; 32] = [7u8; 32];
    const NJ: usize = 3;
    const H: usize = 4;

    fn roundtrip(msg: &HilMsg<NJ, H>) -> (Header, HilMsg<NJ, H>) {
        let mut buf = Vec::new();
        encode(&KEY, 0x1122_3344_5566_7788, 9, 1_000, msg, &mut buf);
        decode::<NJ, H>(&KEY, &buf).expect("decodes")
    }

    fn sample_command() -> HilMsg<NJ, H> {
        let mut actions = [[0.0f64; NJ]; H];
        actions[0] = [1.0, -2.5, f64::NAN];
        actions[1] = [0.25, 0.5, 0.75];
        HilMsg::Command(Command {
            obs_tick: PhysTick(42),
            rows: 2,
            actions,
        })
    }

    #[test]
    fn every_kind_round_trips() {
        let msgs: [HilMsg<NJ, H>; 6] = [
            HilMsg::Hello {
                nj: NJ as u32,
                h: H as u32,
                deployment_hash: [0xAB; 32],
            },
            HilMsg::HelloAck {
                rate_num: 1000,
                rate_den: 1,
                start_tick: 7,
            },
            HilMsg::State {
                tick: PhysTick(11),
                q: [0.1, 0.2, 0.3],
                qd: [-1.0, 0.0, 1.0],
            },
            sample_command(),
            HilMsg::Heartbeat,
            HilMsg::Bye {
                reason: BYE_SHUTDOWN,
            },
        ];
        for msg in &msgs {
            let (hdr, back) = roundtrip(msg);
            assert_eq!(hdr.kind, msg.kind());
            assert_eq!(hdr.session_id, 0x1122_3344_5566_7788);
            assert_eq!(hdr.seq, 9);
            assert_eq!(hdr.sender_mono_ns, 1_000);
            match (msg, &back) {
                // `NaN != NaN`, so the command is compared on bit patterns instead.
                (HilMsg::Command(a), HilMsg::Command(b)) => {
                    assert_eq!(a.obs_tick, b.obs_tick);
                    assert_eq!(a.rows, b.rows);
                    for (ra, rb) in a.actions.iter().zip(&b.actions) {
                        for (x, y) in ra.iter().zip(rb) {
                            assert_eq!(x.to_bits(), y.to_bits());
                        }
                    }
                }
                _ => assert_eq!(*msg, back),
            }
        }
    }

    /// The header offsets of design note section 7.2, on bytes assembled by hand.
    #[test]
    fn header_offsets_match_the_design_note() {
        let mut buf = Vec::new();
        encode::<NJ, H>(
            &KEY,
            0x0807_0605_0403_0201,
            0x0002,
            0x0003,
            &HilMsg::Bye { reason: 0x1234 },
            &mut buf,
        );
        let mut want = Vec::new();
        want.extend_from_slice(b"ESH1"); // 0..4 magic
        want.push(1); // 4 version
        want.push(6); // 5 kind = Bye
        want.extend_from_slice(&[0, 0]); // 6..8 reserved
        want.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]); // 8..16 session_id
        want.extend_from_slice(&2u64.to_le_bytes()); // 16..24 seq
        want.extend_from_slice(&3u64.to_le_bytes()); // 24..32 sender_mono_ns
        want.extend_from_slice(&2u32.to_le_bytes()); // 32..36 body_len
        want.extend_from_slice(&0x1234u16.to_le_bytes()); // 36..38 body
        let tag = blake3::keyed_hash(&KEY, &want);
        want.extend_from_slice(&tag.as_bytes()[..TAG_LEN]);
        assert_eq!(buf, want);
        assert_eq!(buf.len(), HEADER_LEN + 2 + TAG_LEN);
    }

    #[test]
    fn rows_over_the_horizon_are_rejected() {
        // Hand-assembled: a `Command` claiming H + 1 rows, with a body long enough to hold
        // them, so only the row count can reject it.
        let rows = (H + 1) as u16;
        let mut body = Vec::new();
        body.extend_from_slice(&5u64.to_le_bytes());
        body.extend_from_slice(&rows.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes());
        body.extend_from_slice(&vec![0u8; 8 * NJ * rows as usize]);
        let buf = frame(MsgKind::Command, &body);
        assert_eq!(
            decode::<NJ, H>(&KEY, &buf),
            Err(WireError::BadRows { rows, h: H })
        );
    }

    #[test]
    fn a_body_len_that_disagrees_with_the_datagram_is_rejected() {
        let mut buf = Vec::new();
        encode::<NJ, H>(&KEY, 1, 1, 0, &sample_command(), &mut buf);
        let good = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        for wrong in [good - 8, good + 8, u32::MAX] {
            let mut bad = buf.clone();
            bad[32..36].copy_from_slice(&wrong.to_le_bytes());
            assert_eq!(decode::<NJ, H>(&KEY, &bad), Err(WireError::BadLength));
        }
        // Truncating the datagram keeps `body_len` but loses the tag.
        let mut short = buf.clone();
        short.truncate(buf.len() - 1);
        assert_eq!(decode::<NJ, H>(&KEY, &short), Err(WireError::BadLength));
    }

    #[test]
    fn an_oversize_datagram_is_rejected() {
        let buf = vec![0u8; MAX_DATAGRAM + 1];
        assert_eq!(
            decode::<NJ, H>(&KEY, &buf),
            Err(WireError::TooLarge(MAX_DATAGRAM + 1))
        );
    }

    #[test]
    fn a_wrong_key_fails_the_tag() {
        let mut buf = Vec::new();
        encode::<NJ, H>(&KEY, 1, 1, 0, &HilMsg::Heartbeat, &mut buf);
        assert_eq!(decode::<NJ, H>(&[8u8; 32], &buf), Err(WireError::BadTag));
        assert!(decode::<NJ, H>(&KEY, &buf).is_ok());
    }

    /// Wraps `body` in a correctly tagged header, so a test can aim at the body parser.
    fn frame(kind: MsgKind, body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_LEN + body.len() + TAG_LEN);
        buf.extend_from_slice(&MAGIC);
        buf.push(VERSION);
        buf.push(kind as u8);
        buf.extend_from_slice(&[0, 0]);
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
        buf.extend_from_slice(body);
        let tag = blake3::keyed_hash(&KEY, &buf);
        buf.extend_from_slice(&tag.as_bytes()[..TAG_LEN]);
        buf
    }

    proptest! {
        /// Arbitrary bytes never panic the decoder.
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..600)) {
            let _ = decode::<NJ, H>(&KEY, &bytes);
        }

        /// Arbitrary *authenticated* bodies never panic it either: re-tagging gets past the
        /// cheap rejections and into the body parser, which is the part with the arithmetic.
        #[test]
        fn arbitrary_tagged_bodies_never_panic(
            kind in 0u8..=8,
            body in proptest::collection::vec(any::<u8>(), 0..600),
        ) {
            let Ok(kind) = MsgKind::from_u8(kind) else { return Ok(()) };
            let _ = decode::<NJ, H>(&KEY, &frame(kind, &body));
        }
    }
}
