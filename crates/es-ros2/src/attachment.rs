//! The 33-byte `rmw_zenoh` attachment (`docs/api-notes/rmw-zenoh.md` "Attachment: 33 bytes") and
//! the GID derivation used for it (`simplified_XXH3_128bits`, `docs/api-notes/rmw-zenoh.md`
//! "GID").

pub use crate::error::AttachmentError;

/// `sequence_number` (i64 LE, per publisher, starts at 1), `source_timestamp` (i64 LE, ns since
/// the UNIX epoch), then the 16-byte GID behind a one-byte LEB128 length prefix (always `0x10`:
/// `attachment_helpers.cpp` serializes the GID as a zenoh-ext sequence, and 16 fits in one
/// LEB128 byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attachment {
    pub seq: i64,
    pub source_timestamp_ns: i64,
    pub gid: [u8; 16],
}

/// Byte 16 of an encoded attachment: the LEB128 length of the 16-byte GID sequence. Always
/// `0x10` because 16 < 128 needs only the single low-order LEB128 byte.
const GID_LEN_PREFIX: u8 = 0x10;

impl Attachment {
    #[must_use]
    pub fn encode(&self) -> [u8; 33] {
        let mut out = [0u8; 33];
        out[0..8].copy_from_slice(&self.seq.to_le_bytes());
        out[8..16].copy_from_slice(&self.source_timestamp_ns.to_le_bytes());
        out[16] = GID_LEN_PREFIX;
        out[17..33].copy_from_slice(&self.gid);
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, AttachmentError> {
        if bytes.len() != 33 {
            return Err(AttachmentError::BadLength(bytes.len()));
        }
        if bytes[16] != GID_LEN_PREFIX {
            return Err(AttachmentError::BadGidLength(bytes[16]));
        }
        let seq = i64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes"));
        let source_timestamp_ns = i64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
        let mut gid = [0u8; 16];
        gid.copy_from_slice(&bytes[17..33]);
        Ok(Self {
            seq,
            source_timestamp_ns,
            gid,
        })
    }
}

/// `simplified_XXH3_128bits(token_key_expr)`, split low-64-then-high-64, both little-endian
/// (`docs/api-notes/rmw-zenoh.md` "GID": `memcpy(gid, &low64, 8)` then `&high64` on a
/// little-endian host). Stock XXH3-128, seed 0 — the same function `PyPI` `xxhash` exposes as
/// `xxh3_128_intdigest`, which is what `tests/golden/ros2/gid.json` was generated with.
#[must_use]
pub fn gid_of(token_key_expr: &str) -> [u8; 16] {
    let h = xxhash_rust::xxh3::xxh3_128(token_key_expr.as_bytes());
    let low = (h & u128::from(u64::MAX)) as u64;
    let high = (h >> 64) as u64;
    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&low.to_le_bytes());
    out[8..16].copy_from_slice(&high.to_le_bytes());
    out
}
