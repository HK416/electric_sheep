//! Shared formatting helpers.

use std::fmt::Write as _;

/// Lowercase hex, the format every `*_hash` is printed in (spec 5.3).
pub fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}
