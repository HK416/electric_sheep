//! Stable content-derived identifiers (shared with `es-ir`, spec Appendix B.1).

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Error;

/// A 128-bit identifier derived from a name path such as `"robot/arm/joint_1"`.
///
/// Derivation is `blake3(path)` truncated to 16 bytes: stable across runs, machines and
/// processes, unlike `DefaultHasher` or pointer identity. `Ord` so containers keyed by
/// `StableId` iterate deterministically (§3.4).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableId([u8; 16]);

impl StableId {
    /// Derives the id of a name path.
    pub fn from_path(path: &str) -> Self {
        let hash = blake3::hash(path.as_bytes());
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&hash.as_bytes()[..16]);
        Self(bytes)
    }

    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Parses the 32-character lowercase hex form produced by [`Display`](fmt::Display).
    pub fn from_hex(hex: &str) -> Result<Self, Error> {
        let bad = || Error::InvalidStableId(hex.to_owned());
        if hex.len() != 32 {
            return Err(bad());
        }
        let mut bytes = [0u8; 16];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).map_err(|_| bad())?;
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for StableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for StableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StableId({self})")
    }
}

impl Serialize for StableId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for StableId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let hex = String::deserialize(deserializer)?;
        Self::from_hex(&hex).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_path_is_stable_and_distinct() {
        let a = StableId::from_path("robot/arm/joint_1");
        assert_eq!(a, StableId::from_path("robot/arm/joint_1"));
        assert_ne!(a, StableId::from_path("robot/arm/joint_2"));
        // Golden vector, mirrored in `python/es/selfcheck.py` (S-14): a change here breaks
        // every hash chain downstream (§5.3) *and* silently desyncs the Python builder's
        // reimplementation, so both sides pin the same hex literal.
        assert_eq!(a.to_string(), "6d2e29e8077ed3c571f21602d29c7145");
    }

    #[test]
    fn hex_round_trip() {
        let id = StableId::from_path("robot/arm/joint_1");
        assert_eq!(StableId::from_hex(&id.to_string()).unwrap(), id);
        assert!(StableId::from_hex("nope").is_err());
        assert!(StableId::from_hex(&"z".repeat(32)).is_err());
    }

    #[test]
    fn serde_is_a_hex_string() {
        let id = StableId::from_path("scene/table");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{id}\""));
        assert_eq!(serde_json::from_str::<StableId>(&json).unwrap(), id);
    }
}
