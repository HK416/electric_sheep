//! The diagnostic code dictionary (P27).
//!
//! Every code the spec prints verbatim, plus the codes the checks mandated by spec 5.4
//! (type system) and spec 11.1 (semantic pass) need in order to be reportable. A code's
//! severity and title live here and nowhere else, so a message site only names the code.

use crate::diag::Severity;

// --- Type system (spec 5.4) ---------------------------------------------------------------
pub const TYPE_001: &str = "TYPE-001";
pub const TYPE_002: &str = "TYPE-002";
pub const TYPE_003: &str = "TYPE-003";
pub const TYPE_004: &str = "TYPE-004";
pub const TYPE_010: &str = "TYPE-010";
pub const TYPE_011: &str = "TYPE-011";
pub const TYPE_014: &str = "TYPE-014";
pub const TYPE_020: &str = "TYPE-020";

// --- Graph / semantic pass (spec 11.1) ----------------------------------------------------
pub const GRAPH_001: &str = "GRAPH-001";
pub const GRAPH_002: &str = "GRAPH-002";
pub const GRAPH_003: &str = "GRAPH-003";
pub const GRAPH_010: &str = "GRAPH-010";

// --- Canonical encoding / hash chain (spec 5.3) -------------------------------------------
pub const HASH_001: &str = "HASH-001";

// --- Observation (spec 7.2) ---------------------------------------------------------------
pub const OBS_021: &str = "OBS-021";
pub const OBS_034: &str = "OBS-034";

// --- Learning (spec 8.4) ------------------------------------------------------------------
pub const LRN_052: &str = "LRN-052";

// --- Deployment (spec 9.3, spec 11.6) -----------------------------------------------------
pub const DEP_031: &str = "DEP-031";
pub const DEP_114: &str = "DEP-114";

// --- Determinism (spec 6.6) ---------------------------------------------------------------
pub const DET_001: &str = "DET-001";
pub const DET_002: &str = "DET-002";
pub const DET_010: &str = "DET-010";
pub const DET_020: &str = "DET-020";
pub const DET_021: &str = "DET-021";
pub const DET_030: &str = "DET-030";
pub const DET_040: &str = "DET-040";

/// One row of the dictionary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeEntry {
    pub code: &'static str,
    pub severity: Severity,
    /// One-line English title, printed on the first line next to the code.
    pub title: &'static str,
    /// Spec section that defines the check, e.g. `"7.2"`.
    pub section: &'static str,
}

const fn e(
    code: &'static str,
    severity: Severity,
    title: &'static str,
    section: &'static str,
) -> CodeEntry {
    CodeEntry {
        code,
        severity,
        title,
        section,
    }
}

use Severity::{Error, Warning};

/// The full dictionary. Sorted by code so a reader can scan it.
#[rustfmt::skip]
pub const CODES: &[CodeEntry] = &[
    e(DEP_031, Error, "deployment envelope exceeds robot capability", "9.3"),
    e(DEP_114, Warning, "not supported by the selected runtime", "11.6"),
    e(DET_001, Error, "global RNG is forbidden; use TaskRng", "6.6"),
    e(DET_002, Error, "wall-clock access is forbidden", "6.6"),
    e(DET_010, Error, "standard transcendental is forbidden; use es-math::approx", "6.6"),
    e(DET_020, Error, "unstable iteration order", "6.6"),
    e(DET_021, Warning, "undeclared RNG stream", "6.6"),
    e(DET_030, Error, "unordered Reduce is rejected in deterministic mode", "6.6"),
    e(DET_040, Error, "tier 1 cannot be declared with an external backend", "6.6"),
    e(GRAPH_001, Error, "the graph contains a cycle", "11.1"),
    e(GRAPH_002, Error, "edge or boundary port refers to an unknown node", "11.1"),
    e(GRAPH_003, Error, "input port has more than one incoming edge", "11.1"),
    e(GRAPH_010, Error, "unknown port name", "11.1"),
    e(HASH_001, Error, "value is not encodable in canonical form", "5.3"),
    e(LRN_052, Error, "inference latency exceeds the control period", "8.4"),
    e(OBS_021, Error, "color space mismatch", "7.2"),
    e(OBS_034, Error, "intrinsics were not updated for the resize", "7.2"),
    e(TYPE_001, Error, "element type mismatch", "5.4"),
    e(TYPE_002, Error, "shape mismatch", "5.4"),
    e(TYPE_003, Error, "unit mismatch", "5.4"),
    e(TYPE_004, Error, "frame mismatch", "5.4"),
    e(TYPE_010, Error, "unit algebra is undefined for this unit", "5.4"),
    e(TYPE_011, Error, "policy input must be Normalized, Dimensionless or Token", "5.4"),
    e(TYPE_014, Error, "time alignment is not specified", "5.4"),
    e(TYPE_020, Error, "ImageSpec mismatch", "7.2"),
];

/// Looks a code up in the dictionary.
pub fn lookup(code: &str) -> Option<&'static CodeEntry> {
    CODES.iter().find(|entry| entry.code == code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn codes_are_unique_and_well_formed() {
        let mut seen = BTreeSet::new();
        for entry in CODES {
            assert!(seen.insert(entry.code), "duplicate code {}", entry.code);
            let (prefix, number) = entry.code.split_once('-').expect(entry.code);
            assert!(
                (2..=6).contains(&prefix.len())
                    && prefix.chars().all(|c| c.is_ascii_uppercase())
                    && number.len() == 3
                    && number.chars().all(|c| c.is_ascii_digit()),
                "malformed code {}",
                entry.code
            );
            assert!(!entry.title.is_empty() && !entry.section.is_empty());
        }
    }

    #[test]
    fn codes_are_sorted() {
        let sorted: Vec<_> = {
            let mut v: Vec<_> = CODES.iter().map(|e| e.code).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(sorted, CODES.iter().map(|e| e.code).collect::<Vec<_>>());
    }

    #[test]
    fn every_exported_constant_has_an_entry() {
        // The constants used across the crate; a code with no entry has no severity or title.
        for code in [
            TYPE_001, TYPE_002, TYPE_003, TYPE_004, TYPE_010, TYPE_011, TYPE_014, TYPE_020,
            GRAPH_001, GRAPH_002, GRAPH_003, GRAPH_010, HASH_001, OBS_021, OBS_034, LRN_052,
            DEP_031, DEP_114, DET_001, DET_002, DET_010, DET_020, DET_021, DET_030, DET_040,
        ] {
            assert!(lookup(code).is_some(), "{code} missing from CODES");
        }
    }

    #[test]
    fn spec_printed_codes_keep_their_severity() {
        assert_eq!(lookup(DEP_114).unwrap().severity, Severity::Warning);
        assert_eq!(lookup(TYPE_014).unwrap().severity, Severity::Error);
        assert!(lookup("NOPE-999").is_none());
    }
}
