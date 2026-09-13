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
/// A condition expression is not boolean (spec 6.2 IR-C, spec 6.5).
pub const TYPE_030: &str = "TYPE-030";

// --- Graph / semantic pass (spec 11.1) ----------------------------------------------------
pub const GRAPH_001: &str = "GRAPH-001";
pub const GRAPH_002: &str = "GRAPH-002";
pub const GRAPH_003: &str = "GRAPH-003";
pub const GRAPH_010: &str = "GRAPH-010";

// --- Canonical encoding / hash chain (spec 5.3) -------------------------------------------
pub const HASH_001: &str = "HASH-001";
pub const HASH_002: &str = "HASH-002";

// --- Task (spec 7.4) ----------------------------------------------------------------------
pub const TASK_001: &str = "TASK-001";
pub const TASK_002: &str = "TASK-002";

// --- Control graph / IR-C (spec 6.2) ------------------------------------------------------
pub const CTRL_001: &str = "CTRL-001";
pub const CTRL_002: &str = "CTRL-002";
pub const CTRL_003: &str = "CTRL-003";
pub const CTRL_004: &str = "CTRL-004";
pub const CTRL_005: &str = "CTRL-005";
pub const CTRL_006: &str = "CTRL-006";

// --- Observation (spec 7.2 .. 7.5) --------------------------------------------------------
pub const OBS_021: &str = "OBS-021";
pub const OBS_034: &str = "OBS-034";
pub const OBS_040: &str = "OBS-040";
pub const OBS_041: &str = "OBS-041";
pub const OBS_042: &str = "OBS-042";
pub const OBS_043: &str = "OBS-043";

// --- Learning (spec 8.2 .. 8.6) -----------------------------------------------------------
pub const LRN_001: &str = "LRN-001";
pub const LRN_002: &str = "LRN-002";
pub const LRN_010: &str = "LRN-010";
pub const LRN_011: &str = "LRN-011";
pub const LRN_020: &str = "LRN-020";
pub const LRN_021: &str = "LRN-021";
pub const LRN_022: &str = "LRN-022";
pub const LRN_023: &str = "LRN-023";
pub const LRN_030: &str = "LRN-030";
pub const LRN_052: &str = "LRN-052";

// --- Deployment (spec 9.2 .. 9.4, spec 11.6) ----------------------------------------------
pub const DEP_001: &str = "DEP-001";
pub const DEP_010: &str = "DEP-010";
pub const DEP_011: &str = "DEP-011";
pub const DEP_012: &str = "DEP-012";
pub const DEP_013: &str = "DEP-013";
pub const DEP_014: &str = "DEP-014";
pub const DEP_015: &str = "DEP-015";
pub const DEP_020: &str = "DEP-020";
pub const DEP_021: &str = "DEP-021";
pub const DEP_022: &str = "DEP-022";
pub const DEP_023: &str = "DEP-023";
pub const DEP_024: &str = "DEP-024";
pub const DEP_030: &str = "DEP-030";
pub const DEP_031: &str = "DEP-031";
pub const DEP_040: &str = "DEP-040";
pub const DEP_114: &str = "DEP-114";

// --- Evaluation (spec 10.2, spec 10.4) ----------------------------------------------------
pub const EVAL_001: &str = "EVAL-001";
pub const EVAL_002: &str = "EVAL-002";
pub const EVAL_003: &str = "EVAL-003";
pub const EVAL_004: &str = "EVAL-004";
pub const EVAL_005: &str = "EVAL-005";
pub const EVAL_006: &str = "EVAL-006";

// --- Determinism (spec 6.6) ---------------------------------------------------------------
pub const DET_001: &str = "DET-001";
pub const DET_002: &str = "DET-002";
pub const DET_010: &str = "DET-010";
pub const DET_020: &str = "DET-020";
pub const DET_021: &str = "DET-021";
pub const DET_030: &str = "DET-030";
pub const DET_040: &str = "DET-040";

// --- Node factories (spec 6.3, spec 8.3) --------------------------------------------------
pub const FACTORY_001: &str = "FACTORY-001";
pub const FACTORY_002: &str = "FACTORY-002";
pub const FACTORY_003: &str = "FACTORY-003";

// --- Cross-IR checks (spec 11.1) ----------------------------------------------------------
pub const XIR_001: &str = "XIR-001";
pub const XIR_002: &str = "XIR-002";
pub const XIR_010: &str = "XIR-010";
pub const XIR_011: &str = "XIR-011";
pub const XIR_020: &str = "XIR-020";
pub const XIR_021: &str = "XIR-021";
pub const XIR_022: &str = "XIR-022";
pub const XIR_023: &str = "XIR-023";
pub const XIR_024: &str = "XIR-024";
pub const XIR_030: &str = "XIR-030";
pub const XIR_031: &str = "XIR-031";
pub const XIR_032: &str = "XIR-032";
pub const XIR_040: &str = "XIR-040";
pub const XIR_050: &str = "XIR-050";
pub const XIR_051: &str = "XIR-051";

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
    e(CTRL_001, Error, "control node reference names a node that does not exist", "6.2"),
    e(CTRL_002, Error, "control node is unreachable from the root", "6.2"),
    e(CTRL_003, Error, "control node has more than one parent", "6.2"),
    e(CTRL_004, Error, "the control graph contains a cycle", "6.2"),
    e(CTRL_005, Error, "Repeat count or SubTask timeout must be greater than 0", "6.2"),
    e(CTRL_006, Error, "SubTask names a stage, reward or channel the Task IR does not declare", "6.2"),
    e(DEP_001, Error, "unsupported Deployment IR schema version", "9.2"),
    e(DEP_010, Error, "size does not match robot.n_joints", "9.2"),
    e(DEP_011, Error, "safety limit is not finite", "9.3"),
    e(DEP_012, Error, "safety limit interval is empty", "9.3"),
    e(DEP_013, Error, "safety limit must be positive", "9.3"),
    e(DEP_014, Error, "soft margin leaves no room inside the limit", "9.3"),
    e(DEP_015, Error, "workspace is empty or not finite", "9.3"),
    e(DEP_020, Error, "deadline is 0", "9.4"),
    e(DEP_021, Error, "timing budget cannot be met at this rate", "9.4"),
    e(DEP_022, Error, "watchdog timeout exceeds the deadline it guards", "9.4"),
    e(DEP_023, Error, "watchdog parameter is out of range", "9.4"),
    e(DEP_024, Error, "duplicate watchdog", "9.4"),
    e(DEP_030, Error, "fallback policy is not executable", "9.4"),
    e(DEP_031, Error, "deployment envelope exceeds robot capability", "9.3"),
    e(DEP_040, Error, "action contract is inconsistent", "9.2"),
    e(DEP_114, Warning, "not supported by the selected runtime", "11.6"),
    e(DET_001, Error, "global RNG is forbidden; use TaskRng", "6.6"),
    e(DET_002, Error, "wall-clock access is forbidden", "6.6"),
    e(DET_010, Error, "standard transcendental is forbidden; use es-math::approx", "6.6"),
    e(DET_020, Error, "unstable iteration order", "6.6"),
    e(DET_021, Warning, "undeclared RNG stream", "6.6"),
    e(DET_030, Error, "unordered Reduce is rejected in deterministic mode", "6.6"),
    e(DET_040, Error, "tier 1 cannot be declared with an external backend", "6.6"),
    e(EVAL_001, Error, "acceptance names an undeclared metric or suite", "10.2"),
    e(EVAL_002, Error, "threshold or perturbation parameter is not finite", "10.2"),
    e(EVAL_003, Error, "episode batch is empty or its seed list does not match", "10.4"),
    e(EVAL_004, Error, "duplicate episode seed or perturbation stream", "10.4"),
    e(EVAL_005, Error, "perturbation range runs backwards", "10.2"),
    e(EVAL_006, Error, "augmentation allow-list carries no justification", "10.2"),
    e(FACTORY_001, Error, "unknown node kind", "6.3"),
    e(FACTORY_002, Error, "kind already owned by another registered factory", "6.3"),
    e(FACTORY_003, Error, "params do not match the node's shape", "6.3"),
    e(GRAPH_001, Error, "the graph contains a cycle", "11.1"),
    e(GRAPH_002, Error, "edge or boundary port refers to an unknown node", "11.1"),
    e(GRAPH_003, Error, "input port has more than one incoming edge", "11.1"),
    e(GRAPH_010, Error, "unknown port name", "11.1"),
    e(HASH_001, Error, "value is not encodable in canonical form", "5.3"),
    e(HASH_002, Error, "graph too symmetric to canonicalize", "5.3"),
    e(LRN_001, Error, "schema version disagrees with the graph", "8.2"),
    e(LRN_002, Error, "wrong number of input ports for this node kind", "8.3"),
    e(LRN_010, Error, "graph boundary does not match the declared tensor ports", "8.2"),
    e(LRN_011, Error, "boundary port disagrees with the policy contract", "8.4"),
    e(LRN_020, Error, "execute_chunk exceeds horizon", "8.4"),
    e(LRN_021, Error, "contract disagrees with the policy head it describes", "8.4"),
    e(LRN_022, Error, "observation_window disagrees with the temporal node", "7.5"),
    e(LRN_023, Error, "contract field that must be positive is 0", "8.4"),
    e(LRN_030, Error, "normalizer direction disagrees with the units it produces", "8.3"),
    e(LRN_052, Error, "inference latency exceeds the control period", "8.4"),
    e(OBS_021, Error, "color space mismatch", "7.2"),
    e(OBS_034, Error, "intrinsics were not updated for the resize", "7.2"),
    e(OBS_040, Error, "Normalize output does not carry a Normalized unit", "7.3"),
    e(OBS_041, Error, "augmentation node is not training_only", "7.3"),
    e(OBS_042, Error, "TemporalWindow reaches further back than History keeps", "7.5"),
    e(OBS_043, Error, "named output does not name a node port", "7.4"),
    e(TASK_001, Error, "ObservationSpec declaration and graph disagree", "7.4"),
    e(TASK_002, Error, "unsupported Task IR schema version", "6.1"),
    e(TYPE_001, Error, "element type mismatch", "5.4"),
    e(TYPE_002, Error, "shape mismatch", "5.4"),
    e(TYPE_003, Error, "unit mismatch", "5.4"),
    e(TYPE_004, Error, "frame mismatch", "5.4"),
    e(TYPE_010, Error, "unit algebra is undefined for this unit", "5.4"),
    e(TYPE_011, Error, "policy input must be Normalized, Dimensionless or Token", "5.4"),
    e(TYPE_014, Error, "time alignment is not specified", "5.4"),
    e(TYPE_020, Error, "ImageSpec mismatch", "7.2"),
    e(TYPE_030, Error, "condition expression is not boolean", "6.2"),
    e(XIR_001, Error, "observation IR belongs to a different Task IR", "7.4"),
    e(XIR_002, Error, "observation source reads a channel the Task IR does not declare", "7.4"),
    e(XIR_010, Error, "observation output and policy input do not name the same tensors", "8.4"),
    e(XIR_011, Error, "TemporalWindow n_steps disagrees with observation_window", "7.5"),
    e(XIR_020, Error, "action_dim disagrees with the deployed action contract", "8.5"),
    e(XIR_021, Error, "execution mode disagrees between Learning IR and Deployment IR", "8.5"),
    e(XIR_022, Error, "horizon or execute_chunk disagrees with the deployed action contract", "8.5"),
    e(XIR_023, Error, "replanning_hz is not an integer divisor of the control rate", "8.4"),
    e(XIR_024, Error, "runtime.deadline_ms does not fit the deployment inference budget", "8.4"),
    e(XIR_030, Error, "task ActionSpec dim disagrees with action_dim", "8.4"),
    e(XIR_031, Error, "task ActionSpec space disagrees with the deployed action space", "8.5"),
    e(XIR_032, Error, "task control_rate_hz disagrees with Deployment.rate.control", "9.2"),
    e(XIR_040, Error, "evaluation references a different Task or Observation IR", "10.4"),
    e(XIR_050, Error, "augmentation node would stay on during evaluation", "7.3"),
    e(XIR_051, Error, "allow-list names a node that is not an Observation Augment node", "10.4"),
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
                (2..=7).contains(&prefix.len())
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
            TYPE_001,
            TYPE_002,
            TYPE_003,
            TYPE_004,
            TYPE_010,
            TYPE_011,
            TYPE_014,
            TYPE_020,
            TYPE_030,
            GRAPH_001,
            GRAPH_002,
            GRAPH_003,
            GRAPH_010,
            HASH_001,
            HASH_002,
            TASK_001,
            TASK_002,
            CTRL_001,
            CTRL_002,
            CTRL_003,
            CTRL_004,
            CTRL_005,
            CTRL_006,
            OBS_021,
            OBS_034,
            OBS_040,
            OBS_041,
            OBS_042,
            OBS_043,
            LRN_001,
            LRN_002,
            LRN_010,
            LRN_011,
            LRN_020,
            LRN_021,
            LRN_022,
            LRN_023,
            LRN_030,
            LRN_052,
            DEP_001,
            DEP_010,
            DEP_011,
            DEP_012,
            DEP_013,
            DEP_014,
            DEP_015,
            DEP_020,
            DEP_021,
            DEP_022,
            DEP_023,
            DEP_024,
            DEP_030,
            DEP_031,
            DEP_040,
            DEP_114,
            EVAL_001,
            EVAL_002,
            EVAL_003,
            EVAL_004,
            EVAL_005,
            EVAL_006,
            DET_001,
            DET_002,
            DET_010,
            DET_020,
            DET_021,
            DET_030,
            DET_040,
            FACTORY_001,
            FACTORY_002,
            FACTORY_003,
            XIR_001,
            XIR_002,
            XIR_010,
            XIR_011,
            XIR_020,
            XIR_021,
            XIR_022,
            XIR_023,
            XIR_024,
            XIR_030,
            XIR_031,
            XIR_032,
            XIR_040,
            XIR_050,
            XIR_051,
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
