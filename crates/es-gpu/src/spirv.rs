//! A hand SPIR-V reader/writer, just enough for spec §3.4 step 3.
//!
//! Two jobs:
//!
//! * **patch** — put the execution modes and `NoContraction` decorations that `slangc` does
//!   not emit into a compiled module ([`apply_exec_modes`]);
//! * **inspect** — prove they are there ([`spirv_has_execution_mode`],
//!   [`spirv_no_contraction_count`]), so a test asserts on the binary instead of trusting a
//!   compiler flag.
//!
//! No dependency: the format is a header plus a flat stream of `(word_count, opcode)`
//! instructions, and the logical layout (capabilities → extensions → memory model → entry
//! points → execution modes → debug → annotations → types/functions) tells us where an
//! insertion is legal.
//!
//! Hand-written SPIR-V is checked by [`validate_spirv`] (`spirv-val` from the Vulkan SDK)
//! rather than trusted, because a header parse cannot tell a legal module from a plausible
//! one.

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;

use crate::caps::ExecModes;
use crate::error::GpuError;

const MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;

// Opcodes used here.
const OP_CAPABILITY: u16 = 17;
const OP_EXTENSION: u16 = 10;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_DECORATION_GROUP: u16 = 73;
const OP_GROUP_DECORATE: u16 = 74;
const OP_GROUP_MEMBER_DECORATE: u16 = 75;

/// `Decoration::NoContraction`.
pub const DECORATION_NO_CONTRACTION: u32 = 42;

/// `ExecutionMode` values of `SPV_KHR_float_controls`.
pub const EXEC_MODE_DENORM_PRESERVE: u32 = 4459;
pub const EXEC_MODE_DENORM_FLUSH_TO_ZERO: u32 = 4460;
pub const EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4461;
pub const EXEC_MODE_ROUNDING_MODE_RTE: u32 = 4462;
pub const EXEC_MODE_ROUNDING_MODE_RTZ: u32 = 4463;

/// The float width every mode this module adds applies to (spec §3.3: f32 only).
const PATCH_WIDTH: u32 = 32;

/// The mode that contradicts `mode` at the same width. A module declaring both is invalid
/// SPIR-V, so the patcher refuses rather than producing one.
fn conflicting_mode(mode: u32) -> Option<u32> {
    match mode {
        EXEC_MODE_DENORM_FLUSH_TO_ZERO => Some(EXEC_MODE_DENORM_PRESERVE),
        EXEC_MODE_DENORM_PRESERVE => Some(EXEC_MODE_DENORM_FLUSH_TO_ZERO),
        EXEC_MODE_ROUNDING_MODE_RTE => Some(EXEC_MODE_ROUNDING_MODE_RTZ),
        EXEC_MODE_ROUNDING_MODE_RTZ => Some(EXEC_MODE_ROUNDING_MODE_RTE),
        _ => None,
    }
}

/// `Capability` values of `SPV_KHR_float_controls`.
const CAP_DENORM_FLUSH_TO_ZERO: u32 = 4465;
const CAP_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4466;
const CAP_ROUNDING_MODE_RTE: u32 = 4467;

const FLOAT_CONTROLS_EXTENSION: &str = "SPV_KHR_float_controls";

/// Float arithmetic opcodes whose result `NoContraction` applies to. Result id is always
/// operand word 2 (after the result type).
const FLOAT_ARITHMETIC: &[u16] = &[
    127, // OpFNegate
    129, // OpFAdd
    131, // OpFSub
    133, // OpFMul
    136, // OpFDiv
    140, // OpFRem
    141, // OpFMod
    142, // OpVectorTimesScalar
    143, // OpMatrixTimesScalar
    144, // OpVectorTimesMatrix
    145, // OpMatrixTimesVector
    146, // OpMatrixTimesMatrix
    147, // OpOuterProduct
    148, // OpDot
];

/// One instruction, as an index range into the module's word slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Inst {
    op: u16,
    start: usize,
    len: usize,
}

impl Inst {
    fn end(&self) -> usize {
        self.start + self.len
    }
}

/// Split a module into instructions. `None` if the header or the stream is malformed.
fn instructions(words: &[u32]) -> Option<Vec<Inst>> {
    if words.len() < HEADER_WORDS || words[0] != MAGIC {
        return None;
    }
    let mut out = Vec::new();
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let len = (words[i] >> 16) as usize;
        let op = (words[i] & 0xffff) as u16;
        if len == 0 || i + len > words.len() {
            return None;
        }
        out.push(Inst { op, start: i, len });
        i += len;
    }
    Some(out)
}

/// Encode one instruction.
fn inst(op: u16, operands: &[u32]) -> Vec<u32> {
    let len = u32::try_from(operands.len() + 1).expect("instruction too long");
    let mut v = Vec::with_capacity(operands.len() + 1);
    v.push((len << 16) | u32::from(op));
    v.extend_from_slice(operands);
    v
}

/// Encode a SPIR-V literal string: nul-terminated UTF-8, packed little-endian, zero-padded.
fn literal_string(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decode a nul-terminated literal string starting at `words[0]`.
fn decode_string(words: &[u32]) -> String {
    let mut bytes = Vec::new();
    for w in words {
        for b in w.to_le_bytes() {
            if b == 0 {
                return String::from_utf8_lossy(&bytes).into_owned();
            }
            bytes.push(b);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Whether the module declares `mode` as an `OpExecutionMode` on any entry point.
///
/// The inspector the tests use to prove spec §3.4 step 3 was actually applied.
pub fn spirv_has_execution_mode(words: &[u32], mode: u32) -> bool {
    let Some(insts) = instructions(words) else {
        return false;
    };
    insts
        .iter()
        .any(|i| i.op == OP_EXECUTION_MODE && i.len >= 3 && words[i.start + 2] == mode)
}

/// Number of `OpDecorate <id> NoContraction` in the module.
pub fn spirv_no_contraction_count(words: &[u32]) -> usize {
    let Some(insts) = instructions(words) else {
        return 0;
    };
    insts
        .iter()
        .filter(|i| {
            i.op == OP_DECORATE && i.len >= 3 && words[i.start + 2] == DECORATION_NO_CONTRACTION
        })
        .count()
}

/// Number of float arithmetic instructions in the module — the ceiling for the count above.
pub fn spirv_float_arithmetic_count(words: &[u32]) -> usize {
    let Some(insts) = instructions(words) else {
        return 0;
    };
    insts
        .iter()
        .filter(|i| FLOAT_ARITHMETIC.contains(&i.op))
        .count()
}

/// Entry-point names declared by the module.
pub fn spirv_entry_points(words: &[u32]) -> Vec<String> {
    let Some(insts) = instructions(words) else {
        return Vec::new();
    };
    insts
        .iter()
        .filter(|i| i.op == OP_ENTRY_POINT && i.len >= 4)
        .map(|i| decode_string(&words[i.start + 3..i.end()]))
        .collect()
}

/// Every `OpExecutionMode <entry> <mode> <literal>` in the module, as `(entry, mode, width)`.
///
/// Restricted to the one-literal-operand form, which is what every `SPV_KHR_float_controls`
/// mode uses; `LocalSize` and the other multi-operand modes carry a different word count and
/// so can never be mistaken for one.
fn declared_modes(words: &[u32], insts: &[Inst]) -> Vec<(u32, u32, u32)> {
    insts
        .iter()
        .filter(|i| i.op == OP_EXECUTION_MODE && i.len == 4)
        .map(|i| (words[i.start + 1], words[i.start + 2], words[i.start + 3]))
        .collect()
}

/// Add the execution modes and decorations of `modes` to a compiled module (spec §3.4
/// step 3).
///
/// Applied to **every** `OpEntryPoint`, not just the first. Idempotent, and idempotent per
/// `(entry point, mode, width)`: `DenormFlushToZero 16` does not suppress the requested
/// `DenormFlushToZero 32`, and a mode already present on entry point A is still added to
/// entry point B. `Err` if `words` is not a SPIR-V module with an entry point, or if an
/// entry point already declares a mode that contradicts a requested one at the same width
/// (`DenormPreserve 32` vs `DenormFlushToZero 32`) — declaring both is invalid SPIR-V, so
/// the patcher refuses instead of emitting it.
pub fn apply_exec_modes(words: &[u32], modes: ExecModes) -> Result<Vec<u32>, GpuError> {
    let insts =
        instructions(words).ok_or_else(|| GpuError::Spirv("not a SPIR-V module".to_owned()))?;
    let entries: Vec<u32> = insts
        .iter()
        .filter(|i| i.op == OP_ENTRY_POINT && i.len >= 3)
        .map(|i| words[i.start + 2])
        .collect();
    if entries.is_empty() {
        return Err(GpuError::Spirv("module has no entry point".to_owned()));
    }

    // Section boundaries. The layout is fixed by the SPIR-V spec, so "after the last X" is
    // a legal insertion point for another X.
    let after = |ops: &[u16], fallback: usize| -> usize {
        insts
            .iter()
            .filter(|i| ops.contains(&i.op))
            .map(Inst::end)
            .max()
            .unwrap_or(fallback)
    };
    let cap_end = after(&[OP_CAPABILITY], HEADER_WORDS);
    let ext_end = after(&[OP_EXTENSION], cap_end);
    let em_end = after(&[OP_EXECUTION_MODE, OP_ENTRY_POINT], ext_end);
    // Annotations come after the debug section; when a module has none, the first
    // instruction that is neither a header-section nor a debug instruction starts the types.
    let annotations = [
        OP_DECORATE,
        OP_MEMBER_DECORATE,
        OP_DECORATION_GROUP,
        OP_GROUP_DECORATE,
        OP_GROUP_MEMBER_DECORATE,
    ];
    let dec_end = insts
        .iter()
        .filter(|i| annotations.contains(&i.op))
        .map(Inst::end)
        .max()
        .unwrap_or_else(|| first_type_index(&insts, em_end));

    let mut new_caps = Vec::new();
    let mut new_modes = Vec::new();
    let declared = declared_modes(words, &insts);
    let has_cap = |c: u32| {
        insts
            .iter()
            .any(|i| i.op == OP_CAPABILITY && i.len >= 2 && words[i.start + 1] == c)
    };
    for (want, cap, mode) in [
        (
            modes.denorm_flush_to_zero_f32,
            CAP_DENORM_FLUSH_TO_ZERO,
            EXEC_MODE_DENORM_FLUSH_TO_ZERO,
        ),
        (
            modes.rounding_mode_rte_f32,
            CAP_ROUNDING_MODE_RTE,
            EXEC_MODE_ROUNDING_MODE_RTE,
        ),
        (
            modes.signed_zero_inf_nan_preserve_f32,
            CAP_SIGNED_ZERO_INF_NAN_PRESERVE,
            EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE,
        ),
    ] {
        if !want {
            continue;
        }
        // Refuse before emitting anything, so a conflicting module is never half-patched.
        for &e in &entries {
            if let Some(other) = conflicting_mode(mode) {
                if declared.contains(&(e, other, PATCH_WIDTH)) {
                    return Err(GpuError::Spirv(format!(
                        "entry point %{e} already declares execution mode {other} at width \
                         {PATCH_WIDTH}; mode {mode} contradicts it"
                    )));
                }
            }
        }
        if !has_cap(cap) {
            new_caps.extend(inst(OP_CAPABILITY, &[cap]));
        }
        for &e in &entries {
            if !declared.contains(&(e, mode, PATCH_WIDTH)) {
                new_modes.extend(inst(OP_EXECUTION_MODE, &[e, mode, PATCH_WIDTH]));
            }
        }
    }

    let mut new_exts = Vec::new();
    let have_ext = insts.iter().any(|i| {
        i.op == OP_EXTENSION
            && decode_string(&words[i.start + 1..i.end()]) == FLOAT_CONTROLS_EXTENSION
    });
    if !new_caps.is_empty() && !have_ext {
        let mut operands = Vec::new();
        operands.extend(literal_string(FLOAT_CONTROLS_EXTENSION));
        new_exts.extend(inst(OP_EXTENSION, &operands));
    }

    let mut new_decorations = Vec::new();
    if modes.no_contraction {
        let mut decorated: Vec<u32> = insts
            .iter()
            .filter(|i| {
                i.op == OP_DECORATE && i.len >= 3 && words[i.start + 2] == DECORATION_NO_CONTRACTION
            })
            .map(|i| words[i.start + 1])
            .collect();
        for i in insts.iter().filter(|i| FLOAT_ARITHMETIC.contains(&i.op)) {
            if i.len < 3 {
                continue;
            }
            let id = words[i.start + 2];
            if decorated.contains(&id) {
                continue;
            }
            decorated.push(id);
            new_decorations.extend(inst(OP_DECORATE, &[id, DECORATION_NO_CONTRACTION]));
        }
    }

    let mut out = Vec::with_capacity(
        words.len() + new_caps.len() + new_exts.len() + new_modes.len() + new_decorations.len(),
    );
    out.extend_from_slice(&words[..cap_end]);
    out.extend_from_slice(&new_caps);
    out.extend_from_slice(&words[cap_end..ext_end]);
    out.extend_from_slice(&new_exts);
    out.extend_from_slice(&words[ext_end..em_end]);
    out.extend_from_slice(&new_modes);
    out.extend_from_slice(&words[em_end..dec_end]);
    out.extend_from_slice(&new_decorations);
    out.extend_from_slice(&words[dec_end..]);
    Ok(out)
}

/// Counter for `spirv-val` scratch names, so concurrent validations never share a file.
static VAL_COUNTER: AtomicU64 = AtomicU64::new(0);
static VAL_MISSING_NOTE: Once = Once::new();

/// Run `spirv-val` (Vulkan SDK) over a module.
///
/// `Ok(true)` — it ran and accepted the module. `Ok(false)` — it is not installed, so
/// nothing was checked; a note is printed once per process. `Err` — it rejected the module,
/// or it is missing and `ES_REQUIRE_SPIRV_VAL=1` says that is not acceptable. The executable
/// is `spirv-val` on `PATH`, or `ES_SPIRV_VAL`.
///
/// This is the only thing in the crate that can catch a bad hand-written patch: `spirv.rs`
/// writes SPIR-V, and a header parse only proves the word stream is well-formed.
pub fn validate_spirv(words: &[u32]) -> Result<bool, GpuError> {
    let exe = std::env::var("ES_SPIRV_VAL").unwrap_or_else(|_| "spirv-val".to_owned());
    let n = VAL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("es-spirv-val-{}-{n}.spv", std::process::id()));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes)?;
    let out = Command::new(&exe).arg(&path).output();
    let _ = std::fs::remove_file(&path);
    match out {
        Ok(o) if o.status.success() => Ok(true),
        Ok(o) => Err(GpuError::Spirv(format!(
            "spirv-val rejected the module: {}",
            String::from_utf8_lossy(&o.stderr).trim()
        ))),
        Err(e) => {
            if std::env::var("ES_REQUIRE_SPIRV_VAL").as_deref() == Ok("1") {
                return Err(GpuError::Spirv(format!(
                    "ES_REQUIRE_SPIRV_VAL=1 but `{exe}` did not run: {e}"
                )));
            }
            VAL_MISSING_NOTE.call_once(|| {
                println!(
                    "NOTE spirv-val is not on PATH: patched SPIR-V is unvalidated (set \
                     ES_REQUIRE_SPIRV_VAL=1 to make this an error)"
                );
            });
            Ok(false)
        }
    }
}

/// Index of the first instruction that starts the type/constant section, for a module with
/// no annotations at all.
fn first_type_index(insts: &[Inst], fallback: usize) -> usize {
    // Debug-section opcodes: OpNop, OpSourceContinued, OpSource, OpSourceExtension, OpName,
    // OpMemberName, OpString, OpLine, OpModuleProcessed — plus the header sections.
    const BEFORE_TYPES: &[u16] = &[
        0,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        330,
        OP_EXTENSION,
        OP_EXT_INST_IMPORT,
        OP_MEMORY_MODEL,
        OP_ENTRY_POINT,
        OP_EXECUTION_MODE,
        OP_CAPABILITY,
    ];
    insts
        .iter()
        .find(|i| !BEFORE_TYPES.contains(&i.op))
        .map_or(fallback, |i| i.start)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smallest module the patcher accepts: header, `OpCapability` Shader, memory model,
    /// entry point, `LocalSize`, one `OpFMul` in a function.
    /// `extra` adds float-control execution modes in the section SPIR-V's layout requires.
    fn module_with_modes(extra: &[(u32, u32)]) -> Vec<u32> {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 20, 0];
        w.extend(inst(OP_CAPABILITY, &[1])); // Shader
        w.extend(inst(OP_MEMORY_MODEL, &[0, 1]));
        let mut entry = vec![5, 4]; // GLCompute, %4
        entry.extend(literal_string("main"));
        w.extend(inst(OP_ENTRY_POINT, &entry));
        w.extend(inst(OP_EXECUTION_MODE, &[4, 17, 64, 1, 1])); // LocalSize
        for (mode, width) in extra {
            w.extend(inst(OP_EXECUTION_MODE, &[4, *mode, *width]));
        }
        w.extend(inst(OP_DECORATE, &[9, 30])); // some unrelated decoration
        w.extend(inst(133, &[6, 7, 8, 8])); // OpFMul %6 %7 = %8 * %8
        w
    }

    fn minimal_module() -> Vec<u32> {
        module_with_modes(&[])
    }

    #[test]
    fn parses_and_rejects_garbage() {
        assert!(instructions(&minimal_module()).is_some());
        assert!(instructions(&[0, 0, 0]).is_none());
        assert!(
            instructions(&[MAGIC, 0, 0, 0, 0, 0]).is_none(),
            "zero length"
        );
        assert!(
            instructions(&[MAGIC, 0, 0, 0, 0, 99 << 16]).is_none(),
            "length past the end"
        );
    }

    #[test]
    fn round_trips_literal_strings() {
        for s in ["main", "abc", "abcd", "SPV_KHR_float_controls"] {
            assert_eq!(decode_string(&literal_string(s)), s);
        }
        // "abcd" needs a whole extra word for the terminator.
        assert_eq!(literal_string("abcd").len(), 2);
    }

    #[test]
    fn reads_entry_points() {
        assert_eq!(spirv_entry_points(&minimal_module()), ["main"]);
    }

    #[test]
    fn patch_adds_modes_capabilities_extension_and_decoration() {
        let m = minimal_module();
        assert!(!spirv_has_execution_mode(&m, EXEC_MODE_ROUNDING_MODE_RTE));
        assert_eq!(spirv_no_contraction_count(&m), 0);
        assert_eq!(spirv_float_arithmetic_count(&m), 1);

        let p = apply_exec_modes(&m, ExecModes::deterministic()).unwrap();
        for mode in [
            EXEC_MODE_DENORM_FLUSH_TO_ZERO,
            EXEC_MODE_ROUNDING_MODE_RTE,
            EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE,
        ] {
            assert!(spirv_has_execution_mode(&p, mode), "missing mode {mode}");
        }
        assert_eq!(spirv_no_contraction_count(&p), 1);
        assert_eq!(spirv_entry_points(&p), ["main"]);

        // Still a well-formed instruction stream, and the extension is declared once.
        let insts = instructions(&p).unwrap();
        assert_eq!(
            insts
                .iter()
                .filter(|i| i.op == OP_EXTENSION
                    && decode_string(&p[i.start + 1..i.end()]) == FLOAT_CONTROLS_EXTENSION)
                .count(),
            1
        );
        // Capabilities precede extensions, which precede execution modes.
        let pos = |op: u16| insts.iter().position(|i| i.op == op).unwrap();
        assert!(pos(OP_CAPABILITY) < pos(OP_EXTENSION));
        assert!(pos(OP_EXTENSION) < pos(OP_EXECUTION_MODE));
    }

    #[test]
    fn patch_is_idempotent() {
        let once = apply_exec_modes(&minimal_module(), ExecModes::deterministic()).unwrap();
        let twice = apply_exec_modes(&once, ExecModes::deterministic()).unwrap();
        assert_eq!(once, twice);
    }

    /// Review `docs/reviews/M4.md` S-5: the old "already present?" test compared the mode
    /// word only, so a mode declared for f16 suppressed the requested f32 one.
    #[test]
    fn a_mode_at_another_width_does_not_suppress_the_requested_one() {
        let w = module_with_modes(&[(EXEC_MODE_DENORM_FLUSH_TO_ZERO, 16)]);
        let p = apply_exec_modes(&w, ExecModes::deterministic()).unwrap();
        let insts = instructions(&p).unwrap();
        let widths: Vec<u32> = declared_modes(&p, &insts)
            .into_iter()
            .filter(|(_, mode, _)| *mode == EXEC_MODE_DENORM_FLUSH_TO_ZERO)
            .map(|(_, _, width)| width)
            .collect();
        assert_eq!(
            widths,
            [16, 32],
            "the f32 mode must be added next to the f16 one"
        );
    }

    /// Two entry points: every mode lands on both, and the patch stays idempotent.
    #[test]
    fn every_entry_point_gets_every_mode() {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 20, 0];
        w.extend(inst(OP_CAPABILITY, &[1]));
        w.extend(inst(OP_MEMORY_MODEL, &[0, 1]));
        for (id, name) in [(4u32, "main"), (5, "other")] {
            let mut entry = vec![5, id];
            entry.extend(literal_string(name));
            w.extend(inst(OP_ENTRY_POINT, &entry));
        }
        w.extend(inst(OP_EXECUTION_MODE, &[4, 17, 64, 1, 1])); // LocalSize, three operands
        w.extend(inst(133, &[6, 7, 8, 8])); // OpFMul

        let p = apply_exec_modes(&w, ExecModes::deterministic()).unwrap();
        assert_eq!(spirv_entry_points(&p), ["main", "other"]);
        let insts = instructions(&p).unwrap();
        let declared = declared_modes(&p, &insts);
        for entry in [4, 5] {
            for mode in [
                EXEC_MODE_DENORM_FLUSH_TO_ZERO,
                EXEC_MODE_ROUNDING_MODE_RTE,
                EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE,
            ] {
                assert!(
                    declared.contains(&(entry, mode, 32)),
                    "entry %{entry} is missing mode {mode}"
                );
            }
        }
        assert_eq!(
            apply_exec_modes(&p, ExecModes::deterministic()).unwrap(),
            p,
            "still idempotent with two entry points"
        );
    }

    /// A contradiction is refused, not patched: `DenormPreserve 32` plus the requested
    /// `DenormFlushToZero 32` would be invalid SPIR-V.
    #[test]
    fn patch_refuses_a_conflicting_mode_at_the_same_width() {
        let w = module_with_modes(&[(EXEC_MODE_DENORM_PRESERVE, 32)]);
        let err = apply_exec_modes(&w, ExecModes::deterministic()).unwrap_err();
        assert!(
            err.to_string()
                .contains(&EXEC_MODE_DENORM_PRESERVE.to_string()),
            "unexpected error: {err}"
        );

        // The same mode at another width is not a conflict: f16 preserve, f32 flush.
        let ok = module_with_modes(&[(EXEC_MODE_DENORM_PRESERVE, 16)]);
        assert!(apply_exec_modes(&ok, ExecModes::deterministic()).is_ok());
    }

    #[test]
    fn patch_with_no_modes_changes_nothing() {
        let m = minimal_module();
        assert_eq!(apply_exec_modes(&m, ExecModes::none()).unwrap(), m);
    }

    #[test]
    fn patch_rejects_a_module_without_an_entry_point() {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 20, 0];
        w.extend(inst(OP_CAPABILITY, &[1]));
        assert!(apply_exec_modes(&w, ExecModes::deterministic()).is_err());
        assert!(apply_exec_modes(&[0, 0, 0], ExecModes::none()).is_err());
    }

    #[test]
    fn patch_works_without_an_annotation_section() {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 20, 0];
        w.extend(inst(OP_CAPABILITY, &[1]));
        w.extend(inst(OP_MEMORY_MODEL, &[0, 1]));
        let mut entry = vec![5, 4];
        entry.extend(literal_string("main"));
        w.extend(inst(OP_ENTRY_POINT, &entry));
        w.extend(inst(22, &[10])); // OpTypeFloat
        w.extend(inst(129, &[10, 11, 12, 12])); // OpFAdd
        let p = apply_exec_modes(&w, ExecModes::deterministic()).unwrap();
        assert_eq!(spirv_no_contraction_count(&p), 1);
        let insts = instructions(&p).unwrap();
        // The decoration must land before the type it cannot follow.
        let dec = insts.iter().position(|i| i.op == OP_DECORATE).unwrap();
        let ty = insts.iter().position(|i| i.op == 22).unwrap();
        assert!(dec < ty);
    }
}
