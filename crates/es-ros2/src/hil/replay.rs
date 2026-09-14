//! Offline replay of a `.eshil` log (`docs/design/ros2-boundary.md` section 7.5, spec 24.2).
//!
//! The M3 gate: a live HIL run, whose *arrival times* are not deterministic, replays to
//! byte-identical Safety Plane decisions, because the log holds the plane's inputs after tick
//! binning. The plane is rebuilt with [`SafetyPlane::from_ir`] from the IR embedded in the
//! header and driven through the same [`EmbeddedCore::step`] the live run used — same code,
//! not a model of it (spec 9.5).
//!
//! What this does **not** prove: that a physical controller, network or robot produces those
//! inputs, or that a `thumbv7em` build decides identically (spec 28.7 gate 14,
//! `Target / Status: unverified`).

use es_core::PhysTick;
use es_ir::deployment::DeploymentIr;
use es_runtime_embedded::core_rt::EmbeddedCore;
use es_safety::{ActionChunk, ExecutionMode, Micros, SafetyPlane};

use super::log::{self, Cur, HilLogError};

/// The outcome of replaying one log (design note section 7.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayReport {
    /// `Decision` records replayed.
    pub steps: u64,
    /// Every replayed decision matched the logged one, byte for byte.
    pub identical: bool,
    /// `(index, tick)` of the first decision that did not.
    pub first_divergence: Option<(u64, PhysTick)>,
    /// The log's own decisions hash: its trailer, or — for a log without one — the hash of
    /// the `Decision` records it does contain.
    pub live_hash: [u8; 32],
    /// blake3 of the `Decision` records this replay produced.
    pub replay_hash: [u8; 32],
    /// The log ends inside a record or without a trailer; the complete prefix was replayed.
    pub truncated: bool,
}

/// Replays `log_bytes`. `NJ` and `H` must match the log's header ([`HilLogError::Shape`]
/// otherwise), and the embedded IR must still hash to the header's `deployment_hash`
/// ([`HilLogError::DeploymentHash`] otherwise).
pub fn replay<const NJ: usize, const H: usize>(
    log_bytes: &[u8],
) -> Result<ReplayReport, HilLogError> {
    let (header, json) = log::parse_header(log_bytes)?;
    if header.nj as usize != NJ || header.h as usize != H {
        return Err(HilLogError::Shape {
            nj: header.nj,
            h: header.h,
            want_nj: NJ,
            want_h: H,
        });
    }
    let ir: DeploymentIr =
        serde_json::from_slice(json).map_err(|e| HilLogError::Ir(e.to_string()))?;
    // JSON formatting is irrelevant: the canonical hash decides (design note section 7.4).
    if ir
        .deployment_hash()
        .map_err(|d| HilLogError::Ir(format!("{d}")))?
        != header.deployment_hash
    {
        return Err(HilLogError::DeploymentHash);
    }
    let plane: SafetyPlane<NJ, H> =
        SafetyPlane::from_ir(&ir).map_err(|e| HilLogError::Ir(e.to_string()))?;
    // `replan_every` never enters `step`; the external controller decided when to replan.
    let mut core: EmbeddedCore<NJ, H> = EmbeddedCore::with_plane(plane, ir.execution, 1);

    let mut live = blake3::Hasher::new();
    let mut replayed = blake3::Hasher::new();
    let mut steps = 0u64;
    let mut identical = true;
    let mut first_divergence = None;
    let mut trailer_hash = None;
    let mut pending: Option<(PhysTick, u64, Option<ActionChunk<NJ, H>>)> = None;
    let mut i = header.end;

    while i < log_bytes.len() {
        // A record the log ends inside is truncation (a crashed run), not corruption.
        let Some((tag, body, raw, next)) = log::read_record(log_bytes, i) else {
            break;
        };
        i = next;
        match tag {
            log::TAG_OBSERVE => {
                let (q, qd) = decode_observe::<NJ>(body)?;
                core.plane_mut().observe_state(&q, &qd);
            }
            log::TAG_HEARTBEAT => {
                let mut c = Cur::new(body);
                let tick = c
                    .u64()
                    .filter(|_| c.at_end())
                    .ok_or(HilLogError::Record("heartbeat"))?;
                core.plane_mut().heartbeat(PhysTick(tick));
            }
            log::TAG_STEP => pending = Some(decode_step::<NJ, H>(body, ir.execution)?),
            log::TAG_DECISION => {
                let (tick, obs_age, chunk) = pending
                    .take()
                    .ok_or(HilLogError::Record("decision without a step"))?;
                let action = core.step(chunk, tick, Micros(obs_age));
                let mine = log::record(log::TAG_DECISION, &log::decision_body(tick, &action));
                live.update(raw);
                replayed.update(&mine);
                if mine != raw && first_divergence.is_none() {
                    identical = false;
                    first_divergence = Some((steps, tick));
                }
                steps += 1;
            }
            log::TAG_STATS => {}
            log::TAG_TRAILER => {
                trailer_hash = Some(
                    body.get(..32)
                        .and_then(|b| <[u8; 32]>::try_from(b).ok())
                        .ok_or(HilLogError::Record("trailer"))?,
                );
                break;
            }
            _ => return Err(HilLogError::Record("unknown record tag")),
        }
    }

    Ok(ReplayReport {
        steps,
        identical,
        first_divergence,
        // The trailer is the live run's own claim about its decisions, so it is preferred over
        // the recomputation: a tampered trailer must show up as a hash mismatch.
        live_hash: trailer_hash.unwrap_or_else(|| *live.finalize().as_bytes()),
        replay_hash: *replayed.finalize().as_bytes(),
        truncated: trailer_hash.is_none(),
    })
}

fn decode_observe<const NJ: usize>(body: &[u8]) -> Result<([f64; NJ], [f64; NJ]), HilLogError> {
    let mut c = Cur::new(body);
    (|| {
        // The tick is recorded for a reader; `observe_state` itself is timeless.
        c.u64()?;
        let q = c.f64s::<NJ>()?;
        let qd = c.f64s::<NJ>()?;
        c.at_end().then_some((q, qd))
    })()
    .ok_or(HilLogError::Record("observe_state"))
}

fn decode_step<const NJ: usize, const H: usize>(
    body: &[u8],
    mode: ExecutionMode,
) -> Result<(PhysTick, u64, Option<ActionChunk<NJ, H>>), HilLogError> {
    let mut c = Cur::new(body);
    (|| {
        let tick = PhysTick(c.u64()?);
        let obs_age = c.u64()?;
        let chunk = if c.u8()? == 0 {
            None
        } else {
            let rows = c.u16()? as usize;
            if rows > H {
                return None;
            }
            let mut actions = [[0.0f64; NJ]; H];
            for row in actions.iter_mut().take(rows) {
                *row = c.f64s::<NJ>()?;
            }
            Some(ActionChunk::new(actions, rows, mode))
        };
        c.at_end().then_some((tick, obs_age, chunk))
    })()
    .ok_or(HilLogError::Record("step"))
}
