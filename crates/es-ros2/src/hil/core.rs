//! [`HilCore`]: tick binning, the Safety Plane, and the input log
//! (`docs/design/ros2-boundary.md` sections 7.1 and 7.3, spec 9.5, spec 24.2).
//!
//! The control loop is *not* reimplemented here: [`HilCore::tick`] binds the events that
//! arrived since the last tick into the integer inputs of one
//! [`es_runtime_embedded::core_rt::EmbeddedCore::step`], which is the same loop the robot
//! runs. What leaves this type is a [`SafeAction`] and nothing else (INV-12, INV-13); the
//! chunk execution mode is the Deployment IR's `execution`, never the controller's choice.
//!
//! No clock is read here. Every plane input is an integer tick, a microsecond count or an
//! `f64` bit pattern, all of them in the log — that is what makes the replay of a
//! non-deterministic live run byte-identical (spec 3.4, spec 3.5, design note section 7.3).

use es_core::PhysTick;
use es_ir::deployment::DeploymentIr;
use es_runtime_embedded::core_rt::EmbeddedCore;
use es_safety::{ActionChunk, ExecutionMode, Micros, SafeAction, SafetyCounters, SafetyPlane};

use super::log::{self, LogWriter};
use super::stats::HilStats;
use super::wire::Command;
use super::HilError;

/// The external controller replans on its own schedule, so the core's internal chunk cursor
/// is never the thing that asks for a new chunk: only [`EmbeddedCore::step`] is used, and
/// `replan_every` does not enter it.
const REPLAN_EVERY: u64 = 1;

/// The Safety Plane, the input log and the statistics of one HIL run; transport-agnostic.
///
/// [`super::HilLink`] is the UDP front end that feeds it. A ROS 2 front end would reuse this
/// type unchanged (design note section 7.1).
#[derive(Debug)]
pub struct HilCore<const NJ: usize, const H: usize> {
    core: EmbeddedCore<NJ, H>,
    log: LogWriter,
    stats: HilStats,
    mode: ExecutionMode,
    /// Control period in whole microseconds, from the IR's rate.
    period_us: u64,
    /// `ceil(inference_budget / control_period)`: how many ticks a command may lag its
    /// observation before the link counts a deadline miss (design note section 7.3 step 3).
    deadline_ticks: u64,
    deployment_hash: [u8; 32],
    rate_num: u64,
    rate_den: u64,
    /// The command bound into the next tick. Private, and there is no accessor: nothing can
    /// read it back out un-validated.
    pending_cmd: Option<Command<NJ, H>>,
    pending_beat: bool,
    last_obs_tick: Option<PhysTick>,
}

impl<const NJ: usize, const H: usize> HilCore<NJ, H> {
    /// Builds the plane from `ir` and writes the `.eshil` header to `log`.
    ///
    /// Rejects an IR whose joint count or horizon disagrees with `NJ` / `H`, or that its own
    /// validator complains about — [`SafetyPlane::from_ir`] decides, this adds nothing.
    pub fn from_ir(
        ir: &DeploymentIr,
        log: Box<dyn std::io::Write + Send>,
    ) -> Result<Self, HilError> {
        let plane: SafetyPlane<NJ, H> =
            SafetyPlane::from_ir(ir).map_err(|e| HilError::Ir(e.to_string()))?;
        let deployment_hash = ir
            .deployment_hash()
            .map_err(|d| HilError::Ir(format!("{d}")))?;
        let json = serde_json::to_vec(ir).map_err(|e| HilError::Ir(e.to_string()))?;
        let (rate_num, rate_den) = (ir.rate.control.num(), ir.rate.control.den());
        let mut log = LogWriter::new(log);
        log.raw(&log::encode_header(
            NJ,
            H,
            rate_num,
            rate_den,
            &deployment_hash,
            &json,
        ));
        let period_us = ir.rate.control_period().0.max(1);
        Ok(Self {
            core: EmbeddedCore::with_plane(plane, ir.execution, REPLAN_EVERY),
            log,
            stats: HilStats::default(),
            mode: ir.execution,
            period_us,
            deadline_ticks: ir.deadlines.inference_budget.0.div_ceil(period_us),
            deployment_hash,
            rate_num,
            rate_den,
            pending_cmd: None,
            pending_beat: false,
            last_obs_tick: None,
        })
    }

    /// The measured joint state at `now`: seeds the plane's hold target and is logged.
    pub fn observe_state(&mut self, now: PhysTick, q: &[f64; NJ], qd: &[f64; NJ]) {
        self.core.plane_mut().observe_state(q, qd);
        self.log
            .write(log::TAG_OBSERVE, &log::observe_body(now, q, qd));
    }

    /// A heartbeat arrived. It is applied once, at the next [`Self::tick`], because the plane's
    /// clock is the tick counter, not the wall clock (design note section 7.3 step 2).
    pub fn on_heartbeat(&mut self) {
        self.pending_beat = true;
    }

    /// A command arrived. The link only hands over commands whose `seq` strictly increased, so
    /// the last one handed over in a tick window is the highest-`seq` one; any earlier one it
    /// replaces is counted `superseded` (design note section 7.3 step 3).
    pub fn on_command(&mut self, cmd: Command<NJ, H>) {
        if self.pending_cmd.replace(cmd).is_some() {
            self.stats.superseded += 1;
        }
    }

    /// One control tick: bin the events, step the plane, log the inputs and the decision.
    ///
    /// **Cannot fail** (INV-13). A log-write error is latched and reported by [`Self::finish`].
    pub fn tick(&mut self, now: PhysTick) -> SafeAction<NJ> {
        if self.pending_beat {
            self.pending_beat = false;
            self.core.plane_mut().heartbeat(now);
            self.stats.heartbeats += 1;
            self.log
                .write(log::TAG_HEARTBEAT, &log::heartbeat_body(now));
        }
        let chunk = match self.pending_cmd.take() {
            Some(cmd) => {
                self.stats.commands += 1;
                if now.0 > cmd.obs_tick.0.saturating_add(self.deadline_ticks) {
                    self.stats.deadline_miss += 1;
                }
                self.last_obs_tick = Some(cmd.obs_tick);
                Some(ActionChunk::new(cmd.actions, cmd.rows as usize, self.mode))
            }
            None => None,
        };
        let obs_age = Micros(self.last_obs_tick.map_or(0, |t| {
            now.ticks_since(t)
                .unwrap_or(0)
                .saturating_mul(self.period_us)
        }));
        self.log.write(
            log::TAG_STEP,
            &log::step_body(now, obs_age.0, chunk.as_ref()),
        );
        let action = self.core.step(chunk, now, obs_age);
        self.log
            .write(log::TAG_DECISION, &log::decision_body(now, &action));
        self.stats.ticks += 1;
        action
    }

    pub fn stats(&self) -> HilStats {
        self.stats
    }

    /// The Safety Plane's own counters (spec 10.3): what the run did, not what the link saw.
    pub fn counters(&self) -> &SafetyCounters {
        self.core.counters()
    }

    /// The `deployment_hash` a controller must present in its `Hello`.
    pub fn deployment_hash(&self) -> [u8; 32] {
        self.deployment_hash
    }

    /// The control rate as `(num, den)`, for `HelloAck`.
    pub fn control_rate(&self) -> (u64, u64) {
        (self.rate_num, self.rate_den)
    }

    /// Writes the trailer and returns the decisions hash (design note section 7.4).
    pub fn finish(self) -> Result<[u8; 32], HilError> {
        self.log.finish().map_err(HilError::Io)
    }

    pub(crate) fn stats_mut(&mut self) -> &mut HilStats {
        &mut self.stats
    }
}
