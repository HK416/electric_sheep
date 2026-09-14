//! The scripted pick-and-place expert (packet `docs/packets/M5/V1-expert-dataset.md`,
//! design note `docs/design/visible-learning.md` section 5).
//!
//! Two things, both concrete structs -- no new trait, `INV-17` allows seven extension points
//! and this is none of them:
//!
//! * [`so101_ik`], a **closed-form** inverse kinematics for a base yaw plus a planar 3R arm.
//!   SO-101 is exactly that shape (design note section 2.8), so there is no Jacobian, no
//!   iteration and no convergence failure mode. Out of reach or out of joint range is
//!   [`None`] -- never a clamped approximation (spec 17.2 applied to kinematics).
//! * [`ScriptedExpert`], a waypoint state machine over [`Stage`] whose only entropy is the
//!   Task IR's own reset draw, so `(task_hash, seed, episode)` fixes the demonstration.
//!
//! Every length, angle and axis it uses is **derived from the parsed scene** by
//! [`Links::from_scene`] -- walking the body tree from the tool site up to the root and
//! composing the poses the importer read. A transcribed constant is a number nobody can
//! check (design note section 4.2), and `ik_uses_derived_link_lengths` scans this file for
//! one.
//!
//! Transcendentals go through [`es_math::approx`] (spec 3.4): the demonstration's `ctrl` rows
//! land in a dataset whose content hash is compared across machines, and the host `libm` is
//! not the same function on two platforms. That costs `f32` accuracy in the solve -- about
//! ten micrometres at the tool, which `ik_round_trips_through_forward_kinematics` pins
//! against MuJoCo's own forward kinematics.

use es_assets::scene::{ActuatorTarget, Body, JointKind, SceneDesc};
use es_core::StableId;
use es_math::{axis::UP, units::DEG_TO_RAD, Pose, Vec3};
use es_physics_core::backend::{ModelInfo, StateView};

use crate::EnvError;

/// The site whose pose the IK solves for: upstream SO-101's tool frame.
const TOOL_SITE: &str = "gripperframe";

fn unsupported(what: impl std::fmt::Display) -> EnvError {
    EnvError::Unsupported(format!("scripted expert: {what}"))
}

// --- transcendentals --------------------------------------------------------------------------

fn sin(x: f64) -> f64 {
    f64::from(es_math::approx::sin(x as f32))
}

fn cos(x: f64) -> f64 {
    f64::from(es_math::approx::cos(x as f32))
}

fn atan2(y: f64, x: f64) -> f64 {
    f64::from(es_math::approx::atan2(y as f32, x as f32))
}

fn sqrt(x: f64) -> f64 {
    f64::from(es_math::approx::sqrt(x as f32))
}

/// `acos` built from the two `approx` primitives, because `es-math` has no `acos` and adding
/// one is `es-math`'s packet, not this one. Exact at the ends: `x = 1` gives `atan2(0, 1) = 0`.
fn acos(x: f64) -> f64 {
    let x = x.clamp(-1.0, 1.0);
    atan2(sqrt(1.0 - x * x), x)
}

// --- the arm, derived from the scene ------------------------------------------------------------

/// One hinge of the chain, in the world frame at the zero configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Hinge {
    id: StableId,
    axis: Vec3,
    anchor: Vec3,
    range: (f64, f64),
}

/// The arm's geometry, derived from a parsed [`SceneDesc`] -- never literals
/// (design note section 4.2).
///
/// The chain is read as a base yaw (`pan`) plus three parallel pitches in the plane that yaw
/// sweeps, ending at the tool site. Everything below is that plane's 2-D description:
/// `l1`/`l2`/`l3` are the in-plane link lengths, `a1`/`a2`/`a3` the in-plane direction of each
/// link at the zero configuration, `r0`/`z0` the first pitch axis relative to the yaw axis and
/// `lateral` the constant out-of-plane offset of the tool.
#[derive(Clone, Debug, PartialEq)]
pub struct Links {
    /// `[pan, lift, elbow, wrist_flex]` plus any remaining chain hinges (roll, gripper).
    chain: Vec<Hinge>,
    /// A point on the yaw axis, and the yaw axis itself.
    pan_anchor: Vec3,
    pan_sign: f64,
    /// Azimuth of the arm plane at the zero configuration.
    psi0: f64,
    /// In-plane radial / vertical offset of the first pitch axis from the yaw axis.
    r0: f64,
    z0: f64,
    /// Signed distance from the yaw axis to the tool, along the pitch axis.
    lateral: f64,
    /// Upper arm, lower arm, and the wrist-plus-gripper offset to the tool site.
    l1: f64,
    l2: f64,
    l3: f64,
    /// In-plane direction of each link at the zero configuration.
    a1: f64,
    a2: f64,
    a3: f64,
}

/// World pose of every body at the zero configuration, by composing parent poses.
fn world_poses(scene: &SceneDesc) -> Result<Vec<(StableId, Pose)>, EnvError> {
    let mut out: Vec<(StableId, Pose)> = Vec::with_capacity(scene.bodies.len());
    // `SceneDesc::bodies` is parent-before-child (the importers walk the tree), so one pass
    // suffices; a body whose parent has not been seen is a malformed scene, not a reorder.
    for body in &scene.bodies {
        let pose = match body.parent {
            None => body.pose,
            Some(parent) => {
                let p = out
                    .iter()
                    .find(|(id, _)| *id == parent)
                    .ok_or_else(|| unsupported(format!("body {} precedes its parent", body.name)))?
                    .1;
                p.compose(body.pose)
            }
        };
        out.push((body.id, pose));
    }
    Ok(out)
}

impl Links {
    /// Reads the arm out of `scene`: the body carrying the `gripperframe` site is the tool,
    /// and the chain is its ancestry up to the root.
    pub fn from_scene(scene: &SceneDesc) -> Result<Self, EnvError> {
        let world = world_poses(scene)?;
        let pose_of = |id: StableId| {
            world
                .iter()
                .find(|(b, _)| *b == id)
                .map_or(Pose::IDENTITY, |(_, p)| *p)
        };
        let tool_body = scene
            .bodies
            .iter()
            .find(|b| b.sites.iter().any(|s| s.name == TOOL_SITE))
            .ok_or_else(|| unsupported(format!("no body carries a site named `{TOOL_SITE}`")))?;
        let site = tool_body
            .sites
            .iter()
            .find(|s| s.name == TOOL_SITE)
            .expect("just found");
        let tool = pose_of(tool_body.id).transform_point(site.pose.position);

        // Root-first ancestry of the tool body.
        let mut ancestry: Vec<&Body> = Vec::new();
        let mut cursor = Some(tool_body);
        while let Some(body) = cursor {
            ancestry.push(body);
            cursor = body
                .parent
                .and_then(|p| scene.bodies.iter().find(|b| b.id == p));
        }
        ancestry.reverse();

        // The chain is the tool body's ancestry, plus the bodies hanging off it: the gripper's
        // own joint moves a jaw *below* the tool frame, not above it.
        let jaws = scene
            .bodies
            .iter()
            .filter(|b| b.parent == Some(tool_body.id));
        let mut chain: Vec<Hinge> = Vec::new();
        for body in ancestry.iter().copied().chain(jaws) {
            for joint in scene.joints.iter().filter(|j| j.body == body.id) {
                if joint.kind != JointKind::Hinge {
                    return Err(unsupported(format!(
                        "joint `{}` is not a hinge; the closed form is a yaw plus a planar 3R",
                        joint.name
                    )));
                }
                let pose = pose_of(body.id);
                chain.push(Hinge {
                    id: joint.id,
                    axis: pose.orientation.rotate(joint.axis).normalize(),
                    anchor: pose.transform_point(joint.anchor),
                    range: joint.range.ok_or_else(|| {
                        unsupported(format!("joint `{}` has no range", joint.name))
                    })?,
                });
            }
        }
        if chain.len() < 4 {
            return Err(unsupported(format!(
                "the tool's chain has {} hinges; the closed form needs a yaw and three pitches",
                chain.len()
            )));
        }

        let up = UP;
        let pitch = chain[1].axis;
        for h in &chain[2..4] {
            if (h.axis.dot(pitch).abs() - 1.0).abs() > 1e-6 {
                return Err(unsupported(
                    "the three pitch axes are not parallel; this is not a planar 3R arm",
                ));
            }
        }
        if chain[0].axis.dot(pitch).abs() > 1e-6 {
            return Err(unsupported(
                "the yaw axis is not perpendicular to the pitches",
            ));
        }
        // The plane's radial direction at the zero configuration, and the sign a positive yaw
        // turns it by (the axis may point down, as SO-101's does).
        let radial = pitch.cross(up);
        if radial.norm() < 1e-9 {
            return Err(unsupported("the pitch axis is vertical"));
        }
        let radial = radial.normalize();
        let plane = |v: Vec3| (v.dot(radial), v.dot(up));
        let link = |from: Vec3, to: Vec3| {
            let (dr, dz) = plane(to - from);
            (sqrt(dr * dr + dz * dz), atan2(dz, dr))
        };
        let pan_anchor = chain[0].anchor;
        let (r0, z0) = plane(chain[1].anchor - pan_anchor);
        let (l1, a1) = link(chain[1].anchor, chain[2].anchor);
        let (l2, a2) = link(chain[2].anchor, chain[3].anchor);
        let (l3, a3) = link(chain[3].anchor, tool);
        Ok(Self {
            pan_anchor,
            pan_sign: if chain[0].axis.dot(up) < 0.0 {
                1.0
            } else {
                -1.0
            },
            psi0: atan2(radial.y, radial.x),
            r0,
            z0,
            lateral: (tool - pan_anchor).dot(pitch),
            l1,
            l2,
            l3,
            a1,
            a2,
            a3,
            chain,
        })
    }

    /// The chain's joint ids in order: yaw, the three pitches, then whatever follows
    /// (`wrist_roll` and the gripper on SO-101).
    pub fn joints(&self) -> Vec<StableId> {
        self.chain.iter().map(|h| h.id).collect()
    }

    /// `(lo, hi)` of the `i`-th chain joint.
    pub fn range(&self, i: usize) -> Option<(f64, f64)> {
        self.chain.get(i).map(|h| h.range)
    }

    /// The farthest the tool can be from the yaw axis, in the arm plane.
    pub fn max_reach(&self) -> f64 {
        self.l1 + self.l2 + self.l3
    }
}

/// Closed-form IK for a base yaw plus a planar 3R, solved for the tool site at `target` with
/// the final link at `approach_pitch` (radians, `-pi/2` is straight down).
///
/// `None` -- never an approximation (spec 17.2) -- when the target is out of reach, inside the
/// arm's lateral offset, or when any joint would leave its range. `wrist_roll` is not solved
/// for: the caller holds it at zero, which is what makes the tool's lateral offset constant.
#[must_use]
pub fn so101_ik(links: &Links, target: Vec3, approach_pitch: f64) -> Option<[f64; 4]> {
    let dx = target.x - links.pan_anchor.x;
    let dy = target.y - links.pan_anchor.y;
    let dz = target.z - links.pan_anchor.z;
    let rho2 = dx * dx + dy * dy;
    if rho2 < links.lateral * links.lateral {
        return None;
    }
    // The arm plane is offset from the yaw axis by `lateral`, so the azimuth the yaw must take
    // is the target's azimuth minus the angle that offset subtends.
    let radial = sqrt(rho2 - links.lateral * links.lateral);
    let azimuth = atan2(dy, dx) - atan2(links.lateral, radial);
    let pan = links.pan_sign * (links.psi0 - azimuth);

    // 2-D problem in the plane, measured from the first pitch axis: the wrist centre is the
    // target minus the last link along the approach direction.
    let (wrist_r, wrist_z) = (
        radial - links.r0 - links.l3 * cos(approach_pitch),
        dz - links.z0 - links.l3 * sin(approach_pitch),
    );
    let chord2 = wrist_r * wrist_r + wrist_z * wrist_z;
    let chord = sqrt(chord2);
    if chord > links.l1 + links.l2 || chord < (links.l1 - links.l2).abs() {
        return None;
    }
    // Elbow-up: the only branch SO-101's joint ranges admit (design note section 5.2).
    let upper = atan2(wrist_z, wrist_r)
        + acos((chord2 + links.l1 * links.l1 - links.l2 * links.l2) / (2.0 * chord * links.l1));
    let lower = atan2(
        wrist_z - links.l1 * sin(upper),
        wrist_r - links.l1 * cos(upper),
    );
    let lift = links.a1 - upper;
    let elbow = links.a2 - lift - lower;
    let flex = links.a3 - lift - elbow - approach_pitch;

    let out = [pan, lift, elbow, flex];
    for (i, q) in out.iter().enumerate() {
        let (lo, hi) = links.range(i)?;
        if !q.is_finite() || *q < lo || *q > hi {
            return None;
        }
    }
    Some(out)
}

// --- the waypoint state machine -----------------------------------------------------------------

/// Where the demonstration is (design note section 5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Above the cube, gripper open.
    Approach,
    /// Down onto the cube.
    Descend,
    /// Squeeze, for `close_ticks` control steps.
    Close,
    /// Straight up to carry height.
    Lift,
    /// Across to the bin, at carry height.
    Transport,
    /// Down into the bin. Without it the cube is released above the wall top, bounces out as
    /// often as not, and -- because the Task IR's success predicate cannot see `z` -- the
    /// episode ends with the cube still in the air (design note section 5.4).
    Lower,
    /// Open, and let it drop.
    Release,
    /// Hold the last command; the episode ends on the Task IR's own `Terminate` node.
    Done,
}

impl Stage {
    fn next(self) -> Self {
        match self {
            Self::Approach => Self::Descend,
            Self::Descend => Self::Close,
            Self::Close => Self::Lift,
            Self::Lift => Self::Transport,
            Self::Transport => Self::Lower,
            Self::Lower => Self::Release,
            Self::Release | Self::Done => Self::Done,
        }
    }
}

/// What the demonstration aims at, in metres and radians.
///
/// `cube_joint` is the cube's **free joint**, not its body: the collector hands a scripted
/// intervener the `qpos ‖ qvel` observation row, which has no `xpos` in it, and a free joint's
/// `qpos` is its world pose exactly (design note section 5.4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertCfg {
    pub cube_joint: StableId,
    /// Where the cube is dropped; `z` is the height the arm carries at, above the bin's walls.
    pub bin_center: Vec3,
    /// Tool height at the release, inside the bin: low enough that the cube is in the bin's
    /// volume when the jaws open, not above it.
    pub drop_height: f64,
    pub hover_height: f64,
    /// How far below the cube's centre the tool tip goes to grip it.
    pub grasp_depth: f64,
    /// Tool pitch for the approach and the grasp; `-pi/2` is straight down.
    pub approach_pitch: f64,
    /// Tool pitch while carrying. SO-101 cannot hold `-pi/2` at carry height
    /// (design note section 5.2), so this is a second angle rather than a clamp.
    pub carry_pitch: f64,
    pub grip_open: f64,
    pub grip_closed: f64,
    pub close_ticks: u32,
    /// Joint error, rad, at which a stage counts as reached.
    pub pos_tol: f64,
    /// How far a commanded joint may move in one control tick, rad, and how much that step may
    /// change from tick to tick. A demonstration that steps straight to its waypoint is
    /// clamped by the Safety Plane at every tick and trips the envelope-violation watchdog
    /// (measured, packet M5/V1), so the expert paces itself to what the envelope allows; the
    /// caller reads both out of the Deployment IR the demonstration is recorded through.
    pub step_max: f64,
    pub accel_max: f64,
    /// Rows per action chunk: the deployment's `horizon`.
    pub horizon: u32,
    /// Rows of each chunk that actually execute before the next one: the deployment's
    /// `execute_chunk`. The expert continues the next ramp from there, because that is where
    /// the *command* got to -- starting again from the measured joints would step backwards by
    /// the servo's following error, which the Safety Plane sees as a violation.
    pub execute: u32,
}

/// One scripted demonstration, stage by stage.
///
/// A concrete struct reached through the intervener hook `es-data`'s collector already takes,
/// so a demonstration is recorded by machinery that already exists -- and passes through the
/// same Safety Plane as a policy chunk (`INV-12`).
#[derive(Clone, Debug)]
pub struct ScriptedExpert {
    cfg: ExpertCfg,
    links: Links,
    stage: Stage,
    stage_tick: u32,
    /// The cube pose the grasp was planned from, latched when the episode starts: once the
    /// jaws touch it the cube moves, and re-planning from the moved cube chases it.
    grasp: Option<Vec3>,
    /// Actuator ids in chain order, so `action` writes to the right `ctrl` slots.
    actuators: Vec<StableId>,
    /// Where the command got to, and how fast it was moving, per chain joint. Empty until the
    /// first step of an episode, which starts it at the measured joints.
    command: Vec<f64>,
    speed: Vec<f64>,
}

impl ScriptedExpert {
    pub fn new(scene: &SceneDesc, cfg: ExpertCfg) -> Result<Self, EnvError> {
        let links = Links::from_scene(scene)?;
        let joints = links.joints();
        let mut actuators = Vec::with_capacity(joints.len());
        for joint in &joints {
            let actuator = scene
                .actuators
                .iter()
                .find(|a| a.target == ActuatorTarget::Joint(*joint))
                .ok_or_else(|| unsupported(format!("joint {joint} has no actuator")))?;
            actuators.push(actuator.id);
        }
        Ok(Self {
            cfg,
            links,
            stage: Stage::Approach,
            stage_tick: 0,
            grasp: None,
            actuators,
            command: Vec::new(),
            speed: Vec::new(),
        })
    }

    /// Back to [`Stage::Approach`] with no latched cube pose: call it once per episode.
    pub fn reset(&mut self) {
        self.stage = Stage::Approach;
        self.stage_tick = 0;
        self.grasp = None;
        self.command.clear();
        self.speed.clear();
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    pub fn links(&self) -> &Links {
        &self.links
    }

    /// The tool target, pitch and gripper command of `stage`.
    fn waypoint(&self, cube: Vec3) -> (Vec3, f64, f64) {
        let cfg = &self.cfg;
        let grip = cfg.grip_closed;
        match self.stage {
            Stage::Approach => (
                Vec3::new(cube.x, cube.y, cube.z + cfg.hover_height),
                cfg.approach_pitch,
                cfg.grip_open,
            ),
            Stage::Descend => (
                Vec3::new(cube.x, cube.y, cube.z + cfg.grasp_depth),
                cfg.approach_pitch,
                cfg.grip_open,
            ),
            Stage::Close => (
                Vec3::new(cube.x, cube.y, cube.z + cfg.grasp_depth),
                cfg.approach_pitch,
                grip,
            ),
            Stage::Lift => (
                Vec3::new(cube.x, cube.y, cfg.bin_center.z),
                cfg.carry_pitch,
                grip,
            ),
            Stage::Transport => (cfg.bin_center, cfg.carry_pitch, grip),
            Stage::Lower => (
                Vec3::new(cfg.bin_center.x, cfg.bin_center.y, cfg.drop_height),
                cfg.approach_pitch,
                grip,
            ),
            Stage::Release | Stage::Done => (
                Vec3::new(cfg.bin_center.x, cfg.bin_center.y, cfg.drop_height),
                cfg.approach_pitch,
                cfg.grip_open,
            ),
        }
    }

    /// One control step of demonstration: the first row of [`chunk`](Self::chunk).
    pub fn action(
        &mut self,
        model: &ModelInfo,
        state: &StateView<'_>,
        env: u32,
    ) -> Option<Vec<f64>> {
        self.chunk(model, state, env)?.into_iter().next()
    }

    /// `horizon` control steps of demonstration, or [`None`] when this stage's waypoint is out
    /// of reach -- which ends the episode as a failed demonstration rather than driving
    /// somewhere close (spec 17.2).
    ///
    /// The rows ramp from where the joints are *now* toward the waypoint, paced by `step_max`
    /// and `accel_max`, so the Safety Plane has nothing to clamp: a demonstration the envelope
    /// corrects at every tick is a demonstration of the envelope, not of the task. The pace
    /// starts from the measured joint velocity, so consecutive chunks join without a step.
    pub fn chunk(
        &mut self,
        model: &ModelInfo,
        state: &StateView<'_>,
        env: u32,
    ) -> Option<Vec<Vec<f64>>> {
        let cube = self.cube_pose(model, state, env)?;
        let grasp = *self.grasp.get_or_insert(cube);
        let (target, pitch, grip) = self.waypoint(grasp);
        let solution = so101_ik(&self.links, target, pitch)?;

        let qpos = state.qpos_of(env);
        let horizon = self.cfg.horizon.max(1) as usize;
        let executed = (self.cfg.execute.max(1) as usize).min(horizon);
        if self.command.len() != self.actuators.len() {
            // The first step of an episode: the command starts where the joints are.
            self.command = Vec::with_capacity(self.actuators.len());
            for hinge in &self.links.chain {
                self.command
                    .push(*qpos.get(model.qpos.get(&hinge.id)?.start as usize)?);
            }
            self.speed = vec![0.0; self.actuators.len()];
        }
        let mut rows = vec![vec![0.0; model.nu as usize]; horizon];
        let mut reached = true;
        for (i, actuator) in self.actuators.iter().enumerate() {
            let want = match i {
                0..=3 => solution[i],
                // `wrist_roll` is held at zero, which is what keeps the tool's lateral offset
                // constant; the last chain joint is the gripper.
                _ if i + 1 == self.actuators.len() => grip,
                _ => 0.0,
            };
            let joint = self.links.chain[i].id;
            let at = *qpos.get(model.qpos.get(&joint)?.start as usize)?;
            let slot = model.actuator.get(actuator)?.start as usize;
            // Anti-windup. The chunk is executed with an inference latency, so a command that
            // only ever integrates forward drifts ahead of the joint it commands; the Safety
            // Plane then clamps every tick and the envelope-violation watchdog latches
            // (measured, packet M5/V1). The command may lead the measured joint, but only by
            // a few steps.
            let lead = 4.0 * self.cfg.step_max;
            self.command[i] = self.command[i].clamp(at - lead, at + lead);
            let ramp = self.ramp(self.command[i], self.speed[i], want, horizon);
            for (k, q) in ramp.iter().enumerate() {
                *rows.get_mut(k)?.get_mut(slot)? = *q;
            }
            let last = ramp[executed - 1];
            self.speed[i] = last
                - if executed > 1 {
                    ramp[executed - 2]
                } else {
                    self.command[i]
                };
            self.command[i] = last;
            if i < 4 {
                reached &= (at - want).abs() <= self.cfg.pos_tol;
            }
        }

        self.stage_tick += 1;
        let advance = match self.stage {
            Stage::Close | Stage::Release => self.stage_tick >= self.cfg.close_ticks,
            Stage::Done => false,
            _ => reached,
        };
        if advance {
            self.stage = self.stage.next();
            self.stage_tick = 0;
        }
        Some(rows)
    }

    /// `horizon` commanded positions ramping from `at`, already moving at `speed` rad per
    /// control tick, toward `want`: bounded by `step_max` and `accel_max`, and slowing in time
    /// to stop on the target rather than overshooting it.
    fn ramp(&self, at: f64, speed: f64, want: f64, horizon: usize) -> Vec<f64> {
        let (step_max, accel) = (self.cfg.step_max.abs(), self.cfg.accel_max.abs());
        let mut q = at;
        let mut v = speed.clamp(-step_max, step_max);
        let mut out = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let remaining = want - q;
            // The fastest step this joint can still stop on the target from.
            let stopping = sqrt(2.0 * accel * remaining.abs());
            let wanted = remaining.signum() * step_max.min(stopping);
            v = wanted.clamp(v - accel, v + accel);
            if v.abs() > remaining.abs() {
                v = remaining;
            }
            q += v;
            out.push(q);
        }
        out
    }

    /// The cube's world position, read from its free joint's `qpos` (the first three of the
    /// seven values `MuJoCo` stores for a free joint).
    fn cube_pose(&self, model: &ModelInfo, state: &StateView<'_>, env: u32) -> Option<Vec3> {
        let range = model.qpos.get(&self.cfg.cube_joint)?;
        if range.len < 3 {
            return None;
        }
        let row = state.qpos_of(env);
        let at = range.start as usize;
        Some(Vec3::new(
            *row.get(at)?,
            *row.get(at + 1)?,
            *row.get(at + 2)?,
        ))
    }
}

/// A `StateView` over one env's `qpos ‖ qvel` row -- the raw observation a scripted intervener
/// is handed (`crates/es-env/src/domains.rs:374-392`). `xpos` is deliberately empty: the
/// expert reads joints, not body rows.
#[must_use]
pub fn state_of_row<'a>(model: &ModelInfo, row: &'a [f64]) -> StateView<'a> {
    let nq = (model.nq as usize).min(row.len());
    StateView {
        n_envs: 1,
        qpos: &row[..nq],
        qvel: &row[nq..],
        ..StateView::default()
    }
}

/// The demo's own tuning, measured on the oracle server against
/// `tests/fixtures/mjcf/so101_pick_place.xml` (packet M5/V1): these values put every cube the
/// Task IR's `Randomization` node can draw into the bin.
///
/// Not a default `impl`: they describe *this* scene, and a second scene wants its own.
#[must_use]
pub fn demo_cfg(cube_joint: StableId) -> ExpertCfg {
    ExpertCfg {
        cube_joint,
        // The bin's interior centre, at the height the arm carries and releases at -- above
        // the wall top, so the carried cube clears it on the way in.
        bin_center: Vec3::new(0.14, -0.10, 0.14),
        drop_height: 0.06,
        hover_height: 0.045,
        // Five millimetres below the cube's centre: the jaws grip a band around their own
        // tips, and the cube is 40 mm tall.
        grasp_depth: -0.005,
        // Not straight down: at -90 degrees the far half of the cube's draw leaves
        // `wrist_flex`'s range, and the closed form refuses rather than approximating.
        approach_pitch: -85.0 * DEG_TO_RAD,
        carry_pitch: -45.0 * DEG_TO_RAD,
        // Wide open, not just wide enough: a jaw holding the 30 mm cube stalls at about 0.30,
        // so an opening the Task IR's success predicate can tell apart from "still holding it"
        // has to be well clear of that (packet M5/V1).
        grip_open: 0.9,
        grip_closed: -0.05,
        close_ticks: 25,
        // 0.01 rad is about two millimetres at the tool. Looser than that and the arm starts
        // its descent while still a jaw-clearance away from over the cube, and shoves it
        // (measured, packet M5/V1).
        pos_tol: 0.01,
        // What `tests/fixtures/visible-learning/deployment.toml` allows at 50 Hz with a tenth
        // held back: velocity 3 rad/s is 0.06 rad a tick, acceleration 20 rad/s^2 is 0.008.
        step_max: 0.054,
        accel_max: 0.0072,
        horizon: 16,
        execute: 10,
    }
}

// Latching and reachability are exact properties here: a tolerance would be the bug.
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> SceneDesc {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/mjcf/so101_pick_place.xml");
        let xml = std::fs::read_to_string(&path).expect("the V0 fixture is in the repo");
        es_assets::parse_mjcf(&xml)
            .expect("the V0 fixture parses")
            .scene
    }

    fn links() -> Links {
        Links::from_scene(&fixture()).expect("the fixture is a yaw plus a planar 3R")
    }

    /// The numbers come out of the parse, not out of this file: a second parse gives the same
    /// `Links`, and none of them is written down anywhere in `expert.rs`.
    #[test]
    fn ik_uses_derived_link_lengths() {
        assert_eq!(links(), links());
        let l = links();
        assert!(l.l1 > 0.0 && l.l2 > 0.0 && l.l3 > 0.0, "{l:?}");
        assert_eq!(l.joints().len(), 6, "six hinges from base to gripper");

        // A source scan: no length from the model may appear as a literal in the solver.
        // `demo_cfg` below it is deliberately excluded -- those numbers are measured tuning
        // for this scene (hover heights, pitches, a pace), not geometry that can be derived,
        // and the packet requires them to be written down somewhere.
        let whole = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/expert.rs"),
        )
        .expect("read own source");
        let src = whole
            .split_once("pub fn demo_cfg")
            .expect("demo_cfg is in this file")
            .0;
        for value in [l.l1, l.l2, l.l3, l.r0, l.z0, l.lateral, l.a1, l.a2, l.a3] {
            for digits in 3..=6 {
                let text = format!("{:.*}", digits, value.abs());
                let bare = text.trim_start_matches("0.");
                assert!(
                    bare.len() < 3 || !src.contains(bare),
                    "a derived value ({text}) is transcribed into expert.rs"
                );
            }
        }
    }

    #[test]
    fn ik_refuses_rather_than_clamps() {
        let l = links();
        let down = -std::f64::consts::FRAC_PI_2;
        // Beyond the reach.
        assert_eq!(so101_ik(&l, Vec3::new(1.0, 0.0, 0.2), down), None);
        // Behind the arm at full stretch: the yaw cannot turn that far.
        assert_eq!(so101_ik(&l, Vec3::new(-0.24, 0.0, 0.02), down), None);
        // Straight up out of the pitch range.
        assert_eq!(so101_ik(&l, Vec3::new(0.05, 0.0, 0.45), down), None);

        // Nothing that does come back is outside a range.
        let mut solved = 0;
        for x in [0.14, 0.18, 0.22, 0.26] {
            for y in [-0.08, 0.0, 0.08] {
                for z in [0.015, 0.06, 0.12] {
                    for pitch in [down, -1.2, -0.9] {
                        let Some(q) = so101_ik(&l, Vec3::new(x, y, z), pitch) else {
                            continue;
                        };
                        solved += 1;
                        for (i, v) in q.iter().enumerate() {
                            let (lo, hi) = l.range(i).expect("four ranges");
                            assert!(*v >= lo && *v <= hi, "joint {i} = {v} outside [{lo}, {hi}]");
                        }
                    }
                }
            }
        }
        assert!(solved > 20, "only {solved} of the grid solved");
    }

    /// The elbow-up branch is the one SO-101's ranges admit, and the solution is continuous:
    /// a millimetre of target moves the joints by much less than a radian.
    #[test]
    fn the_solution_is_continuous_in_the_target() {
        let l = links();
        let down = -std::f64::consts::FRAC_PI_2;
        let a = so101_ik(&l, Vec3::new(0.24, 0.0, 0.02), down).expect("reachable");
        let b = so101_ik(&l, Vec3::new(0.241, 0.0, 0.02), down).expect("reachable");
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 0.05, "{a:?} vs {b:?}");
        }
    }

    fn model_of(links: &Links, cube: StableId) -> ModelInfo {
        use es_physics_core::backend::IndexRange;
        let mut model = ModelInfo {
            nq: 13,
            nv: 12,
            nu: 6,
            ..ModelInfo::default()
        };
        for (i, j) in links.joints().iter().enumerate() {
            model.qpos.insert(*j, IndexRange::new(i as u32, 1));
            model.dof.insert(*j, IndexRange::new(i as u32, 1));
        }
        model.qpos.insert(cube, IndexRange::new(6, 7));
        model
    }

    /// `ScriptedExpert` with the fixture's own cube joint and a `ModelInfo` shaped like
    /// `MuJoCo`'s for that scene.
    fn expert() -> (ScriptedExpert, ModelInfo) {
        let scene = fixture();
        let cube = scene
            .joints
            .iter()
            .find(|j| j.name == "cube_free")
            .expect("the fixture has a cube")
            .id;
        let mut cfg = demo_cfg(cube);
        cfg.close_ticks = 3;
        let expert = ScriptedExpert::new(&scene, cfg).expect("the fixture builds an expert");
        let mut model = model_of(expert.links(), cube);
        for (i, a) in scene.actuators.iter().enumerate() {
            model
                .actuator
                .insert(a.id, es_physics_core::backend::IndexRange::new(i as u32, 1));
        }
        (expert, model)
    }

    fn row(cube: Vec3, joints: [f64; 6]) -> Vec<f64> {
        let mut r = vec![0.0; 25];
        r[..6].copy_from_slice(&joints);
        r[6] = cube.x;
        r[7] = cube.y;
        r[8] = cube.z;
        r[9] = 1.0;
        r
    }

    #[test]
    fn the_state_machine_advances_only_on_its_predicate() {
        let (mut expert, model) = expert();
        let cube = Vec3::new(0.24, 0.0, 0.02);
        let frozen = row(cube, [0.0; 6]);
        let state = state_of_row(&model, &frozen);
        assert_eq!(expert.stage(), Stage::Approach);
        for _ in 0..20 {
            let ctrl = expert.action(&model, &state, 0).expect("reachable");
            assert_eq!(ctrl.len(), 6);
            assert_eq!(expert.stage(), Stage::Approach, "a frozen state moved on");
        }

        // A robot that follows the commanded ramp walks the stages, in order, exactly once
        // each. The expert commands a ramp rather than a step (design note section 7.5), so a
        // waypoint takes several steps to reach -- what is pinned here is the order, not the
        // count.
        let mut joints = [0.0; 6];
        let mut seen = vec![expert.stage()];
        for _ in 0..4000 {
            let at = row(cube, joints);
            let ctrl = expert
                .action(&model, &state_of_row(&model, &at), 0)
                .expect("reachable");
            joints = [ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5]];
            if *seen.last().expect("non-empty") != expert.stage() {
                seen.push(expert.stage());
            }
            if expert.stage() == Stage::Done {
                break;
            }
        }
        assert_eq!(
            seen,
            vec![
                Stage::Approach,
                Stage::Descend,
                Stage::Close,
                Stage::Lift,
                Stage::Transport,
                Stage::Lower,
                Stage::Release,
                Stage::Done,
            ]
        );
    }

    #[test]
    fn an_unreachable_cube_is_none_not_an_approximation() {
        let (mut expert, model) = expert();
        let far = row(Vec3::new(0.9, 0.0, 0.02), [0.0; 6]);
        assert_eq!(expert.action(&model, &state_of_row(&model, &far), 0), None);
    }

    #[test]
    fn reset_forgets_the_latched_cube() {
        let (mut expert, model) = expert();
        // The end of the chunk, where the ramp has had time to head somewhere: its first row
        // is one step from wherever the joints are and says little about the target.
        let aim = |expert: &mut ScriptedExpert, row: &[f64]| {
            expert
                .chunk(&model, &state_of_row(&model, row), 0)
                .expect("reachable")
                .pop()
                .expect("a horizon of rows")
        };
        let first = row(Vec3::new(0.24, 0.0, 0.02), [0.0; 6]);
        let second = row(Vec3::new(0.22, 0.06, 0.02), [0.0; 6]);
        let a = aim(&mut expert, &first);
        let b = aim(&mut expert, &second);
        assert_eq!(
            a[0], b[0],
            "the grasp pose is latched for the episode: {a:?} vs {b:?}"
        );
        expert.reset();
        let c = aim(&mut expert, &second);
        assert_ne!(a[0], c[0], "reset re-plans from the new cube");
        assert_eq!(expert.stage(), Stage::Approach);
    }
}
