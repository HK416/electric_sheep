//! The spec 7.4 declaration Task IR hands to Observation IR: [`ObservationSpec`], where each
//! channel comes from ([`ObsSource`]) and how the simulation renders a sensor channel
//! ([`SensorRender`]).

use std::collections::BTreeMap;

use es_core::StableId;
use serde::{Deserialize, Serialize};

use super::{wdbg, wid, JointQuantity};
use crate::hash::CanonWriter;
use crate::image::ChannelFormat;
use crate::types::PortType;

/// Which renderer produces a [`ObsSource::Sensor`] channel (spec 15.3, packet M7/R5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "path", rename_all = "lowercase")]
pub enum SensorPath {
    /// The rasterizer: today's observation path, and the default.
    #[default]
    Rs,
    /// The path tracer, at this many samples and bounces per pixel.
    Pt { spp: u32, bounces: u32 },
}

/// How linear radiance becomes a display value, mirroring `es_render::Tonemap` (which is
/// layer 5 and out of reach here, spec 4.2). `Pt` only: the `Rs` path clamps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tonemap {
    #[default]
    Reinhard,
    Aces,
}

/// Which seed the path tracer's sample keys are addressed from (packet M10/W1a, spec 6).
///
/// The observation path never accumulates, so a `Pt` frame's sample keys are a pure function
/// of `(pixel, pose)` and 64 spp leaves a **fixed grain per pose** — a texture a policy can
/// use as a fingerprint instead of learning the task (`docs/reviews/M7.md` R13,
/// `docs/design/visible-learning.md` 7.32).
///
/// [`Self::Fixed`] is the default, absent is the default, and the default is today's bytes
/// (spec 28.10 rule 1). It is not a quality knob: `Tick` renders the same estimator at the
/// same `spp`, only keyed differently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SeedStream {
    /// One seed for the whole run: a pose always carries the same grain.
    #[default]
    Fixed,
    /// The **episode-relative** tick is mixed into the seed, so the same pose at two ticks
    /// carries different grain while the collector and the evaluator still agree bit for bit
    /// at the same `(episode, tick)` (spec 3.4: the draw stays addressed, never stepped).
    Tick,
}

impl SeedStream {
    /// Whether this is the stream an absent `seed` means — the `skip_serializing_if` and the
    /// canonical-form rule are the same predicate.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed)
    }
}

/// How the simulation *produces* one sensor channel (packet M7/R5).
///
/// Not an `ImageSpec` field and not an Observation IR one: what the sensor **is** —
/// resolution, colour space, intrinsics — is `ImageSpec`'s, and how the simulation **makes**
/// it is the Task IR's. One Observation IR therefore serves an `Rs` and a `Pt` task alike.
///
/// [`Self::default`] is the rasterizer at neutral exposure, and [`ObsSource::canonical`]
/// writes the block **only when it is not the default**, so an absent or default `render` is
/// byte for byte today's canonical form and no committed `task_hash` moves (spec 28.10
/// rule 1). A `Pt` sensor does move it, and is therefore a new document (spec 13.3).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SensorRender {
    #[serde(flatten)]
    pub path: SensorPath,
    /// Linear multiplier applied before [`Self::tonemap`].
    pub exposure: f32,
    pub tonemap: Tonemap,
    /// `Pt` only: which seed the sample keys are addressed from (packet M10/W1a). Absent =
    /// [`SeedStream::Fixed`] = today's bytes, so appending it moved no committed `task_hash`
    /// (`cargo test -p es-ir committed_task_hashes_are_unmoved_by_seed_stream`).
    #[serde(default, skip_serializing_if = "SeedStream::is_fixed")]
    pub seed: SeedStream,
    /// `Pt` only: run the renderer's single-frame SVGF filter on the sensor's frame (packet
    /// M11/X6, spec 28.14 rule 5). No accumulation: without a history the luminance weight is
    /// exactly 1.0, so the frame stays a pure function of `(pose, episode, tick)`. Absent =
    /// `false` = today's bytes; `true` on `Rs` is `TASK-003`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub svgf: bool,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

impl Default for SensorRender {
    fn default() -> Self {
        Self {
            path: SensorPath::Rs,
            exposure: 1.0,
            tonemap: Tonemap::Reinhard,
            seed: SeedStream::Fixed,
            svgf: false,
        }
    }
}

impl SensorRender {
    /// Whether this is the block an absent `render` means. The serde `skip_serializing_if`
    /// and the canonical-form rule are the same predicate, so a document that omits it and a
    /// document that writes it out cannot disagree.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    fn canonical(&self, w: &mut CanonWriter) {
        match self.path {
            SensorPath::Rs => w.str("rs"),
            SensorPath::Pt { spp, bounces } => {
                w.str("pt");
                w.u32(spp);
                w.u32(bounces);
            }
        }
        w.f32(self.exposure);
        wdbg(w, &self.tonemap);
        // Appended, and written **only** when it is not the default: the two committed
        // documents (`task.toml`, `task-pt.toml`) keep their canonical bytes and their
        // `task_hash`es, and `seed = "tick"` is a new document (spec 13.3, 28.10 rule 1).
        if !self.seed.is_fixed() {
            wdbg(w, &self.seed);
        }
        // Packet M11/X6: the same rule, after `seed`, so no committed document moves.
        if self.svgf {
            w.str("svgf");
        }
    }
}

/// Where one declared observation channel comes from (spec 7.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObsSource {
    Sensor {
        id: StableId,
        format: ChannelFormat,
        /// Absent = [`SensorRender::default`] = today's canonical form (packet M7/R5).
        #[serde(default, skip_serializing_if = "SensorRender::is_default")]
        render: SensorRender,
    },
    JointState {
        body: StableId,
        dof: u32,
        /// Which joint quantity the channel carries, mirroring
        /// [`TaskNode::GetJointState`](super::TaskNode::GetJointState)'s own field (spec 6.3).
        ///
        /// Absent = [`JointQuantity::Position`] = today's canonical form (packet M8/S4e,
        /// the rule packet M7/R5 set for [`SensorRender`]), so no committed `task_hash`
        /// moves. A `Velocity` channel does move it, and is therefore a new document
        /// (spec 13.3).
        #[serde(default = "joint_position", skip_serializing_if = "is_joint_position")]
        quantity: JointQuantity,
    },
    BodyPose(StableId),
    Language,
    /// The action row the policy emitted for the previous control tick -- before the Safety
    /// Plane, in actuator units, in actuator order -- and `initial` (absent = zeros) on the
    /// first tick of an episode (packet M11/X2). Isaac Lab's `last_action` and Playground's
    /// `last_act` are this row with the action tail folded out; the importer folds it.
    ///
    /// **The last variant, and it must stay last**: nothing hashes a variant's index, but a
    /// variant inserted above another would reorder every `Debug`-derived listing, and the
    /// rule for an addition is that it leaves every committed document byte-identical.
    PreviousAction {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        initial: Option<Vec<f64>>,
    },
}

impl ObsSource {
    /// The id a [`ObsSource::PreviousAction`] channel is keyed by: the `StateInput.source`
    /// `XIR-002` matches it against and the name of its observation-plan input buffer. One
    /// per task: there is one previous action.
    pub fn previous_action_id() -> StableId {
        StableId::from_path("es.previous_action")
    }
}

fn joint_position() -> JointQuantity {
    JointQuantity::Position
}

/// Whether this is the quantity an absent `quantity` means. The serde `skip_serializing_if`
/// and the canonical-form rule are the same predicate, so a document that omits it and a
/// document that writes it out cannot disagree.
#[allow(clippy::trivially_copy_pass_by_ref)] // serde's `skip_serializing_if` wants `&T`
fn is_joint_position(q: &JointQuantity) -> bool {
    matches!(q, JointQuantity::Position)
}

impl ObsSource {
    fn canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::Sensor { id, format, render } => {
                w.str("Sensor");
                wid(w, id);
                wdbg(w, format);
                // Only when it is not the default: see [`SensorRender`].
                if !render.is_default() {
                    w.str("render");
                    render.canonical(w);
                }
            }
            Self::JointState {
                body,
                dof,
                quantity,
            } => {
                w.str("JointState");
                wid(w, body);
                w.u32(*dof);
                // Only when it is not the default: see the field's own note.
                if !is_joint_position(quantity) {
                    wdbg(w, quantity);
                }
            }
            Self::BodyPose(id) => {
                w.str("BodyPose");
                wid(w, id);
            }
            Self::Language => w.str("Language"),
            Self::PreviousAction { initial } => {
                w.str("PreviousAction");
                let initial = initial.as_deref().unwrap_or_default();
                w.seq(initial.len());
                for v in initial {
                    w.f64(*v);
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObsChannel {
    pub source: ObsSource,
    pub ty: PortType,
}

/// The input contract handed to Observation IR (spec 7.4): named channels with a source and a
/// type, and **nothing else**. Resize, color transform, normalization and the time window are
/// Observation IR's (spec 5.1, rule 6), so this type has no field for any of them — several
/// Observation IRs share one `task_hash` precisely because none of that is declared here.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObservationSpec {
    pub channels: BTreeMap<String, ObsChannel>,
}

impl ObservationSpec {
    pub(super) fn canonical(&self, w: &mut CanonWriter) {
        w.seq(self.channels.len());
        for (name, ch) in &self.channels {
            w.str(name);
            ch.source.canonical(w);
            ch.ty.canonical(w);
        }
    }
}
