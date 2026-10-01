//! Observation capture: where each plan input's values come from ([`Capture`]), resolved
//! once against the documents and filled every step from the physics state or a frame
//! source, and [`LiveObservation`], the same path for `es loop collect`.

use std::collections::BTreeMap;

use es_compile::{CpuPlan, Home, PlanMode, Tensor, TensorRef};
use es_core::StableId;
use es_env::randomize::RenderOverrides;
use es_ir::observation::{ObservationIr, ObservationNode};
use es_ir::task::TaskIr;
use es_ir::types::ElemType;
use es_physics_core::backend::{ModelInfo, StateView};

use super::CellFrames;
use crate::perturb::LightOverride;
use crate::EvalError;

/// Where an image observation input's pixels come from (§7.2, §10.1).
///
/// Given this episode's lighting, the loaded model and the current state, one camera frame in
/// exactly the dtype, count and layout the plan declared, or the reason there is none. A
/// closure rather than the renderer itself: `es-eval` is layer 10 and `es-render` layer 5, and
/// linking Vulkan here just to *refuse* an image would be the wrong trade — the caller owns
/// `es_env::render::EnvRenderer` (feature `render`) and hands its `frame` in through this.
/// It is not told which image input it serves, so it serves a plan with one camera (`Rollout`);
/// several cameras are a [`ViewSource`].
///
/// The [`LightOverride`] is this cell and episode's draw (§10.2). It is constant for a whole
/// episode, so a caller rebuilds its renderer only when it changes.
///
/// Nothing on this path resamples, converts or reorders: a frame that is not what the plan
/// declared is an error, and every conversion is an Observation IR node (§7.2, `INV-14`).
pub type FrameSource<'a> =
    dyn FnMut(&LightOverride, &ModelInfo, &StateView<'_>) -> Result<Vec<u8>, String> + 'a;

/// A [`FrameSource`] told which image input it serves (packet M15/N2): the observation plan's
/// input name, which is the hex of the `ImageInput`'s sensor id -- the id of the Task IR
/// channel whose camera it is (spec 7.4). An Observation IR with several cameras has several
/// image inputs, and each gets its own camera's frame, never one frame reused.
pub type ViewSource<'a> =
    dyn FnMut(&str, &LightOverride, &ModelInfo, &StateView<'_>) -> Result<Vec<u8>, String> + 'a;

/// A [`ViewSource`] for a whole evaluation: handed each episode's [`RenderOverrides`] — the
/// Task IR's render draws `Env::render_overrides` recorded at reset, with the suite's
/// [`LightOverride`] folded into its `light` (packet M11/R2, spec 28.14 rule 4) — so the
/// frame at `(seed, episode, tick)` is the one `es loop collect --frames` and `Rollout` render.
///
/// A task that declares no render target draws the identity, and the overrides are then the
/// suite's light and nothing else: exactly what the frame source was handed before R2.
pub type DrawnFrameSource<'a> =
    dyn FnMut(&str, &RenderOverrides, &ModelInfo, &StateView<'_>) -> Result<Vec<u8>, String> + 'a;

/// The plan's input buffers, filled from the physics state.
///
/// Returns the descriptors and the owned bytes separately so the caller can build the
/// borrowed `TensorRef`s over them, plus the index of the frame this step wrote, if any.
pub type Captured = (Vec<(String, ElemType, Vec<u64>)>, Vec<Vec<u8>>, Option<u64>);

/// Where one plan input's values come from (§7.4, §10.1).
///
/// Resolved once, against the *documents* rather than guessed per step: an input is an image
/// because the Observation IR says `ImageInput`, not because nothing else matched it.
#[derive(Clone, Copy, Debug)]
pub enum Capture {
    /// One joint's `qpos` range in the loaded model.
    Qpos(es_physics_core::backend::IndexRange),
    Sensor(es_physics_core::backend::IndexRange),
    /// A Task IR `ObsSource::JointState { body, dof }` channel: the leading `dof` joint
    /// positions of env 0. The same convention [`joint_state`] uses for the Safety Plane —
    /// which is why `run_episode` refuses a model carrying fewer than `NJ` of them — and the
    /// only reading available, because that channel names a body and a count, not joints.
    Joints(usize),
    /// A `quantity = Velocity` channel whose id **is** a joint of the loaded model: `dof`
    /// velocities from that joint's first dof (packet M8/S4e). The channel names the block's
    /// first joint and how wide the block is; the model says where that joint's dofs start,
    /// which is the one thing the documents cannot know.
    Qvel(es_physics_core::backend::IndexRange),
    /// The leading `dof` joint velocities of env 0 — the mirror of [`Self::Joints`], for a
    /// `Velocity` channel that names a body rather than a joint.
    JointsVel(usize),
    /// An `ObsSource::BodyPose(b)` channel: body `b`'s row of `xpos` and `xquat`, read as
    /// `pos[3] ‖ quat[4]` with the quaternion in the order `StateView` documents — xyzw
    /// (spec 3.1). A **free-joint** body never reaches this arm: its pose is in `qpos` and
    /// the [`Self::Qpos`] arm answers it first, which is what serves the demo's
    /// `sim_cube_pose`.
    BodyPose(usize),
    /// An `ObservationNode::ImageInput`: the frame source's bytes, unconverted.
    Image,
    /// A Task IR `ObsSource::PreviousAction` channel of this width: the row the policy emitted
    /// for the previous control tick, which the *loop* holds and hands [`capture_at`]
    /// (packet M11/X2). Nothing in the physics state carries it.
    PreviousAction(usize),
}

/// A task's `PreviousAction` channel on the first tick of an episode: its declared `initial`,
/// or zeros of the channel's width when absent. `None` when the task declares no such channel,
/// and then the loop keeps no previous action at all.
pub fn previous_action_initial(task: &TaskIr) -> Option<Vec<f64>> {
    task.observation_spec
        .channels
        .values()
        .find_map(|c| match &c.source {
            es_ir::task::ObsSource::PreviousAction { initial } => {
                Some(initial.clone().unwrap_or_else(|| {
                    vec![0.0; c.ty.shape.dims().iter().product::<u64>() as usize]
                }))
            }
            _ => None,
        })
}

/// Resolves every plan input **before the first episode**, so an observation this build
/// cannot capture is one error at the start of the run rather than a surprise mid-table.
///
/// `model` is `None` when the frames are *recorded* rather than simulated (`crate::bake`): a
/// dataset carries `observation.state` and tiles, so the two arms that index a loaded model
/// have no reading there and the input is refused by name instead of guessed at.
pub fn input_sources(
    plan: &CpuPlan,
    obs: &ObservationIr,
    task: &TaskIr,
    model: Option<&ModelInfo>,
) -> Result<BTreeMap<String, Capture>, EvalError> {
    use es_ir::task::{JointQuantity, ObsSource};

    let state_channels = task
        .observation_spec
        .channels
        .values()
        .filter(|c| matches!(c.source, ObsSource::JointState { .. }))
        .count();
    let mut out = BTreeMap::new();
    for (name, id) in &plan.inputs {
        if !matches!(plan.buffers[id.0].home, Home::Input(_)) {
            continue;
        }
        let source = StableId::from_hex(name).map_err(|e| EvalError::Plan(e.to_string()))?;
        if source == ObsSource::previous_action_id() {
            if let Some(initial) = previous_action_initial(task) {
                out.insert(name.clone(), Capture::PreviousAction(initial.len()));
                continue;
            }
        }
        let is_image =
            obs.graph.nodes.values().any(
                |n| matches!(n, ObservationNode::ImageInput { sensor, .. } if *sensor == source),
            );
        // Every channel that names this plan input. `CpuPlan` allocates **one** input buffer
        // per source id (`Home::Input(id.to_string())`), so a second channel on the same id
        // would be handed the first one's bytes with no error anywhere — which is exactly the
        // silent wrong number §10.1 forbids. The IRs can express the pair (a joint block read
        // at two quantities, packet M8/S4e); the lowering cannot serve it yet, so it is
        // refused here, by name, where the wrongness would happen.
        let claimed: Vec<(&String, &ObsSource)> = task
            .observation_spec
            .channels
            .iter()
            .filter(|(_, c)| {
                matches!(c.source,
                    ObsSource::JointState { body, .. } | ObsSource::BodyPose(body)
                        if body == source)
            })
            .map(|(n, c)| (n, &c.source))
            .collect();
        if claimed.len() > 1 {
            let names: Vec<&str> = claimed.iter().map(|(n, _)| n.as_str()).collect();
            return Err(EvalError::Plan(format!(
                "observation input \"{name}\" is named by {} channels ({}); the observation \
                 plan allocates one input buffer per source id, so they would be served the \
                 same bytes",
                claimed.len(),
                names.join(", ")
            )));
        }
        let declared = claimed.first().map(|(_, s)| *s);
        // A channel the model's own maps cannot answer says so itself: a velocity, or the
        // pose of a body that has no joint. Both are read before the `qpos` / `sensor` arms
        // so that a `Position` channel resolves exactly as it did before this packet.
        let velocity = match declared {
            Some(ObsSource::JointState {
                dof,
                quantity: JointQuantity::Velocity,
                ..
            }) => Some(*dof),
            Some(ObsSource::JointState {
                quantity: q @ JointQuantity::Torque,
                ..
            }) => {
                return Err(EvalError::Plan(format!(
                    "observation input \"{name}\" declares quantity {q:?}; StateView carries \
                     qpos, qvel and sensordata, and no joint torque array"
                )))
            }
            _ => None,
        };
        let body_pose = match declared {
            // A **free-joint** body's pose is in `qpos`, and the arm below answers it — which
            // is what serves the demo's cube. Every other body is only in `xpos` / `xquat`.
            Some(ObsSource::BodyPose(b)) if !model.is_some_and(|m| m.qpos.contains_key(b)) => {
                Some(*b)
            }
            _ => None,
        };
        let joints = match declared {
            Some(ObsSource::JointState { dof, .. }) => Some(*dof as usize),
            _ => None,
        };
        let how = if is_image {
            Capture::Image
        } else if let Some(dof) = velocity {
            let Some(m) = model else {
                return Err(EvalError::Plan(format!(
                    "observation input \"{name}\" is a joint **velocity** channel, and there \
                     is no loaded model here (the frames are recorded): a dataset row carries \
                     observation.state, not qvel"
                )));
            };
            match m.dof.get(&source) {
                // The channel names the block's first joint and how wide the block is; the
                // model says where that joint's dofs start.
                Some(r) => Capture::Qvel(es_physics_core::backend::IndexRange::new(r.start, dof)),
                None => Capture::JointsVel(dof as usize),
            }
        } else if let Some(b) = body_pose {
            let Some(m) = model else {
                return Err(EvalError::Plan(format!(
                    "observation input \"{name}\" is a BodyPose channel, and there is no \
                     loaded model here (the frames are recorded): a dataset row carries \
                     observation.state, not xpos and xquat"
                )));
            };
            let row = m.body.get(&b).ok_or_else(|| {
                EvalError::Plan(format!(
                    "observation input \"{name}\" is a BodyPose channel naming a body that is \
                     not in the loaded model"
                ))
            })?;
            Capture::BodyPose(row.start as usize)
        } else if let Some(r) = model.and_then(|m| m.qpos.get(&source)) {
            Capture::Qpos(*r)
        } else if let Some(r) = model.and_then(|m| m.sensor.get(&source)) {
            Capture::Sensor(*r)
        } else if let Some(dof) = joints {
            // "The leading `dof` of the row" can place at most one channel. Once the
            // `ObservationSpec` declares a second `JointState` channel, which of them starts
            // at `qpos[0]` is not in the documents — it is in the model that ran — so a
            // model-free resolution is refused by name here rather than silently served the
            // wrong values (packet M5/V7a). With a model, the joint-named channel has already
            // been answered by the `Qpos` arm above and never reaches this one.
            if model.is_none() && state_channels > 1 {
                return Err(EvalError::Plan(format!(
                    "observation input \"{name}\" is one of {state_channels} JointState \
                     channels, and there is no loaded model here (the frames are recorded): \
                     only one channel can be the leading {dof} values of the row, and the \
                     documents do not say which"
                )));
            }
            Capture::Joints(dof)
        } else {
            let model = if model.is_some() {
                "a joint or sensor of the loaded model, "
            } else {
                // No model here means recorded frames, and saying "the loaded model" would
                // send the reader looking for a scene that this path never opens.
                "(there is no loaded model here: the frames are recorded) "
            };
            return Err(EvalError::Plan(format!(
                "observation input \"{name}\" is none of: {model}an ImageInput of the \
                 Observation IR, or a JointState or BodyPose channel of the Task IR's \
                 ObservationSpec"
            )));
        };
        out.insert(name.clone(), how);
    }
    Ok(out)
}

pub fn capture(
    plan: &CpuPlan,
    sources: &BTreeMap<String, Capture>,
    model: &ModelInfo,
    state: &StateView<'_>,
    frames: Option<&mut FrameSource<'_>>,
    light: &LightOverride,
    cell_frames: Option<&mut CellFrames>,
) -> Result<Captured, EvalError> {
    capture_at(plan, sources, model, state, frames, light, cell_frames, &[])
}

/// [`capture`], with the previous control tick's policy row a `PreviousAction` input reads
/// (packet M11/X2). The loop owns that row -- `run_episode` and `es_py::Rollout` -- and an
/// empty slice is "this caller keeps none", which a `PreviousAction` input refuses by name.
#[allow(clippy::too_many_arguments)]
pub fn capture_at(
    plan: &CpuPlan,
    sources: &BTreeMap<String, Capture>,
    model: &ModelInfo,
    state: &StateView<'_>,
    frames: Option<&mut FrameSource<'_>>,
    light: &LightOverride,
    cell_frames: Option<&mut CellFrames>,
    previous: &[f64],
) -> Result<Captured, EvalError> {
    let mut named = frames.map(|frame| {
        move |_: &str, light: &LightOverride, model: &ModelInfo, state: &StateView<'_>| {
            frame(light, model, state)
        }
    });
    capture_views(
        plan,
        sources,
        model,
        state,
        named.as_mut().map(|f| f as &mut ViewSource<'_>),
        light,
        cell_frames,
        previous,
    )
}

/// [`capture_at`] with a [`ViewSource`]: every image input is asked for by its own name, so
/// several cameras each feed their own port (packet M15/N2).
#[allow(clippy::too_many_arguments)]
pub fn capture_views(
    plan: &CpuPlan,
    sources: &BTreeMap<String, Capture>,
    model: &ModelInfo,
    state: &StateView<'_>,
    mut frames: Option<&mut ViewSource<'_>>,
    light: &LightOverride,
    mut cell_frames: Option<&mut CellFrames>,
    previous: &[f64],
) -> Result<Captured, EvalError> {
    let mut descs = Vec::new();
    let mut bytes = Vec::new();
    let mut rendered = None;
    for (name, id) in &plan.inputs {
        let desc = &plan.buffers[id.0];
        let Home::Input(_) = &desc.home else {
            continue;
        };
        let how = sources.get(name).copied().ok_or_else(|| {
            EvalError::Plan(format!("observation input \"{name}\" is unresolved"))
        })?;
        let values: Vec<f64> = match how {
            Capture::Qpos(r) => state.qpos_of(0)[r.as_range()].to_vec(),
            Capture::PreviousAction(n) => {
                if previous.len() != n {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\" is the previous action ({n} values); \
                         this loop holds {} -- a caller that keeps no previous policy row \
                         cannot serve a PreviousAction channel",
                        previous.len()
                    )));
                }
                previous.to_vec()
            }
            Capture::Sensor(r) => state.sensordata[r.as_range()].to_vec(),
            Capture::Joints(dof) => {
                let q = state.qpos_of(0);
                if q.len() < dof {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\" wants {dof} joint positions; the model \
                         carries {}",
                        q.len()
                    )));
                }
                q[..dof].to_vec()
            }
            Capture::Qvel(r) => {
                let qd = state.qvel_of(0);
                let end = (r.start + r.len) as usize;
                if qd.len() < end {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\" wants qvel[{}..{end}]; the model carries \
                         {}",
                        r.start,
                        qd.len()
                    )));
                }
                qd[r.as_range()].to_vec()
            }
            Capture::JointsVel(dof) => {
                let qd = state.qvel_of(0);
                if qd.len() < dof {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\" wants {dof} joint velocities; the model \
                         carries {}",
                        qd.len()
                    )));
                }
                qd[..dof].to_vec()
            }
            // `pos[3] ‖ quat[4]`, the quaternion in `StateView`'s own order: xyzw (spec 3.1).
            // Env 0's row, like every other arm here; `es_py::rollout` hands this function a
            // one-env view of the env it means.
            Capture::BodyPose(row) => {
                let (p, q) = (row * 3, row * 4);
                if state.xpos.len() < p + 3 || state.xquat.len() < q + 4 {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\" wants body row {row} of xpos/xquat; the \
                         state carries {} and {}",
                        state.xpos.len(),
                        state.xquat.len()
                    )));
                }
                let mut v = state.xpos[p..p + 3].to_vec();
                v.extend_from_slice(&state.xquat[q..q + 4]);
                v
            }
            Capture::Image => {
                // Without a frame source there is no renderer in this build (§4.3, es-render is
                // layer 5): feeding it zeros would produce a number, and a wrong number in the
                // §10.1 table is worse than no table.
                let Some(frame) = frames.as_deref_mut() else {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\" is an image; image inputs need a \
                         renderer, which this build has none of"
                    )));
                };
                // Exactly what the plan declared, or nothing: the frame is not resized, converted
                // or reordered here — every such step is an Observation IR node (§7.2, INV-14).
                let data = frame(name, light, model, state).map_err(EvalError::Plan)?;
                let want = desc.elems * elem_bytes(desc.dtype);
                if data.len() != want {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\": the plan wants {} {:?} elements \
                         ({want} bytes), the frame supplies {} bytes",
                        desc.elems,
                        desc.dtype,
                        data.len()
                    )));
                }
                // The frame the policy sees is the frame on disk: written here, from the same
                // bytes, before anything downstream can touch them.
                if let Some(cell) = cell_frames.as_deref_mut() {
                    rendered = Some(cell.write(name, desc.dtype, &desc.shape, &data)?);
                }
                descs.push((name.clone(), desc.dtype, desc.shape.clone()));
                bytes.push(data);
                continue;
            }
        };
        descs.push((name.clone(), desc.dtype, desc.shape.clone()));
        bytes.push(encode_state(name, desc.dtype, desc.elems, &values)?);
    }
    Ok((descs, bytes, rendered))
}

/// The Observation IR run on a live state, one control step at a time: what `es loop collect`
/// hands a trained policy (packet M16/H5), which before this was the raw `qpos ‖ qvel` row.
///
/// Nothing here is a second implementation: the `CpuPlan` [`Evaluation::run`] compiles, the
/// [`input_sources`] resolution against the loaded model, [`capture_at`] and `CpuPlan::run`,
/// and the previous action as `run_episode` keeps it -- the task's `initial` on an episode's
/// first step, then the row the loop last executed. The sources are resolved on the first
/// call, because the model is only loaded once the collector's `Env` exists.
///
/// No frame source: an image input is refused at construction, by name, rather than failing
/// on the first step with "no renderer in this build".
#[derive(Debug)]
pub struct LiveObservation {
    plan: CpuPlan,
    obs: ObservationIr,
    task: TaskIr,
    sources: Option<BTreeMap<String, Capture>>,
    initial: Vec<f64>,
    previous: Vec<f64>,
}

impl LiveObservation {
    pub fn new(obs: &ObservationIr, task: &TaskIr) -> Result<Self, EvalError> {
        if let Some(sensor) = obs.graph.nodes.values().find_map(|n| match n {
            ObservationNode::ImageInput { sensor, .. } => Some(sensor),
            _ => None,
        }) {
            return Err(EvalError::Plan(format!(
                "observation input \"{sensor}\" is an image; a live observation captures state \
                 only (it has no frame source)"
            )));
        }
        let plan = CpuPlan::compile(obs, PlanMode::Release)
            .map_err(|d| EvalError::Plan(d.iter().map(ToString::to_string).collect()))?;
        let initial = previous_action_initial(task).unwrap_or_default();
        Ok(Self {
            plan,
            obs: obs.clone(),
            task: task.clone(),
            sources: None,
            previous: initial.clone(),
            initial,
        })
    }

    /// One step: `first` opens an episode (the plan's history and the previous action
    /// restart); `previous` is the policy row the last step executed, if this episode has one.
    pub fn observe(
        &mut self,
        first: bool,
        previous: Option<&[f64]>,
        model: &ModelInfo,
        state: &StateView<'_>,
    ) -> Result<BTreeMap<String, Tensor>, EvalError> {
        let sources = match &mut self.sources {
            Some(s) => s,
            None => self.sources.insert(input_sources(
                &self.plan,
                &self.obs,
                &self.task,
                Some(model),
            )?),
        };
        if first {
            self.plan.reset();
            self.previous.clone_from(&self.initial);
        }
        if let (Some(row), false) = (previous, self.initial.is_empty()) {
            self.previous = row.to_vec();
        }
        let (names, bytes, _) = capture_at(
            &self.plan,
            sources,
            model,
            state,
            None,
            &LightOverride::default(),
            None,
            &self.previous,
        )?;
        let inputs: BTreeMap<String, TensorRef<'_>> = names
            .iter()
            .zip(&bytes)
            .map(|((name, dtype, shape), data)| {
                (
                    name.clone(),
                    TensorRef::new(*dtype, shape.clone(), data.as_slice()),
                )
            })
            .collect();
        self.plan
            .run(&inputs)
            .map_err(|e| EvalError::Plan(e.to_string()))
    }
}

/// State values into one plan input buffer.
///
/// The one place this conversion happens. `crate::bake` calls it with a recorded
/// `observation.state` row where [`capture`] calls it with a live `qpos` slice, so a training
/// input and an inference input cannot be two different roundings of the same number
/// (design note `docs/design/visible-learning.md` section 7.9).
pub(crate) fn encode_state(
    name: &str,
    dtype: ElemType,
    elems: usize,
    values: &[f64],
) -> Result<Vec<u8>, EvalError> {
    if values.len() != elems {
        return Err(EvalError::Plan(format!(
            "observation input \"{name}\": the plan wants {elems} elements, the source supplies \
             {}",
            values.len()
        )));
    }
    Ok(match dtype {
        ElemType::F32 => values
            .iter()
            .flat_map(|v| (*v as f32).to_le_bytes())
            .collect(),
        ElemType::F64 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        other => {
            return Err(EvalError::Plan(format!(
                "observation input \"{name}\" is {other:?}; state capture produces floats"
            )))
        }
    })
}

/// Bytes one element of `e` occupies in a plan buffer.
pub(crate) fn elem_bytes(e: ElemType) -> usize {
    match e {
        ElemType::F32 | ElemType::I32 => 4,
        ElemType::F16 | ElemType::Bf16 => 2,
        ElemType::F64 => 8,
        ElemType::U8 | ElemType::Bool => 1,
    }
}

/// The first `NJ` joint positions and velocities of env 0. Nothing is padded: `run_episode`
/// refused the run unless the model carries at least `NJ` of each.
pub fn joint_state<const NJ: usize>(state: &StateView<'_>) -> ([f64; NJ], [f64; NJ]) {
    let (mut q, mut qd) = ([0.0; NJ], [0.0; NJ]);
    q.copy_from_slice(&state.qpos_of(0)[..NJ]);
    qd.copy_from_slice(&state.qvel_of(0)[..NJ]);
    (q, qd)
}
