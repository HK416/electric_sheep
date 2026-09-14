//! Running the Observation IR over *recorded* frames (§7.2, §19.2) — `es dataset bake`.
//!
//! This exists because training had a second implementation of one Observation IR node.
//! `python/es/train_act.py` re-implemented `Op::Dequantize` for the image and fed the state
//! port the raw `observation.state` row, while `capture` ran the compiled plan — which puts a
//! `Normalize{Range −1..1}` on that port. The policy trained on `q` and was evaluated on
//! `(q + 1) / 2`, and every success rate in `docs/design/visible-learning.md` section 7.8 is
//! depressed by it (open question 11).
//!
//! The fix is not a third implementation. [`ObservationBake`] is [`crate::runner::capture`]
//! with a dataset row where the physics state was: the same [`CpuPlan`], the same
//! [`crate::runner::input_sources`] resolution, the same [`crate::runner::encode_state`]
//! conversion. Only the origin of the raw values differs, which is the entire point — what
//! training reads is what inference computes.
//!
//! Layering (§4.2): `es-data` is layer 10 and so is this crate, so nothing here opens a
//! dataset. The caller (`es dataset bake`, layer 12) reads the rows and writes the
//! safetensors; this is only the executor.

use std::collections::BTreeMap;

use es_compile::{CpuPlan, Home, PlanMode, Tensor, TensorRef};
use es_ir::observation::ObservationIr;
use es_ir::task::TaskIr;
use es_ir::types::ElemType;

use crate::runner::{elem_bytes, encode_state, input_sources, Capture};
use crate::EvalError;

/// The Observation IR's compiled plan, driven from recorded frames.
#[derive(Debug)]
pub struct ObservationBake {
    plan: CpuPlan,
    sources: BTreeMap<String, Capture>,
}

impl ObservationBake {
    /// Compiles the plan and resolves every input once, before any frame.
    ///
    /// `PlanMode::Release` and no `ModelInfo`, both deliberately: release is what
    /// `Evaluation::run` compiles, and a recorded dataset carries `observation.state` plus
    /// tiles and nothing else — so an input that would have been read out of a loaded model's
    /// `qpos` or `sensordata` is named and refused here rather than turned into a guessed
    /// offset into a row.
    pub fn new(obs: &ObservationIr, task: &TaskIr) -> Result<Self, EvalError> {
        let plan = CpuPlan::compile(obs, PlanMode::Release)
            .map_err(|d| EvalError::Plan(d.iter().map(ToString::to_string).collect()))?;
        let sources = input_sources(&plan, obs, task, None)?;
        Ok(Self { plan, sources })
    }

    /// The episode boundary — exactly the `plan.reset()` `run_episode` does per episode, so a
    /// `TemporalWindow` cannot reach back into the previous episode in training either (§7.5).
    pub fn reset(&mut self) {
        self.plan.reset();
    }

    /// `compiler_hash` of the plan the bake ran, for the manifest's §5.3 slot.
    pub fn compiler_hash(&self) -> [u8; 32] {
        self.plan.compiler_hash()
    }

    /// Every output port the bake produces, with the dtype and shape of one frame.
    pub fn outputs(&self) -> impl Iterator<Item = (&str, ElemType, &[u64])> {
        self.plan.outputs.iter().map(|(name, id)| {
            let desc = &self.plan.buffers[id.0];
            (name.as_str(), desc.dtype, desc.shape.as_slice())
        })
    }

    /// One frame: `state` is the recorded `observation.state` row — which is the arm's own
    /// `qpos`, the way `es loop collect` records it — and `image` supplies a named image
    /// input's raw tile, unconverted.
    pub fn frame(
        &mut self,
        state: &[f64],
        image: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
    ) -> Result<BTreeMap<String, Tensor>, EvalError> {
        let mut bytes: Vec<(String, ElemType, Vec<u64>, Vec<u8>)> = Vec::new();
        for (name, id) in &self.plan.inputs {
            let desc = &self.plan.buffers[id.0];
            let Home::Input(_) = &desc.home else {
                continue;
            };
            let how = self.sources.get(name).copied().ok_or_else(|| {
                EvalError::Plan(format!("observation input \"{name}\" is unresolved"))
            })?;
            let data = match how {
                Capture::Joints(dof) => {
                    if state.len() < dof {
                        return Err(EvalError::Plan(format!(
                            "observation input \"{name}\" wants {dof} joint positions; the \
                             recorded row carries {}",
                            state.len()
                        )));
                    }
                    encode_state(name, desc.dtype, desc.elems, &state[..dof])?
                }
                // Exactly what the plan declared, or nothing — the tile is not resized,
                // converted or reordered here; every such step is an Observation IR node
                // (§7.2, INV-14). The same refusal `capture` makes, with the same wording.
                Capture::Image => {
                    let data = image(name).map_err(EvalError::Plan)?;
                    let want = desc.elems * elem_bytes(desc.dtype);
                    if data.len() != want {
                        return Err(EvalError::Plan(format!(
                            "observation input \"{name}\": the plan wants {} {:?} elements \
                             ({want} bytes), the recorded frame supplies {} bytes",
                            desc.elems,
                            desc.dtype,
                            data.len()
                        )));
                    }
                    data
                }
                // Unreachable: `input_sources` refused both at construction, because it was
                // handed no model. Kept as a refusal rather than a panic so a later arm added
                // to `Capture` fails loudly here instead of silently baking the wrong bytes.
                Capture::Qpos(_) | Capture::Sensor(_) => {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\" reads a loaded model; recorded frames \
                         carry no model"
                    )))
                }
            };
            bytes.push((name.clone(), desc.dtype, desc.shape.clone(), data));
        }
        let inputs: BTreeMap<String, TensorRef<'_>> = bytes
            .iter()
            .map(|(name, dtype, shape, data)| {
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
