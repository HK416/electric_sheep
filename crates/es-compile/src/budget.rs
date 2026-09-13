//! Memory and bandwidth budget model (spec 20.2, 20.3).
//!
//! `MemoryBudget::estimate` totals the memory items spec 20.2 lists from static IR/config data
//! alone — no GPU, no running backend. It is the M2 W5 offline deliverable; the ±10% accuracy
//! gate (spec 28.4, 28.7 gate 13) needs a real device run and is `unverified` here (see
//! `docs/design/memory-budget.md`). Every [`BudgetItem`] carries the formula it used so the
//! number can be checked by hand against spec 20.1's reference measurements.

use std::collections::BTreeMap;
use std::fmt;

use es_ir::graph::IrNode;
use es_ir::image::{ChannelFormat, ImageDType, ImageSpec};
use es_ir::learning::LearningGraph;
use es_ir::observation::{ObservationIr, ObservationNode};
use es_ir::types::ElemType;

use crate::plan::{CpuPlan, Home, PlanMode};

/// A GPU/CPU float precision choice for the tensors this model does not pin a dtype for
/// (inference activations run at whatever precision the runtime picks, spec 20.2).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    F16,
}

impl Precision {
    fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// The nq/nv/nu/nsensordata sizes `MemoryBudget` needs from a physics model.
///
/// Mirrors the fields of `es_physics_core::backend::ModelInfo` this budget uses. `es-compile`
/// (layer 7) does not depend on `es-physics-core` (layer 3) for this alone — the CLI, which
/// already depends on both, copies the four numbers across at the call site.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModelSizes {
    pub nq: u32,
    pub nv: u32,
    pub nu: u32,
    pub nsensordata: u32,
}

/// The four batch-domain sizes of spec 5.2. Shaped like `es_env::scheduler::BatchDomains` but
/// defined locally: `es-compile` cannot depend on `es-env` (layer 9 depends on layer 7, not the
/// other way around).
#[derive(Clone, Copy, Debug)]
pub struct BudgetDomains {
    pub n_sim_envs: u32,
    pub n_obs_envs: u32,
    pub n_views: u32,
    pub inference_batch: u32,
}

/// A camera tile atlas layout (spec 15.2): tiles are packed `tiles_per_row` wide, so a tile
/// count that does not divide evenly pads the last row.
#[derive(Clone, Copy, Debug)]
pub struct TileAtlasCfg {
    pub tile_w: u32,
    pub tile_h: u32,
    pub tiles_per_row: u32,
}

#[derive(Debug)]
pub struct BudgetInputs<'a> {
    pub domains: BudgetDomains,
    pub obs: &'a ObservationIr,
    pub learning: Option<&'a LearningGraph>,
    pub model: Option<&'a ModelSizes>,
    pub precision: Precision,
    pub tile_atlas: Option<TileAtlasCfg>,
    /// Total device memory, for the spec 20.3 "total <= device memory - reserve" rule. `None`
    /// skips that rule (no device to check against).
    pub device_bytes: Option<u64>,
}

/// One line item of the budget, with the formula that produced `bytes` spelled out so a human
/// can audit it against spec 20.1/20.2. `bytes == 0` with a formula ending in "unavailable"
/// means the model has no data to size the item from, not that the item costs nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetItem {
    pub name: String,
    pub bytes: u64,
    pub formula: String,
}

impl BudgetItem {
    fn unavailable(name: &str, why: &str) -> Self {
        Self {
            name: name.to_owned(),
            bytes: 0,
            formula: format!("unavailable: {why}"),
        }
    }
}

/// A spec 20.3 rule the report violates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetViolation {
    pub rule: &'static str,
    pub message: String,
}

#[derive(Debug)]
pub struct MemoryReport {
    pub items: Vec<BudgetItem>,
    pub total_bytes: u64,
    /// Bytes attributed to each of the spec 5.2 domains plus `"control"` for the chunk
    /// buffers, which belong to none of the four.
    pub per_domain: BTreeMap<String, u64>,
    pub bandwidth_per_tick: Option<BudgetItem>,
}

// ponytail: fixed 512 MiB driver/OS reserve; calibrate against a real device once the ±10%
// gate (spec 28.4) has GPU-in-the-loop CI to run against.
const DEFAULT_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
/// `maxImageDimension2D` on the desktop GPUs spec 15.2 targets.
const MAX_IMAGE_DIMENSION_2D: u32 = 16384;

fn elem_bytes(e: ElemType) -> u64 {
    match e {
        ElemType::F32 | ElemType::I32 => 4,
        ElemType::F16 | ElemType::Bf16 => 2,
        ElemType::F64 => 8,
        ElemType::U8 | ElemType::Bool => 1,
    }
}

fn image_elem_bytes(d: ImageDType) -> u64 {
    match d {
        ImageDType::U8 => 1,
        ImageDType::U16 | ImageDType::F16 => 2,
        ImageDType::F32 => 4,
    }
}

fn channel_count(f: ChannelFormat) -> u64 {
    match f {
        ChannelFormat::Rgba => 4,
        ChannelFormat::Rgb | ChannelFormat::Normal => 3,
        ChannelFormat::Flow => 2,
        ChannelFormat::Gray | ChannelFormat::Depth | ChannelFormat::Seg => 1,
    }
}

/// The raw sensor `ImageSpec` of the first `ImageInput` node, in node-id order. The render
/// tile atlas is sized from the sensor's own format (spec 15.2), not a downstream tensor's.
fn first_image_spec(obs: &ObservationIr) -> Option<ImageSpec> {
    obs.graph.nodes.values().find_map(|n| match n {
        ObservationNode::ImageInput { io, .. } => io.output.image,
        _ => None,
    })
}

/// Per-sample bytes of the node that owns `sensor` (an `ImageInput` or `StateInput`), for the
/// spec 20.2 history-buffer item.
fn sensor_sample_bytes(obs: &ObservationIr, sensor: es_core::StableId) -> Option<u64> {
    obs.graph.nodes.values().find_map(|n| {
        let (matches_sensor, ty) = match n {
            ObservationNode::ImageInput { sensor: s, io }
            | ObservationNode::StateInput { source: s, io } => (*s == sensor, &io.output),
            _ => return None,
        };
        matches_sensor.then(|| ty.shape.elem_count() * elem_bytes(ty.elem))
    })
}

impl MemoryReport {
    /// Every spec 20.3 rule this report can check on its own. `obs_batch <= sim_batch` always
    /// runs; the device-headroom rule only when `inputs.device_bytes` is given.
    pub fn violations(&self, inputs: &BudgetInputs) -> Vec<BudgetViolation> {
        let mut out = Vec::new();
        let d = inputs.domains;
        if d.n_obs_envs > d.n_sim_envs {
            out.push(BudgetViolation {
                rule: "obs_batch_le_sim_batch",
                message: format!(
                    "n_obs_envs ({}) exceeds n_sim_envs ({}) -- spec 5.2 observation batch is a \
                     subset of the simulation batch",
                    d.n_obs_envs, d.n_sim_envs
                ),
            });
        }
        if let Some(device) = inputs.device_bytes {
            let budget = device.saturating_sub(DEFAULT_RESERVE_BYTES);
            if self.total_bytes > budget {
                out.push(BudgetViolation {
                    rule: "total_le_device_minus_reserve",
                    message: format!(
                        "total {} bytes exceeds device budget {} bytes ({} device - {} reserve)",
                        self.total_bytes, budget, device, DEFAULT_RESERVE_BYTES
                    ),
                });
            }
        }
        if let Some(atlas) = inputs.tile_atlas {
            let tiles = u64::from(d.n_obs_envs) * u64::from(d.n_views);
            let rows = tiles.div_ceil(u64::from(atlas.tiles_per_row).max(1));
            let w = u64::from(atlas.tiles_per_row) * u64::from(atlas.tile_w);
            let h = rows * u64::from(atlas.tile_h);
            if w > u64::from(MAX_IMAGE_DIMENSION_2D) || h > u64::from(MAX_IMAGE_DIMENSION_2D) {
                out.push(BudgetViolation {
                    rule: "tile_atlas_within_max_image_dimension_2d",
                    message: format!(
                        "atlas {w}x{h} exceeds maxImageDimension2D {MAX_IMAGE_DIMENSION_2D} (spec 15.2)"
                    ),
                });
            }
        }
        out
    }
}

impl fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        writeln!(f, "{:<28} {:>10}  formula", "item", "GiB")?;
        for item in &self.items {
            writeln!(
                f,
                "{:<28} {:>10.4}  {}",
                item.name,
                item.bytes as f64 / GIB,
                item.formula
            )?;
        }
        writeln!(f, "{:<28} {:>10.4}", "total", self.total_bytes as f64 / GIB)?;
        writeln!(f, "\nper domain:")?;
        for (domain, bytes) in &self.per_domain {
            writeln!(f, "  {domain:<24} {:>10.4} GiB", *bytes as f64 / GIB)?;
        }
        if let Some(bw) = &self.bandwidth_per_tick {
            writeln!(
                f,
                "\nbandwidth per tick: {:.4} GiB ({})",
                bw.bytes as f64 / GIB,
                bw.formula
            )?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct MemoryBudget;

impl MemoryBudget {
    /// Estimate every spec 20.2 item from static config, with no GPU and no running backend
    /// (spec 28.4's ±10% accuracy gate is unverified until one is available, see
    /// `docs/design/memory-budget.md`).
    pub fn estimate(inputs: &BudgetInputs<'_>) -> MemoryReport {
        let d = inputs.domains;
        let mut per_domain: BTreeMap<String, u64> = BTreeMap::new();
        let mut add_domain = |domain: &str, bytes: u64| {
            *per_domain.entry(domain.to_owned()).or_insert(0) += bytes;
        };

        // -- physics state (backend-owned sizing, spec 20.2) --
        let physics_state = match inputs.model {
            Some(m) => {
                let per_env = u64::from(m.nq + m.nv + m.nu + m.nsensordata);
                let bytes = per_env * u64::from(d.n_sim_envs) * 8;
                BudgetItem {
                    name: "physics_state".to_owned(),
                    bytes,
                    formula: format!(
                        "(nq={} + nv={} + nu={} + nsensordata={}) x n_sim_envs={} x 8B (f64)",
                        m.nq, m.nv, m.nu, m.nsensordata, d.n_sim_envs
                    ),
                }
            }
            None => BudgetItem::unavailable(
                "physics_state",
                "no ModelInfo given -- the physics backend owns this sizing (spec 20.2)",
            ),
        };
        add_domain("simulation", physics_state.bytes);

        // -- render tile atlas (spec 15.2, 20.2) --
        let render_tile_atlas = match first_image_spec(inputs.obs) {
            None => BudgetItem::unavailable(
                "render_tile_atlas",
                "observation IR has no ImageInput node",
            ),
            Some(spec) => {
                let ch = channel_count(spec.channels);
                let dtype_bytes = image_elem_bytes(spec.dtype);
                let tiles = u64::from(d.n_obs_envs) * u64::from(d.n_views);
                if let Some(atlas) = inputs.tile_atlas {
                    let per_row = u64::from(atlas.tiles_per_row).max(1);
                    let rows = tiles.div_ceil(per_row);
                    let bytes = rows
                        * per_row
                        * u64::from(atlas.tile_w)
                        * u64::from(atlas.tile_h)
                        * ch
                        * dtype_bytes
                        * 2;
                    BudgetItem {
                        name: "render_tile_atlas".to_owned(),
                        bytes,
                        formula: format!(
                            "rows={rows} x tiles_per_row={per_row} x {}x{} x ch={ch} x {dtype_bytes}B x 2 (double buffer)",
                            atlas.tile_w, atlas.tile_h
                        ),
                    }
                } else {
                    let bytes = tiles
                        * u64::from(spec.width)
                        * u64::from(spec.height)
                        * ch
                        * dtype_bytes
                        * 2;
                    BudgetItem {
                        name: "render_tile_atlas".to_owned(),
                        bytes,
                        formula: format!(
                            "n_obs_envs={} x n_views={} x {}x{} x ch={ch} x {dtype_bytes}B x 2 (double buffer)",
                            d.n_obs_envs, d.n_views, spec.width, spec.height
                        ),
                    }
                }
            }
        };
        add_domain("observation", render_tile_atlas.bytes);

        // -- observation intermediates: the CPU reference plan's own arena buffers (spec 11.1
        // liveness analysis), scaled by the observation batch. --
        let observation_intermediates = match CpuPlan::compile(inputs.obs, PlanMode::Release) {
            Err(_) => BudgetItem::unavailable(
                "observation_intermediates",
                "observation IR does not compile to a CPU plan",
            ),
            Ok(plan) => {
                let per_env: u64 = plan
                    .buffers
                    .iter()
                    .filter(|b| matches!(b.home, Home::Arena(_)))
                    .map(|b| b.elems as u64 * elem_bytes(b.dtype))
                    .sum();
                BudgetItem {
                    name: "observation_intermediates".to_owned(),
                    bytes: per_env * u64::from(d.n_obs_envs),
                    formula: format!(
                        "sum(CPU plan arena buffers)={per_env}B x n_obs_envs={}",
                        d.n_obs_envs
                    ),
                }
            }
        };
        add_domain("observation", observation_intermediates.bytes);

        // -- history buffers: History.depth x per-sensor bytes x envs (spec 7.5 layer 1, 20.2) --
        let mut history_bytes = 0u64;
        let mut history_parts = Vec::new();
        for (sensor, history) in &inputs.obs.temporal.history {
            if let Some(sample_bytes) = sensor_sample_bytes(inputs.obs, *sensor) {
                let bytes = u64::from(history.depth) * sample_bytes * u64::from(d.n_obs_envs);
                history_bytes += bytes;
                history_parts.push(format!(
                    "depth={} x {sample_bytes}B x n_obs_envs={}",
                    history.depth, d.n_obs_envs
                ));
            }
        }
        let history_buffers = BudgetItem {
            name: "history_buffers".to_owned(),
            bytes: history_bytes,
            formula: if history_parts.is_empty() {
                "no History entries in observation IR".to_owned()
            } else {
                history_parts.join(" + ")
            },
        };
        add_domain("observation", history_buffers.bytes);

        // -- policy weights: WeightsRef never carries a byte size (INV-16 keeps it to
        // path+hash), so this is always unavailable against the current IR. --
        let policy_weights = BudgetItem::unavailable(
            "policy_weights",
            "WeightsRef carries a path and hash, no byte size (spec 8.3)",
        );
        add_domain("inference", policy_weights.bytes);

        // -- inference activations: sum of LearningNode output shapes x inference batch -- //
        let inference_activations = match inputs.learning {
            None => BudgetItem::unavailable("inference_activations", "no Learning IR given"),
            Some(lg) => {
                let bpe = inputs.precision.bytes();
                let per_sample: u64 = lg
                    .nodes
                    .nodes
                    .values()
                    .flat_map(IrNode::outputs)
                    .map(|p| p.ty.shape.elem_count() * bpe)
                    .sum();
                BudgetItem {
                    name: "inference_activations".to_owned(),
                    bytes: per_sample * u64::from(d.inference_batch),
                    formula: format!(
                        "sum(node output elems)={per_sample}B x inference_batch={} ({bpe}B/elem, {:?})",
                        d.inference_batch, inputs.precision
                    ),
                }
            }
        };
        add_domain("inference", inference_activations.bytes);

        // -- chunk buffers: n_sim_envs x horizon x action_dim x 4B x 2 (spec 20.2) -- //
        let chunk_buffers = match inputs.learning {
            None => BudgetItem::unavailable("chunk_buffers", "no Learning IR given"),
            Some(lg) => {
                let c = &lg.policy.contract;
                let bytes = u64::from(d.n_sim_envs)
                    * u64::from(c.horizon)
                    * u64::from(c.action_dim)
                    * 4
                    * 2;
                BudgetItem {
                    name: "chunk_buffers".to_owned(),
                    bytes,
                    formula: format!(
                        "n_sim_envs={} x horizon={} x action_dim={} x 4B x 2 (double buffer)",
                        d.n_sim_envs, c.horizon, c.action_dim
                    ),
                }
            }
        };
        add_domain("control", chunk_buffers.bytes);

        let bandwidth_per_tick = Some(BudgetItem {
            name: "observation_bandwidth_per_tick".to_owned(),
            bytes: render_tile_atlas.bytes + observation_intermediates.bytes,
            formula: "render_tile_atlas + observation_intermediates (moved once per tick)"
                .to_owned(),
        });

        let items = vec![
            physics_state,
            render_tile_atlas,
            observation_intermediates,
            history_buffers,
            policy_weights,
            inference_activations,
            chunk_buffers,
        ];
        let total_bytes = items.iter().map(|i| i.bytes).sum();

        MemoryReport {
            items,
            total_bytes,
            per_domain,
            bandwidth_per_tick,
        }
    }
}

impl fmt::Debug for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use es_core::StableId;
    use es_ir::graph::{Graph, NodeId, PortRef};
    use es_ir::learning::{
        ActionExecutionMode, ArchKind, LearningNode, PolicyContract, PolicyHandle, RuntimeHints,
        StateEncoderKind, WeightsRef,
    };
    use es_ir::observation::{History, Io, NormalizeStats, ObservationOutput, TemporalModel};
    use es_ir::types::{Frame, PortType, Shape, TimeRef, Unit};

    use super::*;

    fn robot() -> StableId {
        StableId::from_path("robot")
    }

    fn state_ty(unit: Unit) -> PortType {
        PortType {
            elem: ElemType::F32,
            shape: Shape::new([4]),
            unit,
            frame: Frame::Joint(robot()),
            time: TimeRef::Tick,
            image: None,
        }
    }

    /// `StateInput -> Normalize`, no cameras: one arena buffer (the `Normalize` output), one
    /// `History` entry on the raw state so the history-buffer formula has something to sum.
    fn obs_fixture() -> ObservationIr {
        let raw = state_ty(Unit::Dimensionless);
        let norm = state_ty(Unit::Normalized { lo: -1.0, hi: 1.0 });
        let mut ir = ObservationIr {
            schema_version: 1,
            task_ref: [0; 32],
            graph: Graph::new(1),
            temporal: TemporalModel::default(),
            outputs: BTreeMap::new(),
        };
        ir.graph.insert(
            NodeId(0),
            ObservationNode::StateInput {
                source: robot(),
                io: Io::source(raw.clone()),
            },
        );
        ir.graph.insert(
            NodeId(1),
            ObservationNode::Normalize {
                stats: NormalizeStats::Range { lo: -1.0, hi: 1.0 },
                io: Io::unary(raw, norm.clone()),
            },
        );
        ir.graph.connect(NodeId(0), "out", NodeId(1), "in0");
        ir.temporal.history.insert(robot(), History { depth: 3 });
        ir.outputs.insert(
            "state".to_owned(),
            ObservationOutput {
                port: PortRef::new(NodeId(1), "out"),
                ty: norm,
            },
        );
        ir
    }

    fn learning_fixture() -> LearningGraph {
        let mut nodes = Graph::new(1);
        nodes.insert(
            NodeId(0),
            LearningNode::StateEncoder {
                inputs: vec![],
                kind: StateEncoderKind::Identity,
                out_dim: 4,
            },
        );
        LearningGraph {
            schema_version: 1,
            inputs: vec![],
            nodes,
            outputs: vec![],
            policy: PolicyHandle {
                architecture: ArchKind::Act,
                base_model: None,
                weights: WeightsRef::Safetensors {
                    path: "w.safetensors".to_owned(),
                    hash: [0; 32],
                },
                contract: PolicyContract {
                    inputs: BTreeMap::new(),
                    observation_window: 1,
                    action_dim: 2,
                    horizon: 5,
                    execute_chunk: 5,
                    replanning_hz: 10.0,
                    execution_mode: ActionExecutionMode::RecedingHorizon,
                    runtime: RuntimeHints {
                        dtype: ElemType::F32,
                        expected_latency_ms: 0.0,
                        deadline_ms: 0.0,
                    },
                },
            },
        }
    }

    fn inputs<'a>(
        obs: &'a ObservationIr,
        learning: &'a LearningGraph,
        model: &'a ModelSizes,
    ) -> BudgetInputs<'a> {
        BudgetInputs {
            domains: BudgetDomains {
                n_sim_envs: 4,
                n_obs_envs: 2,
                n_views: 1,
                inference_batch: 8,
            },
            obs,
            learning: Some(learning),
            model: Some(model),
            precision: Precision::F32,
            tile_atlas: None,
            device_bytes: None,
        }
    }

    #[test]
    fn hand_computed_totals_match() {
        let obs = obs_fixture();
        let learning = learning_fixture();
        let model = ModelSizes {
            nq: 2,
            nv: 2,
            nu: 1,
            nsensordata: 3,
        };
        let report = MemoryBudget::estimate(&inputs(&obs, &learning, &model));

        // physics: (2+2+1+3) x 4 envs x 8B = 256
        let physics = report
            .items
            .iter()
            .find(|i| i.name == "physics_state")
            .unwrap();
        assert_eq!(physics.bytes, 256);

        // no camera in the fixture -- unavailable, not zero-cost.
        let atlas = report
            .items
            .iter()
            .find(|i| i.name == "render_tile_atlas")
            .unwrap();
        assert_eq!(atlas.bytes, 0);
        assert!(atlas.formula.starts_with("unavailable"));

        // one arena buffer (Normalize output): 4 elems x 4B x 2 obs envs = 32
        let obs_inter = report
            .items
            .iter()
            .find(|i| i.name == "observation_intermediates")
            .unwrap();
        assert_eq!(obs_inter.bytes, 32);

        // history: depth=3 x (4 elems x 4B = 16B) x 2 obs envs = 96
        let history = report
            .items
            .iter()
            .find(|i| i.name == "history_buffers")
            .unwrap();
        assert_eq!(history.bytes, 96);

        // policy weights: always unavailable against the current WeightsRef.
        let weights = report
            .items
            .iter()
            .find(|i| i.name == "policy_weights")
            .unwrap();
        assert_eq!(weights.bytes, 0);
        assert!(weights.formula.starts_with("unavailable"));

        // inference: StateEncoder out_dim=4 -> 4 elems x 4B x inference_batch=8 = 128
        let inference = report
            .items
            .iter()
            .find(|i| i.name == "inference_activations")
            .unwrap();
        assert_eq!(inference.bytes, 128);

        // chunk: 4 sim envs x horizon=5 x action_dim=2 x 4B x 2 = 320
        let chunk = report
            .items
            .iter()
            .find(|i| i.name == "chunk_buffers")
            .unwrap();
        assert_eq!(chunk.bytes, 320);

        assert_eq!(report.total_bytes, 256 + 32 + 96 + 128 + 320);
        assert_eq!(report.per_domain["simulation"], 256);
        assert_eq!(report.per_domain["control"], 320);

        // Every item states its formula, even the unavailable ones.
        assert!(report.items.iter().all(|i| !i.formula.is_empty()));

        let bw = report.bandwidth_per_tick.unwrap();
        assert_eq!(bw.bytes, atlas.bytes + obs_inter.bytes);
    }

    #[test]
    fn obs_batch_over_sim_batch_is_a_violation() {
        let obs = obs_fixture();
        let learning = learning_fixture();
        let model = ModelSizes::default();
        let mut inp = inputs(&obs, &learning, &model);
        inp.domains.n_obs_envs = inp.domains.n_sim_envs + 1;
        let report = MemoryBudget::estimate(&inp);
        let violations = report.violations(&inp);
        assert!(violations
            .iter()
            .any(|v| v.rule == "obs_batch_le_sim_batch"));
    }

    #[test]
    fn device_headroom_violation_when_total_exceeds_budget() {
        let obs = obs_fixture();
        let learning = learning_fixture();
        let model = ModelSizes {
            nq: 2,
            nv: 2,
            nu: 1,
            nsensordata: 3,
        };
        let mut inp = inputs(&obs, &learning, &model);
        inp.device_bytes = Some(1024); // far below the 512 MiB reserve alone
        let report = MemoryBudget::estimate(&inp);
        let violations = report.violations(&inp);
        assert!(violations
            .iter()
            .any(|v| v.rule == "total_le_device_minus_reserve"));
    }

    #[test]
    fn display_prints_a_gib_table() {
        let obs = obs_fixture();
        let learning = learning_fixture();
        let model = ModelSizes::default();
        let report = MemoryBudget::estimate(&inputs(&obs, &learning, &model));
        let text = report.to_string();
        assert!(text.contains("GiB"));
        assert!(text.contains("physics_state"));
        assert!(text.contains("total"));
        assert!(text.contains("per domain:"));
    }
}
