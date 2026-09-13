//! Observation IR → CPU reference execution plan (spec 11.1 `Lower`, spec 11.3).
//!
//! Compile resolves everything that does not depend on pixel values: the node order, every
//! buffer's dtype and shape, the arena layout, and which kernel each node becomes. `run`
//! (see [`crate::exec`]) then only fills buffers. See
//! `docs/design/observation-lowering.md`.

use std::collections::BTreeMap;

use es_ir::diag::Diagnostic;
use es_ir::graph::{IrNode, NodeId, PortRef};
use es_ir::image::{ChannelFormat, ImageSpec, Rect};
use es_ir::observation::{
    CropMode, NormalizeStats, ObservationIr, ObservationNode, ResizeFilter, OUT,
};
use es_ir::types::{Align, ElemType};
use es_ir::CanonWriter;

use crate::kernels::KERNEL_IDS;

// TODO(codes-merge): move these into `es_ir::codes` when the compiler codes are dictionaried,
// the way the five IR modules held their codes before P27 merged them.
pub const COMPILE_001: &str = "COMPILE-001";
pub const COMPILE_002: &str = "COMPILE-002";
pub const COMPILE_003: &str = "COMPILE-003";
pub const COMPILE_004: &str = "COMPILE-004";
pub const COMPILE_005: &str = "COMPILE-005";

/// Spec 11.5. Both modes keep every intermediate addressable and neither aliases the arena:
/// the CPU path exists *to have* node boundaries (design note §10). The mode still reaches
/// [`CpuPlan::compiler_hash`] so a future aliasing release plan cannot reuse a cached hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanMode {
    Debug,
    Release,
}

impl PlanMode {
    fn tag(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BufferId(pub usize);

/// Where a buffer's bytes come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Home {
    /// Supplied by the caller of `run` under this name (a `StableId` in hex).
    Input(String),
    /// Element offset into the f32 arena.
    Arena(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct BufferDesc {
    pub dtype: ElemType,
    pub shape: Vec<u64>,
    pub elems: usize,
    pub home: Home,
    /// The node whose `out` port this is — debug mode's per-node value view (spec 11.5).
    pub node: NodeId,
}

/// One kernel invocation. Parameters are resolved at compile time; `run` does no arithmetic
/// on shapes.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// HWC u8 → CHW f32 `/255`.
    Dequantize {
        h: usize,
        w: usize,
        c: usize,
    },
    /// HWC u8 → CHW f32 through the 256-entry sRGB LUT (design note §6).
    SrgbToLinearU8 {
        h: usize,
        w: usize,
        c: usize,
    },
    SrgbToLinear,
    Resize {
        filter: ResizeFilter,
        sw: usize,
        sh: usize,
        c: usize,
        dw: usize,
        dh: usize,
    },
    Crop {
        sw: usize,
        sh: usize,
        c: usize,
        rect: Rect,
    },
    NormalizeMeanStd {
        plane: usize,
        mean: Vec<f32>,
        std: Vec<f32>,
    },
    NormalizeRange {
        lo: f32,
        hi: f32,
    },
    /// `Concat` and `Stack` share a kernel; `chunk` is per input.
    Join {
        chunk: Vec<usize>,
        outer: usize,
    },
    /// `TemporalWindow`: push the incoming frame, then gather.
    Window {
        slot: usize,
        depth: usize,
        n: usize,
        stride: usize,
    },
    Cast {
        to: ElemType,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub node: NodeId,
    pub op: Op,
    pub inputs: Vec<BufferId>,
    pub out: BufferId,
}

/// A ring buffer backing one `TemporalWindow` (layer 1 of spec 7.5, owned here only because
/// the CPU reference runs standalone).
#[derive(Clone, Debug, PartialEq)]
pub struct Ring {
    pub slot: usize,
    pub depth: usize,
    pub data: Vec<f32>,
    pub cursor: usize,
    pub pushed: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CpuPlan {
    pub mode: PlanMode,
    pub buffers: Vec<BufferDesc>,
    pub steps: Vec<Step>,
    /// Arena size in f32 elements; `run` allocates it once.
    pub arena_elems: usize,
    /// Input name → buffer, in the order `run` expects to find them.
    pub inputs: BTreeMap<String, BufferId>,
    /// The names Learning IR binds to (`ObservationIr::outputs`).
    pub outputs: BTreeMap<String, BufferId>,
    /// One per `TemporalWindow` step, keyed by node.
    pub rings: BTreeMap<NodeId, Ring>,
    /// Non-fatal IR diagnostics kept with the plan — `OBS-034` for a deliberate
    /// `rescale_intrinsics = false`, most of all.
    pub warnings: Vec<Diagnostic>,
}

fn channel_count(f: ChannelFormat) -> usize {
    match f {
        ChannelFormat::Rgba => 4,
        ChannelFormat::Rgb | ChannelFormat::Normal => 3,
        ChannelFormat::Flow => 2,
        ChannelFormat::Gray | ChannelFormat::Depth | ChannelFormat::Seg => 1,
    }
}

/// `in7` → 7. Sorting on the port *name* would put `in10` before `in2`.
fn port_index(port: &str) -> usize {
    port.strip_prefix("in")
        .and_then(|n| n.parse().ok())
        .unwrap_or(usize::MAX)
}

fn unsupported(what: &str, id: NodeId) -> Diagnostic {
    Diagnostic::new(
        COMPILE_002,
        format!("{what} has no CPU reference kernel yet"),
    )
    .at(id)
    .with_hint("docs/design/observation-lowering.md section 3 lists the supported node set")
}

impl CpuPlan {
    /// Compile an Observation IR into the CPU reference plan.
    ///
    /// Errors are returned whole, never as the first one: a graph with three wrong nodes
    /// should be fixed in one pass.
    pub fn compile(ir: &ObservationIr, mode: PlanMode) -> Result<Self, Vec<Diagnostic>> {
        let mut diags = ir.validate();
        if diags.iter().any(Diagnostic::is_error) {
            return Err(diags);
        }
        let specs = ir.propagate_image_specs()?;
        let order = ir.graph.topo_order().map_err(|d| vec![d])?;

        let mut plan = Self {
            mode,
            buffers: Vec::new(),
            steps: Vec::new(),
            arena_elems: 0,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            rings: BTreeMap::new(),
            warnings: Vec::new(),
        };
        // node -> the buffer its `out` port lives in.
        let mut produced: BTreeMap<NodeId, BufferId> = BTreeMap::new();

        for id in order {
            let node = &ir.graph.nodes[&id];
            let feeds = Self::feeds(ir, id, &produced, &mut diags);
            plan.lower(ir, id, node, &specs, &feeds, &mut produced, &mut diags);
        }

        for (name, out) in &ir.outputs {
            match produced.get(&out.port.node) {
                Some(b) => {
                    plan.outputs.insert(name.clone(), *b);
                }
                None => diags.push(
                    Diagnostic::new(
                        COMPILE_001,
                        format!("output \"{name}\" is produced by no plan step"),
                    )
                    .at(out.port.node),
                ),
            }
        }

        if diags.iter().any(Diagnostic::is_error) {
            return Err(diags);
        }
        plan.warnings = diags;
        Ok(plan)
    }

    /// The buffers feeding `id`, in declared input-port order.
    fn feeds(
        ir: &ObservationIr,
        id: NodeId,
        produced: &BTreeMap<NodeId, BufferId>,
        diags: &mut Vec<Diagnostic>,
    ) -> Vec<BufferId> {
        let mut edges: Vec<_> = ir
            .graph
            .edges
            .iter()
            .filter(|e| e.to.node == id)
            .map(|e| (port_index(&e.to.port), e.from.node))
            .collect();
        edges.sort_unstable();
        edges
            .into_iter()
            .filter_map(|(_, from)| {
                produced.get(&from).copied().or_else(|| {
                    diags.push(
                        Diagnostic::new(COMPILE_001, "input is fed by an unlowered node").at(id),
                    );
                    None
                })
            })
            .collect()
    }

    fn alloc(&mut self, node: NodeId, dtype: ElemType, shape: Vec<u64>, home: Home) -> BufferId {
        let elems = shape.iter().product::<u64>() as usize;
        if let Home::Arena(_) = home {
            self.arena_elems += elems;
        }
        self.buffers.push(BufferDesc {
            dtype,
            shape,
            elems,
            home,
            node,
        });
        BufferId(self.buffers.len() - 1)
    }

    fn emit(
        &mut self,
        produced: &mut BTreeMap<NodeId, BufferId>,
        id: NodeId,
        feeds: &[BufferId],
        op: Op,
        out: BufferId,
    ) {
        self.steps.push(Step {
            node: id,
            op,
            inputs: feeds.to_vec(),
            out,
        });
        produced.insert(id, out);
    }

    fn alloc_arena(&mut self, node: NodeId, dtype: ElemType, shape: Vec<u64>) -> BufferId {
        let at = self.arena_elems;
        self.alloc(node, dtype, shape, Home::Arena(at))
    }

    /// Geometry of the image on a node's `out` port, as `(c, h, w)`.
    fn chw(specs: &BTreeMap<PortRef, ImageSpec>, id: NodeId) -> Option<(usize, usize, usize)> {
        let s = specs.get(&PortRef::new(id, OUT))?;
        Some((
            channel_count(s.channels),
            s.height as usize,
            s.width as usize,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower(
        &mut self,
        ir: &ObservationIr,
        id: NodeId,
        node: &ObservationNode,
        specs: &BTreeMap<PortRef, ImageSpec>,
        feeds: &[BufferId],
        produced: &mut BTreeMap<NodeId, BufferId>,
        diags: &mut Vec<Diagnostic>,
    ) {
        let ty = node.io().output.clone();
        let shape = ty.shape.dims().to_vec();
        // The geometry of what arrives, from the *producer's* propagated ImageSpec (INV-14:
        // the plan reads intrinsics-carrying specs, it never recomputes geometry itself).
        let src_geom = feeds
            .first()
            .and_then(|b| Self::chw(specs, self.buffers[b.0].node));
        let dst_geom = Self::chw(specs, id);

        match node {
            ObservationNode::ImageInput { sensor, .. }
            | ObservationNode::StateInput { source: sensor, .. } => {
                let name = sensor.to_string();
                let b = self.alloc(id, ty.elem, shape, Home::Input(name.clone()));
                self.inputs.insert(name, b);
                produced.insert(id, b);
            }
            ObservationNode::Dequantize { .. } => {
                let Some((c, h, w)) = src_geom else {
                    diags.push(unsupported("Dequantize without an image input", id));
                    return;
                };
                let out = self.alloc_arena(id, ElemType::F32, shape);
                self.emit(produced, id, feeds, Op::Dequantize { h, w, c }, out);
            }
            ObservationNode::ColorTransform { src, dst, .. } => {
                use es_ir::image::ColorSpace::{Linear, SRgb};
                if (*src, *dst) != (SRgb, Linear) {
                    diags.push(unsupported(
                        &format!("ColorTransform {src:?} -> {dst:?}"),
                        id,
                    ));
                    return;
                }
                let from_u8 = feeds
                    .first()
                    .is_some_and(|b| self.buffers[b.0].dtype == ElemType::U8);
                let out = self.alloc_arena(id, ElemType::F32, shape);
                if from_u8 {
                    // Fused dequantize + EOTF through the LUT: the input takes 256 values, so
                    // the transcendental is evaluated 256 times instead of once per pixel.
                    let Some((c, h, w)) = src_geom else {
                        diags.push(unsupported("ColorTransform without an image input", id));
                        return;
                    };
                    self.emit(produced, id, feeds, Op::SrgbToLinearU8 { h, w, c }, out);
                } else {
                    self.emit(produced, id, feeds, Op::SrgbToLinear, out);
                }
            }
            ObservationNode::Resize { filter, .. } => {
                if !matches!(filter, ResizeFilter::Bilinear | ResizeFilter::Nearest) {
                    diags.push(unsupported(&format!("Resize with {filter:?}"), id));
                    return;
                }
                let (Some((c, sh, sw)), Some((_, dh, dw))) = (src_geom, dst_geom) else {
                    diags.push(unsupported("Resize without an image input", id));
                    return;
                };
                let out = self.alloc_arena(id, ElemType::F32, shape);
                let op = Op::Resize {
                    filter: *filter,
                    sw,
                    sh,
                    c,
                    dw,
                    dh,
                };
                self.emit(produced, id, feeds, op, out);
            }
            ObservationNode::Crop { mode, .. } => {
                let Some((c, sh, sw)) = src_geom else {
                    diags.push(unsupported("Crop without an image input", id));
                    return;
                };
                // Same rectangle the type system used, so pixels and intrinsics agree.
                let src_spec = specs[&PortRef::new(self.buffers[feeds[0].0].node, OUT)];
                let rect = crop_rect(*mode, &src_spec);
                if rect.x + rect.width > sw as u32 || rect.y + rect.height > sh as u32 {
                    diags.push(
                        Diagnostic::new(
                            COMPILE_003,
                            format!(
                                "crop {},{} {}x{} leaves a {sw}x{sh} image",
                                rect.x, rect.y, rect.width, rect.height
                            ),
                        )
                        .at(id),
                    );
                    return;
                }
                let out = self.alloc_arena(id, ElemType::F32, shape);
                self.emit(produced, id, feeds, Op::Crop { sw, sh, c, rect }, out);
            }
            ObservationNode::Normalize { stats, .. } => {
                let elems = shape.iter().product::<u64>() as usize;
                let op = match stats {
                    NormalizeStats::Range { lo, hi } => Op::NormalizeRange {
                        lo: *lo as f32,
                        hi: *hi as f32,
                    },
                    NormalizeStats::MeanStd { mean, std } => {
                        let c = src_geom.map_or(mean.len(), |(c, _, _)| c);
                        if mean.len() != std.len() || mean.len() != c || c == 0 || elems % c != 0 {
                            diags.push(
                                Diagnostic::new(
                                    COMPILE_004,
                                    format!(
                                        "Normalize has {} mean and {} std values for {c} channels",
                                        mean.len(),
                                        std.len()
                                    ),
                                )
                                .at(id),
                            );
                            return;
                        }
                        Op::NormalizeMeanStd {
                            plane: elems / c,
                            mean: mean.iter().map(|v| *v as f32).collect(),
                            std: std.iter().map(|v| *v as f32).collect(),
                        }
                    }
                };
                let out = self.alloc_arena(id, ElemType::F32, shape);
                self.emit(produced, id, feeds, op, out);
            }
            ObservationNode::Concat { axis, .. } | ObservationNode::Stack { axis, .. } => {
                let axis = *axis as usize;
                if axis >= shape.len() {
                    diags.push(
                        Diagnostic::new(COMPILE_005, format!("join axis {axis} is out of range"))
                            .at(id),
                    );
                    return;
                }
                let outer = shape[..axis].iter().product::<u64>() as usize;
                let chunk: Vec<usize> = feeds
                    .iter()
                    .map(|b| self.buffers[b.0].elems.checked_div(outer).unwrap_or(0))
                    .collect();
                let total: usize = chunk.iter().sum::<usize>() * outer;
                if total != shape.iter().product::<u64>() as usize {
                    diags.push(
                        Diagnostic::new(
                            COMPILE_005,
                            format!(
                                "join inputs hold {total} elements, the output declares {}",
                                shape.iter().product::<u64>()
                            ),
                        )
                        .at(id),
                    );
                    return;
                }
                let out = self.alloc_arena(id, ElemType::F32, shape);
                self.emit(produced, id, feeds, Op::Join { chunk, outer }, out);
            }
            ObservationNode::TemporalWindowNode { window, .. } => {
                if window.align != Align::Hold {
                    diags.push(unsupported(
                        &format!("TemporalWindow {:?}", window.align),
                        id,
                    ));
                    return;
                }
                let Some(&frame) = feeds.first() else {
                    diags.push(unsupported("TemporalWindow without an input", id));
                    return;
                };
                let slot = self.buffers[frame.0].elems;
                let depth = ir
                    .temporal
                    .history
                    .values()
                    .map(|h| h.depth as usize)
                    .max()
                    .unwrap_or_else(|| window.required_depth() as usize)
                    .max(window.required_depth() as usize);
                let out = self.alloc_arena(id, ElemType::F32, shape);
                self.rings.insert(
                    id,
                    Ring {
                        slot,
                        depth,
                        data: vec![0.0; slot * depth],
                        cursor: 0,
                        pushed: 0,
                    },
                );
                let op = Op::Window {
                    slot,
                    depth,
                    n: window.n_steps as usize,
                    stride: window.stride as usize,
                };
                self.emit(produced, id, feeds, op, out);
            }
            other => diags.push(unsupported(other.kind(), id)),
        }

        // A declared output element type narrower than f32 is an explicit cast step, never an
        // implicit one (spec 3.3: the pipeline is f32 by default, f16/bf16 on request).
        if let Some(&b) = produced.get(&id) {
            let want = ty.elem;
            if matches!(want, ElemType::F16 | ElemType::Bf16) && self.buffers[b.0].dtype != want {
                let shape = self.buffers[b.0].shape.clone();
                let out = self.alloc_arena(id, want, shape);
                self.steps.push(Step {
                    node: id,
                    op: Op::Cast { to: want },
                    inputs: vec![b],
                    out,
                });
                produced.insert(id, out);
            }
        }
    }

    /// The `compiler` slot of `execution_hash` (spec 5.3, spec 11.2): crate version, plan
    /// mode, and the kernel table in order. Ids are hashed rather than source, so a numerics
    /// change means bumping an id — which is the point.
    pub fn compiler_hash(&self) -> [u8; 32] {
        let mut w = CanonWriter::new();
        w.str("es.compiler_hash.observation.v1");
        w.str(env!("CARGO_PKG_VERSION"));
        w.str(self.mode.tag());
        w.seq(KERNEL_IDS.len());
        for k in KERNEL_IDS {
            w.str(k);
        }
        w.hash().expect("no float is written")
    }
}

/// The rectangle a [`CropMode`] resolves to — the same one `ObservationNode`'s type
/// propagation used, so the plan's pixels and the propagated intrinsics cannot disagree
/// (`INV-14`). `Random` resolves to the centred rectangle: a per-sample offset is
/// augmentation, and augmentation is `training_only` (spec 7.3).
fn crop_rect(mode: CropMode, spec: &ImageSpec) -> Rect {
    match mode {
        CropMode::Rect(r) => r,
        CropMode::Center { width, height } | CropMode::Random { width, height } => Rect {
            x: spec.width.saturating_sub(width) / 2,
            y: spec.height.saturating_sub(height) / 2,
            width,
            height,
        },
    }
}
