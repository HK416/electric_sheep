//! Observation IR → GPU execution plan (spec 7.6, spec 11.4).
//!
//! The GPU path is not a second compiler. [`GpuPlan::compile`] runs [`CpuPlan::compile`] and
//! then *mirrors* it: same topological order, same buffer table, same kernel per node. What
//! it adds is one device arena, one compute pipeline per `(kernel id, defines)`, and a
//! recorded dispatch list. The claim the tests make is bit equality with the CPU reference
//! wherever `docs/design/observation-lowering.md` claims it, which is why nothing here is
//! free to choose a different index order or a different association.
//!
//! This module is the part that needs no GPU: the buffer layout and the pipeline list are
//! pure functions of the CPU plan, so [`GpuPlan::pipeline_plan`] can be gated in CI on a
//! machine with no Vulkan at all.

use std::collections::BTreeMap;

use es_ir::diag::Diagnostic;
use es_ir::graph::NodeId;
use es_ir::observation::{ObservationIr, ObservationNode, ResizeFilter};
use es_ir::types::ElemType;

use crate::kernels::{self, KERNEL_IDS};
use crate::plan::{BufferId, CpuPlan, Home, Op};

pub mod exec;
pub mod plan;

pub use exec::GpuRunError;
pub use plan::GpuPlan;

// TODO(codes-merge): move into `es_ir::codes` with `COMPILE-001..005` when the compiler
// codes are dictionaried.
pub const COMPILE_006: &str = "COMPILE-006";

/// Threads per workgroup. Matches `[numthreads(64, 1, 1)]` in `slang/observation.slang`.
pub const WORKGROUP: usize = 64;

fn unsupported(what: impl Into<String>) -> Vec<Diagnostic> {
    vec![Diagnostic::new(COMPILE_006, what).with_hint(
        "docs/design/observation-lowering.md section 11 lists the GPU mirror of each kernel",
    )]
}

/// One compute pipeline: a kernel id of [`KERNEL_IDS`], the Slang entry point that
/// implements it, and the defines that specialise it.
///
/// Two dispatches share a pipeline exactly when these three agree, so shapes and offsets
/// being defines means a plan that resizes twice to the same size compiles one pipeline.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KernelKey {
    /// An entry of [`KERNEL_IDS`] — the table stays the source of truth for what a kernel
    /// *is*, and `compiler_hash` still covers it.
    pub kernel: &'static str,
    /// Entry point in `slang/observation.slang`.
    pub entry: &'static str,
    pub defines: BTreeMap<String, String>,
}

/// One dispatch of one pipeline over `threads` output elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Dispatch {
    pub key: KernelKey,
    pub threads: usize,
}

impl Dispatch {
    pub(crate) fn groups(&self) -> u32 {
        u32::try_from(self.threads.div_ceil(WORKGROUP)).unwrap_or(u32::MAX)
    }
}

/// Where every plan buffer lives on the device.
///
/// Four device buffers, not one per plan buffer: a descriptor set per plan buffer would put
/// the offsets out of reach of the defines, and the defines are what make a pipeline a pure
/// function of `(kernel, shape)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Per plan buffer: element offset into the `arena` buffer.
    pub at: Vec<usize>,
    /// Per plan buffer: byte offset into the `words` buffer (u8 plan inputs only).
    pub byte_at: Vec<usize>,
    /// Per plan buffer: word index into the `words` buffer (f16/bf16 outputs only).
    pub narrow_at: Vec<usize>,
    pub arena_elems: usize,
    pub word_elems: usize,
    pub ring_at: BTreeMap<NodeId, usize>,
    pub state_at: BTreeMap<NodeId, usize>,
    pub ring_elems: usize,
    pub state_elems: usize,
}

/// What `compile` needs and what `pipeline_plan` exposes without a device.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Lowered {
    pub dispatches: Vec<Dispatch>,
    pub layout: Layout,
    /// Binding 2: the 256-entry sRGB LUT, then every `Normalize{MeanStd}` step's mean and
    /// std. Uploaded once at compile; it depends on no input.
    pub aux: Vec<f32>,
}

fn def(map: &mut BTreeMap<String, String>, k: &str, v: usize) {
    map.insert(k.to_owned(), v.to_string());
}

/// Assign every plan buffer a device home. Arena offsets are the CPU plan's own, so a
/// `BufferId` addresses the same elements on both sides.
fn layout(cpu: &CpuPlan) -> Result<Layout, Vec<Diagnostic>> {
    let mut l = Layout {
        at: vec![0; cpu.buffers.len()],
        byte_at: vec![0; cpu.buffers.len()],
        narrow_at: vec![0; cpu.buffers.len()],
        arena_elems: cpu.arena_elems,
        ..Layout::default()
    };
    for (i, b) in cpu.buffers.iter().enumerate() {
        match &b.home {
            Home::Arena(at) => l.at[i] = *at,
            Home::Input(name) => match b.dtype {
                ElemType::U8 => {
                    l.byte_at[i] = l.word_elems * 4;
                    l.word_elems += b.elems.div_ceil(4);
                }
                ElemType::F32 => {
                    l.at[i] = l.arena_elems;
                    l.arena_elems += b.elems;
                }
                other => {
                    return Err(unsupported(format!(
                        "plan input \"{name}\" is {other:?}; the GPU path carries u8 and f32"
                    )))
                }
            },
        }
        if matches!(b.dtype, ElemType::F16 | ElemType::Bf16) {
            l.narrow_at[i] = l.word_elems;
            l.word_elems += b.elems;
        }
    }
    for (node, ring) in &cpu.rings {
        l.ring_at.insert(*node, l.ring_elems);
        l.ring_elems += ring.slot * ring.depth;
        l.state_at.insert(*node, l.state_elems);
        l.state_elems += 2;
    }
    Ok(l)
}

/// Lower a compiled CPU plan into the dispatch list. No device is touched.
pub(crate) fn lower(cpu: &CpuPlan, ir: &ObservationIr) -> Result<Lowered, Vec<Diagnostic>> {
    let layout = layout(cpu)?;
    let mut aux: Vec<f32> = kernels::srgb_to_linear_lut().to_vec();
    let mut dispatches = Vec::new();

    let src_at = |b: BufferId| layout.at[b.0];
    let byte_at = |b: BufferId| layout.byte_at[b.0];

    for step in &cpu.steps {
        let out = step.out;
        let n = cpu.buffers[out.0].elems;
        let dst = layout.at[out.0];
        let mut d = BTreeMap::new();
        def(&mut d, "ES_N", n);
        def(&mut d, "ES_DST", dst);

        let (kernel, entry) = match &step.op {
            Op::Dequantize { h, w, c } => {
                def(&mut d, "ES_SRC", byte_at(step.inputs[0]));
                def(&mut d, "ES_H", *h);
                def(&mut d, "ES_W", *w);
                def(&mut d, "ES_C", *c);
                ("cast_u8_hwc_to_f32_chw.v1", "cast_u8_hwc_to_f32_chw")
            }
            Op::SrgbToLinearU8 { h, w, c } => {
                def(&mut d, "ES_SRC", byte_at(step.inputs[0]));
                def(&mut d, "ES_H", *h);
                def(&mut d, "ES_W", *w);
                def(&mut d, "ES_C", *c);
                def(&mut d, "ES_AUX", 0);
                ("srgb_to_linear.v1", "srgb_to_linear_u8")
            }
            Op::SrgbToLinear => {
                def(&mut d, "ES_SRC", src_at(step.inputs[0]));
                ("srgb_to_linear.v1", "srgb_to_linear")
            }
            Op::Resize {
                filter,
                sw,
                sh,
                c,
                dw,
                dh,
            } => {
                def(&mut d, "ES_SRC", src_at(step.inputs[0]));
                def(&mut d, "ES_SW", *sw);
                def(&mut d, "ES_SH", *sh);
                def(&mut d, "ES_C", *c);
                def(&mut d, "ES_DW", *dw);
                def(&mut d, "ES_DH", *dh);
                match filter {
                    ResizeFilter::Bilinear => ("resize_bilinear.v1", "resize_bilinear"),
                    ResizeFilter::Nearest => ("resize_nearest.v1", "resize_nearest"),
                    other => return Err(unsupported(format!("Resize with {other:?}"))),
                }
            }
            Op::Crop { sw, sh, c, rect } => {
                def(&mut d, "ES_SRC", src_at(step.inputs[0]));
                def(&mut d, "ES_SW", *sw);
                def(&mut d, "ES_SH", *sh);
                def(&mut d, "ES_C", *c);
                def(&mut d, "ES_RX", rect.x as usize);
                def(&mut d, "ES_RY", rect.y as usize);
                def(&mut d, "ES_RW", rect.width as usize);
                def(&mut d, "ES_RH", rect.height as usize);
                ("crop.v1", "crop")
            }
            // Refused by name, not mirrored (packet M7/T6): a Slang kernel needs an entry in
            // `KERNEL_IDS`, and appending one moves `compiler_hash` for every plan in the
            // repository — including the committed training and evaluation runs T6 may not
            // move. A silent identity here would be worse than a refusal: the CPU plan pads
            // and this one would not, and the two paths exist to be bit-equal.
            Op::Pad { .. } => {
                return Err(unsupported(
                    "Pad has no GPU kernel: it is lowered on the CPU reference path only \
                     (design note section 3). Bake with the CPU plan, or drop the Pad node \
                     from the document this GPU plan compiles",
                ))
            }
            Op::NormalizeMeanStd { plane, mean, std } => {
                def(&mut d, "ES_SRC", src_at(step.inputs[0]));
                def(&mut d, "ES_PLANE", *plane);
                def(&mut d, "ES_C", mean.len());
                def(&mut d, "ES_AUX", aux.len());
                aux.extend_from_slice(mean);
                aux.extend_from_slice(std);
                ("normalize_mean_std.v1", "normalize_mean_std")
            }
            Op::NormalizeRange { lo, hi } => {
                def(&mut d, "ES_SRC", src_at(step.inputs[0]));
                def(&mut d, "ES_LO_BITS", lo.to_bits() as usize);
                def(&mut d, "ES_HI_BITS", hi.to_bits() as usize);
                ("normalize_range.v1", "normalize_range")
            }
            Op::Join { chunk, outer } => {
                // One dispatch per input: the destination's slot for input `i` is a fixed
                // offset, so a copy kernel needs no variable-length define list.
                let stride: usize = chunk.iter().sum();
                let stack = matches!(
                    ir.graph.nodes.get(&step.node),
                    Some(ObservationNode::Stack { .. })
                );
                let (kernel, entry) = if stack {
                    ("stack.v1", "stack_copy")
                } else {
                    ("concat.v1", "concat_copy")
                };
                let mut lead = 0usize;
                for (b, c) in step.inputs.iter().zip(chunk) {
                    let mut d = BTreeMap::new();
                    def(&mut d, "ES_N", c * outer);
                    def(&mut d, "ES_DST", dst);
                    def(&mut d, "ES_SRC", src_at(*b));
                    def(&mut d, "ES_CHUNK", *c);
                    def(&mut d, "ES_STRIDE_OUT", stride);
                    def(&mut d, "ES_LEAD", lead);
                    dispatches.push(Dispatch {
                        key: KernelKey {
                            kernel,
                            entry,
                            defines: d,
                        },
                        threads: c * outer,
                    });
                    lead += c;
                }
                continue;
            }
            Op::Window {
                slot,
                depth,
                n: steps,
                stride,
            } => {
                let ring = layout.ring_at[&step.node];
                let state = layout.state_at[&step.node];
                let mut push = BTreeMap::new();
                def(&mut push, "ES_N", *slot);
                def(&mut push, "ES_SRC", src_at(step.inputs[0]));
                def(&mut push, "ES_SLOT", *slot);
                def(&mut push, "ES_DEPTH", *depth);
                def(&mut push, "ES_RING", ring);
                def(&mut push, "ES_STATE", state);
                dispatches.push(Dispatch {
                    key: KernelKey {
                        kernel: "history_push.v1",
                        entry: "history_push",
                        defines: push,
                    },
                    threads: *slot,
                });

                def(&mut d, "ES_SLOT", *slot);
                def(&mut d, "ES_DEPTH", *depth);
                def(&mut d, "ES_RING", ring);
                def(&mut d, "ES_STATE", state);
                def(&mut d, "ES_STEPS", *steps);
                def(&mut d, "ES_STEP_STRIDE", *stride);
                ("window_gather.v1", "window_gather")
            }
            Op::Cast { to } => {
                def(&mut d, "ES_SRC", src_at(step.inputs[0]));
                def(&mut d, "ES_NARROW", layout.narrow_at[out.0]);
                match to {
                    ElemType::F16 => ("cast_f32_to_f16.v1", "cast_f32_to_f16"),
                    ElemType::Bf16 => ("cast_f32_to_bf16.v1", "cast_f32_to_bf16"),
                    other => return Err(unsupported(format!("Cast to {other:?}"))),
                }
            }
        };

        debug_assert!(
            KERNEL_IDS.contains(&kernel),
            "{kernel} is not a KERNEL_IDS id"
        );
        dispatches.push(Dispatch {
            key: KernelKey {
                kernel,
                entry,
                defines: d,
            },
            threads: n,
        });
    }

    Ok(Lowered {
        dispatches,
        layout,
        aux,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_lowered_kernel_id_is_in_the_table() {
        // The mapping in `lower` names ids as literals; this is the guard that one of them
        // cannot drift away from `KERNEL_IDS`, which `compiler_hash` covers.
        for id in [
            "cast_u8_hwc_to_f32_chw.v1",
            "srgb_to_linear.v1",
            "resize_bilinear.v1",
            "resize_nearest.v1",
            "crop.v1",
            "normalize_mean_std.v1",
            "normalize_range.v1",
            "concat.v1",
            "stack.v1",
            "history_push.v1",
            "window_gather.v1",
            "cast_f32_to_f16.v1",
            "cast_f32_to_bf16.v1",
        ] {
            assert!(KERNEL_IDS.contains(&id), "{id}");
        }
    }

    #[test]
    fn a_dispatch_covers_every_thread() {
        let key = KernelKey {
            kernel: "crop.v1",
            entry: "crop",
            defines: BTreeMap::new(),
        };
        for threads in [1usize, 63, 64, 65, 1000] {
            let d = Dispatch {
                key: key.clone(),
                threads,
            };
            assert!(d.groups() as usize * WORKGROUP >= threads, "{threads}");
        }
    }
}
