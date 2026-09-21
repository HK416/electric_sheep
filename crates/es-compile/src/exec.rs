//! Running a [`CpuPlan`] (spec 11.3). The plan decided everything but the pixel values.
//!
//! `es-ir` has no runtime tensor type and must not grow one — a tensor is a runtime object and
//! the IR is a contract — so the minimal one lives here.

use std::collections::BTreeMap;

use es_ir::observation::ResizeFilter;
use es_ir::types::ElemType;

use crate::kernels;
use crate::plan::{BufferId, CpuPlan, Home, Op};

/// Element width in bytes. `Bool` is one byte, as in every tensor library.
// `pub(crate)` for `crate::gpu::exec`, which validates the same inputs before uploading them.
pub(crate) fn width(dtype: ElemType) -> usize {
    match dtype {
        ElemType::F64 => 8,
        ElemType::F32 | ElemType::I32 => 4,
        ElemType::F16 | ElemType::Bf16 => 2,
        ElemType::U8 | ElemType::Bool => 1,
    }
}

/// A tensor the plan produced: row-major, tightly packed, no batch axis (spec 5.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub dtype: ElemType,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn as_ref(&self) -> TensorRef<'_> {
        TensorRef {
            dtype: self.dtype,
            shape: self.shape.clone(),
            data: &self.data,
        }
    }
}

/// A tensor the caller supplies. Borrowed, because the sensor boundary owns its frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRef<'a> {
    pub dtype: ElemType,
    pub shape: Vec<u64>,
    pub data: &'a [u8],
}

impl<'a> TensorRef<'a> {
    pub fn new(dtype: ElemType, shape: impl Into<Vec<u64>>, data: &'a [u8]) -> Self {
        Self {
            dtype,
            shape: shape.into(),
            data,
        }
    }
}

/// What `run` returns: the names Learning IR binds to (spec 7.4).
pub type Outputs = BTreeMap<String, Tensor>;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExecError {
    #[error("input \"{0}\" was not supplied")]
    MissingInput(String),
    #[error("input \"{name}\": plan expects {want:?} {want_shape:?}, got {got:?} {got_shape:?}")]
    InputMismatch {
        name: String,
        want: ElemType,
        want_shape: Vec<u64>,
        got: ElemType,
        got_shape: Vec<u64>,
    },
    #[error("input \"{name}\": {want} elements declared, {got} bytes of payload")]
    InputSize {
        name: String,
        want: usize,
        got: usize,
    },
    #[error("no kernel for {0} on this path")]
    Unsupported(&'static str),
}

impl CpuPlan {
    /// Execute the plan over one set of inputs.
    ///
    /// `&mut self` because the `TemporalWindow` rings are state (spec 7.5 layer 1): a window
    /// is defined over successive calls, so consecutive `run`s are a stream, not independent.
    pub fn run(&mut self, inputs: &BTreeMap<String, TensorRef<'_>>) -> Result<Outputs, ExecError> {
        // One arena, allocated once (spec 11.1 `Memory Plan`). Neither mode aliases it.
        let mut arena = vec![0.0f32; self.arena_elems];
        // Cloned so the ring state (`&mut self.rings`) stays reachable inside the loop; the
        // step list is a few dozen entries of shape metadata.
        let steps = self.steps.clone();

        for step in &steps {
            let (dst_at, dst_len) = match &self.buffers[step.out.0].home {
                Home::Arena(at) => (*at, self.buffers[step.out.0].elems),
                Home::Input(_) => continue, // an input buffer: nothing to compute
            };

            // Gather inputs first: a kernel takes `&[..]` and `&mut [..]`, and an arena
            // buffer can be on both sides, so the read side is copied out.
            let mut f32_in: Vec<Vec<f32>> = Vec::with_capacity(step.inputs.len());
            let mut u8_in: Vec<&[u8]> = Vec::with_capacity(step.inputs.len());
            for b in &step.inputs {
                let d = &self.buffers[b.0];
                match &d.home {
                    Home::Arena(at) => f32_in.push(arena[*at..*at + d.elems].to_vec()),
                    Home::Input(name) => {
                        let t = read_input(inputs, name, d.dtype, &d.shape, d.elems)?;
                        if d.dtype == ElemType::U8 {
                            u8_in.push(t);
                        } else {
                            f32_in.push(
                                t.chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect(),
                            );
                        }
                    }
                }
            }
            let dst = &mut arena[dst_at..dst_at + dst_len];

            match &step.op {
                Op::Dequantize { h, w, c } => {
                    kernels::cast_u8_hwc_to_f32_chw(u8_in[0], *h, *w, *c, dst);
                }
                Op::SrgbToLinearU8 { h, w, c } => {
                    // Fused dequantize + EOTF: 256 evaluations instead of one per pixel.
                    let lut = kernels::srgb_to_linear_lut();
                    let src = u8_in[0];
                    for ch in 0..*c {
                        for y in 0..*h {
                            for x in 0..*w {
                                dst[ch * h * w + y * w + x] =
                                    lut[src[(y * w + x) * c + ch] as usize];
                            }
                        }
                    }
                }
                Op::SrgbToLinear => kernels::srgb_to_linear(&f32_in[0], dst),
                Op::Resize {
                    filter,
                    sw,
                    sh,
                    c,
                    dw,
                    dh,
                } => match filter {
                    ResizeFilter::Bilinear => {
                        kernels::resize_bilinear(&f32_in[0], *sw, *sh, *c, *dw, *dh, dst);
                    }
                    ResizeFilter::Nearest => {
                        kernels::resize_nearest(&f32_in[0], *sw, *sh, *c, *dw, *dh, dst);
                    }
                    other => return Err(ExecError::Unsupported(filter_name(*other))),
                },
                Op::Crop { sw, sh, c, rect } => {
                    kernels::crop(&f32_in[0], *sw, *sh, *c, *rect, dst);
                }
                // Replicate padding, written here rather than in `kernels` because it has no
                // kernel id: `KERNEL_IDS` is append-only *with* a `compiler_hash` move, and a
                // packet that may not move a committed hash cannot append to it (packet M7/T6,
                // design note `observation-lowering.md` section 3). Same fixed loop order as
                // every kernel beside it.
                Op::Pad {
                    sw,
                    sh,
                    c,
                    left,
                    top,
                    right,
                    bottom,
                } => {
                    let (dw, dh) = (sw + left + right, sh + top + bottom);
                    let src = &f32_in[0];
                    for ch in 0..*c {
                        for y in 0..dh {
                            let sy = y.saturating_sub(*top).min(sh - 1);
                            for x in 0..dw {
                                let sx = x.saturating_sub(*left).min(sw - 1);
                                dst[ch * dh * dw + y * dw + x] = src[ch * sh * sw + sy * sw + sx];
                            }
                        }
                    }
                }
                Op::NormalizeMeanStd { plane, mean, std } => {
                    kernels::normalize_mean_std(&f32_in[0], *plane, mean, std, dst);
                }
                Op::NormalizeRange { lo, hi } => {
                    kernels::normalize_range(&f32_in[0], *lo, *hi, dst);
                }
                Op::Join { chunk, outer } => {
                    let srcs: Vec<&[f32]> = f32_in.iter().map(Vec::as_slice).collect();
                    kernels::concat(&srcs, chunk, *outer, dst);
                }
                Op::Window {
                    slot,
                    depth,
                    n,
                    stride,
                } => {
                    let ring = self
                        .rings
                        .get_mut(&step.node)
                        .expect("compile inserts a ring for every window step");
                    // The cursor names the newest frame, so it only advances after the first.
                    if ring.pushed > 0 {
                        ring.cursor += 1;
                    }
                    ring.pushed += 1;
                    kernels::history_push(&mut ring.data, *slot, *depth, ring.cursor, &f32_in[0]);
                    kernels::window_gather(
                        &ring.data,
                        *slot,
                        *depth,
                        ring.cursor,
                        ring.pushed,
                        *n,
                        *stride,
                        dst,
                    );
                }
                // The narrowing happens in `materialize`, so the arena keeps the f32 value
                // that the per-node debug view (spec 11.5) is there to show.
                Op::Cast { .. } => dst.copy_from_slice(&f32_in[0]),
            }
        }

        let mut out = Outputs::new();
        for (name, id) in &self.outputs {
            out.insert(name.clone(), self.materialize(*id, &arena, inputs)?);
        }
        Ok(out)
    }

    fn materialize(
        &self,
        id: BufferId,
        arena: &[f32],
        inputs: &BTreeMap<String, TensorRef<'_>>,
    ) -> Result<Tensor, ExecError> {
        let d = &self.buffers[id.0];
        let data = match &d.home {
            Home::Input(name) => read_input(inputs, name, d.dtype, &d.shape, d.elems)?.to_vec(),
            Home::Arena(at) => {
                let v = &arena[*at..*at + d.elems];
                match d.dtype {
                    ElemType::F16 => {
                        let mut bits = vec![0u16; d.elems];
                        kernels::cast_f32_to_f16(v, &mut bits);
                        bits.iter().flat_map(|b| b.to_le_bytes()).collect()
                    }
                    ElemType::Bf16 => {
                        let mut bits = vec![0u16; d.elems];
                        kernels::cast_f32_to_bf16(v, &mut bits);
                        bits.iter().flat_map(|b| b.to_le_bytes()).collect()
                    }
                    _ => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                }
            }
        };
        Ok(Tensor {
            dtype: d.dtype,
            shape: d.shape.clone(),
            data,
        })
    }

    /// The f32 value of any node's output buffer after a `run` — spec 11.5's "per-node values:
    /// all". Available in both modes here, because neither aliases the arena.
    pub fn buffer_of(&self, node: es_ir::NodeId) -> Option<BufferId> {
        self.buffers
            .iter()
            .rposition(|b| b.node == node)
            .map(BufferId)
    }
}

// `pub(crate)` for `crate::gpu::exec`: the GPU path must reject a mismatched input with the
// same `ExecError` the CPU path does, not a different one.
pub(crate) fn read_input<'a>(
    inputs: &BTreeMap<String, TensorRef<'a>>,
    name: &str,
    dtype: ElemType,
    shape: &[u64],
    elems: usize,
) -> Result<&'a [u8], ExecError> {
    let t = inputs
        .get(name)
        .ok_or_else(|| ExecError::MissingInput(name.to_owned()))?;
    if t.dtype != dtype || t.shape != shape {
        return Err(ExecError::InputMismatch {
            name: name.to_owned(),
            want: dtype,
            want_shape: shape.to_vec(),
            got: t.dtype,
            got_shape: t.shape.clone(),
        });
    }
    if t.data.len() != elems * width(dtype) {
        return Err(ExecError::InputSize {
            name: name.to_owned(),
            want: elems * width(dtype),
            got: t.data.len(),
        });
    }
    Ok(t.data)
}

fn filter_name(f: ResizeFilter) -> &'static str {
    match f {
        ResizeFilter::Nearest => "Resize{Nearest}",
        ResizeFilter::Bilinear => "Resize{Bilinear}",
        ResizeFilter::Bicubic => "Resize{Bicubic}",
        ResizeFilter::Area => "Resize{Area}",
    }
}
