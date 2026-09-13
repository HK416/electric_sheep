//! Compiling the GPU plan (spec 11.4): one device arena, one pipeline per
//! `(kernel id, defines)`, one recorded dispatch list.

use es_gpu::{
    BindingDesc, BindingKind, Buffer, ComputePipeline, Gpu, GpuError, SlangCompiler, Usage,
};
use es_ir::diag::Diagnostic;
use es_ir::observation::ObservationIr;
use es_ir::CanonWriter;

use crate::gpu::{lower, KernelKey, Layout, COMPILE_006};
use crate::plan::{CpuPlan, PlanMode};

/// `slang/observation.slang`, embedded so a compiled binary carries its own kernels.
const SOURCE: &str = include_str!("../../slang/observation.slang");

/// `approx.slang` still has to be a file on disk, because `slangc` resolves `#include`
/// against `-I` directories. It is the only one.
fn math_slang_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../es-math/slang")
}

/// The five bindings of `slang/observation.slang`, fixed for every kernel.
const BINDINGS: [BindingDesc; 5] = [
    BindingDesc {
        binding: 0,
        kind: BindingKind::Storage,
    },
    BindingDesc {
        binding: 1,
        kind: BindingKind::Storage,
    },
    BindingDesc {
        binding: 2,
        kind: BindingKind::Storage,
    },
    BindingDesc {
        binding: 3,
        kind: BindingKind::Storage,
    },
    BindingDesc {
        binding: 4,
        kind: BindingKind::Storage,
    },
];

pub(crate) fn gpu_error(e: &GpuError) -> Vec<Diagnostic> {
    vec![Diagnostic::new(COMPILE_006, format!("GPU lowering: {e}"))]
}

/// The GPU mirror of [`CpuPlan`] (spec 11.4).
pub struct GpuPlan<'gpu> {
    pub(crate) gpu: &'gpu Gpu,
    /// The CPU plan this mirrors. Its `rings` carry the `cursor`/`pushed` state; the ring
    /// *data* lives on the device, so `Ring::data` is unused here.
    pub(crate) cpu: CpuPlan,
    pub(crate) layout: Layout,
    pub(crate) keys: Vec<KernelKey>,
    pub(crate) pipelines: Vec<ComputePipeline<'gpu>>,
    /// SPIR-V content hashes, pipeline order — the `compile_hash` ingredient of spec 11.4.
    pub(crate) module_hashes: Vec<String>,
    /// `(pipeline index, workgroup count)` in recorded order.
    pub(crate) calls: Vec<(usize, u32)>,
    pub(crate) arena: Buffer<'gpu>,
    pub(crate) words: Buffer<'gpu>,
    pub(crate) aux: Buffer<'gpu>,
    pub(crate) rings: Buffer<'gpu>,
    pub(crate) state: Buffer<'gpu>,
}

impl std::fmt::Debug for GpuPlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPlan")
            .field("mode", &self.cpu.mode)
            .field("pipelines", &self.pipelines.len())
            .field("dispatches", &self.calls.len())
            .field("arena_elems", &self.layout.arena_elems)
            .finish_non_exhaustive()
    }
}

fn bytes(elems: usize) -> u64 {
    (elems as u64) * 4
}

impl<'gpu> GpuPlan<'gpu> {
    /// Compile an Observation IR for `gpu`.
    ///
    /// Runs [`CpuPlan::compile`] first and keeps it: the topological order, the buffer table
    /// and the node→kernel choice are the CPU plan's, so the two paths cannot drift apart by
    /// compiling differently. Diagnostics are the CPU plan's, plus `COMPILE-006` for
    /// anything the device or `slangc` refuses.
    pub fn compile(
        gpu: &'gpu Gpu,
        ir: &ObservationIr,
        mode: PlanMode,
    ) -> Result<Self, Vec<Diagnostic>> {
        let cpu = CpuPlan::compile(ir, mode)?;
        let lowered = lower(&cpu, ir)?;

        // Spec 3.4 step 3: the execution modes are a *compile* input, derived from the
        // capability query. Nothing here sets a device state.
        let modes = gpu.deterministic_execution_modes();
        let compiler = SlangCompiler::new()
            .map_err(|e| gpu_error(&e))?
            .with_include(math_slang_dir());

        let mut keys: Vec<KernelKey> = Vec::new();
        let mut pipelines = Vec::new();
        let mut module_hashes = Vec::new();
        let mut calls = Vec::new();
        for d in &lowered.dispatches {
            let index = if let Some(i) = keys.iter().position(|k| *k == d.key) {
                i
            } else {
                let module = compiler
                    .compile(SOURCE, d.key.entry, "glsl_450", &d.key.defines, modes)
                    .map_err(|e| gpu_error(&e))?;
                let pipeline =
                    ComputePipeline::new(gpu, &module, &BINDINGS).map_err(|e| gpu_error(&e))?;
                keys.push(d.key.clone());
                module_hashes.push(module.hash);
                pipelines.push(pipeline);
                keys.len() - 1
            };
            calls.push((index, d.groups()));
        }

        let layout = lowered.layout;
        let mut plan = Self {
            arena: Buffer::new(gpu, bytes(layout.arena_elems), Usage::Storage)
                .map_err(|e| gpu_error(&e))?,
            words: Buffer::new(gpu, bytes(layout.word_elems), Usage::Storage)
                .map_err(|e| gpu_error(&e))?,
            aux: Buffer::from_f32(gpu, &lowered.aux, Usage::Storage).map_err(|e| gpu_error(&e))?,
            rings: Buffer::new(gpu, bytes(layout.ring_elems), Usage::Storage)
                .map_err(|e| gpu_error(&e))?,
            state: Buffer::new(gpu, bytes(layout.state_elems), Usage::Storage)
                .map_err(|e| gpu_error(&e))?,
            gpu,
            cpu,
            layout,
            keys,
            pipelines,
            module_hashes,
            calls,
        };
        // Vulkan does not promise zeroed memory, and the rings are read before they are
        // fully written (`Align::Hold` repeats the oldest slot).
        plan.reset().map_err(|e| gpu_error(&e))?;
        Ok(plan)
    }

    /// The pipelines [`compile`](Self::compile) will build, computed without a device.
    ///
    /// This is the CI-gateable half of the lowering: whether a plan needs one pipeline per
    /// kernel or twenty is decided here, on a machine with no Vulkan.
    pub fn pipeline_plan(
        ir: &ObservationIr,
        mode: PlanMode,
    ) -> Result<Vec<KernelKey>, Vec<Diagnostic>> {
        let cpu = CpuPlan::compile(ir, mode)?;
        let lowered = lower(&cpu, ir)?;
        let mut keys: Vec<KernelKey> = Vec::new();
        for d in lowered.dispatches {
            if !keys.contains(&d.key) {
                keys.push(d.key);
            }
        }
        Ok(keys)
    }

    /// The CPU plan this mirrors — the oracle, and where the warnings are.
    pub fn cpu(&self) -> &CpuPlan {
        &self.cpu
    }

    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    pub fn dispatch_count(&self) -> usize {
        self.calls.len()
    }

    pub fn keys(&self) -> &[KernelKey] {
        &self.keys
    }

    /// The `compiler` slot of `execution_hash` (spec 5.3, spec 11.2) for the GPU path: the
    /// CPU plan's hash — which covers the kernel table and the plan mode — plus every
    /// SPIR-V content hash, so editing a `.slang` file changes it even when no kernel id
    /// moved (spec 3.4 item 7).
    pub fn compiler_hash(&self) -> [u8; 32] {
        let mut w = CanonWriter::new();
        w.str("es.compiler_hash.observation.gpu.v1");
        w.digest(&self.cpu.compiler_hash());
        w.seq(self.module_hashes.len());
        for (key, hash) in self.keys.iter().zip(&self.module_hashes) {
            w.str(key.kernel);
            w.str(key.entry);
            w.str(hash);
        }
        w.hash().expect("no float is written")
    }

    /// Clear every history ring — the device mirror of [`CpuPlan::reset`].
    ///
    /// `es-eval` calls this after every `env.reset`, and it must leave the plan in exactly
    /// the state `compile` did, or the spec 10.1 table depends on suite order (spec 10.4).
    pub fn reset(&mut self) -> Result<(), GpuError> {
        self.cpu.reset();
        let zeros = vec![0u8; self.rings.size() as usize];
        self.rings.upload(&zeros)
    }
}
