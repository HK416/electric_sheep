//! Running a [`GpuPlan`] (spec 7.6): upload, record every dispatch on the single compute
//! queue, submit once, download.
//!
//! One submission per `run`, in recording order, with a full barrier between dispatches
//! (`es_gpu::CommandRecorder`) — spec 3.4's "deterministic scheduling: static partitioning,
//! single queue". Nothing here is asynchronous and nothing overlaps, which is also why two
//! runs over the same input have to be bit-identical.

use std::collections::BTreeMap;

use es_gpu::{CommandRecorder, GpuError};
use es_ir::types::ElemType;

use crate::exec::{read_input, Outputs, Tensor, TensorRef};
use crate::gpu::plan::GpuPlan;
use crate::plan::{Home, Op};
use crate::ExecError;

#[derive(Debug, thiserror::Error)]
pub enum GpuRunError {
    #[error(transparent)]
    Exec(#[from] ExecError),
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

impl GpuPlan<'_> {
    /// Execute the plan over one set of inputs.
    ///
    /// `&mut self` for the same reason [`crate::CpuPlan::run`] takes it: the history rings
    /// are state and consecutive runs are a stream (spec 7.5 layer 1).
    pub fn run(
        &mut self,
        inputs: &BTreeMap<String, TensorRef<'_>>,
    ) -> Result<Outputs, GpuRunError> {
        // The whole arena is re-uploaded every run rather than patched in place. It costs one
        // transfer of a buffer that is already small, and it removes the only way two runs
        // with the same input could differ: a stale element nothing wrote.
        let mut arena = vec![0u8; self.layout.arena_elems * 4];
        let mut words = vec![0u8; self.layout.word_elems * 4];
        for (name, id) in &self.cpu.inputs {
            let d = &self.cpu.buffers[id.0];
            let src = read_input(inputs, name, d.dtype, &d.shape, d.elems)?;
            let at = match d.dtype {
                ElemType::U8 => self.layout.byte_at[id.0],
                _ => self.layout.at[id.0] * 4,
            };
            let dst = match d.dtype {
                ElemType::U8 => &mut words[at..at + src.len()],
                _ => &mut arena[at..at + src.len()],
            };
            dst.copy_from_slice(src);
        }

        // Ring cursors advance exactly as `CpuPlan::run` advances them; the kernels read
        // them out of the `state` buffer because a pipeline's defines are compile-time.
        let mut state = vec![0u8; self.layout.state_elems * 4];
        for step in &self.cpu.steps {
            if !matches!(step.op, Op::Window { .. }) {
                continue;
            }
            let ring = self
                .cpu
                .rings
                .get_mut(&step.node)
                .expect("compile inserts a ring for every window step");
            if ring.pushed > 0 {
                ring.cursor += 1;
            }
            ring.pushed += 1;
            let at = self.layout.state_at[&step.node] * 4;
            state[at..at + 4].copy_from_slice(&(ring.cursor as u32).to_le_bytes());
            state[at + 4..at + 8].copy_from_slice(&(ring.pushed as u32).to_le_bytes());
        }

        self.arena.upload(&arena)?;
        self.words.upload(&words)?;
        self.state.upload(&state)?;

        for p in &self.pipelines {
            p.reset_descriptors()?;
        }
        let mut recorder = CommandRecorder::new(self.gpu)?;
        let bound = [
            &self.arena,
            &self.words,
            &self.aux,
            &self.rings,
            &self.state,
        ];
        for (index, groups) in &self.calls {
            recorder.dispatch(&self.pipelines[*index], &bound, [*groups, 1, 1])?;
        }
        recorder.submit_and_wait()?;

        let arena = self.arena.download()?;
        let words = self.words.download()?;

        let mut out = Outputs::new();
        for (name, id) in &self.cpu.outputs {
            let d = &self.cpu.buffers[id.0];
            let data = match (&d.home, d.dtype) {
                (Home::Input(from), _) => {
                    read_input(inputs, from, d.dtype, &d.shape, d.elems)?.to_vec()
                }
                (_, ElemType::F16 | ElemType::Bf16) => {
                    // The cast kernel wrote the narrowed bit pattern one per `uint`; the
                    // arena still holds the f32, which is what spec 11.5's per-node view is.
                    let at = self.layout.narrow_at[id.0] * 4;
                    words[at..at + d.elems * 4]
                        .chunks_exact(4)
                        .flat_map(|w| [w[0], w[1]])
                        .collect()
                }
                _ => {
                    let at = self.layout.at[id.0] * 4;
                    arena[at..at + d.elems * 4].to_vec()
                }
            };
            out.insert(
                name.clone(),
                Tensor {
                    dtype: d.dtype,
                    shape: d.shape.clone(),
                    data,
                },
            );
        }
        Ok(out)
    }
}
