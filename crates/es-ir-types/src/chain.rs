//! The execution hash chain of spec 5.3 / Appendix B.6.
//!
//! Digests only: nothing here computes a graph hash, so it sits with the canonical encoder
//! rather than with the graph canonicalizer in `es-ir` (spec 1.5 context budget;
//! `docs/packets/M4/P-M4-S16.md`). `es-ir` re-exports every item from `es_ir::hash`.

use serde::{Deserialize, Serialize};

use crate::canon::CanonWriter;

const CHAIN_TAG: &str = "es.execution_hash.v1";

/// `content + schema + split` — the split is part of the identity because the same data with a
/// different train/val/test split is a different training run (spec 5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetHash {
    pub content: [u8; 32],
    pub schema: [u8; 32],
    pub split: [u8; 32],
}

/// Digest of the capability report of the machine that ran the execution. The report itself is
/// produced outside this crate (GPU limits, driver, CPU features); the IR only needs its
/// identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareCapability(pub [u8; 32]);

/// The hash chain of spec 5.3 / Appendix B.6.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashChain {
    pub asset: Vec<[u8; 32]>,
    pub scene: [u8; 32],
    /// Authoring identity: includes user-assigned node ids (spec 11.2).
    pub task_graph: [u8; 32],
    /// Semantic identity: after normalization and relabelling.
    pub task: [u8; 32],
    pub observation: [u8; 32],
    pub learning: [u8; 32],
    pub policy: [u8; 32],
    pub dataset: DatasetHash,
    pub deployment: [u8; 32],
    pub evaluation: Option<[u8; 32]>,
    pub compiler: [u8; 32],
    pub runtime: [u8; 32],
    pub hardware: HardwareCapability,
}

/// One component of the chain; the unit in which `diff` reports change (spec 27.1
/// `revalidation_trigger`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ChangedComponent {
    Asset,
    Scene,
    TaskGraph,
    Task,
    Observation,
    Learning,
    Policy,
    Dataset,
    Deployment,
    Evaluation,
    Compiler,
    Runtime,
    Hardware,
}

impl HashChain {
    /// `H(task, observation, learning, policy, dataset, deployment, compiler, runtime,
    /// hardware_capability)` — the tier-1 reproducibility condition (spec 5.3).
    ///
    /// `asset`, `scene` and `task_graph` are upstream of `task` and `evaluation` does not
    /// affect what is executed, so none of them are inputs.
    pub fn execution_hash(&self) -> [u8; 32] {
        let mut w = CanonWriter::new();
        w.str(CHAIN_TAG);
        for d in [
            &self.task,
            &self.observation,
            &self.learning,
            &self.policy,
            &self.dataset.content,
            &self.dataset.schema,
            &self.dataset.split,
            &self.deployment,
            &self.compiler,
            &self.runtime,
            &self.hardware.0,
        ] {
            w.digest(d);
        }
        w.hash()
            .expect("the hash chain contains no floats, so encoding cannot fail")
    }

    /// What changed between two chains, in declaration order.
    pub fn diff(&self, other: &Self) -> Vec<ChangedComponent> {
        use ChangedComponent as C;
        let mut out = Vec::new();
        let mut push = |changed: bool, c: C| {
            if changed {
                out.push(c);
            }
        };
        push(self.asset != other.asset, C::Asset);
        push(self.scene != other.scene, C::Scene);
        push(self.task_graph != other.task_graph, C::TaskGraph);
        push(self.task != other.task, C::Task);
        push(self.observation != other.observation, C::Observation);
        push(self.learning != other.learning, C::Learning);
        push(self.policy != other.policy, C::Policy);
        push(self.dataset != other.dataset, C::Dataset);
        push(self.deployment != other.deployment, C::Deployment);
        push(self.evaluation != other.evaluation, C::Evaluation);
        push(self.compiler != other.compiler, C::Compiler);
        push(self.runtime != other.runtime, C::Runtime);
        push(self.hardware != other.hardware, C::Hardware);
        out
    }
}
