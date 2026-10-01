//! The loop ledger, `loop.jsonl` (spec 13.3): what each step of the learning loop consumed and
//! produced, one line per step, and the chain check across them.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{write_file, DataError};

/// Where the loop ledger lives, relative to a dataset root.
pub const LOOP_FILE: &str = "loop.jsonl";

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// --- the loop ledger (spec 13.3) ------------------------------------------------------------

/// Which step of spec 13.1's loop a [`LoopStep`] records.
///
/// [`Train`](Self::Train) and [`Evaluate`](Self::Evaluate) are `es loop cycle`'s (packet
/// M7/T2): the ledger stopped at `distill` and so never recorded that a dataset trained a
/// policy or that a policy was judged. Adding variants leaves every line an older `es` wrote
/// readable — the tag is the only thing serde matches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopKind {
    Collect,
    Intervene,
    Distill,
    Train,
    Evaluate,
}

/// The key prefix a [`LoopKind::Train`] step's checkpoint outputs carry: `checkpoint.<step>`
/// -> the spec 19.3 `policy_hash` of that mark. One key per checkpoint rather than one packed
/// value, because membership is what [`check_chain`] asks of it.
pub const CHECKPOINT: &str = "checkpoint.";

/// One line of `loop.jsonl`: what a loop step consumed and what it produced (spec 13.3).
///
/// `created` is the only non-reproducible value in the file, for the same reason
/// `evaluation.lock`'s is (spec 10.4): provenance is not identity. Nothing here feeds a hash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopStep {
    pub kind: LoopKind,
    pub inputs: BTreeMap<String, String>,
    pub outputs: BTreeMap<String, String>,
    pub created: u64,
}

impl LoopStep {
    pub fn new(kind: LoopKind) -> Self {
        Self {
            kind,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            created: now_unix(),
        }
    }

    #[must_use]
    pub fn input(mut self, key: &str, value: &impl ToString) -> Self {
        self.inputs.insert(key.to_owned(), value.to_string());
        self
    }

    #[must_use]
    pub fn output(mut self, key: &str, value: &impl ToString) -> Self {
        self.outputs.insert(key.to_owned(), value.to_string());
        self
    }
}

/// Appends one step to `<root>/loop.jsonl`. Append-only: the ledger is never rewritten, so a
/// chain that does not link up stays visible (spec 13.3).
pub fn append_loop_step(root: &Path, step: &LoopStep) -> Result<(), DataError> {
    let path = root.join(LOOP_FILE);
    let mut text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(DataError::io(&path, e)),
    };
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(
        &serde_json::to_string(step).map_err(|source| DataError::Json {
            path: path.clone(),
            source,
        })?,
    );
    text.push('\n');
    write_file(&path, text.as_bytes())
}

/// Reads `<root>/loop.jsonl`; an absent file is an empty ledger.
pub fn read_loop_steps(root: &Path) -> Result<Vec<LoopStep>, DataError> {
    let path = root.join(LOOP_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(DataError::io(&path, e)),
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).map_err(|source| DataError::Json {
                path: path.clone(),
                source,
            })
        })
        .collect()
}

/// Spec 13.3's chain, checked over one ledger: a `train` consumed the dataset the `collect`
/// before it produced, and an `evaluate` judged a checkpoint the `train` before it wrote.
///
/// Both halves are conditional on there being an earlier step to chain to — a ledger that
/// only holds an `evaluate` (a bare `es eval run` against a bundle from elsewhere) claims no
/// chain and breaks none. The expert gate of §28.9 rule 1 is an `evaluate` step carrying an
/// `expert` input: it judges the scripted demonstrator rather than a checkpoint, so it is
/// skipped by name rather than by position.
pub fn check_chain(steps: &[LoopStep]) -> Result<(), DataError> {
    let mut content: Option<&str> = None;
    let mut checkpoints: Vec<(&str, &str)> = Vec::new();
    for step in steps {
        match step.kind {
            LoopKind::Collect | LoopKind::Intervene | LoopKind::Distill => {
                if let Some(c) = step.outputs.get("content") {
                    content = Some(c);
                }
            }
            LoopKind::Train => {
                let read = step.inputs.get("content").map_or("-", String::as_str);
                if let Some(want) = content {
                    if read != want {
                        return Err(refuse(format!(
                            "the ledger does not chain (spec 13.3): the train step reads dataset \
                             content {read} and the collect step before it wrote {want}"
                        )));
                    }
                }
                for (key, hash) in &step.outputs {
                    if let Some(mark) = key.strip_prefix(CHECKPOINT) {
                        checkpoints.push((mark, hash));
                    }
                }
            }
            LoopKind::Evaluate => {
                if step.inputs.contains_key("expert") {
                    continue;
                }
                let judged = step.inputs.get("policy_hash").map_or("-", String::as_str);
                if !checkpoints.is_empty() && !checkpoints.iter().any(|(_, h)| *h == judged) {
                    let known: Vec<String> = checkpoints
                        .iter()
                        .map(|(m, h)| format!("{m}={h}"))
                        .collect();
                    return Err(refuse(format!(
                        "the ledger does not chain (spec 13.3): the evaluate step judged \
                         policy_hash {judged}, which is not a checkpoint of any train step \
                         before it ({})",
                        known.join(" ")
                    )));
                }
            }
        }
    }
    Ok(())
}

/// The `evaluation_hash` of the last `evaluate` step in a ledger, if there is one.
///
/// Spec 13.3's discipline — hold the evaluation conditions fixed while data and policy move —
/// is a comparison between this and the next iteration's; `es loop cycle` refuses rather than
/// warns when the two differ.
pub fn last_evaluation_hash(steps: &[LoopStep]) -> Option<&str> {
    steps
        .iter()
        .rev()
        .filter(|s| s.kind == LoopKind::Evaluate)
        .find_map(|s| s.inputs.get("evaluation_hash").map(String::as_str))
}

fn refuse(msg: impl Into<String>) -> DataError {
    DataError::Loop(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ledger_is_append_only() {
        let dir = std::env::temp_dir().join(format!("es-data-loop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        append_loop_step(
            &dir,
            &LoopStep::new(LoopKind::Collect).output("content", &"aa"),
        )
        .unwrap();
        append_loop_step(
            &dir,
            &LoopStep::new(LoopKind::Intervene).input("content", &"aa"),
        )
        .unwrap();
        let steps = read_loop_steps(&dir).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].outputs["content"], steps[1].inputs["content"]);
    }

    /// Packet M7/T2 oracle 2: `collect -> train -> evaluate` chains, and each half fails by
    /// name when it does not (spec 13.3).
    #[test]
    fn loop_train_and_evaluate_steps_chain() {
        let collect = LoopStep::new(LoopKind::Collect)
            .output("content", &"aa")
            .output("schema", &"bb");
        let train = LoopStep::new(LoopKind::Train)
            .input("content", &"aa")
            .input("identity_hash", &"11")
            .output("training_hash", &"22")
            .output("checkpoint.20000", &"cc");
        let evaluate = LoopStep::new(LoopKind::Evaluate)
            .input("policy_hash", &"cc")
            .input("evaluation_hash", &"ee")
            .output("passed", &true);
        let chain = [collect.clone(), train.clone(), evaluate.clone()];
        check_chain(&chain).expect("collect -> train -> evaluate chains");
        assert_eq!(last_evaluation_hash(&chain), Some("ee"));

        // A train step that read another directory's dataset.
        let moved = LoopStep::new(LoopKind::Train).input("content", &"ff");
        let e = check_chain(&[collect.clone(), moved]).expect_err("refused");
        assert!(
            e.to_string().contains("ff") && e.to_string().contains("aa"),
            "{e}"
        );

        // An evaluate step judging a policy no train step in this ledger wrote.
        let stranger = LoopStep::new(LoopKind::Evaluate).input("policy_hash", &"dd");
        let e = check_chain(&[collect.clone(), train.clone(), stranger]).expect_err("refused");
        assert!(
            e.to_string().contains("dd") && e.to_string().contains("20000=cc"),
            "{e}"
        );

        // The expert gate judges the demonstrator, not a checkpoint, and chains to nothing.
        let gate = LoopStep::new(LoopKind::Evaluate)
            .input("expert", &"so101-pick-place")
            .input("policy_hash", &"00")
            .input("evaluation_hash", &"ee");
        check_chain(&[collect, gate, train, evaluate]).expect("the gate is skipped by name");
    }
}
