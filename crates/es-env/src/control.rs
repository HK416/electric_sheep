//! IR-C execution: the per-env control-graph state machine (spec 6.2, spec 6.4).
//!
//! `es-ir` owns the IR-C schema; this is its CPU reference executor, the same split
//! [`crate::plan::ScalarPlan`] has for reward cones. One [`ControlExecutor::step`] per env per
//! control step, after physics and before reward.
//!
//! Deterministic by construction (spec 6.6): integer stage time, no RNG, no wall clock, and the
//! only data-dependent scheduling is `Branch::condition` and `RepeatUntil::Until` — both
//! evaluated with `Expr::eval`, which is already specified as deterministic. See
//! `docs/design/control-graph.md`.

use std::collections::{BTreeMap, BTreeSet};

use es_ir::control::{ControlGraph, ControlNode, RepeatUntil};
use es_ir::graph::NodeId;
use es_ir::task::TaskIr;

/// Descent steps one [`ControlExecutor::step`] may take before giving up. A tree deep or wide
/// enough to exhaust this is a generated graph with a bug, not an authored task; the env is
/// ended rather than left spinning.
const MAX_DESCENT: u32 = 256;

/// One level of the path from the root down to the active `SubTask`: which node, and which of
/// its children the path went through (the `Sequence` position, the `Repeat` iteration, or
/// 0 = `then` / 1 = `else` for a `Branch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Frame {
    node: NodeId,
    index: u32,
}

/// One env's position in the control graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageState {
    stack: Vec<Frame>,
    /// Steps spent in the active `SubTask`.
    ticks: u32,
    /// The stage that scored this step, so the reward sum can be restricted to it.
    scored: Option<NodeId>,
    started: bool,
    done: bool,
    failed: bool,
}

/// What one control step did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageOutcome {
    /// The active stage's weight in the episode reward sum; 0 once the graph is finished.
    pub reward_weight: f64,
    /// The control graph ran out of stages, or a `SubTask` timed out.
    pub done: bool,
    /// `done` because of a `SubTask` timeout (spec 6.2): the episode is a failure.
    pub failed: bool,
    /// The active stage changed on this step.
    pub transitioned: bool,
    /// A stage entered on this step declares `reset_on_entry`. Wiring it into `Env` is
    /// deferred (`docs/design/control-graph.md` §3.4); the flag is reported, never ignored.
    pub reset_requested: bool,
}

/// Advances one control graph, one state per env.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlExecutor {
    graph: ControlGraph,
    stage: Vec<StageState>,
}

impl ControlExecutor {
    /// `None` for an IR-D-only task, which is what keeps [`crate::env::Env`] unchanged when no
    /// control graph is authored.
    pub fn new(task: &TaskIr, n_envs: u32) -> Option<Self> {
        task.control.as_ref().map(|graph| Self {
            graph: graph.clone(),
            stage: vec![StageState::default(); n_envs as usize],
        })
    }

    /// Puts `env` back at the root. Called from `Env::reset`, so an episode boundary is a
    /// control-graph boundary.
    pub fn reset_env(&mut self, env: u32) {
        self.stage[env as usize] = StageState::default();
    }

    /// The stage that scored this step: its reward-term names and its weight.
    pub fn scored_stage(&self, env: u32) -> Option<(&BTreeSet<String>, f64)> {
        let node = self.stage[env as usize].scored?;
        match self.graph.nodes.get(&node)? {
            ControlNode::SubTask { task, .. } => Some((&task.rewards, task.weight)),
            _ => None,
        }
    }

    /// The active stage's name, for a test or a telemetry label.
    pub fn stage_name(&self, env: u32) -> Option<&str> {
        let frame = self.stage[env as usize].stack.last()?;
        match self.graph.nodes.get(&frame.node)? {
            ControlNode::SubTask { task, .. } => Some(task.name.as_str()),
            _ => None,
        }
    }

    /// Advances `env` by one control step, reading and writing `ports` — the same map the
    /// reward and termination cones use, extended with `stage.index` / `stage.ticks` /
    /// `stage.done`.
    pub fn step(&mut self, env: u32, ports: &mut BTreeMap<String, f64>) -> StageOutcome {
        let i = env as usize;
        if self.stage[i].done {
            return StageOutcome {
                done: true,
                failed: self.stage[i].failed,
                ..StageOutcome::default()
            };
        }

        // Entry happens on the first step of the episode, not at reset: a `Branch` must see the
        // bound ports, and at reset there are none.
        let mut reset_requested = false;
        if !self.stage[i].started {
            self.stage[i].started = true;
            reset_requested = self.run(i, ports, Some(self.graph.root));
        }

        let Some(frame) = self.stage[i].stack.last().copied() else {
            self.stage[i].done = true;
            self.stage[i].scored = None;
            return StageOutcome {
                done: true,
                reset_requested,
                ..StageOutcome::default()
            };
        };
        self.stage[i].ticks += 1;
        self.stage[i].scored = Some(frame.node);
        self.write_ports(i, ports, 0.0);

        let (weight, timeout, complete) = {
            let Some(ControlNode::SubTask {
                task,
                timeout_ticks,
            }) = self.graph.nodes.get(&frame.node)
            else {
                // `ControlGraph::validate` rules this out; ending the env beats spinning.
                self.stage[i].done = true;
                self.stage[i].failed = true;
                return StageOutcome {
                    done: true,
                    failed: true,
                    reset_requested,
                    ..StageOutcome::default()
                };
            };
            let complete = match &task.success {
                None => true,
                Some(e) => e.eval(ports).is_some_and(|v| v != 0.0),
            };
            (task.weight, *timeout_ticks, complete)
        };

        if self.stage[i].ticks >= timeout {
            self.stage[i].done = true;
            self.stage[i].failed = true;
            return StageOutcome {
                reward_weight: weight,
                done: true,
                failed: true,
                reset_requested,
                ..StageOutcome::default()
            };
        }
        if !complete {
            return StageOutcome {
                reward_weight: weight,
                reset_requested,
                ..StageOutcome::default()
            };
        }

        self.write_ports(i, ports, 1.0);
        reset_requested |= self.run(i, ports, None);
        StageOutcome {
            reward_weight: weight,
            done: self.stage[i].done,
            failed: self.stage[i].failed,
            transitioned: true,
            reset_requested,
        }
    }

    /// `stage.index` / `stage.ticks` / `stage.done` (`docs/design/control-graph.md` §3.1).
    fn write_ports(&self, i: usize, ports: &mut BTreeMap<String, f64>, done: f64) {
        let stack = &self.stage[i].stack;
        let index = stack
            .len()
            .checked_sub(2)
            .and_then(|p| stack.get(p))
            .filter(|f| {
                matches!(
                    self.graph.nodes.get(&f.node),
                    Some(ControlNode::Sequence { .. })
                )
            })
            .map_or(0, |f| f.index);
        ports.insert("stage.index".to_owned(), f64::from(index));
        ports.insert("stage.ticks".to_owned(), f64::from(self.stage[i].ticks));
        ports.insert("stage.done".to_owned(), done);
    }

    fn push(&mut self, i: usize, node: NodeId, index: u32) {
        self.stage[i].stack.push(Frame { node, index });
    }

    /// Runs the machine until the top frame is a `SubTask` or the stack empties.
    ///
    /// `pending = Some(n)` descends into `n`; `pending = None` pops the finished frame and asks
    /// its parent for the next child. Returns whether the stage it stopped on wants a reset
    /// draw — at most one stage is entered per call, since reaching one ends the walk.
    fn run(
        &mut self,
        i: usize,
        ports: &BTreeMap<String, f64>,
        mut pending: Option<NodeId>,
    ) -> bool {
        for _ in 0..MAX_DESCENT {
            if let Some(node) = pending.take() {
                match self.graph.nodes.get(&node) {
                    Some(ControlNode::Sequence { children }) => {
                        let first = children.first().copied();
                        self.push(i, node, 0);
                        if first.is_some() {
                            pending = first;
                            continue;
                        }
                        // An empty Sequence has nothing to enter; fall through and pop it.
                    }
                    Some(ControlNode::Branch {
                        condition,
                        then_,
                        else_,
                    }) => {
                        // Evaluated once, here. The arm lives in the frame from now on.
                        let take_then = condition.eval(ports).is_some_and(|v| v != 0.0);
                        let next = if take_then { *then_ } else { *else_ };
                        self.push(i, node, u32::from(!take_then));
                        pending = Some(next);
                        continue;
                    }
                    Some(ControlNode::Repeat { body, .. }) => {
                        let body = *body;
                        self.push(i, node, 0);
                        pending = Some(body);
                        continue;
                    }
                    Some(ControlNode::SubTask { task, .. }) => {
                        let wants_reset = task.reset_on_entry;
                        self.push(i, node, 0);
                        self.stage[i].ticks = 0;
                        return wants_reset;
                    }
                    // A dangling reference (`CTRL-001`): pop and carry on.
                    None => {}
                }
            }

            let Some(frame) = self.stage[i].stack.pop() else {
                self.stage[i].done = true;
                return false;
            };
            match self.graph.nodes.get(&frame.node) {
                Some(ControlNode::Sequence { children }) => {
                    let next = frame.index as usize + 1;
                    if let Some(child) = children.get(next).copied() {
                        self.push(i, frame.node, frame.index + 1);
                        pending = Some(child);
                    }
                }
                Some(ControlNode::Repeat { body, until }) => {
                    let again = match until {
                        RepeatUntil::Count(n) => frame.index + 1 < *n,
                        RepeatUntil::Until(e) => !e.eval(ports).is_some_and(|v| v != 0.0),
                    };
                    if again {
                        let body = *body;
                        self.push(i, frame.node, frame.index + 1);
                        pending = Some(body);
                    }
                }
                // A Branch runs one arm; a SubTask is a leaf. Both are finished when popped.
                _ => {}
            }
        }
        self.stage[i].done = true;
        self.stage[i].failed = true;
        false
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::env::tests::{fake_scene, task_with, FakeBackend};
    use crate::scheduler::{BatchDomains, DomainCfg};
    use crate::{Env, Termination};
    use es_ir::control::SubTaskRef;
    use es_ir::task::{Aggregation, CmpOp, Expr, JointQuantity, TaskNode, TerminationKind};
    use es_ir::types::{ElemType, Frame as IrFrame, PortType, Shape, TimeRef, Unit};

    fn gt(port: &str, rhs: f64) -> Expr {
        Expr::Compare {
            op: CmpOp::Gt,
            lhs: Box::new(Expr::Port(port.to_owned())),
            rhs: Box::new(Expr::Const(rhs)),
        }
    }

    fn scalar_ty() -> PortType {
        PortType {
            elem: ElemType::F32,
            shape: Shape::new([1]),
            unit: Unit::Angle,
            frame: IrFrame::World,
            time: TimeRef::Tick,
            image: None,
        }
    }

    /// A task with one `Reward` per name, each reading the hinge angle, so `qpos[0]` is bound
    /// and every stage has a term it can be restricted to.
    fn staged_task(names: &[&str]) -> TaskIr {
        let mut nodes = vec![TaskNode::GetJointState {
            body: fake_scene().bodies[0].id,
            joints: vec!["hinge".to_owned()],
            quantity: JointQuantity::Position,
        }];
        for name in names {
            nodes.push(TaskNode::Reward {
                name: (*name).to_owned(),
                weight: 1.0,
                aggregation: Aggregation::Sum,
                ty: scalar_ty(),
            });
        }
        let mut task = task_with(&nodes);
        for i in 1..=names.len() as u32 {
            task.graph.connect(NodeId(0), "value", NodeId(i), "value");
        }
        task
    }

    fn stage(name: &str, rewards: &[&str], ticks: f64, timeout: u32) -> ControlNode {
        ControlNode::SubTask {
            task: SubTaskRef {
                name: name.to_owned(),
                rewards: rewards.iter().map(|s| (*s).to_owned()).collect(),
                observation: BTreeSet::new(),
                success: Some(gt("stage.ticks", ticks)),
                reset_on_entry: false,
                weight: 1.0,
            },
            timeout_ticks: timeout,
        }
    }

    fn env_of(task: &TaskIr) -> Env<FakeBackend> {
        let domains = BatchDomains {
            simulation: DomainCfg::new(1, 1),
            observation: DomainCfg::new(1, 10),
            inference: DomainCfg::new(1, 20),
            training: None,
        };
        Env::new(task, &fake_scene(), FakeBackend::new(), &domains, 7).expect("the task compiles")
    }

    /// `pick -> move -> place`, each stage two steps long.
    fn pick_move_place() -> TaskIr {
        let mut task = staged_task(&["pick", "move", "place"]);
        task.control = Some(ControlGraph {
            root: NodeId(0),
            nodes: BTreeMap::from([
                (
                    NodeId(0),
                    ControlNode::Sequence {
                        children: vec![NodeId(1), NodeId(2), NodeId(3)],
                    },
                ),
                (NodeId(1), stage("pick", &["pick"], 1.0, 50)),
                (NodeId(2), stage("move", &["move"], 1.0, 50)),
                (NodeId(3), stage("place", &["place"], 1.0, 50)),
            ]),
        });
        task
    }

    fn run(env: &mut Env<FakeBackend>, steps: usize) -> (Vec<f64>, Vec<bool>) {
        let mut rewards = Vec::with_capacity(steps);
        let mut dones = Vec::with_capacity(steps);
        for _ in 0..steps {
            let out = env.step(&[0.05]).expect("step");
            rewards.push(out.rewards[0]);
            dones.push(out.dones[0]);
        }
        (rewards, dones)
    }

    #[test]
    fn three_stages_run_in_order() {
        let task = pick_move_place();
        let mut env = env_of(&task);
        let mut names = Vec::new();
        for _ in 0..6 {
            env.step(&[0.05]).expect("step");
            names.push(env.stage_name(0).map(str::to_owned));
        }
        // Two steps per stage (`stage.ticks > 1.0`), then the graph is finished.
        assert_eq!(
            names,
            [
                Some("pick".to_owned()),
                Some("move".to_owned()),
                Some("move".to_owned()),
                Some("place".to_owned()),
                Some("place".to_owned()),
                None,
            ]
        );
    }

    #[test]
    fn only_the_active_stages_reward_term_scores() {
        // Every term reads the same angle, so a whole-episode sum would be 3x the stage sum.
        let task = pick_move_place();
        let mut staged = env_of(&task);
        let mut plain = env_of(&staged_task(&["pick", "move", "place"]));
        let (staged_rewards, _) = run(&mut staged, 3);
        let (plain_rewards, _) = run(&mut plain, 3);
        for (s, p) in staged_rewards.iter().zip(&plain_rewards) {
            assert_eq!(*s * 3.0, *p, "one of three terms, at stage weight 1.0");
        }
    }

    #[test]
    fn finishing_every_stage_ends_the_episode() {
        let task = pick_move_place();
        let mut env = env_of(&task);
        let (_, dones) = run(&mut env, 7);
        assert_eq!(dones, [false, false, false, false, false, true, false]);
    }

    #[test]
    fn a_branch_takes_the_arm_the_port_selects() {
        let arms = |rhs: f64| {
            let mut task = staged_task(&["left", "right"]);
            task.control = Some(ControlGraph {
                root: NodeId(0),
                nodes: BTreeMap::from([
                    (
                        NodeId(0),
                        ControlNode::Branch {
                            // The hinge starts at 0 and the constant torque drives it positive.
                            condition: gt("qpos[0]", rhs),
                            then_: NodeId(1),
                            else_: NodeId(2),
                        },
                    ),
                    (NodeId(1), stage("left", &["left"], 20.0, 50)),
                    (NodeId(2), stage("right", &["right"], 20.0, 50)),
                ]),
            });
            task
        };
        let taken = |rhs: f64| {
            let task = arms(rhs);
            let mut env = env_of(&task);
            env.step(&[0.05]).expect("step");
            env.stage_name(0).map(str::to_owned)
        };
        assert_eq!(taken(-1.0), Some("left".to_owned()));
        assert_eq!(taken(1.0), Some("right".to_owned()));
    }

    #[test]
    fn repeat_count_executes_the_body_three_times() {
        let mut task = staged_task(&["body"]);
        task.control = Some(ControlGraph {
            root: NodeId(0),
            nodes: BTreeMap::from([
                (
                    NodeId(0),
                    ControlNode::Repeat {
                        body: NodeId(1),
                        until: RepeatUntil::Count(3),
                    },
                ),
                // One step per iteration: `stage.ticks > 0`.
                (NodeId(1), stage("body", &["body"], 0.0, 50)),
            ]),
        });
        let mut env = env_of(&task);
        let (_, dones) = run(&mut env, 4);
        assert_eq!(dones, [false, false, true, false], "three iterations");
    }

    #[test]
    fn a_subtask_timeout_ends_the_episode_with_the_failure_flag() {
        let mut task = staged_task(&["stuck"]);
        task.control = Some(ControlGraph {
            root: NodeId(0),
            // `stage.ticks > 100` never holds before the timeout at 3.
            nodes: BTreeMap::from([(NodeId(0), stage("stuck", &["stuck"], 100.0, 3))]),
        });
        let mut env = env_of(&task);
        let out = env.step(&[0.05]).expect("step");
        assert!(!out.dones[0]);
        env.step(&[0.05]).expect("step");
        let out = env.step(&[0.05]).expect("step");
        assert!(out.dones[0], "the timeout elapsed");
        assert_eq!(out.episodes.len(), 1);
        assert_eq!(out.episodes[0].termination, Termination::Failure);
        assert!(*out.episodes[0].done.last().expect("a step"));
    }

    /// A task-level `Terminate` is a whole-episode predicate and a stage may not suppress it.
    #[test]
    fn a_task_terminate_still_wins() {
        let mut task = pick_move_place();
        let n = NodeId(9);
        task.graph.insert(
            n,
            TaskNode::Compare {
                op: CmpOp::Gt,
                rhs: Some(-1.0),
                ty: scalar_ty(),
            },
        );
        task.graph.insert(
            NodeId(10),
            TaskNode::Terminate {
                kind: TerminationKind::Success,
            },
        );
        task.graph.connect(NodeId(0), "value", n, "a");
        task.graph.connect(n, "value", NodeId(10), "value");
        let mut env = env_of(&task);
        let out = env.step(&[0.05]).expect("step");
        assert!(out.dones[0], "qpos[0] > -1.0 holds on the first step");
    }

    #[test]
    fn two_runs_are_bitwise_identical() {
        let task = pick_move_place();
        let trace = || {
            let mut env = env_of(&task);
            let mut out = Vec::new();
            for _ in 0..12 {
                let step = env.step(&[0.05]).expect("step");
                out.push((
                    step.rewards[0].to_bits(),
                    step.dones[0],
                    env.stage_name(0).map(str::to_owned),
                ));
            }
            out
        };
        assert_eq!(trace(), trace());
    }
}
