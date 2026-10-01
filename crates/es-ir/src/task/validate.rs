//! [`TaskIr::validate`]: every check spec 6 mandates.

use std::collections::BTreeSet;

use super::{ObsSource, SensorPath, TaskIr, TaskNode, SCHEMA_VERSION};
use crate::codes;
use crate::diag::Diagnostic;
use crate::graph::NodeId;

impl TaskIr {
    /// Every check spec 6 mandates, in one pass: port and edge typing, acyclicity, the
    /// determinism rules of spec 6.6, and the spec 7.4 declaration link.
    ///
    /// "No neural nets in Task IR" needs no check — [`TaskNode`] has no such variant.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut diags = self.graph.validate_declared_ports();
        if let Err(d) = self.graph.topo_order() {
            diags.push(d);
        }
        // A file from a newer writer carries nodes and fields this build cannot see, and a `0`
        // is an unwritten field; either way `task_hash` would pin something never validated.
        // Older supported versions stay readable -- the version is hashed, so their hashes
        // survive (`docs/design/ir-types.md`).
        if self.schema_version == 0 || self.schema_version > SCHEMA_VERSION {
            diags.push(Diagnostic::new(
                codes::TASK_002,
                format!(
                    "schema_version {} is not supported (expected 1..={SCHEMA_VERSION})",
                    self.schema_version
                ),
            ));
        }

        let stream_check = |diags: &mut Vec<Diagnostic>, id: NodeId, stream: &String| {
            if stream.is_empty() {
                diags.push(
                    Diagnostic::new(codes::DET_001, "the RNG stream name is empty")
                        .at(id)
                        .with_hint("every draw names a TaskRng(seed, EnvId, PhysTick, stream)"),
                );
            } else if !self.config.rng_streams.contains(stream) {
                diags.push(
                    Diagnostic::new(
                        codes::DET_021,
                        format!("stream \"{stream}\" is not declared in the task config"),
                    )
                    .at(id),
                );
            }
        };

        for (id, node) in &self.graph.nodes {
            match node {
                TaskNode::GetRandom { stream, .. }
                | TaskNode::Randomization { stream, .. }
                | TaskNode::ResetState { stream, .. } => stream_check(&mut diags, *id, stream),
                TaskNode::MathFn { func, approx, .. } if func.is_transcendental() && !approx => {
                    diags.push(
                        Diagnostic::new(
                            codes::DET_010,
                            format!("{func:?} uses the standard library implementation"),
                        )
                        .at(*id)
                        .with_hint("set approx = true (es-math::approx)"),
                    );
                }
                TaskNode::Reduce { unordered, .. } if *unordered && self.config.deterministic => {
                    diags.push(
                        Diagnostic::new(codes::DET_030, "unordered Reduce in deterministic mode")
                            .at(*id),
                    );
                }
                TaskNode::Terminate {
                    hold_ticks: Some(0),
                    ..
                } => diags.push(
                    Diagnostic::new(codes::TASK_004, "Terminate holds for 0 control ticks")
                        .at(*id)
                        .with_hint("hold for at least 1 tick, or leave hold_ticks out"),
                ),
                TaskNode::Reward { ty, name, .. } => {
                    if let Err(d) = ty.check_policy_input() {
                        diags.push(d.at(*id).with_hint(format!(
                            "reward term \"{name}\" must be dimensionless; normalize it first"
                        )));
                    }
                }
                TaskNode::ObservationSpec { channel, ty } => {
                    match self.observation_spec.channels.get(channel) {
                        Some(ch) if ch.ty == *ty => {}
                        Some(_) => diags.push(
                            Diagnostic::new(
                                codes::TASK_001,
                                format!("channel \"{channel}\" is declared with a different type"),
                            )
                            .at(*id),
                        ),
                        None => diags.push(
                            Diagnostic::new(
                                codes::TASK_001,
                                format!("channel \"{channel}\" is not in the ObservationSpec"),
                            )
                            .at(*id),
                        ),
                    }
                }
                _ => {}
            }
        }

        let bound: BTreeSet<&str> = self
            .graph
            .nodes
            .values()
            .filter_map(|n| match n {
                TaskNode::ObservationSpec { channel, .. } => Some(channel.as_str()),
                _ => None,
            })
            .collect();
        for (name, ch) in &self.observation_spec.channels {
            // Packet M11/X6: SVGF filters the path tracer's noise; the rasterizer has none.
            if let ObsSource::Sensor { render, .. } = &ch.source {
                if render.svgf && render.path == SensorPath::Rs {
                    diags.push(
                        Diagnostic::new(
                            codes::TASK_003,
                            format!("channel \"{name}\" declares svgf on the rasterizer"),
                        )
                        .with_hint("use path = \"pt\""),
                    );
                }
            }
            // A `PreviousAction` channel is served by the control loop, not computed by the
            // graph, so no ObservationSpec node binds it (packet M11/I3).
            let served = matches!(ch.source, ObsSource::PreviousAction { .. });
            if !served && !bound.contains(name.as_str()) {
                diags.push(Diagnostic::new(
                    codes::TASK_001,
                    format!("declared channel \"{name}\" has no ObservationSpec node"),
                ));
            }
        }

        if let Some(control) = &self.control {
            diags.extend(control.validate(self));
        }

        if let Err(d) = self.task_hash() {
            diags.push(d);
        }
        diags
    }
}
