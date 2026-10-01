//! [`LearningGraph::validate`]: every check of spec 8.4 the Learning IR answers on its own.

use super::{feature, FusionKind, HeadKind, LearningGraph, LearningNode, NormalizeDir};
use crate::codes;
use crate::diag::Diagnostic;
use crate::graph::IrNode;

impl LearningGraph {
    /// Every check of spec 8.4 that this IR can answer on its own. Cross-IR checks (dataset
    /// shapes, `dt_ctrl`, the Safety Plane fallback) belong to P26.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut diags = self.nodes.validate_declared_ports();
        if let Err(d) = self.nodes.topo_order() {
            diags.push(d);
        }
        if self.schema_version != self.nodes.schema_version {
            diags.push(Diagnostic::new(
                codes::LRN_001,
                format!(
                    "wrapper says schema {} but the graph says {}",
                    self.schema_version, self.nodes.schema_version
                ),
            ));
        }
        self.check_arity(&mut diags);
        self.check_boundary(&mut diags);
        self.check_contract(&mut diags);
        self.check_normalizers(&mut diags);
        self.check_squash(&mut diags);
        self.check_sum(&mut diags);
        self.check_share(&mut diags);
        diags
    }

    /// Spec 8.3: a `Sum` adds its terms element by element, so every input has the node's own
    /// output shape — one width, one token count, and `out_dim` equal to that width.
    fn check_sum(&self, diags: &mut Vec<Diagnostic>) {
        for (id, node) in &self.nodes.nodes {
            let LearningNode::Fusion {
                inputs,
                kind: FusionKind::Sum,
                out_dim,
                token_count,
            } = node
            else {
                continue;
            };
            let out = feature("out", *out_dim, *token_count).ty.shape;
            for p in inputs.iter().filter(|p| p.ty.shape != out) {
                diags.push(
                    Diagnostic::new(
                        codes::LRN_032,
                        format!(
                            "Sum input \"{}\" is {:?} but the fusion's output is {:?} (out_dim {out_dim}, token_count {token_count})",
                            p.name,
                            p.ty.shape.dims(),
                            out.dims()
                        ),
                    )
                    .at(*id)
                    .on_port(p.name.clone())
                    .with_hint("every term of a Sum has the same width and token count, and out_dim is that width"),
                );
            }
        }
    }

    /// Spec 8.3 `share`: it names the owner of a weight group. Looked up by id, one sharer at a
    /// time, so neither node order nor where the owner sits in the document matters.
    fn check_share(&self, diags: &mut Vec<Diagnostic>) {
        for (id, node) in &self.nodes.nodes {
            let LearningNode::VisionEncoder {
                share: Some(owner),
                backbone,
                pretrained,
                frozen,
                out_dim,
                token_count,
                ..
            } = node
            else {
                continue;
            };
            let refuse = |why: String| {
                Diagnostic::new(codes::LRN_033, format!("share = {}: {why}", owner.0))
                    .at(*id)
                    .with_hint("share names a VisionEncoder of the same backbone, out_dim, token_count, pretrained and frozen that shares nothing itself")
            };
            if owner == id {
                diags.push(refuse("an encoder cannot share its own weights".to_owned()));
                continue;
            }
            let Some(other) = self.nodes.nodes.get(owner) else {
                diags.push(refuse(format!("node {} does not exist", owner.0)));
                continue;
            };
            let LearningNode::VisionEncoder {
                share: next,
                backbone: their_backbone,
                pretrained: their_pretrained,
                frozen: their_frozen,
                out_dim: their_out_dim,
                token_count: their_token_count,
                ..
            } = other
            else {
                diags.push(refuse(format!(
                    "node {} is a {}, not a VisionEncoder",
                    owner.0,
                    other.kind()
                )));
                continue;
            };
            if let Some(next) = next {
                diags.push(refuse(format!(
                    "node {} itself shares node {}; sharing does not chain, name the group's owner",
                    owner.0, next.0
                )));
            }
            for (field, mine, theirs) in [
                (
                    "backbone",
                    format!("{backbone:?}"),
                    format!("{their_backbone:?}"),
                ),
                ("out_dim", out_dim.to_string(), their_out_dim.to_string()),
                (
                    "token_count",
                    token_count.to_string(),
                    their_token_count.to_string(),
                ),
                (
                    "pretrained",
                    pretrained.to_string(),
                    their_pretrained.to_string(),
                ),
                ("frozen", frozen.to_string(), their_frozen.to_string()),
            ] {
                if mine != theirs {
                    diags.push(refuse(format!(
                        "{field} is {mine} here but {theirs} on node {}",
                        owner.0
                    )));
                }
            }
        }
    }

    /// Spec 8.3: only a `Regression` head produces the point estimate a squash is defined on.
    /// A sampler head's output comes out of a denoiser or a codebook, where `tanh` would
    /// change the distribution rather than the range (packet M8/S2a).
    fn check_squash(&self, diags: &mut Vec<Diagnostic>) {
        for (id, node) in &self.nodes.nodes {
            let LearningNode::PolicyHead { kind, squash, .. } = node else {
                continue;
            };
            if squash.is_default() || matches!(kind, HeadKind::Regression) {
                continue;
            }
            // The variant name alone: a `Diffusion` head's `Debug` is seven scheduler fields.
            let debug = format!("{kind:?}");
            let head = debug.split(' ').next().unwrap_or(&debug);
            diags.push(
                Diagnostic::new(
                    codes::LRN_031,
                    format!("{squash:?} squash on a {head} head"),
                )
                .at(*id)
                .with_hint("squash is defined on Regression only; leave it None"),
            );
        }
    }

    fn check_arity(&self, diags: &mut Vec<Diagnostic>) {
        for (id, node) in &self.nodes.nodes {
            let (min, max) = node.arity();
            let n = node.input_ports().len();
            if n < min || max.is_some_and(|m| n > m) {
                diags.push(
                    Diagnostic::new(
                        codes::LRN_002,
                        format!("{} declares {n} input ports, expected {min}..", node.kind()),
                    )
                    .at(*id),
                );
            }
        }
    }

    /// The declared tensor ports, the graph boundary and `PolicyContract::inputs` must agree.
    /// Boundary inputs feed the network, so spec 5.4's policy-input rule applies to each.
    fn check_boundary(&self, diags: &mut Vec<Diagnostic>) {
        for (declared, refs, side) in [
            (&self.inputs, &self.nodes.inputs, "input"),
            (&self.outputs, &self.nodes.outputs, "output"),
        ] {
            if declared.len() != refs.len() {
                diags.push(Diagnostic::new(
                    codes::LRN_010,
                    format!(
                        "{} declared {side} ports but the graph boundary has {}",
                        declared.len(),
                        refs.len()
                    ),
                ));
                continue;
            }
            for (port, at) in declared.iter().zip(refs) {
                // A missing node is already reported as GRAPH-002.
                let Some(node) = self.nodes.nodes.get(&at.node) else {
                    continue;
                };
                let ports = if side == "input" {
                    node.inputs()
                } else {
                    node.outputs()
                };
                let Some(found) = ports.into_iter().find(|p| p.name == at.port) else {
                    continue; // already reported as GRAPH-010
                };
                if let Err(d) = port.ty.compatible(&found.ty) {
                    diags.push(
                        Diagnostic::new(
                            codes::LRN_010,
                            format!(
                                "{side} \"{}\" does not match the node: {}",
                                port.name, d.message
                            ),
                        )
                        .at(at.node)
                        .on_port(at.port.clone()),
                    );
                }
            }
        }

        for port in &self.inputs {
            if let Err(d) = port.ty.check_policy_input() {
                diags.push(d.on_port(port.name.clone()));
            }
            match self.policy.contract.inputs.get(&port.name) {
                None => diags.push(
                    Diagnostic::new(
                        codes::LRN_011,
                        format!("PolicyContract has no input named \"{}\"", port.name),
                    )
                    .on_port(port.name.clone()),
                ),
                Some(declared) => {
                    if let Err(d) = port.ty.compatible(&declared.ty) {
                        diags.push(
                            Diagnostic::new(
                                codes::LRN_011,
                                format!("contract input \"{}\": {}", port.name, d.message),
                            )
                            .on_port(port.name.clone()),
                        );
                    }
                }
            }
        }
    }

    fn check_contract(&self, diags: &mut Vec<Diagnostic>) {
        let c = &self.policy.contract;
        for (name, v) in [
            ("action_dim", c.action_dim),
            ("horizon", c.horizon),
            ("execute_chunk", c.execute_chunk),
            ("observation_window", c.observation_window),
        ] {
            if v == 0 {
                diags.push(Diagnostic::new(
                    codes::LRN_023,
                    format!("{name} must be positive"),
                ));
            }
        }
        if c.replanning_hz <= 0.0 || !c.replanning_hz.is_finite() {
            diags.push(Diagnostic::new(
                codes::LRN_023,
                format!(
                    "replanning_hz must be positive and finite, got {}",
                    c.replanning_hz
                ),
            ));
        }
        if c.execute_chunk > c.horizon {
            diags.push(
                Diagnostic::new(
                    codes::LRN_020,
                    format!(
                        "execute_chunk {} exceeds horizon {}",
                        c.execute_chunk, c.horizon
                    ),
                )
                .with_hint("K must be at most H (spec 8.5)"),
            );
        }

        let head = self.nodes.nodes.values().find_map(|n| match n {
            LearningNode::PolicyHead {
                action_dim,
                horizon,
                ..
            }
            | LearningNode::PolicyBundle {
                action_dim,
                horizon,
                ..
            } => Some((*action_dim, *horizon)),
            _ => None,
        });
        match head {
            None => diags.push(
                Diagnostic::new(codes::LRN_021, "the graph has no policy head")
                    .with_hint("add a PolicyHead node, or reference a whole VLA with PolicyBundle"),
            ),
            Some((action_dim, horizon)) => {
                if action_dim != c.action_dim {
                    diags.push(Diagnostic::new(
                        codes::LRN_021,
                        format!(
                            "contract action_dim {} but the head produces {action_dim}",
                            c.action_dim
                        ),
                    ));
                }
                if horizon != c.horizon {
                    diags.push(Diagnostic::new(
                        codes::LRN_021,
                        format!(
                            "contract horizon {} but the head produces {horizon}",
                            c.horizon
                        ),
                    ));
                }
            }
        }

        // The chunker, when present, must agree with the contract it implements.
        for (id, node) in &self.nodes.nodes {
            if let LearningNode::ActionChunker {
                horizon,
                execute_chunk,
                ..
            } = node
            {
                if *execute_chunk > *horizon {
                    diags.push(
                        Diagnostic::new(
                            codes::LRN_020,
                            format!("execute_chunk {execute_chunk} exceeds horizon {horizon}"),
                        )
                        .at(*id),
                    );
                }
                if (*horizon, *execute_chunk) != (c.horizon, c.execute_chunk) {
                    diags.push(
                        Diagnostic::new(
                            codes::LRN_021,
                            format!(
                                "chunker is ({horizon}, {execute_chunk}) but the contract is ({}, {})",
                                c.horizon, c.execute_chunk
                            ),
                        )
                        .at(*id),
                    );
                }
            }
        }

        let window = self
            .nodes
            .nodes
            .values()
            .find_map(|n| match n {
                LearningNode::TemporalEncoder { n_frames, .. } => Some(*n_frames),
                _ => None,
            })
            .unwrap_or(1);
        if window != c.observation_window {
            diags.push(
                Diagnostic::new(
                    codes::LRN_022,
                    format!(
                        "observation_window {} but the temporal node sees {window} frames",
                        c.observation_window
                    ),
                )
                .with_hint("the TemporalEncoder is layer 3 of the spec 7.5 time model"),
            );
        }

        // Spec 8.4: inference must fit inside the deadline and inside the replanning period.
        let replan_ms = 1000.0 / c.replanning_hz;
        let budget = c.runtime.deadline_ms.min(replan_ms);
        if c.runtime.expected_latency_ms > budget {
            diags.push(
                Diagnostic::new(
                    codes::LRN_052,
                    format!(
                        "measured latency {:.1} ms, budget {budget:.1} ms (deadline {:.1} ms, replanning {:.1} Hz)",
                        c.runtime.expected_latency_ms, c.runtime.deadline_ms, c.replanning_hz
                    ),
                )
                .with_hint(
                    "async inference with chunk buffering (spec 8.6), a Safety Plane fallback (spec 9.4), or a faster policy",
                ),
            );
        }
    }

    fn check_normalizers(&self, diags: &mut Vec<Diagnostic>) {
        for (id, node) in &self.nodes.nodes {
            let LearningNode::Normalizer {
                direction,
                out_unit,
                ..
            } = node
            else {
                continue;
            };
            if (*direction == NormalizeDir::Forward) != out_unit.is_policy_input() {
                diags.push(
                    Diagnostic::new(
                        codes::LRN_030,
                        format!("{direction:?} normalizer produces {out_unit:?}"),
                    )
                    .at(*id)
                    .with_hint(
                        "Forward produces Normalized/Dimensionless/Token; Inverse leaves that space",
                    ),
                );
            }
        }
    }
}
