//! The `success` and `failure` sections and the timeout: each clause's predicate, the fold
//! into `Terminate`, the sparse and shaping terms, and which fields a relation takes.

use es_assets::scene::JointKind;
use es_ir::graph::NodeId;
use es_ir::task::{ArithOp, CmpOp, JointQuantity, LogicOp, MathFunc, TaskNode, TerminationKind};
// A number written into a document has musl's bits, not the host's (spec 5.3, M17 G3d).
use es_math::approx;

use super::Compiler;
use crate::spec::{refuse, vocab, Clause, Relation, Shaping, SpecError};

/// `InverseAngle`'s knots: `s = √(8 (1 − d))` at which `1 / (s + 0.1)` is sampled; the last is
/// `√8`, `d = 0` (plan H, `crates/es/tests/shadow_hand.rs`).
const KNOTS: [f64; 9] = [
    0.0,
    0.025,
    0.05,
    0.1,
    0.2,
    0.4,
    0.8,
    1.6,
    2.828_427_124_746_190_3,
];

impl Compiler<'_> {
    /// The success, failure and timeout terminations and the sparse terms.
    pub(super) fn ends(&mut self) -> Result<(), SpecError> {
        let spec = self.spec;
        let scale = self.scale();
        let reward = spec.reward.clone().unwrap_or_default();
        let groups = [
            (
                "success",
                Some(&spec.success),
                LogicOp::And,
                TerminationKind::Success,
                reward.success,
            ),
            (
                "failure",
                spec.failure.as_ref(),
                LogicOp::Or,
                TerminationKind::Failure,
                reward.failure,
            ),
        ];
        for (section, clauses, op, kind, bonus) in groups {
            let Some(clauses) = clauses else { continue };
            if clauses.clauses.is_empty() {
                if section == "success" {
                    return refuse(
                        section,
                        "clauses",
                        "a task needs at least one success clause",
                    );
                }
                continue;
            }
            let mut acc = None;
            for (i, c) in clauses.clauses.iter().enumerate() {
                let at = format!("{section}[{i}] ({} {})", c.subject, c.relation.name());
                let p = self.clause(&at, c)?;
                self.clauses.push((kind, i, p));
                acc = Some(acc.map_or(p, |a| self.logic(op, a, p)));
            }
            let acc = acc.expect("non-empty");
            let stop = self.add(TaskNode::Terminate { kind });
            self.graph.connect(acc, "value", stop, "value");
            if let Some(w) = bonus {
                self.reward(acc, section.to_owned(), w * scale);
            }
        }
        let time = self.source(TaskNode::GetTime { since_reset: true });
        let late = self.compare(time, CmpOp::Ge, spec.timeout_s);
        let stop = self.add(TaskNode::Terminate {
            kind: TerminationKind::Timeout,
        });
        self.graph.connect(late, "value", stop, "value");
        Ok(())
    }

    /// One clause's predicate (a `[1]` bool) and its shaping terms.
    fn clause(&mut self, at: &str, c: &Clause) -> Result<NodeId, SpecError> {
        if c.relation == Relation::Touches {
            return refuse(
                at,
                "relation",
                "`touches` waits for GetContact lowering in es-env",
            );
        }
        check_fields(at, c)?;
        let scale = self.scale();
        let weight = c.weight.unwrap_or_default() * scale;
        let term = |kind: &str| {
            c.term
                .clone()
                .unwrap_or_else(|| format!("{}_{kind}", c.subject))
        };
        match c.relation {
            // A body inside a region: each world axis of its position strictly inside the box;
            // shaped, its distance to the box's centre (design note section 4.7.1).
            Relation::Inside if c.object.is_some() => {
                let (centre, half) = self.region(at, c.object.as_deref().expect("matched"))?;
                let body = self.body(at, "subject", &c.subject)?;
                let mut acc = None;
                for axis in 0..3 {
                    let lane = self.coordinate(at, body, axis as u64, JointQuantity::Position)?;
                    let (gt, lt) = (
                        self.compare(lane, CmpOp::Gt, centre[axis] - half[axis]),
                        self.compare(lane, CmpOp::Lt, centre[axis] + half[axis]),
                    );
                    let within = self.logic(LogicOp::And, gt, lt);
                    acc = Some(acc.map_or(within, |prev| self.logic(LogicOp::And, prev, within)));
                }
                if c.shaping.is_some() {
                    let p = self.pose(body.id);
                    let offset = self.minus_point((p, "pos"), centre);
                    let dist = self.norm((offset, "value"));
                    self.distance_term(dist, term("distance"), weight);
                }
                Ok(acc.expect("three axes"))
            }
            // A body above (below) another: the difference of their world heights beyond `m`.
            Relation::Above | Relation::Below if c.object.is_some() => {
                let subject = self.body(at, "subject", &c.subject)?;
                let other = self.body(at, "object", c.object.as_deref().expect("matched"))?;
                let z_s = self.coordinate(at, subject, 2, JointQuantity::Position)?;
                let z_o = self.coordinate(at, other, 2, JointQuantity::Position)?;
                let (hi, lo) = if c.relation == Relation::Above {
                    (z_s, z_o)
                } else {
                    (z_o, z_s)
                };
                let ty = self.out((hi, "value"));
                let node = TaskNode::Arith {
                    op: ArithOp::Sub,
                    ty,
                };
                let gap = self.pair((hi, "value"), (lo, "value"), node);
                Ok(self.compare(gap, CmpOp::Gt, c.m.unwrap_or(0.0)))
            }
            // A body (not a hinge or slide joint, not `<body>.<axis>`): its speed `‖v‖`, and its
            // angular rate when `angular` bounds it. A free joint named as its body (an
            // unnamed one, packet M17/G9) is that body's, as `vocab::relations` reads it.
            Relation::Still
                if self.dotted(&c.subject).is_none()
                    && !(self.scene.joints.iter()).any(|j| {
                        j.name == c.subject && matches!(j.kind, JointKind::Hinge | JointKind::Slide)
                    }) =>
            {
                let body = self.body(at, "subject", &c.subject)?;
                self.free_joint(at, "subject", body)?;
                let vel = self.velocity(body.id);
                let slow = self.norm_below((vel, "linear"), c.speed.expect("checked"));
                Ok(match c.angular {
                    Some(rate) => {
                        let calm = self.norm_below((vel, "angular"), rate);
                        self.logic(LogicOp::And, slow, calm)
                    }
                    None => slow,
                })
            }
            Relation::Inside => {
                let [lo, hi] = c.range.expect("checked");
                let x = self.scalar(at, &c.subject, JointQuantity::Position)?;
                let (a, b) = (
                    self.compare(x, CmpOp::Gt, lo),
                    self.compare(x, CmpOp::Lt, hi),
                );
                if let Some([r0, r1]) = c.ramp {
                    let ramp = self.normalize((x, "value"), vec![r0], vec![r1], [0.0, 1.0]);
                    self.reward(ramp, term("ramp"), weight);
                }
                Ok(self.logic(LogicOp::And, a, b))
            }
            Relation::Above | Relation::Below => {
                let x = self.scalar(at, &c.subject, JointQuantity::Position)?;
                let op = if c.relation == Relation::Above {
                    CmpOp::Gt
                } else {
                    CmpOp::Lt
                };
                Ok(self.compare(x, op, c.value.expect("checked")))
            }
            Relation::Still => {
                if c.angular.is_some() {
                    return refuse(
                        at,
                        "angular",
                        "a body's bound; the subject is one coordinate",
                    );
                }
                let v = self.scalar(at, &c.subject, JointQuantity::Velocity)?;
                let speed = c.speed.expect("checked");
                let (a, b) = (
                    self.compare(v, CmpOp::Gt, -speed),
                    self.compare(v, CmpOp::Lt, speed),
                );
                Ok(self.logic(LogicOp::And, a, b))
            }
            Relation::Near | Relation::FartherThan => {
                let m = c.m.expect("checked");
                let s = self.body(at, "subject", &c.subject)?.id;
                let s = self.pose(s);
                let offset = if let Some(o) = &c.object {
                    let o = self.body(at, "object", o)?.id;
                    let o = self.pose(o);
                    let ty = self.out((s, "pos"));
                    let node = TaskNode::Arith {
                        op: ArithOp::Sub,
                        ty,
                    };
                    self.pair((s, "pos"), (o, "pos"), node)
                } else {
                    if m >= 1.0 {
                        return refuse(at, "m", "against a point, offsets clamp at 1 m per axis");
                    }
                    self.minus_point((s, "pos"), c.point.expect("checked"))
                };
                let dist = self.norm((offset, "value"));
                if c.shaping.is_some() {
                    self.distance_term(dist, term("distance"), weight);
                }
                let op = if c.relation == Relation::Near {
                    CmpOp::Lt
                } else {
                    CmpOp::Gt
                };
                Ok(self.compare(dist, op, m))
            }
            Relation::OrientationMatches => {
                let s = self.body(at, "subject", &c.subject)?.id;
                let o = self
                    .body(at, "object", c.object.as_deref().expect("checked"))?
                    .id;
                let (s, o) = (self.pose(s), self.pose(o));
                let ty = self.out((s, "quat"));
                let dot = self.pair((s, "quat"), (o, "quat"), TaskNode::Dot { ty });
                let d = self.math(dot, MathFunc::Abs);
                // `|q·g| ≥ cos(theta/2)`: the rotation between them is at most theta.
                let half = c.within_deg.expect("checked").to_radians() / 2.0;
                let reached = self.compare(d, CmpOp::Ge, approx::sin_cos_f64(half).1);
                if c.shaping.is_some() {
                    self.inverse_angle(d, &term("inverse_angle"), weight);
                }
                Ok(reached)
            }
            Relation::Touches => unreachable!("refused above"),
        }
    }

    fn math(&mut self, from: NodeId, func: MathFunc) -> NodeId {
        self.feed((from, "value"), "value", |ty| TaskNode::MathFn {
            func,
            approx: false,
            ty,
        })
    }

    /// `weight / (s + 0.1)` as one clamped ramp per knot interval (a `Reward` cannot take a
    /// reciprocal: `Arith Div` needs a dimensionless divisor), exact at every knot, plus the
    /// floor `1 / (√8 + 0.1)` paid every step (`time since reset ≥ 0`).
    fn inverse_angle(&mut self, d: NodeId, term: &str, weight: f64) {
        let gap = self.normalize((d, "value"), vec![1.0], vec![0.0], [0.0, 8.0]);
        let s = self.math(gap, MathFunc::Sqrt);
        let f: Vec<f64> = KNOTS.iter().map(|s| 1.0 / (s + 0.1)).collect();
        for k in 0..KNOTS.len() - 1 {
            // 1 at s ≤ s_k, 0 at s ≥ s_{k+1}.
            let ramp = self.normalize((s, "value"), vec![KNOTS[k + 1]], vec![KNOTS[k]], [0.0, 1.0]);
            self.reward(ramp, format!("{term}_{k}"), weight * (f[k] - f[k + 1]));
        }
        let time = self.source(TaskNode::GetTime { since_reset: true });
        let always = self.compare(time, CmpOp::Ge, 0.0);
        self.reward(always, format!("{term}_floor"), weight * f[KNOTS.len() - 1]);
    }
}

/// Which fields a relation takes, and which it needs.
fn check_fields(at: &str, c: &Clause) -> Result<(), SpecError> {
    let vocab::Takes {
        fields,
        needs,
        shaping,
    } = vocab::takes(c.relation, c.object.is_some());
    let present = [
        ("object", c.object.is_some()),
        ("point", c.point.is_some()),
        ("range", c.range.is_some()),
        ("value", c.value.is_some()),
        ("m", c.m.is_some()),
        ("speed", c.speed.is_some()),
        ("angular", c.angular.is_some()),
        ("within_deg", c.within_deg.is_some()),
        ("shaping", c.shaping.is_some()),
        ("weight", c.weight.is_some()),
        ("ramp", c.ramp.is_some()),
        ("term", c.term.is_some()),
    ];
    let has = |f: &str| present.iter().any(|(n, p)| *p && *n == f);
    // Where `inside`, `above` / `below` and `near` / `farther_than` measure from: one of two.
    let either = match c.relation {
        Relation::Inside => Some("range"),
        Relation::Above | Relation::Below => Some("value"),
        Relation::Near | Relation::FartherThan => Some("point"),
        _ => None,
    };
    if let Some(alt) = either.filter(|alt| has("object") && has(alt)) {
        return refuse(at, "object", format!("either `object` or `{alt}`"));
    }
    let shaped: &[&str] = match (c.shaping, shaping) {
        (None, _) => &[],
        (Some(s), Some(t)) if s == t && s == Shaping::Ramp => {
            &["shaping", "weight", "term", "ramp"]
        }
        (Some(s), Some(t)) if s == t => &["shaping", "weight", "term"],
        (Some(_), t) => {
            let want = match t {
                Some(Shaping::Distance) => "`distance`",
                Some(Shaping::Ramp) => "`ramp`",
                Some(Shaping::InverseAngle) => "`inverse_angle`",
                None => "no shaping",
            };
            return refuse(
                at,
                "shaping",
                format!("`{}` takes {want}", c.relation.name()),
            );
        }
    };
    if let Some((f, _)) = present
        .iter()
        .find(|(f, p)| *p && !fields.contains(f) && !shaped.contains(f))
    {
        let why = if ["weight", "ramp", "term"].contains(f) {
            "needs `shaping`".to_owned()
        } else {
            format!("does not apply to `{}`", c.relation.name())
        };
        return refuse(at, *f, why);
    }
    for f in needs {
        if !has(f) {
            return refuse(at, *f, "required");
        }
    }
    if c.shaping.is_some() && !has("weight") {
        return refuse(at, "weight", "required with `shaping`");
    }
    if c.shaping == Some(Shaping::Ramp) && !has("ramp") {
        return refuse(at, "ramp", "required with `shaping = \"ramp\"`");
    }
    if matches!(c.relation, Relation::Near | Relation::FartherThan) && has("object") == has("point")
    {
        return refuse(at, "object", "either `object` or `point`");
    }
    Ok(())
}
