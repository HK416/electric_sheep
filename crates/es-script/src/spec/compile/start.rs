//! The `start` section: each start item as a `ResetState` or `Randomization` node, the scene's
//! own poses for every free body no item places, and the robot's tendon couplings.

use es_assets::scene::{Joint, JointKind, TendonKind};
use es_core::StableId;
use es_ir::task::{Distribution, TaskNode};
// A number written into a document has musl's bits, not the host's (spec 5.3, M17 G3d).
use es_math::approx;

use super::{no, Compiler};
use crate::spec::{refuse, Draw, SpecError, StartItem};

impl<'a> Compiler<'a> {
    fn reset(&mut self, target: String, dist: Distribution, stream: String, dice: bool) {
        self.add(if dice {
            TaskNode::Randomization {
                target,
                dist,
                stream: stream.clone(),
            }
        } else {
            TaskNode::ResetState {
                target,
                dist,
                stream: stream.clone(),
            }
        });
        self.streams.insert(stream);
    }

    pub(super) fn start_item(
        &mut self,
        at: &str,
        item: &StartItem,
        strength: Option<f64>,
    ) -> Result<(), SpecError> {
        let dice = item.dice == Some(true);
        let k = strength.filter(|_| dice);
        let present = [
            ("value", item.value.is_some()),
            ("noise", item.noise.is_some()),
            ("range", item.range.is_some()),
            ("draw", item.draw.is_some()),
            ("tilt", item.tilt.is_some()),
            ("tilt_max_deg", item.tilt_max_deg.is_some()),
            ("coupled", item.coupled.is_some()),
            ("stream", item.stream.is_some()),
        ];
        let only = |allowed: &[&str]| match present.iter().find(|(f, p)| *p && !allowed.contains(f))
        {
            Some((f, _)) => refuse(at, *f, format!("does not apply to `{}`", item.what)),
            None => Ok(()),
        };
        if item.what == "robot.joints" {
            only(&["noise", "coupled"])?;
            let Some(noise) = item.noise else {
                return refuse(at, "noise", "required: the fraction of each joint's range");
            };
            let noise = k.map_or(noise, |k| noise * k);
            let coupled = if item.coupled == Some(true) {
                self.couplings(at)?
            } else {
                Vec::new()
            };
            for j in self.joints.clone() {
                let range = |j: &Joint| match j.range {
                    Some(r) => Ok(r),
                    None => refuse(at, "noise", format!("joint `{}` has no range", j.name)),
                };
                let (stream, (lo, hi)) = match coupled.iter().find(|(_, j0, _)| j0.id == j.id) {
                    Some((j1, _, c)) => {
                        let (lo, hi) = range(j1)?;
                        (*j1, (lo * c, hi * c))
                    }
                    None => (j, range(j)?),
                };
                let dist = between(noise * lo, noise * hi);
                self.reset(
                    format!("joint.{}.qpos", j.name),
                    dist,
                    format!("reset.{}", stream.name),
                    dice,
                );
            }
            return Ok(());
        }
        if let Some((body, field)) = self.dotted(&item.what) {
            let j = self.free_joint(at, "what", body)?;
            let q = self.qpos_of(j);
            let stream = item.stream.clone().unwrap_or_else(|| item.what.clone());
            if field == "orientation" {
                // The quaternion `(w, x, y, z)` from `qpos[q + 3]`, one draw per lane; lanes
                // on one stream share it (`es_env::rng` addresses a draw by its stream).
                let lanes: Vec<(Distribution, String)> = match item.draw {
                    Some(Draw::Yaw) => {
                        only(&["draw", "tilt", "stream"])?;
                        let t = item.tilt.unwrap_or(0.0);
                        // `(1, t, −t·u, u)`, each lane linear in one `u ~ U(−1, 1)`.
                        [(1.0, 1.0), (t, t), (t, -t), (-1.0, 1.0)]
                            .into_iter()
                            .map(|(lo, hi)| (between(lo, hi), stream.clone()))
                            .collect()
                    }
                    Some(Draw::Tilt) => {
                        only(&["draw", "tilt_max_deg", "stream"])?;
                        let theta = item.tilt_max_deg.unwrap_or(f64::NAN);
                        if !(theta > 0.0 && theta < 180.0) {
                            return refuse(
                                at,
                                "tilt_max_deg",
                                "required, in (0, 180); 180 is `any`",
                            );
                        }
                        let m =
                            approx::tan_f64(theta.to_radians() / 2.0) / std::f64::consts::SQRT_2;
                        let ab = Distribution::Uniform { lo: -m, hi: m };
                        vec![
                            (Distribution::Constant(1.0), stream.clone()),
                            (ab.clone(), format!("{stream}.x")),
                            (ab, format!("{stream}.y")),
                            (
                                Distribution::Normal {
                                    mean: 0.0,
                                    std: 1.0,
                                },
                                stream.clone(),
                            ),
                        ]
                    }
                    Some(Draw::Any) => {
                        only(&["draw", "stream"])?;
                        ["w", "x", "y", "z"]
                            .iter()
                            .map(|l| {
                                (
                                    Distribution::Normal {
                                        mean: 0.0,
                                        std: 1.0,
                                    },
                                    format!("{stream}.{l}"),
                                )
                            })
                            .collect()
                    }
                    None => return refuse(at, "draw", "required: `yaw`, `tilt` or `any`"),
                };
                for (lane, (dist, stream)) in lanes.into_iter().enumerate() {
                    self.placed.insert(q + 3 + lane);
                    self.reset(format!("qpos[{}]", q + 3 + lane), dist, stream, dice);
                }
                return Ok(());
            }
            let Some(axis) = ["x", "y", "z"].iter().position(|a| *a == field) else {
                return refuse(
                    at,
                    "what",
                    format!("`{field}`: a body places by x, y, z or orientation"),
                );
            };
            only(&["value", "noise", "range", "stream"])?;
            let dist = value_or_range(at, item, k)?;
            self.placed.insert(q + axis);
            self.reset(format!("qpos[{}]", q + axis), dist, stream, dice);
            return Ok(());
        }
        let scene = self.scene;
        let Some(j) = scene.joints.iter().find(|j| j.name == item.what) else {
            return refuse(
                at,
                "what",
                no("`robot.joints`, joint or `<body>.x|y|z|yaw`", &item.what),
            );
        };
        only(&["value", "noise", "range", "stream"])?;
        let dist = value_or_range(at, item, k)?;
        let stream = item
            .stream
            .clone()
            .unwrap_or_else(|| format!("reset.{}", j.name));
        // `joint.<j>.qpos` writes the joint's first lane, a free joint's x too.
        self.placed.insert(self.qpos_of(j));
        self.reset(format!("joint.{}.qpos", j.name), dist, stream, dice);
        Ok(())
    }

    /// Every lane of a free joint outside the robot that no start item writes starts where the
    /// scene puts the body (`qpos0`: its pose, `x y z` then the quaternion `w x y z`, read bit
    /// for bit): a constant `ResetState` on stream `scene.<body>.pos.<i>` / `.quat.<i>`, in the
    /// scene's joint order. `Env::reset` zeroes `qpos` first, so without it the body would start
    /// at the origin (design note section 4.7).
    pub(super) fn scene_poses(&mut self) {
        let scene = self.scene;
        for j in &scene.joints {
            if j.kind != JointKind::Free || self.joints.iter().any(|r| r.id == j.id) {
                continue;
            }
            let Some(body) = scene.bodies.iter().find(|b| b.id == j.body) else {
                continue;
            };
            let (p, o) = (body.pose.position, body.pose.orientation);
            let q = self.qpos_of(j);
            for (lane, v) in [p.x, p.y, p.z, o.w, o.x, o.y, o.z].into_iter().enumerate() {
                if self.placed.contains(&(q + lane)) {
                    continue;
                }
                let stream = match lane {
                    0..3 => format!("scene.{}.pos.{lane}", body.name),
                    _ => format!("scene.{}.quat.{}", body.name, lane - 3),
                };
                let target = format!("qpos[{}]", q + lane);
                self.reset(target, Distribution::Constant(v), stream, false);
            }
        }
    }

    /// `(J1, J0, k)` for every two-joint fixed tendon over the robot's joints: `J0 = k · J1`
    /// holds the tendon `c1·J1 + c0·J0` at zero, `k = −c1 / c0`.
    fn couplings(&self, at: &str) -> Result<Vec<(&'a Joint, &'a Joint, f64)>, SpecError> {
        let robot = |id: StableId| self.joints.iter().copied().find(|j| j.id == id);
        let mut out = Vec::new();
        for t in &self.scene.tendons {
            let TendonKind::Fixed { joints } = &t.kind else {
                continue;
            };
            if !joints.iter().all(|(id, _)| robot(*id).is_some()) {
                continue;
            }
            let [(j1, c1), (j0, c0)] = joints[..] else {
                return refuse(
                    at,
                    "coupled",
                    format!(
                        "tendon `{}` couples {} joints, not two",
                        t.name,
                        joints.len()
                    ),
                );
            };
            out.push((
                robot(j1).expect("checked"),
                robot(j0).expect("checked"),
                -c1 / c0,
            ));
        }
        Ok(out)
    }
}

/// A uniform draw, or the constant when the bounds are the same bits.
fn between(lo: f64, hi: f64) -> Distribution {
    if lo.to_bits() == hi.to_bits() {
        Distribution::Constant(lo)
    } else {
        Distribution::Uniform { lo, hi }
    }
}

/// `value` (± `noise`) or `range`; `k` scales a 🎲 item's spread.
fn value_or_range(at: &str, item: &StartItem, k: Option<f64>) -> Result<Distribution, SpecError> {
    match (item.value, item.range) {
        (Some(v), None) => {
            let n = item.noise.unwrap_or(0.0) * k.unwrap_or(1.0);
            Ok(between(v - n, v + n))
        }
        (None, Some([lo, hi])) if item.noise.is_none() => Ok(match k {
            Some(k) => {
                let (c, h) = (f64::midpoint(lo, hi), (hi - lo) / 2.0);
                Distribution::Uniform {
                    lo: c - k * h,
                    hi: c + k * h,
                }
            }
            None => Distribution::Uniform { lo, hi },
        }),
        (None, Some(_)) => refuse(at, "noise", "a `range` is drawn as written"),
        (Some(_), Some(_)) => refuse(at, "range", "either `value` or `range`"),
        (None, None) => refuse(at, "value", "required: `value` or `range`"),
    }
}
