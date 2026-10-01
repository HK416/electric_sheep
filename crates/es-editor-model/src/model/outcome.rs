//! Why a timed-out attempt failed, read from where the object went (packet M13/Z4).
//!
//! The cube tasks declare no failure predicate, so every failed attempt ends as a timeout and
//! "which `Terminate` fired" names nothing (plan Z, "what the first real runs changed"). The
//! attempt's `.estraj` still holds every body's world pose per tick, and the template's
//! `[outcome]` says which body is the object and which scene geoms are the place it belongs:
//!
//! - [`Outcome::NotReleased`]: last seen inside the target region, and the `[outcome] release`
//!   joint never went past its `above` since the object last went in (packet M13/R4); or, with a
//!   release condition, last seen held over the target - inside its x/y footprint, above its top -
//!   and never let go since it last came over the footprint (a real run, 2026-09-29, carried the
//!   cube right above the bin and held it there until time ran out);
//! - [`Outcome::InsideTooLate`]: otherwise last seen inside, or over the target with a release
//!   condition, and let go, but time ran out (over it: still falling, or perched on a wall);
//! - [`Outcome::NeverLifted`]: otherwise, never `lift_m` above where it started;
//! - [`Outcome::LeftOutside`]: otherwise - lifted, and last seen away from the target.
//!
//! Inside and over are asked first: an object that got there got there, however it did. Without a
//! release condition nothing is over the target: inside is always too late, and an object above
//! the target's top is outside it. The release joint's position is its `qpos` value in the
//! trajectory, at the index the scene's joint order gives it. The target region
//! is the axis-aligned box of the triangles the renderer tessellates for the geoms of the target's
//! stem ([`stem`], the rule ① Scene groups objects by), at the scene's own pose. Nothing here is a
//! second loader: the scene is the Replay panel's ([`load_scene`]), the trajectory `es_env`'s and
//! the triangles `es_render`'s.
//!
//! No trajectory - an old run, one written elsewhere, a file that does not parse - is no class,
//! and the attempt keeps the cause the evaluation recorded (review focus 3).

use std::collections::BTreeMap;
use std::path::Path;

use es_core::StableId;
use es_env::traj::Trajectory;
use es_eval::episodes::EpisodeRow;
use es_eval::run_dir::RunDir;
use es_math::Quat;
use es_render::TriScene;

use crate::model::replay_view::load_scene;
use crate::model::scene_view::stem;
use crate::model::teach::width;
use crate::model::template::OutcomeSpec;

/// `EpisodeRow::termination` of an attempt that ran out of time.
const TIMEOUT: &str = "timeout";

/// Why a timed-out attempt failed; [`classify`] asks the last two first (the module's list).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    /// Never `lift_m` above where it started.
    NeverLifted,
    /// Lifted, and last seen away from the target: neither inside it nor held over it.
    LeftOutside,
    /// Inside the target, or held over it, and the release joint never past its opening since it
    /// came in or over; only with a release condition.
    NotReleased,
    /// Inside the target when time ran out; or, with a release condition, over it and let go.
    InsideTooLate,
}

impl Outcome {
    pub const ALL: [Outcome; 4] = [
        Outcome::NeverLifted,
        Outcome::LeftOutside,
        Outcome::NotReleased,
        Outcome::InsideTooLate,
    ];
}

/// An axis-aligned box, world metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Region {
    fn contains(&self, p: [f64; 3]) -> bool {
        (0..3).all(|i| self.min[i] <= p[i] && p[i] <= self.max[i])
    }

    /// The smallest box holding every point; `None` for none.
    fn around(points: &[[f64; 3]]) -> Option<Self> {
        let first = *points.first()?;
        Some(points.iter().fold(
            Self {
                min: first,
                max: first,
            },
            |r, p| Self {
                min: std::array::from_fn(|i| r.min[i].min(p[i])),
                max: std::array::from_fn(|i| r.max[i].max(p[i])),
            },
        ))
    }
}

/// The class of an object's path, first tick first; `None` for no ticks. `release`, when the
/// template declares one, is the release joint's position at each tick and the opening past
/// which the object counts as let go. Asked in the module's order: inside, then (with `release`)
/// over, then the lift.
pub fn classify(
    path: &[[f64; 3]],
    target: &Region,
    lift_m: f64,
    release: Option<(&[f64], f64)>,
) -> Option<Outcome> {
    let (first, last) = (path.first()?, path.last()?);
    let rise = path
        .iter()
        .map(|p| p[2] - first[2])
        .fold(f64::NEG_INFINITY, f64::max);
    let inside = target.contains(*last);
    // Over the target: inside its x/y footprint, anywhere from its floor up.
    let over = |p: [f64; 3]| target.contains([p[0], p[1], p[2].min(target.max[2])]);
    // The stretch the release is judged over: inside the box for an object that ended in it,
    // over the target for one held over it.
    let within = |p: [f64; 3]| if inside { target.contains(p) } else { over(p) };
    Some(if inside || (release.is_some() && over(*last)) {
        // The first tick of that stretch that the path ends on.
        let entered = (path.iter().rposition(|p| !within(*p))).map_or(0, |i| i + 1);
        let let_go =
            release.is_none_or(|(opening, above)| opening.iter().skip(entered).any(|&q| q > above));
        if let_go {
            Outcome::InsideTooLate
        } else {
            Outcome::NotReleased
        }
    } else if rise < lift_m {
        Outcome::NeverLifted
    } else {
        Outcome::LeftOutside
    })
}

/// One scene under one `[outcome]`: the object's body, the target region and the release
/// joint's `qpos` index, found once.
#[derive(Clone, Debug, PartialEq)]
pub struct Judge {
    object: StableId,
    pub target: Region,
    lift_m: f64,
    release: Option<(usize, f64)>,
}

impl Judge {
    /// `None` when the scene does not load, names no body `object`, has no geom of the
    /// `target` stem, or has no one-value joint the `release` names.
    ///
    /// ponytail: the region is the target at the scene's pose, which is where a fixed bin stays;
    /// a target that moves would be re-posed at the attempt's last tick.
    pub fn new(scene: &Path, spec: &OutcomeSpec) -> Option<Self> {
        let scene = load_scene(scene).ok()?;
        let object = scene.bodies.iter().find(|b| b.name == spec.object)?.id;
        // `qpos` is the scene's joint order laid out `MuJoCo`'s way, as `teach` reads it.
        let release = match &spec.release {
            None => None,
            Some(r) => {
                let at =
                    (scene.joints.iter()).position(|j| j.name == r.joint && width(j.kind) == 1)?;
                let index = scene.joints[..at].iter().map(|j| width(j.kind)).sum();
                Some((index, r.above))
            }
        };
        let tris = TriScene::from_scene(&scene).ok()?;
        let points: Vec<[f64; 3]> = (tris.tris.iter())
            .filter(|t| (tris.names.get(&t.seg)).is_some_and(|n| stem(n) == spec.target))
            .flat_map(|t| t.v.map(|v| v.map(f64::from)))
            .collect();
        Some(Self {
            object,
            target: Region::around(&points)?,
            lift_m: spec.lift_m,
            release,
        })
    }

    /// The attempt's class; `None` when the trajectory is empty, does not hold the object, or
    /// is too narrow for the release joint.
    pub fn classify(&self, traj: &Trajectory) -> Option<Outcome> {
        let path: Vec<[f64; 3]> = (0..traj.ticks())
            .map(|t| {
                let p = traj.poses(t).get(&self.object)?.position;
                Some([p.x, p.y, p.z])
            })
            .collect::<Option<_>>()?;
        let opening: Vec<f64> = match self.release {
            Some((i, _)) => (0..traj.ticks())
                .map(|t| traj.qpos(t).get(i).copied())
                .collect::<Option<_>>()?,
            None => Vec::new(),
        };
        let release = self.release.map(|(_, above)| (&opening[..], above));
        classify(&path, &self.target, self.lift_m, release)
    }
}

/// Each timed-out attempt's class, by cell: those whose `traj/<cell>.estraj` reads and holds the
/// object. Empty when the scene, the object or the target cannot be found.
///
/// ponytail: every timed-out attempt's trajectory is read whole when ⑤ opens a run (about a
/// megabyte each at the demo's length); a summary written beside it at evaluation time would
/// make this free.
pub fn outcomes(
    scene: &Path,
    spec: &OutcomeSpec,
    rows: &[EpisodeRow],
    run: &RunDir,
) -> BTreeMap<String, Outcome> {
    let Some(judge) = Judge::new(scene, spec) else {
        return BTreeMap::new();
    };
    rows.iter()
        .filter(|r| r.termination == TIMEOUT)
        .filter_map(|r| {
            let traj = Trajectory::read(&run.traj_path(&r.cell)).ok()?;
            Some((r.cell.clone(), judge.classify(&traj)?))
        })
        .collect()
}

/// The rotation between two orientations, in degrees: `2 acos |a . b|` (a quaternion and its
/// negation are one orientation).
pub fn angle_deg(a: Quat, b: Quat) -> f64 {
    let dot = (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
        .abs()
        .min(1.0);
    2.0 * dot.acos().to_degrees()
}

/// A reorientation (`[outcome] kind = "reorient"`, packet M16/H7): each timed-out attempt's
/// final angle off the goal, in degrees, by cell - the rotation between the `object` and the
/// `target` bodies' orientations at the last tick of its trajectory. What the success predicate
/// (`|q_cube . q_goal| >= cos(angle / 2)`) still missed when time ran out. Empty when the scene or
/// either body cannot be found; an attempt without a readable trajectory has no angle.
pub fn final_angles(
    scene: &Path,
    spec: &OutcomeSpec,
    rows: &[EpisodeRow],
    run: &RunDir,
) -> BTreeMap<String, f64> {
    let Ok(scene) = load_scene(scene) else {
        return BTreeMap::new();
    };
    let id = |name: &str| (scene.bodies.iter()).find(|b| b.name == name).map(|b| b.id);
    let (Some(object), Some(target)) = (id(&spec.object), id(&spec.target)) else {
        return BTreeMap::new();
    };
    rows.iter()
        .filter(|r| r.termination == TIMEOUT)
        .filter_map(|r| {
            let traj = Trajectory::read(&run.traj_path(&r.cell)).ok()?;
            let poses = traj.poses(traj.ticks().checked_sub(1)?);
            let (o, t) = (poses.get(&object)?, poses.get(&target)?);
            Some((r.cell.clone(), angle_deg(o.orientation, t.orientation)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use es_core::PhysTick;
    use es_physics_core::backend::{IndexRange, ModelInfo, StateView};

    use crate::model::project::tests::{cube, repo};

    fn judge() -> Judge {
        let spec = cube().outcome.expect("[outcome]");
        Judge::new(&repo().join(cube().scene), &spec).expect("the cube and the bin")
    }

    /// A trajectory of one body, `body`, along `path` (identity orientation), with the gripper
    /// at the demo scene's `qpos[5]` at `grip[k]` on tick `k`.
    fn traj(body: StableId, path: &[[f64; 3]], grip: &[f64]) -> Trajectory {
        assert_eq!(path.len(), grip.len());
        let mut model = ModelInfo {
            nq: 6,
            nbody: 1,
            ..ModelInfo::default()
        };
        model.body.insert(body, IndexRange { start: 0, len: 1 });
        let mut traj = Trajectory::new(&model);
        for (k, (p, g)) in path.iter().zip(grip).enumerate() {
            let state = StateView {
                n_envs: 1,
                tick: PhysTick(k as u64),
                qpos: &[0.0, 0.0, 0.0, 0.0, 0.0, *g],
                qvel: &[],
                act: &[],
                sensordata: &[],
                xpos: p,
                xquat: &[0.0, 0.0, 0.0, 1.0],
            };
            traj.push(&model, &state, 0).expect("one body");
        }
        traj
    }

    /// The bin of the demo scene is the box its floor and four walls cover
    /// (`so101_pick_place.xml`: centre 0.14, -0.1; 5.8 cm half-width with the walls; 9 cm tall).
    #[test]
    fn the_target_is_the_box_of_the_bin_geoms() {
        let j = judge();
        let want = Region {
            min: [0.082, -0.158, 0.0],
            max: [0.198, -0.042, 0.09],
        };
        for i in 0..3 {
            assert!((j.target.min[i] - want.min[i]).abs() < 1e-6, "{j:?}");
            assert!((j.target.max[i] - want.max[i]).abs() < 1e-6, "{j:?}");
        }
        let spec = |object: &str, target: &str| OutcomeSpec {
            object: object.into(),
            target: target.into(),
            ..cube().outcome.expect("[outcome]")
        };
        let scene = repo().join(cube().scene);
        assert!(
            Judge::new(&scene, &spec("sphere", "bin")).is_none(),
            "no body"
        );
        assert!(
            Judge::new(&scene, &spec("cube", "crate")).is_none(),
            "no geom"
        );
        assert!(Judge::new(Path::new("no/such.xml"), &spec("cube", "bin")).is_none());
        // The gripper is the sixth hinge, `qpos[5]`; a release joint the scene lacks is no judge.
        assert_eq!(j.release, Some((5, 0.85)));
        let mut spec = spec("cube", "bin");
        spec.release.as_mut().expect("release").joint = "thumb".into();
        assert!(Judge::new(&scene, &spec).is_none(), "no joint");
    }

    /// One synthetic path per class, on the demo scene's own cube body; nothing to read is none.
    #[test]
    fn each_class_from_a_synthetic_trajectory() {
        let j = judge();
        let start = [0.24, 0.0, 0.02];
        let never = [start, [0.22, 0.03, 0.021], [0.20, 0.05, 0.02]];
        let dropped = [start, [0.24, 0.0, 0.10], [0.30, 0.05, 0.02]];
        // Above the bin's walls at tick 1, inside from tick 2.
        let inside = [start, [0.14, -0.1, 0.12], [0.14, -0.1, 0.028]];
        let shut = [0.0, 0.09, 0.09];
        let opened = [0.0, 0.09, 1.2];
        // Opened wide above the bin, then shut again before the cube went in: never let go.
        let reshut = [0.0, 1.2, 0.09];
        for (path, grip, want) in [
            (&never[..], shut, Outcome::NeverLifted),
            (&dropped[..], opened, Outcome::LeftOutside),
            (&inside[..], shut, Outcome::NotReleased),
            (&inside[..], reshut, Outcome::NotReleased),
            (&inside[..], opened, Outcome::InsideTooLate),
            // Exactly at the threshold is not past it (the Task IR's `Gt`).
            (&inside[..], [0.0, 0.09, 0.85], Outcome::NotReleased),
        ] {
            let t = traj(j.object, path, &grip);
            assert_eq!(j.classify(&t), Some(want), "{path:?} {grip:?}");
        }
        // Without a release condition, inside is too late whatever the gripper did.
        let no_release = Judge {
            release: None,
            ..j.clone()
        };
        let t = traj(j.object, &inside, &shut);
        assert_eq!(no_release.classify(&t), Some(Outcome::InsideTooLate));
        assert_eq!(j.classify(&traj(j.object, &[], &[])), None, "no tick");
        let other = StableId::from_path("body/elsewhere");
        let t = traj(other, &dropped, &shut);
        assert_eq!(j.classify(&t), None, "no cube in it");
        // Just under the lift is never lifted.
        let almost = [start, [0.24, 0.0, 0.0399], [0.24, 0.0, 0.02]];
        assert_eq!(
            j.classify(&traj(j.object, &almost, &shut)),
            Some(Outcome::NeverLifted)
        );
    }

    /// Above the bin's walls, inside its footprint, is over it only with a release condition.
    #[test]
    fn an_object_held_over_the_target_is_judged_by_its_release() {
        let j = judge();
        let start = [0.24, 0.0, 0.02];
        // Lifted, carried right above the bin (its top is 0.09 m) and held there.
        let held = [start, [0.14, -0.1, 0.13], [0.139, -0.092, 0.135]];
        // Lifted, and dropped (or held up) just beside the bin's x = 0.198 wall.
        let beside = [start, [0.21, -0.1, 0.13], [0.21, -0.1, 0.02]];
        let beside_high = [start, [0.21, -0.1, 0.13], [0.21, -0.1, 0.13]];
        for (path, grip, want) in [
            (&held[..], [0.0, 0.09, 0.09], Outcome::NotReleased),
            // Opened wide before it came over the bin, shut since: not let go over it.
            (&held[..], [1.2, 0.09, 0.09], Outcome::NotReleased),
            // Let go over the bin, still in the air when time ran out: there, too late.
            (&held[..], [0.0, 0.09, 1.2], Outcome::InsideTooLate),
            (&beside[..], [0.0, 0.09, 1.2], Outcome::LeftOutside),
            (&beside_high[..], [0.0, 0.09, 0.09], Outcome::LeftOutside),
        ] {
            let t = traj(j.object, path, &grip);
            assert_eq!(j.classify(&t), Some(want), "{path:?} {grip:?}");
        }
        // Without a release condition nothing is over the bin: above it is outside.
        let no_release = Judge {
            release: None,
            ..j.clone()
        };
        let t = traj(j.object, &held, &[0.0, 0.09, 0.09]);
        assert_eq!(no_release.classify(&t), Some(Outcome::LeftOutside));
    }

    /// The run that found the held-over case (2026-09-29): two 60,000-step checkpoint preview
    /// attempts lifted the cube about 11.5 cm, carried it right above the bin and held it there,
    /// the gripper shut, until time ran out. Read from the owner's checkout, never committed.
    #[test]
    fn the_real_attempts_held_over_the_bin_are_not_released() {
        const EVIDENCE: &str =
            "F:/Projects/electric_sheep/target/yv/cube-hint-wait1/runs/001/preview/60000/traj";
        let j = judge();
        for cell in ["nominal-01", "nominal-02"] {
            let path = Path::new(EVIDENCE).join(format!("{cell}.estraj"));
            if !path.is_file() {
                eprintln!("skipped: {} is not on this machine", path.display());
                continue;
            }
            let traj = Trajectory::read(&path).expect("the evidence file");
            assert_eq!(j.classify(&traj), Some(Outcome::NotReleased), "{cell}");
            // What the editor said before: lifted but left outside.
            let before = Judge {
                release: None,
                ..j.clone()
            };
            assert_eq!(before.classify(&traj), Some(Outcome::LeftOutside), "{cell}");
        }
    }

    /// The run that found it (packet M13/R4, 2026-09-29): a checkpoint preview attempt put the
    /// cube in the bin at tick 451 of 1,800 and never opened the gripper past 0.09 rad. Read from
    /// the owner's checkout, never committed; elsewhere the test says so and passes.
    #[test]
    fn the_real_attempt_that_never_let_go_is_not_released() {
        const EVIDENCE: &str = "F:/Projects/electric_sheep/target/yv/cube-hint-wait1/runs/001/preview/5000/traj/nominal-02.estraj";
        let path = Path::new(EVIDENCE);
        if !path.is_file() {
            eprintln!("skipped: {EVIDENCE} is not on this machine");
            return;
        }
        let traj = Trajectory::read(path).expect("the evidence file");
        let j = judge();
        assert_eq!(j.classify(&traj), Some(Outcome::NotReleased));
        // What the editor said before the release condition.
        let before = Judge { release: None, ..j };
        assert_eq!(before.classify(&traj), Some(Outcome::InsideTooLate));
    }

    /// Packet M16/H7: the angle between two orientations, either sign of a quaternion.
    #[test]
    fn the_angle_between_orientations() {
        let q = |deg: f64| {
            let h = deg.to_radians() / 2.0;
            Quat {
                x: 0.0,
                y: 0.0,
                z: h.sin(),
                w: h.cos(),
            }
        };
        assert!(angle_deg(q(0.0), q(0.0)).abs() < 1e-9);
        assert!((angle_deg(q(10.0), q(40.0)) - 30.0).abs() < 1e-9);
        let minus = Quat {
            x: -0.0,
            y: -0.0,
            z: -q(40.0).z,
            w: -q(40.0).w,
        };
        assert!((angle_deg(q(10.0), minus) - 30.0).abs() < 1e-9, "-q is q");
        assert!((angle_deg(q(0.0), q(180.0)) - 180.0).abs() < 1e-9);
    }

    /// The student's real evaluation (plan H, E2): every timed-out attempt has an angle, and
    /// none is under the success angle. Read from the owner's checkout, never committed.
    #[test]
    fn the_real_student_attempts_end_off_the_goal() {
        const EVIDENCE: &str = "F:/Projects/electric_sheep/runs/shadow-hand/student-001/eval";
        let dir = Path::new(EVIDENCE);
        if !dir.join("episodes.json").is_file() {
            eprintln!("skipped: {EVIDENCE} is not on this machine");
            return;
        }
        let hand = crate::model::teacher::tests::hand();
        let spec = hand.outcome.as_ref().expect("[outcome]");
        let run = RunDir::open(dir).expect("the evaluation");
        let rows = es_eval::episodes::read_episodes(dir)
            .expect("episodes.json")
            .expect("rows");
        let angles = final_angles(&repo().join(&hand.scene), spec, &rows, &run);
        let timeouts = rows.iter().filter(|r| r.termination == TIMEOUT).count();
        assert!(timeouts > 0 && angles.len() == timeouts, "{angles:?}");
        let success = spec.angle_rad.to_degrees();
        assert!(
            angles.values().all(|&a| a > success && a <= 180.0),
            "{angles:?}"
        );
    }

    /// The committed E2 fixture sweeps the arm and never touches the cube.
    #[test]
    fn the_committed_fixture_never_lifts_the_cube() {
        let traj = Trajectory::read(
            &repo().join("tests/fixtures/visible-learning/run/traj/nominal-00.estraj"),
        )
        .expect("the fixture");
        assert_eq!(judge().classify(&traj), Some(Outcome::NeverLifted));
    }
}
