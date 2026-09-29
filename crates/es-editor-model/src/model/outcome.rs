//! Why a timed-out attempt failed, read from where the object went (packet M13/Z4).
//!
//! The cube tasks declare no failure predicate, so every failed attempt ends as a timeout and
//! "which `Terminate` fired" names nothing (plan Z, "what the first real runs changed"). The
//! attempt's `.estraj` still holds every body's world pose per tick, and the template's
//! `[outcome]` says which body is the object and which scene geoms are the place it belongs:
//!
//! - [`Outcome::NotReleased`]: last seen inside the target region, and the `[outcome] release`
//!   joint never went past its `above` since the object last went in (packet M13/R4);
//! - [`Outcome::InsideTooLate`]: otherwise last seen inside, but time ran out;
//! - [`Outcome::NeverLifted`]: otherwise, never `lift_m` above where it started;
//! - [`Outcome::LeftOutside`]: otherwise - lifted, and last seen outside.
//!
//! Inside is asked first: an object that got there got there, however it did. Without a release
//! condition, inside is always too late. The release joint's position is its `qpos` value in the
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
use es_render::TriScene;

use crate::model::replay_view::load_scene;
use crate::model::scene_view::stem;
use crate::model::teach::width;
use crate::model::template::OutcomeSpec;

/// `EpisodeRow::termination` of an attempt that ran out of time.
const TIMEOUT: &str = "timeout";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    NeverLifted,
    LeftOutside,
    NotReleased,
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
/// which the object counts as let go.
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
    Some(if target.contains(*last) {
        // The first tick of the stretch inside that the path ends on.
        let entered = (path.iter().rposition(|p| !target.contains(*p))).map_or(0, |i| i + 1);
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
