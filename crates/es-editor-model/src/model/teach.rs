//! ② Teach (`docs/design/editor-redesign.md` sections 3 and 11, packet M14/Q3): the project's
//! demonstration program (`teach.toml`) as a list of blocks in plain words, edited one field at
//! a time and checked on every edit, and tried once on the demonstrator.
//!
//! The program is `es_env::program::Program`, the very file `--expert` runs; nothing here reads
//! it a second way. What a program is *refused* for is what the demonstrator refuses it for:
//! [`Program::validate`], then [`ScriptedExpert::with_program`] on the template's scene, paced to
//! its Deployment IR as `es` paces it. What a person is *warned* about is what the demonstrator
//! would fail at in a run: a move the arm cannot reach for some object position the Task IR can
//! draw ([`so101_ik`] at the corners of the box its `Randomization` nodes draw from - never an
//! approximation), or a program that never closes the gripper or never opens it again. Neither
//! blocks saving, and a warning does not block a try either: the demonstrator then fails that
//! block by name (spec 17.2), which is what the try shows.
//!
//! A try is `es eval run --expert <teach.toml>` on a derived Evaluation IR - the template's first
//! suite, one explicit seed, no acceptance, derived as a checkpoint preview derives its own
//! ([`preview_evaluation`]) - into `<project>/try/<n>/`, and it is read back like any run: its one
//! attempt as a result [`Tile`], with the outcome class of packet M13/Z4. The seed is never one
//! the Evaluation IR judges the policy on ([`TRY_SEED_BASE`]).

use std::path::{Path, PathBuf};

use es_assets::scene::{Joint, JointKind, SceneDesc};
use es_data::training::{preview_evaluation, Cycle, PreviewRef};
use es_env::expert::{demo_cfg, so101_ik, Links};
use es_env::program::{place_point, Block, Program, ProgramError};
/// What a block's fields hold, for the screen that edits them.
pub use es_env::program::{Grip, Target, OBJECT};
use es_env::ScriptedExpert;
use es_eval::episodes::read_episodes;
use es_eval::run_dir::RunDir;
use es_ir::deployment::DeploymentIr;
use es_ir::evaluation::{EpisodeBatch, EvaluationIr, SeedPlan};
use es_ir::serial::{
    deployment_from_toml, evaluation_from_toml, evaluation_to_toml, parse_toml, task_from_toml,
    write_toml,
};
use es_ir::task::{Distribution, TaskIr, TaskNode};
use es_math::{units::DEG_TO_RAD, Vec3};

use crate::model::i18n::{fill, t, Lang, Strings};
use crate::model::labels::{cause_key, program_error_key};
use crate::model::outcome;
use crate::model::project::{evaluation_seeds, fresh_seed, Project};
use crate::model::replay_view::load_scene;
use crate::model::results::{tiles, Outcomes, Tile, TileFilter};
use crate::model::scene_view::ScenePreview;
use crate::model::template::{OutcomeSpec, Template};
use crate::model::watch::may_start;
use crate::model::workflow::PhaseState;

/// What a new move block is: the approach of the built-in program, a little higher.
const NEW_ABOVE_M: f64 = 0.05;
const NEW_PITCH_DEG: f64 = -85.0;
/// What a new grip block waits: five of the demo's 0.2 s demonstrator steps.
const NEW_WAIT_S: f64 = 1.0;

/// A project's first try's seed. Never one the template's Evaluation IR resolves, nor is any try
/// after it ([`free_seed`]): a person tunes the program until its try succeeds, and tuning it on
/// the held-out positions the policy is judged on would leak them (spec 13.3, as collection
/// seeds keep off them in packets M13/Z2 and Z4b).
pub const TRY_SEED_BASE: u64 = 1001;

// --- words ---------------------------------------------------------------------------------------

/// A block field, as the inspector edits it, in the file's own units: metres, degrees, seconds.
#[derive(Clone, Debug, PartialEq)]
pub enum Field {
    Target(Target),
    /// Clears `height`: a move has exactly one of the two.
    Above(f64),
    /// Clears `above`.
    Height(f64),
    Pitch(f64),
    Grip(Grip),
    Wait(f64),
}

/// A field's name, for the inspector and for a refusal that names one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldName {
    Target,
    Above,
    Height,
    Pitch,
    Grip,
    Wait,
}

impl FieldName {
    pub const ALL: [FieldName; 6] = [
        FieldName::Target,
        FieldName::Above,
        FieldName::Height,
        FieldName::Pitch,
        FieldName::Grip,
        FieldName::Wait,
    ];

    /// How the file spells it, which is how a [`ProgramError`] names it.
    pub fn file_key(self) -> &'static str {
        match self {
            FieldName::Target => "move",
            FieldName::Above => "above",
            FieldName::Height => "height",
            FieldName::Pitch => "pitch",
            FieldName::Grip => "grip",
            FieldName::Wait => "wait",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            FieldName::Target => "teach.field.target",
            FieldName::Above => "teach.field.above",
            FieldName::Height => "teach.field.height",
            FieldName::Pitch => "teach.field.pitch",
            FieldName::Grip => "teach.field.grip",
            FieldName::Wait => "teach.field.wait",
        }
    }

    /// The unit the inspector shows it in, and what a slider over it spans in that unit: the
    /// heights in centimetres (the file's metres x 100), the wrist in degrees, the wait in
    /// seconds. `None` for a choice.
    pub fn unit(self) -> Option<(&'static str, std::ops::RangeInclusive<f64>)> {
        match self {
            FieldName::Target | FieldName::Grip => None,
            FieldName::Above => Some(("teach.unit.cm", -5.0..=20.0)),
            FieldName::Height => Some(("teach.unit.cm", 0.0..=40.0)),
            FieldName::Pitch => Some(("teach.unit.deg", -90.0..=90.0)),
            FieldName::Wait => Some(("teach.unit.s", 0.2..=10.0)),
        }
    }
}

/// What a move block goes to, as the inspector offers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    Object,
    Place,
    Point,
}

impl TargetKind {
    pub const ALL: [TargetKind; 3] = [TargetKind::Object, TargetKind::Place, TargetKind::Point];

    pub fn of(target: &Target) -> Self {
        match target {
            Target::Named(name) if name == OBJECT => TargetKind::Object,
            Target::Named(_) => TargetKind::Place,
            Target::Point(_) => TargetKind::Point,
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            TargetKind::Object => "teach.target.object",
            TargetKind::Place => "teach.target.place",
            TargetKind::Point => "teach.target.point",
        }
    }
}

pub fn grip_key(grip: Grip) -> &'static str {
    match grip {
        Grip::Open => "teach.grip.open",
        Grip::Closed => "teach.grip.closed",
    }
}

/// ②'s buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    AddMove,
    AddGrip,
    Delete,
    Up,
    Down,
    /// Back to the template's program.
    Reset,
    Save,
    Try,
    /// The next seed: the object somewhere else.
    Reroll,
}

impl Action {
    pub const ALL: [Action; 9] = [
        Action::AddMove,
        Action::AddGrip,
        Action::Delete,
        Action::Up,
        Action::Down,
        Action::Reset,
        Action::Save,
        Action::Try,
        Action::Reroll,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Action::AddMove => "teach.action.add_move",
            Action::AddGrip => "teach.action.add_grip",
            Action::Delete => "teach.action.delete",
            Action::Up => "teach.action.up",
            Action::Down => "teach.action.down",
            Action::Reset => "teach.action.reset",
            Action::Save => "teach.action.save",
            Action::Try => "teach.action.try",
            Action::Reroll => "teach.action.reroll",
        }
    }

    pub fn hint_key(self) -> Option<&'static str> {
        match self {
            Action::Reset => Some("teach.action.reset.hint"),
            Action::Try => Some("teach.action.try.hint"),
            Action::Reroll => Some("teach.action.reroll.hint"),
            Action::AddMove
            | Action::AddGrip
            | Action::Delete
            | Action::Up
            | Action::Down
            | Action::Save => None,
        }
    }
}

/// A number the way a block says it: at most one decimal, no `-0`.
fn short(v: f64) -> String {
    format!("{}", (v * 10.0).round() / 10.0 + 0.0)
}

fn cm(metres: f64) -> String {
    short(metres * 100.0)
}

/// What a block calls its target: the template's `[outcome]` words for its object and its
/// target place when they are the ones named, the scene's own name otherwise, a point by its
/// centimetres.
fn target_name(lang: Lang, target: &Target, object: &str, outcome: Option<&OutcomeSpec>) -> String {
    let table = Strings::get(lang);
    match target {
        Target::Named(name) if name == OBJECT => match outcome {
            Some(o) if o.object == object => table.t(&o.object_name).to_owned(),
            _ => object.to_owned(),
        },
        Target::Named(name) => match outcome {
            Some(o) if o.target == *name => table.t(&o.target_name).to_owned(),
            _ => name.clone(),
        },
        Target::Point([x, y]) => fill(lang, "teach.target.xy", &[&cm(*x), &cm(*y)]),
    }
}

/// One block in plain words, from whatever fields it has: every shape a file can hold reads as
/// something, a malformed one included (its refusal is said beside it).
pub fn block_label(
    lang: Lang,
    block: &Block,
    object: &str,
    outcome: Option<&OutcomeSpec>,
) -> String {
    let mut parts = Vec::new();
    if let Some(target) = &block.target {
        let name = target_name(lang, target, object, outcome);
        parts.push(match (block.above, block.height) {
            (Some(a), _) if a < 0.0 => fill(lang, "teach.label.below", &[&name, &cm(-a)]),
            (Some(a), _) => fill(lang, "teach.label.above", &[&name, &cm(a)]),
            (None, Some(h)) => fill(lang, "teach.label.height", &[&name, &cm(h)]),
            (None, None) => fill(lang, "teach.label.to", &[&name]),
        });
        if let Some(pitch) = block.pitch {
            parts.push(fill(lang, "teach.label.wrist", &[&short(pitch)]));
        }
        let grip = t(lang, grip_key(block.grip));
        parts.push(fill(lang, "teach.label.gripper", &[grip]));
    } else {
        let action = match block.grip {
            Grip::Open => "teach.label.open",
            Grip::Closed => "teach.label.close",
        };
        parts.push(t(lang, action).to_owned());
        if let Some(wait) = block.wait {
            parts.push(fill(lang, "teach.label.wait", &[&short(wait)]));
        }
    }
    parts.join(t(lang, "teach.label.sep"))
}

/// The block a refusal names, counting from 0; `None` for one about the whole program, which
/// the screen says at the top.
pub fn error_block(error: &ProgramError) -> Option<usize> {
    match error {
        ProgramError::Parse(_)
        | ProgramError::Kind(_)
        | ProgramError::Robot(_)
        | ProgramError::Empty
        | ProgramError::UnknownObject(_)
        | ProgramError::Scene(_) => None,
        ProgramError::GripFirst => Some(0),
        ProgramError::BothHeights(i) | ProgramError::NoHeight(i) => Some(*i),
        ProgramError::Missing { block, .. }
        | ProgramError::Misplaced { block, .. }
        | ProgramError::UnknownPlace { block, .. }
        | ProgramError::Wait { block, .. } => Some(*block),
    }
}

/// A refusal in plain words ([`program_error_key`]) with its holes filled.
pub fn error_text(lang: Lang, error: &ProgramError) -> String {
    let field = |name: &str| {
        (FieldName::ALL.iter())
            .find(|f| f.file_key() == name)
            .map_or(name, |f| t(lang, f.key()))
            .to_owned()
    };
    let args = match error {
        ProgramError::Kind(_)
        | ProgramError::Empty
        | ProgramError::BothHeights(_)
        | ProgramError::NoHeight(_)
        | ProgramError::GripFirst => Vec::new(),
        ProgramError::Parse(why) => vec![why.clone()],
        ProgramError::Robot(robot) | ProgramError::UnknownObject(robot) => vec![robot.clone()],
        ProgramError::Missing { field: f, .. } | ProgramError::Misplaced { field: f, .. } => {
            vec![field(f)]
        }
        ProgramError::UnknownPlace { name, .. } => vec![name.clone()],
        ProgramError::Wait { step, .. } => vec![short(*step)],
        ProgramError::Scene(e) => vec![e.to_string()],
    };
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    fill(lang, program_error_key(error), &args)
}

// --- checks ------------------------------------------------------------------------------------

/// What the demonstrator would fail at in a run. Never a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Warning {
    /// For some object position the Task IR can draw, the arm cannot reach this move's target
    /// at any wrist angle.
    Unreachable,
    /// It could, at another wrist angle.
    Pitch,
    /// No block closes the gripper.
    NoClose,
    /// No block opens it after the first that closes it.
    NoOpen,
}

impl Warning {
    pub const ALL: [Warning; 4] = [
        Warning::Unreachable,
        Warning::Pitch,
        Warning::NoClose,
        Warning::NoOpen,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Warning::Unreachable => "teach.warn.unreachable",
            Warning::Pitch => "teach.warn.pitch",
            Warning::NoClose => "teach.warn.no_close",
            Warning::NoOpen => "teach.warn.no_open",
        }
    }
}

/// A program's refusal and its warnings; a `None` block is the whole program's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Check {
    pub error: Option<ProgramError>,
    pub warnings: Vec<(Option<usize>, Warning)>,
}

/// What is said beside one block, or at the top.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Notes {
    pub error: Option<String>,
    pub warnings: Vec<&'static str>,
}

/// One row of the block list.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub label: String,
    pub is_move: bool,
    pub notes: Notes,
}

/// What the checks need from the template's documents, read once when ② opens.
#[derive(Debug)]
struct World {
    scene: SceneDesc,
    links: Links,
    task: TaskIr,
    deploy: DeploymentIr,
    /// Control ticks per demonstrator step: the deployment's re-plan period.
    replan: u32,
    /// The scene's objects: a move's places, but for the program's own object.
    objects: Vec<String>,
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

impl World {
    fn read(template: &Template, root: &Path) -> Result<Self, String> {
        let scene_path = root.join(&template.scene);
        let scene = load_scene(&scene_path).map_err(|e| e.to_string())?;
        let links = Links::from_scene(&scene).map_err(|e| e.to_string())?;
        let deploy = root.join(&template.bundle.deployment);
        let deploy = deployment_from_toml(&read(&deploy)?).map_err(|e| e.to_string())?;
        let replan = es_env::replan_interval(deploy.rate)
            .map_err(|e| e.to_string())?
            .min(deploy.action.execute_chunk as u64);
        let task =
            task_from_toml(&read(&root.join(&template.bundle.task))?).map_err(|e| e.to_string())?;
        let objects = ScenePreview::open(&scene_path)?.contents().objects;
        Ok(Self {
            scene,
            links,
            task,
            deploy,
            replan: u32::try_from(replan).unwrap_or(u32::MAX),
            objects,
        })
    }

    /// The demonstrator `es` builds for `program` (`build_expert`, packet M14/Q2): the free
    /// joint of the body the program calls its object, `demo_cfg` paced to the deployment's
    /// re-plan period - or its refusal, by name.
    ///
    /// ponytail: a copy of `es`'s dozen lines, since `es` is a binary this crate cannot call.
    fn expert(&self, program: &Program) -> Result<&Joint, ProgramError> {
        let scene = &self.scene;
        let joint = (scene.joints.iter())
            .find(|j| {
                j.kind == JointKind::Free
                    && (scene.bodies.iter()).any(|b| b.id == j.body && b.name == program.object)
            })
            .ok_or_else(|| ProgramError::UnknownObject(program.object.clone()))?;
        let mut cfg = demo_cfg(joint.id);
        cfg.pace_to(&self.deploy, self.replan);
        ScriptedExpert::with_program(scene, cfg, program)?;
        Ok(joint)
    }
}

/// `qpos` values a joint takes: `MuJoCo`'s layout, in the scene's joint order.
fn width(kind: JointKind) -> usize {
    match kind {
        JointKind::Free => 7,
        JointKind::Ball => 4,
        JointKind::Hinge | JointKind::Slide => 1,
        JointKind::Fixed => 0,
    }
}

/// Where a distribution can land, as `(lo, hi)`.
///
/// ponytail: a normal draw has no corner; three standard deviations stand in for one.
fn span(dist: &Distribution) -> Option<(f64, f64)> {
    match dist {
        Distribution::Constant(v) => Some((*v, *v)),
        Distribution::Uniform { lo, hi } | Distribution::LogUniform { lo, hi } => Some((*lo, *hi)),
        Distribution::Normal { mean, std } => Some((mean - 3.0 * std, mean + 3.0 * std)),
        Distribution::Choice(values) => {
            let lo = values.iter().copied().reduce(f64::min)?;
            Some((lo, values.iter().copied().reduce(f64::max)?))
        }
    }
}

/// The corners of the box of positions the object can start at: per axis, what the Task IR's
/// nodes on its free joint's `qpos[i]` draw - a `Randomization` over a `ResetState`, since it
/// writes last - or where the scene puts it when neither does.
///
/// ponytail: `qpos[i]` is read as the scene's joint order laid out `MuJoCo`'s way, which is how
/// the importers order joints; `the_object_is_drawn_from_the_task_irs_box` pins it for the demo.
fn corners(task: &TaskIr, scene: &SceneDesc, joint: &Joint) -> Vec<Vec3> {
    let start: usize = (scene.joints.iter())
        .take_while(|j| j.id != joint.id)
        .map(|j| width(j.kind))
        .sum();
    let home = (scene.bodies.iter())
        .find(|b| b.id == joint.body)
        .map_or([0.0; 3], |b| {
            [b.pose.position.x, b.pose.position.y, b.pose.position.z]
        });
    let axis = |a: usize| {
        let slot = format!("qpos[{}]", start + a);
        let drawn = |randomization: bool| {
            task.graph.nodes.values().find_map(|node| match node {
                TaskNode::Randomization { target, dist, .. }
                    if randomization && *target == slot =>
                {
                    span(dist)
                }
                TaskNode::ResetState { target, dist, .. } if !randomization && *target == slot => {
                    span(dist)
                }
                _ => None,
            })
        };
        let (lo, hi) = drawn(true)
            .or_else(|| drawn(false))
            .unwrap_or((home[a], home[a]));
        [lo, hi]
    };
    let (xs, ys, zs) = (axis(0), axis(1), axis(2));
    let mut out = Vec::with_capacity(8);
    for x in xs {
        for y in ys {
            for z in zs {
                out.push(Vec3::new(x, y, z));
            }
        }
    }
    out
}

/// The refusal first, then the warnings: reach only for a program the demonstrator accepts (it
/// is what places a waypoint), the gripper's always.
fn check(program: &Program, world: Option<&World>) -> Check {
    let built = world.map(|w| (w, w.expert(program)));
    let error = (program.validate().err())
        .or_else(|| built.as_ref().and_then(|(_, b)| b.as_ref().err().cloned()));
    let mut warnings = Vec::new();
    if let (None, Some((w, Ok(joint)))) = (&error, &built) {
        let corners = corners(&w.task, &w.scene, joint);
        for (i, (block, point)) in (program.blocks.iter())
            .zip(program.waypoints(&w.scene).unwrap_or_default())
            .enumerate()
        {
            if block.target.is_none() {
                continue;
            }
            let missed: Vec<Vec3> = (corners.iter())
                .map(|&c| point.tool(c))
                .filter(|&at| so101_ik(&w.links, at, point.pitch()).is_none())
                .collect();
            if missed.is_empty() {
                continue;
            }
            let other_pitch = missed.iter().all(|&at| {
                (-90..=90).any(|d| so101_ik(&w.links, at, f64::from(d) * DEG_TO_RAD).is_some())
            });
            let warning = if other_pitch {
                Warning::Pitch
            } else {
                Warning::Unreachable
            };
            warnings.push((Some(i), warning));
        }
    }
    match (program.blocks.iter()).position(|b| b.grip == Grip::Closed) {
        None => warnings.push((None, Warning::NoClose)),
        Some(c) if !program.blocks[c + 1..].iter().any(|b| b.grip == Grip::Open) => {
            warnings.push((None, Warning::NoOpen));
        }
        Some(_) => {}
    }
    Check { error, warnings }
}

// --- the model ---------------------------------------------------------------------------------

/// Why ② is read-only for a project.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TeachError {
    /// A project made before packet M14/Q3: its runs use the template's built-in demonstrator.
    NoFile,
    /// `teach.toml` is there but is not a demonstration program's shape at all.
    Unreadable(String),
}

impl TeachError {
    pub fn text(&self, lang: Lang) -> String {
        match self {
            TeachError::NoFile => t(lang, "teach.readonly.no_file").to_owned(),
            TeachError::Unreadable(why) => fill(lang, "teach.readonly.unreadable", &[why]),
        }
    }
}

/// Why "try once" does nothing now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TryRefused {
    /// A run (or another try) is going ([`may_start`]).
    Busy,
    /// The demonstrator refuses the program; its refusal is shown where it points.
    Invalid,
    /// Something on disk: the template's evaluation, or writing the try.
    Broken(String),
}

impl TryRefused {
    pub fn text(&self, lang: Lang) -> String {
        match self {
            TryRefused::Busy => t(lang, "teach.try.busy").to_owned(),
            TryRefused::Invalid => t(lang, "teach.try.invalid").to_owned(),
            TryRefused::Broken(why) => why.clone(),
        }
    }
}

/// A try just written: where it goes, and `es`'s argv (no program name), to run with the
/// repository root as its working directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TryStart {
    pub dir: PathBuf,
    pub argv: Vec<String>,
}

/// ② of one project.
#[derive(Debug)]
pub struct Teach {
    project: Project,
    /// The template's program, what [`Teach::reset`] goes back to.
    origin: Option<PathBuf>,
    outcome: Option<OutcomeSpec>,
    world: Result<World, String>,
    /// The Evaluation IR a try derives its own from, and the scene it runs on.
    trial: Result<(EvaluationIr, PathBuf), String>,
    program: Program,
    selected: Option<usize>,
    dirty: bool,
    check: Check,
    /// Every seed the template's Evaluation IR resolves: never a try's.
    judged: Vec<u64>,
    /// The seed the next try runs: the latest try's, else [`TRY_SEED_BASE`] - past any judged.
    seed: u64,
}

impl Teach {
    /// The project's `teach.toml`, checked against `template`'s documents under `root`. A file
    /// the demonstrator would refuse still opens: its refusal is shown and it can be fixed.
    pub fn open(project: &Project, template: &Template, root: &Path) -> Result<Self, TeachError> {
        let text = match std::fs::read_to_string(project.teach()) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(TeachError::NoFile),
            Err(e) => return Err(TeachError::Unreadable(e.to_string())),
        };
        let program: Program =
            parse_toml(&text).map_err(|e| TeachError::Unreadable(e.to_string()))?;
        let trial = (|| {
            let cycle_path = root.join(&template.cycle);
            let cycle = Cycle::parse(&read(&cycle_path)?).map_err(|e| e.to_string())?;
            let config = root.join(&cycle.eval.config);
            let ir = evaluation_from_toml(&read(&config)?).map_err(|e| e.to_string())?;
            Ok((ir, root.join(&cycle.scene)))
        })();
        let judged = trial
            .as_ref()
            .map_or_else(|_| Vec::new(), |(ir, _)| evaluation_seeds(ir));
        let latest = project.tries().last().and_then(|r| try_seed(&r.path));
        let seed = free_seed(latest.unwrap_or(TRY_SEED_BASE), &judged);
        let mut teach = Self {
            project: project.clone(),
            origin: template.teach.as_ref().map(|p| root.join(p)),
            outcome: template.outcome.clone(),
            world: World::read(template, root),
            trial,
            program,
            selected: None,
            dirty: false,
            check: Check::default(),
            judged,
            seed,
        };
        teach.recheck();
        Ok(teach)
    }

    /// Read-only: every change goes through an edit, which re-checks it.
    pub fn program(&self) -> &Program {
        &self.program
    }

    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn dirty(&self) -> bool {
        self.dirty
    }

    pub fn check(&self) -> &Check {
        &self.check
    }

    /// The places a move may name ([`TargetKind::Place`]): the scene's objects but the one
    /// the program picks up; none when the scene did not load.
    pub fn places(&self) -> Vec<&str> {
        let objects = self.world.as_ref().map_or(&[][..], |w| &w.objects);
        (objects.iter().map(String::as_str))
            .filter(|o| *o != self.program.object)
            .collect()
    }

    /// Where `target` is on the floor, `[x, y]` in metres: a point's own, a place's
    /// ([`place_point`]), the object's where the scene puts it - where a move switched to a point
    /// starts. `None` when the scene did not load or has no such thing.
    pub fn point_of(&self, target: &Target) -> Option<[f64; 2]> {
        let scene = &self.world.as_ref().ok()?.scene;
        match target {
            Target::Point(p) => Some(*p),
            Target::Named(name) if name == OBJECT => (scene.bodies.iter())
                .find(|b| b.name == self.program.object)
                .map(|b| [b.pose.position.x, b.pose.position.y]),
            Target::Named(name) => place_point(scene, name).map(|p| [p.x, p.y]),
        }
    }

    /// A grip block's wait is a whole number of these seconds: one demonstrator step
    /// (`ExpertCfg::pace_to`: the re-plan period at the control rate).
    pub fn wait_step(&self) -> Option<f64> {
        let w = self.world.as_ref().ok()?;
        Some(f64::from(w.replan.max(1)) / w.deploy.rate.control.as_hz_f64())
    }

    fn recheck(&mut self) {
        self.check = check(&self.program, self.world.as_ref().ok());
    }

    fn changed(&mut self) {
        self.dirty = true;
        self.recheck();
    }

    /// Every block in plain words, with what is said beside it.
    pub fn rows(&self, lang: Lang) -> Vec<Row> {
        let outcome = self.outcome.as_ref();
        (self.program.blocks.iter().enumerate())
            .map(|(i, b)| Row {
                label: block_label(lang, b, &self.program.object, outcome),
                is_move: b.target.is_some(),
                notes: self.notes(lang, Some(i)),
            })
            .collect()
    }

    /// What is said at the top: about the whole program, not one block.
    pub fn top(&self, lang: Lang) -> Notes {
        self.notes(lang, None)
    }

    fn notes(&self, lang: Lang, block: Option<usize>) -> Notes {
        let error = self.check.error.as_ref();
        Notes {
            error: error
                .filter(|e| error_block(e) == block)
                .map(|e| error_text(lang, e)),
            warnings: (self.check.warnings.iter())
                .filter(|(at, _)| *at == block)
                .map(|(_, w)| t(lang, w.key()))
                .collect(),
        }
    }

    /// Whether a button does something now. [`Action::Try`] also waits for no run to be going
    /// ([`Teach::refusal`]).
    pub fn can(&self, action: Action) -> bool {
        let n = self.program.blocks.len();
        match action {
            Action::AddMove | Action::AddGrip | Action::Reset | Action::Reroll => true,
            Action::Delete => self.selected.is_some_and(|i| i < n),
            Action::Up => self.selected.is_some_and(|i| i > 0 && i < n),
            Action::Down => self.selected.is_some_and(|i| i + 1 < n),
            Action::Save => self.dirty,
            Action::Try => self.check.error.is_none(),
        }
    }

    // --- edits: each re-checks -------------------------------------------------------------------

    pub fn select(&mut self, block: Option<usize>) {
        self.selected = block.filter(|&i| i < self.program.blocks.len());
    }

    /// Where an added block goes: after the selected one, else at the end.
    fn insert_at(&self) -> usize {
        let n = self.program.blocks.len();
        self.selected.map_or(n, |i| (i + 1).min(n))
    }

    /// The gripper as the block before `at` leaves it.
    fn grip_before(&self, at: usize) -> Grip {
        (at.checked_sub(1))
            .and_then(|i| self.program.blocks.get(i))
            .map_or(Grip::Open, |b| b.grip)
    }

    fn insert(&mut self, at: usize, block: Block) {
        self.program.blocks.insert(at, block);
        self.selected = Some(at);
        self.changed();
    }

    /// A move above the object, the gripper as it was.
    pub fn add_move(&mut self) {
        let at = self.insert_at();
        let block = Block {
            target: Some(Target::Named(OBJECT.to_owned())),
            above: Some(NEW_ABOVE_M),
            height: None,
            pitch: Some(NEW_PITCH_DEG),
            grip: self.grip_before(at),
            wait: None,
        };
        self.insert(at, block);
    }

    /// A grip block that turns the gripper the other way.
    pub fn add_grip(&mut self) {
        let at = self.insert_at();
        let grip = match self.grip_before(at) {
            Grip::Open => Grip::Closed,
            Grip::Closed => Grip::Open,
        };
        let block = Block {
            target: None,
            above: None,
            height: None,
            pitch: None,
            grip,
            wait: Some(NEW_WAIT_S),
        };
        self.insert(at, block);
    }

    /// The selected block; the one after it (else the new last) is selected.
    pub fn delete(&mut self) -> bool {
        let Some(i) = self.selected.filter(|_| self.can(Action::Delete)) else {
            return false;
        };
        self.program.blocks.remove(i);
        let n = self.program.blocks.len();
        self.selected = (n > 0).then(|| i.min(n - 1));
        self.changed();
        true
    }

    pub fn move_up(&mut self) -> bool {
        let Some(i) = self.selected.filter(|_| self.can(Action::Up)) else {
            return false;
        };
        self.program.blocks.swap(i - 1, i);
        self.selected = Some(i - 1);
        self.changed();
        true
    }

    pub fn move_down(&mut self) -> bool {
        let Some(i) = self.selected.filter(|_| self.can(Action::Down)) else {
            return false;
        };
        self.program.blocks.swap(i, i + 1);
        self.selected = Some(i + 1);
        self.changed();
        true
    }

    /// One field of the selected block. `false`, and nothing marked changed, when there is no
    /// selection or the field already holds the value.
    pub fn edit(&mut self, field: Field) -> bool {
        let Some(block) = self.selected.and_then(|i| self.program.blocks.get_mut(i)) else {
            return false;
        };
        let before = block.clone();
        match field {
            Field::Target(target) => block.target = Some(target),
            Field::Above(v) => (block.above, block.height) = (Some(v), None),
            Field::Height(v) => (block.height, block.above) = (Some(v), None),
            Field::Pitch(v) => block.pitch = Some(v),
            Field::Grip(g) => block.grip = g,
            Field::Wait(v) => block.wait = Some(v),
        }
        if *block == before {
            return false;
        }
        self.changed();
        true
    }

    /// Back to the template's program (the built-in one, for a template that names none).
    /// Unsaved, like any edit.
    pub fn reset(&mut self) -> Result<(), String> {
        self.program = match &self.origin {
            Some(path) => parse_toml(&read(path)?).map_err(|e| e.to_string())?,
            None => Program::builtin(),
        };
        self.selected = None;
        self.changed();
        Ok(())
    }

    /// Writes `teach.toml` whole or not at all: a sibling file, renamed over it. A program the
    /// demonstrator refuses is saved too - it is the person's work in progress.
    pub fn save(&mut self) -> Result<(), String> {
        let path = self.project.teach();
        let text = write_toml(&self.program).map_err(|e| e.to_string())?;
        let partial = path.with_extension("toml.partial");
        std::fs::write(&partial, text)
            .and_then(|()| std::fs::rename(&partial, &path))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        self.dirty = false;
        Ok(())
    }

    // --- trying it -------------------------------------------------------------------------------

    /// 🎲: the object somewhere else.
    pub fn next_seed(&mut self) {
        self.seed = free_seed(self.seed.saturating_add(1), &self.judged);
    }

    /// Why a try would do nothing now, or `None`.
    pub fn refusal(
        &self,
        pid: Option<u32>,
        queued: bool,
        phases: &[PhaseState; 5],
    ) -> Option<TryRefused> {
        if !may_start(pid, queued, phases) {
            return Some(TryRefused::Busy);
        }
        if !self.can(Action::Try) {
            return Some(TryRefused::Invalid);
        }
        self.trial
            .as_ref()
            .err()
            .map(|e| TryRefused::Broken(e.clone()))
    }

    /// Saves unsaved edits (a try runs what is shown), writes `try/<n>/evaluation.toml` and
    /// returns `es eval run`'s argv for it.
    pub fn start_try(
        &mut self,
        pid: Option<u32>,
        queued: bool,
        phases: &[PhaseState; 5],
    ) -> Result<TryStart, TryRefused> {
        if let Some(refused) = self.refusal(pid, queued, phases) {
            return Err(refused);
        }
        let (ir, scene) = self.trial.clone().map_err(TryRefused::Broken)?;
        let ir = try_evaluation(&ir, self.seed).map_err(TryRefused::Broken)?;
        let text = evaluation_to_toml(&ir).map_err(|e| TryRefused::Broken(e.to_string()))?;
        if self.dirty {
            self.save().map_err(TryRefused::Broken)?;
        }
        let dir = self.project.next_try_dir();
        let config = dir.join("evaluation.toml");
        std::fs::create_dir_all(&dir)
            .and_then(|()| std::fs::write(&config, text))
            .map_err(|e| TryRefused::Broken(format!("{}: {e}", dir.display())))?;
        let arg = |p: &Path| p.display().to_string();
        let mut argv = vec!["eval".to_owned(), "run".to_owned()];
        for (flag, value) in [
            ("--config", arg(&config)),
            ("--expert", arg(&self.project.teach())),
            ("--policy", arg(&self.project.bundle())),
            ("--scene", arg(&scene)),
            ("--out", arg(&dir)),
            ("--jobs", "1".to_owned()),
            ("--frames", arg(&dir.join("frames"))),
        ] {
            argv.extend([flag.to_owned(), value]);
        }
        Ok(TryStart { dir, argv })
    }

    /// A try's one attempt, once `es eval run` has written it: whether it succeeded, the cell
    /// to play, and - for one that ran out of time - where the object went.
    pub fn attempt(&self, dir: &Path) -> Option<Tile> {
        let rows = read_episodes(dir).ok()??;
        let outcomes = match (&self.outcome, &self.trial, RunDir::open(dir)) {
            (Some(spec), Ok((_, scene)), Ok(run)) => outcome::outcomes(scene, spec, &rows, &run),
            _ => Outcomes::new(),
        };
        tiles(&rows, TileFilter::All, &outcomes).into_iter().next()
    }

    /// The newest try, and its attempt once there is one.
    pub fn latest_try(&self) -> Option<(PathBuf, Option<Tile>)> {
        let dir = self.project.tries().pop()?.path;
        let attempt = self.attempt(&dir);
        Some((dir, attempt))
    }

    /// What a try came to, in a line: still going, succeeded, failed and why, or never ran to
    /// the end.
    pub fn verdict(&self, lang: Lang, attempt: Option<&Tile>, running: bool) -> String {
        match (attempt, running) {
            (_, true) => t(lang, "teach.try.running").to_owned(),
            (None, false) => t(lang, "teach.try.no_result").to_owned(),
            (Some(tile), false) if tile.success => t(lang, "teach.try.success").to_owned(),
            (Some(tile), false) => match tile.cause {
                None => t(lang, "teach.try.failed").to_owned(),
                Some(cause) => {
                    let table = Strings::get(lang);
                    let names: Vec<&str> = (self.outcome.iter())
                        .flat_map(|o| [table.t(&o.object_name), table.t(&o.target_name)])
                        .collect();
                    let why = fill(lang, cause_key(cause), &names);
                    fill(lang, "teach.try.failed_because", &[&why])
                }
            },
        }
    }
}

/// A try's Evaluation IR: the template's first suite alone, no acceptance (as a preview's), on
/// the one explicit `seed`.
pub fn try_evaluation(ir: &EvaluationIr, seed: u64) -> Result<EvaluationIr, String> {
    let one = PreviewRef {
        episodes: 1,
        suite: None,
        frames: true,
    };
    let mut derived = preview_evaluation(ir, &one).map_err(|e| e.to_string())?;
    derived.episodes = EpisodeBatch {
        n_episodes: 1,
        seeds: SeedPlan::Explicit(vec![seed]),
    };
    Ok(derived)
}

/// The first seed at or after `from` that is none of `judged` ([`fresh_seed`] with one attempt).
fn free_seed(from: u64, judged: &[u64]) -> u64 {
    fresh_seed(&[(from, 0)], judged, 1)
}

/// The seed a try ran, from the Evaluation IR written into it.
fn try_seed(dir: &Path) -> Option<u64> {
    let ir = evaluation_from_toml(&read(&dir.join("evaluation.toml")).ok()?).ok()?;
    evaluation_seeds(&ir).first().copied()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    use es_env::program::SO101_PICK_PLACE;
    use es_env::EnvError;
    use es_eval::episodes::{write_episodes, EpisodeRow};

    use crate::model::outcome::Outcome;
    use crate::model::project::tests::{cube, hint, repo, scratch_project};
    use crate::model::results::Cause;
    use PhaseState::{Done, Locked, NotStarted, Running};

    fn fresh() -> [PhaseState; 5] {
        [Done, Done, NotStarted, Locked, Locked]
    }

    fn open(name: &str) -> (Project, Teach) {
        let project = scratch_project(name);
        let teach = Teach::open(&project, &cube(), &repo()).expect("a template project");
        (project, teach)
    }

    fn labels(teach: &Teach, lang: Lang) -> Vec<String> {
        teach.rows(lang).into_iter().map(|r| r.label).collect()
    }

    fn after<'a>(argv: &'a [String], flag: &str) -> &'a str {
        &argv.windows(2).find(|w| w[0] == flag).expect(flag)[1]
    }

    /// The template's program opens as its seven blocks in plain words, and the demonstrator
    /// accepts it with nothing to warn about: every cube the Task IR draws is in reach.
    #[test]
    fn the_builtin_program_reads_in_plain_words_and_passes() {
        let (project, teach) = open("q3-builtin");
        assert_eq!(teach.program, Program::builtin());
        assert_eq!(teach.check(), &Check::default(), "no refusal, no warning");
        assert!(!teach.dirty());
        assert_eq!(
            labels(&teach, Lang::En),
            [
                "Move above [cube] by 4.5 cm · wrist -85° · gripper open",
                "Move below [cube] by 0.5 cm · wrist -85° · gripper open",
                "Close the gripper · wait 5 s",
                "Move over [cube] to 14 cm above the floor · wrist -45° · gripper closed",
                "Move over [bin] to 14 cm above the floor · wrist -45° · gripper closed",
                "Move over [bin] to 6 cm above the floor · wrist -85° · gripper closed",
                "Open the gripper · wait 5 s",
            ]
        );
        assert_ne!(labels(&teach, Lang::Ko), labels(&teach, Lang::En));
        assert_eq!(teach.places(), ["bin"]);
        assert_eq!(teach.wait_step(), Some(0.2));
        // A target switched to a point starts where it was: the bin's floor exactly.
        let bin = Target::Named("bin".into());
        assert_eq!(teach.point_of(&bin), Some([0.14, -0.1]));
        assert!(teach.point_of(&Target::Named(OBJECT.into())).is_some());
        assert_eq!(teach.point_of(&Target::Point([0.2, 0.0])), Some([0.2, 0.0]));
        assert_eq!(teach.point_of(&Target::Named("shelf".into())), None);
        std::fs::remove_dir_all(&project.root).ok();
    }

    /// The cube-pose card teaches with the same program, and its try derives from its own
    /// Evaluation IR (`evaluation-augmented.toml`).
    #[test]
    fn the_hint_card_teaches_the_same_program() {
        let root = std::env::temp_dir().join(format!("es-q3-hint-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let project = Project::create(&root, "hint", &hint(), &repo()).expect("a project");
        let mut teach = Teach::open(&project, &hint(), &repo()).expect("its program");
        assert_eq!(teach.check(), &Check::default());
        let start = teach.start_try(None, false, &fresh()).expect("a try");
        let ir = evaluation_from_toml(&read(&start.dir.join("evaluation.toml")).unwrap());
        let ir = ir.expect("parses");
        let own = repo().join("tests/fixtures/visible-learning/evaluation-augmented.toml");
        let own = evaluation_from_toml(&read(&own).unwrap()).unwrap();
        assert_eq!((&ir.task, &ir.observation), (&own.task, &own.observation));
        assert!(ir.validate().is_empty(), "{:?}", ir.validate());
        std::fs::remove_dir_all(&root).ok();
    }

    /// Spec 13.3: for both cube cards, no try - the first, 🎲's next ones, one re-opened after
    /// a try made on a judged seed - runs a seed the card's Evaluation IR judges the policy on,
    /// and each is still one nominal attempt with no acceptance.
    #[test]
    fn a_try_never_draws_a_judged_seed() {
        assert_eq!(free_seed(1001, &[1001, 1002]), 1003, "steps past");
        assert_eq!(free_seed(101, &(101..=116).collect::<Vec<_>>()), 117);
        for template in [cube(), hint()] {
            let root = std::env::temp_dir().join(format!(
                "es-q3-judged-{}-{}",
                template.id,
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let project = Project::create(&root, "judged", &template, &repo()).unwrap();
            let cycle = Cycle::parse(&read(&repo().join(&template.cycle)).unwrap()).unwrap();
            let own = read(&repo().join(&cycle.eval.config)).unwrap();
            let judged = evaluation_seeds(&evaluation_from_toml(&own).unwrap());
            assert!(judged.contains(&101), "{}: {judged:?}", template.id);
            let mut teach = Teach::open(&project, &template, &repo()).unwrap();
            for _ in 0..3 {
                let start = teach.start_try(None, false, &fresh()).expect("a try");
                let text = read(&start.dir.join("evaluation.toml")).unwrap();
                let ir = evaluation_from_toml(&text).unwrap();
                let SeedPlan::Explicit(seeds) = &ir.episodes.seeds else {
                    panic!("{:?}", ir.episodes);
                };
                assert_eq!((ir.episodes.n_episodes, seeds.len()), (1, 1));
                assert!(!judged.contains(&seeds[0]), "{}: {seeds:?}", template.id);
                assert_eq!(ir.suites.len(), 1);
                assert_eq!(ir.suites[0].name, "nominal");
                assert!(ir.acceptance.is_empty());
                teach.next_seed();
            }
            // A try written on a judged seed (before this rule) is gone on from, past them.
            let old = try_evaluation(&evaluation_from_toml(&own).unwrap(), 101).unwrap();
            let dir = project.next_try_dir();
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("evaluation.toml"),
                evaluation_to_toml(&old).unwrap(),
            )
            .unwrap();
            let reopened = Teach::open(&project, &template, &repo()).unwrap();
            assert!(!judged.contains(&reopened.seed), "{}", reopened.seed);
            std::fs::remove_dir_all(&root).ok();
        }
    }

    /// The object is drawn from the Task IR's box: `qpos[6]` (x) and `qpos[7]` (y) of the
    /// cube's free joint by `Randomization`, `qpos[8]` (z) by `ResetState`.
    #[test]
    fn the_object_is_drawn_from_the_task_irs_box() {
        let world = World::read(&cube(), &repo()).expect("the demo's documents");
        let joint = world
            .expert(&Program::builtin())
            .expect("the cube's free joint");
        let mut corners: Vec<(f64, f64, f64)> = (corners(&world.task, &world.scene, joint).iter())
            .map(|c| (c.x, c.y, c.z))
            .collect();
        corners.dedup();
        assert_eq!(
            corners,
            [
                (0.21, -0.03, 0.02),
                (0.21, 0.05, 0.02),
                (0.27, -0.03, 0.02),
                (0.27, 0.05, 0.02),
            ]
        );
    }

    /// Every shape a block can have - each target, each height, with and without a wrist, each
    /// gripper; a grip block with and without its wait - reads as words in both languages, with
    /// no hole left and no key showing, and no two shapes alike.
    #[test]
    fn every_block_shape_has_words_in_both_languages() {
        let outcome = cube().outcome;
        let targets = [
            Target::Named(OBJECT.into()),
            Target::Named("bin".into()),
            Target::Named("shelf".into()),
            Target::Point([0.2, -0.1]),
        ];
        let heights = [
            (Some(0.045), None),
            (Some(-0.005), None),
            (None, Some(0.14)),
            (None, None),
        ];
        let mut blocks = Vec::new();
        for grip in [Grip::Open, Grip::Closed] {
            for target in &targets {
                for (above, height) in heights {
                    for pitch in [Some(-85.0), None] {
                        blocks.push(Block {
                            target: Some(target.clone()),
                            above,
                            height,
                            pitch,
                            grip,
                            wait: None,
                        });
                    }
                }
            }
            for wait in [Some(5.0), None] {
                blocks.push(Block {
                    target: None,
                    above: None,
                    height: None,
                    pitch: None,
                    grip,
                    wait,
                });
            }
        }
        for lang in Lang::ALL {
            let mut seen = std::collections::BTreeSet::new();
            for block in &blocks {
                let label = block_label(lang, block, "cube", outcome.as_ref());
                assert!(!label.is_empty() && !label.contains("{}"), "{label}");
                assert!(!label.contains("teach."), "a key shows: {label}");
                seen.insert(label);
            }
            assert_eq!(seen.len(), blocks.len(), "{lang:?}: two shapes read alike");
        }
        let point = Block {
            target: Some(Target::Point([0.2, -0.1])),
            above: None,
            height: Some(0.14),
            pitch: Some(-85.0),
            grip: Grip::Open,
            wait: None,
        };
        assert_eq!(
            block_label(Lang::En, &point, "cube", outcome.as_ref()),
            "Move over [the point (20 cm, -10 cm)] to 14 cm above the floor · wrist -85° · \
             gripper open"
        );
        let shelf = Block {
            target: Some(Target::Named("shelf".into())),
            ..point
        };
        let label = block_label(Lang::En, &shelf, "cube", outcome.as_ref());
        assert!(label.starts_with("Move over [shelf]"), "{label}");
    }

    /// Add, delete, up, down and one field at a time; each marks the program changed and
    /// checks it again. Saving writes the project's file whole and never the template's.
    #[test]
    fn each_edit_operation() {
        let (project, mut teach) = open("q3-edit");
        let n = teach.program.blocks.len();
        assert!(!teach.edit(Field::Pitch(-80.0)), "nothing selected");
        assert!(!teach.can(Action::Delete) && !teach.can(Action::Save));

        // A move after the close keeps the gripper closed; a grip after it opens it.
        teach.select(Some(2));
        teach.add_move();
        assert_eq!(
            (teach.selected, teach.program.blocks.len()),
            (Some(3), n + 1)
        );
        assert_eq!(teach.program.blocks[3].grip, Grip::Closed);
        assert!(teach.program.blocks[3].target.is_some());
        assert!(teach.dirty() && teach.can(Action::Save));
        teach.add_grip();
        assert_eq!(teach.selected, Some(4));
        let grip = &teach.program.blocks[4];
        assert_eq!((grip.target.is_none(), grip.grip), (true, Grip::Open));
        assert!(teach.check().error.is_none(), "{:?}", teach.check());

        assert!(teach.move_up());
        assert_eq!(teach.selected, Some(3));
        assert!(teach.program.blocks[3].target.is_none());
        assert!(teach.move_down());
        assert_eq!(teach.selected, Some(4));
        assert!(teach.delete());
        teach.select(Some(3));
        assert!(teach.delete());
        assert_eq!(teach.program, Program::builtin(), "both added blocks gone");
        assert_eq!(teach.selected, Some(3), "the block that took its place");

        // `above` and `height` replace each other; the same value is no edit.
        assert!(teach.edit(Field::Above(0.06)));
        let lift = &teach.program.blocks[3];
        assert_eq!((lift.above, lift.height), (Some(0.06), None));
        assert!(!teach.edit(Field::Above(0.06)));
        assert!(teach.edit(Field::Height(0.14)));
        assert_eq!(teach.program, Program::builtin());
        assert!(teach.edit(Field::Pitch(-50.0)));
        assert!(teach.edit(Field::Target(Target::Named("bin".into()))));
        teach.select(Some(2));
        assert!(teach.edit(Field::Wait(1.0)));
        assert!(teach.edit(Field::Grip(Grip::Open)));

        teach.select(Some(0));
        assert!(!teach.can(Action::Up) && !teach.move_up());
        teach.select(Some(n - 1));
        assert!(!teach.can(Action::Down) && !teach.move_down());
        teach.select(Some(n));
        assert_eq!(teach.selected, None, "past the end selects nothing");

        let edited = teach.program.clone();
        teach.save().expect("saves");
        assert!(!teach.dirty());
        let partial = std::fs::read_dir(&project.root)
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().ends_with(".partial"));
        assert!(!partial, "the sibling file is renamed over the program");
        let again = Teach::open(&project, &cube(), &repo()).expect("reopens");
        assert_eq!(again.program, edited);
        let template = repo().join(cube().teach.unwrap());
        assert_eq!(std::fs::read_to_string(template).unwrap(), SO101_PICK_PLACE);

        teach.reset().expect("the template's program");
        assert_eq!(teach.program, Program::builtin());
        assert!(teach.dirty());
        std::fs::remove_dir_all(&project.root).ok();
    }

    /// Each check on a crafted program, where it is shown, and that none stops saving.
    #[test]
    fn each_validation_on_a_crafted_program() {
        let (project, mut teach) = open("q3-check");

        // 50 cm above the cube is out of reach wherever it is.
        teach.select(Some(0));
        teach.edit(Field::Above(0.5));
        assert_eq!(
            teach.check(),
            &Check {
                error: None,
                warnings: vec![(Some(0), Warning::Unreachable)]
            }
        );
        let rows = teach.rows(Lang::En);
        assert_eq!(
            rows[0].notes.warnings,
            [t(Lang::En, Warning::Unreachable.key())]
        );
        assert_eq!(rows[1].notes, Notes::default());
        assert_eq!(teach.top(Lang::En), Notes::default());
        assert!(teach.can(Action::Try), "a warning does not stop a try");
        teach.reset().unwrap();

        // Straight down at the carry height: SO-101 cannot hold it there, -45 degrees can.
        teach.select(Some(3));
        teach.edit(Field::Pitch(-90.0));
        assert_eq!(teach.check().warnings, [(Some(3), Warning::Pitch)]);
        teach.reset().unwrap();

        // No block closes; then none opens after the close.
        for i in [2, 3, 4, 5] {
            teach.select(Some(i));
            teach.edit(Field::Grip(Grip::Open));
        }
        assert_eq!(teach.check().warnings, [(None, Warning::NoClose)]);
        assert_eq!(
            teach.top(Lang::En).warnings,
            [t(Lang::En, Warning::NoClose.key())]
        );
        teach.reset().unwrap();
        teach.select(Some(6));
        teach.delete();
        assert_eq!(teach.check().warnings, [(None, Warning::NoOpen)]);
        teach.reset().unwrap();

        // A wait that is not a whole number of 0.2 s steps: the demonstrator's own refusal.
        teach.select(Some(2));
        teach.edit(Field::Wait(0.3));
        let Some(ProgramError::Wait { block: 2, .. }) = teach.check().error else {
            panic!("{:?}", teach.check());
        };
        let said = teach.rows(Lang::En)[2].notes.error.clone();
        assert!(said.as_ref().is_some_and(|s| s.contains("0.2")), "{said:?}");
        teach.reset().unwrap();

        // A place the scene does not have.
        teach.select(Some(4));
        teach.edit(Field::Target(Target::Named("shelf".into())));
        assert_eq!(
            teach.check().error,
            Some(ProgramError::UnknownPlace {
                block: 4,
                name: "shelf".into()
            })
        );
        teach.reset().unwrap();

        // A grip first: refused at parse, shown on block 1, not tryable, still saved and
        // opened again as it was.
        teach.select(Some(0));
        teach.delete();
        teach.delete();
        assert_eq!(teach.check().error, Some(ProgramError::GripFirst));
        assert!(teach.rows(Lang::En)[0].notes.error.is_some());
        assert!(teach.top(Lang::En).error.is_none());
        assert!(!teach.can(Action::Try));
        assert_eq!(
            teach.start_try(None, false, &fresh()),
            Err(TryRefused::Invalid)
        );
        teach.save().expect("a refused program is saved");
        let mut again = Teach::open(&project, &cube(), &repo()).expect("and opens");
        assert_eq!(again.check().error, Some(ProgramError::GripFirst));

        // An empty program is said at the top.
        while !again.program.blocks.is_empty() {
            again.select(Some(0));
            again.delete();
        }
        assert_eq!(again.check().error, Some(ProgramError::Empty));
        assert!(again.top(Lang::En).error.is_some());

        // An object the scene has no free body of: `es`'s own refusal, said at the top.
        let ball = SO101_PICK_PLACE.replace("object = \"cube\"", "object = \"ball\"");
        std::fs::write(project.teach(), ball).unwrap();
        let ball = Teach::open(&project, &cube(), &repo()).expect("opens");
        assert_eq!(
            ball.check().error,
            Some(ProgramError::UnknownObject("ball".into()))
        );
        assert!(ball
            .top(Lang::En)
            .error
            .is_some_and(|e| e.ends_with("ball")));
        assert_eq!(ball.places(), ["bin", "cube"]);
        std::fs::remove_dir_all(&project.root).ok();
    }

    /// The try's argv and the Evaluation IR written beside it: the template's first suite, one
    /// explicit seed, no acceptance. The next seed is the next try's; a re-opened ② goes on from
    /// the latest try; a run going refuses it.
    #[test]
    fn the_try_argv_and_its_derived_evaluation() {
        let (project, mut teach) = open("q3-try");
        assert_eq!(teach.seed, TRY_SEED_BASE);
        teach.select(Some(0));
        teach.edit(Field::Above(0.06));
        let first = teach.start_try(None, false, &fresh()).expect("a try");
        assert!(!teach.dirty(), "saved before it runs");
        let saved = Teach::open(&project, &cube(), &repo()).unwrap();
        assert_eq!(saved.program.blocks[0].above, Some(0.06));
        let argv = &first.argv;
        assert_eq!(argv[..2], ["eval".to_owned(), "run".to_owned()]);
        assert_eq!(first.dir, project.root.join("try").join("001"));
        assert_eq!(PathBuf::from(after(argv, "--out")), first.dir);
        assert_eq!(PathBuf::from(after(argv, "--expert")), project.teach());
        assert_eq!(PathBuf::from(after(argv, "--policy")), project.bundle());
        assert_eq!(
            PathBuf::from(after(argv, "--frames")),
            first.dir.join("frames")
        );
        assert_eq!(after(argv, "--jobs"), "1");
        assert!(Path::new(after(argv, "--scene")).is_file());
        let config = PathBuf::from(after(argv, "--config"));
        assert_eq!(config, first.dir.join("evaluation.toml"));

        let ir = evaluation_from_toml(&std::fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(ir.suites.len(), 1);
        assert_eq!(ir.suites[0].name, "nominal");
        assert_eq!(
            ir.episodes,
            EpisodeBatch {
                n_episodes: 1,
                seeds: SeedPlan::Explicit(vec![1001])
            }
        );
        assert!(ir.acceptance.is_empty());
        assert!(ir.validate().is_empty(), "{:?}", ir.validate());

        teach.next_seed();
        let second = teach.start_try(None, false, &fresh()).expect("another");
        assert_eq!(second.dir, project.root.join("try").join("002"));
        assert_eq!(try_seed(&second.dir), Some(1002));
        assert_eq!(Teach::open(&project, &cube(), &repo()).unwrap().seed, 1002);

        let running = [
            Done,
            Done,
            Running {
                stage: "train".into(),
                fraction: None,
            },
            Locked,
            Locked,
        ];
        for (pid, queued, phases) in [
            (Some(7), false, fresh()),
            (None, true, fresh()),
            (None, false, running),
        ] {
            assert_eq!(teach.start_try(pid, queued, &phases), Err(TryRefused::Busy));
        }
        assert_eq!(project.tries().len(), 2, "a refused try writes nothing");
        std::fs::remove_dir_all(&project.root).ok();
    }

    /// A finished try read back: its one attempt, the cell to play, and where the cube went -
    /// the committed E2 trajectory never lifts it. A try with nothing written yet has none.
    #[test]
    fn a_finished_try_reads_back() {
        let (project, teach) = open("q3-read");
        let dir = project.next_try_dir();
        let fixture = repo().join("tests/fixtures/visible-learning/run");
        std::fs::create_dir_all(dir.join("traj")).unwrap();
        for file in ["report.json", "events.json", "traj/nominal-00.estraj"] {
            std::fs::copy(fixture.join(file), dir.join(file)).unwrap();
        }
        assert_eq!(teach.attempt(&dir), None, "no episodes.json yet");
        assert_eq!(teach.latest_try(), Some((dir.clone(), None)));
        let row = |termination: &str| EpisodeRow {
            suite: "nominal".into(),
            cell: "nominal-00".into(),
            episode: 0,
            seed: 101,
            termination: termination.into(),
            steps: 10,
            changed_steps: 0,
            histogram: [(termination.to_owned(), 1)].into(),
        };
        write_episodes(&[row("timeout")], &dir).unwrap();
        let lost = teach.attempt(&dir).expect("one attempt");
        assert_eq!(
            (lost.cell.as_str(), lost.success, lost.cause),
            (
                "nominal-00",
                false,
                Some(Cause::Outcome(Outcome::NeverLifted))
            )
        );
        assert_eq!(
            teach.verdict(Lang::En, Some(&lost), false),
            "Failed — The cube was never lifted"
        );
        write_episodes(&[row("success")], &dir).unwrap();
        let won = teach.attempt(&dir).expect("one attempt");
        assert!(won.success);
        let unknown = Tile {
            cause: None,
            ..lost.clone()
        };
        for lang in Lang::ALL {
            let lines = [
                teach.verdict(lang, Some(&won), false),
                teach.verdict(lang, Some(&lost), false),
                teach.verdict(lang, Some(&unknown), false),
                teach.verdict(lang, None, true),
                teach.verdict(lang, None, false),
            ];
            for line in &lines {
                assert!(!line.is_empty() && !line.contains("{}"), "{line}");
            }
            let distinct: std::collections::BTreeSet<_> = lines.iter().collect();
            assert_eq!(distinct.len(), lines.len(), "{lines:?}");
        }
        std::fs::remove_dir_all(&project.root).ok();
    }

    /// A project made before this packet has no program file: ② is read-only, with its reason.
    #[test]
    fn a_project_without_a_program_is_read_only() {
        let project = scratch_project("q3-old");
        std::fs::remove_file(project.teach()).unwrap();
        let none = Teach::open(&project, &cube(), &repo()).expect_err("read-only");
        assert_eq!(none, TeachError::NoFile);
        std::fs::write(project.teach(), "kind = [").unwrap();
        let broken = Teach::open(&project, &cube(), &repo()).expect_err("read-only");
        assert!(matches!(broken, TeachError::Unreadable(_)), "{broken:?}");
        for lang in Lang::ALL {
            for e in [&none, &broken] {
                let text = e.text(lang);
                assert!(
                    !text.contains("{}") && !text.starts_with("teach."),
                    "{text}"
                );
            }
        }
        std::fs::remove_dir_all(&project.root).ok();
    }

    /// Every word ② shows has both languages: each refusal (filled), warning, field, unit,
    /// target kind, gripper word, button and hover, and each reason a try is refused.
    #[test]
    fn every_teach_word_is_in_both_tables() {
        let refusals = [
            ProgramError::Parse("x".into()),
            ProgramError::Kind("template".into()),
            ProgramError::Robot("UR5".into()),
            ProgramError::Empty,
            ProgramError::BothHeights(0),
            ProgramError::NoHeight(0),
            ProgramError::Missing {
                block: 1,
                kind: "grip",
                field: "wait",
            },
            ProgramError::Misplaced {
                block: 1,
                kind: "grip",
                field: "pitch",
            },
            ProgramError::GripFirst,
            ProgramError::UnknownPlace {
                block: 0,
                name: "shelf".into(),
            },
            ProgramError::UnknownObject("ball".into()),
            ProgramError::Wait {
                block: 2,
                wait: 0.3,
                step: 0.2,
            },
            ProgramError::Scene(EnvError::Unsupported("x".into())),
        ];
        let mut keys: Vec<&str> = refusals.iter().map(program_error_key).collect();
        keys.extend(Warning::ALL.map(Warning::key));
        keys.extend(FieldName::ALL.map(FieldName::key));
        keys.extend(
            FieldName::ALL
                .iter()
                .filter_map(|f| f.unit().map(|(k, _)| k)),
        );
        keys.extend(TargetKind::ALL.map(TargetKind::key));
        keys.extend([Grip::Open, Grip::Closed].map(grip_key));
        keys.extend(Action::ALL.map(Action::key));
        keys.extend(Action::ALL.iter().filter_map(|a| a.hint_key()));
        for lang in Lang::ALL {
            for key in &keys {
                assert_ne!(Strings::get(lang).t(key), *key, "{lang:?}: {key}");
            }
            for e in &refusals {
                let text = error_text(lang, e);
                assert!(!text.contains("{}"), "{lang:?} {e:?}: {text}");
            }
            for r in [TryRefused::Busy, TryRefused::Invalid] {
                assert!(!r.text(lang).starts_with("teach."), "{r:?}");
            }
        }
        assert!(error_text(Lang::En, &refusals[6]).ends_with(": Wait"));
        assert_eq!(error_block(&ProgramError::GripFirst), Some(0));
        assert_eq!(
            TargetKind::of(&Target::Point([0.0, 0.0])),
            TargetKind::Point
        );
    }
}
