//! The demonstration program (packet `docs/packets/M14/plan-q.md` Task Q1, design note
//! `docs/design/editor-redesign.md` section 11): what [`ScriptedExpert`](crate::ScriptedExpert)
//! does, as data -- *move* blocks (a point, a height, a wrist pitch, the gripper) and *grip*
//! blocks (open or close, then wait).
//!
//! Not an IR (spec 5.1 rule 6): the demonstrator's recipe, like
//! [`ExpertCfg`](crate::ExpertCfg), run by the same concrete struct (`INV-17`). The file says
//! only *what* to do; the tuning that is not the person's to change -- the joint tolerance, the
//! gripper's angles, the pace -- stays in `ExpertCfg`.

use es_assets::scene::SceneDesc;
use es_math::{units::DEG_TO_RAD, Vec3};
use serde::{Deserialize, Serialize};

/// The built-in `so101-pick-place`: `templates/teach/so101-pick-place.toml`, compiled in, so
/// there is one source of the demo's numbers.
pub const SO101_PICK_PLACE: &str = include_str!("../../../templates/teach/so101-pick-place.toml");

const KIND: &str = "demonstration";
const ROBOT: &str = "SO-101";
/// The word a move block uses for the program's object.
pub const OBJECT: &str = "object";

/// A demonstration program file (`kind = "demonstration"`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub kind: String,
    pub robot: String,
    /// The free-joint body a block calls `"object"`, latched where it is at the episode's
    /// first step.
    pub object: String,
    #[serde(default)]
    pub blocks: Vec<Block>,
}

/// One block: a move when `target` is set, a grip otherwise.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    #[serde(rename = "move", default, skip_serializing_if = "Option::is_none")]
    pub target: Option<Target>,
    /// Metres above the target's own centre.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub above: Option<f64>,
    /// Metres above the world origin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<f64>,
    /// Tool pitch, degrees; -90 is straight down.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f64>,
    pub grip: Grip,
    /// Seconds a grip block waits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wait: Option<f64>,
}

/// Where a move block goes: [`OBJECT`], a place (a geom stem in the scene), or `[x, y]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Target {
    Named(String),
    Point([f64; 2]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Grip {
    Open,
    Closed,
}

/// Why a program was refused. `block` is the block's index, shown counting from 1.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ProgramError {
    #[error("demonstration program: {0}")]
    Parse(String),
    #[error("kind = {0:?}; a demonstration program says kind = \"demonstration\"")]
    Kind(String),
    #[error("robot = {0:?}; the demonstrator drives SO-101 only")]
    Robot(String),
    #[error("the program has no blocks")]
    Empty,
    #[error("block {}: both `above` and `height`; a move says exactly one", .0 + 1)]
    BothHeights(usize),
    #[error("block {}: neither `above` nor `height`; a move says exactly one", .0 + 1)]
    NoHeight(usize),
    #[error("block {}: a {kind} block needs `{field}`", .block + 1)]
    Missing {
        block: usize,
        kind: &'static str,
        field: &'static str,
    },
    #[error("block {}: `{field}` does not belong on a {kind} block", .block + 1)]
    Misplaced {
        block: usize,
        kind: &'static str,
        field: &'static str,
    },
    #[error("block 1 is a grip block: there is no pose before it to hold")]
    GripFirst,
    #[error("block {}: no geom in the scene is named `{name}_...` or `{name}`", .block + 1)]
    UnknownPlace { block: usize, name: String },
    #[error("object = {0:?}: that is not the body whose free joint the demonstrator reads")]
    UnknownObject(String),
    #[error(
        "block {}: wait = {wait} s is not a whole, positive number of the demonstrator's {step} s steps",
        .block + 1
    )]
    Wait { block: usize, wait: f64, step: f64 },
    /// The scene is not an arm the closed form solves (`EnvError::Unsupported`).
    #[error(transparent)]
    Scene(#[from] crate::EnvError),
}

impl Program {
    /// Parses and [`validate`](Self::validate)s a program file.
    pub fn parse(text: &str) -> Result<Self, ProgramError> {
        let program: Self =
            es_ir::serial::parse_toml(text).map_err(|e| ProgramError::Parse(e.to_string()))?;
        program.validate()?;
        Ok(program)
    }

    /// [`SO101_PICK_PLACE`], parsed.
    pub fn builtin() -> Self {
        Self::parse(SO101_PICK_PLACE).expect("the built-in program is valid")
    }

    /// Everything that can be checked without a scene.
    pub fn validate(&self) -> Result<(), ProgramError> {
        if self.kind != KIND {
            return Err(ProgramError::Kind(self.kind.clone()));
        }
        if self.robot != ROBOT {
            return Err(ProgramError::Robot(self.robot.clone()));
        }
        if self.blocks.is_empty() {
            return Err(ProgramError::Empty);
        }
        for (i, b) in self.blocks.iter().enumerate() {
            let missing = |kind, field| ProgramError::Missing {
                block: i,
                kind,
                field,
            };
            let misplaced = |kind, field| ProgramError::Misplaced {
                block: i,
                kind,
                field,
            };
            if b.target.is_some() {
                match (b.above, b.height) {
                    (Some(_), Some(_)) => return Err(ProgramError::BothHeights(i)),
                    (None, None) => return Err(ProgramError::NoHeight(i)),
                    _ => {}
                }
                if b.pitch.is_none() {
                    return Err(missing("move", "pitch"));
                }
                if b.wait.is_some() {
                    return Err(misplaced("move", "wait"));
                }
            } else {
                if i == 0 {
                    return Err(ProgramError::GripFirst);
                }
                if b.wait.is_none() {
                    return Err(missing("grip", "wait"));
                }
                for (field, v) in [("above", b.above), ("height", b.height), ("pitch", b.pitch)] {
                    if v.is_some() {
                        return Err(misplaced("grip", field));
                    }
                }
            }
        }
        Ok(())
    }

    /// Every block's waypoint in `scene`: a grip block holds the pose of the block before it,
    /// and a place resolves to its point once, here.
    pub fn waypoints(&self, scene: &SceneDesc) -> Result<Vec<Waypoint>, ProgramError> {
        self.validate()?;
        let mut out: Vec<Waypoint> = Vec::with_capacity(self.blocks.len());
        for (i, b) in self.blocks.iter().enumerate() {
            let w = match &b.target {
                Some(target) => {
                    let aim = match target {
                        Target::Named(name) if name == OBJECT => None,
                        Target::Named(name) => Some(place_point(scene, name).ok_or_else(|| {
                            ProgramError::UnknownPlace {
                                block: i,
                                name: name.clone(),
                            }
                        })?),
                        Target::Point([x, y]) => Some(Vec3::new(*x, *y, 0.0)),
                    };
                    // `validate` above: a move has exactly one of the two, and a pitch.
                    let z = match (b.above, b.height) {
                        (Some(above), _) => Z::Above(above),
                        (_, height) => Z::Height(height.unwrap_or_default()),
                    };
                    Waypoint {
                        aim,
                        z,
                        pitch: b.pitch.unwrap_or_default() * DEG_TO_RAD,
                        grip: b.grip,
                        wait: None,
                    }
                }
                None => Waypoint {
                    grip: b.grip,
                    wait: b.wait,
                    ..*out.last().ok_or(ProgramError::GripFirst)?
                },
            };
            out.push(w);
        }
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Z {
    Above(f64),
    Height(f64),
}

/// One block, resolved against a scene: where the tool goes, at what pitch, with the gripper
/// how, and -- for a grip block -- how long it waits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Waypoint {
    /// `None` is the object, wherever it was latched.
    aim: Option<Vec3>,
    z: Z,
    pitch: f64,
    grip: Grip,
    wait: Option<f64>,
}

impl Waypoint {
    /// The tool target with the object at `object`.
    pub fn tool(&self, object: Vec3) -> Vec3 {
        let at = self.aim.unwrap_or(object);
        let z = match self.z {
            Z::Above(above) => at.z + above,
            Z::Height(height) => height,
        };
        Vec3::new(at.x, at.y, z)
    }

    /// Tool pitch, radians.
    pub fn pitch(&self) -> f64 {
        self.pitch
    }

    pub fn grip(&self) -> Grip {
        self.grip
    }

    /// Seconds a grip block waits; `None` for a move, which waits until it is reached.
    pub fn wait(&self) -> Option<f64> {
        self.wait
    }
}

/// The object a fixed geom belongs to: its name before the first `_` (`bin_floor` is the
/// `bin`) -- the rule `es-editor-model`'s scene view groups a scene's objects by.
fn stem(geom: &str) -> &str {
    geom.split('_').next().unwrap_or_default()
}

/// A place's point: the world centre of the lowest geom of stem `name` (the bin's floor), in
/// the scene's zero configuration.
pub fn place_point(scene: &SceneDesc, name: &str) -> Option<Vec3> {
    let world = crate::expert::world_poses(scene).ok()?;
    scene
        .bodies
        .iter()
        .zip(&world)
        .flat_map(|(body, (_, pose))| {
            (body.geoms.iter())
                .filter(|g| stem(&g.name) == name)
                .map(|g| pose.transform_point(g.pose.position))
        })
        .min_by(|a, b| a.z.total_cmp(&b.z))
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn scene() -> SceneDesc {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/mjcf/so101_pick_place.xml");
        let xml = std::fs::read_to_string(&path).expect("the V0 fixture is in the repo");
        es_assets::parse_mjcf(&xml)
            .expect("the V0 fixture parses")
            .scene
    }

    const HEAD: &str = "kind = \"demonstration\"\nrobot = \"SO-101\"\nobject = \"cube\"\n";
    const MOVE: &str =
        "[[blocks]]\nmove = \"object\"\nabove = 0.05\npitch = -85\ngrip = \"open\"\n";

    fn parse(blocks: &str) -> Result<Program, ProgramError> {
        Program::parse(&format!("{HEAD}{blocks}"))
    }

    #[test]
    fn the_committed_file_is_the_builtin() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../templates/teach/so101-pick-place.toml");
        let text = std::fs::read_to_string(&path).expect("the committed program");
        assert_eq!(text, SO101_PICK_PLACE);
        let program = Program::parse(&text).expect("the committed program parses");
        assert_eq!(program, Program::builtin());
        assert_eq!(program.blocks.len(), 7, "today's seven stages");
        // A program written back reads back the same.
        let again = es_ir::serial::write_toml(&program).expect("serializes");
        assert_eq!(Program::parse(&again).expect("reparses"), program);
    }

    #[test]
    fn the_format_refuses_by_name() {
        assert_eq!(parse(""), Err(ProgramError::Empty));
        let both = "[[blocks]]\nmove = \"object\"\nabove = 0.05\nheight = 0.1\npitch = -85\ngrip = \"open\"\n";
        assert_eq!(parse(both), Err(ProgramError::BothHeights(0)));
        let neither = "[[blocks]]\nmove = \"object\"\npitch = -85\ngrip = \"open\"\n";
        assert_eq!(parse(neither), Err(ProgramError::NoHeight(0)));
        let first = "[[blocks]]\ngrip = \"closed\"\nwait = 1.0\n";
        assert_eq!(parse(first), Err(ProgramError::GripFirst));
        let no_wait = format!("{MOVE}[[blocks]]\ngrip = \"closed\"\n");
        assert!(matches!(
            parse(&no_wait),
            Err(ProgramError::Missing {
                block: 1,
                field: "wait",
                ..
            })
        ));
        let held = format!("{MOVE}[[blocks]]\ngrip = \"closed\"\nwait = 1.0\npitch = -80\n");
        assert!(matches!(
            parse(&held),
            Err(ProgramError::Misplaced {
                block: 1,
                field: "pitch",
                ..
            })
        ));
        let Err(ProgramError::Parse(why)) = parse(&MOVE.replace("pitch", "wrist")) else {
            panic!("an unknown key parsed");
        };
        assert!(why.contains("wrist"), "{why}");
        let robot = Program::parse(&(HEAD.replace("SO-101", "UR5") + MOVE));
        assert_eq!(robot, Err(ProgramError::Robot("UR5".into())));
        let kind = Program::parse(&(HEAD.replace("\"demonstration\"", "\"template\"") + MOVE));
        assert_eq!(kind, Err(ProgramError::Kind("template".into())));
    }

    /// Review focus 2: "bin" is the bin's floor, exactly the literals `demo_cfg` wrote.
    #[test]
    fn a_place_is_its_lowest_geom() {
        let bin = place_point(&scene(), "bin").expect("the scene has a bin");
        assert_eq!((bin.x, bin.y), (0.14, -0.10));
        assert_eq!(bin.z, 0.004, "the floor, not a wall");
        assert_eq!(place_point(&scene(), "shelf"), None);
        let unknown =
            parse("[[blocks]]\nmove = \"shelf\"\nheight = 0.1\npitch = -85\ngrip = \"open\"\n")
                .expect("a place is only checked against a scene")
                .waypoints(&scene());
        assert_eq!(
            unknown,
            Err(ProgramError::UnknownPlace {
                block: 0,
                name: "shelf".into()
            })
        );
    }

    #[test]
    fn a_grip_block_holds_the_pose_before_it() {
        let program = parse(&format!(
            "{MOVE}[[blocks]]\ngrip = \"closed\"\nwait = 1.0\n[[blocks]]\nmove = [0.2, -0.1]\nheight = 0.1\npitch = -45\ngrip = \"closed\"\n"
        ))
        .expect("parses");
        let w = program.waypoints(&scene()).expect("resolves");
        let cube = Vec3::new(0.24, 0.0, 0.02);
        assert_eq!(w[1].tool(cube), w[0].tool(cube));
        assert_eq!(w[1].pitch(), w[0].pitch());
        assert_eq!((w[1].grip(), w[1].wait()), (Grip::Closed, Some(1.0)));
        assert_eq!(w[0].tool(cube), Vec3::new(0.24, 0.0, 0.02 + 0.05));
        assert_eq!(w[0].pitch(), -85.0 * DEG_TO_RAD);
        assert_eq!(w[2].tool(cube), Vec3::new(0.2, -0.1, 0.1));
    }
}
