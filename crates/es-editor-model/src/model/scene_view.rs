//! ① Scene and ② Teach of a template project (packet M12/Y15, `docs/design/editor-redesign.md`
//! sections 3 and 6.3): the template's scene at the model's initial pose, what is in it, and the
//! teaching method in plain words. Read-only; editing a scene is S3/S4's.
//!
//! Nothing here is a second copy: the scene is loaded by the Replay panel's own loader
//! ([`load_scene`]), tessellated by `es_render`'s `TriScene` and projected by
//! `es_render::raster`, exactly as a replay tick is.

use std::path::{Path, PathBuf};

use es_assets::scene::{scene_id, JointKind, SceneDesc, Shape};
use es_core::StableId;
use es_render::raster::{project_scene, Camera, Projected};
use es_render::TriScene;

use crate::model::project::Project;
use crate::model::replay_view::load_scene;
use crate::model::template::{load, templates_root, Method, Template};
use crate::model::viewport::Shot;
use crate::model::watch::source_of;

/// Where the replay camera starts: the demo's showcase view (`es video showcase --eye`).
/// `ui/advanced.rs` re-exports it for the Replay panel, ① Scene and the result tiles.
pub const SHOWCASE_CAMERA: Camera = Camera {
    eye: [0.55, -0.45, 0.42],
    look_at: [0.12, -0.02, 0.08],
    // 45 degrees, the showcase default. `to_radians` is not `const`.
    fov_y: std::f64::consts::FRAC_PI_4,
    width: 640,
    height: 400,
};

/// A scene file, parsed, and its triangles at the initial pose.
#[derive(Debug)]
pub struct ScenePreview {
    path: PathBuf,
    scene: SceneDesc,
    tris: TriScene,
}

/// What the step panel lists, by scene names in scene order: the robot's root body, the things
/// it handles, and the cameras.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneContents {
    pub robot: Option<String>,
    pub objects: Vec<String>,
    pub cameras: Vec<String>,
}

impl ScenePreview {
    /// The template's scene, posed at `qpos0`: `MuJoCo`'s initial state is every hinge and slide
    /// at zero and every free body where the file puts it, which is the scene's own body poses -
    /// what `TriScene::from_scene` tessellates. No joint value is made up here.
    pub fn open(scene: &Path) -> Result<Self, String> {
        let path = scene.to_path_buf();
        let scene = load_scene(scene).map_err(|e| e.to_string())?;
        let tris = TriScene::from_scene(&scene).map_err(|e| e.to_string())?;
        Ok(Self { path, scene, tris })
    }

    /// The scene at its initial pose from `camera`, for the slower looks (packet M16/H8).
    pub fn shot(&self, camera: &Camera) -> Shot {
        Shot {
            scene: self.path.clone(),
            traj: None,
            tick: 0,
            camera: *camera,
        }
    }

    /// The triangles at the initial pose, for the material look.
    pub fn tris(&self) -> TriScene {
        self.tris.clone()
    }

    /// The scene seen from `camera`, back to front; empty for a camera with no orientation.
    pub fn project(&self, camera: &Camera) -> Projected {
        camera
            .view()
            .map_or_else(|_| Vec::new(), |view| project_scene(&self.tris, &view))
    }

    /// The robot is the top body of every chain a joint other than a free one moves; an object
    /// is a free body, or a group of fixed geoms named alike (`bin_floor`, `bin_wall_nx` are
    /// one `bin`). The ground plane and what the renderer lights the scene with are neither.
    ///
    /// ponytail: fixed geoms are grouped by the name before their first `_`; a scene whose
    /// unrelated geoms share a stem shows them as one object. A scene document that names its
    /// objects (S3) replaces the guess.
    pub fn contents(&self) -> SceneContents {
        let bodies = &self.scene.bodies;
        // An MJCF `<worldbody>` is a body of its own (`es_assets::mjcf`); what hangs from it
        // is at the top of its chain, as a URDF's root link is.
        let world = scene_id("body", "world");
        let top = |mut id: StableId| {
            for _ in 0..bodies.len() {
                let parent = bodies.iter().find(|b| b.id == id).and_then(|b| b.parent);
                match parent.filter(|p| *p != world) {
                    Some(parent) => id = parent,
                    None => break,
                }
            }
            id
        };
        let moved = |free: bool| {
            let joints = self.scene.joints.iter();
            let ids = joints.filter(move |j| (j.kind == JointKind::Free) == free);
            ids.map(|j| top(j.body)).collect::<Vec<_>>()
        };
        let (robots, free) = (moved(false), moved(true));
        // The renderer's own rule for what emits light, read back from its triangles.
        let lights: Vec<&String> = (self.tris.lights.iter())
            .filter_map(|&i| self.tris.tris.get(i as usize))
            .filter_map(|tri| self.tris.names.get(&tri.seg))
            .collect();
        let mut objects: Vec<String> = Vec::new();
        for body in bodies {
            if free.contains(&body.id) {
                objects.push(body.name.clone());
            }
            let root = top(body.id);
            if robots.contains(&root) || free.contains(&root) {
                continue;
            }
            for geom in &body.geoms {
                let stem = stem(&geom.name);
                let ground = matches!(geom.shape, Shape::Plane { .. });
                if stem.is_empty() || ground || lights.contains(&&geom.name) {
                    continue;
                }
                if !objects.iter().any(|o| o == stem) {
                    objects.push(stem.to_owned());
                }
            }
        }
        let name = |id: &StableId| bodies.iter().find(|b| b.id == *id).map(|b| b.name.clone());
        SceneContents {
            robot: robots.first().and_then(name),
            objects,
            cameras: self.scene.cameras.iter().map(|c| c.name.clone()).collect(),
        }
    }
}

/// The object a fixed geom belongs to: its name before the first `_` (`bin_floor` is the `bin`).
/// [`ScenePreview::contents`] groups by it, and a template's `[outcome] target` names one
/// ([`crate::model::outcome`]).
pub(crate) fn stem(geom: &str) -> &str {
    geom.split('_').next().unwrap_or_default()
}

/// ① and ② of a project: its template and that template's scene, or `None` when the editor
/// finds no template for it (it is not run from a checkout, or the template is gone).
pub fn open_step(
    project: &Project,
    repo_root: Option<PathBuf>,
) -> Option<(Template, Result<ScenePreview, String>)> {
    let (template, root) = source_of(project, repo_root).ok()?;
    let preview = ScenePreview::open(&root.join(&template.scene));
    Some((template, preview))
}

/// Where the outside camera starts on `template`'s scene: its `viewport`, else the showcase
/// view (packet M16/H7: the hand sits a metre from the arm's origin, out of that view).
pub fn camera_of(template: Option<&Template>) -> Camera {
    match template.and_then(|t| t.viewport) {
        Some(v) => Camera {
            eye: v.eye,
            look_at: v.look_at,
            ..SHOWCASE_CAMERA
        },
        None => SHOWCASE_CAMERA,
    }
}

/// [`camera_of`] the checkout's template whose scene `scene` is; a motion re-posed in ③, ④ or
/// ⑤ knows its scene, not its template.
pub fn camera_for(scene: Option<&Path>) -> Camera {
    let found = scene
        .zip(templates_root())
        .and_then(|(scene, root)| (load(&root).0.into_iter()).find(|t| scene.ends_with(&t.scene)));
    camera_of(found.as_ref())
}

/// ②'s sentence for a teaching method; `{}` is the robot.
pub fn method_key(method: Method) -> &'static str {
    match method {
        Method::Blocks => "teach.blocks",
        Method::Teacher => "teach.by_teacher",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::i18n::{fill, Lang};
    use crate::model::project::tests::{repo, scratch_project};

    fn fixture() -> PathBuf {
        repo().join("tests/fixtures/mjcf/so101_pick_place.xml")
    }

    /// The cube template's scene projects from the viewport's first camera, and its contents
    /// are one robot, the bin and the cube, and the overhead camera.
    #[test]
    fn the_cube_scene_projects_and_names_what_is_in_it() {
        let preview = ScenePreview::open(&fixture()).expect("the demo scene");
        let projected = preview.project(&SHOWCASE_CAMERA);
        assert!(projected.len() > 100, "{} triangles", projected.len());
        assert_eq!(projected, preview.project(&SHOWCASE_CAMERA), "not pure");
        assert_eq!(
            preview.contents(),
            SceneContents {
                robot: Some("base".into()),
                objects: vec!["bin".into(), "cube".into()],
                cameras: vec!["overhead".into()],
            }
        );
    }

    /// Packet M16/H7: a re-posed motion of the hand scene starts on the hand template's
    /// viewport, which sees the hand; the cube cards keep the showcase view.
    #[test]
    fn the_hand_scene_starts_on_its_template_viewport() {
        let hand = crate::model::teacher::tests::hand();
        let scene = repo().join(&hand.scene);
        let camera = camera_for(Some(&scene));
        assert_eq!(camera, camera_of(Some(&hand)));
        assert_ne!(camera, SHOWCASE_CAMERA);
        let preview = ScenePreview::open(&scene).expect("the hand scene");
        let seen = preview.project(&camera).len();
        assert!(seen > 1000, "{seen}");
        assert_eq!(camera_for(Some(&fixture())), SHOWCASE_CAMERA);
        assert_eq!(camera_for(None), SHOWCASE_CAMERA);
    }

    #[test]
    fn a_missing_scene_is_an_error_not_a_panic() {
        let err = ScenePreview::open(Path::new("no/such/scene.xml")).expect_err("missing");
        assert!(err.contains("scene.xml"), "{err}");
    }

    /// A project made from the cube template opens its template's scene; without a checkout it
    /// has none, and says so rather than failing.
    #[test]
    fn a_template_project_opens_its_scene() {
        let project = scratch_project("y15-scene");
        let (template, preview) = open_step(&project, Some(repo())).expect("the template");
        assert_eq!(template.id, "cube-into-bin");
        assert!(preview.is_ok(), "{:?}", preview.err());
        assert!(open_step(&project, None).is_none());
        for lang in Lang::ALL {
            let line = fill(lang, method_key(template.method), &[&template.robot]);
            assert!(line.contains("SO-101") && !line.contains("{}"), "{line}");
        }
        std::fs::remove_dir_all(&project.root).ok();
    }
}
