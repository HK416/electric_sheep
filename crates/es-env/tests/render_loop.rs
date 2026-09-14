//! Oracles for the renderer in the env loop (packet `docs/packets/M5/V0b-render-in-the-loop.md`,
//! design note `docs/design/visible-learning.md` section 7).
//!
//! The **CPU** reference is the golden path (spec 3.5, `crates/es-render/tests/render.rs:14-16`):
//! `tests/golden/render/so101_frame0.*` is produced by `es_render::cpu` from V0's fixture
//! scene at a fixed state, so no driver's arithmetic is committed. The GPU leg renders the
//! same state through `EnvRenderer` and is compared to that reference bit for bit; with no
//! Vulkan device or no `slangc` it prints `SKIP <test>: <reason>` and returns, exactly as the
//! `es-render` suite already does.
//!
//! "Fixed state" here is the scene's rest configuration — every arm joint at `qpos = 0`, which
//! is what the `SceneDesc` body poses already are — with the cube's free body pinned at a
//! world pose written below. No physics runs: a golden that needed `MuJoCo` to reproduce would
//! be a golden most machines could not check.

#![cfg(feature = "render")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_env::render::{body_poses, camera_view, check_image_spec, image_spec, render_config};
use es_env::{EnvError, EnvRenderer, EnvRendererCfg};
use es_gpu::{Gpu, GpuOptions, SlangCompiler};
use es_ir::image::{ChannelFormat, ColorSpace, ImageDType};
use es_math::{Pose, Quat, Vec3};
use es_physics_core::backend::{IndexRange, ModelInfo, StateView};
use es_render::{cpu, Channel, Tile, TriScene};

const W: u32 = 96;
const H: u32 = 96;
const GOLDEN: &str = "so101_frame0";

/// The cube's pinned world pose: lifted off the table and turned 45 degrees about `+Z`, so the
/// frame exercises the rotation half of the pose map and not only the translation.
const CUBE_POS: Vec3 = Vec3::new(0.20, 0.03, 0.06);
const CUBE_QUAT: Quat = Quat::from_xyzw(0.0, 0.0, 0.382_683_432_365_089_8, 0.923_879_532_511_286_7);

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden_dir() -> PathBuf {
    repo_root().join("tests/golden/render")
}

/// V0's fixture scene (`tests/fixtures/mjcf/so101_pick_place.xml`).
fn scene() -> SceneDesc {
    let path = repo_root().join("tests/fixtures/mjcf/so101_pick_place.xml");
    let xml = std::fs::read_to_string(&path).expect("the V0 fixture is in the repo");
    es_assets::mjcf::parse_str(&xml)
        .expect("the V0 fixture parses")
        .scene
}

fn by_name<'a>(scene: &'a SceneDesc, name: &str) -> &'a es_assets::scene::Body {
    scene
        .bodies
        .iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("body `{name}` is in the V0 fixture"))
}

fn overhead(scene: &SceneDesc) -> StableId {
    scene
        .cameras
        .iter()
        .find(|c| c.name.ends_with("overhead"))
        .expect("the V0 fixture declares the overhead camera")
        .id
}

fn cfg(scene: &SceneDesc) -> EnvRendererCfg {
    EnvRendererCfg::rgb(overhead(scene), W, H)
}

/// A `ModelInfo` / `StateView` pair standing for one control step of a running env.
///
/// Only the cube is in `ModelInfo::body`: every other body is at its scene pose, which is the
/// arm's rest configuration, and a body absent from the map keeps exactly that (the property
/// `an_absent_body_keeps_its_scene_pose` pins). That is also what a backend that reports a
/// partial `xpos` would produce.
struct Fixed {
    model: ModelInfo,
    xpos: Vec<f64>,
    xquat: Vec<f64>,
}

impl Fixed {
    fn new(scene: &SceneDesc, pose: Pose) -> Self {
        let nbody = scene.bodies.len() as u32;
        let row = scene
            .bodies
            .iter()
            .position(|b| b.name == "cube")
            .expect("the V0 fixture has a cube");
        let mut model = ModelInfo {
            nbody,
            n_envs: 1,
            ..ModelInfo::default()
        };
        model
            .body
            .insert(by_name(scene, "cube").id, IndexRange::new(row as u32, 1));
        let mut xpos = vec![0.0; nbody as usize * 3];
        let mut xquat = vec![0.0; nbody as usize * 4];
        xpos[row * 3..row * 3 + 3].copy_from_slice(&[
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ]);
        xquat[row * 4..row * 4 + 4].copy_from_slice(&[
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]);
        Self { model, xpos, xquat }
    }

    fn state(&self) -> StateView<'_> {
        StateView {
            n_envs: 1,
            xpos: &self.xpos,
            xquat: &self.xquat,
            ..StateView::default()
        }
    }
}

fn fixed(scene: &SceneDesc) -> Fixed {
    Fixed::new(scene, Pose::new(CUBE_POS, CUBE_QUAT))
}

/// The CPU reference frame for the fixed state: the golden path.
fn cpu_tile(scene: &SceneDesc) -> Tile {
    let f = fixed(scene);
    let state = f.state();
    let world = body_poses(&f.model, &state, 0);
    let tri = TriScene::from_scene_with_poses(scene, &world).expect("the fixture tessellates");
    let cfg = cfg(scene);
    let view = camera_view(scene, &cfg, &world).expect("the overhead camera resolves");
    cpu::rasterize(&tri, &view, &render_config(&cfg), 0)
        .tile(Channel::Rgb8)
        .expect("Rgb8 was requested")
        .clone()
}

/// `Some(gpu)` or a printed SKIP, the same two reasons `crates/es-render/tests/render.rs:57-72`
/// reports.
fn open(test: &str) -> Option<Gpu> {
    if let Err(e) = SlangCompiler::new() {
        println!("SKIP {test}: no slangc ({e})");
        return None;
    }
    match Gpu::open(GpuOptions::default()) {
        Ok(gpu) => {
            println!("RAN {test} on {}", gpu.capabilities().device_name);
            Some(gpu)
        }
        Err(e) => {
            println!("SKIP {test}: no Vulkan device ({e})");
            None
        }
    }
}

// --- animation ------------------------------------------------------------------------------

#[test]
fn poses_from_state_move_the_triangles() {
    let scene = scene();
    let offset = Vec3::new(0.5, -0.25, 0.125);
    let cube = by_name(&scene, "cube");
    let moved = Pose::new(cube.pose.position + offset, cube.pose.orientation);
    let world: BTreeMap<StableId, Pose> = [(cube.id, moved)].into_iter().collect();

    let base = TriScene::from_scene(&scene).expect("tessellates");
    let posed = TriScene::from_scene_with_poses(&scene, &world).expect("tessellates");
    assert_eq!(
        base.tris.len(),
        posed.tris.len(),
        "the tessellation changed"
    );
    assert_eq!(base.names, posed.names, "segmentation ids changed");

    // The cube's own segmentation ids, so "that body's triangles" is not a guess.
    let cube_segs: Vec<u32> = base
        .names
        .iter()
        .filter(|(_, name)| name.contains("cube"))
        .map(|(seg, _)| *seg)
        .collect();
    assert!(!cube_segs.is_empty(), "the cube has a geom");
    let mut moved_tris = 0;
    for (a, b) in base.tris.iter().zip(&posed.tris) {
        if cube_segs.contains(&a.seg) {
            for v in 0..3 {
                for (c, d) in [offset.x, offset.y, offset.z].into_iter().enumerate() {
                    let got = f64::from(b.v[v][c] - a.v[v][c]);
                    assert!(
                        (got - d).abs() < 1e-6,
                        "cube vertex {v} axis {c} moved by {got}, not {d}"
                    );
                }
            }
            moved_tris += 1;
        } else {
            assert_eq!(a, b, "a body outside the pose map moved");
        }
    }
    assert_eq!(moved_tris, 12, "a box is 12 triangles");
}

#[test]
fn an_absent_body_keeps_its_scene_pose() {
    let scene = scene();
    let empty = BTreeMap::new();
    assert_eq!(
        TriScene::from_scene(&scene).expect("tessellates"),
        TriScene::from_scene_with_poses(&scene, &empty).expect("tessellates"),
        "an empty pose map must reproduce the static scene exactly"
    );

    // One body known, the rest unknown: only that body moves.
    let cube = by_name(&scene, "cube");
    let world: BTreeMap<StableId, Pose> = [(cube.id, Pose::new(CUBE_POS, CUBE_QUAT))]
        .into_iter()
        .collect();
    let partial = TriScene::from_scene_with_poses(&scene, &world).expect("tessellates");
    let statics = TriScene::from_scene(&scene).expect("tessellates");
    let cube_segs: Vec<u32> = statics
        .names
        .iter()
        .filter(|(_, name)| name.contains("cube"))
        .map(|(seg, _)| *seg)
        .collect();
    for (a, b) in statics.tris.iter().zip(&partial.tris) {
        if !cube_segs.contains(&a.seg) {
            assert_eq!(a, b, "an absent body did not keep its scene pose");
        }
    }
}

// --- the golden -------------------------------------------------------------------------------

/// Regenerates `tests/golden/render/so101_frame0.*` from the **CPU** reference. Run explicitly:
/// `cargo test -p es-env --features render --test render_loop -- --ignored generate_so101_golden`.
#[test]
#[ignore = "golden generator; run explicitly"]
fn generate_so101_golden() {
    let tile = cpu_tile(&scene());
    tile.write_to(&golden_dir(), GOLDEN).expect("write golden");
    println!(
        "wrote {GOLDEN} ({} bytes, shape {:?})",
        tile.to_bytes().len(),
        tile.shape
    );
}

#[test]
fn cpu_frame_matches_the_golden() {
    let tile = cpu_tile(&scene());
    let path = golden_dir().join(format!("{GOLDEN}.bin"));
    let expected = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run the generate_so101_golden test)",
            path.display()
        )
    });
    let got = tile.to_bytes();
    assert_eq!(got.len(), expected.len(), "{GOLDEN} size");
    assert!(got == expected, "{GOLDEN} differs from its golden");
    // A blank frame would match a blank golden: the camera must actually see the scene.
    assert!(
        got.iter().any(|b| *b != 0),
        "the overhead camera rendered nothing"
    );
    println!("bit-equal CPU vs golden: {GOLDEN}");
}

#[test]
fn two_runs_of_the_same_state_are_byte_identical() {
    let scene = scene();
    assert!(
        cpu_tile(&scene).to_bytes() == cpu_tile(&scene).to_bytes(),
        "two CPU renders of the same state differ"
    );
}

#[test]
fn frames_written_to_disk_round_trip() {
    let dir = std::env::temp_dir().join(format!("es-env-frames-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let tile = cpu_tile(&scene());
    tile.write_to(&dir, "000000").expect("write frame");

    let bin = std::fs::read(dir.join("000000.bin")).expect("read frame");
    assert!(bin == tile.to_bytes(), "the frame did not round trip");
    let sidecar: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("000000.json")).expect("read sidecar"))
            .expect("the sidecar is json");
    assert_eq!(sidecar["dtype"], "u8");
    assert_eq!(sidecar["shape"][0], H);
    assert_eq!(sidecar["shape"][1], W);
    assert_eq!(sidecar["shape"][2], 3);
    // The same three keys the existing goldens carry.
    let golden: serde_json::Value = serde_json::from_slice(
        &std::fs::read(golden_dir().join("cornell_rs_rgb8.json")).expect("read golden sidecar"),
    )
    .expect("the golden sidecar is json");
    assert_eq!(sidecar["layout"], golden["layout"]);
    let _ = std::fs::remove_dir_all(&dir);
}

// --- the image observation port ----------------------------------------------------------------

/// V0's own Observation IR (`tests/fixtures/visible-learning/observation.toml`) declares the
/// `ImageSpec` the renderer has to match; a size, channel-count or colour-space difference is
/// an error naming the field, and nothing is resized to make it fit (spec 7.2, `INV-14`).
#[test]
fn the_declared_image_spec_is_checked_not_coerced() {
    let scene = scene();
    let cfg = cfg(&scene);
    let ours = image_spec(&scene, &cfg).expect("the overhead camera resolves");

    let toml = std::fs::read_to_string(
        repo_root().join("tests/fixtures/visible-learning/observation.toml"),
    )
    .expect("V0's observation IR is in the repo");
    let obs = es_ir::serial::observation_from_toml(&toml).expect("it parses");
    let declared = obs
        .graph
        .nodes
        .values()
        .find_map(|n| match n {
            es_ir::observation::ObservationNode::ImageInput { io, .. } => io.output.image,
            _ => None,
        })
        .expect("V0 declares one image input");
    assert_eq!(check_image_spec(&ours, &declared), Ok(()));

    for (field, broken) in [
        (
            "width",
            es_ir::image::ImageSpec {
                width: declared.width + 32,
                ..declared
            },
        ),
        (
            "height",
            es_ir::image::ImageSpec {
                height: declared.height / 2,
                ..declared
            },
        ),
        (
            "channels",
            es_ir::image::ImageSpec {
                channels: ChannelFormat::Gray,
                ..declared
            },
        ),
        (
            "color_space",
            es_ir::image::ImageSpec {
                color_space: ColorSpace::Linear,
                ..declared
            },
        ),
        (
            "dtype",
            es_ir::image::ImageSpec {
                dtype: ImageDType::F32,
                ..declared
            },
        ),
    ] {
        match check_image_spec(&ours, &broken) {
            Err(EnvError::ImageSpec { field: got, .. }) => assert_eq!(got, field),
            other => panic!("{field}: expected a named mismatch, got {other:?}"),
        }
    }
}

// --- GPU ---------------------------------------------------------------------------------------

#[test]
fn gpu_frame_matches_the_cpu_frame() {
    let test = "gpu_frame_matches_the_cpu_frame";
    let Some(gpu) = open(test) else { return };
    let scene = scene();
    let f = fixed(&scene);
    let dir = std::env::temp_dir().join(format!("es-env-gpu-frames-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut renderer = EnvRenderer::new(
        &gpu,
        &scene,
        EnvRendererCfg {
            frames_dir: Some(dir.clone()),
            ..cfg(&scene)
        },
    )
    .expect("renderer");

    let a = renderer
        .frame(&f.model, &f.state(), 0)
        .expect("frame 0")
        .to_bytes();
    let want = cpu_tile(&scene).to_bytes();
    let diff = a.iter().zip(&want).filter(|(x, y)| x != y).count();
    println!(
        "{test}: {diff} of {} bytes differ from the CPU reference",
        a.len()
    );
    assert!(
        a == want,
        "the GPU frame must be bit-equal to the CPU golden"
    );

    // ... and rendering the same state twice is the same frame (spec 3.5 tier 1, same device).
    let b = renderer
        .frame(&f.model, &f.state(), 0)
        .expect("frame 1")
        .to_bytes();
    assert!(a == b, "two GPU frames of the same state differ");

    // Both frames landed on disk under their own index.
    assert_eq!(renderer.frames(), 2);
    assert!(
        std::fs::read(dir.join("000000.bin")).expect("frame 0 on disk") == a,
        "frame 0 on disk differs from the tile"
    );
    assert!(
        std::fs::read(dir.join("000001.bin")).expect("frame 1 on disk") == b,
        "frame 1 on disk differs from the tile"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
