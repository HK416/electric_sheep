//! Plan H, task H1 oracle 3: the Shadow Hand's three cameras see the cube.
//!
//! `tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.xml` at its reset pose (every joint at
//! `qpos0`, so the scene's static body poses) seen from `top`, `front` and `side` at the Task
//! IR's 96x96:
//!
//! * the CPU reference (`Rs`) draws the held cube's face slabs and the goal cube's in every
//!   camera, and draws no alpha-0 geom (the hand's collision capsules, the floor);
//! * the GPU rasterizer reproduces the CPU reference bit for bit (`Rgb8`, `SegmentationId`), as
//!   the cornell oracles in `render.rs` do;
//! * the GPU path tracer at the X7 rerun's settings (4 spp, 3 bounces, exposure 64) gives the
//!   cube segmentation pixels in all three cameras.
//!
//! `ES_RENDER_DUMP=<dir>` writes each frame's `Rgb8` bytes (`<camera>_<path>.rgb`, 96x96x3)
//! there, for a person to look at. GPU tests print `SKIP` without a device or `slangc`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use es_assets::scene::SceneDesc;
use es_gpu::{Gpu, GpuOptions, SlangCompiler};
use es_math::{Pose, Quat, Vec3};
use es_render::{
    cpu, CameraView, Frame, ImageSpec, RenderConfig, Renderer, TileAtlasCfg, TriScene,
};
use es_sensor::Channel;

const SIDE: u32 = 96;
const CAMERAS: [&str; 3] = ["top", "front", "side"];

fn scene() -> SceneDesc {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf/shadow_hand");
    let xml = std::fs::read_to_string(dir.join("shadow_hand_repose.xml")).expect("the fixture");
    let mut scene = es_assets::parse_mjcf(&xml)
        .expect("the fixture parses")
        .scene;
    es_assets::mesh::load(&mut scene, &dir).expect("the STLs load");
    scene
}

/// A world-fixed MJCF camera in the `OpenCV` frame the renderer takes: MJCF looks down `-Z`
/// with `+Y` up, `OpenCV` down `+Z` with `+Y` down -- a half turn about `X`
/// (`es_env::render::camera_view`, which this crate cannot import: layer 9 over layer 5).
fn view(scene: &SceneDesc, name: &str) -> CameraView {
    let camera = scene
        .cameras
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no camera `{name}`"));
    assert!(camera.body.is_none(), "`{name}` is fixed to the world");
    CameraView {
        pose: camera
            .pose
            .compose(Pose::new(Vec3::ZERO, Quat::from_xyzw(1.0, 0.0, 0.0, 0.0))),
        spec: ImageSpec::pinhole(SIDE, SIDE, camera.fovy),
    }
}

fn rs() -> RenderConfig {
    RenderConfig::rs(TileAtlasCfg::row(SIDE, SIDE, 1))
}

/// The observation path the Task IR will declare (`render = { path = "pt", spp = 4,
/// bounces = 3, exposure = 64 }`, as `task-reach-vision-pt-4spp.toml`).
fn pt() -> RenderConfig {
    let mut cfg = RenderConfig::pt(TileAtlasCfg::row(SIDE, SIDE, 1), 4, 3);
    cfg.exposure = 64.0;
    cfg
}

/// Pixels per geom-name prefix (`object_`, `target_`), read off a segmentation tile.
fn counts(tri: &TriScene, seg: &[u32]) -> BTreeMap<&'static str, usize> {
    let mut out = BTreeMap::new();
    for prefix in ["object_", "target_"] {
        let n = seg
            .iter()
            .filter(|s| tri.names.get(s).is_some_and(|n| n.starts_with(prefix)))
            .count();
        out.insert(prefix, n);
    }
    out
}

fn dump(name: &str, path: &str, rgb: &[u8]) {
    if let Ok(dir) = std::env::var("ES_RENDER_DUMP") {
        let file = PathBuf::from(dir).join(format!("{name}_{path}.rgb"));
        std::fs::write(&file, rgb).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    }
}

fn cpu_frame(tri: &TriScene, cam: &CameraView) -> Frame {
    cpu::rasterize(tri, cam, &rs(), 0)
}

#[test]
fn every_camera_sees_both_cubes_and_no_alpha_zero_geom() {
    let scene = scene();
    let tri = TriScene::from_scene(&scene).expect("the hand tessellates");
    // Alpha 0 draws nothing: no triangle carries a collision geom's or the floor's id.
    let hidden: Vec<u32> = tri
        .names
        .iter()
        .filter(|(_, n)| {
            n.starts_with("robot0:C_") || *n == "floor0" || *n == "object" || *n == "target"
        })
        .map(|(s, _)| *s)
        .collect();
    // 20 collision geoms, the floor and the two cubes' own boxes (their slabs are drawn).
    assert_eq!(hidden.len(), 20 + 1 + 2, "{hidden:?}");
    assert!(tri.tris.iter().all(|t| !hidden.contains(&t.seg)));

    for name in CAMERAS {
        let frame = cpu_frame(&tri, &view(&scene, name));
        let seg = frame
            .tile(Channel::SegmentationId)
            .unwrap()
            .as_u32()
            .unwrap();
        let n = counts(&tri, seg);
        println!("{name}: Rs segmentation pixels {n:?}");
        dump(
            name,
            "rs",
            frame.tile(Channel::Rgb8).unwrap().as_u8().unwrap(),
        );
        assert!(n["object_"] > 0, "`{name}` does not see the cube");
        assert!(n["target_"] > 0, "`{name}` does not see the goal cube");
    }
}

/// `Some(gpu)` or a printed SKIP.
fn open(test: &str) -> Option<Gpu> {
    if let Err(e) = SlangCompiler::new() {
        println!("SKIP {test}: no slangc ({e})");
        return None;
    }
    match Gpu::open(GpuOptions::default()) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            println!("SKIP {test}: no Vulkan device ({e})");
            None
        }
    }
}

#[test]
fn gpu_rasterizer_matches_the_cpu_on_the_hand() {
    let test = "gpu_rasterizer_matches_the_cpu_on_the_hand";
    let Some(gpu) = open(test) else { return };
    let scene = scene();
    let tri = TriScene::from_scene(&scene).expect("the hand tessellates");
    let cams: Vec<CameraView> = CAMERAS.iter().map(|n| view(&scene, n)).collect();
    let mut renderer = Renderer::new(&gpu, rs()).expect("renderer");
    renderer.upload_tris(tri.clone()).expect("upload");
    for (name, cam) in CAMERAS.iter().zip(&cams) {
        let mut atlas = renderer.render(std::slice::from_ref(cam)).expect("render");
        let want = cpu_frame(&tri, cam);
        for channel in [Channel::Rgb8, Channel::SegmentationId] {
            let got = atlas.read_tile(0, channel).expect("tile");
            assert!(
                got.to_bytes() == want.tile(channel).unwrap().to_bytes(),
                "{name}: GPU {channel:?} differs from the CPU reference"
            );
        }
        println!("{name}: GPU Rs bit-equal to the CPU reference (Rgb8, SegmentationId)");
    }
}

#[test]
fn gpu_path_tracer_sees_the_cube_from_every_camera() {
    let test = "gpu_path_tracer_sees_the_cube_from_every_camera";
    let Some(gpu) = open(test) else { return };
    let scene = scene();
    let tri = TriScene::from_scene(&scene).expect("the hand tessellates");
    let mut renderer = Renderer::new(&gpu, pt()).expect("renderer");
    renderer.upload_tris(tri.clone()).expect("upload");
    for name in CAMERAS {
        let mut atlas = renderer.render(&[view(&scene, name)]).expect("render");
        let seg = atlas.read_tile(0, Channel::SegmentationId).expect("seg");
        let n = counts(&tri, seg.as_u32().unwrap());
        let rgb = atlas.read_tile(0, Channel::Rgb8).expect("rgb");
        dump(name, "pt", rgb.as_u8().unwrap());
        println!("{name}: Pt segmentation pixels {n:?}");
        assert!(n["object_"] > 0, "`{name}` does not see the cube on Pt");
    }
}
