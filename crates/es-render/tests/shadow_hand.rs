//! Plan H, task H1 oracle 3: the Shadow Hand's three cameras see the cube.
//!
//! `tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.xml` at its reset pose (every joint at
//! `qpos0`, so the scene's static body poses) seen from `top`, `front` and `side` at the Task
//! IR's 96x96:
//!
//! * the CPU reference (`Rs`) draws the held cube and the goal cube in every camera, and
//!   draws no alpha-0 geom (the hand's collision capsules, which name a material, and the
//!   floor);
//! * the GPU rasterizer reproduces the CPU reference bit for bit (`Rgb8`, `SegmentationId`), as
//!   the cornell oracles in `render.rs` do;
//! * the GPU path tracer at the X7 rerun's settings (4 spp, 3 bounces, exposure 64) gives the
//!   cube segmentation pixels in all three cameras;
//! * packet H1b: the two cubes wear the bundle's `block.png` where `MuJoCo`'s own renderer puts
//!   it (`mujoco.Renderer`, as `textured.rs`'s placement oracle; needs `ES_PYTHON`).
//!
//! `ES_RENDER_DUMP=<dir>` writes each frame's `Rgb8` bytes (`<camera>_<path>.rgb`, 96x96x3)
//! there, for a person to look at. GPU tests print `SKIP` without a device or `slangc`.

#![allow(clippy::many_single_char_names)]

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
    view_at(scene, name, SIDE)
}

fn view_at(scene: &SceneDesc, name: &str, side: u32) -> CameraView {
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
        spec: ImageSpec::pinhole(side, side, camera.fovy),
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

/// Pixels per cube (`object`, `target`), read off a segmentation tile.
fn counts(tri: &TriScene, seg: &[u32]) -> BTreeMap<&'static str, usize> {
    let mut out = BTreeMap::new();
    for cube in ["object", "target"] {
        let n = seg
            .iter()
            .filter(|s| tri.names.get(s).is_some_and(|n| n == cube))
            .count();
        out.insert(cube, n);
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
        .filter(|(_, n)| n.starts_with("robot0:C_") || *n == "floor0")
        .map(|(s, _)| *s)
        .collect();
    // 20 collision geoms (material `robot0:MatColl`, geom alpha 0) and the floor.
    assert_eq!(hidden.len(), 20 + 1, "{hidden:?}");
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
        assert!(n["object"] > 0, "`{name}` does not see the cube");
        assert!(n["target"] > 0, "`{name}` does not see the goal cube");
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
        assert!(n["object"] > 0, "`{name}` does not see the cube on Pt");
    }
}

// --- packet H1b: the block texture where MuJoCo puts it ------------------------------------------

/// `MuJoCo` drawing the fixture at `qpos0` from each camera, one process for all three (its
/// GL context has failed to start now and then on this Windows desktop, so the test launches
/// it once and skips on failure). Two edits to what it draws, both
/// shading rather than placement: the headlight's specular is 0 (the cube materials'
/// `specular="1"` would wash a face turned to the camera towards white, and this renderer's
/// `Rs` Lambert draws no highlight), and the goal material's alpha is 1 (this renderer draws the
/// goal opaque). Alpha-0 geoms go to group 5, which is not drawn, so `MuJoCo`'s transparent pass
/// cannot occlude anything with them.
const MUJOCO: &str = r"
import os
import sys
import mujoco
path, cameras, side = sys.argv[1], sys.argv[2].split(','), int(sys.argv[3])
m = mujoco.MjModel.from_xml_path(path)
m.vis.headlight.specular[:] = 0
m.mat_rgba[m.material('material:target').id, 3] = 1
m.geom_group[m.geom_rgba[:, 3] == 0] = 5
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
opt = mujoco.MjvOption()
opt.geomgroup[5] = 0
r = mujoco.Renderer(m, side, side)
for camera in cameras:
    r.update_scene(d, camera=camera, scene_option=opt)
    sys.stdout.buffer.write(r.render().tobytes())
sys.stdout.buffer.flush()
os._exit(0)  # past the GL context's teardown, which has crashed on this Windows desktop
";

/// One `side x side x 3` image per camera of [`CAMERAS`], in order.
fn mujoco_render(side: u32) -> Result<Vec<Vec<u8>>, String> {
    let python = match std::env::var("ES_PYTHON") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => "python".to_owned(),
    };
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf/shadow_hand");
    let out = std::process::Command::new(&python)
        .args(["-c", MUJOCO])
        .arg(dir.join("shadow_hand_repose.xml"))
        .arg(CAMERAS.join(","))
        .arg(side.to_string())
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("`{python}`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let one = (side * side * 3) as usize;
    if out.stdout.len() != one * CAMERAS.len() {
        return Err(format!("{} bytes from MuJoCo", out.stdout.len()));
    }
    Ok(out.stdout.chunks(one).map(<[u8]>::to_vec).collect())
}

fn srgb_decode(b: u8) -> f32 {
    let c = f32::from(b) / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear RGB over its largest channel: what two lighting models leave alone.
fn chroma(lin: [f32; 3]) -> [f32; 3] {
    let m = lin[0].max(lin[1]).max(lin[2]).max(1e-6);
    lin.map(|c| c / m)
}

fn nearest(c: [f32; 3], palette: &[[f32; 3]]) -> usize {
    let d = |p: &[f32; 3]| (0..3).map(|i| (c[i] - p[i]).powi(2)).sum::<f32>();
    (0..palette.len())
        .min_by(|a, b| d(&palette[*a]).total_cmp(&d(&palette[*b])))
        .unwrap_or(0)
}

/// Packet H1b's placement oracle, `textured.rs`'s method on the hand: every visible face of the
/// held cube and of the goal (a segmentation id and a world-axis normal, borders dropped, at
/// least 200 of the 192x192 pixels) is classified per pixel against `block.png`'s six backdrop
/// colours and white in both images; the dominant backdrop must agree, and at least 85 % of the pixels.
/// At `qpos0` both cubes sit unrotated, so `top` sees their `+Z` faces and `front` and `side`
/// their `-Y` and `-X` faces.
#[test]
fn mujoco_puts_the_block_texture_where_we_do() {
    const PX: u32 = 192;
    let scene = scene();
    let tri = TriScene::from_scene(&scene).expect("the hand tessellates");
    let block = scene
        .assets
        .iter()
        .find(|a| a.name == "texture:object")
        .expect("the block texture");
    let block = scene.textures[&block.id].data.as_ref().expect("decoded");
    let mut palette: Vec<[f32; 3]> = (0..6)
        .map(|f| chroma(block.texel(f, 5, 5).map(|b| f32::from(b) / 255.0)))
        .collect();
    palette.push([1.0; 3]);
    let seg_of = |name: &str| *tri.names.iter().find(|(_, n)| *n == name).unwrap().0;
    let cubes = [seg_of("object"), seg_of("target")];
    let s = PX as usize;
    let images = match mujoco_render(PX) {
        Ok(images) => images,
        Err(why) => {
            println!("SKIP mujoco_puts_the_block_texture_where_we_do: {why}");
            return;
        }
    };
    let (mut faces, mut wrong) = (0, Vec::new());
    for (name, theirs) in CAMERAS.into_iter().zip(images) {
        let cam = view_at(&scene, name, PX);
        let ours = cpu::rasterize(
            &tri,
            &cam,
            &RenderConfig::rs(TileAtlasCfg::row(PX, PX, 1)),
            0,
        );
        let rgb = ours.tile(Channel::Rgb8).unwrap().as_u8().unwrap();
        let seg = ours
            .tile(Channel::SegmentationId)
            .unwrap()
            .as_u32()
            .unwrap();
        let normal = ours.tile(Channel::Normal).unwrap().as_f32().unwrap();
        if let Ok(dir) = std::env::var("ES_RENDER_DUMP") {
            let dir = PathBuf::from(dir);
            std::fs::write(dir.join(format!("{name}_placement_ours.rgb")), rgb).unwrap();
            std::fs::write(dir.join(format!("{name}_placement_mujoco.rgb")), &theirs).unwrap();
        }
        let axis = |i: usize| {
            let n = &normal[i * 3..i * 3 + 3];
            let w = cam.pose.orientation.rotate(Vec3::new(
                f64::from(n[0]),
                f64::from(n[1]),
                f64::from(n[2]),
            ));
            let c = [w.x, w.y, w.z];
            let k = (0..3)
                .max_by(|x, y| c[*x].abs().total_cmp(&c[*y].abs()))
                .unwrap();
            (k, c[k] > 0.0)
        };
        let interior = |i: usize| {
            let (x, y) = (i % s, i / s);
            x > 0
                && y > 0
                && x + 1 < s
                && y + 1 < s
                && [i - 1, i + 1, i - s, i + s]
                    .iter()
                    .all(|j| seg[*j] == seg[i] && axis(*j) == axis(i))
        };
        let mut regions: BTreeMap<(u32, usize, bool), Vec<usize>> = BTreeMap::new();
        for i in (0..s * s).filter(|i| cubes.contains(&seg[*i]) && interior(*i)) {
            let (k, pos) = axis(i);
            regions.entry((seg[i], k, pos)).or_default().push(i);
        }
        for ((id, k, pos), pixels) in &regions {
            let class = |lin: [f32; 3]| nearest(chroma(lin), &palette);
            let mine: Vec<usize> = pixels
                .iter()
                .map(|i| class([0, 1, 2].map(|c| srgb_decode(rgb[i * 3 + c]))))
                .collect();
            let mj: Vec<usize> = pixels
                .iter()
                .map(|i| class([0, 1, 2].map(|c| f32::from(theirs[i * 3 + c]) / 255.0)))
                .collect();
            // A face is its backdrop: the most frequent class other than the letters' white.
            let dominant = |c: &[usize]| {
                let mut n = [0usize; 6];
                c.iter().filter(|x| **x < 6).for_each(|x| n[*x] += 1);
                (0..6).max_by_key(|j| n[*j]).unwrap()
            };
            let agree =
                mine.iter().zip(&mj).filter(|(x, y)| x == y).count() as f64 / pixels.len() as f64;
            let face = format!(
                "{} {}{}",
                tri.names[id],
                if *pos { '+' } else { '-' },
                ['x', 'y', 'z'][*k]
            );
            println!(
                "{name}: {face}: {} px, dominant ours {} / MuJoCo {}, agreement {agree:.3}",
                pixels.len(),
                dominant(&mine),
                dominant(&mj)
            );
            if pixels.len() < 200 {
                // A face seen edge-on (the goal's -X, ~100 px, 0.64) is where MuJoCo's
                // trilinear mipmapping blurs the letters most (renderer.md 15.4).
                continue;
            }
            if dominant(&mine) != dominant(&mj) || agree < 0.85 {
                wrong.push(format!("{name}: {face}"));
            }
            faces += 1;
        }
    }
    assert!(wrong.is_empty(), "faces that disagree: {wrong:?}");
    assert!(faces >= 6, "only {faces} faces large enough to judge");
}
