//! Oracles for `es-render` (spec 1.4).
//!
//! * the CPU reference reproduces the golden files bit for bit (they were generated from it);
//! * the GPU rasterizer reproduces them too — `Rgb8` and `SegmentationId` bit for bit,
//!   `Depth32` to a measured ULP count;
//! * the GPU path tracer at 1 spp equals the CPU path tracer bit for bit;
//! * spec 15.3's channel agreement: `PT` and `RS` produce identical depth, segmentation and
//!   normal;
//! * two GPU renders of the same input are bit-identical (spec 3.5 tier 1, same device).
//!
//! Every GPU test prints `SKIP <test>: <reason>` and returns when no device or no `slangc` is
//! found, so the suite passes on a CI box without a GPU and says so.
//!
//! Goldens are regenerated only by `cargo test -p es-render -- --ignored generate_goldens`,
//! which runs the **CPU** path. Never the GPU path: a golden produced on one driver would
//! bake that driver's arithmetic into the repository.

use std::collections::BTreeSet;
use std::path::PathBuf;

use es_gpu::{Gpu, GpuOptions, SlangCompiler};
use es_render::cornell::{cornell_box, cornell_camera};
use es_render::{
    cpu, Atlas, CameraView, Frame, RenderConfig, RenderPath, Renderer, Tile, TileAtlasCfg, TriScene,
};
use es_sensor::Channel;

const TILE: u32 = 64;
const DEPTH: Channel = Channel::Depth32 { unit_m: 1.0 };

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/render")
}

fn scene() -> TriScene {
    TriScene::from_scene(&cornell_box()).expect("cornell box tessellates")
}

fn rs_cfg() -> RenderConfig {
    RenderConfig::rs(TileAtlasCfg::row(TILE, TILE, 1))
}

fn pt_cfg(spp: u32, bounces: u32) -> RenderConfig {
    RenderConfig::pt(TileAtlasCfg::row(TILE, TILE, 1), spp, bounces)
}

fn cpu_rs() -> Frame {
    cpu::rasterize(&scene(), &cornell_camera(TILE, TILE), &rs_cfg(), 0)
}

fn cpu_pt1() -> Frame {
    cpu::path_trace(&scene(), &cornell_camera(TILE, TILE), &pt_cfg(1, 2), 0)
}

/// `Some((gpu, _))` or a printed SKIP. `slangc` is checked too: without it no kernel compiles
/// and the failure would look like a GPU bug.
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

fn render_gpu<'g>(
    gpu: &'g Gpu,
    cfg: RenderConfig,
    cameras: &[CameraView],
    frames: u32,
) -> Atlas<'g> {
    let mut renderer = Renderer::new(gpu, cfg).expect("renderer");
    renderer.upload_tris(scene()).expect("upload");
    let mut atlas = renderer.render(cameras).expect("render");
    for _ in 1..frames {
        atlas = renderer.render(cameras).expect("render");
    }
    atlas
}

/// Max ULP distance between two `f32` slices, and the index where it happened.
fn max_ulp(a: &[f32], b: &[f32]) -> (u64, usize) {
    let mut worst = (0u64, 0usize);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let key = |v: f32| {
            let bits = i64::from(v.to_bits());
            if bits < 0 {
                i64::from(i32::MIN) - bits
            } else {
                bits
            }
        };
        let d = (key(*x) - key(*y)).unsigned_abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

/// Max absolute difference divided by the reference image's peak value.
///
/// Not a per-element relative error: most of a rendered image is near zero, and dividing a
/// 1e-7 difference by a 1e-7 pixel reports 100% error for an image nobody could tell apart.
/// The peak of the reference is the scale the eye and the encoder both work in.
fn max_normalized(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
        / scale
}

// --- goldens -------------------------------------------------------------------------------

struct Golden {
    name: &'static str,
    channel: Channel,
    dtype: &'static str,
}

const GOLDENS: [Golden; 4] = [
    Golden {
        name: "cornell_rs_rgb8",
        channel: Channel::Rgb8,
        dtype: "u8",
    },
    Golden {
        name: "cornell_rs_depth",
        channel: DEPTH,
        dtype: "f32",
    },
    Golden {
        name: "cornell_rs_seg",
        channel: Channel::SegmentationId,
        dtype: "u32",
    },
    Golden {
        name: "cornell_pt1spp",
        channel: Channel::PtRadiance,
        dtype: "f32",
    },
];

fn golden_tile(g: &Golden, rs: &Frame, pt: &Frame) -> Tile {
    let frame = if g.channel == Channel::PtRadiance {
        pt
    } else {
        rs
    };
    frame.tile(g.channel).expect("channel rendered").clone()
}

/// Regenerate `tests/golden/render/*`. Run once, from the **CPU** reference, then
/// `GOLDEN_UPDATE=1 cargo xtask verify-goldens`.
#[test]
#[ignore = "golden generator; run explicitly"]
fn generate_goldens() {
    let dir = golden_dir();
    std::fs::create_dir_all(&dir).expect("golden dir");
    let (rs, pt) = (cpu_rs(), cpu_pt1());
    for g in &GOLDENS {
        let tile = golden_tile(g, &rs, &pt);
        std::fs::write(dir.join(format!("{}.bin", g.name)), tile.to_bytes()).expect("write bin");
        let sidecar = serde_json::json!({
            "name": g.name,
            "dtype": g.dtype,
            "layout": "row-major, little-endian, tightly packed",
            "shape": tile.shape,
            "kernel": if g.channel == Channel::PtRadiance { "pt.v1 (1 spp, 2 bounces)" } else { "raster.v1" },
            "pins": "OpenCV camera frame and top-left image origin (spec 3.1); depth in metres, unit_m = 1",
            "oracle": "es_render::cpu, the pure-Rust mirror of the Slang kernels",
            "generator": "cargo test -p es-render -- --ignored generate_goldens",
            "spec": "docs/design/renderer.md",
        });
        std::fs::write(
            dir.join(format!("{}.json", g.name)),
            format!(
                "{}\n",
                serde_json::to_string_pretty(&sidecar).expect("json")
            ),
        )
        .expect("write json");
        println!("wrote {} ({} bytes)", g.name, tile.to_bytes().len());
    }
}

#[test]
fn cpu_reference_reproduces_the_goldens_bit_for_bit() {
    let (rs, pt) = (cpu_rs(), cpu_pt1());
    for g in &GOLDENS {
        let path = golden_dir().join(format!("{}.bin", g.name));
        let expected = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run the generate_goldens test)", path.display()));
        let got = golden_tile(g, &rs, &pt).to_bytes();
        assert_eq!(got.len(), expected.len(), "{} size", g.name);
        assert!(got == expected, "{} differs from its golden", g.name);
        println!("bit-equal CPU vs golden: {}", g.name);
    }
}

// --- GPU -------------------------------------------------------------------------------------

#[test]
fn gpu_rasterizer_matches_the_cpu_goldens() {
    let test = "gpu_rasterizer_matches_the_cpu_goldens";
    let Some(gpu) = open(test) else { return };
    let cams = [cornell_camera(TILE, TILE)];
    let mut atlas = render_gpu(&gpu, rs_cfg(), &cams, 1);
    let want = cpu_rs();

    let rgb = atlas.read_tile(0, Channel::Rgb8).expect("rgb");
    let want_rgb = want.tile(Channel::Rgb8).unwrap();
    let diff = rgb
        .as_u8()
        .unwrap()
        .iter()
        .zip(want_rgb.as_u8().unwrap())
        .filter(|(a, b)| a != b)
        .count();
    println!("Rgb8: {diff} of {} bytes differ", rgb.len());
    assert_eq!(diff, 0, "Rgb8 must be bit-equal to the golden");

    let seg = atlas.read_tile(0, Channel::SegmentationId).expect("seg");
    assert!(
        seg.as_u32().unwrap()
            == want
                .tile(Channel::SegmentationId)
                .unwrap()
                .as_u32()
                .unwrap(),
        "SegmentationId must be bit-equal to the golden"
    );
    println!("SegmentationId: bit-equal");

    let depth = atlas.read_tile(0, DEPTH).expect("depth");
    let (ulp, at) = max_ulp(
        depth.as_f32().unwrap(),
        want.tile(DEPTH).unwrap().as_f32().unwrap(),
    );
    println!("Depth32: max ULP {ulp} (at index {at})");
    assert!(ulp <= 1, "Depth32 max ULP {ulp} exceeds the documented 1");

    let normal = atlas.read_tile(0, Channel::Normal).expect("normal");
    let (nulp, _) = max_ulp(
        normal.as_f32().unwrap(),
        want.tile(Channel::Normal).unwrap().as_f32().unwrap(),
    );
    println!("Normal: max ULP {nulp}");
    assert!(nulp <= 1, "Normal max ULP {nulp} exceeds 1");
}

#[test]
fn gpu_renders_are_bit_identical_across_runs() {
    let test = "gpu_renders_are_bit_identical_across_runs";
    let Some(gpu) = open(test) else { return };
    let cams = [cornell_camera(TILE, TILE)];
    let mut a = render_gpu(&gpu, rs_cfg(), &cams, 1);
    let mut b = render_gpu(&gpu, rs_cfg(), &cams, 1);
    for channel in [
        Channel::Rgb8,
        DEPTH,
        Channel::SegmentationId,
        Channel::Normal,
    ] {
        let (x, y) = (
            a.read_tile(0, channel).expect("a"),
            b.read_tile(0, channel).expect("b"),
        );
        assert!(
            x.to_bytes() == y.to_bytes(),
            "{channel:?} differs between runs"
        );
        println!("bit-identical across runs: {channel:?}");
    }
}

#[test]
fn gpu_path_tracer_matches_the_cpu_reference_at_1spp() {
    let test = "gpu_path_tracer_matches_the_cpu_reference_at_1spp";
    let Some(gpu) = open(test) else { return };
    let cams = [cornell_camera(TILE, TILE)];
    let mut atlas = render_gpu(&gpu, pt_cfg(1, 2), &cams, 1);
    let want = cpu_pt1();
    let got = atlas.read_tile(0, Channel::PtRadiance).expect("radiance");
    let expected = want.tile(Channel::PtRadiance).unwrap();
    let (ulp, at) = max_ulp(got.as_f32().unwrap(), expected.as_f32().unwrap());
    let norm = max_normalized(got.as_f32().unwrap(), expected.as_f32().unwrap());
    println!("PtRadiance 1 spp: max ULP {ulp} (index {at}), normalized {norm:e}");
    assert!(
        got.to_bytes() == expected.to_bytes(),
        "PtRadiance at 1 spp must be bit-equal (max ULP {ulp}, normalized {norm:e})"
    );
}

/// Spec 15.3: the render paths agree on depth, segmentation and normal, bit for bit.
#[test]
fn gpu_pt_and_rs_agree_on_geometry() {
    let test = "gpu_pt_and_rs_agree_on_geometry";
    let Some(gpu) = open(test) else { return };
    let cams = [cornell_camera(TILE, TILE)];
    let mut rs = render_gpu(&gpu, rs_cfg(), &cams, 1);
    let mut pt = render_gpu(&gpu, pt_cfg(2, 2), &cams, 1);
    for channel in [DEPTH, Channel::SegmentationId, Channel::Normal] {
        let (a, b) = (
            rs.read_tile(0, channel).expect("rs"),
            pt.read_tile(0, channel).expect("pt"),
        );
        assert!(
            a.to_bytes() == b.to_bytes(),
            "{channel:?} differs between RS and PT"
        );
        println!("PT/RS bit-equal: {channel:?}");
    }
}

#[test]
fn gpu_restir_and_svgf_match_the_cpu_within_tolerance() {
    let test = "gpu_restir_and_svgf_match_the_cpu_within_tolerance";
    let Some(gpu) = open(test) else { return };
    let mut cfg = pt_cfg(1, 2);
    cfg.path = RenderPath::Pt {
        spp: 1,
        bounces: 2,
        restir: true,
        svgf: true,
    };
    let cams = [cornell_camera(TILE, TILE)];
    let mut atlas = render_gpu(&gpu, cfg.clone(), &cams, 1);
    let want = cpu::path_trace(&scene(), &cams[0], &cfg, 0);
    let got = atlas.read_tile(0, Channel::PtRadiance).expect("radiance");
    let want_f = want.tile(Channel::PtRadiance).unwrap().as_f32().unwrap();
    let norm = max_normalized(got.as_f32().unwrap(), want_f);
    let (ulp, _) = max_ulp(got.as_f32().unwrap(), want_f);
    println!("ReSTIR+SVGF: normalized max error {norm:e}, max ULP {ulp} (tolerance 1e-5)");
    assert!(norm <= 1e-5, "ReSTIR/SVGF diverged by {norm:e}");
}

/// Spec 15.2: several cameras in one atlas, each tile equal to rendering that camera alone.
#[test]
fn gpu_atlas_packs_several_cameras() {
    let test = "gpu_atlas_packs_several_cameras";
    let Some(gpu) = open(test) else { return };
    let mut cams = Vec::new();
    for i in 0..3u32 {
        let mut cam = cornell_camera(TILE, TILE);
        cam.pose.position.z += f64::from(i) * 0.25;
        cams.push(cam);
    }
    let atlas_cfg = TileAtlasCfg {
        tile_w: TILE,
        tile_h: TILE,
        tiles_per_row: 2,
        n_tiles: 4, // one padding tile, on purpose
    };
    let mut cfg = RenderConfig::rs(atlas_cfg);
    cfg.channels = BTreeSet::from([Channel::Rgb8, Channel::SegmentationId]);
    let mut atlas = render_gpu(&gpu, cfg.clone(), &cams, 1);
    assert_eq!(atlas.layout().rows, 2);
    for (i, cam) in cams.iter().enumerate() {
        let single = cpu::rasterize(&scene(), cam, &cfg, i as u32);
        let tile = atlas.read_tile(i as u32, Channel::Rgb8).expect("tile");
        assert!(
            tile.to_bytes() == single.tile(Channel::Rgb8).unwrap().to_bytes(),
            "tile {i} does not match the single-camera render"
        );
    }
    println!("3 cameras + 1 padding tile: every tile bit-equal to its own render");
    assert!(
        atlas.read_tile(3, Channel::Rgb8).is_err(),
        "tile 3 has no view"
    );
}

// --- profile (packet M7/R1 step 0) -----------------------------------------------------------

/// The demo scene: the SO-101 pick-and-place cell `es video showcase` renders.
fn so101() -> es_assets::scene::SceneDesc {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/mjcf/so101_pick_place.xml");
    let xml = std::fs::read_to_string(&path).expect("the SO-101 fixture");
    es_assets::mjcf::parse_str(&xml)
        .expect("the SO-101 fixture parses")
        .scene
}

/// `es video showcase`'s free camera at the angle the acceptance uses, built here rather than
/// imported: `es_env::render::look_at` is layer 9 and this crate is layer 5 (spec 4.2).
fn showcase_camera(width: u32, height: u32) -> CameraView {
    use es_math::{Pose, Quat, Vec3};
    let eye = Vec3::new(0.66, -0.46, 0.52);
    let target = Vec3::new(0.14, -0.04, 0.04);
    let forward = (target - eye).normalize();
    let x = forward.cross(Vec3::new(0.0, 0.0, 1.0)).normalize();
    let y = forward.cross(x);
    let z = forward;
    let s = (x.x + y.y + z.z + 1.0).sqrt() * 2.0;
    let quat = Quat::from_xyzw((y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, 0.25 * s);
    CameraView {
        pose: Pose::new(eye, quat),
        spec: es_render::ImageSpec::pinhole(width, height, 36f64.to_radians()),
    }
}

fn median_p95(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[(v.len() * 95) / 100])
}

/// The four phases of one frame, 100 frames, median and p95, at the two sizes that matter:
/// the showcase (1280x720) and one observation frame (96x96).
///
/// `upload` is `Renderer::upload_tris`; `dispatch+wait` is `Renderer::render`, which also
/// uploads the parameter buffer and blocks on the fence; `readback` is `Atlas::read_tile`.
/// Reported with spec 12.4's `camera_frames_per_sec` and `pixels_per_sec`, never a single
/// `step/s`. Run with
/// `cargo test -p es-render --release -- --ignored --nocapture frame_profile`.
#[test]
#[ignore = "timing; run explicitly"]
fn frame_profile() {
    let test = "frame_profile";
    let Some(gpu) = open(test) else { return };
    let scene = so101();
    let world = std::collections::BTreeMap::new();
    let n_tri = TriScene::from_scene(&scene)
        .expect("tessellates")
        .tris
        .len();
    for (w, h) in [(1280u32, 720u32), (96, 96)] {
        let cfg = RenderConfig::rs(TileAtlasCfg::row(w, h, 1));
        let mut renderer = Renderer::new(&gpu, cfg).expect("renderer");
        let cams = [showcase_camera(w, h)];
        let (mut tess, mut up, mut disp, mut read) = (vec![], vec![], vec![], vec![]);
        for _ in 0..100 {
            let t0 = std::time::Instant::now();
            let tri = TriScene::from_scene_with_poses(&scene, &world).expect("tessellates");
            let t1 = std::time::Instant::now();
            renderer.upload_tris(tri).expect("upload");
            let t2 = std::time::Instant::now();
            let mut atlas = renderer.render(&cams).expect("render");
            let t3 = std::time::Instant::now();
            let tile = atlas.read_tile(0, Channel::Rgb8).expect("readback");
            let t4 = std::time::Instant::now();
            assert_eq!(tile.len(), (w as usize) * (h as usize) * 3);
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
            tess.push(ms(t0, t1));
            up.push(ms(t1, t2));
            disp.push(ms(t2, t3));
            read.push(ms(t3, t4));
        }
        let total: Vec<f64> = (0..tess.len())
            .map(|i| tess[i] + up[i] + disp[i] + read[i])
            .collect();
        println!(
            "\n{w}x{h}, {n_tri} triangles, 100 frames, {}",
            gpu.capabilities().device_name
        );
        println!("| phase | median ms | p95 ms |");
        println!("|---|---|---|");
        for (name, v) in [
            ("tessellate", &tess),
            ("upload", &up),
            ("dispatch+wait", &disp),
            ("readback", &read),
            ("frame total", &total),
        ] {
            let (med, p95) = median_p95(v.clone());
            println!("| {name} | {med:.3} | {p95:.3} |");
        }
        let (med, _) = median_p95(total);
        println!(
            "camera_frames_per_sec {:.1}, pixels_per_sec {:.3e} (median frame)",
            1e3 / med,
            f64::from(w) * f64::from(h) * 1e3 / med
        );
        // Where the readback goes. `Buffer::download` on device-local memory allocates a
        // host-visible staging buffer, copies into it and reads it back with `to_vec`;
        // host-visible memory is write-combined, so the *read* is most of the cost. This
        // measures that read alone, on a buffer of the same size. `crates/es-gpu/**` is out
        // of this packet's scope, so the number is recorded, not fixed.
        let bytes = u64::from(w) * u64::from(h) * 4;
        let mut staging =
            es_gpu::Buffer::new(&gpu, bytes, es_gpu::Usage::Staging).expect("staging buffer");
        let t = std::time::Instant::now();
        let got = staging.download().expect("host-visible download");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "host-visible read of {} KiB: {ms:.3} ms ({:.0} MiB/s)",
            bytes / 1024,
            got.len() as f64 / ms / 1048.576
        );
    }
}
