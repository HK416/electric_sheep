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

// --- cached tessellation and the BVH (packet M7/R1) ------------------------------------------

/// Three "ticks" of body poses for `scene`: the home pose, then two hand-set rigid motions
/// applied to every body. `.estraj` replay hands `from_scene_with_poses` exactly this shape of
/// map, so re-posing here exercises the same path with no `es-env` dependency (layer 5).
fn ticks(
    scene: &es_assets::scene::SceneDesc,
) -> Vec<std::collections::BTreeMap<es_core::StableId, es_math::Pose>> {
    use es_math::{Pose, Quat, Vec3};
    let mut out = vec![std::collections::BTreeMap::new()];
    for (k, angle) in [(1.0, 0.37f64), (2.0, -0.91)] {
        let mut m = std::collections::BTreeMap::new();
        for (i, body) in scene.bodies.iter().enumerate() {
            let s = (i as f64) * 0.013 + k * 0.05;
            let (sin, cos) = (angle * 0.5).sin_cos();
            m.insert(
                body.id,
                Pose::new(
                    Vec3::new(s, -s * 0.5, s * 0.25),
                    Quat::from_xyzw(sin * 0.6, sin * 0.8, 0.0, cos).normalize(),
                ),
            );
        }
        out.push(m);
    }
    out
}

/// Packet M7/R1 oracle 2: a `SceneCache` reused across ticks and scenes produces exactly the
/// `TriScene` a per-call tessellation does — every `f32` bitwise, the light list, the names.
///
/// The cache is warmed on Cornell first and then used for the SO-101 cell, so the `Shape`
/// guard (geom ids come from names, and two scenes can share one) is exercised too.
#[test]
fn cached_tessellation_is_bit_identical() {
    let mut cache = es_render::SceneCache::default();
    for scene in [cornell_box(), so101()] {
        for (tick, world) in ticks(&scene).iter().enumerate() {
            let cached = cache.tri_scene(&scene, world).expect("cached");
            let fresh = TriScene::from_scene_with_poses(&scene, world).expect("fresh");
            assert_eq!(
                cached.names, fresh.names,
                "{} tick {tick}: names",
                scene.name
            );
            assert_eq!(
                cached.lights, fresh.lights,
                "{} tick {tick}: light list",
                scene.name
            );
            assert_eq!(
                cached.tris.len(),
                fresh.tris.len(),
                "{} tick {tick}: triangle count",
                scene.name
            );
            // Bitwise on the floats, not `==` on `f32`: the point of the oracle is that no
            // vertex, normal, albedo or emission moved by one ULP.
            let (a, b) = (cached.to_floats(), fresh.to_floats());
            let diff = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                diff,
                0,
                "{} tick {tick}: {diff} of {} floats differ",
                scene.name,
                a.len()
            );
        }
        println!(
            "cached == uncached, bitwise, 3 ticks of {} ({} geoms)",
            scene.name,
            scene.bodies.iter().map(|b| b.geoms.len()).sum::<usize>()
        );
    }
}

/// Packet M7/R1 oracle 3: over 10,000 deterministic rays per scene, the BVH descent returns
/// the flat scan's `Option<Hit>` — bitwise on `t`, the same triangle index — and the any-hit
/// descent returns the flat shadow scan's boolean.
///
/// Rays are counter-based (`es_render::rng`, spec 3.4: no global RNG): origins on a sphere
/// around the scene's centre aimed back through a jittered point near it, so they cross the
/// geometry from every direction rather than sampling one camera's frustum.
#[test]
fn bvh_traversal_is_the_flat_scan() {
    use es_render::bvh::Bvh;
    use es_render::rng;
    for (name, tri) in [
        (
            "cornell",
            TriScene::from_scene(&cornell_box()).expect("cornell"),
        ),
        ("so101", TriScene::from_scene(&so101()).expect("so101")),
    ] {
        // Scene bounds, to aim the rays at something.
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for t in &tri.tris {
            for v in &t.v {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
        }
        let mid = [
            f32::midpoint(lo[0], hi[0]),
            f32::midpoint(lo[1], hi[1]),
            f32::midpoint(lo[2], hi[2]),
        ];
        let radius = (0..3).fold(0.0f32, |m, k| m.max(hi[k] - lo[k])) + 1.0;

        let bvh = Bvh::build(&tri.tris);
        let (mut hits, mut shadows) = (0u32, 0u32);
        for r in 0..10_000u32 {
            let key = rng::key(0x5eed, 0, r, r ^ 0x9e37, 0, 0, 0);
            let cos_theta = rng::uniform(key, 0) * 2.0 - 1.0;
            let phi = rng::uniform(key, 1) * std::f32::consts::TAU;
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let origin = [
                mid[0] + radius * sin_theta * phi.cos(),
                mid[1] + radius * sin_theta * phi.sin(),
                mid[2] + radius * cos_theta,
            ];
            // Aim back through a point inside the scene box, so most rays actually hit.
            let aim = [
                lo[0] + (hi[0] - lo[0]) * rng::uniform(key, 2),
                lo[1] + (hi[1] - lo[1]) * rng::uniform(key, 3),
                lo[2] + (hi[2] - lo[2]) * rng::uniform(key, 4),
            ];
            let dir = [aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]];
            let (near, far) = (0.01f32, 1e3f32);

            let got = cpu::nearest_hit(&tri.tris, &bvh, origin, dir, near, far);
            let want = cpu::nearest_hit_flat(&tri.tris, origin, dir, near, far);
            match (got, want) {
                (Some(tree), Some(scan)) => {
                    assert_eq!(
                        (tree.t.to_bits(), tree.tri),
                        (scan.t.to_bits(), scan.tri),
                        "{name} ray {r}: BVH {tree:?} vs flat scan {scan:?}"
                    );
                    hits += 1;
                }
                (None, None) => {}
                (tree, scan) => panic!("{name} ray {r}: BVH {tree:?} vs flat scan {scan:?}"),
            }
            let got_any = cpu::any_hit(&tri.tris, &bvh, origin, dir, near, far);
            assert_eq!(
                got_any,
                cpu::any_hit_flat(&tri.tris, origin, dir, near, far),
                "{name} ray {r}: any-hit disagrees"
            );
            shadows += u32::from(got_any);
        }
        assert!(
            hits > 1_000,
            "{name}: only {hits} of 10000 rays hit anything"
        );
        assert!(
            bvh.depth <= es_render::bvh::STACK,
            "{name}: tree depth {} exceeds the {} traversal stack",
            bvh.depth,
            es_render::bvh::STACK
        );
        println!(
            "{name}: {} triangles, {} nodes, max traversal depth {} (stack {}), \
             10000 rays, {hits} nearest hits, {shadows} any-hits, all equal to the flat scan",
            tri.tris.len(),
            bvh.nodes.len(),
            bvh.depth,
            es_render::bvh::STACK
        );
    }
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
    let fwd = (target - eye).normalize();
    let right = fwd.cross(Vec3::new(0.0, 0.0, 1.0)).normalize();
    let down = fwd.cross(right);
    // Shepperd's positive-trace branch; the showcase camera never looks along world up.
    let den = (right.x + down.y + fwd.z + 1.0).sqrt() * 2.0;
    let quat = Quat::from_xyzw(
        (down.z - fwd.y) / den,
        (fwd.x - right.z) / den,
        (right.y - down.x) / den,
        0.25 * den,
    );
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
        let mut cache = es_render::SceneCache::default();
        renderer
            .upload_tris(cache.tri_scene(&scene, &world).expect("tessellates"))
            .expect("upload");

        // Sustained rendering, no readback: 100 `render()` calls back to back, after a
        // warm-up of 30. This is the renderer's own cost and the number that reproduces.
        // The four-phase loop below spends most of its wall clock inside an uncached host
        // read, and an idle GPU on this machine drops to P8 and **PCIe gen 1**
        // (`nvidia-smi --query-gpu=pstate,pcie.link.gen.current`), which slows every phase
        // of the next frame — so an end-to-end total measured on a cold card is a
        // measurement of the driver's power policy, not of the renderer (spec 28.9 rule 3).
        // The warm-up is what makes the before/after tables comparable.
        let mut sustained = Vec::new();
        for i in 0..130 {
            let t = std::time::Instant::now();
            let _ = renderer.render(&cams).expect("render");
            if i >= 30 {
                sustained.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }

        let (mut tess, mut up, mut disp, mut read) = (vec![], vec![], vec![], vec![]);
        for _ in 0..100 {
            let t0 = std::time::Instant::now();
            let tri = cache.tri_scene(&scene, &world).expect("tessellates");
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
        let (smed, sp95) = median_p95(sustained);
        let total: Vec<f64> = (0..tess.len())
            .map(|i| tess[i] + up[i] + disp[i] + read[i])
            .collect();
        println!(
            "\n{w}x{h}, {n_tri} triangles, 100 frames, {}",
            gpu.capabilities().device_name
        );
        println!("| phase | median ms | p95 ms |");
        println!("|---|---|---|");
        println!("| render(), sustained, no readback | {smed:.3} | {sp95:.3} |");
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
        println!(
            "sustained: camera_frames_per_sec {:.1}, pixels_per_sec {:.3e}",
            1e3 / smed,
            f64::from(w) * f64::from(h) * 1e3 / smed
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
