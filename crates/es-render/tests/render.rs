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
//!
//! `d`, `n`, `p`, `t` are ray, normal, point and ray parameter here as they are in `cpu.rs`
//! and in the Slang; a test that re-derives a pixel by hand has to read like the code it
//! checks, so the lint is off for this file too.
#![allow(clippy::many_single_char_names)]

use std::collections::BTreeSet;
use std::path::PathBuf;

use es_gpu::{Gpu, GpuOptions, SlangCompiler};
use es_render::cornell::{cornell_box, cornell_camera};
use es_render::{
    cpu, Atlas, CameraView, Frame, RenderConfig, RenderPath, Renderer, Shading, Temporal, Tile,
    TileAtlasCfg, TriScene,
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

/// The opt-in look of packet M7/R2: shadows, hemisphere ambient, Blinn-Phong, `ssaa: 2`.
fn rs_full_cfg() -> RenderConfig {
    RenderConfig::rs_full(TileAtlasCfg::row(TILE, TILE, 1))
}

fn cpu_rs_full() -> Frame {
    cpu::rasterize(&scene(), &cornell_camera(TILE, TILE), &rs_full_cfg(), 0)
}

/// The `rs_full` preset with `shadows` and `ssaa` overridden — the two knobs the tests move.
fn rs_full_with(shadows: bool, ssaa: u32) -> RenderConfig {
    let mut cfg = rs_full_cfg();
    let Shading::Full {
        specular,
        shininess,
        sky_rgb,
        ground_rgb,
        ..
    } = cfg.shading
    else {
        unreachable!("rs_full is Full")
    };
    cfg.shading = Shading::Full {
        shadows,
        specular,
        shininess,
        sky_rgb,
        ground_rgb,
        ssaa,
    };
    cfg
}

fn cpu_pt1() -> Frame {
    cpu::path_trace(&scene(), &cornell_camera(TILE, TILE), &pt_cfg(1, 2), 0)
}

/// The config `cornell_pt_nee_rgb8` is generated from (packet M7/R3): 4 spp, 3 bounces, NEE
/// on, Reinhard, exposure 1 — every one of those the `RenderConfig` default except `nee`.
fn pt_nee_cfg() -> RenderConfig {
    RenderConfig::pt_nee(TileAtlasCfg::row(TILE, TILE, 1), 4, 3)
}

fn cpu_pt_nee() -> Frame {
    cpu::path_trace(&scene(), &cornell_camera(TILE, TILE), &pt_nee_cfg(), 0)
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
    /// Which render the tile comes from, and the `kernel` line of the sidecar.
    source: Source,
    kernel: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Rs,
    RsFull,
    Pt,
    PtNee,
    PtAccum8,
}

const GOLDENS: [Golden; 7] = [
    Golden {
        name: "cornell_rs_rgb8",
        channel: Channel::Rgb8,
        dtype: "u8",
        source: Source::Rs,
        kernel: "raster.v1",
    },
    Golden {
        name: "cornell_rs_depth",
        channel: DEPTH,
        dtype: "f32",
        source: Source::Rs,
        kernel: "raster.v1",
    },
    Golden {
        name: "cornell_rs_seg",
        channel: Channel::SegmentationId,
        dtype: "u32",
        source: Source::Rs,
        kernel: "raster.v1",
    },
    Golden {
        name: "cornell_pt1spp",
        channel: Channel::PtRadiance,
        dtype: "f32",
        source: Source::Pt,
        kernel: "pt.v1 (1 spp, 2 bounces)",
    },
    // Packet M7/R2. Only `Rgb8`: `Full` writes the same depth/seg/normal buffers `Lambert`
    // does (the centre ray's), and `cpu_full_shading_reproduces_its_golden` asserts it
    // instead of committing three more files that would say the same thing.
    Golden {
        name: "cornell_rs_full_rgb8",
        channel: Channel::Rgb8,
        dtype: "u8",
        source: Source::RsFull,
        kernel: "raster.v1 (Shading::Full, ssaa 2)",
    },
    // Packet M7/R3: the first `Rgb8` the path tracer can produce. `PtRadiance` is not pinned
    // a second time — `cornell_pt1spp` already pins the estimator, and this file pins what
    // the tone map does to it.
    Golden {
        name: "cornell_pt_nee_rgb8",
        channel: Channel::Rgb8,
        dtype: "u8",
        source: Source::PtNee,
        kernel: "pt.v2 (4 spp, 3 bounces, NEE, Reinhard, exposure 1)",
    },
    // Packet M7/R4: eight frames of one sample each, accumulated for a still camera. Pinning
    // the `Rgb8` pins the whole chain — the sample indexing, the running sum and the single
    // divide — because `accumulation_of_n_frames_is_n_spp` asserts these bytes are also one
    // 8 spp frame's.
    Golden {
        name: "cornell_pt_accum8_rgb8",
        channel: Channel::Rgb8,
        dtype: "u8",
        source: Source::PtAccum8,
        kernel: "pt.v3 (8 frames x 1 spp, 3 bounces, NEE, Reinhard, exposure 1, max_history 8)",
    },
];

fn golden_tile(
    g: &Golden,
    rs: &Frame,
    rs_full: &Frame,
    pt: &Frame,
    pt_nee: &Frame,
    pt_accum8: &Frame,
) -> Tile {
    let frame = match g.source {
        Source::Rs => rs,
        Source::RsFull => rs_full,
        Source::Pt => pt,
        Source::PtNee => pt_nee,
        Source::PtAccum8 => pt_accum8,
    };
    frame.tile(g.channel).expect("channel rendered").clone()
}

/// Regenerate `tests/golden/render/*`. Run once, from the **CPU** reference, then
/// `GOLDEN_UPDATE=1 cargo xtask verify-goldens`.
///
/// `ES_GENERATE_GOLDENS=1` is required on top of `--ignored`: `cargo test -- --ignored` is a
/// thing people run to see the measurement tests, and a golden generator that rewrites the
/// repository's oracle as a side effect of that is a foot-gun (spec 1.4: goldens are CI
/// read-only).
#[test]
#[ignore = "golden generator; run explicitly"]
fn generate_goldens() {
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_goldens: set ES_GENERATE_GOLDENS=1 to rewrite the goldens");
        return;
    }
    let dir = golden_dir();
    std::fs::create_dir_all(&dir).expect("golden dir");
    let (rs, rs_full, pt, pt_nee) = (cpu_rs(), cpu_rs_full(), cpu_pt1(), cpu_pt_nee());
    let pt_accum8 = cpu_pt_accum8();
    for g in &GOLDENS {
        let tile = golden_tile(g, &rs, &rs_full, &pt, &pt_nee, &pt_accum8);
        std::fs::write(dir.join(format!("{}.bin", g.name)), tile.to_bytes()).expect("write bin");
        let sidecar = serde_json::json!({
            "name": g.name,
            "dtype": g.dtype,
            "layout": "row-major, little-endian, tightly packed",
            "shape": tile.shape,
            "kernel": g.kernel,
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
    let (rs, rs_full, pt, pt_nee) = (cpu_rs(), cpu_rs_full(), cpu_pt1(), cpu_pt_nee());
    let pt_accum8 = cpu_pt_accum8();
    for g in &GOLDENS {
        let path = golden_dir().join(format!("{}.bin", g.name));
        let expected = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run the generate_goldens test)", path.display()));
        let got = golden_tile(g, &rs, &rs_full, &pt, &pt_nee, &pt_accum8).to_bytes();
        assert_eq!(got.len(), expected.len(), "{} size", g.name);
        assert!(got == expected, "{} differs from its golden", g.name);
        println!("bit-equal CPU vs golden: {}", g.name);
    }
}

// --- the Rs look (packet M7/R2) ---------------------------------------------------------------

/// The three geometry channels of two frames, bit for bit.
fn same_geometry(a: &Frame, b: &Frame, what: &str) {
    for channel in [DEPTH, Channel::SegmentationId, Channel::Normal] {
        let (x, y) = (
            a.tile(channel).expect("channel a"),
            b.tile(channel).expect("channel b"),
        );
        assert!(
            x.to_bytes() == y.to_bytes(),
            "{what}: {channel:?} is not bit-identical"
        );
    }
    println!("{what}: Depth32, SegmentationId and Normal all bit-identical");
}

/// Oracle 2, and spec 28.10 rule 1 stated at the API: the default look is `Lambert`, and a
/// render through the new branch is the committed golden byte for byte.
#[test]
fn lambert_is_the_default_and_is_byte_identical() {
    assert_eq!(Shading::default(), Shading::Lambert);
    assert_eq!(rs_cfg().shading, Shading::Lambert);
    assert_eq!(
        RenderConfig::pt(rs_cfg().atlas, 1, 2).shading,
        Shading::Lambert
    );
    assert_eq!(rs_full_cfg().shading, Shading::FULL);

    let got = cpu_rs().tile(Channel::Rgb8).expect("rgb").to_bytes();
    let want = std::fs::read(golden_dir().join("cornell_rs_rgb8.bin")).expect("the golden");
    assert!(
        got == want,
        "the default look moved: Rgb8 differs from its golden"
    );
    println!("Shading::Lambert is the default and reproduces cornell_rs_rgb8 byte for byte");
}

/// Oracle 3, CPU half: the `Full` look reproduces its own golden, and its geometry channels
/// are `Lambert`'s — they come from the centre ray whatever the shading is, which is why no
/// second depth/seg/normal golden exists.
#[test]
fn cpu_full_shading_reproduces_its_golden() {
    let full = cpu_rs_full();
    let got = full.tile(Channel::Rgb8).expect("rgb").to_bytes();
    let want = std::fs::read(golden_dir().join("cornell_rs_full_rgb8.bin"))
        .expect("the golden (run the generate_goldens test)");
    assert_eq!(got.len(), want.len(), "cornell_rs_full_rgb8 size");
    assert!(got == want, "cornell_rs_full_rgb8 differs from its golden");

    let lambert = cpu_rs();
    same_geometry(&full, &lambert, "CPU Full vs Lambert");
    let changed = got
        .iter()
        .zip(lambert.tile(Channel::Rgb8).unwrap().to_bytes())
        .filter(|(a, b)| **a != *b)
        .count();
    assert!(
        changed > got.len() / 10,
        "only {changed} of {} bytes differ: the Full look is barely doing anything",
        got.len()
    );
    println!(
        "cornell_rs_full_rgb8 bit-equal to its golden; {changed} of {} bytes differ from Lambert",
        got.len()
    );
}

/// Oracle 4: a shadow ray can only darken, and only where something stands between the hit
/// point and the light. `ssaa: 1` so one pixel is one ray and the claim is checkable per
/// pixel.
#[test]
fn a_shadow_ray_darkens_only_occluded_pixels() {
    use es_render::bvh::Bvh;
    let (scene, cam) = (scene(), cornell_camera(TILE, TILE));
    let (on, off) = (rs_full_with(true, 1), rs_full_with(false, 1));
    let lit = cpu::rasterize(&scene, &cam, &off, 0);
    let shadowed = cpu::rasterize(&scene, &cam, &on, 0);
    let (a, b) = (
        lit.tile(Channel::Rgb8).unwrap().as_u8().unwrap().to_vec(),
        shadowed
            .tile(Channel::Rgb8)
            .unwrap()
            .as_u8()
            .unwrap()
            .to_vec(),
    );

    let vp = es_render::ViewParams::new(&cam);
    let bvh = Bvh::build(&scene.tris);
    let light = [
        on.light_dir.x as f32,
        on.light_dir.y as f32,
        on.light_dir.z as f32,
    ];
    let mut changed = 0;
    for py in 0..TILE {
        for px in 0..TILE {
            let i = (py * TILE + px) as usize;
            if a[i * 3..i * 3 + 3] == b[i * 3..i * 3 + 3] {
                continue;
            }
            changed += 1;
            for c in 0..3 {
                assert!(
                    b[i * 3 + c] <= a[i * 3 + c],
                    "pixel ({px}, {py}) channel {c} got brighter with shadows on"
                );
            }
            let d = cpu::primary_dir(&vp, px, py);
            let hit = cpu::nearest_hit(&scene.tris, &bvh, vp.pos, d, vp.near, vp.far)
                .unwrap_or_else(|| panic!("pixel ({px}, {py}) changed but hits nothing"));
            let tri = &scene.tris[hit.tri as usize];
            let n = cpu::face_forward(tri, d);
            // The hit point pushed off the surface, exactly as `shade_full` does it.
            let p = [
                vp.pos[0] + d[0] * hit.t + n[0] * 1e-4,
                vp.pos[1] + d[1] * hit.t + n[1] * 1e-4,
                vp.pos[2] + d[2] * hit.t + n[2] * 1e-4,
            ];
            assert!(
                cpu::any_hit(&scene.tris, &bvh, p, light, 0.0, 1e30),
                "pixel ({px}, {py}) darkened but nothing occludes it"
            );
        }
    }
    assert!(changed > 0, "the shadow ray changed no pixel at all");
    println!(
        "{changed} of {} pixels darkened, every one of them occluded",
        TILE * TILE
    );
}

/// The shading mix is energy-conserving (amended at review): a surface in full light returns
/// its albedo, a shadowed one returns `albedo * hemi`, and neither can exceed the albedo.
///
/// This is not covered by `cornell_rs_full_rgb8`: the Cornell box is a closed room, so every
/// shadow ray hits the ceiling and the golden's `diffuse` term is zero at every pixel — the
/// golden pins the hemisphere and the shadow, and *nothing* in it would notice the mix
/// changing. Hence a direct test of `shade_full`, which is the function both texts mirror.
#[test]
fn full_shading_is_energy_conserving() {
    use es_render::bvh::Bvh;
    let mut cfg = rs_full_with(true, 1);
    let Shading::Full {
        sky_rgb,
        ground_rgb,
        shininess,
        ..
    } = cfg.shading
    else {
        unreachable!("rs_full is Full")
    };
    // No highlight: `spec` is additive on purpose and would mask the term under test.
    cfg.shading = Shading::Full {
        shadows: true,
        specular: 0.0,
        shininess,
        sky_rgb,
        ground_rgb,
        ssaa: 1,
    };
    let light = [
        cfg.light_dir.x as f32,
        cfg.light_dir.y as f32,
        cfg.light_dir.z as f32,
    ];
    let albedo = [0.725, 0.71, 0.68];
    let tri = es_render::Tri {
        v: [[0.0; 3]; 3],
        n: light,
        albedo,
        emission: [0.0; 3],
        seg: 1,
    };
    let hemi = {
        // The same two operations `shade_full` and `common.slang` use, not `f32::midpoint`.
        #[allow(clippy::manual_midpoint)]
        let t = (light[2] + 1.0) * 0.5;
        [0, 1, 2].map(|c| ground_rgb[c] + (sky_rgb[c] - ground_rgb[c]) * t)
    };
    // The old form `albedo * (hemi + diffuse)` clipped exactly here: hemi alone is 0.79 and
    // the surface faces the light, so the sum passed 1 and every channel saturated.
    assert!(
        hemi[2] + 1.0 > 1.0,
        "the case under test must be one the additive form would have clipped"
    );

    // Lit: nothing to occlude it (an empty scene), so `vis = 1` and `diffuse = dot(n, L) = 1`.
    let empty: [es_render::Tri; 0] = [];
    let lit = cpu::shade_full(
        &empty,
        &Bvh::build(&empty),
        &tri,
        light,
        [0.0; 3],
        [0.0, 0.0, 1.0],
        &cfg,
    );
    for c in 0..3 {
        assert!(
            (lit[c] - albedo[c]).abs() <= 1e-6,
            "a fully lit surface returned {} on channel {c}, not its albedo {}",
            lit[c],
            albedo[c]
        );
    }

    // Shadowed: any point inside the closed Cornell room is, towards this light.
    let sc = scene();
    let bvh = Bvh::build(&sc.tris);
    let dark = cpu::shade_full(
        &sc.tris,
        &bvh,
        &tri,
        light,
        [1.5, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        &cfg,
    );
    for c in 0..3 {
        let want = albedo[c] * hemi[c];
        assert!(
            (dark[c] - want).abs() <= 1e-6,
            "a shadowed surface returned {} on channel {c}, not albedo * hemi {want}",
            dark[c]
        );
        assert!(dark[c] < lit[c], "the shadow must darken channel {c}");
    }
    println!(
        "lit {lit:?} == albedo {albedo:?}; shadowed {dark:?} == albedo * hemi {:?}",
        [0, 1, 2].map(|c| albedo[c] * hemi[c])
    );
}

/// Oracle 5: `ssaa: 2` is the box filter of the four sub-samples summed in row-major order,
/// on a one-triangle scene where the arithmetic can be written out by hand.
#[test]
fn ssaa_is_a_fixed_order_box_filter() {
    use es_render::bvh::Bvh;
    const N: u32 = 8;
    let scene = TriScene {
        tris: vec![es_render::Tri {
            v: [[2.0, -1.0, 1.0], [2.0, 1.0, 1.0], [2.0, 0.0, 3.0]],
            n: [-1.0, 0.0, 0.0],
            albedo: [0.6, 0.5, 0.4],
            emission: [0.0; 3],
            seg: 1,
        }],
        lights: Vec::new(),
        names: std::collections::BTreeMap::new(),
    };
    let cam = CameraView {
        pose: cornell_camera(N, N).pose,
        spec: es_render::ImageSpec::pinhole(N, N, 1.2),
    };
    let cfg = RenderConfig::rs_full(TileAtlasCfg::row(N, N, 1));
    assert_eq!(
        cfg.shading,
        Shading::FULL,
        "this test hand-computes the preset's four sub-samples"
    );
    let frame = cpu::rasterize(&scene, &cam, &cfg, 0);
    let got = frame.tile(Channel::Rgb8).unwrap().as_u8().unwrap().to_vec();

    let vp = es_render::ViewParams::new(&cam);
    let bvh = Bvh::build(&scene.tris);
    let (mut covered, mut partial) = (0, 0);
    for py in 0..N {
        for px in 0..N {
            // The hand computation: row-major over the 2x2 block, one sequential `+=`, then
            // one multiply by 1/4, then the sRGB encode.
            let mut acc = [0.0f32; 3];
            let mut hits = 0;
            for sy in 0..2 {
                for sx in 0..2 {
                    let d = cpu::primary_dir_sub(&vp, px, py, sx, sy, 0.5);
                    let Some(hit) = cpu::nearest_hit(&scene.tris, &bvh, vp.pos, d, vp.near, vp.far)
                    else {
                        continue;
                    };
                    hits += 1;
                    let tri = &scene.tris[hit.tri as usize];
                    let n = cpu::face_forward(tri, d);
                    let p = [
                        vp.pos[0] + d[0] * hit.t,
                        vp.pos[1] + d[1] * hit.t,
                        vp.pos[2] + d[2] * hit.t,
                    ];
                    let s = cpu::shade_full(&scene.tris, &bvh, tri, n, p, d, &cfg);
                    for c in 0..3 {
                        acc[c] += s[c];
                    }
                }
            }
            covered += usize::from(hits == 4);
            partial += usize::from(hits > 0 && hits < 4);
            let i = (py * N + px) as usize;
            for c in 0..3 {
                let want = cpu::to_u8(acc[c] * (1.0 / 4.0));
                assert_eq!(
                    got[i * 3 + c],
                    want,
                    "pixel ({px}, {py}) channel {c}: {hits} of 4 sub-samples hit"
                );
            }
        }
    }
    assert!(
        partial > 0 && covered > 0,
        "the triangle must both fill and partly cover pixels for this to mean anything \
         ({covered} full, {partial} partial)"
    );
    println!(
        "{covered} fully covered and {partial} partially covered pixels of {}, all equal to the \
         hand-computed row-major box filter",
        N * N
    );
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

/// The `Full` `Rgb8` of one config on the device against the CPU reference, with the
/// edge-pixel tolerance of `docs/design/renderer.md` section 9.3.
fn gpu_full_rgb8_matches_the_cpu(gpu: &Gpu, cfg: &RenderConfig, label: &str) {
    let cams = [cornell_camera(TILE, TILE)];
    let mut full = render_gpu(gpu, cfg.clone(), &cams, 1);
    let rgb = full.read_tile(0, Channel::Rgb8).expect("rgb");
    let want = cpu::rasterize(&scene(), &cams[0], cfg, 0);
    let want_rgb = want.tile(Channel::Rgb8).unwrap();
    let diff = rgb
        .as_u8()
        .unwrap()
        .iter()
        .zip(want_rgb.as_u8().unwrap())
        .filter(|(a, b)| a != b)
        .count();
    let (gpu_px, cpu_px) = (rgb.as_u8().unwrap(), want_rgb.as_u8().unwrap());

    // Every byte outside a geometry edge is bit-equal; a pixel the two disagree about must be
    // a pixel whose four sub-samples do not all land on the same triangle. See
    // `docs/design/renderer.md` section 9.3: `dot`'s summation order is not pinned across the
    // two implementations, so a sub-sample ray that grazes a shared triangle edge can pick a
    // different winner there, and the box filter then shows a quarter of a sample's shading.
    let vp = es_render::ViewParams::new(&cams[0]);
    let sc = scene();
    let bvh = es_render::bvh::Bvh::build(&sc.tris);
    let mut edges = 0;
    for py in 0..TILE {
        for px in 0..TILE {
            let i = ((py * TILE + px) as usize) * 3;
            if gpu_px[i..i + 3] == cpu_px[i..i + 3] {
                continue;
            }
            let worst = (0..3)
                .map(|c| u32::from(gpu_px[i + c].abs_diff(cpu_px[i + c])))
                .max()
                .unwrap_or(0);
            let hits: Vec<Option<u32>> = (0..4)
                .map(|k| {
                    let d = cpu::primary_dir_sub(&vp, px, py, k % 2, k / 2, 0.5);
                    cpu::nearest_hit(&sc.tris, &bvh, vp.pos, d, vp.near, vp.far).map(|h| h.tri)
                })
                .collect();
            assert!(
                hits.iter().any(|h| *h != hits[0]),
                "{label}: pixel ({px}, {py}) differs by {worst} but all four sub-samples hit \
                 {:?}: that is a shading divergence, not a coverage tie",
                hits[0]
            );
            // 255/4: the ceiling on what one sub-sample of four can move an 8-bit channel by,
            // in the linear domain the box filter averages in. A backstop only — the
            // discriminating assertion is the one above.
            assert!(
                worst <= 64,
                "{label}: pixel ({px}, {py}) is an edge pixel but differs by {worst} levels, \
                 more than one sub-sample of four can account for"
            );
            edges += 1;
            println!("  edge pixel ({px}, {py}): sub-sample hits {hits:?}, worst {worst} levels");
        }
    }
    let n_px = (TILE * TILE) as usize;
    assert!(
        edges * 1000 <= n_px,
        "{label}: {edges} of {n_px} pixels differ: more than the 0.1% of edge pixels the tie \
         explains"
    );
    println!(
        "{label}: {diff} of {} bytes differ from the CPU, in {edges} of {n_px} edge pixels",
        rgb.len()
    );
}

/// Oracle 3, GPU half (packet M7/R2): the `Full` look on the device is the CPU reference's
/// `Rgb8` bit for bit, and its geometry channels are the `Lambert` render's — same device,
/// so "the centre ray is the centre ray" is checkable without a ULP budget.
///
/// Two configs, because the Cornell box is a closed room: under the preset every shadow ray
/// is occluded, so `shadows: false` is the only way the device's `diffuse` term — the
/// energy-conserving mix amended at review — is executed at all.
#[test]
fn gpu_full_shading_matches_the_cpu() {
    let test = "gpu_full_shading_matches_the_cpu";
    let Some(gpu) = open(test) else { return };
    gpu_full_rgb8_matches_the_cpu(&gpu, &rs_full_cfg(), "Full Rgb8 (preset, all shadowed)");
    gpu_full_rgb8_matches_the_cpu(
        &gpu,
        &rs_full_with(false, 2),
        "Full Rgb8 (shadows off, lit)",
    );

    let cams = [cornell_camera(TILE, TILE)];
    let mut full = render_gpu(&gpu, rs_full_cfg(), &cams, 1);
    let mut lambert = render_gpu(&gpu, rs_cfg(), &cams, 1);
    for channel in [DEPTH, Channel::SegmentationId, Channel::Normal] {
        let (a, b) = (
            full.read_tile(0, channel).expect("full"),
            lambert.read_tile(0, channel).expect("lambert"),
        );
        assert!(
            a.to_bytes() == b.to_bytes(),
            "{channel:?} differs between Full and Lambert on the GPU"
        );
        println!("GPU Full/Lambert bit-equal: {channel:?}");
    }
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
        nee: false,
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

// --- the tone map, NEE, the unbiased ReSTIR and SSIM (packet M7/R3) --------------------------

/// The tile the convergence oracles work on. Small on purpose: a 4,096 spp reference is a
/// CPU path trace and `cargo xtask ci` runs the suite in debug.
const REF_TILE: u32 = 16;
const REF_SPP: u32 = 4096;

fn small_cam() -> CameraView {
    cornell_camera(REF_TILE, REF_TILE)
}

/// A `Pt` config on the small tile emitting `PtRadiance` only — the tone map is a different
/// oracle and its per-pixel loop is pure cost here.
fn small_pt(spp: u32, bounces: u32, nee: bool) -> RenderConfig {
    let atlas = TileAtlasCfg::row(REF_TILE, REF_TILE, 1);
    let mut cfg = if nee {
        RenderConfig::pt_nee(atlas, spp, bounces)
    } else {
        RenderConfig::pt(atlas, spp, bounces)
    };
    cfg.channels = BTreeSet::from([Channel::PtRadiance]);
    cfg
}

fn radiance_of(cfg: &RenderConfig) -> Vec<f32> {
    cpu::path_trace(&scene(), &small_cam(), cfg, 0)
        .tile(Channel::PtRadiance)
        .expect("radiance")
        .as_f32()
        .expect("f32")
        .to_vec()
}

/// The converged image the two estimators must agree on. Computed once per test binary —
/// oracles 3 and 4 both want it, and it is the expensive thing in this file.
fn reference(nee: bool, bounces: u32) -> &'static Vec<f32> {
    use std::sync::OnceLock;
    static NEE3: OnceLock<Vec<f32>> = OnceLock::new();
    static PLAIN4: OnceLock<Vec<f32>> = OnceLock::new();
    static PLAIN2: OnceLock<Vec<f32>> = OnceLock::new();
    match (nee, bounces) {
        (true, _) => NEE3.get_or_init(|| radiance_of(&small_pt(REF_SPP, 3, true))),
        (false, 4) => PLAIN4.get_or_init(|| radiance_of(&small_pt(REF_SPP, 4, false))),
        (false, _) => PLAIN2.get_or_init(|| radiance_of(&small_pt(REF_SPP, 2, false))),
    }
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len() as f64
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x - *y).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn rmse(a: &[f32], b: &[f32]) -> f64 {
    (a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64)
        .sqrt()
}

/// Oracle 2, the half that needs no device: the tone map is monotone per channel, so a
/// brighter linear radiance never encodes to a darker byte.
fn tonemap_is_monotone() {
    for map in [es_render::Tonemap::Reinhard, es_render::Tonemap::Aces] {
        for exposure in [0.25f32, 1.0, 4.0] {
            let mut last = 0u8;
            let mut x = 0.0f32;
            while x < 64.0 {
                let got = cpu::tonemap_to_u8([x; 3], exposure, map)[0];
                assert!(
                    got >= last,
                    "{map:?} at exposure {exposure}: {x} encoded to {got} after {last}"
                );
                last = got;
                x += 1.0 / 512.0;
            }
            assert!(
                last > 200,
                "{map:?} at exposure {exposure} never got bright"
            );
        }
    }
    println!("tone map monotone on [0, 64) for both operators at three exposures");
}

/// Oracle 2: a `PtRadiance` tile through `Reinhard` and `Aces`, on the CPU and on the GPU,
/// encodes to **bitwise equal** `Rgb8`. Not a ULP budget: neither operator contains a
/// transcendental, so the only approximation left is the sRGB transfer both sides already
/// share (spec 28.7 gate 3).
#[test]
fn tonemap_is_bitwise_on_both_sides() {
    let test = "tonemap_is_bitwise_on_both_sides";
    tonemap_is_monotone();
    let Some(gpu) = open(test) else { return };
    let cams = [cornell_camera(TILE, TILE)];
    for map in [es_render::Tonemap::Reinhard, es_render::Tonemap::Aces] {
        for exposure in [1.0f32, 8.0] {
            // 1 spp / 2 bounces: `gpu_path_tracer_matches_the_cpu_reference_at_1spp` already
            // pins that `PtRadiance` bit-equal, so any `Rgb8` difference here is the tone
            // map's and nothing else's.
            let mut cfg = pt_cfg(1, 2);
            cfg.tonemap = map;
            cfg.exposure = exposure;
            let mut atlas = render_gpu(&gpu, cfg.clone(), &cams, 1);
            let got = atlas.read_tile(0, Channel::Rgb8).expect("rgb8");
            let want = cpu::path_trace(&scene(), &cams[0], &cfg, 0);
            let want_rgb = want.tile(Channel::Rgb8).expect("rgb8");
            let diff = got
                .as_u8()
                .unwrap()
                .iter()
                .zip(want_rgb.as_u8().unwrap())
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "{map:?} at exposure {exposure}: {diff} of {} bytes differ",
                got.len()
            );
            assert_eq!(diff, 0, "{map:?} at exposure {exposure} is not bitwise");
        }
    }
}

/// Oracle 3: NEE and the BSDF-only estimator are two estimators of the *same* integral, so
/// they agree **in expectation** — the image means are within 1% of each other at 4,096 spp.
///
/// **Two deviations from the packet's wording**, both forced by what the two estimators
/// actually are:
///
/// 1. NEE at `B` bounces is the BSDF-only tracer at `B + 1`, not at `B`. NEE's last vertex
///    gathers light the BSDF-only path never reaches, because its continuation ray is never
///    traced. So this compares NEE at 3 bounces with no-NEE at 4. Comparing both at 3 leaves
///    a real +3.7% difference that is a *truncation* difference, not a bias.
/// 2. The assertion is on the image means, not on the per-pixel mean absolute difference.
///    At 4,096 spp each estimator still carries its own Monte-Carlo noise:
///    `nee_at_low_spp_has_lower_variance` measures the BSDF-only RMSE at 16 spp as ~0.024,
///    which is ~0.0015 at 4,096 — the size of the per-pixel difference this test prints.
///    Testing expectation means averaging that noise away. The per-pixel figure is printed
///    beside it and given a loose backstop, so a real estimator mismatch (a missing MIS
///    weight, a wrong pdf) still fails loudly.
#[test]
fn nee_converges_to_the_same_image() {
    let with = reference(true, 3);
    let without = reference(false, 4);
    let (m_on, m_off) = (mean(with), mean(without));
    let d = mean_abs_diff(with, without);
    let bias = m_on - m_off;
    println!(
        "NEE on (3 bounces) vs off (4 bounces), {REF_SPP} spp, {REF_TILE}x{REF_TILE}: mean \
         radiance {m_off:.6} (off) vs {m_on:.6} (on), difference of means {bias:+.6} = \
         {:+.3}%; per-pixel mean |difference| {d:.6} = {:.3}% of the mean (the Monte-Carlo \
         floor)",
        100.0 * bias / m_off,
        100.0 * d / m_off
    );
    assert!(m_off > 0.0, "the reference image is black");
    assert!(
        bias.abs() < 0.01 * m_off,
        "the two estimators disagree in expectation by {:+.3}% of the mean radiance",
        100.0 * bias / m_off
    );
    assert!(
        d < 0.10 * m_off,
        "the per-pixel difference is {:.3}% of the mean: too large to be noise",
        100.0 * d / m_off
    );
}

/// Oracle 4: what NEE buys. Same 16 samples, same seed, same bounces — the RMSE against the
/// converged image is smaller with NEE on.
#[test]
fn nee_at_low_spp_has_lower_variance() {
    let converged = reference(true, 3);
    let on = radiance_of(&small_pt(16, 3, true));
    let off = radiance_of(&small_pt(16, 3, false));
    let (a, b) = (rmse(&on, converged), rmse(&off, converged));
    println!(
        "16 spp RMSE against the {REF_SPP} spp reference: NEE on {a:.6}, NEE off {b:.6} \
         ({:.2}x lower)",
        b / a
    );
    assert!(a < b, "NEE did not reduce the RMSE ({a} vs {b})");
}

/// Oracle 5: the spatial reuse is unbiased. 256 independent 1 spp `ReSTIR` frames, averaged,
/// against a 4,096 spp path trace of the *same* integral — direct lighting only, which is
/// what `ReSTIR` DI estimates, so the reference is two bounces without NEE (primary emission
/// plus one cosine-sampled bounce onto the light).
#[test]
fn restir_is_unbiased_within_tolerance() {
    const SEEDS: u32 = 256;
    let converged = reference(false, 2);
    let mut acc = vec![0.0f64; converged.len()];
    for s in 0..SEEDS {
        let mut cfg = small_pt(1, 2, false);
        cfg.path = RenderPath::Pt {
            spp: 1,
            bounces: 2,
            nee: false,
            restir: true,
            svgf: false,
        };
        cfg.seed = 0x5eed_1234u32.wrapping_add(s.wrapping_mul(0x9e37_79b9));
        for (a, x) in acc.iter_mut().zip(&radiance_of(&cfg)) {
            *a += f64::from(*x);
        }
    }
    let got: Vec<f32> = acc.iter().map(|x| (*x / f64::from(SEEDS)) as f32).collect();
    let m = mean(converged);
    let bias = mean(&got) - m;
    println!(
        "ReSTIR (spatial, pairwise MIS), {SEEDS} seeds x 1 spp vs {REF_SPP} spp direct: \
         reference mean {m:.6}, ReSTIR mean {:.6}, bias {bias:+.6} = {:+.3}%",
        mean(&got),
        100.0 * bias / m
    );
    assert!(m > 0.0, "the reference image is black");
    assert!(
        bias.abs() < 0.01 * m,
        "ReSTIR is biased by {:+.3}% of the mean radiance",
        100.0 * bias / m
    );
}

/// Oracle 6: the new golden is the CPU's bit for bit, and the device reproduces it.
#[test]
fn gpu_pt_nee_matches_the_cpu() {
    let test = "gpu_pt_nee_matches_the_cpu";
    // The CPU half runs everywhere, device or not.
    let cpu_frame = cpu_pt_nee();
    let cpu_rgb = cpu_frame.tile(Channel::Rgb8).expect("rgb8").to_bytes();
    let want = std::fs::read(golden_dir().join("cornell_pt_nee_rgb8.bin"))
        .expect("the golden (run the generate_goldens test)");
    assert!(
        cpu_rgb == want,
        "cornell_pt_nee_rgb8 differs from its golden"
    );
    println!("bit-equal CPU vs golden: cornell_pt_nee_rgb8");

    let Some(gpu) = open(test) else { return };
    let cams = [cornell_camera(TILE, TILE)];
    let mut atlas = render_gpu(&gpu, pt_nee_cfg(), &cams, 1);
    let rad = atlas.read_tile(0, Channel::PtRadiance).expect("radiance");
    let want_rad = cpu_frame
        .tile(Channel::PtRadiance)
        .expect("radiance")
        .as_f32()
        .unwrap();
    let norm = max_normalized(rad.as_f32().unwrap(), want_rad);
    let (ulp, _) = max_ulp(rad.as_f32().unwrap(), want_rad);
    println!("PtRadiance 4 spp NEE: normalized max error {norm:e}, max ULP {ulp}");
    assert!(norm <= 1e-5, "PtRadiance diverged by {norm:e}");

    let rgb = atlas.read_tile(0, Channel::Rgb8).expect("rgb8");
    let diff = rgb
        .as_u8()
        .unwrap()
        .iter()
        .zip(cpu_frame.tile(Channel::Rgb8).unwrap().as_u8().unwrap())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "Rgb8 4 spp NEE: {diff} of {} bytes differ from the CPU",
        rgb.len()
    );
    assert!(
        diff * 1000 <= rgb.len(),
        "{diff} of {} bytes differ, more than the 0.1% a shadow-ray tie explains",
        rgb.len()
    );
}

/// Oracle 7: `ssim(a, a)` is exactly 1, and the score falls as the noise grows.
#[test]
#[allow(clippy::float_cmp)]
fn ssim_is_one_for_identical_and_falls_with_noise() {
    let base = cpu_rs_full()
        .tile(Channel::Rgb8)
        .expect("rgb8")
        .as_u8()
        .unwrap()
        .to_vec();
    assert_eq!(es_render::ssim(&base, &base, TILE, TILE), 1.0);
    println!("ssim(a, a) = 1.0 exactly");

    let mut last = 1.0;
    for amp in [2.0f64, 4.0, 8.0, 16.0, 32.0] {
        // One fixed unit-noise field scaled by `amp`, so a larger `amp` is strictly more
        // noise at every pixel rather than a different draw.
        let noisy: Vec<u8> = base
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let u = f64::from(es_render::rng::mix32(i as u32)) / f64::from(u32::MAX);
                let n = (u * 2.0 - 1.0) * amp;
                (f64::from(*v) + n).clamp(0.0, 255.0).round() as u8
            })
            .collect();
        let s = es_render::ssim(&base, &noisy, TILE, TILE);
        println!("ssim(a, a + {amp} * unit noise) = {s:.6}");
        assert!(s < last, "SSIM did not fall at noise amplitude {amp}");
        last = s;
    }
}

/// The §15.3 `RS`/`PT` colour similarity, measured for the first time (packet M7/R3): `Rs`
/// `Full` at the R2 preset (`ssaa 2`) against a converged `Pt` NEE render, on Cornell and on
/// the SO-101 cell through V9's showcase camera.
///
/// Not an assertion. Spec 15.3 asks for an SSIM *threshold* and this is the measurement that
/// number has to be set from; asserting one here would be asserting a number nobody has
/// looked at yet. The figures land in `docs/design/renderer.md` section 10 and the raw tiles
/// under `target/plan-u/r3/` for a human to look at. Run with
/// `cargo test -p es-render --release -- --ignored --nocapture rs_pt_ssim`.
///
/// The SO-101 half needs a device: 320x180 at 1,024 spp is minutes of CPU. It is split into
/// chunks of 64 spp with different seeds and averaged on the host, so no single dispatch can
/// trip a driver watchdog — a different estimator from one 1,024 spp dispatch, equally
/// unbiased, and the only one that runs on a desktop Windows box.
#[test]
#[ignore = "measurement; run explicitly"]
fn rs_pt_ssim() {
    const W: u32 = 320;
    const H: u32 = 180;
    const CHUNKS: u32 = 16;
    const CHUNK_SPP: u32 = 64;
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/plan-u/r3");
    std::fs::create_dir_all(&out).expect("output dir");
    let dump = |name: &str, bytes: &[u8], w: u32, h: u32| {
        let path = out.join(format!("{name}.bin"));
        std::fs::write(&path, bytes).expect("write");
        std::fs::write(
            out.join(format!("{name}.json")),
            format!("{{\"dtype\":\"u8\",\"shape\":[{h},{w},3]}}\n"),
        )
        .expect("write");
        println!("wrote {}", path.display());
    };

    // Exposure sweep, because a single exposure conflates two different things: the two
    // paths model *different lighting* (`Rs Full` is a hemisphere ambient plus a directional
    // light with no interreflection; `Pt` is one emissive panel with global illumination) and
    // SSIM punishes a brightness offset as hard as a structural one. Sweeping says what the
    // best any exposure can do is, which is the number a structural threshold belongs on.
    let sweep = |label: &str, rs: &[u8], radiance: &[f32], w: u32, h: u32| -> [Vec<u8>; 2] {
        let mut best = (0.0f64, 1.0f32);
        let mut kept = [Vec::new(), Vec::new()];
        for exposure in [0.5f32, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0] {
            let pt: Vec<u8> = radiance
                .chunks_exact(3)
                .flat_map(|p| {
                    cpu::tonemap_to_u8([p[0], p[1], p[2]], exposure, es_render::Tonemap::Reinhard)
                })
                .collect();
            let s = es_render::ssim(rs, &pt, w, h);
            let mean = pt.iter().map(|b| u32::from(*b)).sum::<u32>() / pt.len() as u32;
            println!(
                "| {label} {w}x{h} | Reinhard, exposure {exposure} | SSIM {s:.4} | mean byte \
                 {mean} |"
            );
            if s > best.0 {
                best = (s, exposure);
            }
            // The default exposure, and the one whose mean byte is closest to the
            // rasterizer's — the two a person would want to look at.
            if exposure.to_bits() == 1.0f32.to_bits() {
                kept[0] = pt;
            } else if exposure.to_bits() == 32.0f32.to_bits() {
                kept[1] = pt;
            }
        }
        println!(
            "| {label} {w}x{h} | **best** at exposure {} | **SSIM {:.4}** |",
            best.1, best.0
        );
        kept
    };

    // Cornell, on the CPU: small enough that the reference stays the golden generator's.
    let cam = cornell_camera(TILE, TILE);
    let rs = cpu::rasterize(&scene(), &cam, &rs_full_cfg(), 0);
    let mut pt_cfg = RenderConfig::pt_nee(TileAtlasCfg::row(TILE, TILE, 1), 1024, 3);
    pt_cfg.channels = BTreeSet::from([Channel::PtRadiance]);
    let pt = cpu::path_trace(&scene(), &cam, &pt_cfg, 0);
    let a = rs.tile(Channel::Rgb8).unwrap().as_u8().unwrap().to_vec();
    let radiance = pt.tile(Channel::PtRadiance).unwrap().as_f32().unwrap();
    let [b1, b32] = sweep("cornell", &a, radiance, TILE, TILE);
    dump("cornell_rs_full", &a, TILE, TILE);
    dump("cornell_pt_nee_1024_e1", &b1, TILE, TILE);
    dump("cornell_pt_nee_1024_e32", &b32, TILE, TILE);

    let test = "rs_pt_ssim (SO-101 half)";
    let Some(gpu) = open(test) else { return };
    let cam = showcase_camera(W, H);
    let tri = TriScene::from_scene(&so101()).expect("so101 tessellates");

    let mut rs_cfg = RenderConfig::rs_full(TileAtlasCfg::row(W, H, 1));
    rs_cfg.channels = BTreeSet::from([Channel::Rgb8]);
    let mut renderer = Renderer::new(&gpu, rs_cfg).expect("renderer");
    renderer.upload_tris(tri.clone()).expect("upload");
    let rs_bytes = renderer
        .render(&[cam])
        .expect("render")
        .read_tile(0, Channel::Rgb8)
        .expect("rgb8")
        .as_u8()
        .unwrap()
        .to_vec();

    // Accumulate linear radiance over the chunks, then tone-map once on the host through the
    // same `cpu::tonemap_to_u8` the kernel mirrors.
    let mut acc = vec![0.0f64; (W as usize) * (H as usize) * 3];
    for c in 0..CHUNKS {
        let mut cfg = RenderConfig::pt_nee(TileAtlasCfg::row(W, H, 1), CHUNK_SPP, 3);
        cfg.channels = BTreeSet::from([Channel::PtRadiance]);
        cfg.seed = 0x5eed_1234u32.wrapping_add(c.wrapping_mul(0x9e37_79b9));
        let mut r = Renderer::new(&gpu, cfg).expect("renderer");
        r.upload_tris(tri.clone()).expect("upload");
        let tile = r
            .render(&[cam])
            .expect("render")
            .read_tile(0, Channel::PtRadiance)
            .expect("radiance");
        for (a, x) in acc.iter_mut().zip(tile.as_f32().unwrap()) {
            *a += f64::from(*x);
        }
        println!("  chunk {}/{CHUNKS} done", c + 1);
    }
    let mean_radiance: Vec<f32> = acc
        .iter()
        .map(|x| (*x / f64::from(CHUNKS)) as f32)
        .collect();
    println!("so101: {} spp total", CHUNKS * CHUNK_SPP);
    let [pt1, pt32] = sweep("so101", &rs_bytes, &mean_radiance, W, H);
    dump("so101_rs_full", &rs_bytes, W, H);
    dump("so101_pt_nee_e1", &pt1, W, H);
    dump("so101_pt_nee_e32", &pt32, W, H);
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
        // And the path `Atlas::read_tile` actually takes (M7/R1b): a device-local buffer
        // through `Buffer::download`, staged through host-cached memory since R1b.
        let mut storage =
            es_gpu::Buffer::new(&gpu, bytes, es_gpu::Usage::Storage).expect("storage buffer");
        let t = std::time::Instant::now();
        let got = storage.download().expect("device-local download");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "device-local download of {} KiB via Buffer::download: {ms:.3} ms ({:.0} MiB/s)",
            bytes / 1024,
            got.len() as f64 / ms / 1048.576
        );
    }
}

// --- temporal accumulation for a still camera (packet M7/R4) ---------------------------------

/// The accumulating config: R3's golden config (NEE, 3 bounces, Reinhard, exposure 1) with
/// `spp` split across frames and a history cap.
fn pt_accum_cfg(spp: u32, max_history: u32) -> RenderConfig {
    let mut cfg = RenderConfig::pt_nee(TileAtlasCfg::row(TILE, TILE, 1), spp, 3);
    cfg.temporal = Some(Temporal { max_history });
    cfg
}

/// `frames` accumulated frames of one still camera, and the history they left.
fn cpu_accum(cfg: &RenderConfig, frames: u32) -> (Frame, cpu::History) {
    let (sc, cam) = (scene(), cornell_camera(TILE, TILE));
    let mut history = cpu::History::default();
    let mut frame = cpu::path_trace_accum(&sc, &cam, cfg, 0, &mut history);
    for _ in 1..frames {
        frame = cpu::path_trace_accum(&sc, &cam, cfg, 0, &mut history);
    }
    (frame, history)
}

/// The config the new golden `cornell_pt_accum8_rgb8` is generated from: 8 frames x 1 spp.
fn cpu_pt_accum8() -> Frame {
    cpu_accum(&pt_accum_cfg(1, 8), 8).0
}

/// Oracle 2, and the whole point of the packet: `N` frames of `spp` samples are **bitwise**
/// one frame of `N * spp` samples. Two things make it exact and neither is negotiable -- the
/// sample index is `n * spp + s`, so the frames draw the same samples in the same order, and
/// the kernel's accumulator *starts* at the history sum, so the additions happen in one
/// unbroken left-to-right chain and the divide happens once at output.
#[test]
fn accumulation_of_n_frames_is_n_spp() {
    for (frames, spp) in [(8u32, 1u32), (8, 4)] {
        let (acc, history) = cpu_accum(&pt_accum_cfg(spp, 32), frames);
        let mut one = RenderConfig::pt_nee(TileAtlasCfg::row(TILE, TILE, 1), frames * spp, 3);
        one.channels = acc.channels.keys().copied().collect();
        let want = cpu::path_trace(&scene(), &cornell_camera(TILE, TILE), &one, 0);
        for channel in [Channel::PtRadiance, Channel::Rgb8] {
            let (a, b) = (
                acc.tile(channel).expect("accumulated"),
                want.tile(channel).expect("one frame"),
            );
            assert!(
                a.to_bytes() == b.to_bytes(),
                "{frames} frames x {spp} spp is not bitwise {} spp in {channel:?}",
                frames * spp
            );
        }
        assert!(
            history.n().iter().all(|n| *n == frames),
            "the history length is not {frames} everywhere"
        );
        println!(
            "{frames} frames x {spp} spp == 1 frame x {} spp, bitwise, in PtRadiance and Rgb8",
            frames * spp
        );
    }
}

/// Cornell with the tall block translated along -Y, for the disocclusion oracle.
fn cornell_moved_block() -> TriScene {
    let mut desc = cornell_box();
    for geom in &mut desc.bodies[0].geoms {
        if geom.name == "tall" {
            geom.pose.position.y -= 0.6;
        }
    }
    TriScene::from_scene(&desc).expect("cornell tessellates")
}

/// Oracle 3: the history is kept exactly where the pixel's depth, normal and primitive id are
/// bitwise the previous frame's, and dropped exactly where they are not. A changed
/// `CameraView` drops the whole slot.
#[test]
fn history_drops_where_the_scene_moved() {
    const FRAMES: u32 = 4;
    let cfg = pt_accum_cfg(1, 32);
    let (sc, moved, cam) = (scene(), cornell_moved_block(), cornell_camera(TILE, TILE));
    let mut history = cpu::History::default();
    for _ in 0..FRAMES {
        cpu::path_trace_accum(&sc, &cam, &cfg, 0, &mut history);
    }
    assert!(
        history.n().iter().all(|n| *n == FRAMES),
        "the static scene did not accumulate {FRAMES} frames"
    );

    // The block moves: every pixel whose primary hit changed starts over at 1, every other
    // one carries its history.
    let frame = cpu::path_trace_accum(&moved, &cam, &cfg, 0, &mut history);
    let n = frame
        .tile(Channel::History)
        .expect("the History channel")
        .as_u32()
        .expect("u32")
        .to_vec();
    assert_eq!(n, history.n(), "the channel is not the history length");

    // The rule the renderer applies, computed here from the primary hit: depth, camera-space
    // normal and **primitive id**, all bitwise. The primitive id is the triangle index, not
    // the geom id, so a pixel that crossed a quad's diagonal while staying on the same flat
    // face counts as changed — stricter than the eye, and the disocclusion test is the one
    // place that costs nothing (`docs/design/renderer.md` section 11).
    let primary_hits = |sc: &TriScene| -> Vec<(u32, u32, [u32; 3])> {
        let vp = es_render::ViewParams::new(&cam);
        let bvh = es_render::bvh::Bvh::build(&sc.tris);
        let mut out = Vec::with_capacity((TILE * TILE) as usize);
        for py in 0..TILE {
            for px in 0..TILE {
                let d = cpu::primary_dir(&vp, px, py);
                match cpu::nearest_hit(&sc.tris, &bvh, vp.pos, d, vp.near, vp.far) {
                    Some(hit) => {
                        let tri = &sc.tris[hit.tri as usize];
                        let nrm = cpu::quat_rotate_inv(vp.quat, cpu::face_forward(tri, d));
                        out.push((hit.t.to_bits(), hit.tri + 1, nrm.map(f32::to_bits)));
                    }
                    None => out.push((vp.far.to_bits(), 0, [0u32; 3])),
                }
            }
        }
        out
    };
    let (before, after) = (primary_hits(&sc), primary_hits(&moved));
    let (mut kept, mut dropped) = (0, 0);
    for i in 0..(TILE * TILE) as usize {
        let same = before[i] == after[i];
        if same {
            assert_eq!(
                n[i],
                FRAMES + 1,
                "pixel {i} did not change but lost its history"
            );
            kept += 1;
        } else {
            assert_eq!(n[i], 1, "pixel {i} changed but kept its history");
            dropped += 1;
        }
    }
    assert!(kept > 0 && dropped > 0, "{kept} kept, {dropped} dropped");
    println!(
        "the block moved: {kept} pixels kept their history ({}), {dropped} dropped it (1)",
        FRAMES + 1
    );

    // A different camera resets the slot: every pixel is back to one frame.
    let mut moved_cam = cam;
    moved_cam.pose.position.z += 0.05;
    let after = cpu::path_trace_accum(&sc, &moved_cam, &cfg, 0, &mut history);
    let n = after
        .tile(Channel::History)
        .expect("the History channel")
        .as_u32()
        .expect("u32")
        .to_vec();
    assert!(
        n.iter().all(|v| *v == 1),
        "a changed CameraView did not reset the whole slot"
    );
    println!("a changed CameraView resets every pixel to 1");
}

/// Oracle 4: the variance of the accumulated estimate falls as the history grows -- it is the
/// quantity the luminance weight divides by, so if it did not fall the filter would never
/// narrow. `n < 4` is the 7x7 spatial estimate, `n >= 4` the moments.
///
/// **Deviation from the packet, and the measurement behind it.** The packet asks for the
/// *mean* per-pixel variance to fall. It does not, and the reason is not the estimator: a
/// 1 spp path trace has a heavy tail, so a single firefly pixel out of 4,096 (variance 17.9
/// at n = 16, against a median of 2.3e-5) owns the mean. The estimate of `sigma^2` from `n`
/// frames also *grows* towards the truth as `n` grows — four frames usually miss the tail
/// entirely — which is exactly what the mean is picking up. What the packet is really asking
/// is whether pixels get more certain, so the assertion is on the **median and the p99** and
/// the mean is printed beside them. `docs/design/renderer.md` section 11 has the table.
#[test]
fn variance_falls_with_history() {
    let cfg = pt_accum_cfg(1, 32);
    let mut history = cpu::History::default();
    let (sc, cam) = (scene(), cornell_camera(TILE, TILE));
    let mut seen = Vec::new();
    for frame in 1..=16u32 {
        cpu::path_trace_accum(&sc, &cam, &cfg, 0, &mut history);
        if [1u32, 4, 16].contains(&frame) {
            let v = history.variance();
            let mean = f64::from(v.iter().sum::<f32>()) / v.len() as f64;
            let mut sorted = v.to_vec();
            sorted.sort_by(f32::total_cmp);
            let (median, p99, max) = (
                sorted[sorted.len() / 2],
                sorted[sorted.len() * 99 / 100],
                sorted[sorted.len() - 1],
            );
            println!(
                "n = {frame:>2}: per-pixel variance mean {mean:.9}, median {median:e}, \
                 p99 {p99:e}, max {max:e}"
            );
            seen.push((frame, median, p99));
        }
    }
    for pair in seen.windows(2) {
        assert!(
            pair[1].1 < pair[0].1,
            "the median variance did not fall from n = {} ({:e}) to n = {} ({:e})",
            pair[0].0,
            pair[0].1,
            pair[1].0,
            pair[1].1
        );
        assert!(
            pair[1].2 < pair[0].2,
            "the p99 variance did not fall from n = {} ({:e}) to n = {} ({:e})",
            pair[0].0,
            pair[0].2,
            pair[1].0,
            pair[1].2
        );
    }
}

/// Oracle 6: the luminance weight narrows the filter where the estimate has converged. A
/// synthetic tile with flat geometry (so the depth and normal weights are 1 everywhere and
/// the luminance term is the only thing under test): the left half carries structure and zero
/// variance, the right half is noise with a large variance. With the weight on, the converged
/// half moves less.
#[test]
fn luminance_weight_narrows_the_filter_where_variance_is_low() {
    const W: u32 = 64;
    const H: u32 = 32;
    let n = (W * H) as usize;
    let depth = vec![1.0f32; n];
    let mut normal = vec![0.0f32; n * 3];
    for i in 0..n {
        normal[i * 3 + 2] = 1.0;
    }
    let (mut color, mut variance) = (vec![0.0f32; n * 3], vec![0.0f32; n]);
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            let converged = x < W / 2;
            let v = if converged {
                // Structure a blur would destroy: a 4-pixel stripe pattern.
                f32::from(u8::from((x / 4) % 2 == 0))
            } else {
                f32::from((es_render::rng::mix32(i as u32) >> 24) as u8) / 255.0
            };
            for c in 0..3 {
                color[i * 3 + c] = v;
            }
            variance[i] = if converged { 0.0 } else { 0.25 };
        }
    }
    let (with, _) = cpu::atrous(&color, Some(&variance), &depth, &normal, W, H, 4);
    let (without, _) = cpu::atrous(&color, None, &depth, &normal, W, H, 4);

    let half = |out: &[f32], converged: bool| -> f64 {
        let (mut acc, mut count) = (0.0f64, 0usize);
        for y in 0..H {
            for x in 0..W {
                if (x < W / 2) != converged {
                    continue;
                }
                let i = (y * W + x) as usize;
                let d = f64::from(out[i * 3] - color[i * 3]);
                acc += d * d;
                count += 1;
            }
        }
        (acc / count as f64).sqrt()
    };
    let (a, b) = (half(&with, true), half(&without, true));
    let (na, nb) = (half(&with, false), half(&without, false));
    println!(
        "RMSE against the input after 4 a-trous iterations: converged half {a:.6} with the \
         luminance weight vs {b:.6} without; noisy half {na:.6} vs {nb:.6}"
    );
    assert!(
        a < b,
        "the luminance weight did not narrow the filter on the converged half ({a} vs {b})"
    );
    assert!(
        na > a,
        "the noisy half must still be filtered ({na} vs {a} on the converged half)"
    );
}

/// Oracle 5: the new golden is the CPU reference's bit for bit, and the device reproduces the
/// accumulation -- the `History` channel bitwise (it is integer logic over geometry each side
/// compares with its own previous frame) and the radiance within the PT tolerance.
#[test]
fn gpu_accumulation_matches_the_cpu() {
    let test = "gpu_accumulation_matches_the_cpu";
    // The CPU half runs everywhere, device or not.
    let cpu_frame = cpu_pt_accum8();
    let cpu_rgb = cpu_frame.tile(Channel::Rgb8).expect("rgb8").to_bytes();
    let want = std::fs::read(golden_dir().join("cornell_pt_accum8_rgb8.bin"))
        .expect("the golden (run the generate_goldens test)");
    assert!(
        cpu_rgb == want,
        "cornell_pt_accum8_rgb8 differs from its golden"
    );
    println!("bit-equal CPU vs golden: cornell_pt_accum8_rgb8");

    let Some(gpu) = open(test) else { return };
    let cams = [cornell_camera(TILE, TILE)];
    let mut atlas = render_gpu(&gpu, pt_accum_cfg(1, 8), &cams, 8);

    let n = atlas.read_tile(0, Channel::History).expect("history");
    let want_n = cpu_frame.tile(Channel::History).unwrap().as_u32().unwrap();
    assert!(
        n.as_u32().unwrap() == want_n,
        "the GPU History channel is not the CPU's"
    );
    println!(
        "History after 8 frames: bit-equal to the CPU ({} everywhere)",
        want_n[0]
    );

    let rad = atlas.read_tile(0, Channel::PtRadiance).expect("radiance");
    let want_rad = cpu_frame
        .tile(Channel::PtRadiance)
        .expect("radiance")
        .as_f32()
        .unwrap();
    let norm = max_normalized(rad.as_f32().unwrap(), want_rad);
    let (ulp, _) = max_ulp(rad.as_f32().unwrap(), want_rad);
    println!("PtRadiance after 8 x 1 spp: normalized max error {norm:e}, max ULP {ulp}");
    assert!(
        norm <= 1e-5,
        "the accumulated radiance diverged by {norm:e}"
    );

    let rgb = atlas.read_tile(0, Channel::Rgb8).expect("rgb8");
    let diff = rgb
        .as_u8()
        .unwrap()
        .iter()
        .zip(&cpu_rgb)
        .filter(|(a, b)| *a != *b)
        .count();
    println!(
        "Rgb8 after 8 x 1 spp: {diff} of {} bytes differ from the golden",
        rgb.len()
    );
    assert!(
        diff * 1000 <= rgb.len(),
        "{diff} of {} bytes differ, more than the 0.1% a shadow-ray tie explains",
        rgb.len()
    );

    // The variance-guided filter, whose only GPU oracle is this: the same accumulation with
    // SVGF on, against the CPU reference.
    let mut cfg = pt_accum_cfg(1, 8);
    cfg.path = RenderPath::Pt {
        spp: 1,
        bounces: 3,
        nee: true,
        restir: false,
        svgf: true,
    };
    let mut filtered = render_gpu(&gpu, cfg.clone(), &cams, 8);
    let (sc, cam) = (scene(), cams[0]);
    let mut history = cpu::History::default();
    let mut cpu_svgf = cpu::path_trace_accum(&sc, &cam, &cfg, 0, &mut history);
    for _ in 1..8 {
        cpu_svgf = cpu::path_trace_accum(&sc, &cam, &cfg, 0, &mut history);
    }
    let got = filtered
        .read_tile(0, Channel::PtRadiance)
        .expect("radiance");
    let want_f = cpu_svgf
        .tile(Channel::PtRadiance)
        .unwrap()
        .as_f32()
        .unwrap();
    let norm = max_normalized(got.as_f32().unwrap(), want_f);
    let rgb = filtered.read_tile(0, Channel::Rgb8).expect("rgb8");
    let want_rgb = cpu_svgf.tile(Channel::Rgb8).unwrap().as_u8().unwrap();
    let bytes = rgb
        .as_u8()
        .unwrap()
        .iter()
        .zip(want_rgb)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "accumulated + variance-guided SVGF: normalized max error {norm:e}, {bytes} of {} \
         Rgb8 bytes differ",
        rgb.len()
    );
    // 1e-3, not the 1e-5 the unguided filter holds to, and the luminance weight is why:
    // `exp(-|l_p - l_q| / (sigma_l * sqrt(var) + 1e-10))` is a near-discontinuous function of
    // its inputs where the variance is small, so the ~1e-7 the two path tracers already
    // disagree by (216 ULP, section 10.3) is amplified into a different tap weight. The
    // `Rgb8` assertion below is the one that says the pictures are the same.
    assert!(norm <= 1e-3, "the filtered image diverged by {norm:e}");
    assert!(
        bytes * 100 <= rgb.len(),
        "{bytes} of {} Rgb8 bytes differ after the guided filter",
        rgb.len()
    );
}
