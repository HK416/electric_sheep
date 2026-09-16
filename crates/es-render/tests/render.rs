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
    cpu, Atlas, CameraView, Frame, RenderConfig, RenderPath, Renderer, Shading, Tile, TileAtlasCfg,
    TriScene,
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
    /// Which render the tile comes from, and the `kernel` line of the sidecar.
    source: Source,
    kernel: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Rs,
    RsFull,
    Pt,
}

const GOLDENS: [Golden; 5] = [
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
];

fn golden_tile(g: &Golden, rs: &Frame, rs_full: &Frame, pt: &Frame) -> Tile {
    let frame = match g.source {
        Source::Rs => rs,
        Source::RsFull => rs_full,
        Source::Pt => pt,
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
    let (rs, rs_full, pt) = (cpu_rs(), cpu_rs_full(), cpu_pt1());
    for g in &GOLDENS {
        let tile = golden_tile(g, &rs, &rs_full, &pt);
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
    let (rs, rs_full, pt) = (cpu_rs(), cpu_rs_full(), cpu_pt1());
    for g in &GOLDENS {
        let path = golden_dir().join(format!("{}.bin", g.name));
        let expected = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run the generate_goldens test)", path.display()));
        let got = golden_tile(g, &rs, &rs_full, &pt).to_bytes();
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
    let with_shadows = |shadows: bool| {
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
            ssaa: 1,
        };
        cfg
    };
    let (on, off) = (with_shadows(true), with_shadows(false));
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

/// Oracle 3, GPU half (packet M7/R2): the `Full` look on the device is the CPU reference's
/// `Rgb8` bit for bit, and its geometry channels are the `Lambert` render's — same device,
/// so "the centre ray is the centre ray" is checkable without a ULP budget.
#[test]
fn gpu_full_shading_matches_the_cpu() {
    let test = "gpu_full_shading_matches_the_cpu";
    let Some(gpu) = open(test) else { return };
    let cams = [cornell_camera(TILE, TILE)];
    let mut full = render_gpu(&gpu, rs_full_cfg(), &cams, 1);
    let mut lambert = render_gpu(&gpu, rs_cfg(), &cams, 1);

    let rgb = full.read_tile(0, Channel::Rgb8).expect("rgb");
    let want = cpu_rs_full();
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
                "pixel ({px}, {py}) differs by {worst} but all four sub-samples hit {:?}:                  that is a shading divergence, not a coverage tie",
                hits[0]
            );
            assert!(
                worst <= 16,
                "pixel ({px}, {py}) is an edge pixel but differs by {worst} levels, more than                  one sub-sample of four can account for"
            );
            edges += 1;
            println!("  edge pixel ({px}, {py}): sub-sample hits {hits:?}, worst {worst} levels");
        }
    }
    let n_px = (TILE * TILE) as usize;
    assert!(
        edges * 1000 <= n_px,
        "{edges} of {n_px} pixels differ: more than the 0.1% of edge pixels the tie explains"
    );
    println!(
        "Full Rgb8: {diff} of {} bytes differ from the CPU, in {edges} of {n_px} edge pixels",
        rgb.len()
    );

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
