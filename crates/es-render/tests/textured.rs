//! Plan H, packet HT1: textures and metallic-roughness materials on both render paths.
//!
//! * **Placement** (oracle 2): `tests/fixtures/mjcf/textured/textured.xml` rendered by
//!   `MuJoCo`'s own renderer (`mujoco.Renderer`, offscreen) and by the CPU reference from the
//!   same camera: on each visible face of the two block cubes, and on the checker floor, the
//!   pixels' texel classes agree — which face of `block.png` sits where, which way up, and
//!   where the checker squares fall. Needs `ES_PYTHON` with `mujoco`; prints `SKIP` without.
//! * **BRDF** (oracle 3): a white furnace — a white PBR plane under a white sky, one bounce —
//!   whose `Pt` estimate converges to the directional albedo integrated by quadrature here, in
//!   `f64`, from the glTF formulas written out independently of `es_render::material`; the
//!   albedo stays at or below one. Metallic 0 and 1, roughness 0.05 to 1, NEE off and on.
//! * **GPU vs CPU** (oracle 4): the textured scene on `Rs` `Lambert` (bitwise), `Rs` `Full`
//!   (renderer.md 9.3's edge rule), `Pt` 1 spp (bitwise), `Pt` NEE and `ReSTIR` (1e-5 of the
//!   peak, renderer.md 5.1), and the furnace on the device.
//! * **Goldens** (oracle 5): `textured_*` written by `generate_textured_goldens` from the CPU
//!   reference, reproduced by the CPU and the GPU.

#![allow(clippy::many_single_char_names)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use es_assets::scene::SceneDesc;
use es_gpu::{Gpu, GpuOptions, SlangCompiler};
use es_math::{Pose, Quat, Vec3};
use es_render::{
    cpu, CameraView, Frame, ImageSpec, RenderConfig, RenderPath, Renderer, TileAtlasCfg, TriScene,
};
use es_sensor::Channel;

const TILE: u32 = 64;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf/textured")
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/render")
}

fn scene() -> SceneDesc {
    let xml = std::fs::read_to_string(fixture_dir().join("textured.xml")).expect("fixture");
    let mut scene = es_assets::parse_mjcf(&xml).expect("parses").scene;
    es_assets::mesh::load(&mut scene, &fixture_dir()).expect("the textures decode");
    scene
}

/// The MJCF camera in the `OpenCV` frame (MJCF looks down -Z with +Y up): a half turn about X,
/// as `shadow_hand.rs` does it.
fn view(scene: &SceneDesc, side: u32) -> CameraView {
    let camera = scene.cameras.iter().find(|c| c.name == "cam").expect("cam");
    CameraView {
        pose: camera
            .pose
            .compose(Pose::new(Vec3::ZERO, Quat::from_xyzw(1.0, 0.0, 0.0, 0.0))),
        spec: ImageSpec::pinhole(side, side, camera.fovy),
    }
}

fn atlas(side: u32) -> TileAtlasCfg {
    TileAtlasCfg::row(side, side, 1)
}

/// `Some(gpu)` or a printed SKIP.
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

fn srgb_decode(b: u8) -> f32 {
    let c = f32::from(b) / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// --- oracle 2: placement against MuJoCo ---------------------------------------------------------

const MUJOCO: &str = r"
import sys
import mujoco
path, side = sys.argv[1], int(sys.argv[2])
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, side, side)
r.update_scene(d, camera='cam')
sys.stdout.buffer.write(r.render().tobytes())
";

fn mujoco_render(side: u32) -> Result<Vec<u8>, String> {
    let python = match std::env::var("ES_PYTHON") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => "python".to_owned(),
    };
    let out = Command::new(&python)
        .args(["-c", MUJOCO])
        .arg(fixture_dir().join("textured.xml"))
        .arg(side.to_string())
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("`{python}`: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_owned());
    }
    if out.stdout.len() != (side * side * 3) as usize {
        return Err(format!("{} bytes from MuJoCo", out.stdout.len()));
    }
    Ok(out.stdout)
}

/// A pixel's colour with its brightness divided out: what survives two different lighting
/// models. Linear RGB over its largest channel.
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

/// Oracle 2. The block's six backdrop colours and the letters' white are the palette of the
/// cubes, the checker's two colours the floor's. Every pixel of a face region (a cube's
/// segmentation id and one world-axis normal, the region's border dropped) is classified in
/// both images; the two must agree on the region's dominant colour and on at least 85 % of
/// its pixels — which is a statement about the face (colour) *and* its orientation (where
/// the white letter's pixels fall).
#[test]
fn mujoco_places_the_textures_where_we_do() {
    const SIDE: u32 = 160;
    let theirs = match mujoco_render(SIDE) {
        Ok(img) => img,
        Err(why) => {
            println!("SKIP mujoco_places_the_textures_where_we_do: {why}");
            return;
        }
    };
    let scene = scene();
    let tri = TriScene::from_scene(&scene).expect("tessellates");
    let cam = view(&scene, SIDE);
    let ours = cpu::rasterize(&tri, &cam, &RenderConfig::rs(atlas(SIDE)), 0);
    let rgb = ours.tile(Channel::Rgb8).unwrap().as_u8().unwrap();
    let seg = ours
        .tile(Channel::SegmentationId)
        .unwrap()
        .as_u32()
        .unwrap();
    let normal = ours.tile(Channel::Normal).unwrap().as_f32().unwrap();
    if let Ok(dir) = std::env::var("ES_RENDER_DUMP") {
        let dir = PathBuf::from(dir);
        std::fs::write(dir.join("textured_ours.rgb"), rgb).unwrap();
        std::fs::write(dir.join("textured_mujoco.rgb"), &theirs).unwrap();
    }

    // The palettes, in chroma space. Cube: the backdrop of each face of block.png (the
    // decoded texels at a corner of each face), and white.
    let block_id = scene
        .assets
        .iter()
        .find(|a| a.name == "block" && a.kind == es_assets::scene::AssetKind::Texture)
        .unwrap()
        .id;
    let block = scene.textures[&block_id].data.as_ref().unwrap();
    let mut cube_palette: Vec<[f32; 3]> = (0..6)
        .map(|f| chroma(block.texel(f, 5, 5).map(|b| f32::from(b) / 255.0)))
        .collect();
    cube_palette.push([1.0; 3]);
    let floor_palette = [chroma([0.9, 0.2, 0.1]), chroma([0.1, 0.3, 0.9])];

    let seg_of = |name: &str| *tri.names.iter().find(|(_, n)| *n == name).unwrap().0;
    let (a, b, floor) = (seg_of("cube_a"), seg_of("cube_b"), seg_of("floor"));
    let world_axis = |i: usize| {
        let n = [normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2]];
        let q = cam.pose.orientation;
        let w = q.rotate(Vec3::new(f64::from(n[0]), f64::from(n[1]), f64::from(n[2])));
        let c = [w.x, w.y, w.z];
        let k = (0..3)
            .max_by(|x, y| c[*x].abs().total_cmp(&c[*y].abs()))
            .unwrap();
        (k, c[k] > 0.0)
    };
    let s = SIDE as usize;
    let interior = |i: usize| {
        let (x, y) = (i % s, i / s);
        if x == 0 || y == 0 || x + 1 == s || y + 1 == s {
            return false;
        }
        [i - 1, i + 1, i - s, i + s]
            .iter()
            .all(|j| seg[*j] == seg[i] && world_axis(*j) == world_axis(i))
    };
    let ours_lin = |i: usize| [0, 1, 2].map(|c| srgb_decode(rgb[i * 3 + c]));
    let theirs_lin = |i: usize| [0, 1, 2].map(|c| f32::from(theirs[i * 3 + c]) / 255.0);

    // Regions: (seg, axis, sign) for the cubes, the floor as one region.
    let mut regions: BTreeMap<(u32, usize, bool), Vec<usize>> = BTreeMap::new();
    #[allow(clippy::needless_range_loop)] // `i` indexes three tiles and the neighbour test
    for i in 0..s * s {
        if !interior(i) {
            continue;
        }
        if seg[i] == a || seg[i] == b {
            let (k, pos) = world_axis(i);
            regions.entry((seg[i], k, pos)).or_default().push(i);
        } else if seg[i] == floor {
            regions.entry((floor, 2, true)).or_default().push(i);
        }
    }
    let mut faces_seen = std::collections::BTreeSet::new();
    for ((id, k, pos), pixels) in &regions {
        let palette: &[[f32; 3]] = if *id == floor {
            &floor_palette
        } else {
            &cube_palette
        };
        let classes = |lin: &dyn Fn(usize) -> [f32; 3]| -> Vec<usize> {
            pixels
                .iter()
                .map(|i| nearest(chroma(lin(*i)), palette))
                .collect()
        };
        let (mine, mj) = (classes(&ours_lin), classes(&theirs_lin));
        let dominant = |c: &[usize]| {
            let mut n = vec![0usize; palette.len()];
            for x in c {
                if *id == floor || *x < 6 {
                    n[*x] += 1;
                }
            }
            (0..n.len()).max_by_key(|j| n[*j]).unwrap()
        };
        let agree =
            mine.iter().zip(&mj).filter(|(x, y)| x == y).count() as f64 / pixels.len() as f64;
        let name = tri.names[id].as_str();
        let axis = format!("{}{}", if *pos { '+' } else { '-' }, ['x', 'y', 'z'][*k]);
        println!(
            "{name} {axis}: {} px, dominant ours {} / MuJoCo {}, agreement {:.3}",
            pixels.len(),
            dominant(&mine),
            dominant(&mj),
            agree
        );
        if pixels.len() < 40 {
            continue; // a sliver seen edge-on says nothing about placement
        }
        assert_eq!(
            dominant(&mine),
            dominant(&mj),
            "{name} {axis}: a different face of the texture"
        );
        assert!(
            agree >= 0.85,
            "{name} {axis}: only {agree:.3} of the pixels agree"
        );
        if *id != floor {
            faces_seen.insert(dominant(&mine));
        }
    }
    assert_eq!(
        faces_seen.len(),
        6,
        "all six faces of the block, between the two cubes"
    );
}

// --- oracle 3: the white furnace ------------------------------------------------------------------

fn fresnel(f0: f64, c: f64) -> f64 {
    f0 + (1.0 - f0) * (1.0 - c).powi(5)
}

fn half(v: [f64; 3], l: [f64; 3]) -> [f64; 3] {
    let s = [v[0] + l[0], v[1] + l[1], v[2] + l[2]];
    let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    s.map(|x| x / len)
}

/// The specular lobe `F(v.h) D V` in `f64`, from glTF 2.0 Appendix B (base colour 1), written
/// out here rather than taken from `es_render::material`: the reference is independent.
fn spec_part(metallic: f64, rough: f64, n_v: f64, l: [f64; 3], v: [f64; 3]) -> f64 {
    let nl = l[2];
    if nl <= 0.0 || n_v <= 0.0 {
        return 0.0;
    }
    let a2 = (rough * rough).powi(2);
    let h = half(v, l);
    let vh = (v[0] * h[0] + v[1] * h[1] + v[2] * h[2]).max(0.0);
    let f = fresnel(0.04 * (1.0 - metallic) + metallic, vh);
    let dd = h[2].max(0.0).powi(2) * (a2 - 1.0) + 1.0;
    let d = a2 / (std::f64::consts::PI * dd * dd);
    let vis = 0.5
        / (nl * (n_v * n_v * (1.0 - a2) + a2).sqrt() + n_v * (nl * nl * (1.0 - a2) + a2).sqrt());
    f * d * vis
}

/// Directional albedo `rho(v) = int f(l, v) cos dw` by deterministic quadrature: the diffuse
/// share over a cosine-mapped grid, the specular share over a grid in GGX's `h` distribution
/// (`dw_l = 4 v.h dw_h`), both midpoint rules, `f64` throughout.
fn albedo_ref(metallic: f64, rough: f64, cos_v: f64, gltf: bool) -> f64 {
    const N: usize = 256;
    let v = [(1.0 - cos_v * cos_v).max(0.0).sqrt(), 0.0, cos_v];
    let a = rough * rough;
    let a2 = a * a;
    let pi = std::f64::consts::PI;
    let (mut diffuse, mut specular) = (0.0, 0.0);
    for i in 0..N {
        for j in 0..N {
            let (u1, u2) = ((i as f64 + 0.5) / N as f64, (j as f64 + 0.5) / N as f64);
            let phi = 2.0 * pi * u2;
            // Cosine-weighted l: pdf cos / pi, so the weight is pi f cos / cos.
            let (r, z) = (u1.sqrt(), (1.0 - u1).sqrt());
            let l = [r * phi.cos(), r * phi.sin(), z];
            diffuse += diffuse_part(metallic, cos_v, l, v, gltf) * pi;
            // h from GGX's D(h) (n.h): cos^2 = (1 - u) / (1 + (a^2 - 1) u).
            let c2 = (1.0 - u1) / (1.0 + (a2 - 1.0) * u1);
            let (ch, sh) = (c2.sqrt(), (1.0 - c2).max(0.0).sqrt());
            let h = [sh * phi.cos(), sh * phi.sin(), ch];
            let vh = v[0] * h[0] + v[1] * h[1] + v[2] * h[2];
            if vh <= 0.0 {
                continue;
            }
            let l = [0, 1, 2].map(|k| 2.0 * vh * h[k] - v[k]);
            if l[2] <= 0.0 {
                continue;
            }
            let dd = ch * ch * (a2 - 1.0) + 1.0;
            let d = a2 / (pi * dd * dd);
            // E over h ~ D (n.h) of  f_s(l) cos_l 4 v.h / (D (n.h)).
            specular += spec_part(metallic, rough, cos_v, l, v) * l[2] * 4.0 * vh / (d * ch);
        }
    }
    (diffuse + specular) / (N * N) as f64
}

/// The Lambert lobe. `gltf`: Appendix B's `(1 - F(v.h))`; otherwise the renderer's
/// `(1 - F(n.l)) (1 - F(n.v))`.
fn diffuse_part(metallic: f64, n_v: f64, l: [f64; 3], v: [f64; 3], gltf: bool) -> f64 {
    if l[2] <= 0.0 || n_v <= 0.0 {
        return 0.0;
    }
    let f0 = 0.04 * (1.0 - metallic) + metallic;
    let keep = if gltf {
        let h = half(v, l);
        1.0 - fresnel(f0, (v[0] * h[0] + v[1] * h[1] + v[2] * h[2]).max(0.0))
    } else {
        (1.0 - fresnel(f0, l[2])) * (1.0 - fresnel(f0, n_v))
    };
    keep * (1.0 - metallic) / std::f64::consts::PI
}

/// A white plane of material `(metallic, roughness)` under a white sky, seen at 60 degrees
/// from the normal by an 8x8 camera: every pixel's first hit is the plane, every bounce
/// escapes, so a pixel's estimate is the plane's directional albedo at that pixel's view.
fn furnace(metallic: f64, rough: f64) -> (TriScene, CameraView) {
    let xml = format!(
        r#"<mujoco><asset><material name="m" rgba="1 1 1 1" metallic="{metallic}" roughness="{rough}"/></asset>
           <worldbody><geom name="plane" type="plane" size="0 0 1" material="m"/></worldbody></mujoco>"#
    );
    let scene = es_assets::parse_mjcf(&xml).unwrap().scene;
    let tri = TriScene::from_scene(&scene).unwrap();
    // Camera at height 1 looking 60 degrees off the vertical, narrow field of view.
    let tilt = 60.0f64.to_radians();
    let (s, c) = ((tilt / 2.0).sin(), (tilt / 2.0).cos());
    // OpenCV frame: +Z forward. Start looking down (-Z world): a half turn about X, then tilt
    // about the camera's X.
    let down = Quat::from_xyzw(1.0, 0.0, 0.0, 0.0);
    let tilt_q = Quat::from_xyzw(s, 0.0, 0.0, c);
    let pose = Pose::new(Vec3::new(0.0, 0.0, 1.0), down * tilt_q);
    let cam = CameraView {
        pose,
        spec: ImageSpec::pinhole(8, 8, 0.1),
    };
    (tri, cam)
}

fn furnace_cfg(spp: u32, nee: bool) -> RenderConfig {
    let mut cfg = if nee {
        RenderConfig::pt_nee(atlas(8), spp, 2)
    } else {
        RenderConfig::pt(atlas(8), spp, 2)
    };
    cfg.sky = [1.0, 1.0, 1.0];
    cfg
}

fn mean(xs: &[f32]) -> f64 {
    xs.iter().map(|x| f64::from(*x)).sum::<f64>() / xs.len() as f64
}

/// Oracle 3, on the CPU reference.
#[test]
fn white_furnace_converges_to_the_quadrature_albedo() {
    // The view is 60 degrees off the normal at the centre; the 8x8 tile spans +-0.05 rad.
    let cos_v = 60.0f64.to_radians().cos();
    for metallic in [0.0, 1.0] {
        for rough in [0.05, 0.25, 0.5, 0.75, 1.0] {
            let reference = albedo_ref(metallic, rough, cos_v, false);
            let gltf = albedo_ref(metallic, rough, cos_v, true);
            println!("quadrature m{metallic} r{rough}: rho {reference:.5}; with glTF's (1 - F(v.h)) {gltf:.5}");
            assert!(
                reference <= 1.0 + 1e-9,
                "m{metallic} r{rough}: rho {reference} > 1"
            );
            let (tri, cam) = furnace(metallic, rough);
            for nee in [false, true] {
                let frame = cpu::path_trace(&tri, &cam, &furnace_cfg(512, nee), 0);
                let r = frame.tile(Channel::PtRadiance).unwrap().as_f32().unwrap();
                let got = mean(r);
                let rel = (got - reference).abs() / reference;
                println!(
                    "furnace metallic {metallic} roughness {rough} nee {nee}: Pt {got:.5}, quadrature {reference:.5}, rel {rel:.2e}"
                );
                assert!(got <= 1.0 + 5e-3, "an estimate above one: {got}");
                assert!(
                    rel < 1e-2,
                    "m{metallic} r{rough} nee {nee}: {got} vs {reference}"
                );
            }
        }
    }
}

// --- oracle 4: the GPU against the CPU ------------------------------------------------------------

fn gpu_frame(
    gpu: &Gpu,
    tri: &TriScene,
    cam: &CameraView,
    cfg: RenderConfig,
) -> BTreeMap<Channel, es_render::Tile> {
    let channels: Vec<Channel> = cfg.channels.iter().copied().collect();
    let mut r = Renderer::new(gpu, cfg).expect("renderer");
    r.upload_tris(tri.clone()).expect("upload");
    let mut atlas = r.render(std::slice::from_ref(cam)).expect("render");
    channels
        .into_iter()
        .map(|c| (c, atlas.read_tile(0, c).expect("tile")))
        .collect()
}

fn max_ulp(a: &[f32], b: &[f32]) -> u64 {
    let key = |v: f32| {
        let bits = i64::from(v.to_bits());
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    };
    a.iter()
        .zip(b)
        .map(|(x, y)| (key(*x) - key(*y)).unsigned_abs())
        .max()
        .unwrap_or(0)
}

fn max_normalized(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
        / scale
}

fn differing_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn pt_cfg(spp: u32, bounces: u32, nee: bool, restir: bool) -> RenderConfig {
    let mut cfg = RenderConfig::pt(atlas(TILE), spp, bounces);
    cfg.path = RenderPath::Pt {
        spp,
        bounces,
        nee,
        restir,
        svgf: false,
    };
    cfg.sky = [0.3, 0.35, 0.4];
    cfg.exposure = 4.0;
    cfg
}

#[test]
fn gpu_matches_the_cpu_on_textures_and_materials() {
    let test = "gpu_matches_the_cpu_on_textures_and_materials";
    let Some(gpu) = open(test) else { return };
    let scene = scene();
    let tri = TriScene::from_scene(&scene).expect("tessellates");
    assert!(
        tri.tris.iter().any(|t| t.mat != 0),
        "the scene wears materials"
    );
    let cam = view(&scene, TILE);

    // Rs Lambert: `Rgb8` and segmentation bitwise; depth and normal to renderer.md 14.4's
    // relative 1e-5 at a camera whose quaternion is not a golden's (the `dot` order of 9.3).
    let cfg = RenderConfig::rs(atlas(TILE));
    let want = cpu::rasterize(&tri, &cam, &cfg, 0);
    let got = gpu_frame(&gpu, &tri, &cam, cfg);
    for (channel, tile) in &got {
        let w = want.tile(*channel).unwrap();
        if let (Some(g), Some(w)) = (tile.as_f32(), w.as_f32()) {
            let rel = g
                .iter()
                .zip(w)
                .map(|(x, y)| (x - y).abs() / y.abs().max(1e-6))
                .fold(0.0f32, f32::max);
            println!(
                "Rs Lambert {channel:?}: max ULP {}, max relative {rel:e}",
                max_ulp(g, w)
            );
            assert!(rel <= 1e-5, "Rs Lambert {channel:?}");
        } else {
            let n = differing_bytes(&tile.to_bytes(), &w.to_bytes());
            println!("Rs Lambert {channel:?}: {n} bytes differ");
            assert_eq!(n, 0, "Rs Lambert {channel:?} differs");
        }
    }

    // Rs Full: renderer.md 9.3's rule -- a differing pixel is one whose sub-samples straddle
    // a triangle edge, and no more than 0.1 % of them.
    let cfg = RenderConfig::rs_full(atlas(TILE));
    let want = cpu::rasterize(&tri, &cam, &cfg, 0);
    let got = gpu_frame(&gpu, &tri, &cam, cfg);
    let (a, b) = (
        got[&Channel::Rgb8].as_u8().unwrap(),
        want.tile(Channel::Rgb8).unwrap().as_u8().unwrap(),
    );
    let pixels = a.chunks(3).zip(b.chunks(3)).filter(|(x, y)| x != y).count();
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0);
    println!(
        "Rs Full: {} of {} bytes differ, {pixels} of {} pixels, worst {worst} levels",
        differing_bytes(a, b),
        a.len(),
        TILE * TILE
    );
    assert!(pixels * 1000 <= (TILE * TILE) as usize && worst <= 64);

    // Pt 1 spp, no NEE: the estimator's own arithmetic, bitwise.
    let cfg = pt_cfg(1, 3, false, false);
    let want = cpu::path_trace(&tri, &cam, &cfg, 0);
    let got = gpu_frame(&gpu, &tri, &cam, cfg);
    let (g, w) = (
        got[&Channel::PtRadiance].as_f32().unwrap(),
        want.tile(Channel::PtRadiance).unwrap().as_f32().unwrap(),
    );
    println!(
        "Pt 1 spp: PtRadiance max ULP {}, normalized {:e}",
        max_ulp(g, w),
        max_normalized(g, w)
    );
    assert!(max_normalized(g, w) <= 1e-5);

    // Pt NEE and ReSTIR: renderer.md 5.1's 1e-5 of the peak.
    for (name, cfg) in [
        ("Pt NEE 4 spp", pt_cfg(4, 3, true, false)),
        ("Pt ReSTIR 1 spp", pt_cfg(1, 2, false, true)),
    ] {
        let want = cpu::path_trace(&tri, &cam, &cfg, 0);
        let got = gpu_frame(&gpu, &tri, &cam, cfg);
        let (g, w) = (
            got[&Channel::PtRadiance].as_f32().unwrap(),
            want.tile(Channel::PtRadiance).unwrap().as_f32().unwrap(),
        );
        let (rg, rw) = (
            got[&Channel::Rgb8].as_u8().unwrap(),
            want.tile(Channel::Rgb8).unwrap().as_u8().unwrap(),
        );
        println!(
            "{name}: PtRadiance max ULP {}, normalized {:e}; Rgb8 {} of {} bytes differ",
            max_ulp(g, w),
            max_normalized(g, w),
            differing_bytes(rg, rw),
            rg.len()
        );
        assert!(max_normalized(g, w) <= 1e-5, "{name}");
        assert!(differing_bytes(rg, rw) * 1000 <= rg.len(), "{name}");
    }
}

/// Oracle 3 on the device: the GPU's furnace mean is the CPU's.
#[test]
fn gpu_white_furnace_matches_the_cpu() {
    let test = "gpu_white_furnace_matches_the_cpu";
    let Some(gpu) = open(test) else { return };
    for (metallic, rough) in [(0.0, 0.25), (1.0, 0.5), (0.0, 1.0)] {
        let (tri, cam) = furnace(metallic, rough);
        for nee in [false, true] {
            let cfg = furnace_cfg(64, nee);
            let want = cpu::path_trace(&tri, &cam, &cfg, 0);
            let got = gpu_frame(&gpu, &tri, &cam, cfg);
            let (g, w) = (
                got[&Channel::PtRadiance].as_f32().unwrap(),
                want.tile(Channel::PtRadiance).unwrap().as_f32().unwrap(),
            );
            println!(
                "GPU furnace m{metallic} r{rough} nee {nee}: mean {:.5} vs CPU {:.5}, max ULP {}, normalized {:e}",
                mean(g),
                mean(w),
                max_ulp(g, w),
                max_normalized(g, w)
            );
            assert!(max_normalized(g, w) <= 1e-5);
        }
    }
}

// --- oracle 5: goldens ----------------------------------------------------------------------------

struct Golden {
    name: &'static str,
    kernel: &'static str,
}

const GOLDENS: [Golden; 3] = [
    Golden {
        name: "textured_rs_rgb8",
        kernel: "raster.v2 (Lambert, textured)",
    },
    Golden {
        name: "textured_rs_full_rgb8",
        kernel: "raster.v2 (Shading::Full, ssaa 2, textured, glTF metallic-roughness)",
    },
    Golden {
        name: "textured_pt_nee_rgb8",
        kernel: "pt.v4 (4 spp, 3 bounces, NEE, sky 0.3 0.35 0.4, Reinhard, exposure 4, glTF metallic-roughness)",
    },
];

fn golden_cfg(name: &str) -> RenderConfig {
    match name {
        "textured_rs_rgb8" => RenderConfig::rs(atlas(TILE)),
        "textured_rs_full_rgb8" => RenderConfig::rs_full(atlas(TILE)),
        _ => pt_cfg(4, 3, true, false),
    }
}

fn cpu_golden(name: &str, tri: &TriScene, cam: &CameraView) -> Frame {
    let cfg = golden_cfg(name);
    match cfg.path {
        RenderPath::Rs => cpu::rasterize(tri, cam, &cfg, 0),
        RenderPath::Pt { .. } => cpu::path_trace(tri, cam, &cfg, 0),
    }
}

/// Writes the `textured_*` goldens from the CPU reference. Its own generator, so it cannot
/// rewrite an older golden:
///
///     ES_GENERATE_GOLDENS=1 cargo test -p es-render --test textured -- --ignored generate_textured_goldens
#[test]
#[ignore = "golden generator; run explicitly"]
fn generate_textured_goldens() {
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_textured_goldens: set ES_GENERATE_GOLDENS=1");
        return;
    }
    let scene = scene();
    let tri = TriScene::from_scene(&scene).expect("tessellates");
    let cam = view(&scene, TILE);
    for g in &GOLDENS {
        let tile = cpu_golden(g.name, &tri, &cam)
            .tile(Channel::Rgb8)
            .unwrap()
            .clone();
        std::fs::write(
            golden_dir().join(format!("{}.bin", g.name)),
            tile.to_bytes(),
        )
        .unwrap();
        let sidecar = serde_json::json!({
            "name": g.name,
            "dtype": "u8",
            "layout": "row-major, little-endian, tightly packed",
            "shape": tile.shape,
            "kernel": g.kernel,
            "scene": "tests/fixtures/mjcf/textured/textured.xml, camera `cam`",
            "pins": "OpenCV camera frame and top-left image origin (spec 3.1)",
            "oracle": "es_render::cpu, the pure-Rust mirror of the Slang kernels",
            "generator": "cargo test -p es-render --test textured -- --ignored generate_textured_goldens",
            "spec": "docs/design/renderer.md section 15",
        });
        std::fs::write(
            golden_dir().join(format!("{}.json", g.name)),
            format!("{}\n", serde_json::to_string_pretty(&sidecar).unwrap()),
        )
        .unwrap();
        println!("wrote {}", g.name);
    }
}

#[test]
fn textured_goldens_are_reproduced_by_the_cpu_and_the_gpu() {
    let scene = scene();
    let tri = TriScene::from_scene(&scene).expect("tessellates");
    let cam = view(&scene, TILE);
    let gpu = open("textured_goldens_are_reproduced_by_the_cpu_and_the_gpu");
    for g in &GOLDENS {
        let path = golden_dir().join(format!("{}.bin", g.name));
        let expected = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run generate_textured_goldens)", path.display()));
        let cpu = cpu_golden(g.name, &tri, &cam)
            .tile(Channel::Rgb8)
            .unwrap()
            .to_bytes();
        assert!(cpu == expected, "{}: the CPU reference moved", g.name);
        if let Some(gpu) = &gpu {
            let got = gpu_frame(gpu, &tri, &cam, golden_cfg(g.name))[&Channel::Rgb8].to_bytes();
            let n = differing_bytes(&got, &expected);
            println!(
                "{}: GPU vs golden, {n} of {} bytes differ",
                g.name,
                got.len()
            );
            let allowed = if g.name == "textured_rs_rgb8" {
                0
            } else {
                got.len() / 1000
            };
            assert!(n <= allowed, "{}: {n} bytes", g.name);
        }
    }
}
