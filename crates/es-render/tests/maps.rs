//! Plan H, packet HT2: normal and emissive maps, and glTF materials, on both render paths.
//!
//! * **Normal map vs geometry** (oracle 2): a flat plane wearing the normal map of a height
//!   field, against the same height field tessellated and displaced, whose facet normals are
//!   what the map encodes: the `Rs` `Lambert` images agree, where the flat plane without the
//!   map does not.
//! * **A textured emitter** (oracle 3): an emissive-mapped quad lights a diffuse floor on `Pt`;
//!   NEE on and off converge to the same image, and the floor under the quad to the radiance
//!   integrated here by quadrature, from a bilinear written out in the test.
//! * **glTF = MJCF** (oracle 4): `tests/fixtures/gltf/textured_box/` declares one box and its
//!   four maps in glTF and in MJCF; both import to the same triangles and material table and
//!   render the same bytes.
//! * **GPU vs CPU** (oracle 5) on all three scenes, at `renderer.md` 15.4's tolerances.
//! * **Goldens** (oracle 6): `maps_*`, written by `generate_maps_goldens` from the CPU
//!   reference, reproduced by the CPU and the GPU.

#![allow(clippy::many_single_char_names, clippy::cast_precision_loss)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use es_assets::scene::{scene_id, AssetKind, AssetRef, Body, Geom, Material, SceneDesc, Shape};
use es_assets::texture::{TexKind, Texture, TextureData, TextureSpec, Wrap};
use es_core::StableId;
use es_gpu::{Gpu, GpuOptions, SlangCompiler};
use es_math::{Pose, Quat, Vec3};
use es_render::{
    cpu, CameraView, Frame, ImageSpec, RenderConfig, RenderPath, Renderer, TileAtlasCfg, TriScene,
};
use es_sensor::Channel;

// --- scene building ------------------------------------------------------------------------------

fn geom(name: &str, shape: Shape, pose: Pose, rgba: [f64; 4], material: Option<StableId>) -> Geom {
    Geom {
        id: scene_id("geom", name),
        name: name.to_owned(),
        shape,
        pose,
        friction: [1.0, 0.005, 0.0001],
        contype: 0,
        conaffinity: 0,
        condim: 3,
        priority: 0,
        density: 1000.0,
        mass: None,
        margin: 0.0,
        gap: 0.0,
        solref: [0.02, 1.0],
        solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
        material,
        rgba,
        visual_only: true,
    }
}

fn world(geoms: Vec<Geom>) -> SceneDesc {
    SceneDesc {
        name: "maps".to_owned(),
        bodies: vec![Body {
            id: scene_id("body", "world"),
            name: "world".to_owned(),
            parent: None,
            pose: Pose::IDENTITY,
            inertial: None,
            geoms,
            sites: Vec::new(),
        }],
        ..SceneDesc::default()
    }
}

/// A decoded 2D texture on `scene`, linear unless `srgb`.
fn add_texture(
    scene: &mut SceneDesc,
    name: &str,
    side: u32,
    rgb: Vec<u8>,
    wrap: [Wrap; 2],
) -> StableId {
    let data = TextureData {
        kind: TexKind::TwoD,
        width: side,
        height: side,
        srgb: false,
        rgb,
        wrap,
    };
    let id = scene_id("asset", &format!("texture/{name}"));
    scene.assets.push(AssetRef {
        id,
        name: name.to_owned(),
        kind: AssetKind::Texture,
        path: format!("{name}.png"),
        hash: data.content_hash(),
    });
    scene.textures.insert(
        id,
        Texture {
            spec: TextureSpec {
                kind: TexKind::TwoD,
                ..TextureSpec::default()
            },
            data: Some(data),
        },
    );
    id
}

fn add_material(scene: &mut SceneDesc, name: &str, m: Material) -> StableId {
    let id = scene_id("asset", &format!("material/{name}"));
    scene.materials.insert(id, m);
    id
}

/// A camera at `pos` looking straight down (`OpenCV` frame: a half turn about X, so the image's
/// top is world +Y).
fn down_camera(pos: Vec3, side: u32, fovy: f64) -> CameraView {
    CameraView {
        pose: Pose::new(pos, Quat::from_xyzw(1.0, 0.0, 0.0, 0.0)),
        spec: ImageSpec::pinhole(side, side, fovy),
    }
}

fn atlas(side: u32) -> TileAtlasCfg {
    TileAtlasCfg::row(side, side, 1)
}

fn rgb8(frame: &Frame) -> &[u8] {
    frame.tile(Channel::Rgb8).unwrap().as_u8().unwrap()
}

fn radiance(frame: &Frame) -> &[f32] {
    frame.tile(Channel::PtRadiance).unwrap().as_f32().unwrap()
}

// --- oracle 2: a normal map against the geometry it stands for -----------------------------------

const BUMP_N: usize = 64;
const BUMP_HALF: f64 = 0.5;
const BUMP_SIDE: u32 = 64;

/// The height field: a product of sines, 1.5 periods per metre, slopes up to ~0.28.
fn height(x: f64, y: f64) -> f64 {
    let k = 2.0 * std::f64::consts::PI * 1.5;
    0.03 * (k * x).sin() * (k * y).sin()
}

/// Texel `(r, c)`'s corners: `x` left to right, `y` top to bottom (row 0 is the image's top,
/// world +Y, as the plane's UVs put it).
fn texel_corners(r: usize, c: usize) -> (f64, f64, f64, f64) {
    let step = 2.0 * BUMP_HALF / BUMP_N as f64;
    let x0 = -BUMP_HALF + c as f64 * step;
    let y0 = BUMP_HALF - r as f64 * step;
    (x0, x0 + step, y0, y0 - step)
}

/// The normal map: per texel, the finite-difference normal of the height field over the
/// texel's four corners, which is the plane through the mesh quad that texel covers.
fn bump_texels() -> Vec<u8> {
    let step = 2.0 * BUMP_HALF / BUMP_N as f64;
    let mut out = Vec::with_capacity(BUMP_N * BUMP_N * 3);
    for r in 0..BUMP_N {
        for c in 0..BUMP_N {
            let (x0, x1, yt, yb) = texel_corners(r, c);
            let gx = ((height(x1, yt) + height(x1, yb)) - (height(x0, yt) + height(x0, yb)))
                / (2.0 * step);
            let gy = ((height(x0, yt) + height(x1, yt)) - (height(x0, yb) + height(x1, yb)))
                / (2.0 * step);
            let n = [-gx, -gy, 1.0];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            for v in n {
                out.push(((v / len * 0.5 + 0.5) * 255.0).round() as u8);
            }
        }
    }
    out
}

fn plane(half: f64) -> Shape {
    Shape::Plane {
        half_x: half,
        half_y: half,
        grid: 0.0,
    }
}

/// The flat plane, with the normal map when `mapped` (its green channel turned over when
/// `flip_green`: the other handedness of the bitangent).
fn bump_plane(mapped: bool, flip_green: bool) -> SceneDesc {
    let mut scene = world(Vec::new());
    let material = if mapped {
        let mut texels = bump_texels();
        if flip_green {
            for px in texels.chunks_mut(3) {
                px[1] = 255 - px[1];
            }
        }
        let tex = add_texture(&mut scene, "bump", BUMP_N as u32, texels, [Wrap::Repeat; 2]);
        Some(add_material(
            &mut scene,
            "bumpy",
            Material {
                rgba: [0.8, 0.8, 0.8, 1.0],
                normal_map: Some(tex),
                ..Material::default()
            },
        ))
    } else {
        None
    };
    scene.bodies[0].geoms.push(geom(
        "floor",
        plane(BUMP_HALF),
        Pose::IDENTITY,
        [0.8, 0.8, 0.8, 1.0],
        material,
    ));
    scene
}

/// The height field itself: a vertex at every texel corner, two triangles per texel.
fn bump_mesh() -> SceneDesc {
    let n = BUMP_N + 1;
    let step = 2.0 * BUMP_HALF / BUMP_N as f64;
    let mut positions = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let (x, y) = (-BUMP_HALF + i as f64 * step, -BUMP_HALF + j as f64 * step);
            positions.push([x as f32, y as f32, height(x, y) as f32]);
        }
    }
    let mut indices = Vec::new();
    for j in 0..BUMP_N {
        for i in 0..BUMP_N {
            let a = (j * n + i) as u32;
            let (b, c, d) = (a + 1, a + 1 + n as u32, a + n as u32);
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let mut scene = world(Vec::new());
    let asset = scene_id("asset", "mesh/heightfield");
    scene.assets.push(AssetRef {
        id: asset,
        name: "heightfield".to_owned(),
        kind: AssetKind::Mesh,
        path: "heightfield.obj".to_owned(),
        hash: [9; 32],
    });
    scene.meshes.insert(
        asset,
        es_assets::gltf::MeshData {
            id: asset,
            name: "heightfield".to_owned(),
            positions,
            normals: None,
            uvs: None,
            indices,
            material: None,
        },
    );
    scene.bodies[0].geoms.push(geom(
        "floor",
        Shape::Mesh { asset },
        Pose::IDENTITY,
        [0.8, 0.8, 0.8, 1.0],
        None,
    ));
    scene
}

fn bump_camera() -> CameraView {
    // Straight down from 1.5 m, seeing +-0.35 m of the plane: no silhouette in the frame.
    down_camera(
        Vec3::new(0.0, 0.0, 1.5),
        BUMP_SIDE,
        2.0 * (0.35f64 / 1.5).atan(),
    )
}

fn abs_diffs(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).collect()
}

fn mean_u8(d: &[u8]) -> f64 {
    d.iter().map(|x| f64::from(*x)).sum::<f64>() / d.len() as f64
}

fn percentile(d: &[u8], q: f64) -> u8 {
    let mut s = d.to_vec();
    s.sort_unstable();
    s[((s.len() - 1) as f64 * q) as usize]
}

/// Oracle 2 on the CPU reference.
#[test]
fn a_normal_map_shades_like_the_geometry_it_encodes() {
    let cam = bump_camera();
    let cfg = RenderConfig::rs(atlas(BUMP_SIDE));
    let render = |s: SceneDesc| cpu::rasterize(&TriScene::from_scene(&s).unwrap(), &cam, &cfg, 0);
    let a = render(bump_plane(true, false));
    let f = render(bump_plane(false, false));
    let flipped = render(bump_plane(true, true));
    let m = render(bump_mesh());
    let bump = abs_diffs(rgb8(&a), rgb8(&m));
    let control = abs_diffs(rgb8(&f), rgb8(&m));
    let other = abs_diffs(rgb8(&flipped), rgb8(&m));
    for (what, d) in [
        ("normal map", &bump),
        ("flat plane", &control),
        ("green turned over", &other),
    ] {
        println!(
            "{what} vs displaced mesh: mean {:.3}, p99 {}, max {} levels",
            mean_u8(d),
            percentile(d, 0.99),
            d.iter().max().unwrap()
        );
    }
    // The geometry channels are the flat plane's: the map shades, it does not move surfaces.
    for c in [Channel::SegmentationId, Channel::Normal] {
        assert_eq!(
            a.tile(c).unwrap().to_bytes(),
            f.tile(c).unwrap().to_bytes(),
            "{c:?} moved under a normal map"
        );
    }
    assert!(
        mean_u8(&control) > 8.0 * mean_u8(&bump),
        "the map changes too little"
    );
    assert!(
        mean_u8(&other) > 8.0 * mean_u8(&bump),
        "the bitangent's handedness is not pinned"
    );
    assert!(mean_u8(&bump) <= 1.0 && percentile(&bump, 0.99) <= 4);
}

// --- oracle 3: an emissive-mapped quad -------------------------------------------------------

const QUAD_HALF: f64 = 0.2;
const QUAD_Z: f64 = 0.4;
const GLOW_SIDE: u32 = 4;
const GLOW_REPEAT: f64 = 1.5;
const GLOW: [f64; 3] = [3.0, 2.5, 2.0];
const FLOOR: f64 = 0.8;

fn glow_texels() -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..(GLOW_SIDE * GLOW_SIDE) {
        let v = [
            (i * 67 + 13) % 256,
            (i * 151 + 90) % 256,
            (i * 37 + 200) % 256,
        ];
        out.extend(v.map(|x| if i % 5 == 3 { 0 } else { x as u8 }));
    }
    out
}

/// A floor, and above it a quad facing down whose emission is `GLOW` times a 4 x 4 texture
/// repeated 1.5 times, mirrored along `s` and clamped along `t`.
fn glow_scene() -> SceneDesc {
    let mut scene = world(Vec::new());
    let tex = add_texture(
        &mut scene,
        "glow",
        GLOW_SIDE,
        glow_texels(),
        [Wrap::Mirror, Wrap::Clamp],
    );
    let mat = add_material(
        &mut scene,
        "glow",
        Material {
            rgba: [0.0, 0.0, 0.0, 1.0],
            emissive: Some(GLOW),
            emissive_map: Some(tex),
            texrepeat: [GLOW_REPEAT, GLOW_REPEAT],
            ..Material::default()
        },
    );
    scene.bodies[0].geoms.push(geom(
        "floor",
        plane(1.0),
        Pose::IDENTITY,
        [FLOOR, FLOOR, FLOOR, 1.0],
        None,
    ));
    scene.bodies[0].geoms.push(geom(
        "panel",
        plane(QUAD_HALF),
        Pose::new(
            Vec3::new(0.0, 0.0, QUAD_Z),
            Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
        ),
        [0.5, 0.5, 0.5, 1.0],
        Some(mat),
    ));
    scene
}

fn fold(i: i64, n: i64, wrap: Wrap) -> usize {
    let m = match wrap {
        Wrap::Repeat => i.rem_euclid(n),
        Wrap::Clamp => i.clamp(0, n - 1),
        Wrap::Mirror => {
            let k = i.rem_euclid(2 * n);
            if k < n {
                k
            } else {
                2 * n - 1 - k
            }
        }
    };
    m as usize
}

/// The panel's emitted radiance at texture coordinate `(s, t)`, bilinear with texel centres at
/// +0.5, written out here in `f64`.
fn glow_at(texels: &[u8], s: f64, t: f64) -> [f64; 3] {
    let n = i64::from(GLOW_SIDE);
    let x = s * n as f64 - 0.5;
    let y = t * n as f64 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let texel = |ix: i64, iy: i64| {
        let (c, r) = (fold(ix, n, Wrap::Mirror), fold(iy, n, Wrap::Clamp));
        let i = (r * n as usize + c) * 3;
        [0, 1, 2].map(|k| f64::from(texels[i + k]) / 255.0)
    };
    let (ix, iy) = (x0 as i64, y0 as i64);
    let (c00, c10, c01, c11) = (
        texel(ix, iy),
        texel(ix + 1, iy),
        texel(ix, iy + 1),
        texel(ix + 1, iy + 1),
    );
    [0, 1, 2].map(|k| {
        let top = c00[k] * (1.0 - fx) + c10[k] * fx;
        let bottom = c01[k] * (1.0 - fx) + c11[k] * fx;
        GLOW[k] * (top * (1.0 - fy) + bottom * fy)
    })
}

/// The floor's outgoing radiance at `(px, py, 0)`, `rho / pi * integral(Le cos cos / r^2 dA)`, by
/// the midpoint rule on a `m x m` grid over the panel. The panel is a plane turned half a turn
/// about X: its local `(lx, ly)` sits at world `(lx, -ly, QUAD_Z)`, and its UV is `MuJoCo`'s
/// `((lx / h + 1) / 2, 1 - (ly / h + 1) / 2)`, times the repeat.
fn glow_quadrature(px: f64, py: f64, m: usize) -> [f64; 3] {
    let texels = glow_texels();
    let step = 2.0 * QUAD_HALF / m as f64;
    let mut sum = [0.0f64; 3];
    for j in 0..m {
        for i in 0..m {
            let lx = -QUAD_HALF + (i as f64 + 0.5) * step;
            let ly = -QUAD_HALF + (j as f64 + 0.5) * step;
            let s = f64::midpoint(lx / QUAD_HALF, 1.0) * GLOW_REPEAT;
            let t = (1.0 - f64::midpoint(ly / QUAD_HALF, 1.0)) * GLOW_REPEAT;
            let le = glow_at(&texels, s, t);
            let (dx, dy) = (lx - px, -ly - py);
            let r2 = dx * dx + dy * dy + QUAD_Z * QUAD_Z;
            let g = QUAD_Z * QUAD_Z / (r2 * r2);
            for k in 0..3 {
                sum[k] += le[k] * g * step * step;
            }
        }
    }
    sum.map(|e| FLOOR / std::f64::consts::PI * e)
}

fn glow_cfg(side: u32, spp: u32, nee: bool) -> RenderConfig {
    // NEE at one bounce estimates what the BSDF-only tracer estimates at two
    // (`cpu::nee_direct`): floor, then the panel.
    let mut cfg = if nee {
        RenderConfig::pt_nee(atlas(side), spp, 1)
    } else {
        RenderConfig::pt(atlas(side), spp, 2)
    };
    cfg.sky = [0.0; 3];
    cfg.exposure = 4.0;
    cfg
}

/// Oracle 3 on the CPU reference.
#[test]
fn a_textured_emitter_is_estimated_without_bias() {
    let tri = TriScene::from_scene(&glow_scene()).unwrap();
    assert_eq!(tri.lights.len(), 2, "the panel's two triangles are lights");

    // The probe: one pixel, its ray straight down onto a floor point off the panel's centre
    // (so a mirrored or shifted texture lookup would not integrate to the same value).
    let (px, py) = (0.1, 0.05);
    let probe = down_camera(Vec3::new(px, py, 0.2), 1, 0.2);
    let reference = glow_quadrature(px, py, 1024);
    for (nee, spp, tol) in [(true, 16_384, 0.01), (false, 65_536, 0.02)] {
        let got = radiance(&cpu::path_trace(&tri, &probe, &glow_cfg(1, spp, nee), 0)).to_vec();
        let rel: Vec<f64> = (0..3)
            .map(|k| (f64::from(got[k]) - reference[k]).abs() / reference[k])
            .collect();
        println!(
            "probe, NEE {nee}, {spp} spp: {got:?} vs quadrature {reference:?}, relative {rel:?}"
        );
        assert!(rel.iter().all(|r| *r < tol), "NEE {nee}");
    }

    // The image: NEE on and off converge to the same one.
    let cam = down_camera(Vec3::new(0.0, 0.0, 0.2), 8, 0.9);
    let on = cpu::path_trace(&tri, &cam, &glow_cfg(8, 1024, true), 0);
    let off = cpu::path_trace(&tri, &cam, &glow_cfg(8, 4096, false), 0);
    let mean = |r: &[f32], k: usize| r.chunks(3).map(|p| f64::from(p[k])).sum::<f64>() / 64.0;
    for k in 0..3 {
        let (a, b) = (mean(radiance(&on), k), mean(radiance(&off), k));
        println!(
            "image mean, channel {k}: NEE {a:.5}, no NEE {b:.5}, relative {:.2e}",
            (a - b).abs() / b
        );
        assert!((a - b).abs() / b < 0.01, "channel {k}");
    }
}

// --- oracle 4: the glTF box and the MJCF box -------------------------------------------------------

fn box_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/gltf/textured_box")
}

fn gltf_box() -> SceneDesc {
    let bytes = std::fs::read(box_dir().join("textured_box.gltf")).expect("fixture");
    let import = es_assets::gltf::import_gltf(&bytes, Some(&box_dir())).expect("imports");
    assert!(import.warnings.is_empty(), "{:?}", import.warnings);
    import.scene
}

fn mjcf_box() -> SceneDesc {
    let xml = std::fs::read_to_string(box_dir().join("textured_box.xml")).expect("fixture");
    let mut scene = es_assets::parse_mjcf(&xml).expect("parses").scene;
    es_assets::mesh::load(&mut scene, &box_dir()).expect("loads");
    scene
}

/// Looking at the box's +X, -Y and +Z faces from above and to the side.
fn box_camera(side: u32) -> CameraView {
    let pos = Vec3::new(1.2, -1.0, 0.9);
    let forward = (Vec3::ZERO - pos).normalize();
    let right = forward.cross(Vec3::new(0.0, 0.0, 1.0)).normalize();
    let down = forward.cross(right);
    CameraView {
        pose: Pose::new(pos, from_columns(right, down, forward)),
        spec: ImageSpec::pinhole(side, side, 0.9),
    }
}

/// The rotation whose columns are `x`, `y`, `z` (orthonormal, right-handed), as a quaternion.
fn from_columns(x: Vec3, y: Vec3, z: Vec3) -> Quat {
    let m = [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]];
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quat::from_xyzw(
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s,
        )
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        Quat::from_xyzw(
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        )
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        Quat::from_xyzw(
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        )
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        Quat::from_xyzw(
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        )
    };
    q.normalize()
}

fn box_pt_cfg(side: u32, spp: u32, bounces: u32, nee: bool, restir: bool) -> RenderConfig {
    let mut cfg = RenderConfig::pt(atlas(side), spp, bounces);
    cfg.path = RenderPath::Pt {
        spp,
        bounces,
        nee,
        restir,
        svgf: false,
    };
    cfg.sky = [0.3, 0.35, 0.4];
    cfg.light_rgb = [1.0, 0.95, 0.9];
    cfg.exposure = 2.0;
    cfg
}

#[test]
fn a_gltf_material_renders_as_its_mjcf_declaration() {
    let (g, m) = (gltf_box(), mjcf_box());
    let (tg, tm) = (
        TriScene::from_scene(&g).unwrap(),
        TriScene::from_scene(&m).unwrap(),
    );
    assert_eq!(tg.tris.len(), 12);
    let mat = tg.materials.mats[0];
    assert!(
        mat.pbr
            && mat.rgb != 0
            && mat.metal_tex != 0
            && mat.normal_tex != 0
            && mat.emissive_tex != 0
    );
    assert_eq!(tg.tris, tm.tris, "the triangles differ");
    assert_eq!(*tg.materials, *tm.materials, "the material tables differ");
    let cam = box_camera(64);
    for (name, cfg) in [
        ("Rs Lambert", RenderConfig::rs(atlas(64))),
        ("Rs Full", RenderConfig::rs_full(atlas(64))),
        ("Pt NEE", box_pt_cfg(64, 4, 3, true, false)),
    ] {
        let render = |t: &TriScene| match cfg.path {
            RenderPath::Rs => cpu::rasterize(t, &cam, &cfg, 0),
            RenderPath::Pt { .. } => cpu::path_trace(t, &cam, &cfg, 0),
        };
        let (a, b) = (render(&tg), render(&tm));
        let lit = rgb8(&a).iter().filter(|x| **x > 0).count();
        println!(
            "{name}: glTF and MJCF, {lit} lit bytes, identical {}",
            rgb8(&a) == rgb8(&b)
        );
        assert!(lit > 1000, "{name}: the box is not in view");
        assert!(
            rgb8(&a) == rgb8(&b),
            "{name}: glTF and MJCF render differently"
        );
    }
}

// --- oracle 5: the GPU against the CPU --------------------------------------------------------------

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

/// One scene on every path, GPU against CPU, at `renderer.md` 15.4's tolerances.
fn compare(
    gpu: &Gpu,
    label: &str,
    tri: &TriScene,
    cam: &CameraView,
    side: u32,
    pt_base: &RenderConfig,
) {
    // Rs Lambert: `Rgb8` and segmentation bitwise; depth and normal to relative 1e-5.
    let cfg = RenderConfig::rs(atlas(side));
    let want = cpu::rasterize(tri, cam, &cfg, 0);
    let got = gpu_frame(gpu, tri, cam, cfg);
    for (channel, tile) in &got {
        let w = want.tile(*channel).unwrap();
        if let (Some(g), Some(w)) = (tile.as_f32(), w.as_f32()) {
            let rel = g
                .iter()
                .zip(w)
                .map(|(x, y)| (x - y).abs() / y.abs().max(1e-6))
                .fold(0.0f32, f32::max);
            println!(
                "{label} Rs Lambert {channel:?}: max ULP {}, max relative {rel:e}",
                max_ulp(g, w)
            );
            assert!(rel <= 1e-5, "{label} Rs Lambert {channel:?}");
        } else {
            let n = differing_bytes(&tile.to_bytes(), &w.to_bytes());
            println!("{label} Rs Lambert {channel:?}: {n} bytes differ");
            assert_eq!(n, 0, "{label} Rs Lambert {channel:?}");
        }
    }

    // Rs Full: renderer.md 9.3's rule, no more than 0.1 % of the pixels.
    let cfg = RenderConfig::rs_full(atlas(side));
    let want = cpu::rasterize(tri, cam, &cfg, 0);
    let got = gpu_frame(gpu, tri, cam, cfg);
    let (a, b) = (got[&Channel::Rgb8].as_u8().unwrap(), rgb8(&want));
    let pixels = a.chunks(3).zip(b.chunks(3)).filter(|(x, y)| x != y).count();
    println!(
        "{label} Rs Full: {} of {} bytes differ, {pixels} pixels",
        differing_bytes(a, b),
        a.len()
    );
    assert!(pixels * 1000 <= (side * side) as usize, "{label} Rs Full");

    // Pt 1 spp, NEE 4 spp, ReSTIR 1 spp: renderer.md 5.1's 1e-5 of the peak.
    for (name, spp, bounces, nee, restir) in [
        ("Pt 1 spp", 1, 3, false, false),
        ("Pt NEE 4 spp", 4, 3, true, false),
        ("Pt ReSTIR 1 spp", 1, 2, false, true),
    ] {
        let mut cfg = pt_base.clone();
        cfg.atlas = atlas(side);
        cfg.path = RenderPath::Pt {
            spp,
            bounces,
            nee,
            restir,
            svgf: false,
        };
        let want = cpu::path_trace(tri, cam, &cfg, 0);
        let got = gpu_frame(gpu, tri, cam, cfg);
        let (g, w) = (got[&Channel::PtRadiance].as_f32().unwrap(), radiance(&want));
        let (rg, rw) = (got[&Channel::Rgb8].as_u8().unwrap(), rgb8(&want));
        println!(
            "{label} {name}: PtRadiance max ULP {}, normalized {:e}; Rgb8 {} of {} bytes differ",
            max_ulp(g, w),
            max_normalized(g, w),
            differing_bytes(rg, rw),
            rg.len()
        );
        assert!(max_normalized(g, w) <= 1e-5, "{label} {name}");
        assert!(differing_bytes(rg, rw) * 1000 <= rg.len(), "{label} {name}");
    }
}

#[test]
fn gpu_matches_the_cpu_on_normal_and_emissive_maps() {
    let test = "gpu_matches_the_cpu_on_normal_and_emissive_maps";
    let Some(gpu) = open(test) else { return };
    let side = 64;
    let bump = TriScene::from_scene(&bump_plane(true, false)).unwrap();
    let mut pt = box_pt_cfg(side, 1, 1, false, false);
    pt.light_rgb = [0.0; 3];
    compare(&gpu, "bump", &bump, &bump_camera(), side, &pt);
    let glow = TriScene::from_scene(&glow_scene()).unwrap();
    let pt = glow_cfg(side, 1, false);
    compare(
        &gpu,
        "glow",
        &glow,
        &down_camera(Vec3::new(0.0, 0.0, 0.2), side, 0.9),
        side,
        &pt,
    );
    let tri = TriScene::from_scene(&mjcf_box()).unwrap();
    compare(
        &gpu,
        "box",
        &tri,
        &box_camera(side),
        side,
        &box_pt_cfg(side, 1, 1, false, false),
    );
}

// --- oracle 6: goldens --------------------------------------------------------------------------------

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/render")
}

const GOLDEN_SIDE: u32 = 64;

struct Golden {
    name: &'static str,
    kernel: &'static str,
}

const GOLDENS: [Golden; 3] = [
    Golden {
        name: "maps_rs_rgb8",
        kernel: "raster.v3 (Lambert, textured, normal and emissive maps)",
    },
    Golden {
        name: "maps_rs_full_rgb8",
        kernel: "raster.v3 (Shading::Full, ssaa 2, glTF metallic-roughness, normal and emissive maps)",
    },
    Golden {
        name: "maps_pt_nee_rgb8",
        kernel: "pt.v5 (4 spp, 3 bounces, NEE, sky 0.3 0.35 0.4, light 1 0.95 0.9, Reinhard, exposure 2, normal and emissive maps)",
    },
];

fn golden_frame(name: &str, tri: &TriScene) -> Frame {
    let cam = box_camera(GOLDEN_SIDE);
    match name {
        "maps_rs_rgb8" => cpu::rasterize(tri, &cam, &RenderConfig::rs(atlas(GOLDEN_SIDE)), 0),
        "maps_rs_full_rgb8" => {
            cpu::rasterize(tri, &cam, &RenderConfig::rs_full(atlas(GOLDEN_SIDE)), 0)
        }
        _ => cpu::path_trace(tri, &cam, &box_pt_cfg(GOLDEN_SIDE, 4, 3, true, false), 0),
    }
}

fn golden_cfg(name: &str) -> RenderConfig {
    match name {
        "maps_rs_rgb8" => RenderConfig::rs(atlas(GOLDEN_SIDE)),
        "maps_rs_full_rgb8" => RenderConfig::rs_full(atlas(GOLDEN_SIDE)),
        _ => box_pt_cfg(GOLDEN_SIDE, 4, 3, true, false),
    }
}

/// Writes the `maps_*` goldens from the CPU reference. Its own generator, so it cannot rewrite
/// an older golden:
///
///     ES_GENERATE_GOLDENS=1 cargo test -p es-render --test maps -- --ignored generate_maps_goldens
#[test]
#[ignore = "golden generator; run explicitly"]
fn generate_maps_goldens() {
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_maps_goldens: set ES_GENERATE_GOLDENS=1");
        return;
    }
    let tri = TriScene::from_scene(&mjcf_box()).expect("tessellates");
    for g in &GOLDENS {
        let tile = golden_frame(g.name, &tri)
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
            "scene": "tests/fixtures/gltf/textured_box/textured_box.xml (= textured_box.gltf), the camera of crates/es-render/tests/maps.rs `box_camera`",
            "pins": "OpenCV camera frame and top-left image origin (spec 3.1)",
            "oracle": "es_render::cpu, the pure-Rust mirror of the Slang kernels",
            "generator": "cargo test -p es-render --test maps -- --ignored generate_maps_goldens",
            "spec": "docs/design/renderer.md section 16",
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
fn maps_goldens_are_reproduced_by_the_cpu_and_the_gpu() {
    let tri = TriScene::from_scene(&mjcf_box()).expect("tessellates");
    let gpu = open("maps_goldens_are_reproduced_by_the_cpu_and_the_gpu");
    for g in &GOLDENS {
        let path = golden_dir().join(format!("{}.bin", g.name));
        let expected = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run generate_maps_goldens)", path.display()));
        let cpu = golden_frame(g.name, &tri)
            .tile(Channel::Rgb8)
            .unwrap()
            .to_bytes();
        assert!(cpu == expected, "{}: the CPU reference moved", g.name);
        if let Some(gpu) = &gpu {
            let got = gpu_frame(gpu, &tri, &box_camera(GOLDEN_SIDE), golden_cfg(g.name))
                [&Channel::Rgb8]
                .to_bytes();
            let n = differing_bytes(&got, &expected);
            println!(
                "{}: GPU vs golden, {n} of {} bytes differ",
                g.name,
                got.len()
            );
            let allowed = if g.name == "maps_rs_rgb8" {
                0
            } else {
                got.len() / 1000
            };
            assert!(n <= allowed, "{}: {n} bytes", g.name);
        }
    }
}
