//! Oracle for `docs/packets/M3/W3-splat-importer.md` (spec 16, spec 28.5 W3).
//!
//! Fixtures live in `tests/fixtures/splat/` at the workspace root (inputs, not golden
//! outputs), the same convention as `tests/fixtures/gltf/`. They are generated, not captured:
//! there is no real 3DGS scan in this repository, and the API note they follow is
//! `unverified` throughout.
//!
//! Exact float equality is the assertion in several tests here, not an oversight: a byte-exact
//! round trip, a welded weight of exactly 1.0 and a translation that moves a Gaussian by
//! exactly the same vector are the properties being pinned.
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;

use es_assets::scene::{scene_id, Body, Geom, SceneDesc, Shape};
use es_core::StableId;
use es_math::{Pose, Quat, Vec3};
use es_splat::{import_ply, skin, Binding, ColorAffine, Similarity, SplatError, SplatScene};

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/../../tests/fixtures/splat/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

// ---------------------------------------------------------------- PLY import / export

#[test]
fn the_binary_fixture_round_trips_byte_for_byte() {
    let bytes = fixture("cube20_binary.ply");
    let scene = import_ply(&bytes).expect("import");
    assert_eq!(scene.len(), 20);
    assert_eq!(scene.sh_degree, 3);
    assert_eq!(scene.sh_rest.len(), 45 * 20);
    assert_eq!(scene.write_ply(), bytes);
}

#[test]
fn ascii_and_binary_decode_to_the_same_gaussians_and_the_same_hash() {
    let a = import_ply(&fixture("cube20_ascii.ply")).expect("ascii");
    let b = import_ply(&fixture("cube20_binary.ply")).expect("binary");
    assert_eq!(a.positions, b.positions);
    assert_eq!(a.scales, b.scales);
    assert_eq!(a.rotations, b.rotations);
    assert_eq!(a.opacities, b.opacities);
    assert_eq!(a.sh_dc, b.sh_dc);
    assert_eq!(a.sh_rest, b.sh_rest);
    // Spec 5.3: the hash is over decoded content, so the container cannot change it.
    assert_eq!(a.asset_hash(), b.asset_hash());
    assert_eq!(a.asset.hash, a.asset_hash());
}

#[test]
fn the_axis_fix_is_the_documented_swizzle_and_quaternions_are_canonical() {
    let scene = import_ply(&fixture("cube20_binary.ply")).expect("import");
    // Fixture vertex 0 is at file (x, y, z) = (-0.25, 0.5, -0.125); Z-up is (x, z, -y).
    assert_eq!(&scene.positions[..3], &[-0.25f32, -0.125, -0.5]);
    for q in scene.rotations.chunks_exact(4) {
        assert!(q[3] >= 0.0, "w >= 0 (spec 3.1)");
        let n = f64::from(q[0]).mul_add(
            f64::from(q[0]),
            f64::from(q[1]).mul_add(
                f64::from(q[1]),
                f64::from(q[2]).mul_add(f64::from(q[2]), f64::from(q[3]) * f64::from(q[3])),
            ),
        );
        assert!((n - 1.0).abs() < 1e-6, "unit norm, got {n}");
    }
}

#[test]
fn a_non_fixed_point_activation_round_trips_to_a_relative_1e_6() {
    // The complement of the byte-exact case: `exp`/`ln` and `sigmoid`/`logit` are not exact
    // inverses in f32, so the guarantee outside the fixture's chosen values is accuracy,
    // not bits. See docs/design/splat-real2sim.md section 1.1.
    let mut scene = import_ply(&fixture("cube20_binary.ply")).expect("import");
    for (i, s) in scene.scales.iter_mut().enumerate() {
        *s = 0.001 + (i % 37) as f32 * 0.0137;
    }
    for (i, o) in scene.opacities.iter_mut().enumerate() {
        *o = 0.01 + (i % 41) as f32 * 0.0231;
    }
    let again = import_ply(&scene.write_ply()).expect("reimport");
    for (a, b) in scene.scales.iter().zip(&again.scales) {
        assert!((a - b).abs() <= a.abs() * 1e-6, "scale {a} -> {b}");
    }
    for (a, b) in scene.opacities.iter().zip(&again.opacities) {
        assert!((a - b).abs() <= a.abs() * 1e-6, "alpha {a} -> {b}");
    }
}

#[test]
fn an_unknown_property_is_a_warning_not_a_rejection() {
    let text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n\
                property float z\nproperty float f_dc_0\nproperty float f_dc_1\n\
                property float f_dc_2\nproperty float opacity\nproperty float scale_0\n\
                property float scale_1\nproperty float scale_2\nproperty float rot_0\n\
                property float rot_1\nproperty float rot_2\nproperty float rot_3\n\
                property float confidence\nend_header\n\
                1 2 3 0 0 0 0 -1 -1 -1 1 0 0 0 0.5\n";
    let scene = import_ply(text.as_bytes()).expect("import");
    assert_eq!(scene.len(), 1);
    assert_eq!(scene.sh_degree, 0);
    assert!(scene.sh_rest.is_empty());
    assert!(
        scene.warnings.iter().any(|w| w.contains("confidence")),
        "{:?}",
        scene.warnings
    );
}

/// Every malformed input is a typed error naming what is wrong. None of these panics.
#[test]
fn malformed_input_is_a_typed_error() {
    let head = "ply\nformat ascii 1.0\nelement vertex 1\n";
    let cases: Vec<(&str, String)> = vec![
        ("not a ply", "obj\n".to_owned()),
        ("no end_header", format!("{head}property float x\n")),
        (
            "unknown format",
            "ply\nformat binary_big_endian 1.0\nend_header\n".to_owned(),
        ),
        (
            "missing x",
            format!("{head}property float y\nend_header\n0\n"),
        ),
        (
            "bad sh count",
            format!("{head}property float f_rest_0\nend_header\n0\n"),
        ),
        (
            "list property",
            format!("{head}property list uchar int vertex_indices\nend_header\n"),
        ),
        (
            "unsupported type",
            format!("{head}property decimal x\nend_header\n"),
        ),
        ("garbage statement", format!("{head}wibble 3\n")),
    ];
    for (what, text) in cases {
        let err = import_ply(text.as_bytes()).expect_err(what);
        assert!(!format!("{err}").is_empty(), "{what}: empty message");
    }

    // The specific variants the packet names, so a refactor cannot quietly merge them.
    assert!(matches!(
        import_ply(b"obj\n").unwrap_err(),
        SplatError::NotPly(_)
    ));
    assert!(matches!(
        import_ply(format!("{head}property float f_rest_0\nend_header\n0\n").as_bytes())
            .unwrap_err(),
        SplatError::BadShCount { count: 1 }
    ));
    // A well-formed header over a short payload: the real capture, cut mid-record.
    let mut short = fixture("cube20_binary.ply");
    short.truncate(short.len() - 7);
    assert!(matches!(
        import_ply(&short).unwrap_err(),
        SplatError::Truncated { count: 20, .. }
    ));
}

/// P-M3-R3's oracle: a hostile ascii header can declare far more vertices than the file has
/// room for. Before the fix, the six `SplatScene` array reserves were sized off the declared
/// `count` alone, so this ~1 MB file (500,000 one-token lines against a 59-property header)
/// asked for roughly 118 MB before the very first line's field count was even checked.
#[test]
fn a_hostile_ascii_header_does_not_amplify_the_reserve() {
    const N: usize = 500_000;
    let mut names = vec!["x".to_owned(), "y".to_owned(), "z".to_owned()];
    names.extend((0..3).map(|i| format!("f_dc_{i}")));
    names.extend((0..45).map(|i| format!("f_rest_{i}")));
    names.push("opacity".to_owned());
    names.extend((0..3).map(|i| format!("scale_{i}")));
    names.extend((0..4).map(|i| format!("rot_{i}")));

    let mut text = format!("ply\nformat ascii 1.0\nelement vertex {N}\n");
    for name in &names {
        use std::fmt::Write as _;
        writeln!(text, "property float {name}").expect("write to String cannot fail");
    }
    text.push_str("end_header\n");
    for _ in 0..N {
        text.push_str("0\n");
    }
    assert!(
        (900_000..1_100_000).contains(&text.len()),
        "fixture should be about 1 MB, got {}",
        text.len()
    );

    let before = es_core::alloc_count::allocation_count();
    let err = import_ply(text.as_bytes()).unwrap_err();
    let allocations = es_core::alloc_count::allocation_count() - before;

    assert!(
        matches!(
            err,
            SplatError::BadFieldCount {
                vertex: 0,
                found: 1,
                expected: 59,
            }
        ),
        "{err}"
    );
    // `allocation_count` counts calls, not bytes, so it cannot pin the "under 8 MB" bound by
    // itself — that is pinned directly against the reserve formula in
    // `ply::reserve_tests::ascii_reserve_is_bounded_by_file_size_not_declared_count`. What it
    // catches here is the other half of the regression: an unbounded reserve would still be
    // one `Vec::with_capacity` call each (cheap in count, catastrophic in bytes), but a
    // reserve that silently fell back to *no* upfront reserve would instead show up as an
    // allocation per `push`/`extend_from_slice` scaling with the header's declared count —
    // header parsing plus the six array reserves is on the order of a hundred calls, not
    // hundreds of thousands.
    assert!(
        es_core::alloc_count::counting_enabled(),
        "the alloc-count feature must be enabled for this assertion to mean anything"
    );
    assert!(
        allocations < 1_000,
        "unexpectedly many allocations: {allocations}"
    );
}

#[test]
fn a_non_finite_value_is_rejected_rather_than_poisoning_the_hash() {
    let text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n\
                property float z\nproperty float f_dc_0\nproperty float f_dc_1\n\
                property float f_dc_2\nproperty float opacity\nproperty float scale_0\n\
                property float scale_1\nproperty float scale_2\nproperty float rot_0\n\
                property float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n\
                nan 2 3 0 0 0 0 -1 -1 -1 1 0 0 0\n";
    assert!(matches!(
        import_ply(text.as_bytes()).unwrap_err(),
        SplatError::NotFinite { vertex: 0, .. }
    ));
}

// ---------------------------------------------------------------- alignment

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in `[-1, 1)`.
fn signed_unit(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / 9_007_199_254_740_992.0f64 * 2.0 - 1.0
}

#[test]
fn similarity_fit_recovers_a_known_transform() {
    for seed in 0..64u64 {
        let mut st = 0xA5A5_0000_0000_0000 ^ seed;
        let axis = Vec3::new(
            signed_unit(&mut st),
            signed_unit(&mut st),
            signed_unit(&mut st),
        )
        .normalize();
        let axis = if axis.norm() > 0.5 {
            axis
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };
        let angle = signed_unit(&mut st) * 3.0;
        let (s, c) = ((angle / 2.0).sin(), (angle / 2.0).cos());
        let rot = Quat::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c).normalize();
        let scale = 0.2 + (signed_unit(&mut st) + 1.0) * 1.5;
        let trans = Vec3::new(
            signed_unit(&mut st) * 4.0,
            signed_unit(&mut st) * 4.0,
            signed_unit(&mut st) * 4.0,
        );
        let truth = Similarity { scale, rot, trans };

        let src: Vec<Vec3> = (0..10)
            .map(|_| {
                Vec3::new(
                    signed_unit(&mut st) * 2.0,
                    signed_unit(&mut st) * 2.0,
                    signed_unit(&mut st) * 2.0,
                )
            })
            .collect();
        let dst: Vec<Vec3> = src.iter().map(|p| truth.apply(*p)).collect();

        let fit = Similarity::fit(&src, &dst).expect("fit");
        assert!(
            (fit.scale - scale).abs() < 1e-6 * scale,
            "seed {seed}: scale {} vs {scale}",
            fit.scale
        );
        for (p, q) in src.iter().zip(&dst) {
            let got = fit.apply(*p);
            assert!((got - *q).norm() < 1e-6, "seed {seed}: {got:?} vs {q:?}");
        }
    }
}

#[test]
fn similarity_fit_refuses_degenerate_input() {
    let p = vec![Vec3::new(1.0, 0.0, 0.0); 4];
    assert!(matches!(
        Similarity::fit(&p, &p).unwrap_err(),
        SplatError::DegenerateFit
    ));
    assert!(matches!(
        Similarity::fit(&p[..2], &p[..2]).unwrap_err(),
        SplatError::TooFewPoints(2)
    ));
    assert!(matches!(
        Similarity::fit(&p[..3], &p[..4]).unwrap_err(),
        SplatError::PointCountMismatch { src: 3, dst: 4 }
    ));
}

/// P-M3-R6's oracle: four points on a line give a rotation that is fixed about the line but
/// arbitrary around it — an error, not the silent identity-adjacent rotation the pre-fix code
/// returned (the doc at `Similarity::fit` already promised this; the check was missing).
#[test]
fn similarity_fit_refuses_collinear_input() {
    let p = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    assert!(matches!(
        Similarity::fit(&p, &p).unwrap_err(),
        SplatError::DegenerateFit
    ));

    // Collinear but not axis-aligned, and `dst` a different (still collinear) line: the
    // check must not be a special case of "src equals dst" or "src is along an axis".
    let src = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(2.0, 4.0, 6.0),
        Vec3::new(-1.0, -2.0, -3.0),
    ];
    let dst = vec![
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(5.0, 1.0, 0.0),
        Vec3::new(5.0, 2.0, 0.0),
        Vec3::new(5.0, -1.0, 0.0),
    ];
    assert!(matches!(
        Similarity::fit(&src, &dst).unwrap_err(),
        SplatError::DegenerateFit
    ));
}

#[test]
fn applying_a_similarity_moves_the_capture_and_its_hash() {
    let mut scene = import_ply(&fixture("cube20_binary.ply")).expect("import");
    let before = scene.asset_hash();
    let first = scene.scales[0];
    let xf = Similarity {
        scale: 2.0,
        rot: Quat::IDENTITY,
        trans: Vec3::new(1.0, 0.0, 0.0),
    };
    xf.apply_scene(&mut scene);
    assert_eq!(scene.scales[0], first * 2.0);
    assert_ne!(scene.asset_hash(), before);
    assert_eq!(scene.asset.hash, scene.asset_hash());
}

#[test]
fn color_affine_recovers_a_known_gain_and_bias() {
    let gain = [1.25f64, 0.8, 1.05];
    let bias = [-0.05f64, 0.12, 0.02];
    let mut st = 0x0C01_0400_0000_0001u64;
    let src: Vec<[f32; 3]> = (0..24)
        .map(|_| {
            [
                (signed_unit(&mut st) * 0.5 + 0.5) as f32,
                (signed_unit(&mut st) * 0.5 + 0.5) as f32,
                (signed_unit(&mut st) * 0.5 + 0.5) as f32,
            ]
        })
        .collect();
    let dst: Vec<[f32; 3]> = src
        .iter()
        .map(|s| {
            let mut out = [0.0f32; 3];
            for c in 0..3 {
                out[c] = gain[c].mul_add(f64::from(s[c]), bias[c]) as f32;
            }
            out
        })
        .collect();

    let fit = ColorAffine::fit(&src, &dst).expect("fit");
    for c in 0..3 {
        assert!((fit.gain[c] - gain[c]).abs() < 1e-5, "gain {c}");
        assert!((fit.bias[c] - bias[c]).abs() < 1e-5, "bias {c}");
    }

    // A constant source channel degrades to a pure offset instead of dividing by zero.
    let flat = vec![[0.5f32, 0.5, 0.5]; 4];
    let shifted = vec![[0.7f32, 0.7, 0.7]; 4];
    let offset = ColorAffine::fit(&flat, &shifted).expect("fit");
    assert!((offset.gain[0] - 1.0).abs() < 1e-12);
    assert!((offset.bias[0] - 0.2).abs() < 1e-6);
}

#[test]
fn applying_a_colour_map_survives_the_dc_round_trip() {
    let mut scene = import_ply(&fixture("cube20_binary.ply")).expect("import");
    let before: Vec<[f32; 3]> = (0..scene.len()).map(|i| scene.rgb(i)).collect();
    let map = ColorAffine {
        gain: [1.5, 1.0, 0.5],
        bias: [0.1, 0.0, -0.1],
    };
    scene.apply_color(&map);
    for (i, b) in before.iter().enumerate() {
        let want = map.apply(*b);
        let got = scene.rgb(i);
        for c in 0..3 {
            assert!(
                (want[c] - got[c]).abs() <= want[c].abs() * 1e-5 + 1e-5,
                "gaussian {i} channel {c}: {} vs {}",
                want[c],
                got[c]
            );
        }
    }
}

// ---------------------------------------------------------------- binding and skinning

fn sphere_body(name: &str, at: Vec3, radius: f64) -> Body {
    let id = scene_id("body", name);
    Body {
        id,
        name: name.to_owned(),
        parent: None,
        pose: Pose::new(at, Quat::IDENTITY),
        inertial: None,
        geoms: vec![Geom {
            id: scene_id("geom", &format!("{name}/ball")),
            name: "ball".to_owned(),
            shape: Shape::Sphere { radius },
            pose: Pose::IDENTITY,
            friction: [1.0, 0.005, 0.0001],
            contype: 1,
            conaffinity: 1,
            condim: 3,
            density: 1000.0,
            mass: None,
            margin: 0.0,
            gap: 0.0,
            solref: [0.02, 1.0],
            solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
            material: None,
            rgba: [0.5, 0.5, 0.5, 1.0],
            visual_only: false,
        }],
        sites: Vec::new(),
    }
}

/// Two bodies 2 m apart, and one Gaussian sitting on each one's surface.
fn two_body_setup() -> (SplatScene, SceneDesc, StableId, StableId) {
    let left = sphere_body("left", Vec3::ZERO, 0.25);
    let right = sphere_body("right", Vec3::new(2.0, 0.0, 0.0), 0.25);
    let (lid, rid) = (left.id, right.id);
    let desc = SceneDesc {
        name: "two".to_owned(),
        bodies: vec![left, right],
        ..SceneDesc::default()
    };

    let mut scene = import_ply(&fixture("cube20_binary.ply")).expect("import");
    scene.positions.truncate(6);
    scene.scales.truncate(6);
    scene.rotations.truncate(8);
    scene.opacities.truncate(2);
    scene.sh_dc.truncate(6);
    scene.sh_rest.truncate(90);
    scene.positions = vec![0.25, 0.0, 0.0, 2.25, 0.0, 0.0];
    scene.recompute_bounds();
    (scene, desc, lid, rid)
}

#[test]
fn a_gaussian_on_a_surface_is_welded_to_that_body() {
    let (scene, desc, ..) = two_body_setup();
    let binding = Binding::bind(&scene, &desc, 2);
    assert_eq!(binding.weights[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(binding.indices[0][0], 0, "nearest body is `left`");
    assert_eq!(binding.weights[1], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(binding.indices[1][0], 1, "nearest body is `right`");
}

#[test]
fn skinning_follows_a_body_translation_exactly() {
    let (scene, desc, lid, rid) = two_body_setup();
    let binding = Binding::bind(&scene, &desc, 2);

    let mut poses = BTreeMap::new();
    poses.insert(lid, Pose::new(Vec3::ZERO, Quat::IDENTITY));
    poses.insert(rid, Pose::new(Vec3::new(2.0, 0.0, 0.5), Quat::IDENTITY));

    let skinned = skin(&binding, &scene, &poses);
    assert_eq!(&skinned.positions[..3], &[0.25f32, 0.0, 0.0]);
    assert_eq!(&skinned.positions[3..], &[2.25f32, 0.0, 0.5]);
    // Orientations are untouched by a pure translation.
    assert_eq!(skinned.rotations, scene.rotations);
}

#[test]
fn a_gaussian_between_two_bodies_blends_and_a_missing_pose_holds_still() {
    let (mut scene, desc, lid, _rid) = two_body_setup();
    scene.positions = vec![0.25, 0.0, 0.0, 0.8, 0.0, 0.0];
    let binding = Binding::bind(&scene, &desc, 2);
    let w = binding.weights[1];
    assert!(w[0] > 0.0 && w[1] > 0.0, "blended: {w:?}");
    assert!(
        (f64::from(w[0] + w[1]) - 1.0).abs() < 1e-6,
        "normalised: {w:?}"
    );
    assert!(w[0] > w[1], "closer to `left`: {w:?}");

    // Only `left` is posed; `right` keeps its rest pose, so nothing collapses to the origin.
    let mut poses = BTreeMap::new();
    poses.insert(lid, Pose::new(Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY));
    let skinned = skin(&binding, &scene, &poses);
    assert!(skinned.positions[2] > 0.0 && skinned.positions[2] <= 1.0);
    assert_eq!(&skinned.positions[..2], &[0.25f32, 0.0]);
}

#[test]
fn binding_a_scene_with_no_bodies_leaves_every_gaussian_at_rest() {
    let (scene, ..) = two_body_setup();
    let empty = SceneDesc::default();
    let binding = Binding::bind(&scene, &empty, 4);
    let skinned = skin(&binding, &scene, &BTreeMap::new());
    assert_eq!(skinned.positions, scene.positions);
    assert_eq!(skinned.rotations, scene.rotations);
}
