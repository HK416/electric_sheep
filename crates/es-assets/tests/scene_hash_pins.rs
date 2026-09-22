//! Packet M10/W0b oracle 2: `scene_hash` is one number on every platform.
//!
//! The importers turn `euler=`, `axisangle=`, `zaxis=` and `rpy` into quaternions whose bits
//! enter `scene_hash` (spec 5.3), so the trigonometry there goes through
//! `es_math::approx::{sin_cos_f64, acos_f64}` (the pure-Rust `libm` port, spec 3.2) and not the
//! host's `libm`, which gave the SO-101 scene two digests for one file (M8 S-2, measured on
//! Windows against Linux). The hex below is typed in on purpose: the same constants are asserted
//! on the Windows tree and on the Linux oracle server.
//!
//! `go1_primitives.xml` is the control: it spells every orientation with `xyaxes`, which is
//! `sqrt` only (IEEE, correctly rounded), so its hash was pinned before the change and must
//! not move.

use std::fmt::Write as _;

use es_assets::urdf::{parse_urdf, PackageResolver};

fn fixture(dir: &str, name: &str) -> String {
    let path = format!(
        "{}/../../tests/fixtures/{dir}/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn mjcf_hash(name: &str) -> String {
    let import =
        es_assets::parse_mjcf(&fixture("mjcf", name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    hex(&import.scene.scene_hash())
}

fn urdf_hash(name: &str) -> String {
    let import = parse_urdf(&fixture("urdf", name), &PackageResolver::default())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    hex(&import.scene.scene_hash())
}

/// The control: `xyaxes` only, no transcendental on its path, pinned before the change.
#[test]
fn quadruped_scene_hash_is_unmoved() {
    assert_eq!(
        mjcf_hash("go1_primitives.xml"),
        "3c9348ea4be1fd36223cda47bac7b447c6dfccc831aaa7d26550e2f189c49a99"
    );
}

/// The SO-101 scene (two `euler=` geoms and `fromto` capsules, which go through `acos`),
/// `orientations.xml` (every spelling), `arm2.xml` (`fromto` only — the `acos` path alone
/// moved it) and `arm2.urdf` (`rpy`): the digests the regenerated documents carry, equal on
/// Windows and Linux.
#[test]
fn scene_hashes_are_platform_stable() {
    assert_eq!(
        mjcf_hash("so101_pick_place.xml"),
        "882e7d0b67d32fe8df9260d753d5e985a2bd641de557882bc4e654b5a03b5ba4"
    );
    assert_eq!(
        mjcf_hash("orientations.xml"),
        "958ceaaeef82517e6b0c23e7ac5cb4b41a3a16cfe6201602afe31bfa88458f7f"
    );
    assert_eq!(
        mjcf_hash("arm2.xml"),
        "4602d675ad1990d2cec8a3f3351507c02a062b6ec3dd1f1c6846d35856d5da57"
    );
    assert_eq!(
        urdf_hash("arm2.urdf"),
        "4708aba6dc5d7cf00c472ec4aaaf2018f1c2958610b010b47a33f783350c06da"
    );
}

/// Prints every pinned digest, for the old -> new table in the packet note.
#[test]
fn print_scene_hashes() {
    for name in [
        "so101_pick_place.xml",
        "go1_primitives.xml",
        "orientations.xml",
        "pendulum.xml",
        "actuated.xml",
        "arm2.xml",
    ] {
        println!("mjcf {name} {}", mjcf_hash(name));
    }
    println!("urdf arm2.urdf {}", urdf_hash("arm2.urdf"));
}
