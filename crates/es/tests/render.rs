//! `es render` (packet M16/H8): one frame of a free camera through `es-render`'s GPU path, so
//! the editor's viewport can show the textured and the path-traced scene without a Vulkan
//! device of its own (`docs/design/editor-redesign.md` section 5, S4).
//!
//! The oracle is the renderer's own CPU reference (`docs/design/renderer.md` sections 5.1,
//! 15.4): `Rs` is bit for bit the reference's frame, `Pt` is the reference's accumulated frame
//! within one 8-bit level (section 5.1's 1e-5 of the peak, tone-mapped and rounded). A machine
//! without a Vulkan device skips the render half with the reason printed.
#![cfg(feature = "render")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use es_env::render::{config, look_at};
use es_env::traj::Trajectory;
use es_render::cpu::{path_trace_accum, rasterize, History};
use es_render::{Channel, RenderPath, Shading, Temporal, TriScene};

fn root() -> PathBuf {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).to_path_buf()
}

const TEXTURED: &str = "tests/fixtures/mjcf/textured/textured.xml";
const EYE: [f64; 3] = [0.32, -0.3, 0.3];
const LOOK_AT: [f64; 3] = [0.0, 0.0, 0.04];
const FOV_DEG: f64 = 50.0;
const W: u32 = 64;
const H: u32 = 48;

fn es(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_es"))
        .arg("render")
        .args(args)
        .current_dir(root())
        .output()
        .expect("run es render")
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn csv(v: [f64; 3]) -> String {
    format!("{},{},{}", v[0], v[1], v[2])
}

/// One binary PPM (`P6`) from the front of `bytes`: `(samples, w, h, rgb, rest)`.
fn ppm(bytes: &[u8]) -> (u32, u32, u32, Vec<u8>, &[u8]) {
    let mut fields = Vec::new();
    let mut samples = 0;
    let mut at = 0;
    while fields.len() < 4 {
        let end = at
            + bytes[at..]
                .iter()
                .position(|b| *b == b'\n')
                .expect("header");
        let line = std::str::from_utf8(&bytes[at..end]).expect("ascii");
        if let Some(n) = line.strip_prefix("# samples ") {
            samples = n.parse().expect("samples");
        } else {
            fields.extend(line.split_whitespace().map(str::to_owned));
        }
        at = end + 1;
    }
    assert_eq!(fields[0], "P6");
    let (w, h): (u32, u32) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    let n = (w * h * 3) as usize;
    (samples, w, h, bytes[at..at + n].to_vec(), &bytes[at + n..])
}

/// The reference frame: the scene loaded the way `es` loads it, posed at `tick` of `traj`.
fn reference(
    scene: &str,
    traj: Option<(&str, usize)>,
    path: RenderPath,
    look: Shading,
    frames: u32,
) -> Vec<u8> {
    let scene_desc =
        es_tools::backend::load_scene(root().join(scene).to_str().unwrap()).expect("scene");
    let poses = traj.map_or_else(BTreeMap::new, |(t, tick)| {
        Trajectory::read(&root().join(t)).expect("traj").poses(tick)
    });
    let tris = TriScene::from_scene_with_poses(&scene_desc, &poses).expect("tessellate");
    let view = look_at(EYE, LOOK_AT, FOV_DEG.to_radians(), W, H).expect("view");
    let mut cfg = config(W, H, Channel::Rgb8, path);
    cfg.shading = look;
    let mut history = History::default();
    let frame = match path {
        RenderPath::Rs => rasterize(&tris, &view, &cfg, 0),
        RenderPath::Pt { .. } => {
            cfg.exposure = 8.0;
            cfg.temporal = Some(Temporal {
                max_history: frames,
            });
            let mut last = None;
            for _ in 0..frames {
                last = Some(path_trace_accum(&tris, &view, &cfg, 0, &mut history));
            }
            last.expect("one frame")
        }
    };
    frame.tile(Channel::Rgb8).expect("rgb8").to_bytes()
}

fn camera_args() -> Vec<String> {
    [
        "--eye",
        &csv(EYE),
        "--look-at",
        &csv(LOOK_AT),
        "--fov",
        &FOV_DEG.to_string(),
        "--width",
        &W.to_string(),
        "--height",
        &H.to_string(),
    ]
    .map(str::to_owned)
    .to_vec()
}

/// Renders through `es render` into a file; `None` (and a printed SKIP) without a device.
fn render(test: &str, extra: &[&str]) -> Option<(u32, Vec<u8>)> {
    let out = std::env::temp_dir().join(format!("es-render-{test}-{}.ppm", std::process::id()));
    let cam = camera_args();
    let mut args: Vec<&str> = cam.iter().map(String::as_str).collect();
    args.extend(extra);
    args.extend(["--out", out.to_str().unwrap()]);
    let got = es(&args);
    if got.status.code() != Some(0) {
        assert_eq!(got.status.code(), Some(1), "{test}: {}", text(&got));
        println!("SKIP {test}: {}", text(&got).trim());
        return None;
    }
    let bytes = std::fs::read(&out).expect("the frame");
    std::fs::remove_file(&out).ok();
    let (samples, w, h, rgb, rest) = ppm(&bytes);
    assert_eq!((w, h), (W, H));
    assert!(rest.is_empty(), "one frame in a file");
    Some((samples, rgb))
}

/// `--path rs` and `--path full` are the CPU reference's `Rgb8`, byte for byte, on the
/// textured fixture (section 15.4: 0 bytes differ on `Rs` `Lambert` and `Full`).
#[test]
fn rs_is_the_cpu_reference_bit_for_bit() {
    for (flag, look) in [("rs", Shading::Lambert), ("full", Shading::FULL)] {
        let test = format!("rs_is_the_cpu_reference_{flag}");
        let Some((samples, rgb)) = render(&test, &["--scene", TEXTURED, "--path", flag]) else {
            return;
        };
        assert_eq!(samples, 1);
        let want = reference(TEXTURED, None, RenderPath::Rs, look, 1);
        let differ = rgb.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            differ,
            0,
            "--path {flag}: {differ} of {} bytes differ",
            want.len()
        );
        println!("RAN {test}: 0 of {} bytes differ", want.len());
    }
}

/// A trajectory poses the scene at `--tick`: the arm demo's fixture replay at tick 30 is the
/// reference posed from the same `.estraj`, and not the scene's own pose.
#[test]
fn a_trajectory_tick_poses_the_scene() {
    let (scene, traj) = (
        "tests/fixtures/mjcf/so101_pick_place.xml",
        "tests/fixtures/visible-learning/run/traj/nominal-00.estraj",
    );
    let Some((_, rgb)) = render(
        "a_trajectory_tick_poses_the_scene",
        &["--scene", scene, "--traj", traj, "--tick", "30"],
    ) else {
        return;
    };
    let want = reference(scene, Some((traj, 30)), RenderPath::Rs, Shading::Lambert, 1);
    assert_eq!(rgb, want, "tick 30");
    let still = reference(scene, None, RenderPath::Rs, Shading::Lambert, 1);
    assert_ne!(rgb, still, "the tick moved nothing");
    let past = es(&[
        "--scene",
        scene,
        "--traj",
        traj,
        "--tick",
        "9999",
        "--eye",
        "1,1,1",
        "--look-at",
        "0,0,0",
        "--out",
        "-",
    ]);
    assert_eq!(past.status.code(), Some(2), "{}", text(&past));
    assert!(text(&past).contains("--tick 9999"), "{}", text(&past));
}

/// `--path pt --accumulate K` is K frames of the reference's temporal accumulation (renderer
/// note section 11), within one 8-bit level; `--out -` streams every one of the K frames as a
/// PPM whose sample count grows, and the last is the file's frame.
#[test]
fn pt_accumulates_and_streams_within_tolerance() {
    let test = "pt_accumulates_and_streams_within_tolerance";
    let pt = [
        "--scene",
        TEXTURED,
        "--path",
        "pt",
        "--spp",
        "2",
        "--bounces",
        "3",
    ];
    let mut file_args = pt.to_vec();
    file_args.extend(["--accumulate", "3", "--exposure", "8"]);
    let Some((samples, rgb)) = render(test, &file_args) else {
        return;
    };
    assert_eq!(samples, 6, "3 frames of 2 samples");
    let path = RenderPath::Pt {
        spp: 2,
        bounces: 3,
        nee: true,
        restir: false,
        svgf: false,
    };
    let want = reference(TEXTURED, None, path, Shading::Lambert, 3);
    let worst = rgb.iter().zip(&want).map(|(a, b)| a.abs_diff(*b)).max();
    let differ = rgb.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert!(worst <= Some(1), "worst {worst:?}, {differ} bytes differ");

    let cam = camera_args();
    let mut args: Vec<&str> = cam.iter().map(String::as_str).collect();
    args.extend(file_args.iter().copied());
    args.extend(["--out", "-"]);
    let streamed = es(&args);
    assert_eq!(streamed.status.code(), Some(0), "{}", text(&streamed));
    let mut rest = streamed.stdout.as_slice();
    let mut counts = Vec::new();
    let mut last = Vec::new();
    while !rest.is_empty() {
        let (n, _, _, frame, tail) = ppm(rest);
        counts.push(n);
        last = frame;
        rest = tail;
    }
    assert_eq!(counts, [2, 4, 6]);
    assert_eq!(last, rgb, "the stream's last frame is the file's");
    println!(
        "RAN {test}: worst {worst:?} level(s), {differ} of {} bytes differ",
        want.len()
    );
}

/// Usage errors are exit 2 and say which flag, before any device is opened.
#[test]
fn usage_errors_name_the_flag() {
    for (args, says) in [
        (
            vec!["--eye", "1,1,1", "--look-at", "0,0,0", "--out", "-"],
            "--scene",
        ),
        (
            vec!["--scene", TEXTURED, "--look-at", "0,0,0", "--out", "-"],
            "--eye",
        ),
        (
            vec![
                "--scene",
                TEXTURED,
                "--eye",
                "1,1",
                "--look-at",
                "0,0,0",
                "--out",
                "-",
            ],
            "--eye",
        ),
        (
            vec![
                "--scene",
                TEXTURED,
                "--eye",
                "1,1,1",
                "--look-at",
                "0,0,0",
                "--path",
                "x",
                "--out",
                "-",
            ],
            "--path",
        ),
        (
            vec![
                "--scene",
                TEXTURED,
                "--eye",
                "1,1,1",
                "--look-at",
                "0,0,0",
                "--accumulate",
                "4",
                "--out",
                "-",
            ],
            "--accumulate",
        ),
        (
            vec![
                "--scene",
                TEXTURED,
                "--eye",
                "1,1,1",
                "--look-at",
                "0,0,0",
                "--tick",
                "3",
                "--out",
                "-",
            ],
            "--tick",
        ),
        (
            vec!["--scene", TEXTURED, "--eye", "1,1,1", "--look-at", "0,0,0"],
            "--out",
        ),
    ] {
        let got = es(&args);
        assert_eq!(got.status.code(), Some(2), "{args:?}: {}", text(&got));
        // The first line: the help text below it names every flag.
        let first = text(&got).lines().next().unwrap_or_default().to_owned();
        assert!(first.contains(says), "{args:?}: {first}");
    }
}
