//! Integration tests for `es video mosaic` (M5 V4, design note
//! `docs/design/visible-learning.md` sections 7.2, 8, 9; spec 1.4, 3.4, 25.1).
//!
//! `es` has no `[lib]` target, so every test here drives the compiled binary as a subprocess,
//! the same way `tests/cli.rs` does.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_es"))
}

fn scratch_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("es-video-test-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn workspace_root() -> PathBuf {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).to_path_buf()
}

/// The committed fixture (context: `tests/fixtures/visible-learning/**`), generated once by
/// `generate_fixture_and_golden` below.
fn fixture_frames_dir() -> PathBuf {
    workspace_root().join("tests/fixtures/visible-learning/frames")
}

fn fixture_events_path() -> PathBuf {
    workspace_root().join("tests/fixtures/visible-learning/events.json")
}

fn golden_dir() -> PathBuf {
    workspace_root().join("tests/golden/video")
}

// --- tiny synthetic fixtures (built per test, never committed) --------------------------------

/// One solid-colour `[h, w, 3]` frame, tightly packed row-major.
fn solid(h: usize, w: usize, color: [u8; 3]) -> Vec<u8> {
    color.repeat(h * w)
}

/// Writes one cell directory: `layout.json` plus `NNNNNN.bin` per entry of `frames`.
fn write_cell(cells_root: &Path, name: &str, h: u64, w: u64, frames: &[Vec<u8>]) {
    let dir = cells_root.join(name);
    std::fs::create_dir_all(&dir).expect("create cell dir");
    std::fs::write(
        dir.join("layout.json"),
        serde_json::json!({"shape": [h, w, 3], "dtype": "u8"}).to_string(),
    )
    .expect("write layout.json");
    for (i, bytes) in frames.iter().enumerate() {
        std::fs::write(dir.join(format!("{i:06}.bin")), bytes).expect("write frame");
    }
}

/// `events.json`: cell name -> `[(frame, source), ...]`.
fn write_events(path: &Path, cells: &BTreeMap<&str, Vec<&str>>) {
    let obj: BTreeMap<&&str, Vec<serde_json::Value>> = cells
        .iter()
        .map(|(name, sources)| {
            let records: Vec<_> = sources
                .iter()
                .enumerate()
                .map(|(i, s)| serde_json::json!({"frame": i, "tick": i, "source": s, "events": []}))
                .collect();
            (name, records)
        })
        .collect();
    std::fs::write(path, serde_json::to_string_pretty(&obj).expect("json"))
        .expect("write events.json");
}

fn write_report(path: &Path, n_episodes: u64, success_rate: f64) {
    let v = serde_json::json!({
        "cells": [
            {"metric": "success_rate", "value": {"scalar": success_rate}, "n_episodes": n_episodes},
        ],
    });
    std::fs::write(path, serde_json::to_string_pretty(&v).expect("json"))
        .expect("write report.json");
}

struct Mosaic {
    frames: PathBuf,
    events: PathBuf,
    report: PathBuf,
    out: PathBuf,
    grid: String,
}

fn run_mosaic(m: &Mosaic) -> Output {
    bin()
        .args(["video", "mosaic"])
        .arg("--frames")
        .arg(&m.frames)
        .arg("--events")
        .arg(&m.events)
        .arg("--report")
        .arg(&m.report)
        .arg("--grid")
        .arg(&m.grid)
        .arg("--out")
        .arg(&m.out)
        .output()
        .expect("run es video mosaic")
}

// --- the committed-fixture / golden generator --------------------------------------------------

/// Regenerates `tests/fixtures/visible-learning/**` and `tests/golden/video/mosaic_4x4_frame0.*`.
/// Run once, explicitly; both are then read-only (spec 1.4). 16 cells, 2 frames each, every
/// `ActionSource` covered at least once (`Policy`, `Human`, `Clamped`, `Fallback`).
#[test]
#[ignore = "fixture/golden generator; run explicitly"]
fn generate_fixture_and_golden() {
    let frames_root = fixture_frames_dir();
    std::fs::create_dir_all(&frames_root).expect("frames dir");
    let mut sources: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for i in 0..16u32 {
        let name = format!("{i:02}");
        let color = [(i * 16) as u8, (255 - i * 16) as u8, 100];
        let cell_sources: Vec<&str> = match i {
            1 => vec!["Clamped", "Policy"],
            2 => vec!["Fallback", "Policy"],
            3 => vec!["Human", "Human"],
            _ => vec!["Policy", "Policy"],
        };
        let frames: Vec<Vec<u8>> = (0..cell_sources.len())
            .map(|_| solid(6, 6, color))
            .collect();
        write_cell(&frames_root, &name, 6, 6, &frames);
        sources.insert(name, cell_sources);
    }
    let events_path = fixture_events_path();
    let by_ref: BTreeMap<&str, Vec<&str>> = sources
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();
    write_events(&events_path, &by_ref);

    let dir = scratch_dir("golden-gen");
    let report_path = dir.join("report.json");
    write_report(&report_path, 16, 0.75);
    let out = dir.join("out");
    let result = run_mosaic(&Mosaic {
        frames: frames_root,
        events: events_path,
        report: report_path,
        out: out.clone(),
        grid: "4x4".to_owned(),
    });
    assert!(result.status.success(), "mosaic run: {}", stderr(&result));

    let golden = golden_dir();
    std::fs::create_dir_all(&golden).expect("golden dir");
    std::fs::copy(out.join("000000.bin"), golden.join("mosaic_4x4_frame0.bin")).expect("copy bin");
    let layout = std::fs::read_to_string(out.join("layout.json")).expect("read layout.json");
    std::fs::write(golden.join("mosaic_4x4_frame0.json"), layout).expect("write golden json");
    println!("wrote {}", golden.join("mosaic_4x4_frame0.bin").display());
}

// --- byte-identical determinism -----------------------------------------------------------------

fn golden_report() -> (u64, f64) {
    (16, 0.75)
}

fn run_committed_fixture(out: &Path) -> Output {
    let dir = out.parent().expect("out has a parent").to_path_buf();
    let report_path = dir.join("report.json");
    let (n, sr) = golden_report();
    write_report(&report_path, n, sr);
    run_mosaic(&Mosaic {
        frames: fixture_frames_dir(),
        events: fixture_events_path(),
        report: report_path,
        out: out.to_path_buf(),
        grid: "4x4".to_owned(),
    })
}

#[test]
fn mosaic_matches_the_golden() {
    let dir = scratch_dir("golden-match");
    let out = dir.join("out");
    let result = run_committed_fixture(&out);
    assert!(result.status.success(), "{}", stderr(&result));

    let golden = golden_dir();
    let got_bin = std::fs::read(out.join("000000.bin")).expect("read out frame0");
    let want_bin = std::fs::read(golden.join("mosaic_4x4_frame0.bin"))
        .expect("read golden bin (run generate_fixture_and_golden first)");
    assert_eq!(got_bin, want_bin, "frame0 bytes differ from the golden");

    let got_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("layout.json")).unwrap()).unwrap();
    let want_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(golden.join("mosaic_4x4_frame0.json"))
            .expect("read golden json (run generate_fixture_and_golden first)"),
    )
    .unwrap();
    assert_eq!(got_json, want_json, "layout.json differs from the golden");
}

#[test]
fn two_runs_are_byte_identical() {
    let dir = scratch_dir("determinism");
    let (out1, out2) = (dir.join("out1"), dir.join("out2"));
    let r1 = run_committed_fixture(&out1);
    let r2 = run_committed_fixture(&out2);
    assert!(r1.status.success() && r2.status.success());

    for name in ["000000.bin", "000001.bin", "layout.json"] {
        let a = std::fs::read(out1.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let b = std::fs::read(out2.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(a, b, "{name} differs between two runs of the same inputs");
    }
}

// --- red border ------------------------------------------------------------------------------

#[test]
fn a_clamped_record_draws_a_red_border() {
    let dir = scratch_dir("border");
    let frames_root = dir.join("frames");
    let (color_a, color_b) = ([10u8, 20, 30], [200u8, 210, 220]);
    write_cell(
        &frames_root,
        "00",
        5,
        5,
        &[solid(5, 5, color_a), solid(5, 5, color_a)],
    );
    write_cell(
        &frames_root,
        "01",
        5,
        5,
        &[solid(5, 5, color_b), solid(5, 5, color_b)],
    );
    let events = dir.join("events.json");
    write_events(
        &events,
        &BTreeMap::from([
            ("00", vec!["Clamped", "Policy"]),
            ("01", vec!["Policy", "Policy"]),
        ]),
    );
    let report = dir.join("report.json");
    write_report(&report, 2, 0.5);
    let out = dir.join("out");
    let result = run_mosaic(&Mosaic {
        frames: frames_root,
        events,
        report,
        out: out.clone(),
        grid: "1x2".to_owned(),
    });
    assert!(result.status.success(), "{}", stderr(&result));

    let total_w = 10usize; // 2 cols * 5px
    let read_px = |buf: &[u8], x: usize, y: usize| -> [u8; 3] {
        let d = (y * total_w + x) * 3;
        [buf[d], buf[d + 1], buf[d + 2]]
    };

    // Frame 0: cell "00" is Clamped -> red ring, colour_a interior untouched.
    let f0 = std::fs::read(out.join("000000.bin")).unwrap();
    assert_eq!(
        read_px(&f0, 0, 0),
        [255, 0, 0],
        "top-left corner of a clamped cell"
    );
    assert_eq!(
        read_px(&f0, 4, 0),
        [255, 0, 0],
        "top-right corner of a clamped cell"
    );
    assert_eq!(
        read_px(&f0, 0, 4),
        [255, 0, 0],
        "bottom-left corner of a clamped cell"
    );
    assert_eq!(
        read_px(&f0, 4, 4),
        [255, 0, 0],
        "bottom-right corner of a clamped cell"
    );
    assert_eq!(
        read_px(&f0, 2, 2),
        color_a,
        "interior of a clamped cell is untouched"
    );
    // Cell "01" is Policy this frame: no red anywhere in its tile.
    for y in 0..5 {
        for x in 0..5 {
            assert_eq!(
                read_px(&f0, 5 + x, y),
                color_b,
                "Policy cell must not get a border"
            );
        }
    }

    // Frame 1: both cells are Policy -> "00" no longer has a border.
    let f1 = std::fs::read(out.join("000001.bin")).unwrap();
    for y in 0..5 {
        for x in 0..5 {
            assert_eq!(
                read_px(&f1, x, y),
                color_a,
                "border must not persist once the cell is Policy"
            );
        }
    }
}

// --- fixed bitmap digits ----------------------------------------------------------------------

#[test]
fn the_overlay_digits_are_a_fixed_bitmap() {
    let dir = scratch_dir("digits");
    let frames_root = dir.join("frames");
    write_cell(&frames_root, "00", 20, 40, &[solid(20, 40, [0, 0, 0])]);
    let events = dir.join("events.json");
    write_events(&events, &BTreeMap::from([("00", vec!["Policy"])]));
    let report = dir.join("report.json");
    write_report(&report, 0, 0.0); // n_episodes=0, success=0% -> digit '0' at every drawn glyph
    let out = dir.join("out");
    let result = run_mosaic(&Mosaic {
        frames: frames_root,
        events,
        report,
        out: out.clone(),
        grid: "1x1".to_owned(),
    });
    assert!(result.status.success(), "{}", stderr(&result));

    let total_w = 40usize;
    let y0 = 20 + 1; // grid height (20) + the label strip's 1px top margin
    let f0 = std::fs::read(out.join("000000.bin")).unwrap();
    let is_white = |buf: &[u8], x: usize, y: usize| -> bool {
        let d = (y * total_w + x) * 3;
        buf[d..d + 3] == [255, 255, 255]
    };
    // Digit '0' bitmap: 111 / 101 / 101 / 101 / 111, drawn at (x=1, y=y0).
    let want = [0b111u8, 0b101, 0b101, 0b101, 0b111];
    for (ry, row) in want.iter().enumerate() {
        for rx in 0..3 {
            let expect_white = row & (1 << (2 - rx)) != 0;
            assert_eq!(
                is_white(&f0, 1 + rx, y0 + ry),
                expect_white,
                "digit '0' pixel ({rx},{ry})"
            );
        }
    }
}

// --- padding: a short cell holds its last frame -------------------------------------------------

#[test]
fn a_short_cell_holds_its_last_frame() {
    let dir = scratch_dir("padding");
    let frames_root = dir.join("frames");
    let (short_color, long0, long1, long2) =
        ([1u8, 2, 3], [10u8, 10, 10], [20u8, 20, 20], [30u8, 30, 30]);
    write_cell(&frames_root, "00", 4, 4, &[solid(4, 4, short_color)]);
    write_cell(
        &frames_root,
        "01",
        4,
        4,
        &[solid(4, 4, long0), solid(4, 4, long1), solid(4, 4, long2)],
    );
    let events = dir.join("events.json");
    write_events(
        &events,
        &BTreeMap::from([
            ("00", vec!["Policy"]),
            ("01", vec!["Policy", "Policy", "Policy"]),
        ]),
    );
    let report = dir.join("report.json");
    write_report(&report, 1, 1.0);
    let out = dir.join("out");
    let result = run_mosaic(&Mosaic {
        frames: frames_root,
        events,
        report,
        out: out.clone(),
        grid: "1x2".to_owned(),
    });
    assert!(result.status.success(), "{}", stderr(&result));
    assert!(
        out.join("000002.bin").exists(),
        "mosaic frame count must be the longest cell's"
    );

    let total_w = 8usize;
    let read_px = |buf: &[u8], x: usize, y: usize| -> [u8; 3] {
        let d = (y * total_w + x) * 3;
        [buf[d], buf[d + 1], buf[d + 2]]
    };
    let f2 = std::fs::read(out.join("000002.bin")).unwrap();
    assert_eq!(
        read_px(&f2, 0, 0),
        short_color,
        "the short cell repeats its only frame"
    );
    assert_eq!(
        read_px(&f2, 4, 0),
        long2,
        "the long cell shows its own frame 2"
    );
}

// --- rejected inputs ---------------------------------------------------------------------------

#[test]
fn mismatched_layouts_are_rejected() {
    let dir = scratch_dir("mismatch");
    let frames_root = dir.join("frames");
    write_cell(&frames_root, "00", 4, 4, &[solid(4, 4, [1, 1, 1])]);
    write_cell(&frames_root, "01", 5, 5, &[solid(5, 5, [2, 2, 2])]);
    let events = dir.join("events.json");
    write_events(
        &events,
        &BTreeMap::from([("00", vec!["Policy"]), ("01", vec!["Policy"])]),
    );
    let report = dir.join("report.json");
    write_report(&report, 1, 1.0);
    let out = dir.join("out");
    let result = run_mosaic(&Mosaic {
        frames: frames_root,
        events,
        report,
        out,
        grid: "1x2".to_owned(),
    });
    assert_eq!(result.status.code(), Some(1), "{}", stdout(&result));
    assert!(stderr(&result).contains("expected"), "{}", stderr(&result));
}

#[test]
fn malformed_inputs_are_rejected_before_allocation() {
    // (a) a byte length disagreeing with layout.json.
    {
        let dir = scratch_dir("malformed-len");
        let frames_root = dir.join("frames");
        write_cell(&frames_root, "00", 4, 4, &[vec![0u8; 3]]); // declares 48 bytes, has 3
        let events = dir.join("events.json");
        write_events(&events, &BTreeMap::from([("00", vec!["Policy"])]));
        let report = dir.join("report.json");
        write_report(&report, 1, 1.0);
        let out = dir.join("out");
        let result = run_mosaic(&Mosaic {
            frames: frames_root,
            events,
            report,
            out,
            grid: "1x1".to_owned(),
        });
        assert_eq!(
            result.status.code(),
            Some(1),
            "byte-length mismatch: {}",
            stderr(&result)
        );
    }
    // (b) an events.json frame index that does not correspond to any real frame.
    {
        let dir = scratch_dir("malformed-frame-idx");
        let frames_root = dir.join("frames");
        write_cell(&frames_root, "00", 4, 4, &[solid(4, 4, [1, 1, 1])]);
        let events = dir.join("events.json");
        std::fs::write(
            &events,
            serde_json::json!({"00": [{"frame": 5, "tick": 0, "source": "Policy", "events": []}]})
                .to_string(),
        )
        .unwrap();
        let report = dir.join("report.json");
        write_report(&report, 1, 1.0);
        let out = dir.join("out");
        let result = run_mosaic(&Mosaic {
            frames: frames_root,
            events,
            report,
            out,
            grid: "1x1".to_owned(),
        });
        assert_eq!(
            result.status.code(),
            Some(1),
            "frame index past the end: {}",
            stderr(&result)
        );
    }
    // (c) a cell count that does not match the declared grid.
    {
        let dir = scratch_dir("malformed-cell-count");
        let frames_root = dir.join("frames");
        write_cell(&frames_root, "00", 4, 4, &[solid(4, 4, [1, 1, 1])]);
        let events = dir.join("events.json");
        write_events(&events, &BTreeMap::from([("00", vec!["Policy"])]));
        let report = dir.join("report.json");
        write_report(&report, 1, 1.0);
        let out = dir.join("out");
        let result = run_mosaic(&Mosaic {
            frames: frames_root,
            events,
            report,
            out,
            grid: "1x2".to_owned(), // declares 2 cells, only 1 exists
        });
        assert_eq!(
            result.status.code(),
            Some(1),
            "cell count mismatch: {}",
            stderr(&result)
        );
    }
    // (d) a layout.json shape product that overflows u64.
    {
        let dir = scratch_dir("malformed-overflow");
        let frames_root = dir.join("frames");
        let cell_dir = frames_root.join("00");
        std::fs::create_dir_all(&cell_dir).unwrap();
        std::fs::write(
            cell_dir.join("layout.json"),
            serde_json::json!({"shape": [u64::MAX, u64::MAX, 3], "dtype": "u8"}).to_string(),
        )
        .unwrap();
        std::fs::write(cell_dir.join("000000.bin"), [0u8]).unwrap();
        let events = dir.join("events.json");
        write_events(&events, &BTreeMap::from([("00", vec!["Policy"])]));
        let report = dir.join("report.json");
        write_report(&report, 1, 1.0);
        let out = dir.join("out");
        let result = run_mosaic(&Mosaic {
            frames: frames_root,
            events,
            report,
            out,
            grid: "1x1".to_owned(),
        });
        assert_eq!(
            result.status.code(),
            Some(1),
            "shape overflow: {}",
            stderr(&result)
        );
    }
}

/// No `proptest` dev-dependency is declared for this crate (out of the packet's context, and
/// the packet forbids a new dependency), so this is a hand-rolled adversarial-input sweep
/// rather than a `proptest!` property: every payload here must make the process exit cleanly
/// (never a signal, never a Rust panic backtrace on stderr), whatever exit code it picks.
#[test]
fn arbitrary_events_json_never_panics() {
    let dir = scratch_dir("fuzz");
    let frames_root = dir.join("frames");
    write_cell(&frames_root, "00", 4, 4, &[solid(4, 4, [1, 1, 1])]);
    let report = dir.join("report.json");
    write_report(&report, 1, 1.0);

    let payloads = [
        "",
        "not json at all {{{",
        "null",
        "[]",
        "{}",
        r#"{"00": "not an array"}"#,
        r#"{"00": [{"frame": "not a number", "source": "Policy"}]}"#,
        r#"{"00": [{"frame": 99999999999999999999999999999999999, "source": "Policy"}]}"#,
        r#"{"00": [{"source": "Policy"}]}"#,
        r#"{"00": [{"frame": 0, "source": 123}]}"#,
        r#"{"00": [{"frame": 0, "source": "Bogus"}]}"#,
        r#"{"00": [{"frame": 0, "source": "Policy"}, {"frame": 1, "source": "Policy"}]}"#,
        "\u{0}\u{1}\u{2} garbage \u{feff}",
    ];
    for (i, payload) in payloads.iter().enumerate() {
        let events = dir.join(format!("events-{i}.json"));
        std::fs::write(&events, payload).unwrap();
        let out = dir.join(format!("out-{i}"));
        let result = run_mosaic(&Mosaic {
            frames: frames_root.clone(),
            events,
            report: report.clone(),
            out,
            grid: "1x1".to_owned(),
        });
        assert!(
            result.status.code().is_some(),
            "payload {i} {payload:?} crashed the process"
        );
        assert!(
            !stderr(&result).contains("panicked"),
            "payload {i} {payload:?} panicked:\n{}",
            stderr(&result)
        );
    }
}

// --- the mp4 encoder (needs cv2) -----------------------------------------------------------------

fn cv2_python() -> Option<String> {
    let candidates: Vec<String> = match std::env::var("ES_CV2_PYTHON") {
        Ok(p) if !p.trim().is_empty() => vec![p],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    };
    candidates.into_iter().find(|p| {
        Command::new(p)
            .args(["-c", "import cv2"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

#[test]
#[ignore = "needs a Python with opencv-python; see ES_CV2_PYTHON in the packet oracle"]
fn encode_produces_a_playable_file() {
    let Some(python) = cv2_python() else {
        println!("SKIP encode_video: no Python with cv2 found (set ES_CV2_PYTHON)");
        return;
    };

    let dir = scratch_dir("encode");
    let frames_root = dir.join("frames");
    write_cell(
        &frames_root,
        "00",
        4,
        4,
        &[solid(4, 4, [10, 20, 30]), solid(4, 4, [40, 50, 60])],
    );
    write_cell(
        &frames_root,
        "01",
        4,
        4,
        &[solid(4, 4, [70, 80, 90]), solid(4, 4, [100, 110, 120])],
    );
    let events = dir.join("events.json");
    write_events(
        &events,
        &BTreeMap::from([
            ("00", vec!["Policy", "Policy"]),
            ("01", vec!["Policy", "Policy"]),
        ]),
    );
    let report = dir.join("report.json");
    write_report(&report, 2, 1.0);
    let out = dir.join("out");
    let mosaic_result = run_mosaic(&Mosaic {
        frames: frames_root,
        events,
        report,
        out: out.clone(),
        grid: "1x2".to_owned(),
    });
    assert!(mosaic_result.status.success(), "{}", stderr(&mosaic_result));

    let script = workspace_root().join("python/es/encode_video.py");
    let mp4 = dir.join("demo.mp4");
    let enc = Command::new(&python)
        .arg(&script)
        .arg("--frames")
        .arg(&out)
        .arg("--out")
        .arg(&mp4)
        .args(["--fps", "5"])
        .output()
        .expect("run encode_video.py");
    assert!(enc.status.success(), "encode_video.py: {}", stderr(&enc));
    let printed: serde_json::Value =
        serde_json::from_str(stdout(&enc).trim()).expect("one JSON line");
    assert_eq!(printed["frames"], 2);
    assert_eq!(printed["codec"], "mp4v");

    let readback = Command::new(&python)
        .args([
            "-c",
            "import cv2, json, sys; cap = cv2.VideoCapture(sys.argv[1]); print(json.dumps({\
             'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), \
             'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
             'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}))",
            mp4.to_str().unwrap(),
        ])
        .output()
        .expect("run cv2 readback");
    assert!(readback.status.success(), "{}", stderr(&readback));
    let got: serde_json::Value =
        serde_json::from_str(stdout(&readback).trim()).expect("one JSON line");
    assert_eq!(got["frames"], printed["frames"]);
    assert_eq!(got["width"], printed["width"]);
    assert_eq!(got["height"], printed["height"]);
    println!("RAN encode_video");
}

// --- es video showcase (packet M7/R2) ---------------------------------------------------------

/// The `--look` flag is parsed, `lambert` and `full` are the only two values, and the default
/// is `lambert` — so a re-render of a committed run still reproduces its recorded frames.
///
/// Parsing only: a real render needs a Vulkan device and a run directory. A build without the
/// `render` feature links no renderer at all and says so on stderr, which is what this test
/// then reports instead of pretending to have run.
#[test]
fn showcase_look_flag_is_parsed() {
    let out = scratch_dir("showcase-look");
    let base = |look: Option<&str>| {
        let mut cmd = bin();
        cmd.args(["video", "showcase"])
            .args(["--run", out.join("no-such-run").to_str().unwrap()])
            .args(["--scene", "tests/fixtures/mjcf/so101_pick_place.xml"])
            .args(["--out", out.join("frames").to_str().unwrap()])
            .args(["--eye", "0.66,-0.46,0.52"])
            .args(["--look-at", "0.14,-0.04,0.04"]);
        if let Some(look) = look {
            cmd.args(["--look", look]);
        }
        cmd.current_dir(workspace_root())
            .output()
            .expect("run es video showcase")
    };

    let help = bin()
        .args(["video", "showcase", "--help"])
        .output()
        .expect("run --help");
    if stderr(&help).contains("needs the `render` feature") {
        println!("SKIP showcase_look_flag_is_parsed: built without the `render` feature");
        return;
    }
    assert!(
        stdout(&help).contains("--look lambert|full"),
        "the help text does not document --look:\n{}",
        stdout(&help)
    );

    let bad = base(Some("cinematic"));
    assert_eq!(bad.status.code(), Some(2), "--look cinematic must be usage");
    assert!(
        stderr(&bad).contains("--look: expected lambert or full"),
        "{}",
        stderr(&bad)
    );

    // Accepted: parsing gets past `--look` and the command fails later, on the run directory
    // that is not there or on the missing Vulkan device — a runtime error, not a usage one.
    for look in [None, Some("lambert"), Some("full")] {
        let got = base(look);
        assert_eq!(
            got.status.code(),
            Some(1),
            "--look {look:?} should parse and then fail on the missing run: {}",
            stderr(&got)
        );
        assert!(
            !stderr(&got).contains("--look:"),
            "--look {look:?} was rejected: {}",
            stderr(&got)
        );
    }
    println!("RAN showcase_look_flag_is_parsed: lambert (default), full, and a rejected value");
}

// --- es video showcase --path pt (packet M7/R3) ----------------------------------------------

/// Oracle 8: `--path pt` and its four knobs parse, the two enumerated ones reject a value
/// they do not name, and `--spp 0` is a usage error rather than a division by zero later.
///
/// Parsing only, as above: a real render needs a Vulkan device and a run directory.
#[test]
fn showcase_pt_flags_are_parsed() {
    let out = scratch_dir("showcase-pt");
    let base = |extra: &[&str]| {
        let mut cmd = bin();
        cmd.args(["video", "showcase"])
            .args(["--run", out.join("no-such-run").to_str().unwrap()])
            .args(["--scene", "tests/fixtures/mjcf/so101_pick_place.xml"])
            .args(["--out", out.join("frames").to_str().unwrap()])
            .args(["--eye", "0.66,-0.46,0.52"])
            .args(["--look-at", "0.14,-0.04,0.04"])
            .args(extra);
        cmd.current_dir(workspace_root())
            .output()
            .expect("run es video showcase")
    };

    let help = bin()
        .args(["video", "showcase", "--help"])
        .output()
        .expect("run --help");
    if stderr(&help).contains("needs the `render` feature") {
        println!("SKIP showcase_pt_flags_are_parsed: built without the `render` feature");
        return;
    }
    assert!(
        stdout(&help).contains("--path rs|pt") && stdout(&help).contains("--tonemap"),
        "the help text does not document the PT flags:\n{}",
        stdout(&help)
    );

    for (args, expected) in [
        (vec!["--path", "raytrace"], "--path: expected rs or pt"),
        (
            vec!["--path", "pt", "--tonemap", "filmic"],
            "--tonemap: expected reinhard or aces",
        ),
        (vec!["--path", "pt", "--spp", "0"], "--spp"),
        (vec!["--path", "pt", "--bounces", "0"], "--bounces"),
    ] {
        let got = base(&args);
        assert_eq!(got.status.code(), Some(2), "{args:?} must be usage");
        assert!(
            stderr(&got).contains(expected),
            "{args:?}: {}",
            stderr(&got)
        );
    }

    // Accepted: parsing gets past every flag and the command fails later, on the run
    // directory that is not there or on the missing Vulkan device — a runtime error.
    for args in [
        vec!["--path", "pt"],
        vec!["--path", "pt", "--spp", "8"],
        vec![
            "--path",
            "pt",
            "--spp",
            "64",
            "--bounces",
            "3",
            "--exposure",
            "1.5",
            "--tonemap",
            "aces",
        ],
        vec!["--path", "pt", "--tonemap", "reinhard"],
    ] {
        let got = base(&args);
        assert_eq!(
            got.status.code(),
            Some(1),
            "{args:?} should parse and then fail on the missing run: {}",
            stderr(&got)
        );
        assert!(
            !stderr(&got).contains("expected"),
            "{args:?} was rejected: {}",
            stderr(&got)
        );
    }
    println!(
        "RAN showcase_pt_flags_are_parsed: --path rs|pt, --spp, --bounces, --exposure, \
         --tonemap reinhard|aces"
    );
}

// --- es video showcase --accumulate (packet M7/R4) --------------------------------------------

/// Oracle 7: `--accumulate` and `--max-history` parse, `--accumulate` without `--path pt` is a
/// usage error rather than a silently ignored flag, and `--max-history 0` is rejected before it
/// can divide anything.
///
/// Parsing only: a real render needs a Vulkan device and a run directory.
#[test]
fn showcase_accumulate_flags_are_parsed() {
    let out = scratch_dir("showcase-accumulate");
    let base = |extra: &[&str]| {
        let mut cmd = bin();
        cmd.args(["video", "showcase"])
            .args(["--run", out.join("no-such-run").to_str().unwrap()])
            .args(["--scene", "tests/fixtures/mjcf/so101_pick_place.xml"])
            .args(["--out", out.join("frames").to_str().unwrap()])
            .args(["--eye", "0.66,-0.46,0.52"])
            .args(["--look-at", "0.14,-0.04,0.04"])
            .args(extra);
        cmd.current_dir(workspace_root())
            .output()
            .expect("run es video showcase")
    };

    let help = bin()
        .args(["video", "showcase", "--help"])
        .output()
        .expect("run --help");
    if stderr(&help).contains("needs the `render` feature") {
        println!("SKIP showcase_accumulate_flags_are_parsed: built without the `render` feature");
        return;
    }
    assert!(
        stdout(&help).contains("--accumulate") && stdout(&help).contains("--max-history"),
        "the help text does not document the accumulation flags:\n{}",
        stdout(&help)
    );

    for (args, expected) in [
        (vec!["--accumulate"], "--accumulate is a `pt` flag"),
        (
            vec!["--path", "pt", "--accumulate", "--max-history", "0"],
            "--max-history must be greater than zero",
        ),
    ] {
        let got = base(&args);
        assert_eq!(got.status.code(), Some(2), "{args:?} must be usage");
        assert!(
            stderr(&got).contains(expected),
            "{args:?}: {}",
            stderr(&got)
        );
    }

    // Accepted: parsing gets past both flags and the command fails later, on the run
    // directory that is not there or on the missing Vulkan device.
    for args in [
        vec!["--path", "pt", "--accumulate"],
        vec!["--path", "pt", "--accumulate", "--max-history", "32"],
        vec![
            "--path",
            "pt",
            "--spp",
            "4",
            "--accumulate",
            "--max-history",
            "8",
            "--exposure",
            "32",
        ],
    ] {
        let got = base(&args);
        assert_eq!(
            got.status.code(),
            Some(1),
            "{args:?} should parse and then fail on the missing run: {}",
            stderr(&got)
        );
        assert!(
            !stderr(&got).contains("--accumulate") && !stderr(&got).contains("--max-history"),
            "{args:?} was rejected: {}",
            stderr(&got)
        );
    }
    println!(
        "RAN showcase_accumulate_flags_are_parsed: --accumulate, --max-history, and two rejections"
    );
}
