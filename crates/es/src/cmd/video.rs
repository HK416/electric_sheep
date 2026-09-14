//! `es video mosaic` -- the 4x4 grid video assembly for the visible-learning demo (design note
//! `docs/design/visible-learning.md` sections 7.2, 8, 9; spec 1.4, 2.4, 3.4, 25.1, 26.1).
//!
//! Pure Rust, no GPU, no Vulkan, no Python: this module only reads raw `.bin` + `layout.json`
//! frame directories (the same format `tests/golden/render/*.json` uses) and an `events.json`
//! sidecar, tiles them into a grid, draws a fixed red border on a step the Safety Plane clamped
//! or fell back on, and a fixed-bitmap label strip from the evaluation report's success rate.
//! Encoding the result to an mp4 is `python/es/encode_video.py`, a separate step this binary
//! never calls (spec 2.4): the container is outside the execution hash chain (spec 5.3).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

const HELP: &str = "\
es video mosaic --frames <dir> --events <events.json> --report <report.json>
                 --grid RxC --out <dir> [--label-height N]

Reads one cell per subdirectory of <frames>, each holding a raw `NNNNNN.bin` + `layout.json`
frame sequence (row-major, tightly packed, u8, 3 channels). Every byte length, frame index and
cell count is validated before the mosaic buffer is allocated (spec 25.1). Cells are tiled
row-major in cell-name order (an RxC grid) with no scaling, cropping or resampling: mismatched
cell layouts are refused, never resized. A cell whose <events.json> record for a frame is
`Clamped` or `Fallback` gets a fixed-width red border that frame; `Policy` and `Human` get
none. A label strip under the grid draws the episode count and success rate from
<report.json>'s `success_rate` metric, from an embedded bitmap digit font. The mosaic's frame
count is the longest cell; shorter cells repeat their last frame. Writes <out>/NNNNNN.bin (one
per timestep) plus one shared <out>/layout.json.

    --label-height N   pixel height of the label strip under the grid (default 8)
";

pub fn dispatch(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("mosaic") => match parse_and_run(&args[1..]) {
            Ok(()) => 0,
            Err(VideoError::Usage(m)) => {
                eprintln!("{m}");
                2
            }
            Err(VideoError::Runtime(m)) => {
                eprintln!("error: {m}");
                1
            }
        },
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            0
        }
        Some(other) => {
            eprintln!("es video: unknown subcommand '{other}'\n\n{HELP}");
            2
        }
    }
}

enum VideoError {
    Usage(String),
    Runtime(String),
}

struct Opts {
    frames: PathBuf,
    events: PathBuf,
    report: PathBuf,
    rows: usize,
    cols: usize,
    out: PathBuf,
    label_height: usize,
}

fn parse_and_run(args: &[String]) -> Result<(), VideoError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let (mut frames, mut events, mut report, mut grid, mut out) = (None, None, None, None, None);
    let mut label_height = 8usize;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| VideoError::Usage(format!("{a}: missing value\n\n{HELP}")))
        };
        match a.as_str() {
            "--frames" => frames = Some(PathBuf::from(val()?)),
            "--events" => events = Some(PathBuf::from(val()?)),
            "--report" => report = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--grid" => grid = Some(parse_grid(val()?)?),
            "--label-height" => {
                label_height = val()?.parse().map_err(|_| {
                    VideoError::Usage(format!("--label-height: invalid value\n\n{HELP}"))
                })?;
            }
            other => {
                return Err(VideoError::Usage(format!(
                    "unknown flag '{other}'\n\n{HELP}"
                )))
            }
        }
    }
    let (Some(frames), Some(events), Some(report), Some((rows, cols)), Some(out)) =
        (frames, events, report, grid, out)
    else {
        return Err(VideoError::Usage(format!(
            "--frames, --events, --report, --grid and --out are all required\n\n{HELP}"
        )));
    };
    run(&Opts {
        frames,
        events,
        report,
        rows,
        cols,
        out,
        label_height,
    })
}

fn parse_grid(raw: &str) -> Result<(usize, usize), VideoError> {
    let (r, c) = raw.split_once(['x', 'X']).ok_or_else(|| {
        VideoError::Usage(format!(
            "--grid: expected RxC (e.g. 4x4), got {raw:?}\n\n{HELP}"
        ))
    })?;
    let rows: usize = r
        .parse()
        .map_err(|_| VideoError::Usage(format!("--grid: invalid rows {r:?}\n\n{HELP}")))?;
    let cols: usize = c
        .parse()
        .map_err(|_| VideoError::Usage(format!("--grid: invalid cols {c:?}\n\n{HELP}")))?;
    if rows == 0 || cols == 0 {
        return Err(VideoError::Usage(format!(
            "--grid: rows and cols must both be > 0, got {raw:?}\n\n{HELP}"
        )));
    }
    Ok((rows, cols))
}

// --- input model (untrusted; spec 25.1) ------------------------------------------------------

#[derive(Deserialize)]
struct LayoutFile {
    shape: [u64; 3],
    dtype: String,
}

#[derive(Deserialize)]
struct EventRecord {
    frame: u64,
    source: String,
}

#[derive(Deserialize)]
struct ReportFile {
    cells: Vec<ReportCell>,
}

#[derive(Deserialize)]
struct ReportCell {
    metric: String,
    #[serde(default)]
    n_episodes: u64,
    #[serde(default)]
    value: Option<MetricValueJson>,
}

#[derive(Deserialize)]
struct MetricValueJson {
    #[serde(default)]
    scalar: Option<f64>,
}

struct Cell {
    shape: [u64; 3],
    /// Frame `i`'s bytes are `fs::read(frames[i])`; `layout.json` already pinned their length.
    frames: Vec<PathBuf>,
}

fn rt(msg: impl Into<String>) -> VideoError {
    VideoError::Runtime(msg.into())
}

/// Every subdirectory of `dir`, cell name -> path, in file order (a `BTreeMap`, never a
/// `HashMap`: spec 3.4 forbids hash-iteration dependence).
fn list_cells(dir: &Path) -> Result<BTreeMap<String, PathBuf>, VideoError> {
    let rd = fs::read_dir(dir).map_err(|e| rt(format!("{}: {e}", dir.display())))?;
    let mut cells = BTreeMap::new();
    for entry in rd {
        let entry = entry.map_err(|e| rt(format!("{}: {e}", dir.display())))?;
        let path = entry.path();
        if path.is_dir() {
            cells.insert(entry.file_name().to_string_lossy().into_owned(), path);
        }
    }
    Ok(cells)
}

/// Reads and validates one cell's `layout.json` and its `NNNNNN.bin` frame sequence. Every
/// check here runs before the caller allocates the mosaic buffer (spec 25.1).
fn load_cell(name: &str, dir: &Path) -> Result<Cell, VideoError> {
    let layout_path = dir.join("layout.json");
    let text = fs::read_to_string(&layout_path)
        .map_err(|e| rt(format!("{}: {e}", layout_path.display())))?;
    let layout: LayoutFile =
        serde_json::from_str(&text).map_err(|e| rt(format!("{}: {e}", layout_path.display())))?;
    if layout.dtype != "u8" {
        return Err(rt(format!(
            "{}: dtype {:?} unsupported (only raw u8 frames are read)",
            layout_path.display(),
            layout.dtype
        )));
    }
    let [h, w, c] = layout.shape;
    if h == 0 || w == 0 || c != 3 {
        return Err(rt(format!(
            "{}: shape {:?} must be [height>0, width>0, 3]",
            layout_path.display(),
            layout.shape
        )));
    }
    let frame_bytes = h
        .checked_mul(w)
        .and_then(|v| v.checked_mul(c))
        .ok_or_else(|| {
            rt(format!(
                "{}: shape {:?} overflows",
                layout_path.display(),
                layout.shape
            ))
        })?;

    let mut numbered: Vec<(u64, PathBuf)> = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| rt(format!("{}: {e}", dir.display())))? {
        let path = entry
            .map_err(|e| rt(format!("{}: {e}", dir.display())))?
            .path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let idx: u64 = stem.parse().map_err(|_| {
            rt(format!(
                "{}: frame file name is not a plain number",
                path.display()
            ))
        })?;
        numbered.push((idx, path));
    }
    if numbered.is_empty() {
        return Err(rt(format!(
            "cell {name:?}: no frames under {}",
            dir.display()
        )));
    }
    numbered.sort_by_key(|(i, _)| *i);
    for (want, (idx, path)) in numbered.iter().enumerate() {
        if *idx != want as u64 {
            return Err(rt(format!(
                "cell {name:?}: frame files are not contiguous from 0 (found {} at position {want})",
                path.display()
            )));
        }
        let len = fs::metadata(path)
            .map_err(|e| rt(format!("{}: {e}", path.display())))?
            .len();
        if len != frame_bytes {
            return Err(rt(format!(
                "{}: {len} byte(s), layout.json declares {frame_bytes}",
                path.display()
            )));
        }
    }
    Ok(Cell {
        shape: layout.shape,
        frames: numbered.into_iter().map(|(_, p)| p).collect(),
    })
}

/// Reads `events.json` -- one `Vec<EventRecord>` per cell, keyed exactly like `cells` -- and
/// returns each cell's frame-ordered `source` list. `rec.frame` must equal its position: an
/// `events.json` that disagrees with the frames it claims to describe is untrusted input, not
/// a hint to reconcile (spec 25.1).
fn load_events(
    path: &Path,
    cells: &BTreeMap<String, Cell>,
) -> Result<BTreeMap<String, Vec<String>>, VideoError> {
    let text = fs::read_to_string(path).map_err(|e| rt(format!("{}: {e}", path.display())))?;
    let raw: BTreeMap<String, Vec<EventRecord>> =
        serde_json::from_str(&text).map_err(|e| rt(format!("{}: {e}", path.display())))?;

    let want: BTreeSet<&String> = cells.keys().collect();
    let got: BTreeSet<&String> = raw.keys().collect();
    if want != got {
        return Err(rt(format!(
            "{}: cells {got:?} do not match the frame directories under --frames ({want:?})",
            path.display()
        )));
    }

    let mut sources = BTreeMap::new();
    for (name, records) in raw {
        let cell = &cells[&name]; // safe: key sets just proven equal
        if records.len() != cell.frames.len() {
            return Err(rt(format!(
                "{}: cell {name:?} has {} event record(s), {} frame(s) on disk",
                path.display(),
                records.len(),
                cell.frames.len()
            )));
        }
        let mut src = Vec::with_capacity(records.len());
        for (i, rec) in records.into_iter().enumerate() {
            if rec.frame != i as u64 {
                return Err(rt(format!(
                    "{}: cell {name:?} record {i} has frame index {}, expected {i}",
                    path.display(),
                    rec.frame
                )));
            }
            if !matches!(
                rec.source.as_str(),
                "Policy" | "Human" | "Clamped" | "Fallback"
            ) {
                return Err(rt(format!(
                    "{}: cell {name:?} frame {i}: unknown action source {:?}",
                    path.display(),
                    rec.source
                )));
            }
            src.push(rec.source);
        }
        sources.insert(name, src);
    }
    Ok(sources)
}

/// `(success_rate, n_episodes)` from `report.json`'s `success_rate` cell (spec 12.4's
/// `MetricSpec::SuccessRate`); nothing is invented when it is missing.
fn load_success_rate(path: &Path) -> Result<(f64, u64), VideoError> {
    let text = fs::read_to_string(path).map_err(|e| rt(format!("{}: {e}", path.display())))?;
    let report: ReportFile =
        serde_json::from_str(&text).map_err(|e| rt(format!("{}: {e}", path.display())))?;
    report
        .cells
        .iter()
        .filter(|c| c.metric == "success_rate")
        .find_map(|c| c.value.as_ref()?.scalar.map(|s| (s, c.n_episodes)))
        .ok_or_else(|| {
            rt(format!(
                "{}: no success_rate cell with a scalar value",
                path.display()
            ))
        })
}

// --- mosaic assembly --------------------------------------------------------------------------

fn run(opts: &Opts) -> Result<(), VideoError> {
    let cell_count = opts
        .rows
        .checked_mul(opts.cols)
        .ok_or_else(|| rt("--grid: rows * cols overflows"))?;
    let dirs = list_cells(&opts.frames)?;
    if dirs.len() != cell_count {
        return Err(rt(format!(
            "{}: expected {cell_count} cell(s) for a {}x{} grid, found {}",
            opts.frames.display(),
            opts.rows,
            opts.cols,
            dirs.len()
        )));
    }
    let mut cells = BTreeMap::new();
    for (name, dir) in &dirs {
        cells.insert(name.clone(), load_cell(name, dir)?);
    }
    let shape = cells
        .values()
        .next()
        .expect("dirs.len() == cell_count > 0")
        .shape;
    for (name, cell) in &cells {
        if cell.shape != shape {
            return Err(rt(format!(
                "cell {name:?} is {:?}, expected {shape:?} (every cell in a mosaic shares one \
                 layout; a mismatch is refused, never resized)",
                cell.shape
            )));
        }
    }
    let sources = load_events(&opts.events, &cells)?;
    let (success_rate, n_episodes) = load_success_rate(&opts.report)?;

    let (th, tw) = (shape[0] as usize, shape[1] as usize);
    let total_w = opts
        .cols
        .checked_mul(tw)
        .ok_or_else(|| rt("mosaic width overflows"))?;
    let grid_h = opts
        .rows
        .checked_mul(th)
        .ok_or_else(|| rt("mosaic height overflows"))?;
    let total_h = grid_h
        .checked_add(opts.label_height)
        .ok_or_else(|| rt("mosaic height overflows"))?;
    let frame_pixels = total_w
        .checked_mul(total_h)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| rt("mosaic frame size overflows"))?;

    let max_frames = cells.values().map(|c| c.frames.len()).max().unwrap_or(0);
    fs::create_dir_all(&opts.out).map_err(|e| rt(format!("{}: {e}", opts.out.display())))?;

    for t in 0..max_frames {
        let mut buf = vec![0u8; frame_pixels];
        for (i, (name, cell)) in cells.iter().enumerate() {
            let (row, col) = (i / opts.cols, i % opts.cols);
            let idx = t.min(cell.frames.len() - 1);
            let bytes = fs::read(&cell.frames[idx])
                .map_err(|e| rt(format!("{}: {e}", cell.frames[idx].display())))?;
            let (ox, oy) = (col * tw, row * th);
            copy_tile(&mut buf, total_w, ox, oy, tw, th, &bytes);
            let source = sources
                .get(name)
                .and_then(|v| v.get(idx))
                .ok_or_else(|| rt(format!("cell {name:?}: no event record for frame {idx}")))?;
            if matches!(source.as_str(), "Clamped" | "Fallback") {
                draw_border(&mut buf, total_w, ox, oy, tw, th);
            }
        }
        draw_label(&mut buf, total_w, total_h, grid_h, n_episodes, success_rate);
        fs::write(opts.out.join(format!("{t:06}.bin")), &buf)
            .map_err(|e| rt(format!("{}: {e}", opts.out.display())))?;
    }

    let layout = serde_json::json!({
        "shape": [total_h, total_w, 3],
        "dtype": "u8",
        "layout": "row-major, little-endian, tightly packed",
        "grid": [opts.rows, opts.cols],
        "cell_shape": shape,
        "label_height": opts.label_height,
        "frames": max_frames,
    });
    fs::write(
        opts.out.join("layout.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&layout).expect("Value always serializes")
        ),
    )
    .map_err(|e| rt(format!("{}: {e}", opts.out.display())))?;

    println!(
        "wrote {max_frames} frame(s) to {} ({total_w}x{total_h}, {} cell(s))",
        opts.out.display(),
        cells.len()
    );
    println!(
        "note: the mp4 container is not part of the execution hash chain (spec 5.3); these \
         frames are the evidence -- see python/es/encode_video.py."
    );
    Ok(())
}

/// Copies one `[th, tw, 3]` tile verbatim into `buf` at `(ox, oy)`: no scaling, no filtering,
/// no colour conversion (INV-14's rule against a silent resample, applied to a byte copy).
fn copy_tile(
    buf: &mut [u8],
    total_w: usize,
    ox: usize,
    oy: usize,
    tw: usize,
    th: usize,
    src: &[u8],
) {
    let stride = tw * 3;
    for ry in 0..th {
        let d = ((oy + ry) * total_w + ox) * 3;
        let s = ry * stride;
        buf[d..d + stride].copy_from_slice(&src[s..s + stride]);
    }
}

const RED: [u8; 3] = [255, 0, 0];
const WHITE: [u8; 3] = [255, 255, 255];

fn set_pixel(buf: &mut [u8], total_w: usize, x: usize, y: usize, color: [u8; 3]) {
    let d = (y * total_w + x) * 3;
    buf[d..d + 3].copy_from_slice(&color);
}

/// A fixed-width (1px) red ring over a tile's outermost pixels; the interior is untouched.
fn draw_border(buf: &mut [u8], total_w: usize, ox: usize, oy: usize, tw: usize, th: usize) {
    for x in 0..tw {
        set_pixel(buf, total_w, ox + x, oy, RED);
        set_pixel(buf, total_w, ox + x, oy + th - 1, RED);
    }
    for y in 0..th {
        set_pixel(buf, total_w, ox, oy + y, RED);
        set_pixel(buf, total_w, ox + tw - 1, oy + y, RED);
    }
}

/// One bit per pixel, 3 wide x 5 tall, digits 0-9 -- the "embedded bitmap font table" the
/// acceptance criteria ask for: no `fontdb`, no system font, no locale, no float formatting.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b001, 0b001, 0b001],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

fn draw_digit(buf: &mut [u8], total_w: usize, total_h: usize, x: usize, y: usize, digit: usize) {
    for (ry, row) in DIGITS[digit].iter().enumerate() {
        for rx in 0..3usize {
            if row & (1 << (2 - rx)) == 0 {
                continue;
            }
            let (px, py) = (x + rx, y + ry);
            if px < total_w && py < total_h {
                set_pixel(buf, total_w, px, py, WHITE);
            }
        }
    }
}

/// Draws `value` as `digits` zero-padded glyphs (clamped to that width) and returns the x just
/// past the last one.
fn draw_number(
    buf: &mut [u8],
    total_w: usize,
    total_h: usize,
    mut x: usize,
    y: usize,
    value: u64,
    digits: u32,
) -> usize {
    let value = value.min(10u64.saturating_pow(digits) - 1);
    for place in (0..digits).rev() {
        let d = (value / 10u64.pow(place) % 10) as usize;
        draw_digit(buf, total_w, total_h, x, y, d);
        x += 4; // 3px glyph + 1px gap
    }
    x
}

/// The label strip: episode count, then success rate as a percentage, digits only (spec 12.4:
/// a rate, never a `step/s` figure).
fn draw_label(
    buf: &mut [u8],
    total_w: usize,
    total_h: usize,
    y0: usize,
    n_episodes: u64,
    success_rate: f64,
) {
    let pct = (success_rate.clamp(0.0, 1.0) * 100.0).round() as u64;
    let x = draw_number(buf, total_w, total_h, 1, y0 + 1, n_episodes, 3);
    draw_number(buf, total_w, total_h, x + 3, y0 + 1, pct, 3);
}
