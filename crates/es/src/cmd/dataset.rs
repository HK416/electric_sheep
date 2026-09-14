//! `es dataset info` / `es dataset export` (spec 19: `LeRobot` dataset, Dataset Identity).

use std::path::{Path, PathBuf};

use es_data::{export_v3, DatasetIdentity, LeRobotDataset, Split};

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es dataset info <root>
es dataset export --lerobot-v3 <root> --out <dir> [--frames <dir>]

`info` opens a LeRobot dataset at <root>, and prints its features (as the spec 5.4 PortType
they present at an Observation IR port), episode count, and the dataset identity's three
hashes (spec 19: content/schema/split) -- computed over a display-only all-train split,
since `dataset info` has no split the caller actually intends to train with.

`export --lerobot-v3` converts the v2.1 dataset at <root> into the v3.0 layout `lerobot`
0.6.1 reads, under <dir>; the source is not modified. <dir>/meta/es_provenance.json records
the source's content/schema/split hashes (spec 19.2).

--frames <dir>  Raw rendered frames to embed as PNG for each `observation.images.<name>`
                feature: <dir>/<name>/<NNNNNN>.bin, row-major u8, the feature's own
                [height, width, 3], in dataset-global frame order -- the layout
                `es_env::render::EnvRenderer` writes. Without it, a declared camera is
                dropped from the export and reported, because there are no pixels for it.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("info") => info(&args[1..]),
        Some("export") => export(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es dataset: unknown subcommand '{other}'"
        ))),
    }
}

fn export(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let (mut root, mut out, mut frames) = (None, None, None);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let slot = match arg.as_str() {
            "--lerobot-v3" => &mut root,
            "--out" => &mut out,
            "--frames" => &mut frames,
            other => {
                return Err(CliError::Usage(format!(
                    "es dataset export: unexpected argument '{other}'\n\n{HELP}"
                )))
            }
        };
        *slot = Some(PathBuf::from(rest.next().ok_or_else(|| {
            CliError::Usage(format!("es dataset export: {arg} needs a value\n\n{HELP}"))
        })?));
    }
    let (Some(root), Some(out)) = (root, out) else {
        return Err(CliError::Usage(format!(
            "es dataset export: --lerobot-v3 <root> and --out <dir> are both required\n\n{HELP}"
        )));
    };

    let dataset = LeRobotDataset::open(root).map_err(|e| CliError::Runtime(e.to_string()))?;
    let report = export_v3(&dataset, &out, frames.as_deref())
        .map_err(|e| CliError::Runtime(e.to_string()))?;
    println!("wrote: {}", out.display());
    println!("episodes: {}   frames: {}", report.episodes, report.frames);
    println!("cameras: {}", camera_list(&report.cameras));
    for dropped in &report.dropped {
        println!(
            "warning: camera {dropped:?} dropped -- no frames under --frames for it, and an \
             image feature with no pixels is a dangling reference"
        );
    }
    println!("provenance: {}", provenance(&out).display());
    Ok(0)
}

fn camera_list(cameras: &[String]) -> String {
    if cameras.is_empty() {
        "(none)".to_owned()
    } else {
        cameras.join(", ")
    }
}

fn provenance(out: &Path) -> PathBuf {
    out.join("meta").join("es_provenance.json")
}

fn info(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let [root] = args else {
        return Err(CliError::Usage(HELP.to_owned()));
    };
    let dataset =
        LeRobotDataset::open(root.as_str()).map_err(|e| CliError::Runtime(e.to_string()))?;

    println!("root: {}", dataset.root().display());
    println!("episodes: {}", dataset.episodes().len());
    println!();
    println!("features:");
    for (name, spec) in dataset.features() {
        match spec.port_type() {
            Ok(pt) => println!("  {name:<28} {pt:?}"),
            Err(e) => println!("  {name:<28} <no PortType: {e}>"),
        }
    }

    // `dataset info` has no caller-chosen split, so identity is computed over the whole
    // dataset as a single (all-train) split purely for display.
    let split = Split::deterministic(dataset.episodes().len() as u32, [1.0, 0.0, 0.0], 0);
    let identity =
        DatasetIdentity::compute(&dataset, &split).map_err(|e| CliError::Runtime(e.to_string()))?;
    println!();
    println!("dataset identity (spec 19; split shown is display-only, all-train):");
    println!("  content: {}", hex(&identity.content));
    println!("  schema:  {}", hex(&identity.schema));
    println!("  split:   {}", hex(&identity.split));
    Ok(0)
}
