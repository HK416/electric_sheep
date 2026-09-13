//! `es dataset info` (spec 19: `LeRobot` dataset, Dataset Identity).

use es_data::{DatasetIdentity, LeRobotDataset, Split};

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es dataset info <root>

Opens a LeRobot dataset at <root>, and prints its features (as the spec 5.4 PortType they
present at an Observation IR port), episode count, and the dataset identity's three
hashes (spec 19: content/schema/split) -- computed over a display-only all-train split,
since `dataset info` has no split the caller actually intends to train with.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("info") => info(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es dataset: unknown subcommand '{other}'"
        ))),
    }
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
