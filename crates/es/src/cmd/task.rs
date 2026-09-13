//! `es task compile` (spec 11.1 Lower, spec 11.3 CPU reference plan).

use es_ir::types::ElemType;

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es task compile <task.toml> <observation.toml> [--release]

Validates the Task IR (structurally only -- the task executor that runs it lives in
es-env, not in this CLI) and compiles the Observation IR to a CPU reference execution
plan (spec 11.1 Lower), printing the node execution order, the buffer table, the total
arena size, and the compiler_hash slot of the hash chain (spec 5.3).

    --release   compile with PlanMode::Release instead of the default PlanMode::Debug
";

/// Bytes per element. The CPU plan's arena is f32-addressed regardless of a buffer's own
/// dtype (`es_compile::plan::Home::Arena` doc comment), so this is informational only.
fn elem_bytes(e: ElemType) -> usize {
    match e {
        ElemType::F32 | ElemType::I32 => 4,
        ElemType::F16 | ElemType::Bf16 => 2,
        ElemType::F64 => 8,
        ElemType::U8 | ElemType::Bool => 1,
    }
}

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("compile") => compile(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es task: unknown subcommand '{other}'"
        ))),
    }
}

fn compile(args: &[String]) -> Result<u8, CliError> {
    let mut release = false;
    let mut positional = Vec::new();
    for a in args {
        match a.as_str() {
            "--release" => release = true,
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(0);
            }
            other => positional.push(other.to_owned()),
        }
    }
    let [task_path, obs_path] = positional.as_slice() else {
        return Err(CliError::Usage(HELP.to_owned()));
    };

    let task_raw = std::fs::read_to_string(task_path)
        .map_err(|e| CliError::Runtime(format!("{task_path}: {e}")))?;
    let task = es_ir::serial::task_from_toml(&task_raw)
        .map_err(|e| CliError::Runtime(format!("{task_path}: {e}")))?;
    let mut had_error = false;
    for d in task.validate() {
        print!("{d}");
        had_error |= d.is_error();
    }

    let obs_raw = std::fs::read_to_string(obs_path)
        .map_err(|e| CliError::Runtime(format!("{obs_path}: {e}")))?;
    let observation = es_ir::serial::observation_from_toml(&obs_raw)
        .map_err(|e| CliError::Runtime(format!("{obs_path}: {e}")))?;

    let mode = if release {
        es_compile::PlanMode::Release
    } else {
        es_compile::PlanMode::Debug
    };
    let plan = es_compile::CpuPlan::compile(&observation, mode).map_err(|diags| {
        let mut msg = String::new();
        for d in &diags {
            msg.push_str(&d.to_string());
        }
        CliError::Runtime(msg)
    })?;

    println!("plan ({} mode):", if release { "release" } else { "debug" });
    println!("nodes in execution order:");
    for step in &plan.steps {
        println!(
            "  node {:>4}  {:?}  in={:?} -> out=buf{}",
            step.node.0, step.op, step.inputs, step.out.0
        );
    }
    println!();
    println!("buffers:");
    let mut total_bytes = 0usize;
    for (i, b) in plan.buffers.iter().enumerate() {
        let bytes = b.elems * elem_bytes(b.dtype);
        total_bytes += bytes;
        println!(
            "  buf{i:<4} node={:<4} {:?}  shape={:?}  home={:?}  {bytes}B",
            b.node.0, b.dtype, b.shape, b.home
        );
    }
    println!();
    println!(
        "total: {total_bytes} bytes across {} buffers, {} steps (arena: {} f32 elems)",
        plan.buffers.len(),
        plan.steps.len(),
        plan.arena_elems
    );
    println!("compiler_hash: {}", hex(&plan.compiler_hash()));
    println!();
    println!("note: the task executor lives in es-env; this command only validates the Task IR.");

    Ok(u8::from(had_error))
}
