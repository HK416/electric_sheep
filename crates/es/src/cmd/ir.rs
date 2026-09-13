//! `es ir validate` / `es ir check` (spec 11.1 Cross-IR Check, spec 5.3 hash chain).

use es_ir::cross::{self, IrBundle};
use es_ir::deployment::DeploymentIr;
use es_ir::evaluation::EvaluationIr;
use es_ir::learning::LearningGraph;
use es_ir::observation::ObservationIr;
use es_ir::serial::{
    deployment_from_toml, evaluation_from_toml, learning_from_toml, observation_from_toml,
    task_from_toml, SerialError,
};
use es_ir::task::TaskIr;
use es_ir::Diagnostic;

use crate::error::CliError;
use crate::util::hex;

const VALIDATE_HELP: &str = "\
es ir validate <file.toml>...

Detects the IR kind from each file's envelope (es_ir::serial), runs that IR's own
validate(), prints diagnostics in the spec block format, and prints the IR's own content
hash (task/observation/learning/deployment/evaluation _hash) as hex.
";

const CHECK_HELP: &str = "\
es ir check <task.toml> <obs.toml> <learning.toml> <deploy.toml> [eval.toml]

Builds an IrBundle from the five files and runs the spec 11.1 Cross-IR Check, then prints
the hash chain slots known at authoring time: task, observation, learning, policy,
deployment, evaluation (if given), and compiler. dataset/runtime/hardware are only known
at run time and always print as `unset` here.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("validate") => validate(&args[1..]),
        Some("check") => check(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("es ir <validate|check> ...\n\n{VALIDATE_HELP}\n{CHECK_HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es ir: unknown subcommand '{other}'"
        ))),
    }
}

/// One of the five typed IRs, loaded from its own envelope kind.
enum AnyIr {
    Task(TaskIr),
    Observation(ObservationIr),
    Learning(LearningGraph),
    Deployment(DeploymentIr),
    Evaluation(EvaluationIr),
}

impl AnyIr {
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Task(_) => "task",
            Self::Observation(_) => "observation",
            Self::Learning(_) => "learning",
            Self::Deployment(_) => "deployment",
            Self::Evaluation(_) => "evaluation",
        }
    }

    fn validate(&self) -> Vec<Diagnostic> {
        match self {
            Self::Task(ir) => ir.validate(),
            Self::Observation(ir) => ir.validate(),
            Self::Learning(ir) => ir.validate(),
            Self::Deployment(ir) => ir.validate(),
            Self::Evaluation(ir) => ir.validate(),
        }
    }

    fn hash(&self) -> Result<[u8; 32], Diagnostic> {
        match self {
            Self::Task(ir) => ir.task_hash(),
            Self::Observation(ir) => ir.observation_hash(),
            Self::Learning(ir) => ir.learning_hash(),
            Self::Deployment(ir) => ir.deployment_hash(),
            Self::Evaluation(ir) => ir.evaluation_hash(),
        }
    }
}

fn read_file(path: &str) -> Result<String, CliError> {
    std::fs::read_to_string(path).map_err(|e| CliError::Runtime(format!("{path}: {e}")))
}

fn load_any(path: &str) -> Result<AnyIr, CliError> {
    let raw = read_file(path)?;
    let envelope: toml::Value =
        toml::from_str(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
    let kind = envelope
        .get("kind")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| CliError::Runtime(format!("{path}: missing 'kind' in IR envelope")))?;
    Ok(match kind {
        "task" => AnyIr::Task(
            task_from_toml(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?,
        ),
        "observation" => AnyIr::Observation(
            observation_from_toml(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?,
        ),
        "learning" => AnyIr::Learning(
            learning_from_toml(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?,
        ),
        "deployment" => AnyIr::Deployment(
            deployment_from_toml(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?,
        ),
        "evaluation" => AnyIr::Evaluation(
            evaluation_from_toml(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?,
        ),
        other => {
            return Err(CliError::Runtime(format!(
                "{path}: unknown IR kind '{other}'"
            )))
        }
    })
}

fn validate(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{VALIDATE_HELP}");
        return Ok(0);
    }
    if args.is_empty() {
        return Err(CliError::Usage(VALIDATE_HELP.to_owned()));
    }
    let mut had_error = false;
    for path in args {
        println!("== {path} ==");
        let ir = match load_any(path) {
            Ok(ir) => ir,
            Err(e) => {
                println!("{e}");
                had_error = true;
                continue;
            }
        };
        for d in ir.validate() {
            print!("{d}");
            had_error |= d.is_error();
        }
        match ir.hash() {
            Ok(h) => println!("{}_hash: {}", ir.kind_name(), hex(&h)),
            Err(d) => {
                print!("{d}");
                had_error = true;
            }
        }
        println!();
    }
    Ok(u8::from(had_error))
}

fn parse_positional<T>(
    path: &str,
    parse: fn(&str) -> Result<T, SerialError>,
) -> Result<T, CliError> {
    let raw = read_file(path)?;
    parse(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))
}

fn print_hash(name: &str, h: Result<[u8; 32], Diagnostic>) {
    match h {
        Ok(h) => println!("  {name}: {}", hex(&h)),
        Err(d) => println!("  {name}: unset ({d})"),
    }
}

fn check(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{CHECK_HELP}");
        return Ok(0);
    }
    if !(4..=5).contains(&args.len()) {
        return Err(CliError::Usage(CHECK_HELP.to_owned()));
    }

    let task = parse_positional(&args[0], task_from_toml)?;
    let observation = parse_positional(&args[1], observation_from_toml)?;
    let learning = parse_positional(&args[2], learning_from_toml)?;
    let deployment = parse_positional(&args[3], deployment_from_toml)?;
    let evaluation = args
        .get(4)
        .map(|p| parse_positional(p, evaluation_from_toml))
        .transpose()?;

    let diags = cross::check(&IrBundle {
        task: &task,
        observation: &observation,
        learning: &learning,
        deployment: &deployment,
        evaluation: evaluation.as_ref(),
    });
    let mut had_error = false;
    for d in &diags {
        print!("{d}");
        had_error |= d.is_error();
    }

    println!("hash chain (spec 5.3):");
    print_hash("task", task.task_hash());
    print_hash("observation", observation.observation_hash());
    print_hash("learning", learning.learning_hash());
    print_hash("policy", learning.policy_hash());
    print_hash("deployment", deployment.deployment_hash());
    match &evaluation {
        Some(e) => print_hash("evaluation", e.evaluation_hash()),
        None => println!("  evaluation: unset"),
    }
    match es_compile::CpuPlan::compile(&observation, es_compile::PlanMode::Debug) {
        Ok(plan) => println!("  compiler: {}", hex(&plan.compiler_hash())),
        Err(_) => println!("  compiler: unset (observation IR did not compile)"),
    }
    println!("  dataset: unset");
    println!("  runtime: unset");
    println!("  hardware: unset");

    Ok(u8::from(had_error))
}
