//! The trainer subprocess: the interpreter probe, the backbone fetch, the one spawn, and
//! the trainer's stdout read as it goes. `es` finds its own scripts with [`installed`].

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use es_core::child::{retry_start, transient_start_failure};
use es_data::training::{Plan, Recipe, Route, Step};
use serde_json::Value;

use super::checkpoints::Early;
use super::lerobot::{stream_lerobot, training_bar, LerobotProgress};
use super::{bad, TrainWatch};
use crate::error::CliError;

/// Packet M12/R8: the plan's `# fetch:` line, run when `[policy] base_model` is not on disk --
/// with the run's interpreter, before anything that takes time. What it wrote is then
/// `Backbone::verify`'s to judge, exactly as a file fetched by hand always was.
pub(crate) fn fetch_base_model(plan: &Plan, recipe: &Recipe) -> Result<(), CliError> {
    let (Some(words), Some(path)) = (&plan.fetch, &recipe.policy.base_model) else {
        return Ok(());
    };
    if Path::new(path).exists() {
        return Ok(());
    }
    let command = words.join(" ");
    println!("$ {command}");
    let mut fetch = Command::new(&words[0]);
    fetch.arg(installed(&words[1])).args(&words[2..]);
    let why = match fetch.status() {
        Ok(status) if status.success() => return Ok(()),
        Ok(status) => format!("it exited with {}", status.code().unwrap_or(-1)),
        Err(e) => format!("{}: {e}", words[0]),
    };
    Err(bad(format!(
        "the pretrained backbone {path} is missing and could not be fetched ({why}).\n\
         It is downloaded once, by `{command}`: the interpreter needs torchvision and blake3 \
         (set ES_PYTHON to one that has them) and, the first time, the network. Nothing else \
         has run."
    )))
}

/// One of `es`'s own scripts (`python/es/train_ppo.py`, ...): as the plan names it when it is
/// there from the working directory (the repository root, as before), else the same path
/// under the nearest folder above this executable that holds it. An authored project runs `es`
/// in its own folder (packet M17/G9), where the repository's scripts are not.
pub(super) fn installed(script: &str) -> PathBuf {
    let rel = Path::new(script);
    if rel.is_absolute() || rel.exists() {
        return rel.to_path_buf();
    }
    let exe = std::env::current_exe().ok();
    (exe.iter().flat_map(|e| e.ancestors().skip(1)))
        .map(|dir| dir.join(rel))
        .find(|p| p.is_file())
        .unwrap_or_else(|| rel.to_path_buf())
}

/// The one interpreter probe: it is both the refusal the packet names (an interpreter that
/// cannot import what the route needs, reported in the interpreter's own words) and the
/// source of `hardware.json`'s device and library versions.
pub(super) fn probe(interpreter: &str, route: Route) -> Result<Value, CliError> {
    let script = "\
import json, sys
d = {}
try:
    import torch
    d['torch'] = torch.__version__
    d['cuda'] = torch.version.cuda
    d['device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
except Exception as e:
    sys.stderr.write('%s: %s\\n' % (type(e).__name__, e)); raise SystemExit(1)
if len(sys.argv) > 1 and sys.argv[1] == 'lerobot':
    try:
        import lerobot
        d['lerobot'] = getattr(lerobot, '__version__', 'unknown')
    except Exception as e:
        sys.stderr.write('%s: %s\\n' % (type(e).__name__, e)); raise SystemExit(1)
print(json.dumps(d))
";
    let mut cmd = Command::new(interpreter);
    cmd.args(["-c", script]);
    if route == Route::External {
        cmd.arg("lerobot");
    }
    // Many interpreters importing CUDA torch at once on Windows sometimes fail to start (packet
    // M15/R1); such a failure is retried with a fresh interpreter, which changes nothing.
    retry_start(interpreter, || {
        let out = cmd.output().map_err(|e| {
            let msg = format!("{interpreter}: {e}\nSet ES_PYTHON or [run] interpreter.");
            (msg, false)
        })?;
        let stderr = String::from_utf8_lossy(&out.stderr);
        let printed = !out.stdout.trim_ascii().is_empty();
        let transient = transient_start_failure(out.status.code(), &stderr, printed);
        if !out.status.success() {
            // The exit status too: a process the OS ends (an access violation, a failed DLL
            // load) leaves no Python traceback, and an empty message says nothing (review M14
            // N-5).
            let msg = format!(
                "{interpreter} cannot import what the {} route needs ({}):\n{}",
                route.as_str(),
                exit_words(out.status),
                stderr.trim_end()
            );
            return Err((msg, transient));
        }
        serde_json::from_slice(&out.stdout)
            .map_err(|e| (format!("{interpreter}: the probe printed {e}"), transient))
    })
    .map_err(|(msg, attempts)| match attempts {
        1 => bad(msg),
        n => bad(format!("{msg}\n(after {n} start attempts)")),
    })
}

/// An exit status in words, the Windows NTSTATUS in hex where it is one (0xC0000005 is an
/// access violation, 0xC0000135 a DLL that did not load).
fn exit_words(status: std::process::ExitStatus) -> String {
    match status.code() {
        Some(code) if code < 0 => format!("exit status 0x{:08X}", code as u32),
        Some(code) => format!("exit status {code}"),
        None => "ended by a signal".to_owned(),
    }
}

/// Runs the one subprocess. `capture` is for `train_act.py`, whose whole report is a single
/// JSON line on stdout; `lerobot-train` streams a progress log instead and inherits.
///
/// With someone listening -- a publisher, or `es loop cycle`'s preview queue -- the run is
/// read while it goes: the captured path becomes a *streamed* one, the same stdout read line
/// by line so a `{"progress": ...}` line reaches a viewer, and `lerobot-train`'s console is
/// relayed and read for its bar and metric lines (packet M12/R2). Either way each checkpoint
/// is bundled as soon as the trainer is past it (`early`, packet M13/Z1). The summary is still
/// the last line and still parsed the same way, which is what keeps `training.lock`
/// byte-identical (packet M7/E7).
pub(super) fn spawn(
    step: &Step,
    route: Route,
    batch: Option<u32>,
    watch: &mut TrainWatch<'_>,
    early: &mut Early<'_>,
) -> Result<Value, CliError> {
    let capture = route.captures_trainer_stdout();
    let mut cmd = Command::new(&step.prefix[0]);
    let script = step.prefix.get(1).map(|s| installed(s));
    cmd.args(script)
        .args(step.prefix.iter().skip(2))
        .args(&step.args);
    let extra = watch.trainer_flags(route);
    if !extra.is_empty() {
        println!("  + {}", extra.join(" "));
        cmd.args(&extra);
    }
    let named = || format!("{}: ", step.prefix[0]);
    let (ok, code, summary) = if capture && watch.listening() {
        stream(&mut cmd, watch, early).map_err(|e| bad(format!("{}{e}", named())))?
    } else if capture {
        let out = cmd.output().map_err(|e| bad(format!("{}{e}", named())))?;
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        print!("{text}");
        // Captured means captured: a trainer that failed has to be able to say why.
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        let last = text.lines().last().unwrap_or_default();
        (
            out.status.success(),
            out.status.code(),
            serde_json::from_str(last).unwrap_or(Value::Null),
        )
    } else if watch.listening() {
        let mut progress = LerobotProgress::default();
        let status = stream_lerobot(&mut cmd, |piece| {
            if let (Some(p), Some(row)) = (watch.publisher.as_deref(), progress.read(piece, batch))
            {
                p.train_row(row);
            }
            if let Some((past, _, _)) = training_bar(piece) {
                early.past(past, watch);
            }
        })
        .map_err(|e| bad(format!("{}{e}", named())))?;
        (status.success(), status.code(), Value::Null)
    } else {
        let status = cmd.status().map_err(|e| bad(format!("{}{e}", named())))?;
        (status.success(), status.code(), Value::Null)
    };
    if ok {
        Ok(summary)
    } else {
        Err(bad(format!(
            "the trainer exited with {}",
            code.unwrap_or(-1)
        )))
    }
}

/// The trainer's stdout, line by line, published as it arrives.
///
/// stderr is inherited rather than piped: reading two pipes from one thread deadlocks when
/// either fills, and the non-streamed path's `eprint!` of the captured stderr and this go to
/// the same place. stdout is read on its own thread, so the trainer never waits on a full pipe
/// while a mark is packed here. A line that is not one of the two the trainer publishes is
/// printed and remembered as a candidate summary, so the last non-progress line is the report
/// -- exactly what `Command::output`'s `text.lines().last()` picks.
fn stream(
    cmd: &mut Command,
    watch: &mut TrainWatch<'_>,
    early: &mut Early<'_>,
) -> std::io::Result<(bool, Option<i32>, Value)> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("the trainer's stdout was not piped"))?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut last = String::new();
    for line in rx {
        println!("{line}");
        let parsed = serde_json::from_str::<Value>(&line).unwrap_or(Value::Null);
        let publisher = watch.publisher.as_deref_mut();
        match (publisher, parsed.get("progress"), parsed.get("sample")) {
            (Some(p), Some(progress), _) => {
                if let Some(row) = progress_row(progress) {
                    p.train_row(row);
                }
                if let Some(row) = rl_row(progress) {
                    p.rl_row(row);
                }
            }
            (Some(p), None, Some(Value::String(path))) => p.sample(Path::new(path)),
            _ => {}
        }
        if let Some(past) = progress_step(&parsed) {
            early.past(past, watch);
        }
        if parsed.get("progress").is_none() && parsed.get("sample").is_none() {
            last = line;
        }
    }
    let status = child.wait()?;
    Ok((
        status.success(),
        status.code(),
        serde_json::from_str(&last).unwrap_or(Value::Null),
    ))
}

/// One `{"progress": {...}}` object of `train_act.py` / `train_ppo.py` as the stream-5 row
/// `[step, loss, lr, samples_per_s]` (packet M7/E7); a line without a step or a loss is none.
///
/// A `null` loss is NaN (packet P-M14-R1): JSON has no NaN, so the trainers write a non-finite
/// loss as `null`, and a diverged run must be drawn as one -- the editor's light reads a
/// non-finite loss as broken. An absent `lr` or rate is NaN, never a stand-in.
pub(super) fn progress_row(progress: &Value) -> Option<[f64; 4]> {
    let at = |k: &str| progress.get(k).and_then(Value::as_f64);
    let loss = match progress.get("loss")? {
        Value::Null => f64::NAN,
        v => v.as_f64()?,
    };
    Some([
        at("step")?,
        loss,
        at("lr").unwrap_or(f64::NAN),
        at("samples_per_s").unwrap_or(f64::NAN),
    ])
}

/// A `train_ppo.py` progress line's learning numbers as the stream-6 row `[step, return,
/// episode_len, success, entropy, envelope_violation_rate]` (packet M16/H4). A line without a
/// `return` -- every `train_act.py` line -- is none; a `null` success (no episode ended) is NaN.
pub(super) fn rl_row(progress: &Value) -> Option<[f64; 6]> {
    let at = |k: &str| progress.get(k).and_then(Value::as_f64);
    Some([
        at("step")?,
        at("return")?,
        at("episode_len").unwrap_or(f64::NAN),
        at("success").unwrap_or(f64::NAN),
        at("entropy").unwrap_or(f64::NAN),
        at("envelope_violation_rate").unwrap_or(f64::NAN),
    ])
}

/// The step `train_act.py` or `train_ppo.py` says it has finished: `{"progress": {"step": N}}`.
pub(super) fn progress_step(line: &Value) -> Option<u64> {
    line.get("progress")?.get("step")?.as_u64()
}
