//! What a finished run records: the post-run slots of `training/` and `training.lock`.

use std::path::Path;

use es_data::training::{Route, Training, TRAIN_ACT, TRAIN_PPO};
use serde_json::{json, Value};

use super::write_file;
use crate::error::CliError;
use crate::util::hex;

pub(super) fn metrics(out: &Path, summary: &Value, route: Route) -> Value {
    // The RL route's curve is per iteration and carries nine fields per row, not one loss
    // per optimizer step, so it is a different file rather than the same name holding two
    // shapes (packet M8/S4b).
    let name = match route {
        Route::Rl => "metrics/loss-curve.json",
        _ => "metrics/loss.json",
    };
    let mut curve: Option<Value> = std::fs::read_to_string(out.join(name))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    // `metrics.json` is a `training_hash` slot, so what goes in it has to be what the run
    // *computed* -- and `samples_per_sec` is a measurement of the machine, not of the run.
    // The same recipe on a faster box is the same run, and leaving a wall-clock number in
    // here would make `training_hash` unreproducible by construction, which is the one thing
    // spec 3.5 tier 1 asks of it. The number stays in `metrics/loss-curve.json` on disk,
    // where a person reads it, and the machine it describes is `hardware.json`'s (M8/S4b).
    if route == Route::Rl {
        if let Some(Value::Array(rows)) = &mut curve {
            for row in rows {
                if let Some(row) = row.as_object_mut() {
                    row.remove("samples_per_sec");
                }
            }
        }
    }
    match (curve, summary) {
        (None, Value::Null) => json!({
            "loss": {"unset": true},
            "note": "the external trainer reports its curve to its own logs, not to a file \
                     this command reads",
        }),
        (curve, summary) => json!({
            "loss": curve.unwrap_or(Value::Null),
            "summary": summary.clone(),
        }),
    }
}

pub(super) fn hardware(probe: &Value, route: Route, device: &str, interpreter: &str) -> Value {
    json!({
        "device": device,
        "device_name": probe.get("device_name").cloned(),
        "torch": probe.get("torch").cloned(),
        "cuda": probe.get("cuda").cloned(),
        "lerobot": probe.get("lerobot").cloned(),
        // No build script and no committed Cargo.lock (M5 review S-8/R7), so a build cannot
        // be named here yet; claiming a revision would be exactly the fabricated slot spec
        // 28.10 rule 2 forbids.
        "driver": {"unset": true},
        "git_describe": {"unset": true},
        "es_version": env!("CARGO_PKG_VERSION"),
        "interpreter": interpreter,
        "trainer": match route {
            Route::Ir => TRAIN_ACT,
            Route::Rl => TRAIN_PPO,
            Route::External => "lerobot-train",
        },
    })
}

/// `optimizer.json` declares the betas and eps `train_act.py`'s `AdamW` leaves at torch's
/// defaults, beside the lr and weight decay the recipe tells it to use (packet M7/T4). If
/// torch ever moves a default the declaration is stale, so the trainer reports what it built
/// and the two are compared out loud.
pub(super) fn warn_on_optimizer(training: &Training, summary: &Value) {
    let Some(reported) = summary.get("optimizer") else {
        return;
    };
    let Ok(declared) = serde_json::from_str::<Value>(training.file("optimizer.json")) else {
        return;
    };
    let same = ["lr", "betas", "eps", "weight_decay"]
        .iter()
        .all(|k| declared.get(*k) == reported.get(*k));
    if !same {
        println!(
            "warning: the trainer reports {reported} and optimizer.json declares {declared}; \
             training_hash names the declaration, so it is now stale"
        );
    }
}

pub(super) fn write_lock(
    out: &Path,
    identity: &[u8; 32],
    training_hash: Option<&[u8; 32]>,
    training: &Training,
    checkpoints: &[Value],
) -> Result<(), CliError> {
    let id = training.identity();
    let per_checkpoint: Vec<Value> = checkpoints
        .iter()
        .map(|c| {
            let digest = c["weights_blake3"]
                .as_str()
                .and_then(from_hex)
                .unwrap_or([0; 32]);
            json!({
                "step": c["step"],
                "bundle": c["bundle"],
                // Spec 19.3: policy_hash = H(training_hash, checkpoint_hash).
                "policy_hash": id.policy_hash(&digest).ok().as_ref().map(hex),
            })
        })
        .collect();
    let lock = json!({
        "schema_version": 1,
        "identity_hash": hex(identity),
        "training_hash": training_hash.map_or(json!({"unset": true}), |h| json!(hex(h))),
        "files": training.digests(),
        "checkpoints": per_checkpoint,
    });
    let mut text = lock.to_string();
    text.push('\n');
    write_file(&out.join("training.lock"), text.as_bytes())
}

fn from_hex(text: &str) -> Option<[u8; 32]> {
    let bytes: Vec<u8> = (0..text.len() / 2)
        .filter_map(|i| u8::from_str_radix(text.get(i * 2..i * 2 + 2)?, 16).ok())
        .collect();
    bytes.try_into().ok()
}
