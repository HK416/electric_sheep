# M5 V18 — the envelope under the declared cadence: an ablation, not a decision

Design note: `docs/design/visible-learning.md` **section 7.26**, and 7.25 (V17, the chunk that
predicted the release is executed), open question 25 (this packet), open question 22 (the
release is not observable, V16/V17).
Spec: §9.3, §9.4, §10.3, §1.4. Depends on V17.

## the question

V17 made both paths execute a chunk's rows 0..9 in order at the declared 5 Hz re-plan.
Under that cadence the command stream reaching `SafetyPlane` asks for 0.02 – 0.03 rad of
travel per control tick while `acceleration_max = 20 rad/s²` allows 0.008 rad of change in the
per-tick step, so `envelope_violation_rate` is 0.998 — `violation.acceleration` on ~90 % of
ticks, `violation.velocity` on roughly half — and the `EnvelopeViolationRate` watchdog
(`max_frac 0.9`, window 200) latches `hold_position` for about a tenth of every run. Open
question 25 named three ends that could move; **(a)** — widen `acceleration_max` to what the
STS3215 servo can actually deliver — is a Deployment IR decision the owner makes, and the owner
wants data before making it. This packet produces that data. It changes no checked-in fixture
and retrains nothing.

## the single variable

`body.safety.acceleration_max` (all six joints) in *ablation copies* of
`tests/fixtures/visible-learning/deployment.toml`, four levels: **20** (the current value, the
baseline that must reproduce V17 bit for bit), **40**, **80**, and **80 with `velocity_max`
also widened 3.0 → 4.4 rad/s** (the servo's unloaded rating per the fixture's own comment), so
the fourth level shows whether velocity binds once acceleration is out of the way. Everything
else is V17's own re-measurement, unchanged: V15's `model-40000.safetensors` (the bytes pinned
in `~/artifacts/plan-v/v17/weights.sha256`), the current Task/Observation/Learning documents,
the 1,800-step budget, nominal training seeds 1–16 and held-out seeds 101–116, one run per
level (V17's bit-identical repeat is not needed here). The `EnvelopeViolationRate` watchdog is
kept exactly as declared — it is part of what is being measured, never disabled (INV-12); the
only legal move is widening a limit in an ablation copy, never touching the checked-in file or
bypassing the plane.

## how a level is built

There is no CLI path that packs a policy against an arbitrary Deployment IR file —
`es policy pack` carries the input bundle's own deployment across unchanged
(`crates/es/src/cmd/policy.rs`, `pack()`), and `write_demo_bundle` (`crates/es/tests/cli.rs`)
always reads
`tests/fixtures/visible-learning/deployment.toml` from disk. V17 hit the same wall moving the
checked-in file's `inference_budget`, and rebuilt its untrained bundle from the (then-updated)
checked-in documents. This packet cannot update the checked-in file, so each level: (1) copies
the checked-in `deployment.toml` aside, (2) patches the six-entry `acceleration_max` array (and
`velocity_max` for the fourth level) in place with a small textual patch
(`target/plan-v/v18/scripts/patch_deployment.py` — a targeted regex over the named bracketed
array, nothing else in the document moves), (3) reruns
`dataset_bake_writes_safetensors_and_a_manifest` (the same oracle V17's `pb.sh`/`pc.sh` use to
produce an untrained bundle from the current documents) to get a bundle carrying the ablated
deployment, (4) restores the checked-in file immediately, before evaluation, and (5) runs
`es policy pack --policy <untrained>.esb --weights model-40000.safetensors --out <trained>.esb`,
which carries the ablated deployment across because that is what the input bundle now holds.
This needed no Rust change: `es-ir::deployment`'s validator has no rule tying
`acceleration_max` or `velocity_max` to a robot capability bound (`codes::DEP_031` is declared
but not implemented — `crates/es-ir/src/deployment.rs:27-28` — that belongs to a cross-IR pass
that does not exist yet), so every level validates and packs on the first try.

## scope

* **`context`** — `docs/design/visible-learning*.md` section 7.26 and open question 25,
  `docs/packets/M5/V18-envelope-ablation*.md`, `target/plan-v/v18/` (untracked: patch script,
  run script, latch-overlap script, mirrored reports and tables), server-side scripts under
  `~/artifacts/plan-v/v18/`.
* **`forbidden`** — every IR schema; `crates/es-safety`; `crates/es-ir`;
  `tests/fixtures/visible-learning/deployment.toml` itself (the checked-in file); the
  acceptance threshold (0.5, unchanged); retraining; V15's weights; sections 7.19 – 7.25;
  `docs/ARCHITECTURE*.md`; a six-suite perturbation sweep or showcase video for any level —
  promoting a level past this ablation is the fixture decision the packet exists to inform, not
  to make.
* **`INV-17`** — no new trait, no new extension point, and (per "how a level is built" above)
  no new Rust at all: the ablation is two Python scripts and four invocations of commands that
  already exist.
* **`INV-12`** — nothing is disabled. Every tick still goes buffer → `plane_chunk` →
  `SafetyPlane::validate` → `ctrl`, and the watchdog whose 0.998 latch rate is being explained
  is kept on at every level, including the ones where it barely fires.

## oracles

1. **The baseline reproduces V17 bit for bit.** `es policy pack`'s printed `weights_hash`,
   `lowering_hash`, `task_hash`, `observation_hash`, `learning_hash`, `policy_hash` at level 20
   equal V17's own (`~/artifacts/plan-v/v17/pb.log`, `pc.log`); both suites' `evaluation_hash`
   in `report.json` equal V17's; every scalar and the full `failure_mode_histogram` in
   `report.json` equal V17's stored `report.json`
   (`~/artifacts/plan-v/v17/e40000-{train,holdout-a}/report.json`).
2. **Every level's Deployment IR validates and packs.** `es policy pack`'s exit code is 0 for
   all four levels; a failure here would force the smallest Rust fix plus a test (not needed —
   see "how a level is built").
3. **The ablation is reproducible from the checked-in repo plus the pinned weights** — see
   "reproducing this" below; a reader can rerun every level from the commands listed there.
4. **`cargo xtask ci`** green — fmt, clippy `-D warnings`, context budget, layering, spec-refs,
   goldens. No golden file, no IR schema, no fixture moves.

## acceptance

The baseline's hashes and numbers match V17's exactly (deliverable 1). The four-level table,
the release-window/watchdog-latch overlap check on the baseline, and the design note section
and packet document exist with the measured numbers (deliverables 1–4). If any level's held-out
`success_rate` reaches 0.5, that is reported prominently and the packet stops there — no
six-suite sweep, no showcase video, no second variable; the fixture decision stays the owner's.
`cargo xtask ci` is green. `deployment.toml` is byte-identical to what V17 left it.

## reproducing this

On the oracle server (RTX 4090), from a tree shipped with `git archive HEAD | gzip | ssh ...`
(never `git clone`), with `ES_PYTHON` set to a Python that has `torch` and `mujoco`:

```
cargo build --release -p es --features render

# baseline (level 20): no patch, rebuild from the checked-in documents as-is
cargo test --release -p es --features render --test cli \
  dataset_bake_writes_safetensors_and_a_manifest -- --nocapture
cp /tmp/es-cli-test-dataset-bake-*/policy.esb untrained-L20.esb
es policy pack --policy untrained-L20.esb --weights model-40000.safetensors \
  --out trained-L20.esb

# a widened level (40, 80, or 80+vel): patch a scratch copy, rebuild, restore, pack
cp tests/fixtures/visible-learning/deployment.toml /tmp/deployment.orig.toml
python3 patch_deployment.py tests/fixtures/visible-learning/deployment.toml \
  acceleration_max=40.0            # or 80.0; add velocity_max=4.4 for the fourth level
cargo test --release -p es --features render --test cli \
  dataset_bake_writes_safetensors_and_a_manifest -- --nocapture
cp /tmp/es-cli-test-dataset-bake-*/policy.esb untrained-L40.esb
cp /tmp/deployment.orig.toml tests/fixtures/visible-learning/deployment.toml   # restore first
es policy pack --policy untrained-L40.esb --weights model-40000.safetensors \
  --out trained-L40.esb

# evaluate both suites per level
es eval run --config eval-trainseeds.toml --policy trained-L40.esb \
  --scene tests/fixtures/mjcf/so101_pick_place.xml --out eL40-train --frames fL40-train
es eval run --config eval-holdout.toml --policy trained-L40.esb \
  --scene tests/fixtures/mjcf/so101_pick_place.xml --out eL40-holdout --frames fL40-holdout
```

`eval-trainseeds.toml` and `eval-holdout.toml` are V15's own, unchanged
(`~/artifacts/plan-v/v15/`); `patch_deployment.py` is
`target/plan-v/v18/scripts/patch_deployment.py`, a
self-contained textual patch over one or two named arrays. The exact scripts run on the server
are `target/plan-v/v18/scripts/{patch_deployment.py,run-level.sh,latch_overlap.py}`.

## measured

Oracle server (RTX 4090), `~/venvs/es-lerobot-cuda` for `es` (torch 2.11.0+cu129, per this
packet's own server instructions — see design note section 7.26 for why that alone moves
`execution_hash` and nothing else), 2026-09-16, tree `~/Projects/es-v18`. Artifacts under
`~/artifacts/plan-v/v18/`, small results mirrored to `target/plan-v/v18/`
(`probe-all.txt`, `latch-overlap.txt`, `reports/report-<level>-<suite>.json`,
`deployment-<level>.toml`, `scripts/`). The full tables, the hash citations and the verdict are
in design note section 7.26; the two headline answers are:

1. **`acceleration_max = 40` alone already clears the stop rule.** Held-out `success_rate`
   reaches **0.625** at the first level past the baseline (`success_rate` 0.0625 at the
   baseline) and stays there at 80, while `envelope_violation_rate` keeps falling — 0.998 → 
   0.90–0.92 → 0.56–0.64 — and the watchdog-latched fraction of the run falls from about a
   tenth to well under 1 %. No level here is taken to a six-suite sweep or a showcase video:
   the stop rule fires on reaching the threshold precisely so that this ablation stays data,
   not a fixture change.
2. **Widening `velocity_max` alongside `acceleration_max` is measurably worse, and the
   histogram names the mechanism.** At `velocity_max = 4.4` the per-tick allowance
   `velocity_max * dt = 0.088` rad crosses `action_rate.first_diff_max = 0.08` rad — a bound
   the fixture's own comment says never fires at `velocity_max <= 3.0` — and a new violation
   kind, `violation.rate_limit`, appears at no other level. The combined level scores worse on
   both suites than `acceleration_max = 80` alone. The owner's option (a) is narrower than
   either default going in: move `acceleration_max`, not `velocity_max`, and 40 already does
   the job that matters.

Held-out `success_rate` at levels 40 and 80 is **0.625**, above the unchanged acceptance
threshold of 0.5 — reported prominently per the stop rule. The checked-in `deployment.toml`
is unchanged; the recommendation and its physical cost (the rotor-torque margin the fixture's
comment derives at 20 rad/s² is mostly consumed at 80) are the owner's to weigh, in design note
section 7.26.
