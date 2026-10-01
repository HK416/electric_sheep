//! Spec 19.3's `training/`: the twelve files a run is identified by, held in memory as
//! canonical JSON, hashed into `identity_hash` and `training_hash`, and written out -- and the
//! augmentation chain's JSON, the one shape `augmentation.json` and a baked manifest share.

use std::collections::BTreeMap;
use std::path::Path;

use es_compile::plan::{augmentation_chains, AugmentStep};
use es_ir::observation::{AugmentKind, ObservationIr};
use es_ir::DatasetHash;
use serde_json::{json, Value};

use super::{refuse, s, Backbone, Plan, Recipe, Route, FILES, INIT_LOCK, UNSET};
use crate::collect::hex;
use crate::identity::{BaseModel, TrainingIdentity};
use crate::DataError;

// --- spec 19.3's `training/` -------------------------------------------------------------------

/// Compact, key-sorted, `float_roundtrip` JSON with a trailing newline.
///
/// `serde_json::Map` is a `BTreeMap` in this workspace (no `preserve_order` feature), so key
/// order is a property of the names and not of the code that built the value — which is what
/// makes the digest of one of these files reproducible (spec 3.4).
pub fn canon_json(value: &Value) -> String {
    let mut text = value.to_string();
    text.push('\n');
    text
}

/// What the dataset says about itself, read once by the shell and passed in here.
#[derive(Clone, Debug)]
pub struct DatasetFacts {
    pub hashes: DatasetHash,
    pub episodes: u32,
    pub frames: u64,
    /// The `es:task:<hex>` name `es loop collect` records, when there is one.
    pub recorded_task: Option<String>,
    /// Whether the split came from `es loop distill`'s `split.json` or is the all-train one.
    pub split_source: &'static str,
}

/// The spec 19.3 `training/` bundle held in memory: twelve files, each canonical JSON.
#[derive(Clone, Debug)]
pub struct Training {
    files: BTreeMap<String, String>,
    dataset: DatasetHash,
    base_source: String,
    base_license: String,
}

impl Training {
    /// The nine pre-run slots, from the recipe and the dataset alone. The three post-run ones
    /// are [`UNSET`] until [`Training::finish`].
    ///
    /// `observation` is the policy's Observation IR when the run has one on disk — the source
    /// of `augmentation.json` and of `seed.json`'s augmentation slot (packet M7/T6). `None`
    /// is the honest answer for a caller that has not opened it, and it writes the same two
    /// slots this command wrote before the packet.
    pub fn pre_run(
        recipe: &Recipe,
        plan: &Plan,
        out: &Path,
        interpreter: &str,
        data: &DatasetFacts,
        backbone: Option<&Backbone>,
        observation: Option<&ObservationIr>,
    ) -> Result<Self, DataError> {
        let run = &recipe.run;
        let route = plan.route;
        // The plan is stored `<out>`-relative, exactly as `--dry-run` prints it: config.json
        // must not carry the scratch directory, or the same recipe would have two identities.
        let rendered = plan.render(out);
        let plan_lines: Vec<&str> = rendered.lines().skip(1).collect();
        let recipe_json = serde_json::to_value(recipe)
            .map_err(|e| refuse(format!("the recipe does not serialise: {e}")))?;

        // The betas and weight decay `train_act.py`'s `AdamW(params, lr=lr)` leaves at
        // torch's documented defaults: declared here, and compared against what the trainer
        // reports when the run ends. `lerobot`'s own optimizer block is not this side's to
        // declare, so it stays unset (T4 owns the optimizer and the schedule).
        let optimizer = match route {
            // `train_ppo.py` builds `Adam`, not `AdamW`: PPO's reference implementations use
            // it, and decoupled weight decay on a policy that is already entropy-regularised
            // is a second regulariser nobody asked for. Its decay is therefore torch's
            // `Adam` default -- zero -- unless the recipe names one (packet M8/S4b).
            Route::Rl => {
                let mut optimizer = json!({
                    "kind": "Adam", "lr": run.lr, "betas": [0.9, 0.999], "eps": 1e-8,
                    "weight_decay": run.weight_decay.unwrap_or(0.0),
                    "declared_by": "train_ppo.py",
                });
                if let Some(clip) = run.grad_clip {
                    optimizer["grad_clip"] = json!(clip);
                }
                optimizer
            }
            Route::Ir => {
                let mut optimizer = json!({
                    "kind": "AdamW", "lr": run.lr, "betas": [0.9, 0.999], "eps": 1e-8,
                    // Torch's own default until packet M7/T4, and now the number the trainer
                    // is *told* to use -- the same value, declared instead of assumed.
                    "weight_decay": run.weight_decay.unwrap_or(0.01),
                    "declared_by": "train_act.py",
                });
                if let Some(clip) = run.grad_clip {
                    // Only when there is one: no clip is the absence of a clip, and a key
                    // that appeared unconditionally would move every pre-T4 identity_hash.
                    optimizer["grad_clip"] = json!(clip);
                }
                optimizer
            }
            Route::External => json!({
                "kind": "AdamW", "lr": run.lr, "betas": {"unset": true},
                "weight_decay": {"unset": true}, "declared_by": "lerobot-train",
            }),
        };
        let base_model = match route {
            // Packet M7/T5: a *verified* provenance. `es train` hashed the file, agreed with
            // the lock beside it and with the pin, and what goes into the slot is what the
            // lock said about the weights -- not about the machine that fetched them. A PPO
            // run on a `VisionEncoder { pretrained = true }` loads it too (packet M11/R10);
            // one without starts from `[init]` or the lowering's own draw.
            Route::Ir | Route::Rl => match backbone {
                Some(lock) => serde_json::to_value(lock)
                    .map_err(|e| refuse(format!("base_model.lock does not serialise: {e}")))?,
                None => json!({"source": "none"}),
            },
            // A *declared* provenance: `lerobot`'s ACT builds `vision_backbone` with these
            // weights unless `extra` overrides it (docs/api-notes/lerobot-config.md). The
            // weights' own digest and licence are packet T5's; claiming them here would be
            // the fabrication spec 28.10 rule 2 forbids.
            Route::External => {
                let extra = recipe.policy.lerobot.as_ref().map(|l| &l.extra);
                let pick = |flag: &str, default: &str| -> String {
                    extra
                        .into_iter()
                        .flatten()
                        .find_map(|a| a.strip_prefix(flag).map(s))
                        .unwrap_or_else(|| s(default))
                };
                json!({
                    "source": "lerobot:torchvision",
                    "vision_backbone": pick("--policy.vision_backbone=", "resnet18"),
                    "pretrained_backbone_weights": pick(
                        "--policy.pretrained_backbone_weights=",
                        "ResNet18_Weights.IMAGENET1K_V1",
                    ),
                    "license": {"unset": true},
                    "weights_hash": {"unset": true},
                })
            }
        };
        let base_source = base_model["source"].as_str().unwrap_or("none").to_owned();

        let mut files = BTreeMap::new();
        let mut put = |name: &str, v: Value| {
            files.insert(s(name), canon_json(&v));
        };
        put(
            "config.json",
            json!({
                "schema_version": 1, "route": route.as_str(), "recipe": recipe_json,
                "interpreter": interpreter, "plan": plan_lines,
            }),
        );
        put("optimizer.json", optimizer);
        // `total_steps` is part of the schedule and not decoration: the cosine's period is
        // the length of the run, so two runs of one `warmup`/`lr_min` pair at different
        // `steps` are two schedules (packet M7/T4).
        put(
            "scheduler.json",
            match run.schedule.as_ref().filter(|s| s.kind == "warmup_cosine") {
                None => json!({"kind": "constant", "lr": run.lr}),
                Some(schedule) => json!({
                    "kind": "warmup_cosine", "lr": run.lr, "lr_min": schedule.lr_min,
                    "warmup": schedule.warmup, "total_steps": run.steps,
                }),
            },
        );
        // Packet M7/T6: real when the document declares a chain, and `{"unset": true}` when it
        // does not -- a run with nothing to randomise has no augmentation seed, and inventing
        // one would move every measured run's `training_hash` for a number nothing read.
        let chains = match observation {
            Some(obs) => augmentation_chains(obs).map_err(|d| {
                refuse(
                    d.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            })?,
            None => BTreeMap::new(),
        };
        let augmentation_seed = run.augmentation_seed.unwrap_or(run.seed);
        put(
            "seed.json",
            json!({
                "global": run.seed, "dataloader": run.seed,
                "augmentation": if chains.is_empty() {
                    json!({"unset": true})
                } else {
                    json!(augmentation_seed)
                },
            }),
        );
        // `{"unset": true}`, and hashed as such, for a run whose data is the rollout it
        // generates (spec 28.10 rule 2: real or unset, never fabricated). A zero digest here
        // would claim a dataset of thirty-two zero bytes, and a synthesised one would claim a
        // dataset that does not exist (packet M8/S4b).
        match &recipe.dataset {
            Some(dataset) if route != Route::Rl => put(
                "dataset.lock",
                json!({
                    "root": dataset.root, "episodes": data.episodes, "frames": data.frames,
                    "content": hex(&data.hashes.content), "schema": hex(&data.hashes.schema),
                    "split": hex(&data.hashes.split), "split_source": data.split_source,
                    "recorded_task": data.recorded_task,
                }),
            ),
            // `canon_json` of this is [`UNSET`] byte for byte, which is what makes the slot
            // the same "this run does not know" every other unset slot is.
            _ => put("dataset.lock", json!({"unset": true})),
        }
        put("base_model.lock", base_model);
        // The slot the trainer is *given* (`--augmentation`), not a description of one: the
        // chains here are the nodes `python/es/augment.py` applies, and the seed is the one it
        // keys its counter RNG with. `{"kind": "none"}` stays the whole file for a document
        // that declares no augmentation, so an unaugmented run's digest is the digest it had.
        put(
            "augmentation.json",
            if chains.is_empty() {
                json!({"kind": "none"})
            } else {
                let observation_hash = observation
                    .expect("a chain came from a document")
                    .observation_hash()
                    .map_err(|e| refuse(format!("the Observation IR does not hash: {e}")))?;
                json!({
                    "kind": "observation-ir",
                    "observation_hash": hex(&observation_hash),
                    "seed": augmentation_seed,
                    "chains": chains.iter()
                        .map(|(port, chain)| (port.clone(), augmentation_json(chain)))
                        .collect::<serde_json::Map<_, _>>(),
                })
            },
        );
        put(
            "precision.json",
            json!({
                "dtype": "fp32", "amp": "off",
                "gradient_accumulation": match route {
                    Route::Ir => run.batch.unwrap_or_default(),
                    // One optimizer step per minibatch, nothing accumulated across them.
                    Route::External | Route::Rl => 1,
                },
            }),
        );
        put("topology.json", json!({"world_size": 1}));
        for name in ["checkpoint.manifest", "metrics.json", "hardware.json"] {
            files.insert(s(name), s(UNSET));
        }
        Ok(Self {
            files,
            dataset: data.hashes,
            base_source,
            // The licence slot of `TrainingIdentity`, real at last on the IR route: spec 19.3
            // makes the backbone's licence part of the run's identity, so a run that changed
            // nothing but the licence of its base model is a different run.
            base_license: backbone.map_or_else(|| s("unset"), |b| b.license.clone()),
        })
    }

    /// The thirteenth slot (packet M8/S1): `training/init.lock`, and its digest inside
    /// `config.json`.
    ///
    /// **Why the digest lives in `config.json`.** Spec 19.3 names twelve files and
    /// `TrainingIdentity` has twelve fields; a thirteenth field would be a change to
    /// `es_data::identity`, which this packet does not own. `config.json` is the "this is the
    /// run as configured" slot, and a run that starts from a policy is configured by that
    /// lock as much as by its recipe — so the lock's digest goes in there, the way
    /// `TrainingIdentity.base_model.hash` is the digest of `base_model.lock` rather than the
    /// file's contents. `identity_hash`, `training_hash` and §19.3's
    /// `policy_hash = H(training_hash, checkpoint_hash)` all follow from it, and a recipe
    /// without `[init]` never calls this, so its `config.json` is byte-for-byte the one it
    /// always was.
    ///
    /// Called before [`Training::hash`] is read for the first time, i.e. while the identity
    /// is still the pre-run one: what a run starts from is known before it starts.
    pub fn set_init(&mut self, lock: &Value) -> Result<(), DataError> {
        let text = canon_json(lock);
        let digest = hex(blake3::hash(text.as_bytes()).as_bytes());
        let mut config: Value = serde_json::from_str(self.file("config.json"))
            .map_err(|e| refuse(format!("config.json does not parse: {e}")))?;
        config["init"] = json!(digest);
        self.files.insert(s("config.json"), canon_json(&config));
        self.files.insert(s(INIT_LOCK), text);
        Ok(())
    }

    /// The three post-run slots, from what the run actually produced.
    pub fn finish(&mut self, checkpoints: &Value, metrics: &Value, hardware: &Value) {
        self.files
            .insert(s("checkpoint.manifest"), canon_json(checkpoints));
        self.files.insert(s("metrics.json"), canon_json(metrics));
        self.files.insert(s("hardware.json"), canon_json(hardware));
    }

    pub fn file(&self, name: &str) -> &str {
        self.files.get(name).map_or(UNSET, String::as_str)
    }

    fn digest(&self, name: &str) -> [u8; 32] {
        *blake3::hash(self.file(name).as_bytes()).as_bytes()
    }

    /// Every file's digest, for `training.lock` — the twelve, and `init.lock` when the run
    /// has one (packet M8/S1).
    pub fn digests(&self) -> BTreeMap<String, String> {
        FILES
            .iter()
            .copied()
            .chain(self.files.contains_key(INIT_LOCK).then_some(INIT_LOCK))
            .map(|n| (s(n), hex(&self.digest(n))))
            .collect()
    }

    /// Spec 19.3's twelve slots, each the blake3 of the file that holds it.
    ///
    /// `base_model` carries the provenance strings and the digest of `base_model.lock`
    /// itself; the backbone weights' own blake3 is inside that file — real on the IR route
    /// when the recipe names one (packet M7/T5), still a *declared* string on the external
    /// route, where nothing on this side downloaded or verified `lerobot`'s backbone.
    pub fn identity(&self) -> TrainingIdentity {
        TrainingIdentity {
            config: self.digest("config.json"),
            optimizer: self.digest("optimizer.json"),
            scheduler: self.digest("scheduler.json"),
            seed: self.digest("seed.json"),
            dataset: self.dataset,
            base_model: BaseModel {
                source: self.base_source.clone(),
                hash: self.digest("base_model.lock"),
                license: self.base_license.clone(),
            },
            augmentation: self.digest("augmentation.json"),
            precision: self.digest("precision.json"),
            topology: self.digest("topology.json"),
            checkpoint_manifest: self.digest("checkpoint.manifest"),
            metrics: self.digest("metrics.json"),
            hardware: self.digest("hardware.json"),
        }
    }

    pub fn hash(&self) -> Result<[u8; 32], DataError> {
        self.identity().training_hash()
    }

    /// Writes the twelve files under `<dir>`, and `init.lock` beside them when there is one.
    pub fn write(&self, dir: &Path) -> Result<(), DataError> {
        for name in FILES
            .iter()
            .copied()
            .chain(self.files.contains_key(INIT_LOCK).then_some(INIT_LOCK))
        {
            crate::write_file(&dir.join(name), self.file(name).as_bytes())?;
        }
        Ok(())
    }
}

/// One `training_only` chain as JSON — the list `training/augmentation.json` carries and the
/// list `es dataset bake --for-training` writes into its manifest, written once here so the
/// trainer reads one shape from either file (packet M7/T6).
///
/// `node` is the Observation IR node id, and it is not decoration: it is the `node_index`
/// coordinate of the augmentation RNG's key, so two ports' chains draw different streams at
/// the same sample and step (`python/es/augment.py`, design note section 13).
pub fn augmentation_json(chain: &[AugmentStep]) -> Value {
    Value::Array(
        chain
            .iter()
            .map(|step| {
                let mut entry = match step.kind {
                    AugmentKind::RandomCrop { width, height } => {
                        json!({"kind": "RandomCrop", "width": width, "height": height})
                    }
                    AugmentKind::ColorJitter {
                        brightness,
                        contrast,
                        saturation,
                        hue,
                    } => json!({
                        "kind": "ColorJitter", "brightness": brightness, "contrast": contrast,
                        "saturation": saturation, "hue": hue,
                    }),
                    AugmentKind::RandomErasing { probability } => {
                        json!({"kind": "RandomErasing", "probability": probability})
                    }
                    AugmentKind::GaussianNoise { sigma } => {
                        json!({"kind": "GaussianNoise", "sigma": sigma})
                    }
                };
                entry["node"] = json!(step.node.0);
                entry
            })
            .collect(),
    )
}
