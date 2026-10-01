//! `[rl]`, the reinforcement-learning route's table (packet M8/S4b): PPO's coefficients, the
//! three knobs whose default serialises like absence, and the `train_ppo.py` flags they
//! validate into.

use serde::{Deserialize, Serialize};

use super::{refuse, s, Recipe};
use crate::DataError;

/// `[rl]` — PPO over rollouts in our own `Env` (packet M8/S4b, spec 13.4).
///
/// **What is not here is the point** (`docs/design/rl-continuation.md` rule 1). PPO is a
/// trainer, not an IR: the value head, the state-independent `log_std`, GAE and the entropy
/// coefficient live in `python/es/train_ppo.py` and in spec 19.3's `training/`, and they move
/// `training_hash` and never `learning_hash`. The deployed graph is whatever `[policy] bundle`
/// already says it is, and this table does not touch it.
///
/// `[run] steps` is the iteration count beside this table, `[run] batch` is refused by name
/// (PPO's batch is `envs * horizon`, derived, not declared), and `[dataset]` is optional
/// because a rollout generates its own data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rl {
    /// `"ppo"`, and nothing else yet. Named rather than assumed, so a second algorithm is a
    /// value here and not a new table.
    pub algo: String,
    /// Environments stepped in lockstep — the simulation batch (spec 5.2).
    pub envs: u32,
    /// Control steps per env per iteration. One iteration collects `envs * horizon` rows.
    pub horizon: u32,
    /// Passes over each iteration's rows.
    pub epochs: u32,
    /// Minibatches per epoch; it has to divide `envs * horizon`.
    pub minibatches: u32,
    /// Discount.
    pub gamma: f64,
    /// GAE's `lambda`.
    pub lam: f64,
    /// The clipped objective's `epsilon`.
    pub clip: f64,
    /// Entropy bonus coefficient.
    pub entropy: f64,
    /// Value-loss coefficient.
    pub value_coef: f64,
    /// The Gaussian's initial `log_std`, which is training-only state and never enters a
    /// document or a bundle. Absent is the importer's own when `[init]` carries one, and
    /// `-0.5` otherwise — the trainer decides, because it is the side that can see both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_log_std: Option<f64>,
    /// Which action PPO's gradient is computed at (packet M9/R5, `docs/reviews/M9.md` S-7).
    /// Absent is `Sampled`, and spelled out it serialises like absence, so every recipe
    /// measured before the packet keeps the `training_hash` it was measured under.
    #[serde(default, skip_serializing_if = "Estimator::is_default")]
    pub estimator: Estimator,
    /// The physics backend `Rollout` steps (packet M11/X1). Absent is `mujoco-cpu`, and spelled
    /// out it serialises like absence, so no measured recipe's `training_hash` moves; any other
    /// value reaches `train_ppo.py --backend` and, through the plan and this JSON, the hash.
    #[serde(default, skip_serializing_if = "RlBackend::is_default")]
    pub backend: RlBackend,
    /// What the value network reads (packet M11/R10). Absent is `Observation`, and spelled out
    /// it serialises like absence, as `estimator` does.
    #[serde(default, skip_serializing_if = "Critic::is_default")]
    pub critic: Critic,
}

/// `[rl] critic` -- the value network's input (packet M11/R10, the asymmetric actor-critic of
/// Pinto et al., arXiv:1710.06542). The value is training-only state either way (design note
/// rule 1): it moves `training_hash` and never `learning_hash`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Critic {
    /// Every Observation IR port, flattened: every row before M11/R10, and the default.
    #[default]
    Observation,
    /// Every non-image port and `Rollout.qpos`, which carries simulator state a deployment
    /// never has (the cube's free joint). An image port never reaches it.
    Privileged,
}

impl Critic {
    /// `skip_serializing_if`: the default serialises exactly like absence.
    fn is_default(&self) -> bool {
        matches!(self, Self::Observation)
    }
}

/// `[rl] backend` -- the engines `es_native.Rollout` has a closed-loop path for (packets
/// M11/X1, M11/R1). Newton (no actuators in its adapter) is refused at parse, by the word the
/// recipe used.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RlBackend {
    /// `MuJoCo` on the CPU, the reference and the default: every row before M11/X1.
    #[default]
    #[serde(rename = "mujoco-cpu")]
    MujocoCpu,
    /// `MuJoCo` Warp on the GPU: tier 2, never bitwise against the reference.
    #[serde(rename = "mjwarp")]
    MjWarp,
    /// `PhysX` through Isaac Sim (packet M11/R1): tier 2, never bitwise against the reference.
    #[serde(rename = "physx")]
    PhysX,
}

impl RlBackend {
    /// `skip_serializing_if`: the default serialises exactly like absence.
    fn is_default(&self) -> bool {
        matches!(self, Self::MujocoCpu)
    }
}

/// `[rl] estimator` — which action PPO's log-probability is evaluated at (packet M9/R5).
///
/// Every RL row measured before this packet was clamped on every tick
/// (`executed_ne_sampled_rate = 1.00`), so the gradient was computed at the log-probability of
/// an action the environment never executed. `Executed` treats the Safety Plane as part of the
/// environment instead. Nothing about the plane, the envelope or what is *recorded* moves —
/// the choice is the estimator's alone, and it is hashed because it is a property of the run.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Estimator {
    /// The Gaussian's own sample: every row before M9/R5, and the default (spec 13.4).
    #[default]
    Sampled,
    /// What `Rollout::act` returned — the action the plane let through to the actuator.
    Executed,
}

impl Estimator {
    /// `skip_serializing_if`: the default has to serialise exactly like absence, or
    /// `estimator = "sampled"` written out would move a `training_hash` nothing else moved.
    fn is_default(&self) -> bool {
        matches!(self, Self::Sampled)
    }
}

impl Recipe {
    /// The `[rl]` table validated, as `train_ppo.py`'s flags (packet M8/S4b).
    ///
    /// Empty when there is no `[rl]`, which is what keeps every pre-S4b plan and its golden
    /// byte-identical. Every bound here is a refusal rather than a clamp: a coefficient
    /// silently moved is a run nobody can reproduce from its own recipe.
    pub fn rl_args(&self) -> Result<Vec<String>, DataError> {
        let Some(rl) = &self.rl else {
            return Ok(Vec::new());
        };
        if rl.algo != "ppo" {
            return Err(refuse(format!(
                "[rl] `algo` is {:?}; this packet implements \"ppo\"",
                rl.algo
            )));
        }
        for (field, value) in [("envs", rl.envs), ("horizon", rl.horizon)] {
            if value == 0 {
                return Err(refuse(format!("[rl] `{field}` is 0")));
            }
        }
        for (field, value) in [("epochs", rl.epochs), ("minibatches", rl.minibatches)] {
            if value == 0 {
                return Err(refuse(format!("[rl] `{field}` is 0")));
            }
        }
        let rows = u64::from(rl.envs) * u64::from(rl.horizon);
        if rows % u64::from(rl.minibatches) != 0 {
            return Err(refuse(format!(
                "[rl] `minibatches` is {} and an iteration collects {rows} rows \
                 (`envs` {} * `horizon` {}); it has to divide them, because a trailing short \
                 minibatch would weight the last rows of every epoch differently",
                rl.minibatches, rl.envs, rl.horizon
            )));
        }
        for (field, value) in [("gamma", rl.gamma), ("lam", rl.lam)] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(refuse(format!(
                    "[rl] `{field}` is {value}; it is in [0, 1]"
                )));
            }
        }
        if !rl.clip.is_finite() || rl.clip <= 0.0 {
            return Err(refuse(format!(
                "[rl] `clip` is {}; the clipped objective's epsilon is positive",
                rl.clip
            )));
        }
        for (field, value) in [("entropy", rl.entropy), ("value_coef", rl.value_coef)] {
            if !value.is_finite() || value < 0.0 {
                return Err(refuse(format!("[rl] `{field}` is {value}")));
            }
        }
        if let Some(log_std) = rl.init_log_std {
            if !log_std.is_finite() {
                return Err(refuse(format!("[rl] `init_log_std` is {log_std}")));
            }
        }
        let mut args = vec![
            s("--envs"),
            rl.envs.to_string(),
            s("--horizon"),
            rl.horizon.to_string(),
            s("--epochs"),
            rl.epochs.to_string(),
            s("--minibatches"),
            rl.minibatches.to_string(),
            s("--gamma"),
            rl.gamma.to_string(),
            s("--lam"),
            rl.lam.to_string(),
            s("--clip"),
            rl.clip.to_string(),
            s("--entropy"),
            rl.entropy.to_string(),
            s("--value-coef"),
            rl.value_coef.to_string(),
        ];
        // Only when the recipe names one: absent means the trainer decides between the
        // importer's own `log_std` and the constant, and a flag carrying a default would
        // take that decision away from the side that can see both.
        if let Some(log_std) = rl.init_log_std {
            args.push(s("--init-log-std"));
            args.push(log_std.to_string());
        }
        // Last in the argv and only when the recipe asks for it (packet M9/R5): the default
        // is what every committed row was measured with, and a flag carrying it would move
        // three plan goldens and their `training_hash`es for a value nobody changed.
        match rl.estimator {
            Estimator::Sampled => {}
            Estimator::Executed => {
                args.push(s("--estimator"));
                args.push(s("executed"));
            }
        }
        // Last, and only off the default, for the same reason (packet M11/X1).
        match rl.backend {
            RlBackend::MujocoCpu => {}
            RlBackend::MjWarp => {
                args.push(s("--backend"));
                args.push(s("mjwarp"));
            }
            RlBackend::PhysX => {
                args.push(s("--backend"));
                args.push(s("physx"));
            }
        }
        // Last, and only off the default, for the same reason (packet M11/R10).
        if rl.critic == Critic::Privileged {
            args.push(s("--critic"));
            args.push(s("privileged"));
        }
        Ok(args)
    }
}
