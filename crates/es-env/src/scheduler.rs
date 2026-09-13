//! The batch-domain scheduler (§12.1–12.3, App. B.5).
//!
//! Pure data: a [`Schedule`] is a static per-tick plan computed from [`BatchDomains`] alone, so
//! two machines interleave the four domains identically and the plan is testable without a
//! backend. Periods are integer counts of simulation ticks — there is no `f64` in this module
//! (§18.1 forbids float time accumulation).

use serde::{Deserialize, Serialize};

use crate::EnvError;

/// Where a domain's work runs. Not a capability query — just where the caller put it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    #[default]
    Cpu,
    Gpu(u32),
}

/// One batch domain: how many items, how often, and where (§12.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainCfg {
    /// Batch size, `>= 1`.
    pub batch: u32,
    /// Tick period, in simulation ticks. The simulation domain's is always 1 — it *is* the
    /// tick. Never a duration (§18.1).
    pub period: u32,
    pub device: Device,
}

impl DomainCfg {
    pub const fn new(batch: u32, period: u32) -> Self {
        Self {
            batch,
            period,
            device: Device::Cpu,
        }
    }

    #[must_use]
    pub const fn on(self, device: Device) -> Self {
        Self { device, ..self }
    }
}

/// The four domains of §12.1. Sizes and schedules are independent by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchDomains {
    pub simulation: DomainCfg,
    pub observation: DomainCfg,
    pub inference: DomainCfg,
    pub training: Option<DomainCfg>,
}

impl BatchDomains {
    /// A single-env, everything-every-tick configuration — the smallest valid one.
    pub fn single_env() -> Self {
        Self {
            simulation: DomainCfg::new(1, 1),
            observation: DomainCfg::new(1, 1),
            inference: DomainCfg::new(1, 1),
            training: None,
        }
    }
}

/// What fires on one simulation tick. Within a tick the order is §6.4's phase order
/// (physics, then observation, then inference), which is fixed and therefore not data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickPlan {
    /// Offset within the hyper-period, `0..period()`.
    pub offset: u64,
    pub observation: bool,
    pub inference: bool,
    pub training: bool,
}

/// A static, deterministic per-tick plan over one hyper-period.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    domains: BatchDomains,
    plan: Vec<TickPlan>,
}

const fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

impl Schedule {
    /// Validates `domains` (§12.1, App. B.5) and expands one hyper-period.
    pub fn build(domains: &BatchDomains) -> Result<Self, EnvError> {
        let bad = |m: String| Err(EnvError::Schedule(m));
        let named = [
            ("simulation", domains.simulation),
            ("observation", domains.observation),
            ("inference", domains.inference),
        ];
        for (name, cfg) in named
            .into_iter()
            .chain(domains.training.map(|t| ("training", t)))
        {
            if cfg.batch == 0 {
                return bad(format!("{name} batch must be >= 1"));
            }
            if cfg.period == 0 {
                return bad(format!("{name} period must be >= 1 simulation tick"));
            }
        }
        if domains.simulation.period != 1 {
            return bad(format!(
                "simulation period must be 1: it defines the tick, got {}",
                domains.simulation.period
            ));
        }
        let (obs, inf) = (domains.observation, domains.inference);
        if inf.period % obs.period != 0 {
            return bad(format!(
                "inference period {} is not an integer multiple of the observation period {}",
                inf.period, obs.period
            ));
        }
        if let Some(train) = domains.training {
            if train.period % inf.period != 0 {
                return bad(format!(
                    "training period {} is not an integer multiple of the inference period {}",
                    train.period, inf.period
                ));
            }
        }
        if obs.batch > domains.simulation.batch {
            return bad(format!(
                "observation batch {} exceeds the simulation batch {}",
                obs.batch, domains.simulation.batch
            ));
        }
        if inf.batch > obs.batch {
            return bad(format!(
                "inference batch {} exceeds the observation batch {}",
                inf.batch, obs.batch
            ));
        }

        let period = domains
            .training
            .map_or(u64::from(inf.period), |t| {
                lcm(u64::from(inf.period), u64::from(t.period))
            })
            .max(1);
        let fires = |p: u32, t: u64| t % u64::from(p) == 0;
        let plan = (0..period)
            .map(|offset| TickPlan {
                offset,
                observation: fires(obs.period, offset),
                inference: fires(inf.period, offset),
                training: domains.training.is_some_and(|t| fires(t.period, offset)),
            })
            .collect();
        Ok(Self {
            domains: *domains,
            plan,
        })
    }

    pub fn domains(&self) -> &BatchDomains {
        &self.domains
    }

    /// Length of the hyper-period in simulation ticks.
    pub fn period(&self) -> u64 {
        self.plan.len() as u64
    }

    /// One hyper-period of the plan.
    pub fn ticks(&self) -> impl Iterator<Item = TickPlan> + '_ {
        self.plan.iter().copied()
    }

    /// The plan for an absolute simulation tick.
    pub fn at(&self, tick: u64) -> TickPlan {
        self.plan[(tick % self.period()) as usize]
    }

    /// The envs whose cameras are active on `tick`, as a half-open range (§12.2 `round_robin`,
    /// §12.3 "a deterministic function of `(seed, tick, period)`" — no seed is needed because
    /// the cycle is positional and env-ID ascending).
    ///
    /// Empty when `tick` is not an observation tick.
    pub fn observation_envs(&self, tick: u64) -> std::ops::Range<u32> {
        if !self.at(tick).observation {
            return 0..0;
        }
        let (n_sim, n_obs) = (
            self.domains.simulation.batch,
            self.domains.observation.batch,
        );
        let cycle = u64::from(n_sim.div_ceil(n_obs));
        let slot = (tick / u64::from(self.domains.observation.period)) % cycle;
        let start = (slot * u64::from(n_obs)) as u32;
        start..(start + n_obs).min(n_sim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domains(sim: (u32, u32), obs: (u32, u32), inf: (u32, u32)) -> BatchDomains {
        BatchDomains {
            simulation: DomainCfg::new(sim.0, sim.1),
            observation: DomainCfg::new(obs.0, obs.1),
            inference: DomainCfg::new(inf.0, inf.1),
            training: None,
        }
    }

    /// The §12.1 example: 4096 sim @ 1 kHz, 512 obs @ 30 Hz, 256 inf @ 10 Hz, 64 train.
    fn spec_example() -> BatchDomains {
        BatchDomains {
            training: Some(DomainCfg::new(64, 990).on(Device::Gpu(1))),
            ..domains((4096, 1), (512, 33), (256, 99))
        }
    }

    #[test]
    fn good_and_bad_configurations() {
        let cases: [(&str, BatchDomains, bool); 9] = [
            ("spec example", spec_example(), true),
            ("single env", BatchDomains::single_env(), true),
            (
                "everything every tick",
                domains((8, 1), (8, 1), (8, 1)),
                true,
            ),
            ("zero sim batch", domains((0, 1), (1, 1), (1, 1)), false),
            ("zero obs period", domains((4, 1), (4, 0), (4, 1)), false),
            ("sim period != 1", domains((4, 2), (4, 2), (4, 2)), false),
            (
                "inf period not a multiple",
                domains((4, 1), (4, 10), (4, 15)),
                false,
            ),
            (
                "obs batch > sim batch",
                domains((4, 1), (8, 1), (4, 1)),
                false,
            ),
            (
                "inf batch > obs batch",
                domains((8, 1), (4, 1), (8, 1)),
                false,
            ),
        ];
        for (name, cfg, ok) in cases {
            assert_eq!(Schedule::build(&cfg).is_ok(), ok, "{name}");
        }
        let train_bad = BatchDomains {
            training: Some(DomainCfg::new(4, 50)),
            ..domains((4, 1), (4, 10), (4, 30))
        };
        let err = Schedule::build(&train_bad).unwrap_err();
        assert!(err.to_string().contains("training period 50"), "{err}");
    }

    #[test]
    fn the_plan_is_a_pure_function_of_the_domains() {
        let s = Schedule::build(&spec_example()).unwrap();
        assert_eq!(s, Schedule::build(&spec_example()).unwrap());
        // lcm(99, 990) = 990.
        assert_eq!(s.period(), 990);
        let fired = |f: fn(&TickPlan) -> bool| s.ticks().filter(f).count();
        assert_eq!(fired(|t| t.observation), 30); // 990 / 33
        assert_eq!(fired(|t| t.inference), 10); // 990 / 99
        assert_eq!(fired(|t| t.training), 1);
        // Every inference tick is also an observation tick: inference never reads an
        // observation from an undefined tick (§12.3).
        assert!(s.ticks().all(|t| !t.inference || t.observation));
        assert_eq!(s.at(0).offset, 0);
        assert_eq!(s.at(990).offset, 0);
        assert_eq!(s.at(1023).offset, 33);
    }

    #[test]
    fn camera_round_robin_is_positional_and_covers_every_env() {
        let s = Schedule::build(&domains((8, 1), (2, 3), (2, 3))).unwrap();
        assert_eq!(s.observation_envs(0), 0..2);
        assert_eq!(s.observation_envs(1), 0..0, "not an observation tick");
        assert_eq!(s.observation_envs(3), 2..4);
        assert_eq!(s.observation_envs(9), 6..8);
        assert_eq!(s.observation_envs(12), 0..2, "wraps after ceil(8/2) slots");

        // A cycle visits every env exactly once.
        let mut seen: Vec<u32> = (0..4).flat_map(|k| s.observation_envs(k * 3)).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn a_ragged_last_slot_is_clipped_not_wrapped() {
        let s = Schedule::build(&domains((5, 1), (2, 1), (2, 1))).unwrap();
        assert_eq!(s.observation_envs(2), 4..5);
    }

    #[test]
    fn serde_round_trip() {
        let s = Schedule::build(&spec_example()).unwrap();
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Schedule>(&json).unwrap(), s);
    }
}
