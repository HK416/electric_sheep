//! The batch-domain scheduler (§12.1–12.3, App. B.5).
//!
//! Pure data: a [`Schedule`] is a static per-tick plan computed from [`BatchDomains`] alone, so
//! two machines interleave the four domains identically and the plan is testable without a
//! backend. Periods are integer counts of simulation ticks — there is no `f64` in this module
//! (§18.1 forbids float time accumulation).

use es_core::TickRate;
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
    ///
    /// One control step is then one simulation tick. That is what a unit test with a mock
    /// backend wants; a loop driven by a Deployment IR wants [`Self::single_env_at`], because
    /// there the control rate is declared and the scene's timestep is whatever the scene says.
    pub fn single_env() -> Self {
        Self {
            simulation: DomainCfg::new(1, 1),
            observation: DomainCfg::new(1, 1),
            inference: DomainCfg::new(1, 1),
            training: None,
        }
    }

    /// A single env whose control step is one control *period* (§12.1, packet M5/V11).
    ///
    /// `physics` is the rate the scene is stepped at and `control` the Deployment IR's
    /// `rate.control`, so one [`Env::step`](crate::Env::step) advances `physics / control`
    /// simulation ticks — the substeps of one control step — and records one row. The
    /// observation domain shares that period: the state does not change inside the window, so
    /// observing every simulation tick would push the same reading into a `TemporalWindow`
    /// several times and re-render the same frame.
    ///
    /// A scene whose timestep does not divide the control period is refused by name rather
    /// than rounded (App. B.5): a control step that is 3.5 substeps long is not a control step.
    pub fn single_env_at(physics: TickRate, control: TickRate) -> Result<Self, EnvError> {
        // `physics / control` as an exact rational: (pn/pd) / (cn/cd) = (pn*cd) / (pd*cn).
        let (num, den) = (physics.num() * control.den(), physics.den() * control.num());
        if den == 0 || num % den != 0 {
            return Err(EnvError::Schedule(format!(
                "the scene steps at {} Hz and the deployment declares rate.control = {} Hz:                  one control period is {} simulation ticks, which is not a whole number",
                physics.as_hz_f64(),
                control.as_hz_f64(),
                num as f64 / den as f64,
            )));
        }
        let period = u32::try_from(num / den).map_err(|_| {
            EnvError::Schedule(format!(
                "a control period of {} simulation ticks does not fit in u32",
                num / den
            ))
        })?;
        Ok(Self {
            simulation: DomainCfg::new(1, 1),
            observation: DomainCfg::new(1, period),
            inference: DomainCfg::new(1, period),
            training: None,
        })
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

    /// Packet M5/V11 oracle 1: one control step is one control period.
    ///
    /// The demo's own numbers -- a scene of `timestep="0.005"` under a deployment that declares
    /// `rate.control = 50` -- plus the two edges: a timestep that does not divide the control
    /// period is refused by name, and a scene already at the control rate needs no substep.
    #[test]
    fn a_control_period_is_a_whole_number_of_substeps() {
        let at = |timestep: f64, hz: u64| {
            BatchDomains::single_env_at(
                TickRate::from_period_secs(timestep).expect("a positive timestep"),
                TickRate::hz(hz),
            )
        };
        // The demo: 200 Hz physics, 50 Hz control.
        let demo = at(0.005, 50).expect("0.005 s divides a 20 ms control period");
        assert_eq!(demo.inference.period, 4);
        assert_eq!(
            demo.observation.period, 4,
            "one observation per control step"
        );
        assert_eq!(demo.simulation.period, 1, "the tick is still the tick");
        Schedule::build(&demo).expect("the derived schedule is valid");
        // Not a divisor: refused, and the message names both rates.
        let err = at(0.006, 50)
            .expect_err("6 ms does not divide 20 ms")
            .to_string();
        assert!(
            err.contains("166.6") && err.contains("50"),
            "the refusal must name both rates: {err}"
        );
        // Already at the control rate: one substep, which is what `single_env` means.
        assert_eq!(at(0.02, 50).expect("20 ms is 20 ms").inference.period, 1);
        // A control rate faster than the physics is the same refusal, not a zero period.
        assert!(at(0.02, 100).is_err());
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
