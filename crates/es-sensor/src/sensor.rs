//! Runtime-side sensor bookkeeping: a [`SensorDesc`] mirrors one sensor instance (id, kind,
//! rate, realism model) and [`SensorBank`] steps many of them with static partitioning — no
//! threads, just a deterministic (`BTreeMap`) iteration order (spec 3.4: no `HashMap`-iteration
//! dependence).

use std::collections::BTreeMap;

use es_core::{PhysTick, StableId, TickRate};
use thiserror::Error;

use crate::noise::{sensor_seed, SensorModel};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SensorError {
    #[error("sensor {0} is not registered in this bank")]
    UnknownSensor(StableId),
}

/// One sensor instance: identity, a free-form kind tag (e.g. `"camera"`, `"imu"`, `"encoder"`
/// per the spec 18.3 table), its sample rate and its realism pipeline.
#[derive(Clone, Debug)]
pub struct SensorDesc {
    pub id: StableId,
    pub kind: String,
    pub rate: TickRate,
    pub model: SensorModel,
}

/// Steps every registered sensor's [`SensorModel`] once per tick, in `StableId` order (stable
/// across runs regardless of registration order).
#[derive(Clone, Debug, Default)]
pub struct SensorBank {
    sensors: BTreeMap<StableId, SensorDesc>,
}

impl SensorBank {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, desc: SensorDesc) {
        self.sensors.insert(desc.id, desc);
    }

    #[must_use]
    pub fn get(&self, id: StableId) -> Option<&SensorDesc> {
        self.sensors.get(&id)
    }

    /// Applies one sensor's model in place, keying its RNG stream on `(base_seed, sensor id,
    /// env, tick)` (spec 3.4 `DET-001`).
    ///
    /// # Errors
    /// [`SensorError::UnknownSensor`] if `id` was never [`insert`](Self::insert)ed.
    pub fn step_one(
        &mut self,
        id: StableId,
        env: u32,
        tick: PhysTick,
        base_seed: u64,
        values: &mut [f64],
    ) -> Result<(), SensorError> {
        let desc = self
            .sensors
            .get_mut(&id)
            .ok_or(SensorError::UnknownSensor(id))?;
        let seed = sensor_seed(base_seed, id);
        desc.model.apply(env, tick, values, seed);
        Ok(())
    }

    /// Steps every sensor that has a matching entry in `values`, in deterministic `StableId`
    /// order. Sensors with no entry in `values` are skipped (static partitioning: which
    /// sensors run is decided by the caller's `values` map, not by iteration side effects).
    pub fn step_all(
        &mut self,
        env: u32,
        tick: PhysTick,
        base_seed: u64,
        values: &mut BTreeMap<StableId, Vec<f64>>,
    ) {
        for (id, desc) in &mut self.sensors {
            if let Some(v) = values.get_mut(id) {
                let seed = sensor_seed(base_seed, *id);
                desc.model.apply(env, tick, v, seed);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::noise::Stage;

    fn desc(name: &str) -> SensorDesc {
        SensorDesc {
            id: StableId::from_path(name),
            kind: "test".to_owned(),
            rate: TickRate::hz(30),
            model: SensorModel {
                stages: vec![Stage::bias(1.0, 0.0)],
            },
        }
    }

    #[test]
    fn step_one_applies_model_and_rejects_unknown_id() {
        let mut bank = SensorBank::new();
        let id = StableId::from_path("cam/front");
        bank.insert(desc("cam/front"));
        let mut v = [0.0];
        bank.step_one(id, 0, PhysTick(0), 1, &mut v).unwrap();
        assert_eq!(v, [1.0]);

        let missing = StableId::from_path("cam/back");
        let mut v2 = [0.0];
        assert_eq!(
            bank.step_one(missing, 0, PhysTick(0), 1, &mut v2),
            Err(SensorError::UnknownSensor(missing))
        );
    }

    #[test]
    fn step_all_iterates_in_stable_id_order_and_skips_absent_entries() {
        let mut bank = SensorBank::new();
        bank.insert(desc("z_last"));
        bank.insert(desc("a_first"));
        let mut values = BTreeMap::new();
        values.insert(StableId::from_path("a_first"), vec![0.0]);
        // "z_last" has no entry: step_all must not panic, and must not fabricate one.
        bank.step_all(0, PhysTick(0), 1, &mut values);
        assert_eq!(values.len(), 1);
        assert_eq!(values[&StableId::from_path("a_first")], vec![1.0]);
    }
}
