# W2 — Sensor realism basics, channel contract, and actuator model (CPU)

Spec: spec 18.2 (actuators), spec 18.3 (sensors and realism: noise, latency, dropout, rolling
shutter, exposure), spec 15.1 (render -> observation path, channel contract), spec 7.2
(`ImageSpec` fields a camera sensor populates: shutter, exposure, rate), spec 3.4 (no global
RNG, deterministic execution), spec 18.4 (deterministic reduction context).

## context

```
crates/es-sensor/**
crates/es-actuator/**
docs/packets/M1/W2-sensor-actuator.md
```

## spec

- `es-sensor::channel` — the render -> observation channel contract (spec 15.1): `Channel`
  (`Rgb8 | RgbF32Linear | Depth32{unit_m} | Normal | SegmentationId | Flow | PtRadiance`) with
  `dtype()`, `n_components()` and `unit()` (a fixed string vocabulary — `es-sensor` is layer 3,
  below `es-ir`'s layer 6, and cannot use `es_ir::Unit`); `CameraContract{channels, width,
  height, rate, shutter, exposure_ticks}` carrying the spec 7.2 `ImageSpec` timing fields.
  `Channel` implements `Ord`/`Eq` by hand (its `Depth32` payload is an `f64`) so it can live in
  a `BTreeSet`, ordering by IEEE-754 bit pattern rather than by value.
- `es-sensor::noise` — deterministic realism stages over `f64` slices, fixed op order:
  `Gaussian{sigma}`, `Bias{offset, drift_per_tick}` (stateless: a pure function of the absolute
  tick), `Quantize{step}`, `Dropout{p_drop, hold_last}`, `Latency{ticks}` (pre-allocated ring,
  zero pre-fill), plus `RollingShutterSkew` (a standalone per-row tick-offset table, not a
  `Stage`, since it needs image height rather than a flat buffer). `SensorModel{stages}` runs
  them in declared order. `NoiseRng` is a ~30-line splitmix64 core copied from
  `crates/es-env/src/rng.rs::EnvRng` (comment names the source) because `es-sensor` (layer 3)
  cannot depend on `es-env` (layer 9); `sensor_seed(seed, StableId)` folds a sensor's identity
  into the base seed so distinct sensors draw independent streams under the same seed. The one
  transcendental (Box-Muller `sqrt`/`ln`/`cos`) goes through `es_math::approx` (spec 3.2
  `DET-010`).
- `es-sensor::sensor` — `SensorDesc{id, kind, rate, model}` and `SensorBank`, a `BTreeMap`
  keyed by `StableId` so iteration order is deterministic without threads (static
  partitioning, spec 3.4: no `HashMap`-iteration dependence).
- `es-actuator::model` — `ActuatorModel::{Motor{gear}, Position{kp,kd,gear}, Velocity{kv,gear},
  General{gain,bias,gear}}` mirroring `crates/es-assets::scene::ActuatorKind`'s field names
  (mirrored rather than imported, to keep this layer-3 crate free of the layer-2 dependency);
  `force(ctrl, q, qd)` implements MuJoCo's motor/position/velocity laws directly and the
  `gaintype=affine, biastype=affine` general-actuator law inferred from the `es-assets` doc
  comment (marked `unverified` against upstream MuJoCo source). `Saturation` clamps
  `ctrl_range`/`force_range`; `ActuatorDelay` is a fixed-tick transport delay (pre-allocated
  ring, zero pre-fill, mirrors `noise::Stage::Latency`); `Backlash{deadband}` is a minimal
  dead-zone hysteresis model (spec 18.2 lists a further two-mass model as a later realism
  step — noted, not built here).

## oracle

```
cargo fmt -p es-sensor -p es-actuator --check
cargo clippy -p es-sensor -p es-actuator --all-targets -- -D warnings
cargo test -p es-sensor -p es-actuator
cargo xtask layering
cargo xtask check-spec-refs
```

## acceptance

- Each noise stage matches a hand-computed vector; `Gaussian` is bit-identical for the same
  `(seed, sensor id, env, tick)` and differs when the sensor id differs; mean/variance of
  10k Gaussian draws are within tolerance of `(0, 1)`.
- `Latency` shifts output by exactly `ticks` ticks with zero pre-fill; `Dropout{hold_last:
  true}` repeats the previous output while dropping.
- `RollingShutterSkew::row_offset_ticks` is monotonic non-decreasing in row and bounded by
  `readout_ticks` (0 at the top row, `readout_ticks` at the bottom row).
- `CameraContract` round-trips through `serde_json` losslessly and rejects an empty channel
  set or a zero width/height.
- `proptest`: the full noise pipeline (`Gaussian` + `Bias` + `Quantize` + `Dropout` +
  `Latency`) produces finite output for finite input, any parameter combination in range.
- `ActuatorModel::force` matches hand-computed cases for all four variants; `Saturation`
  clamps and rejects an inverted range; `Backlash` rejects a negative deadband and holds
  within the dead zone before tracking; `ActuatorDelay` shifts by exactly `ticks` ticks.
- `proptest`: `force`, `ActuatorDelay::step` and `Backlash::step` are finite for finite,
  bounded-range input; repeating the same call gives bitwise-identical output.
- Gate green: `cargo fmt -p es-sensor -p es-actuator --check`, `cargo clippy -p es-sensor -p
  es-actuator --all-targets -- -D warnings`, `cargo test -p es-sensor -p es-actuator`,
  `cargo xtask layering`, `cargo xtask check-spec-refs`.

## forbidden

- Any file outside `context`. In particular the root `Cargo.toml`, `crates/es`,
  `crates/es-policy`, `crates/es-physics-backend` (other agents' in-flight work), and any M0
  crate a reviewer may be reading.
- A dependency on `es-env` (layer 9) or `es-ir` (layer 6) from either crate — both are layer 3;
  copy/mirror instead (`noise::NoiseRng`, `model::ActuatorModel`'s field names).
- Vulkan/rendering code or a tile atlas — that is a later packet; this one is CPU math only.
- `HashMap`/`HashSet` for anything iterated in a deterministic path; `std` transcendentals
  (`f64::sin`/`cos`/`exp`/`ln`/`sqrt`) in a noise or force kernel — use `es_math::approx`.
- A new trait — the seven extension points of `INV-17` are the whole list; neither crate adds
  one.
- Disabling or widening the Safety Plane, or touching anything under `crates/es-safety` — out
  of scope and not needed here.
