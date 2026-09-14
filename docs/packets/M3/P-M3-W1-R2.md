# P-M3-W1-R2 — a partial `JointState` never reaches the Safety Plane

Spec: §9.4 (`SafetyPlane::observe_state` / `sensor_seen` inputs), §25.1 (a message from the network
is untrusted, and a default is not a measurement), §26.1 ("what is not validated is not executed"),
INV-12. Design note `docs/design/ros2-boundary.md` section 4.5.
Closes **S-1** of `docs/reviews/M3-W1.md`.

## context

```
crates/es-ros2/src/actuator.rs
crates/es-ros2/src/error.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R2.md
```

## spec

`reorder_joint_state` (`actuator.rs:65-81`) finds each configured joint by name and then reads

```rust
q[i]  = state.position.get(idx).copied().unwrap_or(0.0);
qd[i] = state.velocity.get(idx).copied().unwrap_or(0.0);
```

`sensor_msgs/JointState` documents that `position`, `velocity` and `effort` "may be empty" and only
"should" match `name` in length, so this is not a hostile case — a driver that publishes names and
positions but no velocities is ordinary, and it currently produces `qd = [0.0; NJ]`, a fabricated
zero velocity handed to `SafetyPlane::observe_state`. A short `position` fabricates a joint angle
the same way. Both feed the plane's hold-position target, its velocity estimate and every
rate-limit stage that differentiates against them.

The module's own doc comment (`:61-64`) and design note 4.5 already state the intended rule: "a
joint the sample does not carry rejects the whole sample (`ROS2-013`) rather than silently leaving
it at zero, so a partial state never reaches `SafetyPlane::observe_state` / `sensor_seen`". Make the
code match it.

- A `position` shorter than the matched index is `Ros2Error::MissingJoint`-shaped but a different
  condition; add one variant, `Ros2Error::PartialJointState { joint: String, field: &'static str }`,
  mapped to the existing `ROS2-013` in `error.rs:129-131` (the code is the "actuator lookup"
  catch-all; no new number).
- `velocity` is the one field where "empty" has a defensible reading. Choose **explicitly** and
  document it in design note 4.5: an entirely empty `velocity` (`state.velocity.is_empty()`) is
  "this driver does not report velocity" and yields `qd = [0.0; NJ]` with the decision recorded;
  a `velocity` that is non-empty but too short is `PartialJointState`. `position` has no such
  reading: short or empty is always an error.

No signature change to `reorder_joint_state`; it keeps returning `Result<([f64; NJ], [f64; NJ]),
Ros2Error>`.

## oracle

```
cargo test -p es-ros2 --features zenoh actuator
cargo fmt --check
cargo clippy -p es-ros2 --all-targets --features zenoh -- -D warnings
cargo xtask check-spec-refs
```

Unit tests in `actuator.rs`'s existing `mod tests`:

- `a_short_position_array_is_refused` — `name = ["j1","j2"]`, `position = [1.0]`,
  `velocity = [10.0, 20.0]`: `ROS2-013`, and the error names `j2` and `position`. **FAILS before the
  fix** (it returns `q = [1.0, 0.0]`).
- `a_short_velocity_array_is_refused` — `velocity = [10.0]` with a full `position`: `ROS2-013`.
- `an_empty_velocity_array_is_the_documented_zero` — `velocity = []`: `Ok`, `qd == [0.0; NJ]`,
  `q` exactly the reordered positions (bit-exact, as the existing test does).
- `an_empty_position_array_is_refused` — `position = []`: `ROS2-013`.
- The existing `reorders_by_name_and_rejects_a_missing_joint` still passes unchanged.

## acceptance

- The five tests pass; the first is confirmed to fail on the unfixed function.
- `Ros2Error::code()` still returns only the codes design note 4.2 lists plus `ROS2-000`,
  `ROS2-013`, `ROS2-020`; no new code string.
- Design note 4.5 gains one sentence naming the empty-`velocity` reading; the `.ko.md` sibling is
  updated in the same commit.
- The 13 `session_loopback` tests still pass; `actuator_publisher_sends_the_safe_action` unchanged.
- No allocation added on the accept path, no `HashMap`, no new trait.

## forbidden

- `crates/es-safety`, `crates/es-ir`, every other crate.
- `camera`, `hil`, `session`, `config`, `cdr`, `msg` (`msg::JointState`'s field types are correct
  and stay as they are).
- Adding a configuration switch for "tolerate a partial state": §26.1 does not admit one.
- Widening `ROS2-013`'s meaning to non-actuator conditions, or renumbering any code.
- Any other M3 W1 finding.
