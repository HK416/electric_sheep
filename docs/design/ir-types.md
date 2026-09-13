# IR common type system (`es-ir::types`) — design

Spec refs: spec 5.4 (공통 타입 시스템), spec 5.2 (batch semantics), spec 3.1 (conventions),
Appendix B.1. Packet: `docs/packets/M0/P18.md`.

## `PortType`

```
PortType = (ElemType, Shape, Unit, Frame, TimeRef, Option<ImageSpec>)
```

`Shape` carries **only the per-sample dims**. The batch axis is never in the shape: each of the
four batch domains (simulation / observation / inference / training, spec 5.2) picks its own
size, so a shape that pinned one of them would be wrong in the other three.

Two ports connected by an edge must match on `elem`, `shape`, `unit` and `time` exactly, and on
`frame` up to the `Policy` exemption (below). `image` must be both-absent or equal.

## Unit — named units as `m^a · kg^b · s^c · rad^d`

Multiplication and division are defined on the four-exponent basis. A named unit that has an
exponent vector converts to it and back:

| Unit | m | kg | s | rad |
|---|---|---|---|---|
| `Dimensionless` | 0 | 0 | 0 | 0 |
| `Length` | 1 | 0 | 0 | 0 |
| `Mass` | 0 | 1 | 0 | 0 |
| `Time` | 0 | 0 | 1 | 0 |
| `Angle` | 0 | 0 | 0 | 1 |
| `Velocity` | 1 | 0 | −1 | 0 |
| `Acceleration` | 1 | 0 | −2 | 0 |
| `AngularVelocity` | 0 | 0 | −1 | 1 |
| `Force` | 1 | 1 | −2 | 0 |
| `Torque` | 2 | 1 | −2 | 0 |
| `Pressure` | −1 | 1 | −2 | 0 |

The table is bidirectional and the mapping is injective, so `Length / Time = Velocity` and
`Velocity * Time = Length` name themselves again. An exponent vector with no name becomes
`Composite(UnitPowers)`; `Composite` with all-zero powers normalizes back to `Dimensionless`.

### Opaque units

`Current`, `Voltage`, `Quaternion`, `RotationMatrix`, `Normalized { lo, hi }`, `Pixel`,
`Luminance`, `Depth` and `Token` have **no** exponent vector and are rejected by `mul` / `div`
(`TYPE-010`). Reasons:

- `Current` / `Voltage` need an ampere axis the spec's four-exponent basis does not have.
- `Quaternion` / `RotationMatrix` are not products of base units at all — multiplying two
  rotations is a group operation, not a dimensional one.
- `Normalized { lo, hi }` has already been mapped onto an arbitrary interval; the physical
  dimension it came from is gone by construction.
- `Pixel`, `Luminance`, `Token` are counts in a domain-specific alphabet.
- `Depth` is metres, but keeping it opaque is deliberate: it stops a depth image from silently
  becoming a `Length` (and then a `Velocity`) through unit algebra. Conversion is an explicit
  node, not an inference.

`checked_add` requires the two `Unit` values to be **equal as enum values** — not merely
equal in exponents. `Depth + Length` is rejected, and so is `Normalized{0,1} + Normalized{-1,1}`.

### Policy-input rule

A policy input port accepts only `Normalized`, `Dimensionless` or `Token`
(`PortType::is_policy_input`, diagnostic `TYPE-011`). Feeding raw metres or radians into a
network is the common bug this rule catches at compile time (spec 5.4).

## Frame

`World | LocalOrigin | Body(id) | Sensor(id) | Joint(id) | Camera(id) | Image(id) | Policy`,
ids are `es_core::StableId`. Frames must be equal — except that `Policy` is compatible with
everything: it denotes "inside the network, no frame checking" (spec 5.4). `Camera(id)` is the
OpenCV optical frame and `Image(id)` is pixel coordinates, both per spec 3.1.

## TimeRef and the join rule (`TYPE-014`)

```
Tick                                   the current physics tick
Sensor { id, align }                   align = Hold | Interpolate | Reject
Window { base, n, stride }             a TemporalWindow
```

An **edge** requires equal `TimeRef`s. A **combination** of several inputs (`Concat`, `Arith`,
…) calls `TimeRef::join`, which is where `TYPE-014` comes from:

| a | b | result |
|---|---|---|
| equal | equal | that `TimeRef` |
| `Tick` | `Sensor { align: Hold \| Interpolate }` | `Tick` — the sensor is resampled onto the tick |
| `Sensor { x, align_x }` | `Sensor { y, align_y }`, x ≠ y | `Tick` if `align_x == align_y` and it is not `Reject`, else `TYPE-014` |
| `Window { b, n, s }` | identical `Window` | that window |
| anything else | | `TYPE-014` |

`Align::Reject` means "do not combine across this clock" — a join that reaches it is an error,
not a silent hold. Joining two different `Window`s is also an error: the learning input meaning
(spec 7.4) differs, and picking one for the user would change what the network sees.

## Canonical encoding

`PortType::canonical` writes the whole type into a `CanonWriter` (`hash.rs`) so node params that
carry a type hash identically across machines. `Normalized`'s two `f64`s go through the writer's
float rule (−0.0 → 0.0, NaN rejected); no other float reaches the hash from this module.

## Schema migrations

`docs/packets/M0/P29.md` freezes the builtin node kind lists (spec 28.7 gate 10) and
`factory.rs` asserts each list's blake3 digest, so a rename, an addition or a removal fails CI
rather than silently moving `task_hash` / `learning_hash`. Every such change is recorded here.

| version | change |
|---|---|
| Task IR `1` | initial IR-D schema (P20) |
| Task IR `2` | IR-C: `TaskIr.control: Option<ControlGraph>`, mixed into `task_hash`. `None` is the default and every pre-IR-C file parses unchanged, but the version itself is hash input, so every `task_hash` moves. New frozen list `BUILTIN_CONTROL_KINDS` = `Sequence`, `Branch`, `Repeat`, `SubTask`, digest `2438218d1bb498aae98de89569fd62bbe1820ea1c860da0471a5e98235a21158`. `BUILTIN_TASK_KINDS` and its digest are unchanged: a control node is not a `TaskNode`, so `TaskNodeFactory` does not claim these kinds. See `docs/design/control-graph.md`. |
