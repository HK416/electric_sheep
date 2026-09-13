# Learning loop — collect / intervene / distill

Spec: §13.1 (the loop is a first-class workflow, each step a CLI command *and* an artifact),
§13.2 (the status of intervention data — labels, provenance, how interventions enter the
dataset), §13.3 (loop reproducibility: every iteration is linked by the hash chain), §19.1
(LeRobot format, intervention labels are episode metadata), §19.2 (Dataset Identity),
§19.3 (Training Identity), §9.3/§9.4 (Safety Plane), `INV-12` (no code path disables it).

Packet: `docs/packets/M3/W7-learning-loop.md`. This note is the human-review artifact; the
code in `crates/es-data/src/{collect,intervention}.rs` is downstream of it.

---

## 1. What this wave implements, and what it does not

§13.1's loop has six commands. This wave owns three of them:

```
es loop collect    policy.esb + scene  ->  dataset/ (+ loop.jsonl)
es loop intervene  dataset/ + segments.json  ->  dataset/ (labels)      (+ loop.jsonl)
es loop distill    dataset/...  ->  dataset/ + training_identity.json   (+ loop.jsonl)
```

`train` is **not** here and will not be: the training run itself is on the PyTorch side of
§2.3's split (Python is a first-class dependency on the learning path only). What `distill`
produces is the *input identity* of that run — the merged dataset, the deterministic split,
and the `training_identity.json` whose `dataset.lock` slot §19.3 requires. The remaining
slots of `TrainingIdentity` (`config`, `optimizer`, `scheduler`, `seed`, `base_model`,
`augmentation`, `precision`, `topology`, `checkpoint_manifest`, `metrics`, `hardware`) are
written as all-zero digests, because this side of the boundary does not know them. A zero
digest is legible as "unset"; a fabricated one would make `training_hash` a lie.

`es eval` and `es deploy` already exist or are owned by other packets; the loop's `FAILURE
SET` arrow (§13.1) is `distill` with several dataset roots on the input side.

---

## 2. The intervention label schema (§13.2)

§13.2 puts interventions in *episode metadata*, as a list of segments with a reason and an
operator. That is the shape this implements, with per-frame columns added so that a training
loader can weight frames without re-deriving them from the segment list.

### 2.1 Segments — `meta/interventions.jsonl`

```rust
pub struct InterventionSegment {
    pub episode: u32,
    pub start_frame: u32,
    pub end_frame: u32,            // inclusive
    pub source: InterventionSource,
    pub operator_id: Option<String>,
    pub note: Option<String>,
}

pub enum InterventionSource { Teleop, Scripted, Corrective }
```

One JSON object per line, sorted by `(episode, start_frame)`. This is the provenance record:
§13.2's `reason` is `note`, and `operator_id` is carried verbatim because the point of the
field is attribution, not analysis. `Scripted` is the machine-driven intervener (a classical
controller, a replayed demonstration, the test hook below); `Teleop` is a human at a device;
`Corrective` is a human correcting a running policy — the HIL-SERL case §13.2 names.

Segments are **not** merged or normalized: two overlapping teleop segments from two operators
stay two lines, because collapsing them would destroy the attribution that is the reason the
file exists. Overlap is resolved only in the per-frame column, where it is a boolean.

### 2.2 Per-frame columns

Two columns join the LeRobot feature set:

| feature | dtype | shape | meaning |
|---|---|---|---|
| `intervention` | `int64` | `[1]` | `0` or `1`: is this frame inside any segment |
| `action_source` | `int64` | `[1]` | `0` policy, `1` clamped, `2` fallback, `3` human |

`intervention` is a `u8` value in §13.2's sense; it is stored as `int64` because the four
column dtypes this crate reads and writes are `float32 / float64 / int64 / bool` (see
`docs/api-notes/lerobot-dataset.md`) and inventing a fifth physical type to save seven bytes
a frame is not worth the interop risk.

**`action_source` is not the same question as `intervention`.** §18.5 and `es-safety`'s
`ActionSource` are explicit that a clamp or a fallback is *normal operation*, not a failure
and not a human: the Safety Plane changed the action, nobody intervened. So the two columns
are independent, and the four `action_source` codes map one-to-one onto what the plane
reports:

```
es_safety::ActionSource::Policy      -> 0 policy
es_safety::ActionSource::Clamped     -> 1 clamped
es_safety::ActionSource::Fallback(_) -> 2 fallback   (the kind is in the safety counters)
(no plane counterpart)               -> 3 human
```

A human action that the plane then clamped is recorded as `action_source = 1` (clamped) with
`intervention = 1`. That is the honest pair: the frame *was* an intervention, and the action
on the actuator was *not* the one the human asked for. Collapsing it to `3` would hide a
clamp, which is exactly the evidence §9.3 exists to preserve.

### 2.3 What `label()` may and may not rewrite

`es_data::intervention::label(root, segments)`:

- writes `meta/interventions.jsonl`;
- rewrites the `intervention` column of every affected episode (`1` inside a segment, `0`
  outside), streaming one episode at a time;
- adds `intervention` / `action_source` to `meta/info.json` if the dataset predates them
  (then, and only then, `dataset_schema_hash` changes);
- **never touches `action_source`.** That column is collection-time provenance: it records
  what the Safety Plane actually emitted while the robot was moving. A label applied hours
  later cannot know that, and overwriting it would turn a measurement into an assertion.

Consequences, which are the point of §19.2's three-way hash:

```
dataset_content_hash   changes   (parquet bytes changed)
dataset_schema_hash    unchanged (unless a column had to be added)
dataset_split_hash     unchanged (the split is not the labeller's business)
```

§13.2 also notes that intervention segments may be *weighted* differently in training and
that the weighting policy is part of `dataset_hash`. The weighting policy is a training-side
document; it enters the hash through `TrainingIdentity::config`, not through these columns.

---

## 3. From a collected episode to LeRobot frames

`es_env::Episode` is columnar per env (`qpos`, `qvel`, `ctrl`, `sensordata`, `reward`,
`done`, `failure`). The mapping is deliberately narrow — one feature per thing a policy
consumes or produces, nothing speculative:

| LeRobot feature | dtype | shape | from |
|---|---|---|---|
| `observation.state` | `float32` | `[nq + nv]` | `qpos ‖ qvel`, the raw observation of §12.2's plan-free path |
| `action` | `float32` | `[nu]` | `ctrl` — the action **after** the Safety Plane, i.e. what the actuator saw |
| `reward` | `float64` | `[1]` | `reward` |
| `intervention` | `int64` | `[1]` | §2.2 |
| `action_source` | `int64` | `[1]` | §2.2 |
| `timestamp` | reserved | | `frame / fps`, `fps` = the deployment's control rate |
| `task_index` | reserved | | all `0` — one task per collect run |

`action` is the post-plane action on purpose. A dataset whose `action` column is the policy's
raw output would train the next policy to reproduce actions that the plane refuses, and the
loop would never converge. The raw output is recoverable where it matters: `action_source`
says on which frames the two differ.

`tasks.jsonl` gets one entry, the string `es:task:<task_hash hex>`. A collect run has no
natural-language instruction to put there; the hash is at least resolvable back to the Task
IR that produced it.

### 3.1 Image channels: `VideoRef` placeholders and a warning

If the bundle's `ObservationSpec` declares channels carrying an `ImageSpec`, a `video`
feature is written into `meta/info.json` for each — so `info.video_path` is populated and
`LeRobotDataset::read_episode` hands back one `VideoRef` per frame per camera — but **no mp4
is written**, because there is no renderer in this build (`es-render` is layer 5 and is a
later milestone; `es eval run` refuses image observations for the same reason).

Every such run emits a warning naming each camera:

```
warning: camera "cam_front": info.json declares a video feature and read_episode will
         produce VideoRef placeholders, but no mp4 was written (no renderer in this build)
```

The alternative — omitting the feature — would make the dataset's `dataset_schema_hash`
describe an observation the task never declared, and a later `es loop distill` would silently
merge it with a real multi-camera dataset. A declared-but-absent video is a dangling
reference the reader can detect; a missing declaration is a lie the reader cannot.

---

## 4. `loop.jsonl` — the loop is reproducible (§13.3)

§13.3 wants each iteration linked: `dataset_hash` of iteration *n* is a function of iteration
*n-1*'s data plus the new intervention episodes, with `evaluation_hash` held fixed. The
ledger that makes that checkable is one append-only file per dataset root:

```rust
pub struct LoopStep {
    pub kind: LoopKind,                       // Collect | Intervene | Distill
    pub inputs: BTreeMap<String, String>,     // name -> hex digest or path
    pub outputs: BTreeMap<String, String>,
    pub created: u64,                         // unix seconds
}
```

One JSON object per line in `<root>/loop.jsonl`, appended, never rewritten. What each step
records:

| step | inputs | outputs |
|---|---|---|
| `collect` | `task`, `observation`, `learning`, `deployment` (the bundle manifest's hashes), `seed`, `episodes` | `content`, `schema` |
| `intervene` | `content` (before), `segments` (count) | `content` (after), `schema` |
| `distill` | `content`/`schema` of each input root, `seed`, `ratios` | `content`, `schema`, `split`, `training_hash` |

The chain property is what a reviewer checks by eye and what the oracle checks in CI: step
*n*'s input `content` is step *n-1*'s output `content`. A `distill` step is appended to
`loop.jsonl` of **every input root as well as the output root**, so a dataset's own ledger
records that it was consumed, and the output's ledger records what it was made of.

`created` is the only non-reproducible value in the file, for the same reason `RunConfig`'s
is (§10.4): provenance is not identity. Nothing in `loop.jsonl` feeds a hash.

`loop.jsonl` deliberately does **not** record `policy_hash` of a *trained* checkpoint or
`evaluation_hash`: neither is produced by these three commands. §13.3's discipline — hold
`evaluation_hash` fixed while data and policy move — is enforced by `es eval compare`
(already implemented), which is where a changed `evaluation_hash` invalidates a comparison.

---

## 5. Collection: the Safety Plane is on the only path (`INV-12`)

`Collector::run` drives `es_env::Env::step_with_policy`, which is the single actuator path:
chunk buffer -> `SafetyPlane::validate` -> `ctrl`. There is no collect-mode branch around it,
including for injected human actions.

```
        intervener says Some(a)
                 |
                 v
policy  ->  [ wrapper ] -> action chunk -> chunk buffer -> SafetyPlane -> ctrl -> physics
                 ^
        intervener says None
```

The intervener is wired in as a `PolicyRuntime` wrapper rather than as a post-plane override,
and that choice is the whole safety argument: an injected action is *just another chunk*. It
is bounded by the same envelope, rate-limited by the same limiter, and counted by the same
counters as a policy chunk. Nothing about collection widens the envelope, and nothing
disables the plane — which `INV-12` forbids outright, tests included.

```rust
// Fn(episode, frame, &obs) -> Option<[f64; NJ]>
pub type Intervener<'a, const NJ: usize> =
    &'a mut dyn FnMut(u32, u32, &[f64]) -> Option<[f64; NJ]>;
```

`obs` is the observation the wrapper was handed for that control tick (the `qpos ‖ qvel` row
of §12.2's raw path), so a scripted intervener is a pure function of the state and the run is
bit-reproducible for a seed. That is what makes it usable as an oracle: the test's intervener
injects on known frames, and the dataset's segments must come back exactly.

### 5.1 Which *frames* an injection labels

An injection is a decision made at control tick `t`, but the action it produces reaches the
actuator later and for longer: §12.3's deterministic latency delays it by
`latency_ticks(expected_latency_ms)` and §8.5's `execute_chunk = K` keeps it driving for `K`
ticks. So an injection at tick `t` labels frames

```
[ t + latency_ticks , t + latency_ticks + K )
```

not the single frame the intervener looked at. Recording the decision tick instead would
mislabel every deployment with non-zero inference latency — which is the normal case (§8.6).

Under `ChunkBlendPolicy::TemporalEnsemble` the emitted action is an average of overlapping
chunks, so a labelled frame is one the injected chunk *contributed to* rather than one it
solely determined. That is recorded here rather than modelled: a per-frame blend weight is a
learning-side concern, and inventing one would be a number nobody measured (§1.7).

### 5.2 Recovering `action_source` per frame

`DomainRunner::emit_actions` writes `ctrl` and does not hand back the `SafeAction`, so the
per-frame source is read from the plane's own counters across the step:

```
fallback_activations grew  -> 2 fallback
else clamped_steps grew    -> 1 clamped
else the frame is labelled -> 3 human
else                       -> 0 policy
```

One env per collect run, so exactly one `validate` call happens per step and the delta is
unambiguous. Reading the counters rather than plumbing a return value through `es-env` keeps
this packet out of a neighbouring crate's API, and the counters are the record §10.3 already
relies on.

---

## 6. Distillation: merge, split, identity

`distill(inputs, SplitSpec { ratios, seed }, out_root)`:

1. Every input's `meta/info.json` `features` map and `fps` must be **equal**. A mismatch is
   refused, naming the feature: merging two datasets whose `observation.state` means
   different things produces a `dataset_schema_hash` that describes neither.
2. Episodes are re-indexed `0..N` in input order, then by original index — a total order, so
   two runs of the same inputs produce byte-identical parquet.
3. `meta/interventions.jsonl` of each input is merged with the episode indices remapped, so
   labels survive the merge. (An input with no such file contributes none.)
4. `Split::deterministic(N, ratios, seed)` — the existing §19.2 seeded Fisher-Yates cut.
   `train` and `val` take `floor(N * ratio)` and `test` takes the remainder, so the three
   lists are always an exact partition of `0..N`.
5. `DatasetIdentity::compute` -> `training_identity.json`, `split.json` (the lists themselves,
   so `dataset_split_hash` is reproducible by hand and not only by re-running this function)
   -> a `Distill` `LoopStep`.

Ratios are not required to sum to 1: the split function clamps and the remainder lands in
`test`, which is the conservative direction (a mis-specified ratio shrinks training data
rather than leaking test data into it). A ratio sum above 1 is refused, because there it is
not clamping but silent truncation of the test set.

---

## 7. Deliberate omissions

- **No teleop device driver.** `--teleop <device>` of §13.1 is real-robot I/O (M3 W1). The
  `Teleop` label variant exists so that a dataset recorded by that path is describable now;
  the driver is not here and `es loop collect` has no `--teleop` flag.
- **No failure-set filter.** §13.1's `FAILURE SET` arrow is `es loop distill` over several
  roots; selecting which episodes failed is `es eval`'s output, and wiring the two together
  is a later packet. `distill` merges what it is given.
- **No dataset deduplication.** Distilling the same root twice doubles its episodes. That is
  arguably the caller's intent (weighting by repetition), and guessing otherwise is worse
  than the surprise.
- **No video.** §3.1.
