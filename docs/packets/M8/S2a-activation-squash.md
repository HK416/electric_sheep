# M8 S2a — `StateEncoder{Mlp}` gains an activation, `PolicyHead{Regression}` a squash, and no committed hash moves

Spec: §8.3 (the node set), §14.4 ("activation and squash are node parameters; absent = default =
today's canonical form"), §28.11 wave 1, §1.5 (`es-ir` is at 6,008 code lines: over the 6,000
target, under the 10,000 cap — keep this addition small and report the new count). Precedent:
packet M7/R5's `SensorRender` (`crates/es-ir/src/task.rs` ~723–790: `#[serde(default,
skip_serializing_if)]` + canonical bytes written only when not default). Gaps it closes:
`docs/design/quadruped-track.md` 3.4 items 1–3 (ReLU fixed, no activation before the head, no
`tanh`). Design note to extend: `docs/design/learning-lowering.md` (+ `.ko.md`); the decision is
recorded in `docs/design/rl-continuation.md` section 8 question 3.

## the question

brax's MLP is `Dense swish, Dense swish, Dense swish, Dense`; rsl_rl's is `Linear ELU … Linear`;
ours lowers `Mlp { hidden }` to `Linear ReLU … Linear` with no activation before the head and no
`tanh` anywhere. **Can the node set say which activation, whether the last hidden layer is
activated, and whether the regression output is squashed — without moving `learning_hash` of
any committed document or any lowered module's bytes?**

## spec

* `StateEncoderKind::Mlp { hidden: Vec<u32>, activation: Activation, activate_output: bool }`
  with `enum Activation { Relu, Elu, Swish, Tanh }`; defaults `Relu` / `false`, both
  `#[serde(default, skip_serializing_if = …)]`. **The canonical bytes of a default node are
  byte-identical to today's** — today `canonical` writes `format!("{kind:?}")`, i.e. the string
  `Mlp { hidden: [256] }`; write that exact string for the default and append the two fields only
  when non-default. Pin the committed `learning.toml` and `learning-pretrained.toml` hashes as hex
  literals **before** touching anything (compute them on `main` first).
* `LearningNode::PolicyHead { …, squash: Squash }` with `enum Squash { None, Tanh }`, default
  `None`, same serde/canonical rule. Validation: `squash != None` on any head kind but
  `Regression` is a named diagnostic (follow `learning.rs`'s existing error-code series).
* `NodeSchema` (the inspector's parameter table, packet M7/E3 — find where `StateEncoder` and
  `PolicyHead` declare their `ParamType`s) gains the three parameters so the editor's inspector
  shows them; `python/es/builder.py` gets keyword arguments with the same defaults if it spells
  these nodes.
* Lowering (`crates/es-policy/src/lower/torch.rs` ~648–665 and ~754): `Elu → nn.ELU()`,
  `Swish → nn.SiLU()`, `Tanh → nn.Tanh()`; `activate_output` appends the activation after the last
  `nn.Linear` of the encoder; `Squash::Tanh` wraps the regression head's output in `torch.tanh`
  before the reshape. A default graph lowers to **byte-identical** source (`lowering_hash`
  unmoved: pin it too).
* Weight keys and shapes are unchanged by any of the three (activations carry no weights).

## context

```
crates/es-ir/src/learning.rs
crates/es-ir/src/schema.rs
crates/es-ir/src/**
crates/es-ir/tests/**
crates/es-policy/src/lower/torch.rs
crates/es-policy/tests/**
crates/es-policy/python/**
python/es/builder.py
crates/es-ir-types/src/codes.rs
crates/es-data/src/lerobot_config.rs
crates/es-policy/src/reference.rs
crates/es-editor/tests/common/mod.rs
crates/es-runtime-embedded/tests/embedded.rs
crates/es/tests/cli.rs
docs/design/learning-lowering.md
docs/design/learning-lowering.ko.md
docs/packets/M8/S2a-activation-squash.md
docs/packets/M8/S2a-activation-squash.ko.md
```

`learning.rs` (the two enums, the fields, canonical, validation), the schema file wherever it
lives in `es-ir`, `torch.rs` (three match arms), tests, the builder only if it spells the nodes,
the note, this packet.

## oracle

1. `cargo test -p es-ir committed_learning_hashes_are_unmoved_by_activation_and_squash` — the
   two committed documents parse, their `learning_hash`es equal the hex literals pinned from
   `main`, and a document that spells `activation = "Relu"`, `activate_output = false`,
   `squash = "None"` explicitly hashes the same as one that omits them; a non-default value
   moves the hash.
2. `cargo test -p es-ir squash_is_refused_off_a_regression_head` — the named diagnostic.
3. `cargo test -p es-policy lower_mlp_activations_source` — the generated source of the
   committed graph is byte-identical to before (pinned `lowering_hash`); for each of
   `Elu/Swish/Tanh × activate_output × squash` the source contains the expected `nn.` spellings.
4. `cargo test -p es-policy lower_mlp_activations_match_torch -- --ignored` (`ES_PYTHON`, server):
   a 15 → [32, 32] → 6 graph per combination, random weights, 64 random inputs — the lowered
   module vs a hand-written torch reference in `crates/es-policy/python/` **bitwise** on CPU.
5. `cargo xtask context-budget` (report `es-ir`'s new code-line count; ≤ 6,100);
   `cargo xtask ci`; `cargo xtask check-scope docs/packets/M8/S2a-activation-squash.md`.

## acceptance

Oracles 1–5. `learning-lowering.md` gains a short subsection naming the three parameters, the
`nn.` mapping and the hash rule; Korean sibling updated.

## forbidden

Moving any committed `learning_hash` or `lowering_hash`; a new node (this is three parameters,
not a node); touching `es-policy/src/lerobot.rs` or the ACT lowering; `docs/ARCHITECTURE*.md`;
`tests/golden/**`; a Python dependency in `es-ir`. INV-17: no new trait.
