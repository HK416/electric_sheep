<!-- English translation of docs/ARCHITECTURE.ko.md (canonical). Keep both files in sync; the pre-commit hook enforces staging them together. -->
# Electric Sheep — Robot Learning Compiler & Runtime Technical Specification (v1.0)

> **Electric Sheep** — an open platform that compiles and tracks the entire span of vision-based robot learning: task, observation, learning, and deployment
>
> Electric Sheep defines its product as a **Robot Learning Compiler & Runtime**. Physics simulation is one backend of this pipeline, and the center of the product is the **Policy layer**.
>
> **This document is the Electric Sheep v1.0 technical specification.** It defines the 5 IRs (Task / Observation / Learning / Deployment / Evaluation), a Safety Plane independent of the policy, hash-chain-based reproducibility, and the Learning Loop that closes through collection → learning → evaluation → deployment → intervention → retraining. The rationale for the design and its positioning within the 2026 ecosystem are covered in §0.

---

## 0. Positioning

### 0.1 One-Sentence Definition

> **Electric Sheep is the compiler for the robot learning pipeline.** When you describe task, observation, learning, and deployment in a typed intermediate representation, it validates them, compiles them into GPU kernels and an execution plan, executes them with identical semantics in simulation and on real hardware, and tracks the entire process with a hash chain.

Physics simulation is **one backend** of this pipeline. It is not the product.

### 0.2 Five Core Goals

1. **The entire span is described in a single semantic representation.** From scene to actuator, no step in between remains as "Python code outside the platform"
2. **It reproduces.** Under identical conditions, bit-for-bit identical results come out, and the task definition, observation pipeline, learning configuration, dataset splits, and policy weights are all tracked with a hash chain
3. **Vision-based policies are first-class.** ACT, Diffusion Policy, the π₀ family, SmolVLA, and the GR00T family are defined, trained, evaluated, and deployed on the same IR
4. **It carries through to real hardware with identical semantics.** The observation preprocessing and safety constraints used in simulation are not re-implemented on real hardware but executed as-is
5. **The learning loop closes.** Collection → learning → evaluation → deployment → intervention → failure data → retraining is the product's workflow

### 0.3 The 2026 Ecosystem

**The physics engine layer is already saturated.**

| Project | Status |
|---|---|
| **Newton 1.0** (2026-03, Linux Foundation, Apache-2.0) | NVIDIA·DeepMind·Disney. Warp + OpenUSD. MuJoCo Warp main backend, Kamino (maximal coordinate, closed-loop), VBD deformables, MPM, SDF collision, hydroelastic. Differentiable. **NVIDIA-only** |
| **MuJoCo Warp** | On RTX PRO 6000 Blackwell, reports locomotion 252×, manipulation 475× vs. MJX |
| **Genesis World 1.0 + Quadrants** (2026-05) | JITs Python kernels to CUDA·ROCm·Metal·**Vulkan**·CPU. Reverse-mode autodiff first-class across all backends. Its own renderer Nyx |
| **Isaac Lab** | PhysX + Newton dual backend, RTX tiled rendering |
| **ManiSkill3** | Vulkan-based GPU parallel simulation·rendering |
| **RoboVerse / MetaSim** | simulator-agnostic configuration, 276 tasks, 500K+ trajectories |

**In contrast, the learning loop layer is rapidly standardizing, and its center is LeRobot.**

LeRobot has bundled dataset, real-hardware control, policy, evaluation, and deployment into a single stack, and its supported policy families have expanded from ACT, Diffusion Policy, VQ-BET, HIL-SERL, TD-MPC to π₀, π₀-FAST, π₀.₅, GR00T N1.7, SmolVLA, XVLA, EO-1, MolmoAct2, WALL-OSS, EVO1. It has even brought in the World Model (VLA-JEPA, LingBot-VA, FastWAM) and Reward Model (SARM, TOPReward, Robometer) categories, and provides unified evaluation scripts on standard benchmarks such as LIBERO and MetaWorld.

**The actual shape of policy architectures** (the basis for why the `policy(obs) → action` model is insufficient)

| Policy | Parameters | Input | Temporal model | Action |
|---|---|---|---|---|
| ACT | 52–80M | RGB multi-view + state | `n_obs_steps = 1` | CVAE + Transformer, horizon 50, execution chunk 20 |
| Diffusion Policy | 263M | time-series visual conditioning | observation window | conditional denoising, receding horizon |
| SmolVLA | 450M | RGB + state + **language** | observation window | flow matching, chunk 16, execution horizon 4–8. Compressed to 64 visual tokens per frame |
| π₀ | 3.3–3.5B | RGB 3 views + sensorimotor + language | observation window | VLM + flow matching action chunk |
| π₀.₅ | — | RGB 2 views [3,256,256] + state[15] | — | chunk 16 |

**RTX 4090 inference latency measurements (reported by LeRobot)**

| Policy | Latency |
|---|---|
| ACT (52M) | **5.0 ms** |
| SmolVLA (450M) | **99.2 ms** |
| π₀ (3.5B) | **209.4 ms** |
| Diffusion Policy (263M) | **369.8 ms** |

**This table dictates the entire performance design.** A physics step is 1 ms, but policy inference is 5–370 ms. In vision-based learning, the bottleneck is not physics throughput but **policy inference latency and vision bandwidth**. GPU contact-solver optimization is the wrong target.

### 0.4 Gap Analysis

| Axis | LeRobot | Newton/MJWarp | Genesis | Isaac Lab | RoboVerse | **Electric Sheep** |
|---|---|---|---|---|---|---|
| Policy family support | best | none | none | RL-centric | medium | **described & compiled as IR** |
| Simulation | none (external dependency) | best | best | best | integrated | **adopted as backend** |
| **Observation pipeline specification** | code | code | code | config | config | **typed IR** |
| **Learning graph specification** | Python class | — | — | — | — | **typed IR** |
| **Reproducibility guarantee** | config logging | not guaranteed | not guaranteed | not guaranteed | not guaranteed | **bit-for-bit + hash chain** |
| **Safety runtime** | none | none | none | none | none | **policy-independent Safety Plane** |
| **Evaluation normalization** | benchmark scripts | — | — | — | 4-level generalization | **Evaluation IR** |
| Real-hardware deployment | yes | none | none | limited | none | **same IR executed** |
| Deployment form | Python | Python | Python | Omniverse | Python | **single static binary** |

**The five items in bold are the reason for existence.** If LeRobot solved "what can be used," Electric Sheep solves **"what exactly it was, why it behaved that way, and whether it can be made that way again."**

**Why it is defensible**
- LeRobot is a Python library. Because observation preprocessing is a Python function, it gets re-implemented in real-hardware deployment, and at that moment sim and real diverge. When described as IR, the same thing executes on both sides
- Reproducibility cannot be secured retroactively. The hash chain is a choice made at design time
- The safety runtime is not the concern of a policy library. Yet under the EU Machinery Regulation (effective 2027-01-20), this becomes a condition of market access (§27)
- Evaluation normalization is needed for both paper reproducibility and industrial adoption, yet no one owns the standard

### 0.5 Non-Goals

- **We do not compete on physics throughput.** If Newton/MJWarp is faster, we use it as a backend
- **A proprietary physics engine is not a required deliverable.** It is an optional item after M4, and the first thing to cut when scaling down (§1.9)
- **We do not invent a proprietary policy architecture.** We merely represent, execute, and evaluate the ACT·DP·VLA families
- **We do not build a proprietary learning framework.** We delegate to PyTorch/JAX
- **This is not a general-purpose visual scripting language.** Adding a node is subject to an RFC
- **We are not a certification body.** We are an evidence-collection tool (§27.1)
- **We do not build a cloud service or managed learning platform**
- **We do not promise a differentiable simulation.** Newton·Genesis already do this. It is an M5 experimental path

### 0.6 Target Tasks

**Primary (M1–M2): vision-based manipulation.** Franka/UR-class 7 DoF + parallel gripper, RGB 1–2 views (224×224 or 256×256) + joint state, ACT/Diffusion Policy/SmolVLA.

**Secondary (M3): vision-based locomotion.** Quadruped visual navigation, mobile robots.

**Tertiary (M4+): bimanual·humanoid.** Language-conditioned, the π₀ family.

For each robot group we define **1 benchmark scene + 3 standard tasks + 1 set of Evaluation Suite** (§10).

---

## 1. Development Model: Agent-Driven Development

Implementation is performed entirely by AI coding agents (Claude, Codex, etc.). Humans do not write code; they write specifications, design oracles, and adjudicate.

### 1.1 Premises

| | Human team | **Agent-driven** |
|---|---|---|
| Scarce resource | implementation time | **specification precision, oracle strength, review attention** |
| Failure mode | fails to finish | **finished but subtly wrong** |
| Scale limit | headcount | **context window, integration consistency** |
| Cost of rollback | high | low (regeneration is cheap) |

**Proposition 1: The quality ceiling of an agent's output is the strength of the oracle that adjudicates it.** A weak test *approves* weak code, so it is far more expensive than a human team.

**Proposition 2: Because rollback is cheap, exploration is cheap.** Creating 3 implementation candidates and choosing among them by benchmark is a normal tactic.

### 1.2 Work Packets

The minimum unit of planning is the packet. If it does not have the five elements, it is not a packet.

```
id / type / crate / depends
context      The range of files to touch. CI restricts the diff to this range
spec         What is to be implemented. There must be no ambiguity
oracle       An executable command that adjudicates pass/fail without a human
acceptance   Acceptance-criteria checklist
forbidden    What must not be done. Specifies the jurisdiction of adjacent packets
```

**A task for which an oracle cannot be written is an under-designed task.** Such an item is converted into not an implementation packet but a design-document packet.

### 1.3 Task Types

| Type | Definition | Human's role | Examples in this project |
|---|---|---|---|
| **A. Autonomous** | The specification is external and adjudication is mechanical | Write packets, sample review once a week | importers, serialization, Python bindings, IR round-trip, task converters, LeRobot datasets, CLI, test harnesses, sensor noise models |
| **B. Review** | The boundary is clear but design taste is involved | Write packets, **review the entire diff (≤800 lines)** | ECS, resource management, editor UI, telemetry, IR core types, Safety Plane implementation |
| **C. Design-delegated** | The human does the algorithm/type design and only the implementation is delegated | **Write the design document**, precisely review the result | IR type system, normalization·hash, Learning IR lowering, transcendental functions, deterministic accumulator, batch-domain scheduler |
| **D. Human-driven** | Failure is silent and diagnosis is exploratory | **Formulate hypotheses**, the agent runs experiments | numerical consistency, policy latency optimization, vision bandwidth tuning, vendor drivers, sim2real gap diagnosis |

- **A**: green-light auto-merge
- **B**: merge after diff review. If it exceeds 800 lines, require splitting
- **C**: `docs/design/<topic>.md` is a prerequisite deliverable. The agent does not create the design
- **D**: the packet is an "experiment," not an "implementation"

**The type composition of this project favors agent-driven development.** This is because it places the proprietary GPU physics solver (Type-D-heavy) in a lower priority and puts IR·data·deployment (Type A/B/C-heavy) up front.

### 1.4 Oracle-First Principle

**The validation harness comes before the implementation. No exceptions.**

The reference oracles this project has:

| Target | Oracle |
|---|---|
| Physics backend | MuJoCo (Python subprocess), MJWarp, Newton |
| **Learning IR lowering** | **PyTorch reference implementation — the result of running the same IR with PyTorch is the ground truth** |
| **Observation IR** | **Numerical match with LeRobot preprocessing, distribution comparison with real-hardware camera captures** |
| **Policy equivalence** | **Loading a LeRobot checkpoint and checking whether the same input yields the same action** |
| Transcendental functions | MPFR / `rug` |
| IR normalization | property test that hashes match after shuffling |
| Deterministic accumulator | bit-comparison after shuffling in arbitrary order |
| Renderer | golden image, PT/RS channel match |
| Safety Plane | exhaustive violation-scenario fixtures |

**It is important that the oracle for the Learning IR is PyTorch.** Whether the proprietary inference runtime loads LeRobot's ACT/SmolVLA checkpoint and produces identical output is the M1 gate. If this passes, "the IR represents a real policy" is proven.

### 1.5 Context Budget

```
A single core crate must fit within a single context window.
  target ≤ 6,000 lines / cap ≤ 10,000 lines (source, excluding tests)
  If it exceeds, a splitting packet is mandatory. No exceptions.
```

`cargo xtask context-budget` blocks in CI. Work that crosses a crate boundary is automatically Type B or higher, and a human finalizes the interface first.

### 1.6 Repository Structure

```
AGENTS.md / CLAUDE.md        root rules
docs/
  conventions.md             §3 conventions
  invariants.md              absolute invariants + machine/human check distinction
  design/                    Type C prerequisite design documents
  api-notes/                 actual signatures of ash·Slang·torch·LeRobot
  packets/<M>/               packet definitions + BACKLOG.md
crates/<name>/AGENTS.md      per-crate guidelines
xtask/                       single entry point for all validation
tests/golden/                read-only in CI
```

The full text is in Appendix C.

### 1.7 Agent Failure Modes and Defenses

| Failure mode | Concrete risk in v1.0 | Defense |
|---|---|---|
| Plausible wrong answer | **Learning IR lowering runs and the shapes match, but the numbers are subtly different → the policy silently degrades** | **PyTorch reference comparison as the M1 gate**. Bit-proximity comparison of output after checkpoint load |
| Hallucinated API | The `ash`, Slang, `torch` C++ API, and LeRobot schema are stale in the training data | Auto-generate `docs/api-notes/`, pin versions, forbid completion reports before `cargo check` |
| Fixing the test | Fitting golden images·golden tensors to the output | Goldens are read-only in CI. `xtask verify-goldens` checks git history |
| Silent scope creep | Refactoring on the side | `xtask check-scope` compares the diff against the packet range |
| Re-implementation | Own polynomial instead of `es-math::approx` → §3.4 determinism collapse | The "what already exists" list in the crate `AGENTS.md` + clippy lints |
| Over-abstraction | Trait proliferation for a single implementation | **Only 7 allowed extension points**: `PhysicsBackend`, `PolicyRuntime`, `TaskNodeFactory`, `LearningNodeFactory`, `InferenceBackend`, `Scalar`, `DeterministicAcc` |
| Cross-session drift | Inconsistency in naming·error types·allocation patterns | Root `AGENTS.md` conventions + weekly coherence audit |
| Determinism erosion | `HashMap` traversal, `f64` time accumulation, FP atomic summation | Blocked by types + clippy + `invariants.md` |
| **Safety bypass** | **Adding a code path that disables the Safety Plane "for testing convenience"** | **Design so that a Safety-Plane-disabled path does not exist. Mandatory pass at compile time (§9.4)** |
| Doc-code mismatch | Spec section-number references break | `xtask check-spec-refs` |

### 1.8 The Human's Role

**What is not delegated:** changes to the §0.5 non-goals and the Appendix A.1 finalized items / Type C design documents / Type D hypotheses·conclusions / oracle approval / gate adjudication / hash definition·public schema changes / **Safety Plane requirements definition** / weekly coherence audit.

**Review budget** (2–3 people, based on a 40-hour week): packet writing 8–12h / design documents 6–10h / diff review 6–10h / Type D 8–14h / audit 2–3h / management 2–3h.

Type A is not in this table. **A must exceed half of the total for this allocation to hold.**

### 1.9 Scope Reduction Order

1. **The entire proprietary physics solver** — the MJWarp/Newton backend is sufficient
2. Path tracer (M4)
3. 3DGS real-to-sim (M3)
4. IR-C (Control Graph)
5. Editable Visual Graph — keep read-only only
6. USD native reader — replaced by Bake
7. Multi-GPU heterogeneous configuration
8. NPU inference
9. Teleoperation OpenXR

**What is never cut:** the 5 IRs and their validators, the **Safety Plane**, the Evaluation IR, the hash chain, the oracle infrastructure, LeRobot compatibility.

---
## 2. Language and Toolchain

### 2.1 Pure Rust Core

No C++ in the core. C libraries are permitted via `bindgen`. The Slang compiler runs only at build time.

**One boundary that is relaxed on the learning path:** learning is delegated to PyTorch/JAX, so **on the learning path Python is a first-class dependency.** This is design, not compromise. It is enough that the core runtime and the real hardware deployment path work without Python.

```
Python required       learning (PyTorch), LeRobot datasets, USD Bake, xacro, MuJoCo reference
Python not required   sim execution, observation pipeline, policy inference, Safety Plane, real hardware deployment, editor
```

### 2.2 Core Crates

`ash` (Vulkan), `gpu-allocator`, `wide`/`multiversion` (SIMD), `crossbeam`, `loom`, `PyO3`+`maturin`, `wasmtime`, `egui`+`winit`, `egui-snarl` (node graph), `gltf`, `openusd`, `zenoh`, `quinn`, `serde`, `proptest`, `blake3`, `ort` (ONNX Runtime, optional), `safetensors`.

### 2.3 Shaders: Slang

A single shader language, SPIR-V output, offline compilation + content-hash cache. Uses of generics: precision parameterization (§3.3), **Observation IR node specialization** (§7.6), **Learning IR preprocessing kernel specialization** (§8.7).

### 2.4 Policy Inference Backends

Three implementations sit behind the `PolicyRuntime` trait.

| Backend | Use | When |
|---|---|---|
| **`TorchRuntime`** (libtorch FFI or Python subprocess) | in-loop inference during training, reference ground truth | M1 |
| **`OnnxRuntime`** (`ort`) | deployment, quantization validation, NPU EP | M2 |
| **`VulkanRuntime`** (Slang kernels + `cooperative_matrix`) | massively parallel small policies, vendor-neutral deployment | M3 |

**Vulkan Compute inference alone cannot run π₀ 3.5B.** Large VLAs are handled by `TorchRuntime`/`OnnxRuntime`, and the Vulkan path handles ACT-class (52M) small policies and vendor-neutral on-robot deployment.

### 2.5 External Tool Dependencies

| Feature | Dependency | When | Required? |
|---|---|---|---|
| Slang → SPIR-V | Slang compiler | build | developers only |
| observation/preprocessing GPU lowering | runtime Slang | compile time | **not needed on cache hit** |
| learning | Python + PyTorch | learning time | learning workflow only |
| large VLA inference | libtorch or ONNX Runtime | runtime | that policy only |
| LeRobot datasets | Python (reading is Rust-native) | interop | export only |
| USD Bake / xacro | Python | import | that format only |
| PyTorch zero-copy | CUDA/HIP driver | runtime | §21 capability |

`es --check-deps` prints the range of what is possible per environment.

---

## 3. Conventions, Precision, and Determinism Contract

### 3.1 Coordinate System and Unit Conventions

| Item | Convention |
|---|---|
| Coordinate system | right-handed, **Z-up**, X-forward (ROS/MJCF/URDF) |
| length/angle/mass/time | m, rad, kg, s |
| Quaternion | **xyzw**, unit norm, w ≥ 0 |
| Inertia | body-frame 3×3 symmetric, principal-axis decomposition cached |
| **Image coordinates** | **origin top-left, x right, y down (OpenCV convention)** |
| **Camera coordinates** | **+Z forward, +X right, +Y down (OpenCV). Same as the ROS `REP-103` optical frame** |
| **Color space default** | **sRGB (nonlinear). Linear conversion only via an explicit node** |

The last three rows are the core conventions of the vision pipeline. Without coordinate/color-space conventions, reimplementing preprocessing silently diverges. `ImageSpec` in §7 carries these in the type.

### 3.2 Transcendental Functions

SPIR-V `GLSL.std.450` only specifies a ULP upper bound; the exact result differs per vendor and driver. `es-math` provides its own minimax polynomials, and Rust and Slang share the coefficients and operation order. Accuracy target ≤ 2 ULP, reference is MPFR.

Calls to standard transcendental functions in physics, observation, and reward kernels are blocked with a clippy lint. **Neural network kernels are an exception** (§8.7). Bit reproducibility of activation functions depends on the policy weights and the inference backend, so a separate contract is used.

### 3.3 Precision

| Hardware | `shaderFloat64` | GPU physics | observation pipeline |
|---|---|---|---|
| NVIDIA Turing+ | supported (1/64) | possible | possible |
| AMD RDNA2+ | supported (1/16–1/32) | possible | possible |
| Intel Arc Alchemist | emulation | DoubleFloat only | possible |
| Intel Xe2 Battlemage | supported | possible | possible |
| Apple / MoltenVK | unsupported | not possible (CPU) | possible |

- Scalar representation is a type parameter: `F64Native` / `DoubleFloat` / `F32`. The default is finalized by M2 measurement
- **The observation/learning pipeline defaults to FP32 with optional FP16/BF16 support.** This is because the policy is trained at that precision
- Rendering FP32, camera-relative coordinates

### 3.4 Deterministic Execution Contract

Vulkan float-controls items such as `shaderDenormFlushToZeroFloat32` are **device capability query values indicating whether the feature is supported**, not settings that force behavior. Actual application is done via SPIR-V execution mode/control.

**The correct 4 steps**

```
1. Capability Query          query VkPhysicalDeviceFloatControlsProperties
        ↓
2. Required Capability Check reject deterministic mode if a required item is unsupported
        ↓
3. SPIR-V Execution Mode     specify DenormFlushToZero / RoundingModeRTE / SignedZeroInfNanPreserve
                             in the shader. NoContraction decoration
        ↓
4. Deterministic Pipeline    an execution plan that satisfies the entire contract below
```

**Deterministic execution contract = all of the following**

```
float controls        (capability check + execution mode specified)
+ no contraction      (NoContraction decoration)
+ deterministic reduction   (RFA/binned summation, §18.4)
+ deterministic scheduling  (static partitioning, single queue)
+ fixed subgroup behavior   (subgroup size logically fixed)
+ fixed algorithm           (no branching derived from device properties)
+ fixed compiler/SPIR-V     (Slang version + SPIR-V content hash)
+ fixed driver/device       (tier 1 condition, §3.5)
```

**`NoContraction` alone does not guarantee deterministic GPU execution.** All eight items are required.

### 3.5 Determinism Tiers

| Tier | Condition | Guarantee |
|---|---|---|
| **0. Semantically identical** | same `*_hash` | the definitions are the same (not a numeric guarantee) |
| **1. Bitwise** | same `execution_hash` + same device/driver | bit-exact match |
| **2. Cross-backend** | CPU ↔ GPU, or backend swap | defined tolerance |
| **3. Physical meaning** | vs MuJoCo, vs real measurement | physics-metric tolerance |
| **4. Policy equivalence** | **run the same IR with PyTorch vs the in-house runtime** | **action tensor tolerance** |

Tier 4 is defined in §8.9.

**Prohibited in deterministic mode:** atomic FP summation, fast-math, workgroup count derived from device properties, floating-point time accumulation, standard transcendental functions in physics/observation kernels, multiple compute queues for physics kernels, global RNG, dependence on `HashMap` iteration order, **bypassing the Safety Plane**.

---

## 4. System Architecture

### 4.1 Plane Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│ AUTHORING PLANE                                                      │
│  Python │ TOML │ Visual Graph │ LLM generation │ external conversion(LeRobot/IsaacLab)│
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ TASK IR                 §6                                           │
│  scene reference / goal / reward / termination / randomization / reset / ObservationSpec declaration  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OBSERVATION IR          §7        sensor → tensor                    │
│  ImageSpec / camera synchronization / crop·resize·normalize / color-space conversion      │
│  TemporalWindow / masking / state fusion                                 │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LEARNING IR             §8        tensor → action chunk              │
│  VisionEncoder / StateEncoder / LanguageEncoder / Fusion             │
│  TemporalEncoder / PolicyHead(Regression·Diffusion·FlowMatching)     │
│  ActionChunker / ActionUnnormalizer                                  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ POLICY RUNTIME          §2.4                                         │
│  Torch │ ONNX │ Vulkan │ NPU     async inference / chunk buffer / latency budget   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ DEPLOYMENT IR + SAFETY PLANE    §9                                   │
│  torque·velocity·workspace limits / collision constraints / rate limit / action validity        │
│  NaN watchdog / inference deadline / stale observation / E-stop / fallback controller │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
                    CONTROL / ACTUATOR  →  ROBOT (or sim)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Piercing all layers:  hash · version · provenance · telemetry · replay · evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────────────────────────────────────────────────────────┐
│ EVALUATION IR           §10    perturbation suite × metric × acceptance criteria │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ EXECUTION SUBSTRATE                                                  │
│  PhysicsBackend (MJWarp │ Newton │ PhysX │ in-house)  §17            │
│  Vision Data Plane (render → observation capture)  §15              │
│  ECS · jobs · time model · failure semantics       §5, §18          │
│  GPU resources · TensorTransport                   §21              │
│  asset pipeline (USD/glTF/MJCF/URDF, 3DGS)         §16              │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Crate Layering

| Layer | Crate |
|---|---|
| 0 | `es-math` |
| 1 | `es-core` (ECS, jobs, time, failure semantics) |
| 2 | `es-gpu`, `es-assets`, `es-usd` |
| 3 | `es-actuator`, `es-sensor`, `es-physics-core` |
| 4 | `es-physics-backend` (MJWarp/Newton/PhysX adapters), `es-physics-cpu`, `es-physics-gpu` |
| 5 | `es-render`, `es-splat` (3DGS) |
| 6 | **`es-ir`** (the 5 IR schemas, type system, normalization, hash, validation) |
| 7 | **`es-compile`** (scheduling, lowering, backend codegen) |
| 8 | **`es-policy`** (PolicyRuntime: Torch/ONNX/Vulkan), **`es-safety`** (Safety Plane) |
| 9 | `es-env` (execution orchestration, batch domain) |
| 10 | `es-data`, `es-telemetry`, `es-eval` |
| 11 | `es-ros2`, `es-py`, `es-script`, `es-transport` |
| 12 | `es-editor` |

**Rules (CI-enforced)**
1. Upper → lower only. Dependency between crates in the same layer is forbidden
2. `es-physics-cpu` ⇎ `es-physics-gpu`. Sharing goes through `es-physics-core`
3. Layers ≤2 do not know Vulkan symbols (`es-gpu` exception)
4. **No crate depends on `es-editor`**
5. The core does not link CUDA/HIP symbols. Only `es-transport` is the exception
6. **`es-ir` does not know the compiler, backends, or PyTorch.** IR is neutral with respect to the means of execution
7. **`es-ir` does not know UI types.** Layout is an `.eslayout` sidecar
8. **`es-safety` does not depend on `es-policy`.** Safety does not trust the policy. It flows in only one direction
9. Crate source ≤ 10,000 lines (§1.5)

**Rule 8 is one of the core structural decisions of v1.0.** If the Safety Plane depended on the Policy layer, the policy's assumptions would leak into safety validation.

### 4.3 PhysicsBackend

**An external physics backend is the default path.**

```
PhysicsBackend
├── MuJoCoWarpBackend    default. GPU parallel, MJCF native            M1
├── NewtonBackend        when Kamino·VBD·hydroelastic are needed       M2
├── MuJoCoCpuBackend     deterministic reference, CI oracle            M1
├── PhysXBackend         Isaac Lab asset compatibility                 M3
└── NativeBackend        in-house CPU/Vulkan solver (optional, §17.4)  M4+
```

Backends declare a capability set, and the compiler checks it against IR requirements (§11.6). Backends **also declare a determinism grade**, and external backends do not declare tier 1.

**Effects of this shift**
- In M1–M3, physics engine development is off the critical path. Newton's progress becomes a direct benefit
- Instead, **semantic mapping between backends** (§17.2) becomes a new core task. The same Task IR must not behave differently on MJWarp and Newton
- This is why item 1 in the reduction order of §1.9 is "the entire in-house physics solver"

---
## 5. IR Family

### 5.1 Boundary Definitions

The responsibilities of the five IRs are clearly divided. **If the boundaries blur, Task IR bloats like a Blueprint.**

| IR | Question | Owns | Batch semantics |
|---|---|---|---|
| **Task IR** | What happens in the world and what is good | Scene references, reward, termination, randomization, reset, ObservationSpec **declaration** | env batch enforced |
| **Observation IR** | How sensor outputs become tensors | ImageSpec, synchronization, geometric transforms, normalization, time windows | env batch by default, camera axis allowed |
| **Learning IR** | How tensors become action chunks | Encoders, fusion, temporal models, policy heads, action decoding | **batch free. Decoupled from env** |
| **Deployment IR** | Under what constraints actions execute | Safety limits, execution modes, deadlines, fallbacks | per robot |
| **Evaluation IR** | How goodness is judged | perturbation suite, metrics, acceptance criteria | episode batch |

**Task IR only declares `ObservationSpec` and does not implement it.** "The forward camera RGB and joint positions are the observation" belongs to Task IR, and "resize it to 224×224, normalize with ImageNet statistics, and stack 2 frames" belongs to Observation IR.

It must be possible to attach different Observation IRs to the same Task IR. ACT uses `n_obs_steps=1` while Diffusion Policy uses an observation window, but the task is identical.

### 5.2 Differential Application of Batch Semantics

Every task node has env batch semantics. **However, this principle is not applied to Learning IR.**

```
simulation batch      4,096 env          physics step
     ↓ (active cameras only)
observation batch     1,024 env × 1 view  render + preprocessing
     ↓ (inference batch)
inference batch       256 samples         policy inference
     ↓ (training minibatch)
training batch        64 sequences        gradient step
```

The four domains may have different sizes, periods, and devices. §12 defines the orchestration.

### 5.3 Hash Scheme

```
asset_hash ──┐
             ├──► scene_hash ──┐
             │                 │
             │   task_graph_hash (authoring identity, includes node IDs)
             │           ↓
             └──►    task_hash    (semantic identity, after normalization)
                         │
        observation_hash ┤
           learning_hash ┤
             policy_hash ┤   (weights + architecture + base model)
            dataset_hash ┤   (content + schema + split)
          deployment_hash┤
          evaluation_hash┘
                         ↓
                  compiler_hash · runtime_hash · hardware_capability
                         ↓
                  ┌──────────────────┐
                  │  execution_hash  │
                  └──────────────────┘
```

```
execution_hash = H(
    task_hash, observation_hash, learning_hash, policy_hash,
    dataset_hash, deployment_hash,
    compiler_hash, runtime_hash, hardware_capability
)
```

**`execution_hash` is the condition for §3.5 Tier 1.** And because it is clear which hash changes when something changes, the impact scope of a "substantive change" can be judged mechanically (§27.1).

Why `dataset_hash` is split into three: even for the same dataset, **a different split produces a different training sample composition.** If the train/val/test split is not hashed, reproduction does not hold.

### 5.4 Common Type System

The port types shared by the five IRs.

```
PortType = (ElemType, Shape, Semantics)

ElemType    f32 | f16 | bf16 | f64 | i32 | u8 | bool
Shape       [D1, D2, ...]           the batch axis is decided by the domain (§5.2)
Semantics   Unit × Frame × TimeRef  (+ images have ImageSpec, §7.2)
```

**Unit** — physical unit algebra

```
Dimensionless | Length(m) | Angle(rad) | Mass(kg) | Time(s)
Velocity | AngularVelocity | Acceleration | Force | Torque
Pressure | Current | Voltage | Quaternion | RotationMatrix
Normalized(lo, hi)            normalized value
Pixel | Luminance | Depth(m)  image domain
Token                          language / discrete representation
```

Multiplication and division follow the unit algebra, and addition requires matching units. **A policy input port only accepts `Normalized`, `Dimensionless`, or `Token`.** The common mistake of feeding un-normalized raw values into a neural network is caught at compile time.

**Frame**

```
World | LocalOrigin | Body(id) | Sensor(id) | Joint(id)
Camera(id)            camera optical frame (§3.1)
Image(id)             pixel coordinates
Policy                policy-internal representation (no frame checking)
```

**TimeRef**

```
Tick(PhysTick)                        current physics tick
Sensor { id, align }                  align = Hold | Interpolate | Reject
Window { base, n_steps, stride }      TemporalWindow (§7.5)
```

A combination in which the time alignment is not specified is a compile error.

```
ERROR TYPE-014  time alignment not specified

  Inputs of Concat "policy_obs":
    joint_pos     TimeRef = Sensor(encoder, 1000 Hz)
    camera_front  TimeRef = Sensor(cam_front, 30 Hz)

  Resolution: time_align = "hold" | "interpolate" | "reject"
              (ACT / Diffusion Policy typically use "hold")
```

---

## 6. Task IR

### 6.1 Scope

Contains only the semantics of the world and the task.

```
Allowed   scene references, goals, reward terms, termination conditions, domain randomization, reset distributions,
          curriculum, sensor schedules, ObservationSpec declaration, dataset recording targets

Forbidden direct ECS mutation, physics solver internals, blocking, GPU synchronization, threads,
          raw device access, file/network I/O, neural networks, preprocessing implementation
```

Forbidding is implemented not as "block it at runtime" but as **"there is no such node in the IR."**

### 6.2 IR-D / IR-C Separation

| | **IR-D (Dataflow)** | **IR-C (Control)** |
|---|---|---|
| Meaning | pure DAG, no side effects | sequencing, branching, subtasks |
| Nodes | observation, reward, termination, randomization, reset | `Sequence`, `Branch`, `SubTask`, `Repeat` |
| Schedule | **M0 schema / M1 lowering** | **M4** |

Standard vision manipulation tasks (pick&place, reach, push) are expressed with IR-D alone. What requires IR-C is multi-stage assembly, and that is M4.

### 6.3 Node Set

**Source** (phase = Observation)
```
GetJointState  GetBodyPose  GetBodyVelocity  GetContact
GetSensor      GetTime      GetRandom(stream required)
GetLanguage    task instruction (for VLA, added in v1.0)
```

**Transform** (pure functions)
```
Transform  Normalize  Clamp  Arith  MathFn  Norm  Dot  Cross
Compare    Logic      Select  Concat  Slice  Reduce
```

**Sink**
```
ObservationSpec   observation declaration (implemented in Observation IR)
ActionSpec        action space declaration (§9.2)
Reward            reward term (name, weight, aggregation)
Terminate         success | failure | timeout
Randomization     target parameters + distribution
ResetState        initial state distribution
Record            dataset recording target
```

**A sensor's render path.** A `Sensor` source declares how the simulation produces it with `render = { path = "rs" | "pt", spp, bounces, exposure, tonemap }` (packet M7/R5). The default is `rs`, and **absent = default = today's canonical form**, so no committed `task_hash` moves; `pt` moves `task_hash` and is therefore a new document (§13.3). The Observation IR does not know about it — what the sensor *is* belongs to `ImageSpec`, how the simulation *makes* it belongs to the Task IR.

`Parallel` does not exist. Parallelization is decided by the compiler. `Wait` / `Repeat` / `Condition` are replaced by IR-C or `Compare`+`Select`.

### 6.4 Execution Semantics

The execution order is determined by **phase → explicit data dependency → tick schedule → node ID**. The graph placement position does not affect the semantics.

```
PrePhysics → Physics → PostPhysics → Observation → Reward → Termination → Record
```

A backward dependency across phases is a compile error. Every node implicitly receives the active `EnvMask` as input, and isolated envs (§18.5) are excluded from computation.

### 6.5 Expressions

The syntax is kept Rhai-like, but **the parser of `es-ir` reads it and expands it into an IR subgraph.** The Rhai interpreter is not invoked.

```
User input:
    norm(pose("gripper").pos - pose("cube").pos) < 0.02

Expansion result:
    GetBodyPose("gripper") ─┐
                            ├─ Arith(-) ─ Norm(L2) ─ Compare(<, 0.02) ─ bool[env]
    GetBodyPose("cube")   ─┘
```

The supported functions correspond 1:1 to the §6.3 nodes. Loops, variable assignment, and function definitions are parse errors. Rhai remains only on low-frequency paths such as editor automation and offline processing.

### 6.6 Determinism Rules

| Code | Rule |
|---|---|
| `DET-001` | Global RNG forbidden. `TaskRng(seed, EnvId, PhysTick, stream)` required |
| `DET-002` | Wall-clock access forbidden (structural) |
| `DET-010` | Standard transcendental functions forbidden. Only `es-math::approx` |
| `DET-020` | Unstable iteration order forbidden |
| `DET-021` | Undeclared RNG stream warning |
| `DET-030` | `Reduce(unordered=true)` rejected in deterministic mode |
| `DET-040` | Tier 1 declaration not possible when using an external backend |

**`sqrt` is not a transcendental (packet M8/S4d, 2026-09-21).** IEEE 754 requires a correctly rounded square root, so `f64::sqrt` is a basic arithmetic operation that is bit-identical on every platform and is not what `DET-010` names. A reward or termination cone's `Norm{L2}` lowers to `Expr::Sqrt`; `exp`, `log` and the trigonometric functions remain `MathFn{approx = true}` only.

---
## 7. Observation IR

### 7.1 Why a Separate Layer

Sensor outputs are assigned `ElemType, Shape, Unit, Frame, TimeRef`, and `GetSensor` uses these. **However, for images this is not enough.**

The simulator already models lens distortion, rolling shutter, motion blur, exposure, and shot noise (§18.3). Yet that information was not passed to the learning input. As a result, two problems arise.

1. **Preprocessing gets reimplemented in Python.** The simulator uses the LeRobot transforms, while real hardware uses separate code. At that moment sim and real diverge and tracing the cause becomes impossible
2. **Camera parameters do not travel with the policy.** The intrinsics of the real hardware camera differ from those of the simulator, but the policy does not know it

Observation IR **describes the entire span from the sensor to the policy input tensor as a single typed graph**, and executes it identically in simulation, in the dataset, and on real hardware.

### 7.2 `ImageSpec`

An image port carries the following contract.

```rust
pub struct ImageSpec {
    pub width: u32,
    pub height: u32,
    pub channels: ChannelFormat,   // Rgb | Rgba | Gray | Depth | Seg | Normal | Flow
    pub dtype: ImageDType,         // U8 | U16 | F16 | F32
    pub color_space: ColorSpace,   // SRgb | Linear | Rec709 | Raw
    pub camera_model: CameraModel, // Pinhole | Fisheye | Equirect | OrthoDepth
    pub intrinsics: Intrinsics,    // fx fy cx cy (+ skew)
    pub extrinsics: Transform,     // T_body_camera, used for Frame checking
    pub distortion: DistortionModel, // None | BrownConrady(k1..k3,p1,p2) | KannalaBrandt
    pub shutter: ShutterModel,     // Global | Rolling { readout: Duration, dir }
    pub exposure: Duration,
    pub rate_hz: f32,
    pub depth_scale: Option<f32>,  // unit of the Depth channel (m/LSB)
}
```

**What the compiler checks**

```
ERROR OBS-021  color space mismatch

  VisionEncoder "dinov2" expects Linear input.
  camera_front is ColorSpace::SRgb.

  Fix: insert a ColorTransform(SRgb → Linear) node, or
       change the encoder normalization spec to an sRGB basis.
       (ImageNet statistics are on an sRGB basis)

ERROR OBS-034  intrinsic not reflected in the resize

  After Resize(640×480 → 224×224), the CameraProjection node
  uses the original intrinsic.

  Fix: set rescale_intrinsics = true so the Resize node
       automatically scales the intrinsic.
```

The second is a bug that occurs especially often in practice. If the intrinsic is not updated after a resize or crop, 3D inference is silently wrong. **The fact that the `Resize` and `Crop` nodes transform the `Intrinsics` is built into the type system.**

### 7.3 Node Set

**Input**
```
ImageInput(sensor_id)      fetches the ImageSpec from the sensor registry
StateInput(spec)           vector state such as joints, IMU, F/T
LanguageInput(source)      task instruction or external input (VLA)
```

**Geometric**
```
Resize(w, h, filter, rescale_intrinsics)
Crop(rect | center | random, rescale_intrinsics)
Pad / Undistort / Rectify / Warp(homography)
CameraProjection           3D point ↔ pixel. consumes intrinsic and extrinsic
```

**Photometric**
```
ColorTransform(src → dst)  sRGB ↔ Linear ↔ Rec709
ToGray / ChannelSelect
Normalize(mean, std | range)   Unit → Normalized
QuantizeU8 / Dequantize
```

**Temporal**
```
TemporalWindow(n_steps, stride, align)   learning input semantics (§7.5)
FrameStack(n)                            channel-axis concatenation (special case)
Delta(n)                                 frame difference (event-like)
```

**Structural**
```
Concat(axis, time_align)   Stack(axis)   Mask(source)
MultiViewPack(cameras)     multiple cameras → [V, C, H, W]
```

**Augmentation** (active only during training, inactive during evaluation)
```
RandomCrop / ColorJitter / RandomErasing / GaussianNoise
```

Augmentation nodes have a `training_only = true` flag. **When executed as Evaluation IR they are automatically disabled, and this is guaranteed at the IR level.** It structurally prevents the accident of running evaluation with augmentation left on.

### 7.4 Connection to `ObservationSpec`

The `ObservationSpec` declared by the Task IR becomes the input contract of the Observation IR.

```
Task IR
  └── ObservationSpec
        ├── "rgb_front"    : Sensor("cam_front"), ChannelFormat::Rgb
        ├── "rgb_wrist"    : Sensor("cam_wrist"), ChannelFormat::Rgb
        └── "joint_state"  : JointState("franka"), 7 DoF + gripper
              │
              ▼
Observation IR
        ├── ImageInput(cam_front) → Resize(224) → ColorTransform → Normalize
        ├── ImageInput(cam_wrist) → Resize(224) → ColorTransform → Normalize
        ├── StateInput(joint)     → Normalize(joint_limits)
        └── TemporalWindow(n=2, align=Hold) → Concat
              │
              ▼
        policy_input : { rgb: f32[2, 2, 3, 224, 224], state: f32[2, 8] }
                              ↑  ↑                         ↑
                           time  view                    time
```

**Multiple Observation IRs can be attached to the same Task IR.** One for ACT (`n_steps=1`), one for Diffusion Policy (`n_steps=2`), and one for π₀ (3 views + language) share the same task. The `task_hash` is the same and only the `observation_hash` differs.

### 7.5 Three-Layer Time Model

If time is handled with a single `History<T,N>`, the signal-processing buffer, the learning input semantics, and the neural network time representation are not distinguished. **So the three are separated.**

```
History<T, N>          system-level buffer
   = ring buffer. models sensor latency, holds the most recent N samples. outside Observation IR.
   owned by: es-core, es-sensor

TemporalWindow(n, stride, align)    learning input semantics
   = "the policy sees the most recent 2 frames at 200ms intervals"
   owned by: Observation IR
   affects: tensor shape, memory budget, dataset sampling

TemporalEncoder(kind, params)       neural network time representation
   = Transformer / TemporalConv / GRU / none (single frame)
   owned by: Learning IR
```

```
FrameStack(4)            and   TemporalTransformer(8 frames)
have completely different meanings.
The former is preprocessing that quadruples the channel axis,
the latter is an architecture choice that applies attention over an 8-token sequence.
```

This distinction is what makes it possible to express ACT (`n_obs_steps=1`, no time encoder), Diffusion Policy (observation window + time-series conditioning), and π₀ (VLM token sequence) with the same IR.

### 7.6 Execution

- **Simulation:** the render output (§15) stays resident on the GPU while the preprocessing kernels execute. No host round-trip
- **Dataset:** the `ImageSpec` is stored together at record time, and the same IR is executed on CPU or GPU during training
- **Real hardware:** the same IR is executed on the camera driver output. **`es-runtime-embedded` includes the Observation IR evaluator** (§9.6)

Because the three paths execute the same IR, preprocessing mismatches do not structurally occur. Validation: for the same input, the outputs of the three paths are bit-identical (integer arithmetic) or within tolerance (floating point).

### 7.7 Oracle

| Item | Validation |
|---|---|
| LeRobot preprocessing equivalence | LeRobot transform vs Observation IR on the same image → within tolerance |
| intrinsic transform correctness | 3D→2D projection error < 0.1 px after resize and crop |
| color space round-trip | sRGB → Linear → sRGB lossless |
| real-hardware distribution comparison | channel statistics and frequency spectrum of real-hardware camera capture vs simulator render |
| three-path agreement | outputs of the simulation, dataset, and real-hardware paths agree |

---
## 8. Learning IR

### 8.1 Purpose and Boundaries

**Do not put neural networks into Task IR.** If Task IR inflates like a Blueprint, the very problem §0.5 guards against occurs. Learning IR is a separate IR.

```
Allowed   encoder selection·configuration, fusion method, temporal model, policy head, action decoding,
          normalization statistics, chunk·horizon settings, pretrained backbone reference

Forbidden layer-level network authoring (that is what PyTorch does)
          arbitrary tensor operation graphs
          learning loop control (optimizer·scheduler are training config)
```

**Core principle: the network internals are opaque, the interface semantics are typed.**

```
                LearningGraph
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   Preprocessor              PolicyHandle (opaque)
   (fully described in IR)           │
                        ┌───────────┼───────────┬──────────┐
                        ▼           ▼           ▼          ▼
                       ACT     Diffusion    FlowMatch    VLA
```

`PolicyHandle` does not know the internals of the weights, but **it must have a metadata contract.** That contract is what Learning IR type-checks.

### 8.2 Structure

```rust
pub struct LearningGraph {
    pub schema_version: u32,
    pub inputs: Vec<TensorPort>,     // contract with Observation IR output
    pub nodes: Vec<LearningNode>,
    pub outputs: Vec<TensorPort>,    // contract with ActionSpec
    pub policy: PolicyHandle,
}
```

### 8.3 Node Set

**Encoder**
```
VisionEncoder { backbone, pretrained, frozen, out_dim, token_count }
    backbone = ResNet18 | ResNet34 | ViT{size} | DINOv2 | SigLIP | SmolVLM | Custom(hash)
StateEncoder { kind, out_dim }          MLP | Identity
LanguageEncoder { tokenizer, model, max_len }
```

**Fusion**
```
Concat | CrossAttention | FiLM | AdaLN | TokenConcat
```

**Temporal**
```
TemporalEncoder { kind, n_frames, ... }
    kind = None | TemporalConv | Transformer | GRU | Mamba
```

**Head**
```
RegressionHead   { horizon, action_dim }              ACT, BC
DiffusionHead    { horizon, n_steps, scheduler }      Diffusion Policy
FlowMatchingHead { horizon, n_steps }                 π₀, SmolVLA
DiscreteHead     { vocab, horizon }                   VQ-BET, RT-2 family
EnergyHead       { ... }                              IBC family
```

**Action**
```
ActionChunker    { horizon H, execute_chunk K, replan_hz }
ActionUnnormalizer { stats_source }       dataset statistics or explicit
```

**Reference**
```
PolicyBundle(uri | hash)   references the whole thing as a single opaque bundle (large VLAs such as π₀)
```

The last one matters. **π₀ 3.5B is not decomposed into nodes.** It is referenced whole via `PolicyBundle`, but its input/output contract (§8.4) is type-checked. Conversely, an ACT-class model is composed as nodes, so experiments such as backbone swapping·freezing are possible at the IR level.

### 8.4 Policy Metadata Contract

```yaml
policy:
  architecture: act
  base_model: null                       # or "lerobot/pi05_base"
  base_model_hash: "b3:9c4e..."
  input:
    rgb_front:   { shape: [3, 224, 224], dtype: f32, space: normalized_imagenet }
    rgb_wrist:   { shape: [3, 224, 224], dtype: f32, space: normalized_imagenet }
    joint_state: { shape: [8], dtype: f32, space: normalized_joint_limits }
    language:    null
  temporal:
    observation_window: 1                # ACT default
  output:
    action_dim: 8
    horizon: 50                          # prediction length
  control:
    execute_chunk: 20                    # execution length
    replanning_hz: 10
    execution_mode: receding_horizon
  runtime:
    dtype: f32
    expected_latency_ms: 5.0             # RTX 4090 basis (§0.3)
    deadline_ms: 50.0                    # §9.4 fallback if exceeded
```

**What the compiler checks**
- Match between Observation IR output shape·normalization space and `input`
- Match between `action_dim` of `ActionSpec` (Task IR) and `output`
- `execute_chunk ≤ horizon`
- Integer relationship between `replanning_hz` and `dt_ctrl`
- **Whether `deadline_ms` fits within the control cycle** — if not, a Safety Plane fallback strategy is mandatory (§9.4)

The last check is important in practice. Diffusion Policy is 370 ms on the RTX 4090. Putting it into 10 Hz control (100 ms) exceeds the deadline every step. This must be revealed at compile time.

```
ERROR LRN-052  inference latency exceeds the control cycle

  policy:      diffusion_policy (263M)
  measured latency: 369.8 ms (RTX 4090)
  control cycle: 100 ms (10 Hz)
  execute_chunk: 20 → effective replanning cycle 2000 ms

  this configuration requires one of the following:
    (a) async inference + chunk buffering (§8.6) — requires execute_chunk ≥ 4, currently 20 OK
    (b) specify a Safety Plane fallback policy (§9.4)
    (c) a faster policy (ACT 5.0 ms, SmolVLA 99.2 ms)

  currently (a) is satisfied but no (b) fallback is specified.
```

### 8.5 Action Modeling

The `policy(obs) → action` model is not used.

```
obs_t
  ↓
policy
  ↓
[a_t, a_{t+1}, ..., a_{t+H-1}]     action chunk, H = horizon
  ↓
execute first K                     K = execute_chunk
  ↓
receding horizon replan
  ↓
controller (dt_ctrl)
```

```rust
pub struct ActionSpec {
    pub space: ActionSpace,        // JointPosition | JointVelocity | JointTorque
                                   // | EEPose | EEDelta | Gripper | Composite
    pub dim: usize,
    pub horizon: usize,            // prediction length H
    pub execute_chunk: usize,      // execution length K ≤ H
    pub control_rate_hz: f32,
    pub execution_mode: ActionExecutionMode,
    pub normalization: NormalizationSpec,
    pub limits: ActionLimits,      // consumed by the Safety Plane (§9)
}

pub enum ActionExecutionMode {
    OpenLoopChunk,        // replan after executing the entire chunk
    RecedingHorizon,      // replan after executing K (default)
    TemporalEnsemble,     // ACT method: exponentially weighted average of overlapping predictions
    RealTimeChunking,     // compute the next chunk while executing the previous one (§8.6)
}
```

`TemporalEnsemble` is what ACT actually uses, so it is treated as first-class. Weighted-averaging the multiple predictions at overlapping time points is describable at the IR level, and it must execute identically on real hardware and in simulation.

**Execution semantics of the incremental action space (plan T, §28.12).** `ActionSpace::JointDelta` (new) and `EeDelta` say the policy output is an *increment on the current target*. The runtime integrates it in one place (`es-env`) — `target_t = target_{t−1} + Δ_t`, and the integration state resets at episode boundaries to the measured pose the plane seeds (§9.3) — and the Safety Plane validates the **absolute target** as it does today. The envelope does not change, and per-tick change shrinks because policy noise lands on the increment. An incremental policy's `Normalizer{Inverse}` statistics are in increment units (rad/tick), and the adapter document declares that (§14.4). Absolute-target policies (ACT, brax/rsl_rl's `default_pose + a·scale`) keep `JointPosition` as is — the increment is an addition, not a replacement.

### 8.6 Async Inference and Chunk Buffer

Since inference latency being larger than the control cycle is the normal situation, the runtime treats this as first-class.

```
control thread (dt_ctrl = 100 ms)
   t=0    start executing chunk A[0..20], output A[0]
   t=100  output A[1]
   ...
   t=400  output A[4]  ← at this point the inference thread starts computing B with obs_400
   ...
   t=770  B arrives (took 370 ms)
   t=800  switch to B[0] instead of A[8]?  ← a switching policy is needed

inference thread (async)
   capture obs_t → policy inference → chunk arrives → buffer replacement
```

**Switching policy** (`ChunkBlendPolicy`)
```
HardSwitch        replace immediately. discontinuity possible
LinearBlend(n)    linear blend over n steps
TemporalEnsemble  weighted average of the overlapping region (ACT method)
```

**Chunk exhaustion (buffer underrun) is a safety event.** If the chunk runs out and no new chunk has arrived, the Safety Plane intervenes (§9.4). This event is recorded by a counter and becomes a metric in the Evaluation IR (§10.3).

### 8.7 Lowering

```
Preprocessor portion (Observation IR + the encoder front-end of Learning IR)
    → compiled to a Slang kernel. GPU-resident, no host round-trip
    → on real hardware, the CPU/NPU kernel of es-runtime-embedded

PolicyHandle portion
    → delegated to the PolicyRuntime (§2.4)
    → Torch: libtorch or subprocess
    → ONNX: ort
    → Vulkan: Slang + cooperative_matrix (only small, ACT-class)

ActionChunker / Unnormalizer / switching policy
    → deterministic CPU code. identical on real hardware·simulation
```

**The boundary is clear.** Preprocessing and postprocessing are fully owned by the IR and executed deterministically, and only the network forward pass is delegated to the runtime. This is the implementation of §8.1's "internals opaque, interface typed".

### 8.8 Connection to Learning

Learning IR **does not own the learning loop.** PyTorch does. Instead, it generates the following.

```
es learn export --ir task.toml --obs obs.toml --learning policy.toml
  →  lerobot_config.yaml       LeRobot training config
  →  dataset_spec.json         dataset schema + split definition
  →  preprocess.py             PyTorch mirror of the Observation IR (for validation)
  →  policy_stub.py            PyTorch composition of the LearningGraph
  →  training.lock             §19.3 learning identity
```

**`preprocess.py` is a reference implementation, not the ground truth.** The ground truth is the IR, and this file is generated so that the training framework performs the same preprocessing. Output agreement between the two paths is a CI gate (§7.7).

### 8.9 Level 4: Policy Equivalence (§3.5)

**Whether the actions agree when the same Learning IR is executed with different `PolicyRuntime`s.**

| Comparison | Tolerance |
|---|---|
| PyTorch(fp32) ↔ ONNX(fp32) | action max absolute error ≤ 1e-5 |
| PyTorch(fp32) ↔ ONNX(int8) | defined per task. success-rate drop ≤ 2%p |
| PyTorch(fp32) ↔ Vulkan(fp32) | ≤ 1e-4 (small policies only) |
| same runtime re-execution | bit-exact (when using deterministic kernels) |

**M1 gate: load LeRobot's ACT checkpoint and produce the same action for the same observation.** This is the proof of "the IR represents a real policy".

---
## 9. Deployment IR and Safety Plane

### 9.1 Principles

> **The policy is not trusted.** Safety is enforced outside the policy, independently of the policy, and deterministically.

```
Policy
  ↓  action chunk
┌─────────────────────────────────────────┐
│ SAFETY PLANE                            │
│  policy-independent · deterministic · fail-safe │
└─────────────────────────────────────────┘
  ↓  validated action
Controller → Actuator → Robot
```

§4.2 rule 8 guarantees this structurally. `es-safety` does not depend on `es-policy`.

### 9.2 `DeploymentIR`

```rust
pub struct DeploymentIR {
    pub schema_version: u32,
    pub robot: RobotRef,              // scene or real hardware description
    pub action: ActionSpec,           // §8.5
    pub envelope: SafetyEnvelope,
    pub watchdogs: Vec<Watchdog>,
    pub fallback: FallbackPolicy,
    pub rate: RateSpec,
}
```

### 9.3 Safety Envelope

Static and dynamic constraints applied to the policy output. **All are deterministic and all are mandatory.**

| Constraint | Content | On violation |
|---|---|---|
| `torque_limit` | Per-joint torque upper bound | Clamp + counter |
| `velocity_limit` | Joint/EE velocity upper bound | Clamp + counter |
| `position_limit` | Joint limits + soft margin | Clamp |
| `workspace` | EE workspace (box, cylinder, convex polytope) | Projection + counter |
| `collision_constraint` | Self-collision / environment collision minimum distance | Stop or retreat |
| `rate_limit` | Action first/second derivative upper bound | Filtering |
| `action_validity` | Domain / NaN/Inf check | **Immediate fallback** |
| `jerk_limit` | Jerk upper bound (optional) | Filtering |

**The distinction between clamp and fallback is important.** Exceeding the torque upper bound is clamped and recorded. NaN cannot be clamped, so it is an immediate fallback.

### 9.4 Watchdog and Fallback

```rust
pub enum Watchdog {
    InferenceDeadline { budget: Duration },     // inference exceeds budget
    ChunkUnderrun,                              // chunk exhaustion (§8.6)
    StaleObservation { max_age: Duration },     // observation is stale
    NanInf,                                     // NaN/Inf in action
    EnvelopeViolationRate { window: usize, max_frac: f32 },
    ControllerHeartbeat { timeout: Duration },
    SensorDropout { sensor: SensorId, max_gap: Duration },
}

pub enum FallbackPolicy {
    HoldPosition,                  // hold current pose
    ZeroVelocity,                  // decelerate to stop
    RetractToHome { traj },        // precomputed safe trajectory
    HandoffController { id },      // hand off to classical controller
    EmergencyStop,                 // immediate stop + latch
}
```

**The fallback is not a policy.** It is not a neural network, it is deterministic, and it must operate even if the policy runtime dies. `es-safety` operates in no-std on a preallocated workspace.

**It is turned on identically even in sim.** If the Safety Plane is off during learning but turned on at deployment, you learn only just before deployment that the policy was trained in a way that violates safety constraints. **The envelope violation rate during sim learning is a first-class metric of the Evaluation IR** (§10.3).

**An episode is the unit of work (packet M7/R1, owner decision 2026-09-21).**
`SafetyPlane::begin_episode`, besides the latch and the seed, clears the sliding window of
`EnvelopeViolationRate` (`SafetyCounters::window`). Tick 0 of episode `k` is not judged as
the tail of episode `k−1`, and the watchdog reads `0.0` until the window fills again
(§10.3's "a full window, or no verdict"). The envelope, the watchdog, and the cumulative
counters are unchanged (INV-12), and `validate`'s signature is unchanged too (INV-13). This
is the condition that also makes §10.4's "`--jobs N` produces no report that `--jobs 1`
would not" hold for the `(cell, episode)` partition, and every evaluation number committed
before 2026-09-21 was measured under the window's carry-over semantics — mark it as such, do
not delete it (§28.9 rule 2).

### 9.5 Equivalence of sim and real hardware

| | Sim | Real hardware |
|---|---|---|
| Observation IR | GPU kernel | CPU/NPU kernel (same IR) |
| Learning IR preprocessing | GPU kernel | Same |
| PolicyHandle | Torch/ONNX/Vulkan | ONNX/Vulkan/NPU |
| ActionChunker | Deterministic CPU | Same code |
| **Safety Plane** | **Same code** | **Same code** |
| Controller | Sim actuator model | Real drive |

**If the `deployment_hash` is the same, the safety behavior is the same.** This is the core claim of the §27.1 evidence artifact.

### 9.6 `es-runtime-embedded`

Minimal runtime for real hardware deployment. Single static binary, no-std capable, 0 heap allocation.

```
Included:  Observation IR evaluator + Learning IR pre/post-processing
       + PolicyRuntime (ONNX or Vulkan or NPU)
       + full Safety Plane
       + telemetry ring buffer

Excluded:  physics engine, renderer, editor, Python, learning
```

The deployment artifact is a single `policy.esb` bundle.

```
policy.esb
├── manifest.json          execution_hash and the full configuration hash
├── observation.ir         §7 (canonical)
├── learning.ir            §8 (pre/post-processing part)
├── deployment.ir          §9 (safety constraints)
├── policy/                weights (onnx | safetensors | spirv)
├── stats/                 normalization statistics
└── signature              optional signature
```

**You do not deploy the policy weights alone.** You deploy the preprocessing and safety constraints together.

---

## 10. Evaluation IR

### 10.1 Why it is needed

In current robot learning, policy comparison usually takes the following form.

```
Policy A = 82%
Policy B = 81%
```

This tells you nothing. What is needed for industrial adoption is the following.

```
                    A       B
nominal            92%     94%
lighting_shift     87%     71%     ← B is vulnerable to lighting
camera_shift       84%     88%
object_pose_shift  79%     81%
occlusion          61%     63%
latency +20ms      84%     52%     ← B is very vulnerable to latency
sensor_dropout     73%     70%
actuator_noise     88%     85%
─────────────────────────────────
intervention rate  3.1%    7.8%
envelope violation 0.4%    2.1%    ← B frequently touches safety constraints
action smoothness  0.82    0.61
p95 latency        7ms     380ms
```

**The Evaluation IR specifies the generation of this table like a task definition.**

### 10.2 Structure

```yaml
evaluation:
  schema_version: 1
  task: tasks/pick_cube.toml
  observation: obs/two_view_224.toml
  episodes_per_cell: 100
  seed_base: 20260912

  suites:
    - name: nominal
      perturbations: []

    - name: lighting_shift
      perturbations:
        - { kind: light_intensity, range: [0.3, 2.5], dist: loguniform }
        - { kind: light_direction, range_deg: 45 }
        - { kind: color_temperature, range_k: [2700, 7500] }

    - name: camera_shift
      perturbations:
        - { kind: camera_extrinsic, pos_sigma_m: 0.02, rot_sigma_deg: 3 }
        - { kind: camera_intrinsic, focal_rel_sigma: 0.02 }

    - name: object_pose_shift
      perturbations:
        - { kind: object_pose, target: "cube", pos_sigma_m: 0.05, yaw_deg: 180 }

    - name: occlusion
      perturbations:
        - { kind: occluder, count: [1, 3], size_m: [0.03, 0.10] }

    - name: latency_injection
      perturbations:
        - { kind: observation_delay, ms: [0, 20, 50] }
        - { kind: action_delay, ms: [0, 20] }

    - name: sensor_dropout
      perturbations:
        - { kind: frame_drop, prob: 0.05, burst: [1, 3] }

    - name: actuator_noise
      perturbations:
        - { kind: torque_noise, rel_sigma: 0.05 }
        - { kind: backlash, rad: [0.0, 0.01] }

  metrics: [success_rate, intervention_rate, collision_rate,
            envelope_violation_rate, action_smoothness, chunk_underrun_rate,
            p50_latency, p95_latency, episode_length, failure_mode_histogram]

  acceptance:
    nominal.success_rate:              ">= 0.85"
    lighting_shift.success_rate:       ">= 0.75"
    latency_injection.success_rate:    ">= 0.70"
    envelope_violation_rate:           "<= 0.01"
    p95_latency_ms:                    "<= 100"
```

### 10.3 Metric definitions

| Metric | Definition |
|---|---|
| `success_rate` | Proportion reaching Task IR's `Terminate(success)` |
| `intervention_rate` | Proportion of episodes in which human intervention occurred on real hardware / HIL |
| `collision_rate` | Rate of unwanted contact occurrence |
| **`envelope_violation_rate`** | **Proportion of steps that the Safety Plane clamped/projected (§9.3)** |
| `action_smoothness` | Normalized reciprocal of the action first/second derivatives |
| **`chunk_underrun_rate`** | **Rate of chunk exhaustion occurrence (§8.6)** |
| `p50/p95_latency` | Observation capture → action output end-to-end |
| `failure_mode_histogram` | Classification of failure causes (timeout, collision, miss, divergence, fallback) |
| **`domain_gap`** | **Observation distribution distance during real hardware log replay (§24.3)** |

The three shown in bold do not exist on other platforms. Each requires a Safety Plane, asynchronous inference, and a sim2real layer, respectively, to be measurable.

### 10.4 Determinism and fairness

- Every perturbation is sampled from `TaskRng(seed_base, suite_id, episode_idx, stream)`. **Even if you change the policy, the same perturbation sequence comes out**
- Augmentation nodes (§7.3) are automatically disabled during evaluation
- If the `evaluation_hash` is the same, the conditions are the same. It is included in the report
- Evaluation runs are replayable. Failed episodes are rewound and analyzed with the §23 integrated debugger

### 10.5 Artifacts

```
es eval run --config eval/pick_cube.yaml --policy policy.esb
  →  report.json        metrics × all suites
  →  report.html        table + failed episode links
  →  episodes/          replay (failures first)
  →  evaluation.lock    evaluation_hash + environment conditions
```

`es eval compare A.json B.json` outputs the per-suite differences of the two policies and statistical significance.

**`events.json`'s `tick` is an episode-relative clock (packet M7/R1).** It counts from 0 at
that episode's `begin_episode`, not the cell's cumulative physics tick. Since `frame` is
already a dense index within the episode, the absolute tick was only meaningful within a
single cell, and only a relative tick makes `events.json` independent of how the run was
scheduled. So `es eval run --jobs N` may shard workers by `(cell, episode)` rather than by
cell (§10.4), and `report.json`, `events.json`, and `.estraj` are bit-identical to `--jobs
1` (the sole exception is `evaluation.lock`'s `created`).

---
## 11. Unified Compiler

### 11.1 Pipeline

A single compiler handles the five IRs.

```
  Frontend (§14)
       ▼
   Parse          expression → IR subgraph
       ▼
   Normalize      constant folding, dead-node elimination, SubGraph inlining, unit normalization
       ▼
   Type Check     ElemType · Shape · Unit · Frame · TimeRef · ImageSpec
       ▼
   Cross-IR Check ┌ Task.ObservationSpec  ↔ Observation.inputs
                  ├ Observation.outputs   ↔ Learning.inputs
                  ├ Learning.outputs      ↔ Task.ActionSpec
                  ├ Learning.runtime      ↔ Deployment.rate  (§8.4 LRN-052)
                  └ Deployment.envelope   ↔ Robot capability
       ▼
   Semantic       cycles, unconnected nodes, reversed phases, resource references
       ▼
   Determinism    §6.6 rules (deterministic mode)
       ▼
   Capability     PhysicsBackend · PolicyRuntime · GPU (§11.6)
       ▼
   Canonicalize   topological sort + ID renumbering → *_hash (§5.3)
       ▼
   Schedule       phase groups, batch domain assignment (§12), fusion boundaries
       ▼
   Memory Plan    intermediate tensor liveness → buffer reuse (§20)
       ▼
   Lower ──┬──► CPU reference execution plan   (ground-truth reference)
           ├──► Slang kernels + SPIR-V         (sim GPU)
           ├──► PolicyRuntime call plan        (Torch/ONNX/Vulkan)
           ├──► Safety Plane static code       (deterministic, no-std capable)
           └──► runtime commands               (randomization·reset·recording)
       ▼
   compile_hash / execution_hash
```

**`Cross-IR Check` is the most valuable check in v1.0.** All boundaries between the five IRs are verified here. Shape mismatches, normalization-space mismatches, action-dimension mismatches, and inference-latency overruns are all caught at compile time.

### 11.2 Hash Separation

| Hash | Input | Purpose |
|---|---|---|
| `*_graph_hash` | Includes node IDs (user-assigned) | Authoring identity. Debugger tracing, hot-patch target |
| `*_hash` | Renumbered after normalization | Semantic identity. Cache, deduplication, registry |
| `compile_hash` | Above + compiler version + backend + precision + SPIR-V hash | Execution artifact identity |
| `execution_hash` | §5.3 | Tier 1 reproduction condition |

**Normalization invariants** (property test, Appendix B)
```
hash(canon(g)) == hash(canon(shuffle_ids(g)))
hash(canon(g)) == hash(canon(move_ui(g)))
hash(canon(g)) == hash(canon(deser(ser(g))))
hash(canon(g)) != hash(canon(change_any_param(g)))
```

### 11.3 CPU Reference Execution Plan

**The CPU path is not the performance path but the ground-truth reference.**

- Runs all Task IR·Observation IR·Learning IR preprocessing on the CPU
- The oracle for GPU lowering. Tier 2 validation (§3.5) applies not only to physics but also to observation·reward·preprocessing
- Works without Slang (macOS, fast CI gate)
- Per-node intermediate values are always visible

### 11.4 GPU lowering

Fuses consecutive element-wise nodes into a single kernel. Fusion boundaries are the boundaries of reduction·shape-change·sensor-read·debug-mode nodes.

The SPIR-V cache key is `(ir_hash, fusion group, backend, precision)`. Shapes are inserted as specialization constants so that tasks differing only in shape share the cache. When `es task compile` packages the cache into the bundle, Slang is not needed in the deployment environment.

### 11.5 Release Plan vs. Debug Plan

After kernel fusion there are no node boundaries, so per-node timing cannot be obtained. Two plans are produced.

| | Release | Debug |
|---|---|---|
| Fusion | Maximal | Node boundaries preserved |
| Per-node timing | None | Present |
| Per-node value statistics | Sampled | All |
| Full tensors | Selected nodes on request | Selected nodes |
| Performance | Baseline | 2–5× slower |

During training, the release plan runs and only value statistics are sampled. When a specific environment is designated for debugging, only that environment is replayed rewound with the debug plan (§23.3).

**Per-term reward decomposition and Learning IR per-node statistics are exceptions.** The `Reward` node and encoder outputs must emit values anyway, so they are produced even under the release plan.

### 11.6 Capability Validation

```
WARN DEP-114  not supported on the selected runtime

  node:  VisionEncoder "dinov2_vitb14"
  requires:  attention_flash, dtype=bf16

  status per PolicyRuntime:
    Torch (CUDA)    supported
    Torch (ROCm)    supported (flash attention unsupported → standard attention fallback)
    ONNX (CPU)      supported (slow: expected latency 2.1 s)
    ONNX (TensorRT) supported
    Vulkan          unsupported (model size exceeded)

  current selection: Vulkan → compile failure
  recommendation: --policy-runtime onnx or a smaller backbone (ResNet18, SmolVLM)
```

It must appear at compile time. It must not die after a 4,096 env batch has been running for a long time.

---

## 12. Batch Domains and Execution Orchestration

### 12.1 The Four Domains

Treating a single environment as the base unit of every pipeline suits state-based RL but **does not suit vision learning.**

```
┌─────────────────────────────────────────────────────────────┐
│ SIMULATION BATCH         N_sim = 4,096                      │
│  physics steps. dt_phys = 1 ms                              │
│  backend: MJWarp / Newton                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │  select active-camera env (§12.2)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ OBSERVATION BATCH        N_obs = 512 env × 2 view           │
│  render + Observation IR. sensor_dt = 33 ms (30 Hz)         │
│  tile atlas (§15.2)                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │  build inference batch
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ INFERENCE BATCH          N_inf = 256                        │
│  PolicyRuntime. asynchronous. latency 5–370 ms             │
│  results go to the chunk buffer (§8.6)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │  record episodes → dataset
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ TRAINING BATCH           N_train = 64 sequences             │
│  PyTorch. separate process·device possible                 │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 Inter-Domain Policy

| Transition | Decision |
|---|---|
| sim → obs | Which env has its camera turned on. `all` / `subset(k)` / `round_robin(k, period)` |
| obs → inf | Inference batch size, padding policy, wait-time upper bound |
| inf → sim | Chunk buffer management, switching policy (§8.6), depletion handling |
| sim → train | Episode recording period, filter (success/failure/intervention) |

**`round_robin` is important in practice.** Turning cameras on for all 4,096 envs blows out VRAM (§20). Cycling through in 8 periods of 512 each keeps memory at 1/8 while data diversity is preserved. The cycling order is deterministic (based on env ID).

### 12.3 Determinism

Even when batch domains are separated, Tier 1 reproduction must be maintained.

- Active-camera env selection is a deterministic function of `(seed, tick, period)`
- The inference batch construction order is env ID ascending
- The application time of asynchronous inference is determined **not by the arrival time but by the logical tick of the arrived chunk**
- Whether chunk depletion occurs is recorded in the replay

**The key to maintaining determinism in asynchronous inference is to determine application not by "when it arrived" but by "which tick's observation it was computed from."** Even if inference finishes earlier than expected, it waits until the fixed tick (deterministic mode). In real-time mode it applies immediately and records this difference.

### 12.4 Performance Metrics

The single `step/s` metric is abolished.

| Metric | Meaning |
|---|---|
| `physics_steps_per_sec` | Physics backend throughput |
| `camera_frames_per_sec` | Render frames (env × view) |
| `pixels_per_sec` | Resolution-independent render throughput |
| `observation_GB_per_sec` | Observation pipeline bandwidth |
| `policy_inferences_per_sec` | Inference throughput |
| `actions_per_sec` | Effective control steps |
| `p50 / p95 end_to_end_latency` | Observation capture → action output |
| `gpu_memory_peak` | VRAM peak |
| `chunk_underrun_rate` | Chunk depletion rate |

**`simulation throughput ≠ learning throughput`.** Even if physics runs at 100k step/s, if the camera is 30 Hz then observations are a few thousand frames per second, and if the policy is 370 ms then inference is a few batches per second. This table reveals where the bottleneck is.

**Baseline estimation example** (4,096 env × 2 camera × 224×224 × RGB × 30 Hz)
```
frames:  4096 × 2 × 30              = 245,760 frames/s
pixels:  245,760 × 224 × 224        = 12.3 Gpixels/s
bandwidth:  12.3G × 3 bytes (RGB8)  = 37 GB/s   ← render output only
         + after FP32 normalization = 148 GB/s  ← preprocessing intermediate tensors
```

**This calculation is why the `round_robin` of §12.2 is needed.** Since the bandwidth of an RTX 4090 is about 1 TB/s, 148 GB/s is theoretically possible, but it is not realistic when physics·render·inference share the same device. Reducing active cameras to 512 env brings it down to 18.5 GB/s, leaving headroom.

---

## 13. Learning Loop

### 13.1 First-Class Workflow

The real-world robot learning stack of 2026 is iterative. LeRobot also supports a HIL workflow that runs a policy on real hardware and re-collects human intervention data.

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
   ┌─────────┐    ┌───────┐    ┌──────────┐    ┌──────────┐
   │ COLLECT │───►│ TRAIN │───►│ EVALUATE │───►│  DEPLOY  │
   └─────────┘    └───────┘    └──────────┘    └────┬─────┘
        ▲                           │                │
        │                           │ failure episode│
        │                           ▼                ▼
        │                    ┌─────────────┐   ┌──────────────┐
        └────────────────────│ FAILURE SET │◄──│ INTERVENTION │
                             └─────────────┘   └──────────────┘
```

Each stage is both a CLI command and an artifact.

```
es collect   --task T --teleop <device> --episodes N     → dataset/
es train     --task T --obs O --learning L --data D      → checkpoint/ + training.lock
es eval      --config E --policy P                       → report.json + episodes/
es deploy    --policy P --deployment D --target <robot>  → policy.esb
es intervene --session S                                 → dataset/ (includes intervention labels)
es distill   --failures F --into D                       → dataset/ (merge failure set)
```

### 13.2 Status of Intervention Data

Intervention episodes are treated differently from ordinary demos.

```
episode metadata:
  source:        teleop | policy | policy_with_intervention | scripted
  intervention:  [{ start_tick, end_tick, reason, operator_id }]
  outcome:       success | failure | aborted
  failure_mode:  timeout | collision | miss | diverge | fallback_triggered
```

**Intervention segments can be weighted differently during learning, and that weighting policy is included in `dataset_hash`.** HIL-SERL-family workflows require this.

### 13.3 Reproducibility of the Loop

Each iteration is linked by an `execution_hash` chain.

```
iteration 3:
  dataset_hash   = H(iteration 2 data + new intervention episodes)
  policy_hash    = H(base=iteration 2 checkpoint, training.lock)
  evaluation_hash= (fixed)
  →  "did the policy improve" can be judged under the same evaluation conditions
```

**Keeping the evaluation conditions fixed while changing only data·policy is the discipline.** If `evaluation_hash` changes, the comparison is invalid, and the tool warns about this.

### 13.4 Continuing Reinforcement Learning

§13.1's loop began as imitation learning (demonstrate → train), but most 2026 robot policies
are trained with PPO in a different simulator and then deployed (MuJoCo Playground and Isaac
Lab). Having that policy **run bit-identically** on this runtime (§8.9), **score in our
sim** through the Evaluation IR (an honest sim-to-sim number), and **keep training in our
sim** from those weights is plan S (§28.11). This section pins that semantics.

**PPO is a trainer on top of the Learning IR, not an IR.** The deployed graph is written
entirely in §8.3's nodes — for a state policy, `Normalizer{Forward, MeanStd}` →
`StateEncoder{Mlp}` → `PolicyHead{Regression, horizon 1}` → `Normalizer{Inverse}`. The value
head, the log std, GAE, the optimizer, and the entropy coefficient are training-only and
live solely in `python/es/train_ppo.py` and §19.3's `training/`. They are nowhere in the
document and do not move `learning_hash`. Task IR has no RL (rule 6): reward and termination
are exactly §6.3's `Reward` and `Terminate` sinks, and the trainer only reads them, never
defines them.

**The rollout is `es-env`'s simulation batch domain** (§5.2, §12.1). The trainer steps N
envs through `Env::reset`/`Env::step` and reads reward and termination from `StepOutcome`;
it never calls another simulator directly. **The Safety Plane stays on during rollout too**
(§9.4, INV-12): the policy's sample is the action, what the plane emits reaches the
actuator, and the rollout records both (§13.2's `action_source`). Sample ≠ execution is not
an error but a rate the report shows — that rate is exactly the `envelope_violation_rate`
§9.4 calls a first-class metric, and whether it falls during training is the answer to "does
the policy learn within the constraints."

**"Start from this policy" is written into the lock.** `training.toml`'s `[init] policy =
"<bundle>"` copies tensors whose name and shape match and initializes the rest, and
`training/init.lock` records the copied names, the initialized names, and the source
`policy_hash` (the same shape as §19.3's `base_model.lock`). Zero-step training returns the
source's weights bit-identically — that is the oracle.

**Determinism.** On the CPU backend (`mujoco-cpu`) and a fixed seed, the same recipe run
twice is bit-identical through the checkpoint (§3.5 tier 1, torch CPU deterministic
kernels). The GPU backends (MJWarp, Newton) are declared tier 3 (§3.4, §17.3), and numbers
measured under them are marked as such. Throughput is stated only in §12.4's nine metrics,
and anything not measured is `Target / Status: unverified`.

**Importing does not guess** (§14.4). Joint order, units, position target vs. torque, and
the layout of observation channels are declared by a robot-specific **adapter document**,
and a mismatch is rejected as a named error (`IMP-0xx`). The source framework's
`.pt`/`.pth`/orbax files are opened only by the training-path Python (INV-16); the Rust
runtime sees only `safetensors` and the manifest beside it.

---
## 14. Authoring Frontend

### 14.1 Principles

Five frontends produce the same canonical IR. **Equivalence is verified by CI.**

```
es run task.toml       ┐
es run task.esgraph    ├──►  same *_hash  ──►  same execution
python train.py        ┘
```

Everything can be run without an editor.

### 14.2 Python Builder

```python
from electric_sheep import Task, Observation, Learning, Deployment, math as m

# ── Task IR ──────────────────────────────────────────────
task = Task("pick_cube", scene="scenes/table_franka.usda")
arm  = task.robot("franka")

task.observe_spec(
    rgb_front   = task.sensor("cam_front").rgb(),
    rgb_wrist   = task.sensor("cam_wrist").rgb(),
    joint_state = arm.joint_positions(),
)
task.action_spec(space="joint_position", robot=arm, limits="from_scene")

task.reward("reach", weight=1.0,
            value=-m.norm(task.body("cube").pose().pos
                          - arm.link("hand").pose().pos))
task.reward("grasp", weight=5.0,
            value=task.contact(arm.link("finger"), "cube").force.sum() > 1.0)
task.terminate("success",
               when=m.norm(task.body("cube").pose().pos
                           - task.site("target").pos) < 0.02)
task.randomize("cube_pose", stream="init",
               target=task.body("cube").initial_pose,
               dist=("uniform_se2", [-0.1, 0.1], [-0.1, 0.1], [-3.14, 3.14]))

# ── Observation IR ───────────────────────────────────────
obs = Observation("two_view_224", task=task)
obs.image("rgb_front").resize(224, 224).to_linear().normalize("imagenet")
obs.image("rgb_wrist").resize(224, 224).to_linear().normalize("imagenet")
obs.state("joint_state").normalize("joint_limits")
obs.temporal_window(n_steps=1)          # ACT default
obs.time_align("hold")

# ── Learning IR ──────────────────────────────────────────
lrn = Learning("act_r18", observation=obs)
lrn.vision_encoder("resnet18", pretrained="imagenet", frozen=False, shared=True)
lrn.state_encoder("mlp", out_dim=512)
lrn.fusion("concat")
lrn.temporal_encoder("none")
lrn.head("regression", horizon=50)
lrn.chunker(execute_chunk=20, replanning_hz=10, mode="temporal_ensemble")

# ── Deployment IR ────────────────────────────────────────
dep = Deployment("franka_safe", robot=arm, action=task.action_spec_ref())
dep.envelope(torque_limit="from_urdf", velocity_limit=1.5,
             workspace=dep.box([0.2, -0.4, 0.0], [0.8, 0.4, 0.6]),
             rate_limit=2.0)
dep.watchdog("inference_deadline", budget_ms=50)
dep.watchdog("chunk_underrun")
dep.watchdog("stale_observation", max_age_ms=100)
dep.fallback("hold_position")

Task.save_bundle("tasks/pick_cube/", task, obs, lrn, dep)
```

**The builder does not execute immediately.** At `save_bundle` or `compile` time it assembles the IR and runs the Cross-IR Check (§11.1). Type errors surface as Python exceptions, but the messages use the same diagnostic format as the compiler.

**When Python goes beyond the IR (arbitrary callbacks, etc.), it raises an explicit error.** Allowing "anything goes as long as it's Python" would break the IR's guarantees.

### 14.3 Storage Formats

| Format | Purpose |
|---|---|
| `task.toml` / `observation.toml` / `learning.toml` / `deployment.toml` / `evaluation.yaml` | Human-readable and git diff |
| `*.esgraph` | Graph editor storage (explicit nodes/edges, stable IDs) |
| `*.eslayout` | Editor metadata sidecar. **Does not enter the IR** |
| `bundle.eslock` | Hashes of the five IRs + pinned references |

### 14.4 External Conversion

```
LeRobot policy config    ┐
Isaac Lab task config    │
MJCF sensors/actuators   ├──► Converter ──► Semantic Mapping Report ──► IR
RoboVerse / MetaSim      │                        │
Gymnasium spec           ┘                        └─► severity=error → execution blocked
```

**LeRobot conversion is the top-priority conversion target for v1.0.** Reading the `lerobot/act_*`, `lerobot/smolvla_base`, and `lerobot/pi05_base` configs into Learning IR lets existing users bring their own policies as-is and layer Electric Sheep's evaluation, safety, and reproducibility layers on top. **This is the lowest barrier to adoption.**

**Importing an RL policy (plan S, §13.4, §28.11).** `es policy import-rl --from
mujoco-playground | rsl-rl | rl-games` moves the PPO actor MLP into `StateEncoder{Mlp}` +
`PolicyHead{Regression}`, the observation running mean and standard deviation into
Observation IR's `Normalize` node, and the action scale and offset into
`Normalizer{Inverse}`. The activation function (swish, ELU, ReLU) and the output squash
(`tanh`) are parameters of the node, and absent = default = today's canonical form, so the
committed `learning_hash` does not move. Only the front half of the conversion
(`python/es/import_rl.py`: pickle, orbax → `safetensors` + `import.json`) is Python; the
back half (document generation, bundle packing) is `es`. The Semantic Mapping Report is the
crosscheck table against the adapter document, and `severity=error` blocks execution as
above.

### 14.5 LLM Generation

```
es generate task --from-description "pick up the cube and place it at the target point" \
                 --scene scenes/table_franka.usda --candidates 16
es generate reward --task T --candidates 16 --optimize success_rate
```

A typed IR is superior to Python code as an LLM generation target. The validator gives structured diagnostics immediately, non-deterministic or safety-violating configurations cannot be expressed, it compiles to GPU, and `ir_hash` removes duplicate candidates before learning. Since the cost of Eureka-style methods is "thousands of gradient steps per candidate," deduplication is a direct saving.

We do not build an evolutionary-loop orchestrator (§0.5). We let existing tools use Electric Sheep as a backend. We expose `validate` / `compile` / `estimate_cost` / `eval` through an MCP interface (M4).

---

## 15. Vision Data Plane

### 15.1 Render → Observation Path

We place an explicit layer between rendering and the learning input.

```
Camera Renderer (§15.3)
      ↓  RGB / Depth / Seg / Normal / Flow / Velocity   (same tensor contract)
Sensor Realism (§18.3)
      ↓  lens distortion · rolling shutter · motion blur · exposure · shot noise · depth holes
Observation Capture
      ↓  GPU-resident. ImageSpec attached
Observation IR (§7)
      ↓  resize · crop · color · normalize · temporal · mask
Learning IR preprocessing (§8.7)
      ↓
Vision Encoder → Policy
```

**Key: this entire span is GPU-resident with no host round-trips.** And since the `ImageSpec` is transformed and propagated at each stage, the original camera parameters can be traced back even at the point of the encoder input.

### 15.2 Massively Parallel Cameras: Tile Atlas

`maxMultiviewViewCount` is 32 on major desktop GPUs. Multiview cannot handle hundreds of cameras.

- All cameras are laid out as tiles in a single framebuffer and rendered in a single render pass
- Each camera keeps its own intrinsic/pose. Because the tiling layout is deterministic, it is reconstructed into per-environment tensors without host transfer
- The view index is a draw payload. A culling compute compacts `(view, instance)` pairs
- Constraint: `maxImageDimension2D` (typically 16384). With 224×224 tiles, a single atlas holds **5,184**

**Reference figures:** 512 tiles of 224×224 RGB = a 5,376×5,376 atlas. 87 MB as RGB8, 347 MB after FP32 normalization, 2× with double buffering. Accounted for in §20.

### 15.3 Render Paths

| Path | Condition | Purpose |
|---|---|---|
| **RS** raster | Always | **Learning observation default**, viewer, MoltenVK |
| **PT** path tracing | RT extension | Photorealistic datasets, domain-gap experiments, golden |
| **3DGS** splat | §16 | real-to-sim reconstructed scenes |

**Output contract:** All three paths output the same channels (RGB, linear depth, instance/semantic seg, world normal, optical flow, velocity) in the same tensor layout. Depth/seg/normal are bit-identical between RS/PT; RGB is an SSIM threshold.

PT is M4 and is item 2 in the §1.9 reduction order. **The design premise is that vision learning holds up with RS alone.**

### 15.4 Acceleration Structures

One TLAS for the entire scene. N envs of the same robot share one BLAS. Since inter-environment ray isolation is impossible with an instance mask (8 bits), **spatial-separation placement + ray `tMax` limiting** is primary, and an `instanceCustomIndex` (24-bit) any-hit filter is secondary (disabled by default).

---

## 16. Real-to-Sim: 3D Gaussian Splatting

### 16.1 Why It Is Needed

**The sim2real bottleneck for vision policies is not physics accuracy but appearance.** The main reason policies fail in vision-based manipulation is that the simulated image is outside the distribution of the real hardware camera image.

In 2025–2026, 3DGS-based real-to-sim established itself as the practical solution to this problem.

| Research | Contribution |
|---|---|
| RL-GSBridge | 3DGS-based real-to-sim-to-real RL, zero-shot transfer of vision control |
| Real-is-Sim (Embodied Gaussians) | Dynamic digital twin across the entire span of collection, learning, evaluation, and deployment |
| RoboGSim | Interactive real2sim2real platform, demo synthesis, novel scene/object expansion, closed-loop evaluation |
| Real-to-Sim Policy Eval (2511.04665) | Smartphone scan → 3DGS, robot/object/background segmentation, **position/color alignment**, PhysTwin softbody. **Sim rollouts correlate with real-hardware performance at r > 0.9** |
| RialTo | Constructs a digital twin from a small amount of real-hardware data, then hardens the policy with sim RL |
| ReaDy-Go | Composites dynamic human GS avatars into a static scene to learn a navigation policy |

The second-to-last is especially important. **Color alignment** (a polynomial mapping of the initial scan's color space to the robot's actual camera (e.g., RealSense) color space) was the decisive step that makes the policy see in-distribution images.

### 16.2 Its Place in Electric Sheep

3DGS is not a separate feature but **one render path and an `ImageSpec` producer.**

```
smartphone/camera scan
      ↓
COLMAP or similar SfM  (external tool)
      ↓
3DGS training          (external tool or es-splat)
      ↓
es asset import-splat --scan scan/ --segment robot,objects,background
      ↓
SceneDesc + SplatAsset
  ├── background  static splat
  ├── objects     per-object splat + physics proxy (convex decomposition or SDF)
  └── robot       splat bound to URDF links
      ↓
es-render splat path (Vulkan compute rasterization)
      ↓
ImageSpec (set with real-hardware camera parameters)
      ↓
Observation IR  ← sim and real hardware are identical
```

**The three key design decisions**

1. **The splat does not replace physics.** Each object has a separate physics proxy (convex-decomposition mesh or SDF), and the splat follows that proxy's transform (Linear Blend Skinning). Physics is handled by the §17 backend
2. **Color alignment is part of the pipeline.** `es asset align-color --scan S --reference <real-hardware capture>` estimates the polynomial mapping and stores it in the `SplatAsset`. This mapping enters the `scene_hash`
3. **Position alignment is also part of the pipeline.** ICP + RANSAC aligns the scan coordinate frame to the robot base frame. The resulting `T_robot_scan` is recorded in the asset metadata

### 16.3 Deliverables

```
es-splat crate (layer 5)
  ├── .ply / .splat / .spz loaders
  ├── Vulkan compute rasterization (tile alignment + alpha blending)
  ├── LBS binding (link transform → splat transform)
  ├── color/position alignment
  └── segmentation metadata (robot / object / background)
```

**Exact alignment is unnecessary; distribution matching is the goal.** It is sufficient for the splat render to have the same statistical and frequency characteristics as the real-hardware camera image. This is measured with the §10 `domain_gap` metric.

### 16.4 Market Significance

With this path, the user's workflow becomes as follows.

```
1. Scan the actual workspace with a phone               (10 min)
2. es asset import-splat + align                        (automatic)
3. Collect 20–50 real-hardware demos                    (1 hour)
4. Data augmentation + policy training in sim           (automatic)
5. Evaluate perturbation suite with es eval (§10)       (automatic)
6. Deploy policy.esb including the Safety Plane (§9.6)  (automatic)
7. Retrain with real-hardware intervention data (§13)   (repeat)
```

**"Build a vision policy without CAD, without a sim expert, in your own workspace."** This is a workflow that no platform currently provides end-to-end, and it is the main market message of §27.

Schedule: **M3.** It is item 3 in the §1.9 reduction order, and the product holds up even if cut (replaceable with external 3DGS tools + manual import).

---

## 17. Physics (Backend Layer)

### 17.1 Role Shift

**In v1.0, physics is not the product but a backend.** See §4.3.

```
PhysicsBackend
├── MuJoCoWarpBackend    Default. MJCF-native, GPU-parallel          M1
├── MuJoCoCpuBackend     Deterministic reference, CI oracle          M1
├── NewtonBackend        When Kamino·VBD·hydroelastic is needed      M2
├── PhysXBackend         Isaac Lab asset compatibility               M3
└── NativeBackend        Own CPU/Vulkan solver                       M4+ (optional)
```

**We do not use our own solver in M1.** This is the biggest schedule saving, and it removes most of the Type D work (§1.3) from the critical path.

### 17.2 Backend Semantic Mapping (a new core task)

The same Task IR must not behave differently across backends. Instead of building a physics engine, **semantic mapping and its verification** become the core task.

```
Task IR                         MJWarp          Newton          PhysX
─────────────────────────────────────────────────────────────────────
actuator.pd(kp, kd)             position gain   controller      drive stiffness
contact.friction_cone           pyramidal       selectable       pyramidal
contact.soft_params             impedance       solver-dependent contact offset
joint.armature                  armature        armature        unsupported → warning
sensor.contact_force            sensor          contact          contact report
```

`es backend compare --task T --backends mjwarp,newton,mujoco-cpu` runs the same task on the three backends and produces a report comparing them with the §3.5 layer-3 metrics. **Unmapped items block execution if `severity: error`** (§14.4).

### 17.3 Determinism

- Only `MuJoCoCpuBackend` declares layer 1 (bitwise)
- GPU backends declare only layers 2 and 3. **This is the honest state**
- Workflows that require layer 1 (evidence-artifact generation, regression reproduction) run on the CPU backend. Trusting 16 envs 100% is more valuable than roughly reproducing 4,096 envs (§27.1)

### 17.4 Own Solver (M4+, optional)

Design decisions that hold if we build it. **We do not implement it now.**

- TGS + substeps, reduced coordinates (ABA), soft contact (MuJoCo impedance compatible)
- For high-DoF GPU, exclude full Delassus assembly and use Matrix-Free CG
- Contact bucket scheduling: classify contact counts into power-of-two buckets, stable-sort, then indirect dispatch per bucket
- Deterministic reduction: RFA/binned summation (§18.4)
- Performance targets are **described only as targets until verified.** We do not write figures like "average 25 → 6–8" as facts

```
Target:  CG iterations ↓ 3–5×
Success criteria:  median / p95 / worst-case iterations, at 4k env
Status:  unverified (M4 gate)
```

---
## 18. Time, Sensors, Actuators, Failure

### 18.1 Time Model

- `dt_phys` (default 1 ms), `dt_ctrl` (default 50–100 ms, based on vision policies), independent per-sensor periods
- All steps are integer ticks. Floating-point time accumulation prohibited, enforced by type
- Every sensor sample is accompanied by a physics tick number
- Observation latency and action latency are modeled with ring buffers and are targets of domain randomization

**The default value of `dt_ctrl` is 50–100 ms.** The inference latency of vision policies (§0.3) determines the realistic control period.

### 18.2 Actuators

Ideal torque/position/velocity, PD + feedforward, DC motor electrodynamics, gears/backlash, joint elasticity (Two-mass), current loop effective model, communication latency/quantization, bus jitter/packet drop, actuator net (ONNX), gripper/suction.

Map within the range supported by the backend, and record unsupported items in the §17.2 report.

### 18.3 Sensors and Realism

| Sensor | Noise model |
|---|---|
| RGB/depth camera | Gaussian/shot noise, depth holes/flying pixels, **lens distortion (OpenCV convention)**, **rolling shutter**, motion blur, exposure, white balance |
| Stereo | Disparity error |
| IMU | Bias, random walk, scale/alignment error |
| Joint encoder | Quantization, latency, offset |
| F/T | Gaussian, temperature drift |
| Lidar | Range noise, dropout, multiple reflections |
| Tactile | Contact manifold → pressure image |
| Event camera | Log-intensity difference, threshold noise, refractory |

**Camera noise parameters are reflected in `ImageSpec` (§7.2) and passed via the Observation IR.** This is the gap filled in v1.0. Knowing the distortion coefficients lets you decide whether to use the `Undistort` node at the IR level.

All parameters are targets of domain randomization and use the same parameter space as the perturbation of the Evaluation IR (§10.2).

### 18.4 Deterministic Reduction

Demmel & Nguyen family RFA / binned summation. K exponent bins (default 3), single read, single parallel reduction. **The core invariant is that `merge` is associative and commutative**, and when this holds, the result is the same even if the subgroup order, workgroup merge order, and SM count differ.

NVIDIA CCCL/CUB provides a `gpu_to_gpu` determinism level based on the same principle, and reports a 20–30% increase on large problems. **The §26 target of "deterministic mode degradation ≤ 30%" matches the independently verified figure.**

### 18.5 Failure Semantics

```
EnvHealth = Ok | Diverged | Poisoned | Quarantined
```

Quarantined envs do not participate in the reduction (because the accumulator is order-independent, masking does not break determinism). Quarantine is recorded in the replay, and if the quarantine rate exceeds a threshold (default 5%), learning is halted.

**v1.0 addition: `EnvHealth` is linked to Safety Plane events.** An env in which the fallback triggered remains `Ok` rather than `Quarantined`, but the event is recorded. Fallback is normal behavior, not a failure.

---

## 19. Datasets and Identity

### 19.1 Format

**LeRobot compatibility is primary.** Parquet + video chunks. Supports multi-episode packing and streaming of LeRobotDataset v3. RLDS/TFDS, HDF5 (robomimic) export are secondary.

Episode composition:
```
Observation   multi-camera video + low-dimensional sensorimotor + physics tick alignment
Action        recorded per chunk (entire prediction horizon + execution interval marker)
Reward        per-term decomposition
Meta          scene_hash, task_hash, observation_hash, randomization sample,
              convention version, backend, quarantine log, ImageSpec, intervention label (§13.2)
```

**Storing `ImageSpec` in the dataset is important.** It can later be reprocessed with a different Observation IR, and when combining data from multiple sources, you can tell that the camera parameters differ.

### 19.2 Dataset Identity

```
dataset_content_hash   actual sample content
dataset_schema_hash    field composition, ImageSpec, action space
dataset_split_hash     train / val / test split definition
dataset_hash = H(content, schema, split)
```

**If you do not hash the split, the training sample composition differs even for "the same dataset".** The split must be a deterministic function.

```yaml
split:
  method: by_episode_hash          # by_index | by_episode_hash | explicit
  seed: 20260912
  ratios: { train: 0.8, val: 0.1, test: 0.1 }
  stratify_by: [task_variant, outcome]
  holdout:                         # explicit generalization holdout
    objects: ["mug_07", "bowl_03"]
    lighting: ["evening"]
```

`holdout` is linked to the generalization suite of §10. Evaluating on objects/lighting never seen during training is guaranteed at the IR level.

### 19.3 Training Identity

```
training/
├── config.json         all hyperparameters
├── optimizer.json      optimizer + parameter groups
├── scheduler.json      LR schedule
├── seed.json           global seed + dataloader seed + augmentation seed
├── dataset.lock        dataset_hash (content/schema/split)
├── base_model.lock     pretrained backbone provenance + hash + license
├── augmentation.json   §7.3 augmentation node composition
├── precision.json      fp32/bf16/fp16, gradient accumulation
├── topology.json       distributed configuration (world size, parallelism method)
├── checkpoint.manifest per-checkpoint step·metrics·hash
├── metrics.parquet     learning curves
└── hardware.json       GPU model·driver·library versions

training_hash = H(all of the above)
policy_hash   = H(training_hash, checkpoint_hash)
```

**`base_model.lock` is especially important.** π₀ was pretrained on 10,000 hours of cross-embodiment data, and SmolVLA was trained on 30,000 GPU-hours of community data. **Because the provenance of the pretrained backbone directly affects the results, it must be included in the provenance.** It is also the basis for license tracking.

---

## 20. Memory and Bandwidth Budget

### 20.1 Reference Measurements

Isaac Lab (RTX 4090): Cartpole physics only 4,096 env = 3.3 GB / **Cartpole RGB camera 1,024 env = 16.7 GB** / G1 locomotion 4,096 env = 6.1 GB.

**The vision task alone has 1/4 the envs but 5× the VRAM.** This is the basis for the §12.2 `round_robin`.

### 20.2 Budget Model

| Item | Calculation |
|---|---|
| Physics state (owned by backend) | reported by backend |
| **Render tile atlas** | `N_obs × views × H × W × ch × bytes × 2 (double buffer)` |
| **Observation IR intermediate tensors** | §11.1 liveness analysis |
| **Policy weights** | ACT 52M×4B=208MB / SmolVLA 450M=1.8GB / π₀ 3.5B=14GB (fp32) |
| **Inference activations** | `N_inf × per-model peak` |
| **Chunk buffer** | `N_sim × horizon × action_dim × 4B × 2` |
| Snapshot ring buffer (rewind) | state size × number of snapshots |
| 3DGS splats (§16) | number of Gaussians × 59 floats (SH degree 3) |
| Dataset recording buffer | staging before compression |

**Calculation example** (512 obs env × 2 view × 224×224 RGB, ACT, N_inf=256)
```
Atlas RGB8 double buffer      512×2×224×224×3×2        =  308 MB
Normalization FP32 intermediate tensor  512×2×224×224×3×4  =  616 MB
Policy weights (ACT 52M fp32)                          =  208 MB
Inference activations (batch 256)  measurement needed, estimate  = ~2.0 GB
Chunk buffer 4096 env         4096×50×8×4×2            =   13 MB
──────────────────────────────────────────────────────────────
Vision·policy subtotal                                ≈  3.1 GB
+ Physics backend (MJWarp 4096 env)                   ≈  2–4 GB
+ If training is on the same GPU, PyTorch             ≈  8–16 GB
```

### 20.3 Rules

- Compute the budget when loading a scene/task/policy and compare against actual memory. **On overrun, automatically reduce `N_obs`/`N_inf` or fail explicitly.** Does not die from OOM
- `es bench --memory-report` outputs per-item measurements
- The editor displays in real time during editing. **Budget overrun is a compile error** (§11.1 Memory Plan)
- VRAM peak is annotated alongside in the performance report (§12.4)

---

## 21. TensorTransport

### 21.1 Zero-copy is not a promise but a capability

Zero-copy is **not always possible on AMD/Intel/ARM, so it is treated not as a product promise but as a capability negotiation concept.**

```rust
pub enum TensorTransport {
    VulkanShared,        // shared within the same VkDevice. Always possible
    CudaExternal,        // VK_KHR_external_memory + cuImportExternalMemory
    HipExternal,         // hipImportExternalMemory
    DlPack,              // cross-framework standard, includes stream semantics
    HostPinnedFallback,  // pinned host ring buffer. Always possible
}

pub struct TransportNegotiation {
    pub preferred: Vec<TensorTransport>,
    pub selected: TensorTransport,      // runtime decision
    pub reason: String,                 // why that one was selected
}
```

**The selection result is recorded in `runtime_hash` and the performance report.** "This run used the host copy fallback" is made explicit.

### 21.2 Conventions

- Double-buffer slot ownership is enforced by type state (`Slot<SimOwned>` / `Slot<LearnerOwned>`)
- `publish` / `release` pass only timeline values without host waits
- DLPack implements the `__dlpack__(stream=...)` protocol
- **Host synchronization counter:** in debug builds, count `vkWaitSemaphores` / `cudaStreamSynchronize` calls and panic if nonzero on the hot path
- Device UUID validation. Enumeration-order matching prohibited
- Platform handles: Linux `OPAQUE_FD`, Windows `OPAQUE_WIN32`
- **Intel (SYCL/XPU) is experimental, so specify `HostPinnedFallback`**

---

## 22. Multi-GPU

Only environment sharding is supported. Collective communication is delegated to the framework. rank = process = GPU = environment shard.

**v1.0 addition: role separation is first-class.**

```
role = sim | obs | infer | train | all

e.g.) 4-GPU configuration
  GPU0  sim   (MJWarp 4096 env)
  GPU1  obs   (render + Observation IR, 512 env × 2 view)
  GPU2  infer (PolicyRuntime, π₀ 3.5B)
  GPU3  train (PyTorch)
```

The batch domains of §12 fit naturally with device boundaries. Cross-domain transfers follow the §21 `TensorTransport` negotiation.

All ranks must have the same `execution_hash`. At startup, rank 0 compiles and broadcasts, and mismatching ranks terminate immediately. The SPIR-V cache is content-addressed, so it can be shared.

`seed_shard = hash(seed_global, rank)`. The `EnvId` of `TaskRng` is global. When `WORLD_SIZE` changes, tier-1 reproduction is impossible — recorded and warned in the replay.

---

## 23. Editor and Integrated Debugger

### 23.1 Principles

**The editor does not host learning. It is a client that connects to a running process.** Enforced by §4.2 rule 4.

**Backend-neutral:** the telemetry protocol is not exclusive to Electric Sheep. Attaching to LeRobot/Isaac Lab/Newton training via a thin Python adapter is included in the M1 deliverables. This is the wedge for adoption.

### 23.2 Layer Graph View

**See all layers on a single screen.**

```
Scene Graph
   │
Task Graph            reward terms, termination conditions
   │
Observation Graph     ImageInput → Resize → Normalize → TemporalWindow
   │
Learning Graph        VisionEncoder → Fusion → Head
   │
Action Graph          Chunker → Unnormalizer
   │
Safety Graph          Envelope → Watchdog → Fallback
```

**This is the true differentiating feature.** When a user asks "why did the grasp success rate drop?":

```
reward.grasp (0.12 ↓)
   ↑
action quality — the action overshoots the target
   ↑
policy output — variance increases in the latter half of the chunk
   ↑
vision encoder — feature norm is 0.3× the usual
   ↑
Normalize — input mean is outside the distribution
   ↑
camera_front — exposure time is pinned to the randomization upper bound
```

On a single screen you can trace back from reward all the way to camera exposure. This chain is possible only because the Observation IR and Learning IR exist as IR.

### 23.3 Inspection During Learning

- State streaming (not frames). The editor locally replicates the scene and receives only the pose/joints/contacts of selected envs to render locally
- Budgeted sampling (default 20 µs per step). Observation image requests have a separate rate limit
- What is visible: 3D view, **actual observation images (pre- and post-preprocessing simultaneously)**, per-term reward decomposition, **per-node Learning IR statistics**, **Safety Plane event log**, **chunk buffer state and inference latency histogram**, `EnvHealth` distribution, per-rank throughput/VRAM
- Controls: pause/single step, reset, hot patch (permitted parameters only), rewind

**"Simultaneous display of pre- and post-preprocessing images" is practically powerful.** Visually confirming the difference between what the policy actually sees and what was rendered is half of vision debugging.

### 23.4 Three Stages of Graph Functionality

| Stage | Functionality | Schedule |
|---|---|---|
| Read-only view | auto layout, per-node real-time values, backtracking | **M1** |
| Editing | add/connect nodes, edit parameters, real-time validation | M3 |
| Control Graph | IR-C editing | M4 |

Start from read-only. An editor needs layout persistence, undo/redo, search, and large-graph performance all together, but read-only needs none of them, and **most of the debugging value comes from read-only.**

Technology: egui + winit, node graph is `egui-snarl`, remote is QUIC.

**Performance gate: learning degradation from enabling telemetry + graph view < 1%.** Required for M1.

---
## 24. Sim-to-Real

### 24.1 ROS 2 Boundary

`rmw_zenoh` and `zenoh-plugin-ros2dds` do not interoperate because their key expression schemes differ. `rmw_zenoh_cpp` is Tier 1 from Kilted (2025-05) and Tier 1 on all platforms in Lyrical (2026-05).

| Mode | Target | Implementation |
|---|---|---|
| **A. rmw_zenoh native** (default) | ROS 2 Kilted+ | Implements key expression, CDR, attachment, and liveliness with `zenoh-rs` |
| B. DDS bridge | Existing DDS | `zenoh-bridge-ros2dds` as a separate process |
| C. Pure Rust DDS (experimental) | Direct DDS connection | `RustDDS` + `ros2-client` |

A and B cannot be enabled at the same time. They are enforced as mutually exclusive in configuration and validated with liveliness at startup.

### 24.2 Hardware-in-the-Loop

Connect a real controller to the simulation. ROS 2 or low-latency UDP (`es-hil`). Deadline misses and jitter are exposed via telemetry.

**In HIL mode, layer 1 determinism is not guaranteed.** External hardware is non-deterministic. Instead, input logs are recorded to enable post-hoc replay.

**v1.0 addition: verifying that the Safety Plane behaves identically in HIL and on real hardware is the M3 gate.** A safety mechanism that only turns on in simulation is meaningless.

### 24.3 Domain Gap Diagnosis

```
es domain-gap --real <rosbag|dataset> --sim <scene+task+obs>
```

Replays a real-hardware log in the simulation and compares observation distributions.

| Comparison | Metric |
|---|---|
| Image | Channel histogram, frequency spectrum, FID family, encoder feature distance |
| State | Joint trajectory error, velocity distribution |
| Contact | Contact event timing, force distribution |
| Policy response | **Action difference for the same observation** |

The last one is the most practical. **If the policy responds differently to sim and real images, that is the definition of the gap.** The 3DGS color alignment in §16 aims to reduce this metric.

---

## 25. Security, Governance, and Versioning

### 25.1 Security

- Telemetry: token authentication, QUIC TLS, default localhost binding
- Asset parser fuzzing: USD/MJCF/URDF/glTF/**3DGS ply**
- **IR parser and validator fuzzing.** Graph files are the input that crosses the trust boundary
- **Task/Observation/Deployment IR are not a script injection surface.** No arbitrary native call, file, or network node exists
- **Policy weights are input that crosses the trust boundary.** `safetensors` preferred, pickle-based formats warned. Model hash verification
- `CustomKernel` (Slang) runtime compilation is subject to a resource whitelist + time and resource caps
- Native plugin signature verification (optional)

### 25.2 Legal Confirmation Items

| Item | Status |
|---|---|
| MJCF format parsing | No issue |
| MuJoCo Menagerie asset redistribution | **Varies per model. Check individually** |
| **Pretrained backbone licenses** (π₀, SmolVLA, GR00T, DINOv2, SigLIP) | **Varies per model. Recorded in `base_model.lock` and verified at deployment** |
| LeRobot dataset and checkpoint redistribution | **Needs confirmation** |
| Whether conversion outputs from Isaac Lab / RoboVerse are derivative works | **Needs confirmation** |
| Privacy and portrait rights of 3DGS scan data | **User scans may contain people. Policy needs to be established** |
| Newton / MJWarp / USD (Apache-2.0) | No issue |

**Pretrained backbone licenses are a substantive new risk in v1.0.** If a backbone goes into a policy bundle (§9.6), the license travels with it at deployment. `es deploy` checks this and warns.

### 25.3 API Versioning

| Target | Policy |
|---|---|
| **The 5 IR schemas** | Integer schema version. Reading a higher version is refused. Migration tooling required |
| **Built-in node type IDs** | Frozen from M1. Only field additions allowed |
| `policy.esb` bundle format | Format version. Reading older versions supported |
| Python API | SemVer. Unstable before 1.0 |
| Plugin C ABI | Integer ABI version. Fixed from M3 |
| Telemetry protocol | Handshake negotiation, supported down to N-1 |

**IR schemas and node type IDs are a stronger contract than the code API.** Breaking a user's task and policy files is far worse than breaking code.

---

## 26. Non-functional Requirements and CI

### 26.1 Non-functional

- Platform: Linux x86_64 (primary), Windows, Linux aarch64 (Jetson), macOS arm64 (RS·CPU)
- Deployment: single static binary + Python wheel. **`es-runtime-embedded` can be no-std, zero heap allocation**
- Determinism: §3.5 5 layers. Deterministic-mode degradation ≤ 30%
- Memory: §20 budget. Does not terminate on OOM
- **Safety: no path exists where the Safety Plane is disabled.** Compile-time guarantee
- Task and policy: **what is not validated is not executed**
- Observability: Tracy, GPU timestamps, Prometheus

### 26.2 CI Tiers

| Tier | Time | Items |
|---|---|---|
| **PR** | < 10 min | Unit, layering, layout validation, clippy, fmt, IR schema/normalization hash/cycle/type checks, determinism lint, Cross-IR Check fixtures, Safety Plane violation scenarios |
| **Merge** | < 1 hr | Physics regression, golden images, MJCF conformance, Python/TOML/esgraph → IR equivalence, **Observation IR ↔ LeRobot preprocessing equivalence**, CPU/GPU lowering equivalence (small-scale) |
| **Nightly** | Several hours | Performance regression, determinism cross-check, fuzzers (IR·asset·weights), `loom`, TSan, large graph compilation, **policy equivalence (layer 4)**, replay |
| **Weekly** | Several hours | Per-vendor determinism, memory budget, backend comparison (§17.2), **Evaluation IR full suite** |
| **Release** | Several days | Full training curves for standard tasks, real-hardware validation, IR schema compatibility matrix, migration |

**Runners:** CPU×3, NVIDIA (RTX 4090 class)×2, AMD×1, Intel×1, macOS×1, **real robot cell×1** (from M3).

The real robot cell runner is central. Without real-hardware validation, no sim2real claim can be made.

---

## 27. Market Strategy

### 27.1 Regulatory Evidence Artifacts (as of the fixed point 2027-01-20)

The EU Machinery Regulation (EU) 2023/1230 becomes mandatory from January 20, 2027. It replaces 2006/42/EC and, for the first time, **includes AI-based safety functions in the scope of conformity assessment**, and makes it possible to require a new assessment by including safety-function software updates in "substantial modification." Digital documentation is accepted. Harmonized standards for AI safety functions are still under development.

Targets: AMR/AGV, adaptive-control and learning-based grasping robot arms and cobots, autonomous ground robots, hazard-zone-judging vision systems, **learning policies that adjust torque limits**.

**Provenance Bundle + Safety Case**

The Provenance Bundle expresses artifact lineage, but by itself it has **no "requirement → validation evidence" relationship.** So a Safety Case graph is placed alongside it.

```
evidence.esb
├── manifest.json           execution_hash and all configuration hashes (§5.3)
├── safety_case/
│   ├── hazards.json        hazard analysis (list of identified hazards)
│   ├── requirements.json   safety requirements corresponding to each hazard
│   ├── traceability.json   requirement ↔ Deployment IR constraint ↔ validation evidence
│   ├── residual.json       residual risk and rationale
│   └── change_impact.json  requirements affected on change, re-validation scope
├── task/ observation/ learning/ deployment/     the 5 IRs (canonical)
├── scene/                  flattened scene + assets.lock (including licenses)
├── policy/                 weights + base_model.lock (§19.3)
├── training/               all of §19.3
├── validation/
│   ├── determinism.json    §3.5 per-layer results
│   ├── evaluation.json     §10 full suite results
│   ├── safety.json         envelope violation and fallback trigger statistics
│   └── domain_gap.json     §24.3
└── signature
```

**`traceability.json` is the core.**

```json
{
  "REQ-07": {
    "hazard": "HAZ-03 (collision with a human worker)",
    "requirement": "EE velocity does not exceed 0.25 m/s in the human detection zone",
    "implemented_by": {
      "ir": "deployment.ir",
      "constraint": "envelope.velocity_limit.zone_human",
      "hash": "b3:4a71..."
    },
    "evidence": [
      { "kind": "static",  "check": "compile.cross_ir.DEP-031", "result": "pass" },
      { "kind": "runtime", "metric": "envelope_violation_rate.zone_human",
        "suites": ["nominal", "occlusion", "latency_injection"],
        "value": 0.0, "episodes": 800 },
      { "kind": "hil", "session": "hil_2026_11_03", "violations": 0 }
    ],
    "revalidation_trigger": ["deployment_hash", "policy_hash"]
  }
}
```

`revalidation_trigger` links to the §5.3 hash chain. **When a policy is retrained, `policy_hash` changes, and it is automatically determined that re-validation of this requirement is needed.** Handling "substantial modification" becomes mechanical.

`es evidence verify bundle.esb` performs hash chain verification + replay re-execution + metric comparison + traceability completeness checking.

**Positioning caution:** it is not a certification body, and the bundle does not guarantee conformity. **It is a tool for evidence collection and tracing to support technical documentation writing.** Once standards are finalized, a mapping document will be added. We do not overstate.

### 27.2 Adoption Path

**Climb from a low threshold.**

| Step | What the user gains | What the user gives up | Timing |
|---|---|---|---|
| 1. Telemetry client | Attach to and view a running LeRobot/Isaac training | Nothing | M1 |
| 2. Evaluation IR | Evaluate a policy with a perturbation suite | Nothing (only evaluation is swapped) | M2 |
| 3. Observation IR | Sim, dataset, and real-hardware preprocessing become one | Preprocessing code | M2 |
| 4. Safety Plane + deployment | Gain a safety runtime and evidence artifacts | Deployment stack | M3 |
| 5. Learning IR | Policy configuration is type-checked and reproduced | Way of defining policies | M3 |
| 6. Full stack | Hash chain completed, real-to-sim | Simulator | M4 |

**In steps 1 and 2, the user gives up nothing.** This is the core of the adoption strategy. It is not "switch to ours" but "layer it on top of what you use now."

### 27.3 Competitive Positioning Summary

```
LeRobot          "what can you use"          — catalog of policies, data, hardware
Newton/Genesis   "how fast is it"            — physics throughput
Isaac Lab        "inside the NVIDIA stack"   — vertical integration

Electric Sheep   "exactly what it was,
                  why it behaved that way,
                  and whether you can make it that way again"
```

**Becoming "yet another fast robot simulator" has no competitive edge.** Robot Learning Compiler + Reproducibility/Safety Layer is the only defensible position.

---
## 28. Execution Plan

### 28.1 Overall Schedule

| Milestone | Duration | Cumulative | Packets | Type (A/B/C/D) | Gate |
|---|---|---|---|---|---|
| **M0 Contracts** | 1.5 months | 1.5 | ~40 | 60/25/12/3 | 5 IR schemas · normalization · hash, validation infrastructure |
| **M1 Vision vertical slice** | 3.5 months | 5.0 | ~50 | 40/35/18/7 | **Franka + RGB + ACT + pick&place end-to-end** |
| **M2 Evaluation · scaling** | 3 months | 8.0 | ~45 | 45/30/15/10 | Evaluation IR, batch domain, 4k env |
| **M3 Real hardware · safety · real2sim** | 3.5 months | 11.5 | ~48 | 40/30/15/15 | **Real hardware robot deployment + Safety Plane validation** |
| **M4 Expansion** | 3.5 months | **15.0** | ~40 | 35/25/25/15 | PT, IR-C, self-built physics (optional), evidence artifact validation |
| M5 Continuous | | | | | NPU, registry, self-built GPU solver |

**15 months total.** The savings from deprioritizing the self-built physics solver (-3 months) offset the cost of the IR layer (+2 months).

### 28.2 M0 — Contracts (6 weeks, ~40 packets)

**Wave 0 — Validation infrastructure (6, 1 week)** — completed before any feature code
```
P01 xtask skeleton                    A
P02 layering check (§4.2, 9 rules)     A
P03 golden policy + verify-goldens     A
P04 check-scope (diff ↔ packet scope)  A
P05 API digest (ash/Slang/torch/LeRobot schemas)  A
P06 CI layer configuration             A
```

**Wave 1 — Math · conventions (6, 1 week)**
```
P07 Scalar trait + DoubleF32           B
P08 transcendental coefficient generation + Rust  C   ← docs/design/transcendental.md prerequisite
P09 transcendental Slang mirror + bit match   C
P10 convention types + round-trip harness     A
P11 BinnedAcc RFA + property test      C   ← docs/design/deterministic-reduce.md prerequisite
P12 SIMD f64xN + dispatch              B
```

**Wave 2 — Core (5, 1 week)**
```
P13 ECS (archetype, SoA)               B
P14 job system + deterministic partitioning    C
P15 time model types                   A
P16 failure semantics types            A
P17 pool + bump arena + zero-allocation assertion    B
```

**Wave 3 — IR core (12, 2 weeks)** ← **M0's center of gravity**
```
P18 common type system (Unit/Frame/TimeRef)    C  ← docs/design/ir-types.md prerequisite
P19 ImageSpec + transform rules (resize/crop)  C  ← docs/design/image-spec.md prerequisite
P20 Task IR core + node schema               B
P21 Observation IR core + node schema        B
P22 Learning IR core + PolicyHandle contract     B
P23 Deployment IR + SafetyEnvelope schema    B
P24 Evaluation IR schema                     B
P25 normalization + *_hash + 5 property tests    C  ← gate
P26 Cross-IR Check rules                      C
P27 Diagnostic types + diagnostic code dictionary    A
P28 TOML/esgraph serialization + round-trip   A
P29 NodeFactory trait (Task/Learning)        B
```

**Wave 4 — Assets · backend adapters (8, 1 week)**
```
P30 SceneDesc + stable ID              B
P31 glTF importer                      A
P32 MJCF parser (default, compiler)    A
P33 MJCF actuators · sensors · tendons  A
P34 URDF + package://                  A
P35 importer fuzzer                    A
P36 PhysicsBackend trait + MuJoCoCpu    B
P37 openusd spike                      D
```

**Wave 5 — Skeleton (3, 1 week)**
```
P38 RS viewer minimal                  B
P39 editor shell (tab structure)      B
P40 telemetry protocol skeleton        B
```

### 28.3 M1 — Vision vertical slice (14 weeks, ~50 packets)

**Gate: Perform Franka + RGB 2-view + ACT + pick&place end-to-end, from scene assembly → authoring task · observation · learning · deployment → data collection on MJWarp → PyTorch training → evaluation → deployment to the real hardware interface.**

Why pick&place was chosen as the M1 gate: pure locomotion benchmarks like quadruped velocity are a proof of a physics-first platform, not a proof of a vision learning platform. A single pick&place validates the camera · Task IR · temporal observation · policy · dataset · action chunking · sim2real · telemetry · replay **all at once**.

| Wave | Content | Packets | Type |
|---|---|---|---|
| W1 Backend | MJWarp adapter, MJCF scene load, semantic mapping report | 6 | B/A |
| W2 Render · vision | tile atlas, channel contract, sensor realism basics | 7 | B/C |
| W3 Observation IR execution | CPU reference + GPU lowering, LeRobot preprocessing equivalence | 8 | C/A |
| W4 Learning IR execution | PolicyRuntime(Torch), ACT configuration, chunk · ensemble, **LeRobot checkpoint load** | 9 | C/B |
| W5 Safety Plane | Envelope, Watchdog, Fallback, violation scenario suite | 6 | B |
| W6 Env runtime | batch domain basics, reset, randomization, episode recording | 6 | B |
| W7 Dataset | LeRobot read · write, split · identity hash | 4 | A |
| W8 Editor | telemetry, 3D view, **read-only layer graph**, pre- and post-preprocessing images | 8 | B |
| W9 Deployment | policy.esb, es-runtime-embedded skeleton, ONNX export | 4 | A/B |

**Prerequisite design documents:** `observation-lowering.md`, `learning-lowering.md`, `safety-plane.md`, `telemetry-protocol.md`

### 28.4 M2 — Evaluation · scaling (12 weeks, ~45 packets)

| Wave | Content | Packets |
|---|---|---|
| W1 Evaluation IR execution | perturbation kernel, metric computation, report, comparison tools | 10 |
| W2 Batch domain | round_robin, asynchronous inference, chunk buffer, determinism | 8 |
| W3 Policy expansion | Diffusion Policy, SmolVLA, ONNX runtime, layer 4 validation | 9 |
| W4 Newton backend | adapter + backend comparison tool | 5 |
| W5 Performance | 9 metrics instrumentation, memory budget, GPU lowering optimization | 7 |
| W6 Authoring | Python builder completion, LeRobot config conversion | 6 |

**Gate:** Evaluation IR full suite operational, 4,096 sim env × 512 obs env stable, 3 policies passing layer 4, memory budget accuracy ±10%

### 28.5 M3 — Real hardware · safety · real2sim (14 weeks, ~48 packets)

| Wave | Content | Packets |
|---|---|---|
| W1 Real hardware interface | rmw_zenoh native, HIL, real hardware camera drivers | 9 |
| W2 embedded runtime | no-std Observation IR evaluator, Safety Plane, NPU backend | 8 |
| W3 **3DGS real2sim** | es-splat, importer, color · position alignment, LBS binding | 10 |
| W4 Domain gap | diagnostic tools, metrics, reports | 4 |
| W5 Safety Case | traceability, evidence bundle, verify CLI | 6 |
| W6 Editing graph | editable editor, Node SDK | 7 |
| W7 Learning Loop | collect/intervene/distill CLI, intervention labels | 4 |

**Gate: Policy execution on a real hardware robot + Safety Plane operation validation + domain gap report generation.** The real hardware cell enters CI.

### 28.6 M4 — Expansion (14 weeks, ~40 packets)

Path tracer + ReSTIR + SVGF / USD native / IR-C (control) / self-built physics solver prototype (optional) / `es evidence verify` completion / MCP interface / LLM task generation / RoboVerse conversion.

### 28.7 Critical Path Gates

| # | Gate | Timing | Type |
|---|---|---|---|
| 1 | 6 validation infrastructures (Wave 0) | M0 | A |
| 2 | **5 IR normalization invariants** | M0 | C |
| 3 | transcendental ≤2 ULP, CPU/Slang bit match | M0 | C |
| 4 | 5 IR Cross-Check fixtures exhaustive | M0 | C |
| 5 | **LeRobot ACT checkpoint load → identical action** | M1 | C |
| 6 | **Observation IR ↔ LeRobot preprocessing equivalence** | M1 | C |
| 7 | **Franka pick&place end-to-end** | M1 | — |
| 8 | Safety Plane violation scenarios all pass | M1 | B |
| 9 | telemetry + graph view degradation < 1% | M1 | B |
| 10 | built-in node type ID freeze | M1 | — |
| 11 | Evaluation IR full suite deterministic reproduction | M2 | C |
| 12 | 3 policies passing layer 4 | M2 | D |
| 13 | memory budget accuracy ±10% | M2 | D |
| 14 | **real hardware deployment + Safety Plane validation** | M3 | D |
| 15 | domain gap metric computation | M3 | D |
| 16 | Safety Case traceability completeness | M3 | B |
| 17 | evidence artifact bundle validation round-trip | M4 | A |

### 28.8 Operational Rhythm

```
Daily    Start 3–6 dependency-resolved packets in parallel (non-conflicting crates)
         Type A auto-merge / B diff review / C design cross-check / D experiment in progress
         On failure, distinguish spec, oracle, and implementation defects. If a spec defect, fix the packet
Weekly   coherence audit, backlog reordering, type D review
Monthly  gate check, specification update
```

Effective parallelism: type A 4–8 / B 2–3 / C 1 / D 1 per day.

### 28.9 After M5 — Assessment and Improvement Plan

M5's vertical slice (plan V) was the first run of the whole `collect → bake → train → eval → video`
cycle in one pass. This subsection feeds what it exposed back into the specification. The evidence
is the as-built record `docs/design/visible-learning.md` 7.4–7.14 and its open questions 11–15,
`docs/packets/M5/V0…V7a`, `docs/reviews/M0`–`M4` and `M3-W1`, and the 2026-09-15 measurements on
the oracle server (RTX 4090). Gates are referenced by their §28.7 number and risks by their §29
row rather than restated here. Every number that is not a measurement is marked
`Target / Status: unverified`, and performance is stated only through the nine metrics of §12.4
(no single `step/s` figure).

In one paragraph: **the plumbing works, the policy still cannot do the task, and that fact only
became believable after the harness had been fixed three times.** The five IRs, the hash chain, the
Safety Plane and the Evaluation IR all ran end to end in the demo, and everything the collect and
train paths produced (50 demonstrations, the loss curves, `observation_hash`, `lowering_hash`,
`dataset_schema_hash`, the checkpoints) still stands. What was invalidated is the evaluation numbers
alone (visible-learning.md 7.13), and that is what 1, 2 and 3 below are about.

**1. What is lacking**

| # | Item | Spec | As built |
|---|---|---|---|
| L1 | Real robot | §28.7 gates 14/15/16, §24.2 | There is no robot cell. HIL proves the host-side half only (`tests/fixtures/hil/v1_small.eshil` replays identically), and cameras are covered as far as the ROS 2 message boundary through fixture goldens. §29's "Safety Plane requirements do not match the real robot" row is still open |
| L2 | Policy capability (vision) | the demo `evaluation.toml`'s `success_rate >= 0.5` | Re-measured after every harness fix: **0/16** nominal, **0/96** across the six suites (50 demos, 96×96, from scratch, 20,000 steps; visible-learning.md 7.12 phase 2). **V8 (2026-09-15)**: LeRobot's own ACT (pretrained ResNet18, CVAE, DETR decoder) trained by `lerobot-train` for 100,000 steps and evaluated through a bitwise-identical import scores **0/16, 0/16, 1/16** at 20k/50k/100k — the "the model is the problem" hypothesis is rejected (7.16, open question 17) |
| L3 | Policy capability (privileged state) | same | V7a phase 2 measured (2026-09-15): at 20,000 steps nominal **0/16 and 1/16** (two independent runs), 5,000 steps 1/16, six suites **4/96**. Every episode times out at 900 steps and `violation.position` is the largest bucket. `envelope_violation_rate` falls from 0.985 at 1,000 steps to 0.15 at 20,000 — the policy gets smoother as it trains and still never bins the cube (visible-learning.md 7.15). **Stop rule fired**: the next suspect is not model size or schedule length but the physics, the contact model or the expert's trajectories. Before that investigation, rung 3 (V8) is the control experiment — if a verified design trained by a verified trainer also fails, the data, physics or expert are at fault; if it succeeds, our variant was |
| L4 | Oracle-first | §1.4 | Three harness defects were found **after** training finished: collector and evaluation disagreed on what the envelope is measured against (V6), evaluation had no chunk buffer and no temporal ensemble and a seed off-by-one (V6b). Before those, the scripted expert itself scored 0/16 through the harness. The missing oracle was one line: "the harness must pass the expert" |
| L5 | What the evaluation gate is pinned to | §9.4, §10.3 | Through the fixed harness the expert passes 8/8, but its `envelope_violation_rate` is 0.4815–0.5485. The test gate was re-pinned from `< 0.02` to the Deployment IR's own watchdog `max_frac = 0.9` (visible-learning.md 7.13). The argument that §9.4 acts on that number is recorded, but **0.9 is wide enough to catch no policy regression at all.** Whether that is a silent widening is left as a human decision |
| L6 | Safety limits | §9.3 table | `ee_velocity_max`, `contact_force_max`, `min_self_distance` and `min_env_distance` are declared in `tests/fixtures/visible-learning/deployment.toml` and **not enforced** by `es-safety` — the plane has no forward kinematics and no contact query. They are documented limits, not live ones |
| L7 | Inference latency | §8.6, §9.2 | Evaluation models no latency (collection models one tick). The Deployment IR has no latency field and `Evaluation::run` is not given the Learning IR |
| L8 | Physics and render backends | §4.3, §17.2, §15.3 | The backend is a Python subprocess speaking line-JSON and holds one env. `es eval run` uses `MuJoCoCpuBackend` only. The MJWarp and Newton adapters were verified at M4 but are not wired into the demo path. Rendering uses the raster path, §15.3's default, so M4's PT/ReSTIR/SVGF work contributes nothing to this demo |
| L9 | Inference without Python | §2.4 | Both training and evaluation infer through a torch subprocess. The ONNX and Vulkan slots of `InferenceBackend` are reserved by name only |
| L10 | Determinism | §3.5 | On the real MuJoCo/torch CPU stack `--jobs 1` and `--jobs 6` are not bit-identical (6 of 24 cells differ in histograms and episode length; `success_rate` and `envelope_violation_rate` are identical in every cell) — a consequence of `MuJoCoCpuBackend`'s declared tier 3, not a defect in the merge. CUDA training does not reproduce run to run, and `--resident-gpu` is bit-identical on CPU only |
| L11 | Build reproducibility | §5.3's `compiler_hash` | `Cargo.lock` is in `.gitignore` and untracked. The hash chain claims a compiler while the dependency set is not pinned |
| L12 | Privileged observation | §5.1, §7.4 | The `sim_` prefix is a convention with no validator. A Deployment IR in `Real` execution mode referencing a channel no robot can supply is stopped only by a human reading the port name (open question 14) |
| L13 | Pretrained backbone | §8.3 | `pretrained = true` is refused by lowering. There is no path that loads pretrained weights at all. Owner decision on §29's licensing row (2026-09-15): torchvision's ImageNet ResNet18 weights (BSD-3) are allowed for the demo — rung 3 (V8) is the first to use them, through LeRobot ACT's defaults |
| L14 | Context budget | §1.5 | `es-ir` is at 5,947 lines against a 6,000 target. Any improvement that touches the IR (delta action space, a provenance validator, widening `TemporalEncoder`) needs a split packet first |
| L15 | CI stability | §26.2 | `es-telemetry`'s `transport::a_connection_past_max_clients_is_refused` fails intermittently under load on Windows. A refusal must also be accepted as ECONNRESET |
| L16 | Shape of the record | §1.2 | The as-built record lives in one 1,300-line design note. This subsection is the first step of distilling it |
| L18 | Control period | §9.2 `rate.control`, §12.1 batch domains | **The root cause V10 found (2026-09-15)**: the whole demo ran at **200 Hz**, not the declared 50. `Env::new` keeps the physics at the MJCF's `timestep 0.005` and `BatchDomains::single_env()` advances one tick per inference, so one recorded action row is one physics step (measured: the mujoco probe tracks the `es` replay to 0.0002 mm at one substep and diverges by 203 mm at four). Consequences: every dynamic envelope limit is 4x (acceleration 16x) looser per step, the per-tick command increment has a median of 0.0032 rad, a 16-row chunk spans 80 ms instead of 320 ms, and the dataset's `fps = 50` metadata is wrong. The data (replay 50/50), the grasp (50/50 lifted) and the ensemble (range preserved) are not the cause (visible-learning.md 7.18, open question 18). Fix = V11: keep the physics, derive the inference period from `rate.control` and the physics rate (4 substeps), refuse a non-dividing scene, re-collect and re-train |
| L19 | Observation–action pairing | §13.2, the LeRobot convention | **Measured by V10, fixed by V12 (2026-09-16)**: `Env::step` recorded the state *after* advancing the physics beside the action that produced it, and the frame was rendered from that same post-step state; the expert and the evaluator act on the *pre*-step observation, so training learned (s_{t+1} → a_t) while inference asks (s_t → a_t). Noise at 5 ms (0.7 mrad), 0.04 rad per tick at 50 Hz. Fix: the row carries the pre-step state and the frame and trajectory are taken at the same instant (visible-learning.md 7.20). The pairing minimum moved from k=0 to k=+1 by exactly one row; violation rate 0.27 → 0.04; success rates did not move |
| L20 | Trained function ≠ deployed function | §8.3, §2.4 | **V13 (2026-09-16)**: the from-scratch ResNet18 lowered with BatchNorm, and because the lowering is single-sample, `train_act.py --batch 8` accumulates eight one-sample forwards — BatchNorm fits N=1 statistics (instance normalization in effect) while inference uses `eval()`'s running statistics. On the policy's own training episodes the chunk L1 is 0.011 in train() vs 0.031–0.039 in eval() (hold-pose baseline 0.048); recalibrating the running statistics only reaches 0.029. Every IR-graph policy from V2 to V12 was measured on top of this defect. Decision: a from-scratch backbone lowers with `GroupNorm(32)` (the Diffusion Policy precedent; train()==eval() bit-identical is the oracle; visible-learning.md 7.21, learning-lowering.md 5.1). Result: the first grasp, lift and carry to the bin (3/16 training seeds, held without release → timeout); episodes where the cube moves 0 → 14–16/16. Held-out is still 0/16 |
| L21 | The unit of the plane's watchdog clock | §9.4, §12.1 | **V17 (2026-09-16)**: `SafetyPlane`'s `period_us` is the *control* period, and both loops handed it the *simulation* tick as `now` (four per control step since V11). It could not show while a chunk arrived every control tick — `accept` stamps `last_chunk_tick` first, so the gap was always zero — and the moment `rate.inference` was honoured, `inference_deadline` fired on 25,920 of 28,800 ticks. Fix: `DomainRunner` uses its own `control_tick` and `es_eval::runner` the episode step (`es-safety` untouched). A 5 Hz re-plan has up to 180 ms between arrivals, so the demo's `inference_budget` and its watchdog went 40 → 240 ms (INV-12: widen, never disable; every safety limit, `stale_observation` at 80 ms and the 0.5 acceptance are unchanged) |
| L22 | The evaluator's inference-latency model | §12.3 | **Measured by V17 (open question 24)**: collection models `RuntimeHints::expected_latency_ms` through `AsyncInference`, so its first chunk lands at tick 1, while the evaluator executes at tick 0. The `qpos‖qvel` traces of one seed diverge at tick 1 (the cadence is the same on both paths: 12 calls per 120 ticks). This is ladder rung 5's packet **Fixed by M7/T7 (2026-09-16)**: `es_eval::runner` drives `es_env::AsyncInference` with the same `latency_ticks`; the two paths' `qpos ‖ qvel` traces of one seed are identical for all 120 ticks. Under the declared latency V19b's ACT still passes (held-out 0.9375) while the IR-graph policy falls to 0.0625, and the imported ACT's *declared* 200 ms is now load-bearing (visible-learning.md 7.30, open question 32) |
| L23 | The envelope under the declared cadence | §9.3 | **V17 (open question 25)**: once rows 0..9 of a chunk execute in order the policy's `envelope_violation_rate` is 0.998 — `acceleration_max = 20 rad/s²` allows 0.008 rad of change in the per-tick step and the policy's rows move 0.02–0.03 rad — and the `EnvelopeViolationRate` watchdog (`max_frac 0.9`) latches the fallback for about a tenth of every run. The expert is at 0.29–0.30 (down from V6b's 0.48–0.55). Nothing new is commanded: the demonstrations' own command stream reaches the plane unaveraged. Open decision: (a) widen to the acceleration the servo actually delivers (a Deployment IR decision), (b) a delta action space, (c) report only. **V18 measured (a) (2026-09-16, fixture untouched, no retraining)**: `acceleration_max` 20 → 40 takes held-out success from 0.0625 to **0.625**; 80 gives 0.625 / 0.625 (training / held-out) with the violation rate 0.998 → 0.90 → 0.56 and the fallback latch 2,690 → 767 → 53 ticks. Widening velocity to 4.4 as well trips `action_rate.first_diff_max` (0.08 rad) and is worse (0.4375). 40 spends 38 % of the rotor's stall torque (2.94 N·m), 80 spends 76 %. **Decided (owner, 2026-09-16): 80 adopted**, `velocity_max` unchanged; the re-derivation lives in the fixture's comment and the decision in visible-learning.md open question 25 (packet V18) |
| L24 | `execution_hash` under load | §5.3 | **V18b (2026-09-16, open question 26)**: an evaluation reproduced while a render ran on the same GPU gave a bit-identical report and `evaluation_hash` and a different `execution_hash`. The slots computed at run time (`runtime`, `hardware_capability`) read measured values. A provenance question, not a correctness one; a packet should pin the slots to declared values and keep the solo/contended pair as an oracle |
| L17 | How gate 7 is judged | §28.7 gate 7 | The demo is SO-101 cube-into-bin, not a Franka with two RGB views. Whether gate 7 is recorded as met with the substitution named, or stays open as written, is undecided (visible-learning.md open question 9) |

**Three rules this subsection fixes.** The rest is left to human judgement, but these three were
paid for by M5 and therefore become specification.

1. **A vertical slice carries the oracle "the harness passes the expert" before it trains anything**
   (§1.4). If the scripted expert — not a policy — cannot be driven through the same evaluation
   path, the same plane and the same success predicate, then every training number afterwards
   measures the harness rather than the policy. M5 learned this late three times.
2. **An invalidated measurement is marked invalid, never deleted.** The tables of 7.8–7.11 are still
   there and 7.13 wrote the reason for their invalidity above them. A number that disappears gets
   rediscovered; a number carrying its invalidation does not.
3. **No performance claim rests on a metric that does not reproduce** (§12.4). If a metric depends
   on a per-worker clock, the report does not even declare it until it is fixed.

**2. Where the wall-clock goes**

The cycle is measured (visible-learning.md 7.11 and 7.14, `docs/packets/M5/V5-fast-cycle.md`;
oracle server, 2026-09-15).

| Phase | Measured | Cause |
|---|---|---|
| nominal, 16 episodes | 4:16.49 / 4:16.68 (two independent runs, `report.json` and every frame byte-identical to each other) | a physics subprocess holding one env |
| 6 suites, 96 episodes | 25:56 sequential → **5:49** with `--jobs 6` | cell-level sharding. Before the per-shard BLAS/torch thread caps it was *slower* than sequential, projected past four hours (~90 threads on 16 cores, load ~47) |
| training, 20,000 steps at batch 8 | 10:53 default / 10:57 `--resident-gpu` / 12:53 bf16 / **10:00** `--compile` | no knob moves it much on an RTX 4090. At batch 8 the module runs one sample at a time, so it is kernel-launch bound |
| training, batch 64 with linear lr scaling | 1:24:31, `final_loss` **NaN** | the linear-scaling convention does not hold for this model on this 50-episode dataset |

What would remove each, and what it is worth:

- **Remove the per-sample loop in the batch lowering.** This is the root cause of the training
  time, and none of `--resident-gpu`, bf16 or `--compile` touched it. `Target / Status: unverified`.
- **Large batches with warmup and an lr schedule.** Linear scaling was measured to diverge, so what
  is needed is a schedule, not a convention. `Target / Status: unverified`.
- **Episode-level sharding.** Blocked today by `es-env`'s episode counter — episode 5's initial
  state is not reproducible without having run episodes 0..4. With a seek, the single-cell nominal
  run parallelizes too.
- **An in-process, multi-env physics backend.** The only lever that pays on the collect and the
  evaluation path at once, and the adapters already exist (L8).
- **Leave the frame path alone.** Raw `.bin` writes and the `cv2` mosaic are not the dominant term.
- **The collect and bake phases have no wall-clock on record yet.** What is missing from the table
  above is unmeasured, not fast. The next server run takes them in the same format.
  `Target / Status: unverified`.

**The limits of the measurement are themselves an optimization target.** Of §12.4's nine metrics,
`physics_steps_per_sec` and `actions_per_sec` divide each worker's own clock under sharding and do
not reproduce. The demo's `evaluation.toml` declares neither, but claiming throughput requires
fixing them first.

**Stop rule.** Once one cycle (collect, bake, train, eval) is under 30 minutes, stop the speed work.
The largest remaining measured term is training at roughly 11 minutes, and below that the
bottleneck is judgement rather than the machine.

**3. Accuracy levers**

(a) What would make a learned policy actually succeed, ordered by information per hour:

| Rank | Lever | Why |
|---|---|---|
| 1 | Read V7a's privileged state policy result first | If a policy handed the cube's exact pose fails, the remaining suspects are the physics and the expert, not the graph. One experiment separates two questions |
| 2 | 200–500 demonstrations, 224×224 or a wrist camera | Finding a 25 mm cube in 96×96 from scratch is the problem the current vision policy is solving |
| 3 | A pretrained visual backbone | Needs `pretrained` support in lowering plus safetensors weights first (L13); rung 3 (V8) sidesteps this through the LeRobot checkpoint loader |
| 4 | Teacher (privileged state) → student (vision) distillation | The remaining path if 1 passes and 2 fails |
| 5 | Chunk 50 with the temporal-ensemble decay tuned to the envelope | Even the expert produces a 0.48–0.55 violation rate through the blend. A policy being clamped is not the policy's problem alone |
| 6 | Training-time augmentation reusing the existing perturbation kernels | `light_intensity` and `light_direction` already run in evaluation; using them in training makes the evaluation suite the training distribution |
| 7 | 100,000 steps with a warmup schedule | Only meaningful after 2 and 3. At 20,000 steps `final_loss` is already 0.0179 |
| 8 | Smoother expert pacing that survives the blend | The pacing uses 90 % of each limit today, and the blend jitter eats the remaining 10 % |

(b) What would make the measurements trustworthy:

- **Model inference latency in evaluation** (L7). Evaluation is one tick better off than collection.
- **More held-out seeds and per-metric confidence intervals.** Over 16 episodes the difference
  between 0/16 and 1/16 is noise — V3, V2b and V1c oscillating between 0.0625 and 0.1250 is the
  example, and V6b invalidated all of those numbers.
- **Carry the expert as a control group in every evaluation report.** With the expert's own success
  rate and violation rate beside the policy's, on the same seeds and the same suites, the table
  itself says whether a number is the harness's doing or the policy's.
- **Break the clamps down per stage and per joint.** `events.json` already carries it: nominal is
  `violation.position` 2,031 against `acceleration` 843 and `velocity` 439, and the gripper joint
  as the likely tenant is recorded only as a hypothesis.
- **Pin "the harness passes the expert" as a prerequisite oracle of every vertical slice** (L4).
  §1.4 already required it.

**Stop rule (accuracy).** If levers 2 and 3 are both spent — 500 demonstrations, 224×224, a
pretrained backbone — and nominal is still 0/16 while the privileged state policy passed, the answer
is neither more data nor a bigger model. What is left is the mapping from observation to action, and
the next move is lever 4 (distillation). Conversely, if even the privileged state policy is 0/16,
levers 2, 3 and 4 are all skipped: L3's stop rule already names the suspects for that case.

**4. The plan (packet ladder)**

Each row is one §1.2 packet and each oracle is a runnable one-liner. The order follows information
content and blocking relations.

| Rank | Packet | Question it answers | Oracle (one line) | Type |
|---|---|---|---|---|
| 1 | **V7a phase 2** (done) | Can this graph do the task when the observation contains the answer | `es eval run` over 16 nominal seeds → `success_rate` in `report.json` (stop rule L3) | D |
| 2 | **M5-R1 harness-first rule** | How does the same defect stop being found after training | A new vertical-slice packet's acceptance names the expert-through-harness test | A |
| 3 | **V8 external ACT** (done: 0/16, 0/16, 1/16; bitwise equivalence — the model hypothesis is rejected; V19/V19b: the same ACT retrained on V15's demonstrations scores held-out 15/16 on the committed documents — the thesis is met) | Does a policy designed and trained outside (LeRobot ACT, `lerobot-train`) succeed on the same demonstrations through the same harness — and does our runtime reproduce it identically | v3.0 export (+`meta/stats.json`) → `lerobot-train` → the `lerobot.rs` loader → gate-5-style bitwise equality → `es eval run` nominal `success_rate` (the demo's acceptance of 0.5, unchanged) | D |
| 4 | **V9 showcase render** (done: four expert successes, 1280×720 mp4; V18b adds two held-out policy successes + a 16-episode mosaic) | Is there a video a person can watch | Replay-rendering the recorded state trajectory at the observation resolution is bit-identical to the recorded frames, and a 1280×720 H.264 mp4 comes out (the expert's 8/8 first) | B |
| 5 | **Latency in evaluation** (V17 measured: the two traces diverge at tick 1, L22) | Do evaluation and collection share one time contract | `cargo test -p es-eval`: with a declared latency, tick 0 is a chunk underrun | B |
| 6 | **Re-pin the demo acceptance** | If not 0.9, what catches a policy regression | `es eval run` passes the expert and fails a policy whose violation rate exceeds the expert's worst | C |
| 7 | **Commit `Cargo.lock`** | Does `compiler_hash` reproduce a build | `cargo xtask ci` checks the lock file and a `--locked` build | A |
| 8 | **The telemetry flake** | Is CI green under load | `cargo test -p es-telemetry transport` passes 100 repeats | A |
| 9 | **Vectorize the batch lowering** | What dominates training time | A 40-step loss curve byte-identical to the current path at lower wall-clock | B |
| 10 | **`Env` episode seek** | Is episode-level parallelism possible | State after a seek is bit-identical to replaying 0..n | B |
| 11 | **Wire MJWarp into evaluation** | Is the backend really replaceable | `es backend compare --backends mjwarp,mujoco-cpu` reports max \|dqpos\| and both reports judge the same | C |
| 12 | **The unenforced safety limits** (L6) | Are the declared limits alive | `cargo test -p es-safety`: a command past the EE velocity limit is counted `Clamped` (INV-12 — widen, never disable) | B |
| 13 | **Inference without Python** | Does the deployment path run without Python | Tier-4 equivalence (§8.9) with the torch path on the same checkpoint | C |
| 14 | **Split `es-ir` → provenance validator** | Can a privileged channel leak onto a real robot | A `Real`-mode Deployment IR plus a `sim_` channel is a validation error | C |
| 15 | **M5 review and distilling the design note** (done: `docs/reviews/M5.md`, follow-ups R1–R10) | Has the record come back into the specification | `docs/reviews/M5.md` exists and `cargo xtask check-spec-refs` passes | A |

**What is not on the ladder, and why.** The native physics solver stays cut as §1.9 reduction 1, and
rungs 12 and 14 stand in its place — what is needed now is not a new solver but a verdict on the
contact model of the solver already in use. The PT/ReSTIR render path (§1.9 reduction 2) does not
enter the policy's observation: there is no evidence that a 96×96 raster image is what blocks the
policy. Rung 4's showcase render, however, is an offline replay of recorded trajectories, so PT can
be used there for the first time as soon as it runs headless on the server. Editor and authoring
work is absent — not one defect M5 exposed had a UI as its cause. Rungs 1, 2, 4, 6, 7 and 8 wait on
no single human's judgement and can start in parallel (§28.8, types A and B); 3, 13 and 14 each wait
on the row above them or on a split packet.

**M6 is the second track: quadruped locomotion (owner decision, 2026-09-15).** The demo's purpose
is to show "learning visibly worked" to a person, and the project's claim is "a policy designed and
trained outside runs in our runtime with the same semantics" (§8, §1.9). Once rung 3 (V8) proves
that with imitation learning on an arm, M6 proves the same claim again with an entirely different
policy family and morphology: a MuJoCo Playground Go1/Go2 joystick policy trained by *their* PPO
trainer (we build no RL trainer), its MLP imported into the Learning IR, run on our MuJoCo CPU
backend under our Safety Plane and perturbation suites (pushes, friction, payload), and rendered by
rung 4's showcase. Prerequisite packets: a primitives-only derivative of the quadruped model (our
loader rejects `mesh` geoms), the observation stack expressed through the Observation IR's history
window, a parity oracle between the training environment and our physics (both backends on the
same XML, §17.2), and export of the observation-normalization parameters. Research digest:
`docs/api-notes/mujoco-playground-quadruped.md`. The real-robot gates 14/15/16 stay open until a
robot cell exists. If even rung 3 (V8) fails, a **physics, expert and action representation**
investigation is inserted before M6 (**fired, 2026-09-15** — packet V10, a scene diagnosis: replay the
expert's recorded actions through `es eval run`, judge grasp versus push from the gripper's contact
forces, check whether the temporal ensemble survives the grasp window; then resolution and
demonstration count one at a time). **V10's verdict (2026-09-15)**: all three suspects are cleared and the cause is L18's control period. The next packet is V11, and nothing else in plan V moves until its number exists. **Progress (2026-09-16)**: V11 (control period, 0/16) → V12 (pairing, L19, 0/16) → V13 (GroupNorm, L20, the first carry) → V14 (200 demonstrations, held-out carry 9/16, no release) → V15 (the release in the data and the 0.85 predicate, 0/16) → V16 (the frozen-input fixed point diagnosed, open question 23) → **V17 (both paths honour the declared 5 Hz re-plan; V15's 40k checkpoint re-measured without retraining: the first 4 releases in 48 episodes, held-out 1/16 = 0.0625 — L21, L22, L23)**. V15's checkpoint is still the best policy and every evaluation number in sections 7.19–7.24 is re-dated. → **V18 (envelope ablation: `acceleration_max` 40 gives held-out 0.625 — the first time plan V clears the 0.5 acceptance, L23)**. → **fixture at 80 + V18b (full six-suite sweep `passed = true`, nominal 0.625, three showcase mp4s) — the first time plan V's acceptance holds on the documents as committed (visible-learning.md 7.28).** → **V19 (LeRobot's real ACT retrained on V15's demonstrations, bitwise reproduction, held-out 13/16 at 20) → V19b (the same checkpoint on the committed documents: training 16/16, held-out 15/16, six-suite `passed`, zero fallback)**. M5 review: `docs/reviews/M5.md` (B-1 suite independence, F-1..F-10 rule candidates, R1–R10). One variable at a time, each packet exposing the next defect: contact-model verification (judged by §17.2's backend
comparison), a delta action space (§8.5, preceded by the `es-ir` split), and a redesign of the
expert's trajectories. Either way, rungs 2, 5 and 6 of the ladder come first — if the harness
cannot be trusted, neither conclusion is a measurement.

### 28.10 M7 — Usability, Training Throughput, Render Quality (plan U)

After M5 closed (`docs/reviews/M5.md`) the owner asked for four things on 2026-09-16: **make
training easier to use**, **make `es-editor` friendlier**, **raise training speed and task
accuracy together**, and **raise the quality of both render paths, RS and PT**. This subsection
turns the four into a §1.2 packet ladder. §28.9's three rules (the harness passes the expert
first · an invalidated measurement is marked, never deleted · no performance claim rests on a
metric that does not reproduce) apply unchanged, and the M5 follow-ups R1–R10 that this
subsection's oracles lean on (R2 latency in evaluation, R4 dataset provenance, R7 `Cargo.lock`)
are pulled onto the ladder. M6 (quadruped) stays parked until the owner's go/park decision;
this subsection does not replace it.

**Where things stand, measured (2026-09-16, oracle server RTX 4090)**

| Area | As built | Cost |
|---|---|---|
| Using training | The external ACT path is six commands — `export → lerobot-train → import-lerobot → eval run → showcase → encode` — some thirty flags, and hashes copied by hand (`docs/packets/M5/V19`'s server script). The IR path is `bake → lower → train_act.py → pack → eval`. There is no `es train` (§13.1) and §19.3's `training_hash` is all zeros except the `dataset` slot | nine scripts for a person to assemble one cycle |
| Training speed | The lowering is single-sample (`unsqueeze(0)`), so `--batch 8` is eight forwards accumulated. 20,000 steps ≈ 11 min (kernel-launch bound); batch 64 with linear lr scaling is NaN (7.11). No pretrained-backbone path (L13) | one retrain 11 min; the external ACT's 100k steps 34 min |
| Accuracy | IR-graph policy held-out 0.625 (V18b), external ACT 0.9375 (V19b), on the committed documents | — |
| Editor | A bundle path in a text field → four tabs Graph/Telemetry/Images/Diagnostics, six edits over the Task IR. No view opens a run (`report.json`, `events.json`, `.estraj`, frames), no 3D view, no parameter inspector, no telemetry producer (a replay source only) | most of §23.3 is empty |
| Render RS | A flat triangle scan, O(pixels × triangles), re-tessellated and re-uploaded every frame: **80 ms/frame** at 1280×720, one camera (2,978 triangles, 7.17). Lambert plus a constant ambient; no shadows, no highlights, no textures | a 1,407-frame showcase ≈ 2 min |
| Render PT | Diffuse only, emissive triangles the only lights, ReSTIR combined with the biased `1/M` weight, à-trous without SVGF's V, no tone map → no `Rgb8` → the showcase is RS-only (7.17 item 5) | §15.3's SSIM contract unmeasured |

**Three rules this subsection fixes.** They are the fences that keep the four goals off each
other's feet.

1. **The observation pixels of a committed document do not change unless the document does.**
   A renderer improvement arrives as a new `RenderConfig` field whose default is today's output,
   and the default path is pinned bit for bit by `tests/golden/render/` and
   `tests/fixtures/visible-learning/frames`. A change that moves one byte is not a renderer
   change but a **document change**, and it invalidates that document's checkpoints (§5.3, §15.3).
2. **`es train` fills every §19.3 slot with a real value or marks it unset.** No fabricated
   digest enters `training_hash` (the rule of `docs/design/learning-loop.md` section 1, promoted
   to a command). When it calls an external trainer (`lerobot-train`), that trainer's config
   file, version and seed go into `config.json` and `hardware.json`.
3. **The editor decides nothing in `app.rs`** (`docs/design/editor-shell.md` section 2). The run
   browser, the 3D replay, the inspector and the launch panel each get a headless view-model and
   its tests first. What a display-less CI cannot judge is, per §1.4, design and not
   implementation.

**One sentence on accuracy.** The M5 review demoted the IR-graph policy to a lowering oracle.
This subsection does not reverse that, but T3, T4 and T5 change the lowering itself (batch
axis, schedule, pretrained backbone), so their result is re-measured **once** on the committed
documents (V15's 200 demonstrations, the same Evaluation IR). Stop rule: if held-out does not
exceed V18b's 0.625 after T5, IR-graph tuning ends here and the product is the external path's
speed (T1, T2). If it does, the number goes into the as-built record from 7.30 on, beside V19b.

**The packet ladder.** The three tracks touch disjoint crates (T: `es`, `es-policy`, `es-data`,
`python/es`; E: `es-editor`, `es-telemetry`, one hook in `es-eval`; R: `es-render`,
`es-env/render.rs`, `es/cmd/showcase.rs`), so within a wave they run in parallel. Each row is
one §1.2 packet and each oracle is a runnable one-liner. Design notes:
`docs/design/training-recipe.md` (T), `editor-shell.md` extended (E), `renderer.md` extended
(R). Packets: `docs/packets/M7/`.

| Wave | Packet | Question it answers | Oracle (one line) | Type |
|---|---|---|---|---|
| 1 | **T1 `es train`** | Does one document (`training.toml`: dataset, bundle or LeRobot policy type, steps/batch/lr/seed/schedule/checkpoint marks, device, interpreter) drive both the IR path and the external path with one command and write §19.3's `training/` with real values | `cargo test -p es --test cli train_`: the `--dry-run` command plan is byte-identical to a golden; the same recipe twice → one `training_hash`; the IR path at 40 steps on the fixture packs a bundle `TorchRuntime` opens (SKIP with a printed reason without `ES_PYTHON`) | B |
| 1 | **E1 run browser** | Does opening `<run>/` show the cells × outcome × metrics table, a per-episode Safety Plane event timeline (stage, joint, tick) and a frame filmstrip | `cargo test -p es-editor`: `RunView::open` on the fixture run reproduces `report.json`'s numbers and the timeline buckets sum to `events.json`'s counts | B |
| 1 | **E2 3D replay** | Can an `.estraj` be replayed inside the editor with no physics and no GPU (`TriScene::from_scene_with_poses` → project → depth sort → `egui::Mesh`, scrubber and play) | headless: the projected triangle set of tick t is a pure function of (trajectory, camera) — a sort-order golden; `app.rs` compiles only | B |
| 1 | **R1 instance transforms + software two-level BVH** | With one BLAS per geom built once and only poses uploaded per frame (the TLAS rebuilt on the CPU each frame), how far does 80 ms fall, and does preserving the scan order keep the output bit-identical | every existing golden unchanged; BVH == flat scan bitwise (CPU and GPU, Cornell + SO-101 at three ticks); showcase ms/frame before and after recorded (target < 5 ms, `Target / Status: unverified`) | B |
| 2 | **T3 batched lowering** (§28.9 rung 9) | When `lower_to_torch` emits a leading batch axis and `train_act.py` uses real batches, where does the training time go | `cargo test -p es-policy`: the batched module at N=1 vs single-sample ≤ tier-4 fp32 (CPU); `train()==eval()` stays bit-identical; 20k-step wall-clock before and after recorded (observation) | B |
| 2 | **E3 inspector, search, open** | Is there a widget per `NodeSchema` `ParamType`, a node search, and opening by recent files and drag-and-drop | headless: the inspector model covers every `ParamType` and `SetParam` round-trips through it; zero new dependencies | B |
| 2 | **R2 the RS look** | Can shadow rays, a hemisphere ambient, Blinn-Phong and SSAA be switched on (opt-in `Shading::Full`) while the default look stays unchanged | existing goldens unchanged; `cornell_rs_full_*` goldens generated by the CPU generator and matched by the GPU bitwise / ≤ 1 ULP; the `frames` fixture untouched | B |
| 3 | **T2 `es loop cycle`** | Does collect → train → eval (→ showcase) run under one document and one `loop.jsonl` (§13.3) | `--dry-run` plan golden; a fixture cycle with the scripted expert → three `loop.jsonl` rows whose hashes chain (`ES_PYTHON`) | B |
| 3 | **T4 lr schedule + large batch** | With warmup + cosine and a `scheduler.json`, does batch 64 stop diverging | scheduler unit golden (exact floats); server: batch 64 with the schedule has a finite loss at or below batch 8's at equal samples seen (observation, not a gate) | B/D |
| 3 | **T5 pretrained backbone** (L13) | Does `pretrained = true` lower to torchvision's ImageNet ResNet18 weights (BSD-3, §29 licence row) with FrozenBN, and does `base_model.lock` pin their provenance | `train()==eval()` bit-identical; a `base_model.lock` hash mismatch is refused; a provenance test downloads and verifies the blake3 (ignored tier) | B |
| 3 | **E4 live telemetry** | Does `es eval run --telemetry <addr>` publish `StepEvent`s and the §12.4 nine metrics through `es_telemetry::transport::Server`, and does the editor attach | loopback: N frames published → the client receives N in order; gate-9 overhead measured (observation) | B |
| 3 | **R3 PT quality** | With NEE + MIS (emissive, directional, sky), an unbiased ReSTIR combination (pairwise MIS) and an `approx`-based tone map, does PT emit `Rgb8` and does §15.3's SSIM contract get measured | tone map CPU == GPU bitwise; `cornell_pt1spp` unchanged with NEE off; SSIM(RS full, converged PT) recorded (threshold fixed after measurement, `unverified`); `es video showcase --path pt` writes frames | B/C |
| 4 | **T6 training augmentation** | Are random shift/crop and brightness recorded in `augmentation.json` and reproduced from the seed | seed → augmented tensor golden; `training_hash` moves with it | B |
| 4 | **T7 latency in evaluation** (M5 R2, L22; done — visible-learning.md 7.30) | With a declared `expected_latency_ms`, is tick 0 a chunk underrun, and do the collection and evaluation traces agree to the last tick | `cargo test -p es-eval` | B |
| 4 | **T8 episode seek** (§28.9 rung 10) | Does `--jobs` shard per episode so that the 16-episode nominal run parallelises too | state after a seek == replaying 0..n bitwise; nominal wall-clock before and after (observation) | B |
| 4 | **E5 launch panel** | Does the editor spawn `es eval run` / `es train` as a child process and attach over `--telemetry` (§23.1: a client, not a host) | headless: the command line the panel model builds is a golden; the exit status is shown | B |
| 4 | **R4 temporal accumulation + SVGF variance** | On the static showcase camera does frame accumulation stand in for spp, and does the variance term drive the filter width | accumulated N × 1 spp vs 1 × N spp within a stated tolerance; goldens | B |
| 5 | **U-measurement** | The IR graph after T3–T5 retrained on V15's 200 demonstrations and re-measured on the committed documents; one `es loop cycle` wall-clock; showcase mp4s from R1–R3 (RS full, PT) | `report.json`, `training.lock` and ms/frame recorded in the as-built sections from 7.30 on, beside V18b and V19b | D |
| 5 | **M7 review** | Has the record come back into the specification | `docs/reviews/M7.md` + `.ko.md`; `cargo xtask ci` green | A |

**What is not on the ladder, and why.** **Mesh geoms** (the upstream SO-101 STLs) and **MJCF
textures and materials** (builtin checker, `texrepeat`, `specular`, `shininess`) each need
`es-assets` work — loading `Shape::Mesh`, fields on `Material` — and both move `scene_hash`,
invalidating the committed checkpoints; they become R5 and R6 after the owner's decision.
**MJWarp in evaluation** (§28.9 rung 11) opens only after T8 has parallelised the nominal run
and the gain is still there, because a tier-3 backend changes the numbers. **The sensor realism
pass** (§18.3) was never asked for. **M6** is parked.

**Owner decisions.** (1) Allow one accuracy re-measurement of the IR graph (with the stop rule
above) — default: allowed. (2) The hash consequence of R5/R6: accept re-collection and
retraining or not. (3) The `lerobot` version `es train`'s external path pins (0.6.1,
`docs/api-notes/lerobot-config.md`). (4) Committing `Cargo.lock` (M5 R7) — needed if
`es train`'s `hardware.json` is to name a reproducible build.

**M7 outcome (2026-09-21, `docs/reviews/M7.md`).** Every row of the ladder closed — T1–T8, E1–E5,
R1(+R1b)–R4, U — and, on the owner's directive, **E6** outside the ladder (plain words, `i18n` string
tables, a system CJK font fallback, a home screen, native file dialogs). The three rules above held:
not one committed observation pixel moved (every renderer improvement is an opt-in `RenderConfig`
field), `es train` wrote no invented digest, and every editor panel had a headless view-model first.
Measured (RTX 4090): 20,000 steps 675 s → 137 s (T3); batch 64 + warmup-cosine 36 % below batch 8's
loss at equal samples, 26 s against 107 s (T4); one cycle (collect 200 → expert gate → train →
evaluate → showcase) 30:19 (T2, of which evaluation 20:24); one 1280×720 frame 80 → 5.2 ms with every
bit unchanged (R1), the `Full` look 9.9 ms (R2), PT NEE 64 spp 262 ms and 4 spp accumulated 29.8 ms
(R3, R4), ReSTIR bias −1.76 % → +0.10 %; live telemetry +0.16 % (E4). **The re-measurement (U)**: on
the committed documents under T7's latency model, U0 (T3+T4) held-out 0.25, U1 (pretrained) diverged
(NaN at step 4,517), U2 (augmented) 0.1875, **U3 (pretrained + augmented) 0.5625 — the first IR-graph
policy to pass the demo's acceptance** (V18b re-measures at 0.0625 on the same evaluator). The stop
rule above fires on the literal reading (0.5625 < 0.625) and not on the like-for-like one; which to
take is the review's first human decision. T8's episode partition measured 2.0×/2.2× and did not ship:
the Safety Plane's violation-rate window carries across episode boundaries and `StepEvent::tick` is the
cell's cumulative clock, so committed numbers move — also a human decision. Follow-up packets R1–R10
are in the review.

### 28.11 M8 — Continuing Reinforcement Learning of External Policies (plan S)

After M7 closed (`docs/reviews/M7.ko.md`) the owner decided on 2026-09-21: **continuing the
reinforcement learning of a policy trained elsewhere first**, the editor's easy UI deferred
until more ideas gather, M6 (quadruped) still parked. Of the review's human decisions, this
section takes **the episode boundary (S-1, S-2)**, pinned to §9.4, §10.5 in the same commit
as this section (both "yes"; the grounds are §13.1's "an episode is where the stream ends"
and §10.4's schedule independence; demonstration numbers already committed are marked as
measured under the old semantics per §28.9 rule 2 and re-measured in the same packet). The
stop rule's (§28.10) reading and §15.3's SSIM threshold remain the owner's. §28.9's rule set
and §28.10's rule set apply unchanged.

**The claim.** The project's thesis was "our runtime reproduces, with the same semantics, a
policy designed and trained externally" (§8, §1.9), and M5 proved it with imitation learning
(LeRobot ACT). Plan S proves the same thesis once more with **a different policy family (PPO
MLP) and a different training signal (reward)**, and goes one step further: it **keeps
training the reproduced policy in our sim**, and records the success rate before and after
continuation in the same table, from the same Evaluation IR. That table is M8's deliverable.

**Where things stand, as-built (2026-09-21)**

| Area | What exists | What is missing |
|---|---|---|
| Importing | `es import lerobot-config`, `es policy import-lerobot` (ACT checkpoint → bundle, bit-identical oracle); `Normalizer{Forward/Inverse, MeanStd}`, `StateEncoder{Mlp}`, `PolicyHead{Regression}` nodes; the brax parameter format survey in `docs/api-notes/mujoco-playground-quadruped.md`; the three gaps in `quadruped-track.md` 3.4 (activation fixed to ReLU, no activation before the head, `tanh` absent) | an RL checkpoint parser (brax/orbax, rsl_rl, rl_games), activation and squash parameters, adapter documents, `es policy import-rl` |
| Training | `es train` (imitation: the IR path `train_act.py`, the external path `lerobot-train`), real values in §19.3's slots, `base_model.lock`, the schedule, augmentation, and pretrained backbone | an RL trainer, `[init] policy`, `init.lock`, a rollout path (`es-py` has no `Env`/`SafetyPlane` bindings — only the four builders) |
| Evaluation | the six-suite Evaluation IR, `--jobs` (per cell), `Env::seek_episode` (T8, bit-identical) | the `(cell, episode)` partition (T8b, waiting on the episode-boundary decision — resolved in this section) |
| Source policy | None. jax, brax, playground, and rsl_rl are not installed on the server. Playground has no SO-101 task (only quadrupeds like Go1, Panda, etc.) | an SO-101 source checkpoint — train one directly on the 4090 in the source framework to pin the recipe |

**Four rules this section pins down.**

1. **RL is a trainer, not an IR** (§13.4). The deployed graph is nothing but §8.3's nodes;
   the value head, the log std, and GAE live only in `training/`. Task IR has no RL (rule
   6).
2. **The rollout is `es-env` and the Safety Plane stays on** (§13.4, INV-12). The moment a
   trainer calls another simulator directly, that number is no longer sim-to-sim.
3. **The adapter declares, the code does not guess** (§14.4). Joint order, units, position
   vs. torque, and observation layout are a robot-specific document, and a mismatch is a
   named error. Pickle is opened only in Python (INV-16).
4. **No new trait** (INV-17). The trainer is a module of `es-data`, `es`, `python/es`, and
   `es-py`'s bindings are pyclasses, not an extension point. Throughput is `Target / Status:
   unverified` until measured.

**The packet ladder.** Wave 0 is what this campaign leans on among M7 review's follow-ups.
Waves 1–3 run in parallel because the crates don't overlap (R: `es-safety`, `es-eval`,
`es/cmd/eval.rs`; E: `es-editor`; S2a: `es-ir`, `es-policy/lower`; S2c:
`python/es/rl_source` and the server; S1: `es-data/training.rs`, `es/cmd/train.rs`; S4a:
`es-py`), and each row is one §1.2 packet whose oracle is a runnable one-liner. Packets:
`docs/packets/M7/P-M7-R1.md`, `P-M7-R12.md`, `docs/packets/M8/S*.md`. Design notes:
`docs/design/rl-continuation.md` (new), `evaluation-execution.md` extended (R1),
`editor-shell.md` extended (R12).

| Wave | Packet | Question it answers | Oracle (one line) | Type |
|---|---|---|---|---|
| 0 | **R1 episode boundary + T8b partition** (`P-M7-R1`) | If `begin_episode` clears the window and `tick` is episode-relative, is the `(cell, episode)` partition bit-identical to sequential execution, and does the 16-episode nominal run halve | `cargo test -p es-safety window_is_cleared_at_begin_episode`; T8 oracle 3 goes from a tripwire to a parity assertion (`--ignored`, server); the committed `evaluation.toml` at jobs 1 and 4 → per-cell `report.json`, `events.json`, `.estraj` bit-identical, jobs 4 wall-clock ≤ 55% of jobs 1 (nine metrics); a U3 held-out re-measurement row | B |
| 0 | **R12/R16 live table** (`P-M7-R12`) | Are rows keyed by `(stage, cell)`, and does a cell that started before attach still get a row | `cargo test -p es-editor live_run_`: the gate's `nominal-00` followed by the evaluation's `nominal-00` are two rows; a stream with no `cell.begin` gets a "joined late" row; the fixture-fold oracle unchanged | B |
| 1 | **S2a activation and squash** | Do `StateEncoder{Mlp}` gain `activation` (relu, elu, swish, tanh) and `PolicyHead{Regression}` gain `squash` (none, tanh), with absent = default = today's hash | `cargo test -p es-ir committed_learning_hash_is_unmoved_by_activation`; `cargo test -p es-policy lower_mlp_activations` (torch reference vs. lowering, 4×2 combinations, ≤ tier-4) | B |
| 1 | **S2c source policy** | Does a reach policy trained with brax PPO on our `so101` scene (MJX) end up as a reproducible recipe and checkpoint | server: `python/es/rl_source/train_brax_so101.py --seed 0` twice → the same orbax parameter hash (mark `unverified` if it is); pins and numbers in `docs/api-notes/brax-ppo-so101.md` | D |
| 1 | **S4a `es-py` rollout bindings** | Can a Python trainer step our env through `es_native.Env`/`es_native.SafetyPlane` and pass through the plane | `python/es/selfcheck.py --env`: the fixture scene's 100-step `qpos` trajectory is bit-identical to the `es-env` test golden; the plane's `validate` emits the same `events` bits as Rust | B |
| 2 | **S2b `es policy import-rl`** | Do brax, rsl_rl, and rl_games actors become bundles through the adapter document and reproduce the source framework on 1,000 random observations — bit-identical for torch sources (rsl_rl, rl_games), §8.9 tier 4 (≤ 1e-5) with the max error recorded for the JAX source (brax) | `cargo test -p es --test cli import_rl_`: three synthetic checkpoints → document, bundle, `import.json`; server `--ignored`: the S2c checkpoint, 1,000 observations — the torch reconstruction vs our runtime bit-identical in f32, JAX vs our runtime max abs error ≤ 1e-5, hashes recorded; five kinds of adapter mismatch rejected as `IMP-0xx` | B |
| 2 | **S1 `[init] policy`** | Are tensors whose name and shape match copied, is the rest initialized, and does `init.lock` record it | `cargo test -p es --test cli train_init_`: zero steps from the U3 checkpoint → weights bit-identical, the `policy_hash` chain preserved; name mismatches listed in the lock | B |
| 3 | **S4b PPO trainer** | Does `es train --recipe`'s `[rl]` run rollout (§13.4), GAE, the clipped objective, and entropy through `train_ppo.py` and fill §19.3; is it bit-identical run twice on the CPU backend | `cargo test -p es --test cli train_rl_`: `--dry-run` plan golden; `tests/fixtures/rl/task-reach.toml` at seed 0 twice → checkpoint bit-identical (`ES_PYTHON`); target return reached (observation, server) | B |
| 3 | **S4d the reward cone reaches a body** | Do `GetBodyPose`, `Norm{L2}` and `Expr::Sqrt` enter `es-env`'s scalar cone so the reach reward `−‖cube − gripper‖` executes (the gap S4b found on 2026-09-21), with the committed `task.toml`'s lowering unmoved | `cargo test -p es-env body_norm_cone_lowers_and_the_demo_task_is_unmoved`; `--ignored`: `task-reach.toml`'s reward is bit-identical to the distance computed from the backend's own `xpos` | B |
| 3 | **S4e the observation capture reads joint velocities and a body pose** | Does a `JointState` channel name its `quantity` (absent = Position = today's hash) and does `es-eval`'s capture read `qvel` and a non-free-joint body's `xpos ‖ xquat`, so the 26-wide reach observation is captured by `Rollout` bit-identically to the backend state (the gap S4d found); and does the reach training (S4b's deferred oracle 4) then hit its target | `cargo test -p es-eval capture_reads_qvel_and_body_pose`; `--ignored`: `Rollout`'s 26-wide port bit-identical slice by slice to the backend state; server: `training-reach.toml` → `es eval run` `success_rate` (target ≥ 0.8, measured value recorded) | B |
| 3 | **S4c continuation measurement** | Starting from the imported policy (S2b) via `[init]` and continuing PPO in our sim, does the success rate rise on the same Evaluation IR | `es eval run` before/after → `visible-learning.md` 7.33 table (before continuation, after continuation, from scratch, three seeds); `es eval compare` | D |
| 4 | **M8 review** | Has the record come back into the specification | `docs/reviews/M8.md` + `.ko.md`; `cargo xtask ci` green | A |

**What is not on the ladder, and why.** **Real checkpoints of Isaac Lab policies (rsl_rl,
rl_games)** stay in this section only as far as the parser and synthetic fixtures — Isaac
scenes must go through `es-usd`, and, like MuJoCo, there is no source policy, so S2c opens a
second source only after pinning the recipe with brax. **The import wizard UI (S3)** was
deferred by the owner. **MJWarp rollout** is the throughput packet that comes after S4b
establishes bit-identity on CPU, and those numbers are tier 3. **RL-only Task IR nodes**
(observation noise, curriculum) run into rule 1.

**Owner decisions.** (1) Both episode-boundary questions are "yes" — this section records
that decision (2026-09-21). (2) S2c's source framework is brax PPO (MuJoCo Playground 0.2.0
pinned) — Isaac is second. (3) S4c's continuation budget (steps, wall-clock) is set after
S4b's measurement.

**M8 result (2026-09-22, `docs/reviews/M8.md`).** Every row of the ladder closed, with **S4d** (the reward cone: `Expr::Sqrt`,
`Source::Xpos` lanes, `Norm{L2}`) and **S4e** (the observation capture: `JointState.quantity`, `Capture::Qvel/BodyPose`) added to
wave 3 for the gap S4b found. The four rules held and `es-safety` changed by one line (the window). Measured (oracle server): R1's
`(cell, episode)` partition is bit-identical at jobs 1·2·4 on the committed documents and the 16-episode nominal suite drops from
125.7 s to 60.5 s (48 %); U3 re-measures byte-identical (the window only matters near `max_frac 0.9`); S2c trained a reach policy
with brax PPO on our scene (a derived MJX scene with 35 contact pairs excluded) to 1.00 — and **0.00 on the committed scene** (it
reaches through the table); S2b reproduces brax bitwise against its reconstruction, 9.7e-7 against JAX, rsl_rl and rl_games
bitwise; S1 re-packs U3 at 0 steps bitwise; S4b is bitwise across two runs on the CPU backend; S4e's reach training peaks at
held-out **0.5625** at 4,000 iterations (target 0.8 not reached, decaying after); **S4c**: imported 0.00, continued 0.00 (the
weights are bit-identical after 4,000 iterations — the tanh saturates on the committed observations and the gradient is exactly
zero), from scratch with the same graph 0.083, from scratch 64×64 **0.417** (three seeds). Passed as engineering; the campaign's
claim ("continue an imported policy") is not demonstrated, and the cause is the source policy (the scene's contact set, the
normalizer's saturation), not the runtime. The review's human decisions: the plane clamping every RL tick (S-3), the worker
thread pool changing CPU numerics (S-1), `scene_hash` depending on libm (S-2), and where the next source policy is trained.

### 28.12 M9 — Incremental Action Space (plan T)

After M8 closed (`docs/reviews/M8.ko.md`) the owner decided on 2026-09-22: as the answer to the plane/RL problem (S-3), **run the
incremental action space as a campaign**, but put two things ahead of it — the exploration-noise experiment (one variable at a
time, §28.9 rule 3) and the `es-ir` split (§1.5). Grounds: half of real-world policies (Isaac Lab's relative joint action / IK
delta, the HIL-SERL family's EE delta) are incremental spaces, and §8.5's `EeDelta` is name-only, so such policies cannot be
imported; in RL, noise lands on the increment and the envelope is kept automatically. M8's remaining human decisions (S-1 worker
pool, S-2 `scene_hash`, the next source policy's scene) stay open as they were.

**The rules this section pins down.** (1) Increment is an addition, not a replacement: the `JointPosition` policy and its
documents do not move a single byte (absent = default = today's hash). (2) Integration happens in one place, `es-env`, and
collection/evaluation/the trainer (`Rollout`) use the same function; the plane validates the absolute target and `es-safety` is
unchanged (INV-11–13). (3) One variable at a time: the noise experiment's (R1) table comes first, and T3 puts the
incremental-mode row next to its best absolute-mode row. (4) No new trait (INV-17).

| Wave | Packet | Question it answers | Oracle (one line) | Type |
|---|---|---|---|---|
| 0 | **R1 exploration noise** (`docs/packets/M8/P-M8-R1.md`) | Changing only `init_log_std`, the entropy coefficient, and the lr schedule, how far does the clamp rate fall from 1.00, and does the reach curve still hold after 4,000 iterations | server: 4 recipe variants × 4,000/10,000 iterations, seed 0 (best gets seed 1 added) → a table of `executed_ne_sampled_rate` and held-out `success_rate` (`rl-continuation.md` §7) | D |
| 0 | **R6 `es-ir` split** (`P-M8-R6.md`) | Moving `graph`/`hash` (modules that don't know the IR structs) down into `es-ir-types` and re-exporting them, does `es-ir` land under the target with callers unchanged | `cargo xtask context-budget`: `es-ir` < 6,000; every committed hash unchanged (all existing pin tests still green); `cargo xtask layering` | B |
| 1 | **T1 incremental semantics** | Does `JointDelta` appear in `es-ir-types` and the Deployment IR, does a single function in `es-env` integrate it, and does the plane receive the absolute target; are the `JointPosition` document's hash and bytes unchanged | `cargo test -p es-env delta_integrates_to_the_absolute_target`: the incremental column's integral is bitwise equal to the absolute column; the demo document's `.estraj` unchanged; `es-safety` diff 0 lines | B |
| 1 | **T2 incremental source policy and import** | Does a policy trained by switching S2c's brax env to incremental actions get imported through the adapter's `[action] kind = "joint_delta"` and reproduce over 1,000 observations | server: brax reconstruction bitwise, JAX ≤ 1e-5; derived-scene success rate recorded | B/D |
| 2 | **T3 incremental reach RL** | With the same budget and the same seeds, does PPO in the incremental space have a lower clamp rate and a higher held-out than the absolute space (R1's best) | server: 3 seeds, 4,000/10,000 iterations, one table (absolute best / incremental) — `success_rate`, `executed_ne_sampled_rate`, entropy | D |
| 3 | **T4 measurement and review** | Is there grounds to fix §13.4's default RL action space to incremental | `docs/reviews/M9.md` + `.ko.md`; `cargo xtask ci` | A |

**What is not on the ladder.** Real Isaac Lab relative-action policies become importable through T2's adapter, but the scene
(`es-usd`) and the contact-set question (M8 S-5) remain, for the next campaign. `EeDelta`'s IK is not in this section —
`JointDelta` comes first, and the EE space is separate, alongside §8.5's `EePose`.

---

## 29. Risks

| Risk | Severity | Response |
|---|---|---|
| **Learning IR cannot express real policies** | **High** | **M1 gates 5·6 verify this early. If LeRobot checkpoint load fails, fall back to referencing the entire `PolicyBundle` (§8.3)** |
| **Observation IR ↔ LeRobot preprocessing mismatch** | **High** | M1 gate 6. Continuously validated as a CI merge gate |
| **Safety Plane requirements do not fit real hardware** | **High** | Put the real hardware cell into CI at M3. Securing a robot partner is an essential task during M2 |
| Weak oracle approves an incorrect implementation | High | §1.4. Secure PyTorch · MuJoCo · MPFR references. Infrastructure first via gate 1 |
| **Vision bandwidth · VRAM limits scale** | Medium | §12.2 round_robin, §20 budget model, precompute then reject |
| **Policy inference latency dominates the control cycle** | Medium | §8.4 LRN-052 compile check, §8.6 asynchronous chunks, ACT-class priority |
| Pretrained backbone license | Medium | §25.2. `base_model.lock` + check at deployment. Decision 2026-09-15: torchvision ImageNet ResNet18 (BSD-3) allowed for the demo (§28.9 L13) |
| **3DGS alignment quality is insufficient** | Medium | §1.9 reduction #3. Replaceable with external tools + manual import |
| Backend semantic mismatch | Medium | §17.2 comparison tools, block execution on error |
| Integration consistency collapse across sessions | Medium | `AGENTS.md` layer, weekly audit |
| Type D bottleneck (2–3 people) | Medium | Reflected in §28.1. Lower the D proportion significantly by deferring self-built physics |
| Hallucinated APIs (ash/Slang/torch/LeRobot) | Medium | `docs/api-notes/` auto-generated, version-pinned |
| Overstated regulatory positioning | Medium | §27.1 explicit: an evidence-collection tool. Mapping claims prohibited before standards are finalized |
| Scope creep (five IRs) | Medium | §5.1 boundary table as the RFC baseline. Node additions require an RFC |
| CI fixed cost (including real hardware cell) | Medium | §26.2 layering, real hardware only on weekly · release |
| Rust staffing | Low | C ABI/WASM + Node SDK |
| Vendor driver deviation | Low | §3.2 · §3.4, per-vendor CI |

---
## Appendix A. Finalized and Deferred Decisions

### A.1 Finalized

**Architecture**
1. The product is a **Robot Learning Compiler & Runtime**. It is not a physics engine
2. **5 IRs**: Task / Observation / Learning / Deployment / Evaluation. Boundaries are in §5.1
3. **Task IR only declares `ObservationSpec`; it does not implement it**
4. **Neural networks do not go into Task IR.** Learning IR is separate
5. **The network internals are opaque, the interface semantics are typed.** The `PolicyHandle` metadata contract is the subject of type checking
6. **`ImageSpec` is the contract of an image port.** Includes resolution, color space, camera model, intrinsics, extrinsics, shutter, exposure, distortion
7. **`Resize` and `Crop` transform the intrinsics.** Included in the type system
8. **Time has 3 layers**: `History` (system) / `TemporalWindow` (learning input) / `TemporalEncoder` (neural network)
9. **Actions are modeled by chunk, horizon, and execution mode.** `policy(obs) → action` is abolished
10. **The Safety Plane is independent of the policy.** `es-safety` does not depend on `es-policy` (§4.2 rule 8)
11. **No Safety-Plane-disabled path exists**
12. **The 4 batch domains are separated**: simulation / observation / inference / training
13. **Env batch semantics are not enforced on the Learning IR**
14. **Physics is a backend.** MJWarp is the default, an in-house solver is an M4+ option
15. The `PhysicsBackend` / `PolicyRuntime` trait boundaries
16. **The Learning Loop is a first-class workflow** (§13)
17. **3DGS real-to-sim is one of the render paths** (§16)

**Reproducibility and Validation**
18. `execution_hash = H(task, observation, learning, policy, dataset, deployment, compiler, runtime, hardware_capability)`
19. **`dataset_hash` is composed of the three of content, schema, and split**
20. `base_model.lock` is a mandatory item of provenance
21. **5 determinism layers.** Layer 4 is policy equivalence
22. **Deterministic execution contract = all 8 items.** It is not guaranteed by float controls alone
23. **Vulkan float-controls are a capability query, not an enforcement mechanism.** They are applied via SPIR-V execution mode
24. **The GPU physics backend does not declare layer 1.** Only the CPU backend declares it
25. RFA/binned summation is adopted as the deterministic reduction
26. **Augmentation nodes are automatically disabled in evaluation**
27. **Fixing `evaluation_hash` is the premise of policy comparison**
28. **Safety Case traceability is a mandatory element of evidence artifacts.** `revalidation_trigger` is linked to the hash chain

**Engineering**
29. **Zero-copy is a capability, not a product promise.** `TensorTransport` negotiation
30. **Performance is viewed through 9 metrics.** The single `step/s` metric is abolished
31. Separation of `graph_hash` (authoring) / `*_hash` (semantics) / `compile_hash` (execution)
32. The editor layout is an `.eslayout` sidecar. It does not exist in the IR crate
33. CPU lowering is not a performance path but the correctness reference
34. Separation of release plan / debug plan
35. Built-in node type IDs are frozen from M1. Versioned independently from the plugin ABI
36. **Unvalidated tasks and policies do not run**
37. Crate source ≤ 10,000 lines (enforced by CI)
38. **Only 7 permitted extension points**: `PhysicsBackend`, `PolicyRuntime`, `TaskNodeFactory`, `LearningNodeFactory`, `InferenceBackend`, `Scalar`, `DeterministicAcc`

**Development Process**
39. AI agents perform the entire implementation. Humans do specification, oracles, design, and adjudication
40. **Oracle-first principle.** The validation harness comes before the implementation
41. The work packet is the minimal unit of planning (5 elements)
42. Type A auto-merge / B diff ≤ 800 lines + approval / C design document first / D human-led
43. Golden files are read-only in CI. Out-of-scope modifications are blocked by CI
44. Feature packets may not begin before the validation infrastructure is complete

**Adoption and Market**
45. **Adoption starts from "layering on top."** Telemetry → evaluation → observation → safety → learning → the whole (§27.2)
46. LeRobot compatibility is the top-priority interoperability target
47. It is not a certification body but an evidence collection and traceability tool

**Episodes and reinforcement learning (2026-09-21, §9.4·§10.5·§13.4)**
48. **An episode is the unit of work.** `EnvelopeViolationRate`'s window and `events.json`'s `tick` start at `begin_episode`
49. **RL is a trainer on top of the Learning IR.** The deployed graph has no value head, the rollout is `es-env`, and the Safety Plane stays on

### A.2 Deferred / Requiring Validation

| Item | Resolution | Decision criterion |
|---|---|---|
| **Does the Learning IR actually express ACT, DP, and SmolVLA** | **M1** | **Action match after loading a LeRobot checkpoint** |
| **Does the Observation IR match LeRobot preprocessing numerically** | **M1** | Match rate within tolerance |
| **Are the Safety Envelope items sufficient for real hardware** | **M3** | Robot partner feedback, HIL violation scenarios |
| Is treating a π₀-class large VLA as a `PolicyBundle` sufficient | M2 | Real-usage feedback |
| Measured vision bandwidth vs. the §12.4 estimate | M2 | Measurement error within ±20% |
| Accuracy of the memory budget model | M2 | ±10% vs. measurement |
| Default for the GPU scalar representation | M2 | Kernel time + trajectory error |
| **Domain gap reduction after 3DGS color/position alignment** | **M3** | Policy response difference before vs. after alignment |
| **Does `execution_hash` reproduction hold in real-hardware deployment** | **M3** | Comparison of deploying the same bundle twice |
| Distribution of unmapped items in cross-backend semantic mapping | M2 | §17.2 report |
| Can layer 1 be extended to "the same vendor family" | M4 | Cross-replay across 3 vendors |
| Actual success rate of `es generate` | M4 | Validation pass rate, deduplication rate |
| Alignment of the Safety Case schema with conformance standards | M4 | After the AI safety-feature standard draft is published |
| Do Type A packets exceed half | End of M0 | Measurement. If below, re-adjust the §1.8 budget |
| Is it worth building an in-house physics solver | M4 | Grounds for differentiation vs. MJWarp/Newton |

---

## Appendix B. Core Code Design

### B.1 IR Common Types

```rust
// es-ir/src/types.rs
pub struct PortType {
    pub elem: ElemType,          // F32 | F16 | Bf16 | F64 | I32 | U8 | Bool
    pub shape: Shape,
    pub unit: Unit,
    pub frame: Frame,
    pub time: TimeRef,
    pub image: Option<ImageSpec>,   // image ports only
}

pub enum Unit {
    Dimensionless, Length, Angle, Mass, Time,
    Velocity, AngularVelocity, Acceleration, Force, Torque,
    Pressure, Current, Voltage, Quaternion, RotationMatrix,
    Normalized { lo: f64, hi: f64 },
    Pixel, Luminance, Depth, Token,
    Composite(UnitPowers),       // m^a · kg^b · s^c · rad^d
}

pub enum Frame {
    World, LocalOrigin, Body(StableId), Sensor(StableId),
    Joint(StableId), Camera(StableId), Image(StableId), Policy,
}

pub enum TimeRef {
    Tick,
    Sensor { id: StableId, align: Align },          // Hold|Interpolate|Reject
    Window { base: Box<TimeRef>, n: u32, stride: u32 },
}
```

### B.2 `ImageSpec` Transformation Rules

```rust
impl ImageSpec {
    /// Resize scales the intrinsics. If rescale=false, warning OBS-034.
    pub fn resized(&self, w: u32, h: u32, rescale: bool) -> Self {
        let (sx, sy) = (w as f64 / self.width as f64, h as f64 / self.height as f64);
        let intr = if rescale { self.intrinsics.scaled(sx, sy) } else { self.intrinsics };
        Self { width: w, height: h, intrinsics: intr, ..*self }
    }

    /// Crop shifts the principal point.
    pub fn cropped(&self, rect: Rect, rescale: bool) -> Self { /* cx -= x0, cy -= y0 */ }

    /// Undistort sets distortion to None and sets new intrinsics.
    pub fn undistorted(&self, new_intr: Intrinsics) -> Self { /* ... */ }
}
```

**These three functions are the basis of the §7.2 OBS-034 check.** As the compiler propagates `ImageSpec` along the graph, it verifies that the intrinsics are consistent when it reaches a `CameraProjection` node.

### B.3 `LearningGraph` and `PolicyHandle`

```rust
// es-ir/src/learning.rs
pub struct LearningGraph {
    pub schema_version: u32,
    pub inputs: Vec<TensorPort>,
    pub nodes: Vec<LearningNode>,
    pub outputs: Vec<TensorPort>,
    pub policy: PolicyHandle,
}

pub struct PolicyHandle {
    pub architecture: ArchKind,          // Act|Diffusion|FlowMatching|Discrete|Bundle
    pub base_model: Option<BaseModelRef>,// uri + hash + license
    pub weights: WeightsRef,             // safetensors|onnx|spirv + hash
    pub contract: PolicyContract,        // §8.4
}

pub struct PolicyContract {
    pub inputs: BTreeMap<SmolStr, TensorPort>,
    pub observation_window: u32,
    pub action_dim: u32,
    pub horizon: u32,
    pub execute_chunk: u32,
    pub replanning_hz: f32,
    pub execution_mode: ActionExecutionMode,
    pub runtime: RuntimeHints,           // dtype, expected_latency_ms, deadline_ms
}
```

### B.4 Safety Plane

```rust
// es-safety/src/lib.rs   — no_std capable, 0 heap allocation
pub struct SafetyPlane<const NJ: usize, const H: usize> {
    envelope: SafetyEnvelope<NJ>,
    watchdogs: WatchdogSet,
    fallback: FallbackPolicy,
    state: SafetyState<NJ, H>,          // pre-allocated
    counters: SafetyCounters,           // violation/fallback statistics (§10.3)
}

impl<const NJ: usize, const H: usize> SafetyPlane<NJ, H> {
    /// Validates and corrects a chunk. Cannot fail. Always returns a safe action.
    pub fn validate(&mut self, chunk: &ActionChunk<NJ, H>, obs_age: Duration,
                    now: PhysTick) -> SafeAction<NJ> { /* ... */ }
}

// Does not depend on es-policy (§4.2 rule 8).
// validate does not return a Result. A safe-side action always exists.
```

**That `validate` does not return a `Result` is by design.** If the safety layer propagated failures upward, the caller could forget to handle them. It always returns an executable action and records the incident in the counters.

### B.5 Batch Domain Scheduler

```rust
// es-env/src/batch.rs
pub struct BatchPlan {
    pub sim: SimDomain,            // N_sim, backend, dt_phys
    pub obs: ObsDomain,            // N_obs, selection, views, sensor_dt
    pub inf: InfDomain,            // N_inf, runtime, async, deadline
    pub train: Option<TrainDomain>,
}

pub enum EnvSelection {
    All,
    Subset(Vec<EnvId>),
    RoundRobin { k: u32, period: u32 },   // deterministic: (tick / sensor_period) % ceil(N/k)
}

/// The application time of an async inference result is decided not by "arrival time"
/// but by "which tick's observation it was computed from" (§12.3).
pub struct ChunkArrival {
    pub computed_from: PhysTick,
    pub apply_at: PhysTick,        // computed_from + deterministic delay
    pub chunk: ActionChunk,
}
```

### B.6 Hash Chain

```rust
// es-ir/src/hash.rs
pub struct HashChain {
    pub asset: Vec<[u8; 32]>,
    pub scene: [u8; 32],
    pub task_graph: [u8; 32],      // authoring identity
    pub task: [u8; 32],            // semantic identity
    pub observation: [u8; 32],
    pub learning: [u8; 32],
    pub policy: [u8; 32],          // weights + arch + base_model
    pub dataset: DatasetHash,      // content + schema + split
    pub deployment: [u8; 32],
    pub evaluation: Option<[u8; 32]>,
    pub compiler: [u8; 32],
    pub runtime: [u8; 32],         // includes PolicyRuntime + TensorTransport choice
    pub hardware: HardwareCapability,
}

impl HashChain {
    pub fn execution_hash(&self) -> [u8; 32] { /* BLAKE3 over canonical encoding */ }
    /// what, if changed, must be revalidated (§27.1 revalidation_trigger)
    pub fn diff(&self, other: &Self) -> Vec<ChangedComponent> { /* ... */ }
}
```

### B.7 Normalization Invariants (property test)

```rust
proptest! {
    #[test] fn hash_independent_of_node_ids(g in arbitrary_ir()) {
        prop_assert_eq!(ir_hash(&canon(g.clone())), ir_hash(&canon(shuffle_ids(g))));
    }
    #[test] fn hash_independent_of_ui(g in arbitrary_ir(), l in arbitrary_layout()) {
        prop_assert_eq!(ir_hash(&parse_esgraph(&write_esgraph(&g, &l))?), ir_hash(&g));
    }
    #[test] fn roundtrip_preserves_hash(g in arbitrary_ir()) {
        prop_assert_eq!(ir_hash(&parse_toml(&write_toml(&g))?), ir_hash(&g));
    }
    #[test] fn param_change_changes_hash(g in arbitrary_ir(), p in arbitrary_edit()) {
        let g2 = apply(g.clone(), p); prop_assume!(g2 != g);
        prop_assert_ne!(ir_hash(&canon(g)), ir_hash(&canon(g2)));
    }
    #[test] fn unit_algebra_laws(a in arbitrary_unit(), b in arbitrary_unit()) {
        prop_assert_eq!(mul(a, b), mul(b, a));
        prop_assert_eq!(div(mul(a, b), b), a);
    }
}
```

**§28.7 gate 2 is these five.** If this wobbles, the entire §5.3 hash chain is meaningless.

### B.8 Deterministic Accumulator and Layering Check

```rust
pub trait DeterministicAcc<T>: Default + Clone {
    fn add(&mut self, v: T);
    fn merge(&mut self, other: &Self);   // associative and commutative
    fn finish(&self) -> T;
}
pub struct BinnedAcc<const K: usize = 3> { bins: [f64; K], index: Option<i32> }
```

```rust
// xtask: check the 9 §4.2 rules via cargo metadata
const LAYERS: &[(&str, u8)] = &[
    ("es-math",0), ("es-core",1),
    ("es-gpu",2), ("es-assets",2), ("es-usd",2),
    ("es-actuator",3), ("es-sensor",3), ("es-physics-core",3),
    ("es-physics-backend",4), ("es-physics-cpu",4), ("es-physics-gpu",4),
    ("es-render",5), ("es-splat",5),
    ("es-ir",6), ("es-compile",7),
    ("es-policy",8), ("es-safety",8),
    ("es-env",9),
    ("es-data",10), ("es-telemetry",10), ("es-eval",10),
    ("es-ros2",11), ("es-py",11), ("es-script",11), ("es-transport",11),
    ("es-editor",12),
];
// Additional checks:
//   does es-safety not depend on es-policy               (rule 8)
//   is es-ir unaware of es-compile / backends / torch     (rule 6)
//   is es-ir unaware of egui                              (rule 7)
//   absence of CUDA/HIP symbols outside es-transport      (rule 5)
```

---
## Appendix C. Agent Guidelines (Summary)

The full text is in the repository `AGENTS.md`. Only the core items are recorded here.

### C.1 Additional rules for the root `AGENTS.md`

```markdown
## Electric Sheep-specific rules

### IR boundaries (specification §5.1)
Do not put neural networks into the Task IR. The Learning IR is separate.
The Task IR only declares the ObservationSpec. Preprocessing is the responsibility of the Observation IR.
Do not put UI/layout types into the IR. They live in the .eslayout sidecar.

### Safety (specification §9, §4.2 rule 8)
es-safety does not depend on es-policy. Do not create a reverse dependency.
Do not create a code path that disables the Safety Plane.
  Do not create one even in tests. Tests do it by widening the envelope.
SafetyPlane::validate does not return a Result. Do not change the signature.

### Images (specification §7.2)
When implementing Resize / Crop, use ImageSpec::resized / cropped and
do not skip the intrinsic transformation. rescale_intrinsics=false is only when the user
has explicitly chosen it.

### Policy runtime (specification §2.4, §8.7)
Preprocessing and postprocessing are owned by the IR. Do not hand them to the PolicyRuntime.
Prefer safetensors for policy weights. Do not add pickle-based loading.

### Determinism (specification §3.4)
Vulkan float-controls is a capability query. Do not write code or comments
that treat it as a "setting". Application is done via SPIR-V execution mode.

### Performance (specification §12.4)
Do not report performance with the single step/s metric. Use the 9 metrics.
Do not write unvalidated performance figures as fact. Use the "Target / Status: unvalidated" format.
```

### C.2 Additional items for `docs/invariants.md`

| ID | Content | Check |
|---|---|---|
| INV-11 | `es-safety` ⇏ `es-policy` | xtask layering |
| INV-12 | Absence of a Safety Plane disable path | clippy custom + code review |
| INV-13 | `SafetyPlane::validate` signature fixed | compile time |
| INV-14 | intrinsic update on `ImageSpec` transformation | unit test + OBS-034 |
| INV-15 | augmentation nodes disabled during evaluation runs | Evaluation IR execution test |
| INV-16 | Absence of pickle-based weight loading | dependency check |
| INV-17 | Absence of single-implementation traits outside the 7 allowed extension points | weekly audit |

---

## Appendix D. References

**Policy/learning stack**
- LeRobot — https://github.com/huggingface/lerobot
- LeRobot: An Open-Source Library for End-to-End Robot Learning — arXiv:2602.22818 (list of supported policies, measured inference latency)
- SmolVLA — arXiv:2506.01844, https://huggingface.co/blog/smolvla
- ACT (Action Chunking with Transformers) — Zhao et al., 2023
- Diffusion Policy — Chi et al., arXiv:2303.04137
- π₀ — Black et al., arXiv:2410.24164
- π₀.₅ — Physical Intelligence, arXiv:2504.16054
- OpenVLA — Kim et al., arXiv:2406.09246
- OpenVLA-OFT (action chunking, parallel decoding) — arXiv:2502.19645
- Real-Time Execution of Action Chunking Flow Policies — arXiv:2506.07339
- GR00T N1 / N1.5 / N1.7 — NVIDIA

**Real-to-Sim / 3DGS**
- Real-to-Sim Robot Policy Evaluation with Gaussian Splatting — arXiv:2511.04665, https://real2sim-eval.github.io/
- RL-GSBridge — Wu et al., 2025
- Real-is-Sim (Embodied Gaussians) — Abou-Chakra et al., 2025
- RoboGSim — Li et al., 2025
- RialTo — Torne et al., 2024
- ReaDy-Go — arXiv:2602.11575
- PhysGaussian — CVPR 2024

**Simulation backends**
- Newton — https://github.com/newton-physics/newton
- Newton 1.0 GA — https://github.com/newton-physics/newton/discussions/2176
- MuJoCo Warp — https://github.com/google-deepmind/mujoco_warp
- Genesis World / Quadrants — https://github.com/Genesis-Embodied-AI/genesis-world
- Isaac Lab performance benchmark — https://isaac-sim.github.io/IsaacLab/
- Isaac Lab (tiled rendering, sensor throughput) — arXiv:2511.04831
- ManiSkill3 — arXiv:2410.00425
- RoboVerse / MetaSim — arXiv:2504.18904

**Determinism/numerics**
- NVIDIA CCCL floating-point determinism — https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/
- Demmel & Nguyen, Reproducible Floating Point Summation — ACM TOMS 2020, https://dl.acm.org/doi/10.1145/3389360
- ExBLAS / Kulisch accumulator
- Vulkan Float Controls — https://docs.vulkan.org/spec/latest/chapters/shaders.html
- Vulkan Limits — https://docs.vulkan.org/spec/latest/chapters/limits.html
- CUDA–Vulkan external memory/semaphore interop — NVIDIA CUDA Programming Guide

**Platform**
- rmw_zenoh — https://github.com/ros2/rmw_zenoh
- ROS 2 Lyrical Luth RMW tier — https://docs.ros.org/en/lyrical/Releases/Release-Lyrical-Luth.html
- openusd (Rust) — https://github.com/mxpv/openusd
- Slang — https://shader-slang.org/
- egui-snarl — https://github.com/zakarumych/egui-snarl

**Regulation**
- Regulation (EU) 2023/1230 (Machinery Regulation), applies 2027-01-20 — EUR-Lex
- Regulation (EU) 2024/1689 (AI Act)
- Machinery Regulation ↔ AI Act relationship (2026 amendment) — https://ai-resources.eu/en/atti-normativi/ue/macchine/
