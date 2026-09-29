# Multi-camera policies — train with several views, deploy with fewer

Design note, 2026-09-29. Asked for by the owner ("S3 끝나면 멀티 카메라 설계 노트부터 써줘") after the
M13/M14 real runs. Spec: §6 (Task IR), §7.2–§7.4 (Observation IR, `MultiViewPack`, the two-view
example), §8.3–§8.4 (Learning IR, fusion), §14.2 (`shared=True`), §15.2 (one atlas, every camera a
tile), §19.1 (multi-camera datasets), §20.2 (atlas budget), §13.3 (the evaluation stays fixed),
§1.4 (oracle first), §5.3 (the hash chain). Nothing here is built yet; section 6 records the owner's
decisions.

## 1. Why

The hint card's policy fails at the two moments where a single overhead camera sees least: it
never lifts the cube (58–65 of 96 attempts) or it carries the cube to the bin and does not let go
(20–22 of 96) — `docs/packets/M12/YV-verification.md`, `docs/packets/M14/QV-verification.md`.
Grasping and releasing are close-range, depth-dependent acts; the standard remedy in imitation
learning (ALOHA, ACT, spec §28.9's own "a wrist camera" lever) is a camera on the wrist. The
camera-only card, which has never passed, needs it more.

The owner pointed at **MAD** — *Merging and Disentangling Views in Visual Reinforcement Learning
for Robotic Manipulation* (Almuzairee, Patil, Bhatt, Christensen; CoRL 2025; arXiv 2505.04619;
code `github.com/aalmuzairee/mad`, MIT). Its two ideas:

- **Merge:** every view goes through one shared encoder and the per-view features are **summed**
  (not concatenated) into the feature the actor and critic read.
- **Disentangle:** during training the single-view features are also fed, as augmented inputs
  weighted by `alpha`, so the policy works from any one view alone at test time.

It is visual RL (DrQ on SAC), on Meta-World and ManiSkill3 with three cameras (one on the arm, two
third-person), without a real robot, and reports about +30 % over multi-view baselines with all
cameras and robustness when cameras are removed. No application to imitation learning (ACT) was
found; here it would be our experiment, measured, not an assumption.

What it would buy this project: train with the wrist and the overhead camera, deploy with the
wrist camera alone (the cheap real-robot setup — SO-101's `camera_mount`), or keep working when one
camera fails.

## 2. What exists today

A survey of the tree (2026-09-29) found the **types** ready and the **paths** not.

| layer | can say several cameras | runs several cameras |
|---|---|---|
| scene | yes; a `<camera>` inside a body is body-mounted and the renderer follows the body each frame (`es-env/src/render.rs`) | SO-101 has a `camera_mount` body and no camera in it |
| Task IR | yes: `ObservationSpec.channels` is a map; each channel is a `Sensor` bound by an `ObservationSpec` graph node | `es-tools` refuses `--frames` with more than one image channel; no fixture has two |
| renderer | `Renderer::render(&[CameraView])` puts every camera in one atlas tile (§15.2) | `EnvRenderer` renders one camera per env; `EnvBatchRenderer` makes tiles *envs*, not views |
| Observation IR | yes: independent `ImageInput` chains, one output port each (§7.4's example); `MultiViewPack` and `Mask` exist | the CPU plan lowers independent chains; `MultiViewPack` and `Mask` are COMPILE-002; `FrameSource`, `Rollout` and `es dataset bake` assume one image |
| Learning IR | yes: one `VisionEncoder` per image port, `Fusion { Concat \| CrossAttention \| FiLm \| AdaLn \| TokenConcat }` | no `Sum`; no weight sharing — `python/es/builder.py` drops §14.2's `shared=True` ("a compiler-pass concern") and no pass implements it |
| datasets | the LeRobot v3 export writes one `observation.images.<name>` per camera | `mirror_frames` would hard-link the one flat tile directory into every camera's directory — two cameras would get the same frames (a latent bug) |
| LeRobot route | LeRobot's ACT itself takes several cameras through one shared backbone | loading a checkpoint back refuses more than one camera (`es-policy/src/lerobot.rs`) |
| a missing camera | — | nothing: every input port must be produced (XIR-010, LRN-011); a missing image is an error in evaluation, bake and training; `SensorDropout` is a Safety Plane watchdog, not a policy mask |

A second camera moves `scene_hash`, `task_hash`, `observation_hash`, then `learning_hash`,
`policy_hash`, the dataset schema, `evaluation_hash` and `execution_hash`. That is the chain doing
its job: new documents, not edited ones.

## 3. Design

### 3.1 New documents, old ones untouched

A new scene `tests/fixtures/mjcf/so101_pick_place_views.xml` — today's scene plus a camera in
`camera_mount` and one side camera (three views, section 6) — and a new Task
IR, Observation IR, Learning IR and Evaluation IR beside it. Every committed document and number
stays as it is; the new ones get their own hashes.

### 3.2 Several cameras through every path (no spec decision)

- `EnvRenderer` renders every image channel of the Task IR per step, one atlas tile per camera
  (§15.2 already describes it); `EnvBatchRenderer` tiles envs × views. Bitwise oracle: each tile
  equals that camera's single render, as X3b pinned for envs.
- `--frames` writes `<frames>/<channel>/<NNNNNN>.bin` (the layout the v3 export already expects);
  `es dataset bake` reads each image port's own directory; `mirror_frames` goes away.
- `FrameSource`, `Rollout` and the evaluation runner take one frame per channel.
- The budget (§20.2) counts `views` from the Task IR instead of the first `ImageInput`.

### 3.3 MAD in the IR (spec decisions — section 6)

- **Sum fusion.** A `FusionKind::Sum` (all inputs of equal width; the output is their sum). It
  is what lets a view be removed without re-training the fusion: the remaining terms keep their
  meaning. Lowered to `torch.stack(...).sum(0)`; its PyTorch oracle is a hand-written module.
- **One encoder, several views** (decided: section 6). Each view keeps its own `VisionEncoder`
  node, and `share = "<node>"` names the encoder whose weights it uses; today's arity rule stands.
- **Single-view training** is the training recipe's, not the graph's: `[run] single_view =
  { weight = alpha }` makes `train_act.py` add, for each batch, the loss of each view's features
  alone through the same fusion. No IR field; the Learning IR's batch semantics are free (§8).

### 3.4 Deploying with fewer cameras: a derived bundle

With `Sum` fusion and a shared encoder, dropping a view is a graph edit that keeps every weight:
remove the view's `ImageInput` chain, its encoder application and its fusion term. So instead of
optional ports (a runtime semantic nothing has today), **`es policy subset --views <names>`**
writes a new bundle: the same weights, a Learning IR and Observation IR without the dropped ports,
new hashes (it *is* a different deployable). XIR-010 and LRN-011 keep holding. Oracle: the subset
bundle's output equals the full graph evaluated with the dropped views' terms removed, computed in
PyTorch.

### 3.5 Evaluating it (§13.3)

The same Evaluation IR suites, seeds, metrics and acceptance, on each bundle: all views, and each
single view. The subset bundle's Task IR lists fewer cameras, so its Evaluation IR references a
different task and observation — the document differs in exactly those references, and the report
says so; the conditions a person compares (suites, seeds, thresholds) are identical.

## 4. Experiments, in order

1. **Does a second view help at all?** The camera-only IR route (U3's graph without
   `sim_cube_pose`) on the overhead camera, then on overhead + wrist, `Concat` fusion, the same
   200 demonstrations re-rendered. Two training seeds each (±0.3 run to run, M10 review S-4).
   Needs only section 3.2.
2. **MAD's claim.** Overhead + wrist with a shared encoder, `Sum` fusion and single-view training;
   evaluated with both cameras, with the wrist alone and with the overhead alone, against the
   single-camera policies of step 1. Needs 3.3 and 3.4.
3. **The editor.** A template option "cameras: overhead / overhead + wrist", and ⑤ showing the
   views side by side in the player.

Cost on this PC: rendering and data grow with the number of cameras (the camera card's 2.9 GB
becomes about 6 GB for two); training memory grows with the encoder applications, not the weights.

## 5. What this design keeps

The five IRs and their validators; the Safety Plane untouched (a missing camera at runtime stays a
`SensorDropout` watchdog → fallback, never a silently degraded policy); the hash chain honest
(a subset bundle is a new bundle); goldens only through generators; every committed document as it
is.

## 6. Decided by the owner (2026-09-29)

1. **`FusionKind::Sum` is added** to the Learning IR (§8.3's list gains it; the spec text is amended
   with the packet that implements it, in `docs/ARCHITECTURE.ko.md` first).
2. **Weight sharing is a field on each encoder:** every view keeps its own `VisionEncoder` node
   (today's one-input arity rule stands), and `share = "<node>"` names the encoder whose weights it
   uses. The lowering emits one module and applies it to each view; a `share` that names a node of
   another backbone, width or pretrained source is refused by name.
3. **Fewer cameras at deployment is a derived bundle** (`es policy subset --views <names>`), not
   optional inputs.
4. **Three cameras, as in MAD:** wrist (in SO-101's `camera_mount`), the overhead camera of today,
   and one side camera. Experiment 1 compares one view (overhead) with three; experiment 2 deploys
   the three-view MAD policy with each view alone and with the wrist alone as the real-robot case.

Plan N (`docs/packets/M15/plan-n.md`) turns sections 3.1–3.5 into packets in this order: the
several-camera paths (3.2) and the new documents (3.1), experiment 1, then `Sum`, `share`,
single-view training and `es policy subset`, experiment 2, then the editor.
