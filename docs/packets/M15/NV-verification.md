# M15 N-V — plan N's experiments (multi-camera)

Plan N's E1 and E2 on this PC (Windows 11, RTX 3060 12 GB, 64 GB RAM), 2026-09-29/30, with the
editor open on each run. Documents: `tests/fixtures/visible-learning/*-views*`, `*-cam*`, `*-mad*`
(N1, N8b); scene `tests/fixtures/mjcf/so101_pick_place_views.xml` (overhead, wrist in SO-101's
`camera_mount`, side).

## 1. What the cameras see (N1)

At the reset pose the arm lies stretched out over the table: the committed **overhead** camera sees
0–29 cube pixels over five pinned cube positions, and 0 at the demonstrator's approach waypoint for
all five (the gripper covers the cube). The **wrist** camera sees 448–470 cube pixels at the approach
waypoint; the **side** camera 66–107 at reset. The overhead camera — the only camera of every run
before plan N — rarely sees the cube while the robot reaches for it.

## 2. The divergence, and its cause (P-M15-R2)

The first E1 trainings went NaN every time on the camera-only graphs, at U3's settings and at
gentler ones:

| graph | settings | NaN at step |
|---|---|---|
| one view | lr 4e-4 | 3,189 |
| three views | lr 4e-4 | 774 |
| three views | lr 4e-4, grad_clip 1.0 | 778 |
| three views | lr 1e-4, grad_clip 1.0 | 7,928 |

The investigation found it in the numerics, not the data or the learning rate: the lowered
`TemporalEncoder{Transformer}` runs a **one-token** sequence, so softmax is exactly 1 and the q/k
projections' gradient is exactly zero — but on this GPU torch's memory-efficient SDPA kernel returns
rounding noise for dq/dk in its backward (2.9e-5 at logit 1, 0.03 at 1e6, NaN at 1e9) while its
forward is exact. AdamW turns the noise into lr-sized steps, `Wq`/`Wk` drift (|Wq| 16 → 41), the
logits grow exponentially (0.27 → 3e9) and the loss goes NaN. The hint card's graph drifts the same
way; its two 60,000-step runs merely ended before the threshold (|Wq| 16 → 39 and → 25), Q5's went
NaN at ~1,130, and M7's U1 at 4,517 (`docs/design/visible-learning.md` open question 35, now
answered). The fix trains with exact (math) attention on CUDA (`python/es/train_act.py`,
`train_ppo.py`); dq/dk are then exactly zero and a CUDA test pins it. No hash moved; inference was
never affected (the forward is exact). E1 then ran at U3's own settings — lr 4e-4, no clipping —
without divergence.

## 3. E1 — one view against three, same data, same settings

One collection: 200 expert demonstrations (the 1 s-wait demonstration program), 63,901 frames,
rendered by all three cameras (200/200 successful; expert gate passed). Both arms train the
camera-only IR route (U3's graph without `sim_cube_pose`; `Concat` fusion; one ImageNet ResNet18 per
view) for 60,000 steps at lr 4e-4, batch 64, warmup-cosine; the one-view arm reads only
`rgb_overhead` of the same collection. Evaluated by the same suites and seeds.

| training seed 0 | one view (overhead) | three views |
|---|---|---|
| training speed | ~1,100 samples/s | ~440 samples/s (2 h 25 min for 60,000 steps) |
| nominal | 0/16 | 1/16 |
| all suites | 3/96 | 8/96 |
| attempts that lifted the cube | **5/96** | **30/96** |
| ended with the cube in the bin / held over it | 2 / 1 | 8 / 14 |
| `envelope_violation_rate`, nominal | 0.011 | 0.022 |

The success counts are within the run-to-run spread of one training each (M10 review S-4); the
lifting counts are not close: the three-view policy lifted the cube six times as often, which is
what section 1's pixel counts predict. Letting go remains the bottleneck (14 attempts held the cube
over the bin), as in M14's runs. Training seed 1 for both arms: section 4.

## 4. E1, training seed 1

The same two recipes with `seed = 1`, on the same collection (2026-09-30, overnight):

| | nominal | all suites | lifted | ended in the bin / held over it |
|---|---|---|---|---|
| one view, seed 0 | 0/16 | 3/96 | 5/96 | 2 / 1 |
| one view, seed 1 | 0/16 | 1/96 | 6/96 | 1 / 2 |
| three views, seed 0 | 1/16 | 8/96 | 30/96 | 8 / 14 |
| three views, seed 1 | **7/16** | **21/96** | 30/96 | 23 / 2 |

**E1's answer: a second and third view help, on both seeds.** The three-view policy lifts the cube
five to six times as often (30 against 5–6) and succeeds more (8 and 21 against 3 and 1). Seed 1's
nominal 7/16 (0.44) is the best camera-only number this project has measured, one success short of
the demo's 0.5 acceptance. The two seeds of the same arm differ by 8 against 21 successes: the
run-to-run spread the M10 review measured holds here too, so success counts need seeds; the lifting
counts were steady (30 and 30; 5 and 6).

## 5. E2 — MAD, deployed on fewer cameras

The MAD graph (`learning-mad.toml`: one ResNet18 shared by the three views, `Sum` fusion, then the
same downstream), `single_view = { weight = 0.5 }`, U3's settings, 60,000 steps (the whole run 2 h 47 min with
N7b's one-encoder-pass single-view step), seed 0, on E1's collection. Then `es policy subset` to each
single view (same weights), each judged by its derived Evaluation IR (N8c — the same suites, seeds,
metrics and acceptance):

| the MAD policy, evaluated with | nominal | all suites | lifted | ended in the bin / held over it |
|---|---|---|---|---|
| all three cameras | 4/16 | 12/96 | **41/96** | **29** / 10 |
| **the wrist camera alone** | **2/16** | 5/96 | 11/96 | 8 / 3 |
| the overhead camera alone | 0/16 | 1/96 | 4/96 | 1 / 1 |
| the side camera alone | 0/16 | 5/96 | 12/96 | 8 / 1 |

- With all cameras it lifted the cube and got it into the bin more often than any other run
  (41 and 29 of 96).
- **Deployed on one camera without retraining, it still works to a degree.** The wrist alone — the
  cheap real-robot setup — reached nominal 2/16, above both policies trained on the overhead
  camera alone (0/16, 0/16); the overhead alone matched those dedicated policies (1/96 against
  3/96 and 1/96). That is MAD's claim, seen here in imitation learning for the first time in this
  project; it is one training, so within the spread of section 4.
- Dropping to one camera is not free: 12 successes with three cameras, 5 with the best single one.

## 6. What stands after plan N's experiments

- More views help grasping decisively; the overhead camera alone is a poor sensor for this task.
- Letting go is still the thinnest part (M14's H-2): every run ends some attempts with the cube over
  or in the bin and the gripper shut.
- Success rates need several training seeds before a comparison is claimed.
