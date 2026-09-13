"""Reproduces the spec 14.2 Python builder snippet end to end: Task, Observation, Learning and
Deployment IR for a pick-and-place task, saved as the four `.toml` files spec 14.3 describes.

Run with the venv this repo's `maturin develop --release --features python` installed into:

    python python/es/examples/pick_place.py [output_dir]

A few spots adapt the spec's illustrative snippet to what the M0 schema actually has a field
for (each one is commented where it happens, e.g. `Task.action_spec`'s space-name table,
`Reward`'s weight carrying the sign spec 14.2 put on the value expression instead). None of
that touches `es-py`'s generic core — every adaptation is ordinary Python composing `add`/
`connect` calls, exactly as a hand-authored task would.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from es import Deployment, Learning, Observation, Task, math as m  # noqa: E402


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Task IR ------------------------------------------------------------------------
    task = Task("pick_cube", scene="scenes/table_franka.usda", rng_streams=("init",))
    arm = task.robot("franka")

    task.observe_spec(
        rgb_front=task.sensor("cam_front").rgb(),
        rgb_wrist=task.sensor("cam_wrist").rgb(),
        joint_state=arm.joint_positions(n_joints=7),
    )
    task.action_spec(space="joint_position", robot=arm, limits="from_scene")

    # spec 14.2: `reward("reach", weight=1.0, value=-m.norm(...))`. Two adaptations from the
    # spec snippet to what `TaskIr::validate` actually requires (spec 5.4
    # `check_policy_input`): the sign moves onto `weight` (`Arith`'s `Sub`/negate path needs a
    # second operand node this builder does not build a constant-folding path for), and the raw
    # `Length` norm is `Normalize`d to `Dimensionless` before it can be a reward term at all.
    reach_dist = m.norm(task.body("cube").pose().pos - arm.link("hand").pose().pos)
    reach_reward = reach_dist.normalize(lo=0.0, hi=1.5)
    task.reward("reach", weight=-1.0, value=reach_reward)

    grasp_force = task.contact(arm.link("finger"), task.body("cube")).force.sum()
    task.reward("grasp", weight=5.0, value=grasp_force > 1.0)

    close_enough = m.norm(task.body("cube").pose().pos - task.site("target")) < 0.02
    task.terminate("success", when=close_enough)

    task.randomize(
        "cube_pose",
        stream="init",
        target=task.body("cube").initial_pose,
        dist=("uniform", [-0.1, 0.1], [-0.1, 0.1]),
    )

    task.save(str(out_dir / "task.toml"))

    # --- Observation IR -------------------------------------------------------------------
    obs = Observation("two_view_224")
    obs.image("cam_front").resize(224, 224).to_linear().normalize("imagenet")
    obs.image("cam_wrist").resize(224, 224).to_linear().normalize("imagenet")
    obs.state("joint_state").normalize("joint_limits")
    obs.temporal_window(n_steps=1)
    obs.time_align("hold")
    obs.save(str(out_dir / "observation.toml"))

    # --- Learning IR ------------------------------------------------------------------------
    # `state_encoder`/`fusion`/`head`/`chunker`'s `kind`/`mode` strings are the bare
    # `LearningNode` enum tags (`learning.rs` has no `rename_all` on these, unlike `backbone` and
    # `space` above, which is why only those two get a translation table).
    lrn = Learning("act_r18")
    vision = lrn.vision_encoder("resnet18", pretrained="imagenet", frozen=False, shared=True)
    state = lrn.state_encoder("Identity")
    fusion = lrn.fusion("Concat")
    lrn.connect(vision, "out", fusion, "in0")
    lrn.connect(state, "out", fusion, "in1")
    head = lrn.head("Regression", horizon=50)
    lrn.connect(fusion, "out", head, "in0")
    # `PolicyHead`'s output port is `"chunk"`, not `"out"` (`learning.rs`'s `chunk()` helper).
    chunker = lrn.chunker(execute_chunk=20, replanning_hz=10, mode="TemporalEnsemble")
    lrn.connect(head, "chunk", chunker, "in0")
    lrn.set_policy(action_dim=7, horizon=50, execute_chunk=20, replanning_hz=10.0)
    lrn.save(str(out_dir / "learning.toml"))

    # --- Deployment IR ------------------------------------------------------------------------
    dep = Deployment("franka_safe", robot=arm, action=task.action_spec_ref())
    dep.envelope(
        torque_limit="from_urdf",
        velocity_limit=1.5,
        workspace=Deployment.box([0.2, -0.4, 0.0], [0.8, 0.4, 0.6]),
        rate_limit=2.0,
    )
    dep.watchdog("inference_deadline", budget_ms=50)
    dep.watchdog("chunk_underrun")
    dep.watchdog("stale_observation", max_age_ms=100)
    dep.fallback("hold_position")
    dep.save(str(out_dir / "deployment.toml"))

    print(f"wrote task.toml, observation.toml, learning.toml, deployment.toml to {out_dir}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tasks/pick_cube")
    build(target)
