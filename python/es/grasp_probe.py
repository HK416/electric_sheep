"""Does the scripted expert grasp the cube, or push it? (packet M5/V10 measurement 2)

Replays the demonstrations' *executed* actions on the demo scene with plain `mujoco` -- no
`es` runtime in the loop -- and logs, per control tick, the contact normal force between each
jaw and the cube, the cube's lift above the table, and the gripper joint's commanded versus
measured position. The verdict it prints is the fraction of demonstrations in which the cube
leaves the table at all.

The episodes come from `recorded_actions_replay_to_the_same_outcome`'s `ES_V10_DUMP`
directory, because only that side knows the Task IR's randomization draw: line 1 of each file
is the reset `qpos || qvel`, and every line after it is `action[0..6] || cube_xyz[0..3]`, the
executed row and the cube position the `es` replay ended that tick in. The cube column is the
self-check -- a probe that stepped the model differently would drift away from it immediately.

    ES_PYTHON=~/venvs/es/bin/python
    $ES_PYTHON -m es.grasp_probe --dump <dir> --scene tests/fixtures/mjcf/so101_pick_place.xml

Without `mujoco` it prints the reason and exits 0, like every other reference oracle here.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# The cube's `<geom name="cube_geom" size="0.0125 0.0125 0.02">`: half-height 20 mm, so a cube
# at rest has its centre at z = 0.02 and "leaves the table" once the lift passes that.
CUBE = "cube_geom"
CUBE_HALF_HEIGHT = 0.02
# `so101_pick_place.xml`'s bin interior, the same box `expert_solves_the_pinned_seeds` checks.
BIN = ((0.09, 0.19), (-0.15, -0.05), 0.09)
GRIPPER_JOINT = "gripper"
# The Learning IR's chunk horizon (`learning.toml`, `ActionChunker.horizon`).
HORIZON = 16


def jaw_geoms(model, mj):
    """The two jaws' collision geom ids, by the scene's own naming."""
    fixed, moving = [], []
    for g in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("fixed_jaw"):
            fixed.append(g)
        elif name.startswith("moving_jaw"):
            moving.append(g)
    return set(fixed), set(moving)


def replay(mj, model, data, rows, reset, substeps, fixed, moving, cube_geom, cube_adr, grip_adr):
    """One episode: step the recorded ctrl rows and return the per-tick log."""
    mj.mj_resetData(model, data)
    nq, nv = model.nq, model.nv
    data.qpos[:] = reset[:nq]
    data.qvel[:] = reset[nq : nq + nv]
    mj.mj_forward(model, data)
    force = np.zeros(6)
    log = []
    for row in rows:
        data.ctrl[:] = row[:6]
        for _ in range(substeps):
            mj.mj_step(model, data)
        f_fixed = f_moving = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            pair = {int(con.geom1), int(con.geom2)}
            if cube_geom not in pair:
                continue
            other = (pair - {cube_geom}).pop()
            mj.mj_contactForce(model, data, c, force)
            n = abs(float(force[0]))
            if other in fixed:
                f_fixed += n
            elif other in moving:
                f_moving += n
        log.append(
            (
                f_fixed,
                f_moving,
                float(data.qpos[cube_adr + 2]),
                float(row[5]),
                float(data.qpos[grip_adr]),
                np.array([data.qpos[cube_adr + i] for i in range(3)]),
                row[6:9],
                np.array(data.qpos[:6]),
            )
        )
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="ES_V10_DUMP directory of ep-NNN.txt files")
    ap.add_argument("--scene", required=True, help="the demo MJCF")
    ap.add_argument("--substeps", type=int, default=1, help="physics ticks per control tick")
    ap.add_argument("--contact", type=float, default=1e-6, help="normal force counted as contact")
    args = ap.parse_args()

    try:
        import mujoco as mj
    except ImportError as e:  # pragma: no cover - the skip path
        print(f"SKIP grasp_probe: {e}")
        return 0

    model = mj.MjModel.from_xml_path(args.scene)
    data = mj.MjData(model)
    fixed, moving = jaw_geoms(model, mj)
    cube_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, CUBE)
    cube_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_adr = model.jnt_qposadr[cube_joint]
    grip_adr = model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)]
    # `--substeps` is a measurement, not a setting: the cube column of the dump only tracks for
    # the value `Env::step` actually uses, and the drift below is what says so.
    period = args.substeps * model.opt.timestep
    print(
        f"jaws: {len(fixed)} fixed geoms, {len(moving)} moving; timestep {model.opt.timestep} s, "
        f"so one recorded action row is {period} s = {1.0 / period:.1f} Hz"
    )

    files = sorted(glob.glob(os.path.join(args.dump, "ep-*.txt")))
    if not files:
        print(f"SKIP grasp_probe: no ep-*.txt in {args.dump}")
        return 0

    header = (
        f"{'ep':>3} {'lift_mm':>8} {'left':>5} {'both_jaw':>9} {'window':>13} "
        f"{'grip_cmd':>9} {'grip_meas':>10} {'cube_final':>24} {'bin':>4} {'drift_mm':>9}"
    )
    print(header)
    print("-" * len(header))
    lifted, in_bin, lifts, windows, closures, drifts = 0, 0, [], [], [], []
    copy_all, copy_grasp, grasp_frac, same, ahead = [], [], [], [], []
    for path in files:
        # Line 1 is `nq + nv` wide and every line after it is 9, so the two are read apart.
        with open(path, encoding="utf-8") as fh:
            reset = np.array(fh.readline().split(), dtype=float)
        rows = np.loadtxt(path, skiprows=1, ndmin=2)
        log = replay(
            mj, model, data, rows, reset, args.substeps, fixed, moving, cube_geom, cube_adr,
            grip_adr,
        )
        z = np.array([e[2] for e in log])
        lift = float(z.max() - CUBE_HALF_HEIGHT)
        touching = [i for i, e in enumerate(log) if max(e[0], e[1]) > args.contact]
        both = [i for i, e in enumerate(log) if min(e[0], e[1]) > args.contact]
        window = (touching[0], touching[-1]) if touching else (-1, -1)
        if both:
            closures.append(float(np.median([log[i][4] for i in both])))
        grip_cmd = float(min(e[3] for e in log))
        grip_meas = float(min(e[4] for e in log))
        final = log[-1][5]
        inside = (
            BIN[0][0] < final[0] < BIN[0][1]
            and BIN[1][0] < final[1] < BIN[1][1]
            and final[2] < BIN[2]
        )
        # The last row's cube column is the *next* episode's reset draw -- `Env::step` resets on
        # the terminal tick and the dump reads the state after it -- so the self-check stops one
        # tick short.
        drift = float(max(np.linalg.norm(e[5] - e[6]) for e in log[:-1]))
        # How much of the learning objective the grasp actually is. `observation.state[t]` is
        # the row ACT is trained on and `action[t:t+H]` its chunk target, so "repeat the joints
        # you can already see" is a predictor that needs no policy at all. Raw radians -- not
        # the normalized L1 the training curve reports, and not comparable to it.
        arm = np.array([e[7] for e in log])
        act = rows[:, :6]
        per_tick = np.array(
            [np.abs(act[t : t + HORIZON] - arm[t]).mean() for t in range(len(act) - HORIZON)]
        )
        copy_all.append(float(np.mean(per_tick)))
        # Which state row the action belongs to. `Env::step` records `ctrl` beside the qpos the
        # step *ended* in, so `observation.state[t]` is the result of `action[t]` and not the
        # row it was computed from -- LeRobot's pairing is the other one. These two medians say
        # how far apart the two readings are, in radians of joint travel.
        same.append(float(np.median(np.abs(act - arm))))
        ahead.append(float(np.median(np.abs(act[1:] - arm[:-1]))))
        if both:
            held = per_tick[both[0] : min(both[-1] + 1, len(per_tick))]
            if held.size:
                copy_grasp.append(float(np.mean(held)))
                grasp_frac.append(len(held) / len(per_tick))
        left_table = lift > CUBE_HALF_HEIGHT
        lifted += int(left_table)
        in_bin += int(inside)
        lifts.append(lift)
        windows.append(len(both))
        drifts.append(drift)
        ep = int(os.path.basename(path)[3:6])
        print(
            f"{ep:>3} {lift * 1000:>8.2f} {str(left_table):>5} {len(both):>9} "
            f"{window[0]:>6}..{window[1]:<6} {grip_cmd:>9.4f} {grip_meas:>10.4f} "
            f"({final[0]:>6.3f},{final[1]:>7.3f},{final[2]:>6.3f}) {str(inside):>4} "
            f"{drift * 1000:>9.4f}"
        )

    n = len(files)
    closure = float(np.median(closures)) if closures else float("nan")
    print(
        f"\nRAN grasp_probe: {lifted}/{n} demonstrations lift the cube clear of the table "
        f"(> {CUBE_HALF_HEIGHT * 1000:.0f} mm), {in_bin}/{n} end with it in the bin.\n"
        f"  max lift {max(lifts) * 1000:.2f} mm, median {float(np.median(lifts)) * 1000:.2f} mm\n"
        f"  two-jaw contact ticks: median {float(np.median(windows)):.1f}, max {max(windows)}\n"
        f"  gripper joint closes to {closure:.4f} rad while both jaws touch the cube -- the "
        f"geometric closure the blended command has to reach\n"
        f"  worst cube drift from the `es` replay: {max(drifts) * 1000:.4f} mm\n"
        f"  the grasp holds for {float(np.median(grasp_frac)) * 100:.1f} % of an episode; a "
        f"chunk's L1 against 'repeat the\n  joints you can already see' is "
        f"{float(np.median(copy_all)):.4f} rad over the whole trajectory and "
        f"{float(np.median(copy_grasp)):.4f} rad\n  inside the grasp window (raw radians, not "
        f"the normalized training loss)\n"
        f"  |action[t] - qpos[t]| median {float(np.median(same)):.5f} rad, "
        f"|action[t] - qpos[t-1]| median {float(np.median(ahead)):.5f} rad"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
