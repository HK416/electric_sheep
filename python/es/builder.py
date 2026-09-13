"""Python authoring builder (spec 14.2).

Thin wrapper over the ``es_native`` extension module (``crates/es-py``, pyo3), which is itself
a thin wrapper over the language-neutral Rust builder core (``crates/es-py/src/builder.rs``).
Nothing in this file constructs an IR node "by hand" in the sense the spec forbids — every
method here ends in one call to ``es_native``'s generic ``add``/``connect``/``declare_*``/
``build``/``save``. What lives here instead is the same kind of small, local type inference a
graph editor's UI needs to do anyway: knowing, for example, that ``GetBodyPose``'s ``pos`` output
is a 3-vector in ``Length`` before a shape has been run once. Each such formula is commented with
the ``es-ir`` function it mirrors, so a change there is easy to find and match here.

Params cross into the extension module as JSON text (see ``crates/es-py/src/pybind.rs`` for why:
no ``PyObject`` walker, no extra dependency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import blake3

from . import es_native as _native

EsDiagnosticError = _native.EsDiagnosticError


def stable_id(path: str) -> str:
    """Mirrors ``es_core::StableId::from_path`` bit-for-bit: ``blake3(path)`` truncated to 16
    bytes, hex-encoded. Two authoring tools (this one, the MuJoCo/Newton backends, the editor)
    naming the same body/sensor/joint by the same path string get the same id without agreeing
    on anything else.
    """
    return blake3.blake3(path.encode("utf-8")).digest(length=16).hex()


def _port_type(
    elem: str = "F32",
    shape: tuple[int, ...] = (1,),
    unit: Any = "Dimensionless",
    frame: Any = "World",
    time: Any = "Tick",
    image: Any = None,
) -> dict:
    return {
        "elem": elem,
        "shape": list(shape),
        "unit": unit,
        "frame": frame,
        "time": time,
        "image": image,
    }


FLAG = _port_type(elem="Bool", unit="Dimensionless")


@dataclass
class Expr:
    """A Task IR value: a ``(node, port)`` pair plus the ``PortType`` es-ir's
    ``TaskNode::outputs`` would report for it. Operators below build more Task IR nodes through
    ``Task.add``/``Task.connect`` — the generic escape hatch, never a special case.
    """

    task: "Task"
    node: int
    port: str
    ty: dict

    def __sub__(self, other: "Expr") -> "Expr":
        return self.task._arith("Sub", self, other)

    def __add__(self, other: "Expr") -> "Expr":
        return self.task._arith("Add", self, other)

    def norm(self, kind: str = "L2") -> "Expr":
        """``m.norm(x)`` (spec 14.2): ``es-ir`` `Norm` node, output shape `[1]` (`task.rs`
        `TaskNode::Norm` -> `with_dims(ty, vec![1])`)."""
        node = self.task.add("Norm", {"kind": kind, "ty": self.ty})
        self.task.connect(self.node, self.port, node, "value")
        return Expr(self.task, node, "value", {**self.ty, "shape": [1]})

    def sum(self) -> "Expr":
        """``.sum()`` (spec 14.2 `force.sum()`): `Reduce(Sum)` over axis 0, same unit, shape
        `[1]` (`task.rs` `Reduce` output keeps the input `ty`'s unit)."""
        node = self.task.add(
            "Reduce", {"op": "Sum", "axis": 0, "unordered": False, "ty": self.ty}
        )
        self.task.connect(self.node, self.port, node, "value")
        return Expr(self.task, node, "value", {**self.ty, "shape": [1]})

    def normalize(self, lo: float, hi: float, out_lo: float = 0.0, out_hi: float = 1.0) -> "Expr":
        """Maps `[lo, hi] -> [out_lo, out_hi]` (`task.rs` `Normalize`). A reward term must be
        `Dimensionless`/`Normalized`/`Token` (`TaskIr::validate`'s `check_policy_input`, spec
        5.4) before it can be handed to `Task.reward`, so any reward built from a raw physical
        quantity (a `Length` norm, say) needs one of these first.
        """
        n = self.ty["shape"][0]
        node = self.task.add(
            "Normalize",
            {"lo": [lo] * n, "hi": [hi] * n, "out_lo": out_lo, "out_hi": out_hi, "ty": self.ty},
        )
        self.task.connect(self.node, self.port, node, "value")
        out_ty = {**self.ty, "unit": {"Normalized": {"lo": out_lo, "hi": out_hi}}}
        return Expr(self.task, node, "value", out_ty)

    def __lt__(self, rhs: float) -> "Expr":
        return self._compare("Lt", rhs)

    def __gt__(self, rhs: float) -> "Expr":
        return self._compare("Gt", rhs)

    def _compare(self, op: str, rhs: float) -> "Expr":
        # `Compare`'s `rhs: Option<f64>` (`task.rs`) compares against a literal directly, no
        # second node needed — exactly the shape spec 14.2's `< 0.02` / `> 1.0` want.
        node = self.task.add("Compare", {"op": op, "rhs": rhs, "ty": self.ty})
        self.task.connect(self.node, self.port, node, "a")
        return Expr(self.task, node, "value", FLAG)


class Sensor:
    def __init__(self, task: "Task", name: str) -> None:
        self.task = task
        self.id = stable_id(name)

    def rgb(self, width: int = 224, height: int = 224) -> Expr:
        ty = _port_type(
            elem="U8", shape=(height, width, 3), unit="Pixel", frame={"Sensor": self.id}
        )
        node = self.task.add("GetSensor", {"sensor": self.id, "ty": ty})
        return Expr(self.task, node, "value", ty)


class Body:
    def __init__(self, task: "Task", name: str) -> None:
        self.task = task
        self.name = name
        self.id = stable_id(name)

    def pose(self, relative_to: Any = "World") -> "Pose":
        node = self.task.add("GetBodyPose", {"body": self.id, "relative_to": relative_to})
        pos_ty = _port_type(shape=(3,), unit="Length", frame=relative_to)
        quat_ty = _port_type(shape=(4,), unit="Quaternion", frame=relative_to)
        return Pose(
            pos=Expr(self.task, node, "pos", pos_ty),
            quat=Expr(self.task, node, "quat", quat_ty),
        )

    @property
    def initial_pose(self) -> str:
        # `Randomization`/`ResetState` (`task.rs`) name their target with a plain string; the
        # runtime resolves it against the scene, not against a Task IR node.
        return f"{self.name}.initial_pose"


@dataclass
class Pose:
    pos: Expr
    quat: Expr


class Link:
    """`arm.link("hand")` (spec 14.2): a body scoped under a robot, `StableId::from_path`d as
    `"{robot}/{link}"` so two robots' same-named link never collide."""

    def __init__(self, task: "Task", robot_name: str, link_name: str) -> None:
        self._body = Body(task, f"{robot_name}/{link_name}")

    def pose(self, relative_to: Any = "World") -> Pose:
        return self._body.pose(relative_to)

    @property
    def id(self) -> str:
        return self._body.id


class Robot:
    def __init__(self, task: "Task", name: str) -> None:
        self.task = task
        self.name = name
        self.id = stable_id(name)

    def link(self, name: str) -> Link:
        return Link(self.task, self.name, name)

    def joint_positions(self, n_joints: int = 7) -> Expr:
        joints = [f"joint_{i}" for i in range(n_joints)]
        node = self.task.add(
            "GetJointState", {"body": self.id, "joints": joints, "quantity": "Position"}
        )
        ty = _port_type(shape=(n_joints,), unit="Angle", frame={"Joint": self.id})
        return Expr(self.task, node, "value", ty)


class Contact:
    def __init__(self, task: "Task", node: int) -> None:
        self.task = task
        self.node = node

    @property
    def force(self) -> Expr:
        return Expr(self.task, self.node, "force", _port_type(shape=(3,), unit="Force"))

    @property
    def touching(self) -> Expr:
        return Expr(self.task, self.node, "touching", FLAG)


class Task:
    """`es.Task(name, scene=...)` (spec 14.2)."""

    def __init__(
        self,
        name: str,
        scene: str,
        *,
        scene_hash: bytes = bytes(32),
        asset_hash: bytes = bytes(32),
        max_episode_steps: int = 1000,
        control_rate_hz: float = 30.0,
        deterministic: bool = True,
        rng_streams: tuple[str, ...] = ("init",),
    ) -> None:
        self.name = name
        scene_json = json.dumps(
            {"path": scene, "scene_hash": list(scene_hash), "asset_hash": list(asset_hash)}
        )
        config_json = json.dumps(
            {
                "max_episode_steps": max_episode_steps,
                "control_rate_hz": control_rate_hz,
                "deterministic": deterministic,
                "rng_streams": list(rng_streams),
            }
        )
        self._native = _native.Task(scene_json, config_json)
        self._action_spec: int | None = None

    # --- the generic escape hatch --------------------------------------------------------

    def add(self, kind: str, params: dict) -> int:
        return self._native.add(kind, json.dumps(params))

    def connect(self, from_node: int, from_port: str, to_node: int, to_port: str) -> None:
        self._native.connect(from_node, from_port, to_node, to_port)

    def _arith(self, op: str, a: Expr, b: Expr) -> Expr:
        """`Arith` (`task.rs`): ports are `a`/`b`, both `a.ty` for `Add`/`Sub` (the only ops
        `Expr`'s operator overloads use — `Mul`/`Div` give `b` a `Dimensionless` port instead,
        which this builder does not need)."""
        node = self.add("Arith", {"op": op, "ty": a.ty})
        self.connect(a.node, a.port, node, "a")
        self.connect(b.node, b.port, node, "b")
        return Expr(self, node, "value", a.ty)

    # --- spec 14.2 surface ----------------------------------------------------------------

    def robot(self, name: str) -> Robot:
        return Robot(self, name)

    def sensor(self, name: str) -> Sensor:
        return Sensor(self, name)

    def body(self, name: str) -> Body:
        return Body(self, name)

    def site(self, name: str) -> Expr:
        """A fixed scene site's position; sites have no orientation node in spec 6.3, so this
        returns a position-only Expr the way `task.site("target").pos` is used in the spec 14.2
        example."""
        node = self.add("GetBodyPose", {"body": stable_id(name), "relative_to": "World"})
        ty = _port_type(shape=(3,), unit="Length")
        return Expr(self, node, "pos", ty)

    def contact(self, a: Body | Link, b: Body | Link) -> Contact:
        node = self.add("GetContact", {"a": a.id, "b": b.id})
        return Contact(self, node)

    def observe_spec(self, **channels: Expr) -> None:
        """`task.observe_spec(rgb_front=..., joint_state=...)` (spec 14.2): declares each kwarg
        as a Task IR -> Observation IR channel (spec 7.4) *and* binds it with an
        `ObservationSpec` node, since `TaskIr::validate` requires one node per declared channel.
        """
        for name, expr in channels.items():
            # The channel's *provenance* (`ObsSource`) is separate from the *value* (`expr`):
            # for this builder, the sensor/joint id the `Expr` was built from is what matters,
            # so it is re-derived from `expr.ty` rather than tracked twice per accessor.
            channel = {"source": _infer_source(expr), "ty": expr.ty}
            self._native.observe(name, json.dumps(channel["source"]), json.dumps(channel["ty"]))
            node = self.add("ObservationSpec", {"channel": name, "ty": expr.ty})
            self.connect(expr.node, expr.port, node, "value")

    def action_spec(self, space: str, robot: Robot, limits: str = "from_scene", dim: int = 7) -> int:
        del robot, limits  # resolved by the compiler against the scene (spec 9.2), not authored here
        self._action_spec = self.add(
            "ActionSpec", {"space": _ACTION_SPACE[space], "dim": dim, "control_rate_hz": 30.0}
        )
        return self._action_spec

    def action_spec_ref(self) -> int:
        if self._action_spec is None:
            raise ValueError("call action_spec(...) before action_spec_ref()")
        return self._action_spec

    def reward(self, name: str, weight: float, value: Expr) -> int:
        return self._native.reward(name, weight, value.node, value.port, json.dumps(value.ty))

    def terminate(self, name: str, when: Expr) -> int:
        return self._native.terminate(name, when.node, when.port)

    def randomize(self, name: str, stream: str, target: str, dist: tuple) -> int:
        return self.add(
            "Randomization",
            {"target": target, "stream": stream, "dist": _distribution(dist)},
        )

    def save(self, path: str) -> None:
        self._native.save(path)


def _infer_source(expr: Expr) -> dict:
    """`ObsSource` (`task.rs`) names *where* a channel's data comes from; every accessor this
    builder has (`Sensor.rgb`, `Robot.joint_positions`) only ever feeds a channel from one
    `GetSensor`/`GetJointState` node, so the source is recoverable from the `Expr`'s own type
    rather than tracked twice. `Language` and `BodyPose` channels are out of scope for this
    packet's example.
    """
    if expr.ty["unit"] == "Pixel":
        sensor_id = expr.ty["frame"]["Sensor"]
        return {"Sensor": {"id": sensor_id, "format": "Rgb"}}
    if isinstance(expr.ty["frame"], dict) and "Joint" in expr.ty["frame"]:
        return {"JointState": {"body": expr.ty["frame"]["Joint"], "dof": expr.ty["shape"][0]}}
    raise ValueError(f"observe_spec: cannot infer an ObsSource for {expr.ty!r}")


_BACKBONE = {
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "dinov2": "DinoV2",
    "siglip": "SigLip",
    "smolvlm": "SmolVlm",
}

# `task.action_spec(space="joint_position", ...)` (spec 14.2) is snake_case; `TaskNode::ActionSpec`
# (`task.rs`) has no `#[serde(rename_all)]`, so its `ActionSpace` tag is the bare PascalCase
# variant name. This is the one translation table that gap needs.
_ACTION_SPACE = {
    "joint_position": "JointPosition",
    "joint_velocity": "JointVelocity",
    "joint_torque": "JointTorque",
    "ee_pose": "EePose",
    "ee_delta": "EeDelta",
    "gripper": "Gripper",
}


def _distribution(dist: tuple) -> dict:
    """`("uniform_se2", lo, hi, angle)` etc. (spec 14.2) -> `Distribution` (`task.rs`). Only
    `uniform` (the spec 14.2 example) is implemented; the others are a straightforward extension
    of the same table, added when a packet needs them.
    """
    kind, *args = dist
    if kind in ("uniform", "uniform_se2"):
        lo, hi = args[0], args[1]
        return {"Uniform": {"lo": float(lo[0]), "hi": float(hi[0])}}
    raise ValueError(f"unsupported distribution kind: {kind}")


class Observation:
    """`es.Observation(name, task=task)` (spec 14.2). `task_ref_hex` is `task_hash` — pass the
    hex digest a `Task.build()`/hash-chain step produced; the example uses a zero placeholder
    since M0 does not yet expose `task_hash` computation from Python.
    """

    def __init__(self, name: str, task_ref_hex: str = "0" * 64) -> None:
        self.name = name
        self._native = _native.Observation(task_ref_hex)

    def add(self, kind: str, params: dict) -> int:
        return self._native.add(kind, json.dumps(params))

    def connect(self, from_node: int, from_port: str, to_node: int, to_port: str) -> None:
        self._native.connect(from_node, from_port, to_node, to_port)

    def image(self, name: str) -> "ImageChain":
        return ImageChain(self, name)

    def state(self, name: str) -> "StateChain":
        return StateChain(self, name)

    def temporal_window(self, n_steps: int = 1, stride: int = 1, align: str = "Hold") -> None:
        self._native.set_temporal(
            json.dumps({"history": {}, "window": {"n_steps": n_steps, "stride": stride, "align": align}})
        )

    def time_align(self, mode: str) -> None:
        # spec 7.5: alignment lives on the `TemporalWindow`/history join, which this minimal
        # builder folds into `temporal_window`'s `align`; kept as a separate call to match the
        # spec 14.2 surface even though it is a no-op once `temporal_window` already ran.
        del mode

    def save(self, path: str) -> None:
        self._native.save(path)


class ImageChain:
    """`obs.image("rgb_front").resize(224, 224).to_linear().normalize("imagenet")` (spec 14.2)."""

    def __init__(self, obs: Observation, sensor_name: str) -> None:
        self.obs = obs
        sensor_id = stable_id(sensor_name)
        ty = _port_type(elem="U8", shape=(224, 224, 3), unit="Pixel", frame={"Sensor": sensor_id})
        self.node = obs.add("ImageInput", {"sensor": sensor_id, "io": {"inputs": [], "output": ty}})
        self.ty = ty

    def resize(self, width: int, height: int, filter: str = "Bilinear") -> "ImageChain":
        out_ty = {**self.ty, "shape": [height, width, 3]}
        node = self.obs.add(
            "Resize",
            {
                "width": width, "height": height, "filter": filter, "rescale_intrinsics": True,
                "io": {"inputs": [self.ty], "output": out_ty},
            },
        )
        self.obs.connect(self.node, "out", node, "in0")
        self.node, self.ty = node, out_ty
        return self

    def to_linear(self) -> "ImageChain":
        # `ColorTransform` in the full spec 7.3 node set is out of this packet's example scope;
        # this builder leaves the pixel encoding as-is and only tracks the shape/unit chain the
        # rest of the example needs.
        return self

    def normalize(self, _stats: str) -> "ImageChain":
        out_ty = {**self.ty, "elem": "F32", "unit": {"Normalized": {"lo": -1.0, "hi": 1.0}}}
        node = self.obs.add(
            "Normalize",
            {
                "stats": {"Range": {"lo": 0.0, "hi": 255.0}},
                "training_only": False,
                "io": {"inputs": [self.ty], "output": out_ty},
            },
        )
        self.obs.connect(self.node, "out", node, "in0")
        self.node, self.ty = node, out_ty
        return self


class StateChain:
    """`obs.state("joint_state").normalize("joint_limits")` (spec 14.2)."""

    def __init__(self, obs: Observation, source_name: str) -> None:
        self.obs = obs
        source_id = stable_id(source_name)
        ty = _port_type(shape=(7,), unit="Angle")
        self.node = obs.add("StateInput", {"source": source_id, "io": {"inputs": [], "output": ty}})
        self.ty = ty

    def normalize(self, _stats: str) -> "StateChain":
        out_ty = {**self.ty, "unit": {"Normalized": {"lo": -1.0, "hi": 1.0}}}
        node = self.obs.add(
            "Normalize",
            {
                "stats": {"Range": {"lo": -3.14, "hi": 3.14}},
                "training_only": False,
                "io": {"inputs": [self.ty], "output": out_ty},
            },
        )
        self.obs.connect(self.node, "out", node, "in0")
        self.node, self.ty = node, out_ty
        return self


def _feature_ty(dim: int, tokens: int = 0) -> dict:
    """Mirrors `learning.rs`'s private `feature()`: `[dim]` when untokenized, else
    `[tokens, dim]`, always `Dimensionless` in `Frame::Policy` — every encoder's `out` port."""
    shape = [dim] if tokens == 0 else [tokens, dim]
    return _port_type(shape=shape, unit="Dimensionless", frame="Policy")


def _chunk_ty(horizon: int, action_dim: int) -> dict:
    """Mirrors `learning.rs`'s private `chunk()`: `PolicyHead`'s `"chunk"` / `ActionChunker`'s
    `"actions"` port, `Normalized(-1, 1)` (`action_unit()`) since it precedes unnormalization."""
    return _port_type(
        shape=[horizon, action_dim], unit={"Normalized": {"lo": -1.0, "hi": 1.0}}, frame="Policy"
    )


class Learning:
    """`es.Learning(name, observation=obs)` (spec 14.2).

    A `LearningNode`'s own `inputs: Vec<TensorPort>` field *is* its input port list
    (`learning.rs`'s `IrNode::inputs` reads it back verbatim) — unlike Task/Observation IR,
    where ports are fixed by the node kind. So every method below both declares that list *and*
    wires the matching `connect`, computing each port's `PortType` the same way `learning.rs`'s
    private `feature()`/`chunk()` helpers do (mirrored in `_feature_ty`/`_chunk_ty` above) so an
    edge's two ends agree exactly, byte for byte.
    """

    def __init__(self, name: str, observation: Observation | None = None) -> None:
        del observation  # the compiler cross-checks Observation <-> Learning shapes (spec 11.1);
        # this builder does not replay that check client-side.
        self.name = name
        self._native = _native.Learning()
        self._last_head_horizon: int | None = None

    @staticmethod
    def act(vision_out_dim: int = 512, state_out_dim: int = 512, action_dim: int = 7, horizon: int = 50) -> "Learning":
        """`es.Learning.act(...)`: the ACT shape (spec 14.2 example) in one call, built by the
        `es_native.Learning.act` pyo3 convenience constructor (still just `add`/`connect`)."""
        self = Learning.__new__(Learning)
        self.name = "act"
        self._native = _native.Learning.act(vision_out_dim, state_out_dim, action_dim, horizon)
        self._last_head_horizon = horizon
        return self

    def add(self, kind: str, params: dict) -> int:
        return self._native.add(kind, json.dumps(params))

    def connect(self, from_node: int, from_port: str, to_node: int, to_port: str) -> None:
        self._native.connect(from_node, from_port, to_node, to_port)

    def vision_encoder(self, backbone: str, pretrained: str | bool = True, frozen: bool = False, shared: bool = True, out_dim: int = 512) -> int:
        del shared  # multi-camera weight sharing is a compiler-pass concern (spec 8.3), not a field this node carries
        # `VisionBackbone::pretrained` is a plain `bool` (`learning.rs`); spec 14.2's
        # `pretrained="imagenet"` names a *source*, which the current schema has no field for,
        # so any truthy value here just means "yes, start from pretrained weights". Its own
        # input (the camera tensor from Observation IR) is a graph boundary, not an edge from
        # another Learning node, so one placeholder port satisfies `LRN-002`'s minimum-of-1
        # without anything needing to match it.
        self._vision_out = out_dim
        return self.add(
            "VisionEncoder",
            {
                "inputs": [{"name": "in0", "ty": _port_type()}],
                "backbone": _BACKBONE[backbone], "pretrained": bool(pretrained), "frozen": frozen,
                "out_dim": out_dim, "token_count": 0,
            },
        )

    def state_encoder(self, kind: str, out_dim: int = 512) -> int:
        self._state_out = out_dim
        return self.add(
            "StateEncoder",
            {"inputs": [{"name": "in0", "ty": _port_type()}], "kind": kind, "out_dim": out_dim},
        )

    def fusion(self, kind: str, out_dim: int | None = None) -> int:
        out_dim = out_dim or (self._vision_out + self._state_out)
        node = self.add(
            "Fusion",
            {
                "inputs": [
                    {"name": "in0", "ty": _feature_ty(self._vision_out)},
                    {"name": "in1", "ty": _feature_ty(self._state_out)},
                ],
                "kind": kind, "out_dim": out_dim, "token_count": 0,
            },
        )
        self._fusion_out = out_dim
        return node

    def temporal_encoder(self, kind: str) -> int:
        return self.add(
            "TemporalEncoder",
            {"inputs": [{"name": "in0", "ty": _port_type()}], "kind": kind, "n_frames": 1, "out_dim": 1, "token_count": 0},
        )

    def head(self, kind: str, horizon: int, action_dim: int = 7) -> int:
        self._last_head_horizon = horizon
        self._head_action_dim = action_dim
        return self.add(
            "PolicyHead",
            {
                "inputs": [{"name": "in0", "ty": _feature_ty(self._fusion_out)}],
                "kind": kind, "action_dim": action_dim, "horizon": horizon,
            },
        )

    def chunker(self, execute_chunk: int, replanning_hz: float, mode: str, horizon: int | None = None) -> int:
        # `horizon` defaults to the immediately preceding `head(...)`'s (spec 14.2's example
        # never passes it explicitly either) — `LRN-021` requires the chunker and the policy
        # contract (`set_policy`) to both quote the same `(horizon, execute_chunk)`.
        horizon = horizon if horizon is not None else self._last_head_horizon
        return self.add(
            "ActionChunker",
            {
                "inputs": [{"name": "in0", "ty": _chunk_ty(horizon, self._head_action_dim)}],
                "horizon": horizon, "execute_chunk": execute_chunk,
                "replan_hz": replanning_hz, "mode": mode, "blend": "HardSwitch", "buffer_chunks": 1,
            },
        )

    def set_policy(
        self,
        architecture: str = "Act",
        weights_path: str = "policy.safetensors",
        action_dim: int = 7,
        horizon: int = 50,
        execute_chunk: int = 20,
        replanning_hz: float = 10.0,
    ) -> None:
        """`PolicyHandle` (`learning.rs`): `LearningGraph::validate` requires one — spec 14.2's
        snippet does not call this explicitly, but a `PolicyBundle`/weights reference is what a
        Learning IR ultimately compiles to (spec 8.4), so `save`/`build` need it filled in."""
        self._native.set_policy(
            json.dumps(
                {
                    "architecture": architecture,
                    "base_model": None,
                    "weights": {"Safetensors": {"path": weights_path, "hash": [0] * 32}},
                    "contract": {
                        "inputs": {},
                        "observation_window": 1,
                        "action_dim": action_dim,
                        "horizon": horizon,
                        "execute_chunk": execute_chunk,
                        "replanning_hz": replanning_hz,
                        "execution_mode": "TemporalEnsemble",
                        "runtime": {"dtype": "F32", "expected_latency_ms": 20.0, "deadline_ms": 50.0},
                    },
                }
            )
        )

    def save(self, path: str) -> None:
        self._native.save(path)


class Deployment:
    """`es.Deployment(name, robot=arm, action=...)` (spec 14.2). Direct setters (spec 9): a
    `DeploymentIr` is a plain struct, not a graph."""

    def __init__(self, name: str, robot: Robot | None = None, action: int | None = None) -> None:
        # `RobotTarget`, `deployment::ActionSpace`, `ExecutionMode`, `Workspace`, `Watchdog` and
        # `FallbackPolicy` (`deployment.rs`) are all `#[serde(rename_all = "snake_case")]` —
        # unlike Task IR's own enums (see `_ACTION_SPACE`), so every tag below is snake_case.
        self.name = name
        self._native = _native.Deployment()
        if robot is not None:
            self._native.set_robot(
                json.dumps({"name": robot.name, "target": {"simulated": {"scene": robot.name}}, "n_joints": 7})
            )
        del action  # resolved from the Task IR's ActionSpec by the compiler (spec 9.2)
        self._native.set_action(
            json.dumps({"space": "joint_position", "dim": 7, "horizon": 50, "execute_chunk": 20})
        )
        self._native.set_execution(json.dumps("receding_horizon"))
        self._native.set_deadlines(
            json.dumps({"observation_age": 100_000, "inference_budget": 50_000, "actuation_budget": 5_000})
        )
        self._native.set_rate(json.dumps({"control": {"num": 30, "den": 1}, "inference": {"num": 10, "den": 1}}))

    @staticmethod
    def box(min_xyz: list[float], max_xyz: list[float]) -> dict:
        return {"box": {"min": min_xyz, "max": max_xyz}}

    def envelope(
        self,
        torque_limit: str = "from_urdf",
        velocity_limit: float = 1.0,
        workspace: dict | None = None,
        rate_limit: float = 1.0,
        n_joints: int = 7,
    ) -> None:
        del torque_limit  # resolved against the URDF by the compiler (spec 9.2), not authored here
        limit = {"lower": -3.14, "upper": 3.14}
        self._native.envelope(
            json.dumps(
                {
                    "position": [limit] * n_joints,
                    "position_soft_margin": [0.1] * n_joints,
                    "velocity_max": [velocity_limit] * n_joints,
                    "acceleration_max": [10.0] * n_joints,
                    "torque_max": [50.0] * n_joints,
                    "jerk_max": None,
                    "action_rate": {"first_diff_max": [rate_limit] * n_joints, "second_diff_max": [rate_limit] * n_joints},
                    "workspace": workspace or Deployment.box([-1, -1, 0], [1, 1, 1]),
                    "ee_velocity_max": velocity_limit,
                    "min_self_distance": 0.02,
                    "min_env_distance": 0.02,
                    "contact_force_max": 100.0,
                }
            )
        )

    def watchdog(self, kind: str, **params: Any) -> None:
        """`.watchdog("inference_deadline", budget_ms=50)` (spec 14.2, snake_case kinds already
        match `Watchdog`'s `rename_all`). A `*_ms` keyword is converted to the `Micros` field the
        Rust side wants (`budget_ms` -> `budget`, in microseconds) since every duration field in
        `Watchdog` follows that naming pattern.
        """
        fields = {(k[:-3] if k.endswith("_ms") else k): (v * 1000 if k.endswith("_ms") else v) for k, v in params.items()}
        self._native.watchdog(json.dumps({kind: fields} if fields else kind))

    def fallback(self, kind: str, **params: Any) -> None:
        self._native.fallback(json.dumps({kind: params} if params else kind))

    def save(self, path: str) -> None:
        self._native.save(path)
