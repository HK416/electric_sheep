<!-- Korean translation of docs/api-notes/newton.md. The English file is the working copy; regenerate this when it changes. -->

# `newton` (Python) — **부분적으로 검증됨**, `newton==1.6.0`에 고정

`NewtonBackend`(spec 4.3: Newton은 M2 백엔드다, spec 17.2: 백엔드 의미 매핑) 뒤의 참조
프로세스인 `crates/es-physics-backend/python/newton_ref.py`가 사용하는 표면.

> **상태: load / step / state 경로는 검증되었으나, contact와 액추에이션은 검증되지
> 않았다.** 아래 표의 모든 호출은 실제 설치와 실제 디바이스에 대해 실행되었다. 검증되지
> *않은* 것은 그렇게 밝히며, `NewtonBackend`는 이에 대해 어떤 capability도 선언하지 않는다
> — spec 1.7: 그럴듯하지만 지어낸 API 이름은 에이전트가 저지를 수 있는 가장 값싼 실수다.

검증 환경: `newton 1.6.0`, `warp-lang 1.17.0`, Windows 11의 CPython 3.12, CUDA Toolkit
12.9 / 드라이버 13.1, NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB, sm_89). `newton`은
워크스페이스 의존성이 아니다: 선택적이며, 런타임 경로의 어떤 것도 Python을 필요로 해서는
안 된다 (spec 2.4).

## 패키지 받기

PyPI 이름은 `newton-physics`가 아니라 **`newton`**이다. `pip install newton-physics`는
다음을 던지는 것이 유일한 동작인 1.6 kB짜리 스텁을 설치한다:

```
ImportError: The 'newton-physics' package has been renamed to 'newton'.
```

`pip install newton`은 6.2 MB짜리 순수 Python 휠을 받아오며 `warp-lang>=1.17.0`만
필요로 하는데, 이는 이미 이 워크스페이스가 `MjWarpBackend`를 위해 갖고 있는 것이다.
CUDA 전용 휠은 필요 없다 — `warp`가 디바이스 계층을 제공하며, 같은 휠이 CPU만으로도
동작한다.

## 사용되는 호출 — 검증됨

| 호출 | 검증된 시그니처 | 비고 |
|---|---|---|
| `newton.ModelBuilder(up_axis=Axis.Z, gravity=None)` | `-> ModelBuilder` | env 템플릿당 빌더 하나. |
| `ModelBuilder.add_mjcf(source, *, parse_mujoco_options=True, ...)` | `(str, ...) -> None` | MJCF 텍스트를 읽으므로, 같은 `scene_to_mjcf` emitter가 세 백엔드 모두에 입력을 준다. 아래 공백 참고. |
| `ModelBuilder.add_world(builder)` | `(ModelBuilder, ...) -> None` | 복제. `n_envs`번 호출된다; `Model.world_count`가 spec 12.1의 배치가 된다. |
| `ModelBuilder.finalize()` | `(...) -> Model` | 디바이스로 업로드한다. |
| `Model.state()` / `Model.control()` | `-> State` / `-> Control` | 두 개의 `State`를 유지하며 서로 교체하는데, 솔버의 in/out 시그니처에 맞춘 것이다. |
| `newton.solvers.SolverFeatherstone(model)` | `(Model) -> SolverBase` | 실제로 사용되는 솔버. 아래의 `SolverMuJoCo` 차단 항목 참고. |
| `SolverBase.step(state_in, state_out, control, contacts, dt)` | `(State, State, Control \| None, Contacts \| None, float) -> None` | 여기서는 `contacts=None` — 연결되어 있지 않다. |
| `newton.eval_fk(model, joint_q, joint_qd, state)` | `(Model, array, array, State) -> None` | 리셋이나 상태 쓰기 뒤에 호출해 `body_q`가 `joint_q`와 맞도록 한다. `mujoco.mj_forward`에 대응. |
| `State.clear_forces()` | `() -> None` | 각 솔버 스텝 전에. |

`warp`는 배너를 stdout에 출력한다; `newton_ref.py`는 `mjwarp_ref.py`와 동일한
`sys.stdout` 가드를 갖고 있다. 이유는 `mujoco-warp.md` 참고.

## 읽고 쓰는 필드 — 검증됨

| 필드 | Shape | 비고 |
|---|---|---|
| `State.joint_q` | flat, world-major | 일반화 좌표(generalized coordinates). env당 `Model.joint_coord_count / world_count`. |
| `State.joint_qd` | flat, world-major | 일반화 속도(generalized velocities). |
| `State.body_q` | `(nbody, 7)` | Newton의 transform은 **(pos xyz, quat xyzw)**다 — MuJoCo의 wxyz와 달리 이미 spec 3.1 순서이므로 swizzle이 필요 없다. |
| `Model.joint_q_start` / `joint_qd_start` | `njoint + 1` | `joint_q` / `joint_qd`로의 조인트별 주소, fencepost로 종료됨: 조인트 `i`는 `[start[i], start[i+1])`을 차지한다. |
| `Model.joint_label` | `list[str]` | 경로 형태, 예: `zoo/worldbody/rod/hinge`. MJCF 이름은 **마지막 `/` 세그먼트**다. `Model.body_label`도 마찬가지. |
| `Model.joint_type` | `njoint` | `JointType`: `PRISMATIC=0, REVOLUTE=1, BALL=2, FIXED=3, FREE=4, DISTANCE=5, D6=6, ROD=7`. |
| `Model.joint_armature`, `joint_limit_lower/upper`, `joint_damping` | dof당 | MJCF 값을 그대로 담는다. |

배열은 **float32**이며 프로세스 경계에서 f64로 넓혀지므로, `NewtonBackend`는
`FloatPrecision::F32`와 결정성 계층 2를 선언한다 (spec 17.3: GPU 백엔드는 결코 계층 1을
선언하지 않는다).

**`add_mjcf`가 임포트하는 것으로 검증됨:** 네 가지 MJCF 조인트 종류 모두(`freejoint`,
`hinge`, `slide`, `ball`), 조인트 한계(MuJoCo의 `angle="degree"` 기본값이 준수된다 —
`range="-1 1"`이 `-1.745e-2` rad이 되었다), 그리고 `armature`.

## 공백 — *부재*로 검증되었으며, 어댑터가 이에 대해 하는 일

`NewtonBackend`가 씬을 그대로 실행하는 대신 거부하는 이유가 바로 이것들이다:

| 공백 | 근거 | 어댑터의 동작 |
|---|---|---|
| `add_mjcf`는 **`<actuator>`를 전혀 임포트하지 않는다** | `<motor>`가 있는 MJCF를 임포트한 뒤 `Model.actuators == []`, `Model.joint_target_mode == [0]` | `nu = 0`; 모든 액추에이터 feature가 `severity: error`이므로, 액추에이션이 있는 씬은 액추에이션 없이 실행되는 대신 **이름으로 지목되어 거부된다** (spec 14.4) |
| `add_mjcf`는 **`<sensor>`를 전혀 임포트하지 않는다** | `Model`에 센서 배열이 없다 | `nsensordata = 0`; 모든 센서 feature가 차단된다 |
| **Contact가 연결되어 있지 않다** | 이 어댑터는 `contacts=None`으로 스텝한다; `newton.CollisionPipeline`은 존재하지만 사용되지 않는다 | MJCF 기본 cone만 선언되며, 물체가 서로를 통과한다는 quirk와 함께 `approximated`로 표시된다. `ContactElliptic` / `ContactSoftParams` / `ContactCondim6` / `ContactMesh` / `ContactHeightField`는 차단된다 |
| 조인트 **spring**과 **friction loss** | 확인된 적 없음 | `TODO(api-notes)`: 미검증, 경고와 함께 unsupported로 선언됨 |

## `SolverMuJoCo`는 여기서 차단된다 — 버전 충돌

`newton.solvers.SolverMuJoCo`는 MJCF 씬에 대한 자연스러운 선택이었겠지만, newton
1.6.0의 `[sim]` extra는 **`mujoco-warp~=3.12.0`**을 고정하는 반면 이 워크스페이스는
`MjWarpBackend`를 위해 **3.13.0**을 필요로 한다. 3.13.0에 대해서는 `SolverMuJoCo` 생성이
커널 컴파일 중에 실패한다:

```
WarpCodegenError: Error while parsing function "convert_mjw_contacts_to_newton_kernel"
  ... Couldn't find function overload for 'contact_force_fn' ...
```

한 환경이 두 고정 버전을 동시에 담을 수 없으며, `MjWarpBackend`가 이 대립에서 이긴다:
그것이 기본 백엔드이고(spec 4.3), spec 17.1이 지목하는 것이다. 그래서
`NewtonBackend`는 Newton 자체의 축약좌표(reduced-coordinate) articulated-body 솔버인
**`SolverFeatherstone`**로 스텝한다.

이것이 오히려 더 나은 테스트일 수도 있다 — spec 17.2가 존재하는 이유는 같은 Task IR이
백엔드마다 다르게 동작해서는 안 되기 때문이며, 진짜로 다른 솔버야말로 그 비교를 해볼 가치가
있게 만든다. 이는 또한 Newton이 `mujoco-cpu`와 어긋나는 것이 반올림이 아니라 물리라는
뜻이기도 하다; 아래 숫자가 이를 말해준다.

newton이 이 고정을 완화하거나, `mujoco-warp` 3.13 지원이 newton에 들어오면 다시 검토할 것.

## 측정값 (spec 12.4: 무엇이 만들어냈는지와 함께 제시하는 숫자)

위의 하드웨어, `armature = 0.01`, `damping = 0.1`, 수평으로 시작하는 막대인 1 kHz 진자.

| 측정 항목 | 값 | 방법 |
|---|---|---|
| `mujoco-cpu` 대비 `max \|dqpos\|`, 200틱 | **5.88e-4** | `newton::tests::newton_against_mujoco_cpu` |
| `mujoco-cpu` 대비 `max \|dqvel\|`, 200틱 | 5.02e-3 | 동일 |
| 에너지 proxy 델타 | 2.06e-2 | 동일 |
| 1e-6 허용오차를 넘어서는 첫 틱 | **6번 틱** | 동일 |
| 같은 씬에서 `mjwarp`, 비교 참고용 | 7.41e-8 | `mujoco-warp.md` |

두 GPU 백엔드 사이의 네 자릿수 차이가 핵심이다: `mjwarp`는 MuJoCo의 솔버를 실행하며
MuJoCo와 일치한다; Newton은 Featherstone을 실행하며 일치하지 않는다. 단언 경계
(`< 1e-2`)는 두 궤적이 같은 궤적 위에 머무른다는 것을 기록할 뿐이다. 이는 검증된
일치 허용오차가 **아니다**.

## spec 17.2가 고정하는 의미론

| Task IR | Newton 매핑 | `mapping.rs`에서의 상태 |
|---|---|---|
| `actuator.pd(kp, kd)` | 조인트 컨트롤러 (`ControllerPD` / `DrivePD`, `Model.joint_target_ke/kd`를 갖는 `Control.joint_target_q`) | approximated — *엔진*에는 존재한다; `add_mjcf`가 이를 연결하지 않으므로 capability 행이 차단된다 |
| `contact.friction_cone` | pyramidal 또는 elliptic 선택 가능 | spec 표대로라면 native: `SolverMuJoCo(cone=...)`가 이를 받는다. `SolverFeatherstone`을 통해서는 도달 불가 |
| `contact.soft_params` | 솔버 의존적 | approximated |
| `joint.armature` | armature | **native, 검증됨**: `Model.joint_armature`가 MJCF 값을 그대로 담는다 |
| `sensor.contact_force` | contact buffer에서 읽음 | approximated — 이 어댑터에서는 연결되어 있지 않다 |

위 Spec17 행들은 *엔진*이 무엇을 하는지를 말한다; capability 행은 `NewtonBackend`가
실제로 *제공*하는 것으로 검증된 바를 말한다. 둘이 다를 때는 capability 행이 실행을
좌우한다(spec 14.4). 둘 다 `crates/es-physics-backend/src/mapping.rs`에 존재하며,
`newton::tests::the_declaration_is_the_newton_column_of_the_mapping`은 선언과 표가
feature 단위로 일치함을 단언한다.

## 다음 실행 체크리스트

1. `newton.CollisionPipeline`을 연결하고 contact feature를 다시 선언하거나, 계속
   거부한다.
2. `add_mjcf`가 하지 않을 것이므로 MuJoCo 액추에이터를 `Control.joint_f` /
   `joint_target_q`에 명시적으로 매핑한다; 그래야만 비로소 `nu`가 0이 아니게 될 수 있다.
3. `mujoco-warp` 고정이 3.13을 허용하게 되면 `SolverMuJoCo`를 다시 시도해 두 솔버를
   비교한다.
4. 2 world를 넘는 배치를 측정한다. `MAX_ENVS = 8192`는 선언되었을 뿐 측정되지 않았다.
