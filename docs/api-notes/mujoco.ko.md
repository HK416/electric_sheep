<!-- Korean translation of docs/api-notes/mujoco.md. The English file is the working copy; regenerate this when it changes. -->

# `mujoco` (Python), 3.13.0에 고정

`MuJoCoCpuBackend`(spec 17.1)의 참조 프로세스인
`crates/es-physics-backend/python/mujoco_ref.py`가 사용하는 표면. CPython 3.12 /
3.13 (Windows x86-64)에서 `mujoco==3.13.0` 휠에 대해, 크레이트 자체의 오라클
테스트를 실행해 확인했다. 워크스페이스의 고정 의존성은 아니다 — 패키지는
선택적이며, `MuJoCoCpuBackend::is_available()`이 부재를 보고하면 CI는 실패하는
대신 오라클을 건너뛴다 (spec 2.4 — 런타임 경로의 어떤 것도 Python을 필요로 하지
않는다).

3.x 라인이면 무엇이든 동작해야 한다. CI 이미지가 설치하는 버전이 바뀌면 이 파일을
갱신한다.

## Calls used

| Call | 사용된 시그니처 | 비고 |
|---|---|---|
| `mujoco.MjModel.from_xml_string(xml)` | `(str) -> MjModel` | 모델이 잘못되면 `ValueError`를 발생시키며, 메시지는 `PhysicsError::Backend`로 그대로 전달된다. |
| `mujoco.MjData(model)` | `(MjModel) -> MjData` | 환경당 하나씩; `n_envs`는 이들의 리스트를 들고 있는 방식으로 에뮬레이션된다. |
| `mujoco.mj_step(model, data)` | `(MjModel, MjData) -> None` | 물리 틱 하나. `step(n)`당 `n`번 호출된다. |
| `mujoco.mj_forward(model, data)` | `(MjModel, MjData) -> None` | 리셋이나 상태 쓰기 뒤에 호출해, `sensordata` / `xpos` / `xquat`이 `qpos`와 맞도록 한다. |
| `mujoco.mj_resetData(model, data)` | `(MjModel, MjData) -> None` | 모델의 초기 상태로 되돌린다. |
| `mujoco.mj_id2name(model, objtype, id)` | `(MjModel, mjtObj, int) -> str \| None` | 이름 없는 요소면 `None`; emitter는 항상 이름을 쓰므로 여기서 `None`이 나오면 버그이고 `PhysicsError::Protocol`로 드러난다. |
| `mujoco.mjtObj.mjOBJ_JOINT` / `mjOBJ_ACTUATOR` / `mjOBJ_SENSOR` / `mjOBJ_BODY` | enum | 이 네 네임스페이스만 조회한다. |

## Fields read

`MjData`: `qpos`, `qvel`, `act`, `ctrl`, `sensordata`, `xpos`, `xquat` — 모두
`float64`의 `numpy` 배열. `xpos`는 `(nbody, 3)`, `xquat`는 `(nbody, 4)`이며,
**`xquat`는 wxyz**이므로 wire를 건너기 전에 spec 3.1의 xyzw로 재정렬한다.

`MjModel`: `nq`, `nv`, `nu`, `nsensordata`, `nbody`, `njnt`, `nsensor`, `jnt_type`,
`jnt_qposadr`, `jnt_dofadr`, `sensor_adr`, `sensor_dim`, `opt.timestep`.

`jnt_type`은 `mjtJoint`이다: `0 = free` (qpos 7 / dof 6), `1 = ball` (4 / 3), `2 =
slide` (1 / 1), `3 = hinge` (1 / 1). 스크립트는 주소 차이로부터 너비를 유도하는
대신 이 표를 직접 들고 있는다.

## Behaviour worth pinning

- **센서 노이즈는 기본적으로 꺼져 있다.** 센서의 `noise`와 `cutoff`는 MJCF에
  쓰이지만, `MuJoCo`는 `sensornoise` enable 플래그가 있을 때만 노이즈를 적용하며,
  이 어댑터는 그 플래그를 설정하지 않는다. 백엔드 capabilities(spec 17.2)에
  `BackendQuirk`로 선언되어 있다.
- **`autolimits`는 3.x에서 기본값이 true**이므로, `limited` 속성이 없는 joint
  `range`는 제한된 것으로 취급된다. Emitter는 이를 그대로 활용해 `limited`를 쓰지
  않는다.
- **`fullinertia`는 관성 프레임을 덮어쓴다:** `MuJoCo`가 이를 고유분해
  (eigendecompose)해 프레임을 그 결과로 설정하므로, `quat`과 `fullinertia`는
  함께 쓸 수 없다. Emitter는 관성이 대각(diagonal)이면 `quat` + `diaginertia`를,
  프레임이 항등(identity)이면 `fullinertia`를 쓰고, 나머지 경우는 회전을 잃는 대신
  이름을 명시해 거부한다.
- 평면(plane)의 세 번째 `size` 성분은 렌더링 격자 간격이며 양수여야 한다.

## Protocol

`python -c "<embedded script>"`, 양방향으로 줄당 JSON 객체 하나. 요청은 `cmd`를
담고, 응답은 `{"ok": true, ...}` 또는 `{"ok": false, "error": "<ExcType>:
<message>"}`다 — 모델링 오류는 프로세스가 죽는 것이 아니라 값이다. 배열은 숫자의
JSON 리스트이며 env-major다. Python float에 대한 `json.dumps`와
`float_roundtrip`을 쓴 `serde_json`은 둘 다 최단 왕복(shortest-round-trip)이므로
`f64` 값은 정확히 그대로 건너간다 — 비트 단위 run-to-run 테스트가 이에 의존한다.

| Request | Reply |
|---|---|
| `{"cmd":"load","mjcf":str,"n_envs":int,"timestep":float\|null,"seed":int}` | `nq, nv, nu, nsensordata, nbody, joints[{name,qpos:[adr,dim],dof:[adr,dim]}], actuators[name], sensors[{name,adr,dim}], bodies[name]` |
| `{"cmd":"reset","envs":[int]\|null,"state":{...}\|null}` | `{}` |
| `{"cmd":"set_ctrl","ctrl":[float]}` | `{}` |
| `{"cmd":"step","n":int}` | `{"nonfinite":[env]}` |
| `{"cmd":"state"}` | `qpos, qvel, act, sensordata, xpos, xquat` |
| `{"cmd":"set_state","state":{"qpos":[],"qvel":[],"act":[]}}` | `{}` |
| `{"cmd":"quit"}` | *(응답 없음; 프로세스가 종료된다)* |

`ES_PYTHON`이 인터프리터를 선택한다; 설정되지 않으면 `python`, 그다음 `python3`을
시도한다.
