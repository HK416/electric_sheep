<!-- Korean translation of docs/api-notes/mujoco-warp.md. The English file is the working copy; regenerate this when it changes. -->

# `mujoco_warp` + `warp` (Python) — **미검증 (unverified)**, `mujoco-warp==0.1.0`에 고정

`MjWarpBackend`(spec 17.1: 배치 GPU 백엔드, spec 12.1: 시뮬레이션 배치 도메인) 뒤편의 참조 프로세스인
`crates/es-physics-backend/python/mjwarp_ref.py`가 사용하는 표면(surface)이다.

> **상태: 미검증 (unverified).** `docs/api-notes/mujoco.md`와 달리 이 파일의 내용은 아무것도 실행된
> 적이 없다. 이 작업 패킷을 구현한 머신에는 CUDA 디바이스가 없고 `mujoco_warp`도 `warp`도 설치되어
> 있지 않아, `MjWarpBackend::is_available()`가 실패하고 모든 라이브 테스트가 이유를 출력하며
> 스킵된다. `mujoco_warp`는 API가 아직 계속 바뀌는 젊은 패키지다. spec 1.7은 그럴듯하지만 지어낸
> API 이름을 에이전트의 상시적 실패 모드로 지목하므로, **아래의 모든 호출에는 그것이 어떻게
> 도출되었는지를 표시**했고, GPU가 있는 첫 머신에서 반드시 재확인해야 한다. 그때까지
> `MjWarpBackend`는 프로토콜은 완성되었지만 엔진 검증은 되지 않은 상태다.

첫 검증 단계에 고정된 버전: CPython 3.12, CUDA 12.x 기준 `mujoco-warp==0.1.0`, `warp-lang>=1.7`,
`mujoco==3.13.0`. `mujoco_warp`는 워크스페이스 의존성이 아니라 선택적(optional) 의존성이며,
런타임 경로의 어떤 부분도 Python을 필요로 해서는 안 된다(spec 2.4).

## 사용되는 호출 — 전부 미검증 (unverified)

| Call | Signature as used | Derived from |
|---|---|---|
| `mujoco_warp.put_model(mjm)` | `(mujoco.MjModel) -> Model` | 패키지 문서에 명시된 진입점: CPU `MjModel`을 한 번 업로드한다. |
| `mujoco_warp.put_data(mjm, mjd, nworld=n)` | `(MjModel, MjData, int) -> Data` | 동일; `nworld`는 배치 폭(batch width)이며, 이 백엔드가 존재하는 이유 그 자체다. 다른 키워드 인자(`nconmax`, `njmax`)도 존재하지만 기본값으로 둔다. |
| `mujoco_warp.step(m, d)` | `(Model, Data) -> None` | 모든 월드에 대해 한 번에 물리 틱 하나를 진행한다. `step(n)`마다 `n`번 호출된다. |
| `mujoco_warp.forward(m, d)` | `(Model, Data) -> None` | 리셋이나 상태 쓰기 이후 `sensordata` / `xpos` / `xquat`가 `qpos`와 일치하도록 호출한다. `mujoco.mj_forward`에 대응한다. |
| `warp.init()` | `() -> None` | 프로세스 시작 시 한 번 호출된다. CUDA 디바이스가 없으면 여기서 실패하며, 트레이스백이 아니라 프로토콜 오류로 보고된다. |
| `mujoco.MjModel.from_xml_string`, `mujoco.MjData`, `mujoco.mj_forward`, `mujoco.mj_resetData`, `mujoco.mj_id2name` | see `mujoco.md` | 검증됨 — 모델 빌드와 리셋 기준값은 CPU 쪽에 남는다. |

`get_data_into(mjd, mjm, d)`는 의도적으로 **사용하지 않는다**: CPU `MjData`를 거쳐 읽으면 배치가
직렬화(serialise)되어 버린다. 상태는 디바이스 배열에서 직접 읽는다.

## 읽고 쓰는 필드 — 전부 미검증 (unverified)

`Data`: `qpos`, `qvel`, `act`, `ctrl`, `sensordata`, `xpos`, `xquat`. 각각은 `warp.array`이며
**첫 번째 축이 `nworld`**다. 이 덕분에 전치(transposition) 없이 와이어 포맷이 env-major가 된다.
`.numpy()`로 읽고 `.assign(ndarray)`로 쓴다.

- 배열은 **float32**다. 값은 프로세스 경계에서 f64로 확장(widen)되므로, `MjWarpBackend`는
  `FloatPrecision::F32`와 결정성 계층 2(spec 3.5)를 선언한다: `mujoco-cpu`와의 일치는 어디까지나
  허용오차(tolerance)이지 비트 단위 일치가 아니다. 다르게 선언하는 것은 spec 17.3이 경고하는
  정직하지 못한 상태에 해당할 것이다.
- `xquat`는 CPU에서와 마찬가지로 **wxyz** 순서이며, 와이어를 건너기 전에 spec 3.1의 xyzw 순서로
  재정렬된다.
- 모델 메타데이터(`nq`, `nv`, `nu`, `nsensordata`, `nbody`, `jnt_qposadr`, `sensor_adr` 등)는
  `put_model`이 만들어진 원본 CPU `MjModel`에서 읽어온다. 따라서 `load` 응답은 `mujoco_ref.py`가
  반환하는 필드와 정확히 동일하며, 두 어댑터가 하나의 Rust 디코더를 공유한다.

## spec 17.2로 고정된 의미

| Task IR | MJWarp mapping | Status |
|---|---|---|
| `actuator.pd(kp, kd)` | position actuator gain | native |
| `contact.friction_cone` | pyramidal | native; elliptic 씬은 조용히 다시 원뿔화(re-cone)되지 않고 이름으로 지목되어 거부된다(`severity: error`, spec 14.4) |
| `contact.soft_params` | `solref` / `solimp` impedance | native |
| `joint.armature` | armature | native |
| `sensor.contact_force` | — | blocked: 공유 MJCF 이미터가 force/touch 센서를 전혀 기록하지 않는다 |
| `contact.condim = 6` | — | `TODO(api-notes)`: 미검증 (unverified), 경고와 함께 unsupported로 선언됨 |

이 표는 코드상으로 `crates/es-physics-backend/src/mapping.rs`에 존재하며,
`MjWarpBackend::capabilities()`가 이로부터 *도출*되므로 선언과 리포트가 서로 어긋날 수 없다.

## 못 박아 둘 만한 동작

- **리셋은 곧 `mj_resetData`를 의미한다.** 프로세스는 리셋 기준값으로 CPU `MjData` 하나를
  유지하며, 그 `qpos` / `qvel` / `act`를 선택된 월드들에 복사한다. 이로써 '초기 상태'는 두
  백엔드에서 동일한 의미를 가지며, 백엔드 간 비교가 같은 지점에서 시작된다.
- **부분(subset) 리셋은 해당 행(row)만 건드리며**, 틱(tick)을 되감는 것은 전체 배치 리셋뿐이다.
- **발산(divergence)은 예외로 발생시키지 않고 보고된다**: non-finite 상태가 된 월드는 `step`의
  `nonfinite` 목록으로 돌아오고 `StepReport::failures`가 된다(spec 18.5).
- Python 쪽에서 발생하는 어떤 종류의 실패든 `{"ok": false, "error": ...}` 한 줄로 표현된다.
  따라서 잘못된 API 이름은 해당 속성(attribute)을 지목하는 `PhysicsError::Backend`로 나타나며 —
  이것이 바로 첫 GPU 실행이 이 파일의 어느 행이 틀렸는지를 우리에게 알려주는 방식이다.

## 첫 GPU 실행 체크리스트

1. `python -c "import mujoco_warp, warp"` — 이것이 성공하면 `MjWarpBackend::is_available()`가
   `Ok`를 반환한다.
2. `cargo test -p es-physics-backend mjwarp_pendulum -- --nocapture` — 더 이상 `SKIP`을 출력하지
   않는다.
3. `cargo test -p es-physics-backend mjwarp_against_mujoco_cpu` — CPU 오라클과 비교하는 spec 3.5
   계층 3 비교 테스트. 이 테스트의 `max |dqpos| < 1e-2` 경계값은 검증된 허용오차가 아니라
   스모크(smoke) 경계값이므로, 실측값으로 교체하고 9-지표 세트를 기록한다(spec 12.4).
4. 위의 모든 행을 다시 확인하고, 실제로 실행이 확인된 부분에 한해서만 **미검증 (unverified)**
   배너를 삭제한다.
