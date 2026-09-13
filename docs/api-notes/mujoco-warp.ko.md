<!-- Korean translation of docs/api-notes/mujoco-warp.md. The English file is the working copy; regenerate this when it changes. -->

# `mujoco_warp` + `warp` (Python) — **검증됨**, `mujoco-warp==3.13.0`에 고정

`MjWarpBackend`(spec 17.1: 배치 GPU 백엔드, spec 12.1: 시뮬레이션 배치 도메인) 뒤의 참조
프로세스인 `crates/es-physics-backend/python/mjwarp_ref.py`가 사용하는 표면.

> **상태: 2026-09-13 검증됨.** 아래 "Calls used" 표의 모든 호출은 실제 설치와 실제 디바이스에
> 대해 실행되었다; 이 파일이 예전에 달고 있던 **unverified** 배너는 사라졌으며, 아직
> 확인되지 않은 것은 각 행에서 그렇게 밝힌다. 이전에 고정했던 `mujoco-warp==0.1.0`은
> 틀렸다 — 패키지 버전은 `mujoco` 자체와 발맞춰 간다.

검증 환경: `mujoco-warp 3.13.0`, `warp-lang 1.17.0`, `mujoco 3.13.0`, Windows 11의
CPython 3.12, CUDA Toolkit 12.9 / 드라이버 13.1, NVIDIA GeForce RTX 4060 Laptop GPU
(8 GiB, sm_89). `mujoco_warp`는 워크스페이스 의존성이 아니다: 선택적이며, 런타임 경로의
어떤 것도 Python을 필요로 해서는 안 된다 (spec 2.4).

## 사용되는 호출 — 행에 달리 적혀 있지 않은 한 검증됨

| 호출 | 검증된 시그니처 | 비고 |
|---|---|---|
| `mujoco_warp.put_model(mjm, batch_sizes=None)` | `(mujoco.MjModel, dict[str, int] \| None) -> Model` | CPU `MjModel`을 한 번 업로드한다. `batch_sizes`는 기본값 그대로. |
| `mujoco_warp.put_data(mjm, mjd, nworld=1, ...)` | `(MjModel, MjData, int, ...) -> Data` | `nworld`가 배치 폭이며, 이 백엔드가 존재하는 이유 전부다. `nconmax` / `njmax` / `nccdmax` / `naconmax` / `nvmax`는 존재하며 기본값 그대로 둔다. |
| `mujoco_warp.step(m, d)` | `(Model, Data) -> None` | 모든 world에 대해 한 번에 물리 틱 하나. `step(n)`당 `n`번 호출된다. |
| `mujoco_warp.forward(m, d)` | `(Model, Data) -> None` | 리셋이나 상태 쓰기 뒤에 호출해 `sensordata` / `xpos` / `xquat`이 `qpos`와 맞도록 한다. `mujoco.mj_forward`에 대응. |
| `warp.init()` | `() -> None` | 프로세스 시작 시 한 번 호출된다. 아래 stdout 관련 항목 참고 — 이 호출이 프로토콜을 망가뜨린 원흉이다. |
| `mujoco.MjModel.from_xml_string`, `mujoco.MjData`, `mujoco.mj_forward`, `mujoco.mj_resetData`, `mujoco.mj_id2name` | `mujoco.md` 참고 | 모델 빌드와 리셋 기준값은 CPU 쪽에 남는다. |

`get_data_into(result, mjm, d, world_id=0)`는 그 시그니처로 존재하지만 의도적으로
**사용하지 않는다**: 한 번에 한 world씩 CPU `MjData`를 통해 읽으므로 배치를 직렬화해
버린다. 상태는 디바이스 배열에서 직접 읽는다.

## `warp`가 stdout에 쓴다 — 실제로 걸려 넘어진 함정

`warp`는 초기화 배너를 **stdout**에 출력한다:

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.1
   Devices: ...
```

그리고 계속 그곳에 출력한다(모듈 컴파일/로드 줄, deprecation 경고). stdout은 JSON-lines
프로토콜이 쓰는 자리이므로, 이것이 바로 그 첫 응답을 오염시켰고 모든 라이브 테스트가
`Protocol("expected value at line 1 column 1 in \`Warp 1.17.0 initialized:\`")`로
실패했다. 그래서 `mjwarp_ref.py`는 먼저 진짜 핸들을 붙잡아 두고 `sys.stdout`을
`sys.stderr`로 돌려버리며, Rust 쪽은 이를 버린다:

```python
_OUT = sys.stdout
sys.stdout = sys.stderr
```

`warp`를 임포트하는 향후의 out-of-process 백엔드는 무엇이든 이 두 줄이 필요하다;
`newton_ref.py`도 이를 갖고 있다.

## 읽고 쓰는 필드 — 검증됨

`Data`: `qpos`, `qvel`, `act`, `ctrl`, `sensordata`, `xpos`, `xquat`. 각각 **첫 번째
축이 `nworld`**인 `warp.array`이며, 이것이 별도의 전치(transposition) 없이 wire 포맷을
env-major로 만드는 이유다. `.numpy()`로 읽고 `.assign(ndarray)`로 쓴다. 둘 다
검증되었다: 두-world 배치를 `qpos = [0.3, -0.2]`로 설정하고 스텝을 진행하니 서로 다른,
독립적인 답으로 갔다.

- 배열은 **float32**다. 값은 프로세스 경계에서 f64로 넓혀지므로, `MjWarpBackend`는
  `FloatPrecision::F32`와 결정성 계층 2(spec 3.5)를 선언한다: `mujoco-cpu`와의 일치는
  허용오차이지 결코 비트 동일성의 보장이 아니다. 다르게 선언하는 것은 spec 17.3이 경고하는
  정직하지 못한 상태일 것이다 — 아래 측정값 참고.
- `xquat`는 CPU에서와 마찬가지로 **wxyz**이며, wire를 건너기 전에 spec 3.1의 xyzw로
  재정렬된다.
- 모델 메타데이터(`nq`, `nv`, `nu`, `nsensordata`, `nbody`, `jnt_qposadr`, `sensor_adr`
  등)는 `put_model`이 빌드된 원본 CPU `MjModel`에서 읽으므로, `load` 응답은
  `mujoco_ref.py`가 반환하는 것과 정확히 같은 필드를 가지며 세 어댑터가 하나의 Rust
  디코더를 공유한다.

## 측정값 (spec 12.4: 무엇이 만들어냈는지와 함께 제시하는 숫자)

위의 하드웨어, `armature = 0.01`, `damping = 0.1`인 1 kHz 진자.

| 측정 항목 | 값 | 방법 |
|---|---|---|
| Run-to-run 재현성, 200틱, 2 world | **비트 동일**, `max \|delta\| = 0` | `mjwarp::tests::mjwarp_runs_agree_to_the_declared_tier`, 같은 저장 상태에서 시작한 두 개의 새 프로세스 |
| `mujoco-cpu` 대비 `max \|dqpos\|`, 200틱 | **7.41e-8** | 수평으로 시작하는 막대에 대한 `mjwarp::tests::mjwarp_against_mujoco_cpu` |
| `mujoco-cpu` 대비 `max \|dqvel\|`, 200틱 | 7.76e-7 | 동일 |
| 에너지 proxy 델타 | 3.16e-6 | 동일 |

**선언된 계층은 그대로 2다.** 한 GPU, 한 드라이버에서 한 씬이 비트 단위로 재현된다는 것은
관측이지 보장이 아니다: spec 17.3은 `mujoco-cpu`만 계층 1을 선언한다고 말하며, 테스트는
tier-2 계약(`delta < 1e-9`)을 단언(assert)할 뿐, 실행이 비트 단위였는지는 단지 *출력*할
뿐이다. 오늘 재현 가능한 것으로 밝혀진 GPU 백엔드가, 다음 드라이버에서 실패하는 테스트가
되어서는 안 된다.

비교 씬 선택이 중요하다: `compare_backends`는 제어 없이 리셋하고 스텝하므로, 수직으로
매달린 진자는 평형점에 놓여 공허하게 `max |dqpos| = 0`을 기록한다. 그래서 비교 테스트는
수평으로 시작하는 막대를 사용한다.

## spec 17.2가 고정하는 의미론

| Task IR | MJWarp 매핑 | 상태 |
|---|---|---|
| `actuator.pd(kp, kd)` | position actuator gain | native |
| `contact.friction_cone` | pyramidal | native; **elliptic** 씬은 (조용히 다시 pyramidal로 만드는 대신) 이름으로 지목해 거부된다(`severity: error`, spec 14.4) |
| `contact.soft_params` | `solref` / `solimp` impedance | native |
| `joint.armature` | armature | native |
| `sensor.contact_force` | — | 차단됨: 공유 MJCF emitter가 force/touch 센서를 전혀 쓰지 않는다 |
| `contact.condim = 6` | — | `TODO(api-notes)`: 아직 미검증, 경고와 함께 unsupported로 선언됨 |

이 표는 코드 안에 `crates/es-physics-backend/src/mapping.rs`로 존재하며,
`MjWarpBackend::capabilities()`는 이로부터 *도출*되므로 선언과 리포트가 서로 어긋날 수
없다.

## 짚어둘 만한 동작

- **리셋은 `mj_resetData`를 의미한다.** 프로세스는 리셋 기준으로 CPU `MjData` 하나를
  들고 있다가 그 `qpos` / `qvel` / `act`를 선택된 world들에 복사하므로, "초기 상태"는
  두 백엔드에서 같은 것을 뜻하며 크로스-백엔드 비교가 같은 지점에서 시작한다.
- **부분집합 리셋은 자신의 행만 건드리며**, 전체 배치 리셋만이 틱을 되감는다.
- **발산은 보고되지, 예외로 던져지지 않는다**: non-finite인 world는 `step`의
  `nonfinite` 리스트로 돌아와 `StepReport::failures`가 된다 (spec 18.5).
- Python 쪽의 어떤 실패든 `{"ok": false, "error": ...}` 한 줄로 나타나므로, 잘못된 API
  이름은 그 속성을 지목하는 `PhysicsError::Backend`로 드러난다.
- **첫 호출이 느리다.** `put_model` / 첫 `step`이 warp 커널을 컴파일하고 캐시한다; 캐시는
  `%LOCALAPPDATA%\NVIDIA\warp\Cache\<version>` 아래에 있다. 콜드 캐시는 수 초가 걸리고,
  웜 캐시는 수 밀리초다.

## 아직 미검증

- `contact.condim = 6`.
- 2 world를 넘어서는 배치 폭. `MAX_ENVS = 8192`는 측정치가 아니라 *선언된* 상한이다
  (spec 12.4); 실제 한계는 디바이스 메모리다.
- 충돌이 많은 씬. 위에서 측정한 모든 것은 충돌이 없는 단일 힌지다.
- `nconmax` / `njmax` 크기 산정, 충돌이 많은 씬에는 필요할 것이다.
