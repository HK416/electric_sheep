<!-- Korean translation of docs/packets/M2/W4-newton-backend.md. The English file is the working copy; regenerate this when it changes. -->

# W4 — Newton 백엔드 어댑터

Spec: spec 4.3 (`NewtonBackend`는 M2 백엔드다 — "필요할 때의 Kamino / VBD /
hydroelastic"), spec 17.2 (백엔드 의미 매핑과 `es backend compare` — "새로운 핵심
작업"), spec 17.3 (결정성 계층: GPU 백엔드는 계층 2를 선언한다), spec 14.4
(`severity: error`인 미매핑 항목은 실행을 차단한다), spec 11.6 (capability 검사),
spec 12.1 (시뮬레이션 배치 도메인), spec 12.4 (측정된 숫자, 결코 단일 수치가 아니라),
spec 1.7 (지어낸 API 이름은 에이전트의 상시적 실패 모드다), spec 18.1 / spec 18.5
(정수 틱, 보고되는 발산), spec 2.4 (런타임 경로의 어떤 것도 Python을 링크하지
않는다). M2 Wave 4. Review class B.

## context

```
crates/es-physics-backend/src/newton.rs        (신규)
crates/es-physics-backend/python/newton_ref.py (신규)
crates/es-physics-backend/src/mapping.rs       (Newton 컬럼 + 전제가 바뀐 테스트 두 개)
crates/es-physics-backend/src/lib.rs           (`pub mod` 한 줄 + re-export 한 줄)
crates/es-physics-backend/src/proc.rs          (공유되는 spawn / call / drop; W1-mjwarp-live.md 참고)
docs/api-notes/newton.md
docs/packets/M2/W4-newton-backend.md
```

## forbidden

`mjwarp.rs`와 `mujoco.rs`의 의미론, `mjcf_out.rs`, 그리고 `es-physics-backend`
밖의 모든 crate. **`es` CLI는 범위 밖이다**: 아래 "CLI note" 참고.

## spec

`NewtonBackend`는 `MjWarpBackend`를 그대로 따른다 — `newton_ref.py`를 실행하는
Python 서브프로세스로, `proc.rs`의 개행 구분 JSON 프로토콜을 말하며, Newton의
MJCF importer를 통해 로드해 하나의 `scene_to_mjcf` emitter가 세 백엔드 모두에
입력을 주고, Newton의 솔버로 스텝하며, env-major 상태 배열을 반환한다.

- `capabilities()`: 결정성 계층 2(`CrossBackend`), `gpu_resident: true`,
  `max_envs = 8192`, `FloatPrecision::F32`.
- `mapping.rs`의 Newton 컬럼은 엔진이 원리상 할 수 있는 것이 아니라 **검증된**
  것으로 채워졌다.
- `BackendKind::Newton`은 이미 존재했고 `compare_backends`는
  `Capabilities::name`으로부터 컬럼을 해석하므로, 백엔드를 `"newton"`으로
  이름 붙이는 것만으로 더 이상의 변경 없이 연결된다.

## oracle

```
cargo fmt -p es-physics-backend --check
cargo clippy -p es-physics-backend --all-targets -- -D warnings
ES_PYTHON=<venv python> cargo test -p es-physics-backend -- --nocapture | grep -E 'SKIP|RAN|qpos|test result'
cargo xtask check-spec-refs
```

테스트: 매핑 행, 미리 준비된 JSON에 대한 프로토콜 왕복(Python도 GPU도 없이), 그리고
Newton이 임포트되면 `RAN`을 그렇지 않으면 `SKIP <reason>`을 출력하는 라이브
테스트.

## findings

### 패키지는 `newton-physics`가 아니라 `newton`이다

`pip install newton-physics`는 `ImportError: ... has been renamed to 'newton'`을
던지는 1.6 kB짜리 스텁을 설치한다. 진짜 휠은 `newton==1.6.0`, 6.2 MB, 순수
Python이며, `warp-lang>=1.17.0`만 필요로 한다 — `MjWarpBackend`를 위해 이미
있는 것이다. **차단 요인 없음: Windows + CUDA에서 설치되고 실행되었다.**

### `SolverMuJoCo`는 이 환경에서 사용할 수 없다

newton 1.6.0의 `[sim]` extra는 `mujoco-warp~=3.12.0`을 고정한다; 이 워크스페이스는
`MjWarpBackend`를 위해 **3.13.0**을 필요로 한다. 3.13.0에 대해 `SolverMuJoCo`를
생성하면 커널 컴파일 중에 죽는다(`convert_mjw_contacts_to_newton_kernel`의
`WarpCodegenError`). 한 환경이 두 고정 버전을 동시에 담을 수 없으며
`MjWarpBackend`가 이긴다 — 그것이 기본 백엔드다(spec 4.3).

그래서 어댑터는 Newton 자체의 축약좌표 articulated-body 솔버인
**`SolverFeatherstone`**로 스텝한다. 이것은 오히려 더 나은 결과일 수도 있다:
spec 17.2가 존재하는 이유는 같은 Task IR이 백엔드마다 다르게 동작해서는 안 되기
때문이며, 진짜로 다른 솔버야말로 그 비교를 해볼 가치가 있게 만든다. 아래에서
Newton이 `mujoco-cpu`와 어긋나는 것은 반올림이 아니라 물리다.

### `add_mjcf`는 액추에이터도 센서도 임포트하지 않는다

검증됨: `<motor>`를 담은 MJCF를 임포트한 뒤, `Model.actuators == []`이고
`Model.joint_target_mode == [0]`이다. 센서 배열은 아예 존재하지 않는다.

이것이 이 패킷에서 가장 중대한 발견인데, 솔깃한 대응 — 씬을 그냥 실행하고
0을 보고하는 것 — 이 정확히 매핑 리포트가 막기 위해 존재하는, 조용히 잘못된
로봇을 만드는 실패이기 때문이다. 대신 모든 액추에이터와 센서 feature는 Newton
컬럼에서 `severity: error`이므로, 액추에이션이 있는 씬은 **프로세스가 spawn되기도
전에, `load`에서 이름으로 지목되어 거부된다**(spec 14.4). `nu = 0`과
`nsensordata = 0`은 정직하며, 비어 있지 않은 벡터로 하는 `set_ctrl`은 오류다.

Spec17 행 `actuator.pd → controller`는 `approximated`로 남는다: *엔진*에는
`ControllerPD`와 `Control.joint_target_q`가 있다. 코드가 이제 긋는 구분은
엔진이 할 수 있는 것과 이 어댑터가 제공하는 것으로 검증된 것 사이의 구분이다 —
capability 행이 실행을 좌우하는 것이다.

### Contact가 연결되어 있지 않다

어댑터는 `contacts = None`으로 스텝한다. `newton.CollisionPipeline`은 존재하지만
사용되지 않는다. 기본 pyramidal cone을 차단하면 *모든* 씬을 차단하게 될 것이므로
(모든 MJCF는 이를 갖고 있다), 물체가 서로를 통과한다는 무딘 quirk와 함께
`approximated`로 선언되며, `ContactElliptic`, `ContactSoftParams`,
`ContactCondim6`, `ContactMesh`, `ContactHeightField`는 완전히 차단된다.

### `add_mjcf`가 임포트하는 것으로 검증됨

네 가지 MJCF 조인트 종류 모두, 조인트 한계(MuJoCo의 `angle="degree"` 기본값이
준수됨), 그리고 `armature` — 그래서 `joint.armature → armature`는 spec 17.2와
일치하는 **native, 검증됨**이다. 조인트 spring과 friction loss는 확인된 적이
없으며 `TODO(api-notes)`로 남는다.

### 좌표는 MuJoCo의 것이 아니라 Newton의 것이다

`Model.joint_q_start`는 조인트별로 `joint_q`의 주소를 매기며 fencepost로
종료된다. `Model.joint_label`은 경로 형태다(`model/worldbody/rod/hinge`); MJCF
이름은 마지막 `/` 세그먼트다. Newton transform은 `(pos xyz, quat xyzw)`다 —
이미 spec 3.1 순서다 — 하지만 **free joint의 일반화 좌표는 MuJoCo가 wxyz로 쓰는
것을 xyzw로 쓴다**, 그래서 부동 베이스(floating base)에 대해서는 `qpos`가
`mujoco-cpu`와 원소별로 비교 가능하지 않다. quirk로 선언되었다.

## accepted

- `newton_pendulum` **RAN**: `nq = nv = 1`, `n_envs = 2`가 독립적, 100틱이 유한,
  리셋됨.
- `newton_against_mujoco_cpu` **RAN**.
- `an_actuated_scene_is_refused_rather_than_run_unactuated`는 Newton이 설치되어
  있든 없든 통과한다 — 리포트가 게이트이며 무엇이 spawn되기도 전에 실행된다.
- `the_declaration_is_the_newton_column_of_the_mapping`은 capability 선언과
  spec 17.2 표가 feature 단위로 일치함을 단언하므로, 둘이 어긋날 수 없다.
- `es-physics-backend`에서 48개의 테스트가 통과한다.

### measured

1 kHz 진자, 수평으로 시작하는 막대, 200틱, RTX 4060 Laptop GPU, `newton 1.6.0`.

| 측정 항목 | Newton | `mjwarp`, 같은 씬 |
|---|---|---|
| `mujoco-cpu` 대비 `max \|dqpos\|` | **5.88e-4** | 7.41e-8 |
| `mujoco-cpu` 대비 `max \|dqvel\|` | 5.02e-3 | 7.76e-7 |
| 에너지 proxy 델타 | 2.06e-2 | 3.16e-6 |
| 1e-6 허용오차를 넘어서는 첫 틱 | **6** | 없음 |

두 GPU 백엔드 사이의 네 자릿수 차이이며, 이것이 기대되는 결과다: `mjwarp`는
MuJoCo의 솔버를 실행하며 MuJoCo와 일치한다, Newton은 Featherstone을 실행하며
일치하지 않는다. `< 1e-2` 단언은 둘이 같은 궤적 위에 머무른다는 것을 기록할
뿐이다; 이는 검증된 일치 허용오차가 **아니다**(spec 12.4).

## CLI note — 범위 밖, 후속 작업 필요

`BackendKind::Newton`과 `compare_backends`는 더 이상 아무것도 필요하지 않다:
백엔드는 스스로 `"newton"`이라 이름 붙이고 `BackendKind::from_name`이 컬럼을
해석한다. 되어 있지 **않은** 것은 `crates/es-physics-backend` 밖에 있는 `es`
CLI의 `es backend compare --task T --backends mjwarp,newton,mujoco-cpu`
(spec 17.2)다. `docs/packets/M1/CLI-backend-compare.md`를 소유한 누구든 자신의
백엔드-이름 switch에 `"newton" => Box::new(NewtonBackend::new())`를 추가해야
한다; `es_physics_backend::NewtonBackend`는 re-export되어 준비되어 있다.

## still open

- `newton.CollisionPipeline`을 연결하고 contact feature를 다시 선언한다.
- `add_mjcf`가 하지 않을 것이므로 MuJoCo 액추에이터를 `Control.joint_f` /
  `joint_target_q`에 명시적으로 매핑한다; 그래야만 비로소 `nu`가 0이 아니게 될
  수 있다.
- `mujoco-warp` 고정이 3.13을 허용하면 `SolverMuJoCo`를 다시 시도한다.
- 2 world를 넘는 배치 폭; `MAX_ENVS = 8192`는 선언되었을 뿐 측정되지 않았다.
- `docs/api-notes/newton.ko.md`와 `docs/packets/M2/W4-newton-backend.ko.md`는
  존재하지 않는다; 이 저장소는 이들에 대한 한국어 대응 문서를 유지한다. 이
  패킷의 선언된 범위 밖이다.
