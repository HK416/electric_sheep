<!-- Korean translation of docs/packets/M1/W1-mjwarp-adapter.md. The English file is the working copy; regenerate this when it changes. -->

# W1 — MJWarp 백엔드 어댑터 + 백엔드 의미 매핑(semantic-mapping) 리포트

Spec: spec 17.2 (백엔드 의미 매핑과 `es backend compare` — "새로운 핵심 작업"), spec 17.1 / spec
17.3 (백엔드 계층, 결정성 계층: GPU 백엔드는 계층 2 또는 3을 선언하며 계층 1은 오직
`mujoco-cpu`만 주장할 수 있다), spec 4.3 (백엔드가 기본 경로다), spec 14.4 (`severity: error`인
미매핑 항목은 실행을 차단한다), spec 11.6 (컴파일 타임 능력(capability) 검사), spec 12.1
(MJWarp는 배치 GPU 경로다), spec 3.5 (결정성 계층), spec 1.7 (지어낸 API 이름은 에이전트의
상시적 실패 모드다), spec 18.1 / spec 18.5 (정수 틱, 보고되는 발산). M1 Wave 1. Review class B.

## context (범위)

```
crates/es-physics-backend/src/mapping.rs
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/src/lib.rs          (두 개의 `pub mod` 줄과 재익스포트)
crates/es-physics-backend/python/mjwarp_ref.py
docs/api-notes/mujoco-warp.md
docs/packets/M1/W1-mjwarp-adapter.md
```

## spec (사양)

### `mapping.rs` — spec 17.2 표

- `--backends` 표기를 갖는 `BackendKind { MuJoCoCpu, MjWarp, Newton, PhysX }`, 그리고
  `Capabilities::name`을 다시 컬럼으로 매핑해 주는 `from_name`.
- `TaskFeature = Spec17(Spec17Row) | Capability(Feature)`: spec 17.2 표가 명시하는 다섯 행
  (`actuator.pd`, `contact.friction_cone`, `contact.soft_params`, `joint.armature`,
  `sensor.contact_force`)과 `es-physics-core`의 모든 `Feature` variant를 더한 것. 두 종류를
  분리한 이유는 `actuator.pd`가 액추에이터 종류가 아니라 게인(gain) 한 쌍이고,
  `sensor.contact_force`는 두 센서 종류에 던지는 질문 하나이기 때문이다.
- `lookup(TaskFeature, BackendKind) -> Mapping { status, severity }`는 구성상 전역(total)이다;
  `SemanticMapping`은 순회를 위해 데카르트 곱(cross product)을 구체화(materialise)한다.
  `Status = Native(note) | Approximated(note) | Unsupported(note)`,
  `Severity = Info | Warning | Error`.
- 컬럼: `mujoco-cpu`는 `crate::mujoco::capabilities()`에서 도출되고, `mjwarp`는 같은 집합에서
  spec 17.2가 더 좁게 고정한 부분을 뺀 것이므로, 선언과 매핑이 서로 어긋날 수 없다. `newton`과
  `physx`는 spec 17.2의 셀들을 정확히 그대로 담고 있으며, 그 외 모든 행은 `Warning`을 동반한
  `Unsupported("TODO(api-notes): unverified against the engine")`다 — **매핑을 native로
  추측하는 일은 절대 없다**(spec 1.7).
- `mapping_report(&SceneDesc, BackendKind) -> MappingReport { backend, rows, blocked }`는 씬을
  스캔하여 — 액추에이터 종류, 센서 종류, `armature != 0`, friction cone, 기본값이 아닌
  `solref` / `solimp` — 사용된 각 feature의 매핑을 나열한다. feature 행은
  `Requirements::from_scene`에서 오므로, 리포트와 spec 11.6 능력(capability) 검사는 같은 씬을
  보게 된다. `blocked`는 `Unsupported`이면서 동시에 `severity: error`인 모든 행이다(spec 14.4).
  `Display`는 `es backend compare`가 헤더에 출력하는 고정폭 표다.
- `compare_backends(a, b, &SceneDesc, ctrl_seq, n_ticks) -> Result<CompareReport, PhysicsError>`는
  두 백엔드를 동일한 리셋 상태에서 동일한(순환되는) 제어 시퀀스로 실행하고, spec 3.5 계층 3
  지표를 산출한다: `max |dqpos|`, `max |dqvel|`, 에너지 드리프트 **근사값(proxy)**
  (`sum ½ qvel²` — 운동에너지만 계산하며 질량 행렬도 포텐셜 항도 없음; `qvel`만 있으면 되므로
  백엔드 간 비교가 가능한 proxy로 문서화됨), `|dqpos|`가 `DIVERGENCE_TOL = 1e-6`을 넘어서는 첫
  틱, 양쪽의 선언된 결정성 계층, 그리고 양쪽의 매핑 리포트. `Display`는 표다. `es backend
  compare` CLI 연결은 별도 패킷이다.

### `mjwarp.rs` — `MjWarpBackend`

- `python/mjwarp_ref.py`를 통해 `PhysicsBackend` 뒤에 놓인 MuJoCo Warp. `include_str!`로
  임베드되어 `python -c`로 실행되며, `mujoco_ref.py`와 동일한 개행 구분(line-delimited)
  JSON(`load` / `reset` / `set_ctrl` / `step` / `state` / `set_state` / `quit`)을 사용하므로
  `proc.rs`의 Rust 요청·응답 타입을 그대로 재사용한다. `load`는 `n_envs`를 실어 나르며,
  스크립트는 이를 `put_data(..., nworld=n_envs)`로 전달한다 — CPU 어댑터의 에뮬레이션 루프가
  아니라 한 디바이스 위의 진짜 배치다(spec 12.1).
- `is_available()`는 `python -c "import mujoco_warp, warp"`를 실행한다; `ES_PYTHON`이
  인터프리터를 오버라이드한다.
- 선언된 capabilities: **결정성 계층 2(`CrossBackend`)이며 계층 1은 절대 아님**(spec 17.3),
  `gpu_resident: true`, `max_envs: 8192`(측정치가 아니라 선언된 상한 — spec 12.4),
  `float: F32`(디바이스 배열이 float32이므로 CPU 오라클과의 일치는 허용오차임),
  `supports_reset_subset`과 `supports_state_get_set`. 이 feature 집합은 매핑 표의 MJWarp 컬럼
  *그 자체*이며, 테스트가 feature 단위로 이를 단언(assert)한다.
- `load()`는 **아무것도 spawn하기 전에** `mapping_report`를 실행하고, 차단된(blocked) 씬을
  `PhysicsError::Unsupported(report.to_string())`로 거부한다(spec 14.4): 오류 메시지 자체가 그
  표이며 해당 행들을 지목한다. feature는 리포트의 몫이므로, 뒤이은 capability 검사는 요청의
  실행 수준(run-level) 형태(배치 크기, spec 11.6)만을 다룬다.
- `docs/api-notes/mujoco-warp.md`는 버전을 고정하고 **모든 `mujoco_warp` 호출을 미검증
  (unverified)으로 표시**하며, 첫 GPU 실행 체크리스트를 담고 있다.

### 패킷 브리프에서 벗어난 부분과 그 이유

- `Status::Native`와 `Status::Unsupported`도 `Approximated`와 마찬가지로 note를 갖는다. 설명
  없는 `Unsupported` 셀은 정확히 spec 17.2가 막고자 하는 것이며, 브리프 자체의
  `TODO(api-notes)` 요구사항도 어딘가에 담길 곳이 필요하다.
- `compare_backends`는 `CompareReport`가 아니라 `Result<CompareReport, _>`를 반환한다: load나
  step의 실패는 비교 결과가 아니며, 구멍 난 리포트 속으로 삼켜져서는 안 된다.
- `MappingReport`는 자신의 `BackendKind`를 갖고 있으므로, 렌더링된 표가 해당 백엔드를 명시할
  수 있다.
- `proc.rs::Process`는 `mujoco_ref.py`를 하드코딩하고 있고 이 패킷은 이를 수정할 수 없으므로,
  spawn / call / drop 삼종 세트가 `ponytail:` 주석 뒤에서 `mjwarp.rs`에 중복되어 있다. 세 번째
  프로세스 외부(out-of-process) 백엔드가 등장하면 둘을 하나의 `Process::spawn_with(SCRIPT)`로
  합칠 것.
- `es-physics-core`에 `Capabilities`나 `Feature`를 추가할 필요는 없었다.

## oracle (오라클)

```
cargo fmt -p es-physics-backend --check
cargo clippy -p es-physics-backend --all-targets -- -D warnings
cargo test -p es-physics-backend
cargo xtask layering && cargo xtask check-spec-refs && cargo xtask context-budget
```

두 라이브 테스트를 제외한 모든 것은 GPU도 Python도 없이 실행된다. `mjwarp_pendulum`과
`mjwarp_against_mujoco_cpu`는 `is_available()`가 실패하면 `SKIP <name>: <reason>`을 출력하고
통과(pass)한다.

## acceptance (수용 기준)

- 모든 `(TaskFeature, BackendKind)` 쌍은 행을 가지며, 모든 행은 비어 있지 않은 note를 가진다.
  그리고 spec 17.2의 다섯 행은 정확한 상태를 셀 단위로 단언(assert)한다 — spec이 차단(block)이
  아니라 *warning*으로 unsupported 처리하는 PhysX의 armature를 포함해서.
- 확인되지 않은 행은 `Unsupported` + `Warning` + `TODO(api-notes)`이며 차단하지 않는다.
- `tests/fixtures/mjcf/pendulum.xml`에 대한 `mapping_report`는 MJWarp에서 차단되고(픽스처가
  elliptic cone을 요구함) `mujoco-cpu`에서는 차단되지 않는다; `actuated.xml`에서는
  `actuator.pd`, `sensor.contact_force`, `Tendon`을 보고하며, MJWarp에서는 차단되고 Newton에서는
  경고만 한다. 평범한 hinge 씬은 어느 백엔드에서도 차단되지 않는다.
- elliptic 씬에 대한 `MjWarpBackend::load`는 `PhysicsError::Unsupported`로 실패하며, 그
  메시지에는 렌더링된 표와 `ContactElliptic`, `blocked: yes`가 포함된다; `MAX_ENVS`를 초과하는
  배치는 `Unsupported::BatchSize`로 실패한다; `load` 이전의 호출은 `NotLoaded`다.
- 동일한 가짜(fake) 적분기 두 개에 대한 `compare_backends`는 발산이 없고 델타가 0임을
  보고한다; 한쪽을 `1e-9`만큼 교란하면 발산 틱(해당 적분기 기준 tick 44)과 0이 아닌
  에너지-proxy 델타를 찾아낸다.
- `mjwarp_ref.py` 프로토콜은 미리 준비된(canned) JSON에 대해 — `load`, 배치된 `state`,
  non-finite 월드를 포함한 `step`, `Ack`, Python 쪽 오류, 형식이 잘못된(malformed) 줄 — Python
  프로세스 없이도 왕복(round-trip)한다.

## forbidden (금지)

`context` 밖의 모든 파일. `crates/es`, `crates/es-policy`, `crates/es-sensor`,
`crates/es-actuator`(다른 패킷들이 진행 중). `mujoco.rs`, `proc.rs`, `mjcf_out.rs` — 재사용만
하며 절대 수정하지 않는다. 루트 `Cargo.toml`. 새로운 trait(`INV-17`: 일곱 개의 확장 지점은
고정되어 있다). `HashMap` / `HashSet`(spec 3.4). GPU 백엔드에서 결정성 계층 1을 선언하는
것(spec 17.3). 아무도 실행해 본 적 없는 엔진에 대해 native 매핑을 추측하는 것(spec 1.7). 물리를
구현하는 것: 이 crate는 엔진에 매핑할 뿐 엔진 자체가 아니다(spec 17.4는 M4+).
