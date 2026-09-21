<!-- Korean translation of docs/design/python-builder.md. The English file is the working copy; regenerate this when it changes. -->

# Python authoring builder (M2 W6, spec 14.2)

## Shape

세 층, 아래로 갈수록 각각 더 얇다:

```
python/es/*.py            fluent한 spec 14.2 표면 (Task, Observation, Learning, Deployment,
                           Expr 연산자 오버로드, es.math) — 순수 Python, node-kind 로직 없음
        |  pyo3 경계를 건너는 JSON 텍스트
crates/es-py/src/pybind.rs "es_native" pyo3 모듈 — Pythonic한 메서드 이름(`.reward`,
                           `.terminate`, `Learning.act`, `.save`), 각각 core로 전달됨
        |  평범한 Rust 호출, Python 관여 없음
crates/es-py/src/builder.rs 언어 중립적인 core — TaskBuilder / ObservationBuilder / LearningBuilder /
                           DeploymentBuilder: `add(kind, params)`, `connect`, `declare_*`,
                           `build`, `save_toml`, 기존 `es-ir` 팩토리/그래프 위에서
```

가장 아래 층만이 이 패킷의 오라클(`cargo test -p es-py`)이 Python 없이
실행하는 층이다; 이 층에는 node-kind별 코드가 전혀 없다 —
`TaskBuilder::add`/`LearningBuilder::add`는 `TaskNodeRegistry`/`LearningNodeRegistry`
(두 개의 `INV-17` 팩토리 확장 지점)를 통해 라우팅되고,
`ObservationBuilder::add`는 `factory::deserialize_tagged`가 TOML을 다시
태그하는 것과 같은 방식으로 JSON을 다시 태그한다(`ObservationNode`에는
팩토리가 없다 — 7개의 확장 지점 중 하나가 아니다 — 그래서 이는 kind마다가
아니라 한 번 손으로 적용된 같은 serde 트릭이다).

## Deployment에는 왜 `add`/`connect`가 없는가

`DeploymentIr`(`es-ir/src/deployment.rs`)는 그래프가 아니라 평범한 구조체다 —
safety envelope, watchdog, fallback, rate다(`serial.rs`: "`NodeId` 맵이 없으므로
미러도 없다"). `DeploymentBuilder`는 애초에 노드 형태가 아니었던 것에
`add(kind, params)` shape를 강제하는 대신 직접적인 setter
(`envelope`, `watchdog`, `fallback`, `set_robot`, ...)다.

## pyo3 경계에서 파라미터가 `toml::Value`가 아니라 JSON인 이유

`es-ir`의 노드 팩토리는 `&toml::Value`를 요구하므로
(`TaskNodeFactory::create`), `builder::json_to_toml`은 사용 지점에서 한 번
변환한다. `builder.rs`와 `pybind.rs` 둘 다의 *공개* 표면은 대신
`serde_json::Value` / JSON 텍스트를 말한다: Python의 `json` 모듈이 자연스러운
다리이며(dict에 대한 `json.dumps`), 이를 재사용하면 `crates/es-py`는 손으로
작성한 `PyObject` 순회기도, `pyo3` 자체를 넘어서는 추가 의존성도 결코 필요
없게 된다(`docs/api-notes/pyo3.md` 참고).

기억해 둘 만한 비대칭 하나: JSON의 `null`에는 TOML 대응물이 없다.
`json_to_toml`은 오류를 내는 대신 `null` 값을 가진 테이블 키를 버리는데 —
이는 정확히 `Option::None`(예: `PortType::image`)이 이미 TOML로 직렬화되는
방식이며 — 테이블 안에 있지 않은 `null`(최상위, 또는 배열 원소)만을 거부하는데,
이는 실제 노드의 필드가 만들어내는 것이 결코 아니다.

## 타입 추론은 어디에 사는가

Task/Observation 노드 포트는 노드 *kind*로 고정된다(`TaskNode::outputs`,
`ObservationNode`의 `Io`); Learning 노드 포트는 노드 자신이 선언한
`inputs: Vec<TensorPort>` 필드이며, `LearningNode::input_ports`가 이를 그대로
되돌려준다. 어느 쪽이든, 그래프 에디터의 "여기에 무엇을 꽂을 수 있는가"가
그래야 하는 것과 마찬가지로, 다음 노드를 연결하기 *전에* 무언가가 각 노드의
결과 `PortType`을 알아야 한다. `python/es/builder.py`의 `Expr` 클래스와 몇 개의
작은 공식(`_feature_ty`, `_chunk_ty`)이, 관련된 `es-ir` private 헬퍼
(`task.rs`의 `f32v`/`boolish`/`flag`, `learning.rs`의
`feature`/`chunk`/`policy_ty`)를 미러링해 이를 수행한다 — 각각은 자신이
동기 상태를 유지해야 하는 Rust 함수와 함께 주석이 달려 있다. 이는 우회책이
아니라 평범한 프론트엔드 작업이다: LLM 생성기나 그래프 에디터도 동일한
추론이 필요할 것이다.

## rollout 바인딩 (M8 S4a, spec 13.4)

`es-py`는 별개의 두 번째 절반을 얻었다: `es_native.Rollout`. `train_ppo.py`가
자기만의 시뮬레이터가 아니라 우리 `Env`와 Safety Plane을 스텝하기 위한 것이다
(`rl-continuation.md` 규칙 2). 계층은 builder의 것과 같되 한 단 짧다 — fluent한
Python 표면이 없는데, 트레이너는 저작 도구가 아니기 때문이다:

```
python/es/train_ppo.py     트레이너 (S4b) — torch, GAE, 옵티마이저
        |  pyo3 경계를 넘는 float 리스트
crates/es-py/src/pybind.rs `Rollout` pyclass — 고정된 (n_joints, horizon) 디스패치 표
        |  평범한 Rust 호출, Python 없음
crates/es-py/src/rollout.rs `Rollout<NJ, H>` — Env + env당 SafetyPlane 하나, CpuPlan 하나
```

| 메서드 | 무엇인가 |
|---|---|
| `Rollout(task, observation, deployment, scene_xml, seed, n_envs)` | 네 문서를 **텍스트**로 받는다; `BatchDomains::single_env_at`의 제어 주기에 배치만 `n_envs`로 둔 `Env<MuJoCoCpuBackend>`를 만든다 |
| `reset(envs=None)` | `Env::reset` 후 해당 plane에 `begin_episode`, 해당 링에 `CpuPlan::reset` |
| `observe() -> {port: [n_envs * dim]}` | env별 `es_eval::runner::capture` + `CpuPlan::run`, row-major |
| `act([n_envs * nu]) -> (executed, events, rewards, dones)` | env별 `observe_state → heartbeat → validate` 후 plane의 출력으로 `Env::step` 한 번; `events`는 `EventSet::bits()` |
| `model() -> dict` | `nq`, `nv`, `nu`, `n_envs`, 인덱스 순서의 actuator/joint 이름, `ctrlrange` |
| `tick() -> int` | 제어 틱 — plane의 시계이며 `Env::tick`의 시뮬레이션 틱이 아니다 |
| `metrics() -> dict` | spec 12.4의 아홉 필드와 `chunk_underrun_rate`, 측정되지 않은 곳은 `None` |

적어 둘 만한 결정 세 가지:

- **numpy가 아니라 리스트.** 경계는 양방향으로 Python 리스트를 나르고 `es-py`는
  배열 의존성을 얻지 않는다. `train_ppo.py`는 이미 torch를 import하며 자기 쪽에서
  변환한다; layer 11 크레이트에 `numpy`를 넣는 것은 호출자 하나가
  `torch.tensor(...)` 한 번을 아끼자고 런타임에 두 번째 배열 ABI를 두는 일이다.
- **두 번째 관측 캡처는 없다.** `es_eval::runner`의 `capture` / `input_sources` /
  `joint_state`가 `pub`이 되었고(로직 변경 없음) 바인딩이 이를 호출한다. 캡처를
  여기서 다시 쓰는 것이 바로 트레이너와 평가가 하나의 상태에서 서로 다른 두 관측을
  읽기 시작하는 방식이다. 추가된 것은 env별 `StateView` 하나뿐인데, 그 헬퍼들이
  단일 env 평가 경로를 위해 쓰였기에 env 0을 읽기 때문이다 — 덕분에 `n`개의 배치가
  그대로 그 헬퍼에 도달한다. 이미지 입력은 여전히 렌더러가 필요하고 `es-py`는 이를
  링크하지 않으므로 `capture`가 이름을 들어 거절한다;
  `crates/es-py/tests/fixtures/observation-state.toml`은 데모 관측에서 카메라 가지를
  제거한 것으로, 데모 문서에서 생성되며 손으로 타이핑되지 않는다.
- **타입 소거된 plane이 아니라 고정된 `(n_joints, horizon)` 표.**
  `SafetyPlane<NJ, H>`는 const 제네릭이고 문서는 실행 시점에야 그 숫자를 드러낸다;
  pyclass는 이 저장소가 쓰는 쌍들을 `es eval run`의 `dispatch_nj_h!`와 똑같이 나열한다.
  트레이트 객체는 여덟 번째 확장점이 된다(`INV-17`). horizon 1은 트레이너 쪽 의미다 —
  제어 틱마다 새 `seq`로 한 행, 청크 버퍼도 선언된 지연도 없다
  (`rl-continuation.md` 3절) — 반면 `H`는 배포 문서 자신의 값으로 남는데,
  `SafetyPlane::from_ir`이 자기 폭이 아닌 envelope을 거절하기 때문이다.

오라클은 하나의 트레이스를 두 번 구동한다: `cargo test -p es-py
rollout_matches_es_eval_loop -- --ignored`가 스크립트된 제어 100 스텝을 `Rollout`과,
`es_eval::runner::run_episode`가 하는 순서 그대로 손으로 쓴 `Env` + `CpuPlan` +
`SafetyPlane` 루프 양쪽에 흘려보내고 둘 모두를
`tests/golden/rollout/so101_100steps.json`에 고정한다;
`python -m es.selfcheck --env`는 같은 100 스텝을 pyclass로 구동해 같은 골든과
비교하므로, Rust 구조체와 Python 클래스가 조용히 어긋날 수 없다.

## 알려진 공백 (M2 W6 범위, M0/M1 회귀가 아니다)

- `python/es/examples/pick_place.py`는 spec 14.2 스니펫의 노드 *shape*를
  충실히 재현한다; 몇몇 호출 지점은 스키마가 오늘 실제로 갖고 있는 필드에
  맞춰 조정된다(인라인으로 문서화됨 — 예: 이 builder에는 Task IR 산술에
  대한 상수 folding 경로가 없으므로 `Reward`의 부호는 `weight`로 옮겨가며,
  `VisionEncoder.pretrained`는 평범한 `bool`이므로
  `pretrained="imagenet"`은 참으로 읽힌다). 이 조정 중 어느 것도 `es-py`의
  일반 core를 건드리지 않는다.
- Cross-IR 검사(`es-ir::cross`, spec 11.1)는 이 builder가 아니라 컴파일러의
  일이다; 여기서는 이를 클라이언트 쪽에서 재현하지 않는다.
- Evaluation IR에는 builder가 없다 — 이 패킷의 범위 밖이다(spec 14.2 자체의
  스니펫도 이를 다루지 않는다).
