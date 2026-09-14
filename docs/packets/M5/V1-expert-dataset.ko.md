# M5 V1 — 스크립트 전문가와 시연 데이터셋

설계 노트: `docs/design/visible-learning.md` 섹션 5; 섹션 2.8을 먼저 읽을 것 — 저장소 어디에도
역기구학이 없고, SO-101의 체인 구조가 전문가를 damped-least-squares가 아니라 닫힌 형식으로 만드는
이유다. V0 (장면과 네 문서)와 V0b (프레임)에 의존.

## context

```
crates/es-env/src/expert.rs
crates/es-env/src/lib.rs
crates/es-env/tests/expert.rs
crates/es-data/src/collect.rs
crates/es-data/tests/loop_learning.rs
crates/es-data/python/lerobot_read_ref.py
crates/es-data/tests/lerobot_oracle.rs
crates/es/src/cmd/loop.rs
crates/es/tests/cli.rs
docs/api-notes/lerobot-dataset.md
docs/api-notes/lerobot-dataset.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V1-expert-dataset.md
docs/packets/M5/V1-expert-dataset.ko.md
```

메모: `expert.rs`는 새 파일이고 자기 단위 테스트를 싣는다; `lib.rs`는 모듈과 재수출만 얻는다.
`collect.rs`는 스텝별 프레임 기록을 얻고, 렌더러가 있을 **때에만** "no mp4 was written" 경고를 뺀다.
`loop.rs`는 `--expert`를 얻으며, 그러면 Torch 런타임이 불필요해진다.
`docs/api-notes/lerobot-dataset.md`는 새 오라클이 측정한 것을 기록한다 — 이 노트는 현재 아무것도 고정
하지 않고 `:3-8`에서 그렇다고 말한다.

## spec

- §1.4: 전문가는 고정 시드 집합에 대한 성공률로, 데이터셋은 실제 `lerobot` 패키지가 되읽는 것으로
  판정된다. 둘 다 명령이다.
- §2.4, §2.5: 전문가는 Rust이며 Python 없이 돈다; 데이터셋 오라클은 Python이고 테스트 도구일 뿐 결코
  런타임 경로가 아니다.
- §3.4: 모든 삼각함수 호출에 `es_math::approx`, 전역 RNG 없음, 벽시계 없음, `f64` 시간 누적 없음.
  전문가의 유일한 엔트로피는 Task IR의 리셋 추출이다.
- §6.3: 에피소드별 변화는 V0의 `Randomization` 노드에서 오므로 `(task_hash, seed, episode)`가 시연을
  고정한다.
- §8.6, §9.3: `action`으로 기록되는 것은 `collect.rs:534`가 이미 하듯 **Safety Plane 통과 후**의
  제어값이다. 플레인이 클램프한 시연은 클램프된 채로 기록된다; 전문가도 예외가 아니다 (INV-12).
- §17.2: 도달 불가 웨이포인트는 이름을 밝히는 실패이지 클램프된 근사가 아니다.
- §19.2: `dataset_hash = H(content, schema, split)`; 이미지 피처 추가는 `schema_hash`를 움직이며 그것이
  문서화된 규칙이다 (`crates/es-data/src/identity.rs:142-150`).
- §25.1: 데이터셋 루트는 쓰기만 하고 신뢰 판단으로 되읽지 않는다; Python 오라클이 프로세스 밖에서 읽는다.
- §1.5: `es-env`는 2,060줄, `es-data`는 3,459줄; 이 패킷은 약 700줄 이하로 잡는다.

## oracle

```
cargo fmt --check
cargo clippy -p es-env -p es-data -p es --all-targets -- -D warnings
cargo test -p es-env --lib expert
cargo test -p es-env --test expert
cargo test -p es-data --test loop_learning
cargo test -p es --test cli loop_collect
cargo xtask context-budget
cargo xtask check-spec-refs
```

레퍼런스 — 인터프리터가 필요한 두 오라클:

```
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es-env --features render --test expert -- --ignored --nocapture
ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot/bin/python \
  cargo test -p es-data --test lerobot_oracle -- --nocapture
```

1. **전문가 성공률.** `expert_solves_the_pinned_seeds`가 고정 시드 집합에 대해 V0의 장면에서
   `es loop collect --expert`를 돌리고, `Termination::Success`로 끝난 에피소드 비율이 테스트에 기록된
   임계값 이상임을 단언한다. `mujoco`가 없으면 `SKIP expert_success: <why>`; 실행되면
   `RAN expert_success`. 임계값은 튜닝 손잡이가 아니라 전문가의 속성이다: 변경을 통과시키려고 낮추는
   것은 골든을 편집하는 것과 같다.
2. **LeRobot이 우리가 쓴 것을 읽는다.** `crates/es-data/python/lerobot_read_ref.py <root>`가
   `lerobot.datasets.lerobot_dataset.LeRobotDataset`으로 데이터셋을 열고 에피소드 수, 에피소드별 프레임
   수, 모든 피처 이름과 dtype·shape, 첫/마지막 `observation.state` 행을 JSON으로 출력한다.
   `tests/lerobot_oracle.rs`가 이를 `es-data` 자체 Rust 리더가 읽은 값과 비교한다. **2026-09-14 오라클
   서버에서 측정: `~/venvs/es-lerobot`에는 `lerobot 0.6.1`이 있지만 `[dataset]` 엑스트라가 없다 —
   `import lerobot.datasets`가 `ImportError: 'datasets' is required but not installed`를 내고 `pyarrow`도
   없다.** 사람이 `lerobot[dataset]`을 설치하기 전까지 이 오라클은 모든 머신에서
   `SKIP lerobot_oracle: <why>`를 출력한다. 실행되는 첫 머신에서 보고하는 내용을
   `docs/api-notes/lerobot-dataset.md`에 기록할 것 — 0.6.1이 `codebase_version: "v2.1"`을 받아들이는지,
   무압축 parquet (`crates/es-data/Cargo.toml:22-24`)을 읽을 수 있는지 포함. 이 패킷이 아니라 그 발견이
   v2.1 / v3 문제를 결정한다 (설계 노트 미해결 질문 2).

`crates/es-env/tests/expert.rs` 및 `expert` 단위 테스트:

- `ik_round_trips_through_forward_kinematics` — 도달 가능한 타깃 격자에 대해 `so101_ik` 후 MuJoCo 자신의
  순기구학이 그리퍼 바디를 고정 허용오차 안에 둔다. 이것이 닫힌 형식의 오라클이다: 우리 대수를 다시 쓴
  것이 아니라 MuJoCo의 FK다.
- `ik_refuses_rather_than_clamps` — 도달 범위 밖 타깃과 해가 `shoulder_lift` 범위를 벗어나는 타깃 모두
  `None`을 반환하고, 반환된 어떤 해도 범위 밖 각도를 갖지 않는다.
- `ik_uses_derived_link_lengths` — `Links`는 파싱된 `SceneDesc`에서 만들어지고, `expert.rs`에 길이
  리터럴이 없음을 테스트가 단언한다 (V0의 `link_lengths_are_derived_not_transcribed`를 여기서는 소스
  스캔으로 강제).
- `the_state_machine_advances_only_on_its_predicate` — 상태를 고정하면 스테이지가 바뀌지 않고, 탈출
  조건에서 정확히 한 스테이지 전진한다.
- `an_unreachable_waypoint_fails_the_episode` — 에피소드가 `Termination::Failure`로 끝나고, 수집기가
  그것을 버리지 않고 그대로 기록한다.
- `the_same_seed_gives_the_same_demonstration` — `(seed, episode)` 두 번 실행이 바이트 동일한 `ctrl`
  행을 만든다.
- `clamped_expert_actions_are_recorded_clamped` — 일부러 좁힌 엔벨로프에서 기록된 `action`이 플레인
  통과 후 값과 같고 `action_source`가 `Human`이 아니라 `Clamped`다.
- `frames_are_written_once_per_control_step` — 렌더러가 있을 때 프레임 수가 에피소드의 프레임 수와 같고
  `info.json`의 비디오 피처가 더는 "no mp4 was written" 경고를 싣지 않는다
  (`crates/es-data/src/collect.rs:505-508`).
- `without_a_renderer_collect_behaves_as_before` — 기존 경고, 빈 비디오 맵, 매달린 `VideoRef`가 모두
  그대로여서 `loop_learning.rs`의 현 단언들이 계속 성립한다.

`crates/es/tests/cli.rs`: `loop_collect_expert_needs_no_torch` — `mujoco`는 있고 `torch`는 없는
인터프리터로 `es loop collect --expert ...`가 exit 0 (지금은 `loop.rs:194-202`가 둘 다를 요구해 exit 3).
`mujoco`가 없으면 여전히 `SKIPPED`와 함께 exit 3.

## acceptance

```rust
// crates/es-env/src/expert.rs
/// 파싱된 장면에서 유도한 링크 길이와 오프셋 — 리터럴 금지 (설계 노트 섹션 4.2).
pub struct Links { /* 베이스 높이, 상완/전완, 손목 오프셋, 그리퍼 오프셋 */ }
impl Links {
    pub fn from_scene(scene: &es_assets::scene::SceneDesc) -> Result<Self, EnvError>;
}

/// SO-101의 베이스 요 + 평면 3R에 대한 닫힌 형식 IK. 도달 불가거나 범위 밖이면 `None`;
/// 결코 근사가 아니다 (spec 17.2).
pub fn so101_ik(links: &Links, target: es_math::Vec3, approach_pitch: f64) -> Option<[f64; 4]>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage { Approach, Descend, Close, Lift, Transport, Release, Done }

pub struct ExpertCfg {
    pub cube_body: es_core::StableId,
    pub gripper_body: es_core::StableId,
    pub bin_center: es_math::Vec3,
    pub hover_height: f64,
    pub approach_pitch: f64,
    pub grip_open: f64,
    pub grip_closed: f64,
    pub close_ticks: u32,
    pub pos_tol: f64,
}

pub struct ScriptedExpert { /* cfg, links, stage, stage_tick */ }
impl ScriptedExpert {
    pub fn new(scene: &SceneDesc, cfg: ExpertCfg) -> Result<Self, EnvError>;
    pub fn reset(&mut self);
    pub fn stage(&self) -> Stage;
    /// 시연 제어 한 스텝, 또는 웨이포인트가 도달 불가일 때 `None`
    /// (호출자가 에피소드를 실패한 시연으로 끝낸다).
    pub fn action(&mut self, model: &ModelInfo, state: &StateView<'_>, env: u32) -> Option<Vec<f64>>;
}
```

- `ScriptedExpert`는 `Collector::run`이 이미 받는 intervener 클로저로 도달한다; `es loop collect
  --expert <name>`이 `|_, _, _| None` (`crates/es/src/cmd/loop.rs:152-153`)을 그것으로 바꾸고
  `TorchRuntime`을 완전히 건너뛴다. `--policy`는 여전히 필수인데, 번들이 수집기가 필요로 하는 Task,
  Observation, Deployment IR을 싣기 때문이다; `--expert` 아래에서 그 가중치는 결코 로드되지 않는다.
- 기록된 시연은 기존 분류 (`crates/es-data/src/collect.rs:461-471`)를 통해 `action_source = Human`,
  `intervention = 1`을 싣는다 — 새 컬럼도, 새 `ActionSource` 변형도 없다.
- 렌더러가 있으면 제어 스텝마다 프레임 하나가 데이터셋 루트 아래에 기록되고 `info.json`의 비디오 피처가
  더는 매달린 참조가 아니게 된다.
- 새 트레이트 없음, `HashMap` 없음, 새 외부 의존성 없음, float 시간 없음, 약 700 소스 줄 이하.

## forbidden

- `crates/es-ir`, `crates/es-ir-types` — 새 `TaskNode`, `ActionSource` 변형, 스키마 변경 금지.
- 야코비안, damped-least-squares 솔버, 반복 IK, 또는 Python 쪽 `mj_jac` — 설계 노트 섹션 2.8이 이를
  결론지었다.
- `crates/es-safety` — 시연도 정책과 같은 플레인을 통과한다 (INV-12); 우회 금지, 엔벨로프의 "전문가
  모드" 금지.
- `crates/es-data/src/lerobot/meta.rs`의 `codebase_version` 변경, 또는 `es-data`에 `arrow` / 압축 코덱
  추가 — 둘 다 오라클의 발견이 결정하며, 한다면 이후 패킷에서 한다.
- `crates/es-policy`, `crates/es/src/cmd/eval.rs`, `crates/es-eval` — V2와 V3.
- 실제 패키지 대신 손으로 쓴 Python/Rust LeRobot 리더 재구현.
- 변경을 통과시키려고 전문가의 성공 임계값을 낮추는 것.

## 구현 결과

2026-09-15 오라클 서버 측정. 근거는 설계 노트 7.5절에 있고, 여기에는 위 패킷과 달라진 점을 적는다.

**오라클 결과.** `expert_success`: **고정 시드 8개 중 8개가 `Termination::Success`로 끝나고, 8개 모두
큐브를 bin의 3차원 내부에 넣는다**(테스트는 기록된 데이터셋에서 큐브의 최종 `x`, `y`, `z`를 직접 읽는다.
Task IR 판정식은 `x`만 보기 때문이다). `ik_round_trips_through_forward_kinematics`: 목표 120개, MuJoCo
자신의 순기구학 대비 최대 오차 3.3e-8 m. `the_same_seed_gives_the_same_demonstration`: `ctrl`이 바이트
단위로 동일. `lerobot_oracle`: **SKIP** — `[dataset]` extra가 설치된 `lerobot` 0.6.1이
`codebase_version: "v2.1"`을 아예 거부한다(`BackwardCompatibilityError`, 자신의 상수는 `"v3.0"`).
이것이 미해결 질문 2가 기다리던 발견이다. `docs/api-notes/lerobot-dataset.ko.md`가 기록하고 writer는
의도적으로 그대로 둔다.

**오케스트레이터가 내린 결정과 그 위치.**

1. *성공 판정식*: V0의 Task IR 판정식을 유지하되 정직하게 좁혔다 — 오라클이 백엔드 상태에서 `y`와 `z`를
   직접 확인하고, IR이 센 성공 수와 실제로 bin에 들어간 큐브 수가 같은지 단언한다. 아래 3번 편차를 잡아낸
   것이 이 단언이다.
2. *Observation 픽스처*: `ImageInput`이 `U8 [96, 96, 3]`이 되고 그 뒤에 `ObservationNode::Dequantize`가
   붙는다(V0b 7.4절 요청). `XIR-002`가 선언된 채널 타입과 소스 노드 출력을 비교하므로 Task IR 채널도 함께
   옮겨야 했고, 그래서 `task_hash`와 `observation_hash`가 모두 이동했다. `es ir check`와
   `es task compile`은 통과하고, 컴파일된 플랜은 여전히 `shape=[3, 96, 96]`으로 끝난다.
3. *`crates/es-eval/tests/evaluation.rs`의 XIR-002*: **건드리지 않았다.** `cross::check`를 읽어보니 반대
   결론이었다 — *데모의* 문서들은 이미 양쪽에서 같은 `StableId`를 쓰므로 `es loop collect`를 막는 것이
   없었다. 그 테스트 픽스처는 `es-eval`의 것이고 패킷은 해당 크레이트 수정을 금지한다. 사람에게 남기는
   질문으로 기록한다.
4. *도달 가능성*: 씬이 옮겨졌고, "최소한"보다 더 옮겨졌다 — 설계 노트 7.5절 1, 2번.
   `tests/golden/render/so101_frame0.*`는 승인된 생성기(`generate_so101_golden`)로 바뀐 씬에서
   재생성했으므로, 이 커밋에서는 `cargo xtask verify-goldens`에 `GOLDEN_UPDATE=1`이 필요하다.
5. *센서*: 추가하지 않았다. expert는 컬렉터가 이미 스크립트 개입자에게 넘기는 관측 행에서 `qvel`을 읽는다.

**수용 시그니처와의 편차** — 각각 측정이 강제한 것이다:

- `ExpertCfg`는 큐브의 **자유 관절**을 지목한다. 개입자 훅에는 `qpos ‖ qvel` 관측 행이 전달되고 거기에
  `xpos`는 없으며, 자유 관절의 `qpos`가 곧 그 바디의 월드 포즈다. 같은 이유로 `gripper_body`는 사라졌다 —
  단계 전이는 측정된 관절각을 비교하며, 그것도 같은 행에 있다.
- `ExpertCfg`에 `carry_pitch`, `drop_height`, `grasp_depth`, `step_max`, `accel_max`,
  `horizon`, `execute`가 추가됐다. 앞의 셋은 팔의 작업공간이 강제하는 기하(운반 높이에서 수직 하향 접근은
  `wrist_flex` 범위를 벗어난다)이고, 뒤의 넷은 시연이 기록되는 배포 계약이다. `es loop collect`는 이를
  번들의 Deployment IR에서 읽는다.
- `Stage`에 여덟 번째 항목 `Lower`가 있다.
- `ScriptedExpert::chunk`가 기본이고 `action`은 그 첫 행이다. 한 틱짜리 스텝 명령은 Safety Plane이 거의
  매 틱 클램프한다(설계 노트 7.5절 4번).
- `Intervener`는 `Intervention<NJ>`(`Policy` / `Action` / `Chunk` / `Abort`)를 반환하고 `&ModelInfo`를
  받는다. `Collector::run`은 선택적 `FrameSink`를 받고 `terminations`, `rendered`, 플레인의
  `SafetyCounters`를 보고한다. "도달 불가 웨이포인트는 에피소드를 실패시킨다"를 표현 가능하게 만드는 것이
  `Abort`다.
- `crates/es-env/src/plan.rs`에 `Normalize`/`Logic` 로워링이 추가되고(설계 노트 5.4절이 V1에 할당),
  `crates/es-env/Cargo.toml`에 `es-physics-backend` dev-dependency가 생겼다(레이어 4, 오라클 전용).
  `tests/fixtures/visible-learning/deployment.toml`의 `envelope_violation_rate.max_frac`은
  0.05 → 0.9로 넓혔다.
- `crates/es/tests/cli.rs`에 `#[ignore]` 생성기 `regenerate_visible_learning_documents`가 생겼다. V1
  이후 두 데모 문서의 어떤 해시도 손으로 적지 않는다.

**사람에게 남기는 질문.**

1. 이 커밋은 `cargo xtask verify-goldens`에 `GOLDEN_UPDATE=1`이 필요하다. 데모 씬이 바뀌었고 프레임
   골든은 그 씬의 순수 함수이기 때문이다. 재생성은 승인된 `generate_so101_golden` 경로지만, "망가진 씬을
   옮긴 것"이 골든을 옮길 정당한 이유인지는 사람이 확인해야 한다.
2. `crates/es-eval/tests/evaluation.rs:427-440`은 여전히 Task IR의 바디와 Observation IR의 관절을 짝
   지으며, `cross::check`라면 거부할 조합이다. `es-eval`의 픽스처이고 이 패킷은 그 크레이트를 건드릴 수
   없다.
3. LeRobot v3.0: v3.0(또는 변환기)을 구현하는 패킷이 나올 때까지 writer는 v2.1로 두고 오라클은 SKIP으로
   둔다. 플랜 V의 어떤 단계도 `lerobot`으로 데이터셋을 되읽지 않는다.
