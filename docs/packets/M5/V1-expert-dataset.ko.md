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
