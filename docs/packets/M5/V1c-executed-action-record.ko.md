# M5 V1c — 시연은 실행된 액션을 기록한다

설계 노트: `docs/design/visible-learning.md` 7.5–7.9절과 **미해결 질문 12**. 먼저 읽을 것.
V1(수집기, 스크립트 전문가), V1b(v3.0 내보내기), V2/V2b(로워링, 학습 스크립트, 베이크 세트),
V3(이 패킷이 다시 돌리는 측정)에 의존한다. V2b가 측정했지만 건드리는 것이 금지되어 있던 결함을
고친다.

## 결함

같은 Safety Plane을 구동하는 두 코드 경로가, 엔벌로프를 **무엇에 대해** 재는지에 대해 서로
다르게 동작한다.

`es_eval::runner`는 — 그리고 `es_ros2::hil`과 `es_runtime_embedded`도 — 매 `validate` 직전에
측정된 관절 상태로 `SafetyPlane::observe_state(q, qd)`를 호출하며, 이는 `last_safe`,
`prev_safe`, `vel`, `prev_vel`을 다시 심는다. 그러면 속도 단계는 `(cmd − qpos) / dt`를 읽고,
엔벌로프는 **추종 오차**의 상한이 된다. 50 Hz에 `velocity_max = 3.0`이면 명령은 자기가 명령하는
관절을 `0.0600` rad까지만 앞설 수 있다.

`es_data::collect::Collector::run`은 그것을 한 번도 호출하지 않았다. 그쪽에서 `last_safe`는
*직전 명령*이므로 엔벌로프는 명령 대 명령의 이동을 제한하고, `ScriptedExpert`는 정확히 거기에
맞춰 페이스를 잡는다(`step_max = 0.054`). 그러면서 그 안티와인드업은 명령이 측정 관절을
`4 × step_max = 0.216` rad 앞서도록 **일부러** 허용한다. 추론 지연을 가진 위치 서보에 필요한
값이기 때문이다.

그래서 시연의 `action` 컬럼은 자기 `qpos`를 중앙값 **0.2413** rad 앞선다 — 추론 시점의 엔벌로프가
허용하는 값의 네 배다. ACT는 그것을 충실히 모방하고(V2b는 청크의 첫 액션이 기록된 액션과
0.02–0.16 rad 이내임을 측정했다), 모든 스텝이 속도·가속도·레이트 클램프에 걸리며,
`envelope_violation_rate`는 `1.0000`, 위반율 워치독은 스텝의 약 1/10에서 `hold_position`으로
래치되고, **86,400 스텝 중 단 하나도 `ActionSource::Policy`가 아니었다**. 시연과 엔벌로프는 서로
대조된 적이 없었다 — 그것이 미해결 질문 12이며, 여기서는 세 가지 선택지가 아니라 하나의 근본
원인으로 진술한다.

같은 "남아 있는 플레인 상태"가 두 번째 결함이다. `Collector::run`은 에피소드 사이에 env와 청크
버퍼와 e-stop 래치를 리셋하지만 플레인의 홀드 타깃·속도·레이트 이력은 리셋하지 않는다. 그래서
에피소드 1은 플레인이 "팔은 아직 에피소드 0의 마지막 명령 자리에 있다"고 믿는 상태에서 시작한다.
`es loop collect --episodes N`이 에피소드 0만 풀었던 이유가 이것이다(설계 노트 7.6절 발견 5).
`reset`이 이미 올바랐던 스크립트 전문가가 아니었다.

**이 패킷이 적용하는 원칙: 실행된 것이 기록되는 것이다.** `es loop collect`는 다른 모든 액추에이터
경로와 똑같이 플레인에게 로봇이 어디 있는지 알려주고, 플레인이 액추에이터 쪽으로 건넨
`SafeAction`을 기록한다. 원 명령은 `action_commanded`로 그 옆에 남는다. 그러면 시연은 구성상
엔벌로프와 일관되고, 에피소드당 한 번의 리셋은 모든 것의 리셋이 된다.

## context

```
crates/es-env/src/domains.rs
crates/es-env/src/expert.rs
crates/es-data/src/collect.rs
crates/es-data/src/lib.rs
crates/es-data/tests/loop_learning.rs
crates/es-data/tests/lerobot_v3.rs
crates/es/src/cmd/loop.rs
docs/api-notes/lerobot-dataset.md
docs/api-notes/lerobot-dataset.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V1c-executed-action-record.md
docs/packets/M5/V1c-executed-action-record.ko.md
```

주: 플레인 통과 이전 명령은 그것이 건네지는 곳 — `DomainRunner::emit_actions`, `es-env`, 레이어
9 — 에서 붙잡는다. 플레인 밖에서 그 값이 존재하는 유일한 지점이고, 여기서 `es-safety`는 금지이기
때문이다. `es-data`는 접근자 하나로 읽어 컬럼 하나를 쓴다.
`crates/es-data/src/lerobot/{columns,meta,v3}.rs`는 **건드리지 않는다**. v2.1 라이터와 v3.0
익스포터는 이미 `Info::features`에 대해 일반적이어서 피처를 선언하는 것이 변경의 전부이고, 그것이
V1b의 익스포터가 옳게 쓰였다는 증거다.

## spec

- **§9.3, §9.4, INV-12.** 아무것도 우회하지 않고 아무것도 끄지 않는다. 플레인은 이미 API가 있는
  입력 하나를 받을 뿐이고(`observe_state`, 문서 주석 자체가 "실제 자세를 아는 호출자는 이것을
  호출한다"고 말한다), `validate`는 시그니처를 유지한다(INV-13). 액추에이터로 가는 값은 여전히
  `SafeAction`뿐이고, `action_commanded`는 디스크 위의 출처 기록이며 제어 경로로 되읽히지 않는다.
- **§13.2.** 수집된 프레임은 자기 출처를 싣는다. `action_source`는 액추에이터 값이 *어떻게*
  만들어졌는지 이미 말했고, `action_commanded`는 *무엇을 요청했는지*를 말한다. 플레인을 다시
  돌리지 않고 `Clamped` 프레임을 읽을 수 있게 하는 것이 그것이다.
- **§19.1, §19.2.** 데이터셋 스키마는 학습 실행 입력 정체성의 일부다. 컬럼 추가는
  `dataset_schema_hash`를, 따라서 `content`를 이동시킨다. api-note가 그 이동을 기록하고 V1b의
  v3.0 내보내기는 두 컬럼을 모두 실으므로, 하나만 있는 데이터셋을 보는 소비자는 없다.
- **§17.2.** 전문가가 끝내지 못한 에피소드는 여전히 실패한 시연으로 기록되고 버려지지 않는다.
  여기서 바뀌는 것은 없다.
- **§3.5, §5.3.** 수집 실행은 시드의 순수 함수로 남는다. `observe_state`는 백엔드 자신의 상태를
  읽을 뿐 RNG도 부동소수점 누산도 더하지 않는다.
- **INV-17.** 새 트레이트 없음. 필드 하나, 접근자 하나, 상수 하나, 컬럼 하나.
- **§1.5.** `es-env`와 `es-data` 모두 상한에 여유가 크다. 이 패킷은 둘 합쳐 ~150 소스 라인 이내로
  예산을 잡는다.

## oracle

```
cargo fmt --check
cargo clippy -p es-env -p es-data -p es --all-targets -- -D warnings
cargo clippy -p es --features render --all-targets -- -D warnings
cargo test -p es-env
cargo test -p es-data
cargo test -p es --test cli
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
cargo xtask ci
```

- `crates/es-env/src/domains.rs`: **`emit_actions_writes_the_planes_answer_and_records_the_command`**.
  일부러 좁힌 엔벌로프로 16개 env × 8 제어 스텝의 액션 단계를 돌리고, env마다 관절마다
  `ctrl`이 `SafetyPlane::last_safe_action()`과 **비트 단위로** 같음(`f64::to_bits`, 허용오차 없음 —
  여기서의 허용오차는 둘 중 하나가 자기만의 변환을 길렀다는 뜻이다), 기록된 모든 값이 통과한
  엔벌로프 안에 있음, 그리고 플레인이 건네받은 명령이 적어도 한 스텝에서는 그것과 달랐음을
  단언한다. 마지막 단언이 없으면 아무것도 클램프하지 않는 엔벌로프에서도 통과한다.
- `crates/es-data/tests/loop_learning.rs`:
  **`the_dataset_records_the_executed_action_beside_the_raw_command`** — 위치 엔벌로프를 좁히고
  매 틱 모든 관절에 `0.9` rad를 요청하는 개입자를 붙인 수집 실행. `action`과
  `action_commanded`는 적어도 한 원소에서 달라야 하고, 모든 `action` 값은 엔벌로프 안이어야
  하며, 적어도 하나의 `action_commanded` 값은 밖이어야 한다. 출처 컬럼을 클램프한 라이터도,
  명령을 액션으로 기록한 라이터도 이 테스트에 걸린다.
- `crates/es-data/tests/loop_learning.rs`: **`a_second_episode_repeats_the_first_exactly`** —
  에피소드 리셋 회귀. 픽스처의 균등 리셋 추출을 상수로 고정하고 속도 엔벌로프를 좁히면 한 실행의
  두 에피소드는 물리적으로 동일하다. 그러면 `action`, `action_commanded`, `observation.state`,
  `action_source`가 행 단위로 같아야 한다. **이 패킷 이전에는 실패함을 측정했다**(에피소드 1이
  에피소드 0의 마지막 명령에서 이어져 `0.065`로 시작, 에피소드 0은 `0.005`).
- `crates/es-env/src/expert.rs`: `reset_restarts_the_ramp_from_the_measured_joints` — 같은 경계의
  전문가 쪽. 이미 올발랐고 이제 고정된다: `reset` 뒤 첫 청크는 이전 에피소드 적분기의 연장이
  아니라 측정된 관절에서 `step_max` 하나 떨어진 값이다.
- `crates/es-data/tests/lerobot_v3.rs`: `export_layout_is_v3`가 두 액션 컬럼 모두 V1b의 v3.0
  내보내기를 `float32`·`[NJ]`로 통과함을 단언하고, 파일 전체가 공유하는 픽스처가 이제 둘 다 쓴다.
  따라서 실제 `lerobot 0.6.1` 읽기 오라클 `lerobot_v3_export`는 새 컬럼이 실린 데이터셋을 읽는다.
  0.6.1이 추가 피처를 거부한다면 그 오라클이 그것을 말해 줄 것이다.

참조 — 오라클 서버(`ES_PYTHON`, `~/venvs/es-lerobot-cuda/bin/python`):

```
cargo test -p es-data --test lerobot_v3 -- --ignored --nocapture      # RAN lerobot_v3_export
cargo test -p es --test cli -- --ignored expert_solves_the_pinned_seeds
es loop collect --expert so101-pick-place --episodes 3 --seed 1       # 1이 아니라 3 성공
```

**측정**(플랜 V의 목적, CI 티어가 아니라 오라클 서버): 학습 50 에피소드와 홀드아웃 5
에피소드를 프레임과 함께 **한 개의** `es loop collect --episodes 50` 명령으로 재수집하고,
`es dataset bake --policy`, 그리고 `--batch 8 --lr 1e-4 --seed 0`에 1k/5k/20k 체크포인트로
20,000 스텝 재학습 — 학습 노브는 V2·V2b와 전부 동일 — 번들 3개를 패킹한 뒤 V3와 V2b가 돌린 그대로
V3의 측정을 재실행한다: 체크포인트마다 16개 nominal 에피소드, 20k에 대한 6개 스위트 표,
`ES_TRAINED_BUNDLE`을 쓴 `visible_learning_demo_run`, `es video mosaic` +
`python/es/encode_video.py` + `ffmpeg -c:v libx264` 사본. `evaluation.toml`의
`success_rate >= 0.5` 수용 기준은 **낮추지 않는다**. 측정된 수치를 그 기준에 대고 그대로 보고한다.
이 실험에서 움직이는 변수는 하나, 시연의 액션 규약이다.

**먼저 하는 진단**(설계 노트 7.10절): V2b의 기존 20,000 스텝 체크포인트를, 속도·가속도·엔드
이펙터·액션 레이트 한계를 기록된 리드가 들어갈 때까지 넓힌 *스크래치* deployment 문서를 가진
번들에 다시 패킹해 같은 16개 nominal 에피소드로 돌린다. 5분이면 되고, 정책이 애초에 과제를 할 수
있는지를 말해 준다 — 그것이 이 패킷이 내놓는 모든 수치를 읽는 방식을 결정한다. 커밋된
`deployment.toml`은 수정하지 않고, 아무것도 끄지 않는다(INV-12).

## acceptance

```rust
// crates/es-env/src/domains.rs
impl<const NJ: usize, const H: usize> DomainRunner<NJ, H> {
    /// 마지막 `emit_actions`가 Safety Plane에 건넨 값, 플레인이 답하기 전: `n_envs * NJ`,
    /// env 기준 행 우선. 출처 기록이지 액추에이터 값이 아니다(`INV-12`). 청크 버퍼에 그 틱의
    /// 행이 없던 틱에서는 플레인 자신의 답을 되풀이하며, `action_source`는 정확히 그 틱들에서
    /// `Fallback`을 읽는다.
    pub fn commanded(&self) -> &[f64];
}

// crates/es-data/src/collect.rs
/// 플레인 통과 이전의 원 명령, `action` 옆에 (스펙 13.2).
pub const ACTION_COMMANDED: &str = "action_commanded";
```

```
info.json features:
  "action":            { "dtype": "float32", "shape": [nu] }   # SafeAction, 변경 없음
  "action_commanded":  { "dtype": "float32", "shape": [nu] }   # 신규
```

- `Collector::run`은 매 `step_with_policy` 직전에 백엔드 자신의 관절 상태로
  `planes[0].observe_state(&q, &qd)`를 호출한다. `es_eval::runner`가 호출하는 것과 사이클상 같은
  지점이다.
- `action`은 의미도 바이트도 그대로다: `ep.ctrl`, 즉 `emit_actions`와 `Env::step`이 복사한
  `safe.q`. 이 패킷은 `action`이 무엇인지를 바꾸지 않는다. 그것이 무엇인지를 *구성상 참*으로 만들고
  오라클로 고정한다.
- `dataset_schema_hash`가 이동한다. `task_hash`, `observation_hash`, `learning_hash`와 로워링은
  이동하지 않으므로 베이크 세트, `evaluation.toml`, 패킹된 번들은 유효한 문서로 남는다.
- 새 트레이트 없음(INV-17), 새 의존성 없음, ~150 소스 라인 이내.

## forbidden

- `crates/es-safety/**` — 엔벌로프 변경, 새 메서드, 시그니처 변경 금지(INV-11, INV-12, INV-13).
  `observe_state`와 `last_safe_action`은 이미 존재하고 이미 다른 세 크레이트가 호출한다. 이
  패킷은 네 번째 호출자를 추가할 뿐이다.
- `tests/fixtures/visible-learning/deployment.toml` — 데모의 엔벌로프는 움직이지 않는다. 진단은
  커밋되지 않고 어떤 테스트도 읽지 않는 스크래치 사본을 쓰며, 그 외의 모든 것이 돌기 전에 픽스처를
  복원한다. 커밋된 엔벌로프를 넓히는 것은 미해결 질문 12의 선택지 (c)이고, 이 패킷의 요점은
  선택지 (a)다.
- `crates/es-policy/**`, `crates/es-ir/**`,
  `tests/fixtures/visible-learning/learning.toml` — Learning IR과 `act_like` 그래프는 V2b가 둔
  그대로다. 델타 액션 공간(선택지 (b)) 없음, 로워링 변경 없음, `lowering_hash`는 움직이면 안 된다.
- `crates/es-data/src/lerobot/{columns,meta,v3}.rs` — 라이터들은 선언된 피처에 대해 일반적이다.
  새 컬럼이 거기서 수정을 요구했다면 그것이 고침이 아니라 발견이었을 것이다.
- `evaluation.toml`의 `success_rate >= 0.5`나 어디의 임계값도 낮추지 않는다. 수용 기준을 통과하지
  못한 측정은 통과하지 못한 것으로 보고한다.
- V2와 V2b가 쓰지 않은 것으로 재학습하지 않는다: 같은 시드, 같은 배치, 같은 학습률, 같은 스텝 수,
  같은 50 에피소드, 같은 홀드아웃 시드.

### artifacts (오라클 서버, RTX 4090; 아래 중 커밋되는 것은 없다)

전부 `~/artifacts/plan-v/v1c/` 아래에 있고, 작은 것들은 요청자의 `target/plan-v/v1c/`로
복사했다. 프레임·타일·체크포인트는 커밋하지 않는다(설계 노트 9절: 프레임이 증거이고 mp4는 그것의
한 가지 보기다).

| 산출물 | 방법 |
|---|---|
| `ds-train`, `frames-train` | `es loop collect --expert so101-pick-place --episodes 50 --seed 1 --frames …` — **한 개의 명령**, 에피소드 리셋 수정이 사 주는 것 |
| `ds-holdout`, `frames-holdout` | 같은 방식, `--episodes 5 --seed 101` |
| `untrained.esb` | `cargo test -p es --test cli dataset_bake_writes`의 `policy.esb` |
| `baked/` | `es dataset bake --policy untrained.esb --frames frames-train ds-train` |
| `model-{1000,5000,20000}.safetensors` | `train_act.py --baked baked/ --batch 8 --lr 1e-4 --seed 0 --device cuda --checkpoint-at 1000,5000,20000` |
| `trained-{1000,5000,20000}.esb` | `es policy pack --policy untrained.esb --weights model-N.safetensors` |
| `nominal-{1000,5000,20000}/`, `suite-20000/` | `es eval run --config … --frames` |
| `demo-{1000,5000,20000}.mp4`, `-h264.mp4` | `es video mosaic --grid 4x4` + `encode_video.py --fps 50` + `ffmpeg -c:v libx264` |
