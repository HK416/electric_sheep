# M5 V6 — 수집과 평가를 위한 하나의 엔벌로프 의미론

설계 노트: `docs/design/visible-learning.md` **7.12절**, 그리고 여기까지 온 경위는 7.5–7.10절과
열린 질문 12. 스펙: §9.3, §9.4, §9.5, 부록 B.4. 이 파일보다 먼저 그것들과
`docs/design/safety-plane.md`를 읽을 것. V1(스크립트 전문가와 고정 시드 오라클), V1c(에피소드당
리셋, `action_commanded`), V3(스위트와 비공허성 게이트)에 의존한다.

## 결함

**평가 하니스가 전문가를 떨어뜨린다. 그러면 어떤 정책도 통과할 수 없다.**

`Collector::run`은 에피소드당 한 번 `observe_state`로 Safety Plane을 시드했다.
`es_eval::runner`·`es_ros2::hil`·`es_runtime_embedded`는 **매** `validate` 앞에서 불렀다.
`observe_state`는 `last_safe`, `prev_safe`, `vel`, `prev_vel`을 덮어썼다 — 클램프의
3(속도)·4(가속도)·7(rate limit) 단계가 차분을 취하는 바로 그 네 필드다. 그래서 같은 엔벌로프가
두 가지 다른 뜻을 가졌다:

| | `velocity_max` / `acceleration_max` / `action_rate`가 제한한 것 |
|---|---|
| `es loop collect` | plane 자신의 명령, tick 대 tick |
| `es eval run`, HIL, 임베디드 | 명령 **빼기 측정된 관절** — 서보의 추종 오차 |

데모의 숫자에서 구속하는 것은 가속도 단계이고, 두 번째 독해는
`a − q − q̇·dt ≤ acceleration_max · dt² = 0.008` rad으로 무너진다. 스크립트 전문가는 Deployment
IR이 허용하는 것에 정확히 맞춰 페이싱하고 `es loop collect`에서 **50/50**을 내지만
`es eval run`에서는 **16 중 0**을 낸다. V3·V2b·V1c가 보고한 모든 평가 셀이
`envelope_violation_rate 1.0000`이고 `ActionSource::Policy` 스텝이 하나도 없는 이유다 — 0이라는
천장에 대고 측정되었다.

V1c는 이것을 측정했지만 고칠 수 없었다. `es-safety`가 `forbidden` 목록에 있었고, 대신 수집기
쪽에서 닫으면(매 스텝 재시드) 전문가가 자기 골든 임계에서 2/8로 떨어진다. 이 패킷은 마땅한 곳에서
고친다.

## 결정, 그리고 근거가 되는 스펙 문장

**`velocity_limit`, `acceleration_limit`, `rate_limit`은 plane 자신의 명령들의 차분이다. 측정된
상태는 에피소드당 정확히 한 번, 시드로서 엔벌로프에 들어온다.**

* **§9.3.** 표는 *정책 출력에 적용되는 정적·동적 제약*이고 동적 행은 모두 **클램프**라고 말한다:
  `velocity_limit` → "관절·EE 속도 상한 / 클램프 + 카운터", `rate_limit` → "액션 1차·2차 미분
  상한 / 필터링". 측정된 속도는 클램프할 수 없다 — plane이 지금 내보내려는 값만 가능하다. 따라서
  각 행이 제한하는 양은 plane 자신의 것이어야 한다.
* **§9.5.** "**`deployment_hash`가 같으면 안전 동작이 같다** — 이것이 §27.1 증거물의 핵심
  주장이다." 같은 deployment 해시는 시뮬과 실기에서 같은 안전 액션을 주어야 한다. 피드백에서
  계산한 클램프는 그럴 수 없다: `qvel`은 MuJoCo에서는 정확하고, `STS3215`에서는 시리얼 버스를 통해
  양자화되고 지연되어 온다. 그것을 엔벌로프에 읽어들이면 §9.5가 증거물의 핵심이라 부르는 주장이,
  그리고 §3.4의 결정성 규칙이 함께 깨진다.
* **§9.3, 다시 — 추가하지 *않은* 것.** "명령이 측정값에서 멀다"에 해당하는 행은 없다. 명령이
  *어디로* 갈 수 있는지를 제한하는 행은 `position_limit`과 `workspace` 둘이고 둘 다 그대로다.
  오케스트레이터의 기본값 — 측정된 속도를 제한하고 측정 자세를 향해 제동한다 — 은 **채택하지
  않았다**. 스펙이 침묵하지 않고 §9.5가 그것을 배제하기 때문이다.
* **이 로봇에서, 물리적으로.** SO-101은 Feetech `STS3215` 여섯 개이고 MuJoCo `position`
  액추에이터로 구동된다: `kp = 998.22`, `kv = 2.731`, `forcerange = ±2.94 N·m`. 루프는 서보
  안에서 닫히고 — 호스트가 목표를 보내면 서보가 오차로 토크를 만든다 —
  `2.94 / 998.22 = 0.0029` rad이 그 토크의 포화점이다. **3 밀리라디안의 추종 오차면 이미 최대
  토크다.** V6 이전 평가의 `0.008` rad 한계는 서보 오차 신호를 포화점의 2.7배에서 제한한 것이었다:
  속도라고 적힌 행에서 토크를 제한하고 있었고, `torque_limit`은 두 행 위에 따로 있으며 모델 자신의
  `forcerange`가 이미 그것을 강제한다.

**`tests/fixtures/visible-learning/deployment.toml`의 어떤 숫자도 움직이지 않는다.** 결함은
기준이었지 한계가 아니었다. 이 파일은 이제 선언하는 모든 숫자의 유도를 담는다 —
`velocity_max = 3.0` rad/s와 `acceleration_max = 20` rad/s² 뒤의 서보 정격, 50 Hz에서 두
`action_rate` 행이 지배당한다는 사실과 그것이 제어 속도에 의존하지 않기 위해 존재한다는 점, 그리고
`es-safety`가 강제하지 않는 세 개(`ee_velocity_max`, `contact_force_max`, 거리 최솟값 — plane에는
FK도 접촉 질의도 없다).

## context

```
crates/es-safety/src/plane.rs
crates/es-safety/tests/envelope_reference.rs
crates/es-data/src/collect.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/evaluation.rs
crates/es-env/src/expert.rs
crates/es/src/cmd/loop.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/deployment.toml
docs/design/safety-plane.md
docs/design/safety-plane.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V6-envelope-semantics.md
docs/packets/M5/V6-envelope-semantics.ko.md
```

`es-ros2`와 `es-runtime-embedded`는 **변경이 필요 없다**: 둘 다 에피소드가 없고, 새 plane을
만들며, 이미 매 `validate` 앞에서 `observe_state`를 부른다. 규칙이 하나인지 아닌지의 시험이
그것이다 — 소비자별 패치가 필요했다면 그것은 하나의 규칙이 아니었다는 뜻이다.

## spec

- **§9.3, §9.5, INV-12.** 무엇도 비활성화되지 않고, 엔벌로프 숫자는 움직이지 않으며, 어떤 제약도
  건너뛰지 않는다. `observe_state`가 가드를 얻고, `begin_episode`는 래치 해제와 시드 재무장을 한다
  — 그것이 대체하는 `reset_latch`가 이미 할 수 있던 것보다 엄밀히 적다.
- **INV-13.** `SafetyPlane::validate`는 시그니처를 유지하고 여전히 `Result`를 반환하지 않는다.
- **INV-17.** 새 트레이트 없음. `bool` 필드 하나, 공개 메서드 하나, 옮겨진 `impl` 블록 하나.
- **§3.4, §3.5.** plane은 자신의 명령과 시드의 순수 함수로 남는다: float 시간 없음, `HashMap`
  없음, RNG 없음, 새 초월함수 없음. 시드는 에피소드당 `observe_state` 한 번이고 실행 기록에 남는다.
- **§9.6, 부록 B.4.** 여전히 `no_std` 가능하고 여전히 힙 0: 새 필드는 미리 할당된 `SafetyState`
  안의 `bool`이다. `construction_and_validation_allocate_nothing`이 그것을 덮는다.
- **§1.5.** 측정값: 소스 다섯 파일에서 **+101 / -54행, 순 +47행** — `es-safety` +36,
  `es-env` +37(옮겨 온 페이싱), `es-eval` +5, `es-data` -7, `es` -24(페이싱이 떠남). 모든
  크레이트가 상한 안쪽에 넉넉히 있고, 규칙대로 테스트는 제외한다.

## oracle

```
cargo fmt --check
cargo clippy -p es-safety -p es-eval -p es-data -p es-env -p es-ros2 -p es-runtime-embedded -p es --all-targets -- -D warnings
cargo clippy -p es --features render --all-targets -- -D warnings
cargo test -p es-safety -p es-eval -p es-env -p es-data -p es-runtime-embedded
cargo test -p es-ros2 --test hil_gate
cargo test -p es --test cli
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
cargo xtask ci
```

- `crates/es-safety/tests/envelope_reference.rs`:
  **`the_envelope_bounds_commands_not_the_following_error`** — 데모의 *커밋된*
  `deployment.toml`로 plane을 만들고, 전문가 자신의 페이싱 규칙으로, 최대 `0.1774` rad(옛 한계의
  22배) 뒤처지는 1차 지연 플랜트에 대고 구동해, 60번 연속 `ActionSource::Policy`이고 출력이 명령과
  비트 단위로 같다는 것, **그리고** 매 tick 관측하는 호출자가 한 번만 관측하는 호출자와 동일한
  수열을 낸다는 것을 단언한다. V6 이전 코드에서 스텝 1에 실패함을 측정했다. 플랜트가 실제로
  뒤처졌는지도 단언하므로 공허하게 통과할 수 없다.
- `crates/es-safety/tests/envelope_reference.rs`:
  **`a_command_outside_the_envelope_is_still_clamped`** — V6는 아무것도 넓히지 않았다. 3 rad/s
  한계에 대고 50 rad/s는 여전히 `Clamped`, 여전히 `ViolationKind::Acceleration`, 여전히 세어진다.
- `crates/es-safety/tests/envelope_reference.rs`:
  **`begin_episode_re_arms_the_seed_and_clears_the_latch`** — 에피소드 중간의 측정은 클램프 기준을
  결코 움직이지 않고, 다음 에피소드의 첫 측정은 움직인다.
- `crates/es-eval/tests/evaluation.rs`:
  **`the_runner_seeds_the_plane_once_per_episode_and_observes_every_step`** — 이 크레이트가 지는
  호출 규율의 몫. 옆에 있는 `the_plane_is_never_disabled` 스캔과 같은 방식이다: `observe_state`
  하나, 유일한 `validate` 앞에, `begin_episode` 하나, 맨 `reset_latch` 없음.
- `crates/es-data/tests/loop_learning.rs`: **`a_second_episode_repeats_the_first_exactly`** —
  V1c 자신의 회귀 테스트로, 그대로 통과한다. 수집기의 에피소드 경계가 `begin_episode`를 거치게 된
  지금도 여전히 재시드한다는 것을 말해 주는 것이 이것이다.
- `crates/es-ros2/tests/hil_gate.rs`: **`v1_fixture_still_replays_identically`** — 그대로이며
  **재생성하지 않았다**. `tests/fixtures/hil/v1_small.eshil`은 서로 다른 값 112개를 가진
  `observe_state` 레코드 120개를 담고 있어서 산술이 바뀌었다면 그 안의 모든 결정이 움직였을 것이다.
  HIL 리그의 플랜트가 완전한 위치 서보라 두 독해가 그곳에서 일치하고 로그는 바이트 단위로 재생된다.
  가정이 아니라 측정이다.
- `crates/es-safety/tests/scenarios.rs`(§28.7 게이트 8 스위트)와 `properties.rs`: **변경 없음,
  재고정된 픽스처 없음.** 모든 시나리오가 시작에서 한 번 시드하는데, 그것이 바로 V6가 보편화하는
  규율이다.

**서버 오라클** — `crates/es/tests/cli.rs` **`expert_passes_the_evaluation_harness`**.
`ScriptedExpert`를 **정책으로서** `es_eval::Evaluation`에 태워 돌린다 — 진짜 러너, 진짜 plane,
진짜 Task IR 성공 술어 — 같은 고정 시드 `[1, 2, 3, 5, 8, 13, 21, 34]`와
`expert_solves_the_pinned_seeds`와 같은 `0.875` 임계로. 두 상수는 이제 두 테스트가 공유하는 한
쌍이다. SO-101 씬을 구동할 수 있는 것은 `MuJoCoCpuBackend`뿐이므로 `mujoco`가 없으면 이유를 찍고
건너뛰며, 서버 오라클로 여기에 이름을 남긴다. 테스트 자신의 문서 주석에 밝힌 두 좁힘: 시드당
`Evaluation::run` 한 번에 에피소드 하나(`Evaluation::run`은 Task IR의 무작위화를 에피소드
카운터로 키잉하고 `Env`를 `seeds[0]`으로 시드하므로, 8 에피소드 셀 하나는 *한* 시드의 여덟 번
추첨이다), 그리고 상수 96×96 프레임 — 전문가는 관절을 읽고 픽셀을 읽지 않으므로, 이것이 이
오라클을 Vulkan 장치 없이 `mujoco`만으로 돌게 한다.

**V3의 비공허성 규칙은 의미를 유지한다.** V6 이후 `Clamped` 스텝은 정책 자신의 청크에서 온다:
`acceleration_max · dt² = 0.008` rad보다 크게 벌어진 연속 행, 또는 소프트 관절 한계를 넘는 행.
둘 다 `nominal`을 포함한 **모든** 스위트에서 도달 가능하며 — V1c의 확대 엔벌로프 진단이 훈련된
정책을 "거의 전부 `violation.position`"으로 측정했고 그것이 V6가 건드리지 않는 소프트 한계
단계다 — `torque_noise` / `backlash`는 팔을 자기 청크 아래에서 빼내어 정책을 그리로 미는, 여전히
가장 유망한 스위트다. 로컬에서는 정책 없이도 `a_command_outside_the_envelope_is_still_clamped`
(`es-safety`)와 `the_envelope_violation_rate_rises_when_the_policy_leaves_the_envelope`,
`a_tightened_envelope_clamps_and_a_widened_one_does_not`(`es-eval`)로 도달 가능성이 고정된다.
훈련된 번들이 여전히 그것을 건드리는지는 2단계 측정이고, 멈춘다면 그것은 강제할 숫자가 아니라
보고할 발견이다.

**2단계, 오라클 서버에서**(`ES_PYTHON`, `~/venvs/es-lerobot-cuda/bin/python`; 재훈련 없음, 손잡이
변경 없음):

```
cargo test -p es --test cli expert_passes_the_evaluation_harness -- --nocapture
cargo test -p es --test cli expert_solves_the_pinned_seeds       -- --nocapture   # 여전히 8/8
es eval run --config tests/fixtures/visible-learning/evaluation.toml \
            --policy ~/artifacts/plan-v/v1c/trained-20000.esb --frames … --jobs 6
```

첫 번째가 이 패킷 자신의 게이트다. 두 번째와 세 번째는 V1c의 커밋된 20,000 스텝 번들 —
nominal 16과 6 스위트 96 — 을 교정된 엔벌로프 의미론 아래에서 설계 노트 7.10절의 V1c 표에 대고
재측정한다. `evaluation.toml`의 `success_rate >= 0.5`는 낮추지 않는다. 무엇이 나오든 그것에 대고
보고한다.

## acceptance

```rust
// crates/es-safety/src/plane.rs
impl<const NJ: usize, const H: usize> SafetyPlane<NJ, H> {
    /// 측정된 관절 상태로 명령 체인을 시드한다. `begin_episode`(또는 생성) 이후 첫 호출만
    /// 무언가를 움직이므로, 모든 소비자가 매 `validate` 앞에서 불러도 되고 모두 같은
    /// 엔벌로프를 얻는다.
    pub fn observe_state(&mut self, q: &[f64; NJ], qd: &[f64; NJ]);

    /// e-stop 래치를 해제하고 시드를 다시 무장시킨다. 무엇도 약화시키지 않는다(INV-12).
    pub fn begin_episode(&mut self);
}

// crates/es-env/src/expert.rs
impl ExpertCfg {
    /// 전문가를 구동될 엔벌로프에 맞춰 페이싱한다. `replan_every`는 호출자가 다음 청크를 요청하기
    /// 전에 실행되는 행 수다: `es loop collect`는 `action.execute_chunk`를, `es_eval::runner`는
    /// 1을 넘긴다.
    pub fn pace_to(&mut self, deploy: &es_ir::deployment::DeploymentIr, replan_every: u32);
}
```

- 규칙은 하나, 한 곳에서 강제된다. 측정값이 무엇을 뜻하는지는 `observe_state`가 결정하고 어떤
  소비자도 국소적으로 결정하지 않는다. 수집·평가·HIL·임베디드 모두 매 `validate` 앞에서 부른다.
- `Collector::run`은 `if frame == 0`을, `run_episode`는 `reset_latch`를 잃는다. 둘 다 에피소드
  경계에서 `begin_episode`를 부른다.
- `validate`는 시그니처를 유지하고(INV-13), 어떤 경로도 plane을 비활성화하거나 우회하지
  않으며(INV-12), 새 트레이트가 없고(INV-17), 엔벌로프 숫자가 움직이지 않으며, 골든이 편집되지
  않고 안전 시나리오가 재고정되지 않는다.
- `deployment.toml`은 산문만 얻는다. `deployment_hash`는 움직이지 않으므로 패킹된 모든 번들,
  `evaluation.toml`, `.eshil` 헤더가 유효한 채로 남는다.
- `tests/fixtures/hil/v1_small.eshil`은 재생성하지 않으며 여전히 바이트 단위로 재생된다.

## V6b — 구현 결과: 평가가 수집과 같은 방식으로 청크를 실행한다

설계 노트 7.13절. 1단계 이후 오케스트레이터가 지시한 1b 단계로, 7.12절 발견 6이 남긴 것을 닫는다 —
둘 중 하나라도 열어 둔 채 서버에서 재측정하면 다시 해야 하기 때문이다.

**문제의 모양을 바꾸는 정정.** 7.12절 발견 6은 수집기가 "추론 속도로 재계획하고 열 행을 실행한다"고
말했다. 그렇지 않다. `BatchDomains::single_env()`는 추론 주기를 1로 선언하므로 `es loop collect`는
**매** 제어 tick마다 청크를 받는다. 거기서 `action.execute_chunk`와 `execution`을 의미 있게 만드는
것은 `es_env::chunk_buffer::ChunkBuffer`다: 도착을 모두 저장하고, `span`이 한 청크가 몇 tick을
구동할지 정하며, `action_at`이 **겹침을 혼합**한다 — `w_i = exp(-decay · i)`, ACT temporal
ensembling이고 데모가 `learning.toml`과 `deployment.toml` 양쪽에 선언한 것이다.

`es_eval::runner`에는 버퍼가 없었다. 매 원시 추론 결과를 새 `seq`로 plane에 넘겼으므로 plane의
커서가 매 tick 리셋되었고 **모든 청크의 0행만 실행되었다**. 열다섯 청크의 지수 평균에 대고 훈련된
정책이 자신의 원시 마지막 예측으로 평가되었다.

**(a) 규칙 둘이 아니라 공유 함수 하나.** `DomainRunner::emit_actions`의 청크→plane 단계를
`es_env::plane_chunk(buffer, feed, tick, mode)`로 옮기고, 비공개 `Submitted`를 `end_episode`를 가진
공개 `PlaneFeed`로 승격했다. `es_eval::run_episode`는 모든 추론 결과를 **Deployment IR**로 만든
`ChunkBuffer`(`action.execute_chunk`, 그리고 `execution` → `ChunkBlendPolicy`이므로
`TemporalEnsemble { decay }`가 `TemporalEnsemble { weight_decay }`가 된다)에 밀어 넣고 같은 호출로
plane에 공급한다. `infer_chunk`는 `seq` 인자를 잃었다: seq는 버퍼가 받아들인 결과당 한 번 찍히고,
그것이 `SafetyPlane::accept`가 신선도를 판정하는 기준이다(§8.6). `DomainRunner::reset_env`도 이제
`PlaneFeed::end_episode`를 부르므로 에피소드 경계는 양쪽에서 하나의 함수다. **CLI 플래그 없음.
Deployment IR이 결정한다.**

`es loop collect`는 비트 단위로 변하지 않았다 — 추출은 이동이고, `es-env`와 `es-data`의 스위트는
V1c의 `a_second_episode_repeats_the_first_exactly`와
`emit_actions_writes_the_planes_answer_and_records_the_command`를 포함해 그대로 통과한다.

**남은 차이 하나, 숨기지 않고 이름을 붙여서.** 수집은 정책 계약의 `expected_latency_ms`(제어 1
tick)를 모델링하므로 첫 청크가 tick 1에 적용되고 tick 0은 청크 언더런이다. 평가는 결과가 계산된
tick에 적용한다. Deployment IR에는 지연 필드가 없고(`deadlines.inference_budget`은 워치독 경계이지
스케줄이 아니다) `Evaluation::run`은 Learning IR을 받지 않는다. 900 스텝 에피소드당 한 프레임.

**(b) `--seed S`는 하나의 장면을 가리킨다.** `Env::new`가 한 번 리셋하고(`(seed, env, episode)`의
추첨 0, §6.3) 모든 에피소드는 정확히 한 번의 리셋으로 끝난다 — 종료 조건이면 `Env::step` 자신의
리셋, 예산이 다하면 명시적 리셋. `run_episode`는 맨 앞에서 **또** 리셋했으므로 평가의 에피소드 `i`는
추첨 `2i + 1`에서, 수집의 에피소드 `i`는 추첨 `i`에서 돌았다. 그 리셋을 없앴다. 아래쪽 리셋,
`plan.reset()`, 셀마다의 `Env`는 그대로다.

**추가된 오라클.**

- `crates/es-eval/tests/evaluation.rs`: **`episode_zero_runs_on_the_first_randomization_draw`** —
  로컬, 가짜 백엔드. `Env`에서 직접 기준값을 만들고 러너가 처음 제공하는 상태가 `Env::new` 자신의
  추첨과 비트 단위로 같음을 단언한다. **두 번째 리셋을 되살리면 실패함을 측정했다**(`0.5904` 대
  `0.8683`).
- `crates/es-eval/tests/evaluation.rs`:
  **`the_runner_feeds_the_plane_through_the_collectors_chunk_buffer`** — 호출 규율 스캔:
  `buffer.push` 하나, `es_env::plane_chunk` 하나, `feed.end_episode` 하나, 그리고 정확히 하나의
  `env.reset(None)`(아래쪽 것).
- `crates/es-data/tests/loop_learning.rs`:
  **`the_collector_resets_once_per_episode_and_never_before_the_first_observation`** — 같은 규칙의
  반대편이라, 추가 리셋이 이쪽으로 옮겨 올 수 없다.
- `crates/es/tests/cli.rs`: **`collection_and_evaluation_draw_the_same_scene_for_a_seed`** — 교차
  경로 오라클. `es_data::Collector`와 `es_eval::Evaluation`을 데모 씬에서 시드 1로 돌려, 각자가
  자기 정책에 처음 건네는 `qpos ‖ qvel`을 원시 `f64` 비트로 비교한다. SO-101 씬은
  `MuJoCoCpuBackend` 말고 구동할 것이 없으므로 **서버 오라클**이다.

**무효화되는 것.** 설계 노트 7.8–7.11절의 모든 평가 숫자 — V3·V2b·V1c의 성공률, 엔벌로프 위반율,
`ActionSource` 히스토그램, 에피소드 길이, 두 번의 확대 엔벌로프 진단. 전부 청크 버퍼 없이, temporal
ensembling 없이, `execute_chunk`가 죽은 채로, 엔벌로프가 추종 오차를 제한한 채로(V6), 큐브가 한
추첨 어긋난 채로(V6b) 측정되었다. 수집과 훈련 숫자는 그대로 유효하다: 시연, 손실 곡선,
`observation_hash`, `lowering_hash`, `dataset_schema_hash`, 체크포인트는 움직이지 않은 경로의
산물이다.

**소스 줄 변화 (V6b).** `es-env` +104 / -65(순 +39, 대부분 공유 함수와 그 주석), `es-eval`
+59 / -13(순 +46). 두 크레이트 모두 §1.5 상한에서 한참 안쪽이다.

## 2단계 — 측정된 대로

설계 노트 `docs/design/visible-learning.md` 7.12절(하니스 결과, 20,000 스텝 재측정)과 7.13절 발견
7–8(메커니즘, 그리고 그것이 강제한 두 테스트 수정)이 분석을 담고 있다. 이 절은 기록이다.

**서버:** RTX 4090, `~/venvs/es-lerobot-cuda/bin/python`, 트리 `~/Projects/es-v6`(이 브랜치의
`5a7e00e` 헤드에서 빌드된 아카이브 체크아웃 — 아카이브에는 `git log`가 없다),
`~/artifacts/plan-v/v6/` 아래의 아티팩트, 2026-09-15. 재훈련 없음, 손잡이 변경 없음.

**오라클.**

| 테스트 | 결과 | 출처 |
|---|---|---|
| `expert_solves_the_pinned_seeds` | 8/8 `Success` | `oracle-collect.log` |
| `expert_passes_the_evaluation_harness` | 8/8 `Success`, `envelope_violation_rate` 0.4815–0.5485 | `oracle-eval.log` |
| `collection_and_evaluation_draw_the_same_scene_for_a_seed` | `f64`에서는 실패로 측정됨(`0.2550719976425171` 대 `0.255071989355131`), `f32`에서 비교하도록 수정 | `oracle-parity.log` |

**`expert_passes_the_evaluation_harness` 자신의 게이트가 다시 고정된다.** 1단계의
`worst_violation < 0.02`는 V6b 절 자체의 포인트 1이 거짓으로 만든 tick 단위 독해를 가정했다:
plane은 이제 16행 호라이즌(`decay = 0.01`)의 temporal-ensemble 혼합을 재고, 그 지터가 범위 밖
위치를 하나도 요구하지 않는 페이싱된 궤적에서도 속도/가속도 클램프를 건드린다 — 7.10절 자신의
수집-경로 숫자(V1c 시연 프레임의 0.6368이 `Clamped`이거나 `Fallback`)가, 여기서 재측정하지 않는
경로에서 더 먼저 측정된 같은 모양을 보여준다. `crates/es/tests/cli.rs`는 이제 여전히 참인 한계를
읽는다 — 테스트에 타이핑해 넣은 숫자 대신 Deployment IR 자신의 `EnvelopeViolationRate` 워치독,
`max_frac = 0.9`(`tests/fixtures/visible-learning/deployment.toml`)다. 측정된 최악값 0.5485는 거기
한참 못 미치고, `expert_solves_the_pinned_seeds`의 8/8은 두 경로를 맞대보는 골든으로 남으며, 이
패킷의 `forbidden` 목록이 보호하는 임계값은 어느 것도 움직이지 않았다.

패리티 오라클의 수정은 무관하다: 두 경로가 처음 내는 `qpos ‖ qvel` 행을 `f64`가 아니라 두 경로가
실제로 공유하는 폭인 `f32`(수집기 자신의 `state_row` 텐서와 시연의 `observation.state` 컬럼)에서
비교한다. `f64` 비교는 `f32`로 반올림된 수와 반올림되지 않은 수를 비교한 것이었다.

**V1c의 20,000 스텝 번들, 재측정:**

| | V3 | V2b | V1c | V6, 정직한 값 |
|---|---|---|---|---|
| nominal `success_rate` (16) | 0.1250 (2/16) | 0.0000 (0/16) | 0.0625 (1/16) | **0.0000 (0/16)** |
| suite `success_rate` (96) | — | — | — | **0.0000 (0/96)** |

여섯 스위트, 각 14,400 제어 tick (`nominal-20000/report.json`, `suite-20000/report.json`):

| 스위트 | envelope_violation_rate | fallback | clamped | policy |
|---|---|---|---|---|
| nominal | 0.2106 | 220 | 2,813 | 11,367 |
| light_intensity | 0.4790 | 333 | 6,565 | 7,502 |
| light_direction | 0.3609 | 400 | 4,797 | 9,203 |
| observation_delay | 0.2142 | 220 | 2,865 | 11,315 |
| torque_noise | 0.1817 | 140 | 2,476 | 11,784 |
| backlash | 0.2265 | 240 | 3,021 | 11,139 |

`fallback`은 `failure_mode_histogram` 자신의 버킷이고, 모든 스위트에서 히스토그램의
`violation.rate` 버킷과 같다 — 모든 fallback tick은 violation-rate 워치독이며 다른 무엇도 뜨지
않는다. `clamped`는 `envelope_violation_rate · 14,400 − fallback`이다. 모든 스위트에서
`violation.position`이 `violation.acceleration`보다 2배 넘게 큰 가장 큰 클램프 버킷이다(nominal:
2,031 대 843, 그다음 `violation.velocity` 439) — 소프트 관절-한계 단계이지, 위에서 전문가 자신의
실행이 보인 temporal-ensemble 지터가 아니다. `violation.rate_limit`과 `violation.torque`는 어느
스위트에도 나타나지 않는다(deployment.toml 자신의 주석: 두 `action_rate` 행과 `torque_max`는 이
제어 속도에서 관절-위치 명령이 건드리는 것보다 느슨하다).

**V3의 비공허성 규칙이 정직하게 유지된다.** `Clamped ≥ 1` — `nominal`만으로도 2,813, 진짜 클램프
산술이고 V6가 없앤 추종 오차 아티팩트가 아니다.

**결론.** 하니스는 전문가를 통과시킨다. 비전 정책은 과제를 해내지 못한다: nominal 16 중 0, 스위트
96 중 0, 하니스 아티팩트가 아니라 관절-한계 단계에서 클램프된다. 중단 규칙 사다리대로, 다음 심판은
V7a(특권 상태)이며 1단계(설계 노트 7.14절)가 이미 시작되어 있다.

## forbidden

- `tests/fixtures/visible-learning/deployment.toml`의 한계를 바꾸는 것, 또는 어디든 임계값을
  바꾸는 것 — `expert_solves_the_pinned_seeds`의 `0.875`, `evaluation.toml`의
  `success_rate >= 0.5`, V3의 비공허성 게이트. 수용 기준을 못 맞춘 측정은 못 맞춘 대로 보고한다.
- plane을 약화시키는 우회, `cfg`, 기능 플래그, 테스트 훅을 추가하는 것(INV-12), `validate`에
  `Result`를 주는 것(INV-13), `es-safety`가 `es-policy`에 의존하게 하는 것(INV-11).
- 새 확장점, 새 트레이트, 또는 에피소드 경계를 위한 `PolicyRuntime` 메서드. V6 오라클의
  expert-as-policy는 테스트 비계이고 테스트 파일 안에 산다.
- 7.12절 발견 6이 기록한 두 비대칭 — 평가 러너의 tick당 재계획과 추가 리셋 — 을 고치는 것. 둘 다
  플랜 V가 보고한 모든 평가 숫자를 움직이고, 둘 다 엔벌로프 문제가 아니다. 열린 질문 13이다.
- 무엇이든 재훈련하는 것. 2단계는 V1c의 커밋된 번들을 재측정하고 훈련 손잡이를 바꾸지 않는다.
- `crates/es-ros2/**`와 `crates/es-runtime-embedded/**` 소스. 규칙이 거기서 패치를 필요로 했다면
  그것은 하나의 규칙이 아니었다.
