# M5 V7a — 시뮬레이터 특권 상태 정책

설계 노트: `docs/design/visible-learning.ko.md` **7.14절**, 그리고 여기까지 온 경위는 7.5–7.13절과
미해결 질문 11–13. 스펙: §5.1, §6.2, §7.2, §7.4, §8.2, §19.2. 이 파일보다 먼저 읽을 것.
V1(전문가와 데이터셋), V2/V2b(학습과 bake), V3(스위트), V5(`--resident-gpu`), V6·V6b(전문가를
통과시키는 하네스와 시드 자신의 씬)에 의존한다.

## 이 패킷이 존재하는 이유가 되는 질문

**답이 들어 있는 관측을 주면, 데모의 정책은 과제를 해낼 수 있는가?**

데모의 비전 ACT — 시연 50개, 96×96 오버헤드 카메라 하나, 청크 10, 스크래치 ResNet18, 20,000
스텝 — 는 동작하는 정책을 한 번도 내놓지 못했고, V6/V6b 이전에 측정된 모든 평가 수치는 *스크립트
전문가*를 떨어뜨리던 하네스(7.12절)나 모든 청크의 0행만 실행하던 하네스(7.13절)에서 나온 것이다.
그 수치들은 사라졌다. 7.8–7.11절에서 넘어오는 것은 없다.

그래서 다음 측정은 비전 데모가 한꺼번에 묻던 두 질문을 분리해야 한다:

1. 이 그래프, 이 엔벌로프, 이 전문가, 이 학습 루프가 **큐브가 어디 있는지 알려주면** 큐브를 통에
   넣는 정책을 만들어낼 수 있는가,
2. 50개 에피소드로 스크래치 학습한 ResNet18이 96×96 프레임에서 25 mm 큐브를 **찾아낼** 수 있는가.

V7a는 1번이고, 영상의 1단계("상태 정책")다. 2번은 별도 패킷 — 더 높은 해상도, 더 많은 시연 — 이며
이 패킷이 아니다.

**중단 규칙, 그리고 이것은 패킷의 일부다.** 상태 정책이 시드 101–116의 nominal에서
`success_rate ≥ 0.5`에 **도달하지 못하면**, 다음 단계는 더 큰 모델도, 더 긴 스케줄도, 더 많은
시연도 **아니다**. 물리, 접촉 모델, 전문가 궤적을 의심하는 것이다. 큐브의 정확한 포즈와 팔의
정확한 관절각, 그리고 7초짜리 스크립트 동작의 시연 50개를 받은 정책에게 부족한 정보는 남아 있지
않다. 실패를 보고하고 옵티마이저가 아니라 씬을 열어라.

## 결정들과 각각의 근거

### 1. 큐브의 포즈는 어디서 오는가

**두 번째 `ObservationSpec` 채널 `sim_cube_pose`, `ObsSource::JointState { body: <`cube_free`
조인트의 `StableId`>, dof: 7 }`, 타입 `f32[7]`.** Task IR은 이를 *선언*할 뿐 아무것도 구현하지
않고(§5.1 규칙 6), Observation IR이 정규화하며, Learning IR이 인코딩한다. `es-ir` 변경 없음, 새
노드 변형 없음, 새 필드 없음.

형태를 결정한 세 가지 발견이 있고, 모두 저장소가 이미 갖고 있던 제약이다:

* **`ObservationSpec`은 7칸 free 조인트를 실을 수 있고, 해석 경로는 이미 존재한다.**
  `es_eval::runner::input_sources`는 플랜 입력의 소스 `StableId`를 Task IR 채널의 `dof`로
  떨어지기 **전에** `ModelInfo.qpos`에 대해 해석한다(`crates/es-eval/src/runner.rs`).
  `ModelInfo.qpos`는 **조인트** id로 키가 잡히므로, 소스가 `cube_free` 조인트인 채널은
  `Capture::Qpos(6..13)` — 그 조인트 자신의 `qpos` 일곱 값 — 을 정확히 받는다. 소스가 *바디*인
  채널(로봇의 `base`인 `joint_state`가 그렇다)은 `Capture::Joints(dof)`, 즉 "행의 앞 `dof`개"로
  떨어진다. 둘 다 기존 분기이고 V7a는 어느 쪽도 추가하지 않는다.
* **Task IR-D는 `GetJointState`로 free 조인트의 `qpos`를 낼 수 없다.** 그 노드의 출력 타입은
  `f32[joints.len()]` — 조인트 *이름* 하나당 스칼라 하나 — 이고, 이는 설계 노트 5.4절이 이미
  기록한 천장이며 성공 술어가 큐브의 `x`만 보는 이유다. 그래서 그래프는 값을
  `GetBodyPose { body: cube, relative_to: World }` → 그 `pos[3]`과 `quat[4]`의
  `Concat{axis 0}` → `ObservationSpec`으로 *보여준다*. 기존 노드 집합으로 표현 가능하고 간선마다
  타입이 맞는다. **채널**은 바디가 아니라 free 조인트를 가리키는데, 채널이 *읽기*를 결정하기
  때문이다: 조인트를 통하면 `qpos`로 정확하고, 바디를 통하면 `xpos ‖ xquat`이 되는데 기록된
  데이터셋에는 그것이 없다.
* **`Normalize{Range}`는 포트당 `(lo, hi)` 하나다.** 그래서 특권 값들에는 자기 포트가 필요하고,
  이것이 `joint_state`를 6에서 13으로 넓히지 않는 두 번째 이유다. 상태 분기의 `±1`에서 큐브의
  0.06 m 추출 범위는 출력 범위의 0.03을 차지하지만, `±0.3` — 추출 범위(`x ∈ [0.21, 0.27]`)와 통의
  내부(`x ∈ [0.09, 0.19]`, `y ∈ [-0.15, -0.05]`)를 모두 품는 팔의 도달 범위 — 에서는 0.10을
  차지한다. 쿼터니언의 네 값은 이 범위에서 `[0, 1]`을 벗어난다(`w = 1`이 2.17로 간다). `Normalize`는
  아핀이지 클램프가 아니고, 평평하게 놓인 정육면체는 거기에 신호를 싣지 않으므로, 우회하지 않고
  기록한다.

### 2. 데이터셋 — 수집할 것은 없고, 증명할 것이 하나 있다

**`observation.state`는 이미 큐브의 포즈를 싣고 있다.** `es_data::collect::to_lerobot`는 이 행을
env 0의 `qpos` 전체 뒤에 `qvel` 전체를 붙여 쓴다 — 이 씬에서는 `nq + nv = 13 + 12 = 25`이고 큐브의
free 조인트는 `qpos[6..13]`에 있다. 컬럼은 추가되지 않고, v2.1 라이터와 v3.0 익스포트는 그대로이며,
**`dataset_schema_hash`는 움직이지 않는다**. 움직인 것은 Observation IR이 그 행의 어느 구간을
읽는가다. `docs/api-notes/lerobot-dataset.ko.md`가 그 레이아웃과 그것이 왜 설계를 떠받치는지를
기록한다.

그 레이아웃이 bake를 정확하게 만드는 것이기도 하다: `qpos`의 `IndexRange`는 `capture`가
`StateView::qpos_of(0)`을 인덱싱하는 것과 같은 두 경계로 기록된 행을 인덱싱한다. 이 행은 상태의
재인코딩이 아니라 상태에 `qvel`을 덧붙인 것이다. 그래서 `es_eval::ObservationBake`는 분기 하나를
얻고(`Capture::Qpos(r) => row[r]`), `ObservationBake::new`는 V2b에서 거부되었던
`Option<&ModelInfo>`를 얻는다. `qpos` 범위는 실제로 돌았던 모델에서 와야 하기 때문이다.

**함께 오는 거절이 하나 있고, 그쪽이 중요한 절반이다.** 모델이 없으면 `Capture::Joints(dof)`는
"행의 앞 `dof`개"를 뜻하고, 선두일 수 있는 채널은 *하나*뿐이다. 두 번째 `JointState` 채널이
선언되어 있는데 모델이 없으면 어느 쪽이 선두인지는 문서에 없다 — 모델에 있다 — 므로
`input_sources`는 한 채널에 다른 채널의 값을 내주는 대신 이름을 대며 거절한다. 그 거절이 없었다면
데모는 `row[..7]`(팔의 여섯 각과 큐브의 `x`)을 특권 포트에 bake해 넣고 조용히 학습했을 것이다.

### 3. 특권 표시

**`sim_cube_pose`는 시뮬레이터 특권이다: 실제 SO-101에는 큐브가 어디 있는지 알려주는 센서가
없다.** `es_ir::task::ObsChannel`은 `source`와 `ty`만 갖는다 — 나머지는 Observation IR의 것이라고
§7.4가 명시한다 — 므로 **태그할 필드가 없고, 만들어내지도 않았다**. 표시는 `sim_` 접두사이며, Task
IR 채널, Observation IR 출력 포트, Learning IR 입력, 정책 계약이 그대로 이어받으므로 체인의 모든
구간과 `contract.json`에서 보인다. 관례는 이 파일과 픽스처 헤더, 설계 노트 7.14절에 문서화된다.

`sim.cube_pose`가 아니라 `sim_cube_pose`인 이유: 포트 이름은 로워링된 모듈에서
`forward(**inputs)`의 키로 파이썬에 도달하고, LeRobot 피처 이름에서 점은 네임스페이스 구분자다.

더 넓은 `joint_state`가 아니라 *두 번째* 포트로 두는 것이, 비전 패킷이 `joint_state`의 해시를
움직이지 않고 이 포트만 떼어낼 수 있게 한다. 팔의 여섯 각은 진짜 엔코더에서 나오고 있던 그대로
남는다.

### 4. 서버 단계의 학습 노브

의도적으로 V2b와 동일하다. 움직이는 변수는 관측 하나다.

* 시연 50개, 같은 데이터셋과 타일 — 상태 정책에 그 이상은 필요 없고, 재수집은
  `dataset_schema_hash`를 움직여 비교를 흐린다.
* `execute_chunk = 10`, `horizon = 16`, `replan_hz = 5`, `TemporalEnsemble { decay = 0.01 }` —
  선언 그대로. V6b가 평가에서 이를 존중하게 만들었으므로, 지금 바꾸면 V7a는 두 가지를 재는 측정이
  된다.
* 20,000 옵티마이저 스텝, `--batch 8 --lr 1e-4 --seed 0`, 1k / 5k / 20k 체크포인트,
  `--resident-gpu`(V5).
* nominal 평가는 시드 101–116, 같은 홀드아웃 16개. 전체 스위트는 최종 체크포인트에만. 영상은 V3와
  V1c가 만든 방식대로.

## context

허용 파일 범위:

```
tests/fixtures/visible-learning/{task,observation,learning,evaluation}.toml
crates/es/tests/cli.rs                 (픽스처 생성기, bake CLI 테스트)
crates/es/src/cmd/dataset.rs           (`es dataset bake`가 필요할 때 모델을 연다)
crates/es-eval/src/bake.rs             (`qpos` 분기와 `ModelInfo` 파라미터)
crates/es-eval/src/runner.rs           (`input_sources`: 모호성 거절)
crates/es-eval/tests/evaluation.rs     (비트 동일성 오라클, 상태 채널 둘)
docs/api-notes/lerobot-dataset{,.ko}.md
docs/design/visible-learning{,.ko}.md  (7.14절, 미해결 질문)
docs/packets/M5/V7a-privileged-state-policy{,.ko}.md
```

`python/es/train_act.py`는 범위 **밖**이며 변경이 필요 없다. baked 텐서와 `contract.json`의
`inputs`를 읽으므로 새 포트는 저절로 도착한다.

## spec

* **§5.1 규칙 6 — Task IR은 선언하고 Observation IR이 구현한다.** `sim_cube_pose`는
  `ObsChannel { source, ty }` 하나와 `TaskNode::ObservationSpec` 하나다. 모든 변환 —
  `Normalize`, 범위, 레이아웃 — 은 `observation.toml`에 있다. 신경망은 Task IR에 들어가지
  않는다(`TaskNode`에는 그런 변형이 없다).
* **§7.4 / `XIR-002`.** Observation IR의 `StateInput { source }`는 선언된 채널과 키가 맞아야 하고
  출력 타입이 선언된 타입과 호환이어야 한다. 커밋된 픽스처에 대해 `es ir check`가 둘 다 검사한다.
* **§7.2 / `INV-14`.** `Resize`도 `Crop`도 추가하지 않으므로 인트린식 변환 의무가 생기지 않는다.
  이미지 분기는 바이트 단위로 그대로다.
* **§8.2.** 특권 포트는 자기 `StateEncoder { Mlp[256], out_dim 512 }`와
  `Fusion { Concat, out_dim 512 }`의 세 번째 입력을 갖고, 로워링은 이미 `n` 입력을 지원한다.
* **`INV-16`.** 가중치는 `safetensors` 그대로이며 가중치 경로는 움직이지 않는다.
* **`INV-17`.** 새 트레이트 없음. `Capture`는 `es-eval`의 enum이지 확장점이 아니다.
* **§19.2.** 데이터셋 포맷은 움직이지 않는다. `dataset_schema_hash`는 그대로다.

## oracle

### 로컬, 그리고 이것이 게이트다

```
cargo run -p es -- ir check tests/fixtures/visible-learning/task.toml \
    tests/fixtures/visible-learning/observation.toml \
    tests/fixtures/visible-learning/learning.toml \
    tests/fixtures/visible-learning/deployment.toml \
    tests/fixtures/visible-learning/evaluation.toml
cargo test -p es-eval --test evaluation -- --nocapture bake state_channels
cargo test -p es --test cli -- --nocapture policy_lower dataset_bake
cargo test -p es -p es-eval -p es-data -p es-env -p es-policy
cargo xtask ci
```

* `a_baked_frame_is_bit_identical_to_what_capture_serves`(`es-eval`)는 이제 상태 포트 **둘**을
  돌린다: Task IR 채널의 앞-`dof` 읽기를 통과하는 `j0`, 그리고 `qpos` 범위가 **1**에서 시작하는
  `j1`을 `Capture::Qpos`로. 모든 프레임의 모든 출력 텐서가 정책이 받은 것과 바이트 단위로 같아야
  하며 허용 오차는 없다. `j1` 분기는 문제가 되는 성질을 가진 최소 크기의 데모 큐브다: "앞의 한
  값"을 읽는 bake는 거기서 `q[0]`을 내줄 것이고, 테스트는 `q[0] ≠ q[1]`을 단언하므로 우연히
  통과할 수 없다.
* `a_bake_refuses_two_state_channels_without_a_model`(`es-eval`)은 위 2절의 거절과, 모델을
  넘겨주면 같은 쌍이 해석된다는 것을 고정한다.
* `policy_lower_writes_the_module_and_contract`(`es`)는 넓어진 그래프를 로워링하고
  `lowering_hash`와 포트 목록을 출력한다. 로워링을 고정하는 골든이 없기 때문이다.
* `es dataset bake` CLI 테스트 둘은 이제 `MuJoCoCpuBackend`가 필요하고 — 데모의 두 번째 채널이
  씬의 `qpos` 범위로 해석되므로 — 없으면 이유를 출력하고 건너뛴다. 그 픽스처의 행 폭도 `es loop
  collect`가 한 번도 쓴 적 없는 여섯 값 대신 실제 `qpos ‖ qvel` 폭(25)이 된다.
* `dataset_bake_names_both_refusals_when_the_scene_cannot_be_loaded`(`es`)가 백엔드가 없는
  곳 — 즉 PR CI — 에서 그 둘을 대신하는 커버리지다. CLI의 폴백 전체를 돌린다: 모델 없는 해석이
  거절되고, 씬 적재를 시도하고, 두 절반이 한 메시지에 이름과 함께 나온다. 백엔드가 있으면
  위 두 테스트가 성공 경로를 덮으므로 비켜선다. `mujoco`가 없는 기계에서 닿을 수 없는 유일한
  단계는 모델 적재 성공이고, 서버 실행의 첫 명령이 정확히 그것이다.

### 서버, 2단계 — 이 순서대로, 다른 것은 움직이지 않는다

영어판 `docs/packets/M5/V7a-privileged-state-policy.md`의 "server, phase 2" 블록과 동일한
명령 순서를 쓴다: `ir check` → `dataset bake` → `policy lower` → `train_act.py` →
`policy pack` → nominal `eval run` → 전체 스위트 → `video mosaic` + `encode_video.py`.

각 체크포인트의 nominal 셀과 최종 스위트는 손실 곡선과 함께 설계 노트 7.14절에 들어간다.
**임계값을 통과시키려고 낮추는 수치는 없다**(`evaluation.toml`은 `success_rate ≥ 0.5`를 계속
요구한다).

## acceptance

1. 다섯 픽스처가 모두 `es ir check`로 검증·교차검사되고, 움직인 모든 해시가 여기와 7.14절에
   기록된다:

   | 슬롯 | 이전 (V6b) | 이후 (V7a) |
   |---|---|---|
   | `task_hash` | `aec2aea9e04cb0f4b224a49da120a1a39cd04499a6b929ea0bf0c2ccfaf51bf1` | `6cf826c167fef258414ce409ff51e310a1240fdb4b42b510351d5bae3e3f6b7b` |
   | `observation_hash` | `f4a50730ac95b91734c9678e75d9e6bc1845578bc2985e45b409e80f3355f6e0` | `6c18f4552064d24b16cffa770cfc71941566f2b0443fe88569d439f747c72daf` |
   | `learning_hash` | `82faf8c70c804b6fc104f436ad3327e90dd684280f7a7b7c3620797086971b54` | `5dac0a46f56198b1a6d04ead8ee43b18913356c56ee99b01a35decc66dc446f0` |
   | `policy_hash` | `01583940a350cd11a7ae0f304d5e8710db0494c210d904ebb62f1f0d2e4b88cd` | `c94c2732e215e4d8ac39cc0bd280c2d5495e5e36dbbc2da39f41ac952bd84a07` |
   | `evaluation_hash` | `5d70c21c0baa68f3aa9ccb64b2208823ba60e9ba47b16ffb7bc50f32313b9ff7` | `0259fd44ecc87c0e947afb98872984c0607efaf328c775d7f52f8ed00dcf041e` |
   | `lowering_hash` | `956abb67…ec6d` (V2b) | `fdd68ec43a7487dc2eefb373af669a2b6da2b692ff6729c9801242bced840719` |
   | `deployment_hash` | `3b2ad56806899d9d7f381bcfba03e65fdfd1fad0067b2a646f0f371bdf6dbb21` | **변동 없음** |
   | `compiler_hash` | `f2a02e84dbc5d186b9c809d9938ab70df70bcfb7e2e95fbc2344ddf1137270d6` | **변동 없음** |
   | `dataset_schema_hash` | — | **변동 없음**: 추가된 컬럼이 없다 |

   모두 파생된 값이다. `task.toml`, `observation.toml`, `evaluation.toml`은 선언된 생성기인
   `cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents`에서 나오며,
   그 안의 어떤 해시도 손으로 적지 않았다.
2. 비트 동일성 오라클이 포트 셋으로 통과하고 두 비공허성 단언이 모두 발동한다.
3. `cargo fmt --check`, 건드린 크레이트에 대해 두 피처 세트 모두 `clippy -D warnings`,
   `cargo xtask check-spec-refs`, `cargo xtask context-budget`, `cargo xtask ci`.
4. 2단계: 위 표들이 서버 실행으로 채워지고, 중단 규칙이 적힌 대로 적용된다 — 멈추라고 말할 때를
   포함해서.

## forbidden

* **`es-safety`.** 한 줄도 안 된다. 엔벌로프는 V6의 것이고 `deployment.toml`의 어떤 수치도
  움직이지 않는다. `deployment_hash`가 위 표에 있는 이유가 바로 움직이지 않았음을 보이기
  위해서다. `INV-11`, `INV-12`, `INV-13` 모두 유효하다.
* **`es-ir`.** 새 `ObservationNode`도, 새 `TaskNode`도, 새 `ObsSource` 변형도, `ObsChannel`의 새
  필드도 없다. 이 크레이트는 §1.5 목표치에 있고, 기존 노드로 채널을 표현할 수 없었다면 패킷은
  멈추고 보고했어야 했다. 표현할 수 있었다.
* **`es-policy` 로워링.** `lower_to_torch`는 건드리지 않는다. 넓어진 그래프는 이미 `n` 입력을
  처리하는 `Fusion{Concat}` 분기가 로워링한다.
* **이미지 분기.** `observation.toml`의 `ImageInput → Dequantize → Normalize` 체인과 그
  `ImageSpec`은 그대로이므로 비전 패킷이 온전히 물려받는다.
* **데이터셋 포맷.** 새 컬럼 없음, 라이터 변경 없음, 익스포터 변경 없음.
* **`python/es/train_act.py`.** 포트를 읽을 뿐, 이름을 알지 못한다.

## 측정 결과 (2단계, 2026-09-15)

기록은 설계 노트 `docs/design/visible-learning.md` 7.15절이고, 이것은 패킷의 판정이다.
`481e4d4`의 서버 트리, `~/artifacts/plan-v/v7a/`.

| 번들 | nominal `success_rate` | `envelope_violation_rate` | 스위트 (96 에피소드) |
|---|---|---|---|
| 1,000 step | 0/16 | 0.9853 | — |
| 5,000 step | 1/16 | 0.2523 | — |
| 20,000 step | 0/16 (샤딩 1/16) | 0.1460 (0.1628) | 4/96 |

- acceptance 표의 모든 해시가 예측대로 돌아왔고, `deployment_hash`는 움직이지 않았다.
- 큐브의 정확한 자세를 입력에 넣은 학습 손실은 모든 체크포인트에서 비전 전용 실행과 0.5 % 안에
  있다(20,000에서 `0.0175` 대 `0.0176`): L1 목적함수는 큐브가 어디 있는지 아는 정책과 모르는
  정책을 거의 구분하지 못한다.
- **중단 규칙: 발동.** `success_rate >= 0.5`의 acceptance는 충족되지 않았고 낮추지도 않았다. 다음
  용의자는 물리, 접촉 모델, 전문가의 궤적이지(7.15절 발견 8) 모델 크기나 학습 길이가 아니다.
  2단계(더 높은 해상도의 비전)는 다음 패킷이 아니다; `docs/ARCHITECTURE.ko.md` 28.9절은 그 조사
  앞에 외부 ACT(V8)를 대조 실험으로 둔다.
- 판정 밖의 발견 둘: torch 런타임에서 `--jobs N`은 `--jobs 1`과 바이트 동일하지 않고(열린 질문
  16), `cv2`는 `~/venvs/es`가 아니라 `~/venvs/es-lerobot`에 있어 위 서버 단계의 `encode_video.py`
  줄이 잘못된 인터프리터를 적고 있다.
- 실행 뒤 발견해 main에서 고친 회귀(`9087785`): `JointState` 채널이 둘이면 `es dataset bake`가
  Task IR의 저장소 상대 경로 `scene.path`를 여는데 그 경로는 저장소 루트에서만 풀리므로,
  `crates/es/tests/cli.rs`의 모델 기반 bake 오라클 둘이 `mujoco`가 있는 곳에서는 모두 실패했다.
  이제 `es dataset bake --scene <file.xml>`이 `es eval run --scene`과 같은 방식으로 씬을 지정하고,
  오라클은 그것을 넘긴다.
