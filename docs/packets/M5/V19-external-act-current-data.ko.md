# M5 V19 — 현재 시연 데이터 위에서, 수리된 하네스로 돌린 LeRobot 자신의 ACT

설계 노트: `docs/design/visible-learning.ko.md` **7.27절**, 그리고 7.16(V8, 같은 파이프라인을
시연 50개와 옛 하네스에서), 7.22(V14, 시연 200개), 7.25(V17, 재계획 주기), 열린 질문 22–25.
스펙: §1.4, §5.3, §7, §8.1, §8.4, §8.7, §8.9, §9.2, §19.1, §25.1. API 노트:
`docs/api-notes/lerobot-act.ko.md`, `docs/api-notes/lerobot-dataset.ko.md`.
의존: V8(내보내기·가져오기·동등성 게이트), V14(시연 200개), V17(케이던스와 평가가 도는
Deployment IR).

## 이 패킷이 묻는 것

V8은 **바깥에서** 설계·학습된 정책이 여기서 동일한 의미론으로 도는지를 물었고, 의미론에는
비트 단위로 그렇다고, 과제에는 0/16, 0/16, 1/16으로 답했다. 그 숫자는 전부 그 뒤로 세 번
수리된 하네스 위에서 측정된 것이다.

* **V11** — 데모가 50 Hz Deployment IR을 두고 제어 루프를 200 Hz로 돌리고 있었다;
* **V12** — 정책이 행동의 근거로 삼은 관측이 *스텝 이후* 상태와 짝지어져 있었다;
* **V17** — 런타임이 선언된 5 Hz가 아니라 매 제어 틱마다 재계획해서, 청크의 0행을 제외한
  모든 행이 죽어 있었다.

게다가 V8의 체크포인트는 옛 케이던스로 모은 V1c의 시연 50개로 학습한 것이고, V14에는 200개가
있다. 그러니 V8의 판정 — "진짜 LeRobot ACT도 더 낫지 않으니 모델이 문제가 아니다" — 은 지금
트리의 무엇으로도 재현되지 않는 측정 위에 서 있다.

**V19는 V17에 대해 하나만 바꾼다: 학습기.** 같은 시연(V14의 200개), 같은 Task IR, 같은
Deployment IR과 Safety Plane, 같은 시드, 같은 예산, 같은 수용 기준. 정책은 LeRobot 0.6.1
자신의 ACT — CVAE, DETR 인코더/디코더, ImageNet 사전학습 ResNet18 — 이고 `lerobot-train`으로
학습해 `es policy import-lerobot`으로 가져온다.

## 결정과 근거

### 1. 상태 특징은 V17의 두 상태 포트를 이어 붙인 것이다

V17의 최고 정책은 상태 포트를 **둘** 읽는다. 팔의 관절각 6개와, V7a의 시뮬레이터 특권
`sim_cube_pose`(큐브 free joint의 `qpos[6..13]`). V8의 ACT는 앞의 하나만 읽었는데,
`lerobot.utils.feature_utils.dataset_to_policy_features`가 정책에 `observation.state` 특징을
정확히 하나만 주기 때문이고, V8은 *비전* 질문이었기 때문이다.

V8의 6차원을 그대로 쓰면 변수가 둘 움직인다 — 학습기 **그리고** 그것이 볼 수 있는 것. 그래서
두 포트를, 기록된 `observation.state` 행(`qpos || qvel`)이 이미 담고 있는 순서 그대로 이어
붙여 LeRobot이 허용하는 하나의 특징으로 만든다.

* 내보내기는 `--state-dim 13`. `qpos[0..13]`을 남기는데, 이는 관절각 6개 다음에 큐브의 7값
  free-joint 자세다. 플래그는 V8의 것이고 바뀐 것은 그 값뿐이다;
* Observation IR은 observation-v8의 그래프에 observation.toml의 큐브 가지를 붙이고 축 0의
  `ObservationNode::Concat`으로 합친 것으로, 커밋된 두 픽스처에서
  `python/es/make_v19_observation.py`가 쓴다. 두 가지 모두 `Normalize`는 항등
  `Range{0..1}`인데 이유는 V8의 것이다 — ACT는 자신의 `MEAN_STD` 통계를 체크포인트에 지니고
  순전파의 첫 연산으로 적용하므로, 여기서 두 번째 아핀 사상을 걸면 내보낸 parquet에도 똑같이
  걸어야 한다. `Concat`의 포트는 `Frame::Policy`다 — 팔 프레임의 `[6]`과 월드 프레임의 `[7]`은
  공통 프레임이 없고, `Frame::Policy`는 IR이 이미 어떤 프레임과도 호환된다고 정의해 둔 것이며,
  네트워크에 건네지는 특징 벡터란 바로 그것이다.

`es-ir` 변경 없음, 새 노드 종류 없음, 픽스처 수정 없음: `Concat`은 M1부터 `ObservationNode`의
변형이고 `es-compile`이 `Op::Join`으로 낮춘다.

### 2. 나머지는 전부 V17의 것, 그대로

`tests/fixtures/visible-learning/{task,deployment}.toml`은 트리에 있는 파일 그대로다 —
`max_episode_steps = 1800`, 제어 50 Hz, 추론 5 Hz, `horizon = 16`, `execute_chunk = 10`,
`TemporalEnsemble { decay = 0.01 }`, 같은 문서가 선언한 주기에 맞춰 V17이 진술한 240 ms 추론
예산, 그리고 수용 기준 `success_rate >= 0.5`. 평가는 `evaluation-v8.toml`의 `observation`
필드를 병합된 문서로 돌리고 시드 목록을 nominal 한 스위트로 줄인 것이며, 줄이는 일은 V14의
`mkcfg.py`가 한다 — V11 – V17이 쓴 것과 같은 트림이다.

`--policy.chunk_size=16 --policy.n_action_steps=16`인 이유는 Deployment IR이 horizon 16을
버퍼링하고 `es policy import-lerobot`이 이와 어긋나는 체크포인트를 거절하기 때문이다(V8 §4).
나머지는 전부 LeRobot의 ACT 기본값이고, 그것이 이 패킷의 요점이다.

### 3. V19가 고쳐야 했던 것, 그리고 왜 지금까지 보이지 않았는가

`es policy import-lerobot`은 Deployment IR의 `deadlines.inference_budget`을
`RuntimeHints::deadline_ms`와 `RuntimeHints::expected_latency_ms` **양쪽**에 써 넣고 있었다.
후자는 `es_ir::learning::RuntimeHints`에 "스펙 0.3 기준 장치에서의 *측정값*이지 약속이 아니다"
라고 적혀 있고, `LRN-052`는 `expected_latency_ms <= min(deadline_ms, 1000 / replanning_hz)`를
검사한다. 예산이 V8의 40 ms이고 재계획 주기가 200 ms인 동안에는 두 해석이 일치했다. V17이 같은
5 Hz 재계획에 대해 **240 ms** 예산을 진술하자, `LRN-052`는 가져온 모든 체크포인트를 "그 배치가
스스로 선언한 주기가 금지하는 지연을 주장한다"는 이유로 거절했다.

수정은 한 줄과 그것에 이름을 주는 함수다. 가져오기는 *상한* — 추론 예산과 재계획 주기 중 더
빡빡한 쪽 — 을 선언한다. `config.json`에 측정값이 없기 때문이다. V8의 값은 그대로이고
(`min(40, 200) = 40`), V17의 배치에서는 200 ms가 되며, 코드에 적힌 정직한 귀결은 이 경로에서
`LRN-052`가 잡을 것이 남아 있지 않다는 것이다. 같은 명령이 지어낸 숫자를 규칙이 검사할 수는
없다. 측정된 지연은 `es bench`에서 온다.

## context

허용 파일 범위:

```
crates/es/src/cmd/policy.rs               (declared_latency_ms 와 그 단위 테스트)
python/es/make_v19_observation.py         (병합된 Observation IR)
docs/design/visible-learning{,.ko}.md     (7.27절, 열린 질문 22-25)
docs/packets/M5/V19-external-act-current-data{,.ko}.md
target/plan-v/v19/                        (서버 결과의 추적되지 않는 미러)
```

## spec

* **§8.1.** 네트워크는 불투명하고 인터페이스는 타입이 있다. 번들의 `LearningGraph`는 계약의
  포트를 지닌 `LearningNode::PolicyBundle` 하나이며, 이는 V8이 확립한 형태다.
* **§8.4 / `XIR-010`, `XIR-022`–`XIR-024`, `LRN-052`.** 계약은 `config.json`에서 투영되고
  Deployment IR에서 채워진다. 가져오기가 *지어내는* 유일한 숫자는 이제 상한으로 선언되고,
  상한이라고 말해진다.
* **§8.7.** 전/후처리는 IR의 것으로 남는다: `Concat`은 Observation IR 노드이고, 청커·앙상블·
  역정규화기는 Deployment IR에 남는다. `PolicyRuntime`으로 옮겨간 것은 없다.
* **§8.9.** 동등성 게이트는 V8의 것이며, 이 프로젝트 자신의 데이터셋 프레임 위에서 tier 4
  (<= 1e-5)로 — 실제로는 0으로 — 측정된다.
* **§5.3.** `task_hash`, `deployment_hash`, 데이터셋 content 해시는 V17과 V14의 것이고,
  `observation_hash`, `learning_hash`, `policy_hash`는 이 패킷의 것이며 기록된다.
* **§7.** 하나의 Task IR은 여러 Observation IR을 지닌다. 이것이 세 번째다
  (`observation.toml`, `observation-v8.toml`, 병합본).
* **`INV-16`.** safetensors만. **`INV-17`.** 새 트레이트 없음.
  **`INV-11`/`INV-12`/`INV-13`.** `es-safety`는 건드리지 않았고 봉투 수치도 움직이지 않았다.

## oracle

### 로컬, 그리고 이것이 게이트다

```sh
cargo test -p es --bin es declared_latency
cargo xtask ci
```

### 서버 — `~/Projects/es-v19`, 산출물은 `~/artifacts/plan-v/v19/`

영문 패킷의 명령 블록과 동일하다. 애블레이션 1은 같은 순서에서 1단계의 소스를
`~/artifacts/plan-v/v15/ds-train`으로, 태그 `s13`을 `w13`으로 바꾼 것이다. 임계값을 넘은
체크포인트에서 돌린 스윕과 영상도 영문 패킷에 그대로 있다.
`~/artifacts/plan-v/v19/{pa,pb,pb2,pc,pd,pe,pf,pg2,ph}.sh`가 실제로 돌린 형태이며,
`probe.py`, `terms.py`, `summary.py`는 V17의 판독 스크립트다.

## acceptance

1. 새 체크포인트에서 동등성 게이트가 스펙 §8.9 tier 4로 통과하고, 측정된 `max_abs` /
   `max_rel`이 기록된다. **협상 불가** — 이것이 논지다.
2. 체크포인트마다: `success_rate`, V17의 에피소드별 양식으로 들어올림 / 옮김 / 놓음,
   `envelope_violation_rate`, `episode_length`를 학습 시드 1–16과 홀드아웃 101–116에 대해.
   최고 체크포인트에서 홀드아웃을 두 번 돌리고 두 `report.json`이 바이트 단위로 같아야 한다.
3. 어느 체크포인트에서든 홀드아웃 `success_rate >= 0.5`이면: 6-스위트 스윕, 쇼케이스 mp4 둘과
   모자이크를 V9 파이프라인으로. 그 아래면 정지 규칙이 발동한다 — 두 번째 변수는 없다.
4. `cargo xtask ci` 통과. 모든 Rust 변경은 테스트를 동반한다.

## forbidden

* **`es-safety`.** 한 줄도, `deployment.toml`의 숫자 하나도. `INV-11`, `INV-12`, `INV-13`은
  그대로이며, 외부 정책은 전문가와 모든 plan V 정책이 돌던 것과 같은 봉투 뒤에서 돈다.
* **`es-ir`.** 새 노드도, 새 필드도, 새 `ArchKind`도 없다.
* **수용 임계값.** 표를 통과시키려고 `success_rate >= 0.5`를 낮추지 않는다.
* **커밋된 픽스처.** `task.toml`, `deployment.toml`, `observation.toml`,
  `observation-v8.toml`, `evaluation-v8.toml`은 V17의 것이며 수정하지 않는다. 병합된
  Observation IR은 그것들 *옆에* 생성되지, 대신 들어가지 않는다.
* **재수집과 재베이크.** V14의 시연 200개가 데이터다. `es dataset bake`와
  `python/es/train_act.py`는 이 경로에 아예 없다.
* **`lower_act`, `lower_to_torch`.** 아키텍처는 체크포인트에 실려 있다.

## as built

오라클 서버에서 측정(RTX 4090, 학습과 레퍼런스는 `~/venvs/es-lerobot-cuda`, 평가는
`~/venvs/es`, 트리 `~/Projects/es-v19`, 산출물 `~/artifacts/plan-v/v19/`), 2026-09-16.
전체 기록: 설계 노트 **7.27절**.

**판정, 그리고 데모가 선언한 수용 기준.** Task IR이 지금 지닌 술어에 맞춰 수집된 시연 위에서,
LeRobot 자신의 ACT는 60,000스텝에서 **홀드아웃 시드 101–116에 대해 `success_rate = 0.8125`
(13 / 16)** 에 도달하고, Evaluation IR의 6-스위트 스윕은 변경되지 않은
`nominal success_rate >= 0.5`에 대해 `passed: true`를 보고한다. plan V에 데모가 생겼다. 쇼케이스
mp4 둘과 4x4 모자이크가 `target/plan-v/v19/`에 있다.

**이 패킷이 지목한** 시연(`~/artifacts/plan-v/v14/ds-train`) 위에서는 같은 레시피가 모든
체크포인트에서 **0 / 16**을 낸다. 이유는 논변이 아니라 측정이다. V15가 성공 술어의 그리퍼 항을
`> 0.85`로 올리고 시연 200개를 **다시 수집**했으므로, 패킷이 가리키는 35,918 프레임 집합은 턱이
0.573 rad에서 멈추는 놓기를 기록한다 — 그 항에 0.28 rad 모자란다 — 반면 V15의 36,960 프레임
집합은 0.832에서 끝난다. 두 실행 모두 보고한다.

| 실행 | 데이터 | 홀드아웃 `success_rate` |
|---|---|---|
| 패킷의 것 (`s13`), 20k / 50k / 100k | `v14/ds-train`, 35,918 프레임 | 모든 체크포인트에서 0.0000 |
| **애블레이션 1 (`w13`), 60,000** | `v15/ds-train`, 36,960 프레임 | **0.8125 (13/16)** |
| 애블레이션 1, 40k / 50k / 100k | 같음 | 0.6875 / 0.6250 / 0.0625 |
| 애블레이션 2 (`s6`), 100k | `v14/ds-train`, 6차원 상태 | 학습함, 평가하지 않음 |

수용 기준, 항목별로:

1. **충족, 비트 단위로, 여덟 번.** `max_abs 0e0`, `max_rel 0e0`, 청크 `[16, 6]`, 이 프로젝트
   자신의 데이터셋에서 기록한 프레임 위에서. 패킷 실행의 20k / 50k / 100k와 애블레이션 1의
   20k / 40k / 50k / 60k / 100k. `lerobot 0.6.1`, `torch 2.11.0+cu129`.
2. **충족.** 두 실행의 체크포인트별 표(학습 시드 1–16, 홀드아웃 101–116)가 7.27절에 있다. 각
   실행의 최고 체크포인트(`s13` 100,000, `w13` 50,000과 60,000)에서 홀드아웃을 두 번 돌렸고 모든
   `report.json` 쌍이 **바이트 단위로 동일**하다. `w13` 50,000 쌍은 서로 다른 `--jobs` 값에서.
3. **충족, 그리고 그 분기를 탔다.** 홀드아웃 `success_rate >= 0.5`이므로 6-스위트 스윕을 돌렸고
   (nominal 0.7500, `observation_delay` 0.8750, `backlash` 0.7500, `light_direction` 0.6250,
   `light_intensity` 0.3750, `torque_noise` 0.1875; `passed: true`), `es video showcase` /
   `es video mosaic`이 `v19-60k-nominal-00.mp4`, `v19-60k-nominal-11.mp4`,
   `v19-60k-mosaic.mp4`를 만들어 `target/plan-v/v19/`에 미러했다.
4. **충족.** `cargo xtask ci` 통과. 유일한 Rust 변경은
   `the_declared_latency_is_the_tighter_of_the_budget_and_the_replan_period`를 동반한다.

**패킷이 몰랐던 네 가지.**

1. **`LRN-052`가 가져온 모든 체크포인트를 거절했다.** `import-lerobot`이 Deployment IR의 240 ms
   `inference_budget`을 `expected_latency_ms`에 써 넣었는데, IR은 그것을 측정값이라 적어 두었고
   `LRN-052`는 200 ms 재계획 주기로 그것을 묶는다. 위 결정 3대로 고쳤고 V8이 가져오던 값은
   움직이지 않는다.
2. **패킷이 지목한 데이터셋은 V15가 대체한 것이고**, 그 차이가 결과를 정한다. 열린 질문 26.
3. **총량보다 스케줄이 중요하다.** 20,000스텝은 아무것도 놓지 않고, 40,000 – 60,000은 홀드아웃
   16개 중 11 – 13개에서 놓으며, 100,000은 1개에서 놓는다. 놓기는 183프레임 에피소드의 19프레임
   꼬리이므로 적합이 일시적이다. `--steps=100000`만 평가했다면 0.0625를 보고하고 데모를 놓쳤을
   것이다.
4. **다른 다섯 스위트가 문서에 있으면 한 스위트의 숫자가 움직인다** — 단독 0.6250 대 스윕의
   nominal 행 0.5625이며, 서로 다른 `--jobs` 값 셋에서 바이트 단위로 재현되므로 샤딩이 아니라
   문서다. 열린 질문 27.

**두 애블레이션, 그리고 각각을 돌린 이유.** `w13`(V15의 재수집 시연 200개에 같은 ACT)은 발견 2가
예측이었고 그것을 돌리는 것이 가장 싼 검증이었기에 존재한다. `s6`(같은 내보내기에 V8 자신의
6차원 관절 전용 상태)은 "학습기"와 "학습기가 볼 수 있는 것"을 분리하려고 존재하며, 100,000스텝
까지 학습하고 **평가하지 않았다**. 발견 2의 턱 천장이 그대로 적용되고 박스가 `w13`에 필요했기
때문이다. 외부 ACT에 특권 큐브 자세가 필요한지는 따라서 아직 열려 있고, 그 체크포인트는 서버에
있다.
