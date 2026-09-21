# M7 T2 — `es loop cycle`: 문서 하나와 원장 하나 아래의 collect → train → eval (→ showcase)

Spec: §13.1(루프는 일급 워크플로우다. 각 단계는 하나의 명령이자 하나의 아티팩트), §13.3
(모든 반복은 해시 체인으로 연결된다. "데이터·정책만 바꾸면서 평가 조건을 고정하는 것이
규율"이며, 바뀐 `evaluation_hash`는 비교를 무효화하고 도구가 경고한다), §19.2/§19.3,
§10.5, §28.9 규칙 1(무엇이든 학습되기 전에 하니스가 전문가를 통과시킨다), §28.10(T2).
확장할 설계 노트: `docs/design/training-recipe.md`(및 `.ko.md`) — 새 절 "사이클", 그리고
`docs/design/learning-loop.md` 4절(원장이 `train`과 `evaluate` 단계를 얻는다). **T1**
(`es train`, 레시피)과, 현재 있는 그대로의 `es loop collect`·`es eval run`·`es video
showcase`에 의존한다.

## 질문

T1은 학습을 명령 하나로 만들었다. §13.1이 그리는 사이클 — 수집, 학습, 평가, 관찰 — 은
여전히 사람이 경로와 해시를 손으로 꿰는 네 개의 명령이고, `loop.jsonl`은 `distill`에서
멈춘다: 데이터셋이 정책을 학습시켰다는 것도, 정책이 판정받았다는 것도 기록하지 않는다.
**문서 하나가 사이클 전체를 돌리고, §13.3이 원하는 해시와 함께 모든 단계를 원장에
추가하고, 움직인 `evaluation_hash`를 가로지르는 비교를 거부할 수 있는가?**

## spec

`es loop cycle --recipe cycle.toml [--out <dir>] [--dry-run] [--from collect|train|eval|showcase]`.

```toml
kind = "cycle"
scene = "tests/fixtures/mjcf/so101_pick_place.xml"

[collect]                        # 선택적. 없으면 기존 데이터셋에서 시작한다
policy   = "untrained.esb"       # Deployment IR이 plane이 되는 번들 (es loop collect --policy)
expert   = "so101-pick-place"    # 또는 학습된 정책 자신의 롤아웃이면 생략
episodes = 200
seed     = 1
frames   = true                  # es loop collect --frames <out>/collect/frames
# dataset = "runs/collect-001/ds"    # [collect] 대신: 재사용

[train]                          # T1의 레시피, 인라인 또는 경로로
recipe = "training.toml"         # 그 [dataset] root/frames는 이 사이클의 수집 출력이 덮어쓴다

[eval]
config    = "tests/fixtures/visible-learning/evaluation.toml"
checkpoint = "last"              # 또는 레시피의 checkpoint_at 중 한 스텝
jobs      = 6
frames    = true

[showcase]                       # 선택적
cell   = "nominal-00"
eye    = [0.66, -0.46, 0.52]
look_at = [0.14, -0.04, 0.04]
fov    = 36
width  = 1280
height = 720
```

* **단계들은 기존 함수를 in-process로 호출한다**(`cmd::r#loop::collect`, `cmd::train::run`,
  `cmd::eval::run`, `cmd::showcase::run`) — 어느 것도 다시 구현되지 않고, 두 번째 CLI 파서도
  없다. 그것들이 이미 띄우는 것(트레이너, 물리 서브프로세스, `--jobs` 워커)만이
  서브프로세스다. 출력은 `<out>/{collect,train,eval,showcase}/` 아래로 간다.
* **`loop.jsonl`이 두 종류의 단계를 얻는다**(`es_data::collect::LoopKind`): `Train { inputs:
  데이터셋 content/schema/split, 레시피의 identity_hash; outputs: training_hash, checkpoints
  (step → policy_hash) }`와 `Evaluate { inputs: policy_hash, evaluation_hash, deployment,
  observation; outputs: report blake3, passed, acceptance 스위트의 success_rate }`. 둘 다
  데이터셋 root의 원장(학습시킨 것, 또는 판정받은 것)과 `<out>/loop.jsonl`에 추가된다. 체인
  속성이 확장된다: `train.inputs.content == collect.outputs.content`,
  `evaluate.inputs.policy_hash ∈ train.outputs.checkpoints`.
* **§13.3의 경고가 이름을 들어 거부가 된다.** `<out>/loop.jsonl`은 이전 사이클들을 담고 있을
  수 있다(`--out`을 반복 2에 재사용). 새 `evaluate` 단계의 `evaluation_hash`가 같은 원장의
  이전 `evaluate` 단계의 것과 다르면, `--allow-new-evaluation`을 넘기지 않는 한 `es loop
  cycle`은 거부한다 — 그것이 §13.3의 "`evaluation_hash`가 바뀌면 비교는 무효"를 실행
  가능하게 만든 것이다. 반복 ≥ 2에서는 끝에 두 리포트에 대한 `es eval compare`가 출력된다.
* **하니스 우선(§28.9 규칙 1, M5-R1).** `[collect].expert`가 설정되면 사이클은 학습 **전에**
  *같은* `[eval].config`로 `es eval run`을 통해 전문가를 돌리고, 전문가의 acceptance가
  실패하면 학습을 거부한다 — 수집과 평가는 하나의 하니스를 공유해야 하고, 전문가가
  실패하는 하니스는 어떤 정책도 통과할 수 없는 하니스다. `--skip-expert-gate`가 있으며
  원장에 일탈로 기록된다.
* **`--dry-run`** 은 단계 계획(명령당 한 줄, `<out>`에 상대적인 경로)과 `train` 아래에
  중첩된 T1 계획을 출력한다. 바이트 단위로 동일한 골든.
* **`--from <stage>`** 는 중단된 사이클을 어떤 단계부터 재개한다. `<out>` 아래의 앞선 단계
  출력을 읽고, 없거나 그 해시가 원장과 어긋나면 거부한다.

## context

`cargo xtask check-scope`가 이 펜스를 읽는다. 아래 산문은 이유를 붙인 같은 목록이다.

```
crates/es-data/src/collect.rs
crates/es-data/src/training.rs
crates/es-data/src/lib.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/cycle.rs
crates/es/src/cmd/mod.rs
crates/es/src/cmd/train.rs
crates/es/src/cmd/eval.rs
crates/es/src/cmd/showcase.rs
crates/es/src/main.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/cycle.toml
tests/golden/train/plan-cycle.txt
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/design/learning-loop.md
docs/design/learning-loop.ko.md
docs/packets/M7/T2-loop-cycle.md
docs/packets/M7/T2-loop-cycle.ko.md
```

`collect.rs`(두 개의 `LoopKind`와 그 추가), `training.rs`(`Cycle` 레시피 스키마와 계획,
헤드리스), `cycle.rs`(신규, 얇게), `loop.rs`/`mod.rs`/`main.rs`(디스패치 + 도움말),
`train.rs`/`eval.rs`/`showcase.rs`는 **오직** 진입점을 `pub(crate)`로 노출하기 위해서만 —
필요하다면 argv 슬라이스 대신 옵션 구조체와 함께 — `cli.rs`(새 `cycle_*` 테스트), 픽스처
레시피와 그 계획 골든(승인된 생성기), 두 설계 노트, 이 패킷.

## oracle

1. `cargo test -p es --test cli cycle_dry_run_plan_is_the_golden` — 픽스처 사이클의
   `--dry-run` 출력이 `tests/golden/train/plan-cycle.txt`와 바이트 단위로 동일하다. Python 불필요.
2. `cargo test -p es-data loop_train_and_evaluate_steps_chain` — `collect → train → evaluate`
   합성 원장이 체인 검사를 통과하고, 입력 `content`가 `collect`의 출력이 아닌 `train`은
   이름을 들어 실패한다.
3. `cargo test -p es --test cli cycle_refuses_a_moved_evaluation_hash` — 해시 A의 `evaluate`
   단계를 담은 원장과 `[eval].config`가 B로 해싱되는 레시피는 두 해시를 모두 담은 메시지로
   거부되고, `--allow-new-evaluation`은 진행한다(dry-run 수준).
4. `cargo test -p es --test cli cycle_runs_the_expert_through_the_harness_first -- --ignored`
   — `ES_PYTHON`(mujoco)과 함께: 데모 픽스처에서 2에피소드 전문가 수집 + 40스텝 IR 경로
   학습 + 2시드 평가. `loop.jsonl`이 `collect`, `evaluate`(전문가 게이트), `train`,
   `evaluate`를 그 순서로, 연결된 해시와 함께 담는다. `report.json`이 존재한다. `ES_PYTHON`이
   없으면 이유와 함께 `SKIP`.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T2-loop-cycle.md`.

## acceptance

오라클 1–5(4는 오라클 서버에서). 서버에서 실제 사이클 하나: `[collect] episodes = 200`에
전문가, `[train]`은 20,000스텝 IR 경로(T3의 배치 모듈이 들어왔으면 그것, 아니면 현재 것),
`[eval]`은 커밋된 `evaluation.toml`에 `--jobs 6`, `[showcase]`는 `nominal-00`. 각 단계의
월클록과 총합을 §28.9의 "월클록이 어디로 가는가" 표 옆에 설계 노트에 기록한다(목표: 한
사이클 30분 미만 — §28.9의 속도 작업 중단 규칙. 측정 전까지는 `Target / Status:
unverified`). 전문가 게이트 자신의 리포트를 정책의 것 옆에 보관한다.

## forbidden

어떤 단계든 다시 구현하는 것. 두 번째 레시피 파서. `collect`·`train`·`eval`·`showcase`가
계산하는 것을 바꾸는 것. `crates/es-safety/**`; `crates/es-ir/**`; 새로 만든 것 외의 모든
픽스처와 골든; `docs/ARCHITECTURE*.md`. INV-16: safetensors만. INV-17: 새 trait 금지.
