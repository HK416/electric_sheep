# M10 W0b — 호스트 libm 없는 `scene_hash`, 그리고 옮겨진 해시 아래서 다시 잰 기록

스펙: §3.2(2026-09-22에 추가된 단락: 오프라인 에셋 경로의 `f64`는 순수 Rust `libm` 포트인
`es_math::approx::{sin_cos_f64, acos_f64}`를 거친다; `DET-010`이 이름하는 것은 호스트 함수다), §3.4,
§5.3(`scene_hash` → `task_hash` → 전부), §28.13 규칙 1(해시 수정은 해시만 옮긴다)과 파동 0, §28.9 규칙 2
(무효화된 수치는 지우지 않고 표시한다), §1.4(골든 파일은 CI 읽기 전용). 리뷰: `docs/reviews/M8.md` S-2와
사람 결정 "플랫폼 간 `scene_hash`" — 소유자가 2026-09-22에 **새 해시 아래서 재수집·재학습·재채점**을
골랐다. 설계 노트: `docs/design/transcendental.md`(`f64` 단락이 생긴다), `visible-learning.md`(7.36행),
`rl-continuation.md` 7절(다시 잰 A0 행). 유형 B, D 절반.

## 질문

`crates/es-assets/src/mjcf/orient.rs:56`(`f64::sin_cos`, 모든 `euler=`와 `axisangle=`), `:75`(`acos`,
`zaxis=`), `crates/es-assets/src/urdf.rs:716-718`(`sin_cos`, `rpy`)은 플랫폼의 libm을 부르므로 같은
`so101_pick_place.xml`이 Windows와 Linux에서 다른 `scene_hash`가 되고(측정됨, M8/S4d)
`crates/es/tests/cli.rs:12041-12057`이 그것을 우회한다. **그 세 호출을 순수 Rust `libm` 포트로 돌리면
`scene_hash`가 모든 플랫폼에서 한 숫자인가 — 그리고 수정이 SO-101의 모든 `task_hash`를 옮기므로, 옮겨진
해시 아래서 재수집·재학습·재채점한 데모와 reach 과제가 커밋된 모든 수치를 비트 단위로 재현하는가?**

## 명세

* `crates/es-math/Cargo.toml`에 `libm`(이미 `Cargo.lock`에 있고 태생이 `no_std` — `es-safety`가
  `embedded-runtime.md`에서 택한 길). `crates/es-math/src/approx/mod.rs`에 `pub fn sin_cos_f64(x: f64)
  -> (f64, f64)`와 `pub fn acos_f64(x: f64) -> f64`, `libm::sincos` / `libm::acos`의 얇은 래퍼, 문서:
  `f64`, CPU 전용, Slang 미러 없음, 오프라인 에셋 경로 전용; 비트는 플랫폼이 아니라 크레이트가 고정; ≤ 2 ULP
  다항식 계열이 아님. 단위 테스트가 `sin_cos_f64(1.57 * 0.5)`, `sin_cos_f64(-0.55 * 0.5)`,
  `sin_cos_f64(core::f64::consts::FRAC_PI_4)`의 비트 패턴(`to_bits()` hex)을 고정한다 — SO-101 장면의
  두 half-angle과 대조군 — Linux 실행이 같은 상수를 단언하도록.
* `orient.rs`의 `axis_angle`·`from_zaxis`, `urdf.rs`의 `quat_from_rpy`가 래퍼를 부른다; `orient.rs:7-8`
  헤더("스펙 3.2의 제한이 적용되지 않는다")는 왜 적용되는지(비트가 `scene_hash`에 들어간다)로 고쳐 쓴다.
  `f64::sqrt`는 그대로(IEEE, 정확히 반올림).
* **기존 생성기로 한 번 재생성**(`crates/es/tests/cli.rs`의 `generate_pt_fixtures`,
  `generate_rl_learning_document`, `generate_augmented_observation_fixture`, `generate_train_goldens` … —
  각자의 env 플래그로 실행; 생성기가 없는 커밋 문서는 손으로 바이트를 고치지 말고 같은 `#[ignore]` 방식의
  생성기를 더한다): SO-101 task 문서 넷(`tests/fixtures/visible-learning/task.toml`, `task-pt.toml`,
  `tests/fixtures/rl/task-reach.toml`, `task-reach-delta.toml` — `scene_hash`만; 다른 바이트는 불변),
  Observation IR 일곱의 `task_ref`, Evaluation IR들과 task·observation 해시를 고정하는 Learning IR, 해시를
  인용하는 학습 레시피 헤더 주석, `crates/es-ir/tests/{joint_quantity,sensor_render}.rs`·`crates/es/tests/cli.rs`
  (`9581`과 `eb6efefa`, `d546b808`, `b5d3b813`, `4ced8547`, `7caac85d`, `967ea296` …이 나오는 모든 곳)의
  고정 상수. `task.scene.scene_hash = scene.scene_hash()` 덮어쓰기 셋(`cli.rs:3372`, `:11199`, `:12056`)과
  `12041` 주석은 없앤다: 테스트는 이제 등식을 단언한다 — 그 단언이 결함의 오라클이다.
  `tests/fixtures/quadruped/task.toml`(`xyaxes`만, `sqrt`)이 대조군: 변경 전에 해시 불변을 고정한다.
* `tests/golden/**`은 움직이지 않는다(`cargo xtask verify-goldens`); `tests/golden/rollout/so101_100steps.json`은
  옛 불일치를 가로질러 Windows에서 생성되고 Linux에서 재현됐으므로 이 수정을 견뎌야 한다 — 아니라면 재생성이
  아니라 발견이다.
* **서버(D 절반)** — 브랜치의 `git archive`를 `~/Projects/es-w0b`에 새로, 아티팩트 `~/artifacts/plan-w/w0b/`,
  두 큐 락(`~/artifacts/plan-w/queue/{gpu,cpu}.lock`)을 내내 쥔다(W0a의 측정이 CPU 락을 놓은 뒤); 모든
  스테이지는 `nohup` + `<stage>.start/.end/.done/.log`:
  1. 커밋된 장면 다섯의 Linux `scene_hash` == 재생성 문서의 값(결정적 오라클), 고정 비트 패턴 셋.
  2. V15 재수집: `es loop collect` 200 에피소드, seed 1, 전문가, `--frames`, 재생성 문서 아래 →
     `~/artifacts/plan-w/w0b/v15/ds-train`; 데이터셋 `content` 다이제스트와 모든 프레임을
     `~/artifacts/plan-v/v15/ds-train` 옆에(기대: 동일; Linux에서 물리는 안 바뀌었다; 바뀌었다면 쿼터니언이
     움직인 ULP가 노트에).
  3. U3 재학습: `training-u3.toml`의 레시피를 새 데이터셋과 재패킹한 untrained 번들(재생성된
     `observation-augmented.toml`, `learning-pretrained.toml`)로 → 체크포인트 텐서를
     `~/artifacts/plan-v/m7-u/U3/train/checkpoints/20000.esb` 옆에(기대: 비트 동일; 같은 데이터, 같은 시드).
  4. U3′ 채점: 재생성된 `evaluation-augmented`(held-out 시드 16 × 스위트 6)와 train-seed 문서, `--jobs 4`,
     전문가 게이트 → `report.json`을 `~/artifacts/plan-v/m7-u/U3/holdout/report.json` 옆에(기대: 해시 빼고
     동일: 0.5625 / 0.6668 / 1092.1).
  5. reach A0 재측정: `training-reach.toml`(`learning-reach.toml`, 64×64) 시드 셋 × 4,000회,
     `evaluation-reach.toml` → 0.4167(0.5625 / 0.3125 / 0.3750) 옆에, CPU 큐, 셋 동시.
  행: `visible-learning.md` 7.36("플랫폼 안정 `scene_hash` 아래서 다시 잰": 옛 해시와 새 해시를 나란히,
  수치마다 "비트 동일" 또는 이름 붙인 편차), `rl-continuation.md` 7절(A0 행, 같은 규칙). wall-clock 기록;
  안 잰 9-metric은 `Target / Status: unverified`.

## context

```
crates/es-math/Cargo.toml
crates/es-math/src/approx/mod.rs
crates/es-assets/src/mjcf/orient.rs
crates/es-assets/src/urdf.rs
crates/es-assets/tests/**
crates/es-ir/tests/**
crates/es-eval/tests/**
crates/es/tests/cli.rs
tests/fixtures/visible-learning/**
tests/fixtures/rl/**
tests/fixtures/quadruped/**
Cargo.lock
docs/design/transcendental.md
docs/design/transcendental.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M10/W0b-scene-hash-libm.md
docs/packets/M10/W0b-scene-hash-libm.ko.md
```

## 오라클

1. `cargo test -p es-math approx_f64_bit_patterns_are_pinned` — 상수 셋.
2. `cargo test -p es-assets` — `orientations.xml`과 SO-101 장면 해시가 이 플랫폼(Windows)에서 고정 hex와
   같고 **그리고** 같은 테스트가 서버(Linux)에서 녹색; quadruped 해시 불변.
3. `cargo test --workspace` — 재생성된 모든 핀 녹색; `cli.rs`의 scene-hash 등식 단언(덮어쓰기 없음)이 두
   플랫폼에서 녹색.
4. `cargo xtask verify-goldens` — 움직인 골든 없음.
5. 위 서버 스테이지 1~5, 두 행 작성, 아티팩트 `~/artifacts/plan-w/w0b/`.
6. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W0b-scene-hash-libm.md`.

## 수용

오라클 1~6; `transcendental.md` 단락; 7.36행과 A0 행과 한국어 짝; 이 패킷의 노트 절에 옛 → 새 해시 표 하나.

## 금지

`f32` `approx` 계열이나 `crates/es-math/slang/**`를 건드리는 것; `tests/golden/**` 편집; `docs/ARCHITECTURE*.md`
(단락은 들어가 있다); 생성기가 있는 커밋 문서를 손으로 고치는 것; 물리·이미터·수치의 의미 변경;
`tests/fixtures/quadruped/task.toml` 해시 이동; `es-render`, `es-env`, `es-physics-*`(다음 `es-assets` 변경은
W2a 것이며 이 패킷 머지 뒤 시작한다).
