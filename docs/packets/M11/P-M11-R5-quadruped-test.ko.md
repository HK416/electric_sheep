# M11 R5 — 사족보행 관찰-격차 테스트가 고정한 격차에 도달한다

스펙: §1.4(테스트는 돌거나 이유를 말한다), §8.7(런타임이 체크포인트를 낮춰진 그래프에 대해 검사한다).
M10 S-1이 찾고 `docs/reviews/M11.md` S-7이 다시 읽은 것:
`quadruped_eval_run_names_the_observation_gap`(`crates/es/tests/cli.rs`)가 `ES_PYTHON` 아래서
실패해, `cargo xtask ci`의 빠른 실패 테스트 단계를 멈춘다. 테스트는 가중치로
`b"es-m6-b1-untrained-placeholder"`를 쓴다. 런타임은 테스트가 고정하려는 관찰 격차에 이르기 전에,
그 체크포인트를 낮춰진 그래프(텐서 8개 누락)에 대해 거부한다. 유형 B.

## 질문

**테스트가 낮춰진 Learning IR과 텐서(이름, 모양, dtype)가 맞는 체크포인트를 만들어, `es eval run`이
체크포인트 검사를 통과하고 관찰 격차 자체의 메시지로 거부되거나 완료되도록 할 수 있는가?**

## 명세

* 테스트의 가중치는 낮춰진 그래프가 요구하는 모든 텐서를, 그 낮춰진 모양과 dtype으로(0으로 채워도
  된다) 갖춘 온전한 형태의 safetensors다. 런타임이나 `es-compile`이 이미 그래프의 파라미터 목록을
  얻는 데 쓰는 함수를 통해 만든다. 모양을 손으로 쓰지 않는다.
* 세 갈래는 그대로 둔다: `3` = SKIPPED, `1` = 관찰 격차 자체의 메시지로 거부됨, `0` = 격차가 닫힘.
  다른 이유로 나온 `1`은, 체크포인트 거부를 포함해, 여전히 테스트를 실패시킨다.
* 도우미가 필요하면 테스트 파일 안에 둔다. 낮춰진 파라미터 목록을 얻는 것이 도우미 없이는 불가능한
  경우가 아니면 제품 코드는 바꾸지 않는다. 그런 경우라면 가장 작은 `pub` 함수를 추가하고 보고서에
  이유를 밝힌다.

## context

```
crates/es/tests/cli.rs
crates/es/tests/common/**
docs/packets/M11/P-M11-R5-quadruped-test.md
docs/packets/M11/P-M11-R5-quadruped-test.ko.md
```

## 오라클

1. `ES_PYTHON`을 프로젝트 venv로 설정한 채:
   `cargo test -p es --test cli quadruped_eval_run_names_the_observation_gap -- --nocapture`
   가 통과하고 `RAN ... (refused by name)` 또는 `RAN ... (the run completed)`를 찍는다.
2. `ES_PYTHON` 없이도 통과한다(SKIPPED 갈래거나 같은 거부).
3. `cargo xtask ci`가 `ES_PYTHON`을 설정한 채 테스트 단계를 처음부터 끝까지 돌린다.
4. fmt, clippy `-D warnings`, `cargo xtask check-scope docs/packets/M11/P-M11-R5-quadruped-test.md`.

## 수용

오라클 1–4. 보고서는 테스트가 이제 도달하는 거부 메시지를 인용한다.

## 금지

테스트 삭제, `#[ignore]`, 또는 `1` 갈래의 메시지 일치 범위를 넓혀 체크포인트 거부를 받아들이는 것.
커밋된 골든과 픽스처.
