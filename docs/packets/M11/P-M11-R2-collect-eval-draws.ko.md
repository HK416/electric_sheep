# M11 R2 — 수집기와 평가기가 랜덤화된 과제를 그 추출대로 렌더한다

스펙: §28.14 규칙 4(같은 추출은 어디서 렌더하든 같은 프레임을 비트 단위로 낸다)와 규칙 1, §13.1, §10.4.
X5(2026-09-23)에서 나온 것: `Rollout`은 env마다 `RenderOverrides`를 적용하지만 `es loop collect --frames`와
`es eval run --frames`는 렌더 타깃이 있는 과제도 여전히 **추출 전** 장면으로 렌더한다. 그러면 DR 과제의
데이터셋이나 held-out 평가가 학습기가 본 적 없는 픽셀을 보게 된다. X5에서 함께 나온 것: `common.slang`에
남은 `Cross` 내장 함수 두 개(Möller–Trumbore와 면 법선)가 카메라 쪽에서 그랬듯 일부 드라이버에서 FMA로
합쳐질 수 있다. 유형 B.

## 질문

**수집기와 평가기가 각 에피소드를 그 에피소드의 추출대로 렌더하는가 — 같은 `(seed, env, episode, tick)`에서
`Rollout`이 렌더하는 것과 비트 단위로 같은 프레임으로 — 그러면서 렌더 타깃을 선언하지 않은 과제는 모두 오늘과
정확히 같은 바이트를 렌더하는가? 그리고 남은 `Cross` 호출 두 개가 추출된 자세 아래서도 정확한가?**

## 명세

* 수집기의 프레임 소스(`crates/es/src/cmd/loop.rs`)와 평가기의 프레임 소스(`es_eval::runner`,
  `crates/es/src/cmd/eval.rs`)가 `env.render_overrides(env)`(X5의 API)를 넘겨 `EnvRenderer::frame_with`를
  부르고, `Rollout`처럼 추출된 내부 파라미터를 프레임 사이드카에 쓴다.
* `common.slang`: Möller–Trumbore와 면 법선의 `cross`를 풀어 쓴 `es_cross`로 바꾼다. 단, 추출된 자세 아래서
  내장 함수가 어느 한 GPU에서라도 1 ULP 이상 어긋난다는 테스트가 있을 때만 바꾼다. 정확하다면 그 증거를
  기록하고 코드는 그대로 둔다.

## context

```
crates/es/src/cmd/loop.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es-env/tests/render_loop.rs
crates/es-render/slang/common.slang
crates/es-render/tests/render.rs
tests/fixtures/rl/**
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M11/P-M11-R2-collect-eval-draws.md
docs/packets/M11/P-M11-R2-collect-eval-draws.ko.md
```

## 오라클

1. `cargo test -p es --test cli --features render dr_collect_eval_frames_match_rollout` (GPU): 조명, rgba,
   카메라 자세, 시야각 타깃을 가진 vision reach 과제. `(episode, tick)`에서 `loop collect --frames`,
   `eval run --frames`, `Rollout`이 낸 프레임이 비트 단위로 같고, 사이드카의 내부 파라미터가 그 에피소드에
   기록된 `ImageSpec`과 같다.
2. 기존 프레임 골든과 궤적은 모두 그대로다(`so101_frame0`, 데모의 `.estraj`와 `events.json` 고정값).
3. 두 GPU에서 `cargo test -p es-render --test render dr_cross_is_exact`: 그 결과가 `common.slang`을 바꿀
   근거가 되거나, 바꿀 필요가 없음을 보여 준다.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## 수용

오라클 1–4. `renderer.md`(+ko)에 추출이 어디서 적용되는지 한 단락을 적는다.

## 금지

렌더 타깃이 없는 과제의 프레임을 바꾸는 것; 두 번째 추출 구현; 배치 렌더(X3b의 일).
