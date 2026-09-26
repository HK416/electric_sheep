# M11 R9 — M11 리뷰에서 나온 작은 수정 다섯 가지

스펙: §7.2 / INV-14(그려진 fov는 에피소드의 내부 파라미터를 움직인다), §3.5(결정성 등급),
§1.2(check-scope), §1.4. `docs/reviews/M11.md`(S-4와 자잘한 지적들)에서 찾음. 각 항목은 독립적이며
자신의 커밋을 갖는다. 유형 B.

## 질문

**다섯 가지 각각을, 커밋된 해시, 골든, `.estraj`, 프레임, `report.json`의 바이트를 움직이지 않고
고칠 수 있는가?**

## 명세

1. **초점 계산을 하나로(R2 지적).** `RenderOverrides::image_spec`
   (`crates/es-env/src/randomize.rs`)는 `fx_f64 * focal`을 기록하는데, `drawn_frame`
   (`crates/es-env/src/render.rs`)은 `fx_f32 * (focal as f32)`로 투영한다. 둘은 수집기 사이드카에서
   f32 1 ULP만큼 다르다. 둘 다 함수 하나를 거치게 해서, 기록된 내부 파라미터가 프레임이 실제로
   투영된 값이 되게 한다. 렌더된 프레임은 움직이면 안 된다: 그 함수는 오늘 `drawn_frame`이
   계산하는 것을 계산하고, `image_spec`은 값을 거기서 가져온다. 커밋된 사이드카나 `layout.json`이
   움직이면 멈추고 보고한다.
2. **`FrameSink`가 렌더 오버라이드를 본다(R2 지적).** es-data의 `FrameSink`는
   `(&ModelInfo, &StateView)`만 받기 때문에, 수집기가 각 에피소드의 렌더 드로우를
   `RandomizationPlan::apply_render`로 다시 계산한다. env의 현재 `RenderOverrides`를 싱크에 넘기고,
   재계산을 지운다. 프레임은 여전히 비트 단위로 같아야 한다(R2의 패리티 오라클).
3. **실행이 어떤 결정성 등급으로 나왔는지 말한다(S-4).** `es eval run`, `es loop collect`,
   `es train`이 표준출력 한 줄 `determinism tier: <tier> (backend <name>)`을 찍는다. 등급은
   백엔드가 이미 보고하는 능력에서 가져온 §3.5 등급이다(MJWarp = 등급 2). `report.json`은
   바뀌지 않는다. 새 필드는 커밋된 보고서를 전부 움직이게 한다.
4. **`check-scope --base <rev>`(M10 지적).** `cargo xtask check-scope <packet.md> --base <rev>`는
   `git diff --name-only <rev>...HEAD`에 작업 트리를 더해 검사한다. `--base` 없으면 동작은 그대로다.
   `xtask/src/scope.rs`에 단위 테스트를 추가한다.
5. **불안정한 테스트 두 개.**
   * `eval_telemetry_publishes_every_tick_in_order`는 완전 병렬 실행에서 임시
     `es-policy-*.safetensors`를 잃어버린다. 공유되는 임시 이름(고정된 이름이거나, 공유
     디렉터리의 PID만 쓰는 이름)을 찾아 호출마다 고유하게 만든다. 이름이 제품 코드에 있다면
     거기서 고치는 것도 범위 안이다.
   * `es-telemetry`의 `a_connection_past_max_clients_is_refused`가 한 번 `ConnectionReset`으로
     실패했다. 거부된 연결에서 온 리셋도 거부다: 테스트가 오늘 기대하는 거부와 함께 이것도
     받아들인다.

## context

```
crates/es-env/src/randomize.rs
crates/es-env/src/render.rs
crates/es-env/src/env.rs
crates/es-env/tests/**
crates/es-data/src/collect.rs
crates/es-data/src/lib.rs
crates/es-data/tests/**
crates/es/src/**
crates/es/tests/cli.rs
crates/es-py/src/**
crates/es-eval/src/**
crates/es-telemetry/src/transport.rs
crates/es-policy/src/**
xtask/src/scope.rs
xtask/src/main.rs
docs/packets/M11/P-M11-R9-small-fixes.md
docs/packets/M11/P-M11-R9-small-fixes.ko.md
```

## 오라클

1. 항목 1: `crates/es-env/tests/`의 테스트가, 그려진 fov에 대해 수집기가 기록하는 `ImageSpec`과
   `drawn_frame`이 투영에 쓰는 것이 비트 단위로 같음을 보인다. R2의 프레임 패리티 테스트도
   여전히 통과한다.
2. 항목 2: R2의 수집기 == 평가기 == `Rollout` 프레임 패리티 테스트가 통과하고, 수집기는 더 이상
   `apply_render`를 부르지 않는다.
3. 항목 3: cli 테스트가 `mujoco-cpu`(등급 1)에 대한 표준출력 줄을 확인한다. 커밋된 `report.json`
   픽스처는 바이트 단위로 그대로다.
4. 항목 4: `cargo test -p xtask`. 이 브랜치에서
   `cargo xtask check-scope docs/packets/M11/P-M11-R9-small-fixes.md --base main`를 돌리면
   통과한다.
5. 항목 5: 각 테스트가 기본 병렬성의 `cargo test -p es --test cli` 아래서 20번 연달아 통과하고,
   텔레메트리 테스트도 마찬가지다.
6. `cargo xtask ci`, 수정 0건인 verify-goldens, fmt, clippy `-D warnings`.

## 수용

오라클 1–6. 보고서는 항목마다 한 줄, 커밋과 함께.

## 금지

`report.json`의 스키마, 어떤 커밋된 골든·픽스처·해시 바꾸기; `es-safety`; 항목 5가 이름 붙인 것
이상으로 테스트의 어서션을 느슨하게 하기.
