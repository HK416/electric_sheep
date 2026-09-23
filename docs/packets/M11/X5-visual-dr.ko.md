# M11 X5 — 시각적 랜덤화: 조명, 색, ambient, geom 색, 카메라 자세와 시야각

스펙: §28.14 규칙 4(추출은 `(seed, env, episode, stream)`으로 키가 매겨지고 에피소드에 기록된다;
같은 추출은 같은 프레임을 비트 단위로 렌더한다, GPU == CPU; 추출된 시야각은 그 에피소드의
`ImageSpec` 내부 파라미터를 옮긴다, INV-14)와 규칙 1, 파동 2, §6.3(`Randomization`), §7.2
(`ImageSpec`), §10.2(평가 섭동은 그대로 있다), §15. 리뷰: `docs/reviews/M10.md` S-6(`Pt` 센서에는
방향광이 없어 `light_direction`이 거기서는 아무것도 측정하지 않는다). 설계 노트: `renderer.md`
(+ko), `batch-domains.md`(+ko). X3(렌더 오버라이드가 `Rollout`에 닿음)과 X4(`set_params` 이후
랜덤화 계획의 모양)에 의존. 유형 B.

## 질문

시각적 변형은 오직 두 평가 섭동으로만 존재한다 — `crates/es-eval/src/perturb.rs:64-257`의
`EnvRng`가 추출하는 `LightOverride { intensity, yaw_deg }` — 그리고 Task IR의 `Randomization`
타깃은 물리적인 것뿐이다(`crates/es-env/src/randomize.rs:141-200`). 소유자는 조명(세기, 방향, 색,
ambient), geom 색, 카메라 외부 파라미터와 카메라 내부 파라미터를 골랐다. **각각이 Task IR
`Randomization` 타깃이 되어 env마다 에피소드마다 추출되고, `Rs`와 `Pt`에서 렌더 시점에
적용되며(`Pt`에는 방향광과 함께), 비트 재현 가능하고 GPU == CPU이며, 이들 중 아무것도 선언하지
않는 모든 문서는 움직이지 않는 채로 그럴 수 있는가?**

## spec

* 타깃(`randomize.rs`의 문법에 render 클래스가 생긴다; 모르는 타깃은 이름으로 `Unsupported`로
  남는다): `light.intensity`(스케일), `light.direction`(요와 피치, 도 단위, 스트림 둘),
  `light.color`(RGB 스케일, 또는 고정 표를 거친 켈빈 단위 색온도), `light.ambient`(스케일),
  `geom.<name>.rgba`(채널별 스케일 또는 HSV 지터, 선택은 문서에 명시), `camera.<name>.pose`
  (평행이동은 미터, 회전은 카메라 자신의 축 기준 도 단위), `camera.<name>.fov`(수직 fov 스케일).
  추출은 env별 `RenderOverrides`로 들어가고 프레임 소스가 렌더 시점에 읽는다; 물리 스텝은 이를
  결코 보지 않는다.
* `LightOverride` 메커니즘은 es-eval에서 es-env로 옮겨져 유일한 구현이 된다; es-eval의
  `LightIntensity` / `LightDirection` 섭동은 이를 호출하며, 커밋된 보고서는 byte-identical로
  남는다.
* es-render: `RenderConfig`에 조명 색과 `Pt` 경로의 방향광이 생긴다(발광 geom 곁에서 한 방향을
  향한 NEE), Slang 셰이더와 CPU 레퍼런스 양쪽에. 기본값은 오늘의 프레임을 비트 단위로
  재현한다(흰색, `Rs` 기본 방향, 그리고 선언되지 않는 한 `Pt` 방향광 세기 0).
* 카메라: 포즈 추출은 센서의 외부 파라미터와 합성된다; fov 추출은 주점 기준으로 `fx`, `fy`를
  다시 스케일하며, 에피소드의 유효 `ImageSpec`(둘 다)이 에피소드 메타와 프레임의 `layout.json`에
  쓰여 내부 파라미터의 소비자가 추출된 값을 보게 된다(INV-14). 후속 `Resize` / `Crop`은 여전히
  이들을 변환한다.
* 모든 추출은 `param_scales` 곁의 `EpisodeMeta`에 기록된다.

## context

```
crates/es-env/src/randomize.rs
crates/es-env/src/render.rs
crates/es-env/src/env.rs
crates/es-env/src/episode.rs
crates/es-env/tests/**
crates/es-eval/src/perturb.rs
crates/es-eval/tests/**
crates/es-ir/src/task.rs
crates/es-ir/tests/**
crates/es-render/src/**
crates/es-render/slang/**
crates/es-render/tests/**
crates/es-py/src/rollout.rs
tests/golden/render/dr_*
tests/fixtures/rl/**
docs/design/renderer.md
docs/design/renderer.ko.md
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/packets/M11/X5-visual-dr.md
docs/packets/M11/X5-visual-dr.ko.md
```

## 오라클

1. `cargo test -p es-env visual_randomization_` — 각 타깃이 파싱되고 해석된다; 같은
   `(seed, env, episode, stream)`은 같은 값을 추출한다; 다른 에피소드는 다르다; 선언되지 않은
   타깃은 해시를 움직이지 않는다.
2. `cargo test -p es-render --test render dr_`(GPU) — 각 타깃에 대해 추출된 프레임: 그 경로의
   패리티 규칙에서 GPU == CPU 레퍼런스(오늘의 규칙이 0이라 하는 곳은 0 ULP); 기본(추출 없음)
   프레임은 기존 골든 전체와 byte-identical; 새 `dr_*` 골든은 CPU 레퍼런스에서만; 방향광이 있는
   `Pt`는 빛을 받는다(하늘만 있을 때 0이던 곳이 0이 아니게).
3. `cargo test -p es-eval` — `light_intensity` / `light_direction`을 쓰는 커밋된 보고서는
   이동 후에도 byte-identical.
4. `cargo test -p es-env camera_fov_draw_moves_intrinsics` — 에피소드에 기록된 `fx`, `fy`는
   공칭값 × 추출된 스케일과 같다; 후속 `Resize`가 이를 변환한다(INV-14).
5. fmt, clippy `-D warnings`, check-scope, verify-goldens; 두 GPU 모두.

## 수용

오라클 1~5; `renderer.md`에 방향광과 색 절이, `batch-domains.md`에 render-target 문법이
생긴다(+ko).

## 금지

텍스처 / 머티리얼(M7 R6); occluder; 커밋된 프레임·보고서·해시 변경; 물리가 렌더 추출을 보는 것;
벤더 디노이저.
