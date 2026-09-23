# M11 X6 — `Pt` 센서가 SVGF를 선언한다

스펙: §28.14 규칙 5(`Pt` 관측 노이즈는 선언된 문서 필드로만 다룬다 — `render.seed = "tick"`과
`render.svgf` — 관측 경로의 시간 누적이나 벤더 디노이저로는 결코 다루지 않는다)와 규칙 1과 파동 1,
§6(`render` 테이블), §15.3, §3.4. 설계 노트: `renderer.md` 4.3(SVGF; history가 없으면 luminance weight는
정확히 1.0)과 12.3(관측 경로에 accumulation이 없는 이유), 12.8(틱 시드); `visible-learning.md`
7.37(U5). W1a에 의존(병합됨). 유형 B.

## the question

`es_env::render::sensor_cfg`는 `SensorPath::Pt { spp, bounces }`를 `RenderPath::Pt { nee: true, restir:
false, svgf: false }`로 매핑한다 — SVGF는 꺼짐으로 하드코딩되어 있다. 단일 프레임 SVGF(depth/normal
edge-stopping, history 없이 luminance weight 1.0)는 이미 GPU == CPU 비트 동일이다. **센서가 `svgf =
true`를 선언할 수 있다면, 관측 프레임은 여전히 `(pose, episode, tick)`의 순수 함수로 남는가 — 수집기 ==
평가기 비트 동일, GPU == CPU — 커밋된 모든 task 해시는 불변인 채로?**

## spec

* `es_ir::task::SensorRender`에 `svgf: bool`(기본 `false`), `#[serde(default, skip_serializing_if =
  "is_false")]`이 더해지고, `canonical`이 **true일 때만** `seed` 뒤에 쓴다 — 정확히 W1a의 `seed`
  패턴(`crates/es-ir/src/task.rs:752-804`); `is_default()`가 이를 포함한다. `Pt`에서만 유효; `Rs`에서
  `svgf = true`는 새 코드를 가진 검증 에러다.
* `sensor_cfg`는 이를 `RenderPath::Pt { svgf }`로 매핑하며 `svgf_iterations`는 `RenderConfig`의
  기본값(4), `temporal: None`(불변: 이 경로에 accumulation 없음).
* `es video showcase --task`는 센서를 따른다(`seed`에서 그러듯).

## context

```
crates/es-ir/src/task.rs
crates/es-ir/src/validate*.rs
crates/es-ir-types/src/codes.rs
crates/es-ir/tests/sensor_render.rs
crates/es-env/src/render.rs
crates/es-env/tests/render_loop.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/task-pt-tick-svgf.toml
tests/fixtures/visible-learning/observation-pt-tick-svgf.toml
tests/fixtures/visible-learning/evaluation-pt-tick-svgf.toml
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M11/X6-sensor-svgf.md
docs/packets/M11/X6-sensor-svgf.ko.md
```

## oracle

1. `cargo test -p es-ir committed_task_hashes_are_unmoved_by_svgf` — `task.toml`, `task-pt.toml`,
   `task-pt-tick.toml` 불변; 명시한 `svgf = false`는 부재와 같게 해시; `true`는 해시를 옮김; `Rs` +
   `svgf`는 코드로 거부.
2. `cargo test -p es-env --features render --test render_loop pt_svgf_sensor_`(GPU): `seed = "tick"` +
   `svgf`로 같은 `(episode, tick)`의 collector-path와 evaluator-path 프레임이 비트 동일; 한 프레임에서
   GPU == CPU 레퍼런스 비트 동일; SVGF 프레임은 non-SVGF 프레임과 다름(뭔가 했다는 증거); SVGF 있을 때와
   없을 때의 ms/frame 측정.
3. `cargo test -p es --test cli` 불변; `es ir check`는 새 픽스처 셋을 받아들인다.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

오라클 1~4가 RTX 3060과 RTX 4090(GPU 큐)에서; `renderer.md`에 비용 표를 담은 12.9가 생긴다(+ko).

## forbidden

관측 경로의 `temporal: Some(_)`; 센서의 ReSTIR(비트 단위 아님); 벤더 디노이저; 커밋된 해시 옮기기; SVGF의
커널이나 상수 변경.
