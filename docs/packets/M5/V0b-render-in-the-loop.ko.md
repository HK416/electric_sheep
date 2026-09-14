# M5 V0b — env 루프 안의 렌더러: 애니메이션, 프레임 디스크 기록, 이미지 관측 포트

설계 노트: `docs/design/visible-learning.md` 섹션 7; 섹션 2.2를 먼저 읽을 것 — `es-render`는 **어떤
크레이트도 의존하지 않는** 완성된 헤드리스 섬이고, 거기 나열된 네 구멍이 이 패킷이 닫는 것이다.
V0, V4와 독립; V1, V2, V3를 막는다.

이 패킷은 플랜 V의 요청 순서에 자리가 없었기에 새로 생겼다: 저장소의 어떤 것도 렌더러를 env 루프에
연결하거나, 프레임을 디스크에 쓰거나, Observation IR 이미지 입력을 받아들이지 않는다.

## context

```
crates/es-render/src/scene.rs
crates/es-render/src/atlas.rs
crates/es-render/src/lib.rs
crates/es-render/tests/render.rs
crates/es-env/src/render.rs
crates/es-env/src/env.rs
crates/es-env/src/domains.rs
crates/es-env/src/lib.rs
crates/es-env/Cargo.toml
crates/es-eval/src/runner.rs
crates/es-eval/Cargo.toml
crates/es-eval/tests/evaluation.rs
crates/es-env/tests/render_loop.rs
tests/golden/render/so101_frame0.bin
tests/golden/render/so101_frame0.json
Cargo.lock
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V0b-render-in-the-loop.md
docs/packets/M5/V0b-render-in-the-loop.ko.md
```

메모: `es-render/src/scene.rs`에 함수 하나, `atlas.rs`에 `write_to` 하나가 추가된다.
`es-env/src/render.rs`는 새 파일이며 `es_render`를 이름으로 부르는 유일한 파일이다. `env.rs`와
`domains.rs`는 선택적 필드와 이미지 포트를 얻는다. `runner.rs`는 `EvalError::Plan` 분기 하나를 바꾼다.
`es-env`와 `es-eval`은 선택적 `render` 피처만 얻으며, 꺼져 있으면 두 크레이트 모두 지금과 똑같이
빌드된다.

## spec

- §4.2: `es-env`는 레이어 9, `es-eval`은 10이고 `es-render` (5)와 `es-gpu` (2)는 둘 다 그 아래이므로
  `cargo xtask layering`을 통과한다. `es-render`는 의존성을 **얻지 않는다**: `StateView`가 아니라 포즈
  맵을 받으므로 레이어 5는 여전히 `es-physics-core` (3)를 모른다.
- §15.2: 아틀라스 하나, 타일 N개, 닫힌 형식 `tile_origin`; 프레임은 기존 `Atlas::read_tile`에서 나온다.
- §15.4는 이 빌드에 구현되어 있지 않다 (TLAS/BLAS 없음, `crates/es-render/src/lib.rs:11-14`); 이 패킷도
  그것을 바꾸지 않으며, 설계 노트가 재테셀레이션 한계를 명시한다.
- §7.2, §26.1, INV-14: 렌더러의 `ImageSpec` 서브셋은 Observation IR이 선언한 `ImageSpec`과 **대조**될
  뿐 그것을 대체하지 않고, 맞추려고 재샘플링하지도 않는다
  (`crates/es-render/src/renderer.rs:292-302`). 불일치는 필드 이름을 밝히는 에러다.
- §3.4, §3.5: 결정성 주장은 CPU 레퍼런스 경로다; GPU 경로는 기존 GPU 오라클을 이미 통과하는 장치에서
  그것과 비트 비교되며, 그 외에는 아무것도 주장하지 않는다.
- §10.1: 러너가 제공할 수 없는 이미지 입력은 0이 되지 않고 에러로 남는다 — 이 패킷은 조건을 좁힐 뿐
  거부를 없애지 않는다 (`crates/es-eval/src/runner.rs:421-428`).
- §1.5: 현재 `es-env` 2,060줄, `es-eval` 2,139줄; 이 패킷은 네 크레이트 합계 약 800줄 이하로 잡는다.
- INV-17: `EnvRenderer`는 구체 구조체다. 트레이트는 추가되지 않는다.

## oracle

```
cargo fmt --check
cargo clippy -p es-render -p es-env -p es-eval --all-targets -- -D warnings
cargo clippy -p es-env -p es-eval --features render --all-targets -- -D warnings
cargo test -p es-render
cargo test -p es-env
cargo test -p es-env --features render --test render_loop
cargo test -p es-eval --features render
cargo xtask context-budget
cargo xtask layering
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

레퍼런스 — GPU 구간. Vulkan 장치와 `slangc`가 있는 머신에서:

```
cargo test -p es-env --features render --test render_loop -- --nocapture
```

CPU 경로는 어디서나 돌고 골든이다. GPU 경로는 장치나 `slangc`가 없으면
`crates/es-render/tests/render.rs:57-72`가 이미 하는 그대로 `SKIP <test>: <reason>`을 출력하고,
실행되면 `RAN render_loop_gpu`를 출력한다.

골든 생성 (기본 무시, `crates/es-render/tests/render.rs:165-195`의 `generate_goldens`가 이미 그렇다):
새 `tests/golden/render/so101_frame0.{bin,json}`은 V0의 픽스처 장면을 고정 `qpos`에서 **CPU** 레퍼런스로
렌더해 만든다. 어떤 드라이버의 산술도 커밋되지 않는다.

`crates/es-env/tests/render_loop.rs`:

- `poses_from_state_move_the_triangles` — 포즈 맵에서 어떤 바디를 알려진 오프셋만큼 옮기면 그 바디의
  geom 삼각형이 전부 정확히 그만큼 움직이고 다른 삼각형은 움직이지 않는다.
- `an_absent_body_keeps_its_scene_pose` — 바디가 빠진 포즈 맵은 그 바디에 대해 `from_scene`을
  재현한다. 부분적으로만 아는 상태가 원점이 아니라 정적 장면으로 저하된다.
- `cpu_frame_matches_the_golden` — 고정 `qpos`에서 V0 픽스처로 렌더한 프레임이
  `tests/golden/render/so101_frame0.bin`과 바이트 동일하다.
- `two_runs_of_the_same_state_are_byte_identical` — CPU 경로에서 설계 노트 섹션 9의 결정성 주장.
- `gpu_frame_matches_the_cpu_frame` — 이 장치에서 비트 동일; 아니면 `SKIP`.
- `frames_written_to_disk_round_trip` — `write_to` 후 되읽으면 `Tile::to_bytes`와 같고, 사이드카의
  `dtype`, `shape`, `layout`이 기존 `tests/golden/render/*.json` 스키마와 일치한다.
- `the_declared_image_spec_is_checked_not_coerced` — `ImageSpec`이 크기·채널 수·색 공간에서 렌더러와
  다른 Observation IR `ImageInput`은 필드 이름을 밝히는 에러다; 리사이즈는 일어나지 않는다.
- `without_a_renderer_an_image_input_is_still_refused` — 피처가 꺼져 있거나 `EnvRenderer`가 없으면
  `crates/es-eval/src/runner.rs`는 지금과 같은 메시지로 같은 `EvalError::Plan`을 반환한다. 무엇도 0으로
  채워지지 않는다.
- `an_env_without_a_renderer_is_unchanged` — `--features render`로 돌리되 `EnvRenderer`가 없는 기존
  `es-env` 테스트 스위트가 피처 없는 실행과 바이트 동일한 에피소드 기록을 만든다.

## acceptance

```rust
// crates/es-render/src/scene.rs
impl TriScene {
    pub fn from_scene_with_poses(
        scene: &es_assets::scene::SceneDesc,
        world: &std::collections::BTreeMap<es_core::StableId, es_math::Pose>,
    ) -> Result<Self, RenderError>;
}

// crates/es-render/src/atlas.rs
impl Tile {
    /// `<dir>/<stem>.bin` (이 타일의 `to_bytes`)과 `<dir>/<stem>.json`
    /// (`dtype`, `shape`, `layout`)을 쓴다. `tests/golden/render/*`가 이미 쓰는 레이아웃.
    pub fn write_to(&self, dir: &std::path::Path, stem: &str) -> std::io::Result<()>;
}

// crates/es-env/src/render.rs   (cfg(feature = "render"))
pub struct EnvRendererCfg {
    pub camera: es_core::StableId,
    pub width: u32,
    pub height: u32,
    pub channel: es_render::Channel,
    pub path: es_render::RenderPath,
    pub frames_dir: Option<std::path::PathBuf>,
}
pub struct EnvRenderer<'gpu> { /* Renderer, cfg, 캐시된 TriScene, 프레임 카운터 */ }
impl<'gpu> EnvRenderer<'gpu> {
    pub fn new(gpu: &'gpu es_gpu::Gpu, scene: &SceneDesc, cfg: EnvRendererCfg) -> Result<Self, EnvError>;
    /// `state`의 `xpos`/`xquat`로 `env`에 대해 장면을 다시 포즈하고, 렌더하고, 타일을 반환한다.
    /// `frames_dir`이 설정되어 있으면 `<frames_dir>/<NNNNNN>.bin` + `.json`을 쓴다.
    pub fn frame(&mut self, model: &ModelInfo, state: &StateView<'_>, env: u32)
        -> Result<es_render::Tile, EnvError>;
    /// 호출자가 선언된 스펙과 대조할 수 있도록 하는 렌더러의 `ImageSpec` 서브셋.
    pub fn image_spec(&self) -> es_render::ImageSpec;
}
```

- `Env`는 피처 뒤에서 `render: Option<EnvRenderer<'_>>`를 얻고, `domains.rs`의 관측 입력 맵은 그것이
  `Some`일 때 이미지 포트를 얻는다. `None`이면 지금과 바이트 동일하게 동작한다.
- `es-eval`의 러너는 렌더러가 있고 선언된 `ImageSpec`이 일치할 **때에만** `ImageInput`을 받아들인다;
  아니면 기존 `EvalError::Plan` 메시지 그대로다.
- `CpuPlan::run`에 넘기는 이미지 텐서는 선언된 레이아웃의 타일 바이트다. `es-env`에서 재샘플링·색
  변환·채널 재배열은 일어나지 않는다: 불일치는 에러이고, 모든 변환은 Observation IR 노드다 (§7.2).
- 새 트레이트 없음; `HashMap` 없음; 네 크레이트 어디에도 새 외부 의존성 없음; `--features render` 유무
  모두에서 `cargo xtask layering`과 `cargo xtask context-budget` 통과.
- `es-render`, `es-env`, `es-eval` 합계 약 800 소스 줄 이하.

## forbidden

- `crates/es-ir`, `crates/es-ir-types` — 새 관측 노드, `ImageSpec` 필드 금지.
- `MultiViewPack` 구현 (`crates/es-compile/src/plan.rs:545`가 거부한다; 데모는 카메라 하나를 쓴다).
- `es-render`에서 `Shape::Mesh`나 `HeightField` 해석 (`crates/es-render/src/scene.rs:201-202`): V0의
  픽스처는 프리미티브 전용이고 에셋 리졸버는 별도 패킷이다.
- PNG·JPEG·비디오 인코더 또는 새 이미지 의존성: 프레임 포맷은 기존 원시 `.bin` + `.json` 사이드카다
  (설계 노트 섹션 7.2).
- `crates/es-eval/src/perturb.rs` — `LightIntensity` / `LightDirection` 활성화는 V3다.
- `crates/es-data`, `crates/es/src/cmd/*.rs` — CLI `--frames` 플래그는 V3, 데이터셋 프레임은 V1이다.
- `es-env`의 `render` 피처를 기본값으로 만들거나, `es-runtime-embedded`나 `es-ros2`가 Vulkan에 닿게
  하는 것.
- `tests/golden/render/cornell_*` 편집; GPU 경로로 새 골든 생성.
