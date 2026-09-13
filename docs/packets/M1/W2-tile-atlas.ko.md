<!-- Korean translation of docs/packets/M1/W2-tile-atlas.md. The English file is the working copy; regenerate this when it changes. -->

# W2-tile-atlas — `es-render`: 타일 아틀라스, 채널 계약, 컴퓨트 래스터라이저

Spec: §15.1 (렌더 → 관측 경로, 채널 계약), §15.2 (대량 병렬 카메라를 위한 타일 아틀라스), §15.3 (렌더 경로; `RS`가 비전 학습 기본값), §15.4 (가속 구조 — *구현되지 않음*, 아래 참고), §3.1 (OpenCV 카메라 프레임, 좌상단 이미지 원점, sRGB 기본값), §3.3 (렌더링은 FP32, 카메라 상대), §3.4 (결정론적 실행), §7.2 (`ImageSpec`), §20.2 (이것이 거울상인 `render_tile_atlas` 예산 항목), §28.3 W2, §1.4 (골든 이미지가 오라클). 설계: `docs/design/renderer.md`.

`docs/packets/M4/W8-renderer.md`와 짝을 이루며, 같은 crate에서 패스 트레이서를 소유한다.

## context (범위)

```
crates/es-render/**                  (src, slang/raster.slang, slang/common.slang, slang/rng.slang, tests)
tests/golden/render/cornell_rs_*     (new goldens, generated once by the CPU reference)
docs/design/renderer.md
docs/packets/M1/W2-tile-atlas.md
```

## spec (사양)

- `TileAtlasCfg { tile_w, tile_h, tiles_per_row, n_tiles }` →
  `AtlasLayout { rows, width, height }`, 그리고
  `tile_origin(i) = ((i % tiles_per_row) * tile_w, (i / tiles_per_row) * tile_h)` — 닫힌
  형식이므로, 소비자는 산술만으로 환경별 텐서를 재구성하고 호스트 전송을 절대 쓰지 않는다
  (§15.2). `maxImageDimension2D`(16384)보다 넓거나 높은 아틀라스는 거부된다.
- `AtlasLayout::atlas_bytes(channel)`는 `crates/es-compile/src/budget.rs`의 `render_tile_atlas`
  항목을 **항목 단위로** 거울상으로 반영하며, 더블 버퍼 계수도 포함한다:
  `rows * tiles_per_row * tile_w * tile_h * components * dtype_bytes * 2`. `es-compile`은
  layer 7이고 layer 5에서 도달할 수 없으므로, 이 중복은 §15.2 참고 수치를 손으로 재계산하는
  테스트로 고정된다. `device_bytes()`는 렌더러가 실제로 할당하는 더 작은 별도의 수치다: 단일
  버퍼, 컴포넌트당 32비트 워드 하나, `Rgb8`은 `RGBA8` 워드 하나로 패킹된다.
- 채널 계약(§15.1): `Rs` 경로는 `Rgb8`, `Depth32 { unit_m: 1.0 }`, `Normal`(카메라 공간),
  `SegmentationId`(1부터 시작하는 geom id, `0` = 배경)를 쓴다. `Flow`와 `RgbF32Linear`는
  조용히 빈 버퍼가 아니라 `RenderError::UnsupportedChannel`이다.
- `Renderer::new(gpu, RenderConfig)`는 `Gpu::deterministic_execution_modes()`(§3.4 3단계)로
  `es_gpu::SlangCompiler`를 통해 `raster.slang`을 컴파일한다.
  `upload_scene(&SceneDesc)`는 box / plane / sphere / capsule / cylinder / ellipsoid를 고정된
  상수 세분화로 월드 공간 삼각형으로 테셀레이션한다; `Shape::Mesh`와 `Shape::HeightField`는
  `RenderError::UnsupportedShape`다.
  `render(&[CameraView]) -> Atlas`, `Atlas::read_tile(cam, channel) -> Tile`.
- `es_render::ImageSpec`은 §7.2의 layer 5 **부분집합**이다: width, height, 핀홀 intrinsics,
  near, far. `camera_model`은 `Pinhole`, `distortion`은 `None`, `shutter`는 `Global`, `Rgb8`의
  색공간은 sRGB로 고정되어 있다 — 필드가 아닌 이유는 §18.3 센서 리얼리즘이 아틀라스에 대한
  이후 패스이기 때문이다. 타일과 해상도가 다른 뷰는 리샘플되지 않고 거부된다(intrinsics를
  변환하지 않는 리샘플은 §7.2 `OBS-034` / `INV-14`).
- 아틀라스 픽셀당 스레드 하나. 래스터라이저는 오름차순 인덱스로 삼각형을 스캔하고 가장 가까운
  히트를 유지하며, 동률은 낮은 인덱스로 깨뜨린다; 커버리지와 깊이는 패스 트레이서가 쓰는 것과
  동일한 Möller–Trumbore 교차에서 나오며, 이것이 §15.3의 "깊이/세그/노멀이 경로 사이에서 비트
  동일함"이 구성상 성립하게 만드는 이유다. 비닝 없음, 가속 구조 없음: §15.4의 TLAS는 `es-gpu`가
  노출하지 않는 Vulkan 확장이 필요하다.
- 결정론: 원자적 연산 없음, 공유 메모리 없음, 서브그룹 연산 없음, 워크그룹 개수 의존 없음;
  모든 초월함수에 `es_math::approx` / `approx.slang`(§3.2 `DET-010`), sRGB 전달 함수의
  `c^(1/2.4)` 포함; `BTreeMap`만.

## oracle (오라클)

```
cargo fmt -p es-render --check
cargo clippy -p es-render --all-targets -- -D warnings
cargo test -p es-render -- --nocapture
cargo xtask layering && cargo xtask verify-goldens && cargo xtask context-budget && cargo xtask check-spec-refs
```

골든은 오직 `cargo test -p es-render -- --ignored generate_goldens`로만 재생성되며, 이는
**CPU** 레퍼런스를 실행한다. GPU로는 절대 하지 않는다: 한 드라이버에서 생성된 골든은 그
드라이버의 산술을 저장소에 구워 넣을 것이다. Vulkan 디바이스나 `slangc`가 없으면 모든 GPU
테스트가 `SKIP <test>: <reason>`을 출력하고 반환하므로, GPU 없는 CI 박스에서도 스위트가
통과하고 그 사실을 알린다.

## acceptance (수용 기준)

NVIDIA RTX 4060 Laptop GPU(드라이버 592.82, Slang 2026.8), 64×64 타일, 96삼각형 Cornell box에서
측정:

- `cpu_reference_reproduces_the_goldens_bit_for_bit` — 골든 4개 모두.
- `gpu_rasterizer_matches_the_cpu_goldens` — `Rgb8` 12288 바이트 중 0개 차이,
  `SegmentationId` 비트 동일, `Depth32` **최대 ULP 0**, `Normal` **최대 ULP 0**.
- `gpu_renders_are_bit_identical_across_runs` — 네 채널 모두, 두 번 실행.
- `gpu_atlas_packs_several_cameras` — 카메라 3대 + 패딩 타일 1개를 2×2 아틀라스에; 각 타일은
  그 카메라 하나만 렌더링한 것과 비트 동일; `read_tile(3, ..)`는 에러.
- 단위 테스트: 타일 원점과 패딩, 예산 공식 거울상, `maxImageDimension2D` 거부, 테셀레이션
  와인딩과 밀집된 1부터 시작하는 세그멘테이션 id, 중첩된 바디 포즈 합성, RNG 스트림 분리.

## forbidden (금지)

`context` 밖의 모든 파일 — 특히 `crates/es-usd`, `es-script`, `es-data`, `es-eval`, `es-py`,
`es-compile`, `crates/es`, 루트 `Cargo.toml`. 패스 트레이서, ReSTIR, SVGF(그건
`docs/packets/M4/W8-renderer.md`, 같은 crate, 별도 수용 기준). 테스트를 통과시키려고 골든을
편집하는 것(§1.4 — 골든은 CI 읽기 전용; 대신 ULP를 보고할 것). 새 확장 포인트 트레이트
(`INV-17`). `HashMap`/`HashSet`(§3.4). 그래픽스 파이프라인, 레이 트레이싱 확장, 가속
구조 — `es-gpu`는 아무것도 노출하지 않으며 그것을 추가하는 것은 이 패킷이 아니라 그쪽 패킷의
일이다. 스플랫 렌더링(§16.2; 훅은 `RenderPath` variant). 센서 리얼리즘(§18.3). 리사이즈/크롭을
렌더러로 밀어넣는 것(§7.2 — Observation IR이 전처리를 소유한다).
