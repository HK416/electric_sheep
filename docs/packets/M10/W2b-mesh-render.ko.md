# M10 W2b — 래스터라이저·패스 트레이서·CPU 레퍼런스가 메시 geom을 그린다

스펙: §15(테셀레이터 하나가 두 경로와 CPU 레퍼런스를 먹인다), §3.4(GPU가 받는 지오메트리는 레퍼런스가 순회하는
지오메트리와 비트 동일), §1.4(골든은 CI 읽기 전용; 추가는 허용), §28.13 규칙 2와 파동 2, INV-17. 설계 노트:
`renderer.md` 2.1(형상 표의 `Mesh` 행)과 테셀레이션 단락(`scene.rs` 상수는 손잡이가 아니다). **W2a**에 의존
(`SceneDesc.meshes`, `es_assets::mesh::load`, `mesh_box.xml`). 유형 B.

## 질문

`TriScene::from_scene`은 "`SceneDesc`가 정점이 아니라 `AssetRef`를 갖고 있다"며 `Shape::Mesh`를 거부한다. W2a가
정점을 `SceneDesc`에 실었다. **테셀레이터에 팔 하나를 더하면 래스터라이저·패스 트레이서·CPU 레퍼런스가 같은
삼각형에서 메시 geom을 그리고, 기존 모든 골든이 바이트 동일하며, 모든 CLI 경로가 공유하는 장면 로더 하나가
`es-env` 변경 없이 메시를 실어 나르는가?**

## 명세

* `crates/es-render/src/scene.rs`: `tessellate(geom)` → `tessellate(geom, scene)`; `Shape::Mesh { asset }` 팔은
  `scene.meshes.get(&asset)`을 읽어 `indices`를 셋씩 `[Vec3; 3]`로 낸다(`f64::from(f32)`, 정확한 확장; geom 포즈는
  여전히 `push_geom`이 프레임마다 `f64`로 적용한 뒤 `to_f32`); 에셋이 없으면 `UnsupportedShape { shape: "Mesh (asset
  not loaded: es_assets::mesh::load)" }`. `HeightField`는 거부 유지. `SceneCache.local`의 키는 `(Shape, [u8; 32])`가
  되어 `Mesh`는 에셋의 `hash`, 나머지는 0 — 같은 geom id에 다른 메시 콘텐츠를 가진 두 장면이 앨리어싱되지
  않는다. `from_scene_with_poses`, `tri_scene`, `upload_scene` 시그니처 불변; CPU 레퍼런스(`cpu.rs`)와 BVH가 같은
  `TriScene`을 소비하므로 CPU/GPU 패리티는 자동. `renderer.rs:302`의 "프리미티브만" 주석과 `renderer.md` 2.1의 행을
  갱신.
* 배관: `es::cmd::backend::load_scene`(`crates/es-tools/src/backend.rs:125` — `es loop collect`, `es eval run`,
  `es video showcase`, `es backend`가 공유하는 유일한 헬퍼)과 `replay_view::load_scene`
  (`crates/es-editor/src/model/replay_view.rs:514-532`)이 MJCF / URDF 파싱 뒤 `es_assets::mesh::load(&mut scene,
  path.parent())`를 부른다. `EnvRenderer`는 장면을 메시째 복제하므로 **`es-env` 소스 변경 없음**; 편집기 replay는 세 줄.
* 골든: CPU 레퍼런스의 64×64 `mesh_box_rs_rgb8`, `mesh_box_rs_depth`, `mesh_box_rs_seg`, `mesh_box_pt1spp`,
  `parse_mjcf` + `mesh::load`로 적재한 `mesh_box.xml`과 고정 `CameraView`(`cornell_camera` 패턴), 기존
  `generate_goldens`(`crates/es-render/tests/render.rs:281`)를 `MESH_GOLDENS` 표로 확장해 `ES_GENERATE_GOLDENS=1`
  아래 생성; 사이드카는 생성자로 `es_render::cpu`를 적는다. RTX 3060 상자에서 생성, RTX 4090에서 재현(CPU 큐가
  빈 뒤 두 줄짜리 서버 실행).

## context

```
crates/es-render/src/scene.rs
crates/es-render/src/renderer.rs
crates/es-render/src/error.rs
crates/es-render/tests/render.rs
crates/es-env/tests/render_loop.rs
crates/es-tools/src/backend.rs
crates/es/tests/cli.rs
crates/es-editor/src/model/replay_view.rs
tests/golden/render/mesh_box_*
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M10/W2b-mesh-render.md
docs/packets/M10/W2b-mesh-render.ko.md
```

수정: CLI 로더는 W3b에서 `crates/es/src/cmd/backend.rs`에서 `crates/es-tools/src/backend.rs`로 옮겨졌다; 그 오라클은
`crates/es/tests/cli.rs`의 `load_scene_resolves_meshes_relative_to_the_scene_file`이다.

## 오라클

1. `cargo test -p es-render` — `scene::tests::mesh_tessellates_from_the_scene_store`(`scene.meshes`의 손으로 만든
   사면체 → 삼각형 4, 조밀한 분할 id, 감김에서 나온 법선), `mesh_without_store_is_refused_by_name`
   (`mesh_and_height_field_are_refused_by_name`의 개명), 메시 geom을 더한
   `curved_shapes_upload_the_vertices_the_cpu_reference_traverses`, `scene_cache_keys_on_mesh_content`(같은 id,
   다른 해시 → 다른 삼각형).
2. `cargo test -p es-render --test render` — `cpu_reference_reproduces_the_mesh_goldens_bit_for_bit`; GPU가 있으면
   (없으면 SKIP): `gpu_rasterizer_matches_the_mesh_goldens`(rgb8·seg 비트 동일, depth는 기존 골든이 쓰는 측정 ULP),
   `gpu_path_tracer_matches_the_cpu_on_the_mesh_box`(1 spp 비트 동일), `mesh_box`에서 돌린
   `cached_tessellation_is_bit_identical`.
3. `cargo test -p es-env --features render --test render_loop mesh_box_renders_through_env_renderer`(GPU): `cam`으로
   `EnvRenderer::new`; 포즈된 프레임이 `from_scene_with_poses`의 프레임과 비트 동일.
4. `cargo test -p es --test cli`와 `cargo test -p es-editor` 불변(프리미티브; `so101_frame0` 골든 바이트 동일);
   `cargo build -p es-editor`.
5. `cargo xtask verify-goldens`(추가만); 골든 넷을 서버에서 재현; `cargo xtask ci`; `cargo xtask check-scope
   docs/packets/M10/W2b-mesh-render.md`.

## 수용

오라클 1~5; 기존 모든 골든 바이트 동일; `renderer.md` 2.1과 한국어 짝; 이름 붙인 후속: 상위 SO-101(~350k 삼각형)의
9-metric ms/frame, 없는 메시가 첫 프레임이 아니라 생성 시점에 실패해야 한다면 `EnvRenderer::new`의 `tri_scene` 드라이런.

## 금지

`es-assets`, `es-physics-backend` 변경(W2a); 테셀레이션 상수(`scene.rs:22-26`); 기존 골든 편집;
`from_scene_with_poses` / `tri_scene` / `upload_scene` 시그니처 변경; `HeightField`; 메시 전용 셰이딩 경로;
`HashMap` 일체; `docs/ARCHITECTURE*.md`.
