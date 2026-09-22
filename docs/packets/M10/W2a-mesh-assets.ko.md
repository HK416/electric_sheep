# M10 W2a — 메시 geom이 적재되고 콘텐츠로 해시되며 MuJoCo에서 시뮬레이션된다

스펙: §5.3(`asset_hash` → `scene_hash`: 체인이 덮는 것은 메시의 경로가 아니라 콘텐츠다), §3.4 / §3.2(해시
경로에 호스트 libm 없음; `str::parse::<f32>`와 `from_le_bytes`만), §4.3 / §17.2(백엔드 능력은 선언되고, 백엔드가
매핑 못 하는 장면은 무엇을 띄우기 전에 거부된다), §28.13 규칙 2(메시는 추가다: 프리미티브 전용 장면의
`scene_hash`는 움직이지 않는다; 디코딩된 메시는 `SceneDesc`에 실린다; 리졸버는 함수다)와 파동 1, INV-17.
리뷰: `docs/reviews/M7.md` "R5/R6(메시 geom, MJCF 텍스처) … 보류"; §28.10 "사다리에 없는 것". 설계 노트:
`usd-reader.md`(이 패킷이 모든 임포터에 채택하는 콘텐츠 해시 규칙), `renderer.md` 2.1(`Mesh` 행: "에셋 리졸버가
필요"). **W0b**에 의존(같은 크레이트; `scene_hash` 플랫폼 안정이 먼저). 유형 B.

## 질문

`Shape::Mesh { asset }`은 이미 MJCF·URDF·glTF·USD에서 파싱되고 해시되지만, STL이나 OBJ를 정점으로 바꾸는 리더가
없고 `AssetRef::hash`는 *경로 문자열*의 다이제스트이며 MuJoCo 이미터와 렌더러가 그 geom을 거부한다 — 그래서
Menagerie의 모든 로봇이 손 밖에 있고 두 캠페인의 원본 정책은 우리가 손으로 파생한 장면에서 학습됐다. **메시
geom을 가진 MJCF 장면이 적재(STL / OBJ)되고, 메시를 `SceneDesc`에 싣고, Windows와 Linux에서 같은 콘텐츠 해시를
갖고, MuJoCo 백엔드에서 도는가 — 커밋된 프리미티브 전용 장면의 `scene_hash`는 움직이지 않은 채?**

쓰기 전에 확인된 것(에이전트가 해당 줄을 다시 확인한다): 커밋된 두 장면은 `assets: []`
(`crates/es-assets/tests/so101_provenance.rs:374-378`); `AssetRef.path`는 해시 입력(`scene.rs:626-632`)이라
절대 다시 쓰면 안 된다; MJCF 임포터는 geom 질량을 계산하지 않는다(`scene.rs:80-82`) — 이미터가 `mass=` /
`density=`를 쓰고 MuJoCo가 유도하므로 메시 geom에 **Rust 관성 코드가 필요 없다**; `PhysicsBackend::load(&SceneDesc)`와
`EnvRenderer::new(&SceneDesc)`는 사이드 스토어가 지나갈 수 없는 INV-17 trait이다; `mujoco_ref.py`는 `from_xml_string`으로
적재하므로 인라인 `<mesh vertex= face=>`는 와이어에 파일 경로가 필요 없다; MJWarp는 MuJoCo의 능력을 물려받는다
(`mjwarp.rs:52-58`).

## 명세

* `SceneDesc.meshes: BTreeMap<StableId, MeshData>` — `#[serde(default, skip_serializing_if =
  "BTreeMap::is_empty")]`, 해시되지 **않음**(`scene_hash`에 들어가는 것은 `AssetRef.hash` 재작성이다);
  `..Default::default()`가 없는 모든 `SceneDesc { … }` 리터럴에 `meshes: BTreeMap::new()`. `MeshData`는 기존
  `es_assets::gltf::MeshData`(`gltf.rs:97-105`)에 `Serialize, Deserialize`를 더한 것; STL / OBJ는 `positions`와
  `indices`를 채우고 `normals` / `uvs` / `material`은 `None`. `gltf::mesh_content_hash`(`gltf.rs:442`)가 크레이트의
  유일한 콘텐츠 해시 함수가 된다(`pub(crate)`): tag ‖ f32 LE positions ‖ u32 LE indices — 구성상 플랫폼 독립.
* `es_assets::mesh::load(scene: &mut SceneDesc, base_dir: &Path) -> Result<(), MeshError>`: 아직 `scene.meshes`에
  없는 모든 `AssetKind::Mesh` 에셋을 `base_dir.join(asset.path)`(`path`는 이미 `join(meshdir, file)`, `elements.rs:53,397`;
  절대 경로는 그대로)에서 읽어 확장자로 분기(`.stl` / `.obj`; 아니면 `MeshError::Unsupported { path }`), 넣고
  `asset.hash = mesh_content_hash(…)`. `lib.rs:7`("디스크에서 파일을 읽지 않는다")에 예외 하나. `MeshError`는
  경로와 이유(파일 없음, 잘못된 인덱스, 짧은 버퍼)를 이름 붙인다.
* 리더: `es_assets::stl::parse(&[u8])` — `len == 84 + 50·n`이면 바이너리(헤더가 `solid`로 시작할 수 있음), 아니면
  ASCII; 정점은 정확한 f32 비트로 중복 제거(`BTreeMap<[u32; 3], u32>`, 최초 등장 순, 결정적); 저장된 면 법선이
  0이 아니고 `dot(normal, (b−a)×(c−a)) < 0`이면 `b`/`c` 스왑(렌더러의 법선은 감김에서 나온다).
  `es_assets::obj::parse(&str)` — `v`(앞 숫자 셋), `f`는 `a`, `a/b`, `a//c`, `a/b/c` 형태; 1-based, 음수 = 현재 `v`
  수 기준 상대; 팬 삼각분할; 나머지는 무시. 둘 다 `index < positions.len()` 검사.
* MJCF: `elements.rs parse_asset`이 `scale`, `refpos`, `refquat`을 읽어 기본값이 아니면 `MjcfError::Unsupported`
  (`mod.rs:379-385`의 `<compiler coordinate>` 패턴) — 오늘 `report_unknown`은 이를 경고로 낮추는데, 메시가
  렌더되는 순간 조용한 지오메트리 오류가 된다. `scale` 베이킹은 이름 붙인 후속.
* 이미터(`mjcf_out.rs`): `scene.meshes`에서 `<asset><mesh name="…" vertex="x y z …" face="i j k …"/></asset>`
  (f32는 `{:?}`, 최단 왕복 표현), `size` 없는 `<geom type="mesh" mesh="…">`. 에셋이 `scene.meshes`에 없는 메시
  geom → `PhysicsError::Unsupported("geom `g`: mesh `m` is not loaded (es_assets::mesh::load)")`, 여전히
  `Process::spawn` 전(`mujoco.rs:226` 대 `:232`). `file=` 경로 없음(trait을 지나는 경로 채널이나 해시되는
  구조체 안의 머신 경로가 필요하다 — 인라인 텍스트가 너무 느려지면 후속).
* 능력: `mujoco.rs:67-73`에 `Feature::ContactMesh`와 quirk "충돌은 메시의 볼록껍질을 쓴다; 렌더러는 표면을
  그린다"; `mjwarp.rs:58`에서 `ContactMesh` 제거(거기서는 미검증); Newton(`mapping.rs:392`)과 픽스처 백엔드는
  불변 — 그들 위의 메시 장면은 `Requirements::from_scene`을 통해 이름으로 거부된다.
* 픽스처: `tests/fixtures/mjcf/mesh_box.xml` — `<compiler meshdir="meshes"/>`, `<asset><mesh name="box"
  file="box.stl"/>`, 평면, `mesh_box`(freejoint, `type="mesh"`, density 1000, z = 0.3), `prim_box`(freejoint,
  `size="0.05 0.05 0.05"`, x = 0.5), W2b용 `<camera name="cam">` 하나; `tests/fixtures/mjcf/meshes/box.stl` =
  바이너리 684 B(80바이트 헤더 + ±0.05 상자의 면 12개), `ES_GENERATE_GOLDENS=1` 아래 `#[ignore]
  generate_mesh_box_stl`이 쓴다. 출처: `so101_pick_place.PROVENANCE.json`에 상위 STL 19개의 `"mesh_blake3":
  { "<name>": "<hex>" }`(벤더 안 함; 기존 `so101_provenance.rs:109-147` 패턴을 파일 목록으로 확장해
  `target/menagerie/<commit>/`로 fetch); `tests/fixtures/mjcf/panda.PROVENANCE.json` — 핀 커밋의 Menagerie
  `franka_emika_panda`, `panda.xml`과 각 `.obj` / `.stl`의 blake3, 벤더 없음. Panda는 **리더 + 해시만**(`panda.xml`이
  `scene_to_mjcf`가 거부하는 손가락 `<tendon>` / `<equality>`를 가진 것으로 보임 — 미검증; MuJoCo OBJ 적재가
  필요하면 `universal_robots_ur5e`가 후보). `docs/api-notes/mujoco.md`(+ `.ko.md`)에 메시 절: 바이너리 STL
  레이아웃, OBJ 인덱스 규칙, `<mesh>` 속성, MuJoCo 3.13 문서에서 읽은 `inertia="legacy"` 대 `"exact"`,
  Menagerie 커밋.

## context

```
crates/es-assets/src/lib.rs
crates/es-assets/src/scene.rs
crates/es-assets/src/mesh.rs
crates/es-assets/src/stl.rs
crates/es-assets/src/obj.rs
crates/es-assets/src/gltf.rs
crates/es-assets/src/mjcf/elements.rs
crates/es-assets/tests/**
crates/es-physics-backend/src/mjcf_out.rs
crates/es-physics-backend/src/mujoco.rs
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/tests/**
crates/es-physics-core/src/usd.rs
crates/es-render/src/cornell.rs
crates/es-splat/src/**
crates/es-usd/src/**
tests/fixtures/mjcf/mesh_box.xml
tests/fixtures/mjcf/meshes/box.stl
tests/fixtures/mjcf/so101_pick_place.PROVENANCE.json
tests/fixtures/mjcf/panda.PROVENANCE.json
docs/api-notes/mujoco.md
docs/api-notes/mujoco.ko.md
docs/packets/M10/W2a-mesh-assets.md
docs/packets/M10/W2a-mesh-assets.ko.md
```

(`es-physics-core`, `es-render`, `es-splat`, `es-usd` 항목은 `SceneDesc` 리터럴에 `meshes: BTreeMap::new()`를
더하는 것뿐이다.)

## 오라클

1. `cargo test -p es-assets --test mesh_load` — `binary_stl_dedups_shared_vertices`(변을 공유하는 손으로 쓴
   면 둘 → 위치 4, 인덱스 6), `binary_stl_flips_a_facet_against_its_normal`, `ascii_stl_decodes`,
   `obj_negative_indices_and_quad_fans`, `obj_face_forms_slash_and_double_slash`,
   `box_stl_content_hash_is_pinned`(hex 상수), `mesh_box_load_rewrites_asset_hash_and_scene_hash`(전 ≠ 후;
   후 == 고정값), `committed_scenes_are_unmoved_by_load`(so101 / go1: 전 == 후; so101 hex도 고정 — W0b가 플랫폼
   안정으로 만들었다), `scale_refpos_refquat_are_refused_by_name`, `missing_file_names_the_path`,
   `bad_index_is_an_error`.
2. `cargo test -p es-assets --test mjcf_parse assets_are_references_not_files` 불변.
3. `cargo test -p es-physics-backend` — `mjcf_out::tests::mesh_geom_emits_inline_vertex_and_face`,
   `mjcf_out::tests::unloaded_mesh_is_refused_by_name`, `mujoco::tests::capabilities_are_declared_honestly`
   (`:429` 반전), `a_scene_the_backend_cannot_map_is_refused_before_anything_is_spawned`(메시 경우도 여전히
   spawn 전), mjwarp 능력/매핑 일치 테스트.
4. `ES_PYTHON=… cargo test -p es-physics-backend --test mesh_box` — `mesh_box_loads_in_mujoco`(nbody 3, nmesh 1,
   `mesh_vertnum[0] == 8`, `mesh_facenum[0] == 12`, 컴파일 경고 0), `mesh_box_rests_like_the_primitive_box`
   (1 kHz 2,000 스텝, `|z_mesh − z_prim| < 1e-3 m`, 둘 다 `0.05 ± 2e-3`, NaN 없음),
   `mesh_box_mass_and_inertia_match_the_primitive`(`body_mass` 상대 < 1e-9, `body_inertia` 상대 < 1e-6; MuJoCo
   3.13 기본 `<mesh inertia="legacy">`는 볼록 닫힌 상자에서 `exact`와 같아야 함 — **미검증**: 다르면 측정된 차이와
   성립하는 허용오차를 기록). 로컬(`ES_PYTHON` venv) **그리고** 서버(CPU 큐, 수 분)에서 — 같은 바이트.
5. `--ignored`, 서버에서: `so101_provenance::upstream_meshes_load_and_hash`(해시 19개 == 매니페스트;
   `derivative_has_the_upstream_kinematics`가 이미 쌍둥이를 증명); `menagerie_meshes::so101_upstream_loads_in_mujoco_with_meshes`
   (`ES_PYTHON`: 적재; 바디별 `body_mass`가 상위의 직접 `from_xml_path` 적재와 1e-9 상대 이내; `camera_mount`
   0.012 kg); `panda_provenance::panda_meshes_load_and_hash`(실제 파일의 OBJ + STL; 해시와 삼각형 수를 api-note에 고정).
6. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W2a-mesh-assets.md`; `cargo xtask context-budget`
   (`es-assets` 기대 ≈ +290 → ~3,890).

## 수용

오라클 1~4와 6이 CI에서 녹색; 5는 env 설정 시 녹색; 커밋된 두 장면의 `scene_hash` 불변; `tests/golden/**` 불변;
api-note 절과 한국어 짝; 노트에 이름 붙인 후속: `<mesh scale>` 베이킹, 경로 채널을 갖춘 `file=`, glTF `MeshData` →
`scene.meshes`, MJWarp `ContactMesh`, 텍스처 / 머티리얼(M7 R6).

## 금지

`es-render`, `es-env`, `es-editor`, `es` 변경(W2b); `AssetRef.path` 재작성; `scene_hash` 인코딩에 필드 추가;
`PhysicsBackend`나 어떤 trait도 건드리기(INV-17); Menagerie 메시 벤더링; Rust 메시 질량 / 관성 코드; 적재 경로
어디에도 호스트 libm; `scale` 베이킹; 텍스처 / 머티리얼; `docs/ARCHITECTURE*.md`; `tests/golden/**`.
