<!-- Korean translation of docs/packets/M4/W3-usd-reader.md. The English file is the working copy; regenerate this when it changes. -->

# W3-usd-reader — 네이티브 `.usda` 부분집합 리더

Spec: 28.6 (M4 범위: "USD 네이티브"), 1.9 항목 6 (이는 *잘라낼 수 있으며*, Bake로
대체된다 — 그래서 최소한으로 유지한다), 2.5 (Python을 통한 USD Bake가 지원되는
대체 경로다), 3.1 (Z-up, 미터, 사원수 xyzw), 4.2 (계층화), 5.3 (`asset_hash` →
`scene_hash`), 17.2 (백엔드 의미 매핑), 1.2 (work packet), 1.4 (오라클 우선).

노트: `docs/api-notes/usd.md`(포맷 다이제스트, 모든 미검증 항목이 표시됨),
`docs/design/usd-reader.md`(코드 shape).

## context (범위)

```
crates/es-usd/**
crates/es-physics-core/src/usd.rs
crates/es-physics-core/src/lib.rs
crates/es-physics-core/Cargo.toml
crates/es/src/cmd/import.rs
crates/es/tests/cli.rs
tests/fixtures/usd/**
docs/api-notes/usd.md
docs/design/usd-reader.md
docs/packets/M4/W3-usd-reader.md
```

`crates/es-usd/**`는 `parse_usda`, `UsdStage`, `Value`, `UsdError`를 소유한다;
`crates/es-physics-core/src/usd.rs`는 `stage_to_scene`을 소유한다.
`es-physics-core`의 `lib.rs` 변경은 `pub mod usd;`와 문서 한 줄이며, `Cargo.toml`
변경은 mesh 콘텐츠 해시가 필요로 하는 `blake3` 의존성뿐이다 — 그 외에는 없다.
`crates/es/tests/cli.rs`는 **추가 전용**이며, 픽스처는 손으로 작성되고 각각
20 KB 이하다.

## spec (사양)

1. `es_usd::parse_usda(&str) -> Result<UsdStage, UsdError>` — 손으로 작성한
   토크나이저와 재귀 하강 prim 파서, 새 crate 의존성 없음. `UsdStage { meta,
   prims, warnings }`, `Prim { path, type_name, specifier, attrs:
   BTreeMap<String, Value>, rels, api_schemas, children }`.
2. `Value`는 *선언된* attribute 타입으로부터 타입이 매겨진다: `Bool`, `Int`,
   `Float`, `Double`, `Token`, `String`, `Asset`, `Float3`, `Quat`,
   `Matrix4d`, `Array`, `Rel`. `quatf`/`quatd` 리터럴은 파일 안에서
   `(w, x, y, z)`이며 spec 3.1의 `xyzw`로 저장된다.
3. 모든 오류는 소스 줄 번호를 싣는다. 알 수 없는 attribute는 `attrs`에
   보존된다; 알 수 없는 prim 타입은 경고와 함께 유지된다. `references` /
   `payload` / `variantSet` / `inherits` / `subLayers` / `.usdc`나 `.usdz`
   magic은 `UsdError::Unsupported { path, feature }`다.
4. `UsdStage::resolve_xform(path) -> Pose`는 조상들의 로컬 transform을
   합성한다; 로컬 transform은 `xformOpOrder`를 따른다(첫 번째로 나열된 것이
   가장 바깥쪽). Scale은 경고와 함께 버려진다.
5. `es_physics_core::usd::stage_to_scene(&UsdStage) -> Result<(SceneDesc,
   Vec<Warning>), UsdSceneError>` — rigid body, collision geom(콘텐츠 해시된
   `AssetRef` 뒤의 primitive와 `Shape::Mesh`), `PhysicsMassAPI` inertial,
   axis·limit·drive를 가진 revolute/prismatic/fixed joint. Y-up → Z-up과
   `metersPerUnit` 스케일링은 각각 한 번씩 적용된다; revolute limit과 angular
   drive 목표값은 도에서 라디안으로 변환된다. Id는 `scene_id`를 통해 prim
   경로로부터 만들어진다.
6. `es import usd <file.usda> --out scene.json`.
7. 영어만, `BTreeMap`만, 새로운 확장-지점 trait 없음(INV-17), 새 의존성 없음.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-usd -p es-physics-core -p es --all-targets -- -D warnings
cargo test -p es-usd -p es-physics-core -p es
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `tests/fixtures/usd/`의 픽스처:
  - `pendulum.usda` — rigid body 두 개, limit과 angular drive를 가진
    `PhysicsRevoluteJoint` 하나; `Body` 두 개와 drive가 `stiffness`/`damping`/`spring_ref`로,
    limit이 라디안으로 매핑된 `JointKind::Hinge` 하나로 매핑된다.
  - `mesh_cube.usda` — `PhysicsMassAPI`를 가진 `Mesh`; `AssetRef::hash`가 변환된
    point/index의 콘텐츠 해시인 `Shape::Mesh` 하나와 `BodyInertial`로 매핑된다.
  - `yup_cm.usda` — `upAxis = "Y"`, `metersPerUnit = 0.01`; USD 상의
    `(0, 300, 0)`에 있는 body는 spec 3.1의 `(0, 0, 3)` m에 도달하며, 수치로
    단언된다.
  - `referenced.usda` — `references`를 사용한다; `parse_usda`는 prim 경로를
    지목하는 `Unsupported`를 반환한다.
  - `malformed.usda` — 올바른 줄 번호와 함께 `Syntax` 오류를 반환하며,
    패닉하지 않는다.
- `parse_usda` 단위 테스트는 모든 `Value` 종류를 커버하며, `quatf (1, 0, 0,
  0)`이 xyzw identity로, `quatf (0, 1, 0, 0)`이 180도 x-회전으로 파싱됨을
  포함한다.
- `stage_to_scene`이 만들어낸 모든 `SceneDesc`는 `SceneDesc::validate()`를
  통과하며, 같은 텍스트를 두 번 파싱해도 픽스처의 `scene_hash`는 안정적이다.
- `es import usd tests/fixtures/usd/pendulum.usda --out <tmp>/scene.json`은
  0으로 종료하고 씬 해시를 출력하며, 쓰여진 JSON은 검증을 통과하는
  `SceneDesc`로 왕복한다.

## forbidden (금지)

- `es-usd`에 USD crate 의존성, 또는 어떤 의존성이든 추가하는 것(spec 2:
  순수 Rust 코어, C++ 없음; spec 1.9는 이 컴포넌트를 잘라낼 수 있게 만드므로
  의존성 그래프를 키워서는 안 된다).
- `crates/es-gpu`, `es-ir`, `es-env`, `es-script`, `es-data`, `es-eval`,
  `es-physics-backend`, `es-policy`, `es-assets`, `xtask`, `.github`를
  건드리는 것 — 다른 패킷들이 소유한다. 특히 `es-usd`는 `es-assets`에
  의존해서는 안 된다: 둘 다 layer 2이고 spec 4.2는 같은 layer 간 의존을
  금지한다.
- 새로운 확장-지점 trait(INV-17), 임포트 경로 어디에든 `HashMap`(spec 3.4),
  또는 composition arc를 조용히 무시하는 것.
- 어떤 golden 파일이든 편집하는 것, 또는 `stage_to_scene`이 파일이 말하지
  않는 물리를 지어내게 만드는 것(유도된 질량 없음, 지어낸 collision 의도
  없음).
