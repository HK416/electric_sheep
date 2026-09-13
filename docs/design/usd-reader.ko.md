<!-- Korean translation of docs/design/usd-reader.md. The English file is the working copy; regenerate this when it changes. -->

# 설계 — 네이티브 `.usda` 리더 (M4, spec 28.6)

포맷에 관한 사실과 그 출처는 `docs/api-notes/usd.md`에 있다. 이 노트는 오직 코드의
shape에 관한 것이다.

## 1. 이것이 의도적으로 작은 이유

Spec 1.9는 USD 네이티브 리더를 컷 목록에서 여섯 번째로 매긴다: **"USD 네이티브 리더 —
Bake로 대체"**. 지원되는 임포트 경로는 spec 2.5의 Python USD Bake다. 그래서 여기서의
목표는 OpenUSD 구현이 아니다; 목표는: *평탄화된 `.usda`에서 rigid-body 씬을 뽑아낼
만큼 충분히 좋은, 손으로 작성한 부분집합 리더이되, 파일이 그 부분집합을 벗어나는
순간 큰 소리로 거부하는 것*이다.

아래의 모든 결정을 형성하는 두 가지 귀결:

- **Composition 없음.** `references`, `payload`, `variantSet`, `inherits`,
  `subLayers`는 prim 경로를 담은 `UsdError::Unsupported`다. reference를 조용히
  무시하는 리더는 절반의 body가 빠진 씬을 만들어내면서도 오류는 내지 않을 것이다
  — importer로서는 최악의 실패다.
- **새 의존성 없음.** 손으로 작성한 토크나이저는 ~300줄이다; OpenUSD 바인딩은
  C++ 툴체인이다(spec 2, 순수 Rust 코어). 부분집합이 작으므로 게으른 선택이 곧
  올바른 선택이기도 하다.

## 2. 두 crate, 하나의 이음매

| Crate | Layer | 소유하는 것 |
|---|---|---|
| `es-usd` | 2 | `.usda` 텍스트 → `UsdStage`. 물리, body, joint에 대해서는 아무것도 모른다. |
| `es-physics-core::usd` | 3 | `UsdStage` → `SceneDesc`. 텍스트에 대해서는 아무것도 모른다. |

이 분리 자체가 이음매의 핵심이다. Layer 2는 `es-usd`가 `es-assets`에 의존하는 것을
금지하며(같은 layer, spec 4.2 "같은 layer 간 의존 금지"), 이것이 정확히 `UsdStage`가
`SceneDesc`를 언급할 수 없고 매핑이 한 layer 위, 이미 둘 다에 의존하는
`es-physics-core`에 사는 이유다.

이는 또한 파서를 정직하게 유지한다: `UsdStage`는 파일을 충실하게, 그리고
손실이 없다 할 만큼 충분히 미러링하며(알 수 없는 attribute는 보존되고, 알 수 없는
prim 타입은 경고와 함께 유지된다), 어떤 해석도 담지 않는다. 무엇이 body로 치는지,
`Capsule`의 axis가 어떻게 pose가 되는지, 도(degree)가 무엇인지 같은 모든 판단은
mapper 안에 있으며, 한 자리에서 읽을 수 있다.

## 3. `es-usd`

```text
parse_usda(&str) -> Result<UsdStage, UsdError>
UsdStage { meta: StageMeta, prims: Vec<Prim>, warnings: Vec<String> }
Prim { path, type_name, specifier, attrs: BTreeMap<String, Value>, rels, api_schemas, children }
```

- 토크나이저: 한 번의 패스로, 모든 토큰의 줄 번호를 추적하므로 모든
  `UsdError::Syntax`가 줄 번호를 싣는다. 줄 끝까지의 `#`는 주석이다(`#usda` magic은
  예외).
- 파서: `def`/`over`/`class` 블록에 대한 재귀 하강(recursive descent). 깊이는
  상한이 있다(10⁶개의 `{`를 먹이는 퍼저는 스택 오버플로가 아니라 오류를 받아야
  한다).
- `Value`는 리터럴로부터 추측되는 것이 아니라 *선언된* 타입 이름으로 타입이
  매겨진다: `float 1`과 `int 1`은 서로 구분된 채로 남으며, `quatf (w,x,y,z)`는
  파싱 시점에 spec 3.1 xyzw로 변환되어 어떤 소비자도 순서를 틀릴 수 없다.
- `resolve_xform(path) -> Pose`는 조상들의 로컬 transform을 합성한다. 로컬
  transform은 `xformOpOrder`를 정확히 따르며, column-vector로
  `M = op[0] * ... * op[n-1]`이다(api-note 3). Scale은 rigid `Pose`로 표현할 수
  없다: identity가 아닌 scale은 경고이며 버려지는데, 이는 glTF importer가 이미
  균일하지 않은(non-uniform) 노드 scale에 대해 내리는 것과 같은 선택이다.
- `BTreeMap`만 사용한다(spec 3.4는 `HashMap` 순회 의존성을 금지한다). 이 crate 안
  어디에서도 디스크에서 파일을 읽지 않는다; `es`가 `&str`를 건네준다.

## 4. `es-physics-core::usd`

```text
stage_to_scene(&UsdStage) -> Result<(SceneDesc, Vec<Warning>), UsdSceneError>
```

매핑 규칙, 모두 기계적이다:

| USD | `SceneDesc` |
|---|---|
| `PhysicsRigidBodyAPI`를 가진 prim | `Body`; 아래 부모 규칙 참고 |
| `PhysicsCollisionAPI`를 가진 prim | 가장 가까운 rigid-body 조상 위의 `Geom` |
| `Cube size` | `Shape::Box { half_extents: size/2 }` |
| `Sphere radius` | `Shape::Sphere` |
| `Cylinder`/`Capsule`의 `radius`,`height`,`axis` | `Shape::Cylinder`/`Capsule { half_length: height/2 }`, `axis`는 geom pose에 접혀 들어감 |
| `Mesh`의 `points`/`faceVertexIndices`/`faceVertexCounts` | `Shape::Mesh` + 콘텐츠 해시된 `AssetRef` |
| `PhysicsMassAPI` | `BodyInertial { mass, com, inertia, frame }` |
| `PhysicsRevoluteJoint` | `JointKind::Hinge`, 한계값은 도 → 라디안 |
| `PhysicsPrismaticJoint` | `JointKind::Slide`, 한계값은 `metersPerUnit`으로 스케일됨 |
| `PhysicsFixedJoint` | `JointKind::Fixed` |
| `drive:{angular,linear}:physics:*` | `Joint::stiffness`, `damping`, `spring_ref` |

### Body 트리

USD body의 부모 prim은 보통 그것의 부모 *body*가 아니다: `Scope`와 `Xform` prim이
그 사이에 끼어 있으며, 로봇은 흔히 평평한 링크 목록에 조인트가 더해진 형태이고,
articulation은 전적으로 `physics:body0`/`body1`이 실어 나른다. 그래서 body의
부모는

1. 있다면 가장 가까운 rigid-body **조상 prim**이고;
2. 그렇지 않으면 그것을 `physics:body1`로 이름 붙이는 joint의 `physics:body0`이고;
3. 그렇지 않으면 아무것도 아니다 — 그것은 루트다.

그러면 `body.pose`는 `parent_world⁻¹ ∘ body_world`이며, world pose들로부터
계산된다. 모순되는 파일(서로를 상대의 부모로 만드는 두 joint)은 여기의 어떤
규칙이 아니라 `SceneDesc::validate`의 사이클 검사가 잡아낸다.

### 관례 (spec 3.1)

두 가지 변환이, 각각 한 번씩, 모든 prim의 **로컬** transform에 적용되며, 그 뒤
계층을 따라 합성된다 — 이는 정확히 `es-assets`의 glTF importer가 하는 것과
같으며, 같은 이유로 옳다: 켤레(conjugation)는 준동형(homomorphism)이고 회전은
`Pose::compose`에 대해 분배되므로, prim마다 변환하는 것과 마지막에 한 번
변환하는 것은 일치한다.

1. **단위.** 모든 길이는 `metersPerUnit`을 곱한다; 모든 inertia는
   `metersPerUnit²`을 곱한다. Revolute 한계값과 angular drive 목표값은 스키마
   안에서는 도(degree)이며 여기서 라디안이 된다.
2. **축.** USD의 `upAxis = "Y"`는 오른손 Y-up이다 — glTF가 쓰는 것과 같은
   관례 — 그래서 그 수정은 glTF importer가 적용하는 것과 같은 고정된 사원수다:
   `+Y → +Z`, `-Z → +X`, `-X → +Y`. 위치는 이로 회전되고 방향은 이로 켤레화된다.
   Body-local 양들도 같은 회전을 받는다 — 질량 중심, joint anchor와 axis,
   `Cylinder`/`Capsule`의 `axis` 토큰, mesh point — 그래서 결과 안의 모든
   frame이 world만이 아니라 전부 Z-up이 된다. `upAxis = "Z"`는 이 수정을
   항등(identity)으로 만든다.

### Id (spec 5.3)

앞의 `/`가 제거된 prim 경로에 대한 `scene_id(kind, path)`:
`/World/arm/link1` → `body/World/arm/link1`. Prim 경로는 USD에서 구성상 유일하므로
id는 카운터 없이도 유일하며, `.usda`로부터 만들어진 `SceneDesc`는 MJCF나 glTF로부터
만들어진 것과 같은 id 체계를 갖는다.

Mesh `AssetRef::hash`는 변환된 `f32` point와 `u32` index에 대한 blake3다 — prim
경로가 아니라 내용이다 — glTF importer의 규칙과 일치하므로, 같은 mesh가 두 번
authoring되어도 한 번만 해시된다.

## 5. 의도적으로 하지 않는 것

- Material, light, camera, shading, primvar, subdivision: `attrs`로 파싱되어
  무시된다.
- `physics:velocity` / `angularVelocity` / articulation root / collision group /
  filtered pair: `SceneDesc`에는 초기 상태를 위한 자리가 없으므로 이들은
  경고다.
- Time sample: `SceneDesc`는 한 순간이다.
- `.usda` 쓰기. Export는 M4 W3의 범위 밖이다.
- 디코딩된 mesh payload를 돌려주는 것. `stage_to_scene`은 삼각형의 콘텐츠
  해시를 가진 `AssetRef`를 반환한다 — `scene_hash`(spec 5.3)를 위해서나
  백엔드가 캐시 키로 쓰기에는 충분하지만 — `es-assets`의 glTF importer가
  `MeshData`를 반환하는 것처럼 정점 배열 자체를 반환하지는 않는다. 삼각형이
  필요한 소비자는 `UsdStage`에서 `points` / `faceVertexIndices`를 다시 읽는다;
  반환 타입을 넓히는 것은 별도의 패킷이다.

이들 전부는 `Warning`이거나 `Unsupported`이며, 절대 조용히 버려지지 않는다.
