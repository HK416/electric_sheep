<!-- Korean translation of docs/api-notes/usd.md. The English file is the working copy; regenerate this when it changes. -->

# API note — `es-usd`가 읽는 OpenUSD `.usda` 텍스트 포맷과 UsdPhysics

`crates/es-usd`가 파싱하는 외부 표면을 고정한 다이제스트. 여기 있는 것은 설계 문서가
아니다(`docs/design/usd-reader.md` 참고); **포맷이 무엇인지**와 **우리가 그것을 얼마나
믿는지**를 기록한다.

모든 줄에는 태그가 붙어 있다:

- `verified` — M4 W3 패킷(2026-09) 동안 연결된 openusd.org 페이지에서 읽음.
- `unverified` — 포맷에 대한 사전 지식으로부터 작성되었으며 1차 출처에 대해 확인되지
  **않음**. 우리가 만들어낸 shape를 우리 파서가 받아들인다는 픽스처는 검증이 아니다.

USD는 잘라낼 수 있는 의존성이다(spec 1.9 항목 6): 지원되는 경로는 Python **USD Bake**
(spec 2.5)이며, 이 리더는 정직하지만 최소한인 네이티브 대안이다. 아래 부분집합 바깥의
무엇이든 prim 경로를 담은 하드 `UsdError::Unsupported`이며 — 조용한 추측은 결코 없다.

가져온 출처:

- <https://openusd.org/release/tut_helloworld.html>
- <https://openusd.org/release/glossary.html>
- <https://openusd.org/release/api/class_usd_geom_xformable.html>
- <https://openusd.org/release/api/class_gf_quatf.html>
- <https://openusd.org/release/api/usd_physics_page_front.html>

---

## 1. 파일 껍데기

| 항목 | 형태 | 상태 |
|---|---|---|
| Magic 헤더 | 첫 번째 비어 있지 않은 줄로서의 `#usda 1.0` | **verified** (hello-world 튜토리얼) |
| Layer 메타데이터 | 헤더 바로 뒤의 괄호로 묶인 블록 | `upAxis`, `metersPerUnit`, `defaultPrim`이 layer 메타데이터라는 것은 **verified**; 정확한 `(` ... `)` 배치는 **unverified** |
| `upAxis` | `token`, `"Y"` 또는 `"Z"`; 폴백 `"Y"` | 값 설정은 **verified**; `"Y"` 폴백은 **unverified** |
| `metersPerUnit` | `double`; 폴백 `0.01` (센티미터) | 폴백은 **unverified** |
| `kilogramsPerUnit` | 읽지 않음; 질량은 kg로 가정 | **unverified** |
| `defaultPrim` | 루트 prim을 이름 붙이는 `string`/`token` | **unverified** |
| `.usdc`, `.usdz` | 바이너리 crate / zip 컨테이너 | 존재는 **verified** (glossary); 우리는 이들을 절대 파싱하지 않는다 |

리더는 `upAxis`나 `metersPerUnit`이 없을 때 조용히 가정하는 대신 경고한다. 폴백이
틀리면 씬을 두 자릿수만큼 망가뜨릴 수 있는, 위 항목들 중 유일한 것이기 때문이다.

## 2. Prim

```usda
def Xform "hello"
{
    def Sphere "world"
    {
    }
}
```

**verified** (hello-world 튜토리얼): `def <TypeName> "<name>" { ... }`, 중괄호로 중첩됨.

| 항목 | 상태 |
|---|---|
| 지정자 `def`, `over`, `class` | **verified** (glossary: "가능한 지정자 세 가지") |
| 타입 이름을 생략할 수 있음 (`def "name"`) | **unverified** |
| Prim 경로는 `/` + `/`로 이어진 prim 이름들의 체인 | **unverified** |
| 이름과 `{` 사이의 prim별 메타데이터 블록 `( ... )` | **unverified** |

우리가 매핑하는 타입 이름: `Xform`, `Scope`, `Mesh`, `Cube`, `Sphere`, `Cylinder`,
`Capsule`, `PhysicsRevoluteJoint`, `PhysicsPrismaticJoint`, `PhysicsFixedJoint`. 그
외의 타입 이름은 파싱되어 stage에 보관되고 경고로 보고된다 — 오류가 아닌데, 알 수 없는
prim 타입은 보통 물리를 전혀 담지 않는 라이트나 머티리얼이기 때문이다.

## 3. Attribute와 relationship

선언은 `[uniform|custom]* <typeName> <name> = <value>`이며, relationship은
`rel <name> = </Prim/Path>`다. 문법으로서는 **unverified**; 아래 개별 철자는 나열된
스키마 페이지에서 왔으며 그곳에 표시되어 있다.

받아들이는 값 리터럴:

| 리터럴 | 예시 | 상태 |
|---|---|---|
| bool | `true` | **unverified** |
| int / float / double | `-3`, `1.5`, `1e-3` | **unverified** |
| token / string | `"X"` | **unverified** |
| tuple | `(0, 0, 1)`, `(1, 0, 0, 0)` | **unverified** |
| matrix4d | `( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )` | **unverified** |
| array | `[(0,0,0), (1,0,0)]`, `[0, 1, 2]` | **unverified** |
| asset | `@./mesh.usda@` | 참조 문법 `@file1.usd@`에서 **verified** (glossary) |
| rel target | `</World/link0>` | **unverified** |

배열 타입 이름은 `[]`를 붙여 쓴다: `float3[] points`, `int[] faceVertexIndices`,
`uniform token[] xformOpOrder`. **unverified**.

알 수 없는 attribute는 버려지지 않고 `Prim::attrs`에 그대로 보존되므로, 이후의 패킷이
파서를 건드리지 않고도 매핑을 넓힐 수 있다.

### 지오메트리 attribute

| Attribute | Prim | 의미 | 상태 |
|---|---|---|---|
| `points` | `Mesh` | `float3[]` 정점 위치 | **unverified** |
| `faceVertexIndices` | `Mesh` | `int[]` | **unverified** |
| `faceVertexCounts` | `Mesh` | `int[]`, 면당 정점 수 | **unverified** |
| `size` | `Cube` | 전체 변 길이, 기본값 `2.0` | **unverified** |
| `radius` | `Sphere`, `Cylinder`, `Capsule` | 기본값 `1.0` | **unverified** |
| `height` | `Cylinder`, `Capsule` | 기본값 `2.0`; `Capsule`에서는 이것이 캡을 제외한 *원통형* 구간이다 | **unverified** |
| `axis` | `Cylinder`, `Capsule` | token `"X"`/`"Y"`/`"Z"`, 기본값 `"Z"` | **unverified** |

### Transform op

**verified** (`UsdGeomXformable`):

- op 이름 `xformOp:translate`, `xformOp:scale`, `xformOp:orient`, `xformOp:transform`,
  성분별 `xformOp:translateX`/`scaleY`/..., `xformOp:rotateX/Y/Z`(도 단위)와 여섯 가지
  Euler variant; 커스텀 접미사는 `xformOp:<type>:<suffix>`다.
- `xformOpOrder`는 op들을 나열한다. 문서의 표현: 연속된 각 op는 앞의 것보다 "더
  로컬하게" 적용된다 — 즉 **첫 번째로 나열된 op가 가장 바깥쪽**이며, column-vector
  관례로는 `M = op[0] * op[1] * ... * op[n-1]`이다.
- `xformOpOrder` 안의 `"!resetXformStack!"`은 그 prim이 부모의 transform을 상속하지
  않는다는 뜻이다; 마지막 등장까지의 모든 것은 무시된다.

**unverified**: `xformOpOrder`에 이름이 없는 `xformOp:*` attribute는 아무 기여도 하지
않는다는 것. 우리는 이 해석을 따른다(`GetOrderedXformOps`와 일치한다) 그리고 prim이
순서에 없는 op를 authoring했을 때 경고한다.

우리는 `translate`, `orient`, `scale`, `transform`만 구현한다; 성분별 op와 Euler op는
`attrs`로 파싱되며 순서에 포함되어 있으면 미지원 op 경고로 보고된다.

`xformOp:orient`는 `quatf`/`quatd`다. `GfQuatf(float real, float i, float j, float
k)`가 실수부를 앞에 둔다는 것은 **verified**; 그래서 `.usda` 텍스트 직렬화가
`(w, x, y, z)`라는 것은 **unverified**다. 우리는 `(w, x, y, z)`로 읽어 spec 3.1의
xyzw로 저장한다.

## 4. UsdPhysics

이 절 전체는 <https://openusd.org/release/api/usd_physics_page_front.html>에 대해
**verified**다, 달리 표시된 경우 제외.

API 스키마는 prim 메타데이터 블록에 나타난다, 예:
`prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]` (목록 철자는
**unverified**).

| 스키마 | 우리가 읽는 attribute |
|---|---|
| `PhysicsRigidBodyAPI` | 마커일 뿐; `physics:velocity`, `physics:angularVelocity`(도/초), `physics:kinematicEnabled`가 존재하며 `attrs`에 보관된다 |
| `PhysicsCollisionAPI` | 마커일 뿐; `physics:collisionEnabled` (bool) |
| `PhysicsMassAPI` | `physics:mass` (float), `physics:centerOfMass` (point3f), `physics:diagonalInertia` (vector3f), `physics:principalAxes` (quatf), `physics:density` (double) |

Joint — `PhysicsRevoluteJoint`, `PhysicsPrismaticJoint`, `PhysicsFixedJoint`:

| Attribute | 타입 | 비고 |
|---|---|---|
| `physics:body0`, `physics:body1` | rel | body0을 부모로, body1을 자식으로 취급한다 (**unverified**) |
| `physics:localPos0`, `physics:localPos1` | point3f | 각 body의 프레임에서의 anchor |
| `physics:localRot0`, `physics:localRot1` | quatf | |
| `physics:axis` | token | `"X"`, `"Y"`, `"Z"` 중 하나 |
| `physics:lowerLimit`, `physics:upperLimit` | float | revolute는 **도(degree)**, prismatic은 거리 단위 |

`PhysicsDriveAPI`는 multi-apply이며, 자유도당 하나의 인스턴스를 갖는다:
`drive:<dof>:physics:stiffness`, `:damping`, `:targetPosition`, `:targetVelocity`. 우리는
revolute 조인트에 대해서는 `angular` 인스턴스를, prismatic에 대해서는 `linear`를
읽는다; 다른 dof 토큰에 대한 drive는 경고다.

도(degree) 기반 revolute 한계와 목표값은 이 노트에서 가장 값진 검증된 사실이다: 이를
라디안으로 읽으면 모든 조인트 범위가 조용히 57배 줄어든다.

## 5. 완전히 거부되는 것

아래 각각은 prim(또는 layer 메타데이터라면 `/`)을 지목하는
`UsdError::Unsupported { path, feature }`인데, 각각이 파일의 *의미*를 바꾸며 그렇지
않은 척하면 그럴듯하지만 틀린 씬을 만들어내기 때문이다:

| Feature | 거부되는 이유 |
|---|---|
| `references`, `prepend references` | composition; resolver와 layer stack이 필요 |
| `payload` | 마찬가지, 미룸 |
| `variantSet` / `variants` | variant에 따라 달라지는 씬 |
| `inherits`, `specializes` | composition |
| `.usdc`, `.usdz` 입력 | 바이너리 crate 포맷, 범위 밖 |
| `subLayers` | composition |
| time sample (`attr.timeSamples = { 0: ... }`) | 씬은 단일 시점이다 |

이들 중 무엇이든 가는 길은 USD Bake다(spec 2.5): Python에서 flatten하고 `.usda`로
다시 내보낸다.
