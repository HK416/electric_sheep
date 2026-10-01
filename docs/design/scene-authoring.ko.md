<!-- Korean translation of docs/design/scene-authoring.md. The English file is the working copy; regenerate this when it changes. -->
# 장면 작성 — 에디터에서 과제를 처음부터 만든다 (S4)

스펙: §5.3(해시 체계), §6(Task IR, `SceneRef`), §14(작성 프런트엔드, 저장 형식), §17.2(의미
매핑 리포트), §23(에디터), §4.2(계층). 이 노트가 기대는 설계 노트: `editor-redesign.md`(§2
결정, §3 ① 화면, §5 S4 방향), `renderer.md` §15–16(텍스처, PBR), `usd-reader.md`,
`python-builder.md`, `node-sdk.md`.

## 1. 왜, 그리고 소유자의 결정

소유자, 2026-10-01, Shadow Hand를 손으로 이식해야 했던 뒤(플랜 H, H1): "장면에서 구성
요소를 추가 삭제가 가능한가? es-editor의 지향점은 게임엔진 처럼 쉬운 로봇 학습 사용인데 장면을
따로 준비해야 한다면 별로일 것 같아." 지금 ①은 읽기 전용이고, 장면은 템플릿이 가리키는 손으로
쓴 MJCF이며, 다섯 개의 IR 문서는 테스트 파일 안의 생성기가 쓴다. MJCF와 Rust를 쓰지 않는
사람은 새 과제를 만들 수 없다.

| 질문 | 결정 (소유자, 2026-10-01) |
|---|---|
| 첫 설계의 범위 | **편집과 과제 정의를 함께**: 뷰포트에서의 장면 편집, 성공/실패 문장, 리셋과 무작위화, 다섯 IR 문서의 생성 — 설계 하나, 패킷 여러 개 |
| 장면의 단일 출처 | **새 장면 문서**(`*.esscene`)이고 MJCF로 내보낸다. MJCF 자체가 아니다 |

이 아래의 나머지는 모두 기본값이 있는 제안이다. 소유자가 아직 정하지 않은 것은 9절에 모았다.

## 2. 사람이 하는 일 (`editor-redesign.md` §3의 ①과 ② 화면을 구체화)

1. **시작**: 시작 화면의 *빈 프로젝트* 카드(또는 어느 프로젝트에서든 *템플릿으로 저장*).
   빈 프로젝트는 바닥, 조명, 바깥쪽 카메라 하나로 열린다.
2. ①에서 **장면을 만든다**: *추가* → 로봇(라이브러리에서: SO-101, Shadow Hand; 또는 URDF /
   MJCF / USD / glTF 파일), 물체(상자, 구, 원기둥, 캡슐, 또는 메시 파일), 카메라, 조명,
   영역(문장이 이름으로 부를 수 있는 보이지 않는 상자나 구역: "상자", "목표 구역").
   뷰포트나 계층 구조에서 선택하고, 기즈모로 이동 / 회전 / 크기 조절을 한다. 인스펙터에서는
   이름, 모양, 크기, 질량, 마찰, 색 / 재질 / 텍스처, 관절(고정, 자유, 힌지, 슬라이드)과 그
   한계를 고친다. 삭제, 복제, 실행 취소 / 다시 실행. *물리 미리보기* 버튼은 모든 것을 3초 동안
   떨어뜨려 보고 되감아 재생한다. 정책의 카메라 화면은 항상 뷰포트 모서리에 있다(실제 관찰,
   H8의 렌더 모드를 H9가 프로세스 안에서 그린다).
3. ①의 문장 편집기에서 **성공이 무엇인지 말한다**: "[큐브]가 [상자] [안에] 있다", "[큐브]가
   [1초] 동안 [정지]해 있다", "[큐브]의 방향이 [goal]의 방향과 [6°] 이내로 같다", "[큐브]가
   [palm]에서 [24cm]보다 멀어지면 실패", "[8초] 안에 끝나지 않으면 실패". 무엇이 어디서
   시작하는가: "시작할 때 [큐브]는 [테이블 구역] 안 무작위 위치에 놓인다(🎲)", 무작위화 강도
   하나(약 / 보통 / 강)와 속성마다 🎲.
4. **정책이 무엇을 보는지 말한다**: 어느 카메라(와 해상도), 어느 관절. 쓰는 동안 쉬운 말로
   검사가 돈다("앞쪽 카메라는 시작 위치의 큐브를 보지 못한다").
5. **② 가르치기**는 지금과 같다: 동작 블록(S3, IK 필요 — 지금은 SO-101), 보상으로 학습한
   교사(플랜 H가 만든 방법: 보상은 같은 문장에서 나온다), 원격조종(S6).
6. **③ / ④ / ⑤**는 템플릿 프로젝트와 똑같다.

여기서 에디터 안에서 학습이나 물리를 돌리는 것은 없다(§23.1, §4.2 규칙 4). 미리보기와 생성은
에디터가 argv를 넘겨 주는 `es` 하위 명령이거나 문서들의 순수 함수다. 렌더만은 예외다. 소유자의
2026-10-01 결정("에디터도 Vulkan 으로 그려줘", 카메라를 움직일 때마다 재질 보기가 단색으로
떨어진 뒤)에 따라 뷰포트는 Vulkan 장치 하나에서 `es-render`로 프로세스 안에서 그린다(패킷
M16/H9, `editor-redesign.md` 5절, S4). 장치가 없는 기계를 위해 `es render`는 남는다.

## 3. 장면 문서 (`*.esscene`)

프로젝트 안의 TOML 파일로, 사람의 장면이다. `task.toml`(§14.3)과 같은 작성 형식이며 새 IR이
아니다. 다른 모든 리더가 만들어 내는 것과 같은 `SceneDesc`로 읽히므로, **`scene_hash`와
`asset_hash`는 파일 형식과 무관하게 `SceneDesc`의 것이다**(지금의 정의, §5.3).
`SceneRef.path`는 `.esscene`을 가리킨다(경로는 지금처럼 해시 입력이다).

```toml
kind = "scene"
schema = 1

[physics]                       # SceneDesc::options; absent = MuJoCo's defaults as today
timestep = 0.008333333333333333
integrator = "implicitfast"

# A robot or any multi-body asset is a *reference* to its own file, placed with a pose and a
# name prefix; the reader expands it into the SceneDesc (section 3.2).
[[include]]
name = "hand"
source = "robots/shadow_hand/shadow_hand.xml"   # MJCF, URDF, USD or glTF
pos = [1.0, 1.25, 0.15]
quat = [0.0, 0.0, 0.0, 1.0]     # xyzw (spec 3.1)

[[body]]
name = "cube"
pos = [1.0, 0.867, 0.177]
joint = "free"
[[body.geom]]
shape = { box = [0.03, 0.03, 0.03] }
mass = 0.216
friction = [1.0, 0.0, 0.0]
material = "block"

[[texture]]
name = "block"
file = "textures/block.png"
kind = "cube"
gridsize = [3, 4]
gridlayout = ".U..LFRB.D.."

[[material]]
name = "block"
texture = "block"
roughness = 0.6

[[camera]]
name = "top"
pos = [1.04, 0.88, 0.56]
quat = [0.0, 0.0, 0.0, 1.0]
fovy = 45.0

[[light]]
name = "ceiling"
kind = "area"
pos = [1.0, 0.9, 0.9]
size = [0.6, 0.6]
intensity = 1.0

[[region]]                      # a site: no collision, no mass; sentences name it
name = "target_area"
size = [0.1, 0.1, 0.01]
pos = [0.25, 0.0, 0.0]
```

### 3.1 해시 체인을 정직하게 지키는 규칙

- **방향은 쿼터니언으로 저장한다**(`scene_hash`에 들어가는 비트). 인스펙터는 오일러 각(도)을
  보여 주고 `es_math::approx`로 변환하므로(자산 경로의 규칙, §5.3 / M10 W0b), 호스트 `libm`이
  해시에 닿지 않는다.
- **정체성은 이름 경로다**(`StableId::from_path`, 지금의 MJCF 바디와 같다): 바디의 id는 트리
  안의 경로다. 이름 바꾸기는 리팩터링이다. 에디터는 그 이름을 쓰는 문서가 어느 것인지 보여 주고
  다시 생성하며(4.2절, 5절), 끊어진 id는 절대 남기지 않는다.
- **자산은 내용으로**: 메시나 텍스처 파일은 프로젝트의 `assets/` 안에 blake3 이름으로 복사되고
  그 경로로 참조된다. `asset_hash`는 내용 기준이므로(HT1의 규칙) 프로젝트를 옮겨도 해시는
  움직이지 않는다.
- **추가형**: 빠뜨린 필드는 `SceneDesc`의 기본값으로 읽힌다. 기존 MJCF가 말하는 것만 적은
  문서는 *같은* `SceneDesc`로 읽힌다 — 8절의 오라클.

### 3.2 인클루드 (로봇과 그 밖의 자산)

`[[include]]`는 그 형식의 기존 리더(MJCF, URDF, USD, glTF)가 펼친다. 바디, 관절, 액추에이터,
센서, 텐던, 접촉 쌍, 재질, 자산이 인클루드의 이름을 접두어로 달고, 인클루드의 자세에 놓인
고정 또는 자유 루트 아래로 들어간다. 인클루드된 파일은 장면 문서에 복사되지 않는다. 프로젝트는
내용 해시가 붙은 사본을 `assets/` 아래에 둔다. 인스턴스별 덮어쓰기(관절의 범위, 액추에이터의
게인, 재질)는 자산 자신의 이름으로 가리키는 `[include.set]` 테이블이다 — 게임 엔진의 프리팹
덮어쓰기와 같다. 복사본을 납작하게 펴지 않고 참조로 두는 이유: 로봇의 기구학이 사는 곳이 그
파일 하나로 남고, 그 파일을 고치면 그것을 쓰는 모든 장면에 닿으며(해시가 눈에 보이게
움직이면서), 문서는 읽고 diff할 수 있을 만큼 작게 유지된다.

### 3.3 내보내기

`es scene export <file.esscene> --mjcf <out.xml>`은 펼친 `SceneDesc`를 자기 완결적인 MJCF
하나로(카메라, 조명, 재질, 텍스처, 텐던, 쌍, gravcomp — `SceneDesc`가 담는 모든 것) 자산을
옆에 두고 쓴다. `es-assets`(계층 2)의 완전 충실도 라이터이며, 동역학이 없는 것은 일부러
버리는 `es-physics-backend`의 `mjcf_out`이 아니다. URDF와 USD 내보내기는 나중 항목이다.

### 3.4 G1이 정한 세부 (구현된 스키마)

`crates/es-assets/src/esscene/`(`EsScene::from_toml` / `to_toml`, `expand`), 오라클
`crates/es-assets/tests/esscene.rs`. 위 예시와 다른 점과 위에서 열어 둔 점:

- **최상위**: `kind = "scene"`, `schema = 1`, `name`(`SceneDesc::name`, 해시 입력; 없으면
  `"scene"`), `[physics]`(필드마다 선택, 없으면 MuJoCo 기본값; `integrator`는
  `euler|rk4|implicit|implicitfast`), `[[geom]]`(월드 바디에 붙는 정적 배경 — 테이블, 바닥,
  통). 모든 선택 필드는 `Option`이라 쓴 것만 다시 쓴다(읽기 ∘ 쓰기 = 항등, 속성 테스트).
- **인클루드**: `name`은 문서 안의 손잡이(과제가 로봇을 이 이름으로 부른다)이고, 이름 접두사는
  따로 `prefix`다. 없으면 파일 자신의 이름과 **id**를 그대로 쓴다(거울 오라클이 성립하는 이유).
  있으면 모든 이름 앞에 붙이고 id를 MJCF 이름 경로 규칙으로 다시 만든다. 파일의 루트 바디는
  월드 아래로 들어가고(감싸는 바디 없음), `pos`/`quat`가 단위가 아니면 루트 바디와 월드 수준
  요소에 합성된다. 파일이 이름 붙인 메시·텍스처 경로는 문서 디렉터리 기준으로 다시 붙는다.
  파일의 `<option>`과 모델 이름은 읽지 않는다(장면의 것은 문서의 `[physics]`와 `name`).
  `[include.set.joint.<이름>]`(`range`, `damping`, `armature`, `stiffness`, `frictionloss`),
  `[include.set.actuator.<이름>]`(`kp`, `kv`, `ctrlrange`, `forcerange`),
  `[include.set.geom.<이름>]`(`rgba`, `material` — 파일의 재질 또는 문서의 재질). 없는 대상은
  필드 이름으로 거부한다. MJCF·URDF·glTF는 읽고, **USD는 거부한다**: `es-usd`는 같은 계층 2의
  형제이고 `SceneDesc`가 아니라 stage를 준다(USD 인클루드는 나중 항목).
- **바디**: `parent`는 앞서 정의된 바디(인클루드된 것 포함)의 이름, 없으면 월드. `joint = { kind,
  name, axis, pos, range, damping, armature, stiffness, frictionloss, springref }` — `kind`는
  `fixed|free|ball|hinge|slide`(`fixed`와 없음은 용접, MJCF처럼 관절을 내지 않는다), `name`이
  없으면 바디 이름. `inertial = { mass, pos, quat, diaginertia | fullinertia }`, `gravcomp`.
  `[[body.geom]]`의 `shape`는 `{ plane | sphere | capsule | cylinder | box | ellipsoid = 크기 }`
  또는 `{ mesh = "파일" }`(자산 이름은 파일 줄기); 이름이 없으면 MJCF처럼 `geom<n>`.
- **메시 배율** (패킷 M17/R3): `shape = { mesh = "파일", scale = [sx, sy, sz] }`, 꼭짓점마다
  축별로 `scale`을 곱한다. 없음과 `[1, 1, 1]`은 비트까지 같은 `SceneDesc`로 읽힌다. 다른 모양
  옆의 `scale`, 모양 키 둘, 양수가 아닌 배율은 필드를 대며 거부한다. `SceneDesc`에
  `mesh_scales`(메시 자산 id별)가 생겼고, 이것이 MJCF의 `<mesh scale>`이다: MJCF 리더는 이제
  이것을 거부하지 않고 읽으며(`refpos`, `refquat`는 여전히 거부), G2의 라이터가 적는다. 맵이
  비어 있지 않을 때만 있는 별도 구역으로 해시되므로 커밋된 `scene_hash`는 하나도 움직이지 않는다.
  `meshes`는 파일 자신의 꼭짓점을 그대로 두어 G2가 파일을 바꾸지 않고 다시 인코딩한다.
  `SceneDesc::mesh_positions`가 배율을 곱한 꼭짓점(`f32(f64(v) · s)`)을 주고, 백엔드 방출기(모든
  백엔드가 읽는 MJCF)와 렌더러가 그것을 시뮬레이션하고 그린다. 배율이 있는 메시의 자산 이름은
  `줄기@s`(균일하지 않으면 `줄기@x,y,z`)라서, 한 파일을 두 배율로 쓰면 `<mesh>`가 둘이다. URDF의
  `<mesh scale>`은 아직 경고다(나중 항목).
- **텍스처는 이름 붙은 `[[texture]]`**이고 재질은 슬롯(`texture`, `orm`, `metallic_map`,
  `roughness_map`, `normal_map`, `emissive_map`)마다 그 이름을 쓴다: 텍스처 하나를 재질 둘이
  나눠 쓰므로(Shadow Hand의 큐브와 목표) 인라인으로 둘 수 없다. MJCF처럼 `rgba`만 쓴 재질은
  이름일 뿐 그려지는 재질이 아니다.
- **카메라**: `fovy`는 도(MJCF와 같은 변환), `parent`가 없으면 월드에 고정.
- **조명**: 오늘의 렌더러가 받는 그대로 — 월드에 놓인 얇은 발광 상자 `<name>_light`(`size`는
  x·y 반폭, 두께 반폭 0.005 m, `rgba = [rgb × intensity, 1]`, 충돌 없음). `kind`는 지금
  `area` 하나.
- **영역**: 사이트다. `SceneDesc::Site`에 모양이 없으므로 `shape`가 아니라 `size`(반폭).
- **쿼터니언 비트**: 이미 정규형(단위에서 1e-12 안, `w ≥ 0`)인 쿼터니언은 쓴 비트 그대로
  저장하고, 그 밖의 것은 `Quat::normalize`로 정규화한다. 에디터가 `SceneDesc`의 비트를 다시
  쓰므로 왕복이 비트를 바꾸지 않는다.
- **순서**: 월드 바디, 인클루드들(차례로), 그다음 문서 자신의 것 — 월드 지오메트리와 조명,
  텍스처와 재질, 바디, 카메라, 영역 — 각각 문서 순서. 거울 오라클(SO-101, Shadow Hand)은
  `SceneDesc`의 모든 값과 순서, `scene_hash`(커밋된 문서들이 지닌 값), 에셋 내용 해시가 같음을
  확인한다. 집합으로 비교하는 것은 `assets` 목록 하나다: Shadow Hand 파일은 큐브의 텍스처·재질을
  손의 것 사이에 끼워 두는데 인클루드의 자산이 문서의 것보다 먼저 오며, 그 순서를 읽는 것은
  없다(`scene_hash`가 정렬하고, 로더와 렌더러는 id로 찾는다). Shadow Hand의 바닥은 손 파일에
  남는다: 빈 `floor0` 바디가 MuJoCo의 바디 순서에서 손보다 앞서기 때문이다. 에셋 경로는 해시
  입력이므로 Shadow Hand 문서는 원래 파일 옆(`tests/fixtures/mjcf/shadow_hand/`)에 둔다.
- `SceneRef.asset_hash`(커밋된 문서에서는 장면 *파일* 바이트의 blake3, M11의 열린 결정)는 파일이
  다르면 당연히 다르다. 같은 것은 `SceneDesc`의 에셋별 해시다. `.esscene`에 대해 무엇을 넣을지
  (인클루드 파일까지 덮을지)는 G3가 문서를 생성할 때 정한다.

## 4. 과제 명세 (`*.estask`)와 `es project generate`

장면 옆의 사람 쪽 파일 하나 더: **과제가 무엇인지, 문장 편집기의 말로**. `teach.toml`(S3)처럼
이것은 레시피이지 IR이 아니다(§5.1 규칙 6). 장면의 물체와 정해진 어휘의 관계를 이름으로
가리키고, `es project generate`가 이것을 장면과 함께 다섯 IR 문서, 학습 레시피, 사이클로
컴파일한다. 같은 입력이면 같은 바이트다(생성기는 결정적이며 시계가 없다).

```toml
kind = "task-spec"
schema = 1
scene = "scene.esscene"
robot = "hand"                       # an include; its actuators become the ActionSpec
control_hz = 60

[success]                            # all clauses must hold
clauses = [
  { subject = "cube", relation = "orientation_matches", object = "goal", within_deg = 5.73 },
]
[failure]                            # any clause ends the attempt as a failure
clauses = [
  { subject = "cube", relation = "farther_than", object = "palm_ref", m = 0.24 },
]
timeout_s = 8.0

[start]                              # reset; 🎲 = randomized, strength scales the ranges
strength = "medium"
items = [
  { what = "cube.yaw", dice = true },
  { what = "goal.yaw", dice = true },
  { what = "hand.joints", noise = 0.2 },
]

[observe]
cameras = ["top", "front", "side"]   # the student's views; resolution per camera
camera_px = 96
render = { path = "pt", spp = 32, bounces = 3, exposure = 8 }
state = ["hand.joint_pos", "goal.qpos"]
privileged = ["cube.pose", "cube.vel"]     # the teacher's extra inputs (plan H)

[reward]                             # derived from the sentences; weights are check boxes
success = "a lot"
shaping = ["orientation", "distance"]
```

### 4.1 어휘는 `es-env`가 낮추는 것이다

각 관계는 `es-env`의 보상 / 종료 콘 낮추기(`crates/es-env/src/plan.rs`)가 읽는 Task IR
노드로 컴파일된다: 원천은 `GetJointState`(위치, 속도), `GetBodyPose`(`pos`, `quat`),
`GetBodyVelocity`(자유 바디, G3c부터), `GetSensor`, `GetTime` — 모두 세계 좌표계만. 변환은
`Arith`, `Norm{L2}`, `Dot`, `MathFn{Abs,Sqrt}`, `Compare`, `Logic`, `Normalize`, `Clamp`(레인
하나), 그리고 G3c부터 `Slice`, `Concat`, `Reduce`(축 0). 싱크는 `Reward`와 `Terminate`, 리셋은
`ResetState`와 `Randomization`. 콘 안에서 이름으로 거부되는 것: `GetContact`, `GetRandom`,
`Transform`, `Cross`, `Select`, `Norm{L1,Linf}`, 나머지 `MathFn`.

| 관계 | 컴파일 결과 |
|---|---|
| `inside` 영역 | 축마다: 바디 위치의 `Slice`, 상자의 낮은 면보다 `Compare >`, 높은 면보다 `<`, `And`. 세 축을 `And` (4.4절) |
| `inside` 구간 | 스칼라 주어, lo보다 `Compare >`, hi보다 `<`, `And` |
| `above` / `below` 값 | 스칼라 주어, `Compare` |
| `above` / `below` 바디 (m만큼) | 두 위치의 `Slice` z, `Arith Sub`, `Compare > m` |
| `near` / `farther_than` (m) | `Arith Sub`, `Norm L2`, `Compare` |
| `still` (s 동안) | 바디: `GetBodyVelocity.linear`의 `Norm L2 <` speed(`angular`가 있으면 `.angular`도 그 아래). 좌표: 속도가 ±speed 안 — IR-D에는 유지 노드가 없어서 "1초 동안"은 "안에 있고 거의 정지"로 컴파일된다(데모의 안정화 한계, `editor-redesign.md` §5 S4) |
| `orientation_matches` (deg) | 쿼터니언의 `Dot`, `MathFn Abs`, `Compare ≥ cos(θ/2)` (플랜 H의 구성) |
| `joint` `above` / `below` (그리퍼 열림) | `GetJointState`, `Compare` |
| `touches` | `GetContact` 낮추기를 기다린다 — 들어오면 제공 |

절마다의 보상 성형: 거리와 방향 절은 플랜 H가 쓴 조밀한 항을 낸다(`−k·distance`,
`1/(√(8(1−d))+0.1)`을 구간 선형 곡선으로), 성공 절은 보너스, 실패 절은 벌점. 체크 박스가
세 단계 중에서 가중치를 고른다. 그러면 **시도가 왜 실패했는지**는 빠져 있던 절이다(소유자가
승인한 S4 방향): `es eval run`이 각 시도의 끝에 절마다의 참 거짓을 `episodes.json`에
기록하고, ⑤가 그것을 ①의 문장의 말로 알려 준다. 템플릿의 `[outcome]` 테이블(큐브 상자 넣기,
재배치)은 명세에서 생성된다.

### 4.2 `es project generate`가 쓰는 것

`scene.esscene` + `task.estask`에서: `task.toml`(Task IR), `observation-teacher.toml`,
`observation-student.toml`, `learning-*.toml`(있는 템플릿 계열: 상태 MLP 교사, 3시점 ACT
학생 — `tanh` 헤드, H6의 불변식), `deployment.toml`(장면의 관절과 제어 범위에서 나온
엔벨로프), `evaluation*.toml`(보류된 시드, 적용되는 스위트), `training-*.toml`, `cycle.toml`.
모든 해시는 유도된 것이고 직접 적은 것은 없다. `crates/es/tests/{views,shadow_hand}.rs`에 커밋된
생성기는 회귀 오라클이 된다: SO-101과 Shadow Hand 명세는 커밋된 문서를 바이트 단위로 똑같이
다시 만들어 낸다.

### 4.3 G3a가 정한 세부 (구현된 스키마)

`crates/es-script/src/spec/`(`TaskSpec::from_toml` / `to_toml`, `compile_task(spec, root)`),
오라클은 `crates/es-script/tests/estask*.rs`. 두 명세
`tests/fixtures/estask/{shadow_hand_repose,so101_views}.estask`는
`tests/fixtures/shadow-hand/task-repose.toml`과 `tests/fixtures/visible-learning/task-views.toml`의
`task_hash`로 컴파일된다(의미 해시. `task_graph_hash`는 다르다: 컴파일러는 노드에 명세 순서로
번호를 붙이고, 커밋된 문서는 생성기의 순서로 붙였다 — SO-101은 이력대로라서 그리퍼 절이
31–33에 덧붙어 있다). 위 예시와 다른 점, 그리고 본문이 열어 둔 것:

- **최상위**: `scene`은 `compile_task`에 주는 프로젝트 루트 기준 상대 경로이고, 적힌 그대로
  `SceneRef.path`에 들어간다. `asset_hash`는 어떤 종류든 장면 파일 바이트의 blake3이다.
  `.esscene`도 마찬가지다(인클루드가 가져오는 것은 `SceneDesc`와 자산별 내용 해시를 거쳐
  `scene_hash`에 들어 있다). `robot`은 로봇의 **루트 바디** 이름이지 인클루드 핸들이 아니다
  (인클루드에는 루트가 여럿일 수 있다: Shadow Hand 파일의 `floor0`. G3b부터는 핸들도 받는다,
  4.5절). 그 하위 트리의 관절이
  `robot.joints`이고, 장면의 액추에이터 전부가 그 행동이다(`JointPosition`, 과제당 로봇 하나).
  `timeout_s × control_hz`는 정수여야 한다: 그것이 `max_episode_steps`이고,
  `리셋 이후 시간 ≥ timeout_s`가 `Timeout` 노드다.
- **절**(`[success]`는 전부, `[failure]`는 하나라도): `{ subject, relation, ... }`. 문서 순서대로
  `And`로 접어 `Terminate(Success)` 하나에, `Or`로 접어 `Terminate(Failure)` 하나에 잇는다.
  **스칼라** 주어(`inside`, `above`, `below`, `still`)는 관절(로봇 관절은 로봇의 관절 벡터로서
  루트 바디를 통해 읽고, 다른 관절은 자기 자신으로 읽는다)이거나 `<body>.x`, 곧 바디 자유
  관절의 첫 좌표다 — `GetJointState`가 읽는 단 하나의 레인이다. **바디** 주어(`near`,
  `farther_than`, `orientation_matches`)는 바디다. 필드: `inside`는 `range = [lo, hi]`,
  `above` / `below`는 `value`, `still`은 `speed`(속도가 ±speed 안), `near` / `farther_than`은
  `m`과 `object`(바디: `Arith Sub`, `Norm`) 또는 `point = [x, y, z]`(상수 노드가 없으므로
  `p − point`는 `[point − 1, point + 1]` 위의 레인별 `Normalize`, 플랜 H의 구성이고 `m` < 1)
  중 하나, `orientation_matches`는 `object`와 `within_deg`(`|q·g| ≥ cos(θ/2)`). 관계가 받지
  않는 필드나 빠진 필드는 절(`success[1] (cube.x still)`)과 필드를 짚어 거부하고, 모르는 키는
  이름으로 거부한다.
- **G3a 때 낮아지지 않던 것은 이름으로 거부했다**: `touches`(`GetContact`), `<body>.y` / `.z`,
  그리고 그 때문에 영역 `inside`, 다른 바디 기준 `above` / `below`, 3축 `still` — 콘 낮추기에
  `Slice`, `Concat`, `Reduce`, `GetBodyVelocity`가 없어서 `inside`는 주어의 x 구간이고
  `still`은 x 속도였다. 정확히 task.toml의 한계("상자의 x 구간과 안정화 한계")다. G3c가 네
  노드를 낮추고 관계들을 3축으로 만들었다(4.4). `touches`는 여전히 `GetContact`를 기다린다.
- **성형**은 절마다 붙고, 참조 문서가 4.1보다 더 필요로 한 곳은 이름 있는 선택지로 두었다:
  `shaping`은 항의 형태, `weight`는 가중치, `term`은 이름(없으면 `<subject>_<shaping>`)이다.
  `distance`(`near` / `farther_than`: `weight × 거리`, [0, 1] m로 자름), `ramp`(`inside`,
  `ramp = [a, b]`와 함께: `weight × (s − a)/(b − a)`를 [0, 1]로 자름 — SO-101의
  `cube_towards_bin`), `inverse_angle`(`orientation_matches`: `weight / (s + 0.1)`,
  `s = √(8(1 − |q·g|))`를 플랜 H의 아홉 매듭에서의 구간 선형 보간으로, 항은 `<term>_0..7`,
  그리고 매 스텝 주는 `<term>_floor`). `[reward]`: `scale`은 모든 가중치에 곱한다(rl_games의
  `scale_value`, Shadow Hand의 0.01). `success` / `failure`는 같은 이름의 희소 항이고 접힌
  술어가 먹인다. 문서의 가중치는 숫자다. 체크 박스의 세 단계를 숫자로 적는 것은 문장
  편집기(G8)의 몫이다.
- **시작** 항목 `{ what, ... }`: `robot.joints`(`noise`는 각 관절 범위에 대한 비율, Isaac Lab의
  `reset_dof_pos_noise`. `coupled = true`면 두 관절짜리 고정 텐던으로 다른 관절에 묶인 관절은
  그 관절의 스트림에서 결합 비율만큼 비례해 뽑는다 — 플랜 H의 J0/J1), 관절, `<body>.x|y|z`
  (`value`, `value ± noise`, `range`), 또는 `draw`를 가진 `<body>.orientation`(아래).
  `value`는 `Constant`(잡음 0일 때도), `range`는 언제나 `Uniform`이다(task.toml의
  `Uniform{0, 0}` 홈 자세). `stream`은 뽑기의 이름이다(없으면 `what`, 관절은 `reset.<joint>`).
  같은 스트림의 항목들은 한 번의 뽑기를 나눠 쓴다. `dice = true`(🎲)면 `ResetState` 대신
  `Randomization` 노드가 된다(task.toml의 큐브 x / y). `strength`는 모든 🎲 항목의 `range`를
  중심 기준으로, `noise`를 그대로 배율한다(없으면 적힌 대로).
- **방향 뽑기**(소유자, 2026-10-01: Shadow Hand를 yaw만 돌리기에서 큐브 뒤집기로 바꾸는 것이
  한 문장이어야 한다). 리셋 노드는 수 하나를 쓰므로 쿼터니언의 레인마다 뽑기가 하나다(같은
  스트림의 레인은 그 뽑기를 나눠 쓴다). 쓰인 쿼터니언은 백엔드가 정규화한다 — MuJoCo와
  MJWarp 모두 `xquat`은 즉시, `qpos`는 첫 스텝 뒤에(H2b, `any`에 대해서는
  `the_backends_normalize_a_drawn_quaternion`이 다시 확인):
  - `draw = "yaw"`(+ `tilt`): 플랜 H의 `(1, t, −t·u, u)`, `u ~ U(−1, 1)` 하나. 세계 X축에 대한
    안착 기울기 `2·atan(t)`와 반 바퀴 [−90°, 90°] 안의 yaw `2·atan(u)`.
  - `draw = "tilt"`, `tilt_max_deg = θ`(< 180): `(1, a, b, g)`, `a, b ~ U(−m, m)`,
    `m = tan(θ/2)/√2`, `g ~ N(0, 1)`. 수직에서의 기울기는 θ를 넘지 않고
    (`a² + b² ≤ tan²(θ/2)·(1 + g²)`, `g = 0`이고 모서리일 때 닿는다) 모든 방위가 뽑힌다
    (`2·atan(g)`: 68 %가 ±90° 안, 8 %가 ±120° 밖). 균일한 구성이 아니라(캡 위로도, yaw로도)
    가장 가까운 정확한 구성이다: 기울기 한계에는 yaw 쌍 `(w, z)`가 0에서 떨어져 있어야 하는데,
    모든 방위를 덮는 유계 뽑기로는 그럴 수 없다(그 쌍의 집합은 상자이고, 방향이 반 바퀴에
    걸치는 볼록 집합은 원점에 닿는다). 그래서 yaw 레인을 유계가 아니게 했다.
  - `draw = "any"`: `<stream>.w|x|y|z`의 `N(0, 1)` 레인 넷, 백엔드가 정규화: SO(3) 위의 균일
    분포(`EnvRng`의 Box–Muller 정밀도까지).
- **관측**: `cameras`(채널 `rgb_<camera>`, `camera_px` 정사각 RGB8, 렌더러가 제어 주기로
  내놓는 `ImageSpec`)와 모두에 쓰는 `render` 하나. `state`와 `privileged`는 목록이 아니라
  `channel = "source"` **테이블**이다 — 채널 이름이 해시 입력인데 참조 문서들의 이름은 한
  규칙을 따르지 않는다(`object_vel`, `target_qpos`, `sim_cube_pose`). 출처:
  `robot.joint_pos`(루트 바디의 `JointState`: 앞쪽 `dof`개, 로봇의 관절이 장면 관절의 맨
  앞이 아니면 거부), `robot.joint_vel`(첫 관절의 `JointState`: id마다 입력 버퍼가 하나),
  `robot.previous_action`(`initial`은 각 ctrlrange의 중앙), `<body>.pose`(`BodyPose`),
  `<body>.qpos` / `<body>.vel`(자유 관절의 7 / 6). `state`와 `privileged`(교사의 추가 입력)는
  Task IR에서는 `ObservationSpec` 하나이고, 나눔은 G3b가 읽는다.
- **libm에서 오는 수**: `within_deg`의 코사인, 카메라 초점 거리, `tilt_max_deg`의 `m`은 자산
  경로의 수처럼(§5.3, M10 W0b) `es_math::approx`(`sin_cos_f64`, `tan_f64` — `libm` 크레이트,
  musl)에서 오므로, **생성된 문서는 어느 호스트에서나 같은 비트를 가진다**(G3d, 2026-10-01; G3b는
  호스트의 `cos` / `tan`을 두었다). 커밋된 해시는 하나도 움직이지 않았다: 커밋된 각도(0.05 rad의
  코사인, 22.5°와 35°의 `tan`)에서 musl과 이 PC의 UCRT는 같은 비트를 주며, 둘 다 정확히
  반올림된 값이다. 그 밖에서는 둘이 충실(faithful)할 뿐 같지는 않다: 1°–179°를 0.001° 간격으로
  훑으면 `tan`은 356,002개 인자(`x`와 `x / 2`) 중 14,720개에서, 반각 코사인은 178,001개 중
  4,902개에서 한 ULP 다르고, 정확히 반올림된 쪽이 늘 같은 편은 아니다 — 예컨대 68° 카메라가 있는
  문서를 G3d 전에 Windows에서 생성했다면 오늘의 문서와 초점 거리의 마지막 비트가 다르다. (`tan`을
  sin / cos로 구하는 길은 택하지 않았다: 45° 카메라의 초점 거리가 한 ULP 움직인다,
  `115.88225099390856` 대 커밋된 `…857`.) 아직 호스트를 부르는 곳: `crates/es/tests/{shadow_hand,views}.rs`의
  픽스처 생성기들로, G3b의 테스트가 비교 대상으로 삼으며 커밋된 각도에서는 일치한다.

### 4.4 G3c가 정한 세부 (3축 관계)

`crates/es-env/src/plan.rs`(오라클 `slice_concat_reduce_and_body_velocity_lower`)와
`crates/es-script/src/spec/compile.rs`의 절 함수들(오라클 `crates/es-script/tests/estask*.rs`,
픽스처 `tests/fixtures/estask/so101_region.esscene`). 두 참조 명세는 여전히 커밋된
`task_hash`로 컴파일된다. 커밋된 어떤 콘에도 네 노드가 없었으므로(거부되었으니까) 커밋된
문서는 모두 전과 같은 `Expr`로 낮아진다.

- **콘 안의 네 노드.** 콘의 값은 레인 한 줄이라 축은 0 하나뿐이다. 다른 축이나 끝을 넘는
  슬라이스는 이름으로 거부한다. `Slice`는 레인 `start..start+len`을, `Concat`은 입력
  `in0, in1, …`을 차례로 잇고, `Reduce`는 레인 순서로 접는다 — `Norm`과 `Dot`이 이미 쓰던
  결합 순서다(`DET-020`). `Mean`은 그 합을 레인 수로 나눈 것이다. `unordered = true`도 같게
  낮춘다(레인 순서도 허용되는 순서 중 하나다. 결정론 모드에서는 검증기가 거부한다, `DET-030`).
- **`GetBodyVelocity`는 `StateView`가 가진 것에서 정확히 유도한다**: 바디 속도 배열은 없지만,
  자유 바디의 `qvel` 여섯은 MuJoCo의 자유 관절 규약이다 — 바디 원점의 선속도는 세계
  좌표계로, 그다음 각속도는 바디 좌표계로. 모든 백엔드의 뷰가 이를 따른다(`physx_ref.py`가
  PhysX의 값을 이 규약으로 바꾼다). `linear`는 앞의 세 레인 그대로다(`GetJointState(Velocity)`가
  묶는 포트와 같다: `cube.x still`과 `cube still`은 같은 수를 읽는다). `angular`는 뒤의 세
  레인을 `xquat`(`GetBodyPose.quat`이 읽는 방향)으로 세계 좌표계에 돌린 것이다,
  `v + w·t + u × t`, `t = 2 u × v` — 다항식이라 `DET-010` 함수가 없다. `mujoco-cpu`에서
  x축으로 60° 기운 채 자기 z축으로 5 rad/s 도는 큐브는 `5 R e_z`를 2e-15 안으로 읽는다
  (`a_spinning_cubes_angular_velocity_is_read_in_the_world_frame`. 세계 좌표계 값으로 잘못
  읽었다면 `(0, 0, 5)`였을 것이다). 자유 관절이 없는 바디는 야코비안이 필요한데 `StateView`에
  없으므로 이름으로 거부하고, 세계 외의 좌표계도 거부한다.
- **영역**은 사이트다 — `.esscene`의 `[[region]]`이나 MJCF의 `<site>` — 그리고 `size`가
  상자의 반폭이다. 상자는 **세계 좌표계 축에 정렬**된다: 중심은 사이트의 위치에 그 위
  바디들의 위치를 더한 것이고(세계에서 아래로 합한다), 면들은 `Compare`의 리터럴로 콘에
  들어간다. 콘에는 벡터를 회전시킬 상수 노드가 없고(세계의 점은 리터럴로만 들어온다 — 점에
  대한 `near`가 레인별 `Normalize`인 이유다), 에디터가 쓰는 영역은 회전이 없다. 그래서 회전한
  사이트는(자신이든 위의 어느 바디든) 이름으로 거부하고, 움직이는 사이트(자기 바디나 그 위에
  관절이 있는 것)도 거부한다. 안쪽은 엄격하다: 면 위는 바깥이다. 영역 `inside`에는 R5 전까지
  성형이 없었다(4.7.1절).
- **주어.** 자유 바디의 `<body>.x`는 여전히 자유 관절의 첫 `qpos` 레인이다(G3a의 형태.
  SO-101의 해시가 거기에 달려 있다). 나머지 `<body>.x|y|z` — `y`, `z`, 그리고 자유 관절이 없는
  바디의 `x` — 는 `GetBodyPose.pos`의 `Slice`이고, `still`에서는 `GetBodyVelocity.linear`의
  `Slice`다(자유 바디). 두 위치 배열은 MuJoCo에서 같은 좌표계지만 같은 순간은 아니다:
  `mj_step`은 적분하기 전 운동학에서 `xpos`를 계산하므로 스텝 뒤의 `xpos`는 `qpos`보다 물리
  서브스텝 하나 늦다(리셋 직후에는 같다). `near`와 `farther_than`은 이미 `xpos`를 읽는다.
- 바디를 가리키는 주어의 **`still`** — 관절이 아니고(관절 이름이 이긴다: SO-101의 `gripper`는
  둘 다다) `<body>.<axis>`도 아닌 것 — 은 `‖v‖ < speed`이고, `angular`(rad/s, 새 필드)가
  있으면 `‖ω‖ < angular`도 붙는다. 바디에는 자유 관절이 있어야 한다. 좌표 주어는 G3a의
  `−speed < v < speed`를 그대로 쓰고 `angular`를 거부한다.
- **바디 기준 `above` / `below`**: `object`와 `m`(여유, 없으면 0). above는
  `z_subject − z_object > m`, below는 `z_object − z_subject > m`.
- `object`는 `range`(`inside`), `value`(`above` / `below`), `point`(`near`, `farther_than`)와
  함께 쓸 수 없다: "either `object` or …", 절을 짚어서.

### 4.5 G3b가 정한 세부 (`es project generate`)

`crates/es-script/src/spec/`의 `generate(spec, root, out, source)`(`generate.rs`, `learning.rs`,
`recipes.rs`, 섹션은 `project.rs`, `robot.rs`)와 동사
`es project generate --spec <x.estask> --out <dir> [--scene <s>]`(경로는 현재 디렉터리, 곧
프로젝트 루트 기준이고, `--out`은 주어진 그대로 사이클에 적힌다). 오라클은
`crates/es-script/tests/estask_generate.rs`(해시, 결정성, 거부)와 `crates/es/tests/project.rs`(각
팔에 `es ir check`와 `es policy init`, 레시피와 사이클의 dry-run). 명세에 선택 섹션 다섯 개가
붙는다. 모든 기본값은 커밋된 두 세트의 값이므로, 과제가 그것들과 다른 곳에만 필드를 쓴다. 모든
문서는 검증되고, 각 팔은 무엇이든 쓰기 전에 교차 IR 검사를 통과한다.

- **`[teacher]`**(없으면 교사도 없다): `state`는 상태 벡터 순서대로 놓인 채널들이다(`[observe]
  state`와 `privileged` 어느 쪽이든. 없으면 `state` 전부, 이어서 `privileged` 전부). `training`은
  `training.toml`의 어느 부분이든 되며 PPO 프리셋(플랜 H의 `training-teacher-v2.toml`) 위에
  병합된다(모르는 키는 `Recipe`가 이름으로 거부한다). 문서: 플랜 H의 상태 MLP(ELU
  `[512, 256] → 128`, `tanh` 헤드, 틱마다 한 행, ctrlrange 역정규화, 마감은 정수 ms로 된 제어
  주기 하나), 그 배포(horizon 1), 그 평가(공칭 스위트).
- **`[student]`**: `name`(팔 이름: `observation-<name>.toml`, …. 없으면 `student`), `views`
  (`[observe] cameras`에서. 없으면 전부), `state`(`[observe] state` 채널만 — 특권 채널은 거부),
  `family`(`act`는 플랜 N의 뷰마다 ResNet18을 두고 `Concat`으로, `mad`는 뷰들이 첫 인코더를 함께
  쓰고 더해지며 프리셋에 단일 뷰 손실이 들어 있다), `preset` — 커밋된 두 학생 사이에서 다를 뿐
  다른 뜻은 없는 관례: `h3`(기본, 플랜 H의 것: `tanh` 헤드와 ctrlrange 역정규화, H6의 불변식.
  상태는 이어 붙여 표준화한 `state` 입력 하나. 모든 뷰는 카메라 이름의 포트로 융합) 또는 `u3`
  (SO-101 데모의 것: 행이 곧 목표인 무한 헤드. 상태 채널 하나를 제 이름으로 `[-1, 1]`로. 첫 뷰는
  `image` 포트로) — `horizon`과 `execute`(필수. `control_hz / execute`는 정수, XIR-023), 그리고
  ACT 프리셋(플랜 N의 `training-views.toml`)이나 MAD 프리셋(`training-mad.toml`) 위의
  `training`.
- **관측**: 뷰마다 Task IR 채널의 `ImageSpec` 위에 플랜 U의 사슬(`Dequantize`,
  `Normalize [0, 1]`, `Pad 4`, `Crop Random`, 학습 때만 `ColorJitter 0.2 / 0.2`). 상태 통계는
  장면과 절에서 온다: 관절 위치는 그 범위로, 관절 속도는 5 rad/s로, 이전 행동은 ctrlrange로,
  바디 위치는 그것을 한 점에서 재는 거리 절로(그 점, 그 반경: 떨어짐 반경) 아니면 장면 위치
  ± 1 m로, 쿼터니언은 그대로, 속도는 1 m/s와 5 rad/s로.
- **`[deploy]`**: `name`과 `workspace`(없으면 `robot`, 루트 바디 ± 1 m). 포락선은 장면의 것이다 —
  각 ctrlrange와 forcerange, 플랜 H의 속도 규칙(속도 `2 w hz`, 가속도 `4 w hz²`, 차분 `2 w` /
  `4 w`) — 제어 주기는 장면의 timestep을 정수 나노초로 둔 것을 데시메이션으로 정확히 나눈 것이고,
  추론 주기는 그것을 다시 `execute`로 나눈 것이다.
- **`[evaluate]`**: `first_seed`, `episodes`, `success_rate`(없으면 101, 16, 0.5). 학생의 평가는
  플랜 U의 섭동 스위트 다섯 개(지연은 정수 ms로 된 제어 주기 하나와 둘)를 더하고 `-nominal`
  형제가 있다. 교사의 평가는 공칭 스위트다.
- **`[cycle]`**: `runs`(없으면 `runs`), `expert`(없으면 학습된 교사 `<runs>/teacher.esb`),
  `episodes`, `seed`, `success_only`, `nominal_only`, `jobs`, `preview`(없으면 켬), `showcase`.
  레시피의 경로는 `runs`에서 나온다 — `<runs>/teacher-untrained.esb`,
  `<runs>/<name>-untrained.esb`, 데이터셋 `<runs>/<name>-001/collect[/successes]/{ds,frames}` —
  그리고 전문가의 수집은 학생 번들(그 Task IR) 아래 기록된다. 레시피마다 머리말에 문서에서 번들을
  만드는 `es policy init` 줄이 있다. 동사가 번들을 직접 만들지는 않는다.
- **`robot`**은 `.esscene` 인클루드를 가리켜도 된다(오케스트레이터, 2026-10-01): 하위 트리에 관절이
  있는 루트 바디 하나(`hand` → `floor0`을 지나 `robot0:hand mount`). 여럿이면 거부한다.
- **재현된 것**(IR 문서는 의미 해시로, 레시피와 사이클은 파싱한 `Recipe` / `Cycle`로):
  `tests/fixtures/shadow-hand/`의 모든 문서(`evaluation-teacher.toml`은 `episodes = 16`에서,
  `-64`는 명세의 64에서), SO-101의 세 뷰 팔(`*-views.toml`), 한 뷰 팔(`views = ["overhead"]`,
  `*-cam.toml`), MAD 세트(`family = "mad"`, `learning-`, `evaluation-`, `training-`,
  `cycle-mad.toml`). IR 문서 스무 개 중 열두 개는 바이트까지 같다. 나머지는 노드 번호를 다른
  순서로 붙였다. 커밋된 실행이 쓴 경로 가운데 규칙이 주지 않는 것은 두 명세의 덮어쓰기다
  (`teacher-untrained-v2.esb`, 플랜 N의 `runs/collect-001/`). **재현하지 않은 것**:
  `visible-learning/deployment.toml`, 장면의 범위가 주지 않는 손으로 맞춘 포락선(3 rad/s,
  80 rad/s², 0.05 rad 소프트 여유). SO-101 명세는 플랜 H의 규칙을 받는다.

### 4.6 G8이 정한 세부 (문장 편집기)

`crates/es-editor-scene/src/sentence.rs`(문장, 편집, 단계, "과제 정하기", 카메라 검사)와
`crates/es-script/src/spec/vocab.rs`(컴파일러가 읽는 그대로의 어휘)이고, 그리는 쪽은
`crates/es-editor/src/ui/sentence.rs`다. 오라클은 `crates/es-editor-scene/tests/sentences.rs`와
`crates/es-editor/tests/sentences.rs`(문장의 낱말. 옆의 `sentences.txt`에 고정해 두었다. 한국어는
문서와 문자열 표에만 둘 수 있다).

- **명령 하나.** `Command::Spec(Option<Box<TaskSpec>>)`가 명세를 바꿔 넣는다. `SceneModel::apply`는
  이 명령뿐 아니라 **모든** 명령 뒤에 편집된 장면 위에서 `compile_task`로 명세를 컴파일한다(저장하지
  않은 장면은 `.preview.esscene`에서 읽는다). 그래서 문장을 깨뜨릴 장면 편집(절이 가리키는 목표를
  지우는 것)도 거부되고, 모델의 명세는 언제나 컴파일되거나 아예 없다. 편집 전부터 깨져 있던
  명세(손으로 고친 파일) 때문에 장면 편집을 거부하지는 않는다. 그것은 문장에서 고친다. 거부는 G3a의
  오류를 `Refusal`로 옮긴 것이다. 필드는 절의 자리와 키(`success[0].within_deg`, `timeout_s`,
  `observe.state.goal_pose`)이고, 말은 표의 키(`author.task.required`, `.ticks`, `.touches`,
  `.no_success`, 그 밖에는 컴파일러의 이유)다.
- **어휘는 컴파일러의 것이다.** `vocab::takes(relation, object)`는 `check_fields`가 읽는 표를 밖으로
  옮긴 것이다(컴파일된 바이트는 하나도 바뀌지 않았다. G3a와 G3b의 오라클이 그대로 통과한다).
  `vocab::subjects`: 월드를 뺀 모든 바디, 모든 힌지·슬라이드 관절, 모든 자유 바디의
  `<body>.x|y|z`. `vocab::relations(scene, subject)`: 관절이나 좌표는 범위 안, 값보다 위나 아래,
  또는 멈춤(좌표는 자유 바디의 것만). 바디는 영역 안, 다른 바디보다 위나 아래, 바디나 고정된 점에서
  가까움이나 멂, 멈춤(자유 바디), 다른 바디와 같은 방향, 또는 닿음(이유와 함께 비활성으로
  보인다). 관절이면서 바디인 이름(SO-101의 `gripper`)은 둘 다 될 수 있는 곳에서 컴파일러처럼
  관절로 읽고, 바디만의 관계도 받는다. `vocab::regions`: 움직이지 않는 바디 위의 사이트. 영역은
  목적어일 뿐 주어가 되지 않는다(컴파일러가 영역을 주어로 읽지 않는다).
- **문장.** 절, 제한 시간, 시작 항목은 번호 붙은 구멍(`{0}`은 주어, `{1}`은 관계, 그다음은 그
  필드)을 가진 표의 키다. 그래서 언어마다 칸을 읽히는 자리에 둔다. 칸은 이름, 관계, 뽑는 방식,
  숫자, 점 가운데 하나다. 숫자는 cm, °, cm/s, °/s, s, %로 보이고(쉬는 기울기 `t`는 그 각도
  `2·atan t`로) 소수는 둘째 자리까지다. 문서는 제 단위와 비트를 지킨다. 칸은 문서의 값을 들고(없는
  값은 없는 채로), 숫자는 사람이 바꿨을 때만 되써진다(기울기는 `es_math::approx`를 거쳐서).
  오라클 1: 커밋된 두 명세를 문장으로 읽어 칸마다 되쓰면 필드마다 같다.
- **편집.** 새 주어는 지금 관계를 받으면 그대로 두고, 아니면 첫 관계를 받는다. 새 관계는 자기가
  받는 필드는 두고, 꼭 있어야 하는 필드는 채우며(주어 자리 ± 5 cm 범위, 그 자리의 값, 5 cm/s,
  10°, 가까움 5 cm와 멂 25 cm, 첫 영역이나 다른 바디 — 자유 바디가 먼저), 형태가 같으면 셰이핑을
  그대로 둔다. 절은 추가하고(첫 자유 바디가 멈춤), 지우고, 제 섹션 안에서 옮긴다. 비어 버린 실패
  섹션은 없앤다. 시작 항목은 추가하고(첫 자유 바디의 x를 지금 자리 ± 2 cm로, 🎲) 지우며, `what`,
  🎲, 값, 범위, 흔들림, 뽑는 방식이 바뀐다(`tilt`는 30° 한계를 함께 가져온다). 관측은 장면 카메라의
  체크 목록(켠 카메라는 끝에 붙고, `camera_px`가 없으면 96. 끈 카메라는 학생의 `views`에서도
  빠진다), `camera_px`, 그림, 감각이다(새 채널 이름은 그 출처에서 따온다, `cube.pose` →
  `cube_pose`. 지운 채널은 교사와 학생의 `state`에서도 빠진다).
- **단계**(9절 5번. 소유자가 바꿀 수 있다). 셰이핑 가중치는 `0.1`, `1`, `10`이고 부호는 그 형태가
  주는 대로다(거리와 램프는 비용, 각도는 보상). 성공 보너스는 `25`, `100`, `250` 또는 없음.
  `[start] strength`는 `0.5`, 없음, `1.5`다(es-script는 모든 🎲 범위를 그 중심에 대해, 그리고
  흔들림을 이 값만큼 늘이고 줄인다. 보통은 `strength`를 쓰지 않는다. `1.0`이 없는 값과 비트까지
  같지는 않기 때문이다 — `c ± 1·h`가 `lo`와 `hi`를 그대로 돌려주지 않을 수 있다). 커밋된 가중치는
  단계로 읽힌다: Shadow Hand의 회전 `1`은 보통, 큐브 거리 `−10`은 많이, 보너스 `250`은 많이.
  SO-101의 램프 `−1`은 보통. 어느 단계에도 없는 숫자는 "적힌 대로 (0.37)"로 읽히고 그대로
  남는다. 켠 램프는 범위의 아래 끝에서 시작해 위 끝에서 범위 폭만큼 더 간 곳에서 끝난다. 가중치는
  `[reward] scale`을 곱하기 전의 값이고, 문장은 scale을 보여 주지 않는다.
- **그림**: 빠른 그림(`render` 없음) 또는 빛 추적(`pt`, 플랜 H의 샘플 32, 반사 3, 틱마다 시드).
  샘플과 반사는 숫자로 고친다. 뷰포트의 재질 그림은 비활성으로 보인다. Task IR 센서에는
  래스터라이저와 경로 추적기만 있다.
- **과제 정하기**(`task.estask`가 없는 프로젝트): 로봇은 장면 문서의 첫 인클루드, 없으면 첫 힌지나
  슬라이드 관절이 달린 바디의 루트다. `control_hz`는 템플릿 Task IR의 `control_rate_hz`(SO-101은
  50, 없으면 50). 8초. 성공 절 하나, 첫 자유 바디가 멈춤(`speed = 0.05`, SO-101의 안정 한계). 장면의
  모든 카메라를 96 px로 관측한다. 오라클 4: SO-101 복사본에서 상자 위에 영역을 두고 "[cube]가
  [bin_area] 안에 있다"를 더하면 컴파일되고, `generate`가 `task.toml`을 쓴다.
- **카메라 검사.** 관측하는 카메라마다, 그리고 절이 말하는 바디마다(좌표면 그 바디, 관절은
  건너뛴다): 카메라에서 시작 항목이 두는 자리(자유 바디의 `x|y|z` 항목: 그 값이나 범위의 가운데)의
  바디 원점까지 광선 하나를 쏜다. 그 점이 정사각형 그림의 시야 안에 있어야 하고, 가장 가까운
  충돌 — 장면 삼각형 위의 `nearest_hit_flat`, G6의 규칙 — 이 그 바디 자신의 모양이어야 한다.
  SO-101 복사본에서 잰 결과: 위 카메라는 시작 위치의 큐브를 **보지 못한다** — 홈 자세의 아래팔이
  큐브 중심으로 가는 선 위에 있다(모서리 화면도 같다). 오라클 5: 큐브와 같은 높이에서 큐브를 보는
  카메라는 본다. 돌려 놓거나 판 뒤에 두면 못 본다. 큐브의 시작을 판 위로 올리면 다시 본다. 중심으로
  가는 광선 하나가 패킷의 규칙이므로, 반쯤 가려진 큐브는 못 보는 것으로 읽힌다.
- **에디터.** ①의 오른쪽 창에 탭이 둘 있다: 고른 것의 필드와 과제. 문장의 위젯은 읽는 사람의
  어순대로 낱말 사이에 놓인다. 고친 명세는 사람이 손을 뗀 뒤에 넘겨지고 실행 취소 한 단계가 된다.
  거부된 명세는 문장에 그대로 남고, 이유는 위에, 그리고 그 이유가 가리키는 절 아래에 다시
  나오며, "되돌리기"가 있다.

### 4.7 GV가 정한 세부 (정하지 않은 좌표가 어디서 시작하나)

GV의 직접 만든 프로젝트(SO-101, (0.22, 0, 0.025)의 5 cm 상자, 시작 항목은 `box.x`와 `box.y`뿐)에서
드러난 문제: `Env::reset`은 리셋 노드를 돌리기 전에 `qpos`를 0으로 채운다. 그래서 시작 항목이 정하지
않은 자유 바디의 좌표는 **0**에서 시작했다. 상자는 z = 0, 반쯤 바닥에 묻힌 채 시작해 처음 다섯
틱 동안 튀어 올랐다. 쿼터니언은 모두 0이고 백엔드는 이것을 항등으로 읽으므로, 장면이 준 회전이
사라진다. "과제 정하기"(시작 항목 없음)에서는 모든 자유 바디가 원점, 곧 로봇 받침 안에서 시작했다.
오라클은 `crates/es-script/tests/estask.rs`(`scene_pose`).

- **규칙.** 로봇 하위 트리 밖 바디의 자유 관절마다, 시작 항목(`<body>.x|y|z`, `<body>.orientation`,
  관절 이름)이 쓰지 않는 `qpos` 일곱 칸 각각에 `compile_task`가 상수 `ResetState`를 하나씩 낸다.
  값은 장면 자신의 값 `qpos0`이다: 바디의 위치 `x y z`, 이어서 방향 `w x y z`(MuJoCo의 순서)를
  `SceneDesc`에서 비트 그대로, 계산 없이 읽는다. 대상은 `qpos[<칸>]`, 스트림은
  `scene.<body>.pos.<i>`나 `scene.<body>.quat.<i>`. 노드는 시작 항목의 노드 뒤에, 장면의 관절
  순서와 칸 순서대로 붙는다. 상수는 아무것도 뽑지 않으므로 다른 뽑기는 움직이지 않는다. GV의
  장면에서 상자는 z = 0.025와 `(1, 0, 0, 0)`을 받고, 장면에서 돌려 둔 상자는 그 회전을 지키며,
  `[start]`가 아예 없으면 모든 자유 바디가 장면의 자세에서 시작한다(`mujoco-cpu`에서 측정).
- **끄는 법.** `[start] zero_unset = true`는 예전 동작을 지킨다: 그런 노드가 없고, 정하지 않은
  칸은 0이다. 없거나 `false`이면 규칙을 따른다. 커밋된 두 명세가 이것을 켠다(그 이유는 주석 한
  줄: 규칙 이전에 커밋된 문서를 재현한다). 그래서 둘의 `task_hash`와 G3b가 다시 쓰는 문서는 그대로다.
  Shadow Hand 템플릿의 편집용 복사본도 이것을 그대로 복사한다. 문장(G8)이 아니다: 문장은 읽은
  그대로 두고, 이것만 든 `[start]` 표는 지우지 않는다.
- **소유자에게 열린 것.** 끈 상태에서 커밋된 Shadow Hand 과제의 목표(`target`, 중력 보상이 있는
  자유 바디, 방향만 뽑는다)는 매 에피소드 장면이 두는 자리가 아니라 원점 (0, 0, 0)에서 시작한다.
  `shadow_hand_repose.estask`에서 `zero_unset`을 빼면 고쳐지지만, 커밋된 `task_hash`가 움직이고
  그와 함께 학습된 교사의 문서도 움직인다. so101_views의 큐브도 쿼터니언(항등, 백엔드가 이미 그렇게
  읽던 값)을 얻게 된다. 커밋된 대로 둔다.

### 4.7.1 R5가 정한 세부 (영역 쪽으로 이끄는 성형, `qpos0`의 관절)

M17 리뷰의 N-2와 F-9. 오라클은 `crates/es-script/tests/estask.rs`(`region_distance`),
`estask_scripted.rs`(`inside_a_region_pays_the_distance_to_its_centre`),
`crates/es-editor-scene/tests/sentences.rs`(`the_region_distance_toggle_is_one_undo_step`).

- **영역 `inside`는 `shaping = "distance"`를 받는다**: `weight × ‖p − c‖`. `p`는 바디의 세계 위치,
  `c`는 영역의 중심(4.4절의 중심)이고, 거리는 `near`처럼 [0, 1] m로 자른다. 점에 대한 `near`와 같은
  구성이다: `GetBodyPose.pos`, 레인별 `Normalize` `[c − 1, c + 1] → [−1, 1]`, `Norm L2`,
  `Normalize [0, 1] → [0, 1]`, `Reward`. 레인마다 1 m에서 잘리지만 결과는 같다: 한 레인이 1 m를
  넘으면 거리도 1 m를 넘고, 항은 어차피 거기서 잘린다. 항 이름은 `term`, 없으면 `<subject>_distance`.
  켜야 생긴다: `shaping`이 없으면 절은 전과 같이 컴파일된다(G3c의 테스트는 그대로다. GV의
  프로젝트에서 성형 없는 영역 절은 R5 이전과 같은 `task_hash`, 같은 문서 바이트를 낸다).
- **문장.** 영역 절은 다른 형태처럼 성형 토글과 단계를 보인다(거리는 비용: 보통은 `weight = −1`).
  "영역 안"을 새로 고르면 — 새 절이든, 관계나 주어를 바꿔 거기에 이르든 — 보통 단계로 켜진 채
  시작한다. 단, 지킬 거리 항이 이미 있으면 그것을 지킨다(G8의 규칙: 형태가 같으면 성형을 그대로
  둔다. 그래서 많이로 성형된 `near`가 `inside`가 되면 많이 그대로다). 절이 이미 가진 관계를 다시
  고르면 아무것도 바뀌지 않으므로, 꺼 둔 토글은 꺼진 채다. "과제 정하기"의 절은 여전히 첫 자유
  바디의 멈춤이다. 그 절이 영역에 이를 때만, 곧 자유 바디가 없는 장면에서만 이 항을 얻는다.
- **F-9: `qpos0`에서 벗어나 시작하는 관절은 없다.** `SceneDesc`에는 `ref`가 없다: MJCF 파서는 이
  속성이 표현되지 않는다고 경고하고 버린다. `.esscene`에도 그런 필드가 없고, 백엔드의 MuJoCo
  모델은 `SceneDesc`에서 쓴다(`scene_to_mjcf`). 그래서 파이프라인이 돌리는 모델에서 모든 경첩과
  미끄럼 관절의 `qpos0`은 0, 곧 파일이 그린 자세이고, `Env::reset`의 0이 바로 그 `qpos0`이다.
  커밋된 장면에도 `ref`는 없다(볼 관절도 없다. 떠 있는 받침의 로봇은 `go1_primitives.xml` 하나인데
  어느 `.estask`에도 쓰이지 않는다. M6의 `task.toml`은 컴파일되지 않는다). 커밋된 장면, GV의
  장면, 라이브러리의 장면 어디에도 나오는 노드가 없으므로 코드를 더하지 않았다. 이것을 바꾸려면 `SceneDesc`에 `ref`가 들어와야 한다. `es-assets`
  패킷이다(파서, `.esscene` 필드, 두 MJCF 작성기, 외관 블록처럼 0이 아닐 때만 덧붙이는
  `scene_hash`). 그다음 `scene_poses`가 정하지 않은 경첩과 미끄럼 관절 가운데 `ref`가 0이 아닌
  것마다 상수 `ResetState`를 내고(스트림 `scene.<joint>`), `zero_unset = true`에서는 내지 않는다.
  `ref`를 읽게 되는 날 `a_hinges_ref_is_not_represented_and_nothing_is_emitted`가 실패한다.

### 4.8 시도가 실패한 이유: 빠진 절 (패킷 M17/R8의 설계)

소유자가 승인한 S4 방향이다. ⑤는 실패한 시도를, 그 끝에서 성립하지 않은 성공 절로 ①의 말을 써서
설명한다. 예: "상자가 목표 영역 안에 없었다 (실패한 시도 4번 중 3번)". M17의 GV에서 그 필요가 드러났다.
⑤는 네 번의 실패 모두에 "시간 안에 끝내지 못함"이라고만 했다.

**결정: 절은 시도가 기록한 끝 상태 위에서, 환경이 쓰는 것과 같은 lowering으로 사후에 평가한다. IR과
해시는 바뀌지 않는다.**

- **끝 상태**는 궤적의 마지막 행이다(패킷 R2: `6ba08d6`부터 `.estraj`는 에피소드가 끝난 스텝 직후의
  상태로 끝난다). 그 행이 없는 예전 궤적은 마지막 행으로 설명하고, ⑤가 그렇다고 말한다.
- **어느 노드가 절인가.** `compile_task`는 절들을 `Terminate(Success)` 하나(`And`)와
  `Terminate(Failure)` 하나(`Or`)로 접는다. 절 자체가 `And` 트리일 수 있어서(영역 안에 있다) Task IR만
  보고는 이 접기를 되돌릴 수 없다. 그래서 컴파일러는 곁표 `[(section, index, node)]`도 돌려준다. 각 절의
  술어가 끝나는 노드다. IR의 일부가 아니며 어떤 해시에도 들어가지 않는다.
- **노드 평가.** `es-env`에 함수 하나가 생긴다. 노드 하나의 원뿔을 종료 원뿔과 똑같이 lowering하고
  (`plan.rs`), 궤적 한 행으로 만든 `StateView`(`qpos`, `qvel`, 바디 자세) 위에서 평가한다.
- **레이아웃.** 그러려면 모델의 레이아웃이 필요하다. 관절마다 어느 `qpos`, `qvel` 칸을 쓰고 바디가 몇
  번째 행에 있는지다. 패킷이 아래 둘 중 하나를 고르고 이유를 기록한다:
  1. 백엔드의 레이아웃을 재현하는 장면의 순수 함수. 커밋된 모든 장면에서 `mujoco-cpu`의 `ModelInfo`와
     같다는 오라클과 함께;
  2. 실행할 때 평가의 궤적 옆에 써 두는 레이아웃. 고정된 파일이나 해시를 하나도 움직이지 않아야
     한다(`demo_trajectories_are_unmoved`가 `traj/` 폴더의 파일들을 해시한다).
- **말의 출처.** `es-editor-scene`이 명세와 G8의 문장을 가지고 있다. 실패한 시도마다 절을 평가해서 ⑤에
  목록을 넘긴다:
  - 성공 절마다, 그 절 없이 끝난 실패 시도의 수;
  - 실패 절마다, 그 절로 끝난 시도의 수.

  ⑤는 직접 만든 프로젝트에서 결과 분류 자리에 이 목록을 보인다. 템플릿 프로젝트는 자기 `[outcome]`
  분류를 그대로 쓴다. `es-editor-model`(계층 12)은 `es-editor-scene`을 의존할 수 없으므로, G9가 생성
  상태를 넘기듯 `es-editor`가 목록을 넘긴다.
- **오라클.**
  - 스크립트된 끝 상태에서 각 절의 참거짓이 환경 자신의 술어와 같다.
  - GV의 평가(시도 16번, 시간 초과 4번)에서 각 시간 초과의 설명이 끝 행에서 거짓인 성공 절을 가리킨다.
    성공한 열두 번에서는 네 절이 모두 참이다.
  - 설명이 보이는 ⑤의 스크린샷.

**R8이 정한 세부** (리뷰 F-11. `crates/es-script/src/spec/compile.rs`의 `compile_clauses`,
`crates/es-env/src/plan.rs`의 `eval_nodes` / `eval_on_row`, `crates/es-physics-backend/src/mjcf_out.rs`의
`layout`, `crates/es-editor-scene/src/missing.rs`의 `explain`. 그리는 것은 `crates/es-editor/src/ui/results.rs`):

- **곁표.** `compile_clauses(spec, root)`는 Task IR과 `[(TerminationKind, index, node)]`를 돌려준다.
  모든 절이 문서 순서대로 들어 있다. `compile_task`는 같은 호출에서 곁표만 뺀 것이라 컴파일된 바이트는
  하나도 움직이지 않았다(G3a, G3b, GV, R5의 오라클이 그대로 통과한다). `eval_nodes`는 노드마다 `value`
  출력을 종료 원뿔과 같은 lowering으로 낮추고, 포트는 환경과 같은 방식으로 읽는다(`Source::read`.
  `Env::bind_ports`도 이제 이것을 부른다). `eval_on_row`는 이 일을 `.estraj` 한 행에서 한다. 센서를
  읽는 원뿔(궤적에는 센서가 기록되지 않는다)과 궤적에 없는 바디는 거절한다. 오라클은
  `estask_scripted.rs`의 `each_clauses_truth_is_the_envs_own_predicate`: 영역 안, 멈춤, 점 근처, 점보다
  멂을 스크립트한 상태 일곱 개에서 본다. 각 절의 참거짓은 그 절 하나만 둔 과제의 종료와 같고, 네 절을
  모두 둔 과제의 종료는 그것들을 접은 것과 같다.
- **레이아웃은 장면의 순수 함수다(1안)**: `layout(scene)`. MuJoCo가 번호를 매기는 순서를 정하는 작성기
  바로 옆에 둔다. 바디는 `scene_to_mjcf`가 쓰는 순서대로(세계가 0, 그다음 깊이 우선, 자식은 장면 순서),
  바디마다 관절은 장면 순서대로(고정 관절은 관절이 아니다) 두고 주소를 더해 간다. 2안을 고르지 않은
  이유: GV의 실행 001과 R8 이전의 모든 실행에는 레이아웃 파일이 없어 설명되지 못한 채 남고, 실행할 때
  쓰는 파일은 고정된 `traj/` 폴더 바로 옆에 놓인다. 순수 함수는 아무것도 쓰지 않고, 순서를 그 순서를
  만드는 코드 옆에 둔다. 오라클은 `crates/es-physics-backend/tests/layout.rs`: `mujoco-cpu`에서 열리는
  커밋된 장면 17개와 GV의 장면에서 `nq`, `nv`, `nbody`, `qpos`, `dof`, `body`가 `mujoco-cpu`의 것과
  같다(`actuated.xml`과 `urdf/arm2.urdf`는 `mujoco-cpu`에서 열리지 않는다. 읽기 거절용 픽스처와
  `so101_reach_mjx.xml`은 읽히지 않는다). 바디의 행은 레이아웃에서, 자세는 궤적에서 바디 id로 가져온다.
- **어느 실행인가.** 실행의 보고서 해시를 가진 생성된 평가 문서가, 저장된 명세가 컴파일되는 Task IR을
  가리킬 때만 그 실행을 설명한다. 과제가 바뀌기 전의 실행을 그 실행에 없던 절로 설명하지 않는다. ⑤는
  `generated/`가 최신인(`Generated::Fresh`) 직접 만든 프로젝트에서만 이것을 묻는다.
- **끝 행**은 궤적의 마지막 행이다. 프레임 수보다 하나 많으면(프레임을 쓰지 않았으면 스텝 수보다 하나
  많으면) 끝 상태이고, 아니면 R2 이전의 실행이다. 이때 ⑤는 끝나기 한 단계 전, 마지막으로 기록된
  상태로 읽었다고 말한다. 성공 절은 그 절 없이 끝난 실패 시도를, 실패 절은 그 절이 성립한 채 끝난 시도를
  설명한다. 성공이 먼저 접히므로(노드 id 오름차순) 시간 초과의 끝 행에서는 어떤 성공 절이 거짓이다.
  예전의 마지막 행에서도 그렇다. 그 행은 바로 앞 스텝 뒤의 상태이고, 환경은 그 상태에서 계속하기로
  정했기 때문이다.
- **말.** 한 줄은 그 절의 G8 문장을 틀에 넣은 것이다. 성공 절은 "끝났을 때 아니었음: {문장}", 실패 절은
  "이것으로 끝남: {문장}", 이어서 "n번 (실패한 m번 중)". 설명하는 시도가 많은 절이 먼저 오고, 아무것도
  설명하지 않는 절은 빠진다. 타일의 말은 그 시도의 첫 절, "아님: {문장}"이다. 이 줄들은 ⑤의 실패 이유
  목록에서 "시간 안에 끝내지 못함"과 실패 조건 자리에 선다. 다른 원인은 그대로 남는다. 부정문으로 쓰지
  않은 이유: 모든 문장 틀과 관계 낱말의 부정형을 두 언어로 두면 표가 두 배가 되고 어색하게 읽힌다("보다
  느리지 않았다"). 틀은 ①의 말을 사람이 쓴 그대로 지킨다. 소유자가 바꿀 수 있다.
- **GV에서 잰 것** (`crates/es-editor-scene/tests/missing.rs`). GV가 쓴 그대로의 실행 001(성공 12번,
  시간 초과 4번, 끝 행 없음): 네 번의 시간 초과 모두 "box가 target 안에 있다" 없이 끝났다. nominal-03,
  -12, -15는 "box가 (22, 12, 2.5) cm에서 5 cm 안에 있다"도 없이 끝났고, nominal-10은 영역 중심에서
  5 cm 안이지만 상자 밖에서 끝났다. 네 번 모두 상자는 멈춰 있었고 그리퍼는 상자 가까이 있었다. 성공한
  열두 번의 마지막 기록 행에서는 "box가 멈춰 있다"가 거짓이다. 성공하기 한 단계 전의 상태이기 때문이고,
  끝 행은 바로 이 경우를 위해 있다. 학생 체크포인트 20000을 복사본에 다시 평가하자(`es eval run`, 16번,
  종료와 스텝 수가 같다) 끝 행에서 같은 설명이 나왔고, 모든 성공의 끝 행에서 성공 절이 모두 참이었다.
- **가는 길에 고친 것 (R7이 찾은 것).** ⑤의 타일이 줄을 바꾼다: 줄바꿈하는 줄 안의 `ui.vertical`은 줄을
  바꾸지 않고 놓이므로 16개 중 10개만 보이고 나머지는 오른쪽에서 잘렸다. 이제 타일마다 자기 너비로 자리를
  잡고, 위로 맞춘 줄바꿈 줄에 놓인다. ②의 Test 도움말은 "64"가 아니라 교사 평가 자신의 시도 수, 곧
  에피소드 수 × 묶음 수를 말한다(`teacher::attempts`: GV는 16, 손은 64).

## 5. 에디터 (①과 ②)

- **계층 구조 패널**: 장면 트리(인클루드는 접힘), 검색, 가시성. 끌어서 부모를 바꾼다.
- **인스펙터**: 선택한 것의 필드와 🎲 토글. 단위는 말로. 잘못된 값은 그 자리에서 이유와 함께
  거부한다(`SceneDesc::validate`, 매핑 리포트).
- **뷰포트**: 피킹(분할 채널이 이미 geom id를 준다), 스냅이 있는 이동 / 회전 / 크기 기즈모,
  선택 물체로 맞추기, 프로세스 안에서 매 프레임 그리는 H8의 렌더 모드(H9), 그리고 모서리의
  정책 카메라 화면을 같은 프로세스 내 렌더러가 그 카메라에 선언된 해상도와 렌더 경로로
  렌더한다(실제 관찰. 장치가 없으면 `es render`).
- **명령과 실행 취소**: 모든 편집은 `es-editor-scene`(계층 12, 5.1)의 장면 모델에 대한 명령이며,
  문서에 적용되고, 검증되고, 되돌릴 수 있다. 레이아웃(`*.eslayout`, §14.3)은 장면 문서에
  들어가지 않는다.
- **물리 미리보기**: `mujoco-cpu`에서 `es scene simulate --seconds 3 --out <traj>`를 돌려
  뷰포트에서 재생한다(에디터는 물리를 돌리지 않는다).
- **검사**는 쉬운 말로, `es`나 순수 함수가 계산한다: 매핑 리포트("이 백엔드는 텐던을 시뮬레이션할
  수 없다"), 카메라가 시작 위치에서 물체를 보지 못함(카메라마다 분할 렌더 한 번), 미리보기에서
  바닥을 뚫고 떨어지는 물체, 시작 자세에서 한계를 넘는 로봇 관절.
- **저장**은 `scene.esscene`을 쓴다. 생성은 요청이 있을 때(그리고 ③ 앞에서) 돌아가므로,
  Advanced 탭을 열지 않는 한 사람은 IR 문서를 보지 않는다.

### 5.1 G5가 정한 세부 (장면 모델)

`crates/es-editor-scene/`(새 계층 12 크레이트: `SceneModel`, `Command`, `check`, `tree`,
`inspect`, `euler`, `make_editable`), 그리기는 `crates/es-editor/src/ui/author.rs`, 오라클은
`crates/es-editor-scene/tests/scene_model.rs`.

- **크레이트**: `es-editor-model`이 10,000줄 상한 중 8,942줄이었고 장면 모델은 약 1,400줄이라
  §4.2의 새 계층 12 크레이트 `es-editor-scene`으로 갔다(스펙 §4.2 표와 규칙 4, 부록 C.8, xtask의
  `LAYERS`). 둘은 서로를 모르고(규칙 1) `es-editor`만 둘 다 쓴다. 거부 이유의 말은
  `es-editor-model` 문자열 표의 키이고, 표 완전성 테스트가 이 크레이트의 소스도 읽는다.
- **편집 가능한 프로젝트**: 프로젝트 루트에 `scene.esscene`(과 `task.estask`)이 있는 것. 생성 문서는
  `generated/`에 쓰고(`generate`의 `out`이 `"generated"`), 다시 만들기 전에 비우므로 실패하면
  낡은 문서가 남지 않는다. 템플릿은 편집 원본을 `[editable] scene / spec`으로 이름한다. "편집 가능한
  복사본 만들기"는 템플릿의 장면 문서를 바이트 그대로, 그 문서가 이름하는 파일(인클루드, 메시,
  텍스처)을 **같은 상대 경로로** 복사한다. 자산 경로는 해시 입력이라 `assets/<blake3>`로 옮기면
  아무것도 고치지 않은 장면의 `scene_hash`가 움직인다(§3.1의 그 규칙은 G7의 가져오기에 남긴다).
  명세는 `scene` 줄만 `scene.esscene`으로 바꿔(주석은 그대로) 복사한다. 측정(오라클 5): 섀도 핸드
  복사본은 템플릿의 `scene_hash`로 펼쳐지고, 생성 문서는 커밋된 문서와 같다. 다른 것은 장면
  *파일*을 가리키는 것뿐이다: Task IR의 `scene.path`와 `asset_hash`, Deployment IR의 시뮬레이션
  로봇 대상 경로, 그리고 해시 체인을 따라 Observation IR의 `task_ref`와 Evaluation IR의
  `task`·`observation`. Learning IR은 그대로다. SO-101 큐브 템플릿 둘은 장면만 복사한다(그
  `task.toml`을 말하는 명세가 없다).
- **명령**: `Add`(물체와 그 모양, 고정 물체, 카메라, 조명, 영역, 인클루드는 G7), `Delete`(물체는 그
  아래 물체와 거기 붙은 카메라·영역과 함께), `Duplicate`(원본 옆, `<이름>_<n>`, 인클루드 복사본은
  접두사를 받는다), `Set`(필드. 이름은 아니다), `SetPose`(G6의 기즈모용), `Reparent`(부모가 앞에
  오도록 물체 순서를 고치고 순환은 거부), `Rename`(문서의 부모 참조와 `task.estask`의 robot, 절의
  subject·object, start의 `what`, observe의 카메라와 출처, 학생의 views를 고친다. `robot.`은 명세의
  낱말이라 그대로). 정체는 이름, 모양(geom)은 목록 안의 자리.
- **검증 정책: 거부된 명령은 적용하지 않는다.** 검사 순서는 값(음수 질량·밀도·마찰, 0 크기, [0, 1]
  밖의 색, 시야각, 한계 순서, 빈 이름) → G1의 `expand`(같은 이름, 없는 부모, 없는 재질, 없는 파일)
  → `mujoco-cpu`의 매핑 리포트(템플릿이 MJWarp에서 배우면 그것도). 거부는 G1이 부르는 필드 경로 +
  표의 키 + 인자. 적용하고 오류를 다는 방식을 버린 이유: 그러면 문서, 뷰포트, 실행 취소의 모든
  단계, 물리 미리보기가 언제나 펼쳐지는 장면이다. 입력한 값은 인스펙터 칸에 남고 그 칸 이름이
  빨갛게, 이유가 위에 나온다.
- **실행 취소**: 단계마다 문서 전체(장면 + 명세)의 스냅숏. 바뀐 것이 없는 명령은 단계가 아니다.
  dirty는 문서 ≠ 저장본, 저장은 둘을 원자적으로(임시 파일 + 이름 바꾸기) 쓰고 다시 생성한다.
- **보이기/숨기기**는 에디터의 것, 메모리에만 있다(아직 `.eslayout`도 쓰지 않는다).
- **인스펙터**: 각도는 roll/pitch/yaw 도(고정 X, Y, Z 축), `es_math::approx`의 `acos_f64` ·
  `sin_cos_f64`로 변환하고, 각도를 바꿀 때만 쿼터니언을 쓴다. 크기는 전체 폭·지름으로 보이고 절반으로
  저장한다(이진수로 정확). 값은 사람이 손을 뗄 때(포인터를 놓고 글자를 입력하지 않을 때) 명령 하나로
  넘어가므로 끌기 한 번이 실행 취소 한 단계다.
- **뷰포트**: `ScenePreview::from_scene`, 정지 장면 샷의 틱 자리는 문서 리비전이라 편집마다 새로
  그린다. 저장하지 않은 문서는 `es render`·`es scene simulate`를 위해 장면 옆 `.preview.esscene`에
  쓴다. 에디터의 재생용 로더도 `.esscene`을 읽는다.
- **G9까지**: ②–⑤는 여전히 템플릿의 문서로 돈다. ①이 그렇게 말한다.

### 5.2 G6가 정한 세부 (피킹, 손잡이, 모서리 화면)

판단은 `crates/es-editor-scene/src/{view,gizmo,policy}.rs`, 그리기는 `crates/es-editor/src/ui/author.rs`
(오버레이), `ui/corner.rs`, `gpu.rs`(`Sensor`), 오라클은 `crates/es-editor-scene/tests/viewport.rs`.

- **피킹은 누른 지점의 분할 채널 값을 CPU에서 읽는 것이다.** 렌더러가 그 지점으로 쏘는 광선
  (`es_render::cpu::primary_dir`, 여기서는 `f64`)을 그려지는 장면의 삼각형에 렌더러 자신의 최근접
  규칙(`nearest_hit_flat`)으로 맞히고, 맞은 삼각형의 분할 id가 곧 geom이다. 장치에서 읽어 오지도
  않고 `es-render`도 고치지 않는다. 뷰포트는 `Rgb8`만 그리고, 클릭 한 번은 프레임마다 채널 하나를
  더 그리는 것이 아니라 광선 하나이며, 답을 정해 둔 광선으로 시험할 수 있다. 물체의 geom을 누르면
  그 물체, 인클루드가 들여온 것을 누르면 그 인클루드(바꿀 수 있는 것은 자기 자세뿐이고 부품은
  읽기 전용), 문서의 세계 geom은 고정 물체, 조명 판은 그 조명을 고른다. 카메라와 영역은 아무것도
  그리지 않으므로 계층 구조에서 고른다. 숨긴 것은 맞지 않아 광선이 그 뒤로 간다. 빈 곳을 누르면
  아무것도 고르지 않는다.
- **손잡이.** 이동과 회전은 문서가 위치를 적는 좌표계(부모의 것, 대부분은 세계)에서 그것 자신의
  원점을 지나므로, 이동은 `pos`의 좌표 하나만 바꾸고 회전은 그 좌표계에서의 `dq · q`다. 크기는
  모양 자신의 좌표계에서 다룬다: geom, 고정 물체, 영역, 조명, 그리고 geom이 하나뿐인 물체(그 geom).
  카메라, 인클루드, geom이 여럿인 물체에는 크기 손잡이가 없다. 메시의 손잡이는 배율을 바꾸며(패킷
  M17/R3, 5.3절), 세 축에 같은 비율을 곱한다. 손잡이 길이는 눈에서의 거리의 15 %라 화면에서
  크기가 변하지 않는다. W, E, R로 도구를, F로 가운데 맞추기를 한다.
- **맞춤**(기본 켬, 체크박스 하나)은 문서의 단위로 반올림한다: 옮긴 좌표는 1 cm 단위로
  (`(v · 100).round() / 100`, 그래서 문서에 `1.05`라고 적힌다), 끌어서 돌린 각은 15°의 배수로,
  크기의 전체 길이(인스펙터의 폭·높이·지름)는 1 cm 단위로, 최소 1 cm. 회전각은 `es_math::approx`의
  `acos_f64`(인스펙터의 `atan2`)로 재고 `sin_cos_f64`로 적으므로, 끌기가 적는 값은 호스트의
  `libm`에 따라 달라지지 않는다.
- **끌기 한 번이 명령 하나다.** 손잡이를 잡고 있는 동안 문서는 바뀌지 않는다. 뷰포트가 선택의
  색칠을 끌기가 옮길 자리에, 손잡이도 그 자리에 그리고, 얼마나 갔는지(`+5.0 cm`, `+30°`,
  `6.0 → 9.0 cm`)를 보인다. 손을 떼면 `SetPose`(이동, 회전) 하나나 `Set`(크기) 하나를 적용하며,
  이것이 실행 취소 한 단계다. 맞춤 때문에 제자리로 돌아온 끌기는 명령을 만들지 않는다.
- **선택**은 선택한 것이 그리는 삼각형 위에 반투명 색을 그림 위에 덧칠한 것이다(깊이 시험 없는
  투시). **가운데 맞추기**는 카메라 방향을 그대로 두고, 선택한 것이 그리는 것을 감싸는 구(아무것도
  그리지 않으면 그 자리의 10 cm 구)를 그림 가운데에 높이 대부분을 채우게 둔다.
- **모서리**는 정책이 받는 카메라 하나를 보인다. 작업 명세가 있는 편집 가능한 프로젝트: 학생의
  `views`, 없으면 `[observe] cameras`를, 편집 중인 장면(저장하지 않았으면 `.preview.esscene`)에
  `compile_task`로 컴파일해 선언된 대로. 템플릿과 명세 없는 편집 복사본: 묶음의 Observation IR이
  읽는 카메라(`ImageInput` 센서, 노드 순서)를 묶음의 Task IR이 선언한 대로. 정책이 보는 카메라를
  골랐으면 그것, 아니면 첫 번째. 측정: 섀도 핸드 복사본의 모서리 카메라 셋은 `task-repose.toml`의
  채널과 똑같이 선언된다(96 × 96, `Pt` 32 spp, 반사 3번, 노출 8, 틱마다 시드).
- **모서리의 프레임**은 뷰포트의 작업 스레드(H9)가 프로세스 안에서 `es-env` 자신의 단일 카메라
  단계로 그린다 — `sensor_cfg`, 아무것도 뽑지 않은 `drawn_frame`, 틱 0의 `Tick` 시드, 렌더 한 번,
  읽기 한 번 — 그래서 장면 자신의 자세(초기화 추첨이 아닌)에서의 관찰이며, 선언된 크기와 경로로,
  장면 리비전과 카메라마다 한 번 그리고, 최근접 필터로 보인다. 이를 위해 `es-editor`가 `es-env`를
  `render` 기능과 함께 쓴다. 장치가 없으면 모서리가 그렇게 말한다. `es render`는 자유 카메라만
  그리므로 여기서 대신할 수 없다.

### 5.3 G7이 정한 세부 (추가, 가져오기, 표시선, 덮어쓰기)

판단은 `crates/es-editor-scene/src/{add,import,marker,overrides}.rs`, 그리기는
`crates/es-editor/src/ui/author/{add,markers,overrides}.rs`, 오라클은
`crates/es-editor-scene/tests/add.rs`. 로봇 목록은 `templates/robots.toml`, 빈 장면은
`tests/fixtures/esscene/empty.esscene`이다.

- **메뉴**: 물체(상자, 구, 원기둥, 캡슐: 5 cm geom 하나를 단 자유 물체), 고정된 물체(같은 모양을
  세계에 고정), 메시 파일(STL, OBJ: 메시 geom 하나를 단 자유 물체), 로봇(목록의 것, 또는 MJCF,
  URDF, glTF 파일: `[[include]]`), 카메라, 조명, 영역. 추가 한 번은 명령 하나(G5의 `Add`)이고
  실행 취소 한 단계이며, 만든 것이 골라진다.
- **놓이는 곳.** 화면 가운데 광선을 그려지는 삼각형에 렌더러의 최근접 규칙으로 맞힌다
  (`SceneModel::surface`. `hit`의 형제로, 삼각형의 법선도 준다). 맞은 점은 `f64`로 다시 구하고,
  수평면에서는 높이를 삼각형 자신의 값으로 두므로 탁자의 0은 0 그대로다. 새것의 가장 낮은 점이
  그 점에 닿는다: 기본 모양은 법선 방향의 반폭(돌리지 않은 채 추가한다), 메시는 가장 낮은 꼭짓점,
  영역은 그 상자. 맞은 것이 없으면 원점, 바닥면 위에 놓는다. 맞춤(G6의 체크박스)이 켜져 있으면
  수평면 위의 점을 면을 따라 1 cm 단위로 반올림하고, 높이는 그대로 둔다.
  - 카메라는 화면의 눈 위치에서 화면이 보는 곳을 본다. 렌더러의 look-at 쿼터니언을 X축으로 반
    바퀴 돌려 MJCF 카메라 좌표계로 옮기므로 삼각함수를 쓰지 않는다. 맞춤이 켜져 있으면 눈과 보는
    점을 먼저 1 cm 단위로 맞춘다. `fovy`는 화면의 값을 1e-6° 단위로.
  - 조명은 그 점 1 m 위에 단다(G5의 추가처럼 천장 판).
  - 로봇은 그 점에 목록의 오프셋을 더한 곳에, 목록의 방향으로(없으면 파일 자신의 방향) 선다.
- **내용으로 가져오기.**
  - 메시나 그림은 `assets/<blake3>.<ext>`로 간다(리더가 확장자로 고르므로 확장자는 소문자).
  - 로봇 파일은 `assets/<파일의 blake3>/<자기 이름>`으로 가고, 그 파일이 이름 대는 파일들이 상대
    경로 그대로 함께 간다: 그 리더가 읽는 것(메시와 텍스처 경로, 텍스처 파일과 큐브 파일,
    `.gltf`의 버퍼와 이미지 URI). 파일을 자기 폴더에서 홀로 펼쳐서 찾으므로, 깨진 파일은 아무것도
    쓰기 전에 거부된다. 자기 폴더 밖(`../`, 절대 경로)을 이름 대는 파일은 그 이름과 함께 거부된다.
  - 같은 바이트가 이미 있으면 다시 쓰지 않고, 그 경로에 다른 파일이 있으면 거부한다.
  - 모든 명령이 거치는 검사는 디스크에서 펼치므로 파일을 먼저 복사하고, 거부된 추가는 자기가 쓴
    것(만든 파일과 폴더)만 정확히 지운다. 실행 취소는 가져온 파일을 남긴다(다시 실행에 필요하며,
    내용으로 이름 붙은 캐시다). G5의 편집 가능한 복사본은 상대 경로를 그대로 둔다.
- **이름**: 새 물체, 카메라, 조명, 영역은 G5의 이름 공간에서 `unique(stem)`이고, 고정 물체는
  세계의 geom 사이에서 겹치지 않으며, 인클루드의 핸들은 목록의 id 또는 파일 이름이다. 인클루드
  자신의 이름이 장면의 이름과 겹치면(두 번째 SO-101) 복제한 인클루드가 받는 접두사 `<핸들>:`을
  붙여 한 번 더 시도한다.
- **그림 가져오기**는 재질 고르기의 마지막 항목이다. 명령 하나 `Command::Scene`(문서 전체를
  바꾸는, 하나뿐인 새 명령)이다: 그림 파일 이름으로 이름 붙인 `[[texture]]`(2D, 그 파일)와
  `[[material]]`(그 텍스처) — 같은 그림이 전에 만든 짝이 있으면 그것 — 그리고 geom의 `material`.
  텍스처 로더가 디코드하는 PNG만 받고, 다른 것은 거부하며 복사한 것을 되돌린다.
- **표시선.** 카메라와 영역을 선으로 그린다: 카메라의 절두체(눈에서의 거리의 12 % 깊이라
  화면에서 크기가 같다. `fovy`의 정사각형 그림, 윗변의 눈금)와 영역의 상자이며, 고른 것은 밝게
  그린다. 그 선에서 8포인트 안을 누르면 광선이 그려진 geom을 찾기 전에 그 카메라나 영역을 고른다.
  그다음 G6의 손잡이가 붙는다(영역은 크기도).
- **덮어쓰기.** 인클루드의 인스펙터는 그 파일이 선언한 것을 파일의 이름으로 늘어놓는다:
  `SceneModel::brought`가 덮어쓰기와 접두사 없이 파일만 펼친다 — 자유 관절을 뺀 모든 관절,
  모든 액추에이터(`kp`, `kv`는 그 이득이 있을 때만), 모든 geom, 파일의 재질. 값은 덮어쓴 값,
  없으면 파일의 값이고 ↺로 파일의 값으로 돌아간다. 편집은 `set`을 정리한(`overrides::prune`)
  인클루드의 G5 `Set`이다: 비운 필드는 덮어쓰기를 버리고, 빈 대상은 표를 버리며, 빈 `set`은
  없는 것이 된다. 경첩과 볼 관절의 범위는 도로 보인다.
- **목록**(`kind = "robots"`, `[[robot]]`에 `id`, `name` — i18n 키 —, 저장소 루트 기준
  `source`, 그리고 선택적인 `pos`, `quat`, `prefix`). 템플릿 로더는 이 파일을 건너뛴다. SO-101은
  밑판이 그 점에 서는 `so101.xml`, 섀도 핸드는 (-1, -1.25, 0)만큼 옮긴 `shadow_hand.xml`이라
  거치대가 그 점 15 cm 위에 걸린다.
- **`empty.esscene`**: 끝없는 바닥(`plane = [0, 0, 0.05]`), 1.5 m 높이의 60 cm 천장 조명, 그리고
  1.2 m 뒤 1.2 m 위에서 45°로 원점을 내려다보는 카메라 `outside`.
- **측정**: 섀도 핸드 복사본과 `empty.esscene` 양쪽에서 모든 항목이 추가되고 펼쳐지며
  `mujoco-cpu`에 매핑된다. 가져온 섀도 핸드 메시 열두 개는 원본과 같은 해시를 갖는다.
- **메시 단위** (패킷 M17/R3, 스키마는 3.4절). 메시 파일을 고르면 작은 창이 열린다: 미터,
  센티미터, 밀리미터(메시에 적힐 배율 1, 0.01, 0.001이며 처음엔 미터), 고른 단위로 본 파일의
  크기, 그리고 한 변이 5 m보다 길면 "방 하나보다 큽니다. 밀리미터로 그린 파일인 것 같으니
  밀리미터를 고르세요." 추가를 누르면 그 배율로, 배율을 곱한 가장 낮은 꼭짓점이 맞은 점에 닿게
  들어간다. 인스펙터는 메시의 배율을 균일한 동안 숫자 하나로(아니면 셋으로) 보이고, 1은 배율
  없음으로 적는다. 크기 손잡이(5.2절)는 세 축에 같은 비율을 곱하고, 끈 길이는 G6의 크기처럼 1 cm
  단위로 맞추며, 끌기 한 번이 `Set` 하나다.
- **남은 것**: 메시가 자기 폴더 밖에 있는 URDF는 거부되고, USD는 G1과 같이 거부된다.

### 5.4 G9가 정한 세부 (직접 만든 프로젝트가 끝까지 돈다)

`templates/empty.toml`, `crates/es-editor-model/src/model/template.rs`(`source`, `authored`,
`Generated`, `load_saved`, `write_saved`), `project.rs`와 `watch.rs`(`with_source`, `set_source`,
단계 막기), `crates/es-editor-scene/src/{model,copy,check,sentence}.rs`(`on_disk`, `refresh`,
`Regen::Stale`, `copy::documents`, `check::maps_onto`, "과제 정하기"가 쓰는 학습자)가 정하고,
`crates/es-editor/src/ui/{scene,author,teacher,teach,results,train,home}.rs`가 그리고 잇는다.
오라클은 `crates/es-editor/tests/authored.rs`(패킷의 1~4)와
`crates/es-editor-scene/tests/{sentences,documents}.rs`.

- **빈 프로젝트는 문서가 없는 템플릿이다.** 따로 된 프로젝트 종류가 아니다. `Template.bundle`은
  있어도 없어도 되고, `cycle`과 `robot`은 비어 있어도 된다. `templates/empty.toml`은
  `[editable] scene = empty.esscene`만 적고 과제가 갖는 것은 아무것도 적지 않는다. 이유:
  프로젝트의 템플릿을 찾는 모든 곳이 타입 하나, 찾는 법 하나를 그대로 쓴다. 프로젝트 종류를
  따로 두었다면 시작 화면, 프로젝트 파일, 모든 단계의 분기가 두 배가 되었을 것이다. 그 카드는
  (id 순으로 다른 카드 옆에) 폴더를 만든다. 시작 화면이 ①의 편집 가능한 복사본처럼
  (`make_editable`) 장면을 복사해 넣고, 그다음 `Project::create`가 `project.toml`을 쓰며 번들은
  만들지 않는다. 필요한 것은 `mujoco`, `torch`, `vulkan`, `render`이고 `mjwarp`는 아니다. 장면을
  만드는 데는 GPU 시뮬레이터가 필요 없고, 무엇이 빠졌는지는 ②의 교사 카드가 말한다.
- **②~⑤가 무엇을 돌리는지**는 함수 하나 `template::source(project, repo, generated)`가 정한다.
  - 템플릿 프로젝트(편집 불가): 템플릿의 문서, `es`는 저장소 루트에서. 예전과 똑같다.
  - 과제가 없는 편집 가능 프로젝트: 템플릿에 문서가 있으면 그 문서(SO-101 복사본), 없으면
    아무것도 없다("아직 과제가 없습니다").
  - 과제가 있는 편집 가능 프로젝트: `generated/`의 문서가 저장된 장면과 과제로 만든 것과
    정확히 같을 때 그 문서, `es`는 프로젝트 자신의 폴더에서.

  "정확히"는 `es-editor-scene`의 `on_disk`다. `generate`를 메모리에서 돌리고 파일마다 바이트
  단위로 비교한다(생성기에는 시계가 없다). `es-editor-model`에는 생성기가 없으므로 그 답은
  `es-editor`가 나른다. 프로젝트를 열 때는 디스크에서, 그 뒤로는 매 프레임 ①의 모델(저장됨,
  저장 안 됨, 마지막 생성)에서. 답이 바뀌면 ③의 소스, ②의 교사 카드와 프로그램, ⑤를 다시
  읽는다.
- **직접 만든 템플릿**(`template::authored`)은 프로젝트의 템플릿 — 그 문구, 필요한 것, 길이,
  시점 — 에 `generated/`의 학생 팔을 번들로, 그 `[teacher]` 문서, 그 사이클과 `scene.esscene`을
  넣은 것이다. 교사가 있고 사이클이 전문가를 적지 않으면 `method = "teacher"`다. `[outcome]`은
  없다. 템플릿의 결과 설명은 그 템플릿 과제의 것이고, 빠진 절로 실패를 설명하는 것은 S4의 다음
  항목이다. `untrained.esb`는 저장할 때마다 바뀌는 생성 문서로 매 실행 전에 `es policy init`처럼
  다시 만든다.
- **`es`는 프로젝트 루트에서 돈다.** 절대 경로가 아니다. 생성 문서는 `scene.esscene`을 프로젝트
  기준 상대 경로로 적고(해시 입력인 `SceneRef.path`), 레시피는 `runs/…`를 `[cycle] runs`에서
  끌어낸다. 둘 다 G3b의 `generate`가 쓰는 대로 상대 경로다. 실행 모델은 이미 모든 `es`를 어떤
  폴더에서 시작하므로(`start_in`) 문서가 적힌 그대로 돈다. 절대 경로로 했다면 생성 문서를,
  그리고 해시 입력인 장면 경로를 컴퓨터마다 다시 써야 했다. 에디터가 직접 쓰는 것(실행 폴더,
  번들, 실행 레시피)은 예전처럼 절대 경로다.
- **막기.** ①에 저장하지 않은 과제가 있을 때(`watch.task.unsaved`), `generated/`가 저장된 문서로
  만든 것이 아닐 때(`watch.task.stale`: 디스크에서 바뀌었거나 한 번도 만들지 않음), 생성이
  실패했을 때(`watch.task.failed`, 생성기의 이유와 함께) ③은 시작하지 않고 이유를 말한다. 그러면
  ①이 끝나지 않은 단계가 되어 단계 막대가 거기서 열리고, ①의 저장 버튼은 "저장하고 문서
  만들기"가 된다. 저장하면 막힘이 풀린다. ②도 패널에 같은 말을 한다.
- **"과제 정하기"는 누가 배우는지도 말한다**(4.6절의 기본값에 더해):
  - 상태: 로봇의 `joint_pos`, `joint_vel`, `previous_action`. 특권: 첫 자유 물체의 `pose`와 `vel`.
  - `[reward] scale = 0.01`(플랜 H의 것)과 중간 성공 보너스(100).
  - 비어 있는 `[teacher]`: 모든 채널, 플랜 H의 PPO 프리셋(환경 2,048개, `mjwarp`에서 3,000번
    반복, 250번마다 체크포인트).
  - 관측하는 모든 카메라를 보는 `[student]`: `joint_pos`를 읽고, 프리셋 `h3`(`tanh` 헤드),
    16행, `control_hz`를 나누는 1~10 중 가장 큰 수만큼 실행(50 Hz에서 10).
  - `[cycle]`: 학습한 교사가 시드 1001부터 시연 200번, `success_only`.

  모두 소유자가 바꿀 수 있다(9절).
- **`[teacher]`가 있는 모든 프로젝트의 교사 카드**는 플랜 H의 카드(H7)를 생성된 레시피, 평가,
  문서에 쓴 것이다. 문구는 큐브가 아니라 "모든 물체"를 말한다. 무엇이든 돌리기 전에 저장된
  장면을 `mjwarp`의 매핑 보고서로 확인한다(`check::maps_onto`). 막는 기능이 있으면 카드에 쉬운
  말로 적고 학습 버튼을 끈다. 프로젝트를 만드는 동안에는 `mujoco-cpu`로만 확인한다.
- **템플릿으로 저장**: ①의 도구 막대, 이름, 그리고 폴더
  `<문서>/Electric Sheep/templates/<이름>/`(이미 있으면 `<이름> 2`…, 저장소에는 절대 쓰지
  않는다). 저장된 문서(고친 것이 있으면 먼저 저장)를 받는다: `scene.esscene`, `task.estask`,
  그리고 `assets/` 통째로. glTF의 버퍼는 장면의 자산 목록에 없기 때문이다. 그다음 마지막으로
  `template.toml`을 쓴다: 프로젝트 템플릿의 문구, 필요한 것, 길이, 시점에 사람이 정한 이름,
  `id = "saved:<폴더>"`, 문서는 없고, 그것이 나온 기본 템플릿 `base`. 시작 화면은 저장한
  템플릿을 기본 템플릿 다음에 늘어놓는다. 그것으로 만든 프로젝트는 같은 방법으로 문서를 복사하고
  `project.toml`에 `base`를 적으므로, 사람이 지울 수도 있는 폴더에 기대지 않는다.
  `ES_DOCUMENTS`는 테스트와 캡처를 위해 문서 폴더를 바꾼다.
- **가는 길에 고친 것.** 추가 메뉴의 물체는 이름 없는 자유 관절을 갖고, 그 관절은 물체의 이름을
  받는다. 컴파일러는 "[box]가 멈춰 있다"를 그 관절로 읽어 거부했다("경첩이나 슬라이드가 아님").
  이제 이름은 경첩이나 슬라이드의 것일 때만 관절로 읽는다. `vocab::relations`가 이미 그렇게
  읽었다. 커밋된 명세의 출력은 하나도 움직이지 않았다(그 자유 관절은 물체와 다른 이름이다).
  카메라 확인도 같은 방식으로 이름을 읽는다.
- **측정**(오라클 1. 빈 프로젝트에 목록의 SO-101, (0.22, 0, 0.025)의 5 cm 상자,
  (0.22, 0.12, 0.025)의 목표 영역, "과제 정하기"와 "[box]가 [target] 안에 있다"): 문서 열세 개.
  `es ir check`가 교사 팔과, 평가 두 개 각각과 함께한 학생 팔을 통과시킨다. 두 레시피 머리말의
  `es policy init`이 번들을 만든다. ②의 교사 실행에 대한 `es train --dry-run`은 `mjwarp`에서
  `train_ppo`를 계획한다. `es loop cycle --dry-run`은 생성된 사이클과 ③이 쓰는 실행 레시피에서
  통과한다. 모두 프로젝트 폴더에서다. SO-101 장면은 `mjwarp`에 매핑된다(메시는 거기서 경고다).
- **남은 것.** 교사 카드의 시간 문장은 핸드의 것("1~1.5시간")이고, SO-101 교사의 실행은 GV
  전까지 측정되지 않았다. ⑤는 템플릿 자신의 과제에 대해서만 실패를 설명한다. 사이클이 전문가를
  적는 직접 만든 프로젝트는 돌지만 시연 프로그램 파일은 생기지 않는다.
- **R7이 정한 것**(리뷰 F-1~F-3. `teacher.rs`의 `test`, `pace`, `time_text`,
  `results::thumbnail`, `es-eval`의 `run_dir.rs`):
  - **저장한 뒤의 시험.** 체크포인트의 과제·관측·배포 해시가 학습 전 교사의 것과 다르면, 그 시험은
    작업 두 개다. 학습 전 교사는 "교사로 쓰기"가 하듯 생성된 문서로 먼저 다시 만든다. 먼저
    `weights/model-<step>.safetensors`를 그 교사에 `es policy pack`해서
    `teacher/<n>/repacked/<step>.esb`에 쓰고, 이어서 그 번들을 `es eval run`해서 `eval/<step>`에
    쓴다. 예전에 묶어 둔 번들은 먼저 지운다. 문서가 그대로면 시험은 평가 하나이고 argv도 예전과
    같다. 오늘의 생성기로 다시 생성한 GV 프로젝트 사본에서, 이렇게 다시 묶은 체크포인트 2000은
    16/16을 받았다.
  - **카메라가 여럿인 타일.** `es eval run`은 카메라마다 `frames/<cell>/<channel>/`에 쓰는데,
    `RunDir`는 `frames/<cell>/*.bin`만 셌다. 그래서 모든 시도에 그림이 없었다. 이제 자기
    `layout.json`이 없는 칸은 이름순으로 첫 카메라를 읽는다. 이 순서가 Task IR의 채널 순서다.
    타일, 플레이어의 "정책이 본 화면", Advanced의 실행 브라우저가 모두 이 카메라를 보여 준다.
  - **카드의 시간은 잰 값이고, 짐작하지 않는다.** 교사가 학습하는 동안에는 전체 반복 중 마친
    수와, 그 실행 자신의 속도로 남은 시간을 말한다. 그 전에는 가장 최근에 끝난 실행의 속도에
    레시피의 `[run] steps`를 곱한다. 속도는 손실 곡선 옆 `metrics/env-metrics.json`의 벽시계
    시간을 반복 수로 나눈 값이다. 둘 다 없으면 직접 만든 프로젝트에는 첫 실행이 잰다고 말하고,
    기본 템플릿은 원래 문장을 그대로 쓴다. GV의 교사는 반복 3,000번에 2,847초, 한 번에
    0.95초였으니 카드는 47분쯤이라고 말한다. "새 교사 학습시키기" 버튼의 설명에는 이제 시간이
    없다.

## 6. 코드가 가는 곳 (계층, §4.2)

| 무엇 | 크레이트 (계층) |
|---|---|
| `.esscene` 리더와 라이터, 인클루드 펼치기, 완전한 MJCF 내보내기 | `es-assets` (2) |
| 과제 명세 컴파일러와 `es project generate`의 핵심(순수: 장면 + 명세 → 문서) | `es-script` (11), 작성 크레이트. `es-editor-scene`이 직접 부른다 |
| `es scene export / simulate`, `es render`, `es project generate` CLI | `es` |
| 장면 모델, 명령, 실행 취소, 검사 (G5), 피킹·손잡이·모서리의 판단 (G6), 문장 편집기 모델 (G8) | `es-editor-scene` (12) |
| 계층 구조, 인스펙터 위젯, 기즈모 | `es-editor` (13) |

새 확장점은 없다(INV-17). `es-editor-model`은 줄 수 목표를 넘었으므로(10,000 중 8,942) G5가 장면
모델을 새 계층 12 크레이트 `es-editor-scene`에 두었다(5.1).

## 7. 스펙 변경 (ko 먼저, 그것이 필요한 패킷과 함께)

- §14.3 저장 형식: `*.esscene`(장면 문서)과 `*.estask`(과제 명세, IR이 아님). §14.1: 장면 문서는
  프런트엔드 하나가 더 늘어난 것이며, 같은 `SceneDesc`에 대해 그 `scene_hash`는 다른 어떤
  프런트엔드의 것과도 같다.
- §6 (`SceneRef`): 경로가 `.esscene`을 가리킬 수 있다.
- §23: 장면 작성과 문장 편집기가 에디터의 ①이 된다(M12의 읽기 전용 ①은 읽기 전용으로 여는
  템플릿에 남는다).

## 8. 오라클 (§1.4)

1. **같은 장면, 같은 해시**: `so101_pick_place.xml`(그리고 Shadow Hand 장면)이 말하는 것을
   그대로 적은 `.esscene`은 같은 `SceneDesc`로 읽히고 같은 `scene_hash` / `asset_hash`를 낸다.
2. **내보내기 왕복**: 커밋된 모든 장면과 생성된 장면에 대해 `parse_mjcf(export(scene)) ==
   scene`(속성 테스트). MuJoCo가 내보낸 것을 불러와 스텝하면 백엔드 자신의 출력과 비트 단위로
   같다(H1의 동등성 오라클).
3. **생성은 함수다**: SO-101 시점 명세와 Shadow Hand 명세는 커밋된 문서를 바이트 단위로 똑같이
   다시 만든다. 바뀐 문장은 움직여야 할 해시만 정확히 움직인다.
4. **모든 관계를 스크립트된 상태에서 검사한다**(H2의 테스트처럼): 관계를 만족하게, 그리고
   어기게 만든 상태에서의 참 거짓과 보상.
5. **에디터 명령**(헤드리스): 적용 / 취소 / 다시 실행하면 똑같은 문서로 돌아온다. 잘못된 편집은
   이유와 함께 거부된다. 이름을 바꾸면 그 이름을 쓰는 모든 문서가 다시 생성된다.
6. **끝에서 끝까지**(오케스트레이터): 빈 프로젝트에서 에디터로 새 과제를 만든다 — 예를 들어
   SO-101이 큐브를 목표 구역으로 미는 것 — 보상으로 학습한 교사로 가르치고, 카메라 학생을
   학습시키고, ⑤에서 그 결과를 읽는다. 모든 단계를 에디터에서 지켜본다.

## 9. 열린 결정 (기본값은 굵게, 소유자가 바꿀 수 있다)

1. 로봇 가져오기: **덮어쓰기가 있는 참조**(3.2), 또는 납작하게 편 복사.
2. 정체성: **이름 경로, 이름 바꾸기는 문서를 다시 생성**(3.1), 또는 엔티티마다 GUID를 저장
   (이름 변경에 강하지만 `StableId` 옆에 두 번째 정체성이 생긴다).
3. 자산: **내용으로 프로젝트에 복사**, 또는 제자리 참조.
4. 첫 버전의 문장 어휘는 4.1절의 것: **나열한 대로**, `touches`는 `GetContact` 낮추기 뒤에.
5. 보상: **문장에서 유도하고 가중치는 세 단계**. 자유 형식 보상 편집기는 Advanced 그래프(M3
   2단계)의 것이며 ①의 일부가 아니다.
6. 말한 과제를 누가 배우는가 (G9): **5.4절이 적은 기본값** — 관측 채널, `[reward] scale = 0.01`과
   중간 보너스, `mjwarp`에서 도는 플랜 H의 PPO 프리셋, `h3` 카메라 학생, 시드 1001부터
   `success_only`로 시연 200번.

## 10. 플랜 (M17 플랜 G) — 패킷 순서

| 웨이브 | 패킷 | 필요 |
|---|---|---|
| 1 | G1 `.esscene` 스키마, 리더(인클루드), 라이터. 스펙 §14.3 / §6 | — |
| 1 | G2 완전 충실도 MJCF 내보내기 + `es scene export` | — |
| 2 | G3 과제 명세 스키마 + 컴파일러 + `es project generate`. SO-101과 Shadow Hand 문서를 바이트 단위로 똑같이 다시 만든다 | G1 |
| 2 | G4 `es render`(선언된 카메라나 자유 카메라의 프레임 하나)와 `es scene simulate` | G1 |
| 3 | G5 에디터 장면 모델: 계층 구조, 인스펙터, 명령, 실행 취소, 저장, 검사. `es-editor-model` 분할 | G1 |
| 3 | G6 뷰포트 피킹과 기즈모. 카메라 모서리 화면 | G4, G5, H8 |
| 4 | G7 추가: 기본 도형, 메시 / 텍스처 가져오기, 로봇 라이브러리, 카메라, 조명, 영역. 삭제, 복제 | G5 |
| 4 | G8 문장 편집기(성공, 실패, 시작, 관찰, 보상), G3 위에 | G3, G5 |
| 5 | G9 빈 프로젝트 카드, 템플릿으로 저장, 생성된 과제를 위한 ② 교사 | G3, G8 |
| 6 | GV 끝에서 끝까지(오케스트레이터), 리뷰 | 전부 |
