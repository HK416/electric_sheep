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
   H8의 렌더러).
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

여기서 에디터 안에서 학습이나 물리를 돌리는 것은 없다(§23.1, §4.2 규칙 4). 미리보기, 렌더,
생성은 에디터가 argv를 넘겨 주는 `es` 하위 명령이거나 문서들의 순수 함수다.

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

[[material]]
name = "block"
texture = { file = "textures/block.png", kind = "cube", gridsize = [3, 4], gridlayout = ".U..LFRB.D.." }
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
shape = { box = [0.1, 0.1, 0.01] }
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

각 관계는 `es-env`가 이미 낮추는 Task IR 노드로 컴파일된다(`GetBodyPose`, `GetJointState`,
`GetBodyVelocity`, `Arith`, `Norm`, `Dot`, `MathFn{Abs,Sqrt}`, `Compare`, `Logic`, `Reduce`,
`Normalize`, `Clamp`, `Concat`, `Slice`, `ResetState`, `Randomization`, `Terminate`, `Reward`):

| 관계 | 컴파일 결과 |
|---|---|
| `inside` (영역) | 대상의 위치를 영역의 상자와 축마다 `Compare`한 뒤 `And` |
| `above` / `below` (m만큼) | `Slice` z, `Arith Sub`, `Compare` |
| `near` / `farther_than` (m) | `Arith Sub`, `Norm L2`, `Compare` |
| `still` (s 동안) | 속도 `Norm`이 한계 아래 — IR-D에는 유지 노드가 없어서 "1초 동안"은 "안에 있고 거의 정지"로 컴파일된다(데모의 안정화 한계, `editor-redesign.md` §5 S4) |
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

## 5. 에디터 (①과 ②)

- **계층 구조 패널**: 장면 트리(인클루드는 접힘), 검색, 가시성. 끌어서 부모를 바꾼다.
- **인스펙터**: 선택한 것의 필드와 🎲 토글. 단위는 말로. 잘못된 값은 그 자리에서 이유와 함께
  거부한다(`SceneDesc::validate`, 매핑 리포트).
- **뷰포트**: 피킹(분할 채널이 이미 geom id를 준다), 스냅이 있는 이동 / 회전 / 크기 기즈모,
  선택 물체로 맞추기, H8의 렌더 모드, 그리고 모서리의 정책 카메라 화면을 `es render`가
  렌더한다(해상도까지 포함한 실제 관찰).
- **명령과 실행 취소**: 모든 편집은 `es-editor-model`(계층 12)의 장면 모델에 대한 명령이며,
  문서에 적용되고, 검증되고, 되돌릴 수 있다. 레이아웃(`*.eslayout`, §14.3)은 장면 문서에
  들어가지 않는다.
- **물리 미리보기**: `mujoco-cpu`에서 `es scene simulate --seconds 3 --out <traj>`를 돌려
  뷰포트에서 재생한다(에디터는 물리를 돌리지 않는다).
- **검사**는 쉬운 말로, `es`나 순수 함수가 계산한다: 매핑 리포트("이 백엔드는 텐던을 시뮬레이션할
  수 없다"), 카메라가 시작 위치에서 물체를 보지 못함(카메라마다 분할 렌더 한 번), 미리보기에서
  바닥을 뚫고 떨어지는 물체, 시작 자세에서 한계를 넘는 로봇 관절.
- **저장**은 `scene.esscene`을 쓴다. 생성은 요청이 있을 때(그리고 ③ 앞에서) 돌아가므로,
  Advanced 탭을 열지 않는 한 사람은 IR 문서를 보지 않는다.

## 6. 코드가 가는 곳 (계층, §4.2)

| 무엇 | 크레이트 (계층) |
|---|---|
| `.esscene` 리더와 라이터, 인클루드 펼치기, 완전한 MJCF 내보내기 | `es-assets` (2) |
| 과제 명세 컴파일러와 `es project generate`의 핵심(순수: 장면 + 명세 → 문서) | `es-script` (11), 작성 크레이트. `es-editor-model`이 직접 부른다 |
| `es scene export / simulate`, `es render`, `es project generate` CLI | `es` |
| 장면 모델, 명령, 실행 취소, 검사, 문장 편집기 모델 | `es-editor-model` (12) |
| 계층 구조, 인스펙터 위젯, 기즈모 | `es-editor` (13) |

새 확장점은 없다(INV-17). `es-editor-model`은 줄 수 목표를 넘었다(10,000 중 약 8,400). 장면
모델은 분할을 염두에 둔 새 모듈 묶음으로 들어가고, 플랜의 첫 에디터 패킷이 예산이 요구하는
분할을 한다.

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
