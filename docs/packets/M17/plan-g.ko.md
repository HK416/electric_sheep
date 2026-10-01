<!-- Korean translation of docs/packets/M17/plan-g.md. The English file is the working copy; regenerate this when it changes. -->
# M17 플랜 G — 장면 저작 (S4)

> `docs/design/scene-authoring.md`의 패킷이다(오너의 결정은 그 1절이며, 9절의 기본값은 오너가
> 바꾸기 전까지 유효하다). 오너가 2026-10-01에 이 플랜을 시작했다("G1, G2 시작"). 각 작업은
> §1.2의 패킷이며, **Files** 블록이 `cargo xtask check-scope`용 `context`이다.

**목표:** 사람이 빈 프로젝트에서 에디터로 새 작업을 만든다 — 장면, 문장, 생성된 문서 — 그리고
MJCF, TOML IR, Rust를 쓰지 않고 학습시킨다.

## 전역 제약

- `docs/packets/M16/plan-h.md`의 전역 제약이 모두 그대로 유효하다(워크트리 + `--ff-only`, 에이전트마다
  `CARGO_TARGET_DIR` 하나, 골든은 생성기로만, 푸시 금지, 원격 서버 없음, 에이전트 패킷 안에서 학습
  실행 금지, 스펙 먼저 — `ARCHITECTURE.ko.md`와 `.md`를 한 커밋에 — 안전 평면은 손대지 않는다).
- **커밋된 모든 문서와 골든, 해시는 그대로다.** `.esscene`에서 읽은 장면은 MJCF에서 읽은 장면과
  똑같이 `SceneDesc`로 해시된다.
- 계층(§4.2): 리더, 라이터, include 전개는 `es-assets`(2), 작업 명세 컴파일러는 `es-script`(11),
  CLI는 `es`, 장면 모델은 `es-editor-model`(12), 위젯은 `es-editor`(13)에 둔다. 새 확장점은 없다(INV-17).

## 웨이브

| 웨이브 | 작업 | 선행 |
|---|---|---|
| 1 | G1 장면 문서 · G2 완전한 MJCF 내보내기 | — |
| 2 | G3a 작업 명세 → Task IR · G4 `es scene simulate`와 ①의 물리 미리보기(`es render`는 H8/H9로 들어왔다) | G1 |
| 2b | G3b 나머지 문서 + `es project generate` · G3c 세 축 관계(`Slice`, `Concat`, `Reduce`, `GetBodyVelocity` 낮춤) | G3a |
| 3 | G5 에디터 장면 모델 · G6 뷰포트 선택과 기즈모 | G1, G4, H8 |
| 4 | G7 추가(프리미티브, 메시, 로봇, 카메라, 조명, 영역) · G8 문장 편집기 | G5, G3 |
| 5 | G9 빈 프로젝트 카드, 템플릿으로 저장, 생성된 작업을 위한 ② 교사 | G3, G8 |
| 6 | GV 처음부터 끝까지(오케스트레이터), 검토 | 전부 |

### 작업 G1: 장면 문서 (`*.esscene`)

**Files:** `crates/es-assets/src/esscene/**`(새 파일), `crates/es-assets/src/lib.rs`(모듈 줄),
`crates/es-assets/tests/esscene*.rs`, `crates/es-tools/src/backend.rs`(`load_scene`이 `.esscene`을
읽는다), `tests/fixtures/esscene/` 아래의 새 픽스처, `docs/ARCHITECTURE.ko.md` / `.md` §14.3과
§6(`SceneRef`), 스키마가 세부 사항을 확정하면 `docs/design/scene-authoring.md`(+ko).

- 설계의 3절을 담은 문서 모델(`EsScene`, serde, TOML. `kind = "scene"`, `schema = 1`):
  `[physics]`, `[[include]]`(MJCF / URDF / glTF / USD 파일, 자세, 선택적 이름 접두사 — 없으면 파일
  자체의 이름을 쓴다. `[include.set]`은 에셋의 이름으로 덮어쓴다), `[[body]]`(`parent`로 이루는
  트리, 자세는 pos + xyzw 쿼터니언, 조인트는 fixed / free / hinge / slide / ball이며 axis, range,
  damping, armature를 가진다. `[[body.geom]]`의 도형은 box / sphere / capsule / cylinder /
  ellipsoid / plane / mesh이고 mass 또는 density, friction, condim, contype / conaffinity, rgba
  또는 material을 가진다), `[[material]]` / 텍스처(HT1/HT2의 필드), `[[camera]]`, `[[light]]`
  (렌더러가 오늘 조명을 받는 방식 그대로: `_light` 방출 지오메트리 관례, 또는 `SceneDesc`가 담는
  것), `[[region]]`(사이트). **읽기와 쓰기** 모두 한다(에디터는 전개 결과가 아니라 문서를
  왕복시킨다).
- `expand(doc, dir) -> SceneDesc`: 기존 리더로 include를 처리하고, 이어서 문서 자신의 엔터티를
  문서 순서대로 놓는다. 에셋은 해석하고 내용으로 해시한다.
- **오라클:** (1) `tests/fixtures/mjcf/so101_pick_place.xml`의 SO-101 로봇 부분(로봇 파일로
  나눈 픽스처)을 include하고 테이블, 큐브, 빈, 카메라, 조명을 네이티브로 쓴 `.esscene`은
  `parse_mjcf(so101_pick_place.xml)`과 **같은** `SceneDesc`, 같은 `scene_hash` / `asset_hash`로
  전개된다. Shadow Hand 장면(손은 include, 큐브 / 목표 / 카메라 / 조명은 네이티브)도 마찬가지다.
  (2) read ∘ write는 문서에서 항등이다(그리고 생성한 문서에 대한 속성 테스트). (3) 모든 거부는
  필드를 이름으로 댄다(알 수 없는 키, 매달린 parent, 알 수 없는 material, 찾을 수 없는 include).
  (4) `load_scene("x.esscene")`은 `expand`가 주는 것을 준다. 커밋된 해시는 움직이지 않는다.

### 작업 G2: 완전한 충실도의 MJCF 내보내기

**Files:** `crates/es-assets/src/mjcf/write.rs`(새 파일)와 그 모듈 줄, `crates/es-assets/tests/mjcf_write*.rs`,
`crates/es/src/cmd/scene.rs`(새 파일: `es scene export`)와 그 등록 및 도움말, 그 CLI 테스트.

- `write_mjcf(scene: &SceneDesc, assets_out: &Path) -> String`: `SceneDesc`가 담은 모든 것 —
  바디, 조인트, 지오메트리(메시 포함, XML 옆에 파일로 쓴다), 사이트, 카메라, 조명 / 방출체,
  머티리얼과 텍스처(HT1/HT2), 액추에이터, 센서, 텐던, 접촉 쌍 / 제외, gravcomp, 옵션 — 을 써서
  **`parse_mjcf(write_mjcf(s)) == s`** 가 되게 한다(해시도 같다). 동역학이 없는 것은 떨어뜨리는
  `es-physics-backend`의 `mjcf_out`과 달리, 이것은 사람이 MuJoCo 뷰어에서 여는 내보내기다.
- `es scene export <scene> --mjcf <out.xml>`: `load_scene`이 읽는 모든 장면에 쓴다(G1이 들어오면
  `.esscene`도 읽고, 그 전에는 MJCF / URDF).
- **오라클:** (1) 파싱되는 `tests/fixtures/mjcf/**`의 커밋된 모든 장면(SO-101과 그 시점들,
  Shadow Hand, 텍스처 장면, 나머지)에서 왕복 동일성과 같은 `scene_hash` / `asset_hash`, 그리고
  생성한 `SceneDesc`에 대한 속성 테스트. (2) MuJoCo(Python, `ES_PYTHON`이 없으면 건너뜀)가 내보낸
  파일을 로드해 스텝한 결과가, SO-101과 Shadow Hand 장면에서 원본을 로드한 MuJoCo와 비트 단위로
  같다(H1의 일치 방법). (3) 커밋된 골든은 움직이지 않는다.

### 작업 G3a: 작업 명세가 Task IR로 컴파일된다

**Files:** `crates/es-script/src/spec/**`(새 파일: `*.estask` 모델과 그 컴파일러),
`crates/es-script/src/lib.rs`(모듈 줄), `crates/es-script/Cargo.toml`(`es-assets`, 필요하면 `toml`),
`crates/es-script/tests/estask*.rs`, `tests/fixtures/estask/`의 새 픽스처, 스키마가 세부 사항을
확정하면 `docs/design/scene-authoring.md`(+ko) §4.

- 설계 §4의 `*.estask` 모델(serde, TOML, `kind = "task-spec"`, `schema = 1`): `scene`, `robot`
  (include), `control_hz`, `[success]` / `[failure]` 절 목록, `timeout_s`, `[start]`(배치, 🎲 항목,
  강도), `[observe]`(해상도와 render를 가진 카메라, 상태 채널, 특권 채널), `[reward]`(수준, 셰이핑).
  읽기와 쓰기를 하고, 알 수 없는 키는 이름을 대어 거부하며, §4.1의 관계는 `touches`를 뺀 전부를
  다룬다(`GetContact`가 낮춰질 때까지 `touches`는 이름을 대어 거부한다).
- `compile_task(spec, scene_dir) -> TaskIr`: 장면은 G1의 `load_scene` 경로로 읽고, 이름은
  `StableId`로 해석하며(없는 이름은 그 이름을 대어 거부), 각 절은 §4.1의 노드로, 보상은 §4.1대로,
  리셋과 무작위화, `ObservationSpec` 채널과 그 센서(`render`는 Task IR이 선언하는 그대로), 로봇의
  액추에이터로부터 `ActionSpec`을 만든다.
- **오라클:** (1) Shadow Hand repose 작업의 명세는 **`tests/fixtures/shadow-hand/task-repose.toml`과
  `task_hash`가 같은** Task IR로 컴파일된다(커밋된 문서가 기준이며, 컴파일러는
  `crates/es/tests/shadow_hand.rs`가 손으로 만들던 것을 넘겨받는다). (2) SO-101 세 시점 작업
  `tests/fixtures/visible-learning/task-views.toml`도 같다. 커밋된 문서에 어휘로 표현할 수 없는
  구성이 있으면 어휘를 최소한으로 확장하고 무엇을 확장했는지 밝힌다(커밋된 문서는 바꾸지 않는다).
  (3) 각 관계를 스크립트된 상태에서 검사한다: 참 / 거짓과 그 보상(H2의
  `the_task_scores_scripted_states`처럼). (4) 거부는 절과 필드를 이름으로 댄다. (5) 커밋된 해시는
  움직이지 않는다.

### 작업 G3b: 나머지 문서와 `es project generate`

**Files:** `crates/es-script/src/spec/**`, `crates/es/src/cmd/project.rs`(새 파일)와 그 등록, 테스트,
픽스처. G3a 이후.

- 명세와 장면으로부터: Observation IR(교사 상태, 학생 시점), Learning IR(상태 MLP 교사, 세 시점 ACT
  학생, `tanh` 헤드 — H6의 불변식), Deployment IR(조인트와 제어 범위에서 낸 엔벨로프), Evaluation
  IR(홀드아웃 시드, 해당하는 스위트, nominal 전용 짝), 학습 레시피, 사이클을 만든다. `es project
  generate --scene <s> --spec <t> --out <dir>`이 이것들을 쓴다. **오라클:** Shadow Hand와 SO-101
  views 명세는 각 세트의 커밋된 모든 문서를 해시 단위로 재생성한다(커밋된 파일이 스스로 생성된
  것이었다면 본문은 바이트 단위로).

### 작업 G4: `es scene simulate`와 ①의 물리 미리보기

**Files:** `crates/es/src/cmd/scene.rs`(G2의 `export` 옆에 `simulate` 동사), 그 테스트,
`crates/es-editor-model/src/model/{scene_view,viewport}.rs`(미리보기의 결정),
`crates/es-editor/src/**`(버튼과 재생), i18n 표.

- `es scene simulate <scene> --seconds S [--ctrl hold|zero] [--backend mujoco-cpu] --out
  <traj.estraj>`: 장면을 초기 자세에서 시작해, 모든 액추에이터를 초기 목표에 고정하거나(`hold`) 0으로
  두고(`zero`) S초 동안 스텝하며, 궤적은 리플레이가 읽는 `.estraj` 형식으로 쓴다. Task IR는
  필요 없다(장면만으로 된다).
- ①: "물리 미리보기" 버튼이 이를 실행하고(모든 에디터 실행처럼 argv로) 결과를 H8/H9의 렌더러와
  리플레이의 타임라인으로 뷰포트에서 재생한다. 실행 중과 실패했을 때(매핑 보고서의 거부를 이름으로
  댄다)의 쉬운 말 상태를 보여 준다.
- **오라클:** CLI의 궤적은 같은 장면을 `MuJoCoCpuBackend`로 직접 스텝한 것과 같다(`mujoco-cpu`에서
  비트 단위). 평면 위에서 떨어뜨린 큐브는 그 위에 멈춘다. 뷰모델의 결정은 헤드리스로 검사한다.
  Shadow Hand 프로젝트에서 찍은 미리보기 스크린샷.

**G3a 병합 결과(2026-10-01):** 두 명세 모두 커밋된 `task_hash`로 컴파일된다(Shadow Hand
`e16b44a9…`, SO-101 views `33f55c29…`). `task_graph_hash`는 다르다(노드 id가 섞이는데, 커밋된
번호 매김은 생성기의 역사다) — **오케스트레이터는 G3b에서도 의미상의 `task_hash` 일치를 목표로
받아들인다**. `inside`와 `still`이 한 축짜리인 것은 es-env의 콘 낮춤에 `Slice`, `Concat`, `Reduce`,
`GetBodyVelocity`가 없기 때문이다(G3c). 컴파일러는 각도 임계값, 카메라 초점 거리, 기울기 한계에
호스트의 `cos` / `tan`을 호출한다(G3b가 정리한다).

### 작업 G3c: 세 축 관계

**Files:** `crates/es-env/src/plan.rs`(와 그 테스트), `crates/es-script/src/spec/compile.rs`(절
함수만), `crates/es-script/tests/estask_scripted.rs`, 설계 노트 §4.1(+ko).

- es-env가 보상 / 종료 콘에서 `Slice`, `Concat`, `Reduce`, `GetBodyVelocity`를 `es-ir`가 정의한
  의미대로 낮춘다(`crates/es-ir/src/task.rs`의 출력 타입과 CPU 참조 평가기가 있으면 그것을 확인한다).
  그러면 관계가 3-벡터를 읽을 수 있다.
- 그러면 어휘는 장면 **영역**(`.esscene`의 `[[region]]` / MJCF site 박스, 세 축 모두) 안에 있음을
  뜻하는 `inside`, 몸체의 선속도(3차원)와 선택적으로 각속도를 쓰는 `still`, 다른 몸체의 `above` /
  `below`, 그리고 `<body>.y` / `.z` 주어를 제공한다.
- **오라클:** 낮춰진 각 노드를 스크립트된 상태에서 손으로 계산한 값과 비교한다(G3a의
  `estask_scripted.rs`처럼). 커밋된 모든 작업 문서의 평가는 그대로다(`task_hash`와 reach / SO-101 /
  Shadow Hand 보상 골든, 비트 단위 RL 테스트). 두 기준 명세는 여전히 커밋된 `task_hash`로
  컴파일된다.

**G3b 병합 결과 (2026-10-01):** `es project generate`는 두 기준 세트, 곧 Shadow Hand 세트와 SO-101의
views / cam / MAD 팔을 IR 문서 하나하나 의미 해시로, 레시피와 사이클은 동일한 구조체로 다시
생성한다. `visible-learning/deployment.toml`(손으로 조정한 엔벨로프)만이 장면의 범위로는 만들어지지
않는 유일한 커밋 파일이다. 호스트의 `cos` / `tan`은 그대로 둔다(sin/cos 기반 `tan`은 45° 카메라의
초점 거리를 1 ULP 움직인다). musl의 `tan`은 조사한 모든 시야각에서 호스트와 비트가 같았으므로,
G3d는 해시를 움직이지 않고 호스트 호출을 없앤다.

### 작업 G3d: 호스트 libm 없는 생성 문서

**Files:** `crates/es-math/src/approx*`(`sin_cos_f64` 옆에 `tan_f64`), `crates/es-script/src/spec/compile.rs`
(호스트 호출 세 곳), 설계 노트 §4.3(+ko). 각도 임계값의 코사인, 카메라 초점 거리의 탄젠트, 기울기
한계에 `es_math::approx`를 쓴다. **오라클:** 커밋된 모든 `task_hash`와 G3b의 모든 해시가 움직이지
않아야 하며, `tan_f64`를 호스트와 1°–179° 및 커밋된 시야각 전체에서 훑어 비교한다. 비트가 하나라도
움직이면 멈추고 보고한다(결정은 소유자의 몫이다).

### 작업 G5: 편집기의 장면 모델

**Files:** 장면 모델(새 모듈, 또는 `es-editor-model`의 10,000줄 상한 때문에 필요하면 새 layer-12
크레이트 `es-editor-scene` — 그 경우 §4.2 / 부록 C.8의 `LAYERS`를 ko 먼저, xtask의 표도 같은
패킷에서 고친다), `crates/es-editor/src/ui/**`(계층 구조, 인스펙터), i18n.

- 편집 가능한 프로젝트는 `scene.esscene` + `task.estask`를 가진다. 모델은 `EsScene` 문서(G1), 선택,
  **명령**(엔티티 추가 / 삭제 / 복제, 필드 설정, 포즈 설정, 부모 변경, 이름 변경 — 이름을 바꾸면
  `task.estask`의 이름도 함께 다시 쓴다), 실행 취소 / 다시 실행 스택, 매 명령 뒤의 검증(G1의
  `expand` + 프로젝트 백엔드의 매핑 보고서, 거절은 해당 필드에 쉬운 말로), 저장으로 이루어진다.
  템플릿 프로젝트의 장면은 읽기 전용으로 두고 "편집 가능한 사본 만들기"를 제공한다.
- ①: 계층 구조 패널(트리, include 접기, 검색, 가시성)과 인스펙터(`es_math::approx`를 거친 미터 단위
  포즈와 오일러 각도, 모양과 크기, 질량, 마찰, 색 / 재질 / 텍스처, 관절 종류와 범위)를 두고 명령을
  통해 편집한다. 뷰포트(H9)는 변경 시 다시 렌더링한다.
- **오라클:** 헤드리스에서 적용 / 취소 / 다시 실행하면 동일한 문서로 돌아온다(임의 명령 열에 대한
  속성 테스트). 잘못된 편집은 필드를 짚어 거절한다. 이름 변경은 명세를 다시 쓰고, 다시 생성한 문서
  (G3b)는 여전히 검증을 통과한다. 저장한 문서를 다시 읽으면 같다. Shadow Hand 프로젝트 사본에서 ①로
  큐브의 크기와 색을 편집하는 스크린샷.

### 작업 G6: 뷰포트 선택과 기즈모 (G5 이후)

**Files:** `crates/es-editor-model` / 장면 크레이트(선택과 기즈모 결정), `crates/es-editor`(그리기,
입력), `crates/es-render`는 세그멘테이션 읽기에 API가 필요할 때만.

- 클릭으로 선택(프로세스 내 렌더의 세그멘테이션 채널이 geom → 그 엔티티를 준다), 스냅(cm, 15°)이 있는
  이동 / 회전 / 크기 기즈모, 선택 항목 맞춰 보기, 드래그는 하나의 명령(실행 취소 한 단계), 뷰포트
  모서리에 정책 카메라의 화면(선언된 카메라, 그 해상도와 렌더 경로, 프로세스 내).
- **오라클:** 스크립트된 광선에 대한 헤드리스 선택 → 엔티티와 기즈모 드래그 → 포즈 변화량. 드래그는
  실행 취소 한 단계. 기즈모 드래그와 모서리 화면의 스크린샷.

**G3d 병합 결과 (2026-10-01):** 컴파일러는 `cos`와 `tan`을 `es_math::approx`(`libm` 크레이트를 통한
musl)에서 가져오므로 생성된 문서는 호스트에 의존하지 않는다. 커밋된 해시와 G3b가 고정한 해시는
하나도 움직이지 않았다. G3b 문단의 "musl의 `tan`은 조사한 모든 시야각에서 호스트와 비트가
같았다"는 커밋된 각도(45°, 70°)에서만 성립한다. 1°–179°를 0.001° 간격으로 훑으면 값의 4.1 %가 1 ULP
다르다(어느 쪽도 올바르게 반올림된 값이 아니다). 따라서 G3d 이전에 Windows에서 그런 각도(예: 68°
카메라)로 생성한 문서는 다시 생성한 문서와 마지막 비트가 다를 수 있다. 오케스트레이터는 musl을
유지했다 — 해시에 들어가는 수에 대한 명세 §5.3의 규칙이다. `crates/es/tests/{shadow_hand,views}.rs`의
픽스처 생성기는 여전히 호스트를 호출하며, 커밋된 각도에서는 일치한다.
