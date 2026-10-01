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
| 2 | G3 작업 명세 + `es project generate` · G4 `es render` / `es scene simulate` | G1 |
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
