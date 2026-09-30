<!-- Korean translation of docs/packets/M16/plan-h.md. The English file is the working copy; regenerate this when it changes. -->
# M16 플랜 H — 경로 추적 카메라 세 대로 보는 Shadow Hand 큐브 재정향

> 오너의 요청(2026-09-30): "`server-side-rendering/PhysicalAI/ShadowHand` 를 기반으로 여러 대의
> 카메라를 사용하여 비전 기반 로봇 학습을 수행 … PT 기반 시뮬레이션 환경 … 학습 진행을 es-editor 로 관찰".
> 질문을 받은 오너는 **Isaac의 작업(큐브 재정향)** 과 **모방 학습(ACT)** 을 골랐다.
> 각 작업은 §1.2의 패킷이며, **Files** 블록이 `cargo xtask check-scope`용 `context`이다.

**목표:** SSR 번들의 Shadow Hand(`shadow_hand_physics_configured.xml`, Isaac Lab의 repose-cube
작업을 MuJoCo로 옮긴 것)가 이 런타임에서 돈다. `Pt` 카메라 세 대로 보는 비전 정책이 큐브를 목표
방향으로 재정향하며, 모든 실행은 `es-editor`로 지켜본다.

**왜 교사인가.** 손 안에서 큐브를 재정향하는 일은 move/grip 블록으로 스크립트할 수 없다. 그래서
시범자 자체가 정책이다. 큐브의 자세를 읽는(특권 정보) 상태 기반 PPO 교사를 `mjwarp`에서 `[rl]`로
학습하고, 그 교사가 세 카메라 아래에서 `es loop collect --frames`를 돌린다. 학생 — 세 시점과 관절
상태, 목표를 입력으로 받고 큐브 자세는 보지 못하는 ACT — 은 그 시범으로 학습한다(교사–학생 증류,
Chen et al., "Visual Dexterity", arXiv 2211.11744). `--expert` 대신 정책으로 수집하는 것은
`es loop collect` 자체의 경로다.

**계획 전에 측정한 것**(이 PC, RTX 3060 12 GB): 번들의 MJCF는 CPU MuJoCo에서 초당 물리 스텝 39k,
MJWarp에서 256 / 1,024 / 4,096개 월드일 때 22k / 71k / 100k이다(`mujoco-warp` 3.13.0 +
`warp-lang` 1.16.0을 `.venv`에 설치했으며, 1.17은 MJWarp의 CCD 커널에서 실패한다). CPU 백엔드의
JSON 파이프로는 X7 ≈ 초당 제어 스텝 640이 나왔고, 이는 약 10⁸ 스텝이 필요한 교사에게는 너무 느리다.

## 전역 제약

- `docs/packets/M15/plan-n.md`의 전역 제약이 모두 그대로 유효하다(워크트리 + `--ff-only`, 에이전트마다
  `CARGO_TARGET_DIR` 하나, 골든은 생성기로만, 푸시 금지, 원격 서버 없음, 에이전트 패킷 안에서 학습
  실행 금지, IR 변경은 스펙 먼저, 안전 평면은 손대지 않는다).
- **커밋된 모든 문서와 골든, 숫자는 그대로다.** 손은 새 파일이다.
- SSR 번들은 읽기 전용 입력이다. 이 저장소에 필요한 것은 여기서 복사·파생하며, 출처 메모(원본 경로,
  번들의 `FILE_MANIFEST.sha256` 해당 줄, 모든 편집)를 남긴다.

## 웨이브

| 웨이브 | 작업 | 선행 |
|---|---|---|
| 1 | H1 이 런타임에서의 장면 · H0 물리 파이프의 바이너리 상태 페이로드 | — |
| 2 | H2 작업과 교사 문서, 처리량 스모크 · HT1 텍스처와 metallic-roughness 머티리얼 | H1 (HT1은 H1이 머지된 뒤이기도 하다. `es-render/src/scene.rs`를 고친다) |
| 3 | E1 교사 실행과 그 평가(오케스트레이터) · HT2 노멀 맵과 이미시브 맵, glTF 머티리얼 | H2 · HT1 |
| 4 | H1b 손 장면이 번들의 텍스처와 머티리얼을 입는다. H2의 문서를 재생성 | HT1 |
| 5 | H3 학생 문서 · E2 `Pt` 카메라 세 대로 수집, 학습, 평가(오케스트레이터) | E1, H1b |

**오너 결정 2026-09-30 (M7 검토에서 보류된 R6):** "텍스처(PBR 기반 재질) 지원도 plan H에 패킷으로
추가해줘" — 텍스처와 PBR 머티리얼을 넣는다. HT1/HT2는 메시의 선례(M10/W2)를 따라 **가산적**이다.
텍스처도 PBR 머티리얼 속성도 없는 장면은 `scene_hash`를 그대로 유지하고 오늘과 비트 단위로 똑같이
렌더되므로, 커밋된 문서와 체크포인트, 골든은 하나도 움직이지 않는다.

### 작업 H1: 이 런타임에서의 Shadow Hand 장면

**Files:** `tests/fixtures/mjcf/shadow_hand/**`(새 파일), `crates/es-physics-backend/src/{mjcf_out,mapping,mujoco,mjwarp}.rs`와
그 테스트, 파싱된 요소를 더 멀리 전달해야 할 때만 `crates/es-assets/src/mjcf/**`,
`crates/es-render/src/scene.rs`(알파), 알파 규칙을 거기에 적는다면 `docs/design/renderer.md`(+ko).

- **파생 장면** `tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.xml`과 `meshes/*.stl`(업스트림
  STL 12개. 이 파서는 `<mesh scale>`을 거부하므로 `scale="0.001"`인 열 개는 미리 스케일한다),
  `PROVENANCE.json`, gymnasium-robotics 라이선스. 편집은 모두 출처 파일에 나열한다. 이름을 붙인
  스카이박스 텍스처, 값이 하나인 사이트 크기를 세 값으로, `<general>` 서보를 `<position kp kv>`로
  (동일하다: `gainprm = kp`, `biasprm = 0 −kp −kv`), `<touch>` 센서 다섯 개 삭제, 손바닥과 큐브를
  잡는 **카메라 세 대**(`top`, `front`, `side`. 나중에 Task IR에서 96×96, 50–60 Hz), `_light`
  방출체, **면마다 색이 다른 큐브**(시각 전용 슬래브 여섯 개, 각각 다른 색. 렌더러가 텍스처가 아닌
  `rgba`를 그리기 때문이며, 이것이 없으면 카메라에게 큐브의 방향이 90° 단위로 모호하다), 같은 색으로
  칠해 카메라에 보이도록 손 옆에 둔 목표 큐브(`target`).
- **물리**: MJCF 방출기가 **고정 텐던**(J1/J0 결합 네 개)과 `<contact> <pair>` / `<exclude>`, 그리고
  `SceneDesc`가 갖고 있다면 `gravcomp`를 쓴다(파서가 떨어뜨린다면 전달하도록 한다). 매핑 보고서는
  이들을 `mujoco-cpu`와 `mjwarp`에서 네이티브로 표시한다. 아직 매핑되지 않는 것은 지금처럼 이름을
  대어 거부한다.
- **렌더**: `rgba` 알파가 0인 지오메트리는 그리지 않는다(번들의 머티리얼에서 손의 충돌 캡슐과 보이지
  않는 바닥이 알파 0이다). 알파 0 지오메트리가 없는 장면은 비트 단위로 똑같이 렌더된다(렌더 골든은
  움직이지 않는다).
- **오라클:** (1) `es backend compare --scene <derived> --backends mujoco-cpu,mjwarp`가 돈다.
  (2) 한 테스트가 파생 장면을 `MuJoCoCpuBackend`로, 그리고 같은 파일을 MuJoCo로 직접 로드해 같은
  제어로 240틱 스텝하고 qpos가 1e-9 안에서 같다(`ES_PYTHON`이 없으면 이유를 남기고 건너뛴다).
  (3) 각 카메라가 리셋 자세를 `Pt`로 렌더하며 세 카메라 모두에서 큐브의 분할(segmentation) 픽셀이
  0보다 크고, 기존 것들과 같은 래스터 대 CPU 참조 검사를 한다. (4) 렌더와 데이터셋 골든은 그대로다
  (`cargo xtask verify-goldens`).

### 작업 HT1: 텍스처와 metallic-roughness 머티리얼

**Files:** `crates/es-assets/src/{mjcf/**,scene.rs,mesh.rs}`(+ 텍스처 모듈), `crates/es-render/**`
(장면, CPU 참조, `Rs`/`Pt` Slang 커널, 생성기로 만드는 골든), 머티리얼 테이블을 렌더러까지 전달해야
한다면 `crates/es-env/src/render*`, `docs/design/renderer.md`(+ko),
`docs/ARCHITECTURE.ko.md` / `.md` §15.3(ko 먼저, 같은 커밋)과 §28의 R6 메모.

- **에셋.** MJCF `<texture>`: PNG `file`(`gridsize` / `gridlayout`, 그리고 여섯 파일 형태 포함)에서
  읽는 `type="2d"`와 `type="cube"`, 결정적으로 생성하는 `builtin="checker|gradient|flat"`
  (`rgb1`/`rgb2`/`mark`/`markrgb`/`random`), `colorspace`(`auto`/`sRGB`/`linear`).
  `<material>`: `rgba`, `texture`, `texrepeat`, `texuniform`, `emission`, 그리고 PBR 속성
  `metallic`, `roughness`(MuJoCo ≥ 3.2), 여기에 `<layer role="rgb|orm|metallic|roughness">`.
  기존의 `specular`/`shininess` 쌍은 명시적으로 주어졌을 **때만** 글로 적은 공식 하나로 유전체의
  roughness에 대응시킨다(MuJoCo의 기본값이 지오메트리를 PBR로 바꾸지는 않는다). 텍스처 바이트는
  경로가 아니라 내용으로 `asset_hash`에 해시한다. OBJ의 `vt` UV는 읽으며, STL에는 없다.
- **UV.** 프리미티브의 정점별 UV는 MuJoCo 자체의 매핑을 따른다(큐브 텍스처의 박스 면은 MuJoCo가
  `gridlayout`을 배치하는 대로, 평면 위의 2d 텍스처는 `texrepeat`/`texuniform`으로, 구·캡슐·실린더·
  타원체는 MuJoCo가 매핑하는 대로). 메시는 자체 UV를 쓰고, UV가 없으면 MuJoCo의 투영을 쓴다.
- **셰이딩.** 머티리얼 모델은 하나, glTF 2.0 metallic-roughness다. 베이스 컬러(계수 × sRGB 디코딩한
  텍셀), metallic, roughness, `F0 = mix(0.04, base, metallic)`인 GGX / Smith 높이 상관 / Schlick
  프레넬, 그리고 Lambert 확산 로브 × (1 − metallic). `Pt`: GGX 가시 법선의 중요도 샘플링, NEE 및
  ReSTIR DI와의 MIS. `Rs`의 `Full`: PBR 머티리얼의 직접광에 Blinn-Phong 대신 같은 BRDF를 쓴다.
  `Rs`의 `Lambert`: 베이스 컬러만. 샘플링은 쌍선형이고, wrap = repeat, 텍셀 중심은 +0.5, 밉맵은
  없다(경로 추적기의 샘플과 `Full`의 SSAA가 풋프린트를 평균한다. 에일리어싱이 보이면 밉 체인은 뒤의
  행으로 한다).
- **오라클.** (1) 커밋된 모든 렌더 골든과 `scene_hash`, `asset_hash`가 그대로다(`verify-goldens`,
  커밋된 문서의 해시를 다시 유도한다). (2) MuJoCo 자체 렌더러(`mujoco.Renderer`, 오프스크린)와의
  텍스처 배치 비교: 번들의 `block.png` 큐브 텍스처를 입은 박스와 체커 평면을 같은 카메라로 양쪽에서
  렌더하고, 각 면의 지배적인 텍셀 색이 일치한다(배치를 보는 것이지 셰이딩이 아니다). (3) BRDF:
  화이트 퍼니스 테스트(알베도 1, 모든 roughness, metallic 0과 1에서 `Pt` 추정값이 ≤ 1이고, CPU에서
  구적법으로 계산한 참조 적분값으로 수렴한다), CPU BRDF의 상호성과 비음수성. (4) GPU와 CPU 참조가
  비트 단위로 같거나, 경로마다 렌더러 노트가 이미 밝힌 `Full`의 가장자리 픽셀 허용 오차 안에 있다.
  (5) 새 골든은 그 생성기로 쓴다.

### 작업 HT2: 노멀 맵과 이미시브 맵, glTF 머티리얼

**Files:** HT1과 같고, 여기에 `crates/es-assets/src/gltf.rs`. 노멀 맵(UV에서 만드는 탄젠트 프레임,
MikkTSpace 호환), 이미시브 텍스처(이미시브 삼각형은 `Pt`의 광원 목록에 들어간다), 그리고 같은
머티리얼 테이블 위의 glTF 리더 `pbrMetallicRoughness`(베이스 컬러, metallic-roughness, 노멀
텍스처). 오라클은 HT1의 (1), (4), (5)와 같고, 여기에 같은 높이 필드의 기하학적 범프와 견주는
노멀 맵을 입힌 평면이 더해진다.

### 작업 H1b: 손이 번들의 외형을 입는다

**Files:** `tests/fixtures/mjcf/shadow_hand/**`, H2의 문서와 생성기. 큐브와 목표는 번들이 선언한 대로
`block.png`를 입고(면 슬래브는 없앤다), 손은 그 머티리얼을 입는다. Task IR은 새 `scene_hash`에 맞춰
재생성한다(물리는 그대로이므로 오라클은 H1의 qpos 일치다). 이것이 반영되기 전에 학습한 교사는
재생성한 문서로 다시 묶는다(`es policy init` + `es policy pack`: 가중치는 Learning IR에만
의존한다).

### 작업 H2: 재정향 작업과 교사의 문서

**Files:** 새 `tests/fixtures/shadow-hand/*.toml`, 그 생성기(`crates/es/tests/views.rs`를 따른다),
그 테스트. 문서로 표현할 수 없을 때만 코드를 건드린다(그때는 먼저 말한다).

- **Task IR** `task-repose.toml`(Isaac의 `ShadowHandEnv` 의미론이며, 표시한 곳은 단순화했다):
  제어 60 Hz(물리 120 Hz), 20개 액추에이터에 `JointPosition`. 채널은 `joint_pos`, `joint_vel`(24),
  `cube_pose`(특권), `goal_quat`, `last_action`, 그리고 이미지 셋 `rgb_top` / `rgb_front` /
  `rgb_side`(96×96, `render = { path = "pt", … }`, X7 재실행의 4 spp, `seed = "tick"`).
  `d = |q_cube · q_goal|`일 때 성공은 `d ≥ cos 0.05`(0.1 rad). 보상은 `−10 · ‖p_cube − p_ref‖`,
  `1 / (√(8(1 − d)) + 0.1)`(Isaac의 `1 / (rot_dist + 0.1)`의 작은 각도 근사형이다. `acos` 노드는
  없다), `−0.0002 · ‖a‖²`, 성공 보너스. 큐브가 `p_ref`에서 0.24 m 떨어지면 실패, 타임아웃 8 s.
  리셋: 손은 번들의 기본 자세에 작은 관절 잡음을 더하고, 큐브는 `p_ref`에 무작위 **요(yaw)** 로
  놓고, 목표도 무작위 **요**(목표는 `normalize(1, 0, 0, u)`, `u ~ U(−1, 1)`)로 한다. 즉 축소한 목표
  집합 — 수직축 둘레의 회전 — 이다. SO(3) 전체 목표는 이 PC의 예산 밖이므로 전체 집합은 뒤의 행으로
  미룬다.
- **교사**: Observation IR(상태만, 정규화), Learning IR(reach 작업의 것과 같은 MLP 액터, 더 넓게),
  Deployment IR(엔벨로프 = 관절과 제어 범위, 속도 한계는 넉넉하게 — 오너의 검토를 위해 적는다),
  `mjwarp`에서 들어가는 만큼의 env를 쓰는 `[rl]` 레시피.
- **오라클:** 문서들이 검증되고 교차 검증된다. 스크립트한 상태에서 평가한 Task IR이 기대하는 성공 /
  실패 / 보상을 준다. `es train --dry-run`. 아홉 가지 메트릭으로 하는 5회 반복 스모크(오케스트레이터) —
  E1의 크기를 정하는 처리량이다.

### E1(오케스트레이터): 교사

`mjwarp`에서 에디터로 지켜보며 학습하고, `mujoco-cpu`에서 평가한다(제외 시드). 중단 규칙은 실행 전에
적어 둔다. 예산은 H2의 처리량으로 정한다.

### 작업 H3 + E2: 학생

카메라마다 Observation IR(`observation-views.toml`의 체인), 관절 상태와 `goal_quat`, `cube_pose`는
없음. Learning IR은 `learning-views.toml`(또는 `learning-mad.toml`)과 같게, Evaluation IR, 학습
레시피, `[collect] policy = teacher`를 가진 사이클. `Pt` 카메라 세 대 아래에서 교사의 성공한
에피소드를 수집하고, 학습하고, 평가하며, 모두 에디터로 지켜본다.

### 작업 H4: 에디터가 RL 실행의 학습을 보여 준다

**파일:** `python/es/train_ppo.py`, `crates/es-py/src/{rollout,pybind}.rs`(+ `tests/rollout.rs`),
`crates/es/src/cmd/{train,telemetry}.rs`(+ `tests/cli.rs`), `crates/es-editor-model/src/model/{train_view,live_run,recent}.rs`와
`i18n/*.toml`, `crates/es-editor/src/{app.rs,ui/advanced.rs}`, `tests/fixtures/editor/train-rl/**`,
`docs/design/telemetry-protocol.md`(+ko), `docs/design/rl-continuation.md`(+ko).

E1에서 본 것: teacher는 손실 곡선으로만 지켜볼 수 있었고, `es-editor <es train --out>`은 시작
페이지를 열었다. `train_ppo.py`의 진행 줄이 `return`, `episode_len`, `entropy`,
`envelope_violation_rate`의 구간 평균과 성공 비율(`Rollout.successes()`, 추가만 함)을 더하고,
`es train`이 이를 **스트림 6**으로 다시 내보낸다(스트림 5는 네 숫자 그대로: 그 독자는 정확히 넷을
분해한다). Live 창이 이를 그리고, `training.lock`이 있는 폴더는 `loss-curve.json`에서 같은 그림으로
열리며 `--attach`가 그것을 이어 간다. 결정과 이유: `telemetry-protocol.md` 9.2절. **오라클:**
`train_rl_telemetry_publishes_the_learning`(스트림 6 = 곡선의 행, 체크포인트와 `training_hash`는
그대로), `train_rl_two_runs_are_bitwise`, `an_rl_run_folds_its_learning_beside_the_loss`,
`a_train_folder_opens_as_the_finished_live_view`.
