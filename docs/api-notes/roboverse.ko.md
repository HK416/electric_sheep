<!-- Korean translation of docs/api-notes/roboverse.md. The English file is the working copy; regenerate this when it changes. -->

# RoboVerse / MetaSim 태스크 config — shape와 출처(provenance)

**고정 버전: 없음.** 이 워크스페이스에는 `metasim` / `roboverse_pack` 패키지가 설치되어
있지 않다; 여기 있는 어떤 것도 실행 중인 MetaSim으로부터 읽은 것이 아니다. 아래 필드는
2026-09-13에 `https://roboverse.wiki/metasim/concept/config.html`에서 (WebFetch로) 가져온
경우 `verified (fetched)`로 표시되며 — `metasim.cfg.scenario.ScenarioCfg` / `RobotCfg`
소스에 직접 접근할 수 없었기 때문이다(`metasim/cfg/scenario.py`에 대한 GitHub raw/blob
URL이 404를 반환했다) — 그 외에는 `unverified`다 (RoboVerse README, arXiv:2504.18904의
초록, spec §0.3/§14.4의 한 줄 설명 "simulator-agnostic config, 276 tasks"로부터
재구성됨). 사람이 `roboverse_pack`/`metasim` 버전을 고정하고 이 파일을 — 그리고 shape가
다르다면 `crates/es-data/src/roboverse.rs`도 — 바로잡아야, 비로소 여기 있는 내용이 근거로
취급될 수 있다 (spec §1.7).

`crates/es-data/src/roboverse.rs`는 **JSON만** 받아들이며, MetaSim 고유의 Python
`ScenarioCfg`/`TaskCfg` 데이터클래스도, YAML도 받지 않는다. 사람이(또는
`dataclasses.asdict` + `json.dump`를 한 번 호출하는 얇은 Python 스크립트가) 이 컨버터가
보기 전에 시나리오/태스크 config를 JSON으로 내보낸다; 그 단계는 여기 범위 밖이며 이
crate에는 구현되어 있지 않다 (이 crate의 의존성 목록을 새 crate 0개로 유지한다 — `serde_yaml`
없음, PyO3 호출 없음).

## 최상위 shape — verified/unverified 혼합

MetaSim의 실제 루트 config는 `ScenarioCfg`(Python 데이터클래스)이며, `verified (fetched)`가
확인한 바로는 다음을 담는다: `robots: list[RobotCfg]`, `objects: list[BaseObjCfg]`,
`cameras`, `lights`, `scene`, 그리고 시뮬레이션-런타임 필드들(`simulator`, `num_envs`,
`sim_params`, `decimation`, `headless`, `renderer`). 문서는 `ScenarioCfg`가 보상 함수,
관측 정의, 성공 판정기, 태스크 수준 로직, 종료 조건, 알고리즘별 파라미터를 담지
**않는다**고 명시한다(`verified (fetched)`, `concept/config.html`) — 이들은 별도의
`TaskCfg` / 태스크-env 클래스(`BaseTaskEnv`/`RLTaskEnv`, `verified (fetched)`,
`concept/task.html`)에 있으며, 이는 `ScenarioCfg`를 한 필드로 담고 `_reward`,
`_terminated`, `_time_out`, `_observation`, `_action_space`, `_extra_spec`을 구현한다.
이 crate는 양쪽 절반을 모두 가진 **하나의 평탄화된 JSON 문서**(`name`, `robots`,
`objects`, `cameras`, `episode_length`, `checker`, `license`)를 하나의 `RoboVerseTask`로
모델링한다 — 위 export 단계가 `TaskCfg`와 그 `ScenarioCfg`를 하나의 파일로 합쳤다고
가정한다; MetaSim 자체가 그렇게 합쳐진 문서를 쓴 적이 있다는 증거는 없다 (`unverified`).

```json
{
  "name": "pick_cube",
  "version": "unverified",
  "license": "Apache-2.0",
  "robots": [ ... ],
  "objects": [ ... ],
  "cameras": [ ... ],
  "episode_length": 250,
  "checker": { ... },
  "randomization": { ... }
}
```

## `robots[]` — 필드 이름 대부분은 `verified (fetched)`, shape/기본값은 `unverified`

`RobotCfg`(`verified (fetched)`, `concept/config.html`)는 시뮬레이터에 구애받지 않는다:
로봇 설명 하나가 `ScenarioCfg.simulator`가 지목하는 어떤 백엔드로든 컴파일된다
(`isaacgym`/`mujoco`/`sapien`/`genesis`/`pybullet`). 가져온 페이지의 최소 예시가 보여주는
필드:

| 필드 | 타입 | 비고 |
|---|---|---|
| `name` | string | `verified (fetched)` |
| `num_joints` | int | `verified (fetched)` |
| `usd_path` / `mjcf_path` / `urdf_path` | string, 하나만 존재 | `verified (fetched)` — 로봇당 정확히 하나의 asset 포맷; 이 crate는 존재하는 것을 읽고 그로부터 `format`을 추론한다 |
| `fix_base_link` | bool | `verified (fetched)` |
| `enabled_gravity` | bool | `verified (fetched)` |
| `control_type` | `{joint_name: "position"\|"velocity"\|"effort"}` | `verified (fetched)`: "Dict of joint -> control mode" |
| `actuators` | `{joint_name: BaseActuatorCfg}` | `verified (fetched)`: "Dict of joint -> BaseActuatorCfg"; `BaseActuatorCfg` 필드(`stiffness`, `damping`, `effort_limit_sim`)는 예시 하나(`stiffness=500, damping=10`)를 넘어서면 `unverified` |
| `joint_limits` | shape 미검증 | 필드가 존재한다는 것은 `verified (fetched)`; 조인트별 `[lo, hi]`는 이 crate의 추측 |
| `default_joint_positions` | 선택적 | 이름만 `verified (fetched)` |
| `curobo_ref_cfg_name` | 선택적 string | 이름만 `verified (fetched)`, 모션 플래너 상호 참조용이며 이 컨버터는 사용하지 않음 |

이 crate는 `name`, `{usd_path, mjcf_path, urdf_path}` 중 하나(-> `format` = `Usd` /
`Mjcf` / `Urdf`), 그리고 `control_type`의 키 집합을 조인트 이름 목록으로 읽는다 —
`actuators`/`joint_limits`는 파싱하지 **않는다**(모델링되지 않으며, 존재하면 `extra`를
통해 경고로 접혀 들어간다). 로봇의 asset 경로는 `es import mjcf|urdf|usd`가 실제로
임포트할 `SceneRefLike`가 된다 (spec §14.4: 이 컨버터는 config shape를 매핑할 뿐,
메시/기구학은 매핑하지 않는다 — 그것은 이미 `es-assets`가 소유한, 별도로 지목된
외부-컨버터의 일이다).

## `objects[]` — `unverified`

`BaseObjCfg`의 필드를 나열한 가져온 페이지는 없다. `RobotCfg`와의 유비, 그리고
"시뮬레이터에 구애받지 않는 asset 설명"이라는 일반적인 틀에 따른 가정된 shape:

| 필드 | 타입 | 비고 |
|---|---|---|
| `name` | string | unverified |
| `usd_path` / `mjcf_path` / `urdf_path` | string, 하나만 존재 | unverified, `RobotCfg`를 그대로 따름 |
| `pose` | `{pos: [x,y,z], rot: [w,x,y,z]}` | unverified |
| `physics` | `"rigid"` \| `"static"` \| `"articulated"` | unverified; 이 crate는 `rigid`(움직일 수 있으며 `ResetState`/`Randomization` 대상이 됨)만 그 외 모든 것(고정된 씬 지오메트리, `SceneRef`만)과 구분한다 |

## `cameras[]` — `unverified`

| 필드 | 타입 | 비고 |
|---|---|---|
| `name` | string | unverified |
| `resolution` | `[width, height]` | unverified |
| `pose` | `{pos, rot}` | unverified, world frame |
| `intrinsics` | `{fx, fy, cx, cy}` | unverified; 없으면 이 crate가 명목상의 pinhole을 합성한다(`fx=fy=width`, 중앙 principal point) — `lerobot_config.rs::nominal_camera`와 같은 폴백 |

## `checker` — `unverified` (클래스 이름만 `verified (fetched)`이며 필드 shape는 없음)

RoboVerse/MetaSim README와 API 표면은 최소한 `DetectedChecker`, `JointPosChecker`,
`PositionShiftChecker`를 성공-조건 클래스로 이름 붙인다(`verified (fetched)`: 이 세
이름이 프로젝트 자체 문서 검색 결과에 등장한다), 그러나 그 생성자 필드를 드러낸 가져온
페이지는 없다. 이 crate는 태그가 붙은 JSON 객체 `{"kind": "...", ...}`를 받아들이고
다음만 매핑한다:

| `kind` | 가정된 필드 (`unverified`) | Task IR 매핑 |
|---|---|---|
| `"DetectedChecker"` | `{"object": name, "detector": name}` | `Reward`(이진 접촉/감지 proxy, `GetContact`를 통함) + 이에 게이트된 `Terminate(Success)` |
| `"JointPosChecker"` | `{"robot": name, "joint": name, "target": f64, "tolerance": f64}` | `GetJointState` + `|q - target|`에 대한 `Compare(Le, tolerance)` -> `Terminate(Success)` |
| `"PositionShiftChecker"` | `{"object": name, "axis": "x"\|"y"\|"z", "distance": f64}` | `axis`를 따른 `GetBodyPose` 델타를 `distance`와 비교 -> `Terminate(Success)` |
| 그 외 | — | `Unmapped { severity: Error }` (spec §14.4: 매핑되지 않은 항목은 실행을 차단한다) |

## `randomization` — `unverified`

spec §0.3은 RoboVerse에 대해 "simulator-agnostic config"만 이름 붙일 뿐, 무작위화
스키마는 이름 붙이지 않는다. 이 crate는 선택적인 `{"target": "<object>.<field>",
"distribution": {"kind": "uniform", "low": f64, "high": f64}}` 목록을 받아들이며
(Task IR 자체의 `Distribution` enum을 그대로 따라, 모든 항목이 `TaskNode::Randomization`에
1:1로 매핑된다), 그 shape 가정을 넘어서면 `unverified`다. 이 crate가 인식하지 못하는
`distribution.kind`는 오류가 아니라 경고와 함께 버려진다 — 매핑되지 않은 `checker`와
달리, 빠진 무작위화 항목은 태스크를 실행이 무의미해지게 만드는 것이 아니라 성능을
저하시킬 뿐이기 때문이다.

## Dataset / trajectory 포맷 — 설계상 범위 밖 (INV-16)

RoboVerse 자체의 trajectory 저장 형식(`unverified`, 흔한 robomimic 계열 관례에 따라
`.pkl`/`.npz`로 추정됨)은 **이 crate가 읽지 않는다**: `INV-16`은 이 워크스페이스
어디에서도 pickle 기반 로딩을 금지하므로, (만약 pickle이라면) MetaSim trajectory는
`es-data::lerobot`이 읽기 전에 Python 쪽 스크립트가 LeRobot 포맷으로 다시 내보내야
한다. 여기서 변환되는 것은 태스크 *config*(JSON, 이 문서의 주제)뿐이다.

## 라이선스 / 출처 (spec §25.2)

Spec §25.2는 "Isaac Lab / RoboVerse 변환 산출물의 2차 저작물(derivative-work) 지위"를
`확인 필요`로 나열한다 — v1.0에는 답이 없으며, 컨버터가 무엇으로부터 시작했는지를
기록해야 한다는 요구사항만 있다. 이 crate는 라이선스 문제를 해결하지 않는다; 입력 JSON이
이름 붙인 `license` 문자열(있다면 무엇이든)을 조건 없이 `Converted::scene_refs[].license`와
`Converted::provenance`(이름, 버전, 라이선스)로 그대로 담아, 다운스트림에서 결정을 내릴
때 필요한 데이터를 갖도록 한다. `license`가 없으면 그 필드는 `None`이며 경고가
발생한다 — 지어낸 `"unknown"`이나 추측된 OSS 라이선스는 결코 만들지 않는다.
