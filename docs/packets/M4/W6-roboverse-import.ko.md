<!-- Korean translation of docs/packets/M4/W6-roboverse-import.md. The English file is the working copy; regenerate this when it changes. -->

# M4 W6 — RoboVerse / MetaSim 태스크 변환

## context (범위)

- `crates/es-data/src/roboverse.rs` (신규)
- `crates/es-data/src/lib.rs` (`pub mod roboverse;`만)
- `crates/es-data/tests/roboverse.rs` (신규)
- `tests/fixtures/roboverse/*.json` (신규, 손으로 작성, 각각 10 KB 미만)
- `crates/es/src/cmd/import.rs` (`roboverse` 서브커맨드 추가; 최소한의 삽입 —
  형제 패킷이 같은 파일에 `usd` variant를 추가한다)
- `crates/es/tests/cli.rs` (추가 전용)
- `docs/api-notes/roboverse.md` (신규)
- `docs/packets/M4/W6-roboverse-import.md` (이 파일)

## spec (사양)

- §14.4 (외부 변환): `RoboVerse / MetaSim`은 IR로 "Semantic Mapping Report"를
  공급하는 네 개의 이름 붙은 변환 대상(LeRobot config, Isaac Lab 태스크
  config, MJCF, RoboVerse/MetaSim, Gymnasium spec) 중 하나다; 매핑되지 않은
  항목은 `severity: error`이며 실행을 차단한다.
- §0.3 줄 "RoboVerse / MetaSim: simulator-agnostic config, 276 tasks" —
  RoboVerse 자체의 shape에 대한 유일한 spec 수준 설명이며, 그보다 더 구체적인
  모든 것은 `docs/api-notes/roboverse.md`에 있다.
- §25.2 — "Isaac Lab / RoboVerse 변환 산출물의 2차 저작물(derivative-work)
  지위: 확인 필요". 이 패킷은 라이선스 문제를 해결하지 않는다; 원본의
  `license` 필드를 `Converted::provenance`와
  `Converted::scene_refs[].license`로 조건 없이 그대로 담는다.
- §6 (Task IR, 616–709번째 줄)과 §7.4 (Observation IR 선언 링크) — 이 변환이
  대상으로 하는 두 IR, 그리고 `es_ir::cross::check`의 `task_observation`
  패스가 강제하는 경계(`ObservationSpec` 채널 <-> Observation IR 소스 노드).
- INV-16 — 이 워크스페이스 어디에서도 pickle 기반 로딩을 하지 않는다;
  RoboVerse 자체의 trajectory 포맷(`unverified`, `.pkl`/`.npz`로 추정됨)은
  구성상 이 컨버터의 범위 밖이다(JSON 태스크 *config*만 읽힌다).

## oracle (오라클)

```
cargo fmt -p es-data -p es --check
cargo clippy -p es-data -p es --all-targets -- -D warnings
cargo test -p es-data -p es
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `docs/api-notes/roboverse.md`는 구현보다 먼저 작성되며, `roboverse.wiki`/GitHub에서
  가져온 것이 아닌 모든 필드를 `unverified`로 표시한다(가져옴: `ScenarioCfg`의
  최상위 shape, `RobotCfg`의 필드 이름, 세 checker 클래스 이름; 미검증:
  `BaseObjCfg`, camera 필드, checker 생성자 필드, randomization shape,
  trajectory 포맷, 라이선스 해결).
- `RoboVerseTask`는 JSON만 파싱한다(문서화됨: MetaSim의 네이티브 Python
  config는 이 crate가 구현하지 않는 한 줄짜리 export 단계를 필요로 한다);
  알 수 없는 최상위 필드는 `extra`에 담겨 경고로 보고될 뿐 결코 거부되지
  않는다.
- `convert(&RoboVerseTask) -> Result<Converted, ConvertError>`, `Converted`는
  `task: TaskIr`, `observation: ObservationIr`, `scene_refs:
  Vec<SceneAssetRef>`, `provenance: Provenance`, `warnings: Vec<String>`,
  `unmapped: Vec<Unmapped>`를 싣는다.
- 로봇/객체는 `SceneAssetRef { path, format, license }`가 된다; camera는
  Task IR `ObservationSpec` 채널과 Observation IR `ImageInput -> Resize`
  체인 둘 다가 된다; checker는 세 개의 이름 붙은 종류(`DetectedChecker`,
  `JointPosChecker`, `PositionShiftChecker`)에 대해 `Reward` +
  `Terminate(Success)`가 되며, 그 외의 것은 `Unmapped { severity: Error }`다;
  episode length는 `TaskConfig::max_episode_steps`와
  `Terminate(Timeout)` 체인이 된다; 인식되는
  `randomization[].distribution.kind`는 `Randomization` 노드가 되고,
  인식되지 않는 것은 오류가 아니라 경고다.
- 두 개의 픽스처: `DetectedChecker`를 가진 pick-and-place 태스크, 그리고
  `JointPosChecker`와 randomization 항 하나를 가진 reach 태스크. 둘 다
  빈 `unmapped`로 변환되며 깨끗하게 검증된다: `TaskIr::validate()`,
  `ObservationIr::validate()`, 그리고 `es_ir::cross::check`에서 나오는
  `XIR-001`/`XIR-002`(Task <-> Observation) 진단이 모두 비어 있다.
- 인식되지 않는 `checker.kind`를 가진 세 번째 픽스처는 `severity: Error`를
  가진 정확히 하나의 `Unmapped` 항목으로 변환되며, 이에 대한
  `es import roboverse`는 1로 종료한다.
- `task_hash()` / `observation_hash()`는 같은 입력에 대한 반복된 `convert()`
  호출에 걸쳐 안정적이다(`HashMap` 없음, wall-clock 없음, float 비결정성
  없음 — `BTreeMap`/`BTreeSet`만).
- `es import roboverse <task.json> --out <dir>`는 `task.toml`,
  `observation.toml`, `provenance.json`을 쓰고, 경고/미매핑 항목과 두 해시를
  출력하며, `unmapped` 항목 중 하나라도 `severity: Error`이면 1로 종료한다.

## forbidden (금지)

- `es-usd`, `es-physics-core`, `es-script`, `es-ir`, `es-env`, `es-gpu`,
  `es-eval` — 다른 진행 중인 패킷들이 소유한다.
- `crates/es/src/cmd/import.rs` 주변에서 최소한의 `roboverse` 서브커맨드
  삽입을 넘어서는 다른 어떤 파일도 (형제 패킷이 같은 파일의 `usd` variant를
  소유한다).
- 새 crate 의존성 없음(`serde_yaml` 없음, PyO3 호출 없음): YAML/네이티브-Python
  config는 문서화된 한 줄짜리 export 단계이지, 이 crate의 문제가 아니다.
- pickle 기반 trajectory 읽기 없음(INV-16): 여기서 변환되는 것은 JSON 태스크
  config뿐이다.
