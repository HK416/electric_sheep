<!-- Korean translation of docs/packets/M2/W6-lerobot-config.md. The English file is the working copy; regenerate this when it changes. -->
# W6 — LeRobot 정책 config -> IR 변환 (Rust 절반)

사양: §14.4 (외부 변환: LeRobot 변환이 v1.0의 최우선 변환 타깃), §14.2 (Python 빌더가 만드는
Observation IR / Learning IR의 모양 — 변환 결과가 이것과 어긋나면 안 된다), §7.4/§7.5
(`ObservationSpec`과의 연결, 시간 모델 3계층), §8.4 (`PolicyContract`), §5.1 (IR 경계),
§28.5 W6 (`docs/ARCHITECTURE.ko.md` 표: "Python 빌더 완성, LeRobot config 변환").

**이 패킷은 Rust 절반만을 다룬다.** 사양 §14.2가 설명하는 Python 빌더(PyO3, `es-py`)는 W6의
*나머지* 절반이며 **여기서는 구현하지 않는다** — 이는 `crates/es-data`의 범위 밖에 있는,
`crates/es-py`를 대상으로 하는 별도의 패킷이다.

## context (범위)

```
crates/es-data/src/lerobot_config.rs
crates/es-data/src/lib.rs                  (+ `pub mod lerobot_config;`)
crates/es-data/Cargo.toml                  (+ `es-math` dependency, for `ImageSpec::extrinsics`)
crates/es-data/tests/lerobot_config.rs
tests/fixtures/lerobot_config/act_config.json
tests/fixtures/lerobot_config/diffusion_config.json
tests/fixtures/lerobot_config/stats.json
tests/fixtures/lerobot_config/unknown_type.json
docs/api-notes/lerobot-config.md
docs/packets/M2/W6-lerobot-config.md
```

## spec (사양)

사양 §1.7에 따라 이 워크스페이스에는 `lerobot` 버전이 고정되어 있지 않다:
`docs/api-notes/lerobot-config.md`를 먼저 작성하고, 모든 `config.json` / `stats.json`
필드에 미검증 (unverified)을 표시한다 — 이는 `docs/api-notes/lerobot-dataset.md`와 동일한
기준이다.

1. `LeRobotPolicyConfig` (serde, `#[serde(tag = "type")]`, `act` / `diffusion`)와
   `LeRobotPolicyConfig::parse(json) -> Result<Self, ConfigError>`: 인식되지 않는 `"type"`은
   추측된 매핑이 아니라 `ConfigError::Unsupported(type)`이 된다(사양 §14.4: Semantic Mapping
   Report의 `severity=error`는 실행을 차단한다). 이 크레이트가 모델링하지 않는 필드는 각
   variant마다 `#[serde(flatten)] extra: BTreeMap<String, Value>` 가방에 담기며
   [`Converted::warnings`]로 노출된다 — 결코 조용히 버려지지 않고, 결코 하드 에러가 되지
   않는다.

2. `Stats` (`BTreeMap<String, FeatureStats>`)는 `meta/stats.json`의 특성별 mean/std/min/max를
   모델링하며, `docs/api-notes/lerobot-dataset.md`가 그 외에는 *무시된다*고 기록한 파일의
   첫 번째 소비자다.

3. `convert(cfg, stats, dataset_info) -> Result<Converted, ConfigError>`는 하나의 일관된
   `ObservationIr` + `LearningGraph`를 만든다:
   - 각 `VISUAL` 입력 특성: `ImageInput -> [Crop (Diffusion의 `crop_shape`, `CropMode::
     Center`, `rescale_intrinsics = true`, INV-14)] -> Resize -> Normalize (`stats`로부터의
     `MeanStd`, 또는 api-note에 명시된 `ImageNet`/항등 대체값) -> [`n_obs_steps > 1`일 때
     `TemporalWindow` 노드 + `History` 항목]`. `crop_is_random`은 추가로 연결되지 않은
     (unwired), `training_only`인 `Augment::RandomCrop` 노드를 삽입한다(§7.3/INV-15:
     `Evaluation IR`이 이를 구조적으로 비활성화하여, eval/sim/real이 실행하는 것은 결정적인
     중앙 크롭만 남긴다).
   - 하나의 `STATE` 입력 특성: `StateInput -> Normalize -> [TemporalWindow]`, 동일한
     윈도잉 규칙을 따른다.
   - Learning IR은 `crates/es-policy`가 기대하는 로어링을 그대로 반영한다(사양 §8.3):
     카메라당 하나의 `VisionEncoder` + 하나의 `StateEncoder` -> `Fusion(Concat)` ->
     `TemporalEncoder` (`n_obs_steps = 1`일 때 `None`, 그 이상에서는 `Transformer`) ->
     `PolicyHead` (ACT는 `Regression`, Diffusion Policy는 `Diffusion { n_steps, scheduler }`)
     -> `ActionChunker` -> `Normalizer(Inverse)`. `PolicyContract`는 사양 §8.4에 따라
     채워진다: `observation_window = n_obs_steps`, `horizon = chunk_size`/`horizon`,
     `execute_chunk = n_action_steps`, `action_dim`은 `ACTION` 출력 특성으로부터,
     `replanning_hz`는 `dataset_info.fps`로부터 (값이 없을 때는 사실인 척하는 추측이 아니라
     경고).
   - 변환된 쌍은 `ObservationIr::validate`, `LearningGraph::validate`를 통과하며, (테스트
     파일은 `es_ir::cross::check`가 완전한 `IrBundle`을 요구하므로 그 주위에 최소한의
     합성 Task IR / Deployment IR을 구성한다) `es_ir::cross::check`의
     Observation<->Learning 경계 규칙도 통과한다.

## oracle (오라클)

```
cargo fmt -p es-data --check
cargo clippy -p es-data --all-targets -- -D warnings
cargo test -p es-data
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `act_config.json`과 `diffusion_config.json`(직접 작성, 10 KB 미만, `lerobot`의 ACT /
  Diffusion Policy `config.json` 구조를 대표하며 — 모든 필드가 미검증 (unverified))는 둘 다
  변환되고, IR별로 깔끔하게 검증되며, 결과로부터 만들어진 범용 합성 Task/Deployment IR 쌍에
  대해 Observation<->Learning 경계에서 깔끔하게 교차 검사를 통과한다.
- `n_obs_steps > 1`은 스트림별 `History` 깊이를 등록하고 `ObservationIr::temporal.window`를
  설정한다.
- Diffusion의 `crop_shape`는 `rescale_intrinsics = true`이고 주점(principal point)이 이동한
  `Crop` 노드를 만든다(INV-14); `crop_is_random`은 추가로 `training_only`인 `Augment` 노드를
  만든다.
- `unknown_type.json`(`"type": "smolvla"`)은 패닉이 아니고 최선 추측의 ACT/Diffusion 매핑도
  아닌, `ConfigError::Unsupported`가 된다.
- 인식되지 않는 `config.json` 필드(예: `optimizer_lr`)는 오류도 아니고 조용히 버려지지도
  않으며 `Converted::warnings`의 경고 문자열이 된다.
- 동일한 입력에 대해 `convert()`를 두 번 실행하면 동일한 `observation_hash` /
  `learning_hash`가 나온다(사양 §5.3 결정성).

## forbidden (금지)

- Python 빌더(PyO3, `es-py`) — 별도의 패킷이며 이 패킷이 아니다.
- `crates/es-env`, `crates/es-eval`, `crates/es-policy`, `crates/es-compile`, `crates/es`
  (동시 진행 중인 M2 패킷들이 소유) 그리고 루트 `Cargo.toml`.
- `HashMap`/`HashSet`(사양 §3.4 결정성) — `BTreeMap`만 사용.
- 새로운 확장 지점 트레이트(INV-17); 이 패킷은 어떤 것도 추가하지 않는다.
- 커밋 — 오라클은 이 패킷에 의해 실행되고 보고될 뿐, 랜딩되지 않는다.
