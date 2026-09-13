<!-- Korean translation of docs/packets/M2/P-M2-R2.md. The English file is the working copy; regenerate this when it changes. -->

# P-M2-R2 — 변환된 LeRobot config가 계획되고 평가된다

Spec: §7.3 (augmentation 노드), §10.4 (평가는 augmentation을 자동으로 비활성화한다),
§11.3 (`CpuPlan`), §14.4 (LeRobot config 변환).
설계 노트: `docs/design/observation-lowering.md` §3과 §9.1,
`docs/design/evaluation-execution.md` §2.2.
불변식: INV-14 (crop/resize에서의 intrinsics), INV-15 (평가에서는 augmentation이
꺼진다).
리뷰 발견 사항: `docs/reviews/M2.md` — Blocker, `crates/es-data/src/lerobot_config.rs:426`.

W6의 변환기는 `crop_is_random`에 대해 *연결되지 않은(unwired)*
`Augment { training_only: true }` 노드를 삽입했다. `topo_order`는 어쨌든 그것을
방문하는데 planner에는 `Augment` 분기가 없었으므로, `crop_shape`를 지닌 모든 LeRobot
config — 저장소에 커밋된 `diffusion_config.json` 픽스처를 포함해 — 는 컴파일할 수 없는
`ObservationIr`을 만들어냈다. W1과 W6은 서로 맞물리지 않았고, 그 둘을 가로지르는 테스트는
없었다.

## context (범위)

```
crates/es-compile/src/plan.rs
crates/es-compile/tests/observation_cpu.rs
crates/es-data/Cargo.toml
crates/es-data/src/lerobot_config.rs
crates/es-data/tests/lerobot_config.rs
crates/es-eval/src/lib.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/evaluation.rs
docs/design/observation-lowering.md
docs/design/evaluation-execution.md
docs/packets/M2/P-M2-R2.md
```

## spec (사양)

- **`es-data`** — `Augment` 노드가 체인에 연결된다: `ImageInput → Crop(centre) →
  Augment{RandomCrop, training_only} → Resize → Normalize → …`. 이 IR에서 `Augment`는
  기하학을 보존한다(`image_out`은 들어오는 `ImageSpec`을 그대로 전파한다). 따라서
  LeRobot의 random crop은 중심 `Crop` — intrinsics 변환을 지니는(INV-14) — 에 더해
  *그 윈도우의* offset jitter이며, 이것이 바로 그 노드가 이름 붙이는 것이다. 그러므로 그
  `io`는 `unary(raw_ty, cropped_ty)`가 아니라 `unary(cropped_ty, cropped_ty)`다.
- **`es-compile`** — `CpuPlan::lower`가 `Augment` 분기를 얻는다. **새 API는 없다:** 이
  계획은 평가/배포 경로이며, 여기에는 augmentation 커널이 없고 거기 도달할 학습 모드도
  없으므로, `PlanOptions { training }` 플래그는 위치가 하나뿐인 스위치가 될 것이다.
  `training_only: true`는 identity pass-through로 lowering된다 — 스텝도, 버퍼도
  없으며, 소비자는 producer의 버퍼를 읽는다 — 이것이 바로 §10.4의 자동 비활성화다.
  `training_only: false`는 `COMPILE-002`다. 노드는 그래프에서 결코 제거되지 않는다:
  그것을 벗겨내면 `observation_hash`가 저자가 결코 선언한 적 없는 그래프를 기술하게 될
  것이다. 그 플래그는 커널을 들여오는 패킷에 속한다.
- **`es-eval`** — `refuse_augmentation`은 `training_only` 노드를 건너뛰며(계획이 이미
  그것들을 비활성화했다), allow-list 밖의 다른 `Augment` 노드는 여전히 거부한다.
- **`es-data`** (같은 리뷰에서 나온 사소한 지적) — `config.json`의 `shape` 항목에는
  상한이 있다: `checked_dim`은 STATE나 ACTION 차원이 `1..=65_536` 밖이면
  `ConfigError::OutOfRange`로 거부하므로, `vec![0.0; dim]`이 신뢰할 수 없는 입력으로부터
  크기가 정해질 수 없다.

## oracle (오라클)

```
cargo test -p es-data lerobot_config
cargo test -p es-compile a_training_only_augment_is_an_identity_pass_through
cargo test -p es-compile an_augment_that_is_not_training_only_is_a_diagnostic
cargo test -p es-eval a_training_only_augment_node_is_disabled_not_refused
cargo test -p es-eval an_allow_listed_augment_node_gets_past_inv_15_and_dies_in_the_compiler
```

- `es-data`: `every_converted_observation_compiles_to_a_plan`은 두 픽스처(`act_config.json`과
  `crop_shape`를 지닌 `diffusion_config.json`) 각각의 `Converted::observation`에 대해
  두 계획 모드 모두에서 `CpuPlan::compile`을 실행하며, 진단이 — 오류든 경고든 — 없음을
  단언한다. `es-compile`은 dev-dependency일 뿐이다; es-data는 layer 10이고 es-compile은
  layer 7이므로, 이는 정당한, 엄격히 더 낮은 계층으로의 의존이다(`cargo xtask layering`이
  이를 포함한다).
- `es-eval`: end-to-end 실행은 변환된 config가 아니라 손으로 만든 픽스처를 쓰는데,
  es-data와 es-eval이 둘 다 layer 10이라 서로 의존할 수 없고(규칙 1), `capture`가
  `es-render`가 존재하기 전까지는 이미지 입력을 거부하기 때문이다. 이는 `training_only`
  `Augment` 노드가 실행되면서도 아무것도 바꾸지 않음을 단언한다: 그 리포트는 그 노드가
  없는 동일한 그래프의 리포트와 같다. 교차 지점은 위의 es-data 컴파일 오라클이다.

## acceptance (수용 기준)

- 커밋된 두 LeRobot 픽스처 모두 변환**되고** 계획된다, `Debug`와 `Release` 양쪽에서.
- 변환된 `Augment` 노드는 인바운드 엣지와 아웃바운드 엣지를 하나씩 가진다(단언됨). 그래서
  도달 불가능한 노드가 planner에 도달하지 않는다.
- allow-list에 있는 `training_only`가 아닌 노드는 INV-15를 통과하고 컴파일러에 의해
  거부된다 — `Ok(_) | Err(Plan(_))`가 아니라 정확히 그 결과로서 단언된다.
- `cargo test -p es-compile -p es-eval -p es-data`가 green이다; 바뀐 golden 없음.

## forbidden (금지)

- Augmentation *커널*: `training_only` 노드는 여기서는 identity이지, 결코 샘플링된
  crop offset이 아니다. 그것은 RNG 스트림 계약이 필요하며 이후 패킷의 몫이다.
- 계획할 수 있도록 observation 그래프를 다시 쓰거나 가지치기하는 것(INV-15의 근거:
  `observation_hash`는 선언된 것을 계속 기술해야 한다).
- crop/resize 체인 어디에서도 `rescale_intrinsics: false`(INV-14).
- `crates/es-compile/src/budget.rs`, `crates/es-env`, `crates/es-policy`,
  `crates/es-telemetry`, `crates/es`.
