# M5 V3 — 랜덤화 평가, Safety Plane 이벤트 스트림, 그리고 데모 실행

설계 노트: `docs/design/visible-learning.md` 섹션 2.7과 8; 섹션 2.7을 먼저 읽을 것 — Task IR 랜덤화는
큐브를 옮길 수 있지만 `PerturbationKind::ObjectPose`는 못 하고, 시각 계열 넷은 의도적으로
`Unsupported`로 남는다. V2 (학습된 번들)와 V0b (프레임)에 의존; V4가 영상으로 만들 입력을 생산한다.

## context

```
crates/es-eval/src/perturb.rs
crates/es-eval/src/runner.rs
crates/es-eval/src/lib.rs
crates/es-eval/tests/evaluation.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/evaluation.toml
crates/es-eval/tests/visible_learning.rs
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/packets/M5/V3-perturb-safety.md
docs/packets/M5/V3-perturb-safety.ko.md
```

메모: `perturb.rs`는 커널 둘을 얻고 `Unsupported` 분기 둘을 잃는다; `runner.rs`는 스텝별 이벤트
레코드와 프레임 싱크를 얻는다; `eval.rs`는 `--frames <dir>`를 얻는다.
`docs/design/evaluation-execution.md` 섹션 3은 지원 표이며, 어떤 계열이 왜 미지원인지를 말하는 곳이므로
같은 커밋에서 갱신해야 한다.

## spec

- §10.1, §10.2: 스위트는 `섭동 스위트 x 메트릭 x 수용 기준`이다. 두 계열이 `Unsupported`에서 구현으로
  옮겨가고, 나머지는 새 상황에 맞게 다시 쓴 사유를 유지한다.
- §10.4: 공정성 — 모든 추출이 `EnvRng::new(seed, suite_id, episode_idx, stream)`이다
  (`crates/es-eval/src/perturb.rs:9-12`). 조명 계열을 켜는 것이 다른 스트림의 커서를 움직여서는 안 된다.
  움직이면 기존 모든 리포트가 바뀐다.
- §10.5: `es eval compare A.json B.json`이 이 패킷이 쓰는 리포트에 대해 여전히 동작한다.
- §9.3–§9.5, INV-12, INV-13: 붉은 오버레이는 Safety Plane의 기존 스텝별 결과를 읽는다. 어떤 코드 경로도
  플레인을 끄지 않는다; 데모의 클램프는 테스트 훅이 아니라 **Deployment IR에서 좁힌 엔벨로프**에서 온다.
  `SafetyPlane::validate`는 시그니처를 유지한다.
- INV-11: `es-safety`는 아무것도 얻지 않고 새 의존성도 없다.
- §26.1: Observation IR과 렌더러 사이의 `ImageSpec` 불일치는 에러로 남고 (V0b), 이 빌드에 커널이 없는
  섭동은 이름을 밝히는 `EvalError::Unsupported`로 남는다 — 조용한 무동작은 결코 아니다 (`perturb.rs`가
  이미 따르는 §17.2의 규칙).
- §3.5: 리포트는 `evaluation_hash`로 재현되고, 프레임은 CPU 렌더 경로에서 바이트 동일하며, 물리는 비트
  단위가 아니라 `DeterminismTier::PhysicsMeaning`이다. 설계 노트 섹션 9가 그 표다.
- §12.4: 데모의 대표 수치는 초당 무엇이 아니라 성공률이다.
- §1.5: `es-eval`은 2,139 코드 줄; 이 패킷은 약 500줄 이하로 잡는다.

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo clippy -p es-eval --features render --all-targets -- -D warnings
cargo test -p es-eval
cargo test -p es-eval --features render --test visible_learning
cargo test -p es --test cli eval_run
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

레퍼런스 — 데모 실행 자체, 오라클 서버에서:

```
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es-eval --features render --test visible_learning -- --ignored --nocapture
```

`tests/visible_learning.rs`가 `tests/fixtures/visible-learning/evaluation.toml`에 따라 V0의 장면을 V2의
학습된 번들로 돌리고 결과를 판정한다. `mujoco`나 `torch`가 없으면 `SKIP visible_learning: <why>`;
실행되면 `RAN visible_learning`. 진짜 백엔드/런타임 에러는 SKIP이 아니라 **실패**다.

**게이트, 그리고 그것은 비공허하다** (W1d의 게이트가 쓰는 규칙, `docs/design/ros2-boundary.md` 섹션 7.5):
실행 결과에 `Termination::Success` 에피소드가 최소 하나 **그리고** `Clamped`로 분류된 스텝이 최소 하나
있을 때에만 통과한다. Safety Plane이 한 번도 발동하지 않는 데모는 아무것도 증명하지 않고, 성공이 없는
데모도 마찬가지다. 어느 쪽도 플래그로 만들어지지 않는다: 성공은 V2의 정책에서, 클램프는 스위트의 좁힌
엔벨로프에서 온다.

`tests/visible_learning.rs` 및 `tests/evaluation.rs`:

- `sixteen_cells_produce_sixteen_frame_dirs` — 그리드는 **독립적인 단일 env 에피소드 16개**다.
  `Evaluation::run`이 `BatchDomains::single_env()`를 하드코딩하고 (`crates/es-eval/src/runner.rs:132`)
  `MuJoCoCpuBackend`가 `max_envs: 1`을 선언하기 때문이다 (`crates/es-physics-backend/src/mujoco.rs:43`).
  테스트는 셀마다 `frames/<cell>/` 하나씩, `layout.json`이 동일함을 단언한다. V4가 모자이크하려면 그것이
  필요하다.
- `events_json_has_one_record_per_frame` — `events.json` 길이가 프레임 수와 같고, `frame`이 조밀하게
  증가하며, 모든 `tick`이 그 스텝의 `PhysTick`이다.
- `a_tightened_envelope_clamps_and_is_recorded` — 데모 스위트의 제한에서 최소 한 레코드가 `Clamped`이고,
  그 스텝의 `ViolationKind` 비트셋이 0이 아니며, `SafetyCounters`가 레코드 수와 일치한다.
- `a_widened_envelope_never_clamps` — 같은 실행을 넓은 엔벨로프로 하면 `Clamped` 레코드가 0이고 궤적은
  같다. 오버레이가 정책이 아니라 플레인을 읽는다는 증거다.
- `the_plane_is_never_disabled` — `es-eval` 소스 스캔에 `validate`를 건너뛰는 경로가 없고 (INV-12),
  `SafetyPlane::validate`의 시그니처가 그대로다 (INV-13).
- `light_intensity_and_direction_change_the_frame` — 렌더러가 있으면 두 새 계열이 기준 셀과 다른 프레임을
  만들고, 없으면 기존 `NO_RENDERER` 메시지로 `EvalError::Unsupported`로 남는다.
- `the_four_remaining_kinds_are_still_unsupported_by_name` — `ColorTemperature`, `Occluder`,
  `CameraExtrinsic`, `CameraIntrinsic`이 각각 계열 이름과 사유를 밝히는 `EvalError::Unsupported`를
  반환한다; `ObjectPose`는 자기 사유를 유지한다 ("`Env::reset`이 상태 오버라이드를 받지 않는다",
  `crates/es-eval/src/perturb.rs:196-200`). 데모의 큐브 포즈 변화는 이 계열이 아니라 V0의 Task IR
  `Randomization`에서 온다.
- `enabling_the_light_kinds_does_not_move_other_streams` — 기존 평가 픽스처의 리포트가 이 패킷 이전에
  커밋된 것과 바이트 동일하다. §10.4 공정성 검사이며, 부주의한 스트림 인덱스를 가장 잘 잡을 테스트다.
- `the_report_is_reproducible` — 같은 `evaluation_hash`를 두 번 돌리면 `report.json`과
  `evaluation.lock`이 동일하다.
- `the_frames_are_byte_identical_on_the_cpu_path` — 두 번 실행이 바이트 동일한 `frames/**/*.bin`을
  만든다 (설계 노트 섹션 9).

`crates/es/tests/cli.rs`: `eval_run_frames_flag_writes_frames_and_events`,
`eval_run_without_frames_is_unchanged` (기존 리포트 출력이 바이트 동일),
`eval_run_frames_without_a_renderer_is_a_usage_error` — 프레임 없이 조용히 도는 대신 빠진 피처 이름을
밝히며 exit 2.

## acceptance

```rust
// crates/es-eval/src/runner.rs
/// 렌더된 프레임마다 레코드 하나. `frames/` 옆에 `events.json`으로 기록된다.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct StepEvent {
    pub frame: u64,
    pub tick: es_core::PhysTick,
    pub source: ActionSource,     // Policy | Clamped | Fallback | Human
    pub events: u32,              // 이 스텝의 ViolationKind 비트셋
}

/// 실행이 프레임과 이벤트를 두는 곳; `None`이면 정확히 현재 동작.
pub struct FrameSink { pub dir: std::path::PathBuf }
```

```
es eval run --config eval.toml --policy trained.esb --scene so101_pick_place.xml \
            --out demo-out [--frames demo-out/frames]
# demo-out/
#   report.json  evaluation.lock  report.html
#   frames/<cell>/000000.bin ... + layout.json
#   events.json
```

- `ActionSource`는 `es-data`가 이미 `SafetyCounters` 델타에서 유도하는 4분류다
  (`crates/es-data/src/collect.rs:461-471`); V3는 두 번째 유도를 만드는 대신 그것을 재사용한다. 공유를
  위해 옮겨야 한다면 `es-eval` (10)로 가는데 `es-data` (10)는 그것에 의존할 수 없으므로, 원본을 밝히는
  주석과 함께 복제하거나 별도 패킷에서 `es-safety` (8)로 올린다 — 여기서는 아니다.
- `PerturbationKind::LightIntensity`와 `LightDirection`은 `TriScene` 업로드 전 장면 광원을 스케일·회전
  하는 커널로 해석된다; 각각 자기 `stream`에서 추출하고, 기존 어떤 스트림의 커서도 움직이지 않는다.
- `ColorTemperature`, `Occluder`, `CameraExtrinsic`, `CameraIntrinsic`, `ObjectPose`는 이 패킷 이후에도
  참인 사유와 함께 계속 `EvalError::Unsupported`를 반환하고, `docs/design/evaluation-execution.md`
  섹션 3의 표가 그에 맞게 갱신된다.
- 데모 Deployment IR은 한 스위트 셀에 대해 속도·변화율 제한을 좁힌다. 어떤 플래그, `cfg`, 테스트 훅도
  플레인을 끄거나 우회하지 않는다 (INV-12).
- `es-eval/render` 없이 `--frames`는 사용법 오류(exit 2)이지, 조용히 아무것도 쓰지 않는 실행이 아니다.
- 새 트레이트 없음, 새 외부 의존성 없음, 약 500 소스 줄 이하.

## forbidden

- `crates/es-ir`, `crates/es-ir-types` — 새 `PerturbationKind`, `MetricSpec`, 스키마 변경 금지.
- `crates/es-safety` — 엔벨로프는 Deployment IR 문서에서 넓히거나 좁힌다; 크레이트는 건드리지 않고
  `validate`는 시그니처를 유지한다 (INV-11, INV-12, INV-13).
- `ObjectPose` 구현: `Env::reset_with`가 필요하고 그것은 `es-env` 패킷이며, 데모에는 필요 없다 (Task IR
  랜덤화가 이미 큐브를 옮긴다).
- `CameraExtrinsic` / `CameraIntrinsic` 구현: INV-14가 `ImageSpec` 내부 파라미터를 근사하지 말고 카메라를
  따라 움직이라고 요구하며, 그것은 섭동 커널 이상의 작업이다.
- `BatchDomains::single_env()`를 올리거나 `--envs` 플래그를 추가하는 것: 그리드는 독립 에피소드 16개이며
  (설계 노트 섹션 2.1), 시뮬레이션 도메인 배치는 다른 패킷이다.
- `crates/es-render`, `crates/es-env/src/render.rs` — V0b가 렌더러 표면을 소유한다.
- 무엇이든 모자이크·오버레이·인코딩하는 것 — V4.
- 움직인 RNG 스트림을 수용하려고 기존 평가 리포트 골든을 편집하는 것.
