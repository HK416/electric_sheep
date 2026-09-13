<!-- Korean translation of docs/packets/M2/W5-memory-budget.md. The English file is the working copy; regenerate this when it changes. -->

# W5 — 메모리 예산 모델 + `es bench --memory-report` (오프라인 부분)

명세: spec 20.1, spec 20.2, spec 20.3, spec 15.2, spec 12.4, spec 5.2,
spec 28.4, spec 28.7 gate 13. 설계 노트: `docs/design/memory-budget.md`.

## context (범위)

```
crates/es-compile/src/budget.rs      (new)
crates/es-compile/src/lib.rs         (adds `pub mod budget;` and its re-exports)
crates/es-compile/Cargo.toml         (no change)
crates/es/src/cmd/bench.rs           (new)
crates/es/src/cmd/mod.rs             (adds `pub mod bench;`)
crates/es/src/main.rs                (adds `bench` dispatch + usage line)
crates/es/Cargo.toml                 (adds `es-telemetry` dependency)
crates/es/tests/cli.rs               (appends bench tests)
docs/design/memory-budget.md         (new)
docs/packets/M2/W5-memory-budget.md  (new, this file)
```

다른 M2 W5 하위 패킷들(GPU lowering 최적화, 9개 지표 테이블 배후의 실시간
측정 루프, 예산 모델을 `es-env`의 자동 축소와 `es-editor`의 컴파일 타임
검사에 연결하는 것)은 **범위 밖**이다: 이 패킷은 정적/오프라인 절반뿐이며,
달리 될 수도 없다 — 이 환경에는 대조할 GPU가 없다(spec 28.4의 ±10%
정확도 게이트는 여기서는 미검증 (unverified)이다, 설계 노트 §6 참고).

## spec (사양)

- `es_compile::budget::MemoryBudget::estimate(&BudgetInputs) -> MemoryReport`
  — spec 20.2의 모든 항목(`physics_state`, `render_tile_atlas`,
  `observation_intermediates`, `history_buffers`, `policy_weights`,
  `inference_activations`, `chunk_buffers`)을, 각각 감사용으로 공식을
  적어둔 `BudgetItem { name, bytes, formula }`로 낸다. `bytes == 0`이면서
  formula가 `"unavailable: ..."`로 시작하는 것은 "공짜"가 아니라 "데이터
  없음"을 뜻한다 — `policy_weights`는 항상 unavailable이다(`WeightsRef`는
  바이트 크기를 싣지 않는다, `INV-16`).
- `MemoryReport { items, total_bytes, per_domain: BTreeMap<String, u64>,
  bandwidth_per_tick }`와 그 `Display`(GiB 테이블).
- `MemoryReport::violations(&BudgetInputs) -> Vec<BudgetViolation>` — spec
  20.3의 `obs_batch <= sim_batch` 규칙(항상 검사됨)과
  `total <= device_bytes - reserve` 규칙(`device_bytes`가 주어졌을 때
  검사됨), 그리고 `TileAtlasCfg`가 주어졌을 때의 `maxImageDimension2D`
  (spec 15.2) 검사.
- `es bench --memory-report --obs <obs.toml> [--learning <learning.toml>]
  [--scene <mjcf|urdf>] --sim-envs N --obs-envs N --views N
  --inference-batch N [--precision f16|f32] [--device-gib G] [--tile-w N
  --tile-h N --tiles-per-row N]` — 테이블과 위반 사항을 출력한다; 위반 시
  exit 1, 사용법 오류 시 exit 2, 그 외에는 0. `--scene`은 best-effort다:
  사용할 수 없거나 파싱할 수 없는 backend는 명령을 실패시키는 대신 메모를
  출력하고 `physics_state`를 `unavailable`로 남긴다.
- `es bench`(`--memory-report` 없이)는 `es_telemetry::PerfMetrics::default()`를
  통해 spec 12.4의 9개 지표 테이블을 출력한다 — 모든 필드가 `unmeasured`
  — 그리고 한 줄: `Target / Status: 미검증`(측정 루프는 별도의 M2 W5
  하위 패킷이다). 새 지표 타입은 정의하지 않는다: `es-compile`(layer 7)은
  그것이 필요 없고, CLI(layer 11)는 이미 `es-telemetry`(layer 10)를 쓸 수
  있다.
- `ModelSizes { nq, nv, nu, nsensordata }`는
  `es_physics_core::backend::ModelInfo`의 필드를 미러링하지만 `es-compile`에
  로컬로 정의된다: `u32` 네 개를 위해 `es-compile`(layer 7)에
  `es-physics-core`(layer 3) 의존성을 추가하는 것은 이 패킷이 선언한 파일
  범위를 벗어나며, 둘 다에 이미 의존하는 CLI가 호출 지점에서 그것들을
  복사해 넘긴다.
- `BudgetDomains { n_sim_envs, n_obs_envs, n_views, inference_batch }`는
  `es_env::scheduler::BatchDomains`와 같은 모양이지만 같은 계층 이유로
  로컬에 정의된다(`es-env`는 `es-compile` 위의 layer 9다).

## oracle (오라클)

```
cargo fmt -p es-compile -p es --check
cargo clippy -p es-compile -p es --all-targets -- -D warnings
cargo test -p es-compile -p es
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `budget.rs` 단위 테스트: 손으로 만든
  `ObservationIr`/`LearningGraph`/`ModelSizes` fixture에서 모든 바이트
  수를(단지 "> 0"이 아니라) 정확히 단언한다; `render_tile_atlas`와
  `policy_weights`는 그 fixture에서(카메라 노드도 가중치 크기도 없으므로)
  `unavailable`(bytes `0`, formula가 `"unavailable"`로 시작)로 단언된다;
  spec 20.3의 두 규칙 모두 위반으로 실행된다; 모든 항목의 formula
  문자열이 비어 있지 않음을 단언한다; `Display`가 `"GiB"`, `"total"`,
  `"per domain:"`을 포함함을 단언한다.
- `crates/es/tests/cli.rs`: `es bench`(플래그 없이)는 spec 12.4의 9개
  지표 이름 전부와 `unmeasured`, `Target / Status: 미검증` 줄을 담고
  있다. 기존 cross-IR fixture의 Observation/Learning IR
  (`write_fixture_toml`을 통해 TOML로 기록, spec 5.1의 `es_ir::serial`)에
  대한 `es bench --memory-report`는 모든 예산 항목 이름, `GiB`,
  `per domain:`을 출력하며, 균형 잡힌 `--obs-envs <= --sim-envs`에서는
  위반 없이 exit 0이다; 같은 fixture에 `--obs-envs > --sim-envs`를 주면
  exit 1이며 `obs_batch_le_sim_batch`를 명시한다; `--obs` 없이
  `--memory-report`를 주면 사용법 오류다(exit 2).

## forbidden (금지)

- `crates/es-env/**`, `crates/es-eval/**`, `crates/es-policy/**`,
  `crates/es-data/**` — 다른 에이전트들의 동시 진행 중인 M2 패킷.
- `WeightsRef`에 대한 어떤 변경(바이트 크기 필드 추가): `es-ir` 범위의
  패킷이 스키마를 넓히기로 결정하기 전까지 `policy_weights`는
  `unavailable`로 남는다.
- 9개 `PerfMetrics` 필드 배후의 실시간 측정 루프, `N_obs`/`N_inf` 자동
  축소 루프, 에디터의 컴파일 타임 예산 초과 에러(spec 20.3) — 모두 실행
  중인 backend/GPU나 `es-editor`(layer 12)를 필요로 하며 별도 패킷이다.
- 루트 `Cargo.toml`.
