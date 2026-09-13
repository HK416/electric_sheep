<!-- Korean translation of docs/packets/M2/CLI-eval-run-import.md. The English file is the working copy; regenerate this when it changes. -->

# CLI-eval-run-import — `es eval run` and `es import lerobot-config`

Spec: §10.5 (`es eval run --config ... --policy policy.esb` -> `report.json`, `report.html`,
`episodes/`, `evaluation.lock`), §14.4 (외부 변환, LeRobot config가 우선 대상), §1.4
(오라클 우선: 이 머신에서 실제로 실행할 수 없는 평가는 조작하지 않고 거부한다), §9.6
(`policy.esb`), §5.3 (해시 체인).

## context (범위)

```
crates/es/src/cmd/eval.rs      (+ `run` subcommand, alongside the existing `compare`)
crates/es/src/cmd/import.rs    (new: `es import lerobot-config`)
crates/es/src/cmd/mod.rs       (+ `pub mod import;`)
crates/es/src/cmd/backend.rs   (`load_scene` made `pub(crate)`, reused by `eval run`)
crates/es/src/main.rs          (+ `import` dispatch, usage text)
crates/es/Cargo.toml           (+ `es-eval`, `es-env`, `es-safety`, `es-policy` deps)
crates/es/tests/cli.rs
docs/packets/M2/CLI-eval-run-import.md
```

## spec (사양)

1. `es eval run --config <eval.toml> --policy <policy.esb> --scene <file.xml|urdf> [--out
   <dir>] [--backend mujoco-cpu] [--runtime torch]`:
   - 번들을 열고 (`es_compile::PolicyBundle::open`, spec 9.6 — 모든 IR과 해시를 다시 검증한다)
     `--config`로 Evaluation IR을 파싱한다 (`es_ir::serial::evaluation_from_toml`).
   - 씬 파일을 건드리기 *전에* `MuJoCoCpuBackend::is_available()`과
     `es_policy::torch_runtime::is_available()`을 확인한다. 둘 중 하나라도 사용 불가하면
     `SKIPPED (<reason>)`을 출력하고 종료 코드 **3**으로 끝난다 — usage(2), failure(1)와
     구별되는 코드인 이유는 아무것도 실행되지 않았기 때문이다.
   - 그렇지 않으면 씬을 로드하고, 정책을 로드하며 (`TorchRuntime::load`를
     `WeightsSource::InMemory(bundle.weights)`에 대해 실행), `es_eval::Evaluation::run`을
     호출한다. 이 함수의 `NJ`/`H` const 제네릭은 `bundle.deployment.{robot.n_joints,
     action.horizon}`으로부터 작은 고정 디스패치 테이블을 통해 런타임에 결정된다
     (ponytail: 런타임 제네릭 솔버가 아니라 열거된 쌍이다 — 새 로봇/호라이즌 조합이
     필요하면 쌍을 추가한다); 목록에 없는 조합은 패닉이 아니라 `CliError::Runtime`이다.
   - `es_eval::write_artifacts`로 `report.json`과 `evaluation.lock`을 쓰고, `report.html` —
     동일한 데이터 위에 손으로 작성하고 이스케이프한 HTML 테이블이며 템플릿 크레이트는
     쓰지 않는다 — 도 쓴다. **이 빌드는 `episodes/`를 생성하지 않는다** (렌더러가 아직
     없으며, `--help`와 이 패킷 모두 그 사실을 명시한다).
   - 모든 `AcceptanceResult`가 `Determined { passed: true }`이면 종료 코드 0; 하나라도
     `Determined { passed: false }` 또는 `Unavailable`이면 (둘 다 출력하고) 종료 코드 1.

2. `es import lerobot-config --config <config.json> [--stats <stats.json>] [--dataset <root>]
   --out <dir>`: `es_data::lerobot_config::convert`를 실행하고, `es_ir::serial`을 통해
   `<out>/observation.toml`과 `<out>/learning.toml`을 쓰며, 모든 변환 경고와 두 콘텐츠 해시
   (`observation_hash`, `learning_hash`)를 출력한다. `ConfigError`(잘못된 JSON, 지원하지 않는
   정책 타입, 누락된 feature) 또는 I/O 오류 시 종료 코드 1; usage 오류 시 2.

## oracle (오라클)

```
cargo fmt -p es --check
cargo clippy -p es --all-targets -- -D warnings
cargo test -p es
```

## acceptance (수용 기준)

- `PolicyBundle::build`로 테스트 내에서 빌드한 번들(M1 `crates/es/tests/cli.rs`의 cross-IR
  fixture, 가중치는 `crates/es-runtime-embedded/tests/embedded.rs`의 `deployable()`이 하는
  것과 정확히 같은 방식으로 손으로 작성한 바이트 블롭을 가리키도록 재설정)에 대해 `eval
  run`을 실행하면, `mujoco`/`torch`가 없을 때 (CI의 PR job에서는 항상 참이다, spec 1.4)
  종료 코드 **3**으로 끝나고 `SKIPPED`를 출력한다 — 백엔드/런타임 확인은 (존재하지 않는)
  `--scene` 파일을 읽기 전에 실행된다.
- `tests/fixtures/lerobot_config/act_config.json`에 대해 `import lerobot-config`를 실행하면
  `es ir validate`를 통해 깨끗하게 (`ERROR` 없이) 왕복하는 `observation.toml` /
  `learning.toml`을 쓰고, 두 해시를 모두 출력한다.

## forbidden (금지)

- 다른 크레이트의 `src/` (`es-eval`, `es-compile`, `es-policy`, `es-physics-backend`,
  `es-data`, `es-ir`는 소비될 뿐 수정되지 않는다).
- `episodes/` 리플레이 — 아직 렌더러가 존재하지 않으며, 이 패킷은 그 공백을 문서화할 뿐
  메우지 않는다.
- `HashMap`/`HashSet` — `BTreeMap`만 사용한다.
- 새로운 확장 지점 trait (INV-17); 이 패킷은 추가하지 않는다.
- `report.html`을 위한 템플릿 엔진 의존성 — 대신 손으로 작성하고 이스케이프한다.
- 커밋 — 오라클은 이 패킷에 의해 실행되고 보고될 뿐, 반영되지 않는다.
