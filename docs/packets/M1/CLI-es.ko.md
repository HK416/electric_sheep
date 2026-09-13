<!-- Korean translation of docs/packets/M1/CLI-es.md. The English file is the working copy; regenerate this when it changes. -->

# CLI-es — `es` 런타임/CLI

명세: §2.5 (`es --check-deps`, 외부 도구 의존성 표), §5.3 (해시 체인), §10.5 (Evaluation IR
산출물, `es eval compare`), §11.1 (통합 컴파일러 파이프라인, Cross-IR Check), §17.2
(`es backend compare`, 이 패킷의 범위 밖), §9.6 (`policy.esb`, 이 패킷의 범위 밖), §26.2
(CI 커맨드 표면).

## context (범위)

```
crates/es/Cargo.toml
crates/es/src/main.rs
crates/es/src/error.rs
crates/es/src/util.rs
crates/es/src/cmd/mod.rs
crates/es/src/cmd/check_deps.rs
crates/es/src/cmd/ir.rs
crates/es/src/cmd/task.rs
crates/es/src/cmd/eval.rs
crates/es/src/cmd/dataset.rs
crates/es/tests/cli.rs
docs/packets/M1/CLI-es.md
```

## spec (사양)

새 바이너리 크레이트 `es`는 언어 바인딩 레이어(§4.2 표의 레이어 11 -- 레이어 10 이하의
무엇에든 의존할 수 있고, 그 무엇도 이것에 되의존할 수 없다)에 있다. `clap`은 쓰지 않는다:
`xtask`가 이미 세운 관례를 따라 손으로 짠 `std::env::args` 디스패처를 쓴다.
`CliError { Usage(String), Runtime(String) }`가 모든 서브커맨드가 반환하는 유일한 에러
타입이다; usage 에러는 종료 코드 2, 런타임 실패나 `Severity::Error` 진단은 1, 그 외는 0이다.
사용자 입력(파일 경로, 파일 내용, CLI 인자)에는 `unwrap`을 쓰지 않는다.

- **`es --check-deps`** (§2.5) -- 절대 실패하지 않는다. 프로브하는 것: Python
  인터프리터(`python`, `python3`, 또는 `$ES_PYTHON`; `es-physics-backend::proc::
  python_candidates`를 그대로 반영), `mujoco` import 가능 여부(자체 인터프리터 탐색을 이미
  하는 `MuJoCoCpuBackend::is_available`에 위임), `torch`와 `lerobot` import 가능 여부
  (`python -c "import ..."`를, 멈춘 인터프리터가 CLI를 정지시키지 못하도록 `try_wait`을
  폴링하는 10초 타임아웃으로 실행), Vulkan 로더(Unix에서는 통상적인 lib 디렉터리 아래의
  `libvulkan.so.1`, `%PATH%`/System32의 `vulkan-1.dll` -- 파일 존재 여부만 보는 휴리스틱이며
  출력에서도 그렇게 표시되고 결코 드라이버 프로브가 아니다). §2.5의 어떤 능력이 뒤따르는지를
  보고한다: PyTorch 학습 경로, MuJoCo(CPU) 오라클, LeRobot 데이터셋 export(데이터셋 *읽기*는
  Rust 네이티브이며 항상 가능), Slang/Vulkan GPU lowering(캐시 히트 시에는 이 중 아무것도
  필요 없다는 점도 함께 언급).
- **`es ir validate <file.toml>...`** -- 파일의 envelope에서 `IrKind`를 감지하고
  (`es_ir::serial`: 어떤 타입의 `*_from_toml`을 호출할지 정하기 전에 `toml::Value`로 `kind`
  필드를 미리 들여다본다), 해당 IR 자신의 `validate()`를 실행하고, 각 `Diagnostic`을 명세의
  블록 포맷 `Display`로 출력한 뒤, 그 IR 자신의 `*_hash`를 소문자 16진수로 출력한다.
- **`es ir check <task> <obs> <learning> <deploy> [eval]`** -- 다섯 파일을 위치 인자로
  파싱하고(각자의 `*_from_toml`이 이미 `KindMismatch`를 거부한다), `IrBundle`을 만들고,
  `es_ir::cross::check`를 실행하고, 진단을 출력한 뒤, 저작 시점에 알 수 있는 §5.3이 정의하는
  해시 체인 슬롯을 모두 출력한다: task, observation, learning, policy
  (`LearningGraph::policy_hash`), deployment, evaluation(파일이 생략되면 `unset`),
  compiler(`compiler_hash`를 결과에서 읽어내기 위해서만 observation IR을 `PlanMode::Debug`로
  `CpuPlan::compile`한다 -- 이 해시는 크레이트 버전, plan mode, 커널 표에만 의존하고 IR
  내용에는 의존하지 않지만, 그 메서드는 컴파일된 `CpuPlan` 위에 있다).
  `dataset`/`runtime`/`hardware`는 항상 `unset`이다: 이는 런타임에만 알 수 있고 저작
  시점에는 결코 알 수 없다.
- **`es task compile <task> <obs> [--release]`** -- Task IR을 구조적으로 검증하고
  (`--help`는 task *실행기*가 여기가 아니라 `es-env`에 있다고 분명히 밝힌다)
  `CpuPlan::compile`로 Observation IR을 컴파일하며, 노드 실행 순서, 버퍼 표(dtype, shape,
  home, 바이트 크기), 전체 arena 크기, `compiler_hash`를 출력한다.
- **`es eval compare <A.json> <B.json>`** -- 두 파일 모두 `es_ir::evaluation::
  EvaluationReport` JSON이다. suite별/metric별 표를 출력한다: A의 값, B의 값, delta, 그리고
  유의성 열. `EvaluationReport`(spec 10.5)는 셀당 집계값(`MetricValue::Scalar`/
  `::Histogram`)만 실을 뿐 원시 에피소드별 샘플은 결코 싣지 않으므로, 이 커맨드는 각 파일을
  범용 `serde_json::Value`로 다시 파싱해 비표준 확장인 `cells[i].samples: [f64, ...]`을
  찾는다; 일치하는 `(suite, metric)` 셀에 대해 *양쪽 모두* 이를 갖고 있을 때만 양측 Welch
  t-검정 p-value를 계산하고(순수 Rust로: Welch-Satterthwaite 자유도, 이어서 Lentz의
  연분수를 통한 regularized incomplete beta와 Lanczos `ln_gamma` -- 통계 크레이트 없음)
  `|p| < 0.05`를 표시한다. 그렇지 않으면 그 열은 `n/a (aggregate-only report)`로 표시된다.
- **`es dataset info <root>`** -- `LeRobotDataset`을 열어, 모든 feature를 그것이 Observation
  IR 포트에서 나타내는 `PortType`으로 출력하고(`FeatureSpec::port_type`, 해당하는 것이 없는
  `string` feature는 `Err`), 에피소드 개수와, 표시 전용의 all-train `Split`(`dataset info`에는
  식별에 쓸 호출자 지정 split이 없다) 위에서 계산한 세 개의 `DatasetIdentity` 해시
  (content/schema/split)를 출력한다.
- 모든 서브커맨드에 대한 `es --version`, `es --help`, `es <subcommand> --help`.

## oracle (오라클)

```
cargo fmt -p es --check
cargo clippy -p es --all-targets -- -D warnings
cargo test -p es
cargo xtask layering
cargo xtask context-budget
```

## acceptance (수용 기준)

- `--check-deps`는 Python도 Vulkan 로더도 없는 환경을 포함해 모든 환경에서 0으로 종료하며,
  Vulkan 검사를 휴리스틱으로 올바르게 표시한다.
- `ir validate`는 다섯 종류의 IR 각각에 대해 올바른 `validate()`를 찾아 일치하는 `*_hash`를
  출력한다; 알 수 없는 `kind`나 유효한 TOML이 아닌 파일은 패닉이 아니라 런타임 에러(종료 코드
  1)다.
- `crates/es-ir/tests/cross_fixture.rs`의 `Fixture::new()`가 만드는 방식대로 상호참조되고
  완전히 유효한 번들에 대해 `ir check`는 진단 0건을 내며,
  task/observation/learning/policy/deployment/compiler 모두를 실제 해시로, `evaluation`은
  주어졌을 때는 해시로 생략됐을 때는 `unset`으로, dataset/runtime/hardware는 항상 `unset`으로
  낸다.
- 같은 픽스처의 task+observation 쌍에 대해 `task compile`은 성공하며, 그 `compiler_hash`는
  `CpuPlan::compile`을 직접 호출한 것과 일치한다.
- 손으로 작성한 두 `EvaluationReport` JSON 파일에 대해 `eval compare`는 서로 다른
  `(suite, metric)`마다 한 행씩 출력하고, 한쪽에만 있는 셀을 올바르게 표시하며, 양쪽 모두
  해당 셀에 `samples` 배열을 갖고 있을 때만 `p=` 유의성 값을 낸다.
- 픽스처 LeRobot 데이터셋에 대해 `dataset info`는 `string`이 아닌 모든 feature의 `PortType`,
  올바른 에피소드 개수, 0이 아닌 세 개의 identity 해시를 나열한다.
- 어떤 서브커맨드의 구현도 CLI 인자, 파일 내용, 스폰된 프로세스의 출력에서 비롯된 데이터에
  `.unwrap()`/`.expect()`를 호출하지 않는다.
- `cargo xtask layering`은 `es` 크레이트에 대해 위반을 보고하지 않는다(LAYERS 표가
  게이팅하는 `es-*` 패턴에 맞지 않으므로 표 항목이 필요하지 않다; 어차피 바이너리이므로
  무엇도 이것에 되의존할 수 없다).
- `crates/es/**`는 소스 라인 기준 약 1,200줄 이하로 유지된다(테스트 제외).

## forbidden (금지)

- `crates/es-policy/**`와 `docs/ARCHITECTURE*.md` 아래의 모든 파일 -- 동시에 진행 중인 다른
  패킷의 소관.
- layering 검사가 실제로 `es`라는 이름의 크레이트를 거부하지 않는 한 `xtask/src/layering.rs`를
  수정하는 것; 현재는 거부하지 않으므로(`LAYERS`는 `es-`로 시작하는 이름만 게이팅한다) 이
  패킷은 이 파일을 건드리지 않는다.
- 루트 `Cargo.toml`을 수정하는 것 -- `crates/*` glob이 이미 `crates/es`를 포함하며, 여기서의
  모든 의존성은 `{ workspace = true }`다.
- `clap`이나 다른 어떤 CLI 인자 파싱 크레이트.
- `crates/es-ir/tests/cross_fixture.rs`를 수정하는 것 -- `crates/es/tests/cli.rs`는 원본을
  바꾸는 대신 필요한 픽스처 구성 함수들(위반 시나리오 테스트는 제외)을 복사해 쓴다.
- 커밋.
