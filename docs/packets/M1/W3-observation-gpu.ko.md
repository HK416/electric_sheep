<!-- Korean translation of docs/packets/M1/W3-observation-gpu.md. The English file is the working copy; regenerate this when it changes. -->

# W3 — Observation IR → GPU 로워링 (2부)

Spec: 사양 7.6, 사양 7.7, 사양 11.4, 사양 11.5, 사양 2.3, 사양 3.4, 사양 3.5, 사양 26.2,
사양 28.3. 설계 노트: `docs/design/observation-lowering.md` §11 (리뷰 클래스 C — 수치가
바로 그 결과물이다). 1부는 `docs/packets/M1/W3-observation-cpu-ref.md`이며, 그 `CpuPlan`이
이 패킷의 오라클이다.

## context (범위)

```
docs/design/observation-lowering.md      (§11 filled in; .ko.md mirror too)
docs/packets/M1/W3-observation-gpu.md    (new)
crates/es-compile/Cargo.toml             (adds `es-gpu`: layer 2 under layer 7, rule 1)
crates/es-compile/src/lib.rs             (adds `pub mod gpu;` + re-exports)
crates/es-compile/src/gpu/mod.rs         (new — the lowering, no device)
crates/es-compile/src/gpu/plan.rs        (new — compile, pipelines, compiler_hash)
crates/es-compile/src/gpu/exec.rs        (new — upload, dispatch, download)
crates/es-compile/slang/observation.slang (new — one entry point per kernel id)
crates/es-compile/tests/observation_gpu.rs (new)
crates/es-compile/src/exec.rs            (`read_input`, `width`: private -> pub(crate))
```

## forbidden (금지)

`budget.rs`, `bundle.rs`, `kernels.rs`, `plan.rs` — 1부와 번들 패킷이 소유. `exec.rs` 변경은
위의 두 `pub(crate)` 가시성 확장에 한정된다: GPU 경로는 불일치하는 입력을 CPU 경로와 *같은*
`ExecError`로 거부해야 하며, 그 검사의 두 번째 사본은 두 번째 계약이 되어버린다. `KERNEL_IDS`는
추가·재정렬·개명되지 않는다 — 새 수치 동작은 새 id이며 그것은 1부의 테이블이다.
`tests/golden/**`는 CI 읽기 전용이다. `es-render`, `es-usd`, `es-script`, `es-data`, `es-eval`,
`es-py`, `crates/es`는 이웃 패킷 소유다.

## spec (사양)

- `GpuPlan::compile(&Gpu, &ObservationIr, PlanMode) -> Result<GpuPlan, Vec<Diagnostic>>`는
  `CpuPlan::compile`을 호출하고 그것을 **거울상**으로 반영한다 — 같은 위상 순서, 같은 버퍼
  테이블, 노드당 같은 커널. 이것은 두 번째 컴파일러가 아니다; 추가하는 것은 오직 바이트가 어디
  사는지와 어떤 파이프라인이 실행되는지뿐이다. 디바이스나 `slangc`가 거부하는 것은 무엇이든
  `COMPILE-006`이다.
- 계획 버퍼당 버퍼가 아니라 디바이스 아레나 하나. 다섯 개의 스토리지 바인딩이 모든 커널을
  서비스한다: `arena`(f32 중간값, 그다음 f32 계획 입력), `words`(u8 계획 입력, 그다음
  f16 / bf16 출력 비트 패턴), `aux`(sRGB LUT, 그다음 각 `Normalize` 단계의 mean/std),
  `rings`, `state`(윈도우별 `cursor`/`pushed`). 디스크립터 레이아웃 하나가 계획 전체를
  커버하므로 모든 버퍼 오프셋이 Slang define이 될 수 있다.
- `(kernel id, Slang entry, defines)`당 `ComputePipeline` 하나. `KERNEL_IDS`는 계속 커널이
  *무엇인지*에 대한 근원이며, define(dtype, 채널, 크기, 오프셋)은 사양 2.3의 특수화다.
  `GpuPlan::pipeline_plan`은 디바이스 없이 그 목록을 노출하므로, Vulkan이 없는 CI도 이 결정을
  게이트할 수 있다.
- `GpuPlan::run(&mut self, &BTreeMap<String, TensorRef>) -> Result<Outputs, GpuRunError>`:
  업로드, 하나의 큐에 모든 디스패치를 배리어를 사이에 두고 기록, 제출 한 번, 다운로드.
  `&mut self`인 이유는 링이 스트림이기 때문이다(사양 7.5 layer 1).
- `GpuPlan::reset()`은 `CpuPlan::reset`의 디바이스 거울상이다: 링을 0으로, 커서를 0으로 — 정확히
  `compile`이 남기는 상태로. `es-eval`은 매 `env.reset` 뒤에 이것을 호출한다.
- `GpuPlan::compiler_hash()` = CPU 계획의 해시 더하기 모든 파이프라인의 SPIR-V 콘텐츠 해시이며,
  그래서 `.slang` 파일을 편집하면 커널 id가 바뀌지 않아도 `execution_hash`의 `compiler` 슬롯이
  움직인다(사양 3.4 항목 7, 사양 11.2).
- 결정론(사양 3.4): 실행 모드는 `Capabilities::deterministic_execution_modes()`에서 온다 —
  디바이스 설정이 아니라 능력 질의에서 읽은 컴파일 입력이다 — 그리고 `es_gpu::apply_exec_modes`가
  덧붙이는 `NoContraction`. 단일 큐, 실행당 제출 한 번, 원자적 연산 없음, 공유 메모리 없음,
  서브그룹 연산 없음, 워크그룹 개수 의존 없음. `es_math::approx`(`approx.slang`)가 유일한
  초월함수.

## oracle (오라클)

```
cargo test -p es-compile -- --nocapture
cargo xtask layering && cargo xtask verify-goldens && cargo xtask context-budget
```

Vulkan 디바이스나 `slangc`가 없으면 모든 디바이스 테스트가 `SKIP <reason>`을 출력하고
통과하며(사양 1.4), 있으면 디바이스 이름과 함께 `RAN`을 출력한다. CPU/GPU 동치 수치는 단언될
뿐 아니라 출력되므로, 회귀가 숫자로 드러난다.

## acceptance (수용 기준)

- `tests/golden/observation/`의 모든 골든이 `GpuPlan`으로 재현되며, `observation_cpu.rs`가
  쓰는 것과 *같은* 비교 아래서다: 바이트 단위, 다만 `srgb_to_linear_lut256`은 그 사이드카가
  선언하는 7 ULP까지 예외다. 측정값: `dequantize`, `resize_bilinear`, `crop`, `normalize`,
  `history_window`는 비트 동일; LUT는 7 ULP — 이것은 CPU 커널 자신이 f64 레퍼런스 공식과
  갖는 거리이지, 그 위에 얹힌 GPU 오차가 아니다. GPU가 CPU가 업로드한 테이블 바이트를 그대로
  인덱싱하기 때문이다.
- 17×17까지의 의사 무작위 `(source, target)` 크기 쌍 50개: `CpuPlan::run` vs `GpuPlan::run`을
  같은 그래프에서, **비트 단위로**. 측정 50/50, 최악 0 ULP. 불일치는 크기와 ULP 거리만
  실패시키는 대신 함께 출력한다.
- 하나의 계획에 대한 하나의 입력으로 두 번 `run`한 결과는 비트 동일(사양 3.5 tier 1).
- `reset(); run(x)`는 새로 컴파일한 계획의 첫 `run(x)`와 같고, CPU 계획의 것과도 같다.
  히스토리에 민감한 픽스처에서(단언되지 않으면 테스트는 아무것도 증명하지 못한다).
- 어떤 골든도 덮지 않는 커널에 대한 CPU vs GPU 비트 동일성: `resize_nearest`, 원소별
  `srgb_to_linear`(0 ULP), `concat`, `stack`, `normalize_range`, `cast_f32_to_f16`,
  `cast_f32_to_bf16`.
- GPU 없이: 각 커널을 한 번씩 쓰는 체인의 파이프라인 목록은 서로 다른 커널 id당 파이프라인
  하나이며, 같은 크기로의 두 리사이즈는 파이프라인을 공유하고 다른 크기로의 두 리사이즈는
  공유하지 않는다.

## known limits (알려진 한계)

- **퓨전 없음.** 사양 11.4의 원소별 퓨전은 구현되지 않는다; CPU 경로에서와 마찬가지로 두 계획
  모드 모두에서 모든 노드가 자신만의 디스패치다. 릴리스 모드 퓨전과 아레나 앨리어싱은 별도
  패킷이며, `PlanMode`가 `compiler_hash`에 도달하는 것이 바로 그것을 보호한다.
- **비정규수.** 결정론적 실행 모드는 디바이스에서 비정규수를 0으로 플러시하지만, CPU 커널은
  그러지 않는다. 비정규 중간값이 둘이 불일치할 수 있는 유일한 지점이다.
- `f16`/`bf16` 결과는 `uint` 영역에 비트 패턴으로 쓰인다: 디바이스는 16비트 스토리지로 열리지
  않는다. 좁혀진 버퍼를 디바이스에서 다시 f32로 넓히는 것은 지원되지 않는다 — 아레나가 f32를
  유지하므로 아직 아무도 그것을 필요로 하지 않는다.
- `slangc`의 스크래치 파일은 `{pid}-{invocations}`로 이름 붙으며 각 `SlangCompiler`는 그
  카운터를 0에서 시작하므로, 같은 캐시 키를 컴파일하는 한 프로세스의 두 스레드가 경합한다.
  GPU 테스트는 그것을 직렬화하여 우회한다; 수정은 `es-gpu`에 속한다.
