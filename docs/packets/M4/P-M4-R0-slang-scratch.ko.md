<!-- Korean translation of docs/packets/M4/P-M4-R0-slang-scratch.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R0 — `SlangCompiler` 스크래치 파일 경합

Spec: 사양 2.3, 사양 11.4(Slang → SPIR-V 콘텐츠 해시 캐시), 사양 3.4(결정론적 실행 계약 —
한 캐시 키에 대한 컴파일된 바이트는 누가 컴파일했든 동일하다). 세 건의 `es-compile` GPU
테스트 실패에 대한 후속 조치이며, `es-compile` 호출부가 아니라 `es-gpu`의 동시성 구멍으로
접수되었다. 리뷰 클래스 A.

## context (범위)

```
crates/es-gpu/src/slang.rs
crates/es-compile/tests/observation_gpu.rs
docs/packets/M4/P-M4-R0-slang-scratch.md
```

## spec (사양)

`SlangCompiler::compile`은 `slangc` 스크래치 입출력의 이름을 캐시 디렉터리 바로 아래
`{hash}.{pid}-{invocations}.slang` / `.raw.spv`로 지었다. `invocations`는 0부터 시작하는
인스턴스별 카운터이므로, 한 프로세스의 두 스레드나 — 혹은 한 프로세스 안의 두
`SlangCompiler` 인스턴스, 예를 들어 공유 컴파일러와 호출마다 새로 만드는 것 — 가 같은 캐시
키를 동시에 컴파일하면 같은 태그를 만들어 컴파일 도중 서로의 스크래치 파일을 지운다
(`os error 2`). `crates/es-compile/tests/observation_gpu.rs`는 프로세스 전역 `Mutex`로 모든
GPU 테스트를 직렬화해서 이 문제를 우회했다.

수정 내용:

- 프로세스 전역 `static COUNTER: AtomicU64`가 어느 `SlangCompiler` 인스턴스나 스레드가
  호출하든 상관없이 `compile` 호출마다 전역적으로 유일한 `n`을 준다.
- 각 호출은 자신만의 스크래치 디렉터리 `std::env::temp_dir().join(format!("es-slang-
  {pid}-{n}"))`를 받는다 — 캐시 디렉터리가 아니라 OS 임시 디렉터리이므로, 한 호출의 스크래치
  파일이 다른 호출의 것과 절대 인접하지 않는다 — 성공이든 실패든 드롭될 때 스스로
  `remove_dir_all`하는 RAII `ScratchDir` 가드를 통해서다. 내부 파일들은 가드가 강제 프로세스
  종료로 건너뛰어졌을 때도 읽을 수 있는 이름이 되도록 캐시 키 해시, `n`, 스레드 id로 추가로
  이름 붙는다.
- 캐시 항목은 `<hash>.spv.tmp-<n>`(호출마다 유일)에 쓰고 `<hash>.spv`로 `rename`하여
  발행된다. 같은 키를 두고 경합하는 두 호출은 둘 다 독립적으로 컴파일하고 둘 다 rename에
  도달한다; 결정론(사양 3.4)은 그 바이트가 동일함을 의미하므로, 나중에 도착하는 rename이
  같은 내용으로 먼저 것을 무해하게 덮어쓴다 — 캐시 디렉터리는 그 키에 대해 정확히 하나의
  `.spv`로 끝난다. 락도, 중복 제거 로직도 없다.
- `invocations`가 `Cell<usize>`에서 `AtomicUsize`로 옮겨졌으며, 이것이 `&SlangCompiler`를
  `Sync`로 만들어 인스턴스 하나가 여러 스레드에서 컴파일될 수 있게 한다(그러지 않았다면
  오라클의 "공유 컴파일러" 갈래는 애초에 컴파일되지 않았을 것이다).
- 같은 키의 동시 컴파일이 이제 안전해졌으므로 `es-compile`의 우회책
  (`observation_gpu.rs`의 `static DEVICE: Mutex<()>`와 `on_gpu`의 락 획득)이 제거된다;
  그 파일의 GPU 테스트는 다시 테스트 간 직렬화 없이 실행된다.

## oracle (오라클)

```
cargo test -p es-gpu --lib slang::tests::eight_concurrent_compiles_of_the_same_key_do_not_clobber_each_other
cargo test -p es-compile --test observation_gpu
```

첫 번째는 수정 전에 쓰인 오라클이다(옛 코드에서는 `os error 2`나 words 불일치로 실패한다):
8개의 스레드가 하나의 공유 `SlangCompiler`와 호출마다 새로 만든 인스턴스를 섞어서 같은 소스를
하나의 캐시 디렉터리에 대해 컴파일한다. 모든 컴파일이 성공하고, 반환된 모든
`SpirvModule::words`가 동일하며, 이후 캐시 디렉터리에 그 키에 대한 `.spv`가 정확히 하나
있음을 단언한다. `slangc`가 `PATH` / `ES_SLANGC`에 없으면 `SKIP`을 출력하고 반환한다(패닉
없이, 사양 1.4). 두 번째는 `es-compile` 우회책의 제거가 이 crate 자신의(이전에는 직렬화되던)
GPU 테스트에서 그 실패를 재도입하지 않았음을 확인한다.

## acceptance (수용 기준)

- `eight_concurrent_compiles_of_the_same_key_do_not_clobber_each_other`는 `slangc`가
  있을 때 통과하고(이 머신에서 검증됨, RTX 4060 있음) 없으면 깔끔하게 SKIP한다.
- `cargo test -p es-compile --test observation_gpu`는 `DEVICE` mutex와 그 문서 주석이
  제거된 채로 반복 실행해도(3회 확인) 흔들림 없이 통과한다.
- `cargo fmt -p es-gpu -p es-compile --check`와
  `cargo clippy -p es-gpu -p es-compile --all-targets -- -D warnings`가 깨끗하다.
- 새 의존성 없음. `SlangCompiler`의 공개 API(`new`, `with_include`, `version`,
  `invocations`, `cache_dir`, `compile`, `compile_file`)는 시그니처를 유지한다.

## forbidden (금지)

`context` 밖의 모든 파일, 특히 `crates/es-render/src/renderer.rs`의 `COMPILE_LOCK` — 같은
근본 원인에 대한 대응하는 우회책이지만 다른 crate의 파일이고 이 패킷의 범위 밖이다; 제거는
후속 패킷이다. `SlangCompiler` 자체에 락이나 뮤텍스를 추가하는 것(그건 직렬화된 컴파일을
다시 들여오는 것일 뿐이다). 다른 도구가 이미 의존하고 있을 수 있는 디스크상 캐시 파일 이름
스킴(`<hash>.spv`)을 바꾸는 것. `default_cache_dir`, `cache_key`, 또는
`crates/es-gpu/src/spirv.rs`의 무엇이든 건드리는 것.
