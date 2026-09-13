<!-- Korean translation of docs/packets/M1/W9-deployment.md. The English file is the working copy; regenerate this when it changes. -->

# W9 — `policy.esb`와 `es-runtime-embedded`

사양: spec 9.5, spec 9.6, spec 10.5, spec 5.3, spec 8.5, spec 8.6, spec 25.1, spec 25.3,
spec 27.1, spec 4.2. 설계 노트: `docs/design/policy-bundle.md` (먼저 작성됨; 컨테이너
레이아웃과 매니페스트 계약이 산출물이며, 코드는 그 하위에 있다).

M1의 배포 절반: 전처리, 가중치, 안전 제약을 함께 담는 산출물 하나 (spec 9.6), 그리고
시뮬레이션이 실행하는 것과 동일한 코드로 이를 실행하는 최소 런타임 (spec 9.5).

## context (범위)

```
docs/design/policy-bundle.md                      (new, written first)
docs/packets/M1/W9-deployment.md                  (new)
crates/es-compile/src/bundle.rs                   (new)
crates/es-compile/src/lib.rs                      (one `pub mod`, one re-export block)
crates/es-compile/Cargo.toml                      (blake3)
crates/es-runtime-embedded/Cargo.toml             (new crate)
crates/es-runtime-embedded/src/lib.rs             (new)
crates/es-runtime-embedded/src/runtime.rs         (new)
crates/es-runtime-embedded/src/ring.rs            (new)
crates/es-runtime-embedded/src/hardware.rs        (new)
crates/es-runtime-embedded/tests/embedded.rs      (new)
Cargo.toml                                        (one workspace-dependency line)
xtask/src/layering.rs                             (one LAYERS row)
```

## spec (사양)

- **`.esb` 컨테이너** — `ESB1` 매직, `u32` 엔트리 개수, 이름순으로 정렬된 엔트리별
  `{ name_len: u32, name, len: u64, blake3: [u8; 32] }`, 그 뒤에 같은 순서의 페이로드들,
  전부 리틀 엔디안. `write(&BundleManifest, &BTreeMap<String, Vec<u8>>) -> Vec<u8>`와
  `read(&[u8]) -> Result<Bundle, BundleError>`가 있으며, 후자는 모든 페이로드 해시를
  검증하고 중복되거나 순서가 어긋난 이름을 거부한다. 아카이브 크레이트도 압축도 없음:
  설계 노트 2절 참고.
- **`BundleManifest`** — `schema_version`, `kind: Policy | Evidence`, `hashes` (슬롯별
  `Option<[u8; 32]>`로 표현되는 spec 5.3 체인, TOML에서는 16진수, 그리고
  `Option<DatasetHash>`), `created_utc: Option<String>`, `signature: Option<Vec<u8>>`
  (예약됨, spec 25.1 — 아무것도 서명하지 않고 아무것도 검증하지 않는다). `manifest.toml`
  엔트리로 저장된다. 슬롯이 비어 있다는 것은 "주장되지 않음(not claimed)"을 의미할 뿐
  "0으로 주장됨(claimed as zero)"을 의미하지 않는다.
- **`PolicyBundle::build(task, observation, learning, deployment, weights) -> Vec<u8>`** —
  각 IR을 검증하고, `es_ir::cross::check`를 실행하며, `blake3(weights)`를
  `WeightsRef::hash`와 대조 검증하고, `compiler` 슬롯을 위해 observation plan을
  컴파일하며, `task/observation/learning/policy/deployment/compiler`를 채운다. `runtime`과
  `dataset`은 비어 있는 채로 남는다.
- **`PolicyBundle::open(&[u8]) -> Result<PolicyBundle, BundleError>`** — 네 개의 IR을
  다시 파싱하고 다시 검증하며, cross-IR 패스를 다시 실행하고, 모든 해시를 재계산하여
  일치하지 않으면 해당 슬롯 이름을 담은 `HashMismatch { slot }`로 실패한다. 매니페스트는
  증거가 아니라 주장(claim)이다. `compile_plan()`이 (`Serialize`가 아닌) `CpuPlan`을
  다시 빌드한다; 이것이 안전하다는 근거가 `compiler` 해시다. 배포 번들은 `PlanMode::Release`로
  컴파일된다.
- **`EmbeddedRuntime<NJ, H>`** — `from_bundle(bytes, Box<dyn PolicyRuntime>)`가 유일한
  생성자다: 번들을 열고, 플랜을 컴파일하며, (Deployment IR에 대해 `NJ`와 `H`를 강제하는)
  `SafetyPlane::from_ir`을 빌드하고, 가중치를 `WeightsSource::InMemory`로 로드한다.
  `tick(&sensors, now, obs_age) -> SafeAction<NJ>`는 plan -> `infer` -> `ActionChunk` ->
  `SafetyPlane::validate` 순으로 실행하고 `TickRecord`를 하나 밀어 넣는다. 리플랜 주기는
  `execute_chunk`로 상한이 걸린 `rate.control / rate.inference` 컨트롤 틱이다 (spec 8.6);
  리플랜 사이에는 버퍼링된 청크가 다시 제출된다. 어떤 실패든 빈 청크가 되며, plane이
  이를 청크 언더런과 설정된 폴백으로 전환한다 — `tick`에는 에러 반환이 없다 (`INV-13`).
- **`execution_hash()`** — 매니페스트로부터의 spec 5.3 체인에, 로드된 백엔드의
  `runtime_hash()`와 여기서
  `blake3(tag || arch || os || family || pointer width || [feature, present]*)`로
  계산되는 `HardwareCapability`가 더해진 것이며, 기능 프로브 목록은 고정되어
  `hardware.rs`에 문서화되어 있다.
- **레이어링** — `es-runtime-embedded`는 `es-env`와 함께 레이어 9에 위치한다:
  `es-safety` / `es-policy` (8)와 `es-compile` (7)을 사용할 수 있다; 오직 `es-ros2` /
  `es-py` (11)만이 이를 의존할 수 있다. `es-telemetry` (10)는 사용할 수 **없으므로**,
  ~50줄짜리 `TelemetryRing`이 원본과 업그레이드 경로(`RingBuffer`를 `es-core`로 옮기기)를
  명시하는 주석과 함께 `es_telemetry::ring::RingBuffer`를 중복 구현한다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-compile -p es-runtime-embedded --all-targets -- -D warnings
cargo test -p es-compile -p es-runtime-embedded
cargo xtask layering
cargo xtask check-spec-refs
cargo xtask context-budget
```

## acceptance (수용 기준)

- 컨테이너: 라운드트립; **동일한 입력에 대해 바이트 단위로 동일한 출력**; 페이로드 바이트
  하나가 뒤집히면 `EntryHash { name }`가 된다; 알 수 없는 매직과 잘린(truncated) 파일은
  이름 붙은 에러가 된다; 매니페스트는 16진수 해시로 TOML을 라운드트립하며 `None` 필드는
  생략한다.
- `PolicyBundle`: `es-ir`의 cross-IR 픽스처가 빌드되고, 모든 해시가 일치한 채로 다시
  열린다; 매니페스트 해시를 바꿔치기하면 `HashMismatch { slot: "learning" }`로 실패한다;
  `WeightsRef::hash`와 일치하지 않는 가중치는 빌드 시점에 거부된다.
- `#[cfg(test)]`의 `FakeRuntime`을 사용한 픽스처 위의 `EmbeddedRuntime`: 정상적인(benign)
  청크는 `ActionSource::Policy`와 정상 액션을 내보낸다; NaN 청크는 유한한 액션과 함께
  `Fallback(HoldPosition)`을 내보낸다; 8개 관절용 번들은 `EmbeddedRuntime<3, H>`로
  로드되기를 거부한다; 리플랜 주기는 10 컨트롤 틱(100 Hz / 10 Hz)이며 35틱은 정확히
  4번의 추론을 만들어낸다; 청크 재사용 틱은 `es_core::alloc_count::assert_no_alloc`을
  통과한다; `execution_hash`는 프로세스 내에서 안정적이며 같은 번들을 두 번 로드해도
  동일하다.
- 워치독을 비활성화하는 대신 픽스처의 추론 예산(inference budget)을 넓힌다 (`INV-12`).
- `PolicyBundle` 수준의 테스트는 `es-runtime-embedded/tests/embedded.rs`에 있는데,
  cross-IR 픽스처 사본이 거기 있기 때문이다; `es-compile`은 픽스처가 필요 없는 컨테이너
  수준의 단위 테스트를 유지한다.

## forbidden (금지)

- `context` 밖의 어떤 파일도 안 된다. 특히 이번 웨이브에서 다른 에이전트 소유인
  `crates/es-physics-backend/src/mjwarp*`, `crates/es-sensor`, `crates/es-actuator`,
  `docs/reviews/M0.md`, `docs/ARCHITECTURE*.md`; 그리고 API를 소비만 하고 절대 수정하지
  않는 `crates/es-ir`, `crates/es-safety`, `crates/es-policy`.
- 아카이브 의존성 (`zip`, `tar`, `flate2`, `zstd`). 컨테이너는 ~40줄이며 압축은 신뢰
  경계에서의 공격 표면이다 (spec 25.1).
- 새로운 트레이트. `PolicyRuntime`은 `Box<dyn PolicyRuntime>`로 소비된다; `INV-17`의
  일곱 확장 지점이 전체 목록이다.
- `SafetyPlane` 없이 `EmbeddedRuntime`을 생성하는 경로, 또는 plane이 보지 못한 액션을
  반환하는 경로 (`INV-12`, `INV-13`). 여유가 필요한 테스트는 엔벨로프를 넓힌다.
- pickle 경로, 혹은 로드 시 코드를 실행하는 어떤 가중치 포맷도 안 된다 (`INV-16`).
- `HashMap`/`HashSet` (spec 3.4), 스레드, 전역 RNG, 런타임 내 파일시스템 접근, Python,
  물리, 렌더링 (spec 9.6의 제외 목록).
- `signature` 슬롯에 실제 암호화를 넣는 것. 이번 마일스톤에서는 예약된 필드일 뿐이며,
  절반만 검증된 서명은 아예 없는 것보다 나쁘다.
- 증거 번들(evidence bundle) 콘텐츠 (`safety_case/`, `validation/`, `training/`)와
  `es evidence verify` — spec 27.1은 이후 패킷이다. 이 패킷이 빚지고 있는 것은 포맷
  변경 없이 확장 가능한 컨테이너뿐이다.
