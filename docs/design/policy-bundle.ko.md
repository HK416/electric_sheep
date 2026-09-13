<!-- Korean translation of docs/design/policy-bundle.md. The English file is the working copy; regenerate this when it changes. -->

# `policy.esb` — 배포 번들 포맷

`es-compile::bundle`와 `es-runtime-embedded`를 위한 설계 노트. 사양: spec 9.5, spec 9.6,
spec 10.5, spec 5.3, spec 25.1, spec 25.3, spec 27.1.

## 1. 애초에 왜 번들인가

spec 9.6: **"정책 가중치만 배포하지 않는다 — 전처리와 안전 제약이 함께 배포된다."** 체크포인트
단독으로는 배포 가능한 산출물이 아니다: 같은 가중치라도 다르게 리사이즈된 픽셀을 입력받으면 다른
정책이 되고, 같은 정책이라도 다른 엔벨로프 아래에서는 spec 27.1 기준으로 다른 기계가 된다. 따라서
배포 산출물은 로봇의 동작을 결정하는 네 개의 IR과 체크포인트, 그리고 `execution_hash`를 재계산하는
데 필요한 spec 5.3의 모든 해시를 담은 파일 하나다.

spec 9.5가 하고자 하는 주장은 이것이다: *같은 `deployment_hash`는 같은 안전한 동작을 의미한다*.
이는 안전을 강제하는 것과 관측을 전처리하는 것이 가중치와 함께 배포되고, 로드 시점에 함께 검증될
때만 성립한다.

## 2. 컨테이너

`.esb`는 작은 tar 형태의 컨테이너다. 아카이브 크레이트도, 압축도 없다:

- 포맷 자체가 리틀 엔디안 정수 몇십 줄에 불과하므로 의존성을 추가해도 얻는 것이 없다;
- safetensors 페이로드는 압축되지 않는다;
- 압축 해제(inflate) 경로는 신뢰 경계를 넘나드는 산출물에서 공격 표면이 되며 (spec 25.1),
  압축 폭탄(decompression bomb)은 이 파일이 애초에 가질 수 없는 부류의 버그다.

```
offset  size          field
0       4             magic, ASCII "ESB1"
4       4             entry_count : u32

then entry_count headers, sorted by name, ascending, byte-wise:
        4             name_len : u32
        name_len      name     : UTF-8, no interior NUL, no path normalization
        8             len      : u64, payload length
        32            hash     : blake3 of the payload

then the payloads, concatenated, in the same order.
```

모든 정수는 리틀 엔디안이다. 엔트리별 오프셋 필드는 없다: 페이로드는 헤더 순서대로 배치되고
각각의 길이는 `len`이 결정하므로, 주어진 엔트리 집합에 대한 레이아웃은 유일하다.

**결정성.** 엔트리는 `BTreeMap`에서 나오므로 순서는 이름 순이며, 작성기(writer)의 다른 어떤
부분도 달라지지 않는다. 동일한 입력에 대한 두 번의 빌드는 바이트 단위로 동일하며, 이는 테스트
`policy_bundle_round_trips_and_is_byte_identical`이 검증한다. 이 때문에 기본적으로 빌드
타임스탬프가 기록되지 않는다: 매니페스트에 `created_utc`가 존재하기는 하지만
`PolicyBundle::build`는 이를 비워 둔다.

**읽기는 악의적 입력에 안전하다.** 모든 길이 값이 파일에서 나오므로, 리더는 범위를 벗어나
슬라이싱하는 대신 `Truncated`를 반환하는 커서다; 이름은 반드시 UTF-8이어야 한다; 중복되거나
순서가 어긋난 이름은 `Unsorted`가 된다; 그리고 모든 페이로드의 `blake3`가 반환되기 전에
검증된다. 비트 하나만 뒤집혀도 열기에 실패한다.

`ESB1`의 `1`은 매니페스트의 `schema_version`이 아니라 *컨테이너 세대(container generation)*다.
향후 호환되지 않는 레이아웃은 새로운 매직 값을 받아, 오래된 리더가 잘못 해석하는 대신 명확하게
실패하도록 한다 (spec 25.3: 번들 포맷은 버전이 매겨지며 이전 버전도 계속 읽을 수 있어야 한다).

## 3. 엔트리

| 이름 | 필수 | 내용 |
|---|---|---|
| `manifest.toml` | 예 | `BundleManifest`, 아래 참고 |
| `task.toml` | policy | Task IR (spec 6), `es_ir::serial` envelope |
| `observation.toml` | policy | Observation IR (spec 7) |
| `learning.toml` | policy | Learning IR (spec 8) |
| `deployment.toml` | policy | Deployment IR + Safety Plane 설정 (spec 9) |
| `weights.safetensors` | policy | 체크포인트 바이트, 그대로 |
| `evaluation.toml` | 아니오 | Evaluation IR (spec 10), 번들이 스위트도 함께 실을 때 |

리더는 이 이름들 중 어느 것도 알지 못한다; `read`는 모든 엔트리를 반환하고,
`PolicyBundle::open`이 필요한 것만 요청한다. 이 덕분에 **spec 27.1의 증거 번들이 동일한
컨테이너일 수 있다**: `kind = Evidence`에 `safety_case/*.json`, `validation/*.json`,
`training/`, `scene/`, 그리고 채워진 `dataset` 해시가 더해질 뿐이다. 포맷 변경도, 두 번째
리더도 필요 없다.

## 4. 매니페스트

```toml
schema_version = 1
kind = "policy"

[hashes]
task        = "<64 hex>"
observation = "<64 hex>"
learning    = "<64 hex>"
policy      = "<64 hex>"
deployment  = "<64 hex>"
compiler    = "<64 hex>"
# runtime, dataset: absent in a deployment bundle
```

해시가 TOML 정수 배열이 아니라 16진수 문자열인 이유는, 매니페스트가 번들에서 사람이 직접 읽는
유일한 부분이기 때문이다.

모든 슬롯은 `Option`이며, 슬롯이 비어 있다는 것은 **"주장되지 않음(not claimed)"**을 의미할
뿐 "0으로 주장됨(claimed as zero)"을 의미하지 않는다. 두 종류(kind)는 서로 다른 부분집합을
채운다:

- `runtime`은 `policy.esb`에는 없다: 추론 백엔드는 번들이 열리는 곳에서 선택되므로, 산출물
  자체는 이를 알 수 없다. `es-runtime-embedded`가 로드된 `PolicyRuntime::runtime_hash()`로부터
  이를 채운다.
- `dataset`은 `policy.esb`에는 없다: 학습(training)은 배포 경로에 있지 않다. 증거 번들이 이를
  채우며, 그때에만 `execution_hash`가 학습 시점의 값과 일치한다.
- `hardware_capability`는 애초에 매니페스트 필드가 아니다. 이는 산출물이 아니라 이를 실행하는
  기계를 서술한다.

`signature`는 예약된 `Option<Vec<u8>>` 슬롯이다 (spec 25.1은 선택적 서명 검증을 명시한다).
**현재는 아무것도 서명하지 않고 아무것도 검증하지 않으므로**, 리더는 `Some`을 신뢰의 근거로
취급해서는 안 된다. 서명 기능이 도입되면 서명 필드를 비운 채로 컨테이너 바이트 전체를 커버하게
된다.

## 5. `PolicyBundle::build` / `open`

`build`는 각 IR을 검증하고, `es_ir::cross::check`를 실행하며, `compiler` 슬롯을 위해
observation plan을 컴파일하고, 체크포인트를 `WeightsRef::hash`와 대조 검증한 뒤, 계산 가능한
여섯 개의 해시를 채운다.

`open`은 **같은 작업을 다시 수행하며 매니페스트의 어떤 내용도 신뢰하지 않는다**. 매니페스트는
산출물의 주장(claim)이고, IR들이 그 증거다. 재계산한 값이 일치하지 않는 슬롯은 해당 슬롯 이름을
담은 `BundleError::HashMismatch { slot }`가 되며, 이는 spec 27.1의 `revalidation_trigger`가
기술되는 단위이기도 하다.

### 컴파일된 플랜이 번들에 없는 이유

`CpuPlan`은 `Serialize`가 아니다 — 해석된 버퍼 위치, 아레나(arena) 레이아웃, 그리고 살아 있는
`TemporalWindow` 링 상태를 담고 있으며 — 이를 직렬화하면 컴파일러의 내부 표현이 배포 산출물
안에 그대로 얼어붙게 되는데, 이는 spec 25.3이 원하지 않는 바다. 그래서 번들은 Observation IR을
저장하고, `PolicyBundle::compile_plan`이 로드 시점에 플랜을 다시 빌드한다. 이것이 안전하다는
근거는 `compiler` 해시다: `CpuPlan::compiler_hash`는 크레이트 버전, 플랜 모드, 커널 id
테이블을 커버하므로, 다른 수치 결과를 만들어낼 재빌드는 실행되기 전에 열기에서 실패한다. 배포
번들은 항상 `PlanMode::Release` (`BUNDLE_PLAN_MODE`)로 컴파일되며, 이 모드는 `compiler_hash`
안에 포함되어 있으므로 조용한 차이가 될 수 없다.

## 6. `es-runtime-embedded`

spec 9.6이 구성 요소를 나열하며, 이 크레이트는 그것들을 조합할 뿐 아무것도 추가하지 않는다:

```
compiled observation plan   es-compile   (spec 7, spec 11.3)
policy runtime              es-policy    (spec 2.4, one Box<dyn PolicyRuntime>)
Safety Plane                es-safety    (spec 9, whole)
telemetry ring              here         (see below)
```

`EmbeddedRuntime::from_bundle`가 **유일한** 생성자이며 항상 `SafetyPlane`을 빌드한다;
`tick`은 오직 plane만이 만들어낼 수 있는 `SafeAction`을 반환한다 (`INV-12`, `INV-13`).
관절 수가 `NJ`가 아니거나 액션 지평(action horizon)이 `H`가 아닌 번들을 거부하는 것 역시
`SafetyPlane::from_ir`다.

### 리플랜 주기 (spec 8.6)

`rate.control / rate.inference` 컨트롤 틱마다 한 번의 추론이 이루어지며,
`action.execute_chunk`로 상한이 걸린다 — K를 넘어서는 행(row)들은 명령이 아니라 예측이다
(spec 8.5). 이 비율은 두 유리수 `TickRate`로부터 정확히 계산된다; `XIR-023`이 이미 이것이
`PolicyContract::replanning_hz`와 일치함을 검사했으므로, 누적되는 부동소수점 주기는 없다
(spec 3.4). 리플랜 사이에는 버퍼링된 청크가 다시 제출되고, plane이 자신의 커서를 진행시킨다.

### 실패는 에러가 아니라 청크다

누락된 센서 텐서, 플랜 에러, 추론 에러, 혹은 형태(shape)가 잘못된 액션 텐서 — 이들 모두
**빈 청크(empty chunk)**를 만들어낸다. plane은 이를 청크 언더런(chunk underrun)과 설정된
폴백(fallback)으로 전환한다. `tick`에 에러 반환이 없는 이유는, 액션 없는 컨트롤 틱이란
존재하지 않기 때문이다.

### 할당

모든 크기는 `from_bundle`에서 결정된다. **재사용(reuse)** 틱은 아무것도 할당하지 않는다 —
`es_core::alloc_count::assert_no_alloc`으로 단언(assert)된다. **리플랜(replan)** 틱은 이
크레이트가 소유하지 않는 정확히 두 곳에서 할당이 일어난다:

1. `CpuPlan::run`은 호출마다 새로운 f32 아레나를 받고, `tick`은 그것을 위한 borrowed 입력
   맵을 만든다;
2. `PolicyRuntime::infer`는 트레이트 경계다; 구현체들이 각자의 버퍼를 소유한다.

따라서 spec 9.6의 "제로 힙 할당(zero heap allocation)"은 plane, 청크 버퍼, 텔레메트리 링에
대해서는 충족되며, 저 두 경계는 M2에서 다룰 작업이다.

### 텔레메트리 링 — 알려진 중복

`es_telemetry::ring::RingBuffer`가 진짜(real one)이고 더 낫다 (시퀀스 번호, `drain_since`,
드롭 카운트). `es-telemetry`는 레이어 10이고 이 크레이트는 레이어 9이므로, 이를 의존하는 것은
레이어링 위반이다 (spec 4.2). 여기 있는 ~50줄짜리 `TelemetryRing`은 게으른(lazy) 임시방편이다.
**레이어 9 이하의 두 번째 크레이트가 링을 필요로 하게 되면, 올바른 해결책은 `RingBuffer`를
`es-core`(레이어 1)로 옮기고 이것을 삭제하는 것이다.**

## 7. 알려진 한계

- 서명 없음, 암호화 없음. `signature` 슬롯은 존재하지만 아무것도 채우지 않는다 (spec 25.1).
- 압축 없음, 의도적으로 (2절 참고).
- 스트리밍 없음: `read`와 `write`는 전체 `Vec<u8>`에 대해 동작한다. 번들은 정책 하나와 그래프
  네 개이므로, 그 정책을 실행하는 기계의 메모리에 구조적으로 들어맞는다.
- `dataset`은 배포 번들에는 없으므로, `EmbeddedRuntime::execution_hash`는 그 슬롯을 0으로
  두며, 증거 번들이 이를 채워주기 전까지는 학습 시점의 `execution_hash`와 일치하지 않는다.
- `hardware_capability`는 타겟 트리플과 CPU 기능 프로브만을 커버한다. GPU 한계와 드라이버
  버전은 레이어 9에서는 보이지 않는다 (spec 9.6은 렌더러를 제외한다); Vulkan `PolicyRuntime`은
  디바이스 식별 정보를 자신의 `runtime_hash`에 직접 포함시켜야 한다.
- Safety Plane은 새 청크가 저장된 청크와 바이트 단위로 동일할 경우 청크 커서를 그대로
  유지한다 (`docs/design/safety-plane.md`). 따라서 말 그대로 상수인 청크를 내놓는 정책은
  커서가 다 소진되어 폴백에 빠진다 — fail-safe이지만 예상 밖일 수 있다; 테스트용
  `FakeRuntime`이 정확히 이 이유로 드리프트(drift) 항을 가지고 있다.
