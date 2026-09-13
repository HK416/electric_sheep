<!-- Korean translation of docs/design/domain-gap.md. The English file is the working copy; regenerate this when it changes. -->

# 도메인 갭 진단 (`es-eval::domain_gap`) — 설계

Spec 참조: §24.3 (도메인 갭 진단: 무엇을 비교하는지, 왜 정책-응답 갭이 가장 유용한
신호인지), §18.3 (진단된 갭이 가리켜야 할 센서 리얼리즘 노브), §10.3 (§10 지표 집합 안의
`domain_gap` 지표 슬롯), §4.2 (크레이트 계층화), §1.2 (work packets), §28.5 M3 W4, §28.7 게이트
15 ("domain gap report generated").

리뷰 등급: C — 코드를 읽기 전에 이 문서를 먼저 읽을 것.

## 1. 이것이 무엇인지, 그리고 당연해 보이는 시그니처에서 벗어난 한 가지

§24.3은 `es domain-gap --real <rosbag|dataset> --sim <scene+task+obs>`를 기술한다: 실제
로봇 로그를 시뮬레이션에 재생해 관측 분포를 비교하는 것이다. 이 크레이트에는 아직 rosbag
재생이 없으므로(그것은 HIL 인프라, §24.2의 몫이지 이 패킷의 몫이 아니다), W4는 이미 캡처된
두 개의 `LeRobot` 데이터셋 — 하나는 시뮬레이션 롤아웃에서, 하나는 실제 로봇에서 — 사이의
갭을 진단한다. 이는 한 단계 아래에서 이루어지는 같은 비교다: 관측 채널, 액션 채널, 에피소드
결과의 분포, 그리고 실제 쪽이 기록했다면 지연시간(latency)까지.

**한 가지 벗어난 점(Deviation):** 당연해 보이는 시그니처는 `DomainGap::compute(sim:
&LeRobotDataset, real: &LeRobotDataset, opts) -> GapReport`다. 이 크레이트는 그 시그니처를
가질 수 없다: `es-data`와 `es-eval`은 둘 다 layer 10이고(§4.2), 규칙 1은 같은 계층 간
의존을 금지한다 — `cargo xtask layering`은 dev-dependency를 포함해 모든 의존 엣지에
이를 강제하므로, 테스트 전용으로 `es-eval`이 `es-data`를 import하는 것조차 CI를 실패시킨다.
그래서 `es-eval::domain_gap`은 `LeRobotDataset`이 아니라, 평범한 `Vec<f64>` 샘플로 구성된
작고 데이터셋에 무관한 입력 타입([`GapInput`](#3-types))을 받는다. `es-data`와
`es-eval::domain_gap`을 둘 다 import하는 유일한 것은 `es` 바이너리다(계층 표의 제약을 받지
않는다 — `es-*` 크레이트가 아니기 때문이다). `LeRobotDataset`을 `GapInput`으로 바꾸는 곳이
바로 `crates/es/src/cmd/gap.rs`다. 이는 리포지토리 어디에서나 볼 수 있는 것과 같은 역할
분담을 유지한다: IR/런타임 크레이트는 해시 가능하고 의존성이 가벼운 계약이고, CLI는 그 둘을
잇는 접착제다.

## 2. 지표 집합 (§24.3 비교 표, §10.3 `domain_gap`)

숫자형 채널마다(관측 피처 차원, 액션 차원, 또는 `latency.*` 시리즈 — §4 참조):

- 양쪽의 `mean`, `std`, 그리고 그 차이(`real - sim`).
- **KS 통계량 `D`**: 두 표본 콜모고로프-스미르노프(Kolmogorov-Smirnov) 통계량,
  `sup_x |F_sim(x) - F_real(x)|`이며, 정렬 병합(sorted merge)으로 계산한다 — 통계 크레이트
  없음(§1 툴체인 최소주의). 병합은 값 전체의 반복 구간(run of repeats)을 지나칠 때까지 두
  커서를 **모두** 진행시킨 뒤에야 갭을 평가한다(표준 `ks_2samp` 시맨틱스): 경험적 CDF는
  우연속(right-continuous) 계단 함수이므로, 동점 구간 중간의 한 점은 어느 CDF 위에도 있지
  않고, 거기서 측정한 차이는 `D`가 아니다. 로봇 데이터에서는 이것이 예외가 아니라 흔한
  경우다 — 이진 플래그, 양자화된 인코더 카운트, 그리고 §5의 데이터셋별 stride가 두 표본
  개수를 서로 다르게 만든다 — 그리고 동점 중간을 측정하면 두 *동일한* 이진 채널에 대해 300
  대 100 표본에서 예컨대 `0.333`을 보고하게 되는데, 이는 기본 `0.3` 임계값을 넘는다. 단위
  테스트는 반복된 단일 값, 이진 채널, 10단계 양자화 채널 각각에 대해 `n`이 서로 다를 때도
  `D = 0`을 고정한다.
- **Wasserstein-1**(1차원 샘플에 대한 earth mover's distance): 지지집합(support) 위에서
  `|F_sim(x) - F_real(x)|`를 적분한 값이며, 정렬된 샘플로부터 계산한다 — 이 역시 통계 크레이트
  없음.
- 양쪽에 대한 작은 분위수(quantile) 표(p10/p50/p90), nearest-rank 방식(보간 없음,
  `es_eval::metrics::aggregate`의 `P95`와 같은 관례)으로 계산해, 표가 정확히 재현 가능하다.

에피소드 수준(§10.3 `success_rate`, `episode_length`, `envelope_violation_rate`):

- `success_rate`, 평균 에피소드 길이, 평균 envelope 위반율을 양쪽 모두에 대해, 각각
  `es_ir::evaluation::MetricValue`로 담는다 — 데이터셋에 해당 컬럼이 없으면 이유와 함께
  `Unavailable`이며, 절대 조작된 `0.0`이 아니다(`es-eval::metrics`가 이미 따르는 것과 같은
  규칙).

지연시간(§10.3 `p50/p95_latency`, §24.2 deadline/jitter): 호출자가 `latency.obs_age_ms`
/ `latency.inference_ms` / `latency.actuation_ms` 채널(실제 실행과 함께 기록된
`es_telemetry::protocol::PerfMetrics` 샘플에서 읽어온 것)을 제공하면, 이들은 다른 숫자형
채널과 같은 경로를 거친다 — 지연시간 갭도 다른 것과 다를 바 없는 분포 갭이며, p50/p95는
분위수 표에서 그대로 나온다.

## 3. 타입

```
GapOptions { max_samples_per_feature: usize, threshold: f64 }   // KS D threshold, default 0.3
FeatureSamples { dims: usize, values: Vec<f64>, nonfinite_dropped: usize }  // row-major, n*dims
EpisodeSummary { success: Option<bool>, length: u64, envelope_violation: Option<f64> }
GapInput { channels: BTreeMap<String, FeatureSamples>, episodes: Vec<EpisodeSummary> }

ChannelGap { name, sim_n, real_n, sim_mean, real_mean, sim_std, real_std, mean_diff,
             ks_d, wasserstein1, sim_quantiles: [f64; 3], real_quantiles: [f64; 3],
             sim_nonfinite_dropped, real_nonfinite_dropped, flagged }
DimsMismatch { channel: String, sim_dims: usize, real_dims: usize }
EpisodeGap { sim_success_rate, real_success_rate, sim_length_mean, real_length_mean,
             sim_envelope_violation_rate, real_envelope_violation_rate: MetricValue }
Suspect { channel: String, ks_d: f64, knob: &'static str }
GapReport { threshold, channels: Vec<ChannelGap>, unmatched_sim: Vec<String>,
            unmatched_real: Vec<String>, dims_mismatch: Vec<DimsMismatch>,
            episodes: EpisodeGap, suspects: Vec<Suspect> }
```

`BTreeMap`만 사용한다(`HashMap`은 절대 안 됨, §18.4 결정성 관례를 같은 이유로 여기까지
확장한 것이다: 순회 순서가 해셔 상태에 의존해서는 안 된다); 모든 것이 정렬된 키 순서로
순회되므로, 같은 입력에 대한 두 번의 실행이 바이트 단위로 일치한다.

## 4. 페어링 규칙

`GapInput.channels`의 키는 이미 호출자가 네임스페이스를 붙인 것이다: 6차원 관측
피처 `observation.state`는 리포트 행 `observation.state[0]` .. `observation.state[5]`가
되고, 액션 피처도 마찬가지이며, 지연시간 시리즈는 `latency.obs_age_ms` 등이 된다. 페어링은
`sim.channels`와 `real.channels` 사이의 정확한 키 일치로 이루어진다(정렬 순서의 `BTreeMap`
교집합); 한쪽에만 존재하는 이름은 비교되지 않고 — 대신 `unmatched_sim` / `unmatched_real`에
나열된다. 양쪽 모두에 존재하지 않는 피처는 분포 갭이 아니라 스키마 차이이며, 아무것도 없는
것에 대해 거리를 지어내는 것은 정확히 이 코드베이스가 금지하는 "조작된 `0.0`"이 되기
때문이다.

양쪽 모두에 존재하지만 `dims`가 다른 이름은 같은 종류의 스키마 차이다: 그것은
`dims_mismatch`에 들어가고 점수화되지 않는다. 6차원 sim 피처의 차원 `d`를 7차원 real
피처의 차원 `d`와 비교하는 것(혹은 더 나쁘게는, sim 인덱스를 그 마지막 컬럼으로 클램핑하는
것)은 서로 다른 두 물리량을 짝지어 그것들에 대한 `D`를 보고하는 셈이다.

## 5. 서브샘플링 (유계, 결정적)

`GapInput` 구성(`crates/es/src/cmd/gap.rs` 안)은 모든 에피소드의 프레임을 읽지만
채널당 최대 `max_samples_per_feature`개의 샘플만 유지한다: 전체 프레임 수 `n`과 상한 `m`이
주어지면, stride = `ceil(n / m)`이고, 데이터셋 순서로(에피소드는 인덱스순, 프레임은 에피소드
내 순서) 프레임 인덱스 `0, stride, 2*stride, ...`를 유지한다. RNG도, 무작위 치환이 있는
저수지 샘플링(reservoir sampling)도 없다 — 위치 기반의 고정 stride는 결정적이며, 한
에피소드 안의 프레임들은 이미 시간적으로 상관되어 있으므로, 이 리포트가 계산하는 채널별
주변(marginal) 통계에 대해서는 무작위 추출보다 대표성이 떨어지지 않는다.

`LeRobotWriter` 이외의 것이 작성한 데이터셋이 가질 수 있는 두 가지 경우가 있으며, 둘 다
`es-data`가 아니라 여기서 처리된다:

- **비유한(non-finite) 값.** 어느 차원에든 `NaN`/`Inf`가 있는, 유지된 프레임은 버려지고
  `FeatureSamples::nonfinite_dropped`에 집계되며, 채널별·쪽별로 `ChannelGap`에 노출된다.
  KS도 Wasserstein-1도 `NaN`에 대해서는 의미가 없고, `serde_json`도 그것을 인코딩할 방법이
  없다 — 그래서 필터링되지 않은 샘플 하나가 표가 이미 출력된 *이후에* `gap_report.json`을
  무너뜨리곤 했다. `GapReport::to_json`이 `Result`를 반환하는 것도 같은 이유다: 이미 출력을
  낸 명령의 마지막 단계가 패닉이어서는 안 된다.
- **`n * dims`보다 짧은 컬럼.** `dims`는 피처가 선언한 `elem_count`에서 온다; 프레임당
  그만큼의 값을 담고 있지 않은 컬럼은 자기모순적인(inconsistent) 데이터셋이므로, 행
  슬라이스는 `get(..)`으로 이루어지고 miss는 인덱스 패닉이 아니라 `DataError::Inconsistent`
  (`CliError::Runtime`, exit 1)가 된다.

## 6. Suspects: flag된 채널을 §18.3 노브에 매핑하기

채널은 `ks_d > threshold`일 때 "flag"된다(기본값 `0.3`, CLI의 `--threshold`).
flag된 각 채널은 이름-접두어 휴리스틱으로 §18.3 센서 리얼리즘 노브에 매핑된다 — 실제
픽셀/신호 내용에 대한 추론이 아니라 의도적으로 조회 테이블(lookup table)이다:

| channel name contains | §18.3 knob to check |
|---|---|
| `image`, `cam`, `rgb`, `depth` | 노출, 화이트 밸런스, 렌즈 왜곡, 롤링 셔터 |
| `vel`, `velocity` | IMU/관절 인코더 노이즈 모델, 지연시간 |
| `force`, `torque`, `.ft`, `wrench` | F/T 센서 가우시안 노이즈, 온도 드리프트 |
| `state`, `joint`, `position`, `qpos` | 관절 인코더 양자화, 지연, 오프셋 |
| `action` | 액추에이터/컨트롤러 지연시간, 액션 청크 지연 |
| `latency.` | deadline/jitter 예산 (§24.2), 센서 노브가 아님 |
| (위 어느 것도 아님) | 일반 센서 노이즈 모델 (특정 §18.3 노브 없음) |

이는 의도적으로 거칠다(ponytail: 여기서는 키워드 표가 분류기보다 낫다 — 이 리포트는
사람이 가서 살펴볼 곳을 가리키는 포인터이지 진단이 아니다) 그리고 별도 파일이 아니라
`domain_gap.rs` 안의 `const` slice로 존재한다.

## 7. 리포트 스키마

`GapReport::to_json(&self) -> Result<String, serde_json::Error>`는
`serde_json::to_string_pretty`다(struct는 이미 `Serialize`를 derive한다); 파일은 CLI가
`gap_report.json`으로 쓴다. `GapReport`는 CLI가 stdout에도 출력하는 평문 표를 위해
`Display`를 구현한다: 채널당 한 행(`name`, 양쪽의 `n`, 양쪽의 `mean`/`std`, `KS D`, `W1`,
flagged 표시), 그 뒤로 unmatched-feature 목록, `dims_mismatch` 목록, 있을 경우의
non-finite drop 개수, 에피소드 수준 블록, suspects 목록이 이어진다.

## 8. CLI

```
es gap --sim <root> --real <root> [--out gap_report.json] [--max-samples N] [--threshold D]
```

양쪽 `LeRobotDataset`을 열고, 각 쪽에 대해 `GapInput`을 만들며(§5),
`domain_gap::DomainGap::compute`를 호출하고, `Display` 표를 출력하며, `to_json`을 통해
`--out`(기본값 `gap_report.json`)을 쓴다. 어떤 채널이든 flag되면(`ks_d > threshold`) 종료
코드 1, 아니면 0; usage 오류는 2다(다른 모든 `es` 서브커맨드가 쓰는 관례,
`crates/es/src/error.rs`).
