<!-- Korean translation of docs/packets/M3/W4-domain-gap.md. The English file is the working copy; regenerate this when it changes. -->

# W4-domain-gap — 도메인 갭 진단: 지표 + 리포트

Spec: §24.3(도메인 갭 진단: 관측/액션 분포, latency/jitter, 접촉/힘,
success-rate 갭; 리포트 형식), §18.3(진단된 갭이 가리켜야 할 센서 리얼리즘 노브),
§10.3(§10 지표 집합 안의 `domain_gap`), §4.2(크레이트 계층화 — 이 크레이트가
`LeRobotDataset`이 아니라 `GapInput`을 받는 이유는 설계 문서 참조), §1.2(work
packets), §28.5 M3 W4, §28.7 게이트 15 ("domain gap report generated").

설계 노트: `docs/design/domain-gap.md`(먼저 읽을 것 — 지표 집합, 페어링 규칙,
서브샘플링 규칙, suspects-to-knob 표를 싣고 있다).

## context (범위)

```
crates/es-eval/src/domain_gap.rs   (new)
crates/es-eval/src/lib.rs          (+ `pub mod domain_gap;`)
crates/es-eval/tests/domain_gap.rs (new)
crates/es/src/cmd/gap.rs           (new)
crates/es/src/cmd/mod.rs           (+ `pub mod gap;`)
crates/es/src/main.rs              (+ one dispatch arm and the usage line)
crates/es/tests/cli.rs             (append tests)
docs/design/domain-gap.md          (new)
docs/packets/M3/W4-domain-gap.md   (this file)
```

## spec (사양)

1. `es_eval::domain_gap`은 `GapOptions`, `FeatureSamples`, `EpisodeSummary`,
   `GapInput`, `ChannelGap`, `EpisodeGap`, `Suspect`, `GapReport`, `GapError`, 그리고 단일
   항목 `DomainGap::compute(sim: &GapInput, real: &GapInput, opts: &GapOptions) ->
   Result<GapReport, GapError>`를 싣는다(새 확장 지점 trait 없음, INV-17; `BTreeMap`만).
   `DomainGap::compute(sim: &LeRobotDataset, ...)`에서 벗어난 것은 의도적이다: `es-data`와
   `es-eval`은 둘 다 spec-4.2 layer 10이므로, 같은 계층 간 의존(`cargo xtask
   layering`이 dev-dependency도 함께 검사한다)은 테스트 fixture를 위해서도 허용되지
   않는다 — 설계 문서 §1이 이를 설명한다.
2. 숫자형 채널마다: 양쪽의 mean/std와 그 차이, 두 표본 KS 통계량 `D`(정렬
   병합, 통계 크레이트 없음), Wasserstein-1(정렬 병합, 통계 크레이트 없음), 그리고
   p10/p50/p90 분위수 표(nearest-rank, `es_eval::metrics::aggregate`의 `P95`와 같은
   관례). 채널은 정렬 순서에서 정확한 이름 일치로 페어링된다(`BTreeMap` merge-join);
   한쪽에만 존재하는 이름은 `unmatched_sim` / `unmatched_real`에 나열되며, 절대 아무것도
   없는 것에 대해 점수가 매겨지지 않는다.
3. 에피소드 수준: `success_rate`, 평균 `episode_length`, 평균
   `envelope_violation_rate`를 양쪽 모두, 각각 `es_ir::evaluation::MetricValue`로 —
   `EpisodeSummary`가 모든 에피소드에 대해 `None`을 싣고 있으면 이유와 함께
   `Unavailable`이다(조작된 `0.0` 없음, `es_eval::metrics`와 같은 규칙).
4. 채널은 `ks_d > opts.threshold`일 때 "flagged"다; flag된 모든 채널은
   설계 문서 §6의 이름-접두어 표로 §18.3 노브에 매핑되어 `GapReport.suspects`에
   나타난다.
5. `GapReport::to_json`(`serde_json::to_string_pretty`)과
   `impl Display for GapReport`(평문 표), 설계 문서 §7에 따름.
6. `es gap --sim <root> --real <root> [--out gap_report.json] [--max-samples N]
   [--threshold D]`(`crates/es/src/cmd/gap.rs`): 양쪽 `LeRobotDataset`을 열고, 모든
   에피소드를 읽어 모든 숫자형 비예약(non-reserved) 피처를 유지함으로써 쪽별로
   `GapInput` 하나씩을 만들며(`success`와 `action_source` 컬럼은 대신 에피소드 블록에
   공급된다 — 설계 문서의 에피소드 수준 관례 참조), 고정 stride로 채널당
   `--max-samples`(기본값 4096)까지 결정적으로 서브샘플링된다(설계 문서 §5, RNG 없음).
   `Display` 표를 출력하고, `--out`(기본값 `./gap_report.json`)을 쓴다. 어떤 채널이든
   flag되면 종료 코드 1, 아니면 0; usage 오류는 2다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-eval -p es
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- 동일한 두 `GapInput`에 대한 `DomainGap::compute`: 모든 채널의 `ks_d`가
  ~0이며, 어느 것도 flag되지 않는다.
- 한쪽에만 큰 평균 이동(mean shift)이 주어진 피처 하나: 그 채널만(오직 그것만)
  flag되고 `suspects`에 나타난다; 관계없는 채널들은 그렇지 않다.
- 한쪽에만 존재하는 피처는 점수가 매겨지지 않고 `unmatched_sim`/`unmatched_real`에
  나열된다.
- KS `D`와 Wasserstein-1은 작은 고정 샘플에 대해 손으로 계산한 값과
  일치한다(`domain_gap.rs`의 단위 테스트).
- 디스크 위의 두 `LeRobot` fixture 데이터셋(`LeRobotWriter`,
  `crates/es/tests/cli.rs`)에 대한 `es gap`은 표를 출력하고, `GapReport`의
  `Deserialize`를 통해 왕복하는 `gap_report.json`을 쓰며, 이동된 피처가 `--threshold`를
  넘을 때 정확히 1로 종료한다.
- `cargo xtask check-spec-refs`는 이 패킷의 파일들이 추가하는 모든 `§`/`spec
  N.M` 참조를 해석한다.

## forbidden (금지)

- `crates/es-eval/src/evidence.rs`, `crates/es-data/**`, `crates/es-safety/**`,
  `crates/es-runtime-embedded/**`, `crates/es-compile/**`, `crates/es-splat/**`,
  `crates/es-editor/**`, `crates/es/src/cmd/evidence.rs`, `crates/es/src/cmd/loop.rs`를
  수정하는 것 — 다른 진행 중인 패킷들이 이들을 소유한다.
- `es-data`나 `es-telemetry`를 `es-eval`의 의존성으로(dev-dependency를
  포함해 어떤 종류로든) 추가하는 것: 같은 계층이며, `cargo xtask layering`이 거부한다(spec
  4.2 규칙 1).
- 새 확장 지점 trait(INV-17), 리포트 경로 어디에든 있는 `HashMap`(spec
  18.4), 또는 데이터셋에 해당 컬럼이 없는데도 조작된 에피소드 수준 숫자.
- `es_telemetry::protocol::PerfMetrics` 샘플 수집을 CLI에 연결하는 것: 이
  패킷에는 `--latency` 플래그가 요청되지 않았으므로, `latency.*` 채널은 미래의 호출자가
  채울 수 있는 `GapInput`의 능력일 뿐, 아직 `es gap`이 읽는 것은 아니다.
