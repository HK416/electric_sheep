<!-- Korean translation of docs/packets/M3/P-M3-R2.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-R2 — `es gap`이 적대적인(hostile) 데이터셋에서도 살아남기

M3 리뷰 should-fix 항목들(`docs/reviews/M3.md`: `domain_gap.rs:174`,
`cmd/gap.rs:168`, `domain_gap.rs:350`)에 대한 후속 조치.

Spec: §24.3 (domain-gap 진단), §19.1 (LeRobot 데이터셋 레이아웃), §28.7 게이트 15,
§1.2 (work packets), §1.4 (오라클 우선).

설계 노트: `docs/design/domain-gap.md` §3, §4, §5, §7.

## context (범위)

```
crates/es-eval/src/domain_gap.rs   (GapReport::to_json, FeatureSamples, ChannelGap, compute)
crates/es/src/cmd/gap.rs           (build_input)
crates/es/tests/cli.rs             (append tests)
docs/design/domain-gap.md          (§3, §4, §5, §7)
docs/packets/M3/P-M3-R2.md         (this file)
```

세 개의 신뢰 경계(trust-boundary) 구멍이며, 모두 이 저장소가 작성하지 않은 데이터셋에서
`gap_report.json`으로 가는 경로 위에 있다.

## spec (사양)

1. **비유한(non-finite) 샘플.** `build_input`은 모든 차원이 유한할 때만 프레임을
   유지한다; 버려진 프레임은 `FeatureSamples::nonfinite_dropped`를 증가시키고, `compute`는
   이를 `ChannelGap::sim_nonfinite_dropped` / `real_nonfinite_dropped`로 옮기며 `Display`는
   이를 합계로 출력한다. KS도 Wasserstein-1도 `NaN`에 대해서는 의미가 없고, `serde_json`은
   그것을 인코딩하기를 거부한다.
2. **`to_json`이 `Result`를 반환한다.** `GapReport::to_json(&self) -> Result<String,
   serde_json::Error>`; `.expect`는 사라졌다. CLI는 그 오류를 `CliError::Runtime`으로
   매핑한다. 이미 표를 출력한 명령은 디스크에 쓰는 도중 패닉해서는 안 된다.
3. **컬럼 슬라이스를 검사한다.** `flat.get(local * dims..(local + 1) * dims)`를 쓰고,
   miss 시 에피소드, 채널, 그 컬럼이 담고 있는 값의 개수, 그리고 선언된 `elem_count`가
   암시하는 개수를 명시하는 `DataError::Inconsistent`를 낸다. `dims`는 `info.json`에서 온다;
   맞지 않는 컬럼은 인덱스 패닉이 아니라 자기모순적인 데이터셋이다.
4. **폭이 어긋나면 나열하고, 클램핑하지 않는다.** 양쪽이 같은 이름으로 갖고 있지만
   `dims`가 다른 채널은 `GapReport::dims_mismatch` 엔트리(`DimsMismatch
   { channel, sim_dims, real_dims }`)가 되고 점수화되지 않는다. `compare_channel`은 오직
   폭이 같을 때만 호출되므로, 서로 무관한 컴포넌트를 비교했던 `d.min(dims - 1)` 클램프는
   사라졌다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-eval -p es
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `gap_hostile_dataset_errors_instead_of_panicking`: `NaN` 컬럼을 가진 *동시에*
  parquet 컬럼보다 더 넓은 피처를 선언한 `info.json`을 가진 테스트 내 `LeRobot` 데이터셋은
  `es gap`을 종료 코드 1로, stderr에 `error: ... inconsistent ...`와 함께 끝나게 한다 —
  패닉도, 인덱스 범위 초과도 없다.
- `gap_nonfinite_samples_are_dropped_and_counted`: `NaN` 하나만으로는 치명적이지
  않다 — 실행이 완료되고, `gap_report.json`이 파싱되며, 영향받은 채널은 자신의 드롭된
  프레임 개수를 보고하고, 그 `mean`/`std`는 유한하다.
- `a_channel_whose_width_disagrees_is_listed_not_clamped`: 2폭 sim 채널을 3폭 real
  채널과 비교하면 `ChannelGap` 행은 생기지 않고 `dims_mismatch` 엔트리가 하나 생긴다.
- `report_json_round_trips`는 새 `Result` 시그니처를 통해서도 여전히 성립한다.

## forbidden (금지)

- `es_data::DataError`에 변형을 추가하거나 `crates/es-data/**`를 편집하는 것:
  `Inconsistent`는 자기모순적인 데이터셋이 정확히 무엇인지를 나타내며, 그 크레이트는 다른
  패킷이 소유한다.
- 비유한 샘플을 값으로 조용히 대체하거나, 비교된 적 없는 채널에 `0.0`을 대체하는 것 —
  설계 문서의 "아무것도 지어내지 않는다" 규칙이다.
- `crates/es-splat/**`, `xtask/**`, `.github/**`, `crates/es/src/cmd/evidence.rs`를
  건드리는 것.
- 새 확장 지점 trait(INV-17), 또는 리포트 경로 안의 `HashMap`(§18.4).
