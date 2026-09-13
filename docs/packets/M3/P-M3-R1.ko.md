<!-- Korean translation of docs/packets/M3/P-M3-R1.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-R1 — 두 표본 KS 통계량이 동점(tie)을 지나쳐 진행하도록 하기

M3 리뷰 블로커(`docs/reviews/M3.md`, `[BLOCK] domain_gap.rs:444`)에 대한 후속 조치.

Spec: §24.3 (domain-gap 진단), §10.3 (§10 지표 집합 안의 `domain_gap`), §28.7 게이트
15 ("domain gap report generated"), §1.2 (work packets), §1.4 (오라클 우선).

설계 노트: `docs/design/domain-gap.md` §2.

## context (범위)

```
crates/es-eval/src/domain_gap.rs   (ks_statistic + its unit tests)
docs/design/domain-gap.md          (§2)
docs/packets/M3/P-M3-R1.md         (this file)
```

## spec (사양)

1. `ks_statistic`는 두 *경험적(empirical)* CDF에 대해 `D = sup_x |F_sim(x) - F_real(x)|`를
   계산하며, 이 CDF들은 우연속(right-continuous) 계단 함수다: 정렬 병합(sorted merge)의 각
   단계는 `x = min(sim[i], real[j])`를 고르고, `|F_sim - F_real|`을 평가하기 전에 `x`와 같은
   모든 표본을 지나치도록 두 커서를 **모두** 진행시킨다(표준 `ks_2samp` 시맨틱스). 이전
   루프는 반복 하나씩만 진행시켰으므로, 어느 CDF 위에도 있지 않은 지점에서 갭을 측정했다.
2. 비교는 `f64::total_cmp`를 사용한다 — 두 정렬이 사용한 것과 같은 순서다 — 그래서
   병합이 `-0.0`/`NaN`에 대해 정렬과 어긋날 수 없고 항상 진행한다.
3. 시그니처 변경 없음, 새 타입 없음, 새 의존성 없음. `wasserstein1`은 이미 동점을 올바르게
   처리하며(병합된 breakpoint를 중복 제거한다) 손대지 않는다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-eval -p es
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `ks_statistic(&[1.0, 1.0, 1.0], &[1.0]) == 0.0`이고, 인자를 뒤바꿔도 같다(이전에는
  `0.667`이었다).
- 동일한 이진 채널 — 양쪽 다 0이 40% — 이 300 대 100 표본에서 정확히 `0.0`으로
  나온다(이전에는 `0.333`으로, 기본 `0.3` 임계값을 넘었다).
- 동일한 10단계 양자화 채널이 500 대 100 표본에서 정확히 `0.0`으로 나온다.
- 동점이 실제 갭을 억누르지 않는다: `{0,0,0,0}` 대 `{1,1}`은 `1.0`이고, `{0,0,1,1}` 대
  `{1}`은 `0.5`다.
- 이 파일에 이미 있던 손으로 계산한 사례들은 바뀌지 않는다: `[0,1]` 대 `[2,3]`은 `1.0`,
  `v` 대 `v`는 `0.0`, `{0,1,2,3}` 대 `{2,3,4,5}`는 `0.5`다.

## forbidden (금지)

- 통계 크레이트를 추가하는 것(§1 툴체인 최소주의) 또는 `ks_statistic`의 시그니처를
  바꾸는 것.
- `crates/es-splat/**`, `crates/es-data/**`, `xtask/**`, `.github/**`를 건드리는 것 —
  다른 진행 중인 패킷들이 이들을 소유한다.
- 새 확장 지점 trait(INV-17), 또는 리포트 경로 안의 `HashMap`(§18.4).
