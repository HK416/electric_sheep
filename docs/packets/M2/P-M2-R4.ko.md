<!-- Korean translation of docs/packets/M2/P-M2-R4.md. The English file is the working copy; regenerate this when it changes. -->

# P-M2-R4 — inference 제출 큐에 한도 두기

Spec: spec 12.1, spec 12.2, spec 12.3, spec 20.3, spec 28.4. 설계 노트:
`docs/design/batch-domains.md` §10. 리뷰 발견 사항: `docs/reviews/M2.md`의
"Should-fix" `crates/es-env/src/inference.rs:65`.

## context (범위)

```
crates/es-env/src/inference.rs       (the queue + its tests)
crates/es-env/src/lib.rs             (re-exports `default_max_pending`)
docs/design/batch-domains.md         (§10)
docs/packets/M2/P-M2-R4.md           (this file)
```

## spec (사양)

`AsyncInference::queue`는 한도 없는 `VecDeque`였다. `inference.batch`가
도착률보다 낮을 때마다 — spec 12.2가 애초에 카메라 round-robin을 두고 있는
바로 그 통상적인 과다구독 상황 — 이 큐는 한없이 자랐다. spec 28.4의 게이트
구성에서 이는 spec 20.3이 결코 일어나서는 안 된다고 말하는 메모리 부족
상태다.

- `default_max_pending(latency_ticks, batch) = batch × latency_ticks + batch`:
  파이프라인이 뒤처지지 않을 때 실제로 진행 중인 작업량과 정확히 같다(전체
  latency 동안 tick마다 방출되는 배치 하나에, 지금 채워지고 있는 배치를 더한
  것). saturating이며, 배치 하나보다 결코 작아지지 않는다.
- `AsyncInference::with_max_pending(n)`이 이를 재정의하며, `batch`까지로
  클램프된다.
- `submit`은 이 한도를 넘기게 될 제출을 버리고 `dropped_submissions()`에
  집계한다. **가장 최신** 것이 버려지므로 큐는 여전히 FIFO를 유지하며,
  back-pressure는 여전히 뒤섞이지 않고 스케줄 순서대로 작업을 지연시킨다
  (spec 12.3).
- 버려진 관측은 chunk를 만들어내지 않으므로, 다른 모든 누락된 chunk와 정확히
  같은 곳에서 드러난다: underrun, 그다음 Safety Plane의 fallback(spec 8.6,
  spec 9.4). `submitted() + dropped_submissions()`가 모든 시도의 총합이다.

## oracle (오라클)

```
cargo test -p es-env inference
```

`a_full_queue_drops_the_newest_instead_of_growing`: `batch = 1`,
`latency_ticks = 2`(따라서 `max_pending == 3`), 4,096 env × 10,000 tick =
40.96 M 제출. `pending()`은 매 tick `max_pending()` 이하로 유지되고, 초과분은
쌓이는 대신 집계되며, 테스트는 backlog를 할당하지 않고 완료된다. release 순서
테스트들은 그 한도를 명시적으로 해제한다(`with_max_pending(usize::MAX)`):
그것들은 순서에 관한 것이며, drop이 있으면 이를 혼동시킬 것이기 때문이다.

## acceptance (수용 기준)

- `cargo fmt --check`, `cargo clippy -p es-env --all-targets -- -D warnings`,
  `cargo test -p es-env`, `cargo xtask layering`, `cargo xtask check-spec-refs`.
- 새 trait 없음(INV-17). 결정성은 변하지 않는다: drop 규칙은 큐 길이의 순수
  함수이므로, 실행은 여전히 비트 단위로 재생된다.

## forbidden (금지)

- `submit`/`poll` 뒤의 실제 스레딩 — 이후 패킷의 몫이다; release 규칙은 여기
  그대로 남는다.
- `crates/es-env/src/domains.rs`의 action phase(P-M2-R3)와 크기 산정 모델
  (P-M2-R5).
