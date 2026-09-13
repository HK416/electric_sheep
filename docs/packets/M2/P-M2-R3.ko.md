<!-- Korean translation of docs/packets/M2/P-M2-R3.md. The English file is the working copy; regenerate this when it changes. -->

# P-M2-R3 — inference deadline이 domain runner를 통과해 살아남게 하기

Spec: spec 8.5, spec 8.6, spec 9.4, spec 12.3, spec 13.1. 설계 노트:
`docs/design/batch-domains.md` §9. 리뷰 발견 사항: `docs/reviews/M2.md`의
"Should-fix" `crates/es-env/src/domains.rs:271,275`.

## context (범위)

```
crates/es-env/src/domains.rs         (the action phase + its tests)
crates/es-env/src/chunk_buffer.rs    (adds `arrivals`, `action_at`)
docs/design/batch-domains.md         (§9, §11)
docs/packets/M2/P-M2-R3.md           (this file)
```

## spec (사양)

`DomainRunner::emit_actions`는 제어 tick마다 새로 합성한 한 행짜리 chunk에
`ActionChunk::with_seq(self.control_tick)`을 찍었다. `SafetyPlane::accept`는
아직 보지 못한 `seq`를 모두 새 chunk로 취급해 `last_chunk_tick`을 그리로
옮기며, `last_chunk_tick`이 바로 `InferenceDeadline` watchdog이 측정하는
값이다 — 그래서 `DomainRunner`를 거치는 한 그 watchdog은 절대 발화할 수
없었고, 죽은 정책은 오직 `ChunkUnderrun`으로만 나타났다.

이제 `seq`는 `es-runtime-embedded/src/runtime.rs`가 이미 그렇게 하고 있듯이
**정책 호출 결과당 한 번**만 전진한다 — 그 chunk를 만들어낸 추론 완료
시점이다.

- `ChunkBuffer::arrivals()`는 push 횟수를 센다(그 env에 결과가 하나 전달될
  때마다 하나씩).
- `arrivals`가 움직였고 버퍼가 현재 tick을 커버하면, `emit_actions`는 새
  `seq`를 찍고 그 결과가 다음 결과가 올 때까지 구동할 행들로 chunk를 채운다.
  이는 새로 추가된 `ChunkBuffer::action_at`(카운터가 없는 `next_action`)을
  통해 읽힌다. 이 선견(lookahead)은 정확한데, `arrivals`를 움직이지 않고서는
  아무것도 버퍼에 도달할 수 없고, 바로 그것이 다음 재구성을 촉발하기
  때문이다.
- 그 외의 모든 tick은 이전 `seq`를 재제출한다. `accept`는 이미 보유한
  `seq`에 대해서는 아무것도 복사하지 않으므로, 그 페이로드는 절대 읽히지
  않고, plane은 자신의 chunk를 계속 소비하며, `last_chunk_tick`은 마지막
  실제 결과가 놓아둔 자리에 그대로 남는다. chunk의 행들을 넘어서면 plane
  자신의 `ChunkUnderrun`이 fallback을 만들어낸다.
- `reset_env`는 뒤에 행 없이 `seq`를 올린다. 그래서 plane은 끝난 에피소드의
  chunk를 계속 소비하는 대신 그것을 버린다(spec 13.1).

`next_action`은 여전히 env마다 제어 tick마다 한 번씩 호출되므로, §12.4의
`chunk_underrun_rate`는 변하지 않는다.

## oracle (오라클)

```
cargo test -p es-env domains::tests::a_policy_that_stops_producing_trips_the_inference_deadline
```

살아있는 정책으로 12번의 제어 스텝을 진행한 다음 정책 없이 8번을 진행한다:
`InferenceDeadline`은 정책이 도는 동안에는 0번 카운트되고 4번째 무응답
스텝에서 발화한다 — 제어 스텝당 16 ms의 plane 시간에 대해 50 ms의 예산이므로,
"예산 안"이지 늦은 것이 아니다.

## acceptance (수용 기준)

- `cargo fmt --check`, `cargo clippy -p es-env --all-targets -- -D warnings`,
  `cargo test -p es-env`, `cargo xtask layering`, `cargo xtask check-spec-refs`.
- 새 trait 없음(INV-17)이고 action phase에 힙 할당 없음: chunk는 이전과
  마찬가지로 인라인 `[[f64; NJ]; H]`다.
- plane은 여전히 유일한 actuator 경로이며 절대 비활성화되지 않는다(INV-12,
  INV-13).

## forbidden (금지)

- `crates/es-safety/**` — `SafetyPlane::validate`의 시그니처와 `accept`의
  규칙은 고정되어 있다(INV-13); 이 패킷은 호출자를 그것들에 맞춘다.
- `crates/es-runtime-embedded/**` — 이미 올바르게 되어 있으며, 여기서는
  참조 대상이다.
- P-M2-R4의 큐 한도와 P-M2-R5의 크기 산정 모델.
