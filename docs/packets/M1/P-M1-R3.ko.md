<!-- Korean translation of docs/packets/M1/P-M1-R3.md. The English file is the working copy; regenerate this when it changes. -->

# P-M1-R3 — 호출자로부터 오는 chunk 신선도

M1 리뷰의 Should-fix(`docs/reviews/M1.md`, Should-fix,
`crates/es-safety/src/plane.rs:288-302`)를 고친다: `SafetyPlane::accept`는 유효한
prefix에 대한 내용 동등성으로 chunk 신선도를 판단했으므로, 정당한 replan에서(정지된
hold, 포화된 출력) 비트 단위로 동일한 chunk를 다시 내보내는 정책은 `last_chunk_tick`을
전혀 갱신하지 못했고, `InferenceDeadline`이 건강한 정책에서도 트립될 수 있었다. 이제
신선도는 호출자로부터 온다: 호출마다 단조 증가하는 카운터인 `ActionChunk::seq`.

Spec: spec 8.5, spec 8.6, spec 9.4, `INV-13` (변경 없음: `validate`는 여전히
`SafeAction`을 반환하며 `Result`가 아니다).

## context (범위)

```
crates/es-safety/src/types.rs                              (ActionChunk gains seq: u64)
crates/es-safety/src/plane.rs                               (accept() compares seq, not bytes)
crates/es-safety/tests/scenarios.rs                          (fixture Chunk.seq, auto-increment)
crates/es-runtime-embedded/src/runtime.rs                    (increments seq once per infer)
crates/es-runtime-embedded/tests/embedded.rs                 (comment fix only)
crates/es-env/src/domains.rs                                 (feeds control_tick as seq at the
                                                               two direct SafetyPlane::validate
                                                               call sites — es-env already
                                                               constructed ActionChunk before
                                                               this packet)
tests/fixtures/safety/identical_chunks_fresh_seq.json        (new)
tests/fixtures/safety/repeated_seq_is_stale.json             (new)
docs/design/safety-plane.md                                  (chunk-identity section rewritten;
                                                               fixture table +2)
docs/packets/M1/P-M1-R3.md                                   (new)
```

## forbidden (금지)

`crates/es-compile`, `crates/es-telemetry`, `crates/es-eval`, `crates/es-data`,
`.github/**` — 동시 진행 중인 M1 후속 작업이 소유한다. `es-eval/src/runner.rs`도
`ActionChunk`를 생성하지만 범위 밖이다; `ActionChunk::new`/`::empty`가 기존 시그니처를
유지하고 `seq`를 `0`("알 수 없음")으로 기본값 처리하므로 영향받지 않으며, 컴파일을
유지하기 위해 이 패킷의 범위 밖에서 바뀌어야 할 것은 없다. 새 trait 없음(`SafetyPlane`의
확장 지점은 `INV-17`로 고정되어 있다); `SafetyPlane::validate`의 시그니처는 변경되지
않는다.

## spec (사양)

- `ActionChunk<NJ, H>`가 `pub seq: u64`를 갖게 된다. `new`/`empty`는 이를 `0`
  ("알 수 없음")으로 기본값 처리한다; 신선도를 신경 쓰는 호출자는 `.with_seq(seq)`를
  체이닝한다.
- `SafetyPlane::accept`는 `last_seq.is_none() || chunk.seq > last_seq`일 때만 chunk를
  새것으로 취급한다 — plane이 처음 보는 chunk는 그 `seq`가 무엇이든 항상 받아들여진다.
  내용은 절대 검사되지 않는다. `last_seq`는 `SafetyState`에 저장된다(`Option<u64>`, 초기값
  `None`).
- `es-runtime-embedded::EmbeddedRuntime`는 `seq: u64` 필드를 추가하며, `infer()` 호출
  (replan tick)마다 한 번씩 증가하고 plane에 건네는 chunk에 `.with_seq(self.seq)`를 통해
  붙인다; 버퍼링된 chunk를 재사용하는 tick들은 이전과 정확히 똑같이 같은 `ActionChunk`
  (같은 `seq`)를 그대로 재제출한다.
- `es-env::DomainRunner::emit_actions`는 chunk 버퍼로부터 제어 tick마다 한 행짜리
  `ActionChunk`를 합성한다; 이제 각각에 `self.control_tick`(호출마다 엄격히 증가)을
  태그하며, tick마다 내용이 달라지는 것에 의존하던 이전 방식을 대체한다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-safety -p es-runtime-embedded -p es-env --all-targets -- -D warnings
cargo test -p es-safety -p es-runtime-embedded
cargo check --workspace --all-targets
```

## acceptance (수용 기준)

- `tests/fixtures/safety/identical_chunks_fresh_seq.json`: 세 번의 replan이 매번 엄격히
  증가하는 `seq`와 함께 같은 내용을 제출한다; 모든 스텝은 `source: policy`를 단언한다.
- `tests/fixtures/safety/repeated_seq_is_stale.json`: chunk가 (내용이 다르더라도)
  *같은* `seq`로 재제출된다; 그 스텝은 `ChunkUnderrun`을 동반한
  `Fallback(HoldPosition)`이며, 이후의 새로운 `seq`가 `Policy`를 복구한다.
- 기존의 열일곱 개 시나리오와 property 스위트가 변경 없이 통과한다(`cargo test -p
  es-safety`).
- `cargo test -p es-runtime-embedded`가 통과한다. 여기에는 replan 사이의 많은 재사용
  tick을 실행하는 `the_replan_cadence_is_honored`와 `a_chunk_reuse_tick_does_not_allocate`가
  포함된다.
- `cargo check --workspace --all-targets` — `es-eval`의 기존 `ActionChunk::new` 호출
  지점들이 수정 없이 컴파일된다.
