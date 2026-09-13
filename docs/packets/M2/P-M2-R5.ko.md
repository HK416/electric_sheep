<!-- Korean translation of docs/packets/M2/P-M2-R5.md. The English file is the working copy; regenerate this when it changes. -->

# P-M2-R5 — chunk 버퍼 크기 산정 모델 하나로 통일

Spec: spec 4.2, spec 8.6, spec 12.4, spec 20.2, spec 28.7. 설계 노트:
`docs/design/memory-budget.md` §3, `docs/design/batch-domains.md` §11. 리뷰
발견 사항: `docs/reviews/M2.md`의 "Should-fix" `crates/es-compile/src/budget.rs:420`
대 `crates/es-env/src/domains.rs:495`.

## context (범위)

```
crates/es-core/src/sizing.rs         (new: CHUNK_SLOTS + chunk_buffer_bytes)
crates/es-core/src/lib.rs            (adds `pub mod sizing;`)
crates/es-env/src/chunk_buffer.rs    (re-exports the constant)
crates/es-env/src/domains.rs         (`DomainSizing::chunk_buffer_bytes` delegates)
crates/es-compile/src/budget.rs      (the `chunk_buffers` item + its tests)
docs/design/memory-budget.md         (§2, §3)
docs/packets/M2/P-M2-R5.md           (this file)
```

## spec (사양)

같은 양에 대한 두 개의 in-tree 모델이 8배 어긋나 있었다: 예산 쪽은 spec
20.2의 `n_sim_envs × H × NJ × 4B × 2`를 썼고, 런타임은 `f64`의
`CHUNK_SLOTS = 8`개 슬롯을 할당했다. 둘 다 존재하는 한 gate 13의 ±10%는
도달 불가능하다.

`es-compile`은 layer 7이고 `es-env`는 layer 9이므로, 어느 쪽도 다른 쪽을
호출할 수 없다. 그래서 이 공식은 layer 1에 자리 잡는다.

```rust
es_core::sizing::CHUNK_SLOTS: usize            // 8
es_core::sizing::chunk_buffer_bytes(n_envs, nj, h) -> u64
    = n_envs * CHUNK_SLOTS * h * nj * 8        // saturating
```

`es_env::CHUNK_SLOTS`는 이제 이것의 재노출이고, `DomainSizing::chunk_buffer_bytes`는
이를 그대로 위임하며(`chunk_slots` 필드는 사라졌다 — 그것은 그 상수의 두 번째
사본이었다), `MemoryBudget::estimate`의 `chunk_buffers` 항목도 같은 함수를
호출한다.

**spec 20.2로부터의 이탈은 의도적이며** 예산이 출력하는 공식 문자열에
명시되어 있다: 런타임은 env마다 `CHUNK_SLOTS`개의 *겹치는* f64 chunk를
유지하는데, ACT의 temporal ensembling이 살아있는 모든 chunk를 평균하고(spec
8.6) 제어 경로 전체가 f64이기 때문이다. 이는 spec의 f32 더블 버퍼 줄의
4배이며, 옳은 쪽은 구현이다 — spec의 그 줄은 ensembling 버퍼보다 먼저
쓰였다.

## oracle (오라클)

```
cargo test -p es-compile budget
cargo test -p es-core sizing
```

`chunk_buffers_agree_with_the_runtime_sizing_at_the_gate_configuration`은
spec 12.4 게이트 구성(4,096 sim env, `H = 20`, `NJ = 7`)에 대해
`MemoryBudget::estimate(..).chunk_buffers`가 `es_core::sizing::chunk_buffer_bytes`와
같음을 단언한다 — 36,700,160 B로, `es_env::DomainSizing::GATE`가 단언하는
바로 그 수치다 — 그리고 출력된 공식이 그 이탈을 명시함도 단언한다.

## acceptance (수용 기준)

- `cargo fmt --check`,
  `cargo clippy -p es-core -p es-env -p es-compile --all-targets -- -D warnings`,
  `cargo test -p es-core -p es-env -p es-compile`, `cargo xtask layering`,
  `cargo xtask check-spec-refs`.
- 새 의존성 엣지 없음: `es-core`는 이미 두 호출자 모두의 아래에 있다(spec
  4.2).
- 새 trait 없음(INV-17).

## forbidden (금지)

- `CHUNK_SLOTS` 자체나 런타임의 f64 chunk 저장 방식을 바꾸는 것 — 이 패킷은
  두 모델을 통합할 뿐, 어느 쪽도 재조정하지 않는다.
- `crates/es-compile/src/plan.rs`, `exec.rs`와 그 외의 예산 항목들.
