<!-- Korean translation of docs/packets/M0/P-M0-R4.md. The English file is the working copy; regenerate this when it changes. -->

# P-M0-R4 — `es-ir`를 분리하여 §1.5 컨텍스트 예산 이내로 되돌리기

Spec: §1.5 (컨텍스트 예산), §4.2 (크레이트 계층화), Appendix B.8 (`LAYERS`).
`docs/reviews/M0.md`에서 `es-ir`의 라인 수에 대해 나온 `[HUMAN]` 항목의 후속 작업. 리뷰
등급 A.

## context

```
crates/es-ir-types/**
crates/es-ir/src/lib.rs
crates/es-ir/src/graph.rs
crates/es-ir/src/hash.rs
crates/es-ir/Cargo.toml
Cargo.toml
xtask/src/layering.rs
docs/packets/M0/P-M0-R4.md
```

## spec

`es-ir`는 코드 6,589줄(P-M0-R2 이후로는 6,669줄)로, §1.5의 목표치인 6,000줄을 초과했다.

리뷰에서 나온 두 가지 방안은 기각되었다:

- **`testing` 모듈을 밖으로 옮기는 것은 아무 효과가 없다(no-op).**
  `xtask/src/context_budget.rs`는 이미 모든 `#[cfg(...test...)] mod … { }` 블록을
  제외하며, proptest 전략들은 모두 인라인
  `#[cfg(any(test, feature = "testing"))] pub mod testing` 블록이다. 6,589라는 수치는
  이미 그것들을 제외한(net) 값이므로(3,465줄 제외됨), 파일로 옮겨봐야 아무것도 바뀌지
  않는다.
- **레이어 6에 형제 크레이트(`es-ir-deploy`)를 두는 것은 금지되어 있다** — `Graph`/`IrNode`를
  위해 `es-ir`에 의존해야 하는데, 이는 같은 레이어 간 의존(§4.2 rule 1 위반)이다.
- **`es-diag`(`codes.rs` + `diag.rs`만)는 너무 작다**: 코드 290줄뿐이라 `es-ir`는 여전히
  ~6,380줄로 남아 여전히 WARN 상태다.

그래서: **레이어 2에 새 크레이트 `es-ir-types`**를 둔다(의존 대상이 레이어 0인 `es-math`와
레이어 1인 `es-core`이므로, 2가 위치할 수 있는 최저 레이어다). 이 크레이트는 그래프가
무엇인지 알지 못하는 IR 어휘를 담는다:

| 이동 대상 | 출처 |
|---|---|
| `codes.rs`, `diag.rs`, `types.rs`, `image.rs` | `crates/es-ir/src/`에서 그대로 |
| `canon.rs` — `CanonWriter` | `crates/es-ir/src/hash.rs` |
| `NodeId` (`lib.rs`) | `crates/es-ir/src/graph.rs` — `Diagnostic`가 이를 가리킨다 |

모든 공개 경로는 재수출(re-export)을 통해 원래 위치에 그대로 남는다: `es-ir`의 `lib.rs`는
`pub use es_ir_types::{codes, diag, image, types};`를 수행하고, `hash.rs`는
`pub use es_ir_types::canon::CanonWriter;`를, `graph.rs`는 `pub use es_ir_types::NodeId;`를
수행한다. 크레이트 루트에서 모듈을 `pub use`하면 `es-ir` 내부에서도 `crate::codes::…`가
계속 해석되므로, 이동된 파일들의 `use` 구문 세 줄을 제외하면 어떤 모듈 본문도 바뀌지 않는다.

`xtask/src/layering.rs`에 `("es-ir-types", 2)`가 추가된다. 이는 이번 분리보다 앞서 작성된
Appendix B.8 표로부터의 이탈(deviation)이다; Appendix B.8이 다음에 개정될 때 spec에
반영되어야 한다.

## oracle

```
cargo xtask context-budget && cargo xtask layering && cargo test --workspace --features es-ir/testing
```

## acceptance

- `context-budget`는 모든 크레이트에 대해 `OK`를 보고한다: `es-ir` 5,774줄(이전 6,669줄),
  `es-ir-types` 909줄.
- `layering`은 크레이트 18개, 위반 없음으로 통과한다.
- `cargo check --workspace --all-targets --features es-ir/testing`은 `es-compile`,
  `es-env`, `es-policy`, `es-data`, `es-safety`, `es`, `es-runtime-embedded`나 `es-ir`
  통합 테스트의 어떤 파일도 변경 **없이** 깔끔하게 통과한다 — 모든 `es_ir::…` 경로가 여전히
  해석된다.
- `cargo test --workspace --features es-ir/testing`이 통과한다.

## forbidden

`context` 밖의 모든 파일. `Graph`, `IrNode`, 또는 IR 노드 kind를 언급하는 어떤 것이라도
`es-ir-types`로 옮기는 것 — 분리 기준선은 "그래프에 대해 아무것도 모른다"이다. 이전
경로에 재수출을 남기지 않고 공개 경로를 변경하는 것. `docs/ARCHITECTURE.ko.md`를 편집하는
것.
