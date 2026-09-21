# M8 R6 — 목표 아래의 `es-ir`: `graph`와 `hash`가 `es-ir-types`로 내려간다

스펙: §1.5(목표 ≤ 코드 6,000줄, 하드 캡 10,000; 목표 초과분에는 분할 패킷이 필수), §4.2
(`es-ir-types`는 레이어 2, `es-ir`는 레이어 6 — *아래로*의 이동은 의존 방향을 바꾸지 않는다),
§5.3(해시 체인의 정의는 이동하지 않는다), §28.12 파동 0(플랜 T의 `ActionSpace` 변경의
전제조건). 리뷰: `docs/reviews/M8.md` R6. 선례: `es-ir-types`를 만든 첫 분리(`Expr`, `codes`,
`diag`).

## 질문

`es-ir`는 코드 6,148줄이다. `graph.rs`(402)와 `hash.rs`(718)는 `codes`, `diag`, 그리고
서로만을 임포트한다 — IR struct는 없다 — 그리고 `es-ir-types`는 이미 `codes`와 `diag`를 갖고
있다. **그것들을 아래로 옮기고 `es-ir`가 같은 경로로 재수출한다면, 호출자를 하나도 바꾸지
않고 커밋된 해시를 하나도 움직이지 않은 채 `es-ir`를 목표 아래로 되돌릴 수 있는가?**

## 사양

* `crates/es-ir/src/graph.rs`와 `crates/es-ir/src/hash.rs`를 `crates/es-ir-types/src/`로
  옮긴다(`graph`와 `hash` 모듈로), `use` 줄을 제외하면 바이트 그대로. `hash.rs`가
  `codes`/`diag`/`graph`를 넘어 `es-ir` 자체의 무언가를 임포트하는 것으로 밝혀지면
  `graph.rs`만 옮기고 그렇게 말한다; `norm.rs`는 그대로 남는다(세 IR을 임포트하므로).
* `es-ir`는 재수출한다: `pub use es_ir_types::{graph, hash};`, 그래서 `es_ir::graph::…`와
  `es_ir::hash::…`는 계속 해석된다. 워크스페이스 안의 어떤 호출자도 바뀌지 않는다; 두
  크레이트와 그 `Cargo.toml`들 밖에서 `git diff --stat`은 비어 있다.
* 옮겨진 파일 안에 있던 테스트는 그것들과 함께 이동한다; `es-ir`의 다른 곳에서 해시를
  고정하는 테스트들(`committed_*_hash*`, `hash_chain_*`, `testing` property test)은 바뀌지
  않고 돈다.
* `docs/design/hash-canonicalization.md`(+ `.ko.md`)와 `docs/design/ir-types.md`(+ `.ko.md`)는
  각각 모듈이 이제 어디에 사는지 말하는 문장 하나를 얻는다.

## context

```
crates/es-ir/src/lib.rs
crates/es-ir/src/graph.rs
crates/es-ir/src/hash.rs
crates/es-ir/Cargo.toml
crates/es-ir-types/src/**
crates/es-ir-types/Cargo.toml
docs/design/hash-canonicalization.md
docs/design/hash-canonicalization.ko.md
docs/design/ir-types.md
docs/design/ir-types.ko.md
docs/packets/M8/P-M8-R6.md
docs/packets/M8/P-M8-R6.ko.md
```

## 오라클

1. `cargo xtask context-budget` — `es-ir`가 코드 6,000줄 아래(숫자를 보고하라; 두 모듈을 다
   옮기면 ≈ 5,030, `graph`만 옮기면 ≈ 5,750으로 예상).
2. `cargo test --workspace` — 기존 테스트가 전부 green, 특히 고정된 커밋 해시들
   (`task eb6efefa…`, `task-pt d546b808…`, `learning 5dac0a46…`, `fdb5178a…`, evaluation과
   deployment의 고정값)과 `es-ir/testing`의 property test.
3. `cargo xtask layering` — 위반 없음; `cargo xtask check-spec-refs`; `cargo xtask
   verify-goldens`(아무것도 바뀌지 않음); `cargo xtask check-scope docs/packets/M8/P-M8-R6.md`.
4. `git diff --stat main -- crates ':!crates/es-ir' ':!crates/es-ir-types'`가 비어 있음.

## 수용 기준

오라클 1–4; 두 노트의 문장과 한국어 자매 문서.

## 금지

어떤 함수의 본문, 이름, 가시성을 바꾸는 것; `norm.rs`, 다섯 IR 모듈, `cross.rs`,
`factory.rs`, `serial.rs`를 건드리는 것; 새 크레이트; 호출자 수정; `docs/ARCHITECTURE*.md`;
`tests/golden/**`.
