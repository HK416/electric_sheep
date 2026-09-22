# M10 W3a — `es-data`를 목표 아래로: 임포터 셋이 `es-import`로 내려간다

스펙: §1.5(목표 ≤ 6,000 코드 줄, 상한 10,000; 목표를 넘으면 분할 패킷 필수), §4.2(2026-09-22에 추가된 행:
레이어 9의 `es-import` — `es-data`(10) 아래, `rl_import`가 필요로 하는 `es-policy`(8) 위; *아래로* 옮기면
의존 방향은 바뀌지 않는다), §5.3(해시 정의는 움직이지 않는다), §28.13 파동 1. 리뷰: `docs/reviews/M8.md` R6 /
사람 결정 "줄 예산". 선례: `docs/packets/M8/P-M8-R6.md`(`graph` / `hash` → `es-ir-types`, `use` 줄만 빼고
바이트 그대로, re-export, 호출자 편집 0). 유형 B.

## 질문

`es-data`는 7,321 코드 줄로 보고된다(실제 ≈ 6,900 — 아래 오계산 참조). `lerobot_config.rs`(935),
`roboverse.rs`(688), `rl_import.rs`(829)는 크레이트 안에서 LeRobot 피처 스펙과 서로 말고는 아무것도 가져오지
않고, `es-core` / `es-ir` / `es-math` / `es-policy`만 건드리며, 테스트는 별도 파일이고,
`crates/es/src/cmd/{import,policy}.rs`만 쓴다. **이 셋을 레이어 9 크레이트 `es-import`로 옮기고 `es-data`가 같은
경로로 re-export하면, 호출자 무변경·커밋 해시 불변으로 `es-data`가 목표 아래에 오는가?**

## 명세

* 새 크레이트 `crates/es-import`(레이어 9; deps `es-core`, `es-ir`, `es-math`, `es-policy`, `serde`, `serde_json`,
  `thiserror`, 그리고 세 파일이 이미 쓰는 것). `crates/es-data/src/{lerobot_config,roboverse,rl_import}.rs`와
  `crates/es-data/tests/{lerobot_config,roboverse,rl_import}.rs`를 **`use` 줄만 빼고 바이트 그대로** 옮긴다.
  `lerobot_config`가 `es_data::lerobot`(피처 스펙)에서 뭔가 필요하면 그것이 필요한 가장 작은 자족 조각을
  `es-import`로 옮기고 `es-data`에서도 re-export하거나, 타입 하나뿐이면 의존을 반대 방향으로 두거나 — 어느 쪽인지
  적는다.
* 에러 타입: 세 모듈의 에러는 새 크레이트의 것(`ImportError` 또는 모듈별)이 되고 `es_data::DataError`에
  `#[from]`을 더해 기존 모든 `?` 지점이 계속 컴파일된다.
* `es-data` re-export: `pub use es_import::{lerobot_config, roboverse, rl_import};` — `es_data::lerobot_config::…`
  등이 계속 풀린다. `git diff --stat main -- crates ':!crates/es-data' ':!crates/es-import'`는 `Cargo.toml` /
  `Cargo.lock` 줄 빼고 비어 있다.
* `xtask/src/layering.rs` `LAYERS`에 `("es-import", 9)`(스펙 표에는 이미 행이 있다). **context-budget 오계산**:
  `xtask/src/context_budget.rs`의 `brace_delta`가 raw string을 다루지 않아 `crates/es-data/src/training.rs:2194`가
  `mod tests` 안에서 TOML 본문에 중괄호가 있는 `r#"`를 열면 테스트 ≈ 419줄이 코드로 세어진다; 카운터가 raw
  string 본문을 건너뛰게 고친다(몇 줄) — 이 패킷 자체의 오라클이므로 범위 안 — 그리고 수정 전후 숫자를 보고한다.
* 설계 노트: `rl_import`가 설명된 곳(`docs/design/policy-bundle.md` 또는 `python-builder.md`)과 `lerobot-config`
  노트에 모듈이 어디 사는지 한 문장씩(+ `.ko.md`).

## context

```
crates/es-import/**
crates/es-data/src/lib.rs
crates/es-data/src/lerobot_config.rs
crates/es-data/src/roboverse.rs
crates/es-data/src/rl_import.rs
crates/es-data/src/lerobot/**
crates/es-data/tests/**
crates/es-data/Cargo.toml
Cargo.toml
Cargo.lock
xtask/src/layering.rs
xtask/src/context_budget.rs
docs/design/policy-bundle.md
docs/design/policy-bundle.ko.md
docs/design/python-builder.md
docs/design/python-builder.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/api-notes/lerobot-config.md
docs/api-notes/lerobot-config.ko.md
docs/api-notes/roboverse.md
docs/api-notes/roboverse.ko.md
docs/packets/M10/W3a-es-import-split.md
docs/packets/M10/W3a-es-import-split.ko.md
```

## 오라클

1. `cargo xtask context-budget` — `es-data` < 6,000(카운터 수정 뒤 기대 ≈ 4,450, 전 ≈ 4,870), `es-import` ≈ 2,450;
   다른 모든 크레이트 숫자는 카운터 수정에 의한 것 말고 불변이며 그것은 목록으로 적는다.
2. `cargo test --workspace` — 고정된 모든 해시 녹색; `es policy import-rl`과 `es import` CLI 테스트 불변.
3. `cargo xtask layering`; `cargo xtask check-spec-refs`; `cargo xtask verify-goldens`.
4. `git diff --stat main -- crates ':!crates/es-data' ':!crates/es-import'` — Cargo 파일만.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W3a-es-import-split.md`.

## 수용

오라클 1~5; 노트 문장들과 한국어 짝.

## 금지

`use` 줄과 에러 `#[from]` 말고 함수 본문·이름·가시성 변경; 두 크레이트 밖의 호출자 편집; `training.rs`,
`collect.rs`, `identity.rs`, `intervention.rs` 건드리기(이 파동에서 `training.rs`는 P-M9-R5의 것);
`tests/golden/**`; `docs/ARCHITECTURE*.md`; `es-` 접두사가 없는 크레이트 이름.
