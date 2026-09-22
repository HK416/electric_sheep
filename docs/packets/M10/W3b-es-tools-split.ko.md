# M10 W3b — `es`를 목표 아래로: 아티팩트 동사들이 `es-tools`로 옮겨간다

스펙: §1.5, §4.2(2026-09-22에 추가된 행: 레이어 11의 `es-tools` — 아티팩트를 읽는 `es` 동사들: video, showcase,
backend, evidence, gap, bench), §26.2(진입점 하나: `es`가 유일한 바이너리로 남고 `es-tools`는 그것이 부르는
라이브러리다), §28.13 파동 2. 리뷰: `docs/reviews/M8.md` R6. 선례: P-M8-R6와 **W3a**(이 파동의 첫 분할).
**W0a**와 **W1a**가 머지된 뒤(`cmd/eval.rs`, `cmd/showcase.rs`를 거기서 건드린다). 유형 B.

## 질문

`es`는 7,485 코드 줄로 보고된다. `cmd/` 모듈들은 `error` / `util`(35줄)을 중심으로 한 별 모양이다;
`video` + `showcase` + `backend`(1,232, `render` 피처 이음새 — `showcase`가 그 뒤의 유일한 모듈)와 `evidence` +
`gap` + `bench`(871)는 `error` / `util` 말고 크레이트 안 간선이 없다(`showcase` → `backend`, `eval` 제외). **이 여섯
동사를 레이어 11 라이브러리 크레이트 `es-tools`로 옮기고 `CliError` / `hex`를 거기로 재배치해 `es`가 re-export하면,
`crates/es/tests/cli.rs`(12,800줄, 바이너리 실행) 무변경으로 `es`가 목표 아래에 오는가?**

## 명세

* 새 크레이트 `crates/es-tools`(레이어 11, 라이브러리): `video.rs`, `showcase.rs`, `backend.rs`, `evidence.rs`,
  `gap.rs`, `bench.rs`를 `use` 줄만 빼고 바이트 그대로 옮긴다; `error.rs`(`CliError`)와 `util.rs`(`hex`)가 함께
  옮겨가고 `es`가 re-export(`pub use es_tools::{error, util}` 또는 동등한 것)해 남은 `cmd/*` 모듈이 무변경으로
  컴파일된다. `showcase`가 쓰는 `cmd::eval`(`renderer_cfg` 등): 공유 헬퍼의 가장 작은 조각을 `es-tools`로 옮기고
  `es::cmd::eval`이 거기서 부르게(`use` 변경, 본문 변경 없음) — 무엇을 옮겼는지 적는다.
* `render` 피처 이동: `es-tools/Cargo.toml` `render = ["es-env/render", "dep:es-gpu", "dep:es-render"]`,
  `es/Cargo.toml` `render = ["es-tools/render"]`; `es` 안의 다른 것이 필요 없으면 선택적 `es-gpu` / `es-render`
  의존을 뺀다.
* `crates/es/src/main.rs`: 디스패처가 `es_tools::video::run(...)` 등을 부른다; `TOP_HELP` 불변; 모든 명령의
  argv·출력·종료 코드 불변(CLI 테스트가 오라클).
* `xtask/src/layering.rs` `LAYERS`에 `("es-tools", 11)`; `es` 자체는 면제 유지.

## context

```
crates/es-tools/**
crates/es/src/main.rs
crates/es/src/cmd/mod.rs
crates/es/src/cmd/error.rs
crates/es/src/cmd/util.rs
crates/es/src/error.rs
crates/es/src/util.rs
crates/es/src/cmd/video.rs
crates/es/src/cmd/showcase.rs
crates/es/src/cmd/backend.rs
crates/es/src/cmd/evidence.rs
crates/es/src/cmd/gap.rs
crates/es/src/cmd/bench.rs
crates/es/src/cmd/eval.rs
crates/es/Cargo.toml
Cargo.toml
Cargo.lock
xtask/src/layering.rs
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M10/W3b-es-tools-split.md
docs/packets/M10/W3b-es-tools-split.ko.md
```

(`editor-shell.md`는 런치 패널이 `es` 바이너리의 동사를 이름하는 곳만 — 있어도 한 문장.)

## 오라클

1. `cargo xtask context-budget` — `es` < 6,000(기대 ≈ 5,380), `es-tools` ≈ 2,100.
2. `cargo test -p es --test cli`와 `--test video` — 파일 무변경, 녹색; `cargo test -p es-tools`.
3. `cargo build -p es --features render`와 없이; `cargo xtask layering`; `check-spec-refs`; `verify-goldens`
   (`tests/golden/video/**`, `editor/**` 불변).
4. `git diff --stat main -- crates ':!crates/es' ':!crates/es-tools'` — Cargo 파일만.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W3b-es-tools-split.md`.

## 수용

오라클 1~5.

## 금지

어떤 동사의 argv·출력·종료 코드·도움말 변경; 두 번째 바이너리; `crates/es/tests/**` 편집; 재배치된 `error` / `util`을
위한 `use` 줄 말고 `cmd/{train,cycle,loop,dataset,policy,import,ir,task,generate,check_deps,mcp,telemetry}.rs`
건드리기; `tests/golden/**`; `docs/ARCHITECTURE*.md`.
