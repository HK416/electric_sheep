<!-- Korean translation of docs/packets/M12/P-M12-R4-editor-split.md. The English file is the working copy; regenerate this when it changes. -->
# M12 R4 — `es-editor-model`(계층 12)과 `es-editor`(계층 13)

오너 결정 H-2, 2026-09-29(`docs/reviews/M12.md`): `es-editor`는 §1.5의 코드 10,000줄 중 9,763줄에
있어 S2–S6이 들어갈 수 없다. 헤드리스 뷰모델을 별도 크레이트로 떼어 내고, egui 셸은 그 한 계층
위에 둔다. 스펙 §1.5, §4.2(계층 표와 규칙), 부록 C.8(`LAYERS`); `docs/design/editor-shell.md`
2절(model/egui 분리는 이미 규칙이다).

## spec

- 새 크레이트 `crates/es-editor-model`(계층 **12**): 지금 `crates/es-editor/src/model/` 아래에
  있는 모든 것, i18n 표(`crates/es-editor/i18n/` → `crates/es-editor-model/i18n/`), 그리고
  디스플레이가 필요 없는 모든 테스트. egui 의존성은 `model/`이 이미 부르는 것 외에는 두지 않는다
  (`fonts.rs`가 `egui::FontDefinitions`를 만든다. 그 의존성을 유지하고 크레이트 문서에 적거나,
  그것이 유일한 이유라면 `fonts.rs`를 셸로 옮긴다 — `es-editor-model`에 `eframe`이 남지 않는
  쪽을 택한다).
- `crates/es-editor`(계층 **13**): `app.rs`, `ui/`, `main.rs`, `lib.rs`. `es-editor-model`에
  의존한다. `es-editor`에는 아무것도 의존하지 않는다(§4.2 규칙 4 유지). **`es-editor-model`에는
  `es-editor`만 의존한다**(규칙 4에 새 문장).
- 스펙: §4.2 표에 `| 13 | es-editor |` 행이 생기고 12행은 `es-editor-model`이 된다. 규칙 4는 둘
  다 부른다. 부록 C.8의 `LAYERS`에 `("es-editor-model", 12)`와 `("es-editor", 13)`이 들어간다.
  `docs/ARCHITECTURE.ko.md`와 `docs/ARCHITECTURE.md`는 **같은 커밋**에 넣는다(pre-commit 훅이
  검사한다). `xtask/src/layering.rs`의 `LAYERS`와 규칙 4 검사가 따라가고, `CLAUDE.md`의 계층
  줄은 `es-editor-model`(12) → `es-editor`(13)이라고 쓴다.
- 그 밖에는 순수한 이동이다: 동작 변화 없음, 모든 테스트가 새 크레이트에서 계속 통과,
  `tests/golden/editor/` 아래 골든은 바이트 단위로 같음, `include_str!`·픽스처 경로는 맞춘다.
- 가능하면 두 크레이트 모두 §1.5 목표 아래로. 두 숫자를 모두 보고한다.

## context

```
crates/es-editor/**
crates/es-editor-model/**
Cargo.toml
Cargo.lock
xtask/src/layering.rs
xtask/src/context_budget.rs
docs/ARCHITECTURE.md
docs/ARCHITECTURE.ko.md
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/design/editor-redesign.md
docs/design/editor-redesign.ko.md
CLAUDE.md
.githooks/pre-commit
docs/packets/M12/P-M12-R4-editor-split.md
docs/packets/M12/P-M12-R4-editor-split.ko.md
```

## oracle

1. `cargo test -p es-editor-model`과 `cargo test -p es-editor`가 합쳐서, 오늘 main에서
   `cargo test -p es-editor`가 통과하는 테스트를 정확히 통과한다(같은 이름, 같은 개수).
2. 새 행으로 `cargo xtask layering`이 초록. `layering.rs`에 `es-editor-model`이 `es-editor`에
   의존할 수 없고 제3의 크레이트가 `es-editor-model`에 의존할 수 없다는 단위 테스트.
3. `cargo xtask context-budget`: 두 크레이트 모두 10,000 미만.
4. `cargo xtask verify-goldens`, `check-spec-refs`, fmt, clippy `-D warnings`, `check-scope`;
   `cargo build -p es-editor`(바이너리).

## forbidden

모든 동작 변화; 모델 타입 이름 바꾸기; 원격 서버 접속.
