<!-- Korean translation of docs/packets/M12/P-M12-R5-render-default.md. The English file is the working copy; regenerate this when it changes. -->
# M12 R5 — `render`는 `es`의 기본 피처다

소유자 결정 H-1, 2026-09-29: 기본 `cargo build -p es`에는 렌더러가 없어서, Python 도구를 모두
설치해도 시작 화면은 두 큐브 템플릿을 비활성으로 둔다(`"render": false`). `es-render`는 이미
`es-editor`를 위해 빌드되므로 기본 빌드가 치르는 비용은 작다.

## spec

- `crates/es/Cargo.toml`: `render`가 `default`에 들어간다. `--no-default-features`는 여전히 그것
  없이 빌드되고, 모든 `#[cfg(not(feature = "render"))]` 경로가 거기서 계속 동작한다.
- 기본 빌드에 렌더러가 *없음*을 단언하는 테스트(예: `--frames needs the render feature` 거부)는
  `#[cfg(not(feature = "render"))]` 아래로 옮기거나 기본 피처 없이 빌드한 바이너리를 돌린다;
  하나도 지우지 않는다.
- 그러면 `es --check-deps --json`이 기본 빌드에서 `"render": true`를 보고한다; 시작 화면의 render
  설치 줄은 에디터를 전혀 바꾸지 않고도 기본 빌드에서 사라진다.
- "`es`의 기본 빌드는 `es-render`를 끌어오지 않는다"고 말하는 문서(`eval.rs`의 거부 문구,
  `docs/design/*`, `python/es/README*`, 그렇게 말한다면 CLAUDE.md)를 고친다.

## context

```
crates/es/Cargo.toml
crates/es/src/**
crates/es/tests/**
python/es/README.md
python/es/README.ko.md
docs/design/**
Cargo.lock
docs/packets/M12/P-M12-R5-render-default.md
docs/packets/M12/P-M12-R5-render-default.ko.md
```

## oracle

1. `cargo build -p es` 뒤 `target/.../es --check-deps --json` → `"render": true`.
2. `cargo test -p es`(기본 피처)와 `cargo test -p es --no-default-features`가 모두 통과한다.
3. `es`를 빌드하는 `cargo xtask ci` 단계들(fmt, 워크스페이스 전체 clippy `-D warnings`, layering).

## forbidden

렌더러의 동작이나 해시를 바꾸는 것; 원격 서버에 접속하는 것.
