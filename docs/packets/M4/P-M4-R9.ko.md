<!-- Korean translation of docs/packets/M4/P-M4-R9.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R9 — ubuntu PR 게이트가 `es-editor`를 컴파일하도록 eframe에 Linux 윈도잉 기능 추가

`main`의 빨간 `PR gate (spec 26.2)` job(GitHub Actions run 34780514440, 커밋 `c0fd56b`,
ubuntu-latest)을 닫는다. 워크스페이스는 `eframe`을 `default-features = false`에 `glow` +
`default_fonts`만으로 고정하는데(`Cargo.toml:52-55`), 이 때문에 eframe의 기본 `x11`/`wayland`
기능이 빠진다. 둘 다 없으면 `winit 0.30.13`이 자체
`compile_error!("The platform you're compiling for is not supported by winit")`
(`winit-0.30.13/src/platform_impl/mod.rs:78`)에서 멈춘다. `cargo xtask ci`는
`cargo clippy --workspace --all-targets`를 실행하므로, Linux에서는 테스트가 하나도 돌기 전에
게이트가 실패한다 — §26.1이 1차 플랫폼으로 지정한 바로 그 플랫폼에서. `docs/design/editor-shell.md`
7절은 헤드리스 빌드에 이 기능들이 필요 없다고 했다; 크레이트가 검증된 Windows에서는 맞고
Linux에서는 틀리다.

리뷰 항목이 아니다: Linux 검증 호스트(renderer-14, Ubuntu 26.04, RTX 4090, rustc 1.98.1)를
구성하다 발견했고, 그곳에서 `cargo clippy -p es-editor --all-targets -- -D warnings`로
재현했다(exit 101, 같은 `compile_error!`).

Spec: 사양 §26.1(Linux x86_64가 1차 플랫폼), §26.2(PR 계층은 clippy, fmt, 단위 테스트를
실행한다), §23(에디터), §1.4(오라클 — 기존 ubuntu PR 게이트 — 은 이미 실패한다; 이 패킷은
그것을 약화시키지 않고 통과시킨다). 에디터 동작에는 변화가 없다.

## context (범위)

```
Cargo.toml
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M4/P-M4-R9.md
docs/packets/M4/P-M4-R9.ko.md
```

`Cargo.toml`은 `[workspace.dependencies]`의 `eframe` 기능 목록과 그 주석만 바뀐다; 두 설계
노트는 7절만 바뀐다; 패킷과 그 한국어 사본은 새로 추가된다.

## spec (사양)

- `[workspace.dependencies] eframe`은 `default-features = false`, `glow`, `default_fonts`를
  유지하고 `x11`과 `wayland`를 추가한다 — eframe 자신의 Linux 기본값이며, 그 기본 집합의
  다른 것은 넣지 않는다(`accesskit` 없음, `web_screen_reader` 없음, persistence 없음).
- 하나가 아니라 둘 다. `x11`만으로도 컴파일은 되지만, 그러면 네이티브 Wayland 세션(Ubuntu
  기본값)은 XWayland 설치에 의존하게 된다; `wayland`만 켜면 순수 X11 호스트가 빠진다.
- 새 시스템 빌드 의존성이 없다. pkg-config 검색 경로를 비워도(`PKG_CONFIG_LIBDIR=/nonexistent`,
  이때 `pkg-config --exists wayland-client`는 1을 반환한다) `cargo check -p es-editor`는
  성공한다: X11 쪽은 `x11-dl`/`x11rb`/`xkbcommon-dl`이고, `wayland-sys`는 `dlopen` 기능으로
  빌드되므로(glutin과 smithay-clipboard가 켠다) 빌드 스크립트가 아무것도 링크하지 않는다.
  따라서 ubuntu 러너에는 `apt-get install` 단계가 필요 없고 `.github/workflows/ci.yml`은
  바뀌지 않는다.
- Windows와 macOS에서는 두 기능 모두 아무 효과가 없다: 이들이 켜는 모든 크레이트는 winit과
  glutin의 Linux/BSD 전용 타깃 의존성이다.
- `docs/design/editor-shell.md`의 7절(과 그 한국어 사본)은 "헤드리스 빌드에는 필요 없다" 대신
  위 내용을 서술한다.

## oracle (오라클)

변경 전에는 Linux에서 첫 명령이 winit의 `compile_error!`로 실패한다. 변경 후, Linux
(renderer-14, `ES_REQUIRE_GPU=1`, Python 오라클 설치됨)에서:

```
cargo clippy -p es-editor --all-targets -- -D warnings
PKG_CONFIG_LIBDIR=/nonexistent PKG_CONFIG_PATH= cargo check -p es-editor
cargo tree -p es-editor -i winit -e features
cargo xtask ci
cargo xtask check-scope docs/packets/M4/P-M4-R9.md
```

Windows에서:

```
cargo clippy -p es-editor --all-targets -- -D warnings
cargo xtask ci
```

## acceptance (수용 기준)

- 위 Linux 명령이 모두 exit 0이고, `cargo xtask ci`는 모든 단계(fmt, clippy, 테스트,
  context-budget, layering, nostd, spec-refs, goldens)를 통과한다.
- Linux에서 `cargo tree -p es-editor -i winit -e features`가 `winit feature "x11"`과
  `winit feature "wayland"`를 모두 나열한다.
- Windows 게이트는 여전히 통과한다.
- 다음 push에서 GitHub `PR gate (spec 26.2)` job이 초록색이 된다. 형제 job인
  `Reference oracle tier`는 무관한 이유로 실패한다; forbidden 참조.
- Linux에서 에디터 창 열기: `Target / Status: unverified` — 검증 호스트에는 SSH로 접근 가능한
  디스플레이도 Xvfb도 없다.

## forbidden (금지)

- `crates/es-editor/**` — 소스 변경 없음; 결함은 워크스페이스 기능 집합에 있다.
- `.github/workflows/ci.yml` — `Reference oracle tier` job의 SKIP(러너에 slangc, newton,
  diffusers, ACT 체크포인트 없음)은 별도의 설계 선택(설치할지 job을 좁힐지)을 가진 별개의
  문제이며 여기서 건드리지 않는다.
- `eframe`/`egui`를 0.32.3 너머로, 또는 워크스페이스 `rust-version`을 올리는 것:
  `docs/design/editor-shell.md`의 7절이 둘 다 고정한다.
- 다른 패킷의 범위.
