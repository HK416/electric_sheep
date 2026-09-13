<!-- Korean translation of docs/packets/M3/P-M3-R4.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-R4 — `no_std` 빌드를 CI 단계로 만들기

Spec: §1.6 (`xtask`는 단일 검증 진입점), §26.2 (PR 등급 CI), W2
(`docs/packets/M3/W2-embedded-nostd.md`).
리뷰 지적사항: `docs/reviews/M3.md` Should-fix — "`no_std` 분리에 대한 CI 게이트 없음":
`.github/workflows/` 나 `xtask/` 아래 어디에도
`--no-default-features --target thumbv7em-none-eabihf`를 빌드하는 곳이 없어서, W2가
확립한 속성이 다음 `es-core` 변경에서 조용히 썩어버린다.

## context (범위)

```
xtask/src/main.rs
xtask/src/nostd.rs
.github/workflows/ci.yml
docs/packets/M0/P01.md
docs/packets/M3/P-M3-R4.md
```

## spec (사양)

- `xtask/src/nostd.rs`: `cargo xtask nostd`는
  `cargo build -p es-core -p es-safety -p es-runtime-embedded --no-default-features --target
  thumbv7em-none-eabihf`를 실행한다. 타깃 감지 점검은 `rustup target list --installed`가
  출력하는 텍스트에 대한 순수 함수 `target_installed(installed: &str, target: &str) -> bool`
  이므로, 셸아웃 없이 단위 테스트가 가능하다.
- 타깃이 설치되어 있지 않으면 `nostd::run`은
  `SKIP nostd: target thumbv7em-none-eabihf not installed (rustup target add
  thumbv7em-none-eabihf)`를 출력하고 성공을 반환한다 — `oracles` 잡의 mujoco/mjwarp/torch
  스킵과 같은 "SKIP은 실패가 아니다" 규약이다. `run`은 `require: bool`을 받는다;
  `require = true`이면 같은 "타깃 없음" 조건에서 `FAIL nostd: ...`를 stderr에 출력하고
  SKIP 대신 실패를 반환한다.
- `main.rs`: 새 `nostd` match 분기가 선택적 `--require` argv 토큰을 읽어
  `nostd::run(&root, require)`를 호출한다; `cmd_ci`는 `layering::run(root)` 바로 다음에
  `nostd::run(root, false)`를 호출한다(SKIP을 용인함 — 대부분의 개발 머신과 기본 PR
  툴체인은 임베디드 타깃을 갖고 있지 않으므로).
- `.github/workflows/ci.yml`: PR 잡의 툴체인 단계가 `targets: thumbv7em-none-eabihf`를
  얻어 러너가 실제로 그 타깃을 갖도록 하고, `cargo xtask ci` 다음에 오는 두 번째 단계가
  `cargo xtask nostd --require`를 실행한다 — 리뷰가 지목한 두 선택지 중 더 단순한 쪽(
  `cargo xtask ci`의 출력을 tee해서 `^SKIP nostd`를 grep하는 대신, 별도의 필수 단계로
  둔다)이므로, CI에서의 SKIP은 조용한 녹색이 아니라 하드 실패가 된다.
- `docs/packets/M0/P01.md`: 이 패킷이 P02-P06과 같은 방식으로 `nostd`를 디스패처에
  추가한다는 점을 적어 두어, 골격 패킷 자신의 "누가 무엇을 추가하는가" 설명이 계속
  정확하게 유지되도록 한다.

## oracle (오라클)

```
cargo test -p xtask nostd::
cargo xtask nostd
```

`cargo xtask nostd`는 로컬에서 통과한다(이 환경에는 타깃이 설치되어 있다); 단위
테스트는 파서의 설치됨/설치안됨 두 분기를 모두 다룬다.

## acceptance (수용 기준)

- 타깃이 설치되어 있을 때 `cargo xtask nostd`는 `es-core`, `es-safety`,
  `es-runtime-embedded`를 `--no-default-features`로 `thumbv7em-none-eabihf`에 대해
  빌드하고 0으로 종료한다.
- `cargo xtask nostd`(타깃 미설치)는 `SKIP nostd:`로 시작하는 줄을 출력하고 0으로
  종료한다.
- `cargo xtask nostd --require`(타깃 미설치)는 `FAIL nostd:`로 시작하는 줄을 stderr에
  출력하고 0이 아닌 값으로 종료한다.
- `cargo xtask ci`는 `layering` 다음, `check-spec-refs` 이전에 `nostd`를 실행한다.
- `.github/workflows/ci.yml`의 PR 잡 툴체인 단계는 `targets` 아래 `thumbv7em-none-eabihf`
  를 나열하고, `cargo xtask ci` 다음 단계가 `cargo xtask nostd --require`를 실행한다.
- 단위 테스트가 `rustup`을 호출하지 않고 `target_installed`를 직접 검증한다(설치됨,
  없음, 그리고 후행 공백/빈 출력 줄).

## forbidden (금지)

- 단위 테스트에서 `rustup`으로 셸아웃 금지 — `target_installed`는 목록 텍스트를 그냥
  `&str` 인자로 받는다.
- `nostd` 단계 하나를 추가하는 것 이상으로 `cargo xtask ci`가 빌드하는 대상을 바꾸지
  않는다 — 이 패킷은 `context_budget`, `layering`, `spec_refs`, `goldens`를 건드리지
  않는다.
- W2가 이미 확립한 것 이상으로 `crates/**` 아래를 수정하지 않는다; 이 패킷은 기존 빌드
  명령을 `xtask`와 CI에 연결할 뿐이다.
