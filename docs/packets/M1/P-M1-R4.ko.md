<!-- Korean translation of docs/docs/packets/M1/P-M1-R4.ko.md. The English file is the working copy; regenerate this when it changes. -->

# P-M1-R4 — 번들 컨테이너 강화(hardening) (`es-compile::bundle`)

M1 리뷰 후속 조치(`docs/reviews/M1.md`, Should-fix): `read`는 마지막 페이로드 이후의 남는
바이트를 조용히 무시하여 "주어진 엔트리 집합에 대한 레이아웃은 유일하다"라는 주장을 거짓으로
만들고 있었고, `PolicyBundle::open`은 해시 슬롯을 비워 둔 매니페스트가 해당 슬롯에 대한 spec
5.3 검사를 전혀 거치지 않고 열리도록 허용하고 있었다. Spec: spec 5.3(해시 체인), spec
9.6(배포 번들), spec 25.1(신뢰 경계를 넘는 신뢰할 수 없는 입력), spec 25.3(컨테이너 버전
관리). 설계 노트: `docs/design/policy-bundle.md`.

## context (범위)

```
crates/es-compile/src/bundle.rs
docs/design/policy-bundle.md
docs/packets/M1/P-M1-R4.md
```

테스트는 `bundle.rs` 자체의 `#[cfg(test)] mod tests` 안에 있다 — 새 테스트 파일은 없다.

## spec (사양)

- **남는 바이트(trailing bytes)는 거부된다.** `read`는 모든 페이로드를 거치는 동안 커서 위치를
  추적한다; 마지막 페이로드를 소비한 시점에 커서가 정확히 `bytes.len()`에 도달하지 않으면,
  `read`는 남는 바이트를 조용히 받아들이고(버리는) 대신 `BundleError::TrailingBytes { extra }`를
  반환한다.
- **필수 해시 슬롯.** `PolicyBundle::open`은 컨테이너 안의 다른 무엇도 신뢰하기 *전에*
  매니페스트의 `task`, `observation`, `learning`, `deployment`, `compiler` 해시 슬롯이 모두
  `Some`인지 확인한다; 이 중 하나라도 `None`이면 `BundleError::MissingHash { slot }`이 된다.
  `policy`, `runtime`, `dataset`은 `docs/design/policy-bundle.md`가 이미 문서화한 그대로 선택
  사항으로 남는다(`Policy` 번들은 결코 `runtime`/`dataset`을 채우지 않으며, `policy`는 `open`이
  이미 수행하는 직접적인 `weights` blake3 검사와 중복이다).
- **명시적인 체크된 산술, 그리고 상한.** `read`의 모든 길이/오프셋은 이미 `Cursor::take`의
  `checked_add`를 거친다(spec 25.1의 악의적 입력에 안전한 리더). 이 패킷은 개수나 이름 길이가
  어디에든 쓰이기 *전에* 검사되는 `MAX_ENTRIES` / `MAX_NAME_LEN` 상한(각각 4096 — 실제 번들은
  짧은 이름을 가진 소수의 엔트리만 가진다)을 추가하므로, 거대한 엔트리 개수나 이름 길이를
  주장하는 적대적인 헤더는 신뢰할 수 없는 입력의 힘을 빌려 반복하거나 할당하는 대신
  `BundleError`로 빠르게 실패한다. `write`의 두 `u32::try_from(..).unwrap_or(u32::MAX)` 호출
  (오버플로우 시 조용히 잘못된 출력을 내며, 리뷰에서 Nit으로 지적됨)은 동일한 두 오류
  variant에 대한 `?`가 되므로, 지나치게 큰 엔트리 개수/이름은 쓰기 쪽에서도 손상된 개수가
  아니라 오류가 된다.

## oracle (오라클)

```
cargo fmt -p es-compile --check
cargo clippy -p es-compile --all-targets -- -D warnings
cargo test -p es-compile bundle
```

## acceptance (수용 기준)

- 남는 바이트 1개가 추가된 유효한 컨테이너는 `BundleError::TrailingBytes { extra: 1 }`로
  `read`에 실패한다; 바이트 단위로 동일한 라운드 트립 테스트
  (`container_round_trips_and_is_deterministic`)는 영향을 받지 않는다.
- `BundleHashes::default()`(모든 슬롯이 `None`)로 만들어진 매니페스트는
  `BundleError::MissingHash { slot: "task" }`로 `PolicyBundle::open`에 실패한다 — 검사되는
  첫 번째 필수 슬롯이다 — 이 검사가 파싱보다 먼저 실행되므로 다른 엔트리에 유효한 IR TOML이
  있을 필요조차 없다.
- 그 뒤에 페이로드 바이트가 전혀 없이 페이로드 `len`을 `u32::MAX`라고 주장하는 헤더, 버퍼 끝을
  넘어서는 `name_len`, 그리고 10억으로 선언된 엔트리 개수는 각각 `BundleError`(순서대로
  `Truncated`, `Truncated`, `TooManyEntries`)를 반환한다; 패닉이 발생하지 않음을 증명하기
  위해 각 호출은 테스트 안에서 `std::panic::catch_unwind`로 감싸진다.
- 기존의 모든 `bundle::tests` 케이스는 변경 없이 그대로 통과한다.

## forbidden (금지)

- `crates/es-compile/src/budget.rs`, `crates/es-compile/src/lib.rs`,
  `crates/es-compile/src/kernels.rs`, `crates/es-compile/tests/` — 다른 패킷들의 범위다.
- `docs/design/telemetry-protocol.md`, `crates/es-telemetry/*` — P-M1-R5의 범위다.
- 새로운 외부 의존성, 새로운 trait(INV-17), 그리고 `policy`/`runtime`/`dataset`을 필수 슬롯
  목록에 추가하는 것 — 리뷰가 지적한 수정 사항은 전체 `BundleHashes`가 아니라 다섯 개의
  슬롯만을 지목한다.
- 커밋하는 것.
