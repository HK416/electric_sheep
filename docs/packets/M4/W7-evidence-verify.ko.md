<!-- Korean translation of docs/packets/M4/W7-evidence-verify.md. The English file is the working copy; regenerate this when it changes. -->

# M4 W7 — `es evidence verify` 완성: 서명, replay 계획, schema v2

Design note: `docs/design/safety-case.md` (§6 "검증이 증명하지 않는 것"이 이
패킷에 의해 갱신된다 — 먼저 읽을 것, `Valid`/`replayable`이 정확히 무엇을
의미하고 무엇을 의미하지 않는지를 명시한다).

## context (범위)

- `crates/es-eval/src/evidence.rs` (+ 테스트)
- `crates/es-eval/Cargo.toml` (`ed25519-dalek` 추가)
- `crates/es-compile/src/bundle.rs` (기존 `signature` 슬롯에 대한 동반 필드
  `BundleManifest.signer_public_key`; `BUNDLE_SCHEMA_VERSION` 1 -> 2)
- `crates/es/src/cmd/evidence.rs` (+ `keygen`/`sign`/`replay` 서브커맨드,
  `verify --trust/--require-signature`)
- `crates/es/Cargo.toml` (CLI의 키 처리에 필요한 같은 `ed25519-dalek` 고정)
- `crates/es/tests/cli.rs` (추가 전용)
- `.gitignore` (`*.eskey`)
- `docs/design/safety-case.md` (§6 갱신)
- `docs/packets/M4/W7-evidence-verify.md` (이 파일)

## spec (사양)

- §27.1(evidence 번들, `traceability.json`, `revalidation_trigger`)과
  §25.1(보안: 가중치와 그래프 파일은 신뢰 경계를 건넌다; 키는 절대
  저장소 안에 살지 않는다)이 함께, M3 W5가 예약된 슬롯으로 남겨둔 것 —
  컨테이너에 대한 진짜 서명 — 을 정의한다.
- §25.3(버전 관리: 번들 포맷은 버전이 매겨지며, 구버전도 계속 읽을 수 있게
  유지된다)은 `schema_version` 1 -> 2가 호환성을 깨지 않게 만드는 규칙이다:
  v1 매니페스트는 단순히 서명 필드가 둘 다 없을 뿐이며, 이는 서명되지 않은
  v2와 동일하게 검증된다(`signature: Absent`).
- §28.6(M4 범위 줄: "`es evidence verify` 완성")과 §28.7 게이트 17
  ("evidence 번들 검증 왕복")이 이 패킷의 헌장이다; 게이트 17은 이 패킷
  이후에도 열려 있다 — replay 재실행에는 `es`가 링크하지 않는
  `PhysicsBackend`/`PolicyRuntime` 쌍이 필요하므로, `es evidence replay`는
  `SKIPPED`를 출력하며(`es eval run`과 같은 관례), `--dry-run`만이 계획을
  출력한다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-eval -p es-compile -p es --all-targets -- -D warnings
cargo test -p es-eval -p es-compile -p es
cargo xtask check-spec-refs
```

통과/실패 내용, `crates/es-eval/src/evidence.rs` 단위 테스트와
`crates/es/tests/cli.rs`:

- `evidence_sign_then_verify_is_valid_against_the_trusted_key` — 서명하고,
  서명자의 키에 대해(`Valid`) 그리고 신뢰된 키가 없는 상태에서
  (`UntrustedKey`) 검증한다; 어느 쪽이든 `ok()`.
- `evidence_verify_flags_a_tampered_signed_bundle_invalid` — 서명 후 다시
  쓰인 항목(컨테이너가 다시 봉인되어 컨테이너 자체의 해시 검사는
  통과한다)은 `Invalid`로 검증된다.
- `evidence_verify_reports_wrong_trusted_key_as_untrusted` — 진짜 서명,
  서명자의 것이 아닌 `--trust` 키는 `Invalid`가 아니라 `UntrustedKey`를
  보고한다.
- `evidence_verify_a_schema_v1_bundle_reports_absent_signature` — 강제로
  `schema_version: 1`로 되돌린 매니페스트(W7 이전의 모든 번들이 그런
  모습이다)는 여전히 열리고 `Absent`로 검증된다.
- `evidence_verify_catches_a_reindented_report` — `build`의 정규
  정렬-키 pretty 형식 대신 압축된 형태로 재직렬화된 `report.json`은,
  동일한 값으로 파싱됨에도 `EVID_NOT_CANONICAL`을 발동시킨다.
- `evidence_keygen_sign_and_verify_trust_cli_round_trip` — `es evidence
  keygen`은 32바이트 `.eskey`를 쓰고 공개 키를 출력한다; `sign`은 서명된
  번들을 다시 내보내고 같은 키를 출력한다; `--trust` 없는 `verify`는
  "untrusted key"를 보고하고, 일치하는 `--trust pub.hex`가 있으면 "valid"를
  보고한다; `--require-signature`는 신뢰된 키 없이는 1로, 있으면 0으로
  종료한다.
- `evidence_replay_dry_run_prints_the_plan_and_is_skipped_otherwise` —
  `--dry-run` 없는 `es evidence replay`는 3으로 종료하고 `SKIPPED`를
  출력한다; `--dry-run`이 있으면 `ReplayPlan` 표를 출력하고 0으로 종료한다.

여기에 더해, `verify`의 새 `trusted_keys` 매개변수와 정규 JSON writer에 맞춰
갱신된 기존 M3 W5 테스트들(`evidence_bundle_round_trips`,
`evidence_verify_catches_a_tampered_report`,
`evidence_verify_fails_an_uncovered_requirement`,
`evidence_verify_against_lists_revalidation`,
`evidence_build_and_verify_cli_round_trip`).

## acceptance (수용 기준)

- `EvidenceBundle::sign(bytes, &SigningKey) -> Vec<u8>`는 번들을
  `manifest.signature` = 정렬 순서로 놓인 모든 비-매니페스트 항목의
  `(name, hash)` 쌍의 blake3에 대한 ed25519와 함께, 그리고
  `manifest.signer_public_key`가 설정된 채로 다시 내보낸다; `rand_core`도,
  프로세스 내 키 생성도 없다 — 호출자가 제공한 32바이트 시드로부터의
  `SigningKey::from_bytes`만.
- `EvidenceBundle::verify(bytes, against, trusted_keys) -> VerifyReport`는
  `signature: SignatureStatus`(`Valid([u8;32]) | Invalid | Absent |
  UntrustedKey([u8;32])`)와 `replayable: Vec<ReplayPlan>`를 추가한다; 둘
  다 `VerifyReport::ok()`에 영향을 주지 않는다(게이트 16은 여전히 "모든
  요구사항이 커버되고, error 진단이 없음"이다 — 서명이나 replay가
  *요구되는지*는 번들의 속성이 아니라 CLI에서 강제되는 호출자 정책이다).
- `report.json` 정규형 검사: `verify`는 그 바이트가 자기 자신의 파싱된
  형태에 대한 정확한 `serde_json::to_value` -> `to_string_pretty`가 아닌
  모든 `reports/<i>/report.json`을 지목한다(`EVID_NOT_CANONICAL`).
  `build`의 writer(`to_json`)는 이제 정확히 그 형태를 만들어내므로,
  `build`가 쓰는 어떤 것도 자기 자신의 검사에 걸리지 않는다.
- `es evidence keygen --out key.eskey`(저장소 밖에 두라고 경고함;
  `*.eskey`는 gitignore됨), `es evidence sign --key key.eskey --in
  evidence.esb --out signed.esb`, `es evidence verify --trust pub.hex
  [--trust ...] [--require-signature]`(`--require-signature`가 주어졌을
  때 `Valid` 외에는 무엇이든 1로 종료), `es evidence replay --dry-run`
  (계획을 출력함; `--dry-run` 없이는 `SKIPPED`, 3으로 종료).
- `BundleManifest.schema_version` 2; 1도 여전히 열리고 검증된다(`Absent`),
  `build`는 항상 2를 쓴다.
- 새 trait 없음(INV-17), `BTreeMap`만, 영어만, 새 코드 약 600줄 이하.

## forbidden (금지)

- `crates/es-script`, `crates/es-data`, `crates/es-usd`, `crates/es-ir`,
  `crates/es-env`, `crates/es-gpu`, `crates/es-physics-*` — 다른 진행 중인
  패킷들이 소유한다.
- 컨테이너 포맷 자체(`bundle::read`/`write`, `.esb` 바이트 레이아웃)와
  `es_eval::runner`(보고서는 `es eval run`/`write_artifacts`가 이미
  만들어내는 형태로 소비된다) — 매니페스트의 서명 슬롯과 evidence-번들
  JSON writer의 정규화만 바뀐다.
- Replay 재실행: 이 패킷이 게이트 17의 "검증" 절반을 닫은 뒤에도 "재실행
  후 비교" 절반은 여전히 M4의 과제로 남는다; 이는 이 crate가 링크하지
  않는 `PhysicsBackend`/`PolicyRuntime` 쌍이 필요하다.
- PKI, 키 로테이션, 또는 폐기 목록: `--trust`는 설계상 호출자가 이미
  신뢰하기로 결정한 키들의 평평한 목록이다(`docs/design/safety-case.md`
  §6 참고).
