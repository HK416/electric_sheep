<!-- Korean translation of docs/packets/M4/P-M4-R2.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R2 — 항목들뿐 아니라 매니페스트 자체에 서명한다

리뷰 블로커 **B-2**(`docs/reviews/M4.md`, "Follow-up packets" R2)를 닫는다. 설계 노트:
`docs/design/safety-case.md` §6(이 패킷에서 갱신됨).

## context (범위)

- `crates/es-eval/src/evidence.rs` (그 유닛 테스트 포함)
- `crates/es/tests/cli.rs` (추가만, append only)
- `docs/design/safety-case.md` (서명이 무엇을 커버하는지에 대한 §6 항목)
- `docs/packets/M4/P-M4-R2.md` (이 파일)

## spec (사양)

- §25.1 — 가중치와 그래프 파일은 신뢰 경계를 넘나든다; 서명은 번들 자신에 대한 주장을
  단순히 자기 일관적인 것이 아니라 인증된 것으로 만드는 요소다.
- §5.3 — `manifest.hashes`는 선언된 실행-해시 체인 *그 자체*다. 이를 빠뜨리는 서명은
  페이로드는 인증하지만 페이로드에 대한 진술은 인증하지 않으며, 이는 §28.7 gate 17
  입장에서 본말이 전도된 것이다.
- §25.3 — 컨테이너 포맷은 버전이 매겨져 있다. 이 패킷은 포맷의 바이트를 하나도 바꾸지
  않으므로 `schema_version` 증가는 없다: scheme id는 서명된 메시지 내부의 도메인
  구분자다.
- Appendix B.6 — `CanonWriter`는 이 프로젝트의 정본(canonical) 인코더다: 리틀엔디언 정수,
  길이 접두 문자열과 blob. 서명된 메시지를 위한 인코더이기도 하다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-eval -p es-compile -p es-data -p es --all-targets -- -D warnings
cargo test -p es-eval -p es-compile -p es
cargo xtask check-spec-refs
```

Pass/fail 내용:

- `cargo test -p es-eval evidence` —
  `the_signature_covers_every_manifest_field_but_the_signature`: `(manifest, entries)` 쌍에
  서명한 뒤, `hashes`, `kind`, `schema_version`, `created_utc`, `signer_public_key`를 차례로
  다시 쓴다, 매번 동일한 entries와 동일한 서명 바이트로; 모두 `SignatureStatus::Invalid`를
  보고해야 하며, 건드리지 않은 것만 `Valid`를 보고해야 한다.
- `a_name_payload_split_shift_changes_the_digest` — entry `("ab", "c")`와 entry
  `("a", "bc")`는 서로 다른 digest를 낸다, entry를 하나 빼는 것도 마찬가지다(entry 개수가
  메시지 안에 들어 있다).
- `a_v1_signature_is_invalid_under_v2` — 옛 `blake3(name || hash)` 메시지에 대한 서명은
  `Valid`가 아니라 `Invalid`를 보고한다.
- `cargo test -p es --test cli evidence` —
  `evidence_verify_flags_a_rewritten_manifest_on_a_signed_bundle`: 실제로 서명된
  `evidence.esb`에서 `manifest.hashes.task`를 다시 쓰고, 컨테이너를 자체 entry별 blake3
  검사가 통과하도록 다시 봉인한다; `Invalid`를 보고하고, `EVID-007`을 발생시키며, `ok()`는
  false다.
- 기존의 W7 서명 테스트들(`evidence_sign_then_verify_is_valid_against_the_trusted_key`,
  `evidence_verify_flags_a_tampered_signed_bundle_invalid`,
  `evidence_verify_reports_wrong_trusted_key_as_untrusted`,
  `evidence_verify_a_schema_v1_bundle_reports_absent_signature`,
  `evidence_keygen_sign_and_verify_trust_cli_round_trip`)는 변경 없이 통과한다.

## acceptance (수용 기준)

- `signing_digest(manifest, entries)`는 `CanonWriter`로 다음을 인코딩한다: scheme id
  `"ed25519-esb-v2"`, `signature`를 비운 매니페스트를 정본 JSON으로(길이 접두 blob 하나),
  entry 개수, 그리고 각 entry의 길이 접두 이름과 32바이트 페이로드 digest. 모든 문자열과
  blob은 길이 접두이므로, 서로 다른 두 `(manifest, entry set)` 쌍이 바이트를 재분할해서
  메시지를 공유하는 일은 없다.
- 매니페스트는 필드별이 아니라 정본 JSON 형태로 들어가므로, 나중에 `BundleManifest`에
  필드가 추가되어도 자동으로 서명 대상이 된다. `signature`만 유일하게 제외되는 필드다;
  `hashes`, `kind`, `schema_version`, `created_utc`, `signer_public_key`는 모두 커버된다.
- scheme id는 새 매니페스트 필드가 아니라 digest *내부*에 존재한다: v1으로 서명된 번들은
  `Invalid`를 보고하며(`docs/design/safety-case.md` §6에 문서화됨), 새 `SignatureStatus`
  variant는 없고, `.esb` 레이아웃은 그대로다.
- `EvidenceBundle::sign`은 자신이 곧 쓸 매니페스트에 서명한다(`schema_version`은 이미
  올라가 있고 `signer_public_key`도 이미 설정되어 있다), 그래서 `verify`는 자신이 읽은
  것으로부터 동일한 메시지를 다시 유도할 수 있다.
- `verify`는 내장된 policy 번들의 것뿐 아니라, 번들 *자신*의 `manifest.hashes`도 자신의
  `chain.json`과 교차 검사한다(`EVID-007`); `chain_vs_manifest`는 `source` 레이블과,
  evidence 매니페스트만 채우는 `runtime`/`dataset` 슬롯을 얻었다. 이것이 B-2의 서명 없는
  절반이다: 다시 쓰인 매니페스트는 서명이 있든 없든 잡힌다.
- 새 trait 없음(INV-17), `BTreeMap`만 사용, English-only, 새 의존성 없음.

## forbidden (금지)

- 컨테이너 포맷(`bundle::read`/`write`, `.esb` 바이트 레이아웃)과
  `BUNDLE_SCHEMA_VERSION`: 디스크상의 바이트는 어떤 것도 바뀌지 않는다.
- 새 매니페스트 필드, 새 `SignatureStatus` variant, PKI, 키 로테이션이나 폐기(revocation).
- `crates/es-env`, `crates/es-ir`, `crates/es-usd`, `crates/es-script`, `crates/es-gpu`,
  `crates/es-render`, `crates/es-core`, `python/` — 진행 중인 다른 리뷰 패킷들이 소유한다.
- 리플레이 재실행(gate 17의 나머지 절반) — 이 패킷으로 바뀌지 않는다.
