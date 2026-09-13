<!-- Korean translation of docs/packets/M3/P-M3-R5.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-R5 — 오래된(stale) lock은 커버리지가 아니고, 빈 case는 완전하지 않다

M3 리뷰 should-fix 항목들(`docs/reviews/M3.md`: `evidence.rs:456`,
`evidence.rs:612`)에 대한 후속 조치.

Spec: §27.1 (Provenance Bundle + Safety Case, `traceability`, `revalidation_trigger`),
§10.5 (`report.json` / `evaluation.lock`), §5.3 (해시 체인), §28.7 게이트 16,
§1.2 (work packets), §1.4 (오라클 우선).

설계 노트: `docs/design/safety-case.md` §4.

## context (범위)

```
crates/es-eval/src/evidence.rs   (SafetyCase::validate, EvidenceBundle::verify)
crates/es/tests/cli.rs           (append tests)
docs/design/safety-case.md       (§4)
docs/packets/M3/P-M3-R5.md       (this file)
```

## spec (사양)

1. `EvidenceBundle::verify`는 `report.json`뿐 아니라 모든 `reports/<i>/` 엔트리를
   순회한다: `evaluation.lock`은 `EvaluationLock`으로 파싱되고 그 `execution_hash`(hex)는
   `chain.execution_hash()`와 비교되며, 체인이 그 슬롯을 채우고 있을 때는 그
   `evaluation_hash`가 `chain.evaluation`과 비교된다. 둘 중 하나라도 불일치하면
   `EVID-006` 진단이다.
2. 자신의 내용이 다른 실행(run)을 가리키는 엔트리는 커버리지로 세지 않는다. `verify`는
   그런 엔트리 이름들을 모으고, 요구사항별 커버리지 패스는 그것을 가리키는 evidence를
   `stale`에 넣는다. 이는 spec 10.5의 두 아티팩트 모두에 대한 구멍을 한 번에 막는다:
   이전에는 다른 실행의 report가 진단을 일으키면서도 여전히 그 요구사항을 `covered`로
   표시했고, 다른 실행의 *lock*은 아무 진단도 일으키지 않았는데, 왜냐하면 참조되는 유일한
   `execution_hash`가 case 자신이 스스로에 대해 적어둔 것뿐이었기 때문이다.
3. `SafetyCase::validate`는 요구사항이 없는 case에 대해 `EVID-008`을 보고한다.
   `VerifyReport::ok()`는 "모든 요구사항이 커버된다"인데, 이는 빈 표에 대해서는 공허하게
   참(vacuously true)이므로, 그 빈 표 자체가 지적 사항이 되어야 한다.
   `EvidenceBundle::build`는 다른 어떤 유효하지 않은 case를 거부하는 것과 같은 이유로
   이를 거부한다.
4. 새 타입 없음, `verify`의 시그니처 변경 없음, 새 의존성 없음; foreign-entry 집합에는
   `BTreeSet`을 쓴다(§18.4, `HashMap` 금지).

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-eval -p es
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `evidence_verify_treats_a_foreign_lock_as_stale`: `reports/0/evaluation.lock`이
  *다른* 실행의 lock인 번들 — entry-hash 검사가 통과하도록 `safety_case/case.json`이 그
  파일의 blake3를 기록하도록 다시 쓰여 있다 — 에서, 그 lock의 요구사항은 `stale`에 놓이고,
  `covered`가 아니며, `verify`는 ok가 아니고, `EVID-006` 진단이 그 엔트리를 지목한다.
- `evidence_verify_fails_a_case_with_no_requirements` / `a_case_with_no_requirements_is_a_
  diagnostic`: 빈 `requirements` 목록은 정확히 `EVID-008`을 만들고, `VerifyReport::ok()`는
  빈 커버리지 표에 대해 false이며, `es evidence verify`는 1로 종료한다.
- 기존 evidence 테스트들은 바뀌지 않는다: round trip, 변조된 report(여전히 요구사항
  하나는 covered), 커버되지 않은 요구사항, `--against` 재검증.

## forbidden (금지)

- `crates/es/src/cmd/evidence.rs`, `crates/es-compile/**`, `crates/es-data/**`,
  `crates/es-splat/**`, `xtask/**`, `.github/**`를 편집하는 것 — 다른 진행 중인 패킷들이
  이들을 소유한다.
- `--against`가 종료 코드에 영향을 주게 만드는 것(`docs/design/safety-case.md` §5:
  이는 권고적(advisory)이다).
- 번들이 증명하는 것 이상을 주장하는 것: 여전히 서명(§25.1)도 재실행(replay)도
  없으며(§28.6, 게이트 17), 설계 노트 §6이 그렇게 말한다.
- 새 확장 지점 trait(INV-17), 또는 `evidence.rs` 어디에든 있는 `HashMap`(§18.4).
