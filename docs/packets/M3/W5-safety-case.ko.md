<!-- Korean translation of docs/packets/M3/W5-safety-case.md. The English file is the working copy; regenerate this when it changes. -->

# W5-safety-case — traceability 그래프, `evidence.esb`, `es evidence verify`

Spec: §27.1(Provenance Bundle + Safety Case, `evidence.esb` 레이아웃,
`traceability.json`, `revalidation_trigger`), §5.3(해시 체인, `HashChain::diff`),
§10.5(`report.json`, `evaluation.lock`), §25.1(신뢰 경계, 미구현 서명), §25.2(이 번들이
공급하는 법무 체크리스트), §28.5 M3 W5, §28.7 게이트 16(Safety Case traceability
완전성)과 게이트 17(evidence bundle 왕복, M4).

설계 노트: `docs/design/safety-case.md`(먼저 읽을 것 — 완전성 규칙,
`revalidation_trigger` 표, 그리고 검증이 증명하지 *않는* 것들의 목록을 싣고 있다).

## context (범위)

```
crates/es-compile/src/bundle.rs     (+ the Evidence-kind entry name constants only)
crates/es-eval/src/evidence.rs      (new)
crates/es-eval/src/lib.rs           (+ `pub mod evidence;`)
crates/es/src/cmd/evidence.rs       (new)
crates/es/src/cmd/mod.rs            (+ `pub mod evidence;`)
crates/es/src/main.rs               (+ one dispatch arm and the usage line)
crates/es/tests/cli.rs              (append tests; reuses the existing cross-IR `Fixture`)
docs/design/safety-case.md          (new)
docs/packets/M3/W5-safety-case.md   (this file)
```

## spec (사양)

1. `es_eval::evidence`는 그래프 타입들(`Requirement`, `Claim`, `Evidence`,
   `EvidenceKind`, `traceability: BTreeMap<String, Vec<String>>`를 가진 `SafetyCase`)을
   싣는다. 전부 `serde`이고, 새 trait 없으며(INV-17), `BTreeMap`만(§3.4: `HashMap`
   순회 순서 없음).
2. `SafetyCase::validate() -> Vec<Diagnostic>`는 중복 id, 매달린 요구사항 또는
   evidence 참조, evidence가 없는 요구사항을 거부한다.
3. `EvidenceBundle::build(policy_bundle, chain, reports, case, extra)`는
   `chain.json`, `safety_case/case.json`, `reports/<i>/{report.json,evaluation.lock}`을
   담고 정책 번들을 `policy/policy.esb`에 통째로 임베드한 `BundleKind::Evidence`의
   `.esb`를 쓴다; 매니페스트는 정책 번들의 해시 슬롯에 더해 체인으로부터 `runtime`과
   `dataset`을 싣는다. 유효하지 않은 case는 거부된다.
4. `EvidenceBundle::verify(bytes, against)`는 `VerifyReport`를 반환한다: 모든
   엔트리 해시(컨테이너는 자신의 것을 검사하고, `verify`는 각 `Evidence.hash`를 그것이
   가리키는 엔트리와 비교해 검사한다), `chain.execution_hash()`를 모든 리포트의
   `execution_hash`와, 체인을 임베드된 정책 번들의 매니페스트와 슬롯 단위로(
   `learning`/`policy` 불일치는 warning이다 — 설계 노트 §3 참조), 요구사항별
   커버리지(게이트 16), 그리고 `against`가 주어지면 `HashChain::diff` 컴포넌트들을 그것들이
   무효화하는 evidence와 짝지어 반환한다.
5. `es evidence verify <bundle.esb> [--against <other.esb>] [--json]`는 표를
   출력하고 모든 요구사항이 커버되고 오류 진단이 하나도 없을 때만 0으로, 그렇지 않으면
   1로 종료한다. `es evidence build --policy <policy.esb> --chain <chain.json> --report
   <dir>... --case <case.json> --out <evidence.esb>`.
6. `--against`는 권고적이며 절대 종료 코드를 바꾸지 않는다. 번들이 서명되었다고
   주장하는 것은 아무것도 없다: `verify`는 `signature: unverified`를 출력한다(§25.1).

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-compile -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-compile -p es-eval -p es
cargo xtask layering
cargo xtask check-spec-refs
```

pass/fail 내용은 `crates/es/tests/cli.rs`에 있다:

- `evidence_bundle_round_trips` — build 후 verify가 green이며, 모든
  요구사항이 커버된다.
- `evidence_verify_catches_a_tampered_report` — 다시 쓰인 리포트 엔트리(컨테이너가
  다시 만들어져 컨테이너 자신의 해시 검사는 통과한다)는 `Evidence.hash` 검사에 걸린다.
- `evidence_verify_fails_an_uncovered_requirement` — traceability 링크 하나가
  제거되면, 종료 코드 1.
- `evidence_verify_against_lists_revalidation` — 더 엄격한 `contact_force_max`를
  가진 deployment로부터 만들어진 두 번째 번들: `Deployment`가 재실행이 필요한 evidence와
  함께 diff에 나타난다.
- `evidence_verify_cli_round_trip` — cross-IR fixture로부터 테스트 안에서
  만들어진 번들에 대해 `es evidence build` 후 `es evidence verify`; `--json`이
  파싱된다.

여기에 더해 `crates/es-eval/src/evidence.rs`에는 `SafetyCase::validate`와
`revalidation_trigger` 표에 대한 단위 테스트가 있다.

## acceptance (수용 기준)

- 게이트 16: 일치하는 `execution_hash`를 가진 evidence가 없는 요구사항은
  검증에 실패하고, 그래프가 완전한 번들은 통과한다.
- 새 trait 없음, 새 의존성 없음, `HashMap` 없음.
- `docs/design/safety-case.md`는 완전성 규칙, trigger 표, 그리고 한계(서명
  없음, replay 없음, 적합성 주장 없음)를 §27.1 자신의 용어로 진술한다.

## forbidden (금지)

- `crates/es-safety`, `crates/es-runtime-embedded`, `crates/es-data`,
  `crates/es-splat`, `crates/es-editor`, `crates/es-eval/src/domain_gap.rs`,
  `crates/es/src/cmd/loop.rs`, `crates/es/src/cmd/gap.rs` — 이웃한 M3 패킷들이 이들을
  소유한다.
- 루트 `Cargo.toml`, 컨테이너 형식 자체(`bundle.rs`의 `write`/`read`), 그리고
  `es_eval::runner`(리포트는 오늘 쓰이는 그대로 소비된다).
- Replay 재실행과 서명 검증: M4(§28.6), 게이트 17.

## follow-ups (후속 작업)

- `EVID-0xx` 진단 코드는 `es-ir-types::codes` 사전에 없으므로(그 크레이트는
  이 패킷의 범위 밖이다), 빈 제목으로 렌더링된다. M4의 사전 작업에서 등록할 것.
- `es evidence build`는 저작자를 위해 evidence 해시를 채워주지 않는다;
  `case.json`이 그것들을 명시해야 하고 `build`는 맞지 않는 것을 거부한다. 그것들을
  제안해주는 `es evidence link` 도우미가 있으면 저작이 견딜 만해지겠지만 — 의도적으로
  여기서는 만들지 않는다, 나중에 스스로 검사할 해시를 스스로 쓰는 도구는 아무것도
  검증하지 않기 때문이다.
