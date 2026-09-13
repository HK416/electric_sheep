<!-- Korean translation of docs/design/safety-case.md. The English file is the working copy; regenerate this when it changes. -->

# Safety Case 그래프와 `evidence.esb`

`es-eval::evidence`와 `es evidence verify`를 위한 설계 노트. Spec: spec 27.1
(Provenance Bundle + Safety Case, `evidence.esb`, `traceability.json`,
`revalidation_trigger`), spec 5.3 (해시 체인, `HashChain::diff`), spec 10.5
(`report.json` / `evaluation.lock`), spec 25.1 (가중치와 그래프 파일은 신뢰 경계를
건넌다; 서명 슬롯), spec 25.2 (이 번들이 원재료가 되는 법적 체크리스트), spec 28.7
게이트 16과 17.

## 1. 이 그래프가 애초에 왜 존재하는가

provenance 번들은 *계보(lineage)*를 기록한다: 이 정책은 저 데이터셋에서 왔고, 저
컴파일러로 컴파일되었다. Spec 27.1은 이것이 기록하지 않는 부분이 바로 적합성
(conformity) 파일이 필요로 하는 것이라고 말한다: **"요구사항 -> 검증 증거 관계가
존재하지 않는다."** Safety Case 그래프가 바로 그 관계이며 그 이상은 아니다. 이는
세 개의 목록과 하나의 맵이다:

```
Requirement { id, text, source, severity }        무엇이 성립해야 하며, 어떤 규칙이 그렇게 말하는가
Claim       { id, text, requirement }             사람이 하나의 요구사항에 대해 펴는 논증
Evidence    { id, kind, entry, hash,              기계 실행이 만들어낸 사실 하나,
              execution_hash }                      번들 항목으로 주소가 매겨짐
Traceability: requirement id -> [evidence id]     엣지 집합
```

`Evidence.entry`는 번들 *안의* 경로(`reports/0/report.json`)이고, `hash`는 그 항목
바이트의 blake3이며, `execution_hash`는 그것을 만들어낸 실행의 spec 5.3 체인
다이제스트다. 이 세 필드가 엣지를 장식이 아니라 검사 가능하게 만드는 것이다: 증거는
썩어버릴 URL도, 바꿔치기된 파일도, 다른 설정으로 실행된 결과도 될 수 없다.

어떤 노드 타입도 trait가 아니며 어떤 것도 확장 지점이 아니다(INV-17). 그래프는
데이터다.

## 2. 번들 레이아웃

`evidence.esb`는 `policy.esb`(`docs/design/policy-bundle.md`)와 같은 컨테이너이며
`kind = Evidence`에 항목이 더 많다. M3 W5가 채우는 만큼의 spec 27.1 트리:

```
evidence.esb
├── manifest.toml         kind = Evidence; spec 5.3 해시 슬롯 (spec 27.1 manifest.json)
├── chain.json            증명된 실행의 전체 HashChain -- 3절 참고
├── safety_case/case.json 요구사항 + 논증 + 증거 + traceability (spec 27.1)
├── reports/<i>/report.json        spec 10.5 EvaluationReport
├── reports/<i>/evaluation.lock    spec 10.5 EvaluationLock
└── policy/policy.esb     배포 번들, 통째로 임베드됨
```

policy 번들은 `task/ observation/ learning/ deployment/ policy/`로 풀어헤쳐지는 것이
아니라 **자신의 바이트 그대로** 항목 하나로 임베드된다. 다시 나누어 쪼갠다면 evidence
번들이 배포 번들이 이미 봉인한 IR을 다시 직렬화하는 셈이 되고, 직렬화 방식의 어떤
차이도 보이지 않게 될 것이다. 통째로 임베드되므로, `verify`는 이에 대해
`PolicyBundle::open`을 호출해 배포 아티팩트를 여는 것이 이미 수행하는 모든 검사 —
IR validator, cross-IR 패스, 가중치 해시, 체인 슬롯 — 를 그대로 물려받는다.

spec 27.1 트리에 있는 `scene/`, `training/`, `validation/determinism.json`,
`domain_gap.json`, `hazards.json`, `residual.json`, `signature`는 이 패킷이 쓰지
않는다. 이 패킷이 알지 못하는 항목은 `extra_entries`를 통해 함께 실려가므로, 나중에
추가하는 것은 포맷 변경이 아니다.

## 3. 왜 매니페스트뿐 아니라 `chain.json`도 있는가

`BundleManifest.hashes`는 *배포* 아티팩트가 알 수 있는 만큼의 체인 부분집합을
싣는다. 여기에는 `asset`, `scene`, `task_graph`, `evaluation`, `hardware` 슬롯이
없으므로 `execution_hash`를 이로부터 다시 계산할 수 없다 — 그리고 보고서를 이
번들에 묶어주는 유일한 것은 그 `execution_hash`가 번들이 증명하는 바로 그것이라는
사실이다. `chain.json`은 그 실행의 직렬화된 `es_ir::hash::HashChain`이며, 둘 다를
준다: 모든 보고서와 비교할 `chain.execution_hash()`, 그리고 `--against` 아래에서
`HashChain::diff`를 위한 컴포넌트 수준 필드.

`verify`는 체인을 임베드된 policy 번들의 매니페스트 슬롯과 슬롯 단위로 교차
검사한다:

| 슬롯 | 불일치 시 심각도 | 이유 |
|---|---|---|
| `task`, `observation`, `deployment`, `compiler` | error | 양쪽 모두 같은 IR로부터 같은 방식으로 이를 계산한다 |
| `learning`, `policy` | warning | 번들은 *선언된* 그래프를 해시한다(`learning_hash` / `policy_hash`); 런타임 체인은 `PolicyRuntime`이 실제로 로드한 것을 기록한다(`lowering_hash` / `weights_hash`). 이는 서로 다른 두 질문에 대한 두 개의 정직한 답이며, 그 간극은 조용히 숨겨진 것이 아니라 알려진 것이다 |
| `runtime`, `dataset` | policy 번들에는 없음 | 체인으로부터 evidence 매니페스트로 채워진다 |

## 4. 완전성 -- 게이트 16

게이트가 요구사항마다 검사하는 규칙:

> 요구사항은, 그것에 연결된 증거 중 적어도 하나가 번들 항목으로 존재하고, 그
> 기록된 `hash`가 그 항목 바이트의 blake3와 같고, 그 `execution_hash`가 번들
> 자신의 `chain.execution_hash()`와 같을 때 -- *case*가 기록하는 바로도, *항목
> 자신*이 기록하는 바로도 -- **커버된(covered)** 것이다.

이 마지막 조항의 후반부가 바로 `verify`가 case만 읽지 않고 항목들을 파싱하는
이유다. spec 10.5가 정의하는 `reports/<i>/` 아티팩트 각각은 그것을 만들어낸
실행의 해시를 싣는다: `report.json`은 `execution_hash` 바이트로, `evaluation.lock`은
hex로, 여기에 잠긴 스위트의 `evaluation_hash`까지. 둘 다 파싱되어 `chain.json`과
교차 검사된다. `report.json`만 검사했다면 정확히 `EvidenceKind::Lock` 크기만 한
구멍이 남았을 것이다: 다른 실행에서 가져온 lock을, case를 다시 써서 그 파일의
blake3를 기록하게 만들면, case가 자기 자신에 대해 할 수 있는 모든 검사를
통과하면서 커버리지로 집계되었을 것이다. 자신의 내용이 다른 실행(또는 다른
Evaluation IR)을 지목하는 항목은 진단을 발생시키며 *동시에* 그 증거를 `stale`에
넣는다.

존재하고 해시도 올바르지만 다른 실행에 속하는, 연결된 증거는 커버리지가 아니라
**stale**로 보고된다: 그것은 다른 머신에 대한 실제 측정치다. stale 증거만 가진
요구사항은 커버되지 않은 것이며, `es evidence verify`는 1을 반환하고 종료한다.

`SafetyCase::validate`는 그 어떤 것보다 먼저 실행되어 그래프 자체를 거부한다: 중복
id, 존재하지 않는 요구사항을 지목하는 claim이나 traceability 키, 존재하지 않는
증거를 지목하는 traceability 항목, 증거 목록이 아예 없는 요구사항, 그리고
**요구사항이 하나도 없는** case(`EVID-008`). 마지막 것은 트집이 아니다:
`VerifyReport::ok()`는 "모든 요구사항이 커버됨"인데, 빈 표에 대해서는 이것이
공허하게 참이 되므로, 이것이 없으면 `es evidence verify`는 아무것도 논증하지 않는
문서에 대해 초록색 게이트-16 결과를 출력하게 된다. 유효하지 않은 case로부터
`evidence.esb`를 만드는 것은 거부된다 -- 검증 불가능한 아티팩트는 아티팩트가 아예
없는 것보다 나쁘다(spec 26.1: *검증되지 않은 것은 실행되지 않은 것이다*).

## 5. `revalidation_trigger` -- "실질적 변경" 표

Spec 27.1은 `revalidation_trigger`를 요구사항마다 매달지만, 이 패킷은 이를
**증거 종류**에 매단다. 이유는 무엇이 어떤 사실을 무효화하는지가 그 사실이 뒷받침하는
문장의 속성이 아니라 그 사실이 어떻게 만들어졌는지의 속성이기 때문이며,
요구사항별 목록은 같은 진실이 낡아버릴 수 있는 두 번째 자리가 될 뿐이다. 요구사항은
자신의 증거 중 하나가 영향받을 때 정확히 그만큼 영향받는다.

`ChangedComponent`가 단위다(spec 5.3, `HashChain::diff`).

| 증거 종류 | 무엇에 의해 무효화되는가 |
|---|---|
| `EvalReport` | `execution_hash`의 모든 컴포넌트 + `Asset`, `Scene`, `Evaluation` -- 즉 `TaskGraph`를 제외한 전부 |
| `Lock` | `EvalReport`와 동일: 그 보고서가 만들어진 조건들을 기록한다 |
| `SafetyScenarioSuite` | `Asset`, `Scene`, `Task`, `Observation`, `Learning`, `Policy`, `Deployment`, `Compiler`, `Runtime`, `Hardware` -- 봉투, 무엇이 명령되었는지, 그리고 기계가 그것으로 무엇을 하는지 |
| `GoldenTest` | `Asset`, `Scene`, `Task`, `Observation`, `Compiler`, `Runtime`, `Hardware` -- golden은 정책에 대한 주장이 아니라 파이프라인에 대한 비트 단위 정확성 주장이다 |
| `HumanReview` | `Asset`, `Scene`, `Task`, `Deployment` -- 검토자가 읽은 것 |

`TaskGraph`는 아무것도 무효화하지 않는다: 이는 authoring 정체성(node id)이며, spec
5.3도 같은 이유로 이를 `execution_hash` 밖에 둔다. `Dataset`은 `HumanReview`나
`GoldenTest`를 무효화하지 않지만 보고서는 무효화하는데, 이는 `execution_hash`
안에 있기 때문이다.

`es evidence verify a.esb --against b.esb`는 `a.chain.diff(&b.chain)`을 실행하고,
바뀐 컴포넌트마다 표가 다시 실행되어야 한다고 말하는 증거 id를 출력한다. 이것이
spec 27.1의 "실질적 변경"의 기계적인 절반이다. 이는 참고용이다: `--against`는
결코 종료 코드를 바꾸지 않는데, "이 번들이 저 번들과 다르다"는 것이 이 번들의
결함이 아니기 때문이다.

## 6. 검증이 증명하지 않는 것

여기에 명시하는 이유는 spec 27.1이 포지셔닝 경고로 끝나며, CLI가 실제보다 더
많은 것을 증명하는 것처럼 읽혀서는 안 되기 때문이다.

- **서명은 인증(authentication)이지 보증(endorsement)이 아니다.**
  `EvidenceBundle::sign`(M4, W7, spec 25.1)은 컨테이너 전체에 대한 분리된(detached)
  ed25519 서명을 추가하며, `verify(bytes, against, trusted_keys)`는
  `signature: Valid(key) | Invalid | Absent | UntrustedKey`를 보고한다. 서명되는
  메시지(스킴 `ed25519-esb-v2`, P-M4-R2)는 `CanonWriter` 인코딩으로, 스킴 id, 자신의
  `signature` 슬롯만 비워진 매니페스트(그래서 `hashes`, 선언된 spec 5.3 체인,
  `kind`, `schema_version`, `signer_public_key`가 모두 포함된다), 그리고 정렬된
  순서의 모든 항목의 `(name, blake3(payload))` 쌍을 문자열과 blob마다 길이 접두를
  붙여 담는다. 첫 번째 스킴(`v1`, M4 W7)은 항목들만 서명했으므로, 서명된 번들의
  매니페스트가 다시 쓰여도 여전히 검증에 통과할 수 있었다; 스킴 id가 매니페스트
  필드가 아니라 다이제스트 안에 있기 때문에, 이제 `v1` 서명은 `Invalid`로
  보고된다 -- 이 바이너리는 자신이 계산하지 않은 메시지를 보증할 수 없다. `Valid`는
  오직 *호출자가 가진 바이트가 `trusted_keys` 소지자가 서명한 바이트와 같다*는 것만을
  뜻한다 -- 그 서명자가 무언가를 확인했는지는 전혀 말해주지 않으며, 이는 여전히
  [`VerifyReport::ok`]의 일부가 아니다: 번들이 신뢰된 서명을 지녀야 하는지 여부조차
  번들의 속성이 아니라 호출자 정책이다(`es evidence verify --require-signature`).
  `Absent`는 W7 이전에 만들어진 모든 번들이 여전히 보고하는 값이다
  (`schema_version: 1` 매니페스트에는 서명 필드가 아예 없으며, 이는 결함이 아니다 --
  spec 25.3은 구버전을 계속 읽을 수 있게 유지한다). 키는 `--trust pub.hex`가
  어떻게 채워졌는지 이상으로는 신뢰할 수 없다; 이 설계는 PKI도, 폐기 목록도, 키
  로테이션도 추가하지 않으며, 그것들이 세워질 원시 요소만을 추가한다.
- **재실행 계획은 재실행이 아니다.** `VerifyReport::replayable`(spec 28.6 게이트
  17의 전제 조건)은 `reports/<i>/` 항목마다, 재실행이 필요로 할
  `evaluation_hash`/`execution_hash`/`seeds`를 이름 붙이며,
  `es evidence replay --dry-run`이 이를 출력한다. 여기 있는 어떤 것도 셀을 다시
  실행하거나 재실행 지표를 기록된 것과 비교하지 않는다 -- 그러려면 이 crate가
  링크하지 않는 `PhysicsBackend` + `PolicyRuntime` 쌍이 필요하며, 이것이 바로
  평범한 `es evidence replay`가 무언가를 실행하는 척하는 대신
  `SKIPPED`를 출력하는 이유다(백엔드를 쓸 수 없을 때의 `es eval run`과 같은
  관례). `report.json`에 대한 정규형(canonical-form) 검사(파싱된 형태를 정렬된
  키로 `serde_json` 재직렬화한 것과 바이트 단위로 동일함)는 그 항목이
  *기계로 작성되었다*는 것만 증명할 뿐, 다시 실행하면 같은 숫자를 재현하리라는
  것은 증명하지 않는다.
- **적합성(conformity)이 아니다.** 이 번들은 기술 파일을 작성하기 위한 증거
  수집과 추적이다. 이는 인증이 아니며, 초록색 `verify` — 서명되었든 아니든 —는
  적합성 진술이 아니다(spec 27.1).
- **논증에 대한 판단이 아니다.** 모든 요구사항에 증거가 있다는 것은 그
  요구사항들이 옳은 것인지, 논증이 실제로 따라오는지에 대해서는 아무것도 말해주지
  않는다. 그것은 `HumanReview` 증거 종류가 대체하기 위해서가 아니라 기록하기
  위해 존재하는, 사람에 의한 검토의 몫이다.
