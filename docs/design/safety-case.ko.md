<!-- Korean translation of docs/design/safety-case.md. The English file is the working copy; regenerate this when it changes. -->

# Safety Case 그래프와 `evidence.esb`

`es-eval::evidence`와 `es evidence verify`에 대한 설계 노트다. Spec: spec 27.1
(Provenance Bundle + Safety Case, `evidence.esb`, `traceability.json`,
`revalidation_trigger`), spec 5.3(해시 체인, `HashChain::diff`), spec 10.5(`report.json` /
`evaluation.lock`), spec 25.1(가중치와 그래프 파일이 신뢰 경계를 넘는 것; signature 슬롯),
spec 25.2(이 번들이 원재료가 되는 법무 체크리스트), spec 28.7 게이트 16과 17.

## 1. 이 그래프가 애초에 왜 존재하는가

provenance bundle은 *계보(lineage)*를 기록한다: 이 정책은 그 데이터셋에서 왔고, 그
컴파일러가 컴파일했다는 식이다. Spec 27.1은 그것이 기록하지 않는 부분이 바로 적합성 문서에
필요한 부분이라고 말한다: **"요구사항 -> 검증 증거 관계가 없다."** Safety Case 그래프는 바로
그 관계이며, 그 이상도 이하도 아니다. 이는 세 개의 목록과 하나의 맵이다:

```
Requirement { id, text, source, severity }        what must hold, and which rule says so
Claim       { id, text, requirement }             the argument a human makes for one requirement
Evidence    { id, kind, entry, hash,              a fact produced by a machine run,
              execution_hash }                      addressed as a bundle entry
Traceability: requirement id -> [evidence id]     the edge set
```

`Evidence.entry`는 번들 *내부*의 경로이고(`reports/0/report.json`), `hash`는 그
entry 바이트의 blake3이며, `execution_hash`는 그것을 만든 실행의 spec 5.3 체인 다이제스트다.
이 세 필드가 바로 이 엣지를 장식이 아니라 검사 가능하게 만드는 것이다: evidence는 썩어버릴
URL일 수도, 바꿔치기된 파일일 수도, 다른 구성으로 실행된 결과일 수도 없다.

어떤 노드 타입도 trait이 아니고 어느 것도 확장 지점이 아니다(INV-17). 이 그래프는
데이터다.

## 2. 번들 레이아웃

`evidence.esb`는 `policy.esb`(`docs/design/policy-bundle.md`)와 같은 컨테이너이며
`kind = Evidence`이고 엔트리가 더 많다. Spec 27.1의 트리 중 M3 W5가 채우는 부분까지는 다음과
같다:

```
evidence.esb
├── manifest.toml         kind = Evidence; the spec 5.3 hash slots (spec 27.1 manifest.json)
├── chain.json            the full HashChain of the attested run -- see section 3
├── safety_case/case.json requirements + claims + evidence + traceability (spec 27.1)
├── reports/<i>/report.json        spec 10.5 EvaluationReport
├── reports/<i>/evaluation.lock    spec 10.5 EvaluationLock
└── policy/policy.esb     the deployment bundle, embedded whole
```

정책 번들은 `task/ observation/ learning/ deployment/ policy/`로 풀어헤쳐지지 않고
**자신의 바이트 그대로** 하나의 엔트리로 임베드된다. 다시 분해한다면 evidence 번들이
deployment 번들이 이미 봉인한 IR을 재직렬화한다는 뜻이 되며, 직렬화 방식의 어떤 차이든
보이지 않게 될 것이다. 통째로 임베드되므로, `verify`는 그것에 대해 `PolicyBundle::open`을
호출하고 deployment 아티팩트를 여는 것이 이미 수행하는 모든 검사 — IR validator, cross-IR
패스, 가중치 해시, 체인 슬롯 — 를 물려받는다.

spec 27.1 트리의 `scene/`, `training/`, `validation/determinism.json`,
`domain_gap.json`, `hazards.json`, `residual.json`, `signature`는 이 패킷이 쓰지 않는다. 이
패킷이 알지 못하는 엔트리는 `extra_entries`를 통해 함께 실려가므로, 나중에 추가하는 것은
형식 변경이 아니다.

## 3. 왜 매니페스트뿐 아니라 `chain.json`도 있는가

`BundleManifest.hashes`는 *deployment* 아티팩트가 알 수 있는 체인의 부분집합만
싣는다. `asset`, `scene`, `task_graph`, `evaluation`, `hardware` 슬롯이 없으므로
`execution_hash`를 거기서 다시 계산할 수 없다 — 그리고 어떤 리포트를 이 번들에 묶어주는
유일한 것은 그 리포트의 `execution_hash`가 이 번들이 증명하는 값과 같다는 사실뿐이다.
`chain.json`은 그 실행의 직렬화된 `es_ir::hash::HashChain`이며, 두 가지를 모두 제공한다:
모든 리포트와 비교할 `chain.execution_hash()`, 그리고 `--against` 아래에서
`HashChain::diff`를 위한 컴포넌트 단위 필드들이다.

`verify`는 체인을 임베드된 정책 번들의 매니페스트와 슬롯 단위로 교차 검사한다:

| slot | severity on mismatch | why |
|---|---|---|
| `task`, `observation`, `deployment`, `compiler` | error | 양쪽 모두 같은 IR로부터 같은 방식으로 계산한다 |
| `learning`, `policy` | warning | 번들은 *선언된* 그래프(`learning_hash` / `policy_hash`)를 해시하고; 런타임 체인은 `PolicyRuntime`이 실제로 로드한 것(`lowering_hash` / `weights_hash`)을 기록한다. 이 둘은 서로 다른 두 질문에 대한 정직한 두 답이며, 그 갭은 조용히 숨겨진 것이 아니라 알려진 것이다 |
| `runtime`, `dataset` | 정책 번들에는 없음 | 체인으로부터 evidence 매니페스트로 채워진다 |

## 4. 완전성 — 게이트 16

이 게이트가 요구사항마다 검사하는 규칙:

> 요구사항은, 그것에 연결된 적어도 하나의 evidence가 번들 엔트리로 존재하고, 그
> 기록된 `hash`가 그 엔트리 바이트의 blake3와 같고, 그 `execution_hash`가 번들 자신의
> `chain.execution_hash()`와 같을 때 **커버(covered)**된 것이다.

존재하고 해시도 맞지만 *다른* `execution_hash`를 지닌 연결된 evidence는 커버로
보고되지 않고 **stale**로 보고된다: 그것은 다른 machine의 실제 측정값이기 때문이다. stale한
evidence만 가진 요구사항은 커버되지 않은 것이며, `es evidence verify`는 1로 종료한다.

`SafetyCase::validate`는 이 모든 것보다 먼저 실행되어 그래프 자체를 거부할 수
있다: 중복된 id, 존재하지 않는 요구사항을 가리키는 claim이나 traceability 키, 존재하지 않는
evidence를 가리키는 traceability 엔트리, evidence 목록이 아예 없는 요구사항. 유효하지 않은
case로부터 `evidence.esb`를 만드는 것은 거부된다 — 검증 불가능한 아티팩트는 아티팩트가
없는 것보다 나쁘다(spec 26.1: *검증되지 않은 것은 실행되지 않은 것이다*).

## 5. `revalidation_trigger` — "실질적 변경" 표

Spec 27.1은 `revalidation_trigger`를 각 요구사항에 매단다; 이 패킷은 그 대신
**evidence kind**에 매단다. 이유는, 어떤 사실을 무효화하는 것은 그 사실이 뒷받침하는
문장의 속성이 아니라 그 사실이 만들어진 방식의 속성이기 때문이며, 요구사항별 목록은 같은
진실이 두 번째로 낡아버릴 수 있는 자리를 만드는 것이다. 요구사항은 자신의 evidence 중
하나가 영향받을 때 정확히 그만큼 영향받는다.

`ChangedComponent`가 그 단위다(spec 5.3, `HashChain::diff`).

| evidence kind | invalidated by |
|---|---|
| `EvalReport` | `execution_hash` 안의 모든 컴포넌트에 더해 `Asset`, `Scene`, `Evaluation` — 즉 `TaskGraph`를 제외한 전부 |
| `Lock` | `EvalReport`와 동일: 그 리포트가 만들어진 조건을 기록하기 때문이다 |
| `SafetyScenarioSuite` | `Asset`, `Scene`, `Task`, `Observation`, `Learning`, `Policy`, `Deployment`, `Compiler`, `Runtime`, `Hardware` — envelope, 무엇이 명령되는지, 그리고 machine이 그것으로 무엇을 하는지 |
| `GoldenTest` | `Asset`, `Scene`, `Task`, `Observation`, `Compiler`, `Runtime`, `Hardware` — golden은 정책에 대한 것이 아니라 파이프라인에 대한 비트 단위의 주장이다 |
| `HumanReview` | `Asset`, `Scene`, `Task`, `Deployment` — 리뷰어가 읽은 것 |

`TaskGraph`는 아무것도 무효화하지 않는다: 이는 저작 정체성(node id)이며, spec
5.3이 같은 이유로 이를 `execution_hash` 바깥에 둔다. `Dataset`은 `HumanReview`나
`GoldenTest`를 무효화하지 않지만 리포트는 무효화하는데, 이는 `execution_hash` 안에 있기
때문이다.

`es evidence verify a.esb --against b.esb`는 `a.chain.diff(&b.chain)`을 실행하고,
바뀐 컴포넌트마다 위 표가 재실행되어야 한다고 말하는 evidence id들을 출력한다. 이것이
spec 27.1의 "실질적 변경"의 기계적인 절반이다. 이는 권고적(advisory)이다: `--against`는
종료 코드를 절대 바꾸지 않는데, "이 번들이 저 번들과 다르다"는 것은 이 번들의 결함이
아니기 때문이다.

## 6. 검증이 증명하지 않는 것

여기 명시하는 이유는 spec 27.1이 포지셔닝 경고로 끝나며, CLI가 실제보다 더 대단한
것으로 읽혀서는 안 되기 때문이다.

- **서명 없음.** `BundleManifest.signature`는 예약된 슬롯이며, 아무것도 그것을
  채우지도 검사하지도 않는다(spec 25.1: 네이티브 플러그인 서명 검증은 선택적이며 나중
  일이다). 번들 안의 모든 해시는 자기 일관적(self-consistent)이지만, 그중 어느 것도
  *인증(authenticated)*되지 않았다. 파일을 다시 쓸 수 있는 사람은 누구든 그 해시들도 함께
  다시 쓸 수 있다. `verify`는 `signature: unverified`를 출력한다.
- **재실행(replay) 없음.** Spec 27.1은 `es evidence verify`가 replay를 다시
  실행해서 지표를 비교하기도 원하지만, 그것은 M4(spec 28.6)의 몫이다. 오늘은 번들 안의
  리포트들이 번들이 주장하는 그 리포트들이 맞는지만 검사하며, 다시 실행하면 같은 것이
  나올지는 검사하지 않는다. 게이트 17이 바로 이런 이유로 M4 게이트다.
- **적합성 판정 없음.** 이 번들은 기술 문서 작성을 위한 evidence 수집·추적
  도구다. 인증이 아니며, `verify`가 통과했다고 해서 적합성 진술이 되는 것은 아니다(spec
  27.1).
- **논증에 대한 판단 없음.** 모든 요구사항에 evidence가 있다는 것은 그 요구사항이
  옳은지, 그 주장이 타당한지에 대해서는 아무것도 말해주지 않는다. 그것은 `HumanReview`
  evidence kind가 대체하기 위해서가 아니라 기록하기 위해 존재하는 사람의 리뷰다.
