<!-- Korean translation of docs/design/hash-canonicalization.md. The English file is the working copy; regenerate this when it changes. -->

# `canonical_hash`를 위한 그래프 정규화 — 설계

Spec refs: §5.3 (해시 체인), §11.2 (`*_hash`), Appendix B.7 (재라벨링 불변식).
패킷: `docs/packets/M0/P-M0-R2.md`. 코드: `crates/es-ir-types/src/hash.rs`.

이 모듈은 `docs/packets/M8/P-M8-R6.md`에서 (spec 1.5 컨텍스트 예산 때문에) `es-ir`에서
`es-ir-types`로 내려갔고 `es_ir::hash`로 그대로 재수출되므로, 이 문서의 모든 경로와
커밋된 모든 해시는 변함이 없다.

## 문제

`canonical_hash`는 *의미론적(semantic)* 동일성이어야 한다: 두 그래프가 노드 재명명(renaming)을
제외하면 같은 그래프일 때에만 같은 해시를 얻는다. 기존 구현은 색 정제(colour refinement)만으로
이루어져 있었다(1차원 Weisfeiler-Leman): 각 노드를 kind + params로 색칠한 뒤, 어떤 셀도 더 이상
분할되지 않을 때까지 그 노드의 간선들에 대한 `(direction, near port, neighbour colour, far
port)`의 정렬된 다중집합(multiset)으로 반복적으로 재색칠한다. 그런 다음 노드 색의 정렬된
다중집합과 색으로 라벨링된 간선의 정렬된 다중집합을 인코딩하여 해시한다.

1-WL은 건전한(sound) *불변량(invariant)*이다 — 동형(isomorphic)인 그래프는 항상 일치한다 —
그러나 증명 가능하게 불완전(incomplete)하다. M0 리뷰의 반례는
`hash::tests::eight_cycle_differs_from_two_four_cycles`에 있다: 동일한 `Const` 네 개와 동일한
`Add`(포트 `a`/`b`) 네 개를 하나의 8-사이클로 연결한 경우와, 같은 여덟 노드를 서로 분리된 두
개의 4-사이클로 연결한 경우다. 두 그래프 모두 모든 노드가 양방향으로 규칙적(regular)이므로
정제가 어떤 셀도 분할하지 못하고, 두 인코딩은 바이트 단위로 동일해진다. 서로 다른 두 태스크
그래프가 하나의 `task_hash`를 공유하게 되는 것이며, 이는 곧 조용한(silent) 재현성 버그다.

## 알고리즘: 개별화-정제(individualization-refinement)

`canonical_hash`는 이제 단순한 불변량이 아니라 정규 *형태(form)*를 계산한다:

1. 1-WL 고정점(fixpoint)까지 정제한다 (`refine_from`).
2. 색칠이 이산적(discrete)이면(모든 색이 유일하면) 그것을 인코딩하고 해시한다.
3. 그렇지 않으면 **모호한 클래스(ambiguous class)**를 고른다: 구성원이 둘 이상인 색 클래스 중
   가장 작은 것이며, 동률은 색 값 자체로 판가름한다 (`ambiguous_class`). 두 기준 모두 색칠의
   속성이지 결코 `NodeId`의 속성이 아니므로, 그래프를 어떻게 재라벨링하더라도 항상 같은
   클래스가 선택된다.
4. 그 클래스의 **각** 구성원에 대해 차례로: 그 노드만 재색칠하고
   (`distinguish` = `H(INDIV_TAG || colour)`), 다시 정제한 뒤 재귀한다.
5. 모든 분기(branch) 중 사전식으로 가장 작은 인코딩을 취한다.

모든 구성원을 시도하고 최소값을 취하는 것이 바로 결과를 우연히 어느 노드가 선택되었는지와
무관하게 만드는 요소이며, 그 덕분에 Appendix B.7이 여전히 성립한다: 동형인 그래프들은
대응하는 분기 집합을 열거하므로 결국 같은 최소값에서 일치한다. 노드 id는 여전히 인코딩에
절대 기록되지 않는다 — 색 벡터를 인덱싱하는 데에만 쓰인다.

## 상한(cap)

`canonical_hash` 호출 하나당 `REFINE_CAP = 10_000`회의 정제 패스가 있으며, 이는 재귀 전체에
걸쳐 공유된다. 이를 소진한 그래프는 해시 대신 `Diagnostic` **`HASH-002` "graph too symmetric
to canonicalize"**를 반환한다. 거부(refusing)만이 유일하게 안전한 실패 방식이다: 잘못된
해시는 몇 달 뒤에야 드러나는 재현성 버그지만, 진단(diagnostic)은 지금 당장의 빌드 오류이기
때문이다.

## 복잡도

정제 패스 한 번은 `O(n · |E| · n)`이다 — 라운드마다 노드당 전체 간선을 훑는다(이전과 동일;
M0 리뷰에서 `[HUMAN]` 항목으로 기록되었고, 저작된(authored) 그래프에 대해서는 여전히 고칠
가치가 없다). 호출당 정제 패스 수:

| 그래프 | 패스 |
|---|---|
| 단순 정제만으로 이산적이 되는 경우(흔한 경우: kind나 params가 서로 다름) | 1 |
| 크기 `k`인 모호한 클래스가 하나이고, 한 번의 개별화 후 이산적이 되는 경우 | `1 + k` |
| 중첩된 모호성, 깊이 `d`, 클래스 크기 `k₁ … k_d` | `O(∏ kᵢ)` |

따라서 최악의 경우는 지수적(exponential)이며, 이것이 상한이 존재하는 이유다. 실제로 저작된
IR 그래프는 수백 개 노드 수준이고 모호성은 드물다 — 클래스가 정제를 견디고 살아남는 것은
여러 노드가 *구조적으로* 서로 바꿔 쓸 수 있을 때뿐이며, 이는 대개 실제로 서로 바꿔 쓸 수
있다는 뜻이다.

## 이제 "의미론적 동일성"이 보장하는 것

- **건전함(무조건적).** 동형인 그래프 — 같은 구조, 같은 kind, 같은 params, 같은 포트 이름,
  같은 경계 순서, 임의의 `NodeId` 할당 — 는 항상 같은 해시를 낸다. `hash_independent_of_node_ids`
  proptest로 증명된다.
- **완전함(조건부).** 탐색이 `REFINE_CAP` 이내에 끝나기만 하면 동형이 아닌 그래프는 서로 다른
  해시를 낸다 — 이것이 정규 형태(canonical form)가 주는 보장이다. 상한을 넘으면 답은
  `HASH-002`이며, 틀릴 수 있는 해시가 나오는 일은 결코 없다.
- **여전히 주장하지 않는 것.** 여기 있는 어떤 것도 같은 해시를 갖는 두 그래프가 같은 것을
  계산한다고 말하지 않는다: `f32` 필드는 인코딩에서 `f64`로 확장되므로(`CanonWriter::f32`),
  런타임 동작을 바꾸는 정밀도 변경이 저장된 해시를 무효화하지 않는다.
