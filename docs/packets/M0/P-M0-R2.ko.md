<!-- Korean translation of docs/packets/M0/P-M0-R2.md. The English file is the working copy; regenerate this when it changes. -->

# P-M0-R2 — `canonical_hash` 완전성 상한(completeness ceiling)을 결정하고 강제하기

Spec: §5.3 (해시 체인), §11.2 (`*_hash`), Appendix B.7. `docs/reviews/M0.md`에서
`crates/es-ir/src/hash.rs`에 대해 나온 `[SHOULD]` 지적 사항의 후속 작업. 리뷰 등급 A.

## context

```
crates/es-ir/src/hash.rs
crates/es-ir-types/src/codes.rs
docs/design/hash-canonicalization.md
docs/packets/M0/P-M0-R2.md
```

## spec

M0 리뷰는 `canonical_hash`가 정확히 1-WL 불변량이며, 동일한 여덟 노드를 하나의 8-사이클로
연결한 경우와 분리된 두 개의 4-사이클로 연결한 경우가 충돌(collide)함을 확인했다. 결정된
방향은 **상한을 문서화하는 것이 아니라 끌어올리는 것**이다: 개별화-정제
(individualization-refinement)를 추가한다.

- `refine_from(g, order, pos, colour, budget)` — 고정점까지의 한 번의 색 정제 실행이며,
  `budget` 한 단위를 소비한다.
- `ambiguous_class(colour)` — 구성원이 둘 이상인 색 클래스 중 가장 작은 것이며, 동률은 색
  값으로 판가름한다. 색칠의 속성이지 결코 `NodeId`의 속성이 아니다.
- `canonical_encoding(...)` — 이산적인 색칠은 곧바로 인코딩한다; 그렇지 않으면 모호한
  클래스의 모든 구성원을 차례로 구별된(distinguished) 노드로 시도하고(`distinguish`가 도메인
  분리된 다이제스트로 그 노드를 재색칠한다), 다시 정제하고 재귀한 뒤 사전식으로 가장 작은
  인코딩을 취한다. *모든* 선택지에 대해 최소값을 취하는 것이 바로 Appendix B.7을 보존하는
  요소다.
- 호출당 `REFINE_CAP = 10_000`회의 정제 패스가 있으며, 재귀 전체에 걸쳐 공유된다. 이를
  소진하면 틀릴 수 있는 해시 대신 새로운 `HASH-002` "graph too symmetric to canonicalize"를
  반환한다.
- 모듈과 `canonical_hash`의 문서 주석은 무조건적인 "의미론적 동일성" 대신 조건부 보장을
  명시한다; `docs/design/hash-canonicalization.md`에 알고리즘, 상한, 복잡도 표가 실린다.

## oracle

```
cargo test -p es-ir --features testing --lib hash::
```

## acceptance

- `eight_cycle_differs_from_two_four_cycles`는 양쪽 절반을 모두 검증한다: 두 그래프의 단순
  정제 인코딩이 **같고**(이 픽스처가 실제로 1-WL 반례임을 뜻함), 그 `canonical_hash` 값은
  **다르다**. 두 그래프 모두 재라벨링 불변성을 유지한다.
- `exhausted_refinement_budget_is_reported`는 소진된 budget으로 `canonical_encoding`을
  실행시켜 `HASH-002`를 검증한다.
- 기존의 `hash_independent_of_node_ids`와 `param_change_changes_hash` proptest, 그리고
  다른 모든 `hash::` 테스트가 변경 없이 그대로 통과한다.
- `HASH-002`는 제목과 섹션을 갖춘 채 `codes::CODES`에 들어 있으므로, `codes::` 테스트도
  그대로 통과한다.

## forbidden

`context` 밖의 모든 파일. `CANON_TAG`나 이미 이산적인 색칠의 인코딩을 변경하는 것 —
모호하지 않은 그래프에 대해 저장된 해시는 움직이면 안 된다. 완전성 주장을 "탐색이 상한
이내에 끝나는 한"보다 넓게 확장하는 것. 상한에 도달했을 때 해시를 반환하는 것.
