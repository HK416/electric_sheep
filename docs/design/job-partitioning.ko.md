<!-- Korean translation of docs/design/job-partitioning.md. The English file is the working copy; regenerate this when it changes. -->

# 작업 분할 (P14)

§3.4는 *결정적 스케줄링*(정적 분할, 단일 큐)을 결정적 실행 계약의 여덟 조건 중
하나로 나열하며, §18.4는 하드웨어에 걸쳐 작업을 어떻게 나누든 결과가 달라지지 않는
리덕션을 요구한다.

## 규칙

`partition_static(n_items, n_workers)`는 두 인자만의 순수 함수다.

```
base = n_items / n_workers
rem  = n_items % n_workers
chunk i gets base + 1 items for i < rem, else base items
```

범위는 연속적이고, 오름차순이며, 겹치지 않고, `0..n_items`를 정확히 한 번씩
덮는다. 빈 범위는 버려지므로 chunk 개수는 `min(n_items, n_workers)`가 된다.

여분의 항목은 *낮은* chunk에 배분된다 — 높은 쪽이 아니라 — 이는 임의의 선택이지만,
다른 머신에서의 재현이 동일하게 분할되도록 여기서 고정한다.

## 금지 사항

- `n_workers`를 머신에서 유도하는 것(`available_parallelism`, CPU 개수, GPU SM
  개수). 호출자가 이를 넘기며, 재현 시에는 실행 기록에서 그대로 복원한다.
- Work stealing, 동적 청킹, 또는 워커가 자기 항목을 얼마나 빨리 처리했는지에
  의존하는 어떤 분할도 금지한다.
- 리덕션 결과를 완료 순서로 병합하는 것. `par_reduce`는 청크별 결과를 청크
  인덱스로 색인된 슬롯에 모으고 `merge`로 청크 인덱스 순서에 따라 접으므로, 주어진
  `n_workers`에 대해 결과는 스레드 스케줄링과 무관하게 동일하다.

## 워커 개수와 결과

Chunk 경계는 `n_workers`에 의존하므로, `merge`가 결합적일 때만 결과가 워커 개수에
걸쳐 불변이다 (§18.4: 핵심 불변식은 `merge`가 결합적이고 교환적이라는 것). 부동소수점
합은 그렇지 않다 — `es-math`의 `DeterministicAcc`(RFA / binned summation, INV-17)가
이를 결합적이고 교환적으로 만드는 accumulator다. `es-core`는 병합 클로저에 대해
제네릭을 유지할 뿐 float에 대해 알지 못한다.

## 상한

`JobSystem`은 풀을 계속 살려두는 대신, 호출마다 scoped 스레드를 생성한다
(`std::thread::scope`). 여기서 다루는 청크 크기에서는 생성 비용이 무의미하다. 만약
호출당 생성 비용이 프로파일에서 드러나면, 분할 계약을 바꾸지 않고도 같은 API 뒤에
영속적인 풀을 끼워 넣을 수 있다.
