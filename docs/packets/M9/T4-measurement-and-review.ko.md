# M9 T4 — 측정값을 읽고, 리뷰가 플랜 T를 닫는다

스펙: §28.12 파동 3, §13.4(RL 기본값이 증분이 되어야 하는가), §28.9 규칙 2. 유형 A.

## 사양

`docs/reviews/M9.md` + `.ko.md`를 M8 형식으로: 게이트 결과, 플랜 T가 하려던 것과 한 것(R1의
노이즈 표, R6의 줄 수, T1의 비트 단위 적분, T2의 증분 가져오기, T3의 나란한 비교), 발견,
사람의 결정(§13.4 기본값; `EeDelta`의 IK; 두 개의 `ActionSpace` 열거형; 아직 열려 있는 M8
결정 S-1, S-2, 그리고 다음 소스 정책의 장면), 사다리 상태, 판정, 후속 작업. §28.12는
"M9 result" 단락을 얻는다(ko + en, 커밋 하나); `CLAUDE.md`의 상태 단락이 옮겨간다.

## context

```
docs/reviews/M9.md
docs/reviews/M9.ko.md
docs/ARCHITECTURE.ko.md
docs/ARCHITECTURE.md
CLAUDE.md
docs/packets/M9/T4-measurement-and-review.md
docs/packets/M9/T4-measurement-and-review.ko.md
```

## 오라클

리뷰된 커밋에서 `cargo xtask ci`가 green; `check-spec-refs`.

## 수용 기준

모든 숫자가 측정됨 또는 `unverified`로 태그된 채 리뷰가 존재한다; 오케스트레이터가 그것을
쓴다.

## 금지

해시 없는 숫자; 표가 뒷받침하지 않는 판정.
