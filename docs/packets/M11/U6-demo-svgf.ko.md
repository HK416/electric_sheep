# M11 U6 — 데모의 경로 추적 행에 SVGF, 학습 시드 둘

스펙: §28.14 규칙 5, 7과 파동 3; `visible-learning.md` 7.37(U5: held-out 0.25, 학습 시드 0.50,
CUDA 학습 한 번); M10 리뷰 S-2(cycle 레시피는 오늘의 103,881-프레임 수집에 비해 덜 학습됨)와
S-4(단일 CUDA 실행은 ±0.3을 짊어진다). X6에 의존. 유형 D.

## 질문

**U5의 `seed = "tick"` 위에 `svgf = true`를 얹으면, 데모의 경로 추적 ACT 행이 움직이는가, 그리고
학습 시드 둘의 편차보다 더 크게 움직이는가?**

## 명세

* 문서: `task-pt-tick-svgf.toml`과 그 observation/evaluation 형제 문서(X6), 그리고 W1a가 U5의
  것을 유도한 방식대로 유도된 augmented observation과 evaluation.
* `task-pt-tick-svgf` 아래서 W1a의 파이프라인을 돌린다: 수집 200(시드 1, `--expert`) →
  `training-u5.toml`의 설정(변수 하나: 문서)으로 **학습 시드 둘**에서 학습 → held-out과
  학습-시드 평가. U5도 두 번째 학습 시드로 다시 돌려서 U5가 시드 둘을 갖게 한다. 서버, GPU
  lock, nohup + 마커를 `~/artifacts/plan-x/u6/` 아래.
* 먼저 패리티 게이트, W1a가 했듯이: collector와 evaluator의 tick-0..2 프레임이 md5 동일.

## context

```
tests/fixtures/visible-learning/**
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M11/U6-demo-svgf.md
docs/packets/M11/U6-demo-svgf.ko.md
```

## 오라클

1. 서버에서의 패리티 게이트.
2. 표: U5 시드 0(커밋됨), U5 시드 1, U6 시드 0, U6 시드 1 — 스위트별 held-out `success_rate`,
   학습-시드 성공률, 최종 손실, 해시.
3. check-scope, verify-goldens; 원본 프레임 트리는 읽은 뒤 삭제.

## 수용

`visible-learning.md` 7.38(+ko)에 표와 한 문장: SVGF가 행을 두-시드 편차보다 더 크게
움직이는가.

## 금지

레시피 변경(스텝 수 포함: S-2는 이 패킷이 아니라 소유자의 결정); 코드 변경.
