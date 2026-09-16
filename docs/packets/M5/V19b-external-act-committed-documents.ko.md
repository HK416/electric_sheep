# 패킷 M5/V19b — 커밋된 문서 위의 외부 ACT

설계 노트: `docs/design/visible-learning.ko.md` 7.29절. 선행: V19(7.27절, 체크포인트와 병합
Observation IR), V18/V18b(7.26·7.28절, 픽스처 결정).

## 맥락

- 허용 범위: 저장소 파일 없음. 실행은 오라클 서버의 `~/artifacts/plan-v/v19b/run.sh`와
  `showcase.sh`, 트리는 `~/Projects/es-main`(`ebde599`의 아카이브).
- 입력: `~/artifacts/plan-v/v19/train-w13/checkpoints/060000/pretrained_model`(V15의 시연 200편으로
  학습한 LeRobot 0.6.1 ACT), `~/artifacts/plan-v/v19/observation-v19.toml`(`observation_hash 07fad282…`),
  커밋된 `task.toml`·`learning.toml`·`deployment.toml`(`acceleration_max = 80`, 커밋 `74b01a3`), V19의
  `eval-w13-{train,holdout}.toml`과 `evaluation-s13.toml`.

## 사양

머지된 트리의 `es policy import-lerobot`으로 체크포인트를 커밋된 배포에 대해 다시 임포트하고, 다른
것은 바꾸지 않은 채 nominal 스위트(학습 시드 1–16, 홀드아웃 시드 101–116)와 홀드아웃 시드의 6스위트
스윕으로 평가한다. 홀드아웃 성공 2편과 모자이크를 렌더한다.

## 오라클

- `es policy import-lerobot --checkpoint … --task task.toml --observation observation-v19.toml
  --deployment deployment.toml --out w13-060000-a80.esb`가 `deployment_hash f2f9a510…`, 즉 커밋된
  픽스처에 대해 `es ir check`가 찍는 해시를 출력한다.
- 세 문서에 `es eval run`; `report.json`이 `passed`를 담는다.
- `es video showcase` / `es video mosaic` + `python/es/encode_video.py`가 mp4 세 편을 만든다.

## 수용

홀드아웃 nominal 문서에서 `passed = true`(임계값 0.5 불변); 영상이 존재하고 표본 프레임이
잡기·운반·놓기를 보여준다.

## 금지

재학습; 문서나 코드의 어떤 변경; 안전 플레인의 비활성화·우회; V19가 보고한 것과 다른 체크포인트 선택.

## 결과 (2026-09-16, RTX 4090, `ES_PYTHON=~/venvs/es`)

| 문서 | `passed` | `success_rate` | `envelope_violation_rate` | 폴백 틱 |
|---|---|---|---|---|
| nominal, 학습 1–16 | true | 1.0000 | 0.232 | 0 |
| nominal, 홀드아웃 101–116 | true | 0.9375 | 0.208 | 0 |
| 6스위트, 홀드아웃 | true | nominal 0.9375 · light_intensity 0.9375 · light_direction 0.8125 · observation_delay 1.0 · torque_noise 0.3125 · backlash 0.9375 | 0.18–0.50 | 4 (light_direction) |

해시: `task eb6efefa…`, `observation 07fad282…`, `learning 17613075…`, `policy 308a1f53…`,
`deployment f2f9a510…`. 영상 `v19b-a80-holdout-nominal-00.mp4`(시드 101), `-nominal-11.mp4`(시드 112),
`-mosaic.mp4`는 `~/artifacts/plan-v/v19b/`와 `target/plan-v/v19b/showcase/`에; 리포트는
`target/plan-v/v19b/reports/`에 미러. 홀드아웃 nominal의 유일한 실패(`nominal-04`, 시드 105)는 집게가
1,800틱 중 1,475틱 동안 열린 파지 실패다.
