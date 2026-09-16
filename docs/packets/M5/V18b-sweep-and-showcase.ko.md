# 패킷 M5/V18b — V18 단계에서의 Evaluation IR 전체 스윕과 쇼케이스 영상

설계 노트: `docs/design/visible-learning.ko.md` 7.28절. 선행: V18(`acceleration_max` 절제 실험,
7.26절), V9(쇼케이스 파이프라인, 7.17절).

## 맥락

- 허용 범위: 저장소 파일 없음. 서버 트리 `~/Projects/es-v18b`, 산출물 `~/artifacts/plan-v/v18b/`,
  작은 결과는 `target/plan-v/v18b/`에 미러(추적 안 함).
- 입력: V18이 패킹한 번들 `trained-L40.esb`·`trained-L80.esb`(V15의 `model-40000.safetensors`,
  바이트 동일), 체크인된 `tests/fixtures/visible-learning/evaluation.toml`, V15의 `eval-holdout.toml`.

## 사양

체크인된 Evaluation IR을 그대로 — 선언된 모든 스위트, 자체 시드 계획(홀드아웃 16편, 시드 101–116),
예산 1800 — `acceleration_max` 40과 80에서 돌려 `passed`와 스위트별 지표를 보고한다. 두 단계에서
V18의 홀드아웃 nominal 숫자를 재현한다. 통과한 단계에서 홀드아웃 성공 2편과 16편 모자이크를 V9
파이프라인으로 렌더한다.

## 오라클

- 단계마다 `es eval run --config tests/fixtures/visible-learning/evaluation.toml --policy
  trained-L<N>.esb --scene tests/fixtures/mjcf/so101_pick_place.xml --jobs 4 --out … --frames …
  --traj …`; `report.json`이 `passed`를 담는다.
- 홀드아웃 nominal 재실행이 V18의 `report.json`(`evaluation_hash 40623e01…`, 모든 지표와
  `failure_mode_histogram`)을 비트 단위로 재현한다.
- `es ir check task.toml observation.toml learning.toml deployment-L80.toml`이 커밋 `74b01a3` 이후의
  픽스처와 같은 `deployment_hash`를 찍는다(오케스트레이터 확인: 둘 다 `f2f9a510…`, 40 사본은
  `38e56f4a…`).
- `es video showcase` / `es video mosaic`가 만든 mp4의 프레임 수를 `ffprobe`가 확인한다.

## 수용

채택된 단계에서 `passed = true`; 홀드아웃 재현이 비트 동일; 영상 세 편이 존재하고 표본 프레임이
잡기·운반·놓기를 보여준다.

## 금지

`docs/`, `tests/`, 어떤 픽스처든 편집(픽스처 결정은 오너의 것이며 병행해서 내려졌다, 커밋
`74b01a3`); 재학습; 안전 플레인의 비활성화·우회; 40 단계 숫자를 "택하지 않은 대안" 이상으로 승격.

## 결과 (2026-09-16, RTX 4090, `ES_PYTHON=~/venvs/es-lerobot-cuda`)

| 단계 | `passed` | nominal | light_intensity | light_direction | observation_delay | torque_noise | backlash |
|---|---|---|---|---|---|---|---|
| 40 | true | 0.625 | 0.3125 | 0.4375 | 0.25 | 0.125 | 0.5625 |
| **80 (채택)** | **true** | **0.625** | 0.625 | 0.5 | 0.0625 | 0.0625 | 0.625 |

두 단계 모두 `evaluation_hash e52e8360…`(배포는 여기 들어가지 않는다); `execution_hash c2322c2c…`(40),
`52dfeba9…`(80). 홀드아웃 nominal은 두 단계 모두 V18을 재현. 영상: `v18b-L80-holdout-nominal-00.mp4`
(시드 101, 206틱), `v18b-L80-holdout-nominal-12.mp4`(시드 113, 183틱), 둘 다 1280×720 30 fps;
`v18b-L80-holdout-mosaic.mp4`(96×96 관측 4×4 타일, 1,800 프레임 50 fps). 서버
`~/artifacts/plan-v/v18b/showcase/`, 로컬 `target/plan-v/v18b/showcase/`; 전체 표와 명령은
`target/plan-v/v18b/RESULTS.md`.

남긴 것: 80 단계 렌더가 같은 GPU에서 도는 동안 40 단계를 재현하니 숫자는 같고 `execution_hash`만
달랐다(`745c9777…` 대 단독 `c2322c2c…`) — 열린 질문 26.
