# M7 U — 측정: T3–T6 이후의 IR 그래프, 커밋된 문서 위에서

스펙: §28.10("정확도에 대한 한 문장": T3·T4·T5가 lowering을 바꾸므로, IR-그래프 정책은
커밋된 문서 위에서 **한 번** 다시 측정된다; 중단 규칙 — T5 이후 held-out이 V18b의 0.625
이하에 머무르면, IR-그래프 튜닝은 끝나고 제품은 외부 경로의 속도가 된다), §28.9 규칙
2–3(무효화된 측정은 지워지지 않고 표시된다; 재현 가능한 지표 없는 성능 주장은 없다),
§12.4(아홉 개의 지표, 결코 `step/s`가 아님), §13.3(움직인 `evaluation_hash`는 새로운
비교이며, 소리 내어 말해진다), §10.5. 유형 D. 확장할 설계 노트:
`docs/design/visible-learning.md`(+ `.ko.md`) — V18b(7.28절)와 V19b(7.29/7.30절) 옆의
7.31절 "구현된 대로 (M7/U)". 정확도 절반은 **T2, T3, T4, T5, T6, T7**에, 쇼케이스 절반은
**R2, R3, R4**에 의존한다; 각 절반은 자신의 입력이 도착했을 때 돌고 어느 커밋으로부터
돌았는지 말한다.

## 질문

모든 T-패킷이 loss를 보고했다. **그중 어느 것이라도 프로젝트가 커밋한 문서들 위에서
`success_rate`를 움직이는가 — 그리고 §28.10 중단 규칙에 의해, IR-그래프 작업은 M7 이후에도
계속되는가?**

## 사양

* **문서들.** Task/Deployment/Evaluation IR: 커밋된 `tests/fixtures/visible-learning/` 세트
  (`evaluation.toml` = 배제된 시드 101–116 위의 16편 + 여섯 스위트; 학습 시드의 문서는
  서버의 `~/artifacts/plan-v/v15/eval-trainseeds.toml`). 데이터: V15의 200개 시연
  (`~/artifacts/plan-v/v15/ds-train`, `v15/baked`에 bake됨 — U2/U3를 위해 `--for-training`으로
  다시 bake). 모든 실행: T4의 행-D 설정(배치 64, lr 4e-4, `warmup_cosine` warmup 250,
  `lr_min` 1e-6, 시드 0, `--resident-gpu`)에서 20,000 옵티마이저 스텝, 레시피를 가진
  `es train`을 통해, 그러므로 각각에 대해 `training.lock`이 존재한다.

  | run | Learning IR | Observation IR | 무엇을 측정하는가 |
  |---|---|---|---|
  | U0 | 커밋된 `learning.toml` | 커밋된 `observation.toml` | T3 + T4만 — **T4의 행-D 체크포인트, 이미 학습됨**(`~/artifacts/plan-v/m7-t4/model-D.safetensors`, `lr_curve_hash c01d5185…`): 패키징하고 평가할 뿐, 재학습하지 않는다 |
  | U1 | `learning-pretrained.toml`(T5) | 커밋된 것 | + ImageNet 백본, T5의 노트가 달리 말하지 않는 한 `frozen = false` |
  | U2 | 커밋된 것 | `observation-augmented.toml`(T6) | + random shift와 brightness/contrast |
  | U3 | `learning-pretrained.toml` | `observation-augmented.toml` | 둘 다 |

  각각에 대해: 배제된 16편에서의, 학습 시드 16편에서의, 그리고 `--jobs 6`을 가진 커밋된
  `evaluation.toml`로부터의 여섯-스위트 스윕의 `success_rate`, 모두 T7의 지연시간 모델
  아래에서(`es policy pack`이 쓰는 대로의 번들의 `expected_latency_ms` — 그것이 무엇인지
  말할 것; 열린 질문 32). `report.json`, `evaluation.lock`, `training.lock`과 체크포인트는
  `~/artifacts/plan-v/m7-u/U<n>/` 아래에.
* **U2/U3는 새로운 비교다.** 그것들의 Observation IR은 다른 문서이므로, 그것을 이름으로
  지정하는 Evaluation IR은 다른 `evaluation_hash`다(§13.3). 표는 두 해시를 모두 출력하고
  노트는 한 문장으로 U2/U3 행이 체인이 아니라 오직 독자의 판단으로만 U0/U1과 비교 가능하다고
  말한다 — perturbation 스위트, 시드, 지표는 같은 바이트다; 관측 문서는 그렇지 않다.
* **적용된 중단 규칙.** 노트의 첫 문단은 U1의 배제된 수치를 0.625에 대해 진술하고 §28.10이
  정하는 결과를 진술한다. 그것은 측정 이후에 쓰이며 이전에는 쓰이지 않는다.
* **사이클의 벽시계 시간.** T2의 수용 사이클(collect 200 → expert gate → train → eval →
  showcase)은 인용될 뿐 다시 돌지 않는다, T2의 수치가 T4/T5가 도착하기 전에 취해진 것이
  아니라면 — 그렇다면 U3 설정에서 한 번 다시 돌리고, 단계별 시간을 T2의 것 옆에.
* **쇼케이스 절반.** U0의 `nominal-00`으로부터(또는 U0가 모든 에피소드에서 실패하면 V19b의
  것으로 — 어느 쪽인지 말할 것): R2의 `--look full`, R3의 `--path pt --spp 64`를 가진
  1280×720의 `es video showcase` 프레임, 그리고 R4가 도착했다면 R4의 `--path pt --spp 4
  --accumulate`(그렇지 않으면 그 행은 `not run: R4 not landed at <commit>`라고 적힌다),
  각각 `python/es/encode_video.py`로 H.264로 인코딩되며, ms/frame을 R-패킷들 자신의 표
  옆에, mp4는 `~/artifacts/plan-v/m7-u/showcase/` 아래에 그리고 요청자의
  `target/plan-u/m7-u/`로 복사되어.
* **여기서는 아무것도 튜닝되지 않는다.** 발산하거나, 시간 초과되거나, 수용에 실패하는 행은
  그 안에 수치가 있는 행이다. 두 번째 시드, 더 긴 실행, 바뀐 lr은 새 패킷이다.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

```
tests/fixtures/visible-learning/training-u*.toml
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M7/U-measurement.md
docs/packets/M7/U-measurement.ko.md
```

레시피 네 개 `training-u0.toml` … `training-u3.toml`(커밋되어 있어서 실행이 저장소로부터
재현 가능하다 — U0의 것은 행-D 설정을 이름으로 지정하지만 그 체크포인트는 T4의 것이다),
설계 노트 절, 이 패킷. **소스 변경 없음.** 도중에 발견된 결함은 여기서 고쳐지지 않고
보고된다.

## 오라클

1. 표의 모든 행은 출력된 것과 일치하는 `evaluation_hash`를 가진 `report.json`과, 요청자의
   머신에서 레시피의 `--dry-run` 아이덴티티와 일치하는 `identity_hash`를 가진
   `training.lock`을 갖는다(`es train --dry-run`은 Python 없이 로컬에서 판정된다).
2. `es eval compare U0/report.json U1/report.json`(그리고 U0 대 U2, U1 대 U3)가 노트에
   출력된다 — 눈으로 읽은 것이 아니라 도구 자신의 유의성 열.
3. `cargo xtask ci`(소스는 움직이지 않는다; 네 레시피의 `train_dry_run` 플랜은 골든이
   아니지만, 레시피는 파싱된다); `cargo xtask check-scope docs/packets/M7/U-measurement.md`.

## 수용 기준

7.31절의 표(네 행 × 세 열 + 스윕의 여섯), 중단 규칙 문단, 사이클 벽시계 시간, 쇼케이스
ms/frame 표와 세 mp4. 각 수치는 그것이 돌았던 커밋과 트리를 이름으로 지정한다.

## 금지

어떤 소스 변경이든; U0를 재학습하는 것; 다섯 번째 설정; 커밋된 두 세트 말고 다른 시드에서
평가하는 것; `docs/ARCHITECTURE*.md`(스펙을 움직이는 것은 리뷰이지 이 패킷이 아니다); 골든.
