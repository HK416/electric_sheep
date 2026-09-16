# M5 V18 — 선언된 케이던스 아래의 엔벨로프: 결정이 아니라 절제 실험

설계 노트: `docs/design/visible-learning.md` **7.26절**, 그리고 7.25절(V17, 놓기를 예측하던
청크가 실행된다), 미해결 질문 25(이 패킷), 미해결 질문 22(놓기는 관측 가능하지 않다, V16/V17).
스펙: §9.3, §9.4, §10.3, §1.4. 의존: V17.

## 질문

V17은 두 경로 모두 선언된 5 Hz 재계획에서 청크의 0..9행을 순서대로 실행하게 만들었다. 그
케이던스 아래에서 `SafetyPlane`에 닿는 명령 흐름은 제어 틱당 0.02 – 0.03 rad의 이동을
요구하는데 `acceleration_max = 20 rad/s²`는 틱당 스텝 변화를 0.008 rad로 묶으므로
`envelope_violation_rate`가 0.998이다 — 틱의 약 90 %에서 `violation.acceleration`, 대략
절반에서 `violation.velocity` — 그리고 `EnvelopeViolationRate` 워치독(`max_frac 0.9`, 창
200)이 매 실행의 약 1/10 동안 `hold_position`을 걸어 잠근다. 미해결 질문 25는 움직일 수 있는
끝을 셋 꼽았다. **(a)** — STS3215 서보가 실제로 낼 수 있는 값까지 `acceleration_max`를
넓힌다 — 는 오너가 내리는 Deployment IR 결정이고, 오너는 결정 전에 데이터를 원한다. 이
패킷이 그 데이터를 만든다. 커밋된 픽스처는 하나도 바뀌지 않고 재학습도 없다.

## 변수 하나

`tests/fixtures/visible-learning/deployment.toml`의 *절제 사본* 안에서
`body.safety.acceleration_max`(관절 6개 전부)를 네 단계로: **20**(현재 값, V17을 비트
단위로 재현해야 하는 기준선), **40**, **80**, 그리고 **80에 `velocity_max`도 3.0 → 4.4
rad/s로 넓힌 단계**(픽스처 자신의 주석이 인용하는 서보의 무부하 정격) — 넷째 단계는 가속도가
걸림돌에서 빠진 뒤 속도가 병목이 되는지를 보여준다. 나머지는 모두 V17의 재측정 그대로다:
V15의 `model-40000.safetensors`(`~/artifacts/plan-v/v17/weights.sha256`에 고정된 바이트),
현재의 Task/Observation/Learning 문서, 1,800스텝 예산, 학습 시드 1–16과 홀드아웃 시드
101–116, 단계당 한 번(V17의 비트 동일 반복은 여기서는 필요 없다). `EnvelopeViolationRate`
워치독은 선언된 그대로 유지한다 — 측정 대상의 일부이지 결코 비활성화하지 않는다(INV-12).
합법적인 유일한 수는 절제 사본 안에서 한계를 넓히는 것뿐, 커밋된 파일을 건드리거나 플레인을
우회하는 것이 아니다.

## 단계를 만드는 방법

임의의 Deployment IR 파일에 대해 정책을 패킹하는 CLI 경로는 없다 — `es policy pack`은 입력
번들 자신의 deployment를 그대로 가져가고(`crates/es/src/cmd/policy.rs`의 `pack()`),
`write_demo_bundle`(`crates/es/tests/cli.rs`)은 항상 디스크의
`tests/fixtures/visible-learning/deployment.toml`을 읽는다. V17도 커밋된 파일의
`inference_budget`을 옮길 때 같은 벽에 부딪혔고, 그때(갱신된) 커밋 문서로부터 미학습 번들을
다시 만들었다. 이 패킷은 커밋된 파일을 갱신할 수 없으므로, 각 단계는: (1) 커밋된
`deployment.toml`을 옆으로 복사해 두고, (2) 여섯 항목짜리 `acceleration_max` 배열(넷째
단계는 `velocity_max`도)을 작은 텍스트 패치로 제자리에서 고치고
(`target/plan-v/v18/scripts/patch_deployment.py` — 이름 붙은 대괄호 배열 하나를 겨냥한
정규식, 문서의 다른 곳은 전혀 움직이지 않는다), (3) V17의 `pb.sh`/`pc.sh`가 미학습 번들을
현재 문서에서 만들 때 쓴 것과 같은 오라클인
`dataset_bake_writes_safetensors_and_a_manifest`를 다시 돌려 절제된 deployment를 담은
번들을 얻고, (4) 평가 전에 커밋된 파일을 즉시 복원하고, (5)
`es policy pack --policy <untrained>.esb --weights model-40000.safetensors --out <trained>.esb`
를 돌린다 — 입력 번들이 이제 그것을 담고 있으므로 절제된 deployment가 그대로
넘어간다. 이 과정에 Rust 변경은 전혀 필요 없었다: `es-ir::deployment`의 검증기에는
`acceleration_max`나 `velocity_max`를 로봇 성능 한계에 묶는 규칙이 없어서(`codes::DEP_031`은
선언만 되어 있고 구현되지 않았다 — `crates/es-ir/src/deployment.rs:27-28` — 아직 존재하지
않는 Cross-IR 패스의 몫이다) 네 단계 모두 첫 시도에 검증되고 패킹된다.

## 범위

* **`context`** — `docs/design/visible-learning*.md` 7.26절과 미해결 질문 25,
  `docs/packets/M5/V18-envelope-ablation*.md`, `target/plan-v/v18/`(추적 안 됨: 패치 스크립트,
  실행 스크립트, 래치 겹침 스크립트, 미러링된 보고서와 표), `~/artifacts/plan-v/v18/` 아래의
  서버 스크립트.
* **`forbidden`** — 모든 IR 스키마, `crates/es-safety`, `crates/es-ir`,
  `tests/fixtures/visible-learning/deployment.toml` 자체(커밋된 파일), 수용 임계값(0.5,
  불변), 재학습, V15의 가중치, 7.19–7.25절, `docs/ARCHITECTURE*.md`, 어떤 단계든 6스위트
  섭동 스윕이나 쇼케이스 영상으로 승격하는 것 — 이 절제 실험을 넘어 한 단계를 승격하는 것은
  패킷이 알려주려는 픽스처 결정이지 패킷이 내리는 결정이 아니다.
* **`INV-17`** — 새 트레잇도 새 확장점도 없고, ("단계를 만드는 방법" 대로) 새 Rust도 전혀
  없다: 이 절제 실험은 파이썬 스크립트 두 개와 이미 존재하는 명령의 호출 네 번이다.
* **`INV-12`** — 비활성화한 것은 없다. 모든 틱은 여전히 버퍼 → `plane_chunk` →
  `SafetyPlane::validate` → `ctrl`을 지나고, 0.998 래치율을 설명하려는 그 워치독은 거의 걸리지
  않는 단계에서도 모든 단계에서 켜진 채로 유지된다.

## 오라클

1. **기준선이 V17을 비트 단위로 재현한다.** 단계 20에서 `es policy pack`이 찍는
   `weights_hash`, `lowering_hash`, `task_hash`, `observation_hash`, `learning_hash`,
   `policy_hash`가 V17 자신의 것과 같다(`~/artifacts/plan-v/v17/pb.log`, `pc.log`). 두
   스위트의 `report.json`의 `evaluation_hash`도 V17과 같다. `report.json`의 모든 스칼라와
   `failure_mode_histogram` 전체가 V17이 저장한 `report.json`과 같다
   (`~/artifacts/plan-v/v17/e40000-{train,holdout-a}/report.json`).
2. **모든 단계의 Deployment IR이 검증되고 패킹된다.** 네 단계 모두 `es policy pack`의 종료
   코드가 0이다. 여기서 실패했다면 가장 작은 Rust 수정과 테스트 하나를 강제했겠지만(필요
   없었다 — "단계를 만드는 방법" 참고) 이번에는 그렇지 않았다.
3. **이 절제 실험은 커밋된 저장소와 고정된 가중치만으로 재현 가능하다** — 아래 "이것을
   재현하려면"을 보라. 독자는 거기 나열된 명령으로 모든 단계를 다시 돌릴 수 있다.
4. **`cargo xtask ci`** 통과 — fmt, clippy `-D warnings`, 컨텍스트 예산, 레이어링, 스펙 참조,
   골든. 골든 파일도, IR 스키마도, 픽스처도 움직이지 않는다.

## 수용 기준

기준선의 해시와 숫자가 V17과 정확히 일치한다(산출물 1). 네 단계 표, 기준선의 릴리스
구간/워치독 래치 겹침 확인, 그리고 설계 노트 절과 패킷 문서가 측정값과 함께 존재한다(산출물
1–4). 어느 단계든 홀드아웃 `success_rate`가 0.5에 도달하면 그것을 눈에 띄게 보고하고 그
단계에서 멈춘다 — 6스위트 스윕도, 쇼케이스 영상도, 두 번째 변수도 없다. 픽스처 결정은
여전히 오너의 몫이다. `cargo xtask ci`가 통과한다. `deployment.toml`은 V17이 남긴 그대로
바이트 단위로 같다.

## 이것을 재현하려면

오라클 서버(RTX 4090)에서, `git archive HEAD | gzip | ssh ...`로 실어 보낸 트리에서(절대
`git clone` 아님), `torch`와 `mujoco`를 가진 파이썬으로 `ES_PYTHON`을 설정하고:

```
cargo build --release -p es --features render

# 기준선(단계 20): patch 없이 커밋된 문서 그대로 다시 빌드
cargo test --release -p es --features render --test cli \
  dataset_bake_writes_safetensors_and_a_manifest -- --nocapture
cp /tmp/es-cli-test-dataset-bake-*/policy.esb untrained-L20.esb
es policy pack --policy untrained-L20.esb --weights model-40000.safetensors \
  --out trained-L20.esb

# 넓힌 단계(40, 80, 또는 80+속도): 스크래치 사본을 patch, 재빌드, 복원, pack
cp tests/fixtures/visible-learning/deployment.toml /tmp/deployment.orig.toml
python3 patch_deployment.py tests/fixtures/visible-learning/deployment.toml \
  acceleration_max=40.0            # 또는 80.0; 넷째 단계에는 velocity_max=4.4도 추가
cargo test --release -p es --features render --test cli \
  dataset_bake_writes_safetensors_and_a_manifest -- --nocapture
cp /tmp/es-cli-test-dataset-bake-*/policy.esb untrained-L40.esb
cp /tmp/deployment.orig.toml tests/fixtures/visible-learning/deployment.toml   # 먼저 복원
es policy pack --policy untrained-L40.esb --weights model-40000.safetensors \
  --out trained-L40.esb

# 단계마다 두 스위트를 평가
es eval run --config eval-trainseeds.toml --policy trained-L40.esb \
  --scene tests/fixtures/mjcf/so101_pick_place.xml --out eL40-train --frames fL40-train
es eval run --config eval-holdout.toml --policy trained-L40.esb \
  --scene tests/fixtures/mjcf/so101_pick_place.xml --out eL40-holdout --frames fL40-holdout
```

`eval-trainseeds.toml`과 `eval-holdout.toml`은 V15 자신의 것으로 바뀌지 않았다
(`~/artifacts/plan-v/v15/`). `patch_deployment.py`는
`target/plan-v/v18/scripts/patch_deployment.py`로, 이름 붙은 배열 하나 또는 둘에 대한
자기완결적 텍스트 패치다. 서버에서 실제로 돌린 스크립트는
`target/plan-v/v18/scripts/{patch_deployment.py,run-level.sh,latch_overlap.py}`다.

## 측정값

오라클 서버(RTX 4090), `es`에는 `~/venvs/es-lerobot-cuda`(torch 2.11.0+cu129, 이 패킷 자신의
서버 지침대로 — 이것만으로 왜 `execution_hash`가 움직이고 다른 것은 움직이지 않는지는 설계
노트 7.26절 참고), 2026-09-16, 트리 `~/Projects/es-v18`. 산출물은
`~/artifacts/plan-v/v18/` 아래, 작은 결과물은 `target/plan-v/v18/`로 미러링했다
(`probe-all.txt`, `latch-overlap.txt`, `reports/report-<단계>-<스위트>.json`,
`deployment-<단계>.toml`, `scripts/`). 전체 표, 해시 인용, 판정은 설계 노트 7.26절에 있고,
두 핵심 답은:

1. **`acceleration_max = 40`만으로 이미 정지 규칙을 넘는다.** 기준선 다음 첫 단계에서 벌써
   홀드아웃 `success_rate`가 **0.625**에 도달하고(기준선은 0.0625) 80에서도 유지되며,
   `envelope_violation_rate`는 계속 떨어진다 — 0.998 → 0.90–0.92 → 0.56–0.64 — 그리고
   워치독이 래치하는 실행 비율은 약 1/10에서 1 % 미만으로 떨어진다. 어떤 단계도 6스위트
   스윕이나 쇼케이스 영상으로 이어지지 않는다: 정지 규칙이 임계값 도달 시점에 발동하는
   것은 정확히 이 절제 실험이 데이터로 남고 픽스처 변경이 되지 않게 하기 위해서다.
2. **`acceleration_max`와 함께 `velocity_max`를 넓히면 측정 가능하게 더 나빠지고, 히스토그램이
   그 메커니즘을 이름 붙여준다.** `velocity_max = 4.4`에서 틱당 허용량
   `velocity_max * dt = 0.088` rad은 `action_rate.first_diff_max = 0.08` rad을 넘어서는데,
   이는 픽스처 자신의 주석이 `velocity_max <= 3.0`에서는 결코 걸리지 않는다고 적은 경계다.
   그리고 다른 어느 단계에도 없는 새 위반 종류 `violation.rate_limit`이 나타난다. 합친
   단계는 두 스위트 모두에서 `acceleration_max = 80` 단독보다 나쁜 점수를 낸다. 오너의
   선택지 (a)는 처음 생각한 두 기본값보다 더 좁다: `velocity_max`가 아니라
   `acceleration_max`를 움직여라, 그리고 40만으로도 중요한 일은 이미 끝난다.

40단계와 80단계의 홀드아웃 `success_rate`는 **0.625**로 불변인 0.5 수용 임계값 위다 — 정지
규칙에 따라 눈에 띄게 보고한다. 커밋된 `deployment.toml`은 바뀌지 않았다. 권고와 그 물리적
대가(픽스처의 주석이 20 rad/s²에서 이끌어낸 로터 토크 여유가 80에서는 대부분 소진된다)는
설계 노트 7.26절에서 오너가 저울질할 몫이다.
