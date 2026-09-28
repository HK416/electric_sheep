<!-- Korean translation of docs/packets/M12/YV-verification.md. The English file is the working copy; regenerate this when it changes. -->
# M12 Y-V — 플랜 Y가 실제로 도는 것을 아직 봐야 하는 것

플랜 Y(`plan-y.md`, 설계 노트 `docs/design/editor-redesign.md`)는 실제 학습 실행을 단 한 번도
거치지 않고 구현됐다: 이 머신에는 MuJoCo와 LeRobot이 있는 Python 환경이 없었고, 오너는
그 기간 동안 원격 서버를 배제했다(2026-09-28). Python이 필요한 오라클은 모두 출력된
이유와 함께 스킵된다. 이 목록은 그 스킵들이 검증하지 못하고 남긴 것을, 돌려 볼 가치가
있는 순서로 정리한 것이다. **어디서 도는가 — 이 PC의 로컬 venv인지 서버인지 — 는 오너의
결정이고, 그 결정 없이는 여기 있는 어떤 것도 서버에서 시작할 수 없다.**

## 0. 전제 조건

- `mujoco`, `torch`(짧은 프리셋을 뺀 나머지에는 CUDA), `lerobot`이 있는 Python, `ES_PYTHON`으로
  이름함. 확인: `es --check-deps --json` → `modules`가 모두 `true`.
- 렌더러와 함께 빌드된 `es`: `cargo build -p es --features render`(기본 빌드는 `"render": false`를
  보고하고, 시작 화면은 두 큐브 카드를 모두 비활성으로 유지한다). `render`가 `es`의
  기본 feature가 돼야 하는지는 열려 있는 오너 결정이다(M12 리뷰를 보라).
- 체크아웃에서 시작한 에디터(`target/<profile>/es-editor.exe`), 그래야 `templates/`와 자기
  곁의 `es`를 찾는다.

## 1. 두 큐브 템플릿이 끝까지 돈다

각 카드에 대해 — `cube-into-bin`(카메라만, LeRobot 경로, `cycle-vision.toml`)과
`cube-into-bin-hint`(큐브 자세가 주어짐, IR 경로, `cycle.toml`) — 시작 화면에서 프로젝트를
만들고, **short**를 고르고, Start를 누르고, 끝날 때까지 둔다.

합격: 모든 단계가 끝나고(수집 → 전문가 점검 → 학습 → 평가 → 전시), ⑤가 저절로 열리고,
실행 폴더에 `loop.jsonl`, `eval/report.json`, `eval/episodes.json`이 있다. 카드마다 기록할
것: 홀드아웃 성공률, `envelope_violation_rate`, 단계별 wall-clock. 어떤 성공률도 필수는
아니다 — 카메라 전용 카드는 이 과제를 통과한 적이 없다(설계 노트 4절); 수치는 나오는
대로 보고한다.

그런 다음 적어도 힌트 카드에서 **medium**을, M7의 U 시리즈와 비교할 수 있는 수치를 보기
위해.

## 2. `episodes.json`이 `report.json`과 일치한다

`ES_PYTHON=… cargo test -p es --test cli eval_run_episodes_json_agrees_with_the_report`가
스킵 대신 돈다. 그리고 항목 1의 실행 폴더들에서: 모든 스위트에 대해, `episodes.json`으로
센 성공 수 ÷ 그 행 수 = 보고서의 `success_rate`와 정확히 같다.

## 3. "지금까지 배운 것을 평가하기"

실행을 시작하고, 첫 체크포인트 표시를 기다리고, 버튼을 누른다. 예상되는 위험(Y12): 학습
도중 강제 종료한 뒤에는 `training.lock`이 없을 수 있고, `es loop cycle --from eval`이
거부할 수 있다. 합격: 평가가 가장 새 표시 위에서 돌고 ⑤가 열리거나, 거부가 이해되고
버튼이 바뀐다(예를 들어: 표시가 있을 때까지 비활성, 또는 강제 종료된 학습을 `--from eval`이
받아들이게 하는 cycle 변경) — 리뷰에 기록되는 결정.

또한 Windows에서 **멈추기가 프로세스 트리 전체를 끝내는지** 확인한다: `LaunchModel::kill`은
`es.exe`를 끝내고, 그것이 시작한 Python 트레이너는 살아남을 수 있다(Y12). 합격: 멈추기
뒤에 그 실행의 `python` 프로세스가 하나도 남아 있지 않다; 하나라도 남으면 실행 모델을
(job object로) 고치되, 자신만의 패킷으로.

## 4. 신호등의 임계값

항목 1의 성공한 실행 하나가 남긴 진행 지점(`step, loss, samples/s`, 도착 시각)을 픽스처에
기록하고, `crates/es-editor/src/model/health.rs`에
`a_recorded_successful_run_is_green_throughout`을 추가하고, 어떤 합성 테스트도 약화시키지
**않은 채로** 통과할 때까지 `THRESHOLDS`를 조정한다. 고른 값과 그것이 나온 실행을 테스트에
기록한다.

## 5. 게시가 실행에 치르게 하는 비용

에디터가 시작하는 모든 실행은 `--telemetry-image-every 50`을 더한다(Y12). 같은 실행
설정에서 그것을 켠 경우와 끈 경우로 수집과 학습 단계를 측정한다(`editor-shell.md` 13절
게이트 9의 방법). 합격: 1 % 미만(사양 23.4의 게이트); 아니면 간격을 늘린다.

## 6. 분 단위의 길이 프리셋

사용한 머신에서, 카드마다 short / medium / long의 wall-clock. 시작 화면이 추정치를 보여야
한다면, (절대 짐작하지 않고) 이 수치들로부터 더한다.

## 7. 두 카드가 같은 시연을 수집한다

Y5b는 카메라 전용 수집 번들에 제어 주기 하나(20 ms)의 선언된 지연을 줘서, 시연기가 IR
경로 번들 아래서와 정확히 똑같이 돌게 했다. 확인: 같은 시드와 에피소드 수에서 두 카드의
수집 데이터셋이 동일한 `qpos`/`ctrl` 열을 갖는다.

## 8. 손으로 하는 전체 흐름

Windows에서, 처음 쓰는 사람이 에디터를 보는 그대로: 시작 → 템플릿 카드 → Create → Start
(short) → ③과 ④를 지켜봄 → ⑤ → 실패한 에피소드 재생 → 다시 실행. 두 언어 모두로 각
화면의 스크린샷을 `target/plan-y/yv/` 아래에, 그리고 읽는 사람이 추측해야 했던 자리를
모두 후속 사안으로 적어 둔다.
