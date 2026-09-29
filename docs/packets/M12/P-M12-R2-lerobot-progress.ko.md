<!-- Korean translation of docs/packets/M12/P-M12-R2-lerobot-progress.md. The English file is the working copy; regenerate this when it changes. -->
# M12 R2 — LeRobot 경로가 학습 진행 상황을 발행한다

2026-09-29 Y-V 항목 1(카메라 전용 큐브 템플릿, `--from train`으로 재개)에서 발견했다. 스펙
§23.3(실행 지켜보기), `docs/design/editor-shell.md` 16절(학습 스트림),
`docs/design/editor-redesign.md` 6.4절(신호등).

## 결함

`es train`은 스트림 5(`[step, loss, lr, samples_per_s]`)를 `{"progress": …}` JSON 줄에서만
발행한다(`crates/es/src/cmd/train.rs`, "`{\"progress\": ...}` line reaches a viewer" 근처의 줄
리더). `train_act.py`와 `train_ppo.py`는 그 줄을 출력하지만 **`lerobot-train`은 출력하지 않는다**.
그래서 LeRobot 경로에서는 학습 단계 내내(몇 분에서 몇 시간) 지켜보는 쪽에 아무것도 들리지 않는다.
곡선도 ETA도 없고, 학습이 멀쩡한데도 에디터의 신호등은 `THRESHOLDS.silence_s`가 지나면 *응답
없음*으로 바뀐다.

이 실행의 로그에서 측정한, `lerobot-train`(0.6.1)이 실제로 출력하는 것:
- stderr의 tqdm 막대, `\r`로 구분됨:
  `Training:  70%|███████   | 3510/5000 [03:51<01:39, 14.94step/s]` — 정확한 step, 전체, 속도.
- `log_freq` step마다(기본 200) 다음을 담은 INFO 줄:
  `step:3K smpl:26K ep:49 epch:0.25 loss:0.131 grdn:… lr:…` — loss(여기의 step은 축약되어
  있으므로 step으로 쓰면 안 된다).

## 스펙

- 외부(LeRobot) 경로에서만, 트레이너의 stdout **과** stderr도 이 두 형식을 찾아 읽는다(`\n`뿐
  아니라 `\r`로도 나눈다). 지표 줄이 새 loss를 줄 때 진행 행 하나를 발행한다: `step` = 가장
  최근의 정확한 tqdm step, `loss` = 그 줄의 loss, `lr` = 그 줄에 `lr`이 있으면 그 값,
  `samples_per_s` = 가장 최근 tqdm 속도 × `[run] batch`.
- 첫 loss를 알기 전에는 아무것도 발행하지 않는다. 모르는 loss 대신 NaN이나 0을 넣는 일은 절대
  없다(신호등은 NaN을 *고장*으로 읽는다).
- IR 경로와 RL 경로는 바뀌지 않는다. `{"progress": …}` 줄의 의미도 그대로다. 전에 쓰지 않던
  것은 디스크에 아무것도 쓰지 않는다. `training.lock`과 모든 해시는 그대로다.
- 파서는 한 줄을 받는 순수 함수이고 자체 단위 테스트를 갖는다. 표본은 지어낸 것이 아니라 실제
  `lerobot-train` 로그(위의 줄들)에서 복사한다.

## 컨텍스트

```
crates/es/src/cmd/train.rs
crates/es/src/cmd/telemetry.rs
crates/es/tests/cli.rs
docs/packets/M12/P-M12-R2-lerobot-progress.md
docs/packets/M12/P-M12-R2-lerobot-progress.ko.md
```

## 오라클

1. 줄 파서의 단위 테스트: tqdm 형식(step, 전체, 속도), 지표 형식(loss, lr), 둘 다 없는 줄,
   `\r`로 이어진 여러 tqdm 갱신(마지막 것이 이긴다), tqdm 줄보다 먼저 온 지표 줄(아직 행 없음).
2. 기존 train·telemetry 테스트가 그대로 통과한다. `cargo test -p es --test cli train`.
3. fmt, clippy `-D warnings`, `check-scope`, `verify-goldens`.

## 금지

`train_act.py` / `train_ppo.py`가 출력하는 것이나 그 줄을 읽는 방식을 바꾸는 것. 지어낸 loss를
발행하는 것. `es-editor`를 바꾸는 것. 원격 서버에 접속하는 것.
