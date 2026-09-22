# M10 R1 — `ES_PYTHON` 아래의 기존 실패 둘: 전체 배치 리셋 뒤 낡은 `Env::tick()`, 그리고 CRLF를 쓰는 골든 생성기

스펙: §10.5(`events.json`과 텔레메트리 스트림은 에피소드 상대 틱을 싣는다 — 패킷 M7/R1), §23.1(텔레메트리는
실행의 클라이언트이지 두 번째 시계가 아니다), §1.4(골든 파일은 CI 읽기 전용; 한 플랫폼에서 재현 못 하는
생성기는 생성기의 버그다), §28.13 규칙 1. 플랜 W 파동 0의 CI에서 발견(둘 다 플랜 W 이전인 `c7e736e`에서도
실패; 둘 다 `ES_PYTHON`이 필요해 이전 CI에서는 돌지 않았다). 진단(읽기 전용, 2026-09-22): "질문" 참조. 유형 B, 작음.

## 질문

**(A)** `crates/es-compile/tests/gen_goldens.rs`의 `the_goldens_are_what_the_torch_oracle_produces`가 Windows에서
실패한다: `crates/es-compile/python/gen_observation_goldens.py:76`이 `.json` 사이드카를 `newline=` 없는
`Path.write_text(...)`로 써서 Python 텍스트 모드가 Windows에서 `\r\n`을 내는 반면 커밋된 골든은 LF다
(`.gitattributes` `eol=lf`); 모든 `.bin`은 바이트 동일, 모든 사이드카는 줄마다 정확히 1바이트씩 다르다
(`crop_8x6_at_2_1_4x4.json`의 439 대 454). torch 2.14.0 / torchvision 0.29.0은 api-note 핀과 일치한다.
**`newline="\n"`이면 생성기가 Windows에서도 커밋된 골든을 바이트 단위로 재현하는가?**

**(B)** `crates/es/tests/cli.rs`의 `collect_telemetry_publishes_every_episode`(`--features render`):
`es loop collect --telemetry` 자식이 `crates/es/src/cmd/telemetry.rs:326`
(`record.tick.0 -= *self.episode_start.get_or_insert(tick.0)`)에서 `attempt to subtract with overflow`로
패닉한다. 원인: `Env::tick()`(`crates/es-env/src/env.rs:187-189`)은 `Env::step()`(`:287`)에서만 갱신되는
필드를 돌려주고, `Env::reset()`(`:204-261`)은 갱신하지 않는데, MuJoCo 백엔드의 전체 배치 `reset(None)`
(`crates/es-physics-backend/src/mujoco.rs:316-318`)은 자기 시계를 0으로 만든다 — 에피소드가 종료가 아니라
`--max-steps`로 끝날 때 `crates/es-data/src/collect.rs:674`가 타는 분기. 그래서 다음 에피소드의 첫 `Tick`은 이전
에피소드의 낡은 누적 틱을 보고하고, 퍼블리셔가 그것을 `episode_start`로 잡으며, 두 번째 `Tick`(이제 작은 값)에서
언더플로한다. 릴리스 빌드에서는 뺄셈이 감겨 ~1.8e19의 틱이 스트림 2로 나간다 — 플레이크가 아니라 틀린 값이다.
**`Env::reset`이 백엔드에서 캐시된 틱을 재동기화하면 `Env::tick()`이 모든 호출자에게 옳고, 텔레메트리 테스트가
통과하며, 커밋된 바이트는 하나도 움직이지 않는가?**

## 명세

* (A) `gen_observation_goldens.py`: `write_text(..., encoding="utf-8", newline="\n")`(그 생성기의 다른 텍스트 쓰기도).
  골든 변경 없음.
* (B) `crates/es-env/src/env.rs` `Env::reset`: `self.backend.reset(...)` 뒤에 `self.tick = self.backend.state().tick`
  (또는 백엔드가 노출하는 동등한 접근자) — 캐시가 절대 낡지 않도록; 수정은 텔레메트리 호출 지점이 아니라 공유
  함수에 둔다. `telemetry.rs:326`은 그대로(입력이 옳으면 뺄셈은 옳다). `es-env`의 단위 테스트: 스텝 뒤 `reset(None)`
  다음 `env.tick()`이 백엔드 시계와 같다(픽스처 백엔드에서; 픽스처 백엔드가 MuJoCo 백엔드의 전체 배치 리셋
  의미론을 비추지 않아 `reset(None)`에서 시계를 0으로 만들지 않으면 그렇게 말하고, 픽스처 자체의 의미론과 CLI
  테스트로 검증한다).
* 문서: `docs/design/telemetry-protocol.md`(+ `.ko.md`)의 에피소드 상대 리베이스가 설명된 곳에 낡은 캐시 원인과 이
  패킷을 이름하는 한 문장; `docs/design/observation-lowering.md`(+ `.ko.md`) 또는 골든 생성기가 설명된 곳에
  `newline` 규칙 한 문장.

## context

```
crates/es-compile/python/gen_observation_goldens.py
crates/es-compile/tests/gen_goldens.rs
crates/es-env/src/env.rs
crates/es-env/tests/**
crates/es/tests/cli.rs
docs/design/telemetry-protocol.md
docs/design/telemetry-protocol.ko.md
docs/design/observation-lowering.md
docs/design/observation-lowering.ko.md
docs/packets/M10/P-M10-R1-stale-tick-and-crlf.md
docs/packets/M10/P-M10-R1-stale-tick-and-crlf.ko.md
```

## 오라클

1. `ES_PYTHON=… cargo test -p es-compile the_goldens_are_what_the_torch_oracle_produces`가 Windows에서 녹색;
   `cargo xtask verify-goldens` — 움직인 골든 없음.
2. `cargo test -p es-env <새 틱 테스트>`; `ES_PYTHON=… cargo test -p es --features render --test cli
   collect_telemetry_publishes_every_episode` 녹색(단독으로 돌린다; 다른 빌드가 돌면 부하에 민감하다).
3. `cargo test --workspace`(`ES_PYTHON`; 알려진 `quadruped_eval_run_names_the_observation_gap` 실패 제외),
   `cargo xtask verify-goldens`, `cargo xtask ci`, `cargo xtask check-scope docs/packets/M10/P-M10-R1-stale-tick-and-crlf.md`.
4. 데모 궤적과 롤아웃 골든 불변(`tests/golden/rollout/so101_100steps.json`, `crates/es-eval/tests`의 `.estraj` 핀):
   리셋 시점의 재동기화는 스텝된 값을 바꿀 수 없고, 테스트 스위트가 그것을 말한다.

## 수용

오라클 1~4; 두 문서 문장과 한국어 짝.

## 금지

골든 편집; `telemetry.rs`의 뺄셈 건드리기; `Env::step`·백엔드·`collect.rs`·데이터셋 스키마 변경;
`docs/ARCHITECTURE*.md`.
