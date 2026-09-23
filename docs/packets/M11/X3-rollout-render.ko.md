# M11 X3 — `Rollout`이 렌더링한다: RL이 이미지를 관측하며, env마다 틱별 시드로

스펙: §28.14 규칙 1, 4, 5와 파동 2, §4.3(Python은 학습 경로의 일급 의존성; 렌더러는 Rust), §13.4
(`es-env`를 거친 rollout, plane 켜짐), §15, §3.4. 설계 노트: `renderer.md` 12.3 / 12.8(틱 시드와 그 세
호출자), `rl-continuation.md` 2절(+ko). X6(`render.svgf`)가 머지된 것과 X1(`Rollout`의 백엔드 enum)에
의존 — 먼저 `git merge --ff-only main`. 유형 B.

## 질문

`es_native.Rollout::observe`는 es-py가 렌더러를 링크하지 않아(`crates/es-py/src/rollout.rs:235`) 이미지
입력을 거부하고, es-py는 `render` 기능 없이 `es-env`를 빌드한다. PPO는 이미 로워링된 Learning IR을
실행하므로 이미지가 도착하면 `VisionBackbone` 인코더가 동작한다. **es-py `render` 기능이 있으면
`Rollout`이 각 env의 센서를 렌더링하는가 — `Rs`나 `Pt`, `seed = "tick"`이 각 env 자신의 리셋에서 다시
시작하고 `svgf`는 선언대로 — 그래서 `Rollout` 프레임이 같은 `(episode, tick)`에서 수집기의 프레임과
비트 단위로 동일한가?**

## spec

* `es-py`에 `render` 기능(`es-env/render`)이 생긴다; `train_ppo.py`가 쓰는 maturin 빌드가 이를 켠다.
  기능이 없으면 이미지 입력은 여전히 이름으로 거부된다(오늘의 동작).
* `Rollout`은 수집기가 쓰는 것과 같은 `es_env::render::sensor_cfg`로 만든 `EnvRenderer`를 env마다
  하나씩 소유한다; `observe`는 `es_eval::runner`와 같은 `capture` 경로로 이미지 포트를 캡처한다
  (Observation IR의 두 번째 구현 없음); 각 env의 리셋은 그 env의 `EnvRenderer::begin_episode`를
  호출하므로 틱 시드는 env마다 에피소드 상대적이다.
* 프레임당 렌더 비용을 측정해 `Rollout.metrics()`로 반환한다(렌더 ms/프레임, 9지표 집합의 렌더 행;
  단일 `step/s` 수치는 없음).
* `train_ppo.py`는 이미지 포트를 로워링된 모듈이 기대하는 레이아웃의 `[n_envs, C, H, W]` float
  텐서로 쌓는다(계약에서 읽는다, 절대 가정하지 않는다).

## context

```
crates/es-py/Cargo.toml
crates/es-py/src/rollout.rs
crates/es-py/src/lib.rs
crates/es-py/tests/**
crates/es-env/src/render.rs
crates/es-env/tests/render_loop.rs
python/es/train_ppo.py
python/es/pyproject.toml
tests/fixtures/rl/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/X3-rollout-render.md
docs/packets/M11/X3-rollout-render.ko.md
```

## 오라클

1. `cargo test -p es-py --features render rollout_frame_equals_collector_frame`(GPU) — 카메라를 가진
   reach 과제(`tests/fixtures/rl/task-reach-vision.toml`, 다른 reach 문서처럼 생성됨), `Pt` 16 spp,
   `seed = "tick"`: env 2개 × 에피소드 2개에서 `(episode, tick)`의 `Rollout` 프레임 == 같은
   `(episode, tick)`의 `es loop collect --frames` 비트 단위; 기능이 없으면 이미지 입력은 이름으로 거부.
2. `cargo test -p es-py rollout_state_only_is_unchanged` — state-only reach rollout과
   `train_rl_two_runs_are_bitwise`는 불변.
3. `ES_PYTHON=… python/es/train_ppo.py` ResNet18 `VisionBackbone` 그래프로 vision reach 과제에서
   10회 반복 스모크: 실행됨, 유한한 손실, 렌더 ms/프레임 보고.
4. fmt, clippy `-D warnings`(`render` 있음/없음 둘 다), check-scope, verify-goldens.

## 수용

오라클 1~4가 두 GPU 모두에서; 렌더 비용 표(Rs, Pt 4/16/64 spp, SVGF on/off)가
`rl-continuation.md`(+ko)에.

## 금지

es-py 안의 두 번째 관측 구현; temporal accumulation; `EnvRenderer`의 프레임 바이트 변경; env를
가로지르는 배칭(X3b의 몫).
