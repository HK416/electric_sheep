# M8 S2c — 소스 정책: 우리 SO-101 장면 위의 brax PPO, 가져올 수 있도록 고정됨

스펙: §28.11 파동 1과 그 "오너 결정 2"(brax PPO, MuJoCo Playground 0.2.0 고정, Isaac은 두
번째), §14.4, §2.5(외부 도구는 버전과 다이제스트로 고정된다), §1.7(인식되지 않은 업스트림
사실은 경고이고, 결코 추측이 아니다). 선례: `docs/api-notes/mujoco-playground-quadruped.md`
(brax 파라미터 형식, 2절 "Export to numpy" — 먼저 읽을 것). 작업은
`docs/design/rl-continuation.md` 5절에서 한 번 정의되고, 이 패킷은 그것의 **소스 쪽**을
구현한다. 새 api-note: `docs/api-notes/brax-ppo-so101.md`(+ `.ko.md`). 유형 D: 수치는
관측이다.

## 질문

업스트림에 SO-101 PPO 정책이 사전학습되어 존재하지 않는다(Playground는 Go1, Panda, Aloha,
Leap …를 낸다). **우리 자신의 `so101_pick_place.xml` 위에서 업스트림 스택(MJX + brax PPO)으로,
4090에서, 두 번째 사람이 다시 돌릴 수 있는 레시피로부터 reach 정책을 학습할 수 있는가 — 그리고
Rust에서 orbax나 pickle을 결코 열지 않고 S2b가 가져올 수 있는 형태로 내보낼 수 있는가?**

## 사양

* **환경** `python/es/rl_source/so101_reach_env.py`: 커밋된 장면 위의 `mjx_env.MjxEnv`, 50 Hz
  제어(`ctrl_dt = 0.02`, `sim_dt = 0.005`, `n_substeps = 4`), `rl-continuation.md` 5절과
  정확히 같은 관측/행동/보상/성공/타임아웃(그리퍼 점 = 사이트 `gripperframe`; 큐브 포즈는
  커밋된 Task IR의 `Randomization` 노드가 하는 방식으로 에피소드마다 무작위화 —
  `tests/fixtures/visible-learning/task.toml`의 헤더를 읽고 같은 분포를 재현하라; 동일할 수
  없다면 api-note에 그렇게 적어라). 행동은 `[-1, 1]` → `ctrlrange`로 정규화된다(centre +
  half-range × a).
* **MJX 호환성, 가정이 아니라 측정.** 장면은 `integrator="implicitfast"`, `cone="elliptic"`,
  `condim="3"`/`"6"`, `frictionloss`, `armature`, `<position>` 액추에이터를 선언한다. 먼저
  장면을 그대로 시도하라; MJX가 거부하거나 발산하면, **최소한의** 편집으로 파생된
  `python/es/rl_source/so101_reach_mjx.xml`을 쓰고 모든 편집을 api-note에 표로 남겨라 — 각
  편집은 S4c가 측정할 sim-to-sim 간극이다.
* **학습** `python/es/rl_source/train_brax_so101.py --seed S --timesteps N --out <dir>`:
  Playground의 매니퓰레이션 파라미터 스타일의 설정(네트워크 `(256, 256)` 또는 수렴하는
  무엇이든 — 기록하라)으로 `brax.training.agents.ppo.train`, `normalize_observations = True`.
  쓰는 것: orbax 체크포인트, `source.npz`(프레임워크 중립: `obs_mean`, `obs_std`, 저장된
  그대로의 히든 `kernel_i`/`bias_i`, 출력 Dense를 나눈 `mean_kernel`/`mean_bias`와
  `logstd_kernel`/`logstd_bias`), `meta.json`(`framework`, 버전들, `obs_dim`, `action_dim`,
  `hidden`, `activation = "swish"`, `activate_output = true`, `squash = "tanh"`, 관측 배치,
  행동 `scale`/`offset`, MJCF에서 온 관절 순서, 사용된 장면의 blake3, 설정), 보상 곡선 JSON,
  그리고 **`oracle-1000.npz`**: 채널마다 시드 0으로 `U(−1, 1)`에서 뽑은 관측 벡터 1,000개와
  JAX가 그것들에 대해 계산한 결정론적 행동 `tanh(loc)`(f32). 그 파일이 S2b의 오라클 입력이다.
* **소스 프레임워크에서의 평가** `eval_brax_so101.py`: 결정론적 정책으로 64 에피소드에 걸친
  성공률 — S4c가 "소스 프레임워크" 열에 넣을 그 수치.
* **Venv** 서버의 `~/venvs/es-rl`(uv, Python 3.12): `jax[cuda12]`, `mujoco`, `mujoco-mjx`,
  `brax==0.14.2`, `playground==0.2.0`, `orbax-checkpoint`, `numpy`. 모든 버전을 api-note에.
  GPU를 예의 있게 공유하라: `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_MEM_FRACTION=.5`; 다른 작업(U4 평가, `~/artifacts/plan-v/m7-r5/`)이 돌고
  있을 수 있다 — 무엇도 죽이지 마라.
* **결정성**은 관측이다: `XLA_FLAGS=--xla_gpu_deterministic_ops=true`로 시드 0을 두 번 —
  `source.npz`가 비트 단위인지 기록하라(둘의 blake3). 아니라면 그렇게 적고 둘 다의 해시를
  남겨라.
* **아티팩트**는 `~/artifacts/plan-s/s2c/` 아래(체크포인트, `source.npz`, `meta.json`,
  `oracle-1000.npz`, 로그, 곡선); api-note가 경로들과 `source.npz`의 blake3를 이름 짓는다.

## context

```
python/es/rl_source/**
docs/api-notes/brax-ppo-so101.md
docs/api-notes/brax-ppo-so101.ko.md
docs/packets/M8/S2c-source-policy.md
docs/packets/M8/S2c-source-policy.ko.md
```

학습 경로 Python뿐. Rust 파일 변경 없음. `python/es/` 안에서 `rl_source/` 밖은 아무것도 없음.

## 오라클

1. 서버: `train_brax_so101.py --seed 0 --timesteps <N>`이 끝난다; `eval_brax_so101.py`가 64
   에피소드에 걸쳐 성공률 ≥ 0.8을 보고한다(목표; 도달하지 못하면 행이 도달한 값을 적고 실행은
   여전히 실린다 — S4c는 *어떤* 정책이든 필요하고, 약한 것이 없는 것보다 이어하기에는 더 정보를
   준다).
2. `source.npz` / `meta.json` / `oracle-1000.npz`가 쓰인다; 작은 리더
   `python/es/rl_source/check_export.py`가 `source.npz`로부터 numpy에서 1,000개의 행동을
   다시 계산해서(swish, tanh) JAX의 것과 ≤ 1e-5로 일치한다(numpy 대 XLA; 최대 절대오차를
   기록하라).
3. 시드 0 실행 두 번: `source.npz`의 blake3가 기록됨, 비트 단위이든 아니든.
4. api-note가 고정값, 환경 정의, MJX 발견 표, 레시피, 수치(각각
   `measured (server, date, path)`), S2b가 읽을 내보내기 형식과 함께 존재한다.

## 수용 기준

오라클 1–4; `cargo xtask ci`는 영향받지 않음(Rust 없음), 이 패킷 위에서
`cargo xtask check-scope`.

## 금지

어떤 Rust 변경도; 이 Python 말고 어디서든 체크포인트를 여는 것; 커밋된 장면
(`tests/fixtures/mjcf/**`)을 바꾸는 것 — 파생 XML은 `rl_source/` 아래에 산다;
`rl-continuation.md` 5절과 다른 작업 정의를 api-note에 어떻게 왜 다른지 적은 행 없이 쓰는 것.
