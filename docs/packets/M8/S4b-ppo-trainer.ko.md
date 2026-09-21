# M8 S4b — PPO 트레이너: `[rl]`을 가진 `es train --recipe`, `es_native.Rollout`을 거치는 롤아웃

스펙: §13.4(전부), §19.3, §12.4, §3.5 계층 1(CPU 백엔드 비트 단위), §28.10 규칙 2, §28.11
파동 3, INV-12(플레인이 켜져 있다), INV-16, INV-17. **S4a**(바인딩), **S1**(`[init]`),
**S2a**(스쿼시 로워링)에 의존. 설계 노트: `docs/design/rl-continuation.md` 2, 3, 5, 6절이
그 결정들이다; 이 패킷이 7절을 채운다. `training-recipe.md`가 "RL 경로"를 얻는다.

## 질문

**하나의 레시피가 PPO를 돌릴 수 있는가 — Safety Plane이 켜진 우리 `Env`에서의 롤아웃, GAE,
클립된 목적함수, 엔트로피 — 처음부터 또는 `[init]`으로부터, §19.3의 `training/`을 실제 값으로
채우고, CPU 백엔드에서 비트 단위로 재현 가능하도록?**

## 사양

* **레시피.** `[rl] algo = "ppo"`, `envs`, `horizon`(반복마다 env당 스텝), `epochs`,
  `minibatches`, `gamma`, `lam`, `clip`, `entropy`, `value_coef`, `init_log_std`(부재 →
  `[init]`이 하나를 나른다면 임포터의 `log_std`, 아니면 `−0.5`). `[rl]`이 있으면: `[run]
  steps` = PPO 반복 수; `[run] batch`는 이름으로 거절됨; `[dataset]`은 선택적이고
  `dataset.lock`은 RL 실행에서 `unset`을 읽는다; `[policy] bundle`이 그래프를 이름 짓는다
  (문서들의 새 `es policy pack` 또는 가져온 번들); `task` / `observation` / `deployment`는
  번들에서 온다; `[run] seed`, `device`, `lr`, `grad_clip`, `checkpoint_at`, `schedule`은
  자기 의미를 유지한다.
* **경로** `rl`: `es policy lower` → `python/es/train_ppo.py --module … --rollout-docs …
  --scene … [--init-weights …] --envs … --out …` → 체크포인트마다 `es policy pack`(IR 경로가
  하는 대로). 트레이너는 로워링된 모듈(액추에이터 단위의 결정론적 행동)로부터 액터를 짓고,
  상태 독립적인 `log_std`를, Observation IR 포트들의 결합 위의 가치 MLP를 짓는다;
  `a = mu + exp(log_std)·eps`를 샘플링한다; `Rollout.act`를 스텝한다; `obs`, 샘플링된 `a`,
  실행된 행동, 이벤트 비트, 보상, done, 가치, 로그 확률을 저장한다; GAE; 클립된 목적함수 +
  가치 손실 + 엔트로피; `[run] lr`, `grad_clip`을 가진 Adam; `device = cpu`에서
  `torch.use_deterministic_algorithms(True)`와 시드된 RNG.
* **기록되는 것.** `metrics/loss-curve.json`, 반복마다: `loss`, `policy_loss`, `value_loss`,
  `entropy`, `return`, `episode_len`, `envelope_violation_rate`(이벤트 비트에서),
  `executed_ne_sampled_rate`, `samples_per_sec`; 스트림 5 `[step, loss, lr, samples/s]`,
  그래서 에디터의 Live 탭이 그것을 그린다(E7); `training/config.json`이 `[rl]`을 나른다;
  `training/value.safetensors`는 재개용, 결코 패킹되지 않음.
* CPU 백엔드에서의 **결정성**: 같은 레시피의 두 번 실행 → 모든 체크포인트가 비트 단위이고
  같은 `training_hash`. GPU 백엔드는 나중의 처리량 패킷이다(§28.11 "사다리에 없음").
* **오라클 작업** `tests/fixtures/rl/task-reach.toml` + `observation-reach.toml` +
  `deployment-reach.toml` + `evaluation-reach.toml`(`rl-continuation.md` 5절; 무작위
  정책이 클램프되고 래치되지 않을 만큼 넓은 배포 envelope — INV-12: 넓히기만, 결코
  비활성화하지 않기)와 `training-reach.toml`(`[rl]`, 작게: `envs = 8`, `horizon = 64`).

## context

```
crates/es-data/src/training.rs
crates/es-data/tests/**
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
python/es/train_ppo.py
python/es/README.md
python/es/README.ko.md
tests/fixtures/rl/**
tests/golden/train/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M8/S4b-ppo-trainer.md
docs/packets/M8/S4b-ppo-trainer.ko.md
```

## 오라클

1. `cargo test -p es --test cli train_rl_dry_run_plan` — `training-reach.toml`을 위한 plan
   골든(추가분); `[rl]`이 있는 `[run] batch`는 이름으로 거절됨; `[dataset]` 부재는
   받아들여짐.
2. `cargo test -p es --test cli train_rl_two_runs_are_bitwise`(`ES_PYTHON`; 이유와 함께
   SKIP): `envs = 4`, `horizon = 16`, 반복 3회, 시드 0, 두 번 → 체크포인트 비트 단위,
   `training_hash` 같음, `dataset.lock`은 `unset`을 읽음, `loss-curve.json`이 나열된
   필드를 가짐.
3. `cargo test -p es --test cli train_rl_init_from_import`(`ES_PYTHON`): `[init] policy =`
   S2b의 합성 playground 픽스처 번들 → 첫 업데이트 전 반복 0의 액터가 가져온 것과 비트
   단위로 같음(`0.esb`).
4. 서버: 실제 예산(`envs = 16`, `horizon = 64`, N 반복)의 `training-reach.toml` →
   `es eval run --config evaluation-reach.toml`의 성공률 ≥ 0.8; N, 벽시계, 7절의 아홉
   지표(실행이 측정하지 않는 것은 `Target / Status: unverified`).
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M8/S4b-ppo-trainer.md`.

## 수용 기준

오라클 1–5. 날짜와 경로가 있는 7절의 행들; `training-recipe.md`의 "RL 경로".

## 금지

`es-env` 말고 다른 시뮬레이터(규칙 2); 건너뛰는 플레인 상태(INV-12); 문서나 번들 안의 가치
헤드나 `log_std`(규칙 1); RL을 위한 Task IR 변경(규칙 6); pickle(INV-16); 새 trait(INV-17);
`docs/ARCHITECTURE*.md`; `tests/golden/**` 수정.
