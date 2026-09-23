# M11 R1 — 백엔드 후속: 식별자에 어댑터 스크립트, `Rollout`의 PhysX, `es train`을 통한 vision RL

스펙: §28.14 규칙 2(백엔드는 실행의 조건이다)와 규칙 1, §5.3, §13.4. X1 후속·I1·X3 보고(2026-09-23)에서
나온 것: (1) `mjwarp_ref.py` 수정 두 건과 `physx_ref.py` 수정 한 건이 평가 결과를 바꿨는데 어떤
`execution_hash`도 움직이지 않았다 — `backend_identity`가 엔진 버전만 해시하고 어댑터 스크립트는 해시하지
않기 때문이다; (2) `es_native.Rollout`은 여전히 `backend = "physx"`를 거부한다; (3) `es train`의 `[rl]`
경로가 Observation IR에 이미지 입력이 있는 번들을 거부한다(`crates/es/src/cmd/train.rs`). 그래서 vision RL은
`train_ppo.py`를 직접 불러야만 돈다. 유형 B.

## 질문

**기준이 아닌 백엔드의 어댑터 스크립트가 바뀌면 `execution_hash`가 움직이고, `Rollout`이 PhysX에서 돌며,
`es train`이 vision reach 레시피를 끝까지 돌리는가 — 커밋된 `mujoco-cpu` 해시는 모두 그대로인 채로?**

## 명세

* `backend_identity(caps, engine_version, script)`는 기존 필드 뒤에, 백엔드가 내장한 어댑터 스크립트
  (`mjwarp::SCRIPT`, `newton::SCRIPT`, `physx::SCRIPT`)의 blake3를 같은 길이 접두 규칙으로 해시한다.
  `mujoco-cpu`는 모두 0인 슬롯을 유지한다(§28.14 규칙 2: 기준 백엔드는 대신 골든이 고정한다).
  `evaluation.lock`의 `backend` 블록에 `script_blake3`가 추가된다.
* `Rollout(backend = "physx")`: `crates/es-py/src/rollout.rs`의 닫힌 enum에 `Env<PhysXBackend>`를 더한다.
  `[rl] backend = "physx"`를 파싱한다. 롤아웃 루프의 다른 부분은 바꾸지 않는다.
* `es train` `[rl]`은 빌드에 `render` feature가 있으면 이미지 입력을 받는다(maturin 빌드는 X3 이후 항상 이
  feature를 켠다). 롤아웃 문서는 지금처럼 넘기고, 렌더 비용을 `metrics/env-metrics.json`에 기록한다.
  feature가 없으면 여전히 이름으로 거부한다.

## context

```
crates/es-physics-backend/src/lib.rs
crates/es-physics-backend/tests/**
crates/es-eval/src/runner.rs
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
crates/es-data/src/training.rs
crates/es-py/src/rollout.rs
crates/es-py/src/pybind.rs
crates/es-py/tests/**
tests/golden/train/**
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/packets/M11/P-M11-R1-backend-followups.md
docs/packets/M11/P-M11-R1-backend-followups.ko.md
```

## 오라클

1. `cargo test -p es-physics-backend backend_identity` — 스크립트 바이트만 바꿔도 식별자가 바뀐다.
   `mujoco-cpu`의 슬롯은 여전히 모두 0이다. 커밋된 `mujoco-cpu` lock은 모두 그대로다.
2. `cargo test -p es --test cli train_rl_vision_dry_run_plan` — `tests/fixtures/rl/training-reach-vision.toml`의
   plan 골든(추가). `ES_PYTHON`과 GPU가 있으면 3회 반복 `es train`이 끝까지 돌고 체크포인트를 패킹한다.
3. 서버(`ES_ISAAC_PYTHON`, GPU 락): 상태 reach 레시피에 `[rl] backend = "physx"`로 5회 반복 `Rollout`
   PPO 스모크가 끝까지 돈다.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## 수용

오라클 1–4. `evaluation-execution.md` §2.8(+ko)에 스크립트 규칙을 적는다.

## 금지

`mujoco-cpu` 해시를 움직이는 것; 새 trait; X3 프레임 바이트를 바꾸는 것.
