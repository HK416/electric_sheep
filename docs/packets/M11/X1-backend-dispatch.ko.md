# M11 X1 — `--backend`: 정책이 MJWarp에서 폐루프로 돌고, 백엔드가 해시 체인에 들어간다

스펙: §28.14 규칙 2(`mujoco-cpu`가 아닌 백엔드에서의 실행은 `execution_hash`의 `hardware_capability`
슬롯에 `H("es.backend.v1", name, engine version, float, determinism tier)`를 쓴다; `mujoco-cpu`는 오늘의
슬롯을 유지한다)와 규칙 1과 파동 1, §5.3, §17.1–17.2(백엔드는 능력을 선언한다; 백엔드가 매핑 못 하는
장면은 무엇을 띄우기 전에 거부된다), §3.5(계층 2 `CrossBackend`: 결코 비트 단위가 아니고, 차이는 숫자로
보고된다), §14.4, INV-17. 리뷰: `docs/reviews/M4.md` gate 12(MJWarp max |Δqpos| 7.4e-8, Newton 5.9e-4,
개루프). 설계 노트: `evaluation-execution.md`(+ko)에 "backend" 하위 절이 생긴다; `rl-continuation.md`(+ko)
2절에 `[rl] backend` 줄이 생긴다. 유형 B, D 행 하나.

## the question

`Env<B: PhysicsBackend>`는 제네릭이고 `MjWarpBackend` / `NewtonBackend`는 모든 trait 메서드를 구현하지만,
`es eval run`, `es loop collect`, `es_native.Rollout`은 `MuJoCoCpuBackend`를 하드코딩한다
(`crates/es/src/cmd/eval.rs:455,703,776`, `loop.rs:311,330,593,633`,
`crates/es-py/src/rollout.rs:102,173`), 그래서 이 저장소가 할 수 있는 유일한 sim-to-sim은 `es backend
compare`의 개루프 궤적뿐이다. **`BackendKind` 위의 디스패치 하나로, 같은 번들이 MJWarp에서 폐루프로 돌아
— `Rollout`을 통해 평가되고 수집되고 학습되며 — 백엔드가 해시 체인에 기록되고, 커밋된 모든 `mujoco-cpu`
해시는 움직이지 않으며, Newton / PhysX는 이름으로 거부되는가?**

## spec

* 동사마다 함수 하나가 백엔드를 고른다: `--backend mujoco-cpu | mjwarp | newton | physx`(기본
  `mujoco-cpu`)는 `BackendKind`로 파싱되고(`crates/es-physics-backend/src/mapping.rs:26-49`) 고른
  타입으로 단형화된 기존 제네릭 진입점을 호출한다 — `Evaluation::run_shard_with_sink::<B, …>(…, B::new,
  …)`와 `Collector::run_with_sink::<B, …>` — `Env` 안에서 `Box<dyn PhysicsBackend>`는 절대 아니다
  (제네릭은 유지; §3.4 hot path). 가용성(`B::is_available()`)은 문서화된 SKIPPED exit 3을 유지한다.
  `newton`은 `load`에 닿고 거기서 자신의 매핑 보고서로 거부된다(액추에이터와 센서가 선언되지 않았고, 접촉이
  연결되지 않음) — 거부는 행 이름을 댄다; `physx`는 I1이 상륙할 때까지 `not implemented (M11/I1)`을
  찍는다.
* `es_native.Rollout(…, backend="mujoco-cpu")`: `Env<MuJoCoCpuBackend>` / `Env<MjWarpBackend>`의 내부
  enum 뒤에 같은 디스패치(닫힌 enum, trait object가 아니고 새 trait도 아님 — INV-17). 학습 레시피의
  `[rl] backend = "mjwarp"`(부재 = `mujoco-cpu`, 부재처럼 직렬화, M9/R5의 `estimator` 패턴)는
  `train_ppo.py --backend`와 `training_hash`에 닿는다.
* 해시 체인: `es_physics_backend::backend_identity(caps: &Capabilities, engine_version: &str) -> [u8;
  32]` = 규칙 2의 태그된 튜플의 blake3; `RunConfig.hardware`는 `mujoco-cpu`를 뺀 모든 백엔드에서 그
  다이제스트이고, `mujoco-cpu`의 슬롯은 오늘 값을 유지하므로 커밋된 모든 `evaluation.lock`의
  `execution_hash`는 움직이지 않는다. 엔진 버전은 백엔드의 load 응답에서 온다(`proc.rs`의 `LoadReply`에
  `engine_version`을 더하고, 없으면 세 `*_ref.py`에도 더한다; 버전 없음은 빈 문자열이 아니라 에러).
  `evaluation.lock`은 이미 `backend`를 찍는다; 여기에 `engine_version`이 더해진다.
* `es eval run`, `es loop collect`, `train`의 도움말은 네 이름과 오늘 각각이 하는 일을 나열한다.
  `check_deps`는 각 백엔드의 가용성을 찍는다.

## context

```
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/check_deps.rs
crates/es/tests/cli.rs
crates/es-eval/src/runner.rs
crates/es-eval/src/lock.rs
crates/es-eval/tests/**
crates/es-physics-backend/src/lib.rs
crates/es-physics-backend/src/mapping.rs
crates/es-physics-backend/src/proc.rs
crates/es-physics-backend/src/mujoco.rs
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/src/newton.rs
crates/es-physics-backend/python/*_ref.py
crates/es-physics-backend/tests/**
crates/es-py/src/rollout.rs
crates/es-py/src/lib.rs
crates/es-py/tests/**
crates/es-data/src/training.rs
python/es/train_ppo.py
tests/golden/train/**
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/X1-backend-dispatch.md
docs/packets/M11/X1-backend-dispatch.ko.md
```

(나열된 경로가 존재하지 않으면 — 예를 들어 lock 타입이 다른 곳에 있으면 — 그것을 담은 파일을 쓰고 보고서에
이름을 적는다; 나열된 크레이트 밖으로 넓히지 않는다.)

## oracle

1. `cargo test -p es-physics-backend backend_identity` — 다이제스트는 네 이름에 걸쳐, 그리고 두 엔진
   버전에 걸쳐 다르고, 호출마다 안정적이며, `mujoco-cpu`의 슬롯은 오늘 값이다.
2. `cargo test -p es --test cli eval_run_backend_` — 데모 문서에서 `--backend newton`은 어떤 프로세스도
   뜨기 전에 자신의 매핑 행 이름을 대며 거부된다; `--backend physx`는 M11/I1을 이름 댄다; `--backend
   banana`는 사용법 에러(exit 2); `mujoco_warp` 없이 `--backend mjwarp`는 exit 3 SKIPPED. `loop
   collect`도 마찬가지.
3. `cargo test -p es --test cli train_rl_backend_dry_run_plan` — 플랜 골든 추가; 명시한 `backend =
   "mujoco-cpu"`는 부재와 같게 해시; `training-reach.toml`의 `training_hash`는 불변.
4. 기존 테스트가 고정한 커밋된 `evaluation.lock` / 보고서는 모두 불변(알려진 M10 S-1 테스트를 빼고 `es
   --test cli`와 `es-eval` 스위트 전체가 녹색).
5. 서버(`mujoco_warp`가 있는 `ES_PYTHON=~/venvs/es/bin/python`; GPU 큐 lock): `evaluation-reach.toml`의
   reach A0(`~/artifacts/plan-s/s4e/run-4000`)와 `evaluation.toml`의 데모 U3 체크포인트를 각각
   `mujoco-cpu`와 `mjwarp`에서; 쌍마다 `es eval compare`; MJWarp를 두 번 돌림(실행마다 재현되는가?
   추정하지 말고 보고); `mjwarp`에서 20-iteration `Rollout` PPO smoke. 표는 `evaluation-execution.md`의
   새 하위 절에 들어간다.
6. fmt, clippy `-D warnings`, `cargo xtask check-scope docs/packets/M11/X1-backend-dispatch.md`, `cargo
   xtask verify-goldens`.

## acceptance

오라클 1~6; 측정 표에 모든 숫자가 자신의 `execution_hash` 옆에; 설계 노트 하위 절(+ko).

## forbidden

`Env` 안의 `Box<dyn PhysicsBackend>`; 새 trait; `es-safety`에 대한 어떤 변경도; 커밋된 `mujoco-cpu` 해시
옮기기; 매핑 보고서를 억눌러 Newton을 "동작"하게 만들기; MJWarp 결과를 비트 단위라고 보고하기; PhysX
코드(I1).
