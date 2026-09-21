# M8 S4a — `es_native.Rollout`: Python 트레이너가 우리 `Env`와 Safety Plane을 밟는다

스펙: §13.4("롤아웃은 `es-env`의 시뮬레이션 배치 도메인이다; Safety Plane은 켜져 있다"), §2.1
(Python은 학습 경로에서만 1급이다; 바인딩은 `es-py`, 레이어 11이고, `es-env`(9),
`es-safety`(8), `es-compile`(7), `es-physics-backend`(4)에 의존할 수 있다), §4.2(같은 레이어
의존 없음), §28.11 파동 1, INV-11/12/13, INV-17(pyclass는 확장 지점이 아니다). 설계 노트:
`docs/design/python-builder.md`(+ `.ko.md`)가 한 절을 얻는다; 결정들은
`docs/design/rl-continuation.md` 2–3절에 있다. 선례: `crates/es-py/src/pybind.rs`(빌더
pyclass 네 개, `#[pyclass(unsendable)]`, 피처 `python`, `cdylib` + `rlib`).

## 질문

`train_ppo.py`(S4b)는 env N개를 리셋하고 스텝하고, **Observation IR의** 출력 포트를(원시
상태가 아니라) 읽고, 샘플링된 모든 행동을 `SafetyPlane::validate`를 거쳐 밀어넣고, 보상, done,
플레인의 이벤트 비트를 읽어야 한다 — `es eval run`이 쓰는 것과 같은 코드를 통해, Python에서.
**그것을 주는 가장 작은 바인딩은 무엇이고, 그것이 Rust 쪽이 만드는 것과 같은 롤아웃임을 어떻게
보이는가?**

## 사양

* `es_native.Rollout(task_toml: str, observation_toml: str, deployment_toml: str, scene_xml: str,
  seed: int, n_envs: int)` — Rust 구조체 `Rollout`(평범한 구조체, `rlib`에서 보임, Python
  없이 테스트됨) 위의 `#[pyclass(unsendable)]`로, `n_envs` 크기의 `BatchDomains`로 지어진
  `Env<MuJoCoCpuBackend>`를 갖는다(`es_eval::runner`와 `DomainRunner`가 쓰는 것과 같은
  `TickRate`/`rate.control` 유도), Deployment IR로부터 온 env당 `SafetyPlane` 하나,
  Observation IR로부터 온 env당 `CpuPlan` 하나(`PlanMode::Release`), 그리고 문서들에 대해
  한 번 풀린 plan 입력 소스(`es_env`의 plan/observation 헬퍼가 public이면 재사용하라; 유일한
  구현이 `es-eval` 안에 private이면, 그 헬퍼를 `es-eval`에서 `pub`으로 만들고 그것에
  의존하라 — 두 번째 관측-캡처 구현을 쓰지는 **말아라**).
* 메서드: `reset(envs: list[int] | None) -> None`; `observe() -> dict[str, list[float]]`
  (포트 이름 → 모든 env에 대한 plan의 출력, 행 우선 `[n_envs, dim]`, Python float로서의 f32);
  `act(actions: list[float]) -> (executed: list[float], events: list[int], rewards: list[float],
  dones: list[bool])` — `actions`는 액추에이터 단위의 `[n_envs × nu]`이다; env마다:
  `es_eval::runner::run_episode`가 하는 대로 `observe_state`, `heartbeat`, `validate`(한 행의
  chunk, age 0, now = step)를 거친 뒤, 플레인의 출력으로 `Env::step`; `events`는
  `EventSet::bits()`; `dones`는 `StepOutcome`에서 온다; 종료에서의 `Env::step` 자신의 리셋은
  유지되고 그 env의 플레인에 `begin_episode`가 불린다(P-M7-R1 의미론); `model() -> dict`
  (`nq`, `nv`, `nu`, 액추에이터와 관절 이름, `ctrlrange`); `tick() -> int`;
  `metrics() -> dict`(§12.4의 아홉 필드, 측정되지 않은 곳은 `None`).
* 청크 버퍼도 지연시간도 없음: horizon 1, 동기(`rl-continuation.md` 3절).
* 리스트이지 numpy가 아니다: 새 Rust 의존성 없음. `train_ppo.py`가 자기 쪽에서 변환한다.
* 빌드: `crates/es-py`에서 `maturin develop --release --features python`(서버 venv는
  `python -m pip install maturin`으로 `maturin`을 얻는다); `python/es/README.md`가 그것을
  문서화한다.

## context

```
crates/es-py/**
crates/es-eval/src/runner.rs
crates/es-eval/src/lib.rs
python/es/selfcheck.py
python/es/README.md
python/es/README.ko.md
tests/golden/rollout/**
docs/design/python-builder.md
docs/design/python-builder.ko.md
docs/packets/M8/S4a-rollout-binding.md
docs/packets/M8/S4a-rollout-binding.ko.md
```

`es-py`(`es-env`, `es-safety`, `es-eval`, `es-physics-backend`에 대한 Cargo 의존성; 구조체를
위한 `rollout.rs`, 클래스를 위한 `pybind.rs`), `es-eval`은 **오직** 기존의 private
관측-캡처 헬퍼를 `pub`으로 만들기 위해서만(로직 변경 없음), `selfcheck.py --env`, 골든
(추가분), 노트, 이 패킷.

## 오라클

1. `cargo test -p es-py rollout_matches_es_eval_loop -- --ignored`(`ES_PYTHON`, MuJoCo):
   커밋된 문서들 위에서 고정된 스크립트화된 ctrl로 100스텝 돌린 Rust `Rollout`이 스텝마다
   `qpos`를 `run_episode`가 하는 방식으로 테스트 안에 손으로 짠 `Env` + `SafetyPlane` +
   `CpuPlan` 루프와 비트 단위로 같게 내고, 이벤트 비트도 동일하게 낸다; `ES_GENERATE_GOLDENS=1`
   일 때 `tests/golden/rollout/so101_100steps.json`을 쓴다(그렇지 않으면 그것에 대해
   단언한다).
2. `PYTHONPATH=python <venv>/python -m es.selfcheck --env`(`maturin develop` 이후): pyclass를
   거친 같은 100스텝이 골든과 비트 단위로 같다; `RAN` / `SKIP <reason>`을 찍는다.
3. `python` 피처 없이도 `cargo test -p es-py`는 여전히 빌드되고 통과한다(`rlib` 경로).
4. `cargo xtask layering`(es-py의 새 의존성은 모두 더 아래 레이어), `cargo xtask ci`,
   `cargo xtask check-scope docs/packets/M8/S4a-rollout-binding.md`.

## 수용 기준

오라클 1–4(1과 2는 서버에서: 트리 `~/Projects/es-s4a`, venv `~/venvs/es-lerobot-cuda` +
maturin; 끝나면 트리를 지워라). `python-builder.md`가 메서드 표와 no-numpy 결정과 함께
"롤아웃 바인딩"을 얻는다; 한국어 자매 문서.

## 금지

두 번째 관측-캡처 구현; 건너뛰는 플레인 상태(INV-12)나 `validate` 시그니처 변경(INV-13);
`es-safety` 변경; 청크 버퍼나 지연시간 모델(S4b/S4c가 그 질문을 소유한다); Rust 크레이트 안의
numpy; trait(INV-17); `docs/ARCHITECTURE*.md`.
