# M11 I1 — `PhysXBackend`: Isaac Sim이 `PhysicsBackend`로 돈다

스펙: §28.14 규칙 6(계층 `CrossBackend`; 장면은 `scene_to_mjcf`와 Isaac Sim의 MJCF 임포터를 거친다;
임포터의 모든 빈틈은 매핑 보고서의 한 행)과 규칙 1~2, 파동 2, §17.1~17.2, §14.4(severity error로
막힌 매핑은 결코 실행되지 않는다), §3.5, INV-17(`PhysicsBackend`는 일곱 개 중 하나; 새 trait 없음).
API 노트: `docs/api-notes/isaac-sim.md`(W0 + I0의 측정된 충실도 표). I0(venv와 임포터 표)와
X1(`physx`를 선택 가능하게 만드는 디스패치)에 의존. 유형 B/D.

## 질문

`BackendKind::PhysX`는 존재하며 "not implemented"를 출력한다(`crates/es-physics-backend/src/mapping.rs`).
MJWarp와 Newton은 이미 `proc.rs`의 JSON-lines 프로토콜 뒤에서 Python 서브프로세스로 돈다
(`mjwarp.rs`, `newton.rs`, `python/*_ref.py`). **같은 방식으로 만든 `PhysXBackend` — 방출된 MJCF를
Isaac Sim이 headless로 임포트하고, 위치 타깃은 조인트 드라이브를 거치며, 상태를 읽어들이는 — 가
임포터가 정확히 무엇을 유지했는지 말해 주는 능력과 quirk를 갖춘 trait 전체를 구현해, `es backend
compare`와 X1의 `es eval run --backend physx`가 그 위에서 도는가?**

## spec

* `crates/es-physics-backend/src/physx.rs` + `python/physx_ref.py`: `proc.rs` 프로토콜(`Load`,
  `Reset`, `SetCtrl`, `Step`, `State`, `SetState`, 그리고 I0가 필드가 env마다 쓸 수 있음을 보이면
  `SetParams` — 아니면 `ModelParams` 능력 없음). `ES_ISAAC_PYTHON`이 인터프리터를 이름 짓는다
  (Isaac venv는 `ES_PYTHON`의 것이 아니다); 없으면 = 문서화된 SKIPPED exit 3.
* `load`: `scene_to_mjcf` → 임시 파일 → MJCF 임포터 → env마다 하나의 articulation(복제됨); 물리
  dt = 장면의 timestep, substep과 솔버(TGS/PGS, 반복 횟수)는 매핑되는 범위에서 장면의 옵션에서
  가져오고, 아니면 quirk 행; CPU 대 GPU 파이프라인은 `LoadConfig`가 고르고 기록된다.
* 액추에이터: MuJoCo `position` 액추에이터 → 액추에이터의 `kp` / `kv`를 stiffness / damping으로
  쓰는 조인트 드라이브(단위 변환 명시), `motor` → effort; 그 외는 모두 이름으로 거부. I0의 표가
  drop되거나 바뀐 것으로 표시하는 다른 모든 MJCF 기능은 매핑 보고서의 한 행이다.
* 상태 순서: 관절은 이름으로 우리 `qpos` / `qvel` 레이아웃에 매핑된다(free joint: 위치 + 쿼터니언
  순서 변환; 문서화할 것).
* 능력: 계층 `CrossBackend`, `f32`, 측정된 대로 `gpu_resident`; `engine_version` = Isaac Sim +
  PhysX 버전(X1의 해시 규칙).

## context

```
crates/es-physics-backend/src/physx.rs
crates/es-physics-backend/src/lib.rs
crates/es-physics-backend/src/mapping.rs
crates/es-physics-backend/src/proc.rs
crates/es-physics-backend/python/physx_ref.py
crates/es-physics-backend/tests/**
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/check_deps.rs
crates/es-tools/src/backend.rs
crates/es/tests/cli.rs
docs/api-notes/isaac-sim.md
docs/api-notes/isaac-sim.ko.md
docs/packets/M11/I1-physx-backend.md
docs/packets/M11/I1-physx-backend.ko.md
```

## 오라클

1. `cargo test -p es-physics-backend physx_capabilities_are_declared_honestly`와
   `physx_mapping_report_names_every_dropped_feature`(Isaac 불필요).
2. 서버, `ES_ISAAC_PYTHON=~/venvs/es-isaac/bin/python`, GPU 큐 락: `es backend compare --scene
   tests/fixtures/mjcf/so101_pick_place.xml --backends mujoco-cpu,physx --ctrl-random --ticks 500`와
   `mesh_box.xml`에서 같은 것 — 수치(max |Δqpos|, 발산 틱, 에너지 프록시) 기록; 두 번 실행(런 간
   재현성, CPU와 GPU 파이프라인).
3. 서버: `evaluation-reach.toml`(X1의 디스패치)에서 reach A0의 `es eval run --backend physx` —
   완료; `success_rate`가 mujoco-cpu와 mjwarp 옆에.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens; Isaac 없는 로컬 빌드는 깔끔하게 SKIP.

## 수용

오라클 1~4; api-note의 백엔드 절(+ko)에 수치와 모든 quirk.

## 금지

새 trait; 어떤 Rust 크레이트로도 Isaac Sim을 임포트(서브프로세스만); 측정이 아니라 주장된
허용오차; 임포터의 drop 숨기기; `es-safety`.
