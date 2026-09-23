# M11 I0 — 오라클 서버의 Isaac Sim과 Isaac Lab, 헤드리스, 측정됨

스펙: §28.14 규칙 6(Isaac Sim은 소유자가 NVIDIA EULA를 수락한 뒤에만 설치한다; 계층은 `CrossBackend`;
장면은 `scene_to_mjcf`와 그 MJCF 임포터를 거쳐 들어가고, 임포터의 모든 빈틈은 매핑 보고서의 한 행이다)와
파동 1, §2(Python은 학습 경로에서만 일급 의존이다; 코어 런타임은 결코 Isaac을 필요로 하지 않는다),
§17.1. API 다이제스트: `docs/api-notes/isaac-sim.md`, `isaac-lab.md`(M11 W0). 유형 D.

**전제조건:** NVIDIA Omniverse / Isaac Sim EULA에 대한 소유자의 명시적 수락, 이 패킷의 보고서에 날짜와
함께 기록됨. 이것 없이는 패킷이 시작하지 않는다.

## the question

**RTX 4090 서버에서 자신만의 venv에 pip로 설치된 헤드리스 Isaac Sim이 우리 SO-101 장면을
MJCF로(`scene_to_mjcf`가 내는 텍스트) 임포트하고, position target으로 스텝하고, 관절 상태를 보고하는가 —
그리고 그 MJCF 임포터는 무엇을 유지하고, 바꾸고, 버리는가?**

## spec

* api-note가 고정한 파이썬 버전을 쓰는 uv venv `~/venvs/es-isaac`, `isaacsim`(고정된 버전과 extras)과
  Isaac Lab(맞는 릴리스), 그리고 `rsl-rl-lib`. 시스템 패키지 없음; sudo 없음. 설치 로그와 `pip freeze`는
  `~/artifacts/plan-x/i0/` 아래.
* `python/physx_smoke.py`(레포에 있음, 아직 백엔드 아님): `SimulationApp({"headless": True})`를 시작하고,
  MJCF 임포터로 MJCF 파일을 임포트하고, articulation을 만들고, 고정된 position-target 시퀀스로 장면의
  timestep에서 physics 스텝 100번을 밟고, 버전과 관절 궤적을 JSON으로 찍고, 깔끔히 종료한다.
  `tests/fixtures/mjcf/so101_pick_place.xml`(`es`가 낸 MJCF를 거쳐 — 작은 `es backend`나 테스트
  헬퍼로 덤프)과 `mesh_box.xml`에서 실행.
* `docs/api-notes/isaac-sim.md`(+ko)의 임포터 충실도 표, 임포트된 USD stage에서 채움: 관절과 한계,
  액추에이터 → drive type / stiffness / damping, MuJoCo 대비 질량과 관성, 충돌 형상(mesh → convex?),
  friction, contype/conaffinity, 센서, 경고와 함께 버려진 것.
* MuJoCo CPU와의 same-control 비교(100스텝 관절 궤적, max |Δq|) — 숫자만; 허용오차는 주장하지 않음.

## context

```
python/physx_smoke.py
docs/api-notes/isaac-sim.md
docs/api-notes/isaac-sim.ko.md
docs/api-notes/isaac-lab.md
docs/api-notes/isaac-lab.ko.md
docs/packets/M11/I0-isaac-sim-install.md
docs/packets/M11/I0-isaac-sim-install.ko.md
```

## oracle

1. 서버에서: `~/venvs/es-isaac/bin/python python/physx_smoke.py --mjcf <so101.xml>`는 exit 0과 함께
   버전과 100행 궤적을 찍는다; `mesh_box`도 마찬가지.
2. 충실도 표와 Δq 숫자가 날짜·버전과 함께 api-note에 있다.
3. `crates/` 안의 무엇도 바뀌지 않는다; `cargo xtask check-scope docs/packets/M11/I0-isaac-sim-install.md`.

## acceptance

오라클 1~3; 디스크 사용량과 설치 시간 기록; 실행 동안 GPU 큐 lock 유지.

## forbidden

기록된 EULA 수락 없이 설치; sudo; 다른 venv 건드리기; 어떤 Rust 변경도(I1이 백엔드를 소유).
