# M5 V10 — 장면 진단

설계 노트: `docs/design/visible-learning.ko.md` **7.18절**, 그리고 7.15(발견 8, 용의자 셋),
7.16(V8), 7.12–7.13(엔벌로프 의미론, `ChunkBuffer` 경로, 시딩), 7.10(`action`이 무엇인지), 5절과
5.4절(전문가와 그 성공 술어). 스펙: §8.5, §8.6, §9.3, §9.5, §12.1, §13.2. V1·V1c·V6·V6b·V8에
의존하며 그중 무엇도 바꾸지 않는다.

## 이 패킷이 답하려고 존재하는 질문

**SO-101 큐브-담기 데모는 왜 학습되지 않는가?** 하네스는 증명되었다 — 스크립트 전문가가
`es eval run`을 8/8로 통과하고(V6), 두 경로가 한 시드에 대해 같은 장면을 뽑는다(V6b). 모델도
배제되었다 — V8은 LeRobot 자신의 ACT를 임포트해 비트 단위로 검증하고 100,000 스텝 학습해
0/16, 0/16, 1/16을 냈다. 7.15절의 발견 8은 남은 셋을 지목했다.

1. 시연 데이터가 실제로는 풀린 과제를 담고 있지 않다,
2. 전문가가 큐브를 파지하지 않고 밀어낸다,
3. 시간 앙상블이 파지가 액추에이터에 닿기 전에 그것을 무너뜨린다.

V10은 셋 모두를 싼 것부터 측정한다. **학습 없음, 모델 변경 없음, 픽스처 변경 없음, 해시 이동
없음.** 아래 모든 수치는 실행 가능한 명령에서 나온다.

## 범위

* **`context`** — `crates/es/tests/cli.rs`(새 테스트 둘), `python/es/grasp_probe.py`(신규),
  `docs/design/visible-learning*.md`의 7.18절과 미해결 질문 16,
  `docs/packets/M5/V10-scene-diagnosis*.md`.
* **`forbidden`** — `tests/fixtures/` 아래 모든 픽스처, 모든 크레이트의 `src/`,
  `docs/ARCHITECTURE*.md`, 그리고 모든 재학습. V10은 측정하며 고치지 않는다.
* **`INV-17`** — 새 트레이트 없음. 재생은 두 번째 `PolicyRuntime` 구현이 아니라 `es_env::Env`
  위의 평범한 루프이고, 파지 프로브는 런타임 바깥의 파이썬 스크립트다.
* **`INV-12`** — 재생도 모든 액션을 `SafetyPlane::validate`로 통과시킨다. 비활성화되는 것은
  없고 엔벌로프 수치는 움직이지 않는다.

## 세 측정

### 1. 기록된 액션은 성공을 재현하는가?

`recorded_actions_replay_to_the_same_outcome`(`crates/es/tests/cli.rs`). 수집된
`LeRobotDataset`을 열어 각 에피소드의 기록된 액션 행을 같은 리셋에서 개루프로, 같은 `Env`와 같은
`SafetyPlane`을 통해 재생하고, 빈의 3차원 내부 — Task IR의 x 전용 콘 리프가 아니라
`expert_solves_the_pinned_seeds`가 쓰는 정직한 술어(5.4절) — 로 채점한다. 두 액션 열 모두
재생한다: 실행된 `SafeAction`인 `action`과 플레인 이전의 원 명령인 `action_commanded`(7.10절).

명시할 결정 둘. **`PolicyRuntime`이 아니라 평범한 재생 루프** — `INV-17`이 허용하는 확장점은
일곱이고 기록 액션 재생기는 그중 하나가 아니며, `Env::step(ctrl)`이 행을 그대로 받는다.
**청크 `seq`는 에피소드 단위가 아니라 런 단위** — `SafetyPlane::accept`는 마지막으로 본 것보다
큰 `seq`에 대해서만 청크를 받아들이고 `begin_episode`는 의도적으로 그것을 되돌리지 않으므로(§8.6),
에피소드 단위 카운터는 첫 에피소드 이후를 전부 영구 언더런으로 만든다. 측정: 에피소드 단위면
1/50, 런 단위면 50/50.

**오라클.** `ES_V10_DATASET=<root> cargo test --release -p es --test cli
recorded_actions_replay_to_the_same_outcome -- --nocapture`. `action` 재생이 시연 자신이 기록한
것과 정확히 같은 수의 큐브를 빈에 넣는지 단언한다. `mujoco`가 없거나 `ES_V10_DATASET`이 없으면
이유를 출력하고 건너뛴다 — 수집된 데이터셋은 픽스처가 아니라 입력이다.

**결과(V1c `ds-train`, 50 에피소드, 시드 1, 18,263 프레임).** 시연은 50/50으로 큐브를 빈에
기록했다. `action` 재생은 **50/50**을 재현하고(`Success` 49 — 한 에피소드의 큐브는 빈 안에 있는데
x 전용 술어가 아니라고 하며, 이것이 5.4절의 천장이다), 플레인은 10,787틱을 최대 `9.004e-4` rad
고쳤다. `action_commanded` 재생은 **50/50**을 재현하고(`Success` 50), 플레인은 11,589틱을 최대
`3.258e-1` rad 고쳤다. **데이터는 건전하고 용의자 1은 닫혔다.**

### 2. 전문가는 파지하는가, 미는가?

`python/es/grasp_probe.py`를 `~/venvs/es/bin/python`(mujoco 3.13)으로 실행한다. 같은 액션
시퀀스를 픽스처 장면 위에서 순수 `mujoco`로 — 루프 안에 `es` 런타임 없이 — 재생하며, 제어 틱마다
각 조 지옴과 `cube_geom` 사이의 접촉 법선력(`mj_contactForce`), 큐브의 높이, 그리퍼 관절의
명령값 대 측정값을 기록한다.

입력은 재생의 `ES_V10_DUMP` 디렉터리다: `ep-NNN.txt`의 첫 줄이 리셋 `qpos ‖ qvel`이고 그 뒤 모든
줄이 `action[0..6] ‖ cube_xyz[0..3]`이다. Task IR의 무작위화 추첨을 아는 쪽은 `es` 쪽뿐이므로
리셋 상태는 추측이 아니라 전달되어야 하며, 큐브 열이 자체 검증이자 `--substeps`를 설정이 아니라
측정으로 만드는 장치다.

**오라클.** `$ES_PYTHON python/es/grasp_probe.py --dump <dir> --scene
tests/fixtures/mjcf/so101_pick_place.xml`. 표와 판정 줄을 출력하고, `mujoco`가 없으면 이유를
출력하고 0으로 종료한다.

**결과.** **50/50** 시연이 큐브를 테이블에서 완전히 들어 올린다(반높이 20 mm 초과). 들어 올린
높이 중앙값 122.65 mm, 최대 127.67 mm. 양쪽 조가 동시에 접촉한 틱은 중앙값 161.5 — 에피소드의
47.9 % — 이고, 그동안 그리퍼 관절은 0.0950 rad에서 멈춘다. 이는 장면의 25 mm 큐브가 허용하는
기하학적 폐합각이다. 50/50이 큐브를 빈에 넣고 끝난다. **전문가는 파지하며 용의자 2는 닫혔다.**

### 3. 시간 앙상블은 파지 구간을 견디는가?

`the_temporal_ensemble_survives_the_grasp_window`(`crates/es/tests/cli.rs`). 시연 하나를
`es loop collect`와 `es eval run`이 하는 것과 똑같이 `es_env::plane_chunk`로 구동하면서, 두 번째
`ChunkBuffer`에 동일한 청크를 `ChunkBlendPolicy::HardSwitch`로 먹인다. 두 번째 버퍼의 행이 원
명령 — 그 틱에 대한 최신 청크 자신의 행 — 이므로 둘의 차이는 앙상블이고 그 외에는 아무것도
아니다. "원 명령"의 두 번째 정의도, 혼합의 재구현도 없다.

**오라클.** `cargo test --release -p es --test cli
the_temporal_ensemble_survives_the_grasp_window -- --nocapture`. 에피소드가 `Success`로 끝나는
것 **그리고** 모든 관절의 혼합 명령이 최신 청크가 요구한 양 끝을 여전히 포괄하는 것을 단언한다.
허용오차가 아니라 등식인데, 극값이 중첩된 여덟 청크가 모두 동의할 만큼 오래 유지되기 때문이다.
그리퍼를 열어 버리는 `decay`나 `CHUNK_SLOTS`는 이 테스트를 실패시킨다.

**결과(시드 1, `decay = 0.01`, 16행 호라이즌의 중첩 청크 8개).** 에피소드는 351틱이고 `Success`로
끝난다. 118–338틱에서 관절별 최대 편차는 0.35651 rad이고 전부 그리퍼이지만, 모든 관절에서
`blend 최소 == raw 최소 == −0.05000`, `blend 최대 == raw 최대 == 0.90000`이다. 조 관절은 가장
조일 때 0.0678 rad으로, 측정 2가 보고하는 0.0950 rad 폐합각 안쪽이다. **앙상블은 범위의 손실이
아니라 지연이며 용의자 3은 닫혔다.**

## 세 측정이 대신 찾아낸 것

`--substeps 1`은 쉰 에피소드에 걸쳐 프로브의 큐브를 `es` 재생 위에 **0.0002 mm**로 붙여 두고,
`--substeps 4`는 **202.98 mm** 벌어진다. 즉 기록된 액션 한 행은 장면 자신의 `timestep="0.005"`
MuJoCo 스텝 정확히 하나 — **200 Hz** — 인 반면 `deployment.toml`은 `rate.control = 50`을
선언한다. `Env::new`는 `LoadConfig { rate: None }`으로 적재하고 `Env::step`은
`schedule.domains().inference.period`만큼 전진시키는데 `BatchDomains::single_env()`가 그것을 1로
두며, 둘 중 어느 것도 Deployment IR에서 유도되지 않는다.

결과는 모두 산수다: `dt_s`가 Deployment IR에서 오므로 Safety Plane의 모든 동적 상한이 그것이
제한하는 스텝보다 네 배(가속도는 열여섯 배) 헐겁다. 정책이 분해해야 하는 틱당 명령 증분은 관절
범위 ±1.7 rad 위에서 중앙값 0.00322 rad이다. 그리고 16행 청크는 320 ms가 아니라 80 ms를 담는다.

가는 길에 가설 둘이 더 측정되어 닫혔다. 데이터셋은 `observation.state[t]`를 그것을 *만들어 낸*
`action[t]`와 짝지으며 이는 LeRobot의 짝짓기와 반대다 — 그러나 5 ms에서 두 읽기의 차이는
0.7 밀리라디안(0.00251 대 0.00322)으로 잡음이다. 그리고 "이미 보이는 관절을 그대로 반복하라"는
자명한 예측기에 대한 청크의 L1은 궤적 전체에서 0.0982 rad, 파지 구간 안에서 0.0964 rad이므로
파지는 목적함수의 사라질 만한 조각도, 더 어려운 부분도 아니다.

## 수용 기준

* `cargo xtask ci` 통과.
* 두 테스트가 오라클 서버에서 실행되고 `mujoco` 없이는 이유를 출력하며 건너뛴다. 재생은
  `ES_V10_DATASET` 없이도 건너뛴다.
* 7.18절이 표 셋과 네 번째 발견과 판정을 담고, 사람의 결정을 위해 미해결 질문 16이 추가된다.
* 픽스처도, 크레이트 `src/`도, 해시도, 골든도 움직이지 않는다.

## 다음 패킷

**V11 — 제어 한 스텝은 제어 주기 하나다.** `BatchDomains::inference.period`를 Deployment IR의
`rate.control`과 장면의 물리 속도에서 유도하고, 타임스텝이 제어 주기를 나누지 못하는 장면을
거부하고, V1c의 명령으로 다시 수집하고 V2의 손잡이로 다시 학습해 V8의 100,000 스텝 수치와
비교한다. 미해결 질문 16이 대안(`LoadConfig::rate`, 스케줄이 아니라 물리를 바꾼다)과 기본값이
스케줄인 이유를 적어 둔다.
