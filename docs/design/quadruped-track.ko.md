# 사족보행 트랙 (Track B): 외부에서 학습하고, 가져오고, 여기서 실행한다

상태: 섹션 1–3은 패킷 M6/B1이 작성했다. 이 트랙의 첫 패킷이자 LOCAL-ONLY 패킷 — 장면 하나,
IR 문서 네 개, 그리고 그것들을 판정하는 오라클이 전부다. 아직 아무것도 학습하지 않았고 아무것도
가져오지 않았다. 고정된 업스트림 사실은 `docs/api-notes/mujoco-playground-quadruped.md`에 있다
(mujoco_playground 0.2.0 @ `124a73f`, menagerie @ `1b86ece`, brax 0.14.2). 이 노트는 그것을 되풀이
하지 않고, 그에 대해 우리가 무엇을 하는지를 적는다.

이 트랙이 기대는 스펙 섹션: §8 (Learning IR), §1.9 (절대 자르지 않는 것과 가장 먼저 잘리는 것),
§9 (Deployment IR과 Safety Plane), §17 (백엔드), §28.9 (M6이 존재하는 이유).

## 1. 왜 외부 RL과 임포트인가, 여기서 학습하지 않고

Electric Sheep는 **컴파일러이자 런타임**이지 학습 프레임워크가 아니다. §1.9는 네이티브 물리
솔버를 가장 먼저 잘릴 항목으로 놓았고, 네이티브 PPO 구현은 그 목록에조차 없다 — 애초에 범위가
아니었기 때문이다. Learning IR(§8)이 임포트 표면이다: 정책을 인코더·헤드·청커·역정규화기라는
타입 있는 노드로 표현하고, §8.7이 그 노드들을 PyTorch로 낮춰 같은 IR을 PyTorch에서 돌린 것이
정답이 된다. §8 어디에도 가중치가 여기서 만들어져야 한다는 말은 없다.

그래서 Track B는 이렇다. **`mujoco_playground`가 MJX/JAX PPO로 Go1 조이스틱 정책을 학습하고,
우리는 그 가중치를 Learning IR 그래프로 가져와 우리 Observation IR·Deployment IR·Safety Plane을
통해 실행한다.** 이것이 정직한 분업이고, 동시에 이 저장소에서 걷는 로봇에 이르는 가장 싼 길이다.

- 어렵고 우리에게 의견이 없는 부분은 업스트림이 이미 튜닝해 두었다: 열다섯 개의 셰이핑 보상 항,
  도메인 무작위화 범위, PD 게인, 관측 노이즈. 대조할 기준 없이 그것을 다시 유도하는 것은 같은
  목적지로 가는 더 어려운 길이다.
- 업스트림의 학습 물리는 **MuJoCo**(MJX)다. 우리 기준 백엔드도 MuJoCo(CPU)다. 접촉 모델·솔버·
  적분기가 같은 계열이고, 이것이 `legged_gym`이나 Isaac Lab(둘 다 PhysX로 학습한다, api-note
  섹션 4)보다 `mujoco_playground`를 택하는 가장 큰 이유다.
- 우리 스택에서 한 번도 검증되지 않은 부분을 건드린다: 열두 개 구동 관절을 가진 **부유 기저**
  로봇. 그 전의 모든 픽스처 — 진자, 2링크 팔, SO-101 — 는 고정 기저다. 섹션 3은 대부분 그래서
  드러난 것들의 목록이다.

이 저장소가 소유하는 것, 그리고 임포트가 건너뛸 수 없는 것: 우리 로더가 받아들이는 장면, 네 개의
IR 문서, 그 위의 해시 체인, 정책의 행동이 통과하는 Safety Plane, 결과를 판정하는 Evaluation IR.
그중 어느 것도 `mujoco_playground`의 것이 아니다.

## 2. 장면 파생본

`tests/fixtures/mjcf/go1_primitives.xml`은 mujoco_playground의 `go1_mjx_feetonly.xml`에
`scene_mjx_feetonly_flat_terrain.xml`에서 필요한 두 요소를 더한, 단일 파일 프리미티브 전용
파생본이다. 전체 열거는 `tests/fixtures/mjcf/go1_primitives.PROVENANCE.json`에 있고 프로비넌스
오라클이 그것을 읽는다. 형태는 이렇다.

- **치환:** 5개의 Unitree STL 메시를 참조하던 13개의 `class="visual"` geom이 여기서는 박스다.
  로봇에 가한 변경은 이것뿐이고, 동역학을 바꿀 수 없다: 그 geom들은 `contype=0 conaffinity=0`
  이고 모든 body가 명시적 `<inertial>`을 들고 있으며 MuJoCo는 geom 유도 관성보다 그것을
  우선한다. 충돌 geom은 업스트림에서 이미 프리미티브였다 — `feetonly`가 그런 뜻이다 — 따라서
  **충돌 형상은 하나도 근사되지 않았다**. 그것이야말로 물리 변경이 되었을 부분이다.
- **인라인:** 평지 바닥 geom과 `home` 키프레임. 우리 임포터는 `<include>`를 거부하고
  (`MjcfError::Include`), `home`은 리셋 자세이자 모든 행동의 영점이다
  (`motor_targets = default_pose + action * 0.5`).
- **제거:** `<sensor>` 블록 전체, 그 블록만 참조하던 다섯 개의 `<site>`, 그리고 `<light>` 하나.
  `crates/es-physics-backend/src/mjcf_out.rs`는 `jointpos`/`jointvel` 센서만 내보내므로 자이로나
  속도계를 선언한 장면은 MuJoCo에 닿기 전에 `check_requirements`에서 거부된다. 센서는 동역학을
  나르지 않으므로 물리 비용은 없다 — 다만 관측에 비용이 있다, 섹션 3을 보라.
- **바이트 단위 동일:** `<option>` 블록, `<default>` 트리 전체, 모든 관절, 모든 액추에이터, 모든
  충돌 geom, 모든 `<inertial>`, 그리고 `home`. `cargo test -p es-assets --test go1_provenance`가
  고정 커밋에 대해 이를 증명하며, 내려받은 바이트는 **파싱 전에** blake3를 검증한다(§25.1).

STL을 벤더링하지 않은 이유: 이 트랙의 어떤 오라클도 그리지 않는 형상을 위해 라이선스까지 포함해
5개의 바이너리 메시를 떠안아야 한다. SO-101 픽스처가 선례를 만들었고(`so101_provenance.rs`)
여기서는 그것을 그대로 따른다.

라이선스: 로봇은 menagerie의 `unitree_go1`, BSD-3-Clause, HangZhou YuShu TECHNOLOGY CO.,LTD.
mujoco_playground의 수정본과 장면 파일은 Apache-2.0이다. 둘 다 소스 형태의 수정 재배포를
허용한다. `go1_primitives.LICENSE`는 menagerie의 파일을 수정 없이 그대로 둔 것이다.

## 3. 네 개의 문서, 물리 일치 리스크, 그리고 아직 열려 있는 것

`tests/fixtures/quadruped/{task,observation,learning,deployment}.toml`은
`cargo test -p es --test cli -- --ignored regenerate_quadruped_documents`가 장면과 서로로부터
생성한다. 그래서 그 안의 어떤 해시도 어떤 관절 한계도 손으로 타이핑되지 않았다 — 팔 데모의
`regenerate_visible_learning_documents`가 선례다. 각 파일의 헤더가 그 파일이 무엇인지 말하고,
이 섹션은 그것들이 무엇이 *틀렸는지*를 말한다. 적어 둘 가치가 있는 쪽은 그쪽이다.

### 3.1 물리 일치, 그리고 오라클이 그것을 고정하는 방식

리스크: `iterations=1 ls_iterations=5 timestep=0.004 integrator=Euler`에 `eulerdamp`를 끈 상태로
MJX에서 학습된 정책을, 우리가 `parse_mjcf` → `scene_to_mjcf` → MuJoCo로 배포한다.
**그 왕복이 떨어뜨리는 모든 것은 정책이 본 적 없는 물리다.** 우리 CPU 백엔드를 MuJoCo 기본값인
선탐색 50회로 돌리거나 암시적 감쇠를 켜 두면 접촉 해결과 적분이 달라지고, 그것은 발 미끄러짐과
떨림으로 나타난다.

그중 둘은 실제였다. 이 패킷 전에는 `SceneDesc::PhysicsOptions`에 `ls_iterations` 필드가 없었고
`<option><flag>`는 경고만 남기고 버려졌다. 그래서 재생성된 MJCF는 MuJoCo에 `ls_iterations = 50`
(기본값)을 말하고 `eulerdamp`를 켠 채로 두었다. 이제 둘 다 실려서 나간다.
`the_emitted_mjcf_keeps_the_playground_option_block`이 회귀 테스트이고, Python이 필요 없으므로
PR CI에서 돈다.

둘이 더 실제였고, 패킷 M6/B1b 전에 여기 적혀 있던 문장 — "둘 다 행동이 0인 기립 궤적을 움직이지
않는다" — 은 틀렸다. geom `priority`와 `<position inheritrange>`는 둘 다 왕복에서 떨어지고 있었고,
스텝 오라클이 그중 첫 번째를 측정했다: `home`에서 물리 1,250스텝을 선 뒤, 같은 픽스처에 대해
우리 백엔드와 MuJoCo 직접 실행이 좌표별 최대 **|Δqpos| = 7.574e-3**, 임계값 1e-3의 일곱 배로
어긋났다.

MuJoCo 단독 절제 실험이 원인을 정확히 분리했다: `go1_primitives.xml`의 바닥에서 `priority="1"`
*만* 제거하면 7.574e-3이 마지막 자리까지 재현되고, `solimp`나 `inheritrange`를 제거하면 숫자가
전혀 변하지 않는다. `priority`가 없으면 MuJoCo는 접촉 쌍의 마찰을 `max`로 섞으므로, 바닥의
우선순위가 결정하는 대신 발의 `0.4`가 바닥의 `0.6`을 이기고, 자세가 다른 곳에 안착한다.

M6/B1b가 둘 다 싣는다. `Geom.priority: i32`(MuJoCo 기본값 0)는 MJCF 임포터가 파싱하고,
정준 씬 인코딩에서 `condim` 옆에 — 물리 의미이므로 조건 없이(§5.3) — 인코딩되며, 0이 아닐 때
`mjcf_out`이 내보낸다. `<position inheritrange>`는 MuJoCo 컴파일러가 해석하는 방식 그대로
임포트 시점에 액추에이터의 `ctrl_range`로 해석된다: 대상 관절 자신의 범위를 중점 기준으로
스케일한 값이다. 변경 후 같은 오라클이 같은 1,250스텝에서 **최대 |Δqpos| = 8.882e-16**을
측정한다 — 허용오차 안이라기보다 배정밀도 마지막 비트 수준의 일치다.

허용오차는 여전히 비트 동일이 아니라 선언된 `DeterminismTier::PhysicsMeaning` 허용오차다:
§4.3은 외부 백엔드가 tier 1을 선언하는 것을 금지하고, 오라클이 증명하는 것은 왕복에서 *의미*가
떨어지지 않는다는 것이지 독립된 두 바이너리가 비트 단위로 일치한다는 것이 아니다. 인코딩에
필드를 추가하면 모든 `scene_hash`와 그 하류의 모든 해시가 움직이므로,
`tests/fixtures/quadruped/*.toml`과 `tests/fixtures/visible-learning/*.toml`은 각자의
생성기(`regenerate_quadruped_documents`, `regenerate_visible_learning_documents`)로 재생성했다.
`SCENE_TAG`은 `es.scene.v1` 그대로다: 그것은 도메인 분리자이고, 이 저장소는 필드 추가로 그것을
올린 적이 없다(M6/B1이 같은 인코딩에 `ls_iterations`와 `eulerdamp`를 올림 없이 추가했다).

`crates/es-physics-backend/tests/go1_step.rs`(`#[ignore]`, `mujoco`가 있는 `ES_PYTHON` 필요)는
`home`에서 행동 0으로 250 제어 틱 × 5 서브스텝을 우리 백엔드로 밟고, NaN이 없으며 몸통이 여전히
0.20 m 위에 있음 — 즉 **선다** — 을 단언한 뒤, 같은 XML을 `mujoco`로 직접 밟아 `qpos`를
비교한다. MuJoCo가 자기 기본값이 아니라 우리 픽스처에서 `ls_iterations = 5`,
`eulerdamp = false`, `timestep = 0.004`를 읽었음도 단언하고, 두 경로의 물리 스텝 1,000회당
벽시계 시간을 관측값으로 출력한다(§12.4 — 아홉 지표 아니면 없음이고, 단일 `step/s` 숫자는 이
저장소가 하는 주장이 아니다).

### 3.2 관측을 아직 채울 수 없다 — 트랙의 첫 번째 블로킹 항목

Playground의 정책 입력은 48차원이다: `local_linvel(3)`, `gyro(3)`, `gravity(3)`,
`joint_pos(12) - default_pose`, `joint_vel(12)`, `last_action(12)`, `command(3)`. 우리
Observation IR은 정확히 그것을, 그 순서로 선언하고, 네 문서에 대해 `es ir check`는 통과한다.
그러나 우리 런타임은 그것을 *채울* 수 없다.

- `crates/es-eval/src/runner.rs::input_sources`는 관측 입력을 `qpos` 구간, MJCF 센서 구간, 또는
  이미지로 해석한다. 관절 속도는 닿지 않고(`qvel` 캡처가 없다), 기저 선속도와 자이로도 닿지
  않으며(센서를 제거했고 `mjcf_out`이 어차피 내보내지 못한다), 투영 중력은 어떤 노드도 계산하지
  않는 기저 쿼터니언의 회전이고, `last_action`과 `command`는 센서 값이 아니라 런타임 상태다.
- Task IR-D에도 대부분의 소스가 없고, 그래서 `task.toml`의 `ObservationSpec` 노드는 의도적으로
  연결되어 있지 않다. 채널을 선언하는 것 자체는 여전히 옳다 — §7.4가 바로 "Task IR이 선언하고
  Observation IR이 구현한다"이다 — 다만 IR-D가 그중 무엇도 계산하지 않는다.

그래서 오늘 이 네 문서에 대한 `es eval run`은 실행을 흉내 내지 않고 **이름을 대며 거부한다**
(§1.4). `quadruped_eval_run_names_the_observation_gap`이 그 거부 메시지를 고정하므로, 캡처
경로가 기저 상태를 갖게 되는 날 이 테스트가 실패하고 갱신된다 — 구멍 위에서 조용히 초록으로
남아 있지 않는다.

### 3.3 Safety Plane에 고정 기저 로봇의 상태가 들어간다 — 두 번째 블로킹 항목

`crates/es-eval/src/runner.rs::joint_state`는 plane에 `qpos[0..NJ]`와 `qvel[0..NJ]`를 준다.
고정 기저 팔에서는 그것이 구동 관절이다. Go1에서는 몸통의 자유 관절에 앞쪽 힌지 다섯 또는 여섯
개를 더한 것이다: plane이 기저 위치를 무릎의 한계에 대고 판정하게 된다. `deployment.toml`의
엔벨로프는 로봇에 대해 옳고, 거기에 올바른 열두 행을 먹이는 것은 `es-eval` 변경이다(선행 `NJ`
관례가 "`ModelInfo`에서 나온 구동 관절 인덱스"가 되어야 한다). 이 패킷의 범위 밖이다. 우회하지
않았고, 무엇을 끄는 식으로는 더더욱 우회하지 않았다(INV-12).

오늘 증명할 수 있고 실제로 증명하는 것:
`quadruped_bundle_runs_100_ticks_through_the_safety_plane`이 네 문서로 `policy.esb`를 만들고
`deployment.toml`로 만든 진짜 `SafetyPlane::<12, 1>`에 100 제어 틱을 통과시킨다. 두 번 돈다.
학습되지 않은 정책의 행동(`tanh`로 묶여 있어 20 ms 틱당 최대 ±0.5 rad의 명령 도약, 21 rad/s
관절에 대해 약 50 rad/s)은 100틱 모두 클램프되지만 폴백을 래치하지 않고 위치 한계를 벗어나지
않는다. 로봇이 물리적으로 따라갈 수 있는 명령(0.02 rad/틱)은 100틱 모두 `ActionSource::Policy`로
그대로 통과한다. 중요한 쪽은 두 번째다: 그것이 없으면 상수로 클램프되는 엔벨로프도 첫 번째를
"통과"한다 — 패킷 M5/V6이 팔에서 찾아낸 결함이다.

### 3.4 Learning IR은 아직 업스트림이 학습한 네트워크가 아니다

세 개의 간극이고, 전부 문서가 아니라 *낮추기(lowering)*에 있다.

1. **활성화 함수.** brax의 `MLP`은 `linen.swish`를 쓰고 `lower_to_torch`는 `nn.ReLU`를
   내보낸다. 둘 중 하나가 움직이기 전까지 가져온 가중치는 학습된 정책을 재현하지 못한다. 가장
   큰 단일 임포트 리스크이고 다음 패킷의 첫 질문이다.
2. **활성화 위치.** 업스트림은 `Dense(512) swish, Dense(256) swish, Dense(128) swish,
   Dense(24)`다. 우리는 `StateEncoder{Mlp hidden=[512,256], out_dim=128}` →
   `Linear(48,512) ReLU Linear(512,256) ReLU Linear(256,128)`에 헤드의 `Linear(128,12)`이
   붙는다: 선형 층 수는 업스트림과 같은 넷이지만 헤드 앞에 활성화가 없다.
   (`hidden = [512, 256]` + `out_dim = 128`이 우리 노드 집합에서 "512, 256, 128의 MLP"를 쓰는
   방식이다. `hidden = [512,256,128]`은 *다섯 번째* 선형 층이 된다.)
3. **`tanh`.** 업스트림의 결정론적 추론은 24차원 출력의 앞 절반에 대한 `tanh(location)`이고
   (`NormalTanhDistribution.mode`) 로그 스케일 절반은 쓰이지 않는다. 우리 노드 집합에는 활성화
   노드 자체가 없어서 `tanh`는 표현되지 않는다. 행동 포트의 `Normalized { lo = -1, hi = 1 }`
   단위와 Safety Plane의 클램프가 `tanh`의 *범위*는 주지만 모양은 주지 않는다.

*정확한* 쪽: 관측 정규화기는 `Normalizer { Forward, MeanStd }`에 자리표시자
`mean = 0, std = 1` — brax의 `running_statistics`가 내보내는 형태이고 임포트 패킷이 체크포인트
에서 채운다 — 이고, 행동 역정규화기는 `Normalizer { Inverse, MeanStd }`에
`mean = default_pose`(장면의 `home` 키프레임에서)와 `std = 0.5`(`action_scale`)로,
`motor_targets = default_pose + action * action_scale` 그 자체다. 가중치는 전부 0인 해시를 가진
`safetensors` 참조이고, pickle 경로는 어디에도 없다(INV-16).

### 3.5 문서가 말할 수 없는 나머지

- **종료.** 업스트림은 업라이트 벡터의 z 성분이 음수가 되면 종료하고, 보통의 두 번째 가드는 기저
  높이다. 둘 다 표현 불가능하다: `es-env`의 보상/종료 원뿔은 관절의 *첫* `qpos` 인덱스를
  묶는데(`crates/es-env/src/plan.rs::joint_leaf`) 자유 관절의 첫 인덱스는 x다. `task.toml`은
  타임아웃과 "기저가 경기장을 벗어남"(|x| > 3 m)으로 종료하고 헤더에 그렇게 적어 두었다.
- **보상.** 업스트림의 `exp(-err² / tracking_sigma)`는 `MathFn`(원뿔에 없다)과 명령 속도를 뺄
  상수 리프(IR-D에 없다)가 필요하다. 우리 것은 명령 상한으로 정규화한 전진 속도다. 오늘은 비용이
  없다 — 업스트림의 열다섯 항이 `mujoco_playground`에서 정책을 학습시킨다 — 다만 이것이 *우리*
  평가 하네스가 쓸 보상이다.
- **조이스틱 명령.** 업스트림은 OU 비슷한 일정으로 에피소드 중간에 재샘플링한다. IR-D가 말할 수
  있는 것은 "에피소드당 한 번 뽑는다"이고, 세 개의 `Randomization` 노드가 그것이다. 중간
  재샘플링은 IR-C(제어 그래프, §6.2)의 문제다.
- **관측 노이즈.** 업스트림은 자기 env 안에서 채널별 *균등* 노이즈를 더한다. `AugmentKind`에는
  `GaussianNoise`가 있고 균등 변형은 없으며, INV-15는 어차피 평가 중에 `Augment` 노드를 끈다.
  표현하지 않았다. 그것은 우리가 아니라 학습기의 몫이다.

### 3.6 사람에게 남기는 열린 질문

1. `StateEncoder{Mlp}`에 대해 `lower_to_torch`를 swish로 옮길 것인가, 노드에 활성화 파라미터를
   더할 것인가(IR 변경이고, §1.9의 "확장점은 일곱 개뿐"이 두 번 생각하게 만든다), 아니면 ReLU
   아래에서 가져온 정책을 재튜닝할 것인가? (3.4 #1)
2. `tanh`는 Learning IR 노드인가, `PolicyHead` 파라미터인가, 낮추기가 적용하는 행동 단위의
   속성인가? (3.4 #3)
3. "구동 관절 인덱스"는 누구 것인가 — `ModelInfo`인가, Deployment IR인가, Task IR
   `ObsSource::JointState`의 새 필드인가? 3.2와 3.3이 같은 답을 기다린다.
4. Go1인가 Go2인가? 업스트림에는 Go1만 있다(api-note 섹션 1). Go2 트랙은 `Joystick`을
   menagerie의 `unitree_go2`(이미 프리미티브 충돌이다)에 손으로 이식하는 것을 뜻한다. Go1 정책이
   한 번 돌기 전에 내릴 결정이 아니다.
