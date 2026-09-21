# 플랜 V — "보이는 학습": 누구나 볼 수 있는 엔드투엔드 데모 — 설계

Spec: §28.3 및 §28.7 게이트 7 (Franka pick&place 엔드투엔드), §1.2 (작업 패킷), §1.4 (오라클 우선),
§1.5 (컨텍스트 예산), §1.9 (절대 자르지 않을 것 / 가장 먼저 자를 것), §2.3–§2.5 (Python이 허용되는 곳),
§4.2 (레이어링), §4.3 (백엔드), §5.1 (IR 경계), §5.3 (해시 체인), §6.3 (랜덤화, 리셋),
§7.2 (`ImageSpec`), §8.3 및 §8.7 (Learning IR, 정책 런타임), §9.2–§9.4 (Safety Plane), §10.1–§10.4
(Evaluation IR), §12.4 (단일 `step/s` 금지), §15.2 (타일 아틀라스), §17.2 (백엔드 매핑), §18.1
(정수 시간), §19.2 (데이터셋 동일성), §25.1 (신뢰 경계), §26.1–§26.2 (검증, CI 티어).
불변식: INV-11, INV-12, INV-13, INV-14, INV-16, INV-17.
패킷: `docs/packets/M5/V0-scene-task.md`, `V0b-render-in-the-loop.md`, `V1-expert-dataset.md`,
`V2-act-training.md`, `V3-perturb-safety.md`, `V4-video.md`.

이 설계 노트의 절은 "섹션 N"으로 인용하고, `§N`은 언제나 스펙을 뜻한다.

## 1. 무엇이며, 어떤 게이트를 닫는가

실행 가능한 데모 하나: 시뮬레이션 속 SO-101 팔이 큐브를 바구니에 넣는 스크립트 시연을 보고, 그
데이터셋으로 정책을 학습하고, 학습된 정책을 랜덤화된 장면에서 평가하고, 그 결과가 성공률 오버레이와
Safety Plane이 동작을 클램프할 때마다 붉은 테두리가 뜨는 4x4 그리드 영상으로 나온다.

§28.7 게이트 7 ("Franka pick&place 엔드투엔드")은 `docs/reviews/M1.md:30`에서 **미충족**으로 기록되었다
("franka names IR fixtures only; no e2e run exists"). 이후 어떤 리뷰도 이를 다시 다루지 않았다:
`docs/reviews/*.md` 전체에서 `gate 7` / `franka` / `pick&place`를 grep하면 그 한 줄만 나오고 M2, M3, M4
에는 아무것도 없다. 플랜 V는 이를 닫는 패킷 묶음이며, 사용자가 이미 결정한 두 가지 대체를 적용한다:
**Franka가 아니라 SO-101** (나중에 실제로 구입하면 시뮬 데모가 그대로 이어지는 저가 팔), 그리고
**일반적인 pick&place가 아니라 큐브를 바구니에 넣기**.

| 항목 | 패킷 | 오라클 | V 이후 |
|---|---|---|---|
| SO-101 장면 + Task / Observation / Deployment IR | V0 | 상류 모델 교차검증, `es task compile`, MuJoCo 로드 | 어디서나 검증 가능 (fetch: 네트워크) |
| env 루프에서 구동되는 렌더러, 프레임 디스크 기록, 이미지 관측 포트 | V0b | CPU 레퍼런스 골든, GPU/CPU 일치 | PR 티어 CPU; GPU는 장치 필요 |
| 스크립트 전문가 + 시연 데이터셋 | V1 | 전문가 성공률, LeRobot 왕복 | `mujoco` 필요 |
| PyTorch에서 학습된 Learning IR 로워링, `policy.esb`로 패킹 | V2 | 로워링 모듈 동일성, 손실 감소, `eval run` 로드 | `torch` 필요 (서버) |
| 섭동 스위트 + 스텝별 Safety Plane 이벤트 | V3 | Evaluation IR 리포트, 비공허(non-vacuous) 클램프 이벤트 | `mujoco` + `torch` 필요 |
| 프레임 그리드 -> 오버레이 -> mp4 | V4 | 고정 프레임에서 바이트 동일한 모자이크 | 모자이크는 어디서나; mp4는 `cv2` 필요 |
| 임의의 벽시계·처리량·학습시간 수치 | — | 존재하지 않는 측정 루프 | `Target / Status: unverified` |
| 선언된 랜덤화 범위를 넘어선 일반화 | — | 범위 밖 | `Target / Status: unverified` |

## 2. 계획을 바꾼 조사 결과

이 절의 모든 내용은 가정이 아니라 실제로 읽은 것이다. 줄 번호는 이 노트 작성 시점 기준이다.

### 2.1 Rust 물리 스테퍼는 없다; 백엔드는 Python 서브프로세스다

`PhysicsBackend` (`crates/es-physics-core/src/backend.rs:219-255`)의 프로덕션 구현은 셋이고
(`mujoco.rs:218`, `mjwarp.rs:175`, `newton.rs:200`), 셋 다 `Command::new(python).args(["-c", script])`로
줄 단위 JSON을 주고받는다 (`crates/es-physics-backend/src/proc.rs:201-229`); float은 십진 JSON으로 건넌다
(`proc.rs:97-105`). `es-physics-cpu`도 `es-physics-gpu`도 없다;
`crates/es-physics-core/src/lib.rs:7-8`이 직접 그렇게 말한다 ("It contains no engine").

제어 스텝 하나마다 `es eval run`은 **MuJoCo 왕복 3회 + Torch 왕복 1회**를 치른다:
`set_ctrl` (`crates/es-env/src/env.rs:258`), `step` (`env.rs:262`), 무조건적인 전체 `state` 풀
(`crates/es-physics-backend/src/mujoco.rs:354`), 그리고 `policy.infer`
(`crates/es-eval/src/runner.rs:374` -> `crates/es-policy/src/torch_runtime.rs:353`). 코드 자신이 이것이
의도적이며 처리량 경로가 아니라고 적어 두었다 (`mujoco.rs:5-6`, `torch_runtime.rs:6-8`).

**결론.** 4x4 그리드는 16-env 배치가 아니라 **독립적인 단일 env 에피소드 16개를 오프라인에서
모자이크한 것**이다. `Evaluation::run`은 `BatchDomains::single_env()`를 하드코딩하고
(`crates/es-eval/src/runner.rs:132` -> `crates/es-env/src/scheduler.rs:58-65`), `es eval run`에는
`--envs` 플래그가 없으며 (`crates/es/src/cmd/eval.rs:193-229`), `MuJoCoCpuBackend`는 `max_envs: 1`을
선언하고 (`mujoco.rs:43`) `n_envs > 1`은 독립 `MjData`에 대한 직렬 Python 루프로 흉내낸다
(`crates/es-physics-backend/python/mujoco_ref.py:138-140`). 배치 도메인을 키우는 일은 이 데모와 별개이며
플랜 V는 건드리지 않는다.

플랜 V의 어떤 패킷도 스텝 레이트, 에피소드 시간, 학습 시간을 말하지 않는다. `es bench`는 모든 필드를
`unmeasured`로 출력하고 (`crates/es/src/cmd/bench.rs:163-196`) §12.4는 단일 수치를 폐기했다.

### 2.2 `es-render`는 완성되어 있고, 헤드리스이며, 고립된 섬이다

워크스페이스의 어떤 크레이트도 `es-render`에 의존하지 않는다. 그런데도 내용은 진짜다: `Renderer::new` /
`upload_scene` / `render(&[CameraView]) -> Atlas` (`crates/es-render/src/renderer.rs:175`, `:234`, `:315`),
N개 카메라의 닫힌 형식 타일 아틀라스 (`crates/es-render/src/atlas.rs:52-57`, §15.2), `Atlas::read_tile`
(`renderer.rs:82-138`)과 `Buffer::download` (`crates/es-gpu/src/buffer.rs:152-163`)를 통한 GPU->호스트
리드백, 그리고 GPU 경로가 비트 단위로 맞춰야 하는 `tests/golden/render/`의 CPU 레퍼런스 골든
(`crates/es-render/tests/render.rs:197-209`).

**헤드리스는 가정이 아니라 Vulkan 레벨에서 확인했다.** 인스턴스는 활성 확장 없이 만들어지고
(`crates/es-gpu/src/instance.rs:59-64`), 디바이스도 마찬가지이며 (`crates/es-gpu/src/device.rs:80-82`),
큐는 `COMPUTE`만으로 고른다 (`instance.rs:84`). surface/swapchain 모듈 자체가 없다
(`crates/es-gpu/src/lib.rs:21-28`). `winit`은 `es-editor`에만 나온다. 오라클 서버에서 측정 (2026-09-14):
`vulkaninfo --summary`가 디스플레이 없는 SSH 환경에서 `NVIDIA GeForce RTX 4090`을 `apiVersion 1.4.341`로,
그리고 `llvmpipe` 폴백을 보고한다.

데모를 막는 구멍은 넷이고, 소스는 수치를 지어내는 대신 넷 모두를 명시한다:

| 구멍 | 근거 |
|---|---|
| `Tile` -> 디스크 | 유일한 `fs::write`가 `#[ignore]`된 골든 생성기에 있다 (`crates/es-render/tests/render.rs:173`) |
| `Renderer` -> Observation IR `ImageInput` | `crates/es-eval/src/runner.rs:421-428`이 이미지 입력에 `EvalError::Plan`을 반환한다: "0을 먹이면 숫자는 나오지만, §10.1 표의 틀린 숫자는 표가 없는 것보다 나쁘다" |
| 메시 geom | `crates/es-render/src/scene.rs:201`이 `Shape::Mesh`를 거부한다 ("에셋 리졸버가 필요함") |
| 애니메이션 | `TriScene::from_scene`이 **정적** `world_poses(scene)`를 삼각형에 구워 넣는다 (`scene.rs:56-65`) |

`MultiViewPack` (`crates/es-ir/src/observation.rs:378-381`)은 선언만 되어 있고 플랜 로워링이 거부한다
(`crates/es-compile/src/plan.rs:545`). 따라서 데모는 **카메라 하나**를 쓴다.

### 2.3 MJCF 파이프라인은 메시·include·keyframe·카메라를 실어 나르지 못한다

장면 경로는 MJCF 텍스트 -> `SceneDesc` (`crates/es-assets/src/mjcf/mod.rs`) -> 다시 MJCF 텍스트
(`crates/es-physics-backend/src/mjcf_out.rs`) -> `mujoco.MjModel.from_xml_string`이다. 좁은 지점이 셋:

- **`<include>`는 곧바로 거부된다.** 루트에서도 (`crates/es-assets/src/mjcf/mod.rs:247-251`) 바디 안에서도
  (`mod.rs:569-573`). Menagerie의 `scene_box.xml`은 `<include>` 래퍼라 그대로 쓸 수 없다. 데모 장면은
  평평한 단일 파일이어야 한다.
- **`type="mesh"` geom은 `PhysicsError::Unsupported`다** (`mjcf_out.rs:295`, `mjcf_out.rs:551-556`로 고정),
  `es-render`도 거부한다 (`scene.rs:201`). 따라서 메시를 포함한 모델은 이 파이프라인에서 MuJoCo에도
  렌더러에도 도달하지 못한다.
- `<keyframe>`, `<equality>`, `<contact>`, `<visual>`은 경고로 파싱되고 버려진다 (`mod.rs:208-218`);
  사이트·카메라·머티리얼·텐던은 재출력되지 않는다 (`mjcf_out.rs:9-11`).

이것이 에셋 정책(섹션 4)을 결정한다: 상류 메시 모델이 아니라 **프리미티브 전용** 파생본.

### 2.4 데이터 경로는 픽셀을 전혀 기록하지 않으며, LeRobot v2.1을 쓴다

`es loop collect`는 스텁이 아니라 진짜이고 (`crates/es/src/cmd/loop.rs:146-155`, `:223`), 실제 `Env`를
스텝한다 (`crates/es-data/src/collect.rs:355-357`). `mujoco`가 있는 Python과 `torch`가 있는 Python이
둘 다 발견되지 않으면 `SKIPPED` + exit 3을 낸다 (`loop.rs:194-202`).

스텝마다 `observation.state` (`qpos ‖ qvel`, f32), `action` (**Safety Plane 통과 후**의 제어값),
`reward`, `intervention`, `action_source`, `timestamp`를 기록한다 (`collect.rs:514-552`). `Episode`
(`crates/es-env/src/episode.rs:73-96`)에는 이미지 버퍼가 아예 없다. 카메라를 선언하면 `info.json`에
`video` 피처와 경고가 생기고, 비디오 맵은 비어 있으며, mp4 참조는 의도적으로 매달린 채 남는다
(`collect.rs:496-510`, `:549-550`).

포맷: `codebase_version: "v2.1"`, 하드코딩 (`crates/es-data/src/lerobot/meta.rs:148`), 레이아웃
`data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` (`meta.rs:155`),
`parquet 59.3`을 `default-features = false`로 사용 — `arrow` 없음, **무압축**
(`crates/es-data/Cargo.toml:22-24`, `columns.rs:146`). `docs/api-notes/lerobot-dataset.md:3-8`은
**데이터셋 포맷에 대해 어떤 `lerobot` 버전도 고정되어 있지 않으며** 노트 전체가 기억에서 복원된 것이라고
분명히 적고 있다. 기록된 데이터셋을 실제 패키지로 왕복시키는 오라클은 없다: 저장소에서 유일한 `lerobot`
import는 ACT 정책 오라클 `crates/es-policy/python/act_ref.py`뿐이다.

**결론.** V1은 (a) 수집이 프레임을 쓰도록 가르치고, (b) 없는 오라클을 추가해야 한다: 서버의 실제
`lerobot`으로 데이터셋을 되읽기. v2.1 / v3 문제는 논쟁이 아니라 그 오라클이 결정한다 — 섹션 12 미해결
질문 2.

### 2.5 학습 루프는 어디에도 없고, `lower_act`는 평가될 수 없다

`AdamW|optimizer|backward|def train|fn train`를 grep하면 다이제스트 **슬롯**
(`crates/es-data/src/identity.rs:194`), 0 플레이스홀더 (`collect.rs:654`), 셰이더 옵티마이저만 나온다.
`es learn`도 `es train`도 서브커맨드에 없다 (`crates/es/src/main.rs:48-61`).
`crates/es/src/cmd/loop.rs:5-7`이 직접 말한다: "`es loop train` deliberately does not exist: the training
run is on the Python side of §2.3's split".

로워링 경로는 둘인데, `eval run`에서 닿을 수 있는 것은 하나뿐이다:

| 경로 | 입력 | 도달 경로 |
|---|---|---|
| `lower_to_torch(graph: &LearningGraph)` (`crates/es-policy/src/lower/torch.rs:334`) | IR | `TorchRuntime::load` (`torch_runtime.rs:340`), 즉 `es eval run --policy` (`crates/es/src/cmd/eval.rs:387-393`)와 `es loop collect` (`loop.rs:208`) |
| `lower_act(cfg: &ActConfig)` (`crates/es-policy/src/lerobot.rs:467`) | LeRobot `config.json` | `TorchRuntime::load_lowered`뿐이고, 그 유일한 호출자는 `lower_act` 자신이다 (`torch_runtime.rs:278-284`) |

`docs/reviews/M4.md:101-103`이 이 우회를 기록한다. 플랜 V에 대한 결과는 리뷰가 적은 것보다 더 날카롭다:
`load`가 번들의 `LearningGraph`에서 다시 로워링하므로 **`lower_act`로 만든 번들은 `es eval run`으로
실행할 수 없다**.

**결정 (섹션 6).** 플랜 V는 LeRobot의 ACT가 아니라 **IR이 소유한 ACT 형태의 그래프**를 학습한다.
`LearningGraph`는 이미 그것을 표현한다 — `es_ir::learning::testing::act_like`
(`crates/es-ir/src/learning.rs:1090-1152`)가 `ArchKind::Act`로 `VisionEncoder -> StateEncoder -> Fusion
-> TemporalEncoder -> PolicyHead -> ActionChunker`를 만들고, `lower_to_torch`는 그 노드를 전부 로워링한다
(`lower/torch.rs:551`, `:587`, `:617`, `:659`, `:756`, `:769-807`). 이렇게 하면 §1.4의 "동일한 IR을
PyTorch에서 돌린 것이 ground truth"가 문자 그대로 참이 된다 — Python이 최적화하는 모듈이 `eval run`이
추론하는 모듈과 바이트 단위로 같기 때문이다 — 그리고 `lower_act` 부채는 M4가 둔 자리에 그대로 남는다.
플랜 V는 그것을 고치지 않는다; 미해결 질문 1.

### 2.6 Safety Plane은 이미 루프 안에, 스텝마다 있다

`Evaluation::run`은 Deployment IR로부터 셀마다 `SafetyPlane<NJ, H>`를 만들고
(`crates/es-eval/src/runner.rs:149`), `policy.infer`와 `env.step` 사이에서 `safety.validate`를 호출한다
(`runner.rs:374-385`). `SafetyCounters`와 `ViolationKind`는 이미 메트릭이 읽는 것이고
(`crates/es-eval/src/metrics.rs:13`, `:144`), `es-data`는 이미 카운터 델타로 스텝의 액션 소스를
분류한다 — `Policy | Clamped | Fallback | Human` (`crates/es-data/src/collect.rs:461-471`).

**따라서 붉은 오버레이에는 새로운 안전 표면이 필요 없다.** V3는 기존 스텝별 분류를 프레임 인덱스와 함께
내보낸다; `validate`를 다르게 부르지 않고, 무엇도 비활성화하지 않으며 (INV-12), `validate`의 시그니처도
그대로다 (INV-13). `es-safety`는 어떤 의존성도 얻지 않으므로 INV-11도 영향이 없다.

### 2.7 랜덤화: Task IR은 큐브를 옮길 수 있지만 Evaluation IR은 못 한다

`RandomizationPlan`은 타깃을 `Target::Qpos(i)` / `Qvel(i)`로 해석해 리셋 때 기록한다
(`crates/es-env/src/randomize.rs:19-29`, `:109-110`). 큐브는 free joint이므로 그 월드 포즈가 곧 `qpos`다 —
큐브 포즈 랜덤화는 Task IR `Randomization` 노드(§6.3)이고 지금도 동작한다.

Evaluation IR 쪽은 아니다: `PerturbationKind::ObjectPose`는 "`Env::reset`이 상태 오버라이드를 받지
않는다"는 이유로 `Unsupported`이고 (`crates/es-eval/src/perturb.rs:196-200`), 모든 시각 계열
(`LightIntensity`, `LightDirection`, `ColorTemperature`, `Occluder`, `CameraExtrinsic`,
`CameraIntrinsic`)은 렌더러가 없어서 `Unsupported`다 (`perturb.rs:180-195`). 동작하는 것은 다섯:
`ObservationDelay`, `ActionDelay`, `FrameDrop`, `TorqueNoise`, `Backlash` (`perturb.rs:159-179`).

`Target::Scale(BodyMass | GeomFriction | ActuatorGain)`은 **뽑아서 기록되지만 백엔드로는 전혀 전달되지
않는다** — "`PhysicsBackend`에는 파라미터 API가 없다" (`randomize.rs:24-26`). 즉 질량·마찰 랜덤화는 현재
기록상의 거짓말이며, V0는 그것을 선언해서는 안 된다.

**결론.** V3의 스위트는: 기준 셀, 지원되는 다섯 섭동 계열, 그리고 — V0b가 렌더러를 붙인 뒤 — 조명 계열
둘. 거부 사유가 사라지므로 구현 가능해진다. V3는 `LightIntensity`와 `LightDirection`만 구현하고,
`ColorTemperature`, `Occluder`, `CameraExtrinsic`, `CameraIntrinsic`은 `Unsupported`로 남긴다
(미해결 질문 4).

### 2.8 역기구학은 어디에도 없다

`crates/`, `python/`, `xtask/`에서 `inverse_kinematic|jacobian|damped_least|mj_jac|pinv`를 grep하면
MJCF의 `<option jacobian=>` 솔버 속성만 나온다 (`crates/es-assets/src/mjcf/mod.rs:422-426`,
`crates/es-assets/src/scene.rs:370`). `es-math`는 `approx`, `conventions`, `reduce`, `scalar`, `simd`이고
`es-actuator`는 전달 모델뿐이다.

하지만 순기구학은 이미 공짜다: `StateView`가 모든 바디의 `xpos` (`n_envs * nbody * 3`)와 `xquat`
(`n_envs * nbody * 4`)를 싣고 있고 (`crates/es-physics-core/src/backend.rs:165-168`), `ModelInfo.body`가
바디 id를 행으로 매핑한다 (`backend.rs:126-127`).

**결정 (섹션 5).** 전문가는 damped-least-squares도 `mj_jac`도 아닌 **닫힌 형식 IK**를 쓴다. SO-101의
체인 구조가 이것을 편법이 아니라 더 작은 해법으로 만든다: `shoulder_pan`은 베이스 요(yaw)이고
`shoulder_lift`, `elbow_flex`, `wrist_flex`는 그 요가 쓸고 지나가는 평면 안의 연속된 세 피치다
(섹션 4.2). 즉 베이스 회전 + 접근 피치를 고른 평면 3R 팔이며, 야코비안도 반복도 수렴 실패 모드도 새
의존성도 없이 수십 줄이면 끝난다. DLS 솔버라면 백엔드가 노출하지 않는 야코비안이 필요하고, Python 쪽
`mj_jac`은 아무 이득 없이 전문가를 Rust 코어 밖으로 내보낸다.

### 2.9 오라클 서버: 2026-09-14 측정

SSH로, 읽기 전용. 아무것도 설치하거나 변경하지 않았다.

| 항목 | 결과 |
|---|---|
| GPU | `NVIDIA GeForce RTX 4090`, 24564 MiB |
| Vulkan | `~/VulkanSDK/x86_64/bin`의 `vulkaninfo`, `apiVersion 1.4.341`, 헤드리스, 디스플레이 없음 |
| `ffmpeg` 바이너리 | **없음** (`command -v ffmpeg` 빈 출력; `ffprobe`, `gst-launch-1.0`, `convert`도 없음) |
| FFmpeg 공유 라이브러리 | 있음: `libavcodec62` .. `libavutil60`, `7:8.0.1-3ubuntu2` |
| `~/venvs/es` | `torch 2.14.0+cpu`, `torchvision 0.29.0+cpu`, `mujoco 3.13.0`, `mujoco-warp 3.13.0`, `warp-lang 1.17.0`, `newton 1.6.0`, `diffusers 0.40.0`, `pillow 12.3.0`, `numpy 2.5.2` |
| `~/venvs/es-lerobot` | `lerobot 0.6.1`, `torch 2.11.0+cpu`, `torchvision 0.26.0+cpu`, `opencv-python-headless 4.13.0.92`, `pillow 12.3.0`, `numpy 2.2.6` |
| `torch.cuda.is_available()` | **두 venv 모두 `False`** — 둘 다 `+cpu` 빌드 |
| `import lerobot.datasets` | **`ImportError: 'datasets' is required but not installed`**; `pyarrow`도 없음 |
| `cv2.VideoWriter` mp4 | `mp4v`는 열리고 기록됨 (10프레임 -> 1450바이트); `avc1`과 `H264`는 **실패** — 링크된 유일한 H.264 인코더가 `h264_v4l2m2m`이고 장치를 찾지 못함 |
| MuJoCo Menagerie | 서버에 없음 |

이 중 셋이 계획을 바꾸고, 둘은 사람이 처리해야 할 선행 조건이다:

1. **ffmpeg 바이너리 없음.** 인코더는 `~/venvs/es-lerobot`의 `cv2.VideoWriter`, 코덱은 `mp4v`
   (MPEG-4 Part 2), 컨테이너는 `.mp4`다. 위에서 검증했다. H.264는 아니다 (미해결 질문 5).
2. **4090 위의 torch가 CPU 전용.** ResNet-18 백본을 가진 ACT 형태 그래프를 CPU에서 학습하는 것이
   플랜 V의 일정 리스크다. 사람의 선행 조건이며, 섹션 6.4는 어느 쪽이든 데모가 돌아가도록 이미지를
   작게 유지하고 학습 시간을 말하지 않는다.
3. **`lerobot[dataset]`이 설치되어 있지 않아** `lerobot.datasets`가 import조차 되지 않는다. 사람이
   설치하기 전까지 V1의 데이터셋 오라클은 `SKIP`이다. 사람의 선행 조건.

### 2.10 컨텍스트 예산

`cargo xtask context-budget`, 2026-09-14: 모든 크레이트 `OK`; `es-ir`은 **5,947 / 6,000** 코드 줄로
목표까지 53줄 남았다. 예산은 인라인 테스트 모듈 밖의 `src/` 아래 `*.rs`만 센다
(`xtask/src/context_budget.rs:83-156`) — 픽스처, `tests/`, Python 스크립트, 골든은 무료다.

**따라서 플랜 V의 어떤 패킷도 `es-ir`이나 `es-ir-types`에 한 줄도 추가하지 않는다.** 데모에 필요한 것은
이미 표현 가능하다: `TaskNode::Terminate { kind: TerminationKind::Success }`
(`crates/es-ir-types/src/expr.rs:104-108`), `Randomization` 타깃 (섹션 2.7), `MetricSpec::SuccessRate`
(`crates/es-eval/src/metrics.rs:32`, `:97-100`), 그리고 6노드 ACT 형태 `LearningGraph` (섹션 2.5).
각 패킷의 `forbidden`이 이를 명시한다.

## 3. 파이프라인

```
robotstudio_so101 (상류, 고정)                V0
  └─ 프리미티브 전용으로 파생 ─▶ tests/fixtures/mjcf/so101_pick_place.xml
                                     │
                    es-assets ──────▶ SceneDesc ─┬─▶ mjcf_out ─▶ MuJoCo (python 서브프로세스)
                                                 └─▶ TriScene ─▶ es-render                V0b
                                                                    │
 task.toml / observation.toml / deployment.toml (V0) ──▶ policy.esb │
                                     │                              │
                 es loop collect --expert (V1) ◀── ScriptedExpert ──┤
                                     │                              │
                     LeRobot v2.1 데이터셋 + 프레임                  │
                                     │                              │
             es policy lower ─▶ EsPolicy(nn.Module) ─▶ train_act.py (V2, 서버)
                                     │
                     model.safetensors ─▶ es policy pack ─▶ 학습된 policy.esb
                                     │
             es eval run --policy --frames (V3) ─▶ report.json + frames/ + events.json
                                     │
                        es video (V4) ─▶ 모자이크 + 오버레이 ─▶ demo.mp4
```

데모의 증거가 되는 산출물은 넷이다: `report.json` (스위트별 성공률), 프레임 디렉터리,
`events.json` (스텝별 액션 소스), `demo.mp4`.

## 4. 에셋 정책

### 4.1 무엇을 벤더링하고 무엇을 받아오는가

상류는 `google-deepmind/mujoco_menagerie`의 **`robotstudio_so101`**, **Apache-2.0**이며 ("This model is
released under the Apache License 2.0", `robotstudio_so101/README.md`), MuJoCo 3.1.3 이상이 필요하다.
해당 경로의 고정 커밋: **`ac6b2b09983786f3036cab1000221017fa2193b4`** (2026-05-18). 다른 후보인
`trs_so_arm100`은 SO-ARM100이고, 사용자가 고른 SO-101은 `robotstudio_so101`이다.

디렉터리는 약 17.5 MB다: `so101.xml` 16,670 B, `scene.xml`, `scene_box.xml`, `so101.png` 1,010,575 B,
그리고 `assets/` — STL 19개, 합계 17,230,580 B.

**디렉터리를 벤더링하는 것도, 런타임에 받아오는 것도 답이 아니다. 섹션 2.3이 말하듯 메시 모델은 이
파이프라인을 아예 통과하지 못하기 때문이다.** 정책은 이렇다:

- **벤더링**: `tests/fixtures/mjcf/so101_pick_place.xml` (약 10 KB, 평평한 단일 파일, `<include>` 없음).
  상류의 기구학 체인, 관절 축, 범위, 관성, `class="collision"` 프리미티브 geom을 그대로 두고
  `class="visual"` 메시 geom을 전부 제거한 파생본에 큐브·바구니·카메라 하나를 더한 것. 함께
  `tests/fixtures/mjcf/so101_pick_place.LICENSE` (상류 Apache-2.0 원문)와
  `so101_pick_place.PROVENANCE.json` — 상류 저장소, 경로, 커밋, 상류 `so101.xml`의 blake3, 파생 규칙.
  시각 메시를 버리는 것이 바로 이 파일을 벤더링 가능한 크기로 만드는 이유이며, 충돌 geom은 이미
  프리미티브다. 실제로 만든 파일에서는 각 링크가 잃어버린 메시 geom 대신 시각 전용 프리미티브
  **하나**씩 (`contype=0 conaffinity=0`, 이름 `<link>_shell`) 얻어서 렌더러에서도 팔로 알아볼 수 있다.
  `camera_mount_shell`은 대체한 메시의 0.012 kg을 그대로 들고 있고 나머지 링크에는 모두 명시적
  `<inertial>`이 있으므로, 추가된 shell이 질량 특성을 바꾸지 않는다. 권위 있는 목록은 매니페스트의
  `derivation` 배열이고, 그것을 검사하는 것이 `so101_provenance.rs`다.
- **17 MB는 절대 벤더링하지 않고 받아온다.** V0의 프로비넌스 오라클이 고정 커밋의 상류 `so101.xml`을
  받아 매니페스트의 blake3와 대조하고, 우리 파생본이 기구학적으로 동일함을 단언한다: 같은 순서의 같은
  관절 이름, 같은 축, 같은 범위, 같은 바디 오프셋과 방향, 같은 질량과 관성을 파싱된 `SceneDesc`의
  정확한 `f64` 비교로. 네트워크도 캐시도 없으면 사유와 함께 `SKIP`. 받은 바이트는 `target/` 아래에만
  두고 트리에는 넣지 않는다.

이것이 에셋 정책의 "커밋 + blake3로 고정된 fetch" 쪽이며, 빌드 단계가 아니라 **오라클**로 쓰인다:
데모 경로의 어떤 것도 네트워크를 필요로 하지 않는다.

### 4.2 상류에서 온 기구학 체인

관절 6개, `position` 액추에이터 6개, 카메라 하나 (`wrist_cam`), 사이트 `baseframe`·`gripperframe`,
`so101.xml`에는 keyframe 없음. `<compiler angle="radian" meshdir="assets" autolimits="true"/>`,
`<option integrator="implicitfast" timestep="0.005" cone="elliptic" iterations="10" ls_iterations="20"
impratio="10"/>`.

| 바디 | pos | quat (wxyz) | 관절 | 축 | ctrlrange |
|---|---|---|---|---|---|
| `base` | `0 0 0` | `1 0 0 0` | `shoulder_pan` | `0 0 1` | -1.91986 .. 1.91986 |
| `shoulder` | `0.0388353 0 0.0624` | `0 0 -1 0` | `shoulder_lift` | `0 0 1` | -1.74533 .. 1.74533 |
| `upper_arm` | `-0.0303992 -0.0182778 -0.0542` | `1 -1 -1 -1` | `elbow_flex` | `0 0 1` | -1.69 .. 1.69 |
| `lower_arm` | `-0.11257 -0.028 0` | `1 0 0 1` | `wrist_flex` | `0 0 1` | -1.658063 .. 1.658063 |
| `wrist` | `-0.1349 0.0052 0` | `1 0 0 -1` | `wrist_roll` | `0 0 1` | -2.7438473 .. 2.84121 |
| `gripper` | `5.55112e-17 -0.0611 0.0181` | `0.0172091 -0.0172091 0.706897 0.706897` | `gripper` | `0 0 1` | -0.174533 .. 1.74533 |
| `moving_jaw_so101_v1` | `0.0202 0.0188 -0.0234` | `1 1 0 0` | (`gripper`가 구동하는 턱) | | |

`site gripperframe`은 `gripper` 안에서 `pos="0.012 -0.000218 -0.098127"`이다. 고정 바디 회전을 적용하고
나면 `shoulder_lift`, `elbow_flex`, `wrist_flex`는 `shoulder_pan`이 쓸고 지나가는 수직 평면 안의 연속된
세 피치가 된다 — 섹션 2.8의 평면 3R이다. IK가 쓰는 정확한 링크 길이는 Rust 소스에 손으로 옮겨 적지 않고
**프로비넌스 오라클이 파싱된 `SceneDesc`에서 유도한다**: 손으로 옮긴 상수는 아무도 검증할 수 없는
숫자다.

데모 장면은 같은 평평한 파일 안에 다음을 더한다: free joint 큐브 (상류 `scene_box.xml`은
`pos="0.5 0 0.03"`에 `size="0.02 0.02 0.03"`, `condim="3" friction="1 .03 .003"`를 쓴다), 바닥판과 얇은
박스 벽 넷으로 만든 바구니, 고정 오버헤드 `<camera>` 하나, `<light>` 하나, 그리고 패스 트레이스 경로가
발광체를 필요로 하면 이름이 `_light`로 끝나는 geom 하나 (`crates/es-render/src/scene.rs:30`이 이름
접미사로 발광을 판단한다 — `SceneDesc`에 발광 머티리얼 필드가 없기 때문).

## 5. 스크립트 전문가 (V1)

### 5.1 배치

`crates/es-env/src/expert.rs`. `es-env`는 레이어 9다: 이미 스텝 루프를 소유하고, 이미 `ModelInfo`를
들고 있으며 `StateView`를 읽을 수 있고, 둘 다 필요로 하는 `es-data` (10)와 `es` CLI보다 아래에 있다.
`es-math` (0)는 장면을 모르고, `es-actuator` (3)는 전달 모델이다. 구체 구조체 `ScriptedExpert` —
**새 트레이트 없음** (INV-17). `Collector::run`이 이미 받는 intervener 훅을 통해 소비되며, CLI는 현재
그것을 `|_, _, _| None`으로 넘긴다 (`crates/es/src/cmd/loop.rs:152-153`). 따라서 시연은 이미 존재하고
이미 테스트된 기계장치로 `action_source = Human` 및 `intervention` 컬럼과 함께 기록된다
(`crates/es-data/tests/loop_learning.rs:569`).

### 5.2 역기구학

`so101_ik(links: &Links, target: Vec3, approach_pitch: f64) -> Option<[f64; 4]>`:

1. `shoulder_pan = atan2(target.y, target.x)`, 관절 범위를 벗어나면 거부.
2. pan이 쓸고 지나가는 평면에서 접근 방향을 따라 손목 오프셋을 뺀 3R 손목 중심을 구하고, 코사인
   법칙으로 2R 삼각형을 풀어 `shoulder_lift`와 `elbow_flex`를 얻는다 (elbow-up 분기);
   `wrist_flex = approach_pitch - shoulder_lift - elbow_flex`.
3. 도달 불가이거나 어떤 각도가 범위를 벗어나면 근사가 아니라 `None`. 웨이포인트에서의 `None`은
   에피소드를 실패한 시연으로 끝내며, 조용히 클램프되지 않는다 (§17.2의 규칙을 기구학에 적용).

`wrist_roll`은 0으로 고정하고 `gripper`는 상태 기계가 명령하므로, 팔은 실제 그대로인 4-DoF 위치
결정기로 다룬다. 삼각함수는 `es_math::approx`를 쓴다 (§3.4는 골든이 의존하는 경로에서 호스트 `libm`을
금지한다).

### 5.3 웨이포인트 상태 기계

`StateView`에서 읽는 것: 큐브 바디의 `xpos`/`xquat` 행과 `gripper` 바디의 행, 둘 다 `ModelInfo.body`
경유. 여섯 상태이며 각각 목표 포즈 + 그리퍼 명령 + 탈출 조건을 가진다: `Approach` (큐브 위) ->
`Descend` -> `Close` (고정 틱 수만큼 유지) -> `Lift` -> `Transport` (바구니 위) -> `Release`. 전이는
정수 틱 수와 위치 오차 임계값으로 하고, 에피소드는 전문가의 의견이 아니라 Task IR 자신의 `Terminate`
노드로 끝난다.

여기서 벽시계나 전역 RNG를 읽는 것은 없다. 에피소드별 변화는 Task IR의 큐브 포즈 랜덤화(섹션 2.7)에서
오므로, 같은 `(task_hash, seed, episode)`는 같은 시연을 만든다.

### 5.4 성공

`큐브가 N 제어 스텝 연속으로 바구니 부피 안에 있음`, Task IR의 `Success` 종류 `Terminate` 노드로
(`crates/es-ir-types/src/expr.rs:105`). 큐브의 free joint `qpos`가 곧 월드 포즈이고, `es-env`의 태스크
플랜은 `Source::Qpos` 리프를 로워링하므로 (`crates/es-env/src/plan.rs:19-28`), 술어는 새 노드 타입 없는
평범한 `Qpos`/`Qvel` 식이다.

"N 스텝 연속"만은 콘이 셀 수 없다: Task IR-D는 상태 없는 데이터플로 DAG다. V0는 싸게 해결한다 —
바구니가 충분히 깊고 속도 경계가 충분히 빡빡해서 큐브가 안정된 뒤에만 술어가 참이 되게 하고 `N = 1`로
둔다. 진짜 안정 카운터가 필요하다고 밝혀지면 그것은 새 IR-D 노드가 아니라 IR-C (§6, 제어)에 속한다:
미해결 질문 3.

**V0가 측정한 것, 그리고 술어가 "AABB 그리고 `|v|`"보다 좁아진 이유.** *기존* 콘의 세 가지 한계이며,
셋 다 `es-ir` 변경이 아니다:

1. **콘 리프는 스칼라 하나, 그 관절의 첫 인덱스다.** `Ctx::joint_leaf`
   (`crates/es-env/src/plan.rs:181-208`)는 `joints.first()`를 취해 `qpos[range.start]`를 바인딩한다.
   큐브의 free joint에서는 그것이 `x` 하나뿐이고, `y`와 `z`는 `TaskNode`에서 아예 주소 지정이 안 된다 —
   `qpos[i]`를 인덱스로 읽는 노드가 없기 때문이다 (인덱스를 쓰는 것은 `Randomization` / `ResetState`의
   타깃 *문자열*뿐, `randomize.rs:141-153`). 그래서 V0가 작성한 술어는 **바구니의 x 구간 + 정지 경계**다.
   장면은 그 구간이 판별력을 갖도록 배치했다: 큐브는 x ≈ 0.24에서 시작하고 바구니 내부는
   x ∈ [0.050, 0.170]이며, 그 x에서 도달 가능한 y를 바구니의 y 폭(±0.105)이 덮는다.
2. **상수 리프도 절댓값도 없다.** `TaskNode`에 `Const`가 없고, `Arith::Mul`의 오른쪽 피연산자는 §5.4
   단위 대수에서 무차원이다 (`task.rs`의 `inputs()`). 따라서 `x - c`도 `v * v`도 표현할 수 없다.
   유일하게 쓸 수 있는 상수는 `Compare { rhs: Some(c) }`가 접어 넣는 것뿐이므로 `|v| < b`는 `Compare`
   둘과 `And` 하나로 쓰고, 성형 보상은 큐브 x의 `Normalize`로 쓴다 (보상 항은 무차원이거나 정규화되어야
   한다, `TYPE-011`).
3. **`Normalize`와 `Logic`은 콘 로워링이 아직 다루지 않는 `TaskNode` 변형이다.** `Ctx::lower`
   (`plan.rs:125-179`)는 `GetJointState`, `GetSensor`, `GetTime`, `Arith`, `Compare`, `Clamp`만 다루고
   나머지는 이름을 밝힌 `EnvError::Unsupported`다. 거기에 `Normalize`와 `Logic`을 더하는 것은 IR을 건드리지
   않는 **`es-env` (레이어 9)의 V1 작업**이다: `Expr`에는 이미 `Logic`이 있고 정규화는 아핀 `Arith`다.
   V0의 문서는 로워링이 아니라 노드 집합을 기준으로 작성된다.

3축 AABB에는 셋 중 하나가 필요하다: 콘 안의 `GetBodyPose` 리프, free joint의 7폭 `qpos`에 대한 `Slice`
리프, 또는 `jointpos` 계열 센서 셋. 셋 다 V0 범위를 넘는 IR 또는 `es-env` 작업이다 — 미해결 질문 3을
여기까지 넓힌다.

**관측 쪽에도 같은 모양의 간극.** `capture` (`crates/es-eval/src/runner.rs:411-420`)는 관측 입력 이름을
**관절**로 키가 잡힌 `ModelInfo.qpos`나 `ModelInfo.sensor`에 대해 해석한다. `qvel` 경로도 바디 경로도
없다. 교차 IR 검사 (`XIR-002`)는 Task IR 채널과 Observation IR `StateInput`이 *같은* id를 부르기를
요구하고, `DEP-031`은 Safety Plane 엔벨로프를 그 채널의 `dof`와 비교한다. 그래서 6관절 로봇은 로봇 바디가
부르는 `dof = 6` 채널 하나가 되는데, 바로 그것을 `capture`가 해석하지 못한다.
`crates/es-eval/tests/evaluation.rs:427-440`은 Task IR에는 바디를, Observation IR에는 관절을 적어 이를
피해 가지만 그 조합은 `cross::check`를 통과하지 못한다. 수정은 V1의 몫이고 (`capture`를 관절 *집합*과
`qvel`까지 넓히거나, 장면에 `jointpos` / `jointvel` 센서를 내보내거나), V0는 정직한 6폭 상태를 선언하고
간극을 여기 기록한다.

## 6. 학습과 훈련 (V2)

### 6.1 그래프

섹션 2.5의 6노드 ACT 형태 `LearningGraph`: `VisionEncoder { ResNet18 }` -> `StateEncoder` -> `Fusion` ->
`TemporalEncoder { Transformer }` -> `PolicyHead { Regression }` -> `ActionChunker`, 그리고 `Normalizer`
노드들. `lower_to_torch`는 이를 생성 Python이 아니라 IR 안에서 로워링한다
(`crates/es-policy/src/lower/torch.rs:769-807` — `lower_act`가 하지 않는 바로 그것).

이것은 LeRobot의 ACT가 아니다. VAE도 DETR 디코더도 없다. §8.3의 `TemporalEncoder { Transformer }`가
너비만 싣기 때문이며 (`lower_act`가 존재하는 이유 자체다, `crates/es-policy/src/lerobot.rs:9-14`),
데모에서 이것을 "ACT"라고 부르면 작은 거짓말이 된다. 설계 노트와 패킷은 이를 **ACT 형태의 Learning IR
정책**이라고 부르며, 데모 문구도 그래야 한다.

### 6.2 로워, 학습, 팩

세 단계, Rust/Python 선은 정확히 §2.3이 두는 자리에:

```
es policy lower   --policy untrained.esb --out build/
es dataset bake   --policy untrained.esb --out baked/ --frames tiles/ ds/     # V2b
<py> python/es/train_act.py --module build/ --baked baked/ --out model.safetensors
es policy pack    --policy untrained.esb --weights model.safetensors --out trained.esb
```

(`bake` 줄은 V2b의 것이다. V2의 세 단계로 왜 충분하지 않았는지는 섹션 7.9.)

- `lower`는 `TorchModule.source`를 그대로 쓰고, 더해서 `contract.json` — 로워링이 선언하는
  `weight_keys`와 `weight_shapes` (`crates/es-policy/src/lower/torch.rs:297`), 그리고 `lowering_hash`.
- `bake`는 기록된 모든 프레임을 번들 자신의 Observation IR로 통과시킨다 (V2b, 섹션 7.9).
- `train_act.py`는 그 소스를 `exec`하고 `EsPolicy()`를 만들고 **구운** 관측 세트를 읽고
  옵티마이저를 돌린 뒤 **`contract.json`의 키로만** `safetensors`를 쓴다. IR에 대해 아무것도
  모르며 레이어를 정의할 권한이 없고, V2b 이후에는 Observation IR 노드도 구현하지 않는다.
- `pack`은 safetensors를 읽어 모든 키와 모양을 컨트랙트와 대조하고, 그 외 무엇이든 거부하며, 새
  `weights.safetensors`와 재계산된 매니페스트로 번들을 다시 쓴다 (`crates/es-compile/src/bundle.rs:467`,
  `:513`). `safetensors`만 쓰고 어디에도 pickle 경로를 추가하지 않으며 (INV-16), `WeightsSource`에
  변형을 더하지 않는다 (`crates/es-policy/src/runtime.rs:27-32`).

그러면 `es eval run --policy trained.esb`가 **eval 경로를 전혀 바꾸지 않고** 동작한다:
`TorchRuntime::load`가 같은 `LearningGraph`를 다시 로워링해 같은 모듈을 얻는다 (`torch_runtime.rs:340`).

### 6.3 학습 오라클

학습은 §1.4가 "레퍼런스와 일치"로 판정할 수 없는 플랜 V의 유일한 부분이다. 대신 실행 가능한 네 가지
사실로 판정하며, 그 중 어느 것도 "손실 곡선이 좋아 보였다"가 아니다:

1. **모듈이 IR의 것이다.** `train_act.py`는 레이어를 만들지 않는다; 오라클이 로드된 `es_policy.py`와
   `lower_to_torch(bundle.learning).source`를 대조해 동일성을 요구한다.
2. **가중치가 컨트랙트에 맞는다.** `es policy pack`은 누락 키, 추가 키, 틀린 모양을 거부하고, 오라클은
   세 거부를 모두 단언한다.
3. **무언가를 배운다.** 고정된 작은 데이터셋과 고정 시드에서 최종 학습 손실이 초기 손실의 고정 비율
   미만. 골든이 아니라 임계값이다 — CPU PyTorch는 비트 단위로 이식 가능하지 않다.
4. **왕복한다.** 패킹된 번들이 `TorchRuntime`에서 로드되고 고정 관측에 대해 선언된 모양의 유한한 액션
   청크를 낸다.

`torch`가 없으면 넷 다 사유와 함께 `SKIP`한다.

### 6.4 크기, 그리고 CPU 전용 리스크

섹션 2.9: 서버의 두 venv 모두 `+cpu` 빌드다. 따라서 데모의 이미지는 작고 (V0가 `ImageSpec`을 선언한다;
96x96 또는 128x128, 카메라 하나 — 어차피 `MultiViewPack`은 거부된다, 섹션 2.2), 청크 호라이즌도 짧다.
`_backbone("resnet18", 512)` (`crates/es-policy/src/lower/torch.rs:925`)가 ImageNet 사전학습 가중치를
원하는지, 따라서 학습 시 네트워크가 필요한지는 **미검증**이며 미해결 질문 6이다. 어디에도 학습 시간을
적지 않는다 (§12.4).

## 7. 루프 안의 렌더링 (V0b)

### 7.1 애니메이션

`TriScene::from_scene`은 정적 바디 포즈를 굽는다 (섹션 2.2). V0b는 그 옆에 함수 하나를 더한다:

```rust
pub fn from_scene_with_poses(scene: &SceneDesc, world: &BTreeMap<StableId, Pose>) -> Result<TriScene, RenderError>
```

— `from_scene`과 동일하되 바디의 월드 포즈가 있으면 `world`에서 가져온다. 호출자가 `ModelInfo.body`와
`StateView::xpos`/`xquat`로 그 맵을 만든다. **`es-render`는 의존성을 얻지 않는다**: `StateView`가 아니라
포즈 맵을 받으므로 레이어 5는 여전히 레이어 3을 모른다.

숨기지 않고 명시하는 한계: 렌더 프레임마다 장면 전체를 다시 테셀레이트하고 다시 업로드한다.
`es-render`는 가속 구조 없이 평평한 삼각형 배열을 훑고 수백 삼각형 한계를 문서화한다
(`crates/es-render/src/lib.rs:11-14`). 데모 장면은 프리미티브 팔, 큐브, 판 다섯 개짜리 바구니다. 그것이
더는 참이 아니게 되면 해법은 더 큰 CPU 루프가 아니라 셰이더의 바디별 변환 버퍼다.

### 7.2 프레임 디스크 기록

`Tile`은 이미 자기 바이트를 안다 (`crates/es-render/src/atlas.rs:147-153`). 그 레이아웃이 이미 골든
포맷이다 (`tests/golden/render/*.bin` + `.json` 사이드카). V0b는 정확히 그것을 프레임당 한 쌍씩 쓴다:
`frames/<cell>/<NNNNNN>.bin` 그리고 `frames/<cell>/layout.json` 하나.

**PNG는 없다.** 워크스페이스에 PNG나 이미지 인코더가 없고 `[workspace.dependencies]`에도 없다. 추가해도
얻는 게 없다: 유일한 소비자는 numpy인 V4의 모자이크다. 원시 프레임이 바이트 동일한 산출물이고 PNG는 그
두 번째 인코딩일 뿐이다. 사람이 이미지 뷰어로 열어 보고 싶다면 미해결 질문 7.

### 7.3 이미지 관측 포트

좁은 수정 둘:

- `crates/es-env`: 기본 꺼짐인 선택적 `render` 카고 피처 (`es-ros2/zenoh` 선례를 따름)로 `es-render` (5)와
  `es-gpu` (2)를 추가한다 — 둘 다 레이어 9 아래이므로 `cargo xtask layering`을 만족한다. `Env`가
  `Option<EnvRenderer>`를 얻는다 — 트레이트가 아니라 구체 구조체 (INV-17). 피처가 꺼져 있으면 `es-env`는
  지금과 정확히 똑같이 컴파일되고 동작하며, `es-runtime-embedded`는 Vulkan을 보지 못한다.
- `crates/es-eval/src/runner.rs:421-428`이 **렌더러가 있을 때에만** 이미지 입력에 대한
  `EvalError::Plan` 반환을 멈추고 타일을 `ImageInput` 포트의 텐서로 넣는다. 렌더러가 없으면 같은
  메시지로 계속 거부한다. 무엇도 0으로 채워지지 않는다.

INV-14는 건드리지 않는다: 렌더러는 선언된 크기로 정확히 타일을 만들고 재샘플링을 거부한다
(`crates/es-render/src/renderer.rs:292-302`); `Resize`/`Crop`은 내부 파라미터를 변환하는 Observation IR
노드로 남는다. `es-render` 자신의 `ImageSpec` 서브셋 (`crates/es-render/src/view.rs:81-95`)은 플랜 시점에
Observation IR이 선언한 `ImageSpec`과 대조되고, 불일치는 리스케일이 아니라 에러다 — §26.1이 말하는
"검증되지 않은 것은 실행되지 않는다"이고 W1c가 카메라에 적용한 바로 그 규칙이다.

### 7.4 구현 결과 (V0b), 그리고 위 계획이 틀렸던 두 가지

둘 다 취향이 아니라 구조적 제약이다:

- **`Env`는 렌더러를 들고 있지 않는다.** `EnvRenderer<'gpu>`는 `Gpu`를 빌리므로, `Env`의 필드로 두면
  `Env<B>`에 수명이 붙는다 — `crates/es-data/src/collect.rs:297`과 `crates/es-eval/src/runner.rs:143`이
  모두 이름으로 쓰는 타입이고, 둘 다 V0b가 고칠 수 있는 파일이 아니다. 그래서 렌더러는 호출자의 것이고,
  `EnvRenderer::frame(&ModelInfo, &StateView, env)`이 호출자가 이미 들고 있는 상태를 받는다 (수용 기준의
  시그니처가 이미 그 모양이었다). `Env`와 `domains.rs`는 건드리지 않았고, 그래서 "피처가 꺼지면 바이트
  단위로 동일한 동작"은 테스트할 대상이 아니라 자명한 참이다.
- **`es-eval`에는 `render` 피처가 없다.** 대신 프레임 소스 —
  `es_eval::runner::FrameSource = dyn FnMut(&ModelInfo, &StateView) -> Result<Vec<u8>, String>` — 를
  `Evaluation::run_with_frames`로 받고, `Evaluation::run`은 `None`을 넘긴다. 레이어 10은 이미지를
  *거부하기 위해서조차* Vulkan을 링크하지 않고, 거부 경로는 어느 머신에서나 테스트된다. 호출자가
  `EnvRenderer::frame`을 그 클로저에 연결하며, CLI가 그렇게 하는 것은 V3이다.

**V1/V2를 위한 발견.** V0의 `observation.toml`은 `ImageInput`의 *텐서*를 `F32` `[3, 96, 96]` (CHW)로
선언하면서 `ImageSpec`의 `dtype`은 `U8`이라고 말한다 — 즉 렌더러의 `Rgb8` `[96, 96, 3]` 타일은 그 버퍼에
맞지 않고, `capture`는 양쪽 크기를 적어 거부한다. `ImageSpec` 자체는 렌더러와 정확히 일치한다 (V0의 파일을
그대로 읽어 `the_declared_image_spec_is_checked_not_coerced`가 확인한다). 수정은 픽스처를 다음에 고치는
패킷의 몫이다: `ImageInput`을 `U8 [96, 96, 3]`으로 선언하고 뒤에 `ObservationNode::Dequantize`를 두면 된다
— `Op::Dequantize` (`crates/es-compile/src/plan.rs:73`)가 정확히 "HWC u8 → CHW f32 /255"이고 이미 있다.
V0b는 V0의 픽스처를 고치지 않는다.

### 7.5 구현 결과 (V1), 그리고 V0의 씬과 문서가 틀렸던 네 가지

아래는 모두 오라클 서버에서 `tests/fixtures/mjcf/so101_pick_place.xml`에 대해 측정한 값이며,
`es_env::expert::demo_cfg`의 모든 숫자는 이 측정에서 나왔다. 오라클은
`expert_solves_the_pinned_seeds`(`crates/es/tests/cli.rs`)다: 고정된 시드 8개를
`es loop collect --expert`로 돌리고, 큐브의 최종 `x`, `y`, `z`를 기록된 데이터셋에서 직접 읽는다 —
Task IR의 판정식은 `x` 하나만 보기 때문에(5.4절) 거기서 얻은 성공률은 expert가 아니라 판정식에 대한
성공률이 되기 때문이다.

**1. bin이 로봇 안에 있었고, 팔이 닿지도 않았다.** V0의 위치(+X 축 위 `x = 0.11` 중심)에서는
`bin_wall_nx`가 어깨 자신의 충돌 박스 `shoulder_holder_col`을 11 mm 파고들었다. MuJoCo는 정지
자세에서 접촉을 보고했고 `shoulder_pan`은 아예 돌지 못했다. 그뿐 아니라 *bin 내부 위쪽의 어떤 점도 팔의
작업공간 안에 없었다*: 0.16 m 툴과 SO-101의 `wrist_flex` 범위에서는 `x < 0.14`, `z > 0.04`에
elbow-up 해가 존재하지 않는다. bin은 이제 팔 옆에 놓인다(내부 `x ∈ [0.09, 0.19]`,
`y ∈ [-0.15, -0.05]`). 닿고, 어깨와 겹치지 않으며, 벽 높이에서도 오버헤드 카메라 프러스텀 안에 완전히
들어온다. 마지막 조건은 렌더 골든이 의존한다: 벽 윗면이 프러스텀 경계에 걸치게 두었더니 GPU 프레임이 CPU
레퍼런스와 27,648 바이트 중 3 바이트 달라졌다.

**2. 큐브는 30 mm가 아니라 25 mm다.** 집게 한쪽은 고정이다. 툴 사이트가 집게 간격의 중앙이 되는 것은 특정
개방각에서뿐이고, 베이스가 회전한 상태로 접근하면 큐브는 대각선을 내민다. 30 mm 큐브는 뽑기 범위의 가까운
모서리에서 약 2 mm 여유만 남겼고, 내려가는 동안 그 여유를 잃고 큐브를 잡는 대신 밀어냈다. 같은 이유로
`pos_tol`도 0.035 rad에서 0.01(툴 기준 약 2 mm)로 줄였다.

**3. 성공 판정식이 큐브가 아직 집게 안에 있을 때 발화한다.** "큐브 `x`가 bin 구간 안, 정지 상태"는 bin에
놓인 큐브만큼이나 bin 위를 *운반 중인* 큐브에도 참이다. V1은 정확히 그것을 측정했다: 8/8 에피소드가 큐브를
10 cm 공중에 든 채 `Success`로 끝났다. 콘이 읽을 수 있는 스칼라를 하나 더 쓴다 — 그리퍼 자신의 관절 —
그래서 판정식은 이제 "bin의 x 구간 안, 정지, **그리고 쥐고 있지 않음**"이다. 임계값이 0.6 rad인 이유는
25 mm 큐브를 문 집게가 아무리 닫으라고 해도 0.30 부근에서 멈추기 때문이다. 그래서 시연은 0.4가 아니라
0.9까지 연다. expert에는 `Lower` 단계도 생겼다: 운반 높이에서 놓으면 큐브는 bin만큼이나 자주 벽에
떨어지고, 판정식이 `z`를 못 보기 때문에 에피소드는 큐브가 아직 낙하 중일 때 끝난다.

**4. 스텝 명령은 시연이 아니다.** 5 Hz 추론에 `execute_chunk = 10`, `TemporalEnsemble`: expert의 틱당
위치 목표는 청크로 액추에이터에 도달하고, Safety Plane은 각 청크를 엔벨로프가 허용하는 움직임으로 바꾼다.
IK 해로 곧장 점프하는 스크립트는 거의 매 틱 클램프된다 — 측정값: 40스텝 중 39, `Velocity`와
`Acceleration` — 그리고 V0의 `envelope_violation_rate { max_frac = 0.05 }`는 모든 시연에서 1초쯤
뒤에 폴백을 래치해 팔을 끝까지 얼려버렸다. 기록된 변경은 두 가지다: expert가 **램프 청크**를 내보내고
(`step_max`, `accel_max`는 기록에 쓰이는 Deployment IR에서 읽고, 측정 관절에 대한 anti-windup 포함),
데모의 `max_frac`을 0.9로 넓혔다. 엔벨로프를 넓히는 것은 허용된 수단이고 플레인을 끄는 것은 아니다
(`INV-12`). 클램프는 여전히 모두 계수되고 `action_source = Clamped`로 기록된다 — 그것이 V3의 재료다.

**측정 결과.** 고정 시드 8/8이 `Success`로 끝나고 8/8이 큐브를 bin의 3차원 내부에 넣는다. 900스텝(18초)
예산 중 약 350 제어 스텝(7초)이 걸린다. IK 자체는 120개 목표에 대해 MuJoCo 자신의 순기구학과 3.3e-8 m
이내로 일치하고, 같은 시드 두 번은 바이트 단위로 같은 `ctrl` 행을 만든다.

**데이터셋 오라클은 답했고, 답은 거부였다.** `[dataset]` extra가 설치된 `lerobot` 0.6.1은
`codebase_version: "v2.1"`을 아예 읽지 않는다. `BackwardCompatibilityError`를 던지고 자신의
v2.1 → v3.0 변환기를 안내한다. 이는 미해결 질문 2를 v3 writer 쪽으로 결론짓고,
`docs/api-notes/lerobot-dataset.ko.md`가 이를 기록한다. writer는 여기서 의도적으로 건드리지 않는다.
그것을 바꾸는 패킷은 v3.0을 구현하는 패킷이기 때문이다.

### 7.6 V2 실제 구현, 그리고 위 계획이 몰랐던 여섯 가지

`es policy lower`와 `es policy pack` (`crates/es/src/cmd/policy.rs`)이 6.2절의 Rust 쪽 양 끝이고,
`python/es/train_act.py`가 그 사이의 옵티마이저다. `es_policy::lower::Contract`가 모듈 소스를
제외하면 스펙 2.3의 분할선을 넘는 것의 전부다. 아래는 전부 발견 사항이며, 다섯 중 넷은 이
구간을 처음 실제로 돌려보지 않고서는 알 수 없던 것들이다.

**1. 로워링이 실행 불가능한 모듈을 만들고 있었다.** `VisionEncoder{ResNet18}`은
`self.n0(inputs["rgb_overhead"])`로 내려가고, IR 포트는 이미지 한 장 — `[3, 96, 96]`, 배치 축
없음. 스펙 5.2가 추론 도메인에 자체 배치 크기를 주기 때문이다. 그런데 torchvision 백본은
바닥까지 `nn.BatchNorm2d`이고, 이것은 3차원 입력을 그냥 거부한다: *"expected 4D input (got 3D
input)"*. 어떤 테스트도 이를 잡지 못했는데, `torch_equivalence.rs`가 PyTorch에 태우는 유일한
그래프가 state-only MLP(`a_state_only_graph_lowers_to_torch_alone`)이고,
`es_ir::learning::testing::act_like`는 `[3, 224, 224]`을 선언한다 — 같은 모양, 같은 실패.
비전 인코더를 가진 **모든** `lower_to_torch` 소비자가 깨져 있었으므로, 수정은 이 패킷 안의
우회가 아니라 로워링의 한 줄이다: 이미지 한 장짜리 배치로 돌린다 —
`self.n0(x.unsqueeze(0)).squeeze(0)`. 바로 두 갈래 아래의 토큰 없는 `TemporalEncoder` 가지가
이미 하고 있는 것과 같은 거래다. 이로써 V2의 선언된 context에
`crates/es-policy/src/lower/torch.rs`가 하나 늘었다. 비전 인코더가 있는 그래프의
`lowering_hash`는 이동하며, 이를 고정하는 골든은 없다.

**2. 수집기에는 frame sink가 있었지만 호출자가 없었다. 그래서 V2가 연결했다.** V1은
`CollectSpec`의 `FrameSink`를 `es-data`에 구현하고 거기서 테스트했지만, CLI에 그것을 넘겨주는
곳이 없었다: `crates/es/src/cmd/loop.rs`가 `None`을 넘겼고, 그래서 모든 `es loop collect` 실행이
*"camera `rgb_overhead`: `info.json` declares a video feature ... but no mp4 was written"*를
출력했으며 데이터셋에는 픽셀이 전혀 없었다. 이제 `es loop collect --frames <dir>`가 collect 호출
안에서 `Gpu`와 `EnvRenderer`를 만든다 — 바로 그 borrow가 `Env`가 렌더러를 들고 있지 않은
이유다(7.4절) — Task IR 자신의 이미지 채널과 `ImageSpec`으로 설정하고, 컨트롤 스텝마다 raw tile
하나를 `<dir>/<NNNNNN>.bin` + `.json`으로 쓴다. 렌더 골든과 `es video mosaic`이 이미 쓰는 형식이다.
`es`의 새 `render` 피처 뒤에 있고 기본은 꺼져 있으므로, 평소의 CLI는 여전히 Vulkan을 전혀 링크하지
않는다(스펙 4.2). 피처 없는 빌드는 구멍 뚫린 데이터셋을 쓰는 대신 `--frames`를 거부한다. 오라클
서버에서 측정: 352 스텝 시연 하나가 1.8초에 렌더되고, "no mp4 was written" 경고는 사라졌다.

**3. 학습이 Observation IR 노드를 정확히 하나 재구현한다. 그리고 그것은 부채다.** tile은 HWC
`u8`이고 Learning IR 입력은 CHW `f32`다. 추론에서는 Observation IR의 컴파일된 plan이
`Op::Dequantize`(`es_compile::plan::Op::Dequantize`)로 그 변환을 한다. Python 쪽에는 plan 러너가
없으므로 `train_act.py`의 `dequantize`가 **그 노드다**. 이 저장소에서 IR 노드가 두 번째 구현을
갖는 유일한 곳이다. 데모의 Observation IR이 정확히 `ImageInput -> Dequantize -> sink`이기 때문에만
성립한다: 이미지 포트와 Learning IR 입력 사이에 노드가 하나라도 더 생기는 순간 학습과 추론이
조용히 어긋난다. 진짜 해법은 데이터셋 위에 observation plan을 굽는 `es` 단계이고, 그것은 한 줄이
아니라 패킷이다.

**4. `es eval run`은 여전히 이미지를 넣을 수 없다. 그래서 V2는 성공률을 보고하지 않는다.**
`Evaluation::run`은 `frames: None`을 넘기고 `capture`는 이미지 입력을 이름을 대며 거부한다
(`crates/es-eval/src/runner.rs:475-481`). 의도적이다 — "스펙 10.1 표의 틀린 숫자는 표가 없는 것보다
나쁘다". `Evaluation::run_with_frames`와 이 패킷이 `cmd/loop.rs`에 넣은 `renderer_cfg`가 둘이서
V3가 그것을 닫는 데 필요한 전부지만, 닫는 것은 V3의 몫이고 아직 존재하지 않는 데모
`evaluation.toml`도 마찬가지다. 학습된 번들에 대해 V2가 하는 가장 강한 정직한 주장은 패킷 자신의
것이다: `es eval run --policy trained.esb`가 **`TorchRuntime::load`를 통과한다**. `lower_act`
번들이라면 실패할 단언이다(2.5절).

**5. `es loop collect --episodes N`은 0번 에피소드만 푼다.** 오라클 서버에서 측정:
`--episodes 50 --seed 1`은 `success 1, timeout 49`로 끝나는 반면, 같은 시드를 한 에피소드씩
돌리면(`--episodes 1 --seed s`, `s = 1..50`) 49 성공 1 타임아웃이다. 스크립트 전문가 또는
수집기가 에피소드 경계를 넘어 들고 가는 무언가가 리셋되지 않는다 — `Collector::run`의 루프,
`ScriptedExpert`의 웨이포인트 상태, 또는 팔의 자세. 이것은 V1 코드이고 V2는 건드리지 않는다.
V2의 데이터셋은 단일 에피소드 실행 50개를 `es loop distill`로 병합한 것이며, 이 결함은 V1
후속 패킷이 필요하다. 아울러 V1의 오라클(`expert_solves_the_pinned_seeds`, 시드 8개를 각각 한
에피소드씩)은 이것을 볼 수 없었다는 뜻이기도 하다.

**6. 체크포인트가 움직이는 것은 `learning_hash`가 아니라 `policy_hash`다.** 6.2절과 패킷 모두
`policy_hash`라고 했고 그게 맞지만, 못 박아 둘 가치가 있다: `learning_hash`는 그래프를
덮고(`canon_learning`) 움직이지 않으며, `policy_hash`는 `WeightsRef`를 그 안에 해싱하므로
(`LearningGraph::policy_hash`) 움직인다.
`pack_recomputes_the_manifest_rather_than_patching_it`이 네 슬롯을 모두 단언한다: `policy`는
움직이고 `learning`, `task`, `observation`은 움직이지 않는다.

**미결 질문 6은 답이 나왔고, 더 작은 것이 하나 열렸다.** `_backbone`은 `weights=` 인자 없는
`getattr(torchvision.models, name)()`이므로, 학습은 처음부터(from scratch)이고 어느 시점에도
네트워크가 필요 없다. Learning IR 노드는 `pretrained = true`를 들고 있고, 로워링은 그것을
**무시한다** — IR이 선언한 것과 실제로 도는 것 사이의 조용한 불일치다. 오늘은 무해하지만
(우리가 원하는 답이 "from scratch"다) 존중하거나 거부해야지 버려서는 안 된다.

**측정된 것, 오라클 서버에서 (RTX 4090, `~/venvs/es-lerobot-cuda`, torch 2.11.0+cu129).**
시연 50개, 그중 50개가 `Success`, 프레임 17,697개: LeRobot v2.1 parquet 3.4 MB와 렌더된 tile
485 MB. 시드 101-105의 held-out 5개 에피소드, 5개 중 4개 `Success`. 학습은 `--batch 8`로 20,000
optimizer step을 돌았고(로워링된 모듈은 single-sample이다 — `reshape(horizon, action_dim)`과
`[:execute_chunk]`에 배치 축이 없다 — 그래서 배치는 forward 한 번씩 누적된다), 액션 청크 L1 loss는
step 1 / 1,000 / 5,000 / 20,000에서 0.727 -> 0.0507 -> 0.0309 -> 0.0171로 떨어졌다. 모든 벽시계
시간은 관측이지 주장이 아니며, 어떤 처리량 수치도 어디에도 적지 않는다(스펙 12.4). 20k
체크포인트는 61 MB이고 커밋하지 **않는다**. 서버의 `~/artifacts/plan-v/` 아래에 있고 blake3은
`docs/packets/M5/V2-act-training.md`에 기록한다.

**패킷 acceptance로부터의 이탈, 전부 의도적이다.** `train_act.py`는 패킷에 없던 플래그 세 개를
받는다 — `--checkpoint-at`(실행 길이도 상한으로 잡으므로 "20,000 step"이 정확하다),
`--loss-curve`, 그리고 수집기가 이제 쓰는 tile을 읽는 `--frames` — 그리고 JSON 한 줄에 패킷이 못
박은 셋 외에 다섯 개의 키를 더 싣는다.
`the_loss_falls`와 `the_packed_bundle_round_trips`는 한 테스트다. 사실 4가 사실 3의
체크포인트를 필요로 하고, 이름 둘을 지키자고 두 번 학습하는 것은 낭비이기 때문이다.
`eval_run_accepts_the_trained_bundle`은 `crates/es/tests/cli.rs`의
`policy_pack_output_is_accepted_by_eval_run`으로 옮겼다. Evaluation IR 픽스처가 거기 있고
`tests/fixtures/visible-learning/`에는 `evaluation.toml`이 없기 때문이다(그것은 V3의 문서다).
round-trip은 패킷이 요구한 것보다 강해졌다: held-out 관측 하나에 대해 `TorchRuntime::infer`를
*직접* PyTorch forward와 스펙 8.9의 tier-4 fp32 허용오차로 비교한다. safetensors writer,
`pack`의 검증, `nodes.N` -> `nN` 이름 변환, 와이어 프로토콜을 한 숫자로 덮는다.

### 7.7 구현 결과 (V1b): LeRobot v3.0 내보내기, 그리고 포맷이 실제로 요구한 것

V1의 데이터셋 오라클은 거부로 답했고(7.5절), V1b는 그 수리다 — 라이터 교체가 아니라 변환기.
`es loop collect`는 여전히 `codebase_version: "v2.1"`을 쓴다. V2의 학습 스크립트가 그것을 직접 읽고,
동시에 진행 중인 패킷 밑에서 포맷을 옮기는 것은 비싼 쪽의 정답이었을 것이다. 새 명령은
`es dataset export --lerobot-v3 <root> --out <dir>`, 구현 전부는
`crates/es-data/src/lerobot/v3.rs`, 포맷은 오라클 서버의 `lerobot` 0.6.1에 대해 고정해
`docs/api-notes/lerobot-dataset.md`의 "LeRobot v3.0" 절에 적었다 — 설치된 패키지를 읽고 *또한*
그것에 대해 실행해서, 파일 경로 단위로.

**2026-09-15 측정**, `ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot-cuda/bin/python cargo test
-p es-data --test lerobot_v3 -- --nocapture`: `RAN lerobot_v3_export`. `LeRobotDataset`가 내보낸
것을 열어 `codebase_version 3.0`, 에피소드 2개 / 프레임 7개를 보고했고, 7프레임을 모두 순회했으며,
`observation.state`와 `action`을 모든 프레임에서 우리가 쓴 parquet과 1e-5 이내로 일치하게,
카메라를 `[3, 4, 6]` CHW 텐서로, 픽셀 합을 111.6706(우리 계산 28476/255 = 111.67059)으로 돌려줬다.
옆의 v2.1 오라클은 여전히 거부를 사유로 `SKIP`을 출력한다. 0.6.1이 읽지 않는 포맷의 정직한 상태다.

**v2.1 노트가 말하지 못했던, 포맷이 요구한 네 가지.**

1. **`shape: [1]` 피처는 길이 1 리스트가 아니라 스칼라 컬럼이다.** `DatasetInfo.__post_init__`이
   모든 `shape`를 튜플로 바꾸고, `get_hf_features_from_features`가 `len(shape) == 1`보다 먼저
   `shape == (1,)`로 분기한다. v2.1 라이터는 모든 피처를 3-레벨 LIST로 내보내므로 `reward`,
   `timestamp`와 네 개의 인덱스 컬럼이 나가는 길에 모양을 바꾼다. `[6]` 피처는 여전히 리스트다 —
   선언이 `fixed_size_list<T>[6]`인 자리에 parquet 가변 `list<T>`가 와도 `datasets`가 캐스트하므로
   기존 스키마 빌더를 재사용할 수 있다.
2. **다섯 부기 컬럼은 `features`에 필수다.** 그것이 `Dataset.from_parquet(..., features=...)`을
   구동하므로, 파일에는 있고 `info.json`에는 없는 컬럼은 무해한 덤이 아니다. v2.1이 `미검증`으로
   남긴 질문에 v3.0이 답한다.
3. **`meta/tasks.parquet`은 `pandas`로 읽히고, 태스크 문자열은 그 인덱스여야 한다.**
   `dataset_reader.py:352`가 `self._meta.tasks.iloc[task_idx].name`이고, 행의 `.name`은 *인덱스*
   값이다. parquet 컬럼 두 개로는 부족하다 — 파일은 푸터에 `task`를 `index_columns`로 지목하는
   `pandas` key/value 메타데이터도 담는다. 내보내기가 Python 라이브러리의 내부 직렬화 관행을 쓰는
   유일한 지점이고, api-note가 그 블록을 그대로 인용한다.
4. **이미지는 비디오 코덱이 필요 없고, 써서도 안 된다.** `dtype: "image"`는 인코딩된 이미지 파일을
   데이터 parquet 안에 `struct<bytes, path>`로 저장한다. `dtype: "video"`는 디코더가 필요하고,
   오라클 서버의 `torchcodec`은 설치되어 있지만 로드되지 않아(`libnppicc.so.12` 없음) `pyav`로
   폴백한다. 내보내기는 `image`를 쓰고, 어디서도 인코더가 돌지 않는다 — `~/.local/bin/ffmpeg`도,
   Rust에서도, 오라클의 Python 쪽에서도.

**여전히 PNG 크레이트는 없다.** 7.2절은 유일한 소비자가 numpy라서 이미지 인코더 추가를 거부했다.
`lerobot`이 두 번째 소비자이고 디코드 가능한 파일을 원하므로, `v3.rs`가 ~70줄의 PNG를 담는다:
8비트 트루컬러, 필터 0, *stored* deflate 블록의 zlib 스트림. 원시 프레임 대비 약 0.1%를 쓰고 PIL이
여는 파일을 얻는다. 검사는 `png_round_trips`로, 인코더 자신의 출력을 되파싱한다 — 모든 청크 CRC,
LEN/NLEN 쌍, Adler-32 — 그래서 인코더는 인터프리터가 필요 없는 오라클을 갖는다.

**출처는 `info.json`이 아니라 사이드카다.** 스펙 §19.2에 따라 내보낸 것은 파생 산출물이므로
`meta/es_provenance.json`이 원본의 `content`/`schema`/`split` 해시와 `codebase_version`을 기록한다.
`info.json`의 추가 키가 아닌 이유는 `DatasetInfo.from_dict`가 모르는 키를 경고와 함께 버리고
`to_dict`가 그것을 되쓰지 않기 때문이다 — 거기 넣은 출처는 LeRobot 쪽 재기록을 살아남지 못한다.

**의도적으로 뺀 세 가지.**

- **`meta/stats.json`을 쓰지 않는다.** 없으면 `load_stats`가 `None`을 돌려주고 읽기 경로에서
  필요하지 않다. 그것은 학습의 정규화를 위한 것이다. 파일을 채우려고 통계를 지어내는 것은 빼는
  것보다 나쁘고, 플랜 V에서 `lerobot`을 통해 학습하는 것은 없다.
- **청크/파일 분할 없음.** `data/chunk-000/file-000.parquet` 하나와
  `meta/episodes/chunk-000/file-000.parquet` 하나. `data_files_size_in_mb`는 권고값이고(리더는
  `data/*/*.parquet`을 글롭한다), 내보내기는 현재 데이터셋 하나를 메모리에 올린다 — 그 한계는
  소스에 그대로 표시했다.
- **픽셀이 없는 카메라는 내보내지 않고 버린다.** `es loop collect`는 오늘 픽셀을 쓰지 않으므로
  (7.5절) 픽셀은 `--frames <dir>`에서 온다: `<dir>/<name>/<NNNNNN>.bin`, `EnvRenderer`가 이미
  쓰는 원시 덤프를 `observation.images.<name>`마다 하위 디렉터리 하나로. 그것이 없으면 피처를
  버리고 명령이 그렇게 말한다. 아무도 못 읽을 이미지 피처를 선언하지 않는다.

### 7.8 구현 결과 (V3), 그리고 위 계획이 몰랐던 두 가지

`es eval run --frames <dir>`가 섹션 8의 입력 전부다: 제어 스텝마다 Task IR의 이미지 채널
하나를 렌더하고, `<dir>/<suite>-<NN>/<NNNNNN>.bin`과 셀마다 `layout.json` 하나를 쓰며,
`<out>/events.json`에 프레임마다 `{ frame, tick, source, events }` 레코드 하나를 쓴다 —
`es video mosaic`이 이미 읽는 디렉터리 모양이자 JSON 모양이고, GPU도 물리 백엔드도 없이
`eval_run_output_is_what_es_video_mosaic_reads`가 확인한다. 셀 하나는 한 스위트의 한
**에피소드**다. `Evaluation::run`이 에피소드마다 자신의 `Env`를 주기
때문이며(`BatchDomains::single_env()`), 그래서 16 에피소드가 16개 디렉터리이고 4x4 격자는
에피소드의 격자다.

**1. `capture`가 어느 입력이 이미지인지 추측했고, 그 추측은 틀렸다.** 규칙은 "`qpos`도
`sensor`도 아닌 plan 입력은 모두 이미지"였다. `Evaluation::run`이 `frames: None`을 넘기는
동안에는 그런 입력이 전부 거부되었으므로 보이지 않았고, V3가 프레임 소스를 공급하는 순간
버그가 되었다: 데모의 `StateInput`은 로봇 **body**를 지칭하므로
(`ObsSource::JointState { body, dof: 6 }`, 관절 id도 센서 id도 아니다) 첫 실제 실행은
27,648바이트 카메라 타일을 6원소 `F32` 버퍼에 먹이고 크기 불일치로 멈췄다. 이제
`input_sources`가 **첫 에피소드 이전에 한 번만** 문서에 대해 모든 입력을 해석한다: 입력이
이미지인 이유는 Observation IR이 `ImageInput`이라고 말하기 때문이고,
`JointState { body, dof }` 채널은 앞쪽 `dof`개 관절 위치를 읽는다 — `joint_state::<NJ>`가
Safety Plane에 먹이려고 이미 쓰는 것과 같은 규약이다. 그 외의 것은 표 중간의 뜻밖의 사고가
아니라 실행 시작 시점의 오류다. `docs/design/evaluation-execution.md` 섹션 2.3이 그 표다.

**2. 조명 종류는 `EnvRendererCfg`를 통해서는 실현될 수 없었다.** 섹션 2.7은 "V0b가
렌더러를 놓으면 두 조명 종류는 구현 가능해진다"고 기대했고 실제로 그렇지만, 계획이 가정한
자리에서는 아니었다. `Rs` 경로는 `RenderConfig::light_dir`와 `::ambient`로 쉐이딩하는데 둘
다 `Renderer`를 만들 때 고정되고, `EnvRendererCfg`는 둘 다 싣지 않으며, 그 표면은 V0b의
것이다. 데모 씬의 유일한 발광 geom(`ceiling_light`)도 도움이 안 된다: `z = 0.8`로
`z = 0.5`의 오버헤드 카메라 **바로 위**에 있어 카메라 뒤이자 화면 밖이고, `Rs` 경로에서
발광 geom은 자기 픽셀 말고는 아무것도 비추지 않는다. 그래서 커널은 `es-eval` 쪽 순수 함수
둘로 안착했다 — `LightOverride::scene`(모든 geom `rgba`에 곱하는 이득이며, Lambert 항이
`albedo`에 선형이므로 이는 *정확히* 복사휘도 이득이다)와
`LightOverride::rotate_dir`(`light_dir`의 yaw) — 그리고 `es eval run`이
`es_render::Renderer`를 V0b 자신의 공개 `render_config` / `body_poses` / `camera_view`와
조합해 그것을 적용하며, 추첨이 바뀔 때만 렌더러를 다시 만든다. `es-render`와
`es-env/src/render.rs`는 손대지 않았다. `es` 크레이트의 `render` 피처가 `es-render`를 직접
의존성으로 얻었을 뿐이고, 기본 빌드에는 아무 비용이 없다(기본 꺼짐, spec 4.2).

**envelope은 조이지 않았다. 조일 필요가 없었기 때문이다.** 섹션 8은 데모의 clamp가
Deployment IR의 한계를 조여서 나온다고 말한다. 측정은 그것이 공짜로 나온다고 말한다: ACT
청크는 V1의 스크립트 전문가처럼 램프되지 않으므로(섹션 7.5 발견 4) plane이 스스로 clamp
하며, `deployment.toml`을 건드리지 않고도 비공허성 규칙이 성립한다. 그쪽이 더 나은 결과다 —
데모의 envelope이 시연을 기록할 때 쓴 바로 그 envelope이다 — 그리고 어느 쪽이든 INV-12는
건드리지 않는다.

**무엇을 측정했나 — 오라클 서버(RTX 4090, `~/venvs/es/bin/python`, mujoco 3.13.0,
torch 2.14.0+cpu), 2026-09-14.** `evaluation_hash 5d70c21c…3b9ff7`, 시드 101-116 —
시연을 수집한 1-50에서 제외된 홀드아웃이다. 성공률이 헤드라인이고 유일한 헤드라인이다(§12.4):

| 체크포인트 | nominal 성공률 | 평균 에피소드 길이 |
|---|---|---|
| 1,000 스텝 | 0.0625 (1/16) | 851.1 |
| 5,000 스텝 | 0.0000 (0/16) | 900.0 |
| 20,000 스텝 | 0.1250 (2/16) | 883.5 |

그리고 20,000 스텝 체크포인트의 스위트별(각 16 에피소드):

| 스위트 | 성공률 | 평균 에피소드 길이 |
|---|---|---|
| nominal | 0.1250 | 883.5 |
| light_intensity | 0.0625 | 887.4 |
| light_direction | 0.1250 | 883.5 |
| observation_delay | 0.1250 | 883.5 |
| torque_noise | 0.0000 | 900.0 |
| backlash | 0.0000 | 900.0 |

96 에피소드, 렌더된 프레임 85,407장, 28분. 비공허성 규칙은 성립한다 — `Success` 2회,
`Clamped` 76,967 스텝 — 그러나 통과로 세기보다는 그대로 말해두는 편이 나은 이유로
성립한다:

**어느 실행에서도 `ActionSource::Policy` 스텝은 단 하나도 없었다.** 85,407 중 `Clamped`
76,967, `Fallback` 8,440이고, 모든 스위트의 모든 셀에서 `envelope_violation_rate`가 정확히
`1.0`이다. 정책의 원본 청크는 매 스텝 envelope 밖에 있고, rate watchdog이 트립하며,
fallback이 약 10%의 스텝에서 자세를 유지한다. `light_direction` 행이 소수점 넷째 자리까지
`nominal` 행과 같다는 것은 같은 사실의 다른 면이다: 팔이 하는 일을 결정하는 것은 픽셀이
아니라 Safety Plane이다.

**가장 유력한 원인은 학습/추론 observation 불일치이며, 이는 정책에 대한 V3의 발견이 아니라
V2의 부채다.** `train_act.py`는 그래프의 상태 포트에 **원본** `observation.state` 행을
먹이고, Observation IR 노드 중 정확히 하나 `Dequantize`만 재구현한다(섹션 7.6 발견 3).
그러나 데모의 Observation IR은 `ImageInput -> Dequantize -> sink`가 아니다:
`StateInput -> Normalize{Range −1..1}`이고 `ImageInput -> Dequantize ->
Normalize{Range 0..1}`이다. 이미지 가지는 안전하다 — `normalize_range(x, 0, 1)`은
항등이다 — 상태 가지는 그렇지 않다: 추론에서 정책은 `(q + 1) / 2`를 받고, 학습에서는 `q`를
받았다. 고유수용 입력 전체의 아핀 이동이며, 발견 3이 "두 번째 노드가 나타나는 순간"
일어나리라 예측한 바로 그것이다. 플랜 V는 여기서 고치지 않는다: 고침은 학습 쪽의
plan-bake 단계(또는 컴파일된 plan을 통한 재학습)이고 둘 다 V2의 것이며, 이 패킷은 재학습이
금지되어 있다. **미해결 질문 11**이다.

따라서 이 표의 정직한 독법은 이렇다: 파이프라인은 끝에서 끝까지 돌고 실제 §10.1 표를
만들며, Safety Plane이 팔을 실제로 지배하고, *정책* 숫자는 아직 ACT의 측정치가 아니다 —
학습받지 않은 observation을 먹은 ACT의 측정치다.

**비디오.** 16개 nominal 셀에 대한 `es video mosaic --grid 4x4`가 384x392 프레임 900장을
주고(96x96 타일 16개에 8픽셀 레이블 띠), `python/es/encode_video.py --fps 50`이 제어
레이트의 18초 mp4를 준다: 20,000 스텝 실행에 `mp4v` 10.5 MB, 같은 원본 프레임을 서버 자신의
`ffmpeg 7.0.2`로 통과시킨 H.264 사본은 1.5 MB. mp4는 해시 체인에 없다(섹션 9); 프레임이
증거다.

### 7.9 As built (V2b): 학습 세트가 Observation IR를 통과한다

섹션 7.8의 표는 ACT의 측정이 아니었다. 열린 질문 11이 이유를 이름 붙였고, 섹션 7.6의 발견 3이 한
패킷 앞서 그것을 예고했으며, V2b는 그 발견이 이미 명세한 수리다: "데이터셋 위에서 관측 플랜을 굽는
`es` 단계, 한 줄이 아니라 패킷."

**결함을 근본 원인으로 한 번 더.** 증상은 빠진 `Normalize` 하나였다. 원인은 Observation IR 노드
하나가 구현을 둘 가졌다는 것 — `train_act.py`의 `dequantize`가 파이썬으로 다시 쓴 `Op::Dequantize`
였다 — 이고, 노드가 구현을 둘 가질 수 있게 되는 순간 학습이 적용하는 노드의 *집합* 역시 동기화를
지켜야 할 두 번째 대상이 된다. 상태 가지의 `Normalize{Range −1..1}`이 아무도 빼기로 결정하지 않은
채 사라진 경위가 그것이다. 학습도 상태를 정규화하도록 기워 넣었다면 증상은 고쳐지고 메커니즘은
남았을 것이다. 데모의 Observation IR에 세 번째 노드가 생기는 순간 살아남는 그런 수정은 없다.

**`es dataset bake --policy <bundle.esb> --out <dir> [--frames <tiles>] <root>`.** 기록된 모든
프레임을 `es_eval::ObservationBake`로 통과시킨다. 이는 물리 상태가 있던 자리에 데이터셋 행을 놓은
`capture`다: 같은 `CpuPlan::compile(obs, PlanMode::Release)`, 같은 `input_sources` 해석, 같은
`f64 -> f32` 입력 인코딩, 에피소드마다 같은 `plan.reset()`. "같은 로직"이 아니라 — 두 곳에서
호출되는 같은 함수다 (`crates/es-eval/src/bake.rs`, `crates/es-eval/src/runner.rs`). 에피소드당
safetensors 하나를 쓰며, Observation IR **자신의 출력 이름** 아래 `[frames, ...]` 모양 텐서와
`action`을 담고, `observation_hash` · `task_hash` · `compiler_hash` · 데이터셋 content 해시를 담은
`manifest.json`을 함께 쓴다. `train_act.py`는 그것을 읽고 `read_dataset`, `read_frames`,
`dequantize`, `plan_inputs`를 잃었다 — 321줄이 269줄이 되었고, 중요한 숫자는 이 저장소의
Observation IR 노드 구현 수가 둘에서 하나가 되었다는 것이다.

**위 계획이 몰랐던 세 가지.**

**1. `input_sources`는 모델이 없을 수도 있음을 배워야 했다.** 추론에서 입력은 먼저 `ModelInfo`에
대해 풀린다 — 관절의 `qpos` 범위, 센서의 범위 — 그다음에야 Task IR의 `ObservationSpec`에 대해
풀린다. 기록된 데이터셋에는 로드된 모델이 없다. `observation.state`와 타일을 담을 뿐이다. 두 번째
해석기를 쓰는 대신 `input_sources`가 `Option<&ModelInfo>`를 받고, `None`이면 모델을 색인하는 두
갈래가 그냥 사용 불가가 되어, 베이크가 먹일 수 없는 관측은 *첫 프레임보다 먼저 포트 이름을 대는
오류 하나*가 된다 — V3가 러너에 준 것과 같은 규율(섹션 7.8 발견 1), 같은 이유로. 데모의
`StateInput`은 바디를 이름하고 어느 쪽이든 `ObsSource::JointState { body, dof }`로 풀리며, 그래서
애초에 구워진다.

**2. 오라클이 비트 동일성일 수 있고, 파이썬도 GPU도 물리 백엔드도 필요 없다.**
`a_baked_frame_is_bit_identical_to_what_capture_serves`는 데모 모양의 Observation IR
(`StateInput -> Normalize{−1..1}` **그리고** `ImageInput -> Dequantize -> Normalize{0..1}`) 위에서
진짜 평가를 돌린다. 프레임 소스는 자기가 건네받은 `StateView`를 자기가 내준 타일과 함께 기록하고,
`PolicyRuntime`은 자기가 받은 모든 관측 맵을 기록한다. 그 같은 행과 타일이 이어서
`ObservationBake`를 통과하며, 모든 프레임의 모든 출력 텐서가 바이트 단위로 같아야 한다 —
픽스처에서 50 프레임 x 2 포트. 허용오차는 없다. 허용오차가 있다면 두 경로 중 하나가 자기 몫의 변환을
길렀다는 뜻일 테니까. 또한 두 노드가 실제로 *발화*함을 단언한다(상태에는 `(q + 1) / 2`, 이미지에는
바뀐 타일). 그래서 원본 입력의 복사본 둘을 비교해 통과할 수 없는데, 옛 배치라면 정확히 그렇게
통과했을 것이다.

**3. `pretrained`는 거부되고, 열린 질문 6이 그와 함께 닫힌다.** V2는 로워링이
`VisionEncoder{pretrained}`를 무시함을 발견했다. 존중하는 것은 한 줄 — `_backbone`의
`weights="DEFAULT"` — 이고, 그것은 틀린 한 줄이다: `EsPolicy()`가 *매* 생성마다 네트워크에서
ImageNet 가중치를 가져오게 되고, 여기에는 추론 시 `TorchRuntime::load` 내부도 포함되는데 거기서
`load_state_dict(strict=True)`가 잠시 뒤 전부를 덮어쓴다. 인스턴스화에 네트워크가 필요한 로워링은
§2.5에 어긋나고, 아무도 해시하지 않은 가중치는 체인 밖이다(§5.3). 그래서 `lower_to_torch`는 플래그와
두 갈래 출구를 이름 붙인 `LowerError::Unsupported`를 반환하고, 데모의 `learning.toml`은 이제
`pretrained = false`라고 말한다 — V2가 어차피 하고 있던 그대로다. 이는 `learning_hash`와
`policy_hash`를 움직이고, `task_hash`와 `observation_hash`는 움직이지 않으므로 구운 세트와
`evaluation.toml`은 그대로다. `es_ir::learning::testing::act_like`는 계속 `true`를 선언한다. ACT가
그런 것이기 때문이다. 그것을 로워링하는 `es-policy`의 두 테스트가 공유 헬퍼 하나로 플래그를 지운다.

**측정된 것, 오라클 서버에서 (RTX 4090, 학습은 `~/venvs/es-lerobot-cuda`, torch 2.11.0+cu129,
평가는 `~/venvs/es/bin/python`, mujoco 3.13.0), 2026-09-15.** 모든 손잡이가 V2의 것이다: 같은 50
에피소드 데이터셋과 타일, `--batch 8 --lr 1e-4 --seed 0`, 1k/5k/20k 체크포인트. 움직인 변수는
하나이고 그것은 관측이다. 베이크는 17,697 프레임, safetensors 1.9 GB,
`observation_hash f4a50730…55f6e0` — `evaluation.toml`이 선언하는 바로 그 해시이므로 구운 세트,
번들, 평가 문서가 모두 하나의 Observation IR를 이름한다. 로워링은 V2의 것과 바이트 동일하다
(`lowering_hash 956abb67…ec6d`). 아키텍처가 움직이지 않았다는 가장 깔끔한 진술이다.

학습 손실, 행동 청크에 대한 L1, 각 지점에서 끝나는 100 스텝의 평균, V2의 것과 나란히:

| step | 1 | 100 | 1,000 | 5,000 | 20,000 |
|---|---|---|---|---|---|
| V2 (원본 상태) | 0.7267 | 0.1825 | 0.0507 | 0.0309 | 0.0171 |
| V2b (구운 것) | 0.7261 | 0.2085 | 0.0528 | 0.0325 | 0.0166 |

같은 곡선이고, 그것이 예상된 결과다: 상태 수정은 입력 하나의 아핀 사상이고 옵티마이저가 그것을
흡수한다. **손실은 이 결함을 결코 탐지할 수 없었다.** 그래서 이 결함의 오라클이 임계값이 아니라
비트 동일성인 것이다.

성공률, 시드 101-116의 held-out 16 에피소드, 섹션 7.8의 것과 나란히:

| checkpoint | V3 nominal | V2b nominal | V2b 평균 에피소드 길이 |
|---|---|---|---|
| 1,000 steps | 0.0625 (1/16) | 0.1250 (2/16) | 861.8 |
| 5,000 steps | 0.0000 (0/16) | 0.0625 (1/16) | 859.3 |
| 20,000 steps | 0.1250 (2/16) | 0.0000 (0/16) | 900.0 |

어느 쪽이든 48 에피소드 중 성공 3회다. 스윕은 나아지지도 나빠지지도 않았고 그저 이리저리
움직였다. 정책이 아닌 무언가가 결정하는 표는 이렇게 생겼다.

그리고 20,000 스텝 체크포인트의 스위트 전체, 각 16 에피소드:

| suite | V3 | V2b |
|---|---|---|
| nominal | 0.1250 | 0.0000 |
| light_intensity | 0.0625 | 0.0000 |
| light_direction | 0.1250 | 0.0000 |
| observation_delay | 0.1250 | 0.0000 |
| torque_noise | 0.0000 | 0.0000 |
| backlash | 0.0000 | 0.0000 |

V2b의 모든 셀이 `episode_length 900.00`, `envelope_violation_rate 1.0000`이다.

**`evaluation.toml`은 `success_rate >= 0.5`를 요구한다. 가장 좋은 셀이 `0.1250`이었고
20,000 스텝 스위트는 `0.0000`이었다. 미달이고, 미달이 아니게 만들려고 무엇도 낮추지 않았다.** V3 자신의 비공허성 게이트도 실패한다:
`visible_learning_demo_run`이 *"non-vacuity: no suite produced a single Success episode"*로
패닉한다. 86,400 프레임, `Clamped` 77,880, `Fallback` 8,520 — 그리고 V3에서와 똑같이,
**단 한 스텝도 `ActionSource::Policy`가 아니었다**.

**그러므로 팔을 멈추고 있던 것은 관측이 아니었고, V2b의 진짜 결과는 추측이 아니라 측정된 다음
결함이다.** 숫자 셋이 그것이 무엇인지 말한다:

1. **정책은 잘 모방한다.** 구운 학습 프레임에서 청크의 첫 행동이 기록된 행동을 관절당
   0.02-0.16 rad 이내로 추종한다 (에피소드 0의 t = 312: 정책
   `+0.801 −0.727 +0.718 +1.551 −0.008 −0.076`, 기록
   `+0.773 −0.567 +0.661 +1.464 +0.000 −0.050`). 학습한 그 관측을 먹이면 ACT는 시연을 재현한다.
   이 패킷 이전에는 할 수 없던 주장이다.
2. **위반은 `Velocity`이고 스텝의 90%에서 난다.** `events.json`의 `ViolationKind` 비트셋을
   디코드하면: `Velocity` 77,880 (0.901), `Acceleration` 76,868 (0.890), `RateLimit` 72,467
   (0.839), `Position` 16,093 (0.186), `ViolationRate` 8,520 (0.099). 마지막 것이 모든
   `Fallback`을 만드는 워치독이다.
3. **시연의 명령은 자신의 관절 위치를 엔벌로프의 스텝당 예산의 네 배만큼 앞선다.**
   `SafetyPlane`은 속도를 `(cmd − last_safe) / dt`로 `velocity_max = 3.0`에 대고 클램프하므로
   명령은 컨트롤 스텝당 `3.0 × 0.02 = 0.0600` rad만 움직일 수 있다 — 그리고 기록된 `action` 열의
   스텝 간 델타 최대값이 정확히 `0.0600`이다. `es loop collect`가 **플레인을 통과한 뒤의** 명령을
   기록하고 스크립트 전문가가 그것을 포화시키기 때문이다. 한편 위치 서보의 정상상태 추종 오차인
   `|action[t] − qpos[t]|`는 중앙값 **0.2413** rad이고 0.7831까지 간다. ACT는 `qpos + 0.24`를
   내보내도록 배우고, 플레인은 명령이 스텝당 0.06만 전진하도록 허용한다. 간극은 구조적이고 결코
   닫히지 않으므로 팔은 모든 에피소드의 첫 스텝부터 속도 클램프를 받고, violation-rate 워치독이
   약 10분의 1의 시간 동안 `hold_position`으로 래치한다.

그중 무엇도 Observation IR 문제가 아니고, 무엇도 이 패킷 안에서 고칠 수 없다: `es-safety`,
`es-env`, 데모의 `deployment.toml`이 모두 V2b의 `forbidden` 목록에 있고, INV-12가 유일한 지름길을
금지한다. 다음 패킷의 일이고, V1/V3 모양이다 — 시연의 행동 규약과 Deployment IR의 엔벌로프가 서로
대조된 적이 없다. 아래 **열린 질문 12**가 어느 끝이 움직일지 묻는다.

그러니 정직한 독법은: 섹션 7.8의 표는 학습하지 않은 관측을 먹은 ACT의 측정이었고, 이번 것은 같은
엔벌로프의 더 좁은 해석을 통해 기록된 시연을 충실히 모방하는 정책을 Safety Plane이 거절하는
측정이다. 데모는 여전히 작동하지 않지만, 이제는 숫자가 붙은 이유로 실패한다.

**비디오.** V3가 쓴 것과 같은 파이프라인, `es video mosaic --grid 4x4`로 nominal 16 셀을 묶고
`python/es/encode_video.py --fps 50`으로 인코딩했다: 각각 384x392 900 프레임,
`demo-{1000,5000,20000}.mp4`가 `mp4v`로 13.0 / 11.6 / 11.4 MB, 서버의 `ffmpeg 7.0.2`를 통한
H.264 사본이 1.9 / 1.7 / 1.7 MB. mp4는 해시 체인에 없다(섹션 9). 프레임이 있다.

### 7.10 As built (V1c): 실행된 것이 기록되는 것이고, 팔이 실제로 멈추는 지점

7.9절은 미해결 질문 12로 끝났다. 시연의 액션 규약과 Deployment IR의 엔벌로프가 서로 대조된 적이
없고, 움직일 수 있는 끝이 셋이라는 것. V1c가 그 둘을 대조했고, 가장 먼저 발견한 것은 질문의 전제가
한 패킷만큼 어긋나 있었다는 사실이다.

**1. `action`은 이미 실행된 액션이었다.** `DomainRunner::emit_actions`가 `SafeAction::q`를
`ctrl`에 복사하고, `Env::step`이 `ctrl`을 기록하고, `to_lerobot`이 그것을 `action`으로 쓴다.
미해결 질문 12의 선택지 (a) — "`action`을 서보 목표가 아니라 다음 명령 위치로 기록한다" — 는 이미
코드에 들어 있던 변경을 묘사한 것이다. 저장소의 어떤 것도 그것을 단언하지 않았고 플레인 통과 이전
명령은 버려졌기 때문에 아무도 볼 수 없었을 뿐이고, 그래서 `Clamped` 프레임은 플레인을 다시 돌리지
않고는 읽을 수 없었다. V1c는 그것을 고정하고(`ctrl`은 `SafetyPlane::last_safe_action()`과 비트
단위로 같다, `f64::to_bits`, 허용오차 없음) 명령을 두 번째 컬럼 `action_commanded`에 남긴다. 이는
`dataset_schema_hash`를 이동시킨다. `task_hash`, `observation_hash`, `learning_hash`,
`lowering_hash`는 이동하지 않으며, `lerobot 0.6.1`은 V1b의 v3.0 내보내기를 통해 추가 피처를
읽는다(`RAN lerobot_v3_export`, 되돌려주는 features에 `"action_commanded": {"dtype":
"float32"}`가 있다).

**2. "`es loop collect --episodes N`이 에피소드 0만 푼다"는 추론 위상 문제였고, 플레인도
전문가의 `reset`도 아니었다.** 7.6절 발견 5는 용의자 셋을 지목했고 셋 다 틀렸다.
`es loop collect --expert`는 개입자 훅에서 `frame == 0`일 때 `ScriptedExpert`를 리셋한다. 그 훅은
`PolicyRuntime::infer`이고, 이는 제출된 관측이 *릴리스될 때* — 제출로부터 `expected_latency_ms`
틱 뒤에 — 실행된다(스펙 12.3절). 데모의 `learning.toml`은 20 ms 제어 주기에 대해 `15.0` ms를
선언하므로 모든 에피소드의 첫 호출은 frame 1이고 **`frame == 0`은 아예 한 번도 발화하지 않는다**.
갓 생성된 `ScriptedExpert`가 리셋된 상태로 시작하기 때문에 옳아 보였을 뿐이다. 에피소드 0은
리셋이 필요 없어 넘어가고, 그 뒤의 모든 에피소드는 직전 에피소드의 스테이지와 래치된 큐브를 안고
돈다. 에피소드 인덱스로 키를 잡으면 `--episodes 50 --seed 1`이 **1/50에서 50/50으로**, 홀드아웃
다섯은 0/5에서 5/5로 바뀐다 — 각각 명령 하나로. `crates/es-data`의 픽스처는
`expected_latency_ms = 0.0`을 선언했고, 그것이 바로 V1 자신의 오라클이 이것을 볼 수 없었던 이유다.
회귀 테스트가 이제 그 속성을 진술한다(`frame_zero_is_not_a_hook_an_intervener_may_reset_on`:
지연이 선언되면 개입자의 **어떤** 호출도 frame 0이 아니다).

같은 부류의 두 번째, 더 작은 사례: `Collector::run`은 에피소드 사이에 env와 청크 버퍼와 e-stop
래치를 리셋했지만 플레인의 홀드 타깃·속도·레이트 이력은 리셋하지 않았고, 그래서 플레인은 매
에피소드를 "팔은 아직 마지막 명령 자리에 있다"고 믿으며 열었다. 이제는 매 에피소드의 첫
프레임에서 측정된 자세로 심는다 — 실제 자세를 아는 호출자가 첫 `validate` 전에 하는 일이라고
`SafetyPlane` 자신의 문서가 말하는 것 — 그리고 `a_second_episode_repeats_the_first_exactly`는 그것
없이는 실패한다.

**3. 수집 경로와 평가 경로는 엔벌로프를 무엇에 대해 재는지에서 서로 다르고, 그 불일치는
수집기 쪽에서 닫을 수 없다.** `es_eval::runner`, `es_ros2::hil`, `es_runtime_embedded`는 **매**
`validate` 전에 `observe_state`를 호출해 `last_safe`, `prev_safe`, `vel`, `prev_vel`을 다시
심는다. 그러면 엔벌로프는 **추종 오차**의 상한이 된다. 데모의 숫자로는 가속도 단계가 구속조건이고
상한은 `a − q − q̇·dt ≤ acceleration_max · dt² = 0.008` rad이다. `Collector::run`은 에피소드당 한
번 심으므로 에피소드 안에서 엔벌로프는 *명령된* 이동을 제한한다 — `ScriptedExpert::chunk`가 자신을
맞추는 대상이고, 그 문서 주석이 바로 그렇게 말한다.

수집기가 매 스텝 다시 심게 하는 변경은 구현해서 측정했다. 두 경로를 일치시키는 변경이 그것이기
때문이다. 그 결과 스크립트 전문가가 **`expert_solves_the_pinned_seeds`에서 2/8**(임계값 `0.875`,
골든)로, **평가 자신의 시드 101–116에서 0/16**으로 떨어지고, 50 에피소드 수집은 18/50이 된다.
이유는 구조적이다. 전문가는 청크 시작 시점의 자세로부터 16행 청크를 계획해 10틱에 걸쳐 실행하며,
`step_max` / `accel_max` / 안티와인드업 리드를 어떻게 재조정해도 열 틱의 개루프 구간에서 0.008
rad 추종 오차 상한을 만족시키지 못한다. 그것이 가능해지는 두 노브는 `deployment.toml`의
엔벌로프와 `execute_chunk`이고 둘 다 V1c 밖이다. 그래서 이 비대칭은 추측이 아니라 숫자로 적힌 채
남고, 미해결 질문 12도 열린 채 남는다 — 선택지 (a)는 이미 되어 있던 것으로 지워지고, 선택지 (c)는
아래에서 답해진다.

**나머지 전부를 읽는 방식을 결정하는 진단: 미해결 질문 12의 선택지 (c)를 실제로 돌렸다.** V2b의
20,000 스텝 체크포인트를, deployment 문서가 *스크래치* 사본인 번들에 다시 패킹했다 —
`velocity_max` 3.0 → 30.0, `acceleration_max` 20.0 → 400.0, `ee_velocity_max` 0.6 → 6.0,
액션 레이트 한계 둘 다 → 1.0, 워치독을 포함한 나머지 전부는 그대로 — 그리고 같은 16개 nominal
에피소드로 돌렸다. 커밋된 픽스처는 수정하지 않았고 아무것도 끄지 않았다(INV-12).
`weights_hash`, `lowering_hash`, `task_hash`, `observation_hash` 모두 V2b의 것과 일치하므로
움직인 것은 엔벌로프뿐이다.

| | V2b, 커밋된 엔벌로프 | 같은 번들, 넓힌 엔벌로프 |
|---|---|---|
| `success_rate` (nominal, 16) | 0.0000 | **0.0000** |
| `episode_length` | 900.00 | 900.00 (16/16 타임아웃) |
| `envelope_violation_rate` | 1.0000 | **0.0551** |
| `ActionSource::Policy` | 0 / 86,400 | **13,606 / 14,400** |
| `ActionSource::Clamped` | 77,880 | 794 |
| `ActionSource::Fallback` | 8,520 | **0** |

**그러니 팔을 멈추고 있던 것은 Safety Plane이 아니었다.** 싸울 필요가 없는 엔벌로프를 주면 ACT는
스텝의 94.5 %를 직접 몰고, 워치독은 한 번도 래치하지 않으며, 성공률은 0으로 그대로이고 모든
에피소드가 900 스텝 예산을 다 쓴다. 엔벌로프를 넓히면 움직이지만 여전히 과제를 못 하는 정책을
얻는다. 누군가 선택지 (b)에 패킷 하나를 쓰기 전에 알아 둘 값어치가 있고, 아래 V1c의 재학습을
`success_rate >= 0.5`를 통과하려는 시도가 아니라 시연 규약의 측정으로 보고하는 이유다.

**무엇을 측정했는가, 오라클 서버, 2026-09-15.** 학습 노브는 전부 V2와 V2b의 것이다:
`--batch 8 --lr 1e-4 --seed 0`, 1k/5k/20k 체크포인트, 같은 학습 50 에피소드와 같은 홀드아웃 5개.
움직인 것은 수집이다. `es loop distill`로 병합한 단일 에피소드 50번이 아니라 **한 개의**
`es loop collect --episodes 50 --seed 1` 명령이고, 전문가는 에피소드당 리셋되며 플레인은
에피소드당 심어진다. 50/50 `Success`, 홀드아웃 5/5, 18,263 프레임(V2는 17,697),
`observation_hash f4a50730…55f6e0`과 `lowering_hash 956abb67…ec6d`로 1.9 GB 베이크 — 둘 다 V2·V2b의
것과 바이트 동일이므로 관측도 아키텍처도 움직이지 않았다.

시연 자체가 바뀌었고, 그것이 이 패킷의 결과다:

| | V2b의 세트 | V1c의 세트 |
|---|---|---|
| 전체 프레임의 `\|action − qpos\|` 중앙값 | 0.2413 | **0.0027** |
| 같은 값, 팔 관절 0–4만 | — | 0.0009 |
| 같은 값, 그리퍼(관절 5) | — | 0.1436 |
| `action ≠ action_commanded`인 프레임 | 기록 안 됨 | 11,580 / 18,263 |
| `action_source` | — | clamped 11,581, human 6,631, fallback 50, policy 1 |

리드가 100배 줄었다. 에피소드 시작 시 플레인에 팔의 실제 자세를 심어, `last_safe`가 0에서
출발해 표류하지 않게 한 결과다. fallback 50개는 에피소드당 하나 — 첫 추론 결과가 존재하기 전
frame 0의 청크 언더런 — 이고, 위반율 워치독은 한 번도 발화하지 않는다(V2b는 스텝의 약 10 %).

학습 손실, 액션 청크에 대한 L1, 각 지점에서 끝나는 100 스텝의 평균:

| step | 1 | 100 | 1,000 | 5,000 | 20,000 |
|---|---|---|---|---|---|
| V2 (raw state) | 0.7267 | 0.1825 | 0.0507 | 0.0309 | 0.0171 |
| V2b (baked) | 0.7261 | 0.2085 | 0.0528 | 0.0325 | 0.0166 |
| V1c (baked, 재수집) | 0.7476 | 0.2053 | 0.0560 | 0.0319 | **0.0176** |

세 번째로 같은 곡선이다. 성공률, 시드 101–116의 홀드아웃 16 에피소드:

| 체크포인트 | V3 nominal | V2b nominal | V1c nominal | V1c 평균 에피소드 길이 |
|---|---|---|---|---|
| 1,000 스텝 | 0.0625 (1/16) | 0.1250 (2/16) | 0.0000 (0/16) | 900.0 |
| 5,000 스텝 | 0.0000 (0/16) | 0.0625 (1/16) | 0.0000 (0/16) | 900.0 |
| 20,000 스텝 | 0.1250 (2/16) | 0.0000 (0/16) | **0.0625 (1/16)** | 876.7 |

그리고 20,000 스텝 체크포인트의 스위트별 표, 각 16 에피소드:

| 스위트 | V3 | V2b | V1c |
|---|---|---|---|
| nominal | 0.1250 | 0.0000 | 0.0625 |
| light_intensity | 0.0625 | 0.0000 | 0.1250 |
| light_direction | 0.1250 | 0.0000 | 0.0000 |
| observation_delay | 0.1250 | 0.0000 | 0.0000 |
| torque_noise | 0.0000 | 0.0000 | 0.0000 |
| backlash | 0.0000 | 0.0000 | 0.1250 |

96 에피소드 중 `Success` 넷. V2b는 0, V3는 2였다. 그래서 V2b 번들에서 *"non-vacuity: no suite
produced a single Success episode"*로 패닉했던 V3 자신의 비공허성 게이트가 이제 통과한다:
`RAN visible_learning_demo_run`, 1,544초. **`evaluation.toml`은 `success_rate >= 0.5`를
요구한다. nominal 스위트는 `0.0625`를 측정했다. 실패이고, 실패하지 않게 하려고 낮춘 것은 없다.**
V1c의 모든 셀은 여전히 `envelope_violation_rate 1.0000`을 보고하고, 여섯 스위트 84,772 프레임은
`Clamped` 76,412와 `Fallback` 8,360이며 **`ActionSource::Policy`는 단 하나도 없다** — 평가
경로의 엔벌로프 해석이 추종 오차를 0.008 rad로 제한하는데, L1 0.0176의 모방 정책 중 그 정밀도를
가진 것은 없기 때문이다. 시연을 정리한 것은 그것을 바꾸지 못했고 바꿀 수도 없었다. 그것은
데이터의 속성이 아니라 두 해석의 속성이다.

위 진단과 같은 넓힌 스크래치 엔벌로프로 돌리면 V1c의 20,000 스텝 체크포인트는
`envelope_violation_rate 0.4127`, 14,400 중 `Policy` 8,457과 `Clamped` 5,943(거의 전부
`violation.position` — 정책이 관절 소프트 한계를 넘는 자세를 명령한다), 폴백 0,
`success_rate 0.0000`을 보고한다. 두 정책 모두 플레인에서 풀려나면 움직이고, 둘 다 과제를 하지
못한다.

**그러니 V1c의 정직한 독법은: 시연이 고쳐졌고, 수집 루프가 고쳐졌고, 데모는 여전히 작동하지
않는다 — 그리고 처음으로 그 이유가 둘 중 어느 쪽의 하류도 아니다. 병목은 정책이다.** 이제 세
패킷이 각각 교란 요인을 하나씩 제거했고(V2b는 관측, V1c는 시연과 에피소드 루프, 두 진단은
엔벌로프), 남은 것은 시연 50개와 옵티마이저 스텝 20,000번, 그리고 학습 프레임을 L1 0.0176으로
모방하면서 처음 보는 큐브 자세에는 열여섯 번에 한 번쯤 일반화하는 ACT다.

**비디오.** V3와 V2b가 쓴 것과 같은 파이프라인, `es video mosaic --grid 4x4`로 nominal 16 셀을
묶고 `python/es/encode_video.py --fps 50`으로 인코딩했다: 각각 384x392 900 프레임,
`demo-{1000,5000,20000}.mp4`가 `mp4v`로 12.9 / 12.7 / 11.4 MB, 서버의 `ffmpeg 7.0.2`를 통한
H.264 사본이 2.0 / 1.9 / 1.7 MB. mp4는 해시 체인에 없다(섹션 9). 프레임이 있다.

### 7.11 구현 결과 (V5 1단계): 빠른 사이클, 그리고 쓸 수 없었던 분할

패킷 `docs/packets/M5/V5-fast-cycle.ko.md`. 수집 -> 학습 -> 평가 한 사이클은 순차 실행으로 약 한
시간이다. nominal 16 에피소드 약 5분, 6개 스위트 96 에피소드 실행 약 28분, 배치 8로 20,000 스텝 학습
약 11분. V5는 이것을 **숫자를 하나도 바꾸지 않고** 줄인다. 여기서 성공률도, loss도, 프레임도 움직여서는
안 된다.

**1. 에피소드 단위 분할은 존재하지 않고, 그 사실 자체가 발견이다.** 워커 하나당 에피소드 하나라는
당연해 보이는 설계는 쓸 수 없다. 코드를 더 쓴다고 해결되는 이유가 아니다. 하나의 **셀**(스위트 하나와
그 모든 에피소드) 안에서 `Evaluation::run_with_frames`는 `Env` 하나, `SafetyPlane` 하나, 단조 증가하는
청크 `seq` 하나를 셀 전체에 걸쳐 유지하고, `Env::reset`은 태스크 자신의 `RandomizationPlan`을 키로 쓰는
env별 에피소드 카운터를 증가시킨다(`crates/es-env/src/env.rs:224`). 따라서 에피소드 5의 초기 상태는
에피소드 0..4를 실제로 돌리지 않고서는 재현할 수 없다. `envelope_violation_rate`와
`chunk_underrun_rate` 뒤의 안전 카운터는 셀 단위 합계이고, `env.metrics()`도 셀 전체에 걸쳐 누적된다.
에피소드 단위 워커는 자기 앞 에피소드들을 다시 돌리거나(속도 이득 없음) 다른 숫자를 내놓거나(허용 안 됨)
둘 중 하나다. 에피소드 카운터를 임의 위치로 옮기는 것은 `es-env` 변경이고, 이 패킷은 그것을 스스로
금지한다.

> **패킷 M7/T8이 다시 물었고, 답은 바뀌지 않았다.** 장애물의 절반은 사라졌다. `Env::seek_episode`가
> 들어왔고, seek한 env는 데모 씬과 커밋된 Task IR에서 재생한 env와 비트 단위로 같다
> (`cargo test -p es-env --test seek -- --ignored`, 오라클 서버 2026-09-21, `k ∈ {1, 3, 7}`,
> `.estraj` 포함). 나머지 절반은 `es-env`의 문제가 아니다. 같은 커밋된 데모 문서 위에서 T8 이전
> 빌드와 T8 빌드의 `es eval run`을 돌려 보니 산출물이 두 군데에서 움직였다. Safety Plane의
> `ViolationRate` 링이 `begin_episode`를 넘어 살아남고, `StepEvent::tick`이 셀 전체에 누적되는
> 물리 클럭이다. 그래서 단위는 셀로 남았고 플래그는 추가하지 않았다. 증거와 측정된 차이, 그리고
> 리뷰가 내려야 할 두 결정은 `docs/design/evaluation-execution.ko.md` 2.7절에 있다.
>
> **그리고 바뀌었다 (패킷 M7/R1).** 오너가 2026-09-21에 둘 다 결정했다(spec 28.11):
> `begin_episode`가 링을 비우고(spec 9.4), `StepEvent::tick`이 에피소드에서부터 센다
> (spec 10.5). 둘이 들어온 뒤 `(cell, episode)` 분할은 커밋된 데모 문서 위에서 순차 실행과
> 바이트 단위로 같고, `--jobs N`은 셀이 아니라 에피소드를 나누며, 스위트가 하나뿐인 평가도
> 병렬화된다. 동일성 증거와 새 벽시계 표는 같은 2.7절에 있고, 의미론 변경이 이 노트 자신의
> 숫자에 무엇을 했는지는 7.33절에 있다.

반면 **셀**은 진짜로 독립적이다. 자기 `Env`, 자기 `SafetyPlane`, 0에서 시작하는 `seq`, 첫 에피소드를
포함한 모든 에피소드에서의 `plan.reset()`. 셀들이 공유하는 것은 두 가지뿐이다. 컴파일된 `CpuPlan`
— 에피소드마다 reset되므로 새로 컴파일한 plan과 reset한 plan은 같은 plan이다 — 과 `PolicyRuntime`.
후자는 피드포워드다(`TorchRuntime::infer`는 호출 사이에 상태를 갖지 않고, 시간 윈도우는 policy가 아니라
plan에 있다). **그래서 `--jobs N`은 셀을 라운드로빈으로 분할한다. 셀 `c`는 샤드 `c % N`에 속한다.**
데모의 28분은 여섯 개 스위트에 있다. nominal 실행은 스위트 하나라 아무 이득이 없고, `N`은 스위트 수로
클램프된다.

**2. 순차 경로가 곧 샤드 하나짜리 샤드 경로다.** `Evaluation::run_shard`는 자기 샤드가 가진 셀을 돌리고
판정은 하지 않는다. `Evaluation::merge`가 모든 워커의 셀을 `EvaluationIr::suites` 인덱스로 정렬하고
(안정 정렬이라 셀 안의 메트릭 순서는 그대로다) `judge`, 해시 체인, `report.json`, `evaluation.lock`을
부모에서 한 번만 계산한다. `run_with_frames`는 이제 말 그대로 `run_shard((0, 1))` 다음 `merge`이므로,
`--jobs 1`과 기존 경로의 바이트 동일성은 두 번째 구현이 아니라 구조로 보장된다(스펙 3.5 티어 1).
워커가 스레드가 아니라 프로세스인 이유는 물리 백엔드 자체가 env 하나를 담은 파이썬 서브프로세스이고
`TorchRuntime`이 또 하나를 들고 있기 때문이다. 스레드는 같은 인터프리터 두 개 뒤에 줄을 설 뿐이다.

프레임은 병합 단계가 아예 필요 없다. 셀은 `<frames>/<suite>-<NN>/`에 쓰고, 셀 이름은 전역적으로
유일하며 샤드들은 서로 겹치지 않는 셀을 갖는다. 그래서 자식들이 한 디렉터리에 써도 충돌하지 않는다.
각자 `events.json`의 자기 몫을 돌려주고 부모가 (서로소인) 맵들을 이어 붙인다.

**3. 워커 하나를 잃으면 오류이지 짧은 리포트가 아니다.** 병합된 셀 집합이 정확히 `0..suites.len()`,
각각 한 번씩이 아니면 `EvalError::Shard`가 무엇을 받았는지 이름을 붙여 거절한다. 이 검사가 없으면 자식
하나를 잃은 실행이 완벽하게 올바른 `evaluation_hash`를 달고 다섯 개 스위트짜리 리포트를 내놓는다.
실제 측정과 구분이 안 되므로 가능한 실패 모드 중 최악이다. CLI에서 실패한 자식은 샤드 번호, 종료 코드,
자식의 마지막 stderr 줄을 담은 오류 **하나**가 된다.

**4. 아홉 메트릭 중 둘은 이미 재현 불가능하고, `--jobs`는 그것을 드러낸다.** `physics_steps_per_sec`와
`actions_per_sec`은 `EnvMetrics::simulation_wall`로 카운터를 나눈 값이라, 샤드 실행에서는 각 워커가
자기 시계를 잰다. 순차 경로에서도 실행마다 움직인다. 데모의 `evaluation.toml`도 오라클 픽스처도 둘 다
선언하지 않는다. 고치지 않고 기록한다. 고치는 일은 `es-env`의 몫이고, 이 노트의 어떤 숫자도 그것에
의존하지 않는다.

**5. 학습 플래그, 그리고 그중 인용해도 정직한 것.** `--resident-gpu`는 forward마다 샘플 하나를 복사하는
대신 베이크된 세트 전체를 한 번에 디바이스로 올린다. 텐서가 *어디 사는지*만 바꾸고 그 외에는 아무것도
바꾸지 않으므로, 같은 seed에서 loss 곡선은 — 체크포인트까지 — **비트 단위로 동일**하다.
`resident_gpu_does_not_move_the_loss`가 두 `--loss-curve` 파일을 바이트로 비교해 이것을 고정한다.
`--amp bf16`과 `--compile`은 비트를 바꾸기 **때문에** 옵트인이다. 픽스처에서 로컬로 측정한 값: 배치 4로
40 스텝, `initial_loss`가 fp32에서 0.5596105996519327, bf16 autocast에서 0.5595557652413845.
`--batch`는 위 숫자들의 재현성을 위해 8로 유지한다. 올릴 때의 문서화된 관례는 `--lr` 선형 스케일링이다
(`--batch 32 --lr 4e-4`).

**6. 측정 결과 (2단계), 오라클 서버, 2026-09-15.** 아래 모든 실행 전에 `nvidia-smi`는 사용률
0% / 사용 메모리 55 MiB를 보였다. 16코어 박스는 다음 항목에서 스스로 그렇게 만든 경우를 빼면 그
외에는 유휴 상태였다. V1c의 `trained-20000.esb`, `build/`, `baked/`(섹션 7.10)가 아래 모든 행의
픽스처이며, 여기서 섹션 7.9나 7.10이 기록한 어떤 수치도 움직이지 않았다.

| 항목 | Target | Observed |
| --- | --- | --- |
| 6 스위트 96 에피소드, 순차 (`--jobs 1`) | 28분 (V3 기준선) | 25:56 (같은 설정의 이전 실행, 이후 덮어써짐, 25:52 -- 일치) |
| 6 스위트 96 에피소드, `--jobs 6`, 아래 수정 전 | 약 5-6분 | 끝까지 돌리지 못함: 순차의 초당 ~55프레임 대비 합산 초당 ~5프레임, 85분에 96개 중 35개 셀, 끝까지 기다리지 않고 종료(4시간 이상으로 투영됨) |
| 6 스위트 96 에피소드, `--jobs 6`, 아래 수정 후 | 약 5-6분 | **5:49** |
| nominal 단독, 16 에피소드, 순차, 독립된 두 번 실행 | 약 5분 | 4:16.49 / 4:16.68, 서로 바이트 동일(`report.json`과 모든 프레임) |
| `report.json` / `events.json`, `--jobs 1` 대 `--jobs 6`(수정 후) | 바이트 동일 | 실제 백엔드에서는 비트 동일하지 않음 -- 아래 참조; `FakeBackend` 오라클은 바이트 동일 유지 |

nominal 단독 설정은 스위트가 하나뿐이다. `crates/es/src/cmd/eval.rs`는
`a.jobs.min(eval_ir.suites.len().max(1) as u32)`로 제한하므로 그 위에서 `--jobs 6`은 구조적으로
`--jobs 1`로 실행된다 -- 별도로 시간을 잰 실행이 아니라 그 제한 코드를 읽어서 확인한 사실이다.

**첫 `--jobs 6` 실행은 순차보다 5배 *느렸다*.** 각 샤드는 같은 바이너리를 다시 호출한 것이고
(`spawn_shards`), 각 샤드 자신의 `TorchRuntime` 서브프로세스는 스레드 풀을 박스의 모든 코어로
기본 설정한다. 16코어에 그런 여섯 개가 한꺼번에 약 90개의 OS 스레드를 원했고(서브프로세스당
`nlwp` 23을 직접 측정), load average는 ~47을 유지했으며, 박스는 계산이 아니라 컨텍스트 스위칭에
시간을 썼다 -- 85분에 96개 중 35개 셀, 끝나려면 4시간을 넘길 것으로 투영됐다.
`crates/es/src/cmd/eval.rs`에서 수정: 이제 `spawn_shards`는 호출자가 이미 export하지 않은 경우에
한해(`shard_thread_env` -- passthrough가 우선) 각 샤드의 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` /
`OPENBLAS_NUM_THREADS` / `TORCH_NUM_THREADS`를 `cores / jobs`(`shard_thread_cap`)로 설정하며,
둘 다 실제 서브프로세스를 띄우지 않고 유닛 테스트된다. 수정 후 여섯 샤드는 각각 스레드 2-3개
(`nlwp` 3, 각 ~136% CPU)를 측정했고 load average는 ~5.5로 떨어졌으며, 실행은 5:49에 끝났다 --
패킷이 목표한 범위 안이고, 그 사이 박스에서 달라진 다른 것이 없다는 점은 전후의
`nvidia-smi`/`uptime`으로 확인했다.

**그 수정은 실제 백엔드의 바이트 동일성을 대가로 치르며, 이 패킷의 forbidden 목록은 그것을 여기서
닫도록 두지 않는다.** `sharding_the_cells_produces_a_byte_identical_report`(`FakeBackend`, 실제
부동소수점 없음)는 수정 후 재검증해도 여전히 바이트 단위로 통과한다. 그러나 실제 `mujoco-cpu` +
`torch` 스택에서는 `--jobs 1`의 자체 서브프로세스는 (이 패킷 이전과 마찬가지로) 제한되지 않은 채
남아 있는 반면 `--jobs 6` 샤드는 이제 `cores/jobs` 스레드로 제한된다 -- `--jobs 1`이 쓰는 것과
다른 스레드 수이고, CPU 스레드 리덕션은 정확히 결합법칙을 만족하지 않는다. 측정 결과: 병합된
report의 24개 셀 중 6개가 다르다 --

| | `--jobs 1` | `--jobs 6`(수정 후) |
| --- | --- | --- |
| `light_intensity` `episode_length` | 845.3125 | 845.25 |
| `nominal` `failure_mode_histogram` `violation.position` | 2,592 | 2,430 |
| `light_intensity` `failure_mode_histogram` `violation.position` | 1,677 | 1,936 |

-- 그리고 `success_rate`와 `envelope_violation_rate`는 24개 셀 전부 동일하며, 에피소드 후반부에서
어떤 스텝의 위반 분류가 뒤바뀐 뒤로 소수의 개별 프레임이 다르다. 수정 탓으로 돌리기 전에 분리해서
확인했다: 같은 nominal 단독 설정의 독립된 두 `--jobs 1` 실행은 서로 바이트 동일하고(위 표, 프레임
포함 -- 실행마다 달라지는 고유한 비결정성을 배제), `--jobs 1` 실행 전에 `OMP_NUM_THREADS=16`(이
박스의 코어 수)을 명시적으로 export해도 설정하지 않은 기본값 실행과 바이트 단위로 동일하다
("명시적 대 기본값"이 변수라는 가설을 배제) -- 즉 divergence는 프로세스 분리가 아니라 스레드
*수*를 따라간다. 이는 `MuJoCoCpuBackend` 자신이 선언한 `DeterminismTier::PhysicsMeaning`(섹션 9:
tier 3, 비트 단위 아님)과 CPU 스레드 커널 일반의 하류 결과이지, 병합/샤딩 로직의 결함이 아니다.
이를 닫으려면 `--jobs 1` 자체의 스레드 수도 제한해야 하는데, 이미 커밋된 V1c/V2b/V3 수치를 건드릴
위험 없이 이 패킷이 그 값을 고를 근거가 없다 -- forbidden. "수정됨"이 아니라 실제 백엔드의 문서화된
기존 한계로 남긴다.

**학습, V2의 노브, 다섯 행 전부, `--checkpoint-at 20000`으로 20,000 스텝:**

| 플래그 | 실제 시간 | `initial_loss` | `final_loss` |
| --- | --- | --- | --- |
| (a) 기본값 | 10:53.07 | 0.066787 | 0.017945 |
| (b) `--resident-gpu` | 10:57.00 | 0.066568 | 0.017749 |
| (c) `--resident-gpu --amp bf16` | 12:53.49 | 0.066955 | 0.018015 |
| (d) `--resident-gpu --compile` | 10:00.07 | 0.066878 | 0.017895 |
| (e) `--resident-gpu --batch 64 --lr 8e-4` | 1:24:31 | 0.050043 | **NaN** |

(a)와 (b)는 `--device cuda`에서 비트 동일하지 **않다**: `cmp`는 `--loss-curve`와 체크포인트
양쪽에서 다르다고 판정하며, 두 번째 옵티마이저 스텝부터 갈라진다(첫 스텝은 둘 다
0.4806089662...로 일치; 두 번째는 0.48060897 대 0.48060090으로 상대 차이가 약 1e-5이며 그
뒤로 커진다). 이는 평범한 CUDA 커널/알고리즘 선택 비결정성이다 -- `train_act.py`는
`torch.use_deterministic_algorithms`를 설정하지 않으며, resident 텐서의 다른 메모리 레이아웃은
매번 복사되는 텐서와 다른 cuDNN/cuBLAS 커널을 고를 수 있다. 설계 노트와 패킷의 "비트 동일" 주장은
유닛 오라클이 실제로 검사하는 장치(CPU, 아래 7번 항목)에서는 정확하고 CUDA에서는 성립하지 않는다.
둘 다 화해시키지 않고 그대로 기록한다. 화해시키는 일은 이 패킷 밖이다(`train_act.py`에 결정성을
강제하는 코드를 추가하는 것도, 이 패킷의 forbidden 목록 밖에 있는 수치를 움직이는 것도 여기서 할
수 없다).

(c)는 (a)/(b)보다 **느리다**, 빠르지 않다: `--batch 8`에서는 모듈이 샘플을 한 번에 하나씩
처리하므로(섹션 7.11의 5번 항목) forward 하나하나가 작아서 연산량이 아니라 실행 오버헤드에
지배되고, bf16 autocast의 연산당 오버헤드가 거기서는 본전을 뽑지 못한다. (d)는 소폭 더 빠르다
((a)/(b) 대비 약 8%) -- `torch.compile`의 융합은 이 규모에서도 무언가 할 일이 있고, 웜업 비용이
160,000번의 forward/backward 호출에 걸쳐 상각된다. (e)는 같은 20,000 스텝에 대해 문서 자체의
`--batch N` / `--lr` 선형 스케일링 관례(`--batch 64`, 8배, `--lr 8e-4`, 8배)를 그대로 따랐고
`NaN`으로 발산했다. 이 배수에서, 이 모델과 이 50 에피소드 데이터셋에는 그 관례가 성립하지 않는다는
발견이며, (e)가 (a)-(d)와 동등하다는 주장이 아니다(§12.4: 여기서 (e)를 `step/s` 수치나 권고로
인용하지 않는다).

**7. `cargo test -p es-policy --test ir_training -- --ignored --nocapture`, `ES_PYTHON`을 CUDA 학습
venv로 지정, 서버 트리에서:**

```
RAN act_training_uses_baked_observations: loss 0.4456 -> 0.1828 over 40 steps (0.410x), chunk
  [10, 6], max_abs vs a direct forward 0e0 (tol 1e-5), observation_hash
  f4a50730ac95b91734c9678e75d9e6bc1845578bc2985e45b409e80f3355f6e0, torch 2.11.0+cu129
RAN resident_gpu_does_not_move_the_loss: 40 bit-identical steps, 825 bytes of curve
```

`resident_gpu_does_not_move_the_loss`는 먼저 수정이 하나 더 필요했다: 마지막 assertion이 리터럴
부분 문자열 `"resident_gpu":true`를 찾았는데, Python 기본 `json.dumps`는 이를 절대 내지 않는다
(콜론 뒤에 항상 공백을 둔다: `"resident_gpu": true`) -- 테스트 자체의 기존 버그이며, 실제로
돌리려면 `torch`가 필요해서 지금에서야 발견됐다. 패키지가 없으면 CI는 이유를 출력하고 건너뛴다.
`crates/es-policy/tests/ir_training.rs`에서 수정; 이 테스트는 `--device cpu`로 돈다
(`train_act.py`가 기본으로 떨어지는 값이며 이 패킷이 바꾸지 않았다) -- 그리고 그것이 바로 위에서
측정한 CUDA divergence와 이 테스트의 비트 동일 주장이 서로 모순되지 않는 이유다. 서로 다른
장치다.

다시 발견하지 않도록 기록해 두는 로컬 제약 하나. `--compile`은 윈도우 개발 머신에서 시험할 수 없다.
`torch._inductor`가 템플릿을 ANSI 코드페이지로 읽어 `cp949` 로캘에서 `UnicodeDecodeError`로 죽는다.
이 스크립트가 아니라 torch의 버그다. 위 (d) 행이 리눅스 서버에서 받은 측정값이다.

### 7.12 구현 결과 (V6): 하나의 엔벌로프 의미론, 그리고 이제 전문가를 통과시키는 하니스

패킷 `docs/packets/M5/V6-envelope-semantics.md`. 7.10절의 발견 3은 수집 경로와 평가 경로가
Safety Plane의 동적 단계들을 무엇에 대해 재는지에 대해 서로 다르다고 말하고, V1c가 `es-safety`를
건드릴 수 없었기에 열어 두었다. 이 불일치는 취향 문제가 아니다. 스펙에 답이 있고, 그 답은 데모가
지금까지 낸 *모든* 평가 숫자가 의미를 갖는지를 결정한다.

**1. 결함, 정확하게.** `Collector::run`은 에피소드의 첫 프레임에서 `observe_state`를 불렀고,
`es_eval::runner`·`es_ros2::hil`·`es_runtime_embedded`는 **매** `validate` 앞에서 불렀다.
`observe_state`는 `last_safe`, `prev_safe`, `vel`, `prev_vel`을 덮어썼다 — 클램프의 3·4·7단계가
차분을 취하는 바로 그 네 필드다. 그래서 평가 경로에서 `velocity_limit`·`acceleration_limit`·
`rate_limit`은 *명령*과 *측정된 관절* 사이의 차이, 즉 위치 서보의 추종 오차가 되었다. 데모의
숫자에서는 가속도 단계가 구속하고 그 한계는 `a − q − q̇·dt ≤ acceleration_max · dt² = 0.008` rad다.
Deployment IR이 허용하는 것에 정확히 맞춰 스스로를 페이싱하고 `es loop collect`에서 50/50을 내는
스크립트 전문가가 `es eval run`에서는 **16 중 0**을 낸다. V3·V2b·V1c의 모든 평가 셀이 0이라는
천장에 대고 측정된 것이며, 그래서 전부 `envelope_violation_rate 1.0000`이고
`ActionSource::Policy` 스텝이 하나도 없다.

**2. 의미론은 편의가 아니라 스펙에서.** §9.3의 표는 *정책 출력에 적용되는 정적·동적 제약*이고
동적 행은 모두 **클램프**라고 말한다 — `velocity_limit` "관절·EE 속도 상한 / 클램프 + 카운터",
`rate_limit` "액션 1차·2차 미분 상한 / 필터링". 측정된 속도는 클램프할 수 없다. 클램프할 수 있는
것은 plane이 지금 내보내려는 값뿐이다. 따라서 그 행들이 제한하는 양은 plane 자신의 명령들의
차분이고, 측정값은 에피소드의 첫 명령을 팔이 실제로 있는 곳에 놓는 시드로서 딱 한 번 들어온다.

§9.5는 같은 것을 반대쪽에서, 더 강하게 말한다: **"`deployment_hash`가 같으면 안전 동작이
같다"** — 그리고 그것이 §27.1 증거물의 핵심 주장이라고 못박는다. 피드백에서 계산한 클램프는 그
주장을 만족할 수 없다. MuJoCo의 `qvel`은 정확하고 잡음이 없지만, 실제 SO-101에서 같은 값은
시리얼 버스를 통해 더 낮은 속도로, 양자화와 지연을 달고 돌아온다. 그것을 읽는 엔벌로프는 같은
`deployment_hash`에 대해 두 곳에서 다른 액션을 내보내고, §3.4의 결정성 규칙은 잡을 것이 없어진다.
그래서 오케스트레이터의 기본값 — *측정된* 속도를 제한하고 제동한다 — 은 **채택하지 않았다**:
스펙은 침묵하지 않고, §9.5가 그것을 배제한다. 명령과 측정 사이의 점프 검사도 추가하지 않는다.
§9.3의 표에 그런 행이 없기 때문이다. 명령이 *어디로* 갈 수 있는지를 제한하는 행은
`position_limit`과 `workspace` 둘이고, 둘 다 그대로다.

물리적 독해도 같다. SO-101의 서보는 `STS3215`이고 씬은 그것을 MuJoCo `position` 액추에이터로
구동한다: `kp = 998.22`, `kv = 2.731`, `forcerange = ±2.94 N·m`. 루프는 서보 *안에서* 닫힌다 —
호스트는 목표 위치를 보내고 서보가 오차로 토크를 만든다. `2.94 / 998.22 = 0.0029` rad이 그 토크가
포화하는 지점이다. **3 밀리라디안의 추종 오차면 이미 최대 토크다.** 그러니 V6 이전 평가 경로의
`0.008` rad 한계는 서보의 오차 신호를 포화점의 약 2.7배에서 제한한 것이었다. 속도라고 적힌 행에서
토크를, 그것도 잘못 제한하고 있었던 것이며, `torque_limit`은 두 행 위에 따로 있고 모델 자신의
`forcerange`가 이미 그것을 강제한다. `tests/fixtures/visible-learning/deployment.toml`은 이제
선언하는 모든 숫자의 유도 과정을 담고 있다. `es-safety`가 강제하지 *않는* 세 개
(`ee_velocity_max`, `contact_force_max`, 거리 최솟값 — plane에는 FK도 접촉 질의도 없다)도 포함해서.

**3. 바뀐 것, 네 줄이다.** `SafetyPlane::observe_state`는 `SafetyPlane::begin_episode` 직후(또는
생성 직후) 첫 호출에서만 명령 체인을 심고, 그 이후의 호출에서는 아무것도 건드리지 않고 반환한다.
`begin_episode`는 e-stop 래치를 해제하고 *동시에* 시드를 다시 무장시키며, 두 에피소드 경계에
있던 맨 `reset_latch()`를 대체한다. 이제 모든 소비자가 매 `validate` 앞에서 `observe_state`를
불러도 되고 — 실제로 모두 그렇게 한다 — 그것이 무엇을 뜻하는지는 호출자가 아니라 plane이
결정한다. `validate`의 시그니처는 그대로고(INV-13), 엔벌로프 숫자는 하나도 움직이지 않았으며,
무엇도 비활성화되거나 우회되지 않았고(INV-12), 새 트레이트도 없다(INV-17). 수집기는 `if frame ==
0`을, 평가 러너는 `reset_latch`를 잃었다.

**4. 로컬에서 숫자가 말하는 것.** `crates/es-safety/tests/envelope_reference.rs`는 데모의 커밋된
`deployment.toml`로 plane을 만들고, 전문가 자신의 페이싱 규칙(스텝당
`0.9 · min(velocity_max·dt, first_diff_max)`, 스텝 변화량 `0.9 · min(acceleration_max·dt²,
second_diff_max)`)으로, 최대 **0.1774 rad**(옛 한계의 22배) 뒤처지는 플랜트에 대고 구동한 뒤, 60번
연속 `ActionSource::Policy`이고 출력이 명령과 비트 단위로 같다는 것 — *그리고* 매 tick 관측하는
호출자와 한 번만 관측하는 호출자가 동일한 수열을 낸다는 것을 단언한다. V6 이전 코드에서는 스텝
1에서 실패한다. 나머지 둘은 V6가 아무것도 넓히지 않았음을 고정한다: 3 rad/s 한계에 대고 50 rad/s를
요구하는 명령은 여전히 `ViolationKind::Acceleration`과 함께 `Clamped`이고 여전히 세어지며,
`begin_episode`는 두 번째 에피소드가 열리는 곳에서 여전히 다시 시드한다.

**5. 없던 오라클, 그리고 어디서 도는가.** `expert_passes_the_evaluation_harness`
(`crates/es/tests/cli.rs`)는 `ScriptedExpert`를 **정책으로서** `es_eval::Evaluation`에 태워
돌린다 — 진짜 러너, 진짜 plane, 진짜 Task IR 성공 술어 — `expert_solves_the_pinned_seeds`와 같은
고정 시드 8개, 같은 `0.875` 임계로. 두 상수는 이제 두 테스트가 공유하는 한 쌍이다. 이것은 **서버
오라클**이다: SO-101 씬을 구동할 수 있는 것은 `MuJoCoCpuBackend`뿐이고, `mujoco`가 없으면 이유를
찍고 건너뛴다 — 그것이 미러링하는 수집 오라클과 똑같이. 테스트 안에 밝혀 둔 두 가지 좁힘: 시드당
`Evaluation::run` 한 번, 에피소드 하나씩(`Evaluation::run`은 `Env`를 `seeds[0]`으로 시드하고 Task
IR의 무작위화를 에피소드 카운터로 키잉하므로, 8 에피소드 셀 하나는 *한* 시드의 여덟 번 추첨이
된다), 그리고 상수 96×96 프레임 — 전문가는 관절을 읽고 픽셀을 결코 읽지 않으므로, 이것이 이
오라클을 Vulkan 장치 없이 `mujoco`만으로 돌게 해 준다.

**6. V6가 찾았지만 고치지 않은 두 비대칭. 둘 다 엔벌로프가 아니기 때문이다.** (V6b, 7.13절에서
닫혔다 — 그리고 아래 둘 중 첫째는 잘못 서술되어 있다. 수집기도 추론 속도로 재계획하지 *않는다*.
매 tick 재계획하며 `ChunkBuffer`를 통해 temporal ensembling을 하고, 평가에는 그것이 아예 없었다.
7.13절에 정정과 측정이 있다.)

* **평가 러너는 매 제어 tick마다 재계획한다.** `run_episode`는 스텝마다 정책을 부르고 매번 새
  `seq`를 주므로 커서가 리셋되고 모든 청크의 0행만 실행된다 — `action.execute_chunk = 10`은 그
  경로에서 죽어 있고 deployment의 `rate.inference = 5 Hz`도 지켜지지 않는다. 수집기는 추론
  속도로 재계획하고 열 행을 실행한다. `ExpertCfg::pace_to`가 `execute_chunk`를 읽는 대신
  재계획당 행 수를 인자로 받게 된 이유가 이것이다: 그것은 엔벌로프가 아니라 *소비자*의 속성이다.
  ACT 정책 입장에서는 평가가 수집보다 네트워크를 열 배 자주 질의한다는 뜻이고, 이는 실제
  훈련/시험 불일치이며 열린 질문 13이다.
* **평가 러너는 수집기보다 에피소드를 하나 앞서 리셋한다.** `Env::new`가 이미 리셋하고(무작위화
  추첨 0) `run_episode`가 맨 앞에서 또 리셋하므로, 평가의 에피소드 0은 어떤 시드의 추첨 1이고
  수집의 에피소드 0은 추첨 0이다. `--seed 1`이 두 경로에서 다른 큐브 자세를 가리킨다. 고치면
  지금까지 보고된 모든 평가 숫자가 움직이므로 여기에 기록만 하고, V6 오라클은 같은 *추첨*이 아니라
  같은 *시드*와 같은 *임계*를 고정한다.

**7. `tests/fixtures/hil/v1_small.eshil`은 동일하게 재생되며, 그것이 측정이다.** 이 로그는 서로 다른
값 112개를 가진 `observe_state` 레코드 120개를 담고 있어서, `observe_state`의 의미 변경이 그 안의
모든 결정을 움직일 수 있었다. 움직이지 않았다: HIL 테스트 리그의 플랜트는 완전한 위치 서보(`q :=
action.q`)라 측정 자세가 항상 마지막으로 내보낸 명령과 같고, 두 독해가 그곳에서 일치한다. 픽스처는
**재생성하지 않았고** `v1_fixture_still_replays_identically`는 그대로 통과한다 — V6가 엔벌로프의
산술이 아니라 기준을 바꿨다는, 얻을 수 있는 가장 강한 진술이다. `es-ros2`와
`es-runtime-embedded`는 코드 변경이 전혀 필요 없다: 에피소드가 없고, 둘 다 새 plane을 만들며,
둘 다 이미 검증 전에 관측한다.

**8. V3의 비공허성 게이트.** V3는 스위트가 `Success` 에피소드 하나와 `ActionSource::Clamped` 스텝
하나를 내지 못하면 패닉한다. V6 이후 클램프는 서보의 추종 오차에서 오지 않으므로 정책 자신의
청크에서 와야 한다: `acceleration_max · dt² = 0.008` rad보다 크게 벌어진 ACT 청크의 연속 행, 또는
소프트 관절 한계를 넘는 행. 둘 다 `nominal`을 포함한 **모든** 스위트에서 도달 가능하며 — V1c가
확대 엔벌로프 실행을 "거의 전부 `violation.position`"으로 측정했고 그것이 소프트 한계 단계이며
V6가 건드리지 않는다 — `torque_noise` / `backlash`는 팔을 자기 청크 아래에서 빼내어 정책을 그리로
미는, 여전히 가장 유망한 스위트다. 로컬에서는 정책 없이도
`a_command_outside_the_envelope_is_still_clamped`(`es-safety`),
`the_envelope_violation_rate_rises_when_the_policy_leaves_the_envelope`와
`a_tightened_envelope_clamps_and_a_widened_one_does_not`(`es-eval`)로 도달 가능성이 고정된다.
*훈련된 번들*이 여전히 그것을 건드리는지는 2단계 측정이고, 건드리지 않게 된다면 그것은 강제할
숫자가 아니라 보고할 발견이다.

**2단계, 측정된 대로** (오라클 서버, RTX 4090, `~/venvs/es-lerobot-cuda/bin/python`, 트리
`~/Projects/es-v6` — 이 브랜치의 `5a7e00e` 헤드에서 빌드된 아카이브 체크아웃, 그 안에서는 `git log`를
쓸 수 없다 — `~/artifacts/plan-v/v6/`, 2026-09-15. 재훈련 없음, 손잡이 변경 없음.)

**서버가 있어야만 도는 오라클 셋.** `expert_solves_the_pinned_seeds`(수집): 고정 시드 8개 전부
`Success`, 여덟 모두 큐브가 통의 3차원 내부에 들어감(`oracle-collect.log`).
`expert_passes_the_evaluation_harness`(평가): 8/8, 시드별 `envelope_violation_rate` 0.4815–0.5485,
에피소드 길이 330–354(`oracle-eval.log`) — 그 숫자가 무엇이고 이 패킷 자체의 게이트가 왜 그것
때문에 다시 고정되는지는 7.13절 발견 7에 있다.
`collection_and_evaluation_draw_the_same_scene_for_a_seed`: 설계대로 실제 결함을 잡아내며 실패했다.
다만 V6b가 그것을 쓸 때 잡으려던 결함은 아니었다 — 원소 6(큐브의 `y`)이 `es loop collect`에서는
`0.2550719976425171`, `es eval run`에서는 `0.255071989355131`로 돌아왔다. 같은 추첨을 두 번 다르게
반올림한 것이다(`oracle-parity.log`). 수정은 7.13절 발견 8에 있다.

**V1c의 20,000 스텝 번들을, 수정된 엔벌로프 아래에서 재측정:**

| 체크포인트 | V3 nominal | V2b nominal | V1c nominal | V6, 정직한 값 |
|---|---|---|---|---|
| 20,000 스텝 | 0.1250 (2/16) | 0.0000 (0/16) | 0.0625 (1/16) | **0.0000 (0/16)** |

그리고 여섯 스위트 스윕, 각 16 에피소드, 112개 셀 전부가 900 스텝 예산을 다 씀
(`nominal-20000/report.json`, `suite-20000/report.json`):

| 스위트 | success_rate | envelope_violation_rate | fallback | clamped | policy (14,400 중) |
|---|---|---|---|---|---|
| nominal | 0.0000 | 0.2106 | 220 | 2,813 | 11,367 |
| light_intensity | 0.0000 | 0.4790 | 333 | 6,565 | 7,502 |
| light_direction | 0.0000 | 0.3609 | 400 | 4,797 | 9,203 |
| observation_delay | 0.0000 | 0.2142 | 220 | 2,865 | 11,315 |
| torque_noise | 0.0000 | 0.1817 | 140 | 2,476 | 11,784 |
| backlash | 0.0000 | 0.2265 | 240 | 3,021 | 11,139 |

96 중 0. `fallback`은 `failure_mode_histogram` 자신의 버킷이고, `clamped`는
`envelope_violation_rate · 14,400 − fallback`이다. 모든 스위트에서 `fallback`은 히스토그램의
`violation.rate` 버킷과 정확히 같다 — 즉 이 실행에서 나온 모든 fallback tick은
`EnvelopeViolationRate` 워치독 하나뿐이고, `NonFinite`·`StaleObservation`·`InferenceDeadline`·
`HeartbeatLoss`·`SensorDropout`은 한 번도 뜨지 않았다. `clamped` 안에서는 `nominal` 자신의
히스토그램이 `violation.position 2,031`, `violation.acceleration 843`, `violation.velocity 439`다
— position이 다음 단계보다 2배 넘게 앞서고, 같은 순서(position > acceleration > velocity,
`violation.rate_limit`과 `violation.torque`는 여섯 스위트 어디에도 없음)가 여섯 스위트 모두에서
유지된다. §9.3의 `position_limit` 단계는 소프트 관절 마진이다(`position_soft_margin = 0.05` rad,
`deployment.toml`). 가장 비대칭적인 범위를 가지고 7.10절 자신의 진단에서 가장 큰 추종 오프셋을 보인
관절 — 그리퍼, 조인트 5, `[-0.1745, 1.7453]` rad, 중앙값 `|action − qpos| = 0.1436`, 팔은
`0.0009` — 이 유력한 후보다: 큐브를 잡으려 그리퍼를 닫는 정책은 자신의 가속도 예산을 넘기는 것보다
그 관절의 소프트 한계를 넘기는 쪽을 훨씬 자주 저지른다. 이것은 가설이며 가설이라고 밝혀 둔다. 이
실행이 관절별로 나눈 것은 아니다. 나중 패킷이 그 구간을 원한다면 `events.json`이 프레임 단위로
갖고 있다.

**V3의 비공허성 규칙이, 이번에는 정직하게 유지된다.** `Clamped ≥ 1` — `nominal`만으로도 2,813 클램프
스텝, 진짜 클램프 산술(§9.3 2–4단계)이고 V6가 없앤 추종 오차 아티팩트가 아니다. 이 규칙은
V3·V2b·V1c의 모든 숫자에서 공허했다(7.8–7.10절: 그 각각에서 `envelope_violation_rate`가
`1.0000`이었고 모든 스텝이 `Clamped`거나 `Fallback`이었으며 게이트가 가정한 것을 하나도 뜻하지
않았다); 지금은 공허하지 않다.

**2단계가 위해 있던 결론.** 하니스는 전문가를 통과시킨다 — 8/8, Deployment IR 자신의 워치독이 넉넉히
허용하는 엔벌로프 위반율에서. 비전 정책은 과제를 해내지 못한다 — nominal 16 중 0, 스위트 96 중 0,
그 클램프는 전문가 자신의 실행이 보이는 temporal-ensemble 지터(7.13절 발견 7)가 아니라 관절-한계
단계가 지배한다 — 이는 하니스가 원래 성공할 정책을 잘못 고치는 것이 아니라 정책이 팔이 닿을 수 없는
자세를 명령하고 있다는 뜻으로 읽힌다. 7.9절 이래 이 노트가 따라온 중단 규칙 사다리대로, 이 관측에
대한 다음 단계는 더 큰 모델도 더 긴 스케줄도 아니다: 7.14절 자신의 수가 이미 시작되어 있다 —
V7a의 특권 상태 포트, 그 2단계가 이 문서가 빚진 다음 표다.

### 7.13 구현 결과 (V6b): 평가가 수집과 같은 방식으로 청크를 실행하고, 같은 장면을 뽑는다

패킷 `docs/packets/M5/V6-envelope-semantics.md`의 V6b 절. 7.12절의 발견 6은 두 비대칭을 지목하고
남겨 두었다. 오케스트레이터는 2단계 서버 실행 전에 둘 다 닫기로 했다. 열어 둔 채로 재측정하면
다시 해야 하기 때문이다. **그리고 발견 6의 전반부는 틀렸으며, 진실은 그것이 주장한 것보다 나쁘다.**

**1. 정정. 수집도 열 tick마다 재계획하지 않는다 — 매 tick 재계획하고 *앙상블*한다. 평가는 둘 다 하지
않았다.** `BatchDomains::single_env()`는 추론 주기를 1로 선언하므로 `es loop collect`는 **매** 제어
tick마다 관측을 제출하고 청크를 받는다. 거기서 `action.execute_chunk`와 `rate.inference`를 의미 있게
만드는 것은 제출 주기가 아니라 `es_env::chunk_buffer::ChunkBuffer`다. 도착한 청크를 모두 저장하고,
`span`이 한 청크가 몇 tick을 구동할 수 있는지 정하며(`execute_chunk`, `TemporalEnsemble`이면 전체
호라이즌), `action_at`이 **겹치는 청크들을 혼합**한다 — `w_i = exp(-decay · i)`, 즉 ACT의 temporal
ensembling이고, 이는 데모의 `learning.toml`(`mode = "TemporalEnsemble"`, `weight_decay = 0.01`)과
`deployment.toml`(`[body.execution.temporal_ensemble] decay = 0.01`)이 둘 다 선언하는 바로 그것이다.

`es_eval::runner`에는 버퍼가 아예 없었다. 매 추론 결과를 새 `seq`로 plane에 넘겼으므로 plane의
커서가 매 tick 리셋되었고 **모든 청크의 0행만 실행되었다**. Deployment IR의
`action.execute_chunk = 10`은 그 경로에서 죽어 있었고 `execution`도 마찬가지였다 — 열다섯 개
청크의 지수 평균에 대고 훈련된 정책이 자신의 가공되지 않은 마지막 예측으로 평가된 것이다. 이는
"열 배 자주"보다 훨씬 큰 훈련/시험 격차이고, V3·V2b·V1c의 모든 평가 숫자가 그 아래에서 측정되었다.

**2. 수정은 규칙 둘이 아니라 공유 함수 하나다.** `DomainRunner::emit_actions`의 청크→plane 단계는
이제 `chunk_buffer.rs`의 `es_env::plane_chunk(buffer, feed, tick, mode)`이고, `Submitted` 기록은
공개 타입 `PlaneFeed`로 승격되었다. `es_eval::runner`가 같은 함수를 부른다: 모든 추론 결과를
**Deployment IR**로 만든 `ChunkBuffer`(`action.execute_chunk`, 그리고 `execution`이 함의하는 혼합
정책 — `TemporalEnsemble { decay }` → `ChunkBlendPolicy::TemporalEnsemble { weight_decay }`)에
밀어 넣고, 거기서 plane에 공급한다. `infer_chunk`는 `seq` 인자를 잃었다 — seq는 *버퍼가 받아들인
결과*당 한 번 찍히며, 그것이 `SafetyPlane::accept`가 신선도를 판정하는 기준이다(§8.6).
`PlaneFeed::end_episode`가 양쪽의 에피소드 경계이고 `DomainRunner::reset_env`도 이제 그것을 부른다.
Deployment IR이 결정하고, 플래그는 없다.

`es loop collect`는 비트 단위로 변하지 않았다 — 추출은 이동이고, `es-env`와 `es-data`의 스위트는
V1c의 `a_second_episode_repeats_the_first_exactly`와
`emit_actions_writes_the_planes_answer_and_records_the_command`를 포함해 그대로 통과한다. 그것이
요점이다: 공유 함수는 수집기의 것이고, 평가가 그리로 왔다.

**3. 남은 차이 하나, 이름을 붙여서.** 수집은 정책 계약의 `expected_latency_ms`(20 ms 제어 주기에
대해 15 ms = 1 tick)를 모델링하므로 첫 청크가 tick 1에 적용되고 tick 0은 청크 언더런이다 — 7.10절의
"에피소드당 하나씩 50번의 폴백"이 그것이다. 평가는 결과가 계산된 tick에 적용한다. Deployment IR에는
지연 필드가 없고(`deadlines.inference_budget`은 워치독 경계이지 스케줄이 아니다),
`Evaluation::run`은 Learning IR을 받지 않으며, 그것을 위해 필드나 플래그를 만드는 것은 이 패킷의
결정도 IR의 현재 형태도 아니다. 영향은 900 스텝 에피소드당 한 프레임이다. 여기 적어 두고, 다른
어디서도 아닌 척하지 않는다.

**4. 이제 `--seed S`는 하나의 장면을 가리킨다.** `Env::new`가 한 번 리셋하고 — `(seed, env,
episode)`의 추첨 0(§6.3) — 모든 에피소드는 정확히 한 번의 리셋으로 끝난다(종료 조건이면
`Env::step` 자신의 리셋, 스텝 예산이 다하면 명시적 리셋). `Collector::run`은 그것에 의존하고 루프
앞에서 리셋하지 않는다. `run_episode`는 맨 앞에서 **또** 리셋했으므로 평가의 에피소드 `i`는 추첨
`2i + 1`에서, 수집의 에피소드 `i`는 추첨 `i`에서 돌았다: `--seed 1`이 두 경로에서 큐브를 다른 곳에
놓았고, 둘 다 유효한 자세라 조용히 그랬다. 그 리셋을 없앴다. 다른 것은 움직이지 않았다: 예산이 다한
에피소드를 닫는 아래쪽 리셋도, `plan.reset()`도 그대로이고, 셀마다 새 `Env`를 받는 것도 그대로다.

**5. 오라클.** `episode_zero_runs_on_the_first_randomization_draw`(`es-eval`, 로컬, 가짜 백엔드)는
`Env`에서 직접 기준값을 만들고 러너가 처음 제공하는 상태가 `Env::new` 자신의 추첨과 비트 단위로
같음을 단언한다 — **두 번째 리셋을 되살리면 실패함을 측정했다**(`0.5904` 대 `0.8683`).
`the_runner_feeds_the_plane_through_the_collectors_chunk_buffer`(`es-eval`)와
`the_collector_resets_once_per_episode_and_never_before_the_first_observation`(`es-data`)이 두
호출 규율 스캔이며, V6가 시드에 대해 추가한 것과 같은 방식이다 — 추가 리셋이 반대쪽으로 옮겨 갈 수
없게 한다. `collection_and_evaluation_draw_the_same_scene_for_a_seed`(`crates/es/tests/cli.rs`)가
오케스트레이터가 요청한 교차 경로 오라클이다: `es_data::Collector`와 `es_eval::Evaluation`을 데모
씬에서 시드 1로 돌려, 각자가 자기 정책에 처음 건네는 `qpos ‖ qvel`을 원시 `f64` 비트로 비교한다.
SO-101 씬은 `MuJoCoCpuBackend` 말고 구동할 것이 없으므로 두 전문가 오라클 옆의 **서버 오라클**로
이름을 남긴다.

**6. 이것이 무효화하는 것.** 7.8·7.9·7.10·7.11절의 모든 평가 숫자 — V3·V2b·V1c의 성공률,
엔벌로프 위반율, `ActionSource` 히스토그램, 에피소드 길이, 그리고 두 번의 확대 엔벌로프 진단. 전부
청크 버퍼 없이, temporal ensembling 없이, `execute_chunk`가 죽은 채로, 엔벌로프가 추종 오차를
제한한 채로(V6), 큐브가 시드 자신의 것에서 한 추첨 떨어진 채로(V6b) 측정되었다. **그 표들에서
넘어오는 것은 없다.** 수집 숫자는 넘어온다: 시연, 손실 곡선, `observation_hash`,
`lowering_hash`, `dataset_schema_hash`, 훈련된 체크포인트는 모두 수집 경로와 훈련 경로의
산물이고 둘 다 움직이지 않았다. 2단계는 V1c의 커밋된 20,000 스텝 번들을 재측정한다 — 재훈련 없음,
손잡이 변경 없음 — 그리고 그것이 플랜 V가 만든 첫 번째, 하니스가 아니라 정책을 재는 평가 표다.

**7. 2단계는 발견 1이 이름 붙인 메커니즘을 측정했고, 그것이 하니스 자신의 게이트를 다시 고정하게
만들었다.** `expert_passes_the_evaluation_harness`는 1단계 이래 `worst_violation < 0.02`를
단언했다. plane이 전문가의 페이싱된 램프를 전문가가 낸 그대로 tick마다 읽는다는 가정에서였다.
이 절의 구조상 그렇지 않다: 평가 경로는 이제 수집과 같은 `ChunkBuffer` temporal-ensemble
혼합을 거친다 — 16행 호라이즌에 걸쳐 `decay = 0.01` — 그래서 plane이 매 tick 재는 것은
`action_at`이 span 안의 모든 청크를 섞은 결과이지, 어느 한 청크 자신의 페이싱된 행이 아니다.
측정값: 고정 시드 8개에 걸쳐 `envelope_violation_rate` 0.4815–0.5485(`oracle-eval.log`).
기여하는 각 청크는 `velocity_max·dt`와 `acceleration_max·dt²`의 90%로 페이싱되어 있으므로(7.12절
발견 4), 그 페이싱 규칙이 남기는 약 10%의 여유가 정확히, 한 제어 tick 떨어져 계산된 청크들 사이의
혼합 지터가 속도·가속도 클램프 단계를 매 tick 건드리는 데 필요한 여유다 — 범위 밖 위치를 단 한 번도
요구하지 않는 궤적에서다. 수집 경로는 이미 자신의 숫자로 같은 모양을 보였다: 7.10절의 V1c 시연은
18,263프레임 중 11,631이 `Clamped`이거나 `Fallback`, **0.6368**이다 — 같은 현상을 이 절이
재측정하지 않는 경로에서 더 먼저 측정한 것이라서 여기 인용한다.

그러니 `< 0.02` 게이트는 이 패킷이 골라서 느슨하게 한 한계가 아니었다. 1단계가 아직 버퍼를 거치지
않은 경로를 1단계 때 읽은 값이었고, 위 발견 1이 착지한 순간 참이 아니게 되었다. `crates/es/tests/
cli.rs`는 이제 여전히 참인 한계를 문서에서 직접 읽는다. 테스트에 숫자를 타이핑해 넣는 대신 —
Deployment IR 자신의 `EnvelopeViolationRate` 워치독(`max_frac = 0.9`,
`tests/fixtures/visible-learning/deployment.toml`)이다. 스펙 9.4가 실제로 작동시키는 숫자가
그것이기 때문이다: 그 위로 가면 plane이 폴백을 걸어 잠그고 전문가는 과제가 아니라
`hold_position`을 몰게 된다. 최악값 0.5485는 거기 근처에도 가지 않고,
`expert_solves_the_pinned_seeds`의 8/8은 두 경로를 맞대보는 골든으로 남는다. `< 0.02`라는 숫자는
단언문만 조용히 넓히는 대신, 여기에 글로 적어서 은퇴시킨다.

**8. 교차 경로 오라클은 무관한 반올림 하나를 더 측정했고, 두 경로가 공유하는 폭에서 고쳤다.**
`collection_and_evaluation_draw_the_same_scene_for_a_seed`는 두 경로가 처음 건네는 `qpos ‖ qvel`
행을 원시 `f64` 비트로 비교했고 실패했다: 원소 6(큐브의 `y`)이 `es_data::Collector`의 정책 입력에서는
`0.2550719976425171`, `es_eval::Evaluation`의 프레임 소스에서는 `0.255071989355131`로 읽혔다. 같은
추첨을 두 번 반올림한 것이다. 수집기 자신의 정책 입력은 `es_env::domains::state_row`의 `f32`
텐서다 — plan-free 경로는 `f64` 관측을 백엔드 너머로 나르지 않는다 — 그리고 시연의
`observation.state` 컬럼도 같은 `f32`로 쓰인다. 반면 평가 러너의 프레임 소스는 백엔드의 `f64`
`StateView`를 직접 건네받아, 수집기의 정책이 실제로 보는 것보다 변환을 한 단계 덜 거친다. 둘을
`f64`로 비교한 것은 `f32`로 반올림된 수와 반올림되지 않은 수를 비교한 것이지, 한 수를 다르게 읽는
두 경로를 비교한 것이 아니었다. 오라클은 이제 비트 비교 전에 양쪽을 `f32`로 캐스팅한다 — 두 경로가
실제로 같은 객체인 폭이고, 이 프로젝트가 지금까지 기록한 모든 시연이 쓰인 폭이다.

### 7.14 구현 결과 (V7a 1단계): 큐브의 포즈가 상태 포트로 들어온다, 그리고 그것이 증명해도 되는 것

패킷 `docs/packets/M5/V7a-privileged-state-policy.ko.md`. 7.13절이 이 데모가 내놓았던 모든 평가
수치를 무효로 만들었으므로, 다음 수치는 가져갈 가치가 있어야 한다. 비전 데모는 두 질문을 한꺼번에
묻는다 — 이 그래프가 과제를 해낼 수 있는가, 그리고 스크래치 ResNet18이 96×96 픽셀에서 25 mm
큐브를 찾아낼 수 있는가 — 그리고 V7a는 두 번째를 제거한 채 첫 번째에 답함으로써 둘을 가른다:
**큐브의 포즈를 상태 포트에 넣는다.**

영상의 1단계, "상태 정책"이다. 2단계, 더 높은 해상도와 더 많은 시연으로 하는 비전은 별도 패킷이다.

**중단 규칙, 측정 전에 적어 둔다.** 상태 정책이 시드 101–116의 nominal에서 `success_rate ≥ 0.5`에
도달하지 못하면, 다음 단계는 더 큰 모델도, 더 긴 스케줄도, 더 많은 시연도 아니다. 물리, 접촉 모델,
전문가 궤적이다. 큐브의 정확한 포즈와 팔의 정확한 관절각, 7초짜리 스크립트 동작의 시연 50개를 받은
정책에게 부족한 정보는 남아 있지 않다.

**1. 채널, 그리고 그 형태를 결정한 세 제약.** `sim_cube_pose`는 두 번째 `ObservationSpec` 채널이며
`ObsSource::JointState { body: <`cube_free` 조인트의 id>, dof: 7 }`, 타입 `f32[7]`이다. `es-ir`에는
아무것도 추가되지 않았다.

* `es_eval::runner::input_sources`는 플랜 입력의 소스 id를 Task IR 채널의 `dof`로 떨어지기
  **전에** `ModelInfo.qpos`에 대해 해석하고, `ModelInfo.qpos`는 **조인트** id로 키가 잡힌다.
  그래서 `cube_free` 조인트를 가리키는 채널은 `Capture::Qpos(6..13)` — 그 조인트 자신의 일곱 값 —
  을 받고, 로봇의 `base` *바디*를 가리키는 `joint_state`는 계속 `Capture::Joints(6)`, "행의 앞
  여섯 개"로 떨어진다. 두 분기 모두 이미 있었고 이 패킷은 어느 쪽도 추가하지 않는다.
* **Task IR-D는 free 조인트의 `qpos`를 낼 수 없다.** `GetJointState`의 출력 타입은
  `f32[joints.len()]`, 조인트 *이름* 하나당 스칼라 하나 — 5.4절이 기록한 천장이고, 성공 술어가
  큐브의 `x`만 보는 이유다. 그래서 그래프는 값을 `GetBodyPose{cube, World}` → `pos[3]`과
  `quat[4]`의 `Concat{axis 0}` → `ObservationSpec`으로 보여준다. 기존 노드 집합으로 표현되고
  간선마다 타입이 맞는다. **채널**이 바디가 아니라 조인트를 가리키는 것은 채널이 *읽기*를
  결정하기 때문이다: 조인트를 통하면 `qpos`로 정확하고, 바디를 통하면 `xpos ‖ xquat`이 되는데
  기록된 데이터셋에는 그것이 없다.
* **`NormalizeStats::Range`는 포트당 `(lo, hi)` 하나다.** `joint_state`를 그냥 6에서 13으로 넓히지
  않은 두 번째 이유다. 상태 분기의 `±1`에서 큐브의 0.06 m 추출 범위는 출력 범위의 0.03을
  차지하지만, `±0.3` — 추출 범위(`x ∈ [0.21, 0.27]`)와 통 내부(`x ∈ [0.09, 0.19]`,
  `y ∈ [-0.15, -0.05]`)를 모두 품는 팔의 도달 범위 — 에서는 0.10을 차지한다. 쿼터니언의 네 값은 그
  범위에서 `[0, 1]`을 벗어난다(`w = 1`이 2.17로). `Normalize`는 아핀이지 클램프가 아니고, 평평하게
  놓인 정육면체는 거기에 신호를 싣지 않으므로, 우회하지 않고 적어 둔다.

**2. 특권은 이름으로 표시한다. 표시할 필드가 없기 때문이다.** `ObsChannel`은 `source`와 `ty`만
갖는다 — 나머지는 Observation IR의 것이라고 §7.4가 명시한다 — 므로 `provenance`나 `sim_only` 같은
필드를 만들어내지 않았다. 표시는 `sim_` 접두사이며, Task IR 채널, Observation IR 출력 포트,
Learning IR 입력, 정책 계약, `contract.json`이 그대로 이어받으므로 체인의 모든 구간에서 보인다.
`sim.cube_pose`가 아니라 `sim_cube_pose`인 것은, 포트 이름이 `forward(**inputs)`의 키로 파이썬에
도달하고 LeRobot 피처 이름에서 점이 네임스페이스 구분자이기 때문이다.

더 넓은 `joint_state`가 아니라 *두 번째* 포트로 두는 것이, 비전 패킷이 `joint_state`의 해시를
움직이지 않고 이 포트만 떼어낼 수 있게 한다. 팔의 여섯 각은 진짜 엔코더에서 나오고, 큐브의 일곱
숫자는 로봇이 지니지 않은 것에서 나온다. 둘 다 float이라는 이유로 한 텐서에 섞지 않는다.

**3. 데이터셋에는 아무것도 필요 없었고, 그것이 발견이었다.** `es_data::collect::to_lerobot`는
`observation.state`를 env 0의 `qpos` 전체 뒤에 `qvel` 전체를 붙여 쓴다 — 이 씬에서
`13 + 12 = 25` — 므로 큐브의 포즈는 V1 이후 모든 시연에 들어 있었다. 컬럼은 추가되지 않았고, v2.1
라이터와 v3.0 익스포트는 그대로이며, **`dataset_schema_hash`는 움직이지 않는다**. 움직인 것은
Observation IR이 그 행의 어느 구간을 읽는가다. 이제 `docs/api-notes/lerobot-dataset.ko.md`가 그
레이아웃을 기록한다. 다음 세 문장이 거기에 의존하기 때문이다.

**4. bake는 오프셋을 배워야 했고, 분기보다 거절이 더 중요하다.** `qpos`의 `IndexRange`는 `capture`가
`StateView::qpos_of(0)`을 인덱싱하는 것과 같은 두 경계로 기록된 행을 인덱싱한다 — 이 행은 상태의
재인코딩이 아니라 상태에 `qvel`을 덧붙인 것이다. 그래서 `ObservationBake::frame`은
`Capture::Qpos(r) => row[r]`을 얻고, `ObservationBake::new`는 V2b가 거부했던
`Option<&ModelInfo>`를 얻는다. `qpos` 범위는 실제로 돌았던 모델에서 와야 하기 때문이다.
`es dataset bake`는 먼저 모델 없이 해석하고, 거절당했을 때에만 `MuJoCoCpuBackend`로 씬을 연다.

제대로 해야 했던 부분은 거절 쪽이다. 모델이 없으면 `Capture::Joints(dof)`는 "행의 앞 `dof`개"를
뜻하고, **선두일 수 있는 채널은 하나뿐이다**. 두 번째 `JointState` 채널이 선언되어 있는데 모델이
없으면 어느 쪽이 선두인지는 문서에 없다 — 모델에 있다 — 므로 `input_sources`는 포트 이름을 대며
거절한다. 그것이 없었다면 데모는 `row[..7]`, 즉 팔의 여섯 각과 큐브의 `x`를 특권 포트에 bake해
넣고 조용히 학습했을 것이다. 손실은 떨어졌을 것이고 그 수치는 아무 의미도 없었을 것이다. 7.9절과
같은 부류의 결함이며, 표를 만들어낸 뒤가 아니라 만들어내기 전에 잡았다.

**5. 로컬에서 고정되는 것.** `a_baked_frame_is_bit_identical_to_what_capture_serves`는 이제 상태
포트 **둘**을 돌린다: 앞-`dof` 읽기를 통과하는 `j0`, 그리고 `qpos` 범위가 **1**에서 시작하는 `j1`을
`Capture::Qpos`로. 모든 프레임의 모든 출력 텐서가 정책이 받은 것과 바이트 단위로 같고 허용 오차는
없으며, 테스트는 `q[0] ≠ q[1]`을 단언하므로 한 포트에 다른 포트의 값을 내주고 통과할 수 없다.
`a_bake_refuses_two_state_channels_without_a_model`은 그 거절과, 모델을 넘기면 같은 쌍이
해석된다는 것을 고정한다.

`es dataset bake` CLI 테스트 둘은 이제 `MuJoCoCpuBackend`가 필요하고 없으면 이유를 출력하며
건너뛴다. 데모의 두 번째 채널이 씬의 `qpos` 범위로 해석되고, 그 레이아웃의 유일한 권위는 씬을
적재하는 백엔드이기 때문이다. 여기서 MJCF로부터 유도하는 것은 7.9절이 제거한 부류의 두 번째
구현이 된다. 그 픽스처의 행 폭도 `es loop collect`가 한 번도 쓴 적 없는 여섯 값 대신 실제
`qpos ‖ qvel` 폭(25)이 되었다.

**6. 움직인 해시.** 모두 파생값이며 선언된 생성기
(`cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents`)에서 나온다:

| 슬롯 | 이전 (V6b) | 이후 (V7a) |
|---|---|---|
| `task_hash` | `aec2aea9…1bf1` | `6cf826c1…6b7b` |
| `observation_hash` | `f4a50730…f6e0` | `6c18f455…2daf` |
| `learning_hash` | `82faf8c7…1b54` | `5dac0a46…46f0` |
| `policy_hash` | `01583940…88cd` | `c94c2732…4a07` |
| `evaluation_hash` | `5d70c21c…9ff7` | `0259fd44…f041e` |
| `lowering_hash` | `956abb67…ec6d` | `fdd68ec4…0719` |
| `deployment_hash` | `3b2ad568…6db1` | 변동 없음 |
| `compiler_hash` | `f2a02e84…70d6` | 변동 없음 |
| `dataset_schema_hash` | — | 변동 없음 |

`deployment_hash`가 표에 있는 것은 움직이지 않았음을 보이기 위해서다. 엔벌로프는 V6의 것이고
V7a는 `es-safety`가 금지되어 있다.

**아직 측정되지 않았다.** 위의 모든 것은 로컬이다. 서버 실행 — 기존 50 에피소드 세트를 넓어진
Observation IR로 bake, 로워링, V2b의 노브 그대로 `--resident-gpu`로 20,000 스텝 학습, 팩, 세
체크포인트에 대해 시드 101–116 스윕, 그다음 20,000 스텝 번들에 대해 여섯 스위트, 그다음 4×4
모자이크 — 이 2단계이고, 여기의 표들은 그것으로 채워진다. 그 실행의 첫 명령은
`es dataset bake`의 모델 적재를 처음으로 실제로 거치는 명령이기도 하다. 이 기계에는 `mujoco`가
없어 로컬 커버리지가 없다.

**측정됨 (V7a 2단계), 그리고 그것이 7.15절이다.** 위의 모든 해시는 예측한 값 그대로 돌아왔고,
bake의 모델 로드는 첫 명령에서 동작했으며, 상태 정책은 20,000 step에서 nominal
`success_rate 0.0000`이다. **중단 규칙이 발동한다.**

### 7.15 구현 결과 (V7a 2단계): 상태 정책을 재고, 중단 규칙이 발동한다

패킷 `docs/packets/M5/V7a-privileged-state-policy.md`의 "server, phase 2" 절. 오라클 서버(RTX 4090,
16코어)에서 커밋 `481e4d4`의 트리를 `~/Projects/es-v7a`에 풀어 실행했고, `mujoco`는
`ES_PYTHON=~/venvs/es/bin/python`, 학습은 `~/venvs/es-lerobot-cuda`를 썼다. 아래 모든 것은
`~/artifacts/plan-v/v7a/` 아래에 있고, 작은 파일은 요청자의 `target/plan-v/v7a/`로 복사했다.

**답부터.** 자기 관측 포트에 큐브의 정확한 자세를, 팔의 정확한 관절각을, 같은 50편의 시연을, 같은
20,000 옵티마이저 step을 받은 정책이 held-out nominal 시드 열여섯 개에서 `success_rate 0.0000` —
900 step 타임아웃 열여섯 번 — 을 기록했고, 같은 셀의 샤딩된 판독에서는 `0.0625`, 한 에피소드였다.
6스위트 전체 실행에서는 **96 에피소드 중 4 성공**이다. **중단 규칙이 발동한다.** 7.14절은 측정
전에 그것이 무엇을 뜻하는지 적어 두었고, 지금도 같은 뜻이다: 다음 용의자는 물리, 접촉 모델,
전문가의 궤적이지 모델 크기·학습 길이·시연 수가 아니다.

**1. bake가 돌았고, 모델이 뒷받침하는 2채널 경로의 첫 실제 실행이다.** 이 저장소의 CI 머신에는
`mujoco`가 없어 성공적인 모델 로드에 닿는 로컬 테스트가 없으므로, 서버 실행의 첫 명령이 곧 그
커버리지다. 모델 없이 해석하다 거부됐고, `MuJoCoCpuBackend`로
`tests/fixtures/mjcf/so101_pick_place.xml`을 로드했으며, 텐서 네 개를 썼다:

```
wrote: /home/LJM/artifacts/plan-v/v7a/baked
episodes: 50   frames: 18263
  action                       [6]
  joint_state                  [6]
  rgb_overhead                 [3, 96, 96]
  sim_cube_pose                [7]
observation_hash: 6c18f4552064d24b16cffa770cfc71941566f2b0443fe88569d439f747c72daf
```

50 에피소드, 18,263 프레임, 1.9 GB, 6초. 같은 V1 데이터셋과 같은 V1c 타일
(`~/artifacts/plan-v/v1c/ds-train`, `frames-train`) — 아무것도 재수집하지 않았으므로
`dataset_schema_hash`는 움직이지 않았고, V6의 비전 실행과의 비교는 변수 하나만 움직인다.

**2. 패킷이 예측한 모든 해시, 측정됨.** 커밋된 픽스처 다섯 개에 `es ir check`를 돌리면
`task 6cf826c1…6b7b`, `observation 6c18f455…2daf`, `learning 5dac0a46…46f0`,
`policy c94c2732…4a07`, `deployment 3b2ad568…6db1`(**불변**, 금지 조항대로),
`evaluation 0259fd44…f041e`, `compiler f2a02e84…70d6`가 찍혔다 — 일곱 개 모두 acceptance 표의
값 그대로다. `es policy lower`는 `["joint_state", "rgb_overhead", "sim_cube_pose"]` 위에서
`lowering_hash fdd68ec4…0719`와 가중치 키 14개(정확 12, 접두 2)를 찍었고, V2b의 10개와 비교하면
세 번째 인코더가 차이의 전부다. 패킹된 번들 세 개:

| 체크포인트 | 텐서 | `weights_hash` | `policy_hash` |
|---|---|---|---|
| 1,000 | 146 | `4ebf61fb…791c` | `8f62acfa…ea1e` |
| 5,000 | 146 | `017ac45f…a72b` | `0e6d786f…f71ad` |
| 20,000 | 146 | `ac03ceec…a581` | `301647f4…a4df` |

**3. 손실은 떨어졌고, 비전의 손실과 정확히 같은 만큼 떨어졌다.** `--batch 8 --lr 1e-4 --seed 0
--resident-gpu`로 `cuda`에서 20,000 step, 상주 1,927.5 MiB, 10분 35초. 실행 자체의 요약은
`initial_loss 0.06752 -> final_loss 0.01713`(처음과 마지막 10분의 1의 평균)이고, 같은 knob의
V1c 비전 실행과 나란히, 각 체크포인트에서 끝나는 100 step의 평균:

| step | V7a (상태 + 비전) | V1c (비전만) |
|---|---|---|
| 처음 100 | 0.2057 | 0.2053 |
| 1,000 | 0.0557 | 0.0560 |
| 5,000 | 0.0308 | 0.0319 |
| 20,000 | 0.0175 | 0.0176 |
| 보고된 `initial -> final` | 0.06752 -> 0.01713 | 0.06689 -> 0.01799 |

**이것이 첫 번째 발견이고, 성공률 표보다 크다.** 네트워크에 답을 주었더니 학습 손실이 약 0.5 %
움직였다. 목표물의 정확한 좌표가 들어 있는 포트를 알아채지 못하는 회귀 손실은 태스크를 결정하는
행동의 부분을 재고 있지 않다: 7초짜리 스크립트 궤적에 대한 L1 거리는 길고 쉽고 큐브와 무관한
구간이 지배하고, 큐브의 위치가 실제로 동작을 고르는 몇 프레임은 그 안에서 반올림 오차다. 손실
곡선은 정책이 태스크를 배우고 있다는 증거였던 적이 없고, 이제 그것이 얼마나 작은 증거였는지에
대한 수가 생겼다.

**4. nominal, 시드 101–116, 16 에피소드, 체크포인트당 셀 하나.** V6의 정직한 비전 수치(7.13절의
하네스, V1c의 20,000 step 번들, `~/artifacts/plan-v/v6/`)와 나란히:

| 번들 | `success_rate` | `envelope_violation_rate` | `episode_length` | `Policy` / `Clamped` / `Fallback` |
|---|---|---|---|---|
| V7a 상태, 1,000 | 0.0000 | 0.9853 | 900.0 | 212 / 12,908 / 1,280 |
| V7a 상태, 5,000 | 0.0625 | 0.2523 | 855.0 | 10,228 / 3,212 / 240 |
| V7a 상태, 20,000 | **0.0000** | 0.1460 | 900.0 | 12,297 / 1,986 / 117 |
| V7a 상태, 20,000, 샤딩 | 0.0625 | 0.1628 | 854.6 | 11,448 / 2,106 / 120 |
| V6 비전, 20,000 | 0.0000 | 0.2106 | 900.0 | 11,367 / 2,813 / 220 |

`ActionSource` 수는 셀의 전체 프레임 예산(모든 에피소드가 타임아웃이면 14,400) 위의 값이다.
실패 모드 히스토그램, 같은 순서:

| 번들 | `violation.position` | `violation.acceleration` | `violation.velocity` | `violation.rate` | `fallback` | `timeout` | `success` |
|---|---|---|---|---|---|---|---|
| V7a 상태, 1,000 | 11,772 | 1,387 | 358 | 1,280 | 1,280 | 16 | 0 |
| V7a 상태, 5,000 | 2,629 | 585 | 229 | 240 | 240 | 15 | 1 |
| V7a 상태, 20,000 | 1,334 | 668 | 185 | 117 | 117 | 16 | 0 |
| V6 비전, 20,000 | 2,031 | 843 | 439 | 220 | 220 | 16 | 0 |

그러니까 상태 정책은 태스크가 아닌 모든 축에서 비전 정책보다 *얌전하다*: 자기 step의 85.4 %를
스스로 구동하고(비전은 78.9 %), clamp를 3분의 1 덜 맞고, fallback은 절반으로 줄고, 소프트 위치
한계는 3분의 1만큼만 친다. 그래도 큐브를 통에 넣는 일은 결코 없다. 7.10절의 결론 — "싸우지
않는 엔벌로프를 주면 ACT는 step의 94.5 %를 스스로 구동하고 그래도 태스크를 못 한다" — 은
지각 문제를 제거한 뒤에도 살아남는다.

**5. 20,000 step 번들의 전체 스위트**, `--jobs 6`, 96 에피소드, 5분 50초 대 V1c의 순차 27분.
83,429 프레임 위의 `ActionSource`: `Policy` 71,447, `Clamped` 11,449, `Fallback` 533.

| 스위트 | V7a `success_rate` | V7a `envelope_violation_rate` | V7a `episode_length` | V6 비전 `success_rate` |
|---|---|---|---|---|
| nominal | 0.0625 | 0.1628 | 854.6 | 0.0000 |
| light_intensity | 0.0000 | 0.1647 | 900.0 | 0.0000 |
| light_direction | 0.0000 | 0.1462 | 900.0 | 0.0000 |
| observation_delay | 0.0625 | 0.1080 | 853.9 | 0.0000 |
| torque_noise | 0.1250 | 0.1682 | 805.8 | 0.0000 |
| backlash | 0.0000 | 0.1137 | 900.0 | 0.0000 |
| **합계** | **4 / 96 = 0.0417** | | | **0 / 96** |

`evaluation.toml`의 acceptance는 nominal 스위트 하나에 걸려 있고 `success_rate >= 0.5`를 요구한다.
실행 전에도 그것을 요구했고 지금도 요구한다; 모든 리포트에서 `passed`는 `false`이고 어떤 문턱도
손대지 않았다. nominal보다 *높게* 나온 두 교란 스위트 — `torque_noise` 2/16, `observation_delay`
1/16 — 는 정책이 명령하지 않은 움직임을 더하는 둘이고, n=16에서 그것은 잡음이며, 결과로 읽지
않고 잡음으로 기록한다.

**6. 런타임이 torch일 때 `--jobs N`은 `--jobs 1`과 바이트 동일하지 않고, 이 실행이 그것을
쟀다.** 단독 nominal 셀은 0/16이고 같은 셀이 `--jobs 6` 스위트 안에서는 1/16이다 — 같은 번들,
같은 시드, 같은 문서 본문. 단독 셀을
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=TORCH_NUM_THREADS=2`로 다시 돌리면 샤딩된
판독이 **정확히** 재현된다: `success_rate 0.0625`, `envelope_violation_rate 0.16279069767441862`,
`episode_length 854.625`, 그리고 필드 하나하나 같은 히스토그램. 원인은 `56fa49b`로 병합된 샤드별
스레드 상한이다: 16코어 박스에서 `--jobs 6`의 워커는 `16/6 = 2` 스레드로 제한되고, `--jobs 1`
실행은 열여섯 개를 다 쓰며, torch의 리덕션 순서는 스레드 수의 함수다. 다른 합은 다른 액션이고
다른 액션은 다른 궤적이다.

여기서 `es-eval`에 잘못된 것은 없다: V5 오라클이 고정하는 spec 10.4의 주장 — 부모가 분할하고,
워커는 아무것도 판정하지 않으며, 병합이 셀 순서를 복원한다 — 은 성립하고, `N = 4`에서 바이트
동일성을 고정하는 오라클은 결정적 가짜 런타임을 쓰기 때문에 정확히 이것을 볼 수 없다. 잘못된
것은 `es eval run --help`의 한정 없는 "같은 시드에서 산출물은 --jobs 1과 바이트 동일하다"이다.
그것은 하네스에 대해서는 참이고, 하네스와 스레드 수에 민감한 `PolicyRuntime`의 합성에 대해서는
참이 아니다. 열린 질문 16이고, 여기서 판정을 움직이지는 않으며(0/16과 1/16 모두 0.5에서 한참
아래다), 이 쌍의 정직한 읽기는 열여섯 중 한 에피소드가 해결의 경계에 걸쳐 있고 나머지 열다섯은
근처에도 없다는 것이다.

**7. 영상.** 각 체크포인트의 nominal 셀 열여섯 개에 `es video mosaic --grid 4x4`를 걸면
384x392(96x96 타일 열여섯 개 + 8픽셀 라벨 띠)의 900 프레임이 나오고, `encode_video.py --fps 50`이
제어 주기 그대로 18초짜리 mp4를 만든다. `cv2`는 `~/venvs/es`가 아니라 `~/venvs/es-lerobot`에
있는데, 패킷의 명령줄이 잘못된 쪽을 적고 있으므로 적어 둘 가치가 있다.

| 체크포인트 | `mp4v` | H.264 |
|---|---|---|
| 1,000 | `demo-1000.mp4`, 5.31 MB | `demo-1000-h264.mp4`, 0.45 MB |
| 5,000 | `demo-5000.mp4`, 6.46 MB | `demo-5000-h264.mp4`, 0.91 MB |
| 20,000 | `demo-20000.mp4`, 8.18 MB | `demo-20000-h264.mp4`, 1.14 MB |

H.264 사본은 같은 원본 모자이크 프레임에 `ffmpeg 7.0.2`를 건 것이지 `mp4v` 파일의 트랜스코드가
아니다. mp4는 해시 체인에 없고(9절), 프레임은 있다.

**8. 판정, 그리고 사다리를 어디로 보내는가.** 중단 규칙은 적힌 대로 적용된다. 데모에는 더 큰
인코더가 고칠 지각 문제가 없고, 더 긴 스케줄이 고칠 옵티마이저 문제도 없다: 손실은 수렴했고,
엔벌로프는 정책과 싸우지 않으며, step의 85 %는 정책 자신의 것이고, 큐브는 시작한 자리에 그대로
있다. 남은 것은 7.14절이 이름 붙인 것 — 물리, 접촉 모델, 또는 시연 자체다. 가장 싼 것부터 먼저
열어 볼 세 가지, 그리고 그중 어느 것도 모델이 아니다:

* **시연은 잡는가?** V1의 성공 술어는 큐브의 `x`가 통 안으로 넘어가는 것이고(5.4절),
  `es loop collect`는 50/50을 보고한다. 스크립트 전문가는 들어 올리는 대신 밀어서 그 술어를 만족
  시킬 수 있고, 미는 궤적을 재현하다 접촉을 1 mm 놓치는 정책은 아무것도 얻지 못한다. 오라클은
  그리퍼의 접촉력을 기록하며 기록된 에피소드 하나를 재생하는 것이고, 아무도 하지 않은 측정이다.
* **그리퍼는 무엇이든 물기는 하는가?** MuJoCo 기본 접촉 파라미터와 `forcerange ±2.94 N·m` 아래
  25 mm 상자에 닿는 SO-101의 조는 씬이 한 번도 따로 받아 본 적 없는 두 물체 질문이다.
* **실행된 청크가 시연이 담고 있는 궤적인가?** 7.13절은 평가가 수집과 같은 방식으로 청크를
  실행하게 했고, 앙상블은 그중 열다섯을 섞는다; `decay = 0.01` 지수 평균을 통과한 7초 동작은
  매끄러워진 동작이고, 그 매끄러움이 파지 구간을 살려 두는지는 시연 하나를 `ChunkBuffer`에
  재생해 보면 잴 수 있다.

영상의 2단계 — 더 높은 해상도와 더 많은 시연의 비전 — 는 다음 패킷이 **아니다**. V7a는 그것을
결정하려고 존재했고, 결정했다: 답을 쥔 정책이 못 하는 일을 비전 정책이 하리라 기대할 수 없다.

### 7.16 구현 결과 (V8): 외부 ACT를 우리 런타임으로

패킷 `docs/packets/M5/V8-external-act.ko.md`. 이 데모가 측정해 온 정책은 전부 우리 것이었다.
CVAE도 DETR 디코더도 없고 ResNet18을 처음부터 학습시키는 ACT *모양*의 Learning IR 그래프 —
스펙 §8.3의 `TemporalEncoder { Transformer }`가 폭 하나만 싣기 때문이다. 그것은 0/16이고 V7a의
특권 변형은 0–1/16이다. 그래서 숫자로는 가를 수 없는 가설이 둘 남는다. **모델**이거나,
**데이터·물리·전문가**이거나. V8은 모델만 바꾼다. LeRobot 자신의 ACT를 `lerobot-train`으로
학습시켜, 우리 Observation IR·우리 Deployment IR과 Safety Plane·`es eval run`으로 돌린다.

**판정부터. 중단 규칙은 측정 전에 적어 두었기 때문이다.** 외부 ACT는 100,000 스텝에서 nominal
`success_rate = 0.0625`(1/16), 20,000과 50,000에서는 0/16이다. 데모의 합격선은 `0.5`이고 그 숫자는
내리지 않았다. **가설 1은 기각된다.** ImageNet 사전학습 백본과 CVAE와 DETR 디코더를 갖추고 다섯 배
긴 일정으로 학습한 진짜 ACT가, 이 데이터셋에서 IR 모양 그래프보다 낫지 않다. 다음 패킷이 여는 것은
씬·접촉 모델·전문가의 궤적이다. 옵티마이저도 아니고, `es-ir`를 넓히는 것도 아니다.

**1. 실제로 시험대에 오른 주장, 그리고 그것이 요구한 네 가지.**

스펙 §8.1은 네트워크 내부는 불투명하고 인터페이스 의미론만 타입이 붙는다고 말한다 — "우리는 독자적인
정책 아키텍처를 발명하지 않는다". 그것을 반증 가능하게 만드는 데 네 가지가 필요했고, 각각은 프로젝트가
기록해 두고 아직 닫지 않은 구멍이었다.

* **`meta/stats.json`.** 7.7절의 "의도적인 세 가지 누락"이 그것을 첫째로 꼽았다. 당시엔 `lerobot`을
  통해 학습하는 것이 없었기 때문이다. `lerobot-train`은 모든 정책 피처를
  `LeRobotDataset.meta.stats`로 정규화하므로, 이제 `crates/es-data/src/lerobot/v3.rs`에서 러스트로
  직접, parquet을 쓰는 같은 패스 안에서 쓴다 — 값에 대한 순회는 둘이 아니라 하나다. 피처마다
  `min`/`max`/`mean`/모집단 `std`/`count`, 이미지는 채널별 `[3, 1, 1]`에 `[0, 1]` 범위, 그리고
  **`qNN` 키는 없다**. 그것은 LeRobot 자신의 코드에서도 5000 구간 히스토그램 추정치이고
  `NormalizationMode.QUANTILES`만 읽는데 ACT는 전부 `MEAN_STD`다. 오라클은 LeRobot이 자기 포맷을
  판정하는 것이다. `lerobot_stats_ref.py`가 `LeRobotDataset`으로 익스포트를 열고 같은 parquet에
  `compute_episode_stats` + `aggregate_stats`로 통계를 다시 계산해 최대 불일치를 보고한다 —
  **RAN, 최대 `5.5e-08`**, 모든 피처·모든 키에 대해, 빠진 항목 없이.
  `docs/api-notes/lerobot-dataset.ko.md`가 포맷을 고정한다.
* **익스포트 선택 두 개. `lerobot`이 정책 피처를 이름만으로 분류하기 때문이다.**
  `dataset_to_policy_features`는 `action`으로 시작하는 모든 키를 ACTION 피처로 만들므로
  `action_commanded`와 `action_source`가 액션 헤드 둘로 더 도착했을 것이다. `--drop`이 그것을 뺀다.
  그리고 `observation.state`는 env 0의 전체 `qpos` 뒤에 전체 `qvel`을 붙여 기록되고(25개),
  그것은 **추론 시 되돌려줄 수 없다** — `es_eval`의 상태 캡처는 `qpos`를 읽고 `qvel` 갈래가 아예
  없다 — 그래서 `--state-dim 6`이 `qpos[..6]`을 남긴다. 추론 시 `Capture::Joints(6)`이 Observation
  IR에 건네는 바로 그것이다. 이것이 막는 것은, 런타임이 만들어 낼 수 없는 관측 위에서 수렴하는 학습이다.
* **`es policy import-lerobot`, 그리고 외부 아키텍처가 사는 곳.** Learning IR에는 없고, 있을 수도
  없다. 대신 스펙이 내놓는 답이 §8.3이다. *"π급 모델은 노드로 분해하지 않는다. `PolicyBundle`로
  통째로 참조하되 입출력 계약(§8.4)은 타입 검사한다"* — 그리고 §28의 리스크 표는 "Learning IR이 실제
  정책을 표현하지 못함"의 대비책으로 바로 이것을 이미 적어 두었다. 그래서 번들의 `LearningGraph`는
  계약의 포트를 실은 **`LearningNode::PolicyBundle` 하나**이고, 아키텍처 파라미터는 원래 살던 곳을
  타고 온다. `remap_checkpoint`가 체크포인트 자신의 `config.json`을 출력 safetensors의
  `__metadata__`에 기록하고, `TorchRuntime::load`는 그것을 찾으면 `lower_act`로 로워링한다.
  **해시 체인이 여전히 결정한다.** config는 `WeightsRef::hash`가 지목하는 바이트 안에 있고, `load`는
  파이썬이 무엇을 보기 전에 그 해시를 검증하며, `validate_keys`는 그대로 돌고, `INV-16`은 그대로다.
* **`observation-v8.toml`** — 같은 Task IR 위의 두 번째 Observation IR(§7). `evaluation-v8.toml`이
  그것을 가리키고, 합격 임계값은 하나도 움직이지 않았다.

**2. 위의 계획이 몰랐던 두 가지.**

**발견 1 — 상태 포트는 생 라디안을 실을 수 없고, 옳은 답은 정규화가 항등이라고 말하는 것이다.**
계획은 ACT의 정규화기가 체크포인트 안에 있다는 근거로 상태 포트를 `Unit::Angle`로 선언하는 것이었다.
교차 IR 패스가 그것을 거절한다. §5.4의 `TYPE-011`은 그래프 경계뿐 아니라
`PolicyContract::inputs`에 대해서도 검사되므로(`crates/es-ir/src/cross.rs:200`), 정책 입력은
`Normalized`·`Dimensionless`·`Token` 중 하나여야 한다. 예외는 없다. 그것은 프로젝트의 규칙이고 V8은
`es-ir`가 금지되어 있으므로, 따라야 했다.

*진짜* 아핀 사상을 넣어서 따르는 것은 잘못된 수정이었을 것이고, 왜 그런지는 적어 둘 값이 있다. 그것은
7.9절의 결함이 모자만 바꿔 쓴 것이기 때문이다. LeRobot의 정규화기는 정책이 **학습된** 값에 맞춰졌으므로,
추론에 적용되는 어떤 사상이든 내보낸 데이터셋에도 적용되어야 한다 — 즉 `Op::Normalize`를 익스포터에서,
파이썬의 손이 닿는 곳에서, 두 번째로 구현해야 한다. 노드 하나에 구현 둘, 또다시.

그래서 `observation-v8.toml`의 상태 갈래는 `Normalize { Range { lo: 0.0, hi: 1.0 } }`이고, 이는
`(q − 0) / (1 − 0)` — 선언된 항등이다. 포트는 `Normalized { 0, 1 }`이라고 말하는데, 이 코드베이스가
이미 그 단어를 쓰는 방식으로는 참이다. `sim_cube_pose` 헤더가 대놓고 적어 두었다. "`Normalize`는
아핀이지 클램프가 아니다." 그리고 네트워크는 여전히 생값을 먹지 않는다. 자기 첫 층이 체크포인트가
싣고 온 정규화기이기 때문이다. 외부에서 정규화하는 정책에 대한 Observation IR의 정규화 결정은 *항등*이고,
정직한 것은 그 결정을 내리는 노드에 그렇게 적는 것이지 부재로부터 독자가 추론하게 두는 것이 아니다.

그다음 임포트는 각 계약 입력의 **단위**를 Observation IR에서 가져온다(§5.1 규칙 6: 전처리는
Observation IR의 것). 둘이 `elem`과 `shape`에 대해 일치하는지 확인한 뒤에. `config.json`은 단위를
기록하지 않으므로 `act_policy`는 애초에 추측밖에 할 수 없었다.

**발견 2 — `lerobot-train`이 오늘 쓰는 체크포인트는 정규화 통계를 `model.safetensors`에 싣지 않는다.**
0.6.x는 정규화를 `ACTPolicy` 밖 프로세서 파이프라인으로 옮겼고, 그래서 통계는
`policy_preprocessor_step_3_normalizer_processor.safetensors`에, 접두사 없이
`<feature>.{mean,std,…}` 키로 산다. M4 게이트가 쓴 고정 업스트림 체크포인트
(`lerobot/act_aloha_sim_transfer_cube_human`)는 0.6 이전 업로드라 인라인으로 싣고 있고, 그래서 아무도
눈치채지 못했다. `remap_checkpoint`는 이제 두 번째 파일을 옵션으로 받아 접두사 표가 찾지 못한 항목만
채운다. 두 레이아웃이 다 읽히고 어느 쪽도 플래그가 필요 없다. `act_ref.py`도 같은 둘을 읽는다.
`docs/api-notes/lerobot-act.ko.md` 9절이 그 기록이다.

작은 것 하나 더. `config.json`은 체크포인트가 *학습된* 장치를 기록하고 `from_pretrained`가 그것을 따르므로
레퍼런스는 이제 `.to("cpu")`를 강제한다. 장치가 다른 두 fp32 결과를 비교하는 것은 모듈이 아니라
cuDNN의 커널 선택을 재는 일이다.

**3. 등가성 게이트, 이 정책이 실제로 학습한 프레임 위에서.** `act_checkpoint.rs`는 이제
`ES_ACT_OBSERVATION`을 받는다. 기록된 `{state, image}`를 왕복 보장 최단 십진수로 써서 양쪽이
비트 단위로 같은 f32를 쥔다. 프레임은 익스포트의 100번 — 팔이 뻗는 중간, 큐브는 시드가 놓은 자리 —
이고 이미지는 `f32::from(byte) / 255.0f32`, 즉 `es_compile`의 `ToTensor` 커널이 추론 시 내놓는
바로 그 값이다.

| 체크포인트 | 모양 | `max_abs` | `max_rel` | tier |
|---|---|---|---|---|
| 20,000 | `[16, 6]` | `0e0` | `0e0` | 비트 단위 (§8.9 tier 4는 `1e-5`) |
| 100,000 | `[16, 6]` | `0e0` | `0e0` | 비트 단위 |

`lerobot` 0.6.1, `torch` 2.11.0+cu129, CPU. 램프 위의 일치는 두 모듈이 같은 함수를 계산한다는 말이고,
정책이 학습한 프레임 위의 일치는 그것을 데모를 결정하는 입력 위에서 말하는 것이다.

**4. 무엇을 학습시켰는가.** LeRobot ACT 기본값 — ImageNet 사전학습 ResNet18, `use_vae = true`,
`latent_dim = 32`, `dim_model = 512`, 헤드 8, `dim_feedforward = 3200`, 인코더 4층 / 디코더 1층,
`kl_weight = 10.0`, `lr = 1e-5`(백본 포함), `dropout = 0.1`, 모든 피처 `MEAN_STD` — 에 두 가지 이탈,
둘 다 Deployment IR이 요구한 것이고 튜닝 선택이 아니다. `chunk_size = 16`(LeRobot 기본은 100;
`XIR-022`가 런타임이 버퍼링하는 청크와 정책이 예측하는 청크가 같기를 요구한다), 그리고
`n_action_steps = 16`(10이 아니라). `lower_act`의 모듈이 `actions[0][:n_action_steps]`를 돌려주고
런타임은 예측된 청크 전부가 필요하기 때문이다 — `execute_chunk = 10`과 시간 앙상블은 Deployment IR의
것이다(§9.2). `n_action_steps`는 ACT 손실에 들어가지 않는다. 배치 8, 시드 0, 100,000 스텝,
10,000마다 체크포인트.

손실, `lerobot-train`이 기록한 그대로(`l1`은 청크의 L1, `kld`는 CVAE 항 × `kl_weight`):

| 스텝 | 500 | 5,000 | 10,000 | 20,000 | 50,000 | 80,000 | 100,000 |
|---|---|---|---|---|---|---|---|
| 전체 | 4.270 | 0.320 | 0.124 | 0.055 | 0.035 | 0.029 | 0.026 |
| `l1` | 0.247 | 0.092 | 0.070 | 0.053 | 0.035 | 0.029 | 0.026 |
| `kld` | 0.402 | 0.023 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 |

수렴하고, V2b의 IR 모양 그래프가 도달한 것보다 *낮은* L1로 수렴한다(거기서의 20,000 스텝 0.0166은
정규화가 달라 직접 비교 대상은 아니지만, 여기 어디에도 적합 실패로 보이는 것은 없다).
**손실은 애초에 의심 대상이 아니었다.**

**5. 측정.** 시드 101–116의 홀드아웃 16 에피소드, `evaluation-v8.toml`, V6 봉투, `es eval run`,
평가 경로 변경 없음.

| 체크포인트 | `success_rate` | 평균 에피소드 길이 | `envelope_violation_rate` | 실패 |
|---|---|---|---|---|
| 20,000 | 0.0000 (0/16) | 900.0 | 0.180 | timeout 16 |
| 50,000 | 0.0000 (0/16) | 900.0 | 0.370 | timeout 16, fallback 스텝 40 |
| 100,000 | **0.0625 (1/16)** | 865.5 | 0.356 | timeout 15, fallback 스텝 14 |

100,000 스텝 실행은 한 번 더 돌렸고 `report.json`이 **바이트 단위로 동일**하게 나왔다. 해시 체인이
주장하는 결정성(§3.5 tier 1)을 외부 정책 위에서 실제로 거친 것이다.

100,000 스텝 번들에 대한 전체 스위트, 각 16 에피소드, `--jobs 6`:

| 스위트 | `success_rate` | 평균 에피소드 길이 | `envelope_violation_rate` |
|---|---|---|---|
| nominal | 0.0625 | 865.1 | 0.352 |
| light_intensity | 0.0625 | 866.9 | 0.350 |
| light_direction | 0.0625 | 873.0 | 0.253 |
| observation_delay | 0.0625 | 867.3 | 0.329 |
| torque_noise | 0.0625 | 866.2 | 0.352 |
| backlash | 0.0625 | 864.6 | 0.285 |

모든 스위트가 같은 한 에피소드를 성공으로 센다. 움직이지 않는 섭동 스윕은, 섭동이 아닌 다른 것이 표를
결정하고 있을 때의 모습이다 — 정책이 과제에 충분히 가깝지 않아서 조명 변화가 영향을 줄 여지가 없다.

**6. 통계에서 하나 남겨 둘 숫자.** 손목 롤 관절(인덱스 4)은 50회 시연 전체에 걸쳐
`observation.state.std = 0.0001`, `action.std = 0.0`이다. 스크립트 전문가가 그것을 전혀 돌리지 않는다.
LeRobot의 `MEAN_STD`는 `std + 1e-8`로 나누므로 그 채널의 입력은 ~10⁴배로 증폭되고 출력은 평균에 고정된다.
둘 다 상수 컬럼에 대해서는 올바른 동작이고 결함이 아니다 — 다만 진짜 트레이너가 진짜 `stats.json`을
읽어야 비로소 보이는 종류의 것이라, 두 번 발견되지 않도록 여기 적어 둔다.

**7. 벽시계** (RTX 4090, 16코어. 벤치마크가 아니라 관측이다).

| 단계 | 측정 |
|---|---|
| `es dataset export --lerobot-v3` (50 에피소드, 18,263 프레임, 출력 486 MB) | 4 초 |
| `lerobot-train`, 100,000 스텝, 배치 8, CUDA | 33분 50초 |
| `es eval run`, 16 에피소드, `--jobs 1`, `--frames` 포함 | 5분 44초 |
| `es eval run`, 96 에피소드(스위트 6개), `--jobs 6`, `--frames` 포함 | 9분 31초 |

**8. 해시.** `task_hash`와 `deployment_hash`는 움직이지 않는다. 그것이 이 패킷의 요점이다. 같은 과제,
같은 봉투, 다른 정책.

| 슬롯 | 값 |
|---|---|
| `task_hash` | `6cf826c1…6b7b` — V7a에서 변동 없음 |
| `deployment_hash` | `3b2ad568…6db1` — V6에서 변동 없음 |
| `observation_hash` (V8) | `006820aa5f787b4731f5866c57764afb91af814c2e2627d89e8e4aac2e27497c` |
| `evaluation_hash` (V8) | `d1819fca6ccf12597a49946d64b3ea1bb3ae16e8ec14d80e2ea95b46571de29c` |
| `learning_hash` (100k) | `71d08b44f83eb017f24493c374e8f2c5585e71046d8ae9b8702312695595b1a5` |
| `policy_hash` (20k / 50k / 100k) | `0d259d87…0244` / `a7f6acd2…0eda4` / `30ed13fa…42b6` |
| 원본 체크포인트 (100k) | `5a84f19ab3d06f62f699951fe9e36971212c3a67387b2e0c044d4706c5d3dc75` |
| `dataset_schema_hash` | `1f5ddafc…8777c` — 원본 데이터셋의 것, 변동 없음: 컬럼을 더하지 않았다 |

원본 체크포인트의 해시는 `BaseModelRef`로 체인 안에 있다. 그래서 번들은 자기가 어느 LeRobot 파일에서
왔는지와 실제로 적재하는 리맵된 파일을 둘 다 이름 댄다.

**9. 영상.** 100,000 스텝 실행의 nominal 16 셀에 대해 `es video mosaic --grid 4x4`, 384×392로 900
프레임, 그다음 `encode_video.py`와 H.264 복사본: `demo-v8-h264.mp4`, 1.6 MB. 팔이 뻗다가 멈추는 것이
보이고, 플레인이 클램프한 스텝에는 빨간 테두리가 있다.

**열린 질문 (V8-1), 12절 목록에 넣을 것.** *외부 ACT의 실패가 결론 내려도 되는 것.* 이것은 "우리
Learning IR 노드 집합이 데모가 안 되는 이유다"를 배제한다. 배제할 값이 있었고, 이제 비트 단위로 검증된
임포트로 배제되었다. 그러나 "96×96은 너무 작다"나 "50 에피소드는 너무 적다"는 배제하지 **않는다** —
V8은 그것들을 일부러 고정했다. 모델과 동시에 움직였다면 두 가지를 재는 일이 되었을 것이다. 기본값:
데모의 실패를 **데이터와 물리**의 발견으로 취급하고, 다음 패킷을 씬의 진단으로 삼는다. 전문가 자신이
기록한 액션을 `es eval run`으로 재생해 하네스가 그 성공률을 재현하는지 확인하고, 그다음 해상도와 시연
수를 하나씩 바꾼다. 대안 — 곧바로 더 높은 해상도에 더 많은 시연 — 은 더 큰 실행이고, 실패했을 때 같은
모호함을 남긴다.

### 7.17 구현 결과 (V9): 쇼케이스 렌더

패킷 `docs/packets/M5/V9-showcase-render.md`. 이 데모가 지금까지 만들어 낸 것은 숫자, 해시,
그리고 7.8절의 4x4 모자이크뿐이었다 — 96x96 타일 열여섯 개를 바로 위에서 내려다본 384x392
픽셀. 파이프라인 안의 유일한 카메라가 신경망이 읽는 그 카메라이고, 그것은 의도적으로 96
픽셀이기 때문이다. 거기서 팔이 큐브를 집는 장면을 볼 수 있는 사람은 없다. V9는 **이미 끝난
실행**을, 어떤 씬에도 어떤 `ObservationSpec`에도 어떤 해시에도 없는 카메라로 다시 렌더한다.

**1. 두 번째 카메라가 아니라 리플레이이고, 그 선택은 코드가 했다.** `EnvRenderer::frame`은
이미 포즈와 뷰의 순수 함수다 — `body_poses` → `TriScene::from_scene_with_poses` →
`Renderer::render` → `read_tile` — 그리고 실행 중인 env에서 가져가는 것은 `StateView::xpos` /
`xquat`뿐이다. 그래서 "렌더러가 받은 것을 기록한다"와 "로봇이 무엇을 했는지 기록한다"는 같은
파일이고, 그 파일이 생기는 순간 렌더는 물리 백엔드도, 정책도, 파이썬도 프로세스에 없는
리플레이가 된다. 이제 `es eval run`과 `es loop collect`는 언제나
`<run>/traj/<cell>.estraj`를 쓴다 (`crates/es-env/src/traj.rs`, `es_env::Trajectory`): 매직,
`nq`/`nv`/`nbody`/`ticks`, 바디 id 목록, 그리고 제어 틱마다 한 행씩
`qpos ‖ qvel ‖ (pos[3] ‖ quat[4]) * nbody`. 에피소드당 1 MB 미만이며, 같은 에피소드가 이미
쓰는 96x96 프레임 24 MB에 비하면 없는 것과 같다.

대안이었던 `es eval run --showcase WxH` — 실행 중에 두 번째 `ImageSpec`을 렌더하는 안 — 은 몇
줄 더 작지만 리플레이가 하는 세 가지를 못 한다. MuJoCo 900 스텝과 신경망을 다시 돌리지 않고
다른 각도·다른 해상도로 다시 렌더하기, `es eval run`을 아예 거치지 않는 **전문가**를 렌더하기,
그리고 *검증되기*. 루프 안의 두 번째 카메라에는 비교할 기준이 없고, 리플레이에는 있다.

**2. 포즈를 다시 계산하지 않고 저장한다, 그리고 그것이 오라클 전부다.** 리플레이 시점에
`qpos`로부터 바디 포즈를 다시 구한다는 것은 정기구학을 뜻하고, 그것은 리플레이 프로세스에 물리
백엔드를 들이고 `mj_forward`의 두 번째 구현이 어긋날 여지를 만든다는 뜻이다. 저장해 두면
리플레이된 틱은 렌더러가 받았던 바로 그 `BTreeMap<StableId, Pose>`이고, 따라서 끝난 실행을 그
실행 **자신의** `ImageSpec`으로 그 실행 **자신의** 카메라를 통해 다시 렌더하면 기록된 프레임이
바이트 단위로 재현되어야 한다. 사소해 보이지만 결정적인 두 가지: `Trajectory::poses`는
`Pose::new`가 아니라 구조체 리터럴로 `Pose`를 만든다 — 이미 정규화된 단위 쿼터니언을 한 번 더
정규화하면 마지막 비트가 움직이기 때문이다. 그리고 틱은 제어 틱이 시작하는 곳이 아니라
*프레임이 캡처되는* 곳에서 기록된다 — `observation_delay` 아래에서도 궤적 인덱스와 프레임
인덱스가 같은 숫자로 유지된다.

오라클 서버에서 측정, 2026-09-15, 둘 다 통과:

* `a_showcase_replay_reproduces_the_frames_the_policy_saw` — 스크립트 전문가를
  `es_eval::Evaluation`으로 돌리되 프레임 소스를 데모 자신의 96x96 `ImageSpec`에서
  `es_render::cpu::rasterize`(모든 렌더 골든을 생성하는 바로 그 함수)로 두고, 그 다음 모든 틱을
  파일로부터 다시 렌더: **120 프레임 중 120개 동일**, 23.6초. GPU가 아니라 `mujoco`가 필요하다.
* `showcase_replay_of_a_real_run_is_bit_identical` (`#[ignore]`) — 실제 `es eval run --frames`
  결과에 대해 `es video showcase --camera overhead --width 96 --height 96 --cell nominal-00`을
  GPU 경로로: **900 프레임 중 900개 동일**, 5.7초.

**3. 카메라가 명령줄 인자인 이유는 씬이 카메라를 더 가질 수 없기 때문이다.** `scene_hash`는
`task_hash`로, `task_hash`는 `observation_hash`로, 그리고 학습된 번들 전부로 이어진다. 그래서
`tests/fixtures/mjcf/so101_pick_place.xml`에 `<camera>`를 추가하는 것은 이 영상이 보여 주려는
체크포인트들을 무효화하는 일이다. 해시를 움직이는 카메라는 독립적인 카메라가 아니다.
`es_env::render::look_at`이 `--eye`, `--look-at`, `--fov`로부터 스펙 3.1의 `OpenCV` 뷰를 만들고,
월드 업은 `+Z`이며, 시선이 그 축과 평행하면 롤을 임의로 고르는 대신 거절한다. `--camera NAME`은
씬이 *실제로* 선언한 카메라를 쓰며, 그것이 리플레이를 실행과 대조하는 방법이다.

영상이 쓰는 카메라는, 한 에피소드의 세 프레임을 여덟 가지 후보로 렌더해 눈으로 보고 고른
**eye `(0.66, -0.46, 0.52)`, look-at `(0.14, -0.04, 0.04)`, fovy 36°, 1280x720**이다. 작업
공간의 앞쪽·옆쪽·위쪽에서 본 각도이며, 한 에피소드 내내 팔 전체와 통과 큐브를 화면 안에 담는다.

**4. 무엇을 렌더했고 얼마나 걸렸는가.** 벽시계 시간, RTX 4090, `Rs`, 1280x720, 카메라 하나 —
이 씬을 이 크기로 렌더한 *관측값*이지 렌더러에 대한 주장이 아니다 (스펙 12.4):
**프레임당 82.4 ms** (전문가, 1407 프레임), **78.2 ms** (정책, 900 프레임). 비용은 픽셀이
아니다. 7.1절이 이미 적어 둔 천장 — 매 프레임 씬 전체를 다시 테셀레이트하고 다시 업로드한다 —
이 삼각형 2,978개에서 측정된 것이 이 숫자다. 같은 렌더러가 평가 안에서 96x96 관측 프레임을
같은 프레임당 비용으로 만든다.

| 영상 | 출처 | 에피소드 | 프레임 | fps | H.264 크기 |
|---|---|---|---|---|---|
| `expert-showcase.mp4` | `es loop collect --expert`, 시드 1 | 4 (4/4 `Success`) | 1407 | 30 | 2.59 MiB |
| `policy-20000-showcase.mp4` | `es eval run`, V7a 20,000스텝 번들, 시드 101 | 1 | 900 | 30 | 1.34 MiB |

둘 다 기록된 제어 틱마다 한 프레임이므로 30 fps에서는 50 Hz 제어율의 0.6배 속도로 재생된다 —
잡는 순간이 뭉개지지 않고 보이며, 프레임 수가 곧 틱 수다(오라클이 검사하는 것이 그것이다).
전문가는 에피소드당 약 350틱(7초)이 걸리고, 정책의 에피소드는 전부 900틱 예산을 다 쓴다.

**대비가 핵심이고, 보기 좋은 대비는 아니다.** 전문가의 네 에피소드는 네 번 모두 큐브를 집어
통에 넣는다. V7a 20,000스텝 상태 정책은 같은 nominal 스위트의 홀드아웃 시드 네 개(101–104)를
같은 엔벨로프로 통과시켰을 때 `success_rate 0.0000`, `episode_length 900.0`,
`envelope_violation_rate 0.0433` — 타임아웃 4회, 가속도 클램프 155회, 속도 클램프 45회다.
영상으로 보면 팔이 들려 올라가 통 근처에서 맴돌고 큐브에는 끝내 닿지 않는다. 이것은 영상을
위해 만든 네 에피소드 실행이고 자신의 `evaluation_hash`를 가지며, 고정된 열여섯 에피소드
측정이 아니다. 표가 아니라 *영상이 보여 주는 것*으로서 여기 적는다.

**5. 경로 추적기는 여기서 헤드리스로 돌아가지만 쓰지 않는다.** 서버에서
`cargo test --release -p es-render`는 디스플레이 없이 RTX 4090에서
`gpu_path_tracer_matches_the_cpu_reference_at_1spp`,
`gpu_pt_and_rs_agree_on_geometry`, `gpu_restir_and_svgf_match_the_cpu_within_tolerance`를
통과한다. 즉 PT는 쓸 수 있다. 그런데도 `es video showcase`가 `Rs` 전용인 이유는 장치가 아니다.
`PT_CHANNELS`는 `PtRadiance`(선형 `f32`)와 기하 채널 셋이고 `Rgb8`은 거기 없다. 그래서 PT
쇼케이스에는 톤 매핑 — 노출과 커브 — 이 필요하고, 뒷받침할 오라클도 없이 룩(look)에 대한 결정을
CLI 플래그 뒤에 숨기는 것은 이 패킷이 할 일이 아니다. 필요해지는 날을 위해
`EnvRendererCfg::path`는 이미 그 변형을 들고 있다.

**6. 작은 발견 두 가지.** `python/es/encode_video.py`는 이제 `~/venvs/es`에서 돌지 않는다. 그
환경에는 `mujoco`와 `torch`는 있지만 `opencv-python`이 없고, 있는 쪽은 `~/venvs/es-lerobot`이다
(cv2 4.13.0, `mp4v`, 전문가 영상 13.0 MiB). 위 표의 H.264 파일은 7.8절이 이미 기록한 경로 —
서버 자신의 `ffmpeg 7.0.2`로 같은 원시 프레임을 인코딩 — 로 만들었고, 5배 작다. 그리고
`es loop collect --expert`는 렌더러 없이도 궤적을 쓰므로 전문가 영상의 비용은 0.4초짜리 수집
실행 하나다. 전문가는 픽셀을 필요로 한 적이 없었고, 이제 그 영상도 그렇다.

### 7.18 구현 결과 (V10): 장면 진단

패킷 `docs/packets/M5/V10-scene-diagnosis.md`. 7.15절의 발견 8은, 하네스가 스크립트 전문가를
통과시키고(V6/V6b, 7.12–7.13절) 모델이 배제된(V8, 7.16절: LeRobot 자신의 ACT, 비트 단위로 검증된
임포트, 100,000 스텝, 0/16 · 0/16 · 1/16) 지금에도 데모가 학습되지 않는 이유로 용의자 셋을
지목했다. 시연 데이터, 파지, 그리고 시간 앙상블이다. V10은 셋 모두를 측정한다 — 학습 없음, 모델
변경 없음, 픽스처도 해시도 움직이지 않음 — 그리고 **셋 다 건전하다**. 두 번째를 건전하다고
증명한 측정이 아무도 보지 않던 네 번째를 찾아냈고, 그것이 이 패킷의 발견이다.

아래는 모두 V1c 학습 세트다: `~/artifacts/plan-v/v1c/ds-train`, `es loop collect --episodes 50
--seed 1`, 50 에피소드, 18,263 프레임, `task_hash aec2aea9…`, `observation_hash f4a50730…`,
데이터셋 내용 `51a953d6…`. 오라클 서버에서 2026-09-15 측정.

**1. 기록된 액션은 시연을 재현한다: 50/50.**
`recorded_actions_replay_to_the_same_outcome`(`crates/es/tests/cli.rs`)는 각 에피소드의 기록된
액션 행을 같은 리셋에서 개루프로, 같은 `Env`와 같은 `SafetyPlane`을 통해 재생하고, 시연 자신이
기록한 것(빈 내부의 3차원 영역 안에 있는 큐브 — `expert_solves_the_pinned_seeds`가 확인하는 것이고
Task IR의 x 전용 술어가 볼 수 없는 것, 5.4절)과 대조한다. 두 번째 `PolicyRuntime`이 아니라 `Env`
위의 평범한 재생 루프다: `INV-17`이 허용하는 확장점은 일곱이고 기록 액션 재생기는 그중 하나가
아니며, 플레인은 여전히 모든 행을 본다.

| 재생 대상 | 빈 안의 큐브 | `Termination::Success` | 플레인이 고친 틱 | 최대 보정량 |
|---|---|---|---|---|
| 시연 자신의 기록 | **50 / 50** | — | — | — |
| `action`(실행된 `SafeAction`) | **50 / 50** | 49 / 50 | 10,787 / 18,263 | 9.004e-4 rad |
| `action_commanded`(원 명령) | **50 / 50** | 50 / 50 | 11,589 / 18,263 | 3.258e-1 rad |

세 가지가 고정된다. **데이터는 건전하다** — 모든 시연이 재현 가능한 픽앤플레이스이고,
실행값 대 명령값의 구분은 아무 대가도 치르지 않는다: 두 열 모두 큐브 쉰 개를 빈에 넣는다.
**`action`은 정말로 플레인 자신의 출력이다**(7.10절의 주장을 반대편에서 측정한 것): 다시
검증해도 최대 0.9 밀리라디안만 움직이는데 이는 `f32` 저장과 레이트 리미트 반올림이고, 원 명령을
다시 검증하면 최대 0.33 rad 움직인다. **49/50은 물리가 아니라 Task IR 술어다**: 한 에피소드의
큐브는 빈 안에 있는데 x 하나짜리 콘 리프가 아니라고 말하며, 이는 5.4절이 기록한 천장 그대로다.
청크 `seq`를 런 단위가 아니라 에피소드 단위로 돌리면 점수는 1/50이 된다 — 플레인은 마지막으로 본
것보다 큰 `seq`에 대해서만 청크를 받아들이고 `begin_episode`는 의도적으로 그것을 되돌리지 않으므로,
첫 에피소드 이후는 전부 영구 언더런이다. 이는 재생 루프의 성질이며, 다음 사람이 가장 먼저 틀릴
지점이라 여기에 적어 둔다.

**2. 전문가는 파지한다, 밀지 않는다: 50/50.**
`python/es/grasp_probe.py`는 같은 액션 시퀀스 쉰 개를 순수 `mujoco` 3.13으로 — 루프 안에 `es`
런타임 없이 — 재생하며, 틱마다 각 조와 `cube_geom` 사이의 접촉 법선력, 큐브의 높이, 그리퍼 관절의
명령값 대 측정값을 기록한다. 리셋 상태와 액션 행은 위 재생에서 온다(`ES_V10_DUMP`). Task IR의
무작위화 추첨을 아는 쪽이 그쪽뿐이기 때문이며, 덤프의 큐브 열이 자체 검증이다.

| | 측정값 |
|---|---|
| 큐브를 테이블에서 완전히 들어 올린 시연(> 20 mm, 큐브 반높이) | **50 / 50** |
| 들어 올린 높이, 중앙값 / 최대 | 122.65 mm / 127.67 mm |
| **양쪽** 조가 동시에 접촉한 틱, 중앙값 / 최대 | 161.5 / 474 |
| 에피소드 중 파지가 차지하는 비율 | 47.9 % |
| 양쪽 조가 큐브를 쥐고 있을 때의 그리퍼 관절 | 0.0950 rad (명령 −0.05, 열림 0.9) |
| 큐브가 빈 안에서 끝난 시연 | 50 / 50 |
| `es` 재생과의 최대 큐브 편차 | **0.0002 mm** |

25 mm 큐브를 테이블 위 122 mm로 옮기고 에피소드의 절반 동안 두 조 사이에 쥐고 있는 것은 어떤
정의로도 파지다. 0.0950 rad은 큐브를 사이에 두었을 때 조가 멈추는 기하학적 폐합각이며, 5.3절의
`grip_closed = −0.05`는 서보를 포화시켜 단단히 쥐려고 일부러 그것을 지나치도록 잡은 값이다.

**3. 시간 앙상블은 지연될 뿐, 그리퍼를 열지 않는다.**
`the_temporal_ensemble_survives_the_grasp_window`는 시연 하나를 `es loop collect`와 `es eval
run`이 하는 것과 똑같이 `es_env::plane_chunk`로 구동하면서, *두 번째* `ChunkBuffer`에 동일한 청크를
`HardSwitch`로 먹인다. 두 번째 것이 원 명령 — 그 틱에 대한 최신 청크 자신의 행 — 이므로 둘의 차이는
앙상블이고 그 외에는 아무것도 아니다. 시드 1, `decay = 0.01`, 16행 호라이즌의 중첩 청크
`CHUNK_SLOTS = 8`개. 에피소드는 351틱이다: `Approach` 0, `Descend` 71, `Close` 118, `Lift` 143,
`Transport` 202, `Lower` 268, `Release` 314, 그리고 `Success`로 끝난다.

| 관절 | 118–338틱의 max &#124;blend − raw&#124; | raw 최소 | blend 최소 | raw 최대 | blend 최대 |
|---|---|---|---|---|---|
| 0 `shoulder_pan` | 0.22955 | −0.00031 | −0.00031 | 0.77272 | 0.77272 |
| 1 `shoulder_lift` | 0.26197 | −1.37040 | −1.37040 | 0.16505 | 0.16505 |
| 2 `elbow_flex` | 0.20923 | 0.24469 | 0.24469 | 0.71384 | 0.71384 |
| 3 `wrist_flex` | 0.22593 | 1.08776 | 1.08776 | 1.51702 | 1.51702 |
| 4 `wrist_roll` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 5 `gripper` | **0.35651** | **−0.05000** | **−0.05000** | 0.90000 | 0.90000 |

**모든 관절의 혼합 명령은 최신 청크가 요구한 양 끝에 여전히 도달한다.** 그리퍼도 마찬가지로,
혼합값은 정확히 `grip_closed = −0.05`까지 닫히고 정확히 `grip_open = 0.9`까지 열린다. 앙상블은
*지연*이지 — 그리퍼에서 최대 0.357 rad, 램프의 여섯 틱 반 정도 — 범위의 손실이 아니다. 극값은
중첩된 여덟 청크가 모두 동의할 만큼 오래 유지되기 때문이다. 조 관절은 가장 조일 때 0.0678 rad을
가리키며, 측정 2가 큐브를 쥐고 있는 동안 보고하는 0.0950 rad보다 안쪽이다. 테스트는 허용오차가
아니라 등식을 단언하므로, `decay`나 `CHUNK_SLOTS`를 그리퍼가 열릴 만큼 바꾸면 실패한다.

**4. 세 측정이 가는 길에 찾아낸 것: 제어 한 스텝은 20 ms가 아니라 5 ms다.**
`grasp_probe.py`는 `--substeps`를 *설정*이 아니라 *측정*으로 받는다. 값이 맞을 때만 큐브 열이 `es`
재생 위에 남기 때문이다. `--substeps 1`에서 쉰 에피소드에 걸친 최대 편차는 **0.0002 mm**,
`--substeps 4`에서는 **202.98 mm**다. 즉 기록된 액션 한 행은 장면 자신의 `timestep="0.005"`
MuJoCo 스텝 정확히 하나 — **200 Hz** — 인 반면, `tests/fixtures/visible-learning/deployment.toml`은
`rate.control = 50`, `rate.inference = 5`를 선언한다. 둘을 잇는 것은 아무것도 없다. `Env::new`는
장면을 `LoadConfig { rate: None }`으로 적재하므로 물리는 MJCF의 타임스텝을 유지하고, `Env::step`은
`schedule.domains().inference.period`만큼 틱을 전진시키는데, `Collector::run`과 `es_eval::runner`가
만드는 유일한 것인 `BatchDomains::single_env()`가 그것을 1로 둔다.

결과는 셋이고 모두 산수다.

* **Safety Plane의 모든 동적 상한이 그것이 다스리는 물리보다 네 배 헐겁다.** `SafetyPlane`의
  `dt_s`는 Deployment IR에서 오므로 `velocity_max · dt = 3.0 · 0.02 = 0.06` rad이 *5 ms* 스텝마다
  허용된다: 선언된 3 rad/s에 대해 12 rad/s다. `acceleration_max · dt²`는 열여섯 배 헐겁다.
  엔벌로프가 위반된 것이 아니라 단위가 틀린 채로 측정되고 있다.
* **정책이 분해해야 하는 신호가 3 밀리라디안이다.** 쉰 에피소드에서 측정한
  `|action[t] − qpos[t−1]|`의 중앙값은 **0.00322 rad**, `|action[t] − qpos[t]|`는
  **0.00251 rad**이다. 따라서 16행 청크는 약 80 ms, 약 0.05 rad의 이동을 담으며, 관절 범위가
  ±1.7 rad인 액션 공간 위에서 그렇다. `--substeps 4`로 같은 시연을 읽으면 청크 이동량이 0.0982가
  아니라 0.0480 rad이고 틱당 증분은 0.00033으로 떨어진다 — 데이터셋이 저작된 속도로 읽히기
  때문에 수치가 움직인다.
* **시연은 모두가 생각하는 것보다 네 배 짧다.** 351틱 에피소드는 7초가 아니라 시뮬레이션 시간
  1.76초이고, `max_episode_steps = 900`은 4.5초다.

이 측정이 비로소 물을 수 있게 만든 한 틱 짝짓기 문제는 답이 났고 원인이 *아니다*: `Env::step`은
스텝이 끝난 qpos 옆에 `ctrl`을 기록하므로 `observation.state[t]`는 `action[t]`가 계산된 행이 아니라
그 결과이고, 이는 LeRobot의 짝짓기와 반대다 — 그러나 5 ms에서 두 읽기의 차이는 0.7 밀리라디안
(0.00251 대 0.00322)으로 잡음이다. "이미 보이는 관절을 그대로 반복하라"는 자명한 예측기에 대한
청크의 L1은 궤적 전체에서 0.0982 rad, 파지 구간 안에서 0.0964 rad이므로 파지가 목적함수의 사라질
만한 조각인 것도 아니다: 에피소드의 47.9 %이고 나머지보다 예측하기 어렵지도 않다. 두 가설 모두
측정되어 닫힌 것으로 여기에 기록한다.

**판정.** 장면도, 시연도, 전문가도, 앙상블도 원인이 아니다. 데이터는 쉰 에피소드 모두에서
재현 가능한, 양 조로 쥐고 122 mm 들어 올리는 픽앤플레이스이고, 전체 스택으로 재생하면 큐브 쉰 개가
빈에 들어가며, 정책이 학습되고 평가되는 청크 혼합은 모든 관절의 양 끝에서 전문가의 명령을 정확히
보존한다. V10이 대신 찾아낸 것은 **모든 문서가 50 Hz를 선언하는 동안 데모 전체가 200 Hz로
돌아왔다**는 사실이며, 이는 학습 문제를 설계된 것보다 네 배 미세하게 만들고 — 스텝당 명령 운동
3 밀리라디안, 청크당 80 ms — 모든 동적 엔벌로프 수치를 그것이 제한하는 스텝보다 네 배에서 열여섯
배 헐겁게 만든다.

**가장 작은 다음 패킷**은 따라서 모델도, 해상도도, 시연 개수도 아니다. 제어 한 스텝이 제어 주기
하나만큼 지속되게 만드는 것이다. `Env::new`에는 이미 필드가 있고 — `LoadConfig::rate`는 `Some`
하나면 물리 속도를 정하고, `BatchDomains`에는 스케줄이 존중하는 주기가 이미 있다 — 그러므로 변경은
둘 중 하나를 Deployment IR의 `rate.control`에서 유도하고 타임스텝이 그것을 나누지 못하는 장면을
거부하는 것이다. 그런 다음 V2의 정확한 손잡이로 다시 수집하고 다시 학습해 V8의 100,000 스텝
수치와 비교한다. 그 수치가 존재하기 전까지 플랜 V의 다른 무엇도 움직여서는 안 된다. 데모가 낸 모든
학습·평가 수치가 의도된 속도의 네 배에서 얻어진 것이기 때문이다. 미해결 질문 18.

### 7.19 구현 결과 (V11): 제어 한 스텝이 제어 주기 하나

패킷 `docs/packets/M5/V11-control-period.ko.md`. 7.18절의 발견 4 — 모든 문서가 50이라고 말하는
동안 데모는 200 Hz로 돌았다 — 를 여기서 고쳤고, 미해결 질문 18의 기본값 **(b)** 가 만들어진
것이다. 물리는 두고 제어를 데시메이션한다.

**유도.** `BatchDomains::single_env_at(physics, control)`(`crates/es-env/src/scheduler.rs`)는
`period = physics / control`을 정확한 유리수 `(pn·cd) / (pd·cn)`로 계산하고, 타임스텝이 제어
주기를 나누지 못하는 장면은 반올림하지 않고 두 속도를 메시지에 담아 이름으로 거부한다(부록 B.5의
검증 방식). 데모에서는 `200 / 50 = 4`다. `Env::step` 한 번이 `ctrl`을 한 번 설정하고 물리 틱 네
개를 전진시키며 한 행을 기록한다. `LoadConfig::rate`는 `None`으로 남으므로 장면 자신의
`timestep="0.005"`와 그에 따른 접촉 거동은 건드리지 않는다 — 시연과 M6 Go1 트랙이 둘 다 그것에
의존한다.

`observation.period`는 같은 값을 의도적으로 가진다. `DomainRunner::observe_window`는 윈도우
*앞에서* 상태를 한 번 읽고 윈도우의 시뮬레이션 틱들을 순회하므로, `observation.period = 1`이면
같은 판독이 `TemporalWindow`에 네 번 들어가고 같은 프레임이 네 번 렌더링된다. 정책이 소비하는
것은 제어 스텝당 관측 하나다. `rate.inference = 5`는 여전히 "제어 10스텝마다 재계획"을 뜻한다 —
그것은 `ChunkBuffer`를 지나는 `action.execute_chunk`이고(7.13절) 건드리지 않았다.

`Collector::run`, `es_eval::runner`, 그리고 `Env`를 직접 구동하는 7.18절의 테스트 두 개에
연결된다(공유 헬퍼 `demo_domains`를 통해, 그래서 테스트가 수집기와 같은 방식으로 장면을
전진시킨다). `es video showcase`의 `.estraj` 기록은 **제어 틱 단위**이고 그대로다 — 두 루프 모두
`step_with_policy` 반환 뒤에 한 행을 넣는다 — 그래서 이제 진짜 50 Hz 기록이고
`encode_video.py --fps 50`은 1/4배속이 아니라 실시간으로 재생한다. 7.17절의 표는 30 fps
영상이 제어 속도의 0.6배로 재생된다고 적었는데, 200 Hz 틱에서는 사실 0.15배였다. 이 패킷이
7.17절에서 바로잡는 유일한 수치다. 데이터셋의 `fps`는 언제나
`rate.control`이었고 이제 참이다. 행은 실제로 20 ms 간격이다. `TickRate::from_period_secs`는
타임스텝→틱 변환을 `es-core`로 옮겨, `f64` 시간을 금지하는 스케줄러(§18.1)가 나노초 반올림의
사본 없이 닿을 수 있게 한다.

정규 인코딩은 아무것도 바뀌지 않았으므로 **픽스처도 골든도 움직이지 않았다.**
`regenerate_visible_learning_documents`도 `regenerate_quadruped_documents`도 할 일이 없었다.

**수정이 곧바로 드러낸 것: 스크립트 전문가가 동작을 멈췄다.** 진짜 50 Hz에서
`expert_solves_the_pinned_seeds`는 **0/8**, 매 에피소드 타임아웃이었다. 순수 `mujoco`로 기록된
행을 툴 사이트에 대해 재생해 측정했다.

| | 측정값 |
|---|---|
| 호버 자세의 툴, 래치된 큐브까지 xy 거리 | 0.0003 m (정확) |
| 하강 바닥의 툴 z, 목표 0.0146 대비 | **0.0033 m** |
| 큐브를 치는 제어 틱 하나에서의 큐브 변위 | **59 mm** |
| 그 뒤 유지 명령 대비 `shoulder_lift` | 0.018 rad |
| 같은 관절, 같은 `ctrl`, 접촉 없음, 순수 MuJoCo | 4.9e-4 rad |

즉 팔은 정확히 접근하고, **파지 자세를 약 11 mm 지나치며**, 집게를 테이블에 박고, 큐브를 닿지
않는 곳으로 쳐내고, 테이블에 얹힌다 — 거기서 웨이포인트 기계의 `pos_tol = 0.01` 게이트는 다시
닫힐 수 없다. 원인은 장면이 낼 수 없는 한계다. `forcerange` 2.94 N·m에 대해
`velocity_max = 3.0` rad/s와 `acceleration_max = 20` rad/s²는 하강 바닥에서 제동 여력을 남기지
않는다. 이전에는 드러날 수 없었다. 제어 한 스텝이 5 ms 물리 스텝 하나였고 팔은 에피소드 내내 속도
포화 상태였기 때문이다 — 램프를 추종할 수 없었으니 지나칠 수도 없었다. **플랜 V가 수집한 모든
시연은 구동된 것이 아니라 끌려다닌 팔이 만든 것이다.**

`ExpertCfg::pace_to`는 전문가를 엔벌로프에 맞추려고 이미 있었다. 그 계수가 0.9에서
`PACE = 0.5`가 되었다. 진짜 속도에서 고정 시드 여덟 개로 스윕: **0.9 → 0/8, 0.75 → 2/8,
0.6 → 8/8, 0.5 → 8/8, 0.35 → 8/8**. 양쪽에 여유를 둔 절반. 엔벌로프 한계도, 수용 게이트도,
Deployment IR 값도 움직이지 않았다. 장면 자신의 액추에이터에 대한 *전문가*의 보정이며, V11이
스케줄 외에 돌린 유일한 손잡이다.

**V10의 측정이 정확히 뒤집힌다.** `grasp_probe.py --substeps`는 V10이 쓴 바로 그 측정이고,
**새로 수집한** 50 에피소드 집합에 적용했다 — V1c 집합은 200 Hz 데이터라 50 Hz로 재생할 수 없어
재사용이 불가능했다.

| `grasp_probe.py` | V10 (`--substeps 1`이 추종) | V11 (`--substeps 4`가 추종) |
|---|---|---|
| 서브스텝 1에서 `es` 재생 대비 최악 큐브 편차 | 0.0002 mm | **195.0051 mm** |
| 서브스텝 4에서 `es` 재생 대비 최악 큐브 편차 | 202.98 mm | **0.0000 mm** |
| 큐브를 테이블에서 띄운 시연 | 50 / 50 | **50 / 50** |
| 들어올림, 중앙값 | 122.65 mm | 120.75 mm |
| 양 집게 접촉 틱, 중앙값 | 161.5 | 90.0 |
| 양 집게가 큐브를 잡을 때 그리퍼 관절 | 0.0950 rad | 0.0934 rad |

새 집합에 대한 `recorded_actions_replay_to_the_same_outcome`: 기록된 큐브 50개가 통 안,
`action` 재생이 50 재현(`Success` 50), `action_commanded` 재생이 50 재현.

**이제 자신이 제한하는 스텝에 대해 측정되는 엔벌로프.**

| | V6 / V10 (200 Hz) | V11 (50 Hz) |
|---|---|---|
| `expert_passes_the_evaluation_harness` | 8 / 8 | **8 / 8** |
| 최악 `envelope_violation_rate` | 0.48 – 0.55 | **0.2044** (0.1535 – 0.2044) |
| `expert_solves_the_pinned_seeds` | 8 / 8 | **8 / 8** |
| `the_temporal_ensemble_survives_the_grasp_window` | `Success` | **`Success`** |
| 에피소드 길이, 제어 스텝 | ~351 | 225 – 235 |

**이제 제어 한 스텝의 값어치.** 두 열 모두 같은 명령(`--episodes 50 --seed 1`)에 같은 스크립트로
측정했으므로 7.18절의 관절별 수치가 아니라 서로 견줄 수 있다.

| 시연 집합당 | V1c (200 Hz) | V11 (50 Hz) |
|---|---|---|
| 프레임 | 18,263 | 9,038 |
| 에피소드 중앙값, 제어 스텝 | 352 | 179 |
| 에피소드 중앙값, 시뮬레이션 초 | 1.76 | **3.58** |
| `max_j abs(action[t] − action[t−1])`, 중앙값 | 0.01687 rad | **0.03985 rad** |
| `max_j abs(action[t] − qpos[t−1])`, 중앙값 | 0.26506 rad | **0.06223 rad** |
| 16행 청크 이동, 중앙값 | 0.18931 rad | **0.47143 rad** |
| `meta/info.json` `fps` / 실제 행 간격 | 50 / 5 ms | **50 / 20 ms** |

명령이 더 이상 팔을 1/4 라디안 앞서지 않고, 청크는 80 ms와 1/5 라디안이 아니라 320 ms와 0.5
라디안의 이동이다.

**두 정책: 어느 쪽도 과제를 배우지 못했고, 중단 규칙이 발동한다.**

IR 소유 그래프, 새 50 Hz 집합에 V2의 정확한 손잡이(`es policy lower` →
`train_act.py --batch 8 --lr 1e-4 --seed 0 --device cuda --resident-gpu`, 최적화 20,000 스텝,
초기 손실 0.0565 → 최종 **0.0132**, RTX 4090에서 654 s. V1c의 200 Hz 실행은 0.0669에서
0.0180까지 갔다):

| 스위트 | `success_rate` | `envelope_violation_rate` | `episode_length` |
|---|---|---|---|
| nominal (단독, 실행 a) | **0 / 16** | 0.2735 | 900 |
| nominal (단독, 실행 b) | **0 / 16** | 0.2735 | 900 |
| nominal (스위트 안) | 0 / 16 | 0.2706 | 900 |
| light_intensity | 0 / 16 | 0.2922 | 900 |
| light_direction | 0 / 16 | 0.2998 | 900 |
| observation_delay | 0 / 16 | 0.2940 | 900 |
| torque_noise | 0 / 16 | 0.5132 | 900 |
| backlash | 0 / 16 | 0.2003 | 900 |

두 nominal 실행이 마지막 자리까지 일치하므로 이 수치는 스케줄이 아니라 정책의 것이다. 모든
에피소드가 900 스텝 예산을 다 쓴다. 처음 여섯 nominal 셀의 `.estraj` 기록을 읽으면, 팔은 가장 크게
움직이는 관절에서 **1.66 – 1.69 rad** 움직이는데 큐브는 여섯 개 모두 **정확히 시작 자세**로 끝나고
그리퍼는 활짝 열린 채다. 정책은 파지를 놓치는 것이 아니라 큐브에 도달하지 못한다.

외부 ACT, 새 내보내기에 V8의 파이프라인을 그대로(`es dataset export --lerobot-v3
--drop action_commanded,action_source --state-dim 6` → `lerobot-train --policy.type=act
--steps=100000 --batch_size=8 --seed=0`, 2,033 s → `es policy import-lerobot` →
`es eval run`): **nominal 0 / 16**, `envelope_violation_rate` 0.0523, 모든 에피소드 900 스텝.
외부 ACT의 6개 스위트 스윗은 쓰지 **않았다**. 중단 규칙은 nominal 수치에서 발동하고, V8의
0/16 · 0/16 · 1/16도 그 수치이며, 섭동 없이도 0점인 정책의 섭동 스윗은 아무것도 측정하지 않는다.

| | V6 (200 Hz) | V8 (200 Hz) | **V11 (50 Hz)** |
|---|---|---|---|
| IR 소유 그래프, 20,000 스텝, nominal | 0 / 16 | — | **0 / 16** |
| LeRobot ACT, 20,000 / 50,000 / 100,000 스텝, nominal | — | 0/16 · 0/16 · 1/16 | 100,000에서 **0 / 16** |

**벽시계**(처리량 주장이 아니라 관측 — §12.4): 프레임 포함 50 에피소드 수집 50 s, 베이크 1 s,
lower + 20,000 스텝 학습 654 s, 프레임 포함 6개 스위트 평가(96 셀, 렌더 프레임 86,400개,
`--jobs 6`) 393 s, `lerobot-train` 100,000 스텝 2,033 s, 16셀 nominal 실행 하나는 경합에 따라
10 – 25분.

**판정.** 속도는 틀렸었고 이제 맞다. 제어 한 스텝이 제어 주기 하나이고, 엔벌로프는 자신이
제한하는 스텝에 대해 측정되며(전문가 위반율 0.48–0.55 → 0.2044), 청크는 실제 운동 320 ms이고,
데이터셋의 `fps`는 참이다. 속도 수정은 다른 무엇도 찾지 못한 것을 찾았다. **플랜 V가 수집한 모든
시연은 구동된 것이 아니라 끌려다닌 팔이 만든 것이다.** Deployment IR이 장면의 2.94 N·m
액추에이터로 제동할 수 없는 가속도를 선언하고 있고, 200 Hz에서는 팔이 속도 포화 상태라 추종할 수
없는 것을 지나칠 수도 없었기 때문이다. 그것을 고치자 — 전문가가 이제 엔벌로프의 절반에 맞춘다 —
시연 50/50, 전문가 게이트 두 개 모두 8/8, 위반율은 V6의 절반 이하로 돌아왔다.

**그리고 어느 정책도 여전히 배우지 못한다. 0/16과 0/16, V6의 0/16과 V8의 1/16에 대해.** 중단
규칙이 발동한다. 수정은 학습 문제를 오히려 *거칠게* 만들었는데 — 스텝당 17이 아니라 40 mrad,
청크당 1/5가 아니라 0.5 라디안 — 결과는 움직이지 않았다. 잘못된 것이 무엇이든 제어 속도도,
시연도(50/50, 재생과 프로브로 확인), 파지도(122 mm 들어올림, 양 집게, 에피소드의 절반), 앙상블도,
모델도(V8 자신의 ACT, 100,000 스텝) 아니다. **여기서 두 번째 변수를 움직이지 말 것.** 다음 패킷은
가설 하나와 변수 하나 — 해상도 또는 시연 개수 — 를 받고, 비교 기준은 V11의 수치다.

### 7.20 구현 결과 (V12): 정책이 보고 행동하는 관측이 곧 시연이 기록하는 관측

패킷 `docs/packets/M5/V12-observation-pairing.ko.md`. 7.19절은 용의자 넷을 닫고도 "그래도 두
정책 모두 배우지 못한다"로 끝났다. 이 패킷은 다섯 번째를 닫는다. 그것은 내내 기록기 안에
있었다.

**결함.** `Env::step`은 `set_ctrl(ctrl)` -> `backend.step(substeps)` -> `let state =
self.backend.state()` -> `StepRow { qpos/qvel/sensordata: <스텝 이후>, ctrl }` 순서였다. 그래서
시연의 행 `t`는 `observation.state[t]` — `action[t]`가 **실행된 뒤의** 상태 — 를 `action[t]`와
짝지었고, `Collector::run`도 같은 스텝-이후 상태에서 프레임을 렌더하고 `.estraj` 자세를 밀어
넣었다. 그러나 전문가는 스텝 *이전* 상태에서 `action[t]`를 계산했고(`step_with_policy`: 관측 ->
추론 -> 행동 -> 스텝), `es_eval::runner`도 추론에서 똑같이 한다. **학습은
`(s_{t+1}, image_{t+1}) -> a_t`를 배우는데 추론은 `(s_t, image_t) -> a_t`를 묻는다** — 플랜 V가
수집한 모든 시연의 모든 입력에 제어 주기 하나만큼의 지연이 있었다. LeRobot의 관례는 반대다:
`observation[t]`는 `action[t]`를 고른 근거다. V10은 이것을 보고 잡음으로 읽었는데, 당시로서는
옳았다. 7.19절이 고친 200 Hz에서 제어 한 스텝은 5 ms, 0.7 mrad였다.

**수정은 한 곳이다.** `Env::step`이 `backend.step` 전에 `qpos/qvel/sensordata`와 틱을 스냅샷해
방금 적용한 `ctrl`과 함께 *그것을* 기록한다. 보상·종료·실패는 전이의 것으로 남으므로 `done`은
여전히 마지막 프레임을 표시한다. 수집기·평가기·HIL·임베디드 런타임 등 모든 소비자가 env에서
이를 상속한다. `Collector::run`은 프레임 렌더와 `.estraj` 푸시를 같은 순간, 스텝 이전으로 옮겨
이미지·상태 행·궤적이 한 상태의 한 번 읽기가 되게 한다. 덕분에 아무도 이름 붙이지 않았던 더
작은 결함도 사라졌다. 종료 스텝은 `Env::step` 안에서 env를 자동 리셋하므로, 수집기의 스텝-이후
읽기는 모든 에피소드의 마지막 프레임에 **다음 에피소드의 리셋 상태**를 렌더하고 있었다.
`es_eval::runner`는 고칠 것이 없었다 — 이미 옳았고, 그것이 요점이다.

**측정이 정확히 한 행만큼 뒤집힌다.** 각 패킷 자신의 50 에피소드 baked 세트에서(양쪽 9,038
프레임) `max_j abs(action[t] - qpos[t+k])`의 프레임 중앙값:

| `k` | V11(스텝 이후 행) | V12(스텝 이전 행) |
|---|---|---|
| -1 | 0.06159 rad | 0.11149 rad |
| 0 | **0.02094 rad** | 0.06159 rad |
| +1 | 0.06753 rad | **0.02137 rad** |

V12의 `k = 0`은 V11의 `k = -1`과 소수점 다섯 자리까지 같다. 이제 정책이 매핑해야 할 행에는 명령
선행(0.062 rad)이, `t+1`에는 추종 오차(0.021 rad)가 있다. 그 밖에는 움직이지 않았다: 시연
50/50, `recorded_actions_replay_to_the_same_outcome` 두 행동 열 모두 50/50, V9 재생 900 프레임
비트 동일, `dataset_schema_hash` 불변.

**결과, 그리고 중단 규칙.** V2의 노브 그대로 재학습(20,000 스텝, 손실 0.0583 -> 0.0141)한 IR
소유 그래프는 **nominal 0/16(두 번, 비트 동일)이고 자기 학습 시드 1-16에서도 0/16**이다. V11의
0/16, 0/16과 같다. 움직인 것은 엔벌로프다: 위반율 nominal **0.2735 -> 0.0371**, 학습 시드
**0.3152 -> 0.0365**, position 위반은 3,402에서 6으로, fallback은 0으로 떨어졌다. 지연을 없애자
정책의 명령이 그 명령을 낸 상태에서 도달 가능해졌다 — 다만 큐브에 도달하지 못할 뿐이다. 중단
규칙은 학습 시드 수치에서 발동한다: 시드 1이 *곧* 학습 에피소드 0이다.

**그리고 이유는 이제 측정되었다.** 오케스트레이터의 개루프 분석(`~/artifacts/plan-v/v11/
openloop/`, V12 체크포인트로 재실행)이 페어링과 무관하게 이유를 찾았다. `train_act.py --batch 8`
은 **단일 샘플** 순전파 여덟 번을 누적하므로 낮춰진 ResNet18의 모든 `BatchNorm2d`가 `N = 1`
통계로 학습되는데, 추론은 `model.eval()`로 러닝 통계를 쓴다. V12 체크포인트, 학습 에피소드
3개의 청크 L1: **`train()`에서 0.0129 / 0.0127 / 0.0130**(= 보고된 손실) 대
**`eval()`에서 0.0337 / 0.0438 / 0.0441**(= `es eval run`이 실행하는 경로), "현재 자세 유지"
기준선은 0.0565 / 0.0571 / 0.0561. 배포된 적합은 학습된 적합의 세 배이고, 학습에 쓴 데이터에서도
아무것도 하지 않기보다 4분의 1만 낫다. 이 간극은 페어링 수정 앞뒤로 같고, 개루프 표에는 더 이상
지연의 흔적이 없다. 청크의 0행이 `action[t]`에 가장 가깝고 평가기가 실행하는 블렌드는 이제
정렬된 쪽이다. 이것이 **미해결 질문 19**다.

**외부 ACT가 대조군이고, 그 역시 0/16에 머문다.** 새 내보내기에 V8 파이프라인 그대로(100,000
스텝, 2,175 s): `success_rate` **0/16**. V8의 200 Hz 1/16, V11의 0/16과 견준다.
`envelope_violation_rate`는 IR 그래프와 *반대* 방향으로 움직였다(0.0523 -> 0.4267). LeRobot의
ACT에는 train/eval 정규화 간극이 없으므로(`FrozenBatchNorm2d`, 진짜 배치) `BatchNorm`이
이야기의 전부일 수도 없고, 다음 패킷은 미해결 질문 19 옆에 이 수치도 함께 마주해야 한다.

페어링 수정은 그와 무관하게 그 자체로 선다. 그것이 LeRobot이 말하는 시연이고, 평가기와 모든
배포 경로가 이미 가정하던 바이며, 다음 패킷이 물려받는다.

### 7.21 구현 결과 (V13): 배포된 함수가 학습된 함수가 아니었다

패킷 `docs/packets/M5/V13-groupnorm-backbone.ko.md`. 로워링 규칙:
`docs/design/learning-lowering.ko.md` 5.1절.

플랜 V의 IR 소유 비전 정책은 전부 작은 loss를 보고한 뒤 폐루프에서 0점을 받았다. V11이 자기
체크포인트를 자기 학습 에피소드 위에서 개루프로 분석해 이유를 찾았고, 그것은 드리프트도,
블렌드도, 페어링도 아니었다: **`train()`과 `eval()`이 서로 다른 함수였다.**

**메커니즘.** `lower_to_torch`는 `VisionEncoder { ResNet18, pretrained: false }`를 torchvision의
`resnet18()`로 내렸고, 그 기본 정규화 층이 `BatchNorm2d`다 — 20개, 그리고 그래프 전체에서 running
통계를 가진 유일한 모듈이다(`n1`/`n2`/`n4`/`n6`은 `Linear`, `n3`은 `TransformerEncoder`, 즉
LayerNorm). 이 로워링은 구조적으로 단일 샘플이다: spec 8.3의 포트에는 배치 축이 없고 — spec 5.2가
추론 도메인에 자기 배치 크기를 준다 — 이미지는 `.unsqueeze(0)`을, 헤드는 `.reshape(16, 6)`을
거친다. 그래서 `python/es/train_act.py --batch 8`은 단일 샘플 forward 여덟 번을 누적하고, 모든
`BatchNorm2d`가 **N = 1** 통계를 적합했다: 배치 카운터가 달린 instance normalization이다. 그 뒤
`crates/es-policy/python/torch_ref.py:85`가 `model.eval()`을 호출해, 학습이 한 번도 쓰지 않은
running 평균을 끼워 넣는다.

| V11 체크포인트, 자기 학습 에피소드 위의 10행 chunk L1 | |
|---|---|
| `train()` — 학습이 최소화한 것, loss 곡선이 보고한 것 | **0.011** |
| `eval()` — `es eval run`이 실제로 실행한 것 | **0.031 – 0.039** |
| 9,038 프레임 전체를 배치 64로 돌려 running 통계를 재보정한 `eval()` | 0.029 |
| 기준선: 모든 행에서 현재 자세 유지 | 0.048 |

결론을 내리는 행은 재보정 행이다. 가중치 후처리가 할 수 있는 최선 — 배치 forward가 그
BatchNorm들에 건넸을 바로 그 텐서로 학습 세트 전체에 155번 갱신 — 인데 간격의 5분의 1만 좁힌다.
고칠 자리는 로워링이다.

**결정**(아키텍트): `pretrained: false`에 대해 `norm_layer = lambda c: nn.GroupNorm(32, c)`로
내린다. GroupNorm은 눈앞의 샘플 하나를 채널 그룹 단위로 정규화한다. `training` 분기도 running
버퍼도 없으므로 두 모드가 비트 단위로 같은 함수가 되고, 단일 샘플 로워링은 그대로 유효하며,
running 통계 버퍼가 가중치 계약에서 사라진다. Diffusion Policy가 바로 이 이유로 ResNet에 끼워
넣는 것과 같고, 그룹 32는 ResNet의 모든 stage 폭을 나눈다. 이것은 **로워링** 결정이므로 —
§8.3은 정규화를 못 박지 않는다 — `lowering_hash`와 `compiler_hash`가 움직이고 `learning_hash`는
움직이지 않는다. `pretrained: true` 경로는 건드리지 않았다: 여기서는 여전히 거부되고, LeRobot
체크포인트는 `lerobot.rs`를 통해 ImageNet의 BatchNorm을 `FrozenBatchNorm2d`로 유지한다(V8).
고정된 affine 상수에도 학습 모드가 없으므로 그쪽이 맞다.

선언된 가중치 키는 움직이지 않았다 — backbone은 접두사 claim `nodes.0.*`이고 14개 키 전부 V11과
바이트 단위로 같다 — 따라서 바뀐 것은 claim *아래의* 텐서 집합이다: 체크포인트가 146개에서
**86개**로 줄었고, 사라진 60개가 `running_mean`/`running_var`/`num_batches_tracked` × 20이다.

**측정.** V11의 레시피를 V11의 baked 세트 위에서, 변수 하나만 움직여서:
`es policy lower`(`lowering_hash` `fdd68ec4…` → `70a8fec7…`, 가중치 키 불변) →
`train_act.py --batch 8 --lr 1e-4 --seed 0 --device cuda --resident-gpu` 20,000 스텝 →
`es policy pack`.

| | V11 (BatchNorm) | **V13 (GroupNorm)** |
|---|---|---|
| 학습 loss, 초기 → 최종 | 0.0565 → 0.0132 | 0.0607 → **0.0147** |
| 벽시계, 20,000 스텝, RTX 4090(공유) | 654 s | 719 s |
| 체크포인트 텐서 수 | 146 | **86** |
| 모듈 안의 running 통계 버퍼 | 60 | **0** |
| 개루프 chunk L1, `eval()` — 추론 경로 | 0.0313 / 0.0373 / 0.0385 | **0.0132 / 0.0135 / 0.0133** |
| 개루프 chunk L1, `train()` — loss가 보고한 것 | 0.0107 / 0.0122 / 0.0111 | 0.0136 / 0.0139 / 0.0139 |
| 기준선: 현재 자세 유지 | 0.0484 / 0.0488 / 0.0481 | 0.0484 / 0.0488 / 0.0481 |
| 고정 입력에서 backbone만의 `max abs(train() − eval())` | — | **0**(비트 단위 동일) |

두 행이 수렴했다: `eval()`이 이제 보고된 loss의 세 배가 아니라 *보고된 loss와 일치*하고
(0.0147에 대해 0.0133), `train()` 쪽이 아주 약간 더 나쁜데 이는 dropout이 예측하는 바다. 남은
모듈 전체 차이 0.0169는 전부 dropout이다 — `nn.TransformerEncoderLayer`의 `nn.Dropout` 멤버 셋과,
모듈이 아니라 `self.training`을 읽는 float인 `MultiheadAttention`의 attention dropout. 넷을 모두
0으로 두면 모듈 전체가 두 모드에서 비트 단위로 같고, 그것이 `identity.py`의 단언이며 오라클
`the_backbone_computes_the_same_function_in_train_and_eval`이 그 Python 쪽 절반이다. dropout은
의도된 정규화이고 안전한 방향으로 작동한다(추론은 기댓값을 실행한다). 그래서 V11의 0.011은 오히려
*낙관이 아니라 비관*이었다: 3배 간격은 전부 정규화 탓이었다.

**폐루프: 팔이 이제 큐브까지 도달하고, 그래도 과제를 풀지는 못한다.**
프레임을 켜고 `--jobs 6`으로 `es eval run`, V11 자신의 두 스위트 — 학습 시드 1–16과 홀드아웃
시드 101–116:

| | V11 (BatchNorm) | **V13 (GroupNorm)** |
|---|---|---|
| `success_rate`, 학습 시드 1–16 | 0 / 16 | **0 / 16** |
| `success_rate`, 홀드아웃 시드 101–116 | 0 / 16 | **0 / 16** |
| `envelope_violation_rate`, 학습 / 홀드아웃 | 0.3152 / 0.2735 | **0.5770 / 0.6244** |
| `episode_length` | 900(모든 에피소드) | 900(모든 에피소드) |
| 가장 넓게 움직인 관절 이동량, 앞 6 에피소드 | 1.667 – 1.696 rad | 1.515 – 1.623 rad |
| **큐브가 조금이라도 움직인 에피소드** | **0 / 16, 0 / 16** | **14 / 16, 14 / 16** |
| 홀드아웃 에피소드별 큐브 변위 | 16개 전부 0.4 mm(정착) | 0.4 – 200.2 mm, 중앙값 약 9 mm |

마지막 두 행이 결과의 전부다. V11의 정책은 팔을 휘두르고 **큐브에 도달한 적이 없었다** — 두
스위트 모든 에피소드의 0.4 mm는 리셋 때 큐브가 테이블 위에 정착하는 양이다. V13의 정책은 두
스위트 모두 16개 중 14개에서 도달해서 큐브를 밀고, 두 번은 190 mm 넘게 민다. 이 수정은 "정책이
도달하지 못한다"를 "정책이 도달하고 손을 닫지 못한다"로 바꿨다. 다른 종류의 실패이고, 플랜 V의
IR 소유 비전 정책이 만들어낸 첫 폐루프 운동이다. 대신 엔벌로프 여유를 썼다: 홀드아웃 스위트의
`violation.position`이 2,881에서 7,466으로 늘었고, 이제 큰 움직임을 명령하는 정책을 Safety
Plane이 클램프하고 있다.

**V12의 bake에서는 큐브가 테이블을 떠난다.** V12의 페어링 수정 50 Hz 세트(7.20절)가 제때
완성돼 있어서 같은 모듈과 같은 하이퍼파라미터로 여기에도 학습했다: loss 0.0632 → 0.0162, 개루프
`eval()` 0.0137 / 0.0144 / 0.0149 대 `train()` 0.0142 / 0.0148 / 0.0154 — 같은 수렴이고,
"현재 자세 유지" 기준선이 0.0484가 아니라 0.0565인 세트에서다. 페어링 수정이 목표를 실제로 더
어렵게 만들었다는 뜻이다.

| 학습 데이터 | 학습 시드 1–16 | 홀드아웃 101–116 | `envelope_violation_rate` | `episode_length` |
|---|---|---|---|---|
| V11의 bake | 0 / 16 | 0 / 16 | 0.5770 / 0.6244 | 900 / 900 |
| **V12의 bake** | **1 / 16** | 0 / 16 | 0.5396 / 0.6392 | 847.75 / 900 |

저 `1 / 16` 앞에 궤적을 먼저 읽어야 한다. **학습 시드 세 에피소드(03, 11, 15)에서 정책은 큐브를
잡고 118 – 124 mm 들어 올려 통까지 옮긴다** — (0.25, 0.00, 0.02) → (0.13, −0.10, 0.06 … 0.14) —
그리고 900스텝 예산이 다할 때까지 들고 있어서 셋 다 `timeout`으로 채점된다. 하네스가 `success`로
채점한 한 에피소드(02, 틱 64 종료)는 큐브를 1.6 mm 들었을 뿐 운반이 아니며, 따로 볼 문제이지 이
절이 주장하는 바가 아니다. 주장은 세 번의 운반이다: 플랜 V에서 IR 소유 비전 정책이 큐브를 집어
든 첫 사례다. 그중 하나를 렌더한 결과가
`~/artifacts/plan-v/v13/v12/v13-carry-nominal-03-h264.mp4`이다(`es video showcase --cell
nominal-03`, 900틱, 1280x720, 0.9 MiB). 홀드아웃에서 성공한 에피소드는 없다.

**판정과 중단 규칙.** 학습/배포 간극은 실재했고, 측정됐고, 닫혔다: backbone에 대해 `train()`과
`eval()` 차이 0, `eval()` 개루프 L1이 2.4배 내려와 보고된 loss와 만난다. 홀드아웃
`success_rate`는 두 bake 모두 여전히 0/16이므로 중단 규칙이 발동하고 여기서 두 번째 변수는
움직이지 않는다. 수치가 다음으로 가리키는 것은 도달이 아니라 **놓기**다: 행 5(그리퍼)는 모든
개루프 표에서 잔차가 가장 큰 관절이고 — 블렌드 0.0162 대 팔 관절 0.0032 – 0.0103, 현재 자세
대비 0.0823 — "큐브를 통 위까지 옮겨놓고 열지 않는다"가 바로 그것이 예측하는 바다. 두 번째로
볼 것은 900스텝 예산이다: 세 에피소드는 예산이 다할 때 큐브를 올바른 자리에서 들고 있었다.
여기서부터의 비교 기준은 V11이 아니라 V13의 수치다.


### 7.22 구현 결과 (V14): 손은 끝내 열리지 않았고, 시연 4배도 열지 못했다

패킷 `docs/packets/M5/V14-release-and-demos.md`. 질문 둘, 각각 하나씩이며 두 번째가 V13의 중단
규칙이 남겨 둔 단 하나의 변수다.

**1부: 시간을 더 주면 정책은 놓는가?** V13에서 큐브를 옮긴 세 에피소드는 900스텝 예산이 끝나는
순간까지도 여전히 통 위에서 큐브를 들고 있었다. 즉 예산이 답을 가리고 있었다. `max_episode_steps`는
생성되는 값이고(`regenerate_visible_learning_documents`가 소유한다) 따라서 바뀐 것은 생성기의
상수 900/18.0 s → **1800/36.0 s**(`Timeout` 리프가 경과 초를 비교하므로 둘이 함께 움직인다)와
재생성뿐이다. 그 다음 V13의 `model-20000.safetensors`를 재생성된 문서로 만든 번들에 다시 담았다.
**가중치 바이트는 그대로다**(텐서 86개, 동일한 `weights_hash`, 동일한 `lowering_hash`
`70a8fec7…`). `learning_hash` `5dac0a46…`도 그대로이고, `task_hash` `d7a7c061…` → `78814eb4…`,
`observation_hash` `4b069f6c…` → `72b8609a…`, `policy_hash` → `8dbff824…`는 모두 움직인다.
Task IR 값 하나가 바뀌었으니 해시 체인이 제 일을 한 것이다(§5.3).

| 예산 1,800, nominal 스위트 | 학습 시드 1–16 | 홀드아웃 101–116 |
|---|---|---|
| `success_rate`(하네스) | 1 / 16 | 0 / 16 |
| 큐브가 테이블에서 떨어짐 | 3 / 16 | 3 / 16 |
| **큐브를 통 내부 부피까지 옮김** | **3 / 16** — 03 @ 917, 11 @ 147, 15 @ 172 | **2 / 16** — 01 @ 143, 04 @ 142 |
| **통 안에서 큐브를 놓음** | **0 / 16** | **0 / 16** |
| 1800틱 시점에 큐브가 여전히 통 안 | 3 / 16 | 2 / 16 |
| `envelope_violation_rate` | 0.4227 | 0.5457 |
| `episode_length` | 1691.5 | 1800.0 |

옮긴 에피소드는 모두 남은 구간 전체 — 883 ~ 1,653 제어 스텝, 17.7 ~ 33.1 s — 동안 큐브를 통
내부에 든 채로 유지하며, 그리퍼 관절은 0.078 ~ 0.093 rad, 즉 30 mm 큐브를 문 채 멈춘 턱 값이다
(7.19절의 측정값 0.0934). **단 한 에피소드도 손을 열지 않는다.** 18초를 더 줘도 두 스위트 어느
쪽의 판정도 바뀌지 않았다. 그 대가로 얻은 것은 이 유지가 늦은 것이 아니라 무한하다는 사실이다.
유일한 `success`는 V13이 이미 부인한 그 에피소드다: 학습 시드 3, 64틱, 1.8 mm 들어 올림, 운반이
아니다.

**`violation.position`은 그리퍼이고, 오직 그리퍼다.** `events.json`은 프레임당 `ViolationKind`
비트셋만 담고 관절 인덱스는 담지 않으므로, 이 분해는 `.estraj`의 위치를 Deployment IR의 소프트
경계 — `[lower + margin, upper - margin]`, `SafetyPlane::clamp` 2단계가 *명령*을 자르는 바로 그
구간 — 에 대해 재생해서 유도했다.

| 관절 | 학습 시드, 소프트 경계 밖 틱 | 홀드아웃 |
|---|---|---|
| `shoulder_pan` / `shoulder_lift` / `elbow_flex` / `wrist_roll` | 0 | 0 |
| `wrist_flex` | 23 (0.1%) | 18 (0.1%) |
| **`gripper`** | **27,064 중 10,145 (37.5%)** | **28,800 중 14,690 (51.0%)** |
| 하네스 자신의 `violation.position` | 9,265 | 13,380 |

그리퍼의 소프트 구간은 `[-0.1245, 1.6953]` rad이고, 고정된 틱들은 **하한**에서 1 mrad 이내에
있다. 정책은 에피소드의 절반 동안 봉투가 허용하는 것보다 더 세게 턱을 닫으라고 명령하고 안전
평면이 거기서 자른다 — "절대 열지 않는다"는 사실을 평면 쪽에서 본 것과 같다. 유도이므로 단서를
붙인다: `.estraj`는 상태만 기록하므로 이 수치는 잘린 명령의 경계에 *측정된* 위치가 도달한 틱을
세는 것이지 클램프 사건 자체가 아니다. 두 수치는 두 스위트 모두에서 10% 이내로 일치하고 나머지
관절은 전부 0이며, 기록된 데이터가 말할 수 있는 최대치가 이것이다.

**2부: 시연 50개 → 200개.** V12의 명령 그대로, 50 Hz, `PACE = 0.5` 전문가, `--episodes 200
--seed 1` — 50을 넘는 시드는 같은 `Randomization` 스트림의 새 추출이지 반복이 아니다 — 그다음
베이크, 그다음 V2의 손잡이로 V13 그래프 학습.

| | V12 / V13 (50) | **V14 (200)** |
|---|---|---|
| 시연 | 50 / 50 `Success` | **200 / 200 `Success`** |
| 프레임 | 9,038 | **35,918** |
| 학습 손실, 초기 → 20,000 → 40,000 | 0.0632 → 0.0162 → — | 0.1167 → 0.0186 → **0.0148** |

20,000에서도 손실이 계속 떨어지고 있었으므로 40,000도 기록했고 그쪽이 최종 체크포인트다. 평가는
1,800이 아니라 **900** 예산이다. 1부의 답이 "1,800에서도 놓지 않는다"이므로 긴 예산은 아무것도
사지 못하고, 900이 V13의 표가 서 있는 예산이기 때문이다.

| nominal 스위트, 예산 900 | V13, 50개, 20k | V14, 200개, 20k | **V14, 200개, 40k** |
|---|---|---|---|
| `success_rate`, 학습 시드 1–16 | 1 / 16 | 0 / 16 | **1 / 16** |
| `success_rate`, 홀드아웃 101–116 | 0 / 16 | 0 / 16 | **0 / 16**(두 번, 동일) |
| 들어 올림, 학습 / 홀드아웃 | 4 / 16 · — | 8 / 16 · 6 / 16 | **9 / 16 · 9 / 16** |
| **통까지 운반, 학습** | **3 / 16** | 0 / 16 | **8 / 16** |
| **통까지 운반, 홀드아웃** | **0 / 16** | 2 / 16 | **9 / 16** |
| **통 안에서 놓음** | 0 / 16 | 0 / 16 | **0 / 16** |
| `envelope_violation_rate`, 학습 / 홀드아웃 | 0.5396 / 0.6392 | 0.5017 / 0.6120 | **0.1587 / 0.2060** |
| `violation.position`, 학습 / 홀드아웃 | 5,655 / 7,542 | 5,690 / 7,222 | **657 / 1,357** |
| `episode_length`, 학습 / 홀드아웃 | 847.75 / 900 | 900 / 900 | 855.125 / 900 |

40,000 체크포인트의 홀드아웃 두 번은 마지막 자리까지 일치하므로 이 숫자는 스케줄이 아니라 정책의
것이다. 20,000 열은 남겨 둘 가치가 있다. 데이터가 4배가 되면 20,000 스텝은 *부족* 학습이다 —
들어 올리기는 40,000과 비슷하게 하지만 운반은 거의 못 한다 — 즉 "20,000 스텝"은 레시피의 성질이
아니라 시연 50개 집합의 성질이다.

**측정된 벽시계**(처리량 주장이 아니라 관측 — §12.4): 프레임 포함 200 에피소드 수집 208 s, 베이크
8 s(35,918 프레임, 3.8 GB), 로워링 + 40,000 스텝 학습 1,411 s(평가와 공유한 RTX 4090), 프레임
포함 16 에피소드 nominal 스위트 1회는 900 예산에서 265 s, 1,800 예산에서 470 ~ 620 s.

**판정.** 시연 4배는 플랜 V가 측정한 단일 변수 이동 중 가장 크다. 홀드아웃 운반이 **0 / 16에서
9 / 16으로** 올라가고, 봉투 위반율은 3분의 1로, `violation.position`은 7,542 프레임에서 1,357로
떨어진다. 한 번도 본 적 없는 시드의 절반 이상에서 정책은 큐브를 찾고, 쥐고, 약 125 mm 들어 올려
통 위에 올려놓는다. **그런데도 하네스는 여전히 0 / 16이다. 과제에 마지막으로 필요한 한 가지가
정책이 한 번도 해 본 적 없는 그 한 가지 — 손을 여는 것 — 이기 때문이다.** 두 부를 합쳐 평가한 96
에피소드에서, 900 스텝에서도 1,800 스텝에서도 놓음은 0이다. 홀드아웃 `success_rate`는 바뀌지 않은
수용 기준 0.5 아래이므로 중단 규칙이 발동한다: 6-스위트 스윕 없음, 쇼케이스 영상 없음, 여기서 두
번째 변수 없음. 남은 것은 도달도, 파지도, 일반화도, 예산도 아니다. 에피소드 내내 닫으라고 명령되고
거기서 잘리는 관절 하나다.

### 7.23 구현 결과 (V15): 놓기는 데이터 안에 있었고, 길이는 제어 스텝 14개였다

패킷 `docs/packets/M5/V15-release-in-the-data.ko.md`. 이 패킷을 열게 만든 가설 — 기록된 시연이
전문가의 `Release` 단계 **이전에** 끝나서 정책이 손이 열리는 장면을 본 적이 없다는 것 — 은
**거짓**이고, 그것을 반증한 측정이 이 패킷의 쓸모 있는 부분이다.

**1부 a — 모든 시연에 놓기가 들어 있고, 그 첫 제어 스텝 14개만 들어 있다.** V14의 200개 시연
`ds-train`을 에피소드마다 기록된 `action` 열과 옆에 붙은 `.estraj`로 읽었다. "꼬리"는 턱을
닫으라고 명령한 마지막 프레임 뒤에 오는 프레임들이다.

| 시연당, 200개의 중앙값 | V14, 임계값 0.60 |
|---|---|
| 에피소드 길이 | 178 프레임 |
| 꼬리: 마지막 닫힘 명령 뒤 프레임 수 | **14** |
| 그중 0.30 rad 초과로 명령된 프레임 | 8 |
| 그중 0.60 rad 초과로 명령된 프레임 | **1** |
| 꼬리에서 그리퍼 명령의 최댓값 | **0.614** rad (`grip_open` = 0.9 중) |
| 마지막 프레임의 측정된 턱 | 0.573 rad |
| 마지막 프레임의 큐브 z | 0.0276 m — 공중이 아니라 통 바닥에 놓여 있음 |
| 마지막 프레임에 큐브가 3차원 통 부피 안 | 200 / 200 |

가설이 묻는 방식 그대로 — "에피소드가 끝나기 전에 그리퍼 명령이 정지값을 넘은 적이 있는가?" —
물으면 답은 200 / 200이고 에피소드당 74프레임쯤인데, 그것이 틀린 질문이다. 그런 첫 프레임은 9틱
이다: 전문가는 `Approach`와 `Descend` 동안, 즉 *큐브로 내려가는 길에* 턱을 0.9로 활짝 연 채
53프레임쯤 유지한다. 학습 데이터에서 "턱이 활짝 열려 있다"는 압도적으로 접근 구간이다. 놓기는
명령이 `grip_closed`에서 올라가다가 목표인 0.9에 닿지 못한 채 0.614에서 잘리는 14프레임 꼬리다.

**왜 잘리는가.** 에피소드는 Task IR 자신의 `Terminate`로 끝나고, `Terminate`는 술어가 참이 되는
첫 틱에 발화하며, 술어의 그리퍼 항은 `그리퍼 qpos > 0.6`이었다. 전문가의 개방 램프는 다른 모든
관절과 똑같이 `step_max`로 페이싱되고(`ExpertCfg::pace_to`, Deployment IR 엔벨로프의
`PACE = 0.5` — 7.19절), 측정된 턱은 명령보다 늦다. 그래서 0.6을 넘는 데 제어 스텝 14개, 0.85를
넘는 데 19개가 걸린다. 임계값은 "놓았다"를 얼마나 엄격하게 볼지를 고르고 있던 것이 아니라
**놓기의 얼마만큼이 데이터에 남을지**를 고르고 있었다.

**1부 b — 하네스는 이미 놓인 큐브만 세고 있었다.** 그리퍼 항은 V1부터 술어에 있었고
(`crates/es/tests/cli.rs`의 `add_gripper_open_term`), 열린 질문 21이 그것을 적고 있다. V14의
0 / 16은 그 항이고 그것뿐이다. 1,800 스텝 예산에서 틱 단위로:

| 1,800틱 중 만족하는 틱 수 | 홀드아웃 01 | 홀드아웃 04 | 학습 03 | 학습 11 | 학습 15 |
|---|---|---|---|---|---|
| 큐브 x in (0.09, 0.19) | 1,691 | 1,697 | 920 | 1,688 | 1,668 |
| … 그리고 `|vx| < 0.05` | 1,676 | 1,679 | 904 | 1,671 | 1,652 |
| 그리퍼 qpos > 0.6 | 45 | 39 | 779 | 50 | 70 |
| **다섯 항 전부** | **0** | **0** | **0** | **0** | **0** |

정착 항은 접촉 지터에 지지 않는다 — 에피소드의 93 %에서 성립하고, 큐브의 x는 30초 동안 1 mm도
움직이지 않는다 — 그리고 x 구간은 3차원 `cube_in_the_bin` 검사가 쓰는 바로 그 구간이다. 그리퍼가
열려 있는 틱들은 큐브가 구간에 들어오기도 전인 접근 구간이다. 막는 항은 그리퍼이고, 닫히는
순간부터 0.078 ~ 0.093 rad이다.

**2부 — 술어, 전과 후.** 구조는 움직이지 않는다: 이미 있는 `GetJointState` 리프와 `Compare`
노드 위의 다섯 항을 `And`로 묶은 그대로다. 상수 하나가 움직인다.

| | 전 (V1 ~ V14) | 후 (V15) |
|---|---|---|
| 통의 x 구간 | `큐브 x > 0.09` **and** `큐브 x < 0.19` | 그대로 |
| 정착 | `큐브 vx > -0.05` **and** `큐브 vx < 0.05` | 그대로 — 1부 b가 이것이 실패 원인이 아님을 보인다 |
| 손이 열려 있다 | `그리퍼 qpos > 0.6` | **`그리퍼 qpos > 0.85`** |

0.85는 같은 것을 더 엄하게 읽는 값이 아니라 다른 것을 읽는 값이다. 30 mm 큐브를 문 턱은 0.09를
읽고, *열리는 중이지만 아직 큐브에 닿아 있는* 턱은 0.6 언저리까지도 읽는다. 0.85에 닿을 수 있는
턱은 `grip_open`으로 명령되었으면서 **비어 있는** 턱뿐이다. 따라서 에피소드는 놓기의 첫 밀리미터가
아니라 끝난 놓기에서 끝나고, 시연이 기록하는 것은 램프 전체와 열린 손 아래 정지한 큐브가 된다.
`task_hash` `78814eb4…` → **`eb6efefa…`**, `observation_hash` `72b8609a…` → **`899c16a9…`**.
`deployment_hash`는 움직이지 않고 `lowering_hash`(`70a8fec7…`)도 그대로다.

의도가 아니라 결과를 못 박는 오라클이 둘이다. `expert_solves_the_pinned_seeds`는 이제 모든 시연의
꼬리에서 개방 명령 개수를 세고 `RELEASE_FRAMES = 5` 미만이면 실패한다 — `THRESHOLD`와 같은 종류의
못이고, 옛 임계값에서 그 값은 1이었다. `the_demo_task_cones_lower`는 로워링된 성공 `Expr`을 큐브를
문 턱, 열리는 중이지만 아직 벗어나지 않은 턱, 놓은 턱에 대해 백엔드 없이 평가한다.

**그것이 데이터에 사 준 것.** 같은 200개 시드, 같은 전문가, 같은 페이스, 양쪽 모두 200 / 200
`Success`:

| 시연당, 200개의 중앙값 | V14, 0.60 | **V15, 0.85** |
|---|---|---|
| 에피소드 길이 | 178 | **183** |
| 마지막 닫힘 명령 뒤 꼬리 | 14 | **19** |
| 그중 0.30 초과 명령 | 8 | **13** |
| 그중 0.60 초과 명령 | 1 | **6** |
| 꼬리에서 그리퍼 명령의 최댓값 | 0.614 | **0.846** |
| 마지막 프레임의 측정된 턱 | 0.573 | **0.834** |
| 마지막 프레임의 큐브 z | 0.0276 | 0.0279 |
| 총 프레임 | 35,918 | 36,960 |

손을 정지값 너머로 열라고 명령하는 프레임이 여섯 배가 되었고, 마지막 상태가 램프 중간의 턱이
아니라 열린 손 아래 놓인 큐브가 되었다. 전문가 오라클은 양쪽 경로에서 8 / 8 그대로다
(`expert_solves_the_pinned_seeds`, `expert_passes_the_evaluation_harness`의 최악
`envelope_violation_rate` 0.1974).

**3부 — 술어를 유일한 변수로 둔 그 레시피.** 시연 200개, bake, V13의 GroupNorm 그래프를 40,000
스텝(batch 8, lr 1e-4, seed 0, `--resident-gpu`), pack, 학습 시드 1–16과 홀드아웃 시드 101–116
평가. 손실 0.0476 → 20,000에서 0.0193 → 40,000에서 0.0146. `learning_hash` `5dac0a46…`와
`lowering_hash` `70a8fec7…`는 V13의 것이므로, 학습된 함수는 V15 데이터 위의 V13 함수다.

| nominal 스위트 | V14, 200 시연, 40k, **예산 900** | V15, 200 시연, 20k, 예산 1,800 | **V15, 200 시연, 40k, 예산 1,800** |
|---|---|---|---|
| `success_rate`, 학습 1–16 | 1 / 16 | 1 / 16 | **0 / 16** |
| `success_rate`, 홀드아웃 101–116 | 0 / 16 | 0 / 16 | **0 / 16** (두 번, 동일) |
| 테이블에서 들어올림, 학습 / 홀드아웃 | 9 / 16 · 9 / 16 | 3 / 16 · 4 / 16 | 8 / 16 · 9 / 16 |
| **통 부피 안으로 운반, 학습** | **8 / 16** | 1 / 16 | **5 / 16** |
| **통 부피 안으로 운반, 홀드아웃** | **9 / 16** | 0 / 16 | **4 / 16** |
| **통 안에서 놓음** | **0 / 16** | **0 / 16** | **0 / 16** |
| `envelope_violation_rate`, 학습 / 홀드 | 0.1587 / 0.2060 | 0.9025 / 0.9105 | 0.7148 / 0.3381 |
| `violation.position`, 학습 / 홀드 | 657 / 1,357 | 21,868 / 23,403 | 16,529 / 7,801 |
| `episode_length`, 학습 / 홀드 | 855.1 / 900 | 1700.1 / 1800 | 1800 / 1800 |

40,000 체크포인트의 홀드아웃 두 번은 셀 단위로 일치한다. **V14 열과 V15 열의 예산은 같지 않다** —
V15의 Task IR은 1,800을 싣고 있고(열린 질문 21(b)의 기본값을 유지), V14의 200 시연 표는 900에서
찍었다 — 그래서 운반 횟수는 V15에 후하고, 위반율은 애초에 동등 비교가 아니다. 두 배 오래 도는
스위트는 두 배의 클램프 프레임을 쌓는다. 비교 가능한 것은 일어났거나 일어나지 않았거나인 항목,
즉 놓기 행이다: **64 에피소드를 더 돌려도 여전히 0.**

관절별로 보면 V14와 같은 단일 관절 서명이다: 그리퍼가 소프트 경계 밖에 있는 틱이 학습 시드에서
28,800틱 중 18,159틱(63.1 %), 홀드아웃에서 8,557틱(29.7 %)이고 나머지 관절은 0 또는 10틱이다.
운반에 성공한 에피소드는 남은 내내 턱을 0.086 ~ 0.093 rad — 큐브에 물려 멈춘 턱 — 으로 유지한다.

이제 정책은 손을 열기는 한다. 다만 큐브를 쥔 채로는 절대 열지 않는다. 홀드아웃 10과 15는 파지에
실패한 뒤 턱을 0.916, 0.926 rad로 두고 끝나며, 20,000 체크포인트가 기록한 유일한 `success`가 그
가장 날카로운 판본이다: 학습 시드 9, 202틱, 큐브 위치 **(0.166, +0.370, 0.012)** — 통의 x 구간
안, 정지 상태, 열린 손 아래, 그리고 술어가 볼 수 없는 y로 통에서 470 mm 떨어진 곳. 그리퍼 항은
`success`를 *놓았다*로 만들 수는 있어도 *통 안에 놓았다*로 만들지는 못한다. 그것이 정확히 열린
질문 21(a)이고, 노드 집합 안에서 V15가 고칠 수 있는 것이 아니다.

**벽시계**(처리량 주장이 아니라 관측 — §12.4): 프레임 포함 200 시연 수집 217 s, bake 5 s
(36,960 프레임, 3.9 GB), 40,000 스텝 학습 1,332 s(RTX 4090), 1,800 예산에서 프레임 포함 16
에피소드 nominal 스위트 하나 515 ~ 533 s.

**판정.** 가설은 두 번 반증되었다. 놓기는 이미 시연 안에 있었고 — 전부, 큐브가 통에 놓인 채로
끝났다 — 그것을 잘라낸 술어가 자른 것은 14프레임짜리 램프였지 놓기 자체가 아니었다. 그것을
고쳐서 데이터에 손을 열라는 명령 프레임이 여섯 배가 되고 마지막 상태가 열린 손 아래 놓인 큐브가
되었지만, 정책의 행동은 움직이지 않았다: **64 에피소드를 더 평가해 놓기 0회, V14와 V15를 합쳐
128 에피소드**, 900 스텝에서도 1,800 스텝에서도. 홀드아웃 `success_rate`는 0으로 바뀌지 않은
수용 기준 0.5 아래이므로 중단 규칙이 발동한다 — 6-스위트 스윕 없음, 쇼케이스 영상 없음, 여기서
두 번째 변수 없음.

V15가 남기는 것은 받은 질문보다 작은 질문이다. "시연이 손이 열리기 전에 끝난다"는 답이 나왔고
틀렸다. "하네스가 놓인 큐브를 보지 못한다"도 답이 나왔고 틀렸다. 남은 것은 여섯 폭 L1 회귀의 채널
하나이고, 그 "지금 열어라" 프레임은 에피소드의 7 %이며 그 의미는 나머지 다섯이 정한다. 다음 변수는
학습 쪽이다 — 손실에서 그리퍼 채널의 가중치, 또는 이 관절의 *닫힘*을 이미 명백히 지연시키는(블렌드가
명령된 `grip_closed`에 끝내 닿지 않는다, `crates/es/tests/cli.rs`에 못박혀 있다) 시간 앙상블이고,
그것이 열림을 더 잘 다룰 이유는 없다.


### 7.24 구현 결과 (V16): 놓기는 관측의 함수가 아니다

패킷 `docs/packets/M5/V16-the-hand-opens.ko.md`. V15는 6채널 L1 회귀의 채널 하나와 그에 대한
네 개의 질문을 남겼다. 네 개 모두 여기서 답한다. 그리고 그 답은 세 후보 수정 중 어느 것이
겨냥하던 것도 아니다. **시연이 손을 펴는 그 순간, 정책의 입력 전체는 얼어 있고 타깃은 그렇지
않다.**

아래 표는 전부 V15 자신의 산출물 위에서 측정했다 — 200 시연 베이크 세트,
`model-40000.safetensors`, 낮춰진 `build/`, 40,000스텝 평가의 `.estraj`와 렌더된 타일 —
그리고 입력 포트는 Observation IR이 만드는 방식 그대로 재구성했다(`joint_state` →
`(q + 1) / 2`, `sim_cube_pose` → `(p + 0.3) / 0.6`, `rgb_overhead` → `U8 HWC / 255` 후 CHW).
스크립트는 `~/artifacts/plan-v/v16/part1{,b,c,d}.py`.

**1. 릴리스 지점의 오픈루프에서 정책은 열림을 예측한다. 그것도 이르게.** 학습 시연 20개,
개시 시점 = 턱을 닫으라고 명령한 마지막 프레임의 바로 다음(V15의 `tail` 시작), 개시 프레임
중앙값 163, 길이 중앙값 184. 20개 평균:

| rel | 타깃 | row 0 | row 2 | row 4 | row 6 | row 9 | 팔 `|P9−P0|` | 기록된 `|a+9−a|` |
|---|---|---|---|---|---|---|---|---|
| −9 | −0.046 | −0.041 | −0.039 | −0.040 | −0.050 | −0.033 | 0.2016 | 0.1892 |
| −7 | −0.046 | −0.040 | −0.039 | −0.042 | −0.048 | **0.007** | 0.1101 | 0.1033 |
| −5 | −0.046 | −0.039 | −0.039 | −0.038 | −0.027 | 0.073 | 0.0534 | 0.0360 |
| −3 | −0.046 | −0.039 | −0.034 | −0.013 | 0.031 | 0.177 | 0.0193 | 0.0008 |
| −1 | −0.045 | −0.031 | −0.002 | 0.053 | 0.133 | 0.308 | 0.0165 | 0.0036 |
| **0** | −0.036 | −0.012 | 0.040 | 0.118 | 0.216 | 0.391 | 0.0138 | 0.0000 |
| +2 | 0.006 | 0.049 | 0.139 | 0.243 | 0.347 | 0.498 | 0.0138 | 0.0000 |
| +4 | 0.080 | **0.129** | 0.246 | 0.352 | 0.439 | 0.563 | 0.0157 | 0.0000 |
| +8 | 0.306 | 0.318 | 0.431 | 0.501 | 0.566 | 0.711 | 0.0165 | 0.0000 |
| +12 | 0.498 | 0.505 | 0.580 | 0.660 | 0.753 | 0.848 | 0.0249 | 0.0000 |
| +14 | 0.568 | 0.573 | 0.677 | 0.767 | 0.827 | 0.851 | 0.0286 | 0.0000 |

row 9는 개시보다 **7프레임 앞서** 유지값을 벗어나고, row 0은 20개 시연 **전부에서**
개시+3(중앙값; 최소 −1, 최대 +3)에 `grip_closed + 0.1`을 넘는다. 예측은 기록된 램프를
약 0.05 rad 안에서 따라가고 개시부터 개시+6까지는 오히려 앞선다. 팔 행은 "계속 내려감"이
아니라 **정지**를 예측한다: 개시 시점에서 `|P9 − P0|`는 0.014 rad, 기록값은 0.000.
**오픈루프 적합은 결함이 아니다.** V13의 그리퍼 잔차 0.016 rad는 에피소드 전체 평균이었고
여기에 몰려 있지 않다.

**2. 닫힌 루프의 홀드는 매니폴드 위에 있고, 상태의 무엇도 그것을 움직이지 못한다.** V15의
40,000스텝 스위트에서 운반에 성공한 8 에피소드를, 큐브가 빈에 들어간 뒤 300 제어 틱 지점에서
표본화해, 200개 시연 전체의 평균 릴리스 개시 상태와 정책의 정규화 입력 공간에서 비교한다:

| 차원 | 홀드 | 시연 개시 | 차이 | 차이 / σ | 원단위 차이 |
|---|---|---|---|---|---|
| `shoulder_pan` | 0.8853 | 0.8864 | −0.0011 | −1077 | −0.0022 rad |
| `shoulder_lift` | 0.2132 | 0.2176 | −0.0044 | −6.5 | **−0.0088 rad** |
| `elbow_flex` | 0.8325 | 0.8309 | +0.0016 | +54 | +0.0033 rad |
| `wrist_flex` | 1.2309 | 1.2322 | −0.0013 | −750 | −0.0025 rad |
| `wrist_roll` | 0.5009 | 0.5000 | +0.0009 | +891 | +0.0018 rad |
| `gripper` | 0.5467 | 0.5469 | −0.0002 | −0.4 | −0.0003 rad |
| `cube_x` / `cube_y` / `cube_z` | 0.7211 / 0.3408 / 0.6040 | 0.7234 / 0.3436 / 0.6027 | −0.0023 / −0.0028 / +0.0013 | −1.9 / −2.5 / +0.4 | −1.4 / −1.7 / +0.8 mm |

정규화 L2 거리 홀드 → 시연 개시: **0.0120**(팔만 0.0050, 큐브만 0.0109). 최대 원단위 차이는
`shoulder_lift`의 8.8 mrad이고, 나머지 팔 관절은 3.3 mrad 이내, 큐브는 1.7 mm 이내다. 그
홀드에서 오픈루프로 모듈이 예측하는 그리퍼 행은 8 에피소드 평균으로

```
-0.039  -0.036  -0.027  -0.018  0.002  0.025  0.053  0.105  0.150  0.203
```

— 시연 자신의 개시−3과 같은 램프 모양이고 row 0은 닫혀 있다. 그다음 섭동 스윕
(`e40000-train nominal-02`, 자신의 row 0 = −0.0306): 13개 상태 차원 중 **어느 하나를** 시연
평균까지 끝까지 옮겨도 row 0은 `grip_closed + 0.1`을 넘지 않고, **13개를 함께** 옮겨도
−0.0291이며, 홀드의 상태에 **시연 0 자신의 릴리스 개시 타일**을 먹여도 −0.0287이다.

시연의 개시 프레임과 개시+4 프레임 사이에서 포트를 하나씩 바꿔 끼우면 이유가 드러난다 —
시연 20개 평균, 그다음 같은 공여 프레임을 운반 8 에피소드의 홀드에 끼운 결과:

| 바꿔 끼운 포트 | 개시 → row 0 | row 9 | 홀드 → row 0 | row 9 |
|---|---|---|---|---|
| (없음) | −0.0118 | 0.3913 | −0.0387 | 0.2030 |
| `joint_state` | −0.0119 | 0.3911 | −0.0374 | 0.2048 |
| `sim_cube_pose` | −0.0118 | 0.3912 | −0.0382 | 0.2042 |
| **`rgb_overhead`** | **0.1288** | **0.5634** | **0.1344** | **0.5665** |
| (셋 모두) | 0.1286 | 0.5630 | 0.1358 | 0.5690 |

**릴리스 판단 전체를 타일이 지고 있다.** 상태 포트는 어느 쪽이든 row 0을 1 mrad 움직이는 데
그치고, 타일은 0.17 rad를 움직이며 개시 → 개시+4의 변화 전체를 혼자 재현한다 — **홀드에
넣었을 때도** 그렇고, 거기서 명령을 −0.039에서 +0.134로, 턱이 큐브에 물려 멈추는 0.093 rad
너머로 끌어올린다. 그러므로 "홀드에서는 닫힘, 개시에서는 열림을 예측한다면 그것을 뒤집는 가장
작은 섭동은 무엇인가"의 답은 **기록된 상태의 어떤 섭동도 아니며, 시연 개시 4프레임 뒤의 아무
타일이나**이다.

**그리고 타일 신호는 타일 잡음보다 작다.** 시연 40개에 대해, 상대 프레임별로 입력이 개시
프레임의 입력에서 떨어진 차원별 평균 거리(정규화 단위 ×1000):

| rel | `shoulder_lift` | `gripper` | `cube_z` | 타일 | 타깃 그리퍼 |
|---|---|---|---|---|---|
| −15 | 252.51 | 0.68 | 94.12 | 42.22 | −0.0460 |
| −5 | 28.88 | 0.06 | 9.99 | 7.95 | −0.0459 |
| −2 | **0.32** | 0.02 | 0.14 | **0.18** | −0.0459 |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | −0.0360 |
| +2 | **0.89** | 0.02 | 0.13 | **0.38** | 0.0060 |
| +4 | 1.12 | 0.05 | 0.24 | 0.40 | **0.0799** |
| +5 | 1.12 | 0.06 | 0.25 | 0.41 | 0.1289 |
| +6 | 1.14 | **11.60** | 0.23 | 0.41 | 0.1858 |
| +9 | 1.15 | 104.85 | 29.27 | 0.58 | 0.3658 |

개시−2부터 개시+5까지 **입력 전체가 얼어 있다** — 어느 차원도 1.2 × 10⁻³ 정규화 이상
움직이지 않고 타일은 4.3 × 10⁻⁴ — 그 사이 타깃은 −0.046에서 +0.129 rad까지 오른다. 그리퍼
관절 자체가 다르게 읽히기 시작하는 것은 개시+6, 명령이 마침내 멈춤 각도를 넘는 틱이다. 직접
세면: 입력의 모든 차원이 개시 프레임의 값에서 0.002 이내인 프레임은 시연당 **6개**(40개
중앙값)이고, 그 프레임들의 그리퍼 타깃은 −0.0460에서 +0.0807, 중앙값 **+0.0057 rad**이다.

그 중앙값이 결과 전부다. 일대다 군집 위의 L1 적합은 조건부 중앙값을 돌려주고, V15가 홀드에서
실제로 측정한 닫힌 루프 명령은 **+0.0071 rad** — 군집이 예측하는 값과 1.4 mrad 차이다. 턱은
30 mm 큐브에 0.093 rad에서 물려 멈추므로, 0.093 아래의 어떤 명령도 측정 관절을, 따라서 관측
전체를, 있던 자리에 그대로 둔다. **홀드는 일반화 실패가 아니라 산술적 고정점이다.**

이 패킷의 범위 안에서 그것을 빠져나갈 수 없게 만드는 것이 둘 있다. 얼어붙은 창은 약 6 제어
틱인데, Deployment IR의 `acceleration_max = 20 rad/s²`가 50 Hz에서 틱당 스텝의 변화를
0.008 rad로 제한하므로 `grip_closed`에서 멈춤 각도까지의 0.143 rad를 정지 상태에서 지나는 데
약 6틱이 든다 — 창을 거기 놓은 것은 엔벨로프이고, 엔벨로프는 여기서 금지되어 있다. 그리고
닫힌 루프 홀드 타일에서 대응 시연의 **가장 가까운** 타일까지의 거리는 6.2 × 10⁻⁴ ~
2.9 × 10⁻³ (8개 중 8개에서 가장 가까운 프레임이 개시 ± 3), 즉 **릴리스 신호 전체의 1.5 ~
6.8배**다 — "쥐고 있음"과 "지금 열어라"를 가르는 것이 닫힌 루프와 자기 학습 데이터를 가르는
것보다 작다. 수집과 평가는 동일하게 렌더한다: 같은 시드의 틱 0에서 두 타일은 비트 단위로
같고, 이후 갈라지는 것은 전문가와 정책이 다르게 명령하기 때문이다.

**3. Unnormalizer가 없으므로 틀릴 스케일 자체가 없다.** 데모의 Learning IR 그래프는
`VisionEncoder`, `StateEncoder` 둘, `Fusion`, `TemporalEncoder`, `PolicyHead`,
`ActionChunker`다 — **`Unnormalizer` 노드는 없다.** `actions` 출력은
`Normalized { lo: −1, hi: 1 }`을 선언하지만 아무것도 그것을 적용하지 않고, `es dataset bake`는
기록된 `action` 열을 그대로 쓴다(`crates/es/src/cmd/dataset.rs:289`). 타깃, 예측, 액추에이터
명령이 모두 시연과 같은 스케일의 라디안이다. **리스케일링은 정당화되지 않고 IR 문서 변경도
필요 없다.** 분포는 운반 에피소드에서 5 제어 틱마다 한 표본:

| | n | 최소 | p05 | p50 | p95 | 최대 | 소프트 경계 아래 |
|---|---|---|---|---|---|---|---|
| 파지 + 운반 | 25 | −0.0453 | −0.0449 | −0.0308 | 0.9121 | 0.9294 | 0.0 % |
| **홀드** | 170 | −0.0407 | −0.0402 | **−0.0394** | −0.0257 | −0.0153 | **0.0 %** |
| 시연, 개시 전 40프레임 | — | −0.0480 | — | −0.0461 | — | −0.0339 | — |
| 시연, 릴리스 꼬리 | — | −0.0364 | — | 0.4177 | — | 0.8646 | — |

정책의 "닫힘" 회귀값은 −0.039로, 시연의 −0.048 … −0.034 **안쪽**이다. **소프트 경계에 대한
패킷의 전제는 교정이 필요하다**: 틱의 30 – 63 %에서 −0.1245에 붙는 현상은 운반 에피소드의
홀드가 아니다. V15 자신의 에피소드별 표가 보여준다 — 마지막 그리퍼 값은 운반에 성공한 모든
에피소드에서 0.086 – 0.093이고, −0.125는 애초에 파지에 실패한 에피소드에서만 나온다. 거기서
턱은 *비어 있고* −0.04 명령이 자유 관절을 클램프까지 밀어붙인다. 그리퍼가 유일하게 클램프되는
관절이라는 것은 여전히 사실이지만, 그 클램프는 실패한 파지의 증상이지 홀드의 증상이 아니다.

**4. 앙상블은 장애물이 아니라 유일하게 돕고 있는 것이다.** `es_eval::runner`는 **제어** 틱마다
정책을 호출하므로 청크가 매 틱 푸시되고, 유효 구간은 `valid = 10`행, `CHUNK_SLOTS = 8`이
겹침을 여덟 개로 제한하며, `w_i = exp(−0.01 i)`는 균등에서 6.8 % 이내다. 실행되는 명령은
가장 최근 여덟 청크의 0..7행을 거의 균등하게 평균한 값이다.

| 시연 0, 자신의 릴리스 | 최신 청크 row 0 | 앙상블 |
|---|---|---|
| 중점 0.426을 넘는 시점 | 개시 + 10틱 | 개시 + **11**틱 |
| 도달 최대값(타깃 최대 0.900) | 0.847 | **0.838** |

시연이 실제로 담고 있는 스텝에 대해 **제어 틱 1개의 지연과 1 %의 감쇠**다. 홀드에서는 부호가
뒤집힌다 — `e40000-train nominal-02`의 운반 이후 400틱:

| | 최소 | 중앙값 | 최대 | `grip_closed + 0.1` 초과 틱 |
|---|---|---|---|---|
| 최신 청크 row 0만 (`HardSwitch`가 실행할 값) | −0.0428 | **−0.0394** | −0.0061 | 0 / 400 |
| 시간 앙상블 (실제로 도는 것) | −0.0435 | **+0.0071** | +0.0408 | 0 / 400 |

블렌드는 그리퍼 명령을 **릴리스 쪽으로 0.047 rad 옮긴다.** 오르는 램프의 0..7행을 평균하기
때문이다. 어느 쪽도 0.093 멈춤 각도에 닿지 못한다. **후보 (b)는 측정으로 기각된다: Deployment
IR의 다른 `execution` 변형들은 모두 `HardSwitch`로 낮아지고(`es_eval::runner::blend_of`) 그것은
여기서 엄격히 더 나쁘므로, V15의 번들을 그 모드로 다시 돌리지 않았다.** 후보 (c)는 표 2로
기각된다: 홀드의 기록된 13개 상태 차원은 릴리스 매니폴드에서 1 – 9 mrad, 1.7 mm 이내이고, 그중
어느 것도 단독으로든 함께든 예측을 뒤집지 못한다 — 그리고 그것이 시사하던 데이터 쪽 수리(전문가
`Lower` 목표에 지터를 주는 것)는 얼어붙은 군집을 녹이는 대신 넓히므로 조건부 중앙값을 더
나쁘게 만든다.

표 4가 드러내는 것은 이 실패에 대해 지금까지 측정된 가장 큰 지렛대이고, 이 패킷의 변수는
아니다. `deployment.toml`은 `rate.control = 50 Hz`에 대해 `rate.inference = 5 Hz`를,
`learning.toml`은 `replanning_hz = 5.0`을 선언하지만, 평가 러너는 제어 틱마다 재계획하므로
청크의 뒷행들은 결코 실행되지 않는다. 선언된 5 Hz라면 — 열 제어 틱마다 청크 하나가 자기
0..9행을 순서대로 구동한다면 — 홀드에서 명령된 그리퍼는 −0.039 → 0.203을 걸어가고 **운반 8
에피소드 중 8개에서** row 7 또는 8에서 0.093 멈춤 각도를 넘는다. 그것이 열린 질문 23이다.

**Part 2 — 가중치, 그리고 그것이 확인해 준 예측.** 후보 (b)는 표 4로, (c)는 표 2로 기각되므로
움직인 것은 (a)이고, 그것이 패킷이 요구한 손잡이이기도 하다. 진단은 그것으로 충분하지 않을
것이라 예측한다 — 채널 가중치는 그 채널의 오차를 곱할 뿐 얼어붙은 군집의 조건부 중앙값을
움직이지 못한다 — 그리고 이 실행이 22분의 값어치를 하는 이유는 바로 그 예측을 반증 가능하게
만들기 때문이다: 모듈은 이미 4.3 × 10⁻⁴의 타일 신호로 개시와 개시+4를 0.14 rad만큼 가르고
있고, 다섯 배의 기울기는 그 분리를 날카롭게 할 다섯 배의 유인이다.

`python/es/train_act.py --channel-weight 5=5`: `w = [1,1,1,1,1,5]`인 `(|d| * w).mean()`,
플래그 하나, 그리고 전부 `w = 1`이면 정확히 `l1_loss`다 — 오라클
`channel_weight_of_one_is_the_unweighted_loss`가 바이트 단위로 고정하므로 이 절의 모든 손실
곡선은 비교 가능한 채로 남는다. 그 밖에는 아무것도 움직이지 않았다: V15의 베이크 세트, V15의
낮춰진 모듈(`lowering_hash 70a8fec7…`), V15의 미학습 번들, 40,000스텝, batch 8, lr 1e-4,
seed 0, `--resident-gpu`. 이 플래그는 해시 슬롯에 들어가지 않으므로(스펙 8.1)
`task_hash eb6efefa…`, `observation_hash 899c16a9…`, `learning_hash 5dac0a46…`,
`deployment_hash`는 모두 V15의 것이고 가중치 바이트만 움직인다(`policy_hash` → `47516aa7…`).
학습 손실 0.0920 → 0.0241은 *가중된* 목적함수이므로 V15의 0.0476 → 0.0146과 비교할 수 없다.
비교 가능한 숫자는 가중 없는 오픈루프 L1이다:

| 오픈루프, 시연 6개, 3프레임마다 | V15 | **V16 (그리퍼 ×5)** |
|---|---|---|
| 청크 L1, 6채널 전부, 가중 없음 | 0.0116 | 0.0134 |
| 청크 L1, 그리퍼 채널만 | 0.0095 | **0.0109** |

그리퍼 채널 자신의 가중 없는 오차가 *더 나빠졌다*. 목적함수를 가중 없는 최적점에서 멀리
옮기면 일어나는 일이다. 가중이 바꾼 것은 램프의 모양이고, 바꾸지 못한 것은 액추에이터에
닿는 행이다. V15의 운반 8 에피소드 홀드에서, 8개 평균:

| 홀드의 그리퍼 청크 | row 0 | row 2 | row 4 | row 6 | row 7 | row 9 | 0..7행 평균(앙상블이 실행하는 값) |
|---|---|---|---|---|---|---|---|
| V15 | −0.039 | −0.027 | 0.002 | 0.053 | 0.105 | 0.203 | **+0.0071** |
| **V16** | **−0.046** | −0.026 | 0.015 | **0.091** | 0.146 | **0.256** | **+0.0214** |

row 9는 26 %, row 6은 0.038 rad 올랐다 — 램프는 더 가팔라졌다. **row 0은 7 mrad 내려갔고**,
앙상블이 실제로 내는 명령은 여전히 0.093 멈춤 각도의 4분의 1이다. 두 체크포인트 모두 8개 홀드
중 0개에서 row 0이 멈춤 각도를 넘는다. 그것을 깨뜨리려고 설계한 실험이 일대다 논증을 확인해
주었다.

닫힌 루프, 같은 두 스위트, 같은 1,800스텝 예산:

| nominal 스위트 | V15, 200 시연, 40k | **V16, 그리퍼 ×5, 40k** |
|---|---|---|
| `success_rate`, 학습 1–16 | 0 / 16 | **0 / 16** |
| `success_rate`, 홀드아웃 101–116 | 0 / 16 | **0 / 16** (두 번, 동일) |
| 큐브 들어올림, 학습 / 홀드아웃 | 8 / 16 · 9 / 16 | **5 / 16 · 4 / 16** |
| 빈 안까지 운반, 학습 / 홀드아웃 | 5 / 16 · 4 / 16 | **0 / 16 · 0 / 16** |
| **빈 안에서 놓음** | 0 / 16 | **0 / 16** |
| `envelope_violation_rate`, 학습 / 홀드아웃 | 0.7148 / 0.3381 | **0.0510 / 0.0485** |
| `violation.position`, 학습 / 홀드아웃 | 16,529 / 7,801 | **0 / 0** |
| 그리퍼가 소프트 경계 밖인 틱 | 18,159 (63.1 %) / 8,557 (29.7 %) | **0 (0.0 %) / 0 (0.0 %)** |
| `episode_length` | 1800 / 1800 | 1800 / 1800 |

판정과 별개로 남겨둘 행이 둘이다. `violation.position`이 **두 스위트 모두 0**이고, 그리퍼는
57,600 제어 틱 중 단 하나도 소프트 경계 밖에 있지 않다. 다섯 배의 기울기가 닫힘 회귀값을
−0.1245 아래에서 −0.068 … −0.081로, 엔벨로프 안으로 끌어올렸다. V14와 V15가 함께 보고하던
단일 관절 클램프 징후는 사라졌고 릴리스는 그것을 따라 나오지 않았다 — 표 3의 결론이 이제
정책 쪽뿐 아니라 Safety Plane 쪽에서도 확인된 것이다. 그리고 손은 펴진다: 학습 5개, 홀드아웃
2개 에피소드에서 마지막 그리퍼 값이 0.89 – 1.39 rad이고, 그 전부가 애초에 파지에 실패한
에피소드다. V15의 발견 그대로다.

대가는 운반이었다. 8개와 9개의 운반이 **0개와 0개**가 되었고, 들어올림은 8, 9에서 5, 4로
떨어졌다. 그리퍼 채널의 다섯 배는 다섯 팔 채널 각각의 상대 가중이 5분의 1이 된다는 뜻이고,
큐브를 찾아 옮기는 것은 팔이다.

**벽시계**(처리량 주장이 아니라 관측치 — §12.4): 40,000스텝 학습 1,331 s(RTX 4090); 프레임
포함 16 에피소드 nominal 스위트 3개, 1,800 예산 1,583 s; Part 1의 분석 넷은 CPU에서 총 약 4분.

**판정.** 단일 변수를 움직였고, 예측은 유지되었으며, 수정은 자신의 측정으로 기각된다:
그리퍼 램프는 row 9에서 26 % 가팔라지고 **액추에이터에 닿는 유일한 행인 row 0에서 7 mrad
얕아졌으며**, 닫힌 루프는 가지고 있던 운반을 전부 잃었다. 홀드아웃 `success_rate`는 0으로
불변인 수용 임계치 0.5 아래이므로 중지 규칙이 발동한다 — 6스위트 스윕 없음, 쇼케이스 영상
없음, 두 번째 변수 없음. **V15의 체크포인트가 여전히 플랜 V 최고의 정책이다.** V16이 남기는
것은 다음에 시도할 목록이 아니라 세 후보 중 무엇도 통할 수 없었던 이유의 측정된 설명이다 —
릴리스는 Deployment IR 자신의 가속 한계에 의해 여섯 제어 틱 동안 관측에서 얼어붙어 있고,
그것을 이미 예측하고 있는 청크는 첫 행 너머로 실행되지 않는다 — 그리고 사람이 이제 기본값을
골라야 하는 열린 질문 22와 23이다.

### 7.25 구현 결과 (V17): 놓기를 예측하던 청크가 실행되고, 손이 열린다

패킷 `docs/packets/M5/V17-inference-rate.ko.md`. V16의 진단은 스스로 당길 수 없는 레버에서
끝났다. `deployment.toml`은 `rate.control = 50 Hz`와 `rate.inference = 5 Hz`를 선언하는데 두
루프 모두 두 번째 숫자를 읽지 않았다. 이 패킷은 두 경로 모두에서 그것을 지키고 V15의
40,000스텝 체크포인트를 다시 측정한다 — 재학습 없이, 같은 `model-40000.safetensors` 파일,
같은 `lowering_hash 70a8fec7…`, `task_hash eb6efefa…`, `observation_hash 899c16a9…`,
`learning_hash 5dac0a46…`.

**케이던스, 전과 후.** 규칙은 하나다. `es_env::replan_interval(rate) = rate.control /
rate.inference`를 정수 제어 틱으로 주고, 나누어떨어지지 않으면 이름을 붙여 거부한다. 1,800틱
에피소드 기준:

| | 에피소드당 정책 호출 | 액추에이터에 닿는 청크 행 | 살아 있는 청크 |
|---|---|---|---|
| V6b – V16 | **1,800** | 각 청크의 0행, 8겹 앙상블을 거쳐 | 8 |
| **V17** | **180** | **각 청크의 0..9행, 순서대로** | **1** |

버퍼는 특수 처리하지 않았다. 낮춰진 청크가 `[10, 6]`이라 `valid = 10`이고,
`TemporalEnsemble`에서 스팬은 `valid`행이며 다음 도착은 정확히 10틱 뒤이므로 겹침 집합의
원소는 하나이고 지수 평균은 그것으로 축퇴한다. 분기가 아니라 산술이다 — `ChunkBuffer`와
`plane_chunk`는 손대지 않았다.

**이 변경이 드러낸 결함 둘. 둘 다 매 틱 정책을 부르는 동안에는 보이지 않았다.**

1. **플레인의 워치독 시계가 시뮬레이션 틱이었다.** `SafetyPlane`은 틱 차이를 Deployment IR의
   *제어* 주기로 마이크로초로 바꾸는데(`es_safety::config`,
   `period_us = rate.control_period()`), 두 루프 모두 `Env::tick` — V11 이후 이 씬에서 제어
   스텝당 4틱인 **시뮬레이션** 틱(`timestep = 0.005`, 제어 50 Hz) — 을 넘겼다. 매 제어 틱마다
   청크가 도착하는 동안에는 드러날 수 없었다. `accept`가 마감을 재기 전에
   `last_chunk_tick = now`를 찍어 간격이 항상 0이었기 때문이다. V17의 첫 실행이 즉시 드러냈다.
   40 ms 예산에서 `violation.inference_deadline`이 **28,800 제어 틱 중 25,920틱** — 90 %이고,
   이는 `k ≥ 1`에 대해 `80,000 µs × k > 40,000`, 즉 청크가 도착하는 틱을 뺀 전부다.
   `DomainRunner::emit_actions`는 자신의 `control_tick`을, `es_eval::runner`는 에피소드 스텝을
   쓴다. `es-safety` 자체는 그대로다.
2. **5 Hz 재계획은 40 ms 추론 마감을 만족시킬 수 없다.** 시계를 고쳐도 도착 간격은 곧 재계획
   주기 — 최대 180 ms — 다. `deployment.toml`의 `inference_budget`과 `inference_deadline`
   워치독은 **240 ms**가 된다(문서가 스스로 요구하는 200 ms에 추론용으로 이미 허용하던 40 ms).
   `deadlines.observation_age`는 `DEP_021`이 `inference_budget`보다 작은 값을 거부하기 때문에만
   따라간다. 강제되는 신선도 한계(`stale_observation`, 80 ms), 모든 안전 한계, 0.5 수용
   임계값은 움직이지 않았고 워치독은 여전히 켜져 있다. 이것이 미해결 질문 23의 "두 편 중 하나는
   바뀌어야 한다"이며, 결과적으로 양쪽 모두였다. `deployment_hash`가 함께 움직이므로 V15의
   가중치를 현재 문서로 만든 번들에 다시 담았다 — 같은 safetensors 파일, 바이트 그대로.

**오라클**(오라클 서버, RTX 4090, 트리 `~/Projects/es-v17`, `~/artifacts/plan-v/v17/`,
2026-09-16):

| 오라클 | 결과 |
|---|---|
| `the_runner_infers_once_per_declared_replan_period` | 100 제어 스텝 → 10:1에서 **10회**, 1:1에서 **100회** |
| `an_inference_rate_that_does_not_divide_the_control_rate_is_refused` | 두 레이트와 `3.33` 제어 틱을 이름으로 말하는 `EvalError::Env` |
| `collection_and_evaluation_ask_the_policy_at_the_same_cadence` | 120 제어 틱에서 **각 경로 12회** |
| `collection_and_evaluation_draw_the_same_scene_for_a_seed` | 25개 값 동일 |
| `expert_passes_the_evaluation_harness` | **8/8**, `envelope_violation_rate` 0.2893 – **0.3021**, 에피소드 길이 533 – 553 |
| `expert_solves_the_pinned_seeds` | 8/8 |
| V9 `a_showcase_replay_reproduces_the_frames_the_policy_saw` | 통과 |
| V10 `recorded_actions_replay_to_the_same_outcome` | 통과 |
| `the_temporal_ensemble_survives_the_grasp_window` | 통과 |
| `cargo xtask ci` | 통과 |

전문가의 `envelope_violation_rate`는 V6b의 0.4815 – 0.5485에서 0.2893 – 0.3021로 떨어졌고
에피소드는 길어졌다(330 – 354 → 533 – 553). 한 청크의 행을 순서대로 실행하는 것이 서로
어긋나는 여덟 청크의 평균보다 매끄러운 명령 흐름이라 플레인이 덜 자르고, 전문가도 이제 청크당
1행이 아니라 10행이 실행된다는 전제로 페이스를 잡는다(`ExpertCfg::pace_to`가 두 경로에서 같은
수를 받는다).

**두 경로가 여전히 공유하지 않는 것, 측정했고 단언으로 덮지 않았다.** 한 시드에서
`es loop collect`와 `es eval run`의 틱별 `qpos ‖ qvel` 궤적은 **틱 1**에서 갈라진다 — 틱 0도
아니고 재계획 경계도 아니다. `es loop collect`는 Learning IR의
`RuntimeHints::expected_latency_ms = 15.0`(50 Hz에서 제어 1틱)을 `AsyncInference`로 적용해
첫 청크가 틱 1에 플레인에 닿고 틱 0은 기록된 언더런이 되는데, `es_eval::runner`에는 지연
모델이 없어 틱 0에 0행을 실행한다. 관절 0에서 수집은 `1.44 × 10⁻⁷` rad, 평가는
`−4.24 × 10⁻⁴`다. 이것이 미해결 질문 24이며, V17은 추론 지연 모델링을 있던 그대로 둔다.

**재측정.** V15의 40,000스텝 체크포인트, 1,800스텝 예산, 16에피소드 nominal 스위트 3회(빌드,
재패킹, 3회 전부 합쳐 벽시계 649초):

| nominal 스위트 | V15, 40k | V16, 그리퍼 ×5, 40k | **V17, 선언된 5 Hz의 40k** |
|---|---|---|---|
| `success_rate`, 학습 1–16 | 0 / 16 | 0 / 16 | **2 / 16 = 0.1250** |
| `success_rate`, 홀드아웃 101–116 | 0 / 16 | 0 / 16 | **1 / 16 = 0.0625**(두 번, 비트 동일) |
| 큐브 들어올림, 학습 / 홀드아웃 | 8 / 16 · 9 / 16 | 5 / 16 · 4 / 16 | **6 / 16 · 6 / 16** |
| 빈까지 운반, 학습 / 홀드아웃 | 5 / 16 · 4 / 16 | 0 / 16 · 0 / 16 | **5 / 16 · 5 / 16** |
| **빈 안에서 놓기**, 학습 / 홀드아웃 | 0 / 16 · 0 / 16 | 0 / 16 · 0 / 16 | **2 / 16 · 2 / 16** |
| `envelope_violation_rate`, 학습 / 홀드아웃 | 0.7148 / 0.3381 | 0.0510 / 0.0485 | **0.9985 / 0.9982** |
| `episode_length`, 학습 / 홀드아웃 | 1800 / 1800 | 1800 / 1800 | **1627.1 / 1722.4** |

**손이 열린다.** 평가된 48에피소드에서 놓기 4회, 그중 3회는 Task IR 자신의 콘이 `success`로
채점했다. V13·V14·V15·V16의 **128에피소드에서 놓기는 0회**였다. 에피소드별, 학습 시드 1–16
(`~/artifacts/plan-v/v17/e40000-train`):

| 셀 | 틱 | 들어올림 mm | 들어올림 @ | 운반 @ | 놓기 @ | 하니스 | 빈 안 |
|---|---|---|---|---|---|---|---|
| nominal-03 | 1800 | 132.7 | 132 | 220 | — | timeout | 예 |
| nominal-05 | 1800 | 132.0 | 133 | 190 | — | timeout | 예 |
| **nominal-08** | **602** | 124.0 | 125 | 180 | **410** | **success** | 예 |
| nominal-11 | 1800 | 125.4 | 122 | 185 | — | timeout | 예 |
| **nominal-15** | **231** | 117.5 | 115 | 188 | **225** | **success** | 예 |
| 나머지 11개 | 1800 | 3.0 – 26.6 | — | — | — | timeout | 아니오 |

그리고 홀드아웃 101–116(`e40000-holdout-a`, `-b`는 행 단위로 동일):

| 셀 | 틱 | 들어올림 mm | 들어올림 @ | 운반 @ | 놓기 @ | 하니스 | 빈 안 |
|---|---|---|---|---|---|---|---|
| nominal-01 | 1800 | 153.0 | 110 | 243 | — | timeout | 예 |
| nominal-04 | 1800 | 123.3 | 121 | 210 | — | timeout | 예 |
| nominal-05 | 1800 | 124.9 | 121 | 209 | **1040** | timeout | 예 |
| nominal-11 | 1800 | 141.7 | 113 | 176 | — | timeout | 예 |
| **nominal-12** | **559** | 131.9 | 102 | 248 | **280** | **success** | 예 |
| nominal-02 | 1800 | 126.3 | 128 | — | — | timeout | 아니오 |
| 나머지 10개 | 1800 | 2.0 – 14.8 | — | — | — | timeout | 아니오 |

**어느 술어 항이 실패하는가.** 그리퍼, 오직 그리퍼다. 홀드아웃의 운반 5에피소드에서 큐브는
1,800틱 중 1,557 – 1,618틱 동안 빈의 x 구간 **안에 있고** 안정되어 있다(`|vx| < 0.05`). 즉
위치 항도 안정 항도 채점된 성공을 막은 적이 없다. 바로 그 틱들에서 그리퍼 `qpos`의 최댓값은:

| 홀드아웃 셀 | 큐브가 빈 안에서 안정된 동안의 그리퍼 최댓값 | 하니스 |
|---|---|---|
| nominal-11 | 0.227 | timeout |
| nominal-01 | 0.335 | timeout |
| nominal-04 | 0.409 | timeout |
| **nominal-05** | **0.817** | timeout |
| **nominal-12** | **0.820** | **success** |
| *(비교용, 학습 nominal-08)* | *0.844* | *success* |

다섯 중 셋은 0.41 rad을 넘지 못한다 — 예전의 홀드 그대로다. 넷째 nominal-05는 **0.817**까지
열려 술어의 0.85에 **약 33 밀리라디안** 못 미치고(위 표의 0.820은 마지막 틱의 *스텝 후* 상태가
0.85를 넘어 채점된 에피소드의 마지막 *스텝 전* 궤적 표본이다 — 술어는 스텝 후 상태로 판정되고
궤적은 스텝 전 상태를 기록한다, 패킷 V12; 오케스트레이터가 `terms.py`로 확인: 채점된 모든
에피소드에서 `all3 = 0`), 에피소드 끝에는 다시 −0.009 rad로
닫힌다. 즉 놓기는 이제 정책이 하는 일이고, 놓기와 *채점된* 놓기를 가르는 것은 집게 여행의
마지막 몇 센티라디안과 그 상태를 유지하는지 여부다.

**대가: 엔벨로프.** `envelope_violation_rate`가 **0.9985 / 0.9982**가 되었다 —
`violation.acceleration`이 학습 26,033틱 중 23,443틱, 홀드아웃 27,559틱 중 24,815틱,
`violation.velocity`가 대략 절반이다. 청크의 0..9행을 순서대로 실행하면 제어 틱당 0.02 – 0.03
rad의 이동을 요구하는데 `acceleration_max = 20 rad/s²`는 틱당 *스텝* 변화를 0.008 rad로
묶으므로 플레인이 거의 매 틱 레이트 제한을 건다. `EnvelopeViolationRate` 워치독
(`max_frac = 0.9`, 창 200)은 학습 2,547틱, 홀드아웃 2,690틱 — 실행의 약 1/10 — 동안 폴백을
걸어 잠근다. 8겹 앙상블에서는 같은 청크들이 훨씬 평탄한 명령으로 평균되어 0.71 / 0.34였다.
새로운 무언가를 명령하는 것이 아니라 시연 자신의 명령 흐름이 평균되지 않은 채 플레인에 닿는
것이고, 이는 `deployment.toml`이 V1부터 예고한 바다("이상이 아니라 구성상 대부분의 틱에서
잘린다"). 그러나 0.9985에서는 워치독이 걸리며, 그것이 미해결 질문 25다.

**판정.** 선언된 재계획 주기를 지킨 것은 플랜 V에서 놓기를 만들어낸 첫 변경이고, 홀드아웃
시드에서 성공을 만들어낸 첫 변경이다. 재학습 없이 V16이 잃었던 운반도 되찾았다(두 스위트
모두 0 → 16개 중 5개). 홀드아웃 `success_rate`는 **0.0625**로 불변인 수용 임계값 0.5 아래이므로
정지 규칙이 발동한다. 6스위트 스윕도, 쇼케이스 영상도, 두 번째 변수도 없다. **V15의
체크포인트가 여전히 플랜 V의 최고 정책이며, 보고되어 온 것보다 측정 가능하게 더 낫다** —
7.19 – 7.24절의 모든 숫자는 청크의 뒷행이 죽은 채로 측정된 것이다. V17이 남기는 표적은 V16이
남긴 것보다 훨씬 좁다. 큐브는 홀드아웃 16시드 중 5개에서 빈에 도달하고 에피소드의 90 % 동안
그 안에 안정되어 있으며, 그중 하나에서는 집게가 술어에 약 33 밀리라디안까지 다가간다. 미해결 질문
22의 (ii) — 관측이 담고 있는 무언가에 놓기를 맞춘다 — 가 다음 변수이고, 이제는 탈출할 고정점이
아니라 고칠 구체적인 대상이 있다.

### 7.26 구현 결과 (V18): 엔벨로프 절제 실험 — 결정이 아니라 절제 실험

패킷 `docs/packets/M5/V18-envelope-ablation.md`. 미해결 질문 25는
`envelope_violation_rate 0.998` 문제를 움직일 수 있는 세 갈래를 꼽았다. 그중 **(a)** —
STS3215 서보가 실제로 낼 수 있는
값까지 `acceleration_max`를 넓힌다 — 는 오너가 내리는 Deployment IR 결정이고, 오너는 결정 전에
데이터를 요청했다. 이 패킷은 변수를 정확히 하나만 움직인다: `body.safety.acceleration_max`(관절
6개 전부)를, 커밋된 파일을 건드리지 않는 `tests/fixtures/visible-learning/deployment.toml`의
*절제 사본* 안에서 네 단계로 — **20**(현재 값, 기준선으로 비트 단위까지 재현), **40**, **80**,
그리고 **80에 `velocity_max`도 3.0 → 4.4 rad/s로 함께 넓힌 단계**(픽스처 자신의 주석이 이미
인용하는 STS3215의 무부하 정격) — 넷째 단계는 가속도가 걸림돌에서 빠진 뒤 속도가 새로운
병목이 되는지 보기 위한 것이다. 나머지는 모두 V17의 재측정 그대로다: V15의
`model-40000.safetensors` 그대로, 현재의 Task/Observation/Learning 문서, 1,800스텝 예산, 학습
시드 1–16과 홀드아웃 시드 101–116, 단계당 한 번(V17의 비트 동일 반복은 여기서는 필요 없다).
재학습 없음, 커밋된 픽스처 변경 없음 — 절제 사본은 `~/artifacts/plan-v/v18/deployments/`
아래에 있고, 번들을 굽고 팩하는 동안만 픽스처의 스크래치 사본에 patch되었다가 다음 단계를
돌리기 전에 커밋된 파일로 복원된다.

**기준선은 V17을 정확히 재현한다.** `es policy pack`이 직접 찍는 해시들이 V17과 비트 단위로
같다: `weights_hash 4b583636…`, `lowering_hash 70a8fec7…`, `task_hash eb6efefa…`,
`observation_hash 899c16a9…`, `learning_hash 5dac0a46…`, `policy_hash 4dc23c92…`. 두 스위트의
`evaluation_hash`도 마지막 비트까지 같고(학습 `b49fc549…`, 홀드아웃 `40623e01…`)
`report.json`이 담은 숫자도 전부 같다: `success_rate` 0.1250 / 0.0625,
`envelope_violation_rate` 0.99854 / 0.99819, `episode_length` 1627.0625 / 1722.4375, 그리고
`failure_mode_histogram` 전체(`fallback` 2547 / 2690, `violation.acceleration` 23443 / 24815,
`violation.velocity` 13863 / 12661) — 모두 7.25절의 표와 동일하다. 유일하게 **일치하지 않는**
해시는 `execution_hash`(여기서는 `c8400882…`, V17은 `20416dfd…`)이고, 이유는 감춰두지 않고
밝힌다: `execution_hash`의 `runtime` 슬롯은 `PolicyRuntime::runtime_hash`인데, torch
백엔드에서는 `blake3(backend, protocol, torch.__version__)`
(`crates/es-policy/src/torch_runtime.rs:392-395`)이다. V17은 `~/venvs/es`(torch
`2.14.0+cpu`)에서 돌았고, 이 패킷의 서버 지침은 `~/venvs/es-lerobot-cuda`(torch
`2.11.0+cu129`)를 지정한다 — 인터프리터가 다르니 `runtime_hash`도 다른 것이 설계대로다(스펙
5.3: 체인은 *어느* 런타임이 결과를 냈는지도 이름 붙인다). 문서·가중치·정책의 정체성과 측정된
모든 숫자는 영향받지 않는다.

**네 단계.**

| 단계 | 스위트 | `success_rate` | 들어올림 | 운반 | 놓음 | 하니스 성공 | `envelope_violation_rate` | `violation.acceleration` | `violation.velocity` | `violation.rate_limit` | 폴백(래치) | 평균 `episode_length` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20(기준선) | 학습 | 0.1250 | 6/16 | 5/16 | 2/16 | 2/16 | 0.99854 | 23443/26033 | 13863/26033 | 0 | 2547 | 1627.06 |
| 20(기준선) | 홀드아웃 | 0.0625 | 6/16 | 5/16 | 2/16 | 1/16 | 0.99819 | 24815/27559 | 12661/27559 | 0 | 2690 | 1722.44 |
| 40 | 학습 | **0.4375** | 16/16 | 16/16 | 7/16 | 7/16 | 0.92439 | 15467/18173 | 7750/18173 | 0 | 1160 | 1135.81 |
| 40 | 홀드아웃 | **0.6250** | 15/16 | 15/16 | 11/16 | 10/16 | 0.90054 | 12784/15202 | 6105/15202 | 0 | 767 | 950.13 |
| 80 | 학습 | **0.6250** | 15/16 | 15/16 | 10/16 | 10/16 | 0.63672 | 8244/15784 | 5726/15784 | 0 | 93 | 986.50 |
| 80 | 홀드아웃 | **0.6250** | 14/16 | 14/16 | 11/16 | 10/16 | 0.55518 | 7027/14979 | 4865/14979 | 0 | 53 | 936.19 |
| 80 + 속도 4.4 | 학습 | 0.1875 | 15/16 | 15/16 | 5/16 | 3/16 | 0.63213 | 14417/24087 | 8184/24087 | **1010** | 291 | 1505.44 |
| 80 + 속도 4.4 | 홀드아웃 | 0.4375 | 15/16 | 15/16 | 10/16 | 7/16 | 0.60827 | 10953/18781 | 5150/18781 | **644** | 139 | 1173.81 |

(`들어올림`/`운반`/`놓음`/하니스 성공과 에피소드별 `release@` 틱은 `.estraj`에서 재구성한
`probe.py`의 결과이며 `target/plan-v/v18/probe-all.txt`에 전문이 있다. `하니스 성공`은
`report.json` 자신의 `success_rate`이고, 위 표의 모든 칸에서 둘이 일치한다. `probe.py`의
판정은 트래젝토리 행이 기록하는 **스텝 이전** 상태에서 재구성한 것인데(패킷 V12) Task IR의
술어는 **스텝 이후** 상태로 판정하므로, `probe.py`의 몇몇 행은 `report.json`이 그보다 한 틱
빠르거나 늦게 `success`로 채점한 에피소드를 `timeout`으로 찍기도 한다. 위의 모든 스칼라는
`report.json` 쪽 — 하니스 자신의 판정 — 에서 가져온 것이다.)

**40에서 이미 정지 규칙을 넘는다.** 기준선 다음 첫 단계인 `acceleration_max = 40`에서 벌써
홀드아웃 `success_rate`가 **0.625**로, 불변인 0.5 수용 임계값을 크게 넘는다. 80까지 더
넓혀도 홀드아웃 `success_rate`는 다시 오르지 않지만(0.625, 같은 16시드가 대체로 겹친다)
엔벨로프 비용은 계속 줄어든다: `envelope_violation_rate`가 0.998(20) → 0.900–0.924(40) →
0.555–0.637(80)로 떨어지고, 워치독이 래치하는 비율도 실행의 약 1/10(20)에서 5–6 %(40),
1 % 미만(80)으로 떨어진다. `episode_length`도 같이 줄어든다 — 에피소드가 1,800스텝 예산을
다 쓰는 대신 채점된 성공으로 끝나기 때문이다. 이는 V17이 8겹 앙상블에서 청크 자신의 행을
그대로 실행하는 것으로 옮겼을 때 측정한 것과 같은 효과다: 명령 흐름이 그것이 생성된
엔벨로프 안에 있으면 플레인이 계속 보정할 필요가 없다.

**`acceleration_max`와 함께 `velocity_max`도 넓히면 더 나빠진다 — 그리고 히스토그램이 정확히
왜인지 말해준다.** 넷째 단계의 홀드아웃 `success_rate`(0.4375)는 순수 80단계(0.625)보다
나쁘고, 학습 `success_rate`(0.1875)는 40단계(0.4375)보다 나쁘다. `envelope_violation_rate`도
나아지지 않는다(0.608–0.632, 80단계의 0.555–0.637 대비). 실패 모드 히스토그램에는 **다른
어느 단계에도 없는** 위반 종류가 나타난다: `violation.rate_limit` — `ViolationKind::RateLimit`,
즉 `action_rate`의 1·2차 미분 클램프(스펙 9.3, `crates/es-safety/src/plane.rs:419-431`)로,
`EnvelopeViolationRate` 워치독 자신의 표시인 `ViolationKind::ViolationRate`
(`violation.rate`, 위 표의 모든 행에서 폴백 수와 같다)와는 별개다. 픽스처 자신의 주석은
`action_rate.first_diff_max = 0.08`(4.0 rad/s)을 두고 "이 제어 주기에서는 위 두 행보다
느슨해서 50 Hz에서는 둘 다 걸리지 않는다"고 적었는데, 이는 정확히 `velocity_max * dt <= 0.08`,
즉 `velocity_max <= 4.0`인 동안만 참이다. `velocity_max = 3.0`에서 그 곱은 0.06 rad이고,
`acceleration_max`를 40이나 80으로 올리는 것만으로는 이 값이 전혀 바뀌지 않는다 — 그래서
40단계와 80단계 모두 `violation.rate_limit`이 0이다. `velocity_max = 4.4`에서는 그 곱이
0.088 rad로 0.08 rad 경계를 넘어서고, 이 제어 주기에서는 걸리지 않으리라 픽스처가 가정한
레이트 제한 클램프가 가속도·속도 클램프 위에 추가로 걸리기 시작한다 — 학습 24,087틱 중
1,010틱, 홀드아웃 18,781틱 중 644틱 — 그리고 그렇게 만들어진 명령 흐름은 속도를 넓히지 않은
80단계보다도, 덜 넓힌 40단계보다도 나쁜 점수를 낸다. 가속도가 걸림돌에서 빠지면 속도가
병목이 되는 것은 맞다(`violation.velocity`는 80단계에서도 여전히 틱의 27–36 %) — 하지만
이제까지 느슨했던 `action_rate` 경계 안으로 속도를 넓혀 넣는 것은 공짜가 아니고, 이 절제
실험이 그것을 처음으로 보여준 측정이다.

**기준선의 놓기 구간과 워치독 래치, 측정값(산출물 2).** V17의 표가 이름 붙인 네 에피소드 —
학습 `nominal-08`(`release@410`), 학습 `nominal-15`(`release@225`), 홀드아웃
`nominal-05`(`release@1040`), 홀드아웃 `nominal-12`(`release@280`) — 중 `events.json`의
워치독 래치(`Fallback`) 구간이 놓기 이후 구간과 겹치는 것은 넷 중 셋이다:

| 에피소드 | 놓기 틱 | 에피소드 끝 | 놓기 이후 폴백 틱 | 겹침? |
|---|---|---|---|---|
| 학습 `nominal-08` | 410 | 601 | 20 (`479–492, 507, 522, 534, 567, 572, 587`) | **예** |
| 학습 `nominal-15` | 225 | 230 | 0 | 아니오 |
| 홀드아웃 `nominal-05` | 1040 | 1799 | 77 (짧은 구간 10개, `1057–1068` … `1700, 1713`) | **예** |
| 홀드아웃 `nominal-12` | 280 | 558 | 34 (짧은 구간 9개, `283` … `549–550`) | **예** |

`nominal-15`는 놓기 5틱 뒤에 에피소드가 끝나 그 구간에서 워치독이 한 번도 래치하지 않는다.
나머지 셋은 큐브가 놓인 것으로 채점된 뒤에도 수십 틱 동안 `Clamped`와 `Fallback`을 오가며
계속 도는데, 이는 0.998 엔벨로프 위반 이야기가 술어가 참이 된 *뒤*에도 계속되는 것과 같다 —
하니스의 성공 판정은 그 뒤 플레인이 무엇을 하는지에 좌우되지 않지만, 실제 로봇이라면 "끝난"
뒤에도 집게가 간헐적으로 얼었다 풀렸다 할 것이다.

**판정, 그리고 오너가 치를 대가.** 이것은 결정이 아니라 절제 실험이다:
`tests/fixtures/visible-learning/deployment.toml`은 손대지 않았고, 어떤 단계도 6스위트
섭동 스윕이나 쇼케이스 영상으로 승격하지 않는다 — 그것은 패킷이 알려주려는 픽스처 결정이지
패킷이 내리는 결정이 아니다. 측정 자체는 모호하지 않다: `acceleration_max`만 20에서 40
rad/s²로 올려도 V13부터 모든 체크포인트를 막아온 홀드아웃 수용 임계값을 이미 넘고, 80까지
더 올리면 `success_rate`를 깎지 않으면서 엔벨로프 여유를 계속 벌어준다
(`envelope_violation_rate` 0.998 → 0.90–0.92 → 0.56–0.64). 같은 움직임에 `velocity_max`까지
같이 올리는 것은 도움이 되지 않고 측정 가능하게 해롭다 — `acceleration_max`만 바꿔서는 전혀
건드리지 않는, 이제까지 느슨했던 두 번째 경계(`action_rate.first_diff_max`)를 넘기 때문이다.
**오너가 (a)를 택한다면 데이터가 말하는 것은: `acceleration_max`만 움직여라, 그리고 40만으로도
중요한 일(수용 임계값을 넘는 것)은 이미 끝난다. 80은 더 낮은 엔벨로프 위반 수치를 주장할
가치가 있을 때 쓸 수 있다.** 대가는 수치만이 아니라 물리적이다: 픽스처 자신의 주석은 20
rad/s²를 로터 토크 여유(`armature * acceleration = 0.028 * 20 = 0.56` N·m, 스톨의 5분의 1)와
"명령과 측정된 궤적이 가깝게 붙어 있다"는 데서 이끌어냈는데, 80 rad/s²에서 그 여유는
`0.028 * 80 = 2.24` N·m로 STS3215 스톨 토크 2.94 N·m의 대부분을 차지한다. 플레인의 명령이
서보가 실제로 해낼 수 있는 것을 따라간다는 주장은 다시 측정하는 것만으로는 부족하고 다시
유도해야 하는데, 이는 40이나 80을 절제 실험 이상의 것으로 채택하기 전에 필요하다. 그
재유도, 그리고 그것이 (b)의 델타 액션 공간까지 필요로 하는지는 픽스처 변경 자체와 함께
오너에게 남긴다.

### 7.27 구현 결과 (V19): LeRobot 자신의 ACT, 그리고 데모가 동작한다

패킷 `docs/packets/M5/V19-external-act-current-data.ko.md`. V8(7.16절)은 **바깥에서** 설계·학습된
정책이 여기서 동일한 의미론으로 도는지를 물었고, 의미론에는 *그렇다, 비트 단위로*, 과제에는
0/16, 0/16, 1/16으로 답했다. 그 숫자는 전부 그 뒤로 세 번 수리된 하네스 위에서 측정된 것이고 —
V11(데모가 50 Hz Deployment IR을 두고 200 Hz로 돌았다), V12(관측이 스텝 이후 상태와 짝지어져
있었다), V17(청크의 0행을 제외한 모든 행이 죽어 있었다) — 지금 200개가 있는 자리에서 시연
50개로 학습한 것이다. **V19는 V17에 대해 하나만 바꾼다: 학습기.** 같은 Task IR, 같은 Deployment
IR과 Safety Plane, 같은 1,800스텝 예산, 같은 시드, 같은 `success_rate >= 0.5`.
`task_hash eb6efefa…`와 `deployment_hash 5ee54574…`는 V17의 것이고 움직이지 않는다.

**판정부터.** 현재 술어에 맞춰 수집된 시연 위에서 LeRobot 자신의 ACT는 60,000스텝 체크포인트에서
**홀드아웃 시드 101–116에 대해 `success_rate = 0.8125` (13 / 16)** 에 도달한다. 손대지 않은 수용
임계값 0.5에 맞서서이고, Evaluation IR의 6-스위트 스윕도 **통과**한다. plan V가 임계값을 넘은 것은
처음이고, 데모는 이제 영상으로 존재한다. 패킷이 지목한 시연 — *이전* 술어에 맞춰 수집된 V14의 것
— 위에서는 같은 레시피가 0 / 16을 낸다. 그 간극을 설명하는 측정은 정책이 아니라 데이터에 있다.

**상태 특징은 V17의 두 상태 포트를 이어 붙인 것이다.** V17의 정책은 팔의 관절각 6개 **그리고**
V7a의 시뮬레이터 특권 `sim_cube_pose`를 읽는다. V8의 ACT는 앞의 하나만 읽었는데,
`dataset_to_policy_features`가 정책에 `observation.state` 특징을 정확히 하나만 주기 때문이다.
V8의 6차원을 그대로 쓰면 학습기 *그리고* 그것이 볼 수 있는 것이 함께 움직이므로, 둘을 `qpos`가
이미 담고 있는 순서대로 LeRobot이 허용하는 하나의 특징으로 합친다. 내보내기는
`--state-dim 13`(`qpos[0..13]` — 관절각 6개 다음에 큐브의 7값 free-joint 자세)이고, Observation
IR은 observation-v8의 그래프에 observation.toml의 큐브 가지를 축 0의 `ObservationNode::Concat`
으로 붙인 것으로, 커밋된 두 픽스처에서 `python/es/make_v19_observation.py`가 쓰며 그것들을
대체하지 않는다. 두 가지 모두 항등 `Range{0..1}`로 정규화하는데 이유는 V8의 것이다 — ACT 자신의
`MEAN_STD` 통계가 순전파의 첫 연산이다 — 그리고 `Concat`의 포트는 `Frame::Policy`, 즉 IR이 이미
어떤 프레임과도 호환된다고 정의해 둔 프레임이며, 네트워크에 건네지는 특징 벡터란 바로 그것이다.
`es-ir` 변경은 없다. `Concat`은 M1부터 `ObservationNode`였고 `es-compile`이 `Op::Join`으로
낮춘다. `observation_hash`는 **`07fad282…`**, 이 Task IR 위의 세 번째 Observation IR이다(§7).

**논지, 그리고 그것은 충족되었다.** `crates/es-policy/tests/act_checkpoint.rs`는 진짜
`lerobot.policies.act.modeling_act.ACTPolicy`와 우리의 `TorchRuntime`을 이 프로젝트 자신의
데이터셋에서 기록한 같은 프레임(내보내기의 프레임 20: 13값 상태와 96x96 타일) 위에서 돌린다.
평가한 모든 체크포인트에서 — 두 내보내기에 걸쳐 **여덟** 개(패킷 실행의 20k / 50k / 100k,
애블레이션 1의 20k / 40k / 50k / 60k / 100k), `lerobot 0.6.1`, `torch 2.11.0+cu129`:

| | 청크 | `max_abs` | `max_rel` |
|---|---|---|---|
| 두 내보내기의 모든 체크포인트 | `[16, 6]` | **0e0** | **0e0** |

스펙 §8.9 tier 4는 `<= 1e-5`를 요구하고, 측정값은 **0**이다. CVAE와 DETR 디코더와 ImageNet
사전학습 백본을 지닌 정책 — 어느 것도 이 프로젝트의 Learning IR이 기술할 수 없는 것 — 위에서.
그것이 §8.1의 약속을 우리가 저작하지 않은 산출물 위에서, 현재 문서와 현재 케이던스로 이행한
것이다.

**실행이 드러낸 결함 하나, 정책이 아니라 가져오기에 있다.** `es policy import-lerobot`은
Deployment IR의 `deadlines.inference_budget`을 `RuntimeHints::deadline_ms`와
`RuntimeHints::expected_latency_ms` **양쪽**에 써 넣고 있었다. 후자는
`es_ir::learning::RuntimeHints`에 "§0.3 기준 장치에서의 측정값이지 약속이 아니다"라고 적혀
있고, `LRN-052`는 `latency <= min(deadline, 1000 / replanning_hz)`를 검사한다. 예산이 V8의
40 ms이고 재계획 주기가 200 ms인 동안에는 두 해석이 일치했다. V17이 같은 파일이 선언한 5 Hz
재계획에 대해 **240 ms** 예산을 진술하자, `LRN-052`는 가져온 *모든* 체크포인트를 —
`measured latency 240.0 ms, budget 200.0 ms` — 그 배치가 스스로 선언한 주기가 금지하는 지연을
주장한다는 이유로 거절했다. 가져오기는 이제 **상한**, `min(예산, 재계획 주기)`를 선언한다
(`declared_latency_ms`와 그 단위 테스트). V8의 40 ms는 그대로이고 V17의 배치에서는 200 ms이며,
정직한 귀결은 그 숫자가 만들어지는 자리에 적혀 있다 — 같은 명령이 지어낸 숫자를 규칙이 검사할
수는 없으므로 이 경로에서 `LRN-052`가 잡을 것은 남아 있지 않다. 측정된 지연은 `es bench`
(§12.4)에서 온다.

**학습.** LeRobot 자신의 트레이너와 ACT 기본값(`IMAGENET1K_V1`의 `resnet18`, `use_vae`,
`latent_dim 32`, `dim_model 512`, 인코더 4 / 디코더 1층, `kl_weight 10.0`, `lr 1e-5`, 전부
`MEAN_STD`), Deployment IR이 horizon 16을 버퍼링하므로
`--policy.chunk_size=16 --policy.n_action_steps=16`, `--steps=100000 --batch_size=8 --seed=0
--save_freq=10000`. 두 번째 학습과 평가를 함께 돌리는 RTX 4090에서 100,000스텝에 2,239 – 2,773초.
손실 6.3(스텝 200) → 0.115(10k) → 0.053(20k) → 0.038(50k) → **0.028**이며, 두 내보내기에서
세 자리까지 같다.

#### 패킷의 실행 (`s13`): 패킷이 지목한 데이터셋

`~/artifacts/plan-v/v14/ds-train`, 200 에피소드, 35,918 프레임, 내보내기 content 해시
`c08d1a39…`. nominal 스위트, 1,800스텝 예산, V17 케이던스의 `es eval run`. "옮김"은 큐브가 3차원
바구니 부피 안에 든 것, "놓음"은 그 상태에서 하네스의 옛 `gripper > 0.6`이며, 둘 다 V17의
`probe.py`가 `.estraj`에서 읽는다.

| 체크포인트 | 스위트 | `success_rate` | 들어올림 | **바구니로 옮김** | 놓음(>0.6) | `envelope_violation_rate` | `episode_length` |
|---|---|---|---|---|---|---|---|
| 10,000 | 홀드아웃 | 0.0000 | 12/16 | **12/16** | 1/16 | 0.9328 | 1800.0 |
| 20,000 | 학습시드 | 0.0000 | 11/16 | **10/16** | 7/16 | 0.9593 | 1800.0 |
| 20,000 | 홀드아웃 | 0.0000 | 7/16 | **7/16** | 5/16 | 0.9580 | 1800.0 |
| 50,000 | 학습시드 | 0.0000 | 14/16 | **14/16** | 0/16 | 0.9602 | 1800.0 |
| 50,000 | 홀드아웃 | 0.0000 | 10/16 | **9/16** | 0/16 | 0.9684 | 1800.0 |
| 100,000 | 학습시드 | 0.0000 | 12/16 | **12/16** | 0/16 | 0.9610 | 1800.0 |
| 100,000 | 홀드아웃(두 번) | 0.0000 | 9/16 | **9/16** | 0/16 | 0.9678 | 1800.0 |

100,000에서의 홀드아웃 두 번은 **바이트 단위로 동일한 `report.json`**을 냈다. 같은 스위트,
같은 예산에서 V17의 최고 기록과 견주면 홀드아웃 기준으로 들어올림 6/16 → **10/16**, 옮김
5/16 → **9/16**, `success_rate` 0.0625 → 0.0000이다.

**그 에피소드들이 끝나는 자리는 공중이 아니다.** 옮긴 에피소드마다 큐브는 z ~ 0.11 m까지
들어올려져 바구니 위로 운반된 뒤 거기 놓인다. 에피소드의 남은 내내 `z = 0.0279 m`, `|vz| = 0`
이고, 이는 바구니 바닥이며 시연 자신이 끝나는 안착 높이와 같다. 그리퍼는 0.19 – 0.31 rad로,
30 mm 큐브를 문 턱(0.093 rad)의 약 2.5배다. **큐브는 잡혀 있지 않다. 내려놓아졌다.** 성공 콘이
읽는 것은 턱이고, 그것은 `> 0.85`를 요구한다.

**이유, 논변이 아니라 데이터로 측정한 것.** V15(7.23절)는 그 항을 `> 0.6`에서 `> 0.85`로 올리고
시연 200개를 **다시 수집**했다. 에피소드는 술어가 처음 참이 되는 틱에 끝나므로 임계값이 곧
"놓기의 얼마만큼이 데이터에 남는가"를 정하기 때문이다. `~/artifacts/plan-v/v14/ds-train`
(35,918 프레임)은 **V15 이전** 수집이고, V15 자신의 `ds-train`(36,960 프레임)이 V15와 V17이
학습한 것이다. 각 40 에피소드에서, 큐브가 바구니의 x 구간 안에 있는 동안 기록된 턱 각도:

| 시연 | 프레임 | 큐브가 바구니 안일 때 턱의 최대 각 | 마지막 큐브 z |
|---|---|---|---|
| `v14/ds-train` (술어 `> 0.6`) | 35,918 | **0.572 – 0.576 rad** (중앙값 0.573) | 0.0274 – 0.0278 |
| `v15/ds-train` (술어 `> 0.85`) | 36,960 | **0.825 – 0.849 rad** (중앙값 0.832) | 0.0279 |

즉 35,918 프레임 집합에는 천장이 박혀 있다. 그것을 *완벽하게* 모방하는 자는 턱을 약 0.573 rad
까지 열고 멈추며, 이는 Task IR이 이제 요구하는 항에 0.28 rad 모자란다 — 그리고 이 ACT가 실제로
하는 것(바구니 위 최대 0.35 – 0.58)과 몇 밀리라디안 차이다. 위의 0/16은 정책이 자기 시연을 배우지
못한 것이 아니다. 더는 존재하지 않는 술어에 맞춰 수집된 집합을 정책이 배운 것이다.

#### 애블레이션 1 (`w13`): 같은 ACT를 V15의 재수집 시연으로

위 문단이 예측이고, 예측을 시험하는 가장 싼 방법은 그것을 해 보는 것이기에 돌렸다. 나머지는 모두
동일하다 — 같은 Observation IR(`07fad282…`), 같은 Task·Deployment IR, 같은 시드, 같은 레시피 —
`v15/ds-train`(200 에피소드, 36,960 프레임, 내보내기 content 해시 `55a958e1…`)에 대해서.
60,000스텝의 번들은 `~/artifacts/plan-v/v19/w13-060000.esb`다.

| 체크포인트 | 스위트 | `success_rate` | 들어올림 | 옮김 | 놓음 | `envelope_violation_rate` | `episode_length` |
|---|---|---|---|---|---|---|---|
| 20,000 | 학습시드 | 0.0000 | 9/16 | 9/16 | 0/16 | 0.9721 | 1800.0 |
| 20,000 | 홀드아웃 | 0.0000 | 8/16 | 8/16 | 0/16 | 0.9679 | 1800.0 |
| 40,000 | 홀드아웃 | 0.6875 (11/16) | 11/16 | 11/16 | 11/16 | 0.9755 | 744.8 |
| 50,000 | 학습시드 | 0.5000 (8/16) | 9/16 | 8/16 | 8/16 | 0.9677 | 1025.9 |
| 50,000 | 홀드아웃(두 번) | 0.6250 (10/16) | 10/16 | 10/16 | 10/16 | 0.9687 | 831.9 |
| **60,000** | **학습시드** | **0.7500** (12/16) | 13/16 | 12/16 | 12/16 | 0.9509 | 644.9 |
| **60,000** | **홀드아웃(두 번)** | **0.8125** (13/16) | 13/16 | **13/16** | **13/16** | 0.9692 | **539.9** |
| 100,000 | 학습시드 | 0.0000 | 8/16 | 8/16 | 0/16 | 0.9676 | 1800.0 |
| 100,000 | 홀드아웃 | 0.0625 (1/16) | 11/16 | 11/16 | 1/16 | 0.9506 | 1705.2 |

두 번 돌린 홀드아웃은 모두 **바이트 단위로 동일**했고, 50,000의 쌍은 서로 다른 `--jobs`
값(6과 4)에서 돌았는데도 바이트 단위로 일치했다.

60,000에서의 에피소드별 표, 홀드아웃 시드 101–116(`~/artifacts/plan-v/v19/e-w13-060000-holdout`):

| 셀 | 틱 | 들어올림 mm | 들어올림 @ | 옮김 @ | 놓음 @ | 마지막 턱 | 하네스 |
|---|---|---|---|---|---|---|---|
| nominal-00 | 248 | 127.6 | 105 | 163 | 242 | 0.801 | **성공** |
| nominal-01 | 218 | 124.9 | 107 | 166 | 203 | 0.848 | **성공** |
| nominal-03 | 234 | 130.1 | 121 | 180 | 222 | 0.836 | **성공** |
| nominal-04 | 286 | 138.8 | 110 | 196 | 278 | 0.834 | **성공** |
| nominal-05 | 229 | 129.6 | 128 | 195 | 223 | 0.802 | **성공** |
| nominal-06 | 248 | 133.7 | 109 | 183 | 239 | 0.843 | **성공** |
| nominal-07 | 226 | 132.6 | 104 | 182 | 219 | 0.830 | **성공** |
| nominal-08 | 246 | 134.6 | 127 | 195 | 241 | 0.799 | **성공** |
| nominal-10 | 229 | 124.4 | 115 | 164 | 213 | 0.846 | **성공** |
| nominal-11 | 254 | 124.3 | 127 | 183 | 239 | 0.847 | **성공** |
| nominal-12 | 297 | 165.6 | 136 | 230 | 280 | 0.840 | **성공** |
| nominal-13 | 274 | 145.0 | 131 | 210 | 259 | 0.849 | **성공** |
| nominal-15 | 250 | 123.8 | 138 | 193 | 245 | 0.808 | **성공** |
| nominal-02 / -09 / -14 | 1800 | 3.0 – 10.3 | — | — | — | −0.02 – 0.72 | 타임아웃 |

각각 **4.4 – 5.9초** 만의 깔끔한 성공 13번이고, 스크립트 전문가가 약 7초 걸리는 과제다. 실패
셋은 모두 같은 실패다. 집기가 빗나가 큐브가 테이블을 떠나지 않는다(들어올림 3 – 10 mm). 마지막에
기록된 턱은 0.799 – 0.849인데, `.estraj` 행은 **스텝 이전** 상태이고 콘은 스텝 이후 상태로
판정되므로 표의 0.80이 `> 0.85` 성공인 것이다. 그리고 그것은 세 자리까지 시연이 끝나는
0.825 – 0.849와 같다. 정책은 스승의 놓기를 재현하고, 스승의 놓기가 바로 그 술어가 쓰인 대상이다.

**총량보다 스케줄이 중요하다.** 20,000스텝은 아무것도 놓지 않고, 40,000 – 60,000은 홀드아웃 시드
16개 중 11 – 13개에서 놓으며, 100,000은 1개에서 놓는다. 놓기는 183프레임 에피소드의 19프레임
꼬리(7.23절)이므로 그에 대한 적합은 일시적이고 "더 오래 학습"은 틀린 손잡이다. 두 내보내기가 같은
모양을 보이며, 패킷 자신의 `--steps=100000`도 마지막 체크포인트만 평가했다면 이를 놓쳤을 것이다.
이는 ACT가 아니라 이 데이터셋에 관한 교훈이다.

**Evaluation IR의 6-스위트 스윕, 60,000, 홀드아웃 시드.** `passed: true` — 수용 기준
(`nominal success_rate >= 0.5`)이 문서 자신의 판정으로 충족된다.

| 스위트 | `success_rate` | `envelope_violation_rate` | `episode_length` |
|---|---|---|---|
| nominal | **0.7500** | 0.9701 | 657.9 |
| `light_intensity` | 0.3750 | 0.9763 | 1237.8 |
| `light_direction` | 0.6250 | 0.9467 | 925.7 |
| `observation_delay` | **0.8750** | 0.9586 | 490.7 |
| `torque_noise` | 0.1875 | 0.9760 | 1580.8 |
| `backlash` | 0.7500 | 0.9666 | 686.0 |

정책은 제어 스텝 1–2개의 관측 지연(0.875, nominal보다 낫다)과 백래시에 견디고, 조명 세기 게인
아래서는 절반으로 떨어지며 — 한 가지 조명 조건에서 학습한 비전 정책이다 — `torque_noise`
(0.1875)에서 가장 나쁘다. 96x96 프레임이 보여 줄 수 없는 바로 그것을 흔들기 때문이다.

**영상.** `es video showcase`를 60,000 홀드아웃 실행 위에서 V9의 카메라(eye
`(0.66, −0.46, 0.52)`, look-at `(0.14, −0.04, 0.04)`, fovy 36°, 1280x720)로 `nominal-00`,
`nominal-11` 셀에 대해, 그리고 `es video mosaic --grid 4x4`를 같은 실행의 관측 타일 위에
Safety Plane 오버레이와 함께 돌렸고, 전부 `python/es/encode_video.py`가 인코딩했다.
`v19-60k-nominal-00.mp4`, `v19-60k-nominal-11.mp4`, `v19-60k-mosaic.mp4`이며
`target/plan-v/v19/`에 미러했다(50,000스텝 세트가 그 옆에 있다). §5.3에 따라 mp4 컨테이너는 해시
체인 안에 있지 않다. 증거는 프레임이다.

#### 애블레이션 2 (`s6`): V8 자신의 6차원 관절 전용 상태

V8의 ACT에는 큐브 자세가 없었고 V17의 IR 그래프 정책에는 있었으므로, "학습기"와 "학습기가 볼 수
있는 것"을 분리하려고 돌렸다. 35,918 프레임 내보내기에서 100,000스텝까지 학습했고(손실
6.615 → 0.029, 2,767초) **평가하지 않았다**. 박스가 애블레이션 1에 필요했고, 위에서 측정한 천장이
그대로 적용되므로 그 평가는 입력이 하나 줄어든 채 같은 데이터 결함을 다시 재는 일이 된다. 외부
ACT에 특권 큐브 자세가 필요한지는 따라서 **여기서 답하지 않았고**, 남은 실험 중 가장 싸다.
체크포인트는 있다.

#### 여전히 드는 비용, 그리고 실행이 미리 알지 못한 두 가지

`envelope_violation_rate`는 두 내보내기의 모든 실행에서 0.95 – 0.98이고,
`violation.acceleration`이 제어 틱의 대부분에서, `EnvelopeViolationRate` 워치독이 592 – 2,134틱
동안 폴백을 걸어 잠근다. 이는 V17의 열린 질문 25가 종류로도(거기서는 0.998) 원인으로도 그대로인
것이다. 청크의 0..9행을 순서대로 실행하면 `acceleration_max = 20 rad/s²`가 허용하는 것보다 틱당
이동이 커진다. **플레인이 거의 매 틱 클램프하는 가운데 데모가 성공한다**는 점은 분명히 말해
두어야 한다. 이것은 깔끔한 명령이 아니라, 움직이지 않은 봉투(`INV-12`) 안에 거친 명령 흐름을
잡아 두는 Safety Plane이다. `wrist_flex`(틱의 약 0.1 %)를 빼면 어느 팔 관절도 위치 한계를 넘지
않고, V14 에피소드의 절반 동안 아래쪽 소프트 경계에 박혀 있던 그리퍼는 0.0 %다.

1. **10틱 재계획 아래의 16행 청크는 V17과 다른 실행 체제다.** 낮춰진 ACT 청크는 `[16, 6]`이고
   `replan_interval`은 제어 틱 10개이므로 **두 청크가 동시에 살아 있고**, `TemporalEnsemble`이
   새 청크의 `k`행과 이전 청크의 `k + 10`행을 평균한다. V17의 IR 그래프 정책은 10행 청크였고,
   같은 산술이 구성원 하나로 퇴화한다(7.25절). 특별 취급은 없고 숫자는 Deployment IR 자신의
   것이지만, "IR 그래프 대 외부 ACT" 비교는 이 차이를 품고 있으며, 20,000과 100,000이 턱을 덜
   여는데 40,000 – 60,000은 그렇지 않은 이유의 검증되지 않은 후보 하나다.
2. **다른 다섯 스위트가 문서에 있으면 nominal 스위트의 숫자가 움직인다.** 50,000에서 nominal만
   남긴 단독 실행은 0.6250, 같은 체크포인트의 6-스위트 스윕의 nominal 행은 0.5625이고 성공한 셀도
   다르다. 샤딩 때문이 아니다. 단독 문서를 `--jobs 8`로 다시 돌리니 `--jobs 6` 실행과 **바이트
   단위로** 0.6250이 재현되었고, `--jobs 4` 반복도 또 바이트 단위로 일치했다. 그러니 차이는
   문서이고, §10.4의 "모든 스위트는 자기 `stream`에서 뽑으므로 행을 하나 더해도 다른 행의 숫자를
   움직일 수 없다" — 7.8절 이후가 기대 온 문장 — 은 측정된 이 쌍에 대해 성립하지 않는다. 이 절의
   모든 숫자는 측정에 쓰인 문서와 함께 적혀 있다. 그것이 열린 질문 28이다.

**V8과 V17에 견주어 읽기.** V8은 모델이 문제가 아니라고 결론지었다. 우리의 IR 모양 그래프가
0–1/16을 낼 때 진짜 LeRobot ACT가 1/16을 냈기 때문이다. 그 결론은 모델에 대해서는 옳았고 거기서
따라 나오는 것에 대해서는 틀렸다. 하네스가 세 군데 망가져 있었고 학습 집합은 그 뒤 바뀐 술어에
맞춰 잘못된 케이던스로 모은 50 에피소드였다. 세 수리를 모두 적용하고 현재 술어에 맞춰 수집된 시연
200개를 쓰면, 같은 외부 ACT가 **홀드아웃 13/16**을 낸다. 같은 데이터와 같은 하네스에서 IR 모양
그래프는 1/16이다(V17). V8이 가르지 못한 두 가설은 이제 갈라진다. **모델도 문제였고, 하네스도
문제였고, 시연도 문제였다** — 그리고 셋을 모두 고쳤을 때에만 데모가 나왔다. V8이 증명했고 V19가
그대로 다시 증명한 것은 한번 측정된 뒤로는 의심된 적 없는 부분이다. 이 프로젝트 바깥에서 통째로
학습된 ACT가 이 프로젝트의 Observation IR, Deployment IR, Safety Plane, 평가를 거쳐 비트 단위로
같은 함수라는 것. 데모는 존재하고, 그 안의 정책은 우리 것이 아니다.

**검토자 주석(오케스트레이터, 2026-09-16).** 13/16에는 두 가지 단서가 따라간다. 첫째, 이 절의
모든 평가는 V19의 기준 커밋 시점의 배포(`acceleration_max = 20`)로 돌았으므로
`envelope_violation_rate` 0.95–0.98과 워치독 래치는 V18 이전의 엔벌로프다. 픽스처는 이제 80이고
(7.28절) w13 체크포인트는 7.29절에서 커밋된 문서로 재측정한다. 둘째, 60,000 step은 40k/50k/60k/100k를
같은 열여섯 시드로 모두 평가한 뒤 held-out 점수로 고른 것이다. 별도의 검증 분할이 없으므로 0.8125에는
평가 시드에 대한 선택이 섞여 있고, 6스위트 스윕의 nominal 0.75가 더 보수적인 숫자다. 두 단서 모두
논지의 결과(모든 체크포인트에서 정확히 0)에는 닿지 않는다.

### 7.28 만든 대로 (V18b + 픽스처 결정): 시연이 체크인된 문서로 자기 Evaluation IR을 통과한다

패킷 `docs/packets/M5/V18b-sweep-and-showcase.md`. 하루(2026-09-16)에 두 가지가 일어났다. 오너가
열린 질문 25를 결정했다: `tests/fixtures/visible-learning/deployment.toml`의 `acceleration_max`가
80 rad/s²로 옮겨졌다(커밋 `74b01a3`, 재유도는 픽스처의 주석에, `velocity_max` 불변). 그리고 V18b가
Evaluation IR **전체** — 픽스처가 선언한 여섯 스위트, 홀드아웃 시드 101–116, 예산 1800 — 를 V18의
40·80 두 단계에서 돌리고, 80에서 V9 쇼케이스 파이프라인을 돌렸다. V18의 80 단계 절제 사본과
커밋된 픽스처는 같은 문서로 파싱되므로 `es ir check`가 둘에 같은 `deployment_hash`를 찍는다
(`f2f9a510…`; 40 사본은 `38e56f4a…`). 그래서 아래 80 단계 행은 더 이상 절제 실험이 아니라
체크인된 문서의 기록이며, 체크포인트는 V15의 40,000 step 그대로다.

**채택된 값에서의 전체 스윕** (`passed = true`; `evaluation_hash e52e8360…`, 배포는 이 해시에 들어가지
않는다; `execution_hash 52dfeba9…`):

| 스위트 | `success_rate` | `envelope_violation_rate` | 평균 `episode_length` | 폴백 틱 |
|---|---|---|---|---|
| nominal (게이트) | **0.625** | 0.555 | 936 | 53 |
| light_intensity | 0.625 | 0.542 | 889 | 29 |
| light_direction | 0.500 | 0.662 | 1190 | 103 |
| observation_delay | 0.0625 | 0.751 | 1719 | 395 |
| torque_noise | 0.0625 | 0.670 | 1731 | 383 |
| backlash | 0.625 | 0.694 | 1021 | 287 |

택하지 않은 대안 40도 통과하지만(`nominal` 0.625) backlash를 뺀 모든 섭동 스위트에서 더 나쁘고
워치독이 계속 래치한다(nominal 폴백 767틱 대 53틱). 이 체크포인트가 견디지 못하는 두 섭동 —
`observation_delay`와 `torque_noise`, 두 단계 모두 16편 중 1편 — 은 엔벌로프가 아니라 정책의
성질이다. 홀드아웃 nominal 숫자는 두 단계 모두 V18 자신의 것과 비트 단위로 같다(같은
`evaluation_hash 40623e01…`, 같은 리포트). 80에서의 홀드아웃 성공 열 편은 시드 101, 102, 104, 105,
107, 108, 110, 112, 113, 116이다.

**영상.** 기록된 `.estraj` 궤적에서 V9 파이프라인으로, 카메라는 7.17절 그대로:
`v18b-L80-holdout-nominal-00.mp4`(시드 101, 제어 틱 206, 1280×720 30 fps),
`v18b-L80-holdout-nominal-12.mp4`(시드 113, 183틱), `v18b-L80-holdout-mosaic.mp4`(홀드아웃 16편을
정책 자신의 96×96 관측 타일 4×4로, 1,800 프레임 50 fps). 서버 `~/artifacts/plan-v/v18b/showcase/`,
로컬 `target/plan-v/v18b/showcase/`에 미러(추적 안 함; 저장소는 영상을 담지 않는다).
오케스트레이터가 표본 프레임을 확인했다: 팔이 큐브 위에서 시작해 잡고, 통 위로 옮겨, 큐브를 안에
둔 채 집게를 연다.

**이것이 무엇이고 무엇이 아닌가.** 플랜 V의 수용 기준이 커밋된 문서 그대로에서 성립한 첫 번째
경우다: 여기서 학습하고 여기서 돌린 정책이, 자신을 위해 쓰인 Evaluation IR을 통과하고, 사람이 볼 수
있는 영상이 있다. 그러나 아직 프로젝트의 논지(외부에서 학습된 정책의 비트 동일 재현 — V19, 7.27절)는
아니고, 이 정책은 배포 가능하지 않다: 상태 포트로 시뮬레이션의 특권 큐브 자세를 읽으며(V7a), 이를
실기에서 막는 것이 §28.9 사다리 14의 provenance 검증기다. V13부터 V17까지의 모든 숫자는 정책의
틱 99.8 %를 깎던 엔벌로프 뒤에서 측정됐다. 0.0625와 0.625 사이에서 바뀐 것은 런타임과 그 문서였지
가중치가 아니다.

**남긴 것 하나(열린 질문 26).** 80 단계 렌더가 같은 GPU에서 도는 동안 40 단계를 재현하니 리포트와
`evaluation_hash`는 같은데 `execution_hash`만 달랐다(`745c9777…` 대 단독 실행의 `c2322c2c…`). 숫자는
움직이지 않았고 출처만 움직였다. V18b는 추적하지 않았다.

### 7.29 만든 대로 (V19b): 커밋된 문서 위의 외부 ACT

패킷 `docs/packets/M5/V19b-external-act-committed-documents.md`. V19의 숫자(7.27절)는 그 브랜치의
기준 시점 배포 — `acceleration_max = 20` — 로 잰 것이고 그사이 픽스처는 80으로 옮겨졌다(7.28절). 이것은
같은 LeRobot ACT의 같은 60,000 step 체크포인트(`~/artifacts/plan-v/v19/train-w13/checkpoints/060000`,
V15의 시연 200편으로 학습)를 머지된 트리의 `es policy import-lerobot`으로 커밋된 `deployment.toml`에
대해 다시 임포트하고 다른 어떤 것도 바꾸지 않은 채 평가한 것이다: `task_hash eb6efefa…`,
`observation_hash 07fad282…`(V19의 병합 상태 포트), `learning_hash 17613075…`, `policy_hash 308a1f53…`,
`deployment_hash f2f9a510…` — 트리의 픽스처에 대해 `es ir check`가 찍는 해시. 재학습 없음, 코드 변경
없음; 오케스트레이터의 실행, `~/artifacts/plan-v/v19b/run.sh`.

**커밋된 문서 그대로의 수용 기준**(예산 1800, `ES_PYTHON=~/venvs/es`, nominal 두 실행은 `--jobs 4`,
스윕은 `--jobs 6`):

| 문서 | 스위트 | `success_rate` | `envelope_violation_rate` | 평균 `episode_length` | 폴백 틱 |
|---|---|---|---|---|---|
| nominal, 학습 시드 1–16 | nominal | **1.0000** (16/16) | 0.232 | 230 | 0 |
| nominal, 홀드아웃 시드 101–116 | nominal | **0.9375** (15/16) | 0.208 | 324 | 0 |
| 6스위트 스윕, 홀드아웃 시드 | nominal | 0.9375 | 0.185 | 323 | 0 |
| | light_intensity | 0.9375 | 0.177 | 334 | 0 |
| | light_direction | 0.8125 | 0.381 | 582 | 4 |
| | observation_delay | **1.0000** | 0.285 | 262 | 0 |
| | torque_noise | 0.3125 | 0.500 | 1379 | 0 |
| | backlash | 0.9375 | 0.254 | 355 | 0 |

세 문서 모두 `passed = true`. nominal 스위트의 유일한 홀드아웃 실패 `nominal-04`(시드 105)는 파지
실패다: 집게가 1,800틱 중 1,475틱 동안 열린 채이고 큐브는 테이블을 떠나지 않는다. 성공 열다섯 편은
198–304 제어 틱, 즉 4–6초에 끝나며 스크립트 시연은 7초다.

**같은 체크포인트·같은 시드의 `acceleration_max = 20` V19와 비교:** 홀드아웃 13/16 → **15/16**,
`envelope_violation_rate` 0.969 → **0.208**, 워치독 래치 틱 592–2,134 → **0**; 스윕의 nominal 0.75 →
0.9375, `light_intensity` 0.375 → 0.9375, `light_direction` 0.625 → 0.8125, `observation_delay` 0.875 →
1.0, `torque_noise` 0.1875 → 0.3125, `backlash` 0.75 → 0.9375. 정책은 아무것도 바뀌지 않았다. V18이 IR
그래프 정책에서 잰 것과 같은 움직임이다: 스무 틱 중 열아홉을 깎던 플레인이 명령 스트림에서 비켜선
것이다. `torque_noise`는 96×96 프레임이 토크를 볼 수 없는 정책에 여전히 어려운 스위트이고,
`light_direction`은 플레인이 조금이라도 폴백하는 유일한 스위트다(16편에서 4틱).

**같은 문서 위의 IR 그래프 정책과 비교**(V18b, 7.28절: 홀드아웃 nominal 0.625, `light_intensity` 0.625,
`observation_delay` 0.0625, `torque_noise` 0.0625): 외부에서 설계되고 외부에서 학습된 학습기가 모든
스위트에서 더 낫고, IR 그래프 정책이 견디지 못한 두 스위트에서는 큰 차이로 낫다. 같은 시연 200편, 같은
하네스, 같은 문서. 그것이 논쟁이 아니라 측정으로 확인된 오너의 논지다: 학습 설계는 외부에서 오고, 이
프로젝트의 일 — 선언된 의미로 돌리고 비트까지 재현하는 것 — 이 끝난 부분이다. 7.27절의 검토자 주석에서
두 단서가 그대로 따라온다: 60,000 step은 홀드아웃 시드로 골랐고(여기서 스윕의 nominal 행도 같은
0.9375라 선택이 수를 부풀리지는 않았다), 두 정책은 청크 길이가 다르다(16행 대 10행, M5 리뷰의 S-9).

**영상.** 홀드아웃 실행에 V9 카메라로 `es video showcase`, 셀 `nominal-00`(시드 101, 224틱)과
`nominal-11`(시드 112, 234틱), 1280×720 50 fps; 안전 플레인 오버레이를 얹은 홀드아웃 관측 스트림 16개의
`es video mosaic --grid 4x4`. 파일 `v19b-a80-holdout-nominal-00.mp4`, `v19b-a80-holdout-nominal-11.mp4`,
`v19b-a80-holdout-mosaic.mp4`는 `~/artifacts/plan-v/v19b/`와 `target/plan-v/v19b/showcase/`에;
오케스트레이터가 표본 프레임을 확인했다. 7.28절이 아니라 이것이 시연의 영상이다: 그 안의 정책은 우리
것이 아니다.

**이것으로 닫히는 것.** M5가 잘려 남은 세 가지(7.28절): 논지(비트 동일, 7.27절), 커밋된 문서 위의 수용
기준(이 절), 영상. 리뷰는 `docs/reviews/M5.ko.md`.

### 7.30 만든 대로 (T7): 존재하는 로봇 위의 평가기, 그리고 데모의 숫자가 어떻게 되었는가

패킷 `docs/packets/M7/T7-eval-latency.md`. V6b는 두 실행 경로가 `es_env::plane_chunk`를 통해
Safety Plane에 먹이도록 했고 V17은 둘 다 `rate.inference`를 지키도록 했다. 남은 것이 열린 질문
24였다. `DomainRunner`는 제어 틱 `t`에 관측을 제출하고 `t + latency_ticks(expected_latency_ms,
rate.control)`에 청크를 방출하므로 수집기의 plane은 틱 0에 빈 버퍼를 보고 자신의 fallback으로
답하는 반면, `es_eval::runner`는 같은 틱에 정책을 호출하고 0행을 실행했다. "두 경로가 청크를
동일하게 실행한다"는 스케줄에 대해 참이고 궤적에 대해 거짓이었다. T7이 그것을 궤적에 대해서도
참으로 만든다.

**무엇이 바뀌었나, 한 문단으로.** 셀 루프는 이제 `DomainRunner::new`가 넘기는 것과 같은 쌍
(`latency_ticks(contract.runtime.expected_latency_ms, rate.control)`, 추론 도메인의 배치)으로
구성한 `es_env::AsyncInference`를 통해 정책을 호출하고, 방출된 결과를
`apply_at = submit_tick + latency`로 `ChunkBuffer`에 넣는다 — App. B.5의
`computed_from + deterministic latency`이지 도착 틱이 아니다. 두 번째 지연 모델은 없다:
`latency_ticks`와 `AsyncInference`는 `es-env`에서 손대지 않았고 양쪽 경로가 호출하는 것이 바로
그것이며, `crates/es-env`의 `git diff`는 비어 있다. plane에 대해서는 아무것도 움직이지 않았다
(INV-12): 언더런은 plane 자신의 이벤트이고, 수집에서와 똑같이 집계되고 기록되고 처리된다.
문서의 숫자는 `RunConfig::expected_latency_ms`로 `es-eval`에 닿는다. `Evaluation::run`은 자신이
심판하는 네 IR만 받을 뿐 `LearningGraph`는 받지 않기 때문이다. `es eval run`이 자신이 연
번들에서 그것을 채운다(한 줄, 패킷의 context 블록에 명시). `expected_latency_ms = 0`은 계속
합법이고 0틱을 뜻하며, 그것이 이전에 돌던 루프 그대로다.

**없던 오라클.** `collection_and_evaluation_draw_the_same_trajectory`(`crates/es/tests/cli.rs`,
`#[ignore]`, 오라클 서버)는 데모 씬의 시드 1을 같은 스크립트 전문가로 `es loop collect`와
`es_eval::Evaluation`에 각각 돌리고 두 `.estraj`를 비교한다 — 틱별 `qpos ‖ qvel`을 원시 `f64`
비트로, 그 다음 파일 전체를. **수정 전**에는 틱 1, 원소 0에서 실패했다: 수집
`1.4392937947180706 × 10⁻⁷` 대 평가 `−4.3777706204127563 × 10⁻⁴`(열린 질문 24가 인용한 값을 이
트리에서 다시 측정한 것). 수정 후에는 120틱 중 120틱이 동일하고 body pose도 그렇다. V17의 형제
테스트 `collection_and_evaluation_ask_the_policy_at_the_same_cadence`는 이제 갈라짐을 출력하는
대신 사라졌음을 단언하고, `expert_passes_the_evaluation_harness`는 수집에서 늘 그랬던 선언된
지연 아래에서 전문가를 돌려 여전히 0.875로 통과한다.

**재측정** (재훈련 없음, 재import 없음: V19b 자신의 `w13-060000-a80.esb`, V18 자신의
`trained-L80.esb`, 커밋된 문서, `ES_PYTHON=~/venvs/es`, `--jobs 6`). 각 행의 "이전"은 7.29절에서
베껴 온 것이 아니라 **같은 장비에서 같은 venv와 같은 플래그로 병합 기준(main)에서 다시 실행한
것**이며, 7.29절의 커밋된 숫자를 정확히 재현했다. 따라서 아래의 모든 변동은 지연 모델이고 다른
무엇도 아니다. 산출물은 `~/artifacts/plan-v/m7-t7/`(`run.sh`, `control.sh`).

| 문서 | 스위트 | `success_rate` | `envelope_violation_rate` | 평균 `episode_length` | fallback 틱 | 언더런 틱 |
|---|---|---|---|---|---|---|
| V19b, 홀드아웃 nominal | nominal | 0.9375 → **0.9375** | 0.2083 → **0.5455** | 323.8 → **556.8** | 0 → **160** | 0 → **160** |
| V19b, 6-스위트 스윕 | nominal | 0.9375 → **0.9375** | 0.1849 → **0.5935** | 323.1 → **572.0** | 0 → **160** | 0 → **160** |
| | light_intensity | 0.9375 → **0.9375** | 0.1771 → **0.5732** | 333.6 → **586.9** | 0 → **160** | 0 → **160** |
| | light_direction | 0.8125 → **0.5625** | 0.3808 → **0.5600** | 582.2 → **1091.2** | 4 → **191** | 0 → **160** |
| | observation_delay | 1.0000 → **1.0000** | 0.2852 → **0.5814** | 262.1 → **502.4** | 0 → **160** | 0 → **160** |
| | torque_noise | 0.3125 → **0.0625** | 0.4997 → **0.5576** | 1379.2 → **1792.4** | 0 → **171** | 0 → **160** |
| | backlash | 0.9375 → **0.8750** | 0.2537 → **0.5431** | 354.8 → **686.9** | 0 → **160** | 0 → **160** |
| V18b IR-그래프 L80, 홀드아웃 nominal | nominal | 0.6250 → **0.0625** | 0.5552 → **0.6985** | 936.2 → **1692.0** | 53 → **536** | 0 → **16** |

**찾은 그대로의 `passed`.** 외부 ACT는 두 문서 모두에서 여전히 통과한다 — nominal에서
`success_rate ≥ 0.5`, 홀드아웃 실행과 스윕 모두 관측값 0.9375. **IR-그래프 정책은 통과하지
못한다:** 같은 0.5 기준에 0.0625, `passed = false`. 기준값은 어느 방향으로도 건드리지 않았다.

**왜 두 정책의 지연이 다른가 — 이것이 발견 아래의 발견이다.** 그 숫자는 각 번들 자신의 것이다.
픽스처 `learning.toml`은 `expected_latency_ms = 15`, 50 Hz에서 제어 1틱을 선언한다 — IR-그래프
번들이 지닌 값이고 `es loop collect`가 200개 시연을 기록한 값이다. import된 ACT는 **200 ms**,
제어 10틱을 지닌다. `es policy import-lerobot`이 선언할 측정값을 갖고 있지 않아 대신 *경계값*을
말하기 때문이다: `declared_latency_ms = min(inference_budget, 1000 / replanning_hz) =
min(240, 200)`(`crates/es/src/cmd/policy.rs`, 그곳의 doc 주석이 그대로 그렇게 말한다). T7
이전까지 그 숫자는 평가 경로에서 무해했다. 이제는 하중을 받으며, ACT의 경우 우연히도 정확히
재계획 주기와 같아서 파이프라인이 완벽히 채워진다: 틱 0에서 계산된 청크가 틱 10에 도착해 틱
10–19를 구동하고, 틱 10의 청크가 20에 도착한다. 그래서 **에피소드의 처음 열 틱만 언더런한다** —
16 에피소드에 걸쳐 160, 시작의 200 ms 홀드 한 번뿐 그 뒤로는 없다. IR-그래프 정책의 1틱은
에피소드당 언더런 하나, 16 에피소드에 16이다.

**이 변동이 뜻하는 것과 뜻하지 않는 것.** ACT의 `envelope_violation_rate`는 대략 세 배가 되고
(0.21 → 0.55) 에피소드는 약 1.7배 길어진다. 정책도 envelope도 바뀌지 않았다: 명령은 실행될
무렵 200 ms 낡아 있으므로, plane은 팔이 이미 떠난 상태를 뒤쫓는 명령 흐름을 rate-limit하고
과제는 닫히는 데 더 오래 걸린다. 그것이 선언된 그대로의 배포에 대한 정직한 숫자다 — 이전 숫자는
추론이 즉시 끝나는 로봇을 묘사한 것이었다. 성공률을 잃은 세 스위트(`light_direction`
0.81 → 0.56, `torque_noise` 0.31 → 0.06, `backlash` 0.94 → 0.88)는 이미 어려운 셋이었다.
200 ms의 개루프 구멍이 "어렵다"를 "실패한다"로 바꾼다.

IR-그래프 정책이 지연 **1틱**에 0.625에서 0.0625로 무너진 것이 더 날카로운 결과이고, 사람이
볼 값어치가 있는 쪽이다: 536개 fallback 틱 중 520개가 `EnvelopeViolationRate` 워치독의
래치다(이전 53). 즉 단 한 틱의 낡음이 200틱 창 안에서 그 명령 흐름을 `max_frac = 0.9` 위로
충분히 자주 밀어 올려 팔을 통째로 붙잡아 둔다. 7.29절의 "외부에서 훈련된 학습기가 모든
스위트에서 낫다"는 이것을 견디고 오히려 격차가 벌어진다. 7.28절의 "데모가 체크인된 문서 위에서
자신의 Evaluation IR을 통과한다"는 IR-그래프 정책에 대해 지연 0에서 측정된 것이고 **견디지
못한다** — 그 문장은 이 패킷이 다시 날짜를 매기는 것이지 지우는 것이 아니다(7.26절이 세운
규칙). 마일스톤을 닫은 M5 수용은 V19b의 것이고, 그것은 여전히 성립한다.

사람이 결정하고 싶을 만한 것 둘은 12절에 있다: 질문 24는 여기서 답했고, 새 질문 32는 이제
숫자를 결정하게 된 `import-lerobot`의 지어낸 지연이다.

### 7.31 만든 대로 (M7/U): T3–T6 이후의 IR 그래프 재측정, 그리고 중단 규칙이 말하는 것

패킷 `docs/packets/M7/U-measurement.md`. §28.10은 IR-그래프 정책을 커밋된 문서 위에서 **한 번**
재측정하도록 허용한다. T3(배치 축), T4(배치 64에서의 웜업+코사인), T5(ImageNet 백본),
T6(학습 증강)이 모두 그 아래의 로워링이나 트레이너를 움직였기 때문이다. 이 절이 그 측정이다:
설정 네 가지, V15의 같은 시연 200편, 넷 모두 T4의 row-D 설정으로 20,000 옵티마이저 스텝
(배치 64, lr 4e-4, `warmup_cosine` 웜업 250, `lr_min` 1e-6, 시드 0, `--resident-gpu`,
`device = "cuda"`), T7의 지연 모델 아래에서. 아무것도 튜닝하지 않았다. 발산한 행은 숫자가 들어
있는 행이다.

**중단 규칙, 적용. 그리고 한 줄로 끝나는 답이 아니다.** §28.10이 고정한 규칙은 *T5 뒤에도
held-out이 V18b의 0.625를 넘지 못하면 IR 그래프 튜닝은 여기서 끝이고 제품은 외부 경로의
속도다.* **U1 — T5 뒤 — 은 0.0000**이다. 학습이 발산했기 때문이다(아래). **U3 — T5와 T6 뒤 —
은 0.5625**이고 0.5625 < 0.625이므로 **문자 그대로 읽으면 중단 규칙은 발동한다.** 같은 조건끼리
읽으면 반대를 말하고, 사람이 둘 중 하나를 골라야 한다. V18b의 0.625는 평가 지연 0에서 측정된
값이고, 7.30절은 같은 체크포인트를 그 문서가 선언한 지연 아래에서 다시 재어 **0.0625**를 얻었다.
U0–U3가 돌아간 평가기가 내놓는 숫자에 대고 보면 U3는 V18b의 9배이고, **T7의 지연 모델 아래에서
데모의 수용을 통과한 최초의 IR-그래프 정책**이다(`passed = true`, held-out과 학습 시드 양쪽).
두 가지가 그것을 누그러뜨린다: U3의 Observation IR은 *다른 문서*이고(아래 해시 항목), 0.5625는
여전히 외부 ACT의 0.9375(7.30절)보다 낮다. 패킷은 다섯 번째 설정을 금지하므로, 이 노트는 두
읽기를 모두 적고 §28.10이 어느 쪽을 뜻했는지는 M7 리뷰가 결정한다.

**네 행.** 아래 모든 숫자는 `es eval run --jobs 6 --frames`, 커밋된 `evaluation.toml`
(held-out 시드 101–116 16편 × 여섯 스위트)와 `~/artifacts/plan-v/v15/eval-trainseeds.toml`
(nominal, 학습 시드 1–16), 오라클 서버(RTX 4090, 16코어)에서 나왔다. U0/U1과 쇼케이스는
커밋 **`3e6a2e8`**의 `~/Projects/es-u`에서, U2/U3는 같은 경로를 **`7c9d954`**(그 사이에 T6이
들어왔다)로 다시 푼 트리에서 돌았다. `cargo build --release --features render`,
`ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`(torch 2.11.0+cu129).

| 행 | Learning IR | Observation IR | 학습 | held-out `success_rate` | 학습 시드 | held-out `envelope_violation_rate` | `passed` |
|---|---|---|---|---|---|---|---|
| **U0** | `learning.toml` | `observation.toml` | T4 row D, 재학습 안 함 | 0.2500 | 0.1250 | 0.4409 | false |
| **U1** | `learning-pretrained.toml` | `observation.toml` | `es train`, **발산** | **0.0000** | 0.0000 | 1.0000 | false |
| **U2** | `learning.toml` | `observation-augmented.toml` | T6의 자체 실행, 재학습 안 함 | 0.1875 | 0.0625 | 0.4279 | false |
| **U3** | `learning-pretrained.toml` | `observation-augmented.toml` | `es train` | **0.5625** | **0.5625** | 0.6668 | **true** |

패킷 M7/R5가 다섯 번째 행 **U4** — 관측을 래스터화 대신 패스 트레이싱한 U3 — 를 [section 7.32](#732-as-built-m7r5-row-u4--같은-정책을-패스-트레이싱된-관측-위에서)에 더한다. *다른* 문서 집합이므로(`Pt` 센서는 `task_hash`를 움직인다) 체인이 아니라 판단으로 이 표를 늘린다. U2/U3이 U0/U1을 늘리는 것과 정확히 같다.

**여섯 스위트 스윕, held-out 시드** (`success_rate` / `envelope_violation_rate` / 평균
`episode_length`. `nominal` 행이 위 표의 held-out 열이다 — 커밋된 `evaluation.toml`은 문서
하나이고 그 nominal 스위트가 곧 held-out 측정이기 때문이다):

| 스위트 | U0 | U1 | U2 | U3 |
|---|---|---|---|---|
| nominal | 0.2500 / 0.4409 / 1492.7 | 0.0 / 1.0 / 1800.0 | 0.1875 / 0.4279 / 1525.4 | **0.5625** / 0.6668 / 1092.1 |
| light_intensity | 0.0625 / 0.4542 / 1699.4 | 0.0 / 1.0 / 1800.0 | 0.0000 / 0.4392 / 1800.0 | **0.6250** / 0.6236 / 921.1 |
| light_direction | 0.0625 / 0.4865 / 1711.8 | 0.0 / 1.0 / 1800.0 | 0.0000 / 0.4178 / 1800.0 | 0.1875 / 0.7265 / 1634.6 |
| observation_delay | 0.0625 / 0.5550 / 1721.8 | 0.0 / 1.0 / 1800.0 | 0.0000 / 0.5294 / 1800.0 | 0.2500 / 0.6506 / 1502.2 |
| torque_noise | 0.0000 / 0.6193 / 1800.0 | 0.0 / 1.0 / 1800.0 | 0.0625 / 0.7975 / 1709.8 | 0.0000 / 0.7157 / 1800.0 |
| backlash | 0.0625 / 0.4114 / 1699.4 | 0.0 / 1.0 / 1800.0 | 0.0000 / 0.4081 / 1800.0 | 0.4375 / 0.7031 / 1244.0 |

`torque_noise`는 이 표의 어떤 IR-그래프 정책도 살아남지 못하는 스위트이고, 7.28절과 7.30절이
IR 그래프에 대해서도 외부 ACT에 대해서도 이미 기록한 것이다: 96×96 프레임은 토크를 볼 수 없다.

**두 evaluation 해시, 그리고 왜 둘인가.** U0/U1은 커밋된 `evaluation.toml`
— `evaluation_hash e52e8360…`(V18b 자신의 것) — 과 `v15/eval-trainseeds.toml`(`b49fc549…`)로
판정했다. U2/U3는 그럴 수 없다: 그들의 Observation IR은 `observation-augmented.toml`,
`observation_hash cc437a24…`(커밋된 것은 `899c16a9…`)이고, Evaluation IR은 자신이 판정하는
observation을 이름으로 가리킨다. 그래서 그들의 문서는 커밋된 둘에서 그 필드 하나만 바꾼 것이며
(`~/artifacts/plan-v/m7-u/evaluation-augmented.toml`, `eval-trainseeds-augmented.toml`; 필드를
되돌린 `diff`는 비어 있다), 해시는 **`e5705cb0…`**와 **`10af1061…`**이다. §13.3: **U2/U3는 새
비교다.** 시드·스위트·섭동 파라미터·예산·수용 기준은 U0/U1의 것과 같은 바이트이므로 읽는
사람의 판단으로는 비교할 수 있다. 사슬은 비교하지 않으며, 위 어떤 행도 그 경계를 넘어 사슬로
이어졌다고 주장하지 않는다. 짚어 둘 결과 둘: `es eval compare`는 `evaluation_hash`가 다른 두
리포트 쌍을 **아무 말 없이** 받는다(`es loop cycle`은 바로 그것을 거부한다 —
`training-recipe.md` 12.3). 이것이 질문 36이다. 그리고 `execution_hash`는 문서별이 아니라
*정책별*이다 — U0의 held-out 실행과 학습-시드 실행이 모두 `fee0e20b…`를 보고한다 — §5.3의
사슬에 evaluation 슬롯이 없기 때문이다.

행별로: U0 `execution_hash fee0e20b…`, U1 `d85f108b…`, U2 `37e4909f…`, U3 `a2283994…`.

**각 번들이 무엇인가.**

| 행 | `learning_hash` | `observation_hash` | `lowering_hash` | `policy_hash`(§5.3) | 번들 |
|---|---|---|---|---|---|
| U0 | `5dac0a46…` | `899c16a9…` | `3d06811c…` | `f9fb5260…` | `m7-u/U0/u0.esb` |
| U1 | `fdb5178a…` | `899c16a9…` | `41d11a06…` | `e77fbc91…` | `m7-u/U1/train/checkpoints/20000.esb` |
| U2 | `5dac0a46…` | `cc437a24…` | `3d06811c…` | `b6ab2413…` | `m7-t6/run/checkpoints/20000.esb` |
| U3 | `fdb5178a…` | `cc437a24…` | `41d11a06…` | `c109d783…` | `m7-u/U3/train/checkpoints/20000.esb` |

넷 모두 `task_hash eb6efefa…`, `deployment_hash f2f9a510…` — 커밋된 문서다. 두 `learning_hash`와
두 `lowering_hash`는 `learning-lowering.md` 5.3이 기록한 그 쌍 그대로 움직이지 않았고, 증강된
observation은 둘 다 움직이지 않는다. 그래서 이 2×2가 2×2다.

**`expected_latency_ms`는 모든 행에서 15.0**이다(질문 32). 픽스처 `learning.toml` 자신이 선언한
숫자이고 50 Hz에서 제어 한 틱이며, `es policy pack`은 문서의 값을 그대로 옮길 뿐 지어내지
않는다 — 질문 32가 말하는 지어냄은 이 경로에 없는 `import-lerobot`의 것이다. 증거는 리포트
안에 있다: 모든 실행이 16 에피소드에서 정확히 **`violation.chunk_underrun` 16틱**, 에피소드당
한 틱을 센다. 한 틱 지연이 만드는 값이다(7.30절의 ACT는 200 ms에서 열 틱을 낸다).

**U0는 재학습하지 않았다.** 패킷이 요구한 대로, T4의 row-D 체크포인트
(`~/artifacts/plan-v/m7-t4/model-D.safetensors`, `lr_curve_hash c01d5185…`, `final_loss
0.004610`)를 커밋된 네 문서를 담은 번들 `~/artifacts/plan-v/v18/untrained-L80.esb`에
`es policy pack`으로 넣은 것이다. 이 트리에서 그 번들을 다시 로워링하면
`lowering_hash 3d06811c…d8a2d394` — T4가 학습한 모듈과 바이트 단위로 같은 것 — 이 찍히므로,
체크포인트는 자신이 판정받는 문서에 맞는다. `tests/fixtures/visible-learning/training-u0.toml`은
그 실행을 `es train` 레시피로 적은 것이고 `--dry-run`으로만 확인했으며 실행하지 않았다.

**U1은 발산했고, 그것이 그 행이다.** `es train --recipe training-u1.toml`(identity_hash
`ae3bd4b0…`, training_hash `e2a9c789…`)은 `first_nonfinite_step 4517`에 도달했다: 손실은
4,510 스텝까지 0.0419 → 0.0151로 떨어졌다가 4,516에서 0.303으로 튀었고 4,517부터 20,000까지
NaN이었다. 스케줄은 T4의 row D가 적용한 것과 정확히 같게 적용되었고(`lr_curve_hash c01d5185…`,
같은 다이제스트 — 즉 같은 20,000개의 `f64`), row D와 다른 것은 ImageNet 초기화뿐이다. 따라서
20,000 스텝 체크포인트는 NaN 가중치이고, 그것을 평가한 결과가 0.0000 행들이다: Safety Plane이
명령된 **28,800틱 중 28,784틱**을 `violation.non_finite`로 거부했고 팔은 모든 스위트에서, 두
문서 모두에서 움직이지 않았다. `envelope_violation_rate 1.0`이 그것이다. **다른 학습률로 다시
시도하지 않았다** — 패킷이 금지한다 — 그러니 이 행은 "이 조합은 발산한다"로 읽어야 하고,
이는 `training-recipe.md` 10절이 row D에 붙인 단서("여기서 측정된 것은 *이* 조합이 안정하다는
것")와 정확히 같다. U3는 같은 백본을 같은 학습률로, T6의 증강과 함께 돌린 것이고 발산하지
**않는다**(`final_loss 0.006383`, `first_nonfinite_step null`). 이 노트에서 가장 뜻밖의 줄이고
질문 35다.

**U2도 재학습하지 않았다**: T6의 수용 실행이 곧 U2의 설정이므로, 이 패킷은 똑같은 두 번째
실행에 GPU를 쓰는 대신 그 체크포인트(`training_hash 61e6c93a…`, `final_loss 0.006664`)를
평가했다. `training-u2.toml`이 그 실행의 레시피를 그대로 옮긴 것이다.

**U3의 학습**, 이 패킷이 만든 유일한 벽시계: `es train` 전체 — `--for-training` 베이크
(36,960 샘플, `[3, 104, 104]`, `Pad(4) → RandomCrop{96,96}` 경계), 로워링, 20,000 스텝, 팩 —
**4:11(251초)**, 유휴 카드에서(`nvidia-smi`: 컴퓨트 프로세스 없음, 1분 부하 3.20 → 1.03).
U1은 같은 조건에서 **2:57(177초)**였다(부하 2.66 → 1.11). 차이는 더 큰 베이크 텐서와 샘플별
증강이다. 둘 다 §19.3의 열두 슬롯을 모두 채운 `training.lock`을 썼고, 거기에는 실제
`base_model` 슬롯(`torchvision.models.ResNet18_Weights.IMAGENET1K_V1`, BSD-3-Clause)이 들어
있다. 기록해 둘 T1/T6 성질 하나: `es train --dry-run`은 `--for-training`도 `--augmentation`도
찍지 않는다. dry run은 번들을 열지 않고 증강 체인은 번들의 Observation IR의 성질이기 때문이다 —
계획은 레시피만의 함수로 남고(`training-recipe.md` 3절), 그 플래그들은 실행이 로그에 남긴
계획에 나타난다.

**`es eval compare U0/report.json U1/report.json`**, 도구 자신의 출력(`failure_mode_histogram`
행은 폭 때문에 뺐다. 네 비교 전문은 `~/artifacts/plan-v/m7-u/compare-U0-U1.txt`,
`compare-U0-U2.txt`, `compare-U1-U3.txt`, `compare-U2-U3.txt`):

```
SUITE                METRIC                                  A              B          DELTA  SIGNIFICANT
nominal              success_rate                     0.250000       0.000000      -0.250000  n/a (aggregate-only report)
nominal              envelope_violation_rate          0.440941       1.000000      +0.559059  n/a (aggregate-only report)
nominal              episode_length                1492.687500    1800.000000    +307.312500  n/a (aggregate-only report)
light_intensity      success_rate                     0.062500       0.000000      -0.062500  n/a (aggregate-only report)
light_intensity      envelope_violation_rate          0.454231       1.000000      +0.545769  n/a (aggregate-only report)
light_intensity      episode_length                1699.437500    1800.000000    +100.562500  n/a (aggregate-only report)
light_direction      success_rate                     0.062500       0.000000      -0.062500  n/a (aggregate-only report)
light_direction      envelope_violation_rate          0.486473       1.000000      +0.513527  n/a (aggregate-only report)
light_direction      episode_length                1711.812500    1800.000000     +88.187500  n/a (aggregate-only report)
observation_delay    success_rate                     0.062500       0.000000      -0.062500  n/a (aggregate-only report)
observation_delay    envelope_violation_rate          0.555011       1.000000      +0.444989  n/a (aggregate-only report)
observation_delay    episode_length                1721.812500    1800.000000     +78.187500  n/a (aggregate-only report)
torque_noise         success_rate                     0.000000       0.000000      +0.000000  n/a (aggregate-only report)
torque_noise         envelope_violation_rate          0.619340       1.000000      +0.380660  n/a (aggregate-only report)
torque_noise         episode_length                1800.000000    1800.000000      +0.000000  n/a (aggregate-only report)
backlash             success_rate                     0.062500       0.000000      -0.062500  n/a (aggregate-only report)
backlash             envelope_violation_rate          0.411423       1.000000      +0.588577  n/a (aggregate-only report)
backlash             episode_length                1699.437500    1800.000000    +100.562500  n/a (aggregate-only report)

A: passed=false   B: passed=false
```

`U1 → U3`는 같은 표에서 부호가 전부 뒤집힌 것이다 — `nominal +0.5625`, `light_intensity
+0.6250`, `backlash +0.4375`, `envelope_violation_rate −0.3332`, `episode_length −707.9`,
그리고 `A: passed=false   B: passed=true`. `U0 → U2`(증강 단독, 해시 경계를 넘는 비교)는
평평하거나 약간 음수다: `nominal −0.0625`, 네 스위트 −0.0625, `torque_noise +0.0625`.
`SIGNIFICANT` 열은 모든 비교의 모든 행에서 `n/a (aggregate-only report)`다. `report.json`이
셀 집계를 담고 에피소드별 결과를 담지 않기 때문이고, 도구는 자신이 받은 것으로는 유의성을
검정할 수 없다고 말하고 있는 것이다 — 정직하며, 위 어떤 차이도 유의하다고 부르지 않는 이유다.

**사이클 벽시계는 인용이지 재실행이 아니다**(`training-recipe.md` 12.8절): 데모 자신의 문서
위에서 `es loop cycle` 한 번이 2026-09-21에 **30:19** — collect 5:00, 전문가 게이트 2:21,
train 2:26, eval 20:24, showcase 0:08 — 로 §28.9의 "한 사이클 30분 이내"를 19초 차이로
놓쳤다. 그 실행은 T3과 T4가 들어온 뒤에 측정된 것이고, 그것이 패킷이 인용의 조건으로 건 것이다.
T5나 T6가 그 단계들을 바꾸지 않는다: U3의 학습(4:11 대 그 사이클의 2:26)은 `train` 행을 2분
미만 움직이고, `eval` 행은 정책이 얼마나 자주 성공하는지의 함수다 — 열여섯 중 아홉을 성공하는
정책은 에피소드를 일찍 끝낸다. 이 패킷 자신의 여섯 스위트 벽시계는 그것을 **분리하지 못하며**,
그것으로 오인되지 않도록만 적어 둔다: 네 실행은 한 박스를 나눠 쓰는 쌍으로 나갔고(U0 9:23 옆에
U1 9:24, U2 8:19 옆에 U3 8:19), 각 쌍의 숫자는 16코어에서 `--jobs 6` 두 실행 사이의 경합이지
어느 정책의 고유 비용이 아니다.

**쇼케이스.** U0의 held-out `nominal-00`(시드 101)은 U0의 네 성공 가운데 **하나가 아니다** —
1,800틱을 다 돌고 타임아웃한다 — 그러나 U0가 모든 nominal 에피소드에서 실패하지는 않으므로
패킷이 정한 V19b 궤적으로의 후퇴는 쓰지 않았고, 이 프레임들은 U0 자신의 것이다: 그 안의 정책은
큐브를 향해 뻗지만 태스크를 닫지 못한다. R1의 카메라
(`--eye 0.66,-0.46,0.52 --look-at 0.14,-0.04,0.04 --fov 36`), 1280×720, 각 1,800프레임,
ms/프레임은 명령 자신의 요약 줄에서:

| 렌더 | ms/프레임 | 1분 부하 전 → 후 | 옆에 둘 값 |
|---|---|---|---|
| `--look full` (R2) | **9.9** | 4.47 → 3.70 | `renderer.md` 9.6의 유휴 카드 10.1 / 9.9 / 9.9 |
| `--path pt --spp 64 --exposure 32` (R3) | **262.5** | 3.70 → 0.04 | `renderer.md` 11.7의 유휴 카드 254.0 / 254.6 |
| `--path pt --spp 4 --accumulate --max-history 32 --exposure 32` (R4) | **29.8** | 0.04 → 0.23 | `renderer.md` 11.7의 29.2 / 28.5 |

`nvidia-smi`는 전후 모두 다른 컴퓨트 프로세스를 보고하지 않았다. 첫 행은 시작할 때 부하가
패킷의 기준 4를 **넘었고**(병렬로 돌던 평가 둘의 꼬리) 18초 뒤 끝날 때는 밑이었다. 그럼에도
`renderer.md` 9.6의 유휴 카드 숫자를 재현하며, 경로 추적 두 행은 모두 부하 4 아래에서
측정되었다. 각각 같은 원본 프레임에서 두 번 인코딩했다: `python/es/encode_video.py --fps 50`
(mp4v, `cv2`/`numpy` 때문에 `~/venvs/es/bin/python`)와 서버의 정적
`~/.local/bin/ffmpeg 7.0.2`를 통한 H.264 사본 — 파이프 하나, 5배 작음, 질문 5가 정리한 관행.

| 파일 | 서버 `~/artifacts/plan-v/m7-u/showcase/` | 로컬 |
|---|---|---|
| RS `full` | `u0-nominal-00-rs-full{,-h264}.mp4` | `target/plan-u/m7-u/u0-nominal-00-rs-full-h264.mp4` |
| PT 64 spp | `u0-nominal-00-pt-64{,-h264}.mp4` | `target/plan-u/m7-u/u0-nominal-00-pt-64-h264.mp4` |
| PT 4 spp 누적 | `u0-nominal-00-pt-4-accum{,-h264}.mp4` | `target/plan-u/m7-u/u0-nominal-00-pt-4-accum-h264.mp4` |

원본 프레임 디렉터리 — 렌더 셋과 평가 `--frames` 트리 여덟, 50 GB — 는 실행 뒤 지웠고 1.5 GB가
남았다: 여덟 실행 전부의 `report.json`·`evaluation.lock`·`events.json`·`traj/`, U1과 U3의
`training.lock`과 체크포인트 번들, 레시피 넷, 증강된 evaluation 문서 둘, `es eval compare`
출력 넷, mp4 여섯, 그리고 `~/artifacts/plan-v/m7-u/U0/keep/` 아래 U0의 held-out `nominal-00`
`.estraj`와 관측 프레임.

**이 절이 정리하는 것과 정리하지 못하는 것.** 이것은 §28.10이 허용한 그 한 번의 재측정이고,
M7이 만든 네 레버 중 `success_rate`를 조금이라도 움직이는 조합은 **증강 + ImageNet 백본의
결합뿐**이라고 말한다: U0 → U2(증강 단독)는 평평하고, U0 → U1(백본 단독)은 발산하며,
U1 → U3는 nominal에서 +0.5625다. 이것은 7.29절을 뒤집지 **않는다**: 외부 ACT는 여전히 모든
스위트에서 낫고, 거기 도달하는 데 2×2가 필요하지도 않았다. 그리고 7.28절을 다시 열지도
않는다 — 그 문장은 7.30절이 다시 날짜를 매긴 채로 남는다. U3가 세우는 것은 *다른* Observation
IR 위에서, 7.28절에는 없던 지연 모델 아래에서 성립하는 *같은* 주장이다.

### 7.32 As built (M7/R5): row U4 — 같은 정책을 패스 트레이싱된 관측 위에서

패킷 `docs/packets/M7/R5-pt-observations.md`; 렌더러 쪽은 `renderer.ko.md` section 12이다. 이제 Task IR의 센서가 자기를 어떤 렌더러가 그리는지 말할 수 있으므로, U3의 구성을 한 가지만 바꿔서 — 96×96 프레임이 어떻게 만들어졌는지 — 다시 돌리고 `success_rate`를 그 옆에서 읽을 수 있다.

**문서들.** `task-pt.toml`은 커밋된 `task.toml`에 `rgb_overhead` 센서의 `render = { path = "pt", spp = 64, bounces = 3, exposure = 64, tonemap = "reinhard" }`를 더한 것이고 **그 외에는 아무것도 아니다**; 커밋된 `task_hash eb6efefa…`는 움직이지 않는다. 부재하거나 기본값인 `render`가 바이트 단위로 오늘의 정규형이기 때문이다(`cargo test -p es-ir committed_task_hash_is_unmoved_by_sensor_render`).

| 문서 | 해시 | 비고 |
|---|---|---|
| `task-pt.toml` | **`d546b808…`** | 커밋된 태스크, 센서 하나가 `Pt` |
| `observation-pt.toml` | `7caac85d…` | `observation.toml`의 그래프, `task_ref` 재지정 |
| `evaluation-pt.toml` | **`dafc8ce6…`** | `evaluation.toml`의 held-out 16 시드 × 6 스위트 |
| `observation-augmented-pt.toml` (서버) | `e5e72c5a…` | T6의 그래프, `task_ref` 재지정 |
| U4의 Evaluation IR (서버) | *아래 참조* | 이 행 자신의 `evaluation_hash` |

**왜 셋이 아니라 다섯인가.** `ObservationIr::task_ref`는 해시 입력이고 `XIR_001`이 그것을 태스크 자신의 해시와 같게 요구하므로 **하나의 관측 IR이 두 Task IR을 섬길 수 없다**(`renderer.ko.md` 12.2). §7.4의 "여러 관측 IR이 하나의 `task_hash`를 공유한다"에는 거울상이 없고, 따라서 `Pt` 태스크는 그래프의 모든 노드가 복사본임에도 하류 문서 집합 전체를 다시 발행한다. 렌더러 변경에 대해 해시 체인이 매기는 값이고, 이 패킷이 그것을 지불한 첫 패킷이다.

**노출은 패킷의 기본값이 아니라 결정이다.** 중립인 `exposure = 1.0`에서 패스 트레이싱된 관측의 평균 바이트는 래스터화된 관측의 188 대비 34다 — 데모 셀에는 발광 패널 하나뿐이고 `Pt` 경로에는 방향광이 없다. `64`는 평균 바이트(184)가 래스터라이저의 것과 맞는 스윕 단이므로, U4는 밝기 차이가 아니라 렌더 경로를 측정한다(`renderer.ko.md` 12.6).

**비용, 측정됨**(`renderer.ko.md` 12.4): 96×96, 64 spp, 3 bounces, 업로드와 리드백을 포함한 프레임 전체 — **RTX 4090에서 57.50 ms/frame**, **RTX 3060에서 127.53 ms/frame**, `Rs`의 2.90 / 2.77 대비. 패킷의 "오라클 서버에서 ~3 ms"는 19배 틀렸고, 그것이 5분짜리 수집과 36분짜리 수집의 차이다.

**SSIM, 측정됨**(`renderer.ko.md` 12.5): 커밋된 `nominal-00.estraj`의 같은 32틱에 대해 관측 해상도에서 `Rs` 대 `Pt` — **평균 0.3547, 최소 0.3035, 최대 0.4227**, 두 카드에서 소수 넷째 자리까지 동일. 정책이 실제로 읽는 크기에서 얻은 첫 §15.3 숫자이고, section 10.4의 결론을 바꾸지 않는다: 여기서 임계값을 세우지 않는다.

#### 그 행

| row | Learning IR | Observation IR | Task IR | 학습 | held-out `success_rate` |
|---|---|---|---|---|---|
| **U3** | `learning-pretrained.toml` | `observation-augmented.toml` | `task.toml` (`Rs`) | `es train` | **0.5625** |
| **U4** | `learning-pretrained.toml` | `observation-augmented-pt.toml` | `task-pt.toml` (`Pt`) | `es train` | **held-out 0.0 (16편 중 0), `passed = false`**; train-seeds 0.0625(16편 중 1) |

**U4가 나왔다(2026-09-21 12:28 서버 시각, held-out 절반).** 학습: 20,000스텝, loss 0.600 → 0.0045
(마지막 100스텝 평균; U3의 최종은 0.0067). Held-out 평가(`evaluation-augmented-pt`, 시드 16 × 스위트
6, PT 프레임 169,612개, `--jobs 6`으로 2시간 57분): **nominal `success_rate` 0.0 — 16편 중 0,
`passed = false`**(수용 기준 ≥ 0.5); `light_intensity` 0.0625, `light_direction` 0.0,
`observation_delay` 0.0, `torque_noise` 0.0, `backlash` 0.0625; nominal `envelope_violation_rate`
0.0928, nominal 에피소드 전부가 1,800틱 timeout. `evaluation_hash 343d84cd…`, `execution_hash
12dff382…`. 읽는 법은 둘이고, train-seeds 절반(실행 중, `U4/trainseeds`)이 가른다: (a) **포즈별
결의 암기** — loss는 U3보다 *낮은데* 못 본 시드에서는 아무것도 못 한다. 고정된 포즈별 텍스처를
지문으로 쓴 정책이 딱 이렇게 보인다(아래 문단; 후속 R13: `spp`를 올리거나 `Pt` 센서에 `Rs`
조명); (b) **PT 경로의 수집/평가 관측 불일치** — train-seeds 평가가 held-out과 달리 성공할 때만
배제된다. 또 `nominal`과 `light_direction`의 히스토그램이 바이트 단위로 같다: 조명 방향 섭동은
래스터라이저의 방향광을 움직이는데 `Pt` 센서에는 그것이 없다(장면 발광체로만 조명). 이 문서에서
그 스위트는 두 번째 nominal 실행이며, §15.3 결정 전에 메워야 할 빈틈이다.

**train-seeds 절반(12:28 → 12:59, nominal 시드 16): `success_rate` 0.0625 — 16편 중 1, `passed =
false`; U3는 자기 학습 시드에서 0.5625였다**(`evaluation_hash cb6c7522…`). 즉 PT 정책은 학습에 쓴
시드에서도 일반화하지 못한다: 자기 행동이 시범 경로를 벗어나는 순간 포즈가 새것이 되고 결 지문은
쓸모가 없어진다. 읽기 (a)다. (b)가 이 숫자로 배제되지는 않지만, 오라클 3(PT 수집 두 번이 비트 동일)과
두 단계가 같은 `sensor_cfg`를 쓴다는 점이 그 가능성을 낮춘다; 직접 확인 — 평가 시점에 데이터셋의 한
tick을 렌더해 바이트를 비교 — 이 R13의 첫 단계다. 재수집 없이 가장 싼 다음 실험: 같은 PT 데이터셋에
`training_only` 잡음 증강을 넣어 재학습해 결이 지문으로 쓰이지 못하게 하기; 그다음 `Pt` 센서의
프레임별 `svgf` 디노이즈나 `Rs` 조명(방향광의 NEE는 잡음이 없다); `spp` 증가는 마지막(두 배마다
세 시간짜리 평가가 두 배). 원시 프레임 트리와 서버 트리 `es-r5`는 이 숫자를 읽은 뒤 지웠고
`U4/{train,holdout,trainseeds}`는 남긴다.

**이 패킷이 닫힐 때 U4는 아직 돌고 있었고, 숫자는 의도적으로 추측하지 않았다.** 작업은 오라클 서버(RTX 4090, 유휴 카드: `nvidia-smi` 컴퓨트 프로세스 없음, 1분 부하 0.56)에서 2026-09-21 07:43에 `nohup ~/artifacts/plan-v/m7-r5/r5.sh &`로, 커밋 `16f1b6f`의 트리 `~/Projects/es-r5`에서 시작되었고 아래 세 단계를 연달아 실행한다. 산출물은 모두 **`~/artifacts/plan-v/m7-r5/`** 아래에 떨어지고, 각 단계가 `<stage>.start` / `.end`(유닉스 초), `<stage>.log`, `<stage>.done`을 쓰므로 어디까지 갔는지 읽을 수 있다:

```
# 1. 패스 트레이싱된 프레임으로 전문가 시연 200개 (57.5 ms/frame로 약 36분)
es loop collect --policy ~/artifacts/plan-v/m7-r5/untrained-pt.esb \
  --scene tests/fixtures/mjcf/so101_pick_place.xml --episodes 200 --seed 1 \
  --expert so101-pick-place --out ~/artifacts/plan-v/m7-r5/ds-train-pt \
  --frames ~/artifacts/plan-v/m7-r5/frames-train-pt

# 2. 그 프레임 위에서 U3의 설정 (약 4분; tests/fixtures/visible-learning/training-u4.toml)
es train --recipe tests/fixtures/visible-learning/training-u4.toml \
  --out ~/artifacts/plan-v/m7-r5/U4/train

# 3. held-out 16 시드 × 6 스위트, 그다음 학습 시드 16개
es eval run --config ~/artifacts/plan-v/m7-r5/evaluation-augmented-pt.toml \
  --policy ~/artifacts/plan-v/m7-r5/U4/train/checkpoints/20000.esb \
  --scene tests/fixtures/mjcf/so101_pick_place.xml \
  --out ~/artifacts/plan-v/m7-r5/U4/holdout --jobs 6 \
  --frames ~/artifacts/plan-v/m7-r5/U4/frames-holdout
es eval run --config ~/artifacts/plan-v/m7-r5/eval-trainseeds-augmented-pt.toml ... \
  --out ~/artifacts/plan-v/m7-r5/U4/trainseeds --jobs 6 ...
```

`success_rate`, `envelope_violation_rate`, `episode_length`, `passed`는 `~/artifacts/plan-v/m7-r5/U4/holdout/report.json`에 있고, 이 행의 `evaluation_hash`와 `execution_hash`도 같은 파일과 그 옆의 `evaluation.lock`에 있다. **평가 단계가 비싼 쪽이다**: 최대 1,800틱짜리 에피소드 96개 × 57.5 ms/frame은 분이 아니라 시간이고, 프로세스 하나가 이미 GPU를 93 %로 잡고 있으므로 `--jobs 6`이 `Rs` 경로에서만큼 벌어 주지 않는다. 원시 프레임 트리는 숫자를 읽은 뒤 지우라고 만든 것이다.

**숫자가 오기 전에 그림이 이미 말하는 것.** `target/plan-u/r5/contact-sheet.png`는 시연 하나 전체(시드 1, 511틱)를 두 번 수집한 것으로 위가 `Rs`, 아래가 `Pt`이며, 두 실행 모두 같은 데이터셋 `content` 다이제스트 `ef2e904d…`를 보고한다 — 같은 상태를 두 가지로 그린 것이다. 64 spp에서 패스 트레이싱된 관측은 **눈에 띄게 거칠고**, 그 거칠기는 결정론적이므로(고정 `seed`, 픽셀로 주소 지정되는 샘플 키) 정책이 에포크를 거치며 평균 내어 없앨 수 있는 잡음이 아니라 포즈마다 고정된 텍스처다. U4가 U3 아래로 떨어진다면 가장 먼저 볼 것이 그것이고, `spp`가 `SensorRender`의 필드인 이유가 정확히 그것이다 — 다음 실행은 코드 한 줄 건드리지 않고 그것을 바꿀 수 있다.

### 7.33 As built (M7/R1): 에피소드 경계 의미론 아래에서 다시 측정한 숫자

패킷 `docs/packets/M7/P-M7-R1.md`; 메커니즘은 `docs/design/evaluation-execution.ko.md` 2.7절과 `docs/design/safety-plane.ko.md`의 "카운터"에 있다. **7.32절까지 이 노트의 모든 평가 수치는 윈도 이월(carry-over) 아래에서 측정된 것이다**: Safety Plane의 §9.4 `ViolationRate` 링이 `begin_episode`를 넘어 살아남았고, 그래서 에피소드 `k`의 틱 0에서 워치독은 에피소드 `k−1`의 dirty 스텝으로 가득 찬 윈도를 읽어 트립했으며, clamp 경로 대신 fallback 경로를 탔다 — envelope 위반율이 약 0.999인 데모에서는 첫 에피소드를 뺀 모든 에피소드의 처음 200틱 동안. 같은 기간 `StepEvent::tick`은 셀에 누적되는 물리 클럭이었다. 오너가 2026-09-21에 둘 다 결정했고(§28.11) R1이 그것을 넣었으므로, 그 행들은 **pre-R1 semantics**이며 삭제되지 않고 표시된 채로 남는다(§28.9 규칙 2). 이 절이 그 재측정이다.

**재측정, 그리고 아무것도 움직이지 않았다.** 두 실행 모두 R1 빌드
(`~/Projects/es-r1-episodes`, `cargo build --release -p es --features render`,
`ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`)로 오라클 서버에서 2026-09-21에 돌렸고, 산출물은
`~/artifacts/plan-v/m7-r1-episodes/` 아래에 있다:

| 행 | 문서 | pre-R1 의미론 | R1 아래 | `passed` |
|---|---|---|---|---|
| **U3 held-out** (16 시드 × 6 스위트, `--jobs 6`) | `evaluation-augmented.toml`, `evaluation_hash e5705cb0…` | `success_rate` **0.5625**, `envelope_violation_rate` 0.6668, `episode_length` 1092.1 | **같은 숫자** — `report.json`이 `~/artifacts/plan-v/m7-u/U3/holdout/report.json`과 바이트 단위로 같고, `execution_hash a2283994…`와 여섯 스위트의 히스토그램까지 모두 같다 | 둘 다 true |
| **expert 게이트**, nominal (16 시드, `--jobs 6`, `--expert so101-pick-place`) | 커밋된 `evaluation.toml`의 nominal 스위트 | `success_rate` 1.0, `envelope_violation_rate` 0.0432, `episode_length` 507.25 (패킷 M7/T2의 `report-expert-gate.json`) | `success_rate` **1.0**, `envelope_violation_rate` **0.0432**, `episode_length` **507.2** | 둘 다 true |

**왜 움직이지 않았고, 언제라면 움직였을까.** 링은 워치독 하나의 입력이고, 그 워치독은
`envelope_violation_rate > max_frac`에서 발화한다. 데모는 `max_frac = 0.9`를 선언한다. U3는 셀
위반율 0.667로, 스크립트 expert는 0.043으로 돈다. 그래서 에피소드 `k`의 틱 0에서 옛 plane이
이월한 윈도와 새 plane의 빈 윈도는 **둘 다 트립 선 아래**다 — 같은 분기, 같은 액션, 같은 궤적.
T8이 갈라짐을 측정한 정책 `v14/trained-20000.esb`는 **0.9998**로 돌았다. 그 이월된 윈도는 틱
0에서 약 1.0을 읽어 트립했고, 빈 링이라면 clamp 경로를 탈 자리에서 fallback 경로를 탔다. 즉 이
의미론 변경은 *이미* 워치독이 참는 것보다 더 자주 envelope 밖에 있는 정책에서만 보이고, 커밋된
행 중에는 그런 것이 없다.

무조건 움직인 것은 `events.json`의 시계다. U3의 `backlash-01`은 옛 의미론에서 `tick: 1408`에,
`backlash-02`는 `tick: 8608`에 열렸고, 지금은 둘 다 `tick: 0`에 열린다. 어떤 지표도 `tick`을
읽지 않으며, 그래서 리포트가 바뀌지 않았다.

이것이 "pre-R1 semantics" 표시의 정직한 상태다. 그것은 숫자에 대한 정정이 아니라 *조건*에 대한
표시다. 7.32까지의 모든 행은 측정된 그대로 유효하고, 앞으로 슬라이딩 위반율이 자기 `max_frac`을
넘는 정책이 나오면 그것이 재측정이 필요한 행이다 — 그리고 그것은 이미 표에 있는
`envelope_violation_rate` 열에서 독자가 확인할 수 있는 성질이다.

**무엇을 다시 측정하지 않았고, 왜인가.** §28.10은 IR 그래프 정책의 재측정을 한 번 허용하고 그것이 U 웨이브였다. 이것은 다섯 번째 구성이 아니며 아무것도 재학습하지 않았다. U0, U1, U2, U4는 pre-R1 행을 유지한다. 데모의 수용 기준이 걸려 있는 행인 U3와 하네스 자신의 expert 게이트만 다시 돌렸고, 그것이 의미론 변경이 사람이 읽는 숫자에 무엇을 했는지 말해 주는 가장 작은 측정이다.

### 7.34 만든 대로 (M8/S4c): RL 트랙

이 노트의 데모는 시연에서 배우는 모방 학습이고, 플랜 S(`docs/design/rl-continuation.md`)는 반대쪽
입구다 — 다른 프레임워크에서 학습된 PPO 액터를 임포트해 우리 시뮬레이션 안에서 RL로 이어붙인다.
측정은 그 노트 섹션 7에 있고, 여기에는 머리기사 숫자만 적는다. 오라클 서버 2026-09-21, 산출물
`~/artifacts/plan-s/s4c/`, 전부를 채점한 Evaluation IR은 하나
(`tests/fixtures/rl/evaluation-reach.toml`, `evaluation_hash f15fe888…`):

| 행 | `success_rate`, `nominal`, 홀드아웃 시드 16개 |
|---|---|
| 소스 정책을 자기가 학습한 장면에서(brax 자신의 평가) | 1.00 |
| 같은 정책을 임포트해 여기서 채점 | **0.0000** |
| 임포트 + PPO 4,000 반복, 시드 3개 | **0.0000** |
| 같은 그래프를 처음부터, 같은 예산, 시드 3개 | 0.0833 (최고 시드 0.2500) |
| 커밋된 64 × 64 그래프를 처음부터, 같은 예산, 시드 3개 | **0.4167** (최고 시드 0.5625) |

**이어붙이기는 도움이 되지 않았고, 아무것도 하지 않았다**: 임포트된 네트워크의 `tanh` 스쿼시가 우리
관측 위에서 포화해 있고(50 제어 틱에서 측정한 스쿼시 이전 값 여섯 개 중 가장 작은 것이 23.7이며,
f32에서 `d tanh/dx`는 그 지점에서 정확히 0.0이다) PPO 4,000 반복 뒤에도 텐서 여섯 개는 비트 단위로
같고 시드 세 개가 하나의 `policy_hash`를 공유한다. 움직이는 것은 스쿼시 뒤에 있지 않은 단 하나의
파라미터 `log_std`다. 데모 자신의 결론(섹션 7.29)은 이것으로 흔들리지 않는다. 이것이 더하는 것은
reach 작업의 합격 기준이 아직 어느 경로로도 충족되지 않았다는 것, 그리고 이 예산에서는 가장 작은
그래프가 여전히 셋 중 가장 낫다는 것이다.

## 8. 안전 오버레이 (V3)

렌더된 프레임마다 V3는 `events.json`에 레코드 하나를 붙인다:
`{ frame, tick, source, events }`. `source`는 `SafetyCounters` 델타에서 나온 기존 4분류 (섹션 2.6),
`events`는 그 스텝의 `ViolationKind` 비트셋이다. 모자이크는 레코드가 `Clamped`나 `Fallback`인 셀에
붉은 테두리를 그리고, 진행 중 성공률은 `report.json`의 `MetricSpec::SuccessRate`에서 온다 — 둘 다 이미
계산된다 (`crates/es-eval/src/metrics.rs:32`, `:97-100`).

**데모는 비공허해야 한다.** W1d의 게이트가 쓰는 바로 그 규칙이다: V3의 오라클은 실행 결과에 `Clamped`
스텝이 최소 하나, `Success` 에피소드가 최소 하나 있지 않으면 실패한다. 한 번도 발동하지 않는 Safety
Plane 데모는 아무것도 증명하지 않는다. 클램프는 데모 스위트의 Deployment IR 속도·변화율 제한을 좁혀서
만든다 — **엔벨로프를 바꾸는 것이지 플레인을 끄는 것이 아니다** (INV-12).

## 9. 결정성, 그리고 해시 체인이 덮는 범위

`execution_hash = H(task, observation, learning, policy, dataset, deployment, compiler, runtime,
hardware_capability)` (§5.3). 플랜 V의 주장은 "영상이 재현 가능하다"보다 의도적으로 좁다:

| 산출물 | 주장 | 근거 |
|---|---|---|
| `report.json`, `evaluation.lock` | 같은 `evaluation_hash` -> 같은 수치 | 이미 M2의 속성이다; 모든 추출이 `EnvRng::new(seed, suite_id, episode_idx, stream)` (`crates/es-eval/src/perturb.rs:9-12`) |
| **CPU** 렌더 경로의 `frames/**/*.bin` | 같은 `execution_hash`에 대해 바이트 동일 | CPU 레퍼런스가 현재의 골든 경로 (`crates/es-render/tests/render.rs:14-16`, `:197-209`) |
| **GPU** 경로의 `frames/**/*.bin` | 기존 GPU 오라클을 통과하는 장치에서 CPU 레퍼런스와 비트 동일; 벤더 간에는 `Target / Status: unverified` | 저장소는 특정 드라이버의 산술을 골든에 굽기를 거부한다 (`render.rs:14-16`) |
| 물리 궤적 | 비트 단위가 아니라 `DeterminismTier::PhysicsMeaning` | `MuJoCoCpuBackend`가 의도적으로 티어 3을 선언한다 (`crates/es-physics-backend/src/mujoco.rs:28-38`) |
| 학습된 가중치 | 재현 **불가** | 버전 간 CPU PyTorch; 섹션 6.3의 규칙 3이 임계값인 이유 |
| `demo.mp4` | 해시 체인 **밖** | 인코더가 체인 밖 호스트 라이브러리다; 증거는 프레임이고 mp4는 그 한 가지 뷰다 |

따라서 정직한 헤드라인은 이것이다: **같은 `execution_hash` -> CPU 렌더 경로에서 바이트 동일한 프레임**.
인코더 하류와 학습된 체크포인트 상류는 그렇지 않으며, 각 패킷이 자기가 그 선의 어느 쪽인지 밝힌다.

## 10. 오라클과 CI 티어

| 오라클 | 확인 대상 | 필요 | 티어 | 없으면 |
|---|---|---|---|---|
| 상류 모델 교차검증 (V0) | 우리 파생본 == 고정된 `so101.xml` 기구학 | 네트워크 또는 캐시 | oracle job | 사유와 함께 `SKIP` |
| 데모 IR의 `es task compile` (V0) | 네 문서가 검증되고 해시된다 | — | PR | — |
| MuJoCo가 데모 장면을 로드 (V0) | `nq`, `nv`, `nu`, `nbody`가 선언대로 | `mujoco` | oracle job | `SKIP` |
| CPU 렌더 골든 (V0b) | 새 프레임이 CPU 레퍼런스와 일치 | — | PR | — |
| GPU == CPU 프레임 (V0b) | 이 장치에서 비트 동일 | Vulkan + `slangc` | GPU job | `SKIP` (GPU 부류) |
| 전문가 성공률 (V1) | 고정 비율 이상 | `mujoco` | oracle job | `SKIP` |
| LeRobot이 우리 데이터셋을 읽음 (V1) | 실제 `lerobot`이 열고 모양에 동의 | `lerobot[dataset]` | oracle job | `SKIP` — **현재 미설치** |
| 모듈 동일성, 컨트랙트, 손실, 왕복 (V2) | 섹션 6.3의 넷 | `torch` | oracle job | `SKIP` |
| Evaluation IR 리포트 + 비공허성 (V3) | 성공률, `Clamped` >= 1, `Success` >= 1 | `mujoco` + `torch` | oracle job | `SKIP` |
| 고정 프레임에서의 모자이크 (V4) | 바이트 동일한 모자이크 + 오버레이 | — | PR | — |
| mp4 인코드 (V4) | 프레임 수가 맞는 재생 가능한 파일 | `cv2` | oracle job | `SKIP` |

SKIP 사유에는 진짜 GPU 스킵이 아닌 한 `gpu`, `vulkan`, `render`, `slangc`, `device`라는 단어가 들어가면
안 된다: `xtask`가 그것들을 분류한다 (`docs/design/ros2-boundary.md` 섹션 8이 기록한 대로).

## 11. 패킷 순서

```
V0 (장면 + IR)  ∥  V0b (루프 안의 렌더링)  ∥  V4 (영상 조립)
                          └──────────┬──────────┘
                                     ▼
                                 V1 (전문가 + 데이터셋)
                                     ▼
                                 V2 (학습 + 패킹)
                                     ▼
                                 V3 (섭동 + 안전 + 실제 데모 실행)
```

사용자가 요청한 순서는 `V0 ∥ V4 -> V1 -> V2 -> V3`였다. 두 가지를 바꾸었고, 둘 다 섹션 2가 강제한 것이다:

- **`V0b`는 새로 생긴 것이다.** 저장소의 어떤 것도 렌더러를 env 루프에 연결하거나, 프레임을 디스크에
  쓰거나, 이미지 관측을 받아들이지 않는다 (섹션 2.2). V1의 프레임, V2의 이미지 입력, V3의 오버레이,
  V4의 실제 입력이 모두 여기에 의존한다. 플랜 V가 발견한 가장 큰 선행 작업이다.
- **V4는 요청대로 V0과 병렬로 남는다.** 의도적으로 *소비자*로 쓰였기 때문이다: 프레임 디렉터리와
  `events.json`을 받아 mp4를 만들고, 오라클은 체크인된 합성 프레임 픽스처다. V0b가 작성되거나 테스트될
  필요는 없다; *실제* 데모를 만들려면 V0b가 필요하고, 그것은 V3에서 일어난다.

각 패킷은 `src/*.rs` 기준 약 1,000줄 이하로 잡았고 (섹션 2.10), 어떤 패킷도 `es-ir`을 건드리지 않는다.

## 12. 사람을 위한 미해결 질문

1. **`lower_act`.** 플랜 V는 M4의 부채를 고치지 않고 우회한다 (섹션 2.5). 기본값: 그대로 둔다 — 데모는
   IR이 소유한 그래프를 학습하고, `lower_act`는 게이트 5의 체크포인트 오라클을 계속 맡는다. 대안은
   §8.3의 `TemporalEncoder { Transformer }`를 레이어 수·FFN 너비·잠재 차원까지 싣도록 넓히는 것인데,
   이는 스펙 변경이며 예산이 53줄 남은 `es-ir`에 들어간다.
2. **LeRobot 데이터셋 버전.** 저장소는 아무것도 고정하지 않는 api-note
   (`docs/api-notes/lerobot-dataset.md:3-8`)를 배경으로 v2.1을 쓰고
   (`crates/es-data/src/lerobot/meta.rs:148`), 서버의 `lerobot`은 0.6.1이다. 기본값: v2.1을 계속 쓰고
   V1의 오라클이 결정하게 한다 — 0.6.1이 거부하면 그 발견이 api-note에 들어가고 v3 라이터는 V1 안의
   변경이 아니라 별도 패킷이 된다.
3. **"N 스텝 연속".** 기본값 `N = 1` + 속도 경계 (섹션 5.4). 진짜 안정 카운터를 원한다면 새 IR-D 노드가
   아니라 IR-C다.
4. **V3가 `Unsupported`로 남기는 섭동들.** `Occluder`는 장면 편집이 필요하고, `CameraExtrinsic` /
   `CameraIntrinsic`은 INV-14가 요구하듯 `ImageSpec` 내부 파라미터가 카메라를 따라 움직여야 하며 (그게
   애초 거부 사유다), `ColorTemperature`는 렌더러에 없는 분광 광원 모델이 필요하다. 기본값: 넷 다
   `Unsupported`로 두고 V3는 `LightIntensity`와 `LightDirection`만 구현한다.
5. **비디오 코덱.** 측정 결과 서버의 `cv2`로는 `mp4v`만 인코딩된다 (섹션 2.9). 기본값: `mp4v`로 출시.
   **V3에서 덤으로 답이 나왔다:** `~/.local/bin/ffmpeg 7.0.2-static`이 이미 서버에 있으므로, 같은 원시
   모자이크 프레임의 H.264 사본은 파이프 하나면 되고 7배 작다(18초에 10.5 MB 대 1.5 MB). 아무것도
   설치하지 않았고 `encode_video.py`도 그대로다; H.264 파일은 같은 프레임의 두 번째 뷰이지 두 번째
   파이프라인이 아니다.
6. **사전학습 백본.** ~~`_backbone("resnet18", 512)`가 ImageNet 가중치를 로드하는지, 따라서
   학습에 네트워크가 필요한지는 미검증이다.~~ **답함 (V2, V2b).** 로드하지 않는다: `weights=`
   인자가 전달되지 않으므로 학습은 처음부터이고 네트워크가 필요 없다. V2는 IR의 `pretrained = true`가
   *무시*되고 있음을 발견했고, V2b는 `lower_to_torch`가 대신 거부하게 하며, 데모의 `learning.toml`은
   이제 `false`를 선언한다 (섹션 7.9).
7. **PNG.** 기본값: 기존 골든 포맷인 원시 `.bin` + 사이드카 (섹션 7.2). 사람이 이미지 뷰어로 프레임을
   열고 싶을 때만 PNG 인코더를 추가한다.
8. **4090 위의 CPU 전용 PyTorch** (섹션 2.9). 기본값: 받아들이고 학습 시간을 말하지 않는다. 사람이
   `~/venvs/es-lerobot`에 CUDA 빌드 torch를 설치하는 것이 플랜 V 일정에 가장 가치가 큰 선행 조건이다.
9. **`lerobot[dataset]`이 서버에 미설치**라 `lerobot.datasets`가 import되지 않는다. 사람이 설치하기
   전까지 V1의 가장 중요한 오라클은 영구 `SKIP`이다.
10. **게이트 7.** 플랜 V를 닫으면 SO-101 / 큐브-바구니 대체를 기록한 채 §28.7 게이트 7을 충족으로
    기록하는가, 아니면 RGB 2뷰 Franka를 위해 게이트 7을 계속 열어 두는가? 기본값: 충족으로 기록하되,
    대체 사실과 섹션 1의 `Target / Status: unverified` 행들을 M5 리뷰에 명시한다.
11. ~~**상태 포트가 추론에서는 정규화되고 학습에서는 되지 않는다** (섹션 7.8).~~ **답함 (V2b).**
    기본값을 택했다: `es dataset bake`가 `capture`가 쓰는 것과 같은 Observation IR 실행기로
    데이터셋을 통과시키고, `train_act.py`는 구운 세트를 읽으며, 데모는 V2의 손잡이 그대로 그 위에서
    재학습되었다. `observation_hash`는 움직이지 않았으므로 패킹된 번들과 `evaluation.toml`은 유효한 채로
    남았다 — 값싼 대안(상태 `Normalize` 제거)이었다면 움직였을 것이다. 섹션 7.9에 V3의 표 옆에 놀인
    새 표가 있다.
12. **시연의 명령과 Deployment IR의 엔벌로프가 서로 대조된 적이 없고, 팔을 멈추는 것은 그것이다**
    (섹션 7.9). 측정: `SafetyPlane`은 명령이 컨트롤 스텝당 `velocity_max * dt = 3.0 * 0.02 =
    0.0600` rad만 전진하도록 허용하는데, 기록된 `action` 열은 자신의 `qpos`를 중앙값 **0.2413**
    rad 앞선다 — 위치 서보의 정상상태 추종 오차이고, ACT가 충실히 배워서 내보내는 값이다. 그래서
    시연을 완벽하게 모방하는 정책조차 모든 에피소드의 첫 스텝부터 속도 클램프를 받고,
    `envelope_violation_rate`는 `1.0`이며, violation-rate 워치독이 약 10분의 1의 스텝에서
    `hold_position`으로 래치한다. 움직일 수 있는 끝이 셋이고 사람이 하나를 골라야 한다. 셋은
    등가가 아니기 때문이다: **(a)** `action`을 서보 목표가 아니라 *다음 명령 위치*로 기록한다 —
    V1 변경이고 데이터셋을 다시 수집한다; **(b)** Learning IR에 델타 행동 공간을 주어 정책이
    `qpos + delta`를 내보내고 플레인이 램프된 명령을 보게 한다 — IR 변경이고 `learning_hash`가
    움직인다; **(c)** 데모의 Deployment IR의 `velocity_max`를 시연이 실제로 들어맞는 값으로
    넓힌다 — INV-12가 허용하는 수(엔벌로프를 넓히되 플레인을 결코 끄지 않는다)지만 데모의
    엔벌로프가 더 이상 시연이 기록된 그 엔벌로프가 아니게 된다. 기본값이자 가장 값싸고 정직한
    것: **먼저 (c)로 정책이 과제를 아예 할 수 있기는 한지 알아내고**, 그다음 (a) 또는 (b)로 좁은
    엔벌로프를 되찾는다. V2b는 셋 중 무엇도 하지 않는다 — `es-safety`, `es-env`,
    `deployment.toml`이 모두 금지 목록에 있다 — 그리고 V2b의 기여는 위 숫자가 존재한다는 것이다.

    **부분적으로 답해졌다(V1c, 7.10절). 그리고 질문은 더 날카로워졌다.** ~~(a)~~는 이미 되어
    있었다: `action`은 언제나 플레인 통과 이후의 `SafeAction`이었고, V1c는 그것을 단언하고 옆에
    `action_commanded`를 붙였을 뿐이다. **(c)는 실제로 돌렸다.** V2b 자신의 20,000 스텝
    체크포인트를 스크래치 deployment 문서로 돌리니 `envelope_violation_rate`가
    `1.0000 -> 0.0551`로, `Fallback`이 `8,520 -> 0`으로, `ActionSource::Policy`가
    `0 -> 14,400 중 13,606`으로 바뀌었다 — 그리고 `success_rate`는 `0.0000` 그대로, 모든 에피소드가
    900 스텝 타임아웃이었다. **팔을 멈추고 있던 것은 플레인이 아니었다.** V1c는 또한 수집 경로와
    평가 경로가 엔벌로프를 서로 다른 기준(명령된 이동 대 추종 오차)에 대해 잰다는 것, 그것을
    수집기 쪽에서 닫으면 스크립트 전문가가 자기 오라클을 잃는다는 것(고정된 0.875에 대해 2/8),
    그리고 "에피소드 0만 풀린다"가 7.6절의 용의자 셋 중 무엇도 아니라 추론 위상이었다는
    것(지연이 선언되면 `frame == 0`은 발화하지 않는다; 수정 후 1/50 -> 50/50)을 찾아냈다. 사람에게
    남은 것은 더 좁다: **(b)** 델타 행동 공간, 또는 **(c)** 데모 `deployment.toml`의 영구적 확대 —
    그리고 (c)는 이제 성공률이 아닌 근거로 주장되어야 한다. 성공률은 움직이지 않기 때문이다.

    **답함 (V6, 7.12절).** 이 질문의 전제 자체가 한 가지 점에서 틀렸다: `0.2413` rad의 리드는 수집
    경로에서 `velocity_max`에 대고 재어진 적이 아예 없었고, 평가 경로에서는
    `acceleration_max · dt² = 0.008` rad — 서보의 *추종 오차* — 에 대고 재어졌다. 스펙 §9.3에는 그
    행이 없고 §9.5는 plane이 그것을 읽는 것을 금지한다. V6는 측정값이 모든 경로에서 에피소드당
    정확히 한 번, 시드로서만 엔벌로프에 들어오게 만든다. `velocity_max`는 plane 자신의 명령을
    제한하며, 그것은 스크립트 전문가가 늘 맞춰 온 것이고 시연이 늘 기록되어 온 기준이다.
    **`deployment.toml`의 숫자는 하나도 움직이지 않았고**, 선택지 (c)는 철회한다: 엔벌로프는
    애초에 시연이 맞지 않던 대상이 아니었다. 선택지 (b), 델타 행동 공간은 그 자체의 장점으로
    여전히 열려 있지만 — 정책의 출력을 모방이 아니라 구성으로 램프로 만든다 — V6가 고치지 않은
    무엇의 수정책은 더 이상 아니다. 남는 것은 7.10절 자신의 결론이다: 싸우지 않는 엔벌로프를
    주면 ACT는 스텝의 94.5 %를 스스로 몰고도 여전히 과제를 못 한다. **병목은 정책이다.**
13. **평가는 수집보다 열 배 자주 재계획하고, 에피소드를 하나 앞서 리셋한다**(7.12절, 발견 6).
    `es_eval::runner`는 매 *제어* tick마다 정책을 부르므로 `action.execute_chunk = 10`과
    `rate.inference = 5 Hz`가 둘 다 그 경로에서 죽어 있고, 5 Hz로 수집한 시연에 대해 네트워크를
    50 Hz로 질의한다 — 아무도 선택하지 않은 훈련/시험 불일치다. 별개로 `Env::new`가 리셋하고
    `run_episode`가 또 리셋하므로 `--seed S`는 수집할 때 무작위화 추첨 0을, 평가할 때 추첨 1을
    가리킨다. 기본값: 둘 다 그대로 둔다. 어느 쪽을 고쳐도 플랜 V가 보고한 모든 평가 숫자가
    움직이고 둘 다 엔벌로프 문제가 아니다. 두 경로를 프레임 단위로 비교하고 싶은 사람은 재계획
    주기를 먼저 요청해야 한다 — 그쪽이 정책의 성공률을 깎고 있을 개연성이 있는 쪽이다.

    **답함 (V6b, 7.13절), 그리고 질문은 과소평가되어 있었다.** 오케스트레이터가 2단계 실행 전에 둘
    다 닫았다. 재계획 쪽은 *틀리기까지* 했다: 수집도 매 제어 tick 재계획한다
    (`BatchDomains::single_env()`가 추론 주기를 1로 선언한다) — 수집에 있고 평가에 없던 것은
    `ChunkBuffer`이고, `action.execute_chunk`와 ACT temporal ensembling이 사는 곳이 거기다. 평가는
    매 원시 결과를 새 `seq`로 plane에 넘기고 0행만 실행했으므로, 열다섯 청크의 지수 평균에 대고
    훈련된 정책이 자신의 원시 마지막 예측으로 평가되었다. `es_eval::runner`는 이제 수집기 자신의
    함수인 `es_env::plane_chunk`를 통해 plane에 공급하며, 버퍼는 Deployment IR로 만든다. 이중
    리셋은 사라졌다. 7.8–7.11절의 모든 평가 숫자가 무효이고, 모든 수집·훈련 숫자는 유효하다.
14. **특권에는 필드가 없고, `sim_` 접두사가 그 일을 대신하고 있다**(7.14절).
    `es_ir::task::ObsChannel`은 `{ source, ty }`이고 그 밖의 것은 거기 속하지 않는다고 §7.4가
    명시하므로, V7a는 `provenance`나 `sim_only` 플래그를 만들어내는 대신 이름으로
    `sim_cube_pose`를 표시한다. 이를 검증하는 것은 아무것도 없다. 실제 SO-101을 겨냥한 Deployment
    IR이 어떤 로봇도 공급할 수 없는 채널을 읽는 Observation IR을 참조해도 막히지 않고, 막을 수 있는
    것은 포트 이름을 읽는 사람뿐이다. 기본값: **관례를 유지한다** — 사용자가 하나뿐이고 검증기가
    없는 필드는 `INV-17`이 거부하려고 존재하는 투기적 추상화이며, 비전 패킷은 이 채널을 배포하는
    것이 아니라 삭제할 것으로 예상된다. 체인이 이를 *강제*하기를 원하는 사람은 검증기 규칙
    (`Real` 실행 모드의 Deployment IR은, Task IR 채널이 모두 로봇이 공급하는 것이 아닌 Observation
    IR을 거부한다)을 요청해야 하고, 그것은 필드가 아니라 스펙 변경이자 `es-ir` 패킷이다.
15. **상태 정책이 결론지어도 되는 것**(7.14절). V7a의 중단 규칙은, 특권 정책이
    `success_rate ≥ 0.5`에 도달하지 못하면 다음 용의자가 모델 크기가 아니라 물리·접촉·전문가
    궤적이라고 말한다. 그 역이 열린 절반이다: 도달한다면 그것은 그래프·엔벌로프·전문가·학습 루프가
    건전하다는 증거이지, 96×96에 시연 50개로 비전이 동작하리라는 증거는 **아니다**. 기본값:
    통과한 상태 정책을 *더 높은* 해상도와 더 큰 세트로 2단계를 진행해도 좋다는 신호로 취급하고,
    상태 정책의 수치를 비전 정책이 견주어질 천장으로 기록한다.

    **답변됨 (V7a 2단계, 7.15절), 아무도 원치 않던 갈래로.** 상태 정책은 20,000 step에서 nominal
    `0.0000`(같은 셀의 샤딩된 판독에서 `0.0625`, 전체 스위트에서 4/96)을 기록했으므로, 열린
    절반은 생기지 않았고 중단 규칙의 절반이 생겼다. 이 실행이 중단 규칙에 더한 것은 같은
    방향을 가리키는 두 번째 수다: 큐브의 정확한 자세가 입력에 들어갔을 때 학습 손실이 0.5 %
    움직였고, 이는 L1 목적함수가 큐브가 어디 있는지 아는 정책과 모르는 정책을 거의 구분하지
    못한다는 뜻이며, 손실 곡선이 태스크 학습의 증거였던 적이 없다는 뜻이다. 7.15절의 발견 8이
    열어 볼 세 가지 — 시연이 잡는가 미는가, 그리퍼가 25 mm 상자를 물기는 하는가, temporal
    ensemble이 파지 구간을 살려 두는가 — 를 이름 붙였고, 순서는 사람이 고른다. 2단계(비전, 더
    높은 해상도, 더 많은 시연)는 다음 패킷이 **아니다**.
16. **정책 런타임이 torch일 때 `--jobs N`은 `--jobs 1`과 바이트 동일하지 않다**(7.15절, 발견 6).
    측정: nominal 셀은 단독으로 0/16, 같은 문서 본문을 같은 번들·시드로 `--jobs 6` 실행하면
    그 안에서 1/16이고, 단독 실행의 BLAS/torch 스레드를 2로 제한하면 샤딩된 수치가 마지막
    자리까지 재현된다. 샤드별 스레드 상한(`56fa49b`)은 옳고 필요한 것이다 — 제한이 없으면 워커
    여섯이 각자 모든 코어 크기의 풀을 만들고 실행이 *느려진다*(7.11절) — 하지만 스레드 수는
    torch의 리덕션 순서를 바꾸고, 그것은 액션을 바꾸고, 그것은 궤적을 바꾼다. V5 오라클은 이것을
    볼 수 없다: 결정적 가짜 런타임으로 `N = 4`에서 바이트 동일성을 고정한다. 기본값, 그리고
    가장 싼 정직한 것: **스케줄러가 아니라 문서를 고친다** — `es eval run --help`와 7.11절은
    *스레드 수에 무관한 런타임이 주어졌을 때* 바이트 동일하다고 말해야 하고, 보고되는 수치는
    자기 `--jobs`를 달고 다녀야 한다. 더 강한 성질을 원하는 사람에게는 동등하지 않은 두 선택지가
    있다: 모든 워커를 `--jobs 1` 실행과 같은 스레드 수로 고정하거나(`--jobs 6`을 `--jobs 1`보다
    느리게 만든 과다 구독을 되돌려 받는다), 스레드 수를 spec 5.3 체인의 `hardware_capability`에
    넣어 그것이 다른 두 실행을 같은 실행이라 주장하지 않게 하거나.
17. **외부 ACT의 실패가 결론 내려도 되는 것**(7.16절, V8-1). LeRobot 자신의 ACT — 사전학습 ResNet18,
    CVAE, DETR 디코더 — 를 같은 50편의 시연으로 `lerobot-train`이 100,000 step 학습시키고
    `ACTPolicy`와의 비트 단위 동등성을 검증해 가져왔더니 20k/50k/100k에서 0/16, 0/16, 1/16이다.
    이것은 "우리 Learning IR 노드 집합이 데모가 안 되는 이유다"를 배제한다. "96×96은 너무 작다"나
    "50 에피소드는 너무 적다"는 배제하지 **않는다**: V8은 그것들을 일부러 고정했다. 기본값: 데모의
    실패를 **데이터와 물리**의 발견으로 취급하고, 다음 패킷을 씬의 진단으로 삼는다 — 전문가 자신이
    기록한 액션을 `es eval run`으로 재생해 하네스가 그 성공률을 재현하는지 확인하고, 그다음 해상도와
    시연 수를 하나씩 바꾼다. 대안 — 곧바로 더 높은 해상도에 더 많은 시연 — 은 더 큰 실행이고,
    실패했을 때 같은 모호함을 남긴다.
18. **데모는 200 Hz로 도는데 모든 문서는 50이라고 말한다**(7.18절). `Env::new`는 장면을
    `LoadConfig { rate: None }`으로 적재하고 `Env::step`은
    `schedule.domains().inference.period`만큼 시뮬레이션 틱을 전진시키는데, `es loop collect`와
    `es eval run`이 만드는 유일한 것인 `BatchDomains::single_env()`가 그것을 1로 둔다. 따라서
    제어 한 스텝이 MJCF 타임스텝 하나, 5 ms인 반면 `deployment.toml`은 `rate.control = 50`과
    `rate.inference = 5`를 선언한다. 추론이 아니라 측정이다: mujoco 프로브의 재생은 서브스텝
    1에서 `es` 재생을 0.0002 mm까지 따라가고 4에서는 202.98 mm 벌어진다. 움직일 수 있는 끝은
    둘이고 둘은 동등하지 않다. **(a)** `LoadConfig::rate`를 Deployment IR의 `rate.control`에서
    정한다 — *물리* 타임스텝이 바뀌므로 시연이 기록된 접촉 거동도 바뀐다. **(b)**
    `BatchDomains::inference.period`를 `rate.control / <장면의 물리 속도>`에서 유도하고
    타임스텝이 제어 주기를 나누지 못하는 장면을 거부한다 — 물리는 그대로 두고 제어 한 스텝을
    그 네 서브스텝으로 만든다. 기본값이자 장면을 건드리지 않는 쪽은 **(b)**. 어느 쪽이든 플랜
    V가 낸 모든 학습·평가 수치는 의도된 제어 속도의 네 배에서 얻은 것이므로 재수집과 재학습은
    같은 패킷에 속하며, 결과를 견주는 상대는 V8의 100,000 스텝 수치다.
    **답 (V11, 7.19절): (b), 구현됨.** `BatchDomains::single_env_at`이 추론 *과* 관측 주기를
    장면의 물리 속도와 Deployment IR의 `rate.control`에서 유도하고, 제어 주기를 나누지 못하는
    타임스텝을 이름으로 거부한다. `LoadConfig::rate`는 `None`으로 남는다. 답과 함께 질문이
    예상할 수 없던 두 번째 발견이 왔다. 진짜 속도에서 스크립트 전문가는 파지 자세를 11 mm
    지나쳐 큐브를 쳐낸다. 데모가 2.94 N*m 액추에이터로 제동할 수 없는 가속도를 선언하고 있고,
    200 Hz에서는 팔이 속도 포화 상태라 추종할 수 없는 것을 지나칠 수도 없었기 때문이다.
    전문가는 이제 엔벌로프의 9/10이 아니라 1/2에 맞춰 페이싱한다(`PACE`, 스윕해 측정). 엔벌로프
    한계도 게이트도 움직이지 않았다.
19. **IR이 정규화를 명시해야 하는가?** (7.21절, V13.) `VisionEncoder`는 `backbone`, `out_dim`,
    `pretrained`, `frozen`을 담고 정규화에 대해서는 아무 말도 하지 않는다. 그래서 "처음부터
    학습하는 ResNet18은 `GroupNorm(32, c)`로 내린다"는 로워링 규칙이고
    (`learning-lowering.ko.md` 5.1절) `learning_hash`가 아니라 `lowering_hash`에 산다.
    *`pretrained` 값마다 방어 가능한 답이 하나씩 있는 한* 그 자리가 맞고, 오늘은 그렇다: 단일
    샘플 로워링에서 배치 통계는 쓸 수 없고, LeRobot 체크포인트의 고정 BatchNorm은 그 ImageNet
    가중치에 맞는 유일한 것이다. 기본값: **로워링에 그대로 둔다.** 대안 — `VisionEncoder`에
    `norm: { Group { groups }, Frozen, Batch }` 필드 — 은 이 선택을 아키텍처의 일부로 해시
    가능하게 만들고 컴파일러를 바꾸지 않고도 두 실험이 이 항목으로 달라질 수 있게 한다. 두 번째
    backbone 계열이나 BatchNorm에서 이어 학습하는 체크포인트가 생기면 그쪽이 강제된다. 다만 이는
    `es-ir` 변경이고 `es-ir`은 **6,000줄 목표의 5,947줄**(§1.5)에 와 있으므로, 정직한 순서는
    `es-ir`을 먼저 쪼개고 그다음에 결정하는 것이다. 이 로워링에 `pretrained: true` 경로가
    들어오거나 두 번째 정규화가 실제로 필요해질 때 다시 본다.
    덧붙여 기록한다. 플랜 V의 모든 문서가 "16행 청크"라고 말하지만 로워링된 모듈은 그렇지 않다:
    헤드는 `horizon = 16`행을 내고 chunker가 `v4_chunk[:10]`으로 자른다
    (`execute_chunk = 10`, `tests/fixtures/visible-learning/learning.toml:272`). 그래서
    `es_eval::infer_chunk`는 `rows = min(len / NJ, H) = 10`을 취하고, 7.21절의 모든 개루프 L1은
    10행 기준이다. 이는 픽스처의 의도이지(`replanning_hz = 5 = 50 Hz / 10`) 버그가 아니다 —
    다만 산문에서의 "16"은 헤드의 horizon을 뜻하지 실행되는 청크를 뜻하지 않는다.

    *이 질문이 병합되기 전에 V13(7.21절)이 답했다: 처음부터 학습하는 백본은 이제 `GroupNorm`으로 lowering되어 학습된 적합도가 곧 배포되는 적합도다. 배치 lowering의 벡터화는 정확성이 아니라 학습 속도 항목으로 사다리에 남는다.*
20. **처음부터 학습하는 백본의 `BatchNorm`: 학습된 적합이 배포된 적합이 아니다**(7.20절).
    `python/es/train_act.py --batch N`은 **단일 샘플** 순전파 N번을 한 옵티마이저 스텝으로
    누적하므로, 낮춰진 ResNet18의 모든 `BatchNorm2d`가 `N = 1` 배치 통계로 학습된다. 반면
    `crates/es-policy/python/torch_ref.py`는 `model.eval()`을 불러 러닝 통계로 돈다. V12 자신의
    체크포인트와 자신의 학습 에피소드에서 측정: 청크 L1이 `train()`에서 0.0129, `eval()`에서
    0.0337–0.0441, "현재 자세 유지"가 0.0561–0.0571. 움직일 수 있는 끝은 둘이다. **(a)** 낮추기
    에서 배치를 벡터화해 학습 한 스텝이 크기 N의 진짜 배치 한 번의 순전파가 되게 한다 — 정직한
    수정이며 `lower_to_torch`가 내는 모듈과 `train_act.py`를 함께 고쳐야 한다. **(b)** IR의
    `VisionEncoder{ResNet18}`이 frozen 또는 group 정규화로 낮춰지게 한다. LeRobot 자신의 ACT가
    하는 일이고(`FrozenBatchNorm2d`) 두 모드를 구조적으로 같게 만들지만, 낮추기가 torchvision
    기본과 더 이상 일치하지 않는 대가가 있다. 기본값: **(a)**. 배치 축은 스펙 5.2가 이미 학습
    도메인에 있다고 말하는 것이고, (b)는 정규화를 가진 다른 모든 백본에 대해서도 틀린 낮추기를
    가리기 때문이다. 어느 쪽이든 변수는 하나이고 다음 패킷의 내용 전부다.
21. **성공 술어가 x 전용이고, 예산은 이제 1,800이다**(7.22절, V14). 한 항목에 결정 둘, 둘 다
    사람 몫이다. **(a) Task IR이 3차원 통 검사를 표현하려면 무엇이 필요한가?**
    `crates/es/tests/cli.rs`의 `cube_in_the_bin`은
    `x in [0.09, 0.19) and y in [-0.15, -0.05) and z < 0.09`인데, Task IR 자신의 술어는
    `큐브 x in (0.09, 0.19) and 큐브 vx in (-0.05, 0.05) and 그리퍼 > 0.6`이다.
    `GetJointState{joints: ["cube_free"]}`가 스칼라 하나를 내놓고 원뿔 리프는 스칼라 하나이기
    때문에(5.4절) 자유 관절의 y와 z에는 애초에 닿을 수 없다. V14가 그 대가를 측정했다: 홀드아웃
    16개 중 9개가 큐브를 실제 통 부피 안에 넣었는데 하네스는 그 전부를 `timeout`으로 채점했고,
    반대로 하네스가 `success`로 채점한 유일한 에피소드(40,000 체크포인트, 학습 시드 3, 182틱)는
    큐브가 통의 x 구간 안이되 공중 125 mm에 있었다. 이 술어는 데모가 주장하는 과제에 대해 건전하지도
    완전하지도 않다. **(a) 중 결정이 아니라 측정이었던 부분은 V15가 정리했다**(7.23절): V14의
    운반 에피소드 다섯 개를 틱 단위로 보면 1,800틱 중 1,668 ~ 1,697틱이 큐브를 통의 x 구간 안에
    두고 그중 1,652 ~ 1,679틱은 `|vx| < 0.05`도 만족하는데, **다섯 항을 모두 만족하는 틱은 하나도
    없다**. 정착 항은 접촉 지터에 지지 않고 x 구간은 3차원 검사가 쓰는 바로 그 구간이다. 붙잡고
    있는 큐브를 막는 유일한 항은 그리퍼다. 사람 몫으로 남은 것은 y와 z뿐이고, 길은 셋이다. **(i)** `GetJointState`에 성분 인덱스를 주거나(혹은
    `GetBodyPose`처럼 벡터 출력 + `Slice`) 해서 성분 셋에 대한 `Compare` 셋을 `And`로 묶을 수 있게
    한다 — 가장 작은 변경이며 §6 스키마 변경이고 `task_hash`가 움직인다. **(ii)** 이미 3-벡터를
    반환하는 `GetBodyPose`(노드 34가 특권 관측 채널에 쓰고 있다)로 검사를 작성하고, 벡터와 박스를
    받는 비교 노드를 둔다. **(iii)** Task IR을 그대로 두고 데모의 `success_rate`가 대리 지표임을
    받아들이며 3차원 운반 횟수를 영원히 나란히 보고한다. 기본값: **(i)**. "원뿔 리프는 스칼라
    하나"가 실제 한계이고 인덱싱이 그것을 푸는 최소이기 때문이다. (iii)은 V13과 V14가 둘 다 산문으로
    해야 했던 일이고 M6까지 확장되지 않는다. **(b) `max_episode_steps`를 1,800으로 둘 것인가?**
    V14는 정책이 언젠가 놓는지 답하려고 이 값을 올렸고, 답은 어떤 예산에서도 아니오였다. 큰 숫자의
    대가는 평가 스위트마다 시간이 두 배가 되고(16 에피소드당 265 s → 470 ~ 620 s) `episode_length`가
    7.19 ~ 7.21절과 더는 비교되지 않는다는 것이다. 유지하는 쪽이 정직하다 — 시연은 제어 스텝 약
    180개가 걸리므로 1,800은 빡빡한 예산이 아니라 넉넉한 예산이다 — 그러나 측정된 어떤 것도 이 값을
    필요로 하지 않는다. 기본값: **1,800 유지**, 그리고 어떤 정책이 놓는 데 성공하면 그 예산에서
    기준표를 다시 돌린다. 한 번도 구속 조건이 아니었던 예산 쪽이 두 오류 중 안전한 쪽이다.
22. **놓기는 관측 가능하지 않고, 기억 없는 정책이 그것을 보게 만들 수는 없다**
    (7.24절, V16). 측정: 개시−2부터 개시+5까지 정책 입력 전체가 1.2 × 10⁻³ 정규화 이내로
    (타일은 4.3 × 10⁻⁴) 얼어 있는 동안 타깃은 −0.046에서 +0.129 rad까지 오르고, 입력이 개시
    프레임의 값에서 0.002 이내인 시연당 약 6개 프레임은 −0.046 ~ +0.081의 그리퍼 타깃을,
    중앙값 **+0.0057 rad**로 담는다 — 그것이 L1 적합이 돌려주는 값이고, 1.4 mrad 차이로
    닫힌 루프가 명령하는 값이다. 그 창이 약 6 제어 틱인 이유는 Deployment IR의
    `acceleration_max = 20 rad/s²`가 50 Hz에서 틱당 스텝 변화를 0.008 rad로 묶기 때문이므로,
    이 엔벨로프 아래에서는 **시연을 아무리 늘려도 놓기가 관측의 함수가 되지 않는다.** 빠져나갈
    길이 셋 있고, 어느 것도 수정 패킷의 단일 변수는 아니다. **(i) 정책에 시계를 준다**:
    Observation IR의 `TemporalWindow { n_steps > 1 }`와 그에 맞춘 Learning IR의
    `observation_window`, `TemporalEncoder { n_frames }`. `observation_hash`와 `learning_hash`가
    움직이고 재베이크와 재학습이 필요하다 — 그리고 단독으로는 오히려 *악화*시킨다. 시연에는
    놓기 전의 긴 정지 구간이 없으므로(측정된 dwell 중앙값 0프레임) 닫힌 루프 홀드의 이력 역시
    본 적 없는 것이 되기 때문이다. 데이터 변경과 짝지어야만 작동한다. **(ii) 놓기에 신호를
    준다**: 관측이 담고 있는 무언가 — 예컨대 큐브가 빈 테두리를 넘는 것 — 에 맞춰 손을 펴도록
    전문가를 바꿔, 얼어붙은 창이 여섯 개가 아니라 하나의 타깃을 갖게 한다. **(iii) 선언된
    재계획 주기를 지킨다**(질문 23). 그러면 이미 놓기를 예측하고 있는 청크가 실제로 실행된다.
    기본값: **(iii) 먼저.** 운반 홀드 8개 중 8개에서 멈춤 각도를 넘는 것이 측정되었고 재학습
    비용이 없다. 그다음 (ii).

    **답함 (V17, 7.25절): 기본값을 택했고, 부분적으로 통했다.** 선언된 5 Hz 재계획을 두 경로
    모두에서 지키자 — 재학습 없이 V15 자신의 가중치로 — **48에피소드에서 놓기 4회, 하니스
    성공 3회**가 나왔다. V13 – V16의 128에피소드에서는 놓기가 0회였다. V16이 잃었던 운반도
    되찾았다(두 스위트 모두 0 → 16개 중 5개). 홀드아웃 `success_rate`는 0.0625로 여전히 0.5
    아래라 정지 규칙이 발동하지만, 장애물이 더는 고정점이 아니다. 홀드아웃의 모든 운반
    에피소드에서 큐브는 1,800틱 중 1,557 – 1,618틱 동안 빈의 x 구간 안에서 안정되어 있으므로
    채점된 성공을 막는 항은 오직 그리퍼이고, 가장 아깝게 놓친 경우는 술어의 0.85에 대해
    0.817 rad — **약 33 밀리라디안**(채점된 에피소드의 마지막 스텝 전 행은 0.82 – 0.84로
    읽히는데 술어는 스텝 후 상태로 판정되기 때문이다) — 까지 열렸다가 다시 닫힌다. **이제 (ii)가 다음 변수다**:
    관측이 담고 있는 무언가에 놓기를 맞춰 집게가 끝까지 열리고 열린 채 머물게 한다. (i),
    Observation IR에 시계를 주는 것은 비용이 그대로이고 여전히 데이터 변경을 옆에 필요로 한다.
23. **`rate.inference = 5 Hz`와 `replanning_hz = 5.0`은 선언되어 있고 지켜지지 않는다**
    (7.24절, V16). `es_eval::runner`는 Deployment IR의 `rate.inference`가 무엇이든 **제어**
    틱마다 `infer_chunk`를 호출하므로 청크가 매 틱 푸시되고 그 1..9행은 결코 실행되지 않는다.
    시간 앙상블은 가장 새로운 여덟 청크의 0..7행을 평균하고, `HardSwitch`라면 최신 청크의 0행을
    낼 것이다. 홀드에서 측정된 비용: 명령된 그리퍼는 +0.007 rad인데 같은 청크의 7, 8, 9행은
    0.105, 0.150, 0.203이고 0.093 rad 멈춤 각도를 넘는다. 이것이 명백한 결함은 아니다 —
    제어 틱마다 추론하고 시간 앙상블을 쓰는 것은 정당한 ACT 배치이고, 모든 시연을 만들어낸 것도
    그 방식이다 — 그러나 두 문서는 5 Hz라 말하고 런타임은 50 Hz로 돈다. 기본값: **그대로 두고
    기록한다.** 수집 경로도 같은 주기이고 그것을 옮기면 7.11 – 7.24절의 모든 숫자가 다시
    매겨진다. 사람이 결정해야 할 것은 정책이 동기적일 때 `rate.inference`가 무엇을 뜻하는가이며,
    어느 쪽으로 가든 두 편 중 하나는 바뀌어야 한다.

    **답함 (V17, 7.25절), 그리고 양쪽 모두였다.** 런타임은 이제 `es_env::replan_interval`이라는
    하나의 규칙으로 *두* 경로 모두에서 선언된 레이트를 지키며, 제어 레이트를 나누어떨어지지
    않는 레이트는 이름을 붙여 거부한다. 문서도 움직여야 했다. 5 Hz 재계획을 선언한 배치는 청크
    도착 간격이 최대 180 ms라 40 ms `inference_deadline`을 만족시킬 수 없으므로, 데모의
    `inference_budget`과 그 워치독은 같은 파일이 선언한 레이트에 맞춰 240 ms로 기술된다.
    `deadlines.observation_age`는 `DEP_021`이 더 작은 값을 거부하기 때문에만 따라간다. 모든
    안전 한계, `stale_observation` 한계, 0.5 수용 임계값은 그대로다. 가는 길에 결함 둘을
    찾았다 — 플레인의 워치독 시계가 *시뮬레이션* 틱인데 그 주기는 *제어* 주기였다는 것(V17에서
    고쳤고 `es-safety`는 손대지 않았다), 그리고 평가기가 추론 지연을 모델링하지 않는다는
    것(질문 24). 7.19 – 7.24절의 모든 평가 숫자는 청크의 뒷행이 죽은 채 측정된 것이라 이로써
    다시 매겨진다. 수집과 학습 숫자는 그대로 이어진다. 수집도 매 제어 틱 재계획했고 시연은
    청크가 아니라 전문가의 명령 흐름이기 때문이다.
24. **`es_eval::runner`에는 추론 지연 모델이 없고 `es loop collect`에는 있다**
    (7.25절, V17). `DomainRunner`는 모든 청크를
    `latency_ticks(RuntimeHints::expected_latency_ms, rate.control)`만큼 — 데모의 15 ms는
    50 Hz에서 제어 1틱 — 늦추므로 수집의 첫 청크는 틱 1에 플레인에 닿고 틱 0은 기록된
    언더런이지만, 평가 러너는 틱 0에 0행을 실행한다. 한 시드에서 측정: 두 경로의 틱별
    `qpos ‖ qvel` 궤적은 틱 0에서 동일하고 틱 1에서 갈라지며, 관절 0에서 `1.44 × 10⁻⁷` 대
    `−4.24 × 10⁻⁴`다. 케이던스 문제가 아니고(V17의 오라클이 두 경로를 120 제어 틱당 정책 호출
    12회로 고정한다) 안전 문제도 아니다(두 경로 모두 플레인이 유일한 액추에이터 경로다). 다만
    "두 경로가 청크를 동일하게 실행한다"가 스케줄에 대해서는 참이고 궤적에 대해서는 아니라는
    뜻이며, 추론 지연이 0인 상태로 평가된 정책은 존재하지 않는 로봇에서 평가된 것이다.
    기본값: **`es_eval::runner`에 수집이 쓰는 `AsyncInference`를 똑같이 준다.** 그 자체로 한
    패킷이고 평가 숫자를 한 번 더 다시 매긴다. 대안은 시뮬레이션 타깃에 대해
    `expected_latency_ms = 0`을 선언하고 그렇다고 말하는 것인데, 이는 문서 변경이고 실제
    로봇이 지킬 수 없는 주장이다.
    **답함 (T7, 7.30절): 기본값을 택했고, 그것이 숫자를 다시 매겼다.** 셀 루프는 이제 같은
    `latency_ticks` 호출로 만든 같은 `es_env::AsyncInference`를 통해 정책을 호출하며, 이 질문이
    요구한 궤적 오라클이 통과한다 — 120틱 중 120틱이 비트 단위로 같고, 이전에는 위에 인용된 바로
    그 두 값으로 틱 1에서 실패했다. `expected_latency_ms = 0`은 계속 합법이고 여전히 0틱을
    뜻한다. 대가: 외부 ACT는 두 문서 모두에서 `success_rate` 0.9375와 `passed = true`를
    유지하지만 `envelope_violation_rate`가 0.21 → 0.55로, 에피소드 길이가 1.7배로 늘고,
    IR-그래프 정책은 0.625 → 0.0625로 떨어져 더 이상 통과하지 못한다. 표와 대조군은 7.30절에
    있다.
25. **선언된 케이던스에서 `envelope_violation_rate`가 0.998이고 워치독이 걸린다**
    (7.25절, V17). 청크의 0..9행을 순서대로 실행하면 제어 틱당 0.02 – 0.03 rad의 이동을
    요구하는데 Deployment IR의 `acceleration_max = 20 rad/s²`는 틱당 스텝 변화를 0.008 rad로
    묶으므로, `SafetyPlane`이 학습 26,033틱 중 23,443틱에서 레이트 제한을 걸고
    `EnvelopeViolationRate` 워치독(`max_frac = 0.9`, 창 200)이 매 실행의 약 1/10 동안 폴백을
    걸어 잠근다. 8겹 앙상블에서는 같은 청크들이 더 평탄한 흐름으로 평균되어 0.71 / 0.34였다.
    새로 명령되는 것은 없다 — 시연 자신의 명령 흐름이 평균되지 않은 채 플레인에 닿는 것이고
    `deployment.toml`이 V1부터 예고한 바다 — 그러나 틱의 1/10을 `hold_position`으로 보내는
    실행은 정책을 측정하고 있지 않다. 움직일 수 있는 끝이 셋이고 서로 같지 않다. **(a)**
    전문가는 엔벨로프의 `PACE = 0.5`로 페이스를 잡고 정책은 그것이 낸 명령을 모방한다.
    `acceleration_max`를 서보가 실제로 낼 수 있는 값으로 넓히는 것이 INV-12에 부합하는
    수단이며, 패킷이 아니라 Deployment IR의 결정이다. **(b)** 델타 액션 공간(질문 12의 (b),
    여전히 열려 있음)은 정책 출력을 구성상 램프로 만든다. **(c)** 받아들이고
    `envelope_violation_rate`를 `action_source` 히스토그램 옆에 계속 보고한다 — V1 – V16이
    0.71에서 해온 일이다. 기본값: **홀드아웃 시드에서 놓기가 채점될 때까지 (c)**, 그다음 (a).
    이 수치는 청크의 뒷행이 실행되기 시작한 뒤에야 구속력을 가졌고, 구속하는 대상은 모든 시연이
    이미 담고 있는 바로 그 명령 흐름이다.

    **홀드아웃 시드에서 놓기가 이제 채점되고(V17), (a)는 측정되었을 뿐 채택되지는 않았다
    (V18, 7.26절).** 커밋된 파일을 건드리지 않는 `deployment.toml` 스크래치 사본 안에서
    `acceleration_max`만 20(기준선), 40, 80으로 절제 실험을 했고 — V15의 체크포인트는 그대로,
    재학습 없음 — 홀드아웃 `success_rate`는 이미 40에서 **0.625**에 도달하고 80에서도 그대로
    유지된다. `envelope_violation_rate`는 계속 떨어진다(0.998 → 0.90–0.92 → 0.56–0.64). 넷째
    단계는 `acceleration_max = 80`과 동시에 `velocity_max`를 3.0 → 4.4 rad/s로 넓혀 가속도가
    걸림돌에서 빠진 뒤 속도가 새 병목이 되는지 보았다 — 실제로 그렇다(`acceleration_max = 80`
    단독에서도 `violation.velocity`가 여전히 틱의 27–36 %) — 하지만 `velocity_max`를 4.0
    rad/s 너머로 넓히면 `action_rate.first_diff_max = 0.08` rad을 넘어서는데, 이는 픽스처
    자신의 주석이 `velocity_max <= 3.0`에서는 결코 걸리지 않는다고 적은 경계이고, 그 결과
    합친 단계는 두 스위트 모두에서 80 단독보다 *더 나쁜* 점수를 낸다. 그래서 "두 끝 중 하나가
    움직여야 한다"의 답은 두 기본값보다 더 좁다: **`acceleration_max`만, `velocity_max`는
    같이 넓히지 않는다.** 픽스처는 여전히 넓히지 않은 채로 남아 있다 — 이것은 Deployment IR
    결정 자체가 아니라 그 결정을 위한 데이터를 만드는 절제 실험이었고, (a) 자신의 틀대로
    오너의 몫이며, 40이나 80을 실제로 채택하기 전에는 7.26절이 짚은 토크 여유 재유도가 여전히
    필요하다.

    **결정(오너, 2026-09-16, V18 이후): `tests/fixtures/visible-learning/deployment.toml`의
    `acceleration_max`를 80 rad/s²로 옮긴다. `velocity_max`는 3.0 그대로.** 재유도는 픽스처의
    주석에 실었다: 80 rad/s²는 로터에 최대 2.24 N m, STS3215 스톨 토크의 4분의 3을 요구하며
    무부하 서보의 능력 안이고, 장면의 `forcerange`가 플레인이 무엇을 허용하든 시뮬 토크를
    계속 제한한다. 40이 아니라 80인 이유는 80에서 두 스위트가 모두 통과하고 워치독 래치가
    사실상 사라져(767–1,160틱 대 53–93틱) 이후의 평가가 플레인이 아니라 정책을 재기 때문이다.
    `deployment_hash`가 움직인다. V18의 80 단계 절제 사본은 새 문서와 내용이 비트 동일하므로
    그 숫자와 V18b의 80 단계 스윕·쇼케이스가 체크인된 픽스처의 기록이 된다. 이 변경 이후
    수집되는 시연은 최대 40 rad/s²로 가속한다(전문가는 엔벌로프의 절반으로 페이싱). V14
    데이터셋은 10에서 수집됐다.

26. **모든 숫자가 그대로인데 GPU 경합 아래서 `execution_hash`가 움직였다** (7.28절, V18b). 같은
    RTX 4090에서 80 단계 렌더가 도는 동안 40 단계 홀드아웃 스위트를 재현하니 리포트·
    `evaluation_hash`·`failure_mode_histogram`은 단독 실행과 같고 `execution_hash`만 달랐다
    (`745c9777…` 대 `c2322c2c…`). §5.3이 해시하는 아홉 슬롯 중 실행 시점에 계산되는 것은
    `runtime`과 `hardware_capability`뿐이므로, 둘 중 하나가 부하 아래서 다르게 읽힌 것이다 —
    스레드 수, 장치 질의, 능력 프로브. 정확성 문제가 아니라(결과는 비트 동일) 출처 문제다: 같은
    실행 둘이 다른 이름을 가져서는 안 된다. 기본값: **슬롯을 결정적으로 만든다** —
    `hardware_capability`와 `PolicyRuntime::runtime_hash`가 읽는 것을 측정값이 아니라 선언값으로
    고정하고, 단독/경합 쌍을 오라클로 추가한다. M5 리뷰 목록에 넣을 패킷이다.
27. **어느 시연 200개가 "현재 시연"이고, 내보내기는 무엇을 가리키는가** (7.27절, V19).
    `~/artifacts/plan-v/v14/ds-train`(35,918 프레임)은 성공 술어의 옛 `gripper > 0.6` 항에 맞춰
    수집되었다. V15는 그 항을 `> 0.85`로 올리고 `~/artifacts/plan-v/v15/ds-train`
    (36,960 프레임)으로 **다시 수집**했으며, 그것이 V15와 V17이 학습한 것이다. 각 40 에피소드에서
    측정한, 큐브가 바구니 안에 있는 동안의 턱 최대 각은 앞의 것이 **0.573 rad**, 뒤의 것이
    **0.832 rad**다. 즉 옛 집합에는 천장이 박혀 있고, 그것을 완벽히 모방하는 자는 Task IR이 지금
    지닌 술어를 만족시킬 수 없다. V19는 같은 외부 ACT를 둘 다에 학습시켜 홀드아웃 0/16과 13/16을
    측정했다. 트리 안의 무엇도 둘 중 하나를 폐기된 것으로 표시하지 않는다. 둘 다 오라클 서버의
    디렉터리이고, 둘 다 같은 씬의 200 에피소드이며, V19를 연 패킷은 틀린 쪽을 지목했다. 기본값:
    **`v15/ds-train`을 시연으로 삼고, 각 수집이 어떤 술어 아래 이루어졌는지를 내보내기의
    `es_provenance.json`에 기록한다.** 그러면 학습 실행이 은퇴한 술어에 조용히 적합될 수 없다.
    대안 — 술어가 바뀔 때마다 다시 수집하고 최신만 남긴다 — 은 V15가 이미 손으로 한 번 한 일이다.
28. **다른 스위트가 문서에 있으면 한 스위트의 숫자가 움직인다** (7.27절, V19).
    `evaluation-v8.toml`을 nominal 하나로 줄인 문서는 50,000스텝 체크포인트에서
    `success_rate = 0.6250`을 내고, *같은* 체크포인트의 6-스위트 스윕의 nominal 행은 `0.5625`를
    내며 성공한 셀도 다르다. 샤딩 때문이 아니다. 줄인 문서를 `--jobs 8`로 다시 돌리니 `--jobs 6`
    실행에 대해 `0.6250`이 바이트 단위로 재현되었고, `--jobs 4` 두 번째 반복도 또 바이트 단위로
    일치했다. 그러니 차이는 문서이고, §10.4의 "모든 스위트는 자기 `stream`에서 뽑으므로 행을 하나
    더해도 다른 행의 숫자를 움직일 수 없다" — 7.8절 이후가 기대 온 문장 — 은 측정된 이 쌍에 대해
    성립하지 않는다. 사람이 결정해야 할 것은, 한 스위트의 추첨이 그것이 실려 가는 Evaluation IR에
    의존해도 되는가(변호 가능: `evaluation_hash`는 `execution_hash` 안에 있으므로 두 실행은 정직히
    서로 다른 실행이다), 아니면 의존하면 안 되는가(역시 변호 가능: §10.4가 그렇게 말한다)이다.
    기본값: **규칙이 정해질 때까지 모든 숫자 옆에 문서를 명시한다.** 7.27절이 그렇게 한다. V11
    이후의 줄인-문서 표는 전부 이 방식으로 측정되었고 내부적으로 일관되지만, 스윕의 행과 맞바꿀 수
    있는 것은 아니다.
32. **`import-lerobot`이 `expected_latency_ms`를 지어내고, T7 이후 그 지어낸 숫자가 데모의
    숫자를 결정한다** (7.30절). `config.json`에는 지연 측정값이 없으므로, import는 배포가
    감내하는 가장 큰 값 — `min(inference_budget, 1000 / replanning_hz)`, 데모에서는 200 ms —
    을 선언하고 자신의 doc 주석에 그렇다고 적어 둔다. 평가에 지연 모델이 없는 동안 그것은
    무해했다. 이제 그것은 모든 에피소드 시작의 제어 10틱 개루프이고 모든 명령의 200 ms
    낡음이며, 같은 씬 같은 로봇에서 import된 ACT가 IR-그래프 정책의 15 ms보다 열 배 큰 지연
    아래에서 평가되는 이유다. 여기에 틀린 것은 없다 — 문서가 결정한다는 것이 설계 전체다 —
    그러나 ACT의 숫자는 측정된 적이 없다. 움직일 수 있는 끝이 셋이다. **(a)** `es bench`가
    레퍼런스 장치에서 체크포인트의 실제 추론 시간을 측정하고(§12.4) `import-lerobot`이 그것을
    쓴다. 이것이 정직한 답이고 하나의 패킷이다. **(b)** import는 아무것도 선언하지 않고 추측을
    거부한다. 그러면 문서는 사람이 쓸 몫이 된다. **(c)** 경계값을 유지하고 모든 숫자 옆에
    그것을 명시한다. 7.30절이 그렇게 한다. 기본값: **(a)가 생길 때까지 (c)** — 경계값은
    보수적인 방향이기 때문이다. 배포가 감내하는 최악의 지연에서 통과하는 정책은 실제 지연에서도
    통과한다.
35. **사전학습 백본이 row D의 학습률에서 발산하고, 증강이 그것을 구한다** (7.31절). U1과 U3는
    문서 하나 — Observation IR — 만 다른데, U1은 `first_nonfinite_step 4517`에 닿고 U3는
    20,000 스텝을 `final_loss 0.006383`으로 끝낸다. 적용된 학습률은 양쪽에서 같은 `f64`들이다
    (`lr_curve_hash c01d5185…`, T4의 row D). 즉 `lr = 4e-4`는 트레이너의 성질도 스케줄의
    성질도 아니고 *(스케줄, 초기화) 쌍*의 성질이며, 이 프로젝트가 커밋한 두 Learning IR 문서
    가운데 하나는 작동한다고 보여진 학습률을 아직 갖고 있지 않다. 움직일 수 있는 끝이 셋이다.
    **(a)** 패킷 하나가 사전학습 쪽의 학습률을 따로 측정한다 — 이 패킷이 금지당한 다섯 번째
    설정이다 — 그리고 레시피 픽스처가 학습률을 하나가 아니라 둘 담는다. **(b)** T4가 추가했고
    측정된 실행이 한 번도 쓰지 않은 `grad_clip`을 사전학습 쪽에서 켠다. 여섯 스텝 만에
    0.0151 → 0.303 → NaN으로 가는 손실은 노름 클립이 존재하는 이유인 모양이다. **(c)** 아무것도
    움직이지 않고 노트가 그 발산을 있는 그대로의 행으로 싣는다. 기본값: **(c)**. 데모의 결론 —
    7.29절의 것 — 이 여기에 걸려 있지 않기 때문이다. 다만 (b)는 레시피의 필드 하나이고 M7
    사다리 전체에서 가장 싼 실험일 것이다.
36. **`es eval compare`가 움직인 `evaluation_hash`를 넘어 말없이 비교한다** (7.31절). §13.3은
    움직인 evaluation 문서는 새 비교라고 말하고, `es loop cycle`은 두 해시를 모두 찍으며
    이름으로 거부한다(`training-recipe.md` 12.3). 같은 절이 독자에게 가리키는 비교 도구는
    확인하지 않으므로, `es eval compare U0/report.json U2/report.json`은 서로 다른 두 문서를
    가로질러 델타 열을 찍으면서 아무 말도 하지 않는다. 기본값: **거부가 아니라 두 해시를
    이름으로 적은 경고 한 줄**. 그 비교가 바로 사람이 원하는 것일 때가 있고(이 노트의 U0 → U2
    행이 그렇다) 도구의 일은 자기가 무엇을 보여 주는지 말하는 것이다. 이것은 M7 리뷰 항목이지
    이 패킷이 할 수 있는 변경이 아니다.
