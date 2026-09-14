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
es policy lower --policy untrained.esb --out build/
<py> python/es/train_act.py --module build/ --dataset ds/ --out model.safetensors
es policy pack --policy untrained.esb --weights model.safetensors --out trained.esb
```

- `lower`는 `TorchModule.source`를 그대로 쓰고, 더해서 `contract.json` — 로워링이 선언하는
  `weight_keys`와 `weight_shapes` (`crates/es-policy/src/lower/torch.rs:297`), 그리고 `lowering_hash`.
- `train_act.py`는 그 소스를 `exec`하고 `EsPolicy()`를 만들고 LeRobot v2.1 데이터셋을 읽고 옵티마이저를
  돌린 뒤 **`contract.json`의 키로만** `safetensors`를 쓴다. IR에 대해 아무것도 모르며 레이어를 정의할
  권한이 없다.
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
5. **비디오 코덱.** 측정 결과 서버에서는 `mp4v`만 인코딩된다 (섹션 2.9). 기본값: `mp4v`로 출시. H.264
   파일을 원하면 사람이 `ffmpeg`을 (또는 `libx264`가 있는 OpenCV를) 설치한다 — 플랜 V는 아무것도
   설치하지 않는다.
6. **사전학습 백본.** `_backbone("resnet18", 512)`가 ImageNet 가중치를 로드하는지, 따라서 학습에
   네트워크가 필요한지는 미검증이다. 기본값: 그렇다면 데모의 작은 이미지로 처음부터 학습한다.
7. **PNG.** 기본값: 기존 골든 포맷인 원시 `.bin` + 사이드카 (섹션 7.2). 사람이 이미지 뷰어로 프레임을
   열고 싶을 때만 PNG 인코더를 추가한다.
8. **4090 위의 CPU 전용 PyTorch** (섹션 2.9). 기본값: 받아들이고 학습 시간을 말하지 않는다. 사람이
   `~/venvs/es-lerobot`에 CUDA 빌드 torch를 설치하는 것이 플랜 V 일정에 가장 가치가 큰 선행 조건이다.
9. **`lerobot[dataset]`이 서버에 미설치**라 `lerobot.datasets`가 import되지 않는다. 사람이 설치하기
   전까지 V1의 가장 중요한 오라클은 영구 `SKIP`이다.
10. **게이트 7.** 플랜 V를 닫으면 SO-101 / 큐브-바구니 대체를 기록한 채 §28.7 게이트 7을 충족으로
    기록하는가, 아니면 RGB 2뷰 Franka를 위해 게이트 7을 계속 열어 두는가? 기본값: 충족으로 기록하되,
    대체 사실과 섹션 1의 `Target / Status: unverified` 행들을 M5 리뷰에 명시한다.
