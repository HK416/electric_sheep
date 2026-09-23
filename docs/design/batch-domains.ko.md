<!-- Korean translation of docs/design/batch-domains.md. The English file is the working copy; regenerate this when it changes. -->

# 배치 도메인, 리셋, 그리고 env RNG (M1 W6)

Spec §12 (네 가지 배치 도메인, 도메인 간 정책, 결정성, 9개 지표), 부록 B.5
(스케줄러 스케치), §6.4 (실행 의미론), §18.1 (정수 시간), §18.5 (실패 의미론),
§13.1 (에피소드 기록이 학습 루프에 공급됨). 리뷰 등급 C.

이것은 `es-env`가 M1 W6에서 실제로 구현하는 것이며 — 못지않게 중요한 것으로 —
구현하지 않는 것이기도 하다.

## 1. 네 도메인, 네 크기

Spec §12.1은 "모든 것의 단위는 env 하나다"라는 생각을 거부한다. 각 도메인은
자신만의 배치 크기와 자신만의 주기를 가지며, `es-env`는 절대로 하나를 다른
것에서 유도하지 않는다.

| 도메인 | `batch` | `period` (sim tick 단위) | Device |
|---|---|---|---|
| simulation | `N_sim` (예: 4096) | **항상 1** — tick 자체를 정의한다 | `Cpu` / `Gpu(id)` |
| observation | `N_obs` (예: 512) | `sensor_dt / dt_phys`, 예를 들어 1 kHz에서 33 | |
| inference | `N_inf` (예: 256) | `dt_ctrl / dt_phys`, observation 주기의 배수 | |
| training | `N_train` (예: 64 시퀀스), 선택적 | inference 주기의 배수 | |

`DomainCfg.period`는 항상 simulation tick의 정수 개수이며 결코 지속 시간이
아니다 (§18.1: float 시간 없음, 타입으로 강제됨 — `BatchDomains` 어디에도 `f64`는
없다). 주기가 실제로 뜻하는 벽시계 시간은 backend의 `TickRate`에서 오며, `Env`가
`ModelInfo`에서 읽는다.

### 검증 (`Schedule::build`)

다음은 거부되며, 각각 고유한 `EnvError::Schedule` 메시지를 낸다.

- `batch == 0`이거나 `period == 0`인 경우;
- `simulation.period != 1`인 경우;
- `inference.period % observation.period != 0`인 경우 — inference tick은 반드시
  observation tick 위에 있어야 한다. 그렇지 않으면 정의되지 않은 tick의 observation을
  소비하게 된다;
- `training.period % inference.period != 0`인 경우;
- `observation.batch > simulation.batch`이거나 `inference.batch > observation.batch`인
  경우 (§12.1: 깔때기는 오직 좁아지기만 한다);
- `observation.batch == 0`인데 선택이 `All`이 아닌 경우.

## 2. 정적 계획

`Schedule`은 순수한 데이터다: 하이퍼주기 `H = lcm(obs, inf, train)`과, 길이 `H`인
`Vec<TickPlan>`이며, `TickPlan { offset, observation, inference, training }`는
sim tick `t`에서 어떤 도메인이 발화하는지를 말한다(`plan[t % H]`). 이 계획은
무엇이 얼마나 빠르게 도는지에 전혀 의존하지 않으므로, 두 대의 기계가 동일하게
인터리빙된다. `Schedule::ticks()`는 하나의 하이퍼주기를 순회한다.

tick **내부에서의** 발화 순서는 §6.4의 위상 순서에 의해 고정되며 데이터가 아니다.

```
PrePhysics -> Physics -> PostPhysics -> Observation -> Reward -> Termination -> Record
```

따라서 둘 다 발화하는 tick에서는 observation이 inference보다 먼저 오며,
inference는 절대 자신의 tick보다 새로운 observation을 보지 않는다.

### 카메라 env 선택 (§12.2, §12.3)

`N_obs < N_sim`일 때, 활성 카메라 집합은 `(tick, obs_period, N_obs, N_sim)`의
결정적 함수다 — RNG도, 부하 피드백도 없다.

```
cycle  = ceil(N_sim / N_obs)
slot   = (tick / obs_period) % cycle
active = [slot * N_obs, min((slot + 1) * N_obs, N_sim))
```

이는 `k = N_obs`인 §12.2의 `round_robin(k, period)`이다. 모든 env의 카메라는
`cycle`개의 observation tick마다 한 번씩 방문되며, 그 순서는 §12.3이 요구하는
대로 env-ID 오름차순이다.

## 3. 리셋 프로토콜

`reset(None)`은 배치 전체를 리셋하고, `reset(Some(&envs))`는 부분 집합을
리셋하며 `capabilities().supports_reset_subset`을 요구한다(그렇지 않으면
`EnvError::Physics(Unsupported)`가 되며, backend가 그 이름을 밝힌다 — 우리가
그것을 흉내내지 않는다). env마다, 다음 순서로 진행한다.

1. `episode[env] += 1` — 에피소드 카운터는 모든 RNG 키의 일부이므로, 재리셋이
   이전 에피소드의 추첨을 절대 재생하지 않는다.
2. 해당 env의 리셋 버퍼 행(`qpos`, `qvel`)을 0으로 만든 다음, `ResetState`와
   `Randomization` 노드를 오름차순 `NodeId` 순서로 적용한다 (§6.4: node ID가
   동점을 깬다. 즉 ID를 보존하는 그래프 재작성은 추첨 순서도 보존한다).
3. 무작위화된 행들로 `backend.reset(envs, Some(&state))`를 호출하며, 다른
   env들의 행은 변경 없이 그대로 넘어간다.
4. `EpisodeRecorder::start(env)` — 이전 에피소드가 마무리되어 반환된다.

자동 리셋: `step`은 physics step 이후 termination을 평가한다. 종료된 env들은
같은 `step` 호출의 *끝에서* 리셋되므로, 에피소드 경계는 tick 경계와 일치하고
호출자는 절대 반쯤 리셋된 배치를 보지 않는다. 반환되는 done 플래그는 새 에피소드가
아니라 종료된 tick에 속한다.

격리된(quarantined) env들(§18.5)은 여전히 backend에 의해 스텝되지만 보상
리덕션과 기록되는 에피소드에서는 제외된다. `StepReport`의
`FailureKind::NanDetected` / `Diverged`가 `es-core`의 `FailurePolicy`를 거쳐
이를 유도한다.

## 4. RNG

Spec §3.4는 전역 RNG를 금지하며 §6.6 `DET-001`은 모든 무작위 소스에 스트림
이름을 필수로 요구한다. 따라서 `EnvRng`는 순차-상태적(sequential-stateful)이
아니라 **카운터 기반**이다.

```
key     = splitmix64_chain(seed, env_index, episode, stream_id.bytes[0..8], bytes[8..16])
nth     = splitmix64(key ^ (counter * GOLDEN))
```

테스트가 고정하는 결과들:

- 같은 `(seed, env, episode, stream)`은 언제나 같은 시퀀스를 낸다. 어떤 기계에서든,
  다른 env들이 무엇을 했는지와 무관하게 — 공유 상태가 없으므로 env 순서와 스레드
  개수는 무관하다.
- 다른 `env`, `episode`, `stream`은 서로 무관한 시퀀스를 낸다. env들은 같은
  seed를 공유하더라도 서로 상관되지 않는다.
- 스트림은 그 이전 것들을 재생하지 않고도 어느 시점에서든 다시 유도될 수 있다.
  이것이 부분 집합 리셋을 저렴하게 만들고 재생을 정확하게 만드는 요인이다.

`stream_id: StableId`는 `Randomization`/`ResetState` 노드가 선언한 스트림 이름의
`StableId::from_path(stream_name)`이다. 따라서 스트림 이름을 바꾸면 추첨과
task hash가 바뀌고, 그 외에는 아무것도 바뀌지 않는다.

**분포** (`es_ir::task::Distribution`, 다섯 가지 모두 다룸):

| Variant | 방법 |
|---|---|
| `Constant(v)` | `v`, 추첨 소비 없음 |
| `Uniform{lo,hi}` | `lo + (hi - lo) * u`, `u = (x >> 11) as f64 * 2^-53` |
| `LogUniform{lo,hi}` | `exp(ln lo + (ln hi - ln lo) * u)`; `lo > 0` 필요 |
| `Normal{mean,std}` | Box–Muller, `mean + std * sqrt(-2 ln u1) * cos(2 pi u2)`, 고정된 연산 순서, 두 번째 변량은 버림 |
| `Choice(vs)` | `vs[(next_u64 % len)]`, 거부 없는 방식 (현실적인 어떤 len에 대해서도 편향 < 2^-64) |

초월함수는 `std`가 아니라 `es_math::approx`를 거친다(`ln`, `exp`, `cos`, `sqrt`)
(§3.2 `DET-010`). 이들은 `f32`다. 무작위화 추첨은 `f64` 정밀도를 필요로 하지
않으며, x86/ARM/GPU 사이에서 비트가 동일한 추첨이 마지막 몇 개의 가수 비트보다
더 중요하다. 상태 배열이 그러하므로 추첨값은 여전히 `f64`로 반환된다.

## 5. 무작위화 대상

`RandomizationPlan::compile(task, scene)`는 각 노드의 `target` 문자열을 구성
시점에 `SceneDesc`에 대해 **한 번만** 해석한다. 그래서 리셋마다 도는 경로에는
문자열 작업도, 풀어야 할 `Result`도 없다.

```
qpos[i]  qvel[i]                       state indices
joint.<name>.qpos  joint.<name>.qvel   resolved via ModelInfo's joint -> IndexRange maps
body.<name>.mass                       scale factor
geom.<name>.friction                   scale factor
actuator.<name>.gain                   scale factor
```

그 외의 것은 컴파일 시점에 `EnvError::Unsupported(target)`이 되며 절대 조용히
건너뛰지 않는다 (§17.2의 규칙을, 무작위화에 적용한 것).

**모델 파라미터가 물리에 닿는다 (packet M11/X4, spec 28.14 규칙 4).** 세 개의 scale 대상은
전과 똑같이 추첨되어 `EpisodeMeta.param_scales`에 기록되고, 리셋 때 기존 trait의 메서드 하나를
통해 backend로 밀어 넣어진다 (INV-17: 새 trait 없음):

```
PhysicsBackend::set_params(&mut self, envs: &[u32], params: &[(Param, StableId, f64)])
    -> Result<(), PhysicsError>
```

`Env::reset`은 리셋되는 env마다 그 env의 추첨값으로 한 번, 추첨 뒤 state를 밀어 넣기 전에
이것을 부른다 — 계획에 `Scale` 항목이 있을 때만. 그래서 그런 항목이 없는 task는 이것을 결코
부르지 않고 그 바이트도 움직이지 않는다 (규칙 1). 추첨 순서와 기록되는 `param_scales`는
그대로다. `Param`은 `es-physics-core`에 있고 `es_env::randomize`에서 re-export된다.

scale은 모델이 **로드될 때의** 값에 대한 상대값이며, 마지막으로 적용된 값에 대한 것이 결코
아니다. 그래서 같은 추첨으로 두 번 리셋해도 편집은 두 번이 아니라 한 번이다:

| 대상 | 스케일되는 것 | 파생 필드 |
|---|---|---|
| `body.<n>.mass` | `body_mass` | `mj_setConst` (body_subtreemass, body/dof invweight0, actuator_acc0, meaninertia) |
| `geom.<n>.friction` | `geom_friction` 세 개 모두 (sliding, torsional, rolling) | 없음 |
| `actuator.<n>.gain` | `gainprm[0]`, 그리고 servo에서 그것을 거울처럼 따라가는 bias 항 — position actuator는 `biasprm[1]`, velocity actuator는 `biasprm[2]` — 그래서 servo는 servo로 남는다. `kv`는 스케일하지 않는다 | 없음 |

geom은 소유 body의 행과 그 body의 geom 중 몇 번째인지로 지정된다 (emitter는 body의 geom을
scene 순서대로 쓰고 `MuJoCo`는 그것들을 `body_geomadr`부터 연속으로 둔다). 그래서 방출된
geom 이름을 재구성하지 않는다.

backend별로 누가 구현하는지는 `Feature::ModelParams`가 말한다. trait의 기본 구현은
`PhysicsError::Unsupported("set_params")`이므로, 그것이 없는 backend는 추첨을 버리지 않고 첫
리셋에서 scale 대상을 이름으로 거부한다 (spec 17.2):

* **`mujoco-cpu`** — `mujoco_ref.py`는 env마다 `MjModel` 하나를 둔다. 첫 `set_params` 때 로드된
  모델의 복사본으로 만든다. 이것을 한 번도 부르지 않는 run은 모델 하나와 오늘의 바이트를
  유지한다. 측정 (oracle 2): 추첨된 scale은 `MjModel` 직접 편집 + `mj_setConst`를 500 step 동안
  비트 단위로 재현하고, 건드리지 않은 env는 편집 없는 run과 비트 단위로 같으며, 궤적 비트는
  Windows와 Linux 서버에서 같다.
* **`mjwarp`** — `mujoco_warp` 3.13 (서버, 2026-09-23)은 batch 가능한 모든 모델 필드를
  `worldid % shape[0]`로 인덱싱되는 선행 world 축과 함께 가진다 (`put_model` 뒤 `body_mass`,
  `geom_friction`, `actuator_gainprm`/`biasprm`, invweight들, `actuator_acc0`,
  `stat.meaninertia`가 모두 `(1, …)`). `mjwarp_ref.py`는 env별 CPU `MjModel`을 `mujoco-cpu`와
  똑같이 편집하고, 그 편집이 움직인 필드마다 그 world의 행으로 올린다. 처음에는 필드를
  `nworld` 행으로 넓힌다. 측정: 건드리지 않은 world는 편집 없는 run과 비트 단위로 같고, 두 번째
  적용은 아무것도 움직이지 않으며, 편집된 world는 `mujoco-cpu`의 편집된 run에서 9.5e-7
  떨어져 있다 (편집 없는 run에서는 1.6e-2) — 선언한 대로 tier 2.
* **`newton`** — 선언하지 않음. 이름으로 거부한다.

`MuJoCoCpuBackend::applied_params` / `MjWarpBackend::applied_params`는 쓰인 모든 파라미터의
`[nominal, applied]`를 env의 모델에서 다시 읽어 돌려준다: 추첨이 물리에 닿았다는 증거.

**렌더 타깃 (패킷 M11/X5, spec 28.14 규칙 4).** 문법에 렌더 부류가 생긴다. 렌더 추첨은 다른 모든
추첨과 똑같이 `EnvRng(seed, env, episode, stream)`로 키가 정해지지만, `RandomizationPlan::apply_render`가
env별 `RenderOverrides`에 뽑을 뿐 `ResetBuffer`에는 절대 쓰지 않는다. 그래서 물리, `set_params`, 모든
물리 추첨은 건드려지지 않는다(오라클: `visual_randomization_undeclared_targets_move_nothing`).
`Env::reset`은 이를 에피소드에 기록하고(`Episode::render`, `param_scales` 옆), `Env::render_overrides(env)`가
프레임 소스에 넘기며, 프레임 소스는 렌더 시점에 적용한다(`es_env::render::drawn_frame`, `renderer.md` 13절).

```
light.intensity                    모든 광원에 대한 이득 (Rs: albedo와 emission; Pt: 발광체, 태양, 하늘)
light.direction                    yaw와 pitch, 도 -- 스트림 두 개, <stream>.yaw 와 <stream>.pitch
light.direction.yaw | .pitch       둘 중 하나
light.color                        채널별 RGB 이득 -- 스트림 세 개, <stream>.r .g .b
light.color.kelvin                 켈빈 색온도, 고정 표를 거친다
light.ambient                      앰비언트 항에 대한 이득 (Rs: `ambient`, [0, 1]로 클램프; Pt: 하늘)
light.radiance                     Pt 전용: 방향광의 복사휘도, 흰색 (없음 = 0 = 오늘)
light.sky                          Pt 전용: 하늘의 복사휘도, 흰색 (없음 = 0 = 오늘)
geom.<name>.rgba                   그 지오메트리의 채널별 RGB 이득, 알파는 그대로 -- 스트림 세 개
camera.<name>.pose.x | .y | .z     카메라 자신의 OpenCV 축을 따른 오프셋, 미터
camera.<name>.pose.roll|pitch|yaw  카메라 자신의 z | x | y 축에 대한 회전, 도
camera.<name>.fov                  초점 배율 s: 주점을 중심으로 fx, fy에 s를 곱한다
```

패킷이 열어 둔 선택들, 명시:

* **노드 하나, 스트림 여럿.** 채널이나 각도에 걸친 타깃(`light.direction`, `light.color`,
  `geom.<n>.rgba`)은 노드의 분포 하나로 각 성분을 `<stream>.<sub>`에서 뽑는다. 그래서 색의 세 채널은
  독립이고, 노드는 여전히 스트림 하나를 이름 붙인다. 값이 하나인 타깃은 선언된 스트림을 그대로 쓴다.
* **`geom.<n>.rgba`는 HSV 지터가 아니라 채널별 배율이다.** 배율은 Lambert 항에 선형이라
  `light.color`와 곱셈으로 합성되고 색공간 초월함수가 필요 없다. HSV는 렌더러의 `f32` 동등성 규칙이
  떠안을 이유가 없는 색상 회전을 요구한다.
* **`camera.<n>.pose`는 축 하나짜리 타깃 여섯 개다.** 미터 단위 이동과 도 단위 회전은 분포 하나를
  공유할 수 없다. 맨 `camera.<n>.pose`는 이름으로 거부된다. 회전 순서는 yaw, pitch, roll이며 각각
  카메라 자신의 축에 대해서다.
* **`camera.<n>.fov`는 초점 배율을 뽑는다.** 그래서 기록된 `fx`, `fy`는 정확히 공칭값 곱하기 추첨값이다
  (오라클 4). `s > 1`은 시야를 좁히고, 수직 fov는 `2 atan(tan(fovy / 2) / s)`가 된다. 분포는 전체
  지지에서 양수여야 한다 — `Normal`, 0에 닿는 `Uniform`, 양수가 아닌 `Constant`나 `Choice`는 거부된다.
* **`light.radiance`와 `light.sky`는 `Rs` 센서를 선언한 태스크에서 거부된다.** 래스터라이저에는 둘 다
  없고, 어느 프레임에도 보이지 않는 추첨은 조용한 건너뜀이 된다(spec 17.2). 이들에 `Constant` 분포를
  주는 것이 새 IR 필드 없이 문서가 `Pt` 태양을 *선언*하는 방법이다(M10 S-6).
* **켈빈 표**는 Tanner Helland의 흑체 근사를 2000 K부터 10000 K까지 1000 K마다 8비트 sRGB로 샘플링한
  것으로, 255로 나눈 선형 배율로 쓰며 행 사이는 구간 선형, 양 끝은 클램프한다. 표가 곧 정의다.

**뽑힌 시야각이 `ImageSpec`에 하는 일 (`INV-14`).** `Env::new`는 선언된 모든 이미지 센서를
`(camera, ImageSpec)`으로 모으고, 매 리셋은 `Episode::image_specs[camera] =
RenderOverrides::image_spec(camera, declared)`를 쓴다 — `fx`, `fy`에 초점 배율을 곱하고 `cx`, `cy`는
그대로, 뽑힌 오프셋은 `extrinsics`에 합성한다. 그래서 내부 파라미터를 소비하는 쪽은 뽑힌 값을 읽고,
하류의 `ImageSpec::resized` / `cropped`는 선언값을 변환하듯 그것을 변환한다
(`camera_fov_draw_moves_intrinsics`). 렌더러가 쓴 프레임도 같은 숫자를 지닌다:
`EnvRenderer::frame_with`는 카메라 추첨이 있는 프레임의 사이드카에 `intrinsics`를 쓴다
(`Tile::write_to_with_intrinsics`). 카메라 추첨이 없는 프레임은 늘 쓰던 사이드카를 쓴다.

**지금 추첨을 적용하는 쪽.** `es_native.Rollout`(RL 경로, X3의 env별 카메라)은
`Env::render_overrides(env)`를 `EnvRenderer::frame_with`에 넘긴다. `es loop collect --frames`와
`es eval run --frames`는 여전히 `frame` / 자체 렌더러를 불러, 렌더 타깃이 있는 태스크에서도
**추첨되지 않은** 장면을 렌더한다 — X5의 범위 밖이며, 수집기와 평가기가 추첨을 보게 만드는 후속
작업이다(`docs/packets/M11/X5-visual-dr.md`).

## 6. 보상과 종료

`Reward`와 `Terminate`는 표현식 리터럴이 아니라 입력 엣지를 가진 그래프
싱크(sink)다. `es-env`는 각 싱크로 들어오는 *스칼라 원뿔(scalar cone)*을
구성 시점에 (`plan.rs`) `es_ir::task::Expr`로 한 번 낮춘 뒤, env마다 tick마다
이미 존재하며 결정적이라고 이미 명세된 `Expr::eval`로 평가한다(고정된 연산
순서, `NaN` 대신 `None`).

원뿔에서 지원되는 것: `GetJointState`, `GetSensor`, `GetTime`, `GetBodyPose`
(잎, 이름 붙은 포트에 바인딩됨), `Arith`, `Compare`, `Clamp`, `Normalize`,
`Logic`, `Norm { kind: L2 }`. reward나 terminate 원뿔 안의 그 외 모든 노드
종류는 `EnvError::Unsupported("<kind> in a reward or termination cone")`이다 —
이름을 불러 거절하지, 근사하지 않는다. 포트 이름은 그 잎이 읽는 상태
인덱스에서 온다: `qpos[i]`, `qvel[i]`, `sensor[i]`, `xpos[i]`, `time`,
`time.episode`.

**레인(lane)** (패킷 M8/S4d). 낮추기는 `Expr` 하나가 아니라 *레인마다* `Expr`
하나를 돌려준다. 스칼라 잎은 한 레인이고, `GetBodyPose { relative_to: World }`는
세 레인이다 — `ModelInfo::body`가 주는 행에서 읽은 `StateView::xpos`의 월드
위치이며, 덕분에 두 물체 사이의 거리를 드디어 적을 수 있다. 레인 수가 같은
`Arith`는 레인별로 계산되고 한 레인짜리 피연산자는 반대쪽으로 브로드캐스트된다.
`Norm { kind: L2 }`는 `n`개 레인을 제곱의 합의 `Sqrt`로 접으며, 그 합은
**레인 순서로** 누산된다(`DET-020`). 그래서 결합 순서가 모든 백엔드, 모든
실행에서 `Sqrt(((x*x + y*y) + z*z))`로 고정된다. `Compare`, `Clamp`,
`Normalize`, `Logic`과 두 싱크는 정확히 한 레인을 요구하며, 벡터가 거기
도달하면 조용히 첫 레인을 쓰는 대신 이름을 부르며 거절한다: 큐브의 `x`를
거리인 양 점수화하는 것이야말로 이 규칙이 막으려는 버그다. 방향(quaternion)
포트, `World`가 아닌 프레임, `L1` / `Linf`, 로드된 모델이 색인하지 않는 물체도
각각 이름을 불러 거절한다.

**`sqrt`는 `DET-010`의 초월함수가 아니다**(§6.6). IEEE 754는 제곱근을 올바르게
반올림하도록 요구한다 — 하드웨어 명령 하나, 모든 타깃에서 같은 비트 — `exp`나
`sin`과 달리 `es-math::approx`가 필요 없다. 그래서 `Expr::Sqrt`가
`es-ir-types`에서 물체 거리 원뿔이 치르는 비용의 전부이며, 음수 피제곱근은
다른 모든 비유한 결과와 마찬가지로 `NaN`이 아니라 `None`이다.

보상 집계는 `Reward` 노드들에 대해 오름차순 `NodeId`로 `sum(weight * term)`을
계산하는 것이다. `Mean`, `Min`, `Max`는 벡터 항의 원소들을 집계하며 여기서는
단일 원소 no-op이다.

## 7. 지표

`EnvMetrics`는 §12.4의 이름들을 싣고 있으며, 모든 필드가 `Option`이다 — 아무도
측정하지 않은 지표는 조작된 0이 아니라 항상 `None`이며, `step/s` 필드는
**아예 존재하지 않는다**. `es-env`는 layer 9이고 `es-telemetry`는 layer 10이므로
`EnvMetrics`는 telemetry 의존성이 전혀 없는 평범한 구조체다. `es-telemetry`가
위에서 그것을 `PerfMetrics`로 변환한다. W6는 실제로 관측하는
`physics_steps_per_sec`, `actions_per_sec`과 tick/wall-time 카운터를 채운다.
렌더, inference, VRAM 필드는 해당 도메인이 생기기 전까지 `None`으로 남는다.

## 8. 미룬 것

- **비동기 추론과 chunk 버퍼** (§12.3, §8.6, B.5 `ChunkArrival`). 스케줄은 이미
  inference tick을 표시해 두고 있다. `apply_at = computed_from + deterministic_delay`와
  `chunk_underrun_rate`는 `es-policy` (layer 8)와 함께 도착한다.
- **`EnvSelection::Subset(Vec<EnvId>)`** — 지금은 `All`과 위의 round-robin
  유도만 있다.
- **observation 도메인의 실제 작업.** W6는 observation tick을 스케줄할
  뿐이며, 렌더링과 Observation IR 파이프라인은 `es-render` / `es-compile`
  패킷의 몫이다.
- **Safety Plane 통합** (§9). `es-safety`는 이웃 패킷에서 진행 중이며, `Env`는
  아직 액션을 clamp하지 않는다. 도착하면 그 훅은 `step` 안, `set_ctrl` 이전의
  호출 하나가 될 것이며 — 선택적이지 않을 것이다 (INV-12).
- **training 도메인 실행** — 스케줄은 되어 있지만 `es-data`로의 인계는 이후
  W의 몫이다.
- **Backend 모델 파라미터 무작위화** — §5의 한계를 참조.

---

# M2 W2 — 라운드로빈 실행, 비동기 추론, chunk 버퍼

위 §8은 이 중 네 가지를 미룬 것으로 나열했다. 이 절은 그 항목들을 대체하며,
§1–§7은 여전히 M1 W6 기반을 설명하고 변경되지 않는다.

## 9. 실행기

`DomainRunner<NJ, H>`는 `Env` 위에서 `Schedule`을 실행한다. 이것의 시계는
**제어 tick**이다: `inference.period`만큼의 simulation tick 구간이며, 이는
정확히 `Env::step` 하나가 전진시키는 양이다. `Env::step_with_policy` 하나가
§12.1 위상 순서로 진행되는 제어 tick 하나다.

```
observation ticks in [t, t+inference.period)
    └─ Schedule::observation_envs(t)        round_robin, §12.2 — only these envs
         └─ CpuPlan::run (or the raw qpos‖qvel row)  → latest[env]
inference tick in the same window
    └─ submit(env, control_tick) ascending by env id, §12.3
poll(control_tick)
    └─ up to inference.batch, FIFO → one PolicyRuntime::infer over the stacked batch
         └─ ChunkBuffer::push(chunk, submit_tick + latency_ticks)     ← App. B.5 apply_at
every control tick
    └─ ChunkBuffer::next_action → SafetyPlane::validate → Env::step(ctrl)
```

마지막 화살표는 chunk에서 actuator로 가는 **유일한** 경로이며, 두 갈래 모두
그것을 거친다: underrun은 plane에 `ActionChunk::empty`를 건네고, plane은
fallback으로 응답한다(§8.6, §9.4). 어떤 갈래도 plane을 우회해 `set_ctrl`에
도달하지 않는다. 테스트에서도 마찬가지다 — 테스트 범위(envelope)는 넓어질
뿐 절대 비활성화되지 않는다(INV-12).

### 제어 tick마다가 아니라 정책 결과마다 하나의 `seq` (P-M2-R3)

`SafetyPlane::accept`는 아직 보지 못한 `seq`일 때만 chunk를 받아들이며, 그것이
`last_chunk_tick`을 옮기는 유일한 것이기도 하다 — `ViolationKind::InferenceDeadline`이
측정하는 타임스탬프다. 그래서 실행기는 `seq`를 **정책 호출 결과당 한 번** 찍는다.

- 현재 tick을 커버하는 결과는 새 `seq`와, 다음 결과가 도착할 때까지 그것이 구동할
  행들을 받는다(`ChunkBuffer::action_at`을 선견(lookahead)으로 사용). 이후 plane
  자신의 커서가 `es-runtime-embedded`가 재계획하지 않는 tick에서 하는 것과 정확히
  같은 방식으로 그 행들을 따라간다. 이 선견은 정확한데, `ChunkBuffer::arrivals`를
  바꾸지 않고서는 아무것도 버퍼에 도달할 수 없고, 바로 그것이 다음 재구성을
  촉발하기 때문이다.
- 그 외의 모든 tick은 이전 `seq`를 재제출한다. plane은 이미 보유한 `seq`의
  페이로드를 무시하므로, 그런 tick들은 `last_chunk_tick`을 갱신하지도, 행동을
  조작해내지도 않는다.
- 에피소드 리셋은 뒤에 행 없이 `seq`를 올린다. 그래서 plane은 끝난 에피소드의
  chunk를 계속 소비하는 대신 그것을 버린다(§13.1).

제어 tick마다 새 `seq`를 찍던 예전 방식은 죽은 정책과 살아있는 정책을 구별할 수
없게 만들었다: `InferenceDeadline`은 `DomainRunner`를 통해서는 절대 발화할 수
없었고, 멈춘 정책은 오직 `ChunkUnderrun`으로만 나타났다.

두 가지는 단일 인스턴스가 아니라 배열인데, 둘 다 그 배후의 상태가 env마다
다르기 때문이다: `planes: &mut [SafetyPlane]`(hold target, rate-limit 이력,
e-stop 래치는 로봇마다 다르다, §9.3)와 `plans: &mut [CpuPlan]`(`TemporalWindow`
링은 env마다의 이력이다, §7.5). 공유 인스턴스라면 한 env의 이력이 다른 env의
것과 뒤섞이게 된다.

## 10. 시뮬레이션된 지연 (§12.3)

`latency_ticks = ceil(expected_latency_ms × control_rate)`는
`RuntimeHints::expected_latency_ms`로부터 정수로 한 번 계산된다. 이것이 비동기
모델의 전부다.

- 제출(submission)은 `submit_tick + latency_ticks <= control_tick`일 때
  방출되며, 실제 backend가 더 일찍 끝나더라도 절대 더 일찍 방출되지 않는다 —
  §12.3의 결정적 모드는 기다린다.
- chunk는 도착한 tick이 아니라 `submit_tick + latency_ticks`에 적용된다
  (부록 B.5 `ChunkArrival::apply_at`). 좁은 `inference.batch`가 그것을 늦게
  전달했다면, 이미 과거가 된 행들은 그냥 서빙되지 않는다. 이는 조용한 시간
  이동이 아니라 카운터상의 underrun으로 나타난다.
- 그 무엇도 시계를 읽지 않는다. `Instant`는 `es-env`에서
  `EnvMetrics::simulation_wall`에만 등장하며, 이는 측정값일 뿐 어떤 결정에도
  입력되지 않는다.

### 큐는 유한하다 (P-M2-R4)

`max_pending`의 기본값은 `inference.batch × latency_ticks + batch`다 — 파이프라인이
뒤처지지 않을 때 실제로 진행 중인 작업량과 정확히 같다: 전체 latency 동안
tick마다 방출되는 배치 하나에, 지금 채워지고 있는 배치를 더한 것이다.
`with_max_pending`이 이를 재정의한다(최소 배치 하나로 클램프됨).

이 한도를 넘기는 제출은 버려지고 `dropped_submissions`에 집계된다. **가장 최신**
것이 버려지므로 큐는 여전히 FIFO를 유지하며 back-pressure는 여전히 스케줄
순서대로 작업을 지연시킬 뿐 뒤섞지 않는다. 버려진 관측은 chunk를 만들어내지
않으므로, 다른 모든 누락된 chunk와 같은 곳에서 드러난다 — underrun으로, 그다음
plane의 fallback으로. 유한하지 않은 큐였다면 `inference.batch`가 도착률에 못
미칠 때마다(§12.2의 round-robin이 존재하는 이유인, 통상적인 과다구독 상황) 한없이
자라났을 것이고, §28.4 게이트 구성에서 이는 §20.3이 금지하는 메모리 부족
상태다.

실제 스레딩은 이후 패킷의 몫이며 바로 이 인터페이스 뒤에 자리 잡아야 한다:
`submit`이 worker에게 작업을 넘기고 `poll`이 그것을 되받으며, 방출 규칙은
여기 그대로 남아 worker를 바꿔 끼워도 의미론이 바뀌지 않는다. 결정적 방식이
먼저다(§12.3); 실시간 모드는 더 일찍 방출하고 그 편차를 replay에 기록한다.

## 11. Chunk 버퍼 (§8.5, §8.6)

`ChunkBuffer<NJ, H>`는 `CHUNK_SLOTS = 8`개의 chunk를 인라인으로 보유한다.
`next_action`은 아무것도 할당하지 않는다: 커버링 집합은 인라인
`[usize; 8]`이며 push 순서로 삽입 정렬되어 있으므로, 결합 순서는 항상 슬롯
순서가 아니라 도착 순서다 — 이것이 앙상블 합을 비트 단위로 재현 가능하게
만드는 요인이다.

| `ChunkBlendPolicy` | 규칙 | chunk 하나의 span |
|---|---|---|
| `HardSwitch` | 가장 최근의 커버링 chunk가 승리 | `min(valid, K)` |
| `LinearBlend { steps }` | `steps` tick에 걸쳐 이전 → 최신으로 램프 | `min(valid, K)` |
| `TemporalEnsemble { weight_decay }` | `w_i = exp(-m·i)`, `i = 0`이 **가장 오래된 것** | `valid` |

ACT는 가장 오래된 겹치는 예측에서부터 지수 가중치를 인덱싱하므로, `m`이
작을수록 새 관측을 더 빨리 반영한다; `f64::exp`가 아니라
`es_math::approx::exp`가 쓰인다(§3.4는 결정적 경로에서 std 초월함수를
금지한다). `K = execute_chunk`는 앞의 두 블렌드를 제한하는데, §8.5가 `K`
이후 재계획하기 때문이다; `TemporalEnsemble`은 `valid`한 모든 행을
의도적으로 읽는데, 겹치는 부분을 평균하는 것 자체가 그 방법이기 때문이다.

`next_action`은 underrun 시 `None`을 반환하고 그것을 카운트한다. 절대 행을
조작해내지 않는다 — 조작된 행동은 아무 검사도 거치지 않은 채 actuator에
도달하게 될 것이다. `action_at`은 실행기가 구성하는 도착당(per-arrival) chunk를
위해 카운터 없이 수행하는 동일한 조회다. 카운터는 여전히 제어 tick당 하나이며,
이것이 §12.4의 `chunk_underrun_rate`가 나누는 분모다.

`CHUNK_SLOTS` 자체는 `chunk_buffer_bytes`와 함께 `es_core::sizing`에 있다.
`es-compile`의 메모리 예산이 바로 이 버퍼들의 크기를 산정해야 하는데 `es-env`에
의존할 수 없기 때문이다(§4.2). 공식 하나, 호출자 둘(P-M2-R5).

**증분은 여기서, 오직 여기서만 적분된다**(spec 8.5, packet M9/T1). `action.space`가
`JointDelta`인 Deployment IR은 버퍼가 제공하는 행이 현재 목표에 대한 *변화량*이라고
말한다. 그래서 `es_env::chunk_buffer::absolute_target`이 그것을 Safety Plane이 마지막으로
**실행한** 명령에 더한다 — 에피소드의 첫 tick에서 그 값은 `SafetyPlane::observe_state`가
명령 사슬을 시드한 측정 자세다 — 그리고 플레인은 지금처럼 절대 목표를 검증한다. 원시
행이 아니라 실행된 값에서 적분한다는 것이 규칙의 전부다: 클램프된 증분이 팔이 도달할 수
없는 목표로 누적되어서는 안 된다. 그 경로에서 플레인에 건네는 chunk는 제어 tick당 한 행을
새 `seq`로 담는다. 플레인이 이미 수락한 행은 다시 적분할 수 없기 때문이다. 감추지 않고
적는 비용: 증분 정책에서는 `InferenceDeadline`이 발화할 수 없고, 죽은 정책은 재계획 구간
하나만큼 뒤에 `ChunkUnderrun`으로 잡힌다.

## 12. 배치 독립성이 실제로 의미하는 것

16-env 실행은 고정된 `observation.batch`에 대해 비트 단위로 replay된다.
서로 다른 `observation.batch` 사이에서는 그렇지 **않으며**, 이것은 결함이
아니라 옳은 동작이다: round-robin은 어느 tick에 어느 env가 관측되는지를
바꾸고, 관측된 env만 제출하며, 제출한 env만 chunk를 받고, chunk가 없는
env는 Safety Plane의 fallback을 받는다. observation 배치를 절반으로
줄이면 각 env의 관측 빈도가 절반이 되고 `chunk_underrun_rate`가 올라간다.

실제로 성립하는 불변량, 즉 테스트가 단언하는 것은 다음과 같다.

> 두 env는 그들의 observation tick이 일치할 때에 한해 구별 불가능하다.

구체적으로, `simulation.batch = 16`일 때 env들은 `16 / observation.batch`개의
round-robin 그룹으로 나뉜다; 궤적은 그룹 내부에서는 동일하고 그룹 사이에서는
다르며, `underrun_rate(4) > underrun_rate(8) > underrun_rate(16)`이다.

### §28.4 게이트의 규모 산정

`DomainSizing`은 구성(configuration)에 대한 정수 연산이다 — 할당이 없으므로,
게이트 구성을 실제로 돌릴 수 없는 노트북에서도 비용을 산정할 수 있다.
`GATE`(4,096 sim env, 512 obs env, 2 view, 224×224, 30 Hz, `H = 20`,
`NJ = 7`)에 대해:

| 수량 | round-robin 512 | 모든 카메라 켬 |
|---|---|---|
| `camera_frames_per_sec` | 30,720 | 245,760 |
| `pixels_per_sec` | 1.54 G | 12.3 G |
| 렌더 출력 (RGB8) | 4.6 GB/s | 37 GB/s |
| `f32` 정규화 후 | 18.5 GB/s | 148 GB/s |
| chunk 버퍼 (8 slots × `H` × `NJ` × f64) | 36.7 MB | 36.7 MB |

마지막 두 열의 8배 차이가 §12.2의 논거 전부다. 이 모두는
**`Target / Status: 미검증`**이다(§12.4): §28.4 W2 게이트 — 4,096 sim env ×
512 obs env 안정 — 는 여기서는 측정이 아니라 예산(budget)이며, 실제로
실행되는 테스트는 축소된 16-env 버전이다.
