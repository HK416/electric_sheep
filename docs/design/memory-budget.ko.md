<!-- Korean translation of docs/design/memory-budget.md. The English file is the working copy; regenerate this when it changes. -->

# 메모리·대역폭 예산 모델

`crates/es-compile/src/budget.rs`(layer 7)에 대한 설계 노트. 명세: spec 20.1,
spec 20.2, spec 20.3, spec 15.2(타일 아틀라스 크기 산정), spec 12.4(9개 지표
집합), spec 5.2(배치 도메인), spec 28.4 / spec 28.7 gate 13(±10% 정확도
게이트). Work packet: `docs/packets/M2/W5-memory-budget.md`.

## 1. 이것이 무엇이고 무엇이 아닌지

Spec 20.3: 예산은 scene/task/policy가 로드되기 **전**에 계산되어 실제
메모리와 대조된다; 초과하면 `N_obs`/`N_inf`를 축소하거나 프로세스가
OOM으로 죽게 두는 대신 명시적으로 실패한다. `MemoryBudget::estimate`는 그
중 정적인 절반이다: spec 20.2가 나열하는 모든 항목을, IR shape과 설정만으로
계산하며, 사람이 손으로 검산할 수 있도록 항목별로 공식을 그대로 적어둔다
(`BudgetItem::formula`).

**이 노트와 이 패킷에 없는 것:** 실제 실행 측정값(이 환경에는 대조할 GPU가
없다), 자동 `N_obs`/`N_inf` 축소 루프, 에디터의 실시간 표시. 이들은
디바이스를 필요로 하며 이후 작업이다. GPU 실행이 존재하기 전까지, 이
패킷이 산출하는 모든 수치는 spec 20.1의 기준 측정값 대비
`Target / Status: 미검증`이다(spec 28.4의 ±10% 게이트, spec 28.7 gate 13).

## 2. spec 20.1 기준 측정값 (calibration target, 미검증 (unverified))

| 시나리오 | VRAM | 상태 |
|---|---|---|
| Isaac Lab, RTX 4090, Cartpole 물리만, 4,096 env | 3.3 GB | 미검증 (unverified) |
| Isaac Lab, RTX 4090, Cartpole + RGB 카메라, 1,024 env | 16.7 GB | 미검증 (unverified) |
| Isaac Lab, RTX 4090, G1 locomotion, 4,096 env | 6.1 GB | 미검증 (unverified) |

비전 태스크는 env가 1/4인데 VRAM은 5배다 — 이것이 spec 12.2의
`round_robin`이 존재하는 이유이며, 이 모델이 실제 GPU에 맞춰 보정된 뒤에는
향후 GPU-in-the-loop 실행이 ±10% 이내로(spec 28.4) 재현해야 하는 검산
대상이기도 하다.

spec 20.2의 실제 계산 예(512 obs env × 2 view × 224×224 RGB, ACT,
`N_inf = 256`)가 두 번째 보정 목표다.

```
atlas RGB8 double-buffered      512x2x224x224x3x2       =  308 MB
normalized FP32 intermediates   512x2x224x224x3x4       =  616 MB
policy weights (ACT 52M fp32)                           =  208 MB
inference activations (batch 256)   needs measurement    ~ 2.0 GB
chunk buffers, 4096 env          4096x50x8x4x2          =   13 MB
--------------------------------------------------------------------
vision + policy subtotal                                 ≈ 3.1 GB
+ physics backend (MJWarp 4096 env)                       ≈ 2-4 GB
+ PyTorch, if training shares the GPU                     ≈ 8-16 GB
```

`MemoryBudget::estimate`는 그 예시와 같은 형태의
`ObservationIr`/`LearningGraph`로부터 처음 두 줄을 정확히 재현한다(render
tile atlas, observation 중간값); 추론 활성화값과 정책 가중치는 이 모델이
추측하는 대신 `unavailable`로 명시하는 두 항목이다(§4). chunk 버퍼 줄은
의도적으로 재현하지 **않는다**: 13 MB가 아니라 4096×8×50×8×8 = 105 MB인데,
런타임이 env마다 여덟 개의 겹치는 f64 chunk를 유지하기 때문이다 — 아래 §3을
보라.

## 3. 항목과 그 공식

모든 항목은 `BudgetItem { name, bytes, formula }`다. `bytes == 0`이면서
formula가 `"unavailable: ..."`로 시작하는 것은 "이것으로 크기를 셈할
데이터가 없다"는 뜻이지 "이것은 비용이 0이다"라는 뜻이 결코 아니다 —
`Display` 테이블과 CLI가 바로 이 이유로 formula를 함께 출력한다.

| 항목 | 공식 | 출처 |
|---|---|---|
| `physics_state` | `(nq+nv+nu+nsensordata) x n_sim_envs x 8B`(f64, spec 3.1의 상태 표현) | `ModelSizes`(호출자가 제공, `es_physics_core::backend::ModelInfo`를 미러링) |
| `render_tile_atlas` | 명시적 레이아웃 없음: `n_obs_envs x n_views x H x W x ch x dtype_bytes x 2`(더블 버퍼). `TileAtlasCfg`가 있으면: `rows x tiles_per_row x tile_w x tile_h x ch x dtype_bytes x 2`, `rows = ceil(tiles / tiles_per_row)` — 타일 개수가 딱 나누어떨어지지 않을 때의 spec 15.2 행 패딩 | 첫 번째 `ImageInput` 노드의 원본 센서 `ImageSpec` |
| `observation_intermediates` | `sum(CPU plan arena buffers) x n_obs_envs` | `PlanMode::Release`에서의 `es_compile::plan::CpuPlan::compile` — 이것이 *바로* spec 11.1 liveness 분석이며, 다시 유도하지 않고 재사용한다 |
| `history_buffers` | History 항목에 대한 합: `depth x per_sensor_bytes x n_obs_envs` | `ObservationIr::temporal.history`(spec 7.5 layer 1)를 각 sensor id를 소유하는 `ImageInput`/`StateInput` 노드와 조인 |
| `policy_weights` | 항상 `unavailable` | `WeightsRef`(spec 8.3)는 경로와 blake3 해시만 실을 뿐 바이트 크기가 없다(`INV-16`은 로딩을 `safetensors`로 유지할 뿐 크기 필드를 추가하지 않는다) |
| `inference_activations` | `sum(LearningNode output shapes) x inference_batch x precision_bytes` | `IrNode::outputs`를 통한 `LearningGraph::nodes`(각 노드가 선언한 출력 포트, spec 8.3) |
| `chunk_buffers` | `n_sim_envs x CHUNK_SLOTS x horizon x action_dim x 8B`(f64) — `es_core::sizing::chunk_buffer_bytes`, **spec 20.2의 `x 4B x 2`에서 벗어남**, 아래를 보라 | `LearningGraph::policy.contract`(`horizon`, `action_dim`) |

### `chunk_buffers`: 모델은 하나뿐이고, 그것은 spec 20.2의 것이 아니다 (P-M2-R5)

spec 20.2는 `n_sim_envs x H x NJ x 4B x 2`를 예산으로 잡는다 — f32 chunk 하나를
더블 버퍼링한 것이다. 런타임은 env마다 `CHUNK_SLOTS = 8`개의 *겹치는* f64
chunk를 유지하는데, ACT의 temporal ensembling이 살아있는 모든 chunk를
평균하고(spec 8.6) 제어 경로 전체가 f64이기 때문이다. 이는 spec의 수치의
4배이며, 옳은 쪽은 구현이다: spec의 그 줄은 ensembling 버퍼보다 먼저
쓰였다.

같은 양에 대한 두 개의 in-tree 모델이 있으면 spec 28.7 gate 13의 ±10% 검사가
도달 불가능해지므로, 정확히 하나만 존재한다 — layer 1의
`es_core::sizing::chunk_buffer_bytes`이며, `es_env::DomainSizing`(layer 9)과
이 예산(layer 7) 둘 다 이를 호출한다. 두 크레이트는 서로를 볼 수 없으므로
(spec 4.2), 이 공식은 둘 아래에 자리 잡는다. spec 12.4 게이트 구성(4,096
env, `H = 20`, `NJ = 7`)에서 이는 36.7 MB이며, `cargo test -p es-compile
budget`은 예산 항목이 그 함수와 같음을 단언한다.

`bandwidth_per_tick`는 `render_tile_atlas + observation_intermediates`다:
어떤 Hz와도 무관하게 simulation tick 한 번마다 이동하는 바이트 수다(spec
12.4의 대역폭 열은 이를 초당 수치로 바꾸기 위해 rate가 필요하다; 그것은
이 패킷이 아니라 측정 루프 패킷의 몫이다).

`per_domain`은 각 항목을 그것이 속한 spec 5.2 도메인(`simulation`,
`observation`, `inference`) 아래로 묶으며, inference 도메인과 actuation
도메인 사이에 걸쳐 있어 어느 쪽에도 깔끔히 속하지 않는 chunk 버퍼를 위해
`control`을 추가한다.

## 4. 왜 일부 항목이 0이나 추측이 아니라 `unavailable`인가

- **`policy_weights`**: `WeightsRef`는 모든 변형(`Safetensors`, `Onnx`,
  `SpirV`)에 대해 `{ path, hash }`다. 바이트 크기 필드를 추가하는 것은 이
  패킷이 선언한 범위(`crates/es-compile/src/budget.rs`와 그 `lib.rs`
  재노출 줄뿐)를 벗어나는 스키마 변경이다; 이 수치를 원하는 향후 패킷은
  체크포인트 이름으로부터 추정하는 대신 `WeightsRef`를 넓혀야 한다.
- **`physics_state` / `render_tile_atlas`**: 호출자가 `ModelSizes`를
  넘기지 않았을 때 / Observation IR에 `ImageInput` 노드가 없을 때
  `None`이다 — 카메라가 없는 태스크는 정당하게 render 예산이 없으며, 이
  모델은 자신에게 알려진 적 없는 physics backend의 크기를 셈할 방법이
  없다.
- **`observation_intermediates`**: Observation IR이 CPU plan으로
  컴파일되지 않을 때(`CpuPlan::compile`이 diagnostics를 반환) `None`이다
  — 예산 모델은 컴파일러 자신의 검증을 중복하지 않고 그 출력을 재사용한다.

## 5. 검사되는 규칙 (spec 20.3)

`MemoryReport::violations(&BudgetInputs) -> Vec<BudgetViolation>`:

- `obs_batch_le_sim_batch` — `n_obs_envs > n_sim_envs`(spec 5.2:
  observation 배치는 simulation 배치에서 뽑히며 그보다 클 수 없다).
- `total_le_device_minus_reserve` — `BudgetInputs::device_bytes`가
  주어졌을 때만 검사하며 `total_bytes <= device_bytes - reserve`를
  확인한다. reserve는 드라이버/OS 오버헤드를 위한 고정 512 MiB
  자리표시자다(그 상수에 `ponytail` 주석이 붙어 있다): CI에서 실제
  디바이스를 쓸 수 있게 되면 그것에 맞춰 보정이 필요하며, 이것이 바로 이
  모델 전체가 기다리고 있는 ±10% 게이트다.
- `tile_atlas_within_max_image_dimension_2d` — `TileAtlasCfg`가 주어졌을
  때만 검사하며, packed atlas가 spec 15.2의 `maxImageDimension2D`(대상
  데스크톱 GPU에서 16384)에 들어맞는지 확인한다.

spec 20.3는 또한 자동 `N_obs`/`N_inf` 축소 루프와 에디터의 컴파일 타임
에러(그 "Memory Plan", spec 11.1)도 요구한다. 둘 다 여기서는 구현되지
않는다: 둘 다 이 모델이 `es-env`/`es-editor`(layer 9와 12)에 연결되어야
하며, 이는 다른 패킷들의 범위다.

## 6. ±10% 게이트(spec 28.4, spec 28.7 gate 13)에 아직 필요한 것

1. spec 20.1의 시나리오들을 돌려 동일한 `BudgetInputs`에 대한 이 모델의
   출력과 비교할 GPU(또는 캡처된 `nvidia-smi`/`vkQueryMemoryHeap`
   트레이스).
2. 그 트레이스에 맞춰 `DEFAULT_RESERVE_BYTES`와 render tile atlas의
   암묵적 가정(실제 texture format에서의 드라이버 패딩/정렬)을 보정하는
   것.
3. `es bench --memory-report`(또는 그 후속)를 실제 backend/session
   *도중에* 실행되도록 연결해, 오늘 이 CLI 패킷이 가진 `--scene`뿐인
   best-effort 경로 대신 `model: Some(ModelSizes)`와 실제 `device_bytes`를
   쓸 수 있게 하는 것.

그때까지 이 모델이 출력하는 모든 수치는 프로젝트 컨벤션(`CLAUDE.md`)에
따라 `Target / Status: 미검증`이다: 감사할 수 있는 공식일 뿐, 이미 이루어진
측정이 아니다.
