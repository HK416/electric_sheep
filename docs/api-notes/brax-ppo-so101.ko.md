<!-- Korean translation of docs/api-notes/brax-ppo-so101.md. The English file is the working copy; regenerate this when it changes. -->

# 우리 SO-101 씬 위의 brax PPO — S2b가 임포트할 소스 정책

패킷: `docs/packets/M8/S2c-source-policy.md`. 과제 정의: `docs/design/rl-continuation.md`
5절(2026-09-21에 소유자가 확정한 26차원 관측 포함). 파라미터 포맷의 선례:
`docs/api-notes/mujoco-playground-quadruped.md` 2절. 영어 원본: `brax-ppo-so101.md`.

그 사족보행 노트와 달리 **아래 내용은 전부 우리가 직접 실행했다**. 모든 수치는
`measured (server, date, path)` — 서버 `renderer-14`(RTX 4090 24 GB, 드라이버 610.57.04,
16 코어), 2026-09-21, 산출물은 `~/artifacts/plan-s/s2c/` 아래. 처리량 수치 전체에 붙는 단서가
하나 있다: 작업 내내 GPU를 다른 작업(M7 R5 홀드아웃 평가, `es eval run` 6 샤드)과 **공유**했고,
세션 중 측정한 float32 참조 matmul이 **2.1 TFLOP/s**였다(혼자 쓰면 훨씬 높은 카드다). 따라서
모든 처리량은 4090의 능력이 아니라 경합 상태에서 측정한 하한으로 읽어야 한다.

## 1. 고정 버전, 그리고 하나의 충돌

| 패키지 | 고정 | 실제 해결된 버전 |
|---|---|---|
| `playground`(`mujoco_playground`) | **0.2.0** | 0.2.0 |
| `brax` | **0.14.2** | 0.14.2 |
| `jax` / `jaxlib` / `jax-cuda12-{plugin,pjrt}` | **0.9.0**(아래 참조) | 0.9.0 |
| `mujoco` / `mujoco-mjx` | — | 3.13.0 / 3.13.0 |
| `flax` / `optax` / `orbax-checkpoint` | — | 0.12.9 / 0.2.8 / 0.12.4 |
| `numpy` / `blake3` / `jaxopt` / `warp-lang` | — | 2.5.3 / 1.0.9 / 0.8.5 / 1.17.0 |
| Python | 3.12 | 3.12.14 (`~/venvs/es-rl`, uv 0.12.13) |

**충돌과 해결.** `playground` 0.2.0과 `brax` 0.14.2 모두 `jax`에 상한을 선언하지 않아 `uv`는
최신 **0.11.2**를 설치했다. 그런데 `brax` 0.14.2는 JAX가 제거한
`jax.device_put_replicated`(`brax/training/agents/ppo/train.py:756`)를 호출한다:

```
AttributeError: jax.device_put_replicated is deprecated; use jax.device_put instead.
```

측정 결과: 이 속성은 **jax 0.10.0에는 없고**, **0.9.0과 0.8.0에는 있다**. 따라서 venv는
고정된 brax가 동작하는 최신 JAX인 `jax[cuda12]==0.9.0`을 고정한다. `mujoco-mjx` 3.13.0은
그 위에서 그대로 동작한다. venv를 재현하는 사람은 반드시 JAX를 고정해야 한다 —
`jax[cuda12]`를 고정 없이 설치하면 첫 학습 스텝에서 위 오류가 그대로 재현된다.

```
uv venv --python 3.12 ~/venvs/es-rl
uv pip install --python ~/venvs/es-rl/bin/python "jax[cuda12]==0.9.0" mujoco mujoco-mjx \
    brax==0.14.2 playground==0.2.0 orbax-checkpoint numpy blake3
```

## 2. 구현된 과제

`python/es/rl_source/so101_reach_env.py`는 커밋된 씬 위의 `mjx_env.MjxEnv`다:

* **제어** 200 Hz 물리 위의 50 Hz(`ctrl_dt = 0.02`, `sim_dt = 0.005`, `n_substeps = 4`);
* **관측 26차원** — `joint_pos[6] ‖ joint_vel[6] ‖ cube_pose[7] ‖ gripper_pose[7]`, 각 pose는
  월드 프레임 `pos[3] ‖ quat[4]`이고 쿼터니언은 **xyzw** 순서(§3.1; MuJoCo의 `xquat`는 wxyz라
  재정렬한다). `gripper`는 사이트가 아니라 **바디**다;
* **행동 6차원** — 정규화된 위치 목표, 액추에이터 `ctrlrange`별로
  `ctrl = offset + scale · clip(a, −1, 1)`, `offset = (hi + lo)/2`, `scale = (hi − lo)/2`;
* **보상** `−dist`, 성공 시 `+1`, `dist = ‖cube_pos − gripper_body_pos‖`(위치만);
* **성공** `dist < 0.03` m; **에피소드** 200 제어 스텝, 조기 종료 없음;
* **랜덤화** 에피소드마다, 커밋된 Task IR 노드를 그대로 재현
  (`tests/fixtures/visible-learning/task.toml` 노드 28–30): 큐브 `x ~ U(0.21, 0.27)`,
  `y ~ U(−0.03, 0.05)`, `z = 0.02`, 모든 팔 관절 0으로 리셋, `qvel = 0`.

패킷 문구와의 의도적 차이 두 가지:

| # | 차이 | 이유 |
|---|---|---|
| D1 | 큐브를 에피소드마다가 아니라 **환경마다 한 번** 뽑는다(Playground 기본 `BraxAutoResetWrapper`, `full_reset=False`) | `full_reset=True`는 모든 스텝에서 모든 환경에 대해 `env.reset`(즉 전체 `mjx.forward`)을 호출한다. 환경 4,096개 · 2 M 스텝이면 환경당 약 2.4 에피소드이므로 실질적으로 같은 분포에서 4,096개의 독립 표본이고, 재추첨은 경합 중인 GPU에서 스텝 시간의 큰 몫을 대가로 몇 %의 다양성만 얻는다. `eval_brax_so101.py`는 에피소드마다 새로 뽑는다. |
| D2 | 성공을 **두 가지로** 보고한다 — `success_reached`(에피소드 중 한 번이라도 0.03 m 이내)와 `success_final`(마지막 스텝이 이내) | 조기 종료가 없으므로 "에피소드가 성공했다"는 해석이 필요하다. 둘 다 `eval.json`에 있고, 아래 대표 수치는 `success_reached`이며 최종 정책에서는 두 값이 같다. |

## 3. 커밋된 씬의 MJX 호환성 — 측정값

**수정 없이 로드되고 스텝된다.** `tests/fixtures/mjcf/so101_pick_place.xml`에 대한
`mjx.put_model`은 `mujoco-mjx` 3.13.0에서 성공한다(`Impl.JAX`, 1.4 s):
`integrator="implicitfast"`, `cone="elliptic"`, `condim` 3과 6, 관절 `frictionloss`,
`impratio=10`, `<position>` 액추에이터 전부 받아들여진다. 거부하는 것은 없다. 다만 **학습이
불가능**하다:

| 측정(별도 표기 없으면 환경 1개) | 커밋된 씬 | 파생 씬(§3.2) |
|---|---|---|
| 정적 할당 | **접촉 1,054개, `efc_J` 5,292 × 12** | **접촉 24개, `efc_J` 84 × 12** |
| `jit(reset)` / `jit(step)` 컴파일 | 45.4 s / 125.3 s | 15.8 s / 14.9 s |
| 200스텝 에피소드 1회(파이썬 루프) | **45분 내 미완료** | 3.8 s (52.7 제어 스텝/s) |
| 배치 4,096 환경 | 574 env-steps/s(CG; Newton은 OOM) | **10,620 env-steps/s** |
| 배치 8,192 환경 | 4.10 GiB 단일 할당 실패 | 불필요 |

원인은 MJX의 기능 부재가 아니라 `njmax`다. MuJoCo 기본 **Newton** 솔버는 환경마다 밀집
`njmax × njmax` 제약 행렬을 만들고 분해한다. `njmax = 5,292`면 환경당 112 MB이고, 팔 접촉쌍을
제외해 `njmax = 360`(환경당 518 KB)으로 줄여도 8,192 환경은 4.10 GiB를 한 번에 요구하며
736 env-steps/s에 머문다.

시도했다가 기각한 두 가지:

* **`solver="CG"`**(행렬 없이 동작, 접촉 72개에서 3.5배 빠름: 4,096 환경 2,548 env-steps/s)는
  **발산한다**. 그리퍼-큐브 접촉쌍이 남아 있으면, PPO가 시작하는 지점인 무작위 전범위 정책이
  큐브 자유관절을 **제어 스텝 11회** 만에 NaN으로 보낸다(`iterations=4 ls_iterations=8`;
  10/20에서는 60회, 10/50에서는 40회). 같은 200스텝을 Newton 10/20은 견딘다. CG는 탈락.
* **float64**(`jax_enable_x64`)로도 구제되지 않는다: 17스텝에서 NaN, 직전 큐브 자유관절 속도가
  1e52다. 발산의 원인은 반올림이 아니라 접촉 충격량이다 — 팔이 휘둘릴 때 턱의 0.75 mm 구
  geom(`condim=6`, `solref="0.01 1"`)이 5 ms 서브스텝 하나 만에 25 mm 큐브를 관통한다.

### 3.1 같은 씬에서 MJX 대 MuJoCo CPU

동일한 시작 상태, 동일한 제어 시퀀스로 200스텝 에피소드 하나를 MJX float32와 MuJoCo CPU
float64로 각각 실행(`python so101_reach_env.py --xml so101_reach_mjx.xml`):

| 경과 | `qpos` 최대 절대 차 |
|---|---|
| 제어 1스텝 | 6.055e-03 |
| 제어 10스텝 | 9.868e-02 |
| 제어 50 / 200스텝 | 1.619e-01 |

에피소드 전체에서 관측 채널의 최대 차는 3.468(`joint_vel` 채널, rad/s)이다. 이는 `kp = 998`
위치 서보에 무작위 전범위 목표를 넣은 개루프 궤적의 혼돈적 발산이며, 폐루프 롤아웃이나 정책의
거동을 한정하지 않는다. §3.5 티어 3이 허용하는 통상적인 float32-float64 격차이지 결함이 아니다.

### 3.2 파생 씬과 그 안의 모든 편집

`python/es/rl_source/so101_reach_mjx.xml`은 커밋된 씬을 **include**하고 요소 하나만 더한다:

```xml
<mujoco model="so101_reach_mjx">
  <include file="../../../tests/fixtures/mjcf/so101_pick_place.xml"/>
  <contact> … <exclude body1= body2= /> 35줄 … </contact>
</mujoco>
```

그 외에는 없다. **`<option>` 재정의도 없다**: `integrator="implicitfast"`, `cone="elliptic"`,
Newton 솔버, `iterations=10`, `ls_iterations=50`, `timestep=0.005`, `impratio=10`이 모두 커밋된
값이고, 모든 바디·geom·관절·액추에이터·사이트·카메라도 구조상 커밋된 것 그대로다 — 이 파일은
우리 런타임이 읽는 씬에서 드리프트할 수 없다.

| # | 편집 | 제거되는 것 | S4c 격차 |
|---|---|---|---|
| E1 | 팔 링크 대 팔 링크 `<exclude>` 21줄 | 인접하지 않은 자기충돌(shoulder–wrist, upper_arm–gripper 등) | 소스 환경에서 팔이 자기 자신을 통과할 수 있다 |
| E2 | 팔 링크 대 `world` 7줄 | 테이블 평면과 통 geom 5개에 대한 팔의 접촉 | 팔이 테이블과 통을 통과할 수 있다 |
| E3 | 그리퍼가 아닌 링크 대 `cube` 5줄 | 팔꿈치·손목·카메라 마운트가 큐브를 치는 것 | 큐브가 팔 몸통에 밀려나지 않는다 |
| E4 | `gripper`·`moving_jaw_so101_v1` 대 `cube` 2줄 | **턱이 큐브에 닿는 것** | 큐브는 물체가 아니라 목표 위치다: 움직이지 않고, 그리퍼는 통과한다 |

E4는 두 번 읽어야 한다. 이것이 실행 자체를 가능하게 만든 편집이고(§3의 NaN), 여기서 학습한
정책은 큐브를 한 번도 느껴본 적이 없다. S4c가 커밋된 씬의 우리 런타임에서 이 정책을 재생하면
큐브와의 접촉은 처음 겪는 일이고, 큐브가 밀려나는 것은 버그가 아니라 예상된 결과다. 네 편집이
학습된 정책에 실제로 얼마를 물리는지는 5.5절에서 측정했다 — 답은 "전부"이며, 그것이 플랜 S의
지속 학습이 고치려는 정직한 출발점이다.

**편집 비용의 측정(분리 가능한 CPU에서):** 동일 상태에서 동일한 무작위 제어 200스텝으로
*편집된* 모델과 *커밋된* 모델을 MuJoCo CPU로 각각 돌리면 `max|Δqpos| = 0.000e+00` — 비트 단위로
같다. 그 궤적에서는 제외된 접촉쌍이 한 번도 닿지 않았으므로 편집의 비용이 정확히 0이었다는
뜻이고, 편집은 팔이 실제로 테이블·자기 자신·큐브에 닿을 때만 효력을 갖는다. 반대로 얻은 것은
접촉 수 1,054 → 24, 그리고 574 → 10,620 env-steps/s다.

## 4. 레시피

서버에는 체크아웃이 없다(저장소 규칙: `git clone` 금지, `scp`만). 스크립트가 필요로 하는 두
디렉터리를 저장소의 상대 배치 그대로 복사한다 — `so101_reach_env.py`가 씬을
`../../../tests/fixtures/mjcf/so101_pick_place.xml`로 찾고, 파생 XML도 같은 경로로
`include`하기 때문이다:

```bash
scp -r python/es/rl_source LJM@192.168.100.14:~/Projects/es-s2c/python/es/
scp tests/fixtures/mjcf/so101_pick_place.xml \
    LJM@192.168.100.14:~/Projects/es-s2c/tests/fixtures/mjcf/
```

```bash
ssh LJM@192.168.100.14
cd ~/Projects/es-s2c/python/es/rl_source
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.3
export XLA_FLAGS=--xla_gpu_deterministic_ops=true

# 0. 씬 자체 점검: MJX 대 MuJoCo CPU, 그리고 후보 XML의 처리량
python so101_reach_env.py --xml so101_reach_mjx.xml
python so101_reach_env.py --xml so101_reach_mjx.xml --bench 4096

# 1. 학습 (기본값: --xml so101_reach_mjx.xml, 2 M 스텝, 4,096 환경)
python train_brax_so101.py --seed 0 --timesteps 2000000 --num-envs 4096 \
    --out ~/artifacts/plan-s/s2c/seed0-run1

# 2. 두 오라클
python check_export.py --out ~/artifacts/plan-s/s2c/seed0-run1
python eval_brax_so101.py --out ~/artifacts/plan-s/s2c/seed0-run1 --episodes 64
```

PPO 설정(`train_brax_so101.py`, Playground의 `PandaPickCube` 매니퓰레이션 파라미터 스타일에
패킷이 요구한 더 넓은 네트워크):

`policy`·`value` MLP **(256, 256)**, `activation=swish`, `normalize_observations=True`,
`unroll_length=10`, `num_minibatches=32`, `num_updates_per_batch=8`, `batch_size=256`,
`discounting=0.97`, `learning_rate=1e-3`, `entropy_cost=2e-2`, `max_grad_norm=1.0`,
`reward_scaling=1.0`, `episode_length=200`, `num_evals=10`, `num_eval_envs=64`,
`deterministic_eval=True`.

## 5. 측정값

모든 행: 서버 `renderer-14`, 2026-09-21, 작업 내내 GPU를 M7 R5 홀드아웃 평가와 공유.

### 5.1 학습 — `~/artifacts/plan-s/s2c/seed0-run1/`

2,000,000 스텝, 시드 0, 4,096 환경, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`,
**wall clock 540.6 s**(9.0분), 같은 카드에서 두 번째 결정성 실행이 동시에 학습 중이었다. 앞선
동일 설정 한 쌍은 464.0 s였다. 실행당 GPU 메모리 최대 약 1.2 GB.

| 스텝 | 보상(에피소드 합) | 200스텝 중 0.03 m 이내 스텝 수 | Σ 거리 |
|---|---|---|---|
| 0 | −55.74 | 0.00 | 55.74 |
| 245,760 | −66.52 | 0.00 | 66.52 |
| 491,520 | −14.60 | 2.20 | 16.80 |
| 737,280 | 184.27 | 188.36 | 4.09 |
| 1,228,800 | 187.25 | 189.50 | 2.25 |
| 1,474,560 | 56.31 | 94.86 | 38.55 |
| 2,211,840 | **188.15** | **190.95** | 2.80 |

과제는 약 74만 스텝에서 풀린다. 1.47 M의 하락은 통상적인 PPO 흔들림이고 회복한다. 40만 스텝
시험 실행도 이미 성공률 1.00으로 평가됐으므로 2 M 예산은 빠듯하지 않고 여유 있는 쪽이다
(`curve.json`에 전체 행이 있다).

### 5.2 소스 프레임워크에서의 평가 — `eval.json`

`eval_brax_so101.py --episodes 64`, 결정적 정책을 체크포인트가 아니라 **`source.npz`에서**
재구성해(즉 S2b가 임포트할 파일을 채점) 에피소드마다 큐브를 새로 뽑았다:

| 지표 | 측정값 |
|---|---|
| `success_reached`(한 번이라도 0.03 m 이내) | **1.00**(64/64) — 목표 0.8 |
| `success_final`(마지막 스텝 이내) | **1.00** |
| 최종 거리 평균 / 최대 | 8.41 mm / 14.65 mm |
| 리턴 평균 | 188.11 |

### 5.3 결정성

같은 시드·같은 플래그로 두 번 실행, `source.npz`가 **비트 단위로 동일**:

```
blake3 8c0faf01c4d0e4193815cac4af47262a3daa623d1202d75b2873dfe3bf2283b3   seed0-run1
blake3 8c0faf01c4d0e4193815cac4af47262a3daa623d1202d75b2873dfe3bf2283b3   seed0-run2
```

같은 설정으로 앞서 실행한 한 쌍에서도 동일한 해시가 나왔다 — 즉 네 번의 실행이 일치한다. 두
실행이 동시에 돌았고 GPU가 경합 상태였으므로, 이 재현성은 한가한 장비 덕분이 아니다.

### 5.4 `check_export.py` — numpy 대 JAX

| obs 집합 | 최대 절대 오차 | 비고 |
|---|---|---|
| `obs_scaled`(분포 내) | **1.580e-06** | 게이트, 허용 1e-5 — 통과 |
| `obs`(`U(−1, 1)`, 패킷이 지정한 추첨) | 2.287e-05 | 행의 80.2 %에서 tanh 포화, §6 참조 |

S2b가 반드시 알아야 할 측정 하나: 저장된 행동은
`jax.default_matmul_precision("highest")`에서 계산했다. 이 GPU에서 XLA의 **기본** float32
matmul은 TF32이고, 그 경로는 진짜 float32 matmul과 **1.515e-03**만큼 다르다
(`meta.json → oracle.tf32_delta`) — 하드웨어 경로만으로 티어 허용치의 100배다. TF32로 계산된
행동과 비교하는 임포터는 아무리 정확해도 1e-5를 통과할 수 없다.

### 5.5 §3.2의 sim-to-sim 격차를 정책 자체로 측정

같은 `source.npz`, 같은 64 평가 에피소드, 세 가지 씬 — S4c가 어차피 어렵게 발견할 수치다:

| 정책을 평가한 씬 | 접촉 수 | `success_reached` | 최종 거리 평균 |
|---|---|---|---|
| 파생 씬(E1–E4 제외) — 학습한 곳 | 24 | **1.00** | 8.4 mm |
| 파생 씬에서 팔-`world` 행만 되살림(E2 복원) | 552 | **0.00** | 71.7 mm |
| 커밋된 씬, 아무것도 제외하지 않음 | 1,054 | **0.00** | 173.9 mm |

두 실행 모두 발산하지 않았다(NaN 없음, 리턴도 정상). 즉 수치 문제가 아니라 거동이다:
**학습된 정책은 테이블과 자기 자신을 통과해서 큐브에 도달한다.** 접촉을 되돌리면 테이블만으로도
7 cm, 전부면 17 cm 못 미친 곳에서 멈춘다. 오차의 대략 절반이 E2(테이블과 통)이고 나머지가
E1/E3/E4다.

이것이 소스 정책의 정직한 상태다: 자기 환경을 푸는, 재현 가능하고 정확히 임포트 가능한 함수이며,
우리 환경에서 동작하는 정책은 **아니다**. 플랜 S에는 오히려 맞는 출발점이다 — 이걸 고치는 것이
바로 "우리 시뮬레이터에서의 RL 지속 학습"의 목적이고, S4c 표의 before/after에 진짜 "before"가
생겼다. 다만 S2b는 임포트한 정책이 커밋된 씬에서 점수를 낼 것이라 기대하면 안 되고, S4b는 첫
지속 학습 구간이 fine-tune이 아니라 재학습에 가까울 것을 예상해야 한다.
(`~/artifacts/plan-s/s2c/seed0-run1-committed-scene/eval.json`, `…-keep-world/eval.json`.)

### 5.6 경로와 해시

| 무엇 | 위치(서버 `renderer-14`) |
|---|---|
| 산출물 | `~/artifacts/plan-s/s2c/seed0-run1/`, `…/seed0-run2/` |
| orbax 체크포인트 | `…/seed0-run1/checkpoints/000002211840/`(체크포인트 9개, 합계 5.5 MB) |
| 로그·벤치·NaN 프로브 | `~/artifacts/plan-s/s2c/logs/*.log` |
| venv | `~/venvs/es-rl`(유지) |

| 파일 | blake3 |
|---|---|
| `source.npz`(두 실행 모두) | `8c0faf01c4d0e4193815cac4af47262a3daa623d1202d75b2873dfe3bf2283b3` |
| `python/es/rl_source/so101_reach_mjx.xml` | `d193fb75b07f7798d410f024a027d9b70217eef0c5f5042e6b70862ee62460da` |
| `tests/fixtures/mjcf/so101_pick_place.xml` | `94d7fa5fa07e3d0ba23774151bc6cb12339f0882a2e3480cf18e9577043f86ad` |

## 6. S2b가 읽을 익스포트 포맷

`source.npz` — 전부 float32, brax의 저장 규약(`kernel`이 `[in, out]`, 오른쪽 곱; torch의
`[out, in]`로 쓰려면 전치):

| 키 | 모양 | 의미 |
|---|---|---|
| `obs_mean`, `obs_std` | `[26]` | `running_statistics`; `(x − mean) / std` 적용, **클리핑 없음** |
| `kernel_0`, `bias_0` | `[26, 256]`, `[256]` | 은닉 Dense 0, 이후 `swish` |
| `kernel_1`, `bias_1` | `[256, 256]`, `[256]` | 은닉 Dense 1, 이후 `swish` |
| `mean_kernel`, `mean_bias` | `[256, 6]`, `[6]` | 출력 Dense 앞 절반 — brax의 `loc` |
| `logstd_kernel`, `logstd_bias` | `[256, 6]`, `[6]` | 출력 Dense 뒤 절반 |

S2b가 재현해야 하는 결정적 정책의 전부는 다음과 같다:

```
x = (obs - obs_mean) / obs_std
x = swish(x @ kernel_0 + bias_0)
x = swish(x @ kernel_1 + bias_1)
a = tanh(x @ mean_kernel + mean_bias)
```

틀리기 쉬운 사실 셋, 각각 고정된 상류 소스에서 직접 읽었다:

1. **출력 Dense는 선형이다.** `brax.training.networks.MLP`는 `hidden_0 … hidden_n`을 만들고
   마지막을 *제외한* 모든 층에 활성함수를 적용한다(`activate_final=False`, `networks.py:161`).
   따라서 패킷 초안의 `meta.json` 필드 `activate_output = true`는 **`false`**로 기록했다.
   출력의 유일한 비선형은 분포의 `tanh`(`squash`), 즉 `NormalTanhDistribution.mode = tanh(loc)`다.
2. **출력 뒤 절반은 log-std가 아니다.** brax는 이를 `std = softplus(x) + 0.001`로 표준편차로
   바꾼다(`distribution.py:171`). `exp(x)`가 아니다. 추론에는 쓰이지 않으며, S1의
   `--init-log-std`는 저장된 수를 로그로 읽지 말고 이 공식을 적용한 뒤 로그를 취해야 한다.
3. **`obs_std`가 brax의 1e-6 하한일 수 있다.** 이 과제의 세 채널은 상수다 — 큐브 `z`와 큐브
   쿼터니언 성분 둘, E4로 큐브가 건드려지지 않기 때문이다 — 그래서 누적 분산이 0이고
   `running_statistics.update`가 std를 `std_min_value = 1e-6`으로 클리핑한다
   (`running_statistics.py:233`). 관측 자체의 스케일이 아닌 입력은 그 채널에서 1e6배로
   증폭된다. `oracle-1000.npz`가 두 집합을 담는 이유가 이것이다.

`meta.json`에는 해결된 버전들, 시드·스텝 수·wall clock, `obs_dim = 26`, `action_dim = 6`,
`hidden`, `activation`, `activate_output`, `squash`, 정규화·log-std 공식, 구간별 이름·시작·길이·
단위를 담은 `obs_layout`(쿼터니언 순서 명시), 관절 순서, 행동 `offset`/`scale`/공식, 사용한 씬과
그 blake3 *및* 커밋된 씬과 그 blake3, 환경 상수, 전체 PPO 설정이 들어 있다.

`oracle-1000.npz`:

| 키 | 추첨 방식 | 용도 |
|---|---|---|
| `obs`, `actions` | 채널별 `U(−1, 1)`, numpy 시드 0 | 패킷이 지정한 추첨, 그대로 유지 |
| `obs_scaled`, `actions_scaled` | `obs_mean + obs_std · U(−1, 1)`, numpy 시드 1 | 같은 추첨을 관측 자체의 스케일에 올린 것 |

`actions*`는 JAX의 결정적 행동(`make_inference_fn(params, deterministic=True)`)이다. 1e-5
티어가 의미를 갖는 것은 **scaled** 집합뿐이다: `U(−1, 1)` 집합에서는 상수 채널이 ~1e6으로
정규화되어 모든 `swish`가 포화하고 행동의 99.9 %가 ±1에 놓이므로, 같은 matmul의 두 float32
구현이 포화하지 않은 소수의 행에서 ~1e-1까지 벌어진다. `check_export.py`는 둘 다 보고하고
scaled 집합으로 게이트한다 — 패킷의 acceptance 문구에서 벗어난 선택이며, 그러지 않으면 어떤
정확한 임포터도 통과할 수 없는 오라클이 되기 때문이다. S2b는 `obs_scaled`와 비교하고 uniform
집합은 포화 프로브로 다루면 된다.

## 7. 델타 환경 — `--action delta` (패킷 M9/T2)

§1–§6은 그대로 유효하다. 고정 버전도, 파생 씬(blake3 `d193fb75…`, 커밋된 씬 `94d7fa5f…`)도,
26차원 관측도, 보상·성공·타임아웃·랜덤화도, PPO 설정도, 익스포트 포맷도 동일하다. 이 절은 그
**하나의** 차이와 그 측정값이다. 서버 `renderer-14`, 2026-09-21 (UTC), venv `~/venvs/es-rl`,
아티팩트 `~/artifacts/plan-t/t2/seed0-run{1,2}/`.

### 7.1 유일한 변경

```
target_t = clip(target_{t−1} + delta_scale · clip(a, −1, 1), ctrl_lo, ctrl_hi)
target_0 = 리셋 자세
```

`delta_scale = 0.05` rad/제어 틱 — 50 Hz에서 2.5 rad/s이므로 배치 IR 엔벨로프의 3.0 rad/s
아래다. 즉 증분 자체는 Safety Plane이 허용하는 범위 안이고, 플레인이 클램프하는 대상은
**적분된** 절대 목표와 위치 한계다.

두 번 읽을 만한 결정 두 가지:

1. **적분기의 상태는 `state.info`의 새 항목이 아니라 `data.ctrl`이다.** Playground의
   `BraxAutoResetWrapper`는 `done`인 곳에서 `first_state.data`를 복원하지만 `info`는 건드리지
   않는다. 따라서 목표를 `info`에 두면 에피소드 경계를 넘어 새어 나간다 — M9/T1이 우리 런타임에
   대해 금지하는 바로 그 버그다. `mjx_env.step`은 명령한 `ctrl`을 `data`에 쓰므로
   `state.data.ctrl`이 곧 직전 명령이고, 리셋이 `target_0`을 공짜로 복원한다. 환경은 두 번째
   상태를 들고 다니지 않으며, `reset`은 `ctrl`을 0이 아니라 리셋 자세(구동되는 여섯 관절의
   `qpos0`, 이 팔에서는 0)로 시드한다.
2. **저장되는 목표는 클립된 값이다.** 이 클립은 플레인 클램프의 소스 쪽 대역이다(T1: 적분기는
   *실행된* 목표에서 전진하며 원시 행을 쓰지 않는다). 그래서 관절 한계를 향해 +1로 미는 정책이
   팔이 도달할 수 없는 목표를 누적했다가 되감아야 하는 일이 없다. 두 모드 모두 목표를 계산하는
   곳은 `SO101Reach.command` 하나뿐이고, `so101_reach_env.py --action delta`가 이를 자체
   점검한다: `+0.5` 증분 하나는 목표를 `0.5 · delta_scale`만큼 움직이고, `+1`의 연속은
   `ctrlrange`에서 멈춘다.

`meta.json`에 `action = { kind = "joint_delta", offset = [0]×6, scale = [0.05]×6,
unit = "rad per control tick", delta_scale = 0.05, formula = …, ctrl_lo, ctrl_hi }`가 추가된다.
`offset`/`scale`의 의미는 §6과 같다 — 임포터가 만드는 `Normalizer{Inverse}` 그 자체다. 따라서
델타 정책에서 역정규화는 `[−1, 1]`을 **증분** ±0.05 rad에 대응시키고 오프셋은 0이다.
`position_target` 블록은 그대로이며 기본값 `--action position`은 §5를 그대로 재현한다.

### 7.2 학습 — `~/artifacts/plan-t/t2/seed0-run1/`

2,000,000 스텝, 시드 0, 4,096 환경, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`,
wall clock **561.3 s**(결정성 확인용 두 번째 런이 같은 카드에서 동시 학습; run 2는 560.3 s).
이번에는 GPU를 독점했고, 16코어 CPU는 다른 에이전트들의 작업과 공유했다(load average 16–22).
첫 평가 행까지 134 s가 걸린 이유가 그것이다.

| 스텝 | 보상(에피소드 합) | 200 중 0.03 m 안에 있던 스텝 | Σ 거리 |
|---|---|---|---|
| 0 | −69.69 | 0.00 | 69.69 |
| 245,760 | −32.46 | 0.00 | 32.46 |
| 491,520 | 121.05 | 132.55 | 11.50 |
| 737,280 | 73.88 | 97.03 | 23.16 |
| 983,040 | 172.84 | 177.06 | 4.23 |
| 1,228,800 | **178.65** | **181.78** | 3.13 |
| 1,474,560 | 165.82 | 170.50 | 4.68 |
| 1,720,320 | 163.82 | 169.05 | 5.23 |
| 1,966,080 | 174.00 | 177.77 | 3.77 |
| 2,211,840 | 121.23 | 132.39 | 11.16 |

익스포트된 정책은 마지막 행이고, 마지막 행은 흔들림이다. **최종** 정확도는 이 런에서 가장
좋고(3.2 mm, §7.3) 잃은 것은 접근 속도다 — 공 안에 머문 스텝이 182가 아니라 132다. 델타 행동은
도약할 수 없다: 리셋 자세에서 큐브까지 어깨 관절로 ~1.2 rad이므로 상한으로도 24틱 이상이고,
그 틱마다 거리 페널티를 문다. 아래 `return_mean`이 §5.2의 위치 정책 188.1이 아니라 126.8인
이유가 이것이며, 이는 학습이 아니라 행동 공간의 성질이다.

### 7.3 소스 프레임워크에서의 평가 — `seed0-run1/eval.json`

`eval_brax_so101.py --episodes 64`, **`source.npz`에서** 재구성한 결정적 정책, 에피소드마다
새로 뽑는 큐브, `--action`은 `meta.json`에서 읽는다:

| 지표 | 델타(이 절) | 위치(§5.2) |
|---|---|---|
| `success_reached`(한 스텝이라도 0.03 m 안) | **1.00** (64/64) — 목표 0.8 | 1.00 |
| `success_final`(마지막 스텝이 안쪽) | **1.00** | 1.00 |
| 최종 거리, 평균 / 최대 | **3.20 mm / 6.72 mm** | 8.41 / 14.65 mm |
| 리턴, 평균 | 126.81 | 188.11 |

틱당 **명령 변화량** `|target_t − target_{t−1}|`, 64 × 200 × 6 값 전체 — M9/T3이 우리 런타임의
클램프 비율을 비교할 분포:

| | 평균 | p95 | 최대 |
|---|---|---|---|
| 관절·틱별 | **0.01006 rad** | **0.03219 rad** | **0.04991 rad** |
| 틱별, 여섯 관절 중 최대 | 0.02090 rad | 0.04303 rad | 0.04991 rad |

최대값은 소수 넷째 자리까지 `delta_scale`이다 — `tanh`가 포화하므로 상한에 도달하되 넘지
않는다 — 그리고 평균은 그 1/5이다. 정책은 에피소드 대부분을 이동이 아니라 큐브 근처에서
정지한 채 보낸다. 이 런에서 `ctrl_lo`/`ctrl_hi`에 걸린 클립은 없었다: 관측된 최대 증분이 곧
행동이 요청할 수 있는 최대치다.

### 7.4 결정성과 `check_export.py`

같은 시드, 같은 플래그, 한 카드에서 동시에 돌린 두 런 — `source.npz` **비트 단위 동일**:

```
blake3 473b4fde657c2029283a2127dc9ba38128d47f4e93d3d0c05aea218d7914bb55   seed0-run1
blake3 473b4fde657c2029283a2127dc9ba38128d47f4e93d3d0c05aea218d7914bb55   seed0-run2
```

`check_export.py --out …/seed0-run1` (numpy 대 JAX, §5.4의 오라클):

| obs 집합 | 최대 절대 오차 | 비고 |
|---|---|---|
| `obs_scaled` (분포 내) | **9.537e-07** | 게이트, 허용 1e-5 — 통과 |
| `obs` (`U(−1, 1)`) | 1.353e-05 | 행의 32.1 %가 tanh 포화(§5.4는 80.2 %) |

이 정책의 `meta.json → oracle.tf32_delta`는 **1.056e-03**이다 — §5.4가 기록한 것과 같은 1e-3
하드웨어 경로 격차이며, 임포터가 `highest` 정밀도의 `actions_scaled`와 비교해야 하는 같은
이유다.

| 무엇 | 위치(서버 `renderer-14`) |
|---|---|
| 아티팩트 | `~/artifacts/plan-t/t2/seed0-run1/`, `…/seed0-run2/` |
| orbax 체크포인트 | `…/seed0-run1/checkpoints/000002211840/` (9개, 5.5 MB) |
| 로그 | `~/artifacts/plan-t/t2/logs/run{1,2}.log` |
| 미러된 소스 | `~/Projects/es-t2/` (패킷 종료 시 삭제) |
