<!-- Korean translation of python/es/README.md. The English file is the working copy; regenerate this when it changes. -->

# `es` — Python 저작(authoring) 빌더 (spec 14.2)

`es_native` pyo3 확장(`crates/es-py`) 위의 얇은 Python 프론트엔드이며, 그 확장은 다시
언어 중립적인 Rust 빌더 코어를 호출한다. 레이어링에 대해서는 `docs/design/python-builder.md`
참고.

## Rust 쪽이 정본(canonical)이다

`builder.py`의 몇몇 작은 공식들은 `es_native`를 호출하는 대신 로컬에서 Rust 로직을
재구현한다. 노드가 아직 만들어지기 전에 클라이언트 측 타입 추론을 위해 그 값이 필요하기
때문이다(에디터는 그래프를 한 번 실행하기도 전에 포트의 shape를 알아야 한다):

- `stable_id(path)`는 `es_core::StableId::from_path`(`crates/es-core/src/id.rs`)를
  미러링한다: `blake3(path)`를 16바이트로 자르고 hex로 인코딩한 것이다.
- `_feature_ty(dim, tokens)`와 `_chunk_ty(horizon, action_dim)`은
  `crates/es-ir/src/learning.rs`의 `feature()`/`chunk()`를 미러링한다.

**세 가지 모두에 대해 Rust가 진실의 원천이다.** 여기의 값이 Rust 구현과 어긋나면 Rust
구현이 옳고 고쳐야 할 쪽은 이 파일이다. `python/es/selfcheck.py`는 함수마다 골든 벡터
하나씩을 고정하며, 이는 `crates/es-core/src/id.rs`의 `from_path_is_stable_and_distinct`
테스트에 고정된 리터럴과 일치한다. 따라서 어느 쪽이든 어긋나면 잘못된 바디에 대해 검증되는
IR을 조용히 만드는 대신 요란하게 실패한다 (M4 리뷰 S-14).

## 셀프 체크 실행

```
maturin develop --release --features python   # from this directory, once, to build es_native
PYTHONPATH=python <venv>/Scripts/python.exe -m es.selfcheck
```

## `es_native.Rollout` — Python에서 런타임을 스텝하기 (M8 S4a)

확장이 나르는 두 번째 것이며 위 빌더와는 무관하다: rollout 바인딩. 트레이너
(`train_ppo.py`, S4b)가 자기만의 시뮬레이터가 아니라 **우리** `Env`와 **우리**
Safety Plane을 스텝하게 한다 (`docs/design/rl-continuation.md` 규칙 2). 샘플된 모든
액션은 액추에이터에 닿기 전에 `SafetyPlane::validate`를 통과하며, 이를 우회하는 경로는
없다 (`INV-12`).

```python
from es import es_native

roll = es_native.Rollout(task_toml, observation_toml, deployment_toml, scene_xml, seed=7, n_envs=2)
roll.model()                      # nq, nv, nu, n_envs, actuator/joint 이름, ctrlrange
obs = roll.observe()              # {port: [n_envs * dim]} — Observation IR의 출력 포트
executed, events, rewards, dones = roll.act(actions)   # actions: [n_envs * nu], 액추에이터 단위
roll.reset([0])                   # env 하나; None이면 전체
roll.tick(), roll.metrics()       # 제어 틱; spec 12.4의 아홉 필드, 미측정은 None
```

네 문서는 경로가 아니라 **텍스트**로 전달된다. 경계는 양방향으로 리스트를 나른다 —
Rust 크레이트에 numpy는 없고, 변환은 트레이너 쪽에서 한다. `events`는 env별
`EventSet::bits()`이므로, 실행된 액션이 샘플된 액션과 얼마나 자주 달라지는지 트레이너가
측정할 수 있다. 설계 노트의 메서드 표는 `docs/design/python-builder.md`의
"The rollout binding"에 있다.

MuJoCo 참조 백엔드를 별도 프로세스로 실행하므로, 그 프로세스의 인터프리터에는 `mujoco`
패키지가 필요하다 — `ES_PYTHON`이 이를 지정한다(이 인터프리터일 필요는 없다). 오라클 쌍:

```
ES_PYTHON=<venv>/bin/python cargo test -p es-py rollout_matches_es_eval_loop -- --ignored
PYTHONPATH=python ES_PYTHON=<venv>/bin/python <venv>/bin/python -m es.selfcheck --env
```

둘 다 같은 스크립트된 제어 100 제어 스텝을 구동해
`tests/golden/rollout/so101_100steps.json`과 비교한다 — 첫 번째는 추가로,
`es_eval::runner::run_episode`가 하는 순서대로 손으로 쓴 `Env` + `CpuPlan` +
`SafetyPlane` 루프와도 비교한다. `--env`는 `RAN ...` 또는 `SKIP <reason>`을 출력하며,
확장이 빌드되지 않았거나 인터프리터에 `mujoco`가 없으면 실패가 아니라 스킵한다.

## `encode_video.py`

위 빌더와는 무관하다: `encode_video.py`는 `es video mosaic`가 만든 raw 프레임 출력을
`.mp4`로 변환하는 독립 스크립트다 (M5 V4, 설계 노트 `docs/design/visible-learning.md`
2.9절, 9절). `cv2.VideoWriter`를 쓰며 fourcc는 `mp4v`다 — 오라클 서버에는 `ffmpeg`
바이너리가 없고 이 코덱만 열린다. 이 패킷에서 유일한 Python 단계이며, `es video mosaic`
자체는 순수 Rust다. `opencv-python`이 설치된 Python이 필요하다:

```
<venv>/bin/python python/es/encode_video.py --frames <mosaic dir> --out demo.mp4 --fps 10
```

## `train_act.py`

이것도 위 빌더와는 무관하다: `train_act.py`는 스펙 2.3 학습 분할의 옵티마이저 쪽이다
(M5 V2/V2b, 설계 노트 `docs/design/visible-learning.md` 6절, 7.6절, 7.9절). 그 패킷에서
**유일한** Python이다 — `es policy lower`, `es dataset bake`, `es policy pack`은 Rust이고
인터프리터가 필요 없다.

```
es policy lower --policy untrained.esb --out build/
es dataset bake --policy untrained.esb --out baked/ --frames tiles/ ds/
<venv>/bin/python python/es/train_act.py --module build/ --baked baked/ --out model.safetensors \
    [--epochs N] [--batch N] [--lr F] [--seed N] [--device cuda] \
    [--checkpoint-at 1000,5000,20000] [--loss-curve curve.json] \
    [--resident-gpu] [--amp bf16] [--compile]
es policy pack --policy untrained.esb --weights model.safetensors --out trained.esb
```

`es policy lower`가 쓴 `build/es_policy.py` — 번들 자신의 `LearningGraph`로부터
`es_policy::lower::lower_to_torch`가 생성한 모듈 — 을 `exec`하고, 그것만 최적화한다.
**레이어를 하나도 정의하지 않는다**: 아키텍처는 Learning IR에서 오거나 아예 오지 않는다.
이것이 스펙 1.4의 "같은 IR을 PyTorch로 돌린 것이 ground truth"를 근사적으로가 아니라
문자 그대로 참으로 만든다. `crates/es-policy/tests/ir_training.rs`가 양쪽을 모두 검사한다
(로워링과의 바이트 일치 검사, 그리고 이 파일에 대한 소스 스캔).

`build/contract.json`이 선언한 그대로의 키로 safetensors를 쓰므로, `es policy pack`이
번들에 받아들이기 전에 모든 키와 shape을 검사할 수 있다 (스펙 25.1). 이 경로 어디에서도
로드 시 코드를 실행할 수 있는 포맷은 읽지도 쓰지도 않는다 (`INV-16`).

알아둬야 할 세 가지, 모두 설계 노트에 기록되어 있다:

- **Observation IR 노드를 하나도 구현하지 않는다.** `--baked`는 `es dataset bake`의 출력이고,
  그 명령은 기록된 모든 프레임을 `es eval run`이 추론에서 돌리는 것과 같은 `CpuPlan`으로
  통과시켰다. V2는 parquet을 `pyarrow`로 읽고 여기서 `Op::Dequantize`를 다시 구현했으며,
  Observation IR이 정규화하는 상태 포트에 원본 `observation.state` 행을 먹였다 — 그래서 정책은
  한 관측으로 학습하고 다른 관측으로 평가받았다 (7.9절, 열린 질문 11). 이 파일이 이제 포맷
  하나만 읽는 이유다;
- 구운 텐서가 없는 contract 입력은 0으로 채워진 포트가 아니라 거부다. V2가 경고해야 했던
  "조용히 0" 실패 모드가 은퇴한다;
- 로워링된 모듈은 single-sample이므로, `--batch N`은 배치 forward 한 번이 아니라 N개 샘플을
  한 optimizer step으로 누적한다.

속도 플래그 셋(M5 V5, 설계 노트 섹션 7.11)은 숫자를 움직이느냐로 갈린다. `--resident-gpu`는 베이크된
세트를 샘플마다가 아니라 한 번에 `--device`로 올리며 같은 `--seed`에서 기본 경로와 **비트 단위로
동일**하다. `--amp bf16`과 `--compile`은 비트를 바꾸고, 그래서 옵트인이다. `--batch`의 기본값이 8인
이유는 설계 노트의 측정 실행이 8이기 때문이다. 올릴 때는 `--lr`을 선형으로 같이 올린다
(`--batch 32 --lr 4e-4`).

`torch`와 `torchvision`이 설치된 Python이 필요하다. `pyarrow`는 더 이상 여기서 읽지 않는다 —
데이터셋 읽기는 `es dataset bake`가 Rust로 한다.
