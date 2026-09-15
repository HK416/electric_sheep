# M5 V8 — 외부 ACT를 우리 런타임으로

설계 노트: `docs/design/visible-learning.ko.md` **7.16절**, 그리고 이 패킷이 올라타는 하네스는
7.12–7.15절. 스펙: §1.4, §5.3, §7, §8.1, §8.4, §8.7, §8.9, §9.2, §19.1, §25.1. API 노트:
`docs/api-notes/lerobot-act.ko.md`, `docs/api-notes/lerobot-dataset.ko.md`,
`docs/api-notes/lerobot-config.ko.md`. 의존: V1/V1b/V1c(시연과 v3.0 익스포트), V3(스위트),
V5(`--jobs`), V6/V6b(전문가를 통과시키고 수집과 같은 방식으로 청크를 실행하는 하네스).

## 이 패킷이 답하려는 질문

스펙 §8.1은 약속으로 시작한다. **네트워크 내부는 불투명하고 인터페이스 의미론만 타입이 붙는다** —
"우리는 독자적인 정책 아키텍처를 발명하지 않는다". 플랜 V는 그 약속을 아직 시험한 적이 없다.
V2/V2b가 학습시킨 것은 IR *모양*의 ACT였다. `VisionEncoder{ResNet18, pretrained=false}` →
`Fusion{Concat}` → `TemporalEncoder{Transformer}` → `PolicyHead{Regression}` — CVAE도, DETR
디코더도, 사전학습 백본도, 카메라별 토큰 격자도 없다. 스펙 §8.3의 노드 파라미터가 폭 하나만
싣기 때문이다. 그 정책은 하네스를 전부 고친 뒤에도 0/16이고, V7a의 시뮬레이터 특권 변형도
0–1/16이다(7.13–7.15절).

따라서 가설은 둘이고, 지금까지의 측정은 둘을 가르지 못한다.

1. **모델** — 우리의 ACT 모양 그래프는 ACT가 아니며, 진짜 ACT라면 과제를 해낸다;
2. **데이터·물리·전문가** — 96×96에서 7초짜리 스크립트 동작 50회는 부족하거나 접촉 모델이
   학습 가능하지 않으며, 어떤 정책도 해내지 못한다.

**V8은 모델만 바꾼다.** 같은 시연, 같은 Task IR, 같은 Deployment IR과 Safety Plane, 같은
`es eval run`, 같은 시드, 같은 합격선 — 대신 정책은 LeRobot 자신의 ACT(CVAE, DETR
인코더/디코더, ImageNet 사전학습 ResNet18)이고 `lerobot-train`이 학습시키며
`crates/es-policy/src/lerobot.rs`로 임포트한다. 그 로워링은 M4 게이트 5 테스트가
`lerobot/act_aloha_sim_transfer_cube_human`에서 비트 단위 일치를 이미 증명한 경로다.

**어느 쪽 결과든 정보이고, 중단 규칙은 측정 전에 적어 둔다.** V8이 nominal
`success_rate ≥ 0.5`에 도달하면 가설 1이 확인되고, 고쳐야 할 것은 Learning IR의 노드 집합이다.
V8도 실패하면 가설 1은 기각된다. 다음 패킷이 열어야 하는 것은 씬·접촉 모델·전문가 궤적이지
**옵티마이저도 IR도 아니다**. 어느 쪽이든 `evaluation-v8.toml`의 임계값은 내리지 않는다.

## 결정들과 각각의 근거

### 1. 외부 정책의 아키텍처는 어디에 사는가

**자기 체크포인트 안에 살고, IR은 그 바이트의 해시를 선언한다.**

`crates/es-policy/src/lerobot.rs`는 이미 그 이유를 적어 두었다. 스펙 §8.3의
`TemporalEncoder { Transformer }`는 폭 하나만 싣고 레이어 수도, 피드포워드 폭도, 잠재 차원도
싣지 않으므로 `lower_to_torch`는 `LearningGraph`만으로 DETR형 ACT를 만들어 낼 수 없다. 노드를
넓히는 것은 패킷이 아니라 스펙 변경이다. 대신 쓸 수 있는 것이 스펙 §8.1 자신의 분할이다.

```
   LearningGraph → Preprocessor(IR에 전부 기술됨)  +  PolicyHandle(불투명)
```

외부 ACT에는 전처리 노드가 필요 없다 — Observation IR이 전부 한다. 그래서 번들의
`LearningGraph`는 **노드 그래프가 비어 있고 `PolicyHandle`만 완전히 타입이 붙은** 모양이며,
이것이 바로 §8.1이 이 경우에 대해 그린 그림이다. 아키텍처 파라미터는 체크포인트 자신의
`config.json`을 타고 오고, `remap_checkpoint`가 그것을 출력 safetensors의 `__metadata__`에
`es.lerobot.act.config`로 기록하며, `TorchRuntime::load`는 그것을 찾으면 `lower_act`로 로워링한다.

이것이 뒷문이 아니라 정직한 길인 이유는 세 가지다.

* **해시 체인이 그대로 덮는다.** config는 `WeightsRef::hash`가 지목하는 바이트 안에 있으므로
  `policy_hash` 안에 있다(§5.3). `load`는 파이썬이 무엇을 보기 전에 선언된 해시를 검증하므로,
  어떤 모듈이 도는지는 여전히 IR이 — 해시를 통해 간접적으로 — 결정한다.
* **느슨해진 것이 없다.** `validate_keys`는 그대로 돌고, `parse_header`는 원래
  `__metadata__`를 건너뛰었으며, 모듈 소스는 여전히 `check()`를 통과한 config로부터 우리
  `lower_act`가 생성하고, `INV-16`은 그대로다. safetensors 들어와서 safetensors 나가고, 어디에도
  pickle은 없다.
* **원래 있던 자리다.** `act_policy`와 `lower_act`는 M4 이래로 체크포인트의 `config.json`을
  읽어 왔다. V8은 추론 시점에도 그 디렉터리가 디스크에 남아 있어야 한다는 요구만 없앤다.

### 2. `meta/stats.json` — 익스포트에 빠져 있던 단 하나

`lerobot-train`은 모든 정책 피처를 `LeRobotDataset.meta.stats`로 정규화하는데, 7.7절의 "의도적인
세 가지 누락"이 `meta/stats.json`을 첫째로 꼽았다. *"없으면 `load_stats`가 `None`을 돌려주고 어떤
읽기 경로도 그것을 필요로 하지 않는다 … 플랜 V의 어떤 것도 `lerobot`을 통해 학습하지 않는다."*
V8은 학습한다. 그래서 파일을 쓴다 — **러스트로 직접**, `crates/es-data/src/lerobot/v3.rs`에서,
parquet을 쓰는 바로 그 패스 안에서. 값에 대한 순회는 둘이 아니라 하나다.

0.6.1의 `lerobot/datasets/compute_stats.py`와 `io_utils.py`에서 확인한 내용물:

* `{feature: {stat: 중첩 리스트}}` — `load_stats` → `cast_stats_to_numpy`가 캐스팅하는 모양;
* 모든 비문자열 피처(다섯 개의 부기 컬럼 포함)에 대해 `min`, `max`, `mean`,
  `std`(모집단, `ddof = 0`, `RunningQuantileStats`와 동일), `count`;
* `[n]` 피처는 길이 `n`의 평평한 리스트, `shape: [1]` 피처는 길이 1의 리스트, 이미지 피처는
  채널별 `[3, 1, 1]`에 `[0, 1]` 범위 — `compute_stats._validate_stat_value`가 받아들이는 세 모양.

**`qNN` 분위수 키는 쓰지 않는다.** LeRobot 자신의 코드에서도 5000 구간 히스토그램 추정치이고,
`NormalizationMode.QUANTILES` / `QUANTILE10`만 읽으며, ACT는 모든 피처가 `MEAN_STD`다. 아무도
읽지 않는 키를 채우려고 근사의 근사를 다시 구현하는 것은, 7.7절이 파일 자체를 지어내기를 거부했을
때와 같은 실수다.

**익스포트 옵션 두 개가 함께 온다.** 둘 다 `lerobot`이 정책 피처를 **이름만으로** 분류하기
때문에 존재한다(`lerobot.utils.feature_utils.dataset_to_policy_features`: `action`으로 시작하는
모든 키가 ACTION 피처, 모든 `observation.*`가 STATE 또는 VISUAL).

* `--drop <a,b>` — 그렇지 않으면 `action_commanded`와 `action_source`가 액션 헤드 두 개로 더
  도착한다. 그것을 빼는 것은 학습상의 결정이므로 호출자가 명시한다.
* `--state-dim <n>` — `es loop collect`는 `observation.state`를 env 0의 전체 `qpos` 뒤에 전체
  `qvel`을 붙여 기록한다(이 씬에서 25). 25개 전부로 학습한 정책은 **애초에 돌릴 수 없다**.
  `es_eval`의 상태 캡처는 `qpos`를 읽고(`Capture::Qpos`, `Capture::Joints`) `qvel` 갈래는 아예
  없다. `--state-dim 6`은 `qpos[..6]`을 남기며, 이는 추론 시 `Capture::Joints(6)`이 Observation
  IR에 건네는 바로 그것이다. 이 거절이 막는 것은, 런타임이 만들어 낼 수 없는 관측 위에서 수렴하는
  학습이다.

### 3. V8이 받는 것, 그리고 일부러 받지 않는 것

**팔의 관절 각 6개와 96×96 overhead 프레임.** V7a의 `sim_cube_pose`는 아니다. LeRobot은 정책에
`observation.state` 피처를 정확히 하나만 주므로 두 번째 상태 포트가 내려앉을 자리가 없고, V8은
*비전* 질문 — V7a가 일부러 미뤄 둔 그 질문 — 이다.

`tests/fixtures/visible-learning/observation-v8.toml`은 같은 Task IR 위의 두 번째 Observation
IR이고(§7: 같은 `task_hash`, 다른 `observation_hash`), 다른 문서들과 같은 선언된 생성기가 만든다.
`observation.toml`과 세 가지가 다르며 각각은 강제된 것이다.

* **상태 포트가 생 라디안을 싣는다**(`Unit::Angle`, `Normalize` 없음). ACT는 정규화 통계를
  체크포인트 안에(`normalize_inputs.buffer_observation_state.{mean,std}`) 두고 순전파의 첫
  연산으로 적용한다 — 우리 로워링은 그것을 노드 9/10/11로 싣는다 — 그러므로 여기의 `Normalize`는
  같은 아핀 사상을 두 번 적용하는 것이 된다. 네트워크가 생값을 먹는 것이 아니다. 자기 첫 층이
  정규화기일 뿐이다;
* **상태 포트는 하나다**, 위의 이유로;
* **포트 이름은 체크포인트의 피처 이름**에서 점을 밑줄로 바꾼 것이다(`observation_state`,
  `observation_images_rgb_overhead`). `XIR-010`은 Observation IR의 출력 이름을
  `PolicyContract::inputs`와 글자 그대로 맞추고, 그 이름은 `config.json`에서 온다. 점은
  `forward(**inputs)` 키워드에 들어갈 수 없다.

이미지 갈래는 `observation.toml`의 것을 바이트 단위로 그대로 쓴다. `ImageInput`(U8 HWC) →
`Dequantize`(CHW F32, /255) → `Normalize{0..1}`. 이것이 이미 LeRobot의 로더가 PNG로부터 만드는
바로 그 텐서다. `Resize`도 `Crop`도 추가하지 않았으므로 내부 파라미터 변환의 빚도 없다(§7.2,
`INV-14`).

`evaluation-v8.toml`은 `evaluation.toml`에서 한 필드 — 가리키는 `observation` — 만 옮긴
것이다(`XIR-040`). 모든 스위트, 섭동, 지표, 합격 임계값이 동일하고 `success_rate ≥ 0.5`도 그대로다.

### 4. 청크 스케줄링은 Deployment IR의 것

`deployment.toml`은 움직이지 않고 `deployment_hash`도 움직이지 않는다. `horizon = 16`,
`execute_chunk = 10`, `TemporalEnsemble { decay = 0.01 }`, 50 Hz 제어, 5 Hz 추론, 그리고 모든
안전 한계가 V6의 것이다.

학습 실행에 따르는 결과 두 가지.

* **`--policy.chunk_size=16`.** LeRobot ACT 기본값은 100이다. `XIR-022`는 런타임이 버퍼링하는
  청크가 정책이 예측하는 청크와 같기를 요구한다.
* **`--policy.n_action_steps=16`, 10이 아니다.** `lower_act`의 모듈은
  `actions[0][:n_action_steps]`를 돌려주는데, 런타임은 16행 전부가 필요하다. *실행* 케이던스는
  런타임의 것이기 때문이다 — `execute_chunk = 10`과 시간 앙상블은 Deployment IR의 것이지(§9.2)
  체크포인트의 것이 아니다. `n_action_steps`는 ACT 손실에 들어가지 않으므로 이것이 바꾸는 것은
  산출물이지 최적화가 아니다. `es policy import-lerobot`은 어긋나는 체크포인트를 청크를 다시
  모양 맞추는 대신 두 플래그를 짚어 **거절한다**.

`act_policy`는 `replanning_hz`, `execution_mode`, 데드라인을 0으로 남긴다. `config.json`에
제어율이 없기 때문이다. 임포트가 그것을 Deployment IR에서 채우며, 이는 `act_policy`의 주석이
이미 그렇게 될 것이라고 적어 둔 대로다.

### 5. 학습 노브

LeRobot ACT 기본값 — 그것이 이 패킷의 요점이다. `vision_backbone = resnet18`에
`ResNet18_Weights.IMAGENET1K_V1`, `use_vae = true`, `latent_dim = 32`, `dim_model = 512`,
`n_heads = 8`, `dim_feedforward = 3200`, 인코더 4층 / 디코더 1층, `kl_weight = 10.0`,
`optimizer_lr = 1e-5`, `optimizer_lr_backbone = 1e-5`, `dropout = 0.1`, 전부 `MEAN_STD`. 배치 8과
시드 0은 V2b/V7a와 같다. 스텝은 100,000이고 10,000마다 체크포인트 — 측정된 처리량이 패킷의 3시간
예산 안에 충분히 들어왔다. 들어오지 않았다면 50,000에서 멈추고 그렇게 적었을 것이다.

시연은 V1c의 50 에피소드이고 `action`은 *실행된* `SafeAction`이다(7.10절). 그러므로 정책이
모방하는 것은 플레인이 통과시킨 것이다.

## context

허용 파일 범위:

```
crates/es-data/src/lerobot/v3.rs          (meta/stats.json, --drop, --state-dim)
crates/es-data/python/lerobot_stats_ref.py
crates/es-data/tests/lerobot_v3.rs        (통계 오라클)
crates/es-data/src/lerobot/mod.rs, src/lib.rs  (재익스포트)
crates/es-policy/src/lerobot.rs           (내장 config, 랭크 가드)
crates/es-policy/src/weights.rs           (`metadata`)
crates/es-policy/src/torch_runtime.rs     (`load`가 모듈을 고른다)
crates/es-policy/python/act_ref.py, tests/act_checkpoint.rs  (기록된 프레임)
crates/es/src/cmd/dataset.rs              (익스포트 플래그 두 개)
crates/es/src/cmd/policy.rs               (`import-lerobot`)
crates/es/tests/cli.rs                    (픽스처 생성기)
tests/fixtures/visible-learning/{observation-v8,evaluation-v8}.toml
docs/api-notes/lerobot-dataset{,.ko}.md   (stats.json 절)
docs/design/visible-learning{,.ko}.md     (7.16절)
docs/packets/M5/V8-external-act{,.ko}.md
```

## spec

* **§8.1.** 네트워크는 불투명하고 인터페이스는 타입이 붙는다. 번들의 `LearningGraph`는
  `PolicyHandle`을 싣고 노드는 싣지 않는데, 네트워크가 여기서 저작되지 않았을 때 그 문장이 갖는
  모습이 바로 그것이다.
* **§8.4 / `XIR-010`, `XIR-011`, `XIR-020`–`XIR-024`.** 계약은 `act_policy`가 `config.json`에서
  투영하고 임포트가 Deployment IR에서 마저 채우며, 모든 교차 IR 규칙은 새 코드 경로가 아니라
  `PolicyBundle::build`/`open`이 검사한다.
* **§8.7.** 전/후처리는 IR에 남는다. Observation IR이 이미지를 정규화하고, 청커·앙상블·역정규화는
  있던 자리에 있다. `PolicyRuntime`으로 옮겨 간 것은 없다.
* **§8.9.** M1 게이트가 고정된 업스트림 체크포인트에서 *임의의* ACT 체크포인트로, 합성 램프에서
  기록된 프레임으로 일반화된다.
* **§5.3 / §25.1.** 가중치는 신뢰 경계를 넘는다. 아키텍처를 그 바이트에서 읽기 **전에** 선언된
  해시를 검사하고, 원본 체크포인트의 해시는 `BaseModelRef`에 기록된다.
* **§19.1.** `meta/stats.json`은 v3.0 포맷의 일부이며, 설치된 패키지에서 확인하고 그 패키지로
  검사한다.
* **`INV-16`.** safetensors만. **`INV-17`.** 새 트레이트 없음.
  **`INV-11`/`INV-12`/`INV-13`.** `es-safety`는 건드리지 않는다.

## oracle

### 로컬, 그리고 이것이 게이트다

```sh
cargo run -p es -- ir check tests/fixtures/visible-learning/task.toml \
    tests/fixtures/visible-learning/observation.toml \
    tests/fixtures/visible-learning/learning.toml \
    tests/fixtures/visible-learning/deployment.toml \
    tests/fixtures/visible-learning/evaluation.toml
cargo test -p es-data --test lerobot_v3 -- --nocapture
cargo test -p es-policy --test act_checkpoint -- --nocapture
cargo test -p es --test cli -- --nocapture policy
cargo xtask ci
```

`observation-v8.toml` 옆에는 Learning IR 픽스처가 일부러 없다. V8의 Learning IR은
`es policy import-lerobot`이 체크포인트와 Deployment IR에서 *유도*하므로 손으로 적을 것이 없고,
`es ir check`에 줄 다섯 번째 문서도 없다. 그 교차 IR 패스는 `PolicyBundle::build` 안에서 돌고,
임포트 명령이 실패하는 것이 그 거절이다.

`meta/stats.json`이 판정받는 자리는
`ES_LEROBOT_PYTHON=<venv>/bin/python cargo test -p es-data --test lerobot_v3`이다.
`lerobot_stats_ref.py`가 `LeRobotDataset`으로 익스포트를 열고, 같은 parquet 위에서 LeRobot
자신의 `compute_episode_stats` + `aggregate_stats`로 통계를 다시 계산해 가장 큰 불일치를
보고한다. 임계값(`< 1e-5`)과 "모든 피처가 다섯 키를 같은 모양으로 갖는다"는 요구는 러스트 쪽에 있다.

### 서버 — 이 순서로 돌린다. 다른 것은 움직이지 않는다

```sh
# 0. 2절이 설명하는 두 선택과 함께 익스포트
./target/release/es dataset export --lerobot-v3 ~/artifacts/plan-v/v1c/ds-train \
    --out ~/artifacts/plan-v/v8/ds-v3 --frames ~/artifacts/plan-v/v8/frames \
    --drop action_commanded,action_source,intervention --state-dim 6

# 1. LeRobot 자신의 트레이너, LeRobot 자신의 ACT 기본값
lerobot-train --dataset.repo_id=es/v8-so101-cube --dataset.root=.../v8/ds-v3 \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=16 --policy.n_action_steps=16 --wandb.enable=false \
    --steps=100000 --batch_size=8 --seed=0 --save_freq=10000 --num_workers=8 \
    --output_dir=.../v8/train --job_name=v8-external-act

# 2. 체크포인트가 번들이 된다
./target/release/es policy import-lerobot \
    --checkpoint .../v8/train/checkpoints/100000/pretrained_model \
    --task tests/fixtures/visible-learning/task.toml \
    --observation tests/fixtures/visible-learning/observation-v8.toml \
    --deployment tests/fixtures/visible-learning/deployment.toml \
    --out .../v8/v8-100000.esb

# 3. 우리 데이터셋에서 나온 프레임 위에서 등가성 게이트
ES_PYTHON=<lerobot venv> ES_ACT_CHECKPOINT=<같은 디렉터리> \
ES_ACT_OBSERVATION=.../v8/frame.json \
    cargo test --release -p es-policy --test act_checkpoint -- --nocapture

# 4. 데모의 합격선이 판정하는 평가
./target/release/es eval run --policy .../v8-100000.esb \
    --evaluation tests/fixtures/visible-learning/evaluation-v8.toml \
    --suite nominal --jobs 6 --out .../v8/nominal-100000
./target/release/es eval run --policy .../v8-100000.esb \
    --evaluation tests/fixtures/visible-learning/evaluation-v8.toml \
    --jobs 6 --frames .../v8/frames-suite --out .../v8/suite-100000

# 5. V3와 V7a가 만들던 방식 그대로 영상
./target/release/es video mosaic --grid 4x4 --frames .../v8/frames-suite --out .../v8/mosaic
python/es/encode_video.py --frames .../v8/mosaic --fps 50 --out .../v8/demo-v8.mp4
```

## acceptance

1. `meta/stats.json`을 `LeRobotDataset`이 모든 피처와 함께 읽고, 그 숫자가 LeRobot 자신의
   `compute_episode_stats` + `aggregate_stats`와 `< 1e-5`로 일치한다. **RAN**.
2. `lerobot-train`이 트레이너를 하나도 고치지 않고 익스포트 위에서 완주한다.
3. `es policy import-lerobot`이 만든 번들을 `PolicyBundle::open`이 재검증하고,
   `es eval run --policy`가 **평가 경로를 하나도 바꾸지 않고** 읽는다 —
   `policy_pack_output_is_accepted_by_eval_run`이 서술하고 설계 노트 2.5절이 닿을 수 없다고
   기록했던 그 주장이다.
4. 등가성 게이트가 기록된 프레임 위에서 스펙 §8.9 tier 4(≤ 1e-5)로 통과하고, 측정된
   `max_abs` / `max_rel`이 7.16절에 기록된다.
5. 시드 101–116의 nominal 스윕과 전체 스위트를 최종 체크포인트와 최소 두 개의 이전
   체크포인트에서 측정하고, 그 숫자가 **측정된 그대로** 7.16절에 들어간다. 판정은 데모 자신의
   합격선(`success_rate ≥ 0.5`)이며, "이 패킷이 답하려는 질문"의 중단 규칙을 적힌 대로 적용한다 —
   멈추라고 말할 때에도.
6. `cargo fmt --check`, `clippy -D warnings`, `cargo xtask ci`.

## forbidden

* **`es-safety`.** 한 줄도, `deployment.toml`의 숫자 하나도. `INV-11`, `INV-12`, `INV-13`이 모두
  유효하고, 외부 정책은 전문가와 모든 플랜 V 정책이 달렸던 것과 같은 봉투 뒤에서 달린다.
* **`es-ir`.** 새 노드도, 새 필드도, 새 `ArchKind`도 없다. IR이 외부 정책을 기술할 수 없었다면
  패킷은 멈추고 보고했어야 한다. §8.1이 정확히 그것을 기술한다.
* **`evaluation.toml` 임계값.** `success_rate ≥ 0.5`는 데모의 합격선이고 표를 통과시키려고 내리지
  않는다 — 골든을 편집하는 것과 같은 규칙이다.
* **`lower_to_torch`, `python/es/train_act.py`, `es dataset bake`.** V8은 IR 로워링도 베이크도
  전혀 거치지 않는다. LeRobot의 트레이너가 LeRobot 데이터셋을 직접 읽는다.
* **v2.1 라이터와 `es loop collect`.** 컬럼을 더하지 않았고 시연을 다시 수집하지 않았다.
  `dataset_schema_hash`는 움직이지 않는다.

## as built

오라클 서버에서 측정(RTX 4090, 학습과 레퍼런스는 `~/venvs/es-lerobot-cuda`, 평가는 `~/venvs/es`),
2026-09-15. 전체 기록: 설계 노트 **7.16절**.

**판정, 그리고 적용된 중단 규칙.** 100,000 스텝에서 nominal `success_rate = 0.0625`(1/16),
20,000과 50,000에서는 0/16. 모든 섭동 스위트가 같은 1/16이다. 데모의 합격선은 `0.5`이고 **임계값은
하나도 내리지 않았다**. 패킷의 가설 1 — "우리 ACT 모양 그래프가 데모가 안 되는 이유다" — 은
**기각된다**. 사전학습 백본과 CVAE와 DETR 디코더를 갖추고 다섯 배 길게 학습한 진짜 LeRobot ACT가 더
낫지 않다. 중단 규칙에 따라 다음 패킷이 여는 것은 씬·접촉 모델·전문가의 궤적이지, 옵티마이저도
`es-ir`도 아니다.

| 체크포인트 | `success_rate` | 평균 에피소드 길이 | `envelope_violation_rate` |
|---|---|---|---|
| 20,000 | 0.0000 (0/16) | 900.0 | 0.180 |
| 50,000 | 0.0000 (0/16) | 900.0 | 0.370 |
| 100,000 | 0.0625 (1/16) | 865.5 | 0.356 |
| 100,000, 전체 스위트 (6 × 16) | 모든 스위트 0.0625 | 864.6 – 873.0 | 0.253 – 0.352 |

합격 조건, 항목별로:

1. **충족.** `RAN lerobot_v3_stats`, LeRobot 자신의 `compute_episode_stats` + `aggregate_stats`에
   대해 최대 불일치 `5.5e-08`, 빠진 것도 남는 것도 없음.
2. **충족.** `lerobot-train`이 트레이너를 하나도 고치지 않고 익스포트 위에서 100,000 스텝 완주
   (33분 50초; 손실 4.270 → 0.026).
3. **충족.** 번들이 다시 열리고 `es eval run --policy`가 평가 경로를 바꾸지 않고 읽는다.
   `deployment_hash 3b2ad568…6db1`과 `task_hash 6cf826c1…6b7b`는 움직이지 않았다.
4. **충족, 비트 단위.** `max_abs 0e0`, `max_rel 0e0`, 청크 `[16, 6]`, 이 프로젝트 자신의 데이터셋에서
   나온 기록된 프레임 위에서, 20,000과 100,000 체크포인트 양쪽에서.
5. **충족.** 위와 7.16절의 표. 100,000 스텝 nominal 실행을 한 번 더 돌렸고 `report.json`이 바이트
   단위로 동일했다.
6. **충족.**

**패킷이 몰랐던 세 가지, 전문은 7.16절에.**

1. **생 `Unit::Angle` 정책 입력은 `TYPE-011`이 거절한다.** 교차 IR 패스가 그래프 경계뿐 아니라
   `PolicyContract::inputs`에 대해서도 검사하기 때문이다. 해법은 두 번째 아핀 사상이 아니다 —
   그것은 내보낸 데이터셋에도 적용되어야 하고, 즉 `Op::Normalize`의 두 번째 구현이 된다 — 참인 것을
   말하는 것이다. `Normalize{Range{0..1}}`, 항등, 그 결정을 소유한 노드에 선언. 그다음 임포트는 각
   계약 입력의 단위를 Observation IR에서 가져온다. `config.json`이 단위를 기록하지 않기 때문이다.
2. **`lerobot-train` 0.6.1이 쓰는 체크포인트는 정규화 통계를 별도의 프로세서 상태 파일에 둔다.**
   `model.safetensors`가 아니다. M4 게이트가 쓴 고정 업스트림 체크포인트가 옛 레이아웃이다. 이제
   둘 다 읽힌다. `docs/api-notes/lerobot-act.ko.md` 9절.
3. **빈 Learning IR 그래프는 `LRN-021`이 거절하고**, 그 힌트가 옳은 답을 짚는다. 스펙 §8.3도 대놓고
   같은 말을 한다. 정책을 `LearningNode::PolicyBundle`로 통째로 참조하라.

**열린 질문 (V8-1)** 은 7.16절 끝에 적혀 있고 그 노트의 12절 목록에 들어가야 한다. 외부 ACT의 실패가
결론 내려도 되는 것, 그리고 다음 패킷은 더 큰 학습 실행이 아니라 씬의 진단(전문가 자신이 기록한
액션을 `es eval run`으로 재생)이어야 한다는 것.
