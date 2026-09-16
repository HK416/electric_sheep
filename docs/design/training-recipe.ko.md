# 학습 레시피 — `es train`

스펙: §13.1(루프의 각 단계는 CLI 명령이자 산출물이다), §19.3(Training Identity — `training/`
번들과 `training_hash`), §8.8, §2.3(Python은 학습 경로에만), §1.4, §28.10 규칙 2(`es train`은
§19.3의 모든 슬롯을 실제 값으로 채우거나 unset으로 표시한다).

패킷: `docs/packets/M7/T1-es-train.md`. 선행: M5 V2/V2b(`train_act.py`, `es dataset bake`),
V8/V19(`import-lerobot`, `lerobot-train`), M3 W7(`docs/design/learning-loop.md` 1절 —
`training_identity.json`이 dataset 슬롯만 빼고 전부 0인 그 지점). 이 노트가 사람이 검토하는
산출물이고, `crates/es-data/src/training.rs`와 `crates/es/src/cmd/train.rs`가 그 하류다.

---

## 1. 한 명령이 무엇을 대체하는가

이 패킷 이전에 학습 한 사이클은 손으로 쓴 스크립트 아홉 개였다. §28.10의 "as built" 표가 그
수를 세고 있고, `docs/packets/M5/V19-external-act-current-data.md`의 server 절이 증거다.
`mkdir frames-in && ln -s …`, 플래그 여섯 개짜리 `es dataset export`, 열한 개짜리
`lerobot-train`, 다섯 개짜리 `es policy import-lerobot`, 그 사이를 손으로 옮겨 붙인
`observation_hash`와 `task_hash`. IR 경로도 `bake`·`lower`·`train_act.py`·`pack`으로 모양이
같았다. §19.3의 `training/`을 쓰는 것은 아무것도 없었다.

`es train --recipe training.toml [--out <dir>] [--dry-run] [--allow-retired-task <hex>]`이
두 경로 모두에 대해 문서 하나, 명령 하나다. 계획의 `es` 단계들은 **같은 프로세스 안에서**
호출된다 — `crates/es/src/cmd/{dataset,policy}.rs`의 함수들이고, 이 패킷이 그 파일들에 가한
유일한 변경은 그중 다섯 개를 `pub(crate)`로 바꾼 것이다 — 그리고 Python 트레이너만이
하위 프로세스다. §2.3이 선을 긋는 바로 그 자리다.

**저장소 루트에서 실행한다.** IR 경로의 트레이너는 `python/es/train_act.py`이고 Task IR의
`scene.path`는 저장소 기준 상대 경로다. 둘 다 다른 모든 `es` 경로 플래그와 마찬가지로 작업
디렉터리 기준으로 풀린다. 레시피 안의 상대 경로도 같은 방식이며, 그래서 `--dry-run` 계획이
재현 가능하다(3절).

## 2. 레시피

```toml
kind = "training"

[dataset]
root   = "runs/collect-001/ds"        # `es loop collect`가 쓰는 LeRobot v2.1 루트
frames = "runs/collect-001/frames"    # 평평한 <NNNNNN>.bin 타일. 이미지 입력이 있으면 필수

[policy]                              # `bundle`과 `lerobot` 중 정확히 하나
bundle = "runs/collect-001/untrained.esb"
# lerobot     = { type = "act", chunk_size = 16, n_action_steps = 16, extra = [] }
# task        = "…/task.toml"         # import가 번들로 실어 나르는 세 문서
# observation = "…/observation-v8.toml"
# deployment  = "…/deployment.toml"

[run]
steps         = 20000
batch         = 8
lr            = 1e-4
seed          = 0
checkpoint_at = [1000, 5000, 20000]
device        = "cuda"
interpreter   = "python"              # ES_PYTHON이 설정되어 있으면 그쪽이 이긴다
```

모든 테이블이 `deny_unknown_fields`다. 오타는 조용히 무시되는 손잡이가 아니라 거절이다.
스키마·계획 빌더·아이덴티티 조립은 `es-data`에 있고 프로세스도 파일도 GPU도 없이 단위
테스트된다(`crates/es-data/src/training.rs`). CLI는 그 껍데기다. TOML은
`es_ir::serial::parse_toml`로 읽는다 — 이 저장소의 TOML 리더가 이미 사는 곳이고, 레시피도
다른 모든 문서와 같은 문서다. `es-data`에 의존성은 하나도 추가하지 않았다.

이름을 붙여 둘 만한 스키마 결정 둘:

* **`steps`가 실행 길이의 권위다.** 그래서 항상 마지막 체크포인트 마크다.
  `train_act.py --checkpoint-at`은 가장 큰 마크에서 실행을 끊으므로, 다르게 읽으면
  `checkpoint_at`이 조용히 실행을 짧게 만들 수 있다.
* **`amp` 필드는 없다.** `train_act.py`의 독스트링 스스로 `--amp`를 스윕용 플래그라고 부른다 —
  "수치를 인용할 실행이 아니라 스윕에 쓰라" — 그리고 레시피는 수치를 인용할 실행을 기술하려고
  존재한다. 그래서 `precision.json`은 오늘 `fp32 / amp off`다. 기록되는 스윕이 필요해지면 T4가
  필드를 추가한다.

커밋된 레시피 둘은 `tests/fixtures/visible-learning/training.toml`과
`training-lerobot.toml`이다. 외부 경로 쪽은 `observation.toml`이 아니라
**`observation-v8.toml`**을 가리킨다. 외부 체크포인트의 입력 이름은 그 데이터셋의 피처 이름에서
점을 밑줄로 바꾼 것이고(§8.4, `XIR-010`), 포트가 정확히 그 이름을 다는 커밋된 Observation IR이
`observation-v8.toml`이다. `observation.toml`을 가리키면 포트 이름을 대는 거절이 나온다 —
측정했다, 7절.

## 3. 계획, 그리고 그것이 골든인 이유

```
# route: ir
es dataset bake --policy … --out baked --frames … <root>
es policy lower --policy … --out module
python python/es/train_act.py --module module --baked baked --out weights/model.safetensors
    --checkpoint-at 1000,5000,20000 --seed 0 --batch 8 --lr 0.0001 --device cuda
    --loss-curve metrics/loss.json
es policy pack --policy … --weights weights/model-1000.safetensors --out checkpoints/1000.esb
… 마크마다 하나씩
```

```
# route: external
es dataset export --lerobot-v3 <root> --out ds-v3 --frames frames-in
    --drop action_commanded,action_source,intervention --state-dim 6
python -m lerobot.scripts.lerobot_train --dataset.repo_id=es/train --dataset.root=ds-v3
    --policy.type=act --policy.chunk_size=16 --policy.n_action_steps=16 --policy.device=cuda
    --policy.push_to_hub=false --policy.optimizer_lr=0.0001 --steps=20000 --batch_size=8
    --seed=0 --save_freq=10000 --output_dir=lerobot --job_name=es-train --wandb.enable=false
es policy import-lerobot --checkpoint lerobot/checkpoints/010000/pretrained_model
    --task … --observation … --deployment … --out checkpoints/10000.esb
… 마크마다 하나씩
```

`tests/golden/train/plan-ir.txt`와 `plan-lerobot.txt`가 저 둘을 한 단계 한 줄로 그대로 담는다.
이것을 기계가 아니라 레시피만의 성질로 만드는 것이 셋이다.

1. **`<out>` 아래의 모든 경로는 `<out>` 기준 상대 경로로 인쇄된다.** 단어의 머리에서만이 아니라
   단어 어디에 나타나든 그렇다 — `--output_dir=lerobot`처럼 — 그리고 구분자는 전부 `/`다. 그래서
   Windows와 Linux가, 서로 다른 두 스크래치 디렉터리가 하나의 계획을 낸다.
2. **그 밖의 모든 경로는 레시피가 쓴 단어 그대로다.** 레시피가 입력이고, 계획은 그것을 고쳐
   쓰지 않는다.
3. **오라클은 `ES_PYTHON`을 제거한다.** 인터프리터는 계획에서 기계에 의존하는 유일한 단어이고,
   `es train`은 계획에서도 실행에서도 같은 방식으로 그것을 결정한다. 그러니 골든을 고정하는
   정직한 방법은 오버라이드를 끈 채로 찍는 것이지, 계획이 그것을 쓰지 않는 척하는 것이 아니다.

`--dry-run`은 레시피로부터, 그리고 외부 경로에서는 레시피가 가리키는 Observation IR로부터
계획을 만든다. **데이터셋도 번들도 열지 않는다.** 그래서 CI가 둘 다 디스크에 없이, Python 없이
판정할 수 있다. `<out>/training/plan.txt`만 쓰고 그 밖에는 아무것도 쓰지 않는다. 일어나지 않은
실행은 아이덴티티를 주장하지 않는다.

계획의 두 유도 값:

* **`--state-dim`**은 Observation IR의 `StateInput` 출력 폭의 합이다. 데모의 병합 관측에서는
  `6 + 7 = 13`으로, M5/V19가 손으로 넣었던 그 수다. `observation-v8.toml`에서는 V8의 6이다.
  해석된 인덱스 구간이 아니라 합인 이유: 소스 id를 `ModelInfo`에 해석하려면 씬이 필요한데
  `es-data`는 `es-eval`과 같은 레이어 10이라(§4.2) 물어볼 수 없다. `qpos`가 겹치는 두 채널은
  과다 계산되고 export가 아무도 읽지 않는 열을 싣게 된다 — 낭비이지 틀림은 아니다.
* **`--save_freq`**는 가장 작은 체크포인트 마크다. `lerobot-train`은 목록이 아니라 하나의
  주기로 저장하기 때문이다. 그 배수가 아닌 마크는 반올림이 아니라 이름을 대고 거절된다.
  디스크에 없는 체크포인트는 import할 수 없고, 마크를 조용히 옮기면 `checkpoint.manifest`에
  틀린 스텝 번호가 들어간다.

**`frames-in`.** `es dataset export --frames`는 카메라마다 디렉터리 하나를 원하고
`es loop collect --frames`는 평평한 하나를 쓴다. V19는 그 사이를 `mkdir`과 `ln -s`로 손수
이었다. 외부 경로에서 `es train`은 타일을 `<out>/frames-in/<camera>`로 하드링크한다(링크가
실패할 때만 복사하고, 이미 있는 것은 건너뛴다). 손으로 만든 심볼릭 링크가 여전히 필요한
레시피는 한 명령이 아니기 때문이다.

## 4. §19.3의 열두 슬롯 — 트레이너가 돌기 전에 무엇을 아는가

`<out>/training/`은 언제나 파일 열두 개를 담고, `<out>/training.lock`은 언제나 두 개의 다이제스트를
싣는다.

| | 슬롯 | 출처 | 시점 |
|---|---|---|---|
| 1 | `config.json` | 정규 JSON으로 쓴 레시피, 해석된 인터프리터, `<out>` 상대 계획 | 사전 |
| 2 | `optimizer.json` | `AdamW` + `lr`. IR 경로에서는 `AdamW(params, lr=lr)`가 torch 기본값으로 두는 betas/eps/weight decay도 | 사전 |
| 3 | `scheduler.json` | T4까지 `{"kind":"constant","lr":…}` | 사전 |
| 4 | `seed.json` | `[run] seed`에서 온 `global`·`dataloader`, `augmentation`은 unset | 사전 |
| 5 | `dataset.lock` | `es-data::identity`의 content/schema/split, 에피소드·프레임 수, 기록된 `es:task:` 이름 | 사전 |
| 6 | `base_model.lock` | `{"source":"none"}`(IR) 또는 선언된 `vision_backbone` + `pretrained_backbone_weights`(외부) | 사전 |
| 7 | `augmentation.json` | T6까지 `{"kind":"none"}` | 사전 |
| 8 | `precision.json` | `fp32`, `amp off`, IR 경로에서 `gradient_accumulation` = `[run] batch` | 사전 |
| 9 | `topology.json` | `{"world_size":1}` | 사전 |
| 10 | `checkpoint.manifest` | 스텝, 번들 경로, 번들 자신의 §5.3 `policy_hash`, 번들 바이트의 blake3 | 사후 |
| 11 | `metrics.json` | `train_act.py`의 손실 곡선과 JSON 요약 | 사후 |
| 12 | `hardware.json` | 인터프리터 프로브의 응답 + `es` 버전 | 사후 |

`identity_hash`는 10–12가 아직 `{"unset": true}`인 상태의 열둘에 대한
`TrainingIdentity::training_hash`이고, `training_hash`는 그것들이 채워진 뒤의 같은 함수다. 둘 다
`training.lock`에 있고, 각 파일의 blake3와 체크포인트별 §19.3의
`policy_hash = H(training_hash, checkpoint_hash)`가 나란히 있다.

**아이덴티티는 어떤 단계보다 먼저 쓰인다.** 거절 검사 직후, 인터프리터를 찔러 보기도 전이다.
그것이 이 분할의 요점이다. 실행의 이름은 GPU 1초를 쓰기 전에 존재하고, 한 기계에서 한 레시피를
두 번 돌리면 둘 중 어느 것도 끝나기 전에 이미 같은 실행이다.
`crates/es/tests/cli.rs::train_identity_is_a_function_of_the_recipe`가 그 성질을 끝에서 끝까지
확인하고, 7절이 오라클 서버에서 측정한다.

**모든 슬롯은 존재하고 파싱되는 파일의 blake3다.** 실행이 진짜로 모르는 슬롯은
`{"unset":true}`라는 파일이고 그대로 해시된다. `es loop distill`이 0을 쓰는 것은 §2.3 경계의
반대편에 있어 아무것도 모르기 때문이고(`docs/design/learning-loop.md` 1절), `es train`은 이쪽
편에 있어 거의 전부를 아니까 "모른다"도 변호할 수 있는 값이어야 한다. 이 명령이 쓰는
`training.lock` 어디에도 전부 0인 다이제스트는 없고, 오라클이 그것을 단언한다.

**오늘 무엇이 unset이고 왜인가.**

| unset | 이유 | 주인 |
|---|---|---|
| `seed.json.augmentation` | 증강이 없다 | T6 |
| `base_model.lock.weights_hash`·`.license` | *선언된* 출처다. `extra`가 덮지 않는 한 백본은 LeRobot ACT의 기본값이고, 여기서는 그 가중치를 내려받지도 검증하지도 않았다 | T5 |
| 외부 경로의 `optimizer.json.betas`·`.weight_decay` | `lerobot`의 옵티마이저 블록은 이쪽이 선언할 것이 아니다 | T4 |
| 외부 경로의 `metrics.json.loss` | `lerobot-train`은 곡선을 자기 로그에 적지, 이 명령이 읽는 파일에 적지 않는다 | T2 |
| `hardware.json.driver` | `torch`는 CUDA 툴킷을 보고하지 드라이버를 보고하지 않는다 | — |
| `hardware.json.git_describe` | 빌드 스크립트가 없고 `Cargo.lock`이 gitignore다(M5 리뷰 S-8 / R7). 빌드를 이름 붙일 수 없는데 리비전을 주장하면 §28.10 규칙 2가 금지하는 날조다 | M5 R7 |

**`metrics.parquet`이 아니라 `metrics.json`.** §19.3은 `metrics.parquet`이라고 쓴다. `es-data`의
parquet writer(`crates/es-data/src/lerobot/{v3,columns}.rs`)는 LeRobot의 `Info`와 `Episode`
— 피처, 청크, 에피소드 인덱스, 경로 템플릿 — 을 중심으로 짜여 있고 `pub(crate)`다. 두 열짜리
스칼라 표를 그것으로 쓰는 일은 "몇 줄"이 아니라 두 번째 writer다. 패킷 자신의 조건이 그래서
성립하지 않고 슬롯은 JSON이다. 곡선을 읽는 것을 넘어 질의해야 할 날이 오면 그건 패킷이다.

**`hardware.json`은 프로브 하나에서 온다.** bake나 export보다 먼저, `es train`은 인터프리터를
한 번 돌려 `torch`(외부 경로에서는 `lerobot`도)를 import하고 `torch.__version__`,
`torch.version.cuda`, `torch.cuda.get_device_name(0)`을 찍는 네 줄짜리 스크립트를 실행한다. 그
프로브가 동시에 패킷이 말하는 "torch를 import하지 못하는 인터프리터" 거절이며 — 인터프리터
자신의 에러를 그대로 인쇄한다 — 이 슬롯의 출처다. 일부러 앞에 둔다. 망가진 환경은 2초짜리
답이고, 10분짜리 bake 뒤에 그것을 알게 되는 것이야말로 이 명령이 없애려는 종류의 일이다.

**`optimizer.json`은 선언하고 나서 검사한다.** IR 경로에서 `train_act.py`는 이제 자기가 만든
옵티마이저와 `torch.__version__`을 JSON 요약에 보고한다(이 패킷이 그 파일에 가한 유일한
변경이다. 학습 루프는 T3/T4의 것이다). `es train`은 그 보고를 실행 전에 쓴 `optimizer.json`과
비교하고, 다르면 둘 다 이름을 대는 경고를 인쇄한다 — torch가 `AdamW` 기본값을 옮겼고 선언이
낡았다는 뜻이다. 거절하지는 않는다. 그 시점엔 체크포인트가 이미 디스크에 있고, 낡은 선언은
T4에서 고칠 일이지 실행을 버릴 이유가 아니다.

**`checkpoint.manifest`는 §19.3의 것이 아니라 *번들의* `policy_hash`를 싣는다.** §19.3은
`policy_hash = H(training_hash, checkpoint_hash)`로 정의하고 `training_hash`는
`checkpoint.manifest`를 덮는다 — 그 수는 자기가 계산되어 나오는 파일 안에 살 수 없다.
매니페스트는 번들 자신의 §5.3 `policy_hash`(`es policy pack`이 인쇄하는 그것)를
`bundle_policy_hash`로 적고, `training.lock`이 체크포인트별로 §19.3의 것을 싣는다. 둘 다
존재하고 어느 쪽도 순환하지 않는다.

## 5. 거절, 각각 이름을 대고

| 거절 | 메시지가 대는 이름 |
|---|---|
| `[policy]`가 `bundle`과 `lerobot`을 둘 다, 또는 둘 다 안 씀 | 어느 쪽인지, 그리고 각 경로의 뜻 |
| `task`/`observation`/`deployment` 없는 외부 경로 | 빠진 키 |
| 이미지 입력이 있는데 `[dataset] frames`가 없음 | `frames`, 그리고 0으로 채운 이미지 채널은 멀쩡해 보이는 정책을 학습시킨다는 것 |
| `lerobot-train`의 단일 `--save_freq`로 만들 수 없는 `checkpoint_at` | 마크들과 주기 |
| 기록된 `task_hash`가 정책의 것이 아닌 데이터셋 | 두 해시와 `--allow-retired-task <hex>` |
| 경로가 필요로 하는 것을 import 못 하는 인터프리터 | 인터프리터와 그 자신의 stderr |

작업 출처 검사는 M5 리뷰 **S-3/R4**를 여기서 드러낸 것이다. `gripper > 0.6`으로 모은 시연과
`> 0.85`로 모은 시연이 오라클 서버에 나란히 있고, V19의 패킷은 틀린 쪽을 지목해 구조적으로
0/16을 측정했다. `es loop collect`는 `meta/tasks.jsonl`에 `es:task:<hex>`를 기록하고,
`es train`은 그것을 레시피가 가리키는 Task IR과 비교해 `--allow-retired-task`가 **그 폐기된
해시를 이름으로 댈 때만** 통과시킨다 — 받아들이는 일이 의도적이고 기록되는 행위가 되고, 켜 둔
채 잊는 플래그가 되지 않는다. `es:task:` 이름이 없는 데이터셋(외부에서 가져온 것)은 아무 주장도
하지 않으므로 검사하지 않는다.

## 6. 오라클

| # | 명령 | 필요 |
|---|---|---|
| 1 | `cargo test -p es --test cli train_dry_run_plan_is_the_golden` | 없음 |
| 2 | `cargo test -p es --test cli train_identity_is_a_function_of_the_recipe` | 없음 |
| 3 | `cargo test -p es --test cli train_ir_path_packs_a_bundle_torch_opens -- --ignored` | torch가 있는 `ES_PYTHON`, MuJoCo |
| 4 | `cargo test -p es --test cli train_refuses_by_name` | 없음 |
| — | `cargo test -p es-data --lib training` | 없음(헤드리스 절반: 스키마·계획·아이덴티티) |

오라클 2는 존재할 수 없는 인터프리터를 가리키는 레시피로 실제 `es train`을 돌린다. 실행은 항상
멈추고, 단언하는 아이덴티티는 멈추기 전에 쓰인 그것이다. 사전/사후 분할을 실행 가능한 문장으로
만든 것이고, Python이 필요 없다.

골든은 `cargo test -p es --test cli -- --ignored generate_train_goldens`로만 재생성되며 절대
손으로 고치지 않는다(§1.4).

## 7. 측정 — 오라클 서버, RTX 4090, 2026-09-16

오라클 3, `ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`: **통과**. 데모 번들과 bake 픽스처
(에피소드 4 × 프레임 16)로 40 스텝, `TorchRuntime::load`가 `checkpoints/40.esb`를 받고
`infer`가 유한한 청크를 돌려주며, `checkpoint.manifest`가 그 파일을 자기 바이트의 blake3와
함께 이름 짓는다. `training_hash 50f17bc2…` ≠ `identity_hash`.

인수 조건의 외부 실행 — `training-lerobot.toml`의 계획을 V15의 시연 200편에 대해
`steps = 200`으로 한 번 돌려, 이 명령 줄이 `lerobot-train`이 받아들이는 명령 줄임을 보이는 것.
이것은 **관측이지 게이트가 아니다.**

```sh
cd ~/Projects/es-t1
ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python \
  ./target/release/es train --recipe ~/artifacts/plan-v/m7-t1/training-v15-v8.toml \
                            --out ~/artifacts/plan-v/m7-t1/run2
```

`lerobot-train`은 계획이 만든 모든 플래그를 받아들였다. `--policy.optimizer_lr=0.0001`도
포함이다(자기 로그 줄이 `lr:1.0e-04`로 찍힌다). export, 200 스텝, import, lock까지 한 명령에서
**벽시계 19.22초**(학습 자체는 6.4초, 약 57 step/s. 프레임 36,960개, 에피소드 200편, 파라미터
52M, `loss 4.458`, `l1_loss 0.323`).

```json
{ "schema_version": 1,
  "identity_hash": "476ed25a414a95c4dcd44748127a304eccc393ad5caa3ff7eb1e6e932e293e1f",
  "training_hash": "bd1b3c863877ed0ea923f0995019648c3203f58097e488686d9fe17b4162578a",
  "files": {
    "augmentation.json":   "7d75c7e1f7056cd846c4c047af66b8a6ff019e0b8be156c9e7e757814ac7bf8f",
    "base_model.lock":     "00ea2e622bdbbdd477695b9c317337fda29ec400edfaa76410c75345f13f002e",
    "checkpoint.manifest": "18fdf2cb808378d1a0df31f3506b1e9ccb340ed1199dbd614cd698d7781392d6",
    "config.json":         "e0da61a6b4b6a151d1b7e0080f3b5c2706ef1241ac835f893a2994faa674cc64",
    "dataset.lock":        "a7aad6ab4e04c2b9330cf9dcf7cbd3d5b24554389e492b1b9cbf2322316cb2cf",
    "hardware.json":       "a983e8821036497cdb0017a0fff27f81c8383e8458f69a27385ed64267265033",
    "metrics.json":        "4f62d486b5a031896112a1d11b9254b130bf7997c5ef66527ca7362b338960e9",
    "optimizer.json":      "35a78de1a1aa9a7fd8f09a63897fdfbd469739499697057c782699ac3ac2a9ca",
    "precision.json":      "033d85b1122382be956cefc988393ccafa22260cca72ec3fa53d20a41b612cd7",
    "scheduler.json":      "8847154d020d177c6f68fddf7bbd89ffee17591758a4b6719e0d10fd51afe7aa",
    "seed.json":           "2f6d6194c4517524d5cad5522335f2ceb612722721babe40bf3ea83125dace10",
    "topology.json":       "79cf76183503dd7106c635e79a4d57c6042d06059bf6a8ccdba805d1ee777341" },
  "checkpoints": [ { "step": 200, "bundle": "checkpoints/200.esb",
                     "policy_hash": "10a54412b5188c08e27fd4aae09f8edac8c1f25be642331662480d8b550be0b2" } ] }
```

프로브가 채운 `hardware.json`: `torch 2.11.0+cu129`, `cuda 12.9`,
`device_name "NVIDIA GeForce RTX 4090"`, `lerobot 0.6.1`, `trainer "lerobot-train"`,
`es_version 0.1.0`, `driver`와 `git_describe`는 unset. `dataset.lock`: `content 55a958e1…`,
`schema 1f5ddafc…`, `split 872fe162…`(all-train), 에피소드 200편, 프레임 36,960개,
`recorded_task es:task:eb6efefa…` — 커밋된 `task.toml` 자신의 `task_hash`다. 즉 S-3 검사가
픽스처가 아니라 실제 데이터에서 통과했다. import된 번들: `observation_hash c69a8e11…`,
`learning_hash cc4587b9…`, `policy_hash a7e06b32…`, `deployment_hash f2f9a510…`.

**같은 레시피를 다른 `<out>`로 한 번 더**(`run3`, 21.04초): `identity_hash 476ed25a…`로 하나,
`training_hash`는 `452ae7d7…`로 다르다. 열두 파일 중 정확히 하나 — `checkpoint.manifest` —
만 다른데, `lerobot-train`이 같은 시드에서 비트 동일한 가중치를 내지 않기 때문이다. 분할이
제 일을 한 것이다. `identity_hash`는 레시피의 이름이고 `training_hash`는 실행의 이름이다.
외부 트레이너가 비트 재현 가능*해야 하는지*는 아래 열린 질문 29다.

첫 시도(`run1`)는 커밋된 `observation.toml`을 썼고 2절에 인용한 포트 이름 메시지로 import에서
거절됐다 — 거절이 동작한 것이며, `training-lerobot.toml`이 `observation-v8.toml`을 가리키는
이유다.

## 8. 패킷과 달라진 점

1. **트레이너가 보고한 버전은 `config.json`이 아니라 `hardware.json`에 있다.** 패킷은 양쪽에
   적어 두었다. `identity_hash`가 트레이너보다 먼저 존재하려면 `config.json`이 사전 슬롯이어야
   하는데, 트레이너가 보고하는 버전은 그때 알 수 없다.
2. **`checkpoint.manifest`는 `bundle_policy_hash`를**, `training.lock`이 §19.3의 `policy_hash`를
   이름 짓는다 — 패킷의 문장은 순환한다, 4절.
3. **외부 명령 줄에 `--policy.optimizer_lr`이 있다.** 패킷의 플래그 목록에는 없다. 그것이 없으면
   `[run] lr`이 실행에 아무 영향도 없이 `optimizer.json`과 해시에 들어가는데, 그건 거짓말이다.
   `lerobot-train` 0.6.1이 받아들임을 확인했다, 7절.
4. **`--dataset.repo_id=es/train`과 `--job_name=es-train`은 상수다.** 둘 다 `lerobot-train`이
   요구하는 이름이고 여기서 무엇을 식별하지도 않는다. 경로에서 유도하면 스크래치 디렉터리가
   골든에 들어간다.
5. **패킷의 `## context` 절에 펜스로 감싼 glob 목록이 생겼다.** `cargo xtask check-scope`가
   파싱할 수 있도록이고, 산문 목록은 그 옆에 그대로다. 파서는 문자 그대로의 `## context` 제목과
   펜스 또는 불릿 목록을 원하며(`xtask/src/scope.rs::parse_context_globs`), `xtask/`는 이
   패킷의 범위 밖이다.

## 9. 열린 질문

* **29.** `lerobot-train`은 한 시드에서 비트 동일한 가중치를 내지 않으므로(7절), 한 기계에서
  한 레시피를 두 번 돌리면 `training_hash`가 달라진다. §3.5 티어 1은 *우리* 커널에 관한
  것이고 외부 트레이너는 그중 하나가 아니다 — 그러나 증거 번들은 체크포인트의 출처가 어느
  티어에 속하는지를 말해야지, 어긋나는 두 해시로부터 독자가 추론하게 두어서는 안 된다.
* **30.** `--state-dim`은 `StateInput` 폭의 합이다(3절). 정확한 답은 해석된 `qpos` 구간들의
  합집합이고, 그것은 씬과 `es-data`가 닿을 수 없는 레이어를 필요로 한다. Observation IR이
  기록된 행에서 읽는 폭을 스스로 선언하거나, 유도를 `es-eval`에 두고 `es train`이 CLI에 물어야
  한다.
* **31.** `python/es/train_act.py`와 Task IR의 `scene.path`가 저장소 기준 상대 경로라서
  `es train`은 저장소 루트에서 실행해야 한다. 설치된 `es`에는 저장소가 없다. 트레이너의 위치는
  아마 문서 값이어야 하거나, `train_act.py`가 바이너리가 싣고 다니는 데이터여야 한다 — T3가
  어차피 그 파일을 건드린다.
