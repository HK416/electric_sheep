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
# 선택 항목이고, 없으면 진짜로 없는 것이다(10절):
# schedule     = { kind = "warmup_cosine", warmup = 250, lr_min = 1e-6 }
# weight_decay = 0.01                 # AdamW의 것. 없으면 torch 자신의 1e-2
# grad_clip    = 1.0                  # 그래디언트 노름 클립. 없으면 끔
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
| 2 | `optimizer.json` | `AdamW` + `lr`. IR 경로에서는 betas/eps와 트레이너에게 실제로 시킨 weight decay도, 그리고 `[run]`이 클립을 두면 `grad_clip`까지(10절) | 사전 |
| 3 | `scheduler.json` | `{"kind":"constant","lr":…}`, 또는 `[run] schedule`이 이름 붙인 `warmup_cosine` 블록(10절) | 사전 |
| 4 | `seed.json` | `[run] seed`에서 온 `global`·`dataloader`, `augmentation`은 unset | 사전 |
| 5 | `dataset.lock` | `es-data::identity`의 content/schema/split, 에피소드·프레임 수, 기록된 `es:task:` 이름 | 사전 |
| 6 | `base_model.lock` | `[policy] base_model`의 *검증된* 출처(IR, 11절), 그것이 없으면 `{"source":"none"}`, 또는 선언된 `vision_backbone` + `pretrained_backbone_weights`(외부) | 사전 |
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
| `base_model.lock.weights_hash`·`.license`, **외부 경로에서만** | *선언된* 출처다. `extra`가 덮지 않는 한 백본은 LeRobot ACT의 기본값이고, 여기서는 그 가중치를 내려받지도 검증하지도 않았다. IR 경로의 것은 검증된다(11절) | — |
| 외부 경로의 `optimizer.json.betas`·`.weight_decay` | `lerobot`의 옵티마이저 블록은 이쪽이 선언할 것이 아니다. 그래서 T4는 그 경로에서 `[run] schedule`·`weight_decay`·`grad_clip`을 아예 거절한다 — 돌지도 않은 스케줄을 선언하는 대신에(10절) | T2 |
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
| 5 | `cargo test -p es-policy --test ir_training lr_schedule_matches_the_golden` | 없음. 인터프리터 절반은 `ES_PYTHON`이 있으면 돌고 없으면 `SKIP`을 찍는다(10절) |
| 6 | `cargo test -p es --test cli train_identity_moves_with_the_schedule` | 없음 |
| 7 | `cargo test -p es-policy --test ir_training -- --ignored the_default_schedule_is_the_old_run` | torch가 있는 `ES_PYTHON` |

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

**패킷 M7/T4**(10절)는 세 곳에서 더 벗어난다.

6. **측정된 네 행은 `es train`이 아니라 `train_act.py`를 직접 돌린 것이므로, 행마다 남는 것은
   `training.lock`이 아니라 JSON 요약과 `--loss-curve`다.** T4 패킷은 행마다 lock을 요구한다.
   `--resident-gpu`는 트레이너의 플래그이고 의도적으로 레시피 필드가 아니다(텐서가 어디 사는지를
   바꿀 뿐 실행이 무엇인지를 바꾸지 않으며, 2절의 규칙은 레시피가 수치를 인용할 실행을
   기술한다는 것이다). 그래서 `es train`으로 측정한 행은 비교 대상인 T3의 기준선과 다른 실행이
   된다. 대신 남긴 것: `~/artifacts/plan-v/m7-t4/{A,B,C,D}.json`과 `*-curve.json`, 그리고 이
   패킷 이후 다시 돌린 오라클 3의 진짜 `es train` lock 하나(`train-ir-training.lock`) — 그
   `scheduler.json` 다이제스트는 T4 이전의 것이다.
7. **`the_default_schedule_is_the_old_run`은 저장소 안에서 이전 스크립트와 비교할 수 없다.**
   그 파일이 트리에 없고, torch 손실 곡선은 정당한 골든이 아니기 때문이다(이 노트의 9절,
   그리고 `ir_training.rs`의 머리말: 한 옵티마이저의 두 실행은 torch 버전을 가로질러 일치하지
   않는다). 트리 안의 테스트는 기본값과 "새 플래그를 전부 기본값으로 명시한 것"을 비교하고
   옵티마이저 블록을 고정한다. `main` 자신의 스크립트와의 비교는 10절의 서버 측정이다.
8. **픽스처 레시피는 스케줄 없이 그대로 둔다.** 패킷이 이 노트에 판단을 맡긴 부분이고, 10절의
   마지막 소절이 이유를 적는다.

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
* **33.** (32는 `visible-learning.md`의 것이다.) 인터프리터가 `blake3`를 임포트하지 못하면
  `lr_curve_hash`는 `null`이다(10절). 이 저장소의 다른 모든 다이제스트는 blake3이고 Rust에서
  계산되는데, 이것만 Python에서 계산된다 — 어떤 학습률을 실제로 적용했는지 아는 것이 트레이너
  뿐이기 때문이다. 트레이너가 적용한 학습률을 `--loss-curve` 옆 파일로 쓰고 `es train`이 해시를
  계산하거나, `blake3`가 `torch`처럼 학습 경로의 선언된 의존성이 되어야 한다
  (`python/es/pyproject.toml`은 USD bake를 위해 이미 그것을 선언한다).
* **34.** lerobot 경로에서는 스케줄을 번역하지 않고 거절한다(10절). ACT의 설정 자체는
  `optimizer_lr`과 `optimizer_weight_decay`를 갖고 있으므로
  (`docs/api-notes/lerobot-config.md`의 "training-only" 행), `[run]`을 `lerobot-train` 0.6.1이
  자기 스케줄을 부르는 이름들로 사상하는 것은 가능하고, 그러면 한 문서가 두 경로를 모두
  기술하게 된다 — 레시피의 요점이 바로 그것이다. 다만 저쪽 플래그 이름을 api-note에 먼저
  고정해야 하고, 그전까지는 `[policy] lerobot.extra`가 탈출구다.

---

## 10. 스케줄 (패킷 M7/T4)

§28.9의 "월클록은 어디로 가는가"는 큰 배치에 대해 한 줄로 끝난다. *선형 스케일링은 발산하는
것으로 측정되었으므로, 필요한 것은 관례가 아니라 스케줄이다.* 이 절은 그 한 줄을 문서 필드
하나, 함수 하나, 골든 하나, 그리고 측정된 네 번의 실행으로 바꾼 것이다.

**트레이너의 다섯 플래그, 문서의 세 필드.** `train_act.py`는 `--schedule
constant|warmup_cosine`, `--warmup-steps N`, `--lr-min F`, `--weight-decay F`(torch 자신의
`1e-2`를 **명시적으로** — `optimizer.json`이 스크립트가 가정한 수가 아니라 실제로 시킨 수를
이름 붙일 수 있도록), `--grad-clip F`(`0`이면 끔)를 얻는다. 레시피는 `[run]`에서 그것들을
이름 붙인다.

```toml
[run]
steps        = 20000
batch        = 64
lr           = 4e-4
schedule     = { kind = "warmup_cosine", warmup = 250, lr_min = 1e-6 }
weight_decay = 0.01                   # 선택. 없으면 torch의 1e-2
grad_clip    = 1.0                    # 선택. 없으면 끔
```

`es train`은 그것들을 트레이너의 명령줄에 올리고, `scheduler.json`을
`{"kind":"warmup_cosine","lr":…,"lr_min":…,"warmup":…,"total_steps":…}`로,
`optimizer.json`을 weight decay와 클립과 함께 쓴다. 따라서 `identity_hash`는 그 하나하나와
함께 움직인다. **`total_steps`는 장식이 아니라 스케줄의 일부다**: 코사인의 주기가 곧 실행의
길이이므로, 같은 `warmup`/`lr_min` 쌍이라도 2,500 스텝과 20,000 스텝은 서로 다른 스케줄이다 —
아래의 C행과 D행이 정확히 그것이다.

**스케줄은 `torch.optim.lr_scheduler`가 아니라 평범한 함수다.**

```
lr(step) = lr * step / warmup                                          step < warmup
         = lr_min + (lr - lr_min) * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))
```

`python/es/train_act.py::lr_at`이 그 세 줄이고 매 스텝 한 번 호출된다. torch 스케줄러가 내는
float 수열은 torch 버전의 구현 세부사항인데, 실행은 자신의 `scheduler.json`으로부터 버전을
가로질러 재현되어야 한다 — 그래서 수열을 여기에 고정한다.
`tests/golden/train/lr_warmup_cosine.json`은 D행 스케줄(`total 20000, lr 4e-4, lr_min 1e-6,
warmup 250`)의 처음 1,000개 값을 담고, 오라클 서버에서 스크립트 자신으로부터 `#[ignore]`된
`generate_lr_golden`이 한 번 생성했다. `ir_training::lr_schedule_matches_the_golden`은 **같은
세 줄의 Rust 재구현**을 그것과 비트 단위로 비교하고, `ES_PYTHON`이 설정되어 있으면 스크립트
자신의 `lr_at`도 비교한다. Rust 쪽은 인터프리터가 필요 없고, 그것이 CI가 스케줄을 판정할 수
있게 해 준다.

두 쪽이 libm을 가로질러 비트 단위로 일치하는가가 열린 위험이었다 — 여기서 초월함수는 코사인
하나뿐이고, 서버의 glibc `cos`가 만든 골든을 Windows 호스트의 MSVC `cos`가 읽는다. 측정:
**1,000개 중 1,000개가 `f64`로 동일**, warmup 구간도 코사인 구간도. 골든 범위에서 코사인의
인자는 `[0, 0.119]`에 머무르고 그 구간에서는 두 libm 모두 올바르게 반올림한다. 감쇠 전체를
덮는 골든이라면 그렇게 운이 좋지 않을 수 있고, 테스트가 단지 단언하는 대신 몇 개가 다른지를
출력하는 이유가 그것이다.

**`constant`이 기본값이고, 기본값은 예전의 그 실행이다.** 서로 다른 세 진술이고, 각각 자기
검사를 갖는다.

1. *명령줄에 아무것도 없다.* 스케줄을 이름 붙이지 않은 레시피는 늘 렌더링하던 계획을 그대로
   렌더링하므로 `tests/golden/train/plan-ir.txt`는 바이트 하나 움직이지 않는다
   (`cli::train_dry_run_plan_is_the_golden`).
2. *정체성에 아무것도 없다.* 새 키들은 레시피가 요구할 때만 `scheduler.json`과
   `optimizer.json`에 들어가므로, T4 이전의 레시피는 자기 `identity_hash`를 유지한다 — 7절에
   기록된 실행은 여전히 같은 실행이다
   (`es-data::a_recipe_without_a_schedule_is_the_run_of_before`,
   `cli::train_identity_moves_with_the_schedule`). 주장이 아니라 측정이다. 이 패킷 이후 서버에서
   다시 돌린 7절의 오라클 3은 `scheduler.json 8847154d…51afe7aa`를 쓰는데, 그 표가 이미 담고
   있던 바로 그 다이제스트다.
3. *산술에 아무것도 없다.* `constant`에서는 `lr_at`을 아예 참조하지 않고 옵티마이저는 자신이
   생성될 때 받은 스텝을 유지한다. `ir_training::the_default_schedule_is_the_old_run`은 CPU에서
   새 플래그 없이 40스텝, 새 플래그를 전부 기본값으로 명시하고 40스텝을 돌려 두 `--loss-curve`
   파일을 **바이트로** 비교하고, 보고된 옵티마이저 블록이 이 노트의 모든 측정이 취해진 그
   블록임을 단언한다.

세 번째는 이 저장소 *안의* 테스트가 완전히 진술할 수 없다. *이전* 스크립트가 트리에 없기
때문이다. 그래서 서버에서 한 번 그것과 맞대어 측정했다. `git show main:python/es/train_act.py`
(main은 `389ef88`, T 트랙 파일들은 `374d42c`의 것과 동일)를 오라클 자신의 실행과 같은 모듈,
같은 bake, 같은 시드, 같은 40스텝으로 돌렸고 — **두 손실 곡선은 바이트 동일**하다.

**lerobot 경로에서는 거절.** `[run] schedule`·`weight_decay`·`grad_clip`은 `train_act.py`의
플래그다. `lerobot-train`은 자기 옵티마이저와 스케줄러 설정을 들고 다니므로, 그쪽에서 이것들을
받아들이면 실행이 적용한 적 없는 스케줄로 `scheduler.json`을 — 따라서 `training_hash`를 — 쓰게
된다. 레시피는 이름을 대고 거절하며 `[policy] lerobot.extra`를 가리킨다(열린 질문 34).

**`lr_curve_hash`.** 실제로 적용된 학습률들을 해시해(f64 리틀엔디언 위의 blake3) 실행 요약에
보고한다. 그래서 한 스케줄의 두 실행을 한 단어로 구별할 수 있고, `constant` 실행의 해시가
`--lr` 40개의 복사본임을 검사할 수 있다. `blake3`는 선택적 임포트다. 스크립트의 계약이
"torch 이상의 패키지는 없다"이므로, 그것이 없는 인터프리터는 학습 실행을 멈추는 대신 `null`을
보고한다(열린 질문 33). 오라클 서버의 `es-lerobot-cuda` venv에는 `pip`이 없어서, 아래 실행들은
옆의 `es` venv에서 `blake3` 패키지를 복사해 `PYTHONPATH`로 가리켜 얻었다.

### 측정 — 오라클 서버, RTX 4090, 2026-09-16

모듈은 이 트리의 `es policy lower`가 내린 데모의 미훈련 번들이고
(`lowering_hash 3d06811c…d8a2d394`, T3의 것), 데이터는 V15의 200 시연 bake 세트
(36,960 샘플, `observation_hash 899c16a9…`)이며, 모든 실행은 `--seed 0 --device cuda
--resident-gpu`, torch 2.11.0+cu129이다. 실행 전 GPU는 비어 있었고(`nvidia-smi`: 56 MiB, 0 %)
기다릴 필요가 없었다.

| 실행 | batch | lr | 스케줄 | 스텝 | 본 샘플 | 월클록 | 1,000스텝당 초 | 초당 샘플 | `final_loss` | 비유한 스텝 |
|---|---|---|---|---|---|---|---|---|---|---|
| A (기준선) | 8 | 1e-4 | constant | 20,000 | 160,000 | **1:47** (107 s) | 5.4 | 1,495 | 0.018901 | 없음 |
| B | 64 | 1e-4 | constant | 2,500 | 160,000 | **0:26** | 10.4 | 6,154 | 0.020714 | 없음 |
| C | 64 | 4e-4 | warmup_cosine, warmup 250, `lr_min` 1e-6 | 2,500 | 160,000 | **0:26** | 10.4 | 6,154 | **0.012060** | 없음 |
| D | 64 | 4e-4 | warmup_cosine, warmup 250, `lr_min` 1e-6 | 20,000 | 1,280,000 | **2:51** (171 s) | 8.6 | 7,485 | 0.004610 | 없음 |

§12.4에 따라 `step/s` 수치는 인용하지 않는다. 두 비율 열은 그 실행 자신의 단위다. 짧은 실행도
같은 약 5초의 시작 비용(모듈 구축과 3.9 GiB 상주 복사)을 1/8의 스텝 수에 나누어 지므로,
B·C의 10.4와 D의 8.6 차이는 전부 그것이다.

**패킷의 질문, 답.** *같은 샘플 수를 본 시점에서 배치 64는 warmup과 코사인으로 배치 8의 손실
이하에 도달하는가?* **그렇다, 그것도 3분의 1만큼**: C는 0.012060으로 A의 0.018901보다 36 %
낮고, **107초가 아니라 26초**다. 그리고 그것을 해낸 것은 배치가 아니라 스케줄이다. B는 같은
160,000 샘플을 같은 64로, A와 같은 lr로 본 실행인데 A보다 *위인* 0.020714에 앉는다. "관례가
아니라 스케줄"이 바로 그 두 행이다.

**아무것도 발산하지 않았고, 그것이 무엇의 증거이고 무엇의 증거가 아닌가.** 어떤 행에도 비유한
스텝이 없다 — 요약이 이제 `first_nonfinite_step`을 보고하므로 이것은 충돌의 부재가 아니라
하나의 수다. 그러나 C·D는 `visible-learning.md` 7.11의 NaN 실행과 *두 가지*가 다르다. 그쪽은
warmup 없이 학습률을 8e-4로 선형 스케일했고, 이쪽은 4e-4에 250스텝의 warmup과 감쇠를 쓴다.
T3는 스케일하지 않은 1e-4에서의 배치 64가 유한함을 이미 보였다. 따라서 여기서 측정된 것은
**이 조합이 안정적이라는 것**이지 warmup만으로 8e-4가 구제된다는 것이 아니다. 그것을 분리하려면
다섯 번째 실행이 필요하고, 하류의 무엇도 그것을 필요로 하지 않는다.

**D는 A와 적합도로 비교할 수 없고**(데이터가 여덟 배다) 여기서 평가하지 않는다(그것은 5차의
U-측정이다). 체크포인트는 `~/artifacts/plan-v/m7-t4/model-D.safetensors`이고 `lr_curve_hash`는
`c01d5185b03a5bad9fbe1708b5582f504aabf90c745910713ca0ccaefc15365c`다. `final_loss 0.004610`은
이 문서들 위에서 IR 그래프 실행이 기록한 가장 낮은 값이다. 그만큼 낮은 적합이 성공하는 정책인지
아닌지가 바로 §28.10의 중단 규칙이 재측정을 위해 남겨 둔 질문이다.

**조심해서 읽을 수 하나.** A행은 `learning-lowering.md` 5.2에서 T3가 측정한 것과 같은
플래그인데 `final_loss 0.018901`을 보고한다 — 그 실행은 0.019026이었고 월클록도 137초 대
107초다. CUDA 학습은 실행 간에 재현되지 않으므로(§28.9 L10) 동일한 두 호출 사이의 0.7 % 차이는
이 표 모든 수의 잡음 바닥이다. 위의 비교가 1 %가 아니라 36 %·76 % 차이인 이유, 그리고 "기본값은
움직이지 않았다"의 비트 단위 오라클이 CPU에 있는 이유가 그것이다.

### 픽스처 레시피는 의도적으로 그대로다

`tests/fixtures/visible-learning/training.toml`은 여전히 스케줄을 이름 붙이지 않는다. 그것은
배치 8에서 측정된 20,000스텝 실행의 커밋된 레시피이고, 그 옆의 계획 골든은 `--dry-run`이 계속
만들어 내야 하는 것이다. 새 필드를 거기서 보이면 둘 다 움직인다. 필드는 위에 문서화되어 있고,
자기 레시피를 직접 만드는 `cli::train_identity_moves_with_the_schedule`이 그것을 실행한다.

## 11. 사전학습된 backbone과 `base_model.lock` (패킷 M7/T5)

spec 19.3은 사전학습된 backbone의 출처가 "결과에 직접 영향을 주므로 프로버넌스에 포함되어야
하며, 라이선스 추적의 근거이기도 하다"고 말한다. 이 패킷 이전까지 IR 경로의 슬롯은
`{"source":"none"}`이었고 외부 경로의 것은 두 필드가 `unset`인 *선언*이었다. 이제 IR 경로에서
그것은 **검증된** 기록이며, 이 절이 검증된다는 것이 무슨 뜻인지 적는다.

레시피가 필드 하나를 얻는다:

```toml
[policy]
bundle     = "runs/collect-001/untrained.esb"
base_model = "~/artifacts/plan-v/m7-t5/resnet18-imagenet1k-v1.safetensors"
```

Learning IR이 `VisionEncoder { pretrained = true }`를 선언하는 번들은 이것을 요구하고, 그렇지
않은 번들은 이것을 거부하며, `lerobot` 경로에서는 아예 거부된다 — 거기서 backbone은 `lerobot`
자신의 것이고 `[policy] lerobot.extra`를 통해 닿는다. 첫 쌍의 양방향 모두 기본값이 아니라
거부이며, 이유는 같다: ImageNet을 원하는데 받지 못한 번들은 그렇지 않다고 말하는 문서 아래에서
처음부터 학습할 것이고, 아무도 읽지 않는 가중치를 이름 대는 레시피는 실행이 갖지 않은 출처를
`base_model.lock`에 넣게 된다 — §28.10 규칙 2가 금지하는 날조다.

**단 1 GPU-초도 쓰이기 전에 세 주장이 일치해야 한다.** `Backbone::verify`는 파일과 그 옆의
`<stem>.lock.json`을 읽고, 이 순서로 확인한다:

| 확인 대상 | 거부되는 경우 | 왜 이 순서인가 |
|---|---|---|
| lock이 이 바이트를 서술하는가 | `lock.blake3 != blake3(file)` | 이 lock 파일이 애초에 이 아티팩트에 관한 것인가 |
| source | `lock.source`가 `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`가 아님 | 라이선스 결정은 *이름 붙은* source에 관한 것이다(§29 행, 오너 2026-09-15) |
| 라이선스 | `lock.license`가 비어 있음 | §19.3은 이 파일을 라이선스 추적의 근거로 삼는다; 빈 슬롯은 아무것도 추적하지 않는다 |
| 고정값 | `blake3(file) != RESNET18_IMAGENET1K_V1_BLAKE3` | 앞의 셋은 lock이 이 바이트에 대한 잘 형성된 기록인지 묻고, 이것은 이 바이트가 저장소가 실제로 측정한 아티팩트인지 묻는다 |

고정값은 `8511928e…9e801899`, `crates/es-data/src/training.rs`의 `pub const` 하나다. 45MB
파일은 결코 커밋되지 않는다 — 오라클 서버의 `~/artifacts/plan-v/m7-t5/`에 있다 — 그래서 그
상수가 저장소가 그것에 대해 아는 전부이며, 이는 `tests/fixtures/mjcf/*.PROVENANCE.json`이 이
프로젝트의 장면이 파생된 업스트림 MJCF를 고정하는 방식과 같다. 그 문자열의 사본은 정확히
하나다: `python/es/fetch_backbone.py`는 자기 것을 갖는 대신 `--expect`로 그것을 *받고*,
`crates/es-policy/tests/backbone_provenance.rs`는 `training.rs`에서 텍스트로 읽어 낸다 —
`es-data`는 레이어 10이고 `es-policy`는 레이어 8이므로(§4.2) 상수를 위로 import할 수 없기
때문이다. `fetch_backbone.py --repin`이 그것을 옮기는 의도적인 방법이고, 상수와 두 설계 노트가
같은 커밋에서 함께 움직여야 한다고 stderr로 말한다.

**슬롯에 무엇이 들어가고, 무엇이 의도적으로 들어가지 않는가.** `base_model.lock`은 `source`,
`url`, `sha256_upstream`, `blake3`, `dropped`, `license`, `license_url`을 싣는다 — *가중치*를
식별하는 필드들이다. 아티팩트 옆의 lock 파일은 그것을 받아온 `torch`와 `torchvision`도
기록하지만 그것들은 **복사되지 않는다**: 같은 업스트림 파일을 받아오는 두 머신은 바이트 단위로
동일한 텐서와 서로 다른 버전 문자열을 쓰므로(측정: torchvision 0.26.0+cu129와 0.29.0+cpu가
같은 blake3를 낸다), 그것을 실으면 받아온 머신이 `identity_hash`에 들어가고 한 레시피가 두
identity를 갖게 된다. 받아온 환경은 아티팩트 자신의 lock 파일과 실행의 `hardware.json`에 속한다.

따라서 `TrainingIdentity.base_model`은 이 경로에서 실제 `source`와 실제 `license`를 갖게 되며,
이것이 §19.3이 요구한 바다: base model의 라이선스 말고는 아무것도 바꾸지 않은 실행은 다른
실행이다.

**트레이너의 줄은 `--init-backbone <path>`를 얻고, 그 외에는 아무것도 얻지 않는다.** `frozen`은
Learning IR의 필드이므로 lowering된 모듈이 그것을 `requires_grad_(False)`로 싣고 `train_act.py`는
여전히 gradient를 요구하는 파라미터 위에 `AdamW`를 만든다 — `learning-lowering.md` 5.3절 참고.
`--frozen` 플래그는 명령줄 위의 IR 결정 사본이 될 것이고, 그러면 계획 골든이 `--dry-run`이 열지
않는 번들에 의존하게 된다.

### 거부는 번호가 아니라 이름으로 한다

패킷은 `TRAIN-0xx`라고 쓴다. `es_data::training`도 `es train`도 숫자 코드를 가진 적이 없다 —
거기의 모든 거부는 필드와 파일을 이름으로 대며, §17.2의 요구는 거부가 식별 가능해야 한다는
것이지 열거되어야 한다는 것이 아니다. 메시지 다섯 개를 위해 고안된 번호 체계는 사용자가
하나뿐인 체계다. 그 다섯은: 자기 가중치를 서술하지 않는 lock 파일(두 다이제스트와 lock의 경로를
이름으로 댄다), 고정된 것이 아닌 아티팩트(두 다이제스트와 옮길 상수를 이름으로 댄다), 빈
라이선스(파일과 §19.3을 이름으로 댄다), `base_model` 없는 `pretrained = true`(플래그와 fetch
명령을 이름으로 댄다), `pretrained = true` 없는 `base_model`(플래그와 주장될 뻔한 것을 이름으로
댄다). 다섯 모두 `cli::train_refuses_a_mismatched_base_model`이다.

### 여기서도 픽스처 레시피는 그대로다

`tests/fixtures/visible-learning/training.toml`은 `base_model`을 이름 붙이지 않는다. 그 번들이
처음부터 학습하는 쪽이고, 그 옆의 계획 골든은 계속 렌더링되어야 하기 때문이다. 실험의 사전학습
쪽은 `tests/fixtures/visible-learning/learning-pretrained.toml`이고, 그것을 가리키는 레시피는
U-측정의 것이지 커밋된 픽스처가 아니다.

---

## 12. 사이클 (패킷 M7/T2)

§13.1은 루프를 그린다 — 수집, 학습, 평가, 관찰 — 그리고 T1이 그중 한 칸을 명령 하나로
만들었다. 나머지 셋은 사람이 경로와 해시를 손으로 꿰는 네 개의 명령으로 남아 있었고,
`loop.jsonl`은 `distill`에서 멈춰 있었다: 데이터셋이 정책을 학습시켰다는 것도, 정책이
판정받았다는 것도 기록하지 않았다. `es loop cycle`은 그 칸 전체를 문서 하나와 원장 하나
아래에 둔다.

```
es loop cycle --recipe <cycle.toml> [--out <dir>] [--dry-run] [--from <stage>]
              [--allow-new-evaluation] [--skip-expert-gate]
```

### 12.1 문서는 단계를 이름 붙일 뿐, 다시 서술하지 않는다

```toml
kind  = "cycle"
scene = "tests/fixtures/mjcf/so101_pick_place.xml"

[collect]                        # 선택적. 재사용하려면 대신 `dataset = "<root>"`
policy   = "runs/collect-001/untrained.esb"
expert   = "so101-pick-place"    # 학습된 정책 자신의 롤아웃이면 생략
episodes = 200
seed     = 1
frames   = true

[train]
recipe = "tests/fixtures/visible-learning/training.toml"   # T1의 레시피, 경로 또는 인라인

[eval]
config     = "tests/fixtures/visible-learning/evaluation.toml"
checkpoint = "last"              # 또는 레시피의 `checkpoint_at`이 쓰는 mark
jobs       = 6
frames     = true

[showcase]                       # 선택적. `render` 피처가 필요하다
cell    = "nominal-00"
eye     = [0.66, -0.46, 0.52]
look_at = [0.14, -0.04, 0.04]
fov     = 36
width   = 1280
height  = 720
```

2절이 레시피에 대해 말한 규칙이 여기에도 그대로 적용된다: **사이클은 단계가 이미 소유한
파라미터를 나르지 않는다**. `[train]`은 T1의 레시피를 경로로 가리킬 뿐 그 필드의 사본이
아니고, `[eval]`은 Evaluation IR을 경로로 가리키므로 `evaluation_hash`는 그 문서 자신의
것이지 그것을 다시 그린 것이 아니다. 사이클이 *덮어쓰는* 유일한 것은 학습 레시피의
`[dataset] root`/`frames`다: 그것들은 이 사이클의 수집 출력이 된다. 다른 디렉터리를 읽은
학습을 가진 사이클은 아무것도 연결하지 않기 때문이다(§13.3). 그 덮어쓰기는 파일을 고쳐
쓰는 대신 중첩된 계획에서 보인다.

**모든 단계는 그 명령이 이미 그러한 함수다** — `cmd::r#loop::collect`, `cmd::eval::run`,
`cmd::train::run`, `cmd::showcase::run` — 자기 계획이 출력하는 바로 그 단어로 in-process
호출되므로, 출력된 줄과 실행된 단계가 어긋날 수 없다. 그것들이 이미 띄우는 것(물리
서브프로세스, 트레이너, `--jobs` 워커)만이 프로세스다. `crates/es/src/cmd/cycle.rs`가
얇은 이유는 T1과 같다: 문서와 계획과 원장 단계는 `es_data::training`과
`es_data::collect`에 있고, 그것들은 헤드리스이며 단위 테스트되어 있다.

### 12.2 단계 계획은 골든이다

`--dry-run`은 단계마다 한 줄을 출력한다. `<out>` 아래의 모든 경로는 그것에 상대적으로,
모든 구분자는 `/`로, T1의 계획은 `train` 줄 아래에 들여쓰기로 — 학습 계획을 레시피만의
속성으로 만드는 바로 그 세 규칙이다(3절):

```
# cycle: collect -> expert-gate -> train -> eval -> showcase
es loop collect --policy runs/collect-001/untrained.esb --scene .../so101_pick_place.xml --episodes 200 --seed 1 --out collect/ds --frames collect/frames --expert so101-pick-place
es eval run --config .../evaluation.toml --policy runs/collect-001/untrained.esb --scene .../so101_pick_place.xml --out eval-expert --jobs 6 --frames eval-expert/frames --expert so101-pick-place
es train --recipe .../training.toml --out train
  # route: ir
  es dataset bake --policy runs/collect-001/untrained.esb --out train/baked --frames collect/frames collect/ds
  ...
es eval run --config .../evaluation.toml --policy train/checkpoints/20000.esb --scene .../so101_pick_place.xml --out eval --jobs 6 --frames eval/frames
es video showcase --run eval --scene .../so101_pick_place.xml --out showcase --cell nominal-00 --eye 0.66,-0.46,0.52 --look-at 0.14,-0.04,0.04 --fov 36 --width 1280 --height 720
```

`tests/golden/train/plan-cycle.txt`가 그것을 바이트 단위로 고정한다. 그것을 만드는 데
디스크의 어떤 것도 읽지 않는다 — 데이터셋도, 번들도, Python도 — 그것이 그 어느 것도 없는
기계의 CI에서 판정 가능하게 만드는 것이다. 일어나지 않은 실행은 디렉터리조차 쓰지 않는다.

### 12.3 두 개의 거부가 이 명령의 요점이다

**하니스는 전문가를 먼저 통과시킨다**(§28.9 규칙 1, M5-R1). `[collect] expert`가 설정되면
사이클은 *같은* `[eval] config`로 `es eval run`을 통해 전문가를 돌린다 — 무엇이든 학습되기
**전에** — 그리고 전문가의 acceptance가 실패하면 사이클을 멈춘다. 전문가가 통과하지 못하는
하니스는 어떤 정책도 통과하지 못하는 하니스이고, GPU 한 시간은 그것을 알아내는 비싼
방법이다. 게이트의 리포트는 정책의 것 옆 `<out>/eval-expert/`에 보관된다. 증거는 그 뒤에
오는 실행보다 오래 살아남아야 하기 때문이다. `--skip-expert-gate`는 그래도 실행하며 원장에
일탈을 기록하므로(`expert_gate = "skipped (--skip-expert-gate)"`), 게이트 없이 학습한
사이클이 게이트를 통과한 사이클처럼 보이지 않는다.

**움직인 `evaluation_hash`는 이름을 들어 거부된다**(§13.3). `<out>`은 반복 2에서 재사용되므로
`<out>/loop.jsonl`이 이미 `evaluate` 단계를 담고 있을 수 있다. 새 것의 `evaluation_hash`가
다르다면 원장이 초대하는 비교는 비교가 아니다 — §13.3의 "데이터·정책만 바꾸면서 평가 조건을
고정하는 것이 규율" — 그래서 사이클은 **두 해시를 모두 출력하며** GPU에 손대기 전에 거부한다
(평가 조건은 문서의 속성이므로 `--dry-run`에서도 검사가 돈다). `--allow-new-evaluation`이
새 비교를 시작하는 의도적 행위다. 반복 2부터는 끝에서 두 리포트에 `es eval compare`가 돈다:
새 것이 덮어쓰기 전에 이전 `report.json`을 `report-prev.json`으로 옮긴다.

### 12.4 `--from <stage>`는 재개하되, 먼저 검사한다

`--from collect|train|eval|showcase`는 앞선 단계를 건너뛰고 그 출력을 `<out>` 아래에서
읽는다. 그것들이 없거나 원장과 어긋나면 거부한다: 데이터셋을 다시 계산한 `content`를 원장의
`collect` 단계와, 디스크의 체크포인트 번들과 `training.lock` 안의 그 `policy_hash`를,
`--from showcase`면 `eval/report.json`을. 단계들이 한 사이클이 아닌 재개된 사이클이야말로
이것이 막으려는 실패 양상이다.

### 12.5 원장이 루프의 끝까지 닿는다

`loop.jsonl`은 `train`·`evaluate` 단계와 `es_data::check_chain`을 얻는다. 둘 다
`docs/design/learning-loop.ko.md` 4.1절에 서술되어 있고, 거기가 그것들의 집이다. 사이클은
각각을 데이터셋 root의 원장과 `<out>/loop.jsonl`에 추가하고, 실행이 끝난 뒤 `check_chain`을
한 번 호출한다. 실제 사이클 하나의 원장은 순서대로:

```
collect  ->  evaluate (전문가 게이트)  ->  train  ->  evaluate (정책)
```

### 12.6 패킷으로부터의 일탈 — `es eval run --expert`

패킷은 `eval.rs`를 **오직** 진입점을 `pub(crate)`로 노출하기 위해서만 건드리도록 허용한다.
구현은 플래그도 하나 추가했다: `es eval run --expert <name>`은 번들의 가중치 대신 스크립트된
시연자가 운전하게 한다(`crates/es/src/cmd/loop.rs`의 `ExpertPolicy`·`SeenState`,
`crates/es/src/cmd/eval.rs`를 통해 연결). 이것은 변명 대신 일탈로 기록한다. 다만 그것 없이는
게이트가 존재할 수 없다: 패킷 자신의 spec이 게이트를 "같은 `[eval].config`로 `es eval run`을
통해 전문가를 돌린다"라고 말하는데, T2 이전에는 `es eval run`을 정책의 가중치 외의 것에
돌릴 방법이 없었다. 스캐폴드 자체는 새롭지 않다 — `expert_passes_the_evaluation_harness`는
패킷 M5/V6 이래로 전문가로 하니스를 운전해 왔고, T2는 그것을 테스트 파일에서 그것이 필요한
명령으로 승격시켰을 뿐이다. `ExpertPolicy`는 INV-17의 일곱 중 하나인 `PolicyRuntime`의
구현이지, 여덟 번째 확장점이 아니다.

플래그 *없이* 실행할 때 달라지는 것: 없다. 기존 eval 테스트는 전부 그대로이고, 그것을
넘기지 않는 모든 경로에서 `--expert`는 `None`이며, `TorchRuntime` 로드는 그것이 `Some`일
때만 건너뛴다(전문가는 가중치를 로드하지 않는다 — `es loop collect --expert`가 이미 하는
것과 같은 거래다). `--frames` 없는 `--expert`는 usage 오류다. 전문가는 프레임 소스에게
건네지는 상태에서 큐브의 pose를 읽고, 이 경로에서 정책에게 특권적 상태를 주는 것은 그것
말고 없기 때문이다.

**리뷰가 볼 것 하나.** 전문가의 `reset`이 기준으로 삼는 에피소드 경계는 **정확한 0 속도
휴리스틱**으로 검출된다: `Env::reset`은 Task IR의 `Randomization` 노드가 큐브의 pose를 쓰기
전에 `qpos`/`qvel`을 0으로 채우고, 에피소드의 첫 tick은 항상 프레임 소스에 도달하므로(거기서는
관측 링이 비어 있어 `observation_delay`가 그것을 버릴 수 없다), 모든 속도가 *정확히* `0.0`인
상태는 그 tick이고 다른 어떤 tick도 아니다. `Seen::episodes`의 `ponytail:` 주석이 그 천장을
이름 붙인다. 모든 속도가 정확히 0인 에피소드 중간 상태는 전문가의 스테이지 머신을 재시작시킬
텐데, 그것은 게이트를 조용히 통과시키는 대신 그 에피소드를 시끄럽게 실패시킨다 — 게이트가
틀리기에 안전한 방향이다 — 그러나 진짜 빈틈은 러너에 에피소드 훅이 아예 없다는 것이다.
**이것은 M7 리뷰 항목이다**: `es_eval::Evaluation`이 에피소드 경계 콜백을 얻거나(`es loop
collect`가 이미 가진 intervener 훅), 아니면 게이트가 그 휴리스틱을 기록으로 남긴 채
받아들이거나.

### 12.7 오라클

| # | 명령 | 필요한 것 |
|---|---|---|
| 1 | `cargo test -p es --test cli cycle_dry_run_plan_is_the_golden` | 없음 |
| 2 | `cargo test -p es-data loop_train_and_evaluate_steps_chain` | 없음 |
| 3 | `cargo test -p es --test cli cycle_refuses_a_moved_evaluation_hash` | 없음 |
| 4 | `cargo test -p es --test cli cycle_runs_the_expert_through_the_harness_first -- --ignored` | `ES_PYTHON`(torch, mujoco), `--features render` |
| 5 | `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T2-loop-cycle.md` | 없음 |

오라클 서버(RTX 4090, `ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`, `--features render`)의
오라클 4, 33.27초:

```
RAN cycle_runs_the_expert_through_the_harness_first: expert gate success_rate 1, policy success_rate 0 at policy_hash b58893a9a48a9da2020a4414baa87db3517bc7f3fc7ac93130d31041510fd95f
```

게이트는 시드 둘로 자른 하니스를 통과하고, 40스텝 정책은 통과하지 못한다 — 40스텝 정책이란
그런 것이다. 그 비대칭이 이 오라클의 주제다: 게이트가 돌았고, **먼저** 돌았으며, 원장이
`collect -> evaluate -> train -> evaluate`를 연결했다.

### 12.8 측정 — 사이클 하나 전체, 오라클 서버, RTX 4090, 2026-09-21

데모 자신의 문서들 위에서 돌린 `es loop cycle`: 프레임을 남긴 200 에피소드 전문가 수집,
커밋된 `evaluation.toml` 위의 게이트, 20,000스텝 IR 경로(배치 8, lr 1e-4, seed 0,
`extra = ["--resident-gpu"]`, `device = "cuda"`), 같은 `evaluation.toml`을 `--jobs 6`과
프레임으로, 그리고 V19b의 카메라로 찍은 `nominal-00` 쇼케이스. 명령 하나, 문서 하나, 원장
하나. `~/artifacts/plan-v/m7-t2/run.sh`가 그 호출이고, `cycle.log`가 stdout의 모든 줄에
타임스탬프를 붙인다.

| 단계 | 월클록 | 무엇이 돌았나 |
|---|---|---|
| `collect` | **5:00** | 200 에피소드, 103,881 프레임, `success 200 / failure 0 / timeout 0`. safety 4,245 clamped, 1,991 fallback |
| `expert-gate`(`es eval run --expert`) | **2:21** | 6 스위트 × 16 에피소드, `--jobs 6`, 프레임. 69,380 프레임 |
| `train`(T1의 IR 경로, 중첩) | **2:26** | 그중 `dataset bake` 0:24, `policy lower` 1초 미만, **`train_act.py` 2:00**(20,000스텝), `policy pack` ×3 약 1초 |
| `eval` | **20:24** | 6 스위트 × 16 에피소드, `--jobs 6`, 프레임. 171,116 프레임 |
| `showcase` | **0:08** | `nominal-00` 1,800 tick, 1280×720 lambert, 4.7 ms/frame |
| **합계** | **30:19** | §28.9의 중단 규칙은 사이클 하나 30분 미만: **19초 차로 놓침** |

`training_hash 6c81756c…`, `identity_hash 8914b1d6…`, `lowering_hash 3d06811c…`,
`initial_loss 0.052309`에서 `final_loss 0.014186`. 판정된 체크포인트는
`checkpoint.20000 = policy_hash ec8379a9…`이고 그것이 `evaluate` 단계의 `policy_hash`다 —
실제 데이터 위의 체인 속성. `es_data::check_chain` 통과:
`ledger: …/loop.jsonl (chained)`.

**게이트는 통과했고, 정책은 통과하지 못했다.** 둘 다 원장에 나란히 있다:

| 스위트 | 전문가 게이트 `success_rate` | 정책 `success_rate` | 정책 `envelope_violation_rate` |
|---|---|---|---|
| nominal | **1.000** | **0.000** | 0.949 |
| light_intensity | 1.000 | 0.000 | 0.945 |
| light_direction | 1.000 | 0.000 | 0.947 |
| observation_delay | 1.000 | 0.000 | 0.904 |
| torque_noise | 0.000 | 0.062 | 0.956 |
| backlash | 1.000 | 0.000 | 0.937 |

acceptance 기준은 `nominal success_rate >= 0.5`이므로 사이클은
`FAILED suite=Some("nominal") metric=success_rate observed=0`과 함께 1로 끝난다. **그것은
이 패킷의 결함이 아니라 측정이다** — T2의 주제는 문서 하나가 사이클을 돌리고 그것을 연결할
수 있는가이고, 그렇게 했다. 정책의 수는 §28.10의 U-측정이 맡은 것이며, 전문가의 0.04 대비
0.95라는 `envelope_violation_rate`가 어디를 볼지 말해 준다: 학습된 chunk가 스무 tick 중
열아홉에서 Deployment IR의 envelope 바깥에 있고, 따라서 팔이 실제로 따르는 것은 Safety
Plane이다. `final_loss 0.014186`인데 한 번도 성공하지 못하는 정책은 10절이 D행에 대해
지적한 바로 그 불일치다 — 그만큼 낮은 fit이 아직 작동하는 정책은 아니다. 이 실행은
`base_model`(11절)을 쓰지 않았고 T6의 augmentation도 트리에 없었으므로, IR 경로에 대한
판결이 아니라 U-측정이 개선해 나갈 *바닥*이다.

**이 측정이 §28.9의 "월클록이 어디로 가는가"에 대해 말하는 세 가지.**

1. **학습은 더 이상 지배적인 항이 아니다.** §28.9는 배치 8에서 20,000스텝에 10:53을
   기록하고 배치 lowering의 샘플별 루프를 근본 원인으로 지목한다. T3가 그것을 제거했다
   (`lowering_hash 3d06811c…`): 여덟 배의 데이터(50이 아니라 200 에피소드) 위에서 같은
   20,000스텝이 이제 **2:00**이 걸린다. §28.9 사다리의 9번 칸은 끝났다.
2. **평가가 지배적인 항이고, 그 월클록은 정책이 얼마나 좋은가의 함수다.** *같은* 하니스 —
   6 스위트, 96 에피소드, `--jobs 6`, 프레임 — 가 전문가에게는 2:21, 정책에게는 **20:24**가
   걸렸다. 그 사이에 코드는 하나도 없는데 8.7배 차이다. 성공한 에피소드는 큐브가 통에
   들어가면 끝나고, 실패하는 에피소드는 1,800 tick을 모두 돈다. §28.9의 이 스위트에 대한
   5:49는 가끔 성공하는 정책 위에서 측정된 것이므로 *최선*의 수이고, 새 사이클이 무는 것은
   최악의 경우다. 이제 이 경로에서 유효한 유일한 지렛대는 에피소드 단위 샤딩
   (§28.9 10번 칸 / T8)이다.
3. **수집도 같은 세금을 낸다.** 이 실행의 첫 시도는
   `~/artifacts/plan-v/v15/untrained.esb`를 썼는데, 그 `deployment_hash 3b2ad568…`은 커밋된
   `f2f9a510…`(80)이 아니라 V18 이전의 envelope(`acceleration_max 20`)이다. 200 에피소드가
   전부 timeout이 났고(`success 0 / timeout 200`, 360,000 스텝 중 284,163이 fallback) 수집에
   **5분이 아니라 17분**이 걸렸다. 그 실행은 폐기하고, 커밋된 네 문서를 모두 나르는
   `~/artifacts/plan-v/v18/untrained-L80.esb`로 다시 돌렸다. 기록해 둘 가치가 있는 이유는
   **전문가 게이트가 바로 그것을 잡아냈을 것**이기 때문이다: 불일치를 발견했을 때 게이트는
   이미 돌고 있었고, 전문가가 통과할 수 없는 하니스 위에서 학습을 거부했을 것이다 — 그것이
   §28.9 규칙 1의 논지 전부이며, 첫 실제 실행에서 묻지도 않았는데 도착했다.

`~/artifacts/plan-v/m7-t2/`에 보관: `loop.jsonl`, `training.lock`, `report-policy.json`,
`report-expert-gate.json`, `cycle.log`, `cycle.toml`, `training.toml`, `run.sh`, 그리고
`run/` 트리 전체(27 GB: 프레임, 트래젝터리, 체크포인트 번들 셋).

## 13. 증강 (패킷 M7/T6)

spec 7.3는 M1 이래로 `Augment` 노드 가족을 가지고 있었고, 이 패킷 전까지 **모든 경로가 그것을
무시했다**: bake는 Release 플랜을 돌리는데 거기서 `training_only` 노드는 identity이고,
`train_act.py`는 Observation IR을 본 적이 없으며, `augmentation.json`은 지금까지 돌아간 모든
실행에 대해 `{"kind": "none"}`이라고 말했다. 이 절은 선언된 서브그래프가 실제로 일어나게
만드는 경계다 — 학습에서만, 하나의 시드로부터, 아이덴티티로 기록되어 — 그러는 동안 평가
경로는 어제 내던 바이트를 그대로 낸다.

### 13.1 경계, 그림 하나로

```
ImageInput -> Dequantize -> Normalize -> Pad(4) -|- Crop{Random 96x96} -> Augment{ColorJitter} -> rgb_overhead
                                                 |
                                               체인 경계
```

경계 왼쪽은 늘 그랬던 Observation IR이고 `CpuPlan`이 돌린다 — 구현은 하나, 추론이 돌리는
바로 그것이다(패킷 M5/V2b). 오른쪽은 *샘플별*이므로 한 번 구워둘 수 없다:

* `es dataset bake --for-training`은 그 포트를 경계에서 쓴다 — 데모에서는
  `[frames, 3, 96, 96]`이 아니라 `[frames, 3, 104, 104]` — 그리고 체인을 `manifest.json`에
  기록한다;
* `es train`은 같은 체인과 그 시드를 spec 19.3의 `training/augmentation.json`에 쓰고,
  `--augmentation`으로 트레이너에 넘긴다;
* `python/es/augment.py`가 샘플마다, 옵티마이저 스텝마다 그것을 적용한다;
* **평가는 그중 아무것도 적용하지 않는다**(INV-15). Release 플랜은 pad와 crop을 `Pad(4)`
  다음 *중앙* 크롭으로 lowering하고, 대칭 패딩에서 그 합성은 증강되지 않은 이미지와 비트
  단위로 같다 — `crates/es/tests/cli.rs::dataset_bake_for_training_writes_the_chain`이 두
  bake를 파일 통째로 비교한다.

`docs/design/observation-lowering.ko.md` 3.1절이 컴파일러 쪽 절반이다: 어떤 노드가 체인 위에
오는지, Release 플랜이 그것들로 무엇을 하는지, 그리고 왜 GPU 경로가 `Pad`를 이름으로
거부하는지.

### 13.2 문서와 레시피

레시피에 선택적 필드 하나가 생긴다:

```toml
[run]
seed              = 0
augmentation_seed = 7    # 선택; 없으면 `seed`
```

분리되어 있는 이유는 분리 가능하기 때문이다 — 나머지가 동일한 실행에서 증강만 다시 뽑는 것은
다른 실행이고, spec 19.3은 그것에 고유한 `seed.json` 슬롯을 준다. 두 슬롯은 정책의
Observation IR이 체인을 선언할 때에만 실제 값이 된다:

| 슬롯 | 체인 없음 | 체인 있음 |
|---|---|---|
| `seed.json.augmentation` | `{"unset": true}` | 그 숫자 |
| `augmentation.json` | `{"kind": "none"}` | `{"kind": "observation-ir", "observation_hash", "seed", "chains"}` |
| bake 스텝 | 이전과 같음 | `... --for-training <root>` |
| 트레이너 스텝 | 이전과 같음 | `... --augmentation <out>/training/augmentation.json` |

그래서 이 패킷 이전에 쓰인 레시피는 같은 아홉 개의 사전 슬롯, 같은 `identity_hash`, 같은
렌더된 플랜을 가진다 — `tests/golden/train/plan-ir.txt`는 움직이지 않았고,
`train_identity_moves_with_augmentation`이 증강되지 않은 절반을 명시적으로 주장한다.

의도적인 구멍 하나: IR 경로에서 `--dry-run`은 번들을 열지 않으므로(그것이 번들도 데이터셋도
없는 기계에서 플랜을 출력할 수 있게 하는 것이다, 패킷 M7/T1 오라클 1) 증강된 레시피의 dry
run은 두 플래그가 *없는* 플랜을 출력한다. 실제 실행은 번들을 먼저 열고, `config.json`은 실제로
돌아간 줄을 기록한다.

### 13.3 RNG: 주소로 지정되며, 진행되지 않는다

모든 추첨은 그것이 쓰이는 자리의 순수 함수다(spec 3.4는 전역 RNG를 금지한다). 믹서는
Murmur3의 `fmix32` — `crates/es-render/src/rng.rs`가 이미 패스 트레이서에 쓰는 정수 연산 열
줄 — 이고 키는 좌표 다섯 개다:

| 좌표 | 타입 | 어디서 오는가 | 무엇을 분리하는가 |
|---|---|---|---|
| `augmentation_seed` | `u64`, `u32` 둘로 접어 넣음 | `[run] augmentation_seed`, 없으면 `[run] seed` | 한 레시피의 두 실행 |
| `sample_index` | `u32` | 트레이너 자신의 전역 샘플 인덱스(`order[cursor]`, 배치 안의 위치가 아니다) | 한 배치 안의 두 샘플 |
| `step` | `u32` | 옵티마이저 스텝 | 같은 샘플을 두 번 볼 때 |
| `node_index` | `u32` | Observation IR **노드 id** | 두 노드, 그리고 두 포트의 체인 |
| `draw` | `u32` | 한 노드 안에서 `0, 1, …` | 크롭의 x와 y |

`key(seed, sample, step, node)`는 `mix32` 다섯 라운드이고, `uniform(key, i)`는 한 라운드 더로
상위 24비트를 취해 값이 `f32`에서 정확하고 결코 1.0에 닿지 않게 한다. **증강 어디에서도
`torch.Generator`를 쓰지 않으며**, 그것이 핵심이다: 그 스트림은 torch 버전의 구현 세부이므로
다른 torch에서 재현한 실행은 조용히 다른 증강을 보게 되고 `augmentation.json`은 일어나지 않은
일을 기술하게 된다.

구현은 둘이고 `f32`에서 비트 단위로 일치한다:
`crates/es-policy/tests/ir_training.rs`가 인터프리터 없이 체인 전체를 Rust로 다시 유도해
`tests/golden/train/augment_seed0.json`과 비교하며, 그 골든은 `python/es/augment.py` 자신으로
한 번 생성되었다.

**여기서 비트 단위가 합리적인 요구인 이유.** 모든 스칼라는 `f64`로 계산되어 텐서에 닿기
*전에* `f32`로 반올림되므로, 원소별 연산은 두 피연산자가 모두 정확히 표현 가능한 단일 IEEE
`f32` 연산이다 — 중간값을 넓혀 한 번 반올림하는 커널과 그러지 않는 커널이 같은 비트를 낸다.
잘못될 수 있었던 두 자리는 가정이 아니라 측정되었다(10절이 `cos`에 적용한 그 규율):

* **Box-Muller의 `ln`과 `cos`** — 한쪽은 torch의 벡터화된 `f64` 커널, 다른 쪽은 Rust의
  `std`. `f64`에서의 마지막 비트 불일치는 상대적으로 ~2⁻⁵³로 `f32` ulp보다 훨씬 아래이므로
  캐스트가 그것을 흡수한다 — 그리고 골든은 960개 값 전부에서 흡수된다고 말한다.
* **contrast의 평균** — 이 파일의 유일한 리덕션: torch의 `x.double().mean()` 대 Rust의 순차
  `f64` 합. 같은 논증, 같은 측정. 원리적으로 *훨씬* 큰 이미지에서 어긋날 수 있는 유일한
  자리이고 — 천장은 두 합산 순서가 상대적으로 ~2⁻⁵³ 다르다는 것, 그것이 드러나려면 f32
  반올림이 정확히 경계에 떨어져야 한다 — 만약 그런 일이 생기면 고치는 방법은 평균을 양쪽에서
  순서가 정해진 `f32` 합으로 정의하는 것이다.

### 13.4 네 종류와 각각의 상태

| 종류 | 상태 | 트레이너가 하는 일 |
|---|---|---|
| `RandomCrop { width, height }` | **구현됨** | `[0, W-w] × [0, H-h]`에서 균일 정수 오프셋, 추첨 둘; Release 플랜의 중앙 크롭은 오프셋을 고정한 같은 사각형이다 |
| `ColorJitter { brightness, contrast }` | **구현됨** | `x * (1 + u·b)`, 그다음 `(x - mean) * (1 + u·c) + mean`, `u ∈ [-1, 1]`, 추첨 둘 |
| `ColorJitter { saturation, hue }` | **이름으로 거부** | 둘 다 이 파일이 가지고 있지 않은 색 모델이 필요하다(`hue`는 HSV 회전). 0이 아닌 값은 노드 이름과 함께 실행을 멈춘다; 조용히 무시되는 파라미터가 더 나쁘다 |
| `GaussianNoise { sigma }` | **구현됨** | 가산, **원소마다** 균일 추첨 둘로 Box-Muller, `f64`, `f32`로 한 번 캐스트 |
| `RandomErasing { probability }` | **이름으로 거부** | 이 패킷이 구현하지 않는다 |

`GaussianNoise`가 비싼 쪽이다: 원소당 추첨 둘이면 샘플마다 스텝마다 `2 × C × H × W` 인덱스
텐서가 필요하다. 데모의 체인은 그것을 쓰지 않으므로 아래의 측정된 실행은 비용을 전혀 치르지
않는다; 쓰는 문서라면 증강이 스텝의 눈에 띄는 비율이 될 것을 예상해야 한다.

### 13.5 U-측정을 위한 문서

`tests/fixtures/visible-learning/observation-augmented.toml`은 커밋된 `observation.toml`에
이미지 경로 위 노드 셋을 더한 것이고,
`cargo test -p es --test cli -- --ignored generate_augmented_observation_fixture`로 재생성된다:

| 문서 | `observation_hash` |
|---|---|
| `observation.toml`(커밋됨) | `899c16a90033eeb406f328060bbd54ef0632f943a2059db7b3f7c369ee218d81` |
| `observation-augmented.toml` | `cc437a2418c36ac003c68258cc017a3783d9b2d876029463e87a9d38d194fb3e` |

출력 포트는 커밋된 문서의 것과 정확히 같은 `PortType`을 지닌다 — `3×96×96`,
`Normalized{0,1}`, 같은 `ImageSpec` — 그래서 `learning.toml`은 손대지 않았고
`learning_hash`는 움직이지 않는다. **그러나 `evaluation.toml`은 `observation`을 명명하므로**,
이 문서로 학습된 정책을 겨냥한 Evaluation IR은 *다른* `evaluation_hash`다(spec 13.3):
U-측정은 증강된 해시를 자신의 평가 문서에 써넣고 새 아이덴티티를 의도적으로 받아들여야 한다
(`es loop cycle`은 `--allow-new-evaluation` 없이는 움직인 `evaluation_hash`를 거부한다,
12.3절). 평가가 보는 *픽셀*에 대해서는 아무것도 바뀌지 않는다 — 그것이 13.1이 보장하는
바다 — 그러나 그것을 명명하는 문서는 바뀐다.

### 13.6 패킷과 달라진 점

1. **`Crop { CropMode::Random }`이 체인 위에 있고, 데모 문서는 `Augment { RandomCrop }`
   대신 그것을 쓴다.** 강제된 것이며, 이 발견은 리뷰의 시간을 쓸 가치가 있다: `es-ir`의
   `image_out`은 `Augment` 포트에서 들어오는 `ImageSpec`을 유지하므로
   `Pad(4) -> Augment{RandomCrop 96x96}`은 전파가 `104×104`라고 말하는 자리에서 `96×96`을
   광고하고, `es_ir::cross`의 `TYPE-020`이 번들이 만들어지기 전에 거부한다. `es-ir`은 이
   패킷에게 금지된 땅이다. `CropMode::Random`은 — IR 자신의 말로 "샘플별 오프셋이며 명목
   기하는 중앙의 것" — `ImageSpec::cropped`를 통해 전파되고 검증되므로 픽스처가 그것을 쓰고,
   `Augment{RandomCrop}` lowering은 문서가 그것을 담을 수 있게 되는 날을 위해 그 옆에
   구현되고 테스트된다. **M7 리뷰를 위한 열린 질문:** `image_out`이
   `Augment { RandomCrop }`에도 `Crop { Random }`이 받는 크롭된 기하를 주어야 하는가?
2. **`crates/es-compile/src/exec.rs`가 패킷의 `## context` 글롭에 추가되었다.** `Op::Pad`는
   `CpuPlan::run`의 match가 있는 곳에서 실행되어야 한다; 대안인 `kernels.rs` 항목은
   `KERNEL_IDS`에 덧붙이는 일이고 저장소의 모든 플랜의 `compiler_hash`를 움직인다 — 이 패킷이
   스스로 금지한 골든 이동이다. 그 결과로 `Pad`에는 커널 id가 없고 GPU 경로는 그것을 이름으로
   거부한다(observation-lowering.ko.md 3.1).
3. **오라클 3은 테스트 하나가 아니라 둘이다.** `es-eval`이 bake를 소유하고 `es`가 manifest를
   소유한다 — `es-eval`은 레이어 10이고 파일을 쓰지 않는다 — 그래서
   `es-eval::bake_for_training_writes_the_boundary`가 경계 모양과 바이트 동일한 평가 프레임을
   주장하고, `es::dataset_bake_for_training_writes_the_chain`이 `manifest.json`을 주장하며 두
   bake를 safetensors 파일 통째로 비교한다.
4. **`augmentation.json`의 작성자는 `es-data`에, 독자는 `augment.py`에 있다.**
   `crates/es-policy/tests/ir_training.rs`는 그 파일의 자기 사본을 쓰므로(레이어 8은 레이어
   10에 의존할 수 없다) 실제 파일의 *모양*은 대신 `crates/es/tests/cli.rs`에서 고정된다. 양
   끝이 모두 주장되지만, 한 자리에서 주장되지는 않는다.

### 13.7 오라클

| # | 커맨드 | 무엇을 판정하는가 |
|---|---|---|
| 1 | `cargo test -p es-policy --test ir_training augmentation_matches_the_golden` | Rust 재구현이 `tests/golden/train/augment_seed0.json`과 `f32`에서 비트 단위로 같다; `ES_PYTHON`이 있으면 `augment.py`도 같다 |
| 2 | `cargo test -p es-compile a_training_only_random_crop_is_a_centre_crop_in_release` | 패딩 후 중앙 크롭한 Release 플랜이 증강되지 않은 픽셀을 재현하고, 크롭이 `Crop`의 intrinsics 변환을 지닌다 |
| 3 | `cargo test -p es-eval bake_for_training_writes_the_boundary` / `cargo test -p es --test cli dataset_bake_for_training_writes_the_chain` | 경계에서 `104×104`와 manifest 안의 체인; 플래그가 없으면 `96×96`과 증강되지 않은 문서의 바이트 |
| 4 | `cargo test -p es --test cli train_identity_moves_with_augmentation` | 아이덴티티 슬롯 둘 다 실제 값이고 둘 다 움직이며, 증강되지 않은 레시피의 것은 원래 자리에 있다 |
| 5 | `cargo test -p es-policy --test ir_training -- --ignored augmented_training_runs` | 증강된 문서에서 40스텝; 유한한 loss; 한 시드에서의 두 실행은 바이트 동일한 곡선, 두 시드는 아니다 |
| 6 | `cargo xtask ci`, `cargo xtask check-scope docs/packets/M7/T6-augmentation.md` | 골든 변경 0건, 범위 위반 없음 |

골든은 `generate_augment_golden`이 생성하며, 그것은 `#[ignore]`가 붙어 있고 **동시에**
`ES_GENERATE_GOLDENS=1`이 설정되지 않으면 실행을 거부한다 — `cargo test -- --include-ignored`는
워크스페이스의 모든 ignored 테스트를 돌리고, M7 리뷰가 바로 그것이 아무도 건드릴 생각이 없던
골든을 다시 쓰는 것을 발견했다.

### 13.8 측정 — 오라클 서버, RTX 4090, 2026-09-21

증강된 문서의 20,000스텝 실행을 10절의 행-D 설정으로, `es train`을 통해: 배치 64, lr 4e-4,
`warmup_cosine` warmup 250 `lr_min` 1e-6, 시드 0, `--resident-gpu`, `device = "cuda"`,
torch 2.11.0+cu129, V15의 200개 시연 위에서(`~/artifacts/plan-v/v15/ds-train`, 36,960 샘플).
실행 전 GPU는 유휴 상태였다(56 MiB, 0 %). 커맨드는 하나이고, bake는 그 스텝 중 하나이며,
그것이 `--for-training` bake다.

| 스텝 | 벽시계 | 무엇이 돌았나 |
|---|---|---|
| `dataset bake --for-training` | **0:10** | 200 에피소드, 36,960 프레임, `rgb_overhead [3, 104, 104]`, 4.5 GB |
| `policy lower` | < 1 s | `lowering_hash 3d06811c…`, T3와 T4의 것 — Learning IR은 움직이지 않았다 |
| `train_act.py` | **4:21** (261 s) | 20,000 스텝, 본 샘플 1,280,000, resident 복사본 4,577.6 MiB |
| `policy pack` | ~1 s | `checkpoints/20000.esb` |
| **합계** | **4:33** | |

`identity_hash 2d8f6837…`, `training_hash 61e6c93a…`, 체크포인트
`policy_hash df985459…`(spec 19.3의 `H(training_hash, checkpoint)`),
`observation_hash cc437a24…`, `lr_curve_hash c01d5185…` — 행 D와 같은 스케줄이며, 그것이 두
행을 비교 가능하게 만든다. `initial_loss 0.049270`, **`final_loss 0.006664`**, 비유한 스텝
없음. **여기서 평가하지 않았다**: 그것은 U-측정의 몫이고, 새 `evaluation_hash`가 필요하다
(13.5).

**증강이 없는 같은 실행인 행 D와 비교하면:**

| | 행 D (10절) | 이 실행 | |
|---|---|---|---|
| 트레이너 벽시계 | 2:51 (171 s) | **4:21 (261 s)** | +53 % |
| s / 1,000 스텝 | 8.6 | **13.0** | |
| samples/s | 7,485 | **4,904** | |
| `final_loss` | 0.004610 | **0.006664** | +45 % |
| resident baked set | 3.9 GiB | **4.5 GiB** | `104²/96²` = 1.17배 |

숫자 둘, 그리고 둘 다 실망이 아니라 예상된 것이다.

**그 90초.** 스텝당 4.5 ms, 배치 64가 노드 둘짜리 체인을 통과하는 비용 — 크롭을 위한 Python
슬라이스 대입 64번과 지터를 위한 64 × (곱하기 하나, `f64` 평균 하나, 어파인 하나). 이것은
산술의 비용이 아니라 샘플별 Python 루프의 비용이다; 언젠가 문제가 된다면 명백한 지렛대는
배치 전체의 오프셋을 루프가 아니라 인덱스 gather 하나로 뽑는 것이다. 아직은 문제가 아니다:
4:21은 §28.9의 예산 안쪽이고 사이클의 평가 스텝은 20분이다.

**loss가 더 높은 것이 핵심이다.** 증강은 학습 분포를 넓히므로 같은 스텝에서의 적합은 *더
나빠야* 한다 — 증강이 답하려고 존재하는 질문은 **평가** 스위트에서 무슨 일이 일어나는가이고,
거기서 증강 없는 행 D는 `final_loss 0.004610`을 기록했으며 (12.8절에서, 배치 8로)
`nominal success_rate` 0을 기록했다. 더 낮은 학습 loss와 결코 성공하지 않는 정책은 정확히
§28.10의 U-측정이 겨냥하는 그 불일치이고, 이 실행은 각 프레임의 크롭을 하나 넘게 본 쪽의
팔이다.

`~/artifacts/plan-v/m7-t6/`에 보관: `untrained-augmented.esb`(커밋된 네 문서에서
`observation.toml` 자리에 `observation-augmented.toml`), `training.toml`, `train.log`,
그리고 `run/`(4.5 GB baked 세트, `module/`, `metrics/loss.json`, `training/`의 열두 슬롯,
`training.lock`, `weights/model-20000.safetensors`, `checkpoints/20000.esb`).
