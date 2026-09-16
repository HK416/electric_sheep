# M7 T1 — `es train`: 문서 하나, 명령 하나, 진짜 `training_hash`

스펙: §13.1(루프의 각 단계는 CLI 명령*이자* 아티팩트다), §19.3(학습 아이덴티티 — `training/`
번들과 `training_hash`), §8.8, §2.3(Python은 학습 경로에만), §1.4, §28.10 규칙 2. 새로 쓸
설계 노트: `docs/design/training-recipe.md`(+ `.ko.md`).
선행: M5 V2/V2b(`train_act.py`, `es dataset bake`), V8/V19(`import-lerobot`,
`lerobot-train`), M3 W7(`es loop distill`이 모든 슬롯이 0인 `training_identity.json`을 쓴다).

## 질문

오늘 하나의 학습 실행은 아홉 개의 손으로 짠 스크립트다(`docs/packets/M5/V19`의 "server" 절):
`export → lerobot-train → import-lerobot → eval run`이거나
`bake → lower → train_act.py → pack`이고, 그 사이에서 해시를 손으로 옮겨 적으며 §19.3의
`training_hash`는 `dataset`을 뺀 나머지가 전부 0이다. **문서 하나가 명령 하나로 두 경로 중
어느 쪽이든 몰 수 있고, 그 명령이 실제 값으로 §19.3의 `training/`을 쓸 수 있는가?**

## spec

`es train --recipe training.toml [--out <dir>] [--dry-run]`.

레시피(`kind = "training"`, CLI가 아니라 `es-data`에서 파싱되고 해시된다):

```toml
kind = "training"
[dataset]
root   = "runs/collect-001/ds"        # LeRobot v2.1 루트 (es loop collect)
frames = "runs/collect-001/frames"    # 선택; Observation IR에 이미지 입력이 있으면 필수
[policy]                              # 둘 중 정확히 하나
bundle  = "untrained.esb"             # IR 경로: 번들에서 가져온 Learning IR + Observation IR
# lerobot = { type = "act", chunk_size = 16, n_action_steps = 16, extra = [] }   # 외부 경로
# task = "task.toml"; observation = "observation.toml"; deployment = "deployment.toml"
[run]
steps = 20000
batch = 8
lr = 1e-4
seed = 0
checkpoint_at = [1000, 5000, 20000]
device = "cuda"
interpreter = "python"                # ES_PYTHON이 설정되면 그것으로 덮어써진다
```

* **IR 경로**: `es dataset bake --policy … --out <out>/baked [--frames …] <root>` →
  `es policy lower --policy … --out <out>/module` → `<interpreter> python/es/train_act.py
  --module <out>/module --baked <out>/baked --out <out>/weights/model.safetensors --checkpoint-at …
  --seed … --batch … --lr … --device … --loss-curve <out>/metrics/loss.json` → 체크포인트마다
  `es policy pack --policy … --weights … --out <out>/checkpoints/<step>.esb`. 기존 서브커맨드는
  (재기동되지 않고) **프로세스 내부에서** 호출된다(`crates/es/src/cmd`의 함수들이다) — 서브
  프로세스인 것은 트레이너뿐이다.
* **외부 경로**: `es dataset export --lerobot-v3 <root> --out <out>/ds-v3 [--frames …] --drop
  action_commanded,action_source,intervention --state-dim <Observation IR에서>` →
  `<interpreter> -m lerobot.scripts.lerobot_train …`(또는 인터프리터 옆의 `lerobot-train` 엔트리
  포인트)를 `--policy.type`, `--policy.chunk_size`, `--steps`, `--batch_size`,
  `--seed`, `--save_freq`, `--output_dir <out>/lerobot`, `--policy.device`, `--wandb.enable=false`,
  `--policy.push_to_hub=false`와 `extra`를 그대로 붙여 실행 → 저장되는 체크포인트마다
  `es policy import-lerobot --checkpoint … --task … --observation … --deployment … --out
  <out>/checkpoints/<step>.esb`.
* **`<out>/training/`**는 §19.3을 따르며, 모든 파일은 그 실행이 실제로 쓴 값으로부터 쓰인다:
  `config.json`(레시피, 캐노니컬 JSON에 해석된 인터프리터 경로와 트레이너가 보고한 버전을
  더한 것), `optimizer.json`(`AdamW`, lr, 트레이너가 보고하는 그대로의 betas/weight decay),
  `scheduler.json`(T4까지는 `{"kind":"constant"}`), `seed.json`, `dataset.lock`
  (`es-data::identity`가 주는 데이터셋의 콘텐츠/스키마/스플릿 해시), `base_model.lock`(IR
  경로에서는 `{"source":"none"}`; 외부 경로에서는 LeRobot 설정의 `vision_backbone` +
  `pretrained_backbone_weights` 문자열 — **선언된** 프로버넌스이며, 해시는 T5까지 미설정),
  `augmentation.json`(T6까지는 `{"kind":"none"}`), `precision.json`(fp32 / `--amp` 플래그),
  `topology.json`(`{"world_size":1}`), `checkpoint.manifest`(스텝 → 번들 경로, `policy_hash`,
  가중치의 blake3), `metrics.json`(loss 곡선; `es-data`의 기존 parquet 라이터가 스칼라
  테이블을 몇 줄로 받을 수 있을 때에 한해 `metrics.parquet`를 대신 쓴다 — 어느 쪽인지 설계
  노트에 적을 것), `hardware.json`(`torch.cuda.get_device_name` 또는 `cpu`에서 얻은 장치
  이름, torch 버전, 드라이버 문자열, `es` 버전과 git describe). 실행이 정말로 알지 못하는
  슬롯은 `{"unset": true}`로 쓰이고 그렇게 해시된다 — **조작된 다이제스트는 결코 안 된다**
  (§28.10 규칙 2).
* **`<out>/training.lock`**: `training_hash` = 위 각 파일의 blake3에 대한
  `TrainingIdentity::training_hash`이며, 체크포인트마다 `policy_hash`가 더해진다. 같은
  입력에 같은 인터프리터로 같은 레시피를 두 번 실행하면 트레이너가 돌기 **전에** 이미 하나의
  `training_hash`를 내야 한다(`checkpoint.manifest`/`metrics`/`hardware`를 뺀 모든 것이
  사전에 알려져 있다) — 설계 노트가 어떤 슬롯이 실행 전이고 어떤 슬롯이 실행 후인지 적으며,
  `training.lock`은 두 다이제스트를 모두 담는다(실행 전 `identity_hash`, 실행 후
  `training_hash`).
* **`--dry-run`**은 명령 계획을 출력하고 — 단계마다 한 줄, 경로는 `<out>` 기준 상대경로 —
  `<out>/training/plan.txt` 말고는 아무것도 쓰지 않는다. CI가 Python 없이 판정하는 대상이
  바로 이것이다.
* 이름을 대는 거부: `bundle`과 `lerobot`이 둘 다 설정됐거나(또는 둘 다 아니거나); 이미지
  입력에 `frames`가 없거나; 데이터셋에 기록된 `task_hash`가 정책의 Task IR과 다른데
  `--allow-retired-task`가 그것을 지목하지 않거나(M5 리뷰 S-3/R4 — 이 검사는 export/import
  경로의 거부이며 여기서 표면화된다); `import torch`를 할 수 없는 인터프리터(외부 경로에서는
  `import lerobot`) — 인터프리터 자신의 에러를 그대로 출력한다.

## context (허용 범위)

`crates/es-data/src/training.rs`(레시피 스키마, 플랜 빌더, 아이덴티티 조립; 신규),
`crates/es-data/src/lib.rs`(모듈과 재수출만), `crates/es/src/cmd/train.rs`(신규),
`crates/es/src/cmd/mod.rs`, `crates/es/src/main.rs`(디스패치 + `TOP_HELP` 줄),
`crates/es/tests/cli.rs`(새 `train_*` 테스트만; 파일의 다른 곳은 건드리지 않는다),
`tests/fixtures/visible-learning/training.toml`과 `training-lerobot.toml`(새 픽스처),
`tests/golden/train/plan-ir.txt`, `plan-lerobot.txt`(새 골든. `generate_goldens`처럼
`#[ignore]`가 붙은, 새로 추가할 정식 테스트 생성기가 만든다), `docs/design/training-recipe.md`
+ `.ko.md`(신규), `docs/packets/M7/T1-es-train*.md`, `python/es/train_act.py`는 **오직**
JSON 요약에 `torch.__version__`과 옵티마이저 하이퍼파라미터를 출력하도록만(학습 루프 변경은
없음 — 그건 T3/T4의 몫).

## oracle

1. `cargo test -p es --test cli train_dry_run_plan_is_the_golden` — 두 픽스처 레시피에
   `--dry-run`을 돌려 stdout과 `plan.txt`가 골든과 바이트 단위로 같다(경로가 `<out>` 기준
   상대경로이므로 골든은 기계에 무관하다). Python 불필요.
2. `cargo test -p es --test cli train_identity_is_a_function_of_the_recipe` — 같은 레시피를
   서로 다른 두 스크래치 디렉터리에서 돌린 실행 전 `identity_hash`가 같고; `seed`나 `lr`을
   바꾸면 움직이며; `training/` 파일들이 존재하고 각각이 유효한 캐노니컬 JSON이며; 어떤
   슬롯도 전부 0인 다이제스트가 아니다(`{"unset": true}`는 허용되고 그렇게 집계된다). Python
   불필요.
3. `cargo test -p es --test cli train_ir_path_packs_a_bundle_torch_opens -- --ignored` — torch를
   가진 인터프리터로 `ES_PYTHON`을 설정하면: 데모 번들과 bake 픽스처(`write_bake_fixture`)로
   40스텝을 돌리고, `TorchRuntime`이 `checkpoints/40.esb`를 열어 유한한 청크를 추론하며;
   `checkpoint.manifest`가 올바른 blake3로 그것을 이름 붙이고; `training_hash`가 설정되어
   `identity_hash`와 다르다. `ES_PYTHON` 없이는 `SKIP: <reason>`을 출력한다.
4. `cargo test -p es --test cli train_refuses_by_name` — 위의 거부들이 각각 필드를 이름으로
   말한다.
5. `cargo xtask ci` 통과; `cargo xtask check-scope docs/packets/M7/T1-es-train.md` 깨끗함.

## acceptance

오라클 1, 2, 4는 로컬에서 통과한다; 3은 오라클 서버(`~/venvs/es-lerobot-cuda`)에서 통과하고,
외부 경로의 `--dry-run` 계획은 명령줄이 `lerobot-train`이 받아들이는 것임을 증명하기 위해
서버에서 `~/artifacts/plan-v/v15/ds-train`에 대해 `steps = 200`으로 *손으로 한 번* 실행한다
(벽시계 시간과 `training.lock`을 설계 노트에 기록할 것 — 이것은 게이트가 아니라 관측이다).
설계 노트가 기록하는 것: 레시피 스키마, §19.3 슬롯의 실행 전/실행 후 분할, `hardware.json`이
담는 것, 그리고 무엇이 `unset`이고 왜 그런지.

## forbidden

`crates/es-ir/**`(IR 변경 없음; 레시피는 여섯 번째 IR이 아니라 `es-data` 문서다);
`crates/es-policy/**`(로워링은 T3/T5의 것); `crates/es-safety/**`; `train_act.py`의 학습
루프; `crates/es/src/cmd/{dataset,policy,loop}.rs`는 필요하면 함수 하나를 `pub(crate)`로
바꾸는 것 이상은 안 됨; `docs/ARCHITECTURE*.md`; 기존의 모든 골든과 픽스처; `es`나
`es-data`의 새 의존성. INV-16: pickle 기반은 어떤 것도 읽거나 쓰지 않는다. INV-17: 새
트레이트 없음.
