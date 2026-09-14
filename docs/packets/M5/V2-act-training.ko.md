# M5 V2 — Learning IR 로워링을 PyTorch에서 학습하고 다시 패킹하기

설계 노트: `docs/design/visible-learning.md` 섹션 6; 섹션 2.5를 먼저 읽을 것 — 저장소에 학습 루프가
전혀 없고, `lower_act`로 만든 번들은 **`es eval run`으로 실행할 수 없다**. 그래서 이 패킷은 대신 IR이
소유한 그래프를 학습한다. V1 (데이터셋)에 의존.

## context

```
crates/es/src/cmd/policy.rs
crates/es/src/cmd/mod.rs
crates/es/src/main.rs
crates/es/tests/cli.rs
crates/es-policy/src/lower/mod.rs
python/es/train_act.py
python/es/README.md
python/es/README.ko.md
crates/es-policy/tests/ir_training.rs
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V2-act-training.md
docs/packets/M5/V2-act-training.ko.md
```

메모: `crates/es/src/cmd/policy.rs`는 새 파일 — `es policy lower`와 `es policy pack`; `mod.rs`와
`main.rs`는 모듈과 디스패치 분기 하나를 얻는다 (`main.rs:48-61`이 그 표다). `lower/mod.rs`는 컨트랙트
타입만 재수출한다. `python/es/train_act.py`는 **유일한** 새 Python이며 `cargo xtask context-budget`에
집계되지 않는다 (`xtask/src/context_budget.rs:148-152`가 `src/` 아래 `*.rs`만 읽는다).

## spec

- §1.4: 학습은 "레퍼런스와 일치"로 판정할 수 없으므로 실행 가능한 네 사실로 판정한다 (설계 노트 섹션
  6.3): 모듈이 IR의 것이다, 가중치가 컨트랙트에 맞는다, 손실이 떨어진다, 번들이 왕복한다. 그 중 어느
  것도 사람이 곡선을 읽는 것이 아니다.
- §2.3, §2.4: 옵티마이저는 분할선의 Python 쪽에 있다; `es policy lower`와 `es policy pack`은 Rust이며
  Python 없이 돈다. `es-policy`는 이미 가진 것 외의 Python 의존성을 얻지 않는다.
- §8.1: 옵티마이저는 IR에 없다 — `train_act.py`는 하이퍼파라미터를 자기 CLI에서 읽고, 그 중 어느 것도
  `TrainingIdentity` (§19.2)를 통하지 않고는 어떤 해시 슬롯에도 들어가지 않으며, V2는 그것을 채우지
  않는다.
- §8.7, INV-16: 가중치는 `safetensors`다. `WeightsSource` (`crates/es-policy/src/runtime.rs:27-32`)는
  변형을 얻지 않고, pickle로 로드 가능한 파일은 쓰이지도 읽히지도 않는다. IR이 전/후처리를 소유한다:
  `pack`은 정규화기를 런타임으로 옮기지 않는다.
- §5.3: `pack`은 번들 매니페스트를 편집하지 않고 재계산한다 (`crates/es-compile/src/bundle.rs:467`,
  `:513`); 가중치가 바뀌었으므로 `policy_hash`가 바뀐다.
- §25.1: `model.safetensors`는 Rust 코어 밖에서 온다. `pack`은 쓰기 **전에** 모든 키와 모양을 컨트랙트와
  대조하고, 모르는 키는 무시하지 않고 거부한다.
- §12.4: 이 패킷의 출력 어디에도 학습 시간, 처리량, `step/s`가 없다.
- §1.5: `es`는 2,946 코드 줄; 이 패킷은 `src/` 기준 약 400줄 이하로 잡는다.

## oracle

```
cargo fmt --check
cargo clippy -p es -p es-policy --all-targets -- -D warnings
cargo test -p es --test cli policy_
cargo test -p es-policy --test ir_training
cargo xtask context-budget
cargo xtask check-spec-refs
```

레퍼런스 — `torch`가 필요한 학습 구간:

```
ES_PYTHON=$HOME/venvs/es-lerobot/bin/python \
  cargo test -p es-policy --test ir_training -- --ignored --nocapture
```

`tests/ir_training.rs`가 임시 디렉터리의 고정된 작은 데이터셋에서 세 단계 전체를 돌리고 네 사실을
확인한다. `torch`가 없으면 사실별로 `SKIP ir_training: <why>`; 넷 다 실행되면 `RAN ir_training`. 진짜
PyTorch 에러를 "torch가 설치되지 않음"과 같은 SKIP으로 묶어서는 **안 된다** — 그것이
`act_checkpoint.rs:113`에 대한 `docs/reviews/M4.md:67` (S-7)이고, 이 패킷은 그것을 반복하면 안 된다.

**2026-09-14 오라클 서버에서 측정.** 두 venv 모두 CPU 전용 빌드다: `~/venvs/es`는 `torch 2.14.0+cpu`,
`~/venvs/es-lerobot`은 `torch 2.11.0+cpu`이고, RTX 4090 위에서 `torch.cuda.is_available()`이 둘 다
`False`다. V0의 이미지가 96x96이고 `H = 16`인 이유가 이것이다. 이 패킷은 학습 시간도 하드웨어 주장도
말하지 않는다 (§12.4); GPU 문제는 설계 노트 미해결 질문 8이다.

`tests/ir_training.rs`:

- `the_trained_module_is_the_lowering` — `train_act.py`가 로드한 `es_policy.py`가
  `lower_to_torch(&bundle.learning)?.source` (`crates/es-policy/src/lower/torch.rs:334`)와 바이트
  동일하다. 이 패킷이 존재하는 이유 전부가 이것이다: §1.4의 "동일한 IR을 PyTorch에서 돌린 것이 ground
  truth"를 "비슷한 모델"이 아니라 문자 그대로 참으로 만든다.
- `train_act_defines_no_layer` — `python/es/train_act.py`의 소스 스캔에 `nn.Linear`, `nn.Conv`,
  `nn.Transformer`, `nn.Module` 상속, `torchvision.models` 호출이 없다. 아키텍처는 로워링에서 오거나
  오지 않는다.
- `pack_refuses_a_missing_key`, `pack_refuses_an_extra_key`, `pack_refuses_a_wrong_shape` — 거부 셋,
  각각 문제의 키 이름을 밝히고 exit 1.
- `the_loss_falls` — 고정 데이터셋과 `--seed 0`; 최종 손실이 초기 손실의 고정 비율 미만이고 둘 다
  `train_act.py`가 JSON으로 출력한다. 골든이 아니라 임계값이다: CPU PyTorch는 버전 간 비트 단위 이식이
  되지 않는다 (설계 노트 섹션 9).
- `the_packed_bundle_round_trips` — 출력에 대한 `PolicyBundle::open` 성공, `TorchRuntime::load`가
  받아들임 (즉 `load_lowered`가 아니라 `lower_to_torch`를 탄다), 고정 관측에 대한 `infer`가 `[H, NJ]`
  모양의 유한한 청크를 반환.
- `eval_run_accepts_the_trained_bundle` — `es eval run --policy trained.esb`가
  `TorchRuntime::load`를 통과한다 (`crates/es/src/cmd/eval.rs:387-393`). `lower_act`라면 실패했을 단언
  이고, 플랜 V가 그것을 쓰지 않는 이유다.

`crates/es/tests/cli.rs`: `policy_lower_writes_the_module_and_contract`,
`policy_pack_rejects_a_non_safetensors_file` (pickle 경로에 닿을 수 없음, INV-16),
`policy_usage_errors_exit_2`.

## acceptance

```rust
// crates/es/src/cmd/policy.rs
// es policy lower --policy <in.esb> --out <dir>
//   기록: <dir>/es_policy.py           = TorchModule::source, 그대로
//        <dir>/contract.json          = { lowering_hash, weight_keys: [String],
//                                         weight_shapes: { key: [usize] },
//                                         action_dim, horizon, inputs: { port: [usize] } }
// es policy pack --policy <in.esb> --weights <model.safetensors> --out <out.esb>
//   모든 키와 모양을 컨트랙트와 대조한 뒤 새 weights.safetensors와 재계산된 매니페스트로
//   번들을 다시 쓴다.
pub fn dispatch(args: &[String]) -> i32;   // 0 ok, 1 runtime, 2 usage, 3 SKIPPED
```

```
python/es/train_act.py --module <dir> --dataset <root> --out model.safetensors \
    [--epochs N] [--batch N] [--lr F] [--seed N] [--device cpu]
```

- `train_act.py`는 `<dir>/es_policy.py`를 `exec`하고 `EsPolicy()`를 인스턴스화하고 V1이 쓴 LeRobot v2.1
  데이터셋을 읽고 최적화한 뒤, 키 집합이 `contract.json`과 정확히 같은 `safetensors`를 쓴다. stdout에는
  JSON 한 줄 `{"initial_loss": f, "final_loss": f, "steps": n}` 외에 아무것도 출력하지 않는다.
- 레이어를 정의하지 않고 모델 주(zoo)를 import하지 않는다. 최적화하는 모든 파라미터는 로워링에서 왔다.
- `pack`은 누락 키, 추가 키, 모양 불일치를 이름과 함께 exit 1로 거부한다; 텐서를 버리거나 패딩하거나
  reshape하지 않는다.
- 출력 번들은 입력과 같은 `task`, `observation`, `learning`, `deployment` 엔트리와 새
  `weights.safetensors`를 갖는다; 매니페스트는 패치가 아니라 `PolicyBundle::build`가 재계산한다.
- `es eval run --policy <out.esb>`가 **`crates/es/src/cmd/eval.rs`의 정책 경로를 전혀 바꾸지 않고**
  이를 로드한다.
- 새 트레이트 없음, 새 외부 크레이트 없음, `src/` 기준 약 400줄 이하.

## forbidden

- `crates/es-policy/src/lerobot.rs` — `lower_act`와 M4의 우회는 여기서 명시적으로 **고치지 않는다**
  (설계 노트 미해결 질문 1). 호출하지도, 변경하지도, `TemporalEncoder { Transformer }`를 넓히지도 말 것.
- `crates/es-ir`, `crates/es-ir-types` — 우회를 고치려면 §8.3의 스펙 변경이 필요하고, `es-ir`은 §1.5
  예산이 53줄 남았다.
- `crates/es-policy/src/torch_runtime.rs` — 특히 유일한 호출자가 `lower_act`인 `load_lowered`
  (`torch_runtime.rs:278-284`).
- `crates/es-policy/src/runtime.rs` — `WeightsSource` 변형 금지, pickle 금지, `torch.load` 금지 (INV-16).
- `es learn` / `es train` 서브커맨드 추가, 또는 Rust 쪽 옵티마이저: `crates/es/src/cmd/loop.rs:5-7`이
  이유를 말한다.
- `TrainingIdentity`의 `dataset` 외 슬롯을 0이 아닌 값으로 채우는 것
  (`crates/es-data/src/collect.rs:650-665`) — 지어낸 다이제스트는 `training_hash`를 거짓말로 만든다.
- `crates/es-env`, `crates/es-eval`, `crates/es-render`, `crates/es-data` — V0b, V1, V3.
- 벽시계 학습 수치를 사실로 보고하는 것.
