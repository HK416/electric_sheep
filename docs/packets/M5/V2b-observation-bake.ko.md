# M5 V2b — 학습 세트가 Observation IR를 통과한다

설계 노트: `docs/design/visible-learning.ko.md` 섹션 7.6 발견 3, 섹션 7.8의 "가장 그럴듯한 원인",
그리고 열린 질문 11. 먼저 읽을 것. V1(데이터셋과 타일), V2(로워링·학습 스크립트·번들), V3(이
패킷이 다시 돌리는 측정)에 의존한다. 이것은 결함 수정이고, 그 수정은 발견 3이 이미 이름 붙인
"진짜 수정" — **데이터셋 위에서 관측 플랜을 굽는 `es` 단계** — 이다.

## 결함

`python/es/train_act.py`는 그래프의 상태 포트에 **원본** `observation.state` 행을 먹이고, 이미지에
대해서는 Observation IR 노드 하나(`Op::Dequantize`)를 다시 구현한다. 추론에서는 `es eval run`의
`capture`가 Observation IR의 컴파일된 플랜을 돌리고, 데모의 플랜은
`StateInput -> Normalize{Range −1..1}` 와 `ImageInput -> Dequantize -> Normalize{Range 0..1}` 이다.
이미지 가지는 우연히 안전하고(`normalize_range(x, 0, 1)`은 항등), 상태 가지는 아니다 — 평가에서
정책은 `(q + 1) / 2`를 받고 학습에서는 `q`를 받았다. 그래서 섹션 7.8의 표는 ACT의 측정이 아니다:
nominal 2/16, `envelope_violation_rate`가 정확히 `1.0`, 그리고 **85,407 스텝 중 단 한 스텝도**
`ActionSource::Policy`가 아니었다.

근본 원인은 한 노드의 구현이 둘이라는 것이지 증상이 아니다. 증상은 빠진 `Normalize` 하나이고,
수정은 학습이 추론 실행기가 쓴 것을 읽으므로 구현이 언제나 하나뿐이라는 것이다.

## context

```
crates/es-eval/src/bake.rs
crates/es-eval/src/runner.rs
crates/es-eval/src/lib.rs
crates/es-eval/tests/evaluation.rs
crates/es/src/cmd/dataset.rs
crates/es/tests/cli.rs
crates/es-policy/src/lower/torch.rs
crates/es-policy/tests/ir_training.rs
python/es/train_act.py
tests/fixtures/visible-learning/learning.toml
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V2b-observation-bake.md
docs/packets/M5/V2b-observation-bake.ko.md
```

메모: 베이크의 핵심은 `input_sources`, `Capture` enum, `f64 -> f32` 입력 인코딩의 구현이 정확히
하나이도록 `es-eval`의 **`capture` 바로 옆**에 산다. `es-data`도 `es-eval`도 레이어 10이므로
데이터셋 읽기와 safetensors 쓰기는 이미 양쪽에 의존하는 `es`(레이어 12)에 남는다. 새 크레이트도
새 의존성도 없다.

## spec

- §7.2: Observation IR가 전처리를 **소유한다**. 노드의 두 번째 구현은 — 파이썬이든 어디든 — 이
  패킷이 없애는 바로 그것이다. INV-14는 건드리지 않는다: 데모에는 `Resize`도 `Crop`도 없으므로
  내재 파라미터 변환의 빚이 없고, 베이크는 자기 몫의 변환을 전혀 하지 않는다.
- §7.5: 관측 스트림은 에피소드에서 끝난다. 베이크는 에피소드마다 플랜을 리셋하며, 이는
  `run_episode`가 하는 `plan.reset()` 그대로다. 학습에서도 `TemporalWindow`가 에피소드 경계를
  넘어 읽을 수 없다.
- §2.3: 분할선은 그대로다. 옵티마이저는 파이썬의 것, 플랜 러너는 러스트의 것. 이 패킷은 선을
  넘어 새어 나갔던 노드 하나를 러스트 쪽으로 되돌린다.
- §1.4: "같은 IR를 PyTorch에서 돌린 것이 ground truth"가 문자 그대로 참으로 남는다 — 모듈은 여전히
  로워링의 것이고, 이제 *입력*도 실행기의 것이다.
- §5.3, §19.2, §19.3: 베이크는 매니페스트에 `observation_hash`, `task_hash`, 컴파일러 해시,
  데이터셋의 `content` 해시를 기록해 구운 세트가 어느 문서에서 왔는지 이름 붙인다. 새 해시 슬롯을
  채우지 않고 기존 슬롯을 바꾸지 않는다.
- §8.9: 왕복 등가 검사는 tier-4 fp32 허용오차를 유지한다. 여기서 무엇도 느슨해지지 않는다.
- INV-16: safetensors로 나가고 safetensors로 들어온다. `train_act.py`는 헤더 파서(`struct` +
  `json`, 15줄)를 얻고 패키지는 얻지 않는다. `pyarrow`는 잃는다.
- INV-17: 새 trait 없음. 베이크는 메서드 셋을 가진 struct다.
- §12.4: 이 패킷이 다시 재는 헤드라인은 성공률이다.
- §1.5: `es-eval`은 2,489 코드 줄, `es`는 4,066 줄이다. 이 패킷은 둘 합쳐 ~300줄 아래로 예산한다.

**V2의 열린 질문 3("사전학습 백본")이 여기서 답해진다: 거부.** `lower_to_torch`는
`VisionEncoder{pretrained}`를 무시했고, 이는 IR가 선언한 것과 실제로 도는 것 사이의 조용한 괴리다.
이를 존중하는 것은 `_backbone`의 `weights="DEFAULT"` 한 줄이고 — 그것은 틀린 한 줄이다: `EsPolicy()`
생성 **때마다** 네트워크에서 ImageNet 가중치를 가져오게 되며, 여기에는 추론 시
`TorchRuntime::load` 내부도 포함되는데 거기서 `load_state_dict(strict=True)`가 그 전부를 즉시
덮어쓴다. 인스턴스화에 네트워크가 필요한 로워링은 §2.5에 어긋난다. 그래서 `pretrained = true`는
플래그와 두 갈래 출구를 이름 붙이는 `LowerError::Unsupported`가 되고, 데모의 `learning.toml`은
`pretrained = false`라고 말한다 — V2가 실제로 하고 있던 그대로다(설계 노트 7.6, "열린 질문 6이
답해졌다"). `learning_hash`와 `policy_hash`는 움직이고, `task_hash`와 `observation_hash`는 움직이지
않으므로 구운 세트와 `evaluation.toml`은 영향받지 않는다.

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es -p es-policy --all-targets -- -D warnings
cargo clippy -p es --features render --all-targets -- -D warnings
cargo test -p es-eval
cargo test -p es-policy
cargo test -p es --test cli dataset_bake
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
cargo xtask ci
```

**이 패킷의 오라클은 비트 동일성**이고, 파이썬도 물리 백엔드도 GPU도 없이 PR 티어에서 돈다:

- `crates/es-eval/tests/evaluation.rs`: **`a_baked_frame_is_bit_identical_to_what_capture_serves`**.
  데모 모양의 Observation IR(`StateInput -> Normalize{−1..1}` **그리고**
  `ImageInput -> Dequantize -> Normalize{0..1}`) 위에서 평가 하나를 `Evaluation::run_with_frames`로
  돌린다. 프레임 소스는 자기가 호출된 `qpos` 행을 자기가 내준 타일과 함께 기록하고,
  `PolicyRuntime`은 자기가 건네받은 모든 관측 맵을 기록한다. 그 같은 행과 같은 타일이 이어서
  `ObservationBake`를 통과하며, **모든** 프레임의 **모든** 출력 텐서가 바이트 단위로 같아야 한다 —
  `dtype`, `shape`, `data`. "가깝다"가 아니고 "프레임 0에 대해서"도 아니다: 런 전체에 대해 동일.
  베이크와 `capture`가 같은 코드이길 그만두면 실패하는 테스트이고, 이 패킷 이전이었다면 실패했을
  테스트다.
- `crates/es-eval/tests/evaluation.rs`: `a_bake_refuses_an_input_the_dataset_cannot_feed` — 관절을
  이름으로 지목하는 `StateInput`은 추론에서 `ModelInfo`를 통해 풀리고 기록된 데이터셋에는 읽을
  방법이 없다. `ObservationBake::new`가 런 도중에 `observation.state`의 오프셋을 추측하는 대신
  생성 시점에 이름을 대며 거부한다.
- `crates/es-policy/tests/ir_training.rs`: `train_act_defines_no_layer`의 금지 목록에 `permute(`,
  `/ 255`, `pyarrow`가 추가된다. `Op::Dequantize`의 손복사본은 이 테스트를 깨지 않고는 돌아올 수
  없고, 그렇게 말하는 데 인터프리터가 필요 없다.
- `crates/es/tests/cli.rs`: `dataset_bake_writes_safetensors_and_a_manifest` — `es-data`의 자체
  라이터로 쓴 세 프레임 데이터셋을 굽고 되읽는다: 에피소드당 safetensors 하나가 Observation IR
  자신의 출력 이름 + `action` 아래 `[frames, ...]` 모양 텐서를 담고, `manifest.json`의
  `observation_hash`가 번들의 것이다.
  `dataset_bake_without_frames_refuses_an_image_observation` — 포트를 이름 대며 exit 1, 이미지
  채널에 0을 구워 넣는 런은 결코 없다.
- `crates/es-policy/src/lower/torch.rs`: `a_pretrained_vision_encoder_is_refused_not_ignored`.

참조 — 오라클 서버(`ES_PYTHON`, `~/venvs/es-lerobot-cuda/bin/python`):

```
cargo test -p es-policy --test ir_training -- --ignored --nocapture
```

`act_training_uses_baked_observations`(V2의 `the_loss_falls_and_the_packed_bundle_round_trips`를
다시 배선한 것)가 전 과정을 **구운** 입력 위에서 돌린다: `es policy lower` -> `es dataset bake` ->
`train_act.py --baked` -> `es policy pack` -> `TorchRuntime::load` -> `infer`, 그리고 청크를 직접
PyTorch forward와 §8.9의 tier-4 fp32 허용오차에서 비교한다. 손실은 여전히 초기값의
`LOSS_MUST_FALL_TO`까지 떨어져야 한다 — 임계값은 움직이지 않고, 낮추는 것은 골든을 편집하는 것이다.
`torch` 없음 -> 이유를 붙인 `SKIP`. PyTorch *내부*의 오류는 실패다. 돌았으면 ->
`RAN act_training_uses_baked_observations`.

**측정**(플랜 V의 요점, 오라클 서버, CI 티어 아님): 50 에피소드 데이터셋을 굽고, `--batch 8
--lr 1e-4 --seed 0`으로 20,000 스텝 재학습하며 1k/5k/20k에서 체크포인트 — 모든 손잡이가 V2와
동일하다 — 번들 셋을 패킹하고, V3가 돌린 그대로 V3의 측정을 다시 돌린다
(`ES_TRAINED_BUNDLE`을 쓰는 `visible_learning_demo_run`, 그리고 설계 노트 섹션 7.8의 직접
`es eval run` 호출들): 체크포인트당 nominal 16 에피소드, 20k에 대한 6개 스위트 전체 표,
`es video mosaic` + `python/es/encode_video.py`. `evaluation.toml`의 `success_rate >= 0.5`
acceptance는 **낮추지 않는다**. 측정된 수를, 그것이 무엇이든, 그 기준에 대고 보고한다.

## acceptance

```rust
// crates/es-eval/src/bake.rs
/// 기록된 프레임 위에서 Observation IR를 돌린다. 추론에서 `capture`가 쓰는 것과 같은 `CpuPlan`,
/// 같은 `input_sources` 해석, 같은 입력 인코딩을 통해서(§7.2).
pub struct ObservationBake { /* plan + resolved sources */ }

impl ObservationBake {
    /// `ModelInfo` 없음: 기록된 데이터셋은 `observation.state`와 타일만 담으므로, 로드된 모델이
    /// 필요한 입력은 어떤 프레임보다도 먼저 여기서 이름 대고 거부된다.
    pub fn new(obs: &ObservationIr, task: &TaskIr) -> Result<Self, EvalError>;
    /// 에피소드 경계(§7.5), 정확히 `run_episode`의 `plan.reset()`.
    pub fn reset(&mut self);
    pub fn frame(
        &mut self,
        state: &[f64],
        image: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
    ) -> Result<BTreeMap<String, Tensor>, EvalError>;
    pub fn outputs(&self) -> impl Iterator<Item = (&String, ElemType, &[u64])>;
}
```

```
es dataset bake --policy <bundle.esb> --out <dir> [--frames <tiles>] <dataset-root>
# <dir>/episode-000000.safetensors   { "<obs output>": [frames, ...], "action": [frames, dim] }
# <dir>/manifest.json                observation_hash, task_hash, compiler_hash,
#                                    dataset_content_hash, episodes[], tensors{}, frames
```

- `--observation <obs.toml>`이 아니라 `--policy <bundle.esb>`: 번들은 Task IR **와** 추론이 쓸
  Observation IR를 함께 담는 유일한 문서이고, 같은 흐름에서 한 단계 앞인 `es policy lower`의
  입력이며, 그러면 매니페스트의 `observation_hash`가 재유도가 아니라 번들 자신의 것이 된다.
- 베이크는 자기 몫의 변환을 하지 않는다. `observation.state` 또는 원본 타일과 구운 텐서 사이의
  모든 바이트는 `CpuPlan::run`이 만든다.
- `JointState { dof }` 채널보다 좁은 데이터셋 행, 크기가 틀린 타일, `F32`가 아닌 플랜 출력,
  타일 수와 어긋나는 프레임 수 — 각각이 이름 붙은 거부다. 베이크를 완성시키려고 무엇도 패딩되거나
  리샘플되거나 0으로 채워지거나 재모양되지 않는다.
- `train_act.py`는 `--baked <dir>`를 읽고 **그것만** 읽는다: `read_dataset`, `read_frames`,
  `dequantize`, `plan_inputs`는 플래그 뒤로 남지 않고 삭제된다. 구운 텐서가 없는 contract 입력은
  거부이며, 이는 V2가 경고해야 했던 "조용히 0" 실패 모드를 은퇴시킨다.
- 새 trait 없음(INV-17), 러스트에도 파이썬에도 새 외부 의존성 없음, 소스 ~300줄 이하.

## forbidden

- `crates/es-data/src/lerobot/**` — v2.1 라이터는 건드리지도, 업그레이드하지도, 교체하지도 않는다.
  베이크는 **읽는다**. V1b의 `es dataset export --lerobot-v3`이 변환기이고 그대로 남는다.
- `crates/es-env`, `crates/es-safety` — 엔벌로프 변경 없음, 리셋 오버라이드 없음, 렌더러 변경 없음.
  측정은 V3의 문서를 수정 없이 다시 돌린다(INV-11, INV-12, INV-13).
- `crates/es-ir` — 스키마 변경 없음. `VisionEncoder{pretrained}`는 노드에서 제거되는 것이 아니라
  *로워링이 거부*하고, `es_ir::learning::testing::act_like`는 계속 `true`를 선언한다. 그것을
  로워링하는 `es-policy`의 두 테스트가 로컬에서 플래그를 지운다.
- `tests/fixtures/visible-learning/{task,observation,deployment,evaluation}.toml` —
  `observation_hash`를 움직이면 구운 세트와 V3의 `evaluation_hash`가 한 번에 무효가 되고, 열린 질문
  11이 거절하는 싼 대안(상태 `Normalize` 제거)이 정확히 그것이다. `learning.toml`만, 그것도
  `pretrained` 플래그만 움직인다.
- `evaluation.toml`의 `success_rate >= 0.5` acceptance나 `ir_training.rs`의 어떤 임계값도 낮추지
  않는다. acceptance에 미달한 측정은 미달했다고 보고한다.
- V2가 쓰지 않은 무엇으로도 재학습하지 않는다: 같은 시드, 같은 배치, 같은 학습률, 같은 스텝 수.
  이 실험에서 움직이는 변수는 하나이고, 그것은 관측이다.
