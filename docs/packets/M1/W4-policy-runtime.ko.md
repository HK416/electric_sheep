<!-- Korean translation of docs/packets/M1/W4-policy-runtime.md. The English file is the working copy; regenerate this when it changes. -->

# W4 — `PolicyRuntime`과 `PyTorch` lowering (`es-policy`)

명세: spec 2.4, spec 8.1, spec 8.3, spec 8.4, spec 8.5, spec 8.7, spec 8.8, spec 8.9, spec 1.4,
spec 5.3, spec 4.2. 설계 노트: `docs/design/learning-lowering.md` (리뷰 유형 C — 노드 표와
허용오차가 산출물이며, 코드는 그 하위 결과물이다).

spec 8.9의 M1 게이트("LeRobot ACT 체크포인트를 로드해 동일한 관측에 대해 동일한 액션을 얻는다")를
위한 기초 작업이다. 이 패킷은 하네스와 lowering을 만든다; 체크포인트 픽스처와 키 리맵은 게이트
패킷의 몫이다.

## context (범위)

```
docs/design/learning-lowering.md          (new, written first)
docs/api-notes/torch.md                   (new)
docs/packets/M1/W4-policy-runtime.md      (new)
crates/es-policy/Cargo.toml               (new)
crates/es-policy/src/lib.rs               (new)
crates/es-policy/src/runtime.rs           (new)
crates/es-policy/src/lower/mod.rs         (new)
crates/es-policy/src/lower/torch.rs       (new)
crates/es-policy/src/torch_runtime.rs     (new)
crates/es-policy/src/weights.rs           (new)
crates/es-policy/src/equiv.rs             (new)
crates/es-policy/python/torch_ref.py      (new)
crates/es-policy/tests/torch_equivalence.rs (new)
```

## spec (사양)

- `runtime::PolicyRuntime` — object-safe하며, 메서드는 네 개: `load(&LearningGraph,
  &WeightsSource) -> PolicyInfo`, `infer(&BTreeMap<String, Tensor>) -> BTreeMap<String,
  Tensor>`, `info()`, `runtime_hash() -> [u8; 32]`. `INV-17`의 일곱 확장점 중 하나다. 입력은
  `PolicyContract::inputs`에 따라 이름 붙은, **이미 전처리된** 텐서다; 출력은 그래프가 선언한
  출력이다. 런타임은 pre-/post-processing을 전혀 추가하지 않는다: 정규화, chunking,
  unnormalization은 IR 노드이며 따라서 lowering된 그래프 안에 있다 (spec 2.4, spec 8.7).
- `WeightsSource::{Safetensors(PathBuf), InMemory(Vec<u8>)}` — pickle 배리언트는 결코 없다
  (`INV-16`). `InferenceBackend`는 trait가 아니라 **enum**이다(`Torch | Onnx | Vulkan`): 오늘
  시점에는 구현체가 하나뿐이기 때문이다. `INV-17`의 trait 슬롯은 예약된 채로 남는다.
- `lower::torch::lower_to_torch(&LearningGraph) -> Result<TorchModule, LowerError>`, 여기서
  `TorchModule { source, weight_keys, weight_shapes, lowering_hash }`다. `source`는
  `forward(**inputs) -> dict`를 갖는 `class EsPolicy(nn.Module)`을 정의하는 완결된 Python
  파일이다. 코드젠은 `Graph::topo_order()` 순서로, 노드당 멤버 하나와 forward 한 줄; 지원하지
  않는 배리언트는 조용한 no-op이 아니라 항상 `LowerError::Unsupported(kind)`다. 출력은
  결정적이다 — `lowering_hash = blake3(tag || source)`이며 동일한 그래프는 반드시 동일한
  바이트열을 내야 한다.
- `torch_runtime::TorchRuntime` — Python 서브프로세스 뒤에 있는 `PyTorch`이며,
  `crates/es-physics-backend/src/proc.rs`의 JSON-lines 패턴을 그대로 따른다.
  `python/torch_ref.py`는 `include_str!`로 임베드된다. `is_available()`은 `import torch`를
  프로브한다. `runtime_hash = blake3(tag || "torch" || PROTOCOL_VERSION || torch version)`.
- `weights` — safetensors 리더/라이터(헤더 길이 u64 LE + JSON + 원시 바이트), `TorchModule`에
  대한 키·shape 검증(`PolicyError::WeightMismatch { missing, unexpected, shape }`),
  `weights_hash` = 파일 바이트의 blake3이며, 백엔드가 시작되기 **전에** `WeightsRef::hash`와
  교차 검증한다 (spec 5.3).
- `equiv::compare_actions(&Tensor, &Tensor, Tolerance) -> Equivalence { max_abs, max_rel,
  pass }` — spec 8.9의 tier-4 하네스이며, 그 표에서 온 `TIER4_FP32` (1e-5), `TIER4_VULKAN`
  (1e-4), `BITWISE`를 갖는다.

## oracle (오라클)

```
cargo test -p es-policy
ES_PYTHON=<venv with torch>/Scripts/python cargo test -p es-policy   # runs the equivalence test
```

## acceptance (수용 기준)

- `torch_mlp_matches_rust`: `StateEncoder` MLP 4 -> 8 -> 8을 `RegressionHead`(2차원, horizon
  3)에 연결하고, 고정 시드의 합성 가중치를 safetensors로 기록해 `TorchRuntime`으로 실행한 뒤,
  고정된 연산 순서를 갖는 테스트 내 Rust f32 평가와 `Tolerance::TIER4_FP32`에서 비교한다.
  **`torch`가 없으면 사유를 출력하며 SKIP한다** — spec 1.4는 오라클 설치 여부와 무관하게
  하네스가 존재하기를 원하며, wheel 누락이 통과하는 동등성으로 보고되어서는 절대 안 된다.
- 동일하게 로드된 런타임에 같은 입력을 두 번 넣으면 비트 단위로 동일하다 (spec 8.9, 표의 마지막
  행).
- lowering의 결정성: 동일 그래프의 두 lowering 결과는 해시를 포함해 바이트 단위로 동일하다.
- `es-ir`의 ACT 픽스처는 노드당 정확히 forward 한 줄로 lowering되며, 문서화된 모듈 이름과
  문서화된 가중치 키·shape을 갖는다.
- 가중치 검증은 누락된 exact 키, 빈 prefix claim, 잘못된 shape, 낯선 키를 각각 이름과 함께
  잡아낸다.
- 와이어 프로토콜은 미리 준비된 JSON에 대해 인코딩·디코딩되므로, Python 없이도 커버리지가
  확보된다.
- `WeightsSource`는 정확히 두 개의 배리언트를 가지며, 임베드된 스크립트에는 `import pickle`도
  `torch.load(`도 없다 (`INV-16`).
- 게이트: `cargo fmt -p es-policy --check`, `cargo clippy -p es-policy --all-targets -- -D
  warnings`, `cargo test -p es-policy`, `cargo xtask layering`, `cargo xtask check-spec-refs`,
  `cargo xtask context-budget`.

## forbidden (금지)

- `context` 밖의 모든 파일. 특히 `crates/es-safety` — `es-policy`는 그 의존성 그래프에 어느
  방향으로도 절대 나타나서는 안 된다(spec 4.2 규칙 8, `INV-11`) — 그리고 `crates/es-ir`와
  `crates/es-compile`은 API만 소비될 뿐 절대 수정되지 않는다. `crates/es`(CLI)와
  `docs/ARCHITECTURE*.md`는 이번 웨이브의 다른 에이전트 소관이다.
- 어떤 이름으로든, Rust에서든 Python에서든 pickle 경로 (`INV-16`).
- `PolicyRuntime` 내부의 pre-/post-processing (spec 2.4, spec 8.7). lowering에 텐서 연산이
  필요하다면, 그것은 IR 노드이거나 아예 존재하지 않아야 한다.
- 새 trait. `PolicyRuntime`이 이 패킷에 허용된 유일한 것이다 (`INV-17`); `InferenceBackend`는
  두 번째 구현체가 생기기 전까지 enum으로 남는다.
- `tch`, `ort`, `safetensors`, `base64`, `ndarray` — `libtorch`/ONNX 백엔드는 M2/M3이고,
  나머지 세 개는 각각 수십 줄이면 되는, 저장소 안에 있어야 할 코드다.
- `HashMap`/`HashSet` (spec 3.4), 스레드, 전역 RNG.
- 사전학습된 backbone을 재구현하는 것. `torchvision`은 참조될 뿐 복사되지 않는다.
- Diffusion/FlowMatching 샘플러, `PolicyBundle`, `LanguageEncoder`, 배치화, ONNX export —
  각각이 왜 보류되는지는 설계 노트 7절에 나와 있다.
