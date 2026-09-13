<!-- Korean translation of docs/api-notes/torch.md. The English file is the working copy; regenerate this when it changes. -->

# PyTorch — `TorchRuntime`를 위한 고정 API 표면

`crates/es-policy`가 실제로 호출하는 것들을 정리한 문서로, torch를 업그레이드할 때 발굴 작업이
아니라 이 파일에 대한 diff 확인만으로 끝나게 하기 위한 것이다. `docs/api-notes/mujoco.md`와
짝을 이루며, 형태와 이유가 동일하다.

명세: spec 1.4 (PyTorch는 Learning IR의 참조 오라클), spec 2.4 (`TorchRuntime`은 M1 백엔드),
spec 8.7 (lowering), spec 8.9 (tier-4 동등성). lowering 계약 자체는
`docs/design/learning-lowering.md`에 있다.

## 버전

| | |
|---|---|
| 검증 대상 | **torch 2.14.0+cpu** (`--index-url https://download.pytorch.org/whl/cpu`) |
| Python | 3.12.10 |
| `numpy` | **설치하지 않음, 필요 없음** — 아래 "numpy 없음" 참고 |
| `safetensors` 패키지 | **필요 없음** — 리더는 `struct` + `json` 15줄이면 충분하다 |
| `torchvision` | `VisionEncoder`가 있는 그래프에서만 필요; 아직 CI에서 실행되지 않음 |
| `diffusers` | **0.40.0**, `python/ddpm_ref_check.py`(DDPM/DDIM 오라클)에서만 필요; 순수 Python, `pip install diffusers` |

로컬 실행을 위한 설치:

```
python -m venv <short path>            # a deep path fails on Windows without long-path support
<venv>/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
ES_PYTHON=<venv>/Scripts/python cargo test -p es-policy
```

`ES_PYTHON`은 `MuJoCo` 오라클이 쓰는 것과 같은 변수이므로, venv 하나로 둘 다 처리할 수 있다.
이 변수가 없으면 `python`, 그다음 `python3` 순서로 시도한다; 어디에도 `torch`가 없으면 동등성
테스트는 `SKIPPED`를 출력하고 통과하며, 다른 모든 테스트는 그대로 실행된다.

## `diffusers` — 독립적인 DDPM/DDIM 오라클

`python/ddpm_ref_check.py`는 같은 stdin/stdout JSON 형태를 가진 두 번째, 더 작은
스크립트다. 이것이 존재하는 이유는 `src/reference.rs`의 Rust 레퍼런스가 lowering을
미러링하므로 잘못된 *스케줄*을 잡아낼 수 없기 때문이다(M2 리뷰,
`crates/es-policy/src/reference.rs:26`). 이것이 호출하는 것:

| 심볼 | 용도 |
|---|---|
| `diffusers.DDPMScheduler(num_train_timesteps, beta_schedule, variance_type, clip_sample, clip_sample_range, prediction_type)` | 검사 대상 스케줄 |
| `diffusers.DDIMScheduler(...)` | 동일하되 `variance_type`은 없음(그런 것이 없으며, eta = 0은 분산을 쓰지 않는다) |
| `.set_timesteps(n)` / `.timesteps` / `.alphas_cumprod` | 서브샘플과 누적 alpha |
| `._get_variance(t)` / `._get_variance(t, prev_t)` | 스텝별 분산; private이며, 버전이 올라갈 때 가장 먼저 깨질 만한 지점 |
| `.step(model_output, t, sample).prev_sample` | 두 번째 테스트를 위한 전체 샘플러 루프 |
| `diffusers.schedulers.scheduling_ddpm.randn_tensor` | DDPM에 새로운 추출 대신 체크포인트의 `noise_<t>` 버퍼를 넘기도록 monkeypatch되어, 양쪽이 같은 숫자를 소비한다(spec 3.4) |

알아둘 만한 두 가지 shape 사항: `DDPMScheduler.step`은 학습된 분산을 감지하기 위해
`model_output.shape[1]`을 읽으므로, lowering된 head가 갖지 않는 배치 축이 필요하다
(스크립트가 reshape한다); 그리고 `_get_variance`는 private API이므로, `diffusers` 업그레이드는
이 표에 대한 diff가 된다. `diffusers`가 없으면 -> `torch`가 없을 때와 마찬가지로 두 테스트
모두 `SKIPPED`를 출력하고 통과한다.

## 이 코드가 의존하는 Python API

전부 `crates/es-policy/python/torch_ref.py`와 생성된 모듈 안에 있다.

**참조 프로세스가 호출하는 것**

| API | 용도 | 비고 |
|---|---|---|
| `torch.__version__` | `PolicyInfo::version`, 그리고 이를 통해 `runtime_hash` | 문자열이다; 이 값이 바뀌면 해시 체인도 바뀌는데, 이것이 의도다 (spec 5.3) |
| `torch.tensor(list, dtype=torch.float32)` | 와이어를 넘나들거나 체크포인트에서 나오는 모든 텐서 | `frombuffer`가 아니라 float 리스트로 구성한다: buffer-protocol도 numpy 의존도 없다 |
| `Tensor.reshape(shape)` | 선언된 shape 적용 | |
| `Tensor.detach().to(torch.float32).contiguous().reshape(-1).tolist()` | 출력 인코딩 | `tolist`는 f32를 Python float으로 정확히 확장하며, `struct.pack("<f")`가 이를 동일한 비트로 다시 좁힌다 |
| `nn.Module.load_state_dict(sd, strict=True)` | 가중치 로딩 | `strict=True`는 의도적이다: 조용한 부분 로드가 바로 이 패킷 전체가 존재하는 이유인 실패 모드다 |
| `nn.Module.eval()` | 추론 모드 | `nn.TransformerEncoderLayer`의 dropout에 영향을 준다 |
| `torch.inference_mode()` | forward pass | |
| `compile(source, "<es-policy>", "exec")` + `exec` | 생성된 모듈 인스턴스화 | 이 프로세스가 실행하는 *유일한* 코드이며, 파이프를 통해 도착할 뿐 체크포인트에서 오지 않는다 |

**생성된 모듈이 사용하는 것** (`crates/es-policy/src/lower/torch.rs`)

`nn.Module`, `nn.Sequential`, `nn.Linear`, `nn.ReLU`, `nn.Identity`, `nn.TransformerEncoder`,
`nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)`, `nn.Module.register_buffer(...,
persistent=False)`, `torch.cat(tensors, dim=)`, `Tensor.unsqueeze/squeeze/reshape`, 슬라이싱.
vision용으로는: `torchvision.models.resnet18` / `resnet34`이며, 그 `.fc`는 `nn.Linear`로
교체된다.

**의도적으로 호출하지 않는 것**

`torch.load`, `torch.save`, 어떤 형태로든 `pickle` (`INV-16`); `torch.compile`, `torch.jit`,
`torch.onnx` (ONNX는 M2 소관); CUDA 관련 일체 (오라클은 정의상 CPU다).

### numpy 없음

CPU wheel은 `numpy`를 끌어오지 않으며, 이것이 없으면 torch는 import 시
`Failed to initialize NumPy` 경고를 출력한다. 이는 stderr로 가고, Rust 쪽에서 null로 파이프한다.
여기서는 numpy가 전혀 필요 없다: 텐서는 Python 리스트로 만들어져 `struct`로 직렬화된다. numpy를
빼두면 CI venv를 wheel 하나로 유지할 수 있다.

## 와이어 프로토콜

줄 단위로 구분된 JSON으로, 입력은 한 줄에 요청 하나, 출력은 한 줄에 JSON 객체 하나다 —
`crates/es-physics-backend/python/mujoco_ref.py`와 같은 형태다. 프로토콜 버전은 **1**
(`es_policy::torch_runtime::PROTOCOL_VERSION`)이며, 이 값은 `runtime_hash`에 해시되어 들어간다.

```
-> {"cmd": "load",  "source": "<generated python>", "weights_path": "<path>.safetensors"}
<- {"ok": true, "torch_version": "2.14.0+cpu", "protocol": 1}

-> {"cmd": "infer", "inputs": {"joint_state": {"shape": [4], "dtype": "F32",
                                               "data_b64": "…"}}}
<- {"ok": true, "outputs": {"actions": {"shape": [3, 2], "dtype": "F32", "data_b64": "…"}}}

-> {"cmd": "quit"}                      (no reply; the process exits)
```

모든 실패는 `{"ok": false, "error": "<ExceptionType>: <message>"}`다 — 모델링 실수는 죽은
프로세스나 stderr의 traceback이 아니라 항상 타입이 있는 `PolicyError`다.

**텐서 인코딩.** `data_b64`는 텐서의 원시 little-endian f32 바이트를, row-major로, 배치 축
없이(spec 5.4) 표준 base64(RFC 4648, 패딩 포함)로 인코딩한 것이다. `dtype`은 safetensors의
표기(`"F32"`)를 쓰므로 이름 하나가 파이프 양쪽에서 같은 것을 의미한다; 그 외 값은 거부된다. JSON
숫자도 대안이었지만 기각되었다: `[50, 8]` 크기의 청크를 숫자로 표현하면 바이트 수가 약 20배가
되고, spec 8.9가 1e-5 비교를 원하는 바로 그 지점에서 최단 왕복 논쟁을 불러들이게 된다.

**상태 유지.** `load`는 반드시 `infer`보다 먼저 와야 한다; 두 번째 `load`는 모델을 교체한다.
프로세스는 정확히 모델 하나를 보유하므로, `TorchRuntime`은 정확히 프로세스 하나를 보유한다.

## Safetensors

양쪽이 구현하는 그대로의 포맷 전체:

```
[0 ..  8)        header length N, u64 little-endian
[8 .. 8+N)       header, UTF-8 JSON object
[8+N .. )        data segment, raw tensor bytes
```

각 헤더 항목은 `"<key>": {"dtype": "F32", "shape": [..], "data_offsets": [start, stop]}`이며,
오프셋은 파일이 아니라 **데이터 세그먼트의 시작**을 기준으로 한다. 선택적인 `"__metadata__"`
키는 문자열 객체에 매핑되며 텐서가 아니다; 양쪽 리더 모두 이를 건너뛴다. Rust는 헤더를
`BTreeMap`에서 작성하므로 바이트열은 오직 내용의 함수이며(spec 3.4), `weights_hash`는
안정적이다.

키는 `nodes.<node_id>.<param path>`이다(설계 노트 4절); Python 쪽은 생성된 모듈의 멤버에
맞추기 위해 이를 `n<node_id>.<param path>`로 rename하며, `load_state_dict`는 strict이므로
Rust 쪽 검사를 통과한 불일치라도 여전히 요란하게 실패한다.

이 프로젝트는 `safetensors` 크레이트나 패키지에 의존하지 않고 포맷을 직접 읽고 쓴다. 양방향
각각 ~40줄이면 되고, CI venv를 wheel 하나로 유지시켜 주며 — 진짜 이유는 — 이 포맷이
`INV-16`의 핵심을 떠받치는 부분이라, 읽을 수 있는 형태로 저장소 안에 두는 것이 가치가 있기
때문이다.

## 알려진 공백

- `torchvision`은 lowering에서 참조되지만 이를 실행하는 테스트는 아직 없다; M1 ACT 체크포인트
  게이트(spec 8.9)가 그것을 다룰 자리다.
- 실제 LeRobot ACT 체크포인트는 이 스킴으로의 키 리맵이 필요하다 — 설계 노트 5절 참고.
- `"epsilon"`이 아닌 `prediction_type`은 `LowerError::Unsupported`다: lowering된
  denoiser는 `eps_theta`이며, `"sample"` / `"v_prediction"`은 다른 `pred_original_sample`을
  요구한다.
- `timestep_spacing`은 `diffusers`의 `"leading"` 기본값으로 고정되어 있다;
  `"linspace"` / `"trailing"`은 모델링되지 않으며, `steps_offset != 0`,
  `rescale_betas_zero_snr`, `thresholding`, `DDIMScheduler`의 `eta > 0`도 마찬가지다.
- 배치 없음. 생성된 모듈은 샘플 하나를 기준으로 작성된다; `Linear`는 어차피 선행 축을
  브로드캐스트하지만, `PolicyHead`의 `reshape(H, A)`가 그 컨벤션을 고정하고 있어 바꿔야 할
  것이다.
- 시작 비용은 Python 인터프리터 전체에 torch import까지 더해 대략 1초다. 이것은 오라클이지
  처리량 경로가 아니다(spec 2.4); `libtorch` FFI는 같은 trait에 다른 구현체일 뿐이다.
