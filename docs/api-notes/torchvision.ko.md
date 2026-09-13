<!-- Korean translation of docs/api-notes/torchvision.md. The English file is the working copy; regenerate this when it changes. -->

# torchvision / torch — observation golden을 위한 고정 API 표면

`crates/es-compile/python/gen_observation_goldens.py`가 실제로 호출하는 것들을 정리한
문서로, torch를 업그레이드할 때 발굴 작업이 아니라 이 파일에 대한 diff 확인만으로 끝나게
하기 위한 것이다. `docs/api-notes/torch.md`(`TorchRuntime` 표면)와 짝을 이루며, 형태와
이유가 동일하다.

명세: spec 1.4 (golden은 테스트 대상 코드가 아니라 참조 오라클에서 나와야 한다),
spec 7.7 (LeRobot 전처리 동등성), spec 11.3 (CPU 계획이 정답이다). 이 호출들이 고정하는
의미론은 `docs/design/observation-lowering.md`에 있다; 패킷은
`docs/packets/M1/P-M1-R1.md`다.

## 버전

| | |
|---|---|
| 검증 대상 | **torch 2.14.0+cpu**, **torchvision 0.29.0+cpu** (`--index-url https://download.pytorch.org/whl/cpu`) |
| Python | 3.12.10 |
| `numpy` | **2.5.2** — 필요하며, torchvision wheel이 함께 끌어온다; `to_tensor`는 텐서가 아니라 PIL 이미지나 ndarray를 받는다 |

로컬 실행을 위한 설치:

```
python -m venv <short path>            # a deep path fails on Windows without long-path support
<venv>/Scripts/python -m pip install torchvision --index-url https://download.pytorch.org/whl/cpu
ES_PYTHON=<venv>/Scripts/python cargo test -p es-compile --test gen_goldens
```

`ES_PYTHON`은 MuJoCo와 `TorchRuntime` 오라클이 쓰는 것과 같은 변수이므로, venv 하나로
셋 다 처리할 수 있다. 이 변수가 없으면 `python`, 그다음 `python3` 순서로 시도한다;
어디에도 torchvision이 없으면 `gen_goldens`는 `SKIPPED`를 출력하고 통과하며, 다른 모든
테스트는 그대로 실행된다.

## 이 코드가 의존하는 API

| API | golden | 비고 |
|---|---|---|
| `torchvision.transforms.functional.to_tensor(ndarray HWC u8)` | `dequantize_8x6_rgb` | `permute(2,0,1).contiguous().float().div(255)`. 커널의 `f32::from(u8) / 255.0`과 비트 단위로 동일하다 — 양쪽 모두 f32로 나눈다 |
| `torch.nn.functional.interpolate(x, size, mode="bilinear", align_corners=False, antialias=False)` | `resize_bilinear_8x6_to_4x3` | 반화소 중심(half-pixel centres). `antialias`는 **반드시** 명시적으로 넘겨야 한다: torchvision의 `Resize`는 0.17부터 `antialias=True`가 기본값이며, 그것은 이 알고리즘의 개선판이 아니라 다른 필터다 (설계 노트 §11) |
| tensor slicing `chw[:, y:y+h, x:x+w]` | `crop_8x6_at_2_1_4x4` | 의도적으로 `TF.crop`을 *쓰지 않는다* — 그것은 범위를 벗어난 사각형을 패딩하는데, IR은 그런 것을 컴파일 시점에 거부한다(`COMPILE-003`) |
| `torch.float64` arithmetic, `.to(torch.float32)` | `srgb_to_linear_lut256` | IEC 61966-2-1 EOTF, 0.04045 미만은 `x/12.92`, 그 이상은 `((x+0.055)/1.055) ** 2.4`, f64로 계산되고 한 번 반올림된다. torchvision 호출이 아니다 — torchvision에는 sRGB EOTF가 없고, f64에서의 `**`가 구할 수 있는 가장 정확한 참조값이다 |
| `torchvision.transforms.functional.normalize(t, mean, std)` | `normalize_imagenet_4x3` | `sub_(mean).div_(std)`: 나눗셈이며, 역수 곱셈이 아니다 |
| `torch.stack(frames, dim=0)` | `history_window_n2_s1` | 오래된 것 → 새것, 현재 프레임이 마지막 |

## 결정성

스크립트는 `torch.use_deterministic_algorithms(True)`, `torch.manual_seed(0)`,
`torch.set_num_threads(1)`을 설정한다. 오늘은 그중 어느 것도 RNG에서 추첨하지 않는다;
시드는 어떤 golden이 언젠가 그렇게 하더라도 이것이 계속 참이도록 설정되어 있다. 모든
값은 CPU f32 또는 f64다 — CUDA도, TF32도, autocast도 없다.

## 무엇이 비트 단위로 동일하지 않고, 왜 그런가

여섯 golden 중 다섯은 Rust 커널에 의해 비트 단위로 재현된다. 여섯 번째인
`srgb_to_linear_lut256`은 사이드카에 `"tolerance_ulp": 7`을 싣고 있다: `DET-010`이
observation 커널에서 `powf`를 금지하므로, `srgb_eotf`는 2.4 거듭제곱을 f32에서
`es_math::approx::exp(2.4 * ln t)`로 계산하며 f64 참조값의 비트에 도달할 수 없다. 7
ULP(상대 오차 4.8e-7)는 테스트를 통과시키기 위해 고른 여유값이 아니라, 256개 항목
전체에 대해 측정된 최악의 경우다.

`torch.nn.functional.interpolate`는 고정된 golden 크기에서는 비트 단위로 일치하지만
모든 크기에서 그런 것은 아니다: 4,624개의 (source, target) 크기 쌍을 스윕한 결과, 1,585개는
비트 단위로 동일하고 나머지는 1~4 ULP 차이가 난다. 반화소 인덱스는 정확히 일치한다;
탭 가중치는 그렇지 않다. 일치시키지 않고 기록만 해 둔다 — 설계 노트 §12 항목 6 참고.

## 업그레이드 절차

1. 새 wheel들을 설치하고, `ES_PYTHON`을 설정한 채로
   `cargo test -p es-compile --test gen_goldens`를 실행한다. 실패하며, 바이트가 바뀐
   파일을 지목한다.
2. 그 변화가 torch 버그 수정(받아들임)인지 의미론 변경(새 커널 id와 설계 노트 변경이
   먼저 필요함)인지 결정한다.
3. 스크립트를 `tests/golden/observation`을 가리키게 하여 재생성하고, 위의 버전 표를
   갱신하며, `GOLDEN_UPDATE=1 cargo xtask verify-goldens`로 커밋을 게이트한다.
