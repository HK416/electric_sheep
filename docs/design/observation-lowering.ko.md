<!-- Korean translation of docs/design/observation-lowering.md. The English file is the working copy; regenerate this when it changes. -->

# Observation IR → CPU 레퍼런스 실행 계획

`crates/es-compile` (layer 7)를 위한 설계 노트. Spec: spec 7.2, spec 7.3, spec 7.5,
spec 7.6, spec 7.7, spec 11.1, spec 11.3, spec 11.5, spec 3.1, spec 3.4. Work
packet: `docs/packets/M1/W3-observation-cpu-ref.md`.

리뷰 등급 C (수치에 대한 사람 리뷰): 아래 내용 전부가 golden들과, 나중에는 GPU
lowering이 재현해야 할 비트 패턴을 고정한다. resize 컨벤션을 잘못 잡는 실수는
조용하다 — 이미지는 여전히 멀쩡해 보이지만 정책은 학습 때 보지 못한 것을 입력받게
된다.

## 1. 이것은 무엇이고 무엇이 아닌가

Spec 11.3: **CPU 경로는 성능 경로가 아니라 정답(ground truth)이다.** 이것은
Slang/SPIR-V lowering(spec 11.4)이 그것에 대해 판정받는 오라클이며, Slang 없이
동작하므로 CI와 macOS가 이를 게이트로 쓸 수 있고, 모든 노드의 중간값을 보여줄 수
있다.

이 노트에 없는 것: fusion, GPU 버퍼, `PolicyRuntime` 호출 계획. 그것들은 이후
웨이브의 몫이다.

## 2. 텐서 모델

```rust
Tensor { dtype: ElemType, shape: Vec<u64>, data: Vec<u8> }
```

행 우선(row-major), 빈틈없이 채워짐(tightly packed), 패딩 없음, 배치 축 없음
(spec 5.4: 배치 축은 shape가 아니라 배치 도메인에 속한다).

**레이아웃 규칙, 딱 두 가지 레이아웃만:**

| 위치 | 레이아웃 | dtype |
|---|---|---|
| sensor 경계 (`ImageInput`) | **HWC** | `U8` |
| 그 이후 전부 | **CHW** | `F32` (출력에서는 `F16`/`Bf16`도 가능) |

CHW f32는 PyTorch가, 따라서 LeRobot이 정책에 공급하는 형태이며, 그것을
일치시키는 것이 바로 spec 7.7("LeRobot 전처리 동등성")의 목적이다. HWC u8은
모든 카메라 드라이버와 모든 LeRobot parquet/video 프레임이 실제로 넘겨주는
형태다. 따라서 경계가 아닌 다른 곳에서 변환한다면 두 번 변환하는 셈이 된다.

레이아웃 변경은 단 하나의 노드 종류에서만 일어난다: `Dequantize` — 그리고 그
융합된(fused) 짝인, u8 포트를 직접 읽는 `ColorTransform`(§6)은 EOTF가 접혀
들어간 같은 변환이다. 따라서 `Dequantize`는 torchvision `ToTensor()`에
해당하는 IR 표현이다 — HWC u8을 255로 나누어 CHW f32로 만든다. 계획 안의
그 무엇도 이 밖에서는 축을 뒤바꾸지 않는다.

상태 텐서(`StateInput`)는 1차원이며 레이아웃 문제가 없다.

## 3. 노드 → 커널 표

이것은 M1 W3 part-1 서브셋이다. 이 목록 밖의 노드는 조용한 no-op이 아니라
컴파일 시점의 `COMPILE-002` 진단이다.

| Observation IR 노드 | 커널 | 비고 |
|---|---|---|
| `ImageInput` | — | 계획 입력 버퍼를 명명; u8 HWC |
| `StateInput` | — | 계획 입력 버퍼를 명명; f32 1-D |
| `Dequantize` | `cast_u8_hwc_to_f32_chw` | `/255`, 유일한 레이아웃 변경 |
| `Resize{Bilinear}` | `resize_bilinear` | §4 |
| `Resize{Nearest}` | `resize_nearest` | `floor(scale * (d + 0.5))`, PyTorch의 규칙 |
| `Crop{Rect\|Center\|Random}` | `crop` | §5 |
| `ColorTransform{SRgb→Linear}` | `srgb_to_linear` / LUT | §6 |
| `Normalize{MeanStd\|Range}` | `normalize` | §7 |
| `Concat{axis}` | `concat` | §8 |
| `Stack{axis}` | `stack` | §8 |
| `TemporalWindowNode` | `history_push` + `window_gather` | §9 |
| 출력 elem ≠ `F32` | `cast_f32_to_f16` / `_bf16` | `half`, round-to-nearest-even |
| `Pad`, `Undistort`, `Rectify`, `Warp`, `CameraProjection`, `ToGray`, `ChannelSelect`, `QuantizeU8`, `FrameStack`, `Delta`, `Mask`, `MultiViewPack`, `Augment`, `LanguageInput` | — | `COMPILE-002`, 이후 웨이브 |

`Augment`는 RNG 스트림 계약이 생기기 전까지는 이 경로에서 구현되지 않은 채로
남는다. 이것은 `training_only`이며(spec 7.3), 이것이 잘못 동작할 평가 경로가
가장 먼저 중요한 경로이기 때문이다.

### 모든 커널이 따르는 결정성 규칙 (spec 3.4)

단일 스레드, 고정된 루프 순서(작성된 대로 정확히 바깥→안), 원자적 연산 없음,
`HashMap` 순회 없음, `f64` 누산기 없음, `std` 초월함수 없음 — 오직
`es_math::approx`만 사용하므로 Rust와 미래의 Slang 커널이 계수와 평가 순서를
*모두* 공유한다. 이 커널들의 누산은 많아야 네 항에 대해 이루어지므로 아직은
`es_math::reduce`가 필요 없다. 미래의 `Area` resize나 탭이 많은 `ToGray`는
그것이 필요해질 것이다.

## 4. Resize: bilinear, `align_corners = false`, 반화소 중심

목표 의미론: `torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear",
align_corners=False, antialias=False)`, 이는 torchvision의
`Resize(..., antialias=False)`이기도 하다.

길이 `D`인 축의 출력 인덱스 `d`마다, `scale = S / D`일 때:

```
s      = scale * (d + 0.5) - 0.5
s      = max(s, 0.0)                  // PyTorch clamps the negative edge, not the positive
i0     = floor(s)                     // as integer
i1     = min(i0 + 1, S - 1)
l1     = s - i0                       // in [0, 1]
l0     = 1 - l1
```

그리고 표본은, 정확히 다음 결합 순서로 계산된다(이는 PyTorch의 순서이며,
재결합하면 마지막 비트들이 달라진다):

```
out = h0 * (w0 * v[y0][x0] + w1 * v[y0][x1])
    + h1 * (w0 * v[y1][x0] + w1 * v[y1][x1])
```

모든 산술은 f32다. `s`는 f32인 `scale`로부터 f32로 계산된다. f64로 계산한
다음 좁히면 어떤 크기에서는 다른 비트가 나오므로, f32 경로가 규범적이다.

이것은 상수 이미지에 대해서도 **정확하지 않다**. 탭들이 `l0 = 1 - l1`로
결합되는데, 이 f32 합이 정확히 1이 아니므로, 균일한(flat) 입력이 1 ulp
어긋나게 돌아올 수 있다. 그것은 PyTorch의 동작이기도 하며, PyTorch와
일치시키는 것이 계약이다 — proptest는 동등성이 아니라 2 ulp를 확인한다.
상수에 대해 정확한 결합 방식이라면 다른 커널이 될 것이다.

**안티에일리어싱은 구현되지 않는다.** 다운스케일에서 `antialias=True`는
이 알고리즘의 개선판이 아니라 다른 알고리즘(지지 영역을 넓힌 필터)이다.
§11을 참조.

## 5. Crop

`Crop`은 `(rect.x, rect.y)`에서 시작해 `rect.width × rect.height`를 채널
평면별로 복사한다. `CropMode::Center`와 `CropMode::Random`은 중심 사각형
`((W - w) / 2, (H - h) / 2, w, h)`로 해석된다 — 이는 타입 시스템이 `CropMode::rect`에
주는 것과 같은 사각형이므로, 계획의 기하학과 전파된 `ImageSpec`은 서로
어긋날 수 없다. Random offset은 증강(augmentation)이며 `Augment`(§3)에
속한다.

이미지 밖으로 벗어나는 사각형은 실행 시점의 clamp가 아니라 컴파일 시점의
`COMPILE-003`이다.

**INV-14.** 계획은 `Intrinsics`를 절대 건드리지 않는다. 모든 기하학적 노드는
`ObservationIr::propagate_image_specs`가 만들어낸 `ImageSpec`을 받으며, 이는
`ImageSpec::resized` / `ImageSpec::cropped`로 만들어진다. 따라서 연쇄된 crop들은
계획의 픽셀들과 같은 방식으로 합성된다: `cropped(a).cropped(b)`는 주점(principal
point)을 `a.x + b.x`만큼 이동시키고, 합쳐진 사각형으로 한 번에 crop하는 것은
같은 intrinsics와 같은 픽셀을 준다. 그 동등성은 proptest로 검증된다.

## 6. 색: sRGB → linear

Spec 3.1: **sRGB가 기본값이며 선형화는 오직 명시적 노드를 통해서만 일어난다.**
Observation IR은 이미 선언된 `src`가 들어오는 `ColorSpace`와 다른
`ColorTransform`을 거부하므로(`OBS-021`), 커널은 절대 추측할 필요가 없다.

스칼라 EOTF는 채널별로 적용되며, 알파는 건드리지 않는다(지원되는 채널
포맷에는 아직 알파가 없다):

```
srgb_eotf(x) = x <= 0.04045 ? x / 12.92
                            : approx::exp(2.4 * approx::ln((x + 0.055) / 1.055))
```

`powf`는 observation 커널에서 금지되므로(spec 3.2, `DET-010`), 2.4 거듭제곱은
`es_math::approx`를 통한 `exp(2.4 * ln(t))`이며, 그 계수는 Slang 미러와
공유된다. 이는 `libm` 대비 약간의 정확도를 대가로 치르지만, 여기서 중요한
단 하나, 즉 CPU 오라클과 GPU 커널이 *같은* 비트를 낸다는 것을 얻는다.

**LUT-256.** 노드의 입력이 u8일 때(`Dequantize`보다 앞서 `ImageInput`에
직접 연결된 `ColorTransform`), 입력은 256가지 값만 가지므로, 계획은
`srgb_eotf(k / 255)`를 `k`마다 한 번씩 계산해 `[f32; 256]` 테이블에 넣고
그것을 인덱싱하여 CHW f32를 쓴다 — dequantize가 접혀 들어가 있으므로,
두 단계짜리 형태와 이 융합된 형태는 같은 텐서를 낸다. 이 테이블은 같은
`srgb_eotf`로 만들어지므로, 두 경로는 리뷰가 아니라 구성상 일치한다. 이
노드의 golden은 바로 그 테이블이다.

`Linear → SRgb`(역변환, spec 7.7의 왕복 오라클에 필요)는 이 part에서는
구현되지 않는다 — 왕복 오라클은 다음 웨이브이며, LUT는 정확히 역변환되지
않는다.

## 7. Normalize

CHW, 채널 `c`마다:

- `MeanStd { mean, std }`: `(x - mean[c]) / std[c]`. 곱셈이 아니라 나눗셈이다 —
  PyTorch의 `normalize`도 나누며, 역수 형태는 마지막 비트가 다르다.
  `mean`/`std`는 IR에서 `f64`이며 계획 시점에 한 번 `f32`로 좁혀진다.
  `len(mean) == len(std) == C`이거나 `COMPILE-004`이다.
- `Range { lo, hi }`: `(x - lo) / (hi - lo)`, 모든 채널에 같은 스칼라.

`Normalize`의 출력 단위는 반드시 `Unit::Normalized`여야 한다 — IR이 이미
그것을 강제하므로(`OBS-040`), 계획은 그 검사를 반복하지 않는다.

## 8. Concat과 stack

`Concat { axis }`는 기존 축을 따라 입력들을 이어붙이며, 그 외 모든 축은
일치해야 한다. `Stack { axis }`는 `axis`에 길이 `n_inputs`의 새 축을
삽입한다. 모든 입력은 같은 shape여야 한다. 둘 다 선언된 입력 포트 순서
(`in0`, `in1`, …)로 이루어지는 순수한 바이트 복사이며, 이는 엣지 목록이
정렬되는 순서이기도 하다. 따라서 결과는 그래프가 어떻게 작성되었는지에
의존하지 않는다.

`time_align`은 여기서는 실행 시점 연산이 아니다. 시계 간의 정렬은 IR의
타입 검사(`TYPE-014`)와, 입력 버퍼를 채우는 쪽에 의해 결정된다. 계획은
복사만 한다.

## 9. History와 window (spec 7.5)

세 계층이 있으며, 그중 가운데 것만 노드다:

- `History<T, N>` — ring 버퍼, `es-core`/`es-sensor`가 소유하며 IR 바깥에
  있다. 계획은 CPU 레퍼런스가 독립 실행되어야 하기 때문에 *윈도우가 있는
  스트림당 하나의 ring*만을 소유한다. 실제 배포에서는 ring이 외부에서
  건네진다.
- `TemporalWindow(n_steps, stride, align)` — 노드. shape을 가진다.
- `TemporalEncoder` — Learning IR. 여기 없다.

```rust
history_push(ring, slot_len, depth, cursor, frame)   // writes slot cursor % depth
window_gather(ring, slot_len, depth, cursor, n, stride, dst)
```

`window_gather`는 오래된 것에서 새것 순으로 낸다. 즉 `k = 0..n`에 대해
슬롯 `cursor - (n - 1 - k) * stride`이며, 따라서 출력의 마지막 행은 항상
현재 프레임이다 — 이는 음수 오프셋 observation window에 대해 LeRobot의
`delta_timestamps`가 쓰는 컨벤션이다. ring이 아직 채워지기 전에는, 가장
오래된 사용 가능 프레임이 반복된다(`Align::Hold`). `Align::Interpolate`와
`Align::Reject`는 지금은 `COMPILE-002`다.

출력 shape는 `[n_steps, ...frame_shape]`이다.

## 10. 계획, 버퍼, 디버그 대 릴리스 (spec 11.5)

`CpuPlan::compile`은 다음을 실행한다: `ObservationIr::validate`(오류는
중단, 경고는 유지됨), `propagate_image_specs`, `topo_order`, 그다음 각
노드의 `out` 포트에 `BufferId`를 배정하는 한 번의 패스이며, 그 dtype과
shape는 노드의 **선언된** `PortType`에서 온다 — IR이 계약이며, 계획은
자신이 어긋날 수 있는 shape을 다시 유도하지 않는다. 크기들은 `run`이 한
번 할당하는 하나의 평평한 arena로 합산된다. `BufferId`는
`(offset, len, dtype, shape)`이다.

Spec 11.5는 두 가지 계획을 요구한다:

| | 릴리스 | 디버그 |
|---|---|---|
| fusion | 최대 | 노드 경계 유지 |
| 노드별 값 | 샘플링됨 | 전부 |

이 경로에서는 **두 모드 모두 모든 중간값을 주소 지정 가능하게 유지하며
arena는 aliasing되지 않는다.** 이 CPU 레퍼런스 경로에는 fusion이 없다
(fusion은 GPU lowering의 몫이며, CPU 계획이 존재하는 이유가 바로 노드
경계를 *갖기* 위해서다). 그러므로 liveness 기반 aliasing은, 메모리 경로가
아닌 경로에서 메모리를 절약하면서, 그것이 존재하는 유일한 이유를 제거하는
셈이 된다. `PlanMode`는 계획에 기록되어 `compiler_hash`에 도달하므로,
나중에 aliasing을 하는 릴리스 계획이 생기면 캐시된 것을 조용히 재사용하는
대신 다른 해시를 얻게 된다.

`compiler_hash()` = (`es-compile` 크레이트 버전, 계획 모드, 표에 있는 모든
커널 id를 순서대로)에 대한 blake3다. 이는 `execution_hash`(spec 5.3, spec
11.2)의 `compiler` 슬롯을 채운다: 커널의 수치를 바꾸는 일은 반드시 이를
바꿔야 하며, 그것이 소스가 아니라 id들이 해시되는 이유이자, 새 커널이
삽입되지 않고 항상 뒤에 추가되는 이유다.

## 11. GPU lowering (미룸) — 각 커널이 어떻게 미러링될 것인가

지금 적어 두는 이유는 CPU 커널들이 GPU가 따라올 수 없는 모양으로 만들어지지
않게 하기 위해서다.

| 커널 | GPU 형태 |
|---|---|
| `resize_bilinear` / `_nearest` | 출력 픽셀당 스레드 하나; f32로 동일한 인덱스 산술. 텍스처 샘플러는 **사용하지 않는다**: 그 필터링 정밀도는 벤더마다 다르게 정의된다. |
| `crop` | 소비자 안으로 인덱스 오프셋으로 융합됨 |
| `srgb_to_linear` | 원소별, `es-math/slang/approx.slang`의 `approx::exp`/`ln` (같은 계수). LUT는 텍스처가 아니라 256개 항목의 uniform 버퍼가 된다. |
| `normalize` | 원소별, 위와 하나의 커널로 융합됨 |
| `cast` | 원소별; f16은 RTE를 사용하는 SPIR-V `Float16` capability를 통해 |
| `concat` / `stack` | 복사, 또는 producer들을 목적지의 서브 범위로 계획하여 없앰 |
| `history_push` / `window_gather` | ring은 device 메모리에 있음; gather는 인덱스 리맵 |

Fusion 경계 (spec 11.4): 리덕션, shape 변경, sensor 읽기, 그리고 — 디버그
모드에서는 — 모든 노드 경계.

## 12. LeRobot에 대해 무엇이 `unverified`인가

spec 12.4의 규칙에 따라, 미검증은 암시되지 않고 명시된다.

1. **LeRobot이 실제로 어떤 resize를 호출하는지 — `unverified`이며 리뷰의
   최우선 질문.** 이 노트는
   `interpolate(..., align_corners=False, antialias=False)`를 구현한다.
   torchvision의 `Resize`는 0.17부터 `antialias=True`가 기본값이었고,
   서로 다른 LeRobot 정책(ACT, Diffusion Policy, SmolVLA, π₀)은 서로 다른
   곳에서 resize한다 — 데이터셋 transform, processor, 또는 아예 하지
   않음. 그 호출 지점 중 하나를 읽고 확정하기 전까지는, spec 7.7이 주장하는
   일치는 어떤 다운스케일에 대해서도 검증되지 않은 채로 남는다. 답이
   `antialias=True`라면, 안티에일리어싱 커널은 이 커널의 변경이 아니라
   추가 커널 id가 된다.
2. **u8 → f32 스케일링 — `unverified`.** `/255`는 `ToTensor`와 일치한다.
   일부 LeRobot 경로는 torchcodec/ffmpeg가 디코딩한, u8 양자화가 전혀
   일어나지 않은 이미 float인 video 프레임을 넘겨준다.
3. **`Normalize` 통계 — `unverified`.** 계획은 IR이 싣고 있는 것을 그대로
   적용한다. 그것이 LeRobot의 데이터셋별 `mean`/`std`인지 ImageNet의
   것인지는 한 계층 위의, 데이터셋에 관한 질문이다.
4. **f16 반올림 — `unverified`.** `half`는 round-to-nearest-even을 쓴다.
   PyTorch의 `.half()`도 그렇지만, 이는 확인된 것이 아니라 단언된 것이다.
5. **`n_steps` 프레임이 아직 존재하지 않을 때의 ring 버퍼 채움 동작 —
   `unverified`.** LeRobot의 `delta_timestamps`는 에피소드의 첫 프레임으로
   clamp하는데, 이는 여기서 `Align::Hold`가 하는 것과 같지만, 그 동등성은
   아직 실행되어 확인된 적이 없다.

## 13. Goldens

`tests/golden/observation/*.bin`과 파일당 하나의 `.json` 사이드카(shape,
dtype, kernel, 그것이 무엇을 고정하는지에 대한 한 문장). 한 번
`cargo test -p es-compile --test gen_goldens -- --ignored`로 생성된 뒤에는
영원히 읽기 전용이다(spec 1.4; `cargo xtask verify-goldens`가 어떤 수정에
대해서도 실패한다). 하나를 재생성하는 것은 그 뒤에 spec 변경이 있는
의도적인 행위이며, 테스트를 통과시키기 위한 방편이 결코 아니다.

리틀 엔디언 f32/u8, 빈틈없이 채워짐 — arena가 들고 있는 것과 같은 바이트다.
