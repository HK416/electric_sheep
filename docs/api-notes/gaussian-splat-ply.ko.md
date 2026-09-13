<!-- Korean translation of docs/api-notes/gaussian-splat-ply.md. The English file is the working copy; regenerate this when it changes. -->
# 3D 가우시안 스플래팅(Gaussian Splatting) PLY 레이아웃

핀 버전: **없음**. 이 노트의 어떤 내용도 실행 중인 레퍼런스 구현과 대조 검증되지 않았다 — 이 워크스페이스에는 3DGS 트레이너도, `.ply` 캡처도, Python 의존성도 없다. API 노트는 검증된 내용만 기록한다는 §1.7 규칙에 따라 **아래 모든 필드는 `unverified`(미검증)로 표시**되며, 실제 캡처를 신뢰하기 전에 INRIA 레퍼런스(`graphdeco-inria/gaussian-splatting`, `scene/gaussian_model.py`, `save_ply` /
`construct_list_of_attributes`)와 대조 재검증해야 한다.

이 노트는 *포맷* 노트이지 라이브러리 노트가 아니다: `es-splat`은 이 바이트들을 직접 읽고 쓴다(PLY 헤더 하나와 패킹된 `f32` 몇 개는 의존성을 들일 가치가 없다). 따라서 여기서 중요한 것은 프로퍼티 이름, 그 순서, 그리고 각각에 필요한 변환이다.

## 컨테이너

표준 Stanford PLY. 실제로 쓰이는 것은 세 인코딩 중 둘뿐이다.

```
ply
format binary_little_endian 1.0        # or: format ascii 1.0
element vertex <N>
property float x
...
end_header
<N * (sizeof of the declared properties) bytes, or N whitespace-separated lines>
```

- `format binary_big_endian 1.0` — `unverified`(미검증), 관측된 적 없음; `es-splat`은 추측하는 대신 이를 거부한다.
- 레퍼런스 라이터의 줄 끝은 `\n`이다; 실제 리더들은 `\r\n`도 허용한다.
- `element vertex`는 정확히 하나뿐이다; 다른 element(예: `face`)는 나타나지 않는다. 3DGS PLY에는 연결성(connectivity)이 없다.
- 레퍼런스 라이터에서 모든 프로퍼티는 `float`(`f32`)이다. 일부 익스포터에서는 `double`/`uchar` 컬럼이 나타난다 — `unverified`(미검증)이며 지원하지 않는다.

## 정점 프로퍼티 (사실상 표준 순서)

| 프로퍼티 | 개수 | 의미 | 임포트 시 변환 | 상태 |
|---|---|---|---|---|
| `x` `y` `z` | 3 | 가우시안 중심, 재구성 결과의 월드 단위 | 축 보정(아래 참조); 단위 스케일링은 **없음** — 3DGS 월드는 미지의 similarity 변환까지만 정해지며, 이는 §16.2의 위치 정렬이 해결하는 부분이다 | `unverified` (미검증) |
| `nx` `ny` `nz` | 3 | 레퍼런스 라이터가 0으로 기록함; 3D 가우시안에는 법선이 없다 | **무시**(읽고 버림) | `unverified` (미검증) |
| `f_dc_0..2` | 3 | RGB 채널별 SH 0차 계수 | `rgb = 0.5 + C0 * f_dc`, `C0 = 0.28209479177387814` | `unverified` (미검증) |
| `f_rest_0..44` | 45 | SH 1..3차, **채널 우선(channel-major)**: R용 15계수, 이어서 G용 15계수, 이어서 B용 15계수 | 없음(저장된 그대로 유지; 아래 주의사항 참조) | `unverified` (미검증) |
| `opacity` | 1 | 알파의 **로짓(logit)** | `alpha = sigmoid(opacity)` | `unverified` (미검증) |
| `scale_0..2` | 3 | 가우시안 **자체 프레임**에서 축별 표준편차의 **자연로그** | `sigma = exp(scale_i)`; 축 보정 없음 — 이들은 월드 방향이 아니라 로컬 프레임 범위(extent)다 | `unverified` (미검증) |
| — | — | — | 위의 `exp`/`ln`/`sigmoid`/`logit`은 호스트 `libm`이 아니라 `es_math::approx`(`f32`, ULP 단위 오차 제한, IEEE 정확 반올림 아님)를 거친다. 이들이 `SplatScene::asset_hash`(§5.3)에 반영되어 모든 기계에서 동일한 비트를 내야 하기 때문이다(§3.2/§3.4). 정확도에 대한 영향은 `docs/design/splat-real2sim.md` §1.1을 참조. | — |
| `rot_0..3` | 4 | 쿼터니언 **wxyz**, **정규화되지 않은** 상태로 저장(트레이너가 렌더 시점에 정규화) | 정규화 후 §3.1의 xyzw 순서로 재배열, `w >= 0`으로 맞춘 뒤 축 보정 적용 | `unverified` (미검증) |

`f_rest`의 길이는 학습된 SH 차수에 따라 달라진다: `3 * ((d+1)^2 - 1)` = d = 0/1/2/3일 때 각각 0/9/24/45. 3차(45)가 레퍼런스 기본값이다. `es-splat`은 `f_rest_*` 개수로부터 `sh_degree`를 추론하며, 이 네 값 중 하나가 아닌 개수는 거부한다.

**`f_rest`의 채널 우선(channel-major) 순서가 여기서 가장 오류가 나기 쉬운 항목이다.** 레퍼런스 라이터는 `(N, 15, 3)` 텐서를 평탄화하기 전에 `(N, 3, 15)`로 전치(transpose)한다. 따라서 `f_rest_0..14`는 모두 R이다. 일부 서드파티 리더는 계수 우선(coefficient-major)을 가정하여 뷰 의존적(view-dependent) 색상에서만 조용히 틀리게 동작하는데, 이것이 실제 캡처를 나란히 렌더링해 대조해보기 전까지 이 항목이 `unverified`(미검증)로 남는 이유다.

**월드 회전 하에서의 SH.** 0차(`f_dc`)는 회전 불변이다. 1..3차는 그렇지 않다: 월드를 회전시키려면 각 밴드에 위그너-D(Wigner-D) 회전을 적용해야 한다. `es-splat`은 `f_rest`를 회전시키지 **않는다** — 계수를 변경 없이 그대로 나르며 경고를 기록한다. 아직 이를 소비하는 곳이 없고(스플랫 래스터라이저는 §16.3 작업이며 Vulkan이 필요하다), §16.2의 정렬 similarity도 캡처마다 다시 피팅되므로, 올바른 밴드 회전은 뷰 의존적 색상을 처음 렌더링하는 패킷의 몫이다.

## 프로퍼티 순서는 보장되지 않는다

레퍼런스 라이터 이외의 익스포터는 컬럼 순서를 바꾸고, 일부는 자체 프로퍼티(`confidence`, `segment_id`, 스플랫별 id 등)를 추가하기도 한다. 따라서 `es-splat`은 프로퍼티를 오프셋이 아니라 **이름으로** 인덱싱하고, 선언된 타입 순서로부터 각 프로퍼티의 바이트 오프셋을 계산하며, 인식되지 않는 프로퍼티는 오류가 아니라 경고로 보고한다. 자체 라이터는 위의 레퍼런스 순서로 기록하므로 라운드트립은 바이트 단위로 안정적이다.

## 압축 variant — 아직 미지원

| 포맷 | 무엇인가 | 왜 미지원인가 |
|---|---|---|
| `.splat` | antimatter15의 뷰어 포맷: 가우시안당 32바이트 — `f32[3]` 위치, `f32[3]` 스케일(로그가 아닌 선형), `u8[4]` RGBA, `u8[4]` 쿼터니언을 `(q * 128 + 128)`로 양자화. 헤더 없음; 개수는 `len / 32`. | 0차 색상만 존재 — 뷰 의존적 SH가 버려지므로, §16 전체가 전제하는 충실도로 §15.1 렌더 경로에 공급할 수 없다. `unverified` (미검증). |
| `.ksplat` | mkkellogg의 GaussianSplats3D 포맷: 버전이 있는 헤더, 공간적으로 버킷화된 스플랫, 16비트 half float, 여러 압축 레벨. | 레이아웃이 버전에 의존하고, 공간 버킷화가 가우시안 순서를 재배열한다 — 이는 §5.3의 파일 순서 기반 `asset_hash`가 익스포터의 버킷화 방식에 좌우되게 만든다. `unverified` (미검증). |
| `.spz` | Niantic의 압축 포맷, §16.3에서 목표로 명시됨. | gzip과 고정소수점 레이아웃이 필요하며, 아직 아무것도 렌더링하지 않는 시점에 의존성을 들일 가치가 없다. `unverified` (미검증). |

이들 중 하나가 도입될 때는 반드시 동일한 `SplatScene`으로 필드 단위까지 정확히 디코딩되어야 한다. `asset_hash`는 파일 바이트가 아니라 디코딩된 가우시안(§5.3)에 대해 계산되기 때문이며, 이 성질 덕분에 "같은 캡처, 다른 컨테이너"가 동일하게 해시된다.
