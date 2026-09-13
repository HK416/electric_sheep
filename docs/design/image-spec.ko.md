<!-- Korean translation of docs/design/image-spec.md. The English file is the working copy; regenerate this when it changes. -->

# `ImageSpec`과 그 변환 규칙 (`es-ir::image`) — 설계

Spec refs: spec 7.2 (`ImageSpec`, `OBS-021`, `OBS-034`), spec 3.1 (coordinate
conventions), Appendix B.2, `INV-14`. Packet: `docs/packets/M0/P19.md`.

## 규약 (spec 3.1, 협상 불가)

- **이미지 좌표**: 원점은 좌상단, `+x`는 오른쪽, `+y`는 아래쪽 (OpenCV).
- **카메라 좌표**: `+Z`는 전방, `+X`는 오른쪽, `+Y`는 아래쪽 (OpenCV 광학 프레임,
  ROS `REP-103` 광학 프레임과 동일).
- **기본 색공간**: sRGB (비선형). 선형화는 항상 명시적인 노드이며 암묵적으로
  이루어지지 않는다.

`extrinsics`는 `es_math::conventions::Pose`(m 단위 위치 + `w >= 0`인 xyzw 정규 단위
쿼터니언)로 표현된 `T_body_camera`다. `Pose`를 재사용함으로써 저장소 전체가 회전
규약을 하나만 갖는다.

## 변환이 타입 안에 있는 이유

`fx, fy, cx, cy`를 갱신하지 않고 이미지를 리사이즈하거나 크롭해도 실패하지 않는다 —
조용히 잘못된 3D를 돌려줄 뿐이다. 그래서 기하를 바꾸는 세 연산은 *스펙 자체의
메서드*로 두었고, Observation IR의 `Resize` / `Crop` / `Undistort` 노드는 반드시
이들을 거쳐야 한다 (`INV-14`). 컴파일러는 결과로 나온 `ImageSpec`을 그래프를 따라
전파하고, `CameraProjection` 노드에서 `intrinsics_consistent_with`를 호출한다 — 이
비교가 `OBS-034`의 출처다.

## 변환 규칙 (Appendix B.2)

| method | width/height | intrinsics | distortion |
|---|---|---|---|
| `resized(w, h, rescale)` | `w, h` | `rescale`이면 `scaled(w/width, h/height)`, 아니면 변경 없음 | 변경 없음 |
| `cropped(rect, rescale)` | `rect.width, rect.height` | `rescale`이면 `cx -= rect.x`, `cy -= rect.y`, 아니면 변경 없음 | 변경 없음 |
| `undistorted(new_intr)` | 변경 없음 | `new_intr` | `None` |

`Intrinsics::scaled(sx, sy)`는 `fx, skew`를 `sx`로, `fy`를 `sy`로, `cx`를 `sx`로,
`cy`를 `sy`로 스케일한다.

이는 Appendix B.2의 단순 스케일링, 즉 픽셀 *코너(corner)* 규약이다. 픽셀 *중심
(centre)* 규약에서는 정확한 형태가 `cx' = (cx + 0.5)·sx − 0.5`이지만, 그 차이는 반
픽셀 미만이고 사양이 단순한 형태를 고정하므로 그것이 구현된 형태다. 이에 동의하지
않는 리샘플러는 이 규칙을 다시 쓰는 것이 아니라 자신의 노드에서 그 사실을 명시해야
한다.

`rescale = false`는 **사용자의 명시적 옵트아웃**일 뿐, 그 이상도 이하도 아니다.
호출자가 이미 리사이즈된 스트림에 대한 intrinsics를 따로 측정해 두었을 수 있기
때문에 존재한다. 기본값이 아니며, `false`를 넘기는 노드가 그 근거를 대야 하는
노드다. 하위 단계에서 `intrinsics_consistent_with`는 여전히 그 불일치를 `OBS-034`로
보고한다 (그것을 강등할지는 호출자가 결정한다).

## `intrinsics_consistent_with(&other)`

`self`의 intrinsics가, 두 해상도 사이의 순수 리사이즈를 거친 뒤의 `other`의
intrinsics와 같을 때 참이다.

```
sx = self.width / other.width,  sy = self.height / other.height
self.intrinsics ≈ other.intrinsics.scaled(sx, sy)      (relative tolerance 1e-9)
```

따라서 해상도가 같다면 intrinsics도 같아야 하며, 흔한 버그 — 640×480 → 224×224로
리사이즈하면서 원래의 `fx`를 그대로 두는 것 — 는 거짓으로 판정된다. 스케일 인자는
정수의 정확한 비율이지만 `scaled`는 `f64`로 곱하므로 허용오차는 상대적이다.

## 패킷 브리프에 없는 필드

구조체는 spec 7.2를 온전히 따른다: `channels` (`ChannelFormat`), `dtype`
(`ImageDType`), `rate_hz`, `depth_scale`도 함께 담는다. `color_space`는 `SRgb |
Linear | Rec709 | Raw`다 (spec 7.2) — `Gray`와 `Depth`는 색공간이 아니라 채널
포맷이다. `OBS-021`(비전 인코더에서의 색공간 불일치)은 컴파일러의 검사 항목이며,
이 모듈은 그 필드와 코드값만을 담는다.
