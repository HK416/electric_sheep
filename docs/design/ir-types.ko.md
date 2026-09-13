<!-- Korean translation of docs/design/ir-types.md. The English file is the working copy; regenerate this when it changes. -->

# IR 공통 타입 시스템 (`es-ir::types`) — 설계

Spec refs: spec 5.4 (공통 타입 시스템), spec 5.2 (batch semantics), spec 3.1 (conventions),
Appendix B.1. Packet: `docs/packets/M0/P18.md`.

## `PortType`

```
PortType = (ElemType, Shape, Unit, Frame, TimeRef, Option<ImageSpec>)
```

`Shape`는 **샘플당 차원만** 담는다. 배치 축은 shape에 절대 들어가지 않는다: 네 개의
배치 도메인(simulation / observation / inference / training, spec 5.2) 각각이 자기
크기를 고르므로, 그중 하나를 고정한 shape는 나머지 세 도메인에서 틀리게 된다.

엣지로 연결된 두 포트는 `elem`, `shape`, `unit`, `time`이 정확히 일치해야 하고,
`frame`은 아래의 `Policy` 예외를 제외하면 일치해야 한다. `image`는 둘 다 없거나 같아야
한다.

## Unit — `m^a · kg^b · s^c · rad^d` 형태의 명명된 단위

곱셈과 나눗셈은 네 개의 지수로 이루어진 기저 위에서 정의된다. 지수 벡터를 가진
명명된 단위는 그 벡터로 변환했다가 되돌아온다.

| Unit | m | kg | s | rad |
|---|---|---|---|---|
| `Dimensionless` | 0 | 0 | 0 | 0 |
| `Length` | 1 | 0 | 0 | 0 |
| `Mass` | 0 | 1 | 0 | 0 |
| `Time` | 0 | 0 | 1 | 0 |
| `Angle` | 0 | 0 | 0 | 1 |
| `Velocity` | 1 | 0 | −1 | 0 |
| `Acceleration` | 1 | 0 | −2 | 0 |
| `AngularVelocity` | 0 | 0 | −1 | 1 |
| `Force` | 1 | 1 | −2 | 0 |
| `Torque` | 2 | 1 | −2 | 0 |
| `Pressure` | −1 | 1 | −2 | 0 |

이 표는 양방향이고 대응이 단사(injective)이므로, `Length / Time = Velocity`와
`Velocity * Time = Length`는 스스로 다시 이름을 얻는다. 이름이 없는 지수 벡터는
`Composite(UnitPowers)`가 되고, 지수가 모두 0인 `Composite`는 `Dimensionless`로
정규화된다.

### 불투명(opaque) 단위

`Current`, `Voltage`, `Quaternion`, `RotationMatrix`, `Normalized { lo, hi }`, `Pixel`,
`Luminance`, `Depth`, `Token`은 지수 벡터가 **없고**, `mul` / `div`에서 거부된다
(`TYPE-010`). 이유:

- `Current` / `Voltage`는 사양의 4-지수 기저에 없는 암페어 축이 필요하다.
- `Quaternion` / `RotationMatrix`는 애초에 기본 단위의 곱이 아니다 — 두 회전을
  곱하는 것은 군(group) 연산이지 차원 연산이 아니다.
- `Normalized { lo, hi }`는 이미 임의의 구간으로 매핑되어 있어, 원래의 물리적
  차원은 구성상 사라진 상태다.
- `Pixel`, `Luminance`, `Token`은 도메인 고유 알파벳 안의 개수(count)다.
- `Depth`는 미터 단위이지만 불투명하게 남겨두는 것은 의도적이다 — depth 이미지가
  단위 대수를 거쳐 슬그머니 `Length`(그다음 `Velocity`)가 되는 것을 막는다. 변환은
  추론이 아니라 명시적인 노드로 이루어진다.

`checked_add`는 두 `Unit` 값이 지수만이 아니라 **enum 값으로서 동일**할 것을
요구한다. `Depth + Length`는 거부되고, `Normalized{0,1} + Normalized{-1,1}`도
마찬가지다.

### 정책 입력 규칙

정책 입력 포트는 `Normalized`, `Dimensionless`, `Token`만 받아들인다
(`PortType::is_policy_input`, 진단 코드 `TYPE-011`). 미터나 라디안 원값을 네트워크에
그대로 흘려보내는 것은 이 규칙이 컴파일 시점에 잡아내는 흔한 버그다 (spec 5.4).

## Frame

`World | LocalOrigin | Body(id) | Sensor(id) | Joint(id) | Camera(id) | Image(id) |
Policy`이며, id는 `es_core::StableId`다. Frame은 서로 같아야 한다 — 다만 `Policy`는
무엇과도 호환된다: "네트워크 내부이므로 frame 검사를 하지 않는다"는 뜻이다 (spec
5.4). `Camera(id)`는 OpenCV 광학 프레임이고 `Image(id)`는 픽셀 좌표이며, 둘 다 spec
3.1을 따른다.

## TimeRef와 join 규칙 (`TYPE-014`)

```
Tick                                   the current physics tick
Sensor { id, align }                   align = Hold | Interpolate | Reject
Window { base, n, stride }             a TemporalWindow
```

**엣지**는 `TimeRef`가 같을 것을 요구한다. 여러 입력의 **결합**(`Concat`, `Arith`,
…)은 `TimeRef::join`을 호출하며, `TYPE-014`는 여기서 나온다.

| a | b | 결과 |
|---|---|---|
| equal | equal | 그 `TimeRef` |
| `Tick` | `Sensor { align: Hold \| Interpolate }` | `Tick` — 센서가 tick에 맞춰 리샘플링된다 |
| `Sensor { x, align_x }` | `Sensor { y, align_y }`, x ≠ y | `align_x == align_y`이고 `Reject`가 아니면 `Tick`, 아니면 `TYPE-014` |
| `Window { b, n, s }` | 동일한 `Window` | 그 window |
| 그 밖의 모든 경우 | | `TYPE-014` |

`Align::Reject`는 "이 클록을 건너 결합하지 않는다"는 뜻이다 — join이 여기에 도달하면
조용한 hold가 아니라 오류다. 서로 다른 두 `Window`를 join하는 것도 오류다: 학습 입력
의미(spec 7.4)가 다르기 때문에, 사용자를 위해 하나를 골라주면 네트워크가 보게 되는
것이 바뀌어 버린다.

## 정규 인코딩

`PortType::canonical`은 타입 전체를 `CanonWriter`(`hash.rs`)에 기록하여, 타입 해시를
담는 노드 파라미터가 여러 머신에서 동일하게 나오도록 한다. `Normalized`의 두 `f64`는
writer의 float 규칙(−0.0 → 0.0, NaN 거부)을 거치며, 이 모듈에서 해시에 들어가는
float는 이것뿐이다.
