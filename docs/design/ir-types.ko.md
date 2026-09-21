<!-- Korean translation of docs/design/ir-types.md. The English file is the working copy; regenerate this when it changes. -->

# IR 공통 타입 시스템 (`es-ir::types`) — 설계

Spec refs: spec 5.4 (공통 타입 시스템), spec 5.2 (batch semantics), spec 3.1 (conventions),
부록 B.1. Packet: `docs/packets/M0/P18.md`.

## `PortType`

```
PortType = (ElemType, Shape, Unit, Frame, TimeRef, Option<ImageSpec>)
```

`Shape`는 **샘플당 차원만** 싣는다. 배치 축은 절대 shape 안에 없다: 네 배치 도메인
(시뮬레이션 / 관측 / 추론 / 학습, spec 5.2) 각각이 자신만의 크기를 고르므로, 그중
하나를 고정한 shape는 나머지 셋에서는 틀리게 된다.

엣지로 연결된 두 포트는 `elem`, `shape`, `unit`, `time`에서 정확히 일치해야 하고,
`frame`은 아래의 `Policy` 예외를 제외하고 일치해야 한다. `image`는 둘 다 없거나
같아야 한다.

## Unit — `m^a · kg^b · s^c · rad^d` 형태의 명명된 단위

곱셈과 나눗셈은 네 지수(exponent) 기저 위에서 정의된다. 지수 벡터를 갖는 명명된
단위는 그것으로, 그리고 그로부터 변환된다:

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

이 표는 양방향이며 매핑은 단사(injective)이므로, `Length / Time = Velocity`와
`Velocity * Time = Length`는 자기 자신의 이름으로 다시 돌아온다. 이름이 없는 지수
벡터는 `Composite(UnitPowers)`가 된다; 지수가 모두 0인 `Composite`는 다시
`Dimensionless`로 정규화된다.

### 불투명(opaque) 단위

`Current`, `Voltage`, `Quaternion`, `RotationMatrix`, `Normalized { lo, hi }`,
`Pixel`, `Luminance`, `Depth`, `Token`은 지수 벡터를 **갖지 않으며** `mul` / `div`에서
거부된다(`TYPE-010`). 이유:

- `Current` / `Voltage`는 스펙의 네 지수 기저에는 없는 암페어 축이 필요하다.
- `Quaternion` / `RotationMatrix`는 애초에 기본 단위의 곱이 아니다 — 두 회전을
  곱하는 것은 차원 연산이 아니라 군(group) 연산이다.
- `Normalized { lo, hi }`는 이미 임의의 구간으로 매핑되었다; 그것이 유래한 물리적
  차원은 구성상 사라진 상태다.
- `Pixel`, `Luminance`, `Token`은 도메인 특화된 알파벳 안의 개수(count)다.
- `Depth`는 미터 단위이지만, 이를 불투명하게 유지하는 것은 의도적이다: 이는 depth
  이미지가 단위 대수(unit algebra)를 통해 조용히 `Length`가(그리고 그다음
  `Velocity`가) 되는 것을 막는다. 변환은 추론이 아니라 명시적인 노드다.

`checked_add`는 두 `Unit` 값이 지수만 같은 것이 아니라 **enum 값으로서 동일**할
것을 요구한다. `Depth + Length`는 거부되며, `Normalized{0,1} + Normalized{-1,1}`도
마찬가지다.

### 정책 입력 규칙

정책 입력 포트는 `Normalized`, `Dimensionless`, `Token`만 받아들인다
(`PortType::is_policy_input`, 진단 `TYPE-011`). 원시 미터나 라디안을 네트워크에
그대로 먹이는 것은 이 규칙이 컴파일 타임에 잡아내는 흔한 버그다(spec 5.4).

## Frame

`World | LocalOrigin | Body(id) | Sensor(id) | Joint(id) | Camera(id) | Image(id) |
Policy`, id는 `es_core::StableId`다. Frame은 같아야 한다 — 다만 `Policy`는 모든
것과 호환된다: "네트워크 내부이며 frame 검사 없음"을 뜻한다(spec 5.4). `Camera(id)`는
OpenCV 광학 frame이고 `Image(id)`는 픽셀 좌표이며, 둘 다 spec 3.1을 따른다.

## TimeRef와 join 규칙 (`TYPE-014`)

```
Tick                                   현재 물리 틱
Sensor { id, align }                   align = Hold | Interpolate | Reject
Window { base, n, stride }             TemporalWindow
```

**엣지**는 같은 `TimeRef`를 요구한다. 여러 입력의 **조합**(`Concat`, `Arith`, …)은
`TimeRef::join`을 호출하며, `TYPE-014`가 나오는 곳이 바로 여기다:

| a | b | 결과 |
|---|---|---|
| 같음 | 같음 | 그 `TimeRef` |
| `Tick` | `Sensor { align: Hold \| Interpolate }` | `Tick` — 센서가 틱 위로 리샘플된다 |
| `Sensor { x, align_x }` | `Sensor { y, align_y }`, x ≠ y | `align_x == align_y`이고 `Reject`가 아니면 `Tick`, 아니면 `TYPE-014` |
| `Window { b, n, s }` | 동일한 `Window` | 그 window |
| 그 외 | | `TYPE-014` |

`Align::Reject`는 "이 시계(clock)를 가로질러 결합하지 말라"는 뜻이다 — 이에 도달하는
join은 조용한 hold가 아니라 오류다. 서로 다른 두 `Window`를 결합하는 것도
오류다: learning-input 의미(spec 7.4)가 다르며, 사용자를 위해 하나를 고르는 것은
네트워크가 보는 것을 바꿔버릴 것이다.

## 정규 인코딩

`PortType::canonical`은 전체 타입을 `CanonWriter`(`hash.rs`)에 써서, 타입 해시를
싣는 노드 파라미터가 모든 머신에서 동일하도록 한다. `Normalized`의 두 `f64`는
writer의 float 규칙(−0.0 → 0.0, NaN은 거부됨)을 거친다; 이 모듈로부터 해시에
도달하는 다른 float은 없다.

## Task IR의 센서 소스 (`ObsSource::Sensor`, 패킷 M7/R5)

```
Sensor { id, format, render: SensorRender }
SensorRender { path: Rs | Pt { spp, bounces }, exposure: f32, tonemap: Reinhard | Aces }
```

`render`는 *시뮬레이션*이 그 채널을 어떻게 만드는지 말하고, `ImageSpec`은 그 채널이 *무엇*인지 말한다. 둘을 갈라 두는 것이 `observation.toml`이 노드 하나 바꾸지 않고 래스터화 태스크와 패스 트레이싱 태스크를 모두 섬길 수 있게 하고, `render`가 `ImageSpec`의 필드가 아닌 이유이기도 하다 — `ImageSpec`은 실제 로봇도 가질 수 있는 카메라를 기술하는데, "픽셀당 64 샘플"은 실제 카메라가 가진 것이 아니다.

이 필드를 쓰지 않는 모든 문서에 대해 그것을 공짜로 만드는 규칙이 둘 있다:

- **`#[serde(default, skip_serializing_if = "SensorRender::is_default")]`**. `render`를 한 번도 언급하지 않는 문서는 같은 바이트로 왕복한다.
- **`ObsSource::canonical`은 기본값이 아닐 때만 블록을 쓴다.** 그래서 `task_hash`는 타입이 말할 수 있는 것이 아니라 문서가 *말한* 것의 함수다. 부재하는 `render`와 명시적으로 써 넣은 기본값 `render`는 같은 문서이고 같은 해시다; `crates/es-ir/tests/sensor_render.rs`가 커밋된 `task_hash eb6efefa…`에 대해 둘 다 단언한다.

기본값이 오늘의 동작인 어떤 후속 필드에도 같은 수법이 통하고, 이것이 §28.10 규칙 1의 IR 쪽 해석이다. 대가는 정규 인코딩이 더 이상 구조체를 곧이곧대로 순회하는 것이 아니라는 점이다 — `SensorRender` *안에* 필드를 추가하면서 `SensorRender::canonical`을 늘리지 않으면 그 필드는 해시에 보이지 않는다. 그 파일의 두 번째 테스트가 노브를 하나씩 바꿔 가며 해시가 움직이는지 단언하는 이유가 그것이다.

## 어휘가 사는 곳 (`es-ir-types`, layer 2)

`es-ir`은 다섯 개의 IR과 그래프 뼈대, 그래프 정규화기를 담는다; 그 아래에서 그래프가
무엇인지 **모르는** 모든 것은 한 크레이트 아래인 `es-ir-types`에 살며, 그래서 두 크레이트
모두 한 컨텍스트 윈도우에 들어간다(spec 1.5). `es-ir`은 모든 항목을 원래 경로로
재수출하므로 `es_ir::types::PortType`, `es_ir::task::Expr`, `es_ir::HashChain`이 그대로
해석되고 하위 크레이트는 하나도 바뀌지 않는다.

| `es-ir-types` 모듈 | 담는 것 | 패킷 |
|---|---|---|
| `types`, `image`, `diag`, `codes`, `canon` | 타입 시스템, `ImageSpec`, 진단, `CanonWriter` | `P-M0-R4` |
| `expr` | spec 6.3 파라미터 enum들(`ArithOp`, `MathFunc`, `Distribution`, …)과 spec 6.5 `Expr` + 평가기 | `P-M4-S16` |
| `chain` | `HashChain`, `DatasetHash`, `HardwareCapability`, `ChangedComponent` — 다이제스트뿐, 그래프 해싱 없음 | `P-M4-S16` |

그래프 형태인 나머지 절반은 의도적으로 `es-ir`에 남는다: `canonical_hash` /
`canonical_order`는 `Graph<N>`이 필요하고, IR-C의 `ControlGraph`는 `TaskIr`에 대해
검증하므로, 둘 다 그래프를 함께 끌고 내려가지 않고서는 layer를 내릴 수 없다.

## 스키마 버전

각 IR은 자신의 `schema_version`을 싣고, 그것은 **해시 입력**이다 — 상수가 올라간 뒤에도
예전 파일이 자기 해시를 유지하는 이유가 이것이다. `TaskIr::validate`는
`1 ..= SCHEMA_VERSION`을 받아들이고 그 범위를 벗어나면 `TASK-002`를 보고한다.
`DeploymentIr`이 `DEP-001`을 보고하는 것과 같다: `0`은 기록되지 않은 필드이고, 이 빌드보다
높은 버전은 이 빌드가 볼 수 없는 노드와 필드를 싣고 있다는 뜻이므로, 그에 대해 하위
어디에서도 `task_hash`를 고정해서는 안 된다. 마이그레이션 단계는 없다 — 지원되는 예전
버전은 있는 그대로 읽힌다.

## 스키마 마이그레이션

`docs/packets/M0/P29.md`는 내장 노드 kind 목록을 동결하며(spec 28.7 게이트 10),
`factory.rs`는 각 목록의 blake3 다이제스트를 단언(assert)하므로, 이름 변경이나
추가나 제거는 `task_hash` / `learning_hash`를 조용히 움직이는 대신 CI를 실패시킨다.
그러한 모든 변경은 여기에 기록된다.

| 버전 | 변경 |
|---|---|
| Task IR `1` | 초기 IR-D 스키마 (P20) |
| Task IR `2` | IR-C: `TaskIr.control: Option<ControlGraph>`, `task_hash`에 섞여 들어감. `None`이 기본값이며 IR-C 이전의 모든 파일이 그대로 파싱되지만, 버전 자체가 해시 입력이므로 모든 `task_hash`가 움직인다. 새로 동결된 목록 `BUILTIN_CONTROL_KINDS` = `Sequence`, `Branch`, `Repeat`, `SubTask`, 다이제스트 `2438218d1bb498aae98de89569fd62bbe1820ea1c860da0471a5e98235a21158`. `BUILTIN_TASK_KINDS`와 그 다이제스트는 변경되지 않았다: control 노드는 `TaskNode`가 아니므로 `TaskNodeFactory`는 이 kind들을 주장하지 않는다. `docs/design/control-graph.md` 참고. |
