<!-- Korean translation of docs/packets/M1/W2-sensor-actuator.md. The English file is the working copy; regenerate this when it changes. -->

# W2 — 센서 리얼리즘 기초, 채널 계약(channel contract), 액추에이터 모델 (CPU)

Spec: spec 18.2 (액추에이터), spec 18.3 (센서와 리얼리즘: noise, latency, dropout, rolling
shutter, exposure), spec 15.1 (렌더 -> observation 경로, channel contract), spec 7.2 (카메라
센서가 채우는 `ImageSpec` 필드: shutter, exposure, rate), spec 3.4 (전역 RNG 금지, 결정적
실행), spec 18.4 (결정적 리덕션(reduction) 컨텍스트).

## context (범위)

```
crates/es-sensor/**
crates/es-actuator/**
docs/packets/M1/W2-sensor-actuator.md
```

## spec (사양)

- `es-sensor::channel` — 렌더 -> observation 채널 계약(spec 15.1): `dtype()`, `n_components()`,
  `unit()`(고정된 문자열 어휘 — `es-sensor`는 계층 3이며 `es-ir`의 계층 6보다 아래에 있어
  `es_ir::Unit`을 사용할 수 없다)을 갖는 `Channel`(`Rgb8 | RgbF32Linear | Depth32{unit_m} |
  Normal | SegmentationId | Flow | PtRadiance`); spec 7.2의 `ImageSpec` 타이밍 필드를 실어
  나르는 `CameraContract{channels, width, height, rate, shutter, exposure_ticks}`. `Channel`은
  `Ord`/`Eq`를 직접 구현하는데(그 `Depth32` payload가 `f64`이기 때문), 이는 값이 아니라
  IEEE-754 비트 패턴으로 정렬하여 `BTreeSet`에 담을 수 있도록 하기 위함이다.
- `es-sensor::noise` — `f64` 슬라이스 위에서 동작하는 결정적 리얼리즘 스테이지들이며, 연산
  순서는 고정되어 있다: `Gaussian{sigma}`, `Bias{offset, drift_per_tick}`(상태 없음: 절대 틱의
  순수 함수), `Quantize{step}`, `Dropout{p_drop, hold_last}`, `Latency{ticks}`(미리 할당된
  ring, zero pre-fill), 그리고 `RollingShutterSkew`(플랫 버퍼가 아니라 이미지 높이가 필요하므로
  `Stage`가 아니라 독립적인 행(row)별 틱 오프셋 표). `SensorModel{stages}`는 이들을 선언된
  순서대로 실행한다. `NoiseRng`는 `crates/es-env/src/rng.rs::EnvRng`에서 복사한 약 30줄짜리
  splitmix64 코어다(주석에 출처를 명시함). `es-sensor`(계층 3)는 `es-env`(계층 9)에 의존할 수
  없기 때문이다; `sensor_seed(seed, StableId)`는 센서의 identity를 base seed에 접어 넣어,
  동일한 seed 아래에서도 서로 다른 센서가 독립적인 스트림을 뽑도록 한다. 유일한
  초월함수(Box-Muller의 `sqrt`/`ln`/`cos`)는 `es_math::approx`를 거친다(spec 3.2 `DET-010`).
- `es-sensor::sensor` — `SensorDesc{id, kind, rate, model}`과, `StableId`를 키로 하는
  `BTreeMap`인 `SensorBank`. 이로써 스레드 없이도 순회 순서가 결정적이다(정적
  분할(static partitioning), spec 3.4: `HashMap` 순회 의존성 금지).
- `es-actuator::model` — `crates/es-assets::scene::ActuatorKind`의 필드 이름을 그대로 본뜬
  (import가 아니라 미러링한, 이 계층-3 crate를 계층-2 의존성에서 자유롭게 두기 위해)
  `ActuatorModel::{Motor{gear}, Position{kp,kd,gear}, Velocity{kv,gear},
  General{gain,bias,gear}}`; `force(ctrl, q, qd)`는 MuJoCo의 motor/position/velocity 법칙을
  직접 구현하며, `es-assets`의 문서 주석에서 추론한 `gaintype=affine, biastype=affine`
  general-actuator 법칙도 구현한다(업스트림 MuJoCo 소스 대비 미검증 (unverified)으로 표시됨).
  `Saturation`은 `ctrl_range`/`force_range`를 clamp한다; `ActuatorDelay`는 고정 틱 전송
  지연이다(미리 할당된 ring, zero pre-fill, `noise::Stage::Latency`를 그대로 따름);
  `Backlash{deadband}`는 최소한의 데드존 히스테리시스 모델이다(spec 18.2는 이후의 리얼리즘
  단계로 two-mass 모델을 추가로 열거하지만 — 언급만 하고 여기서는 만들지 않는다).

## oracle (오라클)

```
cargo fmt -p es-sensor -p es-actuator --check
cargo clippy -p es-sensor -p es-actuator --all-targets -- -D warnings
cargo test -p es-sensor -p es-actuator
cargo xtask layering
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- 각 noise 스테이지는 손으로 계산한 벡터와 일치한다; `Gaussian`은 동일한 `(seed, sensor id,
  env, tick)`에 대해 비트 단위로 동일하며 sensor id가 다르면 값이 달라진다; 10,000회 Gaussian
  추출의 평균/분산은 `(0, 1)`의 허용오차 내에 있다.
- `Latency`는 zero pre-fill 상태로 출력을 정확히 `ticks`틱만큼 지연시킨다;
  `Dropout{hold_last: true}`는 드롭하는 동안 이전 출력을 반복한다.
- `RollingShutterSkew::row_offset_ticks`는 행(row)에 대해 단조 비감소(monotonic
  non-decreasing)이며 `readout_ticks`로 상한이 정해진다(맨 위 행에서 0, 맨 아래 행에서
  `readout_ticks`).
- `CameraContract`는 `serde_json`을 통해 손실 없이 왕복(round-trip)하며, 빈 채널 집합이나 0인
  width/height는 거부한다.
- `proptest`: 전체 noise 파이프라인(`Gaussian` + `Bias` + `Quantize` + `Dropout` + `Latency`)은
  유효 범위 내 임의의 파라미터 조합에서 유한한(finite) 입력에 대해 유한한 출력을 낸다.
- `ActuatorModel::force`는 네 variant 모두에서 손으로 계산한 케이스와 일치한다; `Saturation`은
  clamp를 수행하고 역전된(inverted) range는 거부한다; `Backlash`는 음수 deadband를 거부하며
  추적(tracking)하기 전까지는 dead zone 내에서 값을 유지한다; `ActuatorDelay`는 정확히
  `ticks`틱만큼 지연시킨다.
- `proptest`: `force`, `ActuatorDelay::step`, `Backlash::step`은 유한하고
  유계(bounded-range)인 입력에 대해 유한한 값을 낸다; 동일한 호출을 반복하면 비트 단위로
  동일한 출력을 낸다.
- 게이트 통과: `cargo fmt -p es-sensor -p es-actuator --check`, `cargo clippy -p es-sensor -p
  es-actuator --all-targets -- -D warnings`, `cargo test -p es-sensor -p es-actuator`,
  `cargo xtask layering`, `cargo xtask check-spec-refs`.

## forbidden (금지)

- `context` 밖의 모든 파일. 특히 루트 `Cargo.toml`, `crates/es`, `crates/es-policy`,
  `crates/es-physics-backend`(다른 에이전트들이 진행 중인 작업), 그리고 리뷰어가 읽고 있을 수
  있는 모든 M0 crate.
- 두 crate 중 어느 쪽에서도 `es-env`(계층 9)나 `es-ir`(계층 6)에 대한 의존성을 갖지 않는다 —
  둘 다 계층 3이다; 대신 복사/미러링한다(`noise::NoiseRng`, `model::ActuatorModel`의 필드
  이름).
- Vulkan/렌더링 코드나 타일 아틀라스(tile atlas) — 이는 이후 패킷의 몫이다; 이 패킷은 CPU
  수학 연산만 다룬다.
- 결정적 경로에서 순회되는 어떤 것에도 `HashMap`/`HashSet`을 쓰지 않는다; noise나 force
  커널에서 `std` 초월함수(`f64::sin`/`cos`/`exp`/`ln`/`sqrt`)를 쓰지 않는다 —
  `es_math::approx`를 사용한다.
- 새로운 trait — `INV-17`의 일곱 확장 지점이 전부이며, 어느 crate도 새로 추가하지 않는다.
- Safety Plane을 비활성화하거나 넓히는 것, 또는 `crates/es-safety` 아래의 무엇이든 건드리는
  것 — 범위 밖이며 여기서는 필요하지 않다.
