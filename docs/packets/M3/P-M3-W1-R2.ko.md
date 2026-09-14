<!-- Korean translation of docs/packets/M3/P-M3-W1-R2.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-W1-R2 — 부분적인 `JointState`는 Safety Plane에 도달하지 않는다

Spec: §9.4(`SafetyPlane::observe_state` / `sensor_seen`의 입력), §25.1(네트워크에서 온 메시지는
신뢰할 수 없고, 기본값은 측정값이 아니다), §26.1("검증되지 않은 것은 실행되지 않는다"), INV-12.
디자인 노트 `docs/design/ros2-boundary.md` section 4.5.
`docs/reviews/M3-W1.md`의 **S-1**을 닫는다.

## context (범위)

```
crates/es-ros2/src/actuator.rs
crates/es-ros2/src/error.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R2.md
```

## spec (사양)

`reorder_joint_state`(`actuator.rs:65-81`)는 설정된 각 관절을 이름으로 찾은 뒤 이렇게 읽는다.

```rust
q[i]  = state.position.get(idx).copied().unwrap_or(0.0);
qd[i] = state.velocity.get(idx).copied().unwrap_or(0.0);
```

`sensor_msgs/JointState`는 `position`, `velocity`, `effort`가 "비어 있을 수 있고" 길이가 `name`과
같아야 한다는 것은 "should"일 뿐이라고 문서화한다. 따라서 이것은 악의적인 경우가 아니다 — 이름과
위치는 publish하지만 속도는 publish하지 않는 드라이버는 흔하며, 지금은 `qd = [0.0; NJ]`라는 조작된
0 속도를 만들어 `SafetyPlane::observe_state`에 넘긴다. 짧은 `position`도 같은 식으로 관절 각도를
지어낸다. 둘 다 plane의 hold-position 목표, 속도 추정, 그리고 그것들에 대해 미분하는 모든
rate-limit 단계에 들어간다.

이 모듈 자신의 doc 주석(`:61-64`)과 디자인 노트 4.5는 의도한 규칙을 이미 적고 있다: "샘플이 싣지
않은 관절이 있으면 조용히 0으로 두는 대신 샘플 전체를 거부(`ROS2-013`)하므로, 부분적인 상태는 절대
`SafetyPlane::observe_state` / `sensor_seen`에 도달하지 않는다." 코드를 거기에 맞춘다.

- 매칭된 인덱스보다 짧은 `position`은 `Ros2Error::MissingJoint`와 모양은 비슷하지만 다른 조건이다.
  variant를 하나 추가한다: `Ros2Error::PartialJointState { joint: String, field: &'static str }`,
  `error.rs:129-131`의 기존 `ROS2-013`에 매핑한다(이 코드는 "actuator lookup" 포괄 코드이며, 새
  번호는 만들지 않는다).
- `velocity`는 "비어 있음"에 변호 가능한 해석이 있는 유일한 필드다. **명시적으로** 고르고 디자인
  노트 4.5에 기록한다: 완전히 빈 `velocity`(`state.velocity.is_empty()`)는 "이 드라이버는 속도를
  보고하지 않는다"이고 `qd = [0.0; NJ]`를 낳으며 그 결정은 기록된다. 비어 있지는 않지만 짧은
  `velocity`는 `PartialJointState`다. `position`에는 그런 해석이 없다: 짧거나 비어 있으면 항상
  에러다.

`reorder_joint_state`의 시그니처는 바뀌지 않는다. 계속
`Result<([f64; NJ], [f64; NJ]), Ros2Error>`를 반환한다.

## oracle (검증)

```
cargo test -p es-ros2 --features zenoh actuator
cargo fmt --check
cargo clippy -p es-ros2 --all-targets --features zenoh -- -D warnings
cargo xtask check-spec-refs
```

`actuator.rs`의 기존 `mod tests`에 넣는 단위 테스트:

- `a_short_position_array_is_refused` — `name = ["j1","j2"]`, `position = [1.0]`,
  `velocity = [10.0, 20.0]`: `ROS2-013`이고 에러가 `j2`와 `position`을 지목한다. 수정 전
  **FAIL**(`q = [1.0, 0.0]`을 반환한다).
- `a_short_velocity_array_is_refused` — `position`은 온전한데 `velocity = [10.0]`: `ROS2-013`.
- `an_empty_velocity_array_is_the_documented_zero` — `velocity = []`: `Ok`, `qd == [0.0; NJ]`,
  `q`는 재정렬된 위치 그대로(기존 테스트처럼 bit-exact 비교).
- `an_empty_position_array_is_refused` — `position = []`: `ROS2-013`.
- 기존 `reorders_by_name_and_rejects_a_missing_joint`는 변경 없이 통과한다.

## acceptance (수용 기준)

- 다섯 테스트가 통과하고, 첫 번째가 수정 전 함수에서 실패함이 확인된다.
- `Ros2Error::code()`는 여전히 디자인 노트 4.2가 나열한 코드들과 `ROS2-000`, `ROS2-013`,
  `ROS2-020`만 반환한다; 새 코드 문자열 없음.
- 디자인 노트 4.5에 빈 `velocity` 해석을 명시하는 문장 하나가 추가되고, `.ko.md` 형제 문서도 같은
  커밋에서 갱신된다.
- `session_loopback` 테스트 13개가 계속 통과하고 `actuator_publisher_sends_the_safe_action`은
  그대로다.
- 수락 경로에 할당 추가 없음, `HashMap` 없음, 새 trait 없음.

## forbidden (금지)

- `crates/es-safety`, `crates/es-ir`, 그 밖의 모든 크레이트.
- `camera`, `hil`, `session`, `config`, `cdr`, `msg`(`msg::JointState`의 필드 타입은 올바르며
  그대로 둔다).
- "부분 상태를 허용" 설정 스위치 추가: §26.1이 그런 것을 인정하지 않는다.
- `ROS2-013`의 의미를 actuator 외 조건으로 넓히거나 코드 번호를 바꾸는 것.
- 그 밖의 모든 M3 W1 발견 사항.
