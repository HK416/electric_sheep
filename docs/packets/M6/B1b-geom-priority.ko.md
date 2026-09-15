# M6 B1b — geom `priority`와 `<position inheritrange>`를 씬 모델로 나르기

설계 노트: `docs/design/quadruped-track.md` 3.1절. Go1 씬을 실었고 이 둘을 알려진 누락으로
지목한 M6/B1의 후속. B1 자신의 오라클이 측정했던 물리 일치 격차를 닫는다:
`go1_stands_from_home_and_agrees_with_mujoco_directly`가 임계값 1e-3에 대해 좌표별 최대
**|Δqpos| = 7.574e-3**으로 실패하고 있었다.

## context

```
crates/es-assets/src/scene.rs
crates/es-assets/src/mjcf/mod.rs
crates/es-assets/src/mjcf/attrs.rs
crates/es-assets/src/mjcf/elements.rs
crates/es-assets/src/gltf.rs
crates/es-assets/src/urdf.rs
crates/es-assets/tests/go1_provenance.rs
crates/es-physics-backend/src/mjcf_out.rs
crates/es-physics-core/src/usd.rs
crates/es-render/src/cornell.rs
crates/es-render/src/scene.rs
crates/es-splat/tests/splat.rs
tests/fixtures/quadruped/{task,observation,learning,deployment}.toml
tests/fixtures/visible-learning/{task,observation,evaluation}.toml
docs/design/quadruped-track.md
docs/design/quadruped-track.ko.md
docs/packets/M6/B1b-geom-priority.md
docs/packets/M6/B1b-geom-priority.ko.md
```

`gltf.rs` / `urdf.rs` / `usd.rs` / `cornell.rs` / `scene.rs` / `splat.rs` 수정은 각각 한 줄
— MuJoCo 기본값인 `priority: 0` — 이다. `Geom`이 `Default` 없는 평범한 구조체이기 때문이다.
`Cargo.toml` 변경 없음, 새 트레이트 없음(INV-17), `docs/` 밖의 새 파일 없음.

## 이 패킷이 존재하는 이유인 측정

**MuJoCo 단독** 절제 실험이므로 숫자는 우리 코드에 관한 것이 아니다:
`tests/fixtures/mjcf/go1_primitives.xml`에서 속성을 하나씩 제거하고, `home`에서 행동 0으로
물리 1,250스텝을 밟은 뒤, 원본 파일에 대해 `qpos`를 비교한다.

| 제거한 것 | 좌표별 최대 \|Δqpos\| |
| --- | --- |
| 바닥 `priority="1"` | **7.574e-3** |
| 발 `solimp` | 0 |
| `<position inheritrange>` | 0 |

7.574e-3은 B1의 오라클이 우리 백엔드와 MuJoCo 직접 실행 사이에서 측정한 불일치와 마지막
자리까지 같다 — 즉 `priority`가 격차의 전부였다. MuJoCo의 접촉 파라미터 규칙: `priority`가
더 높은 geom이 `friction`, `condim`, `solref`, `solimp`를 단독으로 결정하고, 같으면
원소별 `max`로 섞는다. 바닥의 `priority="1"`은 "모든 발 접촉은 내 0.6을 쓴다"는 상류의
선언이다. 이것이 빠지면 발의 0.4가 `max`를 이기고 자세가 다른 곳에 안착한다.

## spec

- §5.3: `scene_hash`는 "시뮬레이션 거동을 바꿀 수 있는 모든 필드에 민감"하므로, `priority`는
  0이 아닐 때만이 아니라 `condim` 옆에서 **조건 없이** 인코딩된다. MuJoCo가 다르게 밟는 두
  씬에 대해 같은 값이 나오는 해시가 바로 §5.3이 금지하는 버그다.
- §17.2: 백엔드는 매핑할 수 없는 것을 추측하지 않고 선언한다. `mjcf_out`은 이제 `priority`를
  내보낸다(기본값 0에서는 생략하므로 생성된 XML은 상류의 모양을 유지한다).
- §1.4: 심판은 리뷰가 아니라 `go1_step.rs`를 통한 MuJoCo(CPU)다.
- §4.3, §3.5: 허용오차는 측정된 숫자가 무엇이든 백엔드가 선언한
  `DeterminismTier::PhysicsMeaning` 허용오차로 남는다. 외부 백엔드는 tier 1을 선언하지 않는다.
- §28.9: 이 트랙이 속한 마일스톤.

`SCENE_TAG`은 `es.scene.v1` 그대로다. 그것은 스키마 버전이 아니라 해시 네임스페이스 사이의
도메인 분리자다. M6/B1이 같은 인코딩에 `ls_iterations`와 `eulerdamp`를 올림 없이 추가했고 이
패킷도 그것을 따른다. 어차피 모든 `scene_hash`가 움직이며, 아래 **해시** 절이 그 이야기다.

## `inheritrange`를 어떻게 날랐나

날랐고, 열 줄이 들었다. `<position inheritrange="f">`는 MuJoCo에서 **컴파일 시점 재작성**이다:
`ctrlrange`가 전달 대상 자신의 범위를 중점 기준으로 `f`배 스케일한 값이 된다. 임포터는
`Actuator`를 만들기 전에 이미 대상 관절을 해석하고, 관절은 `<actuator>`보다 먼저 파싱되므로
(`ROOT_ORDER`) `Actuator::ctrl_range`가 해석된 쌍을 담으면 되고 씬 모델에 필드가 전혀 늘지
않는다. `<position>`에만 적용한다 — 상류는 `intvelocity`에도 허용하지만 우리는 그것을
모델링하지 않고, 거기서는 열거된 경고로 남는다.

절제 실험이 여기서 0을 매김에도 이것을 버리는 것은 중립이 아니었다: `ctrlrange`가 없으면
액추에이터가 클램프되지 않으므로, 정책의 위치 목표가 학습 시보다 넓은 범위로 유지된다.
`home`에서의 행동 0은 한계에 닿지 않고, 그래서 스텝 오라클이 이것을 볼 수 없으며, 그래서
측정이 아니라 논증으로 고친다.

## 해시

정준 인코딩에 필드를 추가하면 모든 `scene_hash`가, 따라서 모든 `task_hash`와 그 하류 전부가
움직인다. **오직** 각자의 인가된 생성기로만 재생성했다 — 이 저장소의 어떤 해시도 손으로
입력되거나 편집되지 않는다:

```
cargo test -p es --test cli -- --ignored regenerate_quadruped_documents
cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents
```

이것이 `tests/fixtures/quadruped/{task,observation,learning,deployment}.toml`과
`tests/fixtures/visible-learning/{task,observation,evaluation}.toml`을 다시 썼다(마지막 것은
`task = "<task_hash>"`를 고정한다). `tests/golden/` 아래 어떤 파일도 씬 해시를 담지 않으므로
골든은 건드리지 않았고 `cargo xtask verify-goldens`는 그대로 통과한다.

## oracle

```
cargo xtask ci
cargo test -p es-assets --test go1_provenance
cargo test -p es-physics-backend --lib mjcf_out
```

GPU 서버에서(`mujoco` / `torch`가 필요한 절반):

```
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es-physics-backend --test go1_step -- --ignored --nocapture
ES_PYTHON=$HOME/venvs/es/bin/python cargo test -p es --test cli --features render -- \
  quadruped dataset_bake expert_passes_the_evaluation_harness collection_and_evaluation
```

## acceptance

- `go1_stands_from_home_and_agrees_with_mujoco_directly`가
  `RAN go1_stands_from_home_and_agrees_with_mujoco_directly`를 최대 |Δqpos| 1e-3 미만으로
  출력한다. **측정값: 1,250 물리 스텝에서 8.882e-16** — 허용오차 안이라기보다 배정밀도의
  마지막 비트. 이 패킷 전: 7.574e-3.
- `derivative_carries_the_home_keyframe`이 바닥의 `priority="1"`을 상류 자신의 바이트에 대해,
  순서를 가정한 하나의 부분 문자열이 아니라 속성 단위로 비교하고, 그 값이 `SceneDesc`에
  도달했음을 단언한다.
- `derivative_parses_with_only_the_enumerated_warnings`가 더는 `` `priority` ``나
  `` `inheritrange` ``를 열거하지 않는다: 실려 나가므로 경고하면 안 된다.
- `actuators_and_sensors_round_trip`이 `priority="2"`와, 관절 자신의 `(-1, 1)`에서
  `(-0.5, 0.5)`로 해석된 `inheritrange="0.5"` 위치 액추에이터를 왕복시킨다.
- 재생성된 데모 픽스처는 위의 `es` CLI 테스트로 서버에서 MuJoCo에 대해 증명된다.

## forbidden

- 픽스처든 골든이든 해시를 손으로 편집하는 것. 생성기만 쓴다.
- `docs/ARCHITECTURE*.md`, `docs/design/visible-learning.md` — 이 패킷이 바꿀 문서가 아니다.
- `crates/es-safety` — 손대지 않는다. `priority`는 아무것도 넓히지 않고 아무것도 끄지 않는다
  (INV-12, INV-13).
- 임포터가 여전히 버리는 다른 모든 MJCF 속성. `group`, `<keyframe>`, 재질 `rgba`, 카메라
  `mode`는 열거된 경고로 남는다. 이 패킷은 오라클이 지목한 둘을 나른다.
