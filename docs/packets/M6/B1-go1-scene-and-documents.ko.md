# M6 B1 — Go1 장면과 네 개의 IR 문서

설계 노트: `docs/design/quadruped-track.md` 섹션 1–3. 고정된 업스트림 사실:
`docs/api-notes/mujoco-playground-quadruped.md`. 사족보행 트랙의 첫 패킷이자 LOCAL-ONLY:
학습 없음, GPU 없음, 서버 없음. 임포트 패킷(brax 체크포인트 → `safetensors` → Learning IR
가중치)을 막는다.

## context

```
tests/fixtures/mjcf/go1_primitives.xml
tests/fixtures/mjcf/go1_primitives.LICENSE
tests/fixtures/mjcf/go1_primitives.PROVENANCE.json
tests/fixtures/quadruped/task.toml
tests/fixtures/quadruped/observation.toml
tests/fixtures/quadruped/learning.toml
tests/fixtures/quadruped/deployment.toml
crates/es-assets/tests/go1_provenance.rs
crates/es-assets/src/scene.rs
crates/es-assets/src/mjcf/mod.rs
crates/es-physics-backend/tests/go1_step.rs
crates/es-physics-backend/src/mjcf_out.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
docs/design/quadruped-track.md
docs/design/quadruped-track.ko.md
docs/packets/M6/B1-go1-scene-and-documents.md
docs/packets/M6/B1-go1-scene-and-documents.ko.md
```

메모: 네 개의 `src/` 파일이 각각 몇 줄씩 바뀌며 이유는 하나다 — Playground로 학습된 모델의
`<option>` 블록이 `SceneDesc`를 살아서 통과해야 한다(아래 **spec**). `crates/es/src/cmd/eval.rs`
는 그 자신의 주석이 권하는 디스패치 표에 `(12, 1)` 한 행을 더한다. `Cargo.toml` 변경 없음:
프로비넌스 테스트는 `so101_provenance.rs`가 그러듯 새 HTTP 의존성이 아니라
`std::process::Command`로 시스템 `curl`을 쓴다.

## spec

- §1.4: 장면은 리뷰가 아니라 고정된 업스트림 파일에 대한 실행 가능한 교차 검사로 판정된다. 물리
  오라클은 MuJoCo(CPU)이고, `mujoco`를 가진 인터프리터가 없으면 통과가 아니라 `SKIP <why>`를
  출력한다.
- §1.9, §8: 정책은 `mujoco_playground`가 **외부에서** 학습하고 가중치로 가져온다. 네이티브 학습
  스택은 애초에 범위가 아니었고, Learning IR이 임포트 표면이다.
- §17.2: 백엔드는 매핑할 수 없는 것을 추측하지 않고 이름을 대어 선언한다. `mjcf_out`은 `<option>`
  전체를 쓰고, 이제 "전체"에 `ls_iterations`와 `<flag eulerdamp>`가 포함된다 — `ls_iterations = 5`
  와 `eulerdamp` 비활성 아래에서 학습된 정책을 MuJoCo 기본값(50, 활성)으로 밟는 것은 정책이 본
  적 없는 물리로 밟는 것이다. 이 패킷이 하는 유일한 `src/` 변경이고, 이 패킷이 레이어 4에
  존재하는 이유다.
- §4.3, §3.5: `MuJoCoCpuBackend`는 tier 1이 아니라 `DeterminismTier::PhysicsMeaning`을 선언한다
  — 외부 백엔드는 절대 tier 1을 선언하지 않는다 — 그래서 일치 오라클의 허용오차는 비트 동일이
  아니라 선언된 물리 의미 허용오차다.
- §5.1: Task IR이 `ObservationSpec`을 선언하고 전처리도 신경망도 소유하지 않는다. Observation
  IR이 채널을 구현하고, Learning IR이 정책을 소유한다. 문서 넷, 해시 넷, `es ir check` 하나.
- §7.4, §7.5: 48차원 채널은 Task IR에서 한 번 선언되고 Observation IR에서 구현된다.
  `temporal.window = { n_steps = 1 }`은 시간 모델의 레이어 2이자 업스트림 `history_len = 1`을
  소리 내어 말한 것이다(XIR-011이 `observation_window`와 대조한다).
- §8.3, §8.4: `StateEncoder{Mlp}` → `PolicyHead{Regression}` → `ActionChunker` →
  `Normalizer{Inverse}`. `replanning_hz = 50`은 50 Hz 제어율의 정수 약수다(XIR-023).
- §9.2–§9.4, INV-12, INV-13: Deployment IR이 관절·속도·가속도·토크·행동 변화율 한계와 워치독과
  폴백을 싣는다. 모든 한계는 정책이 학습된 것보다 최소한 넓다 — 넓히되 절대 끄지 않는다 — 그리고
  `SafetyPlane::validate`의 시그니처는 손대지 않았다.
- §12.4: 벽시계 시간은 물리 스텝 1,000회당 관측값으로 보고한다. `step/s` 주장은 없다.
- §18.1: 제어율과 추론율은 정수 `TickRate`이고, 데드라인은 제어 주기의 정수배다.
- §25.1: 프로비넌스 테스트는 네트워크에서 온 바이트를 **파싱하기 전에** blake3를 검증한다.
- §28: 이 트랙이 속한 마일스톤.

## oracle

```
cargo fmt --check
cargo clippy -p es-assets -p es-physics-backend -p es --all-targets -- -D warnings
cargo test -p es-assets --test go1_provenance
cargo test -p es-physics-backend --test go1_step
cargo test -p es --test cli quadruped
cargo run -p es -- ir check tests/fixtures/quadruped/task.toml \
    tests/fixtures/quadruped/observation.toml \
    tests/fixtures/quadruped/learning.toml \
    tests/fixtures/quadruped/deployment.toml
cargo xtask ci
```

참조 — 이 머신에 없을 수 있는 것을 필요로 하는 절반:

```
# 업스트림 교차 검사: 네트워크 또는 캐시된 사본
ES_PLAYGROUND_CACHE=$HOME/cache/playground \
  cargo test -p es-assets --test go1_provenance -- --nocapture

# 물리 오라클: mujoco가 있는 Python. GPU 서버에서는 ~/venvs/es (mujoco 3.13):
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es-physics-backend --test go1_step -- --ignored --nocapture
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es --test cli quadruped -- --nocapture
```

`go1_provenance.rs`는 `go1_primitives.PROVENANCE.json`에서
`{ repo, path, path_scene, commit, blake3_go1_mjx_feetonly_xml, blake3_scene_flat_terrain_xml,
menagerie_commit, license, derivation }`을 읽고, 두 업스트림 파일을 있으면
`$ES_PLAYGROUND_CACHE/<commit>/<file>`에서, 없으면
`raw.githubusercontent.com/google-deepmind/mujoco_playground/<commit>/.../go1/xmls/<file>`에서
`curl -fsSL`로 `target/`에 받고, **파싱하기 전에** 매니페스트의 blake3와 대조한다. 핀은
`124a73fa3303f75a62f8fe04d329b829ed0ebdfb`(릴리스 v0.2.0)다. 캐시도 네트워크도 없으면
`SKIP go1_provenance: <why>`, 실행되었으면 `RAN go1_provenance (<file>)`. blake3 불일치는 스킵이
아니라 **실패**다. 파싱 전에 업스트림 텍스트에 두 가지 편집을 가하는데, 둘 다 임포터의 한계이고
단언 대상 중 어느 것도 건드리지 않는다: `<mesh class="go1" ...>` → `<mesh ...>`(임포터가
`<default>`보다 `<asset>`을 먼저 읽는다), 그리고 `<site>` 줄 제거(임포터는 3벡터 site `size`를
원하고 업스트림은 MJCF의 스칼라 축약형을 쓴다).

일곱 개의 프로비넌스 테스트가 열거하는 것:

- `derivative_has_the_upstream_joints_and_actuators` — 업스트림 선언 순서 그대로의 열두 힌지와
  바이트 동일한 `axis`, `range`, `damping`, `armature`, `frictionloss`, 그리고 게인·`ctrlrange`·
  `forcerange`·`gear`·구동 관절이 같은 열두 개의 `<position>` 액추에이터.
- `derivative_has_the_upstream_body_frames_and_inertials` — trunk와
  `{FR,FL,RR,RL}_{hip,thigh,calf}`: 바이트 동일한 `pos`, 공유 `mjcf::orient` 정규화 후 바이트
  동일한 방향, 같은 부모, 같은 질량과 관성 대각.
- `derivative_substitutes_only_the_visual_mesh_geoms` — 업스트림의 모든 *프리미티브* geom이 포즈
  그대로 존재하고, 없는 것은 전부 메시이며, **떨어진 메시 하나당 같은 body에 프리미티브 하나가
  더해졌다** — 삭제가 아니라 치환이다. 치환은 정확히 13개이고, 파생본에는 메시도 메시 에셋도
  어디에도 없다.
- `derivative_has_the_upstream_option_block` — `scene.options`가 업스트림과 같고, 풀어 쓰면
  `timestep = 0.004`, `iterations = 1`, `ls_iterations = 5`, `eulerdamp` 비활성,
  `integrator = Euler`, `cone = pyramidal`.
- `derivative_carries_the_home_keyframe` — 파생본의 `home` `qpos`가 고정된 19개 숫자이고,
  *업스트림 장면 파일*의 `home`도 같은 19개다. 즉 행동의 영점을 신뢰하지 않고 업스트림 바이트에
  대조한다. 인라인된 바닥도 업스트림의 것이다.
- `derivative_parses_with_only_the_enumerated_warnings` — 임포터 경고 69개, 하나도 빠짐없이
  `<keyframe>`, geom `priority`/`group`, 머티리얼 `rgba`, 카메라 `mode`, `<position
  inheritrange>` 중 하나 — 매니페스트가 열거한 목록 그대로다. 그리고 파일 텍스트에 `<include`,
  `meshdir`, `type="mesh"`, `<sensor`, `.stl`이 없다.
- `joint_limits_are_derived_not_transcribed` — 모든 힌지가 유한하고 순서가 맞는 `range`를, 모든
  액추에이터가 양쪽이 있는 `forcerange`를 가진다(`deployment.toml`이 복사하는 값이다). 그리고
  장면은 자유 관절 하나 + 힌지 열둘 — 이 픽스처를 그 전의 모든 픽스처와 다르게 만드는 부유 기저.

`go1_step.rs`:

- `the_emitted_mjcf_keeps_the_playground_option_block` — **Python 불필요, PR CI에서 돈다.**
  재생성된 MJCF가 `timestep="0.004" integrator="Euler" iterations="1" ls_iterations="5"
  cone="pyramidal"`과 `<flag eulerdamp="disable"/>`를 싣고, 메시도 `<include>`도 없으며,
  `trunk`·`FR_calf_joint`·`RL_hip`·`floor`를 여전히 이름으로 가진다.
- `go1_stands_from_home_and_agrees_with_mujoco_directly` — `#[ignore]`, `ES_PYTHON` 필요.
  `nq/nv/nu = 19/18/12`. `home`으로 리셋하고 `ctrl`을 `home` 자신의 값(행동 0)으로 두고 250 제어
  틱 × 5 서브스텝 = 1250 물리 스텝. `StepReport` 실패 없음, 상태는 유한하게 유지되고, 몸통은
  0.20 m 위에 남는다. 이어서 같은 XML을 `mujoco`로 직접: MuJoCo가 `iterations = 1`,
  `ls_iterations = 5`, `eulerdamp = false`, `timestep = 0.004`를 보고해야 하고 — 즉 자기 기본값이
  아니라 우리 픽스처의 솔버 설정을 읽었어야 하고 — 기준 실행도 서 있어야 하며, 좌표별 최대
  `|Δqpos|`가 1e-3 미만이어야 한다. 두 경로의 물리 스텝 1,000회당 벽시계 시간을 출력한다.

`crates/es/tests/cli.rs`, `quadruped`에 걸리는 다섯 테스트:

- `regenerate_quadruped_documents`(`#[ignore]`) — 생성기. 파싱된 장면에서 네 문서를 전부 만들어
  쓴다. 그래서 픽스처의 어떤 해시·관절 한계·토크 한계·리셋 자세도 손으로 타이핑되지 않는다.
- `quadruped_documents_validate_and_cross_check` — 넷 각각이 `validate()`를 깨끗하게 통과하고,
  `cross::check`가 비어 있고, `es ir check` 바이너리가 0으로 종료하며 해시 체인을 출력하고,
  문서들이 api-note가 말하는 것을 말한다: 입력 48, 출력 12, `history_len = 1`, 50 Hz, 청크 1.
- `quadruped_documents_are_what_the_generator_produces` — 커밋된 바이트가 생성기 출력과 같다.
  즉 손으로 고치면 테스트가 실패한다.
- `quadruped_bundle_runs_100_ticks_through_the_safety_plane` — 네 문서가 다시 열리는
  `policy.esb`로 묶이고, `deployment.toml`로 만든 진짜 `SafetyPlane::<12, 1>`에 100 제어 틱이 두
  번 통과한다. 학습되지 않은 정책의 `tanh` 제한 행동은 매 틱 클램프되지만 폴백을 래치하지 않고
  위치 한계를 벗어나지 않으며, 물리적으로 따라갈 수 있는 명령(0.02 rad/틱)은 매 틱 그대로
  액추에이터에 도달한다. 두 번째가 상수 클램프 엔벨로프가 첫 번째를 통과하는 것을 막는다.
- `quadruped_eval_run_names_the_observation_gap` — 한 스위트짜리 Evaluation IR에 대한
  `es eval run`은, 문서화된 종료 코드 3으로 스킵하거나(`mujoco`/`torch` 없음), 리포트를 쓰지 않고
  알려진 관측 간극을 **이름을 대며** 거부하거나, 간극이 메워지는 날 0으로 종료한다. 절대
  패닉하지 않고 실행을 흉내 내지 않는다(§1.4).

## acceptance

- `tests/fixtures/mjcf/go1_primitives.xml`이 파싱되고 MuJoCo CPU 백엔드로 로드되며,
  `go1_primitives.PROVENANCE.json`이 열거한 치환 외에는 업스트림과 다르지 않다 — 위의 일곱
  테스트가 증명하는 것이지 여기서 주장하는 것이 아니다.
- `<option timestep / integrator / iterations / ls_iterations>`와 `<flag eulerdamp>`가
  `parse_mjcf` → `scene_to_mjcf`를 살아서 통과한다. 이 패킷 전에는 뒤의 둘이 통과하지 못했다.
- `tests/fixtures/quadruped/`의 네 문서가 개별 검증되고 함께 교차 검사되며, `es ir check`가 그
  위의 해시 체인을 출력한다.
- 학습되지 않은 정책의 100 제어 틱이 관절 12개에서 패닉·폴백·엔벨로프 이탈 없이 Safety Plane을
  통과한다.
- 네 문서의 모든 숫자가 `go1_primitives.xml` 또는 api-note에서 유도되고, 생성기 테스트가 커밋된
  파일이 그 출력임을 증명한다.
- 이 머신에서 **돌릴 수 없었던** 오라클은 정확한 서버 명령과 함께 명시된다(위 **oracle**). 조용히
  건너뛰지 않는다.

## forbidden

- `crates/es-safety` — 어떤 변경도 없음. 엔벨로프는 `deployment.toml`에서 넓힌다.
  `SafetyPlane::validate`의 시그니처는 그대로이고(INV-13) 어떤 것도 plane을 끄지 않는다(INV-12).
- `crates/es-eval`, `crates/es-env` — 부유 기저를 위해 `joint_state`의 선행 `NJ` 관례를 고치거나
  `input_sources`에 기저 속도·자이로·`qvel` 캡처를 가르치는 것은 여기서 `es eval run`을 풀어 주는
  패킷의 일이다(설계 노트 3.2, 3.3). 이 패킷이 아니다.
- `crates/es-ir`, `crates/es-ir-types` — 새 노드, 새 `StateEncoderKind`, 활성화 파라미터, `tanh`
  전부 없음. 설계 노트 3.6이 그것들을 사람에게 남긴 질문으로 적었고, INV-17이 확장점을 일곱 개로
  묶으며 이 패킷은 하나도 더하지 않는다.
- `crates/es-policy/src/lower/torch.rs` — swish 대 ReLU는 임포트 패킷의 첫 질문이고(설계 노트
  3.4), 활성화를 바꾸면 낮춰진 모든 정책의 해시가 움직인다.
- `Geom`에 `priority` 필드를 더하거나, **spec**에 이름을 댄 두 `<option>` 값 외로 MJCF 임포터를
  넓히는 것.
- 로더의 메시 지원, 5개 Unitree STL 벤더링, 파생본을 통과시키려고 `go1_provenance.rs`의 동등성을
  허용오차로 완화하는 것.
- 무엇이든 학습시키는 것, GPU 서버를 건드리는 것, 골든을 수정하는 것.
- `docs/design/visible-learning.md`와 `docs/ARCHITECTURE*.md`.
