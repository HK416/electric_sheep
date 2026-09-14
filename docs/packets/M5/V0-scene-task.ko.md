# M5 V0 — SO-101 큐브-바구니 장면과 네 개의 IR 문서

설계 노트: `docs/design/visible-learning.md` 섹션 4 (에셋 정책)와 5.4 (성공 술어); 섹션 2.3을 먼저
읽을 것 — `<include>`가 거부되고 `type="mesh"` geom은 MuJoCo에도 렌더러에도 도달하지 못하며, 그래서
이 패킷이 프리미티브 전용 파생본을 벤더링한다. V0b, V4와 독립; V1을 막는다.

## context

```
tests/fixtures/mjcf/so101_pick_place.xml
tests/fixtures/mjcf/so101_pick_place.LICENSE
tests/fixtures/mjcf/so101_pick_place.PROVENANCE.json
tests/fixtures/visible-learning/task.toml
tests/fixtures/visible-learning/observation.toml
tests/fixtures/visible-learning/learning.toml
tests/fixtures/visible-learning/deployment.toml
crates/es-assets/tests/so101_provenance.rs
crates/es-physics-backend/tests/so101_scene.rs
crates/es/tests/cli.rs
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V0-scene-task.md
docs/packets/M5/V0-scene-task.ko.md
```

메모: 이 패킷은 **픽스처와 테스트뿐**이다 — 어떤 `src/`에도 한 줄을 더하지 않으므로
`cargo xtask context-budget`은 움직일 수 없다. `crates/es/tests/cli.rs`에 기존 `es task compile`로 네
문서를 컴파일하는 테스트 하나가 추가된다. `Cargo.toml` 변경은 없다: 프로비넌스 테스트는 새 HTTP
의존성이 아니라 시스템 `curl`을 `std::process::Command`로 부른다.

## spec

- §1.4: 픽스처는 리뷰가 아니라 상류 모델과의 실행 가능한 교차검증으로 판정된다.
- §4.3, §17.2: 백엔드가 매핑할 수 없는 장면 항목은 **이름을 밝힌** `Unsupported`다; 파생본은 그 안의
  어떤 것도 매핑 불가가 아니게 하려고 존재하며, 테스트는 파일 전체가 경고가 아니라 로드됨을 증명한다.
- §5.1: Task IR은 `ObservationSpec`을 선언할 뿐 전처리도 신경망도 소유하지 않는다; Observation IR이
  이미지 체인을, Learning IR이 정책을 소유한다. 문서 넷, 해시 넷.
- §6.3: 큐브 포즈 변화는 큐브 free joint의 `qpos`에 대한 Task IR `Randomization` 노드다. 질량·마찰·
  액추에이터 게인 타깃은 **선언하지 않는다**: `es-env`는 추출값을 기록하지만 백엔드로 보내지 않으므로
  (`crates/es-env/src/randomize.rs:24-26`), 선언하면 물리에 도달한 적 없는 숫자가 에피소드 기록에
  들어간다.
- §7.2, INV-14: `ImageSpec`은 렌더러가 만들 크기로 Observation IR에 한 번만 선언된다. 이 패킷에는
  `Resize`도 `Crop`도 없으므로 내부 파라미터 변환의 의무도 없다.
- §9.2–§9.4: Deployment IR이 관절 한계, 속도·변화율 제한, 실행 모드, 폴백을 싣는다; V3가 데모 스위트용
  으로 이를 좁힌다. 엔벨로프는 문서이지 스위치가 아니다 (INV-12).
- §18.1: 제어 레이트는 정수 `TickRate`다; 픽스처의 어떤 것도 float 지속시간이 아니다.
- §25.3: 모든 픽스처 문서는 스키마 버전을 싣는다.
- §28.3, §28.7 게이트 7: M1 이래 이 게이트에 없던 바로 그 장면이다.

## oracle

```
cargo fmt --check
cargo clippy -p es-assets -p es-physics-backend --all-targets -- -D warnings
cargo test -p es-assets --test so101_provenance
cargo test -p es-physics-backend --test so101_scene
cargo test -p es --test cli visible_learning
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

레퍼런스 — 상류 교차검증. 네트워크 또는 캐시가 필요하다:

```
ES_MENAGERIE_CACHE=$HOME/cache/menagerie cargo test -p es-assets --test so101_provenance -- --nocapture
ES_PYTHON=$HOME/venvs/es/bin/python cargo test -p es-physics-backend --test so101_scene -- --nocapture
```

`so101_provenance.rs`는 `so101_pick_place.PROVENANCE.json`에서
`{ repo, path, commit, blake3_so101_xml, derivation }`을 읽고, 상류 `so101.xml`을
`$ES_MENAGERIE_CACHE/<commit>/so101.xml`에서, 없으면
`https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/<commit>/robotstudio_so101/so101.xml`
에서 `curl -fsSL`로 `target/`에 받아, **파싱하기 전에** blake3를 매니페스트와 대조한다 (§25.1:
네트워크에서 온 바이트는 신뢰할 수 없다). 고정 커밋은 `ac6b2b09983786f3036cab1000221017fa2193b4`.
캐시도 네트워크도 없으면 `SKIP so101_provenance: <why>`; 실행되면 `RAN so101_provenance`. blake3
불일치는 **실패**이지 스킵이 아니다: 고정 아래에서 상류가 움직였다는 뜻이다.

`so101_scene.rs`는 픽스처를 `es_assets::parse_mjcf` -> `mjcf_out::scene_to_mjcf` ->
`MuJoCoCpuBackend::load`로 통과시키고 모델 모양을 단언한다. `mujoco`가 없으면
`SKIP so101_scene: <why>`.

V1과 V3가 인용할 수 있도록 여기서 고정하는 픽스처 파라미터: 고정 오버헤드 카메라 하나에서 `96x96`
`Rgb8` 이미지; 제어 레이트 50 Hz; 청크 호라이즌 `H = 16`; `NJ = 6`; `max_episode_steps = 400`.
`NJ = 6`과 `H = 16`은 이미 `es eval run`과 `es loop collect`의 const-generic 디스패치 표에 있으므로
(`crates/es/src/cmd/eval.rs:237-256`, `crates/es/src/cmd/loop.rs:124-144`) 표에 행을 더하지 않는다.

`tests/so101_provenance.rs`:

- `derivative_has_the_upstream_kinematics` — 두 파일을 파싱; 여섯 관절이 같은 순서로 바이트 동일한
  이름으로 나타나고 `axis`, `range`, `damping`, `armature`가 `f64`로 정확히 같다.
- `derivative_has_the_upstream_body_frames` — 체인의 모든 상류 바디가 정확히 같은 `pos`와,
  `es_assets::mjcf::orient` 정규화 후 정확히 같은 방향으로 나타난다.
- `derivative_has_the_upstream_inertials` — 바디별 질량과 대각 관성이 같다.
- `derivative_drops_only_visual_mesh_geoms` — 파생본이 뺀 모든 geom은 상류에서 `type="mesh"`이고,
  상류의 모든 프리미티브 충돌 geom은 여기에 있다.
- `derivative_parses_with_no_warnings` — `parse_mjcf(...).warnings`가 비어 있다: 데모 장면에는
  `SceneDesc`가 버리는 요소가 없다 (`crates/es-assets/src/mjcf/mod.rs:208-218`).
- `derivative_has_no_include` — 파일 본문에 `<include`가 없다. 루트와 바디의 `<include>` 모두
  `MjcfError::Include`이기 때문이다 (`mod.rs:247-251`, `:569-573`).
- `link_lengths_are_derived_not_transcribed` — V1의 IK가 소비할 `Links` 값이 파싱된 `SceneDesc`에서
  계산되어 `target/so101_links.json`에 기록되고, 테스트가 그 유도가 파싱의 순수 함수임을 단언한다.
  따라서 어떤 길이도 Rust 소스에 손으로 복사되지 않는다.

`tests/so101_scene.rs`:

- `the_demo_scene_loads_in_mujoco` — `nu = 6`이고 `ModelInfo.actuator`의 모든 액추에이터 id가 해석된다.
  `nq`는 (팔 관절 + 큐브 free joint의 7 dof) 정확한 값을 여기서 가정하지 않고 `ModelInfo`에 대해
  단언한다.
- `every_scene_item_is_mappable` — `scene_to_mjcf`가 `Ok`를 반환한다: 픽스처의 어떤 geom·액추에이터·
  센서도 `PhysicsError::Unsupported`가 아니다.
- `the_cube_free_joint_is_randomizable` — Task IR의 `RandomizationPlan::compile`이 선언된 모든 타깃을
  `Target::Qpos`로 해석하고, 서로 다른 `(seed, episode)` 쌍은 서로 다른 큐브 포즈를, 같은 쌍은 같은
  포즈를 준다.

`crates/es/tests/cli.rs`에 테스트 하나 `visible_learning_documents_compile`: 네 픽스처 문서에 대한
`es task compile`이 exit 0이고, 출력하는 네 해시가 두 번 실행에서 동일하다.

## acceptance

작성되는 네 문서 (이 패킷은 새 Rust 타입을 도입하지 않는다):

```toml
# task.toml      §6: 장면 참조, ObservationSpec 선언, 보상, Terminate{Success|Failure|Timeout},
#                큐브 free joint qpos에 대한 Randomization, ResetState, max_episode_steps = 400
# observation.toml §7: ImageInput(96x96 Rgb8, 카메라 하나) + qpos/qvel 상태 체인; Resize 없음,
#                Crop 없음, MultiViewPack 없음 (crates/es-compile/src/plan.rs:545가 거부)
# learning.toml  §8: VisionEncoder{ResNet18} -> StateEncoder -> Fusion -> TemporalEncoder{Transformer}
#                -> PolicyHead{Regression} -> ActionChunker, 그리고 Normalizer 노드들; ArchKind::Act
# deployment.toml §9: NJ = 6, H = 16, 50 Hz, 관절/속도/변화율 제한, 실행 모드, 폴백
```

- 성공 술어는 큐브 free joint의 `qpos`와 `qvel`만 쓰는 `Terminate { kind: Success }` 콘이다 — 큐브
  중심이 바구니 AABB 안이고 `|v|`가 경계 미만 (설계 노트 섹션 5.4). 카운터 노드도 새 `TaskNode`
  변형도 없다.
- `so101_pick_place.PROVENANCE.json`은 `repo`, `path`, `commit`, `blake3_so101_xml`, 상류 라이선스
  (`Apache-2.0`), 그리고 파생 규칙을 산문으로 명시한다. `so101_pick_place.LICENSE`는 상류 Apache-2.0
  원문 그대로다.
- 픽스처는 평평한 단일 파일이다: `<include>` 없음, `<keyframe>` 없음, `<equality>` 없음, 메시 geom
  없음, `meshdir` 없음.
- 어떤 `src/` 디렉터리에도 0줄 추가.

## forbidden

- `crates/es-ir`, `crates/es-ir-types` — 새 노드, 새 `PerturbationKind`, 스키마 변경 금지. `es-ir`은
  §1.5 예산이 53줄 남았다 (설계 노트 섹션 2.10).
- `crates/es-env/src/randomize.rs`, `crates/es-eval/src/perturb.rs` — 질량·마찰·게인 랜덤화를
  선언하거나 `Target::Scale`이 백엔드에 도달하게 만드는 것은 `PhysicsBackend` 파라미터 API이며 다른
  패킷이다.
- `crates/es-render`, `crates/es-env/src/expert.rs`, `crates/es-policy`, `crates/es/src/cmd/*.rs` —
  V0b, V1, V2.
- `assets/**` (STL 17,230,580 B), `so101.png`, `scene.xml`, `scene_box.xml` 벤더링.
- HTTP 클라이언트 크레이트 추가, 또는 PR 티어의 어떤 테스트든 네트워크에 의존시키는 것.
- 골든 편집, 또는 파생본을 통과시키려고 `so101_provenance.rs`의 동등 비교를 허용오차로 완화하는 것.
