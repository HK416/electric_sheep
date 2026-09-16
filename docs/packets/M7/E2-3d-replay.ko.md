# M7 E2 — 3D 리플레이: 에디터 안에서 재생되는 `.estraj`, 물리 없음, GPU 없음

스펙: §23.3("에디터는 장면을 로컬에서 복제하며 자세/관절만 받는다" — 기록된 궤적은 오프라인
버전의 그 스트림이다), §23.1, §28.10 규칙 3. 확장할 설계 노트: `docs/design/editor-shell.md`
(+ `.ko.md`) 11절. 선행: M5 V9(`es_env::Trajectory`, `.estraj`는 틱마다
`qpos ‖ qvel ‖ body poses`를 담는다; `TriScene::from_scene_with_poses`는 프로세스 안에
백엔드 없이 그 자세들로부터 장면을 다시 자세 잡는다), E1(선택된 셀).

## 질문

`es video showcase`는 장면 파일과 `.estraj`만으로 궤적을 다시 렌더할 수 있음을 증명한다.
**에디터가 CPU에서, `egui` 캔버스 안에서, 스크러버와 함께 인터랙티브한 속도로 같은 일을 할
수 있어서 — 에피소드를 보는 데 `ffmpeg`도 GPU도 서버도 필요 없게 되는가?**

## spec

헤드리스 뷰모델 `crates/es-editor/src/model/replay_view.rs`:

* `ReplayView::open(scene_path, traj_path) -> Result<ReplayView, ReplayError>` — `es video
  showcase`가 쓰는 것과 같은 로더(`es-assets`의 MJCF/URDF; 에디터는 `es-assets`, `es-env`,
  `es-render`, layer 12에 의존해도 된다)로 `SceneDesc`와 `Trajectory`를 로드한다.
* `ReplayView::ticks()`, `ReplayView::qpos(tick)`, `ReplayView::project(tick, &Camera) ->
  Projected`. 여기서 `Camera { eye, look_at, fov_y, width, height }`는 쇼케이스 자신의 자유
  카메라이고(`es_env::render::look_at`, 재사용할 것, 다시 쓰지 말 것), `Projected`는
  `Vec<Tri2d>`다 — 스크린 공간 점 셋, 플랫 셰이딩된 `[u8; 3]`(렌더러 자신의
  `es_shade_lambert` 공식을 `es_render::cpu`의 공개 헬퍼를 통해 CPU에서 적용하거나, 삼각형
  하나에 대해 `es_render::cpu::rasterize`에 고정시켜 테스트하는 작은 재구현), 그리고 깊이
  키 — **뒤에서 앞으로 정렬**된다(화가 알고리즘; `(depth, triangle index)`에 대한 안정
  정렬이라 순서가 입력의 순수 함수다). 근평면 뒤의 삼각형은 눈을 통해 투영되지 않고 잘려
  나간다.
* `ReplayView::orbit(delta_yaw, delta_pitch)` / `zoom(factor)`는 `look_at`을 중심으로 한
  구면 위에서 카메라를 움직인다; `Camera`에 대한 순수 함수다.
* 재생 상태(`playing`, `tick`, `speed`)는 모델에 살고 `advance(dt_seconds, control_rate_hz)`로
  전진한다; 앱은 벽시계 델타만 먹인다.

`app.rs`: E1의 Run 탭 안의 **Replay** 패널(E1이 아직 착륙하지 않았으면 그 자신의 탭 — 어느
쪽인지 설계 노트에 적을 것). `Projected`를 `egui::Mesh` 하나로 그린다(`Tri2d`마다 삼각형
둘은 틀렸다 — 하나여야 한다; 메시의 정점 색이 곧 플랫 컬러), `0..ticks`에 걸친 스크러버,
재생 / 일시정지 / 스텝 버튼, 틱과 시간 레이블, 드래그로 궤도 회전, 휠로 줌. 그 밖의 것은
거기서 결정되지 않는다.

화가 알고리즘은 의도된 단순화다(그것을 지목하는 `ponytail:` 주석): 장면의 볼록 프리미티브가
서로 겹쳐 들어가지 않는 한 정확하고, 교차하는 곳에서만 틀린다; 업그레이드 경로는 R1의 BVH가
그것을 값싸게 만들고 나면 저해상도에서의 픽셀 단위 `es_render::cpu::rasterize`다.

## context

`cargo xtask check-scope`가 읽는 글롭(파서는 정확히 `## context` 제목과 펜스 블록 또는 불릿 목록을 원한다), 그 아래는 같은 범위를 산문으로:

```
crates/es-editor/src/model/replay_view.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/lib.rs
crates/es-editor/src/app.rs
crates/es-editor/Cargo.toml
tests/fixtures/visible-learning/run/traj/**
tests/golden/editor/**
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E2-3d-replay.md
docs/packets/M7/E2-3d-replay.ko.md
```

`crates/es-editor/src/model/replay_view.rs`(신규), `crates/es-editor/src/model/mod.rs`,
`crates/es-editor/src/lib.rs`, `crates/es-editor/src/app.rs`(패널), `crates/es-editor/Cargo.toml`
(`es-assets`, `render` 기능을 **뺀** `es-env`, `es-render`, `es-math`를 보통 의존성으로),
`tests/fixtures/visible-learning/run/traj/`(데모 장면의 짧은 `.estraj` 하나, ≤ 60틱,
`#[ignore]`가 붙은 생성기가 장면의 홈 자세에서 관절 하나를 움직여 `es_env::Trajectory`를
통해 생성 — E1의 픽스처 디렉터리와 공유), `tests/golden/editor/`(정렬 순서 골든 하나:
쇼케이스 카메라에서 틱 0에 대해 `project`가 내는 삼각형 인덱스 순서),
`docs/design/editor-shell*.md` 11절, `docs/packets/M7/E2-3d-replay*.md`.

## oracle

1. `cargo test -p es-editor projection_is_a_pure_function` — `project(t, cam)`를 두 번
   부르면 같은 `Vec<Tri2d>`가 나온다; 틱 0의 삼각형 인덱스 순서가 골든과 같다; 카메라를
   1도 바꾸면 달라진다.
2. `cargo test -p es-editor a_replay_reposes_what_the_renderer_would` — 틱 `t`에 대해,
   `TriScene::from_scene_with_poses(scene, traj.poses(t))`와 `ReplayView`가 투영하는 월드
   삼각형이 같은 집합이다(투영 전 개수도 정점도 비트 단위로 같다) — 리플레이는 쇼케이스가
   렌더하는 것과 정확히 같은 지오메트리를 그린다.
3. `cargo test -p es-editor flat_shade_matches_the_renderer` — 리플레이가 삼각형 하나에
   매기는 색이, 같은 조명, 같은 앰비언트에서 `es_render::cpu::rasterize`가 그 삼각형
   한가운데의 픽셀에 대해 내는 `Rgb8`과 같다.
4. `cargo test -p es-editor playback_advances_by_the_control_rate` — `advance(0.02 s, 50 Hz)`는
   속도 1에서 한 틱, 속도 2에서 두 틱을 움직이고, 마지막 틱에서 클램프된다.
5. `cargo build -p es-editor`; `cargo xtask ci` 통과; 레이어링 검사 통과(`es-editor`는
   무엇에 의존해도 되고; 새로운 어떤 것도 그것에 의존하지 않는다).

## acceptance

오라클 1–5. 오케스트레이터가 로컬에서 실제 V19b 궤적을 열어 팔이 쓸만한 속도로 움직이는
것을 지켜본다(프레임 시간을 관측으로 보고할 뿐 목표치는 없다). 설계 노트 11절이 뷰모델
API, 화가 알고리즘의 한계, 업그레이드 경로를 기록한다.

## forbidden

`crates/es-render/**`, `crates/es-env/**`, `crates/es-assets/**`(읽기만; 렌더러는 R1의
것); `es-env`의 `render` 기능(에디터 빌드에는 Vulkan 없음); 새 외부 의존성(`egui`가 이미
메시를 그린다); `app.rs`에서의 결정; `docs/ARCHITECTURE*.md`; 기존의 모든 골든.
