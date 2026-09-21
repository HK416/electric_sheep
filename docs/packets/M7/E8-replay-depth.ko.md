# M7 E8 — Replay 패널이 깊이 버퍼로 그린다

스펙: §23.3(3D 뷰; "에디터는 장면을 로컬로 복제하고 선택된 env의 포즈를 스스로 렌더링한다"),
§28.10 규칙 3(헤드리스 먼저), §1.4(골든). 오너 노트 2026-09-21: *렌더링된 뷰의 폴리곤이
잘못되어 보인다 — 마치 깊이 버퍼가 없는 것처럼.* 설계 노트: `docs/design/editor-shell.md`
11절(E2, Replay 패널)이 제자리에서 수정된다. **E2**(`ReplayView`, `Camera`,
`TriScene::from_scene_with_poses`, 투영)에 의존한다.

## 질문

E2는 모든 삼각형을 투영하고, 중심점 깊이로 뒤에서 앞으로 정렬한 다음 `app.rs`에
`egui::Mesh`를 건넨다 — 화가의 알고리즘, 당시 `ponytail:`로 표시되었다. SO-101 팔은 모든
관절에서 서로 맞물리는 링크들이라서, 전체 삼각형 정렬은 두 링크가 겹치는 곳마다 틀린다:
베이스 플레이트가 어깨 위에 그려지고, 손가락이 손목을 뚫고 그려진다. **패널이 CPU에서
픽셀별 깊이 테스트로, 패널 해상도로, 50Hz로 재생하기에 충분히 빠르게, 같은 카메라와 같은
셰이딩으로, 그리고 픽셀을 고정하는 골든과 함께 래스터화할 수 있는가?**

## 사양

* **`model/replay_view.rs`가 래스터화한다.** `Projected`(화면 공간 삼각형, 유지됨 — 테스트와
  E2의 골든이 고정하는 것이다)가 `Raster::draw(&Projected, w, h) -> Raster { w, h, rgb:
  Vec<u8>, depth: Vec<f32> }`을 얻는다: 픽셀별 깊이 테스트(가장 가까운 것이 이긴다, 중심점이
  아니라 세 정점의 카메라 공간 `z`로부터 보간된 깊이)를 가진 스캔라인 / 에지 함수
  래스터라이저, `project`가 오늘 셰이딩하는 것과 정확히 같은 삼각형별 플랫 램버트, 같은
  근평면 클립, 전체에 걸쳐 `f32`, 고정된 반복 순서(`Projected` 순서의 삼각형, 행 우선의
  픽셀), 그래서 이미지는 `(trajectory, tick, camera, w, h)`의 순수 함수다. SIMD 없음, 스레드
  없음 — 960×540에서 약 3,000개 삼각형의 장면은 수백만 개의 에지 테스트이며, 이는 밀리초
  단위다.
* **패널이 텍스처를 보여준다.** `app.rs`는 틱 변경(또는 카메라 변경)마다 한 번씩 `Raster`를
  `egui::ColorImage` 텍스처로 업로드하고 패널에 맞춰 스케일해서 그린다; 래스터의 해상도는
  960×540으로 상한이 걸린 패널의 크기다(`Raster::size_for(panel)`, 모델 함수). 오빗/줌
  제스처, 스크러버, 재생/일시정지는 손대지 않는다. `egui::Mesh`와 정렬은 사라진다;
  `ponytail:` 주석도 그것들과 함께 사라진다.
* **골든.** `tests/golden/editor/replay-tick0-320x180.bin`(+ 카메라를 담은 `.json`): 기본
  `Camera`로부터의 틱 0에서 데모 픽스처 궤적(E2의
  `tests/fixtures/visible-learning/run/traj/*.estraj` 혹은 E2의 테스트가 쓰는 것이 무엇이든),
  320×180 `Rgb8`, `ES_GENERATE_GOLDENS=1` 뒤의 `#[ignore]`된 생성기로 한 번 생성된 다음,
  읽기 전용.
* **뒷면.** `Rs` 경로가 하는 것처럼 양면을 그린다(장면의 테셀레이션에는 보장된 와인딩이
  없다); 노트에 그렇게 말하라.
* **여기 없음.** 안티에일리어싱, 그림자, 텍스처, 에디터 안의 GPU 경로(에디터는 Vulkan을
  링크하지 않는다 — E2의 결정이 유지된다), 피킹.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

```
crates/es-editor/src/model/replay_view.rs
crates/es-editor/src/app.rs
crates/es-editor/tests/**
tests/golden/editor/replay-*.bin
tests/golden/editor/replay-*.json
tests/golden/editor/replay-sort-order.json
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E8-replay-depth.md
docs/packets/M7/E8-replay-depth.ko.md
```

`replay_view.rs`(래스터라이저와 그 테스트; `project`와 정렬-순서 골든은 남는다,
`Projected`가 여전히 래스터라이저의 입력이기 때문이다 — 만약 정렬이 이제 죽은 코드라면,
같은 커밋에서 정렬을 지우고 그 골든을 은퇴시키고, 그렇게 말하라), `app.rs`(메시 대신
텍스처), 새 골든 쌍, 설계 노트, 이 패킷.

## 오라클

1. `cargo test -p es-editor a_depth_test_beats_the_painters_sort` — 교차하는 두 삼각형(옆에서
   본 X: 각각이 한쪽에서 더 가깝다)을 두 가지 색으로: 정렬로는 삼각형 하나 전체가 다른
   것을 덮는다; `Raster::draw`로는 왼쪽 절반이 한 색을, 오른쪽 절반이 다른 색을 보여주며,
   픽셀에 대해 단언된다.
2. `cargo test -p es-editor replay_raster_reproduces_its_golden` — 기본 카메라로부터
   320×180에서 틱 0의 픽스처가 골든과 비트 단위로 같다.
3. `cargo test -p es-editor replay_raster_is_a_function_of_its_inputs` — 같은 입력의 두
   그리기는 비트 단위로 같다; 다른 틱이나 카메라는 그렇지 않다.
4. `cargo test -p es-editor replay_raster_is_fast_enough -- --nocapture` — 960×540에서 픽스처
   장면: 20번 그리기의 중앙값을 출력; 이 박스에서 `< 16 ms`가 목표다(단언이 아니라
   관측이다 — 출력하라).
5. `cargo build -p es-editor`; `cargo xtask ci`; `cargo xtask check-scope
   docs/packets/M7/E8-replay-depth.md`.

## 수용 기준

오라클 1–5. 데모 실행 `target/plan-u/demo/out/eval`(장면
`tests/fixtures/mjcf/so101_pick_place.xml`, 케이스 `nominal-00`, 팔이 접힌 틱)의 전/후
스크린샷을 워크트리의 `target/plan-u/e8/` 아래에; 오케스트레이터가 그 확인을 반복한다.
11절의 `ponytail:` 단락은 이제 참인 것으로 교체된다.

## 금지

에디터 안의 GPU 경로나 그 어떤 새 의존성; `Camera`, `project`의 투영이나 셰이딩 숫자를
바꾸는 것(E2의 정렬-순서 골든은 그것이 존재하는 동안 계속 통과해야 한다); 새 골든 쌍
이외의 골든(그리고 은퇴하는 경우 은퇴되는 것); `docs/ARCHITECTURE*.md`. INV-17: 새
트레이트 없음.
