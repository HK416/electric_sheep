# M5 V9 — 쇼케이스 렌더

설계 노트: `docs/design/visible-learning.md` **섹션 7.17**, 그리고 거기까지 온 경위는 7.1–7.3
(루프 안의 렌더러), 7.8 (V3 모자이크 영상), 7.12–7.13 (전문가를 통과시키는 하네스). 렌더러는
`docs/design/renderer.md` 섹션 2, 3, 5. 스펙: §3.4, §3.5, §4.2, §15.2, §15.3, §25.1. 이 파일보다
먼저 읽을 것.

## 이 패킷이 답하려는 질문

**이 데모를 누가 볼 수 있는가?**

없다. `es video mosaic`은 *정책이* 읽는 프레임을 타일로 붙인다 — Observation IR이 선언한 단
하나의 상단 카메라에서 나온 96×96 `Rgb8` 타일 — 그리고 V3의 4×4 그리드는 썸네일 열여섯 개를
바로 위에서 내려다본 384×392 mp4다. 이 데모가 지금까지 만든 산출물은 숫자이거나 해시이거나 그
모자이크다. 오너가 사람들에게 보여 주고 싶은 한 문장 — "팔이 큐브를 집어 통에 넣는다" — 은 그
어디에도 없고, 있을 수도 없다. 파이프라인 안의 유일한 카메라가 신경망이 학습한 그 카메라이고
의도적으로 96픽셀이기 때문이다.

그래서 V9는 **이미 끝난 실행**을, 어떤 씬에도 어떤 `ObservationSpec`에도 어떤 해시에도 없는
카메라로 다시 렌더하는 사람용 렌더를 추가한다.

## 결정들, 그리고 각각의 근거

### 1. 루프 안의 두 번째 카메라가 아니라 리플레이 렌더

후보는 둘이었다.

**(a) 리플레이 렌더.** 모든 실행이 틱마다의 상태 궤적을 기록하고, 새 명령이 그 파일로부터 씬을
다시 포즈시켜 독립 카메라로 렌더한다.
**(b) 실행 중의 두 번째 `ImageSpec`.** `es eval run --showcase WxH`가 틱마다 두 번째 뷰를 옆
디렉터리에 렌더한다.

**(a)이고, 코드가 그쪽을 더 싸게 만든다.** `es-render`는 이미 삼각형 씬과 카메라의 순수
함수다. `EnvRenderer::frame`은 `body_poses` → `TriScene::from_scene_with_poses` →
`Renderer::render` → `read_tile`이 전부다 (`crates/es-env/src/render.rs`). 실행 중인 env에서
가져가는 것은 `StateView::xpos` / `xquat` — 바디당 double 7개 — 뿐이다. 그래서 "렌더러가 받은
것을 기록한다"와 "로봇이 무엇을 했는지 기록한다"는 같은 파일이고, 그 파일이 생기는 순간 렌더는
물리 백엔드도 정책도 Torch도 MuJoCo도 프로세스에 없는 *리플레이*가 된다.

(b)가 못 하고 (a)가 하는 것:

* 900스텝 에피소드를 MuJoCo와 신경망으로 다시 돌리지 않고, 끝난 실행을 다른 해상도·다른
  각도·다른 렌더 경로로 다시 렌더하기;
* **전문가**를 렌더하기. 전문가는 `es eval run`을 아예 거치지 않고 `es loop collect --expert`로
  가며, 같은 세 줄에서 같은 파일을 얻는다;
* 검증되기. 루프 안의 두 번째 카메라에는 비교할 기준이 없다. 리플레이에는 있다 — 실행 **자신의**
  `ImageSpec`으로 실행 **자신의** 카메라를 통해 궤적을 렌더하면 그 바이트는 실행이 쓴 프레임과
  같아야 한다. 이 패킷의 오라클이 그것이고, (a)에서만 가능하다.

그리고 궤적은 영상과 무관하게 가질 값어치가 있다. 평가 실행은 `report.json`, `events.json`,
픽셀을 기록하지만 팔과 큐브가 어디 있었는지는 아무것도 기록하지 않는다.

### 2. 파일: `qpos` 옆의 포즈, 그리고 왜 둘 다인가

`crates/es-env/src/traj.rs`, `.estraj`:

```text
매직   "ESTRAJ01"          8바이트
nq, nv, nbody, ticks       u32 리틀엔디언 4개
바디 id                    nbody x 16바이트 (`StableId`)
행                         ticks x (nq + nv + nbody*7) x f64 리틀엔디언
```

`qpos ‖ qvel`은 프로버넌스에서 "로봇이 무엇을 했는가"의 뜻이고, 데모 자신의
`observation.state`가 이미 담고 있는 행이다. **바디 포즈**가 함께 있는 이유는 그것이 렌더러가
먹는 것이기 때문이다. 리플레이 시점에 `qpos`로부터 다시 구한다는 것은 정기구학을 뜻하고, 그것은
리플레이 프로세스에 물리 백엔드를 들이고 `mj_forward`의 두 번째 구현이 어긋날 여지를 만든다는
뜻이다. 저장해 두면 리플레이는 *렌더러가 받았던 바로 그 바이트*이고, 아래 오라클에는 허용
오차가 없다. 데모 씬에서 이것은 틱당 f64 109개 — 에피소드당 800 KB 미만이며, 같은 에피소드가
이미 쓰는 96×96 프레임 24 MB에 비하면 없는 것과 같다.

우연이 아니라 결정적인 두 가지:

* `Trajectory::poses`는 `Pose::new`가 아니라 **구조체 리터럴**로 `Pose`를 만든다. 기록된 값은
  이미 `Pose::new`의 출력이고, 정규화된 단위 쿼터니언을 한 번 더 정규화하면 마지막 비트가
  움직인다 (`es_math::Quat::normalize`는 노름으로 나눈다). 비트 단위 일치가 이 파일의 주장
  전부다.
* `Trajectory::push`는 제어 틱이 시작하는 곳이 아니라 **프레임이 캡처되는 틱**을 기록한다.
  `observation_delay` 아래에서 드롭된 관측은 아무것도 렌더하지 않으므로, 모든 스위트에서 궤적
  인덱스와 프레임 인덱스가 같은 숫자로 남는다 — 오라클이 프레임 단위로 비교 가능한 이유다.

경계 있는 파싱 (§25.1): 바이트 길이는 헤더로부터 유도되고 아무것도 할당하기 전에 전부
검사된다. 잘린 파일, 너무 긴 파일, 스트라이드 0 헤더, 길이 계산의 오버플로가 각각 이름과 함께
거절된다.

### 3. 항상 기록, 기억해야 할 플래그가 아니다

`es eval run`은 `<out>/traj/<suite>-<NN>.estraj`를, `es loop collect`는
`<root>/traj/ep-<NNN>.estraj`를 무조건 쓴다. `--traj <dir>`는 위치만 옮긴다. 라이브러리
기본값은 `None`이므로 (`RunConfig::traj_dir`, `CollectSpec::traj_dir`) 기존 호출자와 테스트의
동작은 하나도 바뀌지 않는다.

`--jobs N`에서 샤드들은 프레임 디렉터리를 공유하듯 궤적 디렉터리 하나를 공유한다. 셀 이름이
전역적으로 고유하고 샤드들이 서로소인 셀을 소유하므로 병합할 것이 없다. 수집기의 파일은
`LeRobot` 데이터셋 **옆에** 놓이지 그 파일들 안에 놓이지 않는다. `dataset_content_hash`는
에피소드 메타데이터가 지목하는 parquet 파일만 해시하고 루트를 훑는 코드가 없으므로, 어떤
데이터셋 해시도 움직이지 않는다.

### 4. 카메라가 명령줄에 있는 이유는 씬이 카메라를 더 가질 수 없기 때문이다

`es video showcase --eye X,Y,Z --look-at X,Y,Z [--fov D]`, 월드 업은 `+Z`,
`es_env::render::look_at`이 §3.1의 `OpenCV` 프레임으로 만든다.
`tests/fixtures/mjcf/so101_pick_place.xml`에 `<camera>`를 추가하는 안은 **아니다**.
`scene_hash`는 `task_hash`로, 그것은 `observation_hash`로, 그리고 학습된 번들 전부로 이어지므로,
데모 씬에 카메라를 더하는 것은 이 영상이 보여 주려는 바로 그 체크포인트를 무효화하는 일이다.
해시를 움직이는 카메라는 독립적인 카메라가 아니다.

`--camera NAME`은 씬이 *실제로* 선언한 카메라를 렌더한다. 이것은 편의 기능이 아니다. 실행
자신의 `--width`/`--height`와 함께 쓰면 그것이 곧 실행 자신의 관측을 리플레이하는 것이고, 그것이
리플레이를 실행과 대조하는 방법이다 (아래 오라클).

`INV-14`는 여기 해당하지 않는다. 아무것도 리샘플되지 않는다. 쇼케이스 `ImageSpec`은 다른 모든
카메라와 마찬가지로 `--fov`와 `--width`×`--height`로부터 계산되고, `Resize`/`Crop`은 그대로
Observation IR 노드다. 이 명령은 어떤 IR도 읽거나 쓰거나 해시하지 않는다. mp4는 애초에 해시
체인 안에 없었고 (§5.3) 쇼케이스 렌더의 프레임도 아니다 — 그것은 실행의 *기록*이 아니라 실행의
*재렌더*다.

### 5. `Rs` 전용이며, 경로 추적기는 장치가 없어서 빠진 것이 아니다

`Pt` 경로는 **오라클 서버에서 헤드리스로 돌아간다**. 거기서
`cargo test --release -p es-render`는 디스플레이 없이 RTX 4090에서
`gpu_path_tracer_matches_the_cpu_reference_at_1spp`, `gpu_pt_and_rs_agree_on_geometry`,
`gpu_restir_and_svgf_match_the_cpu_within_tolerance`를 통과한다. 그런데도 `es video showcase`가
`Rs` 전용인 이유는 장치와 무관하다. `PT_CHANNELS`는 `PtRadiance`(선형 `f32`)와 기하 채널 셋이고
`Rgb8`은 거기 없다 (`crates/es-render/src/lib.rs`). 선형 복사휘도를 8비트 프레임으로 바꾸는 것은
톤 매핑 결정이고, 뒷받침할 오라클도 없이 룩(look)에 대한 결정을 CLI 플래그에 넣는 것은 이 패킷이
할 일이 아니다. `--pt` 플래그는 방어할 수 있는 노출과 커브를 가진 다음 패킷의 것이며,
`EnvRendererCfg::path`는 이미 그 변형을 들고 있다.

### 6. 바뀌지 **않은** 것

새 트레이트 없음 (`INV-17`): `Trajectory`는 구조체이고 렌더 경로는 `es-render`가 이미 가진
것이다. Safety Plane 코드 없음, `es-safety` 의존 없음, 엔벨로프 숫자 없음. IR 타입도, 스키마도,
해시도 없음. 골든 파일 없음: 쇼케이스 렌더는 골든을 만들지도 수정하지도 않는다 —
`tests/golden/render/`의 모든 골든을 생성하는 `es_render::cpu`는 오라클에서 *기준*으로 쓰일 뿐
재생성되지 않는다.

## oracle

이 순서로 실행 가능하다.

1. **포맷이 왕복하고 잘린 파일이 거절된다** (§25.1) — `cargo test -p es-env --lib traj`.
   `(nq, nv, nbody, ticks)`에 대한 속성 테스트, 그리고 `a_truncated_file_is_refused` (절단점 6개와
   뒤에 붙인 바이트 1개), `a_hostile_header_allocates_nothing_it_cannot_read`.
   `a_replayed_tick_is_the_pose_map_the_renderer_was_handed`가
   `Trajectory::poses(t) == body_poses(model, state, 0)`을 정확히 고정한다. GPU도 백엔드도 필요
   없고 CI에서 돈다.

2. **리플레이는 곧 실행이다, 실행 자신의 `ImageSpec`에서** —
   `cargo test --release -p es --features render --test cli --
   a_showcase_replay_reproduces_the_frames_the_policy_saw`. 스크립트 전문가가
   `es_eval::Evaluation` — 진짜 러너, 진짜 플레인, 진짜 Task IR — 을 통과하되 프레임 소스는 데모
   자신의 96×96 `ImageSpec`에서 CPU 레퍼런스 래스터라이저 (`es_render::cpu::rasterize`, 모든 렌더
   골든을 생성하는 함수)이고, 실행은 궤적을 기록한다. 그 다음 모든 틱을 **파일로부터** 다시
   렌더해 실행이 쓴 프레임과 바이트 단위로 비교한다. **서버 오라클**: `mujoco`가 필요하고 없으면
   이유를 출력하고 건너뛴다. Vulkan 장치가 아니라 CPU 래스터라이저인 것은 의도적이다 — 주장은
   상태에 관한 것이고, 렌더러는 어느 경로에서든 상태의 순수 함수다.

3. **같은 주장을 실제 실행에서, 실제 명령으로, GPU에서** —
   `ES_SHOWCASE_RUN=<끝난 실행> cargo test --release -p es --features render --test cli --
   --ignored showcase_replay_of_a_real_run_is_bit_identical`. 끝난 `es eval run --frames` 산출물에
   `es video showcase --camera overhead --width 96 --height 96 --cell nominal-00`을 돌려 모든
   프레임을 기록된 것과 비교한다.

4. **영상** — 실행에 `es video showcase`를 돌린 뒤 `python/es/encode_video.py` (또는 V3가 그랬듯
   같은 원시 프레임에 ffmpeg)로, 프레임 수가 기록된 틱 수와 같은 mp4를 만든다.

5. `cargo xtask ci`.

## acceptance

* `.estraj`가 왕복하고, 잘린 파일을 거절하며, `Trajectory::poses`가 `body_poses`와 비트 단위로
  같다 (오라클 1).
* 모든 `es eval run`과 `es loop collect`가 플래그 없이 에피소드당 궤적 하나를 쓰고, 그 때문에
  데이터셋 해시·리포트·락이 하나도 움직이지 않는다.
* 실행 자신의 `ImageSpec`에서의 리플레이가 기록된 관측 프레임을 바이트 단위로 재현한다 — CPU
  레퍼런스 경로 (오라클 2)와 CLI를 통한 GPU 경로 (오라클 3) 양쪽에서.
* `es video showcase`가 `NNNNNN.bin`과 `layout.json` 하나를, 기록된 틱마다 한 프레임씩 쓰고,
  인코더가 그만큼의 프레임을 가진 H.264 mp4로 만든다.
* `render` 피처 없이 빌드된 `es`는 `es video showcase`를 이름과 함께 거절하고 Vulkan을 전혀 링크
  하지 않는다.
* `cargo xtask ci` 통과.

## forbidden

* **씬 파일.** `tests/fixtures/mjcf/so101_pick_place.xml`에 카메라·지옴·바디를 추가하지 말 것.
  `tests/fixtures/visible-learning/` 아래 문서를 재생성하지 말 것. 그것들로부터 이어지는 모든
  해시가 이 패킷이 리플레이해야 하는 학습된 번들을 지목한다.
* **Safety Plane과 `es-safety`.** 건드리지도, 의존하지도, 넓히지도 않는다.
* **모든 IR.** 새 노드·필드·스키마 버전·해시 항 없음. 쇼케이스 카메라는 명령줄 인자이고 계속
  그렇게 남는다.
* **골든.** `tests/golden/render/*`는 읽기 전용이며, 이 패킷은 추가하지도 재생성하지도 않는다.
* **재학습, 그리고 모든 평가 표.** V9는 실행을 렌더할 뿐 정책을 측정하지 않는다. 영상을 위해
  만든 네 에피소드짜리 평가 문서는 자신의 `evaluation_hash`를 가진 별개 파일이고, 고정된 열여섯
  에피소드 측정이 아니다.
* `es video mosaic`. 하던 일을 그대로 한다.
