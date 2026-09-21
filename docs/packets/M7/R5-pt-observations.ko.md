# M7 R5 — 패스 트레이싱된 관측: 센서가 자기 렌더 경로를 선언하고, 그 위에서 정책을 학습시킨다

스펙: §15.3("사실적 데이터셋과 골든에는 PT; RS와 PT 사이 RGB는 SSIM 임계값"), §6 / §7.4(Task IR은
`ObservationSpec`을 *선언*한다: 소스와 타입을 가진 이름 붙은 채널들; 전처리는 관측 IR의 것), §5.3(해시
체인 바깥에는 아무것도 없다), §28.10 규칙 1(커밋된 픽셀은 움직이지 않는다; 움직인 바이트는 문서 변경이다),
§3.4(PT는 결정론적이다: 카운터 RNG, 고정된 샘플 순서), §18.3. 소유자의 질문 2026-09-21: *PT 기반 학습은
없는가?* — 없다. 어떤 문서도 그것을 요청할 수 없기 때문이다: `EnvRendererCfg`를 만드는 세 곳
(`es loop collect --frames`, `es eval run --frames`, `es video showcase`)이 `RenderPath::Rs`를
하드코딩하고, R3은 의도적으로 PT 노브를 `EnvRendererCfg` 바깥에 두었다. 설계 노트:
`docs/design/renderer.md` section 12, `docs/design/ir-types.md`(Task IR의 센서 소스). 의존: **R3**(NEE,
톤맵 → `Rgb8`), **R4**(선택적 누적은 관측에 쓰지 *않는다* — 모든 프레임은 독립이다), **T6**(커밋된 문서
옆에 두 번째 문서를 어떻게 추가하는지 보여 준다), **U**(재측정의 레시피 모양).

## 질문

패스 트레이서는 R3 이후 `Rgb8`을 내놓을 수 있고 오라클 서버에서 96×96 프레임 64 spp에 약 3 ms가 든다.
**Task IR의 센서가 "나를 패스 트레이서로 렌더하라"고 말할 수 있는가? 그래서 수집·평가·쇼케이스가 모두 그
센서를 같은 방식으로 렌더하고, 커밋된 문서의 해시와 픽셀은 건드려지지 않는가 — 그리고 패스 트레이싱된
관측으로 학습한 정책이 자신이 수집된 그 하네스를 통과하는가?**

## 스펙

* **선언은 렌더러 config가 만들어지는 곳, 즉 Task IR의 센서에 산다.**
  `es_ir::task::ObsSource::Sensor { id, format }`이 `#[serde(default)]`인 `render: SensorRender`를
  얻는다:
  ```
  SensorRender { path: Rs | Pt { spp: u32, bounces: u32 }, exposure: f32 (1.0), tonemap: Reinhard | Aces }
  ```
  **해시 규칙:** `canonical`은 `render`가 기본값이 **아닐 때만** 블록을 쓴다 — 부재하거나 기본값인
  `render`는 바이트 단위로 오늘의 정규형이므로 커밋된 모든 `task_hash`(와 그 위에서 해시된 모든 것)는
  움직이지 않는다; 테스트가 커밋된 `task.toml`의 해시를 기록된 값에 대해 단언한다. `Pt` 센서는
  `task_hash`를 움직이므로 자기 Evaluation IR 문서가 필요하고(§13.3: 새 비교), 패킷은
  `tests/fixtures/visible-learning/task-pt.toml`(커밋된 `task.toml`에 카메라 채널의
  `render = { path = "pt", spp = 64, bounces = 3 }`)과 `evaluation-pt.toml`(커밋된 `evaluation.toml`이
  `task-pt`의 해시를 가리키는 것)을 추가하고, 두 해시를 노트에 기록한다. 관측 IR은 바뀌지 않는다: 센서가
  *무엇*인지(해상도, 색 공간, 내부 파라미터)는 `ImageSpec`의 것이고, 시뮬레이션이 그것을 *어떻게*
  만드는지는 Task IR의 것이다 — 이것이 `ImageSpec` 필드가 아닌 이유다. 스펙 문구: §6의
  `ObservationSpec`에 한 문단을 추가한다(한국어 정본과 영어 쌍둥이 — **이 패킷의 유일한 `ARCHITECTURE`
  수정이며, 두 파일은 한 커밋에**).
* **함수 하나가 센서로부터 렌더러 config를 만든다.** `es_env::render::sensor_cfg(camera,
  spec: &ImageSpec, render: &SensorRender, frames_dir) -> EnvRendererCfg`가 손으로 만들던 세
  `EnvRendererCfg::rgb(...)` 호출(`eval.rs::renderer_cfg`, `loop.rs`, 씬 카메라에 대한 `showcase.rs`)을
  대체하고, `Pt { spp, bounces }`를 톤맵·노출과 함께 `RenderPath::Pt { spp, bounces, nee: true,
  restir: false, svgf: false }`로 매핑한다; `Rs`는 정확히 오늘의 config로 매핑된다(비트 단위: frames
  픽스처와 모든 RS 골든이 이것을 고정한다). 따라서 `EnvRendererCfg`는 `exposure`/`tonemap`을 **센서가
  선언한 값의 통과 필드**로 얻지, CLI 노브로 얻지 않는다 — 관측 경로에 노브를 주지 않겠다는 R3의 거절은
  유효하다: 결정하는 것은 *문서*다. PT 프레임은 틱마다 독립이며(누적 없음; `RenderConfig.seed` 고정,
  샘플 키는 이미 픽셀마다 다르다), 그래서 프레임은 포즈의 순수 함수이고 수집/평가 동등성 오라클(T7)이
  `Rs`와 똑같이 `Pt`에도 성립한다.
* **비용은 가정하지 않고 측정한다.** RTX 3060과 4090에서 `task-pt.toml`에 대한 `es loop collect
  --frames` 8 에피소드: 관측 렌더의 ms/frame을 RS의 것 옆에 (측정 전까지 `Target / Status:
  unverified`). PT로 수집된 실행의 `es video showcase`는 *씬 카메라*(`--camera`)에 센서의 경로를 쓰고
  자유 카메라에는 `--path`를 남긴다.
* **재측정(서버).** PT 문서에 대한 U3의 구성: PT 프레임으로 시연 200개 수집(`task-pt.toml`),
  `--for-training`으로 bake, `learning-pretrained.toml` + `observation-augmented.toml`로 row-D 설정에서
  20,000 스텝 학습, `evaluation-pt.toml`의 held-out 16 시드로 평가; `success_rate`를 U3의 0.5625 옆에
  `visible-learning.md` 7.31의 **row U4**로 보고한다(새 `evaluation_hash`를 소리 내어 말한다). 또한 같은
  틱의 RS 관측과 PT 관측 사이 SSIM(R3의 `ssim`)을 32 프레임에 대해, 관측 해상도에서의 첫 §15.3 숫자로.
* **스펙 문단(고정).** 한국어(`docs/ARCHITECTURE.ko.md`, §6 `ObservationSpec`):
  > **센서의 렌더 경로.** `Sensor` 소스는 `render = { path = "rs" | "pt", spp, bounces, exposure, tonemap }`로 시뮬레이션이 그 센서를 어떻게 만드는지 선언한다(패킷 M7/R5). 기본값은 `rs`이며 **부재 = 기본값 = 오늘의 정규형**이라 커밋된 `task_hash`는 움직이지 않는다; `pt`는 `task_hash`를 움직이므로 새 문서다(§13.3). 관측 IR은 이것을 모른다 — 센서가 *무엇*인지는 `ImageSpec`이, 시뮬레이션이 그것을 *어떻게* 만드는지는 Task IR이 말한다.

  영어(`docs/ARCHITECTURE.md`):
  > **A sensor's render path.** A `Sensor` source declares how the simulation produces it with `render = { path = "rs" | "pt", spp, bounces, exposure, tonemap }` (packet M7/R5). The default is `rs`, and **absent = default = today's canonical form**, so no committed `task_hash` moves; `pt` moves `task_hash` and is therefore a new document (§13.3). The Observation IR does not know about it — what the sensor *is* belongs to `ImageSpec`, how the simulation *makes* it belongs to the Task IR.

## context

`cargo xtask check-scope`가 읽는 glob과, 같은 범위를 산문으로:

```
crates/es-ir/src/task.rs
crates/es-ir/src/serial.rs
crates/es-ir/tests/**
crates/es-env/src/render.rs
crates/es-env/tests/**
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/showcase.rs
crates/es/tests/cli.rs
crates/es/tests/video.rs
tests/fixtures/visible-learning/task-pt.toml
tests/fixtures/visible-learning/observation-pt.toml
tests/fixtures/visible-learning/evaluation-pt.toml
tests/fixtures/visible-learning/training-u4.toml
crates/es-data/src/roboverse.rs
crates/es-data/tests/lerobot_config.rs
crates/es-data/tests/loop_learning.rs
crates/es-editor/tests/common/mod.rs
crates/es-runtime-embedded/tests/embedded.rs
docs/ARCHITECTURE.ko.md
docs/ARCHITECTURE.md
docs/design/renderer.md
docs/design/renderer.ko.md
docs/design/ir-types.md
docs/design/ir-types.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M7/R5-pt-observations.md
docs/packets/M7/R5-pt-observations.ko.md
```

`es-ir/src/task.rs`(`SensorRender`, 조건부 정규형, 기본값), TOML 모양에 serde 속성이 필요할 때**만**
`serial.rs`, `es-ir/tests`(해시 안정성 테스트), `es-env/src/render.rs`(`sensor_cfg`, 두 통과 필드),
`es-env/tests`, 세 CLI 호출 지점, `cli.rs`/`video.rs`(PT 수집/평가/쇼케이스 테스트), 새 픽스처들,
스펙 문단(두 파일, 한 커밋), 세 설계 노트, 이 패킷.

**만들면서 수정됨(M7/R5, 조용히 넘어가지 않고 기록한다).** 위 목록이 알 수 없었던 두 가지:

* 공개 구조체 **변형(variant)**은 그것의 모든 구조체 리터럴이 함께 자라지 않으면 필드를 늘릴 수 없다.
  그래서 이웃한 다섯 생성 지점(`es-data/src/roboverse.rs`,
  `es-data/tests/{lerobot_config,loop_learning}.rs`, `es-editor/tests/common/mod.rs`,
  `es-runtime-embedded/tests/embedded.rs`)이 각각 정확히 한 줄
  (`render: SensorRender::default()`)만큼 범위에 들어온다. 그 파일들의 다른 줄은 바뀌지 않았다.
* `observation-pt.toml`은 선택이 아니라 **네 번째** 픽스처다: `ObservationIr::task_ref`가 해시 입력이고
  `XIR_001`이 그것을 태스크 자신의 해시와 같게 요구하므로, `Pt` Task IR은 그래프가 바이트 단위로
  동일하더라도 커밋된 관측 IR 문서를 재사용할 수 없다. `evaluation-pt.toml`은 그 문서의
  `observation_hash`를 가리켜야 하므로, 패킷의 "커밋된 `evaluation.toml`이 `task-pt`의 해시를 가리키는
  것"은 필드 하나가 모자란다.

## oracle

1. `cargo test -p es-ir committed_task_hash_is_unmoved_by_sensor_render` — 커밋된 `task.toml`이 기록된
   `eb6efefa…`로 해시되고, 명시적 기본 `render`를 쓴 같은 문서가 동일하게 해시되며, `task-pt.toml`은
   다르다.
2. `cargo test -p es-env sensor_cfg_rs_is_todays_config` — 기본 render의 `sensor_cfg`가
   `EnvRendererCfg::rgb(...)`와 필드 단위로 같고, 그 `RenderConfig`가 frames 픽스처의 첫 프레임을 비트
   단위로 렌더한다.
3. `cargo test -p es --test cli collect_renders_the_sensor_with_the_path_tracer` — `task-pt.toml`에 대한
   전문가 에피소드 1개를 `--frames`로(GPU): 프레임은 `Rgb8`이고 검지 않으며 같은 틱의 RS 렌더와 다르고,
   두 실행이 비트 단위로 같다(장치나 `ES_PYTHON`이 없으면 `SKIP`).
4. `cargo test -p es --test cli eval_refuses_a_pt_policy_on_the_rs_document` — 해시 불일치가 이름으로
   거절된다(task hash). §13.3이 제 일을 하는 것이다.
5. `cargo test -p es --test video showcase_scene_camera_follows_the_sensor_path`.
6. `cargo xtask verify-goldens` 0 changed; `cargo xtask ci`;
   `cargo xtask check-scope docs/packets/M7/R5-pt-observations.md`.

## acceptance

오라클 1–6(3은 RTX 3060과 서버에서). 노트의 비용 표, SSIM 숫자, row U4; PT로 수집된 한 에피소드의 시연
프레임을 같은 틱의 RS 프레임 옆에 contact-sheet PNG로 워크트리의 `target/plan-u/r5/` 아래 저장.
`training-u4.toml` 커밋(서버 경로, U0–U3의 것처럼).

## forbidden

커밋된 해시나 픽셀을 움직이는 것(오라클 1, 2, 6); CLI의 PT 노브(`--path`는 자유 카메라만의 것으로 남는다);
관측 경로에서의 누적(R4); 관측 IR이나 `ImageSpec` 변경; `crates/es-render/**`(R3에 필요한 것이 있다 —
없다면 멈추고 무엇이 없는지 말할 것); 고정된 문단을 넘는 어떤 `ARCHITECTURE` 수정도. INV-14: 내부
파라미터는 건드리지 않는다. INV-17: 새 트레이트 없음.
