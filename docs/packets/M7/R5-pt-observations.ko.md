# M7 R5 — 경로 추적된 관측: 센서가 자신의 렌더 경로를 선언하고, 정책이 그것으로 학습된다

스펙: §15.3("포토리얼리스틱 데이터셋과 골든을 위한 PT; RS와 PT 사이의 RGB는 SSIM
임계값이다"), §6 / §7.4(Task IR이 `ObservationSpec`을 *선언*한다: 소스와 타입을 가진 이름
붙은 채널들; 관측 IR이 전처리를 소유한다), §5.3(해시 체인 밖에는 아무것도 없다), §28.10
규칙 1(커밋된 픽셀은 결코 움직이지 않는다; 움직인 바이트는 문서 변경이다), §3.4(PT는
결정적이다: 카운터 RNG, 고정된 샘플 순서), §18.3. 오너 질문 2026-09-21: *PT 기반 학습이
없는 것인가?* — 없다, 왜냐하면 그것을 요청할 수 있는 문서가 없기 때문이다: `EnvRendererCfg`를
만드는 세 곳(`es loop collect --frames`, `es eval run --frames`, `es video showcase`)이
`RenderPath::Rs`를 하드코딩하고 있고, R3는 의도적으로 PT 손잡이들을 `EnvRendererCfg` 밖에
두었다. 설계 노트: `docs/design/renderer.md` 12절, `docs/design/ir-types.md`(Task IR의 센서
소스). **R3**(NEE, `Rgb8`로의 톤 맵), **R4**(선택적 누적은 관측에 쓰이지 않는다 — 모든
프레임은 독립적이다), **T6**(`observation-augmented.toml`이 커밋된 문서 옆에 두 번째 문서를
추가하는 방법을 보여준다), **U**(재측정을 위한 레시피 모양)에 의존한다.

## 질문

경로 추적기는 R3 이후로 `Rgb8`을 낼 수 있고 오라클 서버에서 64spp의 96×96 프레임에 대해
약 3ms가 든다. **Task IR 센서가 "나를 경로 추적기로 렌더링하라"고 말할 수 있어서, 수집,
평가, 쇼케이스가 모두 그 센서를 같은 방식으로 렌더링하고, 커밋된 문서들의 해시와 픽셀은
건드리지 않을 수 있는가 — 그리고 경로 추적된 관측으로 학습된 정책이 그것이 수집된 것과
같은 하니스를 통과하는가?**

## 사양

* **선언은 렌더러 설정이 만들어지는 곳에 산다: Task IR의 센서.**
  `es_ir::task::ObsSource::Sensor { id, format }`가 `#[serde(default)]`를 가진 `render:
  SensorRender`를 얻는다:
  ```
  SensorRender { path: Rs | Pt { spp: u32, bounces: u32 }, exposure: f32 (1.0), tonemap: Reinhard | Aces }
  ```
  **해시 규칙:** `canonical`은 `render` 블록이 기본값이 **아닐 때만** 그것을 쓴다 — 부재하거나
  기본값인 `render`는 오늘의 정규형과 바이트 단위로 동일하다, 그래서 커밋된 모든
  `task_hash`(그리고 그 위에서 해시되는 모든 것)는 움직이지 않는다; 테스트가 커밋된
  `task.toml`의 해시를 그 기록된 값에 대해 단언한다. `Pt` 센서는 `task_hash`를 움직이므로,
  자신만의 Evaluation IR 문서가 필요하다(§13.3: 새 비교) — 패킷은
  `tests/fixtures/visible-learning/task-pt.toml`(카메라 채널에 `render = { path = "pt", spp =
  64, bounces = 3 }`을 가진, 커밋된 `task.toml`)과 `evaluation-pt.toml`(`task-pt`의 해시를
  이름으로 부르는, 커밋된 `evaluation.toml`)을 추가하고, 두 해시 모두를 노트에 기록한다.
  관측 IR은 바뀌지 않는다: 센서가 *무엇인지*(해상도, 색 공간, 내부 파라미터)는
  `ImageSpec`의 것이다; 시뮬레이션이 그것을 *어떻게 만드는지*는 Task IR의 것이며, 이것이
  `ImageSpec` 필드가 아닌 이유다. 스펙 텍스트: §6의 `ObservationSpec`에 추가되는 한
  단락(한국어, 정본)과 그 영어 쌍둥이 — **패킷의 유일한 `ARCHITECTURE` 수정, 두 파일 모두
  한 커밋에** — 문구는 아래에 고정된다.
* **하나의 함수가 센서로부터 렌더러 설정을 만든다.** `es_env::render::sensor_cfg(camera,
  spec: &ImageSpec, render: &SensorRender, frames_dir) -> EnvRendererCfg`가 손으로 만든 세
  개의 `EnvRendererCfg::rgb(...)` 호출(`eval.rs::renderer_cfg`, `loop.rs`, 장면 카메라를
  위한 `showcase.rs`)을 대체하며, `Pt { spp, bounces }`를 톤 맵과 노출을 가진
  `RenderPath::Pt { spp, bounces, nee: true, restir: false, svgf: false }`로 매핑한다;
  `Rs`는 정확히 오늘의 설정으로 매핑된다(비트 단위로: 프레임 픽스처와 모든 RS 골든이
  그것을 고정한다). 따라서 `EnvRendererCfg`는 `exposure`/`tonemap`을 CLI 손잡이가 아니라
  **센서가 선언한 패스스루 필드**로 얻는다 — 관측 경로에 손잡이를 주지 않겠다는 R3의
  거부는 유지된다: *문서*가 결정한다. PT 프레임은 틱마다 독립적이다(누적 없음;
  `RenderConfig.seed` 고정, 샘플 키는 이미 픽셀마다 다르다), 그래서 프레임은 포즈의 순수
  함수다 — 수집기/평가기 패리티 오라클(T7)은 RS에서와 정확히 똑같이 PT에서도 성립한다.
* **비용은 측정되지, 가정되지 않는다.** `task-pt.toml`에서 `es loop collect --frames`, 8
  에피소드, RTX 3060과 4090에서: 관측 렌더의 ms/frame을 RS의 것 옆에(측정 전까지 `Target
  / Status: unverified`). PT로 수집된 실행의 `es video showcase`는 장면 카메라(`--camera`)에는
  센서의 경로를 쓰고 자유 카메라에는 `--path`를 유지한다.
* **재측정(서버).** PT 문서들에 대한 U3의 설정: PT 프레임으로 시연 200개 수집
  (`task-pt.toml`), `--for-training`으로 베이크, `learning-pretrained.toml` +
  `observation-augmented.toml`로 row-D 설정에서 20,000스텝 학습, `evaluation-pt.toml`의
  16개 홀드아웃 시드로 평가; `visible-learning.md` 7.31의 **row U4**로서 U3의 0.5625 옆에
  `success_rate`를 보고한다(새 `evaluation_hash`, 소리 내어 말해진다). 또한 같은 틱의 RS와
  PT 관측 사이의 SSIM(R3의 `ssim`)을, 32개 프레임에 대해, 관측 해상도에서의 첫 §15.3
  숫자로서.
* **스펙 문구(고정됨).** 한국어(`docs/ARCHITECTURE.ko.md`, §6 `ObservationSpec`):
  > **센서의 렌더 경로.** `Sensor` 소스는 `render = { path = "rs" | "pt", spp, bounces, exposure, tonemap }`로 시뮬레이션이 그 센서를 어떻게 만드는지 선언한다(패킷 M7/R5). 기본값은 `rs`이며 **부재 = 기본값 = 오늘의 정규형**이라 커밋된 `task_hash`는 움직이지 않는다; `pt`는 `task_hash`를 움직이므로 새 문서다(§13.3). 관측 IR은 이것을 모른다 — 센서가 *무엇*인지는 `ImageSpec`이, 시뮬레이션이 그것을 *어떻게* 만드는지는 Task IR이 말한다.

  영어(`docs/ARCHITECTURE.md`):
  > **A sensor's render path.** A `Sensor` source declares how the simulation produces it with `render = { path = "rs" | "pt", spp, bounces, exposure, tonemap }` (packet M7/R5). The default is `rs`, and **absent = default = today's canonical form**, so no committed `task_hash` moves; `pt` moves `task_hash` and is therefore a new document (§13.3). The Observation IR does not know about it — what the sensor *is* belongs to `ImageSpec`, how the simulation *makes* it belongs to the Task IR.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

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
tests/fixtures/visible-learning/evaluation-pt.toml
tests/fixtures/visible-learning/training-u4.toml
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

`es-ir/src/task.rs`(`SensorRender`, 조건부 정규형, 기본값), `serial.rs`는 **오직** TOML
모양이 serde 속성을 필요로 할 때만, `es-ir/tests`(해시-안정성 테스트),
`es-env/src/render.rs`(`sensor_cfg`, 두 패스스루 필드), `es-env/tests`, 세 CLI 호출 지점,
`cli.rs`/`video.rs`(PT 수집/평가/쇼케이스 테스트), 세 개의 새 픽스처, 스펙 문구(두 파일,
한 커밋), 세 설계 노트, 이 패킷.

## 오라클

1. `cargo test -p es-ir committed_task_hash_is_unmoved_by_sensor_render` — 커밋된
   `task.toml`은 그 기록된 `eb6efefa…`로 해시된다; 명시적인 기본값 `render`를 가진 같은
   문서는 동일하게 해시된다; `task-pt.toml`은 다르다.
2. `cargo test -p es-env sensor_cfg_rs_is_todays_config` — 기본값 `render`를 가진
   `sensor_cfg`는 `EnvRendererCfg::rgb(...)`와 필드마다 같고, 그 `RenderConfig`는 프레임
   픽스처의 첫 프레임을 비트 단위로 렌더링한다.
3. `cargo test -p es --test cli collect_renders_the_sensor_with_the_path_tracer` —
   `task-pt.toml`에서 `--frames`(GPU)를 가진 전문가 에피소드 1개: 프레임들은 `Rgb8`이고,
   검지 않으며, 같은 틱의 RS 렌더와 다르다; 두 번의 실행은 비트 단위로 동일하다(장치나
   `ES_PYTHON`이 없으면 `SKIP`).
4. `cargo test -p es --test cli eval_refuses_a_pt_policy_on_the_rs_document` — 해시 불일치가
   이름으로(태스크 해시) 거부되며, 이것은 §13.3가 제 일을 하는 것이다.
5. `cargo test -p es --test video showcase_scene_camera_follows_the_sensor_path`.
6. `cargo xtask verify-goldens` 0 변경; `cargo xtask ci`; `cargo xtask check-scope
   docs/packets/M7/R5-pt-observations.md`.

## 수용 기준

오라클 1–6(3은 RTX 3060과 서버에서). 노트 안의 비용 표, SSIM 숫자와 row U4; PT로 수집된
한 에피소드의 시연 프레임을 같은 틱들의 RS 프레임 옆, 워크트리의 `target/plan-u/r5/`
아래에 콘택트시트 PNG로 저장. `training-u4.toml` 커밋됨(서버 경로, U0–U3의 것처럼).

## 금지

커밋된 해시나 픽셀을 움직이는 것(오라클 1, 2, 6); CLI에 PT 손잡이(`--path`는 여전히 자유
카메라만의 것); 관측 경로에서의 누적(R4); 관측 IR이나 `ImageSpec`을 바꾸는 것;
`crates/es-render/**`(R3가 이것이 필요로 하는 것을 가지고 있다 — 그렇지 않다면, 멈추고
무엇인지 말하라); 고정된 문구를 넘어서는 그 어떤 `ARCHITECTURE` 수정. INV-14: 내부
파라미터 불변. INV-17: 새 트레이트 없음.
