<!-- Korean translation of docs/packets/M12/plan-y.md. The English file is the working copy; regenerate this when it changes. -->
# M12 plan Y — 에디터의 작업 흐름 셸(S1) — 구현 계획

> **에이전트 작업자용:** 필수 서브스킬: 이 계획을 작업 단위로 구현할 때는
> superpowers:subagent-driven-development(권장) 또는 superpowers:executing-plans을 쓴다. 단계는
> 추적을 위해 체크박스(`- [ ]`) 문법을 쓴다. 아래 각 작업은 §1.2의 작업 패킷이기도 하다:
> **Files** 블록이 패킷의 `context`이며, 그 밖의 것은 바뀔 수 없다.

**목표:** 기반 지식이 없는 사람이 에디터를 열어 큐브 템플릿을 고르고 버튼 하나를 누른 뒤,
신호등으로 실행을 지켜보고 쉬운 말로 된 결과를 읽는다 — 터미널도, 레시피 경로도, TOML 파일도
없이.

**아키텍처:** egui는 그대로 두고, `egui_dock`이 고정된 단계 바 아래 Unity식 도킹을 더한다.
모든 결정은 headless로 테스트되는 `model/` 파일에 있고, `ui/`는 그리기만 한다. 에디터는 여전히
학습을 직접 돌리지 않는다: 실행 레시피를 프로젝트 폴더에 쓰고 `es loop cycle`을 자식
프로세스로 실행해 붙는다. 작은 백엔드 변경 네 가지가 이를 뒷받침한다: `--check-deps --json`,
`episodes.json`, `es policy init`, 그리고 큐브 템플릿의 비전 경로 cycle 레시피.

**기술 스택:** Rust 1.85, egui/eframe 0.32.3, `egui_dock`(신규, 버전은 Y10에서 고정), `rfd`
0.17.2(이미 있음), serde/serde_json/toml, 워크스페이스 크레이트 `es-data`, `es-eval`,
`es-render`, `es-ir`, `es-telemetry`.

**스펙:** `docs/design/editor-redesign.md`(§6이 S1이다). 이 계획과 함께 읽는다; 둘이 다르면
설계 노트가 우선이고, 그 차이는 계획의 버그로 보고한다.

## 전역 제약

- MSRV `rust-version = "1.85"`; `egui`/`eframe`는 `0.32.3`을 유지한다. 유일한 새 의존성은
  `egui_dock`(Y10)이다. 오케스트레이터 없이는 그 외 아무것도 더하지 않는다.
- 계층(§4.2): `es-editor`는 새로 `es-data`(계층 10)에 의존할 수 있다. 같은 계층 의존은 없고,
  `es-editor`에 의존하는 것도 없다. `cargo xtask` 계층 검사는 계속 통과해야 한다.
- **그리는 코드에서는 아무것도 결정하지 않는다**(`editor-shell.md` §2): 무엇을 보여 줄지 고르는
  모든 분기는 `crates/es-editor/src/model/`에 있고 `cargo test -p es-editor`로 판정한다.
- **눈에 보이는 모든 단어**는 `crates/es-editor/i18n/en.toml`과 `ko.toml`을 거친다
  (`editor-shell.md` §15): 두 파일에 같은 키가 있고, 모든 키가 쓰이며, `spec `을 담은
  `*.hint` 밖에는 값이 없다. 한국어 텍스트는 그 외 어디에도 없다 — 테스트에도, 주석에도.
- **레이블 매핑은 와일드카드 가지 없이 전체를 덮는다** — `FailureKind`, `ViolationKind`,
  `PerturbationKind`, `es_env::Termination`, 그리고 레이블을 뽑아내는 모든 새 열거형에 대해.
- `es-ir`은 S1에서 바꾸지 않는다. 안전 평면은 손대지 않는다(INV-11..13); 그것 없이 도는 경로는
  없다(INV-12). 새 트레이트도 없다(INV-17).
- Golden은 읽기 전용이다(§1.4): golden은 정해진 생성기를 거쳐야만 바뀌고 손으로는 절대
  고치지 않으며, 코드를 옮기는 작업은 기존 golden을 모두 바이트 단위로 그대로 둬야 한다.
- Conventional Commits, 영어, **트레일러 없음**; 훅 활성화(`core.hooksPath=.githooks`).
- 워크트리 에이전트: 먼저 `git merge --ff-only main`; **절대 push하지 않음**; 브랜치를
  보고한다.
- 메인 트리에서 빌드할 때는 `CARGO_TARGET_DIR=target/alt`를 쓴다(소유자가 띄워 둔
  `es-editor.exe`가 `target/`를 잠근다).
- 작업마다: `cargo fmt --all -- --check`, `cargo clippy -p <크레이트> --all-targets -- -D
  warnings`, 건드린 테스트, `cargo xtask check-scope`. 오케스트레이터는 병합된 웨이브마다
  `cargo xtask ci`를 한 번 돌린다.
- 이 기기에는 `ES_PYTHON`이 없다: Python이 필요한 모든 오라클은 기존 것과 똑같이 **이유를
  출력하고 건너뛰어야** 한다. 실제 실행은 나중에 Y-V에서 모은다.

## 리뷰 초점

1. **경로에 한글과 공백이 있는 프로젝트 폴더** (`C:\Users\User\문서\내 로봇`): 실행 레시피,
   `--out`, `--recipe`, 자식 프로세스의 작업 디렉터리가 이를 그대로 가지고 다닌다.
   테스트: Y6 `run_recipe_and_argv_keep_a_korean_path_with_spaces`.
2. **시작을 두 번 누르거나 실행 중에 누름**: 자식 하나, 실행 폴더 하나. 테스트: Y6
   `next_run_dir_never_reuses_a_number`와 `start_in_while_running_is_a_no_op`.
3. **실행(또는 에디터)이 죽은 뒤 에디터를 다시 염**: `telemetry.txt`는 있지만 아무도 응답하지
   않는다 — 실행은 재개 지점이 있는 *중단됨*이지 결코 *실행 중*이 아니다. 테스트: Y6
   `a_dead_address_reads_as_interrupted_with_the_next_stage`.
4. **점이 없는 곡선, NaN인 첫 점, 점 하나뿐인 곡선; 사람이 멈춘 실행**: 신호등은 결코
   패닉하지 않고, 데이터가 오기 전에 초록을 보이지 않으며, 강제 종료에는 *멈췄어요*가 아니라
   *당신이 멈췄어요*라고 말한다. 테스트: Y7 `edge_series_never_panic_and_never_start_green`,
   `a_kill_is_stopped_by_you`.
5. **`episodes.json`이 없는 옛 실행; `evaluation_hash`가 다른 지난 실행**: 앞의 경우는
   타일을 보이지 않고 막대를 `report.json`으로 그리며, 뒤의 경우는 숫자 자체를 보이지 않는다. 테스트: Y8
   `old_run_falls_back_to_suite_metrics`와 `different_evaluation_hash_is_not_comparable`.

---

## 웨이브

| 웨이브 | 작업(같은 웨이브 안에서는 병렬) | 필요 |
|---|---|---|
| 1 | Y1 `--check-deps --json` · Y2 `episodes.json` · Y3 래스터 → `es-render` · Y4 실행 읽기 → `es-eval` · Y5 큐브 템플릿 + `es policy init` | — |
| 2 | Y6 프로젝트 + 작업 흐름 · Y7 신호등 · Y8 결과 모델 · Y9 시작 화면 모델 | 웨이브 1 |
| 3 | Y10 도킹 셸 | 웨이브 2 |
| 4 | Y11 시작 화면 · Y12 ③/④ 진행 · Y13 ⑤ 결과 | Y10 |
| 5 | Y14 문서와 Y-V 체크리스트 | 웨이브 4 |
| 나중 | **Y-V** 실제 실행(로컬 venv 또는 서버), 임계값 보정 | 소유자의 승인 |

---

### 작업 Y1: `es --check-deps --json`

**파일:**
- 수정: `crates/es/src/cmd/check_deps.rs`
- 수정: `crates/es/src/main.rs`(`--check-deps` 가지만)
- 테스트: `crates/es/tests/cli.rs`(`check_deps_always_exits_zero` 옆)
- 문서: 없음(`TOP_HELP`의 도움말 줄에 `[--json]`이 추가된다)

**인터페이스:**
- 출력(stdout, 한 줄, 항상 exit 0):

```json
{"schema":1,
 "python":{"found":true,"path":"C:/…/python.exe"},
 "modules":{"mujoco":true,"torch":true,"lerobot":false},
 "vulkan_loader":true,
 "render":false,
 "backends":[{"name":"mujoco-cpu","available":true},
             {"name":"mjwarp","available":false,"reason":"…"}]}
```

`found`가 false이면 `python.path`는 없다. `render`는 `es` 바이너리 자신의
`cfg!(feature = "render")`다 — `--frames`에 필요하고(`crates/es/src/cmd/eval.rs`), 큐브
템플릿이 프레임을 기록하므로 시작 화면이 이를 알아야 한다.

- [ ] **단계 1: 실패하는 테스트 작성**, `crates/es/tests/cli.rs`에:

```rust
#[test]
fn check_deps_json_is_one_object_with_every_field() {
    let out = bin().args(["--check-deps", "--json"]).output().expect("run es");
    assert_eq!(out.status.code(), Some(0));
    let text = stdout(&out);
    let v: serde_json::Value = serde_json::from_str(text.trim()).expect("one JSON object");
    assert_eq!(v["schema"], 1);
    assert!(v["python"]["found"].is_boolean());
    for m in ["mujoco", "torch", "lerobot"] {
        assert!(v["modules"][m].is_boolean(), "modules.{m}");
    }
    assert!(v["vulkan_loader"].is_boolean());
    assert!(v["render"].is_boolean());
    let names: Vec<&str> = v["backends"].as_array().unwrap().iter()
        .map(|b| b["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"mujoco-cpu"));
    for b in v["backends"].as_array().unwrap() {
        if b["available"] == false {
            assert!(b["reason"].is_string(), "an unavailable backend says why");
        }
    }
}
```

- [ ] **단계 2: 돌려서 FAIL을 확인** —
  `cargo test -p es --test cli check_deps_json_is_one_object_with_every_field`(지금 텍스트는
  JSON이 아니다).
- [ ] **단계 3: 구현.** 사실들을 한 번에 비공개 `struct Deps`(python, mujoco, torch, lerobot,
  vulkan, render, backends: `Vec<(String, Result<(), String>)>`)로 모은 뒤,
  `run(json: bool)`이 오늘의 텍스트(바이트 단위로 그대로)나
  `serde_json::to_string(&serde_json::json!({...}))` 중 하나를 출력한다. `main.rs`:
  `Some("--check-deps") => Ok(cmd::check_deps::run(args.iter().any(|a| a == "--json")))`.
- [ ] **단계 4: 두 테스트 모두 실행**; PASS를 확인(텍스트 테스트가 텍스트가 그대로임을
  증명한다).
- [ ] **단계 5: 커밋** `feat(es): --check-deps --json for the editor's start screen`.

**수용 기준:** 두 테스트 모두 통과; 텍스트 출력은 그대로. **금지:** 텍스트 형식 변경;
GPU 드라이버 탐색(Vulkan 점검은 파일 존재 휴리스틱으로 남는다).

---

### 작업 Y2: `episodes.json` — 평가된 에피소드마다 한 행

**파일:**
- 생성: `crates/es-eval/src/episodes.rs`
- 수정: `crates/es-eval/src/lib.rs`(모듈 + 재노출)
- 수정: `crates/es-eval/src/runner.rs`(`cell_name` 추출; `resolve_seeds`를
  `pub(crate)`로; 동작 변화 없음)
- 수정: `crates/es/src/cmd/eval.rs`(`Evaluation::merge` 뒤에 파일을 쓴다)
- 테스트: `episodes.rs`의 모듈 내 테스트; `crates/es/tests/cli.rs`(Python 게이트 상호 검사)

**인터페이스:**
- 입력: `runner::{Shard, ShardCell}`, `metrics::CellSummary`(`histogram`, `steps`,
  `dirty_steps`), `es_ir::evaluation::EvaluationIr`.
- 출력:

```rust
pub const EPISODES_FILE: &str = "episodes.json";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeRow {
    /// `EvaluationIr::suites[cell].name`.
    pub suite: String,
    /// `<suite>-<NN>`: the key of `frames/<cell>/`, `traj/<cell>.estraj`, `events.json`.
    pub cell: String,
    /// Index into the resolved seed list (§10.2).
    pub episode: u64,
    pub seed: u64,
    /// `success` | `failure` | `timeout` | `unfinished` — the one termination bucket.
    pub termination: String,
    /// Steps the Safety Plane validated, and how many of them it changed.
    pub steps: u64,
    pub changed_steps: u64,
    /// This episode's own §10.3 `failure_mode_histogram` buckets.
    pub histogram: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeIndex {
    pub schema_version: u32, // 1
    pub episodes: Vec<EpisodeRow>,
}

/// Rows in `(cell, episode)` order, whatever order the shards arrive in.
pub fn episode_rows(ir: &EvaluationIr, shards: &[Shard]) -> Vec<EpisodeRow>;
pub fn write_episodes(rows: &[EpisodeRow], dir: &Path) -> Result<(), EvalError>;
/// `Ok(None)` when the file is absent (a run from before this packet).
pub fn read_episodes(dir: &Path) -> Result<Option<Vec<EpisodeRow>>, EvalError>;
```

그리고 `runner.rs`에는: `pub fn cell_name(suite: &str, episode: u64) -> String`
(`runner.rs:710`의 `format!("{}-{idx:02}", suite.name)`을, 이제 두 곳에서 호출한다 — 먼저
거기서 `idx`가 에피소드 인덱스임을 확인하고 커밋 메시지에 적을 것).

- [ ] **단계 1: 실패하는 테스트 작성**, `episodes.rs`에:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn summary(termination: &str, steps: u64, changed: u64) -> metrics::CellSummary {
        let mut s = metrics::CellSummary::default();
        s.histogram.insert(termination.to_owned(), 1);
        s.steps = steps;
        s.dirty_steps = changed;
        s.n_episodes = 1;
        s
    }

    fn unit(cell: u32, episode: u64, termination: &str) -> ShardCell {
        ShardCell { cell, episode, summary: summary(termination, 100, 3) }
    }

    #[test]
    fn rows_come_out_in_cell_episode_order_with_names_and_seeds() {
        let ir = two_suite_ir(); // suites "nominal", "light_intensity"; seeds base 101, 2 episodes
        let a = Shard { cells: vec![unit(1, 0, "failure"), unit(0, 1, "success")], ..Default::default() };
        let b = Shard { cells: vec![unit(0, 0, "timeout"), unit(1, 1, "success")], ..Default::default() };
        let rows = episode_rows(&ir, &[a, b]);
        let key: Vec<_> = rows.iter().map(|r| (r.cell.as_str(), r.seed, r.termination.as_str())).collect();
        assert_eq!(key, [
            ("nominal-00", 101, "timeout"), ("nominal-01", 102, "success"),
            ("light_intensity-00", 101, "failure"), ("light_intensity-01", 102, "success"),
        ]);
        assert_eq!(rows[0].changed_steps, 3);
    }

    #[test]
    fn the_file_round_trips_and_an_absent_file_is_none() {
        let dir = std::env::temp_dir().join(format!("es-episodes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(read_episodes(&dir).unwrap(), None);
        let rows = episode_rows(&two_suite_ir(), &[Shard { cells: vec![unit(0, 0, "success")], ..Default::default() }]);
        write_episodes(&rows, &dir).unwrap();
        assert_eq!(read_episodes(&dir).unwrap(), Some(rows));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
```

`two_suite_ir()`는 `runner.rs` 자신의 테스트가 하는 방식으로 최소한의 `EvaluationIr`을
만든다(그 헬퍼가 있으면 재사용하고, 없으면 `SeedPlan::Base(101)`과 `n_episodes = 2`로
만든다). `CellSummary`에 `Default`가 없으면 derive한다(그 타입에는 다른 변경을 하지 않는다).

- [ ] **단계 2: 실행** `cargo test -p es-eval episodes`; FAIL을 확인(모듈이 없다).
- [ ] **단계 3: 구현.** `episode_rows`(펼치고 `(cell, episode)`로 정렬, suite 이름은
  `ir.suites[cell]`에서, seed는 `resolve_seeds(ir)[episode]`에서, termination = 히스토그램에
  있는 `success|failure|timeout|unfinished` 중 하나의 키), writer(pretty JSON + 끝 줄바꿈,
  `write_artifacts`처럼), reader. `crates/es/src/cmd/eval.rs`에서
  `es_eval::write_artifacts(...)` 바로 뒤에:
  `es_eval::episodes::write_episodes(&es_eval::episodes::episode_rows(&eval_ir, &shards), &a.out)`.
  shard-worker 가지(`--shard-out`)는 이보다 먼저 반환하며 새로 쓰는 것이 없다.
- [ ] **단계 4: Python 게이트 상호 검사 추가**, `crates/es/tests/cli.rs`에, Python 없이는
  이미 건너뛰는 기존 `es eval run` 테스트 옆에: 실행 뒤, suite마다
  `count(termination == "success") / n`이 그 suite의 report `success_rate` 값과 **정확히**
  같고, 행 개수가 `n_episodes × suite 수`와 같다. 독립적으로 계산된 두 숫자가 서로를
  검사한다.
- [ ] **단계 5: 실행** `cargo test -p es-eval`와 `cargo test -p es --test cli eval_run`(새
  테스트는 여기서 건너뛴 이유를 출력한다); PASS / SKIP을 확인.
- [ ] **단계 6: 커밋** `feat(es-eval): episodes.json, one row per evaluated episode`.

**수용 기준:** 테스트 통과; `report.json`, `evaluation.lock`, `events.json`은 바이트 단위로
그대로(기존 golden이나 테스트 변경 없음). **금지:** `report.json`에 필드 추가(커밋된 모든
report가 움직인다); 여기서 어떤 지표든 계산하는 것.

---

### 작업 Y3: 재생 래스터라이저가 `es-render`로 옮겨 간다

**파일:**
- 생성: `crates/es-render/src/raster.rs`
- 수정: `crates/es-render/src/lib.rs`(`pub mod raster;`)
- 수정: `crates/es-editor/src/model/replay_view.rs`
- 수정: `crates/es-editor/src/app.rs`(import만)

**인터페이스:**
- 동작 변화 없이 `replay_view.rs`에서 `es_render::raster`로 옮긴다: `Camera`(`view`, `orbit`,
  `zoom`, `PITCH_LIMIT`와 함께), `Tri2d`, `Projected`, `BACKGROUND`, `MAX_PIXELS`,
  `Raster`(`size_for`, `draw`와 함께), `project_scene`, `shading`, `flat_colour`, `edge`,
  `span`, `dot`, `vec3`, `quat_from_basis`, 그리고 이들이 내는 오류 variant(`es-render`의
  `RasterError`; `ReplayError`가 이를 감싼다).
- `replay_view.rs`에 남는다(`es_env::traj`, 계층 9가 필요하다): `ReplayView`, `ReplayError`,
  `panel_height`와 패널 상수들, `load_scene`.
- `flat_colour`는 golden이 바이트 단위로 그대로 남는 경우에 **한해서만**
  `es_render::cpu::shade_lambert`를 다시 구현하지 않고 직접 호출해야 한다; 그렇지 않으면
  복사본을 유지하고 차이를 밝히는 `ponytail:` 주석을 달아 보고한다.

- [ ] **단계 1:** `main`에서 `cargo test -p es-editor replay`를 실행; 통과하는 테스트 목록을
  기록한다.
- [ ] **단계 2:** 코드와 단위 테스트를 옮긴다(래스터라이저만 필요한 테스트는 함께 옮기고,
  `.estraj`를 여는 테스트는 남긴다).
- [ ] **단계 3:** `cargo test -p es-render raster`, `cargo test -p es-editor replay`,
  `cargo xtask verify-goldens`를 실행; 같은 통과 목록과
  `tests/golden/editor/replay-tick0-320x180.bin`, `.json`, `replay_tick0_order.json`이
  그대로임을 확인.
- [ ] **단계 4: 커밋** `refactor(es-render): the editor's CPU replay rasterizer moves to es-render`.

**수용 기준:** golden이 바이트 단위로 그대로; 동작 변화 없음. **금지:** 픽셀, 삼각형 순서,
카메라 수식의 어떤 변경도; 에디터 안의 재노출 shim.

---

### 작업 Y4: 끝난 실행 읽기가 `es-eval`로 옮겨 간다

**파일:**
- 생성: `crates/es-eval/src/run_dir.rs`
- 수정: `crates/es-eval/src/lib.rs`
- 수정: `crates/es-editor/src/model/run_view.rs`, `image_view.rs`, `live_run.rs`,
  `train_view.rs`, `recent.rs`, `crates/es-editor/src/app.rs`(import와 호출부만)

**인터페이스:**
- `es_eval::run_dir`로 옮긴다: `RunError`, `CellRow`, `TickRow`, `FirstSeen`, `KindRow`(데이터만),
  `Timeline`(데이터 + `kind_rows`, `buckets`), `Bucket`, `severity`, `decode_events`,
  `RunView`(**`RunDir`**로 이름 변경; `is_run_dir`, `open`, `set_frames_root`, `frames_root`,
  `cells`, `acceptance`, `sort_by`, `selected_cell`, `select`, `traj_path`, `timeline`,
  `filmstrip`, `frame`)와 비공개 헬퍼들.
- `es_eval::run_dir`로 옮긴다: **`Rgb8Image`**(`{ width, height, data }`), `es-eval`이 쓰는
  프레임의 메모리 표현. 에디터는 거기서 가져다 쓴다.
- 사람이 읽는 것이므로 `es-editor`에 자유 함수로 남는다: `Timeline::heading` →
  `run_view::timeline_heading(&Timeline, cell)`, `KindRow::label` →
  `run_view::kind_label(&KindRow)`, `RunView::columns` → `run_view::columns(&RunDir)`.
  `run_view.rs`는 이것들과 단어를 그리는 나머지만 남긴다.

- [ ] **단계 1:** `cargo test -p es-editor run_view live_run recent`를 실행; 목록을 기록한다.
- [ ] **단계 2:** 코드와, 에디터 타입이 필요 없는 테스트를 옮긴다.
- [ ] **단계 3:** `cargo test -p es-eval run_dir`, `cargo test -p es-editor`,
  `cargo xtask verify-goldens`, `cargo xtask layering`을 실행; PASS와 golden 무변경을 확인.
- [ ] **단계 4: 커밋** `refactor(es-eval): reading a finished run moves from the editor to es-eval`.

**수용 기준:** 같은 테스트가 통과; 계층 검사 green(`es-eval`은 새 크레이트 의존을 얻지
않는다). **금지:** 동작 변경; 사람이 읽는 문자열을 `es-eval`로 옮기는 것.

---

### 작업 Y5: 큐브 템플릿, `es policy init`, 템플릿 리더

**파일:**
- 생성: `templates/cube-into-bin.toml`
- 생성: `tests/fixtures/visible-learning/cycle-vision.toml`
- 생성: `tests/golden/train/plan-cycle-vision.txt`(단계 4의 생성기를 거쳐서만)
- 수정: `crates/es/src/cmd/policy.rs`(`init` 서브커맨드)
- 수정: `crates/es-data/src/training.rs`(CLI와 에디터가 공유하는 라이브러리 함수
  `untrained_bundle` — `es-compile`의 `PolicyBundle::build`와 기존 가중치 초기화를 둘
  다에서 직접 호출할 수 없다면)
- 생성: `crates/es-editor/src/model/template.rs`; 수정: `crates/es-editor/src/model/mod.rs`,
  `crates/es-editor/Cargo.toml`(`es-data` 의존성)
- 테스트: `crates/es/tests/cli.rs`; `template.rs`의 모듈 내 테스트

**인터페이스:**
- 출력, CLI:
  `es policy init --task T --observation O --learning L --deployment D [--seed N] --out X.esb`
  — 같은 문서와 시드에 대해 결정적인 바이트; 다른 `policy` 서브커맨드처럼 exit 0/1/2. 이것이
  필요한 이유는, 새로 받은 저장소는 cycle의 `[collect] policy`가 가리키는 번들을 만들 방법이
  없기 때문이다(그 Deployment IR이 시연기가 그 아래서 도는 평면이다).
- 출력, 라이브러리(에디터가 호출 가능; `es`와 `es-editor` 둘 다 닿을 수 있는 곳에 둔다):

```rust
pub fn untrained_bundle(task: &Path, observation: &Path, learning: &Path,
                        deployment: &Path, seed: u64) -> Result<Vec<u8>, DataError>;
```

- 출력, 템플릿 파일(`templates/cube-into-bin.toml`):

```toml
kind = "template"
id = "cube-into-bin"
name = "template.cube_into_bin.name"        # es-editor i18n key
summary = "template.cube_into_bin.summary"  # es-editor i18n key
robot = "SO-101"
method = "blocks"
needs = ["mujoco", "torch", "lerobot", "vulkan", "render"]
cycle = "tests/fixtures/visible-learning/cycle-vision.toml"
scene = "tests/fixtures/mjcf/so101_pick_place.xml"
demonstrations = 200

# The bundle `es policy init` builds for [collect] policy.
[bundle]
task        = "tests/fixtures/visible-learning/task.toml"
observation = "tests/fixtures/visible-learning/observation.toml"
learning    = "tests/fixtures/visible-learning/learning.toml"
deployment  = "tests/fixtures/visible-learning/deployment.toml"

# Training length presets: the `[run] checkpoint_at` series; the last mark is the length.
[lengths]
short  = [VALUES CHOSEN IN STEP 2]
medium = [VALUES CHOSEN IN STEP 2]
long   = [VALUES CHOSEN IN STEP 2]
```

- 출력, `model/template.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method { Blocks }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Length { Short, Medium, Long }

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Template { pub kind: String, pub id: String, pub name: String, pub summary: String,
    pub robot: String, pub method: Method, pub needs: Vec<String>, pub cycle: String,
    pub scene: String, pub demonstrations: u32, pub bundle: BundleDocs, pub lengths: Lengths }

impl Template {
    /// The `checkpoint_at` series of a preset; its last element is the run's length.
    pub fn marks(&self, length: Length) -> &[u32];
}

/// The directory holding `templates/`: the first ancestor of `exe_dir` that contains both
/// `templates/` and `Cargo.toml`. Pure, for the test; `templates_root()` calls it with
/// `std::env::current_exe()`'s parent.
pub fn find_root(exe_dir: Option<&Path>) -> Option<PathBuf>;
pub fn templates_root() -> Option<PathBuf>;

/// Every `templates/*.toml` that parses, sorted by `id`, and every one that does not, with why.
pub fn load(root: &Path) -> (Vec<Template>, Vec<(PathBuf, String)>);
```

`find_root`의 `ponytail:` 주석이 한계를 밝힌다: 템플릿은 저장소를 받아 둔 상태에서만 찾을 수
있다(설치형 패키지는 나중 항목이다).

- [ ] **단계 1: 열린 사실 세 가지를 확인해 커밋 메시지에 적는다:**
  (a) V19b 비전 경로가 어떻게 수집됐는지 — collect 번들의 문서(위 `[bundle]` 블록은 IR
  경로 문서를 가정한다, 정책이 `--expert` 아래서는 전혀 행동하지 않으므로;
  `docs/design/visible-learning.md` / `training-recipe.md`에서 확인); (b) `es-data`의 기존
  `init_weights`(`crates/es-data/src/training.rs:1166`)나 `es-compile`이 이미 가중치를
  결정적으로 초기화하는지 — 그렇다면 재사용하고 새로 쓰지 않는다; (c) `[run] steps`와
  `checkpoint_at`에 대한 LeRobot 경로의 규칙(`training.rs:467`, `:757`: `lerobot-train`은
  하나의 `--save_freq`로 저장하므로, 마크는 배수 수열이어야 한다).
- [ ] **단계 2: `cycle-vision.toml` 작성**(커밋된 `cycle.toml`에 `[train] recipe =
  "tests/fixtures/visible-learning/training-lerobot.toml"`, `[eval] config =
  "tests/fixtures/visible-learning/evaluation-v8.toml"`를 넣고, 이유를 적은 헤더 주석을
  단다) 하고, 세 길이 프리셋을 `training.rs`가 받아들이는 배수 수열로 고른다. `medium`은
  `training-lerobot.toml` 자신의 마크와 같게 한다.
- [ ] **단계 3: 실패하는 테스트.** `crates/es/tests/cli.rs`:

```rust
#[test]
fn policy_init_is_deterministic_and_the_bundle_opens() {
    let dir = scratch_dir("policy-init");
    let args = |out: &str| vec![
        "policy".to_owned(), "init".to_owned(),
        "--task".to_owned(), "tests/fixtures/visible-learning/task.toml".to_owned(),
        "--observation".to_owned(), "tests/fixtures/visible-learning/observation.toml".to_owned(),
        "--learning".to_owned(), "tests/fixtures/visible-learning/learning.toml".to_owned(),
        "--deployment".to_owned(), "tests/fixtures/visible-learning/deployment.toml".to_owned(),
        "--out".to_owned(), dir.join(out).display().to_string(),
    ];
    for out in ["a.esb", "b.esb"] {
        let o = bin().current_dir(repo_root()).args(args(out)).output().unwrap();
        assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    }
    let (a, b) = (std::fs::read(dir.join("a.esb")).unwrap(), std::fs::read(dir.join("b.esb")).unwrap());
    assert_eq!(a, b);
    es_compile::PolicyBundle::open(&a).expect("the bundle opens");
}

#[test]
fn cycle_vision_dry_run_matches_its_golden() {
    let o = bin().current_dir(repo_root())
        .args(["loop", "cycle", "--recipe", "tests/fixtures/visible-learning/cycle-vision.toml",
               "--out", "runs/y5-dry", "--dry-run"])
        .output().unwrap();
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert_golden("tests/golden/train/plan-cycle-vision.txt", &stdout(&o));
}
```

  저장소 루트, golden, 생성(`ES_GENERATE_GOLDENS=1`)을 위해 파일에 이미 있는 헬퍼를,
  `plan-cycle.txt`의 테스트가 하는 그대로 쓴다.
  `template.rs`:

```rust
#[test]
fn the_committed_templates_parse_and_name_real_files() {
    let root = find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).expect("a checkout");
    let (ok, bad) = load(&root);
    assert!(bad.is_empty(), "{bad:?}");
    let cube = ok.iter().find(|t| t.id == "cube-into-bin").expect("the cube template");
    for p in [&cube.cycle, &cube.scene, &cube.bundle.task, &cube.bundle.observation,
              &cube.bundle.learning, &cube.bundle.deployment] {
        assert!(root.join(p).is_file(), "{p}");
    }
    for l in [Length::Short, Length::Medium, Length::Long] {
        let m = cube.marks(l);
        assert!(!m.is_empty() && m.windows(2).all(|w| w[0] < w[1]), "{l:?}: {m:?}");
    }
    assert!(cube.marks(Length::Short).last() < cube.marks(Length::Long).last());
}

#[test]
fn find_root_walks_up_and_gives_up_cleanly() {
    let tmp = std::env::temp_dir().join(format!("es-tpl-{}", std::process::id()));
    let deep = tmp.join("target").join("debug");
    std::fs::create_dir_all(&deep).unwrap();
    assert_eq!(find_root(Some(&deep)), None);
    std::fs::create_dir_all(tmp.join("templates")).unwrap();
    std::fs::write(tmp.join("Cargo.toml"), "").unwrap();
    assert_eq!(find_root(Some(&deep)), Some(tmp.clone()));
    assert_eq!(find_root(None), None);
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn a_broken_template_is_reported_not_fatal() {
    let tmp = std::env::temp_dir().join(format!("es-tpl-bad-{}", std::process::id()));
    std::fs::create_dir_all(tmp.join("templates")).unwrap();
    std::fs::write(tmp.join("templates").join("x.toml"), "kind = \"template\"\n").unwrap();
    let (ok, bad) = load(&tmp);
    assert!(ok.is_empty());
    assert_eq!(bad.len(), 1);
    std::fs::remove_dir_all(&tmp).unwrap();
}
```

- [ ] **단계 4:** 실행; FAIL을 확인. `init`, `untrained_bundle`, `template.rs`를 구현한다;
  golden을 `ES_GENERATE_GOLDENS=1`로 한 번 생성하고 커밋 전에 읽어 본다(collect →
  expert-gate → train(LeRobot 경로) → eval(`evaluation-v8.toml`) → showcase 순서가 보여야
  한다).
- [ ] **단계 5:** 새 테스트와 `cargo test -p es --test cli plan_cycle`을 실행; PASS를 확인.
- [ ] **단계 6: 커밋** `feat: the cube template, es policy init, and the editor's template reader`.

**수용 기준:** 위 테스트들; golden은 오케스트레이터가 검토한다. **금지:** 커밋된 문서를
복사하는 것(그 경로는 해시 입력이다); `cycle.toml`, `training-lerobot.toml`이나 커밋된 IR
문서를 바꾸는 것; 커밋된 바이너리 번들.

---

### 작업 Y6: 프로젝트 폴더, 실행 레시피, 단계 바의 상태

**파일:**
- 생성: `crates/es-editor/src/model/project.rs`, `crates/es-editor/src/model/workflow.rs`
- 수정: `crates/es-editor/src/model/mod.rs`, `model/launch.rs`(`start_in`,
  `free_local_port`), `model/recent.rs`(`Kind::Project`, `classify`)
- 수정: `crates/es-editor/i18n/en.toml`, `ko.toml`(`phase.*`, `run.*` 키)

**인터페이스:**
- 입력: `template::{Template, Length}`(Y5), `es_data::training::Cycle`(기존,
  `Serialize`/`Deserialize`), `es_data::collect::{read_loop_steps, LoopKind}`,
  `es_data::untrained_bundle`(Y5), `live_run::StageRow`(기존).
- 출력, `project.rs`:

```rust
pub const PROJECT_FILE: &str = "project.toml";
pub const RUNS_DIR: &str = "runs";
pub const RUN_RECIPE: &str = "cycle.toml";
pub const TELEMETRY_FILE: &str = "telemetry.txt";
pub const COLLECT_BUNDLE: &str = "untrained.esb";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectFile { pub kind: String, pub name: String, pub template: String }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Project { pub root: PathBuf, pub file: ProjectFile }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunFolder { pub number: u32, pub path: PathBuf }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartSettings { pub demonstrations: u32, pub length: Length }

impl Project {
    /// Makes `root` (refusing one that already holds a `project.toml`), writes `project.toml`
    /// and builds `untrained.esb` from the template's `[bundle]` documents.
    pub fn create(root: &Path, name: &str, template: &Template, repo_root: &Path)
        -> Result<Self, ProjectError>;
    pub fn open(root: &Path) -> Result<Self, ProjectError>;
    pub fn is_project_dir(path: &Path) -> bool;
    /// `runs/NNN` directories, ascending; anything else under `runs/` is ignored.
    pub fn runs(&self) -> Vec<RunFolder>;
    pub fn latest_run(&self) -> Option<RunFolder>;
    /// One past the largest existing number, three digits; never an existing directory.
    pub fn next_run_dir(&self) -> PathBuf;
}

impl RunFolder {
    pub fn eval_dir(&self) -> PathBuf;          // path/eval
    pub fn report_path(&self) -> PathBuf;       // path/eval/report.json
    pub fn telemetry_addr(&self) -> Option<String>;
}

/// Writes `run/cycle.toml`: the template's recipe with `[collect] episodes`, `[collect]
/// policy` (absolute path of the project's bundle) and an inline `[train]` whose
/// `[run] checkpoint_at` is the preset and `steps` its last mark. Everything else is the
/// template's, unchanged. Returns the argv for `es` (no program name).
pub fn write_run(template: &Template, repo_root: &Path, project: &Project,
                 settings: StartSettings, run: &Path, telemetry: &str)
    -> Result<Vec<String>, ProjectError>;

/// `--from <stage>` argv for resuming `run` (same recipe, same `--out`).
pub fn resume_argv(run: &RunFolder, from: &str, telemetry: &str) -> Vec<String>;
```

- 출력, `launch.rs`:

```rust
impl LaunchModel {
    /// Starts `es` with `args` in `cwd` (the template's repository root), exactly like
    /// `start_program` otherwise; a no-op while a child runs; attach-follows-launch reads the
    /// `--telemetry` value out of `args`.
    pub fn start_in(&mut self, args: &[String], cwd: &Path);
}
/// A port the OS says is free on 127.0.0.1 now (bind `:0`, read it, drop).
pub fn free_local_port() -> u16;
```

- 출력, `workflow.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase { Scene, Teach, Train, Evaluate, Results }
impl Phase { pub const ALL: [Phase; 5]; pub fn key(self) -> &'static str; }

#[derive(Clone, Debug, PartialEq)]
pub enum PhaseState {
    Done,
    NotStarted,
    Locked,
    Running { stage: String, fraction: Option<f32> },
    Failed { stage: String, code: Option<i32> },
    Interrupted { resume_from: Option<String> },
    StoppedByYou { resume_from: Option<String> },
}

/// What disk says about one run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunFacts { pub collected: bool, pub trained: bool, pub evaluated: bool,
                      pub report: bool }
impl RunFacts { pub fn read(run: &RunFolder) -> Self; }

/// What the live connection and the child say. `None` = not attached.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveFacts {
    pub stages: Vec<(String, Option<i32>)>,   // stage name, end code once it ended
    pub fraction: Option<f32>,                // of the stage in progress
    pub child: Child,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Child { Running, Exited(i32), Killed, NotOurs }

/// ③ = collect, expert-gate, train; ④ = eval, showcase. ① ② are Done for a template.
pub fn phases(run: Option<&RunFacts>, live: Option<&LiveFacts>) -> [PhaseState; 5];
```

- [ ] **단계 1: 실패하는 테스트**, `project.rs`에(임시 디렉터리; 테스트 안에서
  `CARGO_MANIFEST_DIR/../..` 아래 실제 픽스처로의 경로를 가진 가짜 `Template`을 만든다):

```rust
#[test]
fn next_run_dir_never_reuses_a_number() {
    let p = scratch_project("next");
    assert!(p.next_run_dir().ends_with("runs/001"));
    std::fs::create_dir_all(p.root.join("runs/001")).unwrap();
    std::fs::create_dir_all(p.root.join("runs/007")).unwrap();
    std::fs::create_dir_all(p.root.join("runs/notes")).unwrap();
    assert!(p.next_run_dir().ends_with("runs/008"));
    assert_eq!(p.runs().iter().map(|r| r.number).collect::<Vec<_>>(), [1, 7]);
}

#[test]
fn create_refuses_an_existing_project() {
    let p = scratch_project("twice");
    assert!(Project::create(&p.root, "again", &cube(), &repo()).is_err());
}

#[test]
fn run_recipe_overrides_only_what_the_person_chose() {
    let p = scratch_project("recipe");
    let run = p.next_run_dir();
    let argv = write_run(&cube(), &repo(), &p,
        StartSettings { demonstrations: 50, length: Length::Short }, &run, "127.0.0.1:7001").unwrap();
    let written: es_data::training::Cycle =
        toml::from_str(&std::fs::read_to_string(run.join("cycle.toml")).unwrap()).unwrap();
    let template: es_data::training::Cycle =
        toml::from_str(&std::fs::read_to_string(repo().join(&cube().cycle)).unwrap()).unwrap();
    assert_eq!(written.collect.as_ref().unwrap().episodes, 50);
    assert_eq!(written.eval, template.eval);
    assert_eq!(written.scene, template.scene);
    // the inline train's marks are the preset, and the length is its last mark
    let marks = cube().marks(Length::Short).to_vec();
    assert_eq!(inline_marks(&written), marks);
    assert_eq!(argv[..2], ["loop".to_owned(), "cycle".to_owned()]);
    assert!(argv.windows(2).any(|w| w[0] == "--telemetry" && w[1] == "127.0.0.1:7001"));
}

#[test]
fn run_recipe_and_argv_keep_a_korean_path_with_spaces() {
    // "\u{bb38}\u{c11c}" and "\u{b0b4} \u{b85c}\u{bd07}" are Korean words; written as escapes
    // because Korean text lives only in the i18n tables.
    let root = std::env::temp_dir().join(format!("es-y6-{}", std::process::id()))
        .join("\u{bb38}\u{c11c}").join("\u{b0b4} \u{b85c}\u{bd07}");
    let p = Project::create(&root, "k", &cube(), &repo()).unwrap();
    let run = p.next_run_dir();
    let argv = write_run(&cube(), &repo(), &p,
        StartSettings { demonstrations: 10, length: Length::Short }, &run, "127.0.0.1:7002").unwrap();
    let out = argv.windows(2).find(|w| w[0] == "--out").unwrap()[1].clone();
    assert_eq!(PathBuf::from(out), run);
    let recipe = argv.windows(2).find(|w| w[0] == "--recipe").unwrap()[1].clone();
    assert!(Path::new(&recipe).is_file());
}
```

  `launch.rs`에:

```rust
#[test]
fn start_in_while_running_is_a_no_op() {
    let mut m = LaunchModel::default();
    let (shell, args) = long_running_shell(); // the platform shell the existing tests use
    m.start_program(&shell, &args);
    let pid = m.pid();
    m.start_in(&["loop".into(), "cycle".into()], Path::new("."));
    assert_eq!(m.pid(), pid, "a second start while running starts nothing");
    m.kill();
}
```

  (셸 자식과 그 pid를 위해 기존 launch 테스트에 이미 있는 헬퍼를 쓴다; `pid()` 접근자는
  없을 때만 추가한다.)

  `workflow.rs`에 — 상태표 전체:

```rust
fn live(stages: &[(&str, Option<i32>)], child: Child) -> LiveFacts {
    LiveFacts { stages: stages.iter().map(|(s, c)| ((*s).to_owned(), *c)).collect(),
                fraction: Some(0.5), child }
}
use PhaseState::*;

#[test]
fn no_run_yet() {
    assert_eq!(phases(None, None), [Done, Done, NotStarted, Locked, Locked]);
}
#[test]
fn collecting_is_train_running() {
    let s = phases(Some(&RunFacts::default()), Some(&live(&[("collect", None)], Child::Running)));
    assert_eq!(s[2], Running { stage: "collect".into(), fraction: Some(0.5) });
    assert_eq!((s[3].clone(), s[4].clone()), (Locked, Locked));
}
#[test]
fn evaluating_after_training() {
    let l = live(&[("collect", Some(0)), ("expert-gate", Some(0)), ("train", Some(0)), ("eval", None)], Child::Running);
    let s = phases(Some(&RunFacts::default()), Some(&l));
    assert_eq!(s[2], Done);
    assert!(matches!(&s[3], Running { stage, .. } if stage == "eval"));
}
#[test]
fn a_stage_that_ends_badly_fails_its_phase() {
    let l = live(&[("collect", Some(0)), ("expert-gate", Some(4))], Child::Exited(4));
    let s = phases(Some(&RunFacts::default()), Some(&l));
    assert_eq!(s[2], Failed { stage: "expert-gate".into(), code: Some(4) });
    assert_eq!(s[3], Locked);
}
#[test]
fn a_child_that_dies_mid_stage_fails_that_stage() {
    let l = live(&[("collect", Some(0)), ("train", None)], Child::Exited(101));
    assert_eq!(phases(Some(&RunFacts::default()), Some(&l))[2],
               Failed { stage: "train".into(), code: Some(101) });
}
#[test]
fn a_kill_is_stopped_by_you_with_a_resume_point() {
    let l = live(&[("collect", Some(0)), ("train", None)], Child::Killed);
    let f = RunFacts { collected: true, ..Default::default() };
    assert_eq!(phases(Some(&f), Some(&l))[2], StoppedByYou { resume_from: Some("train".into()) });
}
#[test]
fn a_finished_run_on_disk() {
    let f = RunFacts { collected: true, trained: true, evaluated: true, report: true };
    assert_eq!(phases(Some(&f), None), [Done, Done, Done, Done, Done]);
}
#[test]
fn a_dead_address_reads_as_interrupted_with_the_next_stage() {
    let f = RunFacts { collected: true, trained: true, ..Default::default() };
    let s = phases(Some(&f), None);
    assert_eq!(s[2], Done);
    assert_eq!(s[3], Interrupted { resume_from: Some("eval".into()) });
    let only_collect = RunFacts { collected: true, ..Default::default() };
    assert_eq!(phases(Some(&only_collect), None)[2], Interrupted { resume_from: Some("train".into()) });
    assert_eq!(phases(Some(&RunFacts::default()), None)[2], Interrupted { resume_from: None });
}
#[test]
fn results_open_whenever_a_report_exists() {
    // showcase failed after eval wrote its report: ④ failed, ⑤ still open
    let l = live(&[("collect", Some(0)), ("train", Some(0)), ("eval", Some(0)), ("showcase", Some(1))], Child::Exited(1));
    let f = RunFacts { collected: true, trained: true, evaluated: true, report: true };
    let s = phases(Some(&f), Some(&l));
    assert_eq!(s[3], Failed { stage: "showcase".into(), code: Some(1) });
    assert_eq!(s[4], Done);
}
```

- [ ] **단계 2:** `cargo test -p es-editor project workflow launch`를 실행; FAIL을 확인.
- [ ] **단계 3:** 구현. `RunFacts::read`는 실행 디렉터리에 `read_loop_steps`를 쓴다
  (collect → `LoopKind::Collect`, train → `LoopKind::Train`, eval →
  `LoopKind::Evaluate`를 비전문가 정책으로; cycle이 어느 ledger 행을 쓰는지 expert gate의
  것까지 확인해서 그 대응을 주석에 적는다)와 `report_path().is_file()`을. 재개 지점은
  사실이 덮지 않는 첫 단계이며, `es loop cycle --from`이 쓰는 표기 그대로 적는다.
  `Kind::Project`가 `recent::classify`에 합류하며, `RunView::is_run_dir`**보다 먼저**
  검사한다.
- [ ] **단계 4:** 실행; PASS를 확인. i18n과 labels 테스트를 실행(새 `phase.*` 키).
- [ ] **단계 5: 커밋** `feat(es-editor): project folders, run recipes and the step bar's state`.

**수용 기준:** 위 테스트들. **금지:** 그리는 코드; 템플릿의 cycle 레시피를 제자리에서
고치는 것; 테스트의 OS temp 디렉터리를 빼고 프로젝트 폴더 밖 어디에든 쓰는 것.

---

### 작업 Y7: 신호등

**파일:**
- 생성: `crates/es-editor/src/model/health.rs`; 수정: `model/mod.rs`, `i18n/en.toml`,
  `i18n/ko.toml`(`health.*` 키: 판정마다 이름 하나와 조언 한 줄)

**인터페이스:**
- 입력: 순수한 숫자뿐(UI가 `LiveRun`, `TrainView`, `LaunchModel`에서 채운다).
- 출력:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Light { Grey, Green, Amber, Red }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict { Starting, GoingWell, StoppedLearning, Slow, NotResponding, Broken,
                   Stopped, StoppedByYou }
impl Verdict {
    pub fn light(self) -> Light;
    pub fn key(self) -> &'static str;        // "health.<name>"
    pub fn advice_key(self) -> &'static str; // "health.<name>.advice"
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point { pub step: u64, pub loss: f64, pub samples_per_s: f64 }

#[derive(Clone, Debug, PartialEq)]
pub struct Input<'a> {
    pub since_start_s: f64,
    /// Seconds since the last telemetry message of any stream; `None` = none yet.
    pub since_last_message_s: Option<f64>,
    pub child_alive: bool,
    pub killed: bool,
    pub exit_code: Option<i32>,
    pub stage_codes: &'a [Option<i32>],
    pub total_steps: Option<u64>,
    pub points: &'a [Point],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    pub silence_s: f64,        // NotResponding after this long without a message
    pub warmup_points: usize,  // Slow and StoppedLearning are not judged before this
    pub slow_fraction: f64,    // latest samples/s below this x the median = Slow
    pub plateau_fraction: f64, // of total steps: no new loss minimum over this = StoppedLearning
}
/// ponytail: first guesses; Y-V calibrates them on a recorded successful run.
pub const THRESHOLDS: Thresholds = Thresholds { silence_s: 180.0, warmup_points: 20,
    slow_fraction: 1.0 / 3.0, plateau_fraction: 0.25 };

pub fn judge(input: &Input<'_>, th: &Thresholds) -> Verdict;
```

순서(맨 처음 맞는 것): `StoppedByYou`(강제 종료) → `Stopped`(exit code ≠ 0, 또는 어떤
stage code든 ≠ 0) → `Broken`(오차가 유한하지 않음) → `NotResponding`(살아 있고, 마지막
메시지 이후 시간 — 없으면 시작 이후 시간 — 이 `silence_s`를 넘음) → `Starting`(아직 메시지
없음) → `Slow` → `StoppedLearning` → `GoingWell`.

- [ ] **단계 1: 실패하는 테스트**(`health.rs`):

```rust
fn base<'a>(points: &'a [Point]) -> Input<'a> {
    Input { since_start_s: 600.0, since_last_message_s: Some(1.0), child_alive: true,
            killed: false, exit_code: None, stage_codes: &[], total_steps: Some(20_000), points }
}
fn falling(n: usize) -> Vec<Point> {
    (0..n).map(|i| Point { step: (i as u64 + 1) * 100, loss: 1.0 / (i as f64 + 1.0),
                           samples_per_s: 40.0 }).collect()
}

#[test] fn a_falling_curve_is_going_well() {
    assert_eq!(judge(&base(&falling(100)), &THRESHOLDS), Verdict::GoingWell);
}
#[test] fn a_kill_is_stopped_by_you() {
    let mut i = base(&[]); i.killed = true; i.child_alive = false; i.exit_code = Some(1);
    assert_eq!(judge(&i, &THRESHOLDS), Verdict::StoppedByYou);
}
#[test] fn a_bad_exit_or_stage_code_is_stopped() {
    let mut i = base(&[]); i.child_alive = false; i.exit_code = Some(4);
    assert_eq!(judge(&i, &THRESHOLDS), Verdict::Stopped);
    let codes = [Some(0), Some(3)];
    let mut j = base(&[]); j.stage_codes = &codes;
    assert_eq!(judge(&j, &THRESHOLDS), Verdict::Stopped);
}
#[test] fn nan_is_broken_even_as_the_first_point() {
    let p = [Point { step: 1, loss: f64::NAN, samples_per_s: 1.0 }];
    assert_eq!(judge(&base(&p), &THRESHOLDS), Verdict::Broken);
    let q = [Point { step: 1, loss: f64::INFINITY, samples_per_s: 1.0 }];
    assert_eq!(judge(&base(&q), &THRESHOLDS), Verdict::Broken);
}
#[test] fn silence_is_not_responding() {
    let mut i = base(&[]); i.since_last_message_s = Some(THRESHOLDS.silence_s + 1.0);
    assert_eq!(judge(&i, &THRESHOLDS), Verdict::NotResponding);
    let mut j = base(&[]); j.since_last_message_s = None; j.since_start_s = THRESHOLDS.silence_s + 1.0;
    assert_eq!(judge(&j, &THRESHOLDS), Verdict::NotResponding);
}
#[test] fn edge_series_never_panic_and_never_start_green() {
    let mut i = base(&[]); i.since_last_message_s = None; i.since_start_s = 5.0;
    assert_eq!(judge(&i, &THRESHOLDS), Verdict::Starting);
    let one = [Point { step: 1, loss: 2.0, samples_per_s: 10.0 }];
    assert_eq!(judge(&base(&one), &THRESHOLDS), Verdict::GoingWell);
    let mut none_total = base(&falling(100)); none_total.total_steps = None;
    let _ = judge(&none_total, &THRESHOLDS); // must not panic
}
#[test] fn a_rate_drop_after_warmup_is_slow() {
    let mut p = falling(60);
    p.last_mut().unwrap().samples_per_s = 5.0; // median 40, 5 < 40/3
    assert_eq!(judge(&base(&p), &THRESHOLDS), Verdict::Slow);
    let mut early = falling(5);
    early.last_mut().unwrap().samples_per_s = 5.0; // before warmup: not judged
    assert_eq!(judge(&base(&early), &THRESHOLDS), Verdict::GoingWell);
}
#[test] fn no_new_minimum_over_a_quarter_of_the_run_is_stopped_learning() {
    let mut p = falling(30); // steps 100..3000, minimum at step 3000
    let last = p.last().unwrap().loss;
    for k in 1..=60 { // 6000 more steps (> 0.25 * 20_000) without going below `last`
        p.push(Point { step: 3000 + k * 100, loss: last * 1.05, samples_per_s: 40.0 });
    }
    assert_eq!(judge(&base(&p), &THRESHOLDS), Verdict::StoppedLearning);
}
```

- [ ] **단계 2:** `cargo test -p es-editor health`를 실행; FAIL을 확인.
- [ ] **단계 3:** `judge`를 구현한다(워밍업 이후 점들의 중앙값; "새 최솟값 없음" =
  `step > last_step - plateau_fraction * total`인 점들의 최솟값이 그 앞 점들의 최솟값보다
  낮지 않음; `total_steps == None`이면 이 규칙을 건너뛴다).
- [ ] **단계 4:** 실행; PASS를 확인; i18n 테스트를 실행.
- [ ] **단계 5: 커밋** `feat(es-editor): the traffic light of a running cycle`.

**수용 기준:** 테스트 통과. **금지:** `judge` 안에서 텔레메트리나 시계를 읽는 것(순수
함수여야 한다); 임계값이 측정된 것이라고 주장하는 것.

---

### 작업 Y8: 결과 모델

**파일:**
- 생성: `crates/es-editor/src/model/results.rs`; 수정: `model/mod.rs`, `model/labels.rs`
  (`cause_key`, `perturbation_key`, 전체성 테스트), `i18n/en.toml`, `ko.toml`(`results.*`,
  `cause.*`, `cause.*.advice`, `perturb.*`)

**인터페이스:**
- 입력: `es_eval::episodes::EpisodeRow`(Y2), `es_eval::run_dir::RunDir`(Y4),
  `es_ir::evaluation::{EvaluationReport, EvaluationIr, PerturbationKind, MetricValue,
  MetricSpec}`, `es_safety::ViolationKind`, `es_core::FailureKind`(경로를 확인할 것).
- 출력:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cause { Timeout, FailureCondition, Unfinished, SafetyLimit, SafetyFallback,
                 MotionGaps, TooLate, Unstable, SensorDrop, ActuatorFault, BackendUnsupported }

/// A histogram bucket's cause. `success` is `None`. Total over every name `es-eval` writes:
/// the four terminations, every `FailureKind`, `fallback`, every `ViolationKind` — the match
/// arms name each one; an unknown name is `None` and is shown raw.
pub fn cause_of(bucket: &str) -> Option<Cause>;

#[derive(Clone, Debug, PartialEq)]
pub struct Card { pub passed: bool, pub successes: u32, pub episodes: u32 }
pub fn card(report: &EvaluationReport, rows: Option<&[EpisodeRow]>) -> Card;

#[derive(Clone, Debug, PartialEq)]
pub enum Comparison { NoPrevious, NotComparable, Delta { success_points: f64 } }
/// Success fraction difference in percentage points, only for an equal `evaluation_hash`.
pub fn compare(current: &EvaluationReport, previous: Option<&EvaluationReport>) -> Comparison;

/// Failed episodes per cause, most first, then `Cause` order; an episode with two causes
/// counts once under each.
pub fn causes(rows: &[EpisodeRow]) -> Vec<(Cause, u32)>;

#[derive(Clone, Debug, PartialEq)]
pub struct Situation { pub suite: String, pub kinds: Vec<PerturbationKind>,
                       pub successes: u32, pub episodes: u32 }
/// Per suite in the evaluation's own order. With rows: counted from them. Without (an old
/// run): from the report's `success_rate` value x `n_episodes`, rounded. `kinds` empty =
/// nominal; `ir == None` leaves `kinds` empty and the UI shows the raw suite name.
pub fn situations(report: &EvaluationReport, rows: Option<&[EpisodeRow]>,
                  ir: Option<&EvaluationIr>) -> Vec<Situation>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileFilter { All, Successes, Failures }
#[derive(Clone, Debug, PartialEq)]
pub struct Tile { pub cell: String, pub suite: String, pub success: bool,
                  pub cause: Option<Cause> }
pub fn tiles(rows: &[EpisodeRow], filter: TileFilter) -> Vec<Tile>;
```

`labels.rs`는 `cause_key(Cause) -> &'static str`, `cause_advice_key(Cause)`,
`perturbation_key(PerturbationKind) -> &'static str`를 얻는다. 각각 variant마다 가지 하나인
match다.

- [ ] **단계 1: 실패하는 테스트**(`results.rs`):

```rust
fn row(suite: &str, ep: u64, termination: &str, extra: &[(&str, u64)]) -> EpisodeRow {
    let mut histogram: BTreeMap<String, u64> = [(termination.to_owned(), 1)].into();
    for (k, n) in extra { histogram.insert((*k).to_owned(), *n); }
    EpisodeRow { suite: suite.into(), cell: format!("{suite}-{ep:02}"), episode: ep,
                 seed: 100 + ep, termination: termination.into(), steps: 10,
                 changed_steps: 0, histogram }
}

#[test] fn every_bucket_es_eval_writes_has_a_cause_or_is_success() {
    assert_eq!(cause_of("success"), None);
    for t in ["failure", "timeout", "unfinished", "fallback"] { assert!(cause_of(t).is_some(), "{t}"); }
    for k in es_safety::ViolationKind::ALL { assert!(cause_of(violation_bucket(k)).is_some(), "{k:?}"); }
    for f in FAILURE_KINDS { assert!(cause_of(f).is_some(), "{f}"); }
}
#[test] fn causes_count_failed_episodes_most_first() {
    let rows = [row("nominal", 0, "timeout", &[]), row("nominal", 1, "failure", &[("fallback", 2)]),
                row("nominal", 2, "timeout", &[]), row("nominal", 3, "success", &[])];
    assert_eq!(causes(&rows), [(Cause::Timeout, 2), (Cause::FailureCondition, 1), (Cause::SafetyFallback, 1)]);
}
#[test] fn different_evaluation_hash_is_not_comparable() {
    let (a, mut b) = (report_with(0.5, "aa"), report_with(0.25, "aa"));
    assert_eq!(compare(&a, Some(&b)), Comparison::Delta { success_points: 25.0 });
    b.evaluation_hash = hash("bb");
    assert_eq!(compare(&a, Some(&b)), Comparison::NotComparable);
    assert_eq!(compare(&a, None), Comparison::NoPrevious);
}
#[test] fn old_run_falls_back_to_suite_metrics() {
    let report = report_with(0.75, "aa"); // one suite "nominal", n_episodes 4
    let s = situations(&report, None, None);
    assert_eq!((s[0].successes, s[0].episodes), (3, 4));
    assert!(tiles(&[], TileFilter::All).is_empty());
    assert_eq!(card(&report, None), Card { passed: report.passed, successes: 3, episodes: 4 });
}
#[test] fn tiles_filter_and_name_their_cause() {
    let rows = [row("nominal", 0, "success", &[]), row("nominal", 1, "timeout", &[])];
    assert_eq!(tiles(&rows, TileFilter::Failures).len(), 1);
    assert_eq!(tiles(&rows, TileFilter::Failures)[0].cause, Some(Cause::Timeout));
    assert_eq!(tiles(&rows, TileFilter::Successes)[0].cell, "nominal-00");
}
```

  `violation_bucket`과 `FAILURE_KINDS`는 **`es-eval`이 쓰는 이름**이어야 한다: 문자열을
  다시 타이핑하지 말고 `metrics::violation_name` / `failure_name`을 public으로 만들거나
  (또는 재노출한다). `report_with(success_rate, hash_hex)`는 suite `nominal` 하나,
  `n_episodes: 4`, `success_rate` 셀, `passed: true`를 가진 `EvaluationReport`를 만든다.
  `labels.rs`에서 `metric_and_launch_labels_are_total`을 확장하거나(또는 형제 테스트를
  추가해) 모든 `Cause`와 모든 `PerturbationKind`가 두 언어 모두에서 구별되는 레이블을 갖게
  한다 — `_` 없이, 날것의 이름 없이.
- [ ] **단계 2:** `cargo test -p es-editor results labels`를 실행; FAIL을 확인.
- [ ] **단계 3:** 구현.
- [ ] **단계 4:** 실행; PASS를 확인; i18n 테스트를 실행.
- [ ] **단계 5: 커밋** `feat(es-editor): the results model — verdict, causes, situations, tiles`.

**수용 기준:** 테스트 통과. **금지:** `cause_of`가 아는 이름, `cause_key`,
`perturbation_key` 안의 와일드카드 가지; 서로 다른 `evaluation_hash`를 가로지르는 비교
숫자; §10.1의 어떤 지표든 다시 계산하는 것(report의 숫자가 그 숫자다).

---

### 작업 Y9: 시작 화면 모델

**파일:**
- 생성: `crates/es-editor/src/model/home.rs`; 수정: `model/mod.rs`, `i18n/en.toml`,
  `ko.toml`(`home.*`, `deps.*`)

**인터페이스:**
- 입력: Y1의 JSON; `template::{Template, templates_root, load}`(Y5); 최근 프로젝트
  카드를 위한 `project::Project`와 `workflow::{RunFacts, phases}`(Y6).
- 출력:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Deps { pub schema: u32, pub python: Python, pub modules: Modules,
                  pub vulkan_loader: bool, pub render: bool, pub backends: Vec<Backend> }
// Python { found: bool, path: Option<String> }, Modules { mujoco, torch, lerobot: bool },
// Backend { name: String, available: bool, reason: Option<String> }

pub fn parse_deps(json: &str) -> Result<Deps, String>;

/// `es --check-deps --json` on its own thread (the probe can take ~30 s).
pub struct DepsProbe { /* receiver */ }
#[derive(Clone, Debug, PartialEq)]
pub enum DepsState { Checking, Ready(Deps), Failed(String) }
impl DepsProbe { pub fn start(es: &Path) -> Self; pub fn poll(&mut self) -> &DepsState; }

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Availability { Ready, Missing(Vec<String>), Unknown }
/// `needs` names: "mujoco" | "torch" | "lerobot" (modules), "vulkan" (loader), "render".
pub fn availability(template: &Template, deps: Option<&Deps>) -> Availability;

/// One recent path, as the start screen lists it.
#[derive(Clone, Debug, PartialEq)]
pub enum RecentCard {
    Project { path: PathBuf, name: String, phases: [PhaseState; 5] },
    Other { path: PathBuf, kind: recent::Kind },
    Missing { path: PathBuf },
}
pub fn recent_cards(recent: &Recent) -> Vec<RecentCard>;
```

- [ ] **단계 1: 실패하는 테스트**(`home.rs`):

```rust
const READY: &str = r#"{"schema":1,"python":{"found":true,"path":"p"},
  "modules":{"mujoco":true,"torch":true,"lerobot":true},"vulkan_loader":true,"render":true,
  "backends":[{"name":"mujoco-cpu","available":true}]}"#;

#[test] fn ready_deps_make_the_cube_template_ready() {
    let d = parse_deps(READY).unwrap();
    assert_eq!(availability(&cube(), Some(&d)), Availability::Ready);
    assert_eq!(availability(&cube(), None), Availability::Unknown);
}
#[test] fn a_missing_module_or_render_names_what_is_missing() {
    let mut d = parse_deps(READY).unwrap();
    d.modules.lerobot = false; d.render = false;
    assert_eq!(availability(&cube(), Some(&d)),
               Availability::Missing(vec!["lerobot".into(), "render".into()]));
}
#[test] fn garbage_is_an_error_not_a_panic() {
    assert!(parse_deps("es --check-deps (spec 2.5)").is_err());
}
#[test] fn recent_cards_mark_what_is_gone() {
    let r = Recent { paths: vec![PathBuf::from("Z:/nowhere/at/all")] };
    assert!(matches!(recent_cards(&r)[0], RecentCard::Missing { .. }));
}
```

- [ ] **단계 2:** `cargo test -p es-editor home`를 실행; FAIL을 확인.
- [ ] **단계 3:** 구현(`DepsProbe`는 `Command`로 `es --check-deps --json`을 실행해,
  스레드에서 stdout을 읽고, `mpsc`로 결과 하나를 보낸다; `poll`은 절대 막지 않는다).
- [ ] **단계 4:** 실행; PASS를 확인; i18n 테스트.
- [ ] **단계 5: 커밋** `feat(es-editor): the start screen's model — PC check, templates, recent projects`.

**수용 기준:** 테스트 통과. **금지:** UI 스레드에서 probe를 도는 것; 준비되지 않은
템플릿을 숨기는 것(무엇이 없는지와 함께 비활성으로 보여 준다).

---

### 작업 Y10: 도킹 셸

**파일:**
- 수정: `Cargo.toml`(워크스페이스 의존성 `egui_dock`), `crates/es-editor/Cargo.toml`
- 생성: `crates/es-editor/src/ui/mod.rs`, `ui/shell.rs`, `ui/advanced.rs`
- 수정: `crates/es-editor/src/app.rs`(`eframe::App` 접착부가 된다: 상태, `update` 디스패치,
  저장), `src/lib.rs`, `src/main.rs`, `model/recent.rs`(`LAYOUT_KEY`), `i18n/en.toml`,
  `ko.toml`(`shell.*`, `menu.*`, `advanced.*`)
- 생성: `crates/es-editor/src/model/layout.rs`(단계마다 어떤 도크 탭이 있는지; 테스트됨)

**인터페이스:**
- 입력: `workflow::{Phase, PhaseState, phases}`(Y6); 기존의 모든 탭 함수.
- 출력:
  - `model/layout.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pane { Viewport, StepPanel, Summary, Console,
                AdvancedGraph, AdvancedSees, AdvancedProblems, AdvancedMetrics, AdvancedLive }
impl Pane { pub const ALL: [Pane; 9]; pub fn key(self) -> &'static str; }
/// The default arrangement for a phase: (left, centre, right, bottom) panes.
pub fn default_layout(phase: Phase) -> Layout;
```

  - `ui/shell.rs`: `pub fn draw(app: &mut EditorApp, ctx: &egui::Context)` — 메뉴 바(File:
    템플릿으로 새 프로젝트, 프로젝트 열기, 파일 열기, 실행 폴더 열기, 붙기, 최근 항목;
    View: 레이아웃 초기화, 언어, 글자 크기), 단계 바(`Phase`마다 버튼 하나, `PhaseState`에
    따라 색이 칠해짐, `Next ▶` 버튼과 "남은 것" 텍스트), `egui_dock` 영역, 상태 줄.
  - `ui/advanced.rs`: 오늘의 `graph_tab`, `run_tab`, `telemetry_tab`, `images_tab`,
    `diagnostics_tab`를 `app.rs`에서 변경 없이 옮긴 것, 각각 도크 탭 하나로 그려진다.
  - 예전 상단 탭 바와 예전 시작 화면은 삭제한다; `labels::Tab`의 역할은 `Pane`이
    대신한다(아무도 쓰지 않게 되면 `Tab`과 그 테스트를 지운다; `labels::Step`은 Y11의
    예전 시작 화면과 함께 없어진다).
  - 프로젝트가 아닌 경로(번들, 문서, 실행 폴더)를 열면 Advanced 패널이 있는 도크를 보여
    주고 단계 바는 `shell.not_a_project`와 함께 비활성이 된다.

- [ ] **단계 1: 크레이트 고정.** `egui` 요구 버전이 `0.32`이고 MSRV ≤ 1.85인 가장 최신
  `egui_dock`을 찾는다(`cargo info egui_dock@<v>`, 그 `Cargo.toml`); 레이아웃 저장이
  필요하면 `default-features = false`에 `serde`를 더해 워크스페이스에 추가한다.
  **하나도 없으면** 멈추고 보고한다: 대체안(egui `SidePanel`, 끌어서 다시 붙이기 없음)은
  오케스트레이터가 소유자에게 가져갈 범위 변경이다.
- [ ] **단계 2: 실패하는 테스트**(`layout.rs`):

```rust
#[test] fn every_phase_has_a_layout_with_the_viewport_in_the_centre() {
    for p in Phase::ALL {
        let l = default_layout(p);
        assert_eq!(l.centre, vec![Pane::Viewport], "{p:?}");
        assert!(l.left.contains(&Pane::StepPanel));
    }
}
#[test] fn every_pane_has_a_label_in_both_languages() {
    for p in Pane::ALL {
        for lang in [Lang::En, Lang::Ko] {
            let s = t(lang, p.key());
            assert!(!s.is_empty() && s != p.key() && !s.contains('_'), "{p:?} {lang:?}");
        }
    }
}
```

- [ ] **단계 3:** 실행; FAIL을 확인; `layout.rs`를 구현; 실행; PASS를 확인.
- [ ] **단계 4:** 탭 함수들을 `ui/advanced.rs`로 옮긴다; 셸을 만든다; 도크 상태를
  `recent::LAYOUT_KEY`(`"es-editor.layout"`) 아래 저장하고, 없거나 읽을 수 없으면 단계별
  기본값으로 돌아간다.
- [ ] **단계 5:** `cargo build -p es-editor`(target/alt), `cargo clippy -p es-editor
  --all-targets -- -D warnings`, `cargo test -p es-editor`; 그런 다음 손으로 실행해
  스크린샷을 찍는다: 빈 셸, 번들을 연 상태(Advanced 패널), 단계 버튼마다.
- [ ] **단계 6: 커밋** `feat(es-editor): the docking shell and the step bar`.

**수용 기준:** 테스트 통과; 스크린샷을 보고서에 첨부; 기존 기능 모두 Advanced 패널에서
여전히 닿을 수 있음. **금지:** `ui/`의 결정; 기능 제거; 다른 의존성.

---

### 작업 Y11: 시작 화면

**파일:**
- 생성: `crates/es-editor/src/ui/home.rs`; 수정: `ui/mod.rs`, `ui/shell.rs`(디스패치),
  `app.rs`(`DepsProbe`, 새 프로젝트 대화상자 상태), `model/labels.rs`(`Step`과
  `home_screen_lists_the_five_steps_in_loop_order` 삭제), `model/dialogs.rs`(새 프로젝트를
  위한 폴더 선택기, 아직 없다면), `i18n/*.toml`

**인터페이스:**
- 입력: `home::{DepsProbe, DepsState, availability, recent_cards}`, `template::load`,
  `project::Project::create`, `recent::Recent`.
- 출력: `pub fn draw(app: &mut EditorApp, ui: &mut egui::Ui)`; 프로젝트나 파일이 열려
  있지 않으면 창 전체.

레이아웃(설계 노트 3절): PC 점검 줄(`needs` 이름마다 쉬운 말로 한 항목, 초록/빨강, 선택
백엔드는 주황, `Checking` 중에는 "확인 중…", `Failed`면 실패 텍스트); 로딩된 템플릿마다
카드 하나로 "무엇을 해 볼까요?"(`Missing`이면 `home.missing` + 없는 항목과 함께 비활성) —
S3–S5의 카드는 아직 없음; 작은 단계 바가 딸린 최근 프로젝트들; 파일 수준의 입구. 템플릿
카드는 작은 대화상자를 연다: 프로젝트 이름(기본값은 `template`의 이름), 폴더(기본값
`<문서>/Electric Sheep/<이름>`; `dialogs`를 통한 OS 폴더 선택기), 만들기. 만들면 프로젝트가
③에서 열린다(①과 ②는 `Done`). 파싱에 실패한 템플릿은 카드들 아래에 경로와 이유와 함께
나열된다.

- [ ] **단계 1:** 구현; `cargo test -p es-editor`(i18n 완전성 테스트가 새 키를 모두
  잡는다); clippy.
- [ ] **단계 2:** 손으로, 이 기기에서(Python 없음): PC 점검이 Python, 학습 도구, MuJoCo를
  없음으로 보여 준다; 큐브 카드는 비활성이고 그것들을 나열한다; 더 이상 없는 최근 경로는
  사라진 것으로 보인다. 프로젝트 만들기는 (공백이 있는 한글 경로를 포함해) Y6의 테스트가
  검증하고, 여기서 손으로 하지 않는다. 스크린샷.
- [ ] **단계 3: 커밋** `feat(es-editor): the start screen`.

**수용 기준:** 스크린샷; 테스트 통과. **금지:** 끝까지 돌지 않는 템플릿이나 기능을 위한
카드.

---

### 작업 Y12: ③과 ④ — 시작, 지켜보기, 멈추기

**파일:**
- 생성: `crates/es-editor/src/ui/train.rs`; 수정: `ui/mod.rs`, `ui/shell.rs`, `app.rs`(실행
  컨트롤러: 어느 실행인지, 그 `LaunchModel`, 그 `LiveRun`/`TrainView`, 시작 시각, 완료 알림
  래치), `i18n/*.toml`

**인터페이스:**
- 입력: `project::{write_run, resume_argv, RunFolder}`, `launch::{LaunchModel::start_in,
  free_local_port}`, `workflow::phases`, `health::{judge, Input, THRESHOLDS}`,
  `live_run::LiveRun`, `train_view::TrainView`, `labels::stage_label`.
- 출력: 단계 패널, 가운데, 요약 패널을 위한 `pub fn draw_train(app, ui)`와
  `pub fn draw_evaluate(app, ui)`.

동작(설계 노트 6.4절):
1. 실행 전 **설정**: 시연 개수(기본값 `template.demonstrations`), 길이(짧게 / 보통 /
   길게). **시작** → `next_run_dir`, `127.0.0.1:{free_local_port()}`로 `write_run`,
   `telemetry.txt` 쓰기, `start_in(argv, repo_root)`.
2. `LiveRun::stages()`로부터 **작업 카드**, `labels::stage_label`을 거친 쉬운 이름; 진행률
   = 수집 중에는 `episode.end` 개수 ÷ 시연 개수, 학습 중에는 걸음 ÷ `TrainView::total()`;
   남은 시간은 `TrainView::eta()`가 있을 때만.
3. **가운데**: 수집 중에는 가장 최근 이미지 스트림 프레임과 `episode.end` 결과로부터의
   성공 카운터; 학습 중에는 샘플 이미지와 오차 곡선(기존 `paint_curve`).
4. **오른쪽**: (위 숫자로 `judge`한) 신호등과 그 이름·조언 줄; 곡선; 초당 샘플 수; GPU
   메모리는 지표 행이 실어 나를 때만.
5. **멈추기** → `LaunchModel::kill`. **지금까지 배운 걸로 평가** → 종료; 실행의
   `cycle.toml` `[eval] checkpoint`를 ledger나 `checkpoint` 이벤트가 보고하는 가장 새
   마크로 설정; `start_in(resume_argv(run, "eval", addr), repo_root)`. (`--from`이 이를
   받아들이는지는 Y-V 항목 3이다; 그때까지는 버튼이 있고 실패는 신호등으로 보여 준다.)
6. **완료**: ④가 `Done`이 되는 첫 프레임에서
   `egui::ViewportCommand::RequestUserAttention(egui::UserAttentionType::Informational)`을
   한 번 보내고 `train.done`을 보여 준다; 현재 단계가 ③이나 ④이면 ⑤로 넘어간다.
7. **다시 열면**: `telemetry.txt`의 주소로 한 번 접속해 본다; 실패하면 단계들은
   *중단됨*으로 읽히고 패널이 [이어서](재개 지점을 담은 `resume_argv`)와
   [처음부터](새 실행)를 내놓는다.
8. **④**는 eval 단계가 있는 같은 화면에, `LiveRun`의 `cell.end` 행이 도착하는 대로
   채워지는 타일 줄이 더해진 것이다.

- [ ] **단계 1:** 구현; 테스트 + clippy.
- [ ] **단계 2:** Python 없이 손으로: 시작 → Python이 필요한 첫 단계에서 자식이 실패 →
  신호등이 그 단계의 쉬운 이름과 함께 *멈췄어요*로 바뀌고 콘솔은 `es` 자신의 메시지를
  담는다; 실행 중인 자식에 멈추기를 누르면 *당신이 멈췄어요*가 보인다. 스크린샷.
- [ ] **단계 3: 커밋** `feat(es-editor): starting and watching a run`.

**수용 기준:** 실패 경로와 강제 종료 경로의 스크린샷. **금지:** 실행 전체의 ETA; 자식
프로세스, 소켓, 또는 레시피보다 큰 파일에 UI 스레드를 막는 것.

---

### 작업 Y13: ⑤ — 결과

**파일:**
- 생성: `crates/es-editor/src/ui/results.rs`; 수정: `ui/mod.rs`, `ui/shell.rs`, `app.rs`,
  `model/dialogs.rs`(`save_file(default_name, filter)`), `i18n/*.toml`

**인터페이스:**
- 입력: `results::{card, compare, causes, situations, tiles, Tile, TileFilter}`,
  `es_eval::episodes::read_episodes`, `es_eval::run_dir::RunDir`, `replay_view::ReplayView`,
  `es_render::raster::{Camera, Raster}`, `labels::{cause_key, cause_advice_key,
  perturbation_key, metric_label}`, 그 실행의 Evaluation IR(`cycle.toml`의
  `[eval] config`에서, 저장소 상대 경로).
- 출력: `pub fn draw(app, ui)`.

동작(설계 노트 6.5절): 실행 목록; 카드(합격/불합격, "n 중 x", 합격 기준 줄마다 ✓/✗와 함께
쉬운 말, 프로젝트의 지난 실행과의 비교 또는 `results.not_comparable`); 조언이 딸린
원인들; 재생기(바깥 궤도 카메라 / `frames/<cell>/`에서 온 정책의 눈 / 나란히; 0.5×/1×/2×;
기존 타임라인 표시); 상황별 막대(레이블 = 이어 붙인 `perturbation_key`들, 없으면
`results.nominal`, 마우스를 올리면 날것의 suite 이름); 필터가 있는 타일; 접힘: 기존 지표
표와 해시들; 버튼: 정책 내보내기(가장 새 체크포인트 번들, `dialogs::save_file`), 다시
실행(같은 설정으로 ③). `episodes.json`이 없는 옛 실행은 타일이 있을 자리에
`results.old_run`을 보여 준다.

- [ ] **단계 1:** 구현; 테스트 + clippy.
- [ ] **단계 2:** 손으로: 임시 프로젝트 사본 안에서
  `tests/fixtures/visible-learning/run`(커밋된 E5 픽스처 실행)을 연다 → 옛 실행 경로;
  그런 다음 손으로 쓴 `episodes.json`을 그 임시 사본 옆에(픽스처 안에는 절대 두지 않는다)
  두고 같은 것을 한다 → 타일, 원인, 막대. 스크린샷.
- [ ] **단계 3: 커밋** `feat(es-editor): the results screen`.

**수용 기준:** 두 경로 모두의 스크린샷. **금지:** 커밋된 픽스처 실행을 수정하는 것.

---

### 작업 Y14: 문서, 그리고 나중을 위한 체크리스트

**파일:**
- 수정: `docs/design/editor-shell.md` + `.ko.md`(새 §17: S1이 무엇을 바꿨는지,
  `editor-redesign.md`를 가리킴), `docs/design/editor-redesign.md` + `.ko.md`(상태 줄;
  구현이 정리한 것들 — 길이 프리셋, ledger 대응, `egui_dock` 버전), `CLAUDE.md`(프로젝트
  상태: M12 plan Y 구현됨, Y-V 대기)
- 생성: `docs/packets/M12/YV-verification.md` + `.ko.md`

Y-V 체크리스트, 각 항목마다 정확한 명령, 어디서 돌 수 있는지(venv가 있는 이 PC, 또는 서버
— 소유자가 정한다), 무엇이 합격인지:

1. 큐브 템플릿의 `es loop cycle --recipe runs/…/cycle.toml`이 LeRobot 경로로 끝까지 돈다;
   수치 기록(held-out 성공률, 봉투 비율), 그 옆에 V19b.
2. `episodes.json`의 성공 수가 그 실행의 `success_rate`와 같다(Y2의 게이트 테스트를
   실제로 돌림).
3. `[eval] checkpoint`가 바뀐 뒤의 `--from eval`("지금까지 평가" 버튼).
4. 신호등: 그 실행의 진행 프레임을 픽스처로 기록하고, `health.rs`에
   `a_recorded_successful_run_is_green_throughout`을 추가하고, 합성 테스트를 약화시키지
   않고 통과할 때까지 `THRESHOLDS`를 조정한다.
5. 쓰인 기기에서 길이 프리셋의 실제 소요 시간, 시작 화면의 추정을 위해.
6. Windows에서 손으로 하는 전체 흐름: 시작 → 템플릿 → 시작 → 지켜보기 → 결과, 스크린샷.

- [ ] **단계 1:** 문서를 쓴다(한국어 자매 문서는 Sonnet 에이전트에게 맡기고 검토할 수
  있다).
- [ ] **단계 2:** `cargo xtask check-spec-refs`; pre-commit 짝짓기 검사.
- [ ] **단계 3: 커밋** `docs: the editor's workflow shell (M12 plan Y) and its verification list`.

---

## 자체 검토(쓰면서 함)

- **스펙 커버리지:** 6.1절 셸/도크/코드 배치 → Y10; 이전 → Y3, Y4; 6.2절 프로젝트/템플릿 →
  Y5, Y6; 6.3절 단계 상태 → Y6; 6.4절 진행/신호등 → Y7, Y12; 6.5절 결과 → Y2, Y8, Y13; 6.6절
  백엔드 → Y1, Y2, Y5; 6.7절 오류 → Y6(중단됨), Y7(멈췄어요/당신이 멈췄어요), Y8(옛 실행),
  Y9(없는 의존성, 깨진 템플릿), Y13(report 없음); 6.8절 오라클 → 각 작업의 테스트 + Y-V;
  8절의 열린 사실 다섯 가지 → Y5 단계 1, Y10 단계 1, Y-V 1–5.
- **자리표시자:** 길이 프리셋은 Y5 단계 2에서 `training.rs`의 규칙으로부터 고른다;
  `THRESHOLDS`는 첫 추측이라고 선언되어 있고 보정 작업이 이름 붙어 있다(Y-V 4).
- **타입 이름:** `es_eval::run_dir::RunDir`(끝난 평가의 파일들, Y4)와
  `project::RunFolder`(프로젝트의 번호 매겨진 `runs/NNN` 폴더, Y6)는 일부러 다르게 이름
  붙인 다른 것들이다; `RunFolder::eval_dir()`이 바로 `RunDir`이 여는 것이다.
