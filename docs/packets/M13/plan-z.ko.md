<!-- Korean translation of docs/packets/M13/plan-z.md. The English file is the working copy; regenerate this when it changes. -->
# M13 plan Z — S2: 학습하는 모습을 지켜보고, 실패에서 배우기

> `docs/design/editor-redesign.md`(3, 5절)의 하위 프로젝트 S2를 위한 패킷. 오너는 2026-09-29에
> 자율권을 부여했다("남은 구현을 진행"): 설계는 합의된 방향을 따르고, 스펙·안전 평면·해시에 관한
> 결정만 오너에게 돌아간다. 각 작업은 §1.2의 패킷이며, **Files** 블록이 `cargo xtask check-scope`용
> `context`이다.

**목표:** 초보자가 학습 *도중*에 정책이 나아지는 모습을 보고(체크포인트마다 짧은 테스트를 영상으로),
실패한 시도가 *왜* 실패했는지 작업의 언어로 읽고, 실패한 부분을 더 많이 담아 다시 학습하는 버튼을
하나 눌러 실행할 수 있다 — 평가에 쓰는 시드로는 학습하지 않으면서.

**첫 실제 실행이 S2의 방향을 바꾼 점(2026-09-29).** 큐브 작업은 실패 조건(failure predicate)을
선언하지 않는다. 실패한 에피소드는 모두 *시간 초과*이다(`YV-verification.md`). 그래서 설계 노트 S2의
"어떤 `Terminate`가 발동했는가"는 아무것도 가리키지 못한다. S2b는 대신 물체의 궤적으로 실패를
분류한다. `es loop collect`에는 섭동이 없으므로, S2c는 Evaluation IR 자체의 `PerturbationKind`로
섭동을 더한다. 설계 노트는 Z7에서 갱신한다.

## 전역 제약

- `plan-y.md`의 전역 제약이 모두 그대로 유효하다(MSRV 1.85, egui 0.32.3, i18n과 레이블 전체
  덮기, 결정은 `es-editor-model`에, 골든은 생성기로만, 트레일러 없음, 워크트리 + `--ff-only`,
  푸시 금지, 주 트리에서는 `CARGO_TARGET_DIR=target/alt`).
- **원격 서버 없음.** 로컬 PC만 쓴다. 실제 실행에는
  `ES_PYTHON=F:/Projects/electric_sheep/.venv/Scripts/python.exe`를 쓴다(LeRobot 오라클은
  `ES_LEROBOT_PYTHON`을 읽는다).
- **다른 패킷이 컴파일되는 동안 에이전트 패킷 안에서 학습 실행 금지**(리뷰 S-6). 실제 실행은
  오케스트레이터가 조용한 PC에서 한다.
- **§13.3 규율:** 미리보기와 재학습은 실행을 판정하는 Evaluation IR을 건드리지 않는다. 미리보기 IR은
  파생되어 결과 옆에 쓰이며, 합격 기준과 비교되지 않는다. 재학습은 모든 평가 시드와 겹치지 않는
  시드로 수집한다.
- **안전 평면:** 손대지 않는다. 섭동을 준 수집도 같은 평면 아래에서 돈다(INV-12).
- **해시:** 기존 해시 정의는 움직이지 않는다. 새 출처 정보(미리보기 IR, 섭동 명세, 병합 데이터셋)는
  기존 해시 입력이 아니라 파일과 원장에 기록한다.

## 리뷰 초점

1. 체크포인트 미리보기는 사람이 알아챌 만큼 학습을 늦추면 안 된다: Y-V에서 측정한다. 미리보기는
   워커 1개, 스위트 1개, 에피소드 4개이다.
2. `lerobot-train`이 아직 쓰는 중인 LeRobot 체크포인트를 읽는 경우: 완전한 디렉터리만 가져온다
   (Z1 테스트).
3. 궤적이 기록되지 않은 실패 에피소드(예전 실행, `--traj` 꺼짐): 원인을 지어내지 않고 일반적인
   종료 원인을 보여 준다(Z4 테스트).
4. 평가에 실패가 없던 실행에서 "다시 학습": 버튼이 그렇다고 알리고 아무것도 하지 않는다(Z4 테스트).
5. 시드: 평가 시드와 겹치는 섭동 재수집은 거부한다(Z2 테스트).

---

## 웨이브

| 웨이브 | 작업 | 필요 |
|---|---|---|
| 1 | Z1 체크포인트 미리보기 (es) · Z2 섭동 수집 (es) | M12 R4–R7 병합 |
| 2 | Z3 "다시" cycle (es, es-data) · Z4 S2 모델 (es-editor-model) | Z1, Z2 |
| 3 | Z5 S2 화면 (es-editor) · Z6 템플릿 결과 필드 | Z3, Z4 |
| 4 | Z7 실제 실행(오케스트레이터), 문서, 리뷰 | 웨이브 3 |

---

### 작업 Z1: 체크포인트마다 짧은 테스트

**Files:** `crates/es/src/cmd/train.rs`, `crates/es/src/cmd/cycle.rs`,
`crates/es/src/cmd/telemetry.rs`, `crates/es-data/src/training.rs`, `crates/es/tests/cli.rs`,
`tests/golden/train/**`(생성기만), `docs/packets/M13/Z1-*.md` + `.ko.md`.

**동작.**
- `Cycle`에 `[eval.preview]`가 생긴다(선택; 템플릿의 cycle에서는 Z6을 통해 기본 켜짐):
  `episodes`(기본 4), `suite`(기본은 Evaluation IR의 첫 스위트, 즉 명목 스위트), `frames`(기본 true).
  테이블이 없으면 오늘의 동작 그대로이며, 커밋된 cycle의 dry-run 계획은 바이트 단위로 같다(골든이
  움직이지 않는다).
- 체크포인트 번들이 쓰인 뒤(IR 경로: 마크마다 `es policy pack` 뒤; LeRobot 경로:
  `lerobot-train`이 `checkpoints/<NNNNNN>/pretrained_model`을 다 쓴 **즉시** — `es train`은
  트레이너가 도는 동안 디렉터리를 지켜보다가, 끝에 한꺼번에 가져오는 대신 완전한 마크마다 그때
  가져온다. 디렉터리는 `model.safetensors`와 `config.json`이 존재하고 그것들이 나타난 뒤 트레이너의
  다음 진행 줄을 읽었을 때 완전하다), `es loop cycle`은 그 번들에 대해 `es eval run`을 **자식
  프로세스**(실행 중인 `es` 바이너리)로 시작한다. 파생 Evaluation IR은 고른 스위트 하나만, 그
  앞쪽 `episodes`개 시드만, 같은 지표로, 합격 기준은 뺀 것이다. 결과는 `<run>/preview/<mark>/`에
  쓰이며(파생 IR을 `evaluation.toml`로, 이어서 평소의 `report.json`, `episodes.json`, `frames/`,
  `traj/`) `--jobs 1`로 돌고, 트레이너를 막지 않는다. 미리보기는 한 번에 하나씩 대기열에 넣는다.
- 텔레메트리(스트림 1): `preview.begin{step, dir}`과 `preview.end{step, dir, successes,
  episodes, code}`. cycle의 원장은 마크마다 `preview` 행을 하나 얻는다(원장 스키마가 해시를 움직이지
  않고 허용할 때만 새 `LoopKind`; 아니면 행은 `<run>/preview/index.jsonl`로 간다 — 정하고 어느
  쪽인지 적는다).
- cycle은 자신의 eval 단계 전에 마지막 대기 중 미리보기를 기다린다.

**오라클.** 단위 테스트: 커밋된 `evaluation.toml`에 대한 파생 IR(스위트 하나, 시드 N개, 합격 기준
없음); 단계별로 쓴 임시 디렉터리에 대한 LeRobot 완전성 규칙; `[eval.preview]`가 있는 cycle의 dry-run
계획을 새 골든으로 고정; 커밋된 cycle의 골든은 그대로. Python 게이트: 필요 없음(Z7이 실제로
돌린다).

### 작업 Z2: Evaluation IR의 섭동 아래에서, 새 시드로 하는 수집

**Files:** `crates/es/src/cmd/loop.rs`, `crates/es-data/src/collect.rs`,
`crates/es-eval/src/perturb.rs`(가시성만), `crates/es/tests/cli.rs`,
`docs/packets/M13/Z2-*.md` + `.ko.md`.

**동작.**
- `es loop collect … --perturb <evaluation.toml> --suites <a,b> [--seed <S>]`: 에피소드 `i`는
  `es_eval::PerturbationPlan`을 통해 스위트 `suites[i % len]`의 섭동 아래에서 돈다. `es eval run`이
  적용하는 방식 그대로(같은 스트림, 같은 키 방식)이며 같은 안전 평면 아래에서 돈다.
- **시드:** 수집 시드 범위 `[S, S+N)`은 Evaluation IR의 확정된 시드와 겹치면 안 된다. 겹치면 이름을
  밝혀 거부한다(`--seed`가 평가 시드 101–116과 겹침).
- 출처: 데이터셋의 에피소드별 메타에 스위트 이름을 기록하고, 원장의 collect 행에 `perturb.config`
  (경로 + `evaluation_hash`)와 `perturb.suites`를 기록한다.
- 전문가는 여전히 시연한다(`--expert`). 그래서 전문가가 할 수 있는 한 시연은 섭동 아래에서도 성공으로
  남는다. collect 요약은 스위트별 성공 수를 출력한다.

**오라클.** 단위 테스트: 인덱스에 의한 스위트 배정; 시드 겹침 거부; 원장 행. Python 게이트 CLI
테스트(`ES_PYTHON`이 없으면 여기서는 건너뜀): 스위트 두 개에 걸친 에피소드 4개에서, 각 에피소드에
기록된 스위트와, 밝기(light) 강도 섭동 에피소드의 렌더된 프레임이 명목 프레임과 다르다.

### 작업 Z3: "다시" cycle

**Files:** `crates/es-data/src/training.rs`, `crates/es/src/cmd/cycle.rs`,
`crates/es/tests/cli.rs`, `tests/golden/train/**`(생성기만), `docs/packets/M13/Z3-*.md` +
`.ko.md`.

**동작.** `Cycle`은 모두 선택인 항목을 얻는다:
- `[collect] perturb = { config = "...", suites = ["..."] }` → Z2의 플래그;
- `[collect] merge = ["<이전 데이터셋 루트>", ...]` → collect 뒤 `es loop distill`이 새 데이터셋을
  그 루트들과 `collect/merged`로 병합하고, 학습은 병합된 루트를 쓴다;
- `[train] init = "<번들 또는 lerobot 체크포인트 디렉터리>"` → IR 경로의 `[init] policy`;
  LeRobot 경로의 `--policy.path`(문서화된 미세조정 경로).
dry-run 계획은 새 단계를 보여 주며, 커밋된 cycle의 골든은 움직이지 않는다.

**오라클.** "다시" cycle의 dry-run 골든(픽스처 경로); `collect` 없는 `merge`, 없는 파일을 가리키는
`init`을 거부하는 스키마 테스트.

### 작업 Z4: S2 모델(`es-editor-model`)

**Files:** `crates/es-editor-model/src/model/{watch.rs, results.rs, project.rs, template.rs}`,
새 `model/preview.rs`와 `model/outcome.rs`, `crates/es-editor-model/i18n/*`,
`docs/packets/M13/Z4-*.md` + `.ko.md`.

**동작.**
- `preview.rs`: `preview.begin/end` 이벤트와 디스크의 `<run>/preview/*/`에서 얻은 실행의 미리보기
  (다시 연 실행도 보여 준다): 스텝, 성공/에피소드, 먼저 재생할 셀.
- `outcome.rs`: 궤적(`.estraj`)과 템플릿의 `[outcome]`(Z6)에서, 물체 바디의 시간에 따른 위치 →
  `NeverLifted`(최대 높이 증가 < `lift_m`), `LeftOutside`(들어 올렸으나 최종 위치가 목표 영역 밖),
  `InsideTooLate`(최종 위치가 목표 영역 안이지만 에피소드가 시간 초과). 목표 영역은 이름이 템플릿의
  `target` 어간으로 시작하는 씬 geom들의 AABB이다(Y15가 이미 "bin"에 쓰는 규칙). 궤적 없음 →
  `None`(일반 원인이 그대로 남는다).
- `results.rs`: 실패한 타일의 원인은 결과 분류를 우선하고, "왜 실패했나" 목록은 결과 분류가 있으면
  그 분류를 센다.
- `project.rs`: `write_run_again(template, repo, project, previous: &RunFolder, results, settings)`
  → Z3 cycle: `perturb.suites` = 성공률이 가장 낮은 스위트들(동률이면 동률 전부; 실패가 없으면
  화면이 보여 줄 이유와 함께 거부), 이전 모든 실행의 collect 시드 뒤이면서 평가 시드 밖인 새 시드
  범위, `merge` = 이전 실행들의 데이터셋 루트, `init` = 이전 실행의 평가된 체크포인트.

**오라클.** 각각에 대한 headless 테스트: 픽스처 이벤트 스트림과 픽스처 디렉터리에서 나온 미리보기
목록; 합성 궤적(들었다 떨어뜨림, 들지 못함, 안에 늦게 들어감)과 커밋된 `.estraj` 픽스처에서의 결과
분류; `write_run_again`의 레시피가 Z3 `Cycle`로 파싱되고, 그 시드가 101–116과 이전 범위를 피하며,
실패가 없는 실행은 거부된다.

### 작업 Z5: S2 화면(`es-editor`)

**Files:** `crates/es-editor/src/ui/{train.rs, results.rs}`, `crates/es-editor/src/app.rs`
(훅만), `docs/packets/M13/Z5-*.md` + `.ko.md`.

**동작.** ③의 가운데는 가장 새 미리보기를 재생하며(기존 재생 캔버스) "4번 중 1번 성공 — 1,000걸음
때"를 띄운다. 곡선은 각 미리보기 스텝을 표시하고, 클릭하면 선택된다. ⑤는 결과 분류와 **"실패 위주로
다시 학습"** 버튼을 보여 주며(거부되면 모델의 이유와 함께 비활성), 이 버튼은 `write_run_again`으로
다음 실행을 쓰고 Start처럼 시작한다.

**오라클.** 빌드, clippy, 테스트; 미리보기가 있는 ③과 버튼이 있는 ⑤의 스크린샷(앱 내 훅, 커밋
안 함).

### 작업 Z6: 템플릿의 결과 및 미리보기 필드

**Files:** `templates/*.toml`, `tests/fixtures/visible-learning/cycle-*.toml`(새 파일로만),
`crates/es-editor-model/src/model/template.rs`(`[outcome]` 테이블), i18n.

`[outcome] object = "cube"`, `target = "bin"`, `lift_m = 0.02`에, 물체와 목표의 쉬운 말 이름을 두
언어로 붙인다. 템플릿의 cycle은 `[eval.preview]`를 얻는다.

### 작업 Z7: 실제 실행과 문서(오케스트레이터)

조용한 PC에서: 미리보기를 켠 medium의 힌트 카드(U3, R7); 그 뒤 "다시 학습" 실행 한 번; 수치, 학습
시간에 대한 미리보기 비용, 결과 분류를 `YV-verification.md`에 기록; 설계 노트의 S2 절을 실제로 만든
것에 맞춰 다시 쓰기; 리뷰.
