<!-- Korean translation of docs/packets/M14/plan-q.md. The English file is the working copy; regenerate this when it changes. -->
# M14 플랜 Q — S3: 동작 블록으로 가르치기

> `docs/design/editor-redesign.md`(3, 5, 11절)의 하위 프로젝트 S3를 위한 패킷. 오너는 2026-09-29에
> 자율권을 부여했다("남은 구현을 진행"): 설계는 합의된 방향을 따르고, 스펙·안전 평면·해시에 관한
> 결정만 오너에게 돌아간다. 각 작업은 §1.2의 패킷이며, **Files** 블록이 `cargo xtask check-scope`용
> `context`이다.

**목표:** ②에서 사람이 시범을 쉬운 말로 된 짧은 블록 목록으로 본다("큐브 위 4.5 cm로 이동",
"집게 닫기", "통 위로 이동"). 블록 하나를 바꾸고 "한 번 해 보기"를 눌러 시범 로봇이 하는 모습을
지켜본 다음, *자신의* 프로그램에서 나온 시범으로 학습한다 — TOML 파일도, 코드 한 줄도 없이.

**하나의 아이디어.** 오늘의 시범기(`es_env::ScriptedExpert`, `--expert so101-pick-place`)는 이미
웨이포인트 목록이다. 일곱 단계의 상태 기계이고, 그 목표는 손으로 맞춘 숫자 묶음 하나(`ExpertCfg`,
`demo_cfg`)에서 나온다. S3는 이 목록을 데이터로 만든다: 같은 구조체가 실행하는 **시범 프로그램**
파일이다. 내장 `so101-pick-place`는 커밋된 프로그램 `templates/teach/so101-pick-place.toml`이
되고, 리팩터링은 오늘의 청크를 비트 단위로 담은 골든으로 증명한다.

## 전역 제약

- `plan-z.md`의 전역 제약이 모두 그대로 유효하다(MSRV 1.85, egui 0.32.3, i18n과 레이블 전체
  덮기, 결정은 `es-editor-model`에, 골든은 생성기로만, 트레일러 없음, 워크트리 + `--ff-only`,
  푸시 금지, `CARGO_TARGET_DIR=target/alt`, 원격 서버 없음, 에이전트 패킷 안에서 학습 실행 금지).
- **IR이 아니다.** 시범 프로그램은 오늘의 `ExpertCfg`처럼 시범기의 레시피이다(Task IR은 정책을
  갖지 않는다, §5.1 규칙 6). IR 타입은 바뀌지 않으며 `es-ir`는 건드리지 않는다.
- **새 트레이트 없음(INV-17).** 프로그램은 기존의 구체 타입 `ScriptedExpert`가 실행하는 데이터이다.
- **안전 평면:** 손대지 않는다. 모든 시범은 여전히 평면을 거치며(INV-12), 오늘과 똑같이
  `ExpertCfg::pace_to`로 Deployment IR의 봉투에 맞춰 속도가 정해진다.
- **해시:** 기존 해시 정의는 움직이지 않는다. 프로그램에서 수집한 데이터셋은 다른 *데이터*이며(내용
  해시가 그렇게 말한다), 원장의 수집 단계는 프로그램의 blake3를 기존 `expert` 입력 옆에 새 출처
  정보로 기록한다.
- **결정론(§3.4):** 프로그램의 숫자는 한 번만 파싱한다. 실행기는 오늘의 `es_math::approx`
  초월함수를 그대로 쓴다. 블록의 각도(degree)는 `demo_cfg`와 똑같이 `DEG_TO_RAD`로 변환하므로,
  내장 프로그램은 오늘의 라디안을 비트 단위로 재현한다.

## 리뷰 초점

1. **리팩터링은 아무것도 움직이지 않는다:** Q1의 오늘 청크열 골든(리팩터링 *전에*, 자기 커밋으로
   기록)은 리팩터링 뒤에도 바이트 단위로 같고, Python 게이트의 `expert_solves_the_pinned_seeds`는
   여전히 8개 중 8개를 푼다.
2. **장소는 정확히 해석된다:** "통"은 스템이 `bin`인 지오메트리 중 가장 낮은 것의 세계 좌표
   위치(`bin_floor`, `0.14 -0.1`)이며, `demo_cfg`가 리터럴로 쓰는 것과 같은 `f64` 값이다.
3. **닿지 못하는 블록은 실행 전에 알린다:** ②는 Task IR이 그릴 수 있는 어떤 물체 위치(그
   `Randomization` 범위의 꼭짓점)에서 목표가 SO-101의 도달 범위를 벗어나는 이동 블록을 쉬운
   말로 표시한다. "한 번 해 보기"는 그래도 그것을 실행한다(시범기가 그 시범을 이름을 대며
   실패시킨다, 스펙 17.2) — 클램프한 근사는 절대 아니다.
4. **템플릿이 아니라 사람의 파일:** `Project::create`는 템플릿의 프로그램을 프로젝트로 복사한다.
   그것을 편집해도 `templates/` 아래에는 절대 쓰지 않는다.

---

## 웨이브

| 웨이브 | 작업 | 필요 |
|---|---|---|
| 1 | Q1 프로그램과 그 실행기 (es-env) | — |
| 2 | Q2 `--expert <file>` (es, es-data) · Q3 가르치기 모델 (es-editor-model) | Q1 |
| 3 | Q4 ② 화면 (es-editor) | Q3 |
| 4 | Q5 실제 실행(오케스트레이터), 문서, 리뷰 | 웨이브 3 |

---

### 작업 Q1: `ScriptedExpert`가 실행하는 시범 프로그램

**Files:** `crates/es-env/src/expert.rs`(`expert.rs`를 읽기 좋게 유지하는 데 필요하면 새
`crates/es-env/src/program.rs`도), `crates/es-env/src/lib.rs`(export),
`crates/es-env/tests/expert.rs`, `tests/golden/expert/**`(생성기로만),
`templates/teach/so101-pick-place.toml`.

**형식**(TOML, `kind = "demonstration"`):
```toml
kind   = "demonstration"
robot  = "SO-101"
object = "cube"          # the free-joint body a block calls "object"; latched at episode start

[[blocks]]               # move: go to a point, gripper held as given
move  = "object"         # "object" | a place (a geom stem in the scene, e.g. "bin") | [x, y]
above = 0.045            # metres above the target's own height  -- or --
# height = 0.14          # metres above the world origin (exactly one of the two)
pitch = -85              # tool pitch, degrees; -90 is straight down
grip  = "open"           # "open" | "closed"

[[blocks]]               # grip: hold the previous block's pose, set the gripper, wait
grip = "closed"
wait = 5.0               # seconds
```
- 이동 블록은 네 팔 관절이 그 IK 해의 `pos_tol` 안에 들어오면 도달한 것이다(오늘의 규칙). 집게
  블록은 `wait`초 뒤에 넘어가며, 이는 시범기 자신의 스텝으로 센다 — 재계획마다 하나, 제어 틱
  `replan_every`마다(`ExpertCfg::pace_to`). 50 Hz 제어율에서 10틱마다 재계획하면 5초는 25스텝이며,
  오늘의 `close_ticks`이다. 스텝의 정수배가 아닌 `wait`는 이름을 대며 거부한다. 마지막 블록 뒤에는
  마지막 명령을 유지한다(오늘의 `Done`).
- `"object"`의 점은 에피소드 첫 스텝에 래치한 물체의 위치이다(오늘의 `grasp`). 장소의 점은 그
  스템의 **가장 낮은** 지오메트리(통의 바닥)의 세계 좌표 위치, x와 y이다. `above`는 목표 자신의
  z(물체의 중심, 장소의 가장 낮은 지오메트리의 중심)에 더하고, `height`는 그것을 대체한다.
- 로봇은 `SO-101`만 지원한다(`so101_ik`). 다른 값은 이름을 대며 거부한다. 사람이 바꿀 일이 아닌
  튜닝 — `pos_tol`, `grip_open`/`grip_closed` 각도, 속도 — 은 코드에 남고(`demo_cfg`의 값), 파일은
  *무엇을* 할지만 말한다.
- `templates/teach/so101-pick-place.toml`은 오늘의 일곱 단계를 블록으로 옮긴 것이다: 접근(물체,
  위 0.045, -85°, 열림), 하강(물체, 위 -0.005, -85°, 열림), 닫기(닫힘, 5초), 들기(물체, 높이
  0.14, -45°, 닫힘), 운반(통, 높이 0.14, -45°, 닫힘), 내리기(통, 높이 0.06, -85°, 닫힘), 놓기(열림,
  5초). `--expert so101-pick-place`는 그 파일을 실행하므로(`include_str!`로 컴파일해 넣는다) 시범의
  숫자 출처는 하나이다.

**오라클.**
1. 첫 커밋, 어떤 리팩터링보다 먼저: 무시 처리된 생성기 `generate_expert_chunks_golden`(과 그
   확인 테스트)이 데모 장면에서 오늘의 `ScriptedExpert`를 결정론적인 **완벽한 추종자** 상태열로
   구동한다 — 관절은 각 청크의 마지막으로 실행된 행을 따르고, 큐브는 그려진 자리에 머문다. 고정한
   큐브 위치 셋과 에피소드 전체 길이에 대해 모든 청크의 `f64` 비트를
   `tests/golden/expert/so101-pick-place-chunks.json`에 쓴다. Python 없음.
2. 리팩터링의 커밋은 그 골든을 바이트 단위로 그대로 둔다(`verify-goldens`에 변경 없음).
3. 단위 테스트: 형식(`above`와 `height` 동시 지정, 알 수 없는 키, 알 수 없는 장소, 정수배가 아닌
   `wait`, 빈 프로그램을 이름을 대며 거부), `bin`의 장소 점이 `(0.14, -0.1)`과 정확히 같음,
   커밋된 파일이 파싱되어 컴파일해 넣은 내장본과 같음.
4. Python 게이트(`ES_PYTHON`이 없으면 건너뜀): 기존 `expert_solves_the_pinned_seeds`.

### 작업 Q2: `--expert <program.toml>`

**Files:** `crates/es/src/cmd/loop.rs`, `crates/es/src/cmd/eval.rs`,
`crates/es-data/src/collect.rs`, `crates/es-data/src/training.rs`(사이클의 `[collect] expert`가
경로를 그대로 넘긴다), `crates/es/tests/cli.rs`, `tests/golden/train/**`(생성기로만).

- `es loop collect`와 `es eval run`(시범기 게이트)은 `--expert <name | path.toml>`을 받는다.
  존재하는 `.toml` 파일을 가리키는 값은 프로그램이고, 그 밖은 내장 이름(오늘은 `so101-pick-place`
  뿐)이다. 두 명령을 위한 해석기는 하나이다.
- 수집 원장 단계는 `expert`(이름, 또는 준 그대로의 경로)를 유지하고 `expert_program` = 프로그램
  바이트의 blake3(이름이면 내장본의 바이트)를 더한다.
- `Cycle`의 `[collect] expert`는 경로일 수 있다. 드라이런 계획은 그것을 그대로 출력한다. 커밋된
  사이클의 골든은 움직이지 않는다.

**오라클.** 단위: 해석기(이름, 파일, 없는 파일, 파싱되지 않는 파일), 원장 필드. CLI: `es eval run
--expert templates/teach/so101-pick-place.toml`이 게이트의 인자 검사를 통과한다. Python 게이트:
`es loop collect --episodes 2 --expert <커밋된 파일>`이 `--expert so101-pick-place`와 같은 데이터셋
`content`를 쓴다.

### 작업 Q3: 가르치기 모델(`es-editor-model`)

**Files:** 새 `crates/es-editor-model/src/model/teach.rs`,
`model/{mod.rs, project.rs, template.rs, labels.rs, workflow.rs}`,
`crates/es-editor-model/i18n/*`, `templates/*.toml`(`teach` 필드).

- 템플릿은 자기 프로그램을 이름으로 부른다(`teach = "templates/teach/so101-pick-place.toml"`).
  `Project::create`는 그것을 `<project>/teach.toml`로 복사하고, `write_run`은 `[collect] expert`를
  그 파일로 정한다. S3 이전에 만든 프로젝트(`teach.toml` 없음)는 템플릿의 내장 이름을 유지한다.
- `Teach`(② 상태): 프로그램, 변경 표시, 불러오기/저장(`teach.toml`, 원자적으로 기록), 템플릿의
  것으로 되돌리기. 선택한 블록 뒤에 이동 블록이나 집게 블록 추가, 삭제, 위/아래로 옮기기, 필드
  하나 편집. 편집마다 다시 검증한다.
- 각 블록의 필드로 만든 쉬운 말(i18n, 두 언어): 예: ko "[큐브] 위 4.5 cm로 이동 · 손목 -85° ·
  집게 열림", "집게 닫기 · 5초 기다리기". 물체와 장소는 템플릿의 `[outcome]` 이름이 맞으면 그
  이름으로 부른다(Z6의 `outcome.cube`, `outcome.bin`).
- 블록별 검증, 각각 i18n 키 하나: Task IR이 그릴 수 있는 어떤 물체 위치에서 닿지 못함(큐브의
  `Randomization` 범위 꼭짓점에서의 IK, 장소 목표라면 그 장소에서의 IK); SO-101이 그 자리에서
  유지할 수 없는 피치; 이동이 아닌 첫 블록; 집게를 닫는 블록이 없음; 닫은 뒤에 여는 블록이 없음.
  경고는 저장이나 해 보기를 막지 않으며, 블록 위에 표시된다.
- "한 번 해 보기": 파생된 Evaluation IR(공칭 스위트, 명시적 시드 하나, acceptance 없음 — Z1의
  `preview_evaluation`이 하나를 파생하는 것과 같다)로 `es eval run --expert <teach.toml>`의
  argv를 만들어 `<project>/try/<n>/`에 쓰고, 결과를 다시 읽는다(성공 여부, 재생할 셀). 🎲 "다른
  위치로"는 다음 시드를 고른다. ②는 실행이 도는 동안 해 보기를 거부한다(`may_start`).

**오라클.** 헤드리스 테스트: 만들 때 복사되는 템플릿의 프로그램; `write_run`의 레시피가 그것을
가리킴; 각 편집 연산; 모든 블록 모양의 레이블이 두 언어에서 `{}`가 남지 않음; 만든 프로그램마다의
검증(닿지 못하는 `above`, 닫기 없음, 첫 블록이 집게); 해 보기 argv와 파생 IR의 파싱(IR은
`es_ir::serial`로, 스위트 하나, 시드 하나, acceptance 없음); 시험용 해 보기 폴더 읽기.

### 작업 Q4: ② 가르치기 화면(`es-editor`)

**Files:** 새 `crates/es-editor/src/ui/teach.rs`, `crates/es-editor/src/ui/{mod.rs, shell.rs}`,
`crates/es-editor/src/app.rs`(훅만).

왼쪽: 블록 목록 — 블록 하나가 쉬운 말 한 행이고 경고 표시가 붙는다. 선택하면 편집, 추가(이동 /
집게), 삭제, 위, 아래, "되돌리기"(템플릿의 프로그램), 저장. 오른쪽(인스펙터): 선택한 블록의 필드 —
목표(물체 / 장소 / 점), 센티미터 슬라이더가 달린 위(above) 또는 높이(height), 도(degree) 단위
손목 각도, 집게 열림/닫힘, 초 단위 기다림. 가운데: "한 번 해 보기"와 "🎲 다른 위치로", 그리고
해 보기의 셀을 보여 주는 공유 재생기(`ui/player.rs`)와 "성공" / "실패 — <원인>"(Z4a의 결과 분류가
다른 실행과 마찬가지로 해 보기에도 적용된다). 에디터가 프로그램을 열 수 없는 템플릿(S3 이전
프로젝트)이면 ②는 이유를 보이며 읽기 전용으로 남는다.

**오라클.** 빌드, clippy, 테스트; 경고가 있는 ②와 해 본 뒤 ②의 스크린샷(앱 내 훅, 커밋하지 않음).

### 작업 Q5: 실제 실행과 문서(오케스트레이터)

조용한 PC에서: 힌트 카드로 프로젝트를 만들고, "큐브 위로 이동"을 4.5 cm에서 6 cm로 바꾸고, 한 번
해 보고, ③에서 짧은 실행 한 번. 수치(해 보기의 시간, 실행의 시범 수와 그 성공, 판정)는
`docs/packets/M14/QV-verification.md`에 적는다. 설계 노트의 S3 절은 만들어진 것으로 다시 쓰고,
리뷰를 한다.
