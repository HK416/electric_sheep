# M7 E6 — 전문가가 아닌 사람을 위한 에디터: 쉬운 말, 홈 화면, 네이티브 파일 대화상자, 한글을 렌더링하는 폰트

스펙: §23.1(에디터는 클라이언트다), §23.2("모든 층을 한 화면에서 본다" — 엔지니어만이
아니라 사람을 위해), §23.3(실행 도중 무엇이 보이는가), §13.1(루프가 워크플로다: 수집 →
학습 → 평가 → 관찰), §28.10 규칙 3(`app.rs`에서는 아무것도 결정되지 않는다; 헤드리스
뷰모델이 먼저다; 화면 없는 CI가 판단할 수 없는 것은 구현이 아니라 설계다), §1.4. 오너
지침 2026-09-21: *에디터는 도메인 지식이 없는 사람에게 너무 어렵다; UI/UX를 개선하고,
다국어 텍스트가 깨지지 않도록 폰트를 고르라.* 설계 노트: `docs/design/editor-shell.md`(+
`.ko.md`) 15절. **E1–E5**에 있는 그대로 의존한다.

## 질문

오늘날 에디터의 모든 화면은 그것을 쓴 엔지니어에게 말을 건다: "bundle.esb, 다섯 개의
.toml 파일이 있는 디렉터리, 또는 실행"을 원하는 맨 텍스트 필드, 레이블이 `--config` /
`--policy` / `--out`인 실행 패널, `envelope_violation_rate`와 `failure_mode_histogram`으로
머리글을 단 결과 표, "(spec 23.3)"을 인용하는 도움말 텍스트, 그리고 한글도 가나도 한자도
없는 기본 폰트 — 그래서 한국어 경로나 레이블은 네모 상자로 렌더링된다. **같은 뷰모델을
쉬운 말로, 읽는 사람의 언어로, 기술적 이름은 호버 한 번 거리에 두고 보여줄 수 있는가 —
그리고 그것을 헤드리스로 판단할 수 있는가?**

## 사양

* **문자열은 `app.rs`가 아니라 표 안에 산다.** `crates/es-editor/i18n/en.toml`과
  `ko.toml`(`key = "text"`, 점으로 구분된 접두사로 묶인 평평한 키들:
  `home.open_project`, `launch.policy`, `metric.success_rate`, …), `include_str!`로
  `model/i18n.rs`에 로드된다: `Lang { En, Ko }`, `Strings::get(lang) -> &Strings`,
  `t(key) -> &str`. 두 표 중 한쪽에라도 키가 없으면 **테스트 실패**이고, 표에는 있지만
  크레이트가 한 번도 쓰지 않는 키도 마찬가지다(테스트는 크레이트 자신의 소스를
  `t("…")`로 그렙한다). `Lang`은 최근 목록(E3) 옆 `eframe::Storage`에 영속화되고 상단
  바에서 토글된다(`English` / `한국어`); 기본값은 `En`이다. 한국어 텍스트는 정확히 이
  두 파일에서만 허용된다: `.githooks/pre-commit`의 `is_doc`이 `*/i18n/*.toml` 케이스를
  얻고, CLAUDE.md의 Conventions 문단이 한 문장으로 그것을 말한다.
* **CJK를 렌더링하는 폰트.** `model/fonts.rs::system_cjk_font() -> Option<(String,
  Vec<u8>)>`은 OS별로 고정된 후보 목록을 탐색한다 — Windows
  `C:\Windows\Fonts\malgun.ttf`(맑은 고딕), `msyh.ttc`, `meiryo.ttc`; macOS
  `/System/Library/Fonts/AppleSDGothicNeo.ttc`, `/System/Library/Fonts/PingFang.ttc`,
  `/System/Library/Fonts/Supplemental/NotoSansCJK*.ttc`; Linux
  `/usr/share/fonts/**/NotoSansCJK*.{ttc,otf}`, `NanumGothic.ttf`,
  `DroidSansFallback*.ttf` — 그리고 존재하는 첫 번째 것을 반환한다(폰트 탐색 의존성
  없음: 목록과 `std::fs`뿐). `main.rs`는 그것을 `FontFamily::Proportional`과
  `FontFamily::Monospace` 둘 다의 **마지막 폴백**으로 밀어 넣는다, 그래서 라틴 문자는
  egui 자신의 글리프를 유지하고 CJK는 시스템 폰트로 떨어진다; 아무것도 찾지 못하면
  상태 바가 어느 경로들을 시도했는지와 어느 패키지가 하나를 설치하는지를 말한다
  (Debian/Ubuntu에서는 `fonts-noto-cjk`). 커밋되는 폰트 파일은 없다. 기본 텍스트
  스타일은 본문 15px / 제목 20px로 올라가고 **Text size** 설정(`S / M / L`, 영속화됨)이
  그것들 전부를 스케일한다 — 13px를 읽을 수 없는 사람이 egui를 알아야 할 필요는 없다.
* **홈 화면.** 아무것도 열려 있지 않을 때, Graph 탭의 빈 상태는 홈이 된다: 세 개의 큰
  버튼 — *Open a project file…*, *Open a run result…*, *Recent* — 그리고 §13.1 순서의
  다섯 단계 스트립, 각각 쉬운 말 한 문장(`home.step.design`, `home.step.collect`,
  `home.step.train`, `home.step.evaluate`, `home.step.watch`)과 그것을 하는 탭이나
  패널로 가는 버튼을 가진다. 탭 이름은 워크플로의 단어가 된다 — `Design`, `Results`,
  `Live`, `What the policy sees`, `Problems`, `Edit` — 옛 이름은 호버 텍스트 안에 둔다.
* **네이티브 파일 대화상자.** 실행 패널과 리플레이 패널의 모든 경로 필드 옆에
  *Open…*, *Browse…*를, `rfd`를 통해 — **하나뿐인 새 의존성**, MIT/Apache, Windows와
  macOS에서 네이티브다. 타깃별 의존성으로 선언된다
  (`[target.'cfg(any(windows, target_os = "macos"))'.dependencies]`) 그 타깃들에서만
  기본값인 `file-dialogs` 피처 뒤에서, 그래서 Linux는 타이핑한 경로와 드래그드롭을
  유지하고 GTK 요구사항을 하나도 얻지 않는다; `model/dialogs.rs`는 그것을
  `pick_file(filter) -> Option<PathBuf>` / `pick_dir()`로 감싸며 `cfg`로 꺼졌을 때는
  `None`을 반환하는 스텁을 가진다, 그리고 `app.rs`는 결코 `rfd`를 이름으로 부르지
  않는다.
* **쉬운 말, 기술적 이름은 호버 한 번 거리.** `model/labels.rs`:
  `metric_label(&MetricSpec)`(`success_rate` → *Success rate*;
  `envelope_violation_rate` → *Safety limit hits*; `failure_mode_histogram` →
  *Failure causes*; `episode_length` → *Episode length*; …)는 `MetricSpec::ALL`을
  빠짐없이 다룬다(와일드카드 갈래는 테스트 실패); `launch_label(LaunchField)`
  (`--config` → *Evaluation settings*, `--policy` → *Policy file*, `--scene` →
  *Scene file*, `--out` → *Output folder*, `--frames` → *Also save frames*,
  `--jobs` → *Parallel workers*, `--telemetry` → *Watch live at*, …)는
  `LaunchField`/`LaunchFlag`를 빠짐없이 다룬다; 모든 표 머리글, 실행 레이블, 탭이
  쉬운 말을 보여주고 원래 이름/플래그를 호버한다. **화면에 보이는 어떤 문자열도
  스펙 절을 인용하지 않는다**: 모든 "(spec N.N)"은 호버로 옮겨진다; i18n 테스트는
  `*.hint` 밖의 어떤 키도 "spec "을 담지 않음을 단언한다. 모든 탭의 빈 상태는 무엇을
  열지와 Open 버튼이 어디 있는지를 한 문장으로 말한다.
* **여기 없음.** `es` 자신의 CLI 출력을 현지화하는 것(실행 로그는 그것을 그대로
  보여준다); 레이어드 그래프의 재설계; egui의 텍스트를 넘어서는 아이콘; `en`/`ko`를
  넘어서는 번역(하나 추가하는 것은 파일 하나다); 저장소에 번들된 폰트.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

```
crates/es-editor/src/app.rs
crates/es-editor/src/main.rs
crates/es-editor/src/lib.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/model/i18n.rs
crates/es-editor/src/model/fonts.rs
crates/es-editor/src/model/labels.rs
crates/es-editor/src/model/dialogs.rs
crates/es-editor/src/model/launch.rs
crates/es-editor/src/model/recent.rs
crates/es-editor/Cargo.toml
crates/es-editor/i18n/en.toml
crates/es-editor/i18n/ko.toml
crates/es-editor/tests/**
Cargo.toml
Cargo.lock
.githooks/pre-commit
CLAUDE.md
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E6-plain-language-editor.md
docs/packets/M7/E6-plain-language-editor.ko.md
```

`model/{i18n,fonts,labels,dialogs}.rs`(신규; 각각 자신의 테스트와 함께), `launch.rs`(표를
통한 레이블; 그 argv 안의 아무것도 바뀌지 않는다 — 골든은 그대로 남는다), `recent.rs`(목록
옆의 영속화된 두 설정), `app.rs`/`main.rs`(배선, 폰트, 홈 화면), `Cargo.toml`(루트:
`[workspace.dependencies]`에 고정된 `rfd`; 크레이트: 타깃별 의존성), 두 문자열 표, 훅의
한 줄짜리 허용 목록, CLAUDE.md의 한 문장, 설계 노트, 이 패킷.

## 오라클

1. `cargo test -p es-editor i18n_tables_are_complete_and_used` — `en.toml`의 모든 키가
   `ko.toml`에 있고 그 반대도 마찬가지다; 크레이트 소스 안의 모든 `t("…")` 키가
   존재한다; 표 안의 모든 키가 쓰인다; `hint`가 아닌 어떤 값도 "spec "을 담지 않는다.
2. `cargo test -p es-editor metric_and_launch_labels_are_total` — `metric_label`이
   `MetricSpec::ALL`을 커버하고, `launch_label`이 모든 `LaunchField`와 `LaunchFlag`를
   커버하며, 레이블들은 두 언어 모두에서 쌍마다 서로 다르다.
3. `cargo test -p es-editor system_cjk_font_is_found_or_the_reason_is_named` — 후보 중
   하나를 가진 머신에서는 로더가 그 바이트와 이름을 반환한다(이 Windows 박스는
   `malgun.ttf`를 가지고 있다); 그렇지 않으면 반환된 이유가 시도된 모든 경로를
   나열한다. 더해서 `font_fallback_is_last`: `fonts::install`이 짓는 `FontDefinitions`는
   egui의 폰트를 먼저 유지한다.
4. `cargo test -p es-editor launch_argv_is_the_golden` 변경 없음 — 레이블은 바뀌었지만
   argv는 바뀌지 않았다.
5. `cargo test -p es-editor home_screen_lists_the_five_steps_in_loop_order` — 헤드리스
   홈 모델이 §13.1 순서의 다섯 단계를 두 언어 모두에서 비어 있지 않은 문장과 함께
   산출한다.
6. `cargo build -p es-editor`(Windows, dialogs 포함) **그리고** `cargo build -p
   es-editor --no-default-features`(Linux 형태); `cargo xtask ci`(레이어링, 예산:
   `es-editor`는 6,000 소스 줄 아래에 머문다 — 숫자를 말하라); `cargo xtask check-scope
   docs/packets/M7/E6-plain-language-editor.md`.

## 수용 기준

오라클 1–6. 워크트리의 `target/plan-u/e6/` 아래에 **두** 언어 모두로 된 스크린샷: 홈
화면; 한국어 레이블과 필드에 타이핑된 한국어 경로를 가진 실행 패널(글리프가 렌더링됨,
네모 상자 없음); 사람이 읽을 수 있게 된 머리글과 원래 메트릭 이름을 보여주는 호버를 가진
Results 탭; `L`로 설정된 text-size 설정. 오케스트레이터가 그 확인을 손으로 반복한다.
노트의 15절이 문자열 표 규칙, 폰트 규칙(시스템 폴백, 번들된 것 없음, CJK 폰트가 없는
머신이 보는 것)과 레이블 표를 기록한다.

## 금지

어떤 뷰모델의 데이터나 어떤 argv 골든을 바꾸는 것; 저장소 안의 폰트 파일; 폰트 탐색
크레이트; `app.rs`에서 직접 또는 Linux에서 기본으로 닿을 수 있는 `rfd`; `app.rs`에서
레이블을 결정하는 것; `crates/es/**`, `crates/es-telemetry/**`, `crates/es-eval/**`;
`docs/ARCHITECTURE*.md`. INV-17: 새 트레이트 없음.
