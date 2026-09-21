# M7 E6 — 전문가가 아닌 사람을 위한 에디터: 쉬운 말, 첫 화면, 네이티브 파일 대화상자, 한국어가 나오는 글꼴

명세: §23.1(에디터는 클라이언트다), §23.2("모든 계층을 한 화면에서" — 엔지니어만이 아니라 사람을
위해), §23.3(실행 중에 보이는 것), §13.1(순환이 곧 작업 흐름이다: 수집 → 학습 → 평가 → 관찰),
§28.10 규칙 3(`app.rs`에서는 아무것도 결정하지 않는다. 헤드리스 뷰모델이 먼저이고, 화면 없는 CI가
판정할 수 없는 것은 구현이 아니라 설계다), §1.4. 2026-09-21 오너 지시: *도메인 지식이 없는
사람에게 에디터가 너무 어렵다. UI/UX를 개선하고, 다국어 텍스트가 깨지지 않도록 글꼴을 고르라.*
설계 노트: `docs/design/editor-shell.md`(+ `.ko.md`) §15. **E1–E5**를 있는 그대로 의존한다.

## 질문

오늘 에디터의 모든 화면은 그것을 쓴 엔지니어에게 말을 건다. "bundle.esb, 다섯 개의 .toml이 든
폴더, 또는 실행"을 달라는 맨 입력칸, 라벨이 `--config` / `--policy` / `--out`인 시작 패널,
`envelope_violation_rate`와 `failure_mode_histogram`이 머리글인 결과 표, "(spec 23.3)"을 인용하는
도움말, 그리고 한글·가나·한자 글리프가 없는 기본 글꼴 — 그래서 한국어 경로나 라벨은 네모로
보인다. **같은 뷰모델을 쉬운 말로, 읽는 사람의 언어로, 기술적 이름은 호버 하나 거리에 두고 보여
줄 수 있는가 — 그리고 그것을 헤드리스로 판정할 수 있는가?**

## 명세

* **문구는 `app.rs`가 아니라 표에 산다.** `crates/es-editor/i18n/en.toml`과 `ko.toml`
  (`key = "text"`, 점 접두사로 묶인 평평한 키: `home.open_project`, `launch.policy`,
  `metric.success_rate`, …). `include_str!`로 `model/i18n.rs`에 들어간다: `Lang { En, Ko }`,
  `Strings::get(lang) -> &Strings`, `t(key) -> &str`. 어느 한쪽 표에 없는 키는 **테스트 실패**이고,
  표에 있는데 크레이트가 쓰지 않는 키도 마찬가지다(테스트가 크레이트 자신의 소스에서 `t("…")`를
  훑는다). `Lang`은 최근 목록(E3) 옆에서 `eframe::Storage`에 보존되고 위쪽 막대에서 전환한다
  (`English` / `한국어`). 기본값은 `En`. 한국어는 정확히 이 두 파일에만 허용된다:
  `.githooks/pre-commit`의 `is_doc`에 `*/i18n/*.toml` 경우가 추가되고, CLAUDE.md의 Conventions
  문단이 그것을 한 문장으로 적는다.
* **CJK를 그리는 글꼴.** `model/fonts.rs::system_cjk_font() -> Option<(String, Vec<u8>)>`이 고정된
  OS별 후보 목록을 훑는다 — Windows `C:\Windows\Fonts\malgun.ttf`(맑은 고딕), `msyh.ttc`,
  `meiryo.ttc`; macOS `/System/Library/Fonts/AppleSDGothicNeo.ttc`,
  `/System/Library/Fonts/PingFang.ttc`, `/System/Library/Fonts/Supplemental/NotoSansCJK*.ttc`;
  Linux `/usr/share/fonts/**/NotoSansCJK*.{ttc,otf}`, `NanumGothic.ttf`,
  `DroidSansFallback*.ttf` — 그리고 처음 존재하는 것을 돌려준다(글꼴 탐색 의존성 없이 목록과
  `std::fs`만). `main.rs`가 그것을 `FontFamily::Proportional`과 `FontFamily::Monospace` 양쪽의
  **마지막 대체**로 밀어 넣어, 라틴은 egui 자신의 글리프를 유지하고 CJK만 시스템 글꼴로 떨어진다.
  하나도 없으면 상태 표시줄이 어떤 경로를 찾아봤는지와 어느 패키지가 하나 설치해 주는지를 말한다
  (데비안·우분투는 `fonts-noto-cjk`). 글꼴 파일은 커밋하지 않는다. 기본 텍스트 스타일은 본문
  15 px / 제목 20 px로 올리고, **글자 크기** 설정(`S / M / L`, 보존됨)이 그 전부를 함께 키운다 —
  13 px를 읽지 못하는 사람이 egui를 알아야 할 이유는 없다.
* **첫 화면.** 아무것도 열지 않았을 때 설계 탭의 빈 상태가 첫 화면이 된다: 큰 버튼 셋 —
  *프로젝트 파일 열기…*, *실행 결과 열기…*, *최근 항목* — 과 §13.1 순서의 다섯 단계 띠. 각
  단계는 쉬운 말 한 문장(`home.step.design`, `home.step.collect`, `home.step.train`,
  `home.step.evaluate`, `home.step.watch`)과 그 일을 하는 탭이나 패널로 가는 버튼을 갖는다. 탭
  이름은 작업의 낱말이 된다 — `설계`, `결과`, `관찰`, `정책이 보는 것`, `문제`, `고치기` — 옛
  이름은 호버에 둔다.
* **네이티브 파일 대화상자.** 시작 패널과 다시 보기 패널의 모든 경로 입력칸 옆에 *열기…*,
  *찾아보기…*. `rfd`를 통하며 — **새 의존성은 이 하나뿐** — MIT/Apache, Windows와 macOS에서
  네이티브다. 타깃별 의존성(`[target.'cfg(any(windows, target_os = "macos"))'.dependencies]`)으로
  선언하고 `file-dialogs` 기능 뒤에 둔다. 그 기능은 그 두 타깃에서만 기본이라, 리눅스는 직접 적는
  경로와 끌어다 놓기를 유지하고 GTK 요구 사항을 얻지 않는다. `model/dialogs.rs`가
  `pick_file(filter) -> Option<PathBuf>` / `pick_dir()`로 감싸고 `cfg`가 꺼진 쪽은 `None`을
  돌려주는 스텁이며, `app.rs`는 `rfd`를 이름조차 부르지 않는다.
* **쉬운 말, 기술적 이름은 호버 하나 거리에.** `model/labels.rs`: `metric_label(&MetricSpec)`
  (`success_rate` → *성공률*; `envelope_violation_rate` → *안전 한계 위반*;
  `failure_mode_histogram` → *실패 원인*; `episode_length` → *에피소드 길이*; …)은
  `MetricSpec::ALL`에 대해 전역적이다(와일드카드 갈래는 테스트 실패). `launch_label(LaunchField)`
  (`--config` → *평가 설정*, `--policy` → *정책 파일*, `--scene` → *장면 파일*, `--out` →
  *결과 폴더*, `--frames` → *사진도 저장할 폴더*, `--jobs` → *동시에 돌릴 개수*, `--telemetry` →
  *실시간으로 볼 주소*, …)는 `LaunchField`/`LaunchFlag`에 대해 전역적이다. 모든 표 머리글, 시작
  라벨, 탭이 쉬운 말을 보이고 원시 이름·플래그를 호버로 단다. **보이는 문자열은 명세 절을
  인용하지 않는다**: 모든 "(spec N.N)"은 호버로 옮기고, i18n 테스트가 `*.hint`가 아닌 키의 값에
  "spec "이 없음을 단언한다. 모든 탭의 빈 상태가 무엇을 열어야 하는지와 열기 버튼이 어디에
  있는지를 한 문장으로 말한다.
* **여기 없는 것.** `es` 자신의 CLI 출력 현지화(시작 로그는 그대로 보여 준다), 계층 그래프
  재설계, egui의 텍스트를 넘는 아이콘, `en`/`ko` 밖의 번역(하나 더하는 것은 파일 하나다),
  저장소에 동봉하는 글꼴.

## 문맥

`cargo xtask check-scope`가 읽는 글롭:

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

`model/{i18n,fonts,labels,dialogs}.rs`(새로 만들고 각자 테스트를 갖는다), `launch.rs`(라벨은 표를
거친다. argv는 아무것도 바뀌지 않으므로 골든은 그대로), `recent.rs`(목록 옆의 보존 설정 둘),
`app.rs`/`main.rs`(배선, 글꼴, 첫 화면), `Cargo.toml`(루트: `[workspace.dependencies]`에 `rfd`
고정. 크레이트: 타깃별 의존성), 두 문자열 표, 훅의 한 줄 허용, CLAUDE.md의 한 문장, 설계 노트,
이 패킷.

## 오라클

1. `cargo test -p es-editor i18n_tables_are_complete_and_used` — `en.toml`의 모든 키가 `ko.toml`에
   있고 그 반대도 그렇다. 크레이트 소스의 모든 `t("…")` 키가 존재한다. 표의 모든 키가 쓰인다.
   `hint`가 아닌 값에 "spec "이 없다.
2. `cargo test -p es-editor metric_and_launch_labels_are_total` — `metric_label`이
   `MetricSpec::ALL`을 덮고, `launch_label`이 모든 `LaunchField`와 `LaunchFlag`를 덮으며, 라벨은
   두 언어 모두에서 서로 다르다.
3. `cargo test -p es-editor system_cjk_font_is_found_or_the_reason_is_named` — 후보 하나를 가진
   기계에서는 로더가 그 바이트와 이름을 돌려준다(이 Windows 상자에는 `malgun.ttf`가 있다).
   아니면 돌려주는 이유가 찾아본 모든 경로를 나열한다. 그리고 `font_fallback_is_last`:
   `fonts::install`이 만든 `FontDefinitions`가 egui의 글꼴을 앞에 유지한다.
4. `cargo test -p es-editor launch_argv_is_the_golden` 그대로 — 라벨은 바뀌었고 argv는 아니다.
5. `cargo test -p es-editor home_screen_lists_the_five_steps_in_loop_order` — 헤드리스 첫 화면
   모델이 §13.1 순서로 다섯 단계를 내놓고, 두 언어 모두에서 각 단계에 비지 않은 문장이 있다.
6. `cargo build -p es-editor`(Windows, 대화상자 포함) **그리고** `cargo build -p es-editor
   --no-default-features`(리눅스 모양). `cargo xtask ci`(계층, 예산: `es-editor`가 소스 6,000줄
   아래를 유지한다 — 숫자를 말할 것),
   `cargo xtask check-scope docs/packets/M7/E6-plain-language-editor.md`.

## 수락

오라클 1–6. 워크트리의 `target/plan-u/e6/` 아래에 **두 언어** 스크린샷: 첫 화면, 한국어 라벨과
입력칸에 적힌 한국어 경로가 있는 시작 패널(글리프가 그려지고 네모가 없을 것), 쉬운 머리글과
원시 지표 이름을 보여 주는 호버가 있는 결과 탭, 글자 크기 `L`. 오케스트레이터가 손으로 다시
확인한다. 노트 §15가 문자열 표 규칙, 글꼴 규칙(시스템 대체, 동봉 없음, CJK 글꼴이 없는 기계가
보는 것)과 라벨 표를 기록한다.

## 금지

뷰모델의 데이터나 argv 골든을 바꾸는 것. 저장소 안의 글꼴 파일. 글꼴 탐색 크레이트. `app.rs`에서
직접 닿거나 리눅스에서 기본으로 켜지는 `rfd`. `app.rs`에서 라벨을 결정하는 것.
`crates/es/**`, `crates/es-telemetry/**`, `crates/es-eval/**`. `docs/ARCHITECTURE*.md`.
INV-17: 새 트레이트 금지.
