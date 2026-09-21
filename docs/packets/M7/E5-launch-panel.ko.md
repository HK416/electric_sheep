# M7 E5 — 실행 패널: 에디터가 `es eval run` / `es train`을 시작하고 붙는다

스펙: §23.1(**에디터는 학습을 호스트하지 않는다; 그것은 실행 중인 프로세스의 클라이언트다**),
§23.3("실행 도중 무엇이 보이는가"; 컨트롤: pause/step/reset은 이 패킷이 *아니다* — 그것들은
실행이 말하지 않는 프로토콜을 필요로 한다), §13.1(각 단계는 명령과 아티팩트다), §28.10 규칙
3(`app.rs`에서는 아무것도 결정되지 않는다)과 (E5). 설계 노트: `docs/design/editor-shell.md`
(+ `.ko.md`) 14절. **E4**(`es eval run --telemetry`, `es-editor --attach`, `LiveRun`)와
**T1**/**T2**(`es train --recipe`, `es loop cycle --recipe`)에 있는 그대로 의존한다.

## 질문

E4 이후 사람은 실행을 지켜볼 수 있다 — 터미널에서 그것을 시작하고 주소를 두 번 타이핑했다면.
**에디터가 이미 아는 것(열린 번들이나 실행, 레시피 경로, Evaluation IR 경로)으로부터 명령줄을
지을 수 있고, 그 의미를 소유하지 않는 자식 프로세스로 그것을 시작할 수 있고, 그 종료 코드와
마지막 줄들을 보여줄 수 있고, 자신이 요청한 텔레메트리에 스스로 붙을 수 있는가 — 클라이언트로
남으면서(§23.1) `app.rs`에서는 아무것도 결정하지 않으면서?**

## 사양

* **`model/launch.rs`** — `LaunchModel { kind: Eval | Train | Cycle, fields }`, 여기서 필드는
  모델이 렌더링하는 세 명령의 플래그와 정확히 같다(`es eval run --config --policy --scene
  --out [--frames] [--jobs] --telemetry <addr>`, `es train --recipe --out`, `es loop cycle
  --recipe --out`), 가능할 때는 세션으로부터 미리 채워진다(`--policy`는 열린 번들의 경로에서,
  `--out`은 실행 디렉터리의 부모에서, `--telemetry` 주소는 Telemetry 탭의 attach 필드에서,
  기본값 `127.0.0.1:7777`). `argv() -> Vec<String>`은 필드의 순수 함수다; `es_binary()`는
  **하나의 규칙으로** `es` 실행 파일을 해석한다 — 설정되어 있으면 `ES_BIN`, 아니면 에디터
  자신의 실행 파일 옆의 `es`, 아니면 `PATH` 위의 `es` — 그리고 어느 것을 찾았는지 말한다.
* **`spawn()`**은 `argv()`로 `std::process::Command`를 시작하고, stdout/stderr는 `mpsc` 채널을
  통해 마지막 200줄의 경계 있는 링을 먹이는 리더 스레드로 파이프된다; `poll()`은 채널을
  비우고 자식을 `try_wait()`한다; `state()`는 `Idle | Running { pid, since } | Exited { code,
  lines } | Failed(String)`이다. `kill()`은 그것을 시작한 사람을 위해 존재한다 — 그것이
  유일한 컨트롤인데, 실행이 어떤 컨트롤 프로토콜도 말하지 않기 때문이다(§23.3의 pause/step은
  여기에 없는 것으로 나열되어 있다). UI 스레드에서 결코 블로킹하지 않는다; `tokio` 없음, 새
  의존성 없음.
* **Attach는 launch를 따른다.** 시작된 명령이 `--telemetry <addr>`를 담고 있으면, 모델은
  `attach: Some(addr)`를 보고하고 앱은 자식이 실행 중인 *뒤에* 그것을 E4의 attach 경로로
  건넨다 — 에디터는 마치 사람이 주소를 타이핑한 것과 정확히 같이 클라이언트로 연결한다.
  종료 코드는 CLI가 문서화하는 의미로 보인다(0 pass, 1 fail/runtime, 2 usage, 3 skipped);
  매핑 표는 패널이 아니라 모델 안에 산다.
* **패널**(`app.rs`, 배선뿐): Run 탭 안의 `Launch` 절 — 세 종류를 콤보로, 필드를 텍스트
  입력으로, 렌더링된 명령줄을 읽기 전용으로, Start / Kill, 상태 줄, 스크롤 영역 안의 마지막
  줄들. 탭 안의 다른 것은 아무것도 바뀌지 않는다.
* **여기 없음.** Pause/step/reset/hot-patch(§23.3) — 프로토콜 없음; job 큐; 원격 호스트;
  `ES_BIN`을 넘어서는 환경 편집. 노트에 "여기 없음" 아래 나열된다.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

```
crates/es-editor/src/model/launch.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/app.rs
crates/es-editor/src/lib.rs
crates/es-editor/tests/**
tests/golden/editor/launch-*.txt
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E5-launch-panel.md
docs/packets/M7/E5-launch-panel.ko.md
```

`model/launch.rs`(신규; 모델 전체와 그 테스트), `model/mod.rs`(export), `app.rs`(그 절),
`lib.rs`(재-export가 필요하다면), `tests/golden/editor/launch-{eval,train,cycle}.txt`(픽스처
입력의 렌더링된 argv, 한 줄에 하나씩, 한 번 생성됨), 설계 노트, 이 패킷. **`Cargo.toml` 변경
없음**: `std::process`와 `std::sync::mpsc`로 충분하다.

## 오라클

1. `cargo test -p es-editor launch_argv_is_the_golden` — 고정된 필드로부터의 세 종류가 세
   골든과 바이트 단위로 같은 argv를 렌더링한다; 설정되지 않았을 때 `--frames`/`--jobs`는
   없다; 모든 경로가 그대로 전달된다(CLI가 하지 않는 정규화는 없음).
2. `cargo test -p es-editor launch_reports_the_exit_code` — 같은 `spawn` 경로를 통한 플랫폼
   셸의 `spawn()`(Windows에서는 `cmd /C exit 3`, 그 외에서는 `sh -c "exit 3"`; 프로그램이
   파라미터다)이 한정된 수의 `poll()` 안에 `Exited { code: 3 }`에 도달한다; stdout 줄들
   (`echo`)이 순서대로 도착한다; 잠든 자식에 대한 `kill()`은 0이 아닌 코드로 `Exited`에
   도달한다.
3. `cargo test -p es-editor launch_attaches_only_after_running` — `attach()`는 `Idle`인 동안은
   `None`이고, `--telemetry`를 담은 명령에 대해서는 `Running`이 되면 `Some(addr)`, 담지 않은
   것에 대해서는 `None`이다.
4. `cargo test -p es-editor es_binary_resolution_is_one_rule` — `ES_BIN`이 이기고, 그다음
   형제 파일, 그다음 `PATH`; 보고된 이유가 어느 것인지 이름으로 지목한다.
5. `cargo build -p es-editor`; `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/E5-launch-panel.md`.

## 수용 기준

오라클 1–5. 오케스트레이터가 픽스처 번들을 열고, `Eval`을 고르고, Evaluation IR 경로를
채우고, Start를 누르고, E4의 attach를 통해 Run 탭에 셀들이 도착하는 것과 끝에서의 종료
코드를 본다; 실행 도중의 `Kill`은 자식을 끝낸다. 노트의 14절이 클라이언트 규칙과 종료 코드
표를 명시한다.

## 금지

무엇이든 호스팅하는 것: 프로세스 내부 평가나 학습 없음; 컨트롤 프로토콜; 새 의존성;
`crates/es/**`, `crates/es-telemetry/**`, `crates/es-eval/**`; `app.rs`에서 명령줄이
무엇인지 결정하는 것; `docs/ARCHITECTURE*.md`; 새로운 세 개 말고 다른 골든. INV-17: 새
트레이트 없음.
