<!-- Korean translation of docs/design/llm-task-generation.md. The English file is the working copy; regenerate this when it changes. -->

# LLM 태스크 생성 (`es-script::generate`, layer 11)

Spec: spec 14.5 (LLM 생성도 같은 validator를 거친다; 루프는 generate -> validate -> repair,
진단이 피드백된다), spec 14.2 (Python 빌더의 어휘는 같은 노드 집합이다), spec 6 (Task IR:
신경망 없음; IR-D만), spec 7.4 (`ObservationSpec`은 선언이지 전처리가 아니다), spec 5.1
(IR 경계 규칙), spec 0.5 (진화적 오케스트레이터 없음 -- 라운드당 후보 하나, 개체군이
아니다).

## 팔레트를 공유하지 않고 다시 도출하는 이유

`crates/es-editor/src/model/palette.rs::Palette::to_json`가 이미 정확히 이 JSON을
내보내지만, es-editor는 layer 12이고 es-script는 layer 11이므로(spec 4.2): es-script가
es-editor에 의존하면 계층 구조가 뒤집힌다. `es_script::generate::Palette::from_builtins()`는
`TaskNodeRegistry` / `LearningNodeRegistry`(`es_ir::factory`)를 직접 읽어 같은
`NodeSchema` 필드(`kind`, `inputs`, `outputs`, `params`)를 JSON으로 투영하며, 에디터의
메뉴 `category`만 뺀다(LLM에게 필요한 것은 타입 계약이지 UI 그룹핑이 아니다). 두 호출
지점 모두 동일한 `NodeSchema`로부터 도출되므로, 새 노드 kind, 이름이 바뀐 파라미터,
바뀐 포트 타입은 어느 파일도 건드리지 않고 양쪽 모두에 나타난다 — 표현에서는 서로
다를 수 있지만(에디터에는 메뉴 카테고리도 있다) 노드가 무엇을 받는지에서는 결코
어긋날 수 없다. 둘을 동기 상태로 유지하려는 세 번째 파일은 의도적으로 없다;
`crates/es-script/tests/generate.rs`의 `palette_lists_every_builtin_kind`가 이 crate
자신의 관점에서 항목 수와 shape를 고정한다.

## Shape

- `TaskSpecPrompt { description, scene: Option<SceneRef>, constraints: BTreeMap<String,
  String> }` -- 호출자가 원하는 것.
- `build_prompt(&Palette, &TaskSpecPrompt) -> Prompt { system, user }` -- `system`은 전체
  문법이다: 팔레트 JSON, `es_ir::serial` TOML 봉투 형태(`es_schema`, `kind`, `[body]`), 그리고
  타협 불가능한 규칙(`ObservationSpec`만 선언, 전처리 없음, 신경망 없음, 팔레트 kind당
  노드 하나). `user`는 설명에 씬과 제약을 더한 것.
- `parse_candidate(text) -> Result<(TaskIr, Option<ObservationIr>), GenerateError>` --
  ```` ``` ````로 감싸진 블록(언어 태그는 무시됨)을 찾으며, 펜스가 전혀 없으면 트리밍된
  응답 전체를 하나의 TOML 문서로 취급하는 것으로 폴백한다. 각 블록은 먼저 Task IR로,
  그다음 Observation IR로 시도된다; `es_ir::serial::{task,observation}_from_toml`이 이미
  일치하지 않는 봉투 `kind`를 거부하므로, 별도의 sniffing은 필요 없다.
- `repair_loop(&Prompt, provider: &mut dyn FnMut(&Prompt) -> Result<String, GenerateError>,
  max_rounds) -> GenerationReport { rounds: Vec<{ candidate_hash, diagnostics }>, result:
  Option<TaskIr> }` -- 각 라운드는 `provider`를 호출하고, 응답을 파싱하고,
  `TaskIr::validate()`(`es ir validate`가 실행하는 것과 *같은* validator)와 이 crate가
  추가하는 완전성 검사 하나, `GEN-001`로 판정한다: `Reward` 노드가 없는 태스크는
  최적화할 대상이 없는데, 이는 `Reward`가 없는 그래프도 구조적으로는 멀쩡하기 때문에
  `validate()` 자체는 잡아내지 못하는 결함이다. `ObservationIr` 후보가 있으면 그것의
  `validate()`도 실행된다. 실패한 라운드의 형식화된 진단(`Diagnostic`에 대한 `Display`,
  `es ir validate`가 출력하는 것과 같은 spec 5.4 블록 형식)은 다음 라운드를 위해 *원래*
  프롬프트의 user 턴에 그대로 덧붙여진다 -- 누적된 대화 기록은 없으므로, 상태 없는
  provider(Messages API 호출 한 번, 또는 프롬프트에 타이핑하는 사람)만으로 충분하다.

**범위 축소:** spec 11.1의 전체 Cross-IR Check(`es_ir::cross::check`)는 Learning IR과
Deployment IR도 필요로 하는데, 이 생성기는 이를 만들어내지 않는다 -- `TaskSpecPrompt`가
요청하는 것에 맞춰 Task IR(그리고 선택적으로 Observation IR)만을 대상으로 한다.
LLM 기반 Learning/Deployment 생성기를 같은 루프로 연결하고, 넷이 모두 존재할 때 전체
cross-check를 실행하는 것은 이 패킷이 아니라 향후 과제다.

## Provider

`repair_loop`의 `provider`는 평범한 클로저 `&mut dyn FnMut(&Prompt) -> Result<String,
GenerateError>`이므로, 테스트는 `Vec<String>` 이터레이터로 미리 준비된 응답 시퀀스를
스크립트하며, CLI는 `repair_loop`가 HTTP에 대해 전혀 모르는 채로 어떤 백엔드든 갈아
끼울 수 있다.

- `--provider stdin`(CLI 기본값)은 라운드당 stdin에서 전체 응답 하나를 읽는다 -- API
  키 없이 어떤 외부 LLM과도 동작한다. 파이프는 응답을 하나만 공급한다; repair를 지난
  라운드는 EOF를 읽어 파싱에 실패하며, 매달리는 대신 `--rounds`에서 루프를 끝낸다.
- `AnthropicProvider`(`es-script`의 `llm` feature, `es`의 `es-script-llm`)는 `ureq`를
  통해 `POST https://api.anthropic.com/v1/messages`를 호출한다(`x-api-key`,
  `anthropic-version: 2023-06-01`, 본문 `{model, max_tokens, system, messages: [{role:
  "user", content}]}`), 모델 id `claude-sonnet-5`(이 글이 쓰인 시점 기준, 번들로 제공되는
  `claude-api` 스킬의 모델 표에서 현재 Sonnet 등급 모델 -- 이것이 낡아 보이면 그 표를
  다시 확인할 것). `ANTHROPIC_API_KEY`는 오직 환경 변수에서만 읽으며, 저장소 안의
  파일에서는 절대 읽지 않는다. 이 feature가 선택적인 것은 정확히 기본 `es-script` /
  `es` 빌드가 TLS 스택을 절대 링크하지 않도록 하기 위해서다. `AnthropicProvider::with_endpoint`는
  실제 API 대신 호출자가 고른 URL로 요청을 보낸다 -- 이것이 존재하는 유일한 이유는
  테스트가 이를 로컬 `TcpListener`로 향하게 할 수 있도록 하기 위해서다.

**타임아웃과 오류 위생 (M4 review S-8):** `AnthropicProvider` 뒤의 `ureq::Agent`는
`timeout_connect`와 `timeout_read` 둘 다 `AnthropicProvider::DEFAULT_TIMEOUT`(60초)로
설정되어 만들어지며, `with_timeout` / `es task generate --timeout SECS`로 재정의할 수
있다 -- `ureq` 자체의 기본값에는 읽기 타임아웃이 *없어서*, 멈춰버린 provider가 그
라운드(그리고 CLI 호출 전체)를 영원히 매달리게 만들곤 했다. 타임아웃은
`GenerateError::Timeout`으로 보고되며, 이는 포괄적인 `GenerateError::Provider(String)`와
구분되게 유지된다. 그래서 호출자가 다르게 반응하려면(재시도할지 포기할지 등) 메시지에서
"timed out"을 문자열 매칭할 필요가 없다. 다른 모든 provider 실패의 메시지는(프록시나
오류 페이지가 요청 헤더를 그대로 돌려보내는 경우를 대비해 API 키가 그대로 나타나는
자리는 모두 `<redacted>`로 대체되어) 수정된 뒤 256바이트로 잘리고(`redact_and_truncate`)
`GenerateError::Provider`로 감싸진다 -- 예전의 `format!("unexpected response shape:
{resp}")`는 무한정 큰 provider 응답 본문을 그대로 출력했는데, 이는 키 유출
위험을(응답에 키가 나타난 적이 있었다면) 안겨줄 뿐 아니라 임의로 큰 덩어리를 로그나
stdout에 쏟아낼 수 있었다.

## CLI

`es task generate --prompt "..." [--scene scene.xml] [--rounds N] [--timeout SECS]
[--provider anthropic|stdin] --out <dir>` (`crates/es/src/cmd/generate.rs`, `main.rs`에서
`es task generate`로부터 라우팅되며, `es task compile`을 위한 기존
`cmd::task::dispatch`로 넘어가기 전에 처리된다). `--scene`은 이름 붙은 파일을 `blake3`로
해시해 `SceneRef::scene_hash`와 `asset_hash` 양쪽에 쓴다 -- 이는 실제 asset-import
해시(그것은 `es-assets`의 일)가 아니라 자리표시자다; 생성 루프는 보통 임포트된 씬의
실제 콘텐츠 해시를 아직 갖고 있지 않을 것이다. 성공하면 `es_ir::serial::task_to_toml`을
통해 `--out` 아래에 `task.toml`을 쓴다.

`--rounds`의 기본값은 3이며 어떤 provider 호출보다도 먼저 `1..=10`에 대해 검사된다:
라운드 0은 절대 결과를 만들어낼 수 없고, 상한 없는 라운드 수에 상한 없는 라운드당
타임아웃을 곱하면 한 번의 호출에 대해 상한 없는 최악 실행 시간이 되어 버렸다(M4
review S-8, `es/src/cmd/generate.rs:49`는 예전에 어떤 `u32`든 받아들였다). 그 범위
밖이면 `dispatch`는 다른 잘못된 인자와 마찬가지로 종료 코드 2인 `CliError::Usage`를
반환한다. `--timeout`(기본값 60)은 초 단위이며 `AnthropicProvider::with_timeout`으로
전달된다; `--provider stdin`은 네트워크 호출을 전혀 하지 않으므로 이를 무시한다.
