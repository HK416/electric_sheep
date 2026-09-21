# M7 R1 — 에피소드 경계, 그리고 `(cell, episode)` 분할 (T8b)

스펙: §9.4(`begin_episode`가 `ViolationRate` 윈도를 비운다 — 2026-09-21에 추가된 문장), §10.5
(`events.json`의 `tick`은 에피소드 상대다 — 같은 커밋), §10.4(`--jobs N`은 `--jobs 1`이 만들지
않을 보고서를 만들 수 없다), §13.1(에피소드는 스트림이 끝나는 곳이다), §28.11 웨이브 0, §28.9
규칙 2(무효화된 측정은 지우지 않고 표시한다). 리뷰: `docs/reviews/M7.md` S-1, S-2, follow-up R1;
소유자의 결정은 §28.11에 기록되어 있다. 이 패킷이 완성하는 패킷: `T8-episode-seek.md`. 확장할
설계 노트: `docs/design/safety-plane.md`("Counters"), `docs/design/evaluation-execution.md`
2.7절(다시 쓰임: 단위가 *곧* `(cell, episode)`다; 그 뒤에 벽시계 표), `docs/design/visible-learning.md`
7.33절(U3를 새 의미론 아래 다시 측정). 한국어 자매 문서는 같은 커밋에.

## 질문

T8은 `Env::seek_episode`가 재생임을(`k ∈ {1, 3, 7}`에 대해 비트 단위로) 증명했고 16편짜리 명목
스위트 위에서 `(cell, episode)` 분할을 2.0×/2.2×(jobs 4/8)로 측정했지만, 그것을 내보내지
않았다: `SafetyCounters::window`가 에피소드 `k−1`의 꼬리를 에피소드 `k`로 들고 가고(첫 차이는
`nominal-01` tick 24), `StepEvent::tick`이 셀의 누적 물리 시계다(`events.json` `nominal-01`
레코드 0: tick 7200). 소유자는 2026-09-21에(§28.11) 둘 다를 결정했다: 윈도는 `begin_episode`에서
비워지고, `tick`은 에피소드로부터 센다. **둘 다 들어가면, `(cell, episode)` 분할은 커밋된
문서들 위에서 순차 실행과 비트 단위로 같은가, 명목 스위트는 이제 얼마나 드는가, 그리고 U3의
홀드아웃 수치는 무엇이 되는가?**

## 사양

* **`es-safety`, 한 줄.** `SafetyPlane::begin_episode`가 `ViolationRate` 링도 비운다
  (`self.counters.window.clear()`). `SafetyCounters` 안의 다른 것은 아무것도 움직이지 않는다:
  `violations`, `steps`, `clamped_steps`, `dirty_steps`, `fallback_activations`는 셀에 걸쳐
  계속 합산되고, `reset_counters`가 그것들을 0으로 되돌리는 유일한 방법으로 남는다. doc
  comment는 §9.4가 이제 말하는 바를 말한다: 래치, 시드, 그리고 링 — 봉투, watchdog들, 그리고
  합산은 손대지 않는다(INV-12), 그리고 `validate`의 시그니처는 손대지 않는다(INV-13). 설계
  노트의 "Counters" 절은 링이 에피소드당이라는 것과 그 이유를(§10.3: 꽉 찬 윈도이거나, 아니면
  아무것도 아니다) 기록한다.
* **`es-eval`, 시계.** `StepEvent::tick`과 `RunEvent::Observation { tick }`은
  `env.tick() − start`를 나른다, 여기서 `start`는 `run_episode`에 들어갈 때 읽힌
  `env.tick()`이다(env는 거기서 막 리셋된 참이다, 패킷 M5/V6b). `frame`은 바뀌지 않는다.
  `events.json`의 스키마는 바뀌지 않는다; 셀의 첫 에피소드 이후의 수치들만 움직인다.
* **컬렉터의 스트림-2 tick.** `es loop collect --telemetry`도 `[frame, tick, source, bits]`를
  발행한다(패킷 M7/E7). 그 `tick`이 `env.tick()`(누적)이면, 발행 지점에서 에피소드 상대로
  만들어 라이브 뷰어가 두 단계에서 하나의 시계를 보게 하라; 이미 그렇다면 노트에 그렇게
  적어라.
* **분할(T8b), T8이 명세한 그대로.** `Evaluation::run_shard_with_sink`의 단위는
  `(cell, episode)` 쌍이다: 단위는 평탄화된 `suites × seeds` 목록 위의
  `cell * n_episodes + episode`이고, 단위 `u`는 샤드 `u % count`로 간다. 각 단위는 자신만의
  `Env`(`episode`로 seek되고, 그 뒤 `reset(Some(&[0]))`), `SafetyPlane`, `ChunkBuffer` +
  `PlaneFeed`, `AsyncInference`를 짓는다. `ShardCell`은 에피소드별 레코드(`Episode`, 플레인의
  `SafetyCounters` 스냅숏, `EnvMetrics`)를 얻고 `record_cell`은 `merge`로 옮겨가며, 이것이
  그것들을 **에피소드 순서로** 합산한다 — 그러므로 `report.json`, `events.json`, 그리고 모든
  `.estraj`는 어느 워커가 만들었든 같은 에피소드별 레코드로부터 계산된다. **경로는 하나뿐**:
  `--jobs 1`은 `count = 1`인 같은 분할이고(T8은 에피소드별 `--jobs 1`을 노이즈 안쪽인 +1초로
  측정했다), 그러므로 순차 실행은 두 번째 구현이 아니다. `merge`는 `(cell, episode)` 전부를
  정확히 한 번씩이 아닌 단위 집합을 거부한다.
* **`es eval run`.** `--jobs`는 `suites`가 아니라 `suites × n_episodes`로 clamp된다;
  `--shard i/N --shard-out` 워커 프로토콜은 그대로 유지된다; 도움말 텍스트의 "한-스위트짜리
  평가는 속도 향상을 얻지 못한다" 문단은 이제 참인 것으로 바뀐다. `--telemetry`는 여전히
  `--jobs 1`을 필요로 한다.
* **골든이 아니라 픽스처.** `tests/fixtures/visible-learning/run/events.json`과
  `tests/fixtures/visible-learning/events.json`은 에디터와 비디오 테스트가 읽는다. 그
  tick들이 새 시계 아래에서 움직인다면, 그것들을 만든 명령을 통해 다시 만들고 노트에 그렇게
  적어라; 움직이지 않는다면(그 셀들이 한 에피소드 길이일 수 있다) 대신 그렇게 적어라.
  `tests/golden/**`는 손대지 않는다.
* **커밋된 수치들의 재확인(§28.9 규칙 2).** `visible-learning.md`의 7.32까지 모든 평가 수치는
  윈도 이월 아래에서 측정된 것이다. 새 7.33절이 한 문단으로 그렇게 말하고, **U3**
  (`~/artifacts/plan-v/m7-u/U3`, `observation-augmented.toml`을 쓴 커밋된 문서들, 홀드아웃
  시드 16개 × 스위트 6개)를 새 빌드로 다시 측정한다: 그 행은 옛 0.5625 옆에 놓이고, 그것은
  남아서 "pre-R1 semantics"로 표시된다. 전문가 게이트의 명목 스위트 위 `success_rate`와
  `envelope_violation_rate`도 그 옆에서 다시 측정되어, 어떤 정책의 것보다 먼저 새 의미론
  아래에서 하네스 자신의 수치가 기록에 남는다.
* **벽시계 시간.** 명목 스위트(16편, `--frames`, T8이 쓴 번들 —
  `~/artifacts/plan-v/v14/trained-20000.esb`, 그 경로가 없어졌다면 U3의 것; 어느 쪽인지
  적어라)를 `--jobs 1`, `4`, `8`로, 두 번씩, 각 행 옆에 1분 로드, T8의 표 형식으로; 행들에
  걸친 동등성은 셀 단위로 단언된다(`report.json`, `events.json`, `.estraj`; `evaluation.lock`의
  `created`는 예외). 실행이 측정하는 곳에서는 §12.4의 아홉 지표로, 측정하지 않는 곳에서는
  `Target / Status: unverified`로 보고한다 — 결코 `step/s` 하나로는 아니다.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

```
crates/es-safety/src/plane.rs
crates/es-safety/src/counters.rs
crates/es-safety/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/src/metrics.rs
crates/es-eval/tests/**
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/telemetry.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/run/events.json
tests/fixtures/visible-learning/events.json
docs/design/safety-plane.md
docs/design/safety-plane.ko.md
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M7/P-M7-R1.md
docs/packets/M7/P-M7-R1.ko.md
```

`plane.rs`(`begin_episode`, 한 줄과 그 주석), `counters.rs`(`window.clear()`가 갖고 있지 않은
`pub(crate)` 경로가 필요할 때만), `es-safety/tests`(시나리오 스위트가 경계 사례를 얻는다),
`runner.rs`(시계, 단위, `ShardCell`의 에피소드별 레코드, merge에서의 `record_cell`),
`metrics.rs`는 **오직** 에피소드별 합산에 헬퍼가 필요할 때만, `es-eval/tests`(오라클 2–3),
`eval.rs`(clamp, 도움말), `loop.rs`/`telemetry.rs`(컬렉터의 tick, 누적일 때만),
`cli.rs`(`eval_jobs_*`), 두 픽스처(움직일 때만), 노트들, 이 패킷.

## 오라클

1. `cargo test -p es-safety window_is_cleared_at_begin_episode` — `envelope_violation_rate()`가
   `0` 위를 읽을 때까지 링을 채우고, `begin_episode`를 부르고, `0.0`을 단언한다; 합산되는
   모든 카운터가 바뀌지 않았고 래치가 clear임을 단언한다. 시나리오 스위트의 두 번째 테스트:
   에피소드 `k−1`에서 `ViolationRate`를 트립시킨 플레인은 에피소드 `k`의 tick 0에서 트립하지
   않는다.
2. `cargo test -p es-eval tick_is_episode_relative` — 픽스처 백엔드 위의 두-에피소드 셀:
   에피소드 1의 첫 `StepEvent.tick`은 `0`이고 그 tick 수열은 에피소드 0의 것과 같다;
   `frame` 수열은 바뀌지 않는다.
3. `cargo test -p es-eval episode_shards_reproduce_the_sequential_run -- --ignored` — T8의
   오라클 3, 이제 **단언**이다: 커밋된 데모 문서들 위에서, 명목 스위트의 4편, `count = 1`
   대 2와 4 워커에 걸친 분할 — `report.json`, `events.json`, 모든 `.estraj`가 바이트 단위로
   같다; `RAN … identical`을 찍는다. `ES_PYTHON` 없이는 `SKIP`.
4. `cargo test -p es --test cli eval_jobs_splits_episodes` — 한-셀짜리 픽스처 위의
   `--jobs 2`는 각각 에피소드 하나씩을 쥔 워커 둘을 낳는다(워커 명령줄에서 되읽음);
   `--jobs 0`과 `--shard` 사용법 거부는 바뀌지 않는다.
5. 서버: 명목 스위트, jobs 1/4/8, 두 번씩 — 행들에 걸친 셀별 아티팩트가 비트 단위로 같음
   (`created` 예외), 그리고 **jobs 4 벽시계 시간 ≤ jobs 1의 55%**. 표는
   `evaluation-execution.md` 2.7절에.
6. 서버: U3 홀드아웃과 명목 전문가 게이트를 다시 측정; `visible-learning.md` 7.33절에 두
   행 모두와, 옛 것들에는 "pre-R1 semantics" 표시.
7. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/P-M7-R1.md`.

## 수용 기준

오라클 1–7(3, 5, 6은 서버에서: 트리의 tarball로부터 `~/Projects/es-r1-episodes`,
`ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`, 아티팩트는
`~/artifacts/plan-v/m7-r1-episodes/` 아래; 박스는 공유된다 — T8이 했듯, 시간을 재는 행 전에
1분 로드가 4 아래로 내려가길 기다려라). 분할은 오직 동일성 위에서만 내보내진다: 오라클 3이
차이를 찾으면, 그 발견(다른 cell / episode / tick과 그 watchdog 이벤트)이 설계 노트에 실리고
패킷은 거기서 멈춘다. 세 노트와 그 한국어 자매 문서가 갱신됨; 패킷의 `.ko.md` 자매 문서가
존재함.

## 금지

`SafetyPlane::validate`의 시그니처(INV-13); `begin_episode`에서 링을 넘어 어떤 플레인 상태를
건너뛰거나 0으로 되돌리는 것(INV-12 — 합산은 셀의 것이다); `es-safety`가 자기 자신의
`begin_episode` 말고 다른 곳으로부터 에피소드를 알게 되는 것(INV-11, 규칙 8); `Env::reset`의
뽑기 순서나 `Env::new`가 하는 일을 바꾸는 것; `record_cell`의 산술(같은 순서의 같은 합이어야
한다); 어떤 지표의 정의; `docs/ARCHITECTURE*.md`(그 문장들은 이미 거기 있다); `tests/golden/**`;
텔레메트리 프로토콜(`crates/es-telemetry/**`); 에디터; 새 분할 옆에 셀 수준 분할을 살려 두는
플래그(경로는 하나뿐). INV-17: 새 트레이트 없음.
