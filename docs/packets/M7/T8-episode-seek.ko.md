# M7 T8 — `Env::seek_episode`, 그리고 셀뿐 아니라 에피소드를 가르는 `--jobs`

스펙: §10.4(`--jobs N`은 `--jobs 1`이 만들지 않을 보고서를 만들 수 없다), §6.3(무작위화는
`(seed, env, episode, stream)`으로 키가 매겨진다), §3.5 tier 1(비트 단위), §28.9 사다리
10단("`Env` 에피소드 seek — seek 뒤의 상태는 0..n을 재생한 것과 비트 단위로 같다"), §28.9
"벽시계 시간이 어디로 가는가"(에피소드 수준 샤딩: "에피소드 카운터가 그것을 막는다; seek가
있으면 한-셀짜리 명목 실행조차 병렬화된다"), §28.10(T8). 확장할 설계 노트:
`docs/design/evaluation-execution.md`(+ `.ko.md`) — 샤딩 절; `docs/design/visible-learning.md`
7.11절의 "샤드는 셀의 분할이다" 문단이 포인터를 얻는다. **T7**(평가기의 지연시간 모델은
컬렉터의 것이다)에 의존한다.

## 질문

`es eval run --jobs 6`은 *셀*을 라운드로빈으로 가르므로, 여섯-스위트 스윕은 여섯-폭으로 돌고
한-셀짜리 명목 스위트(16편, ~5분)는 한-폭으로 돈다. `Env::reset`은 에피소드 `k`의
무작위화를 `(seed, env, episode_counter, stream)`으로부터 뽑고, `k`번의 reset 말고는 아무것도
그 카운터를 설정할 수 없다. **`Env`에게 에피소드 `k`에 있으라고 말할 수 있고, 거기서부터
순차 실행이 만드는 바이트를 만들 수 있는가 — 그렇다면, 평가기의 셀별 상태 중 무엇이 여전히
`(cell, episode)` 분할이 하나가 되는 것을 막는가?**

## 사양

* **`Env::seek_episode(env: u32, episode: u64)`**는 env의 에피소드 카운터를 설정해 *다음*
  `reset`이 에피소드 `episode`를 뽑게 한다. 그 외에는 아무것도 건드리지 않는다 — reset도,
  백엔드 호출도 없다 — 그리고 스텝이 있는 에피소드가 열려 있는 동안은
  (`recorder.open(env).steps() > 0`) 거부된다(`EnvError`), 그러므로 seek는 결코 기록된
  에피소드를 조용히 버리는 일이 아니다. `Env::new`는 오늘처럼 생성 시점에 에피소드 0을
  뽑으므로, 에피소드 `k`를 원하는 갓 만든 `Env`의 호출자는 `seek_episode(0, k)`를 부르고
  `reset(Some(&[0]))`을 부른다.
* **먼저, `Env` 수준의 오라클.** 데모 장면과 커밋된 Task IR로: `k`번 reset된 갓 만든 `Env`
  대 `k`로 seek되어 한 번 reset된 갓 만든 `Env` — `k ∈ {1, 3, 7}`에 대해 reset 뒤의
  `StateView`(`qpos`, `qvel`), 에피소드의 `ParamScales`, 그리고 스크립트화된 전문가가
  이끄는 에피소드의 `.estraj` 전체가 **바이트 단위로 같다**. MuJoCo 백엔드의 `reset`은
  상태를 설정하기 전에 `mj_resetData`를 호출하는데(`python/mujoco_ref.py`), 이것이 이것을
  가능하게 만드는 것이다; 오라클은 그것을 증명하는 것이다.
* **평가기의 셀별 상태, 이름을 붙이면.** `Evaluation::run_shard`는 셀마다, 그리고 그
  에피소드들에 걸쳐 다음을 유지한다: `Env`, `SafetyPlane`(`begin_episode`는 래치를 지우지만
  `ViolationRate` 윈도는 **지우지 않는다** — `SafetyCounters::window`는 에피소드 `k-1`의
  꼬리를 에피소드 `k`로 들고 간다), `ChunkBuffer` + `PlaneFeed`(에피소드마다 지워짐),
  `AsyncInference`(에피소드마다 지워짐), `seq`(셀마다 단조 증가; 순서만 판정된다), 그리고
  `env.metrics()` / `safety.counters()`(`record_cell`이 셀의 에피소드들에 걸쳐 합산). 이들
  중 **윈도**는 그 이월이 합이 아니라 행동인 유일한 것이다. 이 패킷은 그것을 바꾸지 않는다:
  `begin_episode`가 윈도를 지워야 하는지는 M7 리뷰를 위한 `es-safety`/스펙 결정이고, 그것을
  지우면 커밋된 문서들의 결과가 움직일 것이다.
* **`(cell, episode)` 분할, 동등성에 의해 게이트됨.** `run_shard`의 단위는 평탄화된
  `suites × seeds` 목록 위의 `(cell, episode)` 라운드로빈이 된다; 각 단위는 자신만의
  `Env`(seek된), `SafetyPlane`, 버퍼, feed, inference를 짓는다; `ShardCell`은 에피소드별
  `Episode`, `SafetyCounters`, `EnvMetrics` 레코드를 얻고 `record_cell`은 `merge`로
  옮겨가며, 이것이 그것들을 에피소드 순서로 합산한다 — 그러므로 `report.json`, `events.json`,
  그리고 모든 `.estraj`는 어느 워커가 만들었든 같은 에피소드별 레코드로부터 계산된다.
  **분할은 오라클 3이 커밋된 문서들 위에서 성립할 때만 활성화된다**; 윈도 이월이 그것을
  깨뜨리면 분할은 셀 수준에 머무르고, 그 발견은 처음으로 갈라지는 cell/episode/tick과
  함께 설계 노트에 실리며, 그것을 활성화했을 플래그는 존재하지 않는다(YAGNI — M7 리뷰가
  결정한다).
* **벽시계 시간.** 서버에서 `--jobs 1`, `--jobs 4`, `--jobs 8`로 명목 스위트(16편), 전후:
  관측이며, 측정되기 전까지는 `Target / Status: unverified`다.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

```
crates/es-env/src/env.rs
crates/es-env/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/src/metrics.rs
crates/es-eval/tests/**
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M7/T8-episode-seek.md
docs/packets/M7/T8-episode-seek.ko.md
```

`es-env/src/env.rs`(`seek_episode`와 그 거부; `es-env` 안의 다른 것은 없음), `es-env/tests`
(오라클 1), `es-eval/src/runner.rs`(그 단위, `ShardCell`의 에피소드별 레코드, merge에서의
`record_cell`), `es-eval/src/metrics.rs`는 **오직** 에피소드별 합산에 헬퍼가 필요할 때만,
`es-eval/tests`(오라클 2–3), `es/src/cmd/eval.rs`(워커 생성: 셀 대신 단위 — `--shard i/N`
프로토콜은 그대로 유지), `es/tests/cli.rs`(`eval_jobs_*` 테스트), 설계 노트들, 이 패킷.

## 오라클

1. `cargo test -p es-env seek_is_the_replay` — 위의 `Env` 수준 진술, `k ∈ {1, 3, 7}`에 대해
   바이트 단위로 같음, `.estraj` 포함(`ScriptedExpert`로 구동; MuJoCo 백엔드 없이는 이유와
   함께 `SKIP`).
2. `cargo test -p es-env seek_refuses_an_open_episode` — 스텝 이후 reset 이전의 seek는 env와
   그 스텝 수를 이름으로 지목하는 `EnvError`다.
3. `cargo test -p es-eval episode_shards_reproduce_the_sequential_run -- --ignored` — 커밋된
   데모 문서들 위에서, 명목 스위트의 4편: `--jobs 1` 대 2, 4 워커에 걸친 `(cell, episode)`
   분할, `report.json`, `events.json`, 그리고 모든 `.estraj`가 바이트 단위로 같다. 어느
   쪽이든 출력됨: `RAN … identical` 또는 `RAN … first difference at <cell>/<episode>/<tick>`.
   이것이 스펙 네 번째 불릿의 게이트다. `ES_PYTHON` 없이는 `SKIP`.
4. `cargo test -p es --test cli eval_jobs_splits_episodes` — 분할이 활성화된 채: `--jobs 2`로
   돌린 픽스처 실행은 한-셀짜리 픽스처의 에피소드 하나씩을 각각 쥔 워커 두 개를 낳는다(테스트가
   되읽는 워커 명령줄로부터); 비활성화된 채로는, 테스트가 셀 수준 동작을 단언하고 설계
   노트의 발견 문단이 존재함을(grep) 단언한다.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T8-episode-seek.md`.

## 수용 기준

오라클 1–5(3은 서버에서). 벽시계 시간 표(명목, 16편, jobs 1/4/8)는 설계 노트 안에. 분할이
게이트되어 꺼져 있다면, 그 발견은 추측이 아니라 증거(다른 tick과 watchdog 이벤트)와 함께
메커니즘을 이름으로 지목한다.

## 금지

`Env::reset`의 뽑기 순서나 `Env::new`가 하는 일을 바꾸는 것; `crates/es-safety/**`(윈도
결정은 리뷰의 몫이다); `crates/es-physics-backend/**`; 어떤 지표의 정의나 `record_cell`의
산술을 바꾸는 것(합은 같은 순서의 같은 합이어야 한다); 오라클이 증명하지 않은 분할을
활성화하는 플래그; `docs/ARCHITECTURE*.md`; 골든. INV-12: 동등성을 성립시키기 위해 건너뛰는
플레인 상태는 없다. INV-17: 새 트레이트 없음.
