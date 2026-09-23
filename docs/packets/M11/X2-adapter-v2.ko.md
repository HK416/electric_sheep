# M11 X2 — 어댑터 v2: Isaac Lab이나 Playground 정책의 I/O 컨벤션을 선언하고 컴파일한다

스펙: §28.14 규칙 3(외부 정책의 I/O 컨벤션은 IR이 소유한다; 임포터의 어댑터에서 선언되고 번들로
컴파일된다; 추론 시점에는 프레임워크 고유 코드가 전혀 돌지 않는다; IR이 계산할 수 없는 항은 이름으로
거부된다)와 규칙 1과 파동 1, §2.4 / §8.7(IR이 전/후처리를 소유한다), §6.2(`ObservationSpec`),
§7(`TemporalWindow`), §14.4, INV-16(추론 시점에 pickle 없음). 설계 노트: `rl-continuation.md` 8절
"임포터와 어댑터"(+ko); api-notes `isaac-lab.md`(M11 W0: ObservationManager / ActionManager 공식),
`brax-ppo-so101.md`. 유형 B.

## the question

`adapter.toml` v1(`crates/es-import/src/rl_import.rs:116-212`, IMP-001..005)은 액션 관절 순열, 라디안,
`ctrl = offset + scale·a`, Task IR 채널로 매핑된 연속 관측 슬라이스, 첫 레이어에 접힌 mean/std
normalizer를 표현한다. Isaac Lab의 manager 기반 정책은 모두 `joint_pos − default_joint_pos`, 항별
`scale`, `last_action`, 선택적 `history_length`와 `clip`, articulation 순서로 **이름으로** 풀리는 관절,
`decimation × sim.dt` 정책 주기도 쓴다; Playground의 MJX 환경도 비슷한 일을 한다. **이들 각각을 어댑터에서
한 번 선언하고 번들로 컴파일할 수 있는가 — 대수가 정확한 곳에서는 접어서 정확하게 — 번들이 우리 상태
위에서 소스 프레임워크의 관측 → 액션 맵을 재현하도록, 모든 v1 어댑터의 출력은 그대로인 채?**

## spec

* 어댑터 v2 필드, 각각 선택적, `deny_unknown_fields` 유지, 부재 = v1 동작과 바이트 그대로 같음:
  * `[joints] source_names = [...]` — 소스의 articulation 순서에 따른 관절 이름, 우리 액추에이터/관절
    이름에 맞춰 풀림(이름이 다르면 명시적 `[joints.rename]` 테이블); 있으면 `source_order`를 대체(둘 다
    있으면 IMP-008). 채널이 관절별 채널인 관측 슬라이스는 **첫 Dense 레이어의 입력 열을 순열하여** 우리
    순서로 순열된다; 액션은 head 행을 순열하여(정확).
  * `[joints] default_pos = [...]`(라디안, 소스 순서).
  * `[[observation.channels]]`마다: `scale`, `offset`(`"default_pos"` 또는 벡터), `clip`, `history`(N
    프레임, 선언대로 newest-last 또는 newest-first). `(x − offset)·scale`은 MeanStd normalizer로
    접힌다(`mean = offset`, `std = 1/scale`); `clip`은 Observation IR clamp 노드가 된다(기존 노드
    타입이 있으면 그것; 없으면 IMP-009 거부 — 이 패킷에서 새 노드 타입은 없음); `history`는 N의
    `TemporalWindow`가 된다.
  * `[action] use_default_offset = true` → `offset = default_pos`(Isaac의 `JointPositionActionCfg`);
    `clip = [lo, hi]`는 head의 clamp로 접힌다.
  * `[timing] policy_dt` — Deployment IR 제어 주기와 대조; 불일치는 IMP-006.
  * `[actuators]`의 `stiffness`, `damping`, `armature`, `effort_limit`(선택적) — 장면의 액추에이터와
    비교되어 MappingReport에 행으로 쓰임; 절대 변환하지 않음.
  * IR이 계산할 수 없는 채널(`projected_gravity`, `base_lin_vel`, `base_ang_vel`, `velocity_commands`,
    Task IR 목표 채널이 아닌 `generated_commands`) → 항 이름을 대는 IMP-007.
* `prev_action`: 새 Task IR `ObsSource::PreviousAction { initial: Option<Vec<f64>> }` — 정책이 이전
  제어 틱에 낸 액션 행(Safety Plane 이전, 액추에이터 단위), 에피소드 첫 틱에서는 `initial`(부재 = 0).
  어댑터는 `initial = offset`을 써서 리셋 시 접힌 값이 0이 되게 하는데, 이것이 Isaac의 `last_action`이다.
  `ObsSource`의 **마지막** variant라 커밋된 해시는 움직이지 않는다. 그 값은 관측이 오늘 캡처되는 모든
  곳에서 — `es_eval::runner::capture`(평가와 수집)와 `Rollout::observe` — 루프가 이미 쥐고 있는 이전
  정책 출력에서 만들어진다.
* `python/es/import_rl.py`: 경험적 정규화(`obs_normalizer`의 running mean/var → manifest의
  `obs_mean`/`obs_std`)를 쓴 rsl_rl 체크포인트와 Playground brax 파라미터를 읽는다; 소스 설정이 주어지면
  (`--isaac-env-cfg <yaml/json>` / `--playground-config`) `default_joint_pos`, `action_scale`,
  `decimation`, `sim_dt`를 manifest에 기록한다.

## context

```
crates/es-import/src/rl_import.rs
crates/es-import/src/lib.rs
crates/es-import/tests/**
crates/es-ir/src/task.rs
crates/es-ir/src/serial.rs
crates/es-ir/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es-py/src/rollout.rs
crates/es/src/cmd/policy*.rs
crates/es/tests/cli.rs
python/es/import_rl.py
python/es/rl_source/**
tests/fixtures/rl/**
tests/golden/**/import*
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/X2-adapter-v2.md
docs/packets/M11/X2-adapter-v2.ko.md
```

## oracle

1. `cargo test -p es-import adapter_v2_` — 새 필드마다 파싱됨; 거부마다(IMP-006..009) 이름으로; 커밋된
   v1 어댑터의 변환된 Observation IR / Learning IR / 가중치는 모두 이전과 바이트 동일(해시 고정).
2. `ES_PYTHON=… cargo test -p es-import --test isaac_reference` — 생성기가 쓴 합성 소스 둘: Isaac Lab
   스타일 rsl_rl actor(ELU MLP, 경험적 normalizer, `default_joint_pos`, joint_pos_rel ×1,
   joint_vel_rel ×0.05, 목표 자세, last_action, `action_scale 0.5`, `use_default_offset`, 소스 관절
   순서 ≠ 우리, `clip_observations 100`)와 Playground 스타일 brax actor.
   `python/es/rl_source/isaac_reference.py`는 프레임워크의 공식을 **api-note에서, NumPy로** 구현한다;
   임의 상태 256개(이전 액션 포함)에서 `es_policy`를 거쳐 실행한 임포트된 번들은 레퍼런스와 **1e-6**(max
   abs, 액추에이터 단위) 이내로 같다.
3. `cargo test -p es-ir previous_action_` — 커밋된 task 해시는 불변; `PreviousAction` 채널은 해시를
   옮김; `initial`은 라운드트립.
4. `cargo test -p es-eval previous_action_is_the_last_policy_row` — 틱 0 = `initial`, 틱 k = k−1에서의
   정책 행(plane 이전), 버퍼된(chunk) 정책과 horizon-1 정책 둘 다.
5. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

오라클 1~5; `rl-continuation.md` 8절에 v2 표(소스 컨벤션 → 어디에 떨어지는지)가 한국어 짝과 함께 생긴다;
`es policy import-rl --help`가 찍는 거부 목록.

## forbidden

추론 시점의 프레임워크 코드; `import_rl.py` 밖의 pickle(INV-16); 새 Observation IR 노드 타입; 어댑터도
manifest도 말하지 않은 기본 자세·scale·순서 추측; 커밋된 해시 옮기기; `es-safety`.
