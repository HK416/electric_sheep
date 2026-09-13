<!-- Korean translation of docs/packets/M3/W7-learning-loop.md. The English file is the working copy; regenerate this when it changes. -->

# W7-learning-loop — `es loop collect / intervene / distill`과 개입 라벨

Spec: §13.1(루프는 1급 워크플로다; 각 단계는 CLI 명령이자 아티팩트다),
§13.2(개입 데이터의 지위 — 라벨, 출처, 개입이 데이터셋에 들어오는 방식), §13.3(루프
재현성 — 모든 단계가 해시를 싣는다), §19.1(LeRobot 형식, 개입 라벨은 에피소드
메타데이터다), §19.2(Dataset Identity), §19.3(Training Identity), §28.5(M3 W7 행),
§9.3/§9.4와 `INV-12`(Safety Plane은 유일한 actuator 경로 위에 있으며, 어떤 코드
경로도 이를 비활성화하지 않는다).

설계 노트(먼저 읽을 것): `docs/design/learning-loop.md`.

## context (범위)

```
crates/es-data/src/intervention.rs      (new)
crates/es-data/src/collect.rs           (new: Collector, LoopStep, distill)
crates/es-data/src/lib.rs               (+ `pub mod` lines, re-exports, one DataError variant)
crates/es-data/Cargo.toml               (+ es-env, es-safety, es-policy, es-compile,
                                           es-assets, es-physics-core; es-compile promoted
                                           from a dev-dependency)
crates/es-data/tests/loop_learning.rs   (new: one file, because all four tests share one
                                           ~450-line in-test fixture)
crates/es/src/cmd/loop.rs               (new)
crates/es/src/cmd/mod.rs                (+ one `pub mod` line)
crates/es/src/main.rs                   (+ one dispatch arm, usage text)
crates/es/tests/cli.rs                  (append only)
docs/design/learning-loop.md            (new)
docs/packets/M3/W7-learning-loop.md     (this file)
```

## spec (사양)

1. `es_data::intervention`
   - `InterventionSegment { episode, start_frame, end_frame, source: Teleop | Scripted |
     Corrective, operator_id: Option<String>, note: Option<String> }`, `end_frame`은
     포함(inclusive)이다.
   - `ActionSourceCode` — `policy = 0`, `clamped = 1`, `fallback = 2`, `human = 3`;
     `es_safety::ActionSource`로부터의 매핑은 설계 노트 §2.2 표다. clamp나 fallback은
     개입이 **아니지만**(§18.5) 구별되게 기록된다.
   - `label(dataset_root, segments) -> Result<LabelReport>`: `meta/interventions.jsonl`을
     쓰고, 영향받는 에피소드의 `intervention` 컬럼을 다시 쓰며(스트리밍, 한 번에 한
     에피소드씩), 다시 해시한다. `action_source`는 절대 다시 쓰이지 않는다 — 수집
     시점의 출처이기 때문이다. `dataset_content_hash`는 바뀐다; `dataset_schema_hash`는
     `meta/info.json`에 컬럼을 추가해야 했을 때만 바뀐다.
2. `es_data::collect`
   - `Collector::run::<B, F, NJ, H>(&CollectSpec, policy, new_backend, intervener)`는
     번들의 Deployment IR로 만들어진 하나의 `SafetyPlane`으로 `es_env::Env::step_with_policy`를
     구동한다 — 유일한 actuator 경로다(`INV-12`); 주입된 액션은 그저 또 하나의 chunk이며
     다른 것과 똑같이 검증된다.
   - `Intervener = &mut dyn FnMut(episode, frame, &obs) -> Option<[f64; NJ]>`; 제어 틱
     `t`에서의 주입은 프레임 `[t + latency_ticks, t + latency_ticks + execute_chunk)`에
     라벨을 붙인다(설계 노트 §5.1).
   - 설계 노트 §3의 피처 집합으로 `LeRobotWriter`를 통해 에피소드를 쓴다. 프레임별
     `action_source`, `meta/interventions.jsonl`, `Collect` `LoopStep`을 포함한다. 이미지
     채널은 `video` 피처와 경고를 얻는다; mp4는 쓰이지 않는다.
   - `LoopStep { kind: Collect | Intervene | Distill, inputs, outputs, created }`가
     `<root>/loop.jsonl`에 추가된다. 한 줄당 JSON 객체 하나씩, 절대 다시 쓰이지
     않는다.
3. `es_data::distill(inputs, SplitSpec { ratios, seed }, out_root) ->
   Result<TrainingIdentity>`는 데이터셋을 병합하고(에피소드 재색인, `interventions.jsonl`
   재매핑, 피처 맵은 일치해야 함), `DatasetIdentity` + `Split::deterministic`을
   계산하며, `training_identity.json`과 `Distill` `LoopStep`을 출력 root **와** 모든
   입력 root에 쓴다. 학습 실행 자체는 PyTorch 쪽이며 범위 밖이다; 알 수 없는
   `TrainingIdentity` 슬롯들은 모두 0인 다이제스트이며, 절대 조작되지 않는다.
4. CLI
   - `es loop collect --policy policy.esb --scene scene.xml --episodes N --seed S --out
     <root> [--backend mujoco-cpu] [--runtime torch]` — 먼저 backend와 runtime의 가용성을
     검사하고, 어느 쪽이든 사용 불가능하면 `es eval run`이 하는 것과 정확히 같이 종료
     코드 **3**과 함께 `SKIPPED (<reason>)`을 출력한다(§1.4: 실행을 절대 위조하지
     않는다).
   - `es loop intervene --dataset <root> --segments segments.json`
   - `es loop distill --in <root>... --train 0.8 --val 0.1 --test 0.1 --seed S --out
     <root>`
   - 종료 코드: 0 성공, 1 런타임 실패, 2 usage, 3 skipped.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-data -p es --all-targets -- -D warnings
cargo test -p es-data -p es
cargo xtask layering
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- 테스트 내(in-test) `FakeBackend` / `FakePolicy`(테스트 내에서 만들어짐,
  `es-eval` 자신의 오라클이 하는 방식 그대로 — import가 아니라 복사되어, 게이트가 이웃을
  테스트하지 않는다)와 scripted intervener를 사용한 collect는 `action_source` 컬럼이
  정확히 `{policy, human}`을 담고 `meta/interventions.jsonl` 세그먼트가 intervener가
  주입한 프레임과 일치하는 데이터셋을 만든다.
- 쓰인 데이터셋에 대한 `label()`은 `dataset_content_hash`를 바꾸고
  `dataset_schema_hash`는 바꾸지 않은 채 남긴다(스키마가 이미 두 컬럼을 싣고 있다);
  `action_source` 컬럼은 전후로 바이트 단위로 동일하다.
- 두 데이터셋을 증류하면 그 에피소드 수가 합산되고, 정확한 서로소
  분할(disjoint partition)인 split이 만들어지며, 두 번의 실행에 걸쳐 동일한
  `TrainingIdentity`가 나온다.
- 수집된 root의 `loop.jsonl`은 세 단계(`collect`, `intervene`,
  `distill`)를 싣고 있으며, 그 입력 해시들은 이전 단계의 출력 해시로 체인처럼
  이어진다(§13.3).
- CLI: `es loop intervene`과 `es loop distill`은 테스트 내 데이터셋에서
  end-to-end로 동작한다; `es loop collect`는 backend나 runtime을 사용할 수 없을 때
  `SKIPPED`와 함께 3으로 종료한다.

## forbidden (금지)

- `crates/es-safety`, `crates/es-runtime-embedded`,
  `crates/es-eval/src/{evidence,domain_gap}.rs`, `crates/es-compile`, `crates/es-splat`,
  `crates/es-editor`, `crates/es/src/cmd/{evidence,gap}.rs`를 건드리는 것 — 이웃한
  패킷들이 이들을 소유한다.
- 수집을 위해 Safety Plane을 우회하거나, 비활성화하거나, 넓히는 어떤
  분기든("테스트를 위해서만"을 포함해)(`INV-12`): 주입된 사람의 액션은 다른 모든 chunk와
  똑같이 `validate`를 거친다.
- `label()`에서 `action_source`를 다시 쓰는 것, 또는 plane이 아닌 다른
  무언가로부터 그것을 유도하는 것.
- 학습 루프를 실행하는 것, 경계의 이쪽에서 알 수 없는 `TrainingIdentity` 슬롯에
  0이 아닌 다이제스트를 내는 것, 또는 mp4나 조작된 이미지 관측을 쓰는 것.
- 루트 `Cargo.toml`을 수정하는 것, golden 파일을 수정하는 것, 또는 커밋하는
  것.
