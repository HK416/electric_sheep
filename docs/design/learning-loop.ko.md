<!-- Korean translation of docs/design/learning-loop.md. The English file is the working copy; regenerate this when it changes. -->

# 학습 루프 — collect / intervene / distill

Spec: §13.1(루프는 1급 워크플로이며, 각 단계는 CLI 명령 *이자* 아티팩트다),
§13.2(개입 데이터의 지위 — 라벨, 출처, 개입이 데이터셋에 들어오는 방식), §13.3(루프
재현성: 모든 반복이 해시 체인으로 연결된다), §19.1(LeRobot 형식, 개입 라벨은 에피소드
메타데이터다), §19.2(Dataset Identity), §19.3(Training Identity), §9.3/§9.4(Safety
Plane), `INV-12`(어떤 코드 경로도 이를 비활성화하지 않는다).

패킷: `docs/packets/M3/W7-learning-loop.md`. 이 노트는 사람이 리뷰하는
아티팩트이며; `crates/es-data/src/{collect,intervention}.rs`의 코드는 이 문서의 하류에
있다.

---

## 1. 이 wave가 구현하는 것, 그리고 구현하지 않는 것

§13.1의 루프에는 여섯 개의 명령이 있다. 이 wave는 그중 셋을 소유한다:

```
es loop collect    policy.esb + scene  ->  dataset/ (+ loop.jsonl)
es loop intervene  dataset/ + segments.json  ->  dataset/ (labels)      (+ loop.jsonl)
es loop distill    dataset/...  ->  dataset/ + training_identity.json   (+ loop.jsonl)
```

`train`은 여기 **없으며** 앞으로도 없을 것이다: 학습 실행 자체는 §2.3의 분리
기준에서 PyTorch 쪽에 있다(Python은 학습 경로에서만 1급 의존성이다). `distill`이 만드는
것은 그 실행의 *입력 정체성(input identity)*이다 — 병합된 데이터셋, 결정적 분할, 그리고
§19.3이 요구하는 `dataset.lock` 슬롯을 가진 `training_identity.json`이다.
`TrainingIdentity`의 나머지 슬롯들(`config`, `optimizer`, `scheduler`, `seed`,
`base_model`, `augmentation`, `precision`, `topology`, `checkpoint_manifest`, `metrics`,
`hardware`)은 모두 0인 다이제스트로 쓰이는데, 경계의 이쪽에서는 그것들을 알 수 없기
때문이다. 0 다이제스트는 "설정되지 않음"으로 읽히지만, 조작된 값은 `training_hash`를
거짓으로 만들 것이다.

`es eval`과 `es deploy`는 이미 존재하거나 다른 패킷의 소유다; 루프의 `FAILURE
SET` 화살표(§13.1)는 입력 쪽에 여러 데이터셋 root를 가진 `distill`이다.

---

## 2. 개입 라벨 스키마 (§13.2)

§13.2는 개입을 *에피소드 메타데이터*에, 이유와 operator를 가진 세그먼트 목록으로
둔다. 이것이 이 구현이 취하는 형태이며, 여기에 프레임별 컬럼을 더해 학습 로더가 세그먼트
목록으로부터 다시 유도하지 않고도 프레임에 가중치를 줄 수 있게 한다.

### 2.1 세그먼트 — `meta/interventions.jsonl`

```rust
pub struct InterventionSegment {
    pub episode: u32,
    pub start_frame: u32,
    pub end_frame: u32,            // inclusive
    pub source: InterventionSource,
    pub operator_id: Option<String>,
    pub note: Option<String>,
}

pub enum InterventionSource { Teleop, Scripted, Corrective }
```

한 줄에 JSON 객체 하나씩, `(episode, start_frame)`으로 정렬된다. 이것이 출처
기록(provenance record)이다: §13.2의 `reason`은 `note`이며, `operator_id`는 그대로
실린다 — 이 필드의 목적이 분석이 아니라 귀속(attribution)이기 때문이다. `Scripted`는
machine이 주도하는 intervener다(고전적 컨트롤러, 재생된 시연, 아래의 테스트 훅);
`Teleop`은 장치 앞의 사람이며; `Corrective`는 실행 중인 정책을 교정하는 사람 — §13.2가
이름 붙인 HIL-SERL 사례다.

세그먼트는 병합되거나 정규화되지 **않는다**: 두 operator에게서 온 겹치는 두 teleop
세그먼트는 두 줄로 남는데, 그것들을 합치면 이 파일이 존재하는 이유인 귀속 정보가 파괴되기
때문이다. 겹침은 오직 프레임별 컬럼에서만 해소되며, 거기서는 boolean이다.

### 2.2 프레임별 컬럼

세 개의 컬럼이 LeRobot 피처 집합에 합류한다:

| feature | dtype | shape | meaning |
|---|---|---|---|
| `intervention` | `int64` | `[1]` | `0` 또는 `1`: 이 프레임이 어떤 세그먼트 안에 있는가 |
| `action_source` | `int64` | `[1]` | `0` policy, `1` clamped, `2` fallback, `3` human |
| `action_commanded` | `float32` | `[nu]` | 그 틱의 플레인 통과 이전 명령 (패킷 M5/V1c) |

`action`은 플레인이 액추에이터 쪽으로 건넨 `SafeAction`이고, `action_commanded`는 같은 틱에 청크
버퍼가 내어준 행, 즉 플레인이 판정하기 전의 값이다. 실행된 것이 기록되는 것이고 요청된 것은 그
옆에 남는다 — 그래서 `Clamped` 프레임을 플레인을 다시 돌리지 않고 읽을 수 있다. 버퍼에 그 틱의
행이 없었던 틱에는 명령 자체가 없었으므로 컬럼은 플레인 자신의 답을 되풀이하며,
`action_source`는 정확히 그 틱들에서 `2`(fallback)를 읽는다.

`intervention`은 §13.2의 의미에서는 `u8` 값이다; `int64`로 저장되는 이유는 이
크레이트가 읽고 쓰는 네 가지 컬럼 dtype이 `float32 / float64 / int64 / bool`이기
때문이며(`docs/api-notes/lerobot-dataset.md` 참조), 프레임당 7바이트를 아끼려고 다섯 번째
물리 타입을 새로 만드는 것은 상호운용 위험을 감수할 가치가 없다.

**`action_source`는 `intervention`과 같은 질문이 아니다.** §18.5와
`es-safety`의 `ActionSource`는 clamp나 fallback이 실패도 사람도 아닌 *정상 동작*이라고
명시한다: Safety Plane이 액션을 바꾼 것이지 누가 개입한 것이 아니다. 그래서 이 두 컬럼은
독립적이며, 네 가지 `action_source` 코드는 plane이 보고하는 것과 일대일로 대응한다:

```
es_safety::ActionSource::Policy      -> 0 policy
es_safety::ActionSource::Clamped     -> 1 clamped
es_safety::ActionSource::Fallback(_) -> 2 fallback   (the kind is in the safety counters)
(no plane counterpart)               -> 3 human
```

plane이 그 뒤에 clamp한 사람의 액션은 `intervention = 1`과 함께
`action_source = 1`(clamped)로 기록된다. 그것이 정직한 쌍이다: 그 프레임은 개입이었고,
actuator 위의 액션은 사람이 요청한 것이 *아니었다*. 이를 `3`으로 뭉개버리면 clamp를
숨기게 되는데, 그것이야말로 §9.3이 보존하기 위해 존재하는 증거다.

### 2.3 `label()`이 다시 쓸 수 있는 것과 없는 것

`es_data::intervention::label(root, segments)`:

- `meta/interventions.jsonl`을 쓴다;
- 영향받는 모든 에피소드의 `intervention` 컬럼을 다시 쓴다(세그먼트 안이면 `1`,
  밖이면 `0`), 한 번에 한 에피소드씩 스트리밍하며;
- 데이터셋이 그보다 이전 것이라면 `meta/info.json`에 `intervention` /
  `action_source`를 추가한다(그리고 오직 그 경우에만 `dataset_schema_hash`가 바뀐다);
- `action_source`는 **절대 건드리지 않는다.** 그 컬럼은 수집 시점의 출처
  정보다: 로봇이 움직이는 동안 Safety Plane이 실제로 낸 것을 기록한다. 몇 시간 뒤에
  적용되는 라벨은 그것을 알 수 없으며, 그것을 덮어쓰면 측정값이 주장으로 변질된다.

결과들, 이것이 §19.2의 세 방향 해시가 존재하는 이유다:

```
dataset_content_hash   changes   (parquet bytes changed)
dataset_schema_hash    unchanged (unless a column had to be added)
dataset_split_hash     unchanged (the split is not the labeller's business)
```

§13.2는 또한 개입 세그먼트가 학습에서 다르게 *가중치*가 매겨질 수 있으며 그
가중치 정책이 `dataset_hash`의 일부라고 언급한다. 가중치 정책은 학습 쪽 문서이며; 이
컬럼들이 아니라 `TrainingIdentity::config`를 통해 해시로 들어간다.

---

## 3. 수집된 에피소드에서 LeRobot 프레임으로

`es_env::Episode`는 env별로 열 지향(columnar)이다(`qpos`, `qvel`, `ctrl`,
`sensordata`, `reward`, `done`, `failure`). 이 매핑은 의도적으로 좁다 — 정책이 소비하거나
생성하는 것마다 하나의 피처, 그 이상의 추측성 확장은 없다:

| LeRobot feature | dtype | shape | from |
|---|---|---|---|
| `observation.state` | `float32` | `[nq + nv]` | `qpos ‖ qvel`, §12.2의 plan-free 경로의 raw observation |
| `action` | `float32` | `[nu]` | `ctrl` — Safety Plane **이후**의 액션, 즉 actuator가 본 것 |
| `reward` | `float64` | `[1]` | `reward` |
| `intervention` | `int64` | `[1]` | §2.2 |
| `action_source` | `int64` | `[1]` | §2.2 |
| `timestamp` | reserved | | `frame / fps`, `fps` = deployment의 제어 주기율(control rate) |
| `task_index` | reserved | | 모두 `0` — collect 실행당 하나의 task |

`action`이 plane 이후의 액션인 것은 의도적이다. `action` 컬럼이 정책의 raw
출력인 데이터셋은 다음 정책이 plane이 거부하는 액션을 재현하도록 학습시킬 것이고, 루프는
결코 수렴하지 않을 것이다. raw 출력은 중요한 곳에서는 복구 가능하다: `action_source`가
어느 프레임에서 둘이 다른지 말해준다.

`tasks.jsonl`은 엔트리를 하나 얻는다, 문자열 `es:task:<task_hash hex>`다.
collect 실행에는 거기 넣을 자연어 지시문이 없다; 최소한 해시는 그것을 만든 Task IR로
다시 거슬러 갈 수 있다.

### 3.1 이미지 채널: `VideoRef` 플레이스홀더와 경고

번들의 `ObservationSpec`이 `ImageSpec`을 실은 채널을 선언하면, 각각에 대해
`meta/info.json`에 `video` 피처가 쓰인다 — 그래서 `info.video_path`가 채워지고
`LeRobotDataset::read_episode`는 카메라별·프레임별로 `VideoRef` 하나를 돌려준다 — 하지만
**mp4는 쓰이지 않는다**, 이 빌드에는 렌더러가 없기 때문이다(`es-render`는 layer 5이며
이후 마일스톤이다; `es eval run`도 같은 이유로 이미지 관측을 거부한다).

그런 실행마다 각 카메라 이름을 담은 경고를 낸다:

```
warning: camera "cam_front": info.json declares a video feature and read_episode will
         produce VideoRef placeholders, but no mp4 was written (no renderer in this build)
```

대안 — 피처를 생략하는 것 — 은 데이터셋의 `dataset_schema_hash`가 task가 한
번도 선언한 적 없는 관측을 기술하게 만들 것이고, 이후의 `es loop distill`은 그것을 실제
멀티카메라 데이터셋과 조용히 병합해버릴 것이다. 선언되었지만 부재한 video는 리더가
감지할 수 있는 매달린 참조(dangling reference)지만, 누락된 선언은 리더가 감지할 수 없는
거짓말이다.

---

## 4. `loop.jsonl` — 루프는 재현 가능하다 (§13.3)

§13.3은 각 반복이 연결되기를 원한다: 반복 *n*의 `dataset_hash`는 반복
*n-1*의 데이터에 새 개입 에피소드를 더한 것의 함수이며, `evaluation_hash`는 고정된다.
그것을 검사 가능하게 만드는 원장(ledger)은 데이터셋 root마다 하나씩 있는 append-only
파일이다:

```rust
pub struct LoopStep {
    pub kind: LoopKind,                       // Collect | Intervene | Distill
    pub inputs: BTreeMap<String, String>,     // name -> hex digest or path
    pub outputs: BTreeMap<String, String>,
    pub created: u64,                         // unix seconds
}
```

`<root>/loop.jsonl`에 한 줄당 JSON 객체 하나씩, 추가만 되고 절대 다시 쓰이지
않는다. 각 단계가 기록하는 것:

| step | inputs | outputs |
|---|---|---|
| `collect` | `task`, `observation`, `learning`, `deployment`(번들 매니페스트의 해시), `seed`, `episodes` | `content`, `schema` |
| `intervene` | `content`(이전), `segments`(개수) | `content`(이후), `schema` |
| `distill` | 각 입력 root의 `content`/`schema`, `seed`, `ratios` | `content`, `schema`, `split`, `training_hash` |

이 체인 속성은 리뷰어가 눈으로 확인하는 것이자 CI의 오라클이 검사하는 것이다:
단계 *n*의 입력 `content`는 단계 *n-1*의 출력 `content`다. `distill` 단계는
**출력 root뿐 아니라 모든 입력 root**의 `loop.jsonl`에도 추가되므로, 데이터셋 자신의
원장은 그것이 소비되었음을 기록하고, 출력의 원장은 그것이 무엇으로 만들어졌는지를
기록한다.

`created`는 이 파일에서 유일하게 재현 불가능한 값이며, `RunConfig`의 것이 그런
것과 같은 이유다(§10.4): 출처(provenance)는 정체성(identity)이 아니다.
`loop.jsonl` 안의 어떤 것도 해시로 들어가지 않는다.

`loop.jsonl`은 *학습된* 체크포인트의 `policy_hash`나 `evaluation_hash`를
의도적으로 기록하지 **않는다**: 이 세 명령 어느 것도 그것들을 만들지 않기 때문이다.
§13.3의 규율 — 데이터와 정책이 움직이는 동안 `evaluation_hash`를 고정하는 것 —
은 이미 구현된 `es eval compare`가 강제하며, 바뀐 `evaluation_hash`가 비교를 무효화하는
곳이 바로 거기다.

---

## 5. 수집: Safety Plane은 유일한 경로 위에 있다 (`INV-12`)

`Collector::run`은 `es_env::Env::step_with_policy`를 구동하며, 이는 단일
actuator 경로다: chunk buffer -> `SafetyPlane::validate` -> `ctrl`. 그 주위에는
collect-mode 분기가 없으며, 주입된 사람의 액션에 대해서도 마찬가지다.

```
        intervener says Some(a)
                 |
                 v
policy  ->  [ wrapper ] -> action chunk -> chunk buffer -> SafetyPlane -> ctrl -> physics
                 ^
        intervener says None
```

intervener는 plane 이후의 오버라이드가 아니라 `PolicyRuntime` 래퍼로 연결되며,
그 선택이 안전성 논증의 전부다: 주입된 액션은 *그저 또 하나의 chunk*일 뿐이다. 그것은
정책 chunk와 같은 envelope로 유계이고, 같은 limiter로 rate-limit되며, 같은 카운터로
세어진다. 수집과 관련해 envelope를 넓히는 것도, plane을 비활성화하는 것도 전혀
없다 — `INV-12`는 테스트를 포함해 이를 전면적으로 금지한다.

```rust
// Fn(episode, frame, &obs) -> Option<[f64; NJ]>
pub type Intervener<'a, const NJ: usize> =
    &'a mut dyn FnMut(u32, u32, &[f64]) -> Option<[f64; NJ]>;
```

`obs`는 래퍼가 그 제어 틱에 대해 건네받은 관측이며(§12.2의 raw 경로의
`qpos ‖ qvel` 행), 그래서 scripted intervener는 상태의 순수 함수이고 실행은 seed에
대해 비트 단위로 재현 가능하다. 그것이 이것을 오라클로 쓸 수 있게 만드는 이유다:
테스트의 intervener는 알려진 프레임에 주입하고, 데이터셋의 세그먼트는 정확히 그대로
돌아와야 한다.

### 5.1 주입이 라벨을 붙이는 *프레임*은 어느 것인가

주입은 제어 틱 `t`에서 내려진 결정이지만, 그것이 만드는 액션은 더 나중에, 더
오래 actuator에 도달한다: §12.3의 결정적 지연이 `latency_ticks(expected_latency_ms)`만큼
그것을 지연시키고, §8.5의 `execute_chunk = K`가 `K`틱 동안 그것을 계속 구동시킨다.
그래서 틱 `t`에서의 주입은 다음 프레임들에 라벨을 붙인다

```
[ t + latency_ticks , t + latency_ticks + K )
```

intervener가 살펴본 그 한 프레임이 아니다. 대신 결정 틱을 기록한다면 0이 아닌
추론 지연시간을 가진 모든 deployment를 잘못 라벨링하게 될 것이다 — 그리고 그것이 정상적인
경우다(§8.6).

`ChunkBlendPolicy::TemporalEnsemble` 아래에서는 내보내지는 액션이 겹치는
chunk들의 평균이므로, 라벨이 붙은 프레임은 주입된 chunk가 단독으로 결정한 것이 아니라
*기여한* 프레임이다. 이는 모델링되지 않고 여기 기록만 된다: 프레임별 블렌드 가중치는
학습 쪽의 관심사이며, 하나를 지어내는 것은 아무도 측정하지 않은 숫자를 만드는 것이 될
것이다(§1.7).

### 5.2 프레임별 `action_source` 복구하기

`DomainRunner::emit_actions`는 `ctrl`을 쓰고 `SafeAction`을 돌려주지 않으므로,
프레임별 source는 그 스텝에 걸친 plane 자신의 카운터로부터 읽어낸다:

```
fallback_activations grew  -> 2 fallback
else clamped_steps grew    -> 1 clamped
else the frame is labelled -> 3 human
else                       -> 0 policy
```

collect 실행당 env가 하나이므로, 스텝마다 정확히 한 번의 `validate` 호출이
일어나고 델타는 모호하지 않다. 반환값을 `es-env`를 통해 배관(plumbing)하는 대신 카운터를
읽는 것은 이 패킷을 이웃 크레이트의 API 밖에 머물게 하며, 그 카운터는 §10.3이 이미
의존하고 있는 기록이다.

---

## 6. 증류(Distillation): 병합, 분할, 정체성

`distill(inputs, SplitSpec { ratios, seed }, out_root)`:

1. 모든 입력의 `meta/info.json` `features` 맵과 `fps`는 **같아야** 한다.
   불일치는 그 피처 이름과 함께 거부된다: `observation.state`의 의미가 서로 다른 두
   데이터셋을 병합하면 둘 중 어느 쪽도 기술하지 않는 `dataset_schema_hash`가 만들어진다.
2. 에피소드는 입력 순서, 그다음 원래 인덱스 순으로 `0..N`으로 재색인된다 —
   전순서(total order)이므로, 같은 입력에 대한 두 번의 실행은 바이트 단위로 동일한
   parquet를 만든다.
3. 각 입력의 `meta/interventions.jsonl`은 에피소드 인덱스가 재매핑된 채로
   병합되어, 라벨이 병합을 살아남는다. (그런 파일이 없는 입력은 아무것도 기여하지
   않는다.)
4. `Split::deterministic(N, ratios, seed)` — 기존의 §19.2 seed 기반
   Fisher-Yates 절단. `train`과 `val`은 `floor(N * ratio)`를 취하고 `test`는 나머지를
   취하므로, 세 목록은 항상 `0..N`의 정확한 분할(partition)이다.
5. `DatasetIdentity::compute` -> `training_identity.json`, `split.json`(목록
   그 자체이며, 그래서 `dataset_split_hash`는 이 함수를 다시 실행해야만이 아니라 손으로도
   재현 가능하다) -> `Distill` `LoopStep`.

비율의 합이 1이어야 할 필요는 없다: 분할 함수는 clamp하고 나머지는 `test`로
떨어지는데, 이는 보수적인 방향이다(잘못 지정된 비율은 test 데이터를 새어들게 하는 대신
학습 데이터를 줄인다). 비율의 합이 1을 넘으면 거부되는데, 그 경우는 clamp이 아니라
test 세트의 조용한 잘림(truncation)이 되기 때문이다.

---

## 7. 의도적으로 뺀 것들

- **teleop 장치 드라이버 없음.** §13.1의 `--teleop <device>`는 실제 로봇
  I/O다(M3 W1). `Teleop` 라벨 variant는 그 경로로 기록된 데이터셋을 지금 기술할 수
  있도록 존재한다; 드라이버는 여기 없고 `es loop collect`에는 `--teleop` 플래그가
  없다.
- **failure-set 필터 없음.** §13.1의 `FAILURE SET` 화살표는 여러 root에 대한
  `es loop distill`이다; 어떤 에피소드가 실패했는지 고르는 것은 `es eval`의 출력이며,
  둘을 연결하는 것은 이후 패킷의 몫이다. `distill`은 주어진 것을 병합할 뿐이다.
- **데이터셋 중복 제거 없음.** 같은 root를 두 번 증류하면 그 에피소드가
  두 배가 된다. 이는 그럴 법하게 호출자의 의도일 수 있고(반복에 의한 가중치 부여),
  그렇지 않다고 추측하는 것이 그 놀라움보다 더 나쁘다.
- **비디오 없음.** §3.1.
