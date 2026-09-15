# M5 V5 — 빠른 사이클

설계 노트: `docs/design/visible-learning.ko.md` 섹션 7.11. V2b(베이크와 학습 진입점)와 V3(퍼터베이션
스위트, 프레임 덤프, `events.json`)에 의존한다. **결과는 하나도 바꾸지 않는다.** 섹션 7.9와 섹션 7.10이
기록한 모든 숫자가 바이트 단위로 그대로 돌아와야 한다.

## 문제

데모의 수집 -> 학습 -> 평가 한 사이클은 오라클 서버에서 순차 실행 기준 대략 한 시간으로 측정되었다.

| 단계 | 측정값 (V2b/V3) |
| --- | --- |
| `es eval run` nominal, 16 에피소드 | 약 5분 |
| `es eval run` 6 스위트 96 에피소드 | 약 28분 |
| `train_act.py`, 배치 8로 20,000 스텝 | 약 11분 |

그중 어느 것도 기계가 더 빨리 할 수 없는 계산이 아니다. 기계에게 시키지 않고 있는 계산일 뿐이다. 물리
백엔드는 프로세스마다 env 하나를 가진 파이썬 서브프로세스이고(설계 노트 섹션 7.6), `Evaluation::run`은
`BatchDomains::single_env()`를 하드코딩하며, 학습 루프는 베이크된 세트 전체가 디바이스에 올라갈 수 있을
때에도 샘플을 하나씩 호스트 메모리에서 복사한다.

## 순차 러너가 에피소드 사이로 넘기는 상태 — 분할을 넓히기 전에 읽을 것

당연해 보이는 분할은 "워커당 에피소드 하나"다. 여기서는 틀렸고, 이유는 `crates/es-eval/src/runner.rs`에
있다. 하나의 **셀**(스위트 하나) 안에서 `run_with_frames`는,

1. `Env`를 **하나** 만들어 그 스위트의 모든 에피소드에 재사용한다. `Env::reset`은 env별 에피소드
   카운터를 증가시키고(`crates/es-env/src/env.rs:224`) `RandomizationPlan::apply`가 그것을 키로 쓴다.
   따라서 에피소드 5의 초기 상태는 에피소드 0..4를 실제로 돌리지 않으면 재현할 수 없다.
2. 셀마다 `SafetyPlane`을 **하나** 만들고 `safety.counters()`를 셀 전체 기준으로 보고한다.
   `envelope_violation_rate`와 `chunk_underrun_rate`는 셀 단위 합계다.
3. `env.metrics()`를 셀 전체에 걸쳐 누적한다.
4. 단조 증가 `seq`를 셀의 에피소드들 사이로 넘긴다(스펙 8.6: plane은 `seq`가 커졌을 때만 새 청크로 본다).

그러므로 스위트 안의 에피소드들은 **독립이 아니고**, `--jobs`는 독립인 척해서는 안 된다. **셀**은
독립이다. 자기 `Env`, 자기 `SafetyPlane`, 0에서 시작하는 `seq`, 첫 에피소드를 포함한 모든 에피소드에서의
`plan.reset()`. 셀들이 공유하는 것은 컴파일된 `CpuPlan`(에피소드마다 reset되므로 새로 컴파일한 plan과
reset한 plan은 같다)과 `PolicyRuntime`(피드포워드다. `TorchRuntime::infer`는 호출 사이 상태가 없고,
시간 윈도우는 policy가 아니라 plan에 있다)뿐이다.

**따라서 분할은 셀 단위 라운드로빈이다. 셀 `c`는 샤드 `c % N`에 속한다.** 데모 평가에는 스위트가 여섯 개
있고 28분이 거기 있다. nominal 16 에피소드 실행은 스위트 하나라 `--jobs`가 줄여주지 않는다. 그것이 이
패킷의 정직한 한계이고, 우회하지 않고 명시한다. 더 잘게 쪼개려면 `Env`에 에피소드 카운터를 옮길 방법을
줘야 하는데, 여기서 `es-env`는 금지다.

## context

```
crates/es-eval/src/runner.rs
crates/es-eval/src/lib.rs
crates/es-eval/tests/evaluation.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
crates/es-policy/tests/ir_training.rs
python/es/train_act.py
python/es/README.md
python/es/README.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V5-fast-cycle.md
docs/packets/M5/V5-fast-cycle.ko.md
```

새 크레이트도 새 의존성도 없다. 워커는 같은 `es` 바이너리이고(`std::env::current_exe`), 전송은 JSON
파일이다. 둘 다 이미 링크되어 있다.

## spec

- **§10.4, §10.5.** `--jobs N`은 스케줄링 선택이지 다른 평가가 아니다. 부모가 셀을 분할하고, 자식이
  그것을 돌리고, 리포트와 `evaluation_hash`, `execution_hash`, `evaluation.lock`은 오직 부모가
  계산한다. 순차 경로가 만드는 것과 같은 `measured`/`samples` 맵 위에서 같은 `judge`로 계산한다.
  셀 집합이 정확히 `0..suites.len()`, 각각 한 번씩이 아닌 병합은 거절한다. 그러지 않으면 스위트 하나가
  빠진 리포트가 올바른 `evaluation_hash`를 달게 된다.
- **§10.4 (셀 순서).** 각 워커의 셀에는 `EvaluationIr::suites` 인덱스가 붙고 병합은 그 인덱스에 대한
  안정 정렬이다. 그래서 병합된 `cells` 배열은 자식이 어떤 순서로 끝났든 순차 실행이 썼을 순서다.
  `HashMap` 순회도, 벽시계도, 누가 먼저 답했는지에 대한 의존도 없다(§3.4).
- **§3.5 티어 1.** 순차 경로는 말 그대로 `N = 1`짜리 샤드 경로다. `run_with_frames`가
  `run_shard((0, 1))`과 `merge`를 부른다. `N = 1`에서의 바이트 동일성은 구조로 보장되고, 오라클은
  `N = 4`에서 고정한다.
- **§12.4.** 아홉 메트릭 중 둘 — `physics_steps_per_sec`, `actions_per_sec` — 은
  `EnvMetrics::simulation_wall`에서 나오고, 샤드 실행은 각 워커의 벽시계를 잰다. 순차 경로에서도 이미
  실행 간 재현되지 않는다. 데모의 `tests/fixtures/visible-learning/evaluation.toml`도 오라클 픽스처도
  둘 다 선언하지 않는다. 고치지 않고 기록한다. 고치는 일은 `es-env`의 몫이고 여기서는 금지다. 이 패킷의
  어디에서도 단일 `step/s` 수치를 보고하지 않는다.
- **§7.2, INV-14.** 손대지 않는다. 어떤 샤드도 관측을 리샘플하거나 변환하거나 다시 돌리지 않는다. 각
  셀의 프레임은 같은 `CpuPlan`에서 같은 `capture`가 자기 `<frames>/<suite>-<NN>/` 디렉터리에 쓴다.
  셀 이름은 전역적으로 유일하고 샤드는 서로소인 셀을 가지므로, 자식들은 병합 단계도 충돌도 없이 하나의
  프레임 디렉터리에 쓴다.
- **§9.** Safety Plane은 그대로다. 전과 같이 셀당 plane 하나가 모든 스텝을 검증한다. 여기서 추가된 어떤
  경로도 그것을 넓히거나 건너뛰거나 끄지 않으며(INV-12), `SafetyPlane::validate`의 시그니처도 그대로다
  (INV-13).
- **§2.3, §8.1.** `train_act.py`는 계속 옵티마이저만 소유한다. `--resident-gpu`는 수학이 아니라 베이크된
  텐서를 옮긴다. 배치 순서, dtype, loss가 동일하다. `--amp bf16`과 `--compile`은 옵트인이고 비트를
  **바꾼다**. 스윕용이지 숫자를 인용할 실행용이 아니다. 셋 중 무엇도 해시 슬롯에 들어가지 않는다.
- **§8.9.** fp32 추론 동등성 검사(`compare_actions`, `Tolerance::TIER4_FP32`)가 남는 게이트이고, 기본값
  으로 학습한 체크포인트 위에서 남는다.
- **INV-16.** safetensors in, safetensors out. `--resident-gpu`는 텐서가 어디 사는지를 바꿀 뿐 어떤
  포맷에서 읽었는지를 바꾸지 않는다.
- **INV-17.** 새 트레이트 없음. `Shard`/`ShardCell`은 평범한 serde 구조체이고 `Evaluation::run_shard`와
  `Evaluation::merge`는 기존 네임스페이스 구조체의 고유 메서드 둘이다.
- **§1.5.** `es-eval`은 2,595줄, `es`는 4,261줄이다. 이 패킷은 둘 합쳐 약 350줄 이내로 잡는다.

## oracle

PR 티어. 파이썬도 GPU도 네트워크도 필요 없다.

```
cargo test -p es-eval --test evaluation -- sharding
cargo test -p es --test cli -- eval_run_jobs
cargo test -p es --bin es -- cmd::eval
```

1. `sharding_the_cells_produces_a_byte_identical_report` — 인테스트 `FakeBackend`/`FakePolicy`로
   구동하는 4 스위트 이미지 평가 하나를 세 가지로 돌린다. `Evaluation::run_with_frames`로 순차, 샤드
   하나(`N = 1`), 샤드 넷(`N = 4`, **샤드마다 새 `FakePolicy`** — 별도 프로세스가 주는 것과 같다).
   `report.json`, `evaluation.lock`, `events.json`을 `write_artifacts`/`write_events`로 쓰고 **바이트**로
   비교하고, 모든 셀의 프레임 디렉터리를 파일 단위 바이트로 비교한다.
2. `a_merge_missing_a_cell_is_refused` — 샤드 하나의 셀을 빼면 무엇을 받았는지 이름을 붙인
   `EvalError::Shard`이지 리포트가 아니다.
3. `eval_run_jobs_zero_is_a_usage_error` — `es eval run --jobs 0`은 아무것도 열기 전에 2로 종료한다.
4. `cmd::eval` 단위 테스트 — `--shard-out` 없는 `--shard` 거절, `--jobs > 1`과 함께 쓴 `--shard` 거절,
   `--shard 4/4` 거절, 그리고 샤드 번호·종료 코드·자식의 마지막 stderr 줄을 담은 오류 하나를 만드는
   `shard_failed`.

오라클 티어 (`torch`가 있는 `ES_PYTHON` 필요. 없으면 이유를 출력하고 스킵):

```
cargo test -p es-policy --test ir_training -- --ignored --nocapture resident_gpu_does_not_move_the_loss
cargo test -p es-policy --test ir_training -- --ignored --nocapture act_training_uses_baked_observations
```

5. `resident_gpu_does_not_move_the_loss` — 같은 베이크 픽스처, 같은 `--seed`로 옵티마이저 40 스텝을
   `--resident-gpu` 없이 한 번, 있이 한 번 돌리고 두 `--loss-curve` JSON 파일을 **바이트 단위**로
   비교한다. `--amp bf16`과 `--compile`은 이 비교에 일부러 **넣지 않는다**. 비트를 바꾸고, 그래서
   옵트인이다.
6. `act_training_uses_baked_observations` — 기본값에서 그대로, `TorchRuntime`을 통한 티어 4 fp32
   왕복까지 포함해서. 이 패킷은 이것을 움직이면 안 된다.

서버 티어 (2단계, 오라클 서버. 게이트가 아니라 측정이다):

```
# 기준선, 이미 측정됨 (V3, 설계 노트 섹션 7.8/7.10)
es eval run --config evaluation.toml --policy policy.esb --scene scene.xml \
    --out eval-seq --frames frames-seq
# 같은 평가, 워커 여섯
es eval run --config evaluation.toml --policy policy.esb --scene scene.xml \
    --out eval-par --frames frames-par --jobs 6
cmp eval-seq/report.json eval-par/report.json
cmp eval-seq/evaluation.lock eval-par/evaluation.lock   # `created`는 다르다. 나머지를 비교
cmp eval-seq/events.json eval-par/events.json
# 학습, 기본값과 resident
python/es/train_act.py --module build --baked baked --out a.safetensors \
    --batch 8 --seed 0 --checkpoint-at 20000 --device cuda
python/es/train_act.py --module build --baked baked --out b.safetensors \
    --batch 8 --seed 0 --checkpoint-at 20000 --device cuda --resident-gpu
```

| 항목 | Target | Status |
| --- | --- | --- |
| 6 스위트 96 에피소드 실행, `--jobs 6` | 28분에서 약 5-6분 | **unverified** |
| `report.json` / `events.json`의 순차 실행 대비 | 바이트 동일 | **unverified** |
| 배치 8로 20,000 스텝, `--resident-gpu` | 11분보다 빠르게 | **unverified** |
| 20,000 스텝, `--resident-gpu --amp bf16` | 더 빠르게, 비트는 다름 | **unverified** |

위의 모든 행은 서버에서 측정되어 설계 노트 섹션 7.11에 숫자가 적히기 전까지 `Status: unverified`로
남는다. 1단계에서는 이 중 어느 것도 측정값으로 인용할 수 없다.

## acceptance

- `es eval run --jobs N`(기본 1)이 `--jobs 1`과 바이트 동일한 `report.json`, `evaluation.lock`
  (`created` 제외), `events.json`을 만들고, 셀별 프레임도 바이트 동일하다.
- `--jobs 0`은 거절된다. `--shard i/N`은 `--shard-out`을 요구하고 `--jobs > 1`을 거절한다.
- 실패한 워커는 샤드 번호, 종료 코드, 마지막 stderr 줄을 담은 오류 **하나**로 드러난다. 부분 리포트는
  절대 쓰이지 않는다.
- `train_act.py --resident-gpu`가 같은 seed에서 CPU 기본 경로와 비트 단위로 같은 loss 곡선을 만든다.
  `--amp bf16`과 `--compile`은 옵트인이고 비트를 바꾼다고 문서화된다.
- `act_training_uses_baked_observations`가 그대로 통과한다.
- `cargo xtask ci` 통과.

## forbidden

- `es-safety` — 어떤 이유로도 파일 하나 건드리지 않는다. plane은 셀 단위였고 셀 단위로 남는다.
- `es-env` — 에피소드 단위 분할을 가능하게 만들 `Env`의 에피소드 카운터를 포함해서. 그것은 자기 오라클을
  가진 다른 패킷이다.
- 물리 백엔드(`es-physics-backend`, `es-physics-core`)와 백엔드 서브프로세스 프로토콜.
- **측정된 숫자의 변경 일체.** 설계 노트 섹션 7.8/7.9의 성공률도, `ir_training.rs`의 loss 임계값도,
  골든도, `--batch`의 기본값 8도 아니다. 숫자가 움직이면 패킷에 결함이 있는 것이고, 고칠 대상은 숫자가
  아니다.
- `Evaluation::run`과 `run_with_frames`의 기존 시그니처, 그리고 `BatchDomains::single_env()`.
- 런타임, 스레드 풀, 스케줄러 크레이트 추가. 백엔드가 프로세스이고 GIL이 실재하므로 워커는 프로세스다.
