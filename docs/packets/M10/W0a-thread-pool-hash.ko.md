# M10 W0a — 정책 런타임의 스레드 수가 `runtime_hash`에 들어간다

스펙: §5.3(2026-09-22에 추가된 문장: `runtime_hash`는 정책 런타임의 intra-op 스레드 수를 덮는다; 풀은
고정하지 않는다), §10.4(평가의 바이트 동일성), §28.13 규칙 1과 파동 0, §3.5 계층 1. 리뷰:
`docs/reviews/M8.md` S-1과 사람 결정 "워커 스레드 풀" — 소유자가 2026-09-22에 **선택지 2(해시에
말한다)**를 골랐다. 설계 노트: `docs/design/evaluation-execution.md` 2.7(세 선택지; 이 패킷이 그 아래에
결정 단락을 쓴다). 유형 B, D 행 하나.

## 질문

`es eval run --jobs 8`과 `--jobs 1`은 커밋된 데모 문서에서 다른 `report.json` 바이트를 낸다 — 워커마다
Torch 풀이 `cores / N`으로 제한되고 Torch의 CPU 추론은 intra-op 스레드 수에 따라 비트가 다르기 때문이며,
`execution_hash`는 그것을 모른다. **torch 서브프로세스가 실제로 도는 스레드 수를 보고하고
`TorchRuntime::runtime_hash`가 그것을 덮으면, 다른 스레드 수에서 얻은 두 보고서는 두 `execution_hash`를
얻고 같은 스레드 수는 모든 바이트를 유지하며, 다른 것은 아무것도 움직이지 않는가?**

## 명세

* `python/torch_ref.py`: 핸드셰이크 메시지에 `"threads": torch.get_num_threads()`(Torch가
  `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / 기본값에서 해석한 intra-op 풀). `PROTOCOL_VERSION` 1 → 2
  (`crates/es-policy/src/torch_runtime.rs`): 와이어 스키마가 바뀌었고 프로토콜 번호는 이미 `runtime_hash` 입력이다.
* `TorchRuntime::runtime_hash()` = `blake3(RUNTIME_TAG ‖ "torch" ‖ protocol ‖ version ‖ threads u32 LE)`.
  `runtime_hash_of`(`crates/es-policy/src/runtime.rs:143-150`)에 다른 모든 런타임이 `1`을 넘기는
  `threads: u32` 매개변수를 더하거나, `TorchRuntime`이 필드 하나 더한 자기 다이제스트를 계산하거나 —
  에이전트의 선택; `FakePolicy` / `CpuPlan`의 다이제스트는 움직여도 되지만(고정하는 곳 없음) 결정적이어야 한다.
* `EvaluationLock`(`crates/es-eval/src/runner.rs:322-328`)에 `runtime_threads: Option<u32>`
  (`#[serde(default, skip_serializing_if = "Option::is_none")]`), 새 `PolicyRuntime` 비의존 접근자로 채움 —
  평가기가 런타임에 보고된 수를 묻는다(trait이 아니라 `TorchRuntime`의 메서드: INV-17은 런타임 하나를 위해
  trait을 넓히는 것을 금하며, CLI는 자기가 만든 런타임을 안다). 커밋된
  `tests/fixtures/visible-learning/run/evaluation.lock`(placeholder)은 계속 역직렬화된다.
* `es eval run` 도움말(`crates/es/src/cmd/eval.rs:56-81`): 바이트 동일성 주장은 "같은 정책 런타임 스레드
  수에서, 그 수는 `evaluation.lock`이 기록하고 `execution_hash`가 덮는다"로 읽힌다.
* **풀에 관한 것은 아무것도 바뀌지 않는다**: `shard_thread_env`, `cores / N` 제한, `--jobs 1`이 주변 풀을
  물려받는 것은 그대로; 커밋된 수치는 움직이지 않는다.

## context

```
python/torch_ref.py
crates/es-policy/src/torch_runtime.rs
crates/es-policy/src/runtime.rs
crates/es-policy/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/packets/M10/W0a-thread-pool-hash.md
docs/packets/M10/W0a-thread-pool-hash.ko.md
```

## 오라클

1. `cargo test -p es-policy torch_runtime_hash_covers_the_thread_count`(`ES_PYTHON`; 없으면 이유를 찍고
   건너뜀): 자식에 `OMP_NUM_THREADS=1`과 `=2`를 내보내 띄운 두 `TorchRuntime`이 1과 2를 보고하고 다르게
   해시된다; 같은 수의 둘은 같게 해시된다; `PROTOCOL_VERSION == 2`는 옛 `torch_ref.py`가 이름으로 거부한다
   (기존 버전 불일치 거부).
2. `cargo test -p es-eval` — `FakePolicy`의 `--jobs 1 / 2 / 4` 바이트 동일 테스트가 녹색 유지
   (`runtime_threads`는 풀이 없는 런타임에서 `None`).
3. 서버(Linux, 16코어; CPU 큐 `~/artifacts/plan-w/queue/cpu.lock`; 아티팩트 `~/artifacts/plan-w/w0a/`),
   **커밋된** 데모 문서와 U0의 번들로, nominal 스위트만(`evaluation-execution.md` 2.7이 했듯 `evaluation.toml`을
   제한): `--jobs 1`, `--jobs 8`, `OMP_NUM_THREADS=2 --jobs 1`. 표 하나: `evaluation_hash`, `execution_hash`,
   락의 `runtime_threads`, `report.json`의 blake3, wall-clock. 기대: `evaluation_hash`는 셋 다 같다; 2행과 3행은
   wall-clock 빼고 모든 열이 같다; 1행은 `execution_hash`·`runtime_threads`·보고서 다이제스트가 다르다 —
   2.7의 발견이 이제 체인에 보인다. 2.7 아래에 "Decision (M10/W0a)"로 쓴다.
4. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W0a-thread-pool-hash.md`.

## 수용

오라클 1~4; 2.7 결정 단락과 한국어 짝; 도움말 문구.

## 금지

풀을 고정하거나 바꾸는 것(`shard_thread_cap`, `THREAD_ENV_VARS`, `--jobs 1`의 동작); `hardware_capability`
슬롯(L24는 다른 패킷); `PolicyRuntime` 넓히기(INV-17); `docs/ARCHITECTURE*.md`(문장은 들어가 있다);
`tests/golden/**`; 커밋된 픽스처 일체.
