<!-- Korean translation of docs/packets/M4/P-M4-R6.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R6 — GPU SKIP이 필수 실행을 실패로 만들 수 있다

Spec: §1.4(오라클은 머신이 그것을 실행할 수 있든 없든 존재하며 — 어느 쪽인지
말해준다), §1.6 / §26.2(`xtask`가 유일한 검증 진입점이며, PR 티어는 10분 미만이다).
리뷰 발견 사항: `docs/reviews/M4.md` S-7 — `es-gpu/tests/determinism.rs:35,46`과
`es-render/tests/render.rs:57`는 `SKIP`을 출력한 뒤 `ok`로 보고하는데,
`xtask/src/main.rs:45`는 `cargo test`를 **`--nocapture` 없이** 실행했으므로 그 줄들은
로그에 도달하지 못했다. GPU가 없는 머신이 GPU 작업에 대해 녹색(성공)을 보고한 것이다.
`.github/workflows/ci.yml`에 `! grep -q '^SKIP'` 가드가 있지만, 이는 `oracles` job의
Python 크레이트 세 개에만 적용된다. 또한 S-12 — `es-gpu/src/buffer.rs:144`는
`gpu-allocator`가 패딩한 `allocation.size()` 바이트를 반환했으므로
`download_f32().len()`이 요청한 개수를 초과할 수 있었다; 그리고 S-7의 나머지 절반 —
`es-policy/tests/act_checkpoint.rs:113`는 진짜 레퍼런스 오류를 "lerobot not installed"와
같은 SKIP으로 뭉뚱그렸다.

## context (범위)

```
xtask/src/main.rs
crates/es-gpu/src/buffer.rs
crates/es-gpu/tests/determinism.rs
crates/es-policy/tests/act_checkpoint.rs
.github/workflows/ci.yml
docs/design/gpu-foundation.md
docs/packets/M4/P-M4-R6.md
```

## spec (사양)

- `xtask/src/main.rs`: `cmd_ci`의 테스트 단계는 `run_tests(root)`가 된다. 이는 stdout을
  파이프로 연결해 `cargo test --workspace --features es-ir/testing -- --nocapture`를
  실행하고, 도착하는 **모든 줄을 그대로 전달**하며(사람이 보는 로그는 그대로다)
  `is_gpu_skip`이 매치하는 줄들을 수집한다.
- `is_gpu_skip(line: &str) -> bool`는 순수 함수이며 유닛 테스트가 있다: `line`은 0번째
  열에서 `SKIP`으로 시작해야 하고, 나머지 부분은 대소문자 구분 없이 `gpu`, `vulkan`,
  `render`, `slangc`, `device` 중 하나를 포함해야 한다. 그 외의 모든 것 — `SKIP nostd:`,
  `SKIP: ES_ACT_CHECKPOINT is unset`, `SKIP: no Python with torch`, 메시지 안에
  들여써진 `SKIP` — 은 GPU skip이 아니다.
- `ES_REQUIRE_GPU=1`이면 수집된 각 줄은
  `FAIL ES_REQUIRE_GPU=1 but a GPU oracle did not run: …`로 출력되고 해당 단계는
  실패한다; 설정되어 있지 않으면 각각
  `NOTE GPU oracle skipped (ES_REQUIRE_GPU unset): …`로 출력되고 단계는 통과한다.
  `nostd --require`와 같은 형태다.
- `crates/es-gpu/src/buffer.rs`: `Buffer`는 `size`(실제로 할당된 것: `bytes.max(4)`에
  이어 얼로케이터의 패딩) 옆에 호출자가 요청한 바이트 수인 `len: u64`를 추가로 갖는다.
  `download`는 `len`으로 잘라내므로 정확히 요청된 바이트 수만 반환한다.
- `crates/es-policy/tests/act_checkpoint.rs`: `is_missing_dependency(&str) -> bool` —
  `cannot start \``, `ModuleNotFoundError:`, `ImportError:`에 대해서만 참이다. 이와
  매치하는 레퍼런스 실패는 SKIP된다; 그 외에는 `panic!`한다. 설치된 LeRobot이 예외를
  던졌다면 그것은 환경 누락이 아니라 하나의 결과이기 때문이다.
- `.github/workflows/ci.yml`: PR job의 `cargo xtask ci` 단계에 이 러너에는 GPU가
  없다는 것, 그래서 `ES_REQUIRE_GPU`가 의도적으로 설정되어 있지 않다는 것, `1`로
  설정하면 무슨 일이 일어나는지를 기록하는 주석을 단다. `oracles` job은 변경되지
  않는다.

## oracle (오라클)

```
cargo test -p xtask
cargo test -p es-gpu -- --nocapture
ES_REQUIRE_GPU=1 cargo xtask ci
```

`xtask` 자체 테스트는 GPU 없이도 `is_gpu_skip`을 양쪽 방향에서 다룬다(GPU skip 줄
4개는 매치, non-GPU 줄 6개는 거부). `download_returns_exactly_the_requested_length`는
`Storage`와 `Staging` 버퍼 양쪽에서 `f32` 세 개를 다운로드하여 길이와 내용을
단언한다.

## acceptance (수용 기준)

- `cargo xtask ci`는 테스트 출력을 표시한다(`--nocapture`로 실행되므로), 따라서 `SKIP`
  줄이 로그에 보인다.
- GPU가 있는 머신에서는 `ES_REQUIRE_GPU=1 cargo xtask ci`가 0으로 종료하며 건너뛴 GPU
  오라클이 없다고 보고한다; 어떤 GPU 오라클이라도 건너뛰면 해당 줄을 명시하며 0이
  아닌 코드로 종료한다.
- `ES_REQUIRE_GPU`가 없으면 GPU가 없는 머신도 여전히 통과하며, skip마다 `NOTE`
  하나씩을 남긴다.
- `Buffer::download()` / `download_f32()`는 정확히 요청된 바이트/요소 개수를 반환한다.
- 예외를 던지는 LeRobot 레퍼런스는 건너뛰는 대신
  `a_real_lerobot_act_checkpoint_reproduces_its_actions`를 실패시킨다; 인터프리터나
  모듈이 없을 때만 건너뛴다.

## forbidden (금지)

- `cargo xtask ci`가 실행하는 내용은 `--nocapture`와 스캐너를 넘어서서는 바뀌지
  않는다 — 단계 목록, `context-budget`, `layering`, `nostd`, `check-spec-refs`,
  `verify-goldens`는 그대로다.
- 새 CI job 없음, 그리고 `oracles` job의 기존 `tee` + `! grep -q '^SKIP'`에도 변경
  없음.
- 어떤 SKIP도 조용한 통과로 약화시키지 않는다: 스캐너는 skip을 failure로 바꿀 뿐,
  failure를 skip으로 바꾸는 일은 결코 없다.
- `crates/es-render/**`, `crates/es-env/**`, 또는 병렬 진행 중인 다른 M4 패킷들이
  소유한 크레이트에는 손대지 않는다; `es-policy`는 위의 SKIP/FAIL 분리를 위해서만
  건드린다.
