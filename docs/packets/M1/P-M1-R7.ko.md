<!-- Korean translation of docs/docs/packets/M1/P-M1-R7.ko.md. The English file is the working copy; regenerate this when it changes. -->

# P-M1-R7 — 참조 오라클이 설치된 CI 티어

M1 리뷰 후속 조치(`docs/reviews/M1.md`, Human decision): 어떤 CI 환경에도 MuJoCo나 PyTorch가
설치되어 있지 않아서, physics 교차 검사와 learning 교차 검사가 단 한 번도 실행되지 않은 채로
PR 게이트가 그린(green)이 된다 — spec 28.7의 게이트 2, 5, 6은 측정되는 것이 아니라 주장될
뿐이다. Spec: spec 1.4(참조 오라클은 선택이 아니라 필수다), spec 26.2(PR 게이트는 10분
이내로 유지된다), spec 28.7(이 티어가 실제로 실행하기 위해 존재하는 게이트 목록).

## context (범위)

```
.github/workflows/ci.yml
docs/packets/M1/P-M1-R7.md
```

## spec (사양)

- **`.github/workflows/ci.yml`에 두 번째 잡(job)인 `oracles`가 추가된다.** 기존의 `ci`
  잡(PR 게이트, spec 26.2의 10분 미만 예산)은 손대지 않는다 — `cargo xtask ci`는 모든
  PR과 모든 push에서 이전과 정확히 동일하게 실행된다.
- **트리거.** `oracles`는 오직 `if: github.event_name == 'push' || github.event_name ==
  'workflow_dispatch'`일 때만 실행된다 — `pull_request`에서는 절대 실행되지 않는다 —
  Python과 `mujoco`, `torch` 휠 설치가 PR 잡의 10분 예산에 맞지 않고, `main`만 필요로 하는
  검사의 비용을 모든 PR이 치르게 만들기 때문이다. 워크플로의 최상위 `on:` 블록에
  `workflow_dispatch:`가 추가되어 이 잡을 수동으로도 실행할 수 있다. `continue-on-error:
  false`(기본값이며, 명시적으로 설정됨): 여기서의 실패는 정보성 잡이 아니라 실제 CI 실패다.
- **환경.** `ubuntu-latest`, `python-version: "3.12"`의 `actions/setup-python@v5`, 그다음
  `pip install mujoco==3.13.0`(`docs/api-notes/mujoco.md`에 고정됨)과 `pip install
  torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu`(고정되며, 이것이
  전이적으로(transitively) 끌어오는 `torch==2.14.0`과 함께 `docs/api-notes/torchvision.md`와
  `docs/api-notes/torch.md`에 명시됨) — `torch`를 직접 설치하는 대신 `torchvision`을
  설치하면 `torchvision.transforms.functional`이 필요하고 이 같은 잡에서 실행되는
  `es-compile`의 observation-golden 오라클(`docs/packets/M1/P-M1-R1.md`)도 함께 커버된다.
  테스트 단계를 위해 `ES_PYTHON=python`이 export되어, 워크스페이스의 모든 오라클(`proc.rs` /
  `mjwarp.rs` / `torch_runtime.rs`의 인터프리터 검색)이 맨 `python`/`python3` 검색으로
  폴백하는 대신 바로 이 인터프리터를 찾도록 한다.
- **테스트 단계.** `set -euo pipefail` 아래에서 `cargo test -p es-physics-backend -p
  es-policy -p es-compile -- --skip mjwarp --nocapture 2>&1 | tee oracles.log`를 실행하고
  (그래야 출력이 `tee`를 거쳐 파이프되더라도 테스트 실패가 단계를 실패시킨다), 이어서
  `! grep -q '^SKIP' oracles.log`를 실행한다. 이 세 크레이트의 오라클 게이트 테스트는 각각
  의존성을 찾지 못하면 `SKIP`으로 시작하는 줄(또는 마찬가지로 `^SKIP`에 매칭되는 `SKIPPED`)을
  출력하고 그래도 통과한다(spec 1.4 자체의 관례로, 예를 들어
  `crates/es-physics-backend/src/mujoco.rs`, `crates/es-policy/src/reference.rs`);
  `mujoco`/`torch`가 실제로 설치되고 `ES_PYTHON`이 그것들을 가리키는 상태에서는, 이 잡이
  실행하는 테스트 중 어느 것도 그 경로를 타지 않아야 하므로, grep이 아무것도 찾지 못하는
  것이 바로 이 잡을 PR 잡의 두 번째 사본이 아니라 의미 있게 만드는 지점이다.
- **`mujoco_warp`는 조작되는 것이 아니라 제외된다.** `mjwarp.rs`의 두 오라클 테스트
  (`mjwarp_pendulum`, `mjwarp_against_mujoco_cpu`)는 `ubuntu-latest`에는 없는
  GPU(`mujoco_warp` + `warp`)를 필요로 한다. `--skip mjwarp`는 이들이 `SKIP`을 출력해
  grep을 실패시키게 두거나 여기서 실제로 실행될 수 없는 GPU 전용 의존성을 설치하는 대신,
  이 실행에서 그것들을 통째로 제거한다. spec 12.4에 따라 이는 `Target / Status: 미검증
  (unverified)`로 남으며, 숨기지 않고 이 문서에 이름이 명시된다.

## oracle (오라클)

```
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml', encoding='utf-8'))"
```

(잡 자체는 `main`에 대한 push나 수동 dispatch에서 GitHub Actions에 의해서만 실행해 볼 수
있다; 로컬 리뷰는 YAML이 파싱되는지와 잡의 단계들이 이 문서와 일치하는지 확인하는 데
그친다.)

## acceptance (수용 기준)

- `.github/workflows/ci.yml`은 `ci`와 `oracles` 두 개의 잡을 가진 유효한 YAML로 파싱된다.
- `ci` 잡의 단계들은 이 패킷 이전과 바이트 단위로 동일하다(여전히 정확히 `fetch-depth: 0`을
  가진 `actions/checkout@v4`, `rustfmt, clippy`를 가진 `dtolnay/rust-toolchain@stable`,
  `Swatinem/rust-cache@v2`, `cargo xtask ci`).
- `oracles` 잡의 `if:`는 이를 `push`와 `workflow_dispatch`로 제한한다; 이 잡은
  `mujoco==3.13.0`과 `torchvision==0.29.0`(`docs/api-notes/mujoco.md`,
  `docs/api-notes/torch.md`, `docs/api-notes/torchvision.md`에 있는 버전)을 설치하고,
  `ES_PYTHON`을 설정하고, `mjwarp`를 건너뛴 채 이름이 명시된 세 크레이트의 테스트를
  실행하며, 남은 테스트 줄 중 하나라도 `SKIP`으로 시작하면 실패한다.

## forbidden (금지)

- `ci` 잡의 단계, 트리거, `timeout-minutes`에 대한 어떤 변경도 — 이는 오늘 그러한 것과
  정확히 동일하게 spec 26.2의 PR 예산 이내로 유지되어야 한다.
- `crates/es-compile/src/bundle.rs`, `crates/es-telemetry/src/transport.rs`와 이들의 설계
  문서 — P-M1-R4와 P-M1-R5의 범위다.
- `mujoco_warp`를 검증됨으로 표시하는 것, 또는 `mujoco_warp`/`warp`를 설치하는 것 — 이
  패킷으로 GPU 러너가 추가되지 않는다.
- 커밋하는 것.
