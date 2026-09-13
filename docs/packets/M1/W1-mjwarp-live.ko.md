<!-- Korean translation of docs/packets/M1/W1-mjwarp-live.md. The English file is the working copy; regenerate this when it changes. -->

# W1 (후속) — 실제 하드웨어에서의 MJWarp: 라이브 테스트를 돌아가게 하기

Spec: spec 17.1 (MJWarp는 배치 GPU 백엔드다), spec 17.3 (결정성 계층: GPU 백엔드는 계층 2나
3을 선언하며, 계층 1은 오직 `mujoco-cpu`만 주장할 수 있다), spec 3.5 (계층 정의), spec 12.1
(시뮬레이션 배치 도메인), spec 12.4 (측정된 숫자를 보고하라, 결코 단일 수치가 아니라),
spec 1.7 (지어낸 API 이름은 에이전트의 상시적 실패 모드다), spec 4.3 (백엔드가 기본
경로다). M1 Wave 1, `W1-mjwarp-adapter.md`의 후속. Review class B.

`W1-mjwarp-adapter.md`는 `MjWarpBackend`를 프로토콜상으로는 완성되었지만 **엔진으로는
검증되지 않은** 채로 내보냈다: 구현한 머신에는 CUDA 디바이스가 없었으므로 모든 라이브
테스트가 건너뛰어졌고 `docs/api-notes/mujoco-warp.md`의 모든 API 이름은 *unverified*로
표시되었다. 이 패킷은 그것을 마무리한다: RTX 4060 Laptop GPU와 `mujoco-warp`가 설치된
머신, 그리고 뒤따르는 정직한 측정값으로.

## context

```
crates/es-physics-backend/python/mjwarp_ref.py
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/src/proc.rs          (공유되는 spawn / call / drop; 아래 참고)
docs/api-notes/mujoco-warp.md
docs/packets/M1/W1-mjwarp-live.md
```

## forbidden

검증이 바꾸는 것 이상의 `mapping.rs` 의미론, `mujoco.rs`, `mjcf_out.rs`, 그리고
`es-physics-backend` 밖의 모든 crate. `es` CLI(`es backend compare`)는 별도의 패킷이다.

## spec

1. 설치된 `mujoco_warp`에 대해 라이브 테스트를 실행해 통과시키며, `mjwarp_ref.py`를
   추측한 API가 아니라 **설치된** API에 맞춰 고친다.
2. 결정성을 정직하게 측정한다: 같은 저장 상태에서 시작한 두 번의 실행이 비트
   동일한지 아닌지를, 관측된 허용오차와 함께 기록한다.
3. 200틱에 걸쳐 `compare_backends(mujoco-cpu, mjwarp)`를 측정하고 `max |dqpos|`를
   기록한다.
4. `docs/api-notes/mujoco-warp.md`를 갱신한다: 실행된 것에서 *unverified*를 걷어내고,
   실제 버전을 고정하고, 아직 확인되지 않은 것의 명시적 목록을 유지한다.

## oracle

```
ES_PYTHON=<venv python> cargo test -p es-physics-backend -- --nocapture
```

라이브 테스트는 실행되면 `RAN <name>`을, 그렇지 않으면 `SKIP <name>: <reason>`을
출력하므로, 게이트는 통과한 실행과 공허하게 건너뛴 실행을 구분할 수 있다:

```
cargo fmt -p es-physics-backend --check
cargo clippy -p es-physics-backend --all-targets -- -D warnings
ES_PYTHON=... cargo test -p es-physics-backend -- --nocapture | grep -E 'SKIP|RAN|qpos|test result'
```

## 실제로 무엇이 잘못되어 있었는가

**버그는 하나였고, API 이름이 아니었다.** 어댑터가 추측한 모든 이름 — `put_model`,
`put_data(..., nworld=)`, `step`, `forward`, `Data` 필드 이름, `nworld`-major
레이아웃 — 은 `mujoco-warp 3.13.0`에 대해 옳았다. 실패한 것은 **`warp`가
초기화 배너를 stdout에 출력한다**는 것이었고, stdout은 JSON-lines 프로토콜이 쓰는
자리였다:

```
Protocol("expected value at line 1 column 1 in `Warp 1.17.0 initialized:`")
```

수정: 어떤 임포트보다도 먼저 진짜 stdout 핸들을 붙잡아 두고 `sys.stdout`을 이미
Rust 쪽이 버리고 있는 `sys.stderr`로 돌린다. 두 줄. 남겨둘 만한 교훈은 Python
라이브러리의 *stdout에 대한 부작용*이 함수 이름만큼이나 그 API의 일부라는 것이며,
stdout을 소유하는 프로토콜은 그것을 방어해야 한다는 것이다.

진짜로 틀렸던 고정값은 버전이었다: 노트는 `mujoco-warp==0.1.0`을 주장했지만, 패키지
버전은 `mujoco`와 발맞춰 가며 설치된 릴리스는 **3.13.0**이다.

## accepted

- `mjwarp_pendulum` **RAN**: 로드되고, 100틱을 스텝하고, 유한하며, `n_envs = 2`가
  독립적이고, 리셋된다.
- `mjwarp_against_mujoco_cpu` **RAN**.
- `mjwarp_runs_agree_to_the_declared_tier` **RAN** (신규).
- 고정된 버전: `mujoco-warp 3.13.0`, `warp-lang 1.17.0`, `mujoco 3.13.0`, CPython
  3.12, CUDA Toolkit 12.9 / 드라이버 13.1, RTX 4060 Laptop GPU (8 GiB, sm_89).

### measured

| 측정 항목 | 값 |
|---|---|
| Run-to-run, 200틱, 2 world, 같은 저장 상태 | **비트 동일**, `max \|delta\| = 0` |
| `mujoco-cpu` 대비 `max \|dqpos\|`, 200틱 | **7.41e-8** |
| `mujoco-cpu` 대비 `max \|dqvel\|`, 200틱 | 7.76e-7 |
| 에너지 proxy 델타 | 3.16e-6 |
| 발산 틱 (1e-6 허용오차) | 없음 |

**선언된 계층은 그대로 2(`CrossBackend`)다.** Spec 17.3은 오직 `mujoco-cpu`만 계층
1을 선언한다고 명시하며, 한 GPU와 한 드라이버에서 한 씬이 재현된다는 것은 관측이지
보장이 아니다. 그래서 `mjwarp_runs_agree_to_the_declared_tier`는 *tier-2* 계약
(`delta < 1e-9`)을 단언(assert)할 뿐이며, 그 실행이 우연히 비트 단위였는지는 단지
출력할 뿐이다. 오늘 재현 가능한 백엔드가 다음 드라이버 업데이트에서 실패하는
테스트가 되어서는 안 된다.

## 이 패킷이 걸려 넘어졌던 함정과 그 수정

처음 통과한 `mjwarp_against_mujoco_cpu`는 `max |dqpos| = 0.000000e0`을 보고했다 —
그리고 그것은 **무의미했다**. `compare_backends`는 제어 없이 리셋하고 스텝하는데,
픽스처 진자는 수직으로 매달려 있어 평형점이었다: 두 백엔드 모두 200틱 동안 0에
머물렀고 비교는 아무것도 비교하지 않았다. 아무것도 증명하지 않는 초록색 테스트는
빨간색 테스트보다 나쁘다.

이제 compare 테스트는 **수평으로** 시작하는 막대를 사용하므로 `qpos = 0`은
평형점이 아니며 200틱은 실제로 움직인다. 향후의 어떤 크로스-백엔드 픽스처든 같은
점검이 필요하다: 궤적이 자명하지 않음을 단언하라, 그러지 않으면 그 허용오차는
얻어진 것이 아니다.

## 부수적으로: 세 번째 out-of-process 백엔드가 도착했다

`mjwarp.rs`는 자신의 중복된 spawn / call / drop 삼종 세트가 "세 번째
out-of-process 백엔드가 등장하면" `proc.rs`로 합쳐져야 한다는 `ponytail:` 메모를
갖고 있었다. `W4-newton-backend.md`가 바로 그 세 번째 백엔드이므로, 그 합침이
여기서 일어났다: `proc.rs`는 `Process::spawn_with(script, engine)`,
`import_available(modules, what)`, `model_info(...)`(공유되는 `LoadReply` →
`ModelInfo` 이름 해석)을 얻었고, `mjwarp.rs`는 174줄을 잃었다. 두 파일에 걸친 순
변화는 **−63줄**이며, `newton.rs`는 이 모두를 물려받는다.

## still open

- `contact.condim = 6`은 미검증.
- 2 world를 넘는 배치 폭. `MAX_ENVS = 8192`는 선언되었을 뿐 측정되지 않았다
  (spec 12.4).
- 충돌이 많은 씬: 측정된 모든 것은 충돌이 없는 단일 힌지다.
- `docs/api-notes/mujoco-warp.ko.md`는 여전히 예전의 *unverified* 배너와 잘못된
  `0.1.0` 고정값을 담고 있다. 이는 이 패킷의 범위 밖이다; 후속 `docs:` 커밋이
  필요하다.
