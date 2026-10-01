# M18 plan K — 파일 단위 구조 정리와 M17이 남긴 것

> 소유자, 2026-10-02: "남은 작업 수행 및 코드 구조 개선 요청". 범위를 묻자 **파일 단위만**을 골랐다.
> 크레이트 분리와 스펙 변경은 하지 않는다. 세어지는 코드가 1,000줄을 넘는 소스 파일은 모두 모듈로 나누고,
> 크레이트는 그대로 둔다. 각 작업은 §1.2 패킷이며, **Files** 블록이 그 `context`다.

## 전역 제약

- **순수한 이동.** 동작 변경도, 공개 항목의 이름 변경도 없다.
  - 다른 크레이트가 쓰는 경로는 모두 그대로 동작한다. 부모 파일은 원래 경로에 남겨 둔다(Rust의
    `foo.rs` + `foo/part.rs` 배치). 그래서 `docs/`에 적힌 여러 파일 경로가 그대로 맞고, 부모 파일이
    옮긴 것을 다시 내보낸다.
  - 테스트는 자기가 시험하는 코드와 함께 옮기거나 그 자리에 둔다. 어떤 테스트도 약해지지 않는다.
  - 골든, 해시, 문서는 하나도 바뀌지 않는다.
- **책임으로 나눈다.** 줄 수로 자르지 않는다. 새 모듈 하나는 하나의 응집된 일을 하고 그 일로 이름을
  붙이며, 세어지는 줄 기준 약 700줄 이하를 목표로 한다. 이미 하나의 일로 읽히는 파일은 더 커도 되지만
  이유를 적는다.
- **격리.** 에이전트마다 워크트리와 자기 `CARGO_TARGET_DIR=target/wt-<packet>`을 쓰고, 동시에 셋까지만
  돌린다. 푸시하지 않는다. 학습 실행은 하지 않는다. 커밋은 Conventional Commits, 영어, 트레일러 없음.
  문서는 한국어 짝을 함께 둔다.
- **게이트:**
  - `cargo fmt --all --check`.
  - 손댄 크레이트들에 `clippy -D warnings --all-targets`.
  - 손댄 크레이트들의 테스트와, 그것을 다루는 `es` CLI 테스트(`ES_PYTHON`과 함께).
  - xtask `layering`, `context-budget`, `check-spec-refs`, `verify-goldens`.
  - 오케스트레이터는 병합된 웨이브마다 `cargo xtask ci`를 한 번 돌린다.

## 대상 파일 (세어지는 코드 줄, 테스트 제외, 2026-10-02)

| 파일 | 줄 | 패킷 |
|---|---|---|
| `es-data/src/training.rs` | 2,204 | K1 |
| `es-data/src/collect.rs` | 1,070 | K1 |
| `es-editor/src/ui/advanced.rs` | 1,709 | K2 |
| `es-editor/src/ui/author.rs` | 1,039 | K2 |
| `es-ir/src/task.rs` | 1,254 | K3 |
| `es-ir/src/learning.rs` | 1,059 | K3 |
| `es-script/src/spec/compile.rs` | 1,194 | K3 |
| `es-eval/src/runner.rs` | 1,483 | K4 |
| `es/src/cmd/train.rs` | 1,206 | K4 |
| `es/src/cmd/eval.rs` | 1,134 | K4 |
| `es-import/src/rl_import.rs` | 1,421 | K5 |
| `es-render/src/cpu.rs` | 1,317 | K5 |
| `es-editor-scene/src/sentence.rs` | 1,042 | K6 |
| `es-assets/src/scene.rs` | 1,036 | K6 |

## 웨이브

| 웨이브 | 패킷 |
|---|---|
| 1 | K1 (es-data) · K2 (es-editor) · K3 (es-ir, es-script의 컴파일러) |
| 2 | K4 (es-eval, es의 train·eval 동사) · K5 (es-import, es-render의 CPU 경로) · K6 (es-editor-scene의 문장, es-assets의 장면 모델) |
| 3 | K7 유지 노드(M17의 F-8) · K8 URDF 메시 배율 |

- **K7, 한동안 유지하기(M17 F-8).** 설계: `docs/design/scene-authoring.ko.md` 4.9절(새 IR-D 노드가 아니라
  `Terminate` 출력 노드의 `hold_ticks`). 아래 문단은 처음의 초안이다. "[상자]가 [1초] 동안 멈춰 있다"는 지금 "지금 멈춰 있다"로 컴파일된다
  (IR-D에 유지 노드가 없다). 그래서 구르다가 잠깐 느려진 상자도 센다.
  - 상태를 가진 노드(입력이 `n` 제어 틱 동안 이어서 참이면 참, 환경마다, 에피소드와 함께 초기화)는
    Task IR 노드 집합에 대한 스펙 변경이다(§6, ko 먼저).
  - 그것을 요청하는 명세만 쓰므로 커밋된 해시는 바뀌지 않는다.
  - `es-env`는 보상과 종료 원뿔에서 이 노드를 정수 카운터로 lowering한다.
  - `es-script`의 `still`은 `for_s`를 받으며, 이 값은 틱 수로 나누어떨어져야 한다.
  - G8의 문장에 "[1초] 동안" 칸이 생긴다.
  - R8의 사후 읽기는 유지 절을 끝 행에서 안쪽 술어로 설명하고, 그렇게 했다고 말한다.
  - 설계가 먼저다. `docs/design/scene-authoring.md`의 짧은 절(4.9)로, 패킷을 돌리기 전에
    오케스트레이터가 쓴다.
- **K8, URDF 메시 배율.** URDF의 `<mesh scale>`을 경고 대신 R3의 `SceneDesc::mesh_scales`로 옮긴다.

## 병합 결과 (2026-10-02)

여덟 패킷을 모두 병합했다. `cargo xtask ci`는 `ES_PYTHON`과 함께 `ba7c7fe`에서 통과했다. 골든과 커밋된 해시는 하나도 바뀌지 않았다.

**파일 (K1–K6).** 세어지는 줄이 1,000줄을 넘던 파일 14개를 책임별로 나눴다. 부모 파일은 제 경로에 남아
옮긴 것을 다시 내보내므로, 다른 크레이트도, 문서에 인용된 경로도 바뀌지 않았다.

| 패킷 | 파일 | 지금 가장 큰 모듈 |
|---|---|---|
| K1 `f1e32b4`, `c3d9d2a` | `es-data` `training.rs`(모듈 9개), `collect.rs`(5개) | `collect.rs` 514 |
| K2 `1db60d9`, `72c2c0e` | `es-editor` `ui/advanced.rs`(9개), `ui/author.rs`(새 모듈 5개) | `author/inspector.rs` 469 |
| K3 `0dc231c`, `3e6cf85`, `c1168a7` | `es-ir` `task.rs`, `learning.rs`; `es-script` `compile.rs` | `task/node.rs` 582 |
| K4 `361142a`, `4456cfa`, `3e229b8` | `es-eval` `runner.rs`; `es` `cmd/train.rs`, `cmd/eval.rs` | `runner.rs` 629 |
| K5 `c86dab7`, `2e74a21` | `es-render` `cpu.rs`; `es-import` `rl_import.rs` | `rl_import.rs` 480 |
| K6 `16a706d`, `3935fbb` | `es-assets` `scene.rs`; `es-editor-scene` `sentence.rs` | `scene/hash.rs` 480 |

**나눈 뒤 확인한 것.**
- **줄 비교:** 정확한 줄 범위를 옮겨 붙였고, 원본과의 멀티셋 줄 비교에는 import, 모듈 문서, 다시 내보내기,
  `pub(super)`만 남았다.
- **테스트 이름:** 전후가 같다.
- **소스 스캔:** 소스를 텍스트로 읽는 테스트는 모두 그대로 참이다. `es-eval` 테스트 둘이 `run_episode`
  안의 호출 수를 세므로, `run_episode`는 `runner.rs`에 남겼다.
- **렌더러:** CPU 경로가 모든 렌더 골든을 비트까지 재현한다.
- **에디터:** 나누기 전후에 찍은 스크린샷 여섯 쌍이 픽셀까지 같다.
- **해시:** 픽스처 15개를 시험한 결과 `scene_hash`, `Debug`, TOML이 바이트까지 같았다.

**크레이트 총합.** 새 import 줄만큼 조금 늘었다. `es-ir`이 목표 6,000줄을 넘었고(6,031), 이미 넘어
있던 네 크레이트는 그대로 WARN이다. 소유자가 파일 단위만 골랐으므로 크레이트는 나누지 않았다.

**기능 (K7, K8).**
- **K7 `72a7312`…`ba7c7fe`가 M17의 F-8을 닫는다.** `Terminate`가 선택 항목 `hold_ticks`를 받는다
  (스펙 §6.3, ko 먼저. 0은 `TASK-004`로 거절).
  - **유지하는 방식:** 정수 카운터를 환경마다 에피소드 예산을 세는 곳에 두므로, IR-D는 순수한 DAG로
    남는다.
  - **과제 명세:** `[success] hold_s`와 `[failure] hold_s`. ①은 "아래가 모두 [1초] 동안 맞으면
    성공"으로 읽는다.
  - **R8의 설명:** 유지가 짧았던 시도는 "X초 동안만 맞았다"로 설명한다.
  - **측정:** 떨어뜨린 상자는 유지 없이는 튀는 순간 성공하고, 1초 유지로는 멈춘 뒤에만 성공한다(12–14
    스텝 대신 61–66 스텝).
  - **해시:** 커밋된 Task IR 20개(`Terminate` 노드 46개)의 해시가 모두 그대로다.
- **K8 `45e4717`가 M17의 URDF 항목을 닫는다.** URDF의 `<mesh scale>`을 장면 문서와 같은 함수로
  `SceneDesc::mesh_scales`에 옮기므로, 이름과 꼭짓점이 MJCF와 같다. 배율이 없는 URDF의 해시는 그대로다.

**K7이 고른 기본값 (소유자가 바꿀 수 있다)**
- 성공 보너스는 유지하는 동안 매 틱 지급된다. 보너스가 순간 판정의 접은 결과를 읽기 때문이며, 한
  번만 지급하려면 새 노드가 필요하다.
- 유지가 없으면 "0초"로 보인다.
- 실패 묶음이 없는 명세에는 실패 유지 칸이 없다.
