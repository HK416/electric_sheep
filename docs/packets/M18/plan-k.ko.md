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

- **K7, 유지 노드(M17 F-8).** "[상자]가 [1초] 동안 멈춰 있다"는 지금 "지금 멈춰 있다"로 컴파일된다
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
