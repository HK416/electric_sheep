# M11 X3b — 하나의 디스패치에 담긴 N개의 env

스펙: §28.14 파동 2, §15.2(타일 아틀라스), §3.4, §12.4(9개 지표, 단일 `step/s`는 절대 아님). 설계
노트: `renderer.md` §1(타일 아틀라스 + 채널 계약)과 12.4(비용). X3에 의존. 유형 B.

## 질문

아틀라스는 **하나의 장면 위에 여러 카메라**를 배치한다(`renderer.md` §1); X3 이후 `Rollout`은 각 env를
자신의 `EnvRenderer`로, env마다 디스패치 하나씩 렌더링하며, RTX 4090에서 96×96 64-spp `Pt` 프레임 한
장에 57.5 ms — reach PPO 한 런에 약 65시간. **하나의 디스패치가 N개의 env — 같은 지오메트리, env별
바디 변환과 env별 렌더 오버라이드 — 를 N개의 타일로, N번의 단일 렌더와 비트 단위로 동일하게, PT
학습이 감당할 수 있는 env당 비용으로 렌더링할 수 있는가?**

## spec

* 장면은 한 번 업로드된다(삼각형, 로컬 프레임의 강체별 BVH — 오늘의 테셀레이션 캐시가 이미 로컬
  프레임 삼각형을 유지하듯); env별 변환과 env별 오버라이드(빛, 색상, 카메라 — X5의 몫)는 작은
  타일별 버퍼다. BVH 전략(env별 인스턴스 변환을 가진 2레벨 BVH, 또는 공유 BLAS 위의 env별 TLAS
  재구축)은 에이전트가 골라 숫자로 정당화할 몫이다; CPU 레퍼런스는 같은 순회와 같은 샘플 키로 같은
  N개 타일을 렌더링한다.
* 타일마다 샘플 시드는 그 env의 `frame_seed(stream, base, tick)`다 — 타일은 그 env를 단일 렌더링한
  것과 정확히 같다.
* 렌더러가 지원하고 env 수 > 1이면 `Rollout`은 배치 경로를 쓴다; 단일-env 경로는 남아 오라클로
  남는다.

## context

```
crates/es-render/src/**
crates/es-render/shaders/**
crates/es-render/tests/**
crates/es-env/src/render.rs
crates/es-env/tests/render_loop.rs
crates/es-py/src/rollout.rs
tests/golden/render/batch_*
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M11/X3b-batched-render.md
docs/packets/M11/X3b-batched-render.ko.md
```

## 오라클

1. `cargo test -p es-render --test render batched_tiles_equal_single_renders`(GPU): 서로 다른 포즈의
   SO-101 장면 env N ∈ {1, 4, 16}, `Rs`와 `Pt`(1과 16 spp, SVGF on/off): 모든 타일 == 그 env의 단일
   렌더 비트 단위; CPU 레퍼런스 == GPU, 기존 패리티 규칙대로(그 규칙이 0이라 하는 곳은 0 ULP).
2. 기존 골든은 모두 byte-identical; 새 `batch_*` 골든은 CPU 레퍼런스에서만.
3. `pt_batched_cost`(`--ignored`, 두 GPU 모두): `Pt` 4/16/64 spp ± SVGF에서 N = 1, 4, 16, 64의
   env당 ms/프레임, 전체 프레임(업로드, 디스패치, 리드백), X3의 단일-env 수치 옆에.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## 수용

오라클 1~4; `renderer.md`에 비용 표와 BVH 선택 이유를 담은 13절(+ko)이 생긴다.

## 금지

기존 프레임의 바이트 변경; 비결정적 리덕션; 벤더 디노이저; 새 trait.
