# M7 R1b — 리드백: `es-gpu` 안의 호스트 캐시 스테이징 버퍼

스펙: §15.1("이 구간 전체는 호스트 왕복 없이 GPU에 상주한다" — 리드백은 쇼케이스와 프레임
싱크가 호스트로 건너가야만 하는 단 한 곳이므로, 캐시되지 않은 읽기가 아니라 PCIe가 치르는
만큼의 비용을 치러야 한다), §12.4(`camera_frames_per_sec`, `pixels_per_sec`), §3.4(여기는
산술을 전혀 건드리지 않는다), §28.10 규칙 1. 확장할 설계 노트: `docs/design/gpu-foundation.md`
(+ `.ko.md`) — 버퍼 절 — 그리고 `docs/design/renderer.md` 8.4절이 마무리 수치를 얻는다.
**R1**에 의존한다(그 `frame_profile`이 이 오라클의 계측 도구다).

## 질문

R1은 BVH 이후 두 GPU 모두에서 1280×720 프레임을 측정했다: **64–85 ms의 리드백** 안에 든
~1 ms짜리 렌더. 디바이스-로컬 버퍼에 대한 `es_gpu::Buffer::download`는 `Usage::Staging`을
거쳐 스테이징하는데, 이는 `gpu_allocator::MemoryLocation::CpuToGpu` — 호스트 가시적이고,
write-combined이며, *업로드*를 위한 것이다 — 이고, 그런 다음 그 매핑을 통해 27–44 MiB/s로
읽어 낸다(R1의 "host-visible read of 3600 KiB" 줄). **다운로드용 `GpuToCpu`(호스트 캐시)
스테이징 버퍼가 출력 비트 하나 바꾸지 않고 리드백을 복사 자체의 비용으로 끌어내리는가?**

## spec

* `es_gpu::buffer::Usage`가 `Readback` 변종을 얻는다: `MemoryLocation::GpuToCpu`, 사용처는
  `TRANSFER_DST`(그리고 `Buffer::download`가 그것을 매핑하는 데 필요한 것 무엇이든). 매핑되지
  않은 버퍼에 대한 `Buffer::download`는 `Usage::Staging` 대신 `Usage::Readback`을 거쳐
  스테이징한다. `Staging`은 업로드를 위한 의미를 그대로 유지한다. `es-gpu`의 다른 무엇도
  바뀌지 않는다; 산술은 전혀 관여하지 않으므로 어떤 골든도 움직일 수 없다.
* 어떤 디바이스에서 `GpuToCpu` 메모리를 쓸 수 없으면(`gpu-allocator`가 에러를 반환하면),
  프레임을 실패시키는 대신 오늘의 경로로 폴백하고 기존의 capability/report 채널을 통해 그것을
  한 번 알린다 — `Capabilities`가 이미 메모리 힙을 나열한다(§20).
* `frame_profile`(R1의 테스트)은 "host-visible read of N KiB" 줄을 유지하고, `Storage` 버퍼에
  대한 `Buffer::download`를 통한 같은 측정을 더해서, 스테이징 경로 자체가 재는 대상이 되게
  한다.

## context

`cargo xtask check-scope`가 펜스를 읽는다; 아래 산문은 같은 목록에 이유를 단 것이다.

```
crates/es-gpu/src/buffer.rs
crates/es-gpu/src/lib.rs
crates/es-gpu/tests/*.rs
crates/es-render/tests/render.rs
docs/design/gpu-foundation.md
docs/design/gpu-foundation.ko.md
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M7/R1b-readback.md
docs/packets/M7/R1b-readback.ko.md
```

`crates/es-gpu/src/buffer.rs`(변종과 다운로드 경로), `crates/es-gpu/src/lib.rs`(필요하면
재수출만), `crates/es-gpu/tests/*.rs`(왕복 테스트 하나), `crates/es-render/tests/render.rs`
(추가되는 `frame_profile` 줄만), 두 설계 노트, 이 패킷.

## oracle

1. `cargo test -p es-gpu` — 새로운 `download_round_trips_through_a_cached_staging_buffer`:
   카운터 패턴 3,600 KiB를 `Storage` 버퍼에 업로드하고, `download`한 뒤, 바이트 단위 동등성을
   단언한다; 디바이스가 없으면 이유와 함께 `SKIP`한다.
2. `cargo test -p es-render --release` — 기존의 모든 테스트가 불변(`Rgb8`/`Seg`는 비트 단위,
   `Depth`/`Normal`은 ≤ 1 ULP, ReSTIR 허용 오차), `cargo xtask verify-goldens`가 변경 0건.
   리드백 경로는 비트를 바꿀 수 없다; 이것이 그 증명이다.
3. `cargo test -p es-render --release -- --ignored --nocapture frame_profile`을 RTX 3060과
   RTX 4090에서: `readback` 행과 프레임 합계를 전후로 `docs/design/renderer.md` 8.4절에
   기록한다(목표: 리드백이 3.6 MB PCIe 복사의 2배 이내, 즉 수 ms — 측정될 때까지
   `Target / Status: unverified`).
4. 서버에서 1280×720의 V19b `nominal-00`에 대한 `es video showcase`: R1의 118.6 옆에
   ms/frame; 96×96 `overhead` 리플레이는 기록된 프레임들과 여전히 224/224 비트 동일하다.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/R1b-readback.md`.

## acceptance

오라클 1–5; renderer.md 8.4절과 gpu-foundation.md 안의 수치. 폴백 경로가 어느 GPU에서든
밟히면, 그것은 숨기지 않고 보고된다.

## forbidden

어떤 셰이더든; 어떤 산술이든; `RenderConfig`; 골든과 픽스처; `crates/es-render/src/**`;
Vulkan 확장; `docs/ARCHITECTURE*.md`. INV-17: 새 트레이트 없음.
