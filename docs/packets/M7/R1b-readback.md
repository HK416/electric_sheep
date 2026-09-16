# M7 R1b — the readback: a host-cached staging buffer in `es-gpu`

Spec: §15.1 ("this entire span is GPU-resident with no host round-trips" — the readback is
the one place the showcase and the frame sink must cross to the host, so it should cost what
PCIe costs, not what an uncached read costs), §12.4 (`camera_frames_per_sec`,
`pixels_per_sec`), §3.4 (nothing here touches arithmetic), §28.10 rule 1. Design note to
extend: `docs/design/gpu-foundation.md` (+ `.ko.md`) — the buffer section — and
`docs/design/renderer.md` section 8.4 gets its closing number. Depends on **R1** (whose
`frame_profile` is the oracle's instrument).

## the question

R1 measured the 1280×720 frame on both GPUs after the BVH: a ~1 ms render inside **64–85 ms
of readback**. `es_gpu::Buffer::download` on a device-local buffer stages through
`Usage::Staging`, which is `gpu_allocator::MemoryLocation::CpuToGpu` — host-visible,
write-combined, meant for *uploads* — and then reads it back through the mapping at
27–44 MiB/s (R1's `host-visible read of 3600 KiB` line). **Does a `GpuToCpu` (host-cached)
staging buffer for downloads bring the readback to the copy's own cost, with every output
bit unchanged?**

## spec

* `es_gpu::buffer::Usage` gains a `Readback` variant: `MemoryLocation::GpuToCpu`, usage
  `TRANSFER_DST` (plus whatever `Buffer::download` needs to map it). `Buffer::download` on a
  non-mapped buffer stages through `Usage::Readback` instead of `Usage::Staging`. `Staging`
  keeps its meaning for uploads. Nothing else in `es-gpu` changes; no arithmetic is involved,
  so no golden can move.
* If `GpuToCpu` memory is unavailable on a device (`gpu-allocator` returns an error), fall back
  to today's path and say so once through the existing capability/report channel rather than
  failing the frame — `Capabilities` already lists the memory heaps (§20).
* `frame_profile` (R1's test) keeps its "host-visible read of N KiB" line and adds the same
  measurement through `Buffer::download` on a `Storage` buffer, so the staging path itself is
  what is timed.

## context

`cargo xtask check-scope` reads the fence; the prose below is the same list with reasons.

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

`crates/es-gpu/src/buffer.rs` (the variant and the download path), `crates/es-gpu/src/lib.rs`
(re-export only if needed), `crates/es-gpu/tests/*.rs` (a round-trip test), `crates/es-render/tests/render.rs`
(the extra `frame_profile` line only), the two design notes, this packet.

## oracle

1. `cargo test -p es-gpu` — a new `download_round_trips_through_a_cached_staging_buffer`:
   upload 3,600 KiB of a counter pattern into a `Storage` buffer, `download` it, assert
   byte-equality; `SKIP` with reason without a device.
2. `cargo test -p es-render --release` — every existing test unchanged (`Rgb8`/`Seg` bitwise,
   `Depth`/`Normal` ≤ 1 ULP, ReSTIR tolerance), `cargo xtask verify-goldens` 0 changed. A
   readback path cannot change bits; this is the proof.
3. `cargo test -p es-render --release -- --ignored --nocapture frame_profile` on the RTX 3060
   and on the RTX 4090: the `readback` row and the frame total before/after, recorded in
   `docs/design/renderer.md` section 8.4 (target: readback within 2× of a 3.6 MB PCIe copy,
   i.e. a few ms — `Target / Status: unverified` until measured).
4. `es video showcase` on V19b `nominal-00` at 1280×720 on the server: ms/frame beside R1's
   118.6; the 96×96 `overhead` replay still 224/224 bit-identical to the recorded frames.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/R1b-readback.md`.

## acceptance

Oracles 1–5; the numbers in renderer.md 8.4 and gpu-foundation.md. If the fallback path is
taken on either GPU, that is reported, not hidden.

## forbidden

Any shader; any arithmetic; `RenderConfig`; goldens and fixtures; `crates/es-render/src/**`;
a Vulkan extension; `docs/ARCHITECTURE*.md`. INV-17: no new trait.
