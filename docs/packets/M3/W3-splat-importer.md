# W3-splat-importer — `es-splat` offline path: import, align, bind

Spec: §16 (real-to-sim via 3DGS; §16.2 colour and position alignment, LBS binding; §16.3
artifacts), §28.5 W3 row, §15.1 (what a splat asset eventually feeds), §3.1 (Z-up, metres,
xyzw quaternions), §5.3 (`asset_hash` over decoded content), §1.9 item 3 (this path is cut
third — keep it lean).
Design note: `docs/design/splat-real2sim.md`.
API note: `docs/api-notes/gaussian-splat-ply.md` (every field `unverified`).

The offline half of §16 only. The splat rasteriser of §16.3 needs Vulkan and `es-gpu`; it is
not in this packet and nothing here renders.

## context

```
crates/es-splat/src/lib.rs
crates/es-splat/src/ply.rs
crates/es-splat/src/align.rs
crates/es-splat/src/bind.rs
crates/es-splat/tests/splat.rs
tests/fixtures/splat/cube20_ascii.ply
tests/fixtures/splat/cube20_binary.ply
docs/design/splat-real2sim.md
docs/api-notes/gaussian-splat-ply.md
docs/packets/M3/W3-splat-importer.md
```

## spec

1. **`SplatScene`** — structure of arrays, `f32`: `positions` (3N, m, §3.1 Z-up), `scales`
   (3N, m, `exp` applied, Gaussian-local frame), `rotations` (4N, xyzw, unit norm, `w >= 0`),
   `opacities` (N, `sigmoid` applied), `sh_dc` (3N), `sh_rest` (`3 * ((d+1)^2 - 1) * N`, may
   be empty), plus `sh_degree`, `bounds`, `asset: AssetRef`, `warnings`.

2. **`import_ply(bytes) -> Result<SplatScene, SplatError>`** — `binary_little_endian 1.0` and
   `ascii 1.0`. Properties are located **by name**, offsets computed from the declared type
   order, so a reordered or extended exporter still reads. An unrecognised property is a
   warning, not an error. `sh_degree` is inferred from the `f_rest_*` count and must be one of
   0/9/24/45. Axis conversion from the COLMAP/OpenCV Y-down world to §3.1 Z-up is the fixed
   swizzle `(x, z, -y)` applied to positions and to quaternion vector parts —
   **`unverified`**, and deliberately a swizzle rather than a quaternion product so it is
   bit-exact and exactly invertible. `f_rest` is *not* band-rotated; that produces a warning.
   Every malformed input is a typed `SplatError` naming the element or property at fault —
   never a panic, never a silent default.

3. **`SplatScene::write_ply(&self) -> Vec<u8>`** — `binary_little_endian` in the reference
   property order, inverting both activations. One encoding: the reader accepts two, the
   writer picks the interchange one.

4. **`Similarity { scale, rot, trans }`** — `fit(src, dst)` by Umeyama with Horn's quaternion
   method, the 4x4 symmetric eigenproblem solved by a fixed-sweep cyclic Jacobi (`sqrt` and
   division only; no transcendentals, so no `es_math::approx` dependency and no 3x3 SVD).
   `>= 3` correspondences; degenerate input is an error. `apply(p)`, `apply_scene(&mut)`.

5. **`ColorAffine { gain, bias }`** — per-channel ordinary least squares between splat DC
   colours and reference samples; zero-variance channel degrades to a pure offset.
   `SplatScene::apply_color` maps `sh_dc` through RGB space (`0.5 + C0 * dc`) and back.

6. **`Binding::bind(&SplatScene, &SceneDesc, k)`** — `k` nearest bodies by distance to their
   nearest geom *surface* (exact for `Sphere`/`Box`/`Capsule`, geom-origin fallback
   otherwise), inverse-distance weights normalised to 1, `k` clamped to `1..=4`. A Gaussian
   within `WELD_EPS` of a surface is welded: weight `1.0`, one body.
   `skin(&Binding, &SplatScene, &BTreeMap<StableId, Pose>) -> SkinnedSplats` applies
   `delta_b = pose_now_b * rest_b^-1` as linear blend skinning; a body with no pose supplied
   contributes its rest pose.

7. **`asset_hash`** — `blake3` over a domain tag, count, `sh_degree` and every attribute array
   in **file order**, each length-prefixed; over decoded values, so encoding cannot change it
   (§5.3).

## oracle

```
cargo fmt -p es-splat --check
cargo clippy -p es-splat --all-targets -- -D warnings
cargo test -p es-splat
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

## acceptance

- `tests/fixtures/splat/cube20_binary.ply` (20 Gaussians, SH degree 3) imports and
  `write_ply` reproduces the file **byte for byte**. The fixture's raw scale/opacity values
  are drawn from the fixed points of the `exp`/`ln` and `sigmoid`/`logit` pairs on purpose —
  see `docs/design/splat-real2sim.md` §1.1 — and a separate test pins the non-exact case to a
  relative `1e-6`.
- The ascii fixture decodes to the same arrays, bit for bit, as the binary one, and to the
  same `asset_hash`.
- `Similarity::fit` recovers a known `(s, R, t)` to `1e-6` on 10 pseudorandom points, over
  fixed splitmix64 seeds.
- `ColorAffine::fit` recovers a known per-channel gain and bias.
- Gaussians placed on the surface of one of two bodies get weight `1.0` on that body, and
  translating that body translates them by exactly the same vector.
- A truncated header, an unknown `format`, a missing `x`, a bad `f_rest` count and a short
  binary payload each produce their own `SplatError` variant; nothing panics.

## forbidden

- Anything outside `crates/es-splat/**`, `tests/fixtures/splat/**`, and the three docs above.
  `es-assets`, `es-math` and `es-core` are consumed, never edited — including
  `AssetKind`, which gains no `Splat` variant here.
- Rendering, Vulkan, `es-gpu`: the §16.3 rasteriser is a separate packet.
- New dependencies. The PLY reader and writer are written here; a 3DGS PLY is a text header
  and packed `f32`s.
- `.splat` / `.ksplat` / `.spz` decoding — documented as unsupported in the API note.
- ICP and RANSAC (§16.2). `fit` takes correspondences someone else chose.
- `HashMap`/`HashSet` — `BTreeMap` only.
- New extension-point traits (INV-17); this packet adds no trait at all.
- Committing. The oracle is run and reported, not landed.
