# 3D Gaussian Splatting PLY layout

Pinned version: **none**. Nothing in this note was checked against a running reference
implementation — there is no 3DGS trainer, no `.ply` capture and no Python dependency in this
workspace. Per the §1.7 rule that an API note records only what was verified, **every field
below is marked `unverified`** and must be re-checked against the INRIA reference
(`graphdeco-inria/gaussian-splatting`, `scene/gaussian_model.py`, `save_ply` /
`construct_list_of_attributes`) before anyone trusts a real capture.

This is a *format* note, not a library note: `es-splat` reads and writes these bytes itself
(a PLY header plus packed `f32`s is not worth a dependency), so what matters here is the
property names, their order, and what transform each one needs.

## The container

Standard Stanford PLY. Only two of the three encodings occur in the wild:

```
ply
format binary_little_endian 1.0        # or: format ascii 1.0
element vertex <N>
property float x
...
end_header
<N * (sizeof of the declared properties) bytes, or N whitespace-separated lines>
```

- `format binary_big_endian 1.0` — `unverified`, never observed; `es-splat` rejects it rather
  than guessing.
- Line endings are `\n` in the reference writer; readers in the wild also accept `\r\n`.
- There is exactly one `element vertex`; other elements (e.g. `face`) do not appear. A 3DGS
  PLY has no connectivity.
- Every property is `float` (`f32`) in the reference writer. `double`/`uchar` columns appear
  in some exporters — `unverified`, and not supported.

## Vertex properties (the de-facto order)

| Property | Count | Meaning | Transform on import | Status |
|---|---|---|---|---|
| `x` `y` `z` | 3 | Gaussian centre, world units of the reconstruction | axis fix (see below); **no** unit scale — a 3DGS world is up to an unknown similarity, which is what §16.2's position alignment resolves | `unverified` |
| `nx` `ny` `nz` | 3 | Written as zeros by the reference writer; a 3D Gaussian has no normal | **ignored** (read and discarded) | `unverified` |
| `f_dc_0..2` | 3 | SH degree-0 coefficient per RGB channel | `rgb = 0.5 + C0 * f_dc`, `C0 = 0.28209479177387814` | `unverified` |
| `f_rest_0..44` | 45 | SH degrees 1..3, **channel-major**: 15 coefficients for R, then 15 for G, then 15 for B | none (kept as stored; see the caveat below) | `unverified` |
| `opacity` | 1 | **logit** of alpha | `alpha = sigmoid(opacity)` | `unverified` |
| `scale_0..2` | 3 | **natural log** of the per-axis standard deviation, in the Gaussian's *own* frame | `sigma = exp(scale_i)`; no axis fix — these are local-frame extents, not world directions | `unverified` |
| — | — | — | `exp`/`ln`/`sigmoid`/`logit` above run through `es_math::approx` (`f32`, ULP-bounded, not IEEE-correctly-rounded) rather than the host `libm`, because they feed `SplatScene::asset_hash` (§5.3) and must give the same bits on every machine (§3.2/§3.4). See `docs/design/splat-real2sim.md` §1.1 for the accuracy consequence. | — |
| `rot_0..3` | 4 | Quaternion **wxyz**, stored **unnormalised** (the trainer normalises at render time) | normalise, reorder to §3.1 xyzw with `w >= 0`, then apply the axis fix | `unverified` |

`f_rest` length depends on the trained SH degree: `3 * ((d+1)^2 - 1)` = 0 / 9 / 24 / 45 for
d = 0 / 1 / 2 / 3. Degree 3 (45) is the reference default. `es-splat` infers `sh_degree` from
the `f_rest_*` count and rejects a count that is not one of those four.

**`f_rest` channel-major is the single most error-prone item here.** The reference writer
transposes a `(N, 15, 3)` tensor to `(N, 3, 15)` before flattening, so `f_rest_0..14` are all
R. Several third-party readers assume coefficient-major and are silently wrong in
view-dependent colour only, which is why this stays `unverified` until a real capture is
rendered side by side.

**SH under a world rotation.** Degree-0 (`f_dc`) is rotation-invariant. Degrees 1..3 are not:
rotating the world requires the Wigner-D rotation of each band. `es-splat` does **not** rotate
`f_rest` — it carries the coefficients through unchanged and records a warning. Nothing
consumes them yet (the splat rasteriser is §16.3 work and needs Vulkan), and the alignment
similarity of §16.2 is re-fitted per capture anyway, so the correct band rotation belongs to
the packet that first renders view-dependent colour.

## Property order is not guaranteed

Exporters other than the reference one reorder columns, and some add their own (`confidence`,
`segment_id`, per-splat ids). `es-splat` therefore indexes properties **by name**, not by
offset, computes each property's byte offset from the declared type order, and reports an
unrecognised property as a warning rather than an error. Its own writer emits the reference
order above so a round trip is byte-stable.

## Compact variants — not supported yet

| Format | What it is | Why not |
|---|---|---|
| `.splat` | antimatter15's viewer format: 32 bytes per Gaussian — `f32[3]` position, `f32[3]` scale (linear, not log), `u8[4]` RGBA, `u8[4]` quaternion quantised as `(q * 128 + 128)`. Header-less; the count is `len / 32`. | Degree-0 colour only — the view-dependent SH is thrown away, so it cannot feed the §15.1 render path at the fidelity the whole point of §16 depends on. `unverified`. |
| `.ksplat` | mkkellogg's GaussianSplats3D format: versioned header, splats bucketed spatially, 16-bit half floats, several compression levels. | Version-dependent layout, and the spatial bucketing reorders Gaussians — which would make the file-order `asset_hash` of §5.3 depend on the exporter's bucketing. `unverified`. |
| `.spz` | Niantic's compressed format, named in §16.3 as a target. | Needs gzip and a fixed-point layout; not worth a dependency before anything renders. `unverified`. |

When one of these lands, it must decode to the same `SplatScene` field-for-field, because
`asset_hash` is taken over the decoded Gaussians (§5.3) and not over the file bytes — that is
the property that makes "same capture, different container" hash the same.
