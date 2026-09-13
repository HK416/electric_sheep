# Splat real-to-sim: the offline half

`es-splat` (layer 5) is the §16 real-to-sim path. This note covers what M3 W3 implements
*offline*: import a 3D Gaussian Splatting capture, align it to the physics scene in position
and colour, and bind it to articulated bodies so the visual follows the physics. The
rasteriser — the other half of §16.3 — needs Vulkan and is deferred; nothing here renders.

The framing that keeps this crate small is §16.3's own: **exact alignment is unnecessary,
distribution match is the goal.** Every number below is a fit, not a measurement, and §16.1's
result is that getting colour and position roughly right is what moves policy transfer. So:
closed-form fits, no iterative refinement, no ICP yet.

Per §1.9 this whole path is cut third under scope pressure. That is a design constraint, not
a footnote: nothing outside this crate may depend on it.

## 1. `SplatScene`

Structure-of-arrays, `f32`, one `Vec` per attribute, all of length `count` (or a multiple):

```
positions   3 * count   m, §3.1 Z-up world
scales      3 * count   m, per-axis standard deviation in the Gaussian's own frame (exp applied)
rotations   4 * count   xyzw, unit norm, w >= 0 (§3.1)
opacities   1 * count   alpha in [0, 1] (sigmoid applied)
sh_dc       3 * count   degree-0 SH coefficient, RGB
sh_rest     3 * ((d+1)^2 - 1) * count, or empty
```

plus `sh_degree`, `bounds` (axis-aligned min/max over `positions`), `asset: AssetRef`, and
`warnings`. SoA rather than an array of structs because the consumers are a GPU upload and
three passes that each touch one attribute; and because a per-attribute `Vec<f32>` is what
`blake3` hashes without a serialisation step.

The file layout it decodes from is `docs/api-notes/gaussian-splat-ply.md` (everything there is
`unverified`).

### 1.1 Activations are applied on import, and that is lossy

The file stores `log(sigma)` and `logit(alpha)`; `SplatScene` stores `sigma` and `alpha`.
`write_ply` inverts both. `exp` and `ln` are not exact inverses in `f32`: for random inputs
about 10% of scales and about 50% of opacities land one ULP away after a round trip, so
**file -> scene -> file is byte-exact only for values that are fixed points of the activation
pair.** The checked-in fixtures are generated from that set on purpose, so the round-trip test
measures what it is meant to measure — header, property order, byte layout, Gaussian order —
rather than `libm` rounding. A separate test pins the non-exact case to a relative `1e-6`.

The alternative (store the raw file values, activate on access) would make the round trip
exact for any input but push `exp`/`sigmoid` into every consumer, including the GPU upload
path. Decoded-on-import matches what the glTF importer does with vertex data and what §5.3
means by hashing decoded content, so it wins.

**Determinism (P-M3 review, §3.2/§3.4).** `exp`, `ln`, `sigmoid` and `logit` all feed
`SplatScene::asset_hash`, a §5.3 chain input, so they must not depend on the host's `libm`:
two machines importing the same capture must hash it identically. `sigmoid`/`logit`/`scale`
decode and encode therefore route through `es_math::approx::exp` / `es_math::approx::ln` — the
one deterministic transcendental implementation shared with the GPU (`crates/es-math/slang/
approx.slang`) — entirely in `f32`, rather than through `f64::exp`/`f64::ln` rounded once to
`f32`. `es_math::approx` bounds its error by ULPs, not by IEEE correct rounding, so this is a
strictly different (and slightly less accurate) approximation than the host `libm` gave; the
`1e-6` relative tolerance in `a_non_fixed_point_activation_round_trips_to_a_relative_1e_6`
already covers the difference, and the fixed-point set the byte-exact fixtures draw from
changed along with it — `0.0` is a fixed point of both `exp`/`ln` and `sigmoid`/`logit` under
`es_math::approx` (`exp(0) == 1`, `ln(1) == 0`, `sigmoid(0) == 0.5`, `logit(0.5) == 0`, all
bit-exact), so the two `cube20_*.ply` fixtures were regenerated with `scale_0..2` and
`opacity` set to `0.0` for every vertex; every other field (positions, normals, SH
coefficients, rotations) is unchanged.

### 1.2 `AssetKind` has no `Splat` variant

`AssetKind` lives in `es-assets`, which this packet does not own, so `AssetRef::kind` is
`Mesh` and `path` carries the `.ply`. The packet that adds `AssetKind::Splat` should change
it; nothing branches on the variant today.

## 2. Axis conversion

A 3DGS reconstruction inherits COLMAP/OpenCV camera conventions: the world frame is the first
camera's, so **+Y is down and +Z is forward**. §3.1 is right-handed Z-up. The fixed rotation
between them is `R_x(-90 deg)`:

```
x_es = +x_gs        y_es = +z_gs        z_es = -y_gs
```

This is `unverified` — it holds for a COLMAP-derived capture and for the reference trainer's
output, and is wrong for a capture whose SfM stage already applied an up-vector fix. It is a
*convention guess*, which is exactly why §16.2 puts a fitted `T_robot_scan` after it: any
residual rotation is absorbed by the similarity of §3 below, so a wrong guess costs accuracy
in the initial guess only, never correctness of the final alignment.

It is implemented as a **component swizzle with sign flips**, not as a quaternion product.
Same rotation, and it is bit-exact and exactly invertible, which is what makes the PLY round
trip meaningful. Quaternions get the same swizzle on their vector part (conjugation by a
rotation acts on the vector part by that rotation; the scalar part is invariant). Scales are
*not* converted: they are extents in the Gaussian's own frame, not world directions.

## 3. Position alignment: `Similarity`

```
Similarity { scale: f64, rot: Quat, trans: Vec3 }     apply(p) = scale * (rot * p) + trans
Similarity::fit(src: &[Vec3], dst: &[Vec3]) -> Result<Similarity, SplatError>
```

Umeyama's closed form with Horn's quaternion method for the rotation, needing >= 3
correspondences (three non-collinear points fix a rotation; two do not):

1. Centroids, then centred point sets. Sums use a fixed index order.
2. `S = sum_i src_i_centred * dst_i_centred^T` (3x3).
3. Build Horn's symmetric 4x4 `N` from `S` (the `(w, x, y, z)`-ordered form). The eigenvector
   of its largest eigenvalue is the rotation `src -> dst`.
4. `scale = sum_i dst_i_centred . (rot * src_i_centred) / sum_i |src_i_centred|^2`.
5. `trans = dst_mean - scale * (rot * src_mean)`.

**Why the eigen route and not an SVD.** A 3x3 SVD is either a dependency (`nalgebra`, and
layer 5 does not get one for this) or ~200 lines of Jacobi-on-`A^T A` plus sign-correction to
avoid a reflection. Horn's `N` is 4x4 symmetric, so a **cyclic Jacobi eigenvalue sweep** — six
rotations per sweep, fixed `(p, q)` order, a fixed sweep count — solves it in about 60 lines
using nothing but `sqrt` and division. No transcendentals at all, so the §3.2 /
`es_math::approx` rule is satisfied by not needing it, and the fixed operation order makes the
result reproducible without a determinism contract of its own. The quaternion parameterisation
also cannot produce a reflection, which is the failure mode an SVD needs its `det` correction
for.

Degenerate inputs are errors, not silent identity: fewer than 3 points, a mismatched pair
count, a `src` set whose centred points have zero spread (all identical, so no scale is
defined), or a non-finite coordinate.

ICP and RANSAC (§16.2) are not here. `fit` takes correspondences someone else picked — picked
markers, a registration UI, or a detector — and that is the whole M3 W3 scope.

## 4. Colour alignment: `ColorAffine`

§16.1 singles this out: mapping the scan's colour space onto the robot's actual camera is
what put the policy back in-distribution. The spec calls for a polynomial map; the first
useful term of one is per-channel affine, so that is what this is:

```
ColorAffine { gain: [f64; 3], bias: [f64; 3] }        apply(rgb_c) = gain_c * rgb_c + bias_c
ColorAffine::fit(src: &[[f32; 3]], dst: &[[f32; 3]]) -> Result<ColorAffine, SplatError>
```

Ordinary least squares per channel, independently: `gain = cov(src, dst) / var(src)`,
`bias = mean(dst) - gain * mean(src)`. A channel with zero variance in `src` is
`gain = 1, bias = mean(dst) - mean(src)` — a pure offset — rather than a division by zero.
No cross-channel term, no gamma: adding either before there is a real capture to fit against
would be fitting to nothing. The design leaves room for it by keeping `ColorAffine` a value
stored on the scene rather than a baked-in transform.

`SplatScene::apply_color` converts `sh_dc` to RGB (`0.5 + C0 * dc`), applies the map, and
converts back. Degrees 1..3 are left alone: the affine map is not linear (the bias term), so
it has no consistent action on the higher bands, and correcting view-dependent colour needs
the renderer that does not exist yet.

Colour alignment must eventually enter `scene_hash` (§16.2). It does not yet, because nothing
in this packet builds a `SceneDesc` to hash it into.

## 5. LBS binding

§16.2's first design decision: splats do not replace physics. Each Gaussian rides one or more
rigid bodies, and the physics of §17 says where those bodies are.

```
Binding {
    bodies:  Vec<StableId>,   // index space for the weights
    rest:    Vec<Pose>,       // each body's world pose at bind time
    indices: Vec<[u16; 4]>,   // per Gaussian, k <= 4 body indices
    weights: Vec<[f32; 4]>,   // per Gaussian, sums to 1
}
Binding::bind(&SplatScene, &SceneDesc, k: usize) -> Binding
skin(&Binding, &SplatScene, &BTreeMap<StableId, Pose>) -> SkinnedSplats
```

`bind` walks the body tree to world poses, then for each Gaussian measures the distance to
each body's **nearest geom surface** and keeps the `k` smallest (`k` clamped to `1..=4`).
Weights are inverse distance, normalised. A Gaussian sitting on a surface — distance below
`WELD_EPS` — is welded: weight `1.0` to that body and nothing else, which is both the
degenerate-case guard for `1/d` and the common case, since a splat reconstructed from photos
of a link *is* on that link.

Surface distance is exact for `Sphere`, `Box` and `Capsule` and falls back to distance from
the geom origin for everything else (`Mesh`, `HeightField`, `Cylinder`, `Ellipsoid`, `Plane`).
A convex-decomposition proxy — §16.2's actual answer for object geometry — would give the
right distance for a mesh, and belongs to the packet that builds one. Until then the fallback
is documented rather than approximated with something that looks exact and is not.

`skin` computes `delta_b = pose_now_b * rest_b^-1` per body, then per Gaussian blends
`sum_k w_k * (delta_k . p)` for the position and a weighted quaternion sum, sign-aligned to
the highest-weight body and renormalised, for the rotation. That quaternion blend is the
standard LBS approximation: it is not a proper rotation average and it shrinks under large
relative rotations, which for a splat means a slightly wrong ellipsoid orientation between two
bodies rotating apart. Dual quaternion skinning is the fix if it ever shows up in a
`domain_gap` number (§10); it is not worth the code before then. A missing body pose leaves
that Gaussian at its rest position rather than collapsing it to the origin.

This is the CPU path. The GPU path is the same arithmetic in a compute shader and belongs with
the rasteriser.

## 6. Hashing

`asset_hash` is `blake3` over a domain tag, the Gaussian count, `sh_degree`, and then every
attribute array in file order, each length-prefixed. Over **decoded** values, so the same
capture read from an ascii PLY and from a `binary_little_endian` PLY hashes identically —
tested — and a future `.spz` decoder inherits the property for free. Over **file order**,
because a 3DGS PLY has no canonical Gaussian ordering to sort by and inventing one would be a
sort nobody asked for; the file's order is the capture's identity.

Not hashed: `warnings`, `bounds` (derived from `positions`), and the `AssetRef` itself (whose
`hash` field *is* this digest).
