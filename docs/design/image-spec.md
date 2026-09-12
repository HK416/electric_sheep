# `ImageSpec` and its transform rules (`es-ir::image`) — design

Spec refs: spec 7.2 (`ImageSpec`, `OBS-021`, `OBS-034`), spec 3.1 (coordinate conventions),
Appendix B.2, `INV-14`. Packet: `docs/packets/M0/P19.md`.

## Conventions (spec 3.1, not negotiable)

- **Image coordinates**: origin top-left, `+x` right, `+y` down (OpenCV).
- **Camera coordinates**: `+Z` forward, `+X` right, `+Y` down (OpenCV optical frame, identical
  to the ROS `REP-103` optical frame).
- **Default color space**: sRGB (non-linear). Linearization is an explicit node, never implied.

`extrinsics` is `T_body_camera` as an `es_math::conventions::Pose` (position in m + canonical
unit quaternion, xyzw with `w >= 0`). Reusing `Pose` keeps one rotation convention in the repo.

## Why the transforms live in the type

Resizing or cropping an image without updating `fx, fy, cx, cy` does not fail — it silently
returns wrong 3D. So the three geometry-changing operations are *methods on the spec*, and the
Observation IR nodes `Resize` / `Crop` / `Undistort` are required to go through them
(`INV-14`). The compiler propagates the resulting `ImageSpec` down the graph and, at a
`CameraProjection` node, calls `intrinsics_consistent_with` — that comparison is the source of
`OBS-034`.

## Transform rules (Appendix B.2)

| method | width/height | intrinsics | distortion |
|---|---|---|---|
| `resized(w, h, rescale)` | `w, h` | `scaled(w/width, h/height)` if `rescale`, else unchanged | unchanged |
| `cropped(rect, rescale)` | `rect.width, rect.height` | `cx -= rect.x`, `cy -= rect.y` if `rescale`, else unchanged | unchanged |
| `undistorted(new_intr)` | unchanged | `new_intr` | `None` |

`Intrinsics::scaled(sx, sy)` scales `fx, skew` by `sx`, `fy` by `sy`, `cx` by `sx`, `cy` by `sy`.

This is the plain scaling of Appendix B.2, i.e. the pixel *corner* convention. Under the pixel
*centre* convention the exact form is `cx' = (cx + 0.5)·sx − 0.5`; the difference is under half
a pixel and the spec pins the plain form, so that is what is implemented. A resampler that
disagrees must say so in its own node, not by rewriting these rules.

`rescale = false` is the **explicit user opt-out** and nothing else. It exists because a caller
may already have measured intrinsics for the resized stream. It is not a default, and the node
that passes `false` is the node that must justify it; downstream, `intrinsics_consistent_with`
still reports the mismatch as `OBS-034` (the caller decides whether to demote it).

## `intrinsics_consistent_with(&other)`

True when `self`'s intrinsics are what `other`'s become after the pure resize between the two
resolutions:

```
sx = self.width / other.width,  sy = self.height / other.height
self.intrinsics ≈ other.intrinsics.scaled(sx, sy)      (relative tolerance 1e-9)
```

Equal resolutions therefore require equal intrinsics, and the classic bug — resize 640×480 →
224×224 while keeping the original `fx` — is false. The tolerance is relative because the
scale factors are exact ratios of integers but `scaled` multiplies in `f64`.

## Fields not in the packet brief

The struct follows spec 7.2 in full: `channels` (`ChannelFormat`), `dtype` (`ImageDType`),
`rate_hz` and `depth_scale` are carried as well. `color_space` is
`SRgb | Linear | Rec709 | Raw` (spec 7.2) — `Gray` and `Depth` are channel formats, not color
spaces. `OBS-021` (color-space mismatch at a vision encoder) is the compiler's check; this
module only carries the field and its code entry.
