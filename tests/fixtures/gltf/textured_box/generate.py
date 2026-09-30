"""Writes the plan H / HT2 glTF-vs-MJCF fixture: one textured box, declared twice.

    python tests/fixtures/gltf/textured_box/generate.py

Standard library only; deterministic (no clock, no randomness), so re-running it rewrites the
same bytes. Outputs, next to this script:

* ``base.png`` (sRGB), ``mr.png`` (linear; R occlusion, G roughness, B metal),
  ``normal.png`` (linear, tangent space, +Y up the image) and ``emissive.png`` (sRGB), 16 x 16;
* ``textured_box.gltf``: the box as a glTF 2.0 mesh (buffer as a data URI) wearing a
  ``pbrMetallicRoughness`` material with all four maps, ``emissiveFactor`` 1 and
  ``KHR_materials_emissive_strength`` 2;
* ``textured_box.obj`` + ``textured_box.xml``: the same triangles as an OBJ (positions already
  rotated from glTF's Y-up to spec 3.1's Z-up, ``vt`` = ``(u, 1 - v)`` because the OBJ reader
  flips ``v`` as MuJoCo's does) and the same material as MJCF ``<layer>``s with ``emission="2"``.

Every coordinate is dyadic, so both routes produce bit-identical triangles.
"""

import base64
import json
import math
import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SIDE = 16
# Half extents along glTF's x, y (up), z.
HX, HY, HZ = 0.25, 0.375, 0.5


def png(path, pixels, srgb):
    """An 8-bit RGB PNG; `pixels[row][col]` is an (r, g, b) byte triple, row 0 the top."""
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)
    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", SIDE, SIDE, 8, 2, 0, 0, 0))
    if srgb:
        out += chunk(b"sRGB", b"\x00")
    out += chunk(b"IDAT", zlib.compress(raw, 9))
    out += chunk(b"IEND", b"")
    with open(os.path.join(HERE, path), "wb") as f:
        f.write(out)


def grid(fn):
    return [[fn(r, c) for c in range(SIDE)] for r in range(SIDE)]


def base(r, c):
    if (r // 4 + c // 4) % 2 == 0:
        return (230, 60 + 10 * c, 40)
    return (30 + 12 * r, 90, 220)


def mr(r, c):
    rough = 40 + 13 * c
    metal = 255 if (r // 8) == (c // 8) else 0
    return (255, rough, metal)


def normal(r, c):
    # h = 0.5 sin(x) cos(y), one period per 8 texels; the normal (-dh/dx, -dh/dy, 1) with y
    # up the image (row 0 at the top), encoded [-1, 1] -> [0, 255].
    k = 2.0 * math.pi / 8.0
    x, y = (c + 0.5) * k, -(r + 0.5) * k
    dx = 0.5 * math.cos(x) * math.cos(y)
    dy = -0.5 * math.sin(x) * math.sin(y)
    n = (-dx, -dy, 1.0)
    length = math.sqrt(sum(v * v for v in n))
    return tuple(int(round((v / length * 0.5 + 0.5) * 255.0)) for v in n)


def emissive(r, c):
    if r in (3, 12) or c == 7:
        return (255, 200, 90)
    return (0, 0, 0)


def box():
    """24 vertices (4 per face, counter-clockwise from outside) and 12 triangles, glTF axes.

    Each face's `u` runs 0 -> 2 (the texture repeats twice) and `v` 0 -> 1."""
    faces = [
        # (normal axis, sign), then the corner order.
        ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)]),
        ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
        ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]),
        ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
        ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
        ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),
    ]
    uvs = [(0.0, 1.0), (2.0, 1.0), (2.0, 0.0), (0.0, 0.0)]
    positions, normals, texcoords, indices = [], [], [], []
    for n, corners in faces:
        first = len(positions)
        for (sx, sy, sz), uv in zip(corners, uvs):
            positions.append((sx * HX, sy * HY, sz * HZ))
            normals.append(n)
            texcoords.append(uv)
        indices += [first, first + 1, first + 2, first, first + 2, first + 3]
    return positions, normals, texcoords, indices


def check_winding(positions, normals, indices):
    for t in range(0, len(indices), 3):
        a, b, c = (positions[i] for i in indices[t:t + 3])
        e1 = [b[k] - a[k] for k in range(3)]
        e2 = [c[k] - a[k] for k in range(3)]
        cr = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
        n = normals[indices[t]]
        assert sum(cr[k] * n[k] for k in range(3)) > 0, f"triangle {t // 3} winds inward"


def gltf(positions, normals, texcoords, indices):
    blob = b"".join(struct.pack("<3f", *p) for p in positions)
    blob += b"".join(struct.pack("<3f", *n) for n in normals)
    blob += b"".join(struct.pack("<2f", *t) for t in texcoords)
    index_offset = len(blob)
    blob += b"".join(struct.pack("<H", i) for i in indices)
    n = len(positions)
    doc = {
        "asset": {"version": "2.0", "generator": "tests/fixtures/gltf/textured_box/generate.py"},
        "extensionsUsed": ["KHR_materials_emissive_strength"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "box", "mesh": 0}],
        "meshes": [{
            "name": "box",
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                "indices": 3,
                "material": 0,
            }],
        }],
        "materials": [{
            "name": "box",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "baseColorTexture": {"index": 0},
                "metallicFactor": 1.0,
                "roughnessFactor": 1.0,
                "metallicRoughnessTexture": {"index": 1},
            },
            "normalTexture": {"index": 2, "scale": 1.0},
            "emissiveTexture": {"index": 3},
            "emissiveFactor": [1.0, 1.0, 1.0],
            "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 2.0}},
        }],
        "samplers": [{"wrapS": 10497, "wrapT": 10497}],
        "images": [{"uri": u} for u in ("base.png", "mr.png", "normal.png", "emissive.png")],
        "textures": [{"sampler": 0, "source": i} for i in range(4)],
        "buffers": [{
            "byteLength": len(blob),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode(),
        }],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 12 * n},
            {"buffer": 0, "byteOffset": 12 * n, "byteLength": 12 * n},
            {"buffer": 0, "byteOffset": 24 * n, "byteLength": 8 * n},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": 2 * len(indices)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
             "min": [-HX, -HY, -HZ], "max": [HX, HY, HZ]},
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
    }
    with open(os.path.join(HERE, "textured_box.gltf"), "w", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def fmt(v):
    return repr(float(v))


def obj(positions, texcoords, indices):
    lines = ["# tests/fixtures/gltf/textured_box/generate.py: textured_box.gltf's triangles, Z up"]
    for x, y, z in positions:
        # glTF (x, y, z) -> spec 3.1 (-z, -x, y), gltf.rs's AXIS_FIX.
        lines.append(f"v {fmt(-z)} {fmt(-x)} {fmt(y)}")
    for u, v in texcoords:
        lines.append(f"vt {fmt(u)} {fmt(1.0 - v)}")
    for t in range(0, len(indices), 3):
        a, b, c = (i + 1 for i in indices[t:t + 3])
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    with open(os.path.join(HERE, "textured_box.obj"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


MJCF = """<!-- Plan H, HT2: textured_box.gltf's material declared in MJCF, for the glTF-vs-MJCF oracle.
     Written by generate.py; see PROVENANCE.json. -->
<mujoco model="textured_box">
  <compiler meshdir="." texturedir="."/>
  <asset>
    <mesh name="box" file="textured_box.obj"/>
    <texture name="base" type="2d" file="base.png" colorspace="sRGB"/>
    <texture name="mr" type="2d" file="mr.png" colorspace="linear"/>
    <texture name="normal" type="2d" file="normal.png" colorspace="linear"/>
    <texture name="glow" type="2d" file="emissive.png" colorspace="sRGB"/>
    <material name="box" emission="2">
      <layer role="rgb" texture="base"/>
      <layer role="orm" texture="mr"/>
      <layer role="normal" texture="normal"/>
      <layer role="emissive" texture="glow"/>
    </material>
  </asset>
  <worldbody>
    <geom name="box" type="mesh" mesh="box" material="box" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def main():
    png("base.png", grid(base), srgb=True)
    png("mr.png", grid(mr), srgb=False)
    png("normal.png", grid(normal), srgb=False)
    png("emissive.png", grid(emissive), srgb=True)
    positions, normals, texcoords, indices = box()
    check_winding(positions, normals, indices)
    gltf(positions, normals, texcoords, indices)
    obj(positions, texcoords, indices)
    with open(os.path.join(HERE, "textured_box.xml"), "w", newline="\n") as f:
        f.write(MJCF)


if __name__ == "__main__":
    main()
