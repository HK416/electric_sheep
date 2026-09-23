//! STL, binary and ASCII, to positions and triangle indices (packet M10/W2a).
//!
//! STL is a triangle soup: every facet repeats its three vertices in full and carries an
//! advisory normal that exporters often leave at zero. This reader deduplicates vertices by
//! their exact `f32` bits in first-seen order — deterministic, and the only way `blake3` over
//! the result can be one number on every machine (spec 5.3) — and keeps no normals: the
//! renderer derives them from the winding, so a facet wound against its stored normal has its
//! second and third vertices swapped instead.
//!
//! No host `libm` (spec 3.4 `DET-010`): `f32::from_le_bytes` and `str::parse::<f32>` only.

use std::collections::BTreeMap;

/// Positions and a flat triangle index list, three indices per triangle.
pub type Triangles = (Vec<[f32; 3]>, Vec<u32>);

/// Decodes `bytes`, or names what is wrong with them.
///
/// The file is binary exactly when its length is the `80 + 4 + 50 * n` the format pins and the
/// declared triangle count agrees; an 80-byte header may legally begin with `solid`, so
/// sniffing the first word alone would misread real files (see `docs/api-notes/mujoco.md`).
pub fn parse(bytes: &[u8]) -> Result<Triangles, String> {
    let mut out = Builder::default();
    match binary_facets(bytes) {
        Some(n) => binary(&mut out, bytes, n),
        None => ascii(&mut out, bytes)?,
    }
    if out.indices.is_empty() {
        return Err(format!(
            "{} bytes are neither a binary STL (80 + 4 + 50*n) nor an ASCII STL with a facet",
            bytes.len()
        ));
    }
    Ok((out.positions, out.indices))
}

/// The declared triangle count when `bytes` is a well-formed binary STL.
fn binary_facets(bytes: &[u8]) -> Option<usize> {
    let rest = bytes.len().checked_sub(84)?;
    if rest % 50 != 0 {
        return None;
    }
    let declared = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    (declared == rest / 50).then_some(declared)
}

fn binary(out: &mut Builder, bytes: &[u8], facets: usize) {
    let at = |base: usize| {
        f32::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ])
    };
    for i in 0..facets {
        // 50 B per facet: the normal, three vertices, and a two-byte attribute count.
        let base = 84 + 50 * i;
        let vec3 = |k: usize| {
            [
                at(base + 12 * k),
                at(base + 12 * k + 4),
                at(base + 12 * k + 8),
            ]
        };
        out.facet(vec3(0), [vec3(1), vec3(2), vec3(3)]);
    }
}

/// The free-form grammar: `facet normal nx ny nz` / `vertex x y z`, everything else ignored.
fn ascii(out: &mut Builder, bytes: &[u8]) -> Result<(), String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| format!("not a binary STL, and not UTF-8 either: {e}"))?;
    let mut tokens = text.split_whitespace();
    let mut normal = [0.0f32; 3];
    let mut verts: Vec<[f32; 3]> = Vec::new();
    while let Some(token) = tokens.next() {
        let slots = match token {
            "normal" => &mut normal,
            "vertex" => {
                verts.push([0.0; 3]);
                verts.last_mut().expect("just pushed")
            }
            _ => continue,
        };
        for (k, slot) in slots.iter_mut().enumerate() {
            let value = tokens
                .next()
                .ok_or_else(|| format!("ASCII STL: `{token}` wants three numbers, found {k}"))?;
            *slot = value
                .parse::<f32>()
                .map_err(|e| format!("ASCII STL: `{token}` value `{value}`: {e}"))?;
        }
        if verts.len() == 3 {
            out.facet(normal, [verts[0], verts[1], verts[2]]);
            verts.clear();
        }
    }
    if verts.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "ASCII STL: a facet ends with {} vertices, not three",
            verts.len()
        ))
    }
}

/// Deduplicating triangle-soup accumulator.
#[derive(Default)]
struct Builder {
    positions: Vec<[f32; 3]>,
    /// Exact `f32` bits to the index first seen for them.
    seen: BTreeMap<[u32; 3], u32>,
    indices: Vec<u32>,
}

impl Builder {
    fn index_of(&mut self, v: [f32; 3]) -> u32 {
        let key = [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
        if let Some(index) = self.seen.get(&key) {
            return *index;
        }
        let index = self.positions.len() as u32;
        self.positions.push(v);
        self.seen.insert(key, index);
        index
    }

    /// Appends one triangle, swapping `b` and `c` when the winding disagrees with a non-zero
    /// stored normal. The test is `f64` add / sub / mul only — exact, so the decision is the
    /// same on every platform.
    fn facet(&mut self, normal: [f32; 3], [first, second, third]: [[f32; 3]; 3]) {
        let delta = |to: [f32; 3], from: [f32; 3]| {
            [
                f64::from(to[0]) - f64::from(from[0]),
                f64::from(to[1]) - f64::from(from[1]),
                f64::from(to[2]) - f64::from(from[2]),
            ]
        };
        let edge0 = delta(second, first);
        let edge1 = delta(third, first);
        let cross = [
            edge0[1] * edge1[2] - edge0[2] * edge1[1],
            edge0[2] * edge1[0] - edge0[0] * edge1[2],
            edge0[0] * edge1[1] - edge0[1] * edge1[0],
        ];
        let dot = f64::from(normal[0]) * cross[0]
            + f64::from(normal[1]) * cross[1]
            + f64::from(normal[2]) * cross[2];
        let wound = if dot < 0.0 {
            [first, third, second]
        } else {
            [first, second, third]
        };
        for vertex in wound {
            let index = self.index_of(vertex);
            self.indices.push(index);
        }
    }
}
