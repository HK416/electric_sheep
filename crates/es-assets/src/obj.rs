//! Wavefront OBJ geometry — `v`, `vt` and `f` — to positions, UVs and triangle indices
//! (packets M10/W2a, plan H HT1).
//!
//! Only what a collision / visual mesh needs: `v` gives the positions in file order (OBJ
//! already indexes them, so nothing is deduplicated), `f` gives the triangles. The index forms
//! `a`, `a/b`, `a//c` and `a/b/c` are read; indices are 1-based and a negative one counts back
//! from the vertices seen *so far*. A face with more than three vertices is fan-triangulated
//! around its first vertex. `vn`, `o`, `g`, `s`, `mtllib`, `usemtl` and comments are ignored.
//!
//! **UVs** (plan H, HT1): a file whose faces name `vt` indices on every corner gets one output
//! vertex per distinct `(v, vt)` pair, in first-use order, with `v` flipped to `1 - v` — the
//! convention `MuJoCo`'s OBJ decoder applies, so row 0 of a texture is `v = 0` here as in every
//! other UV of this crate. A file with no `vt` on its faces decodes exactly as before: the
//! positions in file order and no UVs, so no committed mesh digest moves.
//!
//! No host `libm` (spec 3.4 `DET-010`): `str::parse::<f32>` only.

use std::collections::BTreeMap;

/// Positions, per-vertex UVs when the faces carry them, and triangle indices.
pub type Decoded = (Vec<[f32; 3]>, Option<Vec<[f32; 2]>>, Vec<u32>);

/// Positions and triangle indices only, the geometry every face describes whether or not it
/// names UVs: [`parse_uv`] without the split.
pub fn parse(text: &str) -> Result<crate::stl::Triangles, String> {
    let (positions, _, corners) = parse_corners(text)?;
    Ok((positions, corners.iter().map(|(v, _)| *v).collect()))
}

/// Decodes `text` with its UVs, or names what is wrong with it.
pub fn parse_uv(text: &str) -> Result<Decoded, String> {
    let (positions, texcoords, corners) = parse_corners(text)?;
    if corners.iter().any(|(_, t)| t.is_none()) {
        return Ok((positions, None, corners.iter().map(|(v, _)| *v).collect()));
    }
    // One vertex per distinct (v, vt), numbered in first-use order.
    let mut seen: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let (mut out_p, mut out_t, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for (v, t) in corners {
        let t = t.unwrap_or(0);
        let next = u32::try_from(out_p.len()).unwrap_or(u32::MAX);
        let i = *seen.entry((v, t)).or_insert_with(|| {
            out_p.push(positions[v as usize]);
            out_t.push(texcoords[t as usize]);
            next
        });
        indices.push(i);
    }
    Ok((out_p, Some(out_t), indices))
}

/// Positions, texcoords (`v` already flipped) and every face corner as (position, texcoord).
type Corners = (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<(u32, Option<u32>)>);

fn parse_corners(text: &str) -> Result<Corners, String> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    // Every face corner as (position index, texcoord index or none).
    let mut corners: Vec<(u32, Option<u32>)> = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let at = |what: &str| format!("OBJ line {}: {what}", line_no + 1);
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            Some("v") => {
                let mut v = [0.0f32; 3];
                for (k, slot) in v.iter_mut().enumerate() {
                    let value = tokens
                        .next()
                        .ok_or_else(|| at(&format!("`v` wants three numbers, found {k}")))?;
                    *slot = value
                        .parse::<f32>()
                        .map_err(|e| at(&format!("`v` value `{value}`: {e}")))?;
                }
                positions.push(v);
            }
            Some("vt") => {
                let mut t = [0.0f32; 2];
                for (k, slot) in t.iter_mut().enumerate() {
                    let value = tokens
                        .next()
                        .ok_or_else(|| at(&format!("`vt` wants two numbers, found {k}")))?;
                    *slot = value
                        .parse::<f32>()
                        .map_err(|e| at(&format!("`vt` value `{value}`: {e}")))?;
                }
                texcoords.push([t[0], 1.0 - t[1]]);
            }
            Some("f") => {
                let mut face = Vec::new();
                for token in tokens {
                    let mut parts = token.split('/');
                    let v = resolve(parts.next().unwrap_or(token), positions.len(), token)
                        .map_err(|e| at(&e))?;
                    let vt = match parts.next() {
                        Some(s) if !s.is_empty() => {
                            Some(resolve(s, texcoords.len(), token).map_err(|e| at(&e))?)
                        }
                        _ => None,
                    };
                    face.push((v, vt));
                }
                if face.len() < 3 {
                    return Err(at(&format!("`f` has {} vertices, not three", face.len())));
                }
                for k in 1..face.len() - 1 {
                    corners.extend_from_slice(&[face[0], face[k], face[k + 1]]);
                }
            }
            _ => {}
        }
    }
    if corners.is_empty() {
        return Err("OBJ: no `f` line, so the file describes no triangle".to_owned());
    }
    Ok((positions, texcoords, corners))
}

/// A 1-based or negative OBJ index into a list of `len` items seen so far.
fn resolve(text: &str, len: usize, token: &str) -> Result<u32, String> {
    let raw = text
        .parse::<i64>()
        .map_err(|e| format!("`f` index `{token}`: {e}"))?;
    let seen = i64::try_from(len).unwrap_or(i64::MAX);
    let resolved = if raw < 0 { seen + raw } else { raw - 1 };
    if resolved < 0 || resolved >= seen {
        return Err(format!(
            "`f` index {raw} is outside the {len} vertices declared so far"
        ));
    }
    Ok(u32::try_from(resolved).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // exact values: the flip of 0, 1, 0.25
    fn vt_splits_vertices_and_flips_v() {
        let text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
                    vt 0.5 0.25\nf 1/1 2/2 3/3 4/4\nf 1/5 3/3 2/2\n";
        let (p, t, i) = parse_uv(text).unwrap();
        let t = t.expect("every corner names a vt");
        assert_eq!(p.len(), 5, "(1, 5) is a new pair, the rest are reused");
        assert_eq!(i, vec![0, 1, 2, 0, 2, 3, 4, 2, 1]);
        assert_eq!(t[0], [0.0, 1.0]);
        assert_eq!(t[2], [1.0, 0.0]);
        assert_eq!(t[4], [0.5, 0.75]);
        assert_eq!(p[4], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_face_without_vt_keeps_the_positions_as_they_were() {
        let (p, t, i) = parse_uv("v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nf 1 2 3/1\n").unwrap();
        assert_eq!((p.len(), t, i), (3, None, vec![0, 1, 2]));
    }
}
