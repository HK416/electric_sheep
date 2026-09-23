//! Wavefront OBJ geometry — `v` and `f` — to positions and triangle indices (packet M10/W2a).
//!
//! Only what a collision / visual mesh needs: `v` gives the positions in file order (OBJ
//! already indexes them, so nothing is deduplicated), `f` gives the triangles. The index forms
//! `a`, `a/b`, `a//c` and `a/b/c` all resolve to the part before the first `/`; indices are
//! 1-based and a negative one counts back from the vertices seen *so far*. A face with more
//! than three vertices is fan-triangulated around its first vertex. Everything else — `vt`,
//! `vn`, `o`, `g`, `s`, `mtllib`, `usemtl`, comments — is ignored, because a mesh without its
//! material is still the right geometry (materials are M7 R6).
//!
//! No host `libm` (spec 3.4 `DET-010`): `str::parse::<f32>` only.

use crate::stl::Triangles;

/// Decodes `text`, or names what is wrong with it.
pub fn parse(text: &str) -> Result<Triangles, String> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices = Vec::new();
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
            Some("f") => {
                let mut face = Vec::new();
                for token in tokens {
                    let vertex = token.split('/').next().unwrap_or(token);
                    let raw = vertex
                        .parse::<i64>()
                        .map_err(|e| at(&format!("`f` index `{token}`: {e}")))?;
                    // 1-based, or negative counting back from the vertices seen so far.
                    let seen = i64::try_from(positions.len()).unwrap_or(i64::MAX);
                    let resolved = if raw < 0 { seen + raw } else { raw - 1 };
                    if resolved < 0 || resolved >= seen {
                        return Err(at(&format!(
                            "`f` index {raw} is outside the {} vertices declared so far",
                            positions.len()
                        )));
                    }
                    face.push(u32::try_from(resolved).unwrap_or(u32::MAX));
                }
                if face.len() < 3 {
                    return Err(at(&format!("`f` has {} vertices, not three", face.len())));
                }
                for k in 1..face.len() - 1 {
                    indices.extend_from_slice(&[face[0], face[k], face[k + 1]]);
                }
            }
            _ => {}
        }
    }
    if indices.is_empty() {
        return Err("OBJ: no `f` line, so the file describes no triangle".to_owned());
    }
    Ok((positions, indices))
}
