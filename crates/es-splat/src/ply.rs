//! The 3D Gaussian Splatting PLY reader and writer.
//!
//! Layout, and why every claim about it is `unverified`: `docs/api-notes/
//! gaussian-splat-ply.md`. Reads `ascii 1.0` and `binary_little_endian 1.0`; writes the
//! latter, in the reference property order.
//!
//! # Axis conversion (spec 3.1) — `unverified`
//!
//! A 3DGS reconstruction inherits COLMAP/OpenCV world conventions: +Y down, +Z forward. Spec
//! 3.1 is right-handed Z-up. The fixed rotation between them is `R_x(-90 deg)`, which as a
//! component map is
//!
//! ```text
//! x_es = +x_file      y_es = +z_file      z_es = -y_file
//! ```
//!
//! and it is implemented as exactly that swizzle rather than as a quaternion product: same
//! rotation, but bit-exact and exactly invertible, which is what makes a byte-exact round
//! trip possible at all. Quaternions get the same swizzle on their vector part (conjugating a
//! quaternion by a rotation rotates its vector part and leaves its scalar part alone).
//!
//! Scales are *not* converted: they are extents in the Gaussian's own frame, not world
//! directions. `f_rest` (SH degrees 1..3) is not band-rotated either — that needs Wigner-D
//! matrices and a renderer to be wrong in front of — so a capture with view-dependent colour
//! imports with a warning.
//!
//! A wrong guess here costs accuracy in the *initial* placement only: spec 16.2 fits
//! `T_robot_scan` afterwards ([`crate::Similarity`]), which absorbs any residual rotation.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::SplatScene;

/// Valid `f_rest_*` counts, indexed by SH degree: `3 * ((d+1)^2 - 1)`.
const SH_REST_COUNTS: [usize; 4] = [0, 9, 24, 45];

/// Alpha is clamped into `[ALPHA_FLOOR, 1 - ALPHA_FLOOR]` before `logit` on write, so a
/// saturated Gaussian writes a large finite value instead of an infinity.
const ALPHA_FLOOR: f32 = 1e-7;

/// Why a byte slice is not a Gaussian splat capture. Also carries the alignment-fit failures
/// of [`crate::Similarity`] and [`crate::ColorAffine`]: one crate, one error type.
///
/// Every variant names the element, property or vertex at fault. Malformed input is one of
/// these, never a panic and never a silent default.
#[derive(Debug, Error)]
pub enum SplatError {
    #[error("not a PLY file: the first line is `{0}`, expected `ply`")]
    NotPly(String),
    #[error("the header has no `end_header` line")]
    UnterminatedHeader,
    #[error("unsupported `format {0}`: only `ascii 1.0` and `binary_little_endian 1.0` are read")]
    UnsupportedFormat(String),
    #[error("header line {line}: `{text}` is not a PLY header statement")]
    BadHeaderLine { line: usize, text: String },
    #[error("property `{property}`: unsupported type `{ty}`")]
    UnsupportedType { property: String, ty: String },
    #[error("property `{0}`: list properties do not occur in a splat PLY and are not read")]
    ListProperty(String),
    #[error("element `{0}` precedes `vertex`; a splat PLY has one `vertex` element and no other")]
    ElementBeforeVertex(String),
    #[error("the header declares no `element vertex`")]
    NoVertexElement,
    #[error("required property `{0}` is missing from `element vertex`")]
    MissingProperty(String),
    #[error("{count} `f_rest_*` properties: expected 0, 9, 24 or 45 (SH degree 0..=3)")]
    BadShCount { count: usize },
    #[error(
        "`f_rest_{index}` is missing: the `f_rest_*` properties must run from 0 without a gap"
    )]
    ShGap { index: usize },
    #[error("vertex data ends after {available} of the {needed} bytes {count} vertices need")]
    Truncated {
        count: usize,
        available: usize,
        needed: usize,
    },
    #[error("vertex {vertex}: {found} values on the line, expected {expected}")]
    BadFieldCount {
        vertex: usize,
        found: usize,
        expected: usize,
    },
    #[error("vertex {vertex}, property `{property}`: `{text}` is not a number")]
    BadNumber {
        vertex: usize,
        property: String,
        text: String,
    },
    #[error("vertex {vertex}, property `{property}`: value is not finite")]
    NotFinite { vertex: usize, property: String },
    #[error("similarity fit: {src} source points and {dst} target points")]
    PointCountMismatch { src: usize, dst: usize },
    #[error("similarity fit needs at least 3 correspondences, got {0}")]
    TooFewPoints(usize),
    #[error(
        "similarity fit: the source points are coincident, so no scale or rotation is defined"
    )]
    DegenerateFit,
    #[error("fit sample {0} contains a non-finite value")]
    NonFiniteSample(usize),
    #[error("colour fit needs at least 2 samples, got {0}")]
    TooFewColorSamples(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Ascii,
    BinaryLe,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScalarTy {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarTy {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Reads one little-endian value. `at + self.size() <= bytes.len()` is the caller's
    /// contract, established once by the [`SplatError::Truncated`] check.
    fn read_le(self, bytes: &[u8], at: usize) -> f64 {
        let b = &bytes[at..at + self.size()];
        match self {
            Self::I8 => f64::from(i8::from_le_bytes([b[0]])),
            Self::U8 => f64::from(b[0]),
            Self::I16 => f64::from(i16::from_le_bytes([b[0], b[1]])),
            Self::U16 => f64::from(u16::from_le_bytes([b[0], b[1]])),
            Self::I32 => f64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            Self::U32 => f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            Self::F32 => f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            Self::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        }
    }
}

/// One declared property of `element vertex`.
#[derive(Clone, Debug)]
struct Prop {
    name: String,
    ty: ScalarTy,
    /// Byte offset within a vertex record (binary only).
    offset: usize,
}

/// The parsed header: what the vertex record contains and where its data starts.
#[derive(Debug)]
struct Header {
    format: Format,
    props: Vec<Prop>,
    stride: usize,
    count: usize,
    data_at: usize,
    /// Property name to index in `props`.
    index: BTreeMap<String, usize>,
}

/// Splits `bytes` into lines at `\n`, trimming a trailing `\r`, and returns the offset just
/// past each line. Header text is ascii by the format's definition; anything else in it is a
/// [`SplatError::BadHeaderLine`] via the statement match, not a decode failure here.
fn header_lines(bytes: &[u8]) -> impl Iterator<Item = (String, usize)> + '_ {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        if at >= bytes.len() {
            return None;
        }
        let end = bytes[at..]
            .iter()
            .position(|b| *b == b'\n')
            .map_or(bytes.len(), |p| at + p);
        let mut line = String::from_utf8_lossy(&bytes[at..end]).into_owned();
        if line.ends_with('\r') {
            line.pop();
        }
        at = (end + 1).min(bytes.len());
        Some((line, at))
    })
}

fn parse_header(bytes: &[u8]) -> Result<Header, SplatError> {
    let mut lines = header_lines(bytes);
    let (magic, _) = lines.next().unwrap_or_default();
    if magic.trim() != "ply" {
        return Err(SplatError::NotPly(magic));
    }

    let mut format = None;
    let mut props: Vec<Prop> = Vec::new();
    let mut count = None;
    let mut in_vertex = false;
    let mut stride = 0usize;

    for (n, (line, next)) in lines.enumerate() {
        let line = line.trim();
        let mut words = line.split_whitespace();
        match words.next() {
            None | Some("comment" | "obj_info") => {}
            Some("format") => {
                let rest = words.collect::<Vec<_>>().join(" ");
                format = Some(match rest.as_str() {
                    "ascii 1.0" => Format::Ascii,
                    "binary_little_endian 1.0" => Format::BinaryLe,
                    _ => return Err(SplatError::UnsupportedFormat(rest)),
                });
            }
            Some("element") => {
                let name = words.next().unwrap_or_default().to_owned();
                if name == "vertex" {
                    in_vertex = true;
                    count = Some(
                        words
                            .next()
                            .and_then(|c| c.parse::<usize>().ok())
                            .ok_or_else(|| SplatError::BadHeaderLine {
                                line: n + 2,
                                text: line.to_owned(),
                            })?,
                    );
                } else {
                    if count.is_none() {
                        return Err(SplatError::ElementBeforeVertex(name));
                    }
                    // A trailing element's properties are not ours; vertex data comes first.
                    in_vertex = false;
                }
            }
            Some("property") => {
                if !in_vertex {
                    continue;
                }
                let ty = words.next().unwrap_or_default();
                let name = words.next().unwrap_or_default().to_owned();
                if ty == "list" {
                    return Err(SplatError::ListProperty(name));
                }
                let ty = ScalarTy::parse(ty).ok_or_else(|| SplatError::UnsupportedType {
                    property: name.clone(),
                    ty: ty.to_owned(),
                })?;
                props.push(Prop {
                    name,
                    ty,
                    offset: stride,
                });
                stride += ty.size();
            }
            Some("end_header") => {
                let format =
                    format.ok_or_else(|| SplatError::UnsupportedFormat("<missing>".to_owned()))?;
                let count = count.ok_or(SplatError::NoVertexElement)?;
                let index = props
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (p.name.clone(), i))
                    .collect();
                return Ok(Header {
                    format,
                    props,
                    stride,
                    count,
                    data_at: next,
                    index,
                });
            }
            Some(_) => {
                return Err(SplatError::BadHeaderLine {
                    line: n + 2,
                    text: line.to_owned(),
                })
            }
        }
    }
    Err(SplatError::UnterminatedHeader)
}

/// Which property index feeds which field of a [`SplatScene`].
#[derive(Debug)]
struct Plan {
    position: [usize; 3],
    dc: [usize; 3],
    rest: Vec<usize>,
    opacity: usize,
    scale: [usize; 3],
    /// File order wxyz.
    rot: [usize; 4],
    sh_degree: u32,
}

/// Properties the reader knows about and does not report as unknown. `nx`/`ny`/`nz` are read
/// and discarded: a 3D Gaussian has no normal and the reference writer stores zeros.
const KNOWN: [&str; 6] = ["x", "y", "z", "nx", "ny", "nz"];

impl Plan {
    fn new(header: &Header, warnings: &mut Vec<String>) -> Result<Self, SplatError> {
        let need = |name: &str| -> Result<usize, SplatError> {
            header
                .index
                .get(name)
                .copied()
                .ok_or_else(|| SplatError::MissingProperty(name.to_owned()))
        };

        let rest_count = header
            .props
            .iter()
            .filter(|p| p.name.starts_with("f_rest_"))
            .count();
        let degree = SH_REST_COUNTS
            .iter()
            .position(|c| *c == rest_count)
            .ok_or(SplatError::BadShCount { count: rest_count })?;
        let mut rest = Vec::with_capacity(rest_count);
        for i in 0..rest_count {
            let name = format!("f_rest_{i}");
            rest.push(
                header
                    .index
                    .get(&name)
                    .copied()
                    .ok_or(SplatError::ShGap { index: i })?,
            );
        }

        for p in &header.props {
            let known = KNOWN.contains(&p.name.as_str())
                || p.name.starts_with("f_dc_")
                || p.name.starts_with("f_rest_")
                || p.name == "opacity"
                || p.name.starts_with("scale_")
                || p.name.starts_with("rot_");
            if !known {
                warnings.push(format!(
                    "property `{}` is not a splat attribute; ignored",
                    p.name
                ));
            }
        }
        if degree > 0 {
            warnings.push(format!(
                "SH degrees 1..={degree} are carried through unrotated; the axis fix is not \
                 applied to `f_rest_*` (see docs/api-notes/gaussian-splat-ply.md)"
            ));
        }

        Ok(Self {
            position: [need("x")?, need("y")?, need("z")?],
            dc: [need("f_dc_0")?, need("f_dc_1")?, need("f_dc_2")?],
            rest,
            opacity: need("opacity")?,
            scale: [need("scale_0")?, need("scale_1")?, need("scale_2")?],
            rot: [
                need("rot_0")?,
                need("rot_1")?,
                need("rot_2")?,
                need("rot_3")?,
            ],
            sh_degree: degree as u32,
        })
    }
}

/// `sigmoid`, entirely in `f32` through [`es_math::approx`] (`exp`) so `asset_hash` is
/// reproducible across libms (spec 3.2/3.4, `es_math::approx` is the one deterministic
/// transcendental implementation) rather than depending on the host's `f64::exp`. This trades
/// `f64::exp`'s correctly-rounded result for `es_math::approx::exp`'s bounded-ULP one; see
/// `docs/design/splat-real2sim.md` section 1.1 for the round-trip consequence.
fn sigmoid(x: f64) -> f32 {
    let x = x as f32;
    1.0 / (1.0 + es_math::approx::exp(-x))
}

/// Inverse of [`sigmoid`]. Saturating alpha is clamped rather than sent to an infinity.
fn logit(y: f32) -> f32 {
    let y = y.clamp(ALPHA_FLOOR, 1.0 - ALPHA_FLOOR);
    es_math::approx::ln(y / (1.0 - y))
}

/// How many vertex records the file could possibly hold, bounding a hostile header's
/// declared `count` by the file's actual size rather than trusting it (P-M3-R3). A binary
/// record needs exactly `stride` bytes; an ascii one needs at least two bytes per property
/// (one digit, one separator). `Vec::with_capacity` calls below use this, not `count`
/// directly, so a header that declares far more vertices than the file has room for cannot
/// amplify the reserve past what the file could actually contain.
fn reserve_count(
    format: Format,
    props: usize,
    stride: usize,
    count: usize,
    available: usize,
) -> usize {
    let bytes_per_record = match format {
        Format::Ascii => 2 * props.max(1),
        Format::BinaryLe => stride.max(1),
    };
    count.min(available / bytes_per_record)
}

/// Reads a Gaussian splat capture. See the module docs for the axis conversion and the API
/// note for the layout; both are `unverified`.
///
/// An unrecognised property is a warning on the returned scene, not a rejection.
pub fn import_ply(bytes: &[u8]) -> Result<SplatScene, SplatError> {
    let header = parse_header(bytes)?;
    let mut warnings = Vec::new();
    let plan = Plan::new(&header, &mut warnings)?;
    let count = header.count;

    // Establishes the slice bounds that `ScalarTy::read_le` relies on, and bounds every
    // allocation below by the file's actual size rather than by its declared count.
    let available = bytes.len().saturating_sub(header.data_at);
    let ascii_lines: Vec<&[u8]> = if header.format == Format::Ascii {
        bytes[header.data_at..]
            .split(|b| *b == b'\n')
            .filter(|l| !l.iter().all(u8::is_ascii_whitespace))
            .collect()
    } else {
        let needed = count.saturating_mul(header.stride);
        if available < needed {
            return Err(SplatError::Truncated {
                count,
                available,
                needed,
            });
        }
        Vec::new()
    };
    if header.format == Format::Ascii && ascii_lines.len() < count {
        return Err(SplatError::Truncated {
            count,
            available: ascii_lines.len(),
            needed: count,
        });
    }

    let per_rest = plan.rest.len();
    let reserve = reserve_count(
        header.format,
        header.props.len(),
        header.stride,
        count,
        available,
    );
    let mut scene = SplatScene {
        positions: Vec::with_capacity(3 * reserve),
        scales: Vec::with_capacity(3 * reserve),
        rotations: Vec::with_capacity(4 * reserve),
        opacities: Vec::with_capacity(reserve),
        sh_dc: Vec::with_capacity(3 * reserve),
        sh_rest: Vec::with_capacity(per_rest * reserve),
        sh_degree: plan.sh_degree,
        bounds: crate::Bounds::default(),
        asset: SplatScene::asset_ref([0u8; 32]),
        warnings,
    };

    let mut fields: Vec<&str> = Vec::new();
    for vi in 0..count {
        if header.format == Format::Ascii {
            let line = ascii_lines
                .get(vi)
                .and_then(|l| std::str::from_utf8(l).ok())
                .unwrap_or_default();
            fields.clear();
            fields.extend(line.split_whitespace());
            if fields.len() != header.props.len() {
                return Err(SplatError::BadFieldCount {
                    vertex: vi,
                    found: fields.len(),
                    expected: header.props.len(),
                });
            }
        }
        let base = header.data_at + vi * header.stride;
        let get = |slot: usize| -> Result<f64, SplatError> {
            let prop = &header.props[slot];
            let value = match header.format {
                Format::BinaryLe => prop.ty.read_le(bytes, base + prop.offset),
                // Parsed at the *declared* width, so an ascii file and its binary twin decode
                // to bit-identical values: `f32::from_str` on a shortest round-trip literal
                // returns the same `f32` the binary record holds, whereas parsing it as
                // `f64` would feed the activations a slightly different number.
                Format::Ascii => {
                    let text = fields[slot];
                    let bad = || SplatError::BadNumber {
                        vertex: vi,
                        property: prop.name.clone(),
                        text: text.to_owned(),
                    };
                    match prop.ty {
                        ScalarTy::F64 => text.parse::<f64>().map_err(|_| bad())?,
                        ScalarTy::F32 => f64::from(text.parse::<f32>().map_err(|_| bad())?),
                        _ => text.parse::<f64>().map_err(|_| bad())?,
                    }
                }
            };
            if value.is_finite() {
                Ok(value)
            } else {
                Err(SplatError::NotFinite {
                    vertex: vi,
                    property: prop.name.clone(),
                })
            }
        };

        // Axis fix, as the exact swizzle documented above.
        let (fx, fy, fz) = (
            get(plan.position[0])?,
            get(plan.position[1])?,
            get(plan.position[2])?,
        );
        scene
            .positions
            .extend_from_slice(&[fx as f32, fz as f32, -fy as f32]);

        for slot in plan.scale {
            // `f32`, through `es_math::approx::exp` (see `sigmoid` above): the same
            // determinism reason applies to every activation that feeds `asset_hash`.
            scene.scales.push(es_math::approx::exp(get(slot)? as f32));
        }

        // File order is wxyz and unnormalised; store xyzw, unit, `w >= 0` (spec 3.1).
        let (qw, qx, qy, qz) = (
            get(plan.rot[0])?,
            get(plan.rot[1])?,
            get(plan.rot[2])?,
            get(plan.rot[3])?,
        );
        let (sx, sy, sz, sw) = (qx, qz, -qy, qw);
        let norm = (sx * sx + sy * sy + sz * sz + sw * sw).sqrt();
        // Divide (never multiply by a reciprocal): a quaternion already unit in `f32` divides
        // by exactly 1.0 and comes back bit-identical, which the round-trip test relies on.
        let divisor = if norm > 0.0 {
            if sw < 0.0 {
                -norm
            } else {
                norm
            }
        } else {
            1.0
        };
        scene.rotations.extend_from_slice(&[
            (sx / divisor) as f32,
            (sy / divisor) as f32,
            (sz / divisor) as f32,
            (sw / divisor) as f32,
        ]);

        scene.opacities.push(sigmoid(get(plan.opacity)?));
        for d in plan.dc {
            scene.sh_dc.push(get(d)? as f32);
        }
        for r in 0..per_rest {
            scene.sh_rest.push(get(plan.rest[r])? as f32);
        }
    }

    scene.recompute_bounds();
    scene.refresh_asset_hash();
    Ok(scene)
}

impl SplatScene {
    /// Serialises to `binary_little_endian 1.0` in the reference property order, inverting
    /// both activations.
    ///
    /// One encoding on the way out: the reader takes two, the writer picks the interchange
    /// one. The inversion is lossy at the last bit — `exp`/`ln` and `sigmoid`/`logit` are not
    /// exact inverses in `f32` — so a round trip is byte-exact only for values that are fixed
    /// points of those pairs; see `docs/design/splat-real2sim.md` section 1.1.
    pub fn write_ply(&self) -> Vec<u8> {
        let count = self.len();
        let per_rest = self.sh_rest.len().checked_div(count).unwrap_or(0);
        let mut names = vec![
            "x".to_owned(),
            "y".to_owned(),
            "z".to_owned(),
            "nx".to_owned(),
            "ny".to_owned(),
            "nz".to_owned(),
        ];
        names.extend((0..3).map(|i| format!("f_dc_{i}")));
        names.extend((0..per_rest).map(|i| format!("f_rest_{i}")));
        names.push("opacity".to_owned());
        names.extend((0..3).map(|i| format!("scale_{i}")));
        names.extend((0..4).map(|i| format!("rot_{i}")));

        let mut out = Vec::with_capacity(64 + names.len() * 20 + count * names.len() * 4);
        out.extend_from_slice(b"ply\nformat binary_little_endian 1.0\n");
        out.extend_from_slice(format!("element vertex {count}\n").as_bytes());
        for name in &names {
            out.extend_from_slice(format!("property float {name}\n").as_bytes());
        }
        out.extend_from_slice(b"end_header\n");

        let put = |value: f32, out: &mut Vec<u8>| out.extend_from_slice(&value.to_le_bytes());
        for i in 0..count {
            // Inverse swizzle of the import axis fix: x_file = x_es, y_file = -z_es,
            // z_file = y_es.
            let (px, py, pz) = (
                self.positions[3 * i],
                self.positions[3 * i + 1],
                self.positions[3 * i + 2],
            );
            put(px, &mut out);
            put(-pz, &mut out);
            put(py, &mut out);
            for _ in 0..3 {
                put(0.0, &mut out);
            }
            for c in 0..3 {
                put(self.sh_dc[3 * i + c], &mut out);
            }
            for r in 0..per_rest {
                put(self.sh_rest[per_rest * i + r], &mut out);
            }
            put(logit(self.opacities[i]), &mut out);
            for c in 0..3 {
                // A non-positive standard deviation has no logarithm; clamp rather than
                // write a NaN that no reader could interpret. `f32`, through
                // `es_math::approx::ln` — see `sigmoid`'s doc comment for why.
                let s = self.scales[3 * i + c].max(f32::MIN_POSITIVE);
                put(es_math::approx::ln(s), &mut out);
            }
            let (qx, qy, qz, qw) = (
                self.rotations[4 * i],
                self.rotations[4 * i + 1],
                self.rotations[4 * i + 2],
                self.rotations[4 * i + 3],
            );
            put(qw, &mut out);
            put(qx, &mut out);
            put(-qz, &mut out);
            put(qy, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod reserve_tests {
    use super::*;

    /// P-M3-R3's oracle in miniature: a ~1 MB ascii file declaring 500,000 vertices at the
    /// full SH-degree-3 property count (59 properties) must not reserve anywhere near the
    /// ~118 MB the pre-fix `3 * count` / `per_rest * count` calls would have asked for — six
    /// `f32` arrays sized off `reserve`, 59 slots per vertex in the worst case (every
    /// property lands in a distinct array), must stay under 8 MB total.
    #[test]
    fn ascii_reserve_is_bounded_by_file_size_not_declared_count() {
        let count = 500_000;
        let available = 1_000_000;
        let props = 59;
        let reserve = reserve_count(Format::Ascii, props, 0, count, available);
        assert!(
            reserve < count / 10,
            "reserve {reserve} barely below {count}"
        );
        let worst_case_bytes = reserve * props * std::mem::size_of::<f32>();
        assert!(
            worst_case_bytes < 8_000_000,
            "reserve {reserve} implies {worst_case_bytes} bytes, expected < 8 MB"
        );
    }

    /// The binary path was already bounded by the `Truncated` check before this fix (the
    /// review's own finding), so this pins that `reserve_count` does not regress it: a wildly
    /// over-declared `count` still caps at what `available / stride` full records fit.
    #[test]
    fn binary_reserve_is_bounded_by_stride() {
        let stride = 4 * 62;
        let available = 1_000_000;
        let count = available / stride + 1_000_000;
        let reserve = reserve_count(Format::BinaryLe, 0, stride, count, available);
        assert_eq!(reserve, available / stride);
    }
}
