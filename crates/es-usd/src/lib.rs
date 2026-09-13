//! `es-usd` (layer 2): a native reader for the subset of the `OpenUSD` `.usda` text format that
//! a rigid-body scene needs.
//!
//! Spec 28.6 puts "USD native" in M4; spec 1.9 item 6 puts it sixth on the cut list, to be
//! replaced by the Python USD Bake of spec 2.5. So this crate is deliberately a *subset*
//! reader: it parses what it documents in `docs/api-notes/usd.md` and refuses everything else
//! with [`UsdError::Unsupported`] naming the prim, rather than silently producing a scene that
//! is missing whatever the file composed in.
//!
//! Layer rule (spec 4.2): this crate depends on `es-core`, `es-math` and external crates only.
//! In particular it must *not* depend on `es-assets`, which is the same layer — which is why
//! nothing here knows about `SceneDesc`. The mapping from a [`UsdStage`] onto a scene lives one
//! layer up, in `es_physics_core::usd`.
//!
//! Nothing here reads a file from disk: [`parse_usda`] takes the text.
//!
//! ```
//! let stage = es_usd::parse_usda("#usda 1.0\ndef Xform \"root\"\n{\n}\n").unwrap();
//! assert_eq!(stage.prims[0].path, "/root");
//! ```
#![forbid(unsafe_code)]

mod parse;

use std::collections::BTreeMap;

use es_math::{Pose, Quat, Vec3};
use thiserror::Error;

pub use parse::parse_usda;

/// Why a `.usda` text could not be turned into a [`UsdStage`].
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum UsdError {
    /// A grammar violation, always with the source line.
    #[error("line {line}: {message}")]
    Syntax { line: usize, message: String },
    /// A real USD feature this reader refuses on purpose. `path` is the prim path, or `/` for
    /// a layer-level feature. Route: flatten the file with USD Bake (spec 2.5).
    #[error("{path}: unsupported USD feature `{feature}` (line {line}); flatten with USD Bake")]
    Unsupported {
        path: String,
        feature: String,
        line: usize,
    },
}

/// Which way is up in the source file (spec 3.1 is Z-up; USD's own fallback is Y).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpAxis {
    #[default]
    Y,
    Z,
}

/// Layer metadata. Defaults are USD's own fallbacks; `*_authored` records whether the file
/// actually said so, because guessing a length unit wrong is a 100x error.
#[derive(Clone, Debug, PartialEq)]
pub struct StageMeta {
    pub up_axis: UpAxis,
    pub meters_per_unit: f64,
    pub default_prim: Option<String>,
    pub up_axis_authored: bool,
    pub meters_per_unit_authored: bool,
}

impl Default for StageMeta {
    fn default() -> Self {
        Self {
            up_axis: UpAxis::Y,
            meters_per_unit: 0.01,
            default_prim: None,
            up_axis_authored: false,
            meters_per_unit_authored: false,
        }
    }
}

/// A prim's specifier (glossary: `def`, `over`, `class`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Specifier {
    Def,
    Over,
    Class,
}

/// One prim, as written. Attributes this reader does not understand are kept in `attrs` and
/// an unknown `type_name` is kept with a warning, so widening the mapping never needs a
/// parser change.
#[derive(Clone, Debug, PartialEq)]
pub struct Prim {
    /// Absolute prim path, e.g. `/World/arm/link1`.
    pub path: String,
    /// Last path segment.
    pub name: String,
    pub specifier: Specifier,
    /// Schema type name (`Xform`, `Mesh`, `PhysicsRevoluteJoint`, ...); empty when the file
    /// wrote `def "name"` with no type.
    pub type_name: String,
    pub attrs: BTreeMap<String, Value>,
    /// Relationship targets, e.g. `physics:body0 -> ["/World/base"]`.
    pub rels: BTreeMap<String, Vec<String>>,
    /// `apiSchemas` from the prim metadata block, verbatim (including multi-apply instance
    /// names such as `PhysicsDriveAPI:angular`).
    pub api_schemas: Vec<String>,
    pub children: Vec<Prim>,
    /// Line the prim was declared on.
    pub line: usize,
}

impl Prim {
    /// Does `apiSchemas` list `name`, either bare or as `name:instance`?
    pub fn has_api(&self, name: &str) -> bool {
        self.api_schemas
            .iter()
            .any(|s| s == name || s.split(':').next() == Some(name))
    }

    /// Depth-first preorder over this prim and its descendants.
    pub fn walk(&self, visit: &mut impl FnMut(&Prim)) {
        visit(self);
        for child in &self.children {
            child.walk(visit);
        }
    }

    /// The prim's own transform, from `xformOpOrder`, in the file's own units and axes.
    ///
    /// `xformOpOrder` is authoritative: an `xformOp:*` attribute the order does not name
    /// contributes nothing (matching `UsdGeomXformable::GetOrderedXformOps`). The first op
    /// listed is the outermost, so with column vectors `M = op[0] * ... * op[n-1]`.
    /// Scale is not representable in a rigid [`Pose`] and is dropped; every drop is reported
    /// through `warnings`.
    pub fn local_xform(&self, warnings: &mut Vec<String>) -> Pose {
        let Some(Value::Array(order)) = self.attrs.get("xformOpOrder") else {
            if self.attrs.keys().any(|k| k.starts_with("xformOp:")) {
                warnings.push(format!(
                    "{}: xformOp attributes are authored but xformOpOrder is not; USD applies none of them",
                    self.path
                ));
            }
            return Pose::IDENTITY;
        };
        let names: Vec<&str> = order.iter().filter_map(Value::as_str).collect();
        // Everything up to the last `!resetXformStack!` is ignored (UsdGeomXformable).
        let start = names
            .iter()
            .rposition(|n| *n == "!resetXformStack!")
            .map_or(0, |i| i + 1);

        let mut pose = Pose::IDENTITY;
        for name in &names[start..] {
            let Some(value) = self.attrs.get(*name) else {
                warnings.push(format!(
                    "{}: xformOpOrder names `{name}` but no such attribute is authored",
                    self.path
                ));
                continue;
            };
            let op = match op_kind(name) {
                Some("translate") => value
                    .as_vec3()
                    .map(|v| Pose::new(v, Quat::IDENTITY))
                    .ok_or("translate is not a 3-vector"),
                Some("orient") => value
                    .as_quat()
                    .map(|q| Pose::new(Vec3::ZERO, q))
                    .ok_or("orient is not a quaternion"),
                Some("transform") => value
                    .as_matrix4d()
                    .map(pose_from_row_major)
                    .ok_or("transform is not a matrix4d"),
                Some("scale") => {
                    if value.as_vec3().is_some_and(is_unit_scale) {
                        Ok(Pose::IDENTITY)
                    } else {
                        Err("a rigid Pose carries no scale; the scale is dropped")
                    }
                }
                _ => Err("unsupported xformOp kind; the op is dropped"),
            };
            match op {
                Ok(op) => pose = pose.compose(op),
                Err(why) => warnings.push(format!("{}: {name}: {why}", self.path)),
            }
        }
        pose
    }
}

/// `xformOp:translate:pivot` -> `translate`.
fn op_kind(name: &str) -> Option<&str> {
    name.strip_prefix("xformOp:")?.split(':').next()
}

fn is_unit_scale(v: Vec3) -> bool {
    v == Vec3::new(1.0, 1.0, 1.0)
}

/// USD matrices are row-vector / row-major: translation sits in the last row.
fn pose_from_row_major(m: [[f64; 4]; 4]) -> Pose {
    let position = Vec3::new(m[3][0], m[3][1], m[3][2]);
    // Rows of a row-vector matrix are the images of the basis vectors, i.e. the columns of
    // the equivalent column-vector rotation.
    let cols = [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ];
    Pose::new(position, quat_from_columns(cols))
}

/// Shepperd's method: rotation matrix (as basis-image columns) to quaternion.
///
/// Local to this crate because `es-math` keeps its own copy private, and a layer-2 crate may
/// not reach into another crate's internals to avoid twenty lines.
#[allow(clippy::many_single_char_names)]
fn quat_from_columns(c: [[f64; 3]; 3]) -> Quat {
    let (m00, m11, m22) = (c[0][0], c[1][1], c[2][2]);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (c[1][2] - c[2][1]) / s,
            (c[2][0] - c[0][2]) / s,
            (c[0][1] - c[1][0]) / s,
            0.25 * s,
        ]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [
            0.25 * s,
            (c[1][0] + c[0][1]) / s,
            (c[2][0] + c[0][2]) / s,
            (c[1][2] - c[2][1]) / s,
        ]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [
            (c[1][0] + c[0][1]) / s,
            0.25 * s,
            (c[2][1] + c[1][2]) / s,
            (c[2][0] - c[0][2]) / s,
        ]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [
            (c[2][0] + c[0][2]) / s,
            (c[2][1] + c[1][2]) / s,
            0.25 * s,
            (c[0][1] - c[1][0]) / s,
        ]
    };
    Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize()
}

/// A parsed layer: metadata plus the root prims, in document order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UsdStage {
    pub meta: StageMeta,
    pub prims: Vec<Prim>,
    /// Everything the reader kept but did not understand.
    pub warnings: Vec<String>,
}

impl UsdStage {
    /// Depth-first preorder over every prim in the stage.
    pub fn walk(&self, visit: &mut impl FnMut(&Prim)) {
        for prim in &self.prims {
            prim.walk(visit);
        }
    }

    /// The prim at an absolute path, e.g. `/World/arm`.
    pub fn find(&self, path: &str) -> Option<&Prim> {
        let mut segments = path.trim_start_matches('/').split('/');
        let first = segments.next()?;
        let mut prim = self.prims.iter().find(|p| p.name == first)?;
        for segment in segments {
            prim = prim.children.iter().find(|p| p.name == segment)?;
        }
        Some(prim)
    }

    /// World transform of `path`: the composition of every ancestor's [`Prim::local_xform`],
    /// **in the file's own units and up-axis**. Converting to spec 3.1 is the caller's job
    /// (`es_physics_core::usd` does it once, for the whole stage).
    ///
    /// An unknown path has no transform, so it gets the identity.
    pub fn resolve_xform(&self, path: &str) -> Pose {
        let mut warnings = Vec::new();
        self.resolve_xform_reporting(path, &mut warnings)
    }

    /// [`UsdStage::resolve_xform`], collecting the warnings the composition produced.
    pub fn resolve_xform_reporting(&self, path: &str, warnings: &mut Vec<String>) -> Pose {
        let mut pose = Pose::IDENTITY;
        let mut prefix = String::new();
        for segment in path.trim_start_matches('/').split('/') {
            prefix.push('/');
            prefix.push_str(segment);
            let Some(prim) = self.find(&prefix) else {
                return pose;
            };
            pose = pose.compose(prim.local_xform(warnings));
        }
        pose
    }
}

/// An attribute value, typed by the attribute's *declared* type name rather than guessed from
/// the literal — `float 1` and `int 1` stay distinguishable.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Double(f64),
    Token(String),
    String(String),
    /// A three-component vector: `float3`, `double3`, `point3f`, `vector3f`, `color3f`, ...
    Float3([f64; 3]),
    /// A `quatf`/`quatd`. The file writes `(w, x, y, z)` (`GfQuatf(real, i, j, k)`); this is
    /// already converted to the spec 3.1 `xyzw` storage order.
    Quat(Quat),
    /// Row-major, row-vector, as USD writes it.
    Matrix4d([[f64; 4]; 4]),
    /// A numeric tuple of some other arity, kept so unknown attributes survive.
    Tuple(Vec<f64>),
    Array(Vec<Value>),
    /// A relationship target or a path-valued attribute.
    Rel(String),
}

impl Value {
    /// Numeric value of a scalar, whatever its declared width.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::Int(v) => Some(v as f64),
            Value::Float(v) | Value::Double(v) => Some(v),
            Value::Bool(v) => Some(f64::from(u8::from(v))),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// Text of a `token`, `string` or asset path.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Token(s) | Value::String(s) | Value::Rel(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_vec3(&self) -> Option<Vec3> {
        match self {
            Value::Float3([x, y, z]) => Some(Vec3::new(*x, *y, *z)),
            _ => None,
        }
    }

    pub fn as_quat(&self) -> Option<Quat> {
        match self {
            Value::Quat(q) => Some(*q),
            _ => None,
        }
    }

    pub fn as_matrix4d(&self) -> Option<[[f64; 4]; 4]> {
        match self {
            Value::Matrix4d(m) => Some(*m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Every element of an array as a number; `None` if any element is not numeric.
    pub fn as_f64_array(&self) -> Option<Vec<f64>> {
        self.as_array()?.iter().map(Value::as_f64).collect()
    }

    /// Every element of an array as a 3-vector.
    pub fn as_vec3_array(&self) -> Option<Vec<Vec3>> {
        self.as_array()?.iter().map(Value::as_vec3).collect()
    }
}
