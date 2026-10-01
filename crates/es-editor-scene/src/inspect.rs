//! What the inspector shows of a record, decided here so the drawing only draws (design section
//! 5): a shape's kind and its sizes as a person measures them — whole widths and diameters, which
//! the document stores halved (exact in binary, so a size shown and left alone keeps its bits) —
//! and the values G1 gives a field the document leaves out.

use es_assets::esscene::ShapeDoc;

/// A geom's `friction` when the document writes none: `MuJoCo`'s (G1's `default_geom`).
pub const FRICTION: [f64; 3] = [1.0, 0.005, 0.0001];
/// `rgba` when the document writes none.
pub const RGBA: [f64; 4] = [0.5, 0.5, 0.5, 1.0];
/// `density` (kg/m³) when the document writes none.
pub const DENSITY: f64 = 1000.0;
/// A camera's `fovy` (degrees) when the document writes none.
pub const FOVY: f64 = 45.0;

/// The shapes a person picks from; a mesh comes from a file (G7's import).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeKind {
    Box,
    Sphere,
    Cylinder,
    Capsule,
    Ellipsoid,
    Plane,
    Mesh,
}

impl ShapeKind {
    /// What the kind picker offers.
    pub const PICK: [Self; 6] = [
        Self::Box,
        Self::Sphere,
        Self::Cylinder,
        Self::Capsule,
        Self::Ellipsoid,
        Self::Plane,
    ];

    pub fn of(s: &ShapeDoc) -> Self {
        match s {
            ShapeDoc::Box(_) => Self::Box,
            ShapeDoc::Sphere(_) => Self::Sphere,
            ShapeDoc::Cylinder(_) => Self::Cylinder,
            ShapeDoc::Capsule(_) => Self::Capsule,
            ShapeDoc::Ellipsoid(_) => Self::Ellipsoid,
            ShapeDoc::Plane(_) => Self::Plane,
            ShapeDoc::Mesh(_) => Self::Mesh,
        }
    }

    /// The name a new body of this shape is offered (`box`, then `box_2`, ...).
    pub fn base(self) -> &'static str {
        match self {
            Self::Box => "box",
            Self::Sphere => "sphere",
            Self::Cylinder => "cylinder",
            Self::Capsule => "capsule",
            Self::Ellipsoid => "ellipsoid",
            Self::Plane => "plane",
            Self::Mesh => "mesh",
        }
    }

    /// The key of its word in the editor's string tables.
    pub fn key(self) -> &'static str {
        match self {
            Self::Box => "author.shape.box",
            Self::Sphere => "author.shape.sphere",
            Self::Cylinder => "author.shape.cylinder",
            Self::Capsule => "author.shape.capsule",
            Self::Ellipsoid => "author.shape.ellipsoid",
            Self::Plane => "author.shape.plane",
            Self::Mesh => "author.shape.mesh",
        }
    }
}

const WIDTH: &str = "author.size.width";
const DEPTH: &str = "author.size.depth";
const HEIGHT: &str = "author.size.height";
const DIAMETER: &str = "author.size.diameter";
const LENGTH: &str = "author.size.length";
const GRID: &str = "author.size.grid";

/// The sizes the inspector shows, each with the key of its word: whole extents in metres. A
/// plane's width or depth of 0 is endless; its grid spacing is as written. A mesh has none.
pub fn dims(s: &ShapeDoc) -> Vec<(&'static str, f64)> {
    match s {
        ShapeDoc::Box([x, y, z]) => vec![(WIDTH, 2.0 * x), (DEPTH, 2.0 * y), (HEIGHT, 2.0 * z)],
        ShapeDoc::Ellipsoid([x, y, z]) => {
            vec![(WIDTH, 2.0 * x), (DEPTH, 2.0 * y), (HEIGHT, 2.0 * z)]
        }
        ShapeDoc::Sphere(r) => vec![(DIAMETER, 2.0 * r)],
        ShapeDoc::Capsule([r, h]) | ShapeDoc::Cylinder([r, h]) => {
            vec![(DIAMETER, 2.0 * r), (LENGTH, 2.0 * h)]
        }
        ShapeDoc::Plane([x, y, grid]) => vec![(WIDTH, 2.0 * x), (DEPTH, 2.0 * y), (GRID, *grid)],
        ShapeDoc::Mesh(_) => Vec::new(),
    }
}

/// `s` with the sizes [`dims`] showed set to `v` (as many as it showed).
pub fn with_dims(s: &ShapeDoc, v: &[f64]) -> ShapeDoc {
    let h = |i: usize| v.get(i).copied().unwrap_or_default() / 2.0;
    match s {
        ShapeDoc::Box(_) => ShapeDoc::Box([h(0), h(1), h(2)]),
        ShapeDoc::Ellipsoid(_) => ShapeDoc::Ellipsoid([h(0), h(1), h(2)]),
        ShapeDoc::Sphere(_) => ShapeDoc::Sphere(h(0)),
        ShapeDoc::Capsule(_) => ShapeDoc::Capsule([h(0), h(1)]),
        ShapeDoc::Cylinder(_) => ShapeDoc::Cylinder([h(0), h(1)]),
        ShapeDoc::Plane(_) => ShapeDoc::Plane([h(0), h(1), v.get(2).copied().unwrap_or_default()]),
        ShapeDoc::Mesh(f) => ShapeDoc::Mesh(f.clone()),
    }
}

/// `s` as a `kind`, about as big: the largest half-extent becomes the radius, and so on.
pub fn reshape(s: &ShapeDoc, kind: ShapeKind) -> ShapeDoc {
    if ShapeKind::of(s) == kind {
        return s.clone();
    }
    let r = match s {
        ShapeDoc::Box(v) | ShapeDoc::Ellipsoid(v) => v[0].max(v[1]).max(v[2]),
        ShapeDoc::Sphere(r) => *r,
        ShapeDoc::Capsule([r, h]) | ShapeDoc::Cylinder([r, h]) => r.max(*h),
        ShapeDoc::Plane([x, y, _]) => x.max(*y),
        ShapeDoc::Mesh(_) => 0.05,
    };
    let r = if r > 0.0 { r } else { 0.05 };
    match kind {
        ShapeKind::Box => ShapeDoc::Box([r; 3]),
        ShapeKind::Ellipsoid => ShapeDoc::Ellipsoid([r; 3]),
        ShapeKind::Sphere => ShapeDoc::Sphere(r),
        ShapeKind::Capsule => ShapeDoc::Capsule([r / 2.0, r]),
        ShapeKind::Cylinder => ShapeDoc::Cylinder([r, r]),
        ShapeKind::Plane => ShapeDoc::Plane([r, r, 0.05]),
        ShapeKind::Mesh => s.clone(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use std::path::Path;

    use es_assets::esscene::{expand, EsScene};

    use super::*;

    /// The defaults shown are the ones G1 expands a bare geom and camera to.
    #[test]
    fn the_shown_defaults_are_the_expanded_ones() {
        let doc = EsScene::from_toml(
            "kind = \"scene\"\nschema = 1\n[[geom]]\nshape = { sphere = 0.1 }\n\
             [[camera]]\nname = \"c\"\n",
        )
        .unwrap();
        let s = expand(&doc, Path::new(".")).unwrap();
        let g = &s.bodies[0].geoms[0];
        assert_eq!((g.friction, g.rgba, g.density), (FRICTION, RGBA, DENSITY));
        assert!((s.cameras[0].fovy.to_degrees() - FOVY).abs() < 1e-12);
    }

    /// Whole sizes in, halves stored, bit for bit both ways; a kind change keeps the size.
    #[test]
    fn sizes_are_whole_and_round_trip() {
        let cube = ShapeDoc::Box([0.03, 0.03, 0.03]);
        let shown: Vec<f64> = dims(&cube).iter().map(|d| d.1).collect();
        assert_eq!(shown, [0.06; 3]);
        assert_eq!(with_dims(&cube, &shown), cube);
        assert_eq!(
            with_dims(&cube, &[0.1, 0.06, 0.06]),
            ShapeDoc::Box([0.05, 0.03, 0.03])
        );
        assert_eq!(reshape(&cube, ShapeKind::Sphere), ShapeDoc::Sphere(0.03));
        assert_eq!(reshape(&cube, ShapeKind::Box), cube);
        for kind in ShapeKind::PICK {
            let s = reshape(&cube, kind);
            assert_eq!(ShapeKind::of(&s), kind);
            assert!(dims(&s).iter().all(|d| d.1 > 0.0), "{s:?}");
        }
    }
}
