//! The type system shared by the five IRs (spec 5.4, Appendix B.1).
//!
//! See `docs/design/ir-types.md` for the unit table, the opaque units and the time-join rule.

use es_core::StableId;
use serde::{Deserialize, Serialize};

use crate::canon::CanonWriter;
use crate::codes;
use crate::diag::Diagnostic;
use crate::image::ImageSpec;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ElemType {
    F32,
    F16,
    Bf16,
    F64,
    I32,
    U8,
    Bool,
}

/// Per-sample dimensions. The batch axis is **not** here: each batch domain picks its own size
/// (spec 5.2), so a shape that pinned one would be wrong in the other three.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Shape(pub Vec<u64>);

impl Shape {
    pub fn new(dims: impl Into<Vec<u64>>) -> Self {
        Self(dims.into())
    }

    pub fn dims(&self) -> &[u64] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Element count of one sample; saturates rather than overflowing.
    pub fn elem_count(&self) -> u64 {
        self.0.iter().fold(1, |a, d| a.saturating_mul(*d))
    }
}

/// Exponents of the base units: `m^m · kg^kg · s^s · rad^rad`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct UnitPowers {
    pub m: i8,
    pub kg: i8,
    pub s: i8,
    pub rad: i8,
}

impl UnitPowers {
    pub const fn new(m: i8, kg: i8, s: i8, rad: i8) -> Self {
        Self { m, kg, s, rad }
    }

    fn combine(self, other: Self, sign: i8) -> Self {
        Self {
            m: self.m + sign * other.m,
            kg: self.kg + sign * other.kg,
            s: self.s + sign * other.s,
            rad: self.rad + sign * other.rad,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Unit {
    Dimensionless,
    Length,
    Angle,
    Mass,
    Time,
    Velocity,
    AngularVelocity,
    Acceleration,
    Force,
    Torque,
    Pressure,
    Current,
    Voltage,
    Quaternion,
    RotationMatrix,
    Normalized { lo: f64, hi: f64 },
    Pixel,
    Luminance,
    Depth,
    Token,
    Composite(UnitPowers),
}

/// The bidirectional name ↔ exponent table. `Dimensionless` is first so the all-zero vector
/// never comes back as `Composite`. Units absent from the table are opaque (design note).
const NAMED: &[(Unit, UnitPowers)] = &[
    (Unit::Dimensionless, UnitPowers::new(0, 0, 0, 0)),
    (Unit::Length, UnitPowers::new(1, 0, 0, 0)),
    (Unit::Mass, UnitPowers::new(0, 1, 0, 0)),
    (Unit::Time, UnitPowers::new(0, 0, 1, 0)),
    (Unit::Angle, UnitPowers::new(0, 0, 0, 1)),
    (Unit::Velocity, UnitPowers::new(1, 0, -1, 0)),
    (Unit::Acceleration, UnitPowers::new(1, 0, -2, 0)),
    (Unit::AngularVelocity, UnitPowers::new(0, 0, -1, 1)),
    (Unit::Force, UnitPowers::new(1, 1, -2, 0)),
    (Unit::Torque, UnitPowers::new(2, 1, -2, 0)),
    (Unit::Pressure, UnitPowers::new(-1, 1, -2, 0)),
];

impl Unit {
    /// The exponent vector, or `None` for an opaque unit.
    pub fn powers(&self) -> Option<UnitPowers> {
        match self {
            Self::Composite(p) => Some(*p),
            _ => NAMED.iter().find(|(u, _)| u == self).map(|(_, p)| *p),
        }
    }

    /// The named unit for an exponent vector, or `Composite`.
    pub fn from_powers(p: UnitPowers) -> Self {
        NAMED
            .iter()
            .find(|(_, q)| *q == p)
            .map_or(Self::Composite(p), |(u, _)| u.clone())
    }

    fn require_powers(&self) -> Result<UnitPowers, Diagnostic> {
        self.powers().ok_or_else(|| {
            Diagnostic::new(
                codes::TYPE_010,
                format!("{self:?} has no dimensional exponents"),
            )
            .with_hint("convert it with an explicit node before doing unit algebra")
        })
    }

    pub fn mul(&self, other: &Self) -> Result<Self, Diagnostic> {
        let (a, b) = (self.require_powers()?, other.require_powers()?);
        Ok(Self::from_powers(a.combine(b, 1)))
    }

    pub fn div(&self, other: &Self) -> Result<Self, Diagnostic> {
        let (a, b) = (self.require_powers()?, other.require_powers()?);
        Ok(Self::from_powers(a.combine(b, -1)))
    }

    /// Addition needs the units to be equal as *values*, not merely equal in dimension:
    /// `Depth + Length` is both metres and a bug.
    pub fn checked_add(&self, other: &Self) -> Result<Self, Diagnostic> {
        if self == other {
            Ok(self.clone())
        } else {
            Err(Diagnostic::new(
                codes::TYPE_003,
                format!("cannot add {self:?} and {other:?}"),
            ))
        }
    }

    /// A policy input port takes only these (spec 5.4).
    pub fn is_policy_input(&self) -> bool {
        matches!(
            self,
            Self::Normalized { .. } | Self::Dimensionless | Self::Token
        )
    }

    fn canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::Normalized { lo, hi } => {
                w.str("Normalized");
                w.f64(*lo);
                w.f64(*hi);
            }
            Self::Composite(p) => {
                w.str("Composite");
                for e in [p.m, p.kg, p.s, p.rad] {
                    w.i64(i64::from(e));
                }
            }
            other => w.str(&format!("{other:?}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Frame {
    World,
    LocalOrigin,
    Body(StableId),
    Sensor(StableId),
    Joint(StableId),
    /// Camera optical frame, `OpenCV` convention (spec 3.1).
    Camera(StableId),
    /// Pixel coordinates, origin top-left (spec 3.1).
    Image(StableId),
    /// Inside the network. No frame checking (spec 5.4).
    Policy,
}

impl Frame {
    pub fn compatible(&self, other: &Self) -> bool {
        self == other || matches!(self, Self::Policy) || matches!(other, Self::Policy)
    }

    fn canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::World => w.str("World"),
            Self::LocalOrigin => w.str("LocalOrigin"),
            Self::Policy => w.str("Policy"),
            Self::Body(id) => tagged_id(w, "Body", id),
            Self::Sensor(id) => tagged_id(w, "Sensor", id),
            Self::Joint(id) => tagged_id(w, "Joint", id),
            Self::Camera(id) => tagged_id(w, "Camera", id),
            Self::Image(id) => tagged_id(w, "Image", id),
        }
    }
}

fn tagged_id(w: &mut CanonWriter, tag: &str, id: &StableId) {
    w.str(tag);
    w.bytes(id.as_bytes());
}

/// How a value on one clock is brought onto another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Align {
    /// Zero-order hold — the usual choice for ACT and Diffusion Policy (spec 5.4).
    Hold,
    Interpolate,
    /// Do not combine across this clock.
    Reject,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeRef {
    /// The current physics tick.
    Tick,
    Sensor {
        id: StableId,
        align: Align,
    },
    /// A `TemporalWindow` (spec 7.5).
    Window {
        base: Box<TimeRef>,
        n: u32,
        stride: u32,
    },
}

impl TimeRef {
    /// The clock a combination of two inputs runs on. An unspecified alignment is `TYPE-014`
    /// (spec 5.4) — see the table in `docs/design/ir-types.md`.
    pub fn join(&self, other: &Self) -> Result<Self, Diagnostic> {
        let unaligned = || {
            Diagnostic::new(
                codes::TYPE_014,
                format!("inputs are on different clocks: {self:?} and {other:?}"),
            )
            .with_hint("time_align = \"hold\" | \"interpolate\" | \"reject\" (ACT and Diffusion Policy usually use \"hold\")")
        };
        if self == other {
            return Ok(self.clone());
        }
        match (self, other) {
            (Self::Tick, Self::Sensor { align, .. }) | (Self::Sensor { align, .. }, Self::Tick)
                if *align != Align::Reject =>
            {
                Ok(Self::Tick)
            }
            (Self::Sensor { align: a, .. }, Self::Sensor { align: b, .. })
                if a == b && *a != Align::Reject =>
            {
                Ok(Self::Tick)
            }
            _ => Err(unaligned()),
        }
    }

    fn canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::Tick => w.str("Tick"),
            Self::Sensor { id, align } => {
                tagged_id(w, "Sensor", id);
                w.str(&format!("{align:?}"));
            }
            Self::Window { base, n, stride } => {
                w.str("Window");
                base.canonical(w);
                w.u32(*n);
                w.u32(*stride);
            }
        }
    }
}

/// `(ElemType, Shape, Unit × Frame × TimeRef, ImageSpec?)` — spec 5.4, Appendix B.1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PortType {
    pub elem: ElemType,
    pub shape: Shape,
    pub unit: Unit,
    pub frame: Frame,
    pub time: TimeRef,
    /// Image ports only.
    pub image: Option<ImageSpec>,
}

impl PortType {
    /// Can a value of type `self` flow into a port of type `other`?
    pub fn compatible(&self, other: &Self) -> Result<(), Diagnostic> {
        if self.elem != other.elem {
            return Err(Diagnostic::new(
                codes::TYPE_001,
                format!("{:?} cannot feed {:?}", self.elem, other.elem),
            ));
        }
        if self.shape != other.shape {
            return Err(Diagnostic::new(
                codes::TYPE_002,
                format!(
                    "{:?} cannot feed {:?}",
                    self.shape.dims(),
                    other.shape.dims()
                ),
            ));
        }
        if self.unit != other.unit {
            return Err(Diagnostic::new(
                codes::TYPE_003,
                format!("{:?} cannot feed {:?}", self.unit, other.unit),
            ));
        }
        if !self.frame.compatible(&other.frame) {
            return Err(Diagnostic::new(
                codes::TYPE_004,
                format!("{:?} cannot feed {:?}", self.frame, other.frame),
            ));
        }
        if self.time != other.time {
            return Err(Diagnostic::new(
                codes::TYPE_014,
                format!("{:?} cannot feed {:?}", self.time, other.time),
            )
            .with_hint("combine the two clocks through an explicit time_align"));
        }
        if self.image != other.image {
            return Err(Diagnostic::new(
                codes::TYPE_020,
                "the two ports carry different ImageSpecs",
            ));
        }
        Ok(())
    }

    /// Rejects a port that a network must not be fed raw (spec 5.4).
    pub fn check_policy_input(&self) -> Result<(), Diagnostic> {
        if self.unit.is_policy_input() {
            Ok(())
        } else {
            Err(Diagnostic::new(
                codes::TYPE_011,
                format!("policy input carries {:?}", self.unit),
            )
            .with_hint("insert a Normalize node, or declare the port Dimensionless"))
        }
    }

    /// Canonical bytes, for node params that carry a type.
    pub fn canonical(&self, w: &mut CanonWriter) {
        w.str(&format!("{:?}", self.elem));
        w.seq(self.shape.rank());
        for d in self.shape.dims() {
            w.u64(*d);
        }
        self.unit.canonical(w);
        self.frame.canonical(w);
        self.time.canonical(w);
        match &self.image {
            None => w.bool(false),
            Some(spec) => {
                w.bool(true);
                spec.canonical(w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(path: &str) -> StableId {
        StableId::from_path(path)
    }

    fn dimensional() -> Vec<Unit> {
        NAMED.iter().map(|(u, _)| u.clone()).collect()
    }

    fn opaque() -> Vec<Unit> {
        vec![
            Unit::Current,
            Unit::Voltage,
            Unit::Quaternion,
            Unit::RotationMatrix,
            Unit::Normalized { lo: -1.0, hi: 1.0 },
            Unit::Pixel,
            Unit::Luminance,
            Unit::Depth,
            Unit::Token,
        ]
    }

    #[test]
    fn named_units_round_trip_through_powers() {
        for u in dimensional() {
            assert_eq!(Unit::from_powers(u.powers().unwrap()), u, "{u:?}");
        }
    }

    #[test]
    fn unit_algebra_names_the_result() {
        assert_eq!(Unit::Length.div(&Unit::Time).unwrap(), Unit::Velocity);
        assert_eq!(Unit::Velocity.mul(&Unit::Time).unwrap(), Unit::Length);
        assert_eq!(Unit::Velocity.div(&Unit::Time).unwrap(), Unit::Acceleration);
        assert_eq!(Unit::Mass.mul(&Unit::Acceleration).unwrap(), Unit::Force);
        assert_eq!(Unit::Force.mul(&Unit::Length).unwrap(), Unit::Torque);
        // Pressure is force per *area*, so two divisions by Length.
        assert_eq!(
            Unit::Force
                .div(&Unit::Length)
                .unwrap()
                .div(&Unit::Length)
                .unwrap(),
            Unit::Pressure
        );
        assert_eq!(Unit::Angle.div(&Unit::Time).unwrap(), Unit::AngularVelocity);
        // No name for kg·m³: Composite, and it still divides back.
        let odd = Unit::Torque.mul(&Unit::Length).unwrap();
        assert_eq!(odd, Unit::Composite(UnitPowers::new(3, 1, -2, 0)));
        assert_eq!(odd.div(&Unit::Length).unwrap(), Unit::Torque);
    }

    #[test]
    fn unit_algebra_laws_hold_for_dimensional_units() {
        for a in dimensional() {
            for b in dimensional() {
                assert_eq!(a.mul(&b).unwrap(), b.mul(&a).unwrap(), "{a:?} {b:?}");
                assert_eq!(a.mul(&b).unwrap().div(&b).unwrap(), a, "{a:?} {b:?}");
                assert_eq!(a.mul(&Unit::Dimensionless).unwrap(), a, "{a:?}");
            }
        }
    }

    #[test]
    fn opaque_units_reject_algebra() {
        for u in opaque() {
            assert_eq!(
                u.mul(&Unit::Length).unwrap_err().code.as_str(),
                codes::TYPE_010,
                "{u:?}"
            );
            assert_eq!(
                Unit::Length.div(&u).unwrap_err().code.as_str(),
                codes::TYPE_010,
                "{u:?}"
            );
            assert!(u.powers().is_none(), "{u:?}");
        }
    }

    #[test]
    fn addition_is_exact_equality() {
        assert_eq!(
            Unit::Length.checked_add(&Unit::Length).unwrap(),
            Unit::Length
        );
        // Both are metres, but a depth image is not a length.
        assert_eq!(
            Unit::Depth
                .checked_add(&Unit::Length)
                .unwrap_err()
                .code
                .as_str(),
            codes::TYPE_003
        );
        let a = Unit::Normalized { lo: 0.0, hi: 1.0 };
        let b = Unit::Normalized { lo: -1.0, hi: 1.0 };
        assert!(a.checked_add(&b).is_err());
        assert!(a.checked_add(&a).is_ok());
    }

    #[test]
    fn composite_of_zero_powers_is_dimensionless() {
        assert_eq!(
            Unit::from_powers(UnitPowers::new(0, 0, 0, 0)),
            Unit::Dimensionless
        );
        assert_eq!(
            Unit::Length.div(&Unit::Length).unwrap(),
            Unit::Dimensionless
        );
    }

    #[test]
    fn policy_inputs_are_restricted() {
        for u in [
            Unit::Normalized { lo: 0.0, hi: 1.0 },
            Unit::Dimensionless,
            Unit::Token,
        ] {
            assert!(port(u.clone()).check_policy_input().is_ok(), "{u:?}");
        }
        for u in [Unit::Length, Unit::Angle, Unit::Pixel, Unit::Depth] {
            let err = port(u.clone()).check_policy_input().unwrap_err();
            assert_eq!(err.code.as_str(), codes::TYPE_011, "{u:?}");
        }
    }

    fn port(unit: Unit) -> PortType {
        PortType {
            elem: ElemType::F32,
            shape: Shape::new([7]),
            unit,
            frame: Frame::World,
            time: TimeRef::Tick,
            image: None,
        }
    }

    #[test]
    fn compatible_reports_one_code_per_class() {
        let base = port(Unit::Length);
        assert!(base.compatible(&base).is_ok());

        let mut other = base.clone();
        other.elem = ElemType::F16;
        assert_eq!(
            base.compatible(&other).unwrap_err().code.as_str(),
            codes::TYPE_001
        );

        let mut other = base.clone();
        other.shape = Shape::new([8]);
        assert_eq!(
            base.compatible(&other).unwrap_err().code.as_str(),
            codes::TYPE_002
        );

        let mut other = base.clone();
        other.unit = Unit::Force;
        assert_eq!(
            base.compatible(&other).unwrap_err().code.as_str(),
            codes::TYPE_003
        );

        let mut other = base.clone();
        other.frame = Frame::Body(id("robot/arm"));
        assert_eq!(
            base.compatible(&other).unwrap_err().code.as_str(),
            codes::TYPE_004
        );

        let mut other = base.clone();
        other.time = TimeRef::Sensor {
            id: id("cam_front"),
            align: Align::Hold,
        };
        assert_eq!(
            base.compatible(&other).unwrap_err().code.as_str(),
            codes::TYPE_014
        );
    }

    #[test]
    fn policy_frame_skips_frame_checking() {
        let mut a = port(Unit::Length);
        a.frame = Frame::Camera(id("cam_front"));
        let mut b = a.clone();
        b.frame = Frame::Policy;
        assert!(a.compatible(&b).is_ok());
        assert!(b.compatible(&a).is_ok());
    }

    #[test]
    fn time_join_follows_the_table() {
        let enc = TimeRef::Sensor {
            id: id("encoder"),
            align: Align::Hold,
        };
        let cam = TimeRef::Sensor {
            id: id("cam_front"),
            align: Align::Hold,
        };
        let cam_interp = TimeRef::Sensor {
            id: id("cam_front"),
            align: Align::Interpolate,
        };
        let cam_reject = TimeRef::Sensor {
            id: id("cam_front"),
            align: Align::Reject,
        };

        assert_eq!(enc.join(&enc).unwrap(), enc);
        assert_eq!(TimeRef::Tick.join(&enc).unwrap(), TimeRef::Tick);
        assert_eq!(enc.join(&TimeRef::Tick).unwrap(), TimeRef::Tick);
        assert_eq!(enc.join(&cam).unwrap(), TimeRef::Tick);

        // Different alignments, or an explicit Reject: TYPE-014.
        for (a, b) in [
            (&enc, &cam_interp),
            (&enc, &cam_reject),
            (&TimeRef::Tick, &cam_reject),
        ] {
            assert_eq!(a.join(b).unwrap_err().code.as_str(), codes::TYPE_014);
        }
    }

    #[test]
    fn windows_join_only_with_themselves() {
        let w = |n| TimeRef::Window {
            base: Box::new(TimeRef::Tick),
            n,
            stride: 1,
        };
        assert_eq!(w(2).join(&w(2)).unwrap(), w(2));
        assert_eq!(w(2).join(&w(3)).unwrap_err().code.as_str(), codes::TYPE_014);
        assert_eq!(
            w(2).join(&TimeRef::Tick).unwrap_err().code.as_str(),
            codes::TYPE_014
        );
    }

    #[test]
    fn canonical_encoding_separates_types() {
        let encode = |p: &PortType| {
            let mut w = CanonWriter::new();
            p.canonical(&mut w);
            w.finish().unwrap()
        };
        let base = port(Unit::Length);
        assert_eq!(encode(&base), encode(&base.clone()));
        assert_ne!(encode(&base), encode(&port(Unit::Force)));
        assert_ne!(
            encode(&port(Unit::Normalized { lo: 0.0, hi: 1.0 })),
            encode(&port(Unit::Normalized { lo: -1.0, hi: 1.0 }))
        );
        assert_ne!(
            encode(&port(Unit::Composite(UnitPowers::new(1, 0, 0, 0)))),
            encode(&port(Unit::Composite(UnitPowers::new(0, 1, 0, 0))))
        );
        let mut other = base.clone();
        other.frame = Frame::Body(id("a"));
        let mut third = base.clone();
        third.frame = Frame::Sensor(id("a"));
        assert_ne!(encode(&other), encode(&third));
    }

    #[test]
    fn serde_round_trip() {
        let mut p = port(Unit::Normalized { lo: -1.0, hi: 1.0 });
        p.frame = Frame::Camera(id("cam_front"));
        p.time = TimeRef::Window {
            base: Box::new(TimeRef::Sensor {
                id: id("cam_front"),
                align: Align::Hold,
            }),
            n: 2,
            stride: 1,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<PortType>(&json).unwrap(), p);
    }
}
