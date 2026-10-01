//! How a number of the document is shown: in the person's unit, at most two decimals.

use es_math::units::{DEG_TO_RAD, RAD_TO_DEG};

/// How a number of the document is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Metres, shown in centimetres.
    Cm,
    /// Radians, shown in degrees.
    Deg,
    /// Degrees, as the document keeps them (`within_deg`, `tilt_max_deg`).
    DegAsIs,
    CmPerS,
    DegPerS,
    Seconds,
    /// A fraction, shown in per cent.
    Percent,
    /// `tan(a / 2)`, shown as the angle `a` (plan H's resting tilt).
    Tilt,
    Plain,
}

impl Unit {
    /// `v` of the document in the person's unit.
    pub fn shown(self, v: f64) -> f64 {
        match self {
            Self::Cm | Self::CmPerS | Self::Percent => v * 100.0,
            Self::Deg | Self::DegPerS => v * RAD_TO_DEG,
            // Display only: what is written back goes through `es_math::approx`.
            Self::Tilt => 2.0 * v.atan() * RAD_TO_DEG,
            Self::DegAsIs | Self::Seconds | Self::Plain => v,
        }
    }

    /// The document's value of `v` in the person's unit.
    pub fn stored(self, v: f64) -> f64 {
        match self {
            Self::Cm | Self::CmPerS | Self::Percent => v / 100.0,
            Self::Deg | Self::DegPerS => v * DEG_TO_RAD,
            Self::Tilt => es_math::approx::tan_f64(v * DEG_TO_RAD / 2.0),
            Self::DegAsIs | Self::Seconds | Self::Plain => v,
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            Self::Cm => " cm",
            Self::Deg | Self::DegAsIs | Self::Tilt => "\u{b0}",
            Self::CmPerS => " cm/s",
            Self::DegPerS => "\u{b0}/s",
            Self::Seconds => " s",
            Self::Percent => " %",
            Self::Plain => "",
        }
    }
}

/// `v` shown in `unit`, at most two decimals, with its unit.
pub fn show(v: f64, unit: Unit) -> String {
    format!("{}{}", decimals(unit.shown(v)), unit.suffix())
}

fn decimals(v: f64) -> String {
    let s = format!("{v:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0" } else { s }.to_owned()
}

/// A fixed point, in centimetres.
pub fn show_point(p: [f64; 3]) -> String {
    let [x, y, z] = p.map(|v| decimals(Unit::Cm.shown(v)));
    format!("({x}, {y}, {z}) cm")
}
