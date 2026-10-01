//! The weight levels a person picks from: a clause's shaping weight, the success bonus and
//! the strength of the start's draws.

use es_script::spec::vocab::takes;
use es_script::spec::{RewardDoc, Shaping, Start};

use super::{form, Clause, TaskSpec};

/// A weight as the person picks it. A number of the document that is no level is `Custom` and
/// is kept as written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Level {
    Low,
    Medium,
    High,
    Custom(f64),
}

/// The levels a person picks from.
pub const LEVELS: [Level; 3] = [Level::Low, Level::Medium, Level::High];
/// A shaping term's weight at each level, before `[reward] scale`; its sign is the term's (a
/// distance or a ramp costs, an angle term pays). The committed references read as levels:
/// Shadow Hand's rotation `1` and cube distance `−10`, SO-101's ramp `−1`. The owner's to change.
pub const SHAPING: [f64; 3] = [0.1, 1.0, 10.0];
/// The success bonus at each level, before `[reward] scale` (Shadow Hand's `250` is high).
pub const BONUS: [f64; 3] = [25.0, 100.0, 250.0];
/// `[start] strength` at low and high; medium is no `strength` (every range as written).
pub const STRENGTH: [f64; 2] = [0.5, 1.5];

impl Level {
    fn of(v: f64, table: &[f64; 3]) -> Self {
        match table.iter().position(|x| x.to_bits() == v.to_bits()) {
            Some(0) => Self::Low,
            Some(1) => Self::Medium,
            Some(2) => Self::High,
            _ => Self::Custom(v),
        }
    }

    fn value(self, table: &[f64; 3]) -> Option<f64> {
        match self {
            Self::Low => Some(table[0]),
            Self::Medium => Some(table[1]),
            Self::High => Some(table[2]),
            Self::Custom(_) => None,
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::Low => "author.task.level.low",
            Self::Medium => "author.task.level.medium",
            Self::High => "author.task.level.high",
            Self::Custom(_) => "author.task.level.custom",
        }
    }
}

fn sign(s: Shaping) -> f64 {
    if s == Shaping::InverseAngle {
        1.0
    } else {
        -1.0
    }
}

/// The shaping term a clause can pay (design section 4.3), `None` when its relation pays none.
pub fn shapable(c: &Clause) -> Option<Shaping> {
    takes(c.relation, form(c).1).shaping
}

/// The clause's shaping weight as a level; `None` when it pays no shaping term.
pub fn shaping_level(c: &Clause) -> Option<Level> {
    let s = c.shaping?;
    Some(Level::of(c.weight.unwrap_or(0.0) * sign(s), &SHAPING))
}

/// Turns the clause's shaping off (`None`) or on at a level. A ramp starts at the range's low
/// edge and ends a range's width past its high one.
pub fn set_shaping(c: &mut Clause, level: Option<Level>) {
    let Some(form) = shapable(c) else { return };
    let Some(level) = level else {
        (c.shaping, c.weight, c.ramp, c.term) = (None, None, None, None);
        return;
    };
    c.shaping = Some(form);
    match level.value(&SHAPING) {
        Some(w) => c.weight = Some(w * sign(form)),
        None => {
            c.weight.get_or_insert(SHAPING[1] * sign(form));
        }
    }
    if form == Shaping::Ramp && c.ramp.is_none() {
        let [lo, hi] = c.range.unwrap_or_default();
        c.ramp = Some([lo, hi + (hi - lo)]);
    }
}

/// The success bonus as a level; `None` without one.
pub fn bonus(spec: &TaskSpec) -> Option<Level> {
    let v = spec.reward.as_ref()?.success?;
    Some(Level::of(v, &BONUS))
}

pub fn set_bonus(spec: &mut TaskSpec, level: Option<Level>) {
    let r = spec.reward.get_or_insert_with(RewardDoc::default);
    match level {
        None => r.success = None,
        Some(l) => r.success = l.value(&BONUS).or(r.success),
    }
    if spec.reward.as_ref() == Some(&RewardDoc::default()) {
        spec.reward = None;
    }
}

/// How strongly the 🎲 items are drawn.
pub fn strength(spec: &TaskSpec) -> Level {
    match spec.start.as_ref().and_then(|s| s.strength) {
        None => Level::Medium,
        Some(v) if v.to_bits() == STRENGTH[0].to_bits() => Level::Low,
        Some(v) if v.to_bits() == STRENGTH[1].to_bits() => Level::High,
        Some(v) => Level::Custom(v),
    }
}

pub fn set_strength(spec: &mut TaskSpec, level: Level) {
    let v = match level {
        Level::Low => Some(STRENGTH[0]),
        Level::Medium => None,
        Level::High => Some(STRENGTH[1]),
        Level::Custom(_) => return,
    };
    if v.is_some() || spec.start.is_some() {
        spec.start.get_or_insert_with(Start::default).strength = v;
    }
}
