//! Violation and fallback statistics (spec 10.3).
//!
//! `envelope_violation_rate` and `chunk_underrun_rate` are two of the three metrics spec 10.3
//! marks as unavailable on other platforms, so they are first-class here rather than derived
//! by a caller. Everything is an integer ratio: nothing accumulates in `f64` (spec 3.4).

use crate::types::ViolationKind;

/// Maximum `EnvelopeViolationRate` window the pre-allocated ring can hold.
pub const WINDOW_CAP: usize = 256;

/// A fixed-capacity ring of "was this step a violation" bits with a running ones count.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViolationWindow {
    bits: [u64; WINDOW_CAP / 64],
    len: usize,
    head: usize,
    filled: usize,
    ones: usize,
}

impl ViolationWindow {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            bits: [0; WINDOW_CAP / 64],
            len: len.clamp(1, WINDOW_CAP),
            head: 0,
            filled: 0,
            ones: 0,
        }
    }

    fn get(&self, i: usize) -> bool {
        self.bits[i / 64] & (1 << (i % 64)) != 0
    }

    fn set(&mut self, i: usize, v: bool) {
        let (w, b) = (i / 64, 1u64 << (i % 64));
        if v {
            self.bits[w] |= b;
        } else {
            self.bits[w] &= !b;
        }
    }

    pub(crate) fn push(&mut self, violated: bool) {
        if self.filled == self.len && self.get(self.head) {
            self.ones -= 1;
        }
        self.set(self.head, violated);
        if violated {
            self.ones += 1;
        }
        self.head = (self.head + 1) % self.len;
        self.filled = (self.filled + 1).min(self.len);
    }

    /// Violated steps over observed steps. Zero before the first step.
    pub(crate) fn fraction(&self) -> f64 {
        if self.filled == 0 {
            0.0
        } else {
            self.ones as f64 / self.filled as f64
        }
    }

    pub(crate) fn clear(&mut self) {
        let len = self.len;
        *self = Self::new(len);
    }
}

/// Per-kind violation counts, fallback activations, and the spec 10.3 sliding rate.
#[derive(Clone, Copy, Debug)]
pub struct SafetyCounters {
    /// Indexed by [`ViolationKind::index`]: the `failure_mode_histogram` of spec 10.3.
    pub violations: [u64; ViolationKind::COUNT],
    /// How many steps ran a fallback (spec 9.4). Normal operation, not a failure (spec 18.5).
    pub fallback_activations: u64,
    /// Steps validated.
    pub steps: u64,
    /// Steps whose action the envelope changed (spec 9.3).
    pub clamped_steps: u64,
    pub(crate) window: ViolationWindow,
}

impl SafetyCounters {
    pub(crate) fn new(window: usize) -> Self {
        Self {
            violations: [0; ViolationKind::COUNT],
            fallback_activations: 0,
            steps: 0,
            clamped_steps: 0,
            window: ViolationWindow::new(window),
        }
    }

    pub fn count(&self, kind: ViolationKind) -> u64 {
        self.violations[kind.index()]
    }

    pub(crate) fn record(&mut self, kind: ViolationKind) {
        self.violations[kind.index()] += 1;
    }

    /// Fraction of steps in the sliding window the plane clamped, projected or fell back on
    /// (spec 10.3 `envelope_violation_rate`, read by the spec 9.4 rate watchdog).
    pub fn envelope_violation_rate(&self) -> f64 {
        self.window.fraction()
    }

    /// Chunk-exhaustion rate over the whole run (spec 8.6, spec 10.3).
    pub fn chunk_underrun_rate(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.count(ViolationKind::ChunkUnderrun) as f64 / self.steps as f64
        }
    }

    /// Zeroes the statistics. Does not touch the emergency-stop latch (INV-12).
    pub fn reset(&mut self) {
        self.violations = [0; ViolationKind::COUNT];
        self.fallback_activations = 0;
        self.steps = 0;
        self.clamped_steps = 0;
        self.window.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_slides_and_counts() {
        let near = |a: f64, b: f64| (a - b).abs() < 1e-12;
        let mut w = ViolationWindow::new(4);
        assert!(near(w.fraction(), 0.0));
        for v in [true, false, false, false] {
            w.push(v);
        }
        assert!(near(w.fraction(), 0.25));
        // Pushing four clean steps evicts the violation.
        for _ in 0..4 {
            w.push(false);
        }
        assert!(near(w.fraction(), 0.0));
        for _ in 0..4 {
            w.push(true);
        }
        assert!(near(w.fraction(), 1.0));
    }

    #[test]
    fn window_len_is_clamped_to_capacity() {
        assert_eq!(ViolationWindow::new(0).len, 1);
        assert_eq!(ViolationWindow::new(10_000).len, WINDOW_CAP);
    }

    #[test]
    fn rates_are_zero_before_any_step() {
        let c = SafetyCounters::new(16);
        assert!(c.envelope_violation_rate() < 1e-12);
        assert!(c.chunk_underrun_rate() < 1e-12);
        assert_eq!(c.count(ViolationKind::Position), 0);
    }
}
