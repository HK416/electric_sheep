//! The solver-level options of a scene, which a backend maps or reports as unmapped.

use es_math::Vec3;
use serde::{Deserialize, Serialize};

/// Solver-level options. Backends map what they support and report the rest (spec 17.2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicsOptions {
    /// Physics step, s.
    pub timestep: f64,
    /// m/s^2, Z-up so the default is `-Z`.
    pub gravity: Vec3,
    pub integrator: Integrator,
    pub cone: FrictionCone,
    pub jacobian: Jacobian,
    pub solver: Solver,
    pub iterations: u32,
    /// Solver line-search iterations. Carried because a policy trained under
    /// `ls_iterations = 5` (`MuJoCo` Playground's locomotion setting) resolves contact
    /// differently from one stepped at `MuJoCo`'s default 50, and a backend that dropped it
    /// would change the physics silently (spec 17.2; packet M6/B1).
    #[serde(default = "default_ls_iterations")]
    pub ls_iterations: u32,
    /// `<option><flag eulerdamp>`: `MuJoCo`'s default integrates joint damping implicitly under
    /// the Euler integrator. MJX-trained models disable it, and the difference is visible in
    /// `qpos` within a few steps, so it is a carried option rather than an ignored flag.
    #[serde(default = "default_eulerdamp")]
    pub eulerdamp: bool,
    /// Frictional-to-normal constraint impedance ratio.
    pub impratio: f64,
}

fn default_ls_iterations() -> u32 {
    50
}

fn default_eulerdamp() -> bool {
    true
}

impl Default for PhysicsOptions {
    fn default() -> Self {
        Self {
            timestep: 0.002,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            integrator: Integrator::Euler,
            cone: FrictionCone::Pyramidal,
            jacobian: Jacobian::Auto,
            solver: Solver::Newton,
            iterations: 100,
            ls_iterations: default_ls_iterations(),
            eulerdamp: default_eulerdamp(),
            impratio: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Integrator {
    Euler,
    Rk4,
    Implicit,
    ImplicitFast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrictionCone {
    Pyramidal,
    Elliptic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Jacobian {
    Dense,
    Sparse,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Solver {
    Pgs,
    Cg,
    Newton,
}
