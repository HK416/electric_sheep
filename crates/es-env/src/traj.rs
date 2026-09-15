//! `.estraj` — the per-tick state trajectory of one episode (packet M5/V9, design note
//! `docs/design/visible-learning.md` section 7.17; spec 25.1 bounded parsing).
//!
//! What "what the robot did" means for provenance, and the input the showcase render replays:
//! one record per control tick holding `qpos`, `qvel` and every body's world pose. The poses
//! are stored rather than re-derived because they are what [`crate::render`] feeds
//! `TriScene::from_scene_with_poses` — so a replay is bit-for-bit the frame the run produced,
//! with no physics backend, no forward kinematics and no second implementation of either.
//!
//! ```text
//! magic  "ESTRAJ01"          8 bytes
//! nq, nv, nbody, ticks       4 x u32 little-endian
//! body ids                   nbody x 16 bytes (`StableId`)
//! rows                       ticks x (nq + nv + nbody*7) x f64 little-endian
//! ```
//!
//! A row is `qpos ‖ qvel ‖ (pos[3] ‖ quat[4]) * nbody`, bodies in `ModelInfo::body` order,
//! which is `StableId` order because it is a `BTreeMap` (spec 3.4). The length is implied by
//! the header and checked before anything is allocated: a truncated or over-long file is
//! refused, never padded (spec 25.1).

use std::collections::BTreeMap;
use std::path::Path;

use es_core::StableId;
use es_math::{Pose, Quat, Vec3};
use es_physics_core::backend::{ModelInfo, StateView};

use crate::EnvError;

/// World pose of every body the model indexes, read out of one env's `xpos` / `xquat` rows.
///
/// A body the backend does not report — an empty `xpos`, a short row — is simply absent from
/// the map, and [`es_render::TriScene::from_scene_with_poses`] then keeps its scene pose. A
/// partially known state degrades to the static scene, never to the origin.
pub fn body_poses(model: &ModelInfo, state: &StateView<'_>, env: u32) -> BTreeMap<StableId, Pose> {
    let nbody = model.nbody as usize;
    let base = env as usize * nbody;
    let mut out = BTreeMap::new();
    for (id, range) in &model.body {
        let row = base + range.start as usize;
        let (Some(p), Some(q)) = (
            state.xpos.get(row * 3..row * 3 + 3),
            state.xquat.get(row * 4..row * 4 + 4),
        ) else {
            continue;
        };
        out.insert(
            *id,
            Pose::new(
                Vec3::new(p[0], p[1], p[2]),
                Quat::from_xyzw(q[0], q[1], q[2], q[3]),
            ),
        );
    }
    out
}

const MAGIC: &[u8; 8] = b"ESTRAJ01";
const HEADER: usize = 8 + 4 * 4;

/// One episode's per-tick state, in memory.
///
/// Built by pushing ticks and written once at the episode boundary: an episode is under a
/// megabyte for the demo scene, and a length nobody can patch back into a header is a length
/// a truncated file cannot fake.
#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory {
    nq: u32,
    nv: u32,
    bodies: Vec<StableId>,
    rows: Vec<f64>,
}

impl Trajectory {
    /// An empty trajectory shaped for `model`.
    pub fn new(model: &ModelInfo) -> Self {
        Self {
            nq: model.nq,
            nv: model.nv,
            bodies: model.body.keys().copied().collect(),
            rows: Vec::new(),
        }
    }

    /// Appends the tick `state` describes, for env `env`.
    ///
    /// The poses come from [`body_poses`] itself, so [`Self::poses`] returns
    /// exactly what the renderer was handed. A body that dropped out of the state between
    /// ticks is an error rather than a short row.
    pub fn push(
        &mut self,
        model: &ModelInfo,
        state: &StateView<'_>,
        env: u32,
    ) -> Result<(), EnvError> {
        let world = body_poses(model, state, env);
        if world.len() != self.bodies.len() || !self.bodies.iter().all(|id| world.contains_key(id))
        {
            return Err(EnvError::Trajectory(format!(
                "the state reports {} of the model's {} bodies",
                world.len(),
                self.bodies.len()
            )));
        }
        let (qpos, qvel) = (state.qpos_of(env), state.qvel_of(env));
        if qpos.len() != self.nq as usize || qvel.len() != self.nv as usize {
            return Err(EnvError::Trajectory(format!(
                "state is {}/{} wide, the model declares {}/{}",
                qpos.len(),
                qvel.len(),
                self.nq,
                self.nv
            )));
        }
        self.rows.extend_from_slice(qpos);
        self.rows.extend_from_slice(qvel);
        for id in &self.bodies {
            let p = world[id];
            self.rows
                .extend_from_slice(&[p.position.x, p.position.y, p.position.z]);
            self.rows.extend_from_slice(&[
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ]);
        }
        Ok(())
    }

    fn stride(&self) -> usize {
        self.nq as usize + self.nv as usize + self.bodies.len() * 7
    }

    pub fn ticks(&self) -> usize {
        self.rows.len().checked_div(self.stride()).unwrap_or(0)
    }

    pub fn bodies(&self) -> &[StableId] {
        &self.bodies
    }

    fn row(&self, tick: usize) -> &[f64] {
        let stride = self.stride();
        &self.rows[tick * stride..(tick + 1) * stride]
    }

    /// `qpos` at `tick`. Panics past [`Self::ticks`], like any slice index.
    pub fn qpos(&self, tick: usize) -> &[f64] {
        &self.row(tick)[..self.nq as usize]
    }

    /// `qvel` at `tick`.
    pub fn qvel(&self, tick: usize) -> &[f64] {
        let q = self.nq as usize;
        &self.row(tick)[q..q + self.nv as usize]
    }

    /// The world poses at `tick`, in the shape `TriScene::from_scene_with_poses` reads — and
    /// bit for bit what [`body_poses`] returned when the tick was recorded.
    pub fn poses(&self, tick: usize) -> BTreeMap<StableId, Pose> {
        let base = self.nq as usize + self.nv as usize;
        let row = self.row(tick);
        self.bodies
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let p = &row[base + i * 7..base + i * 7 + 7];
                // The struct literal and not `Pose::new`: what was recorded is already
                // `Pose::new`'s output, and normalising a canonical unit quaternion a second
                // time moves its last bit. Bit-for-bit is the whole claim of this file.
                (
                    *id,
                    Pose {
                        position: Vec3::new(p[0], p[1], p[2]),
                        orientation: Quat::from_xyzw(p[3], p[4], p[5], p[6]),
                    },
                )
            })
            .collect()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER + self.bodies.len() * 16 + self.rows.len() * 8);
        out.extend_from_slice(MAGIC);
        for n in [
            self.nq,
            self.nv,
            self.bodies.len() as u32,
            self.ticks() as u32,
        ] {
            out.extend_from_slice(&n.to_le_bytes());
        }
        for id in &self.bodies {
            out.extend_from_slice(id.as_bytes());
        }
        for v in &self.rows {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Parses `bytes`, checking every length against the header before allocating (spec 25.1).
    pub fn parse(bytes: &[u8]) -> Result<Self, EnvError> {
        let bad = |m: String| EnvError::Trajectory(m);
        if bytes.len() < HEADER || &bytes[..8] != MAGIC {
            return Err(bad(format!(
                "not an .estraj file: {} byte(s), magic {:?}",
                bytes.len(),
                &bytes[..bytes.len().min(8)]
            )));
        }
        let u32_at = |i: usize| {
            u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize
        };
        let (nq, nv, nbody, ticks) = (u32_at(8), u32_at(12), u32_at(16), u32_at(20));
        let too_big = || bad("header declares more data than can be addressed".to_owned());
        let stride = nbody
            .checked_mul(7)
            .and_then(|b| b.checked_add(nq)?.checked_add(nv))
            .ok_or_else(too_big)?;
        if stride == 0 {
            return Err(bad(
                "a trajectory with no state per tick: nq, nv and nbody are all zero".to_owned(),
            ));
        }
        let expected = ticks
            .checked_mul(stride)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(nbody.checked_mul(16)?))
            .and_then(|n| n.checked_add(HEADER))
            .ok_or_else(too_big)?;
        if bytes.len() != expected {
            return Err(bad(format!(
                "header declares {expected} byte(s) (nq {nq}, nv {nv}, nbody {nbody}, ticks \
                 {ticks}), the file is {}",
                bytes.len()
            )));
        }
        let mut bodies = Vec::with_capacity(nbody);
        for i in 0..nbody {
            let at = HEADER + i * 16;
            let mut id = [0u8; 16];
            id.copy_from_slice(&bytes[at..at + 16]);
            bodies.push(StableId::from_bytes(id));
        }
        let mut rows = Vec::with_capacity(ticks * stride);
        let base = HEADER + nbody * 16;
        for i in 0..ticks * stride {
            let at = base + i * 8;
            let mut v = [0u8; 8];
            v.copy_from_slice(&bytes[at..at + 8]);
            rows.push(f64::from_le_bytes(v));
        }
        Ok(Self {
            nq: nq as u32,
            nv: nv as u32,
            bodies,
            rows,
        })
    }

    pub fn write(&self, path: &Path) -> Result<(), EnvError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| EnvError::Trajectory(format!("{}: {e}", parent.display())))?;
        }
        std::fs::write(path, self.to_bytes())
            .map_err(|e| EnvError::Trajectory(format!("{}: {e}", path.display())))
    }

    pub fn read(path: &Path) -> Result<Self, EnvError> {
        let bytes = std::fs::read(path)
            .map_err(|e| EnvError::Trajectory(format!("{}: {e}", path.display())))?;
        Self::parse(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn model(nq: u32, nv: u32, nbody: u32) -> ModelInfo {
        let mut m = ModelInfo {
            nq,
            nv,
            nbody,
            ..ModelInfo::default()
        };
        for i in 0..nbody {
            m.body.insert(
                StableId::from_path(&format!("body/{i}")),
                es_physics_core::backend::IndexRange { start: i, len: 1 },
            );
        }
        m
    }

    fn built(nq: u32, nv: u32, nbody: u32, ticks: usize) -> Trajectory {
        let m = model(nq, nv, nbody);
        let mut t = Trajectory::new(&m);
        for k in 0..ticks {
            let f = |i: usize| (k * 101 + i) as f64 * 0.125;
            let qpos: Vec<f64> = (0..nq as usize).map(f).collect();
            let qvel: Vec<f64> = (0..nv as usize).map(|i| -f(i)).collect();
            let xpos: Vec<f64> = (0..nbody as usize * 3).map(f).collect();
            let xquat: Vec<f64> = (0..nbody as usize * 4).map(|i| f(i) + 1.0).collect();
            let state = StateView {
                n_envs: 1,
                tick: es_core::PhysTick(k as u64),
                qpos: &qpos,
                qvel: &qvel,
                act: &[],
                sensordata: &[],
                xpos: &xpos,
                xquat: &xquat,
            };
            t.push(&m, &state, 0).expect("push");
        }
        t
    }

    /// The claim the showcase render rests on: a replayed tick is the pose map the renderer
    /// was handed, bit for bit — no tolerance, no re-derivation.
    #[test]
    fn a_replayed_tick_is_the_pose_map_the_renderer_was_handed() {
        let m = model(3, 3, 4);
        let (qpos, qvel) = (vec![0.5, -1.5, 2.25], vec![1.0, 2.0, 3.0]);
        let xpos: Vec<f64> = (0..12).map(|i| f64::from(i) * 0.1).collect();
        let xquat: Vec<f64> = (0..16).map(|i| f64::from(i) * 0.25 - 1.0).collect();
        let state = StateView {
            n_envs: 1,
            tick: es_core::PhysTick(0),
            qpos: &qpos,
            qvel: &qvel,
            act: &[],
            sensordata: &[],
            xpos: &xpos,
            xquat: &xquat,
        };
        let mut t = Trajectory::new(&m);
        t.push(&m, &state, 0).expect("push");
        assert_eq!(t.poses(0), body_poses(&m, &state, 0));
        assert_eq!(t.qpos(0), &qpos[..]);
        assert_eq!(t.qvel(0), &qvel[..]);
    }

    #[test]
    fn a_truncated_file_is_refused() {
        let bytes = built(3, 3, 2, 5).to_bytes();
        for cut in [0, 1, HEADER - 1, HEADER, HEADER + 7, bytes.len() - 1] {
            assert!(
                Trajectory::parse(&bytes[..cut]).is_err(),
                "{cut} byte(s) parsed"
            );
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(
            Trajectory::parse(&longer).is_err(),
            "a trailing byte parsed"
        );
        assert!(Trajectory::parse(&bytes).is_ok());
    }

    proptest! {
        /// Round-trip, and every byte length is the header's own arithmetic.
        #[test]
        fn a_trajectory_round_trips(
            nq in 0u32..8,
            nv in 0u32..8,
            nbody in 1u32..6,
            ticks in 0usize..20,
        ) {
            let t = built(nq, nv, nbody, ticks);
            let bytes = t.to_bytes();
            let back = Trajectory::parse(&bytes).expect("round trip");
            prop_assert_eq!(&back, &t);
            prop_assert_eq!(back.ticks(), ticks);
            prop_assert_eq!(bytes.len(), HEADER + nbody as usize * 16
                + ticks * (nq as usize + nv as usize + nbody as usize * 7) * 8);
            for k in 0..ticks {
                prop_assert_eq!(back.qpos(k), t.qpos(k));
                prop_assert_eq!(back.poses(k), t.poses(k));
            }
        }

        /// Bounded parsing (spec 25.1): no header, however hostile, makes the parser allocate
        /// what it was told rather than what the file holds, or panic. Whatever does parse
        /// re-serializes to the identical bytes.
        #[test]
        fn a_hostile_header_allocates_nothing_it_cannot_read(
            bytes in proptest::collection::vec(any::<u8>(), 0..64),
        ) {
            let mut raw = MAGIC.to_vec();
            raw.extend_from_slice(&bytes);
            if let Ok(t) = Trajectory::parse(&raw) {
                prop_assert_eq!(t.to_bytes(), raw);
            }
        }
    }
}
