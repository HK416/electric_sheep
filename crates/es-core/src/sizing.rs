//! Sizing arithmetic shared by the runtime and the compiler's memory budget (§12.4, §20.2).
//!
//! `es-env` (layer 9) allocates the action chunk buffers and `es-compile` (layer 7) budgets
//! them; neither may depend on the other (§4.2), so the one formula they have to agree on
//! lives here, at layer 1. Two models of the same quantity is how the §28.7 gate 13 ±10%
//! check ends up unreachable.

/// Overlapping action chunks one env's `es_env::ChunkBuffer` keeps.
///
/// The live overlap is `ceil(H / replan_interval)` — 5 for the §8.6 worked example; eight
/// slots leave headroom and keep the scan cheap.
pub const CHUNK_SLOTS: usize = 8;

/// Bytes of action chunk buffer for `n_envs` envs at action dim `nj` and horizon `h`.
///
/// **Deviation from §20.2**, which budgets `N_sim × H × NJ × 4B × 2` (f32, double-buffered):
/// the implementation keeps [`CHUNK_SLOTS`] overlapping chunks of `f64` per env, because ACT
/// temporal ensembling averages *every* live chunk (§8.6) and the control path is f64
/// throughout. That is 4× the spec's figure, and it is the implementation that is right — the
/// spec's line predates the ensembling buffer.
pub const fn chunk_buffer_bytes(n_envs: u64, nj: u64, h: u64) -> u64 {
    n_envs
        .saturating_mul(CHUNK_SLOTS as u64)
        .saturating_mul(h)
        .saturating_mul(nj)
        .saturating_mul(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The §12.4 gate configuration: 4,096 env × 8 slots × H=20 × NJ=7 × 8 B.
    #[test]
    fn the_gate_configuration_costs_36_7_mb() {
        assert_eq!(chunk_buffer_bytes(4096, 7, 20), 36_700_160);
        assert_eq!(chunk_buffer_bytes(0, 7, 20), 0);
        assert_eq!(chunk_buffer_bytes(u64::MAX, 7, 20), u64::MAX, "saturates");
    }
}
