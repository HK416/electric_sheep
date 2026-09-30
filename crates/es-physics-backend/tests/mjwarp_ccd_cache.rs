//! Packet M16/H0: the reproduction of the CCD `KeyError` packet H1 hit, and `mjwarp_ref.py`'s
//! workaround (`patch_ccd_grid_size`).
//!
//! On the Shadow Hand scene, with `mujoco_warp` 3.13.0 and `warp-lang` 1.16.0, `mjw.step` can
//! raise `KeyError: 'ccd_kernel_builder__locals__ccd_kernel_<hash>_cuda_kernel_forward_smem_bytes'`
//! from `collision_convex._ccd_grid_size` -> `wp.get_suggested_block_size`: the loaded CCD
//! module's metadata does not list the kernel asked about. It depends on the state of warp's
//! kernel cache; measured here, without the workaround, a fresh cache failed within its first
//! two processes three times out of three (four concurrent processes, all four failed), and
//! this test, run without it, failed at the first control step. So the test runs the hand
//! through `MjWarpBackend` twice on a fresh `WARP_CACHE_PATH` -- which compiles `mujoco_warp`
//! from scratch, minutes, hence `#[ignore]`.
//!
//! ```text
//! ES_PYTHON=.venv/Scripts/python.exe cargo test -p es-physics-backend --test mjwarp_ccd_cache \
//!   -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use es_core::TickRate;
use es_physics_backend::MjWarpBackend;
use es_physics_core::{LoadConfig, PhysicsBackend};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf/shadow_hand")
}

#[test]
#[ignore = "compiles mujoco_warp into a fresh kernel cache (minutes); needs mujoco_warp"]
fn the_hand_steps_on_a_fresh_and_a_warm_kernel_cache() {
    if let Err(why) = MjWarpBackend::is_available() {
        println!("SKIP the_hand_steps_on_a_fresh_and_a_warm_kernel_cache: {why}");
        return;
    }
    let xml = std::fs::read_to_string(dir().join("shadow_hand_repose.xml")).unwrap();
    let mut scene = es_assets::parse_mjcf(&xml).unwrap().scene;
    es_assets::mesh::load(&mut scene, &dir()).unwrap();
    let cache = std::env::temp_dir().join(format!("es-h0-warp-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    // The only test in this binary, so no other test sees the variable; the adapter process
    // inherits it.
    std::env::set_var("WARP_CACHE_PATH", &cache);
    for run in ["fresh", "warm"] {
        let mut backend = MjWarpBackend::new();
        let info = backend
            .load(
                &scene,
                &LoadConfig {
                    n_envs: 16,
                    rate: Some(TickRate::hz(120)),
                    seed: 0,
                },
            )
            .unwrap_or_else(|e| panic!("{run} cache: load: {e}"));
        for k in 0..20 {
            let ctrl = vec![0.02 * f64::from(k % 5); (info.nu * info.n_envs) as usize];
            backend.set_ctrl(&ctrl).unwrap();
            backend
                .step(2)
                .unwrap_or_else(|e| panic!("{run} cache, control step {k}: {e}"));
        }
        println!("RAN {run} kernel cache: 20 control steps x 16 envs");
    }
    let _ = std::fs::remove_dir_all(&cache);
}
