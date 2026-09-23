//! Packet M11/X1 oracle 1: the backend is a condition of a run (spec 28.14 rule 2).
//!
//! `backend_identity` is `H("es.backend.v1", name, engine version, float, determinism tier)`;
//! `hardware_capability` is what a run writes into the `hardware_capability` slot of
//! `execution_hash` (spec 5.3) -- the identity on every backend except `mujoco-cpu`, the
//! reference (spec 17.1), whose slot stays the all-zero value `RunConfig::default()` has always
//! written, so no committed `evaluation.lock` moves. No Python is needed.
//!
//!     cargo test -p es-physics-backend backend_identity

use es_physics_backend::{
    backend_identity, hardware_capability, mjwarp, mujoco, newton, physx, BackendKind,
};
use es_physics_core::Capabilities;

/// The declared capabilities of every name in the spec 17.2 table.
fn caps(kind: BackendKind) -> Capabilities {
    match kind {
        BackendKind::MuJoCoCpu => mujoco::capabilities(),
        BackendKind::MjWarp => mjwarp::capabilities(),
        BackendKind::Newton => newton::capabilities(),
        BackendKind::PhysX => physx::capabilities(),
    }
}

#[test]
fn backend_identity_differs_across_the_four_names() {
    let digests: Vec<[u8; 32]> = BackendKind::ALL
        .into_iter()
        .map(|k| backend_identity(&caps(k), "1.0.0"))
        .collect();
    for (i, a) in digests.iter().enumerate() {
        for (j, b) in digests.iter().enumerate().skip(i + 1) {
            assert_ne!(
                a,
                b,
                "{} and {} share an identity",
                BackendKind::ALL[i],
                BackendKind::ALL[j]
            );
        }
    }
}

#[test]
fn backend_identity_differs_across_engine_versions() {
    let c = caps(BackendKind::MjWarp);
    assert_ne!(
        backend_identity(&c, "mujoco_warp 3.3.2; warp 1.8.0; mujoco 3.3.2"),
        backend_identity(&c, "mujoco_warp 3.3.3; warp 1.8.0; mujoco 3.3.2"),
    );
}

#[test]
fn backend_identity_is_stable_across_calls() {
    for kind in BackendKind::ALL {
        let c = caps(kind);
        assert_eq!(
            backend_identity(&c, "v"),
            backend_identity(&c, "v"),
            "{kind}"
        );
    }
}

/// The fields are length-prefixed, so moving a byte from one field into its neighbour is a
/// different identity rather than the same concatenation.
#[test]
fn backend_identity_fields_do_not_run_together() {
    let a = Capabilities {
        name: "mjwarp".to_owned(),
        ..mjwarp::capabilities()
    };
    let b = Capabilities {
        name: "mjwar".to_owned(),
        ..mjwarp::capabilities()
    };
    assert_ne!(backend_identity(&a, "1"), backend_identity(&b, "p1"));
}

/// The declaration's float precision and determinism tier are in the tuple: the same name and
/// version declaring f64 is a different condition.
#[test]
fn backend_identity_covers_float_and_tier() {
    let base = caps(BackendKind::MjWarp);
    let f64_caps = Capabilities {
        float: es_physics_core::caps::FloatPrecision::F64,
        ..base.clone()
    };
    let tier = Capabilities {
        determinism: es_physics_core::caps::DeterminismTier::SemanticEqual,
        ..base.clone()
    };
    let id = backend_identity(&base, "v");
    assert_ne!(id, backend_identity(&f64_caps, "v"));
    assert_ne!(id, backend_identity(&tier, "v"));
}

/// `mujoco-cpu` keeps today's slot -- `RunConfig::default().hardware`, all zeros -- whatever
/// its engine version, and every other backend writes its identity.
#[test]
fn backend_identity_mujoco_cpu_slot_is_todays_value() {
    let cpu = caps(BackendKind::MuJoCoCpu);
    assert_eq!(hardware_capability(&cpu, "3.3.2"), [0; 32]);
    assert_eq!(hardware_capability(&cpu, "3.4.0"), [0; 32]);
    for kind in [BackendKind::MjWarp, BackendKind::Newton, BackendKind::PhysX] {
        let c = caps(kind);
        assert_eq!(
            hardware_capability(&c, "v"),
            backend_identity(&c, "v"),
            "{kind}"
        );
        assert_ne!(hardware_capability(&c, "v"), [0; 32], "{kind}");
    }
}
