//! `Ros2Config` parsing (`docs/design/ros2-boundary.md` section 4.1, 4.2). No `zenoh` feature:
//! plain data, checked the same way on every machine (work packet `docs/packets/M3/
//! W1b-ros2-zenoh-session.md`).

use std::time::Duration;

use es_ros2::config::{parse_config, Ros2Mode, ZenohMode};

const DEFAULT_CONNECT: &str = "tcp/localhost:7447";
const DEFAULT_LISTEN: &str = "tcp/localhost:0";

const BASE: &str = r#"
[ros2]
domain_id = 73
node = "es_runtime"
namespace = "/"
enclave = "/"
liveliness_timeout_ms = 1000
"#;

#[test]
fn defaults_mirror_the_rmw_zenoh_session_config() {
    let cfg = parse_config(BASE).unwrap();
    assert_eq!(cfg.domain_id, 73);
    assert_eq!(cfg.liveliness_timeout, Duration::from_millis(1000));
    match cfg.mode {
        Ros2Mode::RmwZenoh(z) => {
            assert_eq!(z.mode, ZenohMode::Peer);
            assert_eq!(z.connect, vec![DEFAULT_CONNECT.to_owned()]);
            assert_eq!(z.listen, vec![DEFAULT_LISTEN.to_owned()]);
        }
        Ros2Mode::DdsBridge { .. } => panic!("no mode table present must imply mode A"),
    }
    assert!(cfg.actuators.is_empty());
}

#[test]
fn both_mode_tables_are_rejected() {
    let toml = format!(
        "{BASE}\n[ros2.rmw_zenoh]\nmode = \"peer\"\n[ros2.dds_bridge]\nbridge_namespace = \"/\"\n"
    );
    assert_eq!(parse_config(&toml).unwrap_err().code(), "ROS2-001");
}

#[test]
fn rust_dds_is_rejected() {
    let toml = format!("{BASE}\n[ros2.rust_dds]\n");
    assert_eq!(parse_config(&toml).unwrap_err().code(), "ROS2-002");
}

#[test]
fn rust_dds_wins_over_both_modes_too() {
    let toml = format!(
        "{BASE}\n[ros2.rust_dds]\n[ros2.rmw_zenoh]\n[ros2.dds_bridge]\nbridge_namespace = \"/\"\n"
    );
    assert_eq!(parse_config(&toml).unwrap_err().code(), "ROS2-002");
}

#[test]
fn unknown_keys_are_rejected() {
    let toml = format!("{BASE}\nunknown_top_level = 1\n");
    assert_eq!(parse_config(&toml).unwrap_err().code(), "ROS2-000");

    let toml = format!("{BASE}\n[ros2.rmw_zenoh]\nbogus = true\n");
    assert_eq!(parse_config(&toml).unwrap_err().code(), "ROS2-000");
}

#[test]
fn mode_b_takes_only_connect_and_bridge_namespace() {
    let toml = format!("{BASE}\n[ros2.dds_bridge]\nbridge_namespace = \"/robot1\"\n");
    let cfg = parse_config(&toml).unwrap();
    match cfg.mode {
        Ros2Mode::DdsBridge {
            endpoints,
            bridge_namespace,
        } => {
            assert_eq!(bridge_namespace, "/robot1");
            assert_eq!(endpoints.connect, vec![DEFAULT_CONNECT.to_owned()]);
            assert!(endpoints.listen.is_empty());
        }
        Ros2Mode::RmwZenoh(_) => panic!("dds_bridge table must select mode B"),
    }
}

#[test]
fn actuator_joint_count_mismatch_is_rejected() {
    let toml =
        format!("{BASE}\n[[ros2.actuator]]\ntopic = \"/cmd\"\njoints = [\"j1\", \"j2\", \"j3\"]\n");
    let cfg = parse_config(&toml).unwrap();
    assert!(cfg.is_actuator_topic("/cmd"));
    assert!(!cfg.is_actuator_topic("/other"));

    let found = cfg.actuator_topic::<3>("/cmd").unwrap();
    assert_eq!(found.joints, vec!["j1", "j2", "j3"]);

    assert_eq!(
        cfg.actuator_topic::<2>("/cmd").unwrap_err().code(),
        "ROS2-013"
    );
    assert_eq!(
        cfg.actuator_topic::<3>("/missing").unwrap_err().code(),
        "ROS2-013"
    );
}

#[test]
fn empty_or_duplicate_actuator_topics_are_rejected() {
    let toml = format!("{BASE}\n[[ros2.actuator]]\ntopic = \"/cmd\"\njoints = []\n");
    assert_eq!(parse_config(&toml).unwrap_err().code(), "ROS2-000");

    let toml = format!(
        "{BASE}\n[[ros2.actuator]]\ntopic = \"/cmd\"\njoints = [\"j1\"]\n[[ros2.actuator]]\ntopic = \"/cmd\"\njoints = [\"j2\"]\n"
    );
    assert_eq!(parse_config(&toml).unwrap_err().code(), "ROS2-000");
}

#[test]
fn syntax_errors_are_ros2_000() {
    assert_eq!(
        parse_config("not valid toml [[[").unwrap_err().code(),
        "ROS2-000"
    );
}
