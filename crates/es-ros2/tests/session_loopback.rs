//! Mode A, the A xor B startup probe, and the actuator path, over two in-process zenoh peers on
//! `127.0.0.1` (`docs/design/ros2-boundary.md` section 4, "In-process loopback"): no router, no
//! multicast, no outside network. PR tier (`zenoh` feature; `cargo xtask ci` enables it).

#![cfg(feature = "zenoh")]

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::Duration;

use es_ros2::attachment::{gid_of, Attachment};
use es_ros2::config::{ActuatorTopic, Ros2Config, Ros2Mode, ZenohEndpoints, ZenohMode};
use es_ros2::msg::{Float64MultiArray, Msg, MsgType, StringMsg};
use es_ros2::names::{topic_key_expr, LivelinessToken};
use es_ros2::session::{Durability, Ros2Node};
use es_safety::{
    ActionChunk, Envelope, ExecutionMode, Fallback, FallbackKind, Limit, Micros, SafetyConfig,
    SafetyPlane, Watchdogs, WorkspaceSpec,
};
use zenoh::Wait;

const PROBE_TIMEOUT: Duration = Duration::from_millis(500);
const RECV_TIMEOUT: Duration = Duration::from_secs(2);

fn reserve_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
    let port = l.local_addr().expect("local addr").port();
    drop(l);
    port
}

/// A mode A config listening on `listen_port` and, if given, connecting straight to another
/// peer's `connect_port` (design note "In-process loopback": one peer listens, the other
/// connects; no router).
fn mode_a_config(
    domain: u32,
    listen_port: u16,
    connect_port: Option<u16>,
    actuators: Vec<ActuatorTopic>,
) -> Ros2Config {
    Ros2Config {
        domain_id: domain,
        node: "es_test".to_owned(),
        namespace: "/".to_owned(),
        enclave: "/".to_owned(),
        liveliness_timeout: PROBE_TIMEOUT,
        mode: Ros2Mode::RmwZenoh(ZenohEndpoints {
            mode: ZenohMode::Peer,
            connect: connect_port
                .map(|p| vec![format!("tcp/127.0.0.1:{p}")])
                .unwrap_or_default(),
            listen: vec![format!("tcp/127.0.0.1:{listen_port}")],
        }),
        actuators,
    }
}

fn mode_b_config(
    domain: u32,
    listen_port: u16,
    connect_port: u16,
    bridge_namespace: &str,
) -> Ros2Config {
    Ros2Config {
        domain_id: domain,
        node: "es_test".to_owned(),
        namespace: "/".to_owned(),
        enclave: "/".to_owned(),
        liveliness_timeout: PROBE_TIMEOUT,
        mode: Ros2Mode::DdsBridge {
            endpoints: ZenohEndpoints {
                mode: ZenohMode::Peer,
                connect: vec![format!("tcp/127.0.0.1:{connect_port}")],
                listen: vec![format!("tcp/127.0.0.1:{listen_port}")],
            },
            bridge_namespace: bridge_namespace.to_owned(),
        },
        actuators: Vec::new(),
    }
}

/// A raw (not `es_ros2`) zenoh peer, for observing what a node under test puts on the wire —
/// exactly the role `docs/api-notes/zenoh-rs.md`'s "In-process loopback" section describes.
fn raw_peer(listen_port: u16, connect_port: u16) -> zenoh::Session {
    let mut cfg = zenoh::Config::default();
    cfg.insert_json5("mode", "\"peer\"").unwrap();
    cfg.insert_json5(
        "connect/endpoints",
        &format!("[\"tcp/127.0.0.1:{connect_port}\"]"),
    )
    .unwrap();
    cfg.insert_json5(
        "listen/endpoints",
        &format!("[\"tcp/127.0.0.1:{listen_port}\"]"),
    )
    .unwrap();
    cfg.insert_json5("scouting/multicast/enabled", "false")
        .unwrap();
    zenoh::open(cfg).wait().expect("open raw peer session")
}

/// Every `MP`/`MS`/`NN` liveliness token currently visible from `session`, parsed.
fn liveliness_tokens(session: &zenoh::Session, pattern: &str) -> Vec<LivelinessToken> {
    let replies = session
        .liveliness()
        .get(pattern)
        .timeout(RECV_TIMEOUT)
        .wait()
        .expect("liveliness get");
    let mut out = Vec::new();
    while let Ok(Some(reply)) = replies.recv_timeout(RECV_TIMEOUT) {
        if let Ok(sample) = reply.result() {
            if let Ok(tok) = LivelinessToken::parse(sample.key_expr().as_str()) {
                out.push(tok);
            }
        }
    }
    out
}

// --- mode A data path -------------------------------------------------------------------------

#[test]
fn mode_a_publisher_sends_cdr_with_a_33_byte_attachment() {
    let (pub_port, sub_port) = (reserve_port(), reserve_port());
    let node = Ros2Node::open(&mode_a_config(1, pub_port, Some(sub_port), Vec::new())).unwrap();
    let observer = raw_peer(sub_port, pub_port);

    let key = topic_key_expr(1, "/chatter", MsgType::String).unwrap();
    let sub = observer.declare_subscriber(&key).wait().unwrap();
    // Give the two peers a moment to route to each other over the fresh TCP link.
    std::thread::sleep(Duration::from_millis(200));

    let publisher = node.publisher("/chatter", MsgType::String, 1).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let mp_tokens: Vec<LivelinessToken> = liveliness_tokens(&observer, "@ros2_lv/**")
        .into_iter()
        .filter(|t| t.kind == es_ros2::names::EntityKind::Publisher)
        .collect();
    assert_eq!(mp_tokens.len(), 1, "exactly one MP token from this node");
    let mp_key = mp_tokens[0].to_key_expr();
    let expected_gid = gid_of(&mp_key);

    for expect_seq in 1..=3i64 {
        publisher
            .put(&Msg::String(StringMsg {
                data: "hello from es".to_owned(),
            }))
            .unwrap();
        let sample = sub.recv_timeout(RECV_TIMEOUT).unwrap().expect("a sample");
        assert_eq!(
            *sample.encoding(),
            zenoh::bytes::Encoding::default(),
            "no encoding is set"
        );
        let payload = sample.payload().to_bytes().to_vec();
        assert_eq!(
            payload,
            StringMsg {
                data: "hello from es".to_owned()
            }
            .to_cdr()
        );
        let att_bytes = sample.attachment().expect("attachment present").to_bytes();
        assert_eq!(att_bytes.len(), 33);
        let att = Attachment::decode(&att_bytes).unwrap();
        assert_eq!(att.seq, expect_seq);
        assert_eq!(att.gid, expected_gid);
    }
}

#[test]
fn mode_a_node_declares_nn_and_mp_tokens_that_parse() {
    let (a_port, b_port) = (reserve_port(), reserve_port());
    let node = Ros2Node::open(&mode_a_config(2, a_port, Some(b_port), Vec::new())).unwrap();
    let observer = raw_peer(b_port, a_port);
    std::thread::sleep(Duration::from_millis(200));
    let _publisher = node.publisher("/chatter", MsgType::String, 1).unwrap();
    std::thread::sleep(Duration::from_millis(300));

    let tokens = liveliness_tokens(&observer, "@ros2_lv/2/**");
    assert!(
        tokens
            .iter()
            .any(|t| t.kind == es_ros2::names::EntityKind::Node),
        "no NN token: {tokens:?}"
    );
    assert!(
        tokens
            .iter()
            .any(|t| t.kind == es_ros2::names::EntityKind::Publisher),
        "no MP token: {tokens:?}"
    );
}

#[test]
fn mode_a_subscriber_decodes_and_drops_samples_without_attachment() {
    let (sub_port, raw_port) = (reserve_port(), reserve_port());
    let node = Ros2Node::open(&mode_a_config(3, sub_port, Some(raw_port), Vec::new())).unwrap();
    let raw = raw_peer(raw_port, sub_port);
    let key = topic_key_expr(3, "/chatter", MsgType::String).unwrap();
    let subscriber = node.subscriber("/chatter", MsgType::String, 4).unwrap();
    std::thread::sleep(Duration::from_millis(200));

    // No attachment at all: rmw_zenoh drops it, so does this crate.
    raw.put(
        &key,
        StringMsg {
            data: "no attachment".to_owned(),
        }
        .to_cdr(),
    )
    .wait()
    .unwrap();
    assert!(subscriber.recv_timeout(RECV_TIMEOUT).unwrap().is_none());
    assert_eq!(subscriber.dropped(), 1);

    // A well-formed one decodes fine.
    let attachment = Attachment {
        seq: 1,
        source_timestamp_ns: 0,
        gid: [7u8; 16],
    };
    raw.put(
        &key,
        StringMsg {
            data: "hi".to_owned(),
        }
        .to_cdr(),
    )
    .attachment(attachment.encode().to_vec())
    .wait()
    .unwrap();
    let received = subscriber
        .recv_timeout(RECV_TIMEOUT)
        .unwrap()
        .expect("decoded sample");
    match received.msg {
        Msg::String(s) => assert_eq!(s.data, "hi"),
        other => panic!("wrong type: {other:?}"),
    }
    assert_eq!(
        subscriber.dropped(),
        1,
        "the first sample is still the only drop"
    );
}

// --- A xor B, enforced at startup and by a later watch --------------------------------------

#[test]
fn mode_a_refuses_to_start_next_to_a_bridge_token() {
    let (node_port, bridge_port) = (reserve_port(), reserve_port());
    let bridge = raw_peer(bridge_port, node_port);
    let bridge_token_key = format!("@/{}/@ros2_lv", bridge.zid());
    let _token = bridge
        .liveliness()
        .declare_token(&bridge_token_key)
        .wait()
        .unwrap();
    std::thread::sleep(Duration::from_millis(200));

    let err =
        Ros2Node::open(&mode_a_config(4, node_port, Some(bridge_port), Vec::new())).unwrap_err();
    assert_eq!(err.code(), "ROS2-003");
}

#[test]
fn mode_b_refuses_without_a_bridge_plugin_token() {
    let (node_port, other_port) = (reserve_port(), reserve_port());
    let _anchor = raw_peer(other_port, node_port);
    let err = Ros2Node::open(&mode_b_config(5, node_port, other_port, "/")).unwrap_err();
    assert_eq!(err.code(), "ROS2-004");
}

#[test]
fn mode_b_refuses_next_to_an_rmw_zenoh_token() {
    let (node_port, rmw_port) = (reserve_port(), reserve_port());
    let rmw_peer = raw_peer(rmw_port, node_port);
    let bridge_token_key = format!("@/{}/@ros2_lv", rmw_peer.zid());
    let _bridge_marker = rmw_peer
        .liveliness()
        .declare_token(&bridge_token_key)
        .wait()
        .unwrap();
    let node_token_key = LivelinessToken {
        domain_id: 6,
        zid: rmw_peer.zid().to_string(),
        node_id: 0,
        entity_id: 0,
        kind: es_ros2::names::EntityKind::Node,
        enclave: "/".to_owned(),
        namespace: "/".to_owned(),
        node_name: "impostor".to_owned(),
        topic: None,
        extra: Vec::new(),
    }
    .to_key_expr();
    let _conflict = rmw_peer
        .liveliness()
        .declare_token(&node_token_key)
        .wait()
        .unwrap();
    std::thread::sleep(Duration::from_millis(200));

    let err = Ros2Node::open(&mode_b_config(6, node_port, rmw_port, "/")).unwrap_err();
    assert_eq!(err.code(), "ROS2-005");
}

#[test]
fn a_bridge_token_appearing_later_fails_closed() {
    let (node_port, peer_port) = (reserve_port(), reserve_port());
    let node = Ros2Node::open(&mode_a_config(7, node_port, Some(peer_port), Vec::new())).unwrap();
    let publisher = node.publisher("/chatter", MsgType::String, 1).unwrap();
    // The node must accept before any conflict exists.
    publisher
        .put(&Msg::String(StringMsg {
            data: "ok".to_owned(),
        }))
        .unwrap();

    let peer = raw_peer(peer_port, node_port);
    let bridge_token_key = format!("@/{}/@ros2_lv", peer.zid());
    let _bridge = peer
        .liveliness()
        .declare_token(&bridge_token_key)
        .wait()
        .unwrap();
    std::thread::sleep(Duration::from_millis(300));

    let err = publisher
        .put(&Msg::String(StringMsg {
            data: "latched".to_owned(),
        }))
        .unwrap_err();
    assert_eq!(err.code(), "ROS2-006");
}

// --- mode B key mapping -----------------------------------------------------------------------

#[test]
fn mode_b_keys_follow_the_bridge_mapping() {
    for (bridge_namespace, name, expect_key) in [
        ("/", "/chatter", "chatter"),
        ("/robot1", "/chatter", "robot1/chatter"),
    ] {
        let (node_port, bridge_port) = (reserve_port(), reserve_port());
        let bridge_peer = raw_peer(bridge_port, node_port);
        let bridge_token_key = format!("@/{}/@ros2_lv", bridge_peer.zid());
        let _bridge_token = bridge_peer
            .liveliness()
            .declare_token(&bridge_token_key)
            .wait()
            .unwrap();
        std::thread::sleep(Duration::from_millis(200));

        let node =
            Ros2Node::open(&mode_b_config(8, node_port, bridge_port, bridge_namespace)).unwrap();
        let sub = bridge_peer.declare_subscriber(expect_key).wait().unwrap();
        std::thread::sleep(Duration::from_millis(200));

        let publisher = node.publisher(name, MsgType::String, 1).unwrap();
        publisher
            .put(&Msg::String(StringMsg {
                data: "bridged".to_owned(),
            }))
            .unwrap();
        let sample = sub
            .recv_timeout(RECV_TIMEOUT)
            .unwrap()
            .expect("mapped key delivers");
        assert!(
            sample.attachment().is_none(),
            "mode B carries no attachment"
        );
    }
}

// --- actuator path (INV-12) --------------------------------------------------------------------

fn safety_plane_fixture() -> SafetyPlane<2, 1> {
    // A deliberately wide envelope (INV-12: widen, never disable) so `validate` passes the
    // action straight through as `ActionSource::Policy`.
    let envelope = Envelope {
        hard: [Limit {
            lower: -100.0,
            upper: 100.0,
        }; 2],
        soft: [Limit {
            lower: -100.0,
            upper: 100.0,
        }; 2],
        vel_max: [1e6; 2],
        acc_max: [1e9; 2],
        tau_max: [1e6; 2],
        jerk_max: None,
        d1_max: [1e6; 2],
        d2_max: [1e6; 2],
        workspace: WorkspaceSpec::Box {
            min: [-100.0; 3],
            max: [100.0; 3],
        },
        ee_velocity_max: 1.0,
        min_self_distance: 0.0,
        min_env_distance: 0.0,
        contact_force_max: 1.0,
        space: es_safety::ActionSpace::JointPosition,
        dt_s: 0.001,
        period_us: 1_000,
        execute_chunk: 1,
    };
    SafetyPlane::from_config(&SafetyConfig {
        envelope,
        watchdogs: Watchdogs::default(),
        fallback: Fallback::stationary(FallbackKind::HoldPosition),
    })
}

#[test]
fn generic_publisher_refuses_actuator_topics() {
    let (node_port, peer_port) = (reserve_port(), reserve_port());
    let actuators = vec![ActuatorTopic {
        topic: "/cmd".to_owned(),
        joints: vec!["j1".to_owned(), "j2".to_owned()],
    }];
    let node = Ros2Node::open(&mode_a_config(9, node_port, Some(peer_port), actuators)).unwrap();
    let err = node
        .publisher("/cmd", MsgType::Float64MultiArray, 1)
        .unwrap_err();
    assert_eq!(err.code(), "ROS2-010");
}

#[test]
fn actuator_publisher_sends_the_safe_action() {
    let (node_port, peer_port) = (reserve_port(), reserve_port());
    let actuators = vec![ActuatorTopic {
        topic: "/cmd".to_owned(),
        joints: vec!["j1".to_owned(), "j2".to_owned()],
    }];
    let node = Ros2Node::open(&mode_a_config(10, node_port, Some(peer_port), actuators)).unwrap();
    let observer = raw_peer(peer_port, node_port);
    let key = topic_key_expr(10, "/cmd", MsgType::Float64MultiArray).unwrap();
    let sub = observer.declare_subscriber(&key).wait().unwrap();
    std::thread::sleep(Duration::from_millis(200));

    let mut plane = safety_plane_fixture();
    plane.observe_state(&[0.0, 0.0], &[0.0, 0.0]);
    let chunk = ActionChunk::new([[1.5, -2.5]], 1, ExecutionMode::RecedingHorizon).with_seq(1);
    let action = plane.validate(&chunk, Micros(0), es_core::PhysTick(0));
    assert!(
        action.is_clean(),
        "the fixture envelope must not clamp: {action:?}"
    );

    let actuator = node.actuator::<2>("/cmd").unwrap();
    actuator.send(&action).unwrap();

    let sample = sub
        .recv_timeout(RECV_TIMEOUT)
        .unwrap()
        .expect("actuator sample");
    let msg = Float64MultiArray::from_cdr(&sample.payload().to_bytes()).unwrap();
    assert!(msg.layout.dim.is_empty());
    assert_eq!(msg.layout.data_offset, 0);
    assert_eq!(msg.data, vec![1.5, -2.5]);
}

#[test]
fn type_mismatch_is_rejected() {
    let (node_port, peer_port) = (reserve_port(), reserve_port());
    let node = Ros2Node::open(&mode_a_config(11, node_port, Some(peer_port), Vec::new())).unwrap();
    let publisher = node.publisher("/chatter", MsgType::String, 1).unwrap();
    let wrong = Msg::Float64MultiArray(Float64MultiArray {
        layout: es_ros2::msg::MultiArrayLayout {
            dim: Vec::new(),
            data_offset: 0,
        },
        data: vec![1.0],
    });
    let err = publisher.put(&wrong).unwrap_err();
    assert_eq!(err.code(), "ROS2-011");
}

#[test]
fn transient_local_is_rejected() {
    let (node_port, peer_port) = (reserve_port(), reserve_port());
    let node = Ros2Node::open(&mode_a_config(12, node_port, Some(peer_port), Vec::new())).unwrap();
    let err = node
        .publisher_with_durability("/chatter", MsgType::String, 1, Durability::TransientLocal)
        .unwrap_err();
    assert_eq!(err.code(), "ROS2-012");
}

// --- committed reference capture ---------------------------------------------------------------

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/ros2/rmw_zenoh")
}

#[test]
fn committed_reference_capture_round_trips() {
    let path = golden_dir().join("talker_capture.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIP committed_reference_capture_round_trips: {} not captured yet (run \
             capture_reference_goldens on a machine with ES_ROS2_ENV)",
            path.display()
        );
        return;
    };
    let json: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");

    for key in [
        "talker_node_token",
        "talker_pub_token",
        "listener_sub_token",
    ] {
        let Some(token_str) = json.get(key).and_then(|v| v.as_str()) else {
            continue;
        };
        let token = LivelinessToken::parse(token_str)
            .unwrap_or_else(|e| panic!("{key} `{token_str}` does not parse: {e}"));
        assert_eq!(
            token.to_key_expr(),
            token_str,
            "{key} must re-encode byte-identically"
        );
    }

    let attachments = json["attachments_hex"]
        .as_array()
        .expect("attachments_hex array");
    assert!(!attachments.is_empty());
    for hex in attachments {
        let bytes = decode_hex(hex.as_str().expect("hex string"));
        Attachment::decode(&bytes).expect("captured attachment decodes");
    }

    let payloads = json["payloads_hex"].as_array().expect("payloads_hex array");
    assert_eq!(payloads.len(), attachments.len());
    for hex in payloads {
        let bytes = decode_hex(hex.as_str().expect("hex string"));
        StringMsg::from_cdr(&bytes).expect("captured talker payload decodes as std_msgs/String");
    }

    if let Some(js_hex) = json.get("joint_state_pub_hex").and_then(|v| v.as_str()) {
        let bytes = decode_hex(js_hex);
        es_ros2::msg::JointState::from_cdr(&bytes)
            .expect("captured `ros2 topic pub` JointState payload decodes");
    }
}

fn decode_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("even-length hex"))
        .collect()
}
