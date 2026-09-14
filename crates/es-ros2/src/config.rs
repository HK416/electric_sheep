//! `Ros2Config` and its TOML shape (`docs/design/ros2-boundary.md` section 4.1, 4.2).
//!
//! Parsing needs no `zenoh` feature: it is plain data, checked the same way on any machine
//! (the `config.rs` test file runs without the feature). `Ros2Node::open` (session.rs, `zenoh`
//! feature only) is the only thing that turns this into a live session.

use std::time::Duration;

use serde::Deserialize;

pub use crate::error::Ros2Error;

/// `DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5`'s connect endpoint (`docs/api-notes/rmw-zenoh.md`
/// "Default configuration"): mode A's default when `[ros2.rmw_zenoh]` omits `connect`, or the
/// whole table is absent.
const DEFAULT_CONNECT: &str = "tcp/localhost:7447";
/// Same table's default `listen.endpoints`.
const DEFAULT_LISTEN: &str = "tcp/localhost:0";

/// `peer` (default, discovers other peers and routers) or `client` (only ever connects out).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZenohMode {
    Peer,
    Client,
}

/// The zenoh transport endpoints a session opens with (design note section 4.1's
/// `[ros2.rmw_zenoh]` / `[ros2.dds_bridge]` tables).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZenohEndpoints {
    pub mode: ZenohMode,
    pub connect: Vec<String>,
    pub listen: Vec<String>,
}

/// Mode A (`rmw_zenoh`, the default) or mode B (a `zenoh-plugin-ros2dds` bridge). Mode C
/// (`RustDDS`) is parsed only far enough to reject it specifically (`ROS2-002`); it has no
/// variant here because nothing is ever built from it (spec 24.1 marks it experimental, design
/// note section 1: "not built").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ros2Mode {
    RmwZenoh(ZenohEndpoints),
    DdsBridge {
        endpoints: ZenohEndpoints,
        bridge_namespace: String,
    },
}

/// One `[[ros2.actuator]]` entry: a reserved topic and the joint order its `Float64MultiArray`
/// command rows are written in (design note section 4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActuatorTopic {
    pub topic: String,
    pub joints: Vec<String>,
}

/// The parsed `[ros2]` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ros2Config {
    pub domain_id: u32,
    pub node: String,
    pub namespace: String,
    pub enclave: String,
    pub liveliness_timeout: Duration,
    pub mode: Ros2Mode,
    pub actuators: Vec<ActuatorTopic>,
}

impl Ros2Config {
    /// Looks up a configured actuator topic and checks its joint count against `NJ`
    /// (design note section 4.5: "`NJ != joints.len()` fails at construction"). Used by
    /// `Ros2Node::actuator` (session.rs, `zenoh` feature) and directly testable without it.
    pub fn actuator_topic<const NJ: usize>(
        &self,
        topic: &str,
    ) -> Result<&ActuatorTopic, Ros2Error> {
        let found = self
            .actuators
            .iter()
            .find(|a| a.topic == topic)
            .ok_or_else(|| Ros2Error::UnknownActuatorTopic(topic.to_owned()))?;
        if found.joints.len() != NJ {
            return Err(Ros2Error::ActuatorJointCount {
                topic: topic.to_owned(),
                expected: NJ,
                found: found.joints.len(),
            });
        }
        Ok(found)
    }

    /// Whether `topic` is reserved for an actuator publisher (`ROS2-010`): a generic
    /// [`crate::session::Publisher`] must refuse it.
    #[must_use]
    pub fn is_actuator_topic(&self, topic: &str) -> bool {
        self.actuators.iter().any(|a| a.topic == topic)
    }
}

// --- TOML shape (private: the public API is `Ros2Config`, never the raw tables) ---------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    ros2: RawRos2,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRos2 {
    domain_id: u32,
    node: String,
    namespace: String,
    enclave: String,
    liveliness_timeout_ms: u64,
    rmw_zenoh: Option<RawZenohEndpoints>,
    dds_bridge: Option<RawDdsBridge>,
    /// Parsed as an opaque table so an unknown key *inside* `[ros2.rust_dds]` does not itself
    /// mask the `ROS2-002` rejection with a `ROS2-000` one; only its presence matters.
    rust_dds: Option<toml::Value>,
    #[serde(default, rename = "actuator")]
    actuator: Vec<RawActuatorTopic>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawZenohEndpoints {
    mode: Option<String>,
    connect: Option<Vec<String>>,
    listen: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDdsBridge {
    connect: Option<Vec<String>>,
    bridge_namespace: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawActuatorTopic {
    topic: String,
    joints: Vec<String>,
}

fn parse_mode(mode: Option<&str>) -> Result<ZenohMode, Ros2Error> {
    match mode {
        None | Some("peer") => Ok(ZenohMode::Peer),
        Some("client") => Ok(ZenohMode::Client),
        Some(other) => Err(Ros2Error::Config(format!(
            "ros2.rmw_zenoh.mode must be \"peer\" or \"client\", got {other:?}"
        ))),
    }
}

impl RawZenohEndpoints {
    fn into_endpoints(self) -> Result<ZenohEndpoints, Ros2Error> {
        Ok(ZenohEndpoints {
            mode: parse_mode(self.mode.as_deref())?,
            connect: self
                .connect
                .unwrap_or_else(|| vec![DEFAULT_CONNECT.to_owned()]),
            listen: self
                .listen
                .unwrap_or_else(|| vec![DEFAULT_LISTEN.to_owned()]),
        })
    }
}

/// Parses a `[ros2]` TOML document into a [`Ros2Config`] (design note section 4.1). Unknown keys
/// are rejected (`#[serde(deny_unknown_fields)]` at every level); `[ros2.rmw_zenoh]` implied
/// when no mode table is present at all mirrors `DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5` (spec
/// 24.1).
pub fn parse_config(toml_text: &str) -> Result<Ros2Config, Ros2Error> {
    let raw: RawFile = toml::from_str(toml_text).map_err(|e| Ros2Error::Config(e.to_string()))?;
    let r = raw.ros2;

    let mode = match (r.rust_dds, r.rmw_zenoh, r.dds_bridge) {
        (Some(_), _, _) => return Err(Ros2Error::ModeCNotBuilt),
        (None, Some(_), Some(_)) => return Err(Ros2Error::BothModesConfigured),
        (None, Some(a), None) => Ros2Mode::RmwZenoh(a.into_endpoints()?),
        (None, None, Some(b)) => Ros2Mode::DdsBridge {
            endpoints: ZenohEndpoints {
                mode: ZenohMode::Peer,
                connect: b
                    .connect
                    .unwrap_or_else(|| vec![DEFAULT_CONNECT.to_owned()]),
                listen: Vec::new(),
            },
            bridge_namespace: b.bridge_namespace,
        },
        (None, None, None) => Ros2Mode::RmwZenoh(ZenohEndpoints {
            mode: ZenohMode::Peer,
            connect: vec![DEFAULT_CONNECT.to_owned()],
            listen: vec![DEFAULT_LISTEN.to_owned()],
        }),
    };

    let mut seen_topics: Vec<&str> = Vec::new();
    let mut actuators = Vec::with_capacity(r.actuator.len());
    for a in &r.actuator {
        if a.joints.is_empty() {
            return Err(Ros2Error::Config(format!(
                "ros2.actuator {:?} has no joints",
                a.topic
            )));
        }
        if seen_topics.contains(&a.topic.as_str()) {
            return Err(Ros2Error::Config(format!(
                "duplicate ros2.actuator topic {:?}",
                a.topic
            )));
        }
        seen_topics.push(&a.topic);
        actuators.push(ActuatorTopic {
            topic: a.topic.clone(),
            joints: a.joints.clone(),
        });
    }

    Ok(Ros2Config {
        domain_id: r.domain_id,
        node: r.node,
        namespace: r.namespace,
        enclave: r.enclave,
        liveliness_timeout: Duration::from_millis(r.liveliness_timeout_ms),
        mode,
        actuators,
    })
}
