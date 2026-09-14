//! The zenoh session: mode A (`rmw_zenoh`), the A xor B startup probe, and the generic
//! publisher/subscriber path (`docs/design/ros2-boundary.md` section 4, spec 24.1). Built only
//! with the `zenoh` feature: everything else in this crate compiles and is tested without it
//! (design note section 2).

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use zenoh::Wait;

use crate::attachment::{gid_of, Attachment};
use crate::config::{Ros2Config, Ros2Mode, ZenohEndpoints, ZenohMode};
pub use crate::error::Ros2Error;
use crate::msg::{Msg, MsgType};
use crate::names::{self, EntityKind, LivelinessToken, QosKey, TopicPart};

/// `TRANSIENT_LOCAL` is refused (`ROS2-012`); this crate only ever declares `VOLATILE`. A plain
/// enum, not a new trait (INV-17), so [`Ros2Node::publisher_with_durability`] can make the
/// refusal testable without a speculative extra parameter on [`Ros2Node::publisher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    Volatile,
    TransientLocal,
}

fn zerr<E: std::fmt::Display>(e: E) -> Ros2Error {
    Ros2Error::Zenoh(e.to_string())
}

fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_nanos()).unwrap_or(i64::MAX))
}

/// A JSON5 array of quoted strings, for `Config::insert_json5`'s `connect/endpoints` and
/// `listen/endpoints` (`docs/api-notes/zenoh-rs.md` "API used").
fn json5_str_array(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("{s:?}")).collect();
    format!("[{}]", inner.join(","))
}

fn zenoh_config(endpoints: &ZenohEndpoints) -> Result<zenoh::Config, Ros2Error> {
    let mode = match endpoints.mode {
        ZenohMode::Peer => "peer",
        ZenohMode::Client => "client",
    };
    let mut cfg = zenoh::Config::default();
    cfg.insert_json5("mode", &format!("{mode:?}"))
        .map_err(zerr)?;
    cfg.insert_json5("connect/endpoints", &json5_str_array(&endpoints.connect))
        .map_err(zerr)?;
    if !endpoints.listen.is_empty() {
        cfg.insert_json5("listen/endpoints", &json5_str_array(&endpoints.listen))
            .map_err(zerr)?;
    }
    // Design note section 2: no outside network by default (spec 25.1).
    cfg.insert_json5("scouting/multicast/enabled", "false")
        .map_err(zerr)?;
    Ok(cfg)
}

/// The first liveliness token seen matching `pattern` within `timeout`, or `None`
/// (`docs/design/ros2-boundary.md` section 4.2's probe). Every wait is bounded (the harness
/// rule this module must also follow): both the query itself and the receive loop are capped
/// by `timeout`.
fn liveliness_any(
    session: &zenoh::Session,
    pattern: &str,
    timeout: Duration,
) -> Result<Option<String>, Ros2Error> {
    let replies = session
        .liveliness()
        .get(pattern)
        .timeout(timeout)
        .wait()
        .map_err(zerr)?;
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(None);
        }
        match replies.recv_timeout(remaining) {
            Ok(Some(reply)) => {
                if let Ok(sample) = reply.result() {
                    return Ok(Some(sample.key_expr().as_str().to_owned()));
                }
                // An error reply: keep draining until a real token or the deadline.
            }
            Ok(None) | Err(_) => return Ok(None),
        }
    }
}

/// Every liveliness token matching `pattern`, seen within `timeout` (bounded, like every wait
/// in this module). Unlike [`liveliness_any`] this always drains the full query window, so it
/// is for callers who want the whole set (e.g. the live interop tests picking a specific remote
/// node's tokens out of several), not the startup probe's fast "does one exist" check.
fn liveliness_all(
    session: &zenoh::Session,
    pattern: &str,
    timeout: Duration,
) -> Result<Vec<String>, Ros2Error> {
    let replies = session
        .liveliness()
        .get(pattern)
        .timeout(timeout)
        .wait()
        .map_err(zerr)?;
    let deadline = Instant::now() + timeout;
    let mut out = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(out);
        }
        match replies.recv_timeout(remaining) {
            Ok(Some(reply)) => {
                if let Ok(sample) = reply.result() {
                    out.push(sample.key_expr().as_str().to_owned());
                }
            }
            Ok(None) | Err(_) => return Ok(out),
        }
    }
}

/// A live `rmw_zenoh`-compatible session (mode A) or a `zenoh-plugin-ros2dds` bridge session
/// (mode B). One node identity (`NN` token in mode A), one shared entity-id counter, one
/// startup probe, one latch that fails every later `put`/`send` closed if a conflicting token
/// appears afterward (`ROS2-006`).
pub struct Ros2Node {
    session: zenoh::Session,
    config: Ros2Config,
    zid: String,
    node_id: u64,
    next_entity_id: Arc<AtomicU64>,
    poisoned: Arc<AtomicBool>,
    _node_token: Option<zenoh::liveliness::LivelinessToken>,
    _conflict_watch: zenoh::pubsub::Subscriber<()>,
}

// zenoh's `Session`/`Subscriber` do not implement `Debug`; a manual, field-free impl satisfies
// the workspace's `missing_debug_implementations` lint without pretending to show internals.
impl std::fmt::Debug for Ros2Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ros2Node").finish_non_exhaustive()
    }
}

impl Ros2Node {
    /// Opens the zenoh session, runs the A xor B startup probe (`ROS2-003`..`ROS2-005`) before
    /// declaring anything of its own, then declares the node's own `NN` token (mode A) and arms
    /// the fail-closed watch for a conflicting token appearing later (`ROS2-006`).
    pub fn open(cfg: &Ros2Config) -> Result<Self, Ros2Error> {
        let endpoints = match &cfg.mode {
            Ros2Mode::RmwZenoh(z) => z,
            Ros2Mode::DdsBridge { endpoints, .. } => endpoints,
        };
        let zconfig = zenoh_config(endpoints)?;
        let session = zenoh::open(zconfig).wait().map_err(zerr)?;

        match &cfg.mode {
            Ros2Mode::RmwZenoh(_) => {
                if let Some(k) =
                    liveliness_any(&session, "@/*/@ros2_lv/**", cfg.liveliness_timeout)?
                {
                    return Err(Ros2Error::ModeAFoundBridge(k));
                }
            }
            Ros2Mode::DdsBridge { .. } => {
                if liveliness_any(&session, "@/*/@ros2_lv", cfg.liveliness_timeout)?.is_none() {
                    return Err(Ros2Error::ModeBNoBridge);
                }
                let conflict = format!("@ros2_lv/{}/**", cfg.domain_id);
                if let Some(k) = liveliness_any(&session, &conflict, cfg.liveliness_timeout)? {
                    return Err(Ros2Error::ModeBFoundRmwZenoh(k));
                }
            }
        }

        let next_entity_id = Arc::new(AtomicU64::new(0));
        let zid = session.zid().to_string();

        let (node_id, node_token) = match &cfg.mode {
            Ros2Mode::RmwZenoh(_) => {
                let id = next_entity_id.fetch_add(1, Ordering::SeqCst);
                let key = LivelinessToken {
                    domain_id: cfg.domain_id,
                    zid: zid.clone(),
                    node_id: id,
                    entity_id: id,
                    kind: EntityKind::Node,
                    enclave: cfg.enclave.clone(),
                    namespace: cfg.namespace.clone(),
                    node_name: cfg.node.clone(),
                    topic: None,
                    extra: Vec::new(),
                }
                .to_key_expr();
                let token = session
                    .liveliness()
                    .declare_token(key)
                    .wait()
                    .map_err(zerr)?;
                (id, Some(token))
            }
            Ros2Mode::DdsBridge { .. } => (0, None),
        };

        let poisoned = Arc::new(AtomicBool::new(false));
        let watch_pattern = match &cfg.mode {
            Ros2Mode::RmwZenoh(_) => "@/*/@ros2_lv/**".to_owned(),
            Ros2Mode::DdsBridge { .. } => format!("@ros2_lv/{}/**", cfg.domain_id),
        };
        let poisoned_cb = poisoned.clone();
        let conflict_watch = session
            .liveliness()
            .declare_subscriber(watch_pattern)
            .callback(move |_sample| {
                poisoned_cb.store(true, Ordering::SeqCst);
            })
            .wait()
            .map_err(zerr)?;

        Ok(Self {
            session,
            config: cfg.clone(),
            zid,
            node_id,
            next_entity_id,
            poisoned,
            _node_token: node_token,
            _conflict_watch: conflict_watch,
        })
    }

    pub fn config(&self) -> &Ros2Config {
        &self.config
    }

    /// Every liveliness token this session can currently see matching `pattern` (e.g. a live
    /// interop test round-tripping a remote node's `MP` token). Not part of the mode A/B
    /// session logic itself — [`Self::open`]'s own probe uses the faster, first-match
    /// `liveliness_any` internally.
    pub fn liveliness_tokens(
        &self,
        pattern: &str,
        timeout: Duration,
    ) -> Result<Vec<String>, Ros2Error> {
        liveliness_all(&self.session, pattern, timeout)
    }

    fn topic_key(&self, name: &str, ty: MsgType) -> Result<String, Ros2Error> {
        match &self.config.mode {
            Ros2Mode::RmwZenoh(_) => names::topic_key_expr(self.config.domain_id, name, ty)
                .map_err(|e| Ros2Error::Config(e.to_string())),
            Ros2Mode::DdsBridge {
                bridge_namespace, ..
            } => Ok(mode_b_key(bridge_namespace, name)),
        }
    }

    fn entity_token_key(
        &self,
        kind: EntityKind,
        entity_id: u64,
        name: &str,
        ty: MsgType,
        depth: u32,
    ) -> String {
        LivelinessToken {
            domain_id: self.config.domain_id,
            zid: self.zid.clone(),
            node_id: self.node_id,
            entity_id,
            kind,
            enclave: self.config.enclave.clone(),
            namespace: self.config.namespace.clone(),
            node_name: self.config.node.clone(),
            topic: Some(TopicPart {
                name: name.to_owned(),
                type_name: ty.dds_name().to_owned(),
                type_hash: ty.rihs01().to_owned(),
                qos: QosKey::with_depth(u64::from(depth)),
            }),
            extra: Vec::new(),
        }
        .to_key_expr()
    }

    /// The mode A / mode B publisher path, without the reserved-actuator-topic check
    /// ([`Self::publisher`] adds that; [`crate::actuator`]'s constructor targets the reserved
    /// topic on purpose and calls this directly).
    pub(crate) fn declare_publisher_raw(
        &self,
        name: &str,
        ty: MsgType,
        depth: u32,
    ) -> Result<Publisher, Ros2Error> {
        names::validate_name(name).map_err(|e| Ros2Error::Config(e.to_string()))?;
        let key = self.topic_key(name, ty)?;
        let zpub = self
            .session
            .declare_publisher(key.clone())
            .wait()
            .map_err(zerr)?;
        let (gid, attach, mp_token) = match &self.config.mode {
            Ros2Mode::RmwZenoh(_) => {
                let entity_id = self.next_entity_id.fetch_add(1, Ordering::SeqCst);
                let token_key =
                    self.entity_token_key(EntityKind::Publisher, entity_id, name, ty, depth);
                let token = self
                    .session
                    .liveliness()
                    .declare_token(token_key.clone())
                    .wait()
                    .map_err(zerr)?;
                (gid_of(&token_key), true, Some(token))
            }
            Ros2Mode::DdsBridge { .. } => ([0u8; 16], false, None),
        };
        Ok(Publisher {
            inner: zpub,
            ty,
            seq: AtomicI64::new(1),
            gid,
            attach,
            poisoned: self.poisoned.clone(),
            _mp_token: mp_token,
        })
    }

    /// Refuses a `[[ros2.actuator]]` topic (`ROS2-010`): use [`Self::actuator`] instead.
    pub fn publisher(&self, name: &str, ty: MsgType, depth: u32) -> Result<Publisher, Ros2Error> {
        self.publisher_with_durability(name, ty, depth, Durability::Volatile)
    }

    /// [`Self::publisher`] with an explicit [`Durability`]. `TransientLocal` always fails with
    /// `ROS2-012` (design note section 4.3: the zenoh-ext advanced-pub/sub key literals it
    /// needs are unverified).
    pub fn publisher_with_durability(
        &self,
        name: &str,
        ty: MsgType,
        depth: u32,
        durability: Durability,
    ) -> Result<Publisher, Ros2Error> {
        if durability == Durability::TransientLocal {
            return Err(Ros2Error::TransientLocalUnsupported);
        }
        if self.config.is_actuator_topic(name) {
            return Err(Ros2Error::ReservedActuatorTopic(name.to_owned()));
        }
        self.declare_publisher_raw(name, ty, depth)
    }

    pub fn subscriber(&self, name: &str, ty: MsgType, depth: u32) -> Result<Subscriber, Ros2Error> {
        names::validate_name(name).map_err(|e| Ros2Error::Config(e.to_string()))?;
        let key = self.topic_key(name, ty)?;
        let cap = usize::try_from(depth.max(1)).unwrap_or(usize::MAX);
        let zsub = self
            .session
            .declare_subscriber(key.clone())
            .with(zenoh::handlers::FifoChannel::new(cap))
            .wait()
            .map_err(zerr)?;
        let (require_attachment, ms_token) = match &self.config.mode {
            Ros2Mode::RmwZenoh(_) => {
                let entity_id = self.next_entity_id.fetch_add(1, Ordering::SeqCst);
                let token_key =
                    self.entity_token_key(EntityKind::Subscriber, entity_id, name, ty, depth);
                let token = self
                    .session
                    .liveliness()
                    .declare_token(token_key)
                    .wait()
                    .map_err(zerr)?;
                (true, Some(token))
            }
            Ros2Mode::DdsBridge { .. } => (false, None),
        };
        Ok(Subscriber {
            inner: zsub,
            ty,
            dropped: AtomicU64::new(0),
            require_attachment,
            _ms_token: ms_token,
        })
    }
}

/// Mode B's key mapping (`docs/api-notes/rmw-zenoh.md` "Topic mapping",
/// `zenoh-plugin-ros2dds`'s `ros2_name_to_key_expr`): the bridge's own namespace, then `name`,
/// each with its leading `/` stripped.
pub(crate) fn mode_b_key(bridge_namespace: &str, name: &str) -> String {
    let name = name.strip_prefix('/').unwrap_or(name);
    if bridge_namespace.is_empty() || bridge_namespace == "/" {
        name.to_owned()
    } else {
        let ns = bridge_namespace
            .strip_prefix('/')
            .unwrap_or(bridge_namespace);
        format!("{ns}/{name}")
    }
}

/// A declared publisher (mode A: with its `MP` liveliness token; mode B: bare). `put` refuses a
/// message of the wrong type (`ROS2-011`) and fails closed once the node's startup-probe watch
/// has latched a conflict (`ROS2-006`).
pub struct Publisher {
    inner: zenoh::pubsub::Publisher<'static>,
    ty: MsgType,
    seq: AtomicI64,
    gid: [u8; 16],
    attach: bool,
    poisoned: Arc<AtomicBool>,
    _mp_token: Option<zenoh::liveliness::LivelinessToken>,
}

impl std::fmt::Debug for Publisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Publisher")
            .field("ty", &self.ty)
            .finish_non_exhaustive()
    }
}

impl Publisher {
    pub fn put(&self, msg: &Msg) -> Result<(), Ros2Error> {
        if self.poisoned.load(Ordering::SeqCst) {
            return Err(Ros2Error::LatchedConflict(
                "a conflicting liveliness token latched this node after startup".to_owned(),
            ));
        }
        if msg.msg_type() != self.ty {
            return Err(Ros2Error::TypeMismatch);
        }
        let bytes = msg.to_cdr();
        let builder = self.inner.put(bytes);
        if self.attach {
            let seq = self.seq.fetch_add(1, Ordering::SeqCst);
            let attachment = Attachment {
                seq,
                source_timestamp_ns: now_ns(),
                gid: self.gid,
            }
            .encode();
            builder.attachment(attachment.to_vec()).wait().map_err(zerr)
        } else {
            builder.wait().map_err(zerr)
        }
    }
}

/// A decoded sample plus the attachment it carried (mode A: always present, or the sample is
/// dropped before reaching here) and the key expression it arrived on.
#[derive(Debug)]
pub struct Received {
    pub msg: Msg,
    pub attachment: Option<Attachment>,
    pub key_expr: String,
}

/// A declared subscriber. Mode A drops and counts a sample with no well-formed 33-byte
/// attachment, exactly as `rmw_zenoh` does (design note section 4.3); mode B never expects one.
pub struct Subscriber {
    inner: zenoh::pubsub::Subscriber<zenoh::handlers::FifoChannelHandler<zenoh::sample::Sample>>,
    ty: MsgType,
    dropped: AtomicU64,
    require_attachment: bool,
    _ms_token: Option<zenoh::liveliness::LivelinessToken>,
}

impl std::fmt::Debug for Subscriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subscriber")
            .field("ty", &self.ty)
            .field("dropped", &self.dropped.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl Subscriber {
    /// Blocks up to `t` for the next sample that decodes cleanly, draining and counting
    /// malformed ones along the way. `Ok(None)` on a plain timeout (spec 25.1: network bytes
    /// are untrusted, so a bad sample is a dropped counter, never a returned error).
    pub fn recv_timeout(&self, t: Duration) -> Result<Option<Received>, Ros2Error> {
        let deadline = Instant::now() + t;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            let Ok(Some(sample)) = self.inner.recv_timeout(remaining) else {
                return Ok(None);
            };
            let attachment = match sample.attachment() {
                Some(bytes) => {
                    if let Ok(a) = Attachment::decode(&bytes.to_bytes()) {
                        Some(a)
                    } else {
                        self.dropped.fetch_add(1, Ordering::SeqCst);
                        continue;
                    }
                }
                None if self.require_attachment => {
                    self.dropped.fetch_add(1, Ordering::SeqCst);
                    continue;
                }
                None => None,
            };
            let payload = sample.payload().to_bytes();
            let Ok(msg) = Msg::from_cdr(self.ty, &payload) else {
                self.dropped.fetch_add(1, Ordering::SeqCst);
                continue;
            };
            return Ok(Some(Received {
                msg,
                attachment,
                key_expr: sample.key_expr().as_str().to_owned(),
            }));
        }
    }

    /// Samples dropped for lacking a well-formed attachment (mode A) or failing to decode.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::mode_b_key;

    #[test]
    fn mode_b_keys_strip_the_leading_slash_and_prepend_the_bridge_namespace() {
        assert_eq!(mode_b_key("/", "/chatter"), "chatter");
        assert_eq!(mode_b_key("/robot1", "/chatter"), "robot1/chatter");
    }
}
