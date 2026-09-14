//! ROS names, `rmw_zenoh` topic key expressions, and liveliness tokens
//! (`docs/api-notes/rmw-zenoh.md`, `docs/design/ros2-boundary.md` section 3).

use crate::msg::MsgType;

pub use crate::error::NameError;

/// Absolute ROS names only: `/`, then `/`-separated `[A-Za-z_][A-Za-z0-9_]*` segments. `~` and
/// relative names are resolved by the caller before reaching this crate.
pub fn validate_name(name: &str) -> Result<(), NameError> {
    let Some(rest) = name.strip_prefix('/') else {
        return Err(NameError::NotAbsolute(name.to_owned()));
    };
    if rest.is_empty() {
        // "/" alone: zero segments, used only as the default namespace, never a topic name.
        return Err(NameError::EmptySegment(name.to_owned()));
    }
    for seg in rest.split('/') {
        if seg.is_empty() {
            return Err(NameError::EmptySegment(name.to_owned()));
        }
        let mut chars = seg.chars();
        let first = chars.next().expect("seg is non-empty");
        let first_ok = first.is_ascii_alphabetic() || first == '_';
        let rest_ok = chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !first_ok || !rest_ok {
            return Err(NameError::InvalidSegment(name.to_owned()));
        }
    }
    Ok(())
}

/// Replaces every `/` with `%` (`liveliness_utils.cpp`'s `mangle_name`; `SLASH_REPLACEMENT`).
#[must_use]
pub fn mangle(s: &str) -> String {
    s.replace('/', "%")
}

fn unmangle(s: &str) -> String {
    s.replace('%', "/")
}

/// `<domain_id>/<name minus one leading and one trailing '/'>/<dds type>/<RIHS01>`
/// (`docs/api-notes/rmw-zenoh.md` "Topic key expression"). Inner `/` in `name` are kept, not
/// mangled: unlike a liveliness token chunk, a topic key expression's own hierarchy separator
/// is `/`, so ROS's `/` already means the same thing zenoh's does.
pub fn topic_key_expr(domain_id: u32, name: &str, ty: MsgType) -> Result<String, NameError> {
    validate_name(name)?;
    // `validate_name` guarantees a leading `/` and rejects a trailing one (an empty last
    // segment), so stripping just the leading `/` is exactly "one leading and one trailing".
    let stripped = &name[1..];
    Ok(format!(
        "{domain_id}/{stripped}/{}/{}",
        ty.dds_name(),
        ty.rihs01()
    ))
}

/// The `QoS` chunk of a liveliness token (`qos_to_keyexpr`): 11 components, each empty when it
/// equals the `rmw_zenoh` default (`RELIABLE:VOLATILE:KEEP_LAST,42:INFINITE:INFINITE:AUTOMATIC,
/// INFINITE`). `es-ros2` only ever emits defaults plus an explicit `depth`; remote components
/// (seen when parsing a token from the wire) are kept as opaque integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QosKey {
    pub reliability: Option<u64>,
    pub durability: Option<u64>,
    pub history: Option<u64>,
    pub depth: Option<u64>,
    pub deadline_sec: Option<u64>,
    pub deadline_nsec: Option<u64>,
    pub lifespan_sec: Option<u64>,
    pub lifespan_nsec: Option<u64>,
    pub liveliness: Option<u64>,
    pub lease_sec: Option<u64>,
    pub lease_nsec: Option<u64>,
}

fn fmt_opt(v: Option<u64>) -> String {
    v.map_or_else(String::new, |v| v.to_string())
}

fn parse_opt(s: &str) -> Option<u64> {
    if s.is_empty() {
        None
    } else {
        s.parse().ok()
    }
}

impl QosKey {
    /// Only an explicit `depth`, everything else default (the shape this crate's own
    /// publishers/subscribers declare): `::,<depth>:,:,:,,`.
    #[must_use]
    pub const fn with_depth(depth: u64) -> Self {
        Self {
            reliability: None,
            durability: None,
            history: None,
            depth: Some(depth),
            deadline_sec: None,
            deadline_nsec: None,
            lifespan_sec: None,
            lifespan_nsec: None,
            liveliness: None,
            lease_sec: None,
            lease_nsec: None,
        }
    }

    fn to_chunk(self) -> String {
        format!(
            "{}:{}:{},{}:{},{}:{},{}:{},{},{}",
            fmt_opt(self.reliability),
            fmt_opt(self.durability),
            fmt_opt(self.history),
            fmt_opt(self.depth),
            fmt_opt(self.deadline_sec),
            fmt_opt(self.deadline_nsec),
            fmt_opt(self.lifespan_sec),
            fmt_opt(self.lifespan_nsec),
            fmt_opt(self.liveliness),
            fmt_opt(self.lease_sec),
            fmt_opt(self.lease_nsec),
        )
    }

    fn parse(chunk: &str) -> Result<Self, NameError> {
        let top: Vec<&str> = chunk.split(':').collect();
        if top.len() != 6 {
            return Err(NameError::BadQos(chunk.to_owned()));
        }
        let (reliability, durability, hd, dl, ls, lv) =
            (top[0], top[1], top[2], top[3], top[4], top[5]);
        let hd: Vec<&str> = hd.splitn(2, ',').collect();
        let dl: Vec<&str> = dl.splitn(2, ',').collect();
        let ls: Vec<&str> = ls.splitn(2, ',').collect();
        let lv: Vec<&str> = lv.splitn(3, ',').collect();
        if hd.len() != 2 || dl.len() != 2 || ls.len() != 2 || lv.len() != 3 {
            return Err(NameError::BadQos(chunk.to_owned()));
        }
        Ok(Self {
            reliability: parse_opt(reliability),
            durability: parse_opt(durability),
            history: parse_opt(hd[0]),
            depth: parse_opt(hd[1]),
            deadline_sec: parse_opt(dl[0]),
            deadline_nsec: parse_opt(dl[1]),
            lifespan_sec: parse_opt(ls[0]),
            lifespan_nsec: parse_opt(ls[1]),
            liveliness: parse_opt(lv[0]),
            lease_sec: parse_opt(lv[1]),
            lease_nsec: parse_opt(lv[2]),
        })
    }
}

/// `NN`/`MP`/`MS`/`SS`/`SC` (`liveliness_utils.cpp`'s `NODE_STR`/`PUB_STR`/`SUB_STR`/
/// `SRV_STR`/`CLI_STR`). This crate only ever declares `Node`, `Publisher` and `Subscriber`
/// tokens (W1b); `Service`/`Client` parse only, for tokens seen on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    Node,
    Publisher,
    Subscriber,
    Service,
    Client,
}

impl EntityKind {
    const fn as_str(self) -> &'static str {
        match self {
            EntityKind::Node => "NN",
            EntityKind::Publisher => "MP",
            EntityKind::Subscriber => "MS",
            EntityKind::Service => "SS",
            EntityKind::Client => "SC",
        }
    }

    fn parse(s: &str) -> Result<Self, NameError> {
        match s {
            "NN" => Ok(EntityKind::Node),
            "MP" => Ok(EntityKind::Publisher),
            "MS" => Ok(EntityKind::Subscriber),
            "SS" => Ok(EntityKind::Service),
            "SC" => Ok(EntityKind::Client),
            _ => Err(NameError::UnknownKind(s.to_owned())),
        }
    }

    /// Whether this kind carries the `TopicName/TopicType/TopicTypeHash/TopicQoS` chunks
    /// (`MP`/`MS`/`SS`/`SC`) or not (`NN`, a bare node token).
    const fn has_topic(self) -> bool {
        !matches!(self, EntityKind::Node)
    }
}

/// The `TopicName/TopicType/TopicTypeHash/TopicQoS` chunks of a `MP`/`MS`/`SS`/`SC` token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicPart {
    pub name: String,
    pub type_name: String,
    pub type_hash: String,
    pub qos: QosKey,
}

/// A parsed or to-be-declared `rmw_zenoh` liveliness token
/// (`docs/api-notes/rmw-zenoh.md` "Liveliness tokens").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivelinessToken {
    pub domain_id: u32,
    pub zid: String,
    pub node_id: u64,
    pub entity_id: u64,
    pub kind: EntityKind,
    pub enclave: String,
    pub namespace: String,
    pub node_name: String,
    pub topic: Option<TopicPart>,
    /// Trailing chunks this crate does not interpret (rolling's optional `/<backends>` suffix):
    /// kept verbatim so a round trip through [`Self::parse`] / [`Self::to_key_expr`] is lossless.
    pub extra: Vec<String>,
}

impl LivelinessToken {
    #[must_use]
    pub fn to_key_expr(&self) -> String {
        let mut parts = vec![
            "@ros2_lv".to_owned(),
            self.domain_id.to_string(),
            self.zid.clone(),
            self.node_id.to_string(),
            self.entity_id.to_string(),
            self.kind.as_str().to_owned(),
            mangle(&self.enclave),
            mangle(&self.namespace),
            mangle(&self.node_name),
        ];
        if let Some(topic) = &self.topic {
            parts.push(mangle(&topic.name));
            parts.push(mangle(&topic.type_name));
            parts.push(mangle(&topic.type_hash));
            parts.push(topic.qos.to_chunk());
        }
        parts.extend(self.extra.iter().cloned());
        parts.join("/")
    }

    pub fn parse(key: &str) -> Result<Self, NameError> {
        let chunks: Vec<&str> = key.split('/').collect();
        if chunks.first() != Some(&"@ros2_lv") {
            return Err(NameError::NotALivelinessToken(key.to_owned()));
        }
        if chunks.len() < 9 {
            return Err(NameError::TooFewChunks(key.to_owned()));
        }
        let domain_id: u32 = chunks[1]
            .parse()
            .map_err(|_| NameError::NotAnInteger(chunks[1].to_owned()))?;
        let zid = chunks[2].to_owned();
        let node_id: u64 = chunks[3]
            .parse()
            .map_err(|_| NameError::NotAnInteger(chunks[3].to_owned()))?;
        let entity_id: u64 = chunks[4]
            .parse()
            .map_err(|_| NameError::NotAnInteger(chunks[4].to_owned()))?;
        let kind = EntityKind::parse(chunks[5])?;
        let enclave = unmangle(chunks[6]);
        let namespace = unmangle(chunks[7]);
        let node_name = unmangle(chunks[8]);

        let (topic, extra_start) = if kind.has_topic() {
            if chunks.len() < 13 {
                return Err(NameError::TooFewChunks(key.to_owned()));
            }
            let topic = TopicPart {
                name: unmangle(chunks[9]),
                type_name: unmangle(chunks[10]),
                type_hash: unmangle(chunks[11]),
                qos: QosKey::parse(chunks[12])?,
            };
            (Some(topic), 13)
        } else {
            (None, 9)
        };
        let extra = chunks[extra_start..]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();

        Ok(Self {
            domain_id,
            zid,
            node_id,
            entity_id,
            kind,
            enclave,
            namespace,
            node_name,
            topic,
            extra,
        })
    }
}

/// Which liveliness token family a key expression belongs to
/// (`docs/api-notes/rmw-zenoh.md` "Mode discriminator"): `@` chunks are verbatim in zenoh key
/// expressions (wildcards never match them), so the two schemes' prefixes cannot collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenScheme {
    /// `@ros2_lv/<domain>/...` (mode A, this crate's own tokens).
    RmwZenoh,
    /// `@/<zenoh_id>/@ros2_lv/...` (`zenoh-plugin-ros2dds`, mode B).
    Ros2DdsBridge,
    Other,
}

#[must_use]
pub fn classify(key: &str) -> TokenScheme {
    let mut parts = key.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("@ros2_lv"), _, _) => TokenScheme::RmwZenoh,
        (Some("@"), Some(_zid), Some("@ros2_lv")) => TokenScheme::Ros2DdsBridge,
        _ => TokenScheme::Other,
    }
}
