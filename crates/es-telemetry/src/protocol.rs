//! Wire protocol skeleton (P40): handshake, version negotiation, and the message/frame schema
//! (spec 23, spec 25.1, spec 25.3, spec 12.4, spec 13.1).
//!
//! This module defines the schema and a length-prefixed encoding only. There is no transport
//! here — no sockets, no QUIC, no TLS (`es-transport`, layer 11, owns that; spec 25.1 assigns
//! it token auth, QUIC TLS, and a localhost-default bind).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use es_core::PhysTick;

/// Current protocol version (spec 25.3: handshake negotiation, server supports N and N-1).
pub const PROTOCOL_VERSION: u32 = 1;

/// Default max encoded frame size: 64 MiB.
pub const DEFAULT_MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024;

/// Which telemetry stream a [`Frame`] belongs to (state stream, image stream, event log, ...;
/// spec 23.3 lists several kinds sharing one wire format).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct StreamId(pub u32);

/// Client to server handshake open (spec 25.1: token auth; spec 25.3: version negotiation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hello {
    pub versions_supported: Vec<u32>,
    pub token: Option<String>,
    pub client: String,
}

/// Server to client handshake reply. `execution_hash` is the hash-chain identity (spec 5.3) of
/// the run being observed, when the server has one yet (e.g. before a task is loaded).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HelloAck {
    pub version: u32,
    pub session_id: u64,
    pub execution_hash: Option<[u8; 32]>,
}

/// Highest protocol version both sides support, or `None` if they share none (spec 25.3: a
/// server must offer at least its current version and the one before it — callers pass that
/// pair, or more, as `server`).
pub fn negotiate(client: &[u32], server: &[u32]) -> Option<u32> {
    client.iter().filter(|v| server.contains(v)).max().copied()
}

/// One control or data message on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    Hello(Hello),
    HelloAck(HelloAck),
    Subscribe { streams: Vec<StreamId> },
    Frame(Frame),
    Ping,
    Pong,
    Bye,
}

/// One sample on a stream: when it happened (sim tick and wall clock) and what it carries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub tick: PhysTick,
    pub wall_ns: u64,
    pub stream: StreamId,
    pub payload: Payload,
}

/// Frame content. Adjacently tagged (`kind`/`value`) rather than internally tagged because
/// several variants — `Scalar`, `Scalars` — do not serialize as a JSON object, which an
/// internally tagged enum requires.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Payload {
    Scalar(f64),
    Scalars(Vec<f64>),
    Tensor {
        shape: Vec<u32>,
        dtype: String,
        bytes: Vec<u8>,
    },
    Image {
        w: u32,
        h: u32,
        format: String,
        bytes: Vec<u8>,
    },
    Event {
        kind: String,
        fields: BTreeMap<String, String>,
    },
    Metrics(PerfMetrics),
}

/// The spec 12.4 performance metric set. `step/s` alone is forbidden (spec 12.4): simulation,
/// observation, inference and control throughput are independent numbers. Every field is
/// `Option<f64>` — a metric the caller has not measured is `None`, never a fabricated zero.
/// The end-to-end latency row of spec 12.4 (`p50 / p95 end_to_end_latency`) becomes two fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfMetrics {
    pub physics_steps_per_sec: Option<f64>,
    pub camera_frames_per_sec: Option<f64>,
    pub pixels_per_sec: Option<f64>,
    pub observation_gb_per_sec: Option<f64>,
    pub policy_inferences_per_sec: Option<f64>,
    pub actions_per_sec: Option<f64>,
    pub p50_end_to_end_latency: Option<f64>,
    pub p95_end_to_end_latency: Option<f64>,
    pub gpu_memory_peak: Option<f64>,
    pub chunk_underrun_rate: Option<f64>,
}

/// Body encoding for a [`Message`]. `Json` is the only one today; a binary codec is a new
/// variant plus a match arm, not an API change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Codec {
    #[default]
    Json = 0,
}

impl Codec {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Json),
            _ => None,
        }
    }
}

/// Errors from encoding, decoding, or framing.
#[derive(Debug, Error)]
pub enum ProtoError {
    #[error("frame body of {size} bytes exceeds the {max} byte limit")]
    TooLarge { size: u32, max: u32 },
    #[error("buffer has {have} bytes, need at least {need}")]
    Incomplete { have: usize, need: usize },
    #[error("byte {0} is not a known codec/wire-version tag")]
    UnknownCodec(u8),
    #[error("JSON codec error: {0}")]
    Json(#[from] serde_json::Error),
}

/// `codec` (1 byte) + body length (4 bytes, little-endian `u32`) + body.
const HEADER_LEN: usize = 5;

/// Encodes `msg` as a length-prefixed JSON frame.
pub fn encode(msg: &Message) -> Vec<u8> {
    encode_with(Codec::Json, msg)
}

/// Encodes `msg` with an explicit [`Codec`].
pub fn encode_with(codec: Codec, msg: &Message) -> Vec<u8> {
    let body = match codec {
        Codec::Json => serde_json::to_vec(msg).expect("Message always serializes to JSON"),
    };
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.push(codec as u8);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Decodes one length-prefixed frame from the front of `buf`, using
/// [`DEFAULT_MAX_FRAME_BYTES`]. Returns the message and the number of bytes it consumed, so the
/// caller can advance past it and try again on the remainder.
pub fn decode(buf: &[u8]) -> Result<(Message, usize), ProtoError> {
    decode_with_max(buf, DEFAULT_MAX_FRAME_BYTES)
}

/// Decodes one length-prefixed frame, rejecting bodies larger than `max` bytes and unknown
/// codec/wire-version tags before touching the body.
pub fn decode_with_max(buf: &[u8], max: u32) -> Result<(Message, usize), ProtoError> {
    if buf.len() < HEADER_LEN {
        return Err(ProtoError::Incomplete {
            have: buf.len(),
            need: HEADER_LEN,
        });
    }
    let codec = Codec::from_byte(buf[0]).ok_or(ProtoError::UnknownCodec(buf[0]))?;
    let len = u32::from_le_bytes(buf[1..5].try_into().expect("4-byte slice"));
    if len > max {
        return Err(ProtoError::TooLarge { size: len, max });
    }
    let total = HEADER_LEN + len as usize;
    if buf.len() < total {
        return Err(ProtoError::Incomplete {
            have: buf.len(),
            need: total,
        });
    }
    let body = &buf[HEADER_LEN..total];
    let msg = match codec {
        Codec::Json => serde_json::from_slice(body)?,
    };
    Ok((msg, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negotiate_picks_highest_common_version() {
        assert_eq!(negotiate(&[1, 2, 3], &[2, 3, 4]), Some(3));
        assert_eq!(negotiate(&[1], &[1, 2]), Some(1)); // client on N-1
        assert_eq!(negotiate(&[2], &[1, 2]), Some(2)); // client on N
        assert_eq!(negotiate(&[1], &[3, 4]), None); // shares nothing
        assert_eq!(negotiate(&[], &[1]), None);
    }

    fn sample_messages() -> Vec<Message> {
        vec![
            Message::Hello(Hello {
                versions_supported: vec![1, 2],
                token: Some("tok".into()),
                client: "es-editor".into(),
            }),
            Message::Hello(Hello {
                versions_supported: vec![1],
                token: None,
                client: "lerobot-adapter".into(),
            }),
            Message::HelloAck(HelloAck {
                version: PROTOCOL_VERSION,
                session_id: 42,
                execution_hash: Some([7u8; 32]),
            }),
            Message::HelloAck(HelloAck {
                version: PROTOCOL_VERSION,
                session_id: 43,
                execution_hash: None,
            }),
            Message::Subscribe {
                streams: vec![StreamId(0), StreamId(3)],
            },
            Message::Frame(Frame {
                tick: PhysTick(100),
                wall_ns: 123_456_789,
                stream: StreamId(1),
                payload: Payload::Scalar(0.5),
            }),
            Message::Frame(Frame {
                tick: PhysTick(101),
                wall_ns: 123_456_999,
                stream: StreamId(1),
                payload: Payload::Scalars(vec![0.1, 0.2, 0.3]),
            }),
            Message::Frame(Frame {
                tick: PhysTick(102),
                wall_ns: 124_000_000,
                stream: StreamId(2),
                payload: Payload::Tensor {
                    shape: vec![3, 224, 224],
                    dtype: "f32".into(),
                    bytes: vec![0, 1, 2, 3],
                },
            }),
            Message::Frame(Frame {
                tick: PhysTick(103),
                wall_ns: 124_100_000,
                stream: StreamId(4),
                payload: Payload::Image {
                    w: 224,
                    h: 224,
                    format: "rgb8".into(),
                    bytes: vec![255, 0, 0],
                },
            }),
            Message::Frame(Frame {
                tick: PhysTick(104),
                wall_ns: 124_200_000,
                stream: StreamId(5),
                payload: Payload::Event {
                    kind: "safety_violation".into(),
                    fields: BTreeMap::from([("limit".to_string(), "joint_1".to_string())]),
                },
            }),
            Message::Frame(Frame {
                tick: PhysTick(105),
                wall_ns: 124_300_000,
                stream: StreamId(6),
                payload: Payload::Metrics(PerfMetrics {
                    physics_steps_per_sec: Some(1.0),
                    ..Default::default()
                }),
            }),
            Message::Ping,
            Message::Pong,
            Message::Bye,
        ]
    }

    #[test]
    fn every_message_variant_round_trips() {
        for msg in sample_messages() {
            let encoded = encode(&msg);
            let (decoded, consumed) = decode(&encoded).unwrap();
            assert_eq!(consumed, encoded.len());
            assert_eq!(decoded, msg);
        }
    }

    #[test]
    fn decode_reports_leftover_bytes_consumed() {
        let a = encode(&Message::Ping);
        let b = encode(&Message::Pong);
        let mut buf = a.clone();
        buf.extend_from_slice(&b);
        let (first, n) = decode(&buf).unwrap();
        assert_eq!(first, Message::Ping);
        assert_eq!(n, a.len());
        let (second, n2) = decode(&buf[n..]).unwrap();
        assert_eq!(second, Message::Pong);
        assert_eq!(n2, b.len());
    }

    #[test]
    fn oversized_frame_is_rejected_before_parsing_body() {
        let encoded = encode(&Message::Ping);
        // The body is a few bytes; a max of 0 must reject it without touching the JSON.
        let err = decode_with_max(&encoded, 0).unwrap_err();
        assert!(
            matches!(err, ProtoError::TooLarge { max: 0, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn unknown_codec_byte_is_rejected() {
        let mut encoded = encode(&Message::Ping);
        encoded[0] = 0xFF;
        assert!(matches!(
            decode(&encoded),
            Err(ProtoError::UnknownCodec(0xFF))
        ));
    }

    #[test]
    fn truncated_buffer_is_incomplete_not_a_panic() {
        let encoded = encode(&Message::Ping);
        assert!(matches!(
            decode(&encoded[..2]),
            Err(ProtoError::Incomplete { .. })
        ));
        assert!(matches!(decode(&[]), Err(ProtoError::Incomplete { .. })));
    }

    #[test]
    fn perf_metrics_serializes_with_every_spec_12_4_metric_name() {
        let m = PerfMetrics {
            physics_steps_per_sec: Some(1.0),
            camera_frames_per_sec: Some(2.0),
            pixels_per_sec: Some(3.0),
            observation_gb_per_sec: Some(4.0),
            policy_inferences_per_sec: Some(5.0),
            actions_per_sec: Some(6.0),
            p50_end_to_end_latency: Some(7.0),
            p95_end_to_end_latency: Some(8.0),
            gpu_memory_peak: Some(9.0),
            chunk_underrun_rate: Some(10.0),
        };
        let json = serde_json::to_string(&m).unwrap();
        for field in [
            "physics_steps_per_sec",
            "camera_frames_per_sec",
            "pixels_per_sec",
            "observation_gb_per_sec",
            "policy_inferences_per_sec",
            "actions_per_sec",
            "p50_end_to_end_latency",
            "p95_end_to_end_latency",
            "gpu_memory_peak",
            "chunk_underrun_rate",
        ] {
            assert!(json.contains(field), "missing {field} in {json}");
        }
        assert_eq!(
            serde_json::from_str::<PerfMetrics>(&json).unwrap(),
            m,
            "must round-trip"
        );
        // Never a single step/s figure: no field is literally named that.
        assert!(!json.contains("\"step"));
    }

    #[test]
    fn unknown_metric_is_none_not_zero() {
        let json = serde_json::to_string(&PerfMetrics::default()).unwrap();
        let back: PerfMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(back.physics_steps_per_sec, None);
        assert_eq!(back.chunk_underrun_rate, None);
    }
}
