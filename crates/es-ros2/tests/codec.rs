//! Pure, PR-tier tests for the CDR codec, names/key-expressions, liveliness tokens and the
//! attachment (work packet `docs/packets/M3/W1a-ros2-cdr-keyexpr.md`). Every byte and hash
//! compared here against a fixed literal traces back to `docs/api-notes/ros2-cdr.md` /
//! `docs/api-notes/rmw-zenoh.md`'s reference tables, or to the checked-in goldens under
//! `tests/golden/ros2/` (produced by `python/gen_ros2_goldens.py`, never by this crate's own
//! encoder — see `tests/gen_goldens.rs` for the provenance check).

use es_core::alloc_count::assert_no_alloc;
use es_ros2::attachment::{Attachment, AttachmentError};
use es_ros2::cdr::{CdrError, CdrReader};
use es_ros2::msg::{
    self, CameraInfo, Float64MultiArray, JointState, Msg, MsgType, RegionOfInterest, StringMsg,
    Time,
};
use es_ros2::names::{
    classify, topic_key_expr, validate_name, EntityKind, LivelinessToken, TokenScheme,
};
use proptest::prelude::*;
use std::path::{Path, PathBuf};

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/ros2")
}

fn golden_bin(name: &str) -> Vec<u8> {
    let path = golden_dir().join("cdr").join(format!("{name}.bin"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn golden_json(path: &str) -> serde_json::Value {
    let p = golden_dir().join(path);
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

// --- CDR: goldens ---------------------------------------------------------------------------

// These floats are literal transcriptions of docs/api-notes/ros2-cdr.md's golden table: exact
// bit-for-bit equality is the point (spec 1.4 bitwise reproducibility), not a fuzzy comparison.
#[allow(clippy::float_cmp)]
#[test]
fn cdr_goldens_match_rosbags() {
    let hello = golden_bin("string_hello");
    let msg = StringMsg::from_cdr(&hello).unwrap();
    assert_eq!(msg.data, "hello");
    assert_eq!(msg.to_cdr(), hello);

    let empty = golden_bin("string_empty");
    let msg = StringMsg::from_cdr(&empty).unwrap();
    assert_eq!(msg.data, "");
    assert_eq!(msg.to_cdr(), empty);

    // Big-endian: decode only (this crate's writer always emits little-endian).
    let hello_be = golden_bin("string_hello_be");
    let msg = StringMsg::from_cdr(&hello_be).unwrap();
    assert_eq!(msg.data, "hello");

    let joint_state = golden_bin("joint_state");
    let msg = JointState::from_cdr(&joint_state).unwrap();
    assert_eq!(msg.header.stamp, Time { sec: 1, nanosec: 2 });
    assert_eq!(msg.header.frame_id, "base");
    assert_eq!(msg.name, vec!["j1".to_string(), "j2".to_string()]);
    assert_eq!(msg.position, vec![0.5, -1.0]);
    assert!(msg.velocity.is_empty());
    assert!(msg.effort.is_empty());
    assert_eq!(msg.to_cdr(), joint_state);

    let image = golden_bin("image_rgb8_2x1");
    let msg = msg::Image::from_cdr(&image).unwrap();
    assert_eq!(msg.header.frame_id, "cam");
    assert_eq!((msg.height, msg.width), (1, 2));
    assert_eq!(msg.encoding, "rgb8");
    assert_eq!(msg.is_bigendian, 0);
    assert_eq!(msg.step, 6);
    assert_eq!(msg.data, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(msg.to_cdr(), image);

    let camera_info = golden_bin("camera_info_plumb_bob");
    let msg = CameraInfo::from_cdr(&camera_info).unwrap();
    assert_eq!(msg.header.frame_id, "cam");
    assert_eq!((msg.height, msg.width), (480, 640));
    assert_eq!(msg.distortion_model, "plumb_bob");
    assert_eq!(msg.d, vec![0.1, 0.01, 0.0, 0.0, 0.0]);
    assert_eq!(msg.k, [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]);
    assert_eq!(msg.r, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(
        msg.p,
        [500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );
    assert_eq!((msg.binning_x, msg.binning_y), (0, 0));
    assert_eq!(
        msg.roi,
        RegionOfInterest {
            x_offset: 0,
            y_offset: 0,
            height: 0,
            width: 0,
            do_rectify: false,
        }
    );
    assert_eq!(msg.to_cdr(), camera_info);

    let fma = golden_bin("float64_multi_array");
    let msg = Float64MultiArray::from_cdr(&fma).unwrap();
    assert!(msg.layout.dim.is_empty());
    assert_eq!(msg.layout.data_offset, 0);
    assert_eq!(msg.data, vec![0.25, -0.5, 1.0]);
    assert_eq!(msg.to_cdr(), fma);
}

#[test]
fn cdr_empty_sequences_get_no_alignment_padding() {
    let joint_state = golden_bin("joint_state");
    assert_eq!(joint_state.len(), 76);
    let msg = JointState::from_cdr(&joint_state).unwrap();
    assert_eq!(msg.to_cdr().len(), 76);
}

#[test]
fn cdr_camera_info_is_357_bytes_without_trailing_padding() {
    let camera_info = golden_bin("camera_info_plumb_bob");
    assert_eq!(camera_info.len(), 357);
    let msg = CameraInfo::from_cdr(&camera_info).unwrap();
    assert_eq!(msg.to_cdr().len(), 357);
}

#[test]
fn cdr_decoder_accepts_options_bits_and_up_to_three_trailing_bytes() {
    for options in [[0x00, 0x00], [0x12, 0x34]] {
        for extra in 0..=3 {
            let mut bytes = vec![0x00, 0x01, options[0], options[1]];
            let s = StringMsg { data: "hi".into() }.to_cdr();
            bytes.extend_from_slice(&s[4..]);
            bytes.extend(std::iter::repeat_n(0xAB, extra));
            let msg = StringMsg::from_cdr(&bytes).unwrap();
            assert_eq!(msg.data, "hi");
        }
    }
}

#[test]
fn cdr_decoder_rejects_a_fourth_trailing_byte_and_unknown_representation_ids() {
    let mut bytes = StringMsg { data: "hi".into() }.to_cdr();
    bytes.extend_from_slice(&[0xAB; 4]);
    assert_eq!(
        StringMsg::from_cdr(&bytes).unwrap_err(),
        CdrError::TrailingBytes(4)
    );

    for id in [[0x00, 0x07], [0x00, 0x09], [0x00, 0x0b], [0x01, 0x00]] {
        let bytes = [id[0], id[1], 0x00, 0x00];
        assert_eq!(CdrReader::new(&bytes).unwrap_err(), CdrError::BadHeader);
    }
}

proptest! {
    #[test]
    fn cdr_decoding_is_total(bytes in proptest::collection::vec(any::<u8>(), 0..96)) {
        for ty in msg::testing::ALL {
            let _ = Msg::from_cdr(ty, &bytes);
        }
    }

    #[test]
    fn liveliness_parse_is_total_and_keeps_unknown_trailing_chunks(key in "[-a-zA-Z0-9_@%/:,.]{0,200}") {
        let _ = LivelinessToken::parse(&key);
    }
}

#[test]
fn cdr_decoding_of_a_huge_sequence_count_in_a_small_buffer_is_truncated_without_allocating() {
    // A `float64[]` count of `u32::MAX` inside a 16-byte JointState-shaped buffer: reading
    // `position` must fail before allocating anything for it.
    let mut bytes = vec![0x00, 0x01, 0x00, 0x00]; // header
    bytes.extend_from_slice(&1i32.to_le_bytes()); // sec
    bytes.extend_from_slice(&2u32.to_le_bytes()); // nanosec
    bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // frame_id "length": bogus, huge
    let result = assert_no_alloc(|| JointState::from_cdr(&bytes));
    assert_eq!(result.unwrap_err(), CdrError::Truncated);
}

// --- RIHS01 -----------------------------------------------------------------------------

#[test]
fn msg_type_table_matches_the_rihs01_golden() {
    let golden = golden_json("rihs01.json");
    for ty in msg::testing::ALL {
        let want = golden[ty.ros_name()].as_str().unwrap();
        assert_eq!(ty.rihs01(), want, "{}", ty.ros_name());
    }
}

// --- Names / key expressions -------------------------------------------------------------

#[test]
fn ros_names_are_validated() {
    for good in ["/chatter", "/a/b_c"] {
        assert!(validate_name(good).is_ok(), "{good}");
    }
    for bad in ["chatter", "//x", "/x/", "~/x", "/1x", "/x y"] {
        assert!(validate_name(bad).is_err(), "{bad}");
    }
}

#[test]
fn topic_key_exprs_match_rmw_zenoh_design_md() {
    let a = topic_key_expr(0, "/chatter", MsgType::String).unwrap();
    assert_eq!(
        a,
        "0/chatter/std_msgs::msg::dds_::String_/RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18"
    );
    let b = topic_key_expr(0, "/robot1/chatter", MsgType::String).unwrap();
    assert_eq!(
        b,
        "0/robot1/chatter/std_msgs::msg::dds_::String_/RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18"
    );
}

const NN_EXAMPLE: &str = "@ros2_lv/0/aac3178e146ba6f1fc6e6a4085e77f21/0/0/NN/%/%/listener";
const MP_EXAMPLE: &str = "@ros2_lv/0/8b20917502ee955ac4476e0266340d5c/0/10/MP/%/%/talker/%chatter/std_msgs::msg::dds_::String_/RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18/::,7:,:,:,,";

#[test]
fn design_md_liveliness_tokens_round_trip_byte_identical() {
    let nn = LivelinessToken::parse(NN_EXAMPLE).unwrap();
    assert_eq!(nn.domain_id, 0);
    assert_eq!(nn.zid, "aac3178e146ba6f1fc6e6a4085e77f21");
    assert_eq!(nn.node_id, 0);
    assert_eq!(nn.entity_id, 0);
    assert_eq!(nn.kind, EntityKind::Node);
    assert_eq!(nn.enclave, "/");
    assert_eq!(nn.namespace, "/");
    assert_eq!(nn.node_name, "listener");
    assert!(nn.topic.is_none());
    assert_eq!(nn.to_key_expr(), NN_EXAMPLE);

    let mp = LivelinessToken::parse(MP_EXAMPLE).unwrap();
    assert_eq!(mp.zid, "8b20917502ee955ac4476e0266340d5c");
    assert_eq!(mp.node_id, 0);
    assert_eq!(mp.entity_id, 10);
    assert_eq!(mp.kind, EntityKind::Publisher);
    assert_eq!(mp.namespace, "/");
    assert_eq!(mp.node_name, "talker");
    let topic = mp.topic.as_ref().unwrap();
    assert_eq!(topic.name, "/chatter");
    assert_eq!(topic.type_name, "std_msgs::msg::dds_::String_");
    assert_eq!(
        topic.type_hash,
        "RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18"
    );
    assert_eq!(topic.qos.depth, Some(7));
    assert_eq!(mp.to_key_expr(), MP_EXAMPLE);
}

#[test]
fn liveliness_parse_keeps_the_rolling_backends_suffix() {
    let with_backends = format!("{MP_EXAMPLE}/some_backend");
    let token = LivelinessToken::parse(&with_backends).unwrap();
    assert_eq!(token.extra, vec!["some_backend".to_string()]);
    assert_eq!(token.to_key_expr(), with_backends);
}

#[test]
fn token_scheme_separates_rmw_zenoh_from_the_dds_bridge() {
    assert_eq!(classify(NN_EXAMPLE), TokenScheme::RmwZenoh);
    assert_eq!(
        classify("@/1234abcd/@ros2_lv/MP/0/chatter/std_msgs::msg::dds_::String_/::,,:,:,:,,"),
        TokenScheme::Ros2DdsBridge
    );
    assert_eq!(
        classify("0/chatter/std_msgs::msg::dds_::String_/RIHS01_x"),
        TokenScheme::Other
    );
}

// --- Attachment ---------------------------------------------------------------------------

#[test]
fn attachment_is_seq_ts_leb128_len_gid() {
    let gid: [u8; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f,
    ];
    let mut expected = [0u8; 33];
    expected[0..8].copy_from_slice(&1i64.to_le_bytes());
    expected[8..16].copy_from_slice(&1_700_000_000_123_456_789i64.to_le_bytes());
    expected[16] = 0x10;
    expected[17..33].copy_from_slice(&gid);

    let a = Attachment {
        seq: 1,
        source_timestamp_ns: 1_700_000_000_123_456_789,
        gid,
    };
    assert_eq!(a.encode(), expected);
    assert_eq!(Attachment::decode(&expected).unwrap(), a);

    assert_eq!(
        Attachment::decode(&expected[..32]).unwrap_err(),
        AttachmentError::BadLength(32)
    );
    let mut too_long = expected.to_vec();
    too_long.push(0);
    assert_eq!(
        Attachment::decode(&too_long).unwrap_err(),
        AttachmentError::BadLength(34)
    );

    let mut bad_len = expected;
    bad_len[16] = 0x11;
    assert_eq!(
        Attachment::decode(&bad_len).unwrap_err(),
        AttachmentError::BadGidLength(0x11)
    );
}

#[test]
fn gid_matches_python_xxhash_for_the_design_md_tokens() {
    let golden = golden_json("gid.json");
    for entry in golden["tokens"].as_array().unwrap() {
        let key = entry["key_expr"].as_str().unwrap();
        let want = entry["gid_hex"].as_str().unwrap();
        let got = hex_encode(&es_ros2::attachment::gid_of(key));
        assert_eq!(got, want, "{key}");
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}
