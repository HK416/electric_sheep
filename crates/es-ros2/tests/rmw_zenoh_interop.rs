//! The live `rmw_zenoh` / `ros2` CLI / `demo_nodes_cpp` oracle (`docs/design/ros2-boundary.md`
//! section 8, work packet `docs/packets/M3/W1b-ros2-zenoh-session.md`). SKIPs, per test, when
//! `ES_ROS2_ENV` is unset; with it, a failure is a real failure, never a SKIP (spec 1.4).
//!
//! Run: `export ES_ROS2_ENV=$HOME/envs/ros2-kilted; cargo test -p es-ros2 --features zenoh
//! --test rmw_zenoh_interop -- --nocapture --test-threads=1`. `--test-threads=1` matters: every
//! non-ignored test shares one `rmw_zenohd` router and one `Ros2Node` (domain 73), started
//! lazily on first use and torn down when the process exits.
//!
//! `z_ran_rmw_zenoh_interop` sorts alphabetically after the five scenario tests (rustc's test
//! harness runs `--test-threads=1` tests in name order); it prints `RAN rmw_zenoh_interop` once
//! a shared counter confirms all five actually ran (not skipped), which is what
//! `docs/packets/M3/W1b-ros2-zenoh-session.md`'s oracle greps for.

#![cfg(feature = "zenoh")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use es_ros2::config::{Ros2Config, Ros2Mode, ZenohEndpoints, ZenohMode};
use es_ros2::msg::{
    CameraInfo, Float64MultiArray, Header, Image, Msg, MsgType, MultiArrayLayout, RegionOfInterest,
    StringMsg, Time,
};
use es_ros2::session::Ros2Node;
use zenoh::Wait;

const N_SCENARIOS: usize = 5;
static COMPLETED: AtomicUsize = AtomicUsize::new(0);

fn skip(test: &str) {
    println!("SKIP {test}: ES_ROS2_ENV is unset");
}

fn env_prefix() -> Option<String> {
    std::env::var("ES_ROS2_ENV").ok()
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Kills and reaps the child on drop, so a failing assertion never leaves an orphaned
/// `rmw_zenohd`/`ros2` process behind (every child in this harness is guarded this way).
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

fn reserve_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
    let port = l.local_addr().expect("local addr").port();
    drop(l);
    port
}

/// `crates/es-ros2/scripts/ros2-env.sh <prefix> <program> <args...>`, run under `bash`
/// explicitly (not relying on the shebang / exec bit surviving a checkout) so the conda
/// activation scripts' own `source` calls resolve under a real bash, not `/bin/sh`
/// (`docs/api-notes/ros2-cdr.md` "Live ROS 2 byte capture").
fn ros2_env_cmd(prefix: &str, program: &str, args: &[&str]) -> Command {
    let script = repo_root().join("crates/es-ros2/scripts/ros2-env.sh");
    let mut cmd = Command::new("bash");
    cmd.arg(script).arg(prefix).arg(program).args(args);
    cmd
}

/// Runs `cmd` to completion, killing it if it outlives `timeout` (every wait in this harness is
/// bounded). Small stdout/stderr only (CLI text, never a media stream), so polling `try_wait`
/// without concurrently draining the pipes cannot deadlock in practice here.
fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<std::process::Output, String> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(Some(_status)) = child.try_wait() {
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("timed out after {timeout:?}"));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    child.wait_with_output().map_err(|e| e.to_string())
}

struct Harness {
    _router: ChildGuard,
    node: Ros2Node,
    port: u16,
    prefix: String,
}

impl Harness {
    /// `None` (never touching the shared `OnceLock`) when `ES_ROS2_ENV` is unset; otherwise the
    /// one shared router + node, started lazily on first use.
    fn get() -> Option<&'static Harness> {
        static HARNESS: OnceLock<Harness> = OnceLock::new();
        env_prefix()?;
        Some(HARNESS.get_or_init(|| Self::start().expect("start rmw_zenohd and our node")))
    }

    fn start() -> Result<Self, String> {
        let prefix = env_prefix().ok_or("ES_ROS2_ENV is unset")?;
        let port = reserve_port();
        let mut cmd = ros2_env_cmd(&prefix, "ros2", &["run", "rmw_zenoh_cpp", "rmw_zenohd"]);
        cmd.env(
            "ZENOH_CONFIG_OVERRIDE",
            format!("listen/endpoints=[\"tcp/127.0.0.1:{port}\"]"),
        );
        cmd.env("RMW_IMPLEMENTATION", "rmw_zenoh_cpp");
        cmd.env("ROS_DOMAIN_ID", "73");
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        let router = ChildGuard(cmd.spawn().map_err(|e| format!("spawn rmw_zenohd: {e}"))?);
        if !wait_for_port(port, Duration::from_secs(20)) {
            return Err("rmw_zenohd did not open its listen port within 20s".into());
        }

        let cfg = Ros2Config {
            domain_id: 73,
            node: "es_interop".to_owned(),
            namespace: "/".to_owned(),
            enclave: "/".to_owned(),
            liveliness_timeout: Duration::from_secs(2),
            mode: Ros2Mode::RmwZenoh(ZenohEndpoints {
                mode: ZenohMode::Peer,
                connect: vec![format!("tcp/127.0.0.1:{port}")],
                listen: Vec::new(),
            }),
            actuators: Vec::new(),
        };
        let node = Ros2Node::open(&cfg).map_err(|e| format!("open our node: {e}"))?;
        Ok(Self {
            _router: router,
            node,
            port,
            prefix,
        })
    }

    fn ros2_cmd(&self, args: &[&str]) -> Command {
        let mut cmd = ros2_env_cmd(&self.prefix, "ros2", args);
        cmd.env("RMW_IMPLEMENTATION", "rmw_zenoh_cpp");
        cmd.env("ROS_DOMAIN_ID", "73");
        cmd.env(
            "ZENOH_CONFIG_OVERRIDE",
            format!("connect/endpoints=[\"tcp/127.0.0.1:{}\"]", self.port),
        );
        cmd
    }

    fn spawn_ros2(&self, args: &[&str]) -> ChildGuard {
        let mut cmd = self.ros2_cmd(args);
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        ChildGuard(
            cmd.spawn()
                .unwrap_or_else(|e| panic!("spawn ros2 {args:?}: {e}")),
        )
    }
}

/// Publishes `msg` at 10 Hz on `ros_topic` while `ros2 topic echo --once` (or any other
/// short-lived `ros2` command) runs, so the echo has something to receive; asserts the command
/// exits 0 and its stdout contains every one of `contains`.
fn publish_while(
    h: &Harness,
    ros_topic: &str,
    ty: MsgType,
    msg: Msg,
    ros2_args: &[&str],
    contains: &[&str],
) -> String {
    let publisher = h
        .node
        .publisher(ros_topic, ty, 1)
        .expect("declare publisher");
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let handle = std::thread::spawn(move || {
        while !stop2.load(Ordering::SeqCst) {
            let _ = publisher.put(&msg);
            std::thread::sleep(Duration::from_millis(100));
        }
    });
    let result = run_with_timeout(h.ros2_cmd(ros2_args), Duration::from_secs(30));
    stop.store(true, Ordering::SeqCst);
    handle.join().expect("publisher thread");
    let out = result.unwrap_or_else(|e| panic!("{ros2_args:?}: {e}"));
    assert!(
        out.status.success(),
        "{ros2_args:?} exited {:?}",
        out.status
    );
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    for c in contains {
        assert!(stdout.contains(c), "missing {c:?} in:\n{stdout}");
    }
    stdout
}

// --- scenarios (run under Harness::get(), name order matters: see the module doc) -------------

#[test]
fn ros2_topic_echo_receives_our_string() {
    const NAME: &str = "ros2_topic_echo_receives_our_string";
    let Some(h) = Harness::get() else {
        skip(NAME);
        return;
    };
    publish_while(
        h,
        "/es_chatter",
        MsgType::String,
        Msg::String(StringMsg {
            data: "hello from es".to_owned(),
        }),
        &[
            "topic",
            "echo",
            "--once",
            "/es_chatter",
            "std_msgs/msg/String",
        ],
        &["data: hello from es"],
    );
    COMPLETED.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn ros2_cli_lists_our_node_and_topic() {
    const NAME: &str = "ros2_cli_lists_our_node_and_topic";
    let Some(h) = Harness::get() else {
        skip(NAME);
        return;
    };
    // Keep a publisher alive so `/es_chatter` is present in the topic list. `--no-daemon`
    // bypasses the `ros2` daemon's own cache, which otherwise can miss a topic declared after
    // the daemon started (recorded here per the work packet: needed, not just defensive).
    let _publisher = h
        .node
        .publisher("/es_chatter2", MsgType::String, 1)
        .expect("declare publisher");

    let nodes = run_with_timeout(
        h.ros2_cmd(&["node", "list", "--no-daemon"]),
        Duration::from_secs(20),
    )
    .expect("ros2 node list");
    assert!(nodes.status.success());
    let nodes_out = String::from_utf8_lossy(&nodes.stdout);
    assert!(nodes_out.contains("/es_interop"), "node list:\n{nodes_out}");

    let topics = run_with_timeout(
        h.ros2_cmd(&["topic", "list", "-t", "--no-daemon"]),
        Duration::from_secs(20),
    )
    .expect("ros2 topic list");
    assert!(topics.status.success());
    let topics_out = String::from_utf8_lossy(&topics.stdout);
    assert!(
        topics_out.contains("/es_chatter2 [std_msgs/msg/String]"),
        "topic list:\n{topics_out}"
    );
    COMPLETED.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn demo_talker_reaches_our_subscriber() {
    const NAME: &str = "demo_talker_reaches_our_subscriber";
    let Some(h) = Harness::get() else {
        skip(NAME);
        return;
    };
    let subscriber = h
        .node
        .subscriber("/chatter", MsgType::String, 10)
        .expect("declare subscriber");
    let _talker = h.spawn_ros2(&["run", "demo_nodes_cpp", "talker"]);

    let mut last_seq: Option<i64> = None;
    let mut got_hello = false;
    let deadline = Instant::now() + Duration::from_secs(25);
    while Instant::now() < deadline && !got_hello {
        let Some(received) = subscriber
            .recv_timeout(Duration::from_secs(3))
            .expect("recv")
        else {
            continue;
        };
        let att = received.attachment.expect("mode A always attaches");
        assert_eq!(att.gid.len(), 16);
        if let Some(last) = last_seq {
            assert!(
                att.seq > last,
                "seq must strictly increase: {} <= {last}",
                att.seq
            );
        }
        last_seq = Some(att.seq);
        if let Msg::String(s) = &received.msg {
            if s.data.starts_with("Hello World: ") {
                got_hello = true;
            }
        }
    }
    assert!(
        got_hello,
        "never received a `Hello World: <n>` string from the talker"
    );

    // The talker's own `MP` token: round trip it and check the attachment's GID against it.
    let tokens = h
        .node
        .liveliness_tokens("@ros2_lv/73/**", Duration::from_secs(3))
        .expect("liveliness get");
    let mp_key = tokens
        .into_iter()
        .find(|k| k.contains("/MP/") && k.contains("chatter"))
        .expect("the talker's MP token is visible");
    let parsed = es_ros2::names::LivelinessToken::parse(&mp_key).expect("MP token parses");
    assert_eq!(parsed.to_key_expr(), mp_key, "round trip byte-identical");
    let expected_gid = es_ros2::attachment::gid_of(&mp_key);
    // The samples already consumed above are gone; a fresh one carries the same publisher GID.
    let received = subscriber
        .recv_timeout(Duration::from_secs(10))
        .expect("recv")
        .expect("one more sample to check the GID against the token");
    let att = received.attachment.expect("attachment");
    assert_eq!(
        att.gid, expected_gid,
        "attachment gid must equal gid_of(talker's MP token)"
    );
    COMPLETED.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn ros2_topic_pub_joint_state_reaches_our_subscriber() {
    const NAME: &str = "ros2_topic_pub_joint_state_reaches_our_subscriber";
    let Some(h) = Harness::get() else {
        skip(NAME);
        return;
    };
    let subscriber = h
        .node
        .subscriber("/es_js", MsgType::JointState, 4)
        .expect("declare subscriber");
    let out = run_with_timeout(
        h.ros2_cmd(&[
            "topic",
            "pub",
            "--once",
            "-w",
            "1",
            "/es_js",
            "sensor_msgs/msg/JointState",
            "{name: [j1, j2], position: [0.5, -1.0]}",
        ]),
        Duration::from_secs(20),
    )
    .expect("ros2 topic pub");
    assert!(
        out.status.success(),
        "ros2 topic pub exited {:?}",
        out.status
    );

    let received = subscriber
        .recv_timeout(Duration::from_secs(15))
        .expect("recv")
        .unwrap_or_else(|| panic!("no JointState arrived; dropped={}", subscriber.dropped()));
    let Msg::JointState(js) = received.msg else {
        panic!("wrong message type");
    };
    assert_eq!(js.name, vec!["j1".to_owned(), "j2".to_owned()]);
    assert_eq!(js.position, vec![0.5, -1.0]);
    COMPLETED.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn our_image_camera_info_and_float64_multi_array_echo_in_ros2() {
    const NAME: &str = "our_image_camera_info_and_float64_multi_array_echo_in_ros2";
    let Some(h) = Harness::get() else {
        skip(NAME);
        return;
    };

    let image = Image {
        header: Header {
            stamp: Time { sec: 1, nanosec: 2 },
            frame_id: "cam".to_owned(),
        },
        height: 1,
        width: 2,
        encoding: "rgb8".to_owned(),
        is_bigendian: 0,
        step: 6,
        data: vec![1, 2, 3, 4, 5, 6],
    };
    publish_while(
        h,
        "/es_image",
        MsgType::Image,
        Msg::Image(image),
        &[
            "topic",
            "echo",
            "--once",
            "/es_image",
            "sensor_msgs/msg/Image",
        ],
        &["encoding: rgb8"],
    );

    let info = CameraInfo {
        header: Header {
            stamp: Time { sec: 1, nanosec: 2 },
            frame_id: "cam".to_owned(),
        },
        height: 480,
        width: 640,
        distortion_model: "plumb_bob".to_owned(),
        d: vec![0.1, 0.01, 0.0, 0.0, 0.0],
        k: [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
        r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p: [
            500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
        binning_x: 0,
        binning_y: 0,
        roi: RegionOfInterest {
            x_offset: 0,
            y_offset: 0,
            height: 0,
            width: 0,
            do_rectify: false,
        },
    };
    publish_while(
        h,
        "/es_camera_info",
        MsgType::CameraInfo,
        Msg::CameraInfo(info),
        &[
            "topic",
            "echo",
            "--once",
            "/es_camera_info",
            "sensor_msgs/msg/CameraInfo",
        ],
        &["distortion_model: plumb_bob"],
    );

    let arr = Float64MultiArray {
        layout: MultiArrayLayout {
            dim: Vec::new(),
            data_offset: 0,
        },
        data: vec![0.25, -0.5, 1.0],
    };
    publish_while(
        h,
        "/es_array",
        MsgType::Float64MultiArray,
        Msg::Float64MultiArray(arr),
        &[
            "topic",
            "echo",
            "--once",
            "/es_array",
            "std_msgs/msg/Float64MultiArray",
        ],
        &["0.25", "-0.5", "1.0"],
    );
    COMPLETED.fetch_add(1, Ordering::SeqCst);
}

/// Sorts alphabetically after the five scenarios above (`z_` prefix; rustc's harness runs
/// `--test-threads=1` tests in name order — see the module doc). Prints the marker the
/// packet's oracle greps for, but only once every scenario actually ran.
#[test]
fn z_ran_rmw_zenoh_interop() {
    if env_prefix().is_none() {
        skip("z_ran_rmw_zenoh_interop");
        return;
    }
    let n = COMPLETED.load(Ordering::SeqCst);
    assert_eq!(
        n, N_SCENARIOS,
        "only {n}/{N_SCENARIOS} interop scenarios completed"
    );
    println!("RAN rmw_zenoh_interop");
}

// --- one-time capture, committed by this packet -------------------------------------------

fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// Version strings for the `RoboStack` packages this oracle depends on, read from `conda-meta`
/// (design note section 8's pinned versions), without invoking Python — raw filesystem only,
/// consistent with this test's "raw zenoh-rs only" rule for the rest of the capture.
fn conda_meta_versions(prefix: &str) -> serde_json::Value {
    let patterns = [
        "ros-kilted-ros-base-",
        "ros-kilted-rmw-zenoh-cpp-",
        "ros-kilted-demo-nodes-cpp-",
        "ros-kilted-cv-bridge-",
        "ros-kilted-image-geometry-",
    ];
    let mut map = serde_json::Map::new();
    if let Ok(entries) = std::fs::read_dir(Path::new(prefix).join("conda-meta")) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            for pat in patterns {
                if let Some(stripped) = name.strip_prefix(pat) {
                    if let Some(json_free) = stripped.strip_suffix(".json") {
                        map.insert(pat.trim_end_matches('-').to_owned(), json_free.into());
                    }
                }
            }
        }
    }
    serde_json::Value::Object(map)
}

/// **Raw zenoh-rs only** (design note section 8, this packet's oracle): no `es_ros2` encoder or
/// parser touches the captured bytes/strings, so `committed_reference_capture_round_trips`
/// (`session_loopback.rs`) is an independent check, not a circular one. `#[ignore]`: a one-time
/// capture run by hand (`docs/packets/M3/W1b-ros2-zenoh-session.md`'s oracle), not part of the
/// regular suite.
#[test]
#[ignore = "one-time capture; run by hand with ES_ROS2_ENV and ES_ROS2_CAPTURE_DIR set"]
fn capture_reference_goldens() {
    let prefix = env_prefix().expect("ES_ROS2_ENV must be set to capture goldens");
    // Cargo runs a test binary with its CWD set to the *package's* manifest directory, not the
    // workspace root the packet's oracle command is written from — resolve the (relative) path
    // the same way `session_loopback.rs`'s `golden_dir()` reads it back.
    let capture_dir_env =
        std::env::var("ES_ROS2_CAPTURE_DIR").expect("ES_ROS2_CAPTURE_DIR must be set");
    let capture_dir = repo_root().join(&capture_dir_env);

    let port = reserve_port();
    let mut router_cmd = ros2_env_cmd(&prefix, "ros2", &["run", "rmw_zenoh_cpp", "rmw_zenohd"]);
    router_cmd.env(
        "ZENOH_CONFIG_OVERRIDE",
        format!("listen/endpoints=[\"tcp/127.0.0.1:{port}\"]"),
    );
    router_cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let _router = ChildGuard(router_cmd.spawn().expect("spawn rmw_zenohd"));
    assert!(
        wait_for_port(port, Duration::from_secs(20)),
        "router did not open its port"
    );

    let mut zcfg = zenoh::Config::default();
    zcfg.insert_json5("mode", "\"peer\"").unwrap();
    zcfg.insert_json5("connect/endpoints", &format!("[\"tcp/127.0.0.1:{port}\"]"))
        .unwrap();
    zcfg.insert_json5("scouting/multicast/enabled", "false")
        .unwrap();
    let session: zenoh::Session = zenoh::open(zcfg).wait().expect("open raw capture session");

    let ros_env = |cmd: &mut Command| {
        cmd.env("RMW_IMPLEMENTATION", "rmw_zenoh_cpp");
        cmd.env("ROS_DOMAIN_ID", "73");
        cmd.env(
            "ZENOH_CONFIG_OVERRIDE",
            format!("connect/endpoints=[\"tcp/127.0.0.1:{port}\"]"),
        );
    };

    let mut talker_cmd = ros2_env_cmd(&prefix, "ros2", &["run", "demo_nodes_cpp", "talker"]);
    ros_env(&mut talker_cmd);
    talker_cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let _talker = ChildGuard(talker_cmd.spawn().expect("spawn talker"));

    let mut listener_cmd = ros2_env_cmd(&prefix, "ros2", &["run", "demo_nodes_cpp", "listener"]);
    ros_env(&mut listener_cmd);
    listener_cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let _listener = ChildGuard(listener_cmd.spawn().expect("spawn listener"));

    let chatter_key = "73/chatter/std_msgs::msg::dds_::String_/\
        RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18";
    let sub = session
        .declare_subscriber(chatter_key)
        .wait()
        .expect("raw subscriber");

    let mut attachments_hex = Vec::new();
    let mut payloads_hex = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(25);
    while attachments_hex.len() < 3 && Instant::now() < deadline {
        if let Ok(Some(sample)) = sub.recv_timeout(Duration::from_secs(5)) {
            if let Some(att) = sample.attachment() {
                attachments_hex.push(encode_hex(&att.to_bytes()));
                payloads_hex.push(encode_hex(&sample.payload().to_bytes()));
            }
        }
    }
    assert!(
        attachments_hex.len() >= 3,
        "only captured {} talker samples",
        attachments_hex.len()
    );
    attachments_hex.truncate(3);
    payloads_hex.truncate(3);

    let tokens = {
        let replies = session
            .liveliness()
            .get("@ros2_lv/73/**")
            .timeout(Duration::from_secs(5))
            .wait()
            .expect("liveliness get");
        let mut out = Vec::new();
        while let Ok(Some(reply)) = replies.recv_timeout(Duration::from_secs(5)) {
            if let Ok(sample) = reply.result() {
                out.push(sample.key_expr().as_str().to_owned());
            }
        }
        out
    };
    let find = |pred: &dyn Fn(&str) -> bool| tokens.iter().find(|k| pred(k)).cloned();
    let talker_node_token = find(&|k| k.contains("/NN/") && k.ends_with("/talker"));
    let talker_pub_token = find(&|k| k.contains("/MP/") && k.contains("chatter"));
    let listener_sub_token = find(&|k| k.contains("/MS/") && k.contains("/listener/"));

    let js_key = "73/es_js_capture/sensor_msgs::msg::dds_::JointState_/\
        RIHS01_a13ee3a330e346c9d87b5aa18d24e11690752bd33a0350f11c5882bc9179260e";
    let js_sub = session
        .declare_subscriber(js_key)
        .wait()
        .expect("raw JointState subscriber");
    // `-w 0`: `ros2 topic pub --once` defaults to waiting for at least one matched RMW
    // subscription (confirmed empirically: it timed out with neither `-w` given nor `-w 1`),
    // which needs a real `MS` liveliness token from an `rclcpp`/`es_ros2` subscriber. The raw
    // zenoh subscriber above has none (deliberately: "raw zenoh-rs only"); zenoh still delivers
    // the `put` to it regardless once the command actually publishes, so forcing the wait to 0
    // is enough.
    let mut pub_cmd = ros2_env_cmd(
        &prefix,
        "ros2",
        &[
            "topic",
            "pub",
            "--once",
            "-w",
            "0",
            "/es_js_capture",
            "sensor_msgs/msg/JointState",
            "{name: [j1, j2], position: [0.5, -1.0]}",
        ],
    );
    ros_env(&mut pub_cmd);
    let _ = run_with_timeout(pub_cmd, Duration::from_secs(20));
    let joint_state_pub_hex = js_sub
        .recv_timeout(Duration::from_secs(15))
        .ok()
        .flatten()
        .map(|s| encode_hex(&s.payload().to_bytes()));

    let json = serde_json::json!({
        "talker_node_token": talker_node_token,
        "talker_pub_token": talker_pub_token,
        "listener_sub_token": listener_sub_token,
        "attachments_hex": attachments_hex,
        "payloads_hex": payloads_hex,
        "joint_state_pub_hex": joint_state_pub_hex,
        "robostack_versions": conda_meta_versions(&prefix),
    });
    std::fs::create_dir_all(&capture_dir).expect("create capture dir");
    let out_path = capture_dir.join("talker_capture.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&json).unwrap()).expect("write capture");
    println!("wrote {}", out_path.display());
}
