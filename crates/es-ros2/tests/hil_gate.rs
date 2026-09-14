//! The M3 gate (spec 24.2, spec 28.5, `docs/design/ros2-boundary.md` section 7.5): a live HIL
//! run over real loopback UDP — with injected delay, 5 % loss, a swapped pair, a 40-tick-late
//! command, a `NaN` row and a heartbeat gap — replays to **byte-identical** Safety Plane
//! decisions.
//!
//! There is no external reference oracle here. The oracle is spec 24.2's own property (live ==
//! replay) plus hand-assembled wire and log layouts, so this file judges itself on any machine.
//!
//! What it does **not** claim: anything about a physical controller, a real network, RT
//! scheduling, or spec 28.7's gate 14, which stays open. Every duration printed below is an
//! *observation* of one CI loopback run, never a performance claim (spec 12.4).

use std::collections::VecDeque;
use std::fs;
use std::io::{self, Write};
use std::net::{SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use es_core::{PhysTick, TickRate};
use es_ir::deployment::{
    ActionContract, ActionSpace, Deadlines, DeploymentIr, ExecutionMode, FallbackPolicy, Limit,
    Micros, RateLimit, RateSpec, RobotRef, RobotTarget, SafetyEnvelope, Watchdog, WatchdogSet,
    Workspace, SCHEMA_VERSION,
};
use es_ros2::hil::wire::{self, Command, HilMsg};
use es_ros2::hil::{replay, HilConfig, HilCore, HilLink, HilLogError, HIL_STATS_STREAM};
use es_safety::{ActionSource, ViolationKind};
use es_telemetry::{Client, Message, Payload, Server, TransportError};

const NJ: usize = 3;
const H: usize = 8;
/// Rows the controller fills per command; also the Deployment IR's `execute_chunk`.
const ROWS: u16 = 4;

/// Fixed test key (design note section 7.7: real rigs read a `*.eshilkey` file outside the
/// repo; a test key is a constant so the run is reproducible).
const KEY: [u8; 32] = *b"es-hil-test-key-0123456789abcdef";

const GATE_TICKS: u64 = 2_000;
/// The committed format fixture is a short run of the same loop.
const FIXTURE_TICKS: u64 = 120;

/// Control period, and the `dt` the plant differentiates with.
const PERIOD: Duration = Duration::from_micros(1_000);
const DT: f64 = 0.001;

// Deliberate perturbations, all placed by the count of *States received*, so that where they
// land in the stream does not change with the host's sleep granularity.
/// Wall-clock silence the controller keeps after sending an injected command — the late one and
/// the `NaN` row. Without it the gate goes vacuous under host load: one tick bins every command
/// that arrived since the last one and the highest `seq` wins (design note section 7.3 step 3),
/// so an injection is judged only if nothing newer shares its bin. Counting that silence in
/// States buys none, because a stalled loop hands the controller a backlog of States it consumes
/// in microseconds; the silence has to be real time.
const QUIET_AFTER_INJECTION: Duration = Duration::from_millis(12);
/// Two commands are held back and released `LATE_BY` States later; one would be enough to
/// prove the deadline-miss path, two give the gate margin on a host that happens to bin them
/// differently.
const LATE_AT: [u64; 2] = [300, 1_200];
const LATE_BY: u64 = 40;
const SWAP_AT: u64 = 500;
const NAN_AT: u64 = 700;
const BEAT_PAUSE: std::ops::Range<u64> = 900..980;
const BURST_EVERY: u64 = 97;

// --- the Deployment IR fixture --------------------------------------------------------------

/// 1 kHz control, an inference budget, a heartbeat timeout, a violation-rate watchdog and
/// `HoldPosition`. The envelope is deliberately wide: widening is the only way to give a test
/// room, because nothing can switch a constraint off (INV-12).
fn fixture_ir() -> DeploymentIr {
    DeploymentIr {
        schema_version: SCHEMA_VERSION,
        robot: RobotRef {
            name: "hil-rig".into(),
            target: RobotTarget::Simulated {
                scene: "scenes/hil.usd".into(),
            },
            n_joints: NJ,
        },
        action: ActionContract {
            space: ActionSpace::JointPosition,
            dim: NJ,
            horizon: H,
            execute_chunk: ROWS as usize,
        },
        safety: SafetyEnvelope {
            position: vec![
                Limit {
                    lower: -2.0,
                    upper: 2.0
                };
                NJ
            ],
            position_soft_margin: vec![0.1; NJ],
            velocity_max: vec![8.0; NJ],
            acceleration_max: vec![20_000.0; NJ],
            torque_max: vec![50.0; NJ],
            jerk_max: None,
            action_rate: RateLimit {
                first_diff_max: vec![0.5; NJ],
                second_diff_max: vec![1.0; NJ],
            },
            workspace: Workspace::Box {
                min: [-1.0; 3],
                max: [1.0; 3],
            },
            ee_velocity_max: 4.0,
            min_self_distance: 0.01,
            min_env_distance: 0.01,
            contact_force_max: 100.0,
        },
        execution: ExecutionMode::RecedingHorizon,
        deadlines: Deadlines {
            observation_age: Micros(200_000),
            inference_budget: Micros(20_000),
            actuation_budget: Micros(500),
        },
        watchdogs: WatchdogSet(vec![
            Watchdog::InferenceDeadline {
                budget: Micros(20_000),
            },
            Watchdog::ControllerHeartbeat {
                timeout: Micros(20_000),
            },
            // Armed, at the only threshold a cold-started run can survive. The plane reads
            // the rate as dirty/observed over the window, and a HIL run's very first tick has
            // no chunk yet (the controller cannot answer a `State` it has not received), so
            // the rate is 1.0 after step one and *any* threshold below 1.0 latches the whole
            // run into fallback -- including the fallback steps that then keep it there.
            // Widening is the sanctioned way to give a test room (INV-12); see the packet
            // report's open question about this interaction.
            Watchdog::EnvelopeViolationRate {
                window: 256,
                max_frac: 1.0,
            },
        ]),
        fallback: FallbackPolicy::HoldPosition,
        rate: RateSpec {
            control: TickRate::hz(1_000),
            inference: TickRate::hz(100),
        },
    }
}

fn deployment_hash() -> [u8; 32] {
    fixture_ir().deployment_hash().expect("hashable IR")
}

// --- a log sink a test can read back --------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct SharedLog(Arc<Mutex<Vec<u8>>>);

impl SharedLog {
    fn bytes(&self) -> Vec<u8> {
        self.0.lock().expect("log lock").clone()
    }
}

impl Write for SharedLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("log lock").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn new_core(log: &SharedLog) -> HilCore<NJ, H> {
    let ir = fixture_ir();
    assert!(ir.validate().is_empty(), "the fixture IR must validate");
    HilCore::from_ir(&ir, Box::new(log.clone())).expect("HilCore from the fixture IR")
}

fn new_link(log: &SharedLog, stats_every: u64) -> HilLink<NJ, H> {
    let cfg = HilConfig {
        bind: "127.0.0.1:0".parse().expect("loopback addr"),
        key: KEY,
        stats_every,
        max_datagram: wire::MAX_DATAGRAM,
    };
    HilLink::bind(&cfg, new_core(log)).expect("bind the HIL link")
}

// --- the controller under test --------------------------------------------------------------

fn xorshift(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

/// A triangle wave, 0.001 rad per tick (1 rad/s, well inside the 8 rad/s limit) and phased per
/// joint, so a clean tick really is clean and only the injected bursts clamp.
fn traj(n: u64, j: usize) -> f64 {
    let p = (n + j as u64 * 137) % 1_000;
    let up = if p < 500 { p } else { 1_000 - p };
    up as f64 * 0.001 - 0.25
}

fn command_for(obs_tick: PhysTick, n: u64) -> HilMsg<NJ, H> {
    let mut actions = [[0.0f64; NJ]; H];
    for (r, row) in actions.iter_mut().enumerate().take(ROWS as usize) {
        for (j, v) in row.iter_mut().enumerate() {
            *v = traj(n + r as u64, j);
        }
    }
    if n % BURST_EVERY == 0 {
        // 0.6 rad in one tick is 600 rad/s: the velocity stage clamps it.
        for v in &mut actions[0] {
            *v += 0.6;
        }
    }
    if n == NAN_AT {
        actions[0] = [f64::NAN; NJ];
    }
    HilMsg::Command(Command {
        obs_tick,
        rows: ROWS,
        actions,
    })
}

/// The external controller: its own thread, its own socket, its own clock. Everything it does
/// to the stream — delay, loss, the swapped pair, the late command, the `NaN`, the heartbeat
/// gap — is placed by a fixed-seed xorshift and by the count of States it has seen, not by the
/// host's wall clock. The one thing the wall clock decides is [`QUIET_AFTER_INJECTION`]: how
/// long the controller stays silent after an injected command, which is what keeps that command
/// the newest one in its tick bin under load.
fn controller(link_addr: SocketAddr, stop: &AtomicBool) {
    let sock = UdpSocket::bind("127.0.0.1:0").expect("controller socket");
    sock.set_nonblocking(true).expect("non-blocking");
    let hash = deployment_hash();
    let started = Instant::now();
    let mut session = 0u64;
    let mut seq = 0u64;
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut states = 0u64;
    let mut queue: VecDeque<(Instant, HilMsg<NJ, H>)> = VecDeque::new();
    let mut last_due = Instant::now();
    let mut held: VecDeque<HilMsg<NJ, H>> = VecDeque::new();
    let mut swap_hold: Option<HilMsg<NJ, H>> = None;
    let mut next_hello = Instant::now();
    let mut quiet_until = Instant::now();
    let mut tx = Vec::new();
    let mut rx = vec![0u8; wire::MAX_DATAGRAM];

    while !stop.load(Ordering::Relaxed) {
        if session == 0 && Instant::now() >= next_hello {
            seq += 1;
            let hello = HilMsg::Hello {
                nj: NJ as u32,
                h: H as u32,
                deployment_hash: hash,
            };
            wire::encode::<NJ, H>(&KEY, 0, seq, 0, &hello, &mut tx);
            let _ = sock.send_to(&tx, link_addr);
            next_hello = Instant::now() + Duration::from_millis(20);
        }

        while let Ok((n, _)) = sock.recv_from(&mut rx) {
            let Ok((hdr, msg)) = wire::decode::<NJ, H>(&KEY, &rx[..n]) else {
                continue;
            };
            match msg {
                HilMsg::HelloAck { .. } => session = hdr.session_id,
                HilMsg::Bye { .. } => return,
                HilMsg::State { tick, .. } => {
                    if session == 0 {
                        continue;
                    }
                    states += 1;
                    if states % 4 == 0 && !BEAT_PAUSE.contains(&states) {
                        enqueue(&mut queue, &mut last_due, &mut rng, HilMsg::Heartbeat);
                    }
                    // The two injections that must survive to be judged -- the late command
                    // and the `NaN` row -- are each followed by real-time silence, so the link
                    // cannot supersede them with a fresher command in the same tick.
                    let quiet = Instant::now() < quiet_until;
                    // 5 % loss: that answer is simply never produced. The `NaN` is exempt.
                    let send = !quiet && (states == NAN_AT || xorshift(&mut rng) % 20 != 0);
                    if send {
                        let cmd = command_for(tick, states);
                        if LATE_AT.contains(&states) {
                            held.push_back(cmd);
                        } else if states == SWAP_AT {
                            swap_hold = Some(cmd);
                        } else {
                            enqueue(&mut queue, &mut last_due, &mut rng, cmd);
                            if let Some(msg) = swap_hold.take() {
                                // The swapped pair: the older observation now travels behind
                                // the newer one.
                                enqueue(&mut queue, &mut last_due, &mut rng, msg);
                            }
                            if states == NAN_AT {
                                quiet_until = last_due + QUIET_AFTER_INJECTION;
                            }
                        }
                    }
                    if LATE_AT.iter().any(|a| states == a + LATE_BY) {
                        if let Some(msg) = held.pop_front() {
                            enqueue(&mut queue, &mut last_due, &mut rng, msg);
                            quiet_until = last_due + QUIET_AFTER_INJECTION;
                        }
                    }
                }
                _ => {}
            }
        }

        let now = Instant::now();
        while queue.front().is_some_and(|(due, _)| *due <= now) {
            let (_, msg) = queue.pop_front().expect("a due message");
            seq += 1;
            let mono = started.elapsed().as_nanos() as u64;
            wire::encode::<NJ, H>(&KEY, session, seq, mono, &msg, &mut tx);
            let _ = sock.send_to(&tx, link_addr);
        }
        // May overshoot badly on a coarse timer; harmless, because the queue is drained by
        // deadline rather than by iteration count, so nothing accumulates.
        thread::sleep(Duration::from_micros(200));
    }
}

/// FIFO delay line: 0-3 ms from the fixed-seed xorshift, never reordering by itself.
fn enqueue(
    queue: &mut VecDeque<(Instant, HilMsg<NJ, H>)>,
    last_due: &mut Instant,
    rng: &mut u64,
    msg: HilMsg<NJ, H>,
) {
    let delay = Duration::from_millis(xorshift(rng) % 4);
    let due = (Instant::now() + delay).max(*last_due);
    *last_due = due;
    queue.push_back((due, msg));
}

/// One live run: `ticks` control ticks at 1 kHz, paced with `thread::sleep` to each tick's
/// absolute deadline. Falling behind is never made up by sleeping less than zero — the loop
/// simply skips the sleep and catches up, which is what keeps the run bounded on a host whose
/// timer granularity is far coarser than the control period.
fn live_run(ticks: u64) -> (Vec<u8>, [u8; 32], LiveOutcome) {
    let log = SharedLog::default();
    let mut link = new_link(&log, 100);
    let addr = link.local_addr();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_ctrl = Arc::clone(&stop);
    let ctrl = thread::spawn(move || controller(addr, &stop_ctrl));

    let start = Instant::now();
    let mut q = [0.0f64; NJ];
    let mut qd = [0.0f64; NJ];
    for t in 0..ticks {
        let action = link.tick(PhysTick(t), &q, &qd);
        // Plant: q := action.q, and the velocity the plant reports is the difference quotient.
        for j in 0..NJ {
            qd[j] = (action.q[j] - q[j]) / DT;
        }
        q = action.q;
        let deadline = start + PERIOD * u32::try_from(t + 1).expect("tick count fits u32");
        if let Some(left) = deadline.checked_duration_since(Instant::now()) {
            thread::sleep(left);
        }
    }
    let wall = start.elapsed();

    stop.store(true, Ordering::Relaxed);
    let stats = link.stats();
    let c = link.counters();
    let outcome = LiveOutcome {
        wall,
        clamped: c.clamped_steps,
        fallback: c.fallback_activations,
        non_finite: c.count(ViolationKind::NonFinite),
        heartbeat_loss: c.count(ViolationKind::HeartbeatLoss),
        commands: stats.commands,
        deadline_miss: stats.deadline_miss,
        superseded: stats.superseded,
        rx_lost: stats.rx_lost,
    };
    let hash = link.finish().expect("write the .eshil trailer");
    ctrl.join().expect("controller thread");
    (log.bytes(), hash, outcome)
}

#[derive(Debug)]
struct LiveOutcome {
    wall: Duration,
    clamped: u64,
    fallback: u64,
    non_finite: u64,
    heartbeat_loss: u64,
    commands: u64,
    deadline_miss: u64,
    superseded: u64,
    rx_lost: u64,
}

// --- the gate ---------------------------------------------------------------------------

#[test]
fn hil_live_run_replays_to_byte_identical_decisions() {
    let (bytes, live_hash, out) = live_run(GATE_TICKS);
    assert!(
        out.commands > 0,
        "the controller never reached the link over loopback UDP; \
         nothing about the Safety Plane can be judged from this run"
    );

    let report = replay::<NJ, H>(&bytes).expect("the log replays");
    assert!(!report.truncated, "a finished run has a trailer");
    assert_eq!(report.steps, GATE_TICKS);
    assert_eq!(
        report.first_divergence, None,
        "first divergence at {:?}",
        report.first_divergence
    );
    assert!(report.identical, "decisions are not byte-identical");
    assert_eq!(report.live_hash, report.replay_hash);
    assert_eq!(report.live_hash, live_hash);
    assert!(report.is_verified(), "{report:?}");

    // Non-vacuity (design note section 7.5): the run must have exercised the plane.
    assert!(out.clamped >= 1, "no clamped step: {out:?}");
    assert!(out.fallback >= 1, "no fallback: {out:?}");
    assert!(out.non_finite >= 1, "no NonFinite event: {out:?}");
    assert!(out.heartbeat_loss >= 1, "no HeartbeatLoss: {out:?}");
    assert!(out.deadline_miss >= 1, "no deadline miss: {out:?}");

    // An observation of one loopback run, not a performance claim (spec 12.4).
    println!(
        "observation: {} ticks of 1 kHz pacing took {:?} wall; \
         commands={} superseded={} rx_lost={}",
        GATE_TICKS, out.wall, out.commands, out.superseded, out.rx_lost
    );
    println!(
        "RAN hil_gate steps={} clamped={} fallback={} deadline_miss={}",
        report.steps, out.clamped, out.fallback, out.deadline_miss
    );
}

// --- the committed format fixture ---------------------------------------------------------

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/hil/v1_small.eshil")
}

fn fixture_bytes() -> Vec<u8> {
    fs::read(fixture_path()).expect("tests/fixtures/hil/v1_small.eshil")
}

/// `tests/fixtures/hil/v1_small.eshil` was produced once by this packet's live run and is
/// committed as a **format fixture**, not a golden: no external reference produces it, and its
/// job is to keep v1 replayable as the code moves (spec 25.3).
///
/// `ES_HIL_WRITE_FIXTURE=1` regenerates it from a fresh short live run; CI never sets it.
#[test]
fn v1_fixture_still_replays_identically() {
    if std::env::var_os("ES_HIL_WRITE_FIXTURE").is_some() {
        let (bytes, _, out) = live_run(FIXTURE_TICKS);
        assert!(out.commands > 0, "regenerating needs a live controller");
        let path = fixture_path();
        fs::create_dir_all(path.parent().expect("fixture dir")).expect("create the fixture dir");
        fs::write(&path, &bytes).expect("write the fixture");
        println!("wrote {} ({} bytes)", path.display(), bytes.len());
    }
    let bytes = fixture_bytes();
    let report = replay::<NJ, H>(&bytes).expect("the v1 fixture replays");
    assert!(!report.truncated);
    assert!(report.identical, "{report:?}");
    assert_eq!(report.live_hash, report.replay_hash);
    assert!(report.steps > 0);
    assert!(report.is_verified(), "{report:?}");
}

// --- log-level oracles ---------------------------------------------------------------------

/// Offset of the first record: the fixed header plus the embedded IR's JSON.
fn first_record(log: &[u8]) -> usize {
    let ir_len = u32::from_le_bytes(log[64..68].try_into().expect("4 bytes")) as usize;
    68 + ir_len
}

/// `(tag, body offset, body length, next offset)` for every record in `log`.
fn records(log: &[u8]) -> Vec<(u8, usize, usize, usize)> {
    let mut out = Vec::new();
    let mut i = first_record(log);
    while i + 5 <= log.len() {
        let len = u32::from_le_bytes(log[i + 1..i + 5].try_into().expect("4 bytes")) as usize;
        if i + 5 + len > log.len() {
            break;
        }
        out.push((log[i], i + 5, len, i + 5 + len));
        i += 5 + len;
    }
    out
}

/// The fixture with one clean `Step`'s first action word flipped: `(bytes, record index of the
/// step, its tick)`.
fn tampered_fixture() -> (Vec<u8>, usize, u64) {
    let mut bytes = fixture_bytes();
    let recs = records(&bytes);
    // A `Step` that carried a chunk and whose `Decision` came out of the plane untouched: a
    // one-bit change to its action must then show up in the decision, with nothing else to
    // blame it on.
    let (idx, body) = recs
        .iter()
        .enumerate()
        .find(|(i, (tag, body, _, _))| {
            *tag == 0x03
                && bytes[body + 16] == 1
                && recs
                    .get(i + 1)
                    .is_some_and(|(t, b, _, _)| *t == 0x10 && bytes[b + 8 + 8 * NJ] == 0)
        })
        .map(|(i, (_, body, _, _))| (i, *body))
        .expect("a clean step in the fixture");
    let tick = u64::from_le_bytes(bytes[body..body + 8].try_into().expect("8 bytes"));
    // Actions start after `tick`, `obs_age_us`, `has_chunk` and `rows`.
    bytes[body + 19] ^= 0x01;
    (bytes, idx, tick)
}

#[test]
fn tampered_step_diverges_at_its_tick() {
    let (bytes, idx, tick) = tampered_fixture();
    let report = replay::<NJ, H>(&bytes).expect("a tampered log still parses");
    assert!(!report.identical);
    assert_ne!(report.live_hash, report.replay_hash);
    let (index, at) = report.first_divergence.expect("a divergence");
    assert_eq!(at, PhysTick(tick), "divergence is at the tampered tick");
    // The `Decision` records before this step are untouched, so the index is its step index.
    let decisions_before = records(&bytes)[..idx]
        .iter()
        .filter(|r| r.0 == 0x10)
        .count();
    assert_eq!(index, decisions_before as u64);
}

#[test]
fn truncated_log_replays_its_complete_prefix() {
    let full = fixture_bytes();
    let recs = records(&full);
    let cut_at = recs[recs.len() / 2];
    // Halfway into a record body: the last complete record is the one before it.
    let mut cut = full.clone();
    cut.truncate(cut_at.1 + cut_at.2 / 2);

    let whole = replay::<NJ, H>(&full).expect("the whole log replays");
    let prefix = replay::<NJ, H>(&cut).expect("the prefix replays");
    assert!(prefix.truncated);
    assert!(!whole.truncated);
    assert!(prefix.identical);
    assert!(prefix.steps > 0 && prefix.steps < whole.steps);
    assert_eq!(prefix.live_hash, prefix.replay_hash);
    // A correct replay of an incomplete run is not a verified one.
    assert!(!prefix.is_verified());
}

/// A header and a trailer with nothing between them satisfies every field the report carries --
/// both hashes are blake3 of nothing -- so the report itself has to say it verified nothing
/// (spec 1.4, design note section 7.5).
#[test]
fn a_log_with_no_decisions_does_not_verify() {
    let full = fixture_bytes();
    let mut empty = full[..first_record(&full)].to_vec();
    let mut body = Vec::new();
    body.extend_from_slice(blake3::hash(&[]).as_bytes());
    body.extend_from_slice(&0u64.to_le_bytes());
    empty.push(0xFF);
    empty.extend_from_slice(&u32::try_from(body.len()).expect("40").to_le_bytes());
    empty.extend_from_slice(&body);

    let report = replay::<NJ, H>(&empty).expect("a header plus a trailer parses");
    assert_eq!(report.steps, 0);
    assert!(!report.truncated, "it has a trailer");
    // Both are the hash of nothing; that is exactly the point.
    assert_eq!(report.live_hash, report.replay_hash);
    assert!(!report.identical, "nothing was compared");
    assert!(!report.is_verified());
}

#[test]
fn a_header_only_log_does_not_verify() {
    let full = fixture_bytes();
    let report = replay::<NJ, H>(&full[..first_record(&full)]).expect("a header alone parses");
    assert!(report.truncated);
    assert_eq!(report.steps, 0);
    assert!(!report.is_verified());
}

/// The one predicate a caller should read, on the three logs whose answers are known.
#[test]
fn the_gate_predicate_is_the_gate() {
    let full = fixture_bytes();
    assert!(replay::<NJ, H>(&full)
        .expect("the fixture replays")
        .is_verified());

    let (tampered, _, _) = tampered_fixture();
    assert!(!replay::<NJ, H>(&tampered)
        .expect("a tampered log still parses")
        .is_verified());

    let recs = records(&full);
    let cut_at = recs[recs.len() / 2];
    let mut cut = full.clone();
    cut.truncate(cut_at.1 + cut_at.2 / 2);
    assert!(!replay::<NJ, H>(&cut)
        .expect("the prefix replays")
        .is_verified());
}

#[test]
fn foreign_deployment_hash_is_rejected() {
    let mut bytes = fixture_bytes();
    // The header claims a deployment the embedded IR does not hash to.
    bytes[32..64].copy_from_slice(&[0xA5u8; 32]);
    assert_eq!(
        replay::<NJ, H>(&bytes),
        Err(HilLogError::DeploymentHash),
        "a log must not replay against a plane it was not recorded under"
    );
}

#[test]
fn shape_mismatch_is_rejected() {
    let bytes = fixture_bytes();
    assert_eq!(
        replay::<4, H>(&bytes),
        Err(HilLogError::Shape {
            nj: NJ as u32,
            h: H as u32,
            want_nj: 4,
            want_h: H,
        })
    );
}

// --- link-level oracles ---------------------------------------------------------------------

/// A raw peer socket: the test's stand-in for a controller, with no delay line and no loss, so
/// every datagram it sends is under the test's control.
struct Peer {
    sock: UdpSocket,
    link: SocketAddr,
    session: u64,
    seq: u64,
    buf: Vec<u8>,
}

impl Peer {
    fn open(link: SocketAddr) -> Peer {
        let sock = UdpSocket::bind("127.0.0.1:0").expect("peer socket");
        sock.set_read_timeout(Some(Duration::from_millis(500)))
            .expect("read timeout");
        Peer {
            sock,
            link,
            session: 0,
            seq: 0,
            buf: Vec::new(),
        }
    }

    /// Encodes with an explicit session and sequence number, so a test can forge both.
    fn frame(&mut self, session: u64, seq: u64, msg: &HilMsg<NJ, H>) -> Vec<u8> {
        wire::encode::<NJ, H>(&KEY, session, seq, 0, msg, &mut self.buf);
        self.buf.clone()
    }

    /// Returns the exact datagram it sent, so a test can keep it the way a recorder would.
    fn put(&mut self, msg: &HilMsg<NJ, H>) -> Vec<u8> {
        self.seq += 1;
        let (session, seq) = (self.session, self.seq);
        let bytes = self.frame(session, seq, msg);
        self.put_raw(&bytes);
        bytes
    }

    fn put_raw(&mut self, bytes: &[u8]) {
        self.sock.send_to(bytes, self.link).expect("send");
    }

    fn recv(&mut self) -> Option<(wire::Header, HilMsg<NJ, H>)> {
        let mut rx = vec![0u8; wire::MAX_DATAGRAM];
        let (n, _) = self.sock.recv_from(&mut rx).ok()?;
        wire::decode::<NJ, H>(&KEY, &rx[..n]).ok()
    }
}

/// Sends `Hello`, ticks the link once so it is drained, and reads the `HelloAck`.
///
/// Everything is settled with a sleep *before* the tick rather than by ticking until something
/// arrives, so the tick at which each datagram lands is fixed and two runs of the same script
/// produce the same log.
/// Returns the raw `Hello` datagram, which a passive recorder on the same host would also have.
fn handshake(link: &mut HilLink<NJ, H>, peer: &mut Peer, t: &mut u64) -> Vec<u8> {
    let hello = peer.put(&HilMsg::Hello {
        nj: NJ as u32,
        h: H as u32,
        deployment_hash: deployment_hash(),
    });
    settle();
    step(link, t);
    peer.session = recv_ack(peer);
    hello
}

/// The session id of the next `HelloAck`, skipping the `State`s the link sends every tick.
fn recv_ack(peer: &mut Peer) -> u64 {
    for _ in 0..64 {
        match peer.recv() {
            Some((hdr, HilMsg::HelloAck { .. })) => return hdr.session_id,
            // A `State`: the link sends one every tick, so they queue up ahead of the ack.
            Some(_) => {}
            None => break,
        }
    }
    panic!("expected a HelloAck");
}

fn settle() {
    thread::sleep(Duration::from_millis(60));
}

fn step(link: &mut HilLink<NJ, H>, t: &mut u64) {
    link.tick(PhysTick(*t), &[0.0; NJ], &[0.0; NJ]);
    *t += 1;
}

fn small_command(obs_tick: PhysTick) -> HilMsg<NJ, H> {
    let mut actions = [[0.0f64; NJ]; H];
    for (r, row) in actions.iter_mut().enumerate().take(ROWS as usize) {
        *row = [0.001 * (r + 1) as f64; NJ];
    }
    HilMsg::Command(Command {
        obs_tick,
        rows: ROWS,
        actions,
    })
}

#[test]
fn bad_tag_wrong_session_and_stale_seq_never_reach_the_plane() {
    // Two runs of the same script; only the first is also fed three refused datagrams.
    let run = |feed_garbage: bool| {
        let log = SharedLog::default();
        let mut link = new_link(&log, 100);
        let mut peer = Peer::open(link.local_addr());
        let mut t = 0u64;
        handshake(&mut link, &mut peer, &mut t);

        peer.put(&small_command(PhysTick(t)));
        settle();
        step(&mut link, &mut t);

        if feed_garbage {
            let (session, seq) = (peer.session, peer.seq);
            let mut bad_tag = peer.frame(session, seq + 1, &HilMsg::Heartbeat);
            let last = bad_tag.len() - 1;
            bad_tag[last] ^= 0xFF;
            peer.put_raw(&bad_tag);

            let wrong_session = peer.frame(session ^ 0xDEAD, seq + 1, &HilMsg::Heartbeat);
            peer.put_raw(&wrong_session);

            // `seq` the link already accepted: stale, whatever it carries.
            let stale = peer.frame(session, seq, &small_command(PhysTick(t)));
            peer.put_raw(&stale);
        }
        settle();
        step(&mut link, &mut t);
        step(&mut link, &mut t);

        let stats = link.stats();
        let c = link.counters();
        let plane = (
            c.steps,
            c.clamped_steps,
            c.fallback_activations,
            c.dirty_steps,
            c.violations,
        );
        link.finish().expect("trailer");
        (log.bytes(), stats, plane)
    };

    let (log_bad, stats_bad, plane_bad) = run(true);
    let (log_clean, stats_clean, plane_clean) = run(false);

    assert_eq!(stats_bad.rx_invalid, 2, "bad tag and wrong session");
    assert_eq!(stats_bad.rx_stale, 1, "a seq the link already accepted");
    assert_eq!((stats_clean.rx_invalid, stats_clean.rx_stale), (0, 0));
    // Same commands, same heartbeats, same decisions: the refused datagrams changed nothing.
    assert_eq!(stats_bad.commands, stats_clean.commands);
    assert_eq!(stats_bad.heartbeats, stats_clean.heartbeats);
    assert_eq!(plane_bad, plane_clean, "the plane's counters must be equal");
    assert_eq!(log_bad, log_clean, "the log must be byte-identical");
}

/// A recorded session must not replay (design note section 7.7, spec 25.1). The `Hello` rewinds
/// `last_seq`, so freshness has to come from the session id.
#[test]
fn a_replayed_session_does_not_reach_the_plane() {
    let log = SharedLog::default();
    let mut link = new_link(&log, 100);
    let mut peer = Peer::open(link.local_addr());
    let mut t = 0u64;
    // Everything a passive recorder on the same host would have kept.
    let hello = handshake(&mut link, &mut peer, &mut t);
    let first_session = peer.session;

    let command = peer.put(&small_command(PhysTick(t)));
    settle();
    step(&mut link, &mut t);
    let before = link.stats();
    assert_eq!(before.commands, 1, "the live command reached the plane");

    // The recorded `Hello`, byte for byte: still authentic, so the link accepts it.
    peer.put_raw(&hello);
    settle();
    step(&mut link, &mut t);
    let second_session = recv_ack(&mut peer);
    assert_ne!(
        second_session, first_session,
        "a Hello must open a fresh session"
    );

    // The recorded `Command` is now authenticated under a dead session.
    peer.put_raw(&command);
    settle();
    step(&mut link, &mut t);
    let after = link.stats();
    assert_eq!(
        after.commands, before.commands,
        "a replayed command must not reach the plane"
    );
    assert_eq!(after.rx_invalid, before.rx_invalid + 1);

    link.finish().expect("trailer");
}

/// The fix must not break reconnection: after the new `Hello`, a freshly framed command on the
/// new session is accepted as usual.
#[test]
fn a_second_hello_starts_a_clean_session() {
    let log = SharedLog::default();
    let mut link = new_link(&log, 100);
    let mut peer = Peer::open(link.local_addr());
    let mut t = 0u64;
    let hello = handshake(&mut link, &mut peer, &mut t);

    peer.put(&small_command(PhysTick(t)));
    settle();
    step(&mut link, &mut t);
    assert_eq!(link.stats().commands, 1);

    peer.put_raw(&hello);
    settle();
    step(&mut link, &mut t);
    peer.session = recv_ack(&mut peer);

    peer.put(&small_command(PhysTick(t)));
    settle();
    step(&mut link, &mut t);
    assert_eq!(
        link.stats().commands,
        2,
        "the controller can still reconnect"
    );

    link.finish().expect("trailer");
}

#[test]
fn hello_with_a_wrong_deployment_hash_gets_bye() {
    let log = SharedLog::default();
    let mut link = new_link(&log, 100);
    let mut peer = Peer::open(link.local_addr());
    peer.put(&HilMsg::Hello {
        nj: NJ as u32,
        h: H as u32,
        deployment_hash: [0xFFu8; 32],
    });
    settle();
    let mut t = 0u64;
    step(&mut link, &mut t);
    match peer.recv() {
        Some((_, HilMsg::Bye { reason })) => assert_eq!(reason, wire::BYE_HASH_MISMATCH),
        other => panic!("expected Bye {{ reason: 1 }}, got {other:?}"),
    }
    link.finish().expect("trailer");
}

#[test]
fn late_command_counts_a_deadline_miss_and_still_reaches_the_plane() {
    // Socket-free: the events go straight into `HilCore` with explicit ticks.
    let log = SharedLog::default();
    let mut core = new_core(&log);
    core.observe_state(PhysTick(0), &[0.0; NJ], &[0.0; NJ]);
    core.on_heartbeat();
    let mut actions = [[0.0f64; NJ]; H];
    actions[0] = [0.001, 0.002, -0.001];
    core.on_command(Command {
        // 90 ticks behind, against a 20 ms budget at 1 kHz: 20 ticks of slack.
        obs_tick: PhysTick(10),
        rows: ROWS,
        actions,
    });
    let action = core.tick(PhysTick(100));

    assert_eq!(core.stats().deadline_miss, 1);
    assert_eq!(core.stats().commands, 1);
    assert_eq!(
        action.source,
        ActionSource::Policy,
        "a late command is the watchdog's call, not the link's: it is still submitted"
    );
    // Bit-exact: a clean tick emits the commanded row unchanged.
    assert_eq!(
        action.q.map(f64::to_bits),
        actions[0].map(f64::to_bits),
        "the command reached the plane and came back unchanged"
    );
    assert_eq!(core.counters().count(ViolationKind::ChunkUnderrun), 0);
    core.finish().expect("trailer");
}

#[test]
fn stats_frames_reach_a_telemetry_client() {
    let server = Server::bind("127.0.0.1:0".parse().expect("addr"), None).expect("bind telemetry");
    let mut client =
        Client::connect(server.local_addr(), None, "hil-gate-test").expect("connect telemetry");
    client
        .subscribe(vec![HIL_STATS_STREAM])
        .expect("subscribe to the HIL stats stream");
    // The server registers the subscription on its own thread.
    thread::sleep(Duration::from_millis(200));

    let log = SharedLog::default();
    let mut link = new_link(&log, 5);
    let mut peer = Peer::open(link.local_addr());
    let mut t = 0u64;
    handshake(&mut link, &mut peer, &mut t);
    // An observation 30 ticks old, against 20 ticks of slack: one deadline miss.
    peer.put(&small_command(PhysTick(0)));
    settle();
    t = 30;
    for _ in 0..40 {
        link.tick(PhysTick(t), &[0.0; NJ], &[0.0; NJ]);
        link.publish_stats(&server, PhysTick(t));
        t += 1;
    }
    assert!(link.stats().deadline_miss >= 1, "{:?}", link.stats());

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut seen = None;
    while Instant::now() < deadline && seen.is_none() {
        match client.try_recv() {
            Ok(Message::Frame(frame)) if frame.stream == HIL_STATS_STREAM => {
                let Payload::Scalars(values) = frame.payload else {
                    panic!("HIL stats must be Payload::Scalars");
                };
                if values[2] >= 1.0 {
                    seen = Some(values);
                }
            }
            Ok(_) | Err(TransportError::WouldBlock) => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("telemetry client: {e}"),
        }
    }
    let values = seen.expect("a HIL stats frame with a deadline miss");
    // Field order of design note section 7.6, positional and fixed.
    assert_eq!(values.len(), 10);
    let as_u64 = |i: usize| values[i] as u64;
    let stats = link.stats();
    assert!(as_u64(0) > 0 && as_u64(0) <= stats.ticks, "ticks");
    assert!(as_u64(1) >= 1, "commands");
    assert!(as_u64(2) >= 1, "deadline_miss");
    assert_eq!(as_u64(3), 0, "rx_lost");
    assert_eq!(as_u64(4), 0, "rx_stale");
    assert_eq!(as_u64(5), 0, "rx_invalid");
    assert_eq!(as_u64(7), 0, "heartbeats: this peer sends none");
    link.finish().expect("trailer");
}

// --- boundary ----------------------------------------------------------------------------

/// `hil` must stay extractable into its own crate by a file move (design note section 2), and
/// the wall clock must stay confined to `link.rs` and `stats.rs` (spec 3.4, spec 3.5).
#[test]
fn hil_imports_no_ros_module() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/hil");
    let ros_modules = [
        "cdr",
        "msg",
        "names",
        "attachment",
        "session",
        "config",
        "actuator",
        "camera",
    ];
    let mut checked = 0;
    for entry in fs::read_dir(&dir).expect("src/hil") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let name = path
            .file_name()
            .expect("file name")
            .to_string_lossy()
            .into_owned();
        let src = fs::read_to_string(&path).expect("read");
        checked += 1;
        for m in ros_modules {
            assert!(
                !src.contains(&format!("crate::{m}")),
                "{name} reaches into the ROS module `{m}`"
            );
        }
        if name != "link.rs" && name != "stats.rs" {
            for clock in ["Instant", "SystemTime"] {
                assert!(
                    !src.contains(clock),
                    "{name} names `{clock}`; the wall clock belongs in link.rs / stats.rs only"
                );
            }
        }
    }
    assert_eq!(
        checked, 7,
        "src/hil has mod, wire, core, link, log, replay, stats"
    );
}
