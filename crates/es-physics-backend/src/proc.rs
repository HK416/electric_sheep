//! The line-delimited JSON protocol to `python/mujoco_ref.py`, and the process that speaks it.
//!
//! One request per line in, one JSON object per line out. Errors on the Python side come back
//! as `{"ok": false, "error": ...}`, so a modelling mistake is a typed [`PhysicsError`] and
//! never a dead process. Bulk floats (`state` replies and `set_ctrl` from/to `mujoco_ref.py`
//! and `mjwarp_ref.py`) cross as raw `f64` bytes after their line instead ([`FrameHeader`],
//! packet M16/H0).
//!
//! The script is embedded with `include_str!` and handed to `python -c`, so there is no
//! installed-data-file lookup at runtime and editing the script forces a rebuild.

use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use es_assets::scene::SceneDesc;
use es_core::child::{retry_start, transient_start_failure};
use es_core::{StableId, TickRate};
use es_physics_core::backend::Param;
use es_physics_core::{IndexRange, ModelInfo, PhysicsError};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// The reference script, embedded at build time.
pub const SCRIPT: &str = include_str!("../python/mujoco_ref.py");

/// One request to the reference process.
#[derive(Debug, Serialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request<'a> {
    Load {
        mjcf: &'a str,
        n_envs: u32,
        /// `None` keeps the timestep in the MJCF.
        timestep: Option<f64>,
        seed: u64,
    },
    Reset {
        envs: Option<&'a [u32]>,
        state: Option<StatePayload<'a>>,
    },
    SetCtrl {
        ctrl: &'a [f64],
    },
    /// `set_ctrl` to `mujoco_ref.py` / `mjwarp_ref.py`: the `frame` values follow the line as
    /// raw little-endian `f64` bytes ([`Process::set_ctrl_frame`], packet M16/H0).
    #[serde(rename = "set_ctrl")]
    SetCtrlFrame {
        frame: usize,
    },
    Step {
        n: u32,
    },
    State,
    SetState {
        state: StatePayload<'a>,
    },
    /// Scales model parameters of `envs` (packet M11/X4); the reply is [`SetParamsReply`].
    SetParams {
        envs: &'a [u32],
        params: &'a [ParamWire],
    },
    Quit,
}

/// One `set_params` entry on the wire: which model field, where in it, and the scale.
///
/// `index` is the engine's body row for `body_mass`, its actuator row for `actuator_gain`,
/// and for `geom_friction` the owning body's row with `sub` the geom's position among that
/// body's geoms -- the emitter writes a body's geoms in scene order and `MuJoCo` keeps them
/// contiguous from `body_geomadr`, so no geom name has to be reconstructed here.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ParamWire {
    pub field: &'static str,
    pub index: u32,
    pub sub: u32,
    pub scale: f64,
}

/// The `set_params` reply: `[nominal, applied]` read back out of each edited model, env-major,
/// one pair per entry.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct SetParamsReply {
    pub values: Vec<[f64; 2]>,
}

/// Where each parameter target of `scene` lives in the loaded model, as `(index, sub)` of a
/// [`ParamWire`]. Built once at load; a target missing from it is refused by name.
pub(crate) fn param_index(
    scene: &SceneDesc,
    info: &ModelInfo,
) -> BTreeMap<(Param, StableId), (u32, u32)> {
    let mut out = BTreeMap::new();
    for body in &scene.bodies {
        let Some(row) = info.body.get(&body.id) else {
            continue;
        };
        out.insert((Param::BodyMass, body.id), (row.start, 0));
        for (k, geom) in body.geoms.iter().enumerate() {
            out.insert((Param::GeomFriction, geom.id), (row.start, k as u32));
        }
    }
    for (id, row) in &info.actuator {
        out.insert((Param::ActuatorGain, *id), (row.start, 0));
    }
    out
}

/// The wire form of `params`, or `Unsupported` naming the first target the model lacks.
pub(crate) fn param_wire(
    index: &BTreeMap<(Param, StableId), (u32, u32)>,
    params: &[(Param, StableId, f64)],
) -> Result<Vec<ParamWire>, PhysicsError> {
    params
        .iter()
        .map(|&(param, id, scale)| {
            let &(index, sub) = index.get(&(param, id)).ok_or_else(|| {
                PhysicsError::Unsupported(format!(
                    "set_params: {param:?} of {id:?} is not in the loaded model"
                ))
            })?;
            let field = match param {
                Param::BodyMass => "body_mass",
                Param::GeomFriction => "geom_friction",
                Param::ActuatorGain => "actuator_gain",
            };
            Ok(ParamWire {
                field,
                index,
                sub,
                scale,
            })
        })
        .collect()
}

/// `(env, param, id)`: one parameter of one env, as `set_params` applied it.
pub type AppliedKey = (u32, Param, StableId);

/// Checks `envs` against the batch and `reply` against the request, then returns
/// `(env, param, id) -> [nominal, applied]` for every entry.
pub(crate) fn applied_values(
    envs: &[u32],
    params: &[(Param, StableId, f64)],
    reply: &SetParamsReply,
) -> Result<Vec<(AppliedKey, [f64; 2])>, PhysicsError> {
    if reply.values.len() != envs.len() * params.len() {
        return Err(PhysicsError::Protocol(format!(
            "set_params answered {} values for {} envs x {} params",
            reply.values.len(),
            envs.len(),
            params.len()
        )));
    }
    Ok(envs
        .iter()
        .flat_map(|env| params.iter().map(move |&(p, id, _)| (*env, p, id)))
        .zip(reply.values.iter().copied())
        .collect())
}

/// `Err` naming the first env outside a batch of `n_envs`.
pub(crate) fn check_envs(envs: &[u32], n_envs: u32) -> Result<(), PhysicsError> {
    match envs.iter().find(|e| **e >= n_envs) {
        Some(bad) => Err(PhysicsError::Backend(format!(
            "env {bad} is out of range for a batch of {n_envs}"
        ))),
        None => Ok(()),
    }
}

/// Env-major state arrays on the wire.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct StatePayload<'a> {
    pub qpos: &'a [f64],
    pub qvel: &'a [f64],
    pub act: &'a [f64],
}

/// The `load` reply: the model's shape and its name-to-index maps.
#[derive(Clone, Debug, Deserialize)]
pub struct LoadReply {
    pub nq: u32,
    pub nv: u32,
    pub nu: u32,
    pub nsensordata: u32,
    pub nbody: u32,
    pub joints: Vec<JointEntry>,
    pub actuators: Vec<String>,
    pub sensors: Vec<SensorEntry>,
    pub bodies: Vec<String>,
    /// The engine the process runs, e.g. `mujoco 3.3.2` -- what
    /// [`backend_identity`](crate::backend_identity) hashes (packet M11/X1). Required: a reply
    /// without it is a protocol error, never an empty string in a hash.
    pub engine_version: String,
}

impl LoadReply {
    /// The engine version, refused when blank.
    pub fn checked_engine_version(&self) -> Result<&str, PhysicsError> {
        let v = self.engine_version.trim();
        if v.is_empty() {
            return Err(PhysicsError::Protocol(
                "the load reply names no engine version".to_owned(),
            ));
        }
        Ok(v)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct JointEntry {
    pub name: String,
    /// `[address, width]` in `qpos`.
    pub qpos: [u32; 2],
    /// `[address, width]` in `qvel`.
    pub dof: [u32; 2],
}

#[derive(Clone, Debug, Deserialize)]
pub struct SensorEntry {
    pub name: String,
    pub adr: u32,
    pub dim: u32,
}

/// The `step` reply: which envs left the finite range (spec 18.5).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct StepReply {
    #[serde(default)]
    pub nonfinite: Vec<u32>,
}

/// The `state` reply.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct StateReply {
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    pub act: Vec<f64>,
    pub sensordata: Vec<f64>,
    pub xpos: Vec<f64>,
    pub xquat: Vec<f64>,
}

/// The `state` reply of `mujoco_ref.py` and `mjwarp_ref.py` (packet M16/H0): a line
/// `{"ok": true, "frame": [n_qpos, n_qvel, n_act, n_sensordata, n_xpos, n_xquat]}`, then those
/// arrays' values as raw little-endian `f64` bytes, in that order. The bytes themselves, so a
/// value arrives bit for bit, and a 1,024-env SO-101 state is 0.8 MB of memcpy rather than
/// 1.8 MB of decimal text for Python to format and Rust to parse. `newton_ref.py` and
/// `physx_ref.py` still answer [`StateReply`] as JSON lists.
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct FrameHeader {
    pub frame: [usize; 6],
}

/// Reads the bytes a [`FrameHeader`] announced off `reader` into a [`StateReply`].
pub fn read_frame(reader: &mut impl Read, header: FrameHeader) -> std::io::Result<StateReply> {
    let total: usize = header.frame.iter().sum();
    let mut bytes = vec![0u8; total * 8];
    reader.read_exact(&mut bytes)?;
    let mut values = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("chunks of 8")));
    let [qpos, qvel, act, sensordata, xpos, xquat] =
        header.frame.map(|n| values.by_ref().take(n).collect());
    Ok(StateReply {
        qpos,
        qvel,
        act,
        sensordata,
        xpos,
        xquat,
    })
}

/// A reply with no payload beyond `ok`.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
pub struct Ack {}

/// Decodes one response line: `{"ok": true, ...}` into `T`, `{"ok": false, ...}` into a
/// [`PhysicsError::Backend`], anything else into [`PhysicsError::Protocol`].
pub fn parse_response<T: DeserializeOwned>(line: &str) -> Result<T, PhysicsError> {
    let value: serde_json::Value = serde_json::from_str(line)
        .map_err(|e| PhysicsError::Protocol(format!("{e} in `{}`", truncate(line))))?;
    match value.get("ok") {
        Some(serde_json::Value::Bool(true)) => serde_json::from_value(value)
            .map_err(|e| PhysicsError::Protocol(format!("{e} in `{}`", truncate(line)))),
        Some(serde_json::Value::Bool(false)) => Err(PhysicsError::Backend(
            value
                .get("error")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unspecified")
                .to_owned(),
        )),
        _ => Err(PhysicsError::Protocol(format!(
            "response has no `ok` field: `{}`",
            truncate(line)
        ))),
    }
}

fn truncate(line: &str) -> String {
    let trimmed = line.trim();
    match trimmed.char_indices().nth(200) {
        Some((cut, _)) => format!("{}...", &trimmed[..cut]),
        None => trimmed.to_owned(),
    }
}

/// Interpreters to try, in order. `ES_PYTHON` overrides the search entirely.
pub(crate) fn python_candidates() -> Vec<String> {
    match std::env::var("ES_PYTHON") {
        Ok(path) if !path.trim().is_empty() => vec![path],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    }
}

/// Whether some interpreter can `import` `modules` (a Python import list).
///
/// `Err` carries what was tried and why it failed, so a CI skip message says something useful.
/// Shared by every out-of-process backend; `what` names the packages in that message.
pub(crate) fn import_available(modules: &str, what: &str) -> Result<(), String> {
    let mut tried = Vec::new();
    for python in python_candidates() {
        match Command::new(&python)
            .args(["-c", &format!("import {modules}")])
            .output()
        {
            Ok(out) if out.status.success() => return Ok(()),
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                tried.push(format!(
                    "`{python}`: {}",
                    stderr.lines().last().unwrap_or("import failed").trim()
                ));
            }
            Err(e) => tried.push(format!("`{python}`: {e}")),
        }
    }
    Err(format!(
        "no Python interpreter with the {what} (set ES_PYTHON to choose one): {}",
        tried.join("; ")
    ))
}

/// Whether a Python with the `mujoco` package can be found.
pub fn is_available() -> Result<(), String> {
    import_available("mujoco", "`mujoco` package")
}

/// How much of a process's stderr a death report can quote: the last this many bytes, and of
/// those the last [`TAIL_LINES`] lines (packet M12/R6).
const TAIL_BYTES: usize = 16 * 1024;
const TAIL_LINES: usize = 64;

/// How long a process whose call failed is given to finish exiting before its last words are
/// read.
const LAST_WORDS_WAIT: Duration = Duration::from_secs(1);

/// The tail of a process's stderr, kept by a reader thread that drains the pipe as it fills,
/// so an unread pipe never blocks the process. Read only when a call fails.
#[derive(Debug)]
struct StderrTail {
    ring: Arc<Mutex<VecDeque<u8>>>,
    reader: JoinHandle<()>,
}

impl StderrTail {
    fn drain(mut stderr: ChildStderr) -> Self {
        let ring = Arc::new(Mutex::new(VecDeque::with_capacity(TAIL_BYTES)));
        let sink = Arc::clone(&ring);
        let reader = std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match stderr.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let mut ring = sink.lock().unwrap_or_else(PoisonError::into_inner);
                        ring.extend(&buf[..n]);
                        let excess = ring.len().saturating_sub(TAIL_BYTES);
                        ring.drain(..excess);
                    }
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(_) => break,
                }
            }
        });
        Self { ring, reader }
    }

    fn text(&self) -> String {
        let mut ring = self.ring.lock().unwrap_or_else(PoisonError::into_inner);
        let text = String::from_utf8_lossy(ring.make_contiguous()).into_owned();
        let lines: Vec<&str> = text.trim().lines().collect();
        lines[lines.len().saturating_sub(TAIL_LINES)..].join("\n")
    }
}

/// A running reference process.
#[derive(Debug)]
pub struct Process {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// `None` when the caller did not pipe stderr.
    stderr: Option<StderrTail>,
    /// Whether a failed call found a line the process had left on stdout (packet M15/R1): a
    /// process that answered, even with an error, did not fail silently.
    wrote: bool,
}

impl Process {
    /// Starts `script` as [`Self::spawn_with`] does and makes the first call, `first` (the
    /// load); the reply comes back with the process. A start that fails the way the Windows
    /// loader fails concurrent starts is retried by [`retry_start`] (packet M15/R1): a fresh
    /// process, the same script and the same request, so a retried start changes no result.
    /// A process that dies after this, mid-episode, is never restarted.
    pub fn start<T: DeserializeOwned>(
        script: &str,
        engine: &str,
        first: &Request<'_>,
    ) -> Result<(Self, T), PhysicsError> {
        retry_start(&format!("the {engine} reference process"), || {
            let mut process = Self::spawn_with(script, engine).map_err(|e| (e, false))?;
            match process.call(first) {
                Ok(reply) => Ok((process, reply)),
                Err(e) => {
                    let died = matches!(e, PhysicsError::ProcessDied(_));
                    Err((e, died && process.failed_to_start()))
                }
            }
        })
        .map_err(|(e, attempts)| match e {
            PhysicsError::ProcessDied(text) if attempts > 1 => {
                PhysicsError::ProcessDied(format!("{text} (after {attempts} start attempts)"))
            }
            e => e,
        })
    }

    /// Whether this process, whose call just died, failed the way [`transient_start_failure`]
    /// retries. Its exit status and stderr are settled: the failed call waited for them.
    fn failed_to_start(&mut self) -> bool {
        let code = self.child.try_wait().ok().flatten().and_then(|s| s.code());
        let stderr = self
            .stderr
            .as_ref()
            .map(StderrTail::text)
            .unwrap_or_default();
        transient_start_failure(code, &stderr, self.wrote)
    }

    /// Starts the first interpreter that spawns, running [`SCRIPT`].
    pub fn spawn() -> Result<Self, PhysicsError> {
        Self::spawn_with(SCRIPT, "MuJoCo")
    }

    /// Starts the first interpreter that spawns, running `script`.
    ///
    /// `engine` names the backend in the failure message. Every out-of-process backend speaks
    /// this one protocol, so they share the spawn / call / drop trio and differ only in which
    /// script is on the other end.
    pub fn spawn_with(script: &str, engine: &str) -> Result<Self, PhysicsError> {
        let mut tried = Vec::new();
        for python in python_candidates() {
            let mut cmd = Command::new(&python);
            cmd.args(["-c", script])
                // Protocol errors come back on stdout as JSON; stderr is for a process that
                // dies before it can write one (a traceback, the DLL loader), and it is drained
                // as it fills, so it cannot deadlock us.
                .stderr(Stdio::piped());
            match Self::spawn_command(cmd) {
                Ok(process) => return Ok(process),
                Err(e) => tried.push(format!("`{python}`: {e}")),
            }
        }
        Err(PhysicsError::Backend(format!(
            "cannot start the {engine} reference process: {}",
            tried.join("; ")
        )))
    }

    /// Starts `cmd` (interpreter, arguments, environment and stderr already set) with the
    /// protocol on its stdin / stdout -- for an engine that cannot run from `python -c`
    /// (packet M11/I1: Isaac Sim's Kit crashes when `sys.argv` is `["-c"]`). A piped stderr is
    /// drained into the tail a death report quotes.
    pub fn spawn_command(mut cmd: Command) -> std::io::Result<Self> {
        let mut child = cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().map(StderrTail::drain);
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::with_capacity(1 << 20, stdout),
            stderr,
            wrote: false,
        })
    }

    /// Sends one request and decodes its reply.
    pub fn call<T: DeserializeOwned>(&mut self, request: &Request<'_>) -> Result<T, PhysicsError> {
        self.call_with(request, &[])
    }

    /// [`Self::call`], with `trailer` written right after the request line.
    fn call_with<T: DeserializeOwned>(
        &mut self,
        request: &Request<'_>,
        trailer: &[u8],
    ) -> Result<T, PhysicsError> {
        let line = serde_json::to_string(request)
            .map_err(|e| PhysicsError::Protocol(format!("cannot encode request: {e}")))?;
        let sent = self
            .stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.write_all(trailer))
            .and_then(|()| self.stdin.flush());
        if let Err(e) = sent {
            return Err(self.last_words(e.to_string(), true));
        }

        let mut reply = String::new();
        match self.stdout.read_line(&mut reply) {
            Ok(0) => Err(self.last_words(
                "the process closed its output without answering".to_owned(),
                false,
            )),
            Ok(_) => parse_response(&reply),
            Err(e) => Err(self.last_words(e.to_string(), false)),
        }
    }

    /// `state` to a script that answers it as a [`FrameHeader`] and its bytes (packet M16/H0).
    pub fn call_state(&mut self) -> Result<StateReply, PhysicsError> {
        let header: FrameHeader = self.call(&Request::State)?;
        read_frame(&mut self.stdout, header).map_err(|e| self.last_words(e.to_string(), false))
    }

    /// `set_ctrl` to a script that reads the values as raw little-endian `f64` bytes after the
    /// line (packet M16/H0): what a JSON list of them is, bit for bit, without Python parsing
    /// `n_envs * nu` decimal numbers every control step.
    pub fn set_ctrl_frame(&mut self, ctrl: &[f64]) -> Result<(), PhysicsError> {
        let bytes: Vec<u8> = ctrl.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _: Ack = self.call_with(&Request::SetCtrlFrame { frame: ctrl.len() }, &bytes)?;
        Ok(())
    }

    /// The [`PhysicsError::ProcessDied`] for a call that failed with `cause` (packet M12/R6):
    /// a line the process left on stdout (a protocol error line's `error` replaces `cause`),
    /// how it exited, and the tail of its stderr. Stdout is read only after a failed write and
    /// only once the process has exited, so this never blocks past [`LAST_WORDS_WAIT`].
    fn last_words(&mut self, cause: String, read_stdout: bool) -> PhysicsError {
        let status = self.wait_briefly();
        let mut what = cause;
        let mut line = String::new();
        if read_stdout && status.is_some() && self.stdout.read_line(&mut line).is_ok() {
            self.wrote |= !line.trim().is_empty();
            match parse_response::<Ack>(&line) {
                Err(PhysicsError::Backend(error)) => what = error,
                _ if !line.trim().is_empty() => {
                    what = format!("{what}; it wrote `{}`", truncate(&line));
                }
                _ => {}
            }
        }
        let status = match status {
            Some(s) => s
                .code()
                .map_or_else(|| s.to_string(), |c| format!("exit code {c}")),
            None => "still running".to_owned(),
        };
        let tail = self
            .stderr
            .as_ref()
            .map(StderrTail::text)
            .unwrap_or_default();
        PhysicsError::ProcessDied(if tail.is_empty() {
            format!("{what} ({status})")
        } else {
            format!("{what} ({status}) — stderr: {tail}")
        })
    }

    /// The exit status once the process has exited and its stderr is read to the end, or
    /// whatever is known at [`LAST_WORDS_WAIT`].
    fn wait_briefly(&mut self) -> Option<ExitStatus> {
        let deadline = Instant::now() + LAST_WORDS_WAIT;
        loop {
            let status = self.child.try_wait().ok().flatten();
            let drained = self.stderr.as_ref().is_none_or(|s| s.reader.is_finished());
            if (status.is_some() && drained) || Instant::now() >= deadline {
                return status;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Ask it to quit, then make sure: a leaked python process would outlive the test run.
        let _ = self
            .stdin
            .write_all(b"{\"cmd\":\"quit\"}\n")
            .and_then(|()| self.stdin.flush());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The tick rate a timestep in seconds stands for (spec 18.1: integer ticks are the model).
pub(crate) fn rate_from_timestep(timestep: f64) -> Result<TickRate, PhysicsError> {
    let nanos = (timestep * 1e9).round();
    if !(nanos.is_finite() && nanos >= 1.0) {
        return Err(PhysicsError::Backend(format!(
            "timestep {timestep} is not a positive number of nanoseconds"
        )));
    }
    let divisor = gcd(1_000_000_000, nanos as u64);
    TickRate::rational(1_000_000_000 / divisor, nanos as u64 / divisor)
        .map_err(|e| PhysicsError::Backend(e.to_string()))
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn id_of(
    map: &BTreeMap<&str, StableId>,
    engine: &str,
    kind: &str,
    name: &str,
) -> Result<StableId, PhysicsError> {
    map.get(name).copied().ok_or_else(|| {
        PhysicsError::Protocol(format!(
            "{engine} reported a {kind} `{name}` the scene does not have"
        ))
    })
}

/// Turns a `load` reply into a [`ModelInfo`] by matching the engine's names back to the scene's
/// [`StableId`]s. Every out-of-process backend answers in this one shape, so they share it.
///
/// A name the scene does not have is a [`PhysicsError::Protocol`], never a silent mismatch: the
/// index ranges are what the runtime addresses state by, so a wrong one is a wrong robot.
pub(crate) fn model_info(
    reply: &LoadReply,
    scene: &SceneDesc,
    engine: &str,
    n_envs: u32,
    rate: TickRate,
) -> Result<ModelInfo, PhysicsError> {
    fn by_name<'a, T: 'a>(
        items: impl IntoIterator<Item = &'a T>,
        field: impl Fn(&'a T) -> (&'a str, StableId),
    ) -> BTreeMap<&'a str, StableId> {
        items.into_iter().map(field).collect()
    }
    let joints = by_name(&scene.joints, |j| (j.name.as_str(), j.id));
    let actuators = by_name(&scene.actuators, |a| (a.name.as_str(), a.id));
    let sensors = by_name(&scene.sensors, |s| (s.name.as_str(), s.id));
    let bodies = by_name(&scene.bodies, |b| (b.name.as_str(), b.id));

    let mut info = ModelInfo {
        nq: reply.nq,
        nv: reply.nv,
        nu: reply.nu,
        nsensordata: reply.nsensordata,
        nbody: reply.nbody,
        n_envs,
        rate,
        ..ModelInfo::default()
    };
    for joint in &reply.joints {
        let id = id_of(&joints, engine, "joint", &joint.name)?;
        info.qpos
            .insert(id, IndexRange::new(joint.qpos[0], joint.qpos[1]));
        info.dof
            .insert(id, IndexRange::new(joint.dof[0], joint.dof[1]));
    }
    for (index, name) in reply.actuators.iter().enumerate() {
        let id = id_of(&actuators, engine, "actuator", name)?;
        info.actuator.insert(id, IndexRange::new(index as u32, 1));
    }
    for sensor in &reply.sensors {
        let id = id_of(&sensors, engine, "sensor", &sensor.name)?;
        info.sensor
            .insert(id, IndexRange::new(sensor.adr, sensor.dim));
    }
    for (index, name) in reply.bodies.iter().enumerate() {
        let id = id_of(&bodies, engine, "body", name)?;
        info.body.insert(id, IndexRange::new(index as u32, 1));
    }
    Ok(info)
}

/// Packet M16/H0 oracle: `script` (`mujoco_ref.py` or `mjwarp_ref.py`) loads the SO-101 scene
/// with three envs and steps it twice (on `mjwarp` the second call replays the captured graph);
/// its `state` is written once by the script's own `answer` (the frame) and once as the JSON
/// float lists the scripts sent before, and the two decodings are asserted equal bit for bit.
/// Run on the interpreter the backends use, after the caller checked it has the engine.
#[cfg(test)]
pub(crate) fn assert_state_matches_the_json_encoding(script: &str) {
    const DRIVER: &str = r#"
import json, sys
g = {"__name__": "es_h0"}
exec(open(sys.argv[1]).read(), g)
out = g.get("_OUT", sys.stdout)
if "wp" in g:
    g["wp"].init()
sim = g["Sim"](open(sys.argv[2]).read(), 3, None, 0)
sim.reset(range(3), None)
model = sim.mjm if hasattr(sim, "mjm") else sim.model
sim.set_ctrl([0.3 * ((k % 5) - 2) for k in range(int(model.nu) * 3)])
sim.step(25)
sim.step(25)
arrays = sim.state()["frame"]
g["answer"](out, {"ok": True, "frame": arrays})
names = ("qpos", "qvel", "act", "sensordata", "xpos", "xquat")
old = dict(zip(names, (a.tolist() for a in arrays)), ok=True)
out.write(json.dumps(old) + "\n")
out.flush()
"#;
    let dir = std::env::temp_dir().join(format!(
        "es-h0-{}-{}",
        std::process::id(),
        blake3::hash(script.as_bytes()).to_hex()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ref.py");
    std::fs::write(&path, script).unwrap();
    let mjcf = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/mjcf/so101_pick_place.xml"
    );
    let python = python_candidates().remove(0);
    // Retried as `Process::start` retries: concurrent starts on Windows can die in the loader.
    let out = retry_start("the M16/H0 driver", || {
        let out = Command::new(&python)
            .args(["-c", DRIVER])
            .arg(&path)
            .arg(mjcf)
            .output()
            .map_err(|e| (e.to_string(), false))?;
        if out.status.success() {
            return Ok(out);
        }
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        let transient = transient_start_failure(out.status.code(), &stderr, !out.stdout.is_empty());
        Err((format!("{}: {stderr}", out.status), transient))
    })
    .unwrap_or_else(|(e, attempts)| panic!("{e} (after {attempts} attempts)"));
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let mut reader = out.stdout.as_slice();
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let header: FrameHeader =
        parse_response(&line).unwrap_or_else(|e| panic!("{e} ({})\n{stderr}", out.status));
    let new = read_frame(&mut reader, header).unwrap();
    line.clear();
    reader.read_line(&mut line).unwrap();
    let old: StateReply = parse_response(&line).unwrap();
    assert!(reader.is_empty(), "{} bytes left over", reader.len());
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (what, a, b) in [
        ("qpos", &new.qpos, &old.qpos),
        ("qvel", &new.qvel, &old.qvel),
        ("act", &new.act, &old.act),
        ("sensordata", &new.sensordata, &old.sensordata),
        ("xpos", &new.xpos, &old.xpos),
        ("xquat", &new.xquat, &old.xquat),
    ] {
        assert_eq!(bits(a), bits(b), "{what}");
    }
    assert!(new.qpos.len() >= 3 * 13 && new.xquat.len() % 4 == 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parser is exercised with canned lines, so the protocol is tested without Python.
    #[test]
    fn a_successful_reply_decodes() {
        let reply: LoadReply = parse_response(
            r#"{"ok":true,"nq":7,"nv":6,"nu":1,"nsensordata":2,"nbody":3,
                "joints":[{"name":"j","qpos":[0,7],"dof":[0,6]}],
                "actuators":["m"],"sensors":[{"name":"s","adr":0,"dim":2}],
                "bodies":["world","b"],"engine_version":"mujoco 3.3.2"}"#,
        )
        .unwrap();
        assert_eq!((reply.nq, reply.nv, reply.nbody), (7, 6, 3));
        assert_eq!(reply.checked_engine_version().unwrap(), "mujoco 3.3.2");
        assert_eq!(reply.joints[0].qpos, [0, 7]);
        assert_eq!(reply.sensors[0].dim, 2);

        let state: StateReply =
            parse_response(r#"{"ok":true,"qpos":[1.5,-0.0],"qvel":[],"act":[],"sensordata":[],"xpos":[],"xquat":[]}"#)
                .unwrap();
        assert_eq!(state.qpos, vec![1.5, -0.0]);
        let step: StepReply = parse_response(r#"{"ok":true,"nonfinite":[3]}"#).unwrap();
        assert_eq!(step.nonfinite, vec![3]);
        let _: Ack = parse_response(r#"{"ok":true}"#).unwrap();
    }

    /// Packet M11/X1: the engine version is what `backend_identity` hashes, so a reply without
    /// one is a protocol error and an empty one is refused -- never an empty string in a hash.
    #[test]
    fn backend_identity_needs_an_engine_version() {
        let body = r#""ok":true,"nq":0,"nv":0,"nu":0,"nsensordata":0,"nbody":1,
            "joints":[],"actuators":[],"sensors":[],"bodies":["world"]"#;
        let missing = parse_response::<LoadReply>(&format!("{{{body}}}")).unwrap_err();
        assert!(matches!(missing, PhysicsError::Protocol(_)), "{missing:?}");
        let empty: LoadReply =
            parse_response(&format!(r#"{{{body},"engine_version":" "}}"#)).unwrap();
        assert!(matches!(
            empty.checked_engine_version(),
            Err(PhysicsError::Protocol(_))
        ));
    }

    #[test]
    fn a_failed_reply_becomes_a_backend_error() {
        let err = parse_response::<Ack>(r#"{"ok":false,"error":"ValueError: no model loaded"}"#)
            .unwrap_err();
        assert_eq!(
            err,
            PhysicsError::Backend("ValueError: no model loaded".to_owned())
        );
    }

    #[test]
    fn malformed_replies_are_protocol_errors() {
        for line in [
            "not json at all",
            r#"{"nq":1}"#,
            r#"{"ok":true,"nq":"seven"}"#,
        ] {
            let err = parse_response::<LoadReply>(line).unwrap_err();
            assert!(
                matches!(err, PhysicsError::Protocol(_)),
                "{line} gave {err:?}"
            );
        }
    }

    /// f64 values must survive the JSON round trip bit for bit; the run-to-run bitwise check in
    /// `mujoco.rs` depends on it (spec 3.5).
    #[test]
    fn floats_round_trip_bit_for_bit() {
        let values = [
            0.1,
            -9.81,
            f64::MIN_POSITIVE,
            1.234_567_890_123_456_7e-13,
            f64::MAX,
        ];
        let line = format!(
            r#"{{"ok":true,"qpos":{},"qvel":[],"act":[],"sensordata":[],"xpos":[],"xquat":[]}}"#,
            serde_json::to_string(&values).unwrap()
        );
        let state: StateReply = parse_response(&line).unwrap();
        for (got, want) in state.qpos.iter().zip(values) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }

    /// Packet M16/H0: a frame decodes to exactly the bits it was written from, split in the
    /// header's order, including what a JSON number cannot carry (`NaN`, infinities); a short
    /// one is an error, never a truncated state.
    #[test]
    fn a_frame_decodes_bit_for_bit() {
        let values = [
            0.1,
            -0.0,
            -9.81,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
        ];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header: FrameHeader =
            parse_response(r#"{"ok": true, "frame": [3, 0, 0, 1, 4, 0]}"#).unwrap();
        let state = read_frame(&mut bytes.as_slice(), header).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&state.qpos), bits(&values[..3]));
        assert_eq!(bits(&state.sensordata), bits(&values[3..4]));
        assert_eq!(bits(&state.xpos), bits(&values[4..]));
        assert!(state.qvel.is_empty() && state.act.is_empty() && state.xquat.is_empty());
        let short = FrameHeader {
            frame: [9, 0, 0, 0, 0, 0],
        };
        assert!(read_frame(&mut bytes.as_slice(), short).is_err());
    }

    /// Packet M16/H0: `mujoco_ref.py`'s state frame is its old JSON state, bit for bit.
    #[test]
    fn mujoco_state_frame_is_the_json_state() {
        if let Err(why) = is_available() {
            println!("SKIP mujoco_state_frame_is_the_json_state: {why}");
            return;
        }
        assert_state_matches_the_json_encoding(SCRIPT);
    }

    /// Packet M16/H0: through the real protocol, a `set_ctrl` sent as a frame steps
    /// `mujoco_ref.py` to the state a JSON list of the same controls does, bit for bit.
    #[test]
    fn mujoco_ctrl_frame_is_the_json_ctrl() {
        if let Err(why) = is_available() {
            println!("SKIP mujoco_ctrl_frame_is_the_json_ctrl: {why}");
            return;
        }
        let mjcf = crate::tests_support::fixture("so101_pick_place.xml");
        let load = Request::Load {
            mjcf: &mjcf,
            n_envs: 2,
            timestep: None,
            seed: 0,
        };
        let run = |frame: bool| {
            let (mut p, info): (Process, LoadReply) =
                Process::start(SCRIPT, "MuJoCo", &load).unwrap();
            let ctrl: Vec<f64> = (0..2 * info.nu)
                .map(|k| 0.1 * f64::from(k) / 3.0 - 0.2)
                .collect();
            if frame {
                p.set_ctrl_frame(&ctrl).unwrap();
            } else {
                let _: Ack = p.call(&Request::SetCtrl { ctrl: &ctrl }).unwrap();
            }
            let _: StepReply = p.call(&Request::Step { n: 40 }).unwrap();
            p.call_state().unwrap()
        };
        let (a, b) = (run(true), run(false));
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.qpos), bits(&b.qpos));
        assert_eq!(bits(&a.qvel), bits(&b.qvel));
        assert_eq!(bits(&a.xpos), bits(&b.xpos));
    }

    /// Packet M16/H0: `mjwarp_ref.py`'s state frame is its old JSON state, bit for bit, after a
    /// replayed step graph.
    #[test]
    fn mjwarp_state_frame_is_the_json_state() {
        if let Err(why) = crate::MjWarpBackend::is_available() {
            println!("SKIP mjwarp_state_frame_is_the_json_state: {why}");
            return;
        }
        assert_state_matches_the_json_encoding(crate::mjwarp::SCRIPT);
    }

    #[test]
    fn requests_encode_as_the_script_expects() {
        let encode = |r: &Request<'_>| serde_json::to_string(r).unwrap();
        assert_eq!(
            encode(&Request::Load {
                mjcf: "<mujoco/>",
                n_envs: 2,
                timestep: Some(0.001),
                seed: 7,
            }),
            r#"{"cmd":"load","mjcf":"<mujoco/>","n_envs":2,"timestep":0.001,"seed":7}"#
        );
        assert_eq!(encode(&Request::Step { n: 3 }), r#"{"cmd":"step","n":3}"#);
        assert_eq!(encode(&Request::State), r#"{"cmd":"state"}"#);
        assert_eq!(
            encode(&Request::SetCtrl { ctrl: &[0.5] }),
            r#"{"cmd":"set_ctrl","ctrl":[0.5]}"#
        );
        assert_eq!(
            encode(&Request::SetCtrlFrame { frame: 12 }),
            r#"{"cmd":"set_ctrl","frame":12}"#
        );
        assert_eq!(
            encode(&Request::Reset {
                envs: Some(&[1]),
                state: None
            }),
            r#"{"cmd":"reset","envs":[1],"state":null}"#
        );
    }

    #[test]
    fn the_embedded_script_is_the_file_on_disk() {
        assert!(SCRIPT.contains("mujoco.mj_step"));
        assert!(SCRIPT.contains("\"cmd\""));
    }

    /// What `mujoco_ref.py` does when its imports fail: one protocol error line, then exit 1.
    const DIES_ON_IMPORT: &str = r#"
import sys
sys.stderr.write("Traceback: the stand-in cannot import mujoco\n")
print('{"ok": false, "error": "import failed: no module named mujoco"}', flush=True)
raise SystemExit(1)
"#;

    /// Writes 2 MiB to stderr before every answer; a `step` makes it exit 3.
    const FLOODS_STDERR: &str = r#"
import sys
while True:
    line = sys.stdin.readline()
    if not line or '"quit"' in line:
        break
    sys.stderr.write(("x" * 1023 + "\n") * 2048)
    sys.stderr.flush()
    if '"step"' in line:
        raise SystemExit(3)
    print('{"ok": true}', flush=True)
"#;

    /// `script` as a reference process, on the interpreter the backends use; `None` (printing
    /// why) where there is none.
    fn stand_in(script: &str) -> Option<Process> {
        if let Err(why) = import_available("sys", "Python interpreter") {
            println!("SKIP: {why}");
            return None;
        }
        Some(Process::spawn_with(script, "stand-in").expect("the stand-in starts"))
    }

    /// Packet M12/R6: a process that reports an import failure and exits before the first
    /// request is reported by what it said and how it exited, not by the write that found its
    /// pipe closed (os error 232 on Windows).
    #[test]
    fn a_process_that_dies_before_the_first_request_says_why() {
        let Some(mut process) = stand_in(DIES_ON_IMPORT) else {
            return;
        };
        while process.child.try_wait().unwrap().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let err = process.call::<Ack>(&Request::State).unwrap_err();
        let PhysicsError::ProcessDied(text) = &err else {
            panic!("{err:?}");
        };
        assert!(
            text.starts_with("import failed: no module named mujoco (exit code 1)"),
            "{text}"
        );
        assert!(
            text.contains("stderr: Traceback: the stand-in cannot import mujoco"),
            "{text}"
        );
        assert!(!text.contains("os error"), "{text}");
    }

    /// `body` behind a prologue that counts its starts in `count` (a fresh file) and, while
    /// the count is under `fail`, dies the way the Windows loader kills a start: a
    /// `[WinError 6]` traceback on stderr, nothing on stdout.
    fn counted(count: &std::path::Path, fail: u32, body: &str) -> String {
        let _ = std::fs::remove_file(count);
        format!(
            "import os, sys\npath = {count:?}\n\
             n = int(open(path).read()) if os.path.exists(path) else 0\n\
             open(path, 'w').write(str(n + 1))\n\
             if n < {fail}:\n    sys.stderr.write('OSError: [WinError 6] The handle is invalid. \
             Error loading \"c10.dll\" or one of its dependencies.\\n')\n    \
             raise SystemExit(1)\n{body}"
        )
    }

    fn starts(count: &std::path::Path) -> String {
        std::fs::read_to_string(count).unwrap()
    }

    /// Packet M15/R1: a start that dies the loader's way twice is retried and the third
    /// answers.
    #[test]
    fn a_start_the_loader_kills_is_retried() {
        if let Err(why) = import_available("sys", "Python interpreter") {
            println!("SKIP: {why}");
            return;
        }
        let count = std::env::temp_dir().join(format!("es-r1-flaky-{}", std::process::id()));
        let answers = "for line in sys.stdin:\n    print('{\"ok\": true}', flush=True)\n";
        let script = counted(&count, 2, answers);
        let (_process, _): (_, Ack) =
            Process::start(&script, "stand-in", &Request::State).expect("the third start answers");
        assert_eq!(starts(&count), "3");
        let _ = std::fs::remove_file(&count);
    }

    /// Packet M15/R1: a clean import failure is an answer, not a flaky start: one start, and
    /// the message is today's. Its only words are the protocol line on stdout, read after a
    /// load too big for the pipe found the process gone (os error 232 on Windows).
    #[test]
    fn an_import_failure_is_not_retried() {
        if let Err(why) = import_available("sys", "Python interpreter") {
            println!("SKIP: {why}");
            return;
        }
        let count = std::env::temp_dir().join(format!("es-r1-import-{}", std::process::id()));
        let body =
            "print('{\"ok\": false, \"error\": \"import failed: no module named mujoco\"}', \
                    flush=True)\nraise SystemExit(1)\n";
        let script = counted(&count, 0, body);
        let mjcf = "x".repeat(1 << 20);
        let load = Request::Load {
            mjcf: &mjcf,
            n_envs: 1,
            timestep: None,
            seed: 0,
        };
        let err = Process::start::<Ack>(&script, "stand-in", &load).unwrap_err();
        assert!(err.to_string().contains("no module named mujoco"), "{err}");
        assert!(!err.to_string().contains("attempts"), "{err}");
        assert_eq!(starts(&count), "1");
        let _ = std::fs::remove_file(&count);
    }

    /// Packet M12/R6: stderr is drained as it is written, so a process that floods it still
    /// answers, and what a death report quotes of it stays bounded.
    #[test]
    fn a_process_that_floods_stderr_still_answers() {
        let Some(mut process) = stand_in(FLOODS_STDERR) else {
            return;
        };
        for _ in 0..2 {
            let _: Ack = process.call(&Request::State).unwrap();
        }
        let err = process.call::<Ack>(&Request::Step { n: 1 }).unwrap_err();
        let PhysicsError::ProcessDied(text) = &err else {
            panic!("{err:?}");
        };
        assert!(text.contains("(exit code 3) — stderr: xxx"), "{text}");
        assert!(
            text.lines().count() <= 64 && text.len() <= 17 * 1024,
            "{text}"
        );
    }
}
