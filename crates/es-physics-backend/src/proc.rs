//! The line-delimited JSON protocol to `python/mujoco_ref.py`, and the process that speaks it.
//!
//! One request per line in, one JSON object per line out. Errors on the Python side come back
//! as `{"ok": false, "error": ...}`, so a modelling mistake is a typed [`PhysicsError`] and
//! never a dead process.
//!
//! The script is embedded with `include_str!` and handed to `python -c`, so there is no
//! installed-data-file lookup at runtime and editing the script forces a rebuild.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use es_assets::scene::SceneDesc;
use es_core::{StableId, TickRate};
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
    Step {
        n: u32,
    },
    State,
    SetState {
        state: StatePayload<'a>,
    },
    Quit,
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

/// A running reference process.
#[derive(Debug)]
pub struct Process {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Process {
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
            let spawned = Command::new(&python)
                .args(["-c", script])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                // Every Python-side failure is reported on stdout as JSON, so stderr carries
                // nothing we need and an unread pipe could only deadlock us.
                .stderr(Stdio::null())
                .spawn();
            match spawned {
                Ok(mut child) => {
                    let stdin = child.stdin.take().expect("stdin was piped");
                    let stdout = child.stdout.take().expect("stdout was piped");
                    return Ok(Self {
                        child,
                        stdin,
                        stdout: BufReader::new(stdout),
                    });
                }
                Err(e) => tried.push(format!("`{python}`: {e}")),
            }
        }
        Err(PhysicsError::Backend(format!(
            "cannot start the {engine} reference process: {}",
            tried.join("; ")
        )))
    }

    /// Sends one request and decodes its reply.
    pub fn call<T: DeserializeOwned>(&mut self, request: &Request<'_>) -> Result<T, PhysicsError> {
        let line = serde_json::to_string(request)
            .map_err(|e| PhysicsError::Protocol(format!("cannot encode request: {e}")))?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .map_err(|e| PhysicsError::ProcessDied(e.to_string()))?;

        let mut reply = String::new();
        match self.stdout.read_line(&mut reply) {
            Ok(0) => Err(PhysicsError::ProcessDied(
                "the process closed its output without answering".to_owned(),
            )),
            Ok(_) => parse_response(&reply),
            Err(e) => Err(PhysicsError::ProcessDied(e.to_string())),
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
                "bodies":["world","b"]}"#,
        )
        .unwrap();
        assert_eq!((reply.nq, reply.nv, reply.nbody), (7, 6, 3));
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
}
