//! `TorchRuntime` — `PyTorch` behind a Python subprocess (spec 2.4).
//!
//! `PyTorch` is the Learning IR's reference oracle (spec 1.4), and spec 2.4 forbids the core
//! runtime from linking Python. So it runs out of process, exactly as `MuJoCoCpuBackend` does
//! for physics: `python/torch_ref.py` is embedded with `include_str!`, handed to `python -c`,
//! and spoken to in line-delimited JSON. That is slow, and deliberately so — this is the thing
//! every other `PolicyRuntime` is judged against, not a throughput path. `libtorch` FFI is the
//! same trait with a different body.
//!
//! Tensors cross as base64'd little-endian f32. JSON numbers would have been simpler but a
//! `[50, 8]` chunk of them is 20x the bytes and invites a shortest-round-trip argument at the
//! exact place where spec 8.9 wants a 1e-5 comparison.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use es_compile::Tensor;
use es_ir::learning::{LearningGraph, PolicyHandle};
use es_ir::types::ElemType;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::lower::{lower_to_torch, TorchModule};
use crate::runtime::{
    runtime_hash_of, InferenceBackend, PolicyError, PolicyInfo, PolicyRuntime, WeightsSource,
};
use crate::weights::{hex, parse_header, validate_keys, weights_hash};

/// The reference script, embedded at build time so there is no data-file lookup at runtime and
/// editing it forces a rebuild.
pub const SCRIPT: &str = include_str!("../python/torch_ref.py");

/// Bumped whenever the wire format changes. Part of [`PolicyRuntime::runtime_hash`].
pub const PROTOCOL_VERSION: u32 = 1;

/// One request to the reference process.
#[derive(Debug, Serialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Request<'a> {
    Load {
        source: &'a str,
        weights_path: &'a str,
    },
    Infer {
        inputs: BTreeMap<String, WireTensor>,
    },
}

/// A tensor on the wire: shape, dtype name (safetensors spelling) and base64 raw bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct WireTensor {
    shape: Vec<u64>,
    dtype: String,
    data_b64: String,
}

#[derive(Debug, Deserialize)]
struct LoadReply {
    torch_version: String,
}

#[derive(Debug, Deserialize)]
struct InferReply {
    outputs: BTreeMap<String, WireTensor>,
}

impl WireTensor {
    fn encode(t: &Tensor) -> Result<Self, PolicyError> {
        if t.dtype != ElemType::F32 {
            return Err(PolicyError::Protocol(format!(
                "only F32 crosses this protocol, got {:?}",
                t.dtype
            )));
        }
        Ok(Self {
            shape: t.shape.clone(),
            dtype: "F32".to_owned(),
            data_b64: b64_encode(&t.data),
        })
    }

    fn decode(self, name: &str) -> Result<Tensor, PolicyError> {
        if self.dtype != "F32" {
            return Err(PolicyError::Protocol(format!(
                "output \"{name}\" has dtype {}",
                self.dtype
            )));
        }
        let data = b64_decode(&self.data_b64)
            .ok_or_else(|| PolicyError::Protocol(format!("output \"{name}\" is not base64")))?;
        let want = self.shape.iter().product::<u64>() as usize * 4;
        if data.len() != want {
            return Err(PolicyError::Protocol(format!(
                "output \"{name}\": shape {:?} needs {want} bytes, got {}",
                self.shape,
                data.len()
            )));
        }
        Ok(Tensor {
            dtype: ElemType::F32,
            shape: self.shape,
            data,
        })
    }
}

/// Decode one response line, as `proc.rs` does for the physics oracle: `{"ok": true, ...}` into
/// `T`, `{"ok": false, ...}` into a backend error, anything else into a protocol error.
fn parse_response<T: DeserializeOwned>(line: &str) -> Result<T, PolicyError> {
    let value: serde_json::Value = serde_json::from_str(line)
        .map_err(|e| PolicyError::Protocol(format!("{e} in `{}`", truncate(line))))?;
    match value.get("ok") {
        Some(serde_json::Value::Bool(true)) => serde_json::from_value(value)
            .map_err(|e| PolicyError::Protocol(format!("{e} in `{}`", truncate(line)))),
        Some(serde_json::Value::Bool(false)) => Err(PolicyError::Backend(
            value
                .get("error")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unspecified")
                .to_owned(),
        )),
        _ => Err(PolicyError::Protocol(format!(
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

/// Interpreters to try, in order. `ES_PYTHON` overrides the search entirely — the same knob
/// the `MuJoCo` oracle uses, so one environment variable points both at the same venv.
pub(crate) fn python_candidates() -> Vec<String> {
    match std::env::var("ES_PYTHON") {
        Ok(path) if !path.trim().is_empty() => vec![path],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    }
}

/// Whether a Python with `torch` can be found. `Err` carries what was tried, so a CI skip
/// message says something useful.
pub fn is_available() -> Result<(), String> {
    let mut tried = Vec::new();
    for python in python_candidates() {
        match Command::new(&python).args(["-c", "import torch"]).output() {
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
        "no Python interpreter with `torch` (set ES_PYTHON to choose one): {}",
        tried.join("; ")
    ))
}

#[derive(Debug)]
struct Process {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Process {
    fn spawn() -> Result<Self, PolicyError> {
        let mut tried = Vec::new();
        for python in python_candidates() {
            match Command::new(&python)
                .args(["-c", SCRIPT])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                // Every Python-side failure comes back on stdout as JSON, so stderr carries
                // nothing we need and an unread pipe could only deadlock us. `torch` writes a
                // NumPy warning there on import.
                .stderr(Stdio::null())
                .spawn()
            {
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
        Err(PolicyError::Unavailable(format!(
            "cannot start the torch reference process: {}",
            tried.join("; ")
        )))
    }

    fn call<T: DeserializeOwned>(&mut self, request: &Request<'_>) -> Result<T, PolicyError> {
        let line = serde_json::to_string(request)
            .map_err(|e| PolicyError::Protocol(format!("cannot encode request: {e}")))?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .map_err(|e| PolicyError::ProcessDied(e.to_string()))?;

        let mut reply = String::new();
        match self.stdout.read_line(&mut reply) {
            Ok(0) => Err(PolicyError::ProcessDied(
                "the process closed its output without answering".to_owned(),
            )),
            Ok(_) => parse_response(&reply),
            Err(e) => Err(PolicyError::ProcessDied(e.to_string())),
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Ask it to quit, then make sure: a leaked interpreter would outlive the test run.
        let _ = self
            .stdin
            .write_all(b"{\"cmd\":\"quit\"}\n")
            .and_then(|()| self.stdin.flush());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `PyTorch` inference, out of process.
#[derive(Debug, Default)]
pub struct TorchRuntime {
    process: Option<Process>,
    info: Option<PolicyInfo>,
    /// A file this runtime wrote for [`WeightsSource::InMemory`], to delete on drop.
    scratch: Option<PathBuf>,
}

impl TorchRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    /// Materialize the checkpoint: the Python side reads a path, so in-memory bytes are spilled
    /// to a temporary file named by their own hash.
    fn checkpoint(&mut self, weights: &WeightsSource) -> Result<(PathBuf, Vec<u8>), PolicyError> {
        match weights {
            WeightsSource::Safetensors(path) => {
                let bytes = std::fs::read(path)
                    .map_err(|e| PolicyError::Io(format!("{}: {e}", path.display())))?;
                Ok((path.clone(), bytes))
            }
            WeightsSource::InMemory(bytes) => {
                let path = std::env::temp_dir().join(format!(
                    "es-policy-{}.safetensors",
                    hex(&weights_hash(bytes))
                ));
                std::fs::write(&path, bytes)
                    .map_err(|e| PolicyError::Io(format!("{}: {e}", path.display())))?;
                self.scratch = Some(path.clone());
                Ok((path, bytes.clone()))
            }
        }
    }

    /// Load a module that was lowered outside [`lower_to_torch`].
    ///
    /// One caller: [`crate::lerobot::lower_act`], whose architecture parameters live in the
    /// checkpoint's own `config.json` because spec 8.3's node parameters do not carry them.
    /// Every check [`PolicyRuntime::load`] performs is performed here too — the declared
    /// weights hash (spec 5.3) and the key/shape contract — because a runtime that loads
    /// whatever it is handed cannot support the hash chain.
    pub fn load_lowered(
        &mut self,
        module: &TorchModule,
        policy: &PolicyHandle,
        weights: &WeightsSource,
    ) -> Result<PolicyInfo, PolicyError> {
        let (path, bytes) = self.checkpoint(weights)?;
        let got = weights_hash(&bytes);
        let expected = *policy.weights.hash();
        if got != expected {
            return Err(PolicyError::WeightsHash {
                expected: hex(&expected),
                got: hex(&got),
            });
        }
        validate_keys(module, &parse_header(&bytes)?)?;

        let mut process = Process::spawn()?;
        let reply: LoadReply = process.call(&Request::Load {
            source: &module.source,
            weights_path: &path.to_string_lossy(),
        })?;

        let info = PolicyInfo {
            backend: InferenceBackend::Torch,
            lowering_hash: module.lowering_hash,
            weights_hash: got,
            action_dim: policy.contract.action_dim,
            horizon: policy.contract.horizon,
            version: reply.torch_version,
        };
        self.process = Some(process);
        self.info = Some(info.clone());
        Ok(info)
    }
}

impl Drop for TorchRuntime {
    fn drop(&mut self) {
        // The process must go first: it has the file open.
        self.process = None;
        if let Some(path) = &self.scratch {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl PolicyRuntime for TorchRuntime {
    fn load(
        &mut self,
        graph: &LearningGraph,
        weights: &WeightsSource,
    ) -> Result<PolicyInfo, PolicyError> {
        // The checkpoint must be the one the IR names (spec 5.3) and must fit the lowered
        // graph (design note section 4). Both before Python sees it.
        let module = lower_to_torch(graph)?;
        self.load_lowered(&module, &graph.policy, weights)
    }

    fn infer(
        &mut self,
        inputs: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>, PolicyError> {
        let process = self.process.as_mut().ok_or(PolicyError::NotLoaded)?;
        let mut wire = BTreeMap::new();
        for (name, t) in inputs {
            wire.insert(name.clone(), WireTensor::encode(t)?);
        }
        let reply: InferReply = process.call(&Request::Infer { inputs: wire })?;
        reply
            .outputs
            .into_iter()
            .map(|(name, t)| {
                let tensor = t.decode(&name)?;
                Ok((name, tensor))
            })
            .collect()
    }

    fn info(&self) -> Option<&PolicyInfo> {
        self.info.as_ref()
    }

    /// Before a load there is no `torch` version to hash, so this covers the protocol alone.
    /// An `execution_hash` is only meaningful once a policy is loaded (spec 5.3).
    fn runtime_hash(&self) -> [u8; 32] {
        let version = self.info.as_ref().map_or("", |i| i.version.as_str());
        runtime_hash_of(InferenceBackend::Torch, PROTOCOL_VERSION, version)
    }
}

// --- base64 (RFC 4648, standard alphabet, padded) ------------------------------------------
//
// Twenty lines against a dependency on the inference path. `std` has no base64 and the
// protocol needs exactly one alphabet.

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[((n >> (18 - 6 * i)) & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn b64_decode(text: &str) -> Option<Vec<u8>> {
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    for c in text.bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = B64.iter().position(|x| *x == c)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The protocol is tested with canned lines, so it is covered without Python (spec 1.4:
    /// the harness must run where the oracle does not).
    #[test]
    fn a_successful_reply_decodes() {
        let reply: LoadReply =
            parse_response(r#"{"ok":true,"torch_version":"2.14.0+cpu"}"#).unwrap();
        assert_eq!(reply.torch_version, "2.14.0+cpu");

        let reply: InferReply = parse_response(
            r#"{"ok":true,"outputs":{"actions":{"shape":[1,2],"dtype":"F32","data_b64":"AACAPwAAAEA="}}}"#,
        )
        .unwrap();
        let t = reply.outputs["actions"].clone().decode("actions").unwrap();
        assert_eq!(t.shape, vec![1, 2]);
        assert_eq!(
            t.data,
            [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat()
        );
    }

    #[test]
    fn a_failed_reply_becomes_a_backend_error() {
        let err = parse_response::<LoadReply>(
            r#"{"ok":false,"error":"RuntimeError: size mismatch for n4.weight"}"#,
        )
        .unwrap_err();
        assert_eq!(
            err,
            PolicyError::Backend("RuntimeError: size mismatch for n4.weight".to_owned())
        );
    }

    #[test]
    fn malformed_replies_are_protocol_errors() {
        for line in [
            "not json at all",
            r#"{"torch_version":"2"}"#,
            r#"{"ok":true,"torch_version":2}"#,
        ] {
            assert!(
                matches!(
                    parse_response::<LoadReply>(line).unwrap_err(),
                    PolicyError::Protocol(_)
                ),
                "{line}"
            );
        }
        // A payload that does not fill its declared shape must not be reinterpreted.
        let short = WireTensor {
            shape: vec![4],
            dtype: "F32".to_owned(),
            data_b64: b64_encode(&[0u8; 8]),
        };
        assert!(matches!(
            short.decode("x").unwrap_err(),
            PolicyError::Protocol(_)
        ));
    }

    #[test]
    fn requests_encode_as_the_script_expects() {
        assert_eq!(
            serde_json::to_string(&Request::Load {
                source: "pass",
                weights_path: "w.safetensors"
            })
            .unwrap(),
            r#"{"cmd":"load","source":"pass","weights_path":"w.safetensors"}"#
        );
        let t = crate::equiv::action_chunk(1, 2, &[1.0, 2.0]);
        let inputs = [("x".to_owned(), WireTensor::encode(&t).unwrap())].into();
        assert_eq!(
            serde_json::to_string(&Request::Infer { inputs }).unwrap(),
            r#"{"cmd":"infer","inputs":{"x":{"shape":[1,2],"dtype":"F32","data_b64":"AACAPwAAAEA="}}}"#
        );
    }

    #[test]
    fn base64_round_trips_every_tail_length() {
        for n in 0..=32usize {
            let data: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            let text = b64_encode(&data);
            assert_eq!(text.len() % 4, 0, "n={n}");
            assert_eq!(b64_decode(&text).as_deref(), Some(data.as_slice()), "n={n}");
        }
        // Known vectors, so a self-consistent-but-wrong alphabet cannot hide.
        assert_eq!(b64_encode(b"f"), "Zg==");
        assert_eq!(b64_encode(b"fo"), "Zm8=");
        assert_eq!(b64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(b64_decode("Zm9vYmFy").unwrap(), b"foobar");
        assert!(b64_decode("not*base64").is_none());
    }

    #[test]
    fn inference_before_a_load_is_not_loaded() {
        let mut rt = TorchRuntime::new();
        assert!(rt.info().is_none());
        assert_eq!(
            rt.infer(&BTreeMap::new()).unwrap_err(),
            PolicyError::NotLoaded
        );
    }

    #[test]
    fn the_embedded_script_is_the_file_on_disk() {
        assert!(SCRIPT.contains("class EsPolicy"));
        assert!(SCRIPT.contains("\"cmd\""));
        assert!(SCRIPT.contains("load_state_dict"));
        // INV-16: the reference process must have no way to unpickle anything.
        assert!(!SCRIPT.contains("import pickle"));
        assert!(!SCRIPT.contains("torch.load("));
    }
}
