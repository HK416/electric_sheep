//! Safetensors: the only weight format on this path (`INV-16`, spec 8.7).
//!
//! A reader, a writer, and the check that a checkpoint is the one the lowered graph asked for.
//! The format is eight bytes of little-endian header length, that many bytes of JSON, then the
//! raw tensor bytes — small enough that a dependency would cost more than it saves, and
//! writing it here means the Python side can read it with `struct` and `json` alone instead of
//! needing the `safetensors` package installed next to `torch`.
//!
//! Nothing here can execute code, which is the whole reason the format was chosen.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Deserialize;

use crate::lower::TorchModule;
use crate::runtime::PolicyError;

/// Every safetensors key this project emits is under this prefix, with the node id next:
/// `nodes.<node_id>.<param path>` (design note section 4).
pub const WEIGHT_PREFIX: &str = "nodes.";

/// A checkpoint held in memory: key -> (shape, row-major f32 values).
pub type Checkpoint = BTreeMap<String, (Vec<u64>, Vec<f32>)>;

/// One tensor in a safetensors file, as the header describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsEntry {
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Byte range within the data segment, i.e. relative to the end of the header.
    pub offsets: (u64, u64),
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// The header's JSON object and the length of the data segment behind it.
fn header_of(bytes: &[u8]) -> Result<(BTreeMap<String, serde_json::Value>, u64), PolicyError> {
    let len_bytes: [u8; 8] = bytes
        .get(..8)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| PolicyError::Safetensors("file is shorter than the 8-byte length".into()))?;
    let n = u64::from_le_bytes(len_bytes) as usize;
    let header = bytes.get(8..8 + n).ok_or_else(|| {
        PolicyError::Safetensors(format!("header claims {n} bytes, file is short"))
    })?;
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(header)
        .map_err(|e| PolicyError::Safetensors(format!("header is not a JSON object: {e}")))?;
    Ok((raw, (bytes.len() - 8 - n) as u64))
}

/// One entry of the header's `__metadata__` string map — safetensors' own free-form slot, the
/// only place a checkpoint may carry something that is not a tensor.
pub fn metadata(bytes: &[u8], key: &str) -> Result<Option<String>, PolicyError> {
    let (raw, _) = header_of(bytes)?;
    Ok(raw
        .get("__metadata__")
        .and_then(|m| m.get(key))
        .and_then(|v| v.as_str())
        .map(str::to_owned))
}

/// Parse a safetensors header. The tensor bytes themselves are never touched — the Python side
/// reads those, and shape checking only needs the header.
pub fn parse_header(bytes: &[u8]) -> Result<BTreeMap<String, SafetensorsEntry>, PolicyError> {
    let (raw, data_len) = header_of(bytes)?;
    let mut out = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry: RawEntry = serde_json::from_value(value)
            .map_err(|e| PolicyError::Safetensors(format!("entry \"{name}\": {e}")))?;
        let [a, b] = entry.data_offsets;
        if a > b || b > data_len {
            return Err(PolicyError::Safetensors(format!(
                "entry \"{name}\": offsets [{a}, {b}] fall outside {data_len} bytes of data"
            )));
        }
        out.insert(
            name,
            SafetensorsEntry {
                dtype: entry.dtype,
                shape: entry.shape,
                offsets: (a, b),
            },
        );
    }
    Ok(out)
}

/// Write a safetensors file. `f32` only: spec 8.4's `runtime.dtype` is `F32` on this path and
/// a second dtype here would be a second thing to keep in sync with the Python reader.
pub fn write_safetensors(tensors: &Checkpoint) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, (shape, values)) in tensors {
        let start = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    // `serde_json::Map` preserves insertion order, and `tensors` is a BTreeMap, so the bytes
    // are a function of the content alone (spec 3.4).
    let header = serde_json::to_vec(&serde_json::Value::Object(header))
        .expect("a map of plain JSON values always serializes");
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    out
}

/// Check a checkpoint against a lowered module (design note section 4).
///
/// - an exact key must be present with the declared shape;
/// - a file key under a `prefix.*` claim is accepted — those names belong to torchvision or
///   `torch.nn`, not to us;
/// - a file key matching neither is `unexpected`.
pub fn validate_keys(
    module: &TorchModule,
    file: &BTreeMap<String, SafetensorsEntry>,
) -> Result<(), PolicyError> {
    let mut missing = Vec::new();
    let mut shape = Vec::new();
    let mut claims = Vec::new();

    for key in &module.weight_keys {
        if let Some(prefix) = key.strip_suffix('*') {
            if !file.keys().any(|k| k.starts_with(prefix)) {
                missing.push(key.clone());
            }
            claims.push(prefix.to_owned());
            continue;
        }
        match file.get(key) {
            None => missing.push(key.clone()),
            Some(entry) => {
                if let Some(want) = module.weight_shapes.get(key) {
                    if entry.shape != *want {
                        shape.push(format!("{key}: want {want:?}, got {:?}", entry.shape));
                    }
                }
                if entry.dtype != "F32" {
                    shape.push(format!("{key}: want dtype F32, got {}", entry.dtype));
                }
            }
        }
    }

    let unexpected: Vec<String> = file
        .keys()
        .filter(|k| {
            !module.weight_shapes.contains_key(*k) && !claims.iter().any(|p| k.starts_with(p))
        })
        .cloned()
        .collect();

    if missing.is_empty() && unexpected.is_empty() && shape.is_empty() {
        Ok(())
    } else {
        Err(PolicyError::WeightMismatch {
            missing,
            unexpected,
            shape,
        })
    }
}

/// `blake3` of the checkpoint bytes — the `WeightsRef::hash` of spec 5.3.
pub fn weights_hash(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

pub fn hex(digest: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in digest {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::act_from_scratch;
    use crate::lower::lower_to_torch;

    /// A checkpoint for the ACT fixture: every exact key at its declared shape, plus two
    /// tensors standing in for each opaque sub-module.
    fn act_checkpoint() -> (TorchModule, Checkpoint) {
        let module = lower_to_torch(&act_from_scratch(8, 512, 8, 50, 20, 1)).unwrap();
        let mut file: Checkpoint = module
            .weight_shapes
            .iter()
            .map(|(k, shape)| {
                let n = shape.iter().product::<u64>() as usize;
                (k.clone(), (shape.clone(), vec![0.5f32; n]))
            })
            .collect();
        for claim in module.weight_keys.iter().filter(|k| k.ends_with(".*")) {
            let prefix = claim.trim_end_matches('*');
            file.insert(format!("{prefix}conv1.weight"), (vec![2], vec![1.0, 2.0]));
        }
        (module, file)
    }

    #[test]
    fn a_written_file_reads_back() {
        let (_, tensors) = act_checkpoint();
        let bytes = write_safetensors(&tensors);
        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.len(), tensors.len());
        let e = &header["nodes.4.bias"];
        assert_eq!((e.dtype.as_str(), e.shape.as_slice()), ("F32", &[400][..]));
        assert_eq!(e.offsets.1 - e.offsets.0, 400 * 4);
        // Deterministic: the same tensors give the same bytes (spec 3.4).
        assert_eq!(bytes, write_safetensors(&tensors));
    }

    #[test]
    fn a_matching_checkpoint_validates() {
        let (module, tensors) = act_checkpoint();
        let header = parse_header(&write_safetensors(&tensors)).unwrap();
        validate_keys(&module, &header).unwrap();
    }

    #[test]
    fn a_missing_key_is_caught() {
        let (module, mut tensors) = act_checkpoint();
        tensors.remove("nodes.4.bias");
        let header = parse_header(&write_safetensors(&tensors)).unwrap();
        let err = validate_keys(&module, &header).unwrap_err();
        let PolicyError::WeightMismatch { missing, .. } = &err else {
            panic!("{err}")
        };
        assert_eq!(missing, &["nodes.4.bias"]);
    }

    #[test]
    fn an_empty_prefix_claim_is_missing_too() {
        let (module, mut tensors) = act_checkpoint();
        tensors.remove("nodes.0.conv1.weight");
        let header = parse_header(&write_safetensors(&tensors)).unwrap();
        let err = validate_keys(&module, &header).unwrap_err();
        let PolicyError::WeightMismatch { missing, .. } = &err else {
            panic!("{err}")
        };
        assert_eq!(missing, &["nodes.0.*"]);
    }

    #[test]
    fn a_wrong_shape_and_a_stray_key_are_caught() {
        let (module, mut tensors) = act_checkpoint();
        tensors.insert("nodes.4.weight".to_owned(), (vec![399, 512], vec![0.0; 1]));
        tensors.insert("nodes.9.weight".to_owned(), (vec![1], vec![0.0]));
        let header = parse_header(&write_safetensors(&tensors)).unwrap();
        let err = validate_keys(&module, &header).unwrap_err();
        let PolicyError::WeightMismatch {
            unexpected, shape, ..
        } = &err
        else {
            panic!("{err}")
        };
        assert_eq!(unexpected, &["nodes.9.weight"]);
        assert_eq!(shape.len(), 1, "{shape:?}");
        assert!(shape[0].starts_with("nodes.4.weight: want [400, 512]"));
    }

    #[test]
    fn a_truncated_or_lying_header_is_an_error() {
        assert!(parse_header(&[0u8; 4]).is_err());
        let mut bytes = write_safetensors(&BTreeMap::new());
        bytes[0] = 0xff;
        assert!(parse_header(&bytes).is_err());
        // An entry pointing past the end of the data segment.
        let header = br#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let mut bad = (header.len() as u64).to_le_bytes().to_vec();
        bad.extend_from_slice(header);
        assert!(parse_header(&bad).is_err());
    }

    #[test]
    fn metadata_is_not_a_tensor() {
        let header = br#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        let parsed = parse_header(&bytes).unwrap();
        assert_eq!(parsed.keys().collect::<Vec<_>>(), vec!["a"]);
    }

    #[test]
    fn hashes_are_content_addressed() {
        let a = weights_hash(b"weights");
        assert_eq!(a, weights_hash(b"weights"));
        assert_ne!(a, weights_hash(b"weightt"));
        assert_eq!(hex(&a).len(), 64);
    }
}
