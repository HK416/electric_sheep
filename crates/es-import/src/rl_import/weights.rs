//! The source's tensors under the keys `lower_to_torch` declares: paired by order, each
//! shape-checked, and moved by adapter v2's column gather and row permutation (packet M11/X2).

use std::collections::BTreeMap;

use es_ir::learning::LearningGraph;
use es_policy::lower::lower_to_torch;
use es_policy::weights::{parse_header, validate_keys, write_safetensors, Checkpoint};

use super::manifest::ImportManifest;
use super::ImportError;

/// The neutral keys onto the keys the lowering declares.
///
/// The pairing is by **order**, not by a formula: `lower_to_torch` numbers an `nn.Sequential`'s
/// members including the activation modules, and recomputing that index here would be a second
/// copy of `torch.rs`'s layout. Every pair is shape-checked, and `validate_keys` re-checks the
/// whole file against the lowering afterwards, so a misalignment is refused rather than packed.
///
/// v2 moves no number, only positions (packet M11/X2): `columns[k]` is the source input the
/// first Dense's column `k` reads, and `rows[i]` is where the head's row `i` goes -- the same
/// function of our state as the source's of its own, exactly. Both `None` for v1.
pub(super) fn remap(
    learning: &LearningGraph,
    manifest: &ImportManifest,
    weights: &[u8],
    columns: Option<&[usize]>,
    rows: Option<&[usize]>,
) -> Result<Vec<u8>, ImportError> {
    let module = lower_to_torch(learning)
        .map_err(|e| ImportError::Ir(format!("lowering the Learning IR: {e}")))?;
    let header = parse_header(weights).map_err(|e| ImportError::Weights(e.to_string()))?;

    let mut neutral: Vec<String> = Vec::new();
    for i in 0..manifest.hidden.len() {
        neutral.push(format!("mlp.{i}.weight"));
        neutral.push(format!("mlp.{i}.bias"));
    }
    neutral.push("head.weight".to_owned());
    neutral.push("head.bias".to_owned());

    let ours: Vec<&String> = module
        .weight_keys
        .iter()
        .filter(|k| !k.ends_with(".*"))
        .collect();
    if ours.len() != neutral.len() {
        return Err(ImportError::Weights(format!(
            "the checkpoint carries {} tensors and the lowered module declares {}: {neutral:?} \
             against {ours:?}",
            neutral.len(),
            ours.len()
        )));
    }

    let mut out: Checkpoint = BTreeMap::new();
    let last = neutral.len() - 2;
    for (index, (from, to)) in neutral.iter().zip(&ours).enumerate() {
        let entry = header
            .get(from)
            .ok_or_else(|| ImportError::Weights(format!("no tensor named \"{from}\"")))?;
        let want = module.weight_shapes.get(*to);
        if want.is_some_and(|w| *w != entry.shape) {
            return Err(ImportError::Weights(format!(
                "\"{from}\" is {:?} but the lowering declares \"{to}\" as {:?}",
                entry.shape,
                want.expect("checked")
            )));
        }
        let mut values = tensor_f32(weights, entry)?;
        // The first hidden Dense's weight `[out, in]`: gather its input columns.
        if let (0, Some(cols)) = (index, columns) {
            let width = entry.shape.get(1).copied().unwrap_or(0) as usize;
            if cols.len() != width {
                return Err(ImportError::Weights(format!(
                    "\"{from}\" reads {width} inputs and the observation folds {}",
                    cols.len()
                )));
            }
            values = values
                .chunks_exact(width)
                .flat_map(|row| cols.iter().map(|c| row[*c]))
                .collect();
        }
        // The head's weight `[action, hidden]` and bias `[action]`: move source row `i` to
        // our row `rows[i]`.
        if let (true, Some(rows)) = (index >= last, rows) {
            let stride = values.len() / rows.len().max(1);
            let mut moved = vec![0.0f32; values.len()];
            for (i, to) in rows.iter().enumerate() {
                moved[to * stride..(to + 1) * stride]
                    .copy_from_slice(&values[i * stride..(i + 1) * stride]);
            }
            values = moved;
        }
        out.insert((*to).clone(), (entry.shape.clone(), values));
    }

    let bytes = write_safetensors(&out);
    let written = parse_header(&bytes).map_err(|e| ImportError::Weights(e.to_string()))?;
    validate_keys(&module, &written)
        .map_err(|e| ImportError::Weights(format!("the remapped checkpoint: {e}")))?;
    Ok(bytes)
}

/// One tensor's f32 values out of a safetensors blob.
fn tensor_f32(
    blob: &[u8],
    entry: &es_policy::weights::SafetensorsEntry,
) -> Result<Vec<f32>, ImportError> {
    if entry.dtype != "F32" {
        return Err(ImportError::Weights(format!(
            "dtype {} is not F32 (spec 8.4's runtime.dtype on this path)",
            entry.dtype
        )));
    }
    let len = u64::from_le_bytes(
        blob.get(..8)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| ImportError::Weights("file is shorter than its length prefix".into()))?,
    ) as usize;
    let base = 8 + len;
    let (a, b) = (
        base + entry.offsets.0 as usize,
        base + entry.offsets.1 as usize,
    );
    let bytes = blob
        .get(a..b)
        .ok_or_else(|| ImportError::Weights("a tensor runs past the end of the file".into()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
