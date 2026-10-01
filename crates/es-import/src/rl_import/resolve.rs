//! The adapter resolved against our robot (packet M11/X2): the joint order and its permutation,
//! the policy period, the source terms the IR cannot compute, and the exact fold of each
//! channel's offset, scale, history order and the joint permutation into the normalizer.

use es_ir::deployment::DeploymentIr;
use es_ir::task::ObsSource;
use es_ir::types::Frame;

use super::adapter::{Adapter, ChannelMap, HistoryOrder, JointBlock, Offset, Scale};
use super::manifest::ImportManifest;
use super::ImportError;

// --- adapter v2 (packet M11/X2) --------------------------------------------------------------

/// Source terms the IR cannot compute on this robot (spec 28.14 "not on the ladder").
pub(super) const UNCOMPUTABLE: [&str; 4] = [
    "projected_gravity",
    "base_lin_vel",
    "base_ang_vel",
    "velocity_commands",
];

/// The term a channel's free-text `source` names: `"mdp.projected_gravity"` and
/// `"projected_gravity(robot)"` are both `projected_gravity`. A v1 `source` such as
/// `"qpos[0:6]"` reads as `qpos`, which names nothing refused.
pub(super) fn term_name(source: &str) -> &str {
    let s = source.trim();
    let s = s.strip_prefix("mdp.").unwrap_or(s);
    s.split(['(', '[', ' ']).next().unwrap_or(s)
}

/// `[joints]` -> (our actuator name of each source joint in source order, the permutation).
/// Exactly one of `source_order` (v1, recorded) and `source_names` (v2, permuted) (`IMP-008`).
pub(super) fn resolve_joints(
    joints: &JointBlock,
    actuators: &[String],
    scene: &str,
    dim: u32,
) -> Result<(Vec<String>, Option<Vec<usize>>), ImportError> {
    let order = |detail: String| ImportError::JointOrder { detail };
    let Some(source) = &joints.source_names else {
        if joints.source_order.is_empty() {
            return Err(order(
                "[joints] declares neither `source_order` nor `source_names`; the source's \
                 joint order is not something the import may assume"
                    .to_owned(),
            ));
        }
        if !joints.rename.is_empty() {
            return Err(order(
                "[joints.rename] renames source names, and a v1 `source_order` is already in \
                 our names; declare `source_names` instead"
                    .to_owned(),
            ));
        }
        return Ok((joints.source_order.clone(), None));
    };
    if !joints.source_order.is_empty() {
        return Err(order(
            "[joints] declares both `source_order` and `source_names`: two statements of one \
             order, and nothing here may pick between them"
                .to_owned(),
        ));
    }
    if let Some(unused) = joints.rename.keys().find(|k| !source.contains(k)) {
        return Err(order(format!(
            "[joints.rename] renames \"{unused}\", which is not one of source_names {source:?}"
        )));
    }
    let mut names = Vec::with_capacity(source.len());
    let mut perm = Vec::with_capacity(source.len());
    for name in source {
        let ours = joints.rename.get(name).unwrap_or(name);
        let at =
            actuators
                .iter()
                .position(|a| a == ours)
                .ok_or_else(|| ImportError::UnknownJoint {
                    joint: ours.clone(),
                    scene: scene.to_owned(),
                    actuators: actuators.to_vec(),
                })?;
        if at >= dim as usize || perm.contains(&at) {
            return Err(order(format!(
                "source joint \"{name}\" resolves to actuator \"{ours}\" (index {at}), which is \
                 {} -- the resolution must be a permutation of the {dim} actuators the \
                 ActionSpec drives",
                if perm.contains(&at) {
                    "already taken by another source joint"
                } else {
                    "past the ActionSpec's width"
                }
            )));
        }
        names.push(ours.clone());
        perm.push(at);
    }
    Ok((names, Some(perm)))
}

/// `IMP-006`: the adapter's `policy_dt` and the manifest's `decimation * sim_dt`, each against
/// the Deployment IR's control period. Returns the report line, `None` when neither states one.
pub(super) fn check_timing(
    adapter: &Adapter,
    manifest: &ImportManifest,
    deployment: &DeploymentIr,
) -> Result<Option<String>, ImportError> {
    let period = 1.0 / deployment.rate.control.as_hz_f64();
    let manifest_dt = match (manifest.decimation, manifest.sim_dt) {
        (Some(n), Some(dt)) => Some((f64::from(n) * dt, format!("decimation {n} x sim_dt {dt}"))),
        _ => None,
    };
    let adapter_dt = adapter
        .timing
        .as_ref()
        .map(|t| (t.policy_dt, "[timing] policy_dt".to_owned()));
    let mut said = Vec::new();
    for (dt, what) in [adapter_dt, manifest_dt].into_iter().flatten() {
        if (dt - period).abs() > 1e-9 * period {
            return Err(ImportError::Timing {
                detail: format!(
                    "{what} = {dt} s per policy step, but the Deployment IR's control period is \
                     {period} s. A policy run at another rate than it was trained at sees a \
                     different world per step; the import checks the period and never \
                     resamples it"
                ),
            });
        }
        said.push(format!("{what} = {dt} s"));
    }
    Ok((!said.is_empty()).then(|| {
        format!(
            "{} == the Deployment IR's control period {period} s",
            said.join(", ")
        )
    }))
}

/// The observation side of the fold: the `Normalize{MeanStd}` stats in *our* flat order
/// (`None`: no source statistics and nothing folded -- v1's identity `Range`), and the first
/// Dense's input column for each of our positions (`None`: the identity).
pub(super) struct Fold {
    pub(super) stats: Option<(Vec<f64>, Vec<f64>)>,
    pub(super) columns: Option<Vec<usize>>,
}

/// Every channel's `(x - offset) * scale`, the joint permutation, the history order and the
/// previous action's un-normalization, folded into one per-element affine map and the
/// source's `(o - mean) / std`:
///
/// ```text
/// n_j = ((x - a_j) * b_j - mean_j) / std_j = (x - (a_j + mean_j / b_j)) / (std_j / b_j)
/// ```
///
/// exact algebra, evaluated once in f64. For a `PreviousAction` channel `x` is our row in
/// actuator units and the source saw its raw action, so `a` and `b` also carry the action
/// tail's inverse: `raw = (x - offset) / scale`. An element with `a = 0, b = 1` keeps the
/// source's numbers bit for bit, which is what leaves every v1 conversion unmoved.
// The names are the formula's above; the float comparisons test for exactly the identity.
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
pub(super) fn fold_channels(
    manifest: &ImportManifest,
    channels: &[(&ChannelMap, &es_ir::task::ObsChannel)],
    perm: Option<&[usize]>,
    action_scale: &[f64],
    action_offset: &[f64],
    default_pos: Option<&[f64]>,
) -> Result<Fold, ImportError> {
    let n = manifest.obs_dim as usize;
    let dim = action_scale.len();
    let bad = |detail: String| ImportError::Channels { detail };
    let (mu, sigma) = match (&manifest.obs_mean, &manifest.obs_std) {
        (Some(m), Some(s)) => {
            if m.len() != n || s.len() != n {
                return Err(ImportError::Manifest(format!(
                    "obs_mean/obs_std are {}/{} long for obs_dim {n}",
                    m.len(),
                    s.len()
                )));
            }
            (Some(m), Some(s))
        }
        _ => (None, None),
    };
    let mut src_of: Vec<usize> = (0..n).collect();
    let mut a = vec![0.0; n];
    let mut b = vec![1.0; n];
    let mut folded = false;
    for (map, declared) in channels {
        let start = map.slice[0] as usize;
        let width = declared.ty.shape.dims().iter().product::<u64>() as usize;
        let history = map.history.unwrap_or(1) as usize;
        let previous = matches!(declared.source, ObsSource::PreviousAction { .. });
        let per_joint = previous
            || matches!(declared.source, ObsSource::JointState { .. })
                && matches!(declared.ty.frame, Frame::Joint(_));
        if (per_joint && (perm.is_some() || previous)) && width != dim {
            return Err(bad(format!(
                "channel \"{}\" is a per-joint channel {width} wide, and the action is {dim}: \
                 a joint permutation or an action tail cannot be applied to it",
                map.channel
            )));
        }
        let term_scale: Vec<f64> = match &map.scale {
            None => vec![1.0; width],
            Some(Scale::One(s)) => vec![*s; width],
            Some(Scale::Each(v)) if v.len() == width => v.clone(),
            Some(Scale::Each(v)) => {
                return Err(bad(format!(
                    "channel \"{}\" has {} scale entries for a {width}-wide channel",
                    map.channel,
                    v.len()
                )))
            }
        };
        let term_offset: Vec<f64> = match &map.offset {
            None => vec![0.0; width],
            Some(Offset::Each(v)) if v.len() == width => v.clone(),
            Some(Offset::Each(v)) => {
                return Err(bad(format!(
                    "channel \"{}\" has {} offset entries for a {width}-wide channel",
                    map.channel,
                    v.len()
                )))
            }
            Some(Offset::Named(name)) if name == "default_pos" => {
                let Some(pose) = default_pos.filter(|_| per_joint && width == dim) else {
                    return Err(bad(format!(
                        "channel \"{}\" declares offset = \"default_pos\", which needs a \
                         per-joint channel as wide as the action and a pose stated by \
                         [joints] default_pos or the manifest's default_joint_pos",
                        map.channel
                    )));
                };
                pose.to_vec()
            }
            Some(Offset::Named(other)) => {
                return Err(ImportError::Adapter(format!(
                    "channel \"{}\": offset = \"{other}\" is not \"default_pos\" or a vector",
                    map.channel
                )))
            }
        };
        if term_scale.iter().any(|s| *s == 0.0 || !s.is_finite()) {
            return Err(bad(format!(
                "channel \"{}\" has a zero or non-finite scale: a term the source multiplied by \
                 zero carries nothing, and no normalizer can divide it back out",
                map.channel
            )));
        }
        if previous {
            if let ObsSource::PreviousAction { initial } = &declared.source {
                // Isaac's `last_action` is zero on the first tick of an episode; ours is
                // `initial` in actuator units, which folds to zero only if it is the offset.
                let want = match perm {
                    Some(p) => permuted(action_offset, p),
                    None => action_offset.to_vec(),
                };
                let have = initial.clone().unwrap_or_else(|| vec![0.0; dim]);
                if have.len() != dim || have.iter().zip(&want).any(|(h, w)| (h - w).abs() > 1e-9) {
                    return Err(bad(format!(
                        "channel \"{}\" is the previous action, whose raw value is zero at an \
                         episode's first tick in the source; ours is `initial` in actuator \
                         units, so the Task IR must declare initial = {want:?} (the action \
                         offset, in our order), not {have:?}",
                        map.channel
                    )));
                }
            }
        }
        for e in 0..width * history {
            let (f, k) = (e / width, e % width);
            let our_k = match perm {
                Some(p) if per_joint => p[k],
                _ => k,
            };
            let our_f = match map.history_order {
                Some(HistoryOrder::NewestFirst) => history - 1 - f,
                _ => f,
            };
            let j = start + e;
            src_of[start + our_f * width + our_k] = j;
            let (aj, bj) = if previous {
                (
                    action_offset[k] + action_scale[k] * term_offset[k],
                    term_scale[k] / action_scale[k],
                )
            } else {
                (term_offset[k], term_scale[k])
            };
            folded |= aj != 0.0 || bj != 1.0;
            a[j] = aj;
            b[j] = bj;
        }
    }
    let identity = src_of.iter().enumerate().all(|(i, j)| i == *j);
    let stats = if mu.is_some() || folded {
        let mut mean = vec![0.0; n];
        let mut std = vec![0.0; n];
        for (ours, &j) in src_of.iter().enumerate() {
            let (m, s) = (mu.map_or(0.0, |v| v[j]), sigma.map_or(1.0, |v| v[j]));
            // Exactly the identity, so v1 keeps its bits.
            let untouched = a[j] == 0.0 && b[j] == 1.0;
            mean[ours] = if untouched { m } else { a[j] + m / b[j] };
            std[ours] = if untouched { s } else { s / b[j] };
        }
        Some((mean, std))
    } else {
        None
    };
    Ok(Fold {
        stats,
        columns: (!identity).then_some(src_of),
    })
}

/// `out[perm[i]] = v[i]`: a source-order vector in our order.
pub(super) fn permuted(v: &[f64], perm: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (i, p) in perm.iter().enumerate() {
        out[*p] = v[i];
    }
    out
}
