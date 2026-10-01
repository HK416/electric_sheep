//! `robot` is a body of the scene or the handle of an `.esscene` `[[include]]` (the orchestrator,
//! 2026-10-01): an include names the robot by its one root body whose subtree has joints (the
//! Shadow Hand file's other root, `floor0`, has none).

use std::collections::BTreeSet;
use std::path::Path;

use es_assets::scene::SceneDesc;

use super::{refuse, SpecError, TaskSpec};

/// The robot's root body: `spec.robot` itself when the scene has a body of that name, else the
/// root of the include `spec.robot` names.
pub(super) fn root_body(
    spec: &TaskSpec,
    root: &Path,
    scene: &SceneDesc,
) -> Result<String, SpecError> {
    if scene.bodies.iter().any(|b| b.name == spec.robot) {
        return Ok(spec.robot.clone());
    }
    let no = || {
        refuse(
            "task-spec",
            "robot",
            format!(
                "no body or `.esscene` include named `{}` in the scene",
                spec.robot
            ),
        )
    };
    let path = root.join(&spec.scene);
    if !path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("esscene"))
    {
        return no();
    }
    let scene_err = |reason: String| SpecError::Scene {
        path: spec.scene.clone(),
        reason,
    };
    let text = std::fs::read_to_string(&path).map_err(|e| scene_err(e.to_string()))?;
    let doc =
        es_assets::esscene::EsScene::from_toml(&text).map_err(|e| scene_err(e.to_string()))?;
    let Some(inc) = doc.includes.iter().find(|i| i.name == spec.robot) else {
        return no();
    };
    // The file's root bodies, by its own names behind the include's prefix (G1's rule).
    let file = path.parent().unwrap_or(Path::new(".")).join(&inc.source);
    let raw = std::fs::read_to_string(&file).map_err(|e| scene_err(e.to_string()))?;
    let sub = if inc.source.to_lowercase().ends_with(".urdf") {
        let resolver = es_assets::urdf::PackageResolver::from_env();
        es_assets::urdf::parse_urdf(&raw, &resolver)
            .map_err(|e| scene_err(e.to_string()))?
            .scene
    } else {
        es_assets::parse_mjcf(&raw)
            .map_err(|e| scene_err(e.to_string()))?
            .scene
    };
    let world = sub.bodies.iter().find(|b| b.name == "world").map(|b| b.id);
    let prefix = inc.prefix.as_deref().unwrap_or("");
    let jointed: Vec<String> = sub
        .bodies
        .iter()
        .filter(|b| Some(b.id) != world && (b.parent.is_none() || b.parent == world))
        .map(|b| format!("{prefix}{}", b.name))
        .filter(|name| has_joints(scene, name))
        .collect();
    match &jointed[..] {
        [one] => Ok(one.clone()),
        _ => refuse(
            "task-spec",
            "robot",
            format!(
                "include `{}` has {} root bodies with joints ({jointed:?}); name the body",
                spec.robot,
                jointed.len()
            ),
        ),
    }
}

/// Whether the subtree of body `name` holds a joint. Bodies come parent first.
fn has_joints(scene: &SceneDesc, name: &str) -> bool {
    let Some(top) = scene.bodies.iter().find(|b| b.name == name) else {
        return false;
    };
    let mut tree = BTreeSet::from([top.id]);
    for b in &scene.bodies {
        if b.parent.is_some_and(|p| tree.contains(&p)) {
            tree.insert(b.id);
        }
    }
    scene.joints.iter().any(|j| tree.contains(&j.body))
}
