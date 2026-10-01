//! "Make an editable copy" of a template project's scene (design sections 3.2 and 5): the
//! template's scene document (G1's — the robot `[[include]]`d by reference, the objects, cameras
//! and lights written in it) becomes the project's `scene.esscene`, and its task specification
//! (G3a) the project's `task.estask`, naming that scene.
//!
//! The files the document names — each include, and the meshes and textures they and it name —
//! are copied into the project **under the same relative paths**: asset paths are hash input
//! (spec 5.3), so the copy expands to the template scene's `SceneDesc` bit for bit and
//! `scene_hash` does not move. The document itself is copied byte for byte (its comments too).
// ponytail: files are copied by their relative names, not under `assets/<blake3>` (design 3.1):
// that renaming would move `scene_hash` for a scene nobody edited. A path out of the template's
// directory (`..`) is refused rather than flattened.

use std::path::{Component, Path, PathBuf};

use es_assets::esscene::{expand, EsScene};
use es_assets::scene::AssetKind;
use es_script::spec::TaskSpec;

use crate::{SCENE_FILE, SPEC_FILE};

/// Writes `root/scene.esscene` from the document `scene`, the files it names beside it, and
/// `root/task.estask` from `spec` with its `scene` naming the copy. Refuses a project that is
/// already editable. The scene document is written last: its presence makes the project
/// editable, so a failure leaves none half made.
pub fn make_editable(root: &Path, scene: &Path, spec: Option<&Path>) -> Result<(), String> {
    let target = root.join(SCENE_FILE);
    if target.exists() {
        return Err(format!("{}: already editable", target.display()));
    }
    let read = |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
    let text = read(scene)?;
    let dir = scene.parent().unwrap_or(Path::new("."));
    let doc = EsScene::from_toml(&text).map_err(|e| format!("{}: {e}", scene.display()))?;
    let desc = expand(&doc, dir).map_err(|e| format!("{}: {e}", scene.display()))?;

    let mut files: Vec<String> = doc.includes.iter().map(|i| i.source.clone()).collect();
    let meshes = desc.assets.iter().filter(|a| a.kind == AssetKind::Mesh);
    files.extend(meshes.map(|a| a.path.clone()));
    for t in desc.textures.values() {
        files.extend(t.spec.file.iter().cloned());
        files.extend(t.spec.cubefiles.iter().flatten().cloned());
    }
    files.sort();
    files.dedup();
    let spec_text = spec.map(|p| adapt(&read(p)?, p)).transpose()?;

    for f in &files {
        let rel = Path::new(f);
        if !rel.components().all(|c| matches!(c, Component::Normal(_))) {
            return Err(format!(
                "{f}: only a file under the scene's own folder can be copied"
            ));
        }
        let to = root.join(rel);
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::copy(dir.join(rel), &to).map_err(|e| format!("{f}: {e}"))?;
    }
    let write =
        |p: &Path, t: &str| std::fs::write(p, t).map_err(|e| format!("{}: {e}", p.display()));
    if let Some(t) = spec_text {
        write(&root.join(SPEC_FILE), &t)?;
    }
    write(&target, &text)
}

/// An editable project's documents copied from `from` into `to` (packet M17/G9): "save as
/// template" (a project into a template's folder) and a project made from a saved template (the
/// other way). `assets/` is copied whole, not by what the scene names: a glTF's buffers are not
/// in its asset list (G7). Then `task.estask` when there is one, and `scene.esscene` last, as
/// [`make_editable`] writes it. Refuses a `to` that already holds a scene document.
pub fn documents(from: &Path, to: &Path) -> Result<(), String> {
    let target = to.join(SCENE_FILE);
    if target.exists() {
        return Err(format!("{}: already editable", target.display()));
    }
    let fail = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
    std::fs::create_dir_all(to).map_err(|e| fail(to, e))?;
    let mut todo = vec![PathBuf::from(crate::import::ASSETS)];
    while let Some(rel) = todo.pop() {
        let Ok(entries) = std::fs::read_dir(from.join(&rel)) else {
            continue;
        };
        std::fs::create_dir_all(to.join(&rel)).map_err(|e| fail(&to.join(&rel), e))?;
        for e in entries.flatten() {
            let rel = rel.join(e.file_name());
            if e.path().is_dir() {
                todo.push(rel);
            } else {
                std::fs::copy(e.path(), to.join(&rel)).map_err(|err| fail(&e.path(), err))?;
            }
        }
    }
    for file in [SPEC_FILE, SCENE_FILE] {
        let src = from.join(file);
        if src.is_file() {
            std::fs::copy(&src, to.join(file)).map_err(|e| fail(&src, e))?;
        }
    }
    Ok(())
}

/// The specification `text` with its `scene` line naming the project's scene document; the
/// rest, comments included, as written. Checked by reading it back.
fn adapt(text: &str, path: &Path) -> Result<String, String> {
    let fail = |why: String| format!("{}: {why}", path.display());
    let mut spec = TaskSpec::from_toml(text).map_err(|e| fail(e.to_string()))?;
    let line = format!("scene = \"{}\"", spec.scene);
    let out = text.replacen(&line, &format!("scene = \"{SCENE_FILE}\""), 1);
    SCENE_FILE.clone_into(&mut spec.scene);
    match TaskSpec::from_toml(&out) {
        Ok(read) if read == spec => Ok(out),
        // The line is written another way: the document as the reader has it.
        _ => spec.to_toml().map_err(|e| fail(e.to_string())),
    }
}
