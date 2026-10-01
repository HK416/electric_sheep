//! Files a person brings into the project, copied by content (packet M17/G7, design section 3.1
//! and section 9's default 3): a mesh or a picture to `assets/<blake3>.<ext>`, a robot file to
//! `assets/<blake3 of the file>/` together with every file it names, each under its own relative
//! path, so the robot's reader resolves them unchanged.
//!
//! Asset paths are hash input (spec 5.3), so the path a scene document names is a function of the
//! bytes, not of where the person's file was: the same file brought twice, into any project,
//! gives the same `scene_hash` and writes nothing the second time. The editable copy of a
//! template (`crate::copy`, G5) keeps its relative paths on purpose and is not this.

use std::path::{Component, Path, PathBuf};

use es_assets::esscene::{expand, EsScene, Include};
use es_assets::scene::AssetKind;

/// Where imports go, relative to the project's root.
pub const ASSETS: &str = "assets";

/// What one import wrote — files and the folders made for them, in the order made — so that a
/// refused import can take back exactly that and nothing that was there before.
#[derive(Debug, Default)]
pub struct Written(pub Vec<PathBuf>);

impl Written {
    /// Removes what this import made, newest first.
    pub fn undo(self) {
        for p in self.0.iter().rev() {
            if std::fs::remove_file(p).is_err() {
                let _ = std::fs::remove_dir(p);
            }
        }
    }

    /// `bytes` at `root/rel`, unless they are there already; a different file there is refused.
    fn put(&mut self, root: &Path, rel: &str, bytes: &[u8]) -> Result<(), String> {
        let to = root.join(rel);
        match std::fs::read(&to) {
            Ok(old) if old == bytes => return Ok(()),
            Ok(_) => return Err(format!("{rel}: a different file is already there")),
            Err(_) => {}
        }
        let fail = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
        let missing: Vec<PathBuf> = (to.ancestors().skip(1))
            .take_while(|d| !d.exists())
            .map(Path::to_path_buf)
            .collect();
        for d in missing.into_iter().rev() {
            std::fs::create_dir(&d).map_err(|e| fail(&d, e))?;
            self.0.push(d);
        }
        std::fs::write(&to, bytes).map_err(|e| fail(&to, e))?;
        self.0.push(to);
        Ok(())
    }

    /// Runs `copy`; when it fails, takes back what it wrote.
    fn guard(copy: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<Self, String> {
        let mut w = Self::default();
        match copy(&mut w) {
            Ok(()) => Ok(w),
            Err(why) => {
                w.undo();
                Err(why)
            }
        }
    }
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// The file's name without its folder.
fn name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// A scene document with nothing in it, the base of a document made here.
pub fn empty() -> EsScene {
    EsScene::from_toml("kind = \"scene\"\nschema = 1\n").expect("the empty document reads")
}

/// A mesh or a picture, copied to `assets/<blake3>.<ext>` (the extension in lower case: the
/// readers choose by it). Returns that path, relative to `root`, and what was written.
pub fn file(root: &Path, src: &Path) -> Result<(String, Written), String> {
    let bytes = read(src)?;
    let ext = src.extension().map_or_else(String::new, |e| {
        format!(".{}", e.to_string_lossy().to_lowercase())
    });
    let rel = format!("{ASSETS}/{}{ext}", hash(&bytes));
    Written::guard(|w| w.put(root, &rel, &bytes)).map(|w| (rel, w))
}

/// A robot file (MJCF, URDF, glTF), copied to `assets/<blake3 of the file>/<its name>` with the
/// files it names under the same relative paths. A file that does not read, or that names a file
/// outside its own folder, is refused before anything is written.
pub fn robot(root: &Path, src: &Path) -> Result<(String, Written), String> {
    let bytes = read(src)?;
    let dir = src.parent().unwrap_or(Path::new("."));
    let file = name(src);
    let named = named(dir, &file, &bytes)?;
    let base = format!("{ASSETS}/{}", hash(&bytes));
    let w = Written::guard(|w| {
        w.put(root, &format!("{base}/{file}"), &bytes)?;
        for f in &named {
            w.put(root, &format!("{base}/{f}"), &read(&dir.join(f))?)?;
        }
        Ok(())
    })?;
    Ok((format!("{base}/{file}"), w))
}

/// The files the robot file `file` in `dir` names, relative to `dir`: what its reader loads.
/// A path with nothing behind it in `dir` (a glTF's embedded data, a texture `MuJoCo` makes) is
/// no file.
fn named(dir: &Path, file: &str, bytes: &[u8]) -> Result<Vec<String>, String> {
    let mut doc = empty();
    doc.includes.push(Include {
        name: "robot".into(),
        source: file.into(),
        prefix: None,
        pos: None,
        quat: None,
        set: None,
    });
    let s = expand(&doc, dir).map_err(|e| format!("{file}: {e}"))?;
    let assets = s.assets.iter();
    let mut files: Vec<String> = (assets
        .filter(|a| matches!(a.kind, AssetKind::Mesh | AssetKind::Texture)))
    .map(|a| a.path.clone())
    .collect();
    for t in s.textures.values() {
        files.extend(t.spec.file.iter().cloned());
        files.extend(t.spec.cubefiles.iter().flatten().cloned());
    }
    files.extend(uris(bytes));
    files.retain(|f| !f.is_empty() && dir.join(f).is_file());
    files.sort();
    files.dedup();
    let outside = |f: &String| {
        !Path::new(f)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
    };
    if let Some(f) = files.iter().find(|f| outside(f)) {
        return Err(format!(
            "{f}: only a file under the robot file's own folder can be copied"
        ));
    }
    Ok(files)
}

/// A glTF's buffer and image files: the `uri`s that are not `data:`. Nothing for anything else.
fn uris(bytes: &[u8]) -> Vec<String> {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return Vec::new();
    };
    (["buffers", "images"].iter())
        .flat_map(|k| v[k].as_array().into_iter().flatten())
        .filter_map(|x| x["uri"].as_str())
        .filter(|u| !u.starts_with("data:"))
        .map(str::to_owned)
        .collect()
}
