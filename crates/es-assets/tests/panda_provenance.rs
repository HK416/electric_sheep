//! The Franka Emika Panda is the OBJ half of the mesh readers' evidence (packet M10/W2a).
//!
//! `robotstudio_so101` is 19 binary STL files; `franka_emika_panda` is 8 STL collision meshes
//! and 59 OBJ visual meshes split per material (link5 has no STL: its collision shape is three
//! `link5_collision_*.obj`), so between them the two robots exercise both readers on real files
//! rather than on hand-written fixtures.
//!
//! **Readers and hashes only.** Nothing here loads the Panda into `MuJoCo`: `panda.xml`
//! carries a `<tendon><fixed>` coupling the two finger joints and an `<equality><joint>`
//! mirroring them, and `scene_to_mjcf` refuses a tendon by name. Making the Panda simulate is
//! not this packet's question.
//!
//! Nothing is vendored: `tests/fixtures/mjcf/panda.PROVENANCE.json` pins a commit and the
//! blake3 of each file, and the files themselves are fetched into `target/menagerie/<commit>/`.
//! A blake3 mismatch is a failure, never a skip — it means upstream moved under the pin.
//!
//!     cargo test -p es-assets --test panda_provenance -- --ignored --nocapture

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

const MANIFEST: &str = "panda.PROVENANCE.json";
const DIR: &str = "franka_emika_panda";

fn read_manifest() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/mjcf")
        .join(MANIFEST);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The one string field `name` holds. No JSON dependency: this crate has none and the packet
/// adds none; the manifest is repo-owned and machine-written, so a flat scan is enough.
fn field(json: &str, name: &str) -> String {
    let after = json
        .split_once(&format!("\"{name}\""))
        .unwrap_or_else(|| panic!("{MANIFEST}: no field \"{name}\""))
        .1;
    let after = after.split_once(':').expect("a field has a value").1;
    let start = after.find('"').expect("a string value");
    let rest = &after[start + 1..];
    rest[..rest.find('"').expect("a terminated string")].to_owned()
}

/// `blake3` of the manifest: repository-relative file name to hex.
fn pins(json: &str) -> BTreeMap<String, String> {
    let fail = |what: &str| -> ! { panic!("{MANIFEST}: \"blake3\" {what}") };
    let Some((_, rest)) = json.split_once("\"blake3\"") else {
        fail("is missing")
    };
    let Some((_, rest)) = rest.split_once('{') else {
        fail("is not an object")
    };
    let Some((block, _)) = rest.split_once('}') else {
        fail("is unterminated")
    };
    let unquote = |s: &str| s.trim().trim_matches('"').to_owned();
    block
        .split(',')
        .filter_map(|entry| entry.split_once(':'))
        .map(|(name, hex)| (unquote(name), unquote(hex)))
        .collect()
}

/// Upstream bytes of `repo_path` at `commit`, cached under `target/menagerie/<commit>/`.
fn fetch(commit: &str, repo_path: &str) -> Result<Vec<u8>, String> {
    let name = repo_path.rsplit('/').next().unwrap_or(repo_path);
    if let Ok(root) = std::env::var("ES_MENAGERIE_CACHE") {
        let cached = PathBuf::from(root).join(commit).join(repo_path);
        if cached.is_file() {
            return std::fs::read(&cached).map_err(|e| format!("{}: {e}", cached.display()));
        }
    }
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/menagerie")
        .join(commit)
        .join(repo_path);
    let out_dir = out.parent().expect("a file has a parent").to_owned();
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    if out.is_file() {
        return std::fs::read(&out).map_err(|e| format!("{}: {e}", out.display()));
    }
    let url = format!(
        "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/{repo_path}"
    );
    let tmp = out_dir.join(format!("{name}.{:?}.part", std::thread::current().id()));
    let status = Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(&tmp)
        .arg(&url)
        .status()
        .map_err(|e| format!("no cache (set ES_MENAGERIE_CACHE) and `curl` failed: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("`curl {url}` exited with {status}"));
    }
    let bytes = std::fs::read(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let _ = std::fs::rename(&tmp, &out);
    let _ = std::fs::remove_file(&tmp);
    Ok(bytes)
}

/// Every pinned Panda file hashes to what the manifest says, and every mesh among them decodes.
///
/// `#[ignore]`: 68 files over the network.
#[test]
#[ignore = "fetches panda.xml and 67 mesh files from GitHub"]
fn panda_meshes_load_and_hash() {
    let json = read_manifest();
    assert_eq!(field(&json, "license"), "Apache-2.0");
    assert_eq!(field(&json, "path"), DIR);
    let commit = field(&json, "commit");
    let pins = pins(&json);
    assert_eq!(pins.len(), 68, "the manifest pins {} files", pins.len());

    let mut computed: Vec<(String, String)> = Vec::new();
    let mut moved: Vec<String> = Vec::new();
    let (mut stl, mut obj, mut triangles) = (0usize, 0usize, 0usize);
    for (name, expected) in &pins {
        let extension = Path::new(name).extension().and_then(OsStr::to_str);
        let repo_path = if extension == Some("xml") {
            format!("{DIR}/{name}")
        } else {
            format!("{DIR}/assets/{name}")
        };
        let bytes = match fetch(&commit, &repo_path) {
            Ok(bytes) => bytes,
            Err(why) => {
                println!("SKIP panda_meshes_load_and_hash: {why}");
                return;
            }
        };
        let got = blake3::hash(&bytes).to_hex().to_string();
        if &got != expected {
            moved.push(name.clone());
        }
        computed.push((name.clone(), got));

        let decoded = if extension == Some("stl") {
            stl += 1;
            Some(es_assets::stl::parse(&bytes))
        } else if extension == Some("obj") {
            obj += 1;
            let text = String::from_utf8(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
            Some(es_assets::obj::parse(&text))
        } else {
            None
        };
        if let Some(decoded) = decoded {
            let (positions, indices) =
                decoded.unwrap_or_else(|e| panic!("{repo_path} does not decode: {e}"));
            assert!(!positions.is_empty() && indices.len() % 3 == 0, "{name}");
            assert!(
                indices.iter().all(|i| (*i as usize) < positions.len()),
                "{name}: an index points past the vertex buffer"
            );
            triangles += indices.len() / 3;
        }
    }
    println!("RAN panda_meshes_load_and_hash: {stl} STL + {obj} OBJ, {triangles} triangles");
    // The block the manifest holds, so regenerating it is a copy rather than a transcription.
    for (name, hex) in &computed {
        println!("    \"{name}\": \"{hex}\",");
    }
    // Measured, not read off a page: the Panda's collision meshes are eight STL files -- link5
    // has none, its collision shape is three `link5_collision_*.obj` among the visual meshes.
    assert_eq!((stl, obj), (8, 59), "the pinned file list changed shape");
    assert!(
        moved.is_empty(),
        "at {commit} these files hash differently from the manifest -- upstream moved under \
         the pin, or a download is corrupt: {moved:?}"
    );
}
