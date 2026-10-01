//! `es scene export` (plan G, packet G2; `docs/design/scene-authoring.md` section 3.3).

use std::path::Path;

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es scene export <scene.xml|urdf|esscene> --mjcf <out.xml>

Writes the scene as one MJCF file a person opens in MuJoCo's viewer: bodies, joints, geoms,
sites, cameras, lights, materials and textures, meshes, tendons, actuators, sensors, contact
pairs and excludes, gravcomp and <option> -- everything the scene description carries, so that
reading the export back gives the same scene and the same scene_hash. Mesh and texture files
are written beside the XML at the scene's own relative paths (an absolute or `..` path goes
under mesh/ or texture/ instead); a file already there with other bytes is not overwritten.
What MJCF cannot say (a height field, a glTF-only material channel, a joint on the world) is
an error naming it.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("export") => export(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es scene: unknown subcommand '{other}'\n\n{HELP}"
        ))),
    }
}

fn export(args: &[String]) -> Result<u8, CliError> {
    let (mut scene, mut out) = (None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(0);
            }
            "--mjcf" => out = it.next().cloned(),
            other if scene.is_none() && !other.starts_with("--") => scene = Some(other.to_owned()),
            other => {
                return Err(CliError::Usage(format!(
                    "es scene export: unexpected '{other}'\n\n{HELP}"
                )))
            }
        }
    }
    let (Some(scene_path), Some(out)) = (scene, out) else {
        return Err(CliError::Usage(HELP.to_owned()));
    };
    let scene = crate::cmd::backend::load_scene(&scene_path)?;
    let out = Path::new(&out);
    let dir = out
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let fail = |e: &dyn std::fmt::Display| CliError::Runtime(format!("{}: {e}", out.display()));
    std::fs::create_dir_all(dir).map_err(|e| fail(&e))?;
    let xml = es_assets::mjcf::write_mjcf(&scene, dir).map_err(|e| fail(&e))?;
    std::fs::write(out, xml).map_err(|e| fail(&e))?;
    println!(
        "wrote {} ({} bodies, {} joints, {} actuators, {} assets)",
        out.display(),
        scene.bodies.len(),
        scene.joints.len(),
        scene.actuators.len(),
        scene.assets.len()
    );
    println!("scene_hash: {}", hex(&scene.scene_hash()));
    Ok(0)
}
