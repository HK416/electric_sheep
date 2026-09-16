//! `es video showcase` -- the human-facing render of a finished run (packet M5/V9, design
//! note `docs/design/visible-learning.md` section 7.17; spec 15.2, 25.1).
//!
//! `es video mosaic` tiles the frames the *policy* saw -- 96x96 observation tiles from the one
//! camera the Observation IR declares -- and nobody can see an arm pick a cube in those. This
//! renders the same run again, from the `.estraj` state trajectory `es eval run` and
//! `es loop collect` now always write (`es_env::traj`), through a camera that is in no scene
//! and in no document: `--eye`, `--look-at`, `--fov` at any resolution.
//!
//! Replay rather than a second camera in the loop, because the trajectory is worth having
//! anyway (it is what "what the robot did" means) and because a finished run can then be
//! re-rendered from any angle, at any resolution, for the expert and for every checkpoint,
//! without re-running the physics or the policy. The frames it writes are the same raw
//! `NNNNNN.bin` + `layout.json` pairs `python/es/encode_video.py` already encodes.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use es_env::render::{camera_view, look_at};
use es_env::traj::Trajectory;
use es_env::EnvRendererCfg;
use es_render::{Channel, RenderPath, Renderer, SceneCache};

use crate::error::CliError;

pub const HELP: &str = "\
es video showcase --run <dir> --scene <file.xml|urdf> --out <dir>
                  (--eye X,Y,Z --look-at X,Y,Z [--fov 45] | --camera NAME)
                  [--width 1280] [--height 720] [--cell NAME]... [--stride N]

Re-renders a finished `es eval run` or `es loop collect` from the per-episode `.estraj` state
trajectories it wrote, through a camera that is not in the scene and not in any IR -- so the
result is watchable rather than the 96x96 tiles the policy reads. <dir> is the run directory
(its `traj/` subdirectory) or a directory of `.estraj` files directly.

Every selected episode is rendered back to back into one <out>/NNNNNN.bin sequence plus one
<out>/layout.json, which `python/es/encode_video.py` (or ffmpeg over the raw frames) encodes.
Nothing is resampled and no observation is involved: the showcase camera has its own
`ImageSpec`, computed from --fov at --width x --height.

    --eye/--look-at a camera that is in no scene and no document, in world metres
    --camera NAME   a camera the scene declares, instead: with the run's own observation
                    --width/--height this reproduces the recorded observation frames bit for
                    bit, which is how a replay is checked against the run it replays
    --cell NAME     render only this episode; repeatable, default every one in name order
    --stride N      render every Nth recorded tick (default 1)
    --fov D         vertical field of view in degrees (default 45)

Needs the `render` feature and a Vulkan device. Exit codes: 0 success, 1 runtime failure,
2 usage error.
";

struct Opts {
    run: PathBuf,
    scene: String,
    out: PathBuf,
    /// A free camera (`--eye` / `--look-at`), or a camera the scene declares (`--camera`).
    camera: Camera,
    width: u32,
    height: u32,
    cells: Vec<String>,
    stride: usize,
}

enum Camera {
    Free {
        eye: [f64; 3],
        target: [f64; 3],
        fov_deg: f64,
    },
    Scene(String),
}

fn usage(msg: impl std::fmt::Display) -> CliError {
    CliError::Usage(format!("{msg}\n\n{HELP}"))
}

fn vec3(raw: &str, flag: &str) -> Result<[f64; 3], CliError> {
    let parts: Vec<&str> = raw.split(',').collect();
    let [x, y, z] = parts.as_slice() else {
        return Err(usage(format!("{flag}: expected X,Y,Z, got {raw:?}")));
    };
    let one = |s: &str| {
        s.trim()
            .parse::<f64>()
            .map_err(|_| usage(format!("{flag}: {s:?} is not a number")))
    };
    Ok([one(x)?, one(y)?, one(z)?])
}

pub fn run(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let (mut run, mut scene, mut out, mut eye, mut target) = (None, None, None, None, None);
    let (mut fov_deg, mut width, mut height, mut stride) = (45.0f64, 1280u32, 720u32, 1usize);
    let (mut cells, mut named) = (Vec::new(), None);

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| usage(format!("{a}: missing value")))
        };
        let num = |v: &str, flag: &str| {
            v.parse::<f64>()
                .map_err(|_| usage(format!("{flag}: {v:?} is not a number")))
        };
        match a.as_str() {
            "--run" => run = Some(PathBuf::from(val()?)),
            "--scene" => scene = Some(val()?.clone()),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--eye" => eye = Some(vec3(val()?, "--eye")?),
            "--look-at" => target = Some(vec3(val()?, "--look-at")?),
            "--fov" => fov_deg = num(val()?, "--fov")?,
            "--width" => width = num(val()?, "--width")? as u32,
            "--height" => height = num(val()?, "--height")? as u32,
            "--stride" => stride = (num(val()?, "--stride")? as usize).max(1),
            "--camera" => named = Some(val()?.clone()),
            "--cell" => cells.push(val()?.clone()),
            other => return Err(usage(format!("unknown flag '{other}'"))),
        }
    }
    let (Some(run), Some(scene), Some(out)) = (run, scene, out) else {
        return Err(usage("--run, --scene and --out are all required"));
    };
    let camera = match (named, eye, target) {
        (Some(name), None, None) => Camera::Scene(name),
        (None, Some(eye), Some(target)) => Camera::Free {
            eye,
            target,
            fov_deg,
        },
        (None, ..) => {
            return Err(usage(
                "a camera is required: --eye X,Y,Z --look-at X,Y,Z, or --camera NAME",
            ))
        }
        (Some(_), ..) => {
            return Err(usage(
                "--camera names a camera the scene already places; it does not take --eye or                  --look-at",
            ))
        }
    };
    if width == 0 || height == 0 {
        return Err(usage("--width and --height must both be greater than zero"));
    }
    render(&Opts {
        run,
        scene,
        out,
        camera,
        width,
        height,
        cells,
        stride,
    })
}

/// The `.estraj` files to render, in the order they will be concatenated.
fn select(opts: &Opts) -> Result<Vec<(String, PathBuf)>, CliError> {
    let rt = |m: String| CliError::Runtime(m);
    let dir = if opts.run.join("traj").is_dir() {
        opts.run.join("traj")
    } else {
        opts.run.clone()
    };
    // A `BTreeSet`, never a `HashMap`: the concatenation order must not depend on how a
    // directory happens to enumerate (spec 3.4).
    let mut found = BTreeSet::new();
    for entry in fs::read_dir(&dir).map_err(|e| rt(format!("{}: {e}", dir.display())))? {
        let path = entry
            .map_err(|e| rt(format!("{}: {e}", dir.display())))?
            .path();
        if path.extension().is_some_and(|e| e == "estraj") {
            let stem = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            found.insert((stem, path));
        }
    }
    if opts.cells.is_empty() {
        if found.is_empty() {
            return Err(rt(format!(
                "no *.estraj trajectories under {}",
                dir.display()
            )));
        }
        return Ok(found.into_iter().collect());
    }
    opts.cells
        .iter()
        .map(|want| {
            found
                .iter()
                .find(|(name, _)| name == want)
                .cloned()
                .ok_or_else(|| rt(format!("no episode {want:?} under {}", dir.display())))
        })
        .collect()
}

fn render(opts: &Opts) -> Result<u8, CliError> {
    let rt = |m: String| CliError::Runtime(m);
    let files = select(opts)?;
    let scene = crate::cmd::backend::load_scene(&opts.scene)?;

    // A free camera is fixed for the whole run; a scene camera may be bolted to a moving
    // body (a wrist camera), so it is resolved per tick from that tick's own poses.
    let mut scene_cfg = None;
    let mut fixed = None;
    match &opts.camera {
        Camera::Free {
            eye,
            target,
            fov_deg,
        } => {
            fixed = Some(
                look_at(*eye, *target, fov_deg.to_radians(), opts.width, opts.height)
                    .map_err(|e| rt(e.to_string()))?,
            );
        }
        Camera::Scene(name) => {
            let cam = scene
                .cameras
                .iter()
                .find(|c| &c.name == name)
                .ok_or_else(|| {
                    rt(format!(
                        "camera {name:?} is not in scene {:?}; it declares {:?}",
                        scene.name,
                        scene.cameras.iter().map(|c| &c.name).collect::<Vec<_>>()
                    ))
                })?;
            scene_cfg = Some(EnvRendererCfg::rgb(cam.id, opts.width, opts.height));
        }
    }

    let gpu = es_gpu::Gpu::open(es_gpu::GpuOptions::default())
        .map_err(|e| rt(format!("no Vulkan device for es video showcase: {e}")))?;
    // The same function every other `Rs` render in this repository goes through, so the
    // showcase and the observation frames cannot drift apart (design note section 7.4).
    let cfg = es_env::render::config(opts.width, opts.height, Channel::Rgb8, RenderPath::Rs);
    let mut renderer = Renderer::new(&gpu, cfg).map_err(|e| rt(format!("renderer: {e}")))?;

    fs::create_dir_all(&opts.out).map_err(|e| rt(format!("{}: {e}", opts.out.display())))?;
    let layout = opts.out.join("layout.json");
    fs::write(
        &layout,
        format!(
            "{{\"dtype\":\"u8\",\"shape\":[{},{},3]}}\n",
            opts.height, opts.width
        ),
    )
    .map_err(|e| rt(format!("{}: {e}", layout.display())))?;

    let start = Instant::now();
    let mut frame = 0u64;
    // The local tessellation of every geom, computed once and re-posed per tick. Bit-identical
    // to tessellating per tick (`docs/design/renderer.md` section 8.2), which is what makes a
    // re-render of a committed run reproduce it (packet M5/V9's bit-identity oracle).
    let mut cache = SceneCache::default();
    for (name, path) in &files {
        let traj = Trajectory::read(path).map_err(|e| rt(e.to_string()))?;
        let ticks = traj.ticks();
        for tick in (0..ticks).step_by(opts.stride) {
            let poses = traj.poses(tick);
            let view = match (&fixed, &scene_cfg) {
                (Some(v), _) => *v,
                (_, Some(cfg)) => {
                    camera_view(&scene, cfg, &poses).map_err(|e| rt(e.to_string()))?
                }
                _ => unreachable!("one of the two camera forms was built above"),
            };
            let tri = cache
                .tri_scene(&scene, &poses)
                .map_err(|e| rt(format!("tessellation: {e}")))?;
            renderer
                .upload_tris(tri)
                .map_err(|e| rt(format!("scene upload: {e}")))?;
            let mut atlas = renderer
                .render(&[view])
                .map_err(|e| rt(format!("render: {e}")))?;
            let tile = atlas
                .read_tile(0, Channel::Rgb8)
                .map_err(|e| rt(format!("readback: {e}")))?;
            let out = opts.out.join(format!("{frame:06}.bin"));
            fs::write(&out, tile.to_bytes()).map_err(|e| rt(format!("{}: {e}", out.display())))?;
            frame += 1;
        }
        println!("{name}: {ticks} tick(s)");
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "wrote {frame} frame(s) of {}x{} to {} in {secs:.1}s ({:.1} ms/frame)",
        opts.width,
        opts.height,
        opts.out.display(),
        if frame == 0 {
            0.0
        } else {
            secs * 1000.0 / frame as f64
        }
    );
    Ok(0)
}
