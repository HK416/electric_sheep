//! `es render` -- one frame of a free camera through `es-render` (packet M16/H8,
//! `docs/design/editor-redesign.md` section 5, S4).
//!
//! The editor's viewport shows the textured and the path-traced scene by asking this verb for
//! the picture, so the editor itself never opens a Vulkan device. The camera is
//! `es video showcase`'s free camera (`es_env::render::look_at`, the arithmetic
//! `es_render::raster::Camera::view` repeats), the scene is loaded and posed the way the
//! showcase loads and poses it, and the config is `es_env::render::config` with the showcase's
//! own three knobs (`shading`, `exposure`, `temporal`). Nothing here is a second renderer.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use es_env::render::look_at;
use es_env::traj::Trajectory;
use es_render::{Channel, RenderPath, Renderer, Shading, Temporal, TriScene};

use crate::error::CliError;
use crate::showcase::vec3;

pub const HELP: &str = "\
es render --scene <file.xml|urdf> [--traj <file.estraj> --tick N]
          --eye X,Y,Z --look-at X,Y,Z [--fov 45] [--width 640] [--height 400]
          [--path rs|full|pt] [--spp 4] [--bounces 3] [--accumulate K] [--exposure 1]
          --out <file.ppm | ->

Renders one picture of the scene from a camera that is in no scene and no document, as
`es video showcase` does, and writes it as a binary PPM (P6) whose comment line says how many
samples per pixel it holds. The editor's viewport shows its textured and path-traced modes
through this command, so the editor never opens a Vulkan device itself.

    --traj/--tick   pose the scene at row N of a recorded `.estraj` (default: the scene's own
                    initial pose)
    --path NAME     `rs` (default): the rasterizer's Lambert with textures; `full`: shadows,
                    sky, PBR highlights and 2x supersampling (M7/R2, HT1); `pt`: the path
                    tracer with next-event estimation (M7/R3)
    --spp N         samples per pixel per frame on `pt` (default 4)
    --bounces B     bounces per sample on `pt` (default 3)
    --accumulate K  `pt` only: K frames of a still camera accumulated (M7/R4), so the last
                    holds K x N samples (default 1)
    --exposure E    linear multiplier before the tone map on `pt` (default 1.0)
    --out FILE      the last frame; `-` writes every accumulated frame to stdout, one PPM
                    after another, as each is done

Needs the `render` feature and a Vulkan device. Exit codes: 0 success, 1 runtime failure,
2 usage error.
";

fn usage(msg: impl std::fmt::Display) -> CliError {
    CliError::Usage(format!("{msg}\n\n{HELP}"))
}

/// A binary PPM: the header, the sample count as its comment, then `rgb` row-major.
pub fn ppm(width: u32, height: u32, samples: u32, rgb: &[u8]) -> Vec<u8> {
    let mut out = format!("P6\n# samples {samples}\n{width} {height}\n255\n").into_bytes();
    out.extend_from_slice(rgb);
    out
}

pub fn run(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let (mut scene, mut traj, mut tick, mut out) = (None, None, None, None);
    let (mut eye, mut target, mut fov_deg) = (None, None, 45.0f64);
    let (mut width, mut height) = (640u32, 400u32);
    let (mut look, mut pt) = (Shading::Lambert, false);
    let (mut spp, mut bounces, mut frames, mut exposure) = (4u32, 3u32, 1u32, 1.0f32);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let val = it
            .next()
            .ok_or_else(|| usage(format!("{a}: missing value")))?;
        let num = |flag: &str| {
            val.parse::<f64>()
                .map_err(|_| usage(format!("{flag}: {val:?} is not a number")))
        };
        match a.as_str() {
            "--scene" => scene = Some(val.clone()),
            "--traj" => traj = Some(PathBuf::from(val)),
            "--tick" => tick = Some(num(a)? as usize),
            "--out" => out = Some(val.clone()),
            "--eye" => eye = Some(vec3(val, "--eye")?),
            "--look-at" => target = Some(vec3(val, "--look-at")?),
            "--fov" => fov_deg = num(a)?,
            "--width" => width = num(a)? as u32,
            "--height" => height = num(a)? as u32,
            "--spp" => spp = num(a)? as u32,
            "--bounces" => bounces = num(a)? as u32,
            "--accumulate" => frames = num(a)? as u32,
            "--exposure" => exposure = num(a)? as f32,
            "--path" => {
                (look, pt) = match val.as_str() {
                    "rs" => (Shading::Lambert, false),
                    "full" => (Shading::FULL, false),
                    "pt" => (Shading::Lambert, true),
                    other => {
                        return Err(usage(format!(
                            "--path: expected rs, full or pt, got {other:?}"
                        )))
                    }
                }
            }
            other => return Err(usage(format!("unknown flag '{other}'"))),
        }
    }
    let Some(scene) = scene else {
        return Err(usage("--scene is required"));
    };
    let (Some(eye), Some(target)) = (eye, target) else {
        return Err(usage("--eye X,Y,Z and --look-at X,Y,Z are required"));
    };
    let Some(out) = out else {
        return Err(usage("--out <file.ppm | -> is required"));
    };
    if width == 0 || height == 0 || spp == 0 || bounces == 0 || frames == 0 {
        return Err(usage(
            "--width, --height, --spp, --bounces and --accumulate must be greater than zero",
        ));
    }
    if frames > 1 && !pt {
        return Err(usage(
            "--accumulate is a `pt` flag: the rasterizer has no samples to accumulate",
        ));
    }
    if tick.is_some() != traj.is_some() {
        return Err(usage("--traj and --tick go together"));
    }
    let poses = match (&traj, tick) {
        (Some(path), Some(tick)) => {
            let traj = Trajectory::read(path)
                .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))?;
            if tick >= traj.ticks() {
                return Err(usage(format!(
                    "--tick {tick}: {} has {} tick(s)",
                    path.display(),
                    traj.ticks()
                )));
            }
            traj.poses(tick)
        }
        _ => BTreeMap::new(),
    };
    let rt = |m: String| CliError::Runtime(m);
    let desc = crate::backend::load_scene(&scene)?;
    let tris = TriScene::from_scene_with_poses(&desc, &poses)
        .map_err(|e| rt(format!("tessellation: {e}")))?;
    let view =
        look_at(eye, target, fov_deg.to_radians(), width, height).map_err(|e| rt(e.to_string()))?;
    // NEE on, ReSTIR and SVGF off, as `es video showcase --path pt` (renderer note 10.5).
    let path = if pt {
        RenderPath::Pt {
            spp,
            bounces,
            nee: true,
            restir: false,
            svgf: false,
        }
    } else {
        RenderPath::Rs
    };
    let mut cfg = es_env::render::config(width, height, Channel::Rgb8, path);
    cfg.shading = look;
    cfg.exposure = exposure;
    // `max_history = K`: below the cap the sum is exact, so K frames are one frame of K x spp
    // samples (renderer note 11.1).
    cfg.temporal = pt.then_some(Temporal {
        max_history: frames,
    });

    let gpu = es_gpu::Gpu::open(es_gpu::GpuOptions::default())
        .map_err(|e| rt(format!("no Vulkan device for es render: {e}")))?;
    let mut renderer = Renderer::new(&gpu, cfg).map_err(|e| rt(format!("renderer: {e}")))?;
    renderer
        .upload_tris(tris)
        .map_err(|e| rt(format!("scene upload: {e}")))?;
    let per_frame = if pt { spp } else { 1 };
    let mut last = Vec::new();
    for f in 1..=frames {
        let mut atlas = renderer
            .render(&[view])
            .map_err(|e| rt(format!("render: {e}")))?;
        let rgb = atlas
            .read_tile(0, Channel::Rgb8)
            .map_err(|e| rt(format!("readback: {e}")))?
            .to_bytes();
        last = ppm(width, height, f * per_frame, &rgb);
        if out == "-" {
            // A reader that went away (the editor moved its camera) ends the render.
            let mut stdout = std::io::stdout().lock();
            stdout
                .write_all(&last)
                .and_then(|()| stdout.flush())
                .map_err(|e| rt(format!("stdout: {e}")))?;
        }
    }
    if out != "-" {
        std::fs::write(&out, &last).map_err(|e| rt(format!("{out}: {e}")))?;
    }
    Ok(0)
}
