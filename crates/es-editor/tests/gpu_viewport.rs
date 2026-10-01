//! Packet M16/H9's oracle: the editor's in-process viewport draws a shot as the references do.
//!
//! One shot of `tests/fixtures/mjcf/textured/textured.xml` through `es_editor::gpu::draw`, the
//! function the viewport's worker runs, in each look, against:
//!
//! * fast (`Rs` `Lambert`): `es_render::cpu::rasterize`, bit for bit
//!   (`docs/design/renderer.md` section 15.4);
//! * material (`Rs` `Full`): the CPU reference under section 9.3's edge rule (at most 0.1 % of
//!   pixels may differ; 0 measured there on this fixture);
//! * realistic (`Pt`, after the slot drew another camera first, so its history had to start
//!   over): the frames `es render --path pt --accumulate K` draws - a fresh `Renderer`, one
//!   upload, `K` renders - bit for bit, and the CPU reference's `K` accumulated frames within
//!   section 5.1's tolerance on the tone-mapped bytes.
//!
//! GPU tests print `SKIP` without a device or `slangc`.

use std::path::Path;

use es_editor::gpu::{config, draw, Frame, Slot};
use es_editor::model::scene_view::ScenePreview;
use es_editor::model::viewport::{Mode, Shot, Source, PT_SPP};
use es_gpu::{Gpu, GpuOptions, SlangCompiler};
use es_render::raster::Camera;
use es_render::{cpu, Channel, Renderer};

const K: u32 = 3;

fn gpu() -> Option<Gpu> {
    if let Err(e) = SlangCompiler::new() {
        eprintln!("SKIP: no slangc ({e})");
        return None;
    }
    match Gpu::open(GpuOptions::default()) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            None
        }
    }
}

fn camera(width: u32, height: u32) -> Camera {
    Camera {
        eye: [0.32, -0.3, 0.3],
        look_at: [0.0, 0.0, 0.04],
        fov_y: 0.9,
        width,
        height,
    }
}

fn owed<'g>(gpu: &'g Gpu, slot: &mut Slot<'g>, mode: Mode, shot: &Shot, source: &Source) -> Frame {
    draw(gpu, slot, mode, shot, source)
        .expect("draw")
        .expect("a frame is owed")
}

#[test]
fn the_in_process_viewport_draws_what_the_references_draw() {
    let Some(gpu) = gpu() else {
        return;
    };
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/mjcf/textured/textured.xml"
    ));
    let preview = ScenePreview::open(path).expect("the fixture");
    let (tris, source) = (preview.tris(), preview.source());
    let shot = preview.shot(&camera(64, 48));
    let view = shot.camera.view().expect("view");
    let mut slot = Slot::default();

    let fast = owed(&gpu, &mut slot, Mode::Fast, &shot, &source);
    let want = cpu::rasterize(&tris, &view, &config(Mode::Fast, 64, 48), 0);
    let want = want.tile(Channel::Rgb8).expect("rgb8").to_bytes();
    assert_eq!(fast.picture.rgb, want, "fast: bit for bit");
    assert!(
        draw(&gpu, &mut slot, Mode::Fast, &shot, &source)
            .expect("draw")
            .is_none(),
        "a drawn shot owes nothing"
    );

    let material = owed(&gpu, &mut slot, Mode::Material, &shot, &source);
    let want = cpu::rasterize(&tris, &view, &config(Mode::Material, 64, 48), 0);
    let want = want.tile(Channel::Rgb8).expect("rgb8").to_bytes();
    assert_ne!(material.picture.rgb, fast.picture.rgb, "the look moved");
    let differ = (material.picture.rgb.chunks(3).zip(want.chunks(3)))
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "material: {differ} of {} pixels differ from the CPU",
        64 * 48
    );
    assert!(differ * 1000 <= 64 * 48, "{differ} pixels");

    // Dirty the path tracer's history with another camera, then come back: what the viewport
    // shows must not depend on where the camera was before.
    let away = preview.shot(&camera(64, 48).orbit(0.7, 0.1));
    owed(&gpu, &mut slot, Mode::Realistic, &away, &source);
    owed(&gpu, &mut slot, Mode::Realistic, &away, &source);
    let mut ours = Vec::new();
    for k in 1..=K {
        let f = owed(&gpu, &mut slot, Mode::Realistic, &shot, &source);
        assert_eq!(f.picture.samples, k * PT_SPP);
        ours.push(f.picture.rgb);
    }

    // `es render --path pt --accumulate K`, step for step.
    let cfg = config(Mode::Realistic, 64, 48);
    let mut fresh = Renderer::new(&gpu, cfg.clone()).expect("renderer");
    fresh.upload_tris(tris.clone()).expect("upload");
    let mut history = cpu::History::default();
    for (k, ours) in ours.iter().enumerate() {
        let theirs = (fresh.render(&[view]).expect("render"))
            .read_tile(0, Channel::Rgb8)
            .expect("readback")
            .to_bytes();
        assert_eq!(*ours, theirs, "frame {k}: es render's bytes");
        let reference = cpu::path_trace_accum(&tris, &view, &cfg, 0, &mut history);
        let reference = reference.tile(Channel::Rgb8).expect("rgb8").to_bytes();
        let off = (ours.iter().zip(&reference))
            .filter(|(a, b)| a.abs_diff(**b) > 1)
            .count();
        println!(
            "pt frame {k}: {off} of {} bytes off the CPU by > 1",
            ours.len()
        );
        assert!(off * 1000 <= ours.len(), "{off} bytes");
    }
    assert!(
        (0..K).all(|k| ours[k as usize] != fast.picture.rgb),
        "a path-traced frame"
    );
}

/// Packet M16/H9's frame times: the Shadow Hand (the `shadow-hand-repose` template's scene and
/// viewport camera) at the viewport's largest size, 960x540, orbiting a step per frame, as a
/// drag does - every frame a new shot - and the path tracer accumulating on a still one.
/// `cargo test -p es-editor --release --test gpu_viewport -- --ignored --nocapture`.
#[test]
#[ignore = "a measurement, not a check"]
fn shadow_hand_frame_times_at_960x540() {
    let Some(gpu) = gpu() else {
        return;
    };
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.xml"
    ));
    let preview = ScenePreview::open(path).expect("the hand");
    let source = preview.source();
    let start = Camera {
        eye: [1.5, 0.4, 0.6],
        look_at: [1.02, 0.95, 0.2],
        fov_y: std::f64::consts::FRAC_PI_4,
        width: 960,
        height: 540,
    };
    let tris = preview.tris();
    let t = std::time::Instant::now();
    let (geo, bvh, mat) = (
        tris.to_floats().len(),
        es_render::bvh::Bvh::build(&tris.tris).to_floats().len(),
        tris.materials.to_floats().len(),
    );
    println!(
        "{} triangles; floats: {geo} triangles, {bvh} tree, {mat} materials; packed in {:.1} ms",
        tris.tris.len(),
        t.elapsed().as_secs_f64() * 1e3
    );
    for mode in Mode::ALL {
        let mut slot = Slot::default();
        // Warm: the renderer, the kernels and the scene upload, once per viewport.
        let t = std::time::Instant::now();
        owed(&gpu, &mut slot, mode, &preview.shot(&start), &source);
        let first = t.elapsed().as_secs_f64() * 1e3;
        let n = 60;
        let t = std::time::Instant::now();
        for i in 1..=n {
            let shot = preview.shot(&start.orbit(0.01 * f64::from(i), 0.0));
            owed(&gpu, &mut slot, mode, &shot, &source);
        }
        let moving = t.elapsed().as_secs_f64() * 1e3 / f64::from(n);
        println!(
            "{mode:?} orbiting: {moving:.1} ms/frame ({:.0} fps), first frame {first:.0} ms",
            1e3 / moving
        );
    }
    // A replay playing: every frame a new tick, so the scene is tessellated (through the
    // slot's `SceneCache`) and uploaded again, textures included.
    let mut slot = Slot::default();
    owed(
        &gpu,
        &mut slot,
        Mode::Material,
        &preview.shot(&start),
        &source,
    );
    let t = std::time::Instant::now();
    for tick in 1..=30 {
        let mut shot = preview.shot(&start);
        shot.tick = tick;
        owed(&gpu, &mut slot, Mode::Material, &shot, &source);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / 30.0;
    println!(
        "Material replaying (upload per frame): {ms:.1} ms/frame ({:.0} fps)",
        1e3 / ms
    );
    // H8's fast look: the CPU raster, projected and drawn per camera change.
    let t = std::time::Instant::now();
    for i in 1..=20 {
        let cam = start.orbit(0.01 * f64::from(i), 0.0);
        let _ = es_render::raster::Raster::draw(&preview.project(&cam), 960, 540);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
    println!(
        "CPU raster orbiting: {ms:.1} ms/frame ({:.0} fps)",
        1e3 / ms
    );
    let mut slot = Slot::default();
    let shot = preview.shot(&start);
    owed(&gpu, &mut slot, Mode::Realistic, &shot, &source);
    let t = std::time::Instant::now();
    let mut frames = 1;
    while let Some(f) = draw(&gpu, &mut slot, Mode::Realistic, &shot, &source).expect("draw") {
        frames = f.picture.samples / PT_SPP;
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "Realistic still: {frames} frames ({} spp) in {ms:.0} ms",
        frames * PT_SPP
    );
}
