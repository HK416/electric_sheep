//! The viewport on the graphics card, in process (packet M16/H9).
//!
//! The owner's decision of 2026-10-01 (`docs/design/editor-redesign.md` section 5, S4): the
//! editor draws its viewport with `es-render` itself, every frame the shot changes, so the
//! material look stays on while the camera moves. One worker thread owns the one `es-gpu`
//! device, opened on the first 3D view; every viewport ([`Look`]) asks it for the shot on
//! screen and shows the newest frame it sent back - the UI thread never waits on the device.
//! Per viewport the worker keeps one `Renderer` (rebuilt only when the size or the path
//! changes) and one `SceneCache`, and uploads the triangles only when the scene or its tick
//! moves; orbiting re-renders and reads back, nothing more.
//!
//! What to draw next is [`viewport::next`]'s, what the line says [`viewport::gpu_status`]'s:
//! the decisions are `es-editor-model`'s, the device is this crate's (spec 4.2). A machine
//! with no device keeps H8's paths ([`viewport::Viewport`]): [`Look::update`] says `false`
//! and the canvas falls back.
//!
//! The same worker draws the viewport's corner (packet M17/G6, [`Sensor`]): one frame of a scene
//! camera under the render its Task IR channel declares, through `es-env`'s own single-camera
//! steps (`drawn_frame`, the episode's first seed) — the observation the policy is given of the
//! scene at its own pose. It has no fallback without a device; the corner says so.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

use es_assets::scene::SceneDesc;
use es_env::randomize::RenderOverrides;
use es_env::render::{drawn_frame, frame_seed, EnvRendererCfg};
use es_gpu::{Gpu, GpuOptions};
use es_ir::task::SeedStream;
use es_render::{
    Channel, RenderConfig, RenderPath, Renderer, SceneCache, Shading, Temporal, TileAtlasCfg,
};

use crate::model::viewport::{
    self, Mode, Next, Picture, Shot, Source, Status, PT_BOUNCES, PT_EXPOSURE, PT_FRAMES, PT_SPP,
};

/// The device, as every viewport sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Health {
    Opening,
    /// Open: its name, and the device memory its allocator holds.
    Ready {
        device: String,
        bytes: u64,
    },
    /// No device, or one that failed; the reason. Every viewport is on H8's paths from here.
    Failed(String),
}

/// One finished frame and the shot and look it is of.
#[derive(Debug)]
pub struct Frame {
    pub mode: Mode,
    pub shot: Shot,
    pub picture: Picture,
    /// Render and readback on the worker, in milliseconds.
    pub ms: f32,
}

struct Want {
    mode: Mode,
    shot: Shot,
    source: Source,
    reply: Sender<Frame>,
}

enum Msg {
    Want(u64, Want),
    Sensor(u64, SensorWant),
    Forget(u64),
}

/// What a corner's frame is of: the scene's revision and the camera's name.
pub(crate) type SensorKey = (usize, String);

/// A frame for the corner (packet M17/G6): the scene, and the camera and render a Task IR
/// channel declares (`es_env::render::sensor_cfg`'s).
pub(crate) struct SensorJob {
    pub scene: Arc<SceneDesc>,
    pub cfg: EnvRendererCfg,
}

struct SensorWant {
    key: SensorKey,
    job: SensorJob,
    reply: Sender<(SensorKey, Result<Picture, String>)>,
}

struct Worker {
    tx: Mutex<Option<Sender<Msg>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    health: Arc<Mutex<Health>>,
}

static WORKER: OnceLock<Worker> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn worker() -> &'static Worker {
    WORKER.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        let health = Arc::new(Mutex::new(Health::Opening));
        let shared = Arc::clone(&health);
        let thread = std::thread::Builder::new()
            .name("es-editor-gpu".into())
            .spawn(move || run(&rx, &shared))
            .ok();
        if thread.is_none() {
            set(&health, Health::Failed("no worker thread".into()));
        }
        Worker {
            tx: Mutex::new(Some(tx)),
            thread: Mutex::new(thread),
            health,
        }
    })
}

fn set(health: &Mutex<Health>, to: Health) {
    *health.lock().unwrap_or_else(PoisonError::into_inner) = to;
}

/// The device's state, or `None` before any viewport asked for it.
pub fn health() -> Option<Health> {
    let w = WORKER.get()?;
    Some(
        w.health
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone(),
    )
}

/// Closes the device: the worker finishes the frame it is on, drops every renderer and the
/// device, and is joined. Called once the window has closed.
pub fn shutdown() {
    let Some(w) = WORKER.get() else {
        return;
    };
    drop(w.tx.lock().unwrap_or_else(PoisonError::into_inner).take());
    if let Some(thread) = w
        .thread
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
    {
        let _ = thread.join();
    }
}

fn send(msg: Msg) -> bool {
    let w = worker();
    let tx = w.tx.lock().unwrap_or_else(PoisonError::into_inner);
    tx.as_ref().is_some_and(|tx| tx.send(msg).is_ok())
}

/// One viewport's end of the worker: what it asked for last and the newest frame back.
pub(crate) struct Look {
    id: u64,
    tx: Sender<Frame>,
    rx: Receiver<Frame>,
    newest: Option<Frame>,
    /// Moves with every frame received, so the shell re-uploads its texture only then.
    revision: u64,
}

impl Default for Look {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            tx,
            rx,
            newest: None,
            revision: 0,
        }
    }
}

impl Drop for Look {
    fn drop(&mut self) {
        // Its renderer and scene go with it; nothing to do if no device was ever opened.
        if WORKER.get().is_some() {
            send(Msg::Forget(self.id));
        }
    }
}

impl Look {
    /// Called once a frame with the shot on screen: takes whatever frames arrived and asks for
    /// the next one if [`viewport::next`] wants one. `false` when there is no device - the
    /// caller draws with H8's paths instead. Never blocks.
    pub fn update(&mut self, mode: Mode, shot: &Shot, source: impl FnOnce() -> Source) -> bool {
        if matches!(health(), Some(Health::Failed(_))) {
            return false;
        }
        for frame in self.rx.try_iter() {
            self.newest = Some(frame);
            self.revision += 1;
        }
        if self.next(mode, shot) != Next::Idle {
            let want = Want {
                mode,
                shot: shot.clone(),
                source: source(),
                reply: self.tx.clone(),
            };
            if !send(Msg::Want(self.id, want)) {
                return false;
            }
        }
        !matches!(health(), Some(Health::Failed(_)))
    }

    fn next(&self, mode: Mode, shot: &Shot) -> Next {
        let (drawn, frames) = self.newest.as_ref().map_or((false, 0), |f| {
            let drawn = f.mode == mode && f.shot == *shot;
            (drawn, f.picture.samples / PT_SPP)
        });
        viewport::next(mode, drawn, frames)
    }

    /// The newest frame, of this shot or of the one before it (a moving camera shows the last
    /// frame until the next one lands, never the flat raster), and its revision.
    pub fn newest(&self) -> Option<(&Frame, u64)> {
        self.newest.as_ref().map(|f| (f, self.revision))
    }

    pub fn status(&self, mode: Mode, shot: &Shot) -> Option<Status> {
        let picture =
            (self.newest.as_ref()).map(|f| (f.mode == mode && f.shot == *shot, f.picture.samples));
        viewport::gpu_status(mode, picture)
    }

    /// Whether the shell should keep repainting: a frame is owed or on its way.
    pub fn busy(&self, mode: Mode, shot: &Shot) -> bool {
        self.next(mode, shot) != Next::Idle
    }
}

/// One corner's end of the worker (packet M17/G6): it asks for a frame once per key and keeps
/// the newest that came back.
pub(crate) struct Sensor {
    id: u64,
    tx: Sender<(SensorKey, Result<Picture, String>)>,
    rx: Receiver<(SensorKey, Result<Picture, String>)>,
    asked: Option<SensorKey>,
    newest: Option<(SensorKey, Result<Picture, String>)>,
    /// Moves with every frame received, so the corner re-uploads its texture only then.
    revision: u64,
}

impl Default for Sensor {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            tx,
            rx,
            asked: None,
            newest: None,
            revision: 0,
        }
    }
}

impl Drop for Sensor {
    fn drop(&mut self) {
        if WORKER.get().is_some() {
            send(Msg::Forget(self.id));
        }
    }
}

impl Sensor {
    /// Takes what arrived and asks for `key`'s frame if it was not asked for yet (`job` is built
    /// only then). `false` without a device. Never blocks.
    pub fn update(&mut self, key: &SensorKey, job: impl FnOnce() -> Option<SensorJob>) -> bool {
        if matches!(health(), Some(Health::Failed(_))) {
            return false;
        }
        for frame in self.rx.try_iter() {
            self.newest = Some(frame);
            self.revision += 1;
        }
        if self.asked.as_ref() != Some(key) {
            self.asked = Some(key.clone());
            if let Some(job) = job() {
                let want = SensorWant {
                    key: key.clone(),
                    job,
                    reply: self.tx.clone(),
                };
                if !send(Msg::Sensor(self.id, want)) {
                    return false;
                }
            }
        }
        !matches!(health(), Some(Health::Failed(_)))
    }

    /// The newest frame (of this key or the one before), whether it is of `key`, and its revision.
    pub fn newest(&self, key: &SensorKey) -> Option<(&Result<Picture, String>, bool, u64)> {
        let (k, frame) = self.newest.as_ref()?;
        Some((frame, k == key, self.revision))
    }
}

/// The worker's renderer for one corner, and the camera config it was built for.
#[derive(Debug, Default)]
struct SensorSlot<'gpu> {
    renderer: Option<(EnvRendererCfg, Renderer<'gpu>)>,
    cache: SceneCache,
}

/// One frame of `job`'s camera, as `es_env::render::EnvRenderer` draws an episode's first frame
/// of one camera: `drawn_frame` with no draws, the renderer built on its config, the lighting
/// set, the `Tick` stream's seed for tick 0, the scene uploaded, one render, the tile read back.
fn sensor_frame<'gpu>(
    gpu: &'gpu Gpu,
    slot: &mut SensorSlot<'gpu>,
    job: &SensorJob,
) -> Result<Picture, String> {
    let none = RenderOverrides::default();
    let (tri, view, rc) = drawn_frame(
        &job.scene,
        &job.cfg,
        &none,
        &BTreeMap::new(),
        &mut slot.cache,
    )
    .map_err(|e| e.to_string())?;
    if slot.renderer.as_ref().is_none_or(|(c, _)| *c != job.cfg) {
        slot.renderer = None;
        let r = Renderer::new(gpu, rc.clone()).map_err(|e| format!("renderer: {e}"))?;
        slot.renderer = Some((job.cfg.clone(), r));
    }
    let Some((_, r)) = slot.renderer.as_mut() else {
        return Err("renderer".into());
    };
    r.set_lighting(&rc);
    if job.cfg.seed_stream == SeedStream::Tick {
        r.set_seed(frame_seed(SeedStream::Tick, rc.seed, 0));
    }
    r.upload_tris(tri)
        .map_err(|e| format!("scene upload: {e}"))?;
    let rgb = (r.render(&[view]))
        .and_then(|mut atlas| atlas.read_tile(0, job.cfg.channel))
        .map_err(|e| format!("render: {e}"))?
        .to_bytes();
    let samples = match job.cfg.path {
        RenderPath::Pt { spp, .. } => spp,
        RenderPath::Rs => 1,
    };
    Ok(Picture {
        width: job.cfg.width,
        height: job.cfg.height,
        rgb,
        samples,
    })
}

/// The config `es render` builds for the same look (`--path rs|full|pt` with H8's budget), so
/// a frame here is that command's frame for the same shot.
pub fn config(mode: Mode, width: u32, height: u32) -> RenderConfig {
    let atlas = TileAtlasCfg::row(width, height, 1);
    let mut cfg = match mode {
        Mode::Realistic => {
            let mut c = RenderConfig::pt(atlas, PT_SPP, PT_BOUNCES);
            c.path = RenderPath::Pt {
                spp: PT_SPP,
                bounces: PT_BOUNCES,
                nee: true,
                restir: false,
                svgf: false,
            };
            c.exposure = PT_EXPOSURE;
            c.temporal = Some(Temporal {
                max_history: PT_FRAMES,
            });
            c
        }
        Mode::Fast | Mode::Material => RenderConfig::rs(atlas),
    };
    cfg.channels = BTreeSet::from([Channel::Rgb8]);
    cfg.shading = if mode == Mode::Material {
        Shading::FULL
    } else {
        Shading::Lambert
    };
    cfg
}

/// One viewport's state on the worker.
#[derive(Debug, Default)]
pub struct Slot<'gpu> {
    /// The renderer and the `(path tracer, width, height)` it was built for.
    renderer: Option<((bool, u32, u32), Renderer<'gpu>)>,
    cache: SceneCache,
    /// The scene file, motion and tick whose triangles the renderer holds.
    scene: Option<(PathBuf, Option<PathBuf>, usize)>,
    drawn: Option<(Mode, Shot)>,
    frames: u32,
}

/// Draws the next frame `slot` owes for `shot` in `mode`, or `None` when it owes none. The
/// whole in-process render: the test renders a shot through this and through `es render`'s
/// steps and compares.
pub fn draw<'gpu>(
    gpu: &'gpu Gpu,
    slot: &mut Slot<'gpu>,
    mode: Mode,
    shot: &Shot,
    source: &Source,
) -> Result<Option<Frame>, String> {
    let t0 = Instant::now();
    let drawn = (slot.drawn.as_ref()).is_some_and(|(m, s)| *m == mode && s == shot);
    let next = viewport::next(mode, drawn, slot.frames);
    if next == Next::Idle {
        return Ok(None);
    }
    let Ok(view) = shot.camera.view() else {
        // A camera with no orientation draws nothing, as the raster does; not a device error.
        slot.drawn = Some((mode, shot.clone()));
        return Ok(None);
    };
    let (width, height) = (shot.camera.width, shot.camera.height);
    let cfg = config(mode, width, height);
    let key = (mode == Mode::Realistic, width, height);
    if slot.renderer.as_ref().is_none_or(|(k, _)| *k != key) {
        // The old one's buffers go first: two atlases of a big viewport are not free.
        slot.renderer = None;
        slot.scene = None;
        let r = Renderer::new(gpu, cfg.clone()).map_err(|e| format!("renderer: {e}"))?;
        slot.renderer = Some((key, r));
    }
    let Some((_, renderer)) = slot.renderer.as_mut() else {
        return Ok(None);
    };
    let scene = (shot.scene.clone(), shot.traj.clone(), shot.tick);
    if slot.scene.as_ref() != Some(&scene) {
        let tris = (slot.cache.tri_scene(&source.0, &source.1))
            .map_err(|e| format!("tessellation: {e}"))?;
        renderer
            .upload_tris(tris)
            .map_err(|e| format!("scene upload: {e}"))?;
        slot.scene = Some(scene);
    }
    // Fast and material share one `Rs` renderer: the shading is a parameter, not a pipeline.
    renderer.set_lighting(&cfg);
    if next == Next::Restart {
        renderer.restart_history();
        slot.frames = 0;
    }
    let rgb = (renderer.render(&[view]))
        .and_then(|mut atlas| atlas.read_tile(0, Channel::Rgb8))
        .map_err(|e| format!("render: {e}"))?
        .to_bytes();
    slot.frames += 1;
    slot.drawn = Some((mode, shot.clone()));
    let samples = if mode == Mode::Realistic {
        slot.frames * PT_SPP
    } else {
        1
    };
    Ok(Some(Frame {
        mode,
        shot: shot.clone(),
        picture: Picture {
            width,
            height,
            rgb,
            samples,
        },
        ms: t0.elapsed().as_secs_f32() * 1000.0,
    }))
}

/// The worker: opens the device, then draws what the viewports ask for, at most one frame per
/// viewport per round, newest request wins. It sleeps whenever no viewport is owed a frame -
/// a hidden viewport asks for nothing, a finished path trace owes nothing.
fn run(rx: &Receiver<Msg>, health: &Mutex<Health>) {
    let gpu = match Gpu::open(GpuOptions::default()) {
        Ok(gpu) => gpu,
        Err(e) => return set(health, Health::Failed(e.to_string())),
    };
    let device = gpu.capabilities().device_name.clone();
    set(
        health,
        Health::Ready {
            device: device.clone(),
            bytes: gpu.memory_reserved(),
        },
    );
    let mut slots: BTreeMap<u64, Slot<'_>> = BTreeMap::new();
    let mut wants: BTreeMap<u64, Want> = BTreeMap::new();
    let mut sensors: BTreeMap<u64, SensorSlot<'_>> = BTreeMap::new();
    loop {
        let first = if wants.is_empty() {
            match rx.recv() {
                Ok(msg) => Some(msg),
                Err(_) => break,
            }
        } else {
            None
        };
        for msg in first.into_iter().chain(rx.try_iter()) {
            match msg {
                Msg::Want(id, want) => {
                    wants.insert(id, want);
                }
                // A corner's frame is one render, drawn as it is asked for: a failure is that
                // frame's, said in the corner, and does not take the viewport's device down.
                Msg::Sensor(id, want) => {
                    let slot = sensors.entry(id).or_default();
                    let frame = sensor_frame(&gpu, slot, &want.job);
                    let _ = want.reply.send((want.key, frame));
                }
                Msg::Forget(id) => {
                    wants.remove(&id);
                    slots.remove(&id);
                    sensors.remove(&id);
                }
            }
        }
        for (id, want) in std::mem::take(&mut wants) {
            let slot = slots.entry(id).or_default();
            match draw(&gpu, slot, want.mode, &want.shot, &want.source) {
                Ok(Some(frame)) => {
                    let _ = want.reply.send(frame);
                }
                Ok(None) => {}
                Err(why) => return set(health, Health::Failed(why)),
            }
        }
        set(
            health,
            Health::Ready {
                device: device.clone(),
                bytes: gpu.memory_reserved(),
            },
        );
    }
}
