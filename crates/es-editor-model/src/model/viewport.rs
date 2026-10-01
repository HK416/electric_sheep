//! The viewport's three looks (packet M16/H8): **fast** (the CPU raster of
//! [`es_render::raster`], as before), **material** (textures and PBR, `es_render`'s own `Rs`
//! `Full` reference on worker threads) and **realistic** (the path tracer, through `es render`
//! in another process, accumulating while the camera holds still). ① Scene, ⑤'s attempts and
//! the Replay panel all show one scene from one camera at one tick - a [`Shot`] - and this
//! decides which picture of it is on screen.
//!
//! The editor still creates no Vulkan device (`docs/design/editor-redesign.md` section 5, S4):
//! the material look is the CPU reference, the realistic one is `es render --out -`, whose
//! frames arrive on a pipe as binary PPMs, one per accumulated frame.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use es_render::raster::Camera;
use es_render::{Channel, RenderConfig, Shading, TileAtlasCfg, TriScene};

/// Which look the viewport draws. Remembered for the session, shared by every viewport.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Fast,
    Material,
    Realistic,
}

impl Mode {
    pub const ALL: [Self; 3] = [Self::Fast, Self::Material, Self::Realistic];

    pub fn key(self) -> &'static str {
        match self {
            Self::Fast => "viewport.fast",
            Self::Material => "viewport.material",
            Self::Realistic => "viewport.realistic",
        }
    }

    /// What the selector's hover says the look costs.
    pub fn hint(self) -> &'static str {
        match self {
            Self::Fast => "viewport.fast.hint",
            Self::Material => "viewport.material.hint",
            Self::Realistic => "viewport.realistic.hint",
        }
    }

    /// `es-editor --viewport fast|material|pt`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "fast" => Some(Self::Fast),
            "material" => Some(Self::Material),
            "pt" => Some(Self::Realistic),
            _ => None,
        }
    }
}

/// One picture's subject: the scene file, the recorded motion and its tick (`None` is the
/// scene at its initial pose), and the camera at its pixel size.
#[derive(Clone, Debug, PartialEq)]
pub struct Shot {
    pub scene: PathBuf,
    pub traj: Option<PathBuf>,
    pub tick: usize,
    pub camera: Camera,
}

/// How long a shot must hold still before a slow look starts on it: a drag changes the camera
/// every frame, and a render per frame of a drag is a render nobody sees.
pub const SETTLE: Duration = Duration::from_millis(200);
/// The path tracer's budget per shot: `PT_FRAMES` frames of `PT_SPP` samples, then it stops,
/// so a still viewport does not hold the GPU forever.
pub const PT_SPP: u32 = 4;
pub const PT_FRAMES: u32 = 64;
pub const PT_BOUNCES: u32 = 3;
/// The exposure the hand's cameras are declared with (`task-repose.toml`: at 8 the white hand
/// keeps its shading); the scenes here are lit by one ceiling panel alike.
pub const PT_EXPOSURE: f32 = 8.0;

/// One finished picture of a shot.
#[derive(Clone, Debug, PartialEq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    /// Samples per pixel it holds: 1 for the material look.
    pub samples: u32,
}

/// What the line under the picture says.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// Nothing of this shot yet; the fast raster stands in.
    Rendering,
    /// The path tracer is still adding samples.
    Samples {
        done: u32,
        total: u32,
    },
    /// It stopped at its budget.
    Finished {
        samples: u32,
    },
    Failed(String),
}

impl Status {
    /// The i18n key and its arguments.
    pub fn line(&self) -> (&'static str, Vec<String>) {
        match self {
            Self::Rendering => ("viewport.rendering", Vec::new()),
            Self::Samples { done, total } => (
                "viewport.samples",
                vec![done.to_string(), total.to_string()],
            ),
            Self::Finished { samples } => ("viewport.finished", vec![samples.to_string()]),
            Self::Failed(why) => ("viewport.failed", vec![why.clone()]),
        }
    }
}

/// What [`plan`] decides for the shot on screen.
#[derive(Debug, PartialEq, Eq)]
pub enum Plan {
    /// The fast look: no render job at all.
    Raster,
    /// The shot moved less than [`SETTLE`] ago; the fast raster stands in.
    Wait,
    Start,
    /// A job for this shot is running, finished or failed: nothing to do.
    Keep,
}

/// The whole decision, as a pure function. `still` is how long the shot has been what it is;
/// `started` is whether a job for exactly this shot was ever started (and not since dropped).
pub fn plan(mode: Mode, still: Duration, started: bool) -> Plan {
    match () {
        () if mode == Mode::Fast => Plan::Raster,
        () if started => Plan::Keep,
        () if still < SETTLE => Plan::Wait,
        () => Plan::Start,
    }
}

#[derive(Debug)]
enum Msg {
    Frame(Picture),
    Done,
    Failed(String),
}

/// A render of one shot, on a worker thread or in an `es render` child.
#[derive(Debug)]
struct Job {
    rx: Receiver<Msg>,
    child: Option<Child>,
    running: bool,
}

impl Drop for Job {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // A shot nobody looks at any more: its frames would never be shown.
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// One viewport's render state: the shot it was last asked for and since when, the job
/// rendering it, and the newest picture.
#[derive(Debug, Default)]
pub struct Viewport {
    wanted: Option<(Mode, Shot, Instant)>,
    job: Option<Job>,
    picture: Option<Picture>,
    failed: Option<String>,
    /// Moves whenever [`Self::picture`] does, so the shell re-uploads its texture only then.
    revision: u64,
}

impl Viewport {
    /// Called once a frame with the shot on screen. Never blocks: a slow look is started on
    /// another thread or in another process, and only once the shot has held still.
    /// `tris` (the posed scene) is asked for only when the material look starts.
    pub fn update(
        &mut self,
        mode: Mode,
        shot: &Shot,
        now: Instant,
        es: &Path,
        tris: impl FnOnce() -> Result<TriScene, String>,
    ) {
        let moved = (self.wanted.as_ref()).is_none_or(|(m, s, _)| *m != mode || s != shot);
        if moved {
            self.wanted = Some((mode, shot.clone(), now));
            self.job = None;
            self.picture = None;
            self.failed = None;
            self.revision += 1;
        }
        self.drain();
        let since = self.wanted.as_ref().map_or(now, |w| w.2);
        let started = self.job.is_some() || self.picture.is_some() || self.failed.is_some();
        if plan(mode, now.saturating_duration_since(since), started) != Plan::Start {
            return;
        }
        match (mode, tris) {
            (Mode::Realistic, _) => self.job = Some(realistic_job(es, shot)),
            (_, tris) => match tris() {
                Ok(tris) => self.job = Some(material_job(shot, tris)),
                Err(why) => self.failed = Some(why),
            },
        }
    }

    /// Whatever the job has sent since the last frame.
    fn drain(&mut self) {
        let Some(job) = &mut self.job else {
            return;
        };
        while let Ok(msg) = job.rx.try_recv() {
            match msg {
                Msg::Frame(picture) => {
                    self.picture = Some(picture);
                    self.revision += 1;
                }
                Msg::Done => job.running = false,
                Msg::Failed(why) => {
                    job.running = false;
                    self.failed = Some(why);
                }
            }
        }
    }

    /// The picture of the shot last passed to [`Self::update`] and its revision, or `None`
    /// while the fast raster stands in.
    pub fn picture(&self) -> Option<(&Picture, u64)> {
        self.picture.as_ref().map(|p| (p, self.revision))
    }

    /// The line under the picture; `None` on the fast look.
    pub fn status(&self) -> Option<Status> {
        let (mode, ..) = self.wanted.as_ref()?;
        if *mode == Mode::Fast {
            return None;
        }
        if let Some(why) = &self.failed {
            return Some(Status::Failed(why.clone()));
        }
        let running = self.job.as_ref().is_some_and(|j| j.running);
        Some(match (&self.picture, *mode) {
            (None, _) => Status::Rendering,
            (Some(_), Mode::Material) => return None,
            (Some(p), _) if running => Status::Samples {
                done: p.samples,
                total: PT_SPP * PT_FRAMES,
            },
            (Some(p), _) => Status::Finished { samples: p.samples },
        })
    }

    /// Whether the shell should keep repainting: something is on its way.
    pub fn busy(&self) -> bool {
        let pending = self.wanted.as_ref().is_some_and(|(m, ..)| *m != Mode::Fast);
        pending
            && (self.job.as_ref().is_some_and(|j| j.running) || self.picture.is_none())
            && self.failed.is_none()
    }
}

/// The `Rs` `Full` look (shadows, sky, PBR, 2x supersampling) of `view`, as the CPU reference
/// draws it, split into row bands over `threads` threads.
///
/// A band is the same camera with a shorter image and its principal point moved up by the
/// band's first row, so every pixel casts the ray the whole frame would cast through it -
/// `px + offset - cx` is exact in `f32` at these sizes either way - and the concatenated bands
/// are the whole frame bit for bit (`material_bands_are_the_cpu_reference_bit_for_bit`).
pub fn material_frame(tris: &TriScene, view: &es_render::CameraView, threads: usize) -> Vec<u8> {
    let (w, h) = (view.spec.width, view.spec.height);
    let rows = h
        .div_ceil(u32::try_from(threads.max(1)).unwrap_or(1))
        .max(1);
    let band = |y0: u32| {
        let mut v = *view;
        v.spec.height = rows.min(h - y0);
        v.spec.intrinsics.cy -= y0 as f32;
        let mut cfg = RenderConfig::rs(TileAtlasCfg::row(w, v.spec.height, 1));
        cfg.shading = Shading::FULL;
        cfg.channels = [Channel::Rgb8].into();
        let frame = es_render::cpu::rasterize(tris, &v, &cfg, 0);
        frame
            .tile(Channel::Rgb8)
            .map(es_render::Tile::to_bytes)
            .unwrap_or_default()
    };
    std::thread::scope(|s| {
        let bands: Vec<_> = (0..h)
            .step_by(rows as usize)
            .map(|y0| s.spawn(move || band(y0)))
            .collect();
        bands
            .into_iter()
            .flat_map(|b| b.join().unwrap_or_default())
            .collect()
    })
}

fn job(rx: Receiver<Msg>, child: Option<Child>) -> Job {
    Job {
        rx,
        child,
        running: true,
    }
}

fn material_job(shot: &Shot, tris: TriScene) -> Job {
    let (tx, rx) = mpsc::channel();
    let camera = shot.camera;
    std::thread::spawn(move || {
        let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        let _ = tx.send(match camera.view() {
            Ok(view) => Msg::Frame(Picture {
                width: camera.width,
                height: camera.height,
                rgb: material_frame(&tris, &view, threads),
                samples: 1,
            }),
            Err(e) => Msg::Failed(e.to_string()),
        });
        let _ = tx.send(Msg::Done);
    });
    job(rx, None)
}

/// `es render`'s command line for `shot` on the path tracer, every frame to stdout.
pub fn argv(shot: &Shot) -> Vec<String> {
    let c = &shot.camera;
    let xyz = |v: [f64; 3]| format!("{},{},{}", v[0], v[1], v[2]);
    let mut args = vec![
        "render".into(),
        "--scene".into(),
        shot.scene.display().to_string(),
    ];
    if let Some(traj) = &shot.traj {
        args.extend(["--traj".into(), traj.display().to_string()]);
        args.extend(["--tick".into(), shot.tick.to_string()]);
    }
    for (flag, value) in [
        ("--eye", xyz(c.eye)),
        ("--look-at", xyz(c.look_at)),
        ("--fov", c.fov_y.to_degrees().to_string()),
        ("--width", c.width.to_string()),
        ("--height", c.height.to_string()),
        ("--path", "pt".to_owned()),
        ("--spp", PT_SPP.to_string()),
        ("--bounces", PT_BOUNCES.to_string()),
        ("--accumulate", PT_FRAMES.to_string()),
        ("--exposure", PT_EXPOSURE.to_string()),
        ("--out", "-".to_owned()),
    ] {
        args.extend([flag.to_owned(), value]);
    }
    args
}

fn realistic_job(es: &Path, shot: &Shot) -> Job {
    let (tx, rx) = mpsc::channel();
    let child = Command::new(es)
        .args(argv(shot))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(e) => {
            let _ = tx.send(Msg::Failed(format!("{}: {e}", es.display())));
            return job(rx, None);
        }
    };
    let (out, err) = (child.stdout.take(), child.stderr.take());
    std::thread::spawn(move || {
        // Stderr on its own thread, so a chatty child can never fill one pipe while this
        // thread waits on the other.
        let err = std::thread::spawn(move || {
            let mut text = String::new();
            if let Some(mut err) = err {
                let _ = err.read_to_string(&mut text);
            }
            text
        });
        let frames = out.map_or(0, |out| stream(BufReader::new(out), &tx));
        let text = err.join().unwrap_or_default();
        let last = text.lines().map(str::trim).rfind(|l| !l.is_empty());
        let _ = tx.send(match (frames, last) {
            (0, why) => Msg::Failed(why.unwrap_or("es render").to_owned()),
            _ => Msg::Done,
        });
    });
    job(rx, Some(child))
}

/// Forwards every PPM of a stream; how many there were.
fn stream(mut from: impl BufRead, tx: &Sender<Msg>) -> usize {
    let mut n = 0;
    while let Ok(Some(picture)) = read_ppm(&mut from) {
        n += 1;
        if tx.send(Msg::Frame(picture)).is_err() {
            break;
        }
    }
    n
}

/// One binary PPM (`es render`'s format: `P6`, `# samples N`, `W H`, `255`, then the bytes)
/// from the front of `from`; `Ok(None)` at a clean end of stream.
pub fn read_ppm(from: &mut impl BufRead) -> std::io::Result<Option<Picture>> {
    let bad = |what: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, what.to_owned());
    let mut fields: Vec<u32> = Vec::new();
    let mut samples = 1;
    let mut line = String::new();
    let mut magic = false;
    while fields.len() < 3 {
        line.clear();
        if from.read_line(&mut line)? == 0 {
            return if magic {
                Err(bad("truncated header"))
            } else {
                Ok(None)
            };
        }
        let text = line.trim();
        if !magic {
            magic = text == "P6";
            if !magic {
                return Err(bad("not a P6 image"));
            }
        } else if let Some(n) = text.strip_prefix("# samples ") {
            samples = n.parse().map_err(|_| bad("sample count"))?;
        } else if !text.starts_with('#') {
            for word in text.split_whitespace() {
                fields.push(word.parse().map_err(|_| bad("header field"))?);
            }
        }
    }
    let (width, height) = (fields[0], fields[1]);
    if fields[2] != 255 || width == 0 || height == 0 || width * height > 4096 * 4096 {
        return Err(bad("unsupported size or depth"));
    }
    let mut rgb = vec![0; (width * height * 3) as usize];
    from.read_exact(&mut rgb)?;
    Ok(Some(Picture {
        width,
        height,
        rgb,
        samples,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::i18n::{fill, Lang, Strings};

    fn root() -> PathBuf {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).to_path_buf()
    }

    fn textured() -> PathBuf {
        root().join("tests/fixtures/mjcf/textured/textured.xml")
    }

    fn shot(width: u32, height: u32, tick: usize) -> Shot {
        Shot {
            scene: textured(),
            traj: None,
            tick,
            camera: Camera {
                eye: [0.32, -0.3, 0.3],
                look_at: [0.0, 0.0, 0.04],
                fov_y: 0.9,
                width,
                height,
            },
        }
    }

    fn tris() -> TriScene {
        let scene = crate::model::replay_view::load_scene(&textured()).expect("fixture");
        TriScene::from_scene(&scene).expect("tessellate")
    }

    #[test]
    fn modes_parse_and_every_word_is_in_both_tables() {
        for mode in Mode::ALL {
            for lang in Lang::ALL {
                for key in [mode.key(), mode.hint()] {
                    assert_ne!(Strings::get(lang).t(key), key, "{key} missing in {lang:?}");
                }
            }
        }
        assert_eq!(Mode::parse("pt"), Some(Mode::Realistic));
        assert_eq!(Mode::parse("material"), Some(Mode::Material));
        assert_eq!(Mode::parse("fast"), Some(Mode::Fast));
        assert_eq!(Mode::parse("nice"), None);
        let lines = [
            Status::Rendering,
            Status::Samples {
                done: 8,
                total: 256,
            },
            Status::Finished { samples: 256 },
            Status::Failed("no Vulkan device".into()),
        ];
        for status in lines {
            let (key, args) = status.line();
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            for lang in Lang::ALL {
                let line = fill(lang, key, &args);
                assert!(!line.contains("{}") && line != key, "{lang:?}: {line}");
                for arg in &args {
                    assert!(line.contains(arg), "{lang:?}: {line}");
                }
            }
        }
    }

    /// The fast look never starts a job; a slow one waits for the shot to hold still for
    /// [`SETTLE`], starts once, and keeps what it started.
    #[test]
    fn a_slow_look_waits_for_a_still_shot_and_starts_once() {
        let ms = Duration::from_millis;
        assert_eq!(plan(Mode::Fast, ms(5000), false), Plan::Raster);
        for mode in [Mode::Material, Mode::Realistic] {
            assert_eq!(plan(mode, ms(0), false), Plan::Wait);
            assert_eq!(plan(mode, SETTLE.saturating_sub(ms(1)), false), Plan::Wait);
            assert_eq!(plan(mode, SETTLE, false), Plan::Start);
            assert_eq!(plan(mode, ms(5000), true), Plan::Keep);
        }
    }

    #[test]
    fn material_bands_are_the_cpu_reference_bit_for_bit() {
        let tris = tris();
        // An odd height, so the last band is shorter than the others.
        let view = shot(64, 45, 0).camera.view().expect("view");
        let mut cfg = RenderConfig::rs(TileAtlasCfg::row(64, 45, 1));
        cfg.shading = Shading::FULL;
        let whole = es_render::cpu::rasterize(&tris, &view, &cfg, 0);
        let whole = whole.tile(Channel::Rgb8).expect("rgb8").to_bytes();
        for threads in [1, 4, 7, 45, 64] {
            assert_eq!(
                material_frame(&tris, &view, threads),
                whole,
                "{threads} bands"
            );
        }
        // Textured: the block's letters and the checker are not one flat colour per geom.
        let distinct: std::collections::BTreeSet<&[u8]> = whole.chunks(3).collect();
        assert!(distinct.len() > 500, "{} colours", distinct.len());
    }

    /// The material look end to end: the fast raster stands in until the shot has held still,
    /// the picture arrives from a worker thread, and moving the tick or the camera drops it
    /// and starts over.
    #[test]
    fn the_material_look_renders_off_thread_and_restarts_when_the_shot_moves() {
        let es = Path::new("es-not-needed");
        let mut vp = Viewport::default();
        let t0 = Instant::now();
        let first = shot(48, 32, 0);
        vp.update(Mode::Material, &first, t0, es, || {
            panic!("not before SETTLE")
        });
        assert_eq!(vp.status(), Some(Status::Rendering));
        assert!(vp.picture().is_none() && vp.busy());
        let wait = |vp: &mut Viewport, shot: &Shot, at: Instant| {
            let mut asked = 0;
            vp.update(Mode::Material, shot, at, es, || {
                asked += 1;
                Ok(tris())
            });
            for _ in 0..600 {
                if vp.picture().is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
                vp.update(Mode::Material, shot, at, es, || panic!("started twice"));
            }
            asked
        };
        assert_eq!(wait(&mut vp, &first, t0 + SETTLE), 1);
        let (picture, revision) = vp.picture().expect("the picture");
        assert_eq!(
            (picture.width, picture.height, picture.samples),
            (48, 32, 1)
        );
        let want = material_frame(&tris(), &first.camera.view().expect("view"), 1);
        assert_eq!(picture.rgb, want);
        assert_eq!(
            vp.status(),
            None,
            "a finished material picture needs no line"
        );
        assert!(!vp.busy());

        let moved = shot(48, 32, 7);
        let t1 = t0 + Duration::from_secs(1);
        vp.update(Mode::Material, &moved, t1, es, || {
            panic!("not before SETTLE")
        });
        assert!(vp.picture().is_none(), "a picture of the old tick");
        assert_eq!(vp.status(), Some(Status::Rendering));
        assert_eq!(wait(&mut vp, &moved, t1 + SETTLE), 1);
        assert!(vp.picture().expect("again").1 > revision);

        vp.update(Mode::Fast, &moved, t1 + SETTLE, es, || panic!("fast"));
        assert!(vp.picture().is_none() && vp.status().is_none() && !vp.busy());
    }

    /// Frames of the realistic look replace each other as they arrive and say how many samples
    /// they hold; the end of the stream finishes it; a child that drew nothing fails with its
    /// own last line.
    #[test]
    fn realistic_frames_count_samples_and_a_silent_child_fails() {
        let mut stream_bytes = Vec::new();
        for (k, level) in [(1u32, 10u8), (2, 20), (3, 30)] {
            stream_bytes.extend(format!("P6\n# samples {}\n2 1\n255\n", k * PT_SPP).bytes());
            stream_bytes.extend([level; 6]);
        }
        let (tx, rx) = mpsc::channel();
        assert_eq!(stream(stream_bytes.as_slice(), &tx), 3);
        let mut vp = Viewport::default();
        let at = Instant::now();
        let s = shot(2, 1, 0);
        vp.update(Mode::Realistic, &s, at, Path::new("unused"), || {
            Err(String::new())
        });
        vp.job = Some(job(rx, None));
        vp.update(Mode::Realistic, &s, at, Path::new("unused"), || {
            Err(String::new())
        });
        let (picture, _) = vp.picture().expect("the newest frame");
        assert_eq!((picture.samples, picture.rgb[0]), (3 * PT_SPP, 30));
        let total = PT_SPP * PT_FRAMES;
        let status = vp.status();
        assert_eq!(
            status,
            Some(Status::Samples {
                done: 3 * PT_SPP,
                total
            })
        );
        tx.send(Msg::Done).expect("send");
        vp.update(Mode::Realistic, &s, at, Path::new("unused"), || {
            Err(String::new())
        });
        assert_eq!(
            vp.status(),
            Some(Status::Finished {
                samples: 3 * PT_SPP
            })
        );
        assert!(!vp.busy());

        // A program that is not there: the reason, not a hang and not a blank.
        let mut vp = Viewport::default();
        let missing = Path::new("no-such-es-binary-h8");
        vp.update(Mode::Realistic, &s, at, missing, || Err(String::new()));
        vp.update(Mode::Realistic, &s, at + SETTLE, missing, || {
            Err(String::new())
        });
        for _ in 0..200 {
            if matches!(vp.status(), Some(Status::Failed(_))) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
            vp.update(Mode::Realistic, &s, at + SETTLE, missing, || {
                Err(String::new())
            });
        }
        let Some(Status::Failed(why)) = vp.status() else {
            panic!("{:?}", vp.status());
        };
        assert!(why.contains("no-such-es-binary-h8"), "{why}");
        assert!(read_ppm(&mut "P5\n1 1\n255\n\0".as_bytes()).is_err());
        assert!(read_ppm(&mut "".as_bytes()).expect("clean end").is_none());
    }

    /// The child's command line: the shot's scene, motion and tick, its camera exactly (the
    /// floats round-trip through their text), and the path tracer's budget.
    #[test]
    fn the_realistic_command_line_is_the_shot() {
        let mut s = shot(320, 200, 12);
        s.traj = Some(PathBuf::from("run/traj/nominal-00.estraj"));
        let args = argv(&s);
        let after = |flag: &str| {
            let at = args.iter().position(|a| a == flag).expect(flag);
            args[at + 1].clone()
        };
        assert_eq!(args[0], "render");
        assert_eq!(after("--tick"), "12");
        assert_eq!(after("--traj"), "run/traj/nominal-00.estraj");
        assert_eq!(after("--eye"), "0.32,-0.3,0.3");
        assert_eq!(
            (after("--width"), after("--height")),
            ("320".into(), "200".into())
        );
        assert_eq!(after("--path"), "pt");
        assert_eq!(after("--out"), "-");
        let fov: f64 = after("--fov").parse().expect("fov");
        assert!((fov.to_radians() - 0.9).abs() < 1e-12);
        let still = argv(&shot(320, 200, 0));
        assert!(!still.contains(&"--tick".to_owned()), "{still:?}");
    }
}
