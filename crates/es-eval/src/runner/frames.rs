//! Where a run's frames land: the [`FrameSink`] and one cell's [`CellFrames`] directory,
//! with the episode's render draws and the intrinsics they give its image sensor.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use es_env::randomize::RenderOverrides;
use es_ir::task::TaskIr;
use es_ir::types::ElemType;

use super::StepEvent;
use crate::perturb::LightOverride;
use crate::EvalError;

/// Where a run puts its frames and events; `None` is today's behaviour exactly.
///
/// One subdirectory per cell — a cell being one episode of one suite, because
/// `BatchDomains::single_env()` makes every episode its own run — holding `<NNNNNN>.bin` plus
/// one `layout.json`, which is the directory shape `es video mosaic` tiles.
#[derive(Clone, Debug, Default)]
pub struct FrameSink {
    pub dir: PathBuf,
    /// Cell name -> its frame-ordered records, in the shape `events.json` is written in.
    pub events: BTreeMap<String, Vec<StepEvent>>,
}

impl FrameSink {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            events: BTreeMap::new(),
        }
    }

    /// Writes `events.json`: `{ "<cell>": [ StepEvent, ... ] }`.
    pub fn write_events(&self, path: &Path) -> Result<(), EvalError> {
        let mut text = serde_json::to_string_pretty(&self.events)
            .map_err(|e| EvalError::Plan(e.to_string()))?;
        text.push('\n');
        fs::write(path, text).map_err(|source| EvalError::Io {
            path: path.display().to_string(),
            source,
        })
    }
}

/// This episode's render draws with the suite's light perturbation folded in (packet M11/R2):
/// intensities multiply and yaws add. A suite with no light perturbation leaves the draws
/// exactly as recorded; a task with no render target is left with the suite's light alone.
pub(super) fn episode_draws(recorded: &RenderOverrides, light: &LightOverride) -> RenderOverrides {
    let mut out = recorded.clone();
    if !light.is_identity() {
        out.light.intensity *= light.intensity;
        out.light.yaw_deg += light.yaw_deg;
    }
    out
}

/// The intrinsics a drawn field of view gives the task's image sensor this episode -- the
/// episode's own `ImageSpec` (`RenderOverrides::image_spec`, `INV-14`) -- or `None` when no
/// camera an image channel names was drawn.
fn drawn_intrinsics(task: &TaskIr, drawn: &RenderOverrides) -> Option<es_ir::image::Intrinsics> {
    task.observation_spec
        .channels
        .values()
        .find_map(|c| channel_intrinsics(c, drawn))
}

/// [`drawn_intrinsics`] of one channel: `None` unless it is an image sensor whose camera was
/// drawn.
fn channel_intrinsics(
    c: &es_ir::task::ObsChannel,
    drawn: &RenderOverrides,
) -> Option<es_ir::image::Intrinsics> {
    match (&c.source, c.ty.frame, c.ty.image) {
        (
            es_ir::task::ObsSource::Sensor { .. },
            es_ir::types::Frame::Camera(camera),
            Some(spec),
        ) if drawn.cameras.contains_key(&camera) => {
            Some(drawn.image_spec(camera, &spec).intrinsics)
        }
        _ => None,
    }
}

/// One cell's frame directory: the raw `.bin` sequence plus the `layout.json` that pins their
/// shape, written as the frames are captured.
///
/// `pub` only because it is an argument of the now-public [`capture`] (packet M8/S4a); it is
/// still built here and nowhere else, and a caller outside this crate can only pass `None`.
#[derive(Debug)]
pub struct CellFrames {
    dir: PathBuf,
    pub(super) n: u64,
    /// The episode's drawn intrinsics, written into `layout.json` beside `dtype` and `shape`
    /// (packet M11/R2, `INV-14`); `None` writes the layout it always did.
    intrinsics: Option<es_ir::image::Intrinsics>,
    /// A plan with several image inputs (packet M15/N2): input name -> its own
    /// `<cell>/<channel>/` directory, each a cell of its own. Empty for one image input,
    /// which writes into `dir` itself exactly as before.
    views: BTreeMap<String, CellFrames>,
}

impl CellFrames {
    /// The cell `dir` of `task`'s episode drawn as `drawn`, for a plan whose image inputs are
    /// `images` (their plan input names). Several inputs each get a subdirectory named after
    /// the Task IR channel whose sensor they read -- the layout `es loop collect --frames`
    /// writes -- and that channel's own drawn intrinsics.
    pub(super) fn new(
        dir: PathBuf,
        task: &TaskIr,
        drawn: &RenderOverrides,
        images: &[&String],
    ) -> Self {
        let views = if images.len() > 1 {
            images
                .iter()
                .map(|input| {
                    let channel = task.observation_spec.channels.iter().find(|(_, c)| {
                        matches!(c.source, es_ir::task::ObsSource::Sensor { id, .. }
                            if id.to_string() == **input)
                    });
                    let name = channel.map_or((*input).clone(), |(name, _)| name.clone());
                    let intrinsics = channel.and_then(|(_, c)| channel_intrinsics(c, drawn));
                    let view = Self {
                        dir: dir.join(name),
                        n: 0,
                        intrinsics,
                        views: BTreeMap::new(),
                    };
                    ((*input).clone(), view)
                })
                .collect()
        } else {
            BTreeMap::new()
        };
        Self {
            intrinsics: drawn_intrinsics(task, drawn),
            dir,
            n: 0,
            views,
        }
    }

    /// Appends one frame of image input `input` and returns its index. The first one writes
    /// `layout.json`, so a cell that rendered nothing leaves no half-described directory
    /// behind. Every input of a step writes one frame, so the indices advance together.
    pub(super) fn write(
        &mut self,
        input: &str,
        dtype: ElemType,
        shape: &[u64],
        data: &[u8],
    ) -> Result<u64, EvalError> {
        if let Some(view) = self.views.get_mut(input) {
            let index = view.write(input, dtype, shape, data)?;
            self.n = view.n;
            return Ok(index);
        }
        let io = |path: &Path| {
            let p = path.display().to_string();
            move |source| EvalError::Io {
                path: p.clone(),
                source,
            }
        };
        if self.n == 0 {
            fs::create_dir_all(&self.dir).map_err(io(&self.dir))?;
            let layout = self.dir.join("layout.json");
            let drawn = match &self.intrinsics {
                Some(i) => format!(
                    ",\"intrinsics\":{}",
                    serde_json::to_string(i).expect("four floats serialize")
                ),
                None => String::new(),
            };
            let text = format!(
                "{{\"dtype\":\"{}\",\"shape\":{:?}{drawn}}}\n",
                dtype_name(dtype),
                shape
            );
            fs::write(&layout, text).map_err(io(&layout))?;
        }
        let path = self.dir.join(format!("{:06}.bin", self.n));
        fs::write(&path, data).map_err(io(&path))?;
        self.n += 1;
        Ok(self.n - 1)
    }
}

/// The `layout.json` spelling of an element type — the one `es video mosaic` and the render
/// goldens read.
fn dtype_name(e: ElemType) -> &'static str {
    match e {
        ElemType::U8 => "u8",
        ElemType::Bool => "bool",
        ElemType::F16 => "f16",
        ElemType::Bf16 => "bf16",
        ElemType::F32 => "f32",
        ElemType::F64 => "f64",
        ElemType::I32 => "i32",
    }
}
