//! The `--frames` renderer: one camera per image input, under each episode's lighting.

use es_compile::PolicyBundle;

use crate::error::CliError;

/// The renderers `--frames` needs, built from what the bundle's Task IR already declares.
///
/// The same rule `es loop collect --frames` follows (`es_tools::frame_cameras`): every image
/// channel, rendered from the camera its `Frame` names, at the `ImageSpec` the IR declares --
/// so a scene whose camera produces something else is refused rather than silently resampled
/// (`INV-14`). Keyed by the name of the observation-plan input that reads the channel, which
/// is its sensor id (`CpuPlan`, spec 7.4): each image port gets its own camera's frame.
pub(super) fn renderer_cfgs(
    bundle: &PolicyBundle,
) -> Result<Vec<(String, es_env::EnvRendererCfg)>, CliError> {
    Ok(es_tools::frame_cameras(&bundle.task, None)?
        .into_iter()
        .filter_map(|(name, cfg)| Some((es_tools::plan_input(&bundle.task, &name)?, cfg)))
        .collect())
}

/// One camera, rendered under one episode's lighting (spec 10.2).
///
/// Not `es_env::EnvRenderer`: a light perturbation sets `RenderConfig::light_dir`, which is
/// fixed when a renderer is built and which `EnvRendererCfg` does not carry -- and V0b owns
/// that surface, so this composes the same public pieces (`es_env::render::render_config`,
/// `body_poses`, `camera_view`) instead of widening it. Everything else is identical, which is
/// why the frames are still the ones the render goldens pin.
///
/// The renderer is rebuilt only when the lighting changes, so a suite with no light
/// perturbation builds exactly one for the whole run.
pub(super) struct LightRig<'gpu> {
    gpu: &'gpu es_gpu::Gpu,
    /// The scene as authored; every episode's scene is derived from this one.
    scene: es_assets::scene::SceneDesc,
    cfg: es_env::EnvRendererCfg,
    lit: Option<(
        es_eval::LightOverride,
        es_assets::scene::SceneDesc,
        es_render::Renderer<'gpu>,
    )>,
    /// `RenderConfig::seed` as `render_config` builds it, and the frames rendered since this
    /// episode's tick 0 — the `Tick` seed stream's clock (packet M10/W1a).
    base_seed: u32,
    episode_frame: u32,
    /// The renderer of a task with render targets, built on the first drawn frame.
    drawn: Option<es_env::EnvRenderer<'gpu>>,
}

impl<'gpu> LightRig<'gpu> {
    pub(super) fn new(
        gpu: &'gpu es_gpu::Gpu,
        scene: es_assets::scene::SceneDesc,
        cfg: es_env::EnvRendererCfg,
    ) -> Self {
        Self {
            gpu,
            base_seed: es_env::render::render_config(&cfg).seed,
            scene,
            cfg,
            lit: None,
            episode_frame: 0,
            drawn: None,
        }
    }

    pub(super) fn frame(
        &mut self,
        drawn: &es_env::randomize::RenderOverrides,
        model: &es_physics_core::backend::ModelInfo,
        state: &es_physics_core::backend::StateView<'_>,
    ) -> Result<Vec<u8>, String> {
        let light = &drawn.light;
        // A task with render targets (packet M11/R2): the episode's draws through
        // `EnvRenderer::frame_with`, the call `es loop collect --frames` and `Rollout` make, so
        // the three render one frame at one `(seed, episode, tick)`. A task without them draws
        // nothing but the suite's light and keeps the path below, byte for byte.
        let undrawn = es_env::randomize::RenderOverrides {
            light: *light,
            ..Default::default()
        };
        if *drawn != undrawn {
            let renderer = match &mut self.drawn {
                Some(r) => r,
                None => self.drawn.insert(
                    es_env::EnvRenderer::new(self.gpu, &self.scene, self.cfg.clone())
                        .map_err(|e| e.to_string())?,
                ),
            };
            if state.tick.0 == 0 {
                renderer.begin_episode();
            }
            return renderer
                .frame_with(model, state, 0, drawn)
                .map(|tile| tile.to_bytes())
                .map_err(|e| e.to_string());
        }
        if self.lit.as_ref().is_none_or(|(l, _, _)| l != light) {
            let scene = light.scene(&self.scene);
            let mut rc = es_env::render::render_config(&self.cfg);
            let d = light.rotate_dir([rc.light_dir.x, rc.light_dir.y, rc.light_dir.z]);
            (rc.light_dir.x, rc.light_dir.y, rc.light_dir.z) = (d[0], d[1], d[2]);
            let renderer =
                es_render::Renderer::new(self.gpu, rc).map_err(|e| format!("renderer: {e}"))?;
            self.lit = Some((*light, scene, renderer));
        }
        // Tick 0 of the episode (packet M10/W1a). Since the `(cell, episode)` partition
        // (packet M8/S1) the runner builds **one `Env` per episode**, freshly reset, so the
        // backend clock reads 0 at an episode's first captured step and nowhere else -- which
        // is the same instant `es loop collect` calls `EnvRenderer::begin_episode` at, and
        // what makes the two stages render the same grain at the same `(episode, tick)`.
        if state.tick.0 == 0 {
            self.episode_frame = 0;
        }
        let ticked = (self.cfg.seed_stream == es_ir::task::SeedStream::Tick).then(|| {
            es_env::render::frame_seed(self.cfg.seed_stream, self.base_seed, self.episode_frame)
        });
        self.episode_frame += 1;
        let (_, scene, renderer) = self.lit.as_mut().expect("just built");
        // `Fixed` is left alone rather than re-set to the same number, so the default path
        // does not even touch the config (the same rule `EnvRenderer::frame` follows).
        if let Some(seed) = ticked {
            renderer.set_seed(seed);
        }
        let world = es_env::render::body_poses(model, state, 0);
        let tri = es_render::TriScene::from_scene_with_poses(scene, &world)
            .map_err(|e| format!("tessellation: {e}"))?;
        let view = es_env::render::camera_view(scene, &self.cfg, &world)
            .map_err(|e| format!("camera: {e}"))?;
        renderer
            .upload_tris(tri)
            .map_err(|e| format!("scene upload: {e}"))?;
        let mut atlas = renderer
            .render(&[view])
            .map_err(|e| format!("render: {e}"))?;
        let tile = atlas
            .read_tile(0, self.cfg.channel)
            .map_err(|e| format!("readback: {e}"))?;
        Ok(tile.to_bytes())
    }
}
