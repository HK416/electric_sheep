//! The GPU renderer: compile the kernels, upload the scene, dispatch one pass per stage,
//! hand back the atlas.
//!
//! Everything is a compute dispatch on `es-gpu`'s single queue with a full barrier after
//! each, so the recorded order *is* the execution order (spec 3.4).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use es_assets::scene::SceneDesc;
use es_gpu::{
    BindingDesc, BindingKind, Buffer, CommandRecorder, ComputePipeline, Gpu, SlangCompiler, Usage,
};
use es_sensor::Channel;

use crate::atlas::{words_per_pixel, AtlasLayout, Tile, TileData};
use crate::bvh::{Bvh, NODE_STRIDE};
use crate::error::RenderError;
use crate::scene::TriScene;
use crate::view::{
    CameraView, RenderConfig, RenderPath, Shading, Tonemap, ViewParams, VIEW_STRIDE,
};

/// Globals before the per-view records in the parameter buffer. Slots 0..20 are M4's and
/// never move; 20..31 are packet M7/R2's shading block, 31..37 packet M7/R3's `Pt` block and
/// 37..39 packet M7/R4's temporal block, each appended at the end (`common.slang` mirrors
/// every number).
const PARAM_VIEW_BASE: usize = 39;
/// Floats per direct-lighting reservoir (mirrors `restir.slang`).
const RES_STRIDE: u64 = 8;
/// Floats per pixel of the temporal history (mirrors `common.slang`'s `ES_HIST_STRIDE` and
/// `cpu::History`): the radiance sum, the two luminance moments, the history length, the
/// previous depth, primitive id and normal, and the variance.
const HIST_STRIDE: u64 = 12;
const WORKGROUP: u32 = 8;
const PROFILE: &str = "glsl_450";

fn storage(binding: u32) -> BindingDesc {
    BindingDesc {
        binding,
        kind: BindingKind::Storage,
    }
}

fn bindings(n: u32) -> Vec<BindingDesc> {
    (0..n).map(storage).collect()
}

fn slang_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("slang")
}

fn math_slang_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../es-math/slang")
}

/// One camera's slice of the atlas, per channel. The buffers stay on the device until
/// [`Atlas::read_tile`] pulls one tile back (spec 15.1: the pipeline is GPU-resident and a
/// readback is the exception, not the step).
pub struct Atlas<'gpu> {
    layout: AtlasLayout,
    n_views: u32,
    channels: BTreeMap<Channel, Buffer<'gpu>>,
}

impl std::fmt::Debug for Atlas<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Atlas")
            .field("layout", &self.layout)
            .field("n_views", &self.n_views)
            .field("channels", &self.channels.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Atlas<'_> {
    pub fn layout(&self) -> AtlasLayout {
        self.layout
    }

    pub fn channels(&self) -> impl Iterator<Item = &Channel> {
        self.channels.keys()
    }

    pub fn tiles(&self) -> u32 {
        self.n_views
    }

    /// Read one camera's tile of one channel back to the host as a row-major `[h, w, c]`
    /// tensor. [`Channel::Rgb8`] is stored as packed `RGBA8` on the device; the unused alpha
    /// is dropped here.
    pub fn read_tile(&mut self, cam: u32, channel: Channel) -> Result<Tile, RenderError> {
        let Some(buffer) = self.channels.get_mut(&channel) else {
            return Err(RenderError::UnsupportedChannel { channel });
        };
        if cam >= self.n_views {
            return Err(RenderError::AtlasTooSmall {
                views: cam as usize + 1,
                capacity: self.n_views as usize,
            });
        }
        let words = buffer.download()?;
        let cfg = self.layout.cfg;
        let (ox, oy) = self.layout.tile_origin(cam);
        let per_px = words_per_pixel(channel) as usize;
        let (tw, th) = (cfg.tile_w as usize, cfg.tile_h as usize);
        let aw = self.layout.width as usize;

        let word = |i: usize| {
            let b = i * 4;
            u32::from_le_bytes([words[b], words[b + 1], words[b + 2], words[b + 3]])
        };
        let mut u8_out = Vec::new();
        let mut u32_out = Vec::new();
        let mut f32_out = Vec::new();
        for y in 0..th {
            for x in 0..tw {
                let src = ((oy as usize + y) * aw + ox as usize + x) * per_px;
                match channel {
                    Channel::Rgb8 => {
                        let w = word(src);
                        u8_out.extend_from_slice(&[
                            (w & 0xff) as u8,
                            ((w >> 8) & 0xff) as u8,
                            ((w >> 16) & 0xff) as u8,
                        ]);
                    }
                    Channel::SegmentationId | Channel::History => u32_out.push(word(src)),
                    _ => {
                        for c in 0..per_px {
                            f32_out.push(f32::from_bits(word(src + c)));
                        }
                    }
                }
            }
        }
        let comps = match channel {
            Channel::Rgb8 => 3,
            _ => per_px,
        };
        let shape = [th, tw, comps];
        let data = match channel {
            Channel::Rgb8 => TileData::U8(u8_out),
            Channel::SegmentationId | Channel::History => TileData::U32(u32_out),
            _ => TileData::F32(f32_out),
        };
        Ok(Tile { shape, data })
    }
}

struct Pipelines<'gpu> {
    primary: ComputePipeline<'gpu>,
    restir: Option<[ComputePipeline<'gpu>; 3]>,
    svgf: Option<ComputePipeline<'gpu>>,
    /// The variance estimate of packet M7/R4, between the tracer and the filter. Present
    /// exactly when `cfg.temporal` is.
    accumulate: Option<ComputePipeline<'gpu>>,
    /// Linear radiance -> `Rgb8` on the `Pt` path (packet M7/R3). Its own dispatch, recorded
    /// last, so it reads what `ReSTIR` and `SVGF` left rather than what the tracer wrote.
    tonemap: Option<ComputePipeline<'gpu>>,
}

/// Keep and grow, never shrink (packet M7/R1 step 2). A frame that needs fewer bytes than the
/// last one reuses the buffer it has; `Buffer::new` is a `vkCreateBuffer` plus an allocation,
/// and the scene size is constant across a replay, so this allocates once.
///
/// A free function, not a method: `self.gpu` and `&mut self.<field>` are disjoint borrows of
/// `Renderer` only when they are named separately.
fn grow<'gpu>(gpu: &'gpu Gpu, slot: &mut Buffer<'gpu>, bytes: u64) -> Result<(), RenderError> {
    if slot.size() < bytes.max(4) {
        *slot = Buffer::new(gpu, bytes.max(4), Usage::Storage)?;
    }
    Ok(())
}

/// The compute renderer (spec 15).
pub struct Renderer<'gpu> {
    gpu: &'gpu Gpu,
    cfg: RenderConfig,
    layout: AtlasLayout,
    pipelines: Pipelines<'gpu>,
    /// The triangles, then the BVH. Persistent across frames.
    tris: Buffer<'gpu>,
    /// Persistent too: the globals, the per-view records and the `ReSTIR` light count.
    params_buf: Buffer<'gpu>,
    /// Primary-hit triangle index per pixel, read by the `ReSTIR` and `SVGF` passes and by
    /// nothing on the host. Persistent, atlas-sized.
    hit_tri: Buffer<'gpu>,
    tri_scene: TriScene,
    n_tri: u32,
    n_lights: u32,
    /// Where the BVH starts inside `tris`, in floats, and how many nodes it has.
    bvh_base: u32,
    bvh_nodes: u32,
    bvh_prim_base: u32,
    /// Previous-frame reservoirs for `ReSTIR` temporal reuse. Empty until the first render
    /// finishes, which is why the temporal pass is a no-op on frame 1 (see the design doc).
    prev_reservoirs: Option<Buffer<'gpu>>,
    /// The temporal history of packet M7/R4, `HIST_STRIDE` floats per atlas pixel, kept
    /// across `render` calls. A 4-byte placeholder while `cfg.temporal` is `None`.
    history: Buffer<'gpu>,
    /// Atlas pixels `history` was allocated and zeroed for.
    history_px: u64,
    /// The views the history was built with, bitwise. A view that moved invalidates its whole
    /// tile — there is no reprojection (`docs/design/renderer.md` section 11).
    prev_views: Vec<[u32; VIEW_STRIDE]>,
    /// Frames rendered since the history was allocated: the sample base `frame * spp + s`.
    frame: u32,
}

impl std::fmt::Debug for Renderer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("layout", &self.layout)
            .field("path", &self.cfg.path)
            .field("n_tri", &self.n_tri)
            .finish_non_exhaustive()
    }
}

impl<'gpu> Renderer<'gpu> {
    /// Compile the kernels for `cfg.path` with the device's deterministic execution modes
    /// (spec 3.4 step 3).
    pub fn new(gpu: &'gpu Gpu, cfg: RenderConfig) -> Result<Self, RenderError> {
        for channel in &cfg.channels {
            if !crate::path_produces(cfg.path, *channel) {
                return Err(RenderError::UnsupportedChannel { channel: *channel });
            }
        }
        let layout = AtlasLayout::new(cfg.atlas)?;
        let compiler = SlangCompiler::new()?
            .with_include(slang_dir())
            .with_include(math_slang_dir());
        let modes = gpu.deterministic_execution_modes();
        let defines = BTreeMap::new();
        let compile = |file: &str, entry: &str, n_bindings: u32| {
            let module =
                compiler.compile_file(slang_dir().join(file), entry, PROFILE, &defines, modes)?;
            ComputePipeline::new(gpu, &module, &bindings(n_bindings))
        };

        let (primary, restir, svgf, tonemap, accumulate) = match cfg.path {
            RenderPath::Rs => (compile("raster.slang", "main", 7)?, None, None, None, None),
            RenderPath::Pt { restir, svgf, .. } => {
                let pt = compile("pt.slang", "main", 10)?;
                let r = if restir {
                    Some([
                        compile("restir.slang", "initial", 9)?,
                        compile("restir.slang", "temporal", 9)?,
                        compile("restir.slang", "spatial", 9)?,
                    ])
                } else {
                    None
                };
                let s = if svgf {
                    Some(compile("svgf.slang", "main", 9)?)
                } else {
                    None
                };
                let t = if cfg.channels.contains(&Channel::Rgb8) {
                    Some(compile("pt.slang", "tonemap", 10)?)
                } else {
                    None
                };
                let a = if cfg.max_history().is_some() {
                    Some(compile("accumulate.slang", "main", 7)?)
                } else {
                    None
                };
                (pt, r, s, t, a)
            }
        };

        Ok(Self {
            gpu,
            cfg,
            layout,
            pipelines: Pipelines {
                primary,
                restir,
                svgf,
                accumulate,
                tonemap,
            },
            tris: Buffer::new(gpu, 4, Usage::Storage)?,
            params_buf: Buffer::new(gpu, 4, Usage::Storage)?,
            hit_tri: Buffer::new(gpu, 4, Usage::Storage)?,
            tri_scene: TriScene::default(),
            n_tri: 0,
            n_lights: 0,
            bvh_base: 0,
            bvh_nodes: 0,
            bvh_prim_base: 0,
            prev_reservoirs: None,
            history: Buffer::new(gpu, 4, Usage::Storage)?,
            history_px: 0,
            prev_views: Vec::new(),
            frame: 0,
        })
    }

    /// Tessellate and upload. Primitives only: `Shape::Mesh` and `Shape::HeightField` are
    /// [`RenderError::UnsupportedShape`] (see `docs/design/renderer.md`).
    pub fn upload_scene(&mut self, scene: &SceneDesc) -> Result<(), RenderError> {
        self.upload_tris(TriScene::from_scene(scene)?)
    }

    /// The tessellated scene, so a caller can map segmentation ids back to geom names
    /// without tessellating twice.
    pub fn tri_scene(&self) -> &TriScene {
        &self.tri_scene
    }

    /// Upload an already-tessellated scene, and the BVH over it.
    ///
    /// The CPU reference takes the same [`TriScene`] and builds the same [`Bvh`] from it,
    /// which is how the two paths are guaranteed to see identical geometry *and* identical
    /// traversal. The tree goes into the same buffer, after the triangles: `es-gpu` binds
    /// seven descriptors to this kernel and the tree is not worth an eighth, a second
    /// staging copy or a renumbering of every shader's bindings.
    pub fn upload_tris(&mut self, tri: TriScene) -> Result<(), RenderError> {
        let bvh = Bvh::build(&tri.tris);
        let mut floats = tri.to_floats();
        self.bvh_base = u32::try_from(floats.len()).unwrap_or(u32::MAX);
        self.bvh_nodes = u32::try_from(bvh.nodes.len()).unwrap_or(u32::MAX);
        self.bvh_prim_base = self.bvh_base + self.bvh_nodes * NODE_STRIDE as u32;
        floats.extend(bvh.to_floats());
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        grow(self.gpu, &mut self.tris, bytes.len() as u64)?;
        self.tris.upload(&bytes)?;
        self.n_tri = u32::try_from(tri.tris.len()).unwrap_or(u32::MAX);
        self.n_lights = u32::try_from(tri.lights.len()).unwrap_or(u32::MAX);
        self.tri_scene = tri;
        Ok(())
    }

    fn params(&self, cameras: &[CameraView], history_valid: &[bool]) -> Vec<f32> {
        let cfg = &self.cfg;
        let mut p = vec![0.0f32; PARAM_VIEW_BASE + cameras.len() * VIEW_STRIDE];
        p[0] = f32::from_bits(self.n_tri);
        p[1] = f32::from_bits(cameras.len() as u32);
        p[2] = f32::from_bits(cfg.atlas.tile_w);
        p[3] = f32::from_bits(cfg.atlas.tile_h);
        p[4] = f32::from_bits(cfg.atlas.tiles_per_row);
        p[5] = f32::from_bits(self.layout.rows);
        p[6] = cfg.light_dir.x as f32;
        p[7] = cfg.light_dir.y as f32;
        p[8] = cfg.light_dir.z as f32;
        p[9] = cfg.ambient;
        p[10] = cfg.sky[0];
        p[11] = cfg.sky[1];
        p[12] = cfg.sky[2];
        p[13] = f32::from_bits(cfg.seed);
        p[14] = f32::from_bits(cfg.spp().max(1));
        p[15] = f32::from_bits(cfg.bounces().max(1));
        p[16] = f32::from_bits(self.n_lights);
        p[17] = f32::from_bits(self.bvh_base);
        p[18] = f32::from_bits(self.bvh_nodes);
        p[19] = f32::from_bits(self.bvh_prim_base);
        // The `Rs` shading block (packet M7/R2). `Lambert` writes the flag and leaves the
        // rest zero; the kernel reads none of it then.
        if let Shading::Full {
            shadows,
            specular,
            shininess,
            sky_rgb,
            ground_rgb,
            ..
        } = cfg.shading
        {
            p[20] = f32::from_bits(1);
            p[21] = f32::from_bits(u32::from(shadows));
            p[22] = specular;
            p[23] = shininess;
            p[24..27].copy_from_slice(&sky_rgb);
            p[27..30].copy_from_slice(&ground_rgb);
            p[30] = f32::from_bits(cfg.shading.ssaa());
        }
        // The `Pt` block (packet M7/R3). At the defaults every one of these is what the
        // pre-R3 kernel behaved as: no NEE, a directional light with zero radiance, exposure
        // 1, Reinhard.
        p[31] = f32::from_bits(u32::from(cfg.nee()));
        p[32..35].copy_from_slice(&cfg.light_rgb);
        p[35] = cfg.exposure;
        p[36] = f32::from_bits(match cfg.tonemap {
            Tonemap::Reinhard => 0,
            Tonemap::Aces => 1,
        });
        // The temporal block (packet M7/R4). Zero is no accumulation at all, which is the
        // pre-R4 kernel byte for byte.
        p[37] = f32::from_bits(cfg.max_history().unwrap_or(0));
        p[38] = f32::from_bits(self.frame);
        for (i, cam) in cameras.iter().enumerate() {
            let base = PARAM_VIEW_BASE + i * VIEW_STRIDE;
            p[base..base + VIEW_STRIDE].copy_from_slice(&ViewParams::new(cam).to_floats());
            // Pad slot 13 of the view record carries the still-camera test: the host owns it
            // because only the host has both frames' `ViewParams`.
            p[base + 13] =
                f32::from_bits(u32::from(history_valid.get(i).copied().unwrap_or(false)));
        }
        p
    }

    /// Per view: is this camera bitwise the one the history at its tile was built with? Also
    /// (re)allocates and zeroes the history buffer when the atlas size changed, and advances
    /// the slot's frame counter (packet M7/R4).
    fn prepare_history(
        &mut self,
        cameras: &[CameraView],
        px: u64,
    ) -> Result<Vec<bool>, RenderError> {
        let Some(_) = self.cfg.max_history() else {
            self.frame = 0;
            return Ok(vec![false; cameras.len()]);
        };
        let bytes = px * HIST_STRIDE * 4;
        if self.history_px != px {
            // Device-local memory is uninitialised; an `n` of garbage would reuse a sample
            // that never existed (the same reason the `ReSTIR` buffer is zeroed).
            self.history = Buffer::new(self.gpu, bytes.max(4), Usage::Storage)?;
            self.history.upload(&vec![0u8; bytes as usize])?;
            self.history_px = px;
            self.prev_views.clear();
            self.frame = 0;
        }
        let now: Vec<[u32; VIEW_STRIDE]> = cameras
            .iter()
            .map(|cam| {
                let mut bits = [0u32; VIEW_STRIDE];
                for (slot, f) in bits.iter_mut().zip(ViewParams::new(cam).to_floats()) {
                    *slot = f.to_bits();
                }
                bits
            })
            .collect();
        let valid = now
            .iter()
            .enumerate()
            .map(|(i, bits)| self.prev_views.get(i) == Some(bits))
            .collect();
        self.prev_views = now;
        Ok(valid)
    }

    fn validate(&self, cameras: &[CameraView]) -> Result<(), RenderError> {
        if cameras.len() > self.cfg.atlas.n_tiles as usize {
            return Err(RenderError::AtlasTooSmall {
                views: cameras.len(),
                capacity: self.cfg.atlas.n_tiles as usize,
            });
        }
        for (i, cam) in cameras.iter().enumerate() {
            if cam.spec.width != self.cfg.atlas.tile_w || cam.spec.height != self.cfg.atlas.tile_h {
                return Err(RenderError::ViewTileMismatch {
                    view: i,
                    width: cam.spec.width,
                    height: cam.spec.height,
                    tile_w: self.cfg.atlas.tile_w,
                    tile_h: self.cfg.atlas.tile_h,
                });
            }
        }
        Ok(())
    }

    fn new_buffer(&self, words: u64) -> Result<Buffer<'gpu>, RenderError> {
        Ok(Buffer::new(self.gpu, (words * 4).max(4), Usage::Storage)?)
    }

    /// Render every camera into one tile of the atlas.
    ///
    /// Temporal `ReSTIR` reuse reads the reservoirs this renderer kept from the *previous*
    /// call, at the same pixel and with no motion-vector reprojection: it is a no-op on the
    /// first call and assumes a static camera afterwards.
    pub fn render(&mut self, cameras: &[CameraView]) -> Result<Atlas<'gpu>, RenderError> {
        self.validate(cameras)?;
        let px = self.layout.pixels();
        let groups = [
            self.layout.width.div_ceil(WORKGROUP),
            self.layout.height.div_ceil(WORKGROUP),
            1,
        ];

        let history_valid = self.prepare_history(cameras, px)?;
        let params_f = self.params(cameras, &history_valid);
        let params_bytes: Vec<u8> = params_f.iter().flat_map(|f| f.to_le_bytes()).collect();
        grow(self.gpu, &mut self.params_buf, params_bytes.len() as u64)?;
        self.params_buf.upload(&params_bytes)?;
        grow(self.gpu, &mut self.hit_tri, px * 4)?;
        let params = &self.params_buf;
        let hit_tri = &self.hit_tri;

        // The four channel buffers are still per-frame: `Atlas` owns them and hands them to
        // the caller, and making it borrow the renderer instead would put a second lifetime
        // on a type every GPU test names. At 0.33 ms of a 123 ms frame that is not where the
        // time is (`docs/design/renderer.md` section 8.1).
        let color_words = match self.cfg.path {
            RenderPath::Rs => 1,
            RenderPath::Pt { .. } => 3,
        };
        let mut color = self.new_buffer(px * color_words)?;
        let depth = self.new_buffer(px)?;
        let seg = self.new_buffer(px)?;
        let normal = self.new_buffer(px * 3)?;
        // The `Pt` path's packed `RGBA8` target (packet M7/R3). Allocated whenever the path
        // is `Pt`, because `pt.slang` declares the binding for both its entry points; only
        // the `tonemap` dispatch writes it, and only when `Rgb8` was asked for.
        let pt_rgb8 = match self.cfg.path {
            RenderPath::Rs => None,
            RenderPath::Pt { .. } => Some(self.new_buffer(px)?),
        };
        // The `History` channel of packet M7/R4: the `n` this frame ended at, one word per
        // pixel. Written by the tracer whether or not a history is kept — without one it is
        // 1, which is the truth.
        let pt_history = match self.cfg.path {
            RenderPath::Rs => None,
            RenderPath::Pt { .. } => Some(self.new_buffer(px)?),
        };
        let history = &self.history;

        self.pipelines.primary.reset_descriptors()?;
        let mut rec = CommandRecorder::new(self.gpu)?;
        match (&pt_rgb8, &pt_history) {
            (Some(rgb8), Some(hist_out)) => rec.dispatch(
                &self.pipelines.primary,
                &[
                    params, &self.tris, &color, &depth, &seg, &normal, hit_tri, rgb8, history,
                    hist_out,
                ],
                groups,
            )?,
            _ => rec.dispatch(
                &self.pipelines.primary,
                &[params, &self.tris, &color, &depth, &seg, &normal, hit_tri],
                groups,
            )?,
        }

        // The variance estimate, after every pixel's accumulated colour is written (its
        // short-history fallback reads a 7x7 neighbourhood) and before the filter reads it.
        let mut variance = None;
        if let Some(accumulate) = &self.pipelines.accumulate {
            accumulate.reset_descriptors()?;
            let var = self.new_buffer(px)?;
            rec.dispatch(
                accumulate,
                &[params, &self.tris, history, &color, &depth, &normal, &var],
                groups,
            )?;
            variance = Some(var);
        }

        // `ReSTIR` DI: initial -> temporal -> spatial, one buffer written per pass. `spatial`
        // writes into the persistent previous-frame buffer, which `temporal` has already
        // read by then (the recorder puts a full barrier between dispatches).
        let mut reservoirs = None;
        // Buffers the recorded command buffer still points at. They must outlive
        // `submit_and_wait`: dropping a `Buffer` destroys the Vulkan handle, and a dispatch
        // reading a destroyed buffer reads whatever the allocator handed out next.
        let mut keep_alive: Vec<Buffer<'gpu>> = Vec::new();
        let recycled = self.prev_reservoirs.take();
        if let Some(passes) = &self.pipelines.restir {
            let res_a = self.new_buffer(px * RES_STRIDE)?;
            let res_b = self.new_buffer(px * RES_STRIDE)?;
            let prev = match recycled {
                Some(b) if b.size() == px * RES_STRIDE * 4 => b,
                // A fresh reservoir buffer is explicitly zeroed: device-local memory is
                // uninitialised, and `M > 0` garbage would make the temporal pass reuse a
                // sample that never existed on frame 1.
                _ => {
                    let mut b = self.new_buffer(px * RES_STRIDE)?;
                    b.upload(&vec![0u8; (px * RES_STRIDE * 4) as usize])?;
                    b
                }
            };
            for p in passes {
                p.reset_descriptors()?;
            }
            let g = [params, &self.tris];
            rec.dispatch(
                &passes[0],
                &[
                    g[0], g[1], &res_a, &res_a, &prev, &depth, &normal, hit_tri, &color,
                ],
                groups,
            )?;
            rec.dispatch(
                &passes[1],
                &[
                    g[0], g[1], &res_b, &res_a, &prev, &depth, &normal, hit_tri, &color,
                ],
                groups,
            )?;
            rec.dispatch(
                &passes[2],
                &[
                    g[0], g[1], &prev, &res_b, &prev, &depth, &normal, hit_tri, &color,
                ],
                groups,
            )?;
            reservoirs = Some(prev);
            keep_alive.push(res_a);
            keep_alive.push(res_b);
        }

        // SVGF a-trous, ping-ponging between `color` and a scratch buffer — and, since packet
        // M7/R4, between two variance buffers alongside it with the squared weights.
        let mut scratch = None;
        if let Some(svgf) = &self.pipelines.svgf {
            svgf.reset_descriptors()?;
            let mut other = self.new_buffer(px * 3)?;
            let mut var_a = match variance.take() {
                Some(v) => v,
                None => self.new_buffer(px)?,
            };
            let mut var_b = self.new_buffer(px)?;
            let mut iters = Vec::new();
            for it in 0..self.cfg.svgf_iterations {
                let mut b = self.new_buffer(1)?;
                b.upload(&(1u32 << it).to_le_bytes())?;
                iters.push(b);
            }
            let mut swapped = false;
            for iter_buf in &iters {
                let (src, dst) = if swapped {
                    (&other, &color)
                } else {
                    (&color, &other)
                };
                let (vsrc, vdst) = if swapped {
                    (&var_b, &var_a)
                } else {
                    (&var_a, &var_b)
                };
                rec.dispatch(
                    svgf,
                    &[
                        params, &self.tris, src, dst, &depth, &normal, iter_buf, vsrc, vdst,
                    ],
                    groups,
                )?;
                swapped = !swapped;
            }
            if swapped {
                std::mem::swap(&mut color, &mut other);
                std::mem::swap(&mut var_a, &mut var_b);
            }
            scratch = Some((other, iters, var_a, var_b));
        }

        // Recorded last, after the swap above, so it reads whichever buffer holds the final
        // radiance (spec 3.4: the recorded order *is* the execution order).
        if let (Some(tm), Some(rgb8), Some(hist_out)) =
            (&self.pipelines.tonemap, &pt_rgb8, &pt_history)
        {
            tm.reset_descriptors()?;
            rec.dispatch(
                tm,
                &[
                    params, &self.tris, &color, &depth, &seg, &normal, hit_tri, rgb8, history,
                    hist_out,
                ],
                groups,
            )?;
        }

        rec.submit_and_wait()?;
        drop(scratch);
        drop(variance);
        drop(keep_alive);
        self.prev_reservoirs = reservoirs;
        self.frame = self.frame.wrapping_add(1);

        let mut channels: BTreeMap<Channel, Buffer<'gpu>> = BTreeMap::new();
        let wanted: BTreeSet<Channel> = self.cfg.channels.clone();
        let mut put = |ch: Channel, b: Buffer<'gpu>| {
            if wanted.contains(&ch) {
                channels.insert(ch, b);
            }
        };
        match self.cfg.path {
            RenderPath::Rs => put(Channel::Rgb8, color),
            RenderPath::Pt { .. } => {
                put(Channel::PtRadiance, color);
                if let Some(rgb8) = pt_rgb8 {
                    put(Channel::Rgb8, rgb8);
                }
                if let Some(hist_out) = pt_history {
                    put(Channel::History, hist_out);
                }
            }
        }
        put(Channel::Depth32 { unit_m: 1.0 }, depth);
        put(Channel::SegmentationId, seg);
        put(Channel::Normal, normal);

        Ok(Atlas {
            layout: self.layout,
            n_views: cameras.len() as u32,
            channels,
        })
    }
}
