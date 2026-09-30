//! `<texture>` and `<material>` beyond their names (plan H, packet HT1).
//!
//! Plan H's HT2 adds the `normal` and `emissive` layer roles: a tangent-space normal map, and
//! an emitted colour map scaled by `emission` (1 when it is not written) — not by `rgba`.
//!
//! A texture's declaration becomes a [`TextureSpec`] on the scene; its texels are decoded by
//! [`crate::mesh::load`]. A material becomes a [`Material`] only when it changes how a geom is
//! drawn: it names a texture (attribute or `<layer>`) or writes `specular`, `shininess`,
//! `metallic`, `roughness` or `emission` explicitly. A material with nothing but `rgba` is left
//! as it was before this packet — a name the geom references and nothing more — which is what
//! keeps every committed scene's digest and every committed frame where they were.

use es_core::StableId;
use roxmltree::Node;

use super::attrs::Attrs;
use super::elements::join;
use super::{MjcfError, Parser};
use crate::scene::Material;
use crate::texture::{Builtin, ColorSpace, Mark, TexKind, Texture, TextureSpec};

/// Attributes whose explicit presence makes a material a drawn one.
const DRAWN: [&str; 6] = [
    "texture",
    "specular",
    "shininess",
    "metallic",
    "roughness",
    "emission",
];

/// The cube's six single-face files, in face order.
const CUBE_FILES: [&str; 6] = [
    "fileright",
    "fileleft",
    "fileup",
    "filedown",
    "filefront",
    "fileback",
];

/// Which slot of a [`Material`] a texture name fills.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Role {
    Rgb,
    Orm,
    Metallic,
    Roughness,
    /// Plan H, HT2.
    Normal,
    Emissive,
}

/// A material's texture reference, resolved once every `<asset>` section is read.
#[derive(Debug)]
pub(crate) struct TextureRef {
    material: StableId,
    role: Role,
    name: String,
    line: u32,
    tag: String,
}

fn pick<T: Copy>(
    attrs: &Attrs<'_>,
    attr: &'static str,
    table: &[(&str, T)],
) -> Result<Option<T>, MjcfError> {
    let Some(text) = attrs.get(attr) else {
        return Ok(None);
    };
    table
        .iter()
        .find(|(k, _)| *k == text)
        .map(|(_, v)| Some(*v))
        .ok_or_else(|| attrs.bad(attr, text))
}

impl<'a> Parser<'a> {
    pub(super) fn parse_texture(
        &mut self,
        attrs: &Attrs<'a>,
        id: StableId,
    ) -> Result<(), MjcfError> {
        let dir = self.compiler.texturedir.clone();
        let mut spec = TextureSpec::default();
        let kinds = [
            ("2d", TexKind::TwoD),
            ("cube", TexKind::Cube),
            ("skybox", TexKind::Skybox),
        ];
        spec.kind = pick(attrs, "type", &kinds)?.unwrap_or(spec.kind);
        let spaces = [
            ("auto", ColorSpace::Auto),
            ("sRGB", ColorSpace::Srgb),
            ("linear", ColorSpace::Linear),
        ];
        spec.colorspace = pick(attrs, "colorspace", &spaces)?.unwrap_or(spec.colorspace);
        let builtins = [
            ("none", Builtin::None),
            ("gradient", Builtin::Gradient),
            ("checker", Builtin::Checker),
            ("flat", Builtin::Flat),
        ];
        spec.builtin = pick(attrs, "builtin", &builtins)?.unwrap_or(spec.builtin);
        let marks = [
            ("none", Mark::None),
            ("edge", Mark::Edge),
            ("cross", Mark::Cross),
            ("random", Mark::Random),
        ];
        spec.mark = pick(attrs, "mark", &marks)?.unwrap_or(spec.mark);
        spec.rgb1 = attrs.fixed::<3>("rgb1")?.unwrap_or(spec.rgb1);
        spec.rgb2 = attrs.fixed::<3>("rgb2")?.unwrap_or(spec.rgb2);
        spec.markrgb = attrs.fixed::<3>("markrgb")?.unwrap_or(spec.markrgb);
        spec.random = attrs.num_or("random", spec.random)?;
        spec.width = attrs.int_or("width", 0)?;
        spec.height = attrs.int_or("height", 0)?;
        spec.file = attrs.get("file").map(|f| join(&dir, f));
        if let Some([r, c]) = attrs.fixed::<2>("gridsize")? {
            if r < 1.0 || c < 1.0 || r.fract() != 0.0 || c.fract() != 0.0 {
                return Err(attrs.bad("gridsize", attrs.get("gridsize").unwrap_or_default()));
            }
            spec.gridsize = [r as u32, c as u32];
        }
        if let Some(layout) = attrs.get("gridlayout") {
            if layout.chars().count() != (spec.gridsize[0] * spec.gridsize[1]) as usize {
                return Err(attrs.bad("gridlayout", layout));
            }
            layout.clone_into(&mut spec.gridlayout);
        }
        for (slot, attr) in spec.cubefiles.iter_mut().zip(CUBE_FILES) {
            *slot = attrs.get(attr).map(|f| join(&dir, f));
        }
        self.scene.textures.insert(id, Texture { spec, data: None });
        Ok(())
    }

    pub(super) fn parse_material(
        &mut self,
        node: Node<'a, 'a>,
        attrs: &Attrs<'a>,
        id: StableId,
    ) -> Result<(), MjcfError> {
        let layers: Vec<Node<'a, 'a>> = node
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "layer")
            .collect();
        if !layers.is_empty() && attrs.has("texture") {
            return Err(MjcfError::Unsupported {
                line: attrs.line(),
                what: "a <material> with both a `texture` attribute and <layer> children"
                    .to_owned(),
            });
        }
        if layers.is_empty() && !DRAWN.iter().any(|a| attrs.has(a)) {
            return Ok(()); // rgba only: drawn as the geom's own colour, exactly as before
        }
        let finite = |v: Option<f64>| v.filter(|x| *x >= 0.0);
        let mut material = Material {
            rgba: attrs.padded("rgba", [1.0; 4])?,
            emission: attrs.num_or("emission", 0.0)?,
            specular: attrs.num("specular")?,
            shininess: attrs.num("shininess")?,
            // `MuJoCo` writes -1 for "unset".
            metallic: finite(attrs.num("metallic")?),
            roughness: finite(attrs.num("roughness")?),
            texrepeat: attrs.fixed::<2>("texrepeat")?.unwrap_or([1.0, 1.0]),
            texuniform: attrs.flag_or("texuniform", false)?,
            ..Material::default()
        };
        let mut refs = Vec::new();
        if let Some(name) = attrs.get("texture") {
            refs.push((Role::Rgb, name.to_owned(), attrs.line()));
        }
        for layer in layers {
            let line = self.line(layer);
            let role = match layer.attribute("role") {
                Some("rgb") => Role::Rgb,
                Some("orm") => Role::Orm,
                Some("metallic") => Role::Metallic,
                Some("roughness") => Role::Roughness,
                Some("normal") => Role::Normal,
                Some("emissive") => Role::Emissive,
                other => {
                    return Err(MjcfError::Unsupported {
                        line,
                        what: format!("<layer role=\"{}\">", other.unwrap_or_default()),
                    })
                }
            };
            let Some(name) = layer.attribute("texture") else {
                return Err(MjcfError::MissingAttr {
                    line,
                    element: "layer".to_owned(),
                    attr: "texture",
                });
            };
            if matches!(role, Role::Emissive) {
                // The map is the emitted colour, scaled by `emission` when it is written.
                let e = attrs.num("emission")?.unwrap_or(1.0);
                material.emissive = Some([e; 3]);
            }
            refs.push((role, name.to_owned(), line));
        }
        for (role, name, line) in refs {
            self.pending_textures.push(TextureRef {
                material: id,
                role,
                name,
                line,
                tag: attrs.tag().to_owned(),
            });
        }
        self.scene.materials.insert(id, material);
        Ok(())
    }

    /// Fills every drawn material's texture slots, or names the texture that does not exist.
    pub(super) fn resolve_material_textures(&mut self) -> Result<(), MjcfError> {
        for r in std::mem::take(&mut self.pending_textures) {
            let Some(tex) = self.names.textures.get(&r.name).copied() else {
                return Err(MjcfError::UnknownRef {
                    line: r.line,
                    element: r.tag,
                    attr: "texture",
                    kind: "texture",
                    name: r.name,
                });
            };
            let Some(m) = self.scene.materials.get_mut(&r.material) else {
                continue;
            };
            let slot = match r.role {
                Role::Rgb => &mut m.rgb,
                Role::Orm => &mut m.orm,
                Role::Metallic => &mut m.metallic_map,
                Role::Roughness => &mut m.roughness_map,
                Role::Normal => &mut m.normal_map,
                Role::Emissive => &mut m.emissive_map,
            };
            *slot = Some(tex);
        }
        Ok(())
    }
}
