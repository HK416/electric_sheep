//! MJCF (`MuJoCo` XML) to [`SceneDesc`] (P32, P33).
//!
//! Text in, scene out: no file is opened here. `<asset>` entries and mesh references become
//! [`AssetRef`]s whose hash covers the *path*, and P31 / P34 / P35 replace that with the
//! content hash once the files are actually read. `<include>` is rejected rather than guessed
//! at, for the same reason.
//!
//! What is covered: `<compiler>` (angle, eulerseq, coordinate, meshdir, texturedir,
//! autolimits, inertiafromgeom), `<option>` (timestep, gravity, integrator, cone, jacobian,
//! solver, iterations, impratio), `<default>` classes including nesting, `class=` and
//! `childclass=`, `<worldbody>` recursion over `<body>`, `<geom>`, `<joint>`, `<freejoint>`,
//! `<site>`, `<inertial>`, `<camera>`, all five orientation spellings (see [`orient`]),
//! `<actuator>`, `<sensor>`, `<tendon>` and `<asset>`.
//!
//! Everything else — `<equality>`, `<contact>`, `<keyframe>`, `<visual>`, unknown elements and
//! unknown attributes — becomes a [`Warning`]. Nothing is dropped in silence.
//!
//! The `<worldbody>` itself becomes a body named `world` (`MuJoCo`'s body 0), so world-level
//! geoms, sites and cameras have an owner like every other element.

mod attrs;
mod elements;
mod orient;

use std::collections::BTreeMap;

use es_core::StableId;
use es_math::units::DEG_TO_RAD;
use es_math::{Inertia, Pose, Quat, Vec3};
use roxmltree::{Document, Node};
use thiserror::Error;

use crate::scene::{
    scene_id, Body, BodyInertial, Camera, FrictionCone, Geom, Integrator, Jacobian, Joint,
    JointKind, PhysicsOptions, SceneDesc, SceneError, Shape, Site, Solver,
};
use attrs::Attrs;

/// Something the source file says that the scene description cannot carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning {
    pub line: u32,
    pub message: String,
}

/// A parsed model plus everything the parser could not represent.
#[derive(Debug)]
pub struct Import {
    pub scene: SceneDesc,
    pub warnings: Vec<Warning>,
}

/// Why an MJCF file could not be turned into a [`SceneDesc`]. Every variant carries the line.
#[derive(Debug, Error)]
pub enum MjcfError {
    #[error("XML: {0}")]
    Xml(#[from] roxmltree::Error),
    #[error("line {line}: root element must be <mujoco>, found <{found}>")]
    NotMjcf { line: u32, found: String },
    #[error("line {line}: <include> is not supported yet (M1)")]
    Include { line: u32 },
    #[error("line {line}: {what} is not supported")]
    Unsupported { line: u32, what: String },
    #[error("line {line}: <{element}> attribute `{attr}` has invalid value `{value}`")]
    BadAttr {
        line: u32,
        element: String,
        attr: &'static str,
        value: String,
    },
    #[error("line {line}: <{element}> is missing required attribute `{attr}`")]
    MissingAttr {
        line: u32,
        element: String,
        attr: &'static str,
    },
    #[error("line {line}: <{element}> `{attr}` references unknown {kind} `{name}`")]
    UnknownRef {
        line: u32,
        element: String,
        attr: &'static str,
        kind: &'static str,
        name: String,
    },
    #[error("line {line}: unknown default class `{class}`")]
    UnknownClass { line: u32, class: String },
    #[error("line {line}: unsupported eulerseq `{seq}`")]
    BadEulerSeq { line: u32, seq: String },
    /// The file parsed but the scene it describes is malformed — duplicate names, for one.
    #[error("{0}")]
    Scene(#[from] SceneError),
}

/// Parses MJCF text into a [`SceneDesc`]. The scene is validated before it is returned.
pub fn parse_str(xml: &str) -> Result<Import, MjcfError> {
    let doc = Document::parse(xml)?;
    let root = doc.root_element();
    let mut parser = Parser::new(&doc);
    if root.tag_name().name() != "mujoco" {
        return Err(MjcfError::NotMjcf {
            line: parser.line(root),
            found: root.tag_name().name().to_owned(),
        });
    }
    root.attribute("model")
        .unwrap_or("mujoco")
        .clone_into(&mut parser.scene.name);
    parser.run(root)?;
    parser.scene.validate()?;
    Ok(Import {
        scene: parser.scene,
        warnings: parser.warnings,
    })
}

/// `<compiler>` settings that change how later elements are read.
#[derive(Debug)]
struct Compiler {
    /// Multiplier from file angle units to radians.
    angle_scale: f64,
    eulerseq: String,
    meshdir: String,
    texturedir: String,
    autolimits: bool,
    /// `None` is `auto`: use `<inertial>` when present.
    inertiafromgeom: Option<bool>,
}

impl Default for Compiler {
    fn default() -> Self {
        Self {
            angle_scale: DEG_TO_RAD,
            eulerseq: "xyz".to_owned(),
            meshdir: String::new(),
            texturedir: String::new(),
            autolimits: true,
            inertiafromgeom: None,
        }
    }
}

/// One `<default>` class: its parent class and its per-element attribute values.
#[derive(Debug, Default)]
struct Class<'a> {
    parent: Option<&'a str>,
    attrs: BTreeMap<&'a str, BTreeMap<&'a str, &'a str>>,
}

/// Name to id lookup, one map per MJCF namespace (they do not share a namespace).
#[derive(Debug, Default)]
pub(crate) struct Names {
    pub(crate) bodies: BTreeMap<String, StableId>,
    pub(crate) joints: BTreeMap<String, StableId>,
    pub(crate) geoms: BTreeMap<String, StableId>,
    pub(crate) sites: BTreeMap<String, StableId>,
    pub(crate) cameras: BTreeMap<String, StableId>,
    pub(crate) tendons: BTreeMap<String, StableId>,
    pub(crate) actuators: BTreeMap<String, StableId>,
    pub(crate) meshes: BTreeMap<String, StableId>,
    pub(crate) materials: BTreeMap<String, StableId>,
    pub(crate) hfields: BTreeMap<String, StableId>,
}

pub(crate) struct Parser<'a> {
    doc: &'a Document<'a>,
    compiler: Compiler,
    classes: BTreeMap<&'a str, Class<'a>>,
    pub(crate) scene: SceneDesc,
    pub(crate) warnings: Vec<Warning>,
    pub(crate) names: Names,
    /// Per-owner counters for elements the file leaves unnamed.
    counters: BTreeMap<String, u32>,
}

impl std::fmt::Debug for Parser<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parser")
            .field("compiler", &self.compiler)
            .field("warnings", &self.warnings.len())
            .finish_non_exhaustive()
    }
}

/// Root children in the order they must be read: `<compiler>` changes angle units, classes
/// must exist before an element names one, assets before a geom references one, bodies before
/// a tendon or sensor references one.
const ROOT_ORDER: [&str; 8] = [
    "compiler",
    "option",
    "asset",
    "default",
    "worldbody",
    "tendon",
    "actuator",
    "sensor",
];

/// Root children that are valid MJCF but have no place in [`SceneDesc`] yet.
const ROOT_IGNORED: [&str; 8] = [
    "size",
    "visual",
    "statistic",
    "custom",
    "extension",
    "equality",
    "contact",
    "keyframe",
];

impl<'a> Parser<'a> {
    fn new(doc: &'a Document<'a>) -> Self {
        let mut classes = BTreeMap::new();
        classes.insert("main", Class::default());
        Self {
            doc,
            compiler: Compiler::default(),
            classes,
            scene: SceneDesc::default(),
            warnings: Vec::new(),
            names: Names::default(),
            counters: BTreeMap::new(),
        }
    }

    pub(crate) fn line(&self, node: Node<'a, 'a>) -> u32 {
        self.doc.text_pos_at(node.range().start).row
    }

    pub(crate) fn warn(&mut self, line: u32, message: String) {
        self.warnings.push(Warning { line, message });
    }

    fn run(&mut self, root: Node<'a, 'a>) -> Result<(), MjcfError> {
        for child in root.children().filter(Node::is_element) {
            let tag = child.tag_name().name();
            if tag == "include" {
                return Err(MjcfError::Include {
                    line: self.line(child),
                });
            }
            if !ROOT_ORDER.contains(&tag) {
                let line = self.line(child);
                let reason = if ROOT_IGNORED.contains(&tag) {
                    "is not represented in SceneDesc"
                } else {
                    "is not a known MJCF element"
                };
                self.warn(line, format!("<{tag}> {reason}, ignored"));
            }
        }
        for wanted in ROOT_ORDER {
            for child in root
                .children()
                .filter(|n| n.is_element() && n.tag_name().name() == wanted)
            {
                match wanted {
                    "compiler" => self.parse_compiler(child)?,
                    "option" => self.parse_option(child)?,
                    "asset" => self.parse_asset(child)?,
                    "default" => self.parse_default(child, None)?,
                    "worldbody" => self.parse_worldbody(child)?,
                    "tendon" => self.parse_tendon(child)?,
                    "actuator" => self.parse_actuator(child)?,
                    _ => self.parse_sensor(child)?,
                }
            }
        }
        Ok(())
    }

    // ---- element plumbing -------------------------------------------------------------

    /// Builds the attribute view of `node`, merging in its default class.
    pub(crate) fn element(
        &self,
        node: Node<'a, 'a>,
        childclass: &'a str,
    ) -> Result<Attrs<'a>, MjcfError> {
        let tag = node.tag_name().name();
        let line = self.line(node);
        let class = node.attribute("class").unwrap_or(childclass);
        let defaults = self.resolve(tag, class, line)?;
        let attrs = Attrs::new(node, tag, line, defaults);
        let _ = attrs.own("class");
        Ok(attrs)
    }

    /// Attribute values `class` inherits for element `tag`: root class first, `class` last.
    fn resolve(
        &self,
        tag: &'a str,
        class: &'a str,
        line: u32,
    ) -> Result<BTreeMap<&'a str, &'a str>, MjcfError> {
        let mut chain = Vec::new();
        let mut cursor = Some(class);
        while let Some(name) = cursor {
            let def = self
                .classes
                .get(name)
                .ok_or_else(|| MjcfError::UnknownClass {
                    line,
                    class: name.to_owned(),
                })?;
            chain.push(def);
            cursor = def.parent;
        }
        let mut merged = BTreeMap::new();
        for def in chain.iter().rev() {
            if let Some(values) = def.attrs.get(tag) {
                merged.extend(values.iter().map(|(k, v)| (*k, *v)));
            }
        }
        Ok(merged)
    }

    /// The element's own `name`, or a generated `<tag><n>` counted per owner.
    pub(crate) fn local_name(&mut self, attrs: &Attrs<'a>, owner: &str) -> String {
        if let Some(name) = attrs.get("name") {
            return name.to_owned();
        }
        let tag = attrs.tag();
        let counter = self.counters.entry(format!("{owner}/{tag}")).or_insert(0);
        *counter += 1;
        format!("{tag}{counter}")
    }

    pub(crate) fn pose(&self, attrs: &Attrs<'a>) -> Result<Pose, MjcfError> {
        let position = attrs.vec3("pos", Vec3::ZERO)?;
        let orientation = orient::read(attrs, self.compiler.angle_scale, &self.compiler.eulerseq)?;
        Ok(Pose::new(position, orientation))
    }

    /// MJCF `*limited` / `autolimits` rule: an explicit `false` wins, an explicit `true` means
    /// the range applies, and `auto` follows `<compiler autolimits>`.
    pub(crate) fn limit(
        &self,
        attrs: &Attrs<'a>,
        limited: &'static str,
        range: &'static str,
        scale: f64,
    ) -> Result<Option<(f64, f64)>, MjcfError> {
        let values = attrs.range(range)?.map(|(lo, hi)| (lo * scale, hi * scale));
        Ok(match attrs.tristate(limited)? {
            Some(true) => Some(values.unwrap_or((0.0, 0.0))),
            None if self.compiler.autolimits => values,
            _ => None,
        })
    }

    // ---- <compiler> / <option> --------------------------------------------------------

    fn parse_compiler(&mut self, node: Node<'a, 'a>) -> Result<(), MjcfError> {
        let attrs = self.element(node, "main")?;
        self.compiler.angle_scale = match attrs.get_or("angle", "degree") {
            "degree" => DEG_TO_RAD,
            "radian" => 1.0,
            other => return Err(attrs.bad("angle", other)),
        };
        if let Some(coordinate) = attrs.get("coordinate") {
            if coordinate != "local" {
                return Err(MjcfError::Unsupported {
                    line: attrs.line(),
                    what: format!("<compiler coordinate=\"{coordinate}\">"),
                });
            }
        }
        self.compiler.eulerseq = attrs.get_or("eulerseq", "xyz").to_owned();
        self.compiler.meshdir = attrs
            .get("meshdir")
            .or_else(|| attrs.get("assetdir"))
            .unwrap_or_default()
            .to_owned();
        self.compiler.texturedir = attrs
            .get("texturedir")
            .or_else(|| attrs.get("assetdir"))
            .unwrap_or_default()
            .to_owned();
        self.compiler.autolimits = attrs.flag_or("autolimits", true)?;
        self.compiler.inertiafromgeom = attrs.tristate("inertiafromgeom")?;
        attrs.report_unknown(&mut self.warnings);
        Ok(())
    }

    fn parse_option(&mut self, node: Node<'a, 'a>) -> Result<(), MjcfError> {
        let attrs = self.element(node, "main")?;
        let defaults = PhysicsOptions::default();
        let options = PhysicsOptions {
            timestep: attrs.num_or("timestep", defaults.timestep)?,
            gravity: attrs.vec3("gravity", defaults.gravity)?,
            integrator: match attrs.get_or("integrator", "Euler") {
                "Euler" => Integrator::Euler,
                "RK4" => Integrator::Rk4,
                "implicit" => Integrator::Implicit,
                "implicitfast" => Integrator::ImplicitFast,
                other => return Err(attrs.bad("integrator", other)),
            },
            cone: match attrs.get_or("cone", "pyramidal") {
                "pyramidal" => FrictionCone::Pyramidal,
                "elliptic" => FrictionCone::Elliptic,
                other => return Err(attrs.bad("cone", other)),
            },
            jacobian: match attrs.get_or("jacobian", "auto") {
                "auto" => Jacobian::Auto,
                "dense" => Jacobian::Dense,
                "sparse" => Jacobian::Sparse,
                other => return Err(attrs.bad("jacobian", other)),
            },
            solver: match attrs.get_or("solver", "Newton") {
                "Newton" => Solver::Newton,
                "CG" => Solver::Cg,
                "PGS" => Solver::Pgs,
                other => return Err(attrs.bad("solver", other)),
            },
            iterations: attrs.int_or("iterations", defaults.iterations)?,
            impratio: attrs.num_or("impratio", defaults.impratio)?,
        };
        self.scene.options = options;
        for child in node.children().filter(Node::is_element) {
            let line = self.line(child);
            let tag = child.tag_name().name().to_owned();
            self.warn(line, format!("<option><{tag}> is not represented, ignored"));
        }
        attrs.report_unknown(&mut self.warnings);
        Ok(())
    }

    // ---- <default> --------------------------------------------------------------------

    fn parse_default(
        &mut self,
        node: Node<'a, 'a>,
        parent: Option<&'a str>,
    ) -> Result<(), MjcfError> {
        let name = node.attribute("class").unwrap_or("main");
        // A top-level unnamed <default> *is* the main class; a named one derives from it.
        let parent = if name == "main" {
            None
        } else {
            parent.or(Some("main"))
        };
        let mut class = Class {
            parent,
            attrs: BTreeMap::new(),
        };
        let mut nested = Vec::new();
        for child in node.children().filter(Node::is_element) {
            let tag = child.tag_name().name();
            if tag == "default" {
                nested.push(child);
                continue;
            }
            let values: BTreeMap<&'a str, &'a str> = child
                .attributes()
                .map(|a| (a.name(), a.value()))
                .filter(|(k, _)| *k != "class")
                .collect();
            class.attrs.entry(tag).or_default().extend(values);
        }
        self.classes.entry(name).or_default();
        let entry = self.classes.get_mut(name).expect("just inserted");
        entry.parent = class.parent;
        for (tag, values) in class.attrs {
            entry.attrs.entry(tag).or_default().extend(values);
        }
        for child in nested {
            self.parse_default(child, Some(name))?;
        }
        Ok(())
    }

    // ---- <worldbody> ------------------------------------------------------------------

    fn parse_worldbody(&mut self, node: Node<'a, 'a>) -> Result<(), MjcfError> {
        let mut world = Body {
            id: scene_id("body", "world"),
            name: "world".to_owned(),
            parent: None,
            pose: Pose::IDENTITY,
            inertial: None,
            geoms: Vec::new(),
            sites: Vec::new(),
        };
        let id = world.id;
        let childclass = node.attribute("childclass").unwrap_or("main");
        let nested = self.parse_body_children(node, &mut world, "world", childclass)?;
        self.names.bodies.insert("world".to_owned(), id);
        self.scene.bodies.push(world);
        for child in nested {
            self.parse_body(child, id, "world", childclass)?;
        }
        Ok(())
    }

    fn parse_body(
        &mut self,
        node: Node<'a, 'a>,
        parent: StableId,
        parent_path: &str,
        childclass: &'a str,
    ) -> Result<(), MjcfError> {
        let attrs = self.element(node, childclass)?;
        let name = self.local_name(&attrs, parent_path);
        let path = format!("{parent_path}/{name}");
        let id = scene_id("body", &path);
        let mut body = Body {
            id,
            name: name.clone(),
            parent: Some(parent),
            pose: self.pose(&attrs)?,
            inertial: None,
            geoms: Vec::new(),
            sites: Vec::new(),
        };
        let childclass = attrs.own("childclass").unwrap_or(childclass);
        attrs.report_unknown(&mut self.warnings);
        let nested = self.parse_body_children(node, &mut body, &path, childclass)?;
        self.names.bodies.insert(name, id);
        self.scene.bodies.push(body);
        for child in nested {
            self.parse_body(child, id, &path, childclass)?;
        }
        Ok(())
    }

    /// Fills `body` from its child elements and returns the nested `<body>` nodes, which the
    /// caller parses once `body` itself is in the scene.
    fn parse_body_children(
        &mut self,
        node: Node<'a, 'a>,
        body: &mut Body,
        path: &str,
        childclass: &'a str,
    ) -> Result<Vec<Node<'a, 'a>>, MjcfError> {
        let mut nested = Vec::new();
        for child in node.children().filter(Node::is_element) {
            match child.tag_name().name() {
                "include" => {
                    return Err(MjcfError::Include {
                        line: self.line(child),
                    })
                }
                "body" => nested.push(child),
                "geom" => {
                    let geom = self.parse_geom(child, path, childclass)?;
                    body.geoms.push(geom);
                }
                "site" => {
                    let site = self.parse_site(child, path, childclass)?;
                    body.sites.push(site);
                }
                "joint" | "freejoint" => {
                    let joint = self.parse_joint(child, body.id, path, childclass)?;
                    self.scene.joints.push(joint);
                }
                "camera" => {
                    let parent = (body.name != "world").then_some(body.id);
                    let camera = self.parse_camera(child, parent, path, childclass)?;
                    self.scene.cameras.push(camera);
                }
                "inertial" => {
                    body.inertial = self.parse_inertial(child, childclass)?;
                }
                other => {
                    let line = self.line(child);
                    self.warn(line, format!("<{other}> is not represented, ignored"));
                }
            }
        }
        Ok(nested)
    }

    fn parse_geom(
        &mut self,
        node: Node<'a, 'a>,
        body_path: &str,
        childclass: &'a str,
    ) -> Result<Geom, MjcfError> {
        let attrs = self.element(node, childclass)?;
        let name = self.local_name(&attrs, body_path);
        let id = scene_id("geom", &format!("{body_path}/{name}"));
        let kind = attrs.get_or("type", "sphere");
        let size = attrs.nums("size")?.unwrap_or_default();
        let fromto = attrs.fixed::<6>("fromto")?;
        let at = |i: usize| -> Result<f64, MjcfError> {
            size.get(i).copied().ok_or_else(|| attrs.missing("size"))
        };
        // `fromto` fixes the pose and the length; otherwise both come from pos/orientation.
        let (pose, half) = match fromto {
            Some([ax, ay, az, bx, by, bz]) => {
                let (a, b) = (Vec3::new(ax, ay, az), Vec3::new(bx, by, bz));
                let delta = b - a;
                let rotation = orient::from_zaxis(delta)
                    .ok_or_else(|| attrs.bad("fromto", attrs.get_or("fromto", "")))?;
                let mid = (a + b).scale(0.5);
                (Pose::new(mid, rotation), Some(delta.norm() * 0.5))
            }
            None => (self.pose(&attrs)?, None),
        };
        let half_length = |i: usize| half.map_or_else(|| at(i), Ok);
        let shape = match kind {
            "plane" => Shape::Plane {
                half_x: size.first().copied().unwrap_or(0.0),
                half_y: size.get(1).copied().unwrap_or(0.0),
                grid: size.get(2).copied().unwrap_or(0.0),
            },
            "sphere" => Shape::Sphere { radius: at(0)? },
            "capsule" => Shape::Capsule {
                radius: at(0)?,
                half_length: half_length(1)?,
            },
            "cylinder" => Shape::Cylinder {
                radius: at(0)?,
                half_length: half_length(1)?,
            },
            "box" => Shape::Box {
                half_extents: Vec3::new(at(0)?, at(1)?, half_length(2)?),
            },
            "ellipsoid" => Shape::Ellipsoid {
                radii: Vec3::new(at(0)?, at(1)?, half_length(2)?),
            },
            "mesh" => {
                let mesh = attrs.get("mesh").ok_or_else(|| attrs.missing("mesh"))?;
                Shape::Mesh {
                    asset: lookup(&self.names.meshes, &attrs, "mesh", "mesh", mesh)?,
                }
            }
            "hfield" => {
                let field = attrs.get("hfield").ok_or_else(|| attrs.missing("hfield"))?;
                Shape::HeightField {
                    asset: lookup(&self.names.hfields, &attrs, "hfield", "hfield", field)?,
                }
            }
            other => return Err(attrs.bad("type", other)),
        };
        let material = match attrs.get("material") {
            None => None,
            Some(name) => Some(lookup(
                &self.names.materials,
                &attrs,
                "material",
                "material",
                name,
            )?),
        };
        let contype = attrs.int_or("contype", 1)?;
        let conaffinity = attrs.int_or("conaffinity", 1)?;
        let geom = Geom {
            id,
            name: name.clone(),
            shape,
            pose,
            friction: attrs.padded("friction", [1.0, 0.005, 0.0001])?,
            contype,
            conaffinity,
            condim: attrs.int_or("condim", 3)?,
            density: attrs.num_or("density", 1000.0)?,
            mass: attrs.num("mass")?,
            margin: attrs.num_or("margin", 0.0)?,
            gap: attrs.num_or("gap", 0.0)?,
            solref: attrs.padded("solref", [0.02, 1.0])?,
            solimp: attrs.padded("solimp", [0.9, 0.95, 0.001, 0.5, 2.0])?,
            material,
            rgba: attrs.padded("rgba", [0.5, 0.5, 0.5, 1.0])?,
            visual_only: contype == 0 && conaffinity == 0,
        };
        attrs.report_unknown(&mut self.warnings);
        self.names.geoms.insert(name, id);
        Ok(geom)
    }

    fn parse_site(
        &mut self,
        node: Node<'a, 'a>,
        body_path: &str,
        childclass: &'a str,
    ) -> Result<Site, MjcfError> {
        let attrs = self.element(node, childclass)?;
        let name = self.local_name(&attrs, body_path);
        let id = scene_id("site", &format!("{body_path}/{name}"));
        let site = Site {
            id,
            name: name.clone(),
            pose: self.pose(&attrs)?,
            size: attrs.vec3("size", Vec3::new(0.005, 0.005, 0.005))?,
        };
        attrs.report_unknown(&mut self.warnings);
        self.names.sites.insert(name, id);
        Ok(site)
    }

    fn parse_joint(
        &mut self,
        node: Node<'a, 'a>,
        body: StableId,
        body_path: &str,
        childclass: &'a str,
    ) -> Result<Joint, MjcfError> {
        let attrs = self.element(node, childclass)?;
        let name = self.local_name(&attrs, body_path);
        let id = scene_id("joint", &format!("{body_path}/{name}"));
        let kind = if attrs.tag() == "freejoint" {
            JointKind::Free
        } else {
            match attrs.get_or("type", "hinge") {
                "hinge" => JointKind::Hinge,
                "slide" => JointKind::Slide,
                "ball" => JointKind::Ball,
                "free" => JointKind::Free,
                other => return Err(attrs.bad("type", other)),
            }
        };
        // Hinge and ball limits are angles; slide limits are lengths, which are already SI.
        let scale = match kind {
            JointKind::Hinge | JointKind::Ball => self.compiler.angle_scale,
            _ => 1.0,
        };
        let joint = Joint {
            id,
            name: name.clone(),
            body,
            kind,
            axis: attrs.vec3("axis", Vec3::new(0.0, 0.0, 1.0))?.normalize(),
            anchor: attrs.vec3("pos", Vec3::ZERO)?,
            range: self.limit(&attrs, "limited", "range", scale)?,
            damping: attrs.num_or("damping", 0.0)?,
            armature: attrs.num_or("armature", 0.0)?,
            stiffness: attrs.num_or("stiffness", 0.0)?,
            friction_loss: attrs.num_or("frictionloss", 0.0)?,
            spring_ref: attrs.num_or("springref", 0.0)? * scale,
        };
        attrs.report_unknown(&mut self.warnings);
        self.names.joints.insert(name, id);
        Ok(joint)
    }

    fn parse_camera(
        &mut self,
        node: Node<'a, 'a>,
        body: Option<StableId>,
        body_path: &str,
        childclass: &'a str,
    ) -> Result<Camera, MjcfError> {
        let attrs = self.element(node, childclass)?;
        let name = self.local_name(&attrs, body_path);
        let id = scene_id("camera", &format!("{body_path}/{name}"));
        let camera = Camera {
            id,
            name: name.clone(),
            body,
            pose: self.pose(&attrs)?,
            // MJCF `fovy` is in degrees whatever `<compiler angle>` says; SI storage is rad.
            fovy: attrs.num_or("fovy", 45.0)? * DEG_TO_RAD,
        };
        attrs.report_unknown(&mut self.warnings);
        self.names.cameras.insert(name, id);
        Ok(camera)
    }

    fn parse_inertial(
        &mut self,
        node: Node<'a, 'a>,
        childclass: &'a str,
    ) -> Result<Option<BodyInertial>, MjcfError> {
        let attrs = self.element(node, childclass)?;
        if self.compiler.inertiafromgeom == Some(true) {
            self.warn(
                attrs.line(),
                "<inertial> ignored: <compiler inertiafromgeom=\"true\">".to_owned(),
            );
            return Ok(None);
        }
        let mass = attrs.num("mass")?.ok_or_else(|| attrs.missing("mass"))?;
        let com = attrs.vec3("pos", Vec3::ZERO)?;
        let frame = orient::read(&attrs, self.compiler.angle_scale, &self.compiler.eulerseq)?;
        let (inertia, frame) =
            if let Some([ixx, iyy, izz, ixy, ixz, iyz]) = attrs.fixed::<6>("fullinertia")? {
                // `fullinertia` is already in the body frame, so it carries no separate frame.
                (Inertia::new(ixx, iyy, izz, ixy, ixz, iyz), Quat::IDENTITY)
            } else if let Some([ixx, iyy, izz]) = attrs.fixed::<3>("diaginertia")? {
                (Inertia::diagonal(ixx, iyy, izz), frame)
            } else {
                return Err(attrs.missing("diaginertia"));
            };
        attrs.report_unknown(&mut self.warnings);
        Ok(Some(BodyInertial {
            mass,
            com,
            inertia,
            frame,
        }))
    }
}

/// Resolves a name in one MJCF namespace, or reports it with the referencing element's line.
pub(crate) fn lookup(
    table: &BTreeMap<String, StableId>,
    attrs: &Attrs<'_>,
    attr: &'static str,
    kind: &'static str,
    name: &str,
) -> Result<StableId, MjcfError> {
    table
        .get(name)
        .copied()
        .ok_or_else(|| MjcfError::UnknownRef {
            line: attrs.line(),
            element: attrs.tag().to_owned(),
            attr,
            kind,
            name: name.to_owned(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn include_is_rejected_with_a_line() {
        let err = parse_str("<mujoco>\n  <include file=\"other.xml\"/>\n</mujoco>").unwrap_err();
        assert!(matches!(err, MjcfError::Include { line: 2 }), "{err}");
    }

    #[test]
    fn non_mujoco_root_is_rejected() {
        assert!(matches!(
            parse_str("<robot/>").unwrap_err(),
            MjcfError::NotMjcf { .. }
        ));
    }

    #[test]
    fn a_duplicate_name_fails_validation() {
        let xml = "<mujoco><worldbody><body name=\"b\"/><body name=\"b\"/></worldbody></mujoco>";
        assert!(matches!(
            parse_str(xml).unwrap_err(),
            MjcfError::Scene(crate::scene::SceneError::DuplicateId { .. })
        ));
    }

    #[test]
    fn global_coordinates_are_rejected_not_guessed() {
        let err = parse_str("<mujoco><compiler coordinate=\"global\"/></mujoco>").unwrap_err();
        assert!(matches!(err, MjcfError::Unsupported { .. }), "{err}");
    }
}
