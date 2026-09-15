//! URDF importer with `package://` resolution (P34).
//!
//! Text in, scene out, same contract as [`crate::mjcf`]: no file but the URDF document itself
//! is read here — a `<mesh filename="package://...">` becomes an [`AssetRef`] whose `hash` is
//! `blake3` of the *resolved path string* (P31 / a later packet own the content hash).
//!
//! URDF is already right-handed, Z-up, SI (spec 3.1), so unlike MJCF there is no unit or axis
//! conversion. What is covered: `<link>` (`<inertial>`, repeated `<visual>` / `<collision>`
//! with box/cylinder/sphere/mesh `<geometry>`), `<joint type="revolute|continuous|prismatic|
//! fixed|floating">` (`<origin>`, `<axis>`, `<limit>`, `<dynamics>`), `<transmission>` (one
//! `Actuator` per `<actuator>` child). `planar` joints, `<mimic>`, `<gazebo>` and unknown
//! elements are reported as [`Warning`]s (or, for `planar`, [`UrdfError::Unsupported`]) rather
//! than silently dropped or guessed at.
//!
//! The kinematic tree is exactly the URDF one: every `<link>` is a [`Body`], every `<joint>`
//! points from a parent link to the child it moves. The link that is no joint's child is the
//! root (`Body::parent == None`); zero or more than one such link is a document error, not a
//! guess.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use es_core::StableId;
use es_math::{Inertia, Pose, Quat, Vec3};
use roxmltree::{Document, Node};
use thiserror::Error;

use crate::mjcf::Warning;
use crate::scene::{
    scene_id, Actuator, ActuatorKind, ActuatorTarget, AssetKind, AssetRef, Body, BodyInertial,
    Geom, Joint, JointKind, SceneDesc, SceneError, Shape,
};

/// A parsed model plus everything the parser could not represent. Same shape as
/// [`crate::mjcf::Import`].
#[derive(Debug)]
pub struct Import {
    pub scene: SceneDesc,
    pub warnings: Vec<Warning>,
}

/// Why a URDF document could not be turned into a [`SceneDesc`].
#[derive(Debug, Error)]
pub enum UrdfError {
    #[error("XML: {0}")]
    Xml(#[from] roxmltree::Error),
    #[error("line {line}: root element must be <robot>, found <{found}>")]
    NotUrdf { line: u32, found: String },
    #[error("line {line}: <{element}> is missing required attribute `{attr}`")]
    MissingAttr {
        line: u32,
        element: String,
        attr: &'static str,
    },
    #[error("line {line}: <{element}> attribute `{attr}` has invalid value `{value}`")]
    BadAttr {
        line: u32,
        element: String,
        attr: &'static str,
        value: String,
    },
    #[error("line {line}: <joint> `{joint}` references unknown link `{link}`")]
    UnknownLink {
        line: u32,
        joint: String,
        link: String,
    },
    #[error("line {line}: <transmission> `{name}` references unknown joint `{joint}`")]
    UnknownJoint {
        line: u32,
        name: String,
        joint: String,
    },
    #[error("link `{link}` is the child of more than one joint; the tree must be a tree")]
    MultipleParents { link: String },
    #[error("no root link: every link is some joint's child (the graph has no root, or a cycle)")]
    NoRoot,
    #[error("multiple root links, no joint connects them: {0:?}")]
    MultipleRoots(Vec<String>),
    #[error("line {line}: {what} is not supported")]
    Unsupported { line: u32, what: String },
    #[error("package `{package}` is not known (known packages: {known:?})")]
    UnknownPackage { package: String, known: Vec<String> },
    #[error("{0}")]
    Scene(#[from] SceneError),
}

impl UrdfError {
    fn missing(line: u32, element: &str, attr: &'static str) -> Self {
        Self::MissingAttr {
            line,
            element: element.to_owned(),
            attr,
        }
    }

    fn bad(line: u32, element: &str, attr: &'static str, value: &str) -> Self {
        Self::BadAttr {
            line,
            element: element.to_owned(),
            attr,
            value: value.to_owned(),
        }
    }
}

/// Resolves `package://<pkg>/<rest>` mesh URIs against known package roots. Pure: no filesystem
/// access happens during resolution itself, only (optionally) while building the map.
#[derive(Debug, Default, Clone)]
pub struct PackageResolver {
    pub roots: BTreeMap<String, PathBuf>,
}

const MAX_SCAN_DEPTH: u32 = 8;

impl PackageResolver {
    pub fn new(roots: BTreeMap<String, PathBuf>) -> Self {
        Self { roots }
    }

    /// Scans every directory in `ROS_PACKAGE_PATH` (`;` on Windows, `:` elsewhere) for
    /// subtrees containing a `package.xml`; the package name is that manifest's `<name>`
    /// element. A directory that is itself a package is not searched further — packages do
    /// not nest.
    #[must_use]
    pub fn from_env() -> Self {
        let path = std::env::var("ROS_PACKAGE_PATH").unwrap_or_default();
        let sep = if cfg!(windows) { ';' } else { ':' };
        let mut roots = BTreeMap::new();
        for dir in path.split(sep).filter(|s| !s.is_empty()) {
            scan_dir(Path::new(dir), 0, &mut roots);
        }
        Self { roots }
    }

    /// Resolves a mesh URI. `package://pkg/rest` looks up `pkg` in `roots`; `file://` and
    /// plain relative/absolute paths pass through unchanged.
    pub fn resolve(&self, uri: &str) -> Result<PathBuf, UrdfError> {
        if let Some(rest) = uri.strip_prefix("package://") {
            let (pkg, tail) = rest.split_once('/').unwrap_or((rest, ""));
            let root = self
                .roots
                .get(pkg)
                .ok_or_else(|| UrdfError::UnknownPackage {
                    package: pkg.to_owned(),
                    known: self.roots.keys().cloned().collect(),
                })?;
            Ok(root.join(tail))
        } else if let Some(rest) = uri.strip_prefix("file://") {
            Ok(PathBuf::from(rest))
        } else {
            Ok(PathBuf::from(uri))
        }
    }
}

fn scan_dir(dir: &Path, depth: u32, roots: &mut BTreeMap<String, PathBuf>) {
    if depth > MAX_SCAN_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    if dir.join("package.xml").is_file() {
        if let Some(name) = package_name(&dir.join("package.xml")) {
            roots.entry(name).or_insert_with(|| dir.to_path_buf());
        }
        return;
    }
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, depth + 1, roots);
        }
    }
}

fn package_name(manifest: &Path) -> Option<String> {
    let xml = std::fs::read_to_string(manifest).ok()?;
    let doc = Document::parse(&xml).ok()?;
    doc.descendants()
        .find(|n| n.has_tag_name("name"))
        .and_then(|n| n.text())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

// ---- parsing ---------------------------------------------------------------------------

/// One `<joint>`, fully read before the kinematic tree is assembled.
struct JointDef {
    line: u32,
    name: String,
    kind: JointKind,
    parent: String,
    child: String,
    origin: Pose,
    axis: Vec3,
    range: Option<(f64, f64)>,
    effort: Option<f64>,
    damping: f64,
    friction_loss: f64,
}

/// Parses URDF text into a [`SceneDesc`]. The scene is validated before it is returned.
pub fn parse_urdf(xml: &str, resolver: &PackageResolver) -> Result<Import, UrdfError> {
    let doc = Document::parse(xml)?;
    let root = doc.root_element();
    let line = |n: Node| doc.text_pos_at(n.range().start).row;
    if root.tag_name().name() != "robot" {
        return Err(UrdfError::NotUrdf {
            line: line(root),
            found: root.tag_name().name().to_owned(),
        });
    }

    let mut scene = SceneDesc {
        name: root.attribute("name").unwrap_or("robot").to_owned(),
        ..SceneDesc::default()
    };
    let mut warnings = Vec::new();
    let mut assets_by_path: BTreeMap<String, StableId> = BTreeMap::new();

    let mut links: BTreeMap<String, Node> = BTreeMap::new();
    let mut joint_nodes = Vec::new();
    let mut transmissions = Vec::new();
    for child in root.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "link" => {
                let name = child
                    .attribute("name")
                    .ok_or_else(|| UrdfError::missing(line(child), "link", "name"))?;
                links.insert(name.to_owned(), child);
            }
            "joint" => joint_nodes.push(child),
            "transmission" => transmissions.push(child),
            "material" | "gazebo" => warnings.push(Warning {
                line: line(child),
                message: format!(
                    "<{}> is not represented in SceneDesc, ignored",
                    child.tag_name().name()
                ),
            }),
            other => warnings.push(Warning {
                line: line(child),
                message: format!("<{other}> is not a known URDF element, ignored"),
            }),
        }
    }

    let mut joints = Vec::with_capacity(joint_nodes.len());
    for node in joint_nodes {
        joints.push(parse_joint(node, &doc, &mut warnings)?);
    }

    // Every child link has at most one parent joint: URDF is a tree, not a graph.
    let mut parent_of: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, j) in joints.iter().enumerate() {
        if !links.contains_key(&j.parent) {
            return Err(UrdfError::UnknownLink {
                line: j.line,
                joint: j.name.clone(),
                link: j.parent.clone(),
            });
        }
        if !links.contains_key(&j.child) {
            return Err(UrdfError::UnknownLink {
                line: j.line,
                joint: j.name.clone(),
                link: j.child.clone(),
            });
        }
        if parent_of.insert(&j.child, i).is_some() {
            return Err(UrdfError::MultipleParents {
                link: j.child.clone(),
            });
        }
    }

    let child_names: BTreeSet<&str> = joints.iter().map(|j| j.child.as_str()).collect();
    let roots: Vec<String> = links
        .keys()
        .filter(|name| !child_names.contains(name.as_str()))
        .cloned()
        .collect();
    match roots.len() {
        0 => return Err(UrdfError::NoRoot),
        1 => {}
        _ => return Err(UrdfError::MultipleRoots(roots)),
    }

    for (name, node) in &links {
        let parent_joint = parent_of.get(name.as_str()).map(|&i| &joints[i]);
        let pose = parent_joint.map_or(Pose::IDENTITY, |j| j.origin);
        let parent = parent_joint.map(|j| scene_id("body", &j.parent));
        let inertial = parse_inertial(*node, &doc)?;
        let geoms = parse_geoms(
            *node,
            name,
            &doc,
            resolver,
            &mut assets_by_path,
            &mut scene,
            &mut warnings,
        )?;
        scene.bodies.push(Body {
            id: scene_id("body", name),
            name: name.clone(),
            parent,
            pose,
            inertial,
            geoms,
            sites: Vec::new(),
        });
    }

    let mut effort_by_joint: BTreeMap<&str, f64> = BTreeMap::new();
    for j in &joints {
        if let Some(effort) = j.effort {
            effort_by_joint.insert(&j.name, effort);
        }
        scene.joints.push(Joint {
            id: scene_id("joint", &j.name),
            name: j.name.clone(),
            body: scene_id("body", &j.child),
            kind: j.kind,
            axis: j.axis,
            anchor: Vec3::ZERO,
            range: j.range,
            damping: j.damping,
            armature: 0.0,
            stiffness: 0.0,
            friction_loss: j.friction_loss,
            spring_ref: 0.0,
        });
    }

    for node in transmissions {
        if let Some(actuator) =
            parse_transmission(node, &doc, &joints, &effort_by_joint, &mut warnings)?
        {
            scene.actuators.push(actuator);
        }
    }

    scene.validate()?;
    Ok(Import { scene, warnings })
}

fn parse_joint<'a>(
    node: Node<'a, 'a>,
    doc: &Document<'a>,
    warnings: &mut Vec<Warning>,
) -> Result<JointDef, UrdfError> {
    let line = doc.text_pos_at(node.range().start).row;
    let name = node
        .attribute("name")
        .ok_or_else(|| UrdfError::missing(line, "joint", "name"))?
        .to_owned();
    let kind_str = node
        .attribute("type")
        .ok_or_else(|| UrdfError::missing(line, "joint", "type"))?;
    let parent = child_attr(node, "parent", "link")
        .ok_or_else(|| UrdfError::missing(line, "joint", "parent"))?;
    let child = child_attr(node, "child", "link")
        .ok_or_else(|| UrdfError::missing(line, "joint", "child"))?;

    let (xyz, rpy) = parse_origin(named_child(node, "origin"), line)?;
    let origin = Pose::new(xyz, quat_from_rpy(rpy));

    let axis = named_child(node, "axis")
        .and_then(|n| n.attribute("xyz"))
        .map_or(Ok(Vec3::new(1.0, 0.0, 0.0)), parse_vec3)
        .map_err(|()| UrdfError::bad(line, "axis", "xyz", ""))?;
    let axis = if axis.norm() > 0.0 {
        axis.normalize()
    } else {
        return Err(UrdfError::bad(line, "axis", "xyz", "0 0 0"));
    };

    let limit_node = named_child(node, "limit");
    let (mut range, mut effort) = (None, None);
    if let Some(limit) = limit_node {
        let lower = num_attr(limit, "lower").unwrap_or(0.0);
        let upper = num_attr(limit, "upper").unwrap_or(0.0);
        range = Some((lower, upper));
        effort = num_attr(limit, "effort");
        if limit.attribute("velocity").is_some() {
            warnings.push(Warning {
                line,
                message: format!(
                    "joint `{name}`: <limit velocity> is not represented in SceneDesc"
                ),
            });
        }
    }

    let (mut damping, mut friction_loss) = (0.0, 0.0);
    if let Some(dyn_node) = named_child(node, "dynamics") {
        damping = num_attr(dyn_node, "damping").unwrap_or(0.0);
        friction_loss = num_attr(dyn_node, "friction").unwrap_or(0.0);
    }

    if named_child(node, "mimic").is_some() {
        warnings.push(Warning {
            line,
            message: format!("joint `{name}`: <mimic> is not supported, ignored"),
        });
    }
    for tag in ["calibration", "safety_controller"] {
        if named_child(node, tag).is_some() {
            warnings.push(Warning {
                line,
                message: format!("joint `{name}`: <{tag}> is not represented in SceneDesc"),
            });
        }
    }

    let kind = match kind_str {
        "fixed" => JointKind::Fixed,
        "floating" => JointKind::Free,
        "continuous" => {
            range = None;
            JointKind::Hinge
        }
        "revolute" => JointKind::Hinge,
        "prismatic" => JointKind::Slide,
        "planar" => {
            return Err(UrdfError::Unsupported {
                line,
                what: "joint type \"planar\"".to_owned(),
            })
        }
        other => return Err(UrdfError::bad(line, "joint", "type", other)),
    };

    Ok(JointDef {
        line,
        name,
        kind,
        parent,
        child,
        origin,
        axis,
        range,
        effort,
        damping,
        friction_loss,
    })
}

fn parse_inertial(link: Node, doc: &Document) -> Result<Option<BodyInertial>, UrdfError> {
    let Some(node) = named_child(link, "inertial") else {
        return Ok(None);
    };
    let line = doc.text_pos_at(node.range().start).row;
    let mass_node =
        named_child(node, "mass").ok_or_else(|| UrdfError::missing(line, "inertial", "mass"))?;
    let mass =
        num_attr(mass_node, "value").ok_or_else(|| UrdfError::missing(line, "mass", "value"))?;

    let (xyz, rpy) = parse_origin(named_child(node, "origin"), line)?;

    let inertia_node = named_child(node, "inertia")
        .ok_or_else(|| UrdfError::missing(line, "inertial", "inertia"))?;
    let comp = |name: &'static str| -> Result<f64, UrdfError> {
        num_attr(inertia_node, name).ok_or_else(|| UrdfError::missing(line, "inertia", name))
    };
    let inertia = Inertia::new(
        comp("ixx")?,
        comp("iyy")?,
        comp("izz")?,
        comp("ixy")?,
        comp("ixz")?,
        comp("iyz")?,
    );

    Ok(Some(BodyInertial {
        mass,
        com: xyz,
        inertia,
        frame: quat_from_rpy(rpy),
    }))
}

#[allow(clippy::too_many_arguments)]
fn parse_geoms(
    link: Node,
    link_name: &str,
    doc: &Document,
    resolver: &PackageResolver,
    assets_by_path: &mut BTreeMap<String, StableId>,
    scene: &mut SceneDesc,
    warnings: &mut Vec<Warning>,
) -> Result<Vec<Geom>, UrdfError> {
    let mut geoms = Vec::new();
    for (tag, visual) in [("visual", true), ("collision", false)] {
        for (i, node) in named_children(link, tag).enumerate() {
            let line = doc.text_pos_at(node.range().start).row;
            let name = node
                .attribute("name")
                .map_or_else(|| format!("{tag}{}", i + 1), str::to_owned);
            let (xyz, rpy) = parse_origin(named_child(node, "origin"), line)?;
            let pose = Pose::new(xyz, quat_from_rpy(rpy));

            let geometry = named_child(node, "geometry")
                .ok_or_else(|| UrdfError::missing(line, tag, "geometry"))?;
            let shape_node = geometry
                .children()
                .find(Node::is_element)
                .ok_or_else(|| UrdfError::missing(line, "geometry", "shape"))?;
            let shape = parse_shape(
                shape_node,
                line,
                &name,
                resolver,
                assets_by_path,
                scene,
                warnings,
            )?;

            let contype = u32::from(!visual);
            geoms.push(Geom {
                id: scene_id("geom", &format!("{link_name}/{name}")),
                name,
                shape,
                pose,
                friction: [1.0, 0.005, 0.0001],
                contype,
                conaffinity: contype,
                condim: 3,
                priority: 0,
                density: 1000.0,
                mass: None,
                margin: 0.0,
                gap: 0.0,
                solref: [0.02, 1.0],
                solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
                material: None,
                rgba: [0.5, 0.5, 0.5, 1.0],
                visual_only: visual,
            });
        }
    }
    Ok(geoms)
}

#[allow(clippy::too_many_arguments)]
fn parse_shape(
    node: Node,
    line: u32,
    name: &str,
    resolver: &PackageResolver,
    assets_by_path: &mut BTreeMap<String, StableId>,
    scene: &mut SceneDesc,
    warnings: &mut Vec<Warning>,
) -> Result<Shape, UrdfError> {
    match node.tag_name().name() {
        "box" => {
            let size = node
                .attribute("size")
                .map(parse_vec3)
                .transpose()
                .map_err(|()| UrdfError::bad(line, "box", "size", ""))?
                .ok_or_else(|| UrdfError::missing(line, "box", "size"))?;
            Ok(Shape::Box {
                half_extents: size.scale(0.5),
            })
        }
        "sphere" => Ok(Shape::Sphere {
            radius: num_attr(node, "radius")
                .ok_or_else(|| UrdfError::missing(line, "sphere", "radius"))?,
        }),
        "cylinder" => Ok(Shape::Cylinder {
            radius: num_attr(node, "radius")
                .ok_or_else(|| UrdfError::missing(line, "cylinder", "radius"))?,
            half_length: num_attr(node, "length")
                .ok_or_else(|| UrdfError::missing(line, "cylinder", "length"))?
                * 0.5,
        }),
        "mesh" => {
            let filename = node
                .attribute("filename")
                .ok_or_else(|| UrdfError::missing(line, "mesh", "filename"))?;
            let resolved = resolver.resolve(filename)?;
            if let Some(scale) = node.attribute("scale") {
                let scale =
                    parse_vec3(scale).map_err(|()| UrdfError::bad(line, "mesh", "scale", scale))?;
                if scale != Vec3::new(1.0, 1.0, 1.0) {
                    warnings.push(Warning {
                        line,
                        message: format!(
                            "mesh `{filename}`: non-unit scale is not represented in SceneDesc"
                        ),
                    });
                }
            }
            let path = resolved.to_string_lossy().into_owned();
            let id = *assets_by_path.entry(path.clone()).or_insert_with(|| {
                let asset = AssetRef::from_path(AssetKind::Mesh, name, &path);
                let id = asset.id;
                scene.assets.push(asset);
                id
            });
            Ok(Shape::Mesh { asset: id })
        }
        other => Err(UrdfError::Unsupported {
            line,
            what: format!("geometry <{other}>"),
        }),
    }
}

fn parse_transmission(
    node: Node,
    doc: &Document,
    joints: &[JointDef],
    effort_by_joint: &BTreeMap<&str, f64>,
    warnings: &mut Vec<Warning>,
) -> Result<Option<Actuator>, UrdfError> {
    let line = doc.text_pos_at(node.range().start).row;
    let trans_name = node.attribute("name").unwrap_or("transmission");
    let Some(joint_node) = named_child(node, "joint") else {
        warnings.push(Warning {
            line,
            message: format!("<transmission name=\"{trans_name}\"> has no <joint>, ignored"),
        });
        return Ok(None);
    };
    let joint_name = joint_node
        .attribute("name")
        .ok_or_else(|| UrdfError::missing(line, "joint", "name"))?;
    if !joints.iter().any(|j| j.name == joint_name) {
        return Err(UrdfError::UnknownJoint {
            line,
            name: trans_name.to_owned(),
            joint: joint_name.to_owned(),
        });
    }

    let actuator_node = named_child(node, "actuator");
    let actuator_name = actuator_node
        .and_then(|n| n.attribute("name"))
        .unwrap_or(trans_name)
        .to_owned();
    let reduction = actuator_node
        .and_then(|n| named_child(n, "mechanicalReduction"))
        .and_then(|n| n.text())
        .and_then(|t| t.trim().parse::<f64>().ok())
        .unwrap_or(1.0);
    let force_range = effort_by_joint.get(joint_name).map(|&e| (-e, e));

    Ok(Some(Actuator {
        id: scene_id("actuator", &actuator_name),
        name: actuator_name,
        kind: ActuatorKind::Motor,
        target: ActuatorTarget::Joint(scene_id("joint", joint_name)),
        gear: [reduction, 0.0, 0.0, 0.0, 0.0, 0.0],
        ctrl_range: None,
        force_range,
    }))
}

// ---- small XML helpers ------------------------------------------------------------------

fn named_child<'a>(node: Node<'a, 'a>, tag: &str) -> Option<Node<'a, 'a>> {
    node.children()
        .find(|n| n.is_element() && n.has_tag_name(tag))
}

fn named_children<'a>(node: Node<'a, 'a>, tag: &'a str) -> impl Iterator<Item = Node<'a, 'a>> {
    node.children()
        .filter(move |n| n.is_element() && n.has_tag_name(tag))
}

/// `<joint><parent link="x"/></joint>`-shaped reference: the attribute lives on a child element.
fn child_attr(node: Node, child_tag: &str, attr: &str) -> Option<String> {
    named_child(node, child_tag)?
        .attribute(attr)
        .map(str::to_owned)
}

fn num_attr(node: Node, name: &str) -> Option<f64> {
    node.attribute(name)?.trim().parse::<f64>().ok()
}

/// Reads `<origin xyz="" rpy=""/>` (each defaulting to zero) — the one pattern joints,
/// inertials and visual/collision geoms all share.
fn parse_origin(node: Option<Node>, line: u32) -> Result<(Vec3, Vec3), UrdfError> {
    let xyz = node
        .and_then(|n| n.attribute("xyz"))
        .map_or(Ok(Vec3::ZERO), parse_vec3)
        .map_err(|()| UrdfError::bad(line, "origin", "xyz", ""))?;
    let rpy = node
        .and_then(|n| n.attribute("rpy"))
        .map_or(Ok(Vec3::ZERO), parse_vec3)
        .map_err(|()| UrdfError::bad(line, "origin", "rpy", ""))?;
    Ok((xyz, rpy))
}

fn parse_vec3(text: &str) -> Result<Vec3, ()> {
    let mut it = text.split_whitespace();
    let mut next = || it.next().and_then(|t| t.parse::<f64>().ok()).ok_or(());
    let v = Vec3::new(next()?, next()?, next()?);
    if it.next().is_some() {
        return Err(());
    }
    Ok(v)
}

/// URDF roll-pitch-yaw: a fixed-axis (extrinsic) rotation about X, then Y, then Z, i.e.
/// `q = qz * qy * qx`. URDF is already Z-up SI (spec 3.1), so this is the only conversion
/// needed — no axis remapping.
fn quat_from_rpy(rpy: Vec3) -> Quat {
    let (sr, cr) = (rpy.x * 0.5).sin_cos();
    let (sp, cp) = (rpy.y * 0.5).sin_cos();
    let (sy, cy) = (rpy.z * 0.5).sin_cos();
    Quat::from_xyzw(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
    .normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolver() -> PackageResolver {
        PackageResolver::new(BTreeMap::from([(
            "robo_pkg".to_owned(),
            PathBuf::from("/pkgs/robo_pkg"),
        )]))
    }

    #[test]
    fn resolves_package_uri() {
        let p = resolver()
            .resolve("package://robo_pkg/meshes/a.stl")
            .unwrap();
        assert_eq!(p, PathBuf::from("/pkgs/robo_pkg").join("meshes/a.stl"));
    }

    #[test]
    fn unknown_package_lists_known_roots() {
        let err = resolver().resolve("package://ghost/a.stl").unwrap_err();
        assert!(matches!(err, UrdfError::UnknownPackage { .. }), "{err}");
    }

    #[test]
    fn file_and_relative_uris_pass_through() {
        assert_eq!(
            resolver().resolve("file:///abs/mesh.stl").unwrap(),
            PathBuf::from("/abs/mesh.stl")
        );
        assert_eq!(
            resolver().resolve("meshes/a.stl").unwrap(),
            PathBuf::from("meshes/a.stl")
        );
    }

    #[test]
    fn scan_dir_finds_a_package_by_its_manifest_name() {
        // `from_env` is a one-line wrapper around `scan_dir` over `ROS_PACKAGE_PATH`; testing
        // the scan directly keeps this test from mutating process-global environment state.
        let dir = std::env::temp_dir().join(format!(
            "es-assets-urdf-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let pkg = dir.join("robo_pkg");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::write(
            pkg.join("package.xml"),
            "<package><name>robo_pkg</name></package>",
        )
        .unwrap();

        let mut roots = BTreeMap::new();
        scan_dir(&dir, 0, &mut roots);
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(roots.get("robo_pkg"), Some(&pkg));
    }

    #[test]
    fn rpy_identity_is_identity_quat() {
        assert_eq!(quat_from_rpy(Vec3::ZERO), Quat::IDENTITY);
    }
}
