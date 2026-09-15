//! `<asset>`, `<tendon>`, `<actuator>` and `<sensor>` (P33).
//!
//! These are the sections that reference elements by name, so they run after `<worldbody>`
//! and resolve every name through [`lookup`] — an unresolved name is an error, never a
//! silently dropped actuator.

use es_core::StableId;
use roxmltree::Node;

use super::attrs::Attrs;
use super::{lookup, MjcfError, Parser};
use crate::scene::{
    scene_id, Actuator, ActuatorKind, ActuatorTarget, AssetKind, AssetRef, Sensor, SensorKind,
    SensorTarget, Tendon, TendonKind,
};

/// Actuator element names this packet maps; anything else is reported, not guessed at.
const ACTUATORS: [&str; 4] = ["motor", "position", "velocity", "general"];

impl<'a> Parser<'a> {
    // ---- <asset> ----------------------------------------------------------------------

    pub(super) fn parse_asset(&mut self, node: Node<'a, 'a>) -> Result<(), MjcfError> {
        for child in node.children().filter(Node::is_element) {
            let tag = child.tag_name().name();
            let kind = match tag {
                "mesh" => AssetKind::Mesh,
                "texture" => AssetKind::Texture,
                "material" => AssetKind::Material,
                "hfield" => AssetKind::HeightField,
                other => {
                    let line = self.line(child);
                    self.warn(
                        line,
                        format!("<asset><{other}> is not represented, ignored"),
                    );
                    continue;
                }
            };
            let attrs = self.element(child, "main")?;
            let file = attrs.get("file");
            let name = match attrs.get("name").or_else(|| file.map(stem)) {
                Some(name) => name.to_owned(),
                None => return Err(attrs.missing("name")),
            };
            let dir = match kind {
                AssetKind::Texture => &self.compiler.texturedir,
                _ => &self.compiler.meshdir,
            };
            // A material has no file of its own: it points at a texture by name.
            let path = match kind {
                AssetKind::Material => attrs.get("texture").unwrap_or_default().to_owned(),
                _ => join(dir, file.unwrap_or_default()),
            };
            let asset = AssetRef::from_path(kind, &name, &path);
            attrs.report_unknown(&mut self.warnings);
            match kind {
                AssetKind::Mesh => self.names.meshes.insert(name, asset.id),
                AssetKind::Material => self.names.materials.insert(name, asset.id),
                AssetKind::HeightField => self.names.hfields.insert(name, asset.id),
                AssetKind::Texture => None,
            };
            self.scene.assets.push(asset);
        }
        Ok(())
    }

    // ---- <tendon> ---------------------------------------------------------------------

    pub(super) fn parse_tendon(&mut self, node: Node<'a, 'a>) -> Result<(), MjcfError> {
        let childclass = node.attribute("childclass").unwrap_or("main");
        for child in node.children().filter(Node::is_element) {
            let tag = child.tag_name().name();
            if tag != "fixed" && tag != "spatial" {
                let line = self.line(child);
                self.warn(line, format!("<tendon><{tag}> is not represented, ignored"));
                continue;
            }
            let attrs = self.element(child, childclass)?;
            let name = self.local_name(&attrs, "tendon");
            let id = scene_id("tendon", &name);
            let kind = if tag == "fixed" {
                TendonKind::Fixed {
                    joints: self.tendon_joints(child, childclass)?,
                }
            } else {
                TendonKind::Spatial {
                    sites: self.tendon_sites(child, childclass)?,
                }
            };
            let tendon = Tendon {
                id,
                name: name.clone(),
                kind,
                range: self.limit(&attrs, "limited", "range", 1.0)?,
                stiffness: attrs.num_or("stiffness", 0.0)?,
                damping: attrs.num_or("damping", 0.0)?,
            };
            attrs.report_unknown(&mut self.warnings);
            self.names.tendons.insert(name, id);
            self.scene.tendons.push(tendon);
        }
        Ok(())
    }

    fn tendon_joints(
        &mut self,
        node: Node<'a, 'a>,
        childclass: &'a str,
    ) -> Result<Vec<(StableId, f64)>, MjcfError> {
        let mut out = Vec::new();
        for child in node.children().filter(Node::is_element) {
            let attrs = self.element(child, childclass)?;
            if attrs.tag() != "joint" {
                let message = format!("<fixed><{}> is not represented, ignored", attrs.tag());
                self.warn(attrs.line(), message);
                continue;
            }
            let joint = attrs.get("joint").ok_or_else(|| attrs.missing("joint"))?;
            let id = lookup(&self.names.joints, &attrs, "joint", "joint", joint)?;
            let coef = attrs.num_or("coef", 1.0)?;
            attrs.report_unknown(&mut self.warnings);
            out.push((id, coef));
        }
        Ok(out)
    }

    fn tendon_sites(
        &mut self,
        node: Node<'a, 'a>,
        childclass: &'a str,
    ) -> Result<Vec<StableId>, MjcfError> {
        let mut out = Vec::new();
        for child in node.children().filter(Node::is_element) {
            let attrs = self.element(child, childclass)?;
            if attrs.tag() != "site" {
                let message = format!("<spatial><{}> is not represented, ignored", attrs.tag());
                self.warn(attrs.line(), message);
                continue;
            }
            let site = attrs.get("site").ok_or_else(|| attrs.missing("site"))?;
            out.push(lookup(&self.names.sites, &attrs, "site", "site", site)?);
            attrs.report_unknown(&mut self.warnings);
        }
        Ok(out)
    }

    // ---- <actuator> -------------------------------------------------------------------

    pub(super) fn parse_actuator(&mut self, node: Node<'a, 'a>) -> Result<(), MjcfError> {
        let childclass = node.attribute("childclass").unwrap_or("main");
        for child in node.children().filter(Node::is_element) {
            let tag = child.tag_name().name();
            if !ACTUATORS.contains(&tag) {
                let line = self.line(child);
                self.warn(
                    line,
                    format!("<actuator><{tag}> is not represented, ignored"),
                );
                continue;
            }
            let attrs = self.element(child, childclass)?;
            let (target, target_name) = self.actuator_target(&attrs)?;
            let name = match attrs.get("name") {
                Some(name) => name.to_owned(),
                None => format!("{tag}_{target_name}"),
            };
            let id = scene_id("actuator", &name);
            let kind = match tag {
                "position" => ActuatorKind::Position {
                    kp: attrs.num_or("kp", 1.0)?,
                    kv: attrs.num_or("kv", 0.0)?,
                },
                "velocity" => ActuatorKind::Velocity {
                    kv: attrs.num_or("kv", 1.0)?,
                },
                "general" => ActuatorKind::General {
                    gain: head(attrs.nums("gainprm")?, [1.0, 0.0, 0.0]),
                    bias: head(attrs.nums("biasprm")?, [0.0, 0.0, 0.0]),
                },
                _ => ActuatorKind::Motor,
            };
            if tag == "general" {
                for (attr, value, default) in [
                    ("dyntype", attrs.get_or("dyntype", "none"), "none"),
                    ("gaintype", attrs.get_or("gaintype", "fixed"), "fixed"),
                    ("biastype", attrs.get_or("biastype", "none"), "none"),
                ] {
                    if value != default {
                        let message = format!(
                            "<general {attr}=\"{value}\">: transduction beyond gain and bias is \
                             the backend's mapping (spec 17.2)"
                        );
                        self.warn(attrs.line(), message);
                    }
                }
            }
            let actuator = Actuator {
                id,
                name: name.clone(),
                kind,
                target,
                gear: head(attrs.nums("gear")?, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ctrl_range: self.actuator_ctrl_range(tag, &attrs, target)?,
                force_range: self.limit(&attrs, "forcelimited", "forcerange", 1.0)?,
            };
            attrs.report_unknown(&mut self.warnings);
            self.names.actuators.insert(name, id);
            self.scene.actuators.push(actuator);
        }
        Ok(())
    }

    /// `ctrlrange`, with `<position inheritrange>` resolved the way `MuJoCo`'s compiler
    /// resolves it: the transmission target's own range, scaled by `inheritrange` about its
    /// midpoint. It is a compile-time rewrite upstream, so resolving it here loses nothing --
    /// and dropping it would leave the actuator unclamped, which silently widens the range
    /// the policy's position targets are held to.
    fn actuator_ctrl_range(
        &self,
        tag: &str,
        attrs: &Attrs<'a>,
        target: ActuatorTarget,
    ) -> Result<Option<(f64, f64)>, MjcfError> {
        let explicit = self.limit(attrs, "ctrllimited", "ctrlrange", 1.0)?;
        // Upstream allows it on `position` and `intvelocity`; we only model the former.
        if tag != "position" {
            return Ok(explicit);
        }
        let Some(factor) = attrs.num("inheritrange")? else {
            return Ok(explicit);
        };
        let ActuatorTarget::Joint(joint) = target else {
            return Ok(explicit);
        };
        let range = self
            .scene
            .joints
            .iter()
            .find(|j| j.id == joint)
            .and_then(|j| j.range);
        let (Some((lo, hi)), true) = (range, factor > 0.0) else {
            return Ok(explicit);
        };
        let (mid, half) = (f64::midpoint(lo, hi), (hi - lo) / 2.0);
        Ok(Some((mid - factor * half, mid + factor * half)))
    }

    fn actuator_target(&self, attrs: &Attrs<'a>) -> Result<(ActuatorTarget, &'a str), MjcfError> {
        if let Some(name) = attrs.get("joint") {
            let id = lookup(&self.names.joints, attrs, "joint", "joint", name)?;
            return Ok((ActuatorTarget::Joint(id), name));
        }
        if let Some(name) = attrs.get("tendon") {
            let id = lookup(&self.names.tendons, attrs, "tendon", "tendon", name)?;
            return Ok((ActuatorTarget::Tendon(id), name));
        }
        if let Some(name) = attrs.get("site") {
            let id = lookup(&self.names.sites, attrs, "site", "site", name)?;
            return Ok((ActuatorTarget::Site(id), name));
        }
        Err(attrs.missing("joint"))
    }

    // ---- <sensor> ---------------------------------------------------------------------

    pub(super) fn parse_sensor(&mut self, node: Node<'a, 'a>) -> Result<(), MjcfError> {
        for child in node.children().filter(Node::is_element) {
            let tag = child.tag_name().name();
            let Some(kind) = sensor_kind(tag) else {
                let line = self.line(child);
                self.warn(line, format!("<sensor><{tag}> is not represented, ignored"));
                continue;
            };
            let attrs = self.element(child, "main")?;
            let (target, target_name) = self.sensor_target(&attrs, kind)?;
            let name = match attrs.get("name") {
                Some(name) => name.to_owned(),
                None => format!("{tag}_{target_name}"),
            };
            let sensor = Sensor {
                id: scene_id("sensor", &name),
                name,
                kind,
                target,
                noise: attrs.num_or("noise", 0.0)?,
                cutoff: attrs.num_or("cutoff", 0.0)?,
            };
            attrs.report_unknown(&mut self.warnings);
            self.scene.sensors.push(sensor);
        }
        Ok(())
    }

    fn sensor_target(
        &self,
        attrs: &Attrs<'a>,
        kind: SensorKind,
    ) -> Result<(SensorTarget, &'a str), MjcfError> {
        match kind {
            SensorKind::JointPos | SensorKind::JointVel => {
                let name = attrs.get("joint").ok_or_else(|| attrs.missing("joint"))?;
                let id = lookup(&self.names.joints, attrs, "joint", "joint", name)?;
                Ok((SensorTarget::Joint(id), name))
            }
            SensorKind::ActuatorFrc => {
                let name = attrs
                    .get("actuator")
                    .ok_or_else(|| attrs.missing("actuator"))?;
                let id = lookup(&self.names.actuators, attrs, "actuator", "actuator", name)?;
                Ok((SensorTarget::Actuator(id), name))
            }
            SensorKind::FramePos | SensorKind::FrameQuat => {
                let name = attrs
                    .get("objname")
                    .ok_or_else(|| attrs.missing("objname"))?;
                let objtype = attrs
                    .get("objtype")
                    .ok_or_else(|| attrs.missing("objtype"))?;
                let target = match objtype {
                    "body" | "xbody" => SensorTarget::Body(lookup(
                        &self.names.bodies,
                        attrs,
                        "objname",
                        "body",
                        name,
                    )?),
                    "site" => SensorTarget::Site(lookup(
                        &self.names.sites,
                        attrs,
                        "objname",
                        "site",
                        name,
                    )?),
                    "geom" => SensorTarget::Geom(lookup(
                        &self.names.geoms,
                        attrs,
                        "objname",
                        "geom",
                        name,
                    )?),
                    "camera" => SensorTarget::Camera(lookup(
                        &self.names.cameras,
                        attrs,
                        "objname",
                        "camera",
                        name,
                    )?),
                    other => return Err(attrs.bad("objtype", other)),
                };
                Ok((target, name))
            }
            SensorKind::Camera => {
                let name = attrs.get("camera").ok_or_else(|| attrs.missing("camera"))?;
                let id = lookup(&self.names.cameras, attrs, "camera", "camera", name)?;
                Ok((SensorTarget::Camera(id), name))
            }
            _ => {
                let name = attrs.get("site").ok_or_else(|| attrs.missing("site"))?;
                let id = lookup(&self.names.sites, attrs, "site", "site", name)?;
                Ok((SensorTarget::Site(id), name))
            }
        }
    }
}

fn sensor_kind(tag: &str) -> Option<SensorKind> {
    Some(match tag {
        "jointpos" => SensorKind::JointPos,
        "jointvel" => SensorKind::JointVel,
        "actuatorfrc" => SensorKind::ActuatorFrc,
        "framepos" => SensorKind::FramePos,
        "framequat" => SensorKind::FrameQuat,
        "accelerometer" => SensorKind::Accelerometer,
        "gyro" => SensorKind::Gyro,
        "force" => SensorKind::Force,
        "torque" => SensorKind::Torque,
        "touch" => SensorKind::Touch,
        "rangefinder" => SensorKind::RangeFinder,
        "camprojection" => SensorKind::Camera,
        _ => return None,
    })
}

/// First `N` values of an MJCF list, keeping `default` for the tail. MJCF pads `gainprm` and
/// friends out to ten entries; only the affine head is represented here.
fn head<const N: usize>(values: Option<Vec<f64>>, default: [f64; N]) -> [f64; N] {
    let mut out = default;
    if let Some(values) = values {
        let n = values.len().min(N);
        out[..n].copy_from_slice(&values[..n]);
    }
    out
}

/// `<compiler meshdir>` + the file attribute, without pretending to be a path library.
fn join(dir: &str, file: &str) -> String {
    if dir.is_empty() || file.is_empty() {
        return file.to_owned();
    }
    format!("{}/{}", dir.trim_end_matches(['/', '\\']), file)
}

/// File name without directories or extension — `MuJoCo`'s rule for an unnamed asset.
fn stem(file: &str) -> &str {
    let name = file.rsplit(['/', '\\']).next().unwrap_or(file);
    name.rsplit_once('.').map_or(name, |(base, _)| base)
}
