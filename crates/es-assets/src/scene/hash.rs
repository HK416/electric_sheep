//! `SceneDesc::scene_hash` (spec 5.3): blake3 over the canonical encoding the parent module
//! describes, and the encoder that writes it.

use es_core::StableId;
use es_math::{Pose, Quat, Vec3};

use super::{
    Actuator, ActuatorKind, ActuatorTarget, Body, ContactPair, Geom, Joint, SceneDesc, Sensor,
    SensorTarget, Shape, Tendon, TendonKind,
};

/// Domain separator: changing the canonical encoding must change every stored hash.
const SCENE_TAG: &str = "es.scene.v1";

impl SceneDesc {
    /// Content identity of the scene (spec 5.3): blake3 of the canonical encoding.
    ///
    /// Independent of element order in the source file, sensitive to every field that can
    /// change simulation behaviour.
    pub fn scene_hash(&self) -> [u8; 32] {
        let mut c = Canon::default();
        c.str(SCENE_TAG);
        c.str(&self.name);
        self.encode_options(&mut c);

        let mut bodies: Vec<&Body> = self.bodies.iter().collect();
        bodies.sort_by_key(|b| b.id);
        c.seq(bodies.len());
        for body in bodies {
            encode_body(&mut c, body);
        }
        encode_sorted(&mut c, &self.joints, |j| j.id, encode_joint);
        encode_sorted(
            &mut c,
            &self.cameras,
            |x| x.id,
            |c, cam| {
                c.id(cam.id);
                c.str(&cam.name);
                c.opt_id(cam.body);
                c.pose(cam.pose);
                c.f64(cam.fovy);
            },
        );
        encode_sorted(&mut c, &self.tendons, |t| t.id, encode_tendon);
        encode_sorted(&mut c, &self.actuators, |a| a.id, encode_actuator);
        encode_sorted(&mut c, &self.sensors, |s| s.id, encode_sensor);
        encode_sorted(
            &mut c,
            &self.assets,
            |a| a.id,
            |c, asset| {
                c.id(asset.id);
                c.str(&asset.name);
                c.str(asset.kind.tag());
                c.str(&asset.path);
                c.raw(&asset.hash);
            },
        );
        self.encode_contact(&mut c);
        self.encode_appearance(&mut c);
        // Packet M17/R3: mesh scales, appended only when a mesh has one, so every scene
        // without one keeps the digest it had (spec 28.13 rule 2).
        if !self.mesh_scales.is_empty() {
            c.str("es.scene.mesh_scale.v1");
            c.seq(self.mesh_scales.len());
            for (id, s) in &self.mesh_scales {
                c.id(*id);
                for v in s {
                    c.f64(*v);
                }
            }
        }
        c.finish()
    }

    /// The drawn materials (plan H, HT1), appended only when the scene has one, so a scene
    /// without textures or PBR attributes keeps the digest it had (spec 28.13 rule 2). The
    /// texels themselves are in the texture assets' content digests.
    fn encode_appearance(&self, c: &mut Canon) {
        if self.materials.is_empty() {
            return;
        }
        c.str("es.scene.appearance.v1");
        c.seq(self.materials.len());
        for (id, m) in &self.materials {
            c.id(*id);
            for v in m.rgba {
                c.f64(v);
            }
            c.f64(m.emission);
            for v in [m.specular, m.shininess, m.metallic, m.roughness] {
                match v {
                    None => c.u8(0),
                    Some(v) => {
                        c.u8(1);
                        c.f64(v);
                    }
                }
            }
            c.f64(m.texrepeat[0]);
            c.f64(m.texrepeat[1]);
            c.u8(u8::from(m.texuniform));
            for t in [m.rgb, m.orm, m.metallic_map, m.roughness_map] {
                c.opt_id(t);
            }
            // Plan H, HT2's maps, in a tagged block written only when the material has one,
            // so an HT1 material keeps the digest it had.
            if m.normal_map.is_some()
                || m.normal_scale.is_some()
                || m.emissive_map.is_some()
                || m.emissive.is_some()
            {
                c.str("es.material.maps.v1");
                c.opt_id(m.normal_map);
                c.f64(m.normal_scale.unwrap_or(1.0));
                c.opt_id(m.emissive_map);
                match m.emissive {
                    None => c.u8(0),
                    Some(e) => {
                        c.u8(1);
                        for v in e {
                            c.f64(v);
                        }
                    }
                }
            }
        }
    }

    /// Pairs, excludes and gravity compensation, appended only when the scene has any, so a
    /// scene without them keeps the digest it had before they were represented (spec 28.13
    /// rule 2). Each list is sorted, so source order cannot move the hash.
    fn encode_contact(&self, c: &mut Canon) {
        if self.contact_pairs.is_empty()
            && self.contact_excludes.is_empty()
            && self.gravcomp.is_empty()
        {
            return;
        }
        c.str("es.scene.contact.v1");
        let mut pairs: Vec<&ContactPair> = self.contact_pairs.iter().collect();
        pairs.sort_by_key(|p| (p.geom1, p.geom2));
        c.seq(pairs.len());
        for p in pairs {
            c.id(p.geom1);
            c.id(p.geom2);
            c.u32(p.condim.map_or(0, |d| d + 1));
            for values in [
                p.friction.as_ref().map(|v| &v[..]),
                p.solref.as_ref().map(|v| &v[..]),
                p.solimp.as_ref().map(|v| &v[..]),
                p.margin.as_ref().map(std::slice::from_ref),
                p.gap.as_ref().map(std::slice::from_ref),
            ] {
                match values {
                    None => c.u8(0),
                    Some(values) => {
                        c.u8(1);
                        for v in values {
                            c.f64(*v);
                        }
                    }
                }
            }
        }
        let mut excludes = self.contact_excludes.clone();
        excludes.sort();
        c.seq(excludes.len());
        for (a, b) in excludes {
            c.id(a);
            c.id(b);
        }
        c.seq(self.gravcomp.len());
        for (body, value) in &self.gravcomp {
            c.id(*body);
            c.f64(*value);
        }
    }

    fn encode_options(&self, c: &mut Canon) {
        let o = &self.options;
        c.f64(o.timestep);
        c.vec3(o.gravity);
        c.u32(o.integrator as u32);
        c.u32(o.cone as u32);
        c.u32(o.jacobian as u32);
        c.u32(o.solver as u32);
        c.u32(o.iterations);
        c.f64(o.impratio);
    }
}

fn encode_sorted<T>(
    c: &mut Canon,
    items: &[T],
    key: impl Fn(&T) -> StableId,
    encode: impl Fn(&mut Canon, &T),
) {
    let mut sorted: Vec<&T> = items.iter().collect();
    sorted.sort_by_key(|item| key(item));
    c.seq(sorted.len());
    for item in sorted {
        encode(c, item);
    }
}

fn encode_body(c: &mut Canon, body: &Body) {
    c.id(body.id);
    c.str(&body.name);
    c.opt_id(body.parent);
    c.pose(body.pose);
    match &body.inertial {
        None => c.u8(0),
        Some(i) => {
            c.u8(1);
            c.f64(i.mass);
            c.vec3(i.com);
            for row in i.inertia.matrix() {
                for v in row {
                    c.f64(v);
                }
            }
            c.quat(i.frame);
        }
    }
    encode_sorted(c, &body.geoms, |g| g.id, encode_geom);
    encode_sorted(
        c,
        &body.sites,
        |s| s.id,
        |c, site| {
            c.id(site.id);
            c.str(&site.name);
            c.pose(site.pose);
            c.vec3(site.size);
        },
    );
}

fn encode_geom(c: &mut Canon, g: &Geom) {
    c.id(g.id);
    c.str(&g.name);
    match g.shape {
        Shape::Plane {
            half_x,
            half_y,
            grid,
        } => {
            c.u8(0);
            c.f64(half_x);
            c.f64(half_y);
            c.f64(grid);
        }
        Shape::Sphere { radius } => {
            c.u8(1);
            c.f64(radius);
        }
        Shape::Capsule {
            radius,
            half_length,
        } => {
            c.u8(2);
            c.f64(radius);
            c.f64(half_length);
        }
        Shape::Cylinder {
            radius,
            half_length,
        } => {
            c.u8(3);
            c.f64(radius);
            c.f64(half_length);
        }
        Shape::Box { half_extents } => {
            c.u8(4);
            c.vec3(half_extents);
        }
        Shape::Ellipsoid { radii } => {
            c.u8(5);
            c.vec3(radii);
        }
        Shape::Mesh { asset } => {
            c.u8(6);
            c.id(asset);
        }
        Shape::HeightField { asset } => {
            c.u8(7);
            c.id(asset);
        }
    }
    c.pose(g.pose);
    for v in g.friction {
        c.f64(v);
    }
    c.u32(g.contype);
    c.u32(g.conaffinity);
    c.u32(g.condim);
    c.i32(g.priority);
    c.f64(g.density);
    match g.mass {
        None => c.u8(0),
        Some(m) => {
            c.u8(1);
            c.f64(m);
        }
    }
    c.f64(g.margin);
    c.f64(g.gap);
    for v in g.solref {
        c.f64(v);
    }
    for v in g.solimp {
        c.f64(v);
    }
    c.opt_id(g.material);
    for v in g.rgba {
        c.f64(v);
    }
    c.u8(u8::from(g.visual_only));
}

fn encode_joint(c: &mut Canon, j: &Joint) {
    c.id(j.id);
    c.str(&j.name);
    c.id(j.body);
    c.u8(j.kind as u8);
    c.vec3(j.axis);
    c.vec3(j.anchor);
    c.opt_range(j.range);
    c.f64(j.damping);
    c.f64(j.armature);
    c.f64(j.stiffness);
    c.f64(j.friction_loss);
    c.f64(j.spring_ref);
}

fn encode_tendon(c: &mut Canon, t: &Tendon) {
    c.id(t.id);
    c.str(&t.name);
    match &t.kind {
        TendonKind::Fixed { joints } => {
            c.u8(0);
            c.seq(joints.len());
            for (id, coef) in joints {
                c.id(*id);
                c.f64(*coef);
            }
        }
        TendonKind::Spatial { sites } => {
            c.u8(1);
            c.seq(sites.len());
            for id in sites {
                c.id(*id);
            }
        }
    }
    c.opt_range(t.range);
    c.f64(t.stiffness);
    c.f64(t.damping);
}

fn encode_actuator(c: &mut Canon, a: &Actuator) {
    c.id(a.id);
    c.str(&a.name);
    match a.kind {
        ActuatorKind::Motor => c.u8(0),
        ActuatorKind::Position { kp, kv } => {
            c.u8(1);
            c.f64(kp);
            c.f64(kv);
        }
        ActuatorKind::Velocity { kv } => {
            c.u8(2);
            c.f64(kv);
        }
        ActuatorKind::General { gain, bias } => {
            c.u8(3);
            for v in gain.into_iter().chain(bias) {
                c.f64(v);
            }
        }
    }
    match a.target {
        ActuatorTarget::Joint(id) => {
            c.u8(0);
            c.id(id);
        }
        ActuatorTarget::Tendon(id) => {
            c.u8(1);
            c.id(id);
        }
        ActuatorTarget::Site(id) => {
            c.u8(2);
            c.id(id);
        }
    }
    for v in a.gear {
        c.f64(v);
    }
    c.opt_range(a.ctrl_range);
    c.opt_range(a.force_range);
}

fn encode_sensor(c: &mut Canon, s: &Sensor) {
    c.id(s.id);
    c.str(&s.name);
    c.u8(s.kind as u8);
    match s.target {
        SensorTarget::Body(id) => {
            c.u8(0);
            c.id(id);
        }
        SensorTarget::Joint(id) => {
            c.u8(1);
            c.id(id);
        }
        SensorTarget::Geom(id) => {
            c.u8(2);
            c.id(id);
        }
        SensorTarget::Site(id) => {
            c.u8(3);
            c.id(id);
        }
        SensorTarget::Camera(id) => {
            c.u8(4);
            c.id(id);
        }
        SensorTarget::Actuator(id) => {
            c.u8(5);
            c.id(id);
        }
    }
    c.f64(s.noise);
    c.f64(s.cutoff);
}

/// Canonical byte encoder: little-endian, length-prefixed, `f64` as normalised IEEE bits.
///
/// Deliberately tiny and local — `es-ir` has its own (spec 5.3 hashes a different layer), and
/// `es-assets` (layer 2) cannot depend on `es-ir` (layer 6).
#[derive(Debug, Default)]
struct Canon {
    buf: Vec<u8>,
}

impl Canon {
    fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn seq(&mut self, len: usize) {
        self.u32(u32::try_from(len).unwrap_or(u32::MAX));
    }

    fn str(&mut self, s: &str) {
        self.seq(s.len());
        self.raw(s.as_bytes());
    }

    /// `-0.0` normalises to `0.0` (they compare equal, so they must hash equal); every `NaN`
    /// normalises to one pattern (no `NaN` is distinguishable from another by comparison).
    fn f64(&mut self, v: f64) {
        let bits = if v == 0.0 {
            0f64.to_bits()
        } else if v.is_nan() {
            f64::NAN.to_bits()
        } else {
            v.to_bits()
        };
        self.buf.extend_from_slice(&bits.to_le_bytes());
    }

    fn id(&mut self, id: StableId) {
        self.raw(id.as_bytes());
    }

    fn opt_id(&mut self, id: Option<StableId>) {
        match id {
            None => self.u8(0),
            Some(id) => {
                self.u8(1);
                self.id(id);
            }
        }
    }

    fn opt_range(&mut self, range: Option<(f64, f64)>) {
        match range {
            None => self.u8(0),
            Some((lo, hi)) => {
                self.u8(1);
                self.f64(lo);
                self.f64(hi);
            }
        }
    }

    fn vec3(&mut self, v: Vec3) {
        self.f64(v.x);
        self.f64(v.y);
        self.f64(v.z);
    }

    fn quat(&mut self, q: Quat) {
        self.f64(q.x);
        self.f64(q.y);
        self.f64(q.z);
        self.f64(q.w);
    }

    fn pose(&mut self, p: Pose) {
        self.vec3(p.position);
        self.quat(p.orientation);
    }

    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.buf).as_bytes()
    }
}
