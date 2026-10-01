//! `[[include]]`: a robot (or any asset file) read by its own reader and merged into the scene.
//!
//! Steps, in order: read the file (MJCF, URDF, glTF; USD has no `SceneDesc` reader in this
//! layer); rebase the files it names onto the document's directory; apply `[include.set]` by the
//! file's own names; with a `prefix`, rename every element and re-derive its id by the MJCF name
//! path scheme (`crate::scene`'s table); place the root bodies and world-level elements at the
//! include's pose; merge. The file's `<option>` and model name are not read: the document's
//! `[physics]` and `name` are the scene's.
//!
//! Without a prefix the file's own names *and ids* are kept, which is what makes a document that
//! includes a robot read to the same `SceneDesc` as the one file that wrote both.

use std::collections::BTreeMap;
use std::path::Path;

use es_core::StableId;

use super::expand::{asset, body_path, pose, world};
use super::{refuse, EsScene, EsSceneError, Include};
use crate::scene::{
    scene_id, ActuatorKind, ActuatorTarget, AssetKind, AssetRef, SceneDesc, SensorTarget, Shape,
    TendonKind,
};

pub(super) fn expand_include(
    s: &mut SceneDesc,
    inc: &Include,
    doc: &EsScene,
    base_dir: &Path,
) -> Result<(), EsSceneError> {
    let field = format!("include[{}]", inc.name);
    let mut sub = read(inc, base_dir, &field)?;
    let dir = inc.source.rsplit_once(['/', '\\']).map_or("", |(d, _)| d);
    rebase(&mut sub, dir);
    if let Some(set) = &inc.set {
        apply_set(&mut sub, set, doc, &format!("{field}.set"))?;
    }
    if let Some(prefix) = &inc.prefix {
        rename(&mut sub, prefix);
    }
    let place = pose(inc.pos, inc.quat, &field)?;
    let w = world();
    // An identity placement leaves every bit as the file wrote it.
    let moved = place != es_math::Pose::IDENTITY;
    for mut b in std::mem::take(&mut sub.bodies) {
        let root = b.id == w || b.parent.is_none() || b.parent == Some(w);
        if moved && root && b.id != w {
            b.pose = place.compose(b.pose);
        }
        if b.id == w {
            if moved {
                for g in &mut b.geoms {
                    g.pose = place.compose(g.pose);
                }
                for site in &mut b.sites {
                    site.pose = place.compose(site.pose);
                }
            }
            s.bodies[0].geoms.append(&mut b.geoms);
            s.bodies[0].sites.append(&mut b.sites);
            continue;
        }
        b.parent = b.parent.or(Some(w));
        s.bodies.push(b);
    }
    for mut c in sub.cameras {
        if moved && c.body.is_none() {
            c.pose = place.compose(c.pose);
        }
        s.cameras.push(c);
    }
    s.joints.append(&mut sub.joints);
    s.actuators.append(&mut sub.actuators);
    s.sensors.append(&mut sub.sensors);
    s.tendons.append(&mut sub.tendons);
    s.assets.append(&mut sub.assets);
    s.contact_pairs.append(&mut sub.contact_pairs);
    s.contact_excludes.append(&mut sub.contact_excludes);
    s.meshes.append(&mut sub.meshes);
    s.gravcomp.append(&mut sub.gravcomp);
    s.materials.append(&mut sub.materials);
    s.textures.append(&mut sub.textures);
    s.mesh_scales.append(&mut sub.mesh_scales);
    Ok(())
}

fn read(inc: &Include, base_dir: &Path, field: &str) -> Result<SceneDesc, EsSceneError> {
    let path = base_dir.join(&inc.source);
    let src = format!("{field}.source");
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => return refuse(src, format!("cannot read `{}`: {e}", path.display())),
    };
    let text = || String::from_utf8_lossy(&bytes).into_owned();
    let fail = |e: &dyn std::fmt::Display| refuse(src.clone(), format!("`{}`: {e}", inc.source));
    match ext.as_str() {
        "xml" | "mjcf" => crate::parse_mjcf(&text())
            .map(|i| i.scene)
            .or_else(|e| fail(&e)),
        "urdf" => crate::urdf::parse_urdf(&text(), &crate::urdf::PackageResolver::from_env())
            .map(|i| i.scene)
            .or_else(|e| fail(&e)),
        "gltf" | "glb" => crate::gltf::import_gltf(&bytes, path.parent())
            .map(|i| i.scene)
            .or_else(|e| fail(&e)),
        // ponytail: es-usd is this crate's layer-2 sibling and gives a stage, not a SceneDesc;
        // a USD include waits for a reader `es-assets` may call.
        "usd" | "usda" | "usdc" => fail(&"USD has no SceneDesc reader in es-assets yet"),
        _ => fail(&"not MJCF (.xml), URDF (.urdf) or glTF (.gltf/.glb)"),
    }
}

/// The files an include names are relative to the include; `mesh::load` resolves against the
/// document's directory, so each path not yet decoded gets the include's directory in front.
fn rebase(sub: &mut SceneDesc, dir: &str) {
    if dir.is_empty() {
        return;
    }
    let join = |p: &str| {
        if p.is_empty() || Path::new(p).is_absolute() {
            p.to_owned()
        } else {
            format!("{dir}/{p}")
        }
    };
    for a in &mut sub.assets {
        let undecoded = match a.kind {
            AssetKind::Mesh => !sub.meshes.contains_key(&a.id),
            AssetKind::Texture => sub.textures.get(&a.id).is_some_and(|t| t.data.is_none()),
            AssetKind::HeightField => true,
            AssetKind::Material => false,
        };
        if undecoded {
            *a = AssetRef::from_path(a.kind, &a.name, &join(&a.path));
        }
    }
    for t in sub.textures.values_mut().filter(|t| t.data.is_none()) {
        t.spec.file = t.spec.file.as_deref().map(join);
        for f in t.spec.cubefiles.iter_mut().flatten() {
            *f = join(f);
        }
    }
}

fn apply_set(
    sub: &mut SceneDesc,
    set: &super::IncludeSet,
    doc: &EsScene,
    field: &str,
) -> Result<(), EsSceneError> {
    for (name, v) in &set.joint {
        let f = format!("{field}.joint.{name}");
        let Some(j) = sub.joints.iter_mut().find(|j| j.name == *name) else {
            return refuse(f, format!("the included file has no joint `{name}`"));
        };
        j.range = v.range.map(|[lo, hi]| (lo, hi)).or(j.range);
        j.damping = v.damping.unwrap_or(j.damping);
        j.armature = v.armature.unwrap_or(j.armature);
        j.stiffness = v.stiffness.unwrap_or(j.stiffness);
        j.friction_loss = v.frictionloss.unwrap_or(j.friction_loss);
    }
    for (name, v) in &set.actuator {
        let f = format!("{field}.actuator.{name}");
        let Some(a) = sub.actuators.iter_mut().find(|a| a.name == *name) else {
            return refuse(f, format!("the included file has no actuator `{name}`"));
        };
        match (&mut a.kind, v.kp, v.kv) {
            (ActuatorKind::Position { kp, kv }, p, v) => {
                *kp = p.unwrap_or(*kp);
                *kv = v.unwrap_or(*kv);
            }
            (ActuatorKind::Velocity { kv }, None, v) => *kv = v.unwrap_or(*kv),
            (_, None, None) => {}
            _ => return refuse(f, "`kp` / `kv` set on an actuator that has no such gain"),
        }
        a.ctrl_range = v.ctrlrange.map(|[lo, hi]| (lo, hi)).or(a.ctrl_range);
        a.force_range = v.forcerange.map(|[lo, hi]| (lo, hi)).or(a.force_range);
    }
    for (name, v) in &set.geom {
        let f = format!("{field}.geom.{name}");
        let material = match &v.material {
            None => None,
            // The file's own material, else one of this document's (its id is its name's).
            Some(m) => Some(
                match asset(sub, AssetKind::Material, m, &format!("{f}.material")) {
                    Ok(id) => id,
                    Err(_) if doc.materials.iter().any(|d| d.name == *m) => {
                        scene_id("asset", &format!("material/{m}"))
                    }
                    Err(e) => return Err(e),
                },
            ),
        };
        let Some(g) = sub
            .bodies
            .iter_mut()
            .flat_map(|b| &mut b.geoms)
            .find(|g| g.name == *name)
        else {
            return refuse(f, format!("the included file has no geom `{name}`"));
        };
        g.rgba = v.rgba.unwrap_or(g.rgba);
        g.material = material.or(g.material);
    }
    Ok(())
}

/// Prefixes every name and re-derives every id by the MJCF name path scheme, then points every
/// reference at the new ids.
fn rename(s: &mut SceneDesc, p: &str) {
    let w = world();
    let mut map: BTreeMap<StableId, StableId> = BTreeMap::new();
    for b in &mut s.bodies {
        if b.id != w {
            b.name = format!("{p}{}", b.name);
        }
    }
    // Paths from the renamed names; computed before any id changes, as parents are looked up
    // by the old ids.
    let paths: BTreeMap<StableId, String> = s
        .bodies
        .iter()
        .map(|b| (b.id, body_path(s, b.id)))
        .collect();
    for b in &mut s.bodies {
        let path = &paths[&b.id];
        for g in &mut b.geoms {
            g.name = format!("{p}{}", g.name);
            map.insert(g.id, scene_id("geom", &format!("{path}/{}", g.name)));
        }
        for site in &mut b.sites {
            site.name = format!("{p}{}", site.name);
            map.insert(site.id, scene_id("site", &format!("{path}/{}", site.name)));
        }
        if b.id != w {
            map.insert(b.id, scene_id("body", path));
        }
    }
    for j in &mut s.joints {
        j.name = format!("{p}{}", j.name);
        let path = paths.get(&j.body).map_or("world", String::as_str);
        map.insert(j.id, scene_id("joint", &format!("{path}/{}", j.name)));
    }
    for c in &mut s.cameras {
        c.name = format!("{p}{}", c.name);
        let path = c
            .body
            .and_then(|b| paths.get(&b))
            .map_or("world", String::as_str);
        map.insert(c.id, scene_id("camera", &format!("{path}/{}", c.name)));
    }
    for a in &mut s.actuators {
        a.name = format!("{p}{}", a.name);
        map.insert(a.id, scene_id("actuator", &a.name));
    }
    for x in &mut s.sensors {
        x.name = format!("{p}{}", x.name);
        map.insert(x.id, scene_id("sensor", &x.name));
    }
    for t in &mut s.tendons {
        t.name = format!("{p}{}", t.name);
        map.insert(t.id, scene_id("tendon", &t.name));
    }
    for a in &mut s.assets {
        a.name = format!("{p}{}", a.name);
        map.insert(
            a.id,
            scene_id("asset", &format!("{}/{}", a.kind.tag(), a.name)),
        );
    }
    for mesh in s.meshes.values_mut() {
        mesh.name = format!("{p}{}", mesh.name);
    }
    remap(s, &map);
}

/// Every id field of `s` through `map` (an id not in it is kept).
fn remap(s: &mut SceneDesc, map: &BTreeMap<StableId, StableId>) {
    let m = |id: &mut StableId| *id = *map.get(id).unwrap_or(id);
    let rekey = |id: StableId| *map.get(&id).unwrap_or(&id);
    for b in &mut s.bodies {
        m(&mut b.id);
        b.parent.iter_mut().for_each(m);
        for g in &mut b.geoms {
            m(&mut g.id);
            g.material.iter_mut().for_each(m);
            if let Shape::Mesh { asset } | Shape::HeightField { asset } = &mut g.shape {
                m(asset);
            }
        }
        b.sites.iter_mut().for_each(|x| m(&mut x.id));
    }
    for j in &mut s.joints {
        m(&mut j.id);
        m(&mut j.body);
    }
    for c in &mut s.cameras {
        m(&mut c.id);
        c.body.iter_mut().for_each(m);
    }
    for a in &mut s.actuators {
        m(&mut a.id);
        let (ActuatorTarget::Joint(t) | ActuatorTarget::Tendon(t) | ActuatorTarget::Site(t)) =
            &mut a.target;
        m(t);
    }
    for x in &mut s.sensors {
        m(&mut x.id);
        let (SensorTarget::Body(t)
        | SensorTarget::Joint(t)
        | SensorTarget::Geom(t)
        | SensorTarget::Site(t)
        | SensorTarget::Camera(t)
        | SensorTarget::Actuator(t)) = &mut x.target;
        m(t);
    }
    for t in &mut s.tendons {
        m(&mut t.id);
        match &mut t.kind {
            TendonKind::Fixed { joints } => joints.iter_mut().for_each(|(j, _)| m(j)),
            TendonKind::Spatial { sites } => sites.iter_mut().for_each(m),
        }
    }
    s.assets.iter_mut().for_each(|a| m(&mut a.id));
    for p in &mut s.contact_pairs {
        m(&mut p.geom1);
        m(&mut p.geom2);
    }
    for (a, b) in &mut s.contact_excludes {
        m(a);
        m(b);
    }
    s.gravcomp = std::mem::take(&mut s.gravcomp)
        .into_iter()
        .map(|(k, v)| (rekey(k), v))
        .collect();
    s.textures = std::mem::take(&mut s.textures)
        .into_iter()
        .map(|(k, v)| (rekey(k), v))
        .collect();
    s.mesh_scales = std::mem::take(&mut s.mesh_scales)
        .into_iter()
        .map(|(k, v)| (rekey(k), v))
        .collect();
    s.meshes = std::mem::take(&mut s.meshes)
        .into_iter()
        .map(|(k, mut v)| {
            m(&mut v.id);
            v.material.iter_mut().for_each(m);
            (rekey(k), v)
        })
        .collect();
    s.materials = std::mem::take(&mut s.materials)
        .into_iter()
        .map(|(k, mut v)| {
            for t in [
                &mut v.rgb,
                &mut v.orm,
                &mut v.metallic_map,
                &mut v.roughness_map,
                &mut v.normal_map,
                &mut v.emissive_map,
            ] {
                t.iter_mut().for_each(m);
            }
            (rekey(k), v)
        })
        .collect();
}
