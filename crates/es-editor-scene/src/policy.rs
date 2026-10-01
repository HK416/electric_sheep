//! The camera the policy sees, for the viewport's corner (packet M17/G6, design section 5): which
//! cameras a project's policy is given, how each is declared (its size and render path, the Task
//! IR channel's), and which one the corner shows. Rendering it is `es-editor`'s, in process.
//!
//! - An editable project with a task specification: the student's `views`, else `[observe]
//!   cameras`, as `compile_task` (G3a) declares them on the scene as edited.
//! - A template, and an editable copy without a specification (it trains on the template's
//!   documents until G9): the cameras its bundle's Observation IR reads, as its Task IR declares
//!   them ([`bundle`]).

use std::path::Path;

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_ir::image::ImageSpec;
use es_ir::observation::{ObservationIr, ObservationNode};
use es_ir::task::{ObsSource, SensorRender, TaskIr};
use es_script::spec::compile_task;

use crate::command::Entity;
use crate::model::SceneModel;

/// One camera the policy is given, as its Task IR channel declares it.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyCamera {
    /// The scene camera's name.
    pub name: String,
    pub sensor: StableId,
    /// The declared picture: its size is the frame's.
    pub image: ImageSpec,
    pub render: SensorRender,
}

/// The cameras a bundle reads: its Observation IR's pictures (the file `observation`), as its
/// Task IR (the file `task`) declares them, named by `scene`'s cameras.
pub fn bundle(
    task: &Path,
    observation: &Path,
    scene: &SceneDesc,
) -> Result<Vec<PolicyCamera>, String> {
    let read = |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
    let task = es_ir::serial::task_from_toml(&read(task)?).map_err(|e| e.to_string())?;
    let obs = es_ir::serial::observation_from_toml(&read(observation)?);
    Ok(declared(
        &task,
        scene,
        &observed(&obs.map_err(|e| e.to_string())?),
    ))
}

/// The sensors an Observation IR reads its pictures from, in node order.
pub fn observed(obs: &ObservationIr) -> Vec<StableId> {
    (obs.graph.nodes.values())
        .filter_map(|n| match n {
            ObservationNode::ImageInput { sensor, .. } => Some(*sensor),
            _ => None,
        })
        .collect()
}

/// The picture channels of `task` from each of `sensors`, in that order, each named by its
/// camera in `scene`. A sensor no channel reads, or that is no camera of `scene`, is left out.
pub fn declared(task: &TaskIr, scene: &SceneDesc, sensors: &[StableId]) -> Vec<PolicyCamera> {
    let channel = |id: &StableId| {
        (task.observation_spec.channels.values()).find_map(|ch| match &ch.source {
            ObsSource::Sensor { id: s, render, .. } if s == id => Some((ch.ty.image?, *render)),
            _ => None,
        })
    };
    (sensors.iter())
        .filter_map(|id| {
            let camera = scene.cameras.iter().find(|c| c.id == *id)?;
            let (image, render) = channel(id)?;
            Some(PolicyCamera {
                name: camera.name.clone(),
                sensor: *id,
                image,
                render,
            })
        })
        .collect()
}

/// The one the corner shows: the selected camera when the policy sees it, else the first.
pub fn shown<'a>(
    cameras: &'a [PolicyCamera],
    selected: Option<&Entity>,
) -> Option<&'a PolicyCamera> {
    let picked = match selected {
        Some(Entity::Camera(n)) => cameras.iter().find(|c| c.name == *n),
        _ => None,
    };
    picked.or_else(|| cameras.first())
}

impl SceneModel {
    /// The cameras this project's policy sees, from its task specification on the scene as
    /// edited: the student's views, else every observed camera. Empty without a specification
    /// (the template's documents stand) or with one that observes no camera.
    pub fn policy_cameras(&mut self) -> Result<Vec<PolicyCamera>, String> {
        let Some(mut spec) = self.spec().cloned() else {
            return Ok(Vec::new());
        };
        let views = spec.student.as_ref().and_then(|s| s.views.clone());
        let observed = spec.observe.as_ref().and_then(|o| o.cameras.clone());
        let names = views.or(observed).unwrap_or_default();
        if names.is_empty() {
            return Ok(Vec::new());
        }
        // The scene as edited: the saved document, or the unsaved one written beside it.
        let file = self.scene_file()?;
        spec.scene = (file.file_name().map(|f| f.to_string_lossy().into_owned()))
            .ok_or_else(|| file.display().to_string())?;
        let task = compile_task(&spec, self.root()).map_err(|e| e.to_string())?;
        let scene = self.scene();
        let sensors: Vec<StableId> = (names.iter())
            .filter_map(|n| scene.cameras.iter().find(|c| c.name == *n).map(|c| c.id))
            .collect();
        Ok(declared(&task, scene, &sensors))
    }
}
