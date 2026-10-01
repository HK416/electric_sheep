//! The `observe` section: camera, state and privileged channels, each bound to its graph value
//! by an `ObservationSpec` node, and the port types and render settings they declare.

use std::time::Duration;

use es_core::StableId;
use es_ir::graph::NodeId;
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    ShutterModel,
};
use es_ir::task::{JointQuantity, ObsChannel, ObsSource, SensorPath, SensorRender, TaskNode};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};
// A number written into a document has musl's bits, not the host's (spec 5.3, M17 G3d).
use es_math::approx;

use super::{no, Compiler};
use crate::spec::{refuse, Observe, RenderDoc, RenderPath, SpecError};

impl Compiler<'_> {
    fn channel(
        &mut self,
        at: &str,
        name: &str,
        source: ObsSource,
        ty: PortType,
    ) -> Result<(), SpecError> {
        if self.channels.contains_key(name) {
            return refuse(at, name, "a second channel of that name");
        }
        self.channels
            .insert(name.to_owned(), ObsChannel { source, ty });
        Ok(())
    }

    fn bind(&mut self, from: (NodeId, &str), channel: &str) -> PortType {
        let ty = self.out(from);
        let channel = channel.to_owned();
        self.feed(from, "value", |ty| TaskNode::ObservationSpec {
            channel,
            ty,
        });
        ty
    }

    pub(super) fn observe(&mut self, o: &Observe) -> Result<(), SpecError> {
        let hz = self.spec.control_hz as f32;
        let render = render(o.render.as_ref())?;
        let cameras = o.cameras.clone().unwrap_or_default();
        if !cameras.is_empty() && o.camera_px.is_none() {
            return refuse("observe", "camera_px", "required with cameras");
        }
        for name in cameras {
            let scene = self.scene;
            let Some(cam) = scene.cameras.iter().find(|c| c.name == name) else {
                return refuse("observe", "cameras", no("camera", &name));
            };
            let ty = camera_ty(cam.id, cam.fovy, o.camera_px.expect("checked"), hz);
            let get = self.add(TaskNode::GetSensor {
                sensor: cam.id,
                ty: ty.clone(),
            });
            let channel = format!("rgb_{name}");
            self.bind((get, "value"), &channel);
            let source = ObsSource::Sensor {
                id: cam.id,
                format: ChannelFormat::Rgb,
                render,
            };
            self.channel("observe.cameras", &channel, source, ty)?;
        }
        for (table, map) in [("state", &o.state), ("privileged", &o.privileged)] {
            for (name, what) in map.iter().flat_map(|m| m.iter()) {
                self.state_channel(&format!("observe.{table}.{name}"), name, what)?;
            }
        }
        Ok(())
    }

    fn state_channel(&mut self, at: &str, name: &str, what: &str) -> Result<(), SpecError> {
        let n = self.joints.len();
        let names: Vec<String> = self.joints.iter().map(|j| j.name.clone()).collect();
        let root = self.root.id;
        let joints = |quantity| TaskNode::GetJointState {
            body: root,
            joints: names.clone(),
            quantity,
        };
        match what {
            "robot.joint_pos" | "robot.joint_vel" => {
                // A `JointState` channel naming a body reads the leading `dof` of the row.
                let leading = self.scene.joints.iter().take(n).map(|j| j.id);
                if !leading.eq(self.joints.iter().map(|j| j.id)) || n == 0 {
                    return refuse(
                        at,
                        name,
                        "the robot's joints are not the scene's leading ones",
                    );
                }
                let (quantity, body) = if what == "robot.joint_pos" {
                    (JointQuantity::Position, self.root.id)
                } else {
                    // The root body is the position channel's id (one input buffer per id), so
                    // the velocity channel names the first joint, whose dofs it reads from.
                    (JointQuantity::Velocity, self.joints[0].id)
                };
                let get = self.source(joints(quantity));
                let ty = self.bind((get, "value"), name);
                let source = ObsSource::JointState {
                    body,
                    dof: n as u32,
                    quantity,
                };
                self.channel(at, name, source, ty)
            }
            "robot.previous_action" => {
                let mut initial = Vec::new();
                for a in &self.scene.actuators {
                    let Some((lo, hi)) = a.ctrl_range else {
                        return refuse(at, name, format!("actuator `{}` has no ctrlrange", a.name));
                    };
                    // The centre: Isaac Lab's zero raw action, in actuator units. Written as the
                    // generator it replaces wrote it, so the bits are the committed ones.
                    #[allow(clippy::manual_midpoint)]
                    initial.push((lo + hi) / 2.0);
                }
                let ty = f32v(initial.len() as u64, Unit::Angle, Frame::World);
                let source = ObsSource::PreviousAction {
                    initial: Some(initial),
                };
                self.channel(at, name, source, ty)
            }
            _ => {
                let Some((body, field)) = self.dotted(what) else {
                    return refuse(at, name, no("source", what));
                };
                let (source, get, ports) = match field.as_str() {
                    "pose" => (
                        ObsSource::BodyPose(body.id),
                        self.pose(body.id),
                        ["pos", "quat"],
                    ),
                    "qpos" => {
                        let j = self.free_joint(at, name, body)?;
                        let source = ObsSource::JointState {
                            body: j.id,
                            dof: 7,
                            quantity: JointQuantity::Position,
                        };
                        (source, self.pose(body.id), ["pos", "quat"])
                    }
                    "vel" => {
                        let j = self.free_joint(at, name, body)?;
                        let source = ObsSource::JointState {
                            body: j.id,
                            dof: 6,
                            quantity: JointQuantity::Velocity,
                        };
                        let get = self.source(TaskNode::GetBodyVelocity {
                            body: body.id,
                            relative_to: Frame::World,
                        });
                        (source, get, ["linear", "angular"])
                    }
                    _ => return refuse(at, name, no("source", what)),
                };
                let parts = vec![self.out((get, ports[0])), self.out((get, ports[1]))];
                let cat = self.add(TaskNode::Concat { parts, axis: 0 });
                self.graph.connect(get, ports[0], cat, "in0");
                self.graph.connect(get, ports[1], cat, "in1");
                let ty = self.bind((cat, "value"), name);
                self.channel(at, name, source, ty)
            }
        }
    }
}

fn render(doc: Option<&RenderDoc>) -> Result<SensorRender, SpecError> {
    let Some(r) = doc else {
        return Ok(SensorRender::default());
    };
    let path = match (r.path, r.spp, r.bounces) {
        (Some(RenderPath::Pt), Some(spp), Some(bounces)) => SensorPath::Pt { spp, bounces },
        (Some(RenderPath::Pt), None, _) => {
            return refuse("observe.render", "spp", "required on `pt`")
        }
        (Some(RenderPath::Pt), _, None) => {
            return refuse("observe.render", "bounces", "required on `pt`")
        }
        (_, Some(_), _) => return refuse("observe.render", "spp", "`pt` only"),
        (_, _, Some(_)) => return refuse("observe.render", "bounces", "`pt` only"),
        _ => SensorPath::Rs,
    };
    let d = SensorRender::default();
    Ok(SensorRender {
        path,
        exposure: r.exposure.unwrap_or(d.exposure),
        tonemap: r.tonemap.unwrap_or(d.tonemap),
        seed: r.seed.unwrap_or(d.seed),
        svgf: r.svgf.unwrap_or(d.svgf),
    })
}

fn f32v(n: u64, unit: Unit, frame: Frame) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([n]),
        unit,
        frame,
        time: TimeRef::Tick,
        image: None,
    }
}

/// A `px`×`px` RGB8 pinhole camera of vertical field of view `fovy` (rad), as the renderer
/// delivers it (`es_env::render::image_spec`).
fn camera_ty(camera: StableId, fovy: f64, px: u32, hz: f32) -> PortType {
    let f = f64::from(px) / 2.0 / approx::tan_f64(fovy / 2.0);
    let c = f64::from(px) / 2.0;
    PortType {
        elem: ElemType::U8,
        shape: Shape::new([u64::from(px), u64::from(px), 3]),
        unit: Unit::Pixel,
        frame: Frame::Camera(camera),
        time: TimeRef::Sensor {
            id: camera,
            align: Align::Hold,
        },
        image: Some(ImageSpec {
            width: px,
            height: px,
            channels: ChannelFormat::Rgb,
            dtype: ImageDType::U8,
            color_space: ColorSpace::SRgb,
            camera_model: CameraModel::Pinhole,
            intrinsics: Intrinsics::new(f, f, c, c),
            extrinsics: es_math::Pose::IDENTITY,
            distortion: DistortionModel::None,
            shutter: ShutterModel::Global,
            exposure: Duration::from_millis(2),
            rate_hz: hz,
            depth_scale: None,
        }),
    }
}
