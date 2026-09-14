//! `ActuatorPublisher`: the only writer allowed on a `[[ros2.actuator]]` topic
//! (`docs/design/ros2-boundary.md` section 4.5, spec 9.1, INV-12). `NJ` is checked against the
//! configured joint count at construction; `send` only ever takes a [`SafeAction`], so there is
//! no path from this crate that puts an unvalidated action on an actuator topic.

use es_safety::SafeAction;

use crate::config::ActuatorTopic;
use crate::error::Ros2Error;
use crate::msg::{Float64MultiArray, JointState, Msg, MsgType, MultiArrayLayout};
use crate::session::{Publisher, Ros2Node};

/// Publishes `std_msgs/msg/Float64MultiArray { layout: { dim: [], data_offset: 0 }, data: q }`:
/// the command type `ros2_control`'s `forward_command_controller` subscribes on `"~/commands"`
/// (`using CmdType = std_msgs::msg::Float64MultiArray;`), which itself refuses a size mismatch.
pub struct ActuatorPublisher<const NJ: usize> {
    inner: Publisher,
    #[allow(dead_code)] // kept for introspection/debugging; not read by `send`.
    topic: ActuatorTopic,
}

impl<const NJ: usize> std::fmt::Debug for ActuatorPublisher<NJ> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActuatorPublisher")
            .field("nj", &NJ)
            .field("topic", &self.topic.topic)
            .finish_non_exhaustive()
    }
}

impl<const NJ: usize> ActuatorPublisher<NJ> {
    pub(crate) fn new(inner: Publisher, topic: ActuatorTopic) -> Self {
        Self { inner, topic }
    }

    pub fn send(&self, action: &SafeAction<NJ>) -> Result<(), Ros2Error> {
        let msg = Msg::Float64MultiArray(Float64MultiArray {
            layout: MultiArrayLayout {
                dim: Vec::new(),
                data_offset: 0,
            },
            data: action.q.to_vec(),
        });
        self.inner.put(&msg)
    }
}

impl Ros2Node {
    /// `NJ != joints.len()` fails here, before any zenoh resource is declared (design note
    /// section 4.5).
    pub fn actuator<const NJ: usize>(
        &self,
        topic: &str,
    ) -> Result<ActuatorPublisher<NJ>, Ros2Error> {
        let cfg_topic = self.config().actuator_topic::<NJ>(topic)?.clone();
        let publisher = self.declare_publisher_raw(topic, MsgType::Float64MultiArray, 1)?;
        Ok(ActuatorPublisher::new(publisher, cfg_topic))
    }
}

/// Reorders an inbound `sensor_msgs/JointState` into the configured joint order (design note
/// section 4.5): a joint the sample does not carry rejects the whole sample (`ROS2-013`) rather
/// than silently leaving it at zero, so a partial state never reaches
/// `SafetyPlane::observe_state` / `sensor_seen`.
///
/// The one exception the message definition forces: `velocity` "may be empty", and an entirely
/// empty `velocity` means "this driver does not report velocity", which reads as `qd = [0.0; NJ]`.
/// A non-empty but too-short `velocity` is a partial sample and is refused; `position` has no such
/// reading, so short or empty is always an error.
pub fn reorder_joint_state<const NJ: usize>(
    joints: &[String],
    state: &JointState,
) -> Result<([f64; NJ], [f64; NJ]), Ros2Error> {
    let velocity_reported = !state.velocity.is_empty();
    let mut q = [0.0f64; NJ];
    let mut qd = [0.0f64; NJ];
    for (i, name) in joints.iter().enumerate() {
        let idx = state
            .name
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| Ros2Error::MissingJoint(name.clone()))?;
        q[i] = *state
            .position
            .get(idx)
            .ok_or_else(|| partial(name, "position"))?;
        if velocity_reported {
            qd[i] = *state
                .velocity
                .get(idx)
                .ok_or_else(|| partial(name, "velocity"))?;
        }
    }
    Ok((q, qd))
}

fn partial(joint: &str, field: &'static str) -> Ros2Error {
    Ros2Error::PartialJointState {
        joint: joint.to_owned(),
        field,
    }
}

#[cfg(test)]
// Exactness is the property under test: `reorder_joint_state` copies values, never computes on
// them, so bit-exact comparison is correct here (see es-safety's `nostd_core.rs` for the same
// pattern).
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::msg::Header;
    use crate::msg::Time;

    fn state(name: &[&str], position: &[f64], velocity: &[f64]) -> JointState {
        JointState {
            header: Header {
                stamp: Time { sec: 0, nanosec: 0 },
                frame_id: String::new(),
            },
            name: name.iter().map(|s| (*s).to_owned()).collect(),
            position: position.to_vec(),
            velocity: velocity.to_vec(),
            effort: Vec::new(),
        }
    }

    #[test]
    fn reorders_by_name_and_rejects_a_missing_joint() {
        let joints = vec!["j2".to_owned(), "j1".to_owned()];
        let s = state(&["j1", "j2"], &[1.0, 2.0], &[10.0, 20.0]);
        let (q, qd) = reorder_joint_state::<2>(&joints, &s).unwrap();
        assert_eq!(q, [2.0, 1.0]);
        assert_eq!(qd, [20.0, 10.0]);

        let joints = vec!["j1".to_owned(), "j3".to_owned()];
        let err = reorder_joint_state::<2>(&joints, &s).unwrap_err();
        assert_eq!(err.code(), "ROS2-013");
    }

    #[test]
    fn a_short_position_array_is_refused() {
        let joints = vec!["j1".to_owned(), "j2".to_owned()];
        let s = state(&["j1", "j2"], &[1.0], &[10.0, 20.0]);
        let err = reorder_joint_state::<2>(&joints, &s).unwrap_err();
        assert_eq!(err.code(), "ROS2-013");
        let msg = err.to_string();
        assert!(msg.contains("j2"), "{msg}");
        assert!(msg.contains("position"), "{msg}");
    }

    #[test]
    fn a_short_velocity_array_is_refused() {
        let joints = vec!["j1".to_owned(), "j2".to_owned()];
        let s = state(&["j1", "j2"], &[1.0, 2.0], &[10.0]);
        let err = reorder_joint_state::<2>(&joints, &s).unwrap_err();
        assert_eq!(err.code(), "ROS2-013");
        let msg = err.to_string();
        assert!(msg.contains("j2"), "{msg}");
        assert!(msg.contains("velocity"), "{msg}");
    }

    #[test]
    fn an_empty_velocity_array_is_the_documented_zero() {
        let joints = vec!["j2".to_owned(), "j1".to_owned()];
        let s = state(&["j1", "j2"], &[1.0, 2.0], &[]);
        let (q, qd) = reorder_joint_state::<2>(&joints, &s).unwrap();
        assert_eq!(q, [2.0, 1.0]);
        assert_eq!(qd, [0.0, 0.0]);
    }

    #[test]
    fn an_empty_position_array_is_refused() {
        let joints = vec!["j1".to_owned(), "j2".to_owned()];
        let s = state(&["j1", "j2"], &[], &[10.0, 20.0]);
        let err = reorder_joint_state::<2>(&joints, &s).unwrap_err();
        assert_eq!(err.code(), "ROS2-013");
        assert!(err.to_string().contains("position"), "{err}");
    }
}
