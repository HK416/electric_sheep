//! The fixed ROS 2 message subset this crate speaks (`docs/api-notes/ros2-cdr.md` "Message
//! subset", `docs/design/ros2-boundary.md` section 5). Field order is verbatim from the
//! upstream `.msg` files; every struct's `to_cdr`/`from_cdr` matches the rosbags 0.11.5 goldens
//! under `tests/golden/ros2/cdr/` byte-for-byte (spec 1.4: the goldens are the reference, this
//! module reproduces them, never the other way around).
//!
//! Each struct also has a private `write_fields`/`read_fields` pair used when it is nested
//! inside another message (`Header` inside `JointState`, `RegionOfInterest` inside
//! `CameraInfo`): a nested message has no struct-level padding of its own, it just aligns as
//! its first field (`docs/api-notes/ros2-cdr.md` "Layout rules"), so nesting must not write or
//! expect a second 4-byte encapsulation header.

use crate::cdr::{CdrError, CdrReader, CdrWriter};

/// `builtin_interfaces/msg/Time`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Time {
    pub sec: i32,
    pub nanosec: u32,
}

impl Time {
    fn write_fields(self, w: &mut CdrWriter) {
        w.write_i32(self.sec);
        w.write_u32(self.nanosec);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        Ok(Self {
            sec: r.read_i32()?,
            nanosec: r.read_u32()?,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// `std_msgs/msg/Header`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub stamp: Time,
    pub frame_id: String,
}

impl Header {
    fn write_fields(&self, w: &mut CdrWriter) {
        self.stamp.write_fields(w);
        w.write_string(&self.frame_id);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        Ok(Self {
            stamp: Time::read_fields(r)?,
            frame_id: r.read_string()?,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// `std_msgs/msg/String`. Named `StringMsg` to not shadow `std::string::String`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringMsg {
    pub data: String,
}

impl StringMsg {
    fn write_fields(&self, w: &mut CdrWriter) {
        w.write_string(&self.data);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        Ok(Self {
            data: r.read_string()?,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// `std_msgs/msg/MultiArrayDimension`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiArrayDimension {
    pub label: String,
    pub size: u32,
    pub stride: u32,
}

impl MultiArrayDimension {
    fn write_fields(&self, w: &mut CdrWriter) {
        w.write_string(&self.label);
        w.write_u32(self.size);
        w.write_u32(self.stride);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        Ok(Self {
            label: r.read_string()?,
            size: r.read_u32()?,
            stride: r.read_u32()?,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// `std_msgs/msg/MultiArrayLayout`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiArrayLayout {
    pub dim: Vec<MultiArrayDimension>,
    pub data_offset: u32,
}

impl MultiArrayLayout {
    fn write_fields(&self, w: &mut CdrWriter) {
        let count = u32::try_from(self.dim.len()).expect("dim fits in u32");
        w.write_u32(count);
        for d in &self.dim {
            d.write_fields(w);
        }
        w.write_u32(self.data_offset);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        let n = r.read_u32()? as usize;
        let mut dim = Vec::new();
        for _ in 0..n {
            dim.push(MultiArrayDimension::read_fields(r)?);
        }
        Ok(Self {
            dim,
            data_offset: r.read_u32()?,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// `std_msgs/msg/Float64MultiArray`.
#[derive(Debug, Clone, PartialEq)]
pub struct Float64MultiArray {
    pub layout: MultiArrayLayout,
    pub data: Vec<f64>,
}

impl Float64MultiArray {
    fn write_fields(&self, w: &mut CdrWriter) {
        self.layout.write_fields(w);
        w.write_f64_seq(&self.data);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        Ok(Self {
            layout: MultiArrayLayout::read_fields(r)?,
            data: r.read_f64_seq()?,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// `sensor_msgs/msg/JointState`.
#[derive(Debug, Clone, PartialEq)]
pub struct JointState {
    pub header: Header,
    pub name: Vec<String>,
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub effort: Vec<f64>,
}

impl JointState {
    fn write_fields(&self, w: &mut CdrWriter) {
        self.header.write_fields(w);
        w.write_string_seq(&self.name);
        w.write_f64_seq(&self.position);
        w.write_f64_seq(&self.velocity);
        w.write_f64_seq(&self.effort);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        Ok(Self {
            header: Header::read_fields(r)?,
            name: r.read_string_seq()?,
            position: r.read_f64_seq()?,
            velocity: r.read_f64_seq()?,
            effort: r.read_f64_seq()?,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// `sensor_msgs/msg/Image`.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub header: Header,
    pub height: u32,
    pub width: u32,
    pub encoding: String,
    pub is_bigendian: u8,
    pub step: u32,
    pub data: Vec<u8>,
}

impl Image {
    fn write_fields(&self, w: &mut CdrWriter) {
        self.header.write_fields(w);
        w.write_u32(self.height);
        w.write_u32(self.width);
        w.write_string(&self.encoding);
        w.write_u8(self.is_bigendian);
        w.write_u32(self.step);
        w.write_u8_seq(&self.data);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        Ok(Self {
            header: Header::read_fields(r)?,
            height: r.read_u32()?,
            width: r.read_u32()?,
            encoding: r.read_string()?,
            is_bigendian: r.read_u8()?,
            step: r.read_u32()?,
            data: r.read_u8_seq()?,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// `sensor_msgs/msg/RegionOfInterest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionOfInterest {
    pub x_offset: u32,
    pub y_offset: u32,
    pub height: u32,
    pub width: u32,
    pub do_rectify: bool,
}

impl RegionOfInterest {
    fn write_fields(&self, w: &mut CdrWriter) {
        w.write_u32(self.x_offset);
        w.write_u32(self.y_offset);
        w.write_u32(self.height);
        w.write_u32(self.width);
        w.write_bool(self.do_rectify);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        Ok(Self {
            x_offset: r.read_u32()?,
            y_offset: r.read_u32()?,
            height: r.read_u32()?,
            width: r.read_u32()?,
            do_rectify: r.read_bool()?,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// `sensor_msgs/msg/CameraInfo`.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraInfo {
    pub header: Header,
    pub height: u32,
    pub width: u32,
    pub distortion_model: String,
    pub d: Vec<f64>,
    pub k: [f64; 9],
    pub r: [f64; 9],
    pub p: [f64; 12],
    pub binning_x: u32,
    pub binning_y: u32,
    pub roi: RegionOfInterest,
}

impl CameraInfo {
    fn write_fields(&self, w: &mut CdrWriter) {
        self.header.write_fields(w);
        w.write_u32(self.height);
        w.write_u32(self.width);
        w.write_string(&self.distortion_model);
        w.write_f64_seq(&self.d);
        for x in self.k {
            w.write_f64(x);
        }
        for x in self.r {
            w.write_f64(x);
        }
        for x in self.p {
            w.write_f64(x);
        }
        w.write_u32(self.binning_x);
        w.write_u32(self.binning_y);
        self.roi.write_fields(w);
    }

    fn read_fields(r: &mut CdrReader) -> Result<Self, CdrError> {
        let header = Header::read_fields(r)?;
        let height = r.read_u32()?;
        let width = r.read_u32()?;
        let distortion_model = r.read_string()?;
        let d = r.read_f64_seq()?;
        let mut k = [0.0; 9];
        for x in &mut k {
            *x = r.read_f64()?;
        }
        let mut rot = [0.0; 9];
        for x in &mut rot {
            *x = r.read_f64()?;
        }
        let mut p = [0.0; 12];
        for x in &mut p {
            *x = r.read_f64()?;
        }
        let binning_x = r.read_u32()?;
        let binning_y = r.read_u32()?;
        let roi = RegionOfInterest::read_fields(r)?;
        Ok(Self {
            header,
            height,
            width,
            distortion_model,
            d,
            k,
            r: rot,
            p,
            binning_x,
            binning_y,
            roi,
        })
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        let mut w = CdrWriter::new();
        self.write_fields(&mut w);
        w.finish()
    }

    pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError> {
        let mut r = CdrReader::new(bytes)?;
        let v = Self::read_fields(&mut r)?;
        r.finish()?;
        Ok(v)
    }
}

/// One entry of the RIHS01 type-hash table: `docs/api-notes/ros2-cdr.md` "RIHS01 type hash" —
/// a const table, never computed (spec 24.1, `docs/design/ros2-boundary.md` section 3).
struct TypeInfo {
    ros_name: &'static str,
    dds_name: &'static str,
    rihs01: &'static str,
}

const TYPE_TABLE: [TypeInfo; 5] = [
    TypeInfo {
        ros_name: "std_msgs/msg/String",
        dds_name: "std_msgs::msg::dds_::String_",
        rihs01: "RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18",
    },
    TypeInfo {
        ros_name: "std_msgs/msg/Float64MultiArray",
        dds_name: "std_msgs::msg::dds_::Float64MultiArray_",
        rihs01: "RIHS01_1025ddc6b9552d191f89ef1a8d2f60f3d373e28b283d8891ddcc974e8c55397f",
    },
    TypeInfo {
        ros_name: "sensor_msgs/msg/JointState",
        dds_name: "sensor_msgs::msg::dds_::JointState_",
        rihs01: "RIHS01_a13ee3a330e346c9d87b5aa18d24e11690752bd33a0350f11c5882bc9179260e",
    },
    TypeInfo {
        ros_name: "sensor_msgs/msg/Image",
        dds_name: "sensor_msgs::msg::dds_::Image_",
        rihs01: "RIHS01_d31d41a9a4c4bc8eae9be757b0beed306564f7526c88ea6a4588fb9582527d47",
    },
    TypeInfo {
        ros_name: "sensor_msgs/msg/CameraInfo",
        dds_name: "sensor_msgs::msg::dds_::CameraInfo_",
        rihs01: "RIHS01_b3dfd68ff46c9d56c80fd3bd4ed22c7a4ddce8c8348f2f59c299e73118e7e275",
    },
];

/// The five top-level message types this crate dispatches on the wire (spec 24.1's message
/// subset). Nested types (`Header`, `Time`, `MultiArrayLayout`, `MultiArrayDimension`,
/// `RegionOfInterest`) never appear bare on a topic, so they have no `MsgType` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgType {
    String,
    Float64MultiArray,
    JointState,
    Image,
    CameraInfo,
}

impl MsgType {
    const fn table_index(self) -> usize {
        match self {
            MsgType::String => 0,
            MsgType::Float64MultiArray => 1,
            MsgType::JointState => 2,
            MsgType::Image => 3,
            MsgType::CameraInfo => 4,
        }
    }

    /// `<package>/msg/<Type>`, the ROS name used in a topic key expression's path segment.
    #[must_use]
    pub const fn ros_name(self) -> &'static str {
        TYPE_TABLE[self.table_index()].ros_name
    }

    /// `<namespace>::msg::dds_::<Type>_`, `type_support_common.cpp`'s mangled DDS type name.
    #[must_use]
    pub const fn dds_name(self) -> &'static str {
        TYPE_TABLE[self.table_index()].dds_name
    }

    /// `RIHS01_` + 64 lowercase hex (REP-2016), from the const table — never computed.
    #[must_use]
    pub const fn rihs01(self) -> &'static str {
        TYPE_TABLE[self.table_index()].rihs01
    }
}

/// One decoded/encodable message, tagged with which [`MsgType`] it is.
// `CameraInfo` (360 bytes) dwarfs `StringMsg`: the work packet's acceptance criteria pin this
// enum's variants to the plain struct types (no `Box`), and messages this size are decoded one
// at a time, never batched in a `Vec<Msg>` hot path, so the size difference is not a real cost.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum Msg {
    String(StringMsg),
    Float64MultiArray(Float64MultiArray),
    JointState(JointState),
    Image(Image),
    CameraInfo(CameraInfo),
}

impl Msg {
    #[must_use]
    pub fn msg_type(&self) -> MsgType {
        match self {
            Msg::String(_) => MsgType::String,
            Msg::Float64MultiArray(_) => MsgType::Float64MultiArray,
            Msg::JointState(_) => MsgType::JointState,
            Msg::Image(_) => MsgType::Image,
            Msg::CameraInfo(_) => MsgType::CameraInfo,
        }
    }

    #[must_use]
    pub fn to_cdr(&self) -> Vec<u8> {
        match self {
            Msg::String(m) => m.to_cdr(),
            Msg::Float64MultiArray(m) => m.to_cdr(),
            Msg::JointState(m) => m.to_cdr(),
            Msg::Image(m) => m.to_cdr(),
            Msg::CameraInfo(m) => m.to_cdr(),
        }
    }

    pub fn from_cdr(ty: MsgType, bytes: &[u8]) -> Result<Self, CdrError> {
        Ok(match ty {
            MsgType::String => Msg::String(StringMsg::from_cdr(bytes)?),
            MsgType::Float64MultiArray => {
                Msg::Float64MultiArray(Float64MultiArray::from_cdr(bytes)?)
            }
            MsgType::JointState => Msg::JointState(JointState::from_cdr(bytes)?),
            MsgType::Image => Msg::Image(Image::from_cdr(bytes)?),
            MsgType::CameraInfo => Msg::CameraInfo(CameraInfo::from_cdr(bytes)?),
        })
    }
}

pub mod testing {
    //! Every [`MsgType`] variant, for property tests that must exercise all of them.
    use super::MsgType;

    pub const ALL: [MsgType; 5] = [
        MsgType::String,
        MsgType::Float64MultiArray,
        MsgType::JointState,
        MsgType::Image,
        MsgType::CameraInfo,
    ];
}
