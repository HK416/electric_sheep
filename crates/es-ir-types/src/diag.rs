//! Compiler diagnostics (P27): the value every IR check returns when it fails.
//!
//! A diagnostic names a code from [`crate::codes`]; the title and the severity come from that
//! dictionary, so a call site only has to supply what is specific to the occurrence.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::codes;
use crate::graph::NodeId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Error => "ERROR",
            Self::Warning => "WARN",
            Self::Info => "INFO",
        })
    }
}

/// A dictionary code such as `"TYPE-014"`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DiagCode(String);

impl DiagCode {
    pub fn new(code: &str) -> Self {
        Self(code.to_owned())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The dictionary row, if the code is known.
    pub fn entry(&self) -> Option<&'static codes::CodeEntry> {
        codes::lookup(&self.0)
    }
}

impl fmt::Display for DiagCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One compiler complaint. Construct with [`Diagnostic::new`] and refine with the builder
/// methods; the severity always comes from the code dictionary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: DiagCode,
    pub severity: Severity,
    pub message: String,
    pub node: Option<NodeId>,
    pub port: Option<String>,
    pub hint: Option<String>,
}

impl Diagnostic {
    /// A diagnostic for `code`; an unknown code is reported as an error.
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        let severity = codes::lookup(code).map_or(Severity::Error, |e| e.severity);
        Self {
            code: DiagCode::new(code),
            severity,
            message: message.into(),
            node: None,
            port: None,
            hint: None,
        }
    }

    #[must_use]
    pub fn at(mut self, node: NodeId) -> Self {
        self.node = Some(node);
        self
    }

    #[must_use]
    pub fn on_port(mut self, port: impl Into<String>) -> Self {
        self.port = Some(port.into());
        self
    }

    #[must_use]
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// The dictionary title, or the empty string for an unknown code.
    pub fn title(&self) -> &'static str {
        self.code.entry().map_or("", |e| e.title)
    }

    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// The spec's block format (spec 5.4, spec 7.2): header line, indented body, indented hint.
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.severity, self.code)?;
        let title = self.title();
        if !title.is_empty() {
            write!(f, "  {title}")?;
        }
        writeln!(f)?;
        writeln!(f)?;
        match (self.node, &self.port) {
            (Some(n), Some(p)) => writeln!(f, "  at node {}, port \"{p}\"", n.0)?,
            (Some(n), None) => writeln!(f, "  at node {}", n.0)?,
            (None, Some(p)) => writeln!(f, "  at port \"{p}\"")?,
            (None, None) => {}
        }
        for line in self.message.lines() {
            writeln!(f, "  {line}")?;
        }
        if let Some(hint) = &self.hint {
            writeln!(f)?;
            writeln!(f, "  hint: {hint}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Diagnostic {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_comes_from_the_dictionary() {
        assert_eq!(
            Diagnostic::new(codes::DEP_114, "x").severity,
            Severity::Warning
        );
        assert_eq!(
            Diagnostic::new(codes::TYPE_014, "x").severity,
            Severity::Error
        );
        // Unknown codes are still reportable, as errors, with no title.
        let unknown = Diagnostic::new("ZZZ-999", "x");
        assert_eq!(unknown.severity, Severity::Error);
        assert_eq!(unknown.title(), "");
    }

    #[test]
    fn display_matches_the_spec_block_format() {
        let d = Diagnostic::new(
            codes::TYPE_014,
            "joint_pos: Sensor(encoder)\ncamera: Sensor(cam)",
        )
        .at(NodeId(7))
        .on_port("policy_obs")
        .with_hint("time_align = \"hold\" | \"interpolate\" | \"reject\"");
        let expected = concat!(
            "ERROR TYPE-014  time alignment is not specified\n",
            "\n",
            "  at node 7, port \"policy_obs\"\n",
            "  joint_pos: Sensor(encoder)\n",
            "  camera: Sensor(cam)\n",
            "\n",
            "  hint: time_align = \"hold\" | \"interpolate\" | \"reject\"\n",
        );
        assert_eq!(d.to_string(), expected);
    }

    #[test]
    fn display_omits_absent_parts() {
        let d = Diagnostic::new(codes::GRAPH_001, "a -> b -> a");
        assert_eq!(
            d.to_string(),
            "ERROR GRAPH-001  the graph contains a cycle\n\n  a -> b -> a\n"
        );
    }

    #[test]
    fn serde_round_trip() {
        let d = Diagnostic::new(codes::OBS_034, "resize 640x480 -> 224x224").at(NodeId(2));
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(serde_json::from_str::<Diagnostic>(&json).unwrap(), d);
    }
}
