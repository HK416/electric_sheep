//! [`CliError`]: the one error type every `es` subcommand returns.

use std::fmt;

/// A CLI-level failure. `Usage` means the arguments themselves are wrong (exit code 2);
/// `Runtime` means the arguments parsed but something they named failed (exit code 1).
#[derive(Debug)]
pub enum CliError {
    Usage(String),
    Runtime(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(m) | Self::Runtime(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for CliError {}
