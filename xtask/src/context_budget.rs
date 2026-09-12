//! `cargo xtask context-budget` — spec 1.5: core crates target <= 6,000 / hard cap
//! <= 10,000 source lines, excluding `tests/` and `benches/` directories.

use std::fs;
use std::path::Path;

const WARN_AT: usize = 6_000;
const FAIL_AT: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    Warn,
    Fail,
}

impl Status {
    pub fn from_lines(lines: usize) -> Self {
        if lines > FAIL_AT {
            Status::Fail
        } else if lines > WARN_AT {
            Status::Warn
        } else {
            Status::Ok
        }
    }

    fn label(self) -> &'static str {
        match self {
            Status::Ok => "OK",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
        }
    }
}

/// Recursively count `*.rs` files under `src/`, skipping any path that has a
/// `tests` or `benches` path component (defense in depth: crates are only
/// expected to keep non-test sources under `src/`).
fn count_rs_lines(dir: &Path) -> usize {
    let mut total = 0;
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "tests" || name == "benches" {
                continue;
            }
            total += count_rs_lines(&path);
        } else if path.extension().is_some_and(|e| e == "rs") {
            total += fs::read_to_string(&path).map_or(0, |s| s.lines().count());
        }
    }
    total
}

pub fn run(crates_dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(crates_dir) else {
        println!("no crates/ directory found; nothing to budget");
        return true;
    };
    let mut rows: Vec<(String, usize, Status)> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let lines = count_rs_lines(&e.path().join("src"));
            (name, lines, Status::from_lines(lines))
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    println!("{:<24} {:>8}  status", "crate", "lines");
    for (name, lines, status) in &rows {
        println!("{name:<24} {lines:>8}  {}", status.label());
    }

    rows.iter().all(|(_, _, status)| *status != Status::Fail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_thresholds() {
        assert_eq!(Status::from_lines(0), Status::Ok);
        assert_eq!(Status::from_lines(6_000), Status::Ok);
        assert_eq!(Status::from_lines(6_001), Status::Warn);
        assert_eq!(Status::from_lines(10_000), Status::Warn);
        assert_eq!(Status::from_lines(10_001), Status::Fail);
    }
}
