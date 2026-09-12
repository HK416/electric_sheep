//! `cargo xtask verify-goldens` — `tests/golden/**` is CI read-only (spec 1.4).
//! Additions are fine; modifications and deletions are rejected unless
//! `GOLDEN_UPDATE=1` is set.

use std::path::Path;
use std::process::Command;

const GOLDEN_DIR: &str = "tests/golden";

/// One `git diff --name-status` entry: first letter of the status code, and
/// the (final) path it applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub status: char,
    pub path: String,
}

/// Pure rule: additions (`A`) are allowed, everything else on an existing
/// golden file is a violation. Takes a fake-able list so this is unit
/// testable without shelling out to git.
pub fn classify(changes: &[Change]) -> Vec<String> {
    changes
        .iter()
        .filter(|c| c.status != 'A')
        .map(|c| {
            format!(
                "{}: golden file `{}` was modified (status {})",
                GOLDEN_DIR, c.path, c.status
            )
        })
        .collect()
}

fn parse_name_status(output: &str) -> Vec<Change> {
    output
        .lines()
        .filter_map(|line| {
            let mut fields = line.split('\t');
            let status = fields.next()?;
            let status_char = status.chars().next()?;
            // Renames/copies carry an extra "old\tnew" pair; keep the final path.
            let path = fields.next_back()?;
            Some(Change {
                status: status_char,
                path: path.to_string(),
            })
        })
        .collect()
}

fn git(workspace_root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(workspace_root)
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        None
    }
}

pub fn run(workspace_root: &Path) -> bool {
    if !workspace_root.join(GOLDEN_DIR).is_dir() {
        println!("verify-goldens: no {GOLDEN_DIR} directory, nothing to check");
        return true;
    }

    let base = git(workspace_root, &["merge-base", "HEAD", "main"])
        .or_else(|| git(workspace_root, &["merge-base", "HEAD", "origin/main"]))
        .map_or_else(|| "HEAD".to_string(), |s| s.trim().to_string());

    let mut changes = Vec::new();
    if let Some(out) = git(
        workspace_root,
        &[
            "diff",
            "--name-status",
            &format!("{base}...HEAD"),
            "--",
            GOLDEN_DIR,
        ],
    ) {
        changes.extend(parse_name_status(&out));
    }
    if let Some(out) = git(
        workspace_root,
        &["diff", "--name-status", "HEAD", "--", GOLDEN_DIR],
    ) {
        changes.extend(parse_name_status(&out));
    }

    let violations = classify(&changes);
    if violations.is_empty() {
        println!(
            "verify-goldens: {} changed golden file(s), all additions",
            changes.len()
        );
        return true;
    }

    if std::env::var("GOLDEN_UPDATE").as_deref() == Ok("1") {
        println!(
            "verify-goldens: GOLDEN_UPDATE=1, bypassing {} violation(s)",
            violations.len()
        );
        return true;
    }

    for v in &violations {
        println!("{v}");
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn change(status: char, path: &str) -> Change {
        Change {
            status,
            path: path.into(),
        }
    }

    #[test]
    fn additions_are_allowed() {
        let changes = vec![change('A', "tests/golden/a.png")];
        assert!(classify(&changes).is_empty());
    }

    #[test]
    fn modifications_and_deletions_are_rejected() {
        let changes = vec![
            change('M', "tests/golden/a.png"),
            change('D', "tests/golden/b.png"),
        ];
        let v = classify(&changes);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn parses_git_name_status_output() {
        let out = "A\ttests/golden/new.png\nM\ttests/golden/old.png\n";
        let changes = parse_name_status(out);
        assert_eq!(
            changes,
            vec![
                change('A', "tests/golden/new.png"),
                change('M', "tests/golden/old.png")
            ]
        );
    }

    #[test]
    fn parses_renames_keeping_final_path() {
        let out = "R100\ttests/golden/old.png\ttests/golden/new.png\n";
        let changes = parse_name_status(out);
        assert_eq!(changes, vec![change('R', "tests/golden/new.png")]);
    }
}
