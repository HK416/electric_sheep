//! `cargo xtask check-spec-refs` — every `spec N.M` / `§N.M` reference in code
//! and docs must resolve to a real heading in `docs/ARCHITECTURE.ko.md`.

use regex::Regex;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

static REF_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:\u{00A7}\s?|spec\s+)(\d+(?:\.\d+)?)").unwrap());
static HEADING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^#{2,3}\s+(\d+(?:\.\d+)?)\b").unwrap());

/// Extract every spec section number referenced in `text` (e.g. "spec 7.2",
/// "§7.2", "(spec 3.4)" all yield "7.2").
pub fn extract_refs(text: &str) -> Vec<String> {
    REF_RE
        .captures_iter(text)
        .map(|c| c[1].to_string())
        .collect()
}

/// Extract the set of section numbers that exist as `## N.` / `### N.M`
/// headings in the canonical spec.
pub fn valid_headings(spec_text: &str) -> BTreeSet<String> {
    HEADING_RE
        .captures_iter(spec_text)
        .map(|c| c[1].to_string())
        .collect()
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs" || e == "md") {
            // The canonical spec and its translation (docs/ARCHITECTURE*.md) are
            // the reference, not something that references itself.
            let name = path.file_name().unwrap().to_string_lossy();
            if name.starts_with("ARCHITECTURE") {
                continue;
            }
            out.push(path);
        }
    }
}

pub fn run(workspace_root: &Path) -> bool {
    let spec_path = workspace_root.join("docs/ARCHITECTURE.ko.md");
    let spec_text = match fs::read_to_string(&spec_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {}: {e}", spec_path.display());
            return false;
        }
    };
    let valid = valid_headings(&spec_text);

    let mut files = Vec::new();
    for sub in ["crates", "xtask", "docs"] {
        collect_files(&workspace_root.join(sub), &mut files);
    }

    let mut unresolved = Vec::new();
    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            for section in extract_refs(line) {
                if !valid.contains(&section) {
                    unresolved.push(format!(
                        "{}:{}: unresolved spec reference `{}`",
                        file.display(),
                        i + 1,
                        section
                    ));
                }
            }
        }
    }

    if unresolved.is_empty() {
        println!(
            "check-spec-refs: {} files scanned, all refs resolve",
            files.len()
        );
        true
    } else {
        for u in &unresolved {
            println!("{u}");
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_various_forms() {
        assert_eq!(extract_refs("see spec 7.2 for details"), vec!["7.2"]);
        assert_eq!(extract_refs("see \u{00A7}7.2 for details"), vec!["7.2"]);
        assert_eq!(extract_refs("(spec 3.4)"), vec!["3.4"]);
        assert_eq!(extract_refs("spec 1"), vec!["1"]);
        assert_eq!(extract_refs("no refs here"), Vec::<String>::new());
    }

    #[test]
    fn extracts_multiple_refs_per_line() {
        assert_eq!(extract_refs("spec 1.2 and \u{00A7}4.2"), vec!["1.2", "4.2"]);
    }

    #[test]
    fn heading_numbers_parsed() {
        let spec = "## 0. intro\n### 1.4 oracle\ntext\n### 1.10 more\n## 4. arch\n### 4.2 layers\n";
        let headings = valid_headings(spec);
        assert!(headings.contains("0"));
        assert!(headings.contains("1.4"));
        assert!(headings.contains("1.10"));
        assert!(headings.contains("4"));
        assert!(headings.contains("4.2"));
        assert!(!headings.contains("9.9"));
    }
}
