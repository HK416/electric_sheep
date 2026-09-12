//! `cargo xtask check-scope <packet.md>` — the working-tree diff must stay
//! inside the globs declared under the packet's `## context` section (spec 1.2,
//! defends against quiet scope creep, spec 1.7).

use regex::Regex;
use std::path::Path;
use std::process::Command;

/// Parse the `## context` section of a packet file into a list of glob
/// patterns. Globs may be listed as a fenced code block or a `-`/`*` bullet
/// list; inline backticks are stripped.
pub fn parse_context_globs(md: &str) -> Vec<String> {
    let mut globs = Vec::new();
    let mut in_section = false;
    let mut in_fence = false;
    for line in md.lines() {
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("## context") {
            in_section = true;
            continue;
        }
        if in_section && trimmed.starts_with("##") {
            break;
        }
        if !in_section {
            continue;
        }
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        let candidate = if in_fence {
            trimmed
        } else if let Some(rest) = trimmed
            .strip_prefix('-')
            .or_else(|| trimmed.strip_prefix('*'))
        {
            rest.trim()
        } else {
            ""
        };
        let candidate = candidate.trim_matches('`').trim();
        if !candidate.is_empty() {
            globs.push(candidate.to_string());
        }
    }
    globs
}

/// Translate a `.gitignore`-style glob (`*`, `**`, `?`, literals) into an
/// anchored regex. Deliberately simple: no `{a,b}` brace or `[abc]` class
/// support, none of the packet globs in this repo need it.
fn glob_to_regex(glob: &str) -> Regex {
    let mut re = String::from("^");
    let chars: Vec<char> = glob.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' if chars.get(i + 1) == Some(&'*') => {
                re.push_str(".*");
                i += 2;
            }
            '*' => {
                re.push_str("[^/]*");
                i += 1;
            }
            '?' => {
                re.push_str("[^/]");
                i += 1;
            }
            c => {
                if "\\.+^$()[]{}|".contains(c) {
                    re.push('\\');
                }
                re.push(c);
                i += 1;
            }
        }
    }
    re.push('$');
    Regex::new(&re).expect("glob_to_regex always builds a valid pattern")
}

pub fn path_matches_any(path: &str, globs: &[String]) -> bool {
    globs.iter().any(|g| glob_to_regex(g).is_match(path))
}

fn git_lines(workspace_root: &Path, args: &[&str]) -> Vec<String> {
    let Ok(out) = Command::new("git")
        .args(args)
        .current_dir(workspace_root)
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

fn changed_files(workspace_root: &Path) -> Vec<String> {
    let mut files = git_lines(workspace_root, &["diff", "--name-only", "HEAD"]);
    for line in git_lines(
        workspace_root,
        &["status", "--porcelain", "--untracked-files=all"],
    ) {
        if let Some(path) = line.strip_prefix("?? ") {
            files.push(path.trim_matches('"').to_string());
        }
    }
    files.sort();
    files.dedup();
    files
}

pub fn run(workspace_root: &Path, packet_path: &Path) -> bool {
    let md = match std::fs::read_to_string(packet_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot read {}: {e}", packet_path.display());
            return false;
        }
    };
    let globs = parse_context_globs(&md);
    if globs.is_empty() {
        eprintln!(
            "{}: no globs found under `## context`",
            packet_path.display()
        );
        return false;
    }

    let files = changed_files(workspace_root);
    let out_of_scope: Vec<&String> = files
        .iter()
        .filter(|f| !path_matches_any(f, &globs))
        .collect();

    if out_of_scope.is_empty() {
        println!(
            "check-scope: {} changed file(s), all within {} glob(s)",
            files.len(),
            globs.len()
        );
        true
    } else {
        for f in &out_of_scope {
            println!("out of scope: {f}");
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_context_block() {
        let md = "## context\n```\nxtask/**\ndocs/packets/M0/P01.md\n```\n## spec\nignored/**\n";
        assert_eq!(
            parse_context_globs(md),
            vec!["xtask/**".to_string(), "docs/packets/M0/P01.md".to_string()]
        );
    }

    #[test]
    fn parses_bullet_context_list() {
        let md = "## context\n- `xtask/**`\n- .gitattributes\n\n## oracle\nfoo\n";
        assert_eq!(
            parse_context_globs(md),
            vec!["xtask/**".to_string(), ".gitattributes".to_string()]
        );
    }

    #[test]
    fn glob_star_star_matches_nested_paths() {
        assert!(path_matches_any("xtask/src/main.rs", &["xtask/**".into()]));
        assert!(path_matches_any("xtask/Cargo.toml", &["xtask/**".into()]));
        assert!(!path_matches_any(
            "crates/es-core/src/lib.rs",
            &["xtask/**".into()]
        ));
    }

    #[test]
    fn glob_literal_file_matches_exactly() {
        let globs = vec![".gitattributes".to_string()];
        assert!(path_matches_any(".gitattributes", &globs));
        assert!(!path_matches_any("a/.gitattributes", &globs));
    }

    #[test]
    fn glob_single_star_does_not_cross_directories() {
        let globs = vec!["docs/packets/M0/*.md".to_string()];
        assert!(path_matches_any("docs/packets/M0/P01.md", &globs));
        assert!(!path_matches_any("docs/packets/M0/sub/P01.md", &globs));
    }
}
