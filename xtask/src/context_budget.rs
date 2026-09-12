//! `cargo xtask context-budget` — spec 1.5: core crates target <= 6,000 / hard cap
//! <= 10,000 source lines, excluding `tests/`/`benches/` directories, inline
//! `#[cfg(test)]` modules, blank lines, and comment-only lines.

use std::fs;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

const WARN_AT: usize = 6_000;
const FAIL_AT: usize = 10_000;

static TEST_CFG_ATTR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^#!?\[cfg\(.*\btest\b.*\)\]$").unwrap());
static MOD_OPEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(pub(\([^)]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\s*\{").unwrap()
});

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

/// Net `{`/`}` on a line, ignoring braces inside `"..."` string literals and
/// after a `//` line comment marker. Not a full lexer (raw strings, block
/// comments, and char-literal braces aren't special-cased) but good enough to
/// find the end of an inline test module.
fn brace_delta(line: &str) -> i32 {
    let mut delta = 0i32;
    let mut in_string = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_string {
            if c == '\\' {
                chars.next();
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '/' if chars.peek() == Some(&'/') => break,
            '"' => in_string = true,
            '{' => delta += 1,
            '}' => delta -= 1,
            _ => {}
        }
    }
    delta
}

fn is_comment_only(trimmed: &str) -> bool {
    trimmed.starts_with("//")
}

/// Count `(code, total)` lines of a single file's source text: `total` is
/// every line outside an excluded inline test module; `code` further drops
/// blank lines and comment-only lines. The context budget (§1.5) is judged on
/// `code`.
fn count_file(text: &str) -> (usize, usize) {
    let lines: Vec<&str> = text.lines().collect();
    let mut code = 0;
    let mut total = 0;
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if TEST_CFG_ATTR.is_match(trimmed) {
            // Skip past any stacked attributes/blank lines to find the `mod` this
            // attribute applies to.
            let mut j = i + 1;
            while j < lines.len() {
                let t = lines[j].trim();
                if t.is_empty() || t.starts_with("#[") || t.starts_with("#![") {
                    j += 1;
                } else {
                    break;
                }
            }
            if j < lines.len() && MOD_OPEN.is_match(lines[j].trim()) {
                let mut depth = 0i32;
                let mut seen_open = false;
                let mut k = j;
                while k < lines.len() {
                    let d = brace_delta(lines[k]);
                    depth += d;
                    seen_open |= depth > 0;
                    k += 1;
                    if seen_open && depth <= 0 {
                        break;
                    }
                }
                i = k; // drop lines[i..k] entirely: not test, not code.
                continue;
            }
        }
        total += 1;
        if !trimmed.is_empty() && !is_comment_only(trimmed) {
            code += 1;
        }
        i += 1;
    }
    (code, total)
}

/// Recursively sum `(code, total)` lines of `*.rs` files under `src/`, skipping
/// any path that has a `tests` or `benches` path component (defense in depth:
/// crates are only expected to keep non-test sources under `src/`).
fn count_dir_lines(dir: &Path) -> (usize, usize) {
    let mut code = 0;
    let mut total = 0;
    let Ok(entries) = fs::read_dir(dir) else {
        return (0, 0);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "tests" || name == "benches" {
                continue;
            }
            let (c, t) = count_dir_lines(&path);
            code += c;
            total += t;
        } else if path.extension().is_some_and(|e| e == "rs") {
            if let Ok(s) = fs::read_to_string(&path) {
                let (c, t) = count_file(&s);
                code += c;
                total += t;
            }
        }
    }
    (code, total)
}

pub fn run(crates_dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(crates_dir) else {
        println!("no crates/ directory found; nothing to budget");
        return true;
    };
    let mut rows: Vec<(String, usize, usize, Status)> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let (code, total) = count_dir_lines(&e.path().join("src"));
            (name, code, total, Status::from_lines(code))
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    println!("{:<24} {:>8} {:>8}  status", "crate", "code", "total");
    for (name, code, total, status) in &rows {
        println!("{name:<24} {code:>8} {total:>8}  {}", status.label());
    }

    rows.iter().all(|(_, _, _, status)| *status != Status::Fail)
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

    #[test]
    fn excludes_inline_cfg_test_mod() {
        let src = r"
fn real_code() {
    1 + 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let map = { let mut m = 0; m += 1; m };
        assert_eq!(map, 1);
    }
}

fn more_code() {
    2 + 2;
}
";
        let (code, _total) = count_file(src);
        // Only `real_code`'s brace line, body, close, blank, and `more_code`'s
        // three lines should remain — none of the `mod tests { ... }` body.
        let src_no_tests = "\nfn real_code() {\n    1 + 1;\n}\n\nfn more_code() {\n    2 + 2;\n}\n";
        let (expected_code, _) = count_file(src_no_tests);
        assert_eq!(code, expected_code);
    }

    #[test]
    fn excludes_cfg_any_test_feature_mod() {
        let src = r#"
pub fn lib_fn() {}

#[cfg(any(test, feature = "testing"))]
pub mod testing {
    pub struct Fixture {
        pub value: i32,
    }

    impl Fixture {
        pub fn new() -> Self {
            Self { value: 0 }
        }
    }
}
"#;
        let (code, total) = count_file(src);
        // Everything from the `#[cfg(any(...))]` line through the closing `}`
        // of `mod testing` is dropped; the two surrounding blank lines still
        // count toward `total` (blank lines are only excluded from `code`).
        assert_eq!(code, 1); // `pub fn lib_fn() {}`
        assert_eq!(total, 3);
    }

    #[test]
    fn braces_in_strings_and_comments_do_not_confuse_brace_matching() {
        let src = r#"
#[cfg(test)]
mod tests {
    // a lone brace in a comment: {
    const S: &str = "also a brace: {";

    #[test]
    fn it_works() {
        assert!(true);
    }
}

fn after() {}
"#;
        let (code, total) = count_file(src);
        assert_eq!(code, 1); // only `fn after() {}`
        assert_eq!(total, 3); // the two surrounding blank lines still count
    }

    #[test]
    fn blank_and_comment_only_lines_excluded_from_code_not_total() {
        let src = "fn a() {}\n\n// a comment\n/// doc\n//! module doc\nfn b() {}\n";
        let (code, total) = count_file(src);
        assert_eq!(code, 2);
        assert_eq!(total, 6);
    }
}
