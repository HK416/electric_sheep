//! A font that renders Korean, and a text size a person can actually read (packet M7/E6).
//!
//! egui's bundled fonts carry no Hangul, Kana or Han glyph, so a Korean label - or a Korean
//! path typed into a field - draws as boxes. The fix is the **system's own** CJK font, found
//! by looking in the places every supported OS keeps one:
//!
//! * **nothing is bundled.** A CJK face is 10-20 MB and licensed per family; committing one
//!   would put a binary blob in a repository whose goldens are read-only for good reasons.
//! * **no font-discovery crate.** The question "does `C:\Windows\Fonts\malgun.ttf` exist" is
//!   [`std::fs`] and a fixed list. A dependency that enumerates every installed face solves a
//!   problem this editor does not have.
//! * **last fallback, never first.** [`install`] appends the face to the end of both
//!   `Proportional` and `Monospace`, so Latin keeps egui's own glyphs - which are better
//!   hinted at small sizes - and only the characters egui cannot draw fall through.
//!
//! When nothing is found the editor still opens and the status bar says which paths were
//! tried and which package installs one; boxes with an explanation beat boxes without.
//!
//! [`TextSize`] is the other half of the same complaint: 13 px is a choice made for people
//! with the eyes of the person who wrote the toolkit. The default is 15 px body / 20 px
//! heading, and S/M/L scales every style at once (persisted beside the recent list).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use egui::{FontData, FontDefinitions, FontFamily, FontId, TextStyle};

/// The name [`install`] registers the system face under.
pub const FAMILY: &str = "system-cjk";

/// Body text, in points, at [`TextSize::M`].
pub const BODY_PX: f32 = 15.0;
/// A heading, in points, at [`TextSize::M`].
pub const HEADING_PX: f32 = 20.0;

/// How deep [`find`] walks a `**` root. Distribution font trees are two or three levels
/// (`/usr/share/fonts/opentype/noto/…`); an unbounded walk of a directory someone has
/// symlinked is not worth a nicer font.
const SCAN_DEPTH: usize = 4;

/// Where Windows keeps a Korean, Chinese and Japanese face, in that order of preference:
/// Malgun Gothic is the Korean UI font every Windows 10/11 install has.
#[cfg(windows)]
const CANDIDATES: &[&str] = &[
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
];

/// macOS ships Apple SD Gothic Neo (Korean) and PingFang (Chinese) in the system font
/// directory; Noto CJK arrives with the optional font packages, under `Supplemental`.
#[cfg(target_os = "macos")]
const CANDIDATES: &[&str] = &[
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/NotoSansCJK*.ttc",
];

/// Linux ships nothing by default: these are what `fonts-noto-cjk`, `fonts-nanum` and the
/// Android fallback package put on disk, wherever the distribution files them.
#[cfg(not(any(windows, target_os = "macos")))]
const CANDIDATES: &[&str] = &[
    "/usr/share/fonts/**/NotoSansCJK*.ttc",
    "/usr/share/fonts/**/NotoSansCJK*.otf",
    "/usr/share/fonts/**/NanumGothic.ttf",
    "/usr/share/fonts/**/DroidSansFallback*.ttf",
];

/// The fixed per-OS list, exactly as it is searched. Reported verbatim when none of them is
/// there, because "no CJK font" is only actionable if it says where it looked.
pub fn candidates() -> &'static [&'static str] {
    CANDIDATES
}

/// The first candidate that exists: its file name and its bytes.
///
/// `None` is an ordinary answer on a machine with no CJK font installed, which is most fresh
/// Linux containers; [`install`] turns it into a sentence.
pub fn system_cjk_font() -> Option<(String, Vec<u8>)> {
    for pattern in CANDIDATES {
        let Some(path) = find(pattern) else { continue };
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if bytes.is_empty() {
            continue;
        }
        let name = path.file_name().map_or_else(
            || (*pattern).to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        return Some((name, bytes));
    }
    None
}

/// Appends the system CJK face to the end of `Proportional` and `Monospace`.
///
/// `Ok(name)` is the face that was installed; `Err(tried)` is [`candidates`] unchanged, for
/// the status bar to name. Either way `fonts` keeps egui's own faces at the front of both
/// families, so nothing about Latin text changes.
pub fn install(fonts: &mut FontDefinitions) -> Result<String, &'static [&'static str]> {
    let Some((name, bytes)) = system_cjk_font() else {
        return Err(CANDIDATES);
    };
    fonts.font_data.insert(
        FAMILY.to_owned(),
        std::sync::Arc::new(FontData::from_owned(bytes)),
    );
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push(FAMILY.to_owned());
    }
    Ok(name)
}

/// The one path a pattern names, if it is on disk.
///
/// Three shapes, which is every shape [`CANDIDATES`] uses: a plain path, a `dir/prefix*.ext`
/// listing, and a `root/**/prefix*.ext` walk. Entries are sorted, so a directory holding two
/// matches resolves to the same one every time.
fn find(pattern: &str) -> Option<PathBuf> {
    if let Some((root, tail)) = pattern.split_once("/**/") {
        return walk(Path::new(root), tail, SCAN_DEPTH);
    }
    let path = Path::new(pattern);
    if !pattern.contains('*') {
        return path.is_file().then(|| path.to_path_buf());
    }
    let name = path.file_name()?.to_str()?;
    walk(path.parent()?, name, 1)
}

/// The first file under `dir`, within `depth` levels, whose name matches `pattern`.
fn walk(dir: &Path, pattern: &str, depth: usize) -> Option<PathBuf> {
    if depth == 0 {
        return None;
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();
    for path in &entries {
        if path.is_file()
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| matches(n, pattern))
        {
            return Some(path.clone());
        }
    }
    entries
        .iter()
        .filter(|p| p.is_dir())
        .find_map(|p| walk(p, pattern, depth - 1))
}

/// `pattern` holds at most one `*`, which stands for any run of characters.
fn matches(name: &str, pattern: &str) -> bool {
    match pattern.split_once('*') {
        None => name == pattern,
        Some((prefix, suffix)) => {
            name.len() >= prefix.len() + suffix.len()
                && name.starts_with(prefix)
                && name.ends_with(suffix)
        }
    }
}

/// How big the text is. Three sizes rather than a slider: a person who cannot read the
/// default wants it bigger, not tuned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextSize {
    S,
    #[default]
    M,
    L,
}

impl TextSize {
    pub const ALL: [TextSize; 3] = [TextSize::S, TextSize::M, TextSize::L];

    /// What is persisted, and what an unrecognised store falls back to.
    pub fn code(self) -> &'static str {
        match self {
            TextSize::S => "s",
            TextSize::M => "m",
            TextSize::L => "l",
        }
    }

    pub fn from_code(code: &str) -> Self {
        match code.trim() {
            "s" => TextSize::S,
            "l" => TextSize::L,
            _ => TextSize::M,
        }
    }

    pub fn label_key(self) -> &'static str {
        match self {
            TextSize::S => "textsize.s",
            TextSize::M => "textsize.m",
            TextSize::L => "textsize.l",
        }
    }

    /// The multiplier every style is scaled by.
    pub fn scale(self) -> f32 {
        match self {
            TextSize::S => 0.85,
            TextSize::M => 1.0,
            TextSize::L => 1.3,
        }
    }
}

/// Every text style at `size`, for `egui::Style::text_styles`. The proportions are egui's;
/// what changes is that the base is [`BODY_PX`] rather than 12.5.
pub fn text_styles(size: TextSize) -> BTreeMap<TextStyle, FontId> {
    let s = size.scale();
    let p = FontFamily::Proportional;
    BTreeMap::from([
        (
            TextStyle::Small,
            FontId::new((BODY_PX - 3.0) * s, p.clone()),
        ),
        (TextStyle::Body, FontId::new(BODY_PX * s, p.clone())),
        (TextStyle::Button, FontId::new(BODY_PX * s, p.clone())),
        (TextStyle::Heading, FontId::new(HEADING_PX * s, p)),
        (
            TextStyle::Monospace,
            FontId::new((BODY_PX - 1.0) * s, FontFamily::Monospace),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::{
        candidates, find, install, matches, system_cjk_font, text_styles, TextSize, BODY_PX,
        FAMILY, HEADING_PX,
    };

    use egui::{FontDefinitions, FontFamily, TextStyle};

    /// Oracle 3 (packet M7/E6). On a machine that has one of the candidates the loader
    /// returns its name and its bytes; on one that has none, the reason names every path it
    /// looked at, so the message in the status bar is actionable rather than a shrug.
    #[test]
    fn system_cjk_font_is_found_or_the_reason_is_named() {
        assert!(!candidates().is_empty(), "every OS has a candidate list");
        let mut fonts = FontDefinitions::default();
        match install(&mut fonts) {
            Ok(name) => {
                assert!(!name.is_empty(), "the face is named");
                let (probed, bytes) = system_cjk_font().expect("the same probe finds it again");
                assert_eq!(probed, name, "the probe is deterministic");
                assert!(bytes.len() > 1024, "a real font file, not a stub");
                assert!(fonts.font_data.contains_key(FAMILY));
            }
            Err(tried) => {
                assert_eq!(tried, candidates(), "the reason lists every path tried");
                for pattern in tried {
                    assert!(find(pattern).is_none(), "{pattern} was reported missing");
                }
                assert!(!fonts.font_data.contains_key(FAMILY), "nothing installed");
            }
        }
    }

    /// egui's own faces stay at the front of both families: Latin text must not change, and
    /// only the glyphs egui cannot draw may fall through to the system face.
    #[test]
    fn font_fallback_is_last() {
        let base = FontDefinitions::default();
        let mut fonts = base.clone();
        let installed = install(&mut fonts).is_ok();
        for family in [FontFamily::Proportional, FontFamily::Monospace] {
            let before = base.families.get(&family).cloned().unwrap_or_default();
            let after = fonts.families.get(&family).cloned().unwrap_or_default();
            assert!(
                after.starts_with(&before),
                "egui's {family:?} faces stay first and in order"
            );
            if installed {
                assert_eq!(after.len(), before.len() + 1, "exactly one face was added");
                assert_eq!(
                    after.last().map(String::as_str),
                    Some(FAMILY),
                    "and it is last"
                );
            } else {
                assert_eq!(after, before, "nothing found, nothing changed");
            }
        }
    }

    /// The pattern matcher, which is the whole of the "no font-discovery crate" claim.
    #[test]
    fn a_pattern_matches_a_prefix_and_a_suffix() {
        assert!(matches("NotoSansCJKkr-Regular.otf", "NotoSansCJK*.otf"));
        assert!(!matches("NotoSansCJKkr-Regular.ttc", "NotoSansCJK*.otf"));
        assert!(!matches("NotoSans-Regular.otf", "NotoSansCJK*.otf"));
        assert!(matches("malgun.ttf", "malgun.ttf"));
        assert!(!matches("malgunbd.ttf", "malgun.ttf"));
        // The star may stand for nothing at all.
        assert!(matches("NotoSansCJK.otf", "NotoSansCJK*.otf"));
        assert!(find("no/such/directory/**/nothing*.ttf").is_none());
    }

    /// S/M/L moves every style at once, and M is the 15 px body / 20 px heading default.
    #[test]
    fn text_size_scales_every_style() {
        let m = text_styles(TextSize::M);
        // The default size is the default: exact, because M's scale is exactly 1.
        assert!((m[&TextStyle::Body].size - BODY_PX).abs() < f32::EPSILON);
        assert!((m[&TextStyle::Heading].size - HEADING_PX).abs() < f32::EPSILON);
        let mut previous = 0.0;
        for size in TextSize::ALL {
            let styles = text_styles(size);
            assert_eq!(styles.len(), m.len(), "the same styles at every size");
            let body = styles[&TextStyle::Body].size;
            assert!(body > previous, "{size:?} is larger than the one before");
            previous = body;
            for style in m.keys() {
                assert!(styles[style].size > 0.0, "{style:?} at {size:?}");
            }
            assert_eq!(TextSize::from_code(size.code()), size, "round-trips");
        }
        assert_eq!(
            TextSize::from_code("huge"),
            TextSize::M,
            "anything else is M"
        );
    }
}
