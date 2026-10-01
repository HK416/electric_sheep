//! Every word the editor shows, in the reader's language (packet M7/E6, spec 23.2).
//!
//! **No visible string is written in `app.rs`.** The two tables `i18n/en.toml` and
//! `i18n/ko.toml` are the only place text lives, which is the same rule spec 28.10 rule 3
//! applies to every other decision: a sentence compiled into the shell is a sentence nothing
//! can check and nobody can translate. It also buys the one exception the repository's
//! English-only rule has — Korean belongs in `*/i18n/*.toml` and nowhere else, so
//! `.githooks/pre-commit` allows those two files and no source file needs to.
//!
//! [`Lang`] is persisted beside the recent list ([`crate::model::recent::LANG_KEY`]) and
//! toggled in the top bar. `En` is the default, because a machine nobody has configured
//! should not guess.
//!
//! A key missing at runtime renders as the key itself rather than panicking — an editor that
//! will not start is worse than one with a visible `tab.design` in it — and
//! `i18n_tables_are_complete_and_used` makes that unreachable in a build that passes CI.

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Which table the editor is reading from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Lang {
    #[default]
    En,
    Ko,
}

impl Lang {
    pub const ALL: [Lang; 2] = [Lang::En, Lang::Ko];

    /// What is persisted. Two letters, so a hand-edited store is readable.
    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ko => "ko",
        }
    }

    /// Anything unrecognised is [`Lang::En`]: a store written by another version must not stop
    /// the editor from opening.
    pub fn from_code(code: &str) -> Self {
        match code.trim() {
            "ko" => Lang::Ko,
            _ => Lang::En,
        }
    }

    /// The toggle's own label, which is **not** translated: a person looking for their
    /// language looks for its name in that language.
    pub fn label_key(self) -> &'static str {
        match self {
            Lang::En => "lang.en",
            Lang::Ko => "lang.ko",
        }
    }

    fn table(self) -> &'static str {
        match self {
            Lang::En => include_str!("../../i18n/en.toml"),
            Lang::Ko => include_str!("../../i18n/ko.toml"),
        }
    }
}

/// One language's table.
#[derive(Debug)]
pub struct Strings {
    entries: BTreeMap<String, String>,
}

impl Strings {
    /// The table, parsed once. The tables are `include_str!`d, so this cannot fail for a
    /// reason the machine could fix; a malformed table is a build that should never have
    /// shipped, and the empty map that results makes every key render as itself.
    pub fn get(lang: Lang) -> &'static Strings {
        static TABLES: OnceLock<[Strings; 2]> = OnceLock::new();
        let tables = TABLES.get_or_init(|| {
            Lang::ALL.map(|lang| Strings {
                entries: toml::from_str(lang.table()).unwrap_or_default(),
            })
        });
        match lang {
            Lang::En => &tables[0],
            Lang::Ko => &tables[1],
        }
    }

    /// The word `key` names, or `key` itself when the tables have no such entry: the two
    /// lifetimes are one because the fallback *is* the key.
    pub fn t<'a>(&'a self, key: &'a str) -> &'a str {
        self.entries.get(key).map_or(key, String::as_str)
    }

    /// The template with each `{}` replaced by the next argument, in order. Deliberately the
    /// whole of the formatting this crate needs: a template holding a number is a sentence
    /// whose word order differs between languages, and nothing here needs more than that.
    pub fn fill(&self, key: &str, args: &[&str]) -> String {
        let mut out = self.t(key).to_owned();
        for arg in args {
            let Some(at) = out.find("{}") else { break };
            out.replace_range(at..at + 2, arg);
        }
        out
    }

    /// Every key, for the completeness test.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }
}

/// The word `key` names, in `lang`. Keys are literals, which is what makes the answer
/// `'static` and lets a label be stored as one.
pub fn t(lang: Lang, key: &'static str) -> &'static str {
    Strings::get(lang).t(key)
}

/// [`Strings::fill`] on `lang`'s table, its Korean particles picked from the words filled in
/// ([`particles`]).
pub fn fill(lang: Lang, key: &str, args: &[&str]) -> String {
    particles("", &Strings::get(lang).fill(key, args))
}

/// How a word ends when a Korean reader reads it aloud, which picks the particle after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    /// No final consonant (batchim).
    Vowel,
    /// A final consonant other than rieul.
    Consonant,
    /// A final rieul: a consonant, except that it takes ro rather than euro.
    Rieul,
}

/// The particle pairs the Korean table writes after a hole, in the textbook's placeholder form,
/// and the form each [`Ending`] takes, in its order. Romanized, since source files hold no
/// Hangul: i(ga), eun(neun), eul(reul), wa(gwa), (eu)ro.
const PAIRS: [(&str, [&str; 3]); 5] = [
    ("\u{c774}(\u{ac00})", ["\u{ac00}", "\u{c774}", "\u{c774}"]),
    ("\u{c740}(\u{b294})", ["\u{b294}", "\u{c740}", "\u{c740}"]),
    ("\u{c744}(\u{b97c})", ["\u{b97c}", "\u{c744}", "\u{c744}"]),
    ("\u{c640}(\u{acfc})", ["\u{c640}", "\u{acfc}", "\u{acfc}"]),
    (
        "(\u{c73c})\u{b85c}",
        ["\u{b85c}", "\u{c73c}\u{b85c}", "\u{b85c}"],
    ),
];

/// How `word` ends when read in Korean (packet M17/R4):
///
/// - a Hangul syllable by its final consonant, `(c - 0xAC00) % 28` (0 none, 8 rieul);
/// - a number's unit as the task's sentences write it (` cm`, `°`, ` s`, `/s`, `%`: senti-miteo,
///   do, cho, peosenteu): a vowel;
/// - a digit as Sino-Korean reads it: 2, 4, 5, 9 a vowel (i, sa, o, gu); 1, 7, 8 rieul (il,
///   chil, pal); 3, 6 and 0 a consonant (sam, yuk; 0 is yeong alone and sip, baek, cheon closing
///   a longer number);
/// - a Latin word by how Korean writes its end (the loanword orthography), a silent `e` after
///   `l`, `m` or `n` dropped: `l` rieul (roll, table); `m`, `n` or `ng` a consonant (arm, pan,
///   zone, ring); `p`, `t`, `k` or `ck` after a single vowel letter a consonant, a short vowel's
///   final stop being written as a final consonant (cup, target, pick); anything else a vowel
///   (object, cube, box, gripper, x);
/// - anything else, the empty word included: a vowel.
pub fn ending(word: &str) -> Ending {
    if [" cm", "\u{b0}", " s", "/s", "%"]
        .iter()
        .any(|u| word.ends_with(u))
    {
        return Ending::Vowel;
    }
    match word.chars().last() {
        Some(c @ '\u{ac00}'..='\u{d7a3}') => match (u32::from(c) - 0xac00) % 28 {
            0 => Ending::Vowel,
            8 => Ending::Rieul,
            _ => Ending::Consonant,
        },
        Some('1' | '7' | '8') => Ending::Rieul,
        Some('0' | '3' | '6') => Ending::Consonant,
        Some(c) if c.is_ascii_alphabetic() => {
            let w = word.to_ascii_lowercase();
            let vowel = |c: Option<char>| c.is_some_and(|c| "aeiou".contains(c));
            let short_stop = ["p", "t", "k", "ck"].iter().any(|stop| {
                w.strip_suffix(stop).is_some_and(|rest| {
                    let mut r = rest.chars().rev();
                    vowel(r.next()) && !vowel(r.next())
                })
            });
            let spoken = (w.strip_suffix('e')).filter(|r| r.ends_with(['l', 'm', 'n']));
            let spoken = spoken.unwrap_or(&w);
            if spoken.ends_with('l') {
                Ending::Rieul
            } else if spoken.ends_with(['m', 'n']) || spoken.ends_with("ng") || short_stop {
                Ending::Consonant
            } else {
                Ending::Vowel
            }
        }
        _ => Ending::Vowel,
    }
}

/// `text` with each particle pair of the Korean table resolved by the word before it
/// ([`ending`]): the text before the pair, or `before` for a pair that opens `text` (a sentence's
/// words after a slot's widget). A text without pairs, English, comes back as it was.
pub fn particles(before: &str, text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if let Some((pair, forms)) = PAIRS.iter().find(|(p, _)| rest.starts_with(p)) {
            let end = ending(if out.is_empty() { before } else { &out });
            out.push_str(forms[end as usize]);
            rest = &rest[pair.len()..];
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{fill, t, Lang, Strings};

    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    /// Every `.rs` file of this crate, source and tests alike, of `es-editor`, the shell that
    /// shows the same tables (packet M12/R4 split the two), of `es-editor-scene`, and of
    /// `es-editor-graph` (packet M17/R1 moved part of this crate there).
    fn sources() -> Vec<String> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut paths = Vec::new();
        walk(&root.join("src"), &mut paths);
        walk(&root.join("tests"), &mut paths);
        let own = paths.len();
        walk(&root.join("../es-editor/src"), &mut paths);
        // The scene model (packet M17/G5) names its refusals by these tables' keys.
        walk(&root.join("../es-editor-scene/src"), &mut paths);
        walk(&root.join("../es-editor-graph/src"), &mut paths);
        walk(&root.join("../es-editor-graph/tests"), &mut paths);
        assert!(
            own > 0 && paths.len() > own,
            "this crate's and the shell's sources"
        );
        paths
            .iter()
            .map(|p| std::fs::read_to_string(p).expect("a source file"))
            .collect()
    }

    /// Every double-quoted literal in `src`, naively: enough to find the keys, since a key
    /// holds neither an escape nor a newline.
    fn literals(src: &str) -> impl Iterator<Item = &str> {
        src.split('"').skip(1).step_by(2)
    }

    /// The suffix that marks a value as hover text.
    const HINT: &str = ".hint";

    /// Oracle 1 (packet M7/E6). The two tables carry the same keys, every key is used by the
    /// crate, every key-shaped literal in the crate is a key, and only a hover cites the spec.
    #[test]
    fn i18n_tables_are_complete_and_used() {
        let en: BTreeSet<&str> = Strings::get(Lang::En).keys().collect();
        let ko: BTreeSet<&str> = Strings::get(Lang::Ko).keys().collect();
        assert!(!en.is_empty(), "the English table parsed");
        let missing_ko: Vec<_> = en.difference(&ko).collect();
        let missing_en: Vec<_> = ko.difference(&en).collect();
        assert!(
            missing_ko.is_empty(),
            "missing from ko.toml: {missing_ko:?}"
        );
        assert!(
            missing_en.is_empty(),
            "missing from en.toml: {missing_en:?}"
        );

        let sources = sources();
        let used: BTreeSet<&str> = sources.iter().flat_map(|s| literals(s)).collect();
        let unused: Vec<_> = en.iter().filter(|k| !used.contains(*k)).collect();
        assert!(
            unused.is_empty(),
            "in the tables but never shown: {unused:?}"
        );

        // The other direction: a literal that is shaped like a key and begins with a prefix
        // the tables use is meant to be a key. A typo therefore fails twice - here, and as the
        // real key going unused above.
        let prefixes: BTreeSet<&str> = en.iter().filter_map(|k| k.split('.').next()).collect();
        let strays: Vec<&str> = used
            .iter()
            .copied()
            .filter(|lit| {
                let mut segments = lit.split('.');
                !en.contains(lit)
                    && segments.next().is_some_and(|p| prefixes.contains(p))
                    // Two or more non-empty segments: `metric.` is a prefix someone strips
                    // off a telemetry field name, not a key that was mistyped.
                    && segments.clone().count() >= 1
                    && lit.split('.').all(|seg| !seg.is_empty())
                    && lit
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._".contains(c))
            })
            .collect();
        assert!(
            strays.is_empty(),
            "key-shaped but not in the tables: {strays:?}"
        );

        // No visible string cites a spec section; a hover is the one place that may.
        for lang in Lang::ALL {
            let table = Strings::get(lang);
            // A hover is the one value that may cite the spec; `.hint` is a key suffix, not
            // a file extension, so the path-aware comparison clippy suggests is wrong here.
            for key in en.iter().filter(|k| !k.contains(HINT)) {
                let value = table.t(key);
                assert!(
                    !value.contains("spec "),
                    "{}/{key} cites the spec in what it shows: {value}",
                    lang.code()
                );
            }
        }
    }

    /// The lookup itself: a key resolves per language, an unknown key is visible rather than
    /// fatal, and `fill` substitutes in order and stops when it runs out.
    #[test]
    fn a_missing_key_is_visible_and_fill_substitutes_in_order() {
        const HOME: &str = concat!("home", ".", "recent");
        const ABSENT: &str = concat!("no", ".", "such", ".", "key");
        assert_ne!(t(Lang::En, HOME), t(Lang::Ko, HOME), "translated");
        assert_eq!(
            t(Lang::En, ABSENT),
            ABSENT,
            "an unknown key shows as itself"
        );
        assert_eq!(Lang::from_code(Lang::Ko.code()), Lang::Ko);
        assert_eq!(Lang::from_code("de"), Lang::En, "anything else is English");

        let filled = fill(Lang::En, concat!("status", ".", "messages"), &["7"]);
        assert!(filled.contains('7') && !filled.contains("{}"), "{filled}");
        // More holes than arguments leaves the rest alone rather than panicking.
        let short = fill(Lang::En, concat!("status", ".", "hash"), &["Task"]);
        assert!(short.contains("Task") && short.contains("{}"), "{short}");
    }

    /// Oracle 1 (packet M17/R4): how a word ends, by script. Hangul is written as escapes,
    /// romanized beside them, since source files hold none.
    #[test]
    fn a_words_ending_is_read_from_its_last_sound() {
        use super::{ending, Ending::*};
        for (word, want) in [
            ("\u{ac00}", Vowel),         // ga
            ("\u{c0c1}\u{c790}", Vowel), // sangja
            ("\u{ac01}", Consonant),     // gak
            ("\u{ac15}", Consonant),     // gang
            ("\u{ac08}", Rieul),         // gal
            ("\u{bb3c}", Rieul),         // mul
            ("object", Vowel),           // obeujekteu
            ("cube", Vowel),             // kyubeu
            ("box", Vowel),              // bakseu
            ("gripper", Vowel),          // geurippeo
            ("cube.x", Vowel),           // ekseu
            ("shoulder_lift", Vowel),    // lipeuteu
            ("seat", Vowel),             // siteu: a long vowel's stop is no final consonant
            ("target", Consonant),       // tagit
            ("robot", Consonant),        // robot
            ("CUP", Consonant),          // keop
            ("pick", Consonant),         // pik
            ("shoulder_pan", Consonant), // paen
            ("arm", Consonant),          // am
            ("ring", Consonant),         // ring
            ("zone", Consonant),         // jon: the e is silent
            ("wrist_roll", Rieul),       // rol
            ("table", Rieul),            // teibeul
            ("l", Rieul),                // el
            ("2", Vowel),                // i
            ("9", Vowel),                // gu
            ("cube2", Vowel),            // i
            ("1", Rieul),                // il
            ("86.7", Rieul),             // chil
            ("8", Rieul),                // pal
            ("3", Consonant),            // sam
            ("6", Consonant),            // yuk
            ("0", Consonant),            // yeong
            ("10", Consonant),           // sip
            ("9 cm", Vowel),             // senti-miteo
            ("(100, 86.7, 17.72) cm", Vowel),
            ("5.73\u{b0}", Vowel), // do
            ("5 cm/s", Vowel),
            ("10\u{b0}/s", Vowel),
            ("8 s", Vowel),  // cho
            ("20 %", Vowel), // peosenteu
            ("", Vowel),
            (")", Vowel),
        ] {
            assert_eq!(ending(word), want, "{word:?}");
        }
    }

    /// Oracle 1 (packet M17/R4): each pair takes the form its word asks for, (eu)ro takes ro
    /// after rieul, a text without pairs is unchanged, and `fill` picks them.
    #[test]
    fn particles_follow_the_word_before_them() {
        use super::particles;
        const I_GA: &str = "\u{c774}(\u{ac00})";
        const WA_GWA: &str = "\u{c640}(\u{acfc})";
        const EURO: &str = "(\u{c73c})\u{b85c}";
        // object-ga target-gwa
        let text = format!("object{I_GA} target{WA_GWA} ");
        assert_eq!(particles("", &text), "object\u{ac00} target\u{acfc} ");
        // A pair that opens the text follows `before`: a slot's widget.
        assert_eq!(particles("target", &format!("{I_GA} ")), "\u{c774} ");
        assert_eq!(particles("", I_GA), "\u{ac00}", "no word: a vowel");
        // mul-lo, gak-euro, cube-ro
        assert_eq!(particles("\u{bb3c}", EURO), "\u{b85c}");
        assert_eq!(particles("\u{ac01}", EURO), "\u{c73c}\u{b85c}");
        assert_eq!(particles("cube", EURO), "\u{b85c}");
        // eun(neun), eul(reul)
        let eun = particles("pan", "\u{c740}(\u{b294})");
        let reul = particles("box", "\u{c744}(\u{b97c})");
        assert_eq!([eun.as_str(), reul.as_str()], ["\u{c740}", "\u{b97c}"]);
        assert_eq!(particles("x", "is inside (2)"), "is inside (2)");
        let missing = fill(
            Lang::Ko,
            concat!("author", ".", "refused", ".", "missing"),
            &["box"],
        );
        assert!(missing.starts_with("box\u{ac00} "), "{missing}");
    }
}
