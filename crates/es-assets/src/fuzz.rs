//! Importer fuzzing harness (P35): mutation-based, shared by every importer.
//!
//! [`mutate`] applies one small, seeded edit to a well-formed seed document — the edits are
//! deliberately dumb (delete/duplicate an element, corrupt an attribute, truncate, inject
//! garbage) because the point is hostile input, not valid-but-unusual input. [`fuzz_importer`]
//! runs an importer over many such mutations and asserts it never panics: a `Result::Err` is a
//! pass, a panic is a bug in the importer (spec 25.1, asset parser fuzzing).
//!
//! Deterministic by construction — a hand-rolled xorshift64 keyed on `(seed_doc, case)`, no
//! external randomness — so a CI failure reproduces from the printed seed alone (spec 1.7,
//! "reimplementation" aside: this one deliberately skips `proptest`'s own PRNG for that reproducibility,
//! not out of NIH).

use std::panic::{self, AssertUnwindSafe};
use std::time::{Duration, Instant};

/// Per-case time budget: importers here parse short in-memory strings, so anything near this
/// means a hang, not slow I/O.
const CASE_BUDGET: Duration = Duration::from_millis(500);

/// Runs `f` over `cases` mutations of each of `seed_docs`, asserting it never panics and never
/// exceeds [`CASE_BUDGET`]. An importer `Err` is success: the fuzzer's job is to find crashes,
/// not to find inputs the importer correctly rejects.
pub fn fuzz_importer<T, E>(seed_docs: &[&str], f: impl Fn(&str) -> Result<T, E>, cases: u32) {
    for (doc_idx, seed) in seed_docs.iter().enumerate() {
        for case in 0..cases {
            let mutation_seed = (doc_idx as u64) << 32 | u64::from(case);
            let mutated = mutate(seed, mutation_seed);
            let started = Instant::now();
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| f(&mutated)));
            assert!(
                outcome.is_ok(),
                "importer panicked on seed doc {doc_idx} case {case} (mutation_seed \
                 {mutation_seed}):\n{mutated}"
            );
            let elapsed = started.elapsed();
            assert!(
                elapsed < CASE_BUDGET,
                "importer took {elapsed:?} (budget {CASE_BUDGET:?}) on seed doc {doc_idx} case \
                 {case}"
            );
        }
    }
}

/// Applies one seeded mutation to `xml`. Deterministic: the same `(xml, seed)` always produces
/// the same output.
pub fn mutate(xml: &str, seed: u64) -> String {
    let mut rng = Rng::new(seed);
    // A chosen strategy can find nothing to act on (e.g. no numeric attribute); retry a few
    // times with freshly-drawn randomness before giving up and returning the seed unchanged.
    for _ in 0..8 {
        let mutated = match rng.below(7) {
            0 => delete_element(xml, &mut rng),
            1 => duplicate_element(xml, &mut rng),
            2 => flip_numeric_attr(xml, &mut rng),
            3 => drop_attr(xml, &mut rng),
            4 => swap_attr_values(xml, &mut rng),
            5 => truncate_doc(xml, &mut rng),
            _ => inject_unknown_element(xml, &mut rng),
        };
        if let Some(out) = mutated {
            return out;
        }
    }
    xml.to_owned()
}

// ---- strategies -------------------------------------------------------------------------

fn delete_element(xml: &str, rng: &mut Rng) -> Option<String> {
    let (s, e) = *rng.choose(&element_spans(xml))?;
    Some(format!("{}{}", &xml[..s], &xml[e..]))
}

fn duplicate_element(xml: &str, rng: &mut Rng) -> Option<String> {
    let (s, e) = *rng.choose(&element_spans(xml))?;
    Some(format!("{}{}{}", &xml[..e], &xml[s..e], &xml[e..]))
}

fn inject_unknown_element(xml: &str, rng: &mut Rng) -> Option<String> {
    let (s, _) = *rng.choose(&element_spans(xml))?;
    Some(format!(
        "{}<es_fuzz_unknown attr=\"1\"/>{}",
        &xml[..s],
        &xml[s..]
    ))
}

fn truncate_doc(xml: &str, rng: &mut Rng) -> Option<String> {
    if xml.is_empty() {
        return None;
    }
    let mut cut = rng.below(xml.len());
    while cut > 0 && !xml.is_char_boundary(cut) {
        cut -= 1;
    }
    Some(xml[..cut].to_owned())
}

const NUMERIC_GARBAGE: [&str; 6] = ["NaN", "inf", "-inf", "1e308", "-1e308", ""];

fn flip_numeric_attr(xml: &str, rng: &mut Rng) -> Option<String> {
    let attrs = find_attrs(xml);
    let numeric: Vec<&Attr> = attrs
        .iter()
        .filter(|a| xml[a.value.0..a.value.1].trim().parse::<f64>().is_ok())
        .collect();
    let a = *rng.choose(&numeric)?;
    let replacement = *rng.choose(&NUMERIC_GARBAGE)?;
    Some(format!(
        "{}{replacement}{}",
        &xml[..a.value.0],
        &xml[a.value.1..]
    ))
}

fn drop_attr(xml: &str, rng: &mut Rng) -> Option<String> {
    let attrs = find_attrs(xml);
    let a = rng.choose(&attrs)?;
    Some(format!("{}{}", &xml[..a.full.0], &xml[a.full.1..]))
}

fn swap_attr_values(xml: &str, rng: &mut Rng) -> Option<String> {
    let attrs = find_attrs(xml);
    if attrs.len() < 2 {
        return None;
    }
    let i = rng.below(attrs.len());
    let j = (i + 1 + rng.below(attrs.len() - 1)) % attrs.len();
    let (first, second) = if attrs[i].value.0 < attrs[j].value.0 {
        (&attrs[i], &attrs[j])
    } else {
        (&attrs[j], &attrs[i])
    };
    Some(format!(
        "{}{}{}{}{}",
        &xml[..first.value.0],
        &xml[second.value.0..second.value.1],
        &xml[first.value.1..second.value.0],
        &xml[first.value.0..first.value.1],
        &xml[second.value.1..]
    ))
}

// ---- tiny XML scanning (deliberately approximate: garbage-in is the point) --------------

/// Byte range of every element `roxmltree` can find, `<tag ...>...</tag>` or `<tag .../>`.
/// Empty when `xml` does not parse — mutations that need element spans just find none then.
fn element_spans(xml: &str) -> Vec<(usize, usize)> {
    let Ok(doc) = roxmltree::Document::parse(xml) else {
        return Vec::new();
    };
    doc.descendants()
        .filter(roxmltree::Node::is_element)
        .map(|n| (n.range().start, n.range().end))
        .collect()
}

struct Attr {
    /// `key="value"` including both quotes, for a clean removal.
    full: (usize, usize),
    /// Just the text between the quotes.
    value: (usize, usize),
}

/// Scans for `name="value"` pairs without a real XML tokenizer — good enough for a fuzzer,
/// since a false positive just mutates a byte range that happens to look like an attribute.
fn find_attrs(xml: &str) -> Vec<Attr> {
    let b = xml.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < b.len() {
        if b[i] != b'=' || b[i + 1] != b'"' {
            i += 1;
            continue;
        }
        let mut key_start = i;
        while key_start > 0 && is_name_byte(b[key_start - 1]) {
            key_start -= 1;
        }
        let value_start = i + 2;
        let mut end = value_start;
        while end < b.len() && b[end] != b'"' {
            end += 1;
        }
        if key_start < i && end < b.len() {
            out.push(Attr {
                full: (key_start, end + 1),
                value: (value_start, end),
            });
            i = end + 1;
        } else {
            i += 1;
        }
    }
    out
}

fn is_name_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b':' | b'-' | b'.')
}

/// Deterministic xorshift64 (Marsaglia): no external randomness, so a seed reproduces exactly.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    fn choose<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            Some(&items[self.below(items.len())])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED_XML: &str =
        "<mujoco model=\"m\"><worldbody><body name=\"b\" pos=\"1 2 3\"/></worldbody></mujoco>";

    #[test]
    fn mutate_is_deterministic() {
        assert_eq!(mutate(SEED_XML, 42), mutate(SEED_XML, 42));
    }

    #[test]
    fn mutate_can_change_the_document() {
        let changed = (0..64).any(|seed| mutate(SEED_XML, seed) != SEED_XML);
        assert!(changed, "no mutation over 64 seeds changed the document");
    }

    #[test]
    fn mutate_never_panics_on_empty_or_garbage_input() {
        for seed in 0..32 {
            let _ = mutate("", seed);
            let _ = mutate("<<not xml>>", seed);
        }
    }

    #[test]
    fn fuzz_importer_reports_a_panic() {
        let result = panic::catch_unwind(|| {
            fuzz_importer(
                &[SEED_XML],
                |s: &str| -> Result<(), ()> { panic!("boom: {s}") },
                4,
            );
        });
        assert!(result.is_err());
    }

    #[test]
    fn fuzz_importer_accepts_a_well_behaved_importer() {
        fuzz_importer(
            &[SEED_XML],
            |s: &str| -> Result<String, ()> { Ok(s.to_owned()) },
            32,
        );
    }
}
