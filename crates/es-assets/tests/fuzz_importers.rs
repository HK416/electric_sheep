//! P35 oracle: run the mutation fuzzer against every MJCF and URDF fixture, ~256 mutations
//! each, and assert neither importer ever panics. Deterministic (see `es_assets::fuzz`), so a
//! CI failure reproduces locally from the same fixture set.

use es_assets::fuzz::fuzz_importer;
use es_assets::urdf::{parse_urdf, PackageResolver};

const CASES: u32 = 256;

fn read_all(dir: &str) -> Vec<(String, String)> {
    let path = format!("{}/../../{dir}", env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<_> = std::fs::read_dir(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "xml" || ext == "urdf")
        })
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            (name, std::fs::read_to_string(&p).unwrap())
        })
        .collect()
}

#[test]
fn fuzz_mjcf_importer_never_panics_on_mutated_fixtures() {
    let fixtures = read_all("tests/fixtures/mjcf");
    let docs: Vec<&str> = fixtures.iter().map(|(_, xml)| xml.as_str()).collect();
    fuzz_importer(&docs, es_assets::parse_mjcf, CASES);
}

#[test]
fn fuzz_urdf_importer_never_panics_on_mutated_fixtures() {
    let fixtures = read_all("tests/fixtures/urdf");
    let docs: Vec<&str> = fixtures.iter().map(|(_, xml)| xml.as_str()).collect();
    // A resolver that knows the one package the fixtures reference: an unresolved `package://`
    // must come back as a normal `Err`, not something the fuzzer should keep tripping over.
    let resolver = PackageResolver::new(std::collections::BTreeMap::from([(
        "robo_pkg".to_owned(),
        std::path::PathBuf::from("/pkgs/robo_pkg"),
    )]));
    fuzz_importer(&docs, |xml| parse_urdf(xml, &resolver), CASES);
}
