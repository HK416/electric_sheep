//! Packet G3b (`docs/packets/M17/plan-g.md`): `generate` writes every document of a project
//! from its specification and scene. The committed sets are the oracle, read only: the Shadow
//! Hand spec regenerates plan H's documents (`tests/fixtures/shadow-hand/`) and the SO-101 views
//! spec plan N's three-view arm (`tests/fixtures/visible-learning/*-views.toml`), its one-view
//! arm (`*-cam.toml`) and its MAD set (`*-mad.toml`) — every IR document by its semantic hash,
//! every recipe and cycle as the parsed `Recipe` / `Cycle`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use es_data::training::{Cycle, Recipe};
use es_script::spec::{compile_task, generate, Family, TaskSpec};

const HAND: &str = "tests/fixtures/estask/shadow_hand_repose.estask";
const VIEWS: &str = "tests/fixtures/estask/so101_views.estask";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    let path = repo().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn spec(rel: &str) -> TaskSpec {
    TaskSpec::from_toml(&read(rel)).expect("the spec reads")
}

fn hex(d: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*d).to_hex().to_string()
}

fn files(spec: &TaskSpec, out: &str) -> BTreeMap<String, String> {
    generate(spec, &repo(), out, "the spec")
        .expect("the spec generates")
        .into_iter()
        .map(|d| (d.file, d.text))
        .collect()
}

/// The document's own hash, by the kind its `kind = "..."` line names.
fn ir_hash(text: &str) -> String {
    use es_ir::serial as s;
    let kind = text
        .lines()
        .find_map(|l| l.strip_prefix("kind = \""))
        .expect("a kind line")
        .trim_end_matches('"');
    let h = match kind {
        "task" => s::task_from_toml(text).unwrap().task_hash(),
        "observation" => s::observation_from_toml(text).unwrap().observation_hash(),
        "learning" => s::learning_from_toml(text).unwrap().learning_hash(),
        "deployment" => s::deployment_from_toml(text).unwrap().deployment_hash(),
        "evaluation" => s::evaluation_from_toml(text).unwrap().evaluation_hash(),
        other => panic!("{other}"),
    };
    hex(&h.expect("hashes"))
}

/// `v`'s leaves (and its tables four levels down) as `path = value` lines.
fn walk(v: &toml::Value, at: &str, out: &mut Vec<String>) {
    match v {
        toml::Value::Table(t) if at.matches('.').count() < 4 => {
            for (k, v) in t {
                walk(v, &format!("{at}.{k}"), out);
            }
        }
        other => out.push(format!("{at} = {other}")),
    }
}

/// Every node, output and field of a document that the other does not have, for a failure.
fn diff(got: &str, want: &str) -> String {
    let lines = |t: &str| -> Vec<String> {
        let mut out = Vec::new();
        walk(&toml::from_str(t).unwrap(), "", &mut out);
        out
    };
    let (g, w) = (lines(got), lines(want));
    let only = |a: &[String], b: &[String]| -> Vec<String> {
        a.iter()
            .filter(|x| !b.contains(x))
            .map(|x| x.chars().take(400).collect())
            .collect()
    };
    format!(
        "only generated:\n  {}\nonly committed:\n  {}",
        only(&g, &w).join("\n  "),
        only(&w, &g).join("\n  ")
    )
}

fn same_ir(docs: &BTreeMap<String, String>, file: &str, committed: &str) -> String {
    let got = &docs[file];
    let want = read(committed);
    let (g, w) = (ir_hash(got), ir_hash(&want));
    assert!(
        g == w,
        "{file} vs {committed}: {g} != {w}\n{}",
        diff(got, &want)
    );
    let body = |t: &str| t.split_once("\nes_schema").map(|(_, b)| b.to_owned());
    let bytes = body(got) == body(&want);
    println!("{file:<34} = {committed:<52} {g} (body bytes equal: {bytes})");
    g
}

fn same_recipe(docs: &BTreeMap<String, String>, file: &str, committed: &str) {
    let got: Recipe = toml::from_str(&docs[file]).expect("generated recipe parses");
    let want: Recipe = toml::from_str(&read(committed)).expect("committed recipe parses");
    assert_eq!(got, want, "{file} vs {committed}");
    println!("{file:<34} = {committed} (Recipe equal)");
}

fn same_cycle(docs: &BTreeMap<String, String>, file: &str, committed: &str) {
    let got: Cycle = toml::from_str(&docs[file]).expect("generated cycle parses");
    let want: Cycle = toml::from_str(&read(committed)).expect("committed cycle parses");
    assert_eq!(got, want, "{file} vs {committed}");
    println!("{file:<34} = {committed} (Cycle equal)");
}

#[test]
fn the_shadow_hand_spec_regenerates_plan_hs_documents() {
    const OUT: &str = "tests/fixtures/shadow-hand";
    let at = |f: &str| format!("{OUT}/{f}");
    let docs = files(&spec(HAND), OUT);
    for (file, committed) in [
        ("task.toml", "task-repose.toml"),
        ("observation-teacher.toml", "observation-teacher.toml"),
        ("learning-teacher.toml", "learning-teacher.toml"),
        ("deployment-teacher.toml", "deployment-hand.toml"),
        ("evaluation-teacher.toml", "evaluation-teacher-64.toml"),
        ("observation-student.toml", "observation-student.toml"),
        ("learning-student.toml", "learning-student.toml"),
        ("deployment-student.toml", "deployment-student.toml"),
        ("evaluation-student.toml", "evaluation-student.toml"),
        (
            "evaluation-student-nominal.toml",
            "evaluation-student-nominal.toml",
        ),
    ] {
        same_ir(&docs, file, &at(committed));
    }
    same_recipe(
        &docs,
        "training-teacher.toml",
        &at("training-teacher-v2.toml"),
    );
    same_recipe(&docs, "training-student.toml", &at("training-student.toml"));
    same_cycle(&docs, "cycle-student.toml", &at("cycle-student.toml"));
    // H2's first evaluation: the same on the teacher's first 16 seeds.
    let mut first = spec(HAND);
    first.evaluate.as_mut().unwrap().episodes = Some(16);
    let docs = files(&first, OUT);
    same_ir(
        &docs,
        "evaluation-teacher.toml",
        &at("evaluation-teacher.toml"),
    );
}

#[test]
fn the_so101_views_spec_regenerates_plan_ns_arms() {
    const OUT: &str = "tests/fixtures/visible-learning";
    let at = |f: &str| format!("{OUT}/{f}");
    let check = |s: &TaskSpec, arm: &str, observation: bool| {
        let docs = files(s, OUT);
        same_ir(&docs, "task.toml", &at("task-views.toml"));
        let name = |kind: &str| format!("{kind}-{arm}.toml");
        let kinds: &[&str] = if observation {
            &["observation", "learning", "evaluation"]
        } else {
            &["learning", "evaluation"]
        };
        for kind in kinds {
            same_ir(&docs, &name(kind), &at(&name(kind)));
        }
        same_recipe(&docs, &name("training"), &at(&name("training")));
        same_cycle(&docs, &name("cycle"), &at(&name("cycle")));
    };
    let views = spec(VIEWS);
    check(&views, "views", true);
    // The one-view arm: the overhead camera alone, the same Task IR.
    let mut cam = views.clone();
    let student = cam.student.as_mut().unwrap();
    student.name = Some("cam".to_owned());
    student.views = Some(vec!["overhead".to_owned()]);
    check(&cam, "cam", true);
    // The MAD set: the three-view arm's observation, one shared encoder, the views summed.
    let mut mad = views;
    let student = mad.student.as_mut().unwrap();
    student.name = Some("mad".to_owned());
    student.family = Some(Family::Mad);
    check(&mad, "mad", false);
}

#[test]
fn generating_is_a_function_of_the_spec_and_the_scene() {
    for rel in [HAND, VIEWS] {
        let once = || generate(&spec(rel), &repo(), "out", rel).unwrap();
        assert_eq!(once(), once(), "{rel}");
    }
}

#[test]
fn the_extended_fixtures_read_back_as_written() {
    for rel in [HAND, VIEWS] {
        let doc = spec(rel);
        assert_eq!(
            TaskSpec::from_toml(&doc.to_toml().unwrap()).unwrap(),
            doc,
            "{rel}"
        );
    }
}

/// `robot` may name an `.esscene` include: the Shadow Hand scene document's `hand` is its root
/// body `robot0:hand mount` (the include's other root, `floor0`, has no joints).
#[test]
fn the_robot_can_be_named_by_its_include() {
    let mut by_body = spec(HAND);
    by_body.scene = "tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.esscene".to_owned();
    let mut by_include = by_body.clone();
    by_include.robot = "hand".to_owned();
    let hash = |s: &TaskSpec| compile_task(s, &repo()).unwrap().task_hash().unwrap();
    assert_eq!(hash(&by_include), hash(&by_body));
    by_include.robot = "hnd".to_owned();
    let err = compile_task(&by_include, &repo()).unwrap_err().to_string();
    assert!(err.contains("robot") && err.contains("hnd"), "{err}");
}

/// `doc` with `from` replaced by `to` (exactly once) must be refused naming every `needle`.
fn refused(rel: &str, from: &str, to: &str, needles: &[&str]) {
    let doc = read(rel);
    assert_eq!(doc.matches(from).count(), 1, "`{from}` is not unique");
    let err = TaskSpec::from_toml(&doc.replace(from, to))
        .and_then(|s| generate(&s, &repo(), "out", rel))
        .expect_err(to)
        .to_string();
    for n in needles {
        assert!(err.contains(n), "`{to}`: `{err}` does not name `{n}`");
    }
}

#[test]
fn refusals_name_the_section_and_the_field() {
    refused(
        HAND,
        "state = [\"joint_pos\", \"target_qpos\"]",
        "state = [\"joint_pos\", \"cube_pose\"]",
        &["student", "state", "cube_pose", "privileged"],
    );
    refused(
        HAND,
        "steps = 60000",
        "stepz = 60000",
        &["student", "training", "stepz"],
    );
    refused(HAND, "execute = 6", "execute = 7", &["student", "execute"]);
    refused(HAND, "[deploy]", "[deploy]\nrobot = 1", &["robot"]);
    refused(
        VIEWS,
        "expert = \"so101-pick-place\"\n",
        "",
        &["cycle", "expert"],
    );
    refused(
        VIEWS,
        "name = \"views\"",
        "name = \"views\"\nviews = [\"top\"]",
        &["student", "views", "top"],
    );
}
