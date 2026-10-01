//! Packet M17/G9 (`docs/packets/M17/plan-g.md`, task G9), oracles 1 to 4: an authored project
//! runs end to end. The empty project is made as the start screen makes it, and ① builds its
//! task as a person does (`es_editor::ui::sentence::authored`: the library's SO-101, a box, a
//! target area, "say the task", "[box] is inside [area]", save). Then:
//!
//! 1. the generated set is whole, every document passes `es ir check`, every recipe's
//!    `es policy init` line builds its bundle, and `es loop cycle --dry-run` passes on the
//!    generated cycle and on the one ③ writes — all run as the editor runs them, in the
//!    project's own folder;
//! 2. the argvs ② and ③ build name the generated documents (pinned), while a template
//!    project's still name the template's in the repository root;
//! 3. an unsaved, stale or failed generation blocks ③ with its reason, and saving clears it;
//! 4. a project saved as a template and made again from it has the same scene, task and
//!    generated documents.
//!
//! `es` is the binary built beside this test (`cargo build -p es`, or `cargo test --workspace`
//! as `cargo xtask ci` runs it), as `crates/es-policy/tests/ir_training.rs` finds it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use es_assets::esscene::{expand, EsScene};
use es_editor::model::home::saved_dir;
use es_editor::model::launch::LaunchModel;
use es_editor::model::project::{self, Project, StartSettings, RUN_RECIPE};
use es_editor::model::teacher;
use es_editor::model::telemetry_view::TelemetryModel;
use es_editor::model::template::{self, Generated, Length, Method, Template};
use es_editor::model::watch::Watch;
use es_editor::model::workflow::{Phase, PhaseState};
use es_editor::ui::scene::{generated, new_documents, save_template};
use es_editor::ui::sentence::authored;
use es_editor_scene::{on_disk, BackendKind, Command as Edit, Regen, SceneModel, SPEC_FILE};

/// Every document "say the task" makes `generate` write (packet M17/G9).
const ALL: [&str; 13] = [
    "task.toml",
    "observation-teacher.toml",
    "learning-teacher.toml",
    "deployment-teacher.toml",
    "evaluation-teacher.toml",
    "training-teacher.toml",
    "observation-student.toml",
    "learning-student.toml",
    "deployment-student.toml",
    "evaluation-student.toml",
    "evaluation-student-nominal.toml",
    "training-student.toml",
    "cycle-student.toml",
];

fn repo() -> PathBuf {
    std::path::absolute(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap()
}

/// `CARGO_BIN_EXE_es` exists only in `es`'s own tests; both binaries share a profile folder.
fn es_bin() -> PathBuf {
    let mut dir = std::env::current_exe().expect("the test has an executable path");
    dir.pop();
    if dir.file_name().is_some_and(|n| n == "deps") {
        dir.pop();
    }
    let exe = dir.join(format!("es{}", std::env::consts::EXE_SUFFIX));
    assert!(
        exe.is_file(),
        "{} is not built; run `cargo build -p es` (or `cargo test --workspace`) first",
        exe.display()
    );
    exe
}

/// `es args` in `cwd`, as the editor's launch model starts it.
fn es(cwd: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(es_bin())
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("run es");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    println!(
        "RAN es {} (in {}): {}",
        args.join(" "),
        cwd.display(),
        out.status
    );
    (out.status.success(), text)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-g9-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn template_named(id: &str) -> Template {
    let (all, bad) = template::load(&repo());
    assert!(bad.is_empty(), "{bad:?}");
    all.into_iter().find(|t| t.id == id).expect(id)
}

/// The empty project as the start screen's card makes it — no task: ② to ⑤ say so — then
/// ①'s task, saved.
fn authored_project(tag: &str) -> (Project, SceneModel) {
    let root = scratch(tag);
    let empty = template_named("empty");
    assert!(empty.bundle.is_none() && empty.editable.is_some());
    new_documents(&root, &empty, &repo()).unwrap();
    let project = Project::create(&root, tag, &empty, &repo()).unwrap();
    assert!(!project.bundle().exists(), "no documents, no bundle");
    let mut m = SceneModel::open(&project.root, vec![BackendKind::MuJoCoCpu]).unwrap();
    m.refresh();
    assert_eq!(m.generated(), &Regen::NoSpec);
    let none = template::source(&project, Some(repo()), Some(&Generated::NoSpec));
    assert_eq!(none.err(), Some(("watch.task.none", String::new())));
    authored(&mut m, &repo()).unwrap();
    (project, m)
}

fn files(m: &SceneModel) -> Vec<String> {
    match m.generated() {
        Regen::Written(files) => files.clone(),
        other => panic!("{other:?}"),
    }
}

/// What ② to ⑤ run for the saved project.
fn source(project: &Project, m: &SceneModel) -> (Template, PathBuf) {
    let now = generated(m.generated(), m.dirty() && m.spec().is_some());
    template::source(project, Some(repo()), Some(&now)).expect("runnable")
}

/// The `es policy init` line a generated recipe's header carries, as its arguments.
fn init_line(recipe: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut on = false;
    for line in recipe.lines() {
        let line = line.trim_start_matches('#').trim();
        on |= line.starts_with("es policy init");
        if on {
            words.extend(
                line.split_whitespace()
                    .filter(|w| *w != "\\")
                    .map(str::to_owned),
            );
            if !line.ends_with('\\') {
                break;
            }
        }
    }
    assert_eq!(words.first().map(String::as_str), Some("es"), "{recipe}");
    words.split_off(1)
}

fn settings() -> StartSettings {
    StartSettings {
        demonstrations: 4,
        length: Length::Short,
    }
}

/// Oracle 1.
#[test]
fn an_authored_project_checks_builds_and_dry_runs_in_its_own_folder() {
    let (project, m) = authored_project("e2e");
    assert_eq!(files(&m), ALL);
    let root = project.root.clone();
    let (template, cwd) = source(&project, &m);
    assert_eq!(cwd, root, "es runs in the project's own folder");
    // Every document, each arm's five together (the student's twice: both its evaluations).
    for (arm, evaluation) in [
        ("teacher", "evaluation-teacher"),
        ("student", "evaluation-student"),
        ("student", "evaluation-student-nominal"),
    ] {
        let doc = |kind: &str| format!("generated/{kind}-{arm}.toml");
        let evaluation = format!("generated/{evaluation}.toml");
        let args = [
            "ir",
            "check",
            "generated/task.toml",
            &doc("observation"),
            &doc("learning"),
            &doc("deployment"),
            &evaluation,
        ];
        let (ok, text) = es(&cwd, &args);
        assert!(
            ok && !text.contains("ERROR"),
            "es ir check ({evaluation})\n{text}"
        );
    }
    for recipe in ["training-teacher.toml", "training-student.toml"] {
        let text = std::fs::read_to_string(root.join("generated").join(recipe)).unwrap();
        let args = init_line(&text);
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let (ok, out) = es(&cwd, &args);
        assert!(ok, "{recipe}: es {args:?}\n{out}");
        let at = args.iter().position(|a| *a == "--out").unwrap() + 1;
        let bundle = std::fs::read(cwd.join(args[at])).unwrap();
        es_compile::PolicyBundle::open(&bundle).expect("the bundle opens");
    }
    let out = scratch("e2e-dry");
    let out = out.to_string_lossy();
    let cycle = ["loop", "cycle", "--recipe", "generated/cycle-student.toml"];
    let (ok, text) = es(&cwd, &[&cycle[..], &["--out", &out, "--dry-run"]].concat());
    assert!(ok && text.contains("runs/teacher.esb"), "{text}");

    // ②'s teacher run as the editor writes it: plan H's PPO on MuJoCo Warp.
    let dir = project.next_teacher_dir();
    let argv = teacher::train(&template, &cwd, &project, &dir, "127.0.0.1:7122").unwrap();
    let mut args: Vec<&str> = argv.iter().map(String::as_str).collect();
    args.retain(|a| *a != "--telemetry" && *a != "127.0.0.1:7122");
    args.push("--dry-run");
    let (ok, text) = es(&cwd, &args);
    assert!(
        ok && text.contains("train_ppo") && text.contains("mjwarp"),
        "{text}"
    );

    // ③ as the editor writes it, once a teacher is chosen (here the untrained one the
    // teacher recipe's header built: no learning runs in a test).
    assert_eq!(template.method, Method::Teacher);
    std::fs::copy(
        root.join("runs/teacher-untrained.esb"),
        project.teacher_bundle(),
    )
    .unwrap();
    let run = project.next_run_dir();
    let argv = project::write_run(
        &template,
        &cwd,
        &project,
        settings(),
        &run,
        "127.0.0.1:7123",
    )
    .unwrap();
    assert!(project.bundle().is_file(), "③ built the student's bundle");
    let mut args: Vec<&str> = argv.iter().map(String::as_str).collect();
    args.retain(|a| *a != "--telemetry" && *a != "127.0.0.1:7123");
    args.push("--dry-run");
    let (ok, text) = es(&cwd, &args);
    assert!(ok && text.contains("evaluation-student.toml"), "{text}");
    let _ = std::fs::remove_dir_all(&root);
}

/// Oracle 2: ② and ③'s argvs on the authored project name its generated documents, with
/// `<root>` for the project's folder; a template project's are as before.
#[test]
fn the_editor_argvs_name_the_generated_documents() {
    let (project, m) = authored_project("argv");
    let (template, cwd) = source(&project, &m);
    let root = project.root.display().to_string();
    let pin = |argv: &[String]| -> Vec<String> {
        (argv.iter())
            .map(|a| a.replace(&root, "<root>").replace('\\', "/"))
            .collect()
    };
    let t = template.teacher.as_ref().expect("[teacher]");
    assert_eq!(
        (
            t.recipe.as_str(),
            t.evaluation.as_str(),
            t.deployment.as_str(),
            t.jobs
        ),
        (
            "generated/training-teacher.toml",
            "generated/evaluation-teacher.toml",
            "generated/deployment-teacher.toml",
            1
        )
    );
    let b = template.bundle.as_ref().expect("[bundle]");
    assert_eq!(
        (
            b.task.as_str(),
            b.observation.as_str(),
            b.learning.as_deref()
        ),
        (
            "generated/task.toml",
            "generated/observation-student.toml",
            Some("generated/learning-student.toml")
        )
    );
    assert_eq!(
        (
            template.cycle.as_str(),
            template.scene.as_str(),
            template.demonstrations
        ),
        ("generated/cycle-student.toml", "scene.esscene", 200)
    );

    // ②: a teacher run, and a test of its checkpoint.
    let dir = project.next_teacher_dir();
    let argv = teacher::train(&template, &cwd, &project, &dir, "127.0.0.1:7124").unwrap();
    assert_eq!(
        pin(&argv),
        [
            "train",
            "--recipe",
            "<root>/teacher/001/recipe.toml",
            "--out",
            "<root>/teacher/001",
            "--telemetry",
            "127.0.0.1:7124"
        ]
    );
    let read = |p: &Path| es_data::training::Recipe::parse(&std::fs::read_to_string(p).unwrap());
    let mut written = read(&dir.join(teacher::RECIPE)).unwrap();
    let generated_recipe = read(&cwd.join("generated/training-teacher.toml")).unwrap();
    let bundle = written.policy.bundle.take().unwrap();
    assert_eq!(pin(&[bundle]), ["<root>/teacher-untrained.esb"]);
    assert_eq!(
        (written.rl, written.run),
        (generated_recipe.rl, generated_recipe.run),
        "the generated PPO recipe, as it is"
    );
    let folder = project.teacher_runs().pop().unwrap();
    let argv = teacher::evaluate(&template, &cwd, &folder, 250).unwrap();
    assert_eq!(
        pin(&argv),
        [
            "eval",
            "run",
            "--config",
            "<root>/generated/evaluation-teacher.toml",
            "--policy",
            "<root>/teacher/001/checkpoints/250.esb",
            "--scene",
            "<root>/scene.esscene",
            "--out",
            "<root>/teacher/001/eval/250",
            "--jobs",
            "1"
        ]
    );

    // ③: the run's recipe names the generated cycle's documents, relative to the folder `es`
    // runs in, and the project's two bundles.
    std::fs::write(project.teacher_bundle(), "chosen").unwrap();
    let run = project.next_run_dir();
    let argv = project::write_run(
        &template,
        &cwd,
        &project,
        settings(),
        &run,
        "127.0.0.1:7125",
    )
    .unwrap();
    assert_eq!(
        pin(&argv),
        [
            "loop",
            "cycle",
            "--recipe",
            "<root>/runs/001/cycle.toml",
            "--out",
            "<root>/runs/001",
            "--telemetry",
            "127.0.0.1:7125"
        ]
    );
    let text = std::fs::read_to_string(run.join(RUN_RECIPE)).unwrap();
    let cycle = es_data::training::Cycle::parse(&text).unwrap();
    let collect = cycle.collect.as_ref().unwrap();
    assert_eq!(
        (
            cycle.scene.as_str(),
            cycle.eval.config.as_str(),
            pin(std::slice::from_ref(&collect.policy)),
            collect.expert.clone(),
            collect.success_only,
        ),
        (
            "scene.esscene",
            "generated/evaluation-student.toml",
            vec!["<root>/teacher.esb".to_owned()],
            None,
            true
        )
    );
    let policy = cycle.train.policy.as_ref().unwrap();
    assert_eq!(
        pin(&[policy.bundle.clone().unwrap()]),
        ["<root>/untrained.esb"]
    );

    // A template project: the template's documents, `es` in the repository root.
    let cube = Project::create(
        &scratch("argv-cube"),
        "cube",
        &template_named("cube-into-bin"),
        &repo(),
    )
    .unwrap();
    let (t, cwd) = template::source(&cube, Some(repo()), None).unwrap();
    assert_eq!(
        (t.id.as_str(), t.generated, cwd),
        ("cube-into-bin", false, repo())
    );
    assert_eq!(t.cycle, "tests/fixtures/visible-learning/cycle-vision.toml");
    let _ = std::fs::remove_dir_all(&cube.root);
    let _ = std::fs::remove_dir_all(&project.root);
}

/// `argv` with the project's folder as `<root>`, separators forward.
fn pin(root: &Path, argv: &[String]) -> Vec<String> {
    let root = root.display().to_string();
    (argv.iter())
        .map(|a| a.replace(&root, "<root>").replace('\\', "/"))
        .collect()
}

/// Packet M17/R7 (F-1): ②'s test of a checkpoint trained before a save regenerated the
/// documents is `es policy pack` of its weights on the untrained teacher rebuilt from the new
/// documents, then `es eval run` of that re-pack (pinned); on unchanged documents, the
/// evaluation alone, as before.
#[test]
fn a_checkpoint_from_before_a_save_is_repacked_for_its_test() {
    let (project, mut m) = authored_project("repack");
    let (template, cwd) = source(&project, &m);
    let dir = project.next_teacher_dir();
    teacher::train(&template, &cwd, &project, &dir, "127.0.0.1:7126").unwrap();
    // Its checkpoint: the untrained teacher on the documents of the time (no learning runs).
    std::fs::create_dir_all(dir.join("checkpoints")).unwrap();
    let ckpt = dir.join("checkpoints").join("250.esb");
    std::fs::copy(project.root.join(teacher::UNTRAINED), &ckpt).unwrap();
    let run = project.teacher_runs().pop().unwrap();
    let same = teacher::test(&template, &cwd, &project, &run, 250).unwrap();
    assert_eq!(
        same,
        [teacher::evaluate(&template, &cwd, &run, 250).unwrap()]
    );

    let mut longer = m.spec().cloned().unwrap();
    longer.timeout_s = 10.0;
    m.apply(&Edit::Spec(Some(Box::new(longer)))).unwrap();
    m.save().unwrap();
    let (template, cwd) = source(&project, &m);
    let jobs = teacher::test(&template, &cwd, &project, &run, 250).unwrap();
    let jobs: Vec<Vec<String>> = jobs.iter().map(|a| pin(&project.root, a)).collect();
    assert_eq!(
        jobs,
        [
            vec![
                "policy",
                "pack",
                "--policy",
                "<root>/teacher-untrained.esb",
                "--weights",
                "<root>/teacher/001/weights/model-250.safetensors",
                "--out",
                "<root>/teacher/001/repacked/250.esb"
            ],
            vec![
                "eval",
                "run",
                "--config",
                "<root>/generated/evaluation-teacher.toml",
                "--policy",
                "<root>/teacher/001/repacked/250.esb",
                "--scene",
                "<root>/scene.esscene",
                "--out",
                "<root>/teacher/001/eval/250",
                "--jobs",
                "1"
            ]
        ]
    );
    let task = |p: &Path| {
        let bundle = es_compile::PolicyBundle::open(&std::fs::read(p).unwrap()).unwrap();
        bundle.manifest.hashes.task
    };
    assert_ne!(
        task(&project.root.join(teacher::UNTRAINED)),
        task(&ckpt),
        "the untrained teacher is rebuilt on the saved documents"
    );
    let _ = std::fs::remove_dir_all(&project.root);
}

/// Packet M17/R7 (F-1) for real, on a copy of GV's `push-box` project named by `ES_R7_PROJECT`
/// (never the owner's folder: its `generated/` is rewritten). Regenerated by today's generator,
/// its documents are not those teacher 001 trained on; ②'s test of checkpoint 2000 packs it and
/// evaluates the re-pack on `mujoco-cpu` (`ES_PYTHON`), and the evaluation passes.
#[test]
fn gv_checkpoint_2000_is_repacked_and_passes() {
    let Some(python) = std::env::var_os("ES_PYTHON") else {
        println!(
            "SKIP gv_checkpoint_2000_is_repacked_and_passes: ES_PYTHON is not set (mujoco-cpu)"
        );
        return;
    };
    let Some(root) = std::env::var_os("ES_R7_PROJECT").map(PathBuf::from) else {
        println!("SKIP gv_checkpoint_2000_is_repacked_and_passes: ES_R7_PROJECT names no copy of GV's project");
        return;
    };
    println!("ES_PYTHON = {}", python.to_string_lossy());
    let project = Project::open(&root).unwrap();
    let mut m = SceneModel::open(&root, vec![BackendKind::MuJoCoCpu]).unwrap();
    m.regenerate();
    let (template, cwd) = source(&project, &m);
    let run = (project.teacher_runs().into_iter())
        .find(|r| r.number == 1)
        .expect("teacher 001");
    let jobs = teacher::test(&template, &cwd, &project, &run, 2000).unwrap();
    for argv in &jobs {
        println!("{:?}", pin(&root, argv));
    }
    assert_eq!(jobs.len(), 2, "the documents moved: pack, then eval");
    let _ = std::fs::remove_dir_all(teacher::eval_dir(&run, 2000));
    // `es eval run` exits 0 only when every acceptance line passed.
    for argv in &jobs {
        let args: Vec<&str> = argv.iter().map(String::as_str).collect();
        let (ok, text) = es(&cwd, &args);
        println!("{text}");
        assert!(ok, "es {args:?}\n{text}");
    }
    let score = teacher::score(&teacher::eval_dir(&run, 2000)).expect("a report");
    println!("checkpoint 2000, re-packed: {}", score.text());
    assert_eq!(score.episodes, 16);
}

/// ③ for `generated`: why it cannot start, or `None` and Start offered.
fn three(
    project: &Project,
    now: &Generated,
) -> (
    Option<(&'static str, String)>,
    Option<&'static str>,
    PhaseState,
) {
    let source = template::source(project, Some(repo()), Some(now));
    let mut watch = Watch::with_source(project, source);
    let (mut launch, telemetry) = (LaunchModel::default(), TelemetryModel::default());
    let at = Instant::now();
    let (_, phases) = watch.tick(&mut launch, &telemetry, Phase::Train, at);
    let view = watch.view(Phase::Train, &launch, &telemetry, &phases, at);
    (view.cannot_start, view.start, phases[0].clone())
}

/// Oracle 3.
#[test]
fn a_stale_or_failed_generation_blocks_training_until_saved() {
    let (project, mut m) = authored_project("block");
    std::fs::write(project.teacher_bundle(), "chosen").unwrap();
    let state = |m: &SceneModel| generated(m.generated(), m.dirty() && m.spec().is_some());
    let ready = (None, Some("watch.start"), PhaseState::Done);
    assert_eq!(three(&project, &state(&m)), ready);

    // An edit in ① not saved yet.
    let spec = m.spec().cloned().unwrap();
    let mut longer = spec.clone();
    longer.timeout_s = 10.0;
    m.apply(&Edit::Spec(Some(Box::new(longer)))).unwrap();
    let blocked =
        |key: &'static str, arg: &str| (Some((key, arg.to_owned())), None, PhaseState::NotStarted);
    assert_eq!(
        three(&project, &state(&m)),
        blocked("watch.task.unsaved", "")
    );
    m.save().unwrap();
    assert_eq!(three(&project, &state(&m)), ready);

    // The specification changed on disk, behind the editor's back.
    let path = project.root.join(SPEC_FILE);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replace("timeout_s = 10.0", "timeout_s = 12.0")).unwrap();
    let disk = generated(&on_disk(&project.root), false);
    assert_eq!(three(&project, &disk), blocked("watch.task.stale", ""));
    std::fs::write(&path, &text).unwrap();
    assert_eq!(
        three(&project, &generated(&on_disk(&project.root), false)),
        ready
    );

    // A specification that compiles but does not generate (a chunk that replans at no whole
    // rate): saved, its generation fails, and says why.
    let mut odd = spec.clone();
    odd.student.as_mut().unwrap().execute = 3;
    m.apply(&Edit::Spec(Some(Box::new(odd)))).unwrap();
    m.save().unwrap();
    let (why, start, first) = three(&project, &state(&m));
    let (key, arg) = why.unwrap();
    assert_eq!(
        (key, start, first),
        ("watch.task.failed", None, PhaseState::NotStarted)
    );
    assert!(arg.contains("execute"), "{arg}");
    m.apply(&Edit::Spec(Some(Box::new(spec)))).unwrap();
    m.save().unwrap();
    assert_eq!(three(&project, &state(&m)), ready);
    let _ = std::fs::remove_dir_all(&project.root);
}

/// The scene document at `root`, expanded.
fn scene_hash(root: &Path) -> [u8; 32] {
    let text = std::fs::read_to_string(root.join("scene.esscene")).unwrap();
    expand(&EsScene::from_toml(&text).unwrap(), root)
        .unwrap()
        .scene_hash()
}

/// Oracle 4.
#[test]
fn a_saved_template_makes_the_same_project_again() {
    let (first, _) = authored_project("saved");
    let docs = scratch("saved-docs");
    let empty = template_named("empty");
    let dir = save_template(&first.root, &empty, "My push task", Some(&docs)).unwrap();
    assert_eq!(dir, saved_dir(Some(&docs)).join("My push task"));
    let (saved, bad) = template::load_saved(&saved_dir(Some(&docs)));
    assert!(bad.is_empty(), "{bad:?}");
    let [t] = &saved[..] else { panic!("{saved:?}") };
    assert_eq!(
        (
            t.id.as_str(),
            t.name.as_str(),
            t.base.as_deref(),
            t.bundle.is_none()
        ),
        ("saved:My push task", "My push task", Some("empty"), true)
    );
    assert_eq!(t.lengths, empty.lengths);

    let root = scratch("saved-again");
    new_documents(&root, t, &repo()).unwrap();
    let again = Project::create(&root, "again", t, &repo()).unwrap();
    assert_eq!(
        again.file.template, "empty",
        "it names the built-in template, not the folder"
    );
    assert_eq!(scene_hash(&again.root), scene_hash(&first.root));
    let spec = |r: &Path| {
        es_editor_scene::sentence::TaskSpec::from_toml(
            &std::fs::read_to_string(r.join(SPEC_FILE)).unwrap(),
        )
        .unwrap()
    };
    assert_eq!(spec(&again.root), spec(&first.root));
    let mut m = SceneModel::open(&again.root, vec![BackendKind::MuJoCoCpu]).unwrap();
    m.regenerate();
    assert_eq!(files(&m), ALL);
    for f in ALL {
        let read = |r: &Path| std::fs::read_to_string(r.join("generated").join(f)).unwrap();
        assert_eq!(read(&again.root), read(&first.root), "{f}");
    }
    let task = |r: &Path| {
        let text = std::fs::read_to_string(r.join("generated/task.toml")).unwrap();
        es_ir::serial::task_from_toml(&text)
            .unwrap()
            .task_hash()
            .unwrap()
    };
    assert_eq!(task(&again.root), task(&first.root));
    for d in [first.root, again.root, docs] {
        let _ = std::fs::remove_dir_all(d);
    }
}
