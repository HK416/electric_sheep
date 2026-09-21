# M7 E6 — the editor for someone who is not an expert: plain words, a home screen, native file dialogs, a font that renders Korean

Spec: §23.1 (the editor is a client), §23.2 ("see every layer on one screen" — for a person, not
only for an engineer), §23.3 (what is visible during a run), §13.1 (the loop is the workflow:
collect → train → evaluate → watch), §28.10 rule 3 (nothing decided in `app.rs`; headless
view-models first; what a display-less CI cannot judge is design, not implementation), §1.4.
Owner directive 2026-09-21: *the editor is too hard for people without domain knowledge;
improve the UI/UX, and pick a font so that multilingual text does not break.* Design note:
`docs/design/editor-shell.md` (+ `.ko.md`) section 15. Depends on **E1–E5** as they are.

## the question

Every screen of the editor today speaks to the engineer who wrote it: a bare text field that
wants "bundle.esb, a directory of the five .toml files, or a run", a launch panel whose labels
are `--config` / `--policy` / `--out`, a results table headed `envelope_violation_rate` and
`failure_mode_histogram`, help text that cites "(spec 23.3)", and a default font with no
Hangul, Kana or Han glyphs, so a Korean path or label renders as boxes. **Can the same
view-models be shown in plain words, in the reader's language, with the technical name one
hover away — and can that be judged headlessly?**

## spec

* **Strings live in a table, not in `app.rs`.** `crates/es-editor/i18n/en.toml` and `ko.toml`
  (`key = "text"`, flat keys grouped by dotted prefix: `home.open_project`, `launch.policy`,
  `metric.success_rate`, …), loaded with `include_str!` into `model/i18n.rs`: `Lang { En, Ko }`,
  `Strings::get(lang) -> &Strings`, `t(key) -> &str`. A missing key in either table is a **test
  failure**, and so is a key present in a table but never used by the crate (the test greps the
  crate's own source for `t("…")`). `Lang` is persisted in `eframe::Storage` beside the recent
  list (E3) and toggled in the top bar (`English` / `한국어`); default `En`. Korean text is
  allowed in exactly these two files: `.githooks/pre-commit`'s `is_doc` gains the case
  `*/i18n/*.toml`, and CLAUDE.md's Conventions paragraph says so in one sentence.
* **A font that renders CJK.** `model/fonts.rs::system_cjk_font() -> Option<(String, Vec<u8>)>`
  probes a fixed per-OS candidate list — Windows `C:\Windows\Fonts\malgun.ttf` (Malgun Gothic),
  `msyh.ttc`, `meiryo.ttc`; macOS `/System/Library/Fonts/AppleSDGothicNeo.ttc`,
  `/System/Library/Fonts/PingFang.ttc`, `/System/Library/Fonts/Supplemental/NotoSansCJK*.ttc`;
  Linux `/usr/share/fonts/**/NotoSansCJK*.{ttc,otf}`, `NanumGothic.ttf`, `DroidSansFallback*.ttf`
  — and returns the first that exists (no font-discovery dependency: a list and `std::fs`).
  `main.rs` pushes it as the **last fallback** of both `FontFamily::Proportional` and
  `FontFamily::Monospace`, so Latin keeps egui's own glyphs and CJK falls through to the system
  font; when none is found the status bar says which paths were tried and which package installs
  one (`fonts-noto-cjk` on Debian/Ubuntu). No font file is committed. The default text style is
  raised to 15 px body / 20 px heading and a **Text size** setting (`S / M / L`, persisted) scales
  all of them — a person who cannot read 13 px should not have to know egui.
* **A home screen.** With nothing open, the Graph tab's empty state becomes a home: three large
  buttons — *Open a project file…*, *Open a run result…*, *Recent* — and a five-step strip in the
  §13.1 order, each with one sentence in plain words (`home.step.design`, `home.step.collect`,
  `home.step.train`, `home.step.evaluate`, `home.step.watch`) and a button that goes to the tab or
  panel that does it. Tab names become the workflow's words — `Design`, `Results`, `Live`, `What
  the policy sees`, `Problems`, `Edit` — with the old name in the hover text.
* **Native file dialogs.** *Open…*, *Browse…* beside every path field of the launch panel and the
  replay panel, through `rfd` — **the one new dependency**, MIT/Apache, native on Windows and
  macOS. It is declared as a target-specific dependency
  (`[target.'cfg(any(windows, target_os = "macos"))'.dependencies]`) behind a `file-dialogs`
  feature that is default on those targets only, so Linux keeps the typed path and drag-drop and
  gains no GTK requirement; `model/dialogs.rs` wraps it in `pick_file(filter) -> Option<PathBuf>`
  / `pick_dir()` with a `cfg`-off stub returning `None`, and `app.rs` never names `rfd`.
* **Plain words, technical name one hover away.** `model/labels.rs`: `metric_label(&MetricSpec)`
  (`success_rate` → *Success rate*; `envelope_violation_rate` → *Safety limit hits*;
  `failure_mode_histogram` → *Failure causes*; `episode_length` → *Episode length*; …) is total
  over `MetricSpec::ALL` (a wildcard arm is a test failure); `launch_label(LaunchField)` (`--config`
  → *Evaluation settings*, `--policy` → *Policy file*, `--scene` → *Scene file*, `--out` → *Output
  folder*, `--frames` → *Also save frames*, `--jobs` → *Parallel workers*, `--telemetry` → *Watch
  live at*, …) is total over `LaunchField`/`LaunchFlag`; every table header, launch label and tab
  shows the plain word and hovers the raw name/flag. **No visible string cites a spec section**:
  every "(spec N.N)" moves to a hover; the i18n test asserts no key outside `*.hint` contains
  "spec ". Empty states of every tab say in one sentence what to open and where the Open button is.
* **Not here.** Localising `es`'s own CLI output (the launch log shows it verbatim); a redesign of
  the layered graph; icons beyond egui's text; translations beyond `en`/`ko` (adding one is one
  file); a font bundled in the repository.

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
crates/es-editor/src/app.rs
crates/es-editor/src/main.rs
crates/es-editor/src/lib.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/model/i18n.rs
crates/es-editor/src/model/fonts.rs
crates/es-editor/src/model/labels.rs
crates/es-editor/src/model/dialogs.rs
crates/es-editor/src/model/launch.rs
crates/es-editor/src/model/recent.rs
crates/es-editor/Cargo.toml
crates/es-editor/i18n/en.toml
crates/es-editor/i18n/ko.toml
crates/es-editor/tests/**
Cargo.toml
Cargo.lock
.githooks/pre-commit
CLAUDE.md
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E6-plain-language-editor.md
docs/packets/M7/E6-plain-language-editor.ko.md
```

`model/{i18n,fonts,labels,dialogs}.rs` (new; each with its tests), `launch.rs` (labels via the
table; nothing in its argv changes — the goldens stay), `recent.rs` (the two persisted settings
beside the list), `app.rs`/`main.rs` (wiring, fonts, the home screen), `Cargo.toml` (root: `rfd`
pinned in `[workspace.dependencies]`; crate: the target-specific dependency), the two string
tables, the hook's one-line allowlist, CLAUDE.md's one sentence, the design note, this packet.

## oracle

1. `cargo test -p es-editor i18n_tables_are_complete_and_used` — every key of `en.toml` is in
   `ko.toml` and vice versa; every `t("…")` key in the crate's source exists; every key in the
   tables is used; no non-`hint` value contains "spec ".
2. `cargo test -p es-editor metric_and_launch_labels_are_total` — `metric_label` covers
   `MetricSpec::ALL`, `launch_label` covers every `LaunchField` and `LaunchFlag`, the labels are
   pairwise distinct in both languages.
3. `cargo test -p es-editor system_cjk_font_is_found_or_the_reason_is_named` — on a machine with
   one of the candidates the loader returns its bytes and name (this Windows box has
   `malgun.ttf`); otherwise the returned reason lists every path tried. Plus
   `font_fallback_is_last`: the `FontDefinitions` built by `fonts::install` keep egui's fonts first.
4. `cargo test -p es-editor launch_argv_is_the_golden` unchanged — labels changed, argv did not.
5. `cargo test -p es-editor home_screen_lists_the_five_steps_in_loop_order` — the headless home
   model yields the five steps in §13.1 order with a non-empty sentence each in both languages.
6. `cargo build -p es-editor` (Windows, with dialogs) **and** `cargo build -p es-editor
   --no-default-features` (the Linux shape); `cargo xtask ci` (layering, budget: `es-editor` stays
   under 6,000 source lines — say the number); `cargo xtask check-scope docs/packets/M7/E6-plain-language-editor.md`.

## acceptance

Oracles 1–6. Screenshots under the worktree's `target/plan-u/e6/`, in **both** languages: the
home screen; the launch panel with Korean labels and a Korean path typed into a field (glyphs
rendered, no boxes); the Results tab with humanised headers and a hover showing the raw metric
name; the text-size setting at `L`. The orchestrator repeats the check by hand. Section 15 of the
note records the string-table rule, the font rule (system fallback, nothing bundled, what a
machine without a CJK font sees) and the label tables.

## forbidden

Changing any view-model's data or any argv golden; a font file in the repository; a font-discovery
crate; `rfd` reachable from `app.rs` directly or on Linux by default; deciding a label in `app.rs`;
`crates/es/**`, `crates/es-telemetry/**`, `crates/es-eval/**`; `docs/ARCHITECTURE*.md`. INV-17:
no new trait.
