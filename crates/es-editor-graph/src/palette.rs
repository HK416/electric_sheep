//! The node palette: every kind the registries can build, with its [`NodeSchema`].
//!
//! This is the editor's whole view of "what nodes exist" (spec 23.4 stage 2, spec 6.3, spec
//! 8.3). It is derived from `TaskNodeRegistry` / `LearningNodeRegistry` — the two node
//! factories `INV-17` allows — so a new node kind appears in the add-node menu with correct
//! ports, parameters and starting values without one line changing here.
//!
//! [`Palette::to_json`] exports the same schema for spec 14.5: an LLM generating a Task or
//! Learning IR reads the palette rather than a prose list of kinds, so the generator and the
//! editor cannot disagree about what a node takes.

use std::collections::{BTreeMap, BTreeSet};

use es_ir::factory::{
    LearningNodeFactory, LearningNodeRegistry, NodeSchema, ParamType, TaskNodeFactory,
    TaskNodeRegistry, BUILTIN_LEARNING_KINDS, BUILTIN_TASK_KINDS,
};
use es_ir::serial::IrKind;
use es_ir::Diagnostic;

/// The two node registries, plus the kinds registered in each.
///
/// The kind list is tracked here because a registry resolves a kind but does not enumerate
/// one; `TaskNodeFactory::kinds` gives the list at registration time, which is the only moment
/// it is needed.
#[derive(Debug)]
pub struct Registries {
    pub task: TaskNodeRegistry,
    pub learning: LearningNodeRegistry,
    task_kinds: BTreeSet<String>,
    learning_kinds: BTreeSet<String>,
}

impl Default for Registries {
    fn default() -> Self {
        Self::with_builtins()
    }
}

impl Registries {
    /// The spec 6.3 and spec 8.3 node sets.
    pub fn with_builtins() -> Self {
        Self {
            task: TaskNodeRegistry::with_builtins(),
            learning: LearningNodeRegistry::with_builtins(),
            task_kinds: BUILTIN_TASK_KINDS.iter().map(|k| (*k).to_owned()).collect(),
            learning_kinds: BUILTIN_LEARNING_KINDS
                .iter()
                .map(|k| (*k).to_owned())
                .collect(),
        }
    }

    /// Adds a third-party Task node factory (`docs/design/node-sdk.md`). Fails with
    /// `FACTORY-002` if it claims a kind that is already owned; the registries are left
    /// unchanged in that case.
    pub fn register_task(&mut self, factory: Box<dyn TaskNodeFactory>) -> Result<(), Diagnostic> {
        let kinds: Vec<String> = factory.kinds().iter().map(|k| (*k).to_owned()).collect();
        self.task.register(factory)?;
        self.task_kinds.extend(kinds);
        Ok(())
    }

    /// Adds a third-party Learning node factory. Same contract as [`Self::register_task`].
    pub fn register_learning(
        &mut self,
        factory: Box<dyn LearningNodeFactory>,
    ) -> Result<(), Diagnostic> {
        let kinds: Vec<String> = factory.kinds().iter().map(|k| (*k).to_owned()).collect();
        self.learning.register(factory)?;
        self.learning_kinds.extend(kinds);
        Ok(())
    }

    pub fn task_kinds(&self) -> impl Iterator<Item = &str> {
        self.task_kinds.iter().map(String::as_str)
    }

    pub fn learning_kinds(&self) -> impl Iterator<Item = &str> {
        self.learning_kinds.iter().map(String::as_str)
    }
}

/// One kind, ready to be offered in a menu or read by a generator.
#[derive(Clone, Debug, PartialEq)]
pub struct PaletteEntry {
    /// Which IR the kind belongs to; a Task kind cannot be added to a Learning graph.
    pub ir: IrKind,
    pub kind: String,
    /// Menu grouping. Derived from the kind, not stored in the IR.
    pub category: &'static str,
    pub schema: NodeSchema,
}

impl PaletteEntry {
    /// A parameter table that type-checks, from the schema's own starting values. This is what
    /// `Edit::AddNode` is given before the user has touched anything.
    pub fn defaults(&self) -> toml::Value {
        let mut table = toml::Table::new();
        for param in &self.schema.params {
            if let Some(value) = &param.default {
                table.insert(param.name.clone(), value.clone());
            }
        }
        toml::Value::Table(table)
    }
}

/// Every kind both registries can build.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Palette {
    pub entries: Vec<PaletteEntry>,
}

impl Palette {
    pub fn from_registries(reg: &Registries) -> Self {
        let mut entries = Vec::new();
        for kind in reg.task_kinds() {
            if let Some(schema) = reg.task.schema(kind) {
                entries.push(PaletteEntry {
                    ir: IrKind::Task,
                    kind: kind.to_owned(),
                    category: task_category(kind),
                    schema,
                });
            }
        }
        for kind in reg.learning_kinds() {
            if let Some(schema) = reg.learning.schema(kind) {
                entries.push(PaletteEntry {
                    ir: IrKind::Learning,
                    kind: kind.to_owned(),
                    category: learning_category(kind),
                    schema,
                });
            }
        }
        Self { entries }
    }

    pub fn get(&self, kind: &str) -> Option<&PaletteEntry> {
        self.entries.iter().find(|e| e.kind == kind)
    }

    /// [`PaletteEntry::defaults`] for one kind.
    pub fn defaults(&self, kind: &str) -> Option<toml::Value> {
        Some(self.get(kind)?.defaults())
    }

    /// The add-node menu: categories in a fixed order, kinds in a fixed order inside each.
    pub fn by_category(&self) -> BTreeMap<&'static str, Vec<&PaletteEntry>> {
        let mut out: BTreeMap<&'static str, Vec<&PaletteEntry>> = BTreeMap::new();
        for entry in &self.entries {
            out.entry(entry.category).or_default().push(entry);
        }
        out
    }

    /// Only the kinds that belong in `ir`'s graph.
    pub fn for_ir(&self, ir: IrKind) -> impl Iterator<Item = &PaletteEntry> {
        self.entries.iter().filter(move |e| e.ir == ir)
    }

    /// The palette as JSON, for the spec 14.5 generator path: the same ports, parameter types
    /// and starting values the editor draws.
    pub fn to_json(&self) -> String {
        let entries: Vec<serde_json::Value> = self.entries.iter().map(entry_json).collect();
        serde_json::to_string_pretty(&serde_json::json!({
            "es_schema": es_ir::serial::ES_SCHEMA_VERSION,
            "entries": entries,
        }))
        .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }
}

fn entry_json(entry: &PaletteEntry) -> serde_json::Value {
    serde_json::json!({
        "ir": format!("{:?}", entry.ir).to_lowercase(),
        "kind": entry.kind,
        "category": entry.category,
        "inputs": ports_json(&entry.schema.inputs),
        "outputs": ports_json(&entry.schema.outputs),
        "params": entry.schema.params.iter().map(|p| serde_json::json!({
            "name": p.name,
            "type": param_type_json(&p.ty),
            "required": p.required,
            "default": p.default.as_ref().and_then(|v| serde_json::to_value(v).ok()),
        })).collect::<Vec<_>>(),
    })
}

fn ports_json(ports: &[es_ir::Port]) -> Vec<serde_json::Value> {
    ports
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.name,
                "ty": serde_json::to_value(&p.ty).unwrap_or(serde_json::Value::Null),
            })
        })
        .collect()
}

fn param_type_json(ty: &ParamType) -> serde_json::Value {
    match ty {
        ParamType::Bool => serde_json::json!("bool"),
        ParamType::Int => serde_json::json!("int"),
        ParamType::Float => serde_json::json!("float"),
        ParamType::String => serde_json::json!("string"),
        ParamType::Shape => serde_json::json!("shape"),
        ParamType::PortType => serde_json::json!("port_type"),
        ParamType::Enum(variants) => serde_json::json!({ "enum": variants }),
    }
}

/// Spec 6.3's own grouping of the Task node set: sources, dataflow operators, and the
/// declarations that make a graph a *task*. A kind no builtin list knows is third-party.
fn task_category(kind: &str) -> &'static str {
    if !BUILTIN_TASK_KINDS.contains(&kind) {
        "Task / custom"
    } else if kind.starts_with("Get") {
        "Task / sources"
    } else if matches!(
        kind,
        "ObservationSpec"
            | "ActionSpec"
            | "Reward"
            | "Terminate"
            | "Randomization"
            | "ResetState"
            | "Record"
    ) {
        "Task / declarations"
    } else {
        "Task / ops"
    }
}

/// Spec 8.3's stages: encoders, fusion and temporal, head, action post-processing.
fn learning_category(kind: &str) -> &'static str {
    match kind {
        "VisionEncoder" | "StateEncoder" | "LanguageEncoder" => "Learning / encoders",
        "Fusion" | "TemporalEncoder" => "Learning / fusion",
        "PolicyHead" | "PolicyBundle" => "Learning / heads",
        "ActionChunker" | "Normalizer" => "Learning / action",
        _ => "Learning / custom",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn palette() -> Palette {
        Palette::from_registries(&Registries::with_builtins())
    }

    #[test]
    fn every_frozen_builtin_kind_is_in_the_palette() {
        let p = palette();
        for kind in BUILTIN_TASK_KINDS {
            let entry = p.get(kind).unwrap_or_else(|| panic!("{kind} is listed"));
            assert_eq!(entry.ir, IrKind::Task);
            assert_eq!(entry.schema.kind, *kind);
        }
        for kind in BUILTIN_LEARNING_KINDS {
            let entry = p.get(kind).unwrap_or_else(|| panic!("{kind} is listed"));
            assert_eq!(entry.ir, IrKind::Learning);
        }
        assert_eq!(
            p.entries.len(),
            BUILTIN_TASK_KINDS.len() + BUILTIN_LEARNING_KINDS.len(),
            "the palette is exactly the two frozen kind lists"
        );
    }

    #[test]
    fn the_defaults_of_every_kind_build_a_node() {
        let reg = Registries::with_builtins();
        let p = Palette::from_registries(&reg);
        for entry in p.for_ir(IrKind::Task) {
            reg.task
                .create(&entry.kind, &entry.defaults())
                .unwrap_or_else(|e| panic!("{} builds from its defaults: {e}", entry.kind));
        }
        for entry in p.for_ir(IrKind::Learning) {
            reg.learning
                .create(&entry.kind, &entry.defaults())
                .unwrap_or_else(|e| panic!("{} builds from its defaults: {e}", entry.kind));
        }
    }

    #[test]
    fn categories_cover_every_entry_and_are_ordered() {
        let p = palette();
        let by_category = p.by_category();
        assert_eq!(
            by_category.values().map(Vec::len).sum::<usize>(),
            p.entries.len()
        );
        assert!(by_category.contains_key("Task / sources"));
        assert!(by_category.contains_key("Learning / heads"));
        assert!(!by_category.contains_key("Task / custom"));
    }

    #[test]
    fn json_carries_the_schema_a_generator_needs() {
        let json = palette().to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let entries = parsed["entries"].as_array().expect("an array");
        assert_eq!(
            entries.len(),
            BUILTIN_TASK_KINDS.len() + BUILTIN_LEARNING_KINDS.len()
        );
        let reward = entries
            .iter()
            .find(|e| e["kind"] == "Reward")
            .expect("Reward is exported");
        assert_eq!(reward["ir"], "task");
        assert_eq!(reward["inputs"][0]["name"], "value");
        assert!(
            reward["params"]
                .as_array()
                .expect("params")
                .iter()
                .any(|p| p["name"] == "weight"),
            "{reward}"
        );
    }

    #[test]
    fn the_palette_is_deterministic() {
        assert_eq!(palette(), palette());
        assert_eq!(palette().to_json(), palette().to_json());
    }
}
