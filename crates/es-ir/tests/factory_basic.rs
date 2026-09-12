//! P29 oracle: `cargo test -p es-ir factory`.
//!
//! Exercises the registries purely through their public API — `schema()` supplies a default
//! for every parameter, so a round trip through `create()` doubles as both "every builtin kind
//! is constructible" and "schema ports match the constructed node's ports".

use es_ir::factory::{
    BuiltinTaskNodes, LearningNodeFactory, LearningNodeRegistry, NodeSchema, TaskNodeFactory,
    TaskNodeRegistry, BUILTIN_LEARNING_KINDS, BUILTIN_TASK_KINDS,
};
use es_ir::task::TaskNode;
use es_ir::{Diagnostic, IrNode};

fn params_from_schema(schema: &NodeSchema) -> toml::Value {
    let mut table = toml::Table::new();
    for param in &schema.params {
        table.insert(
            param.name.clone(),
            param
                .default
                .clone()
                .expect("builtin params carry a default"),
        );
    }
    toml::Value::Table(table)
}

#[test]
fn factory_creates_every_builtin_task_kind_from_its_own_schema() {
    let registry = TaskNodeRegistry::with_builtins();
    for &kind in BUILTIN_TASK_KINDS {
        let schema = registry
            .schema(kind)
            .unwrap_or_else(|| panic!("no schema for {kind}"));
        assert_eq!(schema.kind, kind);
        let node = registry
            .create(kind, &params_from_schema(&schema))
            .unwrap_or_else(|d| panic!("create({kind}) failed: {d}"));
        assert_eq!(node.kind(), kind);
        assert_eq!(node.inputs(), schema.inputs, "inputs for {kind}");
        assert_eq!(node.outputs(), schema.outputs, "outputs for {kind}");
    }
}

#[test]
fn factory_creates_every_builtin_learning_kind_from_its_own_schema() {
    let registry = LearningNodeRegistry::with_builtins();
    for &kind in BUILTIN_LEARNING_KINDS {
        let schema = registry
            .schema(kind)
            .unwrap_or_else(|| panic!("no schema for {kind}"));
        assert_eq!(schema.kind, kind);
        let node = registry
            .create(kind, &params_from_schema(&schema))
            .unwrap_or_else(|d| panic!("create({kind}) failed: {d}"));
        assert_eq!(node.kind(), kind);
        assert_eq!(node.inputs(), schema.inputs, "inputs for {kind}");
        assert_eq!(node.outputs(), schema.outputs, "outputs for {kind}");
    }
}

#[test]
fn factory_rejects_unknown_task_kind() {
    let registry = TaskNodeRegistry::with_builtins();
    let err = registry
        .create("NotAKind", &toml::Value::Table(toml::Table::new()))
        .expect_err("unknown kind must not build");
    assert_eq!(err.code.as_str(), "FACTORY-001");
    assert!(registry.schema("NotAKind").is_none());
}

#[test]
fn factory_rejects_unknown_learning_kind() {
    let registry = LearningNodeRegistry::with_builtins();
    let err = registry
        .create("NotAKind", &toml::Value::Table(toml::Table::new()))
        .expect_err("unknown kind must not build");
    assert_eq!(err.code.as_str(), "FACTORY-001");
}

/// A second factory that claims a kind `BuiltinTaskNodes` already owns.
struct OverlappingTaskFactory;

impl TaskNodeFactory for OverlappingTaskFactory {
    fn kinds(&self) -> &[&'static str] {
        &["GetJointState"]
    }

    fn create(&self, _kind: &str, _params: &toml::Value) -> Result<TaskNode, Diagnostic> {
        unreachable!("registration is rejected before this can be called")
    }

    fn schema(&self, _kind: &str) -> Option<NodeSchema> {
        None
    }
}

#[test]
fn factory_rejects_duplicate_kind_on_registration() {
    let mut registry = TaskNodeRegistry::with_builtins();
    let err = registry
        .register(Box::new(OverlappingTaskFactory))
        .expect_err("GetJointState is already owned by BuiltinTaskNodes");
    assert_eq!(err.code.as_str(), "FACTORY-002");
    // The original owner is untouched: it can still build the kind.
    let schema = registry.schema("GetJointState").unwrap();
    let node = registry
        .create("GetJointState", &params_from_schema(&schema))
        .unwrap();
    assert_eq!(node.kind(), "GetJointState");
}

#[test]
fn factory_builtin_kind_lists_are_exactly_the_ir_node_kinds() {
    let task_kinds = BuiltinTaskNodes.kinds();
    assert_eq!(task_kinds, BUILTIN_TASK_KINDS);
    let learning_kinds = BuiltinTaskNodes.kinds();
    assert_eq!(learning_kinds, BUILTIN_TASK_KINDS);
    assert_eq!(
        es_ir::factory::BuiltinLearningNodes.kinds(),
        BUILTIN_LEARNING_KINDS
    );
}
