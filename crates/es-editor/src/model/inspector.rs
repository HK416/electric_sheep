//! The parameter inspector of spec 23.4 stage 2, as a headless model.
//!
//! `Edit::SetParam` and [`NodeSchema`] have existed since M4 with no widget behind them: a
//! node's parameters could only be changed by editing TOML. This is the half of that widget
//! that *decides* something — which control a [`ParamType`] gets, what a person's text means,
//! and when text becomes an [`Edit`] — so `crate::app` is left with the drawing (spec 28.10
//! rule 3).
//!
//! Three rules shape it:
//!
//! * **One widget per [`ParamType`], and the match is exhaustive.** [`widget_for`] has no
//!   wildcard arm, so a new `ParamType` in `es-ir` breaks this crate's compile rather than
//!   silently rendering as a text box.
//! * **[`Field::parse`] is the only place text becomes a value.** Every other function here
//!   either formats a value into text or hands a parsed value to [`Edit::SetParam`].
//! * **An erroneous field never emits an edit.** [`Inspector::edit`] records the parse error
//!   on the field and returns `None`; the session is not touched.
//!
//! There is no per-kind code here, exactly as in [`crate::model::edit`]: the fields come from
//! the registry's [`NodeSchema`] and the node's own serialization, so a node kind added to
//! `es-ir` is inspectable with no change below this line (`docs/design/node-sdk.md`).

use es_ir::factory::{NodeSchema, ParamType};
use es_ir::graph::NodeId;
use es_ir::IrNode;
use serde::Serialize;

use crate::model::edit::{Edit, EditIr, EditSession};

/// The control one parameter is edited with. One variant per [`ParamType`] — the mapping is a
/// bijection so that [`widget_for`]'s exhaustive match is worth having.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Widget {
    /// [`ParamType::Bool`].
    Checkbox,
    /// [`ParamType::Int`], dragged as an `i64`.
    DragInt,
    /// [`ParamType::Float`], dragged as an `f64`.
    DragFloat,
    /// [`ParamType::String`] — which is also `es-ir`'s catch-all for a value it cannot infer
    /// a narrower type for (an array of floats, a datetime). See [`Field::parse`].
    Text,
    /// [`ParamType::Enum`], as the variants the schema knows plus the one in use.
    Combo(Vec<String>),
    /// [`ParamType::Shape`], typed as `[a, b, ...]`.
    ShapeText,
    /// [`ParamType::PortType`], typed as the inline TOML it prints as.
    TomlText,
}

impl Widget {
    /// What the empty field tells a person to type. A hint is a decision about what someone is
    /// shown, so it lives here and not in `app.rs`.
    pub fn hint(&self) -> &'static str {
        match self {
            Self::Checkbox | Self::DragInt | Self::DragFloat | Self::Combo(_) => "",
            Self::Text => "text",
            Self::ShapeText => "[a, b, ...]",
            Self::TomlText => "{ elem = ..., shape = [...] }",
        }
    }
}

/// The one widget a [`ParamType`] is edited with.
///
/// **No wildcard arm, deliberately**: adding a variant to [`ParamType`] must fail this crate's
/// build, because the alternative is a parameter that silently cannot be edited.
pub fn widget_for(ty: &ParamType) -> Widget {
    match ty {
        ParamType::Bool => Widget::Checkbox,
        ParamType::Int => Widget::DragInt,
        ParamType::Float => Widget::DragFloat,
        ParamType::String => Widget::Text,
        ParamType::Enum(variants) => Widget::Combo(variants.clone()),
        ParamType::Shape => Widget::ShapeText,
        ParamType::PortType => Widget::TomlText,
    }
}

/// One editable parameter: what the schema says it is, what it currently holds, what the
/// person has typed, and why that does not parse.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub name: String,
    /// The schema's type, with one adjustment: an [`ParamType::Enum`] whose variant list does
    /// not contain the value actually in use gains it, so the combo cannot hide the truth.
    pub ty: ParamType,
    pub required: bool,
    /// What is in the box. Seeded from [`Field::current`] and owned by the widget afterwards.
    pub text: String,
    /// Set by [`Inspector::edit`] when [`Field::parse`] refused the text.
    pub error: Option<String>,
    current: toml::Value,
}

impl Field {
    /// The value in the IR right now. `text` is this, formatted.
    pub fn current(&self) -> &toml::Value {
        &self.current
    }

    pub fn widget(&self) -> Widget {
        widget_for(&self.ty)
    }

    /// Text → value. The only such conversion in the editor.
    ///
    /// `Bool` and `Enum` cannot fail from the UI — a checkbox only ever writes `true`/`false`
    /// and a combo only ever writes a variant it was given — but they are fallible here rather
    /// than panicking on text that arrived some other way.
    pub fn parse(&self) -> Result<toml::Value, String> {
        let text = self.text.trim();
        match &self.ty {
            ParamType::Bool => text
                .parse::<bool>()
                .map(toml::Value::Boolean)
                .map_err(|_| format!("expected true or false, got `{text}`")),
            ParamType::Int => text
                .parse::<i64>()
                .map(toml::Value::Integer)
                .map_err(|e| format!("expected a whole number: {e}")),
            ParamType::Float => text
                .parse::<f64>()
                .map(toml::Value::Float)
                .map_err(|e| format!("expected a number: {e}")),
            // `String` is `es-ir`'s catch-all (`infer_param_type`): a real string is edited as
            // itself, and anything it could not name — an array of floats, a datetime — is
            // edited as the TOML it printed as.
            ParamType::String => match &self.current {
                toml::Value::String(_) => Ok(toml::Value::String(self.text.clone())),
                _ => parse_toml(text),
            },
            ParamType::Enum(_) => Ok(self.variant(text)),
            ParamType::Shape => match parse_toml(text)? {
                toml::Value::Array(items)
                    if items.iter().all(|v| matches!(v, toml::Value::Integer(_))) =>
                {
                    Ok(toml::Value::Array(items))
                }
                other => Err(format!(
                    "expected [a, b, ...] of whole numbers, got {other}"
                )),
            },
            ParamType::PortType => match parse_toml(text)? {
                table @ toml::Value::Table(_) => Ok(table),
                other => Err(format!("expected an inline TOML table, got {other}")),
            },
        }
    }

    /// The selected variant as a value: the current one keeps its payload, and any other is
    /// offered to the factory empty — which is `FACTORY-003` if it needs fields, reported by
    /// [`EditSession::apply`] like any other refused edit.
    fn variant(&self, name: &str) -> toml::Value {
        if let toml::Value::Table(table) = &self.current {
            if table.contains_key(name) {
                return self.current.clone();
            }
        }
        let mut table = toml::Table::new();
        table.insert(name.to_owned(), toml::Value::Table(toml::Table::new()));
        toml::Value::Table(table)
    }

    /// What a drag widget shows, and what it writes back. The formatting is here because what
    /// a number looks like as text is what [`Field::parse`] has to read again.
    pub fn number(&self) -> f64 {
        self.text.trim().parse().unwrap_or(0.0)
    }

    pub fn set_number(&mut self, value: f64) {
        self.text = if self.ty == ParamType::Int {
            format!("{}", value as i64)
        } else {
            format!("{value}")
        };
    }

    /// The same pair for a checkbox.
    pub fn flag(&self) -> bool {
        self.text.trim() == "true"
    }

    pub fn set_flag(&mut self, value: bool) {
        self.text = value.to_string();
    }
}

/// Every parameter of one node, ready to be drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Inspector {
    pub node: NodeId,
    pub kind: &'static str,
    fields: Vec<Field>,
}

impl Inspector {
    /// The node's schema from the session's registries, filled in with the node's own current
    /// parameters. `None` for a node that is not there, or whose kind no registry owns — an
    /// Observation IR node, which has no factory at all (`INV-17`).
    pub fn for_node(session: &EditSession, node: NodeId) -> Option<Self> {
        let (kind, params) = params_of(&session.graph, node)?;
        let schema = match session.graph {
            EditIr::Task(_) => session.registries.task.schema(kind),
            EditIr::Learning(_) => session.registries.learning.schema(kind),
            EditIr::Observation(_) => None,
        }?;
        Some(Self::of(node, kind, &schema, &params))
    }

    /// The fields are the schema's, **restricted to the keys the node actually serialized**: a
    /// `None` in an `Option` field is not in the table, and `Edit::SetParam` refuses a key the
    /// node does not have (`FACTORY-003`), so drawing it would be drawing a dead control.
    fn of(node: NodeId, kind: &'static str, schema: &NodeSchema, params: &toml::Table) -> Self {
        let fields = schema
            .params
            .iter()
            .filter_map(|p| {
                let current = params.get(&p.name)?;
                let ty = widen_enum(&p.ty, current);
                Some(Field {
                    name: p.name.clone(),
                    text: text_of(&ty, current),
                    ty,
                    required: p.required,
                    error: None,
                    current: current.clone(),
                })
            })
            .collect();
        Self { node, kind, fields }
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// The widgets write straight into the fields they are bound to.
    pub fn fields_mut(&mut self) -> &mut [Field] {
        &mut self.fields
    }

    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// `text` for `name`, parsed. `Some` is the edit to apply; `None` means the text did not
    /// parse and the reason is now on the field ([`Field::error`]).
    pub fn edit(&mut self, name: &str, text: &str) -> Option<Edit> {
        let node = self.node;
        let field = self.fields.iter_mut().find(|f| f.name == name)?;
        text.clone_into(&mut field.text);
        match field.parse() {
            Ok(value) => {
                field.error = None;
                Some(Edit::SetParam {
                    node,
                    key: name.to_owned(),
                    value,
                })
            }
            Err(e) => {
                field.error = Some(e);
                None
            }
        }
    }
}

/// A one-key table whose key the schema's variant list is missing — a value the example
/// instance the schema was derived from never showed. Adding it keeps the combo honest.
fn widen_enum(ty: &ParamType, current: &toml::Value) -> ParamType {
    let (ParamType::Enum(variants), toml::Value::Table(table)) = (ty, current) else {
        return ty.clone();
    };
    let mut variants = variants.clone();
    for key in table.keys() {
        if !variants.contains(key) {
            variants.push(key.clone());
        }
    }
    ParamType::Enum(variants)
}

/// Value → text, the inverse of [`Field::parse`]. An enum shows its variant, a string shows
/// itself unquoted, and everything else shows the inline TOML it serializes as.
fn text_of(ty: &ParamType, value: &toml::Value) -> String {
    match (ty, value) {
        (ParamType::Enum(_), toml::Value::Table(table)) => {
            table.keys().next().cloned().unwrap_or_default()
        }
        (_, toml::Value::String(s)) => s.clone(),
        _ => value.to_string(),
    }
}

/// One TOML value, parsed by making it a one-key document — `toml`'s `FromStr` reads a
/// document, not a bare value. The error is its first line: the rest is a caret diagram that
/// has no room in a side panel.
fn parse_toml(text: &str) -> Result<toml::Value, String> {
    let mut table: toml::Table = toml::from_str(&format!("v = {text}")).map_err(|e| {
        e.to_string()
            .lines()
            .next()
            .unwrap_or("not valid TOML")
            .to_owned()
    })?;
    table
        .remove("v")
        .ok_or_else(|| "expected a value".to_owned())
}

/// `(kind, the node's own parameter table)` — the same re-serialization `Edit::SetParam` does
/// before handing the table back to the factory, so what the inspector shows and what a
/// `SetParam` overwrites are the same table.
fn params_of(graph: &EditIr, node: NodeId) -> Option<(&'static str, toml::Table)> {
    match graph {
        EditIr::Task(ir) => ir.graph.nodes.get(&node).map(tagged),
        EditIr::Learning(ir) => ir.nodes.nodes.get(&node).map(tagged),
        EditIr::Observation(ir) => ir.graph.nodes.get(&node).map(tagged),
    }
}

fn tagged<N: IrNode + Serialize>(node: &N) -> (&'static str, toml::Table) {
    let kind = node.kind();
    let params = match toml::Value::try_from(node) {
        Ok(toml::Value::Table(mut outer)) => match outer.remove(kind) {
            Some(toml::Value::Table(params)) => params,
            _ => toml::Table::new(),
        },
        _ => toml::Table::new(),
    };
    (kind, params)
}

#[cfg(test)]
mod tests {
    use super::{widget_for, Inspector, Widget};

    use std::collections::BTreeSet;
    use std::path::Path;

    use es_ir::factory::ParamType;
    use es_ir::graph::{Graph, NodeId};
    use es_ir::serial::Layout;
    use es_ir::task::{ObservationSpec, SceneRef, TaskConfig, TaskIr};

    use crate::model::edit::{Edit, EditIr, EditSession};
    use crate::model::palette::Palette;

    fn task_session() -> EditSession {
        EditSession::new(
            EditIr::Task(TaskIr {
                schema_version: 1,
                scene: SceneRef {
                    path: "scenes/table.xml".to_owned(),
                    scene_hash: [0; 32],
                    asset_hash: [0; 32],
                },
                graph: Graph::new(1),
                observation_spec: ObservationSpec::default(),
                control: None,
                config: TaskConfig {
                    max_episode_steps: 200,
                    control_rate_hz: 30.0,
                    deterministic: true,
                    rng_streams: BTreeSet::new(),
                },
            }),
            Layout::default(),
        )
    }

    /// The demo bundle's Learning IR, so that the nine Learning kinds can be added to a real
    /// graph without this test hand-building a `PolicyHandle`.
    fn learning_session() -> EditSession {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .join("tests/fixtures/visible-learning/learning.toml");
        let toml = std::fs::read_to_string(&path).expect("the demo bundle's learning.toml");
        let ir = es_ir::serial::learning_from_toml(&toml).expect("it parses");
        EditSession::new(EditIr::Learning(ir), Layout::default())
    }

    #[track_caller]
    fn add(session: &mut EditSession, kind: &str) -> NodeId {
        let params = Palette::from_registries(&session.registries)
            .defaults(kind)
            .expect("the kind is in the palette");
        let id = session.graph.next_id();
        session
            .apply(Edit::AddNode {
                kind: kind.to_owned(),
                params,
                pos: [0.0, 0.0],
            })
            .unwrap_or_else(|e| panic!("{kind} is added: {e:?}"));
        id
    }

    /// Oracle 1: every `ParamType` gets exactly one widget, and no two share it. The match in
    /// [`widget_for`] has no wildcard arm, so a new variant is a compile error here rather
    /// than a parameter that quietly cannot be edited.
    #[test]
    fn every_param_type_has_one_widget() {
        let table = [
            (ParamType::Bool, Widget::Checkbox),
            (ParamType::Int, Widget::DragInt),
            (ParamType::Float, Widget::DragFloat),
            (ParamType::String, Widget::Text),
            (
                ParamType::Enum(vec!["A".to_owned(), "B".to_owned()]),
                Widget::Combo(vec!["A".to_owned(), "B".to_owned()]),
            ),
            (ParamType::Shape, Widget::ShapeText),
            (ParamType::PortType, Widget::TomlText),
        ];
        for (ty, expected) in &table {
            assert_eq!(&widget_for(ty), expected, "{ty:?}");
            assert_eq!(widget_for(ty), widget_for(ty), "{ty:?} is deterministic");
        }
        for (i, (_, a)) in table.iter().enumerate() {
            for (_, b) in table.iter().skip(i + 1) {
                assert_ne!(a, b, "two ParamTypes share one widget");
            }
        }
    }

    /// Oracle 2: for every kind both registries expose, re-entering each field's own text
    /// emits a `SetParam` carrying the value that is already there — the inspector's
    /// round-trip, which is what makes it safe to open a node and close it again.
    #[test]
    fn set_param_round_trips_through_the_inspector() {
        for (mut session, ir) in [
            (task_session(), es_ir::serial::IrKind::Task),
            (learning_session(), es_ir::serial::IrKind::Learning),
        ] {
            let kinds: Vec<String> = Palette::from_registries(&session.registries)
                .for_ir(ir)
                .map(|e| e.kind.clone())
                .collect();
            assert!(!kinds.is_empty());
            for kind in kinds {
                let id = add(&mut session, &kind);
                let mut inspector =
                    Inspector::for_node(&session, id).unwrap_or_else(|| panic!("{kind}"));
                assert_eq!(inspector.kind, kind);
                let names: Vec<(String, String)> = inspector
                    .fields()
                    .iter()
                    .map(|f| (f.name.clone(), f.text.clone()))
                    .collect();
                for (name, text) in names {
                    let current = inspector
                        .field(&name)
                        .expect("the field is there")
                        .current()
                        .clone();
                    let edit = inspector
                        .edit(&name, &text)
                        .unwrap_or_else(|| panic!("{kind}.{name} = `{text}` parses"));
                    let Edit::SetParam { node, key, value } = edit else {
                        panic!("{kind}.{name} emitted {edit:?}");
                    };
                    assert_eq!(node, id);
                    assert_eq!(key, name);
                    assert_eq!(
                        value, current,
                        "{kind}.{name}: `{text}` is not the value it was printed from"
                    );
                }
            }
        }
    }

    /// The other half of oracle 2: a changed field is accepted by the session, and a
    /// parameter the hash covers moves it.
    #[test]
    fn a_changed_field_is_applied_and_moves_the_hash() {
        let mut session = task_session();
        let id = add(&mut session, "Normalize");
        let before = session.hash().expect("the empty task hashes");
        let mut inspector = Inspector::for_node(&session, id).expect("Normalize is inspectable");
        let edit = inspector.edit("out_hi", "2.5").expect("2.5 is a float");
        assert_eq!(
            edit,
            Edit::SetParam {
                node: id,
                key: "out_hi".to_owned(),
                value: toml::Value::Float(2.5),
            }
        );
        session.apply(edit).expect("the session accepts it");
        assert_ne!(
            before,
            session.hash().expect("it still hashes"),
            "out_hi is a hashed parameter"
        );
        let after = Inspector::for_node(&session, id).expect("still inspectable");
        assert_eq!(after.field("out_hi").expect("out_hi").text, "2.5");
    }

    /// Oracle 3: a field that does not parse records why and emits nothing, and the session
    /// never hears about it.
    #[test]
    fn a_bad_field_never_emits_an_edit() {
        let mut session = task_session();
        let slice = add(&mut session, "Slice");
        let logic = add(&mut session, "Logic");
        let before = session.hash().expect("hashes");

        let mut inspector = Inspector::for_node(&session, slice).expect("Slice is inspectable");
        assert_eq!(inspector.edit("axis", "abc"), None);
        let axis = inspector.field("axis").expect("axis");
        assert_eq!(axis.widget(), Widget::DragInt);
        assert!(axis.error.is_some(), "the reason is on the field");

        let mut inspector = Inspector::for_node(&session, logic).expect("Logic is inspectable");
        assert_eq!(
            inspector.field("shape").expect("shape").widget(),
            Widget::ShapeText
        );
        assert_eq!(inspector.edit("shape", "[1,"), None);
        assert!(inspector.field("shape").expect("shape").error.is_some());
        // A shape of floats is not a shape.
        assert_eq!(inspector.edit("shape", "[1.5]"), None);

        assert_eq!(
            before,
            session.hash().expect("hashes"),
            "nothing reached the session"
        );
    }

    /// A good value after a bad one clears the complaint.
    #[test]
    fn a_repaired_field_stops_complaining() {
        let mut session = task_session();
        let id = add(&mut session, "Slice");
        let mut inspector = Inspector::for_node(&session, id).expect("inspectable");
        assert_eq!(inspector.edit("len", "x"), None);
        assert!(inspector.edit("len", "3").is_some());
        assert_eq!(inspector.field("len").expect("len").error, None);
    }

    /// An Observation IR node has no factory (`INV-17`), so it has no schema and no inspector
    /// — the same boundary `Edit::SetParam` reports as `FACTORY-001`.
    #[test]
    fn an_observation_node_has_no_inspector() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .join("tests/fixtures/visible-learning/observation.toml");
        let toml = std::fs::read_to_string(&path).expect("the demo bundle's observation.toml");
        let ir = es_ir::serial::observation_from_toml(&toml).expect("it parses");
        let first = *ir.graph.nodes.keys().next().expect("it has nodes");
        let session = EditSession::new(EditIr::Observation(ir), Layout::default());
        assert_eq!(Inspector::for_node(&session, first), None);
    }

    /// A node id that is not in the graph.
    #[test]
    fn a_missing_node_has_no_inspector() {
        let session = task_session();
        assert_eq!(Inspector::for_node(&session, NodeId(77)), None);
    }
}
