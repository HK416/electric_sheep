//! Packet M15/N6: the Python builder's `shared=True` writes spec 8.3's `share` (spec 14.2 as
//! amended by M15/N5) -- the first shared encoder is the group's owner, every later one names
//! it, and `shared=False` makes an encoder of its own.
//!
//! `python/es/builder.py` is run with its extension module stood in by a recorder of the
//! `add` calls it makes, so the test needs `ES_PYTHON` (with `blake3`, which the builder
//! imports) but not a maturin build. What the builder wrote is then handed to the same
//! `LearningNodeRegistry` `es_native`'s `add` uses, so the `share` it writes is the one the IR
//! reads. SKIPs, loudly, without `ES_PYTHON`.

use std::path::Path;
use std::process::Command;

use es_ir::factory::LearningNodeRegistry;
use es_ir::graph::NodeId;
use es_ir::learning::LearningNode;
use es_native::builder::json_to_toml;

/// `argv[1]` is `python/es`. Prints `{"ids": [..], "calls": [[kind, params], ..]}`.
const RECORD: &str = r#"
import importlib.util, json, sys, types
root, calls = sys.argv[1], []
class Learning:
    def add(self, kind, params):
        calls.append([kind, json.loads(params)])
        return len(calls) - 1
native = types.ModuleType("es.es_native")
native.Learning, native.EsDiagnosticError = Learning, type("EsDiagnosticError", (Exception,), {})
package = types.ModuleType("es")
package.__path__, package.es_native = [root], native
sys.modules["es"], sys.modules["es.es_native"] = package, native
spec = importlib.util.spec_from_file_location("es.builder", root + "/builder.py")
builder = importlib.util.module_from_spec(spec)
sys.modules["es.builder"] = builder
spec.loader.exec_module(builder)
lrn = builder.Learning("three_views")
ids = [lrn.vision_encoder("resnet18", out_dim=64) for _ in range(3)]
ids.append(lrn.vision_encoder("resnet18", out_dim=64, shared=False))
print(json.dumps({"ids": ids, "calls": calls}))
"#;

#[test]
fn shared_true_writes_share_on_every_later_view() {
    let Ok(python) = std::env::var("ES_PYTHON") else {
        println!("SKIP shared_true_writes_share_on_every_later_view: ES_PYTHON is unset");
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/es");
    let out = Command::new(&python)
        .args(["-c", RECORD, &root.to_string_lossy()])
        .output()
        .unwrap_or_else(|e| panic!("run {python}: {e}"));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let reply: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("one JSON object on stdout");
    let ids: Vec<u64> = reply["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("an id"))
        .collect();
    let calls = reply["calls"].as_array().expect("calls");
    assert_eq!(calls.len(), 4, "{reply}");

    let registry = LearningNodeRegistry::with_builtins();
    let shares: Vec<Option<NodeId>> = calls
        .iter()
        .map(|call| {
            assert_eq!(call[0], "VisionEncoder");
            let params = json_to_toml(&call[1]).expect("JSON params become TOML");
            match registry.create("VisionEncoder", &params) {
                Ok(LearningNode::VisionEncoder { share, .. }) => share,
                other => panic!("{other:?}"),
            }
        })
        .collect();
    let owner = NodeId(u32::try_from(ids[0]).expect("small"));
    assert_eq!(shares, [None, Some(owner), Some(owner), None], "{reply}");
    // Absent, not `null`: an unshared encoder's params are the ones written before the packet.
    assert!(calls[0][1].get("share").is_none(), "{reply}");
    assert!(calls[3][1].get("share").is_none(), "{reply}");
    println!("RAN shared_true_writes_share_on_every_later_view: {shares:?}");
}
