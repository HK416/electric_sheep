//! Packet M8/S4d oracle 1: `Expr::Sqrt` is the IEEE basic operation, and adding it moved
//! nothing that was already written down.
//!
//!     cargo test -p es-ir-types expr_sqrt_is_ieee_and_refuses_negative
//!
//! Why `sqrt` at all, and why it is not a `DET-010` transcendental: spec 6.6. IEEE 754
//! requires a correctly rounded square root, so `f64::sqrt` is one instruction with the same
//! bits on every target -- unlike `exp` or `sin`, which need `es-math::approx`.

use std::collections::BTreeMap;

use es_ir_types::expr::{ArithOp, CmpOp, Expr};

/// The serialized form of an expression that predates this packet. `Expr` is serde-tagged by
/// variant, so a new variant must leave every old one byte-identical -- this is that pin, and
/// it is what a `Sqrt` arm inserted in the wrong place would break.
const PINNED: &str = r#"{"Clamp":{"value":{"Arith":{"op":"Sub","lhs":{"Port":"qpos[6]"},"rhs":{"Const":0.09}}},"lo":0.0,"hi":1.0}}"#;

fn ports() -> BTreeMap<String, f64> {
    [("qpos[6]".to_owned(), 0.25), ("d".to_owned(), 2.0)]
        .into_iter()
        .collect()
}

#[test]
fn expr_sqrt_is_ieee_and_refuses_negative() {
    let ports = ports();

    // Exactly `f64::sqrt`, not an approximation of it: bit equality is the contract.
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(
            Expr::Sqrt(Box::new(Expr::Const(2.0))).eval(&ports),
            Some(2f64.sqrt())
        );
        assert_eq!(
            Expr::Sqrt(Box::new(Expr::Port("d".to_owned()))).eval(&ports),
            Some(2f64.sqrt())
        );
        assert_eq!(
            Expr::Sqrt(Box::new(Expr::Const(0.0))).eval(&ports),
            Some(0.0)
        );
    }

    // A negative radicand is `None` like every other non-finite result, never a `NaN`
    // (`Expr::eval`): a `NaN` is what no hash and no comparison can encode.
    assert_eq!(Expr::Sqrt(Box::new(Expr::Const(-1.0))).eval(&ports), None);
    assert_eq!(
        Expr::Sqrt(Box::new(Expr::Const(-f64::MIN_POSITIVE))).eval(&ports),
        None
    );
    // And a `None` under it stays `None` rather than becoming a zero.
    assert_eq!(
        Expr::Sqrt(Box::new(Expr::Port("nope".to_owned()))).eval(&ports),
        None
    );

    // A `Sqrt`-free expression serializes to exactly what it did before the variant existed.
    let old = Expr::Clamp {
        value: Box::new(Expr::Arith {
            op: ArithOp::Sub,
            lhs: Box::new(Expr::Port("qpos[6]".to_owned())),
            rhs: Box::new(Expr::Const(0.09)),
        }),
        lo: 0.0,
        hi: 1.0,
    };
    assert_eq!(
        serde_json::to_string(&old).expect("Expr serializes"),
        PINNED
    );
    assert_eq!(
        serde_json::from_str::<Expr>(PINNED).expect("Expr round-trips"),
        old
    );

    // The new variant is a peer of the old ones, not a special case: it composes.
    let hypot = Expr::Sqrt(Box::new(Expr::Arith {
        op: ArithOp::Add,
        lhs: Box::new(Expr::Const(9.0)),
        rhs: Box::new(Expr::Const(16.0)),
    }));
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(hypot.eval(&ports), Some(5.0));
    }
    assert_eq!(
        Expr::Compare {
            op: CmpOp::Lt,
            lhs: Box::new(hypot),
            rhs: Box::new(Expr::Const(6.0)),
        }
        .eval(&ports),
        Some(1.0)
    );
    println!("RAN expr_sqrt_is_ieee_and_refuses_negative");
}
