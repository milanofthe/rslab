//! `OrderingPreprocess::Auto` fill verification (feral #91/#92 port).
//!
//! `Auto` must resolve the structural `pick_ordering_preprocess` predicate
//! by *verifying* fill: when the predicate recommends `LdltCompress`, the
//! compressed prefix is adopted only if its exact factor nnz stays within
//! `PREPROCESS_FILL_INFLATION_LIMIT` (2x) of the `None` baseline - the
//! qap15-class misfire feral measured at 6.3x fill / 20x factor time. An
//! explicit (non-`Auto`) preprocess stays honoured unconditionally.

use rslab::symbolic::{
    pick_ordering_preprocess, symbolic_factorize_with_method, OrderingMethod, OrderingPreprocess,
    SupernodeParams,
};
use rslab::CscMatrix;

/// A 2D grid core with a skirt of degree-1 pendant rows: >=30% of columns
/// have <=2 stored nonzeros, so `pick_ordering_preprocess` fires
/// `LdltCompress` (the IPM-regularization-row shape that motivated the
/// verify guard).
fn grid_with_pendants(k: usize, pendants_per_node: usize) -> CscMatrix<f64> {
    let core = k * k;
    let n = core + core * pendants_per_node;
    let idx = |x: usize, y: usize| y * k + x;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for y in 0..k {
        for x in 0..k {
            let p = idx(x, y);
            r.push(p);
            c.push(p);
            v.push(4.0 + pendants_per_node as f64);
            if x + 1 < k {
                r.push(idx(x + 1, y));
                c.push(p);
                v.push(-1.0);
            }
            if y + 1 < k {
                r.push(idx(x, y + 1));
                c.push(p);
                v.push(-1.0);
            }
            for q in 0..pendants_per_node {
                let pd = core + p * pendants_per_node + q;
                r.push(pd);
                c.push(p);
                v.push(-0.5);
                r.push(pd);
                c.push(pd);
                v.push(2.0);
            }
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).unwrap()
}

#[test]
fn auto_preprocess_decision_matches_fill_verify_rule() {
    let a = grid_with_pendants(16, 1);
    assert_eq!(
        pick_ordering_preprocess(&a),
        OrderingPreprocess::LdltCompress,
        "test matrix must trigger the compression predicate"
    );

    let with_pre = |pre: OrderingPreprocess| SupernodeParams {
        preprocess: pre,
        ..SupernodeParams::default()
    };
    let sym_none = symbolic_factorize_with_method(
        &a,
        &with_pre(OrderingPreprocess::None),
        OrderingMethod::Amd,
    )
    .unwrap();
    let sym_comp = symbolic_factorize_with_method(
        &a,
        &with_pre(OrderingPreprocess::LdltCompress),
        OrderingMethod::Amd,
    )
    .unwrap();
    let sym_auto = symbolic_factorize_with_method(
        &a,
        &with_pre(OrderingPreprocess::Auto),
        OrderingMethod::Amd,
    )
    .unwrap();

    // The dispatcher must implement exactly the 2x rule, whatever this
    // matrix's fills turn out to be.
    let expect = if sym_comp.factor_nnz_estimate as f64 <= 2.0 * sym_none.factor_nnz_estimate as f64
    {
        OrderingPreprocess::LdltCompress
    } else {
        OrderingPreprocess::None
    };
    assert_eq!(sym_auto.resolved_preprocess, expect);
    // And the adopted pipeline's fill must match the corresponding
    // explicit run exactly.
    let expect_nnz = match expect {
        OrderingPreprocess::LdltCompress => sym_comp.factor_nnz_estimate,
        _ => sym_none.factor_nnz_estimate,
    };
    assert_eq!(sym_auto.factor_nnz_estimate, expect_nnz);
}

#[test]
fn explicit_preprocess_stays_unverified() {
    // An explicit LdltCompress request is honoured even where Auto's
    // verify might fall back - the guard is an Auto-only policy.
    let a = grid_with_pendants(12, 1);
    let params = SupernodeParams {
        preprocess: OrderingPreprocess::LdltCompress,
        ..SupernodeParams::default()
    };
    let sym = symbolic_factorize_with_method(&a, &params, OrderingMethod::Amd).unwrap();
    assert_eq!(sym.resolved_preprocess, OrderingPreprocess::LdltCompress);
}
