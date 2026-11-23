//! A factorization records into the caller's rsprobe context: its phases as
//! scopes, its counts as measures at them, its log lines as events there. A
//! setting the chosen path does not read is a WARN event the record keeps.

use rslab::{CscMatrix, LdltSolver, SolverSettings};
use rsprobe::{Buffer, Context, Kind, Level};

/// 2D 5-point grid Laplacian (k x k), SPD-shifted, lower triangle.
fn grid_lower(k: usize) -> CscMatrix<f64> {
    let n = k * k;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..k {
        for j in 0..k {
            let p = i * k + j;
            rows.push(p);
            cols.push(p);
            vals.push(4.5);
            for (ii, jj) in [(i + 1, j), (i, j + 1)] {
                if ii < k && jj < k {
                    let q = ii * k + jj;
                    rows.push(q.max(p));
                    cols.push(q.min(p));
                    vals.push(-1.0);
                }
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

#[test]
fn a_factorization_records_its_phases_counts_and_events() {
    let buffer = Buffer::default();
    let ctx = Context::builder()
        .threshold(Level::INFO)
        .sink(buffer.clone())
        .build();
    let a = grid_lower(12);
    let f = {
        let _on = ctx.enter();
        let opts = SolverSettings::exact().with_pivot_threshold(0.5);
        let f = LdltSolver::factor(&a, &opts).unwrap();
        f.solve(&vec![1.0; a.n]).unwrap();
        f
    };
    let d = f.diagnostics();
    let r = ctx.record();
    let at = |path: &[&str], measure: &str| {
        let node = r.find(path).unwrap_or_else(|| panic!("no scope {path:?}"));
        r.nodes[node as usize].get(measure)
    };
    assert_eq!(at(&["analyze"], "calls"), 1);
    assert_eq!(at(&["analyze"], "nonzeros"), a.row_idx.len() as u64);
    assert_eq!(
        at(&["analyze"], "supernodes"),
        d.decisions.n_supernodes as u64
    );
    assert_eq!(at(&["factor"], "fill"), d.factor_nnz);
    assert_eq!(at(&["factor", "scale"], "calls"), 1);
    assert_eq!(at(&["factor", "solve layout"], "calls"), 1);
    assert_eq!(at(&["solve"], "rhs"), 1);
    // DEBUG scopes stay out at INFO
    assert!(r.find(&["solve", "forward"]).is_none());

    let mut events = Vec::new();
    buffer.drain(|e, path| events.push((e.level, path.to_string(), e.kind.clone())));
    let text = |k: &Kind| k.to_string();
    assert!(
        events
            .iter()
            .any(|(l, p, k)| *l == Level::INFO && p == "factor" && text(k).contains("nnz(L)=")),
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .any(|(l, _, k)| *l == Level::WARN && text(k).contains("pivoting.threshold")),
        "{events:?}"
    );
    assert!(r
        .warnings
        .iter()
        .any(|(e, _)| e.kind.to_string().contains("pivoting.threshold")));
}

#[test]
fn finer_scopes_open_with_the_threshold() {
    let ctx = Context::builder().threshold(Level::DEBUG).build();
    let a = grid_lower(12);
    {
        let _on = ctx.enter();
        let f = LdltSolver::factor(&a, &SolverSettings::exact()).unwrap();
        f.solve(&vec![1.0; a.n]).unwrap();
    }
    let r = ctx.record();
    for path in [
        &["analyze", "pattern"][..],
        &["analyze", "ordering"],
        &["analyze", "supernodes"],
        &["factor", "tree"],
        &["solve", "forward"],
        &["solve", "backward"],
    ] {
        assert!(r.find(path).is_some(), "no scope {path:?}");
    }
    let tree = r.find(&["factor", "tree"]).unwrap();
    let fronts = r.nodes[tree as usize].get("fronts");
    assert_eq!(fronts, f_supernodes(&a));
}

/// The supernodes of `a`'s analysis: one front each.
fn f_supernodes(a: &CscMatrix<f64>) -> u64 {
    let f = LdltSolver::factor(a, &SolverSettings::exact()).unwrap();
    f.diagnostics().decisions.n_supernodes as u64
}
