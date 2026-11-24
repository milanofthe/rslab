//! A factorization records into the caller's rsprobe context: its phases as
//! scopes, its counts as measures at them, its log lines as events there. A
//! setting the chosen path does not read is a WARN event the record keeps.

use rslab::{
    CscMatrix, GeneralCsc, KluParallel, KluSettings, KluSymbolic, LdltSolver, LdltSymbolic,
    OrderingMethod, SolverSettings,
};
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
fn a_factorization_reports_its_progress_in_percent() {
    let buffer = Buffer::default();
    let ctx = Context::builder()
        .progress_interval(0)
        .sink(buffer.clone())
        .build();
    let a = grid_lower(12);
    {
        let _on = ctx.enter();
        LdltSolver::factor(&a, &SolverSettings::exact()).unwrap();
    }
    let steps = rslab::constants::PROGRESS_STEPS;
    let mut seen = Vec::new();
    buffer.drain(|e, path| {
        // The analysis parts report with their name.
        if let Kind::Progress { done, total, text } = &e.kind {
            if text.is_empty() {
                assert_eq!((path, *total), ("factor", steps));
                seen.push(*done);
            }
        }
    });
    assert!(seen.len() > 1 && seen.len() as u64 <= steps, "{seen:?}");
    assert_eq!(seen.iter().max(), Some(&steps));
}

/// The progress events of the analysis of `a` under `opts`, per part named.
fn analysis_progress(a: &CscMatrix<f64>, opts: &SolverSettings, part: &str) -> Vec<u64> {
    let buffer = Buffer::default();
    let ctx = Context::builder()
        .progress_interval(0)
        .sink(buffer.clone())
        .build();
    {
        let _on = ctx.enter();
        LdltSymbolic::analyze(a, opts).unwrap();
    }
    let steps = rslab::constants::PROGRESS_STEPS;
    let mut seen = Vec::new();
    buffer.drain(|e, path| {
        if let Kind::Progress { done, total, text } = &e.kind {
            if text == part {
                assert_eq!((path, *total), ("analyze", steps));
                seen.push(*done);
            }
        }
    });
    seen
}

/// Nested dissection joining the ordering race (its gates opened) and AMD
/// alone each report the ordering in percent, ending at the total, with at
/// most one event per step.
#[test]
fn an_analysis_reports_its_ordering_in_percent() {
    let steps = rslab::constants::PROGRESS_STEPS;
    let a = grid_lower(60);
    let mut nd = SolverSettings::default();
    nd.ordering.race.nd_min_n = 0;
    nd.ordering.race.nd_min_work = 0;
    let mut amd = SolverSettings::default();
    amd.ordering.method = OrderingMethod::Amd;
    for (opts, part) in [
        (&nd, "ordering"),
        (&nd, "elimination tree"),
        (&amd, "ordering"),
        (&amd, "elimination tree"),
    ] {
        let seen = analysis_progress(&a, opts, part);
        assert!(
            seen.len() > 1 && seen.len() as u64 <= steps,
            "{part}: {seen:?}"
        );
        assert_eq!(seen.iter().max(), Some(&steps), "{part}");
    }
}

#[test]
fn a_factorization_records_its_phases_counts_and_events() {
    let buffer = Buffer::default();
    let ctx = Context::builder()
        .threshold(Level::INFO)
        .filter("rslab::diagnostics", Level::DEBUG)
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
            .any(|(l, p, k)| *l == Level::DEBUG && p == "factor" && text(k).contains("nnz(L)=")),
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

/// Unsymmetric 2D convection-diffusion grid (k x k), one irreducible block.
fn grid_unsym(k: usize) -> GeneralCsc<f64> {
    let n = k * k;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..k {
        for j in 0..k {
            let p = i * k + j;
            rows.push(p);
            cols.push(p);
            vals.push(4.5);
            for (ii, jj, v) in [(i + 1, j, -1.3), (i, j + 1, -0.7)] {
                if ii < k && jj < k {
                    let q = ii * k + jj;
                    rows.extend([q, p]);
                    cols.extend([p, q]);
                    vals.extend([v, -2.0 - v]);
                }
            }
        }
    }
    GeneralCsc::from_triplets(n, &rows, &cols, &vals).unwrap()
}

#[test]
fn klu_runs_on_the_workers_its_settings_set() {
    // Called from a worker of a wider pool, with every replay forced side by
    // side: the threads busy under each phase are those of the policy, not
    // the caller's pool.
    let width = 2;
    let ctx = Context::builder().threshold(Level::INFO).build();
    let a = grid_unsym(120);
    let s = (KluSettings::default())
        .with_parallel(KluParallel::On)
        .with_threads(width);
    let wide = rayon::ThreadPoolBuilder::new()
        .num_threads(4 * width)
        .build()
        .unwrap();
    wide.install(|| {
        let _on = ctx.enter();
        let sym = KluSymbolic::analyze(&a, &s).unwrap();
        let mut f = sym.factor(&a, &s).unwrap();
        for _ in 0..3 {
            f.refactor(&a).unwrap();
        }
    });
    let r = ctx.record();
    for phase in ["analyze", "factor", "refactor"] {
        let node = &r.nodes[r.find(&[phase]).unwrap() as usize];
        let (wall, busy) = (node.get("wall") as f64, node.get("busy") as f64);
        assert!(
            busy <= width as f64 * wall,
            "{phase}: busy {busy} ns over wall {wall} ns on {width} threads"
        );
    }
}
