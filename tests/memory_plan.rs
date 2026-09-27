//! The memory plan predicts, before any numeric work, the heap the factor
//! will hold and what the first factorization adds to the analysis; both are
//! checked here against the heap the built objects report. (The transient
//! peaks are validated against a counting allocator by
//! `benches/memory_peak.py`.)

use num_complex::Complex64;
use rslab::{
    CscMatrix, GeneralCsc, KluSettings, KluSymbolic, LdltSymbolic, LuSymbolic, MemoryPlan, Scalar,
    SolverSettings, Threads,
};

/// 2D 5-point grid (k x k), diagonally dominant; `full` gives both
/// triangles with a small skew on the couplings, else the lower triangle.
fn grid<T: Scalar>(k: usize, full: bool) -> (usize, Vec<usize>, Vec<usize>, Vec<T>) {
    let n = k * k;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..k {
        for j in 0..k {
            let p = i * k + j;
            rows.push(p);
            cols.push(p);
            vals.push(T::from_real(4.5));
            for (di, dj) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (ii, jj) = (i as i64 + di, j as i64 + dj);
                if ii < 0 || jj < 0 || ii as usize >= k || jj as usize >= k {
                    continue;
                }
                let q = ii as usize * k + jj as usize;
                if full {
                    rows.push(p);
                    cols.push(q);
                    vals.push(T::from_real(if di + dj > 0 { -1.0 } else { -0.9 }));
                } else if q > p {
                    rows.push(q);
                    cols.push(p);
                    vals.push(T::from_real(-1.0));
                }
            }
        }
    }
    (n, rows, cols, vals)
}

/// The prediction within `tol` of what was built, never below it.
fn close(what: &str, predicted: u64, actual: u64, tol: f64) {
    assert!(
        predicted >= actual && predicted as f64 <= actual as f64 * (1.0 + tol),
        "{what}: predicted {predicted}, built {actual}"
    );
}

fn opts() -> SolverSettings {
    SolverSettings {
        threads: Threads::Fixed(4),
        ..SolverSettings::default()
    }
}

fn check_ldlt<T: Scalar>() {
    let (n, r, c, v) = grid::<T>(60, false);
    let a = CscMatrix::from_triplets(n, &r, &c, &v).unwrap();
    let sym = LdltSymbolic::analyze(&a, &opts()).unwrap();
    let plan: MemoryPlan = sym.memory_plan::<T>(&opts(), 3);
    let f = sym.factor(&a, &opts()).unwrap();
    close("ldlt factor", plan.factor_bytes, f.heap_bytes(), 0.01);
    close(
        "ldlt analysis",
        plan.analysis_bytes + plan.analysis_growth_bytes,
        sym.heap_bytes(),
        0.01,
    );
    assert!(plan.peak_bytes() >= plan.resident_bytes() + plan.solve_bytes);
    assert_eq!((plan.threads, plan.nrhs), (4, 3));
    // Factored once, the analysis has nothing more to grow.
    assert_eq!(sym.memory_plan::<T>(&opts(), 1).analysis_growth_bytes, 0);
}

fn check_lu<T: Scalar>() {
    let (n, r, c, v) = grid::<T>(60, true);
    let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
    let sym = LuSymbolic::analyze(&a, &opts()).unwrap();
    let plan = sym.memory_plan::<T>(&opts(), 1);
    let f = sym.factor(&a, &opts()).unwrap();
    close("lu factor", plan.factor_bytes, f.heap_bytes(), 0.01);
    close(
        "lu analysis",
        plan.analysis_bytes + plan.analysis_growth_bytes,
        sym.heap_bytes(),
        0.01,
    );
}

fn check_klu<T: Scalar>() {
    let (n, r, c, v) = grid::<T>(60, true);
    let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
    let sym = KluSymbolic::analyze(&a, &KluSettings::default()).unwrap();
    let plan = sym.memory_plan::<T>(&KluSettings::default(), 1);
    let f = sym.factor(&a, &KluSettings::default()).unwrap();
    // Diagonally dominant: the pivots stay on the diagonal, so the
    // symbolic fill the plan prices is the fill the factor holds.
    close("klu factor", plan.factor_bytes, f.heap_bytes(), 0.01);
    assert_eq!(plan.analysis_bytes, sym.heap_bytes());
}

#[test]
fn ldlt_plan_matches_the_built_factor() {
    check_ldlt::<f64>();
    check_ldlt::<Complex64>();
}

#[test]
fn lu_plan_matches_the_built_factor() {
    check_lu::<f64>();
    check_lu::<Complex64>();
}

#[test]
fn klu_plan_matches_the_built_factor() {
    check_klu::<f64>();
    check_klu::<Complex64>();
}

#[test]
fn scratch_grows_with_the_workers() {
    let (n, r, c, v) = grid::<f64>(60, false);
    let a = CscMatrix::from_triplets(n, &r, &c, &v).unwrap();
    let sym = LdltSymbolic::analyze(&a, &opts()).unwrap();
    let at = |t: usize| {
        let o = SolverSettings {
            threads: Threads::Fixed(t),
            ..SolverSettings::default()
        };
        sym.memory_plan::<f64>(&o, 1)
    };
    let (one, many) = (at(1), at(16));
    assert!(many.factor_peak_bytes > one.factor_peak_bytes);
    assert_eq!(many.factor_bytes, one.factor_bytes);
}
