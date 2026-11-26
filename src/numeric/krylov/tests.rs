use super::*;
use crate::error::RslabError;
use crate::numeric::ldlt::LdltSolver;
use crate::numeric::settings::SolverSettings;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use crate::test_constants::{
    estimate_growth, gamma, gram_schmidt, recomputed_residual, recursive_gap, varah_condition, U64,
};
use num_complex::Complex;

type C = Complex<f64>;

fn grid(m: usize, diag: C, off: C) -> CscMatrix<C> {
    let n = m * m;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            r.push(p);
            c.push(p);
            v.push(diag);
            if b + 1 < m {
                let (hi, lo) = (idx(a, b + 1), p);
                r.push(hi);
                c.push(lo);
                v.push(off);
            }
            if a + 1 < m {
                let (hi, lo) = (idx(a + 1, b), p);
                r.push(hi);
                c.push(lo);
                v.push(off);
            }
        }
    }
    CscMatrix::<C>::from_triplets(n, &r, &c, &v).unwrap()
}

fn two(v: impl Iterator<Item = f64>) -> f64 {
    v.map(|t| t * t).sum::<f64>().sqrt()
}

fn residual_bound<T: Scalar, A: crate::refine::RefineOperator<T> + ?Sized>(
    a: &A,
    x: &[T],
    b: &[T],
    tol: f64,
) -> (f64, f64) {
    let n = b.len();
    let mut ax = vec![T::zero(); n];
    a.apply(x, &mut ax);
    let mut abs_ax = vec![0.0; n];
    a.apply_abs(x, &mut abs_ax);
    let r = two((0..n).map(|i| (b[i] - ax[i]).magnitude()));
    let abs_norm = two((0..n).map(|i| b[i].magnitude() + abs_ax[i]));
    let b_norm = two(b.iter().map(|v| v.magnitude()));
    (r, recomputed_residual(tol, b_norm, abs_norm, n, n, U64))
}

fn inverse_norm(a: &GeneralCsc<C>) -> f64 {
    let inf = |m: &GeneralCsc<C>| {
        let entries = (0..m.n).flat_map(|j| {
            (m.col_ptr[j]..m.col_ptr[j + 1]).map(move |k| (m.row_idx[k], j, m.values[k].norm()))
        });
        let mut rows = vec![0.0; m.n];
        crate::refine::RefineOperator::apply_abs(m, &vec![C::new(1.0, 0.0); m.n], &mut rows);
        varah_condition(m.n, entries) / rows.into_iter().fold(0.0, f64::max)
    };
    (inf(a) * inf(&a.transpose())).sqrt()
}

fn cocg_drift<M: Preconditioner<C> + ?Sized>(
    a: &CscMatrix<C>,
    b: &[C],
    m: &M,
    settings: &crate::KrylovSettings,
    iters: usize,
) -> f64 {
    let n = b.len();
    let b_norm = two(b.iter().map(|v| v.norm()));
    let (mut ax_max, mut r_max) = (0.0f64, b_norm);
    for j in 1..=iters {
        let step = cocg(a, b, m, &settings.clone().with_max_iter(j)).unwrap();
        let mut abs_ax = vec![0.0; n];
        crate::refine::RefineOperator::apply_abs(a, &step.x, &mut abs_ax);
        ax_max = ax_max.max(two(abs_ax.into_iter()));
        r_max = r_max.max(step.final_res * b_norm);
    }
    recursive_gap(iters, n, U64, ax_max, r_max)
}

#[test]
fn cocg_unpreconditioned_solves_complex_symmetric() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(8, c(4.0, 0.5), c(-1.0, 0.1));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-10)
        .with_max_iter(2000);
    let res = cocg(&a, &b, &NoPreconditioner, &settings).unwrap();
    assert!(res.converged, "COCG should converge, res={}", res.final_res);
    assert_eq!(res.stop, StopReason::Converged);
    let (r, bound) = residual_bound(&a, &res.x, &b, settings.tol);
    let drift = cocg_drift(&a, &b, &NoPreconditioner, &settings, res.iters);
    assert!(r <= bound + drift, "residual {}", r);

    let capped = cocg(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-14)
            .with_max_iter(1),
    )
    .unwrap();
    assert!(!capped.converged);
    assert_eq!(capped.stop, StopReason::MaxIter);
}

#[test]
fn cocr_solves_complex_symmetric_pre_and_unpre() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(10, c(4.0, 0.5), c(-1.0, 0.1));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

    let un = cocr(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(3000),
    )
    .unwrap();
    assert!(un.converged, "COCR res={}", un.final_res);
    let mut ax = vec![C::default(); n];
    a.symv(&un.x, &mut ax);
    let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(res < 1e-7, "COCR residual {}", res);

    let m = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let pre = cocr(
        &a,
        &b,
        &m,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(3000),
    )
    .unwrap();
    assert!(pre.converged && pre.iters <= 3, "iters {}", pre.iters);
}

#[test]
fn gmres_solves_unsymmetric_with_lu_preconditioner() {
    use crate::numeric::lu::LuSolver;
    use crate::sparse::general::GeneralCsc;
    let c = |re, im| Complex::new(re, im);
    let m = 8;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(8.0, 1.0));
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.0, 0.2));
                rr.push(q);
                cc.push(p);
                vv.push(c(-2.0, 0.1));
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.5, 0.3));
                rr.push(q);
                cc.push(p);
                vv.push(c(-0.5, 0.4));
            }
        }
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

    let un = gmres(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(2000)
            .with_restart(40),
        None,
    )
    .unwrap();
    assert!(un.converged, "GMRES res={}", un.final_res);

    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-10)
        .with_max_iter(200)
        .with_restart(40);
    let pre = gmres(&a, &b, &lu, &settings, None).unwrap();
    assert!(pre.converged, "preconditioned GMRES res={}", pre.final_res);
    assert!(
        pre.iters <= 3,
        "LU-preconditioned GMRES iters {}",
        pre.iters
    );
    let (res, bound) = residual_bound(&a, &pre.x, &b, settings.tol);
    assert!(res <= bound, "residual {}", res);
}

#[test]
fn gmres_singular_operator_breaks_down_without_nan() {
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let a =
        GeneralCsc::<C>::from_triplets(3, &[0, 1], &[0, 1], &[c(1.0, 0.0), c(1.0, 0.0)]).unwrap();
    let b = vec![c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];
    let res = gmres(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-12)
            .with_max_iter(50)
            .with_restart(10),
        None,
    )
    .unwrap();
    assert!(
        res.x.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
        "solution has NaN/Inf: {:?}",
        res.x
    );
    assert!(
        res.final_res.is_finite(),
        "residual is NaN/Inf: {}",
        res.final_res
    );
    assert!(
        !res.converged,
        "singular system must not report convergence"
    );
    let floor = b[2].norm() / two(b.iter().map(|v| v.norm()));
    assert!(
        res.final_res >= floor * (1.0 - gamma(2 * b.len() + 8, U64)) && res.final_res <= 1.0,
        "residual not in the sane breakdown range: {}",
        res.final_res
    );
}

fn unsym_grid(m: usize) -> crate::sparse::general::GeneralCsc<C> {
    use crate::sparse::general::GeneralCsc;
    let c = |re, im| Complex::new(re, im);
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(8.0, 1.0));
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.0, 0.2));
                rr.push(q);
                cc.push(p);
                vv.push(c(-2.0, 0.1));
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.5, 0.3));
                rr.push(q);
                cc.push(p);
                vv.push(c(-0.5, 0.4));
            }
        }
    }
    GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap()
}

fn diagonal_part(
    a: &crate::sparse::general::GeneralCsc<C>,
) -> crate::sparse::general::GeneralCsc<C> {
    let (mut idx, mut vals) = (Vec::new(), Vec::new());
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            if a.row_idx[k] == j {
                idx.push(j);
                vals.push(a.values[k]);
            }
        }
    }
    crate::sparse::general::GeneralCsc::from_triplets(a.n, &idx, &idx, &vals).unwrap()
}

struct CountingPc<'a, M: ?Sized> {
    inner: &'a M,
    applies: std::sync::atomic::AtomicUsize,
}
impl<T: Scalar, M: Preconditioner<T> + ?Sized> Preconditioner<T> for CountingPc<'_, M> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        self.applies
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.apply(r, z)
    }
}

#[test]
fn fgmres_saves_one_precond_apply_per_restart_cycle() {
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(10);
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let lu = LuSolver::factor(&diagonal_part(&a), &SolverSettings::default()).unwrap();
    let restart = 5;
    let counting = CountingPc {
        inner: &lu,
        applies: std::sync::atomic::AtomicUsize::new(0),
    };
    let res = gmres(
        &a,
        &b,
        &counting,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(2000)
            .with_restart(restart),
        None,
    )
    .unwrap();
    assert!(res.converged, "FGMRES must converge, res={}", res.final_res);
    let applies = counting.applies.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        res.iters > restart,
        "expected multiple cycles, iters={}",
        res.iters
    );
    assert_eq!(
        applies, res.iters,
        "FGMRES precond applies {} must equal iters {} (no per-cycle extra solve)",
        applies, res.iters
    );
}

#[test]
fn gmres_warm_start_cuts_total_iterations_on_related_sequence() {
    let a = unsym_grid(12);
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b0: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let b1: Vec<C> = (0..n)
        .map(|i| c(((i * 3) % 7) as f64 - 3.0, ((i % 3) as f64) - 1.0))
        .collect();
    let steps = 10;
    let (tol, maxit, restart) = (1e-8, 4000, 60);

    let bk = |k: usize| -> Vec<C> {
        let th = 5e-5 * k as f64;
        let (ct, st) = (th.cos(), th.sin());
        (0..n)
            .map(|i| b0[i] * c(ct, 0.0) + b1[i] * c(st, 0.0))
            .collect()
    };

    let mut cold_total = 0usize;
    for k in 0..steps {
        let r = gmres(
            &a,
            &bk(k),
            &NoPreconditioner,
            &crate::KrylovSettings::default()
                .with_tol(tol)
                .with_max_iter(maxit)
                .with_restart(restart),
            None,
        )
        .unwrap();
        assert!(r.converged, "cold solve {k} did not converge");
        cold_total += r.iters;
    }

    let mut warm_total = 0usize;
    let mut prev: Option<Vec<C>> = None;
    for k in 0..steps {
        let r = gmres(
            &a,
            &bk(k),
            &NoPreconditioner,
            &crate::KrylovSettings::default()
                .with_tol(tol)
                .with_max_iter(maxit)
                .with_restart(restart),
            prev.as_deref(),
        )
        .unwrap();
        assert!(r.converged, "warm solve {k} did not converge");
        warm_total += r.iters;
        prev = Some(r.x);
    }

    assert!(
        (warm_total as f64) < 0.7 * (cold_total as f64),
        "warm start did not help enough: cold={cold_total}, warm={warm_total}"
    );
}

#[test]
fn gmres_block_single_rhs_matches_scalar_gmres() {
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(8);
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-10)
        .with_max_iter(200)
        .with_restart(40);
    let single = gmres(&a, &b, &lu, &settings, None).unwrap();
    let blk = gmres_block(&a, &b, 1, &lu, &settings, None, None).unwrap();
    assert!(blk.converged);
    assert!(
        (blk.iters as i64 - single.iters as i64).abs() <= 1,
        "block(s=1) iters {} vs single {}",
        blk.iters,
        single.iters
    );
    let diff = (0..n)
        .map(|i| (blk.x[i] - single.x[i]).norm())
        .fold(0.0, f64::max);
    let gap = residual_bound(&a, &blk.x, &b, settings.tol).1
        + residual_bound(&a, &single.x, &b, settings.tol).1;
    assert!(
        diff <= inverse_norm(&a) * gap,
        "block(s=1) solution differs by {diff}"
    );
}

#[test]
fn gmres_block_multi_rhs_solves_each_column() {
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(10);
    let n = a.n;
    let s = 5;
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-10)
        .with_max_iter(200)
        .with_restart(40);
    let res = gmres_block(&a, &bblk, s, &lu, &settings, None, None).unwrap();
    assert!(
        res.converged,
        "block GMRES must converge; res={:?}",
        res.final_res
    );
    for k in 0..s {
        let col = k * n..k * n + n;
        let (r, bound) = residual_bound(&a, &res.x[col.clone()], &bblk[col], settings.tol);
        assert!(r <= bound, "column {k} residual {r}");
    }
    let inv = inverse_norm(&a);
    for k in 0..s {
        let col = k * n..k * n + n;
        let single = gmres(&a, &bblk[col.clone()], &lu, &settings, None).unwrap();
        let diff = (0..n)
            .map(|i| (res.x[k * n + i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        let gap = residual_bound(&a, &res.x[col.clone()], &bblk[col.clone()], settings.tol).1
            + residual_bound(&a, &single.x, &bblk[col], settings.tol).1;
        assert!(
            diff <= inv * gap,
            "column {k} differs from single solve by {diff}"
        );
    }
}

#[test]
fn gmres_block_within_cycle_deflation_shrinks_applies() {
    use std::sync::Mutex;

    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 8;
    let s = 4;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0 + i as f64, 0.5 + 0.1 * i as f64));
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..=k {
            bblk[k * n + i] = c(1.0, 0.0);
        }
    }

    let op = WidthCountingOp {
        inner: &a,
        widths: Mutex::new(Vec::new()),
    };
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-12)
        .with_max_iter(200)
        .with_restart(40);
    let res = gmres_block(&op, &bblk, s, &NoPreconditioner, &settings, None, None).unwrap();
    assert!(
        res.converged,
        "block GMRES must converge; res={:?}",
        res.final_res
    );

    let inv = inverse_norm(&a);
    for k in 0..s {
        let col = k * n..k * n + n;
        let single = gmres(&a, &bblk[col.clone()], &NoPreconditioner, &settings, None).unwrap();
        let diff = (0..n)
            .map(|i| (res.x[k * n + i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        let gap = residual_bound(&a, &res.x[col.clone()], &bblk[col.clone()], settings.tol).1
            + residual_bound(&a, &single.x, &bblk[col], settings.tol).1;
        assert!(
            diff <= inv * gap,
            "column {k} differs from single solve by {diff}"
        );
    }

    let widths = op.widths.into_inner().unwrap();
    let ncalls = widths.len();
    let total_cols: usize = widths.iter().sum();
    assert_eq!(
        *widths.iter().max().unwrap(),
        s,
        "the first cycle must open at full width"
    );
    assert!(
        *widths.iter().min().unwrap() < s,
        "no apply narrowed: deflation did not shrink the panel"
    );
    assert!(
        widths.contains(&1),
        "the panel must drain to a single active column"
    );
    assert!(
        total_cols < s * ncalls,
        "total column-applies {total_cols} not below the full-width bound {}",
        s * ncalls
    );
}

#[test]
fn gmres_block_dcgs2_bit_identical_across_thread_counts() {
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(60);
    let n = a.n;
    let s = 5;
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let solve = || {
        gmres_block(
            &a,
            &bblk,
            s,
            &lu,
            &crate::KrylovSettings::default()
                .with_tol(1e-10)
                .with_max_iter(300)
                .with_restart(60),
            None,
            None,
        )
        .unwrap()
    };
    let x1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(solve);
    let x8 = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
        .install(solve);
    assert_eq!(
        x1.iters, x8.iters,
        "iteration count must not depend on threads"
    );
    assert!(
        x1.x == x8.x,
        "block solve must be bit-identical across thread counts"
    );
}

#[test]
fn block_gmres_orthogonalization_respects_factor_thread_cap() {
    use crate::numeric::lu::LuSolver;
    use crate::numeric::settings::Threads;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(30);
    let n = a.n;
    let s = 4;
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    let lu = LuSolver::factor(&a, &SolverSettings::default().with_threads(2)).unwrap();
    assert_eq!(
        Preconditioner::<C>::solve_threads(&lu),
        Threads::Fixed(2),
        "factor must carry its resolved solve-phase thread budget"
    );
    let policy = Preconditioner::<C>::solve_threads(&lu);
    let seen = with_threads(8, || policy.run(0, |cap| cap, rayon::current_num_threads));
    assert_eq!(
        seen, 2,
        "the ortho pool must cap to the factor's 2-worker budget"
    );

    let bare = gmres_block(
        &a,
        &bblk,
        s,
        &lu,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(300)
            .with_restart(60),
        None,
        None,
    )
    .unwrap();
    let in_wide = with_threads(8, || {
        gmres_block(
            &a,
            &bblk,
            s,
            &lu,
            &crate::KrylovSettings::default()
                .with_tol(1e-10)
                .with_max_iter(300)
                .with_restart(60),
            None,
            None,
        )
        .unwrap()
    });
    assert!(bare.converged);
    assert!(
        bare.x == in_wide.x && bare.iters == in_wide.iters,
        "capped ortho pool must not perturb the numeric result"
    );
}

#[test]
fn ambient_threads_factor_matches_default_and_runs_on_shared_pool() {
    use crate::numeric::lu::LuSolver;
    use crate::numeric::settings::Threads;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(24);
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let lu_default = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let opts_amb = SolverSettings::default().with_threads(Threads::Ambient);
    let lu_amb = with_threads(2, || {
        assert_eq!(rayon::current_num_threads(), 2);
        LuSolver::factor(&a, &opts_amb).unwrap()
    });
    let x_def = gmres_block(
        &a,
        &b,
        1,
        &lu_default,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(200)
            .with_restart(40),
        None,
        None,
    )
    .unwrap();
    let x_amb = gmres_block(
        &a,
        &b,
        1,
        &lu_amb,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(200)
            .with_restart(40),
        None,
        None,
    )
    .unwrap();
    assert!(
        x_def.x == x_amb.x,
        "ambient-pool factor must be bit-identical to the default factor"
    );
}

#[test]
fn default_thread_policy_caps_at_four() {
    use crate::numeric::settings::Threads;
    assert_eq!(SolverSettings::default().threads, Threads::Auto { max: 4 });
}

#[test]
fn f32_lu_preconditioner_keeps_f64_accuracy_in_gmres() {
    use crate::sparse::general::GeneralCsc;
    let c = |re, im| Complex::new(re, im);
    let m = 10;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(20.0, 2.0));
            if b + 1 < m {
                rr.push(p);
                cc.push(idx(a, b + 1));
                vv.push(c(-1.0, 0.2));
                rr.push(idx(a, b + 1));
                cc.push(p);
                vv.push(c(-2.0, 0.1));
            }
            if a + 1 < m {
                rr.push(p);
                cc.push(idx(a + 1, b));
                vv.push(c(-1.5, 0.3));
                rr.push(idx(a + 1, b));
                cc.push(p);
                vv.push(c(-0.5, 0.4));
            }
        }
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();
    let pc = LowPrecisionLu::factor(&a, &SolverSettings::default()).unwrap();
    assert!(pc.factor_nnz() > 0);
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-6)
        .with_max_iter(200)
        .with_restart(50);
    let res = gmres(&a, &b, &pc, &settings, None).unwrap();
    assert!(res.converged, "mixed-precision GMRES res={}", res.final_res);
    assert!(res.iters <= 6, "iters {}", res.iters);
    let (r, bound) = residual_bound(&a, &res.x, &b, settings.tol);
    assert!(r <= bound, "residual {}", r);
}

#[test]
fn cocr_handles_indefinite_helmholtz() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(10, c(-1.0, 0.3), c(1.0, 0.05));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c(1.0, (i % 3) as f64 - 1.0)).collect();
    let opts = SolverSettings::preconditioner(1e-10);
    let m = LdltSolver::factor(&a, &opts).unwrap();
    let pre = cocr(
        &a,
        &b,
        &m,
        &crate::KrylovSettings::default()
            .with_tol(1e-9)
            .with_max_iter(500),
    )
    .unwrap();
    assert!(
        pre.converged,
        "indefinite COCR res={} iters={}",
        pre.final_res, pre.iters
    );
    let mut ax = vec![C::default(); n];
    a.symv(&pre.x, &mut ax);
    let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(res < 1e-6, "residual {}", res);
}

#[test]
fn f32_preconditioner_keeps_f64_accuracy() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(14, c(4.0, 1.0), c(-1.0, 0.2));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

    let m = LowPrecisionPreconditioner::factor(&a, &SolverSettings::default()).unwrap();
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-10)
        .with_max_iter(500);
    let res = cocg(&a, &b, &m, &settings).unwrap();
    assert!(res.converged, "mixed-precision COCG res={}", res.final_res);
    assert!(res.iters <= 12, "f32-preconditioned iters {}", res.iters);
    let (r, bound) = residual_bound(&a, &res.x, &b, settings.tol);
    let drift = cocg_drift(&a, &b, &m, &settings, res.iters);
    assert!(r <= bound + drift, "mixed-precision residual {}", r);
    assert!(m.factor_nnz() > 0);
}

#[test]
fn rla_preconditioner_collapses_iteration_count() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(12, c(4.0, 1.0), c(-1.0, 0.2));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

    let unpre = cocg(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(5000),
    )
    .unwrap();
    let m = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let pre = cocg(
        &a,
        &b,
        &m,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(5000),
    )
    .unwrap();

    assert!(pre.converged && unpre.converged);
    assert!(
        pre.iters <= 3,
        "complete-factor preconditioner should need <=3 iters, got {}",
        pre.iters
    );
    assert!(
        pre.iters * 5 < unpre.iters,
        "preconditioner should cut iterations sharply: {} vs {}",
        pre.iters,
        unpre.iters
    );
}

fn convection_diffusion(n: usize, gamma: f64) -> crate::sparse::general::GeneralCsc<C> {
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0, 0.02));
        if i + 1 < n {
            rr.push(i);
            cc.push(i + 1);
            vv.push(c(-1.0 + gamma, 0.0));
            rr.push(i + 1);
            cc.push(i);
            vv.push(c(-1.0 - gamma, 0.0));
        }
    }
    GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap()
}

struct WidthCountingOp<'a> {
    inner: &'a GeneralCsc<C>,
    widths: std::sync::Mutex<Vec<usize>>,
}
impl LinearOperator<C> for WidthCountingOp<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn apply(&self, x: &[C], y: &mut [C]) {
        self.widths.lock().unwrap().push(1);
        self.inner.apply(x, y);
    }
    fn apply_block(&self, x: &[C], y: &mut [C], s: usize) {
        self.widths.lock().unwrap().push(s);
        self.inner.apply_block(x, y, s);
    }
}

#[test]
fn gmres_unpreconditioned_nonnormal_needs_many_restarts() {
    let a = convection_diffusion(120, 0.9);
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b: Vec<C> = (0..n).map(|i| c(((i % 5) as f64) - 2.0, 0.5)).collect();
    let restart = 20;
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-8)
        .with_max_iter(8000)
        .with_restart(restart);
    let res = gmres(&a, &b, &NoPreconditioner, &settings, None).unwrap();
    assert!(
        res.converged,
        "non-normal GMRES must converge, res={}",
        res.final_res
    );
    assert!(
        res.iters > restart,
        "must span multiple restart cycles, iters={} (restart={})",
        res.iters,
        restart
    );
    let (r, bound) = residual_bound(&a, &res.x, &b, settings.tol);
    assert!(r <= bound, "true residual {}", r);
}

#[test]
fn gmres_reorthogonalization_keeps_illconditioned_arnoldi_accurate() {
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 32;
    let (diag, sup) = (2.0, 3.0);
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(diag, 0.0));
        if i + 1 < n {
            rr.push(i);
            cc.push(i + 1);
            vv.push(c(sup, 0.0));
        }
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<C> = (0..n).map(|_| c(1.0, 0.2)).collect();
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-10)
        .with_max_iter(4000)
        .with_restart(n);
    let res = gmres(&a, &b, &NoPreconditioner, &settings, None).unwrap();
    assert!(
        res.converged,
        "reorth must keep GMRES converging, res={}",
        res.final_res
    );
    let (r, bound) = residual_bound(&a, &res.x, &b, settings.tol);
    assert!(
        r <= bound,
        "GMRES true residual {r:e} above {bound:e} (lost orthogonality?)"
    );
}

#[test]
fn gmres_happy_breakdown_on_eigenvector_rhs() {
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 12;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0 + i as f64, 0.5 - 0.05 * i as f64));
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let mut b = vec![C::default(); n];
    b[0] = c(1.0, 0.0);
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-12)
        .with_max_iter(50)
        .with_restart(30);
    let res = gmres(&a, &b, &NoPreconditioner, &settings, None).unwrap();
    assert!(
        res.converged,
        "eigenvector RHS must converge, res={}",
        res.final_res
    );
    assert_eq!(
        res.iters, 1,
        "happy breakdown must solve in one step, got {}",
        res.iters
    );
    let (r, bound) = residual_bound(&a, &res.x, &b, settings.tol);
    assert!(r <= bound, "true residual {}", r);
}

#[test]
fn gmres_block_nearby_factor_multirate_deflation() {
    use crate::numeric::lu::LuSolver;
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 8;
    let s = 4;
    let (mut dr, mut dc, mut dv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        dr.push(i);
        dc.push(i);
        dv.push(c(2.0 + i as f64, 0.3));
    }
    let a = GeneralCsc::<C>::from_triplets(n, &dr, &dc, &dv).unwrap();
    let (mut pr, mut pc_, mut pv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        pr.push(i);
        pc_.push(i);
        pv.push(c(1.0 + i as f64, 0.1));
    }
    let pmat = GeneralCsc::<C>::from_triplets(n, &pr, &pc_, &pv).unwrap();
    let lu = LuSolver::factor(&pmat, &SolverSettings::default()).unwrap();

    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..=k {
            bblk[k * n + i] = c(1.0, 0.0);
        }
    }

    let op = WidthCountingOp {
        inner: &a,
        widths: std::sync::Mutex::new(Vec::new()),
    };
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-10)
        .with_max_iter(200)
        .with_restart(40);
    let res = gmres_block(&op, &bblk, s, &lu, &settings, None, None).unwrap();
    assert!(
        res.converged,
        "block GMRES must converge; res={:?}",
        res.final_res
    );

    let inv = inverse_norm(&a);
    for k in 0..s {
        let col = k * n..k * n + n;
        let single = gmres(&a, &bblk[col.clone()], &lu, &settings, None).unwrap();
        let diff = (0..n)
            .map(|i| (res.x[k * n + i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        let gap = residual_bound(&a, &res.x[col.clone()], &bblk[col.clone()], settings.tol).1
            + residual_bound(&a, &single.x, &bblk[col], settings.tol).1;
        assert!(
            diff <= inv * gap,
            "column {k} differs from single solve by {diff}"
        );
    }

    let widths = op.widths.into_inner().unwrap();
    assert!(!widths.is_empty());
    assert_eq!(
        *widths.iter().max().unwrap(),
        s,
        "the first cycle must open at full width"
    );
    assert!(
        *widths.iter().min().unwrap() < s,
        "within-cycle deflation did not shrink the panel: widths={widths:?}"
    );
    assert!(
        widths.contains(&1),
        "the panel must drain to a single active column: {widths:?}"
    );
}

fn diag_op(eigs: &[C]) -> crate::sparse::general::GeneralCsc<C> {
    use crate::sparse::general::GeneralCsc;
    let n = eigs.len();
    let idx: Vec<usize> = (0..n).collect();
    GeneralCsc::<C>::from_triplets(n, &idx, &idx, eigs).unwrap()
}

fn stagnation_spectrum(n: usize, n_small: usize) -> Vec<C> {
    let c = |re: f64, im: f64| Complex::new(re, im);
    (0..n)
        .map(|i| {
            if i < n_small {
                c(0.01 + 0.004 * i as f64, 0.002 * i as f64)
            } else {
                let t = (i - n_small) as f64 / (n - n_small) as f64;
                c(1.0 + 10.0 * t, 0.3 * (i as f64).sin())
            }
        })
        .collect()
}

#[test]
fn gmres_recycled_matches_plain_on_hard_matrix() {
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(12);
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let lu = LuSolver::factor(&diagonal_part(&a), &SolverSettings::default()).unwrap();
    let (tol, maxit, restart) = (1e-10, 4000, 12);
    let plain = gmres(
        &a,
        &b,
        &lu,
        &crate::KrylovSettings::default()
            .with_tol(tol)
            .with_max_iter(maxit)
            .with_restart(restart),
        None,
    )
    .unwrap();
    let mut rec = Recycle::new(8);
    let recd = gmres_recycled(
        &a,
        &b,
        &lu,
        &crate::KrylovSettings::default()
            .with_tol(tol)
            .with_max_iter(maxit)
            .with_restart(restart),
        None,
        &mut rec,
    )
    .unwrap();
    assert!(plain.converged, "plain FGMRES did not converge");
    assert!(recd.converged, "recycled did not converge");
    let diff = (0..n)
        .map(|i| (plain.x[i] - recd.x[i]).norm())
        .fold(0.0, f64::max);
    let gap = residual_bound(&a, &plain.x, &b, tol).1 + residual_bound(&a, &recd.x, &b, tol).1;
    assert!(
        diff <= inverse_norm(&a) * gap,
        "recycled solution differs from plain by {diff}"
    );
    assert!(rec.active() > 0, "recycle subspace was not populated");
}

#[test]
fn gmres_recycled_within_solve_reduces_restarts() {
    let eigs = stagnation_spectrum(40, 4);
    let a = diag_op(&eigs);
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b: Vec<C> = (0..n).map(|i| c(1.0, 0.2 * (i as f64).cos())).collect();
    let (tol, maxit, restart) = (1e-9, 5000, 10);

    let plain = gmres(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(tol)
            .with_max_iter(maxit)
            .with_restart(restart),
        None,
    )
    .unwrap();
    let mut rec = Recycle::new(6);
    let recd = gmres_recycled(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(tol)
            .with_max_iter(maxit)
            .with_restart(restart),
        None,
        &mut rec,
    )
    .unwrap();
    assert!(plain.converged, "plain did not converge ({})", plain.iters);
    assert!(recd.converged, "recycled did not converge ({})", recd.iters);
    let diff = (0..n)
        .map(|i| (plain.x[i] - recd.x[i]).norm())
        .fold(0.0, f64::max);
    let gap = residual_bound(&a, &plain.x, &b, tol).1 + residual_bound(&a, &recd.x, &b, tol).1;
    assert!(diff <= inverse_norm(&a) * gap, "solutions differ by {diff}");
    eprintln!(
        "[within-solve] plain FGMRES(10)={} iters, GCRO-DR(k=6)={} iters",
        plain.iters, recd.iters
    );
    assert!(
        (recd.iters as f64) < 0.75 * (plain.iters as f64),
        "within-solve deflation did not help enough: plain={}, recycled={}",
        plain.iters,
        recd.iters
    );
}

#[test]
fn gmres_recycled_cross_solve_beats_warm_and_cold() {
    let base = stagnation_spectrum(48, 5);
    let n = base.len();
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b0: Vec<C> = (0..n).map(|i| c(1.0, 0.15 * (i as f64).cos())).collect();
    let b1: Vec<C> = (0..n).map(|i| c(0.4 * (i as f64).sin(), 1.0)).collect();
    let steps = 8;
    let (tol, maxit, restart) = (1e-9, 6000, 12);

    let ak = |kk: usize| -> crate::sparse::general::GeneralCsc<C> {
        let eps = 2e-3 * kk as f64;
        let eigs: Vec<C> = base
            .iter()
            .enumerate()
            .map(|(i, &e)| e + c(eps * (1.0 + 0.05 * i as f64), 0.0))
            .collect();
        diag_op(&eigs)
    };
    let bk = |kk: usize| -> Vec<C> {
        let th = 0.02 * kk as f64;
        let (ct, st) = (th.cos(), th.sin());
        (0..n)
            .map(|i| b0[i] * c(ct, 0.0) + b1[i] * c(st, 0.0))
            .collect()
    };

    let mut cold = 0usize;
    for kk in 0..steps {
        let a = ak(kk);
        let r = gmres(
            &a,
            &bk(kk),
            &NoPreconditioner,
            &crate::KrylovSettings::default()
                .with_tol(tol)
                .with_max_iter(maxit)
                .with_restart(restart),
            None,
        )
        .unwrap();
        assert!(r.converged, "cold {kk} stalled");
        cold += r.iters;
    }
    let mut warm = 0usize;
    let mut prev: Option<Vec<C>> = None;
    for kk in 0..steps {
        let a = ak(kk);
        let r = gmres(
            &a,
            &bk(kk),
            &NoPreconditioner,
            &crate::KrylovSettings::default()
                .with_tol(tol)
                .with_max_iter(maxit)
                .with_restart(restart),
            prev.as_deref(),
        )
        .unwrap();
        assert!(r.converged, "warm {kk} stalled");
        warm += r.iters;
        prev = Some(r.x);
    }
    let mut recycled = 0usize;
    let mut rec = Recycle::new(8);
    let mut prevr: Option<Vec<C>> = None;
    for kk in 0..steps {
        let a = ak(kk);
        let r = gmres_recycled(
            &a,
            &bk(kk),
            &NoPreconditioner,
            &crate::KrylovSettings::default()
                .with_tol(tol)
                .with_max_iter(maxit)
                .with_restart(restart),
            prevr.as_deref(),
            &mut rec,
        )
        .unwrap();
        assert!(r.converged, "recycled {kk} stalled");
        recycled += r.iters;
        prevr = Some(r.x);
    }

    eprintln!(
        "[cross-solve] cold={cold}, warm={warm}, recycled={recycled} (total iters over {steps} solves)"
    );
    assert!(warm < cold, "warm ({warm}) not below cold ({cold})");
    assert!(
        recycled < warm,
        "recycled ({recycled}) not below warm ({warm})"
    );
    assert!(
        (recycled as f64) < 0.7 * (cold as f64),
        "recycled ({recycled}) did not beat cold ({cold}) by a clear margin"
    );
}

#[test]
fn gmres_recycled_composes_with_warm_start() {
    let eigs = stagnation_spectrum(40, 4);
    let a = diag_op(&eigs);
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b0: Vec<C> = (0..n).map(|i| c(1.0, 0.1 * (i as f64).cos())).collect();
    let b1: Vec<C> = (0..n).map(|i| c(1.0 + 0.02 * i as f64, 0.1)).collect();
    let (tol, maxit, restart) = (1e-9, 5000, 10);

    let mut rec = Recycle::new(6);
    let first = gmres_recycled(
        &a,
        &b0,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(tol)
            .with_max_iter(maxit)
            .with_restart(restart),
        None,
        &mut rec,
    )
    .unwrap();
    assert!(first.converged);

    let warm_only = gmres(
        &a,
        &b1,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(tol)
            .with_max_iter(maxit)
            .with_restart(restart),
        Some(&first.x),
    )
    .unwrap();
    let warm_rec = gmres_recycled(
        &a,
        &b1,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(tol)
            .with_max_iter(maxit)
            .with_restart(restart),
        Some(&first.x),
        &mut rec,
    )
    .unwrap();
    assert!(warm_only.converged && warm_rec.converged);
    let (res, bound) = residual_bound(&a, &warm_rec.x, &b1, tol);
    assert!(res <= bound, "warm+recycle residual {res}");
    assert!(
        warm_rec.iters <= warm_only.iters,
        "warm+recycle ({}) worse than warm-only ({})",
        warm_rec.iters,
        warm_only.iters
    );
}

#[test]
fn gmres_recycled_real_scalar_path() {
    use crate::sparse::general::GeneralCsc;
    let m = 8;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(4.0f64);
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(-1.0);
                rr.push(q);
                cc.push(p);
                vv.push(-1.8);
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(-0.7);
                rr.push(q);
                cc.push(p);
                vv.push(-1.3);
            }
        }
    }
    let a = GeneralCsc::<f64>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let mut rec = Recycle::<f64>::new(6);
    let settings = crate::KrylovSettings::default()
        .with_tol(1e-9)
        .with_max_iter(5000)
        .with_restart(12);
    let r = gmres_recycled(&a, &b, &NoPreconditioner, &settings, None, &mut rec).unwrap();
    assert!(r.converged, "real recycled solve did not converge");
    let (res, bound) = residual_bound(&a, &r.x, &b, settings.tol);
    assert!(res <= bound, "real recycled residual {res}");
}

fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

fn ladder_rhs(n: usize, s: usize) -> Vec<C> {
    let mut b = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            b[k * n + i] = Complex::new(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    b
}

#[test]
fn block_gmres_continued_is_the_straight_solve() {
    let a = unsym_grid(20);
    let n = a.n;
    let b = ladder_rhs(n, 1);
    let set = crate::KrylovSettings::default()
        .with_max_iter(2000)
        .with_restart(600);
    let straight = gmres_block(
        &a,
        &b,
        1,
        &NoPreconditioner,
        &set.clone().with_tol(1e-10),
        None,
        None,
    )
    .unwrap();
    let mut g = BlockGmres::new(&a, &b, 1, &NoPreconditioner, &set, None).unwrap();
    let (coarse, fine) = (1e-4, 1e-10);
    let r1 = g.solve(coarse, None).unwrap();
    assert!(
        r1.converged && r1.final_res[0] <= coarse,
        "{:?}",
        r1.final_res
    );
    let r2 = g.solve(fine, None).unwrap();
    assert!(
        r2.converged && r2.final_res[0] <= fine,
        "{:?}",
        r2.final_res
    );
    assert!(straight.iters > r1.iters, "the first rung stops early");
    assert_eq!(r1.iters + r2.iters, straight.iters);
    assert!(
        r2.x == straight.x,
        "the continued solve is the straight one"
    );
}

#[test]
fn block_gmres_continued_rung_beats_a_restart() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(20, c(0.6, 0.02), c(-1.0, 0.0));
    let n = a.n;
    let s = 3;
    let b = ladder_rhs(n, s);
    let set = crate::KrylovSettings::default()
        .with_max_iter(2000)
        .with_restart(600);
    let mut g = BlockGmres::new(&a, &b, s, &NoPreconditioner, &set, None).unwrap();
    let tol = 1e-10;
    let r1 = g.solve(1e-6, None).unwrap();
    let r2 = g.solve(tol, None).unwrap();
    let fresh = gmres_block(
        &a,
        &b,
        s,
        &NoPreconditioner,
        &set.clone().with_tol(tol),
        Some(&r1.x),
        None,
    )
    .unwrap();
    assert!(r2.converged && fresh.converged);
    assert!(
        r2.iters < fresh.iters,
        "continued {} against restarted {}",
        r2.iters,
        fresh.iters
    );
    for k in 0..s {
        let col = k * n..(k + 1) * n;
        let (rn, bound) = residual_bound(&a, &r2.x[col.clone()], &b[col], tol);
        assert!(rn <= bound, "column {k}: {rn}");
    }
}

// https://doi.org/10.1016/j.camwa.2005.08.009
#[test]
fn block_gmres_delayed_cgs2_keeps_the_basis_orthonormal() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(20, c(0.6, 0.02), c(-1.0, 0.0));
    let n = a.n;
    let (s, k) = (2, 200);
    let b = ladder_rhs(n, s);
    for flexible in [true, false] {
        let set = crate::KrylovSettings::default()
            .with_max_iter(k)
            .with_restart(k)
            .with_flexible(flexible);
        let mut g = BlockGmres::new(&a, &b, s, &NoPreconditioner, &set, None).unwrap();
        let r = g.solve(1e-15, None).unwrap();
        assert_eq!(r.iters, k);
        let loss = g.orthogonality_loss(k);
        eprintln!("flexible {flexible}: ||V^H V - I||_F = {loss:.3e} over {k} vectors");
        let bound = gram_schmidt(k, n, U64);
        assert!(loss <= bound, "loss {loss:.3e}");
        for (col, &res) in r.final_res.iter().enumerate() {
            let est = g.estimate(col);
            assert!(
                (est - res).abs() <= bound,
                "column {col}: estimate {est:.3e} true {res:.3e}"
            );
        }
    }
}

#[test]
fn block_gmres_memory_plan_matches_the_solve() {
    let a = unsym_grid(16);
    let n = a.n;
    let set = crate::KrylovSettings::default()
        .with_max_iter(2000)
        .with_restart(300);
    for (s, single) in [(1usize, false), (4, false), (1, true), (4, true)] {
        let b = ladder_rhs(n, s);
        let set = set.clone().with_basis_single(single);
        let mut g = BlockGmres::new(&a, &b, s, &NoPreconditioner, &set, None).unwrap();
        g.solve(1e-9, None).unwrap();
        let plan = set.block_bytes::<C>(n, s, g.deepest());
        let held = g.resident_bytes();
        if s == 1 {
            assert_eq!(plan, held);
        } else {
            assert!(plan >= held && held > 0, "plan {plan} held {held}");
        }
    }
}

#[test]
fn block_gmres_deflating_a_harvest_saves_iterations() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(20, c(0.6, 0.02), c(-1.0, 0.0));
    let n = a.n;
    let s = 3;
    let set = crate::KrylovSettings::default()
        .with_max_iter(2000)
        .with_restart(600);
    let first = ladder_rhs(n, s);
    let tol = 1e-10;
    let mut g = BlockGmres::new(&a, &first, s, &NoPreconditioner, &set, None).unwrap();
    let harvested = g.solve(tol, None).unwrap();
    assert!(harvested.converged);
    let rec = g.harvest(20).expect("open cycles to harvest");
    assert!(rec.dim() > 0 && rec.dim() <= 20);
    {
        let (u, cimg) = rec.parts();
        for i in 0..rec.dim() {
            let mut y = vec![C::default(); n];
            a.symv(&u[i * n..(i + 1) * n], &mut y);
            let e = (0..n)
                .map(|r| (y[r] - cimg[i * n + r]).norm())
                .fold(0.0, f64::max);
            let cn: f64 = cimg[i * n..(i + 1) * n]
                .iter()
                .map(|v| v.norm_sqr())
                .sum::<f64>()
                .sqrt();
            eprintln!("rec {i}: |Au - c| {e:.2e}, |c| {cn:.3}");
        }
    }
    let second: Vec<C> = (0..n * s)
        .map(|i| c(((i * 7) % 11) as f64 - 5.0, ((i * 3) % 13) as f64 - 6.0))
        .collect();
    let plain = gmres_block(
        &a,
        &second,
        s,
        &NoPreconditioner,
        &set.clone().with_tol(tol),
        None,
        None,
    )
    .unwrap();
    let mut d = BlockGmres::new(&a, &second, s, &NoPreconditioner, &set, None)
        .unwrap()
        .deflating(&rec)
        .unwrap();
    let r = d.solve(tol, None).unwrap();
    assert!(r.converged && plain.converged);
    assert!(
        r.iters < plain.iters,
        "deflating {} against plain {}",
        r.iters,
        plain.iters
    );
    for k in 0..s {
        let col = k * n..(k + 1) * n;
        let (rn, bound) = residual_bound(&a, &r.x[col.clone()], &second[col], tol);
        assert!(rn <= bound, "column {k}: {rn}");
    }
    assert!(
        d.harvest(20).is_none(),
        "a deflating solve harvests nothing"
    );
    struct OneAtATime<'a>(&'a CscMatrix<C>);
    impl LinearOperator<C> for OneAtATime<'_> {
        fn n(&self) -> usize {
            self.0.n
        }
        fn apply(&self, x: &[C], y: &mut [C]) {
            self.0.symv(x, y);
        }
        fn apply_block(&self, x: &[C], y: &mut [C], s: usize) {
            assert_eq!(s, 1, "applied {s} columns at once");
            self.0.symv(x, y);
        }
    }
    let one = OneAtATime(&a);
    let mut g1 = BlockGmres::new(&one, &first[..n], 1, &NoPreconditioner, &set, None).unwrap();
    let one_column = g1.solve(tol, None).unwrap();
    assert!(one_column.converged);
    assert!(g1.harvest(20).is_some_and(|r| r.dim() > 1));
    let plan =
        set.block_bytes::<C>(n, s, d.deepest()) + set.deflation_bytes::<C>(n, s, d.rec_dim());
    assert!(
        plan >= d.resident_bytes(),
        "plan {plan} held {}",
        d.resident_bytes()
    );
}

#[test]
fn block_gmres_fixed_preconditioner_keeps_one_basis() {
    use crate::numeric::lu::LuSolver;
    let a = unsym_grid(14);
    let n = a.n;
    let s = 4;
    let b = ladder_rhs(n, s);
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let set = crate::KrylovSettings::default()
        .with_max_iter(400)
        .with_restart(12);
    let tol = 1e-11;
    let mut flex = BlockGmres::new(&a, &b, s, &lu, &set, None).unwrap();
    let rf = flex.solve(tol, None).unwrap();
    let fixed_set = set.clone().with_flexible(false);
    let mut fixed = BlockGmres::new(&a, &b, s, &lu, &fixed_set, None).unwrap();
    let rx = fixed.solve(tol, None).unwrap();
    assert!(
        rf.converged && rx.converged,
        "{:?} {:?}",
        rf.final_res,
        rx.final_res
    );
    assert_eq!(rf.iters, rx.iters);
    let inv = inverse_norm(&a);
    for k in 0..s {
        let col = k * n..(k + 1) * n;
        let d = two((0..n).map(|i| (rf.x[k * n + i] - rx.x[k * n + i]).norm()));
        let gap = residual_bound(&a, &rf.x[col.clone()], &b[col.clone()], tol).1
            + residual_bound(&a, &rx.x[col.clone()], &b[col], tol).1;
        assert!(d <= inv * gap, "column {k}: {d}");
    }
    assert!(fixed.resident_bytes() < flex.resident_bytes());
    let plan = fixed_set.block_bytes::<C>(n, s, fixed.deepest());
    assert!(
        plan >= fixed.resident_bytes(),
        "plan {plan} held {}",
        fixed.resident_bytes()
    );
    assert!(plan < set.block_bytes::<C>(n, s, fixed.deepest()));
}

#[test]
fn block_gmres_fixed_preconditioner_harvests() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(20, c(0.6, 0.02), c(-1.0, 0.0));
    let n = a.n;
    let s = 3;
    let set = crate::KrylovSettings::default()
        .with_max_iter(2000)
        .with_restart(600)
        .with_flexible(false);
    let first = ladder_rhs(n, s);
    let mut g = BlockGmres::new(&a, &first, s, &NoPreconditioner, &set, None).unwrap();
    let harvested = g.solve(1e-10, None).unwrap();
    assert!(harvested.converged);
    let rec = g.harvest(20).expect("open cycles to harvest");
    let second: Vec<C> = (0..n * s)
        .map(|i| c(((i * 7) % 11) as f64 - 5.0, ((i * 3) % 13) as f64 - 6.0))
        .collect();
    let plain = gmres_block(
        &a,
        &second,
        s,
        &NoPreconditioner,
        &set.clone().with_tol(1e-10),
        None,
        None,
    )
    .unwrap();
    let mut d = BlockGmres::new(&a, &second, s, &NoPreconditioner, &set, None)
        .unwrap()
        .deflating(&rec)
        .unwrap();
    let r = d.solve(1e-10, None).unwrap();
    assert!(r.converged && plain.converged);
    assert!(
        r.iters < plain.iters,
        "deflating {} against plain {}",
        r.iters,
        plain.iters
    );
}

#[test]
fn block_gmres_room_runs_the_columns_in_waves() {
    let c = |re, im| Complex::new(re, im);
    let (diag, off) = (c(1.2, 0.05), c(-1.0, 0.0));
    let a = grid(14, diag, off);
    let n = a.n;
    let s = 6;
    let b = ladder_rhs(n, s);
    let lu = NoPreconditioner;
    let set = crate::KrylovSettings::default()
        .with_max_iter(4000)
        .with_restart(40)
        .with_flexible(false);
    let tol = 1e-10;
    let mut free = BlockGmres::new(&a, &b, s, &lu, &set, None).unwrap();
    let rf = free.solve(tol, None).unwrap();
    let column = set.block_bytes::<C>(n, 1, 41) - set.block_bytes::<C>(n, 1, 0);
    let room = set.block_bytes::<C>(n, s, 0) + 2 * column;
    let tight = set.clone().with_room(room);
    let mut g = BlockGmres::new(&a, &b, s, &lu, &tight, None).unwrap();
    let (mut peak, mut waited) = (0, 0);
    let mut closings = 0;
    let mut mon = |p: &BlockProgress<'_>| {
        peak = peak.max(p.resident);
        waited = waited.max(p.waiting);
        assert!(p.closing.iter().all(|c| p.checked.contains(c)));
        closings += p.closing.len();
        true
    };
    let r = g.solve(tol, Some(&mut mon)).unwrap();
    assert!(closings <= r.closed);
    eprintln!(
        "room: iters {} -> {}, closed {}, peak {peak} room {room}",
        rf.iters, r.iters, r.closed
    );
    assert!(rf.converged && r.converged, "{:?}", r.final_res);
    assert!(
        r.closed > 0 && waited > 0,
        "closed {} waited {waited}",
        r.closed
    );
    let step = (n * std::mem::size_of::<C>()) as u64;
    assert!(peak <= room + step, "peak {peak} room {room}");
    assert!(r.iters <= s / 2 * rf.iters, "{} {}", r.iters, rf.iters);
    assert!(tight.block_bytes::<C>(n, s, 41) <= room);
    for k in 0..s {
        let col = k * n..(k + 1) * n;
        for (name, x) in [("free", &rf.x), ("within", &r.x)] {
            let (rn, bound) = residual_bound(&a, &x[col.clone()], &b[col.clone()], tol);
            assert!(rn <= bound, "{name} column {k}: residual {rn:e}");
        }
    }
}

#[derive(Default)]
struct Widest(std::sync::atomic::AtomicUsize);

impl Preconditioner<C> for Widest {
    fn apply(&self, r: &[C], z: &mut [C]) -> Result<(), RslabError> {
        self.apply_block(r, z, 1, r.len())
    }
    fn apply_block(&self, r: &[C], z: &mut [C], s: usize, n: usize) -> Result<(), RslabError> {
        self.0.fetch_max(s, std::sync::atomic::Ordering::Relaxed);
        z[..n * s].copy_from_slice(&r[..n * s]);
        Ok(())
    }
}

#[test]
fn block_gmres_within_a_probe_runs_the_columns_in_waves() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let c = |re, im| Complex::new(re, im);
    let (diag, off) = (c(1.2, 0.05), c(-1.0, 0.0));
    let a = grid(14, diag, off);
    let n = a.n;
    let s = 6;
    let b = ladder_rhs(n, s);
    let set = crate::KrylovSettings::default()
        .with_max_iter(4000)
        .with_restart(40);
    let tol = 1e-10;
    let mut free = BlockGmres::new(&a, &b, s, &NoPreconditioner, &set, None).unwrap();
    let rf = free.solve(tol, None).unwrap();
    assert!(rf.converged);

    let unknown = || -> Option<u64> { None };
    let mut g = BlockGmres::new(&a, &b, s, &NoPreconditioner, &set, None)
        .unwrap()
        .within(&unknown);
    let r = g.solve(1e-10, None).unwrap();
    assert_eq!((r.iters, r.closed), (rf.iters, 0));
    assert_eq!(r.x, rf.x);

    let asked = AtomicUsize::new(0);
    let step = (2 * n * std::mem::size_of::<C>()) as u64;
    let probe = || {
        asked.fetch_add(1, Ordering::Relaxed);
        Some(3 * step)
    };
    let pc = Widest::default();
    let mut g = BlockGmres::new(&a, &b, s, &pc, &set, None)
        .unwrap()
        .within(&probe);
    let mut waited = 0;
    let mut mon = |p: &BlockProgress<'_>| {
        waited = waited.max(p.waiting);
        true
    };
    let r = g.solve(tol, Some(&mut mon)).unwrap();
    eprintln!(
        "within: iters {} -> {}, closed {}, probe asked {}",
        rf.iters,
        r.iters,
        r.closed,
        asked.load(Ordering::Relaxed)
    );
    assert!(r.converged, "{:?}", r.final_res);
    assert!(pc.0.load(Ordering::Relaxed) <= 3 && waited == s - 1);
    assert!(
        (s - 1..=s * (s - 1) / 2).contains(&r.closed),
        "closed {}",
        r.closed
    );
    assert!(asked.load(Ordering::Relaxed) > 2 * r.iters);
    for k in 0..s {
        let col = k * n..(k + 1) * n;
        for (name, x) in [("free", &rf.x), ("within", &r.x)] {
            let (rn, bound) = residual_bound(&a, &x[col.clone()], &b[col.clone()], tol);
            assert!(rn <= bound, "{name} column {k}: residual {rn:e}");
        }
    }
}

struct BasisRun {
    iters: usize,
    converged: bool,
    worst: f64,
    first: f64,
    resident: u64,
    plan: u64,
    x: Vec<C>,
}

fn basis_run<A: LinearOperator<C> + ?Sized, M: Preconditioner<C> + ?Sized>(
    a: &A,
    b: &[C],
    s: usize,
    pc: &M,
    set: &crate::KrylovSettings,
    tol: f64,
) -> BasisRun {
    let n = a.n();
    let mut g = BlockGmres::new(a, b, s, pc, set, None).unwrap();
    let mut first = None;
    let mut mon = |p: &BlockProgress<'_>| {
        if p.iterations > 0 && first.is_none() {
            let r = p
                .checked
                .iter()
                .map(|&c| p.residuals[c].unwrap_or(f64::INFINITY));
            first = Some(r.fold(0.0, f64::max));
        }
        true
    };
    let r = g.solve(tol, Some(&mut mon)).unwrap();
    BasisRun {
        iters: r.iters,
        converged: r.converged,
        worst: r.final_res.iter().copied().fold(0.0, f64::max),
        first: first.unwrap_or(f64::NAN),
        resident: g.resident_bytes(),
        plan: set.block_bytes::<C>(n, s, g.deepest()),
        x: r.x,
    }
}

fn column_gap(x: &[C], y: &[C], n: usize) -> f64 {
    (0..x.len() / n)
        .map(|k| {
            let xn = x[k * n..(k + 1) * n]
                .iter()
                .map(|v| v.norm_sqr())
                .sum::<f64>();
            let d = (0..n)
                .map(|i| (x[k * n + i] - y[k * n + i]).norm_sqr())
                .sum::<f64>();
            (d / xn).sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
fn block_gmres_single_basis_reaches_the_tolerance() {
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let u = unsym_grid(20);
    let h = grid(14, c(1.2, 0.05), c(-1.0, 0.0));
    let diag = LuSolver::factor(&diagonal_part(&u), &SolverSettings::default()).unwrap();
    let low = LowPrecisionLu::factor(&u, &SolverSettings::default()).unwrap();
    let s = 4;
    type Case<'a> = (
        &'a str,
        &'a dyn LinearOperator<C>,
        &'a dyn Preconditioner<C>,
        usize,
    );
    let cases: [Case<'_>; 4] = [
        ("unsym", &u, &NoPreconditioner, 30),
        ("unsym diag-lu", &u, &diag, 30),
        ("helmholtz", &h, &NoPreconditioner, 40),
        ("unsym c32-lu", &u, &low, 30),
    ];
    for (name, a, pc, m) in cases {
        let n = a.n();
        let b = ladder_rhs(n, s);
        for flexible in [true, false] {
            let set = crate::KrylovSettings::default()
                .with_max_iter(4000)
                .with_restart(m)
                .with_flexible(flexible);
            let single = set.clone().with_basis_single(true);
            for tol in [1e-8, 1e-10] {
                let f = basis_run(a, &b, s, pc, &set, tol);
                let g = basis_run(a, &b, s, pc, &single, tol);
                let gap = column_gap(&f.x, &g.x, n);
                eprintln!(
                    "{name} flexible {flexible} tol {tol:e}: iters {} -> {}, first check \
                     {:.1e} -> {:.1e}, final {:.1e} -> {:.1e}, gap {gap:.1e}, resident {} -> {}",
                    f.iters, g.iters, f.first, g.first, f.worst, g.worst, f.resident, g.resident
                );
                assert!(
                    f.converged && g.converged,
                    "{name}: {} {}",
                    f.worst,
                    g.worst
                );
                assert!(
                    g.iters <= f.iters + f.iters / 4 + 2,
                    "{name}: {} {}",
                    f.iters,
                    g.iters
                );
                assert!(gap <= 10.0 * tol, "{name}: gap {gap:e}");
                assert!(
                    g.plan >= g.resident,
                    "{name}: plan {} held {}",
                    g.plan,
                    g.resident
                );
                assert!(g.resident < f.resident);
            }
            let bases = |t: &crate::KrylovSettings| {
                t.block_bytes::<C>(n, s, m + 1) - t.block_bytes::<C>(n, s, 0)
            };
            assert_eq!(bases(&set), 2 * bases(&single));
        }
    }
}

#[test]
fn block_gmres_single_basis_cycle_stalls_at_its_rounding() {
    let u = unsym_grid(20);
    let n = u.n;
    let b = ladder_rhs(n, 1);
    let edge = crate::constants::SINGLE_MARGIN * f32::EPSILON as f64 / 2.0;
    for flexible in [true, false] {
        let set = crate::KrylovSettings::default()
            .with_max_iter(4000)
            .with_restart(300)
            .with_flexible(flexible);
        let tol = 1e-13;
        let f = basis_run(&u, &b, 1, &NoPreconditioner, &set, tol);
        let g = basis_run(
            &u,
            &b,
            1,
            &NoPreconditioner,
            &set.with_basis_single(true),
            tol,
        );
        eprintln!(
            "flexible {flexible}: iters {} -> {}, first check {:.1e} -> {:.1e}, final {:.1e} -> {:.1e}",
            f.iters, g.iters, f.first, g.first, f.worst, g.worst
        );
        assert!(f.converged && g.converged, "{} {}", f.worst, g.worst);
        assert!(
            f.first <= tol && f.first * 1e4 < g.first,
            "{} {}",
            f.first,
            g.first
        );
        assert!(g.first <= edge / 16.0, "{}", g.first);
        assert!(g.iters <= f.iters + f.iters / 2, "{} {}", f.iters, g.iters);
    }
}

#[test]
fn block_gmres_single_basis_continues_a_ladder() {
    let u = unsym_grid(20);
    let n = u.n;
    let s = 4;
    let b = ladder_rhs(n, s);
    let set = crate::KrylovSettings::default()
        .with_max_iter(4000)
        .with_restart(80)
        .with_flexible(false);
    let single = set.clone().with_basis_single(true);
    let mut f = BlockGmres::new(&u, &b, s, &NoPreconditioner, &set, None).unwrap();
    let mut g = BlockGmres::new(&u, &b, s, &NoPreconditioner, &single, None).unwrap();
    let (mut fi, mut gi) = (0, 0);
    for tol in [1e-6, 1e-8, 1e-10] {
        let (rf, rg) = (f.solve(tol, None).unwrap(), g.solve(tol, None).unwrap());
        eprintln!("ladder tol {tol:e}: iters {} -> {}", rf.iters, rg.iters);
        assert!(rf.converged && rg.converged, "{:?}", rg.final_res);
        assert!(column_gap(&rf.x, &rg.x, n) <= 10.0 * tol);
        if crate::single_for::<C>(tol) {
            assert_eq!(rf.iters, rg.iters);
        }
        fi += rf.iters;
        gi += rg.iters;
    }
    assert!(gi <= fi + 4, "{fi} {gi}");
}

#[test]
fn block_gmres_lowers_its_bases_before_waves() {
    use crate::numeric::lu::LuSolver;
    let u = unsym_grid(20);
    let n = u.n;
    let s = 4;
    let b = ladder_rhs(n, s);
    let pc = LuSolver::factor(&diagonal_part(&u), &SolverSettings::default()).unwrap();
    let m = 10;
    let deep = m + 1;
    let set = crate::KrylovSettings::default()
        .with_max_iter(4000)
        .with_restart(m)
        .with_downgrade(true);
    let flexible = set.basis();
    let fixed = BlockBasis {
        flexible: false,
        single: false,
    };
    let single = BlockBasis {
        flexible: false,
        single: true,
    };
    let plan = |basis: BlockBasis, d: usize| {
        set.clone()
            .with_flexible(basis.flexible)
            .with_basis_single(basis.single)
            .with_downgrade(false)
            .block_bytes::<C>(n, s, d)
    };
    let column = |basis: BlockBasis| (plan(basis, deep) - plan(basis, 0)) / s as u64;
    for tol in [1e-6, 1e-10] {
        let today = set.clone().with_downgrade(false);
        let rf = BlockGmres::new(&u, &b, s, &pc, &today, None)
            .unwrap()
            .solve(tol, None)
            .unwrap();
        assert!(rf.converged && rf.lowered.is_empty());
        let lowest = if crate::single_for::<C>(tol) {
            vec![fixed, single]
        } else {
            vec![fixed]
        };
        let bottom = *lowest.last().unwrap();
        let mut cases = vec![
            (plan(flexible, deep), vec![], false),
            (plan(fixed, deep), vec![fixed], false),
        ];
        if crate::single_for::<C>(tol) {
            cases.push((plan(single, deep), vec![fixed, single], false));
        }
        cases.push((plan(bottom, 0) + 2 * column(bottom), lowest.clone(), true));
        for (room, path, waves) in cases {
            let tight = set.clone().with_tol(tol).with_room(room);
            let mut g = BlockGmres::new(&u, &b, s, &pc, &tight, None).unwrap();
            let mut peak = 0;
            let mut mon = |p: &BlockProgress<'_>| {
                peak = peak.max(p.resident);
                assert_eq!(p.basis, p.lowered.last().map_or(flexible, |w| w.basis));
                true
            };
            let r = g.solve(tol, Some(&mut mon)).unwrap();
            let gap = column_gap(&rf.x, &r.x, n);
            eprintln!(
                "downgrade tol {tol:e} room {room}: iters {} -> {}, lowered {:?}, closed {}, \
                 peak {peak}, gap {gap:.1e}",
                rf.iters, r.iters, r.lowered, r.closed
            );
            assert!(r.converged, "{:?}", r.final_res);
            let went: Vec<BlockBasis> = r.lowered.iter().map(|w| w.basis).collect();
            assert_eq!(went, path);
            assert_eq!(r.basis, *path.last().unwrap_or(&flexible));
            assert_eq!(g.basis(), r.basis);
            assert_eq!(r.closed > 0, waves, "closed {}", r.closed);
            let step = r.basis.layer_bytes::<C>(n);
            assert!(peak <= room + step, "peak {peak} room {room}");
            if path.is_empty() {
                assert_eq!((r.iters, &r.x), (rf.iters, &rf.x));
            }
            assert!(gap <= 10.0 * tol, "gap {gap:e}");
            let within = tight.block_bytes::<C>(n, s, deep);
            if waves {
                assert!(within <= room, "plan {within} room {room}");
            } else {
                assert_eq!(within, plan(r.basis, deep));
            }
        }
    }
}

#[test]
fn block_gmres_restarts_on_a_changed_preconditioner() {
    use crate::numeric::lu::LuSolver;
    use std::sync::atomic::{AtomicBool, Ordering};
    let a = unsym_grid(20);
    let n = a.n;
    let s = 3;
    let b = ladder_rhs(n, s);
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let base = crate::KrylovSettings::default()
        .with_max_iter(4000)
        .with_restart(200);
    let weak_alone = gmres_block(
        &a,
        &b,
        s,
        &NoPreconditioner,
        &base.clone().with_tol(1e-10),
        None,
        None,
    )
    .unwrap();
    let probe = 6usize;
    for (name, set) in [
        ("flexible", base.clone()),
        ("V alone", base.clone().with_flexible(false)),
    ] {
        let good = AtomicBool::new(false);
        let pc = FnPreconditioner::new(|r: &[C], z: &mut [C], k: usize| {
            if good.load(Ordering::Relaxed) {
                let x = lu.solve_many(&r[..k * n], k)?;
                z[..k * n].copy_from_slice(&x);
            } else {
                z[..k * n].copy_from_slice(&r[..k * n]);
            }
            Ok(())
        });
        let mut g = BlockGmres::new(&a, &b, s, &pc, &set, None).unwrap();
        let mut seen = Vec::new();
        let mut stop = |p: &StepProgress<'_>| {
            assert_eq!(p.estimates.len(), s);
            seen.push(p.worst);
            p.iterations < probe
        };
        let r1 = g.solve_watched(1e-10, None, Some(&mut stop)).unwrap();
        assert_eq!(r1.stop, StopReason::Stalled, "{name}");
        assert_eq!(r1.iters, probe, "{name}");
        assert_eq!(seen.len(), probe, "{name}");
        assert!(
            seen.windows(2).all(|w| w[1] <= w[0] * estimate_growth(U64)),
            "{name}: {seen:?}"
        );
        g.restart();
        good.store(true, Ordering::Relaxed);
        let tol = 1e-10;
        let r2 = g.solve(tol, None).unwrap();
        assert!(r2.converged, "{name}: {:?}", r2.final_res);
        assert!(r1.iters + r2.iters < weak_alone.iters, "{name}");
        for k in 0..s {
            let col = k * n..(k + 1) * n;
            let (rn, bound) = residual_bound(&a, &r2.x[col.clone()], &b[col], tol);
            assert!(rn <= bound, "{name}, column {k}: {rn}");
        }
    }
}
