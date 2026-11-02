use super::util::*;
use super::*;
use crate::error::RslabError;
use crate::numeric::ldlt::LdltSolver;
use crate::numeric::settings::SolverSettings;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use num_complex::Complex;

type C = Complex<f64>;

/// 2D complex-symmetric Helmholtz-style grid (lower triangle).
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

#[test]
fn cocg_unpreconditioned_solves_complex_symmetric() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(8, c(4.0, 0.5), c(-1.0, 0.1));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let res = cocg(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(2000),
    )
    .unwrap();
    assert!(res.converged, "COCG should converge, res={}", res.final_res);
    assert_eq!(res.stop, StopReason::Converged);
    // Verify against the actual residual.
    let mut ax = vec![C::default(); n];
    a.symv(&res.x, &mut ax);
    let r = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-7, "residual {}", r);

    // Starve the budget: the same solve capped at one iteration must report
    // `MaxIter`, not a false `Converged`.
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

    // Unpreconditioned COCR converges to the true solution.
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

    // RLA-preconditioned COCR collapses to a handful of iterations.
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
    // Genuinely unsymmetric complex 2D grid (right != left couplings).
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

    // Unpreconditioned GMRES converges (well-conditioned).
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

    // LU factor as preconditioner -> 1-2 iterations.
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let pre = gmres(
        &a,
        &b,
        &lu,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(200)
            .with_restart(40),
        None,
    )
    .unwrap();
    assert!(pre.converged, "preconditioned GMRES res={}", pre.final_res);
    assert!(
        pre.iters <= 3,
        "LU-preconditioned GMRES iters {}",
        pre.iters
    );
    // Verify the true residual.
    let mut y = vec![C::default(); n];
    a.matvec(&pre.x, &mut y);
    let res = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(res < 1e-8, "residual {}", res);
}

#[test]
fn gmres_singular_operator_breaks_down_without_nan() {
    // Rank-deficient operator: A = diag(1, 1, 0) is singular and
    // `b = (1,1,1)` has a component in the null space (e_2), so GMRES cannot
    // drive the residual to zero - it stagnates. The Krylov subspace is
    // A-invariant with a *singular* restriction (eigenvalue 0), so the upper-
    // triangular Hessenberg factor acquires a ~0 diagonal. The unguarded
    // back-substitution would divide by it and emit NaN/Inf into `x` and the
    // reported residual; the guard must instead truncate to the well-
    // conditioned block, giving a deterministic breakdown: a finite iterate and
    // a truthful non-convergence report.
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    // Only the (0,0) and (1,1) entries; row/col 2 is all-zero -> A e_2 = 0.
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
    // No NaN/Inf reached the solution or the residual.
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
    // Deterministic breakdown: reported as non-converged with a sane residual
    // (the singular direction pins the relative residual near 1/sqrt3 ~ 0.577 - it
    // is bounded well below the blow-up an unguarded divide would produce).
    assert!(
        !res.converged,
        "singular system must not report convergence"
    );
    assert!(
        res.final_res > 1e-12 && res.final_res <= 1.0,
        "residual not in the sane breakdown range: {}",
        res.final_res
    );
}

/// Build the genuinely unsymmetric complex grid used by the GMRES tests.
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

/// The diagonal of `a`: its factor is a deliberately weak preconditioner.
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

/// Wraps a preconditioner and counts every scalar `apply` (the `M^-1` solves).
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
    // FGMRES: the preconditioned basis `Z` is kept, so the restart
    // update `x += Z y` costs **no** extra `M^-1` solve. The loop applies `M^-1`
    // exactly once per inner iteration and never at the restart, so over a
    // multi-cycle solve the total preconditioner-apply count equals the total
    // iteration count. Plain right-preconditioned GMRES would spend one extra
    // `M^-1` per cycle (rebuilding `M^-1(V y)`), i.e. `iters + n_cycles`.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(10); // n = 100
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    // Weak preconditioner so the solve needs several restart cycles at a
    // short restart length - exercising the per-cycle update path.
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
    // Multiple restart cycles actually occurred (proves the saving is nonzero).
    assert!(
        res.iters > restart,
        "expected multiple cycles, iters={}",
        res.iters
    );
    // FGMRES: exactly one apply per inner iteration, none at restart.
    assert_eq!(
        applies, res.iters,
        "FGMRES precond applies {} must equal iters {} (no per-cycle extra solve)",
        applies, res.iters
    );
}

#[test]
fn gmres_warm_start_cuts_total_iterations_on_related_sequence() {
    // Warm start: a sequence of related systems `A x = b_k` with a
    // slowly rotating right-hand side. Cold-starting every solve from 0 pays
    // the full iteration count each time; seeding each solve with the previous
    // solution (which is close, because the RHS barely moved) collapses the
    // per-solve count. Total iterations must drop by a clear margin.
    let a = unsym_grid(12); // n = 144
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    // Two fixed directions; b_k interpolates between them by a small angle step.
    let b0: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let b1: Vec<C> = (0..n)
        .map(|i| c(((i * 3) % 7) as f64 - 3.0, ((i % 3) as f64) - 1.0))
        .collect();
    let steps = 10;
    let (tol, maxit, restart) = (1e-8, 4000, 60);

    let bk = |k: usize| -> Vec<C> {
        let th = 5e-5 * k as f64; // slow rotation
        let (ct, st) = (th.cos(), th.sin());
        (0..n)
            .map(|i| b0[i] * c(ct, 0.0) + b1[i] * c(st, 0.0))
            .collect()
    };

    // Cold: every solve from x0 = 0.
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

    // Warm: first solve from 0, each subsequent seeded with the previous x.
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

    // A meaningful reduction (well beyond noise): warm start must cut the total
    // iteration count by at least 30% over the sequence.
    assert!(
        (warm_total as f64) < 0.7 * (cold_total as f64),
        "warm start did not help enough: cold={cold_total}, warm={warm_total}"
    );
}

#[test]
fn gmres_block_single_rhs_matches_scalar_gmres() {
    // s = 1 block GMRES reduces to the single-RHS path (same Arnoldi, same
    // Givens, default block apply = one single apply): same solution to the
    // requested tolerance, same iteration count up to a +/-1 boundary effect.
    // The paths are NOT bit-identical - block uses CGS2, single uses MGS+DGKS,
    // so the projections sum in a different order and the true residual can
    // straddle `tol` by a rounding ULP. This is documented as a design point in
    // the module-level "Orthogonalization" note, not a defect.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(8);
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let single = gmres(
        &a,
        &b,
        &lu,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(200)
            .with_restart(40),
        None,
    )
    .unwrap();
    let blk = gmres_block(
        &a,
        &b,
        1,
        &lu,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(200)
            .with_restart(40),
        None,
        None,
    )
    .unwrap();
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
    assert!(diff < 1e-7, "block(s=1) solution differs by {diff}");
}

#[test]
fn gmres_block_multi_rhs_solves_each_column() {
    // Several distinct right-hand sides solved in one block iteration; every
    // column must reach its own system's true residual.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(10);
    let n = a.n;
    let s = 5;
    // Column-major nxs block: RHS `k` is shifted/scaled so columns differ.
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let res = gmres_block(
        &a,
        &bblk,
        s,
        &lu,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(200)
            .with_restart(40),
        None,
        None,
    )
    .unwrap();
    assert!(
        res.converged,
        "block GMRES must converge; res={:?}",
        res.final_res
    );
    // True residual per column.
    for k in 0..s {
        let mut y = vec![C::default(); n];
        a.matvec(&res.x[k * n..k * n + n], &mut y);
        let r = (0..n)
            .map(|i| (y[i] - bblk[k * n + i]).norm())
            .fold(0.0, f64::max);
        assert!(r < 1e-8, "column {k} residual {r}");
    }
    // Each column must equal the single-RHS solve of that column.
    for k in 0..s {
        let single = gmres(
            &a,
            &bblk[k * n..k * n + n],
            &lu,
            &crate::KrylovSettings::default()
                .with_tol(1e-10)
                .with_max_iter(200)
                .with_restart(40),
            None,
        )
        .unwrap();
        let diff = (0..n)
            .map(|i| (res.x[k * n + i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-9,
            "column {k} differs from single solve by {diff}"
        );
    }
}

#[test]
fn gmres_block_within_cycle_deflation_shrinks_applies() {
    // Different-convergence-rate regime: a diagonal operator with
    // distinct eigenvalues, unpreconditioned, with RHS `k` supported on `k+1`
    // distinct eigenvalues. GMRES on such a RHS converges in exactly `k+1`
    // steps, so the columns finish at staggered steps within a *single* cycle.
    // The fix must (i) still solve every column exactly like single-RHS GMRES
    // and (ii) actually shrink the batched operator applies as columns deflate
    // mid-cycle - not only at restart. A counting operator records the width
    // of every `apply_block`, and we assert the panel narrows.
    use std::sync::Mutex;

    struct CountingOp<'a> {
        inner: &'a GeneralCsc<C>,
        widths: Mutex<Vec<usize>>,
    }
    impl LinearOperator<C> for CountingOp<'_> {
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

    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 8;
    let s = 4;
    // Diagonal operator with distinct entries -> the minimal polynomial degree
    // of a RHS equals the number of distinct diagonal entries in its support.
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0 + i as f64, 0.5 + 0.1 * i as f64));
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    // RHS `k` = sum of the first `k+1` unit vectors -> converges in `k+1` steps.
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..=k {
            bblk[k * n + i] = c(1.0, 0.0);
        }
    }

    let op = CountingOp {
        inner: &a,
        widths: Mutex::new(Vec::new()),
    };
    let res = gmres_block(
        &op,
        &bblk,
        s,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-12)
            .with_max_iter(200)
            .with_restart(40),
        None,
        None,
    )
    .unwrap();
    assert!(
        res.converged,
        "block GMRES must converge; res={:?}",
        res.final_res
    );

    // (i) Every column matches the single-RHS GMRES solve of that column.
    for k in 0..s {
        let single = gmres(
            &a,
            &bblk[k * n..k * n + n],
            &NoPreconditioner,
            &crate::KrylovSettings::default()
                .with_tol(1e-12)
                .with_max_iter(200)
                .with_restart(40),
            None,
        )
        .unwrap();
        let diff = (0..n)
            .map(|i| (res.x[k * n + i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-9,
            "column {k} differs from single solve by {diff}"
        );
    }

    // (ii) The batched applies actually shrank mid-cycle. Without within-cycle
    // deflation every `apply_block` would run at full width `s`; with it, later
    // steps run narrower. Assert a full-width apply happened (the first step),
    // that some apply narrowed below `s`, that the panel drained to width 1,
    // and that the total column-applies fell below the full-width bound.
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
fn gmres_block_bcgs2_bit_identical_across_thread_counts() {
    // The block-CGS2 orthogonalization reduces over fixed row-chunks folded in
    // chunk order, so the whole block solve is **bit-identical regardless of
    // the thread count** - the determinism guarantee. Solve the same block in
    // a 1-thread and an 8-thread rayon pool and require exact equality.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    // Wide enough that a chunked reduction actually spans several chunks.
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
    // The block-GMRES orthogonalization must run in a pool derived
    // from the *factor's* Threads policy, not the ambient global pool. Factor
    // with a hard cap of 2 workers; the factor then reports `Fixed(2)` as its
    // solve-phase policy, and the pool built from it caps `current_num_threads`
    // to 2 - even when the surrounding (ambient) pool is far wider. The solve
    // stays bit-identical whether run bare or inside a wide ambient pool (the
    // chunk-order reduction is thread-count independent), so the cap changes
    // only the concurrency, never the numbers.
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
    // Factor capped to exactly 2 workers -> solve policy is Fixed(2).
    let lu = LuSolver::factor(&a, &SolverSettings::default().with_threads(2)).unwrap();
    assert_eq!(
        Preconditioner::<C>::solve_threads(&lu),
        Threads::Fixed(2),
        "factor must carry its resolved solve-phase thread budget"
    );
    // The pool the orthogonalization installs caps the worker count to 2,
    // regardless of a wider ambient pool around it.
    let pool = solve_thread_pool(Preconditioner::<C>::solve_threads(&lu));
    let seen = with_threads(8, || {
        pool.as_ref().unwrap().install(rayon::current_num_threads)
    });
    assert_eq!(
        seen, 2,
        "the ortho pool must cap to the factor's 2-worker budget"
    );

    // The full solve is identical bare vs. inside a wide ambient pool: the
    // internal cap governs concurrency only, never the result.
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
    // `Threads::Ambient` factors on the current pool (no new spawn) - the
    // re-factor-in-loop path. Inside a `with_threads(2)` pool the factor must be
    // bit-identical to the normal (scoped-pool) factor: the numeric result is
    // independent of the thread policy.
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
    // The pareto-optimal embedded default: predict per matrix, never exceed 4.
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
            vv.push(c(20.0, 2.0)); // strongly diagonally dominant -> f32 factor is accurate
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
    // The f32 LU factor is a half-memory preconditioner accurate to ~1e-6
    // (the f32 apply floor). f64 GMRES reaches that floor in a couple of
    // iterations - ample for a MoM/FEM Krylov tolerance, at half the factor
    // memory. (A residual below ~1e-6 is not attainable with an f32 apply;
    // use the f64 factor for tighter tolerances.)
    let pc = LowPrecisionLu::factor(&a, &SolverSettings::default()).unwrap();
    assert!(pc.factor_nnz() > 0);
    let res = gmres(
        &a,
        &b,
        &pc,
        &crate::KrylovSettings::default()
            .with_tol(1e-6)
            .with_max_iter(200)
            .with_restart(50),
        None,
    )
    .unwrap();
    assert!(res.converged, "mixed-precision GMRES res={}", res.final_res);
    assert!(res.iters <= 6, "iters {}", res.iters);
    let mut y = vec![C::default(); n];
    a.matvec(&res.x, &mut y);
    let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-5, "residual {}", r);
}

#[test]
fn cocr_handles_indefinite_helmholtz() {
    // Indefinite complex-symmetric: high-frequency 2D Helmholtz with a
    // negative-real diagonal shift (`diag = -1 + 0.3i`). A robust
    // preconditioner (static-pivoted RLA) plus COCR must still converge.
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
    // Factor the preconditioner in Complex<f32> (half memory) but iterate in
    // f64: the solution must still reach f64-level residual, and the f32
    // factor - though approximate - keeps the iteration count tiny.
    let c = |re, im| Complex::new(re, im);
    let a = grid(14, c(4.0, 1.0), c(-1.0, 0.2));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

    let m = LowPrecisionPreconditioner::factor(&a, &SolverSettings::default()).unwrap();
    let res = cocg(
        &a,
        &b,
        &m,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(500),
    )
    .unwrap();
    assert!(res.converged, "mixed-precision COCG res={}", res.final_res);
    // A few iterations suffice; the f32 factor is a strong preconditioner.
    assert!(res.iters <= 12, "f32-preconditioned iters {}", res.iters);
    // Full f64 accuracy recovered despite the single-precision factor.
    let mut ax = vec![C::default(); n];
    a.symv(&res.x, &mut ax);
    let r = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-8, "mixed-precision residual {}", r);
    assert!(m.factor_nnz() > 0);
}

#[test]
fn rla_preconditioner_collapses_iteration_count() {
    // A complete RLA factorization is ~ A^-1, so preconditioned COCG must
    // converge in a handful of iterations - vastly fewer than without.
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

/// Strongly **non-normal** 1D convection-diffusion operator as a general
/// complex matrix: tridiagonal `diag = 2 (+ tiny damping)`, super `= -1+gamma`,
/// sub `= -1-gamma`. For `gamma != 0` it is far from normal - the classic GMRES-hard
/// regime where the residual stagnates for many steps before converging - yet
/// remains (weakly) diagonally dominant, so unpreconditioned GMRES does
/// converge, only after many iterations / restart cycles.
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

/// Wraps an operator and records the width `s` of every block apply - the
/// deflation probe: a shrinking width proves the batched
/// applies narrow as columns converge.
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
    // Unpreconditioned GMRES on a strongly non-normal operator:
    // it must survive the non-normal stagnation phase and multiple restart
    // cycles, then converge to the true solution. Exercises the restart /
    // outer-loop machinery that the diagonally dominant tests never stress.
    let a = convection_diffusion(120, 0.9);
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b: Vec<C> = (0..n).map(|i| c(((i % 5) as f64) - 2.0, 0.5)).collect();
    let restart = 20;
    let res = gmres(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-8)
            .with_max_iter(8000)
            .with_restart(restart),
        None,
    )
    .unwrap();
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
    let mut y = vec![C::default(); n];
    a.matvec(&res.x, &mut y);
    let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-6, "true residual {}", r);
}

#[test]
fn gmres_reorthogonalization_keeps_illconditioned_arnoldi_accurate() {
    // A near-defective, strongly non-normal operator (bidiagonal
    // Jordan-like block: clustered diagonal, dominant super-diagonal) drives
    // the Arnoldi vectors toward linear dependence, so a single MGS sweep
    // collapses the norm and the conditional DGKS second pass (`hn < eta*||w_0||`)
    // must fire to restore orthogonality. Asserting the trigger directly needs
    // an intrusive probe; instead we certify the *effect*: GMRES still drives
    // the true residual to `tol` and matches the exact (direct-LU) solution -
    // which it could not if the ill-conditioned basis went uncorrected.
    use crate::numeric::lu::LuSolver;
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 32;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0, 0.0)); // clustered diagonal -> non-normal, near-defective
        if i + 1 < n {
            rr.push(i);
            cc.push(i + 1);
            vv.push(c(3.0, 0.0)); // dominant super-diagonal
        }
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<C> = (0..n).map(|_| c(1.0, 0.2)).collect();
    // Long restart (single cycle) so the ill-conditioned basis is not masked by
    // a restart - the reorthogonalization alone keeps it usable.
    let res = gmres(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(4000)
            .with_restart(n),
        None,
    )
    .unwrap();
    assert!(
        res.converged,
        "reorth must keep GMRES converging, res={}",
        res.final_res
    );
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let xstar = lu.solve(&b).unwrap();
    let diff = (0..n)
        .map(|i| (res.x[i] - xstar[i]).norm())
        .fold(0.0, f64::max);
    assert!(
        diff < 1e-6,
        "GMRES solution off the direct solve by {} (lost orthogonality?)",
        diff
    );
}

#[test]
fn gmres_happy_breakdown_on_eigenvector_rhs() {
    // Happy breakdown: `b` is an eigenvector of the operator, so
    // the Krylov space `K_1 = span{b}` is already `A`-invariant. The Arnoldi
    // step-1 subdiagonal `h[1][0]` is exactly `0` (the invariant-subspace
    // branch), and GMRES must produce the exact solution in a single iteration.
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 12;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0 + i as f64, 0.5 - 0.05 * i as f64)); // distinct diagonal
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    // `e_0` is an eigenvector (eigenvalue `a[0][0]`); its Krylov space is 1-D.
    let mut b = vec![C::default(); n];
    b[0] = c(1.0, 0.0);
    let res = gmres(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-12)
            .with_max_iter(50)
            .with_restart(30),
        None,
    )
    .unwrap();
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
    let mut y = vec![C::default(); n];
    a.matvec(&res.x, &mut y);
    let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-12, "true residual {}", r);
}

#[test]
fn gmres_block_nearby_factor_multirate_deflation() {
    // Multi-rate within-cycle deflation under a genuine
    // **factor-based** preconditioner. The operator is diagonal with
    // distinct entries `d_i`; the preconditioner is an LU factor of a
    // *different* diagonal matrix `diag(p_i)` - a deliberately imperfect
    // approximate inverse, so the preconditioned operator `M^-1A = diag(d_i/p_i)`
    // still has distinct eigenvalues. Right-hand side `k` is supported on the
    // first `k+1` unit vectors, so its GMRES converges in **exactly** `k+1`
    // steps: the columns finish at staggered steps *within one cycle*. The
    // within-cycle deflation must finalize each fast column and shrink the
    // batched applies to the still-active width, draining the panel to 1 - while
    // every column still matches its single-RHS solve.
    use crate::numeric::lu::LuSolver;
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 8;
    let s = 4;
    // Operator D = diag(d_i), d_i distinct.
    let (mut dr, mut dc, mut dv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        dr.push(i);
        dc.push(i);
        dv.push(c(2.0 + i as f64, 0.3));
    }
    let a = GeneralCsc::<C>::from_triplets(n, &dr, &dc, &dv).unwrap();
    // Preconditioning matrix P = diag(p_i), p_i chosen so d_i/p_i stay distinct
    // (p_i = 1+i => ratios 2, 1.5, 1.33, ... all different): an imperfect factor.
    let (mut pr, mut pc_, mut pv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        pr.push(i);
        pc_.push(i);
        pv.push(c(1.0 + i as f64, 0.1));
    }
    let pmat = GeneralCsc::<C>::from_triplets(n, &pr, &pc_, &pv).unwrap();
    let lu = LuSolver::factor(&pmat, &SolverSettings::default()).unwrap();

    // RHS k = sum of the first k+1 unit vectors -> converges in exactly k+1 steps.
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
    let res = gmres_block(
        &op,
        &bblk,
        s,
        &lu,
        &crate::KrylovSettings::default()
            .with_tol(1e-10)
            .with_max_iter(200)
            .with_restart(40),
        None,
        None,
    )
    .unwrap();
    assert!(
        res.converged,
        "block GMRES must converge; res={:?}",
        res.final_res
    );

    // Every column equals its single-RHS solve (deflation must not corrupt it).
    for k in 0..s {
        let single = gmres(
            &a,
            &bblk[k * n..k * n + n],
            &lu,
            &crate::KrylovSettings::default()
                .with_tol(1e-10)
                .with_max_iter(200)
                .with_restart(40),
            None,
        )
        .unwrap();
        let diff = (0..n)
            .map(|i| (res.x[k * n + i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-8,
            "column {k} differs from single solve by {diff}"
        );
    }

    // Within-cycle deflation fired: the panel opened at full width `s`, narrowed
    // as fast columns deflated, and drained to a single active column.
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

/// A general complex diagonal operator with a prescribed spectrum - the
/// canonical GMRES-DR / GCRO-DR demonstrator: a handful of tiny eigenvalues
/// (which restarted GMRES keeps re-discovering and discarding) sitting far
/// below a cluster of larger ones. Unpreconditioned restarted GMRES stagnates
/// on the small cluster; deflating it is exactly what recycling does.
fn diag_op(eigs: &[C]) -> crate::sparse::general::GeneralCsc<C> {
    use crate::sparse::general::GeneralCsc;
    let n = eigs.len();
    let idx: Vec<usize> = (0..n).collect();
    GeneralCsc::<C>::from_triplets(n, &idx, &idx, eigs).unwrap()
}

/// A spectrum with `n_small` tiny eigenvalues far below a spread cluster.
fn stagnation_spectrum(n: usize, n_small: usize) -> Vec<C> {
    let c = |re: f64, im: f64| Complex::new(re, im);
    (0..n)
        .map(|i| {
            if i < n_small {
                // Tiny, tightly clustered near the origin - the stagnation drivers.
                c(0.01 + 0.004 * i as f64, 0.002 * i as f64)
            } else {
                // Larger eigenvalues spread over [1, 11], mildly complex.
                let t = (i - n_small) as f64 / (n - n_small) as f64;
                c(1.0 + 10.0 * t, 0.3 * (i as f64).sin())
            }
        })
        .collect()
}

#[test]
fn gmres_recycled_matches_plain_on_hard_matrix() {
    // Correctness: the recycled solve must reach the SAME solution as plain
    // FGMRES on a hard preconditioned system (weak diagonal factor, short
    // restart -> many cycles), to the same tolerance.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(12); // n = 144
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
    assert!(
        diff < 1e-7,
        "recycled solution differs from plain by {diff}"
    );
    // The recycle handle came back populated for the next solve.
    assert!(rec.active() > 0, "recycle subspace was not populated");
}

#[test]
fn gmres_recycled_within_solve_reduces_restarts() {
    // Within-solve deflated restarting: on a stagnating, restart-limited solve
    // (tiny eigenvalue cluster, unpreconditioned, short restart) carrying the
    // harmonic-Ritz subspace across restarts must cut total iterations by a
    // real margin versus plain FGMRES - on a SINGLE solve (fresh handle, no
    // cross-solve benefit).
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
    // Both hit the same true solution.
    let diff = (0..n)
        .map(|i| (plain.x[i] - recd.x[i]).norm())
        .fold(0.0, f64::max);
    assert!(diff < 1e-6, "solutions differ by {diff}");
    // Deflated restarting must shave a clear margin off the iteration count.
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
    // Cross-solve recycling on a sequence of related systems: A gets a small
    // diagonal perturbation each step and b rotates slowly. Compare total
    // iterations over the sequence for cold (x0=0), warm (x0=prev x), and
    // recycled+warm (handle carried across solves). Must order
    // recycled < warm < cold, each by a meaningful margin.
    let base = stagnation_spectrum(48, 5);
    let n = base.len();
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b0: Vec<C> = (0..n).map(|i| c(1.0, 0.15 * (i as f64).cos())).collect();
    let b1: Vec<C> = (0..n).map(|i| c(0.4 * (i as f64).sin(), 1.0)).collect();
    let steps = 8;
    let (tol, maxit, restart) = (1e-9, 6000, 12);

    // Slowly varying operator: base spectrum + eps_k on each diagonal entry.
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

    // Cold.
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
    // Warm.
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
    // Recycled (+ warm start, the intended combined use).
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
    // Meaningful reduction, not noise.
    assert!(
        (recycled as f64) < 0.7 * (cold as f64),
        "recycled ({recycled}) did not beat cold ({cold}) by a clear margin"
    );
}

#[test]
fn gmres_recycled_composes_with_warm_start() {
    // Recycle + x0 warm start together: on a related second solve, seeding from
    // the previous solution AND recycling its stagnation subspace must converge
    // to the correct solution and take no more iterations than warm-start alone.
    let eigs = stagnation_spectrum(40, 4);
    let a = diag_op(&eigs);
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b0: Vec<C> = (0..n).map(|i| c(1.0, 0.1 * (i as f64).cos())).collect();
    let b1: Vec<C> = (0..n).map(|i| c(1.0 + 0.02 * i as f64, 0.1)).collect();
    let (tol, maxit, restart) = (1e-9, 5000, 10);

    // First solve seeds both the warm start and the recycle handle.
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

    // Second, related solve: warm-only vs warm+recycle.
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
    // Same true solution.
    let mut ax = vec![C::default(); n];
    a.matvec(&warm_rec.x, &mut ax);
    let res = (0..n).map(|i| (ax[i] - b1[i]).norm()).fold(0.0, f64::max);
    assert!(res < 1e-6, "warm+recycle residual {res}");
    // Composed use is no worse than warm-start alone (typically better).
    assert!(
        warm_rec.iters <= warm_only.iters,
        "warm+recycle ({}) worse than warm-only ({})",
        warm_rec.iters,
        warm_only.iters
    );
}

#[test]
fn gmres_recycled_real_scalar_path() {
    // The real f64 field exercises the conjugate-pair reconstruction in
    // `combine_ritz` (a real diagonal has real eigenvalues, but a real
    // unsymmetric grid produces complex harmonic-Ritz pairs). Must converge to
    // the true solution - correctness of the real recycle path.
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
                vv.push(-1.8); // asymmetric => complex spectrum
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
    let r = gmres_recycled(
        &a,
        &b,
        &NoPreconditioner,
        &crate::KrylovSettings::default()
            .with_tol(1e-9)
            .with_max_iter(5000)
            .with_restart(12),
        None,
        &mut rec,
    )
    .unwrap();
    assert!(r.converged, "real recycled solve did not converge");
    let mut ax = vec![0.0f64; n];
    a.matvec(&r.x, &mut ax);
    let res = (0..n).map(|i| (ax[i] - b[i]).abs()).fold(0.0, f64::max);
    assert!(res < 1e-6, "real recycled residual {res}");
}

/// Run `f` in a fresh rayon pool of `threads` workers.
fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

/// The right-hand sides of the continuation tests: `s` distinct columns.
fn ladder_rhs(n: usize, s: usize) -> Vec<C> {
    let mut b = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            b[k * n + i] = Complex::new(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    b
}

/// A solve continued to a tighter tolerance carries on in its Krylov space:
/// one column's ladder `1e-4 -> 1e-10` is the straight solve to `1e-10`, step
/// for step and bit for bit, and every rung hands out solutions that meet it.
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
    let r1 = g.solve(1e-4, None).unwrap();
    assert!(
        r1.converged && r1.final_res[0] <= 1e-4,
        "{:?}",
        r1.final_res
    );
    let r2 = g.solve(1e-10, None).unwrap();
    assert!(
        r2.converged && r2.final_res[0] <= 1e-10,
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

/// Continuing beats restarting: on an indefinite Helmholtz grid (GMRES
/// converges superlinearly once its space resolves the eigenvalues near zero)
/// the rung `1e-6 -> 1e-10` of several columns takes fewer iterations continued
/// than a fresh solve from the first rung's solutions, and the solutions meet
/// the rung's tolerance.
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
    let r1 = g.solve(1e-6, None).unwrap();
    let r2 = g.solve(1e-10, None).unwrap();
    let fresh = gmres_block(
        &a,
        &b,
        s,
        &NoPreconditioner,
        &set.clone().with_tol(1e-10),
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
        let mut y = vec![C::default(); n];
        a.symv(&r2.x[k * n..(k + 1) * n], &mut y);
        let rn = (0..n)
            .map(|i| (y[i] - b[k * n + i]).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let bn = b[k * n..(k + 1) * n]
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(rn <= 1e-10 * bn * (1.0 + 1e-6), "column {k}: {}", rn / bn);
    }
}

/// The memory plan of a block GMRES ([`KrylovSettings::block_bytes`]) is what a
/// solve holds at the depth it reached: exactly for one column, a bound for
/// several (they stop at different depths), its bases in full precision or single.
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

/// Deflating what a solve of the same operator found slowest pays: on an
/// indefinite Helmholtz grid a second block of right-hand sides solves in fewer
/// iterations with the first solve's harmonic-Ritz subspace, to the same
/// tolerance (true residuals), and a deflating solve harvests nothing.
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
    let mut g = BlockGmres::new(&a, &first, s, &NoPreconditioner, &set, None).unwrap();
    assert!(g.solve(1e-10, None).unwrap().converged);
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
    // other right-hand sides on the same operator
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
    for k in 0..s {
        let mut y = vec![C::default(); n];
        a.symv(&r.x[k * n..(k + 1) * n], &mut y);
        let rn = (0..n)
            .map(|i| (y[i] - second[k * n + i]).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let bn = second[k * n..(k + 1) * n]
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(rn <= 1e-10 * bn * (1.0 + 1e-6), "column {k}: {}", rn / bn);
    }
    assert!(
        d.harvest(20).is_none(),
        "a deflating solve harvests nothing"
    );
    // the harvest applies the operator no wider than the solve does: a one-column solve's
    // harvest of many directions through an operator that takes one column at a time
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
    assert!(g1.solve(1e-10, None).unwrap().converged);
    assert!(g1.harvest(20).is_some_and(|r| r.dim() > 1));
    // the memory it holds is the plan's, the deflation included
    let plan =
        set.block_bytes::<C>(n, s, d.deepest()) + set.deflation_bytes::<C>(n, s, d.rec_dim());
    assert!(
        plan >= d.resident_bytes(),
        "plan {plan} held {}",
        d.resident_bytes()
    );
}

/// A fixed preconditioner (an LU) needs no preconditioned basis
/// ([`KrylovSettings::flexible`] `false`): the block GMRES keeps `V` alone and
/// forms `x0 + M^-1 (V y)`. It reaches the same tolerance in the same steps as the
/// flexible solve, its solutions agree with that solve's to the tolerance, it
/// holds less, and the memory plan is what it holds.
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
    let mut flex = BlockGmres::new(&a, &b, s, &lu, &set, None).unwrap();
    let rf = flex.solve(1e-11, None).unwrap();
    let fixed_set = set.clone().with_flexible(false);
    let mut fixed = BlockGmres::new(&a, &b, s, &lu, &fixed_set, None).unwrap();
    let rx = fixed.solve(1e-11, None).unwrap();
    assert!(
        rf.converged && rx.converged,
        "{:?} {:?}",
        rf.final_res,
        rx.final_res
    );
    assert_eq!(rf.iters, rx.iters);
    for k in 0..s {
        let xn = rf.x[k * n..(k + 1) * n]
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let d = (0..n)
            .map(|i| (rf.x[k * n + i] - rx.x[k * n + i]).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(d <= 1e-9 * xn, "column {k}: {}", d / xn);
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

/// A solve without the preconditioned basis harvests as the flexible one does:
/// its Ritz vectors `M^-1 (V g)` (here `M` the identity) span a subspace whose
/// deflation pays on further right-hand sides.
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
    assert!(g.solve(1e-10, None).unwrap().converged);
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

/// A block GMRES in a room ([`KrylovSettings::room_bytes`]) that holds two of its
/// columns at the restart's depth: the columns with the worst estimates close
/// their cycles and wait while the others run on, every column converges, the
/// solutions agree with the unbounded solve's to the tolerance, and the solve
/// never holds more than the room and a step of one column; the waves take
/// at most the unbounded iterations each.
#[test]
fn block_gmres_room_runs_the_columns_in_waves() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(14, c(1.2, 0.05), c(-1.0, 0.0));
    let n = a.n;
    let s = 6;
    let b = ladder_rhs(n, s);
    let lu = NoPreconditioner;
    let set = crate::KrylovSettings::default()
        .with_max_iter(4000)
        .with_restart(40)
        .with_flexible(false);
    let mut free = BlockGmres::new(&a, &b, s, &lu, &set, None).unwrap();
    let rf = free.solve(1e-10, None).unwrap();
    let column = set.block_bytes::<C>(n, 1, 41) - set.block_bytes::<C>(n, 1, 0);
    let room = set.block_bytes::<C>(n, s, 0) + 2 * column;
    let tight = set.clone().with_room(room);
    let mut g = BlockGmres::new(&a, &b, s, &lu, &tight, None).unwrap();
    let (mut peak, mut waited) = (0, 0);
    let mut closings = 0;
    let mut mon = |p: &BlockProgress<'_>| {
        peak = peak.max(p.resident);
        waited = waited.max(p.waiting);
        // a column closing for memory is one of the checked
        assert!(p.closing.iter().all(|c| p.checked.contains(c)));
        closings += p.closing.len();
        true
    };
    let r = g.solve(1e-10, Some(&mut mon)).unwrap();
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
    // two columns at a time: three waves, each no longer than the unbounded solve
    assert!(r.iters <= s / 2 * rf.iters, "{} {}", r.iters, rf.iters);
    assert!(tight.block_bytes::<C>(n, s, 41) <= room);
    for k in 0..s {
        let xn = rf.x[k * n..(k + 1) * n]
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let d = (0..n)
            .map(|i| (rf.x[k * n + i] - r.x[k * n + i]).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(d <= 1e-7 * xn, "column {k}: {}", d / xn);
    }
}

/// The identity preconditioner, recording the widest block it applied: the most
/// columns a flexible block GMRES stepped at once (it applies it in its steps alone).
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

/// A block GMRES within a live memory probe ([`BlockGmres::within`]) that says the
/// process may take three steps of a column more: at most three columns step at
/// once, the others waiting (one at first, more once finished columns give their
/// bases up); every column converges and the solutions agree with the unbounded
/// solve's to the tolerance.
/// A probe that knows nothing (`None`) bounds nothing: the solve is the unbounded
/// one, bit for bit.
#[test]
fn block_gmres_within_a_probe_runs_the_columns_in_waves() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let c = |re, im| Complex::new(re, im);
    let a = grid(14, c(1.2, 0.05), c(-1.0, 0.0));
    let n = a.n;
    let s = 6;
    let b = ladder_rhs(n, s);
    let set = crate::KrylovSettings::default()
        .with_max_iter(4000)
        .with_restart(40);
    let mut free = BlockGmres::new(&a, &b, s, &NoPreconditioner, &set, None).unwrap();
    let rf = free.solve(1e-10, None).unwrap();
    assert!(rf.converged);

    let unknown = || -> Option<u64> { None };
    let mut g = BlockGmres::new(&a, &b, s, &NoPreconditioner, &set, None)
        .unwrap()
        .within(&unknown);
    let r = g.solve(1e-10, None).unwrap();
    assert_eq!((r.iters, r.closed), (rf.iters, 0));
    assert_eq!(r.x, rf.x);

    let asked = AtomicUsize::new(0);
    // a step of a flexible column: a vector of `V` and one of `Z`
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
    let r = g.solve(1e-10, Some(&mut mon)).unwrap();
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
    assert!(column_gap(&rf.x, &r.x, n) <= 1e-7);
}

/// A block GMRES solve with its bases in full or single precision
/// ([`KrylovSettings::basis_single`]): what it reached and held.
struct BasisRun {
    iters: usize,
    converged: bool,
    worst: f64,
    /// The worst true residual of the first check after a step: where the cycles'
    /// solutions stood when their estimates first met the tolerance.
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

/// The largest relative difference of the columns of two solutions.
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

/// Bases stored in single precision ([`KrylovSettings::basis_single`]) reach the
/// tolerance: on the unsymmetric grid unpreconditioned and with a weak (diagonal)
/// LU, on the indefinite Helmholtz grid, and with a single-precision LU of the
/// operator, flexible and not, to `1e-8` and `1e-10`. Their solutions agree with
/// the full-precision ones to the tolerance, the bases take half the bytes and the
/// memory plan covers what the solve holds.
///
/// A cycle's true residual stalls at the rounding of its stored vectors (about
/// `3e-8` here, [`block_gmres_single_basis_cycle_stalls_at_its_rounding`]), so
/// below it a column restarts from its true residual once more: measured
/// iterations full -> single, flexible and fixed alike, `1e-8` / `1e-10`: unsym
/// 22 -> 24 / 28 -> 34, unsym diag-lu 22 -> 24 / 28 -> 35, helmholtz (restarting
/// every 40 steps anyway) 1025 -> 1076 / 1305 -> 1367, unsym c32-lu 2 -> 3 (flexible),
/// 3 -> 3 (fixed: the single-precision factor stalls the cycle at 2e-7 either way).
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
                // at most a quarter more and the one restart (35 of 28 the most)
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
            // the bases at any depth: half the bytes
            let bases = |t: &crate::KrylovSettings| {
                t.block_bytes::<C>(n, s, m + 1) - t.block_bytes::<C>(n, s, 0)
            };
            assert_eq!(bases(&set), 2 * bases(&single));
        }
    }
}

/// One cycle of single-precision bases (a restart long enough for the whole solve):
/// once its estimate passes far below single precision, its true residual stands at
/// the rounding of the stored vectors, `1.3e-8` flexible and fixed alike (full
/// bases: `3.2e-14`), below the tolerance from which the bases are stored single
/// without a cost ([`single_for`](crate::single_for), `9.5e-7`). One restart from
/// the true residual carries it on to `6.3e-14`, 49 iterations against 35.
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
        let f = basis_run(&u, &b, 1, &NoPreconditioner, &set, 1e-13);
        let g = basis_run(
            &u,
            &b,
            1,
            &NoPreconditioner,
            &set.with_basis_single(true),
            1e-13,
        );
        eprintln!(
            "flexible {flexible}: iters {} -> {}, first check {:.1e} -> {:.1e}, final {:.1e} -> {:.1e}",
            f.iters, g.iters, f.first, g.first, f.worst, g.worst
        );
        assert!(f.converged && g.converged, "{} {}", f.worst, g.worst);
        assert!(
            f.first <= 1e-13 && f.first * 1e4 < g.first,
            "{} {}",
            f.first,
            g.first
        );
        assert!(g.first <= edge / 16.0, "{}", g.first);
        assert!(g.iters <= f.iters + f.iters / 2, "{} {}", f.iters, g.iters);
    }
}

/// Single-precision bases continue a ladder of tolerances in their Krylov space:
/// `1e-6`, then `1e-8`, then `1e-10`. At `1e-6`, above the tolerance from which
/// [`single_for`](crate::single_for) stores single, the solve steps as the full one
/// does (16 iterations each); below, a cycle stalling at its rounding restarts once:
/// 8 iterations against 6 to `1e-8`, 6 against 6 on to `1e-10`.
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

/// A block GMRES that may lower its bases ([`KrylovSettings::downgrade`], here with the
/// diagonal LU, a fixed operator) in rooms that force each rung of the ladder: a room
/// that holds the flexible bases at the restart's depth changes nothing (bit for bit
/// the unbounded flexible solve); one that holds `V` alone drops `Z`; one that holds
/// `V` in single precision rounds it there too (only where the tolerance allows it);
/// a smaller one then runs the columns in waves. Every solve converges, its solutions
/// agree with the unbounded one's to the tolerance, it holds no more than the room and
/// a step of one column, and the plan within the room is the plan of the bases it
/// ended in.
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
    // the plan of the bases as `basis` stores them, unbounded
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
            // the plan within the room: the bases the solve ended in
            let within = tight.block_bytes::<C>(n, s, deep);
            if waves {
                assert!(within <= room, "plan {within} room {room}");
            } else {
                assert_eq!(within, plan(r.basis, deep));
            }
        }
    }
}
