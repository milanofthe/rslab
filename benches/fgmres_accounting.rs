//! **FGMRES preconditioner-apply accounting** (issue #7).
//!
//! Plain right-preconditioned restarted GMRES spends, per restart cycle, one extra
//! `M^{-1}` solve rebuilding the update `x += M^{-1}(V y)`. Flexible GMRES keeps the
//! preconditioned Arnoldi basis `Z = M^{-1} V` and updates `x += Z y` directly, so it
//! spends **exactly one** `M^{-1}` apply per inner iteration and none at the restart -
//! saving one preconditioner solve per cycle. This bench mirrors the unit test
//! `fgmres_saves_one_precond_apply_per_restart_cycle` at benchmark scale by wrapping
//! the operator and the preconditioner in counting adapters. Since FGMRES issues one
//! matvec per Arnoldi step plus one residual matvec per cycle, and one `M^{-1}` apply
//! per step, the cycle count is recovered exactly as `op_applies - pc_applies`, and
//! the plain-GMRES apply count is `pc_applies + cycles`.
//!
//! Run: `cargo bench --features matgen --bench fgmres_accounting`
//!   env: RLA_DROPTOL (default 0.1), RLA_RESTART (default 20), RLA_JSON=<path>.

use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::Complex;
use rslab::matgen::fem::{convection_diffusion, Flow};
use rslab::{
    factor_general_lu, gmres, GeneralCsc, LinearOperator, LuFactors, Preconditioner, RslabError,
    SolverSettings, Threads,
};

type C = Complex<f64>;

fn emit(fields: &str) {
    let Ok(path) = std::env::var("RLA_JSON") else {
        return;
    };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{{{fields}}}");
    }
}

struct CountOp<'a> {
    inner: &'a GeneralCsc<C>,
    applies: AtomicUsize,
}
impl LinearOperator<C> for CountOp<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn apply(&self, x: &[C], y: &mut [C]) {
        self.applies.fetch_add(1, Ordering::Relaxed);
        self.inner.matvec(x, y);
    }
}

struct CountPc<'a> {
    inner: &'a LuFactors<C>,
    applies: AtomicUsize,
}
impl Preconditioner<C> for CountPc<'_> {
    fn apply(&self, r: &[C], z: &mut [C]) -> Result<(), RslabError> {
        self.applies.fetch_add(1, Ordering::Relaxed);
        Preconditioner::<C>::apply(self.inner, r, z)
    }
    fn solve_threads(&self) -> Threads {
        Preconditioner::<C>::solve_threads(self.inner)
    }
}

fn main() {
    let droptol: f64 = std::env::var("RLA_DROPTOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1);
    let restart: usize = std::env::var("RLA_RESTART")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let tol = 1e-8;
    let dims = [60usize, 100, 140];

    println!(
        "FGMRES preconditioner-apply accounting  [drop_tol={droptol:e} restart={restart}]\n\
         n       iters   cycles   fgmres_applies   plain_applies   saved   res"
    );

    for &d in &dims {
        let a = convection_diffusion::<C>(&[d, d], 0.02, Flow::Rotating, true);
        let n = a.n;
        let b: Vec<C> = (0..n)
            .map(|i| Complex::new((i % 7) as f64 - 3.0, (i % 5) as f64 - 2.0))
            .collect();
        let opts = SolverSettings::preconditioner(1e-10).with_drop_tol(droptol);
        let Ok(lu) = factor_general_lu(&a, &opts) else {
            eprintln!("factor failed at n={n}");
            continue;
        };
        let op = CountOp {
            inner: &a,
            applies: AtomicUsize::new(0),
        };
        let pc = CountPc {
            inner: &lu,
            applies: AtomicUsize::new(0),
        };
        let Ok(res) = gmres(&op, &b, &pc, tol, 100_000, restart, None) else {
            eprintln!("gmres failed at n={n}");
            continue;
        };
        let op_applies = op.applies.load(Ordering::Relaxed);
        let pc_applies = pc.applies.load(Ordering::Relaxed);
        let cycles = op_applies.saturating_sub(pc_applies);
        let plain = pc_applies + cycles;
        println!(
            "{n:6}   {:5}   {cycles:5}   {pc_applies:12}   {plain:12}   {cycles:5}   {:.1e}",
            res.iters, res.final_res
        );
        emit(&format!(
            "\"n\":{n},\"drop_tol\":{droptol:e},\"restart\":{restart},\"iters\":{},\
             \"cycles\":{cycles},\"fgmres_applies\":{pc_applies},\"plain_applies\":{plain},\
             \"saved\":{cycles},\"res\":{:e}",
            res.iters, res.final_res
        ));
    }
}
