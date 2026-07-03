//! **Block-GMRES within-cycle deflation**: the shrinking-panel effect at bench scale
//! (issue #4).
//!
//! The benchmark-scale mirror of `gmres_block_within_cycle_deflation_shrinks_applies`
//! (src/numeric/iterative.rs). A diagonal operator with distinct, well-separated
//! eigenvalues has, for a right-hand side supported on `m` unit vectors, minimal-
//! polynomial degree exactly `m`: GMRES converges in `m` steps. We build an `s`-RHS
//! block whose columns are supported on a *staggered* number of unit vectors, so the
//! columns converge at spread rates. Within-cycle deflation compacts a column out of
//! the active panel the instant its Hessenberg estimate reaches `tol`, so the batched
//! operator/preconditioner applies shrink to the still-active width *mid-cycle* - not
//! only at the restart boundary. A counting operator records the width of every
//! `apply_block`; we report the total column-applies against the full-width bound
//! (`calls x s`, the work a no-mid-cycle-deflation schedule would do) and dump the
//! per-step active-width staircase.
//!
//! Run: `cargo bench --bench deflation_study`
//!   env: RLA_N (dimension, default 20000), RLA_MAXDEG (hardest column's degree,
//!        default 192), RLA_RESTART (default 64), RLA_JSON=<path> to emit JSONL.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use num_complex::Complex;
use rslab::{gmres_block, GeneralCsc, LinearOperator, NoPreconditioner};

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

/// Operator wrapper that records the width `s` of every `apply_block` call, so the
/// panel-shrinking effect of within-cycle deflation is directly measurable.
struct CountOp<'a> {
    inner: &'a GeneralCsc<C>,
    cols: AtomicUsize,
    calls: AtomicUsize,
    widths: Mutex<Vec<usize>>,
}
impl LinearOperator<C> for CountOp<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn apply(&self, x: &[C], y: &mut [C]) {
        self.cols.fetch_add(1, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.widths.lock().unwrap().push(1);
        self.inner.matvec(x, y);
    }
    fn apply_block(&self, x: &[C], y: &mut [C], s: usize) {
        self.cols.fetch_add(s, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.widths.lock().unwrap().push(s);
        self.inner.apply_block(x, y, s);
    }
}

fn main() {
    let n: usize = std::env::var("RLA_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20000);
    let maxdeg: usize = std::env::var("RLA_MAXDEG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(192);
    let restart: usize = std::env::var("RLA_RESTART")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let (tol, maxit) = (1e-10, 100_000);
    let ss = [4usize, 8, 16];
    let s_max = *ss.iter().max().unwrap();

    // Diagonal operator with distinct, well-separated eigenvalues.
    let c = |re: f64, im: f64| Complex::new(re, im);
    let idx: Vec<usize> = (0..n).collect();
    let vals: Vec<C> = (0..n).map(|i| c(2.0 + i as f64, 0.5)).collect();
    let a = GeneralCsc::<C>::from_triplets(n, &idx, &idx, &vals).unwrap();

    println!(
        "Block-GMRES within-cycle deflation  [n={n} maxdeg={maxdeg} restart={restart}]\n\
         s    iters   op_cols   full(=calls*s)   reduction   wct(ms)"
    );

    for &s in &ss {
        // RHS `c` supported on its first `deg_c` unit vectors: staggered degrees from
        // ~maxdeg/s up to maxdeg, so the columns converge at spread rates.
        let degs: Vec<usize> = (0..s)
            .map(|cc| {
                let lo = (maxdeg / s).max(2);
                lo + (cc * (maxdeg - lo)) / (s - 1).max(1)
            })
            .collect();
        let mut bblk = vec![C::default(); n * s];
        for (cc, &deg) in degs.iter().enumerate() {
            for i in 0..deg.min(n) {
                bblk[cc * n + i] = c(1.0, 0.0);
            }
        }

        let op = CountOp {
            inner: &a,
            cols: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            widths: Mutex::new(Vec::new()),
        };
        let t = Instant::now();
        let res = gmres_block(&op, &bblk, s, &NoPreconditioner, tol, maxit, restart, None).unwrap();
        let wct = t.elapsed().as_secs_f64() * 1e3;
        assert!(res.converged, "block solve must converge (s={s})");

        let op_cols = op.cols.load(Ordering::Relaxed);
        let op_calls = op.calls.load(Ordering::Relaxed);
        let op_full = op_calls * s;
        let reduction = op_cols as f64 / op_full as f64;
        println!(
            "{s:2}   {:5}   {op_cols:7}   {op_full:12}      {reduction:6.3}    {wct:7.1}",
            res.iters
        );
        emit(&format!(
            "\"kind\":\"summary\",\"s\":{s},\"n\":{n},\"restart\":{restart},\"maxdeg\":{maxdeg},\
             \"iters\":{},\"op_cols\":{op_cols},\"op_calls\":{op_calls},\"op_full\":{op_full},\
             \"reduction\":{reduction:.5},\"wct_ms\":{wct:.3}",
            res.iters
        ));

        if s == s_max {
            let widths = op.widths.into_inner().unwrap();
            let arr: Vec<String> = widths.iter().map(|w| w.to_string()).collect();
            emit(&format!(
                "\"kind\":\"widths\",\"s\":{s},\"n\":{n},\"widths\":[{}]",
                arr.join(",")
            ));
        }
    }
}
