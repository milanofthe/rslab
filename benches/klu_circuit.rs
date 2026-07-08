//! KLU path vs multifrontal LU on circuit-shaped matrices (#15).
//!
//! Generates MNA-like matrices — very sparse (~4-5 nnz/col), unsymmetric,
//! diagonally weighted, with reducible (block upper triangular) structure —
//! and reports for each size:
//!   * KLU: analyze / factor / refactor / solve wall time, factor nnz, blocks
//!   * multifrontal LU (defaults): factor / solve wall time, factor nnz
//!
//! plus the frequency-sweep proxy: 20 refactor+solve cycles KLU vs 20
//! factor+solve cycles multifrontal.
//!
//! Run: `cargo bench --bench klu_circuit`.

use std::time::Instant;

use rslab::{factor_general_lu, solve_lu, GeneralCsc, KluSettings, KluSymbolic, SolverSettings};

/// Deterministic xorshift.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// MNA-like circuit matrix: n nodes in `n_stages` cascaded stages (signal
/// flows one way between stages -> reducible), each stage internally coupled
/// (local 2D neighborhood + a few long-range couplings -> irreducible blocks
/// of stage size). Diagonal ~ sum of couplings (conductance-like), values
/// unsymmetric.
fn circuit(n: usize, n_stages: usize, seed: u64) -> GeneralCsc<f64> {
    let mut rng = Rng(seed | 1);
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let mut colsum = vec![0.0f64; n];
    let stage = n / n_stages;
    for j in 0..n {
        let s = (j / stage).min(n_stages - 1);
        let (lo, hi) = (
            s * stage,
            if s == n_stages - 1 {
                n
            } else {
                (s + 1) * stage
            },
        );
        let span = hi - lo;
        // local neighborhood within the stage (wraps -> strongly connected);
        // slightly unsymmetric conductances
        for &t in &[1usize, 2, 17] {
            let i = lo + (j - lo + t) % span;
            if i != j {
                let g = 0.5 + rng.next_f64().abs();
                r.push(i);
                c.push(j);
                v.push(-g);
                colsum[j] += g;
                let g2 = 0.5 + rng.next_f64().abs();
                r.push(j);
                c.push(i);
                v.push(-g2);
                colsum[i] += g2;
            }
        }
        // one-directional inter-stage feed (previous stage listens)
        if s > 0 && (j - lo) % 3 == 0 {
            let i = lo - stage + (j - lo) % stage;
            let g = 0.2 + 0.3 * rng.next_f64().abs();
            r.push(i);
            c.push(j);
            v.push(-g);
            colsum[j] += g;
        }
    }
    // grounded-capacitor margin keeps every column strictly diagonally
    // dominant: the condition number stays O(1) at every size
    for (j, &cs) in colsum.iter().enumerate() {
        r.push(j);
        c.push(j);
        v.push(cs + 0.1 + 0.1 * rng.next_f64().abs());
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
}

fn resid(a: &GeneralCsc<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; a.n];
    a.matvec(x, &mut ax);
    let num = b
        .iter()
        .zip(&ax)
        .map(|(bi, axi)| (bi - axi).abs())
        .fold(0.0, f64::max);
    num / b.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300)
}

fn main() {
    const SWEEP: usize = 20;
    println!(
        "{:>8} {:>9} | {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} | {:>9} {:>9} {:>9} | {:>8} {:>8}",
        "n",
        "nnz",
        "klu-ana",
        "klu-fac",
        "klu-refac",
        "klu-solve",
        "klu-nnz",
        "blocks",
        "mf-fac",
        "mf-solve",
        "mf-nnz",
        "sweep-klu",
        "sweep-mf"
    );
    for &(n, stages) in &[
        (2_000usize, 8usize),
        (10_000, 16),
        (50_000, 32),
        (200_000, 64),
    ] {
        let a = circuit(n, stages, 0xC0FFEE ^ n as u64);
        let b: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();

        // --- KLU ---
        let t = Instant::now();
        let sym = KluSymbolic::analyze(&a).unwrap();
        let t_ana = t.elapsed();
        let t = Instant::now();
        let mut klu = sym.factor(&a, &KluSettings::default()).unwrap();
        let t_fac = t.elapsed();
        let t = Instant::now();
        klu.refactor(&a).unwrap();
        let t_refac = t.elapsed();
        let t = Instant::now();
        let x = klu.solve(&b).unwrap();
        let t_solve = t.elapsed();
        let klu_res = resid(&a, &x, &b);
        let klu_nnz = klu.factor_nnz();
        let blocks = klu.n_blocks();

        // --- multifrontal LU (defaults) ---
        let t = Instant::now();
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let t_mf_fac = t.elapsed();
        let t = Instant::now();
        let xm = solve_lu(&f, &b).unwrap();
        let t_mf_solve = t.elapsed();
        let mf_res = resid(&a, &xm, &b);
        let mf_nnz = f.factor_nnz();
        assert!(
            klu_res < 1e-8,
            "klu residual {klu_res} (mf residual {mf_res})"
        );
        assert!(mf_res < 1e-8, "multifrontal residual {mf_res}");

        // --- sweep proxy: 20 value sets, same pattern ---
        let t = Instant::now();
        for k in 0..SWEEP {
            let scale = 1.0 + 0.03 * k as f64;
            let a2 = GeneralCsc {
                n: a.n,
                col_ptr: a.col_ptr.clone(),
                row_idx: a.row_idx.clone(),
                values: a.values.iter().map(|&v| v * scale).collect(),
            };
            klu.refactor(&a2).unwrap();
            let _ = klu.solve(&b).unwrap();
        }
        let t_sweep_klu = t.elapsed();
        let t = Instant::now();
        for k in 0..SWEEP {
            let scale = 1.0 + 0.03 * k as f64;
            let a2 = GeneralCsc {
                n: a.n,
                col_ptr: a.col_ptr.clone(),
                row_idx: a.row_idx.clone(),
                values: a.values.iter().map(|&v| v * scale).collect(),
            };
            let f2 = factor_general_lu(&a2, &SolverSettings::default()).unwrap();
            let _ = solve_lu(&f2, &b).unwrap();
        }
        let t_sweep_mf = t.elapsed();

        println!(
            "{:>8} {:>9} | {:>9.2?} {:>9.2?} {:>9.2?} {:>9.2?} {:>9} {:>7} | {:>9.2?} {:>9.2?} {:>9} | {:>8.2?} {:>8.2?}",
            n,
            a.nnz(),
            t_ana,
            t_fac,
            t_refac,
            t_solve,
            klu_nnz,
            blocks,
            t_mf_fac,
            t_mf_solve,
            mf_nnz,
            t_sweep_klu,
            t_sweep_mf
        );
        println!(
            "{:>8} residuals: klu {:.1e}  mf {:.1e}",
            "", klu_res, mf_res
        );
    }
}
