//! The kernel knobs as the kernels take them.

use crate::numeric::settings::KernelSettings;

#[derive(Debug, Clone, Copy)]
pub(crate) struct KernelTuning<'a> {
    pub k: KernelSettings,
    pub pivot_threshold: f64,
    pub interrupt: Option<&'a std::sync::atomic::AtomicBool>,
    /// The workers of the pool the factorization runs on (the whole pool,
    /// never a share of it): what the update of a large node is cut for
    /// ([`unit_share`](crate::numeric::supernodal::unit_share)).
    pub workers: usize,
}

impl KernelTuning<'_> {
    #[inline]
    pub fn interrupted(&self) -> Result<(), crate::error::RslabError> {
        match self.interrupt {
            Some(flag) if flag.load(std::sync::atomic::Ordering::Relaxed) => {
                Err(crate::error::RslabError::Interrupted)
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_constants::{solve_backward_error, UC64};
    use crate::{CscMatrix, KernelSettings, LdltSolver, SolverSettings};
    use num_complex::Complex;

    fn inf_norm(v: &[Complex<f64>]) -> f64 {
        v.iter().map(|z| z.norm()).fold(0.0, f64::max)
    }

    fn row_sums(a: &CscMatrix<Complex<f64>>) -> Vec<Complex<f64>> {
        let mut abs = a.clone();
        abs.values
            .iter_mut()
            .for_each(|v| *v = Complex::new(v.norm(), 0.0));
        let mut sums = vec![Complex::new(0.0, 0.0); a.n];
        abs.symv(&vec![Complex::new(1.0, 0.0); a.n], &mut sums);
        sums
    }

    fn backward_error(a: &CscMatrix<Complex<f64>>, x: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
        let mut ax = vec![Complex::new(0.0, 0.0); a.n];
        a.symv(x, &mut ax);
        let r: Vec<Complex<f64>> = ax.iter().zip(b).map(|(p, q)| p - q).collect();
        inf_norm(&r) / (inf_norm(&row_sums(a)) * inf_norm(x) + inf_norm(b))
    }

    fn helmholtz(m: usize) -> (CscMatrix<Complex<f64>>, Vec<Complex<f64>>) {
        let c = |re, im| Complex::new(re, im);
        let n = m * m;
        let idx = |r: usize, cc: usize| r * m + cc;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..m {
            for cc in 0..m {
                let p = idx(r, cc);
                rows.push(p);
                cols.push(p);
                vals.push(c(4.0, 0.5));
                for (dr, dc) in [(1usize, 0usize), (0, 1)] {
                    if r + dr < m && cc + dc < m {
                        let q = idx(r + dr, cc + dc);
                        let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                        rows.push(hi);
                        cols.push(lo);
                        vals.push(c(-1.0, 0.1));
                    }
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 50.0, 1.0)).collect();
        (a, b)
    }

    #[test]
    fn scalar_serial_and_gemm_parallel_kernels_solve_alike() {
        let (a, b) = helmholtz(10);
        let solve = |s: &SolverSettings| LdltSolver::factor(&a, s).unwrap().solve(&b).unwrap();

        let x_def = solve(&SolverSettings::default().with_threads(0));
        let x_scalar = solve(&SolverSettings {
            kernels: KernelSettings {
                scalar_gate: usize::MAX,
                par_gemm: usize::MAX,
                par_cdiv: usize::MAX,
                ..Default::default()
            },
            ..SolverSettings::default().with_threads(0)
        });
        let x_par = solve(&SolverSettings {
            kernels: KernelSettings {
                scalar_gate: 0,
                par_gemm: 0,
                par_cdiv: 0,
                ..Default::default()
            },
            ..SolverSettings::default().with_threads(0)
        });
        let bound = solve_backward_error(a.n, UC64);
        for (name, x) in [
            ("default", &x_def),
            ("scalar", &x_scalar),
            ("parallel", &x_par),
        ] {
            let omega = backward_error(&a, x, &b);
            assert!(omega <= bound, "{name} path: backward error {omega:e}");
        }
    }

    #[test]
    fn every_panel_width_gives_a_backward_stable_solve() {
        let (a, b) = helmholtz(9);
        for nb in [16usize, 32, 64, 100, 200] {
            let mut s = SolverSettings::default();
            s.kernels.panel_nb = nb;
            let x = LdltSolver::factor(&a, &s).unwrap().solve(&b).unwrap();
            let omega = backward_error(&a, &x, &b);
            assert!(
                omega <= solve_backward_error(a.n, UC64),
                "NB={nb} backward error {omega:.2e}"
            );
        }
    }
}
