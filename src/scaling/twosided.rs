//! Two-sided infinity-norm equilibration of a general matrix, the row and
//! column scalings `A_hat = D_r A D_c` of the LU path.
//!
//! - [`OnePassInfNorm`](super::ScalingStrategy::OnePassInfNorm) (the default):
//!   one step, `d_r[i] = 1/sqrt(max_j |a_ij|)`, `d_c[j] = 1/sqrt(max_i |a_ij|)`,
//!   both from the unscaled matrix.
//! - [`InfNorm`](super::ScalingStrategy::InfNorm): the same step iterated on
//!   the scaled matrix (Ruiz 2001), until every row and column of `A_hat` has
//!   its largest entry within a factor `1 + tol` of one.
//! - [`Identity`](super::ScalingStrategy::Identity): no scaling.
//!
//! The one step leaves a row whose own largest entry is small against the
//! largest entries of its columns well below one (`a / sqrt(r c)` with
//! `c >> r`), and its pivots are then judged against the static pivot floor
//! on that smaller scale. The iteration brings every row and column to the
//! unit scale the floor and the threshold test are stated on.
//!
//! Tolerance. The LU passes the pivot threshold `u` (at least
//! [`INFNORM_TOL`](crate::constants::INFNORM_TOL)): the threshold test keeps a
//! pivot within a factor `u` of its column's largest entry, so scales known to
//! within a factor `1 + u` move no decision by more than the margin the test
//! itself grants, and further steps change the factor in its rounding only.
//! Measured on MoM near-field saddles (rapidmom exp/cond-lu: tia130a and
//! pa180a, 1 and 180 GHz): the floored pivots and the backward error at
//! `1 + u = 1.1` (7 to 8 steps) are those at `1e-8` (31 steps).
//!
//! Convergence. Ruiz (2001, "A scaling algorithm to equilibrate both rows and
//! columns norms in matrices", RAL-TR-2001-034) shows for the infinity norm
//! that every step halves (at least) the logarithm of the deviation of the
//! row and column maxima from one. A matrix whose maxima start within a
//! factor `s` of one (`s` the spread of the first step) thus reaches the
//! tolerance `t` after `log2(ln s / ln(1 + t))` steps; the iteration stops
//! there at the latest (a structurally degenerate matrix, an all-zero row or
//! column, does not contract on that row and would otherwise never stop).

use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;

use super::ScalingStrategy;

/// The steps the iteration takes at most to bring maxima spread over a factor
/// `spread` within `tol` of one, at the rate one half (see the module).
fn steps_bound(spread: f64, tol: f64) -> usize {
    let ln_s = spread.ln();
    let ln_t = tol.ln_1p();
    if ln_s.partial_cmp(&ln_t) != Some(std::cmp::Ordering::Greater) || ln_t <= 0.0 {
        return 1;
    }
    (ln_s / ln_t).log2().ceil() as usize + 1
}

/// What a maximum costs per entry, learned from the passes.
static MAXIMA_RATE: crate::parallel::Rate =
    crate::parallel::Rate::new(crate::constants::SCAN_NS_PER_ENTRY);

/// The row and column maxima of `diag(r) A diag(c)` (the first step, `r = c =
/// 1`, reads `|a_ij|` exactly as the one-pass scaling always has). The columns
/// in pieces of about equal entries on the pool, each piece its own row maxima,
/// taken together after: a maximum does not depend on the order it is taken
/// in, so the result is the same on any number of workers.
fn maxima<T: Scalar>(a: &GeneralCsc<T>, r: &[f64], c: &[f64], rm: &mut [f64], cm: &mut [f64]) {
    let n = a.n;
    let nnz = a.col_ptr[n];
    let pieces = crate::parallel::threads()
        .min(nnz / crate::constants::EQUILIBRATION_PIECE_MIN_NNZ)
        .max(1);
    // piece `p` is the columns from the first whose entries start at or after
    // `p * nnz / pieces`
    let starts: Vec<usize> = (0..=pieces)
        .map(|p| a.col_ptr.partition_point(|&k| k < p * nnz / pieces).min(n))
        .collect();
    let part = |p: usize| {
        let (j0, j1) = (starts[p], if p + 1 == pieces { n } else { starts[p + 1] });
        let mut rows = vec![0.0f64; n];
        let mut cols = vec![0.0f64; j1 - j0];
        for j in j0..j1 {
            let cj = c[j];
            let mut col = 0.0f64;
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[k];
                let m = a.values[k].magnitude() * r[i] * cj;
                if m > rows[i] {
                    rows[i] = m;
                }
                if m > col {
                    col = m;
                }
            }
            cols[j - j0] = col;
        }
        (j0, rows, cols)
    };
    rm.fill(0.0);
    let work = crate::parallel::Work::of(nnz as f64, &MAXIMA_RATE);
    for (j0, rows, cols) in crate::parallel::map(pieces, work, part) {
        cm[j0..j0 + cols.len()].copy_from_slice(&cols);
        for (x, y) in rm.iter_mut().zip(&rows) {
            if *y > *x {
                *x = *y;
            }
        }
    }
}

/// The row and column scalings `(d_r, d_c)` of `a` under `strategy`: the
/// [`OnePassInfNorm`](ScalingStrategy::OnePassInfNorm) step for every
/// strategy but [`InfNorm`](ScalingStrategy::InfNorm) (iterated) and
/// [`Identity`](ScalingStrategy::Identity) (ones); the symmetric strategies
/// `Mc64Symmetric` and `External` have no two-sided meaning (the settings
/// report them as ignored on the LU path). The iteration stops once every
/// row and column maximum is within a factor `1 + tol` of one (`tol` at least
/// [`INFNORM_TOL`](crate::constants::INFNORM_TOL)). Also the steps taken.
pub(crate) fn equilibrate_general<T: Scalar>(
    a: &GeneralCsc<T>,
    strategy: &ScalingStrategy,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, usize) {
    let n = a.n;
    let (mut r, mut c) = (vec![1.0f64; n], vec![1.0f64; n]);
    if *strategy == ScalingStrategy::Identity {
        return (r, c, 0);
    }
    let tol = tol.max(crate::constants::INFNORM_TOL);
    let ln_tol = tol.ln_1p();
    let (mut rm, mut cm) = (vec![0.0f64; n], vec![0.0f64; n]);
    let iterate = *strategy == ScalingStrategy::InfNorm;
    let mut bound = 1;
    let mut steps = 0;
    loop {
        maxima(a, &r, &c, &mut rm, &mut cm);
        // the largest factor between a maximum and one, on the log scale
        let mut dev = 0.0f64;
        for &m in rm.iter().chain(&cm) {
            if m > 0.0 && m.is_finite() {
                dev = dev.max(m.ln().abs());
            }
        }
        if steps > 0 && dev <= ln_tol {
            break;
        }
        if steps == 0 && iterate {
            bound = steps_bound(dev.exp(), tol);
        }
        for (d, &m) in r.iter_mut().zip(&rm).chain(c.iter_mut().zip(&cm)) {
            if m > 0.0 {
                *d = super::kr_guarded_update(*d, m);
            }
        }
        steps += 1;
        if steps >= bound {
            break;
        }
    }
    (r, c, steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csc(n: usize, cols: &[&[(usize, f64)]]) -> GeneralCsc<f64> {
        let mut col_ptr = vec![0];
        let (mut row_idx, mut values) = (Vec::new(), Vec::new());
        for col in cols {
            for &(i, v) in *col {
                row_idx.push(i);
                values.push(v);
            }
            col_ptr.push(row_idx.len());
        }
        GeneralCsc {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    fn maxima_of(a: &GeneralCsc<f64>, r: &[f64], c: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (mut rm, mut cm) = (vec![0.0; a.n], vec![0.0; a.n]);
        maxima(a, r, c, &mut rm, &mut cm);
        (rm, cm)
    }

    /// The one step is the single Knight-Ruiz step on the unscaled matrix,
    /// `1/sqrt(row max)` and `1/sqrt(column max)`, and leaves a row whose max
    /// is small against its column's max below one.
    #[test]
    fn one_pass_is_one_step() {
        // [[1, 1e6], [1e-2, 1]]
        let a = csc(2, &[&[(0, 1.0), (1, 1e-2)], &[(0, 1e6), (1, 1.0)]]);
        let (r, c, steps) = equilibrate_general(&a, &ScalingStrategy::OnePassInfNorm, 0.1);
        assert_eq!(steps, 1);
        assert_eq!(r, vec![1e-3, 1.0]);
        assert_eq!(c, vec![1.0, 1e-3]);
        let (rm, _) = maxima_of(&a, &r, &c);
        assert!(rm[1] < 1e-1, "row 1 max {}", rm[1]);
    }

    /// The iteration brings every row and column maximum within a factor
    /// `1 + tol` of one, in no more steps than the rate one half allows, on a
    /// saddle-like matrix whose couplings span ten orders.
    #[test]
    fn iterated_reaches_unit_maxima_within_the_bound() {
        let a = csc(
            3,
            &[
                &[(0, 1e-4), (1, 3e3), (2, 1.0)],
                &[(0, 2e5), (1, 1.0)],
                &[(0, 1e-6), (2, 7.0)],
            ],
        );
        let (r0, c0) = maxima_of(&a, &[1.0; 3], &[1.0; 3]);
        let spread = r0
            .iter()
            .chain(&c0)
            .map(|m| m.max(1.0 / m))
            .fold(1.0, f64::max);
        for tol in [0.1, 1e-3, crate::constants::INFNORM_TOL] {
            let (r, c, steps) = equilibrate_general(&a, &ScalingStrategy::InfNorm, tol);
            let (rm, cm) = maxima_of(&a, &r, &c);
            for m in rm.iter().chain(&cm) {
                assert!(
                    m.ln().abs() <= tol.ln_1p() * (1.0 + 1e-12),
                    "max {m} at {tol}"
                );
            }
            // the maxima converged (above) within the steps the rate one half allows
            assert!(steps <= steps_bound(spread, tol), "{steps} steps at {tol}");
        }
    }

    /// A zero row has no maximum to equilibrate: it stays unscaled, the other
    /// rows and the columns are balanced.
    #[test]
    fn zero_row_stays_unscaled() {
        let a = csc(3, &[&[(0, 1e4), (2, 1.0)], &[(0, 1e-3)], &[(2, 5.0)]]);
        let (r, c, steps) = equilibrate_general(&a, &ScalingStrategy::InfNorm, 1e-8);
        assert_eq!(r[1], 1.0);
        assert!(steps >= 1);
        let (rm, _) = maxima_of(&a, &r, &c);
        assert!((rm[0] - 1.0).abs() < 1e-6 && (rm[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn identity_is_ones() {
        let a = csc(2, &[&[(0, 3.0)], &[(1, 1e-9)]]);
        let (r, c, steps) = equilibrate_general(&a, &ScalingStrategy::Identity, 0.1);
        assert_eq!((r, c, steps), (vec![1.0, 1.0], vec![1.0, 1.0], 0));
    }
}
