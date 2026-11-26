//! Knight-Ruiz inf-norm iterative equilibration for sparse symmetric
//! matrices.
//!
//! Given a symmetric CSC matrix `A`, compute a diagonal `d` such that
//! each row of `D*A*D` has infinity-norm ~ 1, where `D = diag(d)`,
//! iterating over lower-triangular CSC storage.
//!
//! Inf-norm balancing complements MC64: matching can classify a matrix
//! as already balanced even when its row norms span 4+ orders of
//! magnitude (HYDCAR20 / METHANL8 / SWOPF / HATFLDG are examples), and
//! those matrices need this equilibration before Bunch-Kaufman pivoting.
//!
//! Algorithm (Jacobi-style; converges in about as many iterations as a
//! Gauss-Seidel variant while being simpler to implement over CSC
//! lower-triangle storage):
//!
//! 1. Initialize `d = 1`.
//! 2. Repeat up to `max_iter` times:
//!    a. For each row `i`, compute `max_i = max_j |d[i]*a[i,j]*d[j]|`.
//!    b. Update `d[i] /= sqrt(max_i)` for every row whose `max_i > 0`.
//!    c. Stop when `max_i |1 - max_i|` falls below `tol`.

use crate::scaling::ScalingInfo;
use crate::sparse::csc::CscMatrix;

/// Compute the Knight-Ruiz inf-norm symmetric scaling vector for a
/// lower-triangular symmetric CSC matrix. Returns the diagonal `d`
/// such that `D*A*D` has unit-inf rows, paired with `ScalingInfo::Applied`.
pub fn compute_infnorm(matrix: &CscMatrix) -> (Vec<f64>, ScalingInfo) {
    let n = matrix.n;
    if n == 0 {
        return (Vec::new(), ScalingInfo::Applied);
    }
    let mut d = vec![1.0f64; n];

    let max_iter = crate::constants::INFNORM_MAX_ITER;
    let tol = crate::constants::INFNORM_TOL;

    let mut row_max = vec![0.0f64; n];

    for _ in 0..max_iter {
        for r in row_max.iter_mut() {
            *r = 0.0;
        }

        for j in 0..n {
            let col_start = matrix.col_ptr[j];
            let col_end = matrix.col_ptr[j + 1];
            let dj = d[j];
            let mut col_max = row_max[j];

            for k in col_start..col_end {
                let i = matrix.row_idx[k];
                let v = (d[i] * matrix.values[k] * dj).abs();
                if i != j && v > row_max[i] {
                    row_max[i] = v;
                }
                if v > col_max {
                    col_max = v;
                }
            }
            row_max[j] = col_max;
        }

        let mut max_dev = 0.0f64;
        for i in 0..n {
            let m = row_max[i];
            if m > 0.0 {
                d[i] = crate::scaling::kr_guarded_update(d[i], m);
                let dev = (m - 1.0).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
            }
        }

        if max_dev < tol {
            break;
        }
    }

    (d, ScalingInfo::Applied)
}

/// One-pass symmetric inf-norm equilibration `s_i = 1/sqrt(max_j |A_ij|)` (a single
/// Knight-Ruiz step) and the [`crate::LdltSolver`] default. Cheaper than the
/// iterative [`compute_infnorm`]: it tolerates a zero diagonal (the
/// row max includes the off-diagonals) and never iterates. An all-zero row is
/// left unscaled (`s_i = 1`), surfacing as a singular pivot during factorization.
/// The scan folds in the symmetric partner `(j, i)` of every stored `(i, j)`, so
/// it is correct for a lower-triangular *or* a full symmetric store (`max` is
/// idempotent).
pub fn compute_onepass(matrix: &CscMatrix) -> (Vec<f64>, ScalingInfo) {
    let n = matrix.n;
    let mut row_max = vec![0.0f64; n];
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let m = matrix.values[k].abs();
            if m > row_max[i] {
                row_max[i] = m;
            }
            if i != j && m > row_max[j] {
                row_max[j] = m;
            }
        }
    }
    let s = row_max
        .iter()
        .map(|&r| crate::scaling::inv_sqrt_scale_guarded(r))
        .collect();
    (s, ScalingInfo::Applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::INFNORM_TOL;
    use crate::sparse::csc::CscMatrix;
    use crate::test_constants::knight_ruiz_deviation;

    #[test]
    fn kr_guarded_update_guards_extremes() {
        assert_eq!(crate::scaling::kr_guarded_update(1.0, 4.0), 0.5);
        assert_eq!(crate::scaling::kr_guarded_update(6.0, 9.0), 2.0);
        assert_eq!(
            crate::scaling::kr_guarded_update(f64::MAX, f64::MIN_POSITIVE),
            f64::MAX
        );
        assert_eq!(crate::scaling::kr_guarded_update(1.0, f64::INFINITY), 1.0);
        assert_eq!(crate::scaling::kr_guarded_update(1.0, f64::NAN), 1.0);
    }

    #[test]
    fn kr_subnormal_coupling_stays_finite() {
        let m = CscMatrix::from_triplets(2, &[1, 1], &[0, 1], &[1e-320, 1.0]).unwrap();
        let (d, info) = compute_infnorm(&m);
        assert!(matches!(info, ScalingInfo::Applied));
        for (i, &di) in d.iter().enumerate() {
            assert!(
                di.is_finite() && di > 0.0,
                "d[{i}] = {di} must be finite and strictly positive"
            );
        }
        let scaled = (d[1] * 1e-320 * d[0]).abs();
        assert!(scaled.is_finite(), "scaled entry {scaled} must be finite");
    }

    #[test]
    fn onepass_guard_extremes() {
        assert_eq!(crate::scaling::inv_sqrt_scale_guarded(4.0), 0.5);
        assert_eq!(crate::scaling::inv_sqrt_scale_guarded(0.0), 1.0);
        assert_eq!(crate::scaling::inv_sqrt_scale_guarded(f64::INFINITY), 1.0);
        assert_eq!(crate::scaling::inv_sqrt_scale_guarded(f64::NAN), 1.0);
        let g = crate::scaling::inv_sqrt_scale_guarded(1e-320);
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn diag_3x3() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        let (d, _info) = compute_infnorm(&m);
        for i in 0..3 {
            let scaled = d[i] * m.values[i] * d[i];
            assert!(
                (scaled - 1.0).abs() <= knight_ruiz_deviation(INFNORM_TOL),
                "d[{}] = {} scales the diagonal to {}",
                i,
                d[i],
                scaled
            );
        }
    }

    #[test]
    fn sym_2x2() {
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[4.0, 2.0, 9.0]).unwrap();
        let (d, _) = compute_infnorm(&m);
        let a00 = d[0] * d[0] * 4.0;
        let a01 = d[0] * d[1] * 2.0;
        let a11 = d[1] * d[1] * 9.0;
        let row0 = a00.abs().max(a01.abs());
        let row1 = a01.abs().max(a11.abs());
        let bound = knight_ruiz_deviation(INFNORM_TOL);
        assert!((row0 - 1.0).abs() <= bound, "row0 max = {}", row0);
        assert!((row1 - 1.0).abs() <= bound, "row1 max = {}", row1);
    }

    #[test]
    fn arrow_6x6() {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..6 {
            rows.push(j);
            cols.push(j);
            vals.push((j + 2) as f64);
        }
        for j in 0..5 {
            rows.push(5);
            cols.push(j);
            vals.push(1.0);
        }
        let m = CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap();
        let (d, _) = compute_infnorm(&m);
        for i in 0..6 {
            let mut row_max = 0.0f64;
            for j in 0..6 {
                let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
                let mut v = 0.0;
                for k in m.col_ptr[jj]..m.col_ptr[jj + 1] {
                    if m.row_idx[k] == ii {
                        v = m.values[k];
                        break;
                    }
                }
                let scaled = (d[i] * v * d[j]).abs();
                if scaled > row_max {
                    row_max = scaled;
                }
            }
            assert!(
                (row_max - 1.0).abs() <= knight_ruiz_deviation(INFNORM_TOL),
                "row {} max = {}, expected 1",
                i,
                row_max
            );
        }
    }
}
