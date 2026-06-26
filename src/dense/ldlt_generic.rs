//! Generic dense Bunch-Kaufman LDLᵀ factorization over any [`Scalar`] field.
//!
//! This is a clean, unblocked, correctness-first implementation of the
//! symmetric-indefinite factorization `Pᵀ A P = L D Lᵀ`, where `L` is unit
//! lower triangular and `D` is block diagonal with 1×1 and 2×2 blocks. It is
//! generic over `T: Scalar`, so it serves both the real (`f64`) and the
//! complex-*symmetric* (`Complex<f64>`, PARDISO `mtype 6`) paths.
//!
//! The algorithm is the lower-triangular, right-looking Bunch-Kaufman scheme of
//! LAPACK's `?sytf2` (`xSYTF2`). For the complex-symmetric case this is exactly
//! `zsytf2`: identical control flow to the real `dsytf2`, with magnitudes taken
//! as the complex modulus `|z|` and **no conjugation** anywhere (the matrix is
//! symmetric `A = Aᵀ`, not Hermitian). The pivot threshold is the classical
//! `α = (1 + √17)/8`.
//!
//! This module intentionally does **not** reuse the heavily optimized f64 path
//! in [`crate::dense::factor`] (blocked, SIMD Schur updates, peek-ahead, rook
//! rescue, inertia). That path stays the f64 performance specialization; this
//! one is the shared generic reference and the complex-symmetric kernel.
//! Performance work (blocking, a complex Schur micro-kernel) comes later;
//! correctness first.

use crate::dense::matrix::SymmetricMatrix;
use crate::error::FeralError;
use crate::scalar::Scalar;

/// Result of a generic Bunch-Kaufman LDLᵀ factorization.
///
/// `Pᵀ A P = L D Lᵀ`. The permutation is symmetric (the same `P` acts on rows
/// and columns), so factoring preserves symmetry.
#[derive(Debug, Clone)]
pub struct LdltFactors<T> {
    pub n: usize,
    /// Unit lower triangular `L`, full n×n column-major (entry (i,j) at
    /// `j*n + i`). The diagonal is an explicit `1`; the strict upper triangle
    /// is zero. For a 2×2 pivot at columns `(k, k+1)` the intra-block entry
    /// `L[k+1][k]` is `0` (that coupling lives in `D`, not `L`).
    pub l: Vec<T>,
    /// Diagonal of the block-diagonal `D`, length `n`.
    pub d_diag: Vec<T>,
    /// Sub-diagonal of `D`, length `n`. `d_subdiag[k]` is the `(k+1, k)` entry
    /// of a 2×2 block starting at column `k`; it is `0` for 1×1 pivots and for
    /// the second column of a 2×2 block.
    pub d_subdiag: Vec<T>,
    /// `true` at the starting column of each 2×2 pivot block. The column after
    /// such a start is the block's second column; every other column is a 1×1
    /// pivot.
    pub two_by_two: Vec<bool>,
    /// Symmetric pivot permutation (forward): `perm[i] = j` means original
    /// index `j` occupies pivot position `i`.
    pub perm: Vec<usize>,
}

/// The Bunch-Kaufman pivot threshold `α = (1 + √17)/8 ≈ 0.6404`.
#[inline]
fn bk_alpha() -> f64 {
    (1.0 + 17.0_f64.sqrt()) / 8.0
}

/// Swap symmetric indices `p` and `q` (`p < q`) in a lower-triangle,
/// column-major working matrix. This swaps the corresponding rows *and*
/// columns across the whole matrix — including the already-computed `L`
/// columns to the left — so the partial factorization stays consistent. The
/// crossing element `(q, p)` maps to itself and is left in place.
fn swap_sym_lower<T: Scalar>(a: &mut [T], n: usize, p: usize, q: usize) {
    debug_assert!(p < q && q < n);
    // Column segment strictly below q: (i, p) <-> (i, q) for i > q.
    for i in (q + 1)..n {
        a.swap(p * n + i, q * n + i);
    }
    // Middle cross strip: (i, p) <-> (q, i) for p < i < q.
    for i in (p + 1)..q {
        a.swap(p * n + i, i * n + q);
    }
    // Diagonal: (p, p) <-> (q, q).
    a.swap(p * n + p, q * n + q);
    // Left row segments: (p, j) <-> (q, j) for j < p.
    for j in 0..p {
        a.swap(j * n + p, j * n + q);
    }
}

/// Factor a symmetric matrix `A` as `Pᵀ A P = L D Lᵀ` using unblocked
/// Bunch-Kaufman pivoting.
///
/// Works for any [`Scalar`]; for `T = Complex<f64>` this is the
/// complex-symmetric (`A = Aᵀ`) factorization. Returns
/// [`FeralError::NumericallyRankDeficient`] if a structurally zero pivot (1×1
/// of value 0, or a 2×2 block with zero determinant) is encountered.
pub fn factor_ldlt<T: Scalar>(matrix: &SymmetricMatrix<T>) -> Result<LdltFactors<T>, FeralError> {
    matrix.validate()?;
    let n = matrix.n;
    let alpha = bk_alpha();

    // Working copy; only the lower triangle (i >= j) is read/written.
    let mut a = matrix.data.clone();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut d_diag = vec![T::zero(); n];
    let mut d_subdiag = vec![T::zero(); n];
    let mut two_by_two = vec![false; n];

    let mut k = 0;
    while k < n {
        let absakk = a[k * n + k].magnitude();

        // colmax = largest |A[i][k]| below the diagonal, at row imax.
        let mut colmax = 0.0;
        let mut imax = k;
        for i in (k + 1)..n {
            let m = a[k * n + i].magnitude();
            if m > colmax {
                colmax = m;
                imax = i;
            }
        }

        // Decide pivot size (kstep) and which index to interchange with (kp).
        let kstep;
        let kp;
        if absakk.max(colmax) == 0.0 {
            // Structurally zero column: singular.
            return Err(FeralError::NumericallyRankDeficient);
        } else if absakk >= alpha * colmax {
            kstep = 1;
            kp = k;
        } else {
            // rowmax = largest off-diagonal magnitude in row imax.
            let mut rowmax = 0.0;
            for j in k..imax {
                let m = a[j * n + imax].magnitude(); // A[imax][j], imax > j
                if m > rowmax {
                    rowmax = m;
                }
            }
            for i in (imax + 1)..n {
                let m = a[imax * n + i].magnitude(); // A[i][imax]
                if m > rowmax {
                    rowmax = m;
                }
            }

            if absakk >= alpha * colmax * (colmax / rowmax) {
                kstep = 1;
                kp = k;
            } else if a[imax * n + imax].magnitude() >= alpha * rowmax {
                kstep = 1;
                kp = imax;
            } else {
                kstep = 2;
                kp = imax;
            }
        }

        if kstep == 1 {
            // 1×1 pivot. Interchange index k with kp if needed.
            if kp != k {
                swap_sym_lower(&mut a, n, k, kp);
                perm.swap(k, kp);
            }
            let d = a[k * n + k];
            if d == T::zero() {
                return Err(FeralError::NumericallyRankDeficient);
            }
            d_diag[k] = d;
            let dinv = d.recip();

            // Rank-1 trailing update using the original pivot column, then
            // overwrite the column with the multipliers L[i][k] = A[i][k]/d.
            for j in (k + 1)..n {
                let wj_dinv = a[k * n + j] * dinv; // A[j][k] / d
                if wj_dinv != T::zero() {
                    for i in j..n {
                        a[j * n + i] = a[j * n + i] - a[k * n + i] * wj_dinv;
                    }
                }
            }
            for i in (k + 1)..n {
                a[k * n + i] = a[k * n + i] * dinv;
            }
            k += 1;
        } else {
            // 2×2 pivot at (k, k+1). Interchange index k+1 with kp if needed.
            if kp != k + 1 {
                swap_sym_lower(&mut a, n, k + 1, kp);
                perm.swap(k + 1, kp);
            }
            let d11 = a[k * n + k];
            let d21 = a[k * n + (k + 1)]; // A[k+1][k]
            let d22 = a[(k + 1) * n + (k + 1)];
            let det = d11 * d22 - d21 * d21;
            if det == T::zero() {
                return Err(FeralError::NumericallyRankDeficient);
            }
            let detinv = det.recip();
            d_diag[k] = d11;
            d_subdiag[k] = d21;
            d_diag[k + 1] = d22;
            two_by_two[k] = true;

            // Multiplier columns L_i = D⁻¹ · [A[i][k], A[i][k+1]]ᵀ for i >= k+2,
            // with D⁻¹ = (1/det)·[[d22, -d21], [-d21, d11]].
            let mut l1 = vec![T::zero(); n];
            let mut l2 = vec![T::zero(); n];
            for i in (k + 2)..n {
                let wik = a[k * n + i];
                let wik1 = a[(k + 1) * n + i];
                l1[i] = (d22 * wik - d21 * wik1) * detinv;
                l2[i] = (d11 * wik1 - d21 * wik) * detinv;
            }
            // Trailing update A22[i][j] -= W1_i·l1_j + W2_i·l2_j, reading the
            // original pivot columns (still intact) before overwriting them.
            for j in (k + 2)..n {
                let l1j = l1[j];
                let l2j = l2[j];
                for i in j..n {
                    a[j * n + i] =
                        a[j * n + i] - a[k * n + i] * l1j - a[(k + 1) * n + i] * l2j;
                }
            }
            for i in (k + 2)..n {
                a[k * n + i] = l1[i];
                a[(k + 1) * n + i] = l2[i];
            }
            k += 2;
        }
    }

    // Extract L from the working storage, honoring block structure.
    let mut l = vec![T::zero(); n * n];
    let one = T::one();
    let mut c = 0;
    while c < n {
        if two_by_two[c] {
            l[c * n + c] = one;
            l[(c + 1) * n + (c + 1)] = one;
            // L[c+1][c] stays 0 (intra-block coupling lives in D).
            for i in (c + 2)..n {
                l[c * n + i] = a[c * n + i];
                l[(c + 1) * n + i] = a[(c + 1) * n + i];
            }
            c += 2;
        } else {
            l[c * n + c] = one;
            for i in (c + 1)..n {
                l[c * n + i] = a[c * n + i];
            }
            c += 1;
        }
    }

    Ok(LdltFactors {
        n,
        l,
        d_diag,
        d_subdiag,
        two_by_two,
        perm,
    })
}

/// Solve `A · x = rhs` from a generic LDLᵀ factorization.
///
/// Applies the five-step sequence `x = P L⁻ᵀ D⁻¹ L⁻¹ Pᵀ rhs`.
pub fn solve_ldlt<T: Scalar>(factors: &LdltFactors<T>, rhs: &[T]) -> Result<Vec<T>, FeralError> {
    let n = factors.n;
    if rhs.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: rhs.len(),
        });
    }
    let l = &factors.l;

    // y = Pᵀ · rhs : y[i] = rhs[perm[i]].
    let mut y = vec![T::zero(); n];
    for i in 0..n {
        y[i] = rhs[factors.perm[i]];
    }

    // Forward solve L · z = y (unit lower triangular), in place in y.
    for i in 0..n {
        let mut acc = y[i];
        for j in 0..i {
            acc = acc - l[j * n + i] * y[j];
        }
        y[i] = acc;
    }

    // D-block solve: w = D⁻¹ · z, in place in y.
    let mut k = 0;
    while k < n {
        if factors.two_by_two[k] {
            let d11 = factors.d_diag[k];
            let d21 = factors.d_subdiag[k];
            let d22 = factors.d_diag[k + 1];
            let det = d11 * d22 - d21 * d21;
            if det == T::zero() {
                return Err(FeralError::NumericallyRankDeficient);
            }
            let detinv = det.recip();
            let z0 = y[k];
            let z1 = y[k + 1];
            y[k] = (d22 * z0 - d21 * z1) * detinv;
            y[k + 1] = (d11 * z1 - d21 * z0) * detinv;
            k += 2;
        } else {
            let d = factors.d_diag[k];
            if d == T::zero() {
                return Err(FeralError::NumericallyRankDeficient);
            }
            y[k] = y[k] * d.recip();
            k += 1;
        }
    }

    // Backward solve Lᵀ · v = w (unit upper triangular), in place in y.
    for i in (0..n).rev() {
        let mut acc = y[i];
        for j in (i + 1)..n {
            acc = acc - l[i * n + j] * y[j];
        }
        y[i] = acc;
    }

    // x = P · v : x[perm[i]] = v[i].
    let mut x = vec![T::zero(); n];
    for i in 0..n {
        x[factors.perm[i]] = y[i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    /// ‖A·x − b‖∞ for a real or complex symmetric `A` (via `symv`).
    fn residual_inf<T: Scalar>(a: &SymmetricMatrix<T>, x: &[T], b: &[T]) -> f64 {
        let mut ax = vec![T::zero(); a.n];
        a.symv(x, &mut ax);
        (0..a.n)
            .map(|i| (ax[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    // ---- f64 ----------------------------------------------------------------

    #[test]
    fn f64_indefinite_2x2_pivot() {
        // A = [[0, 1], [1, 0]] has a zero diagonal: forces a 2×2 pivot.
        let a = SymmetricMatrix::<f64>::from_lower_triangle(2, &[(0, 0, 0.0), (1, 0, 1.0), (1, 1, 0.0)]);
        let f = factor_ldlt(&a).unwrap();
        let b = vec![3.0, 5.0];
        let x = solve_ldlt(&f, &b).unwrap();
        // A x = [x1, x0] = b  =>  x = [5, 3].
        assert!((x[0] - 5.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
        assert!(residual_inf(&a, &x, &b) < 1e-12);
    }

    #[test]
    fn f64_spd_matches_reference_solver() {
        // Cross-validate against the validated production f64 solver.
        let entries = [
            (0, 0, 4.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (2, 0, 2.0),
            (2, 1, -1.0),
            (2, 2, 5.0),
        ];
        let a = SymmetricMatrix::<f64>::from_lower_triangle(3, &entries);
        let b = vec![1.0, 2.0, 3.0];

        let f = factor_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();

        let params = crate::dense::factor::BunchKaufmanParams::default();
        let (ref_factors, _inertia) = crate::dense::factor::factor(&a, &params).unwrap();
        let x_ref = crate::dense::solve::solve(&ref_factors, &b).unwrap();

        for i in 0..3 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1e-9,
                "x[{}]={} vs ref {}",
                i,
                x[i],
                x_ref[i]
            );
        }
        assert!(residual_inf(&a, &x, &b) < 1e-12);
    }

    #[test]
    fn f64_larger_indefinite_residual() {
        // A symmetric indefinite 5×5 exercising both 1×1 and 2×2 pivots.
        let entries = [
            (0, 0, 1.0),
            (1, 0, 3.0),
            (1, 1, 2.0),
            (2, 0, 0.5),
            (2, 1, -1.0),
            (2, 2, -4.0),
            (3, 1, 2.0),
            (3, 2, 1.0),
            (3, 3, 0.0),
            (4, 0, -2.0),
            (4, 3, 3.0),
            (4, 4, 1.0),
        ];
        let a = SymmetricMatrix::<f64>::from_lower_triangle(5, &entries);
        let b = vec![1.0, -2.0, 3.0, 0.5, -1.5];
        let f = factor_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-10);
    }

    // ---- Complex symmetric (A = Aᵀ, PARDISO mtype 6) ------------------------

    #[test]
    fn complex_antidiagonal_2x2_pivot() {
        let c = |re, im| Complex::new(re, im);
        // A = [[0, 1], [1, 0]] (complex symmetric, zero diagonal -> 2×2 pivot).
        let a = SymmetricMatrix::<Complex<f64>>::from_lower_triangle(
            2,
            &[(0, 0, c(0.0, 0.0)), (1, 0, c(1.0, 0.0)), (1, 1, c(0.0, 0.0))],
        );
        let f = factor_ldlt(&a).unwrap();
        let b = vec![c(1.0, 1.0), c(2.0, -1.0)];
        let x = solve_ldlt(&f, &b).unwrap();
        // A x = [x1, x0] = b  =>  x = [2 - i, 1 + i].
        assert!((x[0] - c(2.0, -1.0)).norm() < 1e-12);
        assert!((x[1] - c(1.0, 1.0)).norm() < 1e-12);
        assert!(residual_inf(&a, &x, &b) < 1e-12);
    }

    #[test]
    fn complex_diagonal_pivots() {
        let c = |re, im| Complex::new(re, im);
        // Diagonally dominant complex symmetric: all 1×1 pivots.
        let a = SymmetricMatrix::<Complex<f64>>::from_lower_triangle(
            3,
            &[
                (0, 0, c(4.0, 1.0)),
                (1, 0, c(1.0, -1.0)),
                (1, 1, c(5.0, -2.0)),
                (2, 0, c(0.5, 0.0)),
                (2, 1, c(-1.0, 0.5)),
                (2, 2, c(6.0, 1.0)),
            ],
        );
        let b = vec![c(1.0, 0.0), c(0.0, 2.0), c(-1.0, 1.0)];
        let f = factor_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-11);
    }

    #[test]
    fn complex_indefinite_mixed_pivots() {
        let c = |re, im| Complex::new(re, im);
        // 5×5 complex symmetric with small/zero diagonals to force 2×2 pivots.
        let a = SymmetricMatrix::<Complex<f64>>::from_lower_triangle(
            5,
            &[
                (0, 0, c(0.0, 0.0)),
                (1, 0, c(2.0, 1.0)),
                (1, 1, c(1.0, -1.0)),
                (2, 0, c(1.0, 0.0)),
                (2, 1, c(0.0, 1.0)),
                (2, 2, c(0.0, 0.0)),
                (3, 1, c(-1.0, 2.0)),
                (3, 2, c(3.0, 0.0)),
                (3, 3, c(2.0, 1.0)),
                (4, 0, c(1.0, 1.0)),
                (4, 3, c(0.0, -1.0)),
                (4, 4, c(1.0, 0.0)),
            ],
        );
        let b = vec![
            c(1.0, 0.0),
            c(0.0, 1.0),
            c(-1.0, 1.0),
            c(2.0, 0.0),
            c(0.5, -0.5),
        ];
        let f = factor_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-10,
            "residual too large: {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn singular_column_is_rejected() {
        // A fully zero matrix is structurally singular.
        let a = SymmetricMatrix::<f64>::zeros(2);
        assert!(matches!(
            factor_ldlt(&a),
            Err(FeralError::NumericallyRankDeficient)
        ));
    }
}
