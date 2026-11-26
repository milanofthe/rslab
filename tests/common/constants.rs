/// Unit roundoff u of f64, 2^-53 = 1.1e-16, the largest relative error of one
/// rounded operation (Higham, Accuracy and Stability of Numerical Algorithms,
/// 2002, Sec. 2.2).
pub const U64: f64 = f64::EPSILON / 2.0;

/// Unit roundoff u of f32, 2^-24 = 6.0e-8, for factors and solves in single
/// precision.
pub const U32: f64 = f32::EPSILON as f64 / 2.0;

/// gamma_k = k u / (1 - k u) (Higham 2002, Lemma 3.1). A sum or an inner
/// product of k terms computed in any order with the unit roundoff u is within
/// gamma_k |x|^T |y| of the exact value (Higham 2002, Eq. 3.5), and a product
/// of k rounded factors is within gamma_k of the exact product.
pub fn gamma(k: usize, u: f64) -> f64 {
    let ku = k as f64 * u;
    ku / (1.0 - ku)
}

/// Acceptance threshold of the test ratios in LAPACK's test suite (the value
/// 30.0 of `TESTING/dtest.in` in Reference-LAPACK). Its routine `dget02`
/// accepts a solve if ||b - A x|| / (||A|| ||x|| eps) is below it, where eps =
/// `dlamch('Epsilon')` is the unit roundoff.
pub const LAPACK_THRESHOLD: f64 = 30.0;

/// Bound on the normwise backward error omega = ||b - A x|| / (||A|| ||x|| +
/// ||b||) of a solve of order n with an LU or LDL^T factorization in the unit
/// roundoff u. The computed solution satisfies (A + dA) x = b with |dA| <=
/// gamma_3n |L| |U| (Higham 2002, Thm. 9.4), so omega <= gamma_3n g with the
/// growth g = || |L| |U| || / ||A||. Threshold pivoting does not bound g by a
/// small number a priori, so the tests use the acceptance rule of LAPACK's
/// test suite, [`LAPACK_THRESHOLD`] u, and multiply it by n, the length of the
/// longest inner product, which LAPACK leaves out because its test orders are
/// at most 50. The bound 30 n u equals gamma_3n g for g = 10. For n = 1000 it
/// is 3.3e-12 in f64 and 1.8e-3 in f32.
pub fn solve_backward_error(n: usize, u: f64) -> f64 {
    LAPACK_THRESHOLD * n as f64 * u
}

/// Bound on the relative normwise forward error ||x - x*|| / ||x*|| of a
/// solution whose normwise backward error is at most omega, for a matrix with
/// the condition number cond = ||A|| ||A^-1||. Perturbations ||dA|| <= omega
/// ||A|| and ||db|| <= omega ||b|| move the solution by at most 2 omega cond /
/// (1 - omega cond) of itself (Higham 2002, Thm. 7.2).
pub fn forward_error(cond: f64, omega: f64) -> f64 {
    2.0 * omega * cond / (1.0 - omega * cond)
}

/// Bound on the norm of b - A x of order n that a test recomputes for an
/// iterate whose residual norm the solver computed as at most tol ||b||, the
/// stopping test of the Krylov solvers and of iterative refinement. Each
/// component of b - A x is computed with an error of at most sqrt(2)
/// gamma_(k+2) (|b| + |A| |x|)_i, where k is the largest number of entries in
/// a row of A (Higham 2002, Eq. 3.5, with the factor sqrt(2) gamma_2 of a
/// complex product from Lemma 3.5, which covers real operands too). The
/// solver's and the test's evaluations therefore differ by at most 2 sqrt(2)
/// gamma_(k+2) || |b| + |A| |x| ||, which is `abs_norm` here in the norm that
/// the test uses. A 2-norm of n terms is computed with a
/// relative error of at most gamma_(n/2 + 2), so the norms of b and of the
/// residual in the solver and in the test add at most gamma_(2n + 8) of tol
/// ||b||.
pub fn recomputed_residual(
    tol: f64,
    b_norm: f64,
    abs_norm: f64,
    n: usize,
    k: usize,
    u: f64,
) -> f64 {
    tol * b_norm * (1.0 + gamma(2 * n + 8, u))
        + 2.0 * std::f64::consts::SQRT_2 * gamma(k + 2, u) * abs_norm
}

/// Unit roundoff of complex f64 arithmetic in an analysis written for real
/// arithmetic. A complex sum is rounded with a relative error of at most u, a
/// product with at most sqrt(2) gamma_2 and a quotient with at most sqrt(2)
/// gamma_4 (Higham 2002, Lemma 3.5), so sqrt(2) gamma_4 of U64 = 6.3e-16 bounds
/// every complex operation, and the real bounds hold with it in place of u.
pub const UC64: f64 = std::f64::consts::SQRT_2 * 4.0 * U64 / (1.0 - 4.0 * U64);

/// [`UC64`] for complex f32 arithmetic, sqrt(2) gamma_4 of U32 = 3.4e-7.
pub const UC32: f64 = std::f64::consts::SQRT_2 * 4.0 * U32 / (1.0 - 4.0 * U32);

/// Upper bound on the condition number ||A||_inf ||A^-1||_inf of a matrix of
/// order n that is strictly diagonally dominant by rows, from the magnitudes
/// (row, column, |a_ij|) of its entries. Varah's bound ||A^-1||_inf <= 1 /
/// min_i (|a_ii| - sum_(j != i) |a_ij|) (J. M. Varah, Linear Algebra Appl. 11,
/// 1975, pp. 3-5) holds for every such matrix. The function panics if a row is
/// not strictly dominant, where the bound does not apply.
pub fn varah_condition(n: usize, entries: impl IntoIterator<Item = (usize, usize, f64)>) -> f64 {
    let mut diag = vec![0.0f64; n];
    let mut off = vec![0.0f64; n];
    for (i, j, m) in entries {
        if i == j {
            diag[i] += m;
        } else {
            off[i] += m;
        }
    }
    let margin = (0..n)
        .map(|i| diag[i] - off[i])
        .fold(f64::INFINITY, f64::min);
    assert!(
        margin > 0.0,
        "the matrix is not strictly diagonally dominant by rows"
    );
    let norm = (0..n).map(|i| diag[i] + off[i]).fold(0.0f64, f64::max);
    norm / margin
}

/// Bound on the error of log det A, real part log |det A| and imaginary part
/// the phase angle, computed from a factorization whose normwise backward
/// error in the infinity norm is at most omega, with `a_norm` = ||A||_inf and
/// `inverse_sum` the sum of the magnitudes of the entries of A^-1. The factors
/// are those of A + dA with |dA_ij| <= ||dA||_inf <= omega ||A||_inf, so log
/// det (A + dA) - log det A = log det (I + E) with E = A^-1 dA. The sum of the
/// magnitudes of the entries of E is at most e = omega ||A||_inf sum |A^-1_ij|,
/// which bounds its nuclear norm and therefore the sum of the magnitudes of
/// its eigenvalues (Weyl's inequality). With |log (1 + lambda)| <= |lambda| /
/// (1 - e) for |lambda| <= e < 1 the error is at most e / (1 - e). The rounding
/// of the sum of the logarithms of the pivots is a further gamma_n times the
/// sum of their magnitudes and is not included.
pub fn log_det_error(omega: f64, a_norm: f64, inverse_sum: f64) -> f64 {
    let e = omega * a_norm * inverse_sum;
    e / (1.0 - e)
}

/// Relative error of `exp` and `ln` of the platform's math library, at most 1
/// ulp of the result, which is 2 u = 2.2e-16 relative to it. The GNU C
/// Library documents errors of at most 1 ulp for both on x86_64 (glibc
/// manual, Sec. 19.7, Known Maximum Errors in Math Functions).
pub const LIBM_ULP: f64 = 2.0 * U64;

/// Relative error of exp(y) for a y that is formed by exact operations from
/// logarithms that `ln` returns within [`LIBM_ULP`]. The error of y is then
/// at most LIBM_ULP |y|, which changes exp(y) by a factor of exp(LIBM_ULP |y|)
/// = 1 + LIBM_ULP |y| to first order, and `exp` adds LIBM_ULP. The bound is
/// LIBM_ULP (|y| + 1) for |y| LIBM_ULP below 1e-3.
pub fn exp_of_logs(y: f64) -> f64 {
    LIBM_ULP * (y.abs() + 1.0)
}

/// Deviation from one of the row maxima of D A D after the symmetric
/// Knight-Ruiz step d_i <- d_i / sqrt(m_i) that follows a sweep in which every
/// row maximum m_i was within tol of one, the stopping test of
/// `compute_infnorm`. An entry of the new D A D is v_ij / sqrt(m_i m_j) with
/// v_ij <= min(m_i, m_j) by symmetry, so it is at most one, and the entry that
/// attained m_i becomes sqrt(m_i / m_j) >= sqrt((1 - tol) / (1 + tol)) >= 1 -
/// tol. In floating point the row maxima carry the rounding of two products
/// (gamma_2), the new d_i that of a square root and a quotient of the rounded
/// m_i (gamma_3 each for d_i and d_j), and the test's product d_i a_ij d_j
/// another gamma_2, together gamma_10.
pub fn knight_ruiz_deviation(tol: f64) -> f64 {
    tol + gamma(10, U64)
}

/// Acceptance threshold of the test ratios of the nonsymmetric eigenvalue
/// routines in LAPACK's test suite (the value 20.0 of `TESTING/nep.in` in
/// Reference-LAPACK).
pub const NEP_THRESHOLD: f64 = 20.0;

/// Bound on the backward error of a Schur form or an eigenpair of a dense
/// matrix of order n computed by Householder reduction and the shifted QR
/// algorithm in the unit roundoff u, relative to ||A||_1. The QR algorithm is
/// backward stable: the computed Q and T satisfy Q T Q^H = A + E with ||E|| <=
/// p(n) u ||A|| and Q^H Q = I + F with ||F|| <= p(n) u (Golub and Van Loan,
/// Matrix Computations, 4th ed., 2013, Sec. 7.5.6). LAPACK's tests `zhst01`
/// and `zget22` accept ||A - Q T Q^H||_1 / (n ||A||_1 ulp), ||I - Q^H Q||_1 /
/// (n ulp), ||A x - lambda x||_1 / (||A||_1 ||x||_1 ulp) and the deviation of
/// ||x|| from one over n ulp below [`NEP_THRESHOLD`], each computed in
/// floating point by the test, with ulp = 2 u. The bound is NEP_THRESHOLD n
/// ulp for all four, with the factor n also on the residual of an eigenpair,
/// as in [`solve_backward_error`].
pub fn eig_backward_error(n: usize, u: f64) -> f64 {
    NEP_THRESHOLD * n as f64 * 2.0 * u
}

/// First-order bound on the change of a simple eigenvalue of a matrix of order
/// n under a perturbation of 1-norm `perturbation`, for the eigenvalue's
/// condition number cond = ||x||_2 ||l||_2 / |l^T x| with its right and left
/// eigenvectors x and l (Golub and Van Loan 2013, Sec. 7.2.2). The change is
/// at most cond ||dA||_2 + O(||dA||^2), and ||dA||_2 <= sqrt(n) ||dA||_1.
pub fn eigenvalue_error(n: usize, cond: f64, perturbation: f64) -> f64 {
    cond * (n as f64).sqrt() * perturbation
}

/// Loss of orthogonality ||I - V^H V||_2 of k vectors of length n that a
/// Gram-Schmidt process with reorthogonalization (CGS2, MGS with a second
/// pass, or the block variants of the Krylov solvers) computes in the unit
/// roundoff u, and the residual ||R - V S||_2 / ||R||_2 of the factorization
/// R = V S that it computes. Giraud, Langou and Rozloznik (Comput. Math.
/// Appl. 50, 2005, pp. 1069-1075) bound both by c(k, n) u for a numerically
/// nonsingular R, with c a low-degree polynomial that the paper does not fix.
/// The tests take c(k, n) = 2 k n, which is k n eps, the bound that
/// `block_gmres_delayed_cgs2_keeps_the_basis_orthonormal` states. For k = 200
/// and n = 400 it is 1.8e-11 in f64.
pub fn gram_schmidt(k: usize, n: usize, u: f64) -> f64 {
    2.0 * k as f64 * n as f64 * u
}

/// Distance, relative to ||A||_2, of a Ritz value of an Arnoldi basis of k
/// vectors of length n that spans an invariant subspace of a normal operator A
/// from an eigenvalue of A, in the unit roundoff u. The computed Hessenberg
/// matrix H is the projection V^H A V of A onto the basis up to the residual of
/// the Arnoldi relation and up to the loss of orthogonality of V, each at most
/// [`gram_schmidt`] (k, n, u) ||A||_2. The Schur form of H is exact for H + E,
/// where E is the backward error of a sequence of unitary transformations of a
/// k x k matrix, which the same bound covers with n >= k. The restriction of a
/// normal A to an invariant subspace is normal, so by the Bauer-Fike theorem
/// with an eigenvector condition of one (Golub, Van Loan, Matrix
/// Computations, 2013, Thm. 7.2.2) each eigenvalue moves by at most the sum of
/// the three perturbations.
pub fn ritz_value(k: usize, n: usize, u: f64) -> f64 {
    3.0 * gram_schmidt(k, n, u)
}

/// Bound on the drift ||b - A x_j - r_j|| between the true residual and the
/// recursively updated residual r_j of a method that updates x_(j+1) = x_j +
/// alpha p_j and r_(j+1) = r_j - alpha A p_j with A p_j formed by a product
/// with A (COCG and CG), after `iters` steps from x_0 = 0 and r_0 = b, where
/// the drift is zero. In one step with complex operations of unit roundoff u
/// (Higham 2002, Lemma 3.5), alpha p is formed with an error of at most gamma_3
/// |alpha p|, A p with at most gamma_(k+2) |A| |p| for k entries in a row of A,
/// alpha A p with a further gamma_3, and the two sums with u |x_(j+1)| and u
/// |r_(j+1)|. These change the drift by at most gamma_(k+8) |A| |alpha p| + u
/// |A| |x_(j+1)| + u |r_(j+1)|, and |alpha p| <= (1 + gamma_8) (|x_(j+1)| +
/// |x_j|) bounds it by gamma_(k+17) (|A| |x_(j+1)| + |A| |x_j|) + u
/// |r_(j+1)|. The sum over the steps is at most iters (2 gamma_(k+17) ax_max +
/// gamma_2 r_max), where ax_max is the largest norm of |A| |x_j| and r_max the
/// largest computed norm of r_j over the iterates; gamma_2 covers the rounding
/// of the computed norm.
pub fn recursive_gap(iters: usize, k: usize, u: f64, ax_max: f64, r_max: f64) -> f64 {
    iters as f64 * (2.0 * gamma(k + 17, u) * ax_max + gamma(2, u) * r_max)
}

/// Bound, relative to (|L| |U| |x| + |F| |x| + |R A| |x|)_i, on the difference
/// that a test measures between (L (U x) + F x)_i and (P R A Q x)_i for the
/// factors L U + F of the row-scaled and permuted A of order n. The factors
/// satisfy L U = R A + dA with |dA| <= gamma_n |L| |U| (Higham 2002, Thm.
/// 9.3), F holds entries of R A rounded once, the two products U x and L (U
/// x) add gamma_n each, F x and A x add gamma_n each, the scaling by R adds
/// one rounding and the sum one more, so the total is at most gamma_(3n+3) of
/// the sum of magnitudes. In complex arithmetic the products carry sqrt(2)
/// gamma_2 instead of u (Lemma 3.5), which [`UC64`] in place of u covers.
pub fn factor_product(n: usize, u: f64) -> f64 {
    gamma(3 * n + 3, u)
}

/// Factor by which the least-squares residual estimate |g_(j+1)| / ||b|| of a
/// GMRES step may exceed the previous one |g_j| / ||b|| in the unit roundoff
/// u. In exact arithmetic g_(j+1) = -s_j g_j with |s_j| <= 1 for the Givens
/// rotation (c_j, s_j), so the estimates do not increase. The computed s_j =
/// h / r with r = sqrt(|f|^2 + |h|^2) is at most 1 + gamma_6 in magnitude (the
/// squared magnitudes, their sum, the root, the reciprocal and the product),
/// the complex product s_j g_j adds sqrt(2) gamma_2 <= gamma_3, and each
/// magnitude and division by ||b|| adds gamma_3 to the new estimate and
/// removes at most gamma_3 from the old one, together at most 1 + gamma_15.
pub fn estimate_growth(u: f64) -> f64 {
    1.0 + gamma(15, u)
}

/// Bound on the componentwise backward error max_i |b - A x|_i / (|A| |x|)_i
/// after one step of fixed-precision iterative refinement of a solve of order
/// n with the residual computed in the working precision of unit roundoff u.
/// Higham 2002, Thm. 12.3, bounds |b - A x| by 2 gamma_(n+1) |A| |x| for a
/// solver that satisfies its condition (12.5) and a system with u ||
/// |A^-1| |A| |x| ||_inf / min_i (|A| |x|)_i small enough, which a diagonally
/// dominant matrix with a backward stable LU meets. Further steps return the
/// best iterate, which keeps the bound.
pub fn fixed_refinement(n: usize, u: f64) -> f64 {
    2.0 * gamma(n + 1, u)
}

/// Entrywise error, with inner dimension k, of a complex product D = alpha C
/// plus beta A B formed by the conventional method or by Gauss's three-product
/// method, relative to |alpha| |C| plus |beta| M with M = (|Re A| plus |Im A|)
/// (|Re B| plus |Im B|), which is at least |A| |B|. The conventional method is
/// within gamma_k |A| |B| of A B in complex arithmetic, and the three-product
/// method
/// is within gamma_(k+4) M in each part (Higham 2002, Sec. 23.2.4). The scaling
/// by alpha and beta and the final sum add three complex operations, so
/// gamma_(k+7) with the complex unit roundoff [`UC64`] bounds both.
pub fn complex_gemm(k: usize) -> f64 {
    gamma(k + 7, UC64)
}

/// Upper bound on the condition number ||A||_inf ||A^-1||_inf of the
/// five-point Laplacian with Dirichlet boundary, 4 on the diagonal and -1 for
/// each of the at most four grid neighbours, on an m x m grid. ||A||_inf <= 8.
/// A is an M-matrix, so A^-1 >= 0 and ||A^-1||_inf = ||A^-1 1||_inf. The
/// vector w with w = i (m + 1 - i) / 2 at grid row i satisfies A w >= 1
/// entrywise, because the second difference of w across the rows is -1 and
/// the difference along a row is at least 0, so A^-1 1 <= w by the discrete
/// maximum principle, and max w <= (m + 1)^2 / 8. The bound is (m + 1)^2.
pub fn laplacian_condition(m: usize) -> f64 {
    let side = (m + 1) as f64;
    side * side
}

/// Bound on the relative change of the squared 2-norm of a column of a matrix
/// that starts as the identity and is multiplied by `rotations` Givens
/// rotations whose cosine and sine come from `cos` and `sin` of the math
/// library. A rotation changes two entries a, b of the column to c a - s b and
/// s a + c b. Each new entry is within gamma_2 (|c a| + |s b|) <= sqrt(2)
/// gamma_2 (a^2 + b^2)^(1/2) of its value for the rounded c and s, which moves
/// the sum of their squares by at most gamma_6 (a^2 + b^2), and the rounded c
/// and s satisfy c^2 + s^2 = 1 within 2 [`LIBM_ULP`]. To first order the
/// squared norm changes by at most rotations (gamma_6 + 2 LIBM_ULP) of itself.
pub fn givens_drift(rotations: usize) -> f64 {
    rotations as f64 * (gamma(6, U64) + 2.0 * LIBM_ULP)
}

/// Bound on the error of an entry b_i^T x_j of B^T A^-1 B for a complex
/// symmetric A, where x_j is a solution with normwise backward error at most
/// omega in the infinity norm. With r_j = b_j - A x_j and A^-T b_i = x_i, the
/// error is b_i^T A^-1 r_j = x_i^T r_j, at most ||x_i||_1 ||r_j||_inf <= omega
/// ||x_i||_1 (||A||_inf ||x_j||_inf + ||b_j||_inf). The arguments are the
/// largest of these norms over the columns. The bound involves the condition
/// of A only through the computed norms of the solutions.
pub fn bilinear_error(omega: f64, a_norm: f64, x_one: f64, x_inf: f64, b_inf: f64) -> f64 {
    omega * x_one * (a_norm * x_inf + b_inf)
}
