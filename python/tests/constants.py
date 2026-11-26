"""Bounds of the Python tests and the quantities that they are derived from.

U64 is the unit roundoff u of float64, 2^-53 = 1.1e-16, the largest relative
error of one rounded operation (Higham, Accuracy and Stability of Numerical
Algorithms, 2002, Sec. 2.2). U32 is the unit roundoff of float32, 2^-24 =
6.0e-8, for factors and solves in single precision.

LAPACK_THRESHOLD is the acceptance threshold of the test ratios in LAPACK's
test suite (the value 30.0 of TESTING/dtest.in in Reference-LAPACK). Its
routine dget02 accepts a solve if ||b - A x|| / (||A|| ||x|| eps) is below it,
where eps = dlamch('Epsilon') is the unit roundoff.

UC64 is the unit roundoff of complex128 arithmetic in an analysis written for
real arithmetic. A complex sum is rounded with a relative error of at most u, a
product with at most sqrt(2) gamma_2 and a quotient with at most sqrt(2)
gamma_4 (Higham 2002, Lemma 3.5), so sqrt(2) gamma_4 of U64 = 6.3e-16 bounds
every complex operation, and the real bounds hold with it in place of u. UC32
is the same for complex64, sqrt(2) gamma_4 of U32 = 3.4e-7.
"""

import numpy as np

U64 = float(np.finfo(np.float64).eps) / 2
U32 = float(np.finfo(np.float32).eps) / 2
LAPACK_THRESHOLD = 30.0
UC64 = np.sqrt(2) * 4 * U64 / (1 - 4 * U64)
UC32 = np.sqrt(2) * 4 * U32 / (1 - 4 * U32)


def gamma(k, u):
    """gamma_k = k u / (1 - k u) (Higham 2002, Lemma 3.1). A sum or an inner
    product of k terms computed in any order with the unit roundoff u is within
    gamma_k |x|^T |y| of the exact value (Higham 2002, Eq. 3.5)."""
    return k * u / (1 - k * u)


def solve_backward_error(n, u):
    """Bound on the normwise backward error omega = ||b - A x|| / (||A|| ||x||
    + ||b||) of a solve of order n with an LU or LDL^T factorization in the
    unit roundoff u. The computed solution satisfies (A + dA) x = b with |dA|
    <= gamma_3n |L| |U| (Higham 2002, Thm. 9.4), so omega <= gamma_3n g with
    the growth g = || |L| |U| || / ||A||. Threshold pivoting does not bound g
    by a small number a priori, so the tests use the acceptance rule of
    LAPACK's test suite, LAPACK_THRESHOLD u, and multiply it by n, the length
    of the longest inner product, which LAPACK leaves out because its test
    orders are at most 50. The bound 30 n u equals gamma_3n g for g = 10."""
    return LAPACK_THRESHOLD * n * u


def forward_error(cond, omega):
    """Bound on the relative normwise forward error ||x - x*|| / ||x*|| of a
    solution whose normwise backward error is at most omega, for a matrix with
    the condition number cond = ||A|| ||A^-1||. Perturbations ||dA|| <= omega
    ||A|| and ||db|| <= omega ||b|| move the solution by at most 2 omega cond /
    (1 - omega cond) of itself (Higham 2002, Thm. 7.2)."""
    return 2 * omega * cond / (1 - omega * cond)


def recomputed_residual(tol, b_norm, abs_norm, n, k, u):
    """Bound on the norm of b - A x of order n that a test recomputes for an
    iterate whose residual norm the solver computed as at most tol ||b||, the
    stopping test of the Krylov solvers and of iterative refinement. Each
    component of b - A x is computed with an error of at most sqrt(2)
    gamma_(k+2) (|b| + |A| |x|)_i, where k is the largest number of entries in
    a row of A (Higham 2002, Eq. 3.5, with the factor sqrt(2) gamma_2 of a
    complex product from Lemma 3.5, which covers real operands too). The
    solver's and the test's evaluations therefore differ by at most 2 sqrt(2)
    gamma_(k+2) || |b| + |A| |x| ||, which is abs_norm here in the norm that the
    test uses. A 2-norm of n terms is computed with a relative
    error of at most gamma_(n/2 + 2), so the norms of b and of the residual in
    the solver and in the test add at most gamma_(2n + 8) of tol ||b||."""
    return tol * b_norm * (1 + gamma(2 * n + 8, u)) + 2 * np.sqrt(2) * gamma(k + 2, u) * abs_norm


def log_det_error(omega, a_norm, inverse_sum):
    """Bound on the error of log det A, real part log |det A| and imaginary
    part the phase angle, computed from a factorization whose normwise backward
    error in the infinity norm is at most omega, with a_norm = ||A||_inf and
    inverse_sum the sum of the magnitudes of the entries of A^-1. The factors
    are those of A + dA with |dA_ij| <= ||dA||_inf <= omega ||A||_inf, so log
    det (A + dA) - log det A = log det (I + E) with E = A^-1 dA. The sum of the
    magnitudes of the entries of E is at most e = omega ||A||_inf sum |A^-1_ij|,
    which bounds its nuclear norm and therefore the sum of the magnitudes of its
    eigenvalues (Weyl's inequality). With |log (1 + lambda)| <= |lambda| / (1 -
    e) for |lambda| <= e < 1 the error is at most e / (1 - e). The rounding of
    the sum of the logarithms of the pivots is a further gamma_n times the sum
    of their magnitudes and is not included."""
    e = omega * a_norm * inverse_sum
    return e / (1 - e)


def recursive_gap(iters, k, u, ax_max, r_max):
    """Bound on the drift ||b - A x_j - r_j|| between the true residual and the
    recursively updated residual r_j of a method that updates x_(j+1) = x_j +
    alpha p_j and r_(j+1) = r_j - alpha A p_j with A p_j formed by a product with
    A (COCG and CG), after iters steps from x_0 = 0 and r_0 = b, where the drift
    is zero. In one step with complex operations of unit roundoff u (Higham
    2002, Lemma 3.5), alpha p is formed with an error of at most gamma_3 |alpha
    p|, A p with at most gamma_(k+2) |A| |p| for k entries in a row of A, alpha
    A p with a further gamma_3, and the two sums with u |x_(j+1)| and u
    |r_(j+1)|. These change the drift by at most gamma_(k+8) |A| |alpha p| plus
    u |A| |x_(j+1)| plus u |r_(j+1)|, and |alpha p| <= (1 + gamma_8) (|x_(j+1)|
    plus |x_j|) bounds it by gamma_(k+17) (|A| |x_(j+1)| plus |A| |x_j|) plus u
    |r_(j+1)|. The sum over the steps is at most iters (2 gamma_(k+17) ax_max
    plus gamma_2 r_max), where ax_max is the largest norm of |A| |x_j| and r_max
    the largest computed norm of r_j over the iterates. The same function is in
    tests/common/constants.rs of the Rust tests."""
    return iters * (2 * gamma(k + 17, u) * ax_max + gamma(2, u) * r_max)
