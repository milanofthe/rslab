//! FEM-derived structured-grid operators for the hard problem classes a pure
//! stencil Laplacian misses: the **curl-curl** time-harmonic Maxwell operator
//! (complex symmetric indefinite, with a large gradient near-null-space -- the
//! FEM edge-element / EM workload) and the **saddle-point** Stokes/KKT operator
//! (symmetric indefinite, `[A B^T; B -betaC]`). Both are assembled directly on a
//! structured grid with finite differences, no external FEM library, following
//! the standard discretizations (curl-curl `curl curl E - (omega^2 eps - i omega sigma) E`, Nedelec/Yee;
//! Stokes MAC / mixed-FEM `[A B^T; B 0]` with Brezzi-Pitkaranta pressure
//! stabilization `-betaC`). References: Jin, *The FEM in Electromagnetics*;
//! Elman-Silvester-Wathen, *Finite Elements and Fast Iterative Solvers* (IFISS);
//! Benzi-Golub-Liesen, *Numerical solution of saddle point problems*.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use num_complex::Complex;

use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;

fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for d in 1..dims.len() {
        s[d] = s[d - 1] * dims[d - 1];
    }
    s
}

fn divergence_row(k: usize, dims: &[usize], stride: &[usize], n: usize) -> Vec<(usize, f64)> {
    let ndim = dims.len();
    let mut e = Vec::with_capacity(2 * ndim);
    for a in 0..ndim {
        let c = (k / stride[a]) % dims[a];
        if c + 1 < dims[a] {
            e.push((a * n + (k + stride[a]), 0.5));
        }
        if c > 0 {
            e.push((a * n + (k - stride[a]), -0.5));
        }
    }
    e
}

fn vector_laplacian(
    dims: &[usize],
    stride: &[usize],
    n: usize,
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    let ndim = dims.len();
    let ndof = ndim * n;
    let mut diag = vec![0.0f64; ndof];
    let mut tri: Vec<(usize, usize, f64)> = Vec::new();
    for a in 0..ndim {
        let off = a * n;
        for i in 0..n {
            for d in 0..ndim {
                let c = (i / stride[d]) % dims[d];
                if c + 1 < dims[d] {
                    let j = i + stride[d];
                    diag[off + i] += 1.0;
                    diag[off + j] += 1.0;
                    tri.push((off + j, off + i, -1.0));
                } else {
                    diag[off + i] += 1.0;
                }
                if c == 0 {
                    diag[off + i] += 1.0;
                }
            }
        }
    }
    (diag, tri)
}

/// Time-harmonic **curl-curl** Maxwell operator on a structured grid:
/// `A = (curl curl) - omega^2 M + i omega sigma M`, discretized via the identity `curl curl = L - grad div`
/// (`L` the component-wise vector Laplacian, `grad div = D^TD` from the discrete
/// divergence `D`). `M` is the lumped mass (identity). The `-omega^2 M` shift makes the
/// operator **indefinite**; the `i omega sigma M` conductivity term makes it **complex
/// symmetric** (`A = A^T`, not Hermitian); the curl-curl part is singular on
/// discrete gradients, so the system has the gradient near-null-space that makes
/// edge-element EM problems hard for iterative solvers and a stress test for a
/// direct one. `dims` is `[nx, ny]` (2 components) or `[nx, ny, nz]` (3).
pub fn curl_curl(dims: &[usize], omega: f64, sigma: f64) -> CscMatrix<Complex<f64>> {
    let ndim = dims.len();
    let n: usize = dims.iter().product();
    let ndof = ndim * n;
    let stride = strides(dims);

    let mut off: HashMap<(usize, usize), f64> = HashMap::new();
    let mut diag = vec![0.0f64; ndof];

    let (l_diag, l_tri) = vector_laplacian(dims, &stride, n);
    for (i, d) in l_diag.iter().enumerate() {
        diag[i] += *d;
    }
    for (r, c, w) in l_tri {
        *off.entry((r, c)).or_insert(0.0) += w;
    }
    for k in 0..n {
        let e = divergence_row(k, dims, &stride, n);
        for a in 0..e.len() {
            let (p, cp) = e[a];
            diag[p] -= cp * cp;
            for b in (a + 1)..e.len() {
                let (q, cq) = e[b];
                let (hi, lo) = if p > q { (p, q) } else { (q, p) };
                *off.entry((hi, lo)).or_insert(0.0) -= cp * cq;
            }
        }
    }

    let shift = Complex::new(-omega * omega, omega * sigma);
    let mut rows = Vec::with_capacity(ndof + off.len());
    let mut cols = Vec::with_capacity(ndof + off.len());
    let mut vals = Vec::with_capacity(ndof + off.len());
    for i in 0..ndof {
        rows.push(i);
        cols.push(i);
        vals.push(Complex::new(diag[i], 0.0) + shift);
    }
    for ((r, c), w) in off {
        if w != 0.0 {
            rows.push(r);
            cols.push(c);
            vals.push(Complex::new(w, 0.0));
        }
    }
    super::build_sym(ndof, &rows, &cols, &vals)
}

/// **Saddle-point** Stokes/KKT operator `[A B^T; B -betaC]` on a structured grid
/// (symmetric indefinite), generic over the scalar type. `A` is the velocity
/// vector Laplacian (SPD), `B` the discrete divergence coupling velocity to
/// pressure, and `-betaC` a Brezzi-Pitkaranta pressure-Laplacian stabilization
/// (`beta>0`) that regularizes the otherwise-singular zero `(2,2)` block on a
/// collocated grid. DOF layout: velocity `a*n+i` (`a<ndim`), then pressure
/// `ndim*n + i`. The zero/negative pressure block makes the system indefinite --
/// the KKT/constrained-optimization class distinct from an SPD or a shifted
/// Helmholtz operator.
pub fn saddle_point<T: Scalar>(dims: &[usize], beta: f64) -> CscMatrix<T> {
    let ndim = dims.len();
    let n: usize = dims.iter().product();
    let stride = strides(dims);
    let velocity_dofs = ndim * n;
    let ndof = velocity_dofs + n;
    let pdof = |i: usize| velocity_dofs + i;

    let mut diag = vec![0.0f64; ndof];
    let mut tri: Vec<(usize, usize, f64)> = Vec::new();

    let (l_diag, l_tri) = vector_laplacian(dims, &stride, n);
    for (i, d) in l_diag.iter().enumerate() {
        diag[i] += *d;
    }
    tri.extend(l_tri);

    for k in 0..n {
        for (v, coeff) in divergence_row(k, dims, &stride, n) {
            tri.push((pdof(k), v, coeff));
        }
    }

    let (c_diag, c_tri) = vector_laplacian(dims, &stride, n);
    for i in 0..n {
        diag[pdof(i)] -= beta * c_diag[i];
    }
    for (r, c, w) in c_tri {
        if r < n && c < n {
            tri.push((pdof(r), pdof(c), -beta * w));
        }
    }

    let mut rows = Vec::with_capacity(ndof + tri.len());
    let mut cols = Vec::with_capacity(ndof + tri.len());
    let mut vals = Vec::with_capacity(ndof + tri.len());
    for i in 0..ndof {
        rows.push(i);
        cols.push(i);
        vals.push(T::from_real(diag[i]));
    }
    for (r, c, w) in tri {
        rows.push(r);
        cols.push(c);
        vals.push(T::from_real(w));
    }
    super::build_sym(ndof, &rows, &cols, &vals)
}

/// Advection field for [`convection_diffusion`].
#[derive(Clone, Copy, Debug)]
pub enum Flow {
    /// Constant unit velocity along the grid diagonal (equal on every axis) --- a
    /// uniform wind, unsymmetric but spatially constant.
    Diagonal,
    /// Recirculating vortex in the first two axes, `b = (-(y-1/2), (x-1/2))`, zero on
    /// any third axis. The classic double-glazing / recirculating-flow benchmark:
    /// the velocity reverses across the domain, so up-/down-wind coupling flips
    /// sign, a richer unsymmetric structure than a constant wind.
    Rotating,
}

/// Convection--diffusion operator `-epsgrad^2u + b*gradu = f` on a structured `dims` grid
/// with interior unknowns and homogeneous Dirichlet boundaries, finite-differenced
/// --- the canonical source of **unsymmetric** sparse matrices (the LU path's
/// workload). The first-derivative advection term `b*gradu` breaks symmetry; the
/// diffusion term is the usual 5/7-point Laplacian scaled by `eps`.
///
/// * `upwind=false` (central differences) is the sharp, oscillation-prone form
///   that stresses pivoting at small `eps` (high grid-Peclet `|b|h/eps`);
/// * `upwind=true` is the first-order upwind form, a diagonally dominant M-matrix.
///
/// Sweeping `eps` moves the operator from diffusion-dominated (nearly symmetric,
/// small Peclet) to advection-dominated (strongly unsymmetric), and [`Flow`]
/// varies the advection field --- together they span the unsymmetric distribution
/// on a structured grid without an external FEM library.
pub fn convection_diffusion<T: Scalar>(
    dims: &[usize],
    eps: f64,
    flow: Flow,
    upwind: bool,
) -> GeneralCsc<T> {
    let d = dims.len();
    let s = strides(dims);
    let n: usize = dims.iter().product();
    let h: Vec<f64> = dims.iter().map(|&m| 1.0 / (m as f64 + 1.0)).collect();
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for p in 0..n {
        let ci: Vec<usize> = (0..d).map(|k| (p / s[k]) % dims[k]).collect();
        let x: Vec<f64> = (0..d).map(|k| (ci[k] as f64 + 1.0) * h[k]).collect();
        let b: Vec<f64> = match flow {
            Flow::Diagonal => vec![1.0 / (d as f64).sqrt(); d],
            Flow::Rotating => (0..d)
                .map(|k| match k {
                    0 => -(x[1] - 0.5),
                    1 => x[0] - 0.5,
                    _ => 0.0,
                })
                .collect(),
        };
        let mut diag = 0.0;
        for k in 0..d {
            let diff = eps / (h[k] * h[k]);
            diag += 2.0 * diff;
            let has_lo = ci[k] > 0;
            let has_hi = ci[k] + 1 < dims[k];
            let lo = p.wrapping_sub(s[k]);
            let hi = p + s[k];
            if has_lo {
                rows.push(p);
                cols.push(lo);
                vals.push(T::from_real(-diff));
            }
            if has_hi {
                rows.push(p);
                cols.push(hi);
                vals.push(T::from_real(-diff));
            }
            if upwind {
                let a = b[k] / h[k];
                if b[k] >= 0.0 {
                    diag += a;
                    if has_lo {
                        rows.push(p);
                        cols.push(lo);
                        vals.push(T::from_real(-a));
                    }
                } else {
                    diag -= a;
                    if has_hi {
                        rows.push(p);
                        cols.push(hi);
                        vals.push(T::from_real(a));
                    }
                }
            } else {
                let a = b[k] / (2.0 * h[k]);
                if has_hi {
                    rows.push(p);
                    cols.push(hi);
                    vals.push(T::from_real(a));
                }
                if has_lo {
                    rows.push(p);
                    cols.push(lo);
                    vals.push(T::from_real(-a));
                }
            }
        }
        rows.push(p);
        cols.push(p);
        vals.push(T::from_real(diag));
    }
    super::build_gen(n, &rows, &cols, &vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matgen::integration::backward_error;
    use crate::test_constants::{fixed_refinement, gamma, solve_backward_error, U64, UC64};
    use crate::{LdltSymbolic, SolverSettings, ZeroPivotAction};

    fn cube(k: usize) -> [usize; 3] {
        [k, k, k]
    }

    #[test]
    fn curl_curl_is_complex_symmetric_indefinite_and_factors() {
        let a = curl_curl(&cube(8), 2.0, 0.3);
        let max_im = a.values.iter().map(|v| v.im.abs()).fold(0.0, f64::max);
        assert!(max_im > 0.0, "curl-curl carries complex values");
        let sym = LdltSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
        let opts = SolverSettings::default()
            .with_zero_pivot(ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 });
        let solver = sym.factor(&a, &opts).unwrap();
        let n = a.n;
        let b: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i % 5) as f64 - 2.0, 0.5))
            .collect();
        let x = solver
            .solve_refined(&a, &b, &crate::RefinePolicy::steps(40))
            .unwrap()
            .0;
        let omega = backward_error(&a, &x, &b);
        let bound = fixed_refinement(n, UC64) + gamma(n + 1, UC64);
        assert!(omega <= bound, "curl-curl refined backward error {omega}");
    }

    #[test]
    fn saddle_point_is_symmetric_indefinite_and_factors() {
        let a = saddle_point::<f64>(&[10usize, 10], 0.1);
        let sym = LdltSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
        let opts = SolverSettings::default()
            .with_zero_pivot(ZeroPivotAction::PerturbToEps { abs_floor: 1e-12 });
        let solver = sym.factor(&a, &opts).unwrap();
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let x = solver
            .solve_refined(&a, &b, &crate::RefinePolicy::steps(40))
            .unwrap()
            .0;
        let omega = backward_error(&a, &x, &b);
        let bound = fixed_refinement(n, U64) + gamma(n + 1, U64);
        assert!(
            omega <= bound,
            "saddle-point refined backward error {omega}"
        );
    }

    #[test]
    fn convection_diffusion_is_unsymmetric_and_factors() {
        use crate::LuSymbolic;
        let a = convection_diffusion::<f64>(&[20usize, 20], 0.01, Flow::Rotating, true);
        let get = |i: usize, j: usize| -> f64 {
            let (s, e) = (a.col_ptr[j], a.col_ptr[j + 1]);
            a.row_idx[s..e]
                .iter()
                .position(|&r| r == i)
                .map(|k| a.values[s + k])
                .unwrap_or(0.0)
        };
        let mut asymmetric = false;
        for j in 0..a.n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[k];
                if i != j && a.values[k] != get(j, i) {
                    asymmetric = true;
                }
            }
        }
        assert!(asymmetric, "convection-diffusion is unsymmetric");
        let sym = LuSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
        let solver = sym.factor(&a, &SolverSettings::default()).unwrap();
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let x = solver.solve(&b).unwrap();
        let omega = backward_error(&a, &x, &b);
        assert!(
            omega <= solve_backward_error(n, U64),
            "convection-diffusion LU backward error {omega}"
        );
    }
}
