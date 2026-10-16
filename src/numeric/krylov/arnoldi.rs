//! Block Arnoldi: an orthonormal basis of the block Krylov space
//! `K_m(T, R) = span{R, T R, ..., T^(m-1) R}` of a caller's operator `T` and start
//! block `R` (`n x p`), with the block Hessenberg matrix of `T` on it,
//! `T V_m = V_(m+1) Hbar_m`. For eigenvalue and model-order-reduction methods on
//! top of a solver (shift-and-invert spectra, moment matching about a frequency):
//! the operator is a [`LinearOperator`], typically itself a preconditioned solve.
//! The Ritz values are the eigenvalues of `H_m` ([`crate::dense::Schur`]).
//!
//! Orthogonalization, per new column: classical Gram-Schmidt against the basis
//! (the whole block at once, one panel product that reads the basis once) and
//! against the block's columns accepted before it, a second pass (CGS2) where
//! the first kept less than [`KrylovSettings::reorth_eta`] of the column's norm
//! (DGKS). A column reduced to the rounding of its Gram-Schmidt (at most `k` units
//! of roundoff of its norm before, `k` the columns it was projected against, the
//! type's own unit roundoff) is dependent: it is replaced by the first unit vector
//! `e_i` that keeps at least `reorth_eta` of its norm against the basis, with a
//! zero subdiagonal entry, so the other columns of the block go on and the
//! Arnoldi relation holds; [`BlockArnoldi::ranks`] reports how many columns of
//! each block were independent. A block without any is the end: the space is
//! invariant.
//!
//! The basis, `H` and the scratch are allocated once, for `max_steps` steps, and a
//! step allocates nothing. The panel sweeps run in row chunks of
//! [`KrylovSettings::ortho_chunk`] on the caller's current rayon pool (or the pool
//! lent to rslab, [`crate::parallel`]), summed in a fixed order, so the basis does
//! not depend on the thread count.

use super::util::{norm2, panel_project, panel_subtract};
use super::{KrylovSettings, LinearOperator};
use crate::error::RslabError;
use crate::scalar::Scalar;

/// The state of a block Arnoldi iteration ([`BlockArnoldi::new`], [`BlockArnoldi::step`]).
#[derive(Debug, Clone)]
pub struct BlockArnoldi<T> {
    n: usize,
    p: usize,
    max_steps: usize,
    chunk: usize,
    reorth_eta: f64,
    /// The orthonormal basis, `n x (steps + 1) p` column-major (`n x steps p`
    /// once invariant); capacity for `max_steps + 1` blocks.
    basis: Vec<T>,
    /// The columns of `Hbar`, packed: column `j` of step `s = j / p` holds its
    /// `(s + 2) p` entries from row 0 down.
    h: Vec<T>,
    /// `R = V_1 S`: the start block's triangular factor, `p x p` row-major.
    start: Vec<T>,
    /// Independent columns of the start block and of every step's block.
    start_rank: usize,
    ranks: Vec<usize>,
    steps: usize,
    invariant: bool,
    /// Scratch, sized at `new`: the block's coefficients on the basis
    /// (`cols x p`), a column's second-pass coefficients, its coefficients on the
    /// block's columns before it, the block's triangular factor, the columns'
    /// norms, and the panel sums.
    proj: Vec<T>,
    again: Vec<T>,
    own: Vec<T>,
    sub: Vec<T>,
    norms: Vec<f64>,
    scratch: Vec<T>,
}

impl<T: Scalar> BlockArnoldi<T> {
    /// Start from the block `r` (`n x p`, column-major), with room for
    /// `max_steps` steps: the thin QR of `r` is the first basis block (a
    /// dependent column replaced as in a step). `DimensionMismatch` where `r` is
    /// not `n x p`, `InvalidInput` without columns.
    pub fn new(
        r: &[T],
        n: usize,
        p: usize,
        max_steps: usize,
        settings: &KrylovSettings,
    ) -> Result<Self, RslabError> {
        if p == 0 || n == 0 {
            return Err(RslabError::InvalidInput(
                "block Arnoldi needs a start block of at least one column".into(),
            ));
        }
        if r.len() != n * p {
            return Err(RslabError::DimensionMismatch {
                expected: n * p,
                got: r.len(),
            });
        }
        let chunk = settings.ortho_chunk.max(1);
        let cols_max = (max_steps + 1) * p;
        let mut basis = Vec::with_capacity(n * cols_max);
        basis.extend_from_slice(r);
        let mut a = BlockArnoldi {
            n,
            p,
            max_steps,
            chunk,
            reorth_eta: settings.reorth_eta,
            basis,
            h: Vec::with_capacity(Self::h_len(p, max_steps)),
            start: vec![T::zero(); p * p],
            start_rank: 0,
            ranks: Vec::with_capacity(max_steps),
            steps: 0,
            invariant: false,
            proj: vec![T::zero(); cols_max * p],
            again: vec![T::zero(); cols_max],
            own: vec![T::zero(); p],
            sub: vec![T::zero(); p * p],
            norms: vec![0.0; p],
            scratch: vec![T::zero(); n.div_ceil(chunk) * cols_max * p],
        };
        let rank = a.orthonormalize_block(0);
        a.start.copy_from_slice(&a.sub);
        a.start_rank = rank;
        if rank == 0 {
            a.invariant = true;
            a.basis.clear();
        }
        Ok(a)
    }

    /// `H`'s packed length for `max_steps` steps: `p^2 m (m + 3) / 2`.
    fn h_len(p: usize, max_steps: usize) -> usize {
        p * p * max_steps * (max_steps + 3) / 2 // lit: the sum of s + 2 over the steps
    }

    /// The bytes the iteration takes for `max_steps` steps of `n x p` blocks
    /// with the settings' chunk (basis, `H`, scratch): for the caller's memory
    /// admission.
    pub fn bytes(n: usize, p: usize, max_steps: usize, settings: &KrylovSettings) -> usize {
        let cols_max = (max_steps + 1) * p;
        let chunk = settings.ortho_chunk.max(1);
        let scalars = n * cols_max
            + Self::h_len(p, max_steps)
            + cols_max * p
            + cols_max
            + p
            + 2 * p * p
            + n.div_ceil(chunk) * cols_max * p;
        scalars * std::mem::size_of::<T>() + p * std::mem::size_of::<f64>()
    }

    /// The bytes it holds (from the capacities).
    pub fn held_bytes(&self) -> usize {
        (self.basis.capacity()
            + self.h.capacity()
            + self.proj.capacity()
            + self.again.capacity()
            + self.own.capacity()
            + self.sub.capacity()
            + self.start.capacity()
            + self.scratch.capacity())
            * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<f64>()
    }

    /// Orthonormalize the block of `p` columns after the first `cols` basis
    /// columns in place (module doc), without allocating: its triangular factor
    /// goes to `self.sub` (`p x p` row-major, the coefficients on the block's own
    /// columns, zero diagonal for a replaced column), its coefficients on the basis
    /// before it to `self.proj` (`cols x p`, row `j` column `c`). Returns its rank.
    fn orthonormalize_block(&mut self, cols: usize) -> usize {
        let (n, p, chunk) = (self.n, self.p, self.chunk);
        self.sub.iter_mut().for_each(|v| *v = T::zero());
        for c in 0..p {
            self.norms[c] = norm2(&self.basis[(cols + c) * n..(cols + c + 1) * n]);
        }
        // pass 1 against the basis, the whole block at once
        {
            let (v, w) = self.basis.split_at_mut(cols * n);
            panel_project(v, cols, w, p, n, chunk, &mut self.proj, &mut self.scratch);
            panel_subtract(v, cols, w, p, n, chunk, &self.proj);
        }
        let mut rank = 0;
        for c in 0..p {
            let k = cols + c;
            let (vq, rest) = self.basis.split_at_mut(k * n);
            let wc = &mut rest[..n];
            // pass 1 against the block's columns before it
            let own = &mut self.own[..c];
            panel_project(&vq[cols * n..], c, wc, 1, n, chunk, own, &mut self.scratch);
            panel_subtract(&vq[cols * n..], c, wc, 1, n, chunk, own);
            if norm2(wc) < self.reorth_eta * self.norms[c] {
                // pass 2 against all k columns
                let again = &mut self.again[..k];
                panel_project(vq, k, wc, 1, n, chunk, again, &mut self.scratch);
                panel_subtract(vq, k, wc, 1, n, chunk, again);
                for (j, &g) in again[..cols].iter().enumerate() {
                    self.proj[j * p + c] = self.proj[j * p + c] + g;
                }
                for (a, o) in own.iter_mut().enumerate() {
                    *o = *o + again[cols + a];
                }
            }
            for (a, &o) in own.iter().enumerate() {
                self.sub[a * p + c] = o;
            }
            let nrm = norm2(wc);
            // dependent: within the rounding of its k projections
            if nrm.is_nan() || nrm <= T::epsilon() * k.max(1) as f64 * self.norms[c] {
                let again = &mut self.again[..k];
                replace(
                    vq,
                    wc,
                    k,
                    n,
                    chunk,
                    self.reorth_eta,
                    again,
                    &mut self.scratch,
                );
                continue;
            }
            let inv = T::from_real(1.0 / nrm);
            wc.iter_mut().for_each(|v| *v = *v * inv);
            self.sub[c * p + c] = T::from_real(nrm);
            rank += 1;
        }
        rank
    }

    /// One block step: `W = T V_last` written after the basis, orthogonalized
    /// and appended (module doc). Returns how many of its `p` columns were
    /// independent; 0 ends the iteration (the space is invariant, `H` final),
    /// and a step on an invariant space returns 0 again without work.
    /// `DimensionMismatch` where `op` is not `n x n`, `InvalidInput` once
    /// `max_steps` steps were taken.
    pub fn step<A: LinearOperator<T> + ?Sized>(&mut self, op: &A) -> Result<usize, RslabError> {
        if self.invariant {
            return Ok(0);
        }
        if op.n() != self.n {
            return Err(RslabError::DimensionMismatch {
                expected: self.n,
                got: op.n(),
            });
        }
        if self.steps == self.max_steps {
            return Err(RslabError::InvalidInput(format!(
                "block Arnoldi: the basis holds {} steps",
                self.max_steps
            )));
        }
        let (n, p) = (self.n, self.p);
        let cols = (self.steps + 1) * p;
        // within the capacity reserved at new: no reallocation
        self.basis.resize(n * (cols + p), T::zero());
        {
            let (v, w) = self.basis.split_at_mut(cols * n);
            op.apply_block(&v[(cols - p) * n..], w, p);
        }
        let rank = self.orthonormalize_block(cols);
        for c in 0..p {
            for j in 0..cols {
                self.h.push(self.proj[j * p + c]);
            }
            for a in 0..p {
                self.h.push(self.sub[a * p + c]);
            }
        }
        self.steps += 1;
        self.ranks.push(rank);
        if rank == 0 {
            self.invariant = true;
            self.basis.truncate(n * cols);
        }
        Ok(rank)
    }

    /// The block steps taken.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Whether the space is invariant under `T` (a block without an
    /// independent column; no step grows it).
    pub fn is_invariant(&self) -> bool {
        self.invariant
    }

    /// The independent columns of the start block.
    pub fn start_rank(&self) -> usize {
        self.start_rank
    }

    /// The independent columns of every step's block.
    pub fn ranks(&self) -> &[usize] {
        &self.ranks
    }

    /// The square block Hessenberg `H_m = V_m^H T V_m`, `m = steps p`, row-major.
    pub fn hessenberg(&self) -> Vec<T> {
        let (m, p) = (self.steps * self.p, self.p);
        let mut out = vec![T::zero(); m * m];
        let mut off = 0;
        for j in 0..m {
            let len = (j / p + 2) * p;
            for i in 0..len.min(m) {
                out[i * m + j] = self.h[off + i];
            }
            off += len;
        }
        out
    }

    /// The last subdiagonal block `H_(m+1,m)` (`p x p` row-major), `None` before
    /// the first step: the Ritz pair `(theta, V_m y)` of `H_m y = theta y` has the
    /// residual `|H_(m+1,m) y_last|` (`y_last` the last `p` entries of `y`).
    pub fn next_block(&self) -> Option<Vec<T>> {
        if self.steps == 0 {
            return None;
        }
        let (m, p) = (self.steps * self.p, self.p);
        let len = (self.steps + 1) * p;
        let first = self.h.len() - p * len;
        Some(
            (0..p * p)
                .map(|q| {
                    let (a, c) = (q / p, q % p);
                    self.h[first + c * len + (m + a)]
                })
                .collect(),
        )
    }

    /// The start block's triangular factor `S` (`R = V_1 S`), `p x p` row-major.
    pub fn start(&self) -> &[T] {
        &self.start
    }

    /// The basis `V_(m+1)` (`n x (steps + 1) p` column-major; `n x steps p` once
    /// invariant).
    pub fn basis(&self) -> &[T] {
        &self.basis
    }
}

/// Replace the dependent column `wc` by the first unit vector `e_i` that keeps at
/// least `eta` of its norm against the `k` columns `vq` (two Gram-Schmidt passes),
/// normalized, `coef` (`k`) and `scratch` the sweeps' buffers. A basis of all
/// `n` dimensions leaves `wc` zero.
#[allow(clippy::too_many_arguments)]
fn replace<T: Scalar>(
    vq: &[T],
    wc: &mut [T],
    k: usize,
    n: usize,
    chunk: usize,
    eta: f64,
    coef: &mut [T],
    scratch: &mut [T],
) {
    for i in 0..n {
        wc.iter_mut().for_each(|v| *v = T::zero());
        wc[i] = T::one();
        for _ in 0..2 {
            panel_project(vq, k, wc, 1, n, chunk, coef, scratch);
            panel_subtract(vq, k, wc, 1, n, chunk, coef);
        }
        let nrm = norm2(wc);
        if nrm >= eta {
            let inv = T::from_real(1.0 / nrm);
            wc.iter_mut().for_each(|v| *v = *v * inv);
            return;
        }
    }
    wc.iter_mut().for_each(|v| *v = T::zero());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FnOperator;
    use num_complex::Complex;

    type C = Complex<f64>;

    fn random(len: usize, seed: u64) -> Vec<C> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..len).map(|_| C::new(next(), next())).collect()
    }

    /// A dense `n x n` operator (row-major) applied to a column-major block.
    fn dense(a: &[C], n: usize) -> impl FnMut(&[C], &mut [C], usize) + '_ {
        move |x: &[C], y: &mut [C], p: usize| {
            for c in 0..p {
                for i in 0..n {
                    y[c * n + i] = (0..n).map(|j| a[i * n + j] * x[c * n + j]).sum();
                }
            }
        }
    }

    fn dot(a: &[C], b: &[C]) -> C {
        a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
    }

    /// The basis is orthonormal to `tol`.
    fn orthonormal(v: &[C], n: usize, tol: f64) {
        let k = v.len() / n;
        for i in 0..k {
            for j in 0..=i {
                let g = dot(&v[i * n..(i + 1) * n], &v[j * n..(j + 1) * n]);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((g - want).norm() < tol, "V^H V at {i},{j}: {g}");
            }
        }
    }

    /// `T V_m = V_m H_m + V_(m+1) H_(m+1,m) E_m^T` to `tol`.
    fn relation(ar: &BlockArnoldi<C>, a: &[C], n: usize, p: usize, tol: f64) {
        let m = ar.steps() * p;
        let v = ar.basis();
        let h = ar.hessenberg();
        let sub = ar.next_block().unwrap();
        let mut tv = vec![C::new(0.0, 0.0); n * m];
        dense(a, n)(&v[..n * m], &mut tv, m);
        let extra = v.len() / n > m;
        for c in 0..m {
            for i in 0..n {
                let mut r: C = (0..m).map(|k| v[k * n + i] * h[k * m + c]).sum();
                if c >= m - p && extra {
                    r += (0..p)
                        .map(|k| v[(m + k) * n + i] * sub[k * p + (c - (m - p))])
                        .sum::<C>();
                }
                assert!((tv[c * n + i] - r).norm() < tol, "relation at {i},{c}");
            }
        }
    }

    /// The Arnoldi relation holds, the basis is orthonormal, `R = V_1 S`, for one
    /// and for several start columns, and the basis is the same on one thread and
    /// on four (the sweeps' chunk order is fixed).
    #[test]
    fn the_arnoldi_relation_holds() {
        let n = 300;
        let a = random(n * n, 7);
        let settings = KrylovSettings {
            ortho_chunk: 64,
            ..KrylovSettings::default()
        };
        for p in [1usize, 3] {
            let r = random(n * p, 9);
            let run = || {
                let mut ar = BlockArnoldi::new(&r, n, p, 12, &settings).unwrap();
                let op = FnOperator::new(n, dense(&a, n));
                for _ in 0..12 {
                    assert_eq!(ar.step(&op).unwrap(), p);
                }
                assert!(ar.step(&op).is_err(), "past max_steps");
                ar
            };
            let ar = run();
            let four = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap()
                .install(run);
            assert_eq!(ar.basis(), four.basis(), "p {p}: thread count");
            orthonormal(ar.basis(), n, 1e-12);
            relation(&ar, &a, n, p, 1e-10);
            // R = V_1 S
            let (v, s) = (ar.basis(), ar.start());
            for c in 0..p {
                for i in 0..n {
                    let vs: C = (0..p).map(|k| v[k * n + i] * s[k * p + c]).sum();
                    assert!((vs - r[c * n + i]).norm() < 1e-12);
                }
            }
            assert_eq!(
                ar.held_bytes(),
                BlockArnoldi::<C>::bytes(n, p, 12, &settings)
            );
        }
    }

    /// An invariant space containing the start vector (a diagonal operator with
    /// 4 distinct values) ends the iteration after 4 steps, its Ritz values the 4
    /// values; `next_block` is `None` before a step.
    #[test]
    fn an_invariant_space_ends_the_iteration() {
        let n = 50;
        let vals = [1.0, 2.0, 3.0, 4.0];
        let a: Vec<C> = (0..n * n)
            .map(|q| {
                C::new(
                    if q / n == q % n {
                        vals[(q / n) % 4]
                    } else {
                        0.0
                    },
                    0.0,
                )
            })
            .collect();
        let mut ar =
            BlockArnoldi::new(&random(n, 3), n, 1, 10, &KrylovSettings::default()).unwrap();
        assert!(ar.next_block().is_none());
        let op = FnOperator::new(n, dense(&a, n));
        let mut ranks = Vec::new();
        while !ar.is_invariant() {
            ranks.push(ar.step(&op).unwrap());
        }
        assert_eq!(ranks, vec![1, 1, 1, 0]);
        let m = ar.steps();
        let s = crate::dense::Schur::new(&ar.hessenberg(), m).unwrap();
        let mut got = s.eigenvalues();
        got.sort_by(|x, y| x.re.total_cmp(&y.re));
        for (g, w) in got.iter().zip(vals) {
            assert!((g - C::new(w, 0.0)).norm() < 1e-10, "{g} against {w}");
        }
        assert_eq!(ar.step(&op).unwrap(), 0, "a step on an invariant space");
    }

    /// One column of a block dependent (an eigenvector of the operator in the
    /// start block: its image lies in the basis), the others not: that column is
    /// replaced, the step's rank is `p - 1`, the iteration goes on with an
    /// orthonormal basis and the relation holds. A start block with two equal
    /// columns has rank 1 and an orthonormal first block.
    #[test]
    fn a_dependent_column_is_deflated_alone() {
        let n = 60;
        let mut a = random(n * n, 5);
        // e_0 an eigenvector: column 0 of A is 2 e_0
        for i in 0..n {
            a[i * n] = C::new(if i == 0 { 2.0 } else { 0.0 }, 0.0);
        }
        let mut r = random(n * 2, 6);
        for i in 0..n {
            r[i] = C::new(if i == 0 { 1.0 } else { 0.0 }, 0.0);
        }
        let mut ar = BlockArnoldi::new(&r, n, 2, 6, &KrylovSettings::default()).unwrap();
        let op = FnOperator::new(n, dense(&a, n));
        assert_eq!(ar.step(&op).unwrap(), 1, "T e_0 = 2 e_0 lies in the basis");
        for _ in 0..4 {
            ar.step(&op).unwrap();
        }
        orthonormal(ar.basis(), n, 1e-12);
        relation(&ar, &a, n, 2, 1e-10);
        // equal columns
        let col = random(n, 8);
        let twice: Vec<C> = col.iter().chain(col.iter()).copied().collect();
        let ar = BlockArnoldi::new(&twice, n, 2, 2, &KrylovSettings::default()).unwrap();
        assert_eq!(ar.start_rank(), 1);
        orthonormal(ar.basis(), n, 1e-12);
        assert!(
            ar.start()[3].norm() == 0.0,
            "the replaced column's diagonal"
        );
    }

    /// A nearly dependent block (a column in the basis to 1e-9): the second pass
    /// keeps it orthogonal to the basis to the working precision.
    #[test]
    fn a_near_dependent_column_stays_orthogonal() {
        let n = 80;
        let a = random(n * n, 11);
        let r = random(n, 12);
        let mut ar = BlockArnoldi::new(&r, n, 1, 4, &KrylovSettings::default()).unwrap();
        let mut first = true;
        // the operator returns its input's own direction plus 1e-9 of a random one on
        // the first step, then A
        let noise = random(n, 13);
        let op = FnOperator::new(n, |x: &[C], y: &mut [C], _p: usize| {
            if std::mem::take(&mut first) {
                for i in 0..n {
                    y[i] = x[i] + noise[i] * 1e-9;
                }
            } else {
                dense(&a, n)(x, y, 1);
            }
        });
        for _ in 0..4 {
            ar.step(&op).unwrap();
        }
        orthonormal(ar.basis(), n, 1e-13);
    }

    /// In single precision the relation holds to its own roundoff.
    #[test]
    fn single_precision() {
        let n = 100;
        let a: Vec<Complex<f32>> = random(n * n, 21)
            .iter()
            .map(|z| Complex::new(z.re as f32, z.im as f32))
            .collect();
        let r: Vec<Complex<f32>> = random(n, 22)
            .iter()
            .map(|z| Complex::new(z.re as f32, z.im as f32))
            .collect();
        let mut ar = BlockArnoldi::new(&r, n, 1, 8, &KrylovSettings::default()).unwrap();
        let op = FnOperator::new(
            n,
            |x: &[Complex<f32>], y: &mut [Complex<f32>], _p: usize| {
                for i in 0..n {
                    y[i] = (0..n).map(|j| a[i * n + j] * x[j]).sum();
                }
            },
        );
        for _ in 0..8 {
            assert_eq!(ar.step(&op).unwrap(), 1);
        }
        let v = ar.basis();
        for i in 0..9 {
            for j in 0..=i {
                let g: Complex<f32> = v[i * n..(i + 1) * n]
                    .iter()
                    .zip(&v[j * n..(j + 1) * n])
                    .map(|(x, y)| x.conj() * y)
                    .sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((g - want).norm() < 1e-5, "V^H V at {i},{j}: {g}");
            }
        }
    }

    /// Wrong sizes are errors, not panics.
    #[test]
    fn wrong_sizes_are_errors() {
        let s = KrylovSettings::default();
        assert!(matches!(
            BlockArnoldi::new(&[C::new(1.0, 0.0); 5], 3, 2, 4, &s),
            Err(RslabError::DimensionMismatch {
                expected: 6,
                got: 5
            })
        ));
        assert!(BlockArnoldi::<C>::new(&[], 3, 0, 4, &s).is_err());
        let mut ar = BlockArnoldi::new(&random(4, 1), 4, 1, 2, &s).unwrap();
        let op = FnOperator::new(5, |x: &[C], y: &mut [C], _p: usize| y.copy_from_slice(x));
        assert!(matches!(
            ar.step(&op),
            Err(RslabError::DimensionMismatch {
                expected: 4,
                got: 5
            })
        ));
    }
}
