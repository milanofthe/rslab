//! Block Arnoldi: an orthonormal basis of the block Krylov space
//! `K_m(T, R) = span{R, T R, ..., T^(m-1) R}` of a caller's operator `T` and start
//! block `R` (`n x p`), with the block Hessenberg matrix of `T` on it,
//! `T V_m = V_(m+1) Hbar_m`. For eigenvalue and model-order-reduction methods on
//! top of a solver (shift-and-invert spectra, moment matching about a frequency):
//! the operator is a closure, as for the Krylov solvers, and is typically itself a
//! preconditioned solve. The Ritz values are the eigenvalues of `H_m`
//! ([`eigen`](crate::eigen)).
//!
//! Orthogonalization: classical Gram-Schmidt of the new block against the whole
//! basis in the chunked parallel sweeps of block GMRES (fixed chunk order, so the
//! basis does not depend on the thread count), a second pass for a column that
//! kept less than [`KrylovSettings::reorth_eta`] of its norm (DGKS), then a thin QR
//! of the block by modified Gram-Schmidt, twice. A column that the projections
//! reduce to its rounding (`n` units of roundoff of its norm before them) is
//! dependent: the space is invariant and the iteration ends there.

use super::util::{block_project, block_subtract, dotc, norm2};
use super::KrylovSettings;
use crate::scalar::Scalar;

/// The state of a block Arnoldi iteration ([`BlockArnoldi::new`], [`BlockArnoldi::step`]).
#[derive(Debug, Clone)]
pub struct BlockArnoldi<T> {
    n: usize,
    p: usize,
    chunk: usize,
    reorth_eta: f64,
    /// The orthonormal basis, `n x (steps + 1) p` column-major (`n x steps p`
    /// once invariant).
    basis: Vec<T>,
    /// Per column of `Hbar`, its entries from row 0 down to its subdiagonal block.
    hcols: Vec<Vec<T>>,
    /// `R = V_1 S`: the start block's triangular factor, `p x p` row-major.
    start: Vec<T>,
    steps: usize,
    invariant: bool,
    proj: Vec<T>,
    scratch: Vec<T>,
}

impl<T: Scalar> BlockArnoldi<T> {
    /// Start from the block `r` (`n x p`, column-major): its thin QR is the first
    /// basis block. A start block whose columns are dependent gives an invariant,
    /// empty space.
    pub fn new(r: &[T], n: usize, p: usize, settings: &KrylovSettings) -> Self {
        assert_eq!(r.len(), n * p, "the start block is n x p");
        let mut basis = r.to_vec();
        let norms: Vec<f64> = (0..p).map(|c| norm2(&r[c * n..(c + 1) * n])).collect();
        let (start, dependent) = thin_qr(&mut basis, n, p, &norms);
        let mut a = BlockArnoldi {
            n,
            p,
            chunk: settings.ortho_chunk.max(1),
            reorth_eta: settings.reorth_eta,
            basis,
            hcols: Vec::new(),
            start,
            steps: 0,
            invariant: dependent,
            proj: Vec::new(),
            scratch: Vec::new(),
        };
        if dependent {
            a.basis.clear();
        }
        a
    }

    /// One block step: `W = T V_last` (`op(V_last, W)`, both `n x p`
    /// column-major), orthogonalized against the basis and appended. Returns
    /// whether the space grew; `false` once it is invariant (then `H` is final).
    pub fn step(&mut self, op: &mut dyn FnMut(&[T], &mut [T])) -> bool {
        if self.invariant {
            return false;
        }
        let (n, p) = (self.n, self.p);
        let cols = (self.steps + 1) * p;
        let mut w = vec![T::zero(); n * p];
        op(&self.basis[(cols - p) * n..cols * n], &mut w);
        let nchunks = n.div_ceil(self.chunk);
        self.proj.resize(cols, T::zero());
        self.scratch.resize(nchunks * cols, T::zero());
        let mut h: Vec<Vec<T>> = vec![vec![T::zero(); cols]; p];
        let norms: Vec<f64> = (0..p).map(|c| norm2(&w[c * n..(c + 1) * n])).collect();
        for (c, hc) in h.iter_mut().enumerate() {
            let wc = &mut w[c * n..(c + 1) * n];
            let before = norms[c];
            for pass in 0..2 {
                // classical Gram-Schmidt against every basis column: `sa = 1`, so
                // block `i` of the sweep is basis column `i`
                block_project(
                    &self.basis,
                    wc,
                    cols,
                    1,
                    n,
                    self.chunk,
                    &mut self.proj,
                    &mut self.scratch,
                );
                block_subtract(&self.basis, wc, cols, 1, n, self.chunk, &self.proj);
                for (e, &v) in hc.iter_mut().zip(&self.proj) {
                    *e = *e + v;
                }
                // DGKS: a second pass only where the first cancelled most of it
                if pass == 0 && norm2(wc) >= self.reorth_eta * before {
                    break;
                }
            }
        }
        let (sub, dependent) = thin_qr(&mut w, n, p, &norms);
        for (c, mut col) in h.into_iter().enumerate() {
            col.extend((0..p).map(|a| sub[a * p + c]));
            self.hcols.push(col);
        }
        self.steps += 1;
        if dependent {
            self.invariant = true;
            return false;
        }
        self.basis.extend_from_slice(&w);
        true
    }

    /// The block steps taken.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Whether the space is invariant under `T` (no step grows it).
    pub fn is_invariant(&self) -> bool {
        self.invariant
    }

    /// The square block Hessenberg `H_m = V_m^H T V_m`, `m = steps p`, row-major.
    pub fn hessenberg(&self) -> Vec<T> {
        let m = self.steps * self.p;
        (0..m * m)
            .map(|q| {
                self.hcols[q % m]
                    .get(q / m)
                    .copied()
                    .unwrap_or_else(T::zero)
            })
            .collect()
    }

    /// The last subdiagonal block `H_(m+1,m)` (`p x p` row-major): the Ritz pair
    /// `(theta, V_m y)` of `H_m y = theta y` has the residual
    /// `|H_(m+1,m) y_last|` (`y_last` the last `p` entries of `y`).
    pub fn next_block(&self) -> Vec<T> {
        let (m, p) = (self.steps * self.p, self.p);
        (0..p * p)
            .map(|q| {
                self.hcols[m - p + q % p]
                    .get(m + q / p)
                    .copied()
                    .unwrap_or_else(T::zero)
            })
            .collect()
    }

    /// The start block's triangular factor `S` (`R = V_1 S`), `p x p` row-major.
    pub fn start(&self) -> &[T] {
        &self.start
    }

    /// The basis `V_m` (`n x steps p` column-major; one block more until
    /// invariant).
    pub fn basis(&self) -> &[T] {
        &self.basis
    }

    /// The bytes the basis of `steps` steps of `n x p` blocks takes: for the
    /// caller's memory admission.
    pub fn basis_bytes(n: usize, p: usize, steps: usize) -> usize {
        n.saturating_mul(p)
            .saturating_mul(steps + 1)
            .saturating_mul(std::mem::size_of::<T>())
    }
}

/// Thin QR of the `n x q` block `z` in place (modified Gram-Schmidt, twice):
/// `z = Q S`, `S` upper triangular `q x q` row-major. Also whether a column is
/// dependent on the basis and the columns before it: reduced to `n` units of
/// roundoff of `norms` (its norm before any orthogonalization, the rounding of
/// its Gram-Schmidt); its column is left zero then.
fn thin_qr<T: Scalar>(z: &mut [T], n: usize, q: usize, norms: &[f64]) -> (Vec<T>, bool) {
    let mut s = vec![T::zero(); q * q];
    let mut dependent = false;
    for b in 0..q {
        let before = norms[b];
        for _ in 0..2 {
            for a in 0..b {
                let (head, tail) = z.split_at_mut(b * n);
                let qa = &head[a * n..(a + 1) * n];
                let zb = &mut tail[..n];
                let cf = dotc(qa, zb);
                for (w, &v) in zb.iter_mut().zip(qa) {
                    *w = *w - cf * v;
                }
                s[a * q + b] = s[a * q + b] + cf;
            }
        }
        let zb = &mut z[b * n..(b + 1) * n];
        let nrm = norm2(zb);
        // (`<=` and a NaN norm both count as dependent)
        if nrm.is_nan() || nrm <= f64::EPSILON * n as f64 * before {
            dependent = true;
            zb.iter_mut().for_each(|v| *v = T::zero());
            continue;
        }
        let inv = T::from_real(1.0 / nrm);
        zb.iter_mut().for_each(|v| *v = *v * inv);
        s[b * q + b] = T::from_real(nrm);
    }
    (s, dependent)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// A dense `n x n` operator applied to a column-major `n x p` block.
    fn apply(a: &[C], n: usize, x: &[C], y: &mut [C]) {
        let p = x.len() / n;
        for c in 0..p {
            for i in 0..n {
                y[c * n + i] = (0..n).map(|j| a[i * n + j] * x[c * n + j]).sum();
            }
        }
    }

    /// The Arnoldi relation `T V_m = V_m H_m + V_(m+1) H_(m+1,m) E_m^T` holds and
    /// the basis is orthonormal, for one and for several start columns, and the
    /// same on one thread and on four (the sweeps' chunk order is fixed).
    #[test]
    fn the_arnoldi_relation_holds() {
        let n = 300;
        let a = random(n * n, 7);
        let settings = KrylovSettings {
            ortho_chunk: 64,
            ..KrylovSettings::default()
        };
        for p in [1usize, 3] {
            let run = || {
                let mut ar = BlockArnoldi::new(&random(n * p, 9), n, p, &settings);
                for _ in 0..12 {
                    assert!(ar.step(&mut |x, y| apply(&a, n, x, y)));
                }
                ar
            };
            let ar = run();
            let four = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap()
                .install(run);
            assert_eq!(ar.basis(), four.basis(), "p {p}: thread count");
            let m = ar.steps() * p;
            let v = ar.basis();
            for i in 0..=m + p - 1 {
                for j in 0..=i {
                    let g = dotc(&v[i * n..(i + 1) * n], &v[j * n..(j + 1) * n]);
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((g - want).norm() < 1e-12, "p {p}: V^H V at {i},{j}: {g}");
                }
            }
            let h = ar.hessenberg();
            let sub = ar.next_block();
            let mut tv = vec![C::new(0.0, 0.0); n * m];
            apply(&a, n, &v[..n * m], &mut tv);
            for c in 0..m {
                for i in 0..n {
                    let mut r: C = (0..m).map(|k| v[k * n + i] * h[k * m + c]).sum();
                    if c >= m - p {
                        r += (0..p)
                            .map(|k| v[(m + k) * n + i] * sub[k * p + (c - (m - p))])
                            .sum::<C>();
                    }
                    assert!(
                        (tv[c * n + i] - r).norm() < 1e-10,
                        "p {p}: relation at {i},{c}"
                    );
                }
            }
        }
    }

    /// On an operator with an invariant subspace containing the start vector (a
    /// diagonal matrix with 4 distinct values, the start mixing them) the
    /// iteration ends invariant after 4 steps with the 4 values as Ritz values.
    #[test]
    fn an_invariant_space_ends_the_iteration() {
        let n = 50;
        let vals = [1.0, 2.0, 3.0, 4.0];
        let a: Vec<C> = (0..n * n)
            .map(|q| {
                if q / n == q % n {
                    C::new(vals[(q / n) % 4], 0.0)
                } else {
                    C::new(0.0, 0.0)
                }
            })
            .collect();
        let mut ar = BlockArnoldi::new(&random(n, 3), n, 1, &KrylovSettings::default());
        let mut grew = 0;
        while ar.step(&mut |x, y| apply(&a, n, x, y)) {
            grew += 1;
            assert!(grew < 10, "never invariant");
        }
        assert!(ar.is_invariant());
        assert_eq!(ar.steps(), 4);
        let m = ar.steps();
        let mut got = crate::eigen(&ar.hessenberg(), m).expect("converges").values;
        got.sort_by(|x, y| x.re.total_cmp(&y.re));
        for (g, w) in got.iter().zip(vals) {
            assert!((g - C::new(w, 0.0)).norm() < 1e-10, "{g} against {w}");
        }
    }
}
