//! Block Low-Rank (BLR) infrastructure for the multifrontal fronts.
//!
//! Frontal matrices from elliptic-PDE FEM (Helmholtz) and integral-equation MoM
//! near-fields are not low-rank themselves, but their **off-diagonal sub-blocks**
//! — which couple geometrically separated index clusters through a smooth kernel
//! — are numerically low-rank. Compressing those blocks shrinks both the
//! factorization flop count and the front/contribution memory, the two levers
//! behind the PARDISO throughput gap and the transient-memory spike. This module
//! provides the low-rank block type and a pure-Rust rank-revealing compressor;
//! the BLR-aware front factorization builds on it.
//!
//! Compression uses **fully-pivoted Adaptive Cross Approximation (ACA)** — the
//! integral-equation-community standard: a truncated, rank-revealing
//! Gaussian-elimination cross approximation that needs no SVD (which has no
//! pure-Rust complex implementation here). For a dense `m×n` block it is
//! `O(rank·m·n)`, robust, and stops adaptively at a relative Frobenius
//! tolerance.

use crate::scalar::Scalar;

/// A low-rank factorization `B ≈ U · Vᵀ` of a dense `m × n` block, with `U`
/// (`m × rank`) and `V` (`n × rank`) stored column-major. `Vᵀ` (not `Vᴴ`): no
/// conjugation, matching the unconjugated (complex-symmetric / general) algebra
/// the fronts use.
#[derive(Debug, Clone)]
pub struct LowRank<T> {
    pub m: usize,
    pub n: usize,
    pub rank: usize,
    /// `m × rank`, column-major.
    pub u: Vec<T>,
    /// `n × rank`, column-major.
    pub v: Vec<T>,
}

impl<T: Scalar> LowRank<T> {
    /// Stored entries `rank·(m + n)` vs the dense `m·n` — the compression only
    /// pays off when this is smaller.
    pub fn storage(&self) -> usize {
        self.rank * (self.m + self.n)
    }

    /// Reconstruct the dense approximation `U·Vᵀ` (column-major `m × n`). For
    /// tests / diagnostics; the factorization never densifies a compressed block.
    pub fn to_dense(&self) -> Vec<T> {
        let mut d = vec![T::zero(); self.m * self.n];
        for k in 0..self.rank {
            let uk = &self.u[k * self.m..k * self.m + self.m];
            let vk = &self.v[k * self.n..k * self.n + self.n];
            for j in 0..self.n {
                let vj = vk[j];
                if vj != T::zero() {
                    let col = &mut d[j * self.m..j * self.m + self.m];
                    for i in 0..self.m {
                        col[i] = col[i] + uk[i] * vj;
                    }
                }
            }
        }
        d
    }
}

/// Frobenius norm of a dense column-major buffer.
fn frob<T: Scalar>(a: &[T]) -> f64 {
    a.iter().map(|x| x.magnitude_sq()).sum::<f64>().sqrt()
}

/// Fully-pivoted ACA compression of a dense `m × n` block `b` (column-major) to
/// relative Frobenius tolerance `eps`. Caps the rank at `max_rank`.
///
/// Returns the low-rank factors and the achieved rank. If the block does not
/// compress below `eps` within `max_rank` cross steps, `rank == max_rank`
/// (`min(m, n)` cap) and the caller should treat it as not worth compressing.
///
/// Algorithm: repeatedly pick the largest-magnitude residual entry `(i*, j*)`,
/// peel off the rank-1 cross `R[:,j*] · R[i*,:] / R[i*,j*]` (Schur step of
/// Gaussian elimination with full pivoting), and stop when the residual
/// Frobenius norm falls to `eps · ‖B‖_F`.
pub fn compress_aca<T: Scalar>(
    b: &[T],
    m: usize,
    n: usize,
    eps: f64,
    max_rank: usize,
) -> LowRank<T> {
    let cap = max_rank.min(m).min(n);
    let bnorm = frob(b);
    let mut r = b.to_vec(); // residual, column-major m×n
    let mut u: Vec<T> = Vec::with_capacity(m * cap);
    let mut v: Vec<T> = Vec::with_capacity(n * cap);
    let mut rank = 0usize;

    // A structurally-zero block compresses to rank 0.
    if bnorm == 0.0 || cap == 0 {
        return LowRank {
            m,
            n,
            rank: 0,
            u,
            v,
        };
    }
    let tol = eps * bnorm;

    while rank < cap {
        // Full-pivot search: largest |R[i,j]|.
        let mut bi = 0usize;
        let mut bj = 0usize;
        let mut bmag = 0.0f64;
        for j in 0..n {
            let col = &r[j * m..j * m + m];
            for (i, &val) in col.iter().enumerate() {
                let mag = val.magnitude();
                if mag > bmag {
                    bmag = mag;
                    bi = i;
                    bj = j;
                }
            }
        }
        if bmag == 0.0 {
            break; // residual exactly zero — exact low-rank
        }
        let pinv = r[bj * m + bi].recip();
        // Cross factors: u = R[:, bj], v = R[bi, :]·(1/pivot).
        let uk: Vec<T> = (0..m).map(|i| r[bj * m + i]).collect();
        let vk: Vec<T> = (0..n).map(|j| r[j * m + bi] * pinv).collect();
        // Schur update R -= u ⊗ v.
        for j in 0..n {
            let vj = vk[j];
            if vj != T::zero() {
                let col = &mut r[j * m..j * m + m];
                for i in 0..m {
                    col[i] = col[i] - uk[i] * vj;
                }
            }
        }
        u.extend_from_slice(&uk);
        v.extend_from_slice(&vk);
        rank += 1;
        if frob(&r) <= tol {
            break;
        }
    }
    LowRank { m, n, rank, u, v }
}

/// Diagnostic: partition a dense front `f` (`n × n` column-major, of which the
/// leading `ncol` columns are eliminated) into `b × b` blocks and report how
/// compressible its strictly-lower-triangle off-diagonal blocks are at several
/// Frobenius tolerances. This is the empirical BLR-benefit estimate — mean rank
/// and compressed-vs-dense storage of the off-diagonal blocks that BLR would
/// represent in low-rank form. Prints to stderr; gated by the caller.
pub fn probe_front<T: Scalar>(f: &[T], n: usize, ncol: usize, b: usize) {
    let nb = n.div_ceil(b);
    for &eps in &[1e-2f64, 1e-4, 1e-8] {
        let mut dense = 0usize;
        let mut comp = 0usize;
        let mut sumrank = 0usize;
        let mut nblk = 0usize;
        for jb in 0..nb {
            let j0 = jb * b;
            let jn = (j0 + b).min(n);
            for ib in (jb + 1)..nb {
                let i0 = ib * b;
                let im = (i0 + b).min(n);
                let (bm, bn) = (im - i0, jn - j0);
                let mut blk = vec![T::zero(); bm * bn];
                for jj in 0..bn {
                    for ii in 0..bm {
                        blk[jj * bm + ii] = f[(j0 + jj) * n + (i0 + ii)];
                    }
                }
                let lr = compress_aca(&blk, bm, bn, eps, bm.min(bn));
                dense += bm * bn;
                comp += lr.storage();
                sumrank += lr.rank;
                nblk += 1;
            }
        }
        if nblk > 0 {
            eprintln!(
                "[BLR_PROBE] n={n} ncol={ncol} b={b} eps={eps:.0e}: off-diag-blocks={nblk} \
                 mean-rank={:.1}/{b} compressed={:.0}% of dense",
                sumrank as f64 / nblk as f64,
                100.0 * comp as f64 / dense as f64,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn max_abs_diff<T: Scalar>(a: &[T], b: &[T]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).magnitude())
            .fold(0.0, f64::max)
    }

    #[test]
    fn aca_recovers_exact_low_rank() {
        // Build a rank-3 block U0·V0ᵀ (m=40, n=30) and check ACA recovers it at a
        // tiny tolerance with rank ≤ 3.
        let (m, n, r0) = (40usize, 30usize, 3usize);
        let u0: Vec<f64> = (0..m * r0).map(|t| ((t * 7 % 11) as f64 - 5.0)).collect();
        let v0: Vec<f64> = (0..n * r0).map(|t| ((t * 5 % 13) as f64 - 6.0)).collect();
        let mut b = vec![0.0f64; m * n];
        for k in 0..r0 {
            for j in 0..n {
                for i in 0..m {
                    b[j * m + i] += u0[k * m + i] * v0[k * n + j];
                }
            }
        }
        let lr = compress_aca(&b, m, n, 1e-12, m.min(n));
        assert!(lr.rank <= r0, "rank {} should be ≤ {r0}", lr.rank);
        assert!(max_abs_diff(&b, &lr.to_dense()) < 1e-9 * frob(&b));
        assert!(lr.storage() < m * n, "must actually compress");
    }

    #[test]
    fn aca_compresses_smooth_kernel_block() {
        // A smooth-kernel off-diagonal block (1/(2 + |i−j|) type, separated
        // clusters) — the EM near-field analogue — must compress to low rank at a
        // loose preconditioner tolerance.
        let (m, n) = (64usize, 64usize);
        let mut b = vec![Complex::new(0.0, 0.0); m * n];
        for j in 0..n {
            for i in 0..m {
                // rows and cols sit in well-separated 1-D clusters.
                let xi = i as f64;
                let xj = 200.0 + j as f64;
                let d = xj - xi;
                b[j * m + i] = Complex::new(1.0 / d, 0.5 / (d * d));
            }
        }
        let lr = compress_aca(&b, m, n, 1e-6, m.min(n));
        assert!(
            lr.rank <= 8,
            "smooth separated block should be very low rank, got {}",
            lr.rank
        );
        let err = max_abs_diff(&b, &lr.to_dense());
        assert!(err <= 1e-5 * frob(&b), "reconstruction error {err}");
    }

    #[test]
    fn aca_full_rank_block_does_not_falsely_compress() {
        // A diagonal-dominant random-ish block has no low-rank structure: ACA at
        // tight tolerance should need (near) full rank.
        let (m, n) = (24usize, 24usize);
        let mut b = vec![0.0f64; m * n];
        for j in 0..n {
            for i in 0..m {
                b[j * m + i] = if i == j {
                    100.0
                } else {
                    (((i * 31 + j * 17) % 7) as f64) - 3.0
                };
            }
        }
        let lr = compress_aca(&b, m, n, 1e-12, m.min(n));
        assert!(lr.rank >= m - 2, "full-rank block, got rank {}", lr.rank);
    }
}
