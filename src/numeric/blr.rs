//! Block Low-Rank (BLR) infrastructure for the multifrontal fronts.
//!
//! Frontal matrices from elliptic-PDE FEM (Helmholtz) and integral-equation MoM
//! near-fields are not low-rank themselves, but their **off-diagonal sub-blocks**
//! (which couple geometrically separated index clusters through a smooth kernel)
//! are numerically low-rank. Compressing those blocks shrinks both the
//! factorization flop count and the front/contribution memory, the two levers
//! behind the PARDISO throughput gap and the transient-memory spike. This module
//! provides the low-rank block type and a pure-Rust rank-revealing compressor;
//! the BLR-aware front factorization builds on it.
//!
//! Compression uses **fully-pivoted Adaptive Cross Approximation (ACA)** - the
//! integral-equation-community standard: a truncated, rank-revealing
//! Gaussian-elimination cross approximation that needs no SVD (which has no
//! pure-Rust complex implementation here). For a dense `m×n` block it is
//! `O(rank·m·n)`, robust, and stops adaptively at a relative Frobenius
//! tolerance.

use crate::error::RslabError;
use crate::numeric::multifrontal_ldlt::perturb_pivot;
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
    /// Stored entries `rank·(m + n)` vs the dense `m·n` - the compression only
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
            break; // residual exactly zero - exact low-rank
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

/// One tile of a block-partitioned matrix: stored either dense (column-major,
/// `rows × cols`) or in compressed low-rank form `U·Vᵀ`. Diagonal tiles and
/// off-diagonal tiles that do not compress below the break-even rank stay
/// `Dense`; admissible (geometrically separated, smooth-kernel) off-diagonal
/// tiles become `LowRank`. The BLR factorization dispatches on this enum: dense
/// tiles take the ordinary kernels, low-rank tiles take the cheap
/// `U(VᵀU)Vᵀ`-style block arithmetic.
#[derive(Debug, Clone)]
pub enum Block<T> {
    /// Column-major `rows × cols` dense tile.
    Dense {
        rows: usize,
        cols: usize,
        data: Vec<T>,
    },
    /// Compressed tile `U·Vᵀ`.
    LowRank(LowRank<T>),
}

impl<T: Scalar> Block<T> {
    pub fn rows(&self) -> usize {
        match self {
            Block::Dense { rows, .. } => *rows,
            Block::LowRank(lr) => lr.m,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Block::Dense { cols, .. } => *cols,
            Block::LowRank(lr) => lr.n,
        }
    }

    /// Stored scalar count - `rows·cols` dense, `rank·(rows+cols)` low-rank.
    pub fn storage(&self) -> usize {
        match self {
            Block::Dense { rows, cols, .. } => rows * cols,
            Block::LowRank(lr) => lr.storage(),
        }
    }

    pub fn is_low_rank(&self) -> bool {
        matches!(self, Block::LowRank(_))
    }

    /// Dense `rows × cols` column-major reconstruction of the tile.
    pub fn to_dense(&self) -> Vec<T> {
        match self {
            Block::Dense { data, .. } => data.clone(),
            Block::LowRank(lr) => lr.to_dense(),
        }
    }
}

/// Extent `(start, len)` of block index `k` along an axis of length `full`
/// partitioned into tiles of size `b` (the trailing tile may be shorter).
fn block_extent(full: usize, b: usize, k: usize) -> (usize, usize) {
    let start = k * b;
    (start, b.min(full - start))
}

/// A block-partitioned (Block Low-Rank) representation of a dense column-major
/// `nrow × ncol` matrix: a `nbr × nbc` grid of [`Block`]s over a fixed block
/// size `b` (the trailing row/column tile may be smaller). The grid is stored
/// row-major (`blocks[ib·nbc + jb]`).
///
/// This is the data structure the BLR-aware front factorization operates on:
/// Stage 1 builds and reconstructs it; later stages factor it block-by-block
/// with low-rank tile arithmetic.
#[derive(Debug, Clone)]
pub struct BlrMatrix<T> {
    pub nrow: usize,
    pub ncol: usize,
    pub b: usize,
    pub nbr: usize,
    pub nbc: usize,
    blocks: Vec<Block<T>>,
}

impl<T: Scalar> BlrMatrix<T> {
    fn idx(&self, ib: usize, jb: usize) -> usize {
        ib * self.nbc + jb
    }

    pub fn block(&self, ib: usize, jb: usize) -> &Block<T> {
        &self.blocks[self.idx(ib, jb)]
    }

    pub fn block_mut(&mut self, ib: usize, jb: usize) -> &mut Block<T> {
        let i = self.idx(ib, jb);
        &mut self.blocks[i]
    }

    /// Row extent `(start, len)` of block-row `ib`.
    pub fn row_extent(&self, ib: usize) -> (usize, usize) {
        block_extent(self.nrow, self.b, ib)
    }

    /// Column extent `(start, len)` of block-column `jb`.
    pub fn col_extent(&self, jb: usize) -> (usize, usize) {
        block_extent(self.ncol, self.b, jb)
    }

    /// Total stored scalars across all tiles - the BLR memory footprint.
    pub fn storage(&self) -> usize {
        self.blocks.iter().map(Block::storage).sum()
    }

    /// Footprint of the equivalent dense matrix, `nrow·ncol`.
    pub fn dense_storage(&self) -> usize {
        self.nrow * self.ncol
    }

    /// Number of off-diagonal tiles stored compressed.
    pub fn n_low_rank(&self) -> usize {
        self.blocks.iter().filter(|b| b.is_low_rank()).count()
    }

    /// Build a BLR partition of the dense column-major `nrow × ncol` matrix `a`
    /// at block size `b` and per-tile relative Frobenius tolerance `eps`.
    ///
    /// Diagonal tiles (`ib == jb`) are always dense. Each off-diagonal tile is
    /// compressed by ACA, capped at the **break-even rank**
    /// `⌊rows·cols/(rows+cols)⌋` (the rank above which `U·Vᵀ` stores no less than
    /// the dense tile): a tile that reaches that cap without converging is kept
    /// dense, which also bounds the ACA cost on incompressible near-diagonal
    /// tiles to `O(breakeven · rows·cols)`.
    pub fn from_dense(a: &[T], nrow: usize, ncol: usize, b: usize, eps: f64) -> BlrMatrix<T> {
        assert!(b > 0, "block size must be positive");
        let nbr = nrow.div_ceil(b);
        let nbc = ncol.div_ceil(b);
        let mut blocks = Vec::with_capacity(nbr * nbc);
        for ib in 0..nbr {
            let (r0, bm) = block_extent(nrow, b, ib);
            for jb in 0..nbc {
                let (c0, bn) = block_extent(ncol, b, jb);
                // Extract the tile column-major from `a` (col stride `nrow`).
                let mut tile = vec![T::zero(); bm * bn];
                for jj in 0..bn {
                    let src = (c0 + jj) * nrow + r0;
                    tile[jj * bm..jj * bm + bm].copy_from_slice(&a[src..src + bm]);
                }
                let blk = if ib == jb {
                    Block::Dense {
                        rows: bm,
                        cols: bn,
                        data: tile,
                    }
                } else {
                    let breakeven = (bm * bn / (bm + bn)).max(1);
                    let lr = compress_aca(&tile, bm, bn, eps, breakeven);
                    if lr.rank < breakeven && lr.storage() < bm * bn {
                        Block::LowRank(lr)
                    } else {
                        Block::Dense {
                            rows: bm,
                            cols: bn,
                            data: tile,
                        }
                    }
                };
                blocks.push(blk);
            }
        }
        BlrMatrix {
            nrow,
            ncol,
            b,
            nbr,
            nbc,
            blocks,
        }
    }

    /// Reconstruct the dense column-major `nrow × ncol` approximation by writing
    /// every tile into place. For tests / diagnostics and the dense-fallback
    /// path; the factorization never densifies the whole matrix.
    pub fn to_dense(&self) -> Vec<T> {
        let mut out = vec![T::zero(); self.nrow * self.ncol];
        for ib in 0..self.nbr {
            let (r0, bm) = self.row_extent(ib);
            for jb in 0..self.nbc {
                let (c0, bn) = self.col_extent(jb);
                let tile = self.block(ib, jb).to_dense();
                for jj in 0..bn {
                    let dst = (c0 + jj) * self.nrow + r0;
                    out[dst..dst + bm].copy_from_slice(&tile[jj * bm..jj * bm + bm]);
                }
            }
        }
        out
    }
}

// ===========================================================================
// Stage 2 - BLR-aware LU factorization
//
// Block right-looking LU over the tile grid with **block-restricted partial
// pivoting**: pivoting is confined to within each diagonal tile (rows never
// cross a tile boundary), so the low-rank structure of the off-diagonal tiles
// is preserved - a row interchange of a low-rank tile `U·Vᵀ` is just a row
// interchange of `U`. This is weaker than global partial pivoting, but the MoM
// preconditioner runs on a two-sided-equilibrated matrix with static pivot
// perturbation and GMRES refinement, where block-local pivoting is the standard
// (and sufficient) BLR choice.
//
// Off-diagonal tiles stay compressed throughout: the panel triangular solves
// act on the thin `U`/`V` factors, and the Schur updates use low-rank tile
// products `(U₁V₁ᵀ)(U₂V₂ᵀ) = U₁(V₁ᵀU₂)V₂ᵀ`. Updated low-rank tiles accumulate
// rank by factor concatenation and are **recompressed** once the rank exceeds
// the break-even point. Recompression here densifies the (small, ≤b×b) tile and
// re-runs ACA - correct and pure-Rust; a QR-based recompression that avoids the
// densify is the later performance refinement.
// ===========================================================================

/// `C = A·B`, all column-major; `A` is `m×k`, `B` is `k×n`, `C` is `m×n`.
fn dense_matmul<T: Scalar>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];
    for j in 0..n {
        let ccol = &mut c[j * m..j * m + m];
        for p in 0..k {
            let bpj = b[j * k + p];
            if bpj != T::zero() {
                let acol = &a[p * m..p * m + m];
                for i in 0..m {
                    ccol[i] = ccol[i] + acol[i] * bpj;
                }
            }
        }
    }
    c
}

/// `C = Aᵀ·B`; `A` is `p×m`, `B` is `p×n` (both column-major), `C` is `m×n`.
fn matmul_at_b<T: Scalar>(a: &[T], p: usize, m: usize, b: &[T], n: usize) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];
    for j in 0..n {
        let bcol = &b[j * p..j * p + p];
        for i in 0..m {
            let acol = &a[i * p..i * p + p];
            let mut s = T::zero();
            for t in 0..p {
                s = s + acol[t] * bcol[t];
            }
            c[j * m + i] = s;
        }
    }
    c
}

/// `C = A·Bᵀ`; `A` is `m×k`, `B` is `n×k` (both column-major), `C` is `m×n`.
fn matmul_a_bt<T: Scalar>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];
    for t in 0..k {
        let acol = &a[t * m..t * m + m];
        let bcol = &b[t * n..t * n + n];
        for j in 0..n {
            let bjt = bcol[j];
            if bjt != T::zero() {
                let ccol = &mut c[j * m..j * m + m];
                for i in 0..m {
                    ccol[i] = ccol[i] + acol[i] * bjt;
                }
            }
        }
    }
    c
}

/// `dst -= src`, elementwise.
fn dense_sub_inplace<T: Scalar>(dst: &mut [T], src: &[T]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = *d - *s;
    }
}

/// Replay an ordered list of row interchanges on a column-major `rows × cols`
/// buffer (the same swaps the diagonal-tile LU performed).
fn apply_swaps_rows<T: Scalar>(data: &mut [T], rows: usize, cols: usize, swaps: &[(usize, usize)]) {
    for &(a, b) in swaps {
        if a != b {
            for c in 0..cols {
                data.swap(c * rows + a, c * rows + b);
            }
        }
    }
}

/// Apply the diagonal tile's row interchanges to another tile in the same
/// block-row - dense tiles swap rows, low-rank tiles swap rows of `U`.
fn apply_swaps_block<T: Scalar>(blk: &mut Block<T>, swaps: &[(usize, usize)]) {
    match blk {
        Block::Dense { rows, cols, data } => apply_swaps_rows(data, *rows, *cols, swaps),
        Block::LowRank(lr) => apply_swaps_rows(&mut lr.u, lr.m, lr.rank, swaps),
    }
}

/// In-place dense LU with partial pivoting over a column-major `n × n` tile.
/// Returns the ordered row interchanges and the static-perturbation count. The
/// packed result holds unit-`L` below the diagonal and `U` on/above it.
fn dense_lu_inplace<T: Scalar>(
    a: &mut [T],
    n: usize,
    perturb_floor: Option<f64>,
) -> Result<(Vec<(usize, usize)>, usize), RslabError> {
    let mut swaps = Vec::new();
    let mut n_perturbed = 0usize;
    for k in 0..n {
        let mut p = k;
        let mut best = a[k * n + k].magnitude_sq();
        for i in (k + 1)..n {
            let m = a[k * n + i].magnitude_sq();
            if m > best {
                best = m;
                p = i;
            }
        }
        if p != k {
            for c in 0..n {
                a.swap(c * n + k, c * n + p);
            }
            swaps.push((k, p));
        }
        let mut piv = a[k * n + k];
        match perturb_floor {
            Some(floor) if piv.magnitude() < floor => {
                piv = perturb_pivot(piv, floor);
                n_perturbed += 1;
            }
            None if piv == T::zero() => return Err(RslabError::NumericallyRankDeficient),
            _ => {}
        }
        a[k * n + k] = piv;
        let pinv = piv.recip();
        for i in (k + 1)..n {
            a[k * n + i] = a[k * n + i] * pinv;
        }
        for j in (k + 1)..n {
            let ukj = a[j * n + k];
            if ukj != T::zero() {
                for i in (k + 1)..n {
                    a[j * n + i] = a[j * n + i] - a[k * n + i] * ukj;
                }
            }
        }
    }
    Ok((swaps, n_perturbed))
}

/// Solve `L·X = B` in place; `L` is the packed unit-lower factor (`n × n`, only
/// the strictly-lower part read), `X`/`B` are column-major `n × nrhs`.
fn unit_lower_solve_left<T: Scalar>(l: &[T], n: usize, x: &mut [T], nrhs: usize) {
    for jj in 0..nrhs {
        let col = &mut x[jj * n..jj * n + n];
        for k in 0..n {
            let xk = col[k];
            if xk != T::zero() {
                for i in (k + 1)..n {
                    col[i] = col[i] - l[k * n + i] * xk;
                }
            }
        }
    }
}

/// Solve `U·X = B` in place; `U` is the packed upper factor (`n × n`, diagonal
/// carries the pivots), `X`/`B` are column-major `n × nrhs`.
fn upper_solve_left<T: Scalar>(u: &[T], n: usize, x: &mut [T], nrhs: usize) {
    for jj in 0..nrhs {
        let col = &mut x[jj * n..jj * n + n];
        for k in (0..n).rev() {
            let xk = col[k] * u[k * n + k].recip();
            col[k] = xk;
            if xk != T::zero() {
                for i in 0..k {
                    col[i] = col[i] - u[k * n + i] * xk;
                }
            }
        }
    }
}

/// Solve `X·U = B` in place; `U` is the packed upper factor (`n × n`), `X`/`B`
/// are column-major `m × n`. (Right triangular solve for an `L` panel tile.)
fn upper_solve_right<T: Scalar>(u: &[T], n: usize, x: &mut [T], m: usize) {
    for j in 0..n {
        for p in 0..j {
            let upj = u[j * n + p]; // U(p, j)
            if upj != T::zero() {
                for i in 0..m {
                    x[j * m + i] = x[j * m + i] - x[p * m + i] * upj;
                }
            }
        }
        let d = u[j * n + j].recip();
        for i in 0..m {
            x[j * m + i] = x[j * m + i] * d;
        }
    }
}

/// Solve `Uᵀ·Y = B` in place; `U` is the packed upper factor (`n × n`), `Y`/`B`
/// are column-major `n × nrhs`. (`Uᵀ` is lower-triangular → forward sweep.)
fn upper_transpose_solve_left<T: Scalar>(u: &[T], n: usize, y: &mut [T], nrhs: usize) {
    for jj in 0..nrhs {
        let col = &mut y[jj * n..jj * n + n];
        for k in 0..n {
            let mut s = col[k];
            for i in 0..k {
                s = s - u[k * n + i] * col[i]; // u[k*n+i] = U(i,k) = (Uᵀ)(k,i)
            }
            col[k] = s * u[k * n + k].recip();
        }
    }
}

/// `U_kj = L_kk⁻¹ · A_kj` - apply the unit-lower diagonal-tile solve to a panel
/// tile (whole dense tile, or just the `U` factor of a low-rank tile).
fn solve_lower_panel<T: Scalar>(diag: &[T], n: usize, blk: &mut Block<T>) {
    match blk {
        Block::Dense { cols, data, .. } => unit_lower_solve_left(diag, n, data, *cols),
        Block::LowRank(lr) => unit_lower_solve_left(diag, n, &mut lr.u, lr.rank),
    }
}

/// `L_ik = A_ik · U_kk⁻¹` - apply the upper diagonal-tile solve to a panel tile
/// from the right (dense tile), or as `U_kk⁻ᵀ` on the `V` factor (low-rank).
fn solve_upper_panel<T: Scalar>(diag: &[T], n: usize, blk: &mut Block<T>) {
    match blk {
        Block::Dense { rows, data, .. } => upper_solve_right(diag, n, data, *rows),
        Block::LowRank(lr) => upper_transpose_solve_left(diag, n, &mut lr.v, lr.rank),
    }
}

/// Tile product `A·B` as a [`Block`]. Dense·Dense is dense; any product
/// involving a low-rank tile stays low-rank by folding into the thinner factor
/// (`rank = min` of the operands), never densifying.
fn block_product<T: Scalar>(a: &Block<T>, b: &Block<T>) -> Block<T> {
    let m = a.rows();
    let p = a.cols();
    let n = b.cols();
    debug_assert_eq!(p, b.rows());
    match (a, b) {
        (Block::Dense { data: ad, .. }, Block::Dense { data: bd, .. }) => Block::Dense {
            rows: m,
            cols: n,
            data: dense_matmul(ad, m, p, bd, n),
        },
        (Block::Dense { data: ad, .. }, Block::LowRank(lr)) => {
            // (D·U2)·V2ᵀ
            let u = dense_matmul(ad, m, p, &lr.u, lr.rank);
            Block::LowRank(LowRank {
                m,
                n,
                rank: lr.rank,
                u,
                v: lr.v.clone(),
            })
        }
        (Block::LowRank(lr), Block::Dense { data: bd, .. }) => {
            // U1·(Dᵀ·V1)ᵀ
            let v = matmul_at_b(bd, p, n, &lr.v, lr.rank);
            Block::LowRank(LowRank {
                m,
                n,
                rank: lr.rank,
                u: lr.u.clone(),
                v,
            })
        }
        (Block::LowRank(a1), Block::LowRank(b1)) => {
            // U1·(V1ᵀU2)·V2ᵀ - fold the inner r1×r2 matrix into the thinner side.
            let (r1, r2) = (a1.rank, b1.rank);
            let mm = matmul_at_b(&a1.v, p, r1, &b1.u, r2); // r1×r2
            if r1 == 0 || r2 == 0 {
                return Block::LowRank(LowRank {
                    m,
                    n,
                    rank: 0,
                    u: Vec::new(),
                    v: Vec::new(),
                });
            }
            if r1 <= r2 {
                // U' = U1 (m×r1); V' = V2·Mᵀ (n×r1)
                let v = matmul_a_bt(&b1.v, n, r2, &mm, r1);
                Block::LowRank(LowRank {
                    m,
                    n,
                    rank: r1,
                    u: a1.u.clone(),
                    v,
                })
            } else {
                // U' = U1·M (m×r2); V' = V2 (n×r2)
                let u = dense_matmul(&a1.u, m, r1, &mm, r2);
                Block::LowRank(LowRank {
                    m,
                    n,
                    rank: r2,
                    u,
                    v: b1.v.clone(),
                })
            }
        }
    }
}

/// `target -= prod`. Low-rank ⊖ low-rank concatenates factors (rank adds) and
/// recompresses past the break-even rank; mixed cases densify the smaller side.
fn block_sub<T: Scalar>(target: &mut Block<T>, prod: Block<T>, eps: f64) {
    let m = target.rows();
    let n = target.cols();
    let placeholder = Block::Dense {
        rows: 0,
        cols: 0,
        data: Vec::new(),
    };
    let cur = std::mem::replace(target, placeholder);
    *target = match (cur, prod) {
        (Block::Dense { mut data, .. }, Block::Dense { data: pd, .. }) => {
            dense_sub_inplace(&mut data, &pd);
            Block::Dense {
                rows: m,
                cols: n,
                data,
            }
        }
        (Block::Dense { mut data, .. }, Block::LowRank(lr)) => {
            dense_sub_inplace(&mut data, &lr.to_dense());
            Block::Dense {
                rows: m,
                cols: n,
                data,
            }
        }
        (Block::LowRank(lr), Block::Dense { data: pd, .. }) => {
            let mut data = lr.to_dense();
            dense_sub_inplace(&mut data, &pd);
            Block::Dense {
                rows: m,
                cols: n,
                data,
            }
        }
        (Block::LowRank(t), Block::LowRank(p)) => concat_recompress(t, p, eps),
    };
}

/// `T ⊖ P` for two low-rank tiles: concatenate `[U_t | −U_p]`, `[V_t | V_p]`
/// (rank `r_t + r_p`), then recompress if the rank exceeds the break-even point:
/// densify the small tile and re-run ACA, falling back to dense if it no
/// longer pays.
fn concat_recompress<T: Scalar>(t: LowRank<T>, mut p: LowRank<T>, eps: f64) -> Block<T> {
    let (m, n) = (t.m, t.n);
    // Negate the product's contribution (subtraction) via its U factor.
    for x in p.u.iter_mut() {
        *x = T::zero() - *x;
    }
    let rank = t.rank + p.rank;
    let mut u = t.u;
    u.extend_from_slice(&p.u);
    let mut v = t.v;
    v.extend_from_slice(&p.v);
    let breakeven = (m * n / (m + n)).max(1);
    let merged = LowRank {
        m,
        n,
        rank,
        u,
        v,
    };
    if rank <= breakeven {
        return Block::LowRank(merged);
    }
    // Recompress: densify (≤ b×b) and re-ACA at the break-even cap.
    let dense = merged.to_dense();
    let lr = compress_aca(&dense, m, n, eps, breakeven);
    if lr.rank < breakeven && lr.storage() < m * n {
        Block::LowRank(lr)
    } else {
        Block::Dense {
            rows: m,
            cols: n,
            data: dense,
        }
    }
}

/// BLR-factored square matrix `P·A = L·U`: the in-place packed factors plus the
/// per-diagonal-tile local pivot interchanges. Solve with [`blr_lu_solve`].
pub struct BlrLu<T> {
    /// Packed factors: diagonal tiles hold dense `LU`, strictly-lower tiles `L`,
    /// strictly-upper tiles `U` (dense or low-rank).
    pub a: BlrMatrix<T>,
    /// Per-block-row local row interchanges from the diagonal-tile pivoting.
    pub block_swaps: Vec<Vec<(usize, usize)>>,
    pub n_perturbed: usize,
    pub eps: f64,
}

impl<T: Scalar> BlrLu<T> {
    /// Stored scalars across all factor tiles.
    pub fn storage(&self) -> usize {
        self.a.storage()
    }
}

/// BLR-aware LU factorization of a **square** BLR matrix with block-restricted
/// partial pivoting and static pivot perturbation. `eps` is the per-tile
/// recompression tolerance; `perturb_floor` lifts tiny pivots (preconditioner
/// mode) or, when `None`, a zero pivot errors.
pub fn blr_lu_factor<T: Scalar>(
    mut a: BlrMatrix<T>,
    eps: f64,
    perturb_floor: Option<f64>,
) -> Result<BlrLu<T>, RslabError> {
    assert_eq!(a.nrow, a.ncol, "BLR LU requires a square matrix");
    let nb = a.nbr;
    debug_assert_eq!(nb, a.nbc);
    let mut block_swaps: Vec<Vec<(usize, usize)>> = Vec::with_capacity(nb);
    let mut n_perturbed = 0usize;

    for k in 0..nb {
        let (_, nk) = a.row_extent(k);
        // Factor the diagonal tile (always dense).
        let (swaps, np) = match a.block_mut(k, k) {
            Block::Dense { data, .. } => dense_lu_inplace(data, nk, perturb_floor)?,
            Block::LowRank(_) => unreachable!("diagonal tiles are dense"),
        };
        n_perturbed += np;
        // Apply the local row interchanges across the rest of block-row k
        // (already-computed L tiles to the left and the not-yet-factored ones).
        for j in 0..nb {
            if j != k {
                apply_swaps_block(a.block_mut(k, j), &swaps);
            }
        }
        block_swaps.push(swaps);

        // Owned copy of the packed diagonal LU for the panel solves.
        let diag = a.block(k, k).to_dense();
        // Panel: U_kj = L_kk⁻¹ A_kj  (j > k).
        for j in (k + 1)..nb {
            solve_lower_panel(&diag, nk, a.block_mut(k, j));
        }
        // Panel: L_ik = A_ik U_kk⁻¹  (i > k).
        for i in (k + 1)..nb {
            solve_upper_panel(&diag, nk, a.block_mut(i, k));
        }
        // Schur update: A_ij −= L_ik · U_kj.
        for j in (k + 1)..nb {
            for i in (k + 1)..nb {
                let prod = block_product(a.block(i, k), a.block(k, j));
                block_sub(a.block_mut(i, j), prod, eps);
            }
        }
    }
    Ok(BlrLu {
        a,
        block_swaps,
        n_perturbed,
        eps,
    })
}

/// `out -= blk · xj` for a tile (dense or low-rank). `xj` has `blk.cols()`
/// entries, `out` has `blk.rows()`.
fn block_matvec_sub<T: Scalar>(blk: &Block<T>, xj: &[T], out: &mut [T]) {
    match blk {
        Block::Dense { rows, cols, data } => {
            for j in 0..*cols {
                let xv = xj[j];
                if xv != T::zero() {
                    let col = &data[j * rows..j * rows + rows];
                    for i in 0..*rows {
                        out[i] = out[i] - col[i] * xv;
                    }
                }
            }
        }
        Block::LowRank(lr) => {
            // t = Vᵀ·xj  (rank), then out -= U·t.
            let mut t = vec![T::zero(); lr.rank];
            for (r, slot) in t.iter_mut().enumerate() {
                let vcol = &lr.v[r * lr.n..r * lr.n + lr.n];
                let mut s = T::zero();
                for j in 0..lr.n {
                    s = s + vcol[j] * xj[j];
                }
                *slot = s;
            }
            for (r, &tr) in t.iter().enumerate() {
                if tr != T::zero() {
                    let ucol = &lr.u[r * lr.m..r * lr.m + lr.m];
                    for i in 0..lr.m {
                        out[i] = out[i] - ucol[i] * tr;
                    }
                }
            }
        }
    }
}

/// Apply an ordered list of interchanges to a length-`n` RHS segment.
fn apply_swaps_vec<T: Scalar>(x: &mut [T], swaps: &[(usize, usize)]) {
    for &(a, b) in swaps {
        x.swap(a, b);
    }
}

/// Solve `A·x = rhs` in place (`x` starts as the RHS) using BLR factors:
/// permute, block forward-substitute `L·y = P·rhs`, block back-substitute
/// `U·x = y`.
pub fn blr_lu_solve<T: Scalar>(lu: &BlrLu<T>, x: &mut [T]) {
    let a = &lu.a;
    let nb = a.nbr;
    debug_assert_eq!(x.len(), a.nrow);

    // P·rhs - apply each diagonal tile's local interchanges to its row segment.
    for (k, swaps) in lu.block_swaps.iter().enumerate() {
        let (r0, nk) = a.row_extent(k);
        apply_swaps_vec(&mut x[r0..r0 + nk], swaps);
    }
    // Forward: L·y = P·rhs.
    for i in 0..nb {
        let (r0, ni) = a.row_extent(i);
        for j in 0..i {
            let (c0, nj) = a.col_extent(j);
            let xj = x[c0..c0 + nj].to_vec();
            block_matvec_sub(a.block(i, j), &xj, &mut x[r0..r0 + ni]);
        }
        let diag = a.block(i, i).to_dense();
        unit_lower_solve_left(&diag, ni, &mut x[r0..r0 + ni], 1);
    }
    // Backward: U·x = y.
    for i in (0..nb).rev() {
        let (r0, ni) = a.row_extent(i);
        for j in (i + 1)..nb {
            let (c0, nj) = a.col_extent(j);
            let xj = x[c0..c0 + nj].to_vec();
            block_matvec_sub(a.block(i, j), &xj, &mut x[r0..r0 + ni]);
        }
        let diag = a.block(i, i).to_dense();
        upper_solve_left(&diag, ni, &mut x[r0..r0 + ni], 1);
    }
}

/// Diagnostic: partition a dense front `f` (`n × n` column-major, of which the
/// leading `ncol` columns are eliminated) into `b × b` blocks and report how
/// compressible its strictly-lower-triangle off-diagonal blocks are at several
/// Frobenius tolerances. This is the empirical BLR-benefit estimate - mean rank
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
        let u0: Vec<f64> = (0..m * r0).map(|t| (t * 7 % 11) as f64 - 5.0).collect();
        let v0: Vec<f64> = (0..n * r0).map(|t| (t * 5 % 13) as f64 - 6.0).collect();
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
        // clusters) - the EM near-field analogue - must compress to low rank at a
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

    /// A front-like dense matrix: strong diagonal, off-diagonal coupling through
    /// a smooth separated-cluster kernel - compressible away from the diagonal.
    fn smooth_front(n: usize) -> Vec<Complex<f64>> {
        let mut a = vec![Complex::new(0.0, 0.0); n * n];
        for j in 0..n {
            for i in 0..n {
                a[j * n + i] = if i == j {
                    Complex::new(n as f64, 1.0)
                } else {
                    // Smooth kernel in the index separation; rows/cols of a tile
                    // far from the diagonal sit in well-separated clusters.
                    let d = (i as f64 - j as f64).abs() + 1.0;
                    Complex::new(1.0 / d, 0.5 / (d * d))
                };
            }
        }
        a
    }

    #[test]
    fn blr_roundtrip_within_tolerance() {
        // BLR partition of a smooth front must reconstruct within the per-tile
        // tolerance and actually store fewer scalars than dense.
        let (n, b, eps) = (256usize, 64usize, 1e-4);
        let a = smooth_front(n);
        let blr = BlrMatrix::from_dense(&a, n, n, b, eps);
        let recon = blr.to_dense();
        let err = max_abs_diff(&a, &recon);
        // Loose bound: per-tile relative eps accumulates mildly across the grid.
        assert!(
            err <= 1e-2 * frob(&a),
            "reconstruction error {err} vs ‖A‖={}",
            frob(&a)
        );
        assert!(
            blr.storage() < blr.dense_storage(),
            "BLR storage {} should beat dense {}",
            blr.storage(),
            blr.dense_storage()
        );
        assert!(blr.n_low_rank() > 0, "some off-diag tiles should compress");
    }

    #[test]
    fn blr_diagonal_tiles_stay_dense() {
        let (n, b) = (128usize, 32usize);
        let a = smooth_front(n);
        let blr = BlrMatrix::from_dense(&a, n, n, b, 1e-6);
        for k in 0..blr.nbr.min(blr.nbc) {
            assert!(
                !blr.block(k, k).is_low_rank(),
                "diagonal tile ({k},{k}) must stay dense"
            );
        }
    }

    #[test]
    fn blr_ragged_partition_reconstructs() {
        // Block size that does not divide the dimension → trailing short tiles.
        let (m, n, b) = (100usize, 70usize, 32usize);
        let mut a = vec![0.0f64; m * n];
        for j in 0..n {
            for i in 0..m {
                a[j * m + i] = ((i * 13 + j * 7) % 17) as f64;
            }
        }
        let blr = BlrMatrix::from_dense(&a, m, n, b, 1e-12);
        assert_eq!(blr.nbr, 4);
        assert_eq!(blr.nbc, 3);
        let recon = blr.to_dense();
        assert!(max_abs_diff(&a, &recon) < 1e-9 * frob(&a).max(1.0));
    }

    /// Strongly diagonally dominant complex matrix (block-local pivoting is
    /// stable, no equilibration needed) - off-diagonal blocks are full-rank
    /// random and stay dense at tight tolerance.
    fn diag_dominant_complex(n: usize) -> Vec<Complex<f64>> {
        let mut a = vec![Complex::new(0.0, 0.0); n * n];
        for j in 0..n {
            for i in 0..n {
                a[j * n + i] = if i == j {
                    Complex::new(2.0 * n as f64, 1.0)
                } else {
                    Complex::new(
                        (((i * 7 + j * 3) % 11) as f64 - 5.0) / n as f64,
                        (((i * 5 + j * 2) % 7) as f64 - 3.0) / n as f64,
                    )
                };
            }
        }
        a
    }

    fn matvec<T: Scalar>(a: &[T], n: usize, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); n];
        for j in 0..n {
            let xj = x[j];
            for i in 0..n {
                y[i] = y[i] + a[j * n + i] * xj;
            }
        }
        y
    }

    fn rel_resid<T: Scalar>(a: &[T], n: usize, x: &[T], b: &[T]) -> f64 {
        let ax = matvec(a, n, x);
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            num += (ax[i] - b[i]).magnitude_sq();
            den += b[i].magnitude_sq();
        }
        (num / den).sqrt()
    }

    fn rhs_complex(n: usize) -> Vec<Complex<f64>> {
        (0..n)
            .map(|i| Complex::new(((i * 3 % 7) as f64) - 3.0, ((i * 5 % 11) as f64) - 5.0))
            .collect()
    }

    #[test]
    fn blr_lu_dense_single_block_is_exact() {
        // b ≥ n → one dense tile → ordinary dense LU; residual at machine level.
        let n = 40usize;
        let a = diag_dominant_complex(n);
        let b = rhs_complex(n);
        let blr = BlrMatrix::from_dense(&a, n, n, n, 1e-12);
        assert_eq!(blr.nbr, 1);
        let lu = blr_lu_factor(blr, 1e-12, None).unwrap();
        let mut x = b.clone();
        blr_lu_solve(&lu, &mut x);
        assert!(rel_resid(&a, n, &x, &b) < 1e-10);
    }

    #[test]
    fn blr_lu_multiblock_dense_solves() {
        // nb=3 grid, full-rank off-diagonal tiles (stay dense): exercises the
        // block forward/back substitution and the dense Schur path.
        let (n, b) = (96usize, 32usize);
        let a = diag_dominant_complex(n);
        let rhs = rhs_complex(n);
        let blr = BlrMatrix::from_dense(&a, n, n, b, 1e-12);
        assert_eq!(blr.nbr, 3);
        let lu = blr_lu_factor(blr, 1e-12, None).unwrap();
        let mut x = rhs.clone();
        blr_lu_solve(&lu, &mut x);
        assert!(rel_resid(&a, n, &x, &rhs) < 1e-9);
    }

    #[test]
    fn blr_lu_low_rank_front_solves() {
        // Smooth front with genuinely low-rank off-diagonal tiles: exercises the
        // low-rank panel solves, low-rank Schur products and recompression.
        let (n, b, eps) = (256usize, 64usize, 1e-8);
        let a = smooth_front(n);
        let blr = BlrMatrix::from_dense(&a, n, n, b, eps);
        assert!(blr.n_low_rank() > 0, "front should have low-rank tiles");
        let lu = blr_lu_factor(blr, eps, None).unwrap();
        let rhs = rhs_complex(n);
        let mut x = rhs.clone();
        blr_lu_solve(&lu, &mut x);
        let r = rel_resid(&a, n, &x, &rhs);
        assert!(r < 1e-4, "BLR-approximate solve rel resid {r}");
        assert!(
            lu.storage() < n * n,
            "factor storage {} should beat dense {}",
            lu.storage(),
            n * n
        );
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
