//! KLU-style sparse LU: BTF + per-block left-looking Gilbert-Peierls.
//!
//! The third direct path next to the multifrontal LDLᵀ and LU, built for
//! circuit-shaped matrices: extremely sparse, unsymmetric, near-triangularizable,
//! with diagonal blocks far too small for supernodal/BLAS-3 kernels to pay off.
//! Algorithmic reference: SuiteSparse KLU (Davis & Palamadai Natarajan); this is
//! an independent pure-Rust implementation, no FFI.
//!
//! Pipeline:
//!
//! 1. **Analyze** ([`KluSymbolic::analyze`]): maximum transversal + Tarjan SCC
//!    ([`crate::ordering::btf`]) permute the matrix to block *upper* triangular
//!    form with a zero-free diagonal (structural singularity is detected here),
//!    then AMD orders each irreducible diagonal block on its symmetrized
//!    pattern.
//! 2. **Factor** ([`KluSymbolic::factor`]): each diagonal block is factored by
//!    a left-looking Gilbert-Peierls LU — per-column depth-first reach on the
//!    growing L pattern, so the numeric work is proportional to the flop count,
//!    with threshold partial pivoting that prefers the (structurally nonzero)
//!    diagonal. Off-block entries are not factored; they only enter the block
//!    back-substitution. Optional row-max scaling equilibrates the rows first.
//! 3. **Refactor** ([`KluSolver::refactor`]): numeric-only re-factorization for
//!    a matrix with the *same* pattern (frequency sweeps, Newton steps): the
//!    stored pattern and pivot sequence are replayed with no symbolic work and
//!    no pivot search. A changed pattern is detected and rejected; a pivot that
//!    became zero under the frozen pivot order fails cleanly so the caller can
//!    re-[`factor`](KluSymbolic::factor) with pivoting.
//!
//! Every phase is strictly sequential and allocation-deterministic, so results
//! are **bit-identical across runs and thread counts** — this path doubles as
//! the determinism arbiter for the parallel multifrontal paths.

use crate::error::RslabError;
use crate::ordering::btf;
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;

const UNSET: usize = usize::MAX;

/// Options for the KLU path. Defaults follow SuiteSparse KLU: threshold
/// partial pivoting with strong diagonal preference (`pivot_tol = 1e-3`),
/// row-max scaling on, BTF on.
#[derive(Debug, Clone)]
pub struct KluSettings {
    /// Threshold for diagonal preference: the diagonal entry is taken as the
    /// pivot when `|a_jj| >= pivot_tol * max_i |a_ij|` over the eligible
    /// column. `1.0` is plain partial pivoting; small values keep the
    /// BTF/AMD-chosen diagonal (less fill) unless it is numerically tiny.
    pub pivot_tol: f64,
    /// Divide every row by its max-magnitude entry before factoring (and
    /// scale RHS/solution accordingly). Cheap and markedly more robust on
    /// badly row-equilibrated inputs.
    pub row_scaling: bool,
    /// Permute to block upper triangular form first. Disable only for
    /// experiments; without BTF the whole matrix is one block, structural
    /// singularity surfaces as a numeric zero pivot, and the diagonal
    /// preference loses its zero-free guarantee.
    pub btf: bool,
}

impl Default for KluSettings {
    fn default() -> Self {
        Self {
            pivot_tol: 1e-3,
            row_scaling: true,
            btf: true,
        }
    }
}

/// Symbolic analysis for the KLU path: the BTF block structure plus the
/// per-block fill-reducing ordering. Analyze once, then factor any number of
/// matrices sharing the pattern.
#[derive(Debug, Clone)]
pub struct KluSymbolic {
    n: usize,
    nnz: usize,
    /// Pre-pivot row permutation (new-to-old): BTF matching ∘ SCC order ∘
    /// per-block AMD. Partial pivoting at factor time refines this within
    /// each block.
    pre_row_perm: Vec<usize>,
    /// Column permutation (new-to-old); never changed by pivoting.
    col_perm: Vec<usize>,
    /// Diagonal-block boundaries; see [`crate::ordering::btf::BtfForm`].
    block_ptr: Vec<usize>,
}

impl KluSymbolic {
    /// Analyze with default [`KluSettings`].
    pub fn analyze<T: Scalar>(a: &GeneralCsc<T>) -> Result<Self, RslabError> {
        Self::analyze_with(a, &KluSettings::default())
    }

    /// Analyze the pattern of `a`: BTF (unless disabled) + per-block AMD.
    ///
    /// Fails with [`RslabError::StructurallySingular`] when no complete
    /// matching exists (some set of `k` columns has entries in fewer than `k`
    /// rows) — such a matrix is singular for *every* value assignment.
    pub fn analyze_with<T: Scalar>(
        a: &GeneralCsc<T>,
        settings: &KluSettings,
    ) -> Result<Self, RslabError> {
        a.validate()?;
        let n = a.n;

        let (mut pre_row_perm, mut col_perm, block_ptr) = if settings.btf {
            let form = btf::block_triangular_form(n, &a.col_ptr, &a.row_idx)
                .ok_or(RslabError::StructurallySingular)?;
            (form.row_perm, form.col_perm, form.block_ptr)
        } else {
            let ident: Vec<usize> = (0..n).collect();
            let bp = if n == 0 { vec![0] } else { vec![0, n] };
            (ident.clone(), ident, bp)
        };

        // Per-block AMD on the symmetrized block pattern (B + Bᵀ, with
        // diagonal, matching what the multifrontal paths feed rslab-amd).
        // Blocks of size <= 2 have nothing to reorder.
        let mut pinv0 = vec![0usize; n];
        for (k, &r) in pre_row_perm.iter().enumerate() {
            pinv0[r] = k;
        }
        for b in 0..block_ptr.len() - 1 {
            let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
            let bn = be - bs;
            if bn <= 2 {
                continue;
            }
            let mut adj: Vec<Vec<i32>> = vec![Vec::new(); bn];
            for lj in 0..bn {
                let c = col_perm[bs + lj];
                adj[lj].push(lj as i32); // diagonal (zero-free after matching)
                for &r in &a.row_idx[a.col_ptr[c]..a.col_ptr[c + 1]] {
                    let pre = pinv0[r];
                    if pre >= bs && pre < be {
                        let li = pre - bs;
                        if li != lj {
                            adj[lj].push(li as i32);
                            adj[li].push(lj as i32);
                        }
                    }
                }
            }
            let mut colptr_i32 = Vec::with_capacity(bn + 1);
            let mut rowidx_i32 = Vec::new();
            colptr_i32.push(0i32);
            for col in adj.iter_mut() {
                col.sort_unstable();
                col.dedup();
                rowidx_i32.extend_from_slice(col);
                colptr_i32.push(rowidx_i32.len() as i32);
            }
            let pat = rslab_ordering_core::CscPattern::new(bn, &colptr_i32, &rowidx_i32)
                .ok_or_else(|| {
                    RslabError::InvalidInput("klu: malformed block pattern".to_string())
                })?;
            let lperm = rslab_amd::amd_order(&pat).map_err(|e| {
                RslabError::InvalidInput(format!("klu: AMD ordering failed: {e:?}"))
            })?;
            // Apply the local (new-to-old) perm symmetrically to the block's
            // segment of both permutations.
            let old_rows: Vec<usize> = pre_row_perm[bs..be].to_vec();
            let old_cols: Vec<usize> = col_perm[bs..be].to_vec();
            for (i, &lp) in lperm.iter().enumerate() {
                pre_row_perm[bs + i] = old_rows[lp as usize];
                col_perm[bs + i] = old_cols[lp as usize];
            }
        }

        Ok(Self {
            n,
            nnz: a.nnz(),
            pre_row_perm,
            col_perm,
            block_ptr,
        })
    }

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of diagonal blocks in the BTF form.
    pub fn n_blocks(&self) -> usize {
        self.block_ptr.len() - 1
    }

    /// Size of the largest diagonal block (the only part that generates fill).
    pub fn max_block_size(&self) -> usize {
        (0..self.n_blocks())
            .map(|b| self.block_ptr[b + 1] - self.block_ptr[b])
            .max()
            .unwrap_or(0)
    }

    /// Diagonal-block boundaries (`n_blocks + 1` entries).
    pub fn block_ptr(&self) -> &[usize] {
        &self.block_ptr
    }

    /// Numeric factorization of `a`, which must share the analyzed pattern.
    pub fn factor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        settings: &KluSettings,
    ) -> Result<KluSolver<T>, RslabError> {
        let factors = factor_impl(self, a, settings)?;
        Ok(KluSolver { factors })
    }
}

/// The numeric KLU factorization: `P A Q = L U` per diagonal block plus the
/// off-block entries, with row scaling folded in.
#[derive(Debug, Clone)]
struct KluFactors<T> {
    n: usize,
    nnz_a: usize,
    block_ptr: Vec<usize>,
    /// Final row permutation (new-to-old), pivoting included.
    row_perm: Vec<usize>,
    /// Inverse: original row -> final position.
    pinv: Vec<usize>,
    col_perm: Vec<usize>,
    /// Per-original-row reciprocal scale factor (all 1 when scaling is off).
    rs_inv: Vec<f64>,
    scaled: bool,
    /// L: strictly-below-diagonal entries per column, unit diagonal implicit.
    /// Row indices are final positions within the column's block.
    l_colptr: Vec<usize>,
    l_rowidx: Vec<usize>,
    l_val: Vec<T>,
    /// U: strictly-above-diagonal within-block entries per column, stored in
    /// elimination (topological) order — the refactor replay order.
    u_colptr: Vec<usize>,
    u_rowidx: Vec<usize>,
    u_val: Vec<T>,
    udiag: Vec<T>,
    /// Off-block entries (rows in earlier blocks, final positions), per
    /// column in the input's storage order. Not factored; applied in the
    /// block back-substitution.
    f_colptr: Vec<usize>,
    f_rowidx: Vec<usize>,
    f_val: Vec<T>,
}

/// KLU solver handle: factor (or analyze+factor), then solve / refactor.
#[derive(Debug, Clone)]
pub struct KluSolver<T> {
    factors: KluFactors<T>,
}

fn pattern_mismatch() -> RslabError {
    RslabError::InvalidInput(
        "klu: matrix pattern does not match the symbolic analysis / stored factorization"
            .to_string(),
    )
}

/// Row-max scaling reciprocals (1 for empty rows / scaling off).
fn row_scale_inv<T: Scalar>(a: &GeneralCsc<T>, enabled: bool) -> Vec<f64> {
    let mut rs = vec![0.0f64; a.n];
    if enabled {
        for (k, &i) in a.row_idx.iter().enumerate() {
            let m = a.values[k].magnitude();
            if m > rs[i] {
                rs[i] = m;
            }
        }
    }
    rs.iter()
        .map(|&m| {
            if m > 0.0 && m.is_finite() {
                1.0 / m
            } else {
                1.0
            }
        })
        .collect()
}

fn factor_impl<T: Scalar>(
    sym: &KluSymbolic,
    a: &GeneralCsc<T>,
    settings: &KluSettings,
) -> Result<KluFactors<T>, RslabError> {
    a.validate()?;
    let n = sym.n;
    if a.n != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: a.n,
        });
    }
    if a.nnz() != sym.nnz {
        return Err(pattern_mismatch());
    }

    let rs_inv = row_scale_inv(a, settings.row_scaling);
    let mut pinv_pre = vec![0usize; n];
    for (k, &r) in sym.pre_row_perm.iter().enumerate() {
        pinv_pre[r] = k;
    }

    // bpinv[pre_position] = final position; assigned when that row is pivoted.
    let mut bpinv = vec![UNSET; n];
    // Numeric work vector and DFS state, all in pre-pivot position space.
    let mut x = vec![T::zero(); n];
    let mut stamp = vec![0usize; n];
    let mut node_stack = vec![0usize; n];
    let mut cur_stack = vec![0usize; n];
    let mut topo: Vec<usize> = Vec::with_capacity(n);
    let mut nonpiv: Vec<usize> = Vec::with_capacity(n);

    let mut l_colptr = vec![0usize];
    let mut l_rowidx: Vec<usize> = Vec::new();
    let mut l_val: Vec<T> = Vec::new();
    let mut u_colptr = vec![0usize];
    let mut u_rowidx: Vec<usize> = Vec::new();
    let mut u_val: Vec<T> = Vec::new();
    let mut udiag: Vec<T> = Vec::with_capacity(n);
    let mut f_colptr = vec![0usize];
    let mut f_rowidx: Vec<usize> = Vec::new();
    let mut f_val: Vec<T> = Vec::new();

    for b in 0..sym.block_ptr.len() - 1 {
        let (bs, be) = (sym.block_ptr[b], sym.block_ptr[b + 1]);

        if be - bs == 1 {
            // Singleton block: the pivot is the (structurally nonzero)
            // diagonal entry itself; everything else in the column is
            // off-block.
            let c = sym.col_perm[bs];
            let mut diag: Option<T> = None;
            for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                let pre = pinv_pre[a.row_idx[k]];
                let sv = a.values[k] * T::from_real(rs_inv[a.row_idx[k]]);
                if pre == bs {
                    diag = Some(sv);
                } else if pre < bs {
                    f_rowidx.push(bpinv[pre]);
                    f_val.push(sv);
                } else {
                    return Err(pattern_mismatch());
                }
            }
            let d = diag.ok_or(RslabError::SingularBasis { column: c })?;
            if d.magnitude() == 0.0 || !d.is_finite() {
                return Err(RslabError::SingularBasis { column: c });
            }
            bpinv[bs] = bs;
            udiag.push(d);
            l_colptr.push(l_rowidx.len());
            u_colptr.push(u_rowidx.len());
            f_colptr.push(f_rowidx.len());
            continue;
        }

        // General irreducible block: left-looking Gilbert-Peierls.
        for j in bs..be {
            let c = sym.col_perm[j];
            let sj = j + 1; // unique DFS stamp for this column
            topo.clear();
            nonpiv.clear();

            // Pass 1 — symbolic: DFS the reach of the column's within-block
            // pattern over the L columns factored so far. Pivotal nodes come
            // out in `topo` post-order; non-pivotal nodes (pivot candidates)
            // in `nonpiv`.
            for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                let pre = pinv_pre[a.row_idx[k]];
                if pre < bs {
                    continue; // off-block, handled in pass 2
                }
                if pre >= be {
                    return Err(pattern_mismatch());
                }
                if stamp[pre] == sj {
                    continue;
                }
                stamp[pre] = sj;
                if bpinv[pre] == UNSET {
                    nonpiv.push(pre);
                    continue;
                }
                let mut d = 0usize;
                node_stack[0] = pre;
                cur_stack[0] = l_colptr[bpinv[pre]];
                loop {
                    let u = node_stack[d];
                    let endp = l_colptr[bpinv[u] + 1];
                    let mut descended = false;
                    while cur_stack[d] < endp {
                        let ch = l_rowidx[cur_stack[d]];
                        cur_stack[d] += 1;
                        if stamp[ch] == sj {
                            continue;
                        }
                        stamp[ch] = sj;
                        if bpinv[ch] == UNSET {
                            nonpiv.push(ch);
                            continue;
                        }
                        d += 1;
                        node_stack[d] = ch;
                        cur_stack[d] = l_colptr[bpinv[ch]];
                        descended = true;
                        break;
                    }
                    if descended {
                        continue;
                    }
                    topo.push(u);
                    if d == 0 {
                        break;
                    }
                    d -= 1;
                }
            }

            // Pass 2 — scatter the scaled column values (off-block entries go
            // straight to F; earlier blocks are already fully pivoted, so
            // their final positions are known).
            for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                let r = a.row_idx[k];
                let pre = pinv_pre[r];
                let sv = a.values[k] * T::from_real(rs_inv[r]);
                if pre < bs {
                    f_rowidx.push(bpinv[pre]);
                    f_val.push(sv);
                } else {
                    x[pre] = sv;
                }
            }

            // Pass 3 — numeric update in topological order (reverse
            // post-order): each pivotal node's final value feeds its L column
            // into the remaining work vector, and becomes a U entry.
            for &u in topo.iter().rev() {
                let p = bpinv[u];
                let xu = x[u];
                x[u] = T::zero();
                u_rowidx.push(p);
                u_val.push(xu);
                for k in l_colptr[p]..l_colptr[p + 1] {
                    let lr = l_rowidx[k];
                    x[lr] = x[lr] - xu * l_val[k];
                }
            }

            // Pivot: max-magnitude candidate, overridden by the diagonal
            // (pre-position `j`) when it clears the threshold.
            let mut piv = UNSET;
            let mut maxmag = 0.0f64;
            for &np in &nonpiv {
                let m = x[np].magnitude();
                if m > maxmag {
                    maxmag = m;
                    piv = np;
                }
            }
            if piv == UNSET || maxmag == 0.0 || !maxmag.is_finite() {
                // Clear the candidates before failing so the work vector
                // never leaks values into a hypothetical caller retry.
                for &np in &nonpiv {
                    x[np] = T::zero();
                }
                return Err(RslabError::SingularBasis { column: c });
            }
            if stamp[j] == sj && bpinv[j] == UNSET {
                let dm = x[j].magnitude();
                if dm > 0.0 && dm >= settings.pivot_tol * maxmag {
                    piv = j;
                }
            }

            let dval = x[piv];
            x[piv] = T::zero();
            bpinv[piv] = j;
            udiag.push(dval);
            for &np in &nonpiv {
                if np == piv {
                    continue;
                }
                // Keep structural zeros: the pattern must be value-independent
                // for the refactor replay.
                l_rowidx.push(np);
                l_val.push(x[np] / dval);
                x[np] = T::zero();
            }
            l_colptr.push(l_rowidx.len());
            u_colptr.push(u_rowidx.len());
            f_colptr.push(f_rowidx.len());
        }

        // The block is fully pivoted: fix its L row indices from pre-pivot to
        // final positions (U and F indices were final at push time).
        for k in l_colptr[bs]..l_colptr[be] {
            l_rowidx[k] = bpinv[l_rowidx[k]];
        }
    }

    let mut row_perm = vec![0usize; n];
    let mut pinv = vec![0usize; n];
    for (&fin, &orig) in bpinv.iter().zip(&sym.pre_row_perm) {
        debug_assert_ne!(fin, UNSET);
        row_perm[fin] = orig;
        pinv[orig] = fin;
    }

    Ok(KluFactors {
        n,
        nnz_a: sym.nnz,
        block_ptr: sym.block_ptr.clone(),
        row_perm,
        pinv,
        col_perm: sym.col_perm.clone(),
        rs_inv,
        scaled: settings.row_scaling,
        l_colptr,
        l_rowidx,
        l_val,
        u_colptr,
        u_rowidx,
        u_val,
        udiag,
        f_colptr,
        f_rowidx,
        f_val,
    })
}

impl<T: Scalar> KluSolver<T> {
    /// One-shot analyze + factor with the given settings.
    pub fn factor(a: &GeneralCsc<T>, settings: &KluSettings) -> Result<Self, RslabError> {
        KluSymbolic::analyze_with(a, settings)?.factor(a, settings)
    }

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Number of BTF diagonal blocks.
    pub fn n_blocks(&self) -> usize {
        self.factors.block_ptr.len() - 1
    }

    /// Stored factor entries: L + U + diagonal + off-block.
    pub fn factor_nnz(&self) -> usize {
        self.factors.l_val.len()
            + self.factors.u_val.len()
            + self.factors.udiag.len()
            + self.factors.f_val.len()
    }

    /// Solve `A x = b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        let f = &self.factors;
        if b.len() != f.n {
            return Err(RslabError::DimensionMismatch {
                expected: f.n,
                got: b.len(),
            });
        }
        let mut w = vec![T::zero(); f.n];
        for (k, &orig) in f.row_perm.iter().enumerate() {
            w[k] = b[orig] * T::from_real(f.rs_inv[orig]);
        }
        self.solve_permuted(&mut w);
        let mut xout = vec![T::zero(); f.n];
        for (k, &c) in f.col_perm.iter().enumerate() {
            xout[c] = w[k];
        }
        Ok(xout)
    }

    /// Solve for `nrhs` right-hand sides stored row-major (`b[i * nrhs + col]`),
    /// matching [`crate::LuSolver::solve_many`]'s layout.
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
        let f = &self.factors;
        if nrhs == 0 || b.len() != f.n * nrhs {
            return Err(RslabError::DimensionMismatch {
                expected: f.n * nrhs.max(1),
                got: b.len(),
            });
        }
        let mut xout = vec![T::zero(); f.n * nrhs];
        let mut w = vec![T::zero(); f.n];
        for col in 0..nrhs {
            for (k, &orig) in f.row_perm.iter().enumerate() {
                w[k] = b[orig * nrhs + col] * T::from_real(f.rs_inv[orig]);
            }
            self.solve_permuted(&mut w);
            for (k, &c) in f.col_perm.iter().enumerate() {
                xout[c * nrhs + col] = w[k];
            }
        }
        Ok(xout)
    }

    /// Solve with iterative refinement against the exact matrix (up to
    /// `max_iter` refinement steps, keeping the best iterate by residual
    /// max-norm), mirroring [`crate::solve_lu_refined`].
    pub fn solve_refined(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        max_iter: usize,
    ) -> Result<Vec<T>, RslabError> {
        let n = self.factors.n;
        if a.n != n || b.len() != n {
            return Err(RslabError::DimensionMismatch {
                expected: n,
                got: b.len(),
            });
        }
        let mut x = self.solve(b)?;
        let mut ax = vec![T::zero(); n];
        let mut best_x = x.clone();
        let mut best_res = f64::INFINITY;
        for _ in 0..=max_iter {
            a.matvec(&x, &mut ax);
            let r: Vec<T> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
            let res = r.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            if res < best_res {
                best_res = res;
                best_x.clone_from(&x);
            }
            if res == 0.0 {
                break;
            }
            let dx = self.solve(&r)?;
            for (xi, &d) in x.iter_mut().zip(&dx) {
                *xi = *xi + d;
            }
        }
        Ok(best_x)
    }

    /// The block forward/backward substitution on the permuted/scaled vector.
    fn solve_permuted(&self, w: &mut [T]) {
        let f = &self.factors;
        for b in (0..f.block_ptr.len() - 1).rev() {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            // L (unit lower) forward within the block.
            for j in bs..be {
                let xj = w[j];
                if xj != T::zero() {
                    for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                        w[f.l_rowidx[k]] = w[f.l_rowidx[k]] - f.l_val[k] * xj;
                    }
                }
            }
            // U backward within the block.
            for j in (bs..be).rev() {
                let xj = w[j] / f.udiag[j];
                w[j] = xj;
                if xj != T::zero() {
                    for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                        w[f.u_rowidx[k]] = w[f.u_rowidx[k]] - f.u_val[k] * xj;
                    }
                }
            }
            // Off-block columns feed the rows of earlier blocks.
            for j in bs..be {
                let xj = w[j];
                if xj != T::zero() {
                    for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                        w[f.f_rowidx[k]] = w[f.f_rowidx[k]] - f.f_val[k] * xj;
                    }
                }
            }
        }
    }

    /// Numeric-only refactorization: replay the stored pattern and pivot
    /// sequence on new values with the **same** sparsity pattern. No symbolic
    /// work, no pivot search — the fast path for frequency sweeps and Newton
    /// steps. Fails with a pattern-mismatch error if `a`'s pattern deviates
    /// from the factored one, and with [`RslabError::SingularBasis`] if a
    /// frozen pivot becomes zero (re-`factor` with pivoting in that case).
    /// After an error the factorization is invalid; a subsequent successful
    /// `refactor` or a fresh `factor` makes it valid again.
    pub fn refactor(&mut self, a: &GeneralCsc<T>) -> Result<(), RslabError> {
        a.validate()?;
        let f = &mut self.factors;
        if a.n != f.n {
            return Err(RslabError::DimensionMismatch {
                expected: f.n,
                got: a.n,
            });
        }
        if a.nnz() != f.nnz_a {
            return Err(pattern_mismatch());
        }
        let rs_inv = row_scale_inv(a, f.scaled);

        let mut x = vec![T::zero(); f.n];
        let mut scattered: Vec<usize> = Vec::new();

        for b in 0..f.block_ptr.len() - 1 {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            for j in bs..be {
                let c = f.col_perm[j];
                // Scatter (final position space). Off-block entries must hit
                // the stored F pattern slot-for-slot — the same storage-order
                // walk as at factor time.
                scattered.clear();
                let mut fcur = f.f_colptr[j];
                for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                    let r = a.row_idx[k];
                    let fin = f.pinv[r];
                    let sv = a.values[k] * T::from_real(rs_inv[r]);
                    if fin < bs {
                        if fcur >= f.f_colptr[j + 1] || f.f_rowidx[fcur] != fin {
                            return Err(pattern_mismatch());
                        }
                        f.f_val[fcur] = sv;
                        fcur += 1;
                    } else if fin >= be {
                        return Err(pattern_mismatch());
                    } else {
                        x[fin] = sv;
                        scattered.push(fin);
                    }
                }
                if fcur != f.f_colptr[j + 1] {
                    return Err(pattern_mismatch());
                }

                // Replay the elimination in the stored topological order.
                for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                    let p = f.u_rowidx[k];
                    let xu = x[p];
                    x[p] = T::zero();
                    f.u_val[k] = xu;
                    for kl in f.l_colptr[p]..f.l_colptr[p + 1] {
                        let lr = f.l_rowidx[kl];
                        x[lr] = x[lr] - xu * f.l_val[kl];
                    }
                }
                let d = x[j];
                x[j] = T::zero();
                if d.magnitude() == 0.0 || !d.is_finite() {
                    return Err(RslabError::SingularBasis { column: c });
                }
                f.udiag[j] = d;
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    let lr = f.l_rowidx[k];
                    f.l_val[k] = x[lr] / d;
                    x[lr] = T::zero();
                }
                // Every scattered position must now be consumed: a leftover
                // value is an entry outside the stored pattern (the pattern
                // changed) and would otherwise be silently dropped.
                for &pos in &scattered {
                    if x[pos] != T::zero() {
                        return Err(pattern_mismatch());
                    }
                }
            }
        }
        f.rs_inv = rs_inv;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::multifrontal_ldlt::SolverSettings;
    use crate::numeric::multifrontal_lu::{factor_general_lu, solve_lu};
    use num_complex::Complex;

    fn resid<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
        let mut ax = vec![T::zero(); a.n];
        a.matvec(x, &mut ax);
        let num = b
            .iter()
            .zip(&ax)
            .map(|(&bi, &axi)| (bi - axi).magnitude())
            .fold(0.0, f64::max);
        let den = b.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
        num / den.max(1e-300)
    }

    /// Deterministic xorshift for value generation (no rand dependency).
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    /// Circuit-shaped test matrix: sparse, unsymmetric, diagonally weighted,
    /// structurally nonsingular, with genuinely reducible structure (a
    /// one-directional bridge between two internally coupled halves).
    fn circuit_like(n: usize, seed: u64) -> GeneralCsc<f64> {
        let mut rng = Rng(seed | 1);
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        let half = n / 2;
        for j in 0..n {
            r.push(j);
            c.push(j);
            v.push(4.0 + rng.next_f64());
            // couplings within the same half only (keeps two SCC groups)
            let base = if j < half { 0 } else { half };
            let span = if j < half { half } else { n - half };
            for t in 1..=3usize {
                let i = base + (j - base + t * 7 + 1) % span;
                if i != j {
                    r.push(i);
                    c.push(j);
                    v.push(rng.next_f64());
                }
            }
        }
        // one-directional bridge: second half feeds the first (rows in the
        // first half, columns in the second) -> reducible, never a single SCC
        for k in 0..4usize {
            r.push(k * 3 % half);
            c.push(half + (k * 5) % (n - half));
            v.push(0.5 + rng.next_f64().abs());
        }
        GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
    }

    #[test]
    fn klu_solves_circuit_like_and_matches_multifrontal() {
        let a = circuit_like(200, 42);
        let b: Vec<f64> = (0..200).map(|i| (i % 11) as f64 - 5.0).collect();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert!(s.n_blocks() >= 2, "bridge structure must be reducible");
        let x = s.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-12, "residual {}", resid(&a, &x, &b));
        // cross-check against the multifrontal LU
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let xr = solve_lu(&f, &b).unwrap();
        let diff = x
            .iter()
            .zip(&xr)
            .map(|(&p, &q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(diff < 1e-9, "klu vs multifrontal differ by {diff}");
    }

    #[test]
    fn klu_complex_small_diagonal_pivots() {
        // Small diagonal, large off-diagonals: threshold pivoting must
        // abandon the diagonal and still solve accurately (same layout as
        // the multifrontal LU pivoting test).
        let c = |re, im| Complex::new(re, im);
        let m = 6;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(0.3, 0.05));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(2.0, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(1.5, -0.2));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(1.8, 0.1));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(2.2, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let x = s.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-12, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn klu_lower_triangular_needs_no_fill() {
        // Lower bidiagonal: BTF flips it upper triangular; every block is a
        // singleton, so the factor stores no L/U entries at all.
        let n = 50;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(2.0 + (i % 3) as f64);
            if i + 1 < n {
                r.push(i + 1);
                c.push(i);
                v.push(-1.0);
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert_eq!(s.n_blocks(), n);
        // factor_nnz = n diagonal entries + (n-1) off-block entries, zero fill
        assert_eq!(s.factor_nnz(), 2 * n - 1);
        let b: Vec<f64> = (0..n).map(|i| i as f64 - 7.0).collect();
        let x = s.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-14);
    }

    #[test]
    fn klu_structurally_singular_detected() {
        // Column 2 shares its only row pattern with column 0 -> no complete
        // matching regardless of values.
        let a =
            GeneralCsc::<f64>::from_triplets(3, &[0, 1, 0], &[0, 1, 2], &[1.0, 1.0, 5.0]).unwrap();
        match KluSymbolic::analyze(&a) {
            Err(RslabError::StructurallySingular) => {}
            other => panic!("expected StructurallySingular, got {other:?}"),
        }
    }

    #[test]
    fn klu_numerically_singular_detected() {
        // Structurally fine 2x2 block, but rank 1 numerically: the second
        // pivot must come up exactly zero.
        let a = GeneralCsc::<f64>::from_triplets(
            2,
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &[1.0, 2.0, 2.0, 4.0],
        )
        .unwrap();
        match KluSolver::factor(&a, &KluSettings::default()) {
            Err(RslabError::SingularBasis { .. }) => {}
            other => panic!("expected SingularBasis, got {other:?}"),
        }
    }

    #[test]
    fn klu_factor_is_bit_deterministic() {
        let a = circuit_like(150, 7);
        let s1 = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let s2 = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert_eq!(s1.factors.l_val, s2.factors.l_val);
        assert_eq!(s1.factors.u_val, s2.factors.u_val);
        assert_eq!(s1.factors.udiag, s2.factors.udiag);
        assert_eq!(s1.factors.row_perm, s2.factors.row_perm);
        let b: Vec<f64> = (0..150).map(|i| (i % 13) as f64).collect();
        assert_eq!(s1.solve(&b).unwrap(), s2.solve(&b).unwrap());
    }

    #[test]
    fn klu_refactor_replays_and_matches_fresh_factor() {
        let a = circuit_like(150, 99);
        let mut s = KluSolver::factor(&a, &KluSettings::default()).unwrap();

        // Same values: the replay must reproduce the factor bit-identically.
        let (lv, uv, dv) = (
            s.factors.l_val.clone(),
            s.factors.u_val.clone(),
            s.factors.udiag.clone(),
        );
        s.refactor(&a).unwrap();
        assert_eq!(s.factors.l_val, lv);
        assert_eq!(s.factors.u_val, uv);
        assert_eq!(s.factors.udiag, dv);

        // New values, same pattern: the refactored solve must be accurate.
        let a2 = GeneralCsc::from_triplets(
            a.n,
            &{
                let mut rows = Vec::new();
                for j in 0..a.n {
                    for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                        rows.push(a.row_idx[k]);
                    }
                }
                rows
            },
            &{
                let mut cols = Vec::new();
                for j in 0..a.n {
                    for _ in a.col_ptr[j]..a.col_ptr[j + 1] {
                        cols.push(j);
                    }
                }
                cols
            },
            &a.values
                .iter()
                .enumerate()
                .map(|(k, &v)| v * (1.0 + 0.01 * ((k % 17) as f64)))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        s.refactor(&a2).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| (i % 9) as f64 - 4.0).collect();
        let x = s.solve(&b).unwrap();
        assert!(
            resid(&a2, &x, &b) < 1e-11,
            "refactor residual {}",
            resid(&a2, &x, &b)
        );
    }

    #[test]
    fn klu_refactor_rejects_changed_pattern() {
        let a = circuit_like(60, 5);
        let mut s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        // Move one off-diagonal entry to a fresh position (same nnz).
        let (mut rows, mut cols, vals): (Vec<usize>, Vec<usize>, Vec<f64>) = {
            let mut rr = Vec::new();
            let mut cc = Vec::new();
            let mut vv = Vec::new();
            for j in 0..a.n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    rr.push(a.row_idx[k]);
                    cc.push(j);
                    vv.push(a.values[k]);
                }
            }
            (rr, cc, vv)
        };
        let moved = rows.iter().zip(&cols).position(|(&r, &c)| r != c).unwrap();
        rows[moved] = (rows[moved] + 1) % a.n;
        cols[moved] = (cols[moved] + 1) % a.n;
        let a2 = GeneralCsc::from_triplets(a.n, &rows, &cols, &vals).unwrap();
        if a2.nnz() != a.nnz() {
            return; // duplicate collapse: not the case under test
        }
        assert!(s.refactor(&a2).is_err(), "changed pattern must be rejected");
    }

    #[test]
    fn klu_solve_many_matches_single() {
        let a = circuit_like(80, 3);
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let nrhs = 4;
        let b: Vec<f64> = (0..a.n * nrhs).map(|k| (k % 7) as f64 - 3.0).collect();
        let x = s.solve_many(&b, nrhs).unwrap();
        for col in 0..nrhs {
            let bc: Vec<f64> = (0..a.n).map(|i| b[i * nrhs + col]).collect();
            let xc = s.solve(&bc).unwrap();
            for i in 0..a.n {
                assert_eq!(x[i * nrhs + col], xc[i], "rhs {col} row {i}");
            }
        }
    }

    #[test]
    fn klu_solve_refined_tightens_residual() {
        let a = circuit_like(120, 11);
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| ((i * i) % 23) as f64 - 11.0).collect();
        let x = s.solve_refined(&a, &b, 2).unwrap();
        assert!(resid(&a, &x, &b) < 1e-13);
    }

    #[test]
    fn klu_without_btf_still_solves() {
        let a = circuit_like(100, 21);
        let s = KluSolver::factor(
            &a,
            &KluSettings {
                btf: false,
                ..KluSettings::default()
            },
        )
        .unwrap();
        assert_eq!(s.n_blocks(), 1);
        let b: Vec<f64> = (0..a.n).map(|i| (i % 5) as f64).collect();
        let x = s.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-12);
    }

    #[test]
    fn klu_empty_matrix() {
        let a = GeneralCsc::<f64>::from_triplets(0, &[], &[], &[]).unwrap();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert_eq!(s.solve(&[]).unwrap(), Vec::<f64>::new());
    }
}
