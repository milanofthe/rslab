//! Generic **unsymmetric** sparse LU factorization over any [`Scalar`] field —
//! the general (non-symmetric) complex path, complementing the symmetric LDLᵀ
//! path in [`crate::numeric::multifrontal_ldlt`].
//!
//! It targets matrices whose *values* are unsymmetric (e.g. MoM A-EFIE
//! near-field saddle preconditioners, where the symmetric and antisymmetric
//! parts are comparable) but reuses the full symmetric machinery: the
//! fill-reducing ordering, supernodes, assembly tree, level parallelism
//! ([`analyze`](crate::numeric::multifrontal_ldlt::analyze)) and the SIMD
//! `gemm` Schur kernel. Only the per-front kernel changes — an unsymmetric LU
//! producing separate `L` and `U` — and the analysis runs on the **symmetrized
//! pattern** `A ∪ Aᵀ` so the elimination structure carries fill for both
//! factors.
//!
//! ## Scope (v1)
//!
//! * **Static pivoting only** (no row interchange): pivots are taken in order,
//!   with sub-floor pivots perturbed (the [`ZeroPivotAction::PerturbToEps`]
//!   knob). This is the standard, fast choice for a **preconditioner** (PARDISO
//!   defaults to static pivoting too) and is well-suited to the equilibrated,
//!   unit-diagonal MoM matrices. Threshold partial pivoting is a later add.
//! * The reassembled factors are global sparse `L` (CSC, unit lower) and `U`
//!   (CSR, upper with the pivots on the diagonal), in factorization order.

use crate::error::FeralError;
use crate::numeric::blr::BlrMatrix;
use crate::numeric::multifrontal_ldlt::{
    analyze_with, perturb_pivot, AnalyzeOptions, BlrMode, FactorOptions, ZeroPivotAction,
};
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;
use crate::symbolic::SymbolicFactorization;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Reusable dense front-buffer pool. The multifrontal driver factors thousands
/// of fronts, each needing a transient `nrow² ` working buffer. Allocating one
/// per front (and freeing it) churns the system allocator with large, varying
/// sizes; on Windows the heap retains the freed blocks rather than returning
/// them to the OS, so peak RSS balloons far above the live set (the OOM the
/// pure-per-front allocation caused). This pool recycles a handful of buffers
/// (≈ the concurrency level) instead, capping the transient at the live set.
struct FrontPool<T>(Mutex<Vec<Vec<T>>>);

/// Only buffers up to this many entries are recycled. The churning majority of
/// small/medium fronts (which drive fragmentation) stay pooled; the rare huge
/// root/separator fronts are freed immediately rather than pinning their (GB-
/// scale) capacity in the pool for the whole factorization. 4M entries ≈ 64 MB
/// for `Complex<f64>` (front height ~2000).
const POOL_MAX_LEN: usize = 4_000_000;

impl<T: Scalar> FrontPool<T> {
    fn new() -> Self {
        FrontPool(Mutex::new(Vec::new()))
    }
    /// Take a buffer (reused if available) and zero-fill it to `len`.
    fn take(&self, len: usize) -> Vec<T> {
        let mut buf = self
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
            .unwrap_or_default();
        buf.clear();
        buf.resize(len, T::zero());
        buf
    }
    /// Return a buffer for reuse, unless it is an oversized (huge-front) buffer
    /// whose capacity we do not want to pin in the pool — those are dropped.
    fn give(&self, buf: Vec<T>) {
        if buf.capacity() <= POOL_MAX_LEN {
            self.0.lock().unwrap_or_else(|p| p.into_inner()).push(buf);
        }
    }
}

// Opt-in coarse factorization profiler (set `RLA_PROFILE=1`): CPU-nanosecond
// accumulators for the assembly (scatter + extend-add) vs the per-front LU
// kernel, summed across all worker threads. Zero overhead when disabled.
static PROF_ASM_NS: AtomicU64 = AtomicU64::new(0);
static PROF_FRONT_NS: AtomicU64 = AtomicU64::new(0);
static PROF_PANEL_NS: AtomicU64 = AtomicU64::new(0);
static PROF_EXTRACT_NS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Per-worker global→front-local index scratch (`usize`, scalar-independent),
    /// reused across every front a thread factors. Held at the all-`usize::MAX`
    /// invariant between uses; the assembly takes it, sets only the live front
    /// rows, and restores it. Replaces the old `map_init` workspace now that the
    /// driver is a work-stealing tree recursion rather than a level `par_iter`.
    static GLOC_SCRATCH: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// --- BLR contribution-block compression (opt-in `RLA_BLR_CB`) --------------
//
// The dominant factorization transient is the live contribution-block (CB)
// stack — `Σ cnrow²` across the active assembly frontier, ≈5× the L/U volume and
// the source of the memory spike / OOM. A CB is a frontal Schur complement of a
// smooth (MoM near-field) operator, so its off-diagonal tiles are numerically
// low-rank. Storing each large CB block-low-rank on the stack — and densifying
// it tile-by-tile only at the parent's extend-add — shrinks the live stack by
// the tile compression ratio at a small per-tile densify transient. The factor
// then becomes approximate (a preconditioner), exactly the LU/MoM use case
// where GMRES refinement absorbs the BLR tolerance. The fast dense `gemm`
// `lu_front` kernel is unchanged; only the CB *storage* is compressed.

/// A front's contribution block: dense, or BLR-compressed for the stack.
enum Contribution<T> {
    Dense(Vec<T>),
    Blr(BlrMatrix<T>),
}

impl<T: Scalar> Contribution<T> {
    /// Extend-add this CB into front `f` (`nrow`-tall, column-major) at the
    /// parent-local positions `loc` (CB index → front-local). `cn` is the CB
    /// dimension (used by the dense case; the BLR case carries its own).
    fn extend_add_into(&self, loc: &[usize], cn: usize, f: &mut [T], nrow: usize) {
        match self {
            Contribution::Dense(cb) => {
                for jc in 0..cn {
                    let frow = loc[jc] * nrow;
                    let cb_col = &cb[jc * cn..jc * cn + cn];
                    for ic in 0..cn {
                        let dst = frow + loc[ic];
                        f[dst] = f[dst] + cb_col[ic];
                    }
                }
            }
            Contribution::Blr(blr) => {
                // Densify one tile at a time (≤ b×b transient) and scatter.
                for ib in 0..blr.nbr {
                    let (r0, bm) = blr.row_extent(ib);
                    for jb in 0..blr.nbc {
                        let (c0, bn) = blr.col_extent(jb);
                        let tile = blr.block(ib, jb).to_dense();
                        for jj in 0..bn {
                            let frow = loc[c0 + jj] * nrow;
                            let tcol = &tile[jj * bm..jj * bm + bm];
                            for ii in 0..bm {
                                let dst = frow + loc[r0 + ii];
                                f[dst] = f[dst] + tcol[ii];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Aggregate CB-compression accounting (written when CB-BLR is active).
static CB_DENSE_SCALARS: AtomicU64 = AtomicU64::new(0);
static CB_BLR_SCALARS: AtomicU64 = AtomicU64::new(0);

/// Read and reset the BLR-CB accounting: `(dense_equiv_scalars, stored_scalars)`
/// summed over the CBs compressed since the last call — the realized CB-stack
/// compression ratio. Returns `(0, 0)` when CB compression is disabled.
pub fn take_blr_cb_stats() -> (u64, u64) {
    (
        CB_DENSE_SCALARS.swap(0, Ordering::Relaxed),
        CB_BLR_SCALARS.swap(0, Ordering::Relaxed),
    )
}

/// Per-front unsymmetric partial-factorization output, in elimination order
/// (static pivoting → no interchange, so front-local order is pivot order).
struct FrontLu<T> {
    nrow: usize,
    nelim: usize,
    /// Unit-lower `L` of the front, `nrow × nelim` column-major (multipliers
    /// below the diagonal; unit diagonal implicit).
    l: Vec<T>,
    /// Upper `U` of the front as `nrow × nelim` column-major over the *row*
    /// index: `u[c*nrow + r]` is `U(c, r)` for `r >= c` (the eliminated row `c`
    /// against front position `r`). The diagonal `u[c*nrow + c]` is the pivot.
    u: Vec<T>,
    /// Front row permutation from partial pivoting (`rperm[k]` is the original
    /// front-local row that supplied pivot position `k`). Identity on trailing
    /// rows. Drives the global row permutation `perm_row`.
    rperm: Vec<usize>,
    n_perturbed: usize,
}

/// Reassembled per-front result retained for the global pass and the parent's
/// extend-add.
struct NodeLu<T> {
    front: FrontLu<T>,
    row_indices: Vec<usize>,
    /// The `cnrow × cnrow` contribution block `A22 − L21·U12` — dense, or
    /// BLR-compressed for the stack (opt-in `RLA_BLR_CB`).
    contrib: Contribution<T>,
}

/// Stored unsymmetric LU factors, in factorization order. Solve with
/// [`solve_lu`]. The factored system is `Pᵀ A P = L U`, `perm[e]` mapping
/// factorization position `e` to the original index.
pub struct LuFactors<T> {
    pub n: usize,
    /// `L` in CSC (unit lower, explicit unit diagonal).
    pub l_col_ptr: Vec<usize>,
    pub l_row_idx: Vec<usize>,
    pub l_values: Vec<T>,
    /// `U` in CSR (upper; the diagonal entry carries the pivot).
    pub u_row_ptr: Vec<usize>,
    pub u_col_idx: Vec<usize>,
    pub u_values: Vec<T>,
    /// Column permutation: factorization position → original column index
    /// (`Pᵀ A P = L U`, the fill-reducing ordering).
    pub perm: Vec<usize>,
    /// Row permutation: factorization position → original row index. Differs
    /// from `perm` when partial pivoting interchanged rows.
    pub perm_row: Vec<usize>,
    /// Two-sided equilibration: the factor is of `Â = diag(d_row)·A·diag(d_col)`.
    /// Solve applies `D_r` to the RHS and `D_c` to the result. Both length `n`.
    pub d_row: Vec<f64>,
    pub d_col: Vec<f64>,
    /// Number of statically perturbed pivots.
    pub n_perturbed: usize,
}

impl<T: Scalar> LuFactors<T> {
    /// Stored fill: `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.l_values.len() + self.u_values.len()
    }
}

/// Lower triangle of the symmetrized pattern `A ∪ Aᵀ` as CSC `(col_ptr,
/// row_idx)`. The symmetric analysis needs a structurally symmetric pattern so
/// the elimination tree carries fill for both `L` and `U`.
fn symmetrized_lower_pattern<T: Scalar>(a: &GeneralCsc<T>) -> (Vec<usize>, Vec<usize>) {
    let n = a.n;
    // Counting-scatter (no `BTreeSet`: no per-element heap allocation, no
    // pointer-chasing). Each entry contributes a lower-triangle pair `(hi, lo)`
    // to bucket `lo`; buckets are then sorted + deduped into CSC.
    let mut counts = vec![0usize; n];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let lo = if i < j { i } else { j };
            counts[lo] += 1;
        }
    }
    let mut start = vec![0usize; n + 1];
    for j in 0..n {
        start[j + 1] = start[j] + counts[j];
    }
    let total = start[n];
    let mut scattered = vec![0usize; total];
    let mut cursor = start[..n].to_vec();
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
            scattered[cursor[lo]] = hi;
            cursor[lo] += 1;
        }
    }
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx = Vec::with_capacity(total);
    for j in 0..n {
        let seg = &mut scattered[start[j]..start[j + 1]];
        seg.sort_unstable();
        let mut last = usize::MAX;
        for &i in seg.iter() {
            if i != last {
                row_idx.push(i);
                last = i;
            }
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx)
}

/// Partially factor the first `ncol` fully-summed columns of a dense full front
/// `f` (`nrow × nrow`, column-major) by unsymmetric LU with static pivoting.
/// The trailing `[ncol, nrow)` block is returned as the contribution block
/// `A22 − L21·U12` (computed by one `gemm`).
fn lu_front<T: Scalar>(
    f: &mut [T],
    nrow: usize,
    ncol: usize,
    perturb_floor: Option<f64>,
    blr: BlrMode,
    profile: bool,
) -> Result<(FrontLu<T>, Contribution<T>), FeralError> {
    let n = nrow;
    // BLR compressibility probe (opt-in `RLA_BLR_PROBE`): on large fronts, report
    // how low-rank the assembled off-diagonal blocks are — the empirical go/no-go
    // and rank distribution for the BLR front factorization.
    if nrow >= 512 && std::env::var("RLA_BLR_PROBE").is_ok() {
        crate::numeric::blr::probe_front(f, nrow, ncol, 256);
    }
    let t_panel = profile.then(std::time::Instant::now);
    let mut pivots = vec![T::zero(); ncol];
    let mut n_perturbed = 0usize;
    // Row permutation of the front (partial pivoting interchanges rows). Only
    // the fully-summed rows [0, ncol) are ever interchanged.
    let mut rperm: Vec<usize> = (0..nrow).collect();

    // Blocked right-looking LU (LAPACK getrf-style): factor the fully-summed
    // columns in panels of width `NB`. Each panel is factored unblocked (getf2,
    // within-block partial pivoting), then the dominant trailing update runs as
    // a single SIMD `gemm` (rank-`NB`) — routing the O(ncol²·nrow) work through
    // the complex BLAS-3 kernel instead of scalar BLAS-2 column sweeps. This is
    // the structure MKL/PARDISO use for the supernodal panel.
    // Panel width. Smaller than the LAPACK-typical 64 because the `gemm` complex
    // kernel is very fast (≈460 Gflop/s rank-64) while the unblocked getf2 panel
    // is serial BLAS-2: a narrower panel shifts work off the slow serial path
    // onto the fast parallel GEMM. NB=32 measured best on the MoM fronts.
    const NB: usize = 32;
    let mut kb = 0;
    while kb < ncol {
        let ke = (kb + NB).min(ncol);
        // --- Panel factor (getf2) over columns [kb, ke), full height ---
        for k in kb..ke {
            // Partial pivoting within the fully-summed block (rows [k, ncol));
            // compare squared magnitudes (same argmax, no per-candidate sqrt).
            let mut p = k;
            let mut best = f[k * n + k].magnitude_sq();
            for i in (k + 1)..ncol {
                let m = f[k * n + i].magnitude_sq();
                if m > best {
                    best = m;
                    p = i;
                }
            }
            if p != k {
                for c in 0..nrow {
                    f.swap(c * n + k, c * n + p);
                }
                rperm.swap(k, p);
            }
            let mut piv = f[k * n + k];
            match perturb_floor {
                Some(floor) if piv.magnitude() < floor => {
                    piv = perturb_pivot(piv, floor);
                    n_perturbed += 1;
                }
                None if piv == T::zero() => return Err(FeralError::NumericallyRankDeficient),
                _ => {}
            }
            f[k * n + k] = piv;
            let pinv = piv.recip();
            // L multipliers: column k below the diagonal (full height).
            for i in (k + 1)..n {
                f[k * n + i] = f[k * n + i] * pinv;
            }
            // Within-panel Schur update only (columns [k+1, ke)); the trailing
            // block is deferred to the panel GEMM.
            for j in (k + 1)..ke {
                let ukj = f[j * n + k];
                if ukj != T::zero() {
                    for i in (k + 1)..n {
                        f[j * n + i] = f[j * n + i] - f[k * n + i] * ukj;
                    }
                }
            }
        }
        let pw = ke - kb; // panel width
                          // --- TRSM: U[kb:ke, ke:nrow] = L11⁻¹ · A[kb:ke, ke:nrow] ---
                          // L11 is the unit-lower `pw×pw` diagonal block; forward-substitute each
                          // trailing column over the panel rows.
        for j in ke..n {
            for r in (kb + 1)..ke {
                let mut s = f[j * n + r];
                for i in kb..r {
                    s = s - f[i * n + r] * f[j * n + i]; // L11(r,i)·U12(i,j)
                }
                f[j * n + r] = s;
            }
        }
        // --- GEMM: A[ke:, ke:] −= L[ke:, kb:ke] · U[kb:ke, ke:] (rank-`pw`) ---
        let mt = n - ke; // trailing rows = cols
        if mt > 0 && pw > 0 {
            let flops = (mt as u128) * (mt as u128) * (pw as u128);
            let par = if flops >= 8_000_000 {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            // SAFETY: L21 (rows≥ke × cols[kb,ke)), U12 (rows[kb,ke) × cols≥ke)
            // and the A22 dst (rows≥ke × cols≥ke) are pairwise-disjoint
            // sub-blocks of `f`, all in bounds under col-stride `n`. `T` is a
            // supported gemm element type.
            let base = f.as_mut_ptr();
            unsafe {
                gemm::gemm(
                    mt,
                    mt,
                    pw,
                    base.add(ke * n + ke),
                    n as isize,
                    1,
                    true,
                    base.add(kb * n + ke) as *const T,
                    n as isize,
                    1,
                    base.add(ke * n + kb) as *const T,
                    n as isize,
                    1,
                    T::one(),
                    -T::one(),
                    false,
                    false,
                    false,
                    par,
                );
            }
        }
        kb = ke;
    }
    if let Some(t) = t_panel {
        PROF_PANEL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_extract = profile.then(std::time::Instant::now);
    // Pivots from the factored diagonal of the fully-summed block.
    for k in 0..ncol {
        pivots[k] = f[k * n + k];
    }

    // Extract L (nrow × ncol col-major, unit lower) and U (nrow × ncol
    // col-major over the row index, with the pivot on the diagonal), plus the
    // contribution block (`f`'s Schur-updated trailing A22 = A22 − L21·U12).
    // Kept serial: the work-stealing driver already overlaps each front's serial
    // tail with other subtrees' work, so per-front extraction parallelism finds
    // no idle threads to use (measured: no gain).
    let one = T::one();
    let cnrow = nrow - ncol;
    let mut l = vec![T::zero(); nrow * ncol];
    let mut u = vec![T::zero(); nrow * ncol];
    let mut cb = vec![T::zero(); cnrow * cnrow];
    for c in 0..ncol {
        l[c * nrow + c] = one;
        u[c * nrow + c] = pivots[c];
        for r in (c + 1)..nrow {
            l[c * nrow + r] = f[c * nrow + r]; // L(r, c)
            u[c * nrow + r] = f[r * nrow + c]; // U(c, r)
        }
    }
    // Each CB column is a contiguous run of `f`'s trailing block → memcpy.
    for c in 0..cnrow {
        let base = (ncol + c) * n + ncol;
        cb[c * cnrow..c * cnrow + cnrow].copy_from_slice(&f[base..base + cnrow]);
    }

    if let Some(t) = t_extract {
        PROF_EXTRACT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    // Compress large CBs for the stack (BlrMode::ContributionBlocks). The
    // off-diagonal tiles of the frontal Schur complement are low-rank; storing
    // them compressed shrinks the live CB-stack transient. Densified
    // tile-by-tile at the parent extend-add.
    let contribution = match blr {
        BlrMode::ContributionBlocks { eps, min_cnrow, b } if cnrow >= min_cnrow => {
            let blr = BlrMatrix::from_dense(&cb, cnrow, cnrow, b, eps);
            CB_DENSE_SCALARS.fetch_add((cnrow * cnrow) as u64, Ordering::Relaxed);
            CB_BLR_SCALARS.fetch_add(blr.storage() as u64, Ordering::Relaxed);
            Contribution::Blr(blr)
        }
        _ => Contribution::Dense(cb),
    };
    Ok((
        FrontLu {
            nrow,
            nelim: ncol,
            l,
            u,
            rperm,
            n_perturbed,
        },
        contribution,
    ))
}

/// Factor one supernode's full front: build the (symmetric-pattern) row
/// structure, assemble the owned columns (L side), owned rows in trailing
/// columns (U side) and children contribution blocks, then LU-factor.
#[allow(clippy::too_many_arguments)]
fn factor_one_node_lu<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &GeneralCsc<T>,
    a_perm_t: &GeneralCsc<T>,
    child_refs: &[&NodeLu<T>],
    perturb_floor: Option<f64>,
    blr: BlrMode,
    pool: &FrontPool<T>,
    profile: bool,
) -> Result<NodeLu<T>, FeralError> {
    let snode = &sym.supernodes[s];
    let n = sym.n;
    let ncol = snode.ncol;
    let own_last = snode.first_col + ncol;
    let t_asm = profile.then(std::time::Instant::now);

    // Front row structure: own columns ++ sorted trailing rows (symmetrized
    // pattern of own columns plus children contribution rows).
    let mut trailing: Vec<usize> = Vec::new();
    for j in snode.first_col..own_last {
        for k in sym.permuted_pattern.col_ptr[j]..sym.permuted_pattern.col_ptr[j + 1] {
            let r = sym.permuted_pattern.row_idx[k];
            if r >= own_last {
                trailing.push(r);
            }
        }
    }
    for child in child_refs {
        for &r in &child.row_indices[child.front.nelim..] {
            if r >= own_last {
                trailing.push(r);
            }
        }
    }
    trailing.sort_unstable();
    trailing.dedup();
    let mut ri = Vec::with_capacity(ncol + trailing.len());
    ri.extend(snode.first_col..own_last);
    ri.extend(trailing);
    let nrow = ri.len();

    // Front buffer (transient `nrow²`), drawn from the reuse pool so the
    // thousands of large per-front buffers become a handful of recycled ones —
    // the fragmentation fix for the transient-memory OOM. Returned via
    // `pool.give` once `lu_front` has consumed it below.
    let mut fbuf: Vec<T> = pool.take(nrow * nrow);
    let f = &mut fbuf[..];

    // Take the thread-local global→local scratch for the assembly (held at the
    // all-`usize::MAX` invariant). It is returned before `lu_front` so the
    // front GEMM's work-stealing tasks can never re-enter the borrow.
    let mut gloc = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if gloc.len() < n {
        gloc.resize(n, usize::MAX);
    }
    for (li, &g) in ri.iter().enumerate() {
        gloc[g] = li;
    }

    // Owned columns (full): scatter a_perm column c into front column p.
    for p in 0..ncol {
        let c = snode.first_col + p;
        for k in a_perm.col_ptr[c]..a_perm.col_ptr[c + 1] {
            let g = a_perm.row_idx[k];
            let lr = gloc[g];
            if lr != usize::MAX {
                f[p * nrow + lr] = f[p * nrow + lr] + a_perm.values[k];
            }
        }
    }
    // Owned rows, trailing columns only (U12): scatter a_permᵀ column r (=
    // a_perm row r) into front row p for trailing front columns.
    for p in 0..ncol {
        let r = snode.first_col + p;
        for k in a_perm_t.col_ptr[r]..a_perm_t.col_ptr[r + 1] {
            let g = a_perm_t.row_idx[k];
            let lc = gloc[g];
            if lc != usize::MAX && lc >= ncol {
                f[lc * nrow + p] = f[lc * nrow + p] + a_perm_t.values[k];
            }
        }
    }

    // Extend-add each child's full contribution block. Map the child's
    // contribution rows to parent-front-local positions ONCE per child (`loc`,
    // reused across children) rather than re-indexing `gloc` inside the inner
    // loop — turning the cn² global→local lookups into cn, and slicing the
    // contiguous contribution column for the inner accumulation.
    let mut loc: Vec<usize> = Vec::new();
    for child in child_refs {
        let cn = child.front.nrow - child.front.nelim;
        let crows = &child.row_indices[child.front.nelim..];
        loc.clear();
        loc.extend(crows.iter().map(|&g| gloc[g]));
        child.contrib.extend_add_into(&loc, cn, f, nrow);
    }

    // Restore the all-`usize::MAX` invariant and return the scratch to the
    // thread-local before `lu_front` (which spawns work-stealing GEMM tasks).
    for &g in &ri {
        gloc[g] = usize::MAX;
    }
    GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);

    if let Some(t) = t_asm {
        PROF_ASM_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_front = profile.then(std::time::Instant::now);
    let (front, contrib) = lu_front(f, nrow, ncol, perturb_floor, blr, profile)?;
    if let Some(t) = t_front {
        PROF_FRONT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    // `lu_front` has copied L/U/CB out; recycle the front buffer.
    pool.give(fbuf);
    Ok(NodeLu {
        front,
        row_indices: ri,
        contrib,
    })
}

/// A supernode's own factor plus the flat `(supernode-id, factor)` list for the
/// rest of its subtree — the return shape of [`factor_subtree`].
type SubtreeFactors<T> = (NodeLu<T>, Vec<(usize, NodeLu<T>)>);

/// Recursively factor the assembly subtree rooted at supernode `s` with a
/// **work-stealing tree schedule**: the children's subtrees are factored
/// concurrently (`par_iter`) and this node is factored only once they are done.
/// Independent subtrees fill idle threads automatically, and the per-front GEMM
/// shares the *same* rayon pool, so there is no level-barrier stall and no
/// nested-pool contention — the parallel-efficiency win over the old
/// level-synchronous driver.
///
/// Returns this node's factor plus a flat `(supernode-id, factor)` list for the
/// whole subtree, which the caller scatters into `node_results` for the global
/// emit pass.
#[allow(clippy::too_many_arguments)]
fn factor_subtree<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &GeneralCsc<T>,
    a_perm_t: &GeneralCsc<T>,
    perturb_floor: Option<f64>,
    blr: BlrMode,
    pool: &FrontPool<T>,
    profile: bool,
) -> Result<SubtreeFactors<T>, FeralError> {
    let children = &sym.supernodes[s].children;
    // Factor the child subtrees concurrently.
    let mut outs: Vec<SubtreeFactors<T>> = children
        .par_iter()
        .map(|&ch| factor_subtree(ch, sym, a_perm, a_perm_t, perturb_floor, blr, pool, profile))
        .collect::<Result<Vec<_>, _>>()?;
    // Factor this node from the children's own (subtree-root) factors.
    let nf = {
        let child_refs: Vec<&NodeLu<T>> = outs.iter().map(|(own, _)| own).collect();
        factor_one_node_lu(
            s,
            sym,
            a_perm,
            a_perm_t,
            &child_refs,
            perturb_floor,
            blr,
            pool,
            profile,
        )?
    };
    // Free the children's contribution blocks NOW: they have just been
    // extend-added into this front and are never read again (the global emit
    // pass uses only L/U). The CB stack is `Σ cnrow²` ≈ 5× the L/U volume and,
    // when retained to the end, dominated peak memory and caused OOMs. Dropping
    // each CB the moment its parent consumes it keeps only the active
    // contribution frontier live — the standard multifrontal CB-stack.
    for (own, _) in outs.iter_mut() {
        own.contrib = Contribution::Dense(Vec::new());
    }
    // Flatten the subtree's factors for the global pass (child `i` is the i-th
    // entry of `children`).
    let mut subtree = Vec::new();
    for (i, (own, rest)) in outs.into_iter().enumerate() {
        subtree.push((children[i], own));
        subtree.extend(rest);
    }
    Ok((nf, subtree))
}

/// Reusable symbolic analysis for the unsymmetric LU path — the symmetrized
/// pattern `A ∪ Aᵀ` analyzed once. Pass to [`factor_general_lu_numeric`] for
/// each value-set that shares the pattern (frequency sweep / Newton).
pub struct LuSymbolic {
    symb: crate::numeric::multifrontal_ldlt::MultifrontalSymbolic,
    n: usize,
    nnz: usize,
}

impl LuSymbolic {
    /// PARDISO **phase 1**: analyze the symmetrized pattern `A ∪ Aᵀ` of `a`
    /// (values ignored, so any matrix with the target pattern works). Reuse the
    /// result across many [`factor`](Self::factor) calls that share the pattern
    /// — the unsymmetric twin of [`LdltSymbolic::analyze`].
    ///
    /// [`LdltSymbolic::analyze`]: crate::numeric::sparse_solver::LdltSymbolic::analyze
    pub fn analyze<T: Scalar>(a: &GeneralCsc<T>) -> Result<LuSymbolic, FeralError> {
        Self::analyze_with(a, &AnalyzeOptions::default())
    }

    /// [`analyze`](Self::analyze) with explicit composable [`AnalyzeOptions`]
    /// (child-reordering strategy).
    pub fn analyze_with<T: Scalar>(
        a: &GeneralCsc<T>,
        opts: &AnalyzeOptions,
    ) -> Result<LuSymbolic, FeralError> {
        a.validate()?;
        let n = a.n;
        let nnz = a.row_idx.len();
        if n == 0 {
            return Ok(LuSymbolic {
                symb: analyze_with(0, &[0], &[], opts)?,
                n: 0,
                nnz: 0,
            });
        }
        let (col_ptr, row_idx) = symmetrized_lower_pattern(a);
        let symb = analyze_with(n, &col_ptr, &row_idx, opts)?;
        Ok(LuSymbolic { symb, n, nnz })
    }

    /// PARDISO **phases 2–3**: equilibrate and LU-factor `a`, reusing this
    /// analysis, into a ready-to-solve [`LuSolver`]. `a` must share the analyzed
    /// pattern. The unsymmetric twin of [`LdltSymbolic::factor`].
    ///
    /// [`LdltSymbolic::factor`]: crate::numeric::sparse_solver::LdltSymbolic::factor
    pub fn factor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        opts: &FactorOptions,
    ) -> Result<LuSolver<T>, FeralError> {
        Ok(LuSolver {
            factors: factor_general_lu_numeric(self, a, opts)?,
        })
    }

    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)` of the symmetrized
    /// pattern — for factorization-cost diagnostics (front-size distribution and
    /// a factor-flop estimate). See [`MultifrontalSymbolic::front_dims`].
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        self.symb.front_dims()
    }

    /// Number of assembly-tree levels (level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.symb.n_levels()
    }
}

/// A factored unsymmetric LU solver, ready to solve against many right-hand
/// sides — the high-level, equilibrated counterpart of the raw [`LuFactors`]
/// (and the unsymmetric twin of [`LdltSolver`](crate::numeric::sparse_solver::LdltSolver)).
/// Build via [`LuSymbolic::factor`] (analyze once, factor many) or the one-shot
/// [`LuSolver::factor`].
pub struct LuSolver<T> {
    factors: LuFactors<T>,
}

impl<T: Scalar> LuSolver<T> {
    /// One-shot analyze + equilibrate + factor of a general matrix `A`.
    pub fn factor(a: &GeneralCsc<T>, opts: &FactorOptions) -> Result<Self, FeralError> {
        Ok(Self {
            factors: factor_general_lu(a, opts)?,
        })
    }

    /// Solve `A x = b` using the stored factors.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, FeralError> {
        solve_lu(&self.factors, b)
    }

    /// Solve `A · X = B` for `nrhs` right-hand sides at once. `b` and the
    /// returned `x` are **row-major** `n × nrhs` buffers (`b[i*nrhs + c]` is RHS
    /// `c` at row `i`). Faster than `nrhs` separate [`solve`](Self::solve) calls.
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, FeralError> {
        solve_lu_many(&self.factors, b, nrhs)
    }

    /// Solve `A x = b` with iterative refinement against the original matrix `a`
    /// (which must be the matrix this was factored from) — recovers accuracy on
    /// hard systems where the static-pivoted factor alone is insufficient.
    pub fn solve_refined(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        max_iter: usize,
    ) -> Result<Vec<T>, FeralError> {
        solve_lu_refined(&self.factors, a, b, max_iter)
    }

    /// Stored fill `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.factors.factor_nnz()
    }

    /// Number of statically perturbed pivots (preconditioner mode).
    pub fn n_perturbed(&self) -> usize {
        self.factors.n_perturbed
    }

    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Borrow the underlying raw factors (CSC `L` / CSR `U`, permutations,
    /// equilibration), e.g. to use as a [`Preconditioner`](crate::Preconditioner).
    pub fn factors(&self) -> &LuFactors<T> {
        &self.factors
    }
}

/// Factor a general (unsymmetric) sparse matrix `A` as `Pᵀ A P = L U` via
/// generic multifrontal LU with partial pivoting. `a` holds the **full** matrix
/// (both triangles). Convenience wrapper over [`LuSymbolic::analyze`] +
/// [`factor_general_lu_numeric`]; for *analyze once, factor many* keep the
/// [`LuSymbolic`] across calls. Solve with [`solve_lu`] / [`solve_lu_refined`].
pub fn factor_general_lu<T: Scalar>(
    a: &GeneralCsc<T>,
    opts: &FactorOptions,
) -> Result<LuFactors<T>, FeralError> {
    factor_general_lu_numeric(&LuSymbolic::analyze(a)?, a, opts)
}

/// PARDISO phases 2–3 for the general path: numeric LU reusing a [`LuSymbolic`].
/// `a` must share the analyzed pattern (`n`, `nnz`).
#[allow(clippy::needless_range_loop)] // CSC column loops index col_ptr + scaling
pub fn factor_general_lu_numeric<T: Scalar>(
    lusym: &LuSymbolic,
    a: &GeneralCsc<T>,
    opts: &FactorOptions,
) -> Result<LuFactors<T>, FeralError> {
    a.validate()?;
    let n = lusym.n;
    if a.n != n || a.row_idx.len() != lusym.nnz {
        return Err(FeralError::InvalidInput(
            "factor_general_lu_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    if n == 0 {
        return Ok(LuFactors {
            n: 0,
            l_col_ptr: vec![0],
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            u_row_ptr: vec![0],
            u_col_idx: Vec::new(),
            u_values: Vec::new(),
            perm: Vec::new(),
            perm_row: Vec::new(),
            d_row: Vec::new(),
            d_col: Vec::new(),
            n_perturbed: 0,
        });
    }

    let perturb_floor: Option<f64> = match opts.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = a.values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    };
    let blr = opts.blr;

    // The assembly-tree levels are no longer needed: the driver is a
    // work-stealing tree recursion, not a level-synchronous sweep.
    let (sym, _by_level) = lusym
        .symb
        .sym_and_levels()
        .ok_or_else(|| FeralError::InvalidInput("internal: empty symbolic".to_string()))?;

    // Two-sided equilibration Â = D_r A D_c with d_r[i] = 1/√maxⱼ|Aᵢⱼ|,
    // d_c[j] = 1/√maxᵢ|Aᵢⱼ|. Tames the dynamic range (these MoM near-field
    // matrices span ~6 orders) so the LU factor — and any incomplete drop —
    // stays well-scaled; the solve undoes it transparently. Computed from the
    // original (unpermuted) A.
    let mut rmax = vec![0.0f64; n];
    let mut cmax = vec![0.0f64; n];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let m = a.values[k].magnitude();
            if m > rmax[i] {
                rmax[i] = m;
            }
            if m > cmax[j] {
                cmax[j] = m;
            }
        }
    }
    let d_row: Vec<f64> = rmax
        .iter()
        .map(|&r| if r > 0.0 { 1.0 / r.sqrt() } else { 1.0 })
        .collect();
    let d_col: Vec<f64> = cmax
        .iter()
        .map(|&c| if c > 0.0 { 1.0 / c.sqrt() } else { 1.0 })
        .collect();

    // Full permuted, equilibrated matrix Â_perm = Pᵀ (D_r A D_c) P and its
    // transpose (no triangle folding — unsymmetric values kept distinct).
    let nnz = a.row_idx.len();
    let (mut rows, mut cols, mut vals) = (
        Vec::with_capacity(nnz),
        Vec::with_capacity(nnz),
        Vec::with_capacity(nnz),
    );
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            rows.push(sym.perm_inv[i]);
            cols.push(sym.perm_inv[j]);
            vals.push(a.values[k] * T::from_real(d_row[i] * d_col[j]));
        }
    }
    let a_perm = GeneralCsc::<T>::from_triplets(n, &rows, &cols, &vals)?;
    let a_perm_t = a_perm.transpose();

    let profile = std::env::var("RLA_PROFILE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if profile {
        PROF_ASM_NS.store(0, Ordering::Relaxed);
        PROF_FRONT_NS.store(0, Ordering::Relaxed);
    }

    let nsuper = sym.supernodes.len();
    // Roots of the assembly forest: supernodes that are no node's child.
    let mut is_child = vec![false; nsuper];
    for snode in &sym.supernodes {
        for &ch in &snode.children {
            is_child[ch] = true;
        }
    }
    let roots: Vec<usize> = (0..nsuper).filter(|&s| !is_child[s]).collect();
    // Factor every root subtree with the work-stealing tree schedule; the
    // children-before-parent dependency is the recursion structure itself.
    let pool = FrontPool::<T>::new();
    let root_outs: Vec<SubtreeFactors<T>> = roots
        .par_iter()
        .map(|&r| factor_subtree(r, sym, &a_perm, &a_perm_t, perturb_floor, blr, &pool, profile))
        .collect::<Result<Vec<_>, _>>()?;
    // Scatter the subtree factors into `node_results` (indexed by supernode id)
    // for the global emit pass, which still walks supernodes in postorder.
    let mut node_results: Vec<Option<NodeLu<T>>> = (0..nsuper).map(|_| None).collect();
    for (i, (own, subtree)) in root_outs.into_iter().enumerate() {
        node_results[roots[i]] = Some(own);
        for (s, nf) in subtree {
            node_results[s] = Some(nf);
        }
    }
    if profile {
        let asm = PROF_ASM_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let front = PROF_FRONT_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let panel = PROF_PANEL_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let extract = PROF_EXTRACT_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let total = (asm + front).max(1.0);
        eprintln!(
            "[RLA_PROFILE] CPU-ms  assembly {asm:.0} ({:.0}%)  front-LU {front:.0} ({:.0}%)  [panel {panel:.0} | extract {extract:.0}]",
            100.0 * asm / total,
            100.0 * front / total,
        );
    }

    let mut nodes: Vec<&NodeLu<T>> = Vec::with_capacity(nsuper);
    for node_opt in &node_results {
        match node_opt {
            Some(nd) => nodes.push(nd),
            None => {
                return Err(FeralError::InvalidInput(
                    "internal: unfactored supernode".to_string(),
                ))
            }
        }
    }

    // Assign factorization order e (static pivoting → front-local order is just
    // the column order) and the permutation.
    // Two index maps, distinct under row pivoting:
    //   col_pos_of_g[g] = factorization position eliminating COLUMN g
    //                     (→ U column indices, L column indices).
    //   row_pos_of_g[g] = factorization position whose PIVOT ROW is g
    //                     (→ L row indices). Equal to col_pos when no pivoting.
    let mut col_pos_of_g = vec![usize::MAX; n];
    let mut row_pos_of_g = vec![usize::MAX; n];
    let mut perm = vec![0usize; n];
    let mut perm_row = vec![0usize; n];
    let mut e = 0usize;
    for node in &nodes {
        let ff = &node.front;
        for j in 0..ff.nelim {
            // Columns are not interchanged, so position j maps to column
            // `row_indices[j]`; the pivot row is `row_indices[rperm[j]]`.
            let g_col = node.row_indices[j];
            let g_row = node.row_indices[ff.rperm[j]];
            col_pos_of_g[g_col] = e;
            row_pos_of_g[g_row] = e;
            perm[e] = sym.perm[g_col];
            perm_row[e] = sym.perm[g_row];
            e += 1;
        }
    }
    debug_assert_eq!(e, n, "every index eliminated exactly once");

    // Emit global L (CSC, columns in ascending e) and U (CSR, rows in ascending
    // e). A supernode's eliminated columns form a contiguous increasing
    // e-range, so iterating nodes then `j` yields columns/rows in order.
    let one = T::one();
    let (mut l_col_ptr, mut l_row_idx, mut l_values) =
        (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
    let (mut u_row_ptr, mut u_col_idx, mut u_values) =
        (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
    l_col_ptr.push(0);
    u_row_ptr.push(0);
    let mut lcol: Vec<(usize, T)> = Vec::new();
    let mut urow: Vec<(usize, T)> = Vec::new();
    for node in &nodes {
        let ff = &node.front;
        let nr = ff.nrow;
        for j in 0..ff.nelim {
            // Diagonal position (= column position = pivot-row position).
            let diag_e = col_pos_of_g[node.row_indices[j]];
            // L column (unit lower). Below-diagonal rows are indexed by the
            // *pivot-row* position of the front row physically at position `i`
            // (`rperm[i]`), which differs from its column position under
            // pivoting — this is the crux for a correct unsymmetric factor.
            lcol.clear();
            lcol.push((diag_e, one));
            for i in (j + 1)..nr {
                let v = ff.l[j * nr + i];
                if v != T::zero() {
                    lcol.push((row_pos_of_g[node.row_indices[ff.rperm[i]]], v));
                }
            }
            // Incomplete factorization (ILU): drop sub-threshold fill relative
            // to the column's largest multiplier, keeping the unit diagonal.
            if let Some(tau) = opts.drop_tol {
                let colmax = lcol
                    .iter()
                    .filter(|&&(r, _)| r != diag_e)
                    .map(|&(_, v)| v.magnitude())
                    .fold(0.0, f64::max);
                let thr = tau * colmax;
                lcol.retain(|&(r, v)| r == diag_e || v.magnitude() >= thr);
            }
            lcol.sort_unstable_by_key(|&(r, _)| r);
            for &(r, v) in &lcol {
                l_row_idx.push(r);
                l_values.push(v);
            }
            l_col_ptr.push(l_row_idx.len());
            // U row (upper, diagonal carries the pivot). Columns are not
            // interchanged → indexed by column position.
            urow.clear();
            urow.push((diag_e, ff.u[j * nr + j]));
            for i in (j + 1)..nr {
                let v = ff.u[j * nr + i];
                if v != T::zero() {
                    urow.push((col_pos_of_g[node.row_indices[i]], v));
                }
            }
            if let Some(tau) = opts.drop_tol {
                let rowmax = urow
                    .iter()
                    .filter(|&&(cc, _)| cc != diag_e)
                    .map(|&(_, v)| v.magnitude())
                    .fold(0.0, f64::max);
                let thr = tau * rowmax;
                urow.retain(|&(cc, v)| cc == diag_e || v.magnitude() >= thr);
            }
            urow.sort_unstable_by_key(|&(c, _)| c);
            for &(c, v) in &urow {
                u_col_idx.push(c);
                u_values.push(v);
            }
            u_row_ptr.push(u_col_idx.len());
        }
    }

    let n_perturbed = nodes.iter().map(|nd| nd.front.n_perturbed).sum();
    Ok(LuFactors {
        n,
        l_col_ptr,
        l_row_idx,
        l_values,
        u_row_ptr,
        u_col_idx,
        u_values,
        perm,
        perm_row,
        d_row,
        d_col,
        n_perturbed,
    })
}

/// Solve `A x = b` from an unsymmetric LU factorization (`Pᵀ A P = L U`).
#[allow(clippy::needless_range_loop)] // CSC/CSR solves index col_ptr/row_ptr + scaling
pub fn solve_lu<T: Scalar>(f: &LuFactors<T>, b: &[T]) -> Result<Vec<T>, FeralError> {
    let n = f.n;
    if b.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // ŷ = P_row · (D_r b): row-equilibrate then row-permute the RHS.
    let mut y: Vec<T> = (0..n)
        .map(|e| {
            let orig = f.perm_row[e];
            b[orig] * T::from_real(f.d_row[orig])
        })
        .collect();
    // Forward solve L y = ŷ (CSC, unit diagonal). Column-oriented: once y[e] is
    // final, eliminate it from the rows below.
    for e in 0..n {
        let ye = y[e];
        for k in f.l_col_ptr[e]..f.l_col_ptr[e + 1] {
            let i = f.l_row_idx[k];
            if i != e {
                y[i] = y[i] - f.l_values[k] * ye;
            }
        }
    }
    // Backward solve U x = y (CSR by row).
    let mut x = vec![T::zero(); n];
    for e in (0..n).rev() {
        let mut acc = y[e];
        let mut diag = T::one();
        for k in f.u_row_ptr[e]..f.u_row_ptr[e + 1] {
            let c = f.u_col_idx[k];
            if c == e {
                diag = f.u_values[k];
            } else {
                acc = acc - f.u_values[k] * x[c];
            }
        }
        x[e] = acc * diag.recip();
    }
    // Undo the column permutation and apply the column equilibration:
    // x_orig[perm[e]] = D_c[perm[e]] · x̂[e].
    let mut out = vec![T::zero(); n];
    for e in 0..n {
        let orig = f.perm[e];
        out[orig] = x[e] * T::from_real(f.d_col[orig]);
    }
    Ok(out)
}

/// Solve `A · X = B` for `nrhs` right-hand sides at once. `b` and the returned
/// `x` are **row-major** `n × nrhs` buffers (`b[i*nrhs + c]` is RHS `c` at row
/// `i`). The `L`/`U` structure is traversed once and each value applied to all
/// `nrhs` columns — faster than `nrhs` separate [`solve_lu`] calls.
pub fn solve_lu_many<T: Scalar>(
    f: &LuFactors<T>,
    b: &[T],
    nrhs: usize,
) -> Result<Vec<T>, FeralError> {
    let n = f.n;
    if nrhs == 0 || b.len() != n * nrhs {
        return Err(FeralError::DimensionMismatch {
            expected: n * nrhs,
            got: b.len(),
        });
    }
    // Ŷ = P_row · (D_r B): row-equilibrate then row-permute each RHS block.
    let mut y = vec![T::zero(); n * nrhs];
    for e in 0..n {
        let orig = f.perm_row[e];
        let s = T::from_real(f.d_row[orig]);
        let (eb, ob) = (e * nrhs, orig * nrhs);
        for c in 0..nrhs {
            y[eb + c] = b[ob + c] * s;
        }
    }
    // Forward solve L Y = Ŷ (CSC, unit diagonal).
    for e in 0..n {
        let eb = e * nrhs;
        for k in f.l_col_ptr[e]..f.l_col_ptr[e + 1] {
            let i = f.l_row_idx[k];
            if i != e {
                let lval = f.l_values[k];
                let ib = i * nrhs;
                for c in 0..nrhs {
                    y[ib + c] = y[ib + c] - lval * y[eb + c];
                }
            }
        }
    }
    // Backward solve U X = Y (CSR by row), in place in `y`.
    for e in (0..n).rev() {
        let eb = e * nrhs;
        let mut diag = T::one();
        for k in f.u_row_ptr[e]..f.u_row_ptr[e + 1] {
            let c_col = f.u_col_idx[k];
            let uval = f.u_values[k];
            if c_col == e {
                diag = uval;
            } else {
                let cb = c_col * nrhs;
                for c in 0..nrhs {
                    y[eb + c] = y[eb + c] - uval * y[cb + c];
                }
            }
        }
        let dinv = diag.recip();
        for c in 0..nrhs {
            y[eb + c] = y[eb + c] * dinv;
        }
    }
    // Undo column permutation + column equilibration: out[perm[e]] = D_c · x̂[e].
    let mut out = vec![T::zero(); n * nrhs];
    for e in 0..n {
        let orig = f.perm[e];
        let s = T::from_real(f.d_col[orig]);
        let (ob, eb) = (orig * nrhs, e * nrhs);
        for c in 0..nrhs {
            out[ob + c] = y[eb + c] * s;
        }
    }
    Ok(out)
}

/// Solve `A x = b` with iterative refinement against the original matrix `a`.
/// Each step computes the residual `r = b − A x` and applies the correction
/// `x ← x + (LU)⁻¹ r`, stopping once `‖r‖∞` stops improving or `max_iter` is
/// reached. This recovers the accuracy a static / within-block-pivoted factor
/// loses on ill-conditioned matrices, at the cost of a few extra solves.
pub fn solve_lu_refined<T: Scalar>(
    f: &LuFactors<T>,
    a: &GeneralCsc<T>,
    b: &[T],
    max_iter: usize,
) -> Result<Vec<T>, FeralError> {
    let n = f.n;
    if a.n != n || b.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    let mut x = solve_lu(f, b)?;
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
        let dx = solve_lu(f, &r)?;
        for (xi, &d) in x.iter_mut().zip(&dx) {
            *xi = *xi + d;
        }
    }
    Ok(best_x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn resid<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
        let mut y = vec![T::zero(); a.n];
        a.matvec(x, &mut y);
        (0..a.n)
            .map(|i| (y[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    #[test]
    fn f64_unsymmetric_tridiag() {
        // Unsymmetric real tridiagonal (full storage): diag 4, sub -1, super -2.
        let n = 20;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(4.0);
            if i + 1 < n {
                r.push(i + 1);
                c.push(i);
                v.push(-1.0);
                r.push(i);
                c.push(i + 1);
                v.push(-2.0);
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| i as f64 - 9.5).collect();
        let f = factor_general_lu(&a, &FactorOptions::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-10, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn lu_solve_many_matches_single() {
        let n = 12;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(5.0_f64);
            if i + 1 < n {
                r.push(i + 1);
                c.push(i);
                v.push(-1.0);
                r.push(i);
                c.push(i + 1);
                v.push(-2.0);
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let solver = LuSolver::factor(&a, &FactorOptions::default()).unwrap();
        let nrhs = 5;
        let b: Vec<f64> = (0..n * nrhs).map(|k| (k % 7) as f64 - 3.0).collect();
        let x = solver.solve_many(&b, nrhs).unwrap();
        for col in 0..nrhs {
            let bc: Vec<f64> = (0..n).map(|i| b[i * nrhs + col]).collect();
            let xc = solver.solve(&bc).unwrap();
            for i in 0..n {
                assert!(
                    (x[i * nrhs + col] - xc[i]).abs() < 1e-10,
                    "rhs {col} row {i}"
                );
            }
        }
    }

    #[test]
    fn pivoting_triggered_small_diagonal() {
        // Small diagonal, large off-diagonals → partial pivoting fires on
        // (nearly) every column. Well-conditioned overall, so the solve must
        // still hit a tiny residual: this isolates the pivoting/perm logic
        // (correctness) from numerical stability.
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
                vv.push(c(0.3, 0.05)); // small diagonal
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(2.0, 0.3)); // large off-diagonal
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
        let f = factor_general_lu(&a, &FactorOptions::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn complex_unsymmetric_2d_grid() {
        // 2D 5-point grid with unsymmetric neighbor couplings (right ≠ left).
        let c = |re, im| Complex::new(re, im);
        let m = 8;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2)); // p,q
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1)); // q,p (different!)
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let f = factor_general_lu(&a, &FactorOptions::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn complex_f32_lu_solves() {
        // The Complex<f32> LU path (used by the mixed-precision preconditioner).
        let c = |re: f32, im: f32| num_complex::Complex::<f32>::new(re, im);
        let m = 10;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(20.0, 2.0));
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    vv.push(c(-1.0, 0.2));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    vv.push(c(-1.5, 0.3));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<num_complex::Complex<f32>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<num_complex::Complex<f32>> =
            (0..n).map(|i| c((i % 5) as f32 - 2.0, 1.0)).collect();
        let f = factor_general_lu(&a, &FactorOptions::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        let r = resid(&a, &x, &b);
        assert!(r < 1e-3, "f32 LU residual {}", r);
    }

    #[test]
    fn phased_general_lu_analyze_once_factor_many() {
        // PARDISO workflow for the unsymmetric path: analyze the pattern once,
        // factor several value sets that share it — each must match the
        // one-shot factor's solve. The frequency-sweep / Newton use case.
        let c = |re, im| Complex::new(re, im);
        let m = 7;
        let n = m * m;
        let (mut rr, mut cc) = (Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                }
            }
        }
        // Template (values irrelevant) → analyze once.
        let template =
            GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vec![c(1.0, 0.0); rr.len()])
                .unwrap();
        let analysis = LuSymbolic::analyze(&template).unwrap();
        assert_eq!(analysis.n(), n);

        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 4.0, 1.0)).collect();
        for shift in [0.0, 2.5, -1.0] {
            let vv: Vec<Complex<f64>> = rr
                .iter()
                .zip(&cc)
                .map(|(&i, &j)| {
                    if i == j {
                        c(8.0 + shift, 1.0)
                    } else {
                        c(-1.0, 0.2)
                    }
                })
                .collect();
            let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
            let phased =
                factor_general_lu_numeric(&analysis, &a, &FactorOptions::default()).unwrap();
            let one_shot = factor_general_lu(&a, &FactorOptions::default()).unwrap();
            let xp = solve_lu(&phased, &b).unwrap();
            let xo = solve_lu(&one_shot, &b).unwrap();
            for (p, o) in xp.iter().zip(&xo) {
                assert!((p - o).norm() < 1e-10);
            }
            assert!(resid(&a, &xp, &b) < 1e-8);
        }
    }

    #[test]
    fn incomplete_lu_reduces_fill_and_still_solves() {
        use crate::numeric::multifrontal_ldlt::ZeroPivotAction;
        // Unsymmetric grid: incomplete LU (drop_tol) must shrink nnz(L+U) yet
        // still drive iterative refinement to a small residual — the MoM
        // sparse-preconditioner configuration.
        let c = |re, im| Complex::new(re, im);
        let m = 14;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

        let full = factor_general_lu(&a, &FactorOptions::default()).unwrap();
        let opts = FactorOptions {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: Some(5e-2),
            ..Default::default()
        };
        let inc = factor_general_lu(&a, &opts).unwrap();
        assert!(
            inc.factor_nnz() < full.factor_nnz(),
            "ILU should reduce fill: {} vs {}",
            inc.factor_nnz(),
            full.factor_nnz()
        );
        // The incomplete factor + a few refinement steps still solves accurately.
        let x = solve_lu_refined(&inc, &a, &b, 10).unwrap();
        assert!(resid(&a, &x, &b) < 1e-6, "residual {}", resid(&a, &x, &b));
    }
}
