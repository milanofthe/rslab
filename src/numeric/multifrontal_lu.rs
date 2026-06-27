//! Generic **unsymmetric** sparse LU factorization over any [`Scalar`] field —
//! the general (non-symmetric) complex path, complementing the symmetric LDLᵀ
//! path in [`crate::numeric::multifrontal_generic`].
//!
//! It targets matrices whose *values* are unsymmetric (e.g. MoM A-EFIE
//! near-field saddle preconditioners, where the symmetric and antisymmetric
//! parts are comparable) but reuses the full symmetric machinery: the
//! fill-reducing ordering, supernodes, assembly tree, level parallelism
//! ([`analyze`](crate::numeric::multifrontal_generic::analyze)) and the SIMD
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
use crate::numeric::multifrontal_generic::{analyze, perturb_pivot, GenericFactorOptions};
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;
use crate::symbolic::SymbolicFactorization;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

// Opt-in coarse factorization profiler (set `RLA_PROFILE=1`): CPU-nanosecond
// accumulators for the assembly (scatter + extend-add) vs the per-front LU
// kernel, summed across all worker threads. Zero overhead when disabled.
static PROF_ASM_NS: AtomicU64 = AtomicU64::new(0);
static PROF_FRONT_NS: AtomicU64 = AtomicU64::new(0);
static PROF_PANEL_NS: AtomicU64 = AtomicU64::new(0);
static PROF_EXTRACT_NS: AtomicU64 = AtomicU64::new(0);

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
    /// Full `cnrow × cnrow` column-major contribution block `A22 − L21·U12`.
    contrib: Vec<T>,
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
    profile: bool,
) -> Result<(FrontLu<T>, Vec<T>), FeralError> {
    let n = nrow;
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
    Ok((
        FrontLu {
            nrow,
            nelim: ncol,
            l,
            u,
            rperm,
            n_perturbed,
        },
        cb,
    ))
}

fn child_ref<T: Scalar>(
    node_results: &[Option<NodeLu<T>>],
    ch: usize,
) -> Result<&NodeLu<T>, FeralError> {
    node_results
        .get(ch)
        .and_then(|o| o.as_ref())
        .ok_or_else(|| FeralError::InvalidInput("internal: missing child node".to_string()))
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
    node_results: &[Option<NodeLu<T>>],
    perturb_floor: Option<f64>,
    gloc: &mut [usize],
    fbuf: &mut Vec<T>,
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
    for &ch in &snode.children {
        let child = child_ref(node_results, ch)?;
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

    // Caller-owned per-thread scratch held at the all-`usize::MAX` invariant;
    // only `ri`'s entries are set and cleared again below. `fbuf` is the pooled
    // front buffer.
    debug_assert_eq!(gloc.len(), n);
    for (li, &g) in ri.iter().enumerate() {
        gloc[g] = li;
    }

    fbuf.clear();
    fbuf.resize(nrow * nrow, T::zero());
    let f = &mut fbuf[..];

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
    for &ch in &snode.children {
        let child = child_ref(node_results, ch)?;
        let cn = child.front.nrow - child.front.nelim;
        let crows = &child.row_indices[child.front.nelim..];
        let cb = &child.contrib;
        loc.clear();
        loc.extend(crows.iter().map(|&g| gloc[g]));
        for jc in 0..cn {
            let frow = loc[jc] * nrow;
            let cb_col = &cb[jc * cn..jc * cn + cn];
            for ic in 0..cn {
                let dst = frow + loc[ic];
                f[dst] = f[dst] + cb_col[ic];
            }
        }
    }

    // Restore the `gloc` all-`usize::MAX` invariant for the next front.
    for &g in &ri {
        gloc[g] = usize::MAX;
    }

    if let Some(t) = t_asm {
        PROF_ASM_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_front = profile.then(std::time::Instant::now);
    let (front, contrib) = lu_front(f, nrow, ncol, perturb_floor, profile)?;
    if let Some(t) = t_front {
        PROF_FRONT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    Ok(NodeLu {
        front,
        row_indices: ri,
        contrib,
    })
}

/// Reusable symbolic analysis for the unsymmetric LU path — the symmetrized
/// pattern `A ∪ Aᵀ` analyzed once. Pass to [`factor_general_lu_numeric`] for
/// each value-set that shares the pattern (frequency sweep / Newton).
pub struct LuSymbolic {
    symb: crate::numeric::multifrontal_generic::GenericSymbolic,
    n: usize,
    nnz: usize,
}

impl LuSymbolic {
    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)` of the symmetrized
    /// pattern — for factorization-cost diagnostics (front-size distribution and
    /// a factor-flop estimate). See [`GenericSymbolic::front_dims`].
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        self.symb.front_dims()
    }

    /// Number of assembly-tree levels (level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.symb.n_levels()
    }
}

/// PARDISO phase 1 for the general path: analyze the symmetrized pattern of `a`
/// (values ignored). Reuse the result across many [`factor_general_lu_numeric`]
/// calls that share the pattern.
pub fn analyze_general<T: Scalar>(a: &GeneralCsc<T>) -> Result<LuSymbolic, FeralError> {
    a.validate()?;
    let n = a.n;
    let nnz = a.row_idx.len();
    if n == 0 {
        return Ok(LuSymbolic {
            symb: analyze(0, &[0], &[])?,
            n: 0,
            nnz: 0,
        });
    }
    let (col_ptr, row_idx) = symmetrized_lower_pattern(a);
    let symb = analyze(n, &col_ptr, &row_idx)?;
    Ok(LuSymbolic { symb, n, nnz })
}

/// Factor a general (unsymmetric) sparse matrix `A` as `Pᵀ A P = L U` via
/// generic multifrontal LU with partial pivoting. `a` holds the **full** matrix
/// (both triangles). Convenience wrapper over [`analyze_general`] +
/// [`factor_general_lu_numeric`]; for *analyze once, factor many* keep the
/// [`LuSymbolic`] across calls. Solve with [`solve_lu`] / [`solve_lu_refined`].
pub fn factor_general_lu<T: Scalar>(
    a: &GeneralCsc<T>,
    opts: &GenericFactorOptions,
) -> Result<LuFactors<T>, FeralError> {
    factor_general_lu_numeric(&analyze_general(a)?, a, opts)
}

/// PARDISO phases 2–3 for the general path: numeric LU reusing a [`LuSymbolic`].
/// `a` must share the analyzed pattern (`n`, `nnz`).
#[allow(clippy::needless_range_loop)] // CSC column loops index col_ptr + scaling
pub fn factor_general_lu_numeric<T: Scalar>(
    lusym: &LuSymbolic,
    a: &GeneralCsc<T>,
    opts: &GenericFactorOptions,
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
        crate::dense::factor::ZeroPivotAction::Fail => None,
        crate::dense::factor::ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        crate::dense::factor::ZeroPivotAction::ForceAccept => {
            let anorm = a.values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    };

    let (sym, by_level) = lusym
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

    let profile = std::env::var("RLA_PROFILE").map(|v| v == "1").unwrap_or(false);
    if profile {
        PROF_ASM_NS.store(0, Ordering::Relaxed);
        PROF_FRONT_NS.store(0, Ordering::Relaxed);
    }

    let nsuper = sym.supernodes.len();
    let mut node_results: Vec<Option<NodeLu<T>>> = (0..nsuper).map(|_| None).collect();
    for level_nodes in by_level {
        let computed: Vec<(usize, NodeLu<T>)> = level_nodes
            .par_iter()
            .map_init(
                || (vec![usize::MAX; n], Vec::<T>::new()),
                |(gloc, fbuf), &s| {
                    factor_one_node_lu(
                        s,
                        sym,
                        &a_perm,
                        &a_perm_t,
                        &node_results,
                        perturb_floor,
                        gloc,
                        fbuf,
                        profile,
                    )
                    .map(|nf| (s, nf))
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        for (s, nf) in computed {
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
    let (mut l_col_ptr, mut l_row_idx, mut l_values) = (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
    let (mut u_row_ptr, mut u_col_idx, mut u_values) = (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
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
        (0..a.n).map(|i| (y[i] - b[i]).magnitude()).fold(0.0, f64::max)
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
        let f = factor_general_lu(&a, &GenericFactorOptions::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-10, "residual {}", resid(&a, &x, &b));
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
        let f = factor_general_lu(&a, &GenericFactorOptions::default()).unwrap();
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
        let f = factor_general_lu(&a, &GenericFactorOptions::default()).unwrap();
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
        let b: Vec<num_complex::Complex<f32>> = (0..n).map(|i| c((i % 5) as f32 - 2.0, 1.0)).collect();
        let f = factor_general_lu(&a, &GenericFactorOptions::default()).unwrap();
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
        let analysis = analyze_general(&template).unwrap();
        assert_eq!(analysis.n(), n);

        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 4.0, 1.0)).collect();
        for shift in [0.0, 2.5, -1.0] {
            let vv: Vec<Complex<f64>> = rr
                .iter()
                .zip(&cc)
                .map(|(&i, &j)| if i == j { c(8.0 + shift, 1.0) } else { c(-1.0, 0.2) })
                .collect();
            let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
            let phased =
                factor_general_lu_numeric(&analysis, &a, &GenericFactorOptions::default()).unwrap();
            let one_shot = factor_general_lu(&a, &GenericFactorOptions::default()).unwrap();
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
        use crate::dense::factor::ZeroPivotAction;
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

        let full = factor_general_lu(&a, &GenericFactorOptions::default()).unwrap();
        let opts = GenericFactorOptions {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: Some(5e-2),
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
