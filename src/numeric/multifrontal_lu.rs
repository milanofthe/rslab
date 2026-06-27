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
) -> Result<(FrontLu<T>, Vec<T>), FeralError> {
    let n = nrow;
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
    const NB: usize = 64;
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
    // Pivots from the factored diagonal of the fully-summed block.
    for k in 0..ncol {
        pivots[k] = f[k * n + k];
    }

    // Extract L (nrow × ncol col-major, unit lower) and U (nrow × ncol
    // col-major over the row index, with the pivot on the diagonal).
    let one = T::one();
    let mut l = vec![T::zero(); nrow * ncol];
    let mut u = vec![T::zero(); nrow * ncol];
    for c in 0..ncol {
        l[c * nrow + c] = one;
        u[c * nrow + c] = pivots[c];
        for r in (c + 1)..nrow {
            l[c * nrow + r] = f[c * nrow + r]; // L(r, c)
            u[c * nrow + r] = f[r * nrow + c]; // U(c, r)
        }
    }

    // Contribution block: `f`'s trailing block now holds the fully
    // Schur-updated A22 = A22 − L21·U12, accumulated across the per-panel GEMMs
    // above. Copy it out (column-major) for the parent's extend-add.
    let cnrow = nrow - ncol;
    let mut cb = vec![T::zero(); cnrow * cnrow];
    for c in 0..cnrow {
        for r in 0..cnrow {
            cb[c * cnrow + r] = f[(ncol + c) * n + (ncol + r)];
        }
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
) -> Result<NodeLu<T>, FeralError> {
    let snode = &sym.supernodes[s];
    let n = sym.n;
    let ncol = snode.ncol;
    let own_last = snode.first_col + ncol;

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

    // Extend-add each child's full contribution block.
    for &ch in &snode.children {
        let child = child_ref(node_results, ch)?;
        let cn = child.front.nrow - child.front.nelim;
        let crows = &child.row_indices[child.front.nelim..];
        let cb = &child.contrib;
        for jc in 0..cn {
            let lj = gloc[crows[jc]];
            for ic in 0..cn {
                let li = gloc[crows[ic]];
                f[lj * nrow + li] = f[lj * nrow + li] + cb[jc * cn + ic];
            }
        }
    }

    // Restore the `gloc` all-`usize::MAX` invariant for the next front.
    for &g in &ri {
        gloc[g] = usize::MAX;
    }

    let (front, contrib) = lu_front(f, nrow, ncol, perturb_floor)?;
    Ok(NodeLu {
        front,
        row_indices: ri,
        contrib,
    })
}

/// Factor a general (unsymmetric) sparse matrix `A` as `Pᵀ A P = L U` via
/// generic multifrontal LU with static pivoting. `a` holds the **full** matrix
/// (both triangles). Works for `T = f64`/`Complex<f64>` (and the `f32`
/// variants). Solve with [`solve_lu`].
pub fn factor_general_lu<T: Scalar>(
    a: &GeneralCsc<T>,
    opts: &GenericFactorOptions,
) -> Result<LuFactors<T>, FeralError> {
    a.validate()?;
    let n = a.n;
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

    // Analyze the symmetrized pattern (reuses the symmetric ordering / tree).
    let (col_ptr, row_idx) = symmetrized_lower_pattern(a);
    let symb = analyze(n, &col_ptr, &row_idx)?;
    let (sym, by_level) = symb
        .sym_and_levels()
        .ok_or_else(|| FeralError::InvalidInput("internal: empty symbolic".to_string()))?;

    // Full permuted matrix A_perm = Pᵀ A P and its transpose, both in permuted
    // numbering (no triangle folding — unsymmetric values are kept distinct).
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
            vals.push(a.values[k]);
        }
    }
    let a_perm = GeneralCsc::<T>::from_triplets(n, &rows, &cols, &vals)?;
    let a_perm_t = a_perm.transpose();

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
                    )
                    .map(|nf| (s, nf))
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        for (s, nf) in computed {
            node_results[s] = Some(nf);
        }
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
        n_perturbed,
    })
}

/// Solve `A x = b` from an unsymmetric LU factorization (`Pᵀ A P = L U`).
pub fn solve_lu<T: Scalar>(f: &LuFactors<T>, b: &[T]) -> Result<Vec<T>, FeralError> {
    let n = f.n;
    if b.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // ŷ = P_row b (apply the row permutation to the right-hand side).
    let mut y: Vec<T> = (0..n).map(|e| b[f.perm_row[e]]).collect();
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
    // Undo the permutation: out[perm[e]] = x[e].
    let mut out = vec![T::zero(); n];
    for e in 0..n {
        out[f.perm[e]] = x[e];
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
}
