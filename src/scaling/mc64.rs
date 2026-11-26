//! MC64 wrapper: input preprocessing, Hungarian call, symmetric
//! averaging, and output guards.
//!
//! Given a sparse symmetric matrix (lower triangle only in the
//! input CSC), this module produces a symmetric scaling vector
//! `s` such that `D * A * D` (with `D = diag(s)`) has
//! magnitude-bounded off-diagonals and unit-scale diagonals.
//!
//! Algorithm (mirrors SPRAL `src/scaling.f90::hungarian_wrapper` in its
//! non-singular branch):
//!
//!   1. Expand the lower-triangle CSC to a full symmetric pattern,
//!      carrying the original values with the transpose entries.
//!   2. Drop explicit zero entries (log of zero is -inf).
//!   3. Compute `c[k] = log |a[k]|` on the remaining entries.
//!   4. For each column j, compute `cmax[j] = max_k c[k]` and replace
//!      each `c[k]` by `cmax[j] - c[k]`. The cost graph is now
//!      non-negative and has minimum 0 in each column.
//!   5. Run `hungarian_match` on the cost graph.
//!   6. Unwind the normalization (SPRAL scaling.f90:681-682):
//!      - `rscaling[i] = u[i]` (row dual unchanged)
//!      - `cscaling[j] = v[j] - cmax[j]` (column dual minus column max)
//!   7. Symmetric average (SPRAL scaling.f90:169):
//!      `s[i] = exp((rscaling[i] + cscaling[i]) / 2)`.
//!   8. Safety guards: clamp the exponent to avoid overflow,
//!      rewrite any `s[i] == 0` or non-finite result to `1.0`.
//!   9. On partial matching, set `s[i] = 1.0` for any index whose row
//!      OR column is unmatched (the two sets can differ even on a
//!      symmetric pattern) and return
//!      `ScalingInfo::PartialSingular { n_unmatched }`.
//!
//! The partial-singular path deviates from SPRAL, which runs a
//! second Hungarian pass on the full-rank submatrix and then
//! applies a Duff-Pralet correction (scaling.f90:688-800). Here
//! unmatched rows/columns fall back to identity scaling instead,
//! because KKT matrices from interior-point solvers are occasionally
//! structurally rank-deficient and a hard failure would bypass the
//! `ZeroPivotAction::ForceAccept` handling in the factorization.

use super::hungarian::{hungarian_match, CostGraph, Matching};
use super::ScalingInfo;
use crate::constants::LOG_HUGE;
use crate::error::RslabError;
use crate::sparse::csc::CscMatrix;

#[derive(Debug, Clone)]
pub(crate) struct Mc64Cache {
    pub perm: Vec<usize>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub cmax: Vec<f64>,
    pub n_matched: usize,
}

pub(crate) fn compute_matching(matrix: &CscMatrix) -> Result<Mc64Cache, RslabError> {
    let n = matrix.n;
    if n == 0 {
        return Ok(Mc64Cache {
            perm: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            cmax: Vec::new(),
            n_matched: 0,
        });
    }
    let (cost_graph, cmax) = build_cost_graph(matrix)?;
    let Matching {
        perm,
        u,
        v,
        n_matched,
    } = hungarian_match(&cost_graph);

    Ok(Mc64Cache {
        perm,
        u,
        v,
        cmax,
        n_matched,
    })
}

static LOG_RATE: crate::parallel::Rate =
    crate::parallel::Rate::new(crate::constants::SORT_NS_PER_ENTRY);

static COLMAX_RATE: crate::parallel::Rate =
    crate::parallel::Rate::new(crate::constants::SCAN_NS_PER_ENTRY);

pub(crate) fn compute_matching_general<T: crate::scalar::Scalar>(
    a: &crate::sparse::general::GeneralCsc<T>,
) -> Result<Mc64Cache, RslabError> {
    let n = a.n;
    if n == 0 {
        return Ok(Mc64Cache {
            perm: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            cmax: Vec::new(),
            n_matched: 0,
        });
    }
    let entries = a.values.len() as f64;
    let logs = crate::parallel::Work::of(entries, &LOG_RATE);
    let mut cost: Vec<f64> =
        crate::parallel::map::<crate::parallel::Rayon, _>(a.values.len(), logs, |k| {
            let m = a.values[k].magnitude();
            if m == 0.0 || !m.is_finite() {
                f64::NAN
            } else {
                m.ln()
            }
        });
    let mut cols: Vec<&mut [f64]> = Vec::with_capacity(n);
    let mut rest: &mut [f64] = &mut cost;
    for j in 0..n {
        let (head, tail) = rest.split_at_mut(a.col_ptr[j + 1] - a.col_ptr[j]);
        cols.push(head);
        rest = tail;
    }
    let scan = crate::parallel::Work::of(entries, &COLMAX_RATE);
    let cmax: Vec<f64> =
        crate::parallel::map_owned::<crate::parallel::Rayon, _, _>(cols, scan, |col| {
            let top = col
                .iter()
                .copied()
                .filter(|l| !l.is_nan())
                .fold(f64::NEG_INFINITY, f64::max);
            let top = if top.is_finite() { top } else { 0.0 };
            for c in col.iter_mut() {
                *c = top - *c;
            }
            top
        });
    let (col_ptr, row_idx) = if cost.iter().any(|c| c.is_nan()) {
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_idx = Vec::with_capacity(a.row_idx.len());
        let mut kept = 0;
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                if !cost[k].is_nan() {
                    row_idx.push(a.row_idx[k]);
                    cost[kept] = cost[k];
                    kept += 1;
                }
            }
            col_ptr[j + 1] = kept;
        }
        cost.truncate(kept);
        (col_ptr.into(), row_idx.into())
    } else {
        (a.col_ptr[..].into(), a.row_idx[..].into())
    };
    let graph = CostGraph {
        n,
        col_ptr,
        row_idx,
        cost,
    };
    let Matching {
        perm,
        u,
        v,
        n_matched,
    } = {
        let _s = rsprobe::scope!(DEBUG, "hungarian");
        hungarian_match(&graph)
    };
    Ok(Mc64Cache {
        perm,
        u,
        v,
        cmax,
        n_matched,
    })
}

pub(crate) fn unsymmetric_scaling(cache: &Mc64Cache) -> (Vec<f64>, Vec<f64>) {
    let guard = |x: f64| {
        let s = x.clamp(-LOG_HUGE, LOG_HUGE).exp();
        if s == 0.0 || !s.is_finite() {
            1.0
        } else {
            s
        }
    };
    let r: Vec<f64> = cache.u.iter().map(|&u| guard(u)).collect();
    let c: Vec<f64> = cache
        .v
        .iter()
        .zip(&cache.cmax)
        .map(|(&v, &cm)| guard(v - cm))
        .collect();
    (r, c)
}

pub(crate) fn compute_symmetric(matrix: &CscMatrix) -> Result<(Vec<f64>, ScalingInfo), RslabError> {
    let cache = compute_matching(matrix)?;
    Ok(scaling_from_cache(&cache))
}

pub(crate) fn scaling_from_cache(cache: &Mc64Cache) -> (Vec<f64>, ScalingInfo) {
    let n = cache.perm.len();
    if n == 0 {
        return (Vec::new(), ScalingInfo::Applied);
    }
    let Mc64Cache {
        perm,
        u,
        v,
        cmax,
        n_matched,
    } = cache;

    let mut row_matched = vec![false; n];
    for &r in perm.iter() {
        if r != usize::MAX {
            row_matched[r] = true;
        }
    }

    let mut scaling = vec![1.0_f64; n];
    for i in 0..n {
        if !cmax[i].is_finite() {
            scaling[i] = 1.0;
            continue;
        }

        if perm[i] == usize::MAX || !row_matched[i] {
            scaling[i] = 1.0;
            continue;
        }

        let mut arg = (u[i] + v[i] - cmax[i]) / 2.0;

        if !arg.is_finite() {
            scaling[i] = 1.0;
            continue;
        }
        // https://github.com/milanofthe/rslab/issues/291
        arg = arg.clamp(-LOG_HUGE, LOG_HUGE);

        let s = arg.exp();

        if s == 0.0 || !s.is_finite() {
            scaling[i] = 1.0;
        } else {
            scaling[i] = s;
        }
    }

    let info = if *n_matched == n {
        ScalingInfo::Applied
    } else {
        ScalingInfo::PartialSingular {
            n_unmatched: n - *n_matched,
        }
    };

    (scaling, info)
}

fn build_cost_graph(matrix: &CscMatrix) -> Result<(CostGraph<'static>, Vec<f64>), RslabError> {
    let n = matrix.n;

    let mut col_counts = vec![0usize; n];
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let val = matrix.values[k];
            if val == 0.0 {
                continue;
            }
            col_counts[j] += 1;
            if i != j {
                col_counts[i] += 1;
            }
        }
    }

    let mut col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        col_ptr[j + 1] = col_ptr[j] + col_counts[j];
    }
    let nnz_full = col_ptr[n];

    let mut row_idx = vec![0usize; nnz_full];
    let mut cost = vec![0.0_f64; nnz_full];
    let mut offsets: Vec<usize> = col_ptr[..n].to_vec();

    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let val = matrix.values[k];
            if val == 0.0 {
                continue;
            }
            let logabs = val.abs().ln();
            let p = offsets[j];
            row_idx[p] = i;
            cost[p] = logabs;
            offsets[j] += 1;
            if i != j {
                let q = offsets[i];
                row_idx[q] = j;
                cost[q] = logabs;
                offsets[i] += 1;
            }
        }
    }

    let mut pairs: Vec<(usize, f64)> = Vec::new();
    for j in 0..n {
        let start = col_ptr[j];
        let end = col_ptr[j + 1];
        pairs.clear();
        pairs.extend((start..end).map(|k| (row_idx[k], cost[k])));
        // https://github.com/milanofthe/rslab/issues/291
        pairs.sort_by_key(|&(r, _)| r);
        for (k, &(r, c)) in (start..end).zip(&pairs) {
            row_idx[k] = r;
            cost[k] = c;
        }
    }

    let mut cmax = vec![f64::NEG_INFINITY; n];
    for j in 0..n {
        let start = col_ptr[j];
        let end = col_ptr[j + 1];
        if start == end {
            continue;
        }
        let mut m = cost[start];
        for &c in &cost[(start + 1)..end] {
            if c > m {
                m = c;
            }
        }
        cmax[j] = m;
        for c in &mut cost[start..end] {
            *c = m - *c;
        }
    }

    let graph = CostGraph {
        n,
        col_ptr: col_ptr.into(),
        row_idx: row_idx.into(),
        cost,
    };
    Ok((graph, cmax))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_constants::{exp_of_logs, gamma, U64};

    #[test]
    fn diagonal_matrix_produces_inverse_sqrt_scaling() {
        let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        let (s, info) = compute_symmetric(&csc).unwrap();
        assert_eq!(info, ScalingInfo::Applied);
        let expected = [
            1.0 / 2.0_f64.sqrt(),
            1.0 / 3.0_f64.sqrt(),
            1.0 / 5.0_f64.sqrt(),
        ];
        for i in 0..3 {
            let y = csc.values[i].ln() / 2.0;
            let bound = (exp_of_logs(y) + gamma(2, U64)) * expected[i];
            assert!(
                (s[i] - expected[i]).abs() <= bound,
                "s[{}] = {}, expected {}",
                i,
                s[i],
                expected[i]
            );
        }
    }

    #[test]
    fn unmatched_row_with_matched_column_falls_back_to_identity() {
        let cache = Mc64Cache {
            perm: vec![1, usize::MAX],
            u: vec![0.0, 0.0],
            v: vec![2.0, 0.0],
            cmax: vec![1.0, 1.0],
            n_matched: 1,
        };
        let (s, info) = scaling_from_cache(&cache);
        assert_eq!(info, ScalingInfo::PartialSingular { n_unmatched: 1 });
        assert_eq!(
            s[0], 1.0,
            "index 0's ROW is unmatched; the contract requires identity \
             scaling, got {}",
            s[0]
        );
        assert_eq!(
            s[1], 1.0,
            "index 1's column is unmatched; must be identity, got {}",
            s[1]
        );
    }

    #[test]
    fn empty_matrix_returns_empty_scaling() {
        let csc = CscMatrix {
            n: 0,
            col_ptr: vec![0],
            row_idx: vec![],
            values: vec![],
        };
        let (s, info) = compute_symmetric(&csc).unwrap();
        assert!(s.is_empty());
        assert_eq!(info, ScalingInfo::Applied);
    }
}
