//! Per-node machinery shared by the LDL^T and LU kernels: the cmod plan and
//! the global-to-local row map.

use super::Li;
use crate::numeric::supernodal::panel::Layout;
use crate::scalar::Scalar;

// https://github.com/milanofthe/rslab/issues/289
pub(crate) struct CmodPlan {
    /// The updaters with work for this node, in updater order.
    pub spans: Vec<Span>,
    /// The node's update work is large enough to fork (see
    /// [`CMOD_FORK_MIN_FLOPS`], raised to the parallel-GEMM threshold): it
    /// runs in the column blocks of the node's layout, on the pool as wide
    /// as their work is worth. Not bit-identical to the sequential mode
    /// (that one routes small updates through the scalar kernel and cuts no
    /// update at the blocks), which is why the pick is a pure function of
    /// the node.
    pub forks: bool,
}

// https://github.com/milanofthe/rslab/issues/289
pub(crate) fn unit_share(total: f64, workers: usize) -> f64 {
    total / (workers.max(1) * crate::constants::UPDATE_UNITS_PER_WORKER) as f64
}

pub(crate) fn cut(w: &[f64], share: f64, max: usize, first_end: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let (mut start, mut acc) = (0usize, 0.0f64);
    for (b, &x) in w.iter().enumerate() {
        let len = b - start;
        if len > 0 && b >= first_end.min(w.len()) && (len >= max || acc + x > share) {
            ranges.push((start, b));
            (start, acc) = (b, 0.0);
        }
        acc += x;
    }
    if start < w.len() {
        ranges.push((start, w.len()));
    }
    debug_assert!(ranges.iter().all(|&(a, b)| a < b));
    debug_assert!(ranges.iter().skip(1).all(|&(a, b)| b - a <= max.max(1)));
    debug_assert!(ranges
        .first()
        .map_or(w.is_empty(), |&(_, end)| end >= first_end.min(w.len())));
    ranges
}

#[derive(Clone, Copy)]
pub(crate) struct Span {
    pub k: usize,
    pub l: (usize, usize),
    pub u: (usize, usize),
}

impl CmodPlan {
    /// The plan of supernode `s` over its `updaters`; `lists(k)` gives an
    /// updater's pivot count and its off-block `L` row and `U` column lists
    /// (the rows and columns of its panel past its pivots). With `count_u`
    /// the `U12` updates count as work too (the LU kernel).
    pub fn new<'a>(
        sym: &crate::symbolic::SymbolicFactorization,
        s: usize,
        updaters: &[Li],
        lists: impl Fn(usize) -> (usize, &'a [Li], &'a [Li]),
        count_u: bool,
        k: &crate::KernelSettings,
    ) -> Self {
        let (first, ncol) = (sym.supernodes[s].first_col, sym.supernodes[s].ncol);
        let landing = |v: &[Li]| {
            let p0 = v.partition_point(|&g| (g as usize) < first);
            (
                p0,
                p0 + v[p0..].partition_point(|&g| (g as usize) < first + ncol),
            )
        };
        let mut spans = SPANS
            .try_with(|v| v.borrow_mut().pop())
            .ok()
            .flatten()
            .unwrap_or_default();
        spans.clear();
        let mut flops: usize = 0;
        for &k in updaters {
            let k = k as usize;
            let (nck, ol, ou) = lists(k);
            let (l, u) = (landing(ol), landing(ou));
            let lwork = (ol.len() - l.0) * (u.1 - u.0) * nck;
            let uwork = if count_u {
                (l.1 - l.0) * (ou.len() - u.1) * nck
            } else {
                0
            };
            if lwork + uwork == 0 {
                continue;
            }
            flops += lwork + uwork;
            spans.push(Span { k, l, u });
        }
        let forks = flops >= k.fork_min_flops.max(k.par_gemm);
        CmodPlan { spans, forks }
    }
}

impl Drop for CmodPlan {
    fn drop(&mut self) {
        let spans = std::mem::take(&mut self.spans);
        let _ = SPANS.try_with(|v| v.borrow_mut().push(spans));
    }
}

thread_local! {
    // https://github.com/milanofthe/rslab/issues/289
    static SPANS: std::cell::RefCell<Vec<Vec<Span>>> = const { std::cell::RefCell::new(Vec::new()) };
}

thread_local! {
    static GLOC_SCRATCH: [std::cell::RefCell<Vec<Li>>; 2] =
        const { [std::cell::RefCell::new(Vec::new()), std::cell::RefCell::new(Vec::new())] };
}

pub(crate) struct Gloc<'a> {
    map: Vec<Li>,
    rows: &'a [Li],
    slot: usize,
}

impl<'a> Gloc<'a> {
    pub fn new(n: usize, rows: &'a [Li]) -> Self {
        Self::in_slot(0, n, rows)
    }

    /// A second map alongside [`new`](Self::new)'s, on its own scratch.
    pub fn second(n: usize, rows: &'a [Li]) -> Self {
        Self::in_slot(1, n, rows)
    }

    fn in_slot(slot: usize, n: usize, rows: &'a [Li]) -> Self {
        let mut map = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c[slot].borrow_mut()));
        if map.len() < n {
            map.resize(n, Li::MAX);
        }
        for (li, &g) in rows.iter().enumerate() {
            debug_assert_eq!(map[g as usize], Li::MAX);
            map[g as usize] = li as Li;
        }
        Gloc { map, rows, slot }
    }
}

impl std::ops::Deref for Gloc<'_> {
    type Target = [Li];
    fn deref(&self) -> &[Li] {
        &self.map
    }
}

// https://github.com/milanofthe/rslab/issues/289
impl Drop for Gloc<'_> {
    fn drop(&mut self) {
        for &g in self.rows {
            self.map[g as usize] = Li::MAX;
        }
        let map = std::mem::take(&mut self.map);
        GLOC_SCRATCH.with(|c| *c[self.slot].borrow_mut() = map);
    }
}

#[inline]
pub(crate) fn perturb_pivot<T: Scalar>(d: T, abs_floor: f64) -> T {
    let mag = d.magnitude();
    if mag >= abs_floor {
        d
    } else if mag == 0.0 {
        T::from_real(abs_floor)
    } else {
        d * T::from_real(abs_floor / mag)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn schur_tiles<T: Scalar>(
    base: *mut T,
    lay: Layout,
    k0: usize,
    mt: usize,
    pw: usize,
    g: &[T],
    (r, r_ld): (&[T], usize),
    (c0, c1): (usize, usize),
    (tw, rb): (usize, usize),
    par: bool,
    pool: &crate::numeric::supernodal::ScratchPool<Vec<T>>,
    ks: &crate::KernelSettings,
) {
    use crate::constants::PANEL_BLOCK;
    use crate::dense::gemm_backend::{gemm, GemmMode};
    debug_assert!(g.len() >= mt * pw && c1 <= mt);
    debug_assert!(pw == 0 || c1 <= c0 || r.len() >= (pw - 1) * r_ld + c1);
    let (tw, rb) = (tw.max(1), rb.max(1));
    let diag_end = lay.w.saturating_sub(k0).min(mt);
    let mut pieces = Vec::new();
    let mut a = c0;
    while a < c1 {
        let block_end = ((k0 + a) / PANEL_BLOCK + 1) * PANEL_BLOCK - k0;
        let b = (a + tw).min(c1).min(block_end);
        pieces.push((a, b, a, b));
        let mut r0 = b;
        while r0 < mt {
            let cut = if r0 < diag_end { diag_end } else { mt };
            let r1 = (r0 + rb).min(cut);
            pieces.push((a, b, r0, r1));
            r0 = r1;
        }
        a = b;
    }
    let base = crate::numeric::supernodal::PanelPtr(base);
    let mode = GemmMode::new(gemm::Parallelism::None, ks);
    let piece = |q: usize| {
        let (a, b, r0, r1) = pieces[q];
        let w = b - a;
        let is_diagonal_block = r0 == a;
        // SAFETY: entries of the caller's columns on or below the diagonal;
        // the operands are the caller's blocks, disjoint from the panel.
        unsafe {
            if is_diagonal_block {
                let mut diag = pool.take();
                diag.clear();
                diag.resize(w * w, T::zero());
                gemm(
                    w,
                    w,
                    pw,
                    diag.as_mut_ptr(),
                    w as isize,
                    1,
                    false,
                    g.as_ptr().add(a),
                    mt as isize,
                    1,
                    r.as_ptr().add(a),
                    1,
                    r_ld as isize,
                    T::zero(),
                    T::one(),
                    false,
                    false,
                    false,
                    mode,
                );
                for j in 0..w {
                    let col = base.get().add(lay.at(k0 + a + j, k0 + a + j));
                    for i in j..w {
                        let p = col.add(i - j);
                        *p = *p - diag[i + j * w];
                    }
                }
            } else {
                let ld = if k0 + r0 < lay.w {
                    lay.w - (k0 + a) / PANEL_BLOCK * PANEL_BLOCK
                } else {
                    lay.m
                };
                gemm(
                    r1 - r0,
                    w,
                    pw,
                    base.get().add(lay.at(k0 + r0, k0 + a)),
                    ld as isize,
                    1,
                    true,
                    g.as_ptr().add(r0),
                    mt as isize,
                    1,
                    r.as_ptr().add(a),
                    1,
                    r_ld as isize,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    mode,
                );
            }
        }
    };
    let total = pieces.len();
    if par && total > 1 {
        crate::parallel::ranges::<crate::parallel::Rayon>(
            total,
            crate::parallel::Work::ns(f64::INFINITY),
            |r| r.for_each(&piece),
        );
    } else {
        (0..total).for_each(piece);
    }
}
