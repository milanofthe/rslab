//! The supernodal panel form of a triangular factor: the one representation
//! the numeric drivers write and the solves read.
//!
//! A factor `L` (unit lower triangular, or a general lower triangular `U^T`)
//! is stored per supernode as one dense column-major panel of `(w + m) x w`
//! entries with leading dimension `w + m`: `w` columns of the supernode, the
//! `w x w` diagonal block on top (only its lower triangle is meaningful, the
//! unit diagonal is implicit for a unit factor), the `m` off-block rows below
//! in ascending elimination order. The row list is shared by the panel's
//! columns, so the index overhead is `m` words per supernode rather than one
//! per entry, and the values are exactly the dense panels the factorization
//! kernels produce. All panels live in **one buffer in supernode order** (the
//! [`PanelArena`] the drivers factor into), so a sweep streams through memory
//! the way the elimination tree is walked -- a left-looking driver hands its
//! panel over without a copy, and the solves run BLAS-3 style on the same
//! memory. Compared with a compressed-column factor with one `usize` index
//! per entry (24 bytes per complex entry, and a second copy for the solve
//! layout) this is the whole factor at 16 bytes per complex entry, once.
//!
//! [`PanelFactor::to_csc`] materializes the compressed-column form on demand
//! for the reference solves and the public CSC factor types.

use crate::numeric::supernodal::{Cells, PanelPtr, ScratchPool};
use crate::scalar::Scalar;

/// A lower triangular factor in supernodal panel form (see the module docs).
#[derive(Clone, Debug)]
pub struct PanelFactor<T> {
    /// Matrix dimension.
    pub n: usize,
    /// Supernode column starts in elimination order, `ns + 1` entries.
    pub sn_col: Vec<u32>,
    /// Off-block row indices, ascending within each supernode: those of
    /// supernode `s` are `rows[row_ptr[s]..row_ptr[s + 1]]`.
    pub rows: Vec<u32>,
    /// Row starts into `rows`, `ns + 1` entries.
    pub row_ptr: Vec<usize>,
    /// Panel starts into `vals`, `ns + 1` entries.
    pub val_ptr: Vec<usize>,
    /// The panels back to back in supernode order: supernode `s` is the
    /// `(w + m) x w` column-major block `vals[val_ptr[s]..val_ptr[s + 1]]`
    /// (leading dimension `w + m`); entries above the diagonal of the
    /// diagonal block are unused.
    pub vals: Vec<T>,
}

impl<T: Scalar> PanelFactor<T> {
    /// An empty factor of dimension 0.
    pub fn empty() -> Self {
        PanelFactor {
            n: 0,
            sn_col: vec![0],
            rows: Vec::new(),
            row_ptr: vec![0],
            val_ptr: vec![0],
            vals: Vec::new(),
        }
    }

    /// Number of supernodes.
    #[inline]
    pub fn n_supernodes(&self) -> usize {
        self.sn_col.len() - 1
    }

    /// `(c0, w, m)` of supernode `s`: first column, column count, off-block rows.
    #[inline]
    pub fn shape(&self, s: usize) -> (usize, usize, usize) {
        let c0 = self.sn_col[s] as usize;
        let w = self.sn_col[s + 1] as usize - c0;
        (c0, w, self.row_ptr[s + 1] - self.row_ptr[s])
    }

    /// The panel of supernode `s`.
    #[inline]
    pub fn panel(&self, s: usize) -> &[T] {
        &self.vals[self.val_ptr[s]..self.val_ptr[s + 1]]
    }

    /// Structural entry count of the factor: the lower triangles of the
    /// diagonal blocks (diagonal included) plus the off-block rows.
    pub fn nnz(&self) -> usize {
        (0..self.n_supernodes())
            .map(|s| {
                let (_, w, m) = self.shape(s);
                w * (w + 1) / 2 + m * w
            })
            .sum()
    }

    /// Bytes of the panel storage (values plus row indices).
    pub fn bytes(&self) -> usize {
        self.vals.len() * std::mem::size_of::<T>() + self.rows.len() * crate::constants::IDX32_BYTES
    }
}

/// The buffers of a factor handed back for the next factorization of the
/// same analysis to fill: the panels' values and a row list.
pub(crate) struct PanelStorage<T> {
    pub vals: Vec<T>,
    pub rows: Vec<u32>,
}

impl<T> Default for PanelStorage<T> {
    fn default() -> Self {
        PanelStorage {
            vals: Vec::new(),
            rows: Vec::new(),
        }
    }
}

/// Reorder the rows of a column-major `(ld) x w` panel in place so that row
/// `i` of the result is row `order[i]` of the input (`order` is a permutation
/// of `0..ld`). One column of scratch, no allocation per call beyond it.
pub fn permute_panel_rows<T: Copy>(
    panel: &mut [T],
    ld: usize,
    w: usize,
    order: &[u32],
    tmp: &mut Vec<T>,
) {
    debug_assert_eq!(order.len(), ld);
    debug_assert_eq!(panel.len(), ld * w);
    if order.iter().enumerate().all(|(i, &o)| o as usize == i) {
        return;
    }
    tmp.clear();
    for k in 0..w {
        let col = &mut panel[k * ld..(k + 1) * ld];
        tmp.extend_from_slice(col);
        for (i, &o) in order.iter().enumerate() {
            col[i] = tmp[o as usize];
        }
        tmp.clear();
    }
}

/// The factor's buffers while a numeric driver fills them: one value slot
/// per supernode of the analysis, sized `(w + m) x w` from its symbolic
/// shape, and one row slot of at least `m` entries, both back to back in
/// supernode order. Each slot is written by the one task that owns its
/// supernode (the drivers factor straight into it), read by the tasks that
/// update from it once it is published, and finished in place
/// ([`finish_slot`](Self::finish_slot)). [`finish`](Self::finish) then
/// closes the gaps left by dropped rows and yields the [`PanelFactor`].
///
/// An LU front that takes delayed pivots outgrows its slot: it
/// [`grow`](Self::grow)s into buffers of its own, which the accessors serve
/// from then on and [`finish`](Self::finish) moves into the factor. Such an
/// arena is [`lazy`](Self::lazy): a slot is zeroed by its owner when it is
/// [`claim`](Self::claim)ed rather than up front, so the slot of a grown
/// front is never touched (no resident memory), and the factor's buffer has
/// room for the growth.
pub(crate) struct PanelArena<T> {
    /// Slot starts per supernode of the analysis, `nsuper + 1` entries.
    slot_ptr: Vec<usize>,
    vals: Vec<T>,
    base: PanelPtr<T>,
    /// Row slot starts, `nsuper + 1` entries, and the rows.
    row_slot_ptr: Vec<usize>,
    rows: Vec<u32>,
    row_base: PanelPtr<u32>,
    /// The buffers of the supernodes that outgrew their slots (empty for
    /// the others): values and rows.
    grown: Cells<(Vec<T>, Vec<u32>)>,
    /// Slots are zeroed when claimed, not up front.
    lazy: bool,
    /// The scratch of [`finish_slot`](Self::finish_slot), per running call.
    scratch: ScratchPool<PanelScratch<T>>,
    /// Buffers handed in for reuse keep their capacity through
    /// [`finish`](Self::finish), for the next factorization to fill again.
    reused: bool,
}

// SAFETY: slots are disjoint and each has a single writer before any reader
// (the drivers' publication discipline, see the type docs).
unsafe impl<T: Send> Sync for PanelArena<T> {}

impl<T: Scalar> PanelArena<T> {
    /// Allocate the slots of the given sizes `(values, rows)`, one per
    /// supernode (see [`slot_sizes`]). The values are zero-initialized, so a
    /// slot starts as the zero panel the assembly expects.
    pub fn new(slots: impl Iterator<Item = (usize, usize)>) -> Self {
        Self::new_in(PanelStorage::default(), slots, 0, false, false)
    }

    /// [`new`](Self::new) in `storage`, the buffers of an earlier factor of
    /// this analysis: no allocation where their capacity suffices, and the
    /// capacity kept through [`finish`](Self::finish).
    pub fn reuse(storage: PanelStorage<T>, slots: impl Iterator<Item = (usize, usize)>) -> Self {
        Self::new_in(storage, slots, 0, true, false)
    }

    /// The slots of [`new`](Self::new), zeroed as they are claimed, in a
    /// buffer with room for `extra` more values: what the supernodes that
    /// [`grow`](Self::grow) may add to the factor.
    pub fn lazy(slots: impl Iterator<Item = (usize, usize)>, extra: usize) -> Self {
        Self::new_in(PanelStorage::default(), slots, extra, false, true)
    }

    fn new_in(
        storage: PanelStorage<T>,
        slots: impl Iterator<Item = (usize, usize)>,
        extra: usize,
        reused: bool,
        lazy: bool,
    ) -> Self {
        let PanelStorage { mut vals, mut rows } = storage;
        let cap = slots.size_hint().0 + 1;
        let (mut slot_ptr, mut row_slot_ptr) = (Vec::with_capacity(cap), Vec::with_capacity(cap));
        slot_ptr.push(0usize);
        row_slot_ptr.push(0usize);
        for (v, m) in slots {
            slot_ptr.push(slot_ptr.last().copied().unwrap_or(0) + v);
            row_slot_ptr.push(row_slot_ptr.last().copied().unwrap_or(0) + m);
        }
        let total = slot_ptr.last().copied().unwrap_or(0);
        vals.clear();
        vals.reserve(total + extra);
        if !lazy {
            // The whole factor, often hundreds of MB: zero it across the calling
            // pool rather than on one thread (41 ms serial on a 465 MB factor).
            crate::parallel::chunks_mut(
                &mut vals.spare_capacity_mut()[..total],
                crate::constants::ZERO_FILL_CHUNK,
                |_, chunk| {
                    chunk.iter_mut().for_each(|v| {
                        v.write(T::zero());
                    })
                },
            );
            // SAFETY: every element of the first `total` was written above.
            unsafe { vals.set_len(total) };
        }
        rows.clear();
        rows.resize(row_slot_ptr.last().copied().unwrap_or(0), 0);
        let base = PanelPtr(vals.as_mut_ptr());
        let row_base = PanelPtr(rows.as_mut_ptr());
        let ns = slot_ptr.len() - 1;
        PanelArena {
            slot_ptr,
            vals,
            base,
            row_slot_ptr,
            rows,
            row_base,
            grown: Cells::new_default(ns),
            lazy,
            scratch: ScratchPool::new(),
            reused,
        }
    }

    /// Entries of slot `s` (of its own buffer when it grew).
    #[inline]
    pub fn slot_len(&self, s: usize) -> usize {
        // SAFETY: a grown buffer is set by the owner before the slot is
        // published; see `grow`.
        match unsafe { self.grown.get(s) } {
            (g, _) if !g.is_empty() => g.len(),
            _ => self.slot_ptr[s + 1] - self.slot_ptr[s],
        }
    }

    /// Supernode `s`'s slot, zeroed in a lazy arena, for its owner to factor
    /// into.
    ///
    /// # Safety
    /// Only the owner of `s`, once, before anybody reads the slot.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn claim(&self, s: usize) -> &mut [T] {
        if self.lazy {
            let (p0, p1) = (self.slot_ptr[s], self.slot_ptr[s + 1]);
            let start = self.base.get().add(p0);
            let len = p1 - p0;
            if len >= crate::constants::ZERO_FILL_CHUNK {
                let at = PanelPtr(start);
                let chunks = len.div_ceil(crate::constants::ZERO_FILL_CHUNK);
                crate::parallel::for_each(chunks, 1, |c| {
                    let c0 = c * crate::constants::ZERO_FILL_CHUNK;
                    let c1 = (c0 + crate::constants::ZERO_FILL_CHUNK).min(len);
                    std::ptr::write_bytes(at.get().add(c0), 0, c1 - c0);
                });
            } else {
                std::ptr::write_bytes(start, 0, len);
            }
        }
        self.slot_mut(s)
    }

    /// Supernode `s` outgrows its slot: buffers of its own, `vals` zeroed
    /// values and `rows` rows, which the accessors serve from now on.
    ///
    /// # Safety
    /// Only the owner of `s`, once, instead of [`claim`](Self::claim).
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn grow(&self, s: usize, vals: usize, rows: usize) -> &mut [T] {
        self.grown.set(s, (vec![T::zero(); vals], vec![0u32; rows]));
        self.slot_mut(s)
    }

    /// The slot of supernode `s` for writing.
    ///
    /// # Safety
    /// Only the owner of `s` may call this, and not while any reader holds a
    /// reference from [`slot`](Self::slot); in a lazy arena only after
    /// [`claim`](Self::claim) or [`grow`](Self::grow).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slot_mut(&self, s: usize) -> &mut [T] {
        match self.grown.get_mut(s) {
            (g, _) if !g.is_empty() => g,
            _ => std::slice::from_raw_parts_mut(
                self.base.get().add(self.slot_ptr[s]),
                self.slot_ptr[s + 1] - self.slot_ptr[s],
            ),
        }
    }

    /// The slot of supernode `s` for reading.
    ///
    /// # Safety
    /// The owner must have published the slot (all writes done) before any
    /// reader calls this, and must not write again while readers exist.
    #[inline]
    pub unsafe fn slot(&self, s: usize) -> &[T] {
        match self.grown.get(s) {
            (g, _) if !g.is_empty() => g,
            _ => std::slice::from_raw_parts(
                self.base.get().add(self.slot_ptr[s]),
                self.slot_ptr[s + 1] - self.slot_ptr[s],
            ),
        }
    }

    /// The row slot of supernode `s`: room for its `m` off-block rows, for
    /// the emit to fill with their elimination indices before
    /// [`finish_slot`](Self::finish_slot).
    ///
    /// # Safety
    /// As [`slot_mut`](Self::slot_mut): only the owner of `s`.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn rows_mut(&self, s: usize) -> &mut [u32] {
        match self.grown.get_mut(s) {
            (g, r) if !g.is_empty() => r,
            _ => {
                let (r0, r1) = (self.row_slot_ptr[s], self.row_slot_ptr[s + 1]);
                std::slice::from_raw_parts_mut(self.row_base.get().add(r0), r1 - r0)
            }
        }
    }

    /// Finish supernode `s`'s panel of `w` columns and `m` off-block rows in
    /// its slots, the first `m` entries of its row slot filled
    /// ([`finish_panel`]).
    ///
    /// # Safety
    /// As [`slot_mut`](Self::slot_mut): only the owner of `s`, once.
    pub unsafe fn finish_slot(
        &self,
        s: usize,
        w: usize,
        m: usize,
        two_by_two: Option<&[bool]>,
    ) -> PanelOut {
        let mut scratch = self.scratch.take();
        finish_panel(
            self.slot_mut(s),
            &mut self.rows_mut(s)[..m],
            w,
            two_by_two,
            &mut scratch,
        )
    }

    /// Close the arena into the factor: `ncols` gives the column count of
    /// every supernode's finished panel (0 skipped), `outs(s)` the finished
    /// panel's row count and compacted length. Slots are moved in place to
    /// close the gaps of dropped rows and delayed pivots and to make room for
    /// the grown buffers, which are then copied in and freed, so the
    /// factor's buffers are exact. Returns the factor and the zero-slot
    /// count.
    pub fn finish(
        mut self,
        n: usize,
        ncols: impl Iterator<Item = usize>,
        mut outs: impl FnMut(usize) -> PanelOut,
    ) -> (PanelFactor<T>, usize) {
        let cap = ncols.size_hint().0 + 1;
        let mut sn_col: Vec<u32> = Vec::with_capacity(cap);
        let (mut val_ptr, mut row_ptr) = (Vec::with_capacity(cap), Vec::with_capacity(cap));
        sn_col.push(0);
        val_ptr.push(0usize);
        row_ptr.push(0usize);
        let mut zeros = 0usize;
        // Every kept panel: its supernode, length and rows.
        let mut kept: Vec<(usize, usize, usize)> = Vec::with_capacity(cap);
        for (s, w) in ncols.enumerate() {
            if w == 0 {
                continue;
            }
            let out = outs(s);
            debug_assert_eq!(out.len, (w + out.m) * w);
            debug_assert!(out.len <= self.slot_len(s));
            kept.push((s, out.len, out.m));
            sn_col.push(sn_col.last().copied().unwrap_or(0) + w as u32);
            val_ptr.push(val_ptr.last().copied().unwrap_or(0) + out.len);
            row_ptr.push(row_ptr.last().copied().unwrap_or(0) + out.m);
            zeros += out.zeros;
        }
        debug_assert_eq!(sn_col.last().copied(), Some(n as u32));
        let total = val_ptr.last().copied().unwrap_or(0);
        let total_rows = row_ptr.last().copied().unwrap_or(0);
        // A panel in its slot moves to its place in the factor: the ones
        // moving up first, from the last down, then the ones moving down,
        // from the first up; no source is overwritten before it is read (a
        // panel ends within its slot, and the places keep the slots'
        // order). The grown buffers fill their places last.
        let in_slot = |s: usize| unsafe { self.grown.get(s).0.is_empty() };
        let grows = kept.iter().any(|&(s, _, _)| !in_slot(s));
        // Room for the growth was reserved (`lazy`); a factor beyond it is a
        // driver's error.
        assert!(
            total <= self.vals.capacity(),
            "the factor outgrew its buffer"
        );
        if total_rows > self.rows.len() {
            self.rows.resize(total_rows, 0);
        }
        let base = self.vals.as_mut_ptr();
        let mut moves = |up: bool| {
            let order: Box<dyn Iterator<Item = usize>> = if up {
                Box::new((0..kept.len()).rev())
            } else {
                Box::new(0..kept.len())
            };
            for i in order {
                let (s, len, m) = kept[i];
                if !in_slot(s) {
                    continue;
                }
                let (src, dst) = (self.slot_ptr[s], val_ptr[i]);
                let (rsrc, rdst) = (self.row_slot_ptr[s], row_ptr[i]);
                if (dst > src) == up && dst != src {
                    // SAFETY: both ranges lie in the buffer's capacity and the
                    // source holds the finished panel; `copy` handles overlap.
                    unsafe { std::ptr::copy(base.add(src), base.add(dst), len) };
                }
                if (rdst > rsrc) == up && rdst != rsrc {
                    self.rows.copy_within(rsrc..rsrc + m, rdst);
                }
            }
        };
        moves(true);
        moves(false);
        let mut moved = kept
            .iter()
            .enumerate()
            .any(|(i, &(s, _, _))| !in_slot(s) || self.slot_ptr[s] != val_ptr[i]);
        if grows {
            for (i, &(s, len, m)) in kept.iter().enumerate() {
                if in_slot(s) {
                    continue;
                }
                // SAFETY: the record is the finished supernode's own.
                let (g, r) = std::mem::take(unsafe { self.grown.get_mut(s) });
                // SAFETY: the place lies in the capacity; the buffer is separate.
                unsafe { std::ptr::copy_nonoverlapping(g.as_ptr(), base.add(val_ptr[i]), len) };
                self.rows[row_ptr[i]..row_ptr[i] + m].copy_from_slice(&r[..m]);
            }
            moved = true;
        }
        let (mut vals, mut rows) = (self.vals, self.rows);
        // SAFETY: every entry of the first `total` was written: each panel's
        // place by its move or copy, and the places tile `[0, total)`.
        unsafe { vals.set_len(total) };
        if (moved || total < vals.capacity()) && !self.reused {
            vals.shrink_to_fit();
        }
        rows.truncate(total_rows);
        if !self.reused {
            rows.shrink_to_fit();
        }
        // Exact, as the plan keeps it (the memory plan counts it so).
        row_ptr.shrink_to_fit();
        (
            PanelFactor {
                n,
                sn_col,
                rows,
                row_ptr,
                val_ptr,
                vals,
            },
            zeros,
        )
    }
}

/// The arena slot `(values, rows)` of a supernode whose panel is `(w + m) x
/// w` (`w` columns, `m` off-block rows): none for an empty supernode.
pub(crate) fn slot_sizes(w: usize, m: usize) -> (usize, usize) {
    if w == 0 {
        (0, 0)
    } else {
        ((w + m) * w, m)
    }
}

/// One supernode's finished panel, as left in its arena slots: its
/// off-block row count `m` (the rows in the row slot, elimination indices,
/// ascending), the compacted panel length `(w + m) * w`, and the zero-slot
/// count.
#[derive(Default)]
pub(crate) struct PanelOut {
    pub m: usize,
    pub len: usize,
    /// Structural slots (diagonal excluded) holding an exact zero: numeric
    /// cancellation or the symmetrized pattern of an unsymmetric matrix.
    /// `nnz() - zeros` is the stored nonzero count a sparse factor would
    /// report.
    pub zeros: usize,
}

/// The buffers of one [`finish_panel`] call.
pub(crate) struct PanelScratch<T> {
    order: Vec<u32>,
    full: Vec<u32>,
    sorted: Vec<u32>,
    tmp: Vec<T>,
    row_nz: Vec<bool>,
}

impl<T> crate::memory::HeapBytes for PanelScratch<T> {
    fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes as b;
        b(&self.order) + b(&self.full) + b(&self.sorted) + b(&self.tmp) + b(&self.row_nz)
    }
}

impl<T> Default for PanelScratch<T> {
    fn default() -> Self {
        PanelScratch {
            order: Vec::new(),
            full: Vec::new(),
            sorted: Vec::new(),
            tmp: Vec::new(),
            row_nz: Vec::new(),
        }
    }
}

/// Finish one supernode's panel in its slot: rows `0..w` are the supernode's
/// own columns in elimination order, the `m` rows below are permuted into
/// ascending elimination order (`rows[i]` is the elimination index of panel
/// row `w + i` on entry, and ascending on exit), for an LDL^T factor the
/// `(p+1, p)` entry of each 2x2 pivot is cleared (that coupling lives in
/// `D`), and off-block rows without a nonzero in any column are dropped
/// from the panel and from `rows` (compacted in place, the tails of the
/// slots are left behind).
pub(crate) fn finish_panel<T: Scalar>(
    panel: &mut [T],
    rows: &mut [u32],
    w: usize,
    two_by_two: Option<&[bool]>,
    sc: &mut PanelScratch<T>,
) -> PanelOut {
    let m = rows.len();
    let ld = w + m;
    debug_assert!(panel.len() >= ld * w);
    let panel = &mut panel[..ld * w];
    let sorted = rows.windows(2).all(|p| p[0] < p[1]);
    if !sorted {
        sc.order.clear();
        sc.order.extend(0..m as u32);
        sc.order.sort_unstable_by_key(|&i| rows[i as usize]);
        sc.full.clear();
        sc.full.extend(0..w as u32);
        sc.full.extend(sc.order.iter().map(|&i| w as u32 + i));
        permute_panel_rows(panel, ld, w, &sc.full, &mut sc.tmp);
        sc.sorted.clear();
        sc.sorted.extend(sc.order.iter().map(|&i| rows[i as usize]));
        rows.copy_from_slice(&sc.sorted);
    }
    let zero = T::zero();
    if let Some(two_by_two) = two_by_two {
        for p in 0..w {
            if two_by_two[p] && p + 1 < w {
                panel[p * ld + p + 1] = zero;
            }
        }
    }
    // One pass over the strictly lower part: count the zero slots and find
    // the off-block rows without a nonzero in any column (the symmetrized
    // pattern of an unsymmetric matrix, relaxed amalgamation).
    let mut zeros = 0usize;
    let row_nz = &mut sc.row_nz;
    row_nz.clear();
    row_nz.resize(m, false);
    for p in 0..w {
        let col = &panel[p * ld..(p + 1) * ld];
        zeros += col[p + 1..w].iter().filter(|&&v| v == zero).count();
        for (i, &v) in col[w..].iter().enumerate() {
            if v == zero {
                zeros += 1;
            } else {
                row_nz[i] = true;
            }
        }
    }
    let m2 = row_nz.iter().filter(|&&b| b).count();
    let mut len = ld * w;
    if m2 < m {
        let ld2 = w + m2;
        let mut dst = 0;
        for p in 0..w {
            let src = p * ld;
            // Reads stay ahead of writes: `dst <= src` and the kept row `i`
            // lands at an offset no larger than its source.
            panel.copy_within(src..src + w, dst);
            let mut k = w;
            for (i, &keep) in row_nz.iter().enumerate() {
                if keep {
                    panel[dst + k] = panel[src + w + i];
                    k += 1;
                }
            }
            dst += ld2;
        }
        len = ld2 * w;
        let mut k = 0;
        for i in 0..m {
            if row_nz[i] {
                rows[k] = rows[i];
                k += 1;
            }
        }
        zeros -= (m - m2) * w;
    }
    PanelOut { m: m2, len, zeros }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_permutation_in_place() {
        // 3 x 2 panel, column-major.
        let mut panel = vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
        let mut tmp = Vec::new();
        permute_panel_rows(&mut panel, 3, 2, &[2, 0, 1], &mut tmp);
        assert_eq!(panel, vec![2.0, 0.0, 1.0, 12.0, 10.0, 11.0]);
    }

    #[test]
    fn arena_finish_closes_gaps_and_drops_empty_rows() {
        // Supernode 0: w=1, rows {1, 2}; supernode 1: w=1, row {2}; supernode 2: w=1.
        let shapes = [(1usize, 2usize), (1, 1), (1, 0)];
        let arena = PanelArena::<f64>::new(shapes.into_iter().map(|(w, m)| slot_sizes(w, m)));
        unsafe {
            arena.slot_mut(0).copy_from_slice(&[1.0, 0.0, 5.0]); // row 1 is empty
            arena.slot_mut(1).copy_from_slice(&[1.0, 6.0]);
            arena.slot_mut(2).copy_from_slice(&[1.0]);
            arena.rows_mut(0).copy_from_slice(&[2, 1]);
            arena.rows_mut(1).copy_from_slice(&[2]);
        }
        let outs = unsafe {
            [
                arena.finish_slot(0, 1, 2, None),
                arena.finish_slot(1, 1, 1, None),
                arena.finish_slot(2, 1, 0, None),
            ]
        };
        // Rows come in as {2, 1}: sorted to {1, 2}, and row 2 (the zero) is dropped.
        assert_eq!(outs[0].m, 1);
        assert_eq!(unsafe { arena.rows_mut(0) }[0], 1);
        assert_eq!(outs[0].len, 2);
        let mut outs = outs.into_iter();
        let (f, zeros) = arena.finish(3, [1usize, 1, 1].into_iter(), |_| outs.next().unwrap());
        assert_eq!(zeros, 0);
        assert_eq!(f.vals, vec![1.0, 5.0, 1.0, 6.0, 1.0]);
        assert_eq!(f.val_ptr, vec![0, 2, 4, 5]);
        assert_eq!(f.rows, vec![1, 2]);
        assert_eq!(f.row_ptr, vec![0, 1, 2, 2]);
    }
}
