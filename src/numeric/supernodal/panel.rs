//! The supernodal panel form of a triangular factor: the one representation
//! the numeric drivers write and the solves read.
//!
//! A factor `L` (unit lower triangular, or a general lower triangular `U^T`)
//! is stored per supernode as one panel of `w` columns and `m` off-block
//! rows ([`Layout`]): the diagonal part in column blocks of
//! [`PANEL_BLOCK`](crate::constants::PANEL_BLOCK) columns, each block its
//! columns' rows from the block's first column to `w` (column-major, leading
//! dimension `w - jb`), so of the diagonal block's unused upper triangle only
//! the part inside each block's own square is stored; then the `m x w`
//! rectangle of the off-block rows of all columns (column-major, leading
//! dimension `m`), the rows in ascending elimination order. The row list is
//! shared by the panel's columns, so the index overhead is `m` words per
//! supernode rather than one per entry. All panels live in **one buffer**
//! (the [`PanelArena`] the drivers factor into), each at its own offset, in
//! supernode order (or, where fronts outgrew the analysis, the order they
//! started in): the factorization writes this layout in place and the
//! solves read it as it is, so a sweep streams through memory the way the
//! elimination tree is walked. A descendant's update reads its off-block
//! rows as one rectangle of uniform stride.

use crate::constants::PANEL_BLOCK;
use crate::numeric::supernodal::{Cells, PanelPtr, ScratchPool};
use crate::scalar::Scalar;

/// Where the entries of a panel of `w` columns and `m` off-block rows lie in
/// its slot (see the module docs): entry `(i, j)` of the panel (row `i` of
/// `0..w + m`, column `j` of `0..w`) is stored for `i >= jb`, `jb` the first
/// column of `j`'s block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub w: usize,
    pub m: usize,
}

/// A column-major run of a panel's entries: rows `rows` of columns `cols`,
/// the entry `(rows.start, cols.start)` at `off`, columns `ld` apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub off: usize,
    pub ld: usize,
    pub rows: std::ops::Range<usize>,
    pub cols: std::ops::Range<usize>,
}

/// One column of a [`Layout`] ([`Layout::col`]).
#[derive(Clone, Copy)]
pub(crate) struct ColAt {
    w: usize,
    diag: usize,
    rect: usize,
}

impl ColAt {
    /// The offset of the column's entry in row `i`.
    #[inline]
    pub(crate) fn at(&self, i: usize) -> usize {
        if i < self.w {
            self.diag + i
        } else {
            self.rect + i
        }
    }
}

impl Layout {
    #[inline]
    pub(crate) const fn new(w: usize, m: usize) -> Self {
        Layout { w, m }
    }

    /// Entries of the full diagonal blocks before block `k`:
    /// `sum_{i < k} nb (w - i nb)`.
    #[inline]
    fn before(&self, k: usize) -> usize {
        let nb = PANEL_BLOCK;
        k * nb * self.w - nb * nb * (k * k.saturating_sub(1) / 2)
    }

    /// Where the off-block rectangle starts: the entries of the diagonal
    /// blocks.
    #[inline]
    pub(crate) fn rect(&self) -> usize {
        let full = self.w / PANEL_BLOCK;
        let last = self.w - full * PANEL_BLOCK;
        self.before(full) + last * last
    }

    /// The panel's entries.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.rect() + self.m * self.w
    }

    /// The diagonal blocks: `(jb, je, offset, ld)`, block `jb..je` the
    /// column-major `(w - jb) x (je - jb)` block of rows `jb..w` at `offset`.
    pub(crate) fn blocks(
        &self,
    ) -> impl DoubleEndedIterator<Item = (usize, usize, usize, usize)> + ExactSizeIterator {
        let l = *self;
        (0..l.w.div_ceil(PANEL_BLOCK)).map(move |k| {
            let (jb, je) = (k * PANEL_BLOCK, ((k + 1) * PANEL_BLOCK).min(l.w));
            (jb, je, l.before(k), l.w - jb)
        })
    }

    /// The offset of entry `(i, j)`.
    #[inline]
    pub(crate) fn at(&self, i: usize, j: usize) -> usize {
        debug_assert!(j < self.w && i < self.w + self.m);
        if i < self.w {
            let k = j / PANEL_BLOCK;
            let jb = k * PANEL_BLOCK;
            debug_assert!(i >= jb, "entry ({i}, {j}) above its block");
            self.before(k) + (j - jb) * (self.w - jb) + i - jb
        } else {
            self.rect() + j * self.m + i - self.w
        }
    }

    /// Where column `j`'s entries lie: entry `(i, j)` at
    /// [`ColAt::at`]`(i)`, for `i` from `j`'s block's first column on.
    #[inline]
    pub(crate) fn col(&self, j: usize) -> ColAt {
        let k = j / PANEL_BLOCK;
        let jb = k * PANEL_BLOCK;
        ColAt {
            w: self.w,
            diag: self.before(k) + (j - jb) * (self.w - jb) - jb,
            rect: self.rect() + j * self.m - self.w,
        }
    }

    /// The column-major runs that hold rows `rows` of columns `cols`: in the
    /// diagonal part one per block the columns reach (rows from `rows.start`,
    /// at least the block's first column, to `w`), then one in the off-block
    /// rectangle.
    pub(crate) fn runs(
        &self,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
    ) -> impl Iterator<Item = Run> {
        let l = *self;
        let diag_end = rows.end.min(l.w);
        let first = cols.start / PANEL_BLOCK;
        let last = if cols.is_empty() {
            first
        } else {
            cols.end.div_ceil(PANEL_BLOCK)
        };
        let diag = (first..last).filter_map(move |k| {
            let c0 = cols.start.max(k * PANEL_BLOCK);
            let c1 = cols.end.min((k + 1) * PANEL_BLOCK);
            let r0 = rows.start.max(k * PANEL_BLOCK);
            (c0 < c1 && r0 < diag_end).then(|| Run {
                off: l.at(r0, c0),
                ld: l.w - k * PANEL_BLOCK,
                rows: r0..diag_end,
                cols: c0..c1,
            })
        });
        let r0 = rows.start.max(l.w);
        let rect = (r0 < rows.end && !cols.is_empty()).then(|| Run {
            off: l.at(r0, cols.start),
            ld: l.m,
            rows: r0..rows.end,
            cols: cols.clone(),
        });
        diag.chain(rect)
    }
}

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
    /// Panel starts into `vals`, `ns` entries.
    pub val_off: Vec<usize>,
    /// The panels, each at its offset ([`Layout`] of its `w` and `m` from
    /// `vals[val_off[s]]`), in supernode order or, where fronts outgrew the
    /// analysis (delayed pivots), in the order they started; a panel that
    /// dropped rows or pivots leaves the rest of its slot unused.
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
            val_off: Vec::new(),
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

    /// The layout of supernode `s`'s panel.
    #[inline]
    pub(crate) fn layout(&self, s: usize) -> Layout {
        let (_, w, m) = self.shape(s);
        Layout::new(w, m)
    }

    /// The panel of supernode `s` ([`layout`](Self::layout)).
    #[inline]
    pub fn panel(&self, s: usize) -> &[T] {
        &self.vals[self.val_off[s]..self.val_off[s] + self.layout(s).len()]
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
/// and one row slot per supernode of the analysis. Each slot is written by
/// the one task that owns its supernode (the drivers factor straight into
/// it), read by the tasks that update from it once it is published, and
/// finished in place ([`finish_slot`](Self::finish_slot)).
/// [`finish`](Self::finish) then closes the gaps left by dropped rows and
/// yields the [`PanelFactor`].
///
/// The slots are fixed by the analysis, the [`Layout`] of each supernode's
/// symbolic shape, back to back in supernode order and zeroed up front
/// ([`new`](Self::new)); or, where fronts may outgrow the analysis (an LU
/// with delayed pivots), [`lazy`](Self::lazy): each owner claims a slot of
/// the size its front has when it starts, from a buffer of fixed room,
/// zeroed then. The slots then lie in the order the fronts started, so only
/// the values the factor holds are ever resident. A panel stays where it was
/// factored: the factor takes the buffer as it is.
pub(crate) struct PanelArena<T> {
    /// Each supernode's value slot `(start, length)`.
    slots: Cells<(usize, usize)>,
    vals: Vec<T>,
    base: PanelPtr<T>,
    /// Row slot starts, `nsuper + 1` entries, and the rows (fixed slots).
    row_slot_ptr: Vec<usize>,
    rows: Vec<u32>,
    row_base: PanelPtr<u32>,
    /// Lazy: the next free value, and each supernode's own row slot.
    cursor: std::sync::atomic::AtomicUsize,
    lazy_rows: Cells<Vec<u32>>,
    lazy: bool,
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
        Self::new_in(PanelStorage::default(), slots, false)
    }

    /// [`new`](Self::new) in `storage`, the buffers of an earlier factor of
    /// this analysis: no allocation where their capacity suffices, and the
    /// capacity kept through [`finish`](Self::finish).
    pub fn reuse(storage: PanelStorage<T>, slots: impl Iterator<Item = (usize, usize)>) -> Self {
        Self::new_in(storage, slots, true)
    }

    fn new_in(
        storage: PanelStorage<T>,
        slots: impl Iterator<Item = (usize, usize)>,
        reused: bool,
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
        // The whole factor, often hundreds of MB: zero it across the calling
        // pool rather than on one thread (41 ms serial on a 465 MB factor).
        vals.clear();
        vals.reserve(total);
        crate::parallel::chunks_mut(
            &mut vals.spare_capacity_mut()[..total],
            crate::constants::ZERO_FILL_CHUNK,
            crate::parallel::Work::ns(f64::INFINITY),
            |_, chunk| {
                chunk.iter_mut().for_each(|v| {
                    v.write(T::zero());
                })
            },
        );
        // SAFETY: every element of the first `total` was written above.
        unsafe { vals.set_len(total) };
        rows.clear();
        rows.resize(row_slot_ptr.last().copied().unwrap_or(0), 0);
        let ns = slot_ptr.len() - 1;
        let slots = Cells::new(ns, (0usize, 0usize));
        for s in 0..ns {
            // SAFETY: not shared yet.
            unsafe { slots.set(s, (slot_ptr[s], slot_ptr[s + 1] - slot_ptr[s])) };
        }
        Self::with(slots, vals, row_slot_ptr, rows, false, reused)
    }

    /// Slots of `nsuper` supernodes claimed as their fronts start
    /// ([`claim`](Self::claim)), together at most `room` values.
    pub fn lazy(nsuper: usize, room: usize) -> Self {
        let mut vals = Vec::new();
        vals.reserve_exact(room);
        let row_slot_ptr = vec![0usize; nsuper + 1];
        Self::with(
            Cells::new(nsuper, (0usize, 0usize)),
            vals,
            row_slot_ptr,
            Vec::new(),
            true,
            false,
        )
    }

    fn with(
        slots: Cells<(usize, usize)>,
        mut vals: Vec<T>,
        row_slot_ptr: Vec<usize>,
        mut rows: Vec<u32>,
        lazy: bool,
        reused: bool,
    ) -> Self {
        let ns = row_slot_ptr.len() - 1;
        let base = PanelPtr(vals.as_mut_ptr());
        let row_base = PanelPtr(rows.as_mut_ptr());
        PanelArena {
            slots,
            vals,
            base,
            row_slot_ptr,
            rows,
            row_base,
            cursor: std::sync::atomic::AtomicUsize::new(0),
            lazy_rows: Cells::new_default(if lazy { ns } else { 0 }),
            lazy,
            reused,
        }
    }

    /// Entries of slot `s`.
    #[inline]
    pub fn slot_len(&self, s: usize) -> usize {
        // SAFETY: a slot is set before it is published.
        unsafe { self.slots.get(s) }.1
    }

    /// Supernode `s`'s slot for its owner to factor into: `vals` zeroed
    /// values and room for `rows` rows. A fixed slot is the analysis's and
    /// holds them; a lazy one is claimed now, `None` when the arena's room
    /// is spent.
    ///
    /// # Safety
    /// Only the owner of `s`, once, before anybody reads the slot.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn claim(&self, s: usize, vals: usize, rows: usize) -> Option<&mut [T]> {
        if !self.lazy {
            debug_assert!(vals <= self.slot_len(s), "a front within its fixed slot");
            debug_assert!(rows <= self.row_slot_ptr[s + 1] - self.row_slot_ptr[s]);
            return Some(&mut self.slot_mut(s)[..vals]);
        }
        use std::sync::atomic::Ordering;
        let start = self.cursor.fetch_add(vals, Ordering::AcqRel);
        if start + vals > self.vals.capacity() {
            return None;
        }
        self.slots.set(s, (start, vals));
        self.lazy_rows.set(s, vec![0u32; rows]);
        let at = PanelPtr(self.base.get().add(start));
        let chunk = crate::constants::ZERO_FILL_CHUNK;
        if vals <= chunk {
            std::ptr::write_bytes(at.get(), 0, vals);
        } else {
            crate::parallel::for_each(
                vals.div_ceil(chunk),
                crate::parallel::Work::ns(f64::INFINITY),
                |c| {
                    let c0 = c * chunk;
                    // SAFETY: the claimed range, in the buffer's capacity.
                    unsafe { std::ptr::write_bytes(at.get().add(c0), 0, chunk.min(vals - c0)) };
                },
            );
        }
        Some(self.slot_mut(s))
    }

    /// The slot of supernode `s` for writing.
    ///
    /// # Safety
    /// Only the owner of `s` may call this, and not while any reader holds a
    /// reference from [`slot`](Self::slot); in a lazy arena after
    /// [`claim`](Self::claim).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slot_mut(&self, s: usize) -> &mut [T] {
        let (start, len) = *self.slots.get(s);
        std::slice::from_raw_parts_mut(self.base.get().add(start), len)
    }

    /// The slot of supernode `s` for reading.
    ///
    /// # Safety
    /// The owner must have published the slot (all writes done) before any
    /// reader calls this, and must not write again while readers exist.
    #[inline]
    pub unsafe fn slot(&self, s: usize) -> &[T] {
        let (start, len) = *self.slots.get(s);
        std::slice::from_raw_parts(self.base.get().add(start), len)
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
        if self.lazy {
            return self.lazy_rows.get_mut(s);
        }
        let (r0, r1) = (self.row_slot_ptr[s], self.row_slot_ptr[s + 1]);
        std::slice::from_raw_parts_mut(self.row_base.get().add(r0), r1 - r0)
    }

    /// Finish supernode `s`'s panel of `w` columns and `m` off-block rows in
    /// its slots, the first `m` entries of its row slot filled
    /// ([`finish_panel`]), its buffers from `scratch`, the factorization's
    /// pool that every refactorization reuses.
    ///
    /// # Safety
    /// As [`slot_mut`](Self::slot_mut): only the owner of `s`, once.
    pub unsafe fn finish_slot(
        &self,
        s: usize,
        w: usize,
        m: usize,
        two_by_two: Option<&[bool]>,
        scratch: &ScratchPool<PanelScratch<T>>,
    ) -> PanelOut {
        let mut scratch = scratch.take();
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
    /// panel's row count and length. Every panel stays in its slot (a slot
    /// whose panel dropped rows or pivots keeps the rest unused); the rows
    /// close up. Returns the factor and the zero-slot count.
    pub fn finish(
        self,
        n: usize,
        ncols: impl Iterator<Item = usize>,
        mut outs: impl FnMut(usize) -> PanelOut,
    ) -> (PanelFactor<T>, usize) {
        let cap = ncols.size_hint().0 + 1;
        let mut sn_col: Vec<u32> = Vec::with_capacity(cap);
        let mut row_ptr = Vec::with_capacity(cap);
        sn_col.push(0);
        row_ptr.push(0usize);
        let mut zeros = 0usize;
        // Every kept panel: its supernode, its finished length and rows.
        let mut kept: Vec<(usize, usize, usize)> = Vec::with_capacity(cap);
        for (s, w) in ncols.enumerate() {
            if w == 0 {
                continue;
            }
            let out = outs(s);
            debug_assert_eq!(out.len, Layout::new(w, out.m).len());
            debug_assert!(out.len <= self.slot_len(s));
            kept.push((s, out.len, out.m));
            sn_col.push(sn_col.last().copied().unwrap_or(0) + w as u32);
            row_ptr.push(row_ptr.last().copied().unwrap_or(0) + out.m);
            zeros += out.zeros;
        }
        debug_assert_eq!(sn_col.last().copied(), Some(n as u32));
        // SAFETY (the slot reads): every slot is finished and unshared.
        let val_off: Vec<usize> = (kept.iter())
            .map(|&(s, _, _)| unsafe { self.slots.get(s) }.0)
            .collect();
        // The values the slots hold: a fixed arena's whole buffer, a lazy
        // one's claimed part.
        let total = if self.lazy {
            self.cursor.load(std::sync::atomic::Ordering::Acquire)
        } else {
            self.vals.len()
        };
        let total_rows = row_ptr.last().copied().unwrap_or(0);
        let (mut vals, mut rows) = (self.vals, self.rows);
        // SAFETY: the first `total` values are the slots, every one zeroed
        // when made or claimed.
        unsafe { vals.set_len(total) };
        if total < vals.capacity() && !self.reused {
            release_room(&mut vals);
        }
        if self.lazy {
            rows = Vec::with_capacity(total_rows);
            for &(s, _, m) in &kept {
                // SAFETY: the finished supernode's own rows.
                rows.extend_from_slice(&unsafe { self.lazy_rows.get(s) }[..m]);
            }
        } else {
            let mut rdst = 0;
            for &(s, _, m) in &kept {
                let rsrc = self.row_slot_ptr[s];
                if rdst != rsrc {
                    rows.copy_within(rsrc..rsrc + m, rdst);
                }
                rdst += m;
            }
            rows.truncate(total_rows);
            if !self.reused {
                rows.shrink_to_fit();
            }
        }
        // Exact, as the plan keeps it (the memory plan counts it so).
        row_ptr.shrink_to_fit();
        (
            PanelFactor {
                n,
                sn_col,
                rows,
                row_ptr,
                val_off,
                vals,
            },
            zeros,
        )
    }
}

/// Give back the room of `v` past its values where that keeps the buffer
/// in place: a shrink by less than half. Allocators move a block they shrink
/// by more (mimalloc copies it), a second copy of the factor at its peak;
/// such room is a lazy arena's allowance, which was never touched and holds
/// no resident memory.
pub(crate) fn release_room<T>(v: &mut Vec<T>) {
    if 2 * v.len() >= v.capacity() {
        v.shrink_to_fit();
    }
}

/// The arena slot `(values, rows)` of a supernode whose panel has `w`
/// columns and `m` off-block rows ([`Layout`]): none for an empty supernode.
pub(crate) fn slot_sizes(w: usize, m: usize) -> (usize, usize) {
    if w == 0 {
        (0, 0)
    } else {
        (Layout::new(w, m).len(), m)
    }
}

/// One supernode's finished panel, as left in its arena slots: its
/// off-block row count `m` (the rows in the row slot, elimination indices,
/// ascending), the panel's length (its [`Layout`]), and the zero-slot
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
    sorted: Vec<u32>,
    tmp: Vec<T>,
    row_nz: Vec<bool>,
}

impl<T> crate::memory::HeapBytes for PanelScratch<T> {
    fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes as b;
        b(&self.order) + b(&self.sorted) + b(&self.tmp) + b(&self.row_nz)
    }
}

impl<T> Default for PanelScratch<T> {
    fn default() -> Self {
        PanelScratch {
            order: Vec::new(),
            sorted: Vec::new(),
            tmp: Vec::new(),
            row_nz: Vec::new(),
        }
    }
}

/// Finish one supernode's panel ([`Layout`] of `w` columns and `rows.len()`
/// off-block rows) in its slot: the off-block rows are permuted into
/// ascending elimination order (`rows[i]` is the elimination index of
/// off-block row `i` on entry, and ascending on exit), for an LDL^T factor
/// the `(p+1, p)` entry of each 2x2 pivot is cleared (that coupling lives in
/// `D`), and off-block rows without a nonzero in any column are dropped
/// from the rectangle and from `rows` (compacted in place, the tail of the
/// slot is left behind).
pub(crate) fn finish_panel<T: Scalar>(
    panel: &mut [T],
    rows: &mut [u32],
    w: usize,
    two_by_two: Option<&[bool]>,
    sc: &mut PanelScratch<T>,
) -> PanelOut {
    let m = rows.len();
    let lay = Layout::new(w, m);
    debug_assert!(panel.len() >= lay.len());
    let (diag, rect) = panel[..lay.len()].split_at_mut(lay.rect());
    if !rows.windows(2).all(|p| p[0] < p[1]) {
        sc.order.clear();
        sc.order.extend(0..m as u32);
        sc.order.sort_unstable_by_key(|&i| rows[i as usize]);
        permute_panel_rows(rect, m, w, &sc.order, &mut sc.tmp);
        sc.sorted.clear();
        sc.sorted.extend(sc.order.iter().map(|&i| rows[i as usize]));
        rows.copy_from_slice(&sc.sorted);
    }
    let zero = T::zero();
    if let Some(two_by_two) = two_by_two {
        for p in 0..w {
            if two_by_two[p] && p + 1 < w {
                diag[lay.at(p + 1, p)] = zero;
            }
        }
    }
    // One pass over the strictly lower part: count the zero slots and find
    // the off-block rows without a nonzero in any column (the symmetrized
    // pattern of an unsymmetric matrix, relaxed amalgamation).
    let mut zeros = 0usize;
    for p in 0..w {
        let at = lay.at(p, p);
        zeros += diag[at + 1..at + w - p]
            .iter()
            .filter(|&&v| v == zero)
            .count();
    }
    let row_nz = &mut sc.row_nz;
    row_nz.clear();
    row_nz.resize(m, false);
    for col in rect.chunks_exact(m.max(1)).take(if m == 0 { 0 } else { w }) {
        for (i, &v) in col.iter().enumerate() {
            if v == zero {
                zeros += 1;
            } else {
                row_nz[i] = true;
            }
        }
    }
    let m2 = row_nz.iter().filter(|&&b| b).count();
    if m2 < m {
        // Reads stay ahead of writes: the kept row `i` of column `p` lands
        // at an offset no larger than its source.
        let mut dst = 0;
        for p in 0..w {
            for i in 0..m {
                if row_nz[i] {
                    rect[dst] = rect[p * m + i];
                    dst += 1;
                }
            }
        }
        let mut k = 0;
        for i in 0..m {
            if row_nz[i] {
                rows[k] = rows[i];
                k += 1;
            }
        }
        zeros -= (m - m2) * w;
    }
    PanelOut {
        m: m2,
        len: Layout::new(w, m2).len(),
        zeros,
    }
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
    fn layout_places_every_entry_once_and_runs_cover_them() {
        for (w, m) in [(1, 0), (1, 3), (63, 5), (64, 0), (65, 7), (200, 13)] {
            let lay = Layout::new(w, m);
            let mut seen = vec![false; lay.len()];
            for j in 0..w {
                let jb = j / PANEL_BLOCK * PANEL_BLOCK;
                for i in jb..w + m {
                    let at = lay.at(i, j);
                    assert!(!std::mem::replace(&mut seen[at], true), "({i}, {j}) twice");
                }
            }
            assert!(seen.iter().all(|&b| b), "w={w} m={m}: a slot of no entry");
            for j in 0..w {
                let col = lay.col(j);
                for i in j / PANEL_BLOCK * PANEL_BLOCK..w + m {
                    assert_eq!(col.at(i), lay.at(i, j));
                }
            }
            let (r0, c0) = (w / 3, w / 4);
            let mut covered = 0;
            for run in lay.runs(r0.max(c0 / PANEL_BLOCK * PANEL_BLOCK)..w + m, c0..w) {
                for j in run.cols.clone() {
                    for i in run.rows.clone() {
                        let off = run.off + (j - run.cols.start) * run.ld + i - run.rows.start;
                        assert_eq!(off, lay.at(i, j));
                        covered += 1;
                    }
                }
            }
            let want: usize = (c0..w)
                .map(|j| {
                    w + m
                        - r0.max(c0 / PANEL_BLOCK * PANEL_BLOCK)
                            .max(j / PANEL_BLOCK * PANEL_BLOCK)
                })
                .sum();
            assert_eq!(covered, want, "w={w} m={m}");
        }
    }

    #[test]
    fn arena_finish_keeps_slots_and_drops_empty_rows() {
        // Supernode 0: w=1, rows {1, 2}; supernode 1: w=1, row {2}; supernode 2: w=1.
        let shapes = [(1usize, 2usize), (1, 1), (1, 0)];
        let arena = PanelArena::<f64>::new(shapes.into_iter().map(|(w, m)| slot_sizes(w, m)));
        let sc = ScratchPool::new();
        unsafe {
            arena.slot_mut(0).copy_from_slice(&[1.0, 0.0, 5.0]); // row 1 is empty
            arena.slot_mut(1).copy_from_slice(&[1.0, 6.0]);
            arena.slot_mut(2).copy_from_slice(&[1.0]);
            arena.rows_mut(0).copy_from_slice(&[2, 1]);
            arena.rows_mut(1).copy_from_slice(&[2]);
        }
        let outs = unsafe {
            [
                arena.finish_slot(0, 1, 2, None, &sc),
                arena.finish_slot(1, 1, 1, None, &sc),
                arena.finish_slot(2, 1, 0, None, &sc),
            ]
        };
        // Rows come in as {2, 1}: sorted to {1, 2}, and row 2 (the zero) is dropped.
        assert_eq!(outs[0].m, 1);
        assert_eq!(unsafe { arena.rows_mut(0) }[0], 1);
        assert_eq!(outs[0].len, 2);
        let mut outs = outs.into_iter();
        let (f, zeros) = arena.finish(3, [1usize, 1, 1].into_iter(), |_| outs.next().unwrap());
        assert_eq!(zeros, 0);
        // The panels stay in their slots; the dropped row leaves a hole.
        assert_eq!(f.vals, vec![1.0, 5.0, 0.0, 1.0, 6.0, 1.0]);
        assert_eq!(f.val_off, vec![0, 3, 5]);
        assert_eq!(f.rows, vec![1, 2]);
        assert_eq!(f.row_ptr, vec![0, 1, 2, 2]);
    }
}
