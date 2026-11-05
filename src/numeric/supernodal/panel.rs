//! The supernodal panel form of a triangular factor: the one representation
//! the numeric drivers write and the solves read.
//!
//! A factor `L` (unit lower triangular, or a general lower triangular `U^T`)
//! is stored per supernode as one dense panel: `w` columns of the supernode,
//! the `w x w` diagonal block on top (only its lower triangle is meaningful,
//! the unit diagonal is implicit for a unit factor), the `m` off-block rows
//! below in ascending elimination order. The columns go in blocks of
//! [`SOLVE_DIAG_BLOCK`], each block column-major holding the rows from its
//! own first column down ([`packed_blocks`]): of the diagonal block's upper
//! triangle only the part inside each block's square is stored, and a panel
//! of one block is the full column-major `(w + m) x w` one. The row list is
//! shared by the panel's columns, so the index overhead is `m` words per
//! supernode rather than one per entry. All panels live in **one buffer**,
//! in supernode order (or, where fronts outgrew the analysis, the order
//! they were factored in) (the [`PanelArena`] the drivers factor into), so a
//! sweep streams through memory the way the elimination tree is walked, and
//! the solves run BLAS-3 style on the memory the drivers wrote. Compared
//! with a compressed-column factor with one `usize` index per entry (24
//! bytes per complex entry, and a second copy for the solve layout) this is
//! the whole factor at 16 bytes per complex entry, once.

use crate::constants::SOLVE_DIAG_BLOCK;
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
    /// Panel starts into `vals`, `ns` entries.
    pub val_off: Vec<usize>,
    /// The panels back to back, in supernode order or, where fronts outgrew
    /// the analysis (delayed pivots), in the order they were factored:
    /// supernode `s` is the [`packed_len`] entries from `vals[val_off[s]]`
    /// in [`packed_blocks`]; the stored entries above the diagonal of the
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

    /// The panel of supernode `s` ([`packed_blocks`]).
    #[inline]
    pub fn panel(&self, s: usize) -> &[T] {
        let (_, w, m) = self.shape(s);
        &self.vals[self.val_off[s]..self.val_off[s] + packed_len(w, m)]
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

/// Where block `b` of a panel of `w + m = wm` rows starts: `sum_{i < b} nb
/// (wm - i nb)` entries of the blocks before it (`nb` =
/// [`SOLVE_DIAG_BLOCK`]).
#[inline]
fn block_start(wm: usize, b: usize) -> usize {
    let nb = SOLVE_DIAG_BLOCK;
    b * nb * wm - nb * nb * (b * b.saturating_sub(1) / 2)
}

/// The column blocks of a panel of `w` columns and `m` off-block rows as it
/// is stored: `(jb, je, offset, ld)`, block `jb..je` the dense column-major
/// `ld x (je - jb)` block at `offset` from the panel's start, holding rows
/// `jb..w + m` of its columns (`ld = w + m - jb`). Nothing reads an entry
/// above a block's rows (the factorizations work on and below the
/// diagonal, the sweeps read the triangle in column blocks and the rows
/// below in products), so of the diagonal block's upper triangle only the
/// part inside each block's own square is stored.
pub(crate) fn packed_blocks(
    w: usize,
    m: usize,
) -> impl DoubleEndedIterator<Item = (usize, usize, usize, usize)> + ExactSizeIterator {
    let nb = SOLVE_DIAG_BLOCK;
    (0..w.div_ceil(nb)).map(move |b| {
        let (jb, je) = (b * nb, ((b + 1) * nb).min(w));
        (jb, je, block_start(w + m, b), w + m - jb)
    })
}

/// Entries of a panel of `w` columns and `m` off-block rows ([`packed_blocks`]).
pub(crate) fn packed_len(w: usize, m: usize) -> usize {
    packed_blocks(w, m)
        .map(|(jb, je, _, ld)| ld * (je - jb))
        .sum()
}

/// Whether a panel of `w` columns is one block, stored as the full
/// column-major `(w + m) x w` panel the factorization kernels work on.
#[inline]
pub(crate) fn one_block(w: usize) -> bool {
    w <= SOLVE_DIAG_BLOCK
}

/// Column `k` of a panel of `w` columns and `m` off-block rows:
/// `(start, jb)`, its stored rows `jb..w + m` from entry `start` on.
#[inline]
pub(crate) fn packed_col(w: usize, m: usize, k: usize) -> (usize, usize) {
    let b = k / SOLVE_DIAG_BLOCK;
    let jb = b * SOLVE_DIAG_BLOCK;
    (block_start(w + m, b) + (k - jb) * (w + m - jb), jb)
}

/// Store the full column-major `(w + m) x w` panel `full` packed
/// ([`packed_blocks`]) at `dst`.
///
/// # Safety
/// `dst` is valid for [`packed_len`]`(w, m)` writes and does not overlap
/// `full`.
pub(crate) unsafe fn pack_into<T: Copy>(full: &[T], w: usize, m: usize, dst: *mut T) {
    let ld = w + m;
    debug_assert!(full.len() >= ld * w);
    for (jb, je, off, ldb) in packed_blocks(w, m) {
        for k in jb..je {
            std::ptr::copy_nonoverlapping(
                full.as_ptr().add(k * ld + jb),
                dst.add(off + (k - jb) * ldb),
                ldb,
            );
        }
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
/// The slots are fixed by the analysis, the [`packed_len`] of each
/// supernode's symbolic shape, back to back in supernode order
/// ([`new`](Self::new)); or, where fronts may outgrow the analysis (an LU
/// with delayed pivots), [`lazy`](Self::lazy): each owner claims a slot of
/// the size its front has, from a buffer of fixed room. The slots then lie
/// in the order they were claimed, with no holes, so only the values the
/// factor holds are ever resident and closing the gaps moves every panel
/// down. Nothing is zeroed up front: a supernode of one column block
/// factors in its slot, zeroed when claimed ([`claim`](Self::claim)), a
/// wider one in a full panel of its own that it then stores packed
/// ([`place`](Self::place)), so every slot is written before it is read.
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
    /// supernode (see [`slot_sizes`]).
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
        // The values stay unset: every slot is written by its owner before
        // anyone reads it, and the length is set when they are ([`finish`]).
        vals.clear();
        vals.reserve(total);
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
            scratch: ScratchPool::new(),
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
        let at = PanelPtr(self.base.get().add(self.take(s, vals, rows)?));
        zero(at, vals);
        Some(std::slice::from_raw_parts_mut(at.get(), vals))
    }

    /// Store supernode `s`'s factored front, the column-major `ld x cols`
    /// `full`, in its slot with room for `rows` rows: its first `w` columns
    /// the panel, packed ([`pack_into`]), the columns past them (the pivots
    /// an LU front passes on) whole after it. A fixed slot, or one claimed
    /// now from a lazy arena's room (`false` when it is spent).
    ///
    /// # Safety
    /// Only the owner of `s`, once, before anybody reads the slot.
    pub unsafe fn place(
        &self,
        s: usize,
        full: &[T],
        ld: usize,
        w: usize,
        cols: usize,
        rows: usize,
    ) -> bool {
        let len = packed_len(w, ld - w);
        let Some(start) = self.take(s, len + (cols - w) * ld, rows) else {
            return false;
        };
        let at = self.base.get().add(start);
        pack_into(full, w, ld - w, at);
        std::ptr::copy_nonoverlapping(full.as_ptr().add(w * ld), at.add(len), (cols - w) * ld);
        true
    }

    /// The start of supernode `s`'s slot of `vals` values and `rows` rows:
    /// its fixed one, or claimed now from a lazy arena's room (`None` when
    /// it is spent).
    ///
    /// # Safety
    /// Only the owner of `s`, once.
    unsafe fn take(&self, s: usize, vals: usize, rows: usize) -> Option<usize> {
        if !self.lazy {
            debug_assert!(vals <= self.slot_len(s), "a front within its fixed slot");
            debug_assert!(rows <= self.row_slot_ptr[s + 1] - self.row_slot_ptr[s]);
            return Some(self.slots.get(s).0);
        }
        use std::sync::atomic::Ordering;
        let start = self.cursor.fetch_add(vals, Ordering::AcqRel);
        if start + vals > self.vals.capacity() {
            return None;
        }
        self.slots.set(s, (start, vals));
        self.lazy_rows.set(s, vec![0u32; rows]);
        Some(start)
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
    /// panel's row count and compacted length. The panels move down in the
    /// order of their slots to close the gaps of dropped rows and delayed
    /// pivots, so the factor's buffers are exact. Returns the factor and the
    /// zero-slot count.
    pub fn finish(
        mut self,
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
            debug_assert_eq!(out.len, packed_len(w, out.m));
            debug_assert!(out.len <= self.slot_len(s));
            kept.push((s, out.len, out.m));
            sn_col.push(sn_col.last().copied().unwrap_or(0) + w as u32);
            row_ptr.push(row_ptr.last().copied().unwrap_or(0) + out.m);
            zeros += out.zeros;
        }
        debug_assert_eq!(sn_col.last().copied(), Some(n as u32));
        // The values in the order of the slots, each panel moved down to the
        // end of the ones before it (reads stay ahead of writes).
        // SAFETY (the slot reads): every slot is finished and unshared.
        let start = |i: usize| unsafe { self.slots.get(kept[i].0) }.0;
        let mut order: Vec<usize> = (0..kept.len()).collect();
        if self.lazy {
            order.sort_unstable_by_key(|&i| start(i));
        }
        let mut val_off = vec![0usize; kept.len()];
        let (base, mut at) = (self.vals.as_mut_ptr(), 0usize);
        let mut moved = false;
        for &i in &order {
            let (src, len) = (start(i), kept[i].1);
            debug_assert!(at <= src);
            if at != src {
                // SAFETY: both ranges lie in the buffer's capacity, the source
                // holds the finished panel; `copy` handles the overlap.
                unsafe { std::ptr::copy(base.add(src), base.add(at), len) };
                moved = true;
            }
            val_off[i] = at;
            at += len;
        }
        let total = at;
        let total_rows = row_ptr.last().copied().unwrap_or(0);
        let (mut vals, mut rows) = (self.vals, self.rows);
        // SAFETY: the first `total` values are the panels written above,
        // every one written by its owner.
        unsafe { vals.set_len(total) };
        if (moved || total < vals.capacity()) && !self.reused {
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

/// Zero `len` values from `at`, across the calling pool when they are many.
///
/// # Safety
/// `at` is valid for `len` writes that nobody else makes meanwhile.
unsafe fn zero<T>(at: PanelPtr<T>, len: usize) {
    let chunk = crate::constants::ZERO_FILL_CHUNK;
    if len <= chunk {
        std::ptr::write_bytes(at.get(), 0, len);
    } else {
        crate::parallel::for_each(
            len.div_ceil(chunk),
            crate::parallel::Work::ns(f64::INFINITY),
            |c| {
                let c0 = c * chunk;
                // SAFETY: a chunk of the range, which the caller owns.
                unsafe { std::ptr::write_bytes(at.get().add(c0), 0, chunk.min(len - c0)) };
            },
        );
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

/// The arena slot `(values, rows)` of a supernode of `w` columns and `m`
/// off-block rows ([`packed_len`]): none for an empty supernode.
pub(crate) fn slot_sizes(w: usize, m: usize) -> (usize, usize) {
    if w == 0 {
        (0, 0)
    } else {
        (packed_len(w, m), m)
    }
}

/// One supernode's finished panel, as left in its arena slots: its
/// off-block row count `m` (the rows in the row slot, elimination indices,
/// ascending), the compacted panel length [`packed_len`]`(w, m)`, and the
/// zero-slot count.
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

/// Finish one supernode's panel of `w` columns in its slot
/// ([`packed_blocks`]): rows `0..w` are the supernode's own columns in
/// elimination order, the `m` rows below are permuted into ascending
/// elimination order (`rows[i]` is the elimination index of off-block row
/// `i` on entry, and ascending on exit), for an LDL^T factor the `(p+1, p)`
/// entry of each 2x2 pivot is cleared (that coupling lives in `D`), and
/// off-block rows without a nonzero in any column are dropped from the
/// panel and from `rows` (the panel stored again for the rows kept, the
/// tails of the slots are left behind).
pub(crate) fn finish_panel<T: Scalar>(
    panel: &mut [T],
    rows: &mut [u32],
    w: usize,
    two_by_two: Option<&[bool]>,
    sc: &mut PanelScratch<T>,
) -> PanelOut {
    let m = rows.len();
    debug_assert!(panel.len() >= packed_len(w, m));
    let panel = &mut panel[..packed_len(w, m)];
    // Column `k`'s off-block rows start `w - jb` into its stored rows.
    if !rows.windows(2).all(|p| p[0] < p[1]) {
        sc.order.clear();
        sc.order.extend(0..m as u32);
        sc.order.sort_unstable_by_key(|&i| rows[i as usize]);
        for k in 0..w {
            let (start, jb) = packed_col(w, m, k);
            let off = &mut panel[start + w - jb..start + w - jb + m];
            sc.tmp.clear();
            sc.tmp.extend_from_slice(off);
            for (i, &o) in sc.order.iter().enumerate() {
                off[i] = sc.tmp[o as usize];
            }
        }
        sc.sorted.clear();
        sc.sorted.extend(sc.order.iter().map(|&i| rows[i as usize]));
        rows.copy_from_slice(&sc.sorted);
    }
    let zero = T::zero();
    if let Some(two_by_two) = two_by_two {
        for p in 0..w {
            if two_by_two[p] && p + 1 < w {
                let (start, jb) = packed_col(w, m, p);
                panel[start + p + 1 - jb] = zero;
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
        let (start, jb) = packed_col(w, m, p);
        let col = &panel[start..start + w + m - jb];
        zeros += col[p + 1 - jb..w - jb]
            .iter()
            .filter(|&&v| v == zero)
            .count();
        for (i, &v) in col[w - jb..].iter().enumerate() {
            if v == zero {
                zeros += 1;
            } else {
                row_nz[i] = true;
            }
        }
    }
    let m2 = row_nz.iter().filter(|&&b| b).count();
    let mut len = packed_len(w, m);
    if m2 < m {
        // Column by column in order, each to an offset no larger than its
        // own and its kept rows to offsets no larger than theirs, so reads
        // stay ahead of writes.
        for k in 0..w {
            let (src, jb) = packed_col(w, m, k);
            let (dst, _) = packed_col(w, m2, k);
            panel.copy_within(src..src + w - jb, dst);
            let mut at = dst + w - jb;
            for (i, &keep) in row_nz.iter().enumerate() {
                if keep {
                    panel[at] = panel[src + w - jb + i];
                    at += 1;
                }
            }
        }
        len = packed_len(w, m2);
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
        assert_eq!(f.val_off, vec![0, 2, 4]);
        assert_eq!(f.rows, vec![1, 2]);
        assert_eq!(f.row_ptr, vec![0, 1, 2, 2]);
    }

    #[test]
    fn a_wide_panel_is_finished_in_its_column_blocks() {
        // Two column blocks; off-block rows {9, 5, 7}, the one of 5 empty.
        let (w, m) = (SOLVE_DIAG_BLOCK + 6, 3usize);
        let ld = w + m;
        let mut full = vec![0.0f64; ld * w];
        for k in 0..w {
            for i in k..ld {
                full[k * ld + i] = (1000 * k + i + 1) as f64;
            }
            full[k * ld + w + 1] = 0.0;
        }
        let mut panel = vec![0.0; packed_len(w, m)];
        unsafe { pack_into(&full, w, m, panel.as_mut_ptr()) };
        let mut rows = [9u32, 5, 7];
        let out = finish_panel(&mut panel, &mut rows, w, None, &mut PanelScratch::default());
        assert_eq!((out.m, out.len), (2, packed_len(w, 2)));
        assert_eq!(rows[..2], [7, 9]);
        for k in 0..w {
            let (start, jb) = packed_col(w, 2, k);
            for i in k..w {
                assert_eq!(panel[start + i - jb], full[k * ld + i]);
            }
            assert_eq!(panel[start + w - jb], full[k * ld + w + 2]);
            assert_eq!(panel[start + w - jb + 1], full[k * ld + w]);
        }
    }
}
