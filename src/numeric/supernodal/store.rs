//! Per-supernode storage of the left-looking factorizations: the per-index
//! cells of the emit state, the raw panel pointer for disjoint parallel
//! writes, and the scratch pool the node kernels borrow from.

use crate::constants::SCRATCH_KEEP;
use crate::memory::HeapBytes;

pub(crate) struct PanelPtr<T>(pub *mut T);
impl<T> Clone for PanelPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for PanelPtr<T> {}
// SAFETY: the pointer is only dereferenced on disjoint, caller-partitioned cells.
unsafe impl<T> Send for PanelPtr<T> {}
unsafe impl<T> Sync for PanelPtr<T> {}
impl<T> PanelPtr<T> {
    /// Extract the raw pointer. Taking `self` by value forces a closure to
    /// capture the whole (Send+Sync) wrapper rather than disjoint-capturing the
    /// bare field.
    #[inline]
    pub fn get(self) -> *mut T {
        self.0
    }
}

pub(crate) struct Cells<V>(Vec<std::cell::UnsafeCell<V>>);

// SAFETY: each index is written by exactly one owner, and every read of it
// follows a happens-before barrier of the caller (a subtree join or a
// refcount Acquire-Release).
unsafe impl<V: Send> Sync for Cells<V> {}

impl<V> Default for Cells<V> {
    fn default() -> Self {
        Cells(Vec::new())
    }
}

impl<V: Default> Cells<V> {
    /// `n` default-initialized cells (for payloads without a cheap `Clone`).
    pub fn new_default(n: usize) -> Self {
        Cells(
            (0..n)
                .map(|_| std::cell::UnsafeCell::new(V::default()))
                .collect(),
        )
    }
}

impl<V: Clone> Cells<V> {
    pub fn new(n: usize, init: V) -> Self {
        Cells(
            (0..n)
                .map(|_| std::cell::UnsafeCell::new(init.clone()))
                .collect(),
        )
    }
}

impl<V> Cells<V> {
    /// The number of cells.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// SAFETY: `i` is this caller's exclusively owned index.
    #[inline]
    pub unsafe fn set(&self, i: usize, v: V) {
        *self.0[i].get() = v;
    }
    /// SAFETY: the write to `i` happened-before this read.
    #[inline]
    pub unsafe fn get(&self, i: usize) -> &V {
        &*self.0[i].get()
    }
    /// SAFETY: as [`set`](Self::set) - exclusive owner, e.g. for in-place take.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub unsafe fn get_mut(&self, i: usize) -> &mut V {
        &mut *self.0[i].get()
    }
}

impl<V> Cells<V> {
    /// The cells `r` as one slice (`UnsafeCell<V>` is laid out as `V`).
    ///
    /// # Safety
    /// Every write to `r` happened-before this read, and none follows while
    /// the slice lives.
    #[inline]
    pub unsafe fn slice(&self, r: std::ops::Range<usize>) -> &[V] {
        std::slice::from_raw_parts(self.0[r.clone()].as_ptr() as *const V, r.len())
    }
}

pub(crate) struct ScratchPool<S>(std::sync::Mutex<Vec<S>>);

impl<S: Default + HeapBytes> ScratchPool<S> {
    pub fn new() -> Self {
        ScratchPool(std::sync::Mutex::new(Vec::new()))
    }

    /// A scratch object for the caller until the guard drops.
    pub fn take(&self) -> Lent<'_, S> {
        let s = self
            .0
            .lock()
            .ok()
            .and_then(|mut v| v.pop())
            .unwrap_or_default();
        Lent {
            pool: self,
            s: std::mem::ManuallyDrop::new(s),
        }
    }
}

pub(crate) struct Lent<'p, S: HeapBytes> {
    pool: &'p ScratchPool<S>,
    s: std::mem::ManuallyDrop<S>,
}

impl<S: HeapBytes> std::ops::Deref for Lent<'_, S> {
    type Target = S;
    fn deref(&self) -> &S {
        &self.s
    }
}

impl<S: HeapBytes> std::ops::DerefMut for Lent<'_, S> {
    fn deref_mut(&mut self) -> &mut S {
        &mut self.s
    }
}

impl<S: HeapBytes> Drop for Lent<'_, S> {
    fn drop(&mut self) {
        // SAFETY: the object is taken once, here, and never touched again.
        let s = unsafe { std::mem::ManuallyDrop::take(&mut self.s) };
        if s.heap_bytes() > SCRATCH_KEEP {
            return;
        }
        if let Ok(mut v) = self.pool.0.lock() {
            v.push(s);
        }
    }
}
