//! The workers every parallel path of rslab runs on.
//!
//! By default that is the current rayon pool: the one a factorization's
//! [`Threads`](crate::Threads) policy builds, or the caller's own under
//! [`Threads::Ambient`](crate::Threads::Ambient). A host that keeps workers
//! of its own lends them instead ([`install`]): every parallel section of
//! the factorizations and solves called inside then runs on them, the
//! `Threads` policies defer to them, and no rayon pool is entered. So a
//! solver that evaluates its model on the same workers between its
//! factorizations has one set of threads for both, kept awake across the
//! short serial stretches, instead of two pools competing for the cores.
//!
//! Every parallel site goes through the few helpers here. On rayon they are
//! the rayon idioms they replace; on a lent pool they are its sections
//! ([`Pool::run`], [`Pool::join`], [`Pool::scope`]), and a section opened
//! inside one runs on the same pool, which decides whether it nests (a pool
//! that steals work runs it alongside). Every result is the same either
//! way: no section's arithmetic depends on how its pieces are scheduled.

use std::cell::RefCell;
use std::sync::Arc;

/// Workers a host lends rslab.
pub trait Pool: Send + Sync {
    /// Number of workers.
    fn threads(&self) -> usize;
    /// Run `f(i)` for every `i` in `0..n`, returning when all ran.
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync));
    /// Run `f` where handing work to the pool is cheapest; the default runs
    /// it here.
    fn enter(&self, f: &mut (dyn FnMut() + Send)) {
        f()
    }
    /// Run `a` and `b`, side by side where the pool can, returning when
    /// both ran. The default runs one after the other.
    fn join(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        a();
        b();
    }
    /// Run `body` with a [`Spawn`] its tasks start tasks through, returning
    /// once every task ran. The default runs the tasks one after the other
    /// after `body`.
    fn scope<'s>(&self, body: &mut (dyn FnMut(&dyn Spawn<'s>) + Send)) {
        let queue = Serial(std::sync::Mutex::new(std::collections::VecDeque::new()));
        body(&queue);
        loop {
            let task = lock(&queue.0).pop_front();
            match task {
                Some(task) => task(&queue),
                None => return,
            }
        }
    }
}

/// A task of a scope (see [`Pool::scope`]).
pub type Task<'s> = Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>;

/// What a scope's tasks are started through (see [`Pool::scope`]).
pub trait Spawn<'s>: Sync {
    /// Start `task`; its scope returns once it ran.
    fn spawn(&self, task: Task<'s>);
}

/// The tasks of a scope run one after the other.
struct Serial<'s>(std::sync::Mutex<std::collections::VecDeque<Task<'s>>>);

impl<'s> Spawn<'s> for Serial<'s> {
    fn spawn(&self, task: Task<'s>) {
        lock(&self.0).push_back(task);
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<dyn Pool>>> = const { RefCell::new(None) };
}

/// Run `f` with `pool` lent: the factorizations and solves `f` calls run
/// their parallel sections on it. The previous one comes back when `f`
/// returns (or unwinds).
pub fn install<R>(pool: Arc<dyn Pool>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Arc<dyn Pool>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let p = self.0.take();
            CURRENT.with(|c| *c.borrow_mut() = p);
        }
    }
    let _restore = Restore(CURRENT.with(|c| c.replace(Some(pool))));
    f()
}

/// The lent pool, if any.
pub(crate) fn lent() -> Option<Arc<dyn Pool>> {
    CURRENT.with(|c| c.borrow().clone())
}

/// `f` with `pool` lent on this thread, unless it is: what `f` opens runs
/// on the pool too.
fn here<R>(pool: &Arc<dyn Pool>, f: impl FnOnce() -> R) -> R {
    if CURRENT.with(|c| c.borrow().is_some()) {
        f()
    } else {
        install(pool.clone(), f)
    }
}

/// `f(i)` for `i` in `0..n` on the lent pool, each piece with the pool lent
/// on the thread that runs it, so what it opens runs there too.
fn on_pool(pool: &Arc<dyn Pool>, n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n == 1 {
        return f(0);
    }
    pool.run(n, &|i| here(pool, || f(i)));
}

/// The workers of the current section: the lent pool's, or the current
/// rayon pool's.
pub fn threads() -> usize {
    match lent() {
        Some(p) => p.threads().max(1),
        None => rayon::current_num_threads().max(1),
    }
}

/// Run `f` where a parallel section is cheapest to start: on a worker of the
/// pool (a rayon section injected from outside costs a wake per section, one
/// started on a worker almost nothing).
pub(crate) fn enter<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match lent() {
        Some(p) => {
            let (mut f, mut r) = (Some(f), None);
            p.enter(&mut || {
                if let Some(f) = f.take() {
                    r = Some(install(p.clone(), f));
                }
            });
            match r {
                Some(r) => r,
                None => unreachable!("a pool runs what it enters"),
            }
        }
        None if rayon::current_thread_index().is_none() && rayon::current_num_threads() > 1 => {
            rayon::join(f, || ()).0
        }
        None => f(),
    }
}

/// Indices per piece for `n` items, at least `min_len`: about four pieces
/// per worker.
fn piece_len(n: usize, min_len: usize, threads: usize) -> usize {
    n.div_ceil(crate::constants::PIECES_PER_WORKER * threads)
        .max(min_len)
        .max(1)
}

/// `f(i)` for every `i` in `0..n`, `min_len` or more consecutive indices
/// per piece.
pub(crate) fn for_each(n: usize, min_len: usize, f: impl Fn(usize) + Sync + Send) {
    match lent() {
        Some(p) => {
            let len = piece_len(n, min_len, p.threads().max(1));
            on_pool(&p, n.div_ceil(len), &|k| {
                (k * len..((k + 1) * len).min(n)).for_each(&f);
            });
        }
        None => {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .with_min_len(min_len.max(1))
                .for_each(f);
        }
    }
}

/// `f(k, chunk)` for every chunk of `size` entries of `v` (the last one
/// shorter), `k` its index.
pub(crate) fn chunks_mut<T: Send>(
    v: &mut [T],
    size: usize,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    let size = size.max(1);
    match lent() {
        Some(p) => {
            let (len, n) = (v.len(), v.len().div_ceil(size));
            let base = SendPtr(v.as_mut_ptr());
            let per = piece_len(n, 1, p.threads().max(1));
            on_pool(&p, n.div_ceil(per), &|q| {
                for k in q * per..((q + 1) * per).min(n) {
                    let (a, b) = (k * size, ((k + 1) * size).min(len));
                    // SAFETY: the chunks are disjoint ranges of `v`, which
                    // outlives the section.
                    let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(a), b - a) };
                    f(k, chunk);
                }
            });
        }
        None => {
            use rayon::prelude::*;
            v.par_chunks_mut(size)
                .enumerate()
                .for_each(|(k, c)| f(k, c));
        }
    }
}

/// `f(i, &items[i], &mut scratch[i])` for every item and its scratch.
pub(crate) fn zip_mut<A: Sync, S: Send>(
    items: &[A],
    scratch: &mut [S],
    f: impl Fn(usize, &A, &mut S) + Sync + Send,
) {
    let n = items.len().min(scratch.len());
    match lent() {
        Some(p) => {
            let base = SendPtr(scratch.as_mut_ptr());
            on_pool(&p, n, &|i| {
                // SAFETY: piece `i` alone holds `scratch[i]`.
                f(i, &items[i], unsafe { &mut *base.get().add(i) })
            });
        }
        None => {
            use rayon::prelude::*;
            items
                .par_iter()
                .zip(scratch.par_iter_mut())
                .enumerate()
                .for_each(|(i, (a, s))| f(i, a, s));
        }
    }
}

/// `f(i)` for `i` in `0..n`, in order, `min_len` or more per piece.
pub(crate) fn map<R: Send>(
    n: usize,
    min_len: usize,
    f: impl Fn(usize) -> R + Sync + Send,
) -> Vec<R> {
    match lent() {
        Some(p) => {
            let mut out: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(n);
            // SAFETY: every slot is written below before it is read.
            unsafe { out.set_len(n) };
            let base = SendPtr(out.as_mut_ptr());
            let len = piece_len(n, min_len, p.threads().max(1));
            on_pool(&p, n.div_ceil(len), &|k| {
                for i in k * len..((k + 1) * len).min(n) {
                    // SAFETY: piece `k` alone writes slot `i`.
                    unsafe { (*base.get().add(i)).write(f(i)) };
                }
            });
            // SAFETY: all `n` slots written (a panic unwinds past this).
            unsafe { std::mem::transmute::<Vec<std::mem::MaybeUninit<R>>, Vec<R>>(out) }
        }
        None => {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .with_min_len(min_len.max(1))
                .map(f)
                .collect()
        }
    }
}

/// `f(item)` for every item of `items`, consumed, in order.
pub(crate) fn map_owned<A: Send, R: Send>(
    items: Vec<A>,
    min_len: usize,
    f: impl Fn(A) -> R + Sync + Send,
) -> Vec<R> {
    match lent() {
        Some(_) => {
            let slots: Vec<std::sync::Mutex<Option<A>>> = items
                .into_iter()
                .map(|a| std::sync::Mutex::new(Some(a)))
                .collect();
            map(slots.len(), min_len, |i| match lock(&slots[i]).take() {
                Some(a) => f(a),
                None => unreachable!("every item is taken once"),
            })
        }
        None => {
            use rayon::prelude::*;
            items
                .into_par_iter()
                .with_min_len(min_len.max(1))
                .map(f)
                .collect()
        }
    }
}

/// [`map`] with a state per piece, made by `init` (a rayon piece is a
/// thread's run of items).
pub(crate) fn map_init<S, R: Send>(
    n: usize,
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize) -> R + Sync + Send,
) -> Vec<R> {
    match lent() {
        Some(p) => {
            let len = piece_len(n, 1, p.threads().max(1));
            let pieces: Vec<Vec<R>> = map(n.div_ceil(len), 1, |k| {
                let mut s = init();
                (k * len..((k + 1) * len).min(n))
                    .map(|i| f(&mut s, i))
                    .collect()
            });
            pieces.into_iter().flatten().collect()
        }
        None => {
            use rayon::prelude::*;
            (0..n).into_par_iter().map_init(init, f).collect()
        }
    }
}

/// `f(item)` for every item of `items`, consumed, `min_len` or more per
/// piece.
pub(crate) fn for_each_owned<A: Send>(items: Vec<A>, min_len: usize, f: impl Fn(A) + Sync + Send) {
    match lent() {
        Some(_) => {
            map_owned(items, min_len, f);
        }
        None => {
            use rayon::prelude::*;
            items
                .into_par_iter()
                .with_min_len(min_len.max(1))
                .for_each(f);
        }
    }
}

/// `v` sorted by `key`, unstably: on rayon in parallel, on a lent pool on
/// this thread (an analysis sorts once).
pub(crate) fn sort_unstable_by_key<T: Send, K: Ord>(
    v: &mut [T],
    key: impl Fn(&T) -> K + Sync + Send,
) {
    match lent() {
        Some(_) => v.sort_unstable_by_key(key),
        None => {
            use rayon::prelude::*;
            v.par_sort_unstable_by_key(key)
        }
    }
}

/// Whether `f(i)` holds for some `i` in `0..n`, `min_len` or more per piece.
pub(crate) fn any(n: usize, min_len: usize, f: impl Fn(usize) -> bool + Sync + Send) -> bool {
    match lent() {
        Some(_) => {
            let found = std::sync::atomic::AtomicBool::new(false);
            for_each(n, min_len, |i| {
                use std::sync::atomic::Ordering::Relaxed;
                if !found.load(Relaxed) && f(i) {
                    found.store(true, Relaxed);
                }
            });
            found.into_inner()
        }
        None => {
            use rayon::prelude::*;
            (0..n).into_par_iter().with_min_len(min_len.max(1)).any(f)
        }
    }
}

/// `a()` and `b()`, side by side.
pub(crate) fn join<A: Send, B: Send>(
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    match lent() {
        Some(p) => {
            let (mut a, mut b) = (Some(a), Some(b));
            let (mut ra, mut rb) = (None, None);
            p.join(
                &mut || {
                    if let Some(f) = a.take() {
                        ra = Some(here(&p, f));
                    }
                },
                &mut || {
                    if let Some(f) = b.take() {
                        rb = Some(here(&p, f));
                    }
                },
            );
            match (ra, rb) {
                (Some(ra), Some(rb)) => (ra, rb),
                _ => unreachable!("a pool runs every piece"),
            }
        }
        None => rayon::join(a, b),
    }
}

/// A section whose tasks start tasks of their own ([`scope`]).
pub(crate) struct Scope<'a, 's>(ScopeOn<'a, 's>);

enum ScopeOn<'a, 's> {
    Rayon(&'a rayon::Scope<'s>),
    Lent(&'a dyn Spawn<'s>, &'a Arc<dyn Pool>),
}

impl<'a, 's> Scope<'a, 's> {
    /// Start `f` as a task of the scope; the scope returns once it ran.
    pub(crate) fn spawn(&self, f: impl for<'b> FnOnce(&Scope<'b, 's>) + Send + 's) {
        match self.0 {
            ScopeOn::Rayon(sc) => sc.spawn(move |sc| f(&Scope(ScopeOn::Rayon(sc)))),
            ScopeOn::Lent(sp, pool) => {
                let pool = pool.clone();
                sp.spawn(Box::new(move |sp: &dyn Spawn<'s>| {
                    here(&pool, || f(&Scope(ScopeOn::Lent(sp, &pool))))
                }));
            }
        }
    }
}

/// Run `body` with a scope its tasks are started in, returning once every
/// task ran: rayon's `scope`, or the lent pool's.
pub(crate) fn scope<'s, R: Send>(body: impl for<'a> FnOnce(&Scope<'a, 's>) -> R + Send + 's) -> R {
    match lent() {
        Some(p) => {
            let (mut body, mut r) = (Some(body), None);
            p.scope(&mut |sp: &dyn Spawn<'s>| {
                if let Some(f) = body.take() {
                    r = Some(f(&Scope(ScopeOn::Lent(sp, &p))));
                }
            });
            match r {
                Some(r) => r,
                None => unreachable!("a pool runs the body of its scope"),
            }
        }
        None => rayon::scope(|sc| body(&Scope(ScopeOn::Rayon(sc)))),
    }
}

/// `m` locked: a lock is poisoned only by a panic, which its section
/// carries on.
fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A pointer the pieces of a section share, each to its own part.
struct SendPtr<T>(*mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
// SAFETY: the sections hand every piece a part no other piece touches.
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}
