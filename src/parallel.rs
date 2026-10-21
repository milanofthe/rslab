//! The workers every parallel path of rslab runs on, and how wide a section
//! goes on them: [`rspool`]'s, whose items are re-exported here.
//!
//! By default that is the current rayon pool: the one a factorization's
//! [`Threads`](crate::Threads) policy builds, or the caller's own under
//! [`Threads::Ambient`](crate::Threads::Ambient). A host that keeps workers
//! of its own lends them instead ([`install`]): every parallel section of
//! the factorizations and solves called inside then runs on them, the
//! `Threads` policies defer to them, and no rayon pool is entered. A host
//! that lends the same workers to other libraries (a model's evaluations
//! between the factorizations) has one set of threads, one cost model and
//! one rate learning for all of them.
//!
//! A section runs on as many threads as its work is worth, one included:
//! its work in ns (a count of work at a [`Rate`] learned from the serial
//! runs) against what a section costs the pool on `k` threads (its
//! [`Dispatch`]). Every helper here decides so ([`Work::go`] at
//! [`Rayon`]). Every result is the same however a section runs: no
//! section's arithmetic depends on how its pieces are scheduled or on how
//! many threads run them.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

pub use rspool::rayon::{
    dispatch, kept, pool_dispatch as rayon_dispatch, prepare, run, run_rated, run_watched,
    run_width, threads, width, width_on, Rayon,
};
pub use rspool::{
    as_piece, current, install, install_lent, share_of, Dispatch, Go, Lent, Piece, Pool, Rate,
    Serial, Spawn, Task, Timing, Work,
};

use crate::constants::{COMPLEX_OPS, PIECES_PER_WORKER};

/// The lent pool, if any.
pub(crate) fn lent() -> Option<Arc<dyn Pool>> {
    current().map(|l| l.pool)
}

/// Items per piece for `n` items on `k` threads: about
/// [`PIECES_PER_WORKER`] pieces per thread.
fn piece_len(n: usize, k: usize) -> usize {
    n.div_ceil(PIECES_PER_WORKER * k.max(1)).max(1)
}

/// One [`Rate`] per scalar type of a kind of work: real and complex, double
/// and single precision, the complex priors [`COMPLEX_OPS`] times the real.
#[derive(Debug)]
pub(crate) struct ScalarRates {
    real: Rate,
    complex: Rate,
    real32: Rate,
    complex32: Rate,
}

impl ScalarRates {
    /// `prior` ns per real op.
    pub(crate) const fn new(prior: f32) -> Self {
        ScalarRates {
            real: Rate::new(prior),
            complex: Rate::new(prior * COMPLEX_OPS),
            real32: Rate::new(prior),
            complex32: Rate::new(prior * COMPLEX_OPS),
        }
    }

    /// The rate of `T`.
    pub(crate) fn of<T: crate::scalar::Scalar>(&self) -> &Rate {
        let parts = if T::COMPLEX { 2 } else { 1 };
        let single = std::mem::size_of::<T>() < parts * std::mem::size_of::<f64>();
        match (T::COMPLEX, single) {
            (false, false) => &self.real,
            (true, false) => &self.complex,
            (false, true) => &self.real32,
            (true, true) => &self.complex32,
        }
    }
}

/// `f(i)` for every `i` in `0..n`, `work` all of them.
pub(crate) fn for_each(n: usize, work: Work<'_>, f: impl Fn(usize) + Sync + Send) {
    let k = match work.go::<Rayon>(n) {
        Go::Wide(k) => k,
        go => return work.serial(go, || (0..n).for_each(&f)),
    };
    let len = piece_len(n, k);
    work.wide::<Rayon>(n.div_ceil(len), k, &|p| {
        (p * len..((p + 1) * len).min(n)).for_each(&f)
    });
}

/// `f(i)` for every `i` in `0..n`, each its own piece (pieces of very
/// unequal work, such as the walks of a task tree), `work` all of them.
pub(crate) fn each(n: usize, work: Work<'_>, f: impl Fn(usize) + Sync + Send) {
    match work.go::<Rayon>(n) {
        Go::Wide(k) => work.wide::<Rayon>(n, k, &f),
        go => work.serial(go, || (0..n).for_each(&f)),
    }
}

/// `f(k, chunk)` for every chunk of `size` entries of `v` (the last one
/// shorter), `k` its index, `work` all of them.
pub(crate) fn chunks_mut<T: Send>(
    v: &mut [T],
    size: usize,
    work: Work<'_>,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    let size = size.max(1);
    match work.go::<Rayon>(v.len().div_ceil(size)) {
        Go::Wide(k) => chunks_on(v, size, k, work, f),
        go => work.serial(go, || {
            v.chunks_mut(size).enumerate().for_each(|(i, c)| f(i, c))
        }),
    }
}

/// [`chunks_mut`] on `k` threads, the decision taken ([`Work::go`]).
pub(crate) fn chunks_on<T: Send>(
    v: &mut [T],
    size: usize,
    k: usize,
    work: Work<'_>,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    let size = size.max(1);
    let (len, n) = (v.len(), v.len().div_ceil(size));
    let base = SendPtr(v.as_mut_ptr());
    let per = piece_len(n, k);
    work.wide::<Rayon>(n.div_ceil(per), k, &|q| {
        for c in q * per..((q + 1) * per).min(n) {
            let (a, b) = (c * size, ((c + 1) * size).min(len));
            // SAFETY: the chunks are disjoint ranges of `v`, which outlives
            // the section.
            let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(a), b - a) };
            f(c, chunk);
        }
    });
}

/// `f(i, &items[i], &mut scratch[i])` for every item and its scratch, an
/// item a piece, `work` all of them.
pub(crate) fn zip_mut<A: Sync, S: Send>(
    items: &[A],
    scratch: &mut [S],
    work: Work<'_>,
    f: impl Fn(usize, &A, &mut S) + Sync + Send,
) {
    let n = items.len().min(scratch.len());
    let k = match work.go::<Rayon>(n) {
        Go::Wide(k) => k,
        go => {
            return work.serial(go, || {
                for (i, (a, s)) in items.iter().zip(scratch.iter_mut()).enumerate() {
                    f(i, a, s);
                }
            })
        }
    };
    let base = SendPtr(scratch.as_mut_ptr());
    work.wide::<Rayon>(n, k, &|i| {
        // SAFETY: piece `i` alone holds `scratch[i]`.
        f(i, &items[i], unsafe { &mut *base.get().add(i) })
    });
}

/// `f(i)` for `i` in `0..n`, in order, `work` all of them.
pub(crate) fn map<R: Send>(
    n: usize,
    work: Work<'_>,
    f: impl Fn(usize) -> R + Sync + Send,
) -> Vec<R> {
    match work.go::<Rayon>(n) {
        Go::Wide(k) => map_on(n, k, work, f),
        go => work.serial(go, || (0..n).map(f).collect()),
    }
}

/// [`map`] on `k` threads, the decision taken ([`Work::go`]).
pub(crate) fn map_on<R: Send>(
    n: usize,
    k: usize,
    work: Work<'_>,
    f: impl Fn(usize) -> R + Sync + Send,
) -> Vec<R> {
    let mut out: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: every slot is written below before it is read.
    unsafe { out.set_len(n) };
    let base = SendPtr(out.as_mut_ptr());
    let len = piece_len(n, k);
    work.wide::<Rayon>(n.div_ceil(len), k, &|p| {
        for i in p * len..((p + 1) * len).min(n) {
            // SAFETY: piece `p` alone writes slot `i`.
            unsafe { (*base.get().add(i)).write(f(i)) };
        }
    });
    // SAFETY: all `n` slots written (a panic unwinds past this).
    unsafe { std::mem::transmute::<Vec<std::mem::MaybeUninit<R>>, Vec<R>>(out) }
}

/// `f(item)` for every item of `items`, consumed, in order, `work` all of
/// them.
pub(crate) fn map_owned<A: Send, R: Send>(
    items: Vec<A>,
    work: Work<'_>,
    f: impl Fn(A) -> R + Sync + Send,
) -> Vec<R> {
    let k = match work.go::<Rayon>(items.len()) {
        Go::Wide(k) => k,
        go => return work.serial(go, || items.into_iter().map(f).collect()),
    };
    let slots: Vec<std::sync::Mutex<Option<A>>> = items
        .into_iter()
        .map(|a| std::sync::Mutex::new(Some(a)))
        .collect();
    map_on(slots.len(), k, work, |i| match lock(&slots[i]).take() {
        Some(a) => f(a),
        None => unreachable!("every item is taken once"),
    })
}

/// [`map`] with a state per piece, made by `init`.
pub(crate) fn map_init<S, R: Send>(
    n: usize,
    work: Work<'_>,
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize) -> R + Sync + Send,
) -> Vec<R> {
    let k = match work.go::<Rayon>(n) {
        Go::Wide(k) => k,
        go => {
            return work.serial(go, || {
                let mut s = init();
                (0..n).map(|i| f(&mut s, i)).collect()
            })
        }
    };
    let len = piece_len(n, k);
    let pieces = n.div_ceil(len);
    let mut out: Vec<Vec<R>> = (0..pieces).map(|_| Vec::new()).collect();
    let base = SendPtr(out.as_mut_ptr());
    work.wide::<Rayon>(pieces, k, &|p| {
        let mut s = init();
        let part: Vec<R> = (p * len..((p + 1) * len).min(n))
            .map(|i| f(&mut s, i))
            .collect();
        // SAFETY: piece `p` alone writes slot `p`.
        unsafe { *base.get().add(p) = part };
    });
    out.into_iter().flatten().collect()
}

/// `f(item)` for every item of `items`, consumed, `work` all of them.
pub(crate) fn for_each_owned<A: Send>(items: Vec<A>, work: Work<'_>, f: impl Fn(A) + Sync + Send) {
    match work.go::<Rayon>(items.len()) {
        Go::Wide(k) => for_each_owned_on(items, k, work, f),
        go => work.serial(go, || items.into_iter().for_each(f)),
    }
}

/// [`for_each_owned`] on `k` threads, the decision taken ([`Work::go`]).
pub(crate) fn for_each_owned_on<A: Send>(
    items: Vec<A>,
    k: usize,
    work: Work<'_>,
    f: impl Fn(A) + Sync + Send,
) {
    let slots: Vec<std::sync::Mutex<Option<A>>> = items
        .into_iter()
        .map(|a| std::sync::Mutex::new(Some(a)))
        .collect();
    let n = slots.len();
    let len = piece_len(n, k);
    work.wide::<Rayon>(n.div_ceil(len), k, &|p| {
        for slot in &slots[p * len..((p + 1) * len).min(n)] {
            if let Some(a) = lock(slot).take() {
                f(a);
            }
        }
    });
}

/// `v` sorted by `key`, unstably: on rayon in parallel where `work` is
/// worth it, else (and on a lent pool) on this thread.
pub(crate) fn sort_unstable_by_key<T: Send, K: Ord>(
    v: &mut [T],
    work: Work<'_>,
    key: impl Fn(&T) -> K + Sync + Send,
) {
    let go = if current().is_some() {
        Go::Here
    } else {
        work.go::<Rayon>(v.len())
    };
    let Go::Wide(k) = go else {
        return work.serial(go, || v.sort_unstable_by_key(key));
    };
    use ::rayon::prelude::*;
    let n = v.len();
    work.watch::<Rayon, _>(n, k, || v.par_sort_unstable_by_key(key))
}

/// Whether `f(i)` holds for some `i` in `0..n`, `work` all of them.
pub(crate) fn any(n: usize, work: Work<'_>, f: impl Fn(usize) -> bool + Sync + Send) -> bool {
    let k = match work.go::<Rayon>(n) {
        Go::Wide(k) => k,
        go => return work.serial(go, || (0..n).any(f)),
    };
    let found = std::sync::atomic::AtomicBool::new(false);
    let len = piece_len(n, k);
    work.wide::<Rayon>(n.div_ceil(len), k, &|p| {
        for i in p * len..((p + 1) * len).min(n) {
            if found.load(Relaxed) {
                return;
            }
            if f(i) {
                found.store(true, Relaxed);
            }
        }
    });
    found.into_inner()
}

/// `a()` and `b()`, side by side where `work` (both) is worth it, else one
/// after the other on this thread.
pub(crate) fn join<A: Send, B: Send>(
    work: Work<'_>,
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    let go = work.go::<Rayon>(2);
    if !matches!(go, Go::Wide(_)) {
        return work.serial(go, || (a(), b()));
    }
    work.watch::<Rayon, _>(2, 2, || join_on(a, b))
}

/// [`join`] side by side.
fn join_on<A: Send, B: Send>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B) {
    match current() {
        Some(l) => {
            // each side leaves its own sections half this thread's share
            let half = Some((share_of(l.pool.threads()) / 2).max(1));
            let (mut a, mut b) = (Some(a), Some(b));
            let (mut ra, mut rb) = (None, None);
            l.pool.join(
                &mut || {
                    if let Some(f) = a.take() {
                        ra = Some(as_piece(Some(&l), half, f));
                    }
                },
                &mut || {
                    if let Some(f) = b.take() {
                        rb = Some(as_piece(Some(&l), half, f));
                    }
                },
            );
            match (ra, rb) {
                (Some(ra), Some(rb)) => (ra, rb),
                _ => unreachable!("a pool runs every piece"),
            }
        }
        None => {
            let half = Some((share_of(::rayon::current_num_threads()) / 2).max(1));
            ::rayon::join(|| as_piece(None, half, a), || as_piece(None, half, b))
        }
    }
}

/// Run `f` where a parallel section is cheapest to start: on a worker of the
/// pool (a rayon section injected from outside costs a wake per section, one
/// started on a worker almost nothing), if `work` is worth a section at
/// all; else here.
pub(crate) fn enter<R: Send>(work: f64, f: impl FnOnce() -> R + Send) -> R {
    if rspool::rayon::width(work, usize::MAX) < 2 {
        return f();
    }
    match current() {
        Some(l) => {
            let (mut f, mut r) = (Some(f), None);
            l.pool.enter(&mut || {
                if let Some(f) = f.take() {
                    r = Some(rspool::install_lent(l.clone(), f));
                }
            });
            match r {
                Some(r) => r,
                None => unreachable!("a pool runs what it enters"),
            }
        }
        None if ::rayon::current_thread_index().is_none() && ::rayon::current_num_threads() > 1 => {
            ::rayon::join(f, || ()).0
        }
        None => f(),
    }
}

/// A section whose tasks start tasks of their own ([`scope`]).
pub(crate) struct Scope<'a, 's>(ScopeOn<'a, 's>);

/// A task of a scope run on the calling thread.
type HereTask<'s> = Box<dyn for<'b> FnOnce(&Scope<'b, 's>) + Send + 's>;

enum ScopeOn<'a, 's> {
    Rayon(&'a ::rayon::Scope<'s>),
    Lent(&'a dyn Spawn<'s>, &'a Lent),
    /// Not worth the pool: the tasks one after the other on the calling
    /// thread, after the body.
    Here(&'a std::sync::Mutex<std::collections::VecDeque<HereTask<'s>>>),
}

impl<'a, 's> Scope<'a, 's> {
    /// Start `f` as a task of the scope; the scope returns once it ran.
    pub(crate) fn spawn(&self, f: impl for<'b> FnOnce(&Scope<'b, 's>) + Send + 's) {
        match self.0 {
            ScopeOn::Rayon(sc) => {
                sc.spawn(move |sc| as_piece(None, None, || f(&Scope(ScopeOn::Rayon(sc)))))
            }
            ScopeOn::Lent(sp, lent) => {
                let lent = lent.clone();
                sp.spawn(Box::new(move |sp: &dyn Spawn<'s>| {
                    as_piece(Some(&lent), None, || f(&Scope(ScopeOn::Lent(sp, &lent))))
                }));
            }
            ScopeOn::Here(queue) => lock(queue).push_back(Box::new(f)),
        }
    }
}

/// Run `body` with a scope its tasks are started in, returning once every
/// task ran: rayon's `scope`, or the lent pool's, where `work` (the body and
/// its tasks) is worth more than one thread; else the tasks one after the
/// other on this thread after the body, without touching the pool.
pub(crate) fn scope<'s, R: Send>(
    work: Work<'_>,
    body: impl for<'a> FnOnce(&Scope<'a, 's>) -> R + Send + 's,
) -> R {
    let go = work.go::<Rayon>(usize::MAX);
    let Go::Wide(k) = go else {
        return work.serial(go, || {
            let queue = std::sync::Mutex::new(std::collections::VecDeque::new());
            let r = body(&Scope(ScopeOn::Here(&queue)));
            loop {
                let task = lock(&queue).pop_front();
                match task {
                    Some(task) => task(&Scope(ScopeOn::Here(&queue))),
                    None => return r,
                }
            }
        });
    };
    work.watch::<Rayon, _>(usize::MAX, k, || scope_on(body))
}

/// [`scope`] on the pool.
fn scope_on<'s, R: Send>(body: impl for<'a> FnOnce(&Scope<'a, 's>) -> R + Send + 's) -> R {
    match current() {
        Some(l) => {
            let (mut body, mut r) = (Some(body), None);
            l.pool.scope(&mut |sp: &dyn Spawn<'s>| {
                if let Some(f) = body.take() {
                    r = Some(f(&Scope(ScopeOn::Lent(sp, &l))));
                }
            });
            match r {
                Some(r) => r,
                None => unreachable!("a pool runs the body of its scope"),
            }
        }
        None => ::rayon::scope(|sc| body(&Scope(ScopeOn::Rayon(sc)))),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn scalar_rates_price_complex_work_higher() {
        let s = ScalarRates::new(1.0);
        assert_eq!(s.of::<f64>().ns_per_op(), 1.0);
        assert_eq!(
            s.of::<num_complex::Complex<f64>>().ns_per_op(),
            COMPLEX_OPS as f64
        );
    }

    #[test]
    fn rayon_measures_what_a_section_costs_it() {
        let pool = ::rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let d = pool.install(|| {
            prepare();
            dispatch()
        });
        assert_eq!(d.threads(), 3);
        let costs: Vec<f64> = (2..=3).map(|k| d.cost(k)).collect();
        assert!(
            costs.iter().all(|c| c.is_finite() && *c >= 0.0),
            "{costs:?}"
        );
        assert!(costs[0] <= costs[1], "{costs:?}");
    }

    #[test]
    fn small_work_runs_here_large_work_on_the_pool() {
        let pool = ::rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let me = std::thread::current().id();
            let away = AtomicUsize::new(0);
            for_each(64, Work::ns(1.0), |_| {
                if std::thread::current().id() != me {
                    away.fetch_add(1, Relaxed);
                }
            });
            assert_eq!(away.load(Relaxed), 0, "tiny work left the thread");
            let v = map(1000, Work::ns(1e9), |i| i * 2);
            assert!(v.iter().enumerate().all(|(i, &x)| x == 2 * i));
            // a rate that has not had its probes runs its sections here,
            // however large they look at its prior
            let r = Rate::new(1e6);
            for_each(64, Work::of(1e9, &r), |_| {
                if std::thread::current().id() != me {
                    away.fetch_add(1, Relaxed);
                }
            });
            assert_eq!(away.load(Relaxed), 0, "a probe left the thread");
        });
    }
}
