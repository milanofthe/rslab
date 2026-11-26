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
//! between the factorizations) has one set of threads and one cost model
//! for all of them.
//!
//! A section runs on as many threads as its work is worth, one included: its
//! work in ns (a count of work at a constant [`Rate`]) against what a section
//! costs the pool on `k` threads (its [`Dispatch`]). Every helper here decides
//! so ([`Work::go`] at [`Rayon`]). Every result is the same however a section
//! runs: no section's arithmetic depends on how its pieces are scheduled or on
//! how many threads run them.

pub use rspool::rayon::{
    dispatch, kept, pool_dispatch as rayon_dispatch, prepare, run_width, threads, width, width_on,
    Rayon,
};
pub use rspool::{
    as_piece, chunks_mut, current, install, install_lent, is_lent, map, map_init, ranges, share_of,
    Disjoint, Dispatch, Go, Lent, Piece, Pool, Rate, Serial, Spawn, Task, Work,
};

use crate::constants::COMPLEX_OPS;

/// One `X` per scalar type of a kind of work: real and complex, double and
/// single precision.
#[derive(Debug)]
pub(crate) struct PerScalar<X> {
    real: X,
    complex: X,
    real32: X,
    complex32: X,
}

/// One [`Rate`] per scalar type of a kind of work.
pub(crate) type ScalarRates = PerScalar<Rate>;

impl PerScalar<Rate> {
    /// `prior` ns per real op, the complex priors [`COMPLEX_OPS`] times it.
    pub(crate) const fn new(prior: f32) -> Self {
        PerScalar {
            real: Rate::new(prior),
            complex: Rate::new(prior * COMPLEX_OPS),
            real32: Rate::new(prior),
            complex32: Rate::new(prior * COMPLEX_OPS),
        }
    }
}

impl<X> PerScalar<X> {
    /// The one of `T`.
    pub(crate) fn of<T: crate::scalar::Scalar>(&self) -> &X {
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

/// `f(i)` for every `i` in `0..n`, each its own piece (pieces of very
/// unequal work, such as the walks of a task tree), `work` all of them.
pub(crate) fn each(n: usize, work: Work<'_>, f: impl Fn(usize) + Sync + Send) {
    match work.go::<Rayon>(n) {
        Go::Wide(k) => run_width(n, k, &f),
        _ => (0..n).for_each(&f),
    }
}

/// `f(i, &items[i], &mut scratch[i])` for every item and its scratch,
/// `work` all of them.
pub(crate) fn zip_mut<A: Sync, S: Send>(
    items: &[A],
    scratch: &mut [S],
    work: Work<'_>,
    f: impl Fn(usize, &A, &mut S) + Sync + Send,
) {
    let n = items.len().min(scratch.len());
    chunks_mut::<Rayon, _>(&mut scratch[..n], 1, work, |i0, part| {
        for (j, s) in part.iter_mut().enumerate() {
            f(i0 + j, &items[i0 + j], s);
        }
    });
}

/// `f(item)` for every item of `items`, consumed, in order, `work` all of
/// them.
pub(crate) fn map_owned<A: Send, R: Send>(
    items: Vec<A>,
    work: Work<'_>,
    f: impl Fn(A) -> R + Sync + Send,
) -> Vec<R> {
    let mut slots: Vec<(Option<A>, Option<R>)> =
        items.into_iter().map(|a| (Some(a), None)).collect();
    chunks_mut::<Rayon, _>(&mut slots, 1, work, |_, part| {
        for (a, r) in part {
            *r = a.take().map(&f);
        }
    });
    (slots.into_iter())
        .map(|(_, r)| match r {
            Some(r) => r,
            None => unreachable!("every item is mapped"),
        })
        .collect()
}

/// `f(item)` for every item of `items`, consumed, `work` all of them.
pub(crate) fn for_each_owned<A: Send>(items: Vec<A>, work: Work<'_>, f: impl Fn(A) + Sync + Send) {
    let mut slots: Vec<Option<A>> = items.into_iter().map(Some).collect();
    chunks_mut::<Rayon, _>(&mut slots, 1, work, |_, part| {
        part.iter_mut().filter_map(Option::take).for_each(&f)
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
    if go == Go::Here {
        return v.sort_unstable_by_key(key);
    }
    use ::rayon::prelude::*;
    v.par_sort_unstable_by_key(key)
}

/// Whether `f(i)` holds for some `i` in `0..n`, `work` all of them.
pub(crate) fn any(n: usize, work: Work<'_>, f: impl Fn(usize) -> bool + Sync + Send) -> bool {
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    let found = AtomicBool::new(false);
    ranges::<Rayon>(n, work, |r| {
        for i in r {
            if found.load(Relaxed) {
                return;
            }
            if f(i) {
                found.store(true, Relaxed);
                return;
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
    if work.go::<Rayon>(2) == Go::Here {
        return (a(), b());
    }
    join_on(a, b)
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
            // each side leaves its sections the whole pool: rayon runs
            // their pieces on free threads only, and halving it kept the
            // threads a short side leaves behind from the long side's
            let (a, b) = (carried(a), carried(b));
            ::rayon::join(|| as_piece(None, None, a), || as_piece(None, None, b))
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
            let (mut f, mut r) = (Some(carried(f)), None);
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
            ::rayon::join(carried(f), || ()).0
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
                let node = rsprobe::current();
                sc.spawn(move |sc| {
                    let _node = node.enter();
                    as_piece(None, None, || f(&Scope(ScopeOn::Rayon(sc))))
                })
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
    if work.go::<Rayon>(usize::MAX) != Go::Here {
        return scope_on(body);
    }
    let queue = std::sync::Mutex::new(std::collections::VecDeque::new());
    let r = body(&Scope(ScopeOn::Here(&queue)));
    loop {
        let task = lock(&queue).pop_front();
        match task {
            Some(task) => task(&Scope(ScopeOn::Here(&queue))),
            None => return r,
        }
    }
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
        None => {
            let node = rsprobe::current();
            ::rayon::scope(|sc| {
                let _node = node.enter();
                body(&Scope(ScopeOn::Rayon(sc)))
            })
        }
    }
}

/// `f` under the calling thread's rsprobe node on whichever thread runs it:
/// the scopes of work handed to another thread hang under the scope that
/// handed it over, its time is `busy` there.
pub(crate) fn carried<R>(f: impl FnOnce() -> R + Send) -> impl FnOnce() -> R + Send {
    let node = rsprobe::current();
    move || {
        let _node = node.enter();
        f()
    }
}

/// `m` locked: a lock is poisoned only by a panic, which its section
/// carries on.
fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

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
            ranges::<Rayon>(64, Work::ns(1.0), |_| {
                if std::thread::current().id() != me {
                    away.fetch_add(1, Relaxed);
                }
            });
            assert_eq!(away.load(Relaxed), 0, "tiny work left the thread");
            let v = map::<Rayon, _>(1000, Work::ns(1e9), |i| i * 2);
            assert!(v.iter().enumerate().all(|(i, &x)| x == 2 * i));
            // large work at a rate goes wide and computes the same
            let r = Rate::new(1e6);
            let w = map::<Rayon, _>(1000, Work::of(1e9, &r), |i| i * 3);
            assert!(w.iter().enumerate().all(|(i, &x)| x == 3 * i));
        });
    }
}
