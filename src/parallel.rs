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

#[derive(Debug)]
pub(crate) struct PerScalar<X> {
    real: X,
    complex: X,
    real32: X,
    complex32: X,
}

pub(crate) type ScalarRates = PerScalar<Rate>;

impl PerScalar<Rate> {
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

pub(crate) fn each(n: usize, work: Work<'_>, f: impl Fn(usize) + Sync + Send) {
    match work.go::<Rayon>(n) {
        Go::Wide(k) => run_width(n, k, &f),
        _ => (0..n).for_each(&f),
    }
}

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

pub(crate) fn for_each_owned<A: Send>(items: Vec<A>, work: Work<'_>, f: impl Fn(A) + Sync + Send) {
    let mut slots: Vec<Option<A>> = items.into_iter().map(Some).collect();
    chunks_mut::<Rayon, _>(&mut slots, 1, work, |_, part| {
        part.iter_mut().filter_map(Option::take).for_each(&f)
    });
}

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

fn join_on<A: Send, B: Send>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B) {
    match current() {
        Some(l) => {
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
            let (a, b) = (carried(a), carried(b));
            ::rayon::join(|| as_piece(None, None, a), || as_piece(None, None, b))
        }
    }
}

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

pub(crate) struct Scope<'a, 's>(ScopeOn<'a, 's>);

type HereTask<'s> = Box<dyn for<'b> FnOnce(&Scope<'b, 's>) + Send + 's>;

enum ScopeOn<'a, 's> {
    Rayon(&'a ::rayon::Scope<'s>),
    Lent(&'a dyn Spawn<'s>, &'a Lent),
    Here(&'a std::sync::Mutex<std::collections::VecDeque<HereTask<'s>>>),
}

impl<'a, 's> Scope<'a, 's> {
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

pub(crate) fn carried<R>(f: impl FnOnce() -> R + Send) -> impl FnOnce() -> R + Send {
    let node = rsprobe::current();
    move || {
        let _node = node.enter();
        f()
    }
}

pub(crate) fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
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
            let r = Rate::new(1e6);
            let w = map::<Rayon, _>(1000, Work::of(1e9, &r), |i| i * 3);
            assert!(w.iter().enumerate().all(|(i, &x)| x == 3 * i));
        });
    }
}
