//! The workers every parallel path of rslab runs on, and how wide a section
//! goes on them.
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
//! A section runs on as many threads as its work is worth, one included:
//! its work in ns (a count of work at a [`Rate`] learned from the runs)
//! against what a section costs the pool on `k` threads (its [`Dispatch`]:
//! what the lent pool says through [`Pool::dispatch`], for rayon measured
//! on the pool's first use). A section not worth a second thread runs on
//! the calling thread without touching the pool. A host that lends its
//! workers to rsdag too hands rslab the same measured costs, so both
//! libraries decide alike on the threads they share.
//!
//! Every parallel site goes through the few helpers here. On rayon they are
//! the rayon idioms they replace; on a lent pool they are its sections
//! ([`Pool::run_width`], [`Pool::join`], [`Pool::scope`]), and a section
//! opened inside one runs on the same pool, which decides whether it nests
//! (a pool that steals work runs it alongside). Every result is the same
//! either way: no section's arithmetic depends on how its pieces are
//! scheduled or on how many threads run them.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

use crate::constants::{
    ASSUMED_HELPER_NS, ASSUMED_SECTION_NS, CALIBRATION_LINES, CALIBRATION_PIECE_NS,
    CALIBRATION_ROUNDS, COMPLEX_OPS, GAIN, PIECES_PER_WORKER, RATE_MIN_NS, RATE_SAMPLE_EVERY,
    RATE_WEIGHT,
};

/// Workers a host lends rslab.
pub trait Pool: Send + Sync {
    /// Number of workers.
    fn threads(&self) -> usize;
    /// Run `f(i)` for every `i` in `0..n`, returning when all ran.
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync));
    /// [`run`](Self::run) on at most `width` threads, the calling one among
    /// them. The default runs on all.
    fn run_width(&self, n: usize, width: usize, f: &(dyn Fn(usize) + Sync)) {
        let _ = width;
        self.run(n, f)
    }
    /// What a section costs on this pool. The default assumes
    /// ([`Dispatch::assumed`]); a pool that measures itself says so.
    fn dispatch(&self) -> Dispatch {
        Dispatch::assumed(self.threads())
    }
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

/// The pieces of a section: `f(i)` for every `i` of it.
pub type Piece<'a> = dyn Fn(usize) + Sync + 'a;

/// Whether `work` ns is some work (not zero, negative or NaN).
fn has_work(work: f64) -> bool {
    work > 0.0
}

/// What a section costs on a pool: `cost(k)` ns for a section on `k`
/// threads, the calling one among them. A section of `work` ns of serial
/// work in `p` equal pieces takes about `work * ceil(p / k) / p + cost(k)`
/// on `k` threads, and goes as wide as that is shortest ([`width`](
/// Self::width)).
#[derive(Clone, Debug)]
pub struct Dispatch {
    /// `cost[k]` for `k` threads; `cost[0]` and `cost[1]` are zero.
    cost: Arc<[f64]>,
    /// The least work any section goes wide with ([`break_even`](
    /// Self::break_even)).
    least: f64,
}

impl Dispatch {
    /// The costs `cost[k]` of a section on `k` threads, for `k` up to
    /// `cost.len() - 1` (`cost[0]` and `cost[1]` are not read).
    pub fn new(cost: Vec<f64>) -> Self {
        let mut cost = cost;
        if cost.len() < 2 {
            cost.resize(2, 0.0);
        }
        for c in cost.iter_mut().take(2) {
            *c = 0.0;
        }
        // Over many pieces a section on `k` threads takes `work / k +
        // cost[k]`, which saves `GAIN` of `work` from this work on.
        let least = (2..cost.len())
            .filter_map(|k| {
                let saved = 1.0 - GAIN - 1.0 / k as f64;
                (saved > 0.0).then(|| cost[k].max(0.0) / saved)
            })
            .fold(f64::INFINITY, f64::min);
        Dispatch {
            cost: cost.into(),
            least,
        }
    }

    /// Sections cost nothing on `threads` threads: every section of two
    /// pieces or more goes wide (tests, and hosts that know better).
    pub fn free(threads: usize) -> Self {
        Dispatch::new(vec![0.0; threads.max(1) + 1])
    }

    /// No section goes wide.
    pub fn never() -> Self {
        Dispatch::new(vec![0.0; 2])
    }

    /// What a pool of `threads` threads that measures nothing costs:
    /// [`ASSUMED_SECTION_NS`] plus [`ASSUMED_HELPER_NS`] per helper.
    pub fn assumed(threads: usize) -> Self {
        let cost = (0..=threads.max(1))
            .map(|k| ASSUMED_SECTION_NS + ASSUMED_HELPER_NS * k.saturating_sub(1) as f64)
            .collect();
        Dispatch::new(cost)
    }

    /// The most threads a section is costed for.
    pub fn threads(&self) -> usize {
        self.cost.len().saturating_sub(1).max(1)
    }

    /// What a section costs on `k` threads, in ns.
    pub fn cost(&self, k: usize) -> f64 {
        if k < 2 {
            return 0.0;
        }
        self.cost.get(k).copied().unwrap_or(f64::INFINITY)
    }

    /// The least work, in ns, with which a section of any number of pieces
    /// goes wide: below it every section runs serially.
    pub fn break_even(&self) -> f64 {
        self.least
    }

    /// The least work, in ns, with which a section of `pieces` equal pieces
    /// goes wide (infinite where none does).
    pub fn least(&self, pieces: usize) -> f64 {
        let p = pieces.max(1) as f64;
        (2..=pieces.min(self.threads()))
            .filter_map(|k| {
                let saved = 1.0 - GAIN - pieces.div_ceil(k) as f64 / p;
                (saved > 0.0).then(|| self.cost[k].max(0.0) / saved)
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// The threads a section of `work` ns in `pieces` pieces runs on: the
    /// count it takes the shortest on, if that saves at least [`GAIN`] of
    /// the serial time; one (serially) otherwise.
    pub fn width(&self, work: f64, pieces: usize) -> usize {
        self.width_after(work, pieces, 0.0)
    }

    /// [`width`](Self::width) for a section that pays `extra` ns more on
    /// two threads or more (the hand-off into the pool it runs on).
    pub(crate) fn width_after(&self, work: f64, pieces: usize, extra: f64) -> usize {
        if !has_work(work) || pieces < 2 || work < self.least {
            return 1;
        }
        if work.is_infinite() {
            // Worth any section: as wide as the pieces and the costed
            // threads go.
            return pieces.min(self.threads());
        }
        let p = pieces as f64;
        let mut best = (1, work);
        for k in 2..=pieces.min(self.threads()) {
            let t = work * pieces.div_ceil(k) as f64 / p + self.cost[k] + extra;
            if t < best.1 {
                best = (k, t);
            }
        }
        if best.1 <= (1.0 - GAIN) * work {
            best.0
        } else {
            1
        }
    }

    /// What a section costs on a pool of `threads` threads, measured:
    /// `run(k, piece)` runs `piece(i)` for `i` in `0..k` on `k` threads.
    /// Per thread count the median of [`CALIBRATION_ROUNDS`] sections of as
    /// many pieces, each writing [`CALIBRATION_LINES`] cache lines (the
    /// hand-over of a piece's output a real section pays) and then busy for
    /// [`CALIBRATION_PIECE_NS`], less what one piece costs alone; never less
    /// than with fewer threads. A piece lasts long enough for the pool's
    /// other threads to take theirs: on a work-stealing pool whose caller
    /// runs pieces nobody took, instant pieces would all run on the caller
    /// and the section would look free. Run after as many sections on all
    /// threads, so it finds the workers awake as a factorization's sections
    /// do.
    pub fn measure(threads: usize, run: &dyn Fn(usize, &Piece<'_>)) -> Dispatch {
        if threads < 2 {
            return Dispatch::never();
        }
        #[repr(align(64))]
        struct Line(AtomicU64);
        let lines: Vec<Line> = (0..threads * CALIBRATION_LINES)
            .map(|_| Line(AtomicU64::new(0)))
            .collect();
        let busy = std::time::Duration::from_nanos(CALIBRATION_PIECE_NS);
        let piece = |i: usize| {
            let t = crate::clock::Instant::now();
            for l in &lines[i * CALIBRATION_LINES..(i + 1) * CALIBRATION_LINES] {
                l.0.fetch_add(1, Relaxed);
            }
            // An inert clock (no OS clock) reads zero: no busy wait there.
            while cfg!(not(target_arch = "wasm32")) && t.elapsed() < busy {
                std::hint::spin_loop();
            }
        };
        let median = |f: &dyn Fn()| {
            let mut t: Vec<f64> = (0..CALIBRATION_ROUNDS)
                .map(|_| {
                    let s = crate::clock::Instant::now();
                    f();
                    s.elapsed().as_nanos() as f64
                })
                .collect();
            t.sort_by(f64::total_cmp);
            t[t.len() / 2]
        };
        for _ in 0..CALIBRATION_ROUNDS {
            run(threads, &piece);
        }
        let one = median(&|| piece(0));
        let mut cost = vec![0.0; threads + 1];
        for k in 2..=threads {
            let c = median(&|| run(k, &piece)) - one;
            cost[k] = c.max(cost[k - 1]);
        }
        Dispatch::new(cost)
    }
}

/// What an op of some work costs, in ns: a prior until runs are measured,
/// then their running mean. A section's work is its ops at its kind's rate
/// (a replay's fmadds, a product's multiply-adds, a pass's entries). A
/// thread measures one run in [`RATE_SAMPLE_EVERY`] of those estimated at
/// [`RATE_MIN_NS`] or more.
#[derive(Debug)]
pub struct Rate {
    /// ns per op, as `f32` bits.
    ns: AtomicU32,
}

thread_local! {
    /// The runs this thread could have measured, counted for the sampling.
    static TICK: Cell<u32> = const { Cell::new(0) };
}

impl Rate {
    /// A rate of `prior` ns per op until runs are measured.
    pub const fn new(prior: f32) -> Self {
        Rate {
            ns: AtomicU32::new(prior.to_bits()),
        }
    }

    /// ns per op.
    pub fn ns_per_op(&self) -> f64 {
        f32::from_bits(self.ns.load(Relaxed)) as f64
    }

    /// Whether to measure a run of `ops` ops.
    pub fn sample(&self, ops: f64) -> bool {
        ops * self.ns_per_op() >= RATE_MIN_NS
            && TICK.with(|t| {
                let v = t.get();
                t.set(v.wrapping_add(1));
                v.is_multiple_of(RATE_SAMPLE_EVERY)
            })
    }

    /// A run of `ops` ops took `ns` ns of one thread.
    pub fn learn(&self, ns: f64, ops: f64) {
        if !(ns > 0.0 && ops > 0.0) {
            return;
        }
        let seen = (ns / ops) as f32;
        let _ = self.ns.fetch_update(Relaxed, Relaxed, |b| {
            let old = f32::from_bits(b);
            Some((old + RATE_WEIGHT * (seen - old)).to_bits())
        });
    }

    /// `f`, a run of `ops` ops on this thread, measured if sampled.
    pub fn time<R>(&self, ops: f64, f: impl FnOnce() -> R) -> R {
        if !self.sample(ops) {
            return f();
        }
        let t = crate::clock::Instant::now();
        let r = f();
        self.learn(t.elapsed().as_nanos() as f64, ops);
        r
    }
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

/// A section's work: `ops` at a learned rate, which learns from the
/// section's runs, or an estimate in ns.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Work<'r> {
    ops: f64,
    rate: Option<&'r Rate>,
}

impl<'r> Work<'r> {
    /// `ns` ns of work.
    pub(crate) fn ns(ns: f64) -> Self {
        Work {
            ops: ns,
            rate: None,
        }
    }

    /// `ops` ops at `rate`.
    pub(crate) fn of(ops: f64, rate: &'r Rate) -> Self {
        Work {
            ops,
            rate: Some(rate),
        }
    }

    /// The work in ns.
    pub(crate) fn time(&self) -> f64 {
        match self.rate {
            Some(r) => self.ops * r.ns_per_op(),
            None => self.ops,
        }
    }

    /// `f` run on this thread, measured for the rate if sampled.
    fn serial<R>(&self, f: impl FnOnce() -> R) -> R {
        match self.rate {
            Some(r) => r.time(self.ops, f),
            None => f(),
        }
    }
}

/// A lent pool and what a section costs on it.
#[derive(Clone)]
struct Lent {
    pool: Arc<dyn Pool>,
    dispatch: Dispatch,
}

thread_local! {
    static CURRENT: RefCell<Option<Lent>> = const { RefCell::new(None) };
    /// Pieces of sections this thread is running: a rayon pool is not
    /// measured from inside one.
    static IN_PIECE: Cell<u32> = const { Cell::new(0) };
    /// The section cost of the rayon pool this thread last worked in, by
    /// its thread count and whether this thread is one of its workers.
    static RAYON: RefCell<Option<(usize, bool, Dispatch)>> = const { RefCell::new(None) };
}

/// The rayon section costs measured in this process, by thread count and
/// whether measured from a worker.
static MEASURED: std::sync::Mutex<Vec<(usize, bool, Dispatch)>> = std::sync::Mutex::new(Vec::new());

/// Run `f` with `pool` lent: the factorizations and solves `f` calls run
/// their parallel sections on it, as wide as its [`Pool::dispatch`] makes
/// them worth. The previous one comes back when `f` returns (or unwinds).
pub fn install<R>(pool: Arc<dyn Pool>, f: impl FnOnce() -> R) -> R {
    let dispatch = pool.dispatch();
    install_lent(Lent { pool, dispatch }, f)
}

fn install_lent<R>(lent: Lent, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Lent>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let p = self.0.take();
            CURRENT.with(|c| *c.borrow_mut() = p);
        }
    }
    let _restore = Restore(CURRENT.with(|c| c.replace(Some(lent))));
    f()
}

/// The lent pool, if any.
pub(crate) fn lent() -> Option<Arc<dyn Pool>> {
    CURRENT.with(|c| c.borrow().as_ref().map(|l| l.pool.clone()))
}

fn current() -> Option<Lent> {
    CURRENT.with(|c| c.borrow().clone())
}

/// `f` with `lent` installed on this thread, unless one is: what `f` opens
/// runs on the pool too.
fn here<R>(lent: &Lent, f: impl FnOnce() -> R) -> R {
    if CURRENT.with(|c| c.borrow().is_some()) {
        f()
    } else {
        install_lent(lent.clone(), f)
    }
}

/// `f` as a piece of a section on this thread.
fn piece<R>(f: impl FnOnce() -> R) -> R {
    struct Leave;
    impl Drop for Leave {
        fn drop(&mut self) {
            IN_PIECE.with(|p| p.set(p.get() - 1));
        }
    }
    IN_PIECE.with(|p| p.set(p.get() + 1));
    let _leave = Leave;
    f()
}

/// The workers of the current section: the lent pool's, or the current
/// rayon pool's.
pub fn threads() -> usize {
    match lent() {
        Some(p) => p.threads().max(1),
        None => rayon::current_num_threads().max(1),
    }
}

/// What a section costs here: on the lent pool, or on the current rayon
/// pool (measured on its first use).
pub fn dispatch() -> Dispatch {
    if let Some(l) = CURRENT.with(|c| c.borrow().as_ref().map(|l| l.dispatch.clone())) {
        return l;
    }
    let threads = rayon::current_num_threads();
    if threads < 2 {
        return kept(true, 1);
    }
    with_rayon(threads, Dispatch::clone)
}

/// What a section costs on the current rayon pool, lent pool or not: for
/// the code that runs on rayon whatever is lent (the nested dissection).
pub(crate) fn rayon_dispatch() -> Dispatch {
    let threads = rayon::current_num_threads();
    if threads < 2 {
        return kept(true, 1);
    }
    with_rayon(threads, Dispatch::clone)
}

/// The threads `work` ns of work in `pieces` pieces runs on here: as many
/// as the section cost of the lent pool, or of the current rayon pool,
/// makes it worth, one (serially) where none is.
pub fn width(work: f64, pieces: usize) -> usize {
    if pieces < 2 || !has_work(work) {
        return 1;
    }
    let lent = CURRENT.with(|c| {
        c.borrow()
            .as_ref()
            .map(|l| l.dispatch.width(work, pieces).min(l.pool.threads()).max(1))
    });
    if let Some(k) = lent {
        return k;
    }
    let threads = rayon::current_num_threads();
    if threads < 2 {
        return 1;
    }
    with_rayon(threads, |d| d.width(work, pieces))
        .min(threads)
        .max(1)
}

/// The threads `work` ns in `pieces` pieces runs on in a pool of `threads`
/// threads the caller is about to enter: as [`width`] where that pool is the
/// lent one or the current rayon pool, else as a rayon pool of as many
/// threads measured when one was last used from its workers (assumed
/// before), the section into it from here paid on top.
pub(crate) fn width_on(threads: usize, work: f64, pieces: usize) -> usize {
    if threads < 2 || pieces < 2 || !has_work(work) {
        return 1;
    }
    if lent().is_some() || rayon::current_num_threads() == threads {
        return width(work, pieces).min(threads);
    }
    let measured = lock(&MEASURED)
        .iter()
        .find(|(t, inside, _)| *t == threads && *inside)
        .map(|e| e.2.clone());
    let enter = dispatch().cost(2);
    measured
        .unwrap_or_else(|| kept(false, threads))
        .width_after(work, pieces, enter)
        .min(threads)
}

/// The section costs made from a formula: free ([`Dispatch::free`]) or
/// assumed ([`Dispatch::assumed`]) on a thread count, kept once made, so a
/// gate that needs one allocates nothing.
static KEPT: std::sync::Mutex<Vec<(bool, usize, Dispatch)>> = std::sync::Mutex::new(Vec::new());

/// [`Dispatch::free`] (`free`) or [`Dispatch::assumed`] on `threads`
/// threads, made once.
pub(crate) fn kept(free: bool, threads: usize) -> Dispatch {
    let mut k = lock(&KEPT);
    if let Some(e) = k.iter().find(|e| e.0 == free && e.1 == threads) {
        return e.2.clone();
    }
    let d = if free {
        Dispatch::free(threads)
    } else {
        Dispatch::assumed(threads)
    };
    k.push((free, threads, d.clone()));
    d
}

/// `f` of the section cost of the current rayon pool of `threads` threads.
fn with_rayon<R>(threads: usize, f: impl FnOnce(&Dispatch) -> R) -> R {
    let inside = rayon::current_thread_index().is_some();
    let cached =
        RAYON.with(|c| matches!(&*c.borrow(), Some((t, i, _)) if *t == threads && *i == inside));
    if !cached {
        match measured(threads, inside) {
            Some(d) => RAYON.with(|c| *c.borrow_mut() = Some((threads, inside, d))),
            None => return f(&kept(false, threads)),
        }
    }
    RAYON.with(|c| match &*c.borrow() {
        Some((_, _, d)) => f(d),
        None => unreachable!("cached above"),
    })
}

/// The section cost of a rayon pool of `threads` threads, from inside it or
/// not: measured once per process, here and now when nobody did yet. `None`
/// inside a piece of a section, where a measurement would time the
/// section's own work.
fn measured(threads: usize, inside: bool) -> Option<Dispatch> {
    let find = |m: &[(usize, bool, Dispatch)]| {
        m.iter()
            .find(|(t, i, _)| *t == threads && *i == inside)
            .map(|e| e.2.clone())
    };
    if let Some(d) = find(&lock(&MEASURED)) {
        return Some(d);
    }
    if IN_PIECE.with(|p| p.get()) > 0 {
        return None;
    }
    // Not under the lock: a thread waiting in a rayon section runs other
    // work meanwhile, which may ask here too.
    let d = piece(|| Dispatch::measure(threads, &|k, p| rayon_run(k, k, p)));
    let mut m = lock(&MEASURED);
    if let Some(d) = find(&m) {
        return Some(d);
    }
    m.push((threads, inside, d.clone()));
    Some(d)
}

/// Measure what a section costs on the current rayon pool now, if nobody
/// did yet: for a pool just built, whose workers nothing else keeps busy.
pub(crate) fn prepare() {
    let threads = rayon::current_num_threads();
    if lent().is_none() && threads >= 2 {
        with_rayon(threads, |_| ());
    }
}

/// `f(i)` for `i` in `0..n` on `k` threads of the current rayon pool, the
/// calling one among them.
fn rayon_run(n: usize, k: usize, f: &(dyn Fn(usize) + Sync)) {
    use rayon::prelude::*;
    if k >= n {
        (0..n)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|i| piece(|| f(i)));
    } else {
        let next = AtomicUsize::new(0);
        (0..k).into_par_iter().with_max_len(1).for_each(|_| {
            piece(|| loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= n {
                    break;
                }
                f(i);
            })
        });
    }
}

/// `f(i)` for `i` in `0..n` on `k` threads (the calling one among them) of
/// the lent pool, or of the current rayon pool; serially on this thread for
/// `k < 2`, without touching either.
pub(crate) fn run_width(n: usize, k: usize, f: &(dyn Fn(usize) + Sync)) {
    if n < 2 || k < 2 {
        return (0..n).for_each(f);
    }
    let k = k.min(n);
    match current() {
        Some(l) => l.pool.run_width(n, k, &|i| here(&l, || piece(|| f(i)))),
        None => rayon_run(n, k, f),
    }
}

/// `n` pieces of `work` together on `k` threads ([`run_width`]); the
/// work's rate learns from the run when sampled: the threads' time in the
/// pieces, whatever the section cost and however many threads came.
pub(crate) fn run_work(n: usize, k: usize, work: Work<'_>, f: &(dyn Fn(usize) + Sync)) {
    if k < 2 || n < 2 {
        return work.serial(|| (0..n).for_each(f));
    }
    match work.rate {
        Some(rate) if rate.sample(work.ops) => {
            let busy = AtomicU64::new(0);
            run_width(n, k, &|i| {
                let t = crate::clock::Instant::now();
                f(i);
                busy.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            });
            rate.learn(busy.into_inner() as f64, work.ops);
        }
        _ => run_width(n, k, f),
    }
}

/// Items per piece for `n` items on `k` threads: about
/// [`PIECES_PER_WORKER`] pieces per thread.
fn piece_len(n: usize, k: usize) -> usize {
    n.div_ceil(PIECES_PER_WORKER * k.max(1)).max(1)
}

/// `f(i)` for every `i` in `0..n`, `work` all of them.
pub(crate) fn for_each(n: usize, work: Work<'_>, f: impl Fn(usize) + Sync + Send) {
    let k = width(work.time(), n);
    if k < 2 {
        return work.serial(|| (0..n).for_each(&f));
    }
    let len = piece_len(n, k);
    run_work(n.div_ceil(len), k, work, &|p| {
        (p * len..((p + 1) * len).min(n)).for_each(&f)
    });
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
    let (len, n) = (v.len(), v.len().div_ceil(size));
    let k = width(work.time(), n);
    if k < 2 {
        return work.serial(|| v.chunks_mut(size).enumerate().for_each(|(i, c)| f(i, c)));
    }
    let base = SendPtr(v.as_mut_ptr());
    let per = piece_len(n, k);
    run_work(n.div_ceil(per), k, work, &|q| {
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
    let k = width(work.time(), n);
    if k < 2 {
        return work.serial(|| {
            for (i, (a, s)) in items.iter().zip(scratch.iter_mut()).enumerate() {
                f(i, a, s);
            }
        });
    }
    let base = SendPtr(scratch.as_mut_ptr());
    run_work(n, k, work, &|i| {
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
    let k = width(work.time(), n);
    if k < 2 {
        return work.serial(|| (0..n).map(f).collect());
    }
    let mut out: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: every slot is written below before it is read.
    unsafe { out.set_len(n) };
    let base = SendPtr(out.as_mut_ptr());
    let len = piece_len(n, k);
    run_work(n.div_ceil(len), k, work, &|p| {
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
    if width(work.time(), items.len()) < 2 {
        return work.serial(|| items.into_iter().map(f).collect());
    }
    let slots: Vec<std::sync::Mutex<Option<A>>> = items
        .into_iter()
        .map(|a| std::sync::Mutex::new(Some(a)))
        .collect();
    map(slots.len(), work, |i| match lock(&slots[i]).take() {
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
    let k = width(work.time(), n);
    if k < 2 {
        return work.serial(|| {
            let mut s = init();
            (0..n).map(|i| f(&mut s, i)).collect()
        });
    }
    let len = piece_len(n, k);
    let pieces = n.div_ceil(len);
    let mut out: Vec<Vec<R>> = (0..pieces).map(|_| Vec::new()).collect();
    let base = SendPtr(out.as_mut_ptr());
    run_work(pieces, k, work, &|p| {
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
    if width(work.time(), items.len()) < 2 {
        return work.serial(|| items.into_iter().for_each(f));
    }
    map_owned(items, work, f);
}

/// `v` sorted by `key`, unstably: on rayon in parallel where `work` is
/// worth it, else (and on a lent pool) on this thread.
pub(crate) fn sort_unstable_by_key<T: Send, K: Ord>(
    v: &mut [T],
    work: Work<'_>,
    key: impl Fn(&T) -> K + Sync + Send,
) {
    if lent().is_some() || width(work.time(), v.len()) < 2 {
        return work.serial(|| v.sort_unstable_by_key(key));
    }
    use rayon::prelude::*;
    v.par_sort_unstable_by_key(key)
}

/// Whether `f(i)` holds for some `i` in `0..n`, `work` all of them.
pub(crate) fn any(n: usize, work: Work<'_>, f: impl Fn(usize) -> bool + Sync + Send) -> bool {
    if width(work.time(), n) < 2 {
        return work.serial(|| (0..n).any(f));
    }
    let found = std::sync::atomic::AtomicBool::new(false);
    for_each(n, work, |i| {
        if !found.load(Relaxed) && f(i) {
            found.store(true, Relaxed);
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
    if width(work.time(), 2) < 2 {
        return (a(), b());
    }
    match current() {
        Some(l) => {
            let (mut a, mut b) = (Some(a), Some(b));
            let (mut ra, mut rb) = (None, None);
            l.pool.join(
                &mut || {
                    if let Some(f) = a.take() {
                        ra = Some(here(&l, || piece(f)));
                    }
                },
                &mut || {
                    if let Some(f) = b.take() {
                        rb = Some(here(&l, || piece(f)));
                    }
                },
            );
            match (ra, rb) {
                (Some(ra), Some(rb)) => (ra, rb),
                _ => unreachable!("a pool runs every piece"),
            }
        }
        None => rayon::join(|| piece(a), || piece(b)),
    }
}

/// Run `f` where a parallel section is cheapest to start: on a worker of the
/// pool (a rayon section injected from outside costs a wake per section, one
/// started on a worker almost nothing), if `work` is worth a section at
/// all; else here.
pub(crate) fn enter<R: Send>(work: f64, f: impl FnOnce() -> R + Send) -> R {
    if width(work, usize::MAX) < 2 {
        return f();
    }
    match current() {
        Some(l) => {
            let (mut f, mut r) = (Some(f), None);
            l.pool.enter(&mut || {
                if let Some(f) = f.take() {
                    r = Some(install_lent(l.clone(), f));
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

/// A section whose tasks start tasks of their own ([`scope`]).
pub(crate) struct Scope<'a, 's>(ScopeOn<'a, 's>);

/// A task of a scope run on the calling thread.
type HereTask<'s> = Box<dyn for<'b> FnOnce(&Scope<'b, 's>) + Send + 's>;

enum ScopeOn<'a, 's> {
    Rayon(&'a rayon::Scope<'s>),
    Lent(&'a dyn Spawn<'s>, &'a Lent),
    /// Not worth the pool: the tasks one after the other on the calling
    /// thread, after the body.
    Here(&'a std::sync::Mutex<std::collections::VecDeque<HereTask<'s>>>),
}

impl<'a, 's> Scope<'a, 's> {
    /// Start `f` as a task of the scope; the scope returns once it ran.
    pub(crate) fn spawn(&self, f: impl for<'b> FnOnce(&Scope<'b, 's>) + Send + 's) {
        match self.0 {
            ScopeOn::Rayon(sc) => sc.spawn(move |sc| piece(|| f(&Scope(ScopeOn::Rayon(sc))))),
            ScopeOn::Lent(sp, lent) => {
                let lent = lent.clone();
                sp.spawn(Box::new(move |sp: &dyn Spawn<'s>| {
                    here(&lent, || piece(|| f(&Scope(ScopeOn::Lent(sp, &lent)))))
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
    if width(work.time(), usize::MAX) < 2 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_section_goes_as_wide_as_its_work_is_worth() {
        assert_eq!(Dispatch::free(8).width(1.0, 3), 3);
        assert_eq!(Dispatch::free(8).width(1.0, 100), 8);
        assert_eq!(Dispatch::free(8).width(1.0, 1), 1);
        assert_eq!(Dispatch::never().width(1e12, 100), 1);
        // 1 us per section plus 1 us per helper: a microsecond of work stays
        // serial, a millisecond goes to all eight threads, and in between
        // some.
        let d = Dispatch::new((0..=8).map(|k| 1e3 * k as f64).collect());
        assert_eq!(d.width(1e3, 64), 1);
        assert_eq!(d.width(1e6, 64), 8);
        let mid = d.width(2e4, 64);
        assert!(mid > 1 && mid < 8, "{mid}");
        assert_eq!(d.width(f64::NAN, 64), 1);
        // work worth any section: every costed thread
        assert_eq!(d.width(f64::INFINITY, 64), 8);
        assert_eq!(d.width(f64::INFINITY, 3), 3);
        // below the break-even nothing goes wide, at any piece count
        assert!(d.break_even() > 0.0);
        assert_eq!(d.width(d.break_even() * 0.99, usize::MAX), 1);
        assert!(d.width(d.break_even() * 1.01, usize::MAX) > 1);
    }

    #[test]
    fn a_rate_learns_what_an_op_costs() {
        let r = Rate::new(10.0);
        assert_eq!(r.ns_per_op(), 10.0);
        for _ in 0..64 {
            r.learn(1e4, 1e4);
        }
        assert!((r.ns_per_op() - 1.0).abs() < 1e-3, "{}", r.ns_per_op());
        // a run too small to measure is never sampled
        assert!(!(0..64).any(|_| r.sample(1.0)));
        let s = ScalarRates::new(1.0);
        assert_eq!(s.of::<f64>().ns_per_op(), 1.0);
        assert_eq!(
            s.of::<num_complex::Complex<f64>>().ns_per_op(),
            COMPLEX_OPS as f64
        );
    }

    #[test]
    fn rayon_measures_what_a_section_costs_it() {
        let pool = rayon::ThreadPoolBuilder::new()
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
        let pool = rayon::ThreadPoolBuilder::new()
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
        });
    }
}
