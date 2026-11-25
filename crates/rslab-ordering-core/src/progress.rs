//! Progress of an ordering for its caller: a callback in the options of an
//! ordering, called with the vertices ordered so far out of the vertex count,
//! at most [`PROGRESS_STEPS`](crate::constants::PROGRESS_STEPS) times per
//! ordering.

use core::fmt;
use std::panic::RefUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The caller's progress callback: `report(done, n)` with the vertices
/// ordered so far and the vertex count of the graph ordered. It is called
/// from whichever thread makes progress, and with `done == n` once the
/// ordering has placed every vertex. Unwind safe, so options that carry
/// one stay so.
#[derive(Clone)]
pub struct Progress(Arc<dyn Fn(usize, usize) + Send + Sync + RefUnwindSafe>);

impl Progress {
    /// A callback from a closure.
    pub fn new(f: impl Fn(usize, usize) + Send + Sync + RefUnwindSafe + 'static) -> Self {
        Self(Arc::new(f))
    }

    /// Call the callback with `done` of `n` vertices ordered.
    pub fn report(&self, done: usize, n: usize) {
        (self.0)(done, n)
    }
}

impl fmt::Debug for Progress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Progress")
    }
}

/// The vertices of an ordering placed so far, counted from any thread. The
/// callback is called where the count crosses into the next of
/// [`PROGRESS_STEPS`](crate::constants::PROGRESS_STEPS) equal parts of `n`;
/// without a callback nothing is counted.
pub struct Counter<'a> {
    progress: Option<&'a Progress>,
    n: u64,
    done: AtomicU64,
}

impl<'a> Counter<'a> {
    /// A count of `n` vertices reported to `progress`.
    pub fn new(progress: Option<&'a Progress>, n: usize) -> Self {
        Self {
            progress,
            n: n as u64,
            done: AtomicU64::new(0),
        }
    }

    /// `k` more vertices placed.
    #[inline]
    pub fn add(&self, k: usize) {
        if let Some(p) = self.progress {
            let before = self.done.fetch_add(k as u64, Ordering::Relaxed);
            let after = (before + k as u64).min(self.n);
            let part = |d: u64| d * crate::constants::PROGRESS_STEPS / self.n.max(1);
            if part(after) > part(before.min(self.n)) {
                p.report(after as usize, self.n as usize);
            }
        }
    }

    /// Every vertex placed: the rest of the count, so the last call reports
    /// `n` of `n`.
    pub fn finish(&self) {
        if self.progress.is_some() {
            let done = self.done.load(Ordering::Relaxed);
            self.add(self.n.saturating_sub(done) as usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn a_count_calls_at_most_once_per_part_and_ends_at_n() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let p = Progress::new(move |d, n| s.lock().unwrap().push((d, n)));
        let n = 12_345;
        let c = Counter::new(Some(&p), n);
        for _ in 0..n - 7 {
            c.add(1);
        }
        c.finish();
        let seen = seen.lock().unwrap();
        assert!(seen.len() as u64 <= crate::constants::PROGRESS_STEPS);
        assert!(seen.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(seen.last(), Some(&(n, n)));
    }
}
