//! Progress of a long part of a solve in
//! [`PROGRESS_STEPS`](crate::constants::PROGRESS_STEPS) steps: the units done
//! are counted from any thread, and the step reached is returned once per
//! step, for the part's own `rsprobe::progress!` site.

use std::sync::atomic::{AtomicU64, Ordering};

/// The units of a part done so far, out of its total.
pub(crate) struct Steps {
    total: u64,
    done: AtomicU64,
}

impl Steps {
    /// Steps over `total` units; `None` where `active` is false (no
    /// context, or the part's site does not record), so nothing is counted.
    pub(crate) fn new(active: bool, total: u64) -> Option<Steps> {
        (active && total > 0).then(|| Steps {
            total,
            done: AtomicU64::new(0),
        })
    }

    /// `k` more units done: the step reached where they cross into a new one.
    #[inline]
    pub(crate) fn add(&self, k: u64) -> Option<u64> {
        let before = self.done.fetch_add(k, Ordering::Relaxed);
        self.crossed(before, before + k)
    }

    /// `done` units done, a count that only rises: the step reached where it
    /// crosses into a new one.
    pub(crate) fn reach(&self, done: u64) -> Option<u64> {
        let before = self.done.fetch_max(done, Ordering::Relaxed);
        self.crossed(before, done.max(before))
    }

    /// Every unit done: the last step, where it was not reached before.
    pub(crate) fn finish(&self) -> Option<u64> {
        self.reach(self.total)
    }

    fn crossed(&self, before: u64, after: u64) -> Option<u64> {
        let step = |d: u64| d.min(self.total) * crate::constants::PROGRESS_STEPS / self.total;
        (step(after) > step(before)).then(|| step(after))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_step_is_reached_once_and_the_last_is_the_count() {
        let steps = crate::constants::PROGRESS_STEPS;
        let s = Steps::new(true, 12_345).unwrap();
        let mut seen: Vec<u64> = (0..12_000).filter_map(|_| s.add(1)).collect();
        seen.extend(s.reach(12_000));
        seen.extend(s.finish());
        seen.extend(s.finish());
        assert!(seen.windows(2).all(|w| w[0] < w[1]));
        assert!(seen.len() as u64 <= steps);
        assert_eq!(seen.last(), Some(&steps));
        assert!(Steps::new(false, 10).is_none());
    }
}
