//! Progress of the analysis parts that run once per ordering candidate: the
//! orderings and the structures (elimination tree and column counts) under
//! them.
//!
//! A part counts the candidates the analysis waits for, each from 0 to
//! [`PROGRESS_STEPS`]: the candidates of the race's first stage, and the
//! nested dissection where the race may run it. The dissection counts as one
//! candidate, the mean of the seeds that count; a seed started speculatively
//! is not counted until the first stage has decided that it counts, and not
//! at all where it is stopped (the part then counts the dissection as done).
//! An ordering alone is a part of one candidate. An ordering given to the
//! analysis (a permutation, or the automatic route's minimum-degree
//! ordering) is no candidate of the ordering part, only of the structure.

use crate::constants::PROGRESS_STEPS;
use crate::progress::Steps;
use rslab_ordering_core::Progress;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

const UNDECIDED: usize = usize::MAX;

pub(super) struct Part {
    node: rsprobe::Handle,
    emit: fn(u64),
    steps: Steps,
    first: usize,
    slots: Vec<AtomicU64>,
    nd: bool,
    seeds: AtomicUsize,
}

impl Part {
    fn new(active: bool, emit: fn(u64), first: usize, nd: bool, seeds: usize) -> Option<Arc<Part>> {
        let total = (first + usize::from(nd)) as u64 * PROGRESS_STEPS;
        let steps = Steps::new(active, total)?;
        Some(Arc::new(Part {
            node: rsprobe::current(),
            emit,
            steps,
            first,
            slots: (0..first + seeds).map(|_| AtomicU64::new(0)).collect(),
            nd,
            seeds: AtomicUsize::new(if nd { UNDECIDED } else { 0 }),
        }))
    }

    pub(super) fn slot(self: &Arc<Self>, slot: usize) -> Progress {
        let part = self.clone();
        Progress::new(move |done, n| {
            let steps = (done.min(n) as u64 * PROGRESS_STEPS).checked_div(n as u64);
            part.slots[slot].fetch_max(steps.unwrap_or(PROGRESS_STEPS), Ordering::Relaxed);
            part.update();
        })
    }

    pub(super) fn decide(&self, seeds: usize) {
        self.seeds.store(seeds, Ordering::Relaxed);
        self.update();
    }

    pub(super) fn finish(&self) {
        if let Some(step) = self.steps.finish() {
            let _node = self.node.enter();
            (self.emit)(step);
        }
    }

    fn update(&self) {
        let at = |s: &AtomicU64| s.load(Ordering::Relaxed);
        let first: u64 = self.slots[..self.first].iter().map(at).sum();
        let nd = match self.seeds.load(Ordering::Relaxed) {
            UNDECIDED => 0,
            0 => PROGRESS_STEPS,
            k => {
                self.slots[self.first..self.first + k]
                    .iter()
                    .map(at)
                    .sum::<u64>()
                    / k as u64
            }
        };
        let done = first + if self.nd { nd } else { 0 };
        if let Some(step) = self.steps.reach(done) {
            let _node = self.node.enter();
            (self.emit)(step);
        }
    }
}

fn ordering_event(step: u64) {
    rsprobe::progress!(step, PROGRESS_STEPS; "ordering");
}

fn structure_event(step: u64) {
    rsprobe::progress!(step, PROGRESS_STEPS; "elimination tree");
}

pub(super) struct Parts {
    pub ordering: Option<Arc<Part>>,
    pub structure: Option<Arc<Part>>,
}

impl Parts {
    /// Over the first stage's `ordered` candidates (the orderings it runs)
    /// and `structures` (those plus a given one) and, where `nd`, the
    /// dissection with up to `seeds` seeds.
    pub fn new(ordered: usize, structures: usize, nd: bool, seeds: usize) -> Parts {
        let active = rsprobe::active!();
        Parts {
            ordering: Part::new(active, ordering_event, ordered, nd, seeds),
            structure: Part::new(active, structure_event, structures, nd, seeds),
        }
    }

    /// The callbacks of the first stage's ordered candidate `i`.
    pub fn stage(&self, i: usize) -> Reports {
        Reports {
            ordering: self.ordering.as_ref().map(|p| p.slot(i)),
            structure: self.structure.as_ref().map(|p| p.slot(i)),
        }
    }

    /// The callbacks of the given ordering: its structure, the last of the
    /// first stage.
    pub fn known(&self) -> Reports {
        Reports {
            ordering: None,
            structure: self.structure.as_ref().map(|p| p.slot(p.first - 1)),
        }
    }

    /// The callbacks of dissection seed `i`.
    pub fn seed(&self, i: usize) -> Reports {
        Reports {
            ordering: self.ordering.as_ref().map(|p| p.slot(p.first + i)),
            structure: self.structure.as_ref().map(|p| p.slot(p.first + i)),
        }
    }

    pub fn decide(&self, seeds: usize) {
        self.ordering
            .iter()
            .chain(&self.structure)
            .for_each(|p| p.decide(seeds));
    }

    pub fn finish(&self) {
        self.ordering
            .iter()
            .chain(&self.structure)
            .for_each(|p| p.finish());
    }
}

#[derive(Default)]
pub(super) struct Reports {
    pub ordering: Option<Progress>,
    pub structure: Option<Progress>,
}
