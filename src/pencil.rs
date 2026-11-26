//! Matrix families on one layout: `A(alpha) = A0 + alpha*A1 + E`.
//!
//! Shifted systems, the points of a frequency sweep (`G + jw C`) and the
//! stage matrices of implicit time steps (`G + C / (h gamma)`) are members of
//! one family: two real terms on fixed patterns, a scalar `alpha` (real or
//! complex) per member, and a few entries `E` whose values every member
//! brings (terms not affine in `alpha`). [`Pencil`] keeps the terms on the
//! union layout of their patterns, so a member's values are one pass over
//! the slots, written straight into a solver's storage
//! ([`AutoSolver::factor_member`](crate::AutoSolver::factor_member), which
//! keeps the factor of an unchanged member). [`EntryRows`] are a term's
//! entries by rows, for the products with the separate terms.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::constants::SCAN_NS_PER_ENTRY;
use crate::error::RslabError;
use crate::parallel::{chunks_mut, Rayon, ScalarRates, Work};
use crate::scalar::Scalar;
use crate::sparse::layout::CscLayout;

static PASS_RATE: ScalarRates = ScalarRates::new(SCAN_NS_PER_ENTRY);

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn pattern<'a>(
    what: &str,
    (rows, cols): (&'a [usize], &'a [usize]),
) -> Result<(&'a [usize], &'a [usize]), RslabError> {
    if rows.len() != cols.len() {
        return Err(RslabError::InvalidInput(format!(
            "Pencil: {what} has {} rows but {} cols",
            rows.len(),
            cols.len()
        )));
    }
    Ok((rows, cols))
}

fn by_slot(slots: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut entry: Vec<usize> = (0..slots.len()).collect();
    entry.sort_by_key(|&k| slots[k]);
    (entry.iter().map(|&k| slots[k]).collect(), entry)
}

/// The family `A0 + alpha*A1 + E` on one layout: the term values kept in
/// the caller's entry order, `A0` summed on the union, `A1` sorted by union
/// slot.
///
/// Every member's value of a slot sums in one order: `A0`'s entries (entry
/// order), then `alpha` times each `A1` entry (entry order), then the `E`
/// entries (entry order), on any number of threads.
#[derive(Debug)]
pub struct Pencil {
    layout: CscLayout,
    id: u64,
    generation: u64,
    a0: Vec<f64>,
    a1: Vec<f64>,
    base: Vec<f64>,
    a1_slot: Vec<usize>,
    a1_entry: Vec<usize>,
    a1_sorted: Vec<f64>,
    extra_slot: Vec<usize>,
    extra_entry: Vec<usize>,
    a0_rows: EntryRows,
    a1_rows: EntryRows,
}

impl Pencil {
    /// The terms at `(rows, cols)` of an `n x n` matrix in the caller's
    /// entry order (duplicates sum), their values zero; `extra` the entries
    /// whose values every member brings (empty: none). Fails if a part's
    /// lists differ in length or an entry lies outside the matrix.
    pub fn new(
        n: usize,
        a0: (&[usize], &[usize]),
        a1: (&[usize], &[usize]),
        extra: (&[usize], &[usize]),
    ) -> Result<Self, RslabError> {
        let (a0, a1, extra) = (
            pattern("A0", a0)?,
            pattern("A1", a1)?,
            pattern("extra", extra)?,
        );
        let rows: Vec<usize> = (a0.0.iter().chain(a1.0).chain(extra.0)).copied().collect();
        let cols: Vec<usize> = (a0.1.iter().chain(a1.1).chain(extra.1)).copied().collect();
        let layout = CscLayout::new(n, &rows, &cols)?;
        let (n0, n1) = (a0.0.len(), a1.0.len());
        let (a1_slot, a1_entry) = by_slot(&layout.slot()[n0..n0 + n1]);
        let (extra_slot, extra_entry) = by_slot(&layout.slot()[n0 + n1..]);
        Ok(Pencil {
            base: vec![0.0; layout.nnz()],
            layout,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            generation: 0,
            a0: vec![0.0; n0],
            a1: vec![0.0; n1],
            a1_sorted: vec![0.0; n1],
            a1_slot,
            a1_entry,
            extra_slot,
            extra_entry,
            a0_rows: EntryRows::new(n, a0.0, a0.1)?,
            a1_rows: EntryRows::new(n, a1.0, a1.1)?,
        })
    }

    /// The union layout (entry order `A0 ++ A1 ++ extra`), for
    /// [`AutoSymbolic::new`](crate::AutoSymbolic::new) or
    /// [`AutoSymbolic::analyze`](crate::AutoSymbolic::analyze).
    pub fn layout(&self) -> &CscLayout {
        &self.layout
    }

    /// Write the term values in place (entry order), then sum `A0` on the
    /// union and sort `A1` by slot once. A part left unwritten keeps its
    /// values. Every member factored before is a member of other values
    /// after it.
    pub fn update(&mut self, f: impl FnOnce(&mut [f64], &mut [f64])) {
        f(&mut self.a0, &mut self.a1);
        self.base.fill(0.0);
        for (&s, &v) in self.layout.slot().iter().zip(&self.a0) {
            self.base[s] += v;
        }
        for (v, &k) in self.a1_sorted.iter_mut().zip(&self.a1_entry) {
            *v = self.a1[k];
        }
        self.generation += 1;
    }

    /// The values of `A0`, entry order.
    pub fn a0(&self) -> &[f64] {
        &self.a0
    }

    /// The values of `A1`, entry order.
    pub fn a1(&self) -> &[f64] {
        &self.a1
    }

    /// `A0`'s entries by rows, for its products.
    pub fn a0_rows(&self) -> &EntryRows {
        &self.a0_rows
    }

    /// `A1`'s entries by rows, for its products.
    pub fn a1_rows(&self) -> &EntryRows {
        &self.a1_rows
    }

    /// The number of extra entries a member brings.
    pub fn extras(&self) -> usize {
        self.extra_entry.len()
    }

    pub(crate) fn key(&self) -> (u64, u64) {
        (self.id, self.generation)
    }

    /// Member `alpha`'s values, one per union slot, into `out` (a solver's
    /// [`values_mut`](crate::AutoSolver::values_mut)): `A0 + alpha*A1 +
    /// E`, `extra` the values of the `E` entries in their entry order. The
    /// slots run in chunks on the pool where their work is worth it, every
    /// slot summed in the order of the type docs.
    pub fn fill<T: Scalar>(&self, alpha: T, extra: &[T], out: &mut [T]) {
        assert_eq!(
            out.len(),
            self.base.len(),
            "Pencil::fill: one value per slot"
        );
        assert_eq!(
            extra.len(),
            self.extras(),
            "Pencil::fill: one value per extra entry"
        );
        let ops = (self.base.len() + self.a1_slot.len() + self.extra_slot.len()) as f64;
        let work = Work::of(ops, PASS_RATE.of::<T>());
        chunks_mut::<Rayon, _>(out, 1, work, |s0, part| {
            let s1 = s0 + part.len();
            for (o, &b) in part.iter_mut().zip(&self.base[s0..s1]) {
                *o = T::from_real(b);
            }
            let within = |slots: &[usize]| {
                slots.partition_point(|&s| s < s0)..slots.partition_point(|&s| s < s1)
            };
            for i in within(&self.a1_slot) {
                let o = &mut part[self.a1_slot[i] - s0];
                *o = *o + alpha * T::from_real(self.a1_sorted[i]);
            }
            for i in within(&self.extra_slot) {
                let o = &mut part[self.extra_slot[i] - s0];
                *o = *o + extra[self.extra_entry[i]];
            }
        });
    }
}

/// A pattern's entries by rows, entry order within a row: products of value
/// sets given in that entry order. The rows run in chunks on the pool where
/// their work is worth it, each row summed in entry order, so the result
/// has the same bits on every thread count.
#[derive(Clone, Debug)]
pub struct EntryRows {
    start: Vec<usize>,
    entry: Vec<usize>,
    cols: Vec<usize>,
}

impl EntryRows {
    /// The entries at `(rows[k], cols[k])` of an `n x n` matrix. Fails if the
    /// lists differ in length or an entry lies outside the matrix.
    pub fn new(n: usize, rows: &[usize], cols: &[usize]) -> Result<Self, RslabError> {
        if rows.len() != cols.len() {
            return Err(RslabError::InvalidInput(format!(
                "EntryRows: {} rows but {} cols",
                rows.len(),
                cols.len()
            )));
        }
        if let Some((&r, &c)) = (rows.iter().zip(cols)).find(|&(&r, &c)| r >= n || c >= n) {
            return Err(RslabError::InvalidInput(format!(
                "EntryRows: index ({r}, {c}) out of bounds for {n}x{n}"
            )));
        }
        let mut start = vec![0usize; n + 1];
        for &r in rows {
            start[r + 1] += 1;
        }
        for r in 0..n {
            start[r + 1] += start[r];
        }
        let mut next = start[..n].to_vec();
        let mut entry = vec![0usize; rows.len()];
        for (k, &r) in rows.iter().enumerate() {
            entry[next[r]] = k;
            next[r] += 1;
        }
        let cols = entry.iter().map(|&k| cols[k]).collect();
        Ok(EntryRows { start, entry, cols })
    }

    /// The dimension `n`.
    pub fn n(&self) -> usize {
        self.start.len() - 1
    }

    /// The number of entries.
    pub fn entries(&self) -> usize {
        self.entry.len()
    }

    fn rows_into<T: Scalar>(&self, y: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
        let ops = (self.entry.len() + self.n()) as f64;
        let work = Work::of(ops, PASS_RATE.of::<T>());
        chunks_mut::<Rayon, _>(y, 1, work, |r0, part| {
            for (r, o) in (r0..).zip(part) {
                f(r, o);
            }
        });
    }

    /// `y = A x`, `A` the entries with the values `vals` (entry order).
    pub fn mul_into<T: Scalar>(&self, vals: &[f64], x: &[T], y: &mut [T]) {
        self.check(vals.len(), x.len(), y.len());
        self.rows_into(y, |r, o| {
            let p = self.start[r]..self.start[r + 1];
            let mut s = T::zero();
            for (&k, &c) in self.entry[p.clone()].iter().zip(&self.cols[p]) {
                s = s + T::from_real(vals[k]) * x[c];
            }
            *o = s;
        });
    }

    /// `y = |A| |x|`, entry by entry: duplicate positions are not merged,
    /// so the sum bounds the rounding of a sum over the entries.
    pub fn abs_mul_into(&self, vals: &[f64], x: &[f64], y: &mut [f64]) {
        self.check(vals.len(), x.len(), y.len());
        self.rows_into(y, |r, o| {
            let p = self.start[r]..self.start[r + 1];
            let mut s = 0.0;
            for (&k, &c) in self.entry[p.clone()].iter().zip(&self.cols[p]) {
                s += vals[k].abs() * x[c].abs();
            }
            *o = s;
        });
    }

    fn check(&self, vals: usize, x: usize, y: usize) {
        assert_eq!(vals, self.entries(), "EntryRows: one value per entry");
        assert!(
            x == self.n() && y == self.n(),
            "EntryRows: vectors of length n"
        );
    }
}
