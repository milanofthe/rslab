pub mod column_counts;
pub mod ldlt_compress;
mod ordering_graph;
pub mod small_leaf;
pub mod supernode;
pub(crate) mod supervariables;

use crate::error::RslabError;
use crate::ordering::amd::permute_pattern;
use crate::ordering::elimination_tree::EliminationTree;
use crate::ordering::postorder::{biased_postorder, postorder};
use crate::sparse::csc::{CscMatrix, CscPattern};
use ordering_graph::OrderingGraph;

pub use column_counts::{column_counts_gnp, column_counts_permuted, total_factor_nnz};
pub use ldlt_compress::{build_supermap, compress_pattern, expand_permutation, SuperMap};
pub use small_leaf::{find_small_leaf_groups, SmallLeafGroup, SmallLeafParams};
pub use supernode::{
    find_supernodes, pick_amalgamation_strategy, supernode_parents, AmalgamationStrategy,
    OrderingPreprocess, RelaxAmalgamation, Supernode, SupernodeParams,
    AUTO_MULTI_CHILD_FRAC_THRESHOLD,
};

/// Which fill-reducing ordering to use in [`symbolic_factorize_with_method`].
///
/// All methods produce a permutation; the downstream postorder
/// composition, etree construction, column counts, supernode detection,
/// and memory planning are identical regardless of method.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OrderingMethod {
    /// Approximate Minimum Degree (`rslab-amd` crate: approximate
    /// external degree with aggressive element absorption and
    /// supervariable detection, per Amestoy/Davis/Duff 1996+2004).
    /// Default. Matches SuiteSparse/faer on the oracle fixture suite.
    ///
    /// The simplified exact-external-degree implementation at
    /// `src/ordering/amd.rs` remains on disk as a reference for the
    /// algorithm's skeleton but is no longer reachable from the
    /// symbolic pipeline. See
    /// `dev/journal/2026-04-18-03.org` for the retirement evidence
    /// (34-matrix bakeoff: geomean fill tied on parity, crate
    /// 17-23% better and 18-88x faster on large).
    #[default]
    Amd,
    /// Approximate Minimum Fill (`rslab-amf` crate: HAMF4 variant
    /// of Amestoy 1999 - quotient-graph elimination scored by
    /// approximate fill `RMF(i) = (deg(i)*(deg(i)-1+2*degme) -
    /// WF(i)) / (nv(i)+1)` rather than approximate degree).
    /// Same downstream pipeline as `Amd`.
    ///
    /// Default for `n <= 10_000` per `pick_default_method`,
    /// matching MUMPS's `ana_set_ordering.F` rule for SYM=2 small
    /// matrices. Validated against MUMPS HAMF4 on the 183_293-
    /// sidecar corpus by `tests/amf_corpus_oracle.rs`: rslab nnz_L
    /// is within 1.10x MUMPS HAMF4 nnz_L on 183_277 matrices, with
    /// CHARDIS1_0000 the lone documented metric-divergence skip.
    Amf,
    /// rslab-metis multilevel nested dissection: one run with the default
    /// seed (METIS semantics), the two sides of every bisection ordered in
    /// parallel on the rayon pool (the result depends on the seed only, not
    /// on the thread count). The ordering race (`Auto` on large systems,
    /// `AutoRace`) additionally runs a small seed ensemble and keeps the
    /// lowest-fill result when the predicted factorization work is large
    /// enough to pay for the extra runs (`ND_SEED_RACE_MIN_FLOPS`) and the
    /// analysis serves repeated factorizations (`AnalysisUse::Repeated`).
    MetisND,
    /// Reverse Cuthill-McKee band/profile-reducing ordering
    /// (`rslab-ordering-core`: George-Liu degree-sorted BFS from a
    /// pseudo-peripheral start, reversed; Cuthill & McKee 1969,
    /// George & Liu 1981). Targets banded / structured matrices
    /// (stencils, structured FEM) where nested dissection
    /// over-separates and minimum-degree scatters fill: the band
    /// factor has less fill and factors faster. Cheap (one BFS pass).
    /// An [`AutoRace`](OrderingMethod::AutoRace) candidate; also reachable explicitly.
    Rcm,
    /// Adaptive dispatcher: picks a concrete method per-matrix from
    /// cheap pattern features (n and average degree nnz/n).
    ///
    /// Issue #50 plus its F11 follow-up (2026-05-23) collapsed the
    /// per-shape branches to one very-large-and-sparse catch on top
    /// of `pick_default_method`:
    ///   - very-large-and-sparse (n > 100_000, full nnz/n < 5) -> `Amd`
    ///   - everything else delegates to `pick_default_method`
    ///     (`n <= 10_000 -> Amf`, `n > 10_000 -> MetisND`).
    ///
    /// **Opt-in only.** The 154k-matrix IPM bench (2026-04-18) showed
    /// `Auto` regresses sparse factor/MUMPS geomean from 0.44 (AMD)
    /// to 0.58 because the (pre-F11) small-and-sparse branch routed
    /// thousands of n<500 IPM iteration dumps to KaHIP, where K1 +
    /// multilevel setup cost 2-3x per call vs AMD. That branch is
    /// gone - `Auto`'s small-and-sparse path is now AMF via the
    /// default - but the original `Auto` warning is preserved here
    /// since the historical-bench regression evidence remains a
    /// reason to default to `Amd` outside known IPM workloads.
    ///
    /// Use `Auto` only when the workload is known to be dominated by
    /// large or `cresc132`-class matrices where the per-call setup
    /// cost amortizes. The default `symbolic_factorize` keeps `Amd`.
    /// See `dev/tried-and-rejected.md` for the full evidence.
    ///
    /// Applying `Auto` to `Auto` loops once through the dispatcher and
    /// then runs the chosen concrete method.
    Auto,
    /// Race-based dispatcher: runs the cheap symbolic *prefix*
    /// (ordering, postorder, etree, column counts) on each concrete
    /// candidate in {`Amd`, `MetisND`, `Rcm`}
    /// and finishes (supernode detection, memory plan) only the one
    /// with the smallest exact factor nnz (feral #144 port).
    ///
    /// Unlike [`Auto`](OrderingMethod::Auto), which guesses the winner from cheap pattern
    /// features, `AutoRace` measures the actual symbolic outcome. Cost
    /// is Nx the prefix plus ONE pipeline tail (the previous
    /// full-pipeline race paid the tail for every candidate), paid
    /// once per problem because symbolic factorization is reused across
    /// numeric refactorizations with the same sparsity pattern.
    ///
    /// Motivated by issue #8: on `pinene_3200_0009` the
    /// `pick_default_method` heuristic picks `MetisND` (88 s numeric
    /// factor), but `Amd` factors in 19.5 s on the same matrix - a 4.5x
    /// win that the cheap predicate misses. Racing eliminates the
    /// guess: whichever candidate wins on this matrix is the one we
    /// use, no calibration required.
    ///
    /// Candidates that fail (e.g. external crate returns an error) are
    /// skipped; the race succeeds as long as at least one candidate
    /// produces a valid symbolic factorization. `resolved_method` on
    /// the returned `SymbolicFactorization` records the actual winner.
    AutoRace,
}

/// Resolve an `Auto` ordering to a concrete method from cheap pattern
/// features. Non-`Auto` inputs pass through unchanged.
///
/// The rule set adds shape-bakeoff branches on top of
/// [`pick_default_method`]:
///   - very-large-and-sparse (`n > 100_000`, full avg_deg < 5.0) -> `Amd`
///   - arrow/bordered (issue #64): whenever the size rule would pick
///     `MetisND` (`n > 10_000`) but [`is_arrow_bordered`] detects a
///     dense border concentrating the nonzeros, override to `Amf`.
///   - thin-large (issues #67 + #73): whenever the size rule would still
///     pick `MetisND` (after the avg_deg<5 -> AMD and arrow -> AMF catches),
///     override to `Amf` at every `n`. Corpus A/Bs on real factor+solve
///     wall-time found AMF wins or ties MetisND across the whole population:
///     36/36 in the `(10_000, 100_000]` band (#67) and every measured
///     `n > 100_000 && avg_deg >= 5` non-arrow family (#73), including the
///     one matrix (nql180) where MetisND has smaller fill but AMF is still
///     2x faster on the real factor+solve.
///
/// Anything else delegates to `pick_default_method`. `symbolic_factorize`
/// routes through `Auto`, so the no-arg default and `Auto` resolve to the
/// same concrete method on every matrix (issue #64 unified the two paths;
/// previously the no-arg default skipped the very-large-and-sparse and
/// arrow catches).
///
/// The large-and-sparse branch swap from `ScotchND` to `Amd` is the
/// issue #50 fix (2026-05-23). On `powerflow22` (n=2.8 M,
/// full_avg_deg ~ 3.7) the prior ScotchND route took 113.8 s
/// symbolic (15.8 M nnz_L); MetisND was 117.4 s (20.5 M nnz_L); AMD
/// was 55 s (10.4 M nnz_L). The ScotchND advantage at very large n
/// was load-bearing against the same BK pivoting cascade that
/// motivated `pick_default_method`'s chain catches; issue #46 (see
/// `pick_default_method`'s doc comment) eliminated that amplifier in
/// May 2026 and removed the justification for routing very-large
/// sparse matrices through nested dissection at all. Numeric
/// inventory: `dev/research/issue-50-numeric-inventory.csv` shows
/// the IPM corpus's [100k, 200k) bucket has AMD/MetisND num_nnz_l
/// ratio 1.00 on both representatives. See
/// `dev/research/issue-50-metisnd-symbolic-cost.md` section F7-F8.
///
/// The small-and-sparse branch (`n < 10_000 && avg_deg < 15 ->
/// KahipND`) was deleted by the F11 side finding from issue #50
/// (2026-05-23). The corpus inventory in
/// `dev/research/small-sparse-inventory.csv` (838 IPM-corpus
/// matrices factored under AMD/AMF/MetisND/KahipND) shows AMF
/// dominates this population: AMF wins 169/838 per-matrix
/// (vs KahipND's 16), aggregate AMF fill is 0.87x AMD vs KahipND's
/// 0.98x, aggregate AMF time is 0.83x AMD vs KahipND's 0.99x. After
/// deletion these matrices fall through to `pick_default_method`'s
/// `n <= 10_000 -> Amf` rule. (KahipND itself was removed in the 2026-08
/// consolidation; the bakeoff evidence stays in dev/research.)
///
/// `pattern` is expected to be the matrix's full-symmetric pattern (the
/// shape produced by `CscMatrix::symmetric_pattern`); the
/// `pick_default_method` call below converts to a stored-nnz
/// equivalent assuming the diagonal is included.
fn choose_adaptive(pattern: &CscPattern, method: OrderingMethod) -> OrderingMethod {
    if method != OrderingMethod::Auto {
        return method;
    }
    let n = pattern.n;
    let full_nnz = pattern.row_idx.len();
    if n == 0 {
        return OrderingMethod::Amd;
    }
    let avg_deg = full_nnz as f64 / n as f64;
    if n > 100_000 && avg_deg < 5.0 {
        return OrderingMethod::Amd;
    }
    // Convert full-symmetric nnz back to a stored-lower-triangle
    // equivalent so `pick_default_method`'s thresholds (calibrated on
    // stored nnz) apply: stored = (full + n) / 2 when the diagonal is
    // included once on each row of the symmetric pattern.
    let stored_nnz = (full_nnz + n) / 2;
    let base = pick_default_method(n, stored_nnz);
    // Issue #64 arrow/bordered-KKT catch. The size-only
    // `pick_default_method` routes every `n > 10_000` matrix to MetisND,
    // but nested dissection cannot isolate a dense border (a handful of
    // very-high-degree columns concentrating the nonzeros) and the LDL^T
    // factor blows up ~7-9x vs AMF/AMD. Override MetisND -> AMF on the
    // arrow signature. Only the would-be-MetisND decision is touched;
    // the `n <= 10_000 -> AMF` and `n > 100_000 && avg_deg < 5 -> AMD`
    // (returned above) paths are untouched. See
    // `dev/research/issue-64-arrow-bordered-ordering.md`.
    if base == OrderingMethod::MetisND && is_arrow_bordered(pattern) {
        return OrderingMethod::Amf;
    }
    // Issue #67 + #73 thin-large catch. The size-only `pick_default_method`
    // routes every `n > 10_000` matrix to MetisND, but corpus A/Bs on real
    // factor+solve wall-time (not nnz_L alone) show AMF wins or ties MetisND
    // across the whole would-be-MetisND population:
    //   - #67: 36/36 in-scope `(10_000, 100_000]` families, worst case 0.99x
    //     (noise), median ~1.5x, up to 4.5x.
    //   - #73: the `n > 100_000 && avg_deg >= 5` non-arrow families - dtoc2
    //     2.49x, pinene 1.18x, cont5_1_l 2.75x, nql180 2.05x, YATP1NE 2.13x -
    //     AMF wins factor+solve on every measured matrix. Critically nql180 is
    //     the lone case where MetisND has *smaller* symbolic fill (nnz_L 0.98x)
    //     yet AMF is still 2.05x faster on real factor+solve, so fill (nnz_L /
    //     flop_proxy) is NOT a reliable speed predictor and a fill-guarded race
    //     would wrongly demote nql180. The simple unconditional reroute is the
    //     one the evidence supports - see `dev/research/issue-73-n100k-thin-
    //     regime.md` and `dev/research/issue-67-thin-large-ordering.md`.
    //
    // MetisND's separators do not pay off on these uniformly-thin discretization
    // patterns, and its symbolic ordering is 2-5x more expensive than AMF's, so
    // racing the two is a net loss. Route every would-be-MetisND decision to AMF
    // outright. Only the would-be-MetisND decision is touched; the earlier
    // `n > 100_000 && avg_deg < 5 -> Amd` (#50 powerflow) and arrow -> AMF (#64)
    // catches fire first and are untouched.
    if base == OrderingMethod::MetisND {
        return OrderingMethod::Amf;
    }
    base
}

/// Detect the **arrow / bordered-KKT** sparsity signature on a full
/// symmetric pattern: a *small set* of very-high-degree "border" columns
/// carrying a *large share* of the nonzeros, over an otherwise thin body.
///
/// This is the structural fingerprint of an IPM augmented system whose
/// inequality block has a few dense constraint rows (issue #64: r05's
/// iter-0 KKT has 171 of 14 842 columns at degree 502, carrying 38.5% of
/// the nonzeros). On such patterns nested dissection smears the dense
/// border across its separators and the factor blows up, whereas
/// minimum-degree / min-fill orderings (AMD/AMF) defer the border to the
/// end of the elimination where it costs one dense trailing block.
///
/// Predicate (all O(n), allocation-free), on the full symmetric pattern:
///
/// ```text
/// avg_deg   = full_nnz / n
/// heavy_thr = max(HEAVY_DEG_FLOOR, HEAVY_AVG_MULT * avg_deg)
/// heavy     = { columns with degree > heavy_thr }
/// arrow iff  1 <= heavy.count < ARROW_COUNT_FRAC * n   (a *small* set)
///        AND heavy.nnz >= ARROW_NNZ_SHARE * full_nnz   (a *large* share)
/// ```
///
/// The `ARROW_NNZ_SHARE` guard is the discriminating test: it fires on
/// r05 (38.5% share) and rejects bcsstk38 (0.3% share, despite two
/// degree-614 columns). The `ARROW_COUNT_FRAC` guard rejects "many hub"
/// patterns where a large fraction of columns are high-degree (the matrix
/// is then just dense and nested dissection is appropriate). Uniformly
/// thin matrices (PoissonControl, powerflow22, bratu3d, cont-201) have no
/// column above `heavy_thr` and are never flagged. Calibration and the
/// false-positive table are in
/// `dev/research/issue-64-arrow-bordered-ordering.md`.
fn is_arrow_bordered(pattern: &CscPattern) -> bool {
    /// A "heavy" column has degree above this absolute floor regardless
    /// of `avg_deg`, so genuinely dense small matrices (high uniform
    /// degree) are not flagged.
    const HEAVY_DEG_FLOOR: usize = 64;
    /// ...or above this multiple of the average degree.
    const HEAVY_AVG_MULT: f64 = 8.0;
    /// The heavy set must be a *handful* of columns: strictly fewer than
    /// this fraction of `n`.
    const ARROW_COUNT_FRAC: f64 = 0.05;
    /// ...that *concentrate* at least this fraction of the nonzeros.
    const ARROW_NNZ_SHARE: f64 = 0.20;

    let n = pattern.n;
    if n == 0 {
        return false;
    }
    let full_nnz = pattern.row_idx.len();
    if full_nnz == 0 {
        return false;
    }
    let avg_deg = full_nnz as f64 / n as f64;
    let heavy_thr = (HEAVY_AVG_MULT * avg_deg).ceil() as usize;
    let heavy_thr = heavy_thr.max(HEAVY_DEG_FLOOR);

    let mut heavy_count = 0usize;
    let mut heavy_nnz = 0usize;
    for j in 0..n {
        let deg = pattern.col_ptr[j + 1] - pattern.col_ptr[j];
        if deg > heavy_thr {
            heavy_count += 1;
            heavy_nnz += deg;
        }
    }

    if heavy_count == 0 {
        return false;
    }
    let count_ok = (heavy_count as f64) < ARROW_COUNT_FRAC * n as f64;
    let share_ok = (heavy_nnz as f64) >= ARROW_NNZ_SHARE * full_nnz as f64;
    count_ok && share_ok
}

/// The complete output of symbolic factorization.
///
/// Produced before any numeric work begins. Contains everything needed
/// to allocate memory and drive the numeric factorization.
#[derive(Debug)]
pub struct SymbolicFactorization {
    /// Matrix dimension.
    pub n: usize,

    /// Fill-reducing permutation (new-to-old mapping).
    /// Column `perm[k]` of the original matrix becomes column k.
    pub perm: Vec<usize>,

    /// Inverse permutation (old-to-new mapping).
    pub perm_inv: Vec<usize>,

    /// Supernodes in postorder (children before parents).
    pub supernodes: Vec<Supernode>,

    /// Estimated total NNZ in the L factor across all supernodes.
    pub factor_nnz_estimate: usize,

    /// Slack factor applied to factor_nnz_estimate. Default 1.2.
    pub factor_slack: f64,

    /// For each supernode: the size (in f64s) of its contribution block.
    pub contrib_sizes: Vec<usize>,

    /// Peak contribution pool depth (sum of all live contribution blocks
    /// at the deepest point of the tree traversal).
    pub peak_contrib_bytes: usize,

    /// Elimination tree of the permuted matrix.
    pub etree: EliminationTree,

    /// Full symmetric pattern of the permuted matrix.
    pub permuted_pattern: CscPattern,

    /// Column counts of L.
    pub col_counts: Vec<usize>,

    /// Phase 2.9 small-leaf-subtree groups (`dev/plans/phase-2.9-
    /// small-leaf-subtree.md`). Populated unconditionally at
    /// symbolic time; used at numeric time only when
    /// `NumericParams::small_leaf == SmallLeafBatch::On`.
    pub small_leaf_groups: Vec<SmallLeafGroup>,

    /// For each supernode index, `Some(g)` if the supernode is a
    /// member of `small_leaf_groups[g]`, else `None`. Length
    /// `supernodes.len()`.
    pub snode_group: Vec<Option<usize>>,

    /// Concrete ordering method actually dispatched. Records the
    /// `OrderingMethod::Auto -> AMD/AMF/MetisND`
    /// resolution made by `choose_adaptive`. For non-`Auto` callers
    /// this is identical to the requested method.
    pub resolved_method: OrderingMethod,
    /// Concrete amalgamation strategy actually used.
    /// `AmalgamationStrategy::Auto` is resolved by
    /// `pick_amalgamation_strategy` before supernode detection; this
    /// field records the resolved value.
    pub resolved_amalgamation: supernode::AmalgamationStrategy,
    /// Concrete ordering preprocessor actually used.
    /// `OrderingPreprocess::Auto` is resolved by
    /// `pick_ordering_preprocess`; this field records `None` or
    /// `LdltCompress` after that dispatch.
    pub resolved_preprocess: supernode::OrderingPreprocess,
}

/// Size-only base ordering rule from cheap matrix dimensions (no pattern
/// walk). Narrow on purpose - see comment on `Auto` for why a broad
/// dispatcher regressed the IPM bench. `choose_adaptive` calls this for
/// the bulk of patterns, then layers the pattern-aware catches on top
/// (very-large-and-sparse -> AMD; arrow/bordered -> AMF, issue #64).
///
/// Current rule (mirrors MUMPS's `ana_set_ordering.F` AMF-vs-METIS
/// heuristic):
///   - `n == 0`                                        -> `Amd`
///     (avoids /0 and external-crate weirdness on the empty pattern)
///   - `n <= 10_000`                                   -> `Amf`
///     (MUMPS-style "small symmetric" rule: HAMF4 fill metric is
///     within 1.10x of MUMPS HAMF4 on 183_277 of 183_293 sidecar'd
///     matrices in `tests/amf_corpus_oracle.rs`, and the in-tree
///     audit (`diag_amf_vs_amd`) shows AMF strictly better than AMD
///     on 83/782 matrices, tied on 589, AMD better on 110, geomean
///     ratio 1.003. ORBIT2_0000 alone goes from AMD's 1.4M nnz_L
///     down to AMF's 32_105.)
///   - everything else (`n > 10_000`)                  -> `MetisND`,
///     but note this base decision is **always rerouted to AMF** by the
///     issue #67/#73 catches in `choose_adaptive` (measured: AMF wins or
///     ties MetisND on real factor+solve across the whole would-be-MetisND
///     population, and MetisND's symbolic is 2-5x more expensive). It is
///     kept here so the reroute stays a visible, separately-documented
///     decision rather than being silently folded in.
///
/// `nnz` here is the matrix's *stored* nnz (lower triangle for
/// symmetric matrices), not the symmetric pattern's.
///
/// Issue #50 (2026-05-23) deleted two prior escape hatches:
///   - `n >= 5000 && nnz/n < 6 -> MetisND` (bordered-KKT catch, CRESC132);
///   - `n >= 2000 && nnz/n < 4 -> MetisND` (chain-pattern catch,
///     CHAINWOO/HYDROELL/DIXMAANH/VESUVIO).
///
/// Both were calibrated on 2026-04-27 against a Bunch-Kaufman
/// pivoting cascade that fattened the AMD-ordered factor by up to
/// 7.5x on CHAINWOO_0000 and produced a near-dense root frontal on
/// CRESC132_0000. Issue #46's fixes (`42434a5` fine-grained delayed
/// pivoting, `070840b` two-tier 2x2 partner selection) eliminated
/// the amplifier in May 2026: CHAINWOO_0000 now produces 22.9k
/// num_nnz_l with AMD vs the 2.10M it produced before, and the
/// numeric inventory in `dev/research/issue-50-numeric-inventory.csv`
/// shows zero of 250 chain-catch-class corpus matrices have
/// AMD/MetisND num_nnz_l ratio >= 1.5x. The catches now route 113-s
/// nested-dissection symbolic on `powerflow22` (n=2.8 M, stored
/// avg_deg ~ 2.4) where AMD does the same job in 55 s with smaller
/// fill. See `dev/research/issue-50-metisnd-symbolic-cost.md` section F7-F8.
fn pick_default_method(n: usize, _stored_nnz: usize) -> OrderingMethod {
    if n == 0 {
        return OrderingMethod::Amd;
    }
    if n <= 10_000 {
        OrderingMethod::Amf
    } else {
        OrderingMethod::MetisND
    }
}

/// Resolve [`OrderingPreprocess::Auto`] to a concrete preprocessor
/// choice based on cheap O(nnz) shape predicates.
///
/// Returns [`OrderingPreprocess::LdltCompress`] when two conditions hold:
///
/// 1. `n >= MIN_N_FOR_COMPRESSION` (size floor). Below this, numeric
///    factor time is in the sub-ms range and the ~100-400 us compression
///    symbolic overhead dominates. Calibrated from the 154 588-matrix
///    bench: geomean regressed 0.36 -> 0.48 with unconditional
///    compression, driven by small-matrix symbolic overhead.
///
/// 2. `low_degree_cols / n >= LOW_DEGREE_THRESHOLD` (arrow-KKT
///    signature). Columns with stored degree <= 2 (the diagonal plus at
///    most one off-diagonal) are the structural fingerprint of IPM KKT
///    slack blocks (`IpStdAugSystemSolver.cpp:250-305`: `sum_s + delta_s I`
///    coupled to the d-row by a single identity off-diagonal). Many
///    such columns means the MC64 matching has abundant 2-cycle
///    structure for compression to exploit. This broadens the
///    `diag_only / n` predicate from `pick_scaling_strategy` because
///    Ipopt slack columns are degree-2, not degree-1.
///
/// Otherwise returns [`OrderingPreprocess::None`].
///
/// Parallels `crate::scaling::pick_scaling_strategy` in spirit.
/// Both predicates are O(nnz) and allocation-free.
///
/// No published compression-benefit predictor exists in the MUMPS /
/// SPRAL literature (see consult of 2026-04-23). These thresholds are
/// calibrated against the rslab corpus and documented in
/// `dev/journal/2026-04-23-02.org`.
pub fn pick_ordering_preprocess(matrix: &CscMatrix) -> OrderingPreprocess {
    const MIN_N_FOR_COMPRESSION: usize = 128;
    const LOW_DEGREE_THRESHOLD: f64 = 0.30;

    let n = matrix.n;
    if n < MIN_N_FOR_COMPRESSION {
        return OrderingPreprocess::None;
    }

    let mut low_degree = 0usize;
    for j in 0..n {
        let nnz_col = matrix.col_ptr[j + 1] - matrix.col_ptr[j];
        if nnz_col <= 2 {
            low_degree += 1;
        }
    }

    if low_degree as f64 / n as f64 >= LOW_DEGREE_THRESHOLD {
        OrderingPreprocess::LdltCompress
    } else {
        OrderingPreprocess::None
    }
}

/// Perform symbolic factorization of a sparse symmetric matrix.
///
/// Picks the fill-reducing ordering adaptively via [`OrderingMethod::Auto`]
/// (resolved by `choose_adaptive`): AMD for very-large-and-sparse
/// (`n > 100_000`, avg degree `< 5`), **AMF for everything else**, the
/// issue #67/#73 corpus A/Bs rerouted every would-be-MetisND decision to
/// AMF, so `Auto` never resolves to nested dissection. MetisND is reachable
/// only explicitly, via [`OrderingMethod::AutoRace`], or through the
/// `LdltSolver::tuned` nested-dissection bakeoff. Routing through `Auto`
/// keeps this no-arg default and the explicit `Auto` caller in exact
/// agreement (issue #64). Callers who want a specific ordering with no
/// dispatcher should call `symbolic_factorize_with_method` with an explicit
/// `OrderingMethod`.
///
/// Steps:
/// 1. Pick fill-reducing ordering (resolved from `Auto` by `choose_adaptive`)
/// 2. Build elimination tree of the permuted matrix
/// 3. Compute column counts (fill prediction)
/// 4. Detect and amalgamate supernodes
/// 5. Compute MemoryPlan (factor NNZ, contribution sizes, peak memory)
pub fn symbolic_factorize(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
) -> Result<SymbolicFactorization, RslabError> {
    symbolic_factorize_with_method(matrix, snode_params, OrderingMethod::Auto)
}

/// A given ordering, checked to be a permutation of `0..n`.
fn checked_permutation(p: &[usize], n: usize) -> Result<Vec<usize>, RslabError> {
    let mut seen = vec![false; n];
    if p.len() != n
        || !p
            .iter()
            .all(|&k| k < n && !std::mem::replace(&mut seen[k], true))
    {
        return Err(RslabError::InvalidInput(format!(
            "given ordering is not a permutation of 0..{n}"
        )));
    }
    Ok(p.to_vec())
}

/// Seeds of the deterministic nested-dissection ensemble: multilevel ND is
/// seed-sensitive in one direction only (the 40^3 reference measures 11.24 M
/// exact nnz(L) on seeds {1, 3, 4} but 13.16 M - +17% - on seed 2, and on
/// another matrix the default seed can be the outlier), so `MetisND` runs
/// every seed and keeps the best. The candidates run in parallel (the wall
/// cost is roughly one ND plus the exact scoring), the score is the exact
/// scalar nnz(L) via permuted-etree column counts, and the winner is the
/// minimum fill with the LOWEST seed breaking ties - fully deterministic.
/// The candidates run on the ambient rayon pool, which `analyze_with` scopes
/// to the settings' thread budget - a `with_threads(1)` analysis runs the
/// ensemble strictly sequentially.
const ND_SEED_CANDIDATES: &[u64] = &[1, 2, 3];

/// One nested-dissection run: what an explicit `MetisND` request gets
/// (METIS semantics), and what the ordering race uses below
/// [`ND_SEED_RACE_MIN_FLOPS`].
const ND_SINGLE_SEED: &[u64] = &[1];

/// Predicted factorization flops (sum of squared column counts of the cheap
/// champion) from which the ordering race runs the full seed ensemble. Each
/// extra seed costs one nested dissection; the ensemble buys a few percent
/// of fill (up to 15 percent on 3D meshes), which only pays when the numeric
/// factorization is the dominant cost.
const ND_SEED_RACE_MIN_FLOPS: u64 = 50_000_000_000;

/// The cheap candidates of the [`OrderingMethod::AutoRace`] ordering race,
/// always run (each prefix costs a few milliseconds up to mid sizes):
/// minimum-degree, minimum-fill, and the band/profile reducer (which wins on
/// banded / structured patterns where the dissection candidates over-separate
/// and never hurts - it is picked only on the smallest exact factor nnz).
const RACE_CHEAP: &[OrderingMethod] = &[
    OrderingMethod::Amd,
    OrderingMethod::Amf,
    OrderingMethod::Rcm,
];

/// Amortization floor for the expensive `MetisND` race candidate: a
/// multilevel ND ordering costs hundreds of milliseconds at mid sizes, so it
/// only joins the race when the best cheap candidate's predicted factor time
/// ([`prefix_work`]) is large enough that an ND-class win can pay it back
/// (below the floor the whole numeric factor is sub-second and the ordering
/// time cannot amortize - the same work-floor principle as the KLU parallel
/// gates), and above the [`pick_default_method`] size boundary (tiny-n/high-flop
/// shapes are dense-ish, where dissection has nothing to separate).
const ND_RACE_MIN_WORK: u64 = 1_250_000_000;

/// Workers the race's time prediction assumes: the default thread cap.
const RACE_WORKERS: u64 = 4;

fn prefix_flops(px: &SymbolicPrefix) -> u64 {
    px.col_counts.iter().map(|&c| (c * c) as u64).sum()
}

/// Predicted factor time of a prefix in flops: the work shared among
/// [`RACE_WORKERS`], or the longest elimination chain where that is longer,
/// since no worker count shortens it. Minimum-degree orderings of some 3D
/// meshes leave most of the work on one chain (a 23k waveguide: 69 percent);
/// judged by total flops alone they never met the ND floor, although ND
/// halved the factor time there.
fn prefix_work(px: &SymbolicPrefix) -> u64 {
    // Every child precedes its parent in an elimination tree.
    let mut chain = vec![0u64; px.n];
    let mut longest = 0;
    for (j, &c) in px.col_counts.iter().enumerate() {
        let here = chain[j] + (c * c) as u64;
        longest = longest.max(here);
        if let Some(p) = px.etree.parent[j] {
            debug_assert!(p > j, "not an elimination tree");
            chain[p] = chain[p].max(here);
        }
    }
    (prefix_flops(px) / RACE_WORKERS).max(longest)
}

/// Race the [`race_candidates`] orderings at symbolic time and return the
/// `SymbolicFactorization` with the smallest exact scalar factor nnz.
///
/// Implements the [`OrderingMethod::AutoRace`] dispatcher. Feral #144
/// port: each candidate runs only the cheap pipeline *prefix* (ordering,
/// etree, column counts - everything the decision needs); the tail
/// (postorder, supernode detection, small-leaf grouping, memory plan) runs
/// once, for the winner. Candidates that error out (e.g. external
/// crate failure) are skipped; the race succeeds as long as at least one
/// candidate produces a valid prefix. Returns an error only if every
/// candidate fails.
///
/// The decision metric is the prefix's exact `factor_nnz` - the previous
/// full-pipeline race compared `factor_nnz_estimate` (`floor(1.2 * nnz)`),
/// a monotone transform, so the pick can differ only where two candidates'
/// estimates collided by rounding (and then the raw comparison picks the
/// actually-smaller one).
fn symbolic_factorize_race(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
) -> Result<SymbolicFactorization, RslabError> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::OnceLock;
    // Stage 1: the cheap candidates run concurrently (each prefix is itself
    // mostly sequential, so the race wall is roughly the slowest candidate);
    // the pick is deterministic - smallest exact factor nnz, candidate order
    // breaking ties - regardless of completion order.
    //
    // With more than one worker, the first nested-dissection seed starts
    // speculatively instead of after stage 1: the ND prefixes are the longest
    // part of the race, and stage 1 leaves most workers idle. On patterns
    // with at least `EAGER_ND_MIN_NNZ` entries it starts at once; on smaller
    // ones only once the AMD prefix predicts enough work to pass the ND gate,
    // because a discarded ND run on a small pattern outlasts the whole stage 1
    // (it doubled the race on a 128k power grid). The other ensemble seeds
    // start as soon as stage 1 has decided that they count. Stage 2 decides
    // from stage 1 alone which seeds count, exactly as it would have run them,
    // so the result does not depend on the thread count or on whether
    // anything was speculated.
    const EAGER_ND_MIN_NNZ: usize = 2_000_000;
    debug_assert_eq!(RACE_CHEAP[0], OrderingMethod::Amd);
    let parallel = matrix.n > 10_000 && rayon::current_num_threads() > 1;
    let eager = parallel && matrix.row_idx.len() >= EAGER_ND_MIN_NNZ;
    // One symmetric pattern and one ordering graph for every candidate: built
    // per candidate, the pattern was the largest allocation of the race (seven
    // copies at once) and the graph cost as much as a minimum-degree ordering.
    let full = &crate::logging::timed(
        || "analysis: symmetric pattern".into(),
        || matrix.symmetric_pattern(),
    );
    let graph = &OrderingGraph::new(full);
    graph.prepare()?;
    // One slot per ensemble seed, filled by its (at most one) ND prefix.
    let nd: [OnceLock<Option<SymbolicPrefix>>; ND_SEED_CANDIDATES.len()] = Default::default();
    let started: [AtomicBool; ND_SEED_CANDIDATES.len()] = Default::default();
    let (mut best, last_err, seeds) = rayon::scope(|sc| {
        let start = |i: usize| {
            if !started[i].swap(true, Ordering::Relaxed) {
                let nd = &nd;
                sc.spawn(move |_| {
                    let seed = [ND_SEED_CANDIDATES[i]];
                    let prefix = symbolic_prefix(
                        matrix,
                        graph,
                        snode_params,
                        OrderingMethod::MetisND,
                        &seed,
                    );
                    let _ = nd[i].set(prefix.ok());
                });
            }
        };
        if eager {
            start(0);
        }
        let (amd, rest) = rayon::join(
            || {
                let amd =
                    symbolic_prefix(matrix, graph, snode_params, RACE_CHEAP[0], ND_SINGLE_SEED);
                if parallel && matches!(&amd, Ok(p) if prefix_work(p) >= ND_RACE_MIN_WORK) {
                    start(0);
                }
                amd
            },
            || {
                RACE_CHEAP[1..]
                    .par_iter()
                    .map(|&cand| symbolic_prefix(matrix, graph, snode_params, cand, ND_SINGLE_SEED))
                    .collect::<Vec<_>>()
            },
        );
        let mut best: Option<SymbolicPrefix> = None;
        let mut last_err: Option<RslabError> = None;
        for r in std::iter::once(amd).chain(rest) {
            match r {
                Ok(prefix) => {
                    if best
                        .as_ref()
                        .is_none_or(|b| prefix.factor_nnz < b.factor_nnz)
                    {
                        best = Some(prefix);
                    }
                }
                Err(e) => last_err = Some(e),
            }
        }
        // Stage 2: the expensive ND candidate, only where its cost can
        // amortize (see [`ND_RACE_MIN_WORK`]), with the seed ensemble above
        // [`ND_SEED_RACE_MIN_FLOPS`].
        let seeds = match &best {
            Some(champ) if matrix.n > 10_000 && prefix_work(champ) >= ND_RACE_MIN_WORK => {
                if snode_params.nd_ensemble && prefix_flops(champ) >= ND_SEED_RACE_MIN_FLOPS {
                    ND_SEED_CANDIDATES.len()
                } else {
                    ND_SINGLE_SEED.len()
                }
            }
            _ => 0,
        };
        for i in 0..seeds {
            start(i);
        }
        (best, last_err, seeds)
    });
    // The ensemble's pick: the smallest exact factor nnz, the lowest seed
    // breaking ties; it replaces the cheap champion only when smaller.
    let nd = nd
        .into_iter()
        .take(seeds)
        .filter_map(|slot| slot.into_inner().flatten())
        .reduce(|a, b| if b.factor_nnz < a.factor_nnz { b } else { a });
    if let (Some(nd), Some(champ)) = (nd, &best) {
        if nd.factor_nnz < champ.factor_nnz {
            best = Some(nd);
        }
    }
    let Some(winner) = best else {
        return Err(last_err.unwrap_or_else(|| {
            RslabError::InvalidInput("AutoRace: no candidates available".to_string())
        }));
    };
    crate::logging::timed(
        || "analysis: finish".into(),
        || symbolic_finish(winner, full),
    )
}

/// Like [`symbolic_factorize`] but lets the caller pick the
/// fill-reducing ordering via [`OrderingMethod`].
///
/// `symbolic_factorize(m, p) == symbolic_factorize_with_method(m, p,
/// OrderingMethod::Amd)`.
pub fn symbolic_factorize_with_method(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
    method: OrderingMethod,
) -> Result<SymbolicFactorization, RslabError> {
    // AutoRace is resolved by racing each concrete candidate's cheap
    // *prefix* (ordering -> column counts -> factor nnz) and finishing
    // only the winner. The race passes concrete `OrderingMethod`s, so
    // there is no infinite loop.
    if method == OrderingMethod::AutoRace {
        return symbolic_factorize_race(matrix, snode_params);
    }
    let full = &matrix.symmetric_pattern();
    symbolic_finish(
        symbolic_prefix(
            matrix,
            &OrderingGraph::new(full),
            snode_params,
            method,
            ND_SINGLE_SEED,
        )?,
        full,
    )
}

/// Everything the cheap pipeline *prefix* produces: the ordering (incl.
/// preprocess), its elimination tree and column counts, and the exact scalar
/// factor nnz - the quantity every race dispatcher decides on. Produced by
/// [`symbolic_prefix`], consumed by [`symbolic_finish`] (feral #144 port:
/// race candidates run only the prefix; the postorder, supernode detection,
/// small-leaf grouping, and the memory plan run once, for the winner).
struct SymbolicPrefix {
    n: usize,
    /// The ordering, new-to-old, before the postorder.
    perm: Vec<usize>,
    /// Elimination tree and column counts under `perm`.
    etree: EliminationTree,
    col_counts: Vec<usize>,
    factor_nnz: usize,
    resolved_method: OrderingMethod,
    resolved_preprocess: OrderingPreprocess,
    /// The supernode parameters of this candidate (the preprocess variant).
    params: SupernodeParams,
}

/// Ceiling on the fill inflation `OrderingPreprocess::Auto` accepts from
/// `LdltCompress` before falling back to `None` (feral #91/#92 port). NOT
/// smaller-fill-wins: the MC64-matched compression carries a numerical
/// benefit symbolic fill does not capture (oracle-correct inertia through
/// matched 2x2 pivots on near-singular KKTs), and its normal overhead is
/// ~1.1-1.2x - so the guard fires only on a catastrophic misfire (feral
/// measured 6.3x fill / 20x factor-time inflation on the qap15 conic KKT,
/// whose IPM regularization rows fool the 30%-low-degree predicate).
const PREPROCESS_FILL_INFLATION_LIMIT: f64 = 2.0;

/// The cheap pipeline prefix - see [`SymbolicPrefix`]. `method` must be
/// concrete (`AutoRace` is dispatched by the caller).
///
/// Resolves `OrderingPreprocess::Auto` by **verifying** fill rather than
/// trusting the structural predicate (feral #91/#92 port): when
/// [`pick_ordering_preprocess`] recommends `LdltCompress`, both prefixes
/// run (they are cheap post-#144) and the compressed one is kept only if
/// its exact factor nnz stays within [`PREPROCESS_FILL_INFLATION_LIMIT`]
/// of the `None` baseline. An explicit (non-`Auto`) preprocess is honoured
/// unconditionally, exactly as before.
fn symbolic_prefix(
    matrix: &CscMatrix,
    graph: &OrderingGraph,
    snode_params: &SupernodeParams,
    method: OrderingMethod,
    nd_seeds: &[u64],
) -> Result<SymbolicPrefix, RslabError> {
    let resolved_preprocess = match snode_params.preprocess {
        OrderingPreprocess::Auto => pick_ordering_preprocess(matrix),
        other => other,
    };
    let verify = matches!(snode_params.preprocess, OrderingPreprocess::Auto)
        && matches!(resolved_preprocess, OrderingPreprocess::LdltCompress);
    if !verify {
        return symbolic_prefix_with(
            matrix,
            graph,
            snode_params,
            method,
            resolved_preprocess,
            nd_seeds,
        );
    }
    // Verify the predicate's LdltCompress pick against the `None` baseline.
    let variant_params = |pre: OrderingPreprocess| SupernodeParams {
        preprocess: pre,
        ..snode_params.clone()
    };
    let p_none = variant_params(OrderingPreprocess::None);
    let none = symbolic_prefix_with(
        matrix,
        graph,
        &p_none,
        method,
        OrderingPreprocess::None,
        nd_seeds,
    );
    let p_comp = variant_params(OrderingPreprocess::LdltCompress);
    let comp = symbolic_prefix_with(
        matrix,
        graph,
        &p_comp,
        method,
        OrderingPreprocess::LdltCompress,
        nd_seeds,
    );
    let winner = match (none, comp) {
        (Ok(none), Ok(comp)) => {
            let limit = (none.factor_nnz as f64) * PREPROCESS_FILL_INFLATION_LIMIT;
            if (comp.factor_nnz as f64) <= limit {
                comp
            } else {
                none
            }
        }
        // One side failed (e.g. MC64 error): the other one decides.
        (Ok(none), Err(_)) => none,
        (Err(_), Ok(comp)) => comp,
        (Err(e), Err(_)) => return Err(e),
    };
    Ok(winner)
}

/// The prefix pipeline at a **concrete** preprocess - the body of
/// [`symbolic_prefix`] once the `Auto` resolution/verification is done.
fn symbolic_prefix_with(
    matrix: &CscMatrix,
    graph: &OrderingGraph,
    snode_params: &SupernodeParams,
    method: OrderingMethod,
    resolved_preprocess: OrderingPreprocess,
    nd_seeds: &[u64],
) -> Result<SymbolicPrefix, RslabError> {
    let n = matrix.n;

    // beta refactor: scaling is no longer computed here. It moved to
    // `factorize_multifrontal` so that `SymbolicFactorization`
    // depends only on the matrix pattern (not its values) and can
    // be reused across multiple numeric factorizations of
    // structurally identical KKTs. See
    // `dev/plans/scaling-in-numeric.md`.

    // Step 1: Fill-reducing ordering. Dispatch on `method`. The
    // downstream pipeline (postorder composition, etree, column counts,
    // supernode amalgamation, memory plan) is identical regardless of
    // which ordering produced `initial_perm`.
    //
    // If `snode_params.preprocess == LdltCompress`, run MC64 symmetric
    // matching, build the super-variable map, order the compressed
    // graph, and expand the resulting super-permutation back to
    // length `n` before handing it to the rest of the pipeline. See
    // `src/symbolic/ldlt_compress.rs` and
    // `dev/plans/phase-2.6.5-ldlt-compressed-graph.md`.
    // Resolve `OrderingMethod::Auto` against the original matrix's
    // pattern *before* preprocessing. If we resolved against the
    // compressed pattern below, Auto would see a different `n` /
    // `avg_deg` and reach a different conclusion than
    // `symbolic_factorize` (which uses `pick_default_method` on the
    // matrix directly). Issue #3.
    let full_pattern = graph.pattern;
    let method = choose_adaptive(full_pattern, method);

    // `resolved_preprocess` arrives concrete from the dispatcher
    // ([`symbolic_prefix`] resolves and - for `Auto` -> `LdltCompress` -
    // fill-verifies the pick).
    // The fill-reducing ordering and (when enabled) the LdltCompress
    // preprocessor are timed under *separate* stages. The preprocessor's
    // MC64 matching can dwarf the ordering itself - on the pf22 powerflow
    // KKT (n=2.8M) MC64 is ~53s while `rslab_amd::amd_order` is ~0.3s - so
    // folding both into one "ordering" stage mis-attributes the cost and
    // led to the wrong diagnosis in issue #80. `record_ordering` wraps the
    // actual `OrderingGraph::order` call so every path records exactly one
    // `ordering` stage.
    let record_ordering = |g: &OrderingGraph| -> Result<(Vec<usize>, OrderingMethod), RslabError> {
        crate::logging::timed(
            || format!("analysis: ordering {method:?} seeds {nd_seeds:?}"),
            || Ok((g.order(method, nd_seeds)?, method)),
        )
    };
    let (perm, resolved_method): (Vec<usize>, OrderingMethod) = match resolved_preprocess {
        OrderingPreprocess::None => match &snode_params.given_perm {
            Some(p) => (checked_permutation(p, n)?, method),
            None => record_ordering(graph)?,
        },
        OrderingPreprocess::Auto => unreachable!("resolved above"),
        OrderingPreprocess::LdltCompress => {
            // Run the MC64 matching once for the compression supermap. MC64
            // is the expensive part - record it under its own
            // `ldlt_compress` stage (issue #80). NOTE: this matching CANNOT
            // be reused for `Mc64Symmetric` scaling, the generic solver
            // path feeds this function a unit-valued pattern, so the
            // matching carries no value information (see the
            // `scaling::Mc64Cache` note); the retired `cached_mc64` field
            // that promised that reuse was unwireable dead weight.
            let cache = crate::scaling::compute_mc64_cache(matrix)?;
            let map = build_supermap(&cache.perm);
            if map.n_super() == n {
                // Matching gives no compression leverage; fall through
                // to the uncompressed path rather than build and walk
                // an identical-size graph.
                record_ordering(graph)?
            } else {
                let cpat = compress_pattern(full_pattern, &map);
                let (super_perm, resolved) = record_ordering(&OrderingGraph::new(&cpat))?;
                let expanded = expand_permutation(&super_perm, &map);
                (expanded, resolved)
            }
        }
    };

    // Step 2: The elimination tree and the column counts of the ordered
    // pattern, read through the permutation (it is never materialized), and
    // the exact factor nnz: everything the race decides on. Both are
    // invariant under the postorder `symbolic_finish` applies to the winner
    // (a relabelling of the tree), so the candidates skip it.
    let t_tail = crate::clock::Instant::now();
    let mut perm_inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }
    let etree = EliminationTree::from_permuted_pattern(full_pattern, &perm, &perm_inv);
    // Gilbert-Ng-Peyton at O(nnz(A) + n*alpha(n)) (Phase 2.5.1; bit-exact
    // against the elimination simulation on 169585 KKT matrices, see
    // `dev/validation/phase-2.5.1-*`).
    let col_counts = column_counts_permuted(full_pattern, &perm, &perm_inv, &etree);
    let factor_nnz = total_factor_nnz(&col_counts);
    if crate::logging::enabled(crate::logging::LogLevel::Debug) {
        crate::logging::debug(&format!(
            "analysis: prefix tail {resolved_method:?} (etree, column counts): {:.1} ms",
            t_tail.elapsed().as_secs_f64() * 1e3
        ));
    }
    Ok(SymbolicPrefix {
        n,
        perm,
        etree,
        col_counts,
        factor_nnz,
        resolved_method,
        resolved_preprocess,
        params: snode_params.clone(),
    })
}

/// The pipeline tail: supernode detection, small-leaf grouping, memory
/// plan, and struct assembly. Runs once per *adopted* prefix - race losers
/// never get here (feral #144 port).
fn symbolic_finish(
    prefix: SymbolicPrefix,
    full_pattern: &CscPattern,
) -> Result<SymbolicFactorization, RslabError> {
    let SymbolicPrefix {
        n,
        perm: amd_perm,
        etree: amd_etree,
        col_counts: amd_col_counts,
        factor_nnz,
        resolved_method,
        resolved_preprocess,
        params: snode_params,
    } = prefix;
    // Step 3: Postorder the etree (CHOLMOD-style composition).
    // Without this step, supernode amalgamation merges columns whose indices
    // are not consecutive in the column numbering, and downstream code that
    // assumes `first_col..first_col+ncol` is the eliminated set silently
    // factors the wrong columns. See dev/research/postorder-pipeline.md.
    let (post, post_inv) = postorder(&amd_etree);

    // Step 4: Compose AMD perm with the postorder.
    //   final_perm[k] = amd_perm[post[k]]
    // The composition maps postorder position k to the original column.
    let perm: Vec<usize> = post.iter().map(|&p| amd_perm[p]).collect();
    let mut perm_inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }

    // Step 5: Build the final elimination tree by renumbering `amd_etree`
    // through the postorder. Postorder is a topological relabeling of the
    // elimination tree nodes, so `etree(P*A*P^T) = post-renumbering of
    // etree(A)` when P is a postorder of etree(A) - the tree structure is
    // preserved and only the node labels change. This lets us produce the
    // final etree in O(n) instead of re-running `from_pattern` at
    // O(nnz * alpha(n)). A 3-run bench shows ~3% small-frontal p90 improvement
    // over the old two-from_pattern approach.
    let final_parent: Vec<Option<usize>> = (0..n)
        .map(|new| {
            let old_amd = post[new];
            amd_etree.parent[old_amd].map(|old_par| post_inv[old_par])
        })
        .collect();
    let etree = EliminationTree {
        parent: final_parent,
        n,
    };

    // Step 6: The column counts carry over relabelled, as the tree does.
    let mut col_counts: Vec<usize> = post.iter().map(|&old| amd_col_counts[old]).collect();
    debug_assert_eq!(
        col_counts,
        column_counts_permuted(full_pattern, &perm, &perm_inv, &etree)
    );

    // Phase 2.12: optional SSIDS-style merge-biased postorder.
    // Predict desired merges using only the etree + column counts,
    // then re-postorder the etree so desired-merge children are
    // emitted adjacent to their parents. The downstream
    // `find_supernodes` adjacency check then succeeds for those
    // merges naturally.
    //
    // Rebuild path: compose perm with the bias-driven post2,
    // re-permute the matrix, rebuild etree and col_counts. The
    // structural properties are invariant under within-subtree
    // relabeling (CHOLMOD/SSIDS observation, see
    // `dev/research/phase-2.12-column-renumbering.md` section 5.1).
    //
    // Fast-path: when no bias is requested (no desired merges, OR
    // the strategy is `Adjacency`), the second pass is skipped and
    // the pipeline behaves identically to pre-Phase-2.12.
    let mut perm = perm;
    let mut etree = etree;

    // Phase 2.13a: resolve `Auto` to a concrete strategy via a cheap
    // O(n) etree shape predicate. The downstream Renumber gate and
    // `find_supernodes` reverse-iteration check need a concrete
    // variant - `Auto` is a top-level dispatch sentinel only.
    let mut effective_params = snode_params;
    if matches!(
        effective_params.amalgamation_strategy,
        supernode::AmalgamationStrategy::Auto
    ) {
        effective_params.amalgamation_strategy = supernode::pick_amalgamation_strategy(&etree);
    }
    let snode_params: &SupernodeParams = &effective_params;

    if matches!(
        snode_params.amalgamation_strategy,
        supernode::AmalgamationStrategy::Renumber
    ) {
        let bias = supernode::predict_merges(&etree, &col_counts, snode_params);
        if bias.iter().any(|&b| b) {
            let (post2, post2_inv) = biased_postorder(&etree, &bias);
            // Compose: perm_2[k] = perm[post2[k]]; the existing
            // `perm` already encodes AMD o post1.
            let new_perm: Vec<usize> = post2.iter().map(|&p| perm[p]).collect();
            let mut new_perm_inv = vec![0usize; n];
            for (new, &old) in new_perm.iter().enumerate() {
                new_perm_inv[old] = new;
            }
            // `post2` is a postorder of `etree`, so the tree and the column
            // counts carry over relabelled (every topological order of the
            // elimination tree gives the same filled graph). Recomputing both
            // from the renumbered pattern cost as much as the first pass;
            // debug builds still do it, as the check on the invariance.
            let new_etree = EliminationTree {
                parent: post2
                    .iter()
                    .map(|&old| etree.parent[old].map(|p| post2_inv[p]))
                    .collect(),
                n,
            };
            let new_col_counts: Vec<usize> = post2.iter().map(|&old| col_counts[old]).collect();
            #[cfg(debug_assertions)]
            {
                let new_permuted_pattern = permute_pattern(full_pattern, &new_perm);
                debug_assert_eq!(
                    new_etree.parent,
                    EliminationTree::from_pattern(&new_permuted_pattern).parent
                );
                debug_assert_eq!(
                    new_col_counts,
                    column_counts_gnp(&new_permuted_pattern, &new_etree)
                );
            }

            perm = new_perm;
            perm_inv = new_perm_inv;
            etree = new_etree;
            col_counts = new_col_counts;
        }
    }
    let permuted_pattern = permute_pattern(full_pattern, &perm);

    // Step 7: Supernode detection on the postordered etree
    let mut supernodes = find_supernodes(&etree, &col_counts, snode_params);
    // Issue #55 Phase B2: assign per-supernode incoming-delay budget.
    // Bounded-cost postorder pass; runs once per symbolic factor and
    // is cached in `SymbolicFactorization` for reuse across numeric
    // refactors. No effect until the numeric-time enforcement (B3)
    // and CB-rewire (B5) check `Supernode::delayed_capacity`.
    supernode::assign_delayed_capacities(&mut supernodes);

    // Step 7b: Phase 2.9 small-leaf grouping. Runs unconditionally;
    // the groups are consumed at numeric time only when the
    // `small_leaf` gate is `On`. O(n_snodes), no allocations beyond
    // the groups themselves.
    let (small_leaf_groups, snode_group) =
        find_small_leaf_groups(&supernodes, &permuted_pattern, &snode_params.small_leaf);

    // Step 5: Compute contribution sizes and peak memory
    let contrib_sizes: Vec<usize> = supernodes.iter().map(|s| s.contrib_size()).collect();

    let peak_contrib_bytes = compute_peak_contrib(&supernodes, &contrib_sizes);

    let factor_slack = 1.2;

    Ok(SymbolicFactorization {
        n,
        perm,
        perm_inv,
        supernodes,
        factor_nnz_estimate: (factor_nnz as f64 * factor_slack) as usize,
        factor_slack,
        contrib_sizes,
        peak_contrib_bytes,
        etree,
        permuted_pattern,
        col_counts,
        small_leaf_groups,
        snode_group,
        resolved_method,
        resolved_amalgamation: snode_params.amalgamation_strategy,
        resolved_preprocess,
    })
}

/// Compute the peak contribution pool size needed during postorder traversal.
///
/// At any point during traversal, the live contribution blocks are those
/// of nodes that have been factored but whose contribution has not yet
/// been assembled into their parent. In serial postorder, a node's
/// contribution is consumed when its parent is factored.
fn compute_peak_contrib(supernodes: &[Supernode], contrib_sizes: &[usize]) -> usize {
    let n_snodes = supernodes.len();
    if n_snodes == 0 {
        return 0;
    }

    // Simulate the postorder traversal:
    // - When we process supernode k: allocate contrib[k], free contrib[child] for each child
    // - Track peak allocation
    let mut live = vec![false; n_snodes];
    let mut current_size = 0usize;
    let mut peak = 0usize;

    for k in 0..n_snodes {
        // Allocate this node's contribution block
        current_size += contrib_sizes[k];
        live[k] = true;

        if current_size > peak {
            peak = current_size;
        }

        // Free children's contribution blocks (they've been assembled)
        for &child in &supernodes[k].children {
            if live[child] {
                current_size -= contrib_sizes[child];
                live[child] = false;
            }
        }
    }

    peak * std::mem::size_of::<f64>()
}

#[cfg(test)]
mod tests {

    /// The frontal height reported by the symbolic phase must equal the row set
    /// the numeric phase actually builds. They come from different code: the
    /// symbolic tracks the union cardinality through amalgamation, the schedule
    /// materializes the rows from the permuted pattern. A merged supernode is
    /// where the two used to disagree, so the amalgamation is swept.
    #[test]
    fn supernode_nrow_matches_the_built_row_sets() {
        use crate::numeric::supernodal::LlSchedule;
        // 2D 5-point Laplacian, lower triangle.
        let k = 20usize;
        let n = k * k;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        let push = |i: usize,
                    j: usize,
                    v: f64,
                    rows: &mut Vec<usize>,
                    cols: &mut Vec<usize>,
                    vals: &mut Vec<f64>| {
            rows.push(i.max(j));
            cols.push(i.min(j));
            vals.push(v);
        };
        for y in 0..k {
            for x in 0..k {
                let idx = y * k + x;
                push(idx, idx, 4.0, &mut rows, &mut cols, &mut vals);
                if x + 1 < k {
                    push(idx + 1, idx, -1.0, &mut rows, &mut cols, &mut vals);
                }
                if y + 1 < k {
                    push(idx + k, idx, -1.0, &mut rows, &mut cols, &mut vals);
                }
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();

        for nemin in [1usize, 16, 32] {
            let params = SupernodeParams {
                nemin,
                ..Default::default()
            };
            let sym = symbolic_factorize(&a, &params).unwrap();
            let sched = LlSchedule::build(&sym);
            let merged = sym.supernodes.iter().filter(|sn| sn.ncol > 1).count();
            assert!(merged > 0, "nemin={nemin}: nothing merged, test is vacuous");
            for (s, snode) in sym.supernodes.iter().enumerate() {
                assert_eq!(
                    snode.nrow,
                    sched.rows(s).len(),
                    "nemin={nemin}, supernode {s} (ncol={})",
                    snode.ncol
                );
            }
        }
    }
    use super::*;

    #[test]
    fn test_symbolic_factorize_basic() {
        // Simple tridiagonal
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();

        let params = SupernodeParams {
            nemin: 32,
            ..Default::default()
        };
        let sym = symbolic_factorize(&m, &params).unwrap();

        assert_eq!(sym.n, 4);
        assert_eq!(sym.perm.len(), 4);
        assert_eq!(sym.perm_inv.len(), 4);

        // Permutation should be valid
        let mut sorted_perm = sym.perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2, 3]);

        // Factor NNZ estimate should be >= actual NNZ
        assert!(sym.factor_nnz_estimate > 0);

        // Total supernode columns = n
        let total_cols: usize = sym.supernodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 4);
    }

    #[test]
    fn test_symbolic_factorize_dense() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let sym = symbolic_factorize(&m, &params).unwrap();

        // For a dense matrix, factor NNZ = n*(n+1)/2 = 6
        assert!(sym.factor_nnz_estimate >= 6);
    }

    #[test]
    fn test_symbolic_factorize_kkt() {
        // Small KKT matrix
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2],
            &[2.0, 3.0, 1.0, 1.0, -1e-8],
        )
        .unwrap();

        let params = SupernodeParams::default();
        let sym = symbolic_factorize(&m, &params).unwrap();

        assert_eq!(sym.n, 3);
        let total_cols: usize = sym.supernodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 3);
    }

    #[test]
    fn test_perm_inverse_consistency() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();

        let params = SupernodeParams::default();
        let sym = symbolic_factorize(&m, &params).unwrap();

        // perm and perm_inv are inverses
        for i in 0..5 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
            assert_eq!(sym.perm_inv[sym.perm[i]], i);
        }
    }

    #[test]
    fn test_contrib_sizes_nonnegative() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let sym = symbolic_factorize(&m, &params).unwrap();

        for &cs in &sym.contrib_sizes {
            // Contribution sizes should be non-negative (they're usize, always >= 0)
            // and for the root node it should be 0
            assert!(cs < 100000, "unreasonable contrib size: {}", cs);
        }

        // Root supernode should have 0 contribution block
        if let Some(last) = sym.supernodes.last() {
            assert_eq!(
                last.contrib_size(),
                0,
                "root should have no contribution block"
            );
        }
    }

    fn small_grid_5x5() -> CscMatrix {
        // 5x5 grid graph stored as CscMatrix (full symmetric, lower
        // triangle only). Used as a structurally non-trivial test
        // case where AMD, METIS, and SCOTCH all produce permutations
        // and the downstream pipeline must accept any of them.
        let m = 5;
        let n = 5;
        let idx = |r: usize, c: usize| r * n + c;
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                rows.push(k);
                cols.push(k);
                vals.push(4.0);
                if r + 1 < m {
                    rows.push(idx(r + 1, c));
                    cols.push(k);
                    vals.push(-1.0);
                }
                if c + 1 < n {
                    rows.push(idx(r, c + 1));
                    cols.push(k);
                    vals.push(-1.0);
                }
            }
        }
        CscMatrix::from_triplets(m * n, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn symbolic_factorize_amf_produces_valid_perm() {
        // Phase D wire-up smoke test: OrderingMethod::Amf must
        // produce a valid permutation through the full symbolic
        // pipeline (postorder composition, etree, column counts,
        // supernodes). This pins the dispatch wiring; bit-parity vs
        // MUMPS HAMF4 is the job of tests/amf_corpus_oracle.rs.
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amf).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
        assert_eq!(sym.resolved_method, OrderingMethod::Amf);
    }

    #[test]
    fn symbolic_factorize_metis_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::MetisND).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn symbolic_factorize_auto_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::Auto).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn choose_adaptive_rules() {
        // Pattern helper: diagonal pattern with n cols, nnz = density*n.
        fn pat_bufs(n: usize, avg_deg: usize) -> (Vec<usize>, Vec<usize>) {
            let total = n * avg_deg.max(1);
            let mut col_ptr = Vec::with_capacity(n + 1);
            let mut row_idx = Vec::with_capacity(total);
            let per = avg_deg.max(1);
            for j in 0..n {
                col_ptr.push(row_idx.len());
                for t in 0..per {
                    row_idx.push((j + t) % n.max(1));
                }
            }
            col_ptr.push(row_idx.len());
            (col_ptr, row_idx)
        }
        // Very-large-and-sparse (n > 100_000, avg_deg < 5.0) -> AMD.
        // Issue #50 swap (2026-05-23): pre-fix this was the ScotchND
        // branch; see choose_adaptive's doc comment.
        let (cp, ri) = pat_bufs(200_000, 3);
        let p = CscPattern {
            n: 200_000,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::Amd
        );
        // Small-and-sparse (n<10_000, avg_deg<15) -> delegates to
        // pick_default_method, which routes n<=10_000 to AMF. The F11
        // follow-up to issue #50 (2026-05-23) deleted the previous
        // small-and-sparse KahipND branch after the 838-matrix
        // inventory showed AMF aggregate fill 0.870x AMD vs KahipND
        // 0.984x and AMF aggregate time 0.832x AMD vs KahipND 0.990x
        // on that population; KahipND won only 16/838 matrices (1.9%)
        // vs AMF's 169/838 (20.2%). See choose_adaptive's doc comment
        // and dev/research/issue-50-metisnd-symbolic-cost.md section F12.
        let (cp, ri) = pat_bufs(500, 6);
        let p = CscPattern {
            n: 500,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::Amf
        );
        // Thin-large band (issue #67): n in (10_000, 100_000] that the
        // size rule would send to MetisND is overridden to AMF. The corpus
        // A/B (dev/research/issue-67-thin-large-ordering.md) found AMF wins
        // or ties MetisND on factor+solve across this whole band. Here
        // (n=50_000, avg_deg=20, uniform -> not arrow) -> AMF.
        let (cp, ri) = pat_bufs(50_000, 20);
        let p = CscPattern {
            n: 50_000,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::Amf
        );
        // Large-dense (n > 100_000, avg_deg >= 5, non-arrow) now also routes
        // to AMF (issue #73): the real factor+solve A/B found AMF wins on
        // every measured matrix in this regime, so the would-be-MetisND
        // decision is overridden to AMF at every n. The #50 avg_deg < 5 -> AMD
        // catch fires first and is unaffected. (n=150_000, avg_deg=10, uniform
        // -> not arrow.)
        let (cp, ri) = pat_bufs(150_000, 10);
        let p = CscPattern {
            n: 150_000,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::Amf
        );
        // Non-Auto passes through.
        let (cp, ri) = pat_bufs(500, 6);
        let p = CscPattern {
            n: 500,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::MetisND),
            OrderingMethod::MetisND
        );
    }

    /// Build a synthetic full-symmetric `CscPattern` with a prescribed
    /// per-column degree distribution. Connectivity is irrelevant to the
    /// degree-only arrow predicate, so row indices are filled with valid
    /// in-range values without forming a true symmetric pattern.
    fn pattern_with_degrees(degrees: &[usize]) -> CscPattern {
        let n = degrees.len();
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        for (j, &d) in degrees.iter().enumerate() {
            col_ptr.push(row_idx.len());
            for t in 0..d {
                row_idx.push((j + t) % n.max(1));
            }
        }
        col_ptr.push(row_idx.len());
        CscPattern {
            n,
            col_ptr,
            row_idx,
        }
    }

    #[test]
    fn is_arrow_bordered_fires_on_synthetic_arrow() {
        // Issue #64: a small set of very-high-degree border columns
        // carrying a large nnz share = arrow. 11_900 body columns of
        // degree 6 (71_400 nnz) + 100 border columns of degree 600
        // (60_000 nnz). avg_deg~10.95, heavy_thr=max(64,88)=88; border
        // exceeds it. heavy_count=100 (0.83% of n < 5%); heavy_nnz share
        // 60_000/131_400 = 45.7% >= 20% -> arrow.
        let mut degrees = vec![6usize; 11_900];
        degrees.extend(std::iter::repeat_n(600usize, 100));
        let pat = pattern_with_degrees(&degrees);
        assert!(is_arrow_bordered(&pat), "r05-shaped arrow must be detected");
    }

    #[test]
    fn is_arrow_bordered_rejects_uniform_sparse() {
        // Uniformly thin (PoissonControl / powerflow22 / bratu3d shape):
        // no column exceeds heavy_thr -> not an arrow.
        let pat = pattern_with_degrees(&vec![8usize; 12_000]);
        assert!(
            !is_arrow_bordered(&pat),
            "uniform-sparse pattern must not be flagged as arrow"
        );
    }

    #[test]
    fn is_arrow_bordered_rejects_many_hubs() {
        // Exercises the count guard: 1000 columns of degree 1000 (10% of
        // n) carry 99% of the nnz, but a heavy set this large is not a
        // thin border - nested dissection is not obviously wrong, so the
        // arrow override must NOT fire. heavy_count=1000 = 10% > 5%.
        let mut degrees = vec![1000usize; 1000];
        degrees.extend(std::iter::repeat_n(1usize, 9000));
        let pat = pattern_with_degrees(&degrees);
        assert!(
            !is_arrow_bordered(&pat),
            "a large heavy set (10% of n) must be rejected by the count guard"
        );
    }

    #[test]
    fn is_arrow_bordered_rejects_low_nnz_share_border() {
        // bcsstk38 shape: 2 very-high-degree columns but they carry a
        // tiny nnz share (0.3%). The share guard rejects it. n must be
        // small enough that 2 cols < 5%, which is always true here.
        let mut degrees = vec![44usize; 8030];
        degrees.extend([614usize, 614usize]);
        let pat = pattern_with_degrees(&degrees);
        // heavy_thr = max(64, 8*~44) = ~355; the two 614-degree columns
        // are heavy but carry 1228 of ~354_548 nnz = 0.35% << 20%.
        assert!(
            !is_arrow_bordered(&pat),
            "a heavy set carrying a tiny nnz share must be rejected by the share guard"
        );
    }

    #[test]
    fn choose_adaptive_routes_arrow_to_amf() {
        // Issue #64: an arrow pattern with n>10_000 (which would
        // otherwise route to MetisND via pick_default_method) must be
        // overridden to Amf. Mirror the synthetic-arrow degree shape.
        let mut degrees = vec![6usize; 11_900];
        degrees.extend(std::iter::repeat_n(600usize, 100));
        let pat = pattern_with_degrees(&degrees);
        assert_eq!(
            choose_adaptive(&pat, OrderingMethod::Auto),
            OrderingMethod::Amf,
            "arrow/bordered pattern (n>10_000) must route to Amf, not MetisND"
        );
        // A uniform large-dense pattern (n > 100_000, avg_deg >= 5,
        // non-arrow) now routes to AMF via the #73 thin-large catch (the
        // would-be-MetisND decision is overridden to AMF at every n). This
        // does not exercise the arrow catch - the point is only that a
        // non-arrow shape still lands on AMF, just through #73 rather than
        // #64.
        let uniform = pattern_with_degrees(&vec![16usize; 120_000]);
        assert_eq!(
            choose_adaptive(&uniform, OrderingMethod::Auto),
            OrderingMethod::Amf,
            "uniform large-dense non-arrow pattern routes to AMF via the #73 catch"
        );
        // The arrow override must NOT fire below the size floor: a small
        // arrow already routes to Amf via the n<=10_000 rule, but assert
        // the override doesn't accidentally change a non-MetisND base.
        let mut small_arrow = vec![6usize; 4900];
        small_arrow.extend(std::iter::repeat_n(600usize, 100));
        let small = pattern_with_degrees(&small_arrow);
        assert_eq!(
            choose_adaptive(&small, OrderingMethod::Auto),
            OrderingMethod::Amf
        );
    }

    #[test]
    fn symbolic_factorize_default_uses_amf_for_small_matrices() {
        // Per Phase D of dev/plans/amf-clean-room.md: small matrices
        // (n <= 10_000) default to AMF, mirroring MUMPS's
        // ana_set_ordering.F rule for SYM=2 N<=10000.
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let a = symbolic_factorize(&m, &params).unwrap();
        let b = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amf).unwrap();
        assert_eq!(
            a.perm, b.perm,
            "symbolic_factorize on small dense matrices must equal \
             symbolic_factorize_with_method(Amf)"
        );
        assert_eq!(a.factor_nnz_estimate, b.factor_nnz_estimate);
        assert_eq!(a.resolved_method, OrderingMethod::Amf);
    }

    #[test]
    fn pick_default_method_rules() {
        // Issue #50 (2026-05-23): the bordered-KKT and chain-pattern
        // catches that previously routed CRESC132/CHAINWOO/HYDROELL/
        // DIXMAANH/VESUVIO to MetisND were removed after the BK
        // pivoting cascade they defended against was killed by
        // issue #46's fixes (42434a5, 070840b). The numeric
        // inventory in dev/research/issue-50-numeric-inventory.csv
        // shows zero of 250 chain-catch-class matrices now have
        // AMD/MetisND num_nnz_l ratio >= 1.5x.
        //
        // The remaining rule is the MUMPS-style "small symmetric"
        // dispatch: n <= 10_000 -> AMF, n > 10_000 -> MetisND, with
        // the n == 0 sentinel returning AMD.

        // Empty matrix: AMD (avoids /0 and external-crate weirdness).
        assert_eq!(pick_default_method(0, 0), OrderingMethod::Amd);

        // Small matrices (n <= 10_000) -> AMF regardless of avg_deg.
        assert_eq!(pick_default_method(715, 2839), OrderingMethod::Amf); // HAHN1
        assert_eq!(pick_default_method(3000, 8999), OrderingMethod::Amf); // DIXMAANH
        assert_eq!(pick_default_method(3000, 13_000), OrderingMethod::Amf);
        assert_eq!(pick_default_method(3083, 9484), OrderingMethod::Amf); // VESUVIO
        assert_eq!(pick_default_method(4000, 7999), OrderingMethod::Amf); // CHAINWOO
        assert_eq!(pick_default_method(5000, 20_000), OrderingMethod::Amf);
        assert_eq!(pick_default_method(5314, 22566), OrderingMethod::Amf); // CRESC132
        assert_eq!(pick_default_method(10_000, 100_000), OrderingMethod::Amf);

        // Large matrices (n > 10_000) -> MetisND.
        assert_eq!(
            pick_default_method(20_000, 200_000),
            OrderingMethod::MetisND
        );
        // n=2_813_976, stored_nnz=6_622_463 (powerflow22 from #50):
        // -> MetisND now (was -> ScotchND via choose_adaptive's deleted
        // n>100k branch before #50). Issue #50's IPM-loop validation
        // is what justifies the deletion at this size.
        assert_eq!(
            pick_default_method(2_813_976, 6_622_463),
            OrderingMethod::MetisND
        );
    }

    /// PoissonControl KKT lower-triangle CSC, mirrors
    /// `src/bin/diag_poisson_kkt.rs`. n_kkt = 3K^2. K=20 -> n=1200,
    /// large enough to exceed amd_switch=120 (so SCOTCH actually
    /// runs the multilevel pipeline) but small enough to be cheap.
    fn poisson_kkt_csc(k: usize) -> CscMatrix {
        let m = k * k;
        let n_kkt = 3 * m;
        let h = 1.0 / (k as f64 + 1.0);
        let alpha = 0.01;
        let inv_h2 = 1.0 / (h * h);

        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..m {
            rows.push(i);
            cols.push(i);
            vals.push(h * h);
        }
        for i in 0..m {
            rows.push(m + i);
            cols.push(m + i);
            vals.push(alpha * h * h);
        }
        for i in 0..k {
            for j in 0..k {
                let c = i * k + j;
                let con_row = 2 * m + c;
                rows.push(con_row);
                cols.push(c);
                vals.push(4.0 * inv_h2);
                if i > 0 {
                    rows.push(con_row);
                    cols.push((i - 1) * k + j);
                    vals.push(-inv_h2);
                }
                if i + 1 < k {
                    rows.push(con_row);
                    cols.push((i + 1) * k + j);
                    vals.push(-inv_h2);
                }
                if j > 0 {
                    rows.push(con_row);
                    cols.push(i * k + (j - 1));
                    vals.push(-inv_h2);
                }
                if j + 1 < k {
                    rows.push(con_row);
                    cols.push(i * k + (j + 1));
                    vals.push(-inv_h2);
                }
                rows.push(con_row);
                cols.push(m + c);
                vals.push(-1.0);
            }
        }
        CscMatrix::from_triplets(n_kkt, &rows, &cols, &vals).expect("kkt csc")
    }

    #[test]
    fn issue_3_auto_on_kkt_routes_via_pick_default_method() {
        // Issue #3 invariant: the `Auto` path and the no-arg
        // `symbolic_factorize` default must resolve to the *same* concrete
        // ordering on every matrix. PoissonControl K=58 (n=10092, stored
        // avg_deg~2.67) is a uniformly-thin KKT just inside the #67
        // thin-large AMF band ((10_000, 100_000], non-arrow), so both paths
        // now resolve to AMF. (Before #67 this matrix resolved to MetisND
        // via pick_default_method's n>10_000 rule; the MetisND delegation is
        // still covered by the `choose_adaptive_rules` n=150_000 case.)
        let m = poisson_kkt_csc(58);
        let params = SupernodeParams::default();
        let auto = symbolic_factorize_with_method(&m, &params, OrderingMethod::Auto).unwrap();
        let default = symbolic_factorize(&m, &params).unwrap();
        assert_eq!(
            auto.resolved_method, default.resolved_method,
            "Auto must resolve to the same concrete method as \
             `symbolic_factorize` (which also routes through choose_adaptive)"
        );
        assert_eq!(auto.resolved_method, OrderingMethod::Amf);
    }
}
