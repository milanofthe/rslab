//! Single-solve thread-count policy derived from the symbolic analysis.

/// The data-driven single-solve thread-count policy, as a free function over the
/// three predictive features, so the factor path can apply it straight from the
/// symbolic analysis. Returns a worker count in `1..=max_cores`.
pub fn recommend_threads_from(
    factor_flops: u64,
    front_nrow_max: usize,
    tree_width_max: usize,
    max_cores: usize,
) -> usize {
    let cores = max_cores.max(1);
    // Thin fronts + narrow tree: no node-parallelism (tiny fronts) and no
    // tree-parallelism (path-like) to exploit - oversubscription only hurts.
    if front_nrow_max < 512 && tree_width_max < 128 {
        return cores.min(2);
    }
    // Tiny total work: parallel scheduling overhead dominates the factorization.
    if factor_flops < 300_000_000 {
        return cores.min(4);
    }
    cores
}
