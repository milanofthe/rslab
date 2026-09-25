//! Factorization settings as Python objects: `Settings` (the LDL^T / LU
//! paths, wrapping the core's `SolverSettings`), `KluSettings`, and the
//! `Interrupt` cancellation flag both can carry.
//!
//! Every knob is a keyword argument of the constructor; the same keywords are
//! accepted by the factor functions (`rslab.ldlt(A, threads=2)`), which build
//! a `Settings` from them. Parsing strings into the core enums happens here
//! and nowhere else.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rslab::{
    AmalgamationStrategy, KluParallel, OrderingMethod, RelaxAmalgamation, ScalingStrategy,
    SolverSettings, Threads, ZeroPivotAction,
};

/// A caller-owned cancellation flag for a running factorization.
///
/// Pass it as ``interrupt=`` to :class:`Settings` / :class:`KluSettings` (or
/// as a keyword to the factor functions). The numeric phase polls the flag at
/// supernode and panel boundaries and stops with ``RuntimeError("interrupted")``
/// once :meth:`cancel` was called, typically from another thread while the
/// factorization runs with the GIL released. :meth:`reset` re-arms the flag.
///
/// Example
/// -------
/// .. code-block:: python
///
///     stop = rslab.Interrupt()
///     threading.Timer(2.0, stop.cancel).start()      # give up after 2 s
///     f = rslab.lu(A, interrupt=stop)
#[pyclass(name = "Interrupt", module = "rslab")]
#[derive(Clone)]
pub struct PyInterrupt {
    pub flag: Arc<AtomicBool>,
}

#[pymethods]
impl PyInterrupt {
    #[new]
    fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation of the factorization(s) carrying this flag.
    fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Clear the flag so later factorizations run to completion again.
    fn reset(&self) {
        self.flag.store(false, Ordering::Relaxed);
    }

    /// ``True`` once :meth:`cancel` was called and not yet :meth:`reset`.
    #[getter]
    fn is_set(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    fn __repr__(&self) -> String {
        format!("Interrupt(is_set={})", self.is_set())
    }
}

fn bad(key: &str, expected: &str, got: &Bound<'_, PyAny>) -> PyErr {
    PyValueError::new_err(format!("{key} must be {expected}, got {got:?}"))
}

fn lower(key: &str, v: &Bound<'_, PyAny>) -> PyResult<String> {
    v.extract::<String>()
        .map(|s| s.to_ascii_lowercase())
        .map_err(|_| bad(key, "a string", v))
}

pub fn parse_ordering(s: &str) -> PyResult<OrderingMethod> {
    Ok(match s {
        "auto" => OrderingMethod::Auto,
        "amd" => OrderingMethod::Amd,
        "amf" => OrderingMethod::Amf,
        "metis" | "metisnd" | "metis_nd" | "nd" => OrderingMethod::MetisND,
        "rcm" => OrderingMethod::Rcm,
        other => {
            return Err(PyValueError::new_err(format!(
                "ordering must be 'auto', 'amd', 'amf', 'metis' or 'rcm', got '{other}'"
            )))
        }
    })
}

fn ordering_name(o: &OrderingMethod) -> &'static str {
    match o {
        OrderingMethod::Auto => "auto",
        OrderingMethod::Amd => "amd",
        OrderingMethod::Amf => "amf",
        OrderingMethod::MetisND => "metis",
        OrderingMethod::Rcm => "rcm",
    }
}

pub fn parse_scaling(s: &str) -> PyResult<ScalingStrategy> {
    Ok(match s {
        "inf_norm" | "infnorm" | "ruiz" => ScalingStrategy::InfNorm,
        "one_pass" | "one_pass_inf_norm" | "default" => ScalingStrategy::OnePassInfNorm,
        "mc64" | "mc64_symmetric" => ScalingStrategy::Mc64Symmetric,
        "identity" | "none" | "off" => ScalingStrategy::Identity,
        other => {
            return Err(PyValueError::new_err(format!(
                "scaling must be 'inf_norm', 'one_pass', 'mc64' or 'identity', got '{other}'"
            )))
        }
    })
}

fn scaling_name(s: &ScalingStrategy) -> &'static str {
    match s {
        ScalingStrategy::InfNorm => "inf_norm",
        ScalingStrategy::OnePassInfNorm => "one_pass",
        ScalingStrategy::Mc64Symmetric => "mc64",
        ScalingStrategy::Identity => "identity",
        ScalingStrategy::External(_) => "external",
    }
}

fn amalgamation_name(s: AmalgamationStrategy) -> &'static str {
    match s {
        AmalgamationStrategy::Auto => "auto",
        AmalgamationStrategy::Adjacency => "adjacency",
        AmalgamationStrategy::Renumber => "renumber",
    }
}

/// A human word for the expected type of a knob.
fn kind(ty: &str) -> &'static str {
    match ty {
        "bool" => "a bool",
        "f64" => "a float",
        _ => "a non-negative int",
    }
}

/// The settings that are one plain number or flag: the keyword, its type and
/// its field. `apply` and `to_dict` both come from this one list.
macro_rules! plain_knobs {
    ($($key:literal: $ty:ty => $($f:ident).+;)*) => {
        /// Set a plain knob; `false` when `key` is not one.
        fn set_plain(o: &mut SolverSettings, key: &str, v: &Bound<'_, PyAny>) -> PyResult<bool> {
            match key {
                $($key => {
                    o.$($f).+ = v.extract::<$ty>().map_err(|_| bad(key, kind(stringify!($ty)), v))?
                })*
                _ => return Ok(false),
            }
            Ok(true)
        }

        fn plain_dict(o: &SolverSettings, d: &Bound<'_, PyDict>) -> PyResult<()> {
            $(d.set_item($key, o.$($f).+)?;)*
            Ok(())
        }
    };
}

plain_knobs! {
    "pivot_threshold": f64 => pivoting.threshold;
    "matching": bool => matching.enabled;
    "matching_negligible_diagonal": f64 => matching.negligible_diagonal;
    "compress_max_ratio": f64 => ordering.compress_max_ratio;
    "nd_ensemble": bool => ordering.race.ensemble;
    "race_nd_min_n": usize => ordering.race.nd_min_n;
    "race_nd_min_work": u64 => ordering.race.nd_min_work;
    "race_assumed_workers": usize => ordering.race.assumed_workers;
    "race_eager_nd_min_nnz": usize => ordering.race.eager_nd_min_nnz;
    "race_ensemble_size": usize => ordering.race.ensemble_size;
    "race_ensemble_min_flops": u64 => ordering.race.ensemble_min_flops;
    "nd_seed": u64 => ordering.nd.seed;
    "nd_init_trials": u32 => ordering.nd.niparts;
    "nd_coarsen_floor": u32 => ordering.nd.coarsen_floor;
    "nd_leaf_size": u32 => ordering.nd.nd_to_amd_switch;
    "nd_two_hop_ratio": f64 => ordering.nd.two_hop_ratio_threshold;
    "nd_max_imbalance": f64 => ordering.nd.max_imbalance;
    "nd_fm_passes": u32 => ordering.nd.fm_passes;
    "nd_move_limit": usize => ordering.nd.move_limit;
    "nd_max_overshoot": f64 => ordering.nd.max_overshoot;
    "nd_parallel_min_vertices": usize => ordering.nd.parallel_min_vertices;
    "nd_parallel_min_edges": usize => ordering.nd.parallel_min_edges;
    "amd_aggressive": bool => ordering.amd.aggressive;
    "amd_dense_alpha": f64 => ordering.amd.dense_alpha;
    "amf_dense_alpha": f64 => ordering.amf.dense_alpha;
    "nemin": usize => amalgamation.nemin;
    "relax_min_n": usize => amalgamation.relax_min_n;
    "path_like_fraction": f64 => amalgamation.path_like_fraction;
    "root_cap_min_n": usize => amalgamation.root_cap_min_n;
    "root_cap_fraction": f64 => amalgamation.root_cap_fraction;
    "root_cap_max": usize => amalgamation.root_cap_max;
    "panel_nb": usize => kernels.panel_nb;
    "scalar_gate": usize => kernels.scalar_gate;
    "par_gemm": usize => kernels.par_gemm;
    "par_cdiv": usize => kernels.par_cdiv;
    "fork_min_flops": usize => kernels.fork_min_flops;
    "schur_tile": usize => kernels.schur_tile;
    "trailing_block": usize => kernels.trailing_block;
    "complex_split_min_ratio": usize => kernels.complex_split_min_ratio;
    "complex_split_tile": usize => kernels.complex_split_tile;
    "use_gemm_schur": bool => kernels.use_gemm_schur;
    "solve_leaf_subtrees": usize => solve.leaf_subtrees;
    "solve_block": usize => solve.block;
    "solve_ancestor_chunk": usize => solve.ancestor_chunk;
    "solve_apex_min_work": usize => solve.apex_min_work;
}

/// Settings of the symmetric LDL^T and the unsymmetric LU factorizations.
///
/// Wraps the core's ``SolverSettings``. Construct it from keyword arguments
/// (``rslab.Settings(threads=2, ordering="metis")``) and pass it as
/// ``settings=`` to :func:`rslab.ldlt` / :func:`rslab.lu` /
/// :func:`rslab.analyze`, or give the same keywords to those functions
/// directly. Unknown keywords raise ``TypeError``; invalid values ``ValueError``.
/// Every tuning constant of the solver is a keyword; the defaults are the
/// tuned values, listed by :meth:`to_dict`.
///
/// Parameters
/// ----------
/// ordering : {'auto', 'amd', 'amf', 'metis', 'rcm'}, optional
///     Fill-reducing ordering. ``'auto'`` (default) races the orderings on
///     the exact size of their factors: minimum degree, minimum fill and the
///     band reducer always, nested dissection on large systems (with a seed
///     ensemble under ``nd_ensemble``). An explicit value analyzes with
///     exactly that ordering, ``'metis'`` being one nested-dissection run.
///     The ordering used is reported in ``diagnostics()['decisions']``.
/// nd_ensemble : bool, default False
///     Keep the best of several nested-dissection seeds on heavy
///     factorizations: up to a few percent less fill for more dissections in
///     the analysis. Pays over long sweeps that refactor one analysis.
/// nemin : int, default 16
///     Supernode amalgamation threshold. Smaller means finer supernodes:
///     less fill, more per-front overhead.
/// relax : bool or (int, int), default False
///     Relaxed (fill-tolerant) amalgamation. ``True`` uses fronts up to 256
///     columns wide with at most 64 explicit-zero rows per merge, a pair
///     ``(max_width, max_extra_rows)`` sets them.
/// threads : int or 'auto' or ('auto', int) or 'ambient', optional
///     Worker budget of the scoped factorization pool. ``None`` (default) is
///     the per-matrix predictor capped at 4 workers (or the calibrated pick
///     after :func:`rslab.install_diagnose`); an ``int`` pins the count
///     (``0`` = all logical cores); ``'auto'`` is the predictor without the
///     cap, ``('auto', max)`` the predictor capped at ``max``; ``'ambient'``
///     runs on the caller's rayon pool. The factor is bit-identical for
///     every value.
/// preconditioner : float, optional
///     Static-pivot floor: a pivot with magnitude below it is lifted to it,
///     so the factorization never fails and produces the factor of a nearby
///     ``A + E``. Recover accuracy with ``solve(b, refine=k)``. ``1e-4`` is a
///     good start.
/// force_accept : bool, default False
///     In exact mode, accept tiny pivots instead of raising on rank
///     deficiency. Ignored when ``preconditioner`` is set.
/// drop_tol : float, optional
///     Incomplete-factorization threshold: fill below it (relative to the
///     column) is discarded, turning the factor into an ILU-style
///     preconditioner. ``None`` keeps the complete factor.
/// pivot_threshold : float, default 0.1
///     Threshold partial pivoting of the LU path in ``[0, 1]`` (``1.0`` is
///     full partial pivoting). Ignored, and reported in the diagnostics, on
///     the LDL^T path.
/// scaling : {'one_pass', 'inf_norm', 'mc64', 'identity'} or array, optional
///     Symmetric equilibration before the LDL^T factorization: a named
///     strategy, or a float array ``s`` of length ``n`` applying the
///     external scaling ``diag(s) A diag(s)``. The LU path uses its own
///     two-sided scaling and reports a set value.
/// matching : bool, default True
///     Maximum-product row matching (MC64) before the LU analysis, applied
///     where a diagonal entry is below ``matching_negligible_diagonal``
///     (default ``1e-10``) times its column's largest entry. LU path only.
/// interrupt : Interrupt, optional
///     A cancellation flag polled by the numeric phase.
///
/// Other Parameters
/// ----------------
/// race_candidates : list of str, default ['amd', 'amf', 'rcm']
///     The cheap candidates of the ordering race.
/// race_nd_min_n, race_nd_min_work, race_assumed_workers : int
///     Nested dissection joins the race above ``race_nd_min_n`` unknowns
///     (10 000) when the best cheap candidate predicts at least
///     ``race_nd_min_work`` flops (1.25e9) per ``race_assumed_workers`` (4).
/// race_eager_nd_min_nnz : int, default 2 000 000
///     Entries from which the dissection starts speculatively.
/// race_ensemble_size, race_ensemble_min_flops : int
///     Seeds of the ``nd_ensemble`` (3) and the flops from which it runs
///     (5e10).
/// compress_max_ratio : float, default 0.95
///     Order the graph of indistinguishable-vertex groups when they shrink it
///     to at most this share.
/// nd_seed, nd_init_trials, nd_coarsen_floor, nd_leaf_size : int
///     Nested dissection: seed (1), initial bisections (7), coarsest graph
///     size (120), subgraphs ordered by minimum degree (200).
/// nd_two_hop_ratio, nd_max_imbalance, nd_max_overshoot : float
///     Coarsening ratio below which two-hop matching runs (0.85), allowed
///     side imbalance (0.2), separator growth that ends a refinement pass (4).
/// nd_fm_passes, nd_move_limit : int
///     Refinement passes per level (10) and moves without improvement that
///     end one (1 048 576).
/// nd_parallel_min_vertices, nd_parallel_min_edges : int
///     Sizes from which subproblems (4096) and coarsening levels (200 000)
///     run in parallel.
/// amd_aggressive, amd_dense_alpha, amf_dense_alpha
///     Minimum degree: aggressive absorption (True) and the dense-row
///     threshold (10), also of minimum fill.
/// amalgamation : {'auto', 'adjacency', 'renumber'}, default 'auto'
///     How merges reach non-adjacent children; ``'auto'`` renumbers unless
///     fewer than ``path_like_fraction`` (0.05) of the internal tree nodes
///     have several children.
/// relax_min_n : int, default 1024
///     Relaxed amalgamation applies from this many unknowns.
/// root_cap_min_n, root_cap_fraction, root_cap_max
///     Merges into a root stop at ``root_cap_fraction * n`` columns (0.05), at
///     most ``root_cap_max`` (2048), from ``root_cap_min_n`` unknowns (1024).
/// panel_nb, trailing_block, schur_tile : int
///     Bunch-Kaufman panel width (64), its trailing sub-block (16) and the
///     Schur tile (256).
/// scalar_gate, par_gemm, par_cdiv, fork_min_flops : int
///     Update flops below which the scalar loop runs (4096), from which an
///     update GEMM (1e6) and a panel's trailing update (8e6) run in
///     parallel, and from which a supernode's updates fork (1e8).
/// complex_split_min_ratio, complex_split_tile : int
///     Complex products run as real products when their flops are this many
///     times their plane copies (64), in tiles of this edge (256).
/// use_gemm_schur : bool, default True
///     SIMD GEMM (vs the scalar loop) for the LDL^T Schur update.
/// solve_leaf_subtrees, solve_block, solve_ancestor_chunk, solve_apex_min_work : int
///     Triangular solves: leaf subtrees of the parallel sweeps (128), column
///     block (512) and columns per task (32) of the ancestor sweeps, panel
///     entries from which an ancestor uses the blocked sweep (262 144).
#[pyclass(name = "Settings", module = "rslab")]
#[derive(Clone, Default)]
pub struct PySettings {
    pub inner: SolverSettings,
    /// `threads=` was given: an explicit count beats the calibrated pick.
    pub explicit_threads: bool,
    /// The static-pivot floor, kept for `to_dict()` / `repr`.
    preconditioner: Option<f64>,
    force_accept: bool,
    interrupt: Option<PyInterrupt>,
}

impl PySettings {
    /// Build from a keyword dictionary; `None` values keep the default.
    pub fn from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut s = Self::default();
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                let key: String = k.extract()?;
                if !v.is_none() {
                    s.apply(&key, &v)?;
                }
            }
        }
        Ok(s)
    }

    fn apply(&mut self, key: &str, v: &Bound<'_, PyAny>) -> PyResult<()> {
        if set_plain(&mut self.inner, key, v)? {
            return Ok(());
        }
        let o = &mut self.inner;
        match key {
            "threads" => {
                self.explicit_threads = true;
                o.threads = if let Ok(n) = v.extract::<usize>() {
                    Threads::Fixed(n)
                } else if let Ok((word, max)) = v.extract::<(String, usize)>() {
                    if word.to_lowercase() != "auto" {
                        return Err(bad(key, "an int, 'auto', ('auto', max) or 'ambient'", v));
                    }
                    Threads::Auto { max }
                } else {
                    match lower(key, v)?.as_str() {
                        "auto" => Threads::Auto { max: usize::MAX },
                        "ambient" => Threads::Ambient,
                        _ => return Err(bad(key, "an int, 'auto', ('auto', max) or 'ambient'", v)),
                    }
                };
            }
            "preconditioner" => {
                let floor: f64 = v.extract().map_err(|_| bad(key, "a float", v))?;
                self.preconditioner = Some(floor);
                o.pivoting.on_zero_pivot = ZeroPivotAction::PerturbToEps { abs_floor: floor };
            }
            "force_accept" => {
                self.force_accept = v.extract().map_err(|_| bad(key, "a bool", v))?;
            }
            "drop_tol" => o.drop_tol = Some(v.extract().map_err(|_| bad(key, "a float", v))?),
            "ordering" => o.ordering.method = parse_ordering(&lower(key, v)?)?,
            "race_candidates" => {
                let names: Vec<String> = v
                    .extract()
                    .map_err(|_| bad(key, "a list of ordering names", v))?;
                o.ordering.race.candidates = names
                    .iter()
                    .map(|n| parse_ordering(&n.to_ascii_lowercase()))
                    .collect::<PyResult<_>>()?;
            }
            "scaling" => {
                o.scaling = if let Ok(name) = v.extract::<String>() {
                    parse_scaling(&name.to_lowercase())?
                } else if let Ok(arr) = v.extract::<numpy::PyReadonlyArray1<f64>>() {
                    ScalingStrategy::External(arr.as_slice()?.to_vec())
                } else {
                    return Err(bad(key, "a scaling name or a float array of length n", v));
                };
            }
            "relax" => {
                o.amalgamation.relax = if let Ok(on) = v.extract::<bool>() {
                    on.then(RelaxAmalgamation::default)
                } else if let Ok((max_width, max_extra_rows)) = v.extract::<(usize, usize)>() {
                    Some(RelaxAmalgamation {
                        max_width,
                        max_extra_rows,
                    })
                } else {
                    return Err(bad(key, "a bool or (max_width, max_extra_rows)", v));
                };
            }
            "amalgamation" => {
                o.amalgamation.strategy = match lower(key, v)?.as_str() {
                    "auto" => AmalgamationStrategy::Auto,
                    "adjacency" => AmalgamationStrategy::Adjacency,
                    "renumber" => AmalgamationStrategy::Renumber,
                    _ => return Err(bad(key, "'auto', 'adjacency' or 'renumber'", v)),
                };
            }
            "interrupt" => {
                let flag: PyInterrupt =
                    v.extract().map_err(|_| bad(key, "an rslab.Interrupt", v))?;
                o.interrupt = Some(flag.flag.clone());
                self.interrupt = Some(flag);
            }
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "unknown setting '{other}'"
                )))
            }
        }
        Ok(())
    }

    /// The core settings with the exact-mode pivot policy resolved.
    pub fn resolved(&self) -> SolverSettings {
        let mut o = self.inner.clone();
        if self.force_accept && self.preconditioner.is_none() {
            o.pivoting.on_zero_pivot = ZeroPivotAction::ForceAccept;
        }
        o
    }

    fn threads_value(&self, py: Python<'_>) -> PyObject {
        match self.inner.threads {
            Threads::Fixed(n) => n.into_py(py),
            Threads::Auto { max } if max == usize::MAX => "auto".into_py(py),
            Threads::Auto { max } if self.explicit_threads => ("auto", max).into_py(py),
            Threads::Auto { .. } => py.None(),
            Threads::Ambient => "ambient".into_py(py),
        }
    }
}

#[pymethods]
impl PySettings {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::from_kwargs(kwargs)
    }

    /// The settings as a plain dictionary, one key per keyword argument.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let o = &self.inner;
        let d = PyDict::new_bound(py);
        d.set_item("threads", self.threads_value(py))?;
        d.set_item("preconditioner", self.preconditioner)?;
        d.set_item("force_accept", self.force_accept)?;
        d.set_item("drop_tol", o.drop_tol)?;
        d.set_item("ordering", ordering_name(&o.ordering.method))?;
        d.set_item(
            "race_candidates",
            o.ordering
                .race
                .candidates
                .iter()
                .map(ordering_name)
                .collect::<Vec<_>>(),
        )?;
        d.set_item("scaling", scaling_name(&o.scaling))?;
        d.set_item(
            "relax",
            match &o.amalgamation.relax {
                Some(r) => PyTuple::new_bound(py, [r.max_width, r.max_extra_rows])
                    .into_any()
                    .unbind(),
                None => false.into_py(py),
            },
        )?;
        d.set_item("amalgamation", amalgamation_name(o.amalgamation.strategy))?;
        plain_dict(o, &d)?;
        d.set_item("interrupt", self.interrupt.clone().map(|i| i.into_py(py)))?;
        Ok(d.into_any().unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let d = self.to_dict(py)?;
        let d = d.bind(py).downcast::<PyDict>()?;
        let mut parts = Vec::new();
        for (k, v) in d.iter() {
            parts.push(format!("{}={}", k.extract::<String>()?, v.repr()?));
        }
        Ok(format!("Settings({})", parts.join(", ")))
    }
}

/// Settings of the KLU (circuit) path.
///
/// Parameters
/// ----------
/// pivot_threshold : float, default 1e-3
///     Diagonal-preference threshold: the diagonal entry is the pivot when
///     ``|a_jj| >= pivot_threshold * max_i |a_ij|``; ``1.0`` is plain partial
///     pivoting.
/// row_scaling : bool, default True
///     Divide each row by its max-magnitude entry before factoring.
/// btf : bool, default True
///     Permute to block upper triangular form first (keep it on).
/// matching : bool, default True
///     Maximum-product row matching (MC64) as the transversal of the block
///     triangular form, so the diagonal-preference pivoting rarely leaves
///     the diagonal; needs ``btf``.
/// parallel : bool, optional
///     Per-block parallel factor / refactor over the BTF blocks. ``None``
///     (default) is the structural auto gate (several blocks,
///     ``par_min_nnz`` nonzeros, no dominant block); ``True`` / ``False``
///     force it. The result is bit-identical in every mode.
/// par_min_nnz : int, default 8000
///     Nonzeros from which the auto gate factors blocks in parallel.
/// par_min_work : int, default 5e7
///     Replay work a unit of parallel refactorization must carry.
/// par_min_ratio : float, default 2.0
///     Simultaneous work the structure must offer for a parallel refactor.
/// interrupt : Interrupt, optional
///     A cancellation flag polled by the numeric phase.
#[pyclass(name = "KluSettings", module = "rslab")]
#[derive(Clone)]
pub struct PyKluSettings {
    pub inner: rslab::KluSettings,
    interrupt: Option<PyInterrupt>,
}

impl PyKluSettings {
    pub fn from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut s = Self {
            inner: rslab::KluSettings::default(),
            interrupt: None,
        };
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                let key: String = k.extract()?;
                if !v.is_none() {
                    s.apply(&key, &v)?;
                }
            }
        }
        Ok(s)
    }

    fn apply(&mut self, key: &str, v: &Bound<'_, PyAny>) -> PyResult<()> {
        let o = std::mem::take(&mut self.inner);
        self.inner = match key {
            "pivot_threshold" => {
                o.with_pivot_threshold(v.extract().map_err(|_| bad(key, "a float", v))?)
            }
            "par_min_nnz" => rslab::KluSettings {
                par_min_nnz: v.extract().map_err(|_| bad(key, "an int", v))?,
                ..o
            },
            "par_min_work" => rslab::KluSettings {
                par_min_work: v.extract().map_err(|_| bad(key, "an int", v))?,
                ..o
            },
            "par_min_ratio" => rslab::KluSettings {
                par_min_ratio: v.extract().map_err(|_| bad(key, "a float", v))?,
                ..o
            },
            "matching" => o.with_matching(v.extract().map_err(|_| bad(key, "a bool", v))?),
            "row_scaling" => o.with_row_scaling(v.extract().map_err(|_| bad(key, "a bool", v))?),
            "btf" => o.with_btf(v.extract().map_err(|_| bad(key, "a bool", v))?),
            "parallel" => o.with_parallel(match v.extract::<bool>() {
                Ok(true) => KluParallel::On,
                Ok(false) => KluParallel::Off,
                Err(_) => return Err(bad(key, "a bool or None", v)),
            }),
            "interrupt" => {
                let flag: PyInterrupt =
                    v.extract().map_err(|_| bad(key, "an rslab.Interrupt", v))?;
                self.interrupt = Some(flag.clone());
                o.with_interrupt(flag.flag)
            }
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "unknown KLU setting '{other}'"
                )))
            }
        };
        Ok(())
    }
}

#[pymethods]
impl PyKluSettings {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::from_kwargs(kwargs)
    }

    /// The settings as a plain dictionary, one key per keyword argument.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let o = &self.inner;
        let d = PyDict::new_bound(py);
        d.set_item("pivot_threshold", o.pivot_threshold)?;
        d.set_item("row_scaling", o.row_scaling)?;
        d.set_item("btf", o.btf)?;
        d.set_item("matching", o.matching)?;
        d.set_item(
            "parallel",
            match o.parallel {
                KluParallel::Auto => None,
                KluParallel::On => Some(true),
                KluParallel::Off => Some(false),
            },
        )?;
        d.set_item("par_min_nnz", o.par_min_nnz)?;
        d.set_item("par_min_work", o.par_min_work)?;
        d.set_item("par_min_ratio", o.par_min_ratio)?;
        d.set_item("interrupt", self.interrupt.clone().map(|i| i.into_py(py)))?;
        Ok(d.into_any().unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let d = self.to_dict(py)?;
        let d = d.bind(py).downcast::<PyDict>()?;
        let mut parts = Vec::new();
        for (k, v) in d.iter() {
            parts.push(format!("{}={}", k.extract::<String>()?, v.repr()?));
        }
        Ok(format!("KluSettings({})", parts.join(", ")))
    }
}

/// Resolve `settings=` plus keyword overrides into one `Settings`.
pub fn settings_from(
    settings: Option<PySettings>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PySettings> {
    let mut s = settings.unwrap_or_default();
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key: String = k.extract()?;
            if !v.is_none() {
                s.apply(&key, &v)?;
            }
        }
    }
    Ok(s)
}

pub fn klu_settings_from(
    settings: Option<PyKluSettings>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyKluSettings> {
    let mut s = match settings {
        Some(s) => s,
        None => PyKluSettings::from_kwargs(None)?,
    };
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key: String = k.extract()?;
            if !v.is_none() {
                s.apply(&key, &v)?;
            }
        }
    }
    Ok(s)
}
