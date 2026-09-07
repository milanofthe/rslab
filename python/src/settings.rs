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
    BlrMode, FactorMethod, GemmThresholds, KluParallel, MemoryMode, OrderingMethod,
    RelaxAmalgamation, ReorderMode, ScalingStrategy, SolverSettings, Threads, ZeroPivotAction,
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
        "auto_race" | "autorace" | "race" => OrderingMethod::AutoRace,
        "amd" => OrderingMethod::Amd,
        "amf" => OrderingMethod::Amf,
        "metis" | "metisnd" | "metis_nd" | "nd" => OrderingMethod::MetisND,
        "rcm" => OrderingMethod::Rcm,
        other => {
            return Err(PyValueError::new_err(format!(
            "ordering must be 'auto', 'auto_race', 'amd', 'amf', 'metis' or 'rcm', got '{other}'"
        )))
        }
    })
}

fn ordering_name(o: &OrderingMethod) -> &'static str {
    match o {
        OrderingMethod::Auto => "auto",
        OrderingMethod::AutoRace => "auto_race",
        OrderingMethod::Amd => "amd",
        OrderingMethod::Amf => "amf",
        OrderingMethod::MetisND => "metis",
        OrderingMethod::Rcm => "rcm",
    }
}

pub fn parse_scaling(s: &str) -> PyResult<ScalingStrategy> {
    Ok(match s {
        "auto" => ScalingStrategy::Auto,
        "inf_norm" | "infnorm" | "ruiz" => ScalingStrategy::InfNorm,
        "one_pass" | "one_pass_inf_norm" | "default" => ScalingStrategy::OnePassInfNorm,
        "mc64" | "mc64_symmetric" => ScalingStrategy::Mc64Symmetric,
        "identity" | "none" | "off" => ScalingStrategy::Identity,
        other => {
            return Err(PyValueError::new_err(format!(
            "scaling must be 'auto', 'inf_norm', 'one_pass', 'mc64' or 'identity', got '{other}'"
        )))
        }
    })
}

fn scaling_name(s: &ScalingStrategy) -> &'static str {
    match s {
        ScalingStrategy::Auto => "auto",
        ScalingStrategy::InfNorm => "inf_norm",
        ScalingStrategy::OnePassInfNorm => "one_pass",
        ScalingStrategy::Mc64Symmetric => "mc64",
        ScalingStrategy::Identity => "identity",
        ScalingStrategy::External(_) => "external",
    }
}

/// Settings of the symmetric LDL^T and the unsymmetric LU factorizations.
///
/// Wraps the core's ``SolverSettings``. Construct it from keyword arguments
/// (``rslab.Settings(threads=2, ordering="metis")``) and pass it as
/// ``settings=`` to :func:`rslab.ldlt` / :func:`rslab.lu` /
/// :func:`rslab.analyze`, or give the same keywords to those functions
/// directly. Unknown keywords raise ``TypeError``; invalid values ``ValueError``.
///
/// Analysis (pattern) knobs
/// ------------------------
/// ordering : {'auto', 'auto_race', 'amd', 'amf', 'metis', 'rcm'}, optional
///     Fill-reducing ordering. ``None`` (default) uses the heuristic pick,
///     the adaptive ordering plus an exact nested-dissection bakeoff on large
///     systems (with a small seed ensemble once the factorization is heavy
///     enough to pay for it); an explicit value analyzes with exactly that
///     ordering, ``'metis'`` being one nested-dissection run. The ordering
///     actually used is reported in ``diagnostics()['decisions']``.
/// nemin : int, optional
///     Supernode amalgamation threshold (default 16). Smaller means finer
///     supernodes: less fill, more per-front overhead.
/// relax : bool or (int, int), optional
///     Relaxed (fill-tolerant) amalgamation. ``True`` (default) keeps the
///     built-in thresholds, ``False`` disables it, a pair
///     ``(max_width, max_extra_rows)`` sets them explicitly.
/// reorder : {'hybrid_liu', 'off'}, optional
///     Child reordering of the elimination tree: ``'hybrid_liu'`` (default)
///     shrinks the contribution-stack peak, ``'off'`` keeps the natural leaf
///     order for maximum leaf parallelism.
///
/// Numeric knobs
/// -------------
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
/// method : {'left_looking', 'multifrontal'}, default 'left_looking'
///     Numeric schedule; same factor, different transient memory and
///     parallel profile.
/// memory : {'low', 'eager'}, default 'low'
///     Factor emit strategy: ``'low'`` frees each front as soon as it is
///     emitted, ``'eager'`` keeps them resident.
/// pivot_u : float, optional
///     Threshold partial-pivoting tolerance of the LU path in ``[0, 1]``
///     (default 0.1; ``1.0`` is full partial pivoting). Ignored, and reported
///     in the diagnostics, on the LDL^T path.
/// scaling : {'one_pass', 'inf_norm', 'mc64', 'auto', 'identity'} or array, optional
///     Symmetric equilibration before the LDL^T factorization: a named
///     strategy, or a float array ``s`` of length ``n`` applying the
///     external scaling ``diag(s) A diag(s)``. The LU path uses its own
///     two-sided scaling and reports a set value.
/// matching : bool, default True
///     Maximum-product row matching (MC64) before the LU analysis: rows are
///     permuted so the matched entries form the diagonal and both sides are
///     scaled to unit magnitude there, which keeps the element growth of the
///     front-restricted pivoting bounded. LU path only.
/// blr : float or False or dict, optional
///     Block-low-rank compression of the contribution blocks. A float is
///     the relative tolerance with the default block parameters; a dict
///     ``{'eps': tol, 'min_cnrow': 256, 'b': 256, 'adaptive': False}`` sets
///     the smallest contribution block that is compressed, the block size
///     and adaptive per-vector precision; ``False`` (default) keeps exact
///     dense fronts.
/// panel_nb : int, optional
///     Panel width (blocking factor) of the dense kernels, default 64.
/// interrupt : Interrupt, optional
///     A cancellation flag polled by the numeric phase.
///
/// Kernel tuning (benchmark knobs; the defaults are calibrated)
/// -----------------------------------------------------------
/// scalar_gate, par_gemm, par_cdiv : int, optional
///     Flop-count thresholds below which an update runs as a scalar loop, and
///     at or above which the GEMM / the panel-trailing update run in parallel.
/// use_gemm_schur : bool, optional
///     Use the SIMD GEMM (``True``, default) or the scalar loop for the front
///     Schur update.
#[pyclass(name = "Settings", module = "rslab")]
#[derive(Clone, Default)]
pub struct PySettings {
    pub inner: SolverSettings,
    /// `threads=` was given: an explicit count beats the calibrated pick.
    pub explicit_threads: bool,
    /// `ordering=` was given: analyze with it instead of the heuristic pick.
    pub explicit_ordering: bool,
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
        let o = std::mem::take(&mut self.inner);
        self.inner = match key {
            "threads" => {
                self.explicit_threads = true;
                if let Ok(n) = v.extract::<usize>() {
                    o.with_threads(n)
                } else if let Ok((word, max)) = v.extract::<(String, usize)>() {
                    if word.to_lowercase() != "auto" {
                        return Err(bad(key, "an int, 'auto', ('auto', max) or 'ambient'", v));
                    }
                    o.with_thread_policy(Threads::Auto { max })
                } else {
                    match lower(key, v)?.as_str() {
                        "auto" => o.with_thread_policy(Threads::Auto { max: usize::MAX }),
                        "ambient" => o.with_thread_policy(Threads::Ambient),
                        _ => return Err(bad(key, "an int, 'auto', ('auto', max) or 'ambient'", v)),
                    }
                }
            }
            "preconditioner" => {
                let floor: f64 = v.extract().map_err(|_| bad(key, "a float", v))?;
                self.preconditioner = Some(floor);
                o.with_pivot(ZeroPivotAction::PerturbToEps { abs_floor: floor })
            }
            "force_accept" => {
                self.force_accept = v.extract().map_err(|_| bad(key, "a bool", v))?;
                o
            }
            "drop_tol" => o.with_drop_tol(v.extract().map_err(|_| bad(key, "a float", v))?),
            "method" => o.with_method(match lower(key, v)?.as_str() {
                "multifrontal" | "mf" => FactorMethod::Multifrontal,
                "left_looking" | "leftlooking" | "ll" => FactorMethod::LeftLooking,
                _ => return Err(bad(key, "'left_looking' or 'multifrontal'", v)),
            }),
            "memory" => o.with_memory(match lower(key, v)?.as_str() {
                "eager" => MemoryMode::Eager,
                "low" | "low_memory" => MemoryMode::LowMemory,
                _ => return Err(bad(key, "'low' or 'eager'", v)),
            }),
            "ordering" => {
                self.explicit_ordering = true;
                o.with_ordering(parse_ordering(&lower(key, v)?)?)
            }
            "scaling" => {
                if let Ok(name) = v.extract::<String>() {
                    o.with_scaling(parse_scaling(&name.to_lowercase())?)
                } else if let Ok(arr) = v.extract::<numpy::PyReadonlyArray1<f64>>() {
                    o.with_scaling(ScalingStrategy::External(arr.as_slice()?.to_vec()))
                } else {
                    return Err(bad(key, "a scaling name or a float array of length n", v));
                }
            }
            "pivot_u" => o.with_pivot_u(v.extract().map_err(|_| bad(key, "a float", v))?),
            "nemin" => o.with_nemin(v.extract().map_err(|_| bad(key, "an int", v))?),
            "relax" => {
                if let Ok(on) = v.extract::<bool>() {
                    if on {
                        o.with_relax(SolverSettings::default().relax)
                    } else {
                        o.with_relax(None)
                    }
                } else if let Ok((max_width, max_extra_rows)) = v.extract::<(usize, usize)>() {
                    o.with_relax(Some(RelaxAmalgamation {
                        max_width,
                        max_extra_rows,
                    }))
                } else {
                    return Err(bad(key, "a bool or (max_width, max_extra_rows)", v));
                }
            }
            "reorder" => o.with_reorder(match lower(key, v)?.as_str() {
                "hybrid_liu" | "liu" | "hybrid" => ReorderMode::HybridLiu,
                "off" | "none" => ReorderMode::Off,
                _ => return Err(bad(key, "'hybrid_liu' or 'off'", v)),
            }),
            "blr" => {
                if let Ok(false) = v.extract::<bool>() {
                    o.with_blr(BlrMode::Off)
                } else if let Ok(eps) = v.extract::<f64>() {
                    o.with_blr(BlrMode::contribution_blocks(eps))
                } else if let Ok(d) = v.downcast::<PyDict>() {
                    let get = |name: &str| d.get_item(name).ok().flatten();
                    let eps: f64 = get("eps")
                        .ok_or_else(|| bad(key, "a dict with 'eps'", v))?
                        .extract()
                        .map_err(|_| bad(key, "a float 'eps'", v))?;
                    let mut mode = BlrMode::contribution_blocks(eps);
                    if let BlrMode::ContributionBlocks {
                        min_cnrow,
                        b,
                        adaptive,
                        ..
                    } = &mut mode
                    {
                        if let Some(x) = get("min_cnrow") {
                            *min_cnrow = x.extract().map_err(|_| bad(key, "an int 'min_cnrow'", v))?;
                        }
                        if let Some(x) = get("b") {
                            *b = x.extract().map_err(|_| bad(key, "an int 'b'", v))?;
                        }
                        if let Some(x) = get("adaptive") {
                            *adaptive = x.extract().map_err(|_| bad(key, "a bool 'adaptive'", v))?;
                        }
                    }
                    for (k, _) in d.iter() {
                        let k: String = k.extract()?;
                        if !matches!(k.as_str(), "eps" | "min_cnrow" | "b" | "adaptive") {
                            return Err(bad(key, "a dict with eps, min_cnrow, b, adaptive", v));
                        }
                    }
                    o.with_blr(mode)
                } else {
                    return Err(bad(key, "a float tolerance, False or a dict", v));
                }
            }
            "panel_nb" => o.with_panel_nb(v.extract().map_err(|_| bad(key, "an int", v))?),
            "scalar_gate" | "par_gemm" | "par_cdiv" => {
                let n: usize = v.extract().map_err(|_| bad(key, "an int", v))?;
                let mut t = GemmThresholds {
                    scalar_gate: o.scalar_gate,
                    par_gemm: o.par_gemm,
                    par_cdiv: o.par_cdiv,
                };
                match key {
                    "scalar_gate" => t.scalar_gate = n,
                    "par_gemm" => t.par_gemm = n,
                    _ => t.par_cdiv = n,
                }
                o.with_gemm_thresholds(t)
            }
            "use_gemm_schur" => {
                o.with_use_gemm_schur(v.extract().map_err(|_| bad(key, "a bool", v))?)
            }
            "matching" => o.with_lu_matching(v.extract().map_err(|_| bad(key, "a bool", v))?),
            "interrupt" => {
                let flag: PyInterrupt =
                    v.extract().map_err(|_| bad(key, "an rslab.Interrupt", v))?;
                self.interrupt = Some(flag.clone());
                o.with_interrupt(flag.flag)
            }
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "unknown setting '{other}'"
                )))
            }
        };
        Ok(())
    }

    /// The core settings with the exact-mode pivot policy resolved.
    pub fn resolved(&self) -> SolverSettings {
        let mut o = self.inner.clone();
        if self.force_accept && self.preconditioner.is_none() {
            o = o.with_pivot(ZeroPivotAction::ForceAccept);
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
        d.set_item(
            "method",
            match o.method {
                FactorMethod::Multifrontal => "multifrontal",
                FactorMethod::LeftLooking => "left_looking",
            },
        )?;
        d.set_item(
            "memory",
            match o.memory {
                MemoryMode::Eager => "eager",
                MemoryMode::LowMemory => "low",
            },
        )?;
        d.set_item(
            "ordering",
            if self.explicit_ordering {
                Some(ordering_name(&o.ordering))
            } else {
                None
            },
        )?;
        d.set_item("scaling", scaling_name(&o.scaling))?;
        d.set_item("pivot_u", o.pivot_u)?;
        d.set_item("matching", o.lu_matching)?;
        d.set_item("nemin", o.nemin)?;
        d.set_item(
            "relax",
            match &o.relax {
                Some(r) => PyTuple::new_bound(py, [r.max_width, r.max_extra_rows])
                    .into_any()
                    .unbind(),
                None => false.into_py(py),
            },
        )?;
        d.set_item(
            "reorder",
            match o.reorder {
                ReorderMode::HybridLiu => "hybrid_liu",
                ReorderMode::Off => "off",
            },
        )?;
        d.set_item(
            "blr",
            match &o.blr {
                BlrMode::Off => false.into_py(py),
                BlrMode::ContributionBlocks {
                    eps,
                    min_cnrow,
                    b,
                    adaptive,
                } if BlrMode::contribution_blocks(*eps)
                    == (BlrMode::ContributionBlocks {
                        eps: *eps,
                        min_cnrow: *min_cnrow,
                        b: *b,
                        adaptive: *adaptive,
                    }) =>
                {
                    // Default block parameters: the tolerance alone, as given.
                    eps.into_py(py)
                }
                BlrMode::ContributionBlocks {
                    eps,
                    min_cnrow,
                    b,
                    adaptive,
                } => {
                    let bd = PyDict::new_bound(py);
                    bd.set_item("eps", eps)?;
                    bd.set_item("min_cnrow", min_cnrow)?;
                    bd.set_item("b", b)?;
                    bd.set_item("adaptive", adaptive)?;
                    bd.into_any().unbind()
                }
            },
        )?;
        d.set_item("panel_nb", o.panel_nb)?;
        d.set_item("scalar_gate", o.scalar_gate)?;
        d.set_item("par_gemm", o.par_gemm)?;
        d.set_item("par_cdiv", o.par_cdiv)?;
        d.set_item("use_gemm_schur", o.use_gemm_schur)?;
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
/// pivot_tol : float, default 1e-3
///     Diagonal-preference threshold: the diagonal entry is the pivot when
///     ``|a_jj| >= pivot_tol * max_i |a_ij|``; ``1.0`` is plain partial
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
///     (default) is the structural auto gate (at least 4 blocks, 8000
///     nonzeros, no dominant block); ``True`` / ``False`` force it. The
///     result is bit-identical in every mode.
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
            "pivot_tol" => o.with_pivot_tol(v.extract().map_err(|_| bad(key, "a float", v))?),
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
        d.set_item("pivot_tol", o.pivot_tol)?;
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
