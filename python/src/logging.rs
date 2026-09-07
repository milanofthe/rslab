//! The core's logger from Python: level and sink.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rslab::{LogLevel, LogSink};

/// Set the log level of the solver core.
///
/// Parameters
/// ----------
/// level : str
///     ``'debug'``, ``'info'``, ``'warning'`` (default), ``'error'`` or
///     ``'off'``. The environment variable ``RLA_LOG`` sets the initial
///     level.
///
/// Returns
/// -------
/// None
#[pyfunction]
pub fn set_log_level(level: &str) -> PyResult<()> {
    match LogLevel::parse(level) {
        Some(l) => {
            rslab::logging::set_level(l);
            Ok(())
        }
        None => Err(PyValueError::new_err(format!(
            "level must be 'debug', 'info', 'warning', 'error' or 'off', got '{level}'"
        ))),
    }
}

/// The current log level of the solver core.
///
/// Returns
/// -------
/// str
///     The level name in lowercase.
#[pyfunction]
pub fn log_level() -> String {
    rslab::logging::level().label().to_ascii_lowercase()
}

struct PySink(PyObject);

impl LogSink for PySink {
    fn emit(&self, level: LogLevel, msg: &str) {
        // Log calls may come from worker threads while the calling Python
        // thread has released the GIL for the factorization; take it here.
        Python::with_gil(|py| {
            if let Err(e) = self.0.call1(py, (level.label().to_ascii_lowercase(), msg)) {
                e.print(py);
            }
        });
    }
}

/// Route the core's log messages to ``sink(level, message)`` instead of the
/// default stdout / stderr writer; ``None`` restores the default. The sink
/// receives the lowercase level name and the bare message (no timestamp), so
/// a ``logging.Logger`` fits directly::
///
///     log = logging.getLogger("rslab")
///     rslab.set_log_sink(lambda level, msg: log.log(logging.getLevelName(level.upper()), msg))
///
/// The sink may be called from solver worker threads.
///
/// Parameters
/// ----------
/// sink : callable or None
///     ``sink(level, message)``; ``None`` restores the default writer.
///
/// Returns
/// -------
/// None
#[pyfunction]
#[pyo3(signature = (sink))]
pub fn set_log_sink(sink: Option<PyObject>) {
    match sink {
        Some(f) => rslab::logging::set_sink(Box::new(PySink(f))),
        None => rslab::logging::reset_sink(),
    }
}
