//! The core's events from Python: a threshold and a sink. Every call into
//! the core runs without the GIL in the module's rsprobe context ([`run`]),
//! built at the threshold; the events it collects reach the sink on the
//! calling thread once the call returns.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rslab::rsprobe::{Buffer, Context, Level};
use std::sync::{Arc, Mutex};

/// The threshold (`None`: off), the sink and the context built for them.
struct State {
    level: Option<Level>,
    sink: Option<PyObject>,
    ctx: Option<Arc<Context>>,
    buffer: Buffer,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// `f` on the module's state, made on first use at WARN without a sink.
fn state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let s = guard.get_or_insert_with(|| State {
        level: Some(Level::WARN),
        sink: None,
        ctx: None,
        buffer: Buffer::default(),
    });
    f(s)
}

/// The level of a name, as Python's `logging` spells them (`None`: off).
fn parse(name: &str) -> Option<Option<Level>> {
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "trace" => Some(Level::TRACE),
        "debug" => Some(Level::DEBUG),
        "info" => Some(Level::INFO),
        "warning" | "warn" => Some(Level::WARN),
        "error" => Some(Level::ERROR),
        "off" | "none" => None,
        _ => return None,
    })
}

/// The name of a level as Python's `logging` spells it.
fn name(level: Option<Level>) -> &'static str {
    match level {
        None => "off",
        Some(Level::WARN) => "warning",
        Some(l) => l.name(),
    }
}

/// `f` without the GIL in the module's context, then its events to the sink
/// (or to stderr without one).
pub fn run<R: Send>(py: Python<'_>, f: impl Send + FnOnce() -> R) -> R {
    let (ctx, buffer) = state(|s| {
        if s.ctx.is_none() {
            if let Some(level) = s.level {
                let ctx = Context::builder()
                    .threshold(level)
                    .sink(s.buffer.clone())
                    .build();
                s.ctx = Some(Arc::new(ctx));
            }
        }
        (s.ctx.clone(), s.buffer.clone())
    });
    let out = py.allow_threads(|| {
        let _on = ctx.as_ref().map(|c| c.enter());
        f()
    });
    let sink = state(|s| s.sink.as_ref().map(|f| f.clone_ref(py)));
    buffer.drain(|event, path| {
        let level = name(Some(event.level));
        let text = match path {
            "" => event.kind.to_string(),
            p => format!("{p}: {}", event.kind),
        };
        match &sink {
            Some(f) => {
                if let Err(e) = f.call1(py, (level, text)) {
                    e.print(py);
                }
            }
            None => eprintln!("rslab {level}: {text}"),
        }
    });
    out
}

/// Set the log level of the solver core.
///
/// Parameters
/// ----------
/// level : str
///     ``'trace'``, ``'debug'``, ``'info'``, ``'warning'`` (default),
///     ``'error'`` or ``'off'``.
///
/// Returns
/// -------
/// None
#[pyfunction]
pub fn set_log_level(level: &str) -> PyResult<()> {
    let Some(l) = parse(level) else {
        return Err(PyValueError::new_err(format!(
            "level must be 'trace', 'debug', 'info', 'warning', 'error' or 'off', got '{level}'"
        )));
    };
    state(|s| {
        s.level = l;
        s.ctx = None;
    });
    Ok(())
}

/// The current log level of the solver core.
///
/// Returns
/// -------
/// str
///     The level name in lowercase.
#[pyfunction]
pub fn log_level() -> String {
    state(|s| name(s.level).to_string())
}

/// Route the core's log messages to ``sink(level, message)`` instead of
/// stderr; ``None`` restores stderr. The sink receives the lowercase level
/// name and the message, prefixed with the solver phase it was emitted in
/// (``factor: ...``), so a ``logging.Logger`` fits directly::
///
///     log = logging.getLogger("rslab")
///     rslab.set_log_sink(lambda level, msg: log.log(logging.getLevelName(level.upper()), msg))
///
/// The sink is called on the calling thread once the call into the solver
/// returns.
///
/// Parameters
/// ----------
/// sink : callable or None
///     ``sink(level, message)``; ``None`` restores stderr.
///
/// Returns
/// -------
/// None
#[pyfunction]
#[pyo3(signature = (sink))]
pub fn set_log_sink(sink: Option<PyObject>) {
    state(|s| s.sink = sink);
}
