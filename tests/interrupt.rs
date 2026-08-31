//! Cooperative cancellation of the numeric factorization.
//!
//! The contract under test, for all three paths: an armed flag stops the factor
//! at the next supernode / dense-panel / block boundary with
//! `RslabError::Interrupted`, the solver only ever reads the flag, and clearing
//! it makes the next factorization run cleanly. The symbolic analysis is not
//! interruptible and is not exercised here.

use rslab::matgen::{fem, stencil};
use rslab::prelude::*;
use rslab::RslabError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

type C = num_complex::Complex<f64>;

fn helmholtz(k: usize) -> rslab::CscMatrix<C> {
    stencil::helmholtz(
        &[k, k, k],
        C::new(0.05, 0.02),
        &stencil::StencilOpts::default(),
    )
}

fn convdiff(k: usize) -> rslab::GeneralCsc<C> {
    fem::convection_diffusion::<C>(&[k, k], 5e-3, fem::Flow::Rotating, true)
}

/// A pre-armed flag is observed at the first boundary, so the outcome is
/// deterministic and does not race a watchdog.
#[test]
fn armed_flag_interrupts_every_path() {
    let flag = Arc::new(AtomicBool::new(true));

    let a = helmholtz(12);
    let s = SolverSettings::default().with_interrupt(flag.clone());
    let sym = LdltSymbolic::analyze_with(&a, &s).unwrap();
    assert!(matches!(sym.factor(&a, &s), Err(RslabError::Interrupted)));

    let b = convdiff(40);
    let sym = LuSymbolic::analyze_with(&b, &s).unwrap();
    assert!(matches!(sym.factor(&b, &s), Err(RslabError::Interrupted)));

    let ks = KluSettings::default().with_interrupt(flag.clone());
    let sym = KluSymbolic::analyze(&b).unwrap();
    assert!(matches!(sym.factor(&b, &ks), Err(RslabError::Interrupted)));
}

/// Clearing the flag restores a clean factorization: no state is left behind by
/// the interrupted attempt.
#[test]
fn cleared_flag_factors_cleanly() {
    let flag = Arc::new(AtomicBool::new(true));
    let a = helmholtz(12);
    let s = SolverSettings::default().with_interrupt(flag.clone());
    let sym = LdltSymbolic::analyze_with(&a, &s).unwrap();
    assert!(sym.factor(&a, &s).is_err());

    flag.store(false, Ordering::Relaxed);
    let f = sym
        .factor(&a, &s)
        .expect("re-factor after clearing the flag");
    let rhs = vec![C::new(1.0, 0.0); a.n];
    let x = f.solve(&rhs).unwrap();
    let mut ax = vec![C::new(0.0, 0.0); a.n];
    a.symv(&x, &mut ax);
    let num: f64 = ax
        .iter()
        .zip(&rhs)
        .map(|(p, q)| (*p - *q).norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(num.sqrt() < 1e-6, "residual after the clean re-factor");
}

/// Armed mid-flight from another thread: the factor must return `Interrupted`
/// rather than run to completion. The matrix is large enough that a 20 ms delay
/// lands well inside the factorization.
#[test]
fn flag_set_during_the_factor_stops_it() {
    let a = helmholtz(34);
    let flag = Arc::new(AtomicBool::new(false));
    let s = SolverSettings::default().with_interrupt(flag.clone());
    let sym = LdltSymbolic::analyze_with(&a, &s).unwrap();

    let watchdog = {
        let flag = flag.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            flag.store(true, Ordering::Relaxed);
        })
    };
    let outcome = sym.factor(&a, &s);
    watchdog.join().unwrap();
    assert!(
        matches!(outcome, Err(RslabError::Interrupted)),
        "expected Interrupted, got {:?}",
        outcome.map(|f| f.factor_nnz())
    );
}

/// An unarmed flag changes nothing: same factor, bit for bit.
#[test]
fn unarmed_flag_is_bit_identical() {
    let a = helmholtz(14);
    let rhs = vec![C::new(1.0, 0.0); a.n];
    let plain = SolverSettings::default();
    let armed = SolverSettings::default().with_interrupt(Arc::new(AtomicBool::new(false)));

    let x0 = LdltSymbolic::analyze_with(&a, &plain)
        .unwrap()
        .factor(&a, &plain)
        .unwrap()
        .solve(&rhs)
        .unwrap();
    let x1 = LdltSymbolic::analyze_with(&a, &armed)
        .unwrap()
        .factor(&a, &armed)
        .unwrap()
        .solve(&rhs)
        .unwrap();
    for (p, q) in x0.iter().zip(&x1) {
        assert_eq!(p.re.to_bits(), q.re.to_bits());
        assert_eq!(p.im.to_bits(), q.im.to_bits());
    }
}

/// The flag armed for the factorization is carried into the factors, so a KLU
/// refactor on them observes it too.
#[test]
fn klu_refactor_observes_the_stored_flag() {
    let b = convdiff(40);
    let flag = Arc::new(AtomicBool::new(false));
    let ks = KluSettings::default().with_interrupt(flag.clone());
    let sym = KluSymbolic::analyze(&b).unwrap();
    let mut f = sym.factor(&b, &ks).expect("clean factor while unarmed");

    flag.store(true, Ordering::Relaxed);
    assert!(matches!(f.refactor(&b), Err(RslabError::Interrupted)));

    flag.store(false, Ordering::Relaxed);
    f.refactor(&b).expect("refactor after clearing the flag");
}
