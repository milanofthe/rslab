//! Multifrontal LU **memory accounting** — diagnoses the transient peak that
//! causes OOMs. Uses only the (cheap) symbolic analysis `front_dims` so it never
//! runs the (potentially OOM-ing) numeric factorization.
//!
//! For each `*.mtx` it reports, in 16-byte complex-f64 units:
//!   * retained dense `L`+`U` panels   = Σ 2·nrow·ncol  (held until the global emit)
//!   * retained dense contribution `CB` = Σ (nrow−ncol)² (held until the parent consumes — currently to the end)
//!   * peak single-front transient      = max(nrow²[fbuf] + 2·nrow·ncol[l,u] + (nrow−ncol)²[cb])
//!   * the largest front (ncol, nrow)
//!
//! Run: `cargo bench --bench front_mem`.

use rslab::prelude::*;
use rslab::LuSymbolic;

const DIR: &str = r"C:\Repositories\rapidmom\precond_matrices";
const B: f64 = 16.0; // bytes per Complex<f64>
const MB: f64 = 1e6;

fn analyze_file(path: &std::path::Path) {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            println!("{name}: read error {e}");
            return;
        }
    };
    let mtx = match parse_mtx_complex_general(&contents, &name) {
        Ok(m) => m,
        Err(e) => {
            println!("{name}: parse error {e}");
            return;
        }
    };
    drop(contents);
    let a = match mtx.to_general_csc() {
        Ok(a) => a,
        Err(e) => {
            println!("{name}: build error {e}");
            return;
        }
    };
    let n = a.n;
    let sym = match LuSymbolic::analyze(&a) {
        Ok(s) => s,
        Err(e) => {
            println!("{name}: analyze error {e:?}");
            return;
        }
    };
    let dims = sym.front_dims(); // (ncol, nrow) per supernode

    let mut lu_retained = 0.0f64; // Σ 2·nrow·ncol
    let mut cb_retained = 0.0f64; // Σ cnrow²
    let mut peak_front = 0.0f64; // max per-front transient
    let mut peak_dim = (0usize, 0usize);
    for &(ncol, nrow) in &dims {
        let cnrow = nrow - ncol;
        let lu = 2.0 * (nrow as f64) * (ncol as f64);
        let cb = (cnrow as f64) * (cnrow as f64);
        let fbuf = (nrow as f64) * (nrow as f64);
        lu_retained += lu;
        cb_retained += cb;
        let transient = fbuf + lu + cb; // fbuf + l + u + cb all alive in lu_front
        if transient > peak_front {
            peak_front = transient;
            peak_dim = (ncol, nrow);
        }
    }
    // Lower bound on the live heap at the moment the largest front is being
    // extracted: everything retained so far (≈ all L/U + all CB, since nothing is
    // freed until the global emit) plus that front's own fbuf transient.
    let retained = (lu_retained + cb_retained) * B / MB;
    let peak_lo =
        (lu_retained + cb_retained + peak_front - /*its lu+cb already counted*/ 0.0) * B / MB;
    println!(
        "{name:42} n={n:>6}  L+U(dense) {:8.0} MB  CB(dense) {:8.0} MB  retained {:8.0} MB  \
         peak-front {:7.0} MB (ncol={},nrow={})  ≈peak≥ {:8.0} MB",
        lu_retained * B / MB,
        cb_retained * B / MB,
        retained,
        peak_front * B / MB,
        peak_dim.0,
        peak_dim.1,
        peak_lo,
    );
}

fn main() {
    let mut files: Vec<_> = match std::fs::read_dir(DIR) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect(),
        Err(e) => {
            println!("cannot read {DIR}: {e}");
            return;
        }
    };
    files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    let filter = std::env::var("RLA_FILTER").unwrap_or_default();
    println!("Multifrontal LU dense-memory accounting (symbolic only)\n");
    for f in &files {
        if filter.is_empty() || f.file_name().unwrap().to_string_lossy().contains(&filter) {
            analyze_file(f);
        }
    }
}
