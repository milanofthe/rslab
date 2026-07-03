//! Probe which SuiteSparse candidates load, their path (symmetric LDLᵀ vs general
//! LU), size, and whether they carry genuinely complex values (max |imag|). Used
//! to assemble a complex, both-paths benchmark corpus. Hermitian files are rejected
//! by `read_mtx_any` and reported as skipped.
//!
//! Run: `RLA_PROBE=grp/name,grp/name cargo bench --bench probe_complex --features matgen-download`

#[cfg(feature = "matgen-download")]
fn main() {
    let list = std::env::var("RLA_PROBE").unwrap_or_default();
    let mut sym = 0usize;
    let mut gen = 0usize;
    let mut complex = 0usize;
    println!(
        "{:<28}{:>8}{:>10}{:>12}{:>14}",
        "matrix", "type", "n", "nnz", "max|imag|"
    );
    println!("{}", "-".repeat(74));
    for gn in list.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let Some((group, name)) = gn.split_once('/') else {
            continue;
        };
        let path = match rslab::matgen::download::fetch(group, name) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:<28}{:>8}  fetch/skip: {}",
                    name,
                    "-",
                    e.lines().next().unwrap_or("")
                );
                continue;
            }
        };
        match rslab::read_mtx_any(&path) {
            Ok(rslab::MtxLoaded::Symmetric(a)) => {
                let mi = a.values.iter().map(|v| v.im.abs()).fold(0.0, f64::max);
                if mi > 0.0 {
                    complex += 1;
                }
                sym += 1;
                println!(
                    "{:<28}{:>8}{:>10}{:>12}{:>14.2e}",
                    name,
                    "sym",
                    a.n,
                    a.values.len(),
                    mi
                );
            }
            Ok(rslab::MtxLoaded::General(a)) => {
                let mi = a.values.iter().map(|v| v.im.abs()).fold(0.0, f64::max);
                if mi > 0.0 {
                    complex += 1;
                }
                gen += 1;
                println!(
                    "{:<28}{:>8}{:>10}{:>12}{:>14.2e}",
                    name,
                    "general",
                    a.n,
                    a.values.len(),
                    mi
                );
            }
            Err(e) => println!("{:<28}{:>8}  reject: {:?}", name, "-", e),
        }
    }
    println!(
        "\nloaded: {} symmetric (LDLt), {} general (LU); {} carry complex values",
        sym, gen, complex
    );
}

#[cfg(not(feature = "matgen-download"))]
fn main() {
    eprintln!("requires --features matgen-download");
}
