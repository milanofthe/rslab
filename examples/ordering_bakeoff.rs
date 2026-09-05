//! Ordering bake-off on a dumped complex unsymmetric matrix (rapidmom near-field
//! preconditioner): fill, factor time and multi-RHS solve time per `OrderingMethod`.
//!
//! File format (little endian): u64 n, u64 nnz, u32 rows[nnz], u32 cols[nnz], (f64 re, f64 im)[nnz].
//! Duplicates are summed. Run: `cargo run --release --example ordering_bakeoff -- <file> [nrhs]`.
use num_complex::Complex;
use rslab::{GeneralCsc, LuSolver, OrderingMethod, SolverSettings, ZeroPivotAction};
use std::io::Read;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("dump path");
    let nrhs: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let mut buf = Vec::new();
    std::fs::File::open(&path)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let rd64 = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()) as usize;
    let (n, nnz) = (rd64(0), rd64(8));
    let mut o = 16;
    let rows: Vec<usize> = (0..nnz)
        .map(|k| u32::from_le_bytes(buf[o + 4 * k..o + 4 * k + 4].try_into().unwrap()) as usize)
        .collect();
    o += 4 * nnz;
    let cols: Vec<usize> = (0..nnz)
        .map(|k| u32::from_le_bytes(buf[o + 4 * k..o + 4 * k + 4].try_into().unwrap()) as usize)
        .collect();
    o += 4 * nnz;
    let vals: Vec<Complex<f64>> = (0..nnz)
        .map(|k| {
            let re = f64::from_le_bytes(buf[o + 16 * k..o + 16 * k + 8].try_into().unwrap());
            let im = f64::from_le_bytes(buf[o + 16 * k + 8..o + 16 * k + 16].try_into().unwrap());
            Complex::new(re, im)
        })
        .collect();
    let a = GeneralCsc::from_triplets(n, &rows, &cols, &vals).expect("csc");
    println!("n={n} nnz(triplets)={nnz} nrhs={nrhs}");
    let b: Vec<Complex<f64>> = (0..n * nrhs)
        .map(|k| Complex::new((k % 7) as f64 - 3.0, (k % 5) as f64 - 2.0))
        .collect();
    for (name, ord) in [
        ("Amd", OrderingMethod::Amd),
        ("Amf", OrderingMethod::Amf),
        ("MetisND", OrderingMethod::MetisND),
        ("Rcm", OrderingMethod::Rcm),
    ] {
        let opts = SolverSettings::exact()
            .with_pivot(ZeroPivotAction::PerturbToEps { abs_floor: 1e-6 })
            .with_ordering(ord);
        let t0 = Instant::now();
        let lu = match LuSolver::factor(&a, &opts) {
            Ok(s) => s,
            Err(e) => {
                println!("{name:8}: factor failed: {e:?}");
                continue;
            }
        };
        let t_f = t0.elapsed().as_secs_f64();
        let fill = lu.factor_nnz();
        let _ = lu.solve_many(&b, nrhs).unwrap();
        let t1 = Instant::now();
        let reps = 10;
        for _ in 0..reps {
            let _ = lu.solve_many(&b, nrhs).unwrap();
        }
        let t_s = t1.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{name:8}: fill={fill} ({:.2}x nnz, {:.0} MB)  factor={t_f:.2}s  solve({nrhs} rhs)={:.1} ms",
            fill as f64 / nnz as f64,
            fill as f64 * 24.0 / 1e6,
            t_s * 1e3
        );
    }
}
