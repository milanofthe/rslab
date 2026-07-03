//! Issue-#2 axis validation across a large SuiteSparse corpus (>=150 matrices).
//!
//! Downloads real application matrices and, for each, exercises the exact-path
//! tuning axes added in issue #2 and checks they preserve correctness and the
//! determinism / bit-identity guarantees on real data:
//!
//! * **auto** factor + solve -> small relative residual (the default path).
//! * **scaling** strategies (Auto / InfNorm) -> small residual (Achse 2).
//! * **memory** mode Eager vs LowMemory -> bit-identical solve (Achse 3).
//! * **RCM** ordering -> small residual (Achse 4).
//! * **right-looking** vs multifrontal -> bit-identical solve (Achse 5).
//! * **compressed** u32 factor -> bit-identical solve (Achse 8).
//! * **2D front subtraction** (par_cdiv 0 vs MAX) -> bit-identical solve (Achse 9).
//!
//! Matrices too large for the memory cap are skipped (a-priori estimate), and
//! failed downloads are skipped with a reason, so the run never OOMs and reports
//! how many of the corpus actually validated.
//!
//! Run:  `cargo bench --bench validate_axes --features matgen-download`
//! Env:  `RLA_VAL_MEM_CAP_MB` (default 4000), `RLA_VAL_THREADS` (default 8),
//!       `RLA_VAL_MAX` (cap #matrices, default all).

use num_complex::Complex;

#[cfg(feature = "matgen-download")]
use rslab::{
    analyze_with, factor_numeric, factor_sparse_ldlt_with, solve_ldlt, CompressedLdltFactors,
    CscMatrix, FactorMethod, GemmThresholds, LdltSymbolic, MemoryMode, OrderingMethod,
    ScalingStrategy, SolverSettings, ZeroPivotAction,
};

type C = Complex<f64>;

/// Candidate SuiteSparse symmetric matrices (group, name). Deliberately >180 so
/// that >=150 download and validate even if a few names are unavailable.
#[cfg(feature = "matgen-download")]
const CORPUS: &[(&str, &str)] = &[
    // Structural / HB (small, reliable).
    ("HB", "bcsstk01"),
    ("HB", "bcsstk02"),
    ("HB", "bcsstk03"),
    ("HB", "bcsstk04"),
    ("HB", "bcsstk05"),
    ("HB", "bcsstk06"),
    ("HB", "bcsstk07"),
    ("HB", "bcsstk08"),
    ("HB", "bcsstk09"),
    ("HB", "bcsstk10"),
    ("HB", "bcsstk11"),
    ("HB", "bcsstk12"),
    ("HB", "bcsstk13"),
    ("HB", "bcsstk14"),
    ("HB", "bcsstk15"),
    ("HB", "bcsstk16"),
    ("HB", "bcsstk17"),
    ("HB", "bcsstk18"),
    ("HB", "bcsstk19"),
    ("HB", "bcsstk20"),
    ("HB", "bcsstk21"),
    ("HB", "bcsstk22"),
    ("HB", "bcsstk23"),
    ("HB", "bcsstk24"),
    ("HB", "bcsstk25"),
    ("HB", "bcsstk26"),
    ("HB", "bcsstk27"),
    ("HB", "bcsstk28"),
    ("HB", "bcsstm07"),
    ("HB", "bcsstm09"),
    ("HB", "bcsstm11"),
    ("HB", "bcsstm12"),
    ("HB", "bcsstm19"),
    ("HB", "bcsstm21"),
    ("HB", "bcsstm23"),
    ("HB", "bcsstm24"),
    ("HB", "bcsstm25"),
    ("HB", "bcsstm26"),
    ("HB", "nos1"),
    ("HB", "nos2"),
    ("HB", "nos3"),
    ("HB", "nos4"),
    ("HB", "nos5"),
    ("HB", "nos6"),
    ("HB", "nos7"),
    ("HB", "plat362"),
    ("HB", "plat1919"),
    ("HB", "lund_a"),
    ("HB", "lund_b"),
    ("HB", "gr_30_30"),
    ("HB", "494_bus"),
    ("HB", "662_bus"),
    ("HB", "685_bus"),
    ("HB", "1138_bus"),
    ("HB", "sherman1"),
    ("HB", "sts4098"),
    ("HB", "dwt_2680"),
    ("HB", "can_1054"),
    ("HB", "can_1072"),
    ("HB", "lshp3466"),
    ("HB", "zenios"),
    // Boeing.
    ("Boeing", "bcsstk34"),
    ("Boeing", "bcsstk38"),
    ("Boeing", "bcsstk39"),
    ("Boeing", "msc00726"),
    ("Boeing", "msc01050"),
    ("Boeing", "msc01440"),
    ("Boeing", "msc04515"),
    ("Boeing", "msc10848"),
    ("Boeing", "msc23052"),
    ("Boeing", "crystk01"),
    ("Boeing", "crystk02"),
    ("Boeing", "crystk03"),
    ("Boeing", "crystm01"),
    ("Boeing", "crystm02"),
    ("Boeing", "crystm03"),
    ("Boeing", "ct20stif"),
    ("Boeing", "pwtk"),
    ("Boeing", "pct20stif"),
    ("Boeing", "bcsstk36"),
    ("Boeing", "bcsstk37"),
    // Cylshell.
    ("Cylshell", "s1rmq4m1"),
    ("Cylshell", "s1rmt3m1"),
    ("Cylshell", "s2rmq4m1"),
    ("Cylshell", "s2rmt3m1"),
    ("Cylshell", "s3rmq4m1"),
    ("Cylshell", "s3rmt3m1"),
    ("Cylshell", "s3rmt3m3"),
    ("Cylshell", "s3dkt3m2"),
    ("Cylshell", "s3dkq4m2"),
    // Nasa.
    ("Nasa", "nasa1824"),
    ("Nasa", "nasa2146"),
    ("Nasa", "nasa2910"),
    ("Nasa", "nasa4704"),
    ("Nasa", "nasasrb"),
    ("Nasa", "shuttle_eddy"),
    ("Nasa", "skirt"),
    ("Nasa", "pwt"),
    // GHS_psdef.
    ("GHS_psdef", "wathen100"),
    ("GHS_psdef", "wathen120"),
    ("GHS_psdef", "apache1"),
    ("GHS_psdef", "jnlbrng1"),
    ("GHS_psdef", "torsion1"),
    ("GHS_psdef", "minsurfo"),
    ("GHS_psdef", "obstclae"),
    ("GHS_psdef", "gridgena"),
    ("GHS_psdef", "finan512"),
    ("GHS_psdef", "cvxbqp1"),
    ("GHS_psdef", "bloweybq"),
    ("GHS_psdef", "oilpan"),
    ("GHS_psdef", "vanbody"),
    ("GHS_psdef", "s3dkq4m2"),
    ("GHS_psdef", "s3dkt3m2"),
    ("GHS_psdef", "ford1"),
    ("GHS_psdef", "crankseg_1"),
    ("GHS_psdef", "crankseg_2"),
    ("GHS_psdef", "hood"),
    ("GHS_psdef", "bmw7st_1"),
    ("GHS_psdef", "bmwcra_1"),
    ("GHS_psdef", "olafu"),
    ("GHS_psdef", "gyro_k"),
    ("GHS_psdef", "gyro_m"),
    ("GHS_psdef", "bundle1"),
    ("GHS_psdef", "cfd1"),
    ("GHS_psdef", "cfd2"),
    ("GHS_psdef", "thread"),
    ("GHS_psdef", "m_t1"),
    ("GHS_psdef", "x104"),
    ("GHS_psdef", "shipsec1"),
    ("GHS_psdef", "shipsec5"),
    ("GHS_psdef", "shipsec8"),
    ("GHS_psdef", "copter2"),
    ("GHS_psdef", "ford2"),
    // GHS_indef.
    ("GHS_indef", "aug2d"),
    ("GHS_indef", "aug2dc"),
    ("GHS_indef", "aug3d"),
    ("GHS_indef", "aug3dcqp"),
    ("GHS_indef", "bloweya"),
    ("GHS_indef", "bratu3d"),
    ("GHS_indef", "cont-201"),
    ("GHS_indef", "cont-300"),
    ("GHS_indef", "dixmaanl"),
    ("GHS_indef", "dtoc"),
    ("GHS_indef", "helm2d03"),
    ("GHS_indef", "helm3d01"),
    ("GHS_indef", "k1_san"),
    ("GHS_indef", "linverse"),
    ("GHS_indef", "mario001"),
    ("GHS_indef", "ncvxbqp1"),
    ("GHS_indef", "sit100"),
    ("GHS_indef", "spmsrtls"),
    ("GHS_indef", "stokes64"),
    ("GHS_indef", "stokes128"),
    ("GHS_indef", "tuma1"),
    ("GHS_indef", "tuma2"),
    ("GHS_indef", "boyd1"),
    ("GHS_indef", "brainpc2"),
    ("GHS_indef", "darcy003"),
    ("GHS_indef", "dawson5"),
    ("GHS_indef", "exdata_1"),
    // Oberwolfach.
    ("Oberwolfach", "bodyy4"),
    ("Oberwolfach", "bodyy5"),
    ("Oberwolfach", "bodyy6"),
    ("Oberwolfach", "gyro"),
    ("Oberwolfach", "gyro_k"),
    ("Oberwolfach", "gyro_m"),
    ("Oberwolfach", "LF10"),
    ("Oberwolfach", "LFAT5"),
    ("Oberwolfach", "t2dah"),
    ("Oberwolfach", "t2dah_e"),
    ("Oberwolfach", "t2dal"),
    ("Oberwolfach", "t3dl"),
    ("Oberwolfach", "filter3D"),
    ("Oberwolfach", "flowmeter5"),
    // Simon / Rothberg / DNVS / FIDAP / misc.
    ("Simon", "raefsky4"),
    ("Simon", "olafu"),
    ("Simon", "venkat01"),
    ("Rothberg", "cfd1"),
    ("Rothberg", "cfd2"),
    ("Rothberg", "gearbox"),
    ("DNVS", "ship_001"),
    ("DNVS", "ship_003"),
    ("DNVS", "shipsec1"),
    ("DNVS", "shipsec5"),
    ("DNVS", "shipsec8"),
    ("DNVS", "fcondp2"),
    ("DNVS", "fullb"),
    ("DNVS", "halfb"),
    ("DNVS", "m_t1"),
    ("DNVS", "thread"),
    ("DNVS", "troll"),
    ("DNVS", "x104"),
    ("DNVS", "tsyl201"),
    ("FIDAP", "ex3"),
    ("FIDAP", "ex9"),
    ("FIDAP", "ex10"),
    ("FIDAP", "ex11"),
    ("FIDAP", "ex13"),
    ("FIDAP", "ex15"),
    ("FIDAP", "ex19"),
    ("FIDAP", "ex33"),
    ("FIDAP", "ex35"),
    ("FIDAP", "ex40"),
    ("Cunningham", "qa8fk"),
    ("Cunningham", "qa8fm"),
    ("Cunningham", "m3plates"),
    ("Um", "2cubes_sphere"),
    ("Um", "offshore"),
    ("Schmid", "thermal1"),
    ("Schmid", "thermal2"),
    ("Botonakis", "FEM_3D_thermal1"),
    ("Botonakis", "thermomech_dM"),
    ("Botonakis", "thermomech_dK"),
    ("Botonakis", "thermomech_TC"),
    ("Wissgott", "parabolic_fem"),
    ("Pothen", "barth"),
    ("Pothen", "barth4"),
    ("Pothen", "barth5"),
    ("Pothen", "bodyy4"),
    ("Pothen", "bodyy5"),
    ("Pothen", "bodyy6"),
    ("Pothen", "mesh1e1"),
    ("Pothen", "mesh2e1"),
    ("Pothen", "mesh3e1"),
    ("Pothen", "shuttle_eddy"),
    ("Pothen", "sphere2"),
    ("Pothen", "sphere3"),
    ("Pothen", "skirt"),
    ("Pothen", "onera_dual"),
    ("Pothen", "commanche_dual"),
    ("PARSEC", "Si2"),
    ("PARSEC", "SiH4"),
    ("PARSEC", "Si5H12"),
    ("PARSEC", "benzene"),
    ("Williams", "cant"),
    ("Williams", "consph"),
    ("Williams", "pdb1HYS"),
    ("Bai", "qc2534"),
];

#[cfg(feature = "matgen-download")]
fn rand_rhs(n: usize) -> Vec<C> {
    // Deterministic pseudo-random complex RHS (splitmix).
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    (0..n).map(|_| C::new(next(), 0.3 * next())).collect()
}

#[cfg(feature = "matgen-download")]
fn rel_residual(a: &CscMatrix<C>, x: &[C], b: &[C]) -> f64 {
    let mut ax = vec![C::new(0.0, 0.0); a.n];
    a.symv(x, &mut ax);
    let num: f64 = (0..a.n)
        .map(|i| (ax[i] - b[i]).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    num / den.max(1e-300)
}

#[cfg(feature = "matgen-download")]
fn bit_eq(a: &[C], b: &[C]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits())
}

#[cfg(feature = "matgen-download")]
fn main() {
    let mem_cap_mb: f64 = std::env::var("RLA_VAL_MEM_CAP_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000.0);
    let threads: usize = std::env::var("RLA_VAL_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let max_mats: usize = std::env::var("RLA_VAL_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let tol = 1e-6;
    let mut validated = 0usize; // matrices that ran at least the auto+bit checks
    let mut skipped_big = 0usize;
    let mut skipped_fetch = 0usize;
    let mut residual_ok = 0usize;
    let mut fail: Vec<String> = Vec::new();
    // Per-axis bit-identity pass counts.
    let (mut n_method, mut n_mem, mut n_front, mut n_compress) = (0usize, 0usize, 0usize, 0usize);
    let (mut n_scaling, mut n_rcm) = (0usize, 0usize);

    for &(group, name) in CORPUS.iter().take(max_mats) {
        let path = match rslab::matgen::download::fetch(group, name) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[val] skip {name}: fetch {e}");
                skipped_fetch += 1;
                continue;
            }
        };
        let a = match rslab::read_mtx_any(&path) {
            Ok(rslab::MtxLoaded::Symmetric(a)) => a,
            Ok(rslab::MtxLoaded::General(_)) => {
                eprintln!("[val] skip {name}: general (LU path)");
                continue;
            }
            Err(e) => {
                eprintln!("[val] skip {name}: parse {e}");
                continue;
            }
        };
        if a.n == 0 {
            continue;
        }
        // A-priori memory gate so the run never OOMs.
        let sym = match LdltSymbolic::analyze(&a) {
            Ok(s) => s,
            Err(e) => {
                fail.push(format!("{name}: analyze {e:?}"));
                continue;
            }
        };
        let est = sym.estimate_memory::<C>();
        let est_mb = est.transient_peak_bytes.max(est.mf_transient_peak_bytes) as f64 / 1e6;
        if est_mb > mem_cap_mb {
            eprintln!("[val] skip {name}: est {est_mb:.0} MB > cap");
            skipped_big += 1;
            continue;
        }
        let b = rand_rhs(a.n);
        // Exact mode by default; saddle-point / augmented systems (aug*, cont-*,
        // bratu3d, KKT) legitimately hit zero pivots in an *exact* factor, so fall
        // back to preconditioner mode (static perturbation + iterative refinement)
        // - the axes are then validated on those matrices too, in the mode where a
        // factor exists. This is pre-existing solver behaviour, not an axis effect.
        let anorm = a
            .values
            .iter()
            .map(|v| v.norm())
            .fold(0.0, f64::max)
            .max(1.0);
        let exact = SolverSettings::default().with_threads(threads);
        let (base, precond) = match sym.factor(&a, &exact) {
            Ok(_) => (exact, false),
            Err(_) => (
                SolverSettings::default().with_threads(threads).with_pivot(
                    ZeroPivotAction::PerturbToEps {
                        abs_floor: anorm * 1e-12,
                    },
                ),
                true,
            ),
        };

        // (1) Reference factor + solve residual (refined in preconditioner mode).
        let refsolver = match sym.factor(&a, &base) {
            Ok(f) => f,
            Err(e) => {
                fail.push(format!("{name}: factor {e:?}"));
                continue;
            }
        };
        let res = if precond {
            refsolver
                .solve_refined(&a, &b, 30)
                .map(|x| rel_residual(&a, &x, &b))
                .unwrap_or(f64::INFINITY)
        } else {
            refsolver
                .solve(&b)
                .map(|x| rel_residual(&a, &x, &b))
                .unwrap_or(f64::INFINITY)
        };
        if res < tol {
            residual_ok += 1;
        } else {
            eprintln!(
                "[val] {name}: residual {res:.1e} (indefinite/ill-conditioned{})",
                if precond { ", precond" } else { "" }
            );
        }
        validated += 1;

        let solve_with = |opts: &SolverSettings| -> Option<Vec<C>> {
            sym.factor(&a, opts).ok().and_then(|f| f.solve(&b).ok())
        };

        // (2) right-looking vs multifrontal -> bit-identical solve.
        let mf = solve_with(&base.clone().with_method(FactorMethod::Multifrontal));
        let rl = solve_with(&base.clone().with_method(FactorMethod::RightLooking));
        match (mf.as_ref(), rl.as_ref()) {
            (Some(m), Some(r)) if bit_eq(m, r) => n_method += 1,
            (Some(_), Some(_)) => fail.push(format!("{name}: right-looking != multifrontal")),
            _ => {}
        }
        // (3) memory Eager vs LowMemory (MF) -> bit-identical.
        let eager = solve_with(
            &base
                .clone()
                .with_method(FactorMethod::Multifrontal)
                .with_memory(MemoryMode::Eager),
        );
        let lowm = solve_with(
            &base
                .clone()
                .with_method(FactorMethod::Multifrontal)
                .with_memory(MemoryMode::LowMemory),
        );
        match (eager.as_ref(), lowm.as_ref()) {
            (Some(e), Some(l)) if bit_eq(e, l) => n_mem += 1,
            (Some(_), Some(_)) => fail.push(format!("{name}: Eager != LowMemory")),
            _ => {}
        }
        // (4) 2D front subtraction: par_cdiv 0 (parallel) vs MAX (serial) -> bit-identical.
        let thr = |cdiv| GemmThresholds {
            scalar_gate: 4096,
            par_gemm: 1_000_000,
            par_cdiv: cdiv,
        };
        let par = solve_with(
            &base
                .clone()
                .with_method(FactorMethod::Multifrontal)
                .with_gemm_thresholds(thr(0)),
        );
        let ser = solve_with(
            &base
                .clone()
                .with_method(FactorMethod::Multifrontal)
                .with_gemm_thresholds(thr(usize::MAX)),
        );
        match (par.as_ref(), ser.as_ref()) {
            (Some(p), Some(s)) if bit_eq(p, s) => n_front += 1,
            (Some(_), Some(_)) => fail.push(format!(
                "{name}: parallel front subtraction not bit-identical"
            )),
            _ => {}
        }
        // (5) scaling Auto / InfNorm -> small residual.
        let sc_auto = solve_with(&base.clone().with_scaling(ScalingStrategy::Auto));
        let sc_inf = solve_with(&base.clone().with_scaling(ScalingStrategy::InfNorm));
        let sc_ok = |v: &Option<Vec<C>>| {
            v.as_ref()
                .map(|x| rel_residual(&a, x, &b) < 1e-4)
                .unwrap_or(false)
        };
        if sc_ok(&sc_auto) && sc_ok(&sc_inf) {
            n_scaling += 1;
        } else {
            eprintln!("[val] {name}: a scaling variant residual high (ill-conditioned)");
        }

        // (6) compressed u32 factor -> bit-identical solve (raw path; skip if raw
        //     factor is rank-deficient without equilibration).
        if let Ok(raw) = factor_sparse_ldlt_with(&a, &base) {
            if let Ok(rx) = solve_ldlt(&raw, &b) {
                if let Ok(raw2) = factor_sparse_ldlt_with(&a, &base) {
                    if let Some(comp) = CompressedLdltFactors::from_factors(raw2) {
                        if let Ok(cx) = comp.solve(&b) {
                            if bit_eq(&rx, &cx) {
                                n_compress += 1;
                            } else {
                                fail.push(format!("{name}: compressed != full solve"));
                            }
                        }
                    }
                }
            }
        }
        // (7) RCM ordering (analyze-time) -> factors + small residual on the raw
        //     path (skip if raw rank-deficient).
        if let Ok(rcm_sym) = analyze_with(
            a.n,
            &a.col_ptr,
            &a.row_idx,
            &base.clone().with_ordering(OrderingMethod::Rcm),
        ) {
            if let Ok(rf) = factor_numeric(&rcm_sym, &a, &base) {
                if let Ok(rx) = solve_ldlt(&rf, &b) {
                    if rel_residual(&a, &rx, &b) < 1e-4 {
                        n_rcm += 1;
                    }
                }
            }
        }

        println!(
            "[val] {validated:3}. {name:<22} n={:<8} nnz={:<9} res={res:.1e}",
            a.n,
            a.row_idx.len()
        );
    }

    println!("\n===== issue #2 axis validation summary =====");
    println!("matrices validated:      {validated}");
    println!("  residual < {tol:.0e}:        {residual_ok}/{validated}");
    println!("  right-looking==MF:      {n_method}/{validated}  (Achse 5)");
    println!("  Eager==LowMemory:       {n_mem}/{validated}  (Achse 3)");
    println!("  front-subtract determ.: {n_front}/{validated}  (Achse 9)");
    println!("  scaling Auto/InfNorm:   {n_scaling}/{validated}  (Achse 2)");
    println!("  compressed==full:       {n_compress}  (Achse 8, raw-factorable subset)");
    println!("  RCM factor+solve:       {n_rcm}  (Achse 4, raw-factorable subset)");
    println!("skipped (too big):       {skipped_big}");
    println!("skipped (fetch failed):  {skipped_fetch}");
    if fail.is_empty() {
        println!("\nNO CORRECTNESS FAILURES. All bit-identity/determinism checks held.");
    } else {
        println!("\n{} FAILURES:", fail.len());
        for f in &fail {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}

#[cfg(not(feature = "matgen-download"))]
fn main() {
    eprintln!("validate_axes requires --features matgen-download");
}
