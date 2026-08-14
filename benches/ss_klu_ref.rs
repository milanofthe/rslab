//! Shared bench-only loader for the SuiteSparse KLU reference shim
//! (`benches/klu_shim.c`). Compiled at bench runtime against a locally
//! installed SuiteSparse and loaded via `libloading` — the same
//! runtime-reference pattern as the MKL / Accelerate shims; nothing from
//! SuiteSparse is distributed with RSLAB, it is only measured against when
//! present on the machine.
//!
//! Prefix resolution (`RLA_KLU_SS_PREFIX` always wins):
//! * macOS: `/opt/homebrew` (Homebrew), links `-lklu` from `<prefix>/lib`.
//! * Linux: `/usr`, links `-lklu`.
//! * Windows: no default — point `RLA_KLU_SS_PREFIX` at a directory with
//!   `bin/klu.dll` and `include/suitesparse/klu.h` (e.g. a conda-forge
//!   `suitesparse` env's `Library` dir). The shim links directly against the
//!   DLL (MinGW GCC understands that), and `<prefix>/bin` is prepended to
//!   `PATH` so KLU's own dependencies (amd, btf, colamd, suitesparseconfig)
//!   resolve when the shim is loaded.
//!
//! Consumed by `klu_circuit.rs` and `klu_realworld.rs` via `#[path]` module
//! include (benches cannot share a crate-internal module).

use rslab::GeneralCsc;

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct SsKluResult {
    pub ana_s: f64,
    pub fac_s: f64,
    pub refac_s: f64,
    pub slv_s: f64,
    pub sweep_s: f64,
    pub lnz: i64,
    pub unz: i64,
    pub nblocks: i32,
    pub ok: i32,
}

type SsKluFn = unsafe extern "C" fn(
    i32,        // n
    *const i32, // Ap
    *const i32, // Ai
    *const f64, // Ax
    *const f64, // b
    *mut f64,   // x
    i32,        // sweep_len
    *mut SsKluResult,
) -> i32;

pub struct SsKlu {
    _lib: libloading::Library,
    f: SsKluFn,
}

/// `(compile ok, shim path)` or `None` when no prefix/compiler is available.
fn build_shim() -> Option<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("benches/klu_shim.c");
    let prefix = match std::env::var("RLA_KLU_SS_PREFIX") {
        Ok(p) => p,
        Err(_) => {
            if cfg!(target_os = "macos") {
                "/opt/homebrew".into()
            } else if cfg!(target_os = "linux") {
                "/usr".into()
            } else {
                eprintln!(
                    "[ss-klu] set RLA_KLU_SS_PREFIX to a SuiteSparse install \
                     (bin/klu.dll + include/suitesparse/klu.h) to enable the reference"
                );
                return None;
            }
        }
    };
    let (ext, cc, shared_flag) = if cfg!(target_os = "windows") {
        ("dll", "gcc", "-shared")
    } else if cfg!(target_os = "macos") {
        ("dylib", "cc", "-dynamiclib")
    } else {
        ("so", "cc", "-shared")
    };
    let dylib = root.join(format!("target/klu_shim.{ext}"));
    let mtime = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let stale = match (mtime(&src), mtime(&dylib)) {
        (Some(s), Some(d)) => s > d,
        _ => true,
    };
    if stale {
        let mut cmd = std::process::Command::new(cc);
        cmd.args(["-O2", "-std=c11", shared_flag])
            .arg(format!("-I{prefix}/include/suitesparse"));
        if cfg!(target_os = "windows") {
            // MinGW links straight against the DLL; no import lib needed.
            cmd.arg(&src).arg(format!("{prefix}/bin/klu.dll"));
        } else {
            cmd.arg(format!("-L{prefix}/lib")).arg("-lklu").arg(&src);
        }
        if cfg!(target_os = "linux") {
            cmd.arg("-fPIC");
        }
        let st = cmd.arg("-o").arg(&dylib).status();
        if !st.map(|s| s.success()).unwrap_or(false) {
            eprintln!("[ss-klu] shim compile failed (SuiteSparse installed at {prefix}?)");
            return None;
        }
    }
    if cfg!(target_os = "windows") {
        // Make klu.dll and its dependencies resolvable when the shim loads.
        let path = std::env::var("PATH").unwrap_or_default();
        let bin = format!("{prefix}\\bin");
        if !path.split(';').any(|p| p.eq_ignore_ascii_case(&bin)) {
            std::env::set_var("PATH", format!("{bin};{path}"));
        }
    }
    Some(dylib)
}

impl SsKlu {
    pub fn try_new() -> Option<Self> {
        let dylib = build_shim()?;
        let lib = unsafe { libloading::Library::new(&dylib).ok()? };
        let f: SsKluFn = unsafe {
            let s: libloading::Symbol<SsKluFn> = lib.get(b"klu_shim_run").ok()?;
            *s
        };
        Some(SsKlu { _lib: lib, f })
    }

    pub fn run(
        &self,
        a: &GeneralCsc<f64>,
        b: &[f64],
        sweep: usize,
    ) -> Option<(SsKluResult, Vec<f64>)> {
        let ap: Vec<i32> = a.col_ptr.iter().map(|&v| v as i32).collect();
        let ai: Vec<i32> = a.row_idx.iter().map(|&v| v as i32).collect();
        let mut x = vec![0.0f64; a.n];
        let mut r = SsKluResult::default();
        let st = unsafe {
            (self.f)(
                a.n as i32,
                ap.as_ptr(),
                ai.as_ptr(),
                a.values.as_ptr(),
                b.as_ptr(),
                x.as_mut_ptr(),
                sweep as i32,
                &mut r,
            )
        };
        if st != 0 || r.ok != 1 {
            eprintln!("[ss-klu] failed with status {st}");
            return None;
        }
        Some((r, x))
    }
}
