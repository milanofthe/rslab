/* Bench-only FFI shim for Apple Accelerate Sparse Solvers (macOS 15.5+).
 *
 * The public Sparse API (`SparseFactor` / `SparseSolve`) consists of
 * `static inline __attribute__((overloadable))` header wrappers, so it cannot
 * be reached through dlopen alone; this shim compiles those wrappers into a
 * tiny dylib with a plain C ABI that `bench_suite` loads via libloading at
 * runtime (same pattern as the MKL PARDISO reference - the solver library
 * itself stays 100% pure Rust).
 *
 * Configuration mirrors the vendor defaults from SolveImplementation.h
 * (ordering `SparseOrderDefault`, scaling `SparseScalingDefault`,
 * `pivotTolerance = 0.01` for complex double), with two bench-specific
 * changes:
 *   - `reportError` is set to a logging callback so a parameter error cannot
 *     `__builtin_trap()` the whole sweep;
 *   - `malloc`/`free` are instrumented with atomic live/peak counters
 *     (`malloc_size`-based), so the reported peak is the live-bytes peak of
 *     everything Accelerate allocates through the callbacks - the same metric
 *     the Rust solvers report through the counting global allocator.
 *
 * Build (done automatically by bench_suite when the `accel` solver is on):
 *   cc -O2 -std=c11 -dynamiclib -framework Accelerate accel_shim.c -o accel_shim.dylib
 */
#include <Accelerate/Accelerate.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc/malloc.h>
#include <time.h>

typedef struct {
    double re, im;
} shim_cplx;

typedef struct {
    double ana_s;           /* symbolic factorization wall time [s] */
    double fac_s;           /* numeric factorization wall time [s] */
    double slv_s;           /* solve wall time [s] */
    int64_t peak_bytes;     /* live-bytes peak through the instrumented malloc */
    int64_t factor_bytes;   /* a-priori factor size reported by the symbolic phase */
    int64_t workspace_bytes;/* a-priori numeric workspace reported by the symbolic phase */
    int32_t sym_status;     /* SparseStatus_t of the symbolic factorization */
    int32_t num_status;     /* SparseStatus_t of the numeric factorization */
} shim_result;

/* ---- instrumented allocator (live/peak, exact, thread-safe) -------------- */
static _Atomic int64_t live_bytes = 0;
static _Atomic int64_t peak_bytes = 0;

/* Plain system malloc plus `malloc_size` book-keeping. No size header: the LU
 * symbolic path frees pointers through this callback that were NOT allocated
 * through the malloc callback (observed via ___BUG_IN_CLIENT_OF_LIBMALLOC in
 * `_SparseSymbolicFactorLU`), so the pointers handed out must stay ordinary
 * system-malloc pointers for the mixed paths to be harmless. */
static void *counted_malloc(size_t size) {
    void *p = malloc(size);
    if (!p) return NULL;
    int64_t sz = (int64_t)malloc_size(p);
    int64_t now = atomic_fetch_add(&live_bytes, sz) + sz;
    int64_t prev = atomic_load(&peak_bytes);
    while (now > prev && !atomic_compare_exchange_weak(&peak_bytes, &prev, now)) {}
    return p;
}
static void counted_free(void *ptr) {
    if (!ptr) return;
    atomic_fetch_sub(&live_bytes, (int64_t)malloc_size(ptr));
    free(ptr);
}

static void report_error(const char *message) {
    fprintf(stderr, "[accel] parameter error: %s\n", message);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Solve A x = b (complex double) with the requested Accelerate factorization.
 *
 * n         : dimension
 * col_ptr   : CSC column starts, length n+1 (long, as the API wants)
 * row_idx   : CSC row indices (int)
 * vals      : CSC values, {re,im} pairs (layout-identical to double _Complex)
 * symmetric : 1 -> kind = SparseSymmetric, lower triangle stored (LDLT/Cholesky);
 *             2 -> kind = SparseHermitian, lower triangle stored;
 *             0 -> kind = SparseOrdinary (LU/QR)
 * fact_type : raw SparseFactorization_t value (1 = LDLT, 80 = LU, ...)
 * order     : raw SparseOrder_t value, or -1 for SparseOrderDefault
 * pivot_tol : threshold-pivot tolerance, or < 0 for the vendor default (0.01)
 * max_bytes : memory gate; if > 0 and the symbolic phase predicts
 *             factor + workspace above this budget, bail out with -100
 *             before any numeric work (keeps a 16 GB machine out of swap)
 * b, x      : right-hand side / solution, length n
 *
 * Returns 0 on success, -100 if the memory gate fired, the failing
 * SparseStatus_t otherwise. Timings, the live-bytes peak, and the a-priori
 * factor/workspace sizes land in *out.
 */
int accel_sparse_solve_complex(int32_t n, const long *col_ptr, const int32_t *row_idx,
                               const shim_cplx *vals, int32_t symmetric, int32_t fact_type,
                               int32_t order, double pivot_tol, int64_t max_bytes,
                               const shim_cplx *b, shim_cplx *x, shim_result *out) {
    memset(out, 0, sizeof(*out));
    atomic_store(&live_bytes, 0);
    atomic_store(&peak_bytes, 0);

    SparseAttributesComplex_t attr;
    memset(&attr, 0, sizeof(attr));
    if (symmetric == 1) {
        attr.kind = SparseSymmetric;
        attr.triangle = SparseLowerTriangle;
    } else if (symmetric == 2) {
        attr.kind = SparseHermitian;
        attr.triangle = SparseLowerTriangle;
    } else {
        attr.kind = SparseOrdinary;
    }
    SparseMatrixStructureComplex structure = {
        .rowCount = n,
        .columnCount = n,
        .columnStarts = (long *)col_ptr,
        .rowIndices = (int *)row_idx,
        .attributes = attr,
        .blockSize = 1,
    };
    SparseMatrix_Complex_Double a = {
        .structure = structure,
        .data = (__SPARSE_double_complex *)vals,
    };

    /* Vendor defaults (SolveImplementation.h), plus logging + counting hooks. */
    SparseSymbolicFactorOptions sfo = {
        .control = SparseDefaultControl,
        .orderMethod = order < 0 ? SparseOrderDefault : (SparseOrder_t)order,
        .order = NULL,
        .ignoreRowsAndColumns = NULL,
        .malloc = counted_malloc,
        .free = counted_free,
        .reportError = report_error,
    };
    SparseNumericFactorOptions nfo = {
        .control = SparseDefaultControl,
        .scalingMethod = SparseScalingDefault,
        .scaling = NULL,
        .pivotTolerance = pivot_tol < 0.0 ? 0.01 : pivot_tol,
        .zeroTolerance = 0.0,
    };

    double t0 = now_s();
    SparseOpaqueSymbolicFactorization ssym =
        SparseFactor((SparseFactorization_t)fact_type, structure, sfo);
    out->ana_s = now_s() - t0;
    out->sym_status = (int32_t)ssym.status;
    /* "Double the size when used in complex double." */
    out->factor_bytes = 2 * (int64_t)ssym.factorSize_Double;
    out->workspace_bytes = 2 * (int64_t)ssym.workspaceSize_Double;
    if (ssym.status != SparseStatusOK) {
        out->peak_bytes = atomic_load(&peak_bytes);
        return (int)ssym.status;
    }
    if (max_bytes > 0 && out->factor_bytes + out->workspace_bytes > max_bytes) {
        out->peak_bytes = atomic_load(&peak_bytes);
        SparseCleanup(ssym);
        return -100;
    }

    t0 = now_s();
    SparseOpaqueFactorization_Complex_Double f = SparseFactor(ssym, a, nfo);
    out->fac_s = now_s() - t0;
    out->num_status = (int32_t)f.status;
    if (f.status != SparseStatusOK) {
        out->peak_bytes = atomic_load(&peak_bytes);
        SparseCleanup(f);
        SparseCleanup(ssym);
        return (int)f.status;
    }

    memcpy(x, b, (size_t)n * sizeof(shim_cplx));
    DenseVector_Complex_Double xv = {.count = n, .data = (__SPARSE_double_complex *)x};
    t0 = now_s();
    SparseSolve(f, xv); /* in place: x holds b on entry, the solution on exit */
    out->slv_s = now_s() - t0;

    out->peak_bytes = atomic_load(&peak_bytes);
    SparseCleanup(f);
    SparseCleanup(ssym);
    return 0;
}
