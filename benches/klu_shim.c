/* Bench-only FFI shim for SuiteSparse KLU (Davis & Palamadai Natarajan).
 *
 * Compiled at bench runtime against the locally installed SuiteSparse
 * headers/library (Homebrew, LGPL-2.1+) and loaded via libloading, the same
 * runtime-reference pattern as the MKL PARDISO and Apple Accelerate shims:
 * nothing from SuiteSparse is distributed with RSLAB, it is only measured
 * against when present on the machine.
 *
 * Runs the same phases as RSLAB's KLU path on a CSC double matrix:
 * analyze (BTF + per-block AMD, KLU defaults), factor, numeric-only
 * refactor, solve, plus a sweep proxy of `sweep_len` refactor+solve cycles.
 *
 * Build (done automatically by klu_circuit when the library is present):
 *   cc -O2 -std=c11 -I<inc>/suitesparse -L<lib> -lklu -dynamiclib klu_shim.c -o klu_shim.dylib
 */
#include <klu.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

typedef struct {
    double ana_s;
    double fac_s;
    double refac_s;
    double slv_s;
    double sweep_s;   /* sweep_len x (refactor + solve) */
    int64_t lnz;      /* actual nz in L, incl. diagonal */
    int64_t unz;      /* actual nz in U, incl. diagonal */
    int32_t nblocks;  /* BTF diagonal blocks */
    int32_t ok;
} klu_shim_result;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Solve A x = b once with KLU defaults and report per-phase timings.
 * Ap/Ai/Ax: CSC of the n x n matrix; b length n; x out, length n. */
int klu_shim_run(int32_t n, const int32_t *Ap, const int32_t *Ai, const double *Ax,
                 const double *b, double *x, int32_t sweep_len, klu_shim_result *out) {
    memset(out, 0, sizeof(*out));
    klu_common common;
    if (!klu_defaults(&common)) return -1;

    double t0 = now_s();
    klu_symbolic *sym = klu_analyze(n, (int32_t *)Ap, (int32_t *)Ai, &common);
    out->ana_s = now_s() - t0;
    if (!sym) return (int)common.status;

    t0 = now_s();
    klu_numeric *num = klu_factor((int32_t *)Ap, (int32_t *)Ai, (double *)Ax, sym, &common);
    out->fac_s = now_s() - t0;
    if (!num) {
        klu_free_symbolic(&sym, &common);
        return (int)common.status;
    }
    out->lnz = (int64_t)num->lnz;
    out->unz = (int64_t)num->unz;
    out->nblocks = (int32_t)sym->nblocks;

    t0 = now_s();
    if (!klu_refactor((int32_t *)Ap, (int32_t *)Ai, (double *)Ax, sym, num, &common)) {
        klu_free_numeric(&num, &common);
        klu_free_symbolic(&sym, &common);
        return (int)common.status;
    }
    out->refac_s = now_s() - t0;

    memcpy(x, b, (size_t)n * sizeof(double));
    t0 = now_s();
    if (!klu_solve(sym, num, n, 1, x, &common)) {
        klu_free_numeric(&num, &common);
        klu_free_symbolic(&sym, &common);
        return (int)common.status;
    }
    out->slv_s = now_s() - t0;

    /* Sweep proxy: same-pattern value updates, refactor + solve per point. */
    t0 = now_s();
    for (int32_t s = 0; s < sweep_len; s++) {
        if (!klu_refactor((int32_t *)Ap, (int32_t *)Ai, (double *)Ax, sym, num, &common)) break;
        memcpy(x, b, (size_t)n * sizeof(double));
        if (!klu_solve(sym, num, n, 1, x, &common)) break;
    }
    out->sweep_s = now_s() - t0;

    klu_free_numeric(&num, &common);
    klu_free_symbolic(&sym, &common);
    out->ok = 1;
    return 0;
}
