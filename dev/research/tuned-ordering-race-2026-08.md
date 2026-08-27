# tuned ordering: exact two-stage race (2026-08-27)

Trigger: the Accelerate head-to-head exposed the unsym path losing 3-8x
on convection-diffusion. Diagnosis: `SolverSettings::default().ordering`
was pinned to `Amd` (predating the issue-50/F11 fixes), so the shipped
`tuned()` never consulted the shape heuristics at all; the flops-gated
ND bakeoff only rescued the very large classes. On convdiff2d, Amf beats
Amd by 20-45% factor time; a blanket `Auto` default fixes that but
regresses convdiff3d by 48% (Amf loses to Amd there) - per-shape
heuristics keep guessing wrong on this family.

Fix: `tuned()` now resolves the ordering with `AutoRace`, upgraded to an
exact two-stage race: {Amd, Amf, Rcm} prefixes run concurrently and the
smallest EXACT factor nnz wins (candidate order breaks ties); the
expensive MetisND candidate joins only when the champion's exact
predicted flops clear an amortization floor (5e9, the same work-floor
principle as the KLU parallel gates) and n > 10_000. The Amd default
pin, the ND bakeoff and its three gates are gone - one measurement
instead of stacked heuristics. The ND seed ensemble applies inside the
MetisND candidate as before.

Measured vs main (tuned, ana / factor):

| class | fill | factor | ana |
|---|---|---|---|
| convdiff2d 9k | -15% | -40% | = |
| convdiff2d 32k | -10% | -14% | +4 ms |
| convdiff3d 49k | = | = | = |
| helmholtz 32/44^3, curl, saddle | = | noise | = or better |
| bem 8k | -2% | -5% | +112 ms (Amf prefix on a dense-ish graph; factor is 2.3 s there) |

Fill can never regress: Amd is always a candidate and the pick is by
exact fill.
