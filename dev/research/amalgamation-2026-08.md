# Relaxed amalgamation: measured and turned off by default (2026-08-29)

Relaxed amalgamation merges fundamental supernodes into wider ones and pads the
merged front with explicit zeros, trading a little fill for higher-rank dense
updates. It has been the shipped default since June 2026
(`SolverSettings::default().relax = Some { max_width: 256, max_extra_rows: 64 }`,
tuned in commit fc0227b: "-15..25% factor time on MoM+FEM"). The convection-
diffusion classes entered the corpus after that calibration.

## Parameter study

14 configurations (`relax` off, plus `nemin` in {8,16,32,64} crossed with
`max_extra_rows` in {16,64,256} at `max_width` 256), 8 matrices spanning all five
classes, 4 rounds with the configurations as the inner loop so drift hits them
equally, minimum per cell. Geomean of config over default:

| config | geomean | worst matrix |
|---|--:|--:|
| relax off | **0.570** | 1.212 |
| n8 x16 | 0.815 | 1.110 |
| n16 x16 | 0.832 | 1.069 |
| n64 x16 | 0.861 | 1.050 |
| n32 x16 | 0.873 | 1.316 |
| n8 x64 | 0.961 | 1.272 |
| n16 x64 | 0.981 | 1.193 |
| n64 x64 | 1.030 | 1.302 |
| n32 x64 | 1.035 | 1.153 |
| n8 x256 | 1.033 | 1.215 |
| n64 x256 | 1.045 | 1.298 |
| n32 x256 | 1.065 | 1.210 |
| n16 x256 | 1.084 | 1.559 |

Every configuration that tightens the padding budget helps, and switching the
relaxation off entirely helps most. The `nemin` axis barely matters next to it.

## Confirmation over the full grid

Relaxed vs off on all 18 matrices of the head-to-head grid, alternating per
repetition, minimum of three, residual checked on the off side:

| matrix | relaxed | off | ratio | fill relaxed | fill off |
|---|--:|--:|--:|--:|--:|
| helmholtz_4096 | 30.2 | 25.6 | 0.847 | 249179 | 249179 |
| convdiff3d_4096 | 68.7 | 39.8 | 0.580 | 498358 | 498358 |
| helmholtz_5832 | 73.0 | 56.1 | 0.769 | 444037 | 444037 |
| convdiff3d_5832 | 121.3 | 88.4 | 0.729 | 888074 | 888074 |
| helmholtz_9261 | 122.7 | 109.5 | 0.892 | 885249 | 885249 |
| convdiff2d_9216 | 39.8 | 10.5 | **0.263** | 316398 | 316398 |
| curlcurl_14739 | 580.4 | 648.8 | 1.118 | 7158905 | 7151408 |
| convdiff2d_13924 | 95.8 | 23.7 | **0.248** | 515456 | 515456 |
| curlcurl_20577 | 1028.5 | 978.1 | 0.951 | 11492400 | 11492416 |
| convdiff2d_21025 | 111.9 | 33.2 | **0.297** | 837902 | 837902 |
| curlcurl_31944 | 2235.3 | 2124.5 | 0.950 | 22136483 | 22136501 |
| convdiff2d_31684 | 238.8 | 92.6 | **0.388** | 1373040 | 1373040 |
| saddle_48387 | 330.0 | 235.1 | 0.712 | 2588942 | 2588611 |
| mom_48035 | 2731.1 | 1889.6 | 0.692 | 34180453 | 32097207 |
| saddle_73008 | 613.4 | 436.9 | 0.712 | 4577068 | 4288942 |
| mom_72690 | 13968.6 | 14574.9 | 1.043 | 127868972 | 125101635 |
| helmholtz_110592 | 2670.7 | 2229.7 | 0.835 | 23985639 | 23985639 |
| convdiff3d_110592 | 4394.6 | 3743.0 | 0.852 | 47971278 | 47971278 |

Geomean 0.654 for off, 16 of 18 matrices faster, worst case curl-curl 14739 at
+12%. Residuals on the off side are 1e-16 to 9e-13, i.e. unchanged accuracy.
The fill is identical on 12 matrices and *lower* on the rest (MoM 34.2M to 32.1M,
saddle 4.58M to 4.29M) - the relaxation was paying fill as well as time.

The effect survives both timing protocols, so it is not an artifact of cold
caches: analyze once and refactor five times gives 0.314 on convdiff2d_21025 and
0.998 on mom_48035, fresh analyze before every factor gives 0.365 and 0.914.

## Why

The padded rows are explicit zeros, but the numeric path does not know that: they
are carried through every `cmod` as real arithmetic and real memory traffic. On
the classes whose fundamental supernodes are already wide (MoM near-field,
curl-curl) the padding is small relative to the front and the wider GEMM pays for
it. On grid problems, where fundamental supernodes are narrow and numerous, the
padding multiplies the work: convdiff2d_21025 goes from 3878 fundamental
supernodes to 277 relaxed ones at identical output fill, and pays 3.4x the time
for it.

Note also how much the absolute numbers move over a long measuring session on
this fanless machine: the same configuration measured 62 ms early in the session
and 130 ms after hours of load. Every comparison here is interleaved within
minutes for that reason.

## Decision

Default is now `relax: None`. The knob stays, and `with_relax(Some(..))` remains
the opt-in for callers whose fronts are dense enough to want the wider GEMMs -
the two matrices that regress here (curl-curl 14739, MoM 72690) are exactly that
shape. Whether a per-matrix criterion is worth building (padding budget relative
to the front's own entries, rather than an absolute row cap) is open; the data
above is the starting point.
