# backend/tune

Run-time tuning of the kernels' choices (variants, cutoffs, layouts), FFTW-style: issue #54. API:
`tune.hpp` (at the top). Tests: `test.cpp`, `test_no_tuner.cpp` (built with `-DFPS_NO_TUNER`).
Converted choices: small's NTT times behind `small::beats_ntt` ("small.ntt",
`backend/small/tune.hpp`, round 1); factorials' variant per size class ("factorials.variant",
`backend/factorials/tune.hpp`, round 2); small's schoolbook variant per shape class
("small.variant", `backend/small/tune.hpp`, round 3).

## Design

- A choice is a `tune::Table<kDefaults>`: an inline global holding one atomic pointer, constant
  initialized to the compiled defaults (a `std::array` with static storage). `get()` is one acquire
  load (a plain load on x86); the kernel calls it once per call, outside its loops, and indexes the
  values by size class. `set()` publishes a new copy under a mutex and keeps the old copies, so a
  reader holding a reference while another thread sets stays valid (copies are few: one per
  `set()`; they stay reachable, so LeakSanitizer reports nothing).
- Compiled defaults per target: `tools/tune.py <kernel>` builds `backend/<kernel>/tune.cpp` on each
  target's timing machine (lc-bench for the untimed `scalar` and `sse`), runs a cp copy of it on
  the last core under `/tmp/bench.lock` (Log: Zen 5 and page frames), and writes `tuned.json`
  (values as printed, machine, CPU, date, commit) and `tuned.hpp` (one array per field, one row
  per target in `tune::Target` order). The kernel's
  `select.hpp` takes row `int(tune::kTarget)`. `tune::kTarget` comes from the same macros as
  `select.hpp` used (`__znver5__`, `__AVX512F__ && __AVX512BW__`, `__AVX2__`, `__ARM_NEON &&
  __aarch64__`, `__SSE4_1__`); the Mac's builds get the `arm` row. `tune.py --check` (in
  `kernel.py check`): `tuned.hpp` is the rendering of `tuned.json`, every target has a row with the
  same fields, and `tune::Target` lists `TARGETS` in order. It cannot tell whether the kernel got
  faster since the measurement: rerun `tune.py` then (the PR's gate times it).
- Run-time measurement: `backend/<kernel>/tune.hpp` measures a choice on this machine with
  `tune::time` (the candidates timed in turn, median of a few samples) and registers it with
  `tune::add`. `tune::run()` measures every registered choice once per process (`std::call_once`).
  The offline `tune.cpp` calls the same measurement with a larger budget.
- Margin: a measured value within the choice's margin of its compiled default keeps the default,
  so tuning on our own machines mostly reproduces the compiled table, and noise cannot move it.
- Wisdom: `tune::save(path)` writes `fps wisdom 1`, `cpu <name>`, `target <name>`, then one line
  per registered choice, `<name> <values>` (shortest round-trip text). `tune::load(path)` takes it
  only if all three header lines match this process; every line is parsed before any is set;
  names it does not know are skipped. CPU name: x86 the cpuid brand string (`AMD EPYC 7B13`,
  `INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz`, `AMD EPYC 9B45`); Arm Linux the first core's
  implementer, variant, part and revision (`implementer 0x41 variant 0x0 part 0xd4f revision 1`
  on Axion); macOS `machdep.cpu.brand_string`.
- `-DFPS_NO_TUNER` (contest bundles): `Table::get()` is a static constexpr function returning the
  defaults, so lookups fold to constants; no `set()`, registry, `run()` or wisdom, and kernels'
  registrations compile out. The measurement code stays (inline, unused).
- When tuning runs: only when called (`tune::run()`, or `load`). Not lazily on first use of a size
  class: one class of small's choice costs 3.3-6.7 ms up to 2^16 and 8-30 ms at 2^18
  (Measurements), against the issue's "well under a millisecond", and a lazy first call would make
  a product's path, and the gate's benches, depend on noise. Benches and tests read the compiled
  defaults.
- The frontend's `fps::tune_all()`, `save_wisdom` and `load_wisdom` (`include/fps/tune.hpp`, issue
  #66) wrap `tune::run`, `save` and `load`. That header includes every kernel's `tune.hpp` the
  frontend uses, so those choices register; a new kernel tune header is added there. (Not
  `fps::tune()`: a function cannot share this namespace's name.)
- A variant choice (factorials, round 2): the target's variants are a compile-time list
  (`Candidates<Variant<S, K, step>...>`, all instantiated); the table holds one byte per size
  class, the index of the variant (0 the scalar loop); the kernel runs it through a chain of
  compares on the index (`run_variant`), no function pointers. Size classes are quarter octaves
  (e, e + w], w = 2^(k-2): they end at round sizes. `tuned.hpp` also records a hash of the
  target's list, and `select.hpp` static_asserts it, so a table measured for another list does
  not compile; the wisdom line starts with the same hash, and a line for another list, or with a
  variant that cannot run at its class's least n, is malformed (load sets nothing).
- Measuring a variant choice (`factorials::measure`): dense, then sparse. Dense: each class up
  to 2^dense_lg at `points` sizes (just past a round size, and the class end), every variant
  timed in turn by `tune::time_each` (calls per sample from one timed call; a call that alone
  fills a sample is the first sample), the least geometric mean of time / fastest wins unless
  start's is within the margin. Sparse: one size per octave, 2^k, up to 2^sparse_lg, decides the
  octave's classes with a wider margin (one size cannot place a change inside an octave);
  2^sparse_lg decides every larger class. A variant more than 2x the fastest and narrower than
  it is not timed at larger sizes (the loop from about 64 on), except where start uses it.
  tune.cpp runs the dense part to 2^24 at 4 points, 7 samples of 1 ms, against the previous
  compiled table with a 2% margin, so a rerun only moves classes that changed by more than that.
- A two-dimensional variant choice (small, round 3): the class is the octave of m (11) times 4
  ratios n / m (below 2, 4, 8, the rest), 44 classes; the target's 3-7 candidates
  (`Variant<S, R, U, Solo>`) are a compile-time list as in factorials, with the same signature
  check. The dispatch (`run_candidate`) is a fold over the list inlined into the caller, which GCC
  turns into a jump table; each candidate's `run` there is a single call (`panel`, or `panels`
  for b over 1024 words), so the caller grows by a few instructions per candidate. A lambda with
  captures in that place (factorials' `run_variant` form) cost 6 ns per call on Zen 3: its closure
  went through the stack and the call was not inlined (Log). `measure_variants` times every
  candidate at a few shapes per class (m at the octave's end and 3/4 of it, n / m 1 / 3 / 6 / 16
  by ratio class; tune.cpp: also m just past the octave's start, two or three ratios per class)
  and keeps the current variant unless another's geometric mean of time / fastest is better by
  the margin. Run time measures classes with m <= 256 only (beats_ntt sends larger m to the NTT).
  "small.variant" is registered before "small.ntt", which times multiply().

## Measurements

GCC 15.2, `tools/kernel.py` flags, 2026-10-11, this branch on main df74bf7. Machines: `lc-bench`
(Zen 3, avx2), `lc-intel` (Emerald Rapids, avx512), `lc-zen5` (Zen 5), `lc-arm` (Neoverse V2).
Programs: `explore.cpp` in `~/Documents/hpc_library-explore/tune/r1/` (`run.sh` builds and runs it
under the bench lock on the last core).

### Lookup against main's compile-time selection

`beats_ntt` on 4096 random shapes (n, m log-uniform in [1, 2^20]), ns per decision, median of 11
samples of 2 ms, 3 rounds (the middle one shown): main's constexpr cost model (8d79607's code, the
same numbers) against the tuned table (one acquire load), called through a noinline function and
inline in the loop. Same decisions on all 4096 shapes.

| Machine | call: main | tuned | inline: main | tuned | `FPS_NO_TUNER` call: main | tuned |
|---|---:|---:|---:|---:|---:|---:|
| Zen 3 | 4.92 | 3.76 | 4.91 | 2.64 | 4.35 | 3.99 |
| Emerald Rapids | 3.06 | 2.89 | 3.25 | 1.82 | 2.83 | 2.65 |
| Zen 5 | 2.47 | 2.57 | 2.48 | 1.43 | 2.44 | 2.58 |
| Neoverse V2 | 2.67 | 2.20 | 2.64 | 1.90 | 2.63 | 2.33 |

The table costs nothing measurable: the tuned lookup is 0.54-1.04x of main's (the spread is code
generation of the same formula, e.g. Zen 5's call 1.04x and inline 0.58x). A decision costs
1.4-3.8 ns against 34-50 ns for the smallest product (8x8, `backend/small/notes.md`).

### Noise, sample length and the margin

`explore.cpp warm`: the probe's NTT time / compiled default per lg as one run of the run-time
measurement sees it, 10 runs, against samples of 1 ms (`tools/tune.py`'s), in the same binary.

- 50 us samples timed in turn (the first version): Zen 3 lg 7 0.92 (1 ms: 0.99), lg 6 1.03 (0.99);
  Emerald Rapids lg 6-9 1.04-1.15 (0.96-1.00), single runs up to 1.16. Switching between the two
  candidates' code costs a share of a short sample.
- An untimed call before each sample of several calls (now in `tune::time`): Zen 3 within 1% of
  the 1 ms samples at every lg 6-16. Emerald Rapids still 1.06-1.16 at lg 6-9 with 50 us samples;
  with 200 us samples 0.945-1.002 against 0.948-0.995 (1 ms), single runs 0.92-1.12.
- So `small::kRunTime`: samples of 200 us, 5 per candidate, margin 5%.
- Between two binaries (tune.cpp against explore.cpp) the 1 ms ratio itself moved up to 5% (Zen 3
  lg 8 1.047, lg 12 1.057): code layout. tools/tune.py runs reproduce: run 2's table against
  run 1's on run 2's sweep, the same shapes go the slower way on every target (Log).

### Cost of tuning

`tune::run()` for small's choice (`measure_ntt` at `kRunTime`: lg 6-16 measured, 17-25 scaled),
5 runs: Zen 3 46.9-47.0 ms, Emerald Rapids 45.0-49.0, Zen 5 47.0-48.4, Neoverse V2 50.0. Above
the issue's "a few ms per choice" (this choice spans 11 measured lengths), well under a second.
With 50 us samples it took 12.4-20.2 ms, but was biased (above). Values changed per run on the
home machines (margin 5%): Zen 3 1-2 of 20 (lg 7: 0.93 in this binary, a layout difference
against tune.cpp's), Emerald Rapids 0-2, Zen 5 0-1, Neoverse V2 0.

One probe (one length class), ms, median of 5: what lazy tuning on first use would add to the first
product of that class.

| lg | Zen 3 | Emerald Rapids | Zen 5 | Neoverse V2 |
|---:|---:|---:|---:|---:|
| 6 | 3.4 | 4.6 | 4.4 | 4.4 |
| 10 | 4.1 | 4.5 | 4.8 | 4.4 |
| 14 | 6.0 | 3.7 | 3.3 | 4.0 |
| 16 | 3.4 | 3.7 | 4.7 | 6.7 |
| 18 | 15.0 | 16.8 | 8.1 | 29.9 |
| 20 | 64.7 | 76.1 | 35.2 | 135.4 |

A product of class lg 6 takes about 0.1 us: the probe costs 3e4 of them. Lazy tuning is out.

`tools/tune.py small`: 16-26 s per timed target, 77 s for scalar (lc-bench).

### Run-time tuning against the compiled defaults

`explore.cpp validate`: shapes the tables were not fitted on, n + m - 1 = 5 2^(lg-3) and
7 2^(lg-3) for lg 6-20, m = 1-512 within 0.25x-4x of the crossing by the defaults (264-279
shapes), multiply() and the Convolution each timed (median of 5, samples of 1 ms). Shapes sent the
slower way, the worst loss (chosen / faster), and the geomean loss, for the compiled defaults and
for 3 runs of `measure_ntt` at `kRunTime`.

| Build on machine | compiled defaults | tuned at run time (3 runs) |
|---|---|---|
| avx2 on Zen 3 (home) | 1 of 279, 1.001x | 1 of 279, 1.001x |
| avx512 on Emerald Rapids (home) | 2 of 264, 1.041x | 2-4 of 264, 1.041-1.057x |
| zen5 on Zen 5 (home) | 0 of 268 | 0 of 268 |
| arm on Neoverse V2 (home) | 1 of 270, 1.000x | 1 of 270, 1.000x |
| avx2 on Emerald Rapids (Zen 3's table) | 10 of 279, 1.228x, geomean 1.0037 | 0-1 of 279, 1.000-1.025x |
| avx2 on Zen 5 (Zen 3's table) | 2 of 279, 1.077x | 1 of 279, 1.019x |
| avx512 on Zen 5 (Emerald Rapids' table) | 7 of 264, 1.409x, geomean 1.0033 | 0-2 of 264, 1.000-1.034x |

On the home machines tuning keeps what the compiled defaults give; on another CPU it removes
nearly every loss (avx512 on Zen 5: the NTT times at lg 7-9 went up 36-54%).

### factorials.variant (round 2)

GCC 15.2, `tools/kernel.py` flags, 2026-10-11, this branch on main 47736ca. Programs in
`~/Documents/hpc_library-explore/tune/r2/`: `fine.cpp` (every variant at 4 sizes per quarter
octave, 16 to 2^22, 7 samples of 200 us), `runtime.cpp` (validation and run-time cost),
`lookup.cpp`, `analyze.py`, `summarize.py`; outputs in `out/`.

Lookup, ns per decision (`lookup.cpp`: 4096 sizes log-uniform in [1, 2^20], median of 11
samples of 2 ms, 3 rounds): main's compile-time tier compare against `variants.get()[size_class(n)]`.

| Machine | call: main | tuned | inline: main | tuned | `FPS_NO_TUNER` call: tuned |
|---|---:|---:|---:|---:|---:|
| Zen 3 | 1.45 | 2.11-2.13 | 1.14 | 1.04 | 2.23 |
| Emerald Rapids | 1.44 | 1.45-1.54 | 0.42 | 0.76 | 1.17-1.25 |
| Zen 5 | 1.06 | 1.17-1.18 | 0.23 | 1.32 | 1.16-1.17 |
| Neoverse V2 | 0.77-0.79 | 1.44-1.58 | 0.90-0.94 | 0.83 | 1.10-1.14 |

Up to 0.8 ns more per call than main's compares (the class: a leading-zero count and two
shifts), against 140-190 ns for the smallest tiered call (n = 64). In the bench binary,
`tables/6` against the same variant called directly (`tables@<variant>/6`): Zen 3 165 / 164 ns,
Emerald Rapids 166 / 164, Zen 5 140 / 140, Neoverse V2 187 / 187: nothing beyond noise.

Cost of `tune::run()` for this choice (`kRunTime` = dense to 2^17 at 2 points, sparse to 2^20,
3 samples of 10 us, margins 3% and 5%), 5 runs: Zen 3 57-61 ms, Emerald Rapids 49-52, Zen 5
37-39, Neoverse V2 88-90; dense only to 2^16: 47, 43-46, 33, 73. Other builds: avx2 on Emerald
Rapids 45-47, avx2 on Zen 5 37-38, avx512 on Zen 5 38-40; the arm build (clang) on an Apple M4
95-112 (a laptop; indicative). Above "a few ms per choice": the choice has 88 classes and
times 5-9 variants at each; well under a second. Samples of 20 us cost 11-25 ms more and
decided within 0.1% of 10 us; one sample cost half and lost 1.2-2x as much on the home
machines (Zen 5, n <= 2^16: 1.0043-1.0075 against 1.0032-1.0036). The sparse part costs 13-43
ms (Zen 5 to Neoverse V2); without it a table from another CPU keeps its large-n losses
(below).

`tools/tune.py factorials`: 18-33 s per timed target, 56-62 s for scalar on `lc-bench`. A second
run from the first run's table (2% margin) changed 0-5 of 88 classes per target (scalar 0,
sse 5, avx2 4, avx512 2, zen5 4, arm 1); on the second run's points the first table's geomean
loss was 1.0009-1.0080 against 1.0007-1.0039 for the second's.

Decisions (`runtime.cpp`): 240 sizes log-uniform in (16, 2^22], not on the measured grid, every
variant timed there (median of 7, samples of 200 us, fresh std::vectors per size). Per table:
sizes more than 2% slower than the fastest variant, the worst ratio, the geomean of chosen /
fastest; n <= 2^16 (159 sizes) | n > 2^16 (81). Run-time tuning: 5 runs at `kRunTime`.

| Build on machine | compiled defaults | tuned at run time |
|---|---|---|
| avx2 on Zen 3 (home) | 18, 1.081, 1.0061 \| 0, 1.012, 1.0002 | 10-11, 1.081, 1.0040-1.0044 \| 0-1, 1.0002-1.0013 |
| avx512 on Emerald Rapids (home) | 8, 1.050, 1.0026 \| 33, 1.200, 1.0475 | 10-16, 1.050-1.102, 1.0028-1.0047 \| 32-36, 1.049-1.053 |
| zen5 on Zen 5 (home) | 19, 1.155, 1.0076 \| 4, 1.084, 1.0025 | 8, 1.054, 1.0021 \| 4, 1.0026 |
| arm on Neoverse V2 (home) | 2, 1.026, 1.0011 \| 0, 1.019, 1.0003 | the same (no class changed) |
| avx2 on Emerald Rapids (Zen 3's table) | 126, 1.396, 1.1131 \| 77, 1.128, 1.0554 | 7-9, 1.057-1.100, 1.0028-1.0039 \| 0-53, 1.0002-1.0361 |
| avx2 on Zen 5 (Zen 3's table) | 67, 1.198, 1.0402 \| 0 | 5-17, 1.0018-1.0065 \| 0 |
| avx512 on Zen 5 (Emerald Rapids' table) | 82, 1.560, 1.0774 \| 76, 1.353, 1.1967 | 8-13, 1.117, 1.0038-1.0054 \| 24-31, 1.22-1.32, 1.035-1.050 |
| arm on Apple M4 (V2's table; clang, indicative) | 146, 1.606, 1.2244 \| 81, 1.506, 1.4344 | 4-7, 1.0017-1.0026 \| 0, 1.0005 |

On the home machines tuning keeps nearly every class (0-7 of 88 change) and the compiled table
is already within 0.3-0.8% of the fastest variant at sizes it was not measured at; on another
CPU tuning removes most of the loss. What stays:
- Emerald Rapids from 2^19: at sizes off the measured grid zmm K = 2 is often 1.15x faster
  than ymm K = 2, but at the round sizes the table measures (and the gate times) 1.15-1.3x
  slower: zmm's 32 lanes are fast or slow by n (factorials notes, Open). A size class cannot
  tell them apart.
- avx512 on Zen 5 at 2^16-2^18: Zen 5's L2 edge; times there move 10-30% between runs.
- avx2 on Emerald Rapids from 2^17: ymm Montgomery is 3-6% faster than the Shoup that Zen 3's
  table has, near the 5% sparse margin: 2 of 5 runs moved it.
With the sparse margin at 3% (instead of 5%) every run moved it (1.0002), but one of 3 runs on
Emerald Rapids itself then lost 2% at large n (1.0754 against 1.0524).

### small.variant (round 3)

GCC 15.2, `tools/kernel.py` flags, 2026-10-11, this branch on main e452c18 (#37 round 2), then
rebased onto e66ce2f (#5 round 2, #70 round 1) for the third `tune.py` run and the final benches.
Programs in `~/Documents/hpc_library-explore/tune/r3/`: `sweep.cpp` (every candidate on a grid),
`analyze*.py`, `validate.cpp`, `loop.cpp` and `loops.sh` (one call in a loop, paired builds),
`fg.cpp` (f * g in a loop), `fe32.cpp` (f * g against its parts), `lookup.cpp`; outputs in `out/`.

Exploration (`sweep.cpp`, `analyze*.py`): 531 shapes per target (m 1-1024, n / m 1-2048, at most
2^22 terms), 7-12 candidates each (R 2-8, U 1-4, Solo, the narrower structs), 7 samples of 300 us.
Geometric mean of chosen / fastest candidate: one fixed variant (main's) Zen 3 1.023, Emerald
Rapids 1.032, Zen 5 1.021, Neoverse V2 1.031, sse 1.046, scalar 1.050. A table per class, in
sample: by the octave of m alone (11 classes) 1.012-1.017 on the timed four; octave times 4
ratios (44) 1.008-1.011; octave times the octave of n (97) 1.005-1.007. 44 classes, two-fold
cross-validated (decided on half the shapes, scored on the other half): 1.0095-1.015. Greedy
pruning of the candidates (drop the one that costs least): 5-7 keep the 44-class table within
0.0003 of all of them; the lists in `select.hpp` are those.

Lookup and dispatch, in the call (`loops.sh`: `multiply` in a loop against the same variant
called directly, median of 7, 3 interleaved runs):
- A lambda with captures doing the dispatch (factorials' form): 8x8 on Zen 3 40.9 ns against
  34.2 called directly, every shape +6 ns: the closure went through the stack and the call was
  not inlined. As an inlined fold (`run_candidate`): +1.5-2.2 ns on Zen 3, +1.1-1.5 ns on Zen 5
  at 16x16-48x48. A jump table or a chain of compares (`-fno-jump-tables`), and a bounds hint,
  made no difference; `run` out of line cost 1-2 ns more.
- With `run` one call of `panel` (the panel loop out of line, `panels`), against main's direct
  call: Zen 3 8x8 29.0 against 34.9 (the table's variant), 16x16 53.9 / 56.0, 32x32 136.8 /
  136.6; Zen 5 8x8 19.0 / 20.2, 16x16 30.2 / 30.5, 32x32 54.5 / 54.4, 48x48 90.5 / 89.6.
- `lookup.cpp` (the lookup alone, `variants.get()[shape_class(n, m)]`, 4096 random shapes,
  median of 11 samples of 2 ms, 3 runs): inline / through a noinline call, ns per decision: Zen 3
  1.59 / 2.03, Emerald Rapids 1.06 / 1.16, Zen 5 1.15 / 1.22, Neoverse V2 1.33-1.35 / 1.51;
  `FPS_NO_TUNER` (a constant table) Zen 3 1.49 / 1.74, Emerald Rapids 1.05 / 1.17, Zen 5 1.15 /
  1.22, Neoverse V2 1.33 / 1.68. The class costs a leading-zero count and three compares; the
  acquire load costs nothing measurable.

Bench against main e66ce2f (`kernel.py bench`, 11 rounds, the gate's 22 shapes): Zen 3 0.9933,
Emerald Rapids 0.9968, Zen 5 0.9954, Neoverse V2 0.9754; frontend `multiply` cases 1.0044, 1.0013,
0.9932, 0.9937 (`backend/small/notes.md`, Log).

Cost of tuning: `tune::run()` for "small.variant" (`kVariantRunTime`: classes with m <= 256, 2
points, 3 samples of 10 us, margin 3%) 9.1-9.4 ms on Zen 5, 11.6-11.9 Emerald Rapids, 13.0-13.2
Zen 3, 12.7-13.0 Neoverse V2; both of small's choices together 56-68 ms. Measuring 3 m per class
on the Apple M4 cost 38-49 ms against 11-22 and decided no better. `tools/tune.py small`: 22-32 s
per timed target (round 1: 16-26), 82 s for scalar on `lc-bench`. A second `tune.py` run, from the
first run's table with a 1% margin, changed 3-4 of 44 classes per target (Emerald Rapids 21: the
first run had a 2% margin), and the first table scored 1.0034-1.0174 on the second run's points
against 1.0022-1.0088 for the second. A third run (on main e66ce2f) changed 0-6 classes; on its
points the second table scored 1.0024-1.0130, the third 1.0024-1.0093.

Decisions (`validate.cpp`, the final table, on e66ce2f): 240 shapes, m log-uniform in [1, 1024],
n / m in [1, 2^12], at most 2^22 terms, off the measured points; every candidate timed there
(median of 7, samples of 200 us). Geometric mean of chosen / fastest candidate, for the shapes
beats_ntt gives to small (186-197 of 240) \| the rest; run-time tuning: 5 runs at
`kVariantRunTime` (classes changed in brackets).

| Build on machine | main's variant | compiled defaults | tuned at run time |
|---|---|---|---|
| avx2 on Zen 3 (home) | 1.0471 \| 1.0181 | 1.0109 \| 1.0044 | 1.0194-1.0216 \| 1.0044 (1-2) |
| avx512 on Emerald Rapids (home) | 1.0425 \| 1.0179 | 1.0103 \| 1.0076 | 1.0099-1.0130 \| 1.0029-1.0051 (2-4) |
| zen5 on Zen 5 (home) | 1.0279 \| 1.0040 | 1.0161 \| 1.0040 | 1.0170-1.0175 \| 1.0040 (1-2) |
| arm on Neoverse V2 (home) | 1.0299 \| 1.0394 | 1.0204 \| 1.0017 | 1.0206-1.0224 \| 1.0012-1.0017 (2-5) |
| avx2 on Emerald Rapids (Zen 3's table) | 1.0402 \| 1.0071 | 1.0390 \| 1.0039 | 1.0244-1.0265 \| 1.0035 (13-17) |
| avx2 on Zen 5 (Zen 3's table) | 1.0187 \| 1.0031 | 1.0125 \| 1.0024 | 1.0152-1.0164 \| 1.0024 (8-9) |
| avx512 on Zen 5 (Emerald Rapids' table) | 1.0302 \| 1.0073 | 1.0167 \| 1.0070 | 1.0139-1.0158 \| 1.0070-1.0084 (6-10) |
| arm on Apple M4 (V2's table; clang, laptop) | 1.0394 \| 1.0302 | 1.1030 \| 1.2038 | 1.0317-1.0431 \| 1.0292-1.0358 (23-25) |

- On every home machine the compiled table loses less than main's single variant (small: 1.0-2.0%
  against 2.8-4.7%). Run-time tuning there changes 1-5 classes and lands within 0.1-1% of the
  compiled table (2 points and 3 samples of 10 us per class against tune.py's 5-8 points and 7
  samples of 1 ms).
- On another CPU: Zen 3's table on Emerald Rapids is no better than main's variant, and tuning
  removes a third of the loss; on Zen 5 both tables are already better than main's. V2's table on
  the M4 (clang) is worse than main's variant (U = 1 variants run 1.3-1.4x slower there); tuning
  brings it back to main's level. The round-2 table (V2's, before the rebase) gave the same
  picture on the M4: 1.084 \| 1.215 compiled, 1.021-1.029 \| 1.035-1.039 tuned.
- Worst single shapes: m <= 5 (a few vectors, 20-30 ns: 109x1 1.34 on Zen 5, 100x3 1.15 on
  Emerald Rapids).

The frontend bench's NTT-only shapes move with code layout: main against main with one unused
function added to `frontend/bench.cpp` (no kernel changed), 11 rounds: Zen 3 `multiply/6` 1.075,
`/7` 0.973, geomean 1.0022; Zen 5 `/9` 1.031, `/10` 1.023; Emerald Rapids within 0.6%.

## Log

2026-10-11, claude (issue #54, round 1): the mechanism, and small's NTT times converted.
- Representation for small. The decision is two-dimensional (n, m); the round-1 cost model
  (multiply() about term n m + word (n + m) + call + square min^2 against the NTT's time per
  length) missed 2-4 of 186 swept shapes by at most 4-12%. A table of booleans or of cutoffs on
  min(n, m) per (length, size class) loses up to the width of a class at its boundary (a class
  doubles n m, or n + m), so the model stays and its numbers become the tuned values: the shape
  coefficients are fitted offline by `tune.cpp`, and the NTT time per length is the tuned choice,
  measured where the two cross so that the model only interpolates near the crossing.
- Offline (`tools/tune.py small`, `tune.cpp`): 540 shapes, lg 6-17, n + m - 1 at 2^(lg-1) + 1,
  3 2^(lg-2) and 2^lg, m = 1-1024; fit on those within 0.4x-2.5x of the NTT; then the NTT time at
  every lg 6-25 at the crossing (median of 7, samples of 1 ms). 16-26 s per target (scalar 77 s on
  lc-bench). On the swept shapes, round 1's hand fit against the new table: Zen 3 8 shapes the
  slower way (worst 1.12x) against 5 (1.08x); Emerald Rapids 6 (1.07x) against 1 (1.008x); Zen 5
  3 (1.02x) against 2 (1.02x); Neoverse V2 6 (1.13x) against 1 (1.03x); sse 1 against 0; scalar 1
  (1.10x) against 2 (1.01x). In sample: the new table is fitted to those shapes; the out-of-sample
  check is the validation above.
- The run-time measurement first used samples of 50 us without the untimed call: on Zen 3 and
  Emerald Rapids it moved small-lg values by 5-18% on the home machines and made the decisions
  slightly worse than the defaults (Zen 3 worst 1.003x -> 1.09x, Emerald Rapids 1.004x -> 1.11x
  on the validation shapes). Fixed as above.

2026-10-11, claude (issue #54, round 2): factorials' variant per size class converted.
- Representation. Round 3's tiers (min_size per tier, a variant each) became a byte per size
  class. Exploration (`fine.cpp`, 284 sizes per machine, every variant): main's tiers lose
  (geomean of chosen / fastest) 1.0063 on Zen 3, 1.0132 Emerald Rapids, 1.0942 Zen 5, 1.0033
  Neoverse V2, 1.0148 sse, 1.2976 scalar; a per-class table decided on 4 sizes per class
  1.0010, 1.0012, 1.0004-1.0021, 1.0007, 1.0016, 1.0007 (Zen 5's 1.09: zmm K = 1 Shoup, never
  tried before, wins at most sizes).
- Classes [e, e + w) were tried first: one variant per class cannot serve n = 64 (8 lanes of
  8: no slack) and n = 65 (the same lanes cover 128); deciding on the class's lower edge lost
  1.0064-1.0222 at the other sizes, on all 4 sizes it lost at the edges (Zen 3 n = 64: 1.09,
  a gated size). Classes (e, e + w] put the round size at the class end, where every variant
  covers n without slack: on all 4 sizes, 0-1 of 72 round sizes more than 2% slower.
- Run-time measurement, first version: dense classes to 2^16, start's variant kept above. On
  the Apple M4 (arm build, V2's table) it left n > 2^16 at 1.46x; with avx2 on Emerald Rapids
  at 1.05x. Measuring dense to 2^20 cost 145-390 ms. Added the sparse part (one size per
  octave to 2^20). With the dense margin (3%) for it, Zen 5's own table got worse at
  (2^16, 2^17] (1.0102 -> 1.0218): 2^17 alone picked zmm K = 1 for an octave where zmm K = 2
  wins below 1.4e5. Fixed by dense to 2^17 and a separate sparse margin (10% kept the home
  tables but left avx2 on Emerald Rapids at 1.05x; 5% chosen).
- A bug the tests found (lc-amd): with a huge margin measure() still changed a class, because
  start's variant had been dropped as slow at a smaller class (noise at 1 sample of 2 us).
  start's variants are now always timed where start uses them. And the test of the sparse part
  first wanted the first of the fastest variants: on `lc-amd` the clock tied two (sse, one call
  of 1.2 us each) and measure() kept start's, which is one of them. It now wants a fastest.
- Not kept: 4 points per dense class at run time (40-75% more time, decisions no better;
  Zen 5 n <= 2^16 1.0048-1.0067 against 1.0032-1.0036); dense only to 2^14 (Zen 3 n <= 2^16
  1.0065 against 1.0032-1.0044). Kept: dropping a narrow variant once it is 2x the fastest
  (the loop from about 64, Sse.1 from about 1024).

2026-10-11, claude (issue #54, round 3): small's schoolbook variant per shape class converted
("small.variant"). Measurements above (small.variant); the kernel side in `backend/small/notes.md`.
- Representation. Small's R, U, Solo and struct were one variant per target. The time depends on
  m (steps per block, the edges' length) and on n / m (interior blocks against edges), so the
  class is two-dimensional: 11 octaves of m times 4 ratios. Fewer ratio classes (2-3) lost
  0.04-0.2% more on the sweep; the octave of n instead of the ratio (97 classes) gained 0.3% for
  twice the classes. Exact ratio compares (n >= 2m, 4m, 8m) against bit_width(n) -
  bit_width(m): 0-0.0006 better, the same cost. Solo, a threshold on m before, is now a
  candidate's flag: the octave of m already decides it.
- The candidate lists come from the sweep (greedy pruning), 3-7 per target; index 0 is main's
  variant. avx512 and zen5 get different lists (`__znver5__`, as `tune::kTarget`).
- Measurement: run time times 2 points per class (m at the octave's end and 3/4 of it) at one
  ratio per class (two for the long class offline); classes with m > 256 keep the compiled table
  (beats_ntt sends them to the NTT). tune.cpp: 3 m per class and 2-3 ratios, 7 samples of 1 ms.
- Its first run kept main's variant within 2%. On Emerald Rapids and Zen 5 that kept R6 U1 in
  classes where R6 U2 or R8 U2 is 1-1.5% faster (16x16 to 256x256), and the bench there was 1.0071
  and 1.0088 against main. With a 1% margin (the second run, from the first run's table) Emerald
  Rapids reached 1.0008 (and 0.9957 with the change below); Zen 5 kept R6 U1 at those classes.
- Dispatch, in the order tried: a lambda with captures, as factorials does: +6 ns per call on Zen
  3; an inlined fold: +1-2 ns; then each candidate's `run` reduced to one call of `panel` (the
  multi-panel loop and its memset out of line in `panels`): the bench went from 1.0084 to 0.9991
  on Zen 5, from 0.9935 to 0.9896 on Zen 3, from 1.0008 to 0.9957 on Emerald Rapids. Not kept:
  `run` out of line (+1-2 ns), `multiply` out of line (+1-1.5 ns), the `panel` body inlined into
  `run` (Zen 3 8x8 37.0 against 30.2), `-fno-jump-tables` and an unreachable-bounds hint (no
  change).
- A mistake to avoid: `origin/main` moved while this round ran (another agent's fetch: #72, #73),
  so `vm.py --base origin/main` compared later runs against a main with a faster NTT; the
  frontend's NTT-only shapes then read 1.04-1.09 on Emerald Rapids (main's `multiply/21` 10.36 ->
  9.46 ms). Pin the base by its hash. After a rebase onto e66ce2f and a third `tune.py` run, the
  frontend is 0.993-1.004 (above). Zen 3's `multiply/6`-`/7` (1.076, 1.060) are layout: main
  against main with one unused function reads 1.075 and 0.973 there.

2026-10-11, claude (issue #70 round 2, factorials): Zen 5 and page frames. Details and numbers in
`backend/factorials/notes.md` (Log, issue #70 round 2).
- Round 1's factorials tune binary ran zmm K = 1 Shoup 1.2-2.3x slow at n <= 2048 on Zen 5. Rerun
  from its own file it still did; a cp copy of the same file did not; evicted from the page cache
  and read back, another variant (zmm K = 2 Montgomery) ran 1.4-2.3x slow instead. The linker
  leaves a binary on scattered 4 KiB page-cache frames; cp writes 64 KiB folios (contiguous
  frames). Of 6 copies written 4 KiB at a time, 2 had a variant 17-20% slow at some sizes; of 6
  cp copies, none. On Zen 5 the speed of the same code depends on its physical frames (the
  mechanism is a guess), fixed for as long as the file stays cached.
- `tools/tune.py --measure` now runs a cp copy of the tune binary. Zen 5's factorials table from
  two such runs differed in 1 of 88 classes.
- Not changed: `tools/kernel.py` (benches, the gate) still runs freshly linked binaries, so Zen 5
  bench numbers keep a bias per binary (e.g. PR #73's frontend `factorials/10` at 1.294).

## Sources

- FFTW's wisdom and its planner modes (measure once, save, load): M. Frigo, S. G. Johnson, The
  Design and Implementation of FFTW3, Proc. IEEE 93(2), 2005. The idea only; no code read.
