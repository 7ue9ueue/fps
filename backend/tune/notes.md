# backend/tune

Run-time tuning of the kernels' choices (variants, cutoffs, layouts), FFTW-style: issue #54. API:
`tune.hpp` (at the top). Tests: `test.cpp`, `test_no_tuner.cpp` (built with `-DFPS_NO_TUNER`).
The first converted choice: small's NTT times behind `small::beats_ntt` ("small.ntt",
`backend/small/tune.hpp`).

## Design

- A choice is a `tune::Table<kDefaults>`: an inline global holding one atomic pointer, constant
  initialized to the compiled defaults (a `std::array` with static storage). `get()` is one acquire
  load (a plain load on x86); the kernel calls it once per call, outside its loops, and indexes the
  values by size class. `set()` publishes a new copy under a mutex and keeps the old copies, so a
  reader holding a reference while another thread sets stays valid (copies are few: one per
  `set()`; they stay reachable, so LeakSanitizer reports nothing).
- Compiled defaults per target: `tools/tune.py <kernel>` builds `backend/<kernel>/tune.cpp` on each
  target's timing machine (lc-bench for the untimed `scalar` and `sse`), runs it on the last core
  under `/tmp/bench.lock`, and writes `tuned.json` (values as printed, machine, CPU, date, commit)
  and `tuned.hpp` (one array per field, one row per target in `tune::Target` order). The kernel's
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
- The frontend's `fps::tune()`, `save_wisdom` and `load_wisdom` (DESIGN.md) wrap `tune::run`,
  `save` and `load`; they need `backend/small/tune.hpp` included (it pulls in the NTT) for small's
  choice to be registered.

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

## Sources

- FFTW's wisdom and its planner modes (measure once, save, load): M. Frigo, S. G. Johnson, The
  Design and Implementation of FFTW3, Proc. IEEE 93(2), 2005. The idea only; no code read.
