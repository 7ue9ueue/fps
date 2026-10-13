# fps: backend

Status: draft, 2026-10-10. The user API is in `DESIGN.md`.

The backend is the kernels under the API: NTTs, pointwise products, small products, CRT,
element-wise loops. Each kernel has several variants; agents explore them in rounds and keep each
variant where it wins. Header-only: the ISA is chosen at compile time; variants, cutoffs and layouts
that are tuned choices (`backend/tune`) have compiled defaults per target, measured on the target's
machine by `tools/tune.py`, and run-time tuning (explicit, FFTW-style wisdom) can replace them.

## Starting point

lc-opt's kernels (`lib/ntt`, `lib/poly`, `lib/multimod`) are AVX2 only, tuned for Zen 3, with no
fallback and no other ISA. They are the fastest implementations we have: start from their code
(it may move here) and their notes, which hold the measurements and lessons (register pressure,
spills, aliasing, Zen 3 instruction costs, and the attempts that lost). Read both before exploring
a kernel they cover. For other ISAs, port the algorithm and the generators, and measure.

v1 serves one modulus, P = 998244353 (DESIGN.md). Kernels for other moduli (`multimod`, `crt`)
come after v1.

## Targets

A target is a set of compiler flags and the machines that test and time it.

| Target | Flags | Vector registers | Test | Time |
|---|---|---|---|---|
| `scalar` | `-O2` | none | Mac, any VM | none |
| `sse` | `-march=x86-64-v2` | 16 xmm (128-bit) | `lc-amd` | none |
| `avx2` | `-march=x86-64-v3` | 16 ymm (256-bit) | `lc-amd` | `lc-bench` (Zen 3) |
| `avx512` | `-march=x86-64-v4 -mavx512ifma -mavx512vbmi` | 32 zmm, 8 masks | `lc-intel-test` | `lc-intel` (Emerald Rapids) |
| `zen5` | `-march=znver5` | 32 zmm, 8 masks | `lc-zen5-test` | `lc-zen5` (EPYC 9B45) |
| `arm` | `-mcpu=neoverse-v2` | 32 v (128-bit NEON, SVE2) | `lc-arm`, Mac | `lc-arm` (Axion) |

- `scalar` is the reference and the fallback; `sse` checks the generic code at 4 lanes. Neither is tuned.
- Every kernel builds and passes its tests on every target. Guard ISA code with the compiler's
  macros (`__AVX2__`, `__AVX512F__`, `__ARM_NEON`, ...) and keep the scalar path.

## Directions to explore

Starting points, not a list to finish. Agents pick what to try from measurements.

- Registers: 32 registers (AVX-512, ARM) against 16. Spend them on more butterflies in flight, a
  higher radix, or bigger leaves.
- Width: zmm against ymm on AVX-512 machines (AVX-512VL gives ymm 32 registers and masks).
- Mod multiply: Shoup, Montgomery, Barrett, IFMA 52-bit, fp64 with FMA. Costs differ per CPU
  (e.g. `vpmulld` is cheap on Zen 3, not on Intel).
- Lazy reduction bounds, radix (2, 4, 8), leaf size and in-register transposes, traversal (depth
  first with tiles, Stockham, four-step past L2), fusion of levels with pointwise products.
- New instructions: embedded broadcasts, two-source permutes, masks, `vpternlogd`, compress and
  expand, SVE2.
- Multimod: 3 primes of 30 bits against 2 of ~50 bits with IFMA, or a floating-point FFT.
- Cutoffs between small products and the NTT, per target.
- Memory: tile sizes from cache sizes, huge pages, prefetch, non-temporal stores.

Kernels to start with: `simd` (per-ISA vector wrappers), `modarith`, `ntt`, `multimod`, `crt`,
`values` (pointwise products), `small` (schoolbook, Karatsuba, middle products), `elementwise`,
`recurrence`. Agents may add, split or merge kernels.

## How to explore

- Write variants with intrinsics on the `simd` layer first: templates on the ISA, so one source
  runs on every target. Go to generated asm only where a profile shows the compiler is the limit;
  lc-opt's generators (`gen_*.py`) are the model.
- Each kernel has a scalar model that tracks value ranges. Every variant is tested against it.
- Each variant has a name. `backend/<kernel>/select.hpp` picks the variant and size thresholds per
  target from the macros; `-DFPS_<KERNEL>_<OP>=<variant>` forces one. No function pointers or
  virtual calls on hot paths.
- Tuned choices instead of hand-picked thresholds (`backend/tune/notes.md`): a choice is a table
  (`tune::Table`) the kernel reads once per call, outside loops. `backend/<kernel>/tune.hpp`
  measures it at run time and registers it; `backend/<kernel>/tune.cpp` measures it for
  `tools/tune.py <kernel>`, which writes the compiled defaults per target (`tuned.json`,
  `tuned.hpp`; `tune.py --check` runs in `kernel.py check`). After a kernel gets faster, rerun
  `tools/tune.py`. Benches and tests read the compiled defaults, never fresh tuning.
- Bench cases: `<op>/<size>` times the selected variant (gated); `<op>@<variant>/<size>` times one
  variant (reported). Sizes 2^6..2^25, to see L1, L2, L3 and memory.
- Before optimizing, estimate the floor: multiply uops over multiply pipes, or bytes over
  bandwidth. Report the time next to it.
- Where lc-opt has the operation (`../cpp_hpc/lc-opt`, `~/lc-opt` on the VMs; read-only), lc-opt
  is the bar: time it against ours in the same run on `lc-bench` (Zen 3, the machine it was tuned
  for), same compiler and flags, outputs checked equal, at several sizes, and report fps / lc-opt.
  The `Speed` section compares against `main` only, so this is the only check that a move kept
  lc-opt's speed. Keep the harness in the exploration folder (`<kernel>/issue<n>/`) for the next round.
- Measure instruction costs you rely on (`perf` on `lc-intel` and `lc-arm`; probes elsewhere).
  Model rankings (`llvm-mca`) did not predict the hardware in lc-opt.
- Test every variant the machine can run, not only the selected one: ragged sizes (0, 1, powers of
  two and neighbours), random inputs, inputs at the edge of each lazy range. ASan and UBSan.

## Working in parallel

- One agent per issue. An issue is a kernel, a kernel on one target, or a task. Two agents may work
  on the same kernel for different targets.
- Each agent has its own git worktree and branch, `agent/<kernel>-<target>-r<round>`.
- Read the kernel's `notes.md` and skim its `notes/` entries first. Do not repeat a logged attempt without a new reason.
- Agents may change any file. Say in the pull request what you changed outside your kernel.
- Log every attempt, win or loss, in a new file `notes/<issue>-r<round>.md` (AGENTS.md): date, idea,
  result, numbers, CPU, flags, commit.
- Delete losing variants from the repo once logged; keep their files in the exploration folder
  (VM: `~/explore/hpc_library/<kernel>/`; Mac: `~/Documents/hpc_library-explore/<kernel>/<vm>/`).
  Copy VM folders to the Mac before the round ends.
- On a VM, wrap every timing and profiling run in `flock /tmp/bench.lock`. Build and test on the
  test machine, not the timing machine.
- A change merges only if every target's tests pass (`tools/gate.py`, which does not time) and the
  pull request's `Speed` section shows no target slower beyond noise (geomean of new/base at most
  1.005, measured by the author with `kernel.py bench --base`, with the fps / lc-opt figure beside it; `tools/prompt.md`). The agent opens
  the pull request and returns; the coordinator checks the section, runs the gate and sends
  failures back (AGENTS.md, How work flows).
- A round ends with a comment on its issue: what was tried, the numbers, and a last line that is
  exactly `Result: gain` or `Result: no gain`. Speed work stops after 3 rounds, or 2 in a row
  without gain; a feature (a new API, a move, a fix) gets 1 round.
