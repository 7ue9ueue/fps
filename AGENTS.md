# fps

A fast C++ library for univariate polynomials and power series modulo a prime.
It grows out of `../cpp_hpc/lc-opt`: its `lib/` and its per-problem polynomial code move here.

Status: design phase. Agree on the API before writing library code.

## Hard rules

- Write all code ourselves. Papers, docs and other open-source code may be read for ideas, never
  copied. Cite what you used in the module's `notes.md`.
- Code from `../cpp_hpc/lc-opt` may be moved here. Rewrite it to this repo's standards as you go.
- No speed claim without passing tests and a measurement.
- Ask before changing an agreed design decision.

## Performance

Speed is the point of this library. The lc-opt kernels are already fast; the API must not slow them down.

- No overhead from wrapping. A call through the public API must run as fast as calling the kernel
  directly: no extra copies, allocations, bounds checks or virtual calls on hot paths.
- Measure it. For each wrapper, time it against the raw kernel in the same run (see Measuring).
  A gap beyond noise is a bug.
- If an ergonomic API costs speed, keep the fast path reachable (raw spans, plans, transformed values)
  and say so in the header.

## Writing

- English. Short, plain sentences. Numbers over adjectives. Mark guesses as guesses.
- Notes are logs: what was tried, the result, the evidence.
- Commits: imperative one-liner, at most 72 characters.

## Code

- C++23. Readable first: clear names, small functions, no dead code.
- Comments are short and rare: invariants, value ranges, overflow bounds, memory layout.
- Each module has a small documented API (in its header) and its own tests.

## Correctness

- Test against a simple reference (O(n^2) or brute force) on small sizes, edge sizes
  (0, 1, powers of two and their neighbours) and random inputs.
- Run the tests under ASan and UBSan before merging.

## Measuring

- Time old and new code in the same run, on the same inputs, repeated. Report median and spread.
- Record CPU, compiler, flags and commit with the numbers.
- The Mac is ARM64: time x86 code only on x86 machines (the VMs below).

## Machines

We share lc-opt's VMs and add three for the backend targets (`BACKEND.md`). lc-opt's `AGENTS.md`
(Machines, Instruction sets) is the full reference.

| Machine | CPU | Use |
|---|---|---|
| Mac | ARM64 | Edit, build, correctness. x86 binaries run under Rosetta; never time them. |
| `lc-intel` | Xeon 8581C, AVX-512, PMU on | `avx512` timing. Profiling with `perf` (core events, top-down; no L3 events). |
| `lc-amd` | EPYC 7B13 (Zen 3) | Builds and tests. No hardware counters. |
| `lc-bench` | EPYC 7B13 (Zen 3) | Timing only. Do not build or test here. |
| `lc-k68` | EPYC 7B13, Linux 6.8 | Timing of code heavy in page faults or huge pages. Keep it on 6.8. |
| `lc-intel-test` | Xeon 8581C, AVX-512, no PMU | `avx512` builds and tests. |
| `lc-zen5` | EPYC 9B45 (Zen 5), AVX-512, no PMU | `zen5` builds, tests and timing. |
| `lc-arm` | Axion (Neoverse V2), NEON, SVE2, PMU on | `arm` builds, tests and timing; `perf`. |

- GCP project `project-c73e6eb1-e167-4d7a-a31`, region `europe-west2`. Zones: `lc-intel` and
  `lc-intel-test` are in `europe-west2-c`, the rest in `europe-west2-b`.
  Connect: `gcloud compute ssh <name> --zone=<zone>`.
- All run Ubuntu 24.04 with Docker and the `gcc:15.2.0` image. Build with GCC 15.2.
- Copy this repo with `gcloud compute scp --recurse` (or `git` once it has a remote) into
  `~/hpc_library`. Do not touch `~/lc-opt`; other agents use it.
- Flags: `-march=x86-64-v3` (AVX2, no AVX-512) by default. `-march=native` on `lc-intel` lets GCC
  emit AVX-512 unasked; label such timings.
- Wrap every timing or profiling run in `flock /tmp/bench.lock`. Builds and tests may run in parallel.
- Keep VMs running; do not stop them. Never delete a VM you did not create.
- Need more capacity? Create a VM in the same project and region, Ubuntu 24.04, name `lc-<purpose>`,
  and add it to lc-opt's table too.
- Keep exploration files in `~/explore/hpc_library/` on the VM, not `/tmp`. Copy them back to the Mac
  before you finish; a VM can be lost.
