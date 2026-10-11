# fps

A fast C++ library for univariate polynomials and power series modulo a prime.
It grows out of `../cpp_hpc/lc-opt`: its `lib/` and its per-problem polynomial code move here.

Two parts:

- **Backend**: the kernels (NTT now, more later), tuned per CPU. Agents explore them in rounds;
  see Backend and Process.
- **Frontend**: the user API in `DESIGN.md`, still in design. Agree on it before writing it.

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

## Layout

```
AGENTS.md                  rules (CLAUDE.md imports it)
DESIGN.md                  the user API (frontend)
BACKEND.md                 the kernel plan: targets, variants, exploration order
backend/bench.hpp          timing harness for kernel benches
backend/<kernel>/          one kernel: headers (API at the top), test.cpp, bench.cpp, notes.md, gen_*.py
tools/kernel.py            test and time a kernel, per target (VMs; the Mac for scalar)
tools/vm.py                run a command on a VM in a copy of your checkout
tools/gate.py              test and time a pull request on every target; merge it if it passes
tools/prompt.md            brief for one backend round
.claude/commands/work.md   /work: rounds on ready issues with subagents
```

## Backend

`BACKEND.md` is the plan: targets, kernels, variant axes, how variants are written and chosen, and
the exploration order. A kernel is fast inner code with a small API of its own, such as the NTT.
The frontend calls it.

- `test.cpp` checks every variant the machine can run against the kernel's scalar model, and exits
  nonzero on a failure.
- `bench.cpp` uses `backend/bench.hpp`. Case `<op>/<size>` times what the library uses on that
  target; the gate compares these. Case `<op>@<variant>/<size>` times one variant; it is only
  reported. Pass the items per call (coefficients, vectors) for a per-element unit.
- Generated code: commit the generator `gen_*.py` and its output. `gen_*.py --check` exits nonzero
  if they differ.
- `notes.md`: design, measurements per target, and the attempt log.
- Targets and their machines are `TARGETS` in `tools/kernel.py`: `scalar` (tested on `lc-amd` and
  the Mac), `sse` (tested on `lc-amd`), `avx2` and `avx512` (tested and timed). Every kernel builds
  and passes its tests on every target. To add a target, add it there with a machine for it.

## Tools

The Mac drives the VMs. The VMs cannot fetch this private repository: `tools/vm.py` sends trees over ssh.

- `python3 tools/vm.py <vm> [--base REV] '<command>'` copies your working tree to
  `~/hpc_library/<branch>` on the VM and runs the command there. With `--base`, REV goes to
  `$BASE` as well.
- `python3 tools/kernel.py`, on a VM, or on the Mac for `scalar`:
  - `test <kernel> --target <t>`: -O2 and ASan/UBSan builds.
  - `bench <kernel> --target <t> [--base $BASE]`: same-run paired ratios; below 1 is faster.
  - `check`: generators.
- `python3 tools/gate.py <pr> --merge` is the only way into `main`. It tests every affected kernel on
  every target, times each against `main` on the target's timing machine, comments the result on
  the PR, and squash-merges if all tests pass and no target is slower (geomean of new/base <= 1).
  `gate.py <rev> --base <rev>` runs the same checks on local commits, without a PR.

```
python3 tools/vm.py lc-amd 'python3 tools/kernel.py test ntt --target avx2'
python3 tools/vm.py lc-bench --base origin/main 'python3 tools/kernel.py bench ntt --base $BASE'
```

## Process

Backend work happens in rounds: one subagent per round, started by `/work` and briefed with
`tools/prompt.md`. Each GitHub issue is a kernel, a kernel on one target, or a task
(`.github/ISSUE_TEMPLATE/kernel.md`). An issue gets at most 5 rounds and stops after 2 rounds in a
row without gain, unless it says otherwise.

What to try: the variant axes in `BACKEND.md`. Start from the floor (pipe or bandwidth bound) and
the gap to it.

Agents may change any file, including other kernels and the tools, when it helps. The gate re-tests
everything a change touches. Say in the PR what you changed outside your kernel.

## Working in parallel

- One agent per issue at a time. Two agents may work on the same kernel for different targets.
- Each agent works in its own git worktree, on a branch `agent/issue<n>-r<round>` from `origin/main`.
- Read the kernel's `notes.md` first. Do not repeat a logged attempt without a new reason.
- Log every attempt, win or loss: date, idea, result, numbers (CPU, flags, commit).
- Keep exploration files: on a VM in `~/explore/hpc_library/<kernel>/`, on the Mac in
  `~/Documents/hpc_library-explore/<kernel>/<vm>/`. Copy them to the Mac before the round ends.

## Status

- Each issue is labeled `ready`, `running`, `blocked` (needs the user) or `done`.
- Each round ends with a comment on its issue: what was tried, the numbers, and a last line that
  is exactly `Result: gain` or `Result: no gain`. `/work` reads that line.

## Machines

We share lc-opt's VMs. Its `AGENTS.md` (Machines, Instruction sets) is the full reference.

| Machine | CPU | Use |
|---|---|---|
| Mac | ARM64 (M4 Pro), Apple clang | Edit, `scalar` tests (`kernel.py` runs locally). x86 binaries run under Rosetta; never time them. |
| `lc-intel` | Xeon 8581C (Emerald Rapids), AVX-512, PMU on | `avx512` tests and timing. Profiling with `perf` (core events, top-down; no L3 events). |
| `lc-amd` | EPYC 7B13 (Zen 3) | `avx2` builds and tests. No hardware counters. |
| `lc-bench` | EPYC 7B13 (Zen 3) | `avx2` timing only. Build and test elsewhere; `kernel.py` holds the lock for its own builds. |
| `lc-k68` | EPYC 7B13, Linux 6.8 | Timing of code heavy in page faults or huge pages. Keep it on 6.8. |

- GCP project `project-c73e6eb1-e167-4d7a-a31`, region `europe-west2`. Zones: `lc-intel` is in
  `europe-west2-c`, the rest in `europe-west2-b`.
  Connect: `gcloud compute ssh <name> --zone=<zone>`.
- All run Ubuntu 24.04 with Docker and the `gcc:15.2.0` image. Build with GCC 15.2.
- Copies of this repo live in `~/hpc_library/<name>` (`tools/vm.py`). Do not touch `~/lc-opt`;
  other agents use it.
- Flags: `-march=x86-64-v3` (AVX2, no AVX-512) by default. `-march=native` on `lc-intel` lets GCC
  emit AVX-512 unasked; label such timings.
- Wrap every timing or profiling run in `flock /tmp/bench.lock` (`kernel.py bench` does). On
  `lc-amd`, builds and tests may run in parallel.
- Keep VMs running; do not stop them. Never delete a VM you did not create.
- Need more capacity? Create a VM in the same project and region, Ubuntu 24.04, name `lc-<purpose>`,
  and add it to lc-opt's table too.
- Keep exploration files in `~/explore/hpc_library/` on the VM, not `/tmp`. Copy them back to the Mac
  before you finish; a VM can be lost.
