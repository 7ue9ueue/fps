# fps

A fast C++ library for univariate polynomials and power series modulo a prime.
It grows out of `../cpp_hpc/lc-opt`: its `lib/` and its per-problem polynomial code move here.

Two parts:

- **Backend**: the kernels (NTT now, more later), tuned per CPU. Agents explore them in rounds;
  see Backend and Process.
- **Frontend**: the user API, planned in `DESIGN.md`. Built from that plan issue by issue with
  `/build`; see Process.

## Hard rules

- Write all code ourselves. Papers, docs and other open-source code may be read for ideas, never
  copied. Cite what you used in the module's `notes.md`.
- Code from `../cpp_hpc/lc-opt` may be moved here. Rewrite it to this repo's standards as you go.
- No speed claim without passing tests and a measurement.
- `DESIGN.md` and `BACKEND.md` are plans, not contracts. If you find a better approach, change the
  design doc first, then build the new approach. Do not follow the doc into a worse result.
  - Better means measured or proven: numbers, a test, or a short argument in the commit message.
    A guess does not count.
  - Edit the affected rows and sections in place. Do not leave the old text behind.
  - Say what changed and why in the commit message and in the module's `notes.md`.
  - Keep the doc and the code in step: no commit where they disagree.
  - Tell the user in your final report. No need to ask first.
- When a request is unclear, ask before acting. Do not guess at intent; say what is unclear and
  offer the options you see.
- When you need a permission or access you lack (a VM, a gcloud login, a credential, a blocked
  command), stop and ask the user for help. Do not work around it.
- An unattended agent asks on its issue: a comment with the question and the options you see,
  then the `blocked` label. Then it stops and returns.

## Design

- Start simple. Build the smallest design that covers today's callers.
- Add generality (templates, options, extension points, abstractions) only when a real caller needs it.
  Write down which caller needed it.
- Simple must not mean slow: the Performance rules below still apply.

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
include/fps/               the frontend: the user API (`#include <fps/fps.hpp>`)
frontend/                  its tests (test_<area>.cpp), bench.cpp and notes.md
backend/bench.hpp          timing harness for kernel and frontend benches
backend/<kernel>/          one kernel: headers (API at the top), test.cpp, bench.cpp, notes.md, gen_*.py
tools/kernel.py            test and time a unit (a kernel or the frontend), per target
tools/vm.py                run a command on a VM in a copy of your checkout
tools/gate.py              test and time a pull request on every target; merge it if it passes
tools/prompt.md            brief for one backend round
tools/prompt_build.md      brief for building one frontend issue
tools/prompt_review.md     brief for reviewing a frontend pull request against DESIGN.md
.claude/commands/work.md   /work: backend rounds on ready issues with subagents
.claude/commands/build.md  /build: frontend issues, each implemented, reviewed and gated
```

## Backend

`BACKEND.md` is the plan and the working rules: targets, directions to explore, how to write, test
and time variants, and how agents work in parallel. A kernel lives in `backend/<kernel>/`:

- headers, with the kernel's API at the top, and `select.hpp` (the variant per target);
- `test.cpp`: every variant the machine can run against the scalar model; exits nonzero on a failure;
- `bench.cpp`, on `backend/bench.hpp`: `<op>/<size>` cases are gated, `<op>@<variant>/<size>` cases
  are reported. Pass the items per call (coefficients, vectors) for a per-element unit;
- generated code: the generator `gen_*.py` and its output, both committed; `gen_*.py --check`
  exits nonzero if they differ;
- `notes.md`: design, measurements per target, and the attempt log.

Targets and their machines are `TARGETS` in `tools/kernel.py`, kept in step with `BACKEND.md`.

## Tools

The Mac drives the VMs. `tools/vm.py` sends your working tree over ssh, uncommitted changes included.

- `python3 tools/vm.py <vm> [--base REV] '<command>'` copies your working tree to
  `~/hpc_library/<branch>` on the VM and runs the command there. With `--base`, REV goes to
  `$BASE` as well.
- `python3 tools/kernel.py`, on a VM, or on the Mac for `scalar` and `arm`. A unit is a kernel
  name or `frontend`. Builds use `-I. -Iinclude`.
  - `test <unit> --target <t>`: every `test*.cpp` of the unit, -O2 and ASan/UBSan.
  - `bench <unit> --target <t> [--base $BASE]`: same-run paired ratios; below 1 is faster.
  - `check`: generators.
- `python3 tools/gate.py <pr> --merge` is the only way into `main`. It tests every affected unit on
  every target (the frontend too, when `include/`, `frontend/` or a kernel it includes changes),
  times each against `main` on the target's timing machine, comments the result on
  the PR, and squash-merges if all tests pass and no target is slower (geomean of new/base <= 1).
  `gate.py <rev> --base <rev>` runs the same checks on local commits, without a PR.
  Issues and pull requests share numbers: pass the pull request's.

```
python3 tools/vm.py lc-amd 'python3 tools/kernel.py test ntt --target avx2'
python3 tools/vm.py lc-bench --base origin/main 'python3 tools/kernel.py bench ntt --base $BASE'
```

## Process

Backend work happens in rounds: one subagent per round, started by `/work` and briefed with
`tools/prompt.md`. Each GitHub issue (`.github/ISSUE_TEMPLATE/kernel.md`) is labeled `ready`,
`running`, `blocked` (needs the user) or `done`. `/work` reads the `Result:` line each round ends
with (`BACKEND.md`, Working in parallel).

Frontend work is building to a plan, not exploring: `/build` takes issues labeled `build`, one
implementer each (`tools/prompt_build.md`), then a reviewer who checks the pull request against
DESIGN.md (`tools/prompt_review.md`), then the gate. An issue ends with `Result: done`. `/work`
skips `build` issues.

Agents may open issues and comment on any issue. Open one from the template, labeled `ready`, for
work found outside your round: a bug, a new kernel, an idea for another target. Search open issues
first; comment on a match instead of opening a duplicate. Reply to an issue when you have something
useful for it: an answer, a measurement, a related finding.

## Machines

We share lc-opt's VMs and add three for the backend targets (`BACKEND.md`). lc-opt's `AGENTS.md`
(Machines, Instruction sets) is the full reference.

| Machine | CPU | Use |
|---|---|---|
| Mac | ARM64 (M4 Pro), Apple clang | Edit, `scalar` and `arm` tests (`kernel.py` runs locally). x86 binaries run under Rosetta; never time them. |
| `lc-intel` | Xeon 8581C (Emerald Rapids), AVX-512, PMU on | `avx512` timing. Profiling with `perf` (core events, top-down; no L3 events). |
| `lc-amd` | EPYC 7B13 (Zen 3) | `scalar`, `sse` and `avx2` builds and tests. No hardware counters. |
| `lc-bench` | EPYC 7B13 (Zen 3) | `avx2` timing only. Build and test elsewhere; `kernel.py bench` builds its two binaries here under the lock. |
| `lc-k68` | EPYC 7B13, Linux 6.8 | Timing of code heavy in page faults or huge pages. Keep it on 6.8. |
| `lc-intel-test` | Xeon 8581C, AVX-512, no PMU | `avx512` builds and tests. |
| `lc-zen5` | EPYC 9B45 (Zen 5), AVX-512, no PMU | `zen5` builds, tests and timing. |
| `lc-arm` | Axion (Neoverse V2), NEON, SVE2, PMU on | `arm` builds, tests and timing; `perf`. |

- GCP project `project-c73e6eb1-e167-4d7a-a31`, region `europe-west2`. Zones: `lc-intel` and
  `lc-intel-test` are in `europe-west2-c`, the rest in `europe-west2-b`.
  Connect: `gcloud compute ssh <name> --zone=<zone>`.
- All run Ubuntu 24.04 with Docker and the `gcc:15.2.0` image. Build with GCC 15.2.
- Copies of this repo live in `~/hpc_library/<name>` (`tools/vm.py`). Do not touch `~/lc-opt`;
  other agents use it.
- Flags: per target (`TARGETS` in `tools/kernel.py`). `-march=native` on `lc-intel` lets GCC emit
  AVX-512 unasked; label such timings.
- Wrap every timing or profiling run in `flock /tmp/bench.lock` (`kernel.py bench` does). On
  `lc-amd`, builds and tests may run in parallel.
- Keep VMs running; do not stop them. Never delete a VM you did not create.
- Need more capacity? Create a VM in the same project and region, Ubuntu 24.04, name `lc-<purpose>`,
  and add it to lc-opt's table too.
- Keep exploration files in `~/explore/hpc_library/` on the VM, not `/tmp`. Copy them back to the Mac
  before you finish; a VM can be lost.
