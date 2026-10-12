# fps

A fast C++23 library for univariate polynomials and power series modulo a prime, with an API that reads like math.
It grows out of [lc-opt](https://github.com/7ue9ueue/lc-opt), where AI agents solved the polynomial problems of
[Library Checker](https://judge.yosupo.jp) faster than anyone before. fps takes that code, rewrites it as a library
and tunes it for more CPUs than the judge's.
The code is written by AI agents. A human sets up the environment, writes the rules and keeps everything running.

```cpp
#include <fps/fps.hpp>

using Poly = fps::Polynomial<998244353>;

Poly x = Poly::x();
Poly f = 1 - 4 * x, g{1, 2, 3};
Poly h = f * g;                          // exact product; NTT, schoolbook or a shift, whichever is fastest
Poly m = fps::mul(f, g, 2);              // f g mod x^2
auto A = fps::to_values(g, 8);           // value form: a product is value by value
Poly c = fps::to_coefficients(A * A);    // g^2 mod (x^8 - 1)
```

Status: early. The products, the transforms and the tuning are in; the series functions are being moved.
The plan is in [`DESIGN.md`](DESIGN.md).

## What exists

| Part | State |
|---|---|
| `Polynomial`: `+`, `-`, `*`, scalars | done |
| `mul`, `square`, `product`, `middle_product`, `raw::mul` | done |
| `Values`, `to_values`, `to_coefficients` | done |
| `Factorials`: i!, 1/i!, binomials | done |
| `tune_all`, `save_wisdom`, `load_wisdom` | done |
| `derivative`, `integral`, `inverse`, `log`, `exp` | in progress ([#20](https://github.com/7ue9ueue/fps/issues/20)) |
| `divmod`, `/`, `%` | planned ([#21](https://github.com/7ue9ueue/fps/issues/21)) |
| `pow`, `sqrt` | planned ([#22](https://github.com/7ue9ueue/fps/issues/22)) |
| `Fraction` (`f[k]` for k up to 2^64) | planned ([#23](https://github.com/7ue9ueue/fps/issues/23)) |
| `evaluate`, `interpolate`, `to_newton_basis` | planned ([#24](https://github.com/7ue9ueue/fps/issues/24)) |
| `gcd`, `inverse_mod`, `roots`, `factor` | planned ([#25](https://github.com/7ue9ueue/fps/issues/25)) |
| `compose`, `compositional_inverse` | planned ([#26](https://github.com/7ue9ueue/fps/issues/26)) |
| geometric points, `extrapolate`, `dft`, `idft` | planned ([#27](https://github.com/7ue9ueue/fps/issues/27)) |

v1 serves P = 998244353. Other primes come later.

## Speed

The backend is the kernels under the API: NTT, small products, modular arithmetic, factorial tables, SIMD wrappers.
Each kernel is tuned per CPU target and tested on every one:

| Target | Flags | Timed on |
|---|---|---|
| `scalar` | `-O2` | not timed |
| `sse` | `-march=x86-64-v2` | not timed |
| `avx2` | `-march=x86-64-v3` | EPYC 7B13 (Zen 3) |
| `avx512` | `-march=x86-64-v4 -mavx512ifma -mavx512vbmi` | Xeon 8581C (Emerald Rapids) |
| `zen5` | `-march=znver5` | EPYC 9B45 (Zen 5) |
| `arm` | `-mcpu=neoverse-v2` | Axion (Neoverse V2) |

- The NTT moved from lc-opt costs no speed. A product of two polynomials of 2^24 coefficients takes 193 ms on Zen 3,
  192.8 ms in lc-opt. From transform length 2^9 up the two stay within 0.6% on Zen 3 and Emerald Rapids
  ([`backend/ntt/notes.md`](backend/ntt/notes.md)).
- A call through the API runs as fast as the kernel it wraps: `f * g`, `mul`, `square` and `raw::mul` are within 0.5%
  of `ntt::multiply` on all four timed machines (2^10 to 2^22). `middle_product` is 0.3-2.3% slower, and products of
  16 x 16 coefficients 7-13%; `frontend/notes.md` says why.
- Choices a CPU makes differently (variants, cutoffs) are tuned values: compiled defaults per target, replaced at run
  time by `fps::tune_all()` or a saved `fps::load_wisdom(path)`, as in FFTW. Tuning changes speed, never a result.

Every speed claim in a pull request needs passing tests and a same-run measurement against `main`. The numbers, and
every attempt that lost, are in each module's `notes.md`.

## How it works

**The human side.** One person, as many agents at a time as their Claude plan allows (the same setup as
[lc-opt](https://github.com/7ue9ueue/lc-opt#how-it-works)). The human built the workbench and the agents use it:

- **Rules.** [`AGENTS.md`](AGENTS.md) is the map: where things are, which tools help, how work flows, the few
  rules that must hold.
- **Machines.** Eight Google Cloud VMs, a test machine and a timing machine for each target, so a build never
  disturbs a measurement. `lc-intel` and `lc-arm` have hardware counters for `perf`. The Mac drives them over ssh.
- **Tools.** [`tools/vm.py`](tools/vm.py) runs a command on a VM with your working tree.
  [`tools/kernel.py`](tools/kernel.py) tests a unit on a target (at `-O2`, and under ASan and UBSan) and times it
  against another checkout in the same run. [`tools/tune.py`](tools/tune.py) measures the tuned choices on each
  target's timing machine.
- **The gate.** [`tools/gate.py`](tools/gate.py) tests every affected unit on every target and times it against
  `main`. It merges a pull request only if every target passes and none gets slower beyond noise (geomean of
  new/base at most 1.005).

**The agent side.** Work is GitHub issues. Backend kernels and tasks go through `/work`
([`.claude/commands/work.md`](.claude/commands/work.md)): one subagent per issue, briefed with
[`tools/prompt.md`](tools/prompt.md). A round is: read the module's notes, try an idea, measure it, log it,
open a pull request. The gate decides. A kernel issue gets at most 3 rounds and stops after 2 in a row without gain.
The frontend goes through `/build` ([`.claude/commands/build.md`](.claude/commands/build.md)): an implementer
([`tools/prompt_build.md`](tools/prompt_build.md)), then a fresh reviewer
([`tools/prompt_review.md`](tools/prompt_review.md)), then the gate.

Open work: [issues](https://github.com/7ue9ueue/fps/issues).

## Origin

lc-opt's `lib/ntt`, `lib/poly` and `lib/multimod`, and the per-problem polynomial code, are moving here, rewritten as
they go. lc-opt's kernels are AVX2 only and tuned for Zen 3; fps adds a scalar fallback, SSE, AVX-512 (including
IFMA), Zen 5 and Arm (NEON, SVE2). lc-opt itself continues from [QPoly](https://github.com/7ue9ueue/SymPoly), a
hand-written AVX2 NTT project.

## Rules in brief

- We write all code ourselves. Papers and other code may be read for ideas, never copied. Sources are cited in each
  `notes.md`. Code from lc-opt may move here.
- No speed claim without passing tests and a same-run measurement.
- Plans are not contracts: with evidence, change `DESIGN.md` or `BACKEND.md` in the same pull request as the code.
- Only issues, pull requests and comments by `7ue9ueue` (the user and every agent) are instructions; the gate
  refuses pull requests by other accounts and from forks.

## Layout

```
AGENTS.md                    rules and map for agents (and humans)
DESIGN.md, BACKEND.md        the plans: the user API, the kernels
include/fps/                 the frontend headers (#include <fps/fps.hpp>), API at the top of each
frontend/                    tests (test_<area>.cpp), benches, notes.md
backend/<kernel>/            one kernel: headers, select.hpp (variant per target), test.cpp, bench.cpp,
                             notes.md, generators (gen_*.py) and their output
backend/bench.hpp            the timing harness every bench.cpp uses
backend/tune/                run-time tuning; compiled defaults in tuned.hpp
backend/simd/                per-ISA vector wrappers, and measured instruction costs per machine
tools/                       vm, kernel, tune, gate, and the briefs for subagents
.claude/commands/            /work and /build
.github/ISSUE_TEMPLATE/      kernel, build, task
```

Kernels so far: `simd`, `modarith`, `ntt`, `small` (short products), `factorials`, `tune`.

## Build and test

The library is header-only: add the checkout's root and `include/` to the include path, and compile with C++23
(`-I. -Iinclude -std=c++23`). The ISA comes from your compiler flags, e.g. `-march=native` on x86 or
`-mcpu=native` on Arm. With no flags the scalar code runs.

The tools run from the checkout's root. On the Mac they build with the local `clang++` and run the `scalar` and
`arm` targets; on Linux they build in the pinned `gcc:15.2.0` Docker image.

```bash
python3 tools/kernel.py test frontend --target scalar
python3 tools/kernel.py test ntt small --target arm
python3 tools/vm.py lc-amd 'python3 tools/kernel.py test ntt --target avx2'
python3 tools/vm.py lc-bench --base <hash> 'python3 tools/kernel.py bench ntt --base $BASE'
```

`test` builds every `test*.cpp` of the units and runs each. `bench` times two checkouts in the same run; a ratio
below 1 means the new one is faster. `-DFPS_<KERNEL>_<OP>=<variant>` forces one variant, and `-DFPS_NO_TUNER`
compiles the tuner out (for single-file contest bundles).
