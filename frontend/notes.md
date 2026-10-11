# frontend

The user API in `include/fps/`, planned in `DESIGN.md`, built issue by issue with `/build`.
Tests: `frontend/test_*.cpp`, one per area. Bench: `frontend/bench.cpp` (API calls as `<op>/<size>`,
the backend call they wrap as `<op>@kernel/<size>`).

Prototype: a local, unpushed branch (`claude/frontend-lc-opt-backend-8360f6`, worktree
`.claude/worktrees/frontend-lc-opt-backend-8360f6`) implements most of DESIGN.md over a private copy
of lc-opt's kernels. Read it and move code from it; do not copy its `include/fps/lcopt/` (that work
belongs in `backend/`).

## Log

- 2026-10-11: scalars (`mod_mul`, `mod_pow`, `mod_inverse`) from the prototype's `modular.hpp`,
  as the first piece through the pipeline. Tests against brute force on 6 primes from 2 to 2^32 - 5.
- 2026-10-11: `Factorials` (#17): tables of i! and 1 / i!, `binomial(n, k)`, from the prototype's
  `factorials.hpp`. The backend has no kernel for it, so `detail::factorial_tables` is the simple
  loop: two chains of n - 1 dependent `mod_mul` and one `mod_inverse`. lc-opt's 32-lane product
  chains are backend issue #33. `static_assert(P == 998244353)` as DESIGN.md (Moduli) says.
  - DESIGN.md: its line on `Factorials` did not say which n are valid, what `inverse(i)` is, what
    C(n, k) is for k > n, or whether indices are checked. Now: n <= P, as P! = 0 has no inverse
    (n > P throws before the tables are allocated); `inverse(i)` = 1/i!; C(n, k) = 0 for k > n;
    indices unchecked, as `operator[]`.
  - Tests: the tables against running products for n = 0..3, 2^k - 1, 2^k, 2^k + 1 (k <= 20) and
    random n; `detail::factorial_tables` on P = 2, 3, 5, 101 for every n <= P (Wilson at n = P)
    and on 2^32 - 5; binomials against Pascal's triangle (300 rows) and against products of k
    factors (random); both errors. Five injected bugs (bound, binomial edge, index, table entry,
    first entry) all fail the test.
  - Bench (`frontend/bench.cpp`): `factorials/<lg>` builds the tables, `binomial/16` is 2^16
    random C(m, k) from a table of 2^16. No `@kernel` pair: no backend call is wrapped. Median ns
    per entry (min-max), 11 rounds, GCC 15.2, the code of PR #34:

    | Machine (target) | factorials/10 | /14 | /18 | /22 | binomial/16 |
    |---|---:|---:|---:|---:|---:|
    | `lc-bench` (avx2) | 7.13 (7.12-7.15) | 7.02 (7.01-7.03) | 10.26 (10.21-10.30) | 10.60 (10.58-10.63) | 3.58 (3.47-3.64) |
    | `lc-intel` (avx512) | 5.91 (5.91-5.93) | 5.86 (5.86-5.89) | 7.64 (7.62-7.68) | 7.83 (7.83-7.85) | 1.56 (1.54-1.59) |
    | `lc-zen5` (zen5) | 5.95 (5.94-5.96) | 5.86 (5.85-5.86) | 6.89 (6.88-6.97) | 7.11 (7.10-7.14) | 1.78 (1.76-1.78) |
    | `lc-arm` (arm) | 8.24 (8.23-8.25) | 8.09 (8.09-8.10) | 9.66 (9.64-9.72) | 10.12 (10.07-10.14) | 2.90 (2.88-3.01) |

    The loop is latency-bound: each product waits on the one before. lc-opt's chains: about
    0.6 ns per entry (estimate from its notes, #33). The rise from lg 14 to lg 18 is most likely
    page faults on the fresh, zero-filled vectors (guess, not profiled).
