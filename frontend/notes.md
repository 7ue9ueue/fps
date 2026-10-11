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
- 2026-10-11: `Polynomial` (#16): the class, `f + g`, `f - g`, `-f`, `f * g`, scalars, compound
  forms, from the prototype's `polynomial.hpp`. Storage is a `std::vector<std::uint32_t>`.
  `static_assert(P == 998244353)` as DESIGN.md (Moduli) says, so no product for other primes.
  - DESIGN.md: its Examples need `1 - x`, `x + c` and `-1 * f`, which its list of operators did
    not have. Now `f + c`, `c + f`, `f - c`, `c - f` are listed, and a scalar c is any integer
    type, taken mod P. Conventions: a size past a backend limit throws `std::length_error` (an
    NTT product past 2^25 coefficients; `ntt::Convolution` throws it), not `std::domain_error`.
  - Sums, negation and scaling run on the backend's primitives (`detail::lanewise` on
    `simd::Native` with `modarith::Ops`: `add`, `sub`, Shoup's product), since GCC 15.2 does not
    vectorize such loops at -O2.
  - `f * g`, n and m coefficients: a factor with at most one nonzero term (c x^k; constants and
    zero too) is a shift and scale, O(n + m). Else schoolbook if min(n, m) <= 32 or n m < 2^18
    (`detail::schoolbook`, 64-bit sums), else `ntt::Convolution`. Schoolbook is here because a
    short `ntt::Convolution` costs 52 us on `lc-bench` (22 us on `lc-arm`), mostly the first touch
    of a fresh huge page, against 0.43 us for schoolbook at 32 x 32. Cutoffs measured on
    `lc-bench`: schoolbook 0.24 ns per term; it wins up to about 450 x 450 (384 x 384: 36 us
    against 56 us; 512 x 512: 64 against 55) and up to m = 32 at n = 2^14, 2^17, 2^20 (m = 32:
    136 us against 163 at 2^14; m = 48: 192 against 163). Backend issues: #37 (`small`: a tuned
    short product and per-target cutoffs), #38 (`ntt`: memory kept across calls, product into
    caller memory).
  - Tests (`test_polynomial.cpp`): every member and operator against O(n^2) references, on sizes
    0, 1, 2^k - 1, 2^k, 2^k + 1 (k <= 10), every pair of them for `f * g`, random inputs with
    many zeros or all P - 1 (the largest schoolbook sums); shapes at both cutoffs; NTT lengths
    2^10 .. 2^21 at random points; shifts by c x^k with c = 0, 1, random; aliasing (f += f,
    f -= f, f *= f); 4 threads; both errors. 10 injected bugs all fail it: schoolbook's reduction
    cadence and a block bound, the shift's offset and scale, single-term detection, `degree`,
    negation, `f - c` on an empty f, the residue of a negative c, `truncate`.
  - Bench (`frontend/bench.cpp`): `multiply/<lg>` is `f * g` for factors of 2^lg / 2 coefficients,
    `multiply@kernel/<lg>` the same `ntt::Convolution` call by hand (construct, copy in,
    multiply). Below lg = 10 the API runs schoolbook. Per-round ratio API / kernel, median of 11
    rounds (IQR in the PR), GCC 15.2, main at 06feb5c (#35) plus this change; `lc-arm` before #35,
    which did not change its scalar NTT:

    | lg | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm, scalar NTT) |
    |---:|---:|---:|---:|---:|
    | 6 | 0.008 (436 ns, 52 us) | 0.008 (404 ns, 51 us) | 0.010 (303 ns, 30 us) | 0.017 (359 ns, 22 us) |
    | 9 | 0.282 | 0.253 | 0.361 | 0.418 |
    | 10 | 1.006 | 1.015 | 1.006 | 1.006 |
    | 12 | 1.008 | 1.023 | 1.009 | 1.003 |
    | 14 | 1.017 | 1.046 | 1.016 | 1.002 |
    | 16 | 1.026 | 1.044 | 1.023 | 1.002 |
    | 18 | 1.031 | 1.028 | 1.021 | 1.002 |
    | 20 | 1.038 | 1.044 | 1.019 | |
    | 22 | 1.058 | 1.046 | 1.035 | |

    The API is 1-6% slower than the kernel case from 2^10 on x86. `ntt::Convolution` leaves the
    product in its own memory, which it unmaps when it dies, so `f * g` must write it elsewhere.
    That write is the whole gap, and no frontend change can avoid it. Same-run decomposition
    (`lc-intel`, 2^14; us):
    - kernel case: 113.4;
    - plus a copy into one buffer kept across calls: 118.5;
    - plus a new zeroed vector, no copy: 118.7;
    - plus a copy into a new vector: 119.1;
    - the API: 118.9.

    Any write of the product's size after the product costs the same 5 us. The copy alone, timed
    by itself, takes 1.4 us; guess: the fresh mapping per call (#38) makes it dearer. #38 asks
    for a product into caller memory.

    Copy destinations tried (`lc-bench`, out of a page-aligned array, 2^12 to 2^16 words):
    - a 64-byte aligned vector is 4-5x slower (equal page offsets: 4K aliasing);
    - 16 bytes past a page boundary is 15-20x slower;
    - `std::vector`'s own offset is as fast as 64 bytes past.

    On `lc-intel` the offset changes memcpy by at most 10%.
