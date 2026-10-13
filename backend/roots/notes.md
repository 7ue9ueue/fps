# backend/roots

Root finding modulo P = 998244353: the distinct roots in F_P of a polynomial, by one round of the
tangent Graeffe method. API: `roots.hpp` (namespace `fps::root_finding`, as `fps::roots` is the
frontend's function: `roots(f, out)` returns m, out[0, m) increasing). Kernels on `backend/simd`
structs: `kernels.hpp` (the Graeffe step on pairs of leaves, wide sums, zero masks, powers of 3,
the radix sort). Structs per target: `select.hpp`. Tests: `test.cpp`; bench: `bench.cpp`.
Attempts: `notes/` (one file per issue round). From lc-opt's
`problems/polynomial/polynomial_root_finding` (`graeffe.hpp`, `solution.cpp`, its `notes.md`;
lc-opt 13e78b2, issue #122), rewritten on `ntt::Transform`, `series` (per-thread transforms,
`multiply_by_index`), `factorials` and `ntt::middle_product`, so every target runs it.

## Design

lc-opt's design (its notes.md, Design), with these changes:

- Graeffe steps at length bit_floor(d) when d + 1 exceeds it by at most kWrap = 16: the cyclic
  result mod (y^n - 1) holds the coefficients from n on added to the first ones; those (at most
  16 for A, 15 for B) are computed directly from the top 2 kWrap coefficients of the iterate
  (`wrapped`: O(kWrap^2) a step, 2^-31 the step's factor: the Montgomery sums' 2^-32 and the CRT
  without 1/2), subtracted after the inverse, and subtracted twice more for forward_upper's input
  (the result mod (y^n + 1)). The shift does the same for up to kShiftWrap = 8 coefficients
  (added by vector Montgomery products). lc-opt ran degree 2^j at length 2^(j+1): twice the
  transforms.
- l from d <= 8 2^l (lc-opt: d + 1 <= 8 2^l): only d = 2^j changes, to half the points (9 terms
  per row, one fold more).
- On H only A is evaluated, and x A' and B for groups of cosets with zeros of A (all three at
  once while the previous group had zeros): random f (about one root) evaluates a third.
- Groups of 16 cosets (or tracked points) on AVX-512 (`Folds = simd::Avx512`): the folds' sums in
  zmm, two rows of 8 lanes out, each its own transform. zmm vpmuludq runs at ymm's throughput on
  Emerald Rapids and Zen 5 (backend/simd/machines.py), so a fold costs half.
- The Graeffe step on x86 (`even_parts_columns`): the 26 columns of a leaf pair (a, b, P - a odd,
  2 a) through a stack buffer and each lane parity in its own pass of 64-bit sums, so an operand
  is a register or a memory operand. With Sums' EvenOdd operands (two registers each) all live,
  GCC spilled 678 times per 8 pairs.
- Output sorted (the API's order) by a 3-pass radix sort; std::sort took about 1/6 of the time at
  4000 roots.
- Memory: a per-thread workspace kept across calls; tables (leaf weights, factorials) per thread,
  grown on demand; transforms: series::detail::transform. lc-opt mapped one huge page and built
  its tables per run (100-160 us at N = 4000).
- The folds' loop over the terms carries `#pragma GCC novector` (GCC only): GCC 15.2 at -O2 with
  AVX2 vectorizes it wrongly for the Scalar struct's u64 sums (issue #183).

Costs (Zen 3, `lc-bench`, d = 4096, random f): shift 29 us, Graeffe 296 us (14 steps of 21 us:
the pair step 9 us, two inverses and two forward_upper of 4096 2.8-2.9 us each), H 108 us (A only).
With 4096 distinct roots: H 343 us (all three), climbs 184 us (131 multiple zeros, all
collisions). Collisions grow as roots^2 / |H|, so many roots or repeated roots cost up to about
d^2 / 300 in the climbs (issue below).

## Measurements

See `notes/122-r1.md`: against lc-opt in the same run (Zen 3), the phases, the bench on every
target and against the frontend's Cantor and Zassenhaus.
