# backend/taylor_shift

The Taylor shift modulo P = 998244353: `shift(f, c, out)`, out[k] = [x^k] f(x + c) for
k < out.size() <= f.size() (API: `taylor_shift.hpp`). From lc-opt's
`problems/polynomial/polynomial_taylor_shift` (`solution.cpp`, judged as `main.cpp` of lc-opt #251),
rewritten on `backend/simd`, ntt's in-place product and `backend/factorials`' lane grid, so every
target runs it. `select.hpp`: the scans' struct, K and chain step, and the Horner threshold per
target. Tests: `test.cpp`; bench: `bench.cpp`. Attempts: `notes/` (one file per issue round). The
frontend's `detail::composition::taylor_shift` (`include/fps/composition.hpp`) is the plain version
this replaces (issue #26).

## Design

- b_k = (1 / k!) sum_(i >= k) A_i E_(i-k), A_i = f_i i!, E_j = c^j / j!. One cyclic product of
  length L = 2^lg >= 2n (lc-opt's layout): A at [0, n), E_j at L/2 - j, so b_k k! is coefficient
  L/2 + k, the upper half in natural order, and both factors lie in the lower half. E_0 = 1 would
  sit at L/2, where the product does not read: it is left out and comes back in the output as
  A_k = f_k k! (lc-opt fixed it up after its radix-8 pass instead, which needs ntt internals).
- Product: ntt's in place (`SelectedMultiply::multiply(a, b, n, L/2, lg)`, `ntt::Convolution`'s
  path and lc-opt's: both first levels in place on the half-zero factors, both factors' subtrees
  transformed, multiplied and inverted in cache, the scale in the last level). The weights write A
  and E' straight into its arrays a and b; h = a + L/2 afterwards.
- Scans (lc-opt's chains in lanes): W = K S::lanes lanes on `factorials::Chains::lanes` (lane s on
  [a_s, a_s + C), a_s = min(h + s C, n - C), with a_s! and 1 / (a_s + C - 1)! from the factorials
  kernel's lane products and prefix products: no table of (1024 k)!). Lanes past n - C repeat the
  last C positions, so nothing reads or writes past n: f and out need no padding.
  - weights (h = 1): up chain i! 2^32, A_i = montgomery(f_i, .); down chain c^j / j! with
    multipliers j c^-1 (Montgomery, an arithmetic progression mod P), from c^e / e! at
    e = a_s + C - 1, stored at L/2 - j. Lanes from position 1 put the E stores on vector boundaries
    (lc-opt's E lanes run over (s C, (s + 1) C] for the same reason).
  - output: out_k = montgomery(h_k, z_k) + f_k, z = 2^32 / k! down each lane (multipliers k). If
    out overlaps f, the weights also store A to a copy and the output is montgomery(h_k + A_k, z_k)
    (a lane past t - C reads positions another lane has written).
  - The i! and 1 / k! chains step by Montgomery or Shoup products (`factorials::Step`, the
    factorials kernel's trick: the multiplier plain, its quotient a sum of two floors); c^j / j! is
    always Montgomery (its multipliers wrap mod P).
  - Every S::lanes steps an S::lanes x S::lanes transpose (`factorials::transpose`); the chains of
    the next block run interleaved with the stores of the current one.
- n <= kSmall (88): Horner's rule in vectors. c = 0: a copy. t = out.size() < n: the output on a
  grid of t positions (its own lane starts), or scalar below the lanes' minimum.
- Memory: in ntt's per-thread region, a and b (L + 16 words each) and A's copy (n); the product's
  twiddle tables are ntt's.

## Measurements

Round 1 (issue #124), GCC 15.2 in the pinned image, `tools/kernel.py` flags, 3 rounds. ns per
coefficient, n = 2^lg, t = n: `shift/<lg>` / `shift@transform` (the product alone) /
`shift@plain` (the frontend's algorithm: `factorials::tables`, scalar weights,
`ntt::middle_product`, fresh vectors per call).

| lg | Zen 3 `lc-bench` | Emerald Rapids `lc-intel` | Zen 5 `lc-zen5` | Neoverse V2 `lc-arm` |
|---:|---|---|---|---|
| 8 | 6.68 / 3.98 / 11.86 | 5.63 / 3.29 / 9.84 | 3.87 / 1.95 / 7.58 | 8.95 / 6.11 / 13.69 |
| 10 | 6.01 / 4.41 / 11.31 | 5.30 / 3.89 / 10.01 | 3.25 / 2.15 / 7.17 | 9.12 / 7.13 / 14.02 |
| 14 | 7.40 / 6.06 / 12.85 | 6.34 / 5.36 / 11.13 | 3.40 / 2.79 / 7.54 | 11.21 / 9.50 / 16.14 |
| 18 | 9.31 / 7.82 / 14.89 | 8.58 / 7.17 / 14.08 | 4.21 / 3.50 / 8.43 | 14.02 / 12.14 / 19.15 |
| 20 | 10.09 / 8.65 / 17.00 | 9.54 / 8.08 / 14.81 | 4.58 / 3.87 / 9.48 | 15.51 / 13.46 / 20.95 |
| 22 | 11.44 / 9.98 / 28.58 | 10.97 / 8.90 / 25.61 | 5.78 / 4.96 / 15.11 | 17.16 / 14.99 / 29.03 |

shift / plain: 0.40-0.63 (Zen 3), 0.43-0.64 (Emerald Rapids), 0.38-0.51 (Zen 5), 0.59-0.74
(Neoverse V2). shift / product: 1.14-1.23 from 2^14 on (scans, setup, the output), 1.3-2.0 at
2^8-2^10.

Against lc-opt on Zen 3, same binary, outputs equal (`notes/124-r1.md`): fps / lc-opt 0.994 at
n = 2^19, 1.005 at 400000, 1.006 at 2^17, 0.998 at 100000, 1.005 at 2^15; 0.31-0.93 below 2^14;
0.52 at 2^16 and 2^18, where lc-opt's even-lg layout doubles its length.

## Open

- The product at 2^17-2^19 on Zen 3 from spans (issue #177); the in-place product used here does
  not have it.
- Fold the output into the product's top level (lc-opt's note: its halves' top groups into the
  output pass, guess 0.1 ms at 2^19), or the weights into its first level.
- t much smaller than n: a product of n + t - 1 words (dense layout) instead of 2n.
- n > 2^24 (L = 2^25 is the NTT's longest).
- Repeated shifts of one length: with A carrying c^i, E'' = 1 / j! does not depend on c, and its
  leaf form could be kept (one forward transform less per call): a plan object.

## Sources

- lc-opt's polynomial_taylor_shift (`solution.cpp`, notes.md): the layout, the lanes, the chains.
- The convolution form of sum_i a_i C(i, k) c^(i-k): standard.
- Montgomery, "Modular multiplication without trial division", Math. Comp. 44 (1985); Shoup's
  multiplication by a constant with a precomputed quotient (NTL; as `backend/modarith`).
