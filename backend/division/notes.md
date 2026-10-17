# backend/division

Division with remainder modulo P = 998244353: q and r with f = q g + r, deg r < deg g, into caller
memory. API: `division.hpp` (`divide(f, g, q, r[, scratch])`, `quotient_size`, `scratch_words`).
Cost model: `select.hpp` (a tuned table, "division.costs"), measured by `tune.hpp` / `tune.cpp`
(`tools/tune.py division`). Tests: `test.cpp`, `test_tune.cpp`; bench: `bench.cpp`. Attempts:
`notes/` (one file per issue round). From lc-opt's `lib/poly/division.hpp` (its issue #81; its
`lib/poly/notes.md`, Division), rewritten on `ntt::Transform`'s fused ops (issue #112) and
`series::inverse`, so every target runs it.

## Design

With d = deg g, k = n - d, F = rev(f), G = rev(g): Q = rev(q) = F / G mod x^k, r = f - q g mod x^d.

- Plan (`detail::plan`): long division when k d <= 2^12; else the cheapest of the blocked plans
  below by the cost model, long division included. The model: ns per word of forward(in, F),
  cyclic_product, inverse_product (1 term, and each further term), series::inverse
  (per coefficient at 2^lg), all per transform length on this target, and per-word costs of an
  element pass and of long division's terms (also the dot products' terms); small products by
  `small::model_ns`. The search: s = 32, 64, ... up to one block, each with 4 ways for the last
  block and 4 remainder methods.
- Blocks of s (a power of two >= 32) from the top of Q down: blocks 0 .. B-2 of s words, the last
  of c = k - (B-1)s in [1, s] (lc-opt padded F in front instead, so its first block carried up to
  s - 1 zeros). h = 1 / G mod x^s, H = T_2s(h); block j: R_j = (F - G Q)[js, js + s), Q_j = h R_j
  mod x^s by one cyclic_product of length 2s in place.
- Residuals. Tails (d < s): (G Q)[js + i] = (G T)[d + i] for T = Q[js - d, js); one cyclic_product
  of T with Gt = T_Lt(x^(Lt/2 - d) G), Lt = bit_ceil(2d): the words [Lt/2, Lt/2 + d) are (G T)[d,
  2d), so the upper-half output (negated) is the residual's head. Windows (d >= s): the sum over t
  <= min(j, d/s + 1) of (W_t Q_(j-t))[s, 2s), W_t = G[(t-1)s, (t+1)s), as one inverse_product of
  stored transforms (upper half, negated), at most 8 windows (ntt takes 8 terms since #143; before,
  past 4 terms the first ones went through leaf products in memory, multiply and multiply_add).
- The last block: full (cyclic_product at 2s), short (a length >= 2c - 1 below 2s, with its own
  T(h mod x^c)), small (small::multiply of h mod x^c and R), or dot: its residual by c dot products
  of g with Q (no transform, and no T(Q_(B-2))), then small::multiply. For k = 2^j + 1 (the
  divmod bench) the last block is one word.
- Remainder: wrap (q g mod (x^L - 1), L = bit_ceil(d), plus a fold of f), split (L/2 and a middle
  product from one T(q)), tail (one more tail product), direct (small::multiply), as lc-opt.
- Memory: one span (the thread's NTT region, or the caller's scratch): G, h, T(h), then
  series::inverse's scratch under the blocks' arrays; the remainder's arrays from the start.
  Transforms: series' per-thread ntt::Transform per length.

## Measurements

See `notes/102-r1.md`.
