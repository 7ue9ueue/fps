# backend/series

Power series modulo P = 998244353: inverse, log and exp into caller memory. API: `series.hpp`
(`inverse`, `log`, `exp`, each `(f, out)` in this thread's memory or `(f, out, scratch)` with
`*_scratch(n)` words). Tests: `test.cpp`; bench: `bench.cpp`. Attempts: `notes/` (one file per issue round). From lc-opt's
`lib/poly/{inverse,log,exp,calculus}.hpp` (its `lib/poly/notes.md`: Inverse, Log, Exp), rewritten
on `ntt::Transform`'s leaf form and `backend/simd`, so every target runs it (lc-opt's transform
layer is AVX2 asm on Zen 3). One variant per op: no `select.hpp` yet.

## Design

- Transforms: `ntt::Transform` per length, one per thread, kept (as `backend/product_tree`): its
  `forward`, `inverse` in place, `multiply` and `multiply_add` (leaf products), the variants the
  NTT's tuned table picks. Leaf v of a transform of 2L words sits at w_v = (-1)^v r(v / 2), which
  does not depend on the length, so the transform of L words is the lower half of that of 2L.
- Upper halves (lc-opt's `forward_upper`): for a of L words the upper half of T_2L(a) is T_L(a')
  with a'[8j + t] = a[8j + t] rho^j, rho = r(L / 16) (`detail::twist`; product_tree's notes give
  the argument). One Montgomery product per word against a second transform of L. Tested against
  the upper half of T_2L for L = 2^6 .. 2^20.
- Memory: arrays carved from one span (`detail::Arena`): this thread's NTT region
  (`ntt::detail::take_region`, huge pages, kept across calls) or the caller's scratch. Each array
  is 64-byte aligned with 80 words after it (the transforms read 16 words of padding; the rest
  keeps arrays of a power-of-two length 320 bytes apart modulo 4 KiB). `Transform::inverse(F,
  out)` is not used: it takes the thread's region, which a series call already holds, so it would
  map a fresh one per call.
- Reciprocals 1 / i (log's integral, exp's s): Montgomery form 2^32 / i from the factorial
  tables, (i - 1)! / i! 2^64 by two Montgomery products per word (`detail::reciprocals`).
- Recurrences for few coefficients: `dot` sums 16 products < P^2 in 64 bits, then reduces. Below
  32 (inverse) and 64 (log, exp) coefficients. inverse with the recurrence to 64 was slower on the
  Mac (1.07 against 0.69 us at n = 64).

### inverse

Newton k -> 2k with transforms of 2k (lc-opt's step): G = T(g mod x^k), E = T(-f mod x^2k),
E G inverse gives -(f g)[k, 2k) in its upper half; that half alone, transformed, times G,
inverse: g[k, 2k) in the upper half. 5 transforms and 2 leaf products of 2k a step, the sign in
the copy of f. 10 T(n) + 4 LP(n) for n = 2^j.

Last step from k to k + c with c <= k/2 (`inverse_half_step`), at length k: since g f = 1 mod
x^k, the cyclic product (g f_lo) mod (x^k - 1) is 1 + (g f_lo)[k, 2k); with f_hi = f[k, k + c),
g_c = g mod x^c, e = (g f_lo)[k, k + c) + (g_c f_hi) mod x^c from one inverse of E G + F G_c,
then g[k, k + c) = -(g_c e mod x^c). 7 transforms and 3 leaf products of k against 5 and 2 of 2k.
The two extra arrays share G's and E's (G_c at G + k: a transform reads its padding but never
writes it).

### log

lc-opt's blocked division: q = f' / f mod x^(n-1) in B blocks of k coefficients, k the least
power of two >= (n - 1) / 4 (at least 32; B = 2 .. 4), with h = 1 / f mod x^k and transforms of
2k. q_j = h (d - f Q)[jk, (j+1)k) mod x^k (d = f', Q = q mod x^(jk)), the residual from one
inverse of sum over i < j of T(W_(j-i)) T(q_i), W_t = f[(t-1)k, (t+1)k) (upper half exact).
(3 + 5B) transforms and 2 + B + B(B-1)/2 leaf products of 2k: 23 and 12 for B = 4, 11.5 T(n) +
6 LP(n). f is read into d, the W_t and h first, so out may overlap f. Not done: lc-opt's
Toeplitz sharing of blocks 2 and 3 (one leaf product of 2k fewer).

### exp

lc-opt's step from m to 2m with h = 1 / g mod x^(m/2), H = T_m(h), q = x f' (x g' = q g),
Q = q mod x^m:
1. G_lo = T_m(g); e = (g h)[m/2, m) from G_lo H; h[m/2, m) = -(x^(m/2) e h)[m/2, m), H again.
2. T_m(r) = T_m(Q) G_lo, r = Q g mod (x^m - 1) = x g' + (Q g)[m, 2m).
3. H = T_2m(h) (kept for the next step); T_2m(r) = [T_m(r), twisted upper half];
   t = h r mod x^m = Q + h (Q g)[m, 2m).
4. s = (f - log g)[m, 2m) = (q[m + i] + t_i - Q_i) / (m + i).
5. T_2m(g) = [G_lo, twisted upper half]; g[m, 2m) = (x^m s g)[m, 2m).
16 transforms and 7 leaf products of m a step (8 and 3.5 of 2m), with 2 twists.

A last step to n <= 3m/2 (`exp_half_step`, lc-opt's block 2 alone): t = h r mod x^(m/2) needs
h mod x^(m/2) only, and g[m, n) = (g mod x^(m/2)) s mod x^(n-m), whose transform T_m(g mod
x^(m/2)) the previous full step left in Gt. 7 transforms and 3 leaf products of m. Not done:
lc-opt's block 3 (n - m > m/2 at length m: 14 transforms and 7 leaf products of m against 16
and 7; about 6% of a power-of-two exp).

## Measurements

`kernel.py bench series`, GCC 15.2, each target's flags, commit a98c3be, 5 rounds, medians, n =
2^(lg-1); `multiply@ntt/<lg>`: `ntt::multiply` of two factors of n into n words (the frontend's
`mul@kernel` unit). Ratios to it, inverse / log / exp:

| lg | `lc-bench` (Zen 3) | `lc-intel` (Emerald Rapids) | `lc-zen5` (Zen 5) | `lc-arm` (Neoverse V2) |
|---:|---|---|---|---|
| 7 | 4.41 / 4.04 / 6.24 | 4.12 / 3.02 / 5.70 | 4.58 / 3.88 / 7.18 | 2.58 / 3.24 / 4.38 |
| 11 | 2.13 / 3.07 / 4.04 | 2.10 / 2.94 / 3.91 | 2.54 / 3.66 / 4.98 | 1.61 / 2.23 / 2.90 |
| 14 | 1.85 / 2.57 / 3.31 | 1.81 / 2.46 / 3.21 | 2.08 / 2.91 / 3.72 | 1.56 / 2.13 / 2.74 |
| 18 | 1.72 / 2.30 / 2.98 | 1.69 / 2.29 / 2.94 | 1.91 / 2.65 / 3.36 | 1.57 / 2.07 / 2.70 |
| 22 | 1.80 / 2.37 / 3.09 | 1.69 / 2.46 / 3.18 | 1.69 / 2.31 / 2.97 | 1.59 / 2.05 / 2.70 |

ns per coefficient at lg 22 (n = 2^21), inverse / log / exp: `lc-bench` 16.7 / 22.1 / 28.7,
`lc-intel` 16.8 / 24.5 / 31.6, `lc-zen5` 8.1 / 11.1 / 14.2, `lc-arm` 27.8 / 35.8 / 47.2.

- lc-opt's counts give about 1.7 / 2.1 / 2.7 with LP(n) = 0.8 T(n) (Zen 3, leaf products 0.95
  ns per word against transforms of 1.1-1.3 from 2^16): within 6-15% at 2^17 and 2^21. Short
  series cost more: at 2^10, leaf products (0.95 ns per word) are twice a transform's word
  (0.51), and the fused `multiply` the unit runs gains more there.
- Ragged sizes (`_ragged`, n = 1.25 2^(lg-1)): inverse and exp run their half-length last steps;
  ns per coefficient against the power of two at lg 18 on Zen 3: inverse 19.0 / 13.5, log 21.9 /
  18.1, exp 27.1 / 23.4. Before the inverse's half step: inverse_ragged/18 3.72 ms, after 3.12.
