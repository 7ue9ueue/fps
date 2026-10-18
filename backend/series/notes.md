# backend/series

Power series modulo P = 998244353: inverse, log, exp, pow and sqrt into caller memory. API:
`series.hpp` (`inverse`, `log`, `exp` as `(f, out)`, `pow(f, e, c, out)`, `sqrt(f, c, out)`, in
this thread's memory or with a last `scratch` argument of `*_scratch(n)` words). Tests: `test.cpp`,
`test_pow_sqrt.cpp`, `test_ragged.cpp`, `test_inverse_blocks.cpp`; bench: `bench.cpp`,
`bench_pow_sqrt.cpp`, `bench_ragged.cpp` (ragged n). Attempts: `notes/` (one file per
issue round). From lc-opt's `lib/poly/{inverse,log,exp,pow,sqrt,calculus}.hpp` (its
`lib/poly/notes.md`: Inverse, Log, Exp, Power, Sqrt), rewritten
on `ntt::Transform`'s leaf form and `backend/simd`, so every target runs it (lc-opt's transform
layer is AVX2 asm on Zen 3). One variant per op: no `select.hpp` yet.

## Design

- Transforms: `ntt::Transform` per length, one per thread, kept (as `backend/product_tree`), and
  its fused ops (issue #112): `forward(in, F)` from a span (a half-zero source skips its zero
  half), `forward_product`, `forward_upper(in, H)`, `cyclic_product(in, B, work, out, {offset,
  negate})` (forward, leaf products with a kept transform and inverse in one depth-first pass, the
  lower or upper half out, negated by the top level's scale) and `inverse_product(terms, work,
  out, ...)` (an inverse of a sum of up to 8 leaf products). So the steps copy, zero and negate
  nothing (before: 4-6% of each op), and each product is one pass over memory, not three. Leaf v
  of a transform of 2L words sits at w_v = (-1)^v r(v / 2), which does not depend on the length,
  so the transform of L words is the lower half of that of 2L.
- Shifted sources are not needed: (x^(L/2) a) b mod (x^L - 1) is a b rotated by L/2, so a step's
  x^k e g mod (x^2k - 1) upper half is e g mod x^k (no wrap: degree < 2k - 1), from e in place.
- Upper halves (lc-opt's `forward_upper`): `Transform::forward_upper(in, H)` from a span (was a
  twisted copy and a transform of L, `detail::twist`).
- Memory: arrays carved from one span (`detail::Arena`): this thread's NTT region
  (`ntt::detail::take_region`, huge pages, kept across calls) or the caller's scratch. Each array
  is 64-byte aligned with 80 words after it (the transforms read 16 words of padding; the rest
  keeps arrays of a power-of-two length 320 bytes apart modulo 4 KiB). The fused ops take their
  work array from the arena.
- Divisions by indices (log's integral, exp's s): `detail::divide_by_index`, lc-opt's batch
  inversion (4 chains of lanes, 3 Montgomery products a word and one by the dividend), divisors
  below 2^13 from a table built once (the batch's lane totals cost about 0.8 us a call: exp and
  log of 2^10 words were 10-14% slower on Zen 5 without it). Before: a table of 1 / i per call
  from the factorial tables (0.9 ns per word on Zen 3) and a product pass.
- Recurrences for few coefficients: `dot` sums 16 products < P^2 in 64 bits, then reduces. Below
  32 (inverse) and 64 (log, exp) coefficients. inverse with the recurrence to 64 was slower on the
  Mac (1.07 against 0.69 us at n = 64).

### inverse

Newton k -> 2k with transforms of 2k (lc-opt's step): G = T(g mod x^k) (half-zero source), e =
(f g)[k, k + c) by a cyclic product of f with G (upper half out) into g[k, k + c), then
g[k, k + c) = -(e g mod x^c) by a cyclic product of e with G in place (negated). 5 transforms and
2 leaf products of 2k a step. 10 T(n) + 4 LP(n) for n = 2^j.

A last step from k to k + c with c <= k/2 ran at length k (`inverse_half_step`, 6 transforms and
3 leaf products of k) until issue #214 r2, which replaced it by blocks (Ragged n, below).

### log

lc-opt's blocked division: q = f' / f mod x^(n-1) in B blocks of k coefficients, k the least
power of two >= (n - 1) / 4 (at least 32; B = 2 .. 4), with h = 1 / f mod x^k and transforms of
2k. q_j = h (f' - f Q)[jk, (j+1)k) mod x^k (Q = q mod x^(jk)), the residual from one
inverse_product of the sum over i < j of T(W_(j-i)) T(q_i), W_t = f[(t-1)k, (t+1)k) (upper half
exact; on avx2 the sums of 2 and 3 products have asm bottoms of their own). (3 + 5B) transforms
and 2 + B + B(B-1)/2 leaf products of 2k: 23 and 12 for B = 4, 11.5 T(n) + 6 LP(n). Block j
reads f'[jk, (j+1)k) from f (a copy of f if out overlaps it). For B = 4, lc-opt's Toeplitz
sharing (issue #87 r2): block 2's residual and W_3 = W_3 Q_0 + W_2 Q_1 from one
`ntt::Transform::toeplitz_product` (3 leaf products), block 3's from W_1 Q_2 + W_3 (one product
and a term alone, one asm bottom on avx2): 11 leaf products, 5.5 LP(n). h goes to H's lower half
and log's q_j to Q_j's, so T(h) and T(q_j) run in place (`forward(in, F)` with in at F: lc-opt's
one-pass radix 8 on avx2); pow keeps q_j in q (exp reads it), so its T(q_j) stay out of place
(the two-pass radix 8 of ntt's notes/87-r2.md).

### exp

lc-opt's step from m to 2m with h = 1 / g mod x^(m/2), H = T_m(h), q = x f' (x g' = q g),
Q = q mod x^m:
1. G_lo = T_m(g); e = (g h)[m/2, m) by an inverse_product of G_lo H (upper half) into h[m/2, m);
   h[m/2, m) = -(e h mod x^(m/2)) by a cyclic product in place.
2. T_m(r) = forward_product of Q with G_lo, r = Q g mod (x^m - 1) = x g' + (Q g)[m, 2m).
3. H = T_2m(h) (kept for the next step); T_2m(r) = [T_m(r), forward_upper(r)];
   t = h r mod x^m = Q + h (Q g)[m, 2m) (inverse_product, lower half).
4. s = (f - log g)[m, 2m) = (q[m + i] + t_i - Q_i) / (m + i).
5. T_2m(g) = [G_lo, forward_upper(g)]; g[m, 2m) = s g mod x^m (cyclic product).
16 transforms and 7 leaf products of m a step (8 and 3.5 of 2m).

The last step, to n <= 2m (`exp_last_step`, lc-opt's exp_last_step), at length m in blocks of
B = m/2: block 2 as before (t = h0 r mod x^B, g_2 = g0 s), and for n > 3m/2 block 3: (q G)[3B,
4B) for G = g mod x^(3B) is r[B, 2B) - x G'[B, 2B) plus the upper half of T(Q) T(g_2) + T(q_2)
T(g mod x^m) (one inverse_product), times h0, then g_3 = g0 s_3. 14 transforms and 7 leaf
products of m against 16 and 7 for a full step (exp of 2^21 words: 1.08 -> 1.01 of lc-opt).
`exp_steps(q, c, g, ...)` runs the steps from any q (q[0] = 0) and g[0] = c (they do not depend
on the scale of g); `QTransforms` hands the last step T_m(q mod x^m) and T_m(q[m, 3m/2)) when the
caller has them (pow), which saves its forwards of them.

### pow

g = c (f / f[0])^e for residues e, c and f[0] != 0 (f^M = pow(f, M mod P, f[0]^M) for n <= P;
the frontend handles leading zeros). lc-opt's power: g solves x g' = q g with q = e x f' / f,
from log's blocked division (`divide()`, which log and pow share), then exp's steps from q
(`exp_steps`); no integral of f'/f, no derivative in exp, no passes for f / f[0], e and c.
Different from lc-opt: the division runs on N = e x f' (`index_block` with the factor e) over n
coefficients, not on f' over n - 1, so its blocks q_j are q's own, written in place in q (lc-opt
scales each block of f'/f by e and shifts it by one into q, a pass), and its kept transforms give
the last exp step's directly: the division's k is m/2 for m = bit_ceil(n) / 2 at every n >
kExpBase, T_m(q mod x^m) = T(q_0) + s T(q_1) with s = x^k = 1 on the lower half of the leaves
and -1 on the upper (one add/sub pass, `add_lower_sub_upper`), and T_m(q[m, 3m/2)) = T(q_2),
which exists (4 blocks) exactly when exp's block 3 runs. lc-opt gets them from T(f'/f) by a
product with x per leaf (`times_x`: a lane rotation and w_p a_7), and only when n - 1 is not a
power of two. Arrays: q (n words) and the Q_j first, then the division's other arrays, which
exp's reuse. Below kExpBase + 1 coefficients: i g_i = sum_j ((e + 1) j - i) a_j g_(i-j), a = f /
f[0] (`pow_direct`).

### sqrt

lc-opt's: Newton m -> 2m with h = -1 / g mod x^(m/2) and H = T_m(h) (the sign makes both updates
plain products): G = T_m(g); e = (g h)[m/2, m) (inverse_product, upper half) and h[m/2, m) = e h
mod x^(m/2) (cyclic product, in place); s = g^2 mod (x^m - 1) = f mod x^m + (g^2)[m, 2m) by an
inverse_product of G G, r = (s - f[0, m) - f[m, 2m)) / 2 (`half_residual`); H = T_2m(h) (also the
next step's); g[m, 2m) = h r mod x^m (cyclic product of length 2m). 11 transforms and 5 leaf
products of m a step. The last step, to n <= 2m, at length m: d = h r mod x^(n-m) for n - m <=
m/2 (4 transforms, 2 leaf products), else Karp and Markstein's split: d0 = h r0 mod x^(m/2), d1 =
h (r + g d0)[m/2, n - m) with (g d0)[m/2, m) the upper half of a cyclic product with G (8
transforms, 4 leaf products). Below kSqrtBase + 1 coefficients: 2 c g_i = f_i - sum g_j g_(i-j).
Arrays: G, H, R, h, then W (the fused ops' work array) last: an output 320 bytes past W (mod 4
KiB) made the last step's inverse_product of G G at 2^18 cost 590-685 us against 465-500 behind
it (Zen 3, notes/107-r1.md); `half_residual` halves by a shift and a product of the low bit by
(P + 1) / 2, with a fast path when f covers the block.

### Ragged n (issue #214, notes/214-r1.md, notes/214-r2.md)

n = L + r, L = bit_floor(n). inverse for r <= 5L/32 (L/4 on Zen 5 and below L = 2^14): the chain
(`inverse_chain`), steps whose lengths halve down from n so they land on it; a step to n' = L' +
r' at length L' has the power-of-two step's transforms and two products of r' words (the corners:
f_hi g_r and g_r e2 mod x^r'). To r = L/2: the blocked division with N = 1 (`inverse_blocks`,
`divide<true>`: h = 1 / f mod x^k into g, T(h) is Q_0, the blocks' products by h negated), 5
blocks of L/4 to L/4 and 3 of L/2 beyond: 3 blocks cost 2 transforms of L fewer than the steps to
L and the old half step, 5 blocks 2 transforms fewer and 2 leaf products more than the chain's
steps at r = L/4. Past L/2 Newton's steps (4 blocks cost about the step at 2L). exp,
sqrt and the division's last block (log, pow): products of a last step's c coefficients as
`low_product` (the NTT's ragged products, about 2c words) where c <= 9m/32 and its work fits, and to
16 coefficients their windows of full-length products by sums of products (`window()`), no
transform; the step before such a light last step runs as a last step too. The division takes 5
blocks of k/2 where 3 blocks of k would leave a short one.

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
- Against lc-opt in the same run after issue #87 r2 (Zen 3, n = 2^14 .. 2^21): inverse, exp, pow,
  sqrt 0.93-0.99, log 0.95-1.01 (notes/87-r2.md).
- pow and sqrt (`bench_pow_sqrt.cpp`, commit 934233f, 3 rounds), ratios to `multiply@ntt/<lg>`
  at lg 14 / 18 / 22: pow 4.59 / 4.33 / 4.46 (Zen 3), 4.57 / 4.28 / 4.44 (Emerald Rapids), 5.14 /
  4.82 / 4.34 (Zen 5), 4.24 / 4.25 / 4.24 (Neoverse V2); sqrt 1.62 / 1.48 / 1.57, 1.60 / 1.47 /
  1.54, 1.83 / 1.67 / 1.50, 1.53 / 1.50 / 1.51 (lc-opt's counts: about 4.6 and 1.6). Against
  lc-opt on Zen 3: pow 0.87-1.02, sqrt 0.92-0.99 (notes/107-r1.md).
