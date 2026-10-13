# backend/composition

Composition of power series modulo P = 998244353: `compose(f, g, out)`, `power_projection(g,
out)`, `compositional_inverse(f, out)` (API: `composition.hpp`). Tests: `test.cpp`; bench:
`bench.cpp`. Attempts: `notes/` (one file per issue round). From lc-opt's
`lib/poly/{composition,projection,compositional_inverse}.hpp` (its `lib/poly/notes.md`:
Composition, Power projection, Compositional inverse), rewritten on `ntt::Transform`'s public
leaf-form ops and `backend/simd`, so every target runs it. One variant per op: no `select.hpp`.

## Design

- Kinoshita and Li (arXiv 2404.05177) with lc-opt's layout: m = 2^T >= n (at least 64), level s
  stored at stride 2L (x = z, y = z^(2L), x below L = m / 2^s), products as transforms of length
  4m. Q_s(-z)'s leaf is Q_s(z)'s with its odd coefficients negated (leaf moduli z^8 - w are even
  in z), so the Graeffe step and the projection step are per leaf pair (moduli z^8 -+ s, s = r(p))
  and a CRT to leaf p of the length-2m leaf form (s^2 = w_p): no product transform.
- Leaf passes (`detail::Pairs<L>`): lanes are leaf pairs (8 x 8 transposes from
  `ntt::detail::intrinsics::Values`, `Narrow` gives ymm on AVX-512), L of 1, 4 or 8 lanes. Runs
  over a range of pairs (`*_run`), called from ntt::Transform's hooks (issue #149) on each run of
  leaves while it is in cache; runs of fewer pairs than lanes go one pair at a time.
  - Graeffe: c(z) c(-z) mod (z^8 - t), 20 products per leaf, lc-opt's sums (`graeffe_sums`).
  - Projection: the odd coefficients of a(z) c(-z) mod (z^8 - t), 32 products per leaf.
  - Level 1: G(q1), G(q2), E(q1, q2) (even part of q1(x) q2(-x), 32 products per leaf), and for
    power projection O(p1, q1), O(p1, q2).
  - compose's backward step (`product_run`, 1 or 2 terms): R's leaves at 4m from P^(u)'s leaf p
    at 2m (lo + u^4 hi) and Q_s's stored leaves: P^(z^2) mod (z^8 -+ s) = a(u) = lo +- s hi, and
    coefficient k of a(z^2) c(-z) is (-1)^k (a(z^2) c(z))_k: 32 products per leaf, where the
    spread leaf times a full leaf product took 64 and a pass of its own.
  - Every simd struct: 64-bit sums of products of canonical words (negative terms offset by
    multiples of 4 P^2), one Montgomery reduction per output. x86: even words, then odd words
    shifted down (lc-opt's); Neon (issue #149): umull/umull2 into two 64-bit halves per vector,
    the adds folded into umlal (was a Montgomery product per term); Scalar: u64.
  - CRT: (A + B) / 2 + u^4 (A - B) / (2 s); the 1/2 and 1/s are in the constants (Montgomery
    products with 2^32 / 2 and 2^32 / (2 s), a table per pair), the sums' 2^-32 stays: the hooked
    inverse (`inverse_hooked`) takes leaves times 2^-32.
- Pruned transforms (issue #149, `ntt::Transform`'s column bits): a row at stride 2L holds x < L,
  so bit L of the index is zero in Q_s's forward at 4m, and P^'s forward at 2m has bit L/2 zero;
  V's inverse at 2m uses x < L/2, R's at 4m x < L. The levels above the bit run on half the
  columns (the top level included).
- One-dimensional levels (lc-opt's; issue #149):
  - Level 0: Q_1 = 1 - y (g(x) + g(-x)) + y^2 g(x) g(-x): the Graeffe of g at 2m (in the forward's
    hook) and one inverse at m. compose: h = p1(x^2) - p0(x^2) g(-x), the forward of p0 at m and
    one inverse at 2m with the product hook (g's leaf form kept). Power projection: P_1 is a
    shift of g, no product.
  - Level 1 (m >= 128): Q_1 = 1 + y q1 + y^2 q2 at stride m; transforms of q1, q2 (and p1) at m,
    the level-1 hook in q2's forward, inverses at m/2; Q_2 (and P_2) assembled in level 2's
    layout. compose: R's rows 2, 3 = P_2 + P_1 q1(-x) + P_0 q2(-x) and P_3 + P_2 q1(-x) + P_1
    q2(-x), two inverses at m with the 2-term product hook.
  - Levels T - 3, T - 2 (m >= 256, columns L = 8, 4): Q = sum x^k q_k(y) as c_k = (-1)^k q_k,
    products in y of N = 2m/L words (cyclic: the one wrapped coefficient lands on y^0 = 0):
    c'_1 = c1^2 - 2 c2, c'_2 = c2^2 - 2 c1 c3 + 2 c4, c'_3 = c3^2 + 2 c1 c5 - 2 c2 c4 - 2 c6 (6
    products in 3 `inverse_product`s, the factors 2 in scaled leaf forms); compose's R columns
    r_c = sum_(2j+k=c) P_j c_k (up to 4 terms, upper half); power projection's p'_j =
    sum_(a+b=2j+1) p_a c_b (sums of 5 and 7 in two inverses, added). Columns in and out by 8 x 8
    transposes (`columns8_from`, `columns8_to`).
  - Level T - 1: Q_(T-1) = 1 + x q(y), c1 = -q. compose: a cyclic product of length m of f
    reversed with c1, upper half; power projection: a = p1 + p0 c1 (`ntt::multiply`).
- Memory: one region per thread kept across calls (`ntt::detail::map_region`), carved into
  64-byte aligned arrays 80 words apart (`Arena`, which throws std::logic_error past its size);
  compose stores the leaf forms of Q_0 (at 2m), q1, q2 (at m) and Q_2 .. Q_(T-4) (4m words each).
- Few coefficients: Horner (compose) and the powers of g (power projection) up to 16.
- compositional_inverse: power projection, then Lagrange with `series::log` and `series::exp`
  (lc-opt and the frontend: their `power`).

## Measurements

Same run against lc-opt (`lc-bench`, EPYC 7B13, GCC 15.2 in the pinned image, `-O2
-march=x86-64-v3`, one binary with lc-opt e6c6bcb's `lib/poly`, fps and the frontend, 3 rounds,
outputs checked equal at n = 33, 1000, 1024, 4097, 2^17), n = 2^(lg-1), us:

| op | lg | lc-opt | fps | frontend (#130) | fps / lc-opt |
|---|---:|---:|---:|---:|---:|
| `compose` | 6 | 16.4 | 4.6 | 9.1 | 0.28 |
| `compose` | 8 | 16.7 | 10.4 | 37.2 | 0.62 |
| `compose` | 11 | 91.0 | 128.6 | 486.8 | 1.41 |
| `compose` | 14 | 1121 | 1616 | 7227 | 1.44 |
| `compose` | 16 | 5801 | 8450 | 37593 | 1.46 |
| `compose` | 18 | 29498 | 42385 | 176465 | 1.44 |
| `power_projection` | 11 | 81.8 | 122.5 | - | 1.50 |
| `power_projection` | 14 | 990 | 1498 | - | 1.51 |
| `power_projection` | 18 | 25956 | 39714 | - | 1.53 |
| `compositional_inverse` | 11 | 113.2 | 158.0 | 486.5 | 1.39 |
| `compositional_inverse` | 14 | 1227 | 1786 | 6475 | 1.46 |
| `compositional_inverse` | 18 | 30648 | 45443 | 166963 | 1.48 |

One level at m = 2^17 on Zen 3, us (`issue105/compare_parts.cpp`): forward of Q_s at 4m 604,
Graeffe pass 280 (412 with a Montgomery product per term), inverse of V at 2m into q 330, next
level 17; forward of P^ at 2m 304, spread 80, `inverse_product` at 4m (upper half) 950;
projection pass 582 (932). lc-opt's generic level: 760 + 230 + 240 + 790 = 2020 (its notes).

`bench.cpp` per target (kernel.py, 3 rounds), in units of `mul@ntt/<lg>` from the same run
(n x n into n words), against the frontend's plain version (frontend/notes/26.md):

| case | Zen 3 | Emerald Rapids | Zen 5 | Neoverse V2 | frontend (same order) |
|---|---:|---:|---:|---:|---|
| `compose/14` | 34.9 | 35.8 | 40.9 | 32.7 | 132 / 127 / 131 / 122 |
| `compose/18` | 42.4 | 44.2 | 51.2 | 42.4 | 153 / 163 / 162 / 157 |
| `compositional_inverse/18` | 46.1 | 45.5 | 53.3 | 46.9 | 157 / 165 / 166 / 162 |

### Issue #149, round 1

Same run against lc-opt (`lc-bench`, as above; `issue149/compare.cpp`, outputs checked equal at
n = 33, 1000, 1024, 4097, 2^17), n = 2^(lg-1), median of 3 rounds, us:

| op | lg | lc-opt | fps | fps / lc-opt | #105 r1 |
|---|---:|---:|---:|---:|---:|
| `compose` | 6 | 16.1 | 4.1 | 0.25 | 0.28 |
| `compose` | 8 | 16.3 | 9.1 | 0.56 | 0.62 |
| `compose` | 11 | 89.5 | 91.7 | 1.02 | 1.41 |
| `compose` | 14 | 1107 | 1154 | 1.04 | 1.44 |
| `compose` | 16 | 5776 | 6144 | 1.06 | 1.46 |
| `compose` | 18 | 29222 | 31728 | 1.09 | 1.44 |
| `power_projection` | 11 | 81.8 | 90.4 | 1.11 | 1.50 |
| `power_projection` | 14 | 991 | 1122 | 1.13 | 1.51 |
| `power_projection` | 18 | 25841 | 30337 | 1.17 | 1.53 |
| `compositional_inverse` | 11 | 113.4 | 126.6 | 1.12 | 1.39 |
| `compositional_inverse` | 14 | 1228 | 1405 | 1.14 | 1.46 |
| `compositional_inverse` | 18 | 30405 | 36769 | 1.21 | 1.48 |

`bench.cpp` per target against #105 r1 (kernel.py, 11 rounds, new/base) and in units of
`mul@ntt/<lg>` of the same run:

| | Zen 3 | Emerald Rapids | Zen 5 | Neoverse V2 |
|---|---:|---:|---:|---:|
| geomean new/base | 0.786 | 0.805 | 0.808 | 0.679 |
| `compose/18` new/base | 0.749 | 0.727 | 0.766 | 0.684 |
| `power_projection/18` | 0.765 | 0.780 | 0.788 | 0.657 |
| `compositional_inverse/18` | 0.798 | 0.807 | 0.825 | 0.697 |
| `compose/14` units | 25.1 | 25.9 | 29.5 | 21.9 |
| `compose/18` units | 31.9 | 31.9 | 38.2 | 28.9 |
| `compositional_inverse/18` units | 36.6 | 36.9 | 43.0 | 32.5 |

compositional_inverse's tail (log and exp after the projection) is 6.4 ms of 36.8 at 2^17,
lc-opt's (its `power`) 4.6 of 30.4: issue #107.

## Next

- Doubling (lc-opt's `Doubling`, its power projection 23.2 -> 21.3 ms on the generic levels): the
  next level's transform of 4m from V's (and W's) leaves at 2m: the lower half by the levels
  within blocks of L words only (inverse, x < L/2 kept, forward; the y levels commute with the
  truncation and cancel), the upper half by `forward_upper` of the rows mod y^2Y - 1 with row 0
  turned into 2 - row 0. Saves the y levels and the top level of the lower half per array and
  level. Needs ntt ops on blocks of a leaf form (levels below a column bit).
- Memory: compose keeps T - 3 leaf forms of 4m words (1.6 GiB at n = 2^22 with the rest);
  recomputing them from the coefficients (m + L words a level) costs one forward of 4m per level.
