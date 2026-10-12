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
  `ntt::detail::intrinsics::Values`, `Narrow` gives ymm on AVX-512), L of 1, 4 or 8 lanes.
  - Graeffe: c(z) c(-z) mod (z^8 - t), 20 products per leaf, lc-opt's sums (`graeffe_sums`).
  - Projection: the odd coefficients of a(z) c(-z) mod (z^8 - t), 32 products per leaf.
  - x86 and Scalar: 64-bit sums of products of canonical words (negative terms offset by 4 P^2,
    below 8 P^2), then one Montgomery reduction per output (x86: even words, then odd words
    shifted down, as lc-opt). Neon: one Montgomery product per term (`modarith`), summed.
  - CRT: (A + B) / 2 + u^4 (A - B) / (2 s); the 2^-32 of the sums and the 1/2 are in the
    constants (Montgomery products with 2^64 / 2 and 2^64 / (2 s), a table per pair).
  - compose's backward step: the leaf form of P^(u) at 2m spread to that of P^(z^2) at 4m
    (lo +- s hi at the even coefficients), then `Transform::inverse_product` with the stored
    leaf form of Q_s(-z), upper half out.
- Level T - 1 one-dimensional: Q_(T-1) = 1 + x q(y). compose: a cyclic product of length m of
  f reversed with q, upper half, negated; power projection: a = p1 - p0 q (`ntt::multiply`).
- Memory: one region per thread kept across calls (`ntt::detail::map_region`), carved into
  64-byte aligned arrays 80 words apart; compose stores T - 1 leaf forms of 4m words (as lc-opt).
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

## Next

- lc-opt's pruned transforms (one vector bit of the index is zero in the input or unused in
  the output; the y levels skip half the columns) for the forward of Q_s and of P, the inverse
  of V and of R: lc-opt 1445 -> 1332 us at m = 8192.
- The Graeffe and the spread inside the transforms' bottom levels (lc-opt's `LevelBottom`,
  `CompositionBottom`): needs a bottom hook in `ntt::Transform`.
- One-dimensional levels 0, 1, T - 3, T - 2 (lc-opt: 1342 -> 1150 us at m = 8192 with 1 and
  T - 3); power projection's doubling (lc-opt's `Doubling`, the first half of the next level's
  transform without its y levels).
- Neon leaf sums in 64 bits (umull/umlal) instead of a Montgomery product per term.
- Memory: compose keeps T - 1 leaf forms of 4m words (1.5 GiB at n = 2^22); recomputing them from
  the coefficients (m + L words a level) costs one forward of 4m per level.
