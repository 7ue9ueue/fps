# backend/bostan_mori

Coefficient k of p / q modulo P = 998244353 (Bostan and Mori), for `fps::Fraction<P>::operator[]`.
API: `bostan_mori.hpp` (`coefficient(p, q, k)`; the variants `variant::schoolbook`,
`variant::values`, `variant::products`). `select.hpp`: the simd structs and the schoolbook cutoff
per target. Tests: `test.cpp`; bench: `bench.cpp`. Attempts: `notes/` (one file per issue round).
New work: lc-opt has no Bostan-Mori kernel and no kth-term problem.

## Design

- Bostan and Mori: with A = p(x) q(-x) = U_0(x^2) + x U_1(x^2) and q(x) q(-x) = V(x^2),
  [x^k] p / q = [x^(k/2)] U_(k mod 2) / V. p and q are cut to k + 1 words first and after each
  step (coefficient k reads no more), and p's zero top words are dropped.
- `values` (the NTT path): p and q in `ntt::Transform`'s values form of length L = 2N, N > deg q,
  2N > deg p + deg q (N >= 64). In the values form the words of w and -w are rows 2m and 2m + 1
  of a block (word 64g + 8i + t is leaf 8g + t at rho z^bitrev3(i), and z^4 = -1), and w^2 is the
  point of row m (leaves 2u) or m + 4 (leaves 2u + 1) of leaf u = 4g + t/2 of the values form of
  length N. So one pass over the 2N words gives the N values of U and V in the length-N layout:
  U(w^2) = A(w) + A(-w) (even k) or (A(w) - A(-w)) / w (odd k, a table of w^-1 per pair), with
  A(w) = p(w) q(-w), A(-w) = p(-w) q(w) as one 64-bit sum and one Montgomery reduction, and
  V(w^2) = q(w) q(-w). No transform of q(-x), no product transform: the pass reads 2N words of p
  and q and writes N of U and V, in place (output block h overwrites input block h, which output
  block h/2 read; block 0 goes through a copy). Rows of 8 words: ymm on AVX2 and AVX-512 (Avx512Vl),
  pairs of 4 on Neon and SSE, words on scalar; the even/odd lanes of two rows by `unzip` (vshufps +
  vpermq, uzp1/uzp2).
- Doubling: the next step needs U and V at the other N points. U mod (x^N - 1) = U (deg U < N), so
  the lower half of U's values form of length 2N is the pass's output; the upper half is
  `Transform::inverse(F, out, values)` of N into the upper half (the coefficients), then
  `forward_upper(h, values)` of 2N in place. A step: four transforms of N words (two inverses, two
  forward_upper) and the pass, where the frontend's two `ntt::multiply` run six transforms of 2N
  and two leaf-product passes.
- Scale: the pass leaves U times 2 2^-32 and V times 2^-32, so the ratio U / V, the only thing the
  answer needs, picks up 1/2 per step (a table of 2^-h). Nothing is rescaled per word.
- Last step (k becomes 0): [x^0] U / V = U(0) / V(0) = (sum of U's N values) / (sum of V's): no
  transform.
- The cut shortens the transforms: when the degree bounds after the cut (min(k, ...)) allow a
  smaller N, the step runs an inverse of N in place, zeros past the cut, and a forward of the new
  2N. When p and q fit `small` words, the values path hands off to the schoolbook steps (after an
  inverse of N in place).
- `schoolbook` (p and q of at most 256 words): U_0 = qe pe + (-qo) x po, U_1 = qe po + (-qo) pe,
  V = qe qe + (-qo) x qo with pe, po, qe, qo the even and odd halves, kept apart in zero-padded
  arrays. Per block of W output words, U and V together: each broadcast of qe[a] or -qo[a] feeds
  both sums (`backend/small`'s 64-bit lanes: `EvenOdd`, `Halves`, `Single`), only the a whose
  windows meet the block, a fold every 17 terms, one Montgomery reduction per word (U and V both
  times 2^-32: the ratio is kept). The halves of the outputs by `unzip` of two blocks.
- `tiny` (p and q of at most 4 words: Fibonacci and other order-2 recurrences): scalar sums by the
  definition (U[j] = sum s_i p_(2j + r - i), s = q(-x)), exact, one `% P` per word.
- `products`: the frontend's loop (two `ntt::multiply` per step into vectors): the fallback past
  the values form's 2^23 words, and the reference for large tests.
- Memory: one region per thread (huge pages, `ntt::detail::map_region`) for the two arrays of 2N
  words (+ 96), kept across calls; the w^-1 table (N words, built with Shoup products) per thread
  for the longest call (block g's entries do not depend on the length); `ntt::Transform`s per
  length per thread. `Transform::inverse(F, out)` takes the NTT's own region, so the arrays are not
  in it.

## Measurements

Same run against the frontend's `Fraction::operator[]` (main 1e7e7a2 plus this kernel, issue
114's exploration folder, `compare.cpp`: `fraction_at/<d>` as `frontend/bench_fraction.cpp`, p of
d and q of d + 1 random words, d = 2 Fibonacci, k = 10^18), GCC 15.2 in the pinned image, each
target's flags and timing machine, 7 rounds, median (IQRs within 1.5%); outputs equal.

| d | Zen 3 (avx2) | Emerald Rapids (avx512) | Zen 5 (zen5) | Neoverse V2 (arm) |
|---:|---|---|---|---|
| 2 | 6.61 -> 1.15 us (5.74x) | 4.51 -> 0.96 us (4.68x) | 3.92 -> 0.80 us (4.93x) | 5.97 -> 0.97 us (6.15x) |
| 16 | 12.2 -> 3.73 us (3.27x) | 8.85 -> 3.43 us (2.58x) | 6.50 -> 2.67 us (2.44x) | 16.4 -> 4.37 us (3.75x) |
| 64 | 52.3 -> 19.9 us (2.63x) | 39.5 -> 20.7 us (1.91x) | 21.1 -> 14.5 us (1.45x) | 63.5 -> 36.0 us (1.76x) |
| 1024 | 813 -> 498 us (1.63x) | 704 -> 525 us (1.34x) | 406 -> 306 us (1.33x) | 1205 -> 773 us (1.56x) |
| 16384 | 16.2 -> 9.07 ms (1.78x) | 13.6 -> 9.07 ms (1.50x) | 6.79 -> 4.92 ms (1.38x) | 23.2 -> 14.0 ms (1.66x) |
| 100000 | 95.6 -> 38.5 ms (2.49x) | 87.7 -> 39.6 ms (2.22x) | 40.6 -> 20.6 ms (1.97x) | 138 -> 59.4 ms (2.33x) |

Per step (60 steps), in units of `multiply@kernel/<lg>` of the same run (frontend -> kernel):
d = 1024 (lg 11) 2.87 -> 1.76, 2.80 -> 2.09, 2.90 -> 2.18, 2.76 -> 1.77; d = 10^5 (lg 18)
1.53 -> 0.62, 1.53 -> 0.69, 1.61 -> 0.82, 1.54 -> 0.66 (the scheme's four transforms of N are
about 2/3 of one product's six of N, plus the values conversions and the pass).

Parts of a step at N = 2^11 and 2^17 (`parts.cpp` in the exploration folder, ns, median of 3): the values
conversions (join before the inverse, split after forward_upper) against the leaf-form ops:

| | Zen 3 | Emerald Rapids | Zen 5 | Neoverse V2 |
|---|---:|---:|---:|---:|
| pass, even / odd k, 2^11 | 931 / 1379 | 807 / 1147 | 639 / 981 | 1459 / 2142 |
| inverse(F, out) values / leaves, 2^11 | 1986 / 1246 | 2207 / 1131 | 1186 / 552 | 3235 / 2014 |
| forward_upper values / leaves, 2^11 | 2204 / 1317 | 2393 / 1185 | 1297 / 513 | 3517 / 1912 |
| pass, even / odd k, 2^17 | 62186 / 91209 | 55649 / 78716 | 43149 / 66371 | 94282 / 137341 |
| inverse(F, out) values / leaves, 2^17 | 181938 / 141799 | 183813 / 123271 | 97540 / 62488 | 288098 / 214730 |
| forward_upper values / leaves, 2^17 | 196324 / 138673 | 194985 / 121482 | 99426 / 52015 | 303864 / 201207 |
| leaf_product of 2^12 / 2^18 words | 3861 / 254714 | 3278 / 210201 | 2415 / 156090 | 5459 / 350366 |

The values conversions cost 0.3-0.45 ns per word each on every machine; on Zen 5 and Emerald
Rapids, where the transforms are fastest, that is more than the transform itself at 2^11
(issue #98). A step at 2^11 on Zen 5: 5.8 us, of which 2.1 us transforms, 2.8 us conversions.

Schoolbook against values (`small.cpp` there, us, k = 10^18; values is flat per N: 2^6 for
d < 64, 2^7 below 128): crossovers at about d = 95 (Zen 3, Emerald Rapids), 85 (Zen 5), 75
(Neoverse V2), hence `kSchoolbookMax` 96, 96, 88, 76 (sse and scalar, not timed: 64).

## Next

- A leaf-form path: the pass on leaf pairs (U's even or odd words of p q(-x) mod x^8 - c, V's even
  words of q q(-x): 64 products per input leaf, then a CRT of leaves 2u, 2u + 1), with leaf-form
  inverses and forward_upper (no conversions). Estimated from the parts above with the pass at
  1.0-1.3 leaf products per word: 9-17% faster per step at 2^11 on Zen 5 and Emerald Rapids, 1-8%
  at 2^17, 0 to -6% on Zen 3. Faster values conversions in backend/ntt (#98) would help more.
- Odd steps' pass is 1.4-1.5x the even one (the w^-1 table read and one more Montgomery product).
