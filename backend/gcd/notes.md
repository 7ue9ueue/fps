# backend/gcd

Polynomial gcd and the inverse modulo a polynomial, modulo P = 998244353, by half-gcd jumps. API:
`gcd.hpp` (namespace `fps::half_gcd`, since `fps::gcd` is the frontend's function: `gcd(f, g,
out[, scratch])`, `inverse_mod(f, g, h[, scratch])`, `gcd_scratch`, `inverse_mod_scratch`).
Tests: `test.cpp`; bench: `bench.cpp`. Attempts: `notes/` (one file per issue round). From
lc-opt's `lib/poly/gcd.hpp` (its issues #82, #85; its `lib/poly/notes.md`, Gcd), rewritten on
`ntt::Transform`'s values form and `backend/division`, so every target runs it. One variant: no
`select.hpp` yet.

## Design

- Jump of k on (a, b), deg a = n > deg b: R = Q_h ... Q_1 with deg r_h >= n - k > deg r_(h+1); it
  depends only on a, b divided by x^(n - 2k) (Thull and Yap). Fraction-free steps (no inversion
  per step): R (a, b) = (c r_h, c' r_(h+1)).
- Recursion (lc-opt's): jump(k1) on the top 2 k1 + 1 coefficients, k1 = k / 2 for a power of two,
  else the largest power of two below k (lc-opt: 16.2 -> 13.9 ms at n = 50000 against ceil(k / 2)),
  apply R1 to (a, b) at degrees [s2, deg c] by cyclic products of length L = bit_ceil(n / 2) (the
  wrapped part from R1 times (a, b) mod x^(deg c - L + 1), at a shorter length when few words
  wrap), jump(k - progress) on the top of the result, combine R2 R1 by cyclic products of length
  bit_ceil(progress). A first jump with no progress: one division step (`division::divide`, or
  long division for quotients of at most 32 words). Base: Euclid for k <= 64 (kDirect).
- Kept transforms: the combine keeps each product's transform; the parent's transform of twice
  the length is that one plus the upper half from the entry mod x^L + 1 (`forward_upper`).
- Values form (new against lc-opt): every transform is in `ntt::Form::values`, so a product is one
  Montgomery product per word, not an 8 x 8 leaf product (0.95 ns per word on Zen 3), for a split
  after each forward and a join before each inverse (about 0.42 ns per word each on Zen 3).
  Doubling in the values form: `Transform::forward_upper(h, Form::values)` (added to
  `backend/ntt` for this). The matrices' transforms are kept times 2^32: a product of a matrix
  transform with a plain one (apply) is plain after one Montgomery reduction, a product of two
  (combine) stays times 2^32, and goes to the inverse after a second reduction. Sums of two
  products: 64-bit sums, one reduction (x86 `mul_even`, Neon `umull`/`umlal`).
- Euclid: only the coefficients from the floor n - 2k + progress (lc-opt's); a degree-1 quotient
  step is e = alpha^2 c - (alpha beta x + gamma) d over vectors, 3 products in 64-bit sums and one
  Montgomery reduction with factors times 2^32 (x86, Neon); Shoup products elsewhere. Rows of the
  matrix alike. Longer quotients: scalar pseudo-division.
- gcd: the spine of a full jump without the matrix: (a, b) -> (b, a mod b), then jump(k1) on the
  top, apply to the whole of (a, b), repeat while deg a > 64; then fraction-free steps on whole
  remainders, and one inversion for the monic result. About 0.92-0.95 of inverse_mod's time.
- inverse_mod: b = f mod g, the full jump of deg g on (g, b) computing R[0][1] only (and every
  entry's coefficient 0, `at0`): h = R[0][1] / c, c = R[0][0](0) g(0) + R[0][1](0) b(0).
- Memory: one span used as a stack (mark, release): this thread's NTT region or the caller's
  scratch. Bound 48 N words for N = bit_ceil(deg) (lc-opt's) plus the arrays' gaps per level;
  random inputs used 31-34 N from 2^12 on (`peak.cpp` in the exploration folder, issue 103). A
  division step inside a jump takes `division::divide`'s scratch from the stack, or from the heap
  if the stack is short. Transforms: series' per-thread `ntt::Transform` per length.
- Never `Transform::inverse(F, out)` here: it takes this thread's NTT region, which the kernel
  holds, so each call would map a fresh one. Inverses run in place on work arrays.

## Measurements

See `notes/103-r1.md`: against lc-opt in the same run (Zen 3), against the leaf form (every
target), the profile, and the frontend's plain Euclid.
