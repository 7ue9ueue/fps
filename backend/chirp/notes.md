# backend/chirp

Chirps modulo P = 998244353 and what is built on them: evaluation and interpolation on a
geometric sequence, and the shift of sampling points. From lc-opt's `lib/poly/chirp.hpp` and its
problems multipoint_evaluation_on_geometric_sequence, polynomial_interpolation_on_geometric_sequence
and shift_of_sampling_points_of_polynomial (issue #106), rewritten on `backend/simd`, `backend/ntt`'s
public products and `backend/factorials`, so every target runs them.

- `chirp.hpp`: `Terms<S, K>` (x_k = c s^k q^t(k), t(k) = k (k - 1) / 2, K S::lanes per step),
  `chirp`, `multiply_chirp`.
- `geometric.hpp`: `evaluate_geometric`, `interpolate_geometric`.
- `shift.hpp`: `extrapolate` (f(c + k) from f(0 .. n - 1)).
- `scan.hpp`: prefix products of a + b z^s or u0 + s and their inverses (`products`), pointwise
  products, per-thread scratch.
- `select.hpp`: the struct and K per target. Tests: `test.cpp`; bench: `bench.cpp`.
- Attempts, per issue and round: `notes/`.

## Design

- Terms (lc-opt's `detail::Chirp`): x_(k+W) = x_k g_k with g_k = s^W q^(W k + t(W)), W = K
  S::lanes, and g_(k+W) = g_k q^(W^2). g is kept times 2^32, so a step is one Montgomery product
  (x) and one Shoup product (g) per vector; terms in [0, 2P). Passing c 2^32 gives factors for
  `montgomery(f, x)` (multiply_chirp). The object holds words; a pass loads them into local
  vectors (stores may alias an object's arrays, and GCC's ASan stack frames misaligned an object's
  64-byte vector members: a SEGV in the avx512 ASan build of the first version).
- Evaluation, f of n coefficients at a r^i, i < m (chirp z-transform, Bluestein): with
  t(i - j) = t(i) + t(-j) - i j,
  f(a r^i) = r^t(i) sum_j A_j K_(i - j + n - 1), A_j = f_j (a r)^j r^t(j), K_v = r^-t(v - n + 1).
  The sum is `ntt::middle_product(K, A)` (K of n + m - 1 words): no reversal anywhere, and a = 0
  works (0^0 = 1). K_v = r^-t(-g) (r^g)^v (1/r)^t(v), g = n - 1, t(-g) = g (g + 1) / 2: three
  chirp passes (K, A, the output's r^t(i)). r = 0, a = 0 or n = 1: f(a), then f(0); n m <= 2^12:
  Horner at each point.
- Interpolation, Newton's form on x_i = a r^i (lc-opt's derivation, issue #78 there). With
  Q_k = (1 - r) .. (1 - r^k): d_k = a^-k r^-t(k) E_k, E = Y K mod x^n, Y_i = (-1)^i y_i / Q_i,
  K_j = r^t(j) / Q_j, and f_l = (-a)^-l / Q_l sum_k D_k K_(k - l), D_k = (-1)^k Q_k r^-t(k) E_k.
  Here the products A_k = prod_(s < k) (r - r^-s) = (-1)^k Q_k r^-t(k) and B = 1 / A = (-1)^k K_k
  carry Q, its inverse and the chirp r^t at once, and the signs cancel in pairs:
  E' = (y_i B_i (-1)^i r^-t(i)) B mod x^n (= (-1)^k E_k),
  H_l = sum_k A_k E'_k B_(k - l) (the product of (A E') reversed and B, reversed),
  f_l = (-1/a)^l r^-t(l) B_l H_l.
  Two `ntt::multiply` of n words mod x^n, two chirp passes, one pointwise pass, two reversals.
  Points not distinct (a = 0, r^k = 1 for some 0 < k < n, or r = 0 with n > 2) throw
  std::invalid_argument before out is written: A_(n-1) = 0 exactly then. n <= 32 or r = 0: divided
  differences in O(n^2).
- Shift of sampling points: f(x) = W(x) sum_i g_i / (x - i), W(x) = x (x - 1) .. (x - n + 1),
  g_i = f(i) (-1)^(n-1-i) / (i! (n - 1 - i)!) (`factorials::tables`). Outputs at samples (mod P)
  are copied; the others form runs x0 .. x0 + len - 1 inside [n, P) (at most two in practice),
  each one middle product of h_t = 1 / u_t, u_t = x0 - n + 1 + t, with g; W(x0 + k) = u_k ..
  u_(k+n-1). The prefix products of u and their inverses give both h (Pre_t / Pre_(t+1)) and W.
- Prefix products (`scan.hpp`): `products_lanes<S, K>` runs W = K S::lanes lanes, lane l on block
  [l C, (l + 1) C), C a multiple of S::lanes; every S::lanes steps `factorials::transpose` turns
  the tile of lane values into vectors of consecutive positions. Blocks then get the product of
  the blocks before them (one vector pass), [W C, n) runs as a scalar chain, and the inverses run
  down the same lanes from tops derived from one inverse and the block products. Factors in
  Montgomery form: a + b z^s as a 2^32 + (b z^s 2^32), z^s by a Shoup product per step; u0 + s by
  adding 2^32 mod P. `products_scalar`: 8 interleaved scalar chains (the fallback below 1024).
- Memory: one per-thread scratch region (`detail::Scratch`, ntt's huge-page mappings), grown as
  needed; the products run in ntt's own per-thread region.

## Measurements

See `notes/106-r1.md` for the first numbers per target.

## References

- Chirp z-transform: L. Bluestein, "A linear filtering approach to the computation of the discrete
  Fourier transform", 1968; L. Rabiner, R. Schafer, C. Rader, "The chirp z-transform algorithm",
  1969. The identity i j = t(i) + t(-j) - t(i - j) (no square root of r, no reversal) derived here.
- Interpolation on geometric sequences through Newton's form and the q-binomial theorem: derived
  in lc-opt (issue #78 there), no code read; the A, B form derived here.
