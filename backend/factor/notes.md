# backend/factor

Factorization of polynomials modulo P = 998244353 into monic irreducibles with multiplicities. API:
`factor.hpp` (namespace `fps::factorization`, since `fps::factor` is the frontend's function:
`factor(f)`, `factor_as<S>(f)`, `Factor`, `kMaxDegree`). Tests: `test.cpp`; bench: `bench.cpp`.
Attempts: `notes/` (one file per issue round). From lc-opt's
`problems/polynomial/factorization_of_polynomials` (its issue #83, its `notes.md`), rewritten as
templates on `backend/simd` structs, so every target runs it, for the one modulus P. One variant:
no `select.hpp`; `-DFPS_FACTOR_SMALL=0` turns off the ymm register path for moduli of degree <= 8.

## Design

- Files: `kernels.hpp` (Montgomery arithmetic for P, 64-bit-lane sums per struct family, the dense
  kernels: `combine`, `combine_rows`, `lincomb`, `axpy`, `times_x`, `times_x_chain`, `multiply`,
  `transpose_rows`), `poly.hpp` (pooled polynomials, `Ring`: gcd, division, products modulo g by a
  table, powers, Cipolla's square root), `linear.hpp` (fraction-free blocked elimination,
  kernel vectors), `factor.hpp` (the algorithm, the API).
- Algorithm: lc-opt's (Yun; x^P by squarings with a reduction table; linear factors by root
  finding; Berlekamp with the kernel of (Q - I)^T; the factors grouped by the roots of the minimal
  polynomial of a random kernel element), except the root finding (below). `factor.hpp`'s header
  has the steps.
- Words: Montgomery residues (x 2^32 mod P) everywhere inside, canonical. Sums of products in
  64-bit lanes: x86 even and odd words by `mul_even` (`EvenOdd`), Neon `umlal`/`umlal2` on halves
  (`Halves`), scalar `u64` (`Single`). For P a fold s - (s >> 32)(2^32 - 2^32 mod P) every 17
  products (lc-opt's generic p: 14), then fold and Montgomery's step.
- `combine_rows` (new against lc-opt): several output rows at once, R rows x 1 vector of sums in
  registers, so each column vector loaded serves R rows: R = 4 on 16-register structs and Neon,
  8 on AVX-512 (variants measured, notes/123-r1.md). Krylov: the first 8 rows q_j by mat-vecs with
  the multiplication matrix X of x^P, then each block of 8 is the block before times the
  multiplication matrix of q_8 (q_(j+8) = q_j q_8 mod g). The powers of v in the split alike.
- Elimination: lc-opt's fraction-free blocks of 16 columns, but pivot rows are swapped to the
  front of the trailing rows, so the trailing rows stay contiguous and a block's update is one
  `combine_rows`; its coefficients are the panel columns scaled in place (one vector product per
  word; `combine_rows` reads coefficients with a step along s), not a scalar loop.
- `times_x_chain` (multiplication matrices, the reduction table): each step shifts the column
  before in registers (`vperm2i128` + `vpalignr`, `valignd`, `ext`, `palignr`) from aligned loads,
  takes its top word from the register that held it (broadcast by `permute`), and computes that
  vector first: a copy and an unaligned load waited on stores (store forwarding fails) and cost 46
  ns per column at m = 100 on Zen 3; now 37 ns (it is throughput-bound: about 27 instructions per
  vector of 8 words).
- `axpy` (x + c y, in long division and `times_x`): the sums start as x 2^32 (a shift and a mask
  on x86) instead of a product by 1.
- Root finding (new against lc-opt's Cantor-Zassenhaus): P - 1 = 119 2^23, so for a shift a,
  w = (x + a)^((P-1)/2^k) mod Q takes at each root a 2^k-th root of unity w_k^e (w_k = 3^((P-1)/
  2^k)). Roots sharing the low j bits of e (`low`) split by bit j: gcd(Q, w^(2^(k-j-1)) - w_(j+1)^
  low) holds those with bit j = 0, k - j - 1 squarings modulo Q, fewer at each level (C-Z: about 29
  squarings at every node, and a new shift at each node that does not split). k is the least with
  2^k >= 4 d^2 (at most 23), so values collide with probability below 1/8; collided roots (or the
  root -a, whose value is 0) start again with a new shift. For the linear factors a = 0 and
  w = x^((P-1)/2^k) mod L comes from the squarings to x^((P-1)/2) (kept for j = K .. 1). Degree 2:
  the quadratic formula (Cipolla). find_roots of 44 linear factors 65 -> 48 us on Zen 3.
- Memory: polynomials take buffers from a `Pool` (zeroed on return, only the words below n), the
  m x m arrays reuse two vectors (lc-opt's: page faults cost 1.6 us per 4 KiB page on lc-bench).
  Up to `kKeepDegree` = 256 each thread keeps its factorizer between calls (`thread_local`):
  a fresh one per call cost random16 10.1 -> about 12 us (17 more polynomial buffers when the
  powers x^((P-1)/2^j) came in).
- Sorting: the result is sorted (degree, then coefficients from b_0), so every target and every
  random choice give the same list.
- GCC 15.2 miscompiles 64-bit sums on `Scalar` at -O2 -march=x86-64-v3 (#183): the column loops of
  `combine` and `combine_rows` carry `#pragma GCC novector` on `Single` only (on Neon it cost 15%).

## Limits

- deg f <= kMaxDegree = 4096 (std::length_error above): Berlekamp's two m x m matrices are 2 x 64
  MiB there, and the O(m^3) parts take seconds (notes/123-r1.md: times to 4096).
- Degree 2^20: out of reach for this algorithm (matrices of 2^40 words: 4 TiB; m^3 = 1.2e18
  products, years). It would want the distinct-degree split by baby steps and giant steps
  (Kaltofen and Shoup) with modular composition on NTT products, then equal-degree splits, on
  backend/division and backend/gcd (half-gcd). A guess of its cost at 2^20: about sqrt(n) = 1024
  modular compositions of degree 2^20, each about 1024 NTT products of length 2^21 (Brent-Kung),
  so about 10^6 products of ~10 ms each: hours. Not done; no caller needs it yet (#201, backlog).

## Measurements

notes/123-r1.md: same-run against lc-opt on lc-bench (Zen 3), phases, kernel variants, per-target
benches, large degrees, the frontend's `fps::factor`.

## Sources

- E. R. Berlekamp, "Factoring polynomials over large finite fields", Math. Comp. 24 (1970).
- D. G. Cantor, H. Zassenhaus, "A new algorithm for factoring polynomials over finite fields",
  Math. Comp. 36 (1981); the 2-power character splitting is the same idea with the 2^k-th roots
  of unity of F_P in place of +-1 (as in root finding for primes with a large power of 2 in
  P - 1); written here from that idea, no code read.
- D. Y. Y. Yun, "On square-free decomposition algorithms" (1976).
- E. Kaltofen, V. Shoup, "Subquadratic-time factoring of polynomials over finite fields",
  Math. Comp. 67 (1998): the path for large degrees, not implemented.
- lc-opt's code (its `factor.hpp`, `kernel.hpp`, `field.hpp`, `poly.hpp`) moved here, rewritten.
