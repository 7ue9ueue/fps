# backend/evaluation

Multipoint evaluation, interpolation and the Newton basis modulo P = 998244353. API:
`evaluation.hpp` (`evaluate`, `interpolate`, `to_newton_basis`, on spans). The point tree and
its passes: `tree.hpp` (`PointTree`, `Store`, `to_child`, `root_window`). Kernels on
`backend/simd` structs: `kernels.hpp` (middle products and cross sums in lanes, Horner's rule,
the per-base inversion, the direct paths `interpolate_small` and `newton_small`, transposes).
Choices per target: `select.hpp`. Tests: `test.cpp`, `test_units.cpp`, `test_small.cpp`; bench:
`bench.cpp`. From lc-opt's
`lib/poly/{evaluation,interpolation,newton}.hpp` (issue #104), rewritten on this repo's product
trees (`backend/product_tree`, with a `Keep` hook added) and `ntt::Transform` (with
`inverse_upper` added), so every target runs it.

## Design

- Points: m8 = 8 C (C >= kMinSlots = 17 slots, padded with zeros: factors 1 - 0 x = 1). Slot k
  of lane l is x[8 k + l]: point 8 k + l for evaluate and interpolate, point l C + k for the Newton
  basis (each lane a run of consecutive points, so every node covers consecutive points).
- Tree (`PointTree`): product_tree's `LaneTree` over the C slots (factor 1 - a x per lane, in
  Montgomery form) and `StandardTree` over the 8 lane products, each with a `Store` as its Keep:
  every node above the base keeps its children's transforms at its own length (both children,
  or the right one only for the Newton basis), in rooms of one block found by the split point.
  The top tree's root also keeps its product transform T_L(Q) (one more output of the root's
  leaf products), which the division reuses when its length is L. The top tree's leaves are the
  lane root's coefficients as 8 rows, by vector transposes (`lanes_to_rows`; strided loads before).
- Windows (Bostan, Lecerf and Schost's transposed evaluation): with K = max(n, m8) and G =
  rev_K(f), f(a) = [x^(K-1)] G / (1 - a x). Node v of s_v points has W_v = (G / Q_v)[K - s_v, K);
  a child c with sibling d has W_c = (W_v Q_d)[s_d, s_v), exact in a cyclic product of length L
  >= s_v. A node's state is f_v times the transform of length L of its window right-aligned
  (terms below are garbage and stay below). `to_child`: P = state times the sibling's kept
  transform; a child of half the length gets P_lo - T(X mod (x^(L/2) + 1)) = 2 T(X_hi)
  (`inverse_upper` of P_hi and a forward of L / 2), factor doubled (tracked as u = 1 / f, applied
  where coefficients come out); a shorter child gets coefficients and a forward.
- Root (`root_window`): G / Q mod x^K by Karp and Markstein's division with k = ceil(K / 2):
  h = 1 / Q mod x^k (`fps::series::inverse` in the stack, backend/series), q0 = G h mod x^k, r =
  (G - Q q0)[k, K), q1 = h r mod x^(K - k): a forward of h and three `cyclic_product`s of length
  L = bit_ceil(K) (sources of at most L / 2 words), all in the output d (G, then q0 and r, then
  q1); with the tree's T_L(Q) (times 2^32, taken out in r's pass), 7 transforms of L.
- Top tree to lanes: each top leaf's window (an inverse of its parent's product) into a row,
  scaled by its u; the 8 rows transposed into the lanes layout, then one lanes forward: the lane
  root's state at factor 1. (The transform-domain conversion was tried: notes/104-r1.md.)
- Base (lane nodes of at most kLaneBase = 16 slots): the window's coefficients (an inverse of
  the parent's product), the products of the base's nodes by schoolbook (`BaseProducts`, nodes
  split at the middle), then middle products down to single slots (`middle_lanes`: 64-bit sums
  of at most 9 products, one Montgomery step). Every product of factors 1 - a x has constant
  term 1 (kR), so the products' terms by it are added after the Montgomery step, not multiplied
  (`multiply_units`, in product_tree for the tree's own bases too; `middle_lanes<L, true>`,
  `cross_units`). x86 (8 lanes of EvenOdd sums): the unrolled middle products and cross sums
  load their operands per term (`middle_blocks`, blocks of 2 outputs; `cross_loads`) instead of
  holding them in arrays GCC spills.
- Interpolation: M = prod (x - a_i) from Q (M[j] = Q[m - j]); M'(a_i) by the descent from
  (rev(M') / Q mod x^m8); weights y_i / M'(a_i); then R = sum_i w_i Q / (1 - a_i x) up the tree,
  R_v = R_l Q_r + R_r Q_l in the kept transforms (`pointwise_sums`; leaf products in the top),
  doubled to the parent's length as product_tree's nodes; f = rev_m(R). The lane tree runs the
  descent and the sums in one depth-first pass (`interpolate_lanes`); each base inverts its
  values once (`divide_slots`: prefix products in 4 interleaved chains, one vector power, lazy)
  and builds its sums (`cross_units`) from the same products. A zero M'(a_i) (a repeated point)
  throws.
- Newton basis: F = rev_N(f) (N = m8), c_k = [x^(N-1-k)] F / Q_(k+1). Node v's right child's
  window is W_v[0, s_r) (from the state: `to_child` with e = L - s_l, halving when s_l = L / 2),
  its left child's (W_v Q_r)[s_r, s_v) (the product with the right sibling's kept transform).
  C is padded to the power of two above when past 0.65 of it (lc-opt's choice), so every right
  child halves. The top tree's rightmost windows are prefixes of the root's (known coefficients):
  those right children's states are forwards of them (factor 1), no halving, and the last lane's
  row is the root window's first C words (lc-opt's).
- Direct paths (`select.hpp`): Horner's rule (`horner`: blocks of 4 or 8 vectors of points as
  chains, f in parts for short blocks) when its cost n m kHorner is below the tree's estimate
  kFixed + kPerPoint m8 lg m8 + kPerCoefficient K lg K, fitted per target to a grid of n, m in
  8 .. 16384 (notes/136-r1.md). O(m^2) in vectors below kDirectInterpolate points (about 100):
  M = prod (x - a_i) (`product_of_linear`), M'(a_i) by `horner`, then f = sum w_i M / (x - a_i)
  by synthetic divisions, a vector of points per chain (`interpolate_small`). The Newton basis
  below kDirectNewton (180-400 by target): its n - 1 synthetic divisions in a systolic array of
  4 vectors of lanes (`newton_small`: division k in one lane, a lane shift per step, 16-64
  divisions per pass over the coefficients).
- Memory: one region of ntt's per-thread memory (kept across calls) as a product_tree Stack:
  the kept transforms (about 2 m8 words per lane level: 14.7 MB at m = 2^17), the top tree's
  leaf rows (m8 words), the trees, the division's and passes' scratch. Sizes are computed up
  front (`tree_words`, `pass_words`, `division_words`); the tests run with exact sizes and fail
  on a spill to the heap.
- evaluate splits f and the points into chunks of 2^24 (transforms stay within 2^25).

## Measurements

Bench (`bench.cpp`), n = m = 2^lg, GCC 15.2, medians of 11 paired rounds against main e88390e
(in brackets), ns per point, issue #136 round 1 (notes/136-r1.md):

| ns per point (new/base) | lc-bench (Zen 3, avx2) | lc-intel (Emerald Rapids, avx512) | lc-zen5 (Zen 5, zen5) | lc-arm (Neoverse V2, arm) |
|---|---:|---:|---:|---:|
| evaluate/6 | 14.6 (0.999) | 11.2 (0.986) | 9.1 (1.000) | 15.4 (1.000) |
| interpolate/6 | 93.1 (0.376) | 76.1 (0.343) | 56.7 (0.366) | 132.9 (0.415) |
| newton/6 | 14.0 (0.116) | 11.4 (0.118) | 8.2 (0.085) | 22.1 (0.156) |
| evaluate/8 | 43.0 (0.792) | 29.7 (1.006) | 16.4 (1.000) | 58.7 (1.000) |
| interpolate/8 | 63.5 (0.874) | 54.3 (0.863) | 39.2 (0.864) | 79.0 (0.857) |
| newton/8 | 39.1 (0.859) | 27.8 (0.692) | 19.6 (0.683) | 46.9 (0.817) |
| evaluate/10 | 43.5 (0.888) | 37.2 (0.916) | 25.1 (0.904) | 62.2 (0.924) |
| interpolate/10 | 65.7 (0.904) | 55.8 (0.915) | 38.6 (0.911) | 91.6 (0.914) |
| newton/10 | 39.2 (0.876) | 33.9 (0.887) | 22.7 (0.906) | 54.9 (0.901) |
| evaluate/12 | 49.9 (0.906) | 44.8 (0.897) | 27.0 (0.918) | 75.0 (0.948) |
| interpolate/12 | 75.0 (0.915) | 66.8 (0.940) | 41.6 (0.921) | 110.3 (0.935) |
| newton/12 | 45.3 (0.898) | 41.4 (0.946) | 24.5 (0.912) | 67.2 (0.928) |
| evaluate/14 | 60.4 (0.924) | 56.7 (0.948) | 31.0 (0.924) | 91.9 (0.962) |
| interpolate/14 | 89.7 (0.928) | 83.8 (0.972) | 47.4 (0.925) | 134.8 (0.948) |
| newton/14 | 55.1 (0.915) | 51.1 (0.934) | 27.9 (0.915) | 82.6 (0.943) |
| evaluate/16 | 72.4 (0.936) | 68.9 (0.942) | 35.6 (0.933) | 111.2 (0.968) |
| interpolate/16 | 107.0 (0.941) | 100.1 (0.956) | 54.2 (0.935) | 162.6 (0.956) |
| newton/16 | 66.4 (0.928) | 63.6 (0.944) | 32.2 (0.925) | 101.0 (0.951) |
| evaluate/17 | 79.0 (0.942) | 78.2 (0.946) | 38.2 (0.942) | 122.1 (0.969) |
| interpolate/17 | 116.4 (0.947) | 112.2 (0.968) | 58.0 (0.940) | 178.3 (0.960) |
| newton/17 | 72.7 (0.935) | 70.4 (0.954) | 34.6 (0.930) | 110.6 (0.955) |
| evaluate/18 | 90.6 (0.960) | 94.6 (0.980) | 46.2 (0.956) | 133.6 (0.974) |
| interpolate/18 | 131.0 (0.954) | 133.1 (0.970) | 68.5 (0.956) | 195.1 (0.965) |
| newton/18 | 81.4 (0.956) | 81.7 (0.950) | 38.4 (0.945) | 122.0 (0.958) |
| evaluate/20 | 105.6 (0.955) | 130.4 (0.973) | 54.0 (0.958) | 156.9 (0.976) |
| interpolate/20 | 153.4 (0.956) | 182.0 (0.974) | 79.2 (0.962) | 228.8 (0.968) |
| newton/20 | 96.3 (0.945) | 113.5 (0.960) | 47.2 (0.949) | 145.0 (0.959) |

At 6 the direct paths run (evaluate Horner; interpolate and the Newton basis in vectors since
#136, where the trees' fixed cost, 10-21 us, used to set the time); evaluate/8 runs Horner but on
Zen 3, where the refit cost model picks the tree. Against lc-opt on `lc-bench` in the same run (7
rounds, its Arena reused across calls, outputs equal from 2^10 to 2^20), fps / lc-opt: evaluate
0.158 (2^10: lc-opt runs Horner there), 0.886 (2^12), 0.917, 0.938, 0.945 (2^17), 0.965, 0.965
(2^20); interpolate 0.935 (2^10), 0.931, 0.950, 0.956, 0.960, 0.979, 0.972; newton 0.908, 0.906,
0.932, 0.944, 0.948, 0.971, 0.967; geomean 0.943 without evaluate/10 (before #136: 0.97-1.03).
Earlier tables (the move, #104): notes/104-r1.md and this file's history. Against the frontend's
plain `points.hpp` (#24's PR #119) at 2^17 on `lc-bench`: evaluate 60.3 -> 10.4 ms, interpolate
87.0 -> 15.3, newton 38.8 -> 9.5.

## Log

- 2026-10-12, issue #104 round 1 (claude): the move. `notes/104-r1.md`.
- 2026-10-16, issue #136 round 1 (claude): the division on series::inverse and fused products,
  unit terms and loads in the bases, the direct paths in vectors, the cost model refit; 2-7%
  faster from 2^17 on every target, 2-11x below 100-400 points. `notes/136-r1.md`.

## Sources

- lc-opt `lib/poly/{evaluation,interpolation,newton}.hpp` and `lib/poly/notes.md` (Multipoint
  evaluation, Interpolation, Newton basis), `problems/polynomial/{multipoint_evaluation,
  polynomial_interpolation,conversion_from_monomial_basis_to_newton_basis}/notes.md`.
- A. Bostan, G. Lecerf, É. Schost, Tellegen's principle into practice (ISSAC 2003): the
  transposed product tree.
- A. H. Karp, P. Markstein, High-precision division and square root (1997): the division's split.
