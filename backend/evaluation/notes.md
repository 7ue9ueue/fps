# backend/evaluation

Multipoint evaluation, interpolation and the Newton basis modulo P = 998244353. API:
`evaluation.hpp` (`evaluate`, `interpolate`, `to_newton_basis`, on spans). The point tree and
its passes: `tree.hpp` (`PointTree`, `Store`, `to_child`, `root_window`). Kernels on
`backend/simd` structs: `kernels.hpp` (middle products and cross sums in lanes, Horner's rule,
the per-base inversion, the direct paths `interpolate_small` and `newton_small`, transposes).
Choices per target: `select.hpp`. Tests: `test.cpp`, `test_units.cpp`, `test_small.cpp`,
`test_top.cpp`; bench: `bench.cpp`. From lc-opt's
`lib/poly/{evaluation,interpolation,newton}.hpp` (issue #104), rewritten on this repo's product
trees (`backend/product_tree`, with a `Keep` hook added) and `ntt::Transform` (with
`inverse_upper` added), so every target runs it.

## Design

- Points: m8 = 8 C (C >= kMinSlots = 17 slots, padded with zeros: factors 1 - 0 x = 1). Slot k
  of lane l is x[8 k + l]: point 8 k + l for evaluate and interpolate, point l C + k for the Newton
  basis (each lane a run of consecutive points, so every node covers consecutive points).
- Tree (`PointTree`): product_tree's `LaneTree` over the C slots (factor 1 - a x per lane, in
  Montgomery form; length L = bit_ceil(C)) with a `Store` as its Keep: every node above the base
  keeps its children's transforms at its own length (both children, or the right one only for
  the Newton basis), in rooms of one block found by the split point. Over the 8 lane products a
  fixed top tree (since #136 r2; product_tree's StandardTree before): pairs (lanes 2j, 2j + 1) of
  length T = 2L, two halves of 2T, the root of 4T = 8L, each keeping its children's transforms.
  Its leaves are kept as one lanes transform: the lane root's at T (16 L words; its product pass
  writes the lower half through LaneTree's `keep.root` hook, forward_upper of its coefficients
  the upper). The pairs' product transforms come from it by product_tree's `lane_pairs` (lanes
  leaves 8u + i are standard leaf u's 8 roots, ntt's values form order), each doubled to 2T in
  its room; the halves and the root by leaf products; the root also keeps T_L(Q) (one more output
  of its leaf products), which the division reuses when its length is L.
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
- Top tree to lanes (#136 r2): the 4 pairs' states (standard leaf forms at T) to the 8 leaves'
  products in the lanes in one pass (`pairs_to_lanes`: per standard leaf the twist by rho_u^k, an
  8-point transform to its 8 roots' values, the products with the siblings' lanes), then one
  halving of the lanes at T: the lane root's state, factor 8 (no leaf products, inverses of T or
  rows; before: each leaf's window into a row, rows_to_lanes and a lanes forward, notes/104-r1.md
  and 136-r1.md). A lane node's two children's products come from one read of its state
  (`pointwise_twice`); on x86 the halvings' combine is left to them (`kDeferredCombine`). At m8 =
  2^25 the lanes' transforms of 16 L words pass ntt's 2^25 and run twisted (`lanes_forward_upper`
  and the others: coefficient j of each lane times zeta^j, a transform of 8 L).
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
  R_v = R_l Q_r + R_r Q_l in the kept transforms (`pointwise_sums`; leaf products in the top;
  the pairs' from the lanes' sums at T by `lane_pair_sums`, lane_pairs with sums for products),
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
  those right children's states are forwards of them, no halving, scaled by 2 per level as the
  halvings scale the others (lc-opt's path). The leaves: `pairs_to_lanes<L, true>` (right leaves
  take the pair's values alone), then for C = L one halving of the lanes with a sign per lane
  (`combine_alternating`), else the lanes' inverse and each lane's window (`lane_windows`).
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
  the kept transforms (about 2 m8 words per lane level: 14.7 MB at m = 2^17; the top tree's
  leaves 2 m8 words), the trees, the division's and passes' scratch. Sizes are computed up
  front (`tree_words`, `pass_words`, `division_words`); the tests run with exact sizes and fail
  on a spill to the heap.
- evaluate splits f and the points into chunks of 2^24 (transforms stay within 2^25).

## Measurements

Bench (`bench.cpp`), n = m = 2^lg, GCC 16.2, medians of 11 paired rounds against main ff90ae6
(in brackets), ns per point, issue #136 round 2 (notes/136-r2.md; `lc-intel` noisy, its
newton/16 and /18 0.996 and 0.992 in a rerun):

| ns per point (new/base) | lc-bench (Zen 3, avx2) | lc-intel (Emerald Rapids, avx512) | lc-zen5 (Zen 5, zen5) | lc-arm (Neoverse V2, arm) |
|---|---:|---:|---:|---:|
| evaluate/6 | 14.0 (1.000) | 10.2 (1.012) | 9.2 (1.000) | 15.4 (0.999) |
| interpolate/6 | 91.6 (1.001) | 70.6 (1.002) | 57.2 (0.997) | 132.8 (1.000) |
| newton/6 | 13.8 (1.000) | 10.7 (1.008) | 8.2 (1.000) | 22.1 (0.999) |
| evaluate/8 | 37.6 (0.909) | 28.2 (1.011) | 16.4 (1.001) | 58.9 (1.000) |
| interpolate/8 | 54.1 (0.895) | 43.4 (0.878) | 34.1 (0.892) | 70.9 (0.896) |
| newton/8 | 40.0 (1.000) | 25.5 (1.004) | 19.6 (1.000) | 44.0 (0.940) |
| evaluate/10 | 40.7 (0.939) | 33.0 (0.948) | 23.2 (0.942) | 58.2 (0.937) |
| interpolate/10 | 60.3 (0.933) | 49.0 (0.943) | 35.9 (0.950) | 85.3 (0.931) |
| newton/10 | 37.6 (0.966) | 30.9 (0.966) | 21.4 (0.971) | 53.0 (0.965) |
| evaluate/12 | 47.9 (0.953) | 41.1 (0.960) | 25.4 (0.950) | 71.2 (0.947) |
| interpolate/12 | 71.2 (0.949) | 58.8 (0.948) | 39.1 (0.949) | 104.2 (0.942) |
| newton/12 | 44.4 (0.977) | 39.0 (0.990) | 23.5 (0.977) | 65.3 (0.971) |
| evaluate/14 | 58.2 (0.957) | 50.8 (0.948) | 29.5 (0.961) | 87.7 (0.951) |
| interpolate/14 | 85.9 (0.954) | 75.7 (0.956) | 45.1 (0.963) | 128.0 (0.946) |
| newton/14 | 54.3 (0.982) | 48.9 (0.985) | 27.4 (0.992) | 80.7 (0.975) |
| evaluate/16 | 70.0 (0.960) | 64.9 (0.969) | 33.9 (0.958) | 106.0 (0.951) |
| interpolate/16 | 103.3 (0.955) | 93.6 (0.979) | 51.6 (0.961) | 154.4 (0.947) |
| newton/16 | 65.6 (0.980) | 60.5 (1.011) | 31.7 (0.991) | 98.5 (0.975) |
| evaluate/17 | 77.0 (0.958) | 72.3 (0.979) | 36.5 (0.956) | 116.7 (0.954) |
| interpolate/17 | 113.2 (0.955) | 103.1 (0.962) | 55.5 (0.957) | 169.6 (0.948) |
| newton/17 | 72.3 (0.983) | 66.6 (0.991) | 34.1 (0.993) | 108.4 (0.977) |
| evaluate/18 | 87.5 (0.963) | 83.2 (0.973) | 44.0 (0.950) | 127.8 (0.950) |
| interpolate/18 | 126.2 (0.953) | 119.9 (0.975) | 65.3 (0.957) | 185.5 (0.947) |
| newton/18 | 80.6 (0.978) | 76.6 (1.020) | 38.1 (0.982) | 119.5 (0.977) |
| evaluate/20 | 102.1 (0.961) | 112.7 (0.928) | 51.5 (0.962) | 150.7 (0.957) |
| interpolate/20 | 148.1 (0.958) | 160.6 (0.936) | 76.0 (0.962) | 219.0 (0.953) |
| newton/20 | 94.8 (0.981) | 102.7 (0.988) | 46.5 (0.990) | 143.0 (0.980) |

At 6 the direct paths run (evaluate Horner; interpolate and the Newton basis in vectors since
#136 r1); evaluate/8 runs Horner but on Zen 3, where the cost model picks the tree. Against
lc-opt on `lc-bench` in the same run (7 rounds, its Arena reused across calls, outputs equal from
2^10 to 2^20, GCC 16.2), fps / lc-opt: evaluate 0.148 (2^10: lc-opt runs Horner there), 0.827
(2^12), 0.880, 0.887, 0.919 (2^17), 0.913, 0.921 (2^20); interpolate 0.857 (2^10), 0.871, 0.906,
0.914, 0.930, 0.925, 0.921; newton 0.872, 0.883, 0.914, 0.926, 0.937, 0.946, 0.952; geomean
0.905 without evaluate/10 (round 1: 0.943; before #136: 0.97-1.03). Earlier tables (the move,
#104; round 1): notes/104-r1.md, notes/136-r1.md and this file's history. Against the frontend's
plain `points.hpp` (#24's PR #119) at 2^17 on `lc-bench` (round 1): evaluate 60.3 -> 10.4 ms,
interpolate 87.0 -> 15.3, newton 38.8 -> 9.5.

## Log

- 2026-10-12, issue #104 round 1 (claude): the move. `notes/104-r1.md`.
- 2026-10-16, issue #136 round 1 (claude): the division on series::inverse and fused products,
  unit terms and loads in the bases, the direct paths in vectors, the cost model refit; 2-7%
  faster from 2^17 on every target, 2-11x below 100-400 points. `notes/136-r1.md`.
- 2026-10-18, issue #136 round 2 (claude): the top tree's leaves in the lanes (the lane root at
  twice its length; pairs_to_lanes, lane_pair_sums), a lane node's two products from one read of
  its state, the halving's combine in them on x86; 2-6% faster from 2^10 on every target (newton
  1-3%), fps / lc-opt 0.905. `notes/136-r2.md`.

## Sources

- lc-opt `lib/poly/{evaluation,interpolation,newton}.hpp` and `lib/poly/notes.md` (Multipoint
  evaluation, Interpolation, Newton basis), `problems/polynomial/{multipoint_evaluation,
  polynomial_interpolation,conversion_from_monomial_basis_to_newton_basis}/notes.md`.
- A. Bostan, G. Lecerf, É. Schost, Tellegen's principle into practice (ISSAC 2003): the
  transposed product tree.
- A. H. Karp, P. Markstein, High-precision division and square root (1997): the division's split.
