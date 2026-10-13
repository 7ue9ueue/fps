# backend/evaluation

Multipoint evaluation, interpolation and the Newton basis modulo P = 998244353. API:
`evaluation.hpp` (`evaluate`, `interpolate`, `to_newton_basis`, on spans). The point tree and
its passes: `tree.hpp` (`PointTree`, `Store`, `to_child`, `root_window`, `series_inverse`).
Kernels on `backend/simd` structs: `kernels.hpp` (middle products and cross sums in lanes,
Horner's rule, the per-base inversion, `standard_to_lanes`, transposes). Choices per target:
`select.hpp`. Tests: `test.cpp`; bench: `bench.cpp`. From lc-opt's
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
  leaf products), which the division reuses when its length is L.
- Windows (Bostan, Lecerf and Schost's transposed evaluation): with K = max(n, m8) and G =
  rev_K(f), f(a) = [x^(K-1)] G / (1 - a x). Node v of s_v points has W_v = (G / Q_v)[K - s_v, K);
  a child c with sibling d has W_c = (W_v Q_d)[s_d, s_v), exact in a cyclic product of length L
  >= s_v. A node's state is f_v times the transform of length L of its window right-aligned
  (terms below are garbage and stay below). `to_child`: P = state times the sibling's kept
  transform; a child of half the length gets P_lo - T(X mod (x^(L/2) + 1)) = 2 T(X_hi)
  (`inverse_upper` of P_hi and a forward of L / 2), factor doubled (tracked as u = 1 / f, applied
  where coefficients come out); a shorter child gets coefficients and a forward.
- Root (`root_window`): G / Q mod x^K by Karp and Markstein's division with k = ceil(K / 2):
  h = 1 / Q mod x^k (`series_inverse`: Newton with 5 transforms of 2i per step, the recurrence
  below 32), q0 = G h mod x^k, r = (G - Q q0)[k, K), q1 = h r mod x^(K - k), all cyclic of
  length bit_ceil(K). 7 transforms of that length with T_L(Q) from the tree.
- Top tree to lanes: each top leaf's window (an inverse of its parent's product) into a row,
  scaled by its u; the 8 rows transposed into the lanes layout, then one lanes forward: the lane
  root's state at factor 1. (The transform-domain conversion was tried: notes/104-r1.md.)
- Base (lane nodes of at most kLaneBase = 16 slots): the window's coefficients (an inverse of
  the parent's product), the products of the base's nodes by schoolbook (`BaseProducts`, nodes
  split at the middle, `multiply_lanes`), then middle products down to single slots
  (`middle_lanes`: 64-bit sums of at most 9 products, one Montgomery step).
- Interpolation: M = prod (x - a_i) from Q (M[j] = Q[m - j]); M'(a_i) by the descent from
  (rev(M') / Q mod x^m8); weights y_i / M'(a_i); then R = sum_i w_i Q / (1 - a_i x) up the tree,
  R_v = R_l Q_r + R_r Q_l in the kept transforms (`pointwise_sums`; leaf products in the top),
  doubled to the parent's length as product_tree's nodes; f = rev_m(R). The lane tree runs the
  descent and the sums in one depth-first pass (`interpolate_lanes`); each base inverts its
  values once (`divide_slots`: prefix products, one vector power) and builds its sums
  (`cross_lanes`) from the same products. A zero M'(a_i) (a repeated point) throws.
- Newton basis: F = rev_N(f) (N = m8), c_k = [x^(N-1-k)] F / Q_(k+1). Node v's right child's
  window is W_v[0, s_r) (from the state: `to_child` with e = L - s_l, halving when s_l = L / 2),
  its left child's (W_v Q_r)[s_r, s_v) (the product with the right sibling's kept transform).
  C is padded to the power of two above when past 0.65 of it (lc-opt's choice), so every right
  child halves.
- Direct paths (`select.hpp`): Horner's rule (`horner`: blocks of 4 or 8 vectors of points as
  chains, f in parts for short blocks) when its cost n m kHorner is below the tree's estimate
  kFixed + kPerPoint m8 lg m8 + kPerCoefficient K lg K, fitted per target; O(m^2) loops for
  interpolate up to 32 points and the Newton basis up to 64.
- Memory: one region of ntt's per-thread memory (kept across calls) as a product_tree Stack:
  the kept transforms (about 2 m8 words per lane level: 14.7 MB at m = 2^17), the trees, the
  division's and passes' scratch. Sizes are computed up front (`tree_words`, `pass_words`,
  `division_words`); the tests run with exact sizes and fail on a spill to the heap.
- evaluate splits f and the points into chunks of 2^24 (transforms stay within 2^25).

## Measurements

Bench (`bench.cpp`), n = m = 2^lg, GCC 15.2, medians of 7 rounds, ns per point, commit 9323462
(issue #104 round 1; the product_tree reorder after it changes no evaluation code path's work):

| ns per point | lc-bench (Zen 3, avx2) | lc-intel (Emerald Rapids, avx512) | lc-zen5 (Zen 5, zen5) | lc-arm (Neoverse V2, arm) |
|---|---:|---:|---:|---:|
| evaluate/6 | 14.6 | 10.3 | 9.2 | 15.4 |
| interpolate/6 | 244.2 | 204.9 | 153.4 | 329.4 |
| newton/6 | 121.6 | 92.5 | 97.1 | 142.1 |
| evaluate/8 | 54.4 | 27.5 | 16.4 | 58.7 |
| interpolate/8 | 71.5 | 57.1 | 45.5 | 94.4 |
| newton/8 | 44.6 | 36.4 | 28.5 | 58.9 |
| evaluate/10 | 47.9 | 38.0 | 27.3 | 70.7 |
| interpolate/10 | 71.0 | 56.8 | 42.8 | 104.9 |
| newton/10 | 43.3 | 35.0 | 25.1 | 64.3 |
| evaluate/12 | 54.9 | 45.8 | 29.4 | 84.6 |
| interpolate/12 | 81.7 | 64.4 | 46.3 | 125.5 |
| newton/12 | 50.0 | 40.2 | 27.1 | 77.9 |
| evaluate/14 | 65.8 | 54.9 | 34.5 | 103.2 |
| interpolate/14 | 96.9 | 81.1 | 53.4 | 152.6 |
| newton/14 | 60.4 | 50.5 | 31.6 | 95.3 |
| evaluate/16 | 77.9 | 66.8 | 39.9 | 125.8 |
| interpolate/16 | 114.4 | 97.9 | 60.9 | 185.7 |
| newton/16 | 71.9 | 61.7 | 36.7 | 116.7 |
| evaluate/17 | 84.6 | 74.7 | 42.6 | 137.9 |
| interpolate/17 | 124.3 | 105.8 | 65.6 | 202.8 |
| newton/17 | 78.4 | 68.9 | 39.3 | 128.0 |
| evaluate/18 | 95.5 | 86.0 | 50.9 | 150.4 |
| interpolate/18 | 139.2 | 121.3 | 75.9 | 220.6 |
| newton/18 | 86.5 | 76.6 | 43.8 | 140.7 |
| evaluate/20 | 111.5 | 119.1 | 58.9 | 177.7 |
| interpolate/20 | 162.4 | 166.3 | 86.9 | 260.0 |
| newton/20 | 102.8 | 101.4 | 52.7 | 168.2 |

At 6 and 8 the direct paths run where the cost model picks them (evaluate/6 and /8 Horner; at
2^6 the trees' fixed cost, 10-21 us, dominates interpolate and newton). Against lc-opt on
`lc-bench` in the same run (7 rounds, its Arena reused across calls), new/lc-opt: evaluate 0.18
(2^10: lc-opt runs Horner there), 0.97 (2^12), 0.99, 1.00, 1.00 (2^17), 1.01, 1.01 (2^20);
interpolate 1.02-1.03; newton 1.00 (2^10, 2^12), 1.02-1.03 above. Against the frontend's plain
`points.hpp` (#24's PR #119) at 2^17 on `lc-bench`: evaluate 60.3 -> 11.1 ms, interpolate 87.0
-> 16.3, newton 38.8 -> 10.3.

## Log

- 2026-10-12, issue #104 round 1 (claude): the move. `notes/104-r1.md`.

## Sources

- lc-opt `lib/poly/{evaluation,interpolation,newton}.hpp` and `lib/poly/notes.md` (Multipoint
  evaluation, Interpolation, Newton basis), `problems/polynomial/{multipoint_evaluation,
  polynomial_interpolation,conversion_from_monomial_basis_to_newton_basis}/notes.md`.
- A. Bostan, G. Lecerf, É. Schost, Tellegen's principle into practice (ISSAC 2003): the
  transposed product tree.
- A. H. Karp, P. Markstein, High-precision division and square root (1997): the division's split.
