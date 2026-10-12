# backend/product_tree

Products of many polynomials modulo P = 998244353. API: `product_tree.hpp` (`size`, `product`;
both also over any span of items with a projection to their words). Trees: `tree.hpp`
(`LaneTree`, `StandardTree`, memory, tables). Kernels on `backend/simd` structs: `kernels.hpp`
(lane schoolbook, leaf products, pointwise products, twists, conversions). Choices per target:
`select.hpp`. Tests: `test.cpp`; bench: `bench.cpp`. From lc-opt's `lib/poly/product_tree.hpp` and
`problems/polynomial/product_of_polynomial_sequence` (issue #56), rewritten on the NTT's public
`Transform` and on simd templates, so every target runs it.

## Design

- Driver (`detail::product`): constants multiply into one number; one nonconstant factor is
  copied (times that number). The others: those of degree at most 1/64 of the sum of all degrees
  are small; with at least 16 of them (`kMinLaneFactors`) they go to the lanes, sorted by degree,
  largest first (a stable counting sort), and dealt in turn: slot k holds factors 8k .. 8k + 7, a
  missing one is 1. The lane tree multiplies the 8 lanes side by side; the standard tree
  multiplies the 8 lane products (read in place, every 8th word) and the other factors. The
  result leaves Montgomery form times the constant in one pass into the caller's out.
- Node lengths (both trees): a node covers leaves [lo, hi) and splits where the prefix sums of
  the leaves' degrees come closest to halves (lc-opt's split). Its length L is a power of two >=
  its degree (the wrap trick): from its children's transforms of length L its product p comes out
  as p mod (x^L - 1); if deg p = L, p[L] is the product of the leading coefficients (tracked per
  node, per lane in the lanes) and p[0] is corrected by it. A node hands its parent its transform
  at the parent's length; its lower half is the node's own product transform (kept: the product
  pass writes it and a copy, the inverse runs on the copy).
- Upper half of the parent's transform (transform doubling): ntt's leaf form puts leaf v of the
  transform of 2L words at the root w_v = (-1)^v r(v / 2), so the upper leaves are w_(L/8 + u) =
  r(L/16) w_u. ntt's Transform has no forward from group 1 (lc-opt's `forward_upper`), so the
  upper half is a forward of length L of a twisted copy: p mod (x^L + 1) with coefficient group j
  (8 words) times rho^j, rho = r(L/16) (`twiddles`). This works with one factor per 8 words because
  the transform acts lane-wise on the 8 words of a leaf: with p = sum_k x^k C_k(x^8), word k of
  leaf v is C_k(w_v). So the twisted transform's leaf u holds C_k(rho w_u), the upper leaf. Per
  node: one product pass, one inverse and one forward of length L, and a twist pass. A child
  shorter than half its parent extends the same way at 2L, 4L, ... from p (L + 1 coefficients).
- Lanes (`LaneTree`): word 8 j + l = coefficient j of lane l. A transform of 8 m words then holds
  lane l at w_v in word l of leaf v, so products are pointwise (Montgomery, one per word). Nodes
  up to degree 16 (`kLaneBase`) are multiplied out by a lane schoolbook in consecutive pairs
  (64-bit sums of at most 9 products, one Montgomery step, no fold; lc-opt's base was 32 with a
  fold and measured the same as 16), then transformed. Lane length >= 8 (64 words).
- Standard (`StandardTree`): leaf products mod (x^8 - w_v) (`leaf_products`: a window [w A, A]
  per leaf, 8 broadcast products in 64-bit sums, one Montgomery step). Nodes up to degree 32
  (`kStandardBase`) by `small::multiply` in pairs (each product of Montgomery forms carries one
  more 2^32: one pass at the end takes them out). Length >= 64.
- Montgomery form throughout (x 2^32 mod P): pointwise and leaf products need no correction.
- Memory: one region of ntt's per-thread memory (`ntt::detail::take_region`, kept across calls,
  huge pages), used as a stack (`Stack`; past it, blocks from the heap). Nothing of the number of
  factors goes to the heap: fresh pages cost about 1.7 us per 4 KiB page on Zen 3 (lc-opt's notes),
  and `product@standard/18` lost 6 ms to a `std::vector` of leaves (Log). Per thread, kept: ntt's
  `Transform` per length (tables of 2^lg bytes each, so up to 2x the longest's), the twiddles
  rho^j (2 words per lane coefficient of the longest) and the leaf weights w_v 2^32.

## Measurements

Bench (`bench.cpp`), GCC 15.2, medians of 5 rounds, ns per factor, commit 43a68fc (`lc-arm`: 3
rounds of cd70b24, the same trees with `kMinLaneFactors` 64 and no cases below 2^10).
`product/<lg>`: 2^lg factors x - r; `product_mixed`: degrees 0 .. 8, a quarter constants (sum
about 3.4 2^lg); `product_few/<lg>`: 8 factors of 2^lg / 8 or 2^lg / 8 + 1 coefficients.

| case | `lc-bench` (Zen 3, avx2) | `lc-intel` (Emerald Rapids, avx512) | `lc-zen5` (zen5) | `lc-arm` (Neoverse V2) |
|---|---:|---:|---:|---:|
| product/4 | 34.0 | 26.2 | 22.7 | |
| product/6 | 20.7 | 17.4 | 13.0 | |
| product/8 | 19.1 | 18.6 | 13.3 | |
| product/10 | 20.5 | 20.5 | 13.5 | 30.3 |
| product/14 | 28.4 | 27.3 | 16.9 | 44.5 |
| product/18 | 39.8 | 38.8 | 22.0 | 64.6 |
| product@standard/18 | 67.4 | 60.4 | 41.4 | 86.5 |
| product_mixed/18 | 175 | 156 | 93 | 281 |
| product_few/20 (ms) | 11.66 | 10.70 | 5.73 | 20.26 |

Against the frontend's former tree of `f * g` (main a34ce14), frontend `product/<lg>`, 7 rounds,
new/base at lg 10 / 14 / 18: `lc-bench` 0.309 / 0.309 / 0.322 (40.6 ns per factor at 2^18, was
126); `lc-intel` 0.395 / 0.364 / 0.361; `lc-zen5` 0.355 / 0.329 / 0.333; `lc-arm` (5 rounds,
cd70b24) 0.372 / 0.353 / 0.348.

Zen 3 (`lc-bench`), 2^18 linear factors, phases (`explore_profile.cpp`, Log): all 10.39 ms; lane
tree 6.87 (slot loads 0.87, base schoolbook about 0.4, base forwards 0.2, 11 levels of
pointwise + inverse + twist + forward 5.4); standard tree on the 8 lane products 2.54; the rest
about 0.9 (three passes over the factors with the counting sort; one pass alone, `size()`, 0.15;
the output pass 0.04). Per word: forward / inverse 0.36 (2^6) to 1.06 (2^18) ns, pointwise2
0.18, twist 0.17, leaf products 0.83.
lc-opt (issue #56): lane tree 14.4 ms and top tree 5.7 ms for 500000 factors on `lc-amd`, about
40 ns per factor; here 39.8 ns at 2^18 (not the same machine, input or count).

## Log

- 2026-10-12, issue #56 round 1 (claude): the move. Base main a34ce14. Explorations in
  `~/Documents/hpc_library-explore/product_tree/` (profile harness, A/B script, huge products).
  - First version: lanes and standard trees on `ntt::Transform`; the standard tree's upper half by
    a forward of length 2L (no twist). `lc-bench`: product/18 12.0 ms (45.8 ns per factor),
    product@standard/18 19.6, product_few/20 14.0. Profile: lane tree 6.95, standard tree 3.03,
    the rest 2.2 ms: vectors of 2^18 indices (`factors`, `small`, `order`, each 1 MB of fresh
    pages per call) and 8 passes over the factors.
  - Driver: three passes (constants and sums; a degree histogram; placement), the counting sort's
    arrays and the tree's prefix sums in the region: all 12.16 -> 10.92 ms.
  - Standard tree's upper half by the lanes' twist and a forward of length L instead of a forward
    of 2L (Design): standard tree 3.04 -> 2.54 ms; A/B 3 rounds: product@standard/18 25.3 ->
    23.5 ms, product_few/20 13.84 -> 11.50, product_few/16 0.690 -> 0.586.
  - The standard tree's leaves were a `std::vector<Leaf>` of every factor in standard mode:
    product@standard/18 23.5 -> 17.7 ms with them in the region (page faults of a 6 MB vector).
  - Tried, not kept:
    - Slot loads converting each word to Montgomery form in scalar code (instead of one vector
      pass over the slot, which reloads words just stored one by one): 3-5% slower (product/18
      10.39 -> 10.67 ms, product_mixed/18 46.6 -> 48.4).
    - The counting sort in 4 interleaved counters per degree (one counter chains its increments
      through memory): product/18 10.4 -> 11.5 ms. The order is no longer the factors' own, so
      slot loads lose locality; product_mixed unchanged.
    - The 8 lane products moved to 8 arrays in one pass before the standard tree, instead of
      loads of every 8th word at each leaf: geomean 1.0001 (5 rounds, cases at lg >= 10).
    - The counting sort's counts and positions for a run of one degree kept in registers (the
      increments of one counter in memory looked like a chain): geomean 1.0080, 5 rounds
      (product/10 1.016, product_mixed/14 1.016). Not the cost.
  - `kMinLaneFactors` 64 -> 16: lanes against standard, linear factors, `lc-bench`: 16 factors
    416 / 530 ns, 32 factors 714 / 1146, 8 factors 317 / 278 (lanes lose with one slot).
  - Products near 2^25 coefficients (2^22 linear factors; 4 of degree 2^23; 2^24 and 2^20
    quadratics) checked at random points on the Mac, `lc-amd` avx2 and scalar; one more linear
    factor throws std::length_error.
- Next: ntt's `forward_upper` (the subtree of group 1) and an inverse out of place
  would remove the twist pass and the product pass's copy (Zen 3: about 0.35 ns per word per
  node, 1.4 ms of 10.4 at 2^18); one twiddle table for all of ntt's `Transform` lengths (memory);
  slot loads (0.87 ms) by gathers or wider loads; the top tree's 8 forwards of the lane products
  from the lane root's transform (lc-opt's note: about 0.5 ms); leaf products at 0.83 ns per word.

## Sources

- lc-opt `lib/poly/product_tree.hpp` and its notes (`lib/poly/notes.md`, Product tree), and
  `problems/polynomial/product_of_polynomial_sequence/notes.md`: transform doubling, the wrap
  trick, lanes with pointwise products, the split, the lane schoolbook's bounds.
