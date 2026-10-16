# backend/product_tree

Products of many polynomials modulo P = 998244353. API: `product_tree.hpp` (`size`, `product`;
both also over any span of items with a projection to their words). Trees: `tree.hpp`
(`LaneTree`, `StandardTree`, memory, tables). Kernels on `backend/simd` structs: `kernels.hpp`
(lane schoolbook, leaf products, pointwise products, slot loads, conversions). Choices per target:
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
  result leaves Montgomery form times the constant in one pass into the caller's out. Two passes
  over the factors: the first also counts the degrees below 1024 (`kCounted`, on the stack), the
  counting sort's histogram when its cut is below that; the second places them. A slot of 8
  linear factors loads their words in pairs and transposes them in registers (`load_linear`,
  AVX2 and Neon; `kLinearSlots`: not on Zen 3, where it was slower).
- Node lengths (both trees): a node covers leaves [lo, hi) and splits where the prefix sums of
  the leaves' degrees come closest to halves (lc-opt's split). Its length L is a power of two >=
  its degree (the wrap trick): from its children's transforms of length L its product p comes out
  as p mod (x^L - 1); if deg p = L, p[L] is the product of the leading coefficients (tracked per
  node, per lane in the lanes) and p[0] is corrected by it. A node hands its parent its transform
  at the parent's length; its lower half is the node's own product transform (kept: the product
  pass writes it and a copy into the upper half, where the inverse runs).
- Upper half of the parent's transform (transform doubling): `ntt::Transform::forward_upper` of
  length 2L takes p mod (x^L + 1) (L words) to the upper L / 8 leaves of p's leaf form: the
  subtree of group 1 under a radix-2 top, or of groups 2 and 3 under a radix-4 top after one
  radix-2 level with r(1) (lc-opt's `forward_upper`; added to ntt by issue #91). It runs in place
  on the upper half, where the inverse left p mod (x^L - 1), after the wrap fix (p[0] - 2 p[L]).
  The lanes need nothing more: the words' polynomial mod x^(8L) + 1 is each lane's mod X^L + 1.
  Per node: one product pass (with the copy), one inverse and one forward_upper of length L. A
  child shorter than half its parent keeps p (L + 1 coefficients) and extends it at 2L, 4L, ...
  by forward_upper of p itself (p mod (x^m + 1) = p for m > L). Until #91 the upper half was a
  forward of a twisted copy (coefficient group j times r(L/16)^j, a pass of 0.12-0.25 ns per
  word); forward_upper costs 0.02-0.12 ns per word more than forward (its top group has
  twiddles, the forward's is the identity), so it wins at every length (Log).
- Lanes (`LaneTree`): word 8 j + l = coefficient j of lane l. A transform of 8 m words then holds
  lane l at w_v in word l of leaf v, so products are pointwise (Montgomery, one per word). Nodes
  up to degree 16 (`kLaneBase`) are multiplied out by a lane schoolbook in consecutive pairs
  (64-bit sums of at most 9 products, one Montgomery step, no fold; lc-opt's base was 32 with a
  fold and measured the same as 16), then transformed. Lane length >= 8 (64 words).
- Standard (`StandardTree`): leaf products mod (x^8 - w_v) (`leaf_products`: a window [w A, A]
  per leaf, 8 broadcast products in 64-bit sums, one Montgomery step; windows of 16 leaves are
  written before their products read them). Nodes up to degree 32
  (`kStandardBase`) by `small::multiply` in pairs (each product of Montgomery forms carries one
  more 2^32: one pass at the end takes them out). Length >= 64.
- Montgomery form throughout (x 2^32 mod P): pointwise and leaf products need no correction.
- Memory: one region of ntt's per-thread memory (`ntt::detail::take_region`, kept across calls,
  huge pages), used as a stack (`Stack`; past it, blocks from the heap). Nothing of the number of
  factors goes to the heap: fresh pages cost about 1.7 us per 4 KiB page on Zen 3 (lc-opt's notes),
  and `product@standard/18` lost 6 ms to a `std::vector` of leaves (Log). Per thread, kept: ntt's
  `Transform` per length (tables of about 1.25 2^lg bytes each, so up to 2.5x the longest's) and
  the leaf weights w_v 2^32.

## Measurements

Bench (`bench.cpp`), GCC 15.2, medians of 11 rounds, ns per factor, commit 7d34d0c (issue #91
round 1, on main ad1e1b4; `lc-intel` moves 5-10% between runs, the paired ratios less).
`product/<lg>`: 2^lg factors x - r; `product_mixed`: degrees 0 .. 8, a quarter constants (sum
about 3.4 2^lg); `product_few/<lg>`: 8 factors of 2^lg / 8 or 2^lg / 8 + 1 coefficients.

| case | `lc-bench` (Zen 3, avx2) | `lc-intel` (Emerald Rapids, avx512) | `lc-zen5` (zen5) | `lc-arm` (Neoverse V2) |
|---|---:|---:|---:|---:|
| product/4 | 27.9 | 20.9 | 18.5 | 27.5 |
| product/6 | 18.5 | 17.8 | 12.7 | 20.2 |
| product/8 | 18.1 | 17.1 | 12.5 | 22.7 |
| product/10 | 19.6 | 18.5 | 12.7 | 27.2 |
| product/14 | 27.1 | 25.4 | 15.9 | 40.7 |
| product/18 | 37.9 | 38.5 | 21.0 | 59.9 |
| product@standard/18 | 60.5 | 60.4 | 35.3 | 82.1 |
| product_mixed/18 | 166 | 177 | 94.7 | 268 |
| product_few/20 (ms) | 11.31 | 11.60 | 5.45 | 19.86 |

Against main ad1e1b4 (#56's tree), same run, new/base, gated geomean: Zen 3 0.960, Emerald
Rapids 0.919, Zen 5 0.947, Neoverse V2 0.947 (Log, issue #91). Frontend `product/<lg>` at lg
10 / 14 / 18: Zen 3 0.959 / 0.949 / 0.954, Emerald Rapids 0.905 / 0.915 / 0.929, Zen 5 0.972 /
0.939 / 0.933, Neoverse V2 0.898 / 0.917 / 0.938.

Issue #56's tree (commit 328b28a on main a34ce14, 5 rounds; `lc-arm` 3 rounds of 2bd03d3), ns
per factor, lc-bench / lc-intel / lc-zen5 / lc-arm: product/10 20.5 / 20.5 / 13.5 / 30.3,
product/18 39.8 / 38.8 / 22.0 / 64.6, product@standard/18 67.4 / 60.4 / 41.4 / 86.5,
product_mixed/18 175 / 156 / 93 / 281, product_few/20 (ms) 11.66 / 10.70 / 5.73 / 20.26.

Against the frontend's former tree of `f * g` (main a34ce14), frontend `product/<lg>`, 7 rounds,
new/base at lg 10 / 14 / 18: `lc-bench` 0.309 / 0.309 / 0.322 (40.6 ns per factor at 2^18, was
126); `lc-intel` 0.395 / 0.364 / 0.361; `lc-zen5` 0.355 / 0.329 / 0.333; `lc-arm` (5 rounds,
2bd03d3) 0.372 / 0.353 / 0.348.

Zen 3 (`lc-bench`), 2^18 linear factors, phases before #91 (`explore_profile.cpp`, Log): all 10.39 ms; lane
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
- Next (issue #91): ntt's `forward_upper` (the subtree of group 1) and an inverse out of place
  would remove the twist pass and the product pass's copy (Zen 3: about 0.35 ns per word per
  node, 1.4 ms of 10.4 at 2^18); one twiddle table for all of ntt's `Transform` lengths (memory);
  slot loads (0.87 ms) by gathers or wider loads; the top tree's 8 forwards of the lane products
  from the lane root's transform (lc-opt's note: about 0.5 ms); leaf products at 0.83 ns per word.
- 2026-10-12, issue #91 round 1 (claude): the passes outside the transforms. Base main ad1e1b4,
  every timing machine, 11 rounds, same-run ratios new/base. Explorations (per-word benches of
  the node passes, slot loads, A/B scripts) in `~/Documents/hpc_library-explore/product_tree/`.
  - Kept:
    - `ntt::Transform::forward_upper` (added to ntt, all variants; ntt notes.md, Design) in place
      of the twist pass and the forward of the twisted copy, both trees and the longer
      extensions (Design). Per word, forward_upper is 0.02-0.12 ns slower than forward, the twist
      cost 0.12-0.26: product/18 0.974 (Zen 3), 0.974 (Emerald Rapids), 0.976 (Zen 5); gated
      geomean 0.980 / 0.984 / 0.988. The `twiddles` tables and the `twist` kernels are gone.
    - The first pass counts degrees below 1024 (`kCounted`), so the counting sort's own pass is
      skipped when its cut is below that: with forward_upper, gated geomean 0.964 (Zen 3), 0.939
      (Emerald Rapids), 0.969 (Zen 5), 0.961 (Neoverse V2); product/10 0.954 / 0.859 / 0.977 /
      0.935. The skipped pass cost more than a read of the spans (0.15 ms at 2^18): on Emerald
      Rapids product/8 4.05 against 4.60 us with the pass (A/B, 1.13x), probably its increments of
      one counter through memory (Zen 3 renames memory operands; a guess). Standard-only inputs
      (cut 0) pay for the counting they do not use: product@standard/4 1.05 on Emerald Rapids,
      1.03 on Zen 5 (16 factors, 16 ns).
    - Slots of 8 linear factors: the factors' two words loaded as one 64-bit word each and
      transposed in registers (`load_linear`; AVX2 `vpermd` and `vperm2i128`, Neon `uzp`), not
      words stored one by one and reloaded as vectors. Alone: slot loads at 2^18 0.87 -> 0.585 ms
      on Zen 3; in the trees (A/B against the generic path): Emerald Rapids product/8 1.024,
      product/18 1.006 slower without it, Zen 5 1.012 at 2^18, Neoverse V2 1.054 / 1.014, but
      Zen 3 0.977 / 0.988 faster without it (in the tree; the reason is not known). So
      `kLinearSlots` is off on avx2 without AVX-512VL.
    - Leaf products with windows for 16 leaves written before their products (from 8): alone
      Zen 3 0.96, Emerald Rapids 0.93, Zen 5 0.91, Neoverse V2 1.00 of the time; 32 and 64 leaves
      as 16 or 1% slower, 4 leaves 1.05-1.24x.
    - All together (commit 7d34d0c): Measurements.
  - Tried, not kept:
    - The product pass without its copy, the inverse reading the lower half into the upper (the
      variants' inverse from other memory: its subtrees copy their leaves in before their bottom
      level): 1.0095 geomean on Zen 3 (product/18 1.017, product_few/20 1.020). Alone, inverse
      from other memory is 1.08-1.12x the in-place inverse at 2^8 .. 2^21 (Zen 3), while the copy costs the
      product pass 1-12% (it stores into lines it just read). `Transform::inverse(F, out)`
      itself would take ntt's region, which the tree holds (a fresh mapping per call).
    - Leaf products with the odd words by a shift of the even load (one load fewer per term):
      1.04-1.12x slower on x86, equal on Neoverse V2.
    - Leaf products in columns (8 leaves transposed so a vector lane is a leaf; no windows,
      broadcasts or shifted loads; 2 transposes in, 1 out), 2^8 .. 2^18: Zen 3 1.12-1.36x,
      Emerald Rapids 1.05-1.13x, Zen 5 1.05-1.12x slower, Neoverse V2 equal, M4 (clang) 0.82.
    - The counting sort's counters for a run of one degree in registers, both passes (a repeat
      of #56's attempt, for Emerald Rapids): against 7d34d0c, gated geomean 1.013 (Zen 3), 0.991
      (Emerald Rapids), 0.992 (Zen 5), 0.981 (Neoverse V2); product_mixed 1.00-1.02 slower
      everywhere (random degrees flush every factor). Mixed and Zen 3 slower: not kept.
    - Not tried, by arithmetic: the standard tree in ntt's values form (pointwise products, 0.18
      ns per word, instead of leaf products, 0.85): the split and join add 0.85 ns per word per
      node (ntt notes), more than the 0.67 saved.
  - Left (next rounds): the top tree's 8 forwards from the lane root (item 3 of #91); one table
    for all of ntt's Transform lengths (memory, 2.5x the longest's); leaf products, 26 cycles per
    leaf against about 13-15 (#98); the standard-only driver's unused counting (16 ns at 16
    factors on Emerald Rapids).
- 2026-10-16, issue #136 round 1 (claude, for backend/evaluation): `multiply_units` and the
  `kUnitTerms` hook of `LaneTree::coefficients` (leaves whose factors have constant term 1);
  product_tree's own code unchanged (same instructions). `notes/136-r1.md`.

## Sources

- lc-opt `lib/poly/product_tree.hpp` and its notes (`lib/poly/notes.md`, Product tree), and
  `problems/polynomial/product_of_polynomial_sequence/notes.md`: transform doubling, the wrap
  trick, lanes with pointwise products, the split, the lane schoolbook's bounds.
