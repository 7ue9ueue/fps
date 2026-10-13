# backend/ntt

Number-theoretic transforms and products modulo P = 998244353. API: `ntt.hpp` (`multiply`,
`square`, `middle_product`, `Convolution`, `Transform`, the leaf form). Variants: `scalar.hpp` (every target; the model),
`intrinsics.hpp` (`intrinsics<S>`, templates on any `backend/simd` struct S), `avx2.hpp` (lc-opt's
`lib/ntt`), `avx512.hpp` (intrinsics on zmm with generated asm kernels, issue #150).
`lib/ntt`), `neon.hpp` (arm: `intrinsics<Neon>` with generated asm kernels, issue #151).
`select.hpp`: the variant per op and length, a tuned choice ("ntt.variant", issue #78): per target
a list of candidates and a table of one per op (forward, inverse, multiply, product) and lg, with
compiled defaults in `tuned.hpp` (`tools/tune.py ntt`, `tune.cpp`); `tune.hpp` measures it at run
time. `memory.hpp`: `Convolution`'s memory, kept per thread. Tests: `test.cpp`,
`test_no_tuner.cpp`; benches: `bench.cpp`.
Generated, do not edit: `avx2_kernels.hpp` (`gen_avx2_kernels.py`), `avx2_product_kernels.hpp`
(`gen_avx2_product_kernels.py`), `avx512_kernels.hpp` (`gen_avx512_kernels.py`).
(`gen_avx2_product_kernels.py`), `neon_kernels.hpp` (`gen_neon_kernels.py`).

## Design

- One cyclic transform of length L = 2^lg, 2^6 <= L <= 2^25, splits x^L - 1 down to L/8 leaves
  x^8 - w (the leaf form, `ntt.hpp`) and multiplies leaf pairs directly. Only roots of order
  L/8 <= 2^22 are needed; P - 1 = 119 * 2^23.
- Radix-4 groups: group k at stride h holds 4h vectors (a polynomial mod X^4 - r(k)^2,
  X = x^(8h)); its children are 4k + t. L/8 = 4^j starts with one radix-4 group, L/8 = 2 * 4^j
  with a radix-2 level and two groups. Every variant splits the same way, so their forward
  outputs agree word for word once canonical: the scalar variant is the tests' model.
- Twiddle tables: entry k = product of roots[j] over the set bits j of k (roots[j] of order
  2^(j + 2)), so entry 2k is a square root of entry k and entry 2k + 1 = sqrt(-1) times it.
  Blocks of 8 values then their 8 Shoup quotients; L/16 entries each way. Entry k does not
  depend on L: both tables for L <= 2^16 are built at compile time (`kStaticTables`, 2 x 32 KiB);
  longer tables copy them and extend.
- `Convolution` (memory: `memory.hpp`): a and b in one region, 2 MiB aligned, `MADV_HUGEPAGE`,
  kept per thread across calls (one free region per thread, the largest given back); a and b are
  L + 16 words apart, so equal offsets fall in different cache sets. With a in the caller's memory
  (64-byte aligned, L + 16 words), b is in the region at a + L + 16 words modulo 4 KiB: the same
  layout as far as caches and 4K aliasing see it. The constructor zeroes [n, L/2) of a sparse
  factor (n <= L/2; its upper half is not read) and [n, L) of a dense one. Up to 2^16 the tables
  are the compile-time ones; past it, one pair per thread, built for the longest length so far.
- Variant contract for `multiply(a, b, n, m, lg, ...)`: words [n, L) of a are zero, except that
  the upper half of a factor with n <= L/2 may hold anything (avx2 and intrinsics do not read it;
  scalar zeroes it). Same for b.
- `multiply(f, g, out)` (issue #38 round 2): a product from caller spans into a tight output,
  `out = f g mod x^out.size()`; the factors are cut to out.size() first, so a cut product may use
  a shorter transform. Variant op `multiply(const detail::Product&)`: f (n words) and g (m words)
  in caller memory, a and b this thread's region, count = min(n + m - 1, out.size()) words to
  out. Nothing is copied whole: the first level of each transform reads its factor from the
  caller, and the last inverse level writes out.
  - First level (`detail::first_level`): its columns span K blocks (K = 2, 4, 8: the radix-2,
    radix-4 or fused radix-8 top). n = F blocks + c words: columns [0, c0) read F + 1 blocks, the
    rest F, where c0, c1 are c rounded down and up to the level's step (one or two columns: the
    asm and `each2` loops run two per iteration). The step of columns holding the factor's last
    word is staged in a (copied, zero past n) and run in place. Each first level takes the number
    of blocks read, T, as a template argument: blocks t >= T are zero, not loaded, and the
    products by them are not computed (avx2 radix-8 for T <= 4 is the old sparse form, the asm
    `forward_top3_from` for T = 3).
  - Last level (`detail::last_level`): count = F blocks + c words: columns [0, c0) write F + 1
    blocks to out, [c0, block) F; the step holding the last word runs in place in a and one copy
    moves its words of block F. The asm `inverse_top_to<T>` and `inverse_top4_to<T>` write T
    blocks: the generator drops the ops of the others (`prune`). A product cut to half the
    transform (out of L/2 words) computes only the outputs of the top level it writes.
  - The levels pick T with an inlined switch, and each T is its own function. One function for
    every T (a lambda), with two 64-bit divisions per level, cost a 2^7 product 268 instructions
    and 72 cycles on Emerald Rapids (3176 -> 3444 instructions).
  - a and b sit in the region at least 256 bytes from f's, g's and out's page offsets (`place`):
    the bench's spans 64 bytes past a page, against a at 0 and b at 64, ran Zen 3 1.08-1.11x
    slower than multiply() at even lg 2^14-2^22, 1.04-1.06x with the placement (4K aliasing:
    Measurements, Convolution memory).
  - Products in place (`Convolution`) and `Transform` keep their own top levels and kernels,
    unchanged. Through the span levels (one implementation for both) they ran 2-4% slower at
    2^6-2^7 on Emerald Rapids and Zen 5 (Log, issue #38 round 2).
- `square(f, out)` (issue #55): `multiply(f, f, out)` with one forward transform. Variant op
  `square(const detail::Product&)` (g, m, b unused): the first level reads f into a, the tree
  transforms a alone, and the bottom stage takes a's leaves as both factors (intrinsics: `kSquare`
  in `multiply_tree`, `Bottom` and `WideBottom`, whose leaf products read B from A's rows; avx2:
  `PairSquareBottom`, generated kernels `bottom_square_*` without b's forward half, at every
  length). The variant per length is the product row's (`variant::tuned::square`, `kProduct`):
  no row of its own, none measured to pay.
- `middle_product(f, g, out)` (issue #55): out[i] = (f g)[m - 1 + i], out.size() <= n - m + 1.
  A cyclic product of length 2^lg >= n' = m - 1 + out.size() (only f[0, n') is read): the terms
  that wrap land at i + j - 2^lg <= m - 2, below the result. g is copied into b turned by
  x^-(m - 1) (b[0] = g[m - 1], g[0, m - 1) at the top, zeros between), so the result starts at
  word 0 and the variants' `multiply(const Product&)` runs unchanged with g = b, m = 2^lg (the
  first level reads b in place; scalar skips its copy then). Costs a dense first level for b and
  the staging pass (one write of 2^lg words): middle_product/22 (f of 2^22 words, transform 2^22)
  took 1.03 of product/22 (factors of 2^21, sparse) on Zen 3.
- `Transform`: tables built once; forward and inverse in place on caller arrays (32-byte aligned,
  16 words of padding: the AVX2 forward kernels read 4 bytes past their last vector).
- `Transform`'s leaf-form ops (issue #49), for `fps::Values` and the product tree. Each variant
  has `leaf_products`, `forward_values` and `inverse(from, f, out, count, ...)`; the API calls
  `Native` by default (not the per-op selection, so `select.hpp` is untouched).
  - Leaf products (`multiply`, `multiply_add`): `Bottom<N>::products`, a group of 4 leaves per
    step: rows [w A, A] (one Shoup product per leaf), the 8 x 8 products of the fused bottoms
    (64-bit sums), then on x86 an exact reduction (s = hi 2^32 + lo, hi (2^32 mod P) + lo by one
    Shoup product), elsewhere Montgomery's and a Shoup product by 2^32. Group g + 1's rows are
    filled before group g's products. avx2 runs `Bottom<Avx2>` too (lc-opt's asm was slower, Log).
  - Values form (`Form::values`, L <= 2^23): leaf v = 8g + t at its 8 roots rho_v z^j, rho_v =
    r(4v), word 64g + 8i + t at j = bitrev3(i) (common.hpp). Per block of 8 leaves (`Values<N>`):
    a transpose (lanes = leaves), the twist c_k rho^k as two Shoup products, B(g % 8, t)^k per lane
    (a constexpr table, 8 KiB) and A(g / 8)^k broadcast (a table per Transform, L / 16 words for
    both directions), the 8-point transform (5 products), stores by value. The inverse: the
    8-point inverse, the untwist with 1/8 in the lane table, a transpose back. 19 and 20 products
    per 64 words. Fused into the transforms through tile hooks: `forward_tiles` (copies of
    `forward_trees`/`forward_tree` with a hook) call the split on each run of subtrees after its
    bottom level (in L1), `inverse_tiles` join before the bottom level. Below 2^8 the intrinsics' subtrees have 4 leaves (half a block):
    there the split runs after the transform and the join before.
  - Inverse into other memory (`inverse(F, out, form, {offset, add})`): the same tile hook copies
    (or joins) each run of subtrees from F into the work array before its bottom level, so F is
    read once and stays; the top level writes out with the product's `*_out` levels (any
    alignment, count words). With an offset or add, the top level runs in place and one pass
    copies or adds the range. Work array: the thread's region (`place` keeps it 256 bytes from
    F's and out's page offsets), on the stack up to 2^10.
- Hooked and pruned ops (issue #149, for backend/composition; notes/149-r1.md):
  `forward_hooked(in, F, hook, zero)` hands each run of leaves to the caller's hook after its
  bottom level; `inverse_hooked(hook, work, out, o, unused)` lets the hook write each run (times
  2^-32) before its bottom level. A column bit zero in the forward's source (unused in the
  inverse's output) prunes the groups above it to half their columns, the top level included
  (`PrunedLevel` in common.hpp; avx2 `kernels::*_columns`, intrinsics `*_group_columns`).

### avx2

From lc-opt's `lib/ntt` (`ntt.hpp`, `product.hpp`; its `notes.md` has the history and the QPoly
origins). One product, `product<Bottom>` (lc-opt had two paths, `Convolution` and `Product`):

- Depth-first radix-4 recursion; runs of sibling subtrees of at most 256 vectors in all (8 KiB
  per array) go level by level as one tile, so one bottom stage pipelines across all of them.
- Bottom stage: the bottom level (h = 1) fused with the leaf products and the first inverse
  level; batch j + 1's forward half is interleaved with batch j's products. `PairBottom`
  (lc-opt's `Product`): two groups per always-inlined asm statement, leaf weights from the
  group's own twiddles, each coefficient of B broadcast once (145 -> 127 cycles per group on
  Zen 3 in lc-opt). `GroupBottom` (lc-opt's `Convolution`): one group per statement. Pair at
  every length but 2^7 and 2^8, where Group was faster (2026-10-11 log).
- Sparse factors (upper half zero) are not read past the lower half: the first radix-2 level is
  a copy, or the first radix-4 level or `forward_radix8` reads only the lower half.
- Even lg: `forward_radix8` (sparse or dense) does the radix-2 level and both halves' first
  radix-4 groups in one pass; `inverse_top` fuses the last inverse levels and the scale
  (9 Shoup products per column, not 13). Odd lg: `forward_top` (dense) and `inverse_top4`, the
  identity group with the scale in its twiddles (4 Shoup products per column, not 5).
- Arithmetic: Shoup multiplication by twiddles, w < P with q = floor(w 2^32 / P); for any
  x < 2^32, x w - floor(x q / 2^32) P is in [0, 2P). Forward values < 4P, inverse < 2P;
  min(x, x - 2P) reduces lazily. Leaf products: Montgomery reduction (factor 2^-32).
- Kernels: `gen_avx2_kernels.py` builds each as a dataflow graph of AVX2 instructions,
  list-schedules it on a Zen 3 model (4 vector pipes, multiplies and shifts on 2 each, 2 loads,
  1 store per cycle) under a register budget, allocates ymm0-15 by linear scan and emits one asm
  statement. Knobs from QPoly exploration 009 (timed on EPYC 7763).
- Scales come from a constexpr table.
- `Transform::forward`/`inverse` (new here, not in lc-opt): the same recursion and tiles on one
  array with the asm kernels down to h = 4; the h = 1 level is `forward_leaves`/`inverse_leaves`
  (two groups per asm statement; canonical leaves out, canonical leaves in). Odd lg: `forward_top`
  (the first level without input reductions: 20 vector ops per column, not 33) and
  `inverse_top4`. Even lg: a radix-2 pass without reductions (canonical inputs) below 2^16,
  `forward_radix8_dense` from 2^16; `inverse_top`.

### intrinsics

`intrinsics<S>` (`intrinsics.hpp`): the algorithm of the avx2 variant, written once on a simd
struct S and `modarith::Ops<S, P>` (Shoup products by `shoup`, the selected variant). Same
splitting, twiddle tables (`build_table` extends the static prefix N lanes at a time), lazy ranges
(forward < 4P, inverse < 2P) and leaf products (Montgomery, factor 2^-32), so its outputs equal
the other variants' word for word.

- Levels with quarters of at least 32 words run on S, one group at a time. The first level of a
  transform takes inputs < 2P (canonical, or after the radix-2 level), so it skips the input
  reductions: 2 of forward4's, 4 of forward4_identity's (and 2 more for canonical inputs).
- The levels inside a subtree of at most 256 leaves run one call per level (`forward_groups`,
  `inverse_groups`), not one per group. Groups of two columns (quarters of 32 words on zmm) run
  both columns side by side, without the pipeline below. Emerald Rapids, cycles per 64 columns on
  L1 data against one group of 64 columns (1810): a call per group of two columns 2145, a loop of
  pipelined groups 2120 (GCC rematerialized constants per group), two columns side by side with
  the unshifted q below 1919 (Log, issue #5 round 2).
- Each butterfly is two stages on a column of 4 vectors (forward: input reductions and x f2,
  x f3, then the rest; inverse: the sums and y (a - b), z (c - d), then the rest). Groups run
  them software pipelined (`each2`): the first stage of column j + 1, then the second of column j,
  unrolled by 2 so no registers are copied. Shorter dependency chains per loop iteration: on L1
  data, forward and inverse groups 5-8% and 8-9% faster on Neoverse V2, 11% and 7% on Zen 5, 2-4%
  and 4-5% on Emerald Rapids (Measurements). Identity groups (k = 0) only where that won
  (`kPipelined`): Neoverse V2 both (1.5%, 8%), Zen 4/5 the inverse (4%); Emerald Rapids lost 1-3%
  on them.
- Shoup products go through an empty asm statement (`Arith::whole`). Without it GCC distributes
  x w - q P over the product's uses (a + c -> (a - q P) + x w, a - c -> (a + q P) - x w): 24 adds
  and subtracts per forward radix-4 column instead of 19 on zmm (Log, issue #5).
- On x86, `Arith::mul` reads the quotient q unshifted for the odd words (`mul_even` of x >> 32 and
  q): every factor is a broadcast or a half broadcast, so its words 2i and 2i + 1 are equal. One
  p0 shift less per product where q is not loop-invariant: the zmm bottoms (3 to 8 per pair of
  groups) and groups of two columns (3 per group). The join is simd's (`simd::detail::
  kMulhiPermute`: `vpermt2d`, or shift and blend on Zen 4/5).
- Software prefetch, Emerald Rapids only (`kPrefetch`): groups on quarters of at least 2^15 words
  and radix-2 levels on halves of at least 2^20 words prefetch, per column, the line 512 words
  ahead in each stream (`prefetcht0`), and `prefetchw` the lines a pass writes without reading
  (the upper subtree's first level reading the lower half; the sparse radix-4 top's quarters 2
  and 3). In 2 MiB pages, the 4 streams of a group 128 KiB or more apart share an L2 set at each
  column; one pass over 4 quarters of 2 MiB ran at 0.164 ns per word, 0.149 with the quarters
  4 KiB apart, 0.134 with the prefetch (Measurements). Smaller quarters lost with it (2^12-2^16
  1-4%), as did Zen 5 (Log).
- Bottom level (quarters of 8 words, one leaf each): on S up to 8 lanes, one group at a time, its
  columns two at a time on 4-lane structs (Neon, Sse: bottom forward 10%, inverse 8% faster on
  Neoverse V2). On `Avx512` (16 lanes, `WideBottom`): two groups k, k + 1 per step; four
  `vpermt2d` turn their 64 words into vectors [quarter t of group k, quarter t of group k + 1], so
  the butterflies are lane-wise with twiddles [w_k x 8, w_(k+1) x 8] (two `vpermd` of one 16-word
  table load each), and four more turn them back.
- Recursion: depth first; subtrees of at most 256 leaves (8 KiB per array) run level by level.
  Each subtree has its own bottom stage (avx2 now merges sibling subtrees into one tile; not done
  here). Top: a radix-2 level (odd lg - 3) or group 0, with the sparse forms for a factor of at
  most L/2 coefficients. The inverse scale goes into the top groups of the subtrees below the top
  level (`inverse_group_scaled`: twiddles s x, s y, s z and one product more for the first output,
  so 5 products per column, and 4 against 1 for k = 0), when those have a group level (16 leaves
  or more), on Emerald Rapids up to 2^17 (`kFoldMaxLog`; Log); their twiddles come from a
  compile-time table per length (`kScaledRoots`: s r(i) for i < 8 and quotients; computed per
  call they cost Zen 5 6% at 2^9). The top level then only adds: per vector of the transform, 0.5
  products fewer and 1.25 other ops more under a radix-2 top (2 products per pair before), 0.375
  products fewer under a radix-4 top (4 products per column before, as `s (a - b)` and
  `s z (c - d)` served two outputs each; that form stays for the shortest lengths). Sparse with
  the radix-2 level: no copy; the upper subtree runs first and its first level reads the lower
  half (from nv / 2 = 16 leaves; below, the first level is the bottom, which works in place, so
  the half is copied). Sparse with group 0: group 0 reads half the input. No fused radix-8 top:
  slower everywhere it was tried (Log).
- Products, per bottom group: forward butterflies of a and b, leaf products, inverse butterflies,
  one store. Leaf products: x86 up to 8 lanes (`mul_even`): lc-opt's even/odd form, windows
  [w A, A] in a stack buffer read at offsets 1..9, each b_i broadcast once, 64-bit sums < 8 P^2,
  Montgomery reduction, `join_high`; group g + 1's forward half runs before group g's products,
  so the windows are not reloaded right after their stores. `Avx512`, two groups at a time
  (`WideBottom::multiply`), in chunks of 4 pairs: one loop of forward halves stores each pair's
  canonical leaves (rows [w A, A] per half, A, B) in a buffer, then one loop of leaf products and
  inverse halves (two loops: shorter dependency chains per iteration). Window offset j of both
  halves is a ymm load and a `vinserti64x4` from the rows (`load_halves`; still a p5 uop on
  Emerald Rapids, like the `vpermt2d` it replaced), b_i of both halves a broadcast and a masked
  broadcast load (`broadcast_halves`: p0 or p5, where the `vpermd` it replaced is p5 only; the
  leaf products had p5 at 161 uops per pair against p0 at 138), and the twiddles too. Neon: `umull`/`umlal` (+`umull2`/`umlal2`) on
  the windows, `uzp1` for m, `umlal` for m P, `uzp2`; two leaves at a time, both words of each
  (8 independent sums), each b_i broadcast once per leaf. Scalar: one u64 sum per word.
- `intrinsics<Avx512>` runs zmm at every length. Narrow lengths are the tuned table's choice of
  `intrinsics<Avx512Vl>` (issue #78; before it, `kWideMinLog` sent lengths below 2^8 on Emerald
  Rapids and 2^7 on Zen 4/5 to `Narrow<S>` by `__znver4__`/`__znver5__`, so the avx512 build ran
  Emerald Rapids' threshold on Zen 5). Before the zmm bottom level, zmm levels above ymm bottom
  stores were 1.25-1.33x slower at 2^7 and 2^8 on Emerald Rapids (`ld_blocks.store_forward` 1.8M
  against 12k: wide loads of narrow stores still in flight).
- GCC 15 at -O2 kept the fixed 4- and 8-trip loops of the leaf products rolled and did not inline
  the butterfly lambdas of the bottom level; `#pragma GCC unroll` and `[[gnu::always_inline]]`
  fix both (Log). A range-for over the 4 vectors of a column also stays rolled, through the
  stack (the zmm bottom forward had one: 8% of its time on Zen 5), so columns are written out.
  `Arith`'s ops are `always_inline`: in the bench binary (more instantiations than a probe) GCC
  left `Arith::mul` out of line once the groups were pipelined, a call per product (Zen 5 1.4x
  slower; Log).
- Products from spans: the in-place top levels' columns on ranges, `each` and `each2(from, f,
  q, begin, end)` with kLoad quarters read and kStore written (the defaults, 4 and 4, are the
  in-place forms). A factor of at most half the length is read straight from the caller by both
  subtrees' first groups (`Source`: n words, zero past them), as the in-place form reads its
  lower half; a longer one by the radix-2 level. `prefetchw` covers quarters that are written but
  not read (zero quarters past the source): without it, Emerald Rapids ran multiply_dense at odd
  lg from 2^17 2% slower in the unified attempt (Log).

### avx512 (issue #150)

`variant::avx512` (`avx512.hpp`): `intrinsics<S>` on a struct of its own (`avx512::Zmm`, simd's
Avx512 renamed), for which `intrinsics.hpp`'s hook `Generated<S>` swaps in generated asm for some
kernels; every other path is the zmm intrinsics'. One generator, one header per target's machine:
`avx512_kernels.hpp` (Emerald Rapids, avx512 target) and `zen5_kernels.hpp` (Zen 5, zen5 target,
issue #152, below). A candidate on both targets; on zen5 also `Native` (the leaf-form ops).

- Generator: gen_avx2_kernels.py's method on zmm. Dataflow graphs of EVEX instructions (memory
  operands and `{1to16}` broadcasts folded where used once, vpermt2d/vpermi2d writing a source,
  broadcasts merged under a mask), list-scheduled on `backend/simd/machines.py`'s measured model
  of the target (pipes, latencies, uops; Emerald Rapids: 512-bit ops on p0 and p5, vpminud and
  shifts p0 only, permutes p5 only, vpmulld 2 uops), with memory order kept (loads after stores
  that overlap them, stores after loads), allocated on zmm0-31 by linear scan.
- Forward groups k > 0 of 4 columns and more (`forward`, `forward_in2`, prefetching forms): 4
  columns per iteration; the odd words of quarters 2 and 3 from loads 4 bytes on (2 shifts fewer
  per column: p0 uops 452 -> 424 per 16 columns), so they read 4 bytes past a column, inside the
  arrays' padding. Groups of 2 columns, group 0 and the inverse groups stay intrinsics: the asm
  ran as fast (within 1%).
- Product bottoms (`multiply_*`, `square_*`, `stored_*`): WideBottom's pair layout, in two stages
  as gen_avx2_product_kernels.py's: first (forward butterflies of a pair into a leaf buffer: rows
  w A and A, B) and last (leaf products, inverse butterflies, transposed stores); each statement
  runs last of pair j and first of pair j + 1 interleaved. The windows x^(8-j) A of the leaf
  products are vpermi2d of the two rows (indices loaded per use: vpermi2d overwrites them), so no
  row is stored in halves. Kernel time 0.93 of WideBottom::multiply's on L1 data.
- Constants: each statement loads its own (pure loads); masks are "Yk" operands, as a kmovw from
  memory costs a p0/p5 uop (measured, notes/150-r1.md).
- Zen 5 (`zen5_kernels.hpp`, issue #152; notes/152-r1.md): the model in `machines.py` (four 512-bit
  pipes: multiplies and shifts on disjoint pairs, permutes and masked-broadcast merges on a pair
  sharing one pipe with each; 2 zmm loads and 1 store per cycle, a load across a line takes two
  slots), uops matched to pipes per cycle, and only ops that keep the source order feasible in
  registers (no deadlock: `Liveness`, after gen_neon_kernels.py). Kernels: every group loop
  (forward and inverse, k > 0 and identity, 4 columns per iteration, 8 for inverse identity),
  forward groups of two columns (quarters of 32 words) four groups per statement, the bottom
  level forward and inverse in statements of 4 and 2 pairs, and the product bottoms as on Emerald
  Rapids (one pair per stage; `multiply_*` scheduled on the Emerald Rapids model, which ran faster
  on Zen 5). Kernels on L1 data against the intrinsics: groups 0.86-0.91, two-column forward
  groups 0.77, bottom forward 0.80, inverse 0.85, product bottom 0.91, square 0.82.
- Zen 5 cares about code size and store forwarding more than the model says: the product bottom two
  pairs per stage (0.88 on L1 data) and inverse two-column groups (0.86) made whole products slower
  (more hot code than the op cache holds, a guess), and loads 4 bytes past a column (odd words)
  in the two-column groups ran 1.36x slower (loads across two recent stores do not forward).

### neon (issue #151)

The arm target's one candidate: the intrinsics on `neon::Vec` (`simd::Neon` under its own name)
with asm kernels generated for Neoverse V2 (`gen_neon_kernels.py`) where SVE2 is there (`umulh`);
without it (the Mac) the same name runs the intrinsics alone, so every arm build has one candidate
list. `intrinsics::Generated<neon::Vec>` (the hook of the avx512 variant, issue #150, with optional
members) replaces: the bottoms of products, squares and products by stored leaves
(`Bottom::multiply`, `multiply_stored`) and of transforms (`Bottom::forward`, `inverse`) for an
even group count, and the group levels (forward in place with inputs < 4P or < 2P, from other
memory with inputs < 2P; inverse).
- Generator: graph, source order, list schedule on a V2 model under the register budget (an op that
  grows the live set only if the rest can follow in source order within it: no deadlock), then a
  reorder that alternates multiplies and other vector ops (2-5% on V2), linear-scan allocation.
- Product bottoms: two groups per asm statement, the forward half of pair j + 1 interleaved with
  the products of pair j through a leaf buffer; twiddles and B by element (`umlal v, x, b.s[i]`),
  quotients broadcast once per group. Group levels: an asm loop, 4 columns per iteration.
- Round 2 (`notes/151-r2.md`): the same loops for the scaled top groups of subtrees
  (`inverse_group_scaled`), the forward top group on canonical inputs, the inverse top group and
  radix-2 level (in place and into a product's output: a boundary of 8 words reruns the last 16,
  which read only the source), with twiddles in general registers (a block on the stack read by
  vector loads cost inverse/6 12%); the bottom of `inverse_product` of one term (rows from
  leaves, no butterflies). At 2^6 and 2^7 the subtrees are single groups:
  `Generated::kJoinBottoms` makes the top levels run all their bottoms in one call, so the pair
  kernels take them (products 0.74-0.81 there).
- Against `intrinsics<Neon>` on L1 data: product bottom 0.74, group loops 0.84-0.91, transform
  bottoms 0.83 (forward) and 0.95 (inverse). Whole ops, same binary, 2^8-2^22: multiply and
  products 0.80-0.84, forward 0.84-0.89, inverse 0.90-0.98 (round 1; 2^6-2^7 then unchanged).
  `Native` is neon on arm. Attempts and numbers: `notes/151-r1.md`, `notes/151-r2.md`.

### Variant selection (issue #78)

A tuned choice, "ntt.variant" (`backend/tune`): per target a list of candidates (`select.hpp`,
`Selected`) and a table of one candidate per op and lg 6..25 (`Choices`, 80 bytes).

- Candidates: avx512 and zen5 {avx2, intrinsics<Avx512Vl>, intrinsics<Avx512>, avx512} (issues
  #150, #152); avx2 {avx2,
  intrinsics<Avx2>}; arm, sse, scalar intrinsics<Native> alone (no table read: `choice()` is a
  constant 0). Each is fastest somewhere on our machines (Log, issue #78).
- Candidates: avx512 and zen5 {avx2, intrinsics<Avx512Vl>, intrinsics<Avx512>}; avx2 {avx2,
  intrinsics<Avx2>}; sse, scalar intrinsics<Native> alone (no table read: `choice()` is a
  constant 0), arm neon alone (issue #151). Each is fastest somewhere on our machines (Log, issue #78).
- Ops: forward and inverse (`Transform`), multiply (`Convolution::multiply`) and product
  (`multiply(f, g, out)`). The two products have their own rows: their top levels differ (from
  spans: `forward_top_from` and `inverse_top_to<T>` in the asm, `forward_radix4_from` and
  `inverse_radix4_out` in the intrinsics), and those levels are a large share of the time at
  2^6-2^7, where the candidates cross. Emerald Rapids at 2^6: the ymm intrinsics beat the asm by
  9-12% on multiply but by 1-3% on product (tune.cpp, 3 runs). The frontend calls all four.
- Dispatch (`variant::tuned`, which `SelectedForward`, `SelectedInverse` and `SelectedMultiply`
  name): one table read per call (`tune::Table::get()`, an acquire load), then a chain of
  compares on the index inlined into the op (`detail::run_forward` and siblings), as small's
  `run_candidate`. `tuned::build_table` is `Native`'s; every variant builds the same tables.
- `intrinsics<Avx512>` runs zmm at every length; before this, `kWideMinLog` sent lengths below
  2^8 (Emerald Rapids) or 2^7 (`__znver4__`, `__znver5__`) to its ymm form inside the variant.
  That ymm form is the candidate intrinsics<Avx512Vl> now, so the table also decides narrow
  against wide, per CPU at run time (the avx512 build on Zen 5 had Emerald Rapids' threshold).
- `tune.hpp` (`measure`) times the candidates as the library runs them: Transform on an array
  kept across calls, one Convolution (this thread's kept region), multiply(f, g, out) from
  std::vectors; products at factors (L/2, L/2) and (3L/4, L/4 + 1). The least geometric mean of
  time / fastest wins unless the current candidate is within the margin. Run time (`kRunTime`):
  lengths to 2^12, 5 samples of 100 us, 3%; longer lengths keep the table unless the choice at
  2^12 changed. `tune.cpp` (`tools/tune.py ntt`): every length, 7 samples of 1 ms, 1%.
- Before (`by_length`, by hand from the bench's @ cases): avx512 asm below 2^8 for forward and
  at 2^7 for multiply, ymm intrinsics below 2^8 for inverse; zen5 asm below 2^7 (forward) and 2^8
  (inverse, multiply). The data behind it: issue #5 round 2 and issue #4 round 2 (Log).
- `Transform::forward_upper(h)` (issue #91, for the product tree's transform doubling): on a
  Transform of length 2L, h = p mod (x^L + 1) (L words, canonical) -> leaves L/8 .. L/4 - 1 of
  p's leaf form. Under a radix-2 top level (has_radix2(lg)) that is the subtree of group 1 at
  the first radix-4 level; under a radix-4 top (group 0 = x^(2L) - 1, children x^(L/2) -+ 1, -+
  r(1)) the subtrees of groups 2 and 3, after one radix-2 level with r(1) (out < 3P). Every
  variant: scalar (`forward_levels` from a group offset), intrinsics (`forward_tree` of groups
  1, or 2 and 3), avx2 (`forward_trees` of the same); the tuned dispatch uses the forward's
  candidate at L. Cost against `forward` of L words (ns per word, lg 8 / 12 / 18): Zen 3 0.49 /
  0.71 / 1.13 against 0.42 / 0.65 / 1.07; Emerald Rapids 0.43 / 0.65 / 1.04 against 0.36 / 0.57 /
  0.93; Zen 5 0.27 / 0.32 / 0.50 against 0.23 / 0.29 / 0.48; Neoverse V2 0.76 / 1.22 / 1.96
  against 0.64 / 1.11 / 1.86: its top group has twiddles where the forward's is the identity
  (Zen 3 2^9: 0.54 against 0.42). A twist pass (one product per word) cost 0.12-0.26.

## Measurements

### Against lc-opt's `lib/ntt` (issue #3: the move must not cost speed)

`compare_lcopt.cpp` (exploration folder, `ntt/`) links both into one binary and times
`multiply()` on the same inputs, old and new alternating, 11 rounds (each round: median of 5
samples of >= 2 ms). Old: lc-opt `ntt::Product` for "multiply" at lg >= 9 (factors L/2 and L/2),
`ntt::Convolution` otherwise; "multiply_dense": factors 3L/4 and L/4 + 1. GCC 15.2 (pinned image),
`-O2 -march=x86-64-v3`, 2026-10-11, commit 4010a7e. Ratio = fps / lc-opt, median over rounds.

| lg | Zen 3 multiply | Zen 3 dense | Emerald Rapids multiply | Emerald Rapids dense |
|---:|---:|---:|---:|---:|
| 6 | 0.995 | 0.996 | 1.095 | 1.056 |
| 7 | 1.005 | 1.012 | 1.023 | 1.030 |
| 8 | 1.014 | 1.020 | 1.065 | 1.062 |
| 9 | 0.985 | 0.965 | 0.991 | 1.008 |
| 10 | 1.051 | 0.996 | 1.001 | 1.002 |
| 11 | 1.002 | 0.999 | 1.006 | 1.002 |
| 12 | 1.016 | 1.001 | 1.004 | 0.997 |
| 13-19 | 1.004-1.006 | 0.998-1.001 | 0.996-1.001 | 0.994-1.010 |
| 20-25 | 1.000-1.006 | 0.999-1.002 | 0.997-0.999 | 0.997-1.007 |
| geomean (40 cases) | 1.0027 | | 1.0074 | |

Zen 3 = `lc-bench` (EPYC 7B13); Emerald Rapids = `lc-intel` (Xeon 8581C). Absolute times at
lg = 25, Zen 3: multiply 192.6 ms (lc-opt 192.8), dense 198.5 ms (199.0).

From 2^9 up the move is within 0.6% (Zen 3 product path +0.4% median, IQR mostly above 1; Intel
equal). Below 2^9 on Intel the new code is 2-9% slower (216 -> 236 ns at 2^6), reproduced in a
second run of 21 rounds. `perf stat` at 2^6 on `lc-intel`, per call: lc-opt 904 cycles, 2400
instructions, 19 blocked store forwards; fps 990 cycles, 2455 instructions, 28 blocked store
forwards. `perf record`: the same asm `bottom_first` takes 197 -> 234 cycles per call; the other
55 instructions are the out-of-line `build_table`, `convolution` and the variant dispatch.
Forcing `always_inline` on those three changed nothing (2^6: 1.095 -> 1.086). Guess: stack layout
(the leaf buffers) against a and b changes which loads wait on stores; not confirmed.

### Absolute (lc-bench, round 1)

`kernel.py bench ntt --target avx2 --rounds 3`, EPYC 7B13, GCC 15.2, `-O2 -march=x86-64-v3`,
2026-10-11, commit 4010a7e plus the bench fix. ns per call (ns per coefficient):

| lg | forward | inverse | multiply | multiply_dense |
|---:|---:|---:|---:|---:|
| 6 | 25 (0.39) | 131 (2.04) | 249 (3.89) | 250 (3.90) |
| 8 | 111 (0.43) | 216 (0.84) | 703 (2.75) | 715 (2.79) |
| 10 | 541 (0.53) | 646 (0.63) | 2671 (2.61) | 2562 (2.50) |
| 12 | 2669 (0.65) | 2769 (0.68) | 10957 (2.68) | 11197 (2.73) |
| 16 | 61678 (0.94) | 60717 (0.93) | 215964 (3.30) | 233249 (3.56) |
| 20 | 1.29e6 (1.23) | 1.26e6 (1.20) | 4.31e6 (4.11) | 4.65e6 (4.44) |
| 24 | 2.80e7 (1.67) | 2.78e7 (1.66) | 8.59e7 (5.12) | 9.30e7 (5.54) |
| 25 | 5.67e7 (1.69) | 5.75e7 (1.71) | 1.93e8 (5.76) | 1.98e8 (5.91) |

inverse/6 is 5x forward/6: the scale (L/8)^-1 is a modular power per call, about 100 ns
(guess from the difference; multiply pays it too, as in lc-opt).

### Round 2 against main (issue #3)

`kernel.py bench ntt --base` against main 9355374, 7 rounds, GCC 15.2, 2026-10-11, commit d125d53.
`lc-bench` (EPYC 7B13, `-O2 -march=x86-64-v3`): new ns per call (new/base, median of paired ratios).
Geomean over the 80 gated cases: 0.9369.

| lg | forward | inverse | multiply | multiply_dense |
|---:|---:|---:|---:|---:|
| 6 | 23 (0.897) | 21 (0.950) | 127 (0.895) | 129 (0.906) |
| 7 | 39 (0.848) | 46 (0.844) | 233 (0.761) | 248 (0.788) |
| 8 | 101 (0.880) | 103 (0.902) | 478 (0.792) | 503 (0.812) |
| 9 | 211 (0.914) | 222 (0.847) | 1032 (0.776) | 1041 (0.860) |
| 10 | 525 (0.945) | 496 (0.931) | 2143 (0.788) | 2193 (0.894) |
| 11 | 1103 (0.932) | 1123 (0.883) | 4510 (0.882) | 4546 (0.866) |
| 12 | 2624 (0.966) | 2473 (0.961) | 9883 (0.913) | 10066 (0.914) |
| 13 | 5548 (0.949) | 5600 (0.910) | 21457 (0.965) | 21530 (0.905) |
| 14 | 13024 (0.981) | 12248 (0.979) | 46169 (0.986) | 46874 (0.938) |
| 15 | 26961 (0.965) | 26838 (0.920) | 99592 (0.973) | 100118 (0.923) |
| 16 | 60551 (0.992) | 57554 (0.978) | 2.12e5 (0.989) | 2.15e5 (0.948) |
| 17 | 1.26e5 (0.970) | 1.25e5 (0.932) | 4.69e5 (0.990) | 4.69e5 (0.949) |
| 18 | 2.79e5 (0.994) | 2.67e5 (0.984) | 9.82e5 (1.000) | 9.94e5 (0.965) |
| 19 | 5.80e5 (0.982) | 5.74e5 (0.942) | 2.11e6 (0.990) | 2.11e6 (0.956) |
| 20 | 1.26e6 (0.996) | 1.21e6 (0.986) | 4.32e6 (1.001) | 4.37e6 (0.969) |
| 21 | 2.61e6 (0.982) | 2.58e6 (0.948) | 9.19e6 (0.991) | 9.18e6 (0.958) |
| 22 | 5.62e6 (0.997) | 5.44e6 (0.990) | 1.90e7 (1.006) | 1.92e7 (0.974) |
| 23 | 1.18e7 (0.991) | 1.20e7 (0.960) | 4.17e7 (0.995) | 4.17e7 (0.966) |
| 24 | 2.55e7 (0.995) | 2.56e7 (1.000) | 8.56e7 (1.001) | 8.58e7 (0.977) |
| 25 | 5.62e7 (0.993) | 5.55e7 (0.981) | 1.90e8 (0.995) | 1.90e8 (0.963) |

multiply/6 ran at 105-108 ns in five other runs of the same binary (0.74); 127 ns here, not
reproduced. The same commit on the other targets, 5 rounds: `lc-intel` (avx512) 0.9504,
`lc-zen5` 0.9389, `lc-arm` (the scalar variant: only the compile-time tables) 0.9950.

### Costs on Zen 3 (`lc-bench`, round 2 probes)

Clock 3.45 GHz (a chain of dependent adds). Cycles, data in L1, before round 2's changes unless
marked:

- asm radix-4 loops, per column of 4 vectors: `forward` 14.5 (49 vector ops: 84% of 4 pipes),
  `inverse` 14.5 (52 ops), `forward_identity` 9.4, `inverse_identity` 8.6, `forward_top` 5.5 (new).
- h = 1 level of `Transform`, per group: intrinsics 24.1 forward, 17.4 inverse; asm leaves
  kernels (new) 22.8 and 15.8. `inverse_radix4` (intrinsics, odd lg top) 21.4 per column.
- Per 8 vectors: `forward_radix8_sparse` 23, `forward_radix8_dense` 32, radix-2 pass 8 (store
  bound: 1 store per cycle) + `forward_identity` 9.4 + `forward` 14.5; `inverse_top` 44.
- Whole transforms: about 2.0 cycles per vector and radix-2 level from 2^10 to 2^24.
- Instructions (12 independent chains): vpaddd, vpminud, vpblendd 0.25 cycles; vpmulld, vpmuludq,
  vpsrlq, vbroadcastss m32, vmovdqa load 0.5; store 1.0. Pairs: vpmuludq + vpminud 0.633,
  vpsrlq + vpminud 0.60, vpaddd + vpminud 0.50. A multiply or shift costs more than its pipe
  count says when mixed with 4-pipe ops; the 4-pipe op count is only a floor.

### intrinsics (issue #4, round 1)

Against main (6a9c91d, with issue #3's round 2 asm), paired, 5 rounds, `kernel.py bench --base`,
all 80 gated cases, GCC 15.2, `tools/kernel.py` flags, 2026-10-11. Geomean new/base, then new ns
per call (new/base) by lg:

| Target (machine) | Geomean | forward 10 / 16 / 25 | inverse 10 / 16 / 25 | multiply 10 / 16 / 25 |
|---|---:|---|---|---|
| avx2 (`lc-bench`, Zen 3) | 0.999 | 525 (1.000) / 60736 (1.003) / 5.65e7 (1.004) | 498 (1.004) / 57402 (0.996) / 5.59e7 (1.003) | 2152 (1.004) / 2.14e5 (0.997) / 1.91e8 (0.998) |
| avx512 (`lc-intel`, Emerald Rapids) | 0.894 | 502 (0.873) / 60023 (0.863) / 5.98e7 (0.845) | 511 (0.887) / 56624 (0.831) / 6.24e7 (0.848) | 2207 (0.961) / 2.16e5 (0.896) / 2.22e8 (0.826) |
| zen5 (`lc-zen5`, EPYC 9B45) | 0.727 | 288 (0.712) / 31621 (0.667) / 3.33e7 (0.762) | 261 (0.658) / 28179 (0.603) / 3.01e7 (0.663) | 1280 (0.769) / 1.18e5 (0.701) / 1.10e8 (0.740) |
| arm (`lc-arm`, Neoverse V2; lg <= 18) | 0.176 | 1058 (0.169) / 124766 (0.164) / - | 1023 (0.149) / 119537 (0.153) / - | 4855 (0.198) / 4.75e5 (0.180) / - |

avx2: the selected code is unchanged (asm); only the bench binary changed. arm: against the scalar
variant, which main selected there; absolute times at 2^25 not paired. Cases above 1.01: Zen 3
multiply/9 and multiply_dense/9 1.020 (asm unchanged; seen in every run: code layout, guess),
Emerald Rapids inverse/6 1.034 and multiply_dense/6 1.024, Zen 5 inverse/7 1.027 (the asm, called
through `by_length`).

Variants against the selected ones in the same run (`@` cases, median of 5, ns ratio), lg
10 / 16 / 22:

| Machine | Variant / selected | forward | inverse | multiply |
|---|---|---|---|---|
| Zen 3 | intrinsics_avx2 / avx2 | 1.09 / 1.14 / 1.17 | 1.19 / 1.18 / 1.16 | 1.18 / 1.21 / 1.22 |
| Emerald Rapids | avx2 / intrinsics_avx512 | 1.14 / 1.16 / 1.09 | 1.14 / 1.20 / 1.17 | 1.04 / 1.12 / 1.12 |
| Emerald Rapids | intrinsics_avx512vl / intrinsics_avx512 | 1.25 / 1.30 / 1.27 | 1.21 / 1.28 / 1.26 | 1.14 / 1.24 / 1.22 |
| Zen 5 | avx2 / intrinsics_avx512 | 1.40 / 1.50 / 1.52 | 1.53 / 1.66 / 1.71 | 1.30 / 1.41 / 1.41 |
| Zen 5 | intrinsics_avx512vl / intrinsics_avx512 | 1.64 / 1.83 / 1.84 | 1.72 / 1.86 / 1.89 | 1.54 / 1.73 / 1.74 |

- zmm against the same code on ymm (32 registers): 1.2-1.3x on Emerald Rapids, 1.5-1.9x on Zen 5
  (four 512-bit pipes).
- Zen 3: the intrinsics on Avx2 are 1.06-1.22x slower than the scheduled asm, most in the
  products (the asm schedules each bottom group as one dataflow graph, and merges tiles).
- Emerald Rapids, earlier run: Avx512Vl (32 ymm) beat Avx2 (16 ymm) by 5%; on Zen 5 they compile
  alike. The bench times Avx512Vl only there.
- Neoverse V2, ns (ns per coefficient): forward/16 124766 (1.90), inverse/16 119537 (1.82),
  multiply/16 474831 (7.25). Its Shoup products use SVE2 `umulh` (issue #2, round 2).

### Convolution memory (issue #38, round 1)

`kernel.py bench --base`, 7 rounds, GCC 15.2, each target's flags, 2026-10-11. Base: main 7544480
with this round's bench.cpp (its `convolution` case, which maps fresh memory per call there).
`convolution/<lg>`: construct, multiply and destroy a Convolution of 2^lg / 2 and 2^lg / 2
coefficients; `multiply/<lg>`: multiply() alone. ns, medians:

| lg | Zen 3: multiply / convolution / main's convolution | Emerald Rapids | Zen 5 | Neoverse V2 |
|---:|---|---|---|---|
| 6 | 106 / 109 / 56077 | 142 / 144 / 55626 | 89 / 91 / 32105 | 223 / 231 / 23476 |
| 8 | 478 / 490 / 56365 | 474 / 478 / 56043 | 330 / 332 / 32346 | 1033 / 1038 / 24101 |
| 10 | 2141 / 2150 / 57938 | 2207 / 2212 / 60115 | 1278 / 1280 / 34040 | 4841 / 4844 / 28717 |
| 12 | 9895 / 9934 / 65758 | 10114 / 10136 / 68518 | 5694 / 5705 / 38585 | 22604 / 22611 / 46682 |
| 14 | 46216 / 46273 / 102850 | 47870 / 47874 / 106343 | 26284 / 26323 / 59385 | 104608 / 104584 / 130249 |
| 16 | 212962 / 212595 / 270509 | 216431 / 216395 / 283191 | 117797 / 117946 / 150698 | 474581 / 474386 / 504992 |
| 20 | 4.26e6 / 4.27e6 / 4.56e6 | 4.93e6 / 4.92e6 / 5.37e6 | 2.35e6 / 2.35e6 / 2.52e6 | 9.59e6 / 9.59e6 / 9.98e6 |
| 22 | 1.88e7 / 1.88e7 / 2.08e7 | 2.19e7 / 2.19e7 / 2.37e7 | 1.05e7 / 1.05e7 / 1.20e7 | 4.26e7 / 4.26e7 / 4.36e7 |

- Construction now costs 2-12 ns (convolution / multiply 1.01-1.04 at 2^6, 1.00-1.015 from
  2^9), against 23-56 us before: the first touch of a fresh 2 MiB huge page.
- multiply/<lg> against main: 2^6..2^16 unchanged within noise, except Zen 3 at 2^7 (below);
  2^17..2^25 faster, tables kept per thread instead of built per call: Zen 3 0.982-0.992,
  Emerald Rapids 0.959-0.985, Zen 5 0.983-0.990, Neoverse V2 0.987-0.992. multiply_dense alike.
- Zen 3 at 2^7 moves with the memory's placement, not the code: the same binary with the region
  reused or mapped fresh per case gave multiply/7 0.93 and multiply_dense/7 1.07 (11 rounds); main
  against this round gave 1.02-1.04 and 1.05-1.10 in three runs. Round 2 of issue #3 logged the
  same swings at 2^7 and 2^8 from unrelated changes.

Item 2, the product in caller memory (`explore/probe_memory.cpp`, exploration folder): a product
of two factors of 2^lg / 2 coefficients as the frontend runs it, from sources 16 bytes past a page
boundary, 5 rounds, ratio to multiply() alone. "owned": Convolution(n, m), copy f and g in,
multiply, copy the product out; "into": copy f into the caller's a, Convolution(n, m, a), copy g
in, multiply.

| lg | Zen 3 owned / into | Emerald Rapids | Zen 5 | Neoverse V2 |
|---:|---|---|---|---|
| 6 | 1.138 / 1.071 | 1.065 / 1.004 | 1.105 / 1.044 | 1.062 / 1.030 |
| 10 | 1.042 / 1.021 | 1.027 / 1.012 | 1.043 / 1.021 | 1.021 / 1.012 |
| 14 | 1.054 / 1.025 | 1.054 / 1.026 | 1.055 / 1.032 | 1.017 / 1.011 |
| 18 | 1.063 / 1.018 | 1.121 / 1.050 | 1.048 / 1.017 | 1.027 / 1.017 |
| 22 | 1.123 / 1.065 | 1.110 / 1.017 | 1.109 / 1.065 | 1.029 / 1.004 |

The copy out (owned against into) cost 1.5-9% on x86, 0.6-3% on Neoverse V2. What remains of
"into" is the two copies in.

memcpy by destination offset from the source, modulo 4 KiB (same probe, sources 0, 16 and 64
bytes past a page; 5 rounds). The 4-20x of issue #38 did not reproduce here. Slowest against
fastest:
- Zen 3: at most 1.35x (2^10 words, offsets +-16).
- Emerald Rapids: offsets not a multiple of 64 bytes 1.3-1.55x at 2^10..2^12 words, 1.05-1.1x at
  2^14..2^16.
- Zen 5: destination 0 to 128 bytes past the source 1.3-1.7x at 2^14..2^16 words; offsets not a
  multiple of 64 1.1-1.6x at 2^10..2^12.
- Neoverse V2: no effect (within 3%).
- Fast on all four: the destination 64 or 128 bytes before the source, or 256 bytes or more past
  it, a multiple of 64 bytes.

### zmm intrinsics, issue #5 round 1

Against main (03c970b), paired, 5 rounds, `kernel.py bench --base`, the 80 gated cases, GCC 15.2,
`tools/kernel.py` flags, 2026-10-11. Geomean new/base: `lc-intel` (Emerald Rapids) 0.934, `lc-zen5`
(Zen 5) 0.934, `lc-arm` 0.984 (two runs: 0.990 with the asm statement on Neon, then 0.993 of that
without it). New ns per call (new/base):

| lg | EMR forward | EMR inverse | EMR multiply | Zen 5 forward | Zen 5 inverse | Zen 5 multiply |
|---:|---|---|---|---|---|---|
| 6 | 22 (1.000) | 26 (0.772) | 140 (0.989) | 18 (1.006) | 18 (1.000) | 89 (1.001) |
| 8 | 103 (0.999) | 100 (0.955) | 475 (1.000) | 71 (1.022) | 63 (1.006) | 250 (0.748) |
| 10 | 471 (0.940) | 487 (0.957) | 1868 (0.851) | 292 (1.013) | 268 (1.024) | 1061 (0.832) |
| 12 | 2342 (0.926) | 2393 (0.953) | 8902 (0.881) | 1346 (0.968) | 1289 (1.011) | 4924 (0.862) |
| 16 | 55721 (0.932) | 54226 (0.963) | 1.90e5 (0.875) | 29974 (0.956) | 28301 (1.000) | 1.03e5 (0.875) |
| 20 | 1.27e6 (0.935) | 1.21e6 (0.971) | 4.47e6 (0.904) | 6.28e5 (0.947) | 5.81e5 (0.995) | 2.12e6 (0.904) |
| 25 | 5.75e7 (0.950) | 6.14e7 (0.973) | 2.12e8 (0.970) | 3.22e7 (0.974) | 2.95e7 (0.987) | 1.01e8 (0.941) |

Below 2^9 (EMR forward), 2^10 (EMR multiply) and 2^8 (Zen 5) the asm runs, unchanged. Cases above
1.01: Zen 5 forward/8 1.022, forward/9 1.068, forward/10 1.013, inverse/7 1.015, inverse/10 1.024,
inverse/12 1.011. The inverse code at even lg is the same as main's instruction for instruction
(diffed): layout, a guess. forward/9: not the input bound of its first level (forward with
that level's reductions put back: equal, 134 ns both); not found.

Kernels on L1 data, `perf stat` on `lc-intel` (`micro.cpp`): cycles per call and share of the
p0 + p5 floor ((p0 + p5 uops) / 2). Forward group, 16 columns of 4 vectors: 511 (94%) -> 479.
Inverse group: 486 (96%) -> 473. zmm bottom forward, 16 pairs of groups: 932 (84%) -> 906 (84%).
Inverse: 715 (86%) -> 630 (95%). Product bottom, 32 pairs: 10200 (79%, already with `whole`) ->
8990 (90%).

### intrinsics, issue #4 round 2

Against main (2b576e8), paired, 7 rounds, `kernel.py bench --base`, the 80 gated cases, GCC 15.2,
`tools/kernel.py` flags, 2026-10-11, commit bd8e38d (before the 2^17 limit of the folded scale on
Emerald Rapids, Log). Geomean new/base: `lc-arm` (Neoverse V2) 0.877, `lc-zen5` (Zen 5) 0.942,
`lc-intel` (Emerald Rapids) 0.962. New ns per call (new/base):

| lg | V2 forward | V2 inverse | V2 multiply | Zen 5 forward | Zen 5 inverse | Zen 5 multiply | EMR forward | EMR inverse | EMR multiply |
|---:|---|---|---|---|---|---|---|---|---|
| 6 | 32 (0.988) | 34 (1.036) | 198 (0.890) | 18 (1.000) | 18 (1.000) | 89 (1.000) | 22 (1.000) | 27 (1.043) | 141 (1.001) |
| 8 | 162 (0.895) | 172 (0.912) | 885 (0.880) | 60 (0.846) | 62 (0.978) | 233 (0.940) | 89 (0.868) | 97 (0.971) | 380 (0.803) |
| 10 | 897 (0.884) | 916 (0.894) | 4167 (0.879) | 258 (0.889) | 249 (0.922) | 1010 (0.956) | 429 (0.917) | 461 (0.952) | 1856 (0.991) |
| 12 | 4611 (0.876) | 4625 (0.889) | 19412 (0.874) | 1228 (0.905) | 1184 (0.915) | 4676 (0.949) | 2211 (0.944) | 2304 (0.969) | 8754 (0.984) |
| 16 | 1.06e5 (0.870) | 1.06e5 (0.886) | 4.07e5 (0.871) | 27016 (0.903) | 26053 (0.921) | 96260 (0.938) | 53383 (0.953) | 52268 (0.972) | 1.86e5 (0.983) |
| 20 | 2.26e6 (0.866) | 2.27e6 (0.890) | 8.31e6 (0.871) | 5.65e5 (0.904) | 5.42e5 (0.935) | 1.94e6 (0.936) | 1.21e6 (0.961) | 1.17e6 (0.979) | 4.27e6 (0.970) |
| 25 | 9.51e7 (0.866) | 9.91e7 (0.886) | 3.38e8 (0.877) | 3.03e7 (0.938) | 2.86e7 (0.970) | 9.69e7 (0.964) | 5.47e7 (0.957) | 5.74e7 (0.952) | 2.01e8 (0.978) |

Cases above 1.005: Zen 5 inverse/7 1.009 (the asm, unchanged); Emerald Rapids inverse/6 1.043
(IQR 0.996-1.102), multiply/7 1.006 (the asm), and every case at 2^18: inverse 1.037, multiply
1.054, multiply_dense 1.037, convolution 1.039 (the folded scale; off there from 2^18 now: those
cases 0.938-0.946 against bd8e38d, 7 rounds, 2^19 inverse 1.016, the rest of 2^16..2^19 within
0.5%); Neoverse V2 inverse/6 1.036 (one radix-2 level and two bottom groups; not found).

Kernels on L1 data (`micro.cpp` in the exploration folder; 1024 words per array: groups of 4
quarters of 256 words, 32 bottom groups), main -> this round, the same run, 2 rounds each.
Neoverse V2, cycles (`perf stat`): forward group 841 -> 770, inverse group 831 -> 762, identity
groups 665 -> 669 and 673 -> 605, bottom forward 1281 -> 1157, inverse 825 -> 760, product bottom
8608 -> 7666. Zen 5, ns: 63.6 -> 56.5, 60.9 -> 56.6, 38.0 -> 38.1 and 35.5 -> 34.2, bottom forward
137 -> 127, inverse 91.3 -> 92.4, product bottom 630 -> 629. Emerald Rapids, cycles: 503 -> 491,
502 -> 480, 306 -> 306 and 281 -> 283, bottom forward 921 -> 825, inverse 627 -> 652 (the same
instructions as main, diffed), product bottom 4413 -> 4414. Main's forward group alone moved
between 790 and 841 cycles on Neoverse V2 across builds of the probe (code placement, a guess).

Neoverse V2 instruction mixes (`v2mix.cpp`, exploration folder): blocks of independent `mul`
(M, the V0/V2 pipes) and `add` (A, any of V0-V3), instructions per cycle by the order of one
repeated pattern: A 4.0, M 2.0, MA 4.0, MAAA 4.0, MMAA 4.0, MAA 3.0, AMA 3.0, MMAAAA 3.1, MAAAAA
3.5, MAAAAAAA 3.6, MMMAAAAA 3.6. So the order alone moves mixed code between 3 and 4 per cycle,
even with 1 multiply in 8 (guess: pipes chosen at dispatch, and mixed latencies 4 and 2 sharing
V0/V2 write ports; no simple steering rule fit all patterns). The forward group loop's own
instruction sequence, with independent registers: 9.9 cycles per column, against 12.4 in the loop
(main). `str q` takes a V0/V1 uop for its data (LLVM's Neoverse V2 model), and `ld1r` a V uop.

### zmm intrinsics, issue #5 round 2

Against main (d63b99f), paired, 7 rounds, `kernel.py bench --base`, the 114 gated cases (80 ops
and 34 convolution cases), GCC 15.2, `tools/kernel.py` flags, 2026-10-11. Geomean new/base:
`lc-intel` (Emerald Rapids) 0.9628, `lc-arm` (Neoverse V2, 5 rounds) 0.9946, `lc-zen5` (against
e452c18, whose ntt bench binary is byte-identical to d63b99f's) 0.9923. Emerald Rapids, new ns
per call (new/base):

| lg | forward | inverse | multiply | multiply_dense |
|---:|---|---|---|---|
| 6 | 22 (1.005) | 27 (0.978) | 132 (0.938) | 122 (0.920) |
| 8 | 86 (0.965) | 95 (0.971) | 375 (0.985) | 384 (0.982) |
| 10 | 418 (0.973) | 438 (0.948) | 1800 (0.970) | 1828 (0.974) |
| 12 | 2132 (0.965) | 2214 (0.957) | 8604 (0.986) | 8706 (0.990) |
| 16 | 51628 (0.973) | 51120 (0.974) | 1.84e5 (0.986) | 1.86e5 (0.987) |
| 18 | 2.35e5 (0.970) | 2.40e5 (0.983) | 8.46e5 (0.940) | 8.65e5 (0.947) |
| 20 | 1.17e6 (0.968) | 1.14e6 (0.985) | 4.02e6 (0.940) | 4.10e6 (0.942) |
| 22 | 5.38e6 (0.971) | 5.21e6 (0.976) | 1.78e7 (0.928) | 1.81e7 (0.928) |
| 24 | 2.38e7 (0.967) | 2.34e7 (0.984) | 7.94e7 (0.871) | 8.08e7 (0.873) |
| 25 | 4.85e7 (0.880) | 5.23e7 (0.906) | 1.74e8 (0.864) | 1.74e8 (0.859) |

Cases above 1.0 on Emerald Rapids: forward/6 1.0045 (IQR 0.996-1.027; the asm, unchanged).
Convolution cases 0.90-0.99 (2^6: 0.90, the intrinsics). Neoverse V2: no case above 1.008;
forward, inverse, multiply at 2^10 / 2^16 / 2^25 0.99-1.0. Zen 5: inverse 0.95-0.97 from 2^9 to
2^17 (2^10 0.952, 2^16 0.965), 0.97-1.0 above; forward 0.98-1.0, multiply 0.99-1.0. Zen 5 cases
above 1.005: multiply/8 and multiply_dense/8 1.008-1.009, multiply/11 and multiply_dense/11 1.009,
forward/22 1.007, multiply/22 1.008, convolution/8 1.011, convolution/10 1.016,
convolution_into/10 1.023, convolution/11 and convolution_into/11 1.010-1.011, inverse/7 1.009
(the asm). In a probe (`micro.cpp`, 5 rounds) this round's code ran multiply and
construct-and-multiply at 2^8-2^11 as fast as or faster than builds without the unshifted q or
with a and b interleaved per group: code placement, a guess. A run while another agent built on
`lc-zen5` had IQRs up to 1.35; not used.

Page size (`pages.cpp`, exploration folder): multiply() with a and b in fresh 4 KiB-page mappings
(`MADV_NOHUGEPAGE`) against 2 MiB pages, ratio of times, 9 rounds, Emerald Rapids: 2^16 0.998,
2^17 0.978, 2^18 0.955, 2^19 0.938, 2^20 0.947, 2^21 0.939, 2^22 0.940, 2^23 0.943, 2^24 0.948,
2^25 0.965 (2^6..2^15 within 1.5%); forward 2^19..2^25 0.94-0.97, inverse 2^20..2^25 0.94-0.99.
Zen 5: 0.99-1.0 to 2^22, 1.03 from 2^23. The physical layout decides it, and it is not stable:
`pfn.cpp` (pagemap) on `lc-intel` (1.2 GB free, 12.9 GB page cache) found 31 of 2047 neighbouring
pages of a fresh 8 MiB mapping physically adjacent, and 11553 of 16383 in a 64 MiB one. In
`kernel.py bench` with 4 KiB product pages, multiply/18-25 ran 0.94-0.99 but the convolution cases,
which reuse the 2^25 product's region, 0.99-1.01. `perf stat` at 2^20: L2 misses equal (518k per
call both), demand data misses 32k against 22k, 18.6M against 17.5M cycles.

L2 sets (`stride.cpp`): one forward radix-4 pass over 4 quarters of Q words, ns per word,
Emerald Rapids, 11 rounds. Quarters Q apart in 2 MiB pages / Q + 1024 words apart (4 KiB more) /
4 KiB pages / 2 MiB pages with `prefetcht0` 512 words ahead per quarter:

| Q (words) | apart Q | Q + 1024 | 4 KiB pages | prefetch |
|---:|---:|---:|---:|---:|
| 2^17 | 0.148 | 0.139 | 0.147-0.154 | 0.137 |
| 2^19 | 0.164-0.172 | 0.149-0.150 | 0.150 | 0.134 |
| 2^21 | 0.168-0.183 | 0.161-0.168 | 0.161-0.182 | 0.134 |

Other spacings at Q = 2^19: Q + 16 words 0.246, Q + 16384 0.156, Q + 32784 0.179. Prefetch 128 to
1024 words ahead: 0.134-0.138; `prefetcht1` 0.141-0.149; `prefetchw` as `prefetcht0`. So 4
streams in one L2 set at each column (index bits 6-16 equal: quarters 128 KiB apart or more in
2 MiB pages) cost 6-13%, and the prefetch more than recovers it. Zen 5: 0.062-0.073 plain; with
the prefetch 6% slower at Q = 2^15, 8% faster at 2^21.

Whole ops with the prefetch (`micro.cpp`, ns, 3 rounds, Emerald Rapids), from quarters of 2^11,
2^13, 2^15, 2^17 words: 2^11 cost multiply/14 1% and multiply/16 2.5%; 2^15 gained the most at
2^19-2^25 (2^17: multiply/20 0.98 against 0.96, /24 0.925 against 0.91). Radix-2 levels from
halves of 2^19 words against 2^20: inverse/20 1.21 against 1.18 ms, multiply/20 4.17 against
4.23 ms. Distance
256 or 1024 words against 512: within 1% (2^25 forward and inverse 2-3% slower at 256).
`prefetchw` on the write-only streams: multiply/18 -1.5%, /19 -2.8%, /21 -1.5%, /24 -2%, /25 -2.2%.

Kernels on L1 data, `perf stat` cycles, Emerald Rapids, 64 columns of groups of quarters of 32
words: a call per group 2145, a loop of pipelined groups 2120, two columns side by side 2008,
with the unshifted q too 1919; one group of 64 columns 1810. Inverse: 2055, 2100, 1946, 1910;
1840. Quarters of 128 words: 1860 per call, 1873 in a loop (kept: no gain, no loss). Unshifted q
in the bottoms: forward 820 -> 815-827 (p0 + p5 uops 1464 -> 1415), inverse 626-667 -> 598-633,
products 4430 -> 4377 per 32 groups.

### Squares and middle products (issue #55)

`kernel.py bench ntt --base` main a536462, 7 rounds, GCC 15.2, each target's flags, 2026-10-12;
ratios between cases of the new binary, median of the per-round ratios. `square/<lg>`:
`square(f, out)`, f of 2^lg / 2 words, out of 2^lg - 1; `square@multiply`: `multiply(f, f, out)`
on the same spans. `middle_product/<lg>`: f of 2^lg words, g of 2^lg / 2, out of 2^lg / 2 + 1 (a
transform of 2^lg); `middle_product@multiply`: what `fps::middle_product` ran before, `multiply(f,
g, out)` into 2^lg words (a transform of 2^(lg + 1)), then the 2^lg / 2 + 1 words moved down.

| lg | square / @multiply: Zen 3, EMR, Zen 5, V2 | middle_product / @multiply |
|---:|---|---|
| 6 | 0.85, 0.91, 0.89, 0.86 | 0.55, 0.58, 0.59, 0.52 |
| 7 | 0.96*, 0.82, 0.84, 0.84 | 0.54, 0.64, 0.77, 0.52 |
| 8 | 0.87*, 0.86, 0.89, 0.80 | 0.50, 0.51, 0.51, 0.48 |
| 9 | 0.79, 0.84, 0.89, 0.79 | 0.53, 0.48, 0.52, 0.50 |
| 10 | 0.79, 0.82, 0.84, 0.79 | 0.50, 0.49, 0.49, 0.48 |
| 12 | 0.77, 0.78, 0.81, 0.77 | 0.48, 0.49, 0.50, 0.48 |
| 14 | 0.76, 0.78, 0.79, 0.75 | 0.49, 0.49, 0.50, 0.48 |
| 16 | 0.74, 0.77, 0.76, 0.75 | 0.48, 0.50, 0.49, 0.48 |
| 18 | 0.73, 0.73, 0.76, 0.73 | 0.50, 0.50, 0.50, 0.49 |
| 20 | 0.73, 0.72, 0.75, 0.73 | 0.49, 0.52, 0.50, 0.49 |
| 22 | 0.72, 0.72, 0.70, 0.72 | 0.48, 0.52, 0.48, 0.49 |

\* Zen 3 at 2^7 and 2^8: ratio of medians from the run of the pair bottom against `GroupBottom`
(below); the table's run had `GroupBottom` with b = a there, 0.91 and 0.95.

- square: 0.70-0.81 of `multiply(f, f, out)` from 2^12 on every machine, 0.70-0.72 at 2^22; three transforms become two plus the leaf products, so 2/3 is the floor (guess: the leaf
  products and the first and last levels' traffic keep it above). 2^6-2^11 0.77-0.96, the
  shortest lengths inside the spread of code layout (up to 25% on short products, issue #93):
  the unchanged `product/6` and `product/7` moved 1.06 and 1.04 on Zen 3 in the same run.
- middle_product: 0.45-0.52 of the old flow from 2^8, half the transform length; 2^6-2^7
  0.52-0.77 (Zen 5 2^7 0.77: its multiply runs the asm below 2^8, and the new transform is 2^7).
- avx2 bottom of a square at 2^7 and 2^8, where products run `GroupBottom`: `GroupBottom` with b =
  a (its kernels read both batches before they write, so a's forward half ran twice) against
  `PairSquareBottom`, Zen 3, 9 rounds, paired: the pair bottom took 0.96 (2^7) and 0.93 (2^8) of
  the time, the unchanged `square@multiply` 0.91 and 1.01 in the same run (layout).
  The pair bottom runs at every length.

### Products from spans (issue #38, round 2)

`kernel.py bench ntt --base` against main 7f57131, 7 rounds, GCC 15.2, each target's flags,
2026-10-11, the round's final code; ratios between cases of the new binary, median of the
per-round ratios. `product/<lg>`: `multiply(f, g, out)`, f and g of 2^lg / 2 words, out of
2^lg - 1; `product@copy`: a Convolution, f and g copied in, the product copied out (what
`fps::raw::mul` did); `product@into`: f copied into the caller's a, `Convolution(n, n, a)`, g
copied in; `multiply`: `Convolution::multiply()` alone, factors already in place; `product_cut`:
out of 2^lg / 2 words. Spans 64 bytes past a page boundary.

| lg | product / @copy: Zen 3, EMR, Zen 5, V2 | product / @into | product / multiply | product_cut / product |
|---:|---|---|---|---|
| 6 | 0.98, 0.95, 1.01, 1.04 | 1.10, 1.00, 1.07, 1.08 | 1.19, 0.94, 1.12, 1.13 | 0.89, 0.91, 0.93, 0.90 |
| 8 | 1.03, 1.00, 0.98, 0.99 | 1.05, 1.02, 1.01, 1.00 | 1.10, 1.05, 1.04, 1.03 | 0.93, 0.96, 0.96, 0.97 |
| 9 | 0.99, 1.00, 0.99, 0.98 | 0.99, 1.01, 1.02, 0.99 | 1.04, 1.03, 1.04, 1.01 | 0.98, 0.97, 0.96, 0.98 |
| 12 | 0.96, 0.97, 0.98, 0.98 | 0.98, 1.00, 1.01, 0.99 | 1.01, 1.02, 1.03, 1.00 | 0.98, 0.98, 0.98, 0.98 |
| 16 | 0.98, 0.97, 0.98, 0.98 | 1.01, 0.99, 1.03, 0.99 | 1.05, 1.01, 1.06, 1.00 | 0.98, 1.00, 0.98, 0.98 |
| 20 | 0.96, 0.94, 0.97, 0.97 | 1.00, 1.00, 1.01, 0.99 | 1.03, 1.07, 1.05, 1.00 | 0.98, 0.98, 0.99, 0.99 |
| 22 | 0.94, 0.95, 0.98, 0.98 | 0.99, 1.01, 1.03, 1.01 | 1.06, 1.07, 1.12, 1.02 | 0.97, 0.98, 0.98, 0.99 |

- From 2^9 to 2^22, against the copy flow: Zen 3 0.93-1.00, Emerald Rapids 0.94-1.00, Zen 5
  0.95-0.99, Neoverse V2 0.97-0.98. Against the into flow 0.98-1.04, with a result of n + m - 1
  words instead of length + 16. At 2^6-2^8 up to 1.10 slower than the into flow (Zen 3 2^6).
  The cut product: 0.96-1.00 of the whole one, 0.89-0.97 at 2^6-2^8.
- Against multiply() alone, 1.00-1.12: the extra cost is that of the into flow too (both read the
  factors from separate memory and write it), not of copies. Neoverse V2 1.00-1.02. Guess, not
  measured: the separate 4 KiB-page lines of f, g and out against the region's huge pages.
- Placing a and b away from the spans' page offsets (`place`): before it, Zen 3 product /
  multiply at even lg 2^14-2^22 was 1.08-1.11 (a at offset 0 and out at 64: the inverse level's
  loads 2 columns after stores 4 KiB apart), after it 1.04-1.06; odd lg and the other machines
  within 1%.

Gated cases against main, same run (the in-place paths; their code and kernels are main's):
Zen 3 1.0015, Emerald Rapids 0.9999, Zen 5 0.9987, Neoverse V2 0.9996. Cases above 1.02: Zen 3
multiply_dense/7 1.025, convolution_into/7 1.021; Emerald Rapids convolution_into/18 1.028; Zen 5
convolution/7 1.115. In the run before (5 rounds, the same in-place code): Zen 3 multiply/7 1.054,
multiply_dense/7 1.083, convolution*/7 1.07-1.09; Zen 5 convolution*/7 1.18-1.19. Placement,
below.

Zen 3 placement at 2^6-2^9 (`probe_small.cpp`, exploration folder, same-run medians of 5):
main's 2^7 product took 216 ns in one build of the probe and 232 ns in another that only moved
main() into a noinline function; main's top-level code inside this round's tree ran 212-232 ns
across builds, the round's first top levels 244-252 ns. Not the stack (an alloca of 0-4032 bytes
before the run: flat within 1%), not function alignment (`-falign-functions=64`: the gap
remained), not 4K aliasing (`ld_blocks.address_alias` about 1 per product on Emerald Rapids, the
same code).

Instruction counts, `perf stat` on `lc-intel`, a 2^7 product (avx2 code) in place through the
span levels, per product: `-march=x86-64-v4`, main 3176 instructions and 871 cycles, the round's
first unified levels 3444 and 943 (two 64-bit divisions per level, one function holding every
block count); `-march=x86-64-v3`, main 3161 and 868, with shifts and one function per count 3241
and 923. By symbol (cycles), main's unchanged bottom kernels took 54 of the 55 cycles more; front
end and store forwarding were better than main's (`idq.mite_uops` 116 -> 30 per product).

### Tuned selection (issue #78)

The compiled tables (`tools/tune.py ntt`, 7 samples of 1 ms per candidate and point, 1% margin
from main's choices), 2026-10-12. Lengths where a candidate other than zmm is chosen; zmm
(intrinsics<Avx512>) everywhere else. tune.cpp's geometric mean of time / fastest at those
lengths in brackets (asm, ymm intrinsics, zmm):

| Op | Emerald Rapids (lc-intel) | Zen 5 (lc-zen5) |
|---|---|---|
| forward | 2^6 asm (1.000, 1.022, 2.06); 2^7 zmm (1.177, 1.072, 1.000) | 2^6 asm (1.000, 1.117, 1.272) |
| inverse | 2^6 ymm (1.249, 1.000, 1.328); 2^7 asm (1.000, 1.068, 1.251) | 2^6 asm (1.000, 1.109, 1.278) |
| multiply | 2^6 ymm (1.089, 1.000, 1.137); 2^7 asm (1.000, 1.089, 1.203) | 2^6-2^7 asm (2^7: 1.000, 1.160, 1.117) |
| product | 2^6 ymm (1.017, 1.000, 1.135); 2^7 asm (1.000, 1.060, 1.156) | 2^6-2^7 asm (2^7: 1.000, 1.163, 1.127) |

Against main's `by_length`: Emerald Rapids forward/7 zmm (was the asm), inverse/7 the asm (was
ymm); Zen 5 inverse/7 zmm (was the asm). avx2, sse, scalar and arm have one candidate each.

`kernel.py bench --base` against main bfd491b, 11 rounds, the final code: geomean of the gated
cases, new/main.

| Unit | Zen 3 | Emerald Rapids | Zen 5 | Neoverse V2 |
|---|---:|---:|---:|---:|
| ntt | 1.0004 | 1.0002 | 0.9996 | 0.9996 |
| frontend | 0.9991 | 1.0011 | 0.9978 | 1.0013 |
| small | 1.0001 | 1.0004 | 0.9996 | 0.9975 |

(frontend and small on Emerald Rapids, Zen 5 and Neoverse V2: one commit earlier, with the same
code for those targets.) Cases past 1.5%: Emerald Rapids forward/7 0.896 (the gain), inverse/7
1.027 (Log, Open), forward/8, inverse/8, multiply/8, multiply_dense/8 1.015-1.024 (zmm in both),
convolution_into/7 1.025; Zen 3 forward/9, multiply/7, convolution/7-8 1.018-1.026, product_cut/7
0.919 (the asm in both: placement); Zen 5 inverse/25 1.052 (zmm in both; 0.99-1.01 in the run
before), convolution/10 1.019, convolution/11 0.978. The frontend's NTT shapes on Emerald Rapids
multiply/7 and /8 1.036 and 1.029 (the variant is main's at both).

### Leaf-form ops (issue #49)

`lc-bench` (EPYC 7B13), GCC 15.2, `-O2 -march=x86-64-v3`, 3 rounds, 2026-10-12, the round's code
(`bench_leaves.cpp`). ns per word; forward and inverse for scale (bench.cpp):

| lg | forward | inverse | leaf_product | leaf_product_add | forward_values | inverse_values | inverse_to | inverse_to@copy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.36 | 0.33 | 0.97 | 1.06 | 0.89 | 0.83 | 0.48 | 0.41 |
| 8 | 0.39 | 0.41 | 0.94 | 1.00 | 0.85 | 0.83 | 0.52 | 0.49 |
| 10 | 0.51 | 0.49 | 0.93 | 0.99 | 0.95 | 0.88 | 0.54 (region) | 0.57 |
| 12 | 0.64 | 0.61 | 0.94 | 0.99 | 1.08 | 1.00 | 0.67 | 0.70 |
| 16 | 0.92 | 0.88 | 0.94 | 1.01 | 1.36 | 1.27 | 1.04 | 1.07 |
| 20 | 1.20 | 1.15 | 0.95 | 1.01 | 1.63 | 1.54 | 1.33 | 1.43 |
| 22 | 1.33 | 1.29 | 0.95 | 1.02 | 1.77 | 1.68 | 1.58 | 1.72 |
| 23 | 1.43 | 1.45 | 0.93 | 0.98 | 1.86 | 1.78 | 1.68 | 1.83 |

- Leaf products: 0.93-0.97 ns per word at every length (about 26 cycles per leaf; muls are 24 per
  leaf on 2 pipes, loads 25 on 3 ports: the floor is near 13-15 cycles, a guess). The frontend's
  scalar loop took 6.9.
- Values form: the split adds 0.4-0.45 ns per word to the forward (about 97 cycles per block of
  64 words; 19 Shoup products, 2 x 8 x 8 transposes: about 60 cycles of 4-pipe ops, a guess),
  the join as much to the inverse. The frontend's scalar loop took 7.6-10.
- Inverse into other memory, against F copied, the inverse in place and the coefficients copied
  out (`@copy`), same run: Zen 3 0.92-0.97 from 2^10 to 2^23; Emerald Rapids (`lc-intel`) 0.90
  (2^14), 0.89 (2^18), 0.84 (2^22); Zen 5 (`lc-zen5`) 0.95 (2^14), 0.95 (2^18), 1.08 (2^22),
  0.97 (2^23). Below 2^10 it was slower (Emerald Rapids 70 against 37 ns at 2^6: the region's
  take and give, `place`, the span top level), so to 2^10 F is now copied into a stack buffer and
  the inverse runs in place there: 1.19 (2^6), 1.17 (2^7), 1.07 (2^8), 1.02 (2^9) of `@copy` on
  Zen 3, 1.06, 0.99, 1.05, 1.00 on Emerald Rapids.

Frontend (`kernel.py bench frontend --base d16ca08`, 5 rounds, each target's machine), new/base:

| case, lg 10 / 14 / 18 / 22 | Zen 3 | Emerald Rapids | Zen 5 | Neoverse V2 |
|---|---|---|---|---|
| values_product (leaf form) | 0.14 / 0.14 / 0.14 / 0.14 | 0.19 / 0.19 / 0.19 / 0.19 | 0.14 / 0.14 / 0.14 / 0.14 | 0.29 / 0.29 / 0.29 / 0.29 |
| to_values_pointwise | 0.13 / 0.17 / 0.20 / 0.22 | 0.21 / 0.24 / 0.29 / 0.35 | 0.13 / 0.15 / 0.17 / 0.19 | 0.25 / 0.29 / 0.33 / 0.38 |
| to_coefficients_pointwise | 0.11 / 0.13 / 0.17 / 0.18 | 0.17 / 0.19 / 0.24 / 0.28 | 0.10 / 0.11 / 0.13 / 0.15 | 0.20 / 0.24 / 0.28 / 0.33 |
| to_coefficients | 1.01 / 0.94 / 0.96 / 0.88 | 1.00 / 1.08 / 0.92 / 0.92 | 1.00 / 0.94 / 0.95 / 1.06 | 0.97 / 0.97 / 0.98 / 0.98 |
| add_coefficients | 1.04 / 1.01 / 0.98 / 0.98 | 1.07 / 0.97 / 1.03 / 0.92 | 1.09 / 0.98 / 1.00 / 1.10 | 1.03 / 1.00 / 1.00 / 1.02 |

The lg 10 rows of to_coefficients and add_coefficients ran the region path; they now copy F on
the stack (the backend's 2^10 above). `f * g` at 32 x 32 (`multiply/6`, frontend bench.cpp, whose
calls are main's) 1.13 on Zen 3 (152 -> 172 ns), 1.00 on the other machines: code placement
(#81, #93: main's own binary moves over 136-169 ns with the code shifted).

### Zen 5 products at 2^9-2^12 and code layout (issue #101)

`lc-zen5` (EPYC 9B45), GCC 15.2, `-O2 -march=znver5`, 2026-10-12. Commits: b9e53c4 (before #99),
529d281 (#99: square, middle product), ad1e1b4 (#100). The zmm product path (every function from
`ntt::multiply` down: `Ntt<Avx512>::multiply`, `multiply_tree`, `forward_group<4>`,
`WideBottom::multiply`, the `*_part` levels, the boundary functions) is the same instruction stream
in all three (objdump with addresses and jump targets normalized), except register allocation in
`ntt::multiply<tuned>` (region, `place`, dispatch: about 250 scalar instructions) and in the
in-place even-lg `multiply_tree<2>`. What differs is where each function sits.

ns, median over binaries of the median of 3-5 runs [min-max over binaries]; each binary the whole
TU shifted by 0-3584 bytes (512-byte steps, a top-level `.skip`), 8 shifts (ntt bench) or 4:

| case | b9e53c4 | 529d281 | ad1e1b4 |
|---|---|---|---|
| ntt multiply/11 (in place) | 2156 [2148-2160] | 2180 [2172-2191] | 2200 [2193-2213] |
| ntt product/11 | 2199 [2193-2220] | 2210 [2203-2269] | 2205 [2196-2223] |
| ntt product/9 | 526 [523-527] | 526 [523-577] | 524 [522-528] |
| frontend multiply/9 | 540 [538-550] | 554 [540-570] | 561 [549-590] |
| frontend multiply/11 | 2258 [2229-2282] | 2333 [2300-2392] | 2302 [2266-2336] |
| frontend multiply/12 | 4894 [4875-4903] | 4980 [4954-5021] | 4979 [4964-5027] |
| same, product path first in .text (ld `--section-ordering-file`): ntt multiply/11 | 2180 | 2168 | 2170 |
| ordered: ntt product/11 | 2248 | 2226 | 2230 |
| ordered: frontend multiply@kernel/12 | 4890 | 4956 | 4944 |

- The shifts move a binary by about 1% (one shift of 529d281: product/9 and square/9 +10%); the
  binaries differ by more than that and in the same direction at every shift, so it is the
  functions' relative layout, not their absolute addresses. With the zmm product path contiguous
  at the start of .text the three ntt benches agree within 0.5% (b9e53c4 is then no faster). The
  frontend's kernel calls stay 1-1.5% apart at 2^10 and 2^12 (its own code was not ordered).
- Not data placement: tables, f, g, out and a each at 64 page offsets (64-byte steps, the others
  fixed; `probe_place.cpp`): within 1% at 2^9-2^12 (one offset of out 8% at 2^9).
- The gate's 1.34-1.36 at 2^9 (PR #99): its two 21-round runs of the same binary (the build is
  deterministic) gave multiply/9 861 and 501 ns, base 499 and 499. In the slow run every product
  whose subtrees call `multiply_tree<4>` (odd lg 9, 11, 13; every lg from 14) took 90-150 ns more
  per call of it (2^9: 4 calls, +360 ns; 2^20: 680 calls, +74 us; 2^25: 21844 calls, +2.6 ms), and
  nothing else moved (even lg 10 and 12, square, transforms). Not reproduced: 16 more runs of
  `kernel.py bench` (each a fresh build) and 52 processes of fresh copies of the binary all ran
  multiply/9 at 499-513 ns. The only such run in the saved gates (all pooled runs, every target).
- Since issue #110 the gate flags two runs that disagree (a case's new/base moved more than 5%,
  out of both IQRs), times a third and judges on the pair that agrees best. A third gate run of
  PR 99 on lc-zen5 (2026-10-12) found the slow state on the base side this time: base multiply/9
  528 ns (499 before), new 501; ntt 1.0002. Judged on runs 2 and 3, PR 99's zen5 ntt is 1.0017,
  not 1.0291.

### Zen 3 short products and code layout (issue #93)

`lc-k68` (EPYC 7B13, Zen 3, the same CPU as `lc-bench`), GCC 15.2, `-O2 -march=x86-64-v3`, main
4af6aad, 2026-10-12. Probe `probe93.cpp` (exploration folder): `product` = `ntt::multiply(f, g,
out)`, `multiply` = `Convolution::multiply` in place, `square`, `fmul` = the frontend's `f * g`;
factors of 2^lg / 2 words. Four programs (MIX 0-3: the products alone; + `Transform` cases; +
`Values` and `middle_product`; + `exp`, `log`, `divmod`), each with the TU shifted 0-3584 bytes (8
shifts): 32 binaries, 3 runs each. ns: median over binaries [min-max], max/min.

| case | main | pins + placement (4 KiB anchor) | the same, the best of 8 orders (2 binaries) |
|---|---|---|---|
| product/6 | 124 [122-138] 1.13 | 134 [133-137] 1.03 | 130 [128-132] |
| product/7 | 256 [250-266] 1.07 | 264 [257-276] 1.07 | 261 [260-262] |
| product/8 | 535 [518-550] 1.06 | 520 [515-531] 1.03 | 515 [513-516] |
| multiply/6 | 108 [108-132] 1.23 | 128 [127-130] 1.02 | 114 [113-114] |
| multiply/7 | 230 [214-249] 1.17 | 222 [221-241] 1.09 | 223 [222-224] |
| multiply/8 | 486 [472-498] 1.06 | 497 [494-503] 1.02 | 495 [484-507] |
| square/6 | 107 [106-116] 1.09 | 110 [109-120] 1.10 | 112 [110-113] |
| square/7 | 238 [224-242] 1.09 | 247 [245-258] 1.05 | 242 [239-244] |
| square/8 | 454 [447-494] 1.11 | 485 [477-495] 1.04 | 485 [480-490] |
| fmul/6 | 161 [140-186] 1.33 | 178 [167-203] 1.22 | 181 [162-200] |
| 2^9-2^12, every case | 1.01-1.07 | 1.01-1.04 | |

- On main, at 2^7-2^8 the program (MIX) decides and the shift barely matters (multiply/7 214-216
  in MIX 2, 240-249 in MIX 1); at 2^6 both do.
- Not data: f, g and out each at 32 page offsets (128-byte steps): product/6-8 flat within 3% (one
  offset 7%); the heap moved 0-4080 bytes and the stack 0-4000 bytes before the cases: flat within
  1% (two programs 148 and 126 ns at every offset); `kStaticTables` page aligned: no change. Not
  the caller: 8 copies of the calling function 192 bytes apart in one binary, equal within 1%.
- Two causes, both needed. (1) GCC compiles the same COMDAT functions differently per TU: with
  `Transform` in the program, `product<GroupBottom>(const Product&)` is 7216 bytes instead of 7631,
  `Tree<PairBottom>::visit` 940 instead of 2750, `forward_radix4_from` is out of line: inlining
  depends on the TU's other callers. With every product function in one place at the same address
  in two programs (sorted sections, below; 64 KiB anchor) product/6 still took 150 and 125 ns.
  (2) Placement: with inlining pinned (always_inline or noinline on each function of the path) but
  not placed, product/6 131-161 ns across programs.
- Pinned and placed, a short product runs at the same speed in every program (2^6: 1.02-1.03
  spread, against 1.13-1.23), with the group's start 64-byte or 4 KiB aligned alike. But that one
  speed is a lottery: 8 orders of the group and 4 pinning variants each gave a fixed time per case,
  moving by up to 10% between variants (multiply/6 108-145, product/7 254-294) with the same
  instructions (traced, below). Geomean over the 9 cases at 2^6-2^8 against main's median:
  1.02-1.05. `kernel.py bench ntt` (5 rounds, gated cases 2^6-2^12, `bench_leaves.cpp` too) with the
  first pinned order: geomean 1.0013, convolution/6 1.12, convolution/7 0.82, product/7 1.06,
  inverse_values/6 1.13 (the kernels pinned noinline).
- noclone (needed for the sections, Log) costs squares 1-3%: pinned, not placed, with and without
  it: square/8 474 against 461 ns, square/9 897 against 886.
- The code one call runs (a ptrace single-stepper, `trace.hpp`): product/6 executes 1740-1830
  instructions, 1550-1640 distinct, in 146-157 64-byte lines; product/8 7400, 2290-2340 distinct,
  202-211 lines; square/8 5300, 2660-2710, 242-253 lines. An op-cache model of those lines (64
  sets by address bits 6-11, or 32 or 128 sets, or 32-byte windows; 8 ways; 6 or 8 ops per entry)
  does not order the variants by time. `lc-k68` and `lc-bench` have no PMU, so no counters (#94).

The log of rounds up to 2026-10-12 is in `notes/log-to-2026-10-12.md`; later rounds each have a
file in `notes/` (`head -qn1 notes/*.md` lists them).

## Sources

- lc-opt `lib/ntt` (ours): the avx2 variant and the generators, moved and renamed. Its sources:
  our QPoly explorations 008-010; Shoup multiplication (V. Shoup, NTL; D. Harvey, "Faster
  arithmetic for number-theoretic transforms", J. Symb. Comp. 60 (2014),
  https://arxiv.org/abs/1205.2926); splitting to x^k - w leaves (https://codeforces.com/blog/entry/142063,
  https://piskareviv.github.io/blog_aux_ntt_1/; read for ideas, no code used).
