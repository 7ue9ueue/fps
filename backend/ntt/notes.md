# backend/ntt

Number-theoretic transforms and products modulo P = 998244353. API: `ntt.hpp` (`Convolution`,
`Transform`, the leaf form). Variants: `scalar.hpp` (every target; the model), `intrinsics.hpp`
(`intrinsics<S>`, templates on any `backend/simd` struct S), `avx2.hpp` (lc-opt's `lib/ntt`).
`select.hpp` picks one per op and length: `avx2` on the `avx2` target; on `avx512` and `zen5`
`intrinsics<Avx512>` from 2^8-2^10 on (per op and CPU) and `avx2` below; `intrinsics<simd::Native>`
on the other targets. `memory.hpp`: `Convolution`'s memory, kept per thread. Tests: `test.cpp`;
benches: `bench.cpp`.
Generated, do not edit: `avx2_kernels.hpp` (`gen_avx2_kernels.py`), `avx2_product_kernels.hpp`
(`gen_avx2_product_kernels.py`).

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
- `Transform`: tables built once; forward and inverse in place on caller arrays (32-byte aligned,
  16 words of padding: the AVX2 forward kernels read 4 bytes past their last vector).

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

- Levels with quarters of at least 32 words run on S, one group at a time.
- Bottom level (quarters of 8 words, one leaf each): on S up to 8 lanes, one group at a time. On
  `Avx512` (16 lanes, `WideBottom`): two groups k, k + 1 per step; four `vpermt2d` turn their 64
  words into vectors [quarter t of group k, quarter t of group k + 1], so the butterflies are
  lane-wise with twiddles [w_k x 8, w_(k+1) x 8] (two `vpermd` of one 16-word table load each),
  and four more turn them back.
- Recursion: depth first; subtrees of at most 256 leaves (8 KiB per array) run level by level.
  Each subtree has its own bottom stage (avx2 now merges sibling subtrees into one tile; not done
  here). Top: a radix-2 level (odd lg - 3) or group 0, with the sparse forms for a factor of at
  most L/2 coefficients (a copy; group 0 reading half the input), and the scale fused into the
  last inverse level. No fused radix-8 top: slower everywhere it was tried (Log).
- Products, per bottom group: forward butterflies of a and b, leaf products, inverse butterflies,
  one store. Leaf products: x86 up to 8 lanes (`mul_even`): lc-opt's even/odd form, windows
  [w A, A] in a stack buffer read at offsets 1..9, each b_i broadcast once, 64-bit sums < 8 P^2,
  Montgomery reduction, `join_high`; group g + 1's forward half runs before group g's products,
  so the windows are not reloaded right after their stores. `Avx512`: the same form in registers
  for two groups at a time: window offset j of both halves is one `vpermt2d` of w A and A, b_i of
  both halves one `vpermd` of B; no stack buffer. Neon: `umull`/`umlal` (+`umull2`/`umlal2`) on
  the windows, `uzp1` for m, `umlal` for m P, `uzp2`. Scalar: one u64 sum per word.
- Lengths below 2^kWideMinLog run wholly on `Narrow<S>` (S, or `Avx512Vl` for `Avx512`): 2^8 on
  Emerald Rapids, 2^7 on Zen 4/5 (`__znver4__`, `__znver5__`). Before the zmm bottom level, zmm
  levels above ymm bottom stores were 1.25-1.33x slower at 2^7 and 2^8 on Emerald Rapids
  (`ld_blocks.store_forward` 1.8M against 12k: wide loads of narrow stores still in flight).
- GCC 15 at -O2 kept the fixed 4- and 8-trip loops of the leaf products rolled and did not inline
  the butterfly lambdas of the bottom level; `#pragma GCC unroll` and `[[gnu::always_inline]]`
  fix both (Log).
- `variant::by_length<lg, Small, Large>` (`select.hpp`): Small below 2^lg. On `avx512`: avx2
  below 2^9 (forward), 2^8 (inverse), 2^10 (multiply); on `zen5` below 2^8 for all three. From
  @avx2 against @intrinsics_avx512 in one binary, 5 rounds, intrinsics/avx2 at 2^6..2^9:
  Emerald Rapids forward 1.05 1.07 1.04 0.94, inverse 0.79 1.36 0.95 0.99, multiply 0.98 1.20
  1.13 1.11 (0.97 at 2^10); Zen 5 forward 1.17 1.09 0.95 0.76, inverse 1.11 1.06 0.79 0.80,
  multiply 1.08 1.16 0.95 0.90.

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

## Log

2026-10-11, claude (issue #3, round 1): moved lc-opt `lib/ntt` (`ntt.hpp`, `product.hpp`, both
generators and their output) here. Changes: namespaces `fps::ntt`; one `Convolution` picks the
product path (lc-opt `Product`) when it fits, else the convolution path; one templated `Tree`
for both recursions (bottom stage as a policy); `Product`'s `extra` bytes dropped; empty factors
allowed; errors throw. Generated asm: the same text as lc-opt's (diffed: only headers, include
paths and namespaces differ). New: variant `scalar` (the model; every target), `Transform`
(forward and inverse alone, leaf form), `select.hpp`. Tests: tables against the definition of
r(k), forward against the leaf form computed directly (lg <= 11) and against the scalar model
(lg <= 18), inverse undoes forward (every lg; scalar to 2^20), products against schoolbook
(sizes 0..40, random to 1500, lg <= 11) and random-point evaluation (every lg), both avx2 paths
at every lg >= 9, each asm kernel against a lane model with inputs at the edges of its range.
Pass at -O2 and ASan/UBSan: avx2, sse, scalar (`lc-amd`), scalar and arm (Mac).

2026-10-11, claude (issue #3, round 1, second change): two speed-ups on avx2.
- The scales (L/8)^-1 and (L/8)^-1 2^32 were modular powers per call, about 100 ns (inverse/6 was
  131 ns against forward/6 25 ns). Now a constexpr table.
- Even lg: the convolution path had a radix-2 pass (or a copy for a sparse factor), the first
  radix-4 groups in each half, and a separate `scale_radix2` pass; the product path already fused
  them. Now the convolution path and `Transform` use the product path's top: `forward_radix8`
  (new dense variant for a full factor) and `inverse_top`. One pass over memory less each way.
- `lc-bench` (EPYC 7B13), GCC 15.2, `-O2 -march=x86-64-v3`, 7 rounds against main (428ced1),
  `kernel.py bench --base`: geomean 0.9178 over 80 cases. inverse/6 0.168, multiply/6 0.563,
  multiply/8 0.864, inverse/10 0.843, multiply_dense/10 0.957, multiply_dense/16 0.973,
  multiply_dense/20 0.964, multiply_dense/24 0.950, forward/24 0.916, inverse/24 0.935; odd lg
  and large multiply (product path, unchanged) 0.99-1.005.
- forward with the dense radix-8 top was slower below 2^16 (forward/10 1.067, forward/12 1.013;
  the 8 input vectors and 6 twiddle vectors spill); now used from 2^16 only. Second run of
  forward alone, 7 rounds: 2^16..2^24 0.913-1.005; below 2^16 1.003-1.044, including odd lg where
  the code did not change (guess: code layout), multiply/9 and multiply/10 (product path, code
  unchanged) 1.020 and 1.028 in the first run.

2026-10-11, claude (issue #3, round 2): speed-ups on avx2 at every size; the scalar variant only
gets the compile-time tables. `lc-bench`, GCC 15.2, `-O2 -march=x86-64-v3`, each step timed
against the one before, 7 rounds, paired ratios (`kernel.py bench --base`).
- Where the time went (probes above): tables built per call were 75 of 493 cycles of
  multiply/6, 766 of 37302 at 2^12; a small product spent 21-28 cycles per vector beyond 2
  forward and 1 inverse transform at 2^6..2^10, against 12 at 2^14 (short bottom stages: one per
  subtree of 4 or 16 vectors, each with an unpipelined first and last batch); the h = 1 loops of
  `Transform` ran at 24 and 17 cycles per group against 14.5 for a radix-4 column.
- Compile-time tables to 2^16: multiply 2^6..2^16 0.871, 0.831, 0.903, 0.901, 0.954, 0.976,
  0.984, 0.978, 0.984, 0.985, 0.987; dense alike; 2^17 and up 0.995-1.004.
- Sibling subtrees as one tile of up to 256 vectors: product path 2^6..2^12 0.86, 0.86, 0.86,
  0.88, 0.83, 0.92, 0.93. A first version with runtime loop bounds and group indices (base + j)
  / 4h cost 0.3-0.8% at 2^13..2^16 (single 256-vector tiles); templated on the subtree size with
  counters again: 1.000-1.002.
- `PairBottom` for dense factors too (lc-opt kept them on `GroupBottom`): multiply_dense
  2^10..2^25 0.94-0.98, 2^6 0.96; but 2^7 and 2^8 1.17 and 1.12, so `GroupBottom` stays there.
  `convolution()` and `product()` became one `product<Bottom>`.
- `Transform`: radix-2 pass without reductions (canonical inputs; 7 -> 3 ops per pair, but the
  pass is store-bound: 4.4 -> 4.1 cycles per 4 vectors), asm h = 1 kernels, sibling tiles:
  forward/inverse 0.977 geomean; forward/7 1.020.
- `forward_top`: forward, odd lg 7..25: 0.824, 0.954, 0.942, 0.950, 0.969, 0.973, 0.987, 0.986,
  0.995, 0.990. multiply_dense/9 1.011 (only the dense factor's top changed).
- `inverse_top4` for odd lg (scale folded into the twiddles): inverse 0.928, 0.905, 0.919, 0.931,
  0.942, 0.951, 0.956, 0.963, 0.975, 0.971; multiply odd 0.97-0.99. multiply_dense/7 and /8 got
  1.06 and 1.03 slower with no change on their path (lg 8 is even); the same code in a probe
  binary ran at the old speed, and a stack offset sweep (0..4 KiB, 64-byte steps) moved them
  < 3%. Guess: code alignment.
- Tried and dropped: odd lanes from a load 4 bytes on in the h = 1 forward loop, as the asm
  loops do: forward/6..8 2-8% slower (guess: the inputs were just stored by the level above, and a
  load across two stores waits for them; the asm leaves kernels shift instead). Two groups per
  iteration in the intrinsics h = 1 loops: 29.5 and 23 cycles per group, from 24 and 17.4 (guess:
  spills). Leaves kernels with 2 or 4 groups per statement, shared or repeated broadcasts,
  window 12..40, margin 0..2: all 22.3-24.3 (forward) and 15.2-16.9 (inverse) cycles per group;
  not scheduling-bound (guess: the canonical outputs, 14 more ops and a 4-op serial tail per
  group). A fused dense radix-8 top for `Transform` below 2^16 costs the same per 8 vectors as the
  radix-2 pass and the first radix-4 level (32 cycles each); not changed.
- Bench oddity: one full run had multiply/9 and /10 at 1207 and 2576 ns against 1065 and 2143 in
  every other run of the same binary; not reproduced.
- Generator: VEX shifts take no memory operand; `Graph.op` now loads a shift's source (GAS had
  encoded `vpsrlq $32, mem` as EVEX: illegal instruction on Zen 3). No existing kernel changed.

2026-10-11, claude (issue #4, round 1): variant `intrinsics<S>` (`intrinsics.hpp`), `by_length`,
selection, tests, benches. Measurements above (intrinsics). Commits on `agent/ntt-all-r1`, squashed.
- Task: the avx2 variant's algorithm on `simd` and `modarith`; selected on sse, arm and scalar
  (was scalar), and on avx512 and zen5 with `Avx512` from 2^8-2^10 on (was the AVX2 asm). avx2
  keeps the asm. Tests: every `intrinsics<S>` for S in `simd::All` through the variant tests
  (Scalar to 2^20), plus each butterfly, the bottom level, the fused leaf products and the zmm
  pair bottom against lane models with inputs at the edges of their ranges (< 4P forward, < 2P
  inverse). Pass at -O2 and ASan/UBSan on `lc-amd` (scalar, sse, avx2), `lc-intel-test`,
  `lc-zen5`, `lc-arm` and the Mac. Benches: `@intrinsics_<struct>` for the 8- and 16-lane
  structs (Avx512Vl, not Avx2, on AVX-512) and `@avx2` where not selected everywhere, to 2^22.
- Tried, lost: a fused radix-8 top (the radix-2 level and both halves' first radix-4 groups in
  one pass, as avx2's `forward_radix8_dense`, and its inverse with the scale). Against radix-2 tops,
  paired, 5 rounds, the radix-2 form was faster almost everywhere: Zen 3 Avx2 0.85-0.99, Zen 5
  zmm 0.81-1.0, Neoverse V2 0.90-0.98, Emerald Rapids zmm 0.78-1.0 except lg 22 (forward 1.035,
  inverse 1.038). Guess: 8 streams at a power-of-two stride (4K aliasing, set conflicts), and on
  16 registers spills. Removed.
- Won: forcing inlining of the butterflies (`[[gnu::always_inline]]`): before it, Zen 3
  forward/15 and multiply/15 (Avx2) were 1.19x and 1.30x slower.
- Won: `#pragma GCC unroll 8` on the leaf product loops and `[[gnu::always_inline]]` on the
  bottom level's butterfly lambdas. `perf` on `lc-intel` had shown the bottom lambda as its own
  symbol (23% of forward/16) and the leaf product loop rolled (2 dependency chains). Paired, 5
  rounds: multiply 0.76-0.88 on Emerald Rapids zmm, 0.91-0.95 on Zen 5 zmm, 0.91-0.96 on Neoverse
  V2; forward 0.79-0.96 on x86 (Neon unchanged); inverse unchanged.
- Won: the zmm bottom level and zmm leaf products in registers (`WideBottom`, above), against
  the ymm bottom with stack windows, paired, 5 rounds, `@intrinsics_avx512` at 2^10..2^22:
  Emerald Rapids forward 0.87-0.97, inverse 0.91-1.00, multiply 0.91-0.94; Zen 5 forward
  0.81-0.93 (from 2^7), inverse 0.89-0.94, multiply 0.81-0.89. The two steps were not timed
  apart.
- Thresholds (kWideMinLog, `by_length`): from same-binary runs above. Against main before issue
  #3's round 2, intrinsics selected from 2^6 already won (avx512 0.904, zen5 0.780; gate passed
  twice on PR #30); round 2 made the asm 5-25% faster below 2^10, so the asm stays there.
- Not done: merged tiles (one bottom stage for sibling subtrees, as avx2 now), zmm for Neon's
  4-lane equivalent (no two-source permute in `simd::Neon`), a Neon Shoup on x < 2^31 (`sqdmulh`)
  with values kept < 2P, the asm's tricks (windows P - y A, scale folded into twiddles).

## Sources

- lc-opt `lib/ntt` (ours): the avx2 variant and the generators, moved and renamed. Its sources:
  our QPoly explorations 008-010; Shoup multiplication (V. Shoup, NTL; D. Harvey, "Faster
  arithmetic for number-theoretic transforms", J. Symb. Comp. 60 (2014),
  https://arxiv.org/abs/1205.2926); splitting to x^k - w leaves (https://codeforces.com/blog/entry/142063,
  https://piskareviv.github.io/blog_aux_ntt_1/; read for ideas, no code used).
