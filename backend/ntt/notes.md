# backend/ntt

Number-theoretic transforms and products modulo P = 998244353. API: `ntt.hpp` (`Convolution`,
`Transform`, the leaf form). Variants: `scalar.hpp` (every target; the model), `avx2.hpp`
(lc-opt's `lib/ntt`). `select.hpp` picks one per op. Tests: `test.cpp`; benches: `bench.cpp`.
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
  Blocks of 8 values then their 8 Shoup quotients; L/16 entries each way.
- `Convolution`: a, b and the tables in one mapping, 2 MiB aligned, `MADV_HUGEPAGE`; a and b are
  L + 16 words apart, so equal offsets fall in different cache sets. Tables built per call.
- `Transform`: tables built once; forward and inverse in place on caller arrays (32-byte aligned,
  16 words of padding: the AVX2 forward kernels read 4 bytes past their last vector).

### avx2

lc-opt's `lib/ntt` (`ntt.hpp`, `product.hpp`; its `notes.md` has the history and the QPoly
origins). Two product paths, picked per call:

- convolution path (any factors): depth-first radix-4 recursion; subtrees of at most 256 vectors
  (8 KiB per array) run level by level ("tiles"). The bottom level (h = 1) is fused with the leaf
  products and the first inverse level, one group per asm statement, and batch j + 1's forward
  half is interleaved with batch j's products. If a factor's upper half is zero, the first
  radix-2 level is a copy and the first radix-4 level reads only the lower half. The scale
  (L/8)^-1 2^32 is fused into the last inverse level.
- product path (lg >= 9, both factors <= L/2; lc-opt's `ntt::Product`): two groups per
  always-inlined asm statement, leaf weights from the group's own twiddles, each coefficient of
  B broadcast once (145 -> 127 cycles per group on Zen 3 in lc-opt). Even lg: `forward_radix8`
  reads each factor's lower half once and writes both halves' first radix-4 group;
  `inverse_top` fuses the last inverse levels and the scale (9 Shoup products per column, not 13).
  Odd lg: the sparse radix-4 top level. Does not read the upper halves of a and b.
- Arithmetic: Shoup multiplication by twiddles, w < P with q = floor(w 2^32 / P); for any
  x < 2^32, x w - floor(x q / 2^32) P is in [0, 2P). Forward values < 4P, inverse < 2P;
  min(x, x - 2P) reduces lazily. Leaf products: Montgomery reduction (factor 2^-32).
- Kernels: `gen_avx2_kernels.py` builds each as a dataflow graph of AVX2 instructions,
  list-schedules it on a Zen 3 model (4 vector pipes, multiplies and shifts on 2 each, 2 loads,
  1 store per cycle) under a register budget, allocates ymm0-15 by linear scan and emits one asm
  statement. Knobs from QPoly exploration 009 (timed on EPYC 7763).
- Top of an nv = 2 * 4^j transform (even lg >= 8), both paths and `Transform`: one pass for the
  radix-2 level and both halves' first radix-4 groups (`forward_radix8_sparse`, lc-opt's, or
  `forward_radix8_dense`; the dense one only from 2^16 in `Transform::forward`), 8 subtrees, then
  `inverse_top` for the last inverse groups, the radix-2 level and the scale. Scales come from a
  constexpr table.
- `Transform::forward`/`inverse` (new here, not in lc-opt): the same recursion on one array with
  the asm kernels down to h = 4, and an intrinsics loop for the h = 1 level (`forward_bottom`,
  `inverse_bottom`). Not tuned.

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

### Absolute (lc-bench)

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

## Sources

- lc-opt `lib/ntt` (ours): the avx2 variant and the generators, moved and renamed. Its sources:
  our QPoly explorations 008-010; Shoup multiplication (V. Shoup, NTL; D. Harvey, "Faster
  arithmetic for number-theoretic transforms", J. Symb. Comp. 60 (2014),
  https://arxiv.org/abs/1205.2926); splitting to x^k - w leaves (https://codeforces.com/blog/entry/142063,
  https://piskareviv.github.io/blog_aux_ntt_1/; read for ideas, no code used).
