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
- `Transform::forward`/`inverse` (new here, not in lc-opt): the same recursion on one array with
  the asm kernels down to h = 4, and an intrinsics loop for the h = 1 level (`forward_bottom`,
  `inverse_bottom`). Not tuned.

## Measurements

## Log

## Sources

- lc-opt `lib/ntt` (ours): the avx2 variant and the generators, moved and renamed. Its sources:
  our QPoly explorations 008-010; Shoup multiplication (V. Shoup, NTL; D. Harvey, "Faster
  arithmetic for number-theoretic transforms", J. Symb. Comp. 60 (2014),
  https://arxiv.org/abs/1205.2926); splitting to x^k - w leaves (https://codeforces.com/blog/entry/142063,
  https://piskareviv.github.io/blog_aux_ntt_1/; read for ideas, no code used).
