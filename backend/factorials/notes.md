# backend/factorials

Tables fact[i] = i! and inverse[i] = 1 / i! modulo a prime P (2^12 < P < 2^30), i < n <= P,
written into the caller's spans (`factorials.hpp`, API at the top). `select.hpp` picks the simd
struct, vectors per chain and the size below which the scalar loop runs; `model.hpp` is that loop
(the frontend's before this kernel) and the tests' reference; `transpose.hpp` holds the
in-register transposes per vector type. The frontend's `Factorials` calls `tables` (issue #33).

## Design

`Chains<S, K, P>`: W = K S::lanes lanes, lane s covers positions [a_s, a_s + C). From lc-opt's
product chains (`lib/poly/factorials.hpp`: `Chain`, `scan`, `scan_chunk`), rewritten on `simd`
and `modarith` so one source runs on every target.

- Lanes inside the tables: a_s = min(s C, n - C). The lanes past n - C all cover the last C
  positions (the same values, stored twice). So no lane reads a multiplier past n or writes past
  n: no scratch buffer and copy (the prototype's `factorials()` wrote 64 C words to scratch and
  copied n of each table), and no n + 1024 <= P limit (lc-opt's). Any n >= 64 up to P.
- C = chunk(n): n / W rounded up to the vector length; from 64 on, a multiple of 16 with C / 16
  odd (lc-opt's `scan_chunk`, against cache-set conflicts of the W streams). The rounding costs
  at most 32 W positions of duplicate work.
- Pass 1, lane products (a_s, a_s + C - 1]: x <- montgomery(x, g) with g = (a + 2j + 1)(a + 2j + 2)
  in Montgomery form, two factors per step. g moves by its first difference d (a modular add), d
  by 8 2^32 mod P (a second add). lc-opt took one factor per step. Lane k - 1 (k = n / C, the
  last lane at s C) also gives the product up to n - C: the start of the lanes past it.
- Scalar prefix over the lanes: last[s] = (a_s + C - 1)! = last[s - 1] a_s product[s], one
  dependent product per lane; first[s] = (a_s)! = last[s - 1] a_s.
- Pass 2: 1 / last[s] for all lanes at once, x^(P - 2) by right-to-left square and multiply in
  Montgomery form (the square chain and the product chain overlap: 30 dependent steps, not 58).
  lc-opt: Montgomery's batch inversion in scalar code.
- Pass 3, the scan: an up chain from (a_s)! times a_s + 1, a_s + 2, ... and a down chain from
  1 / (a_s + C - 1)! times a_s + C - 1, a_s + C - 2, .... Multipliers in Montgomery form, so x holds
  the plain product: m = base + u, u = j 2^32 mod P shared by all lanes (the down chain uses
  base + P - u). x, m < 2P. Every S::lanes steps the chains' records form a block; an S::lanes x
  S::lanes transpose turns it into vectors of consecutive positions, reduced below P and stored.
  The chains of the next block are interleaved with the transposes and stores of the current one
  (lc-opt's order); the last block is stored without running the chains past the end.
- Code: `[[gnu::flatten]]` on each pass and `unrolled<N>` (index sequences) keep every lambda
  and modarith call inline and every block index constant. Without them GCC 15 kept the scan's
  lambdas as calls and the transposes as loops over stack arrays: scan 0.83-0.86 ns per position
  against 0.53-0.65 (`lc-bench`, avx2, K = 2 and 4).
- Selection (`select.hpp`): S = simd::Native (zmm on `avx512` and `zen5`), K = 2 on x86, 4 on Neon
  and Scalar. The loop below n = 64 (Zen 3, Neoverse V2), 96 (Zen 5), 128 (Emerald Rapids and any
  other AVX-512 target). ymm (Avx512Vl) with K = 2, 16 lanes, from n = 2^19 on `avx512` and
  5 2^20 on `zen5`, where zmm's 32 lanes (64 write streams) stall on memory (Log).

## Tests

`test.cpp`: the model against the definition (fact[i] = fact[i - 1] i, fact[i] inverse[i] = 1);
then `Chains` on every simd struct the machine can run with K = 2, 4, 8, and the selected
`tables`, against the model: every n up to 1100, 2^k - 1, 2^k, 2^k + 1 up to 2^20, both sides of
every change of C up to 2^14 (so every count of collapsed lanes), 20 random n, n = P - 1 and P
for P = 4099 and 65537 (where the last multiplier is P = 0 mod P), and P = 1073741789 (the
largest prime below 2^30). Guard words after n must stay untouched. Mutations caught: a lane
start one lower, the wrong factor for the collapsed lanes, a chain stopped a block early, the down
chain's base off by one. Passes at -O2 and ASan/UBSan on `lc-amd` (scalar, sse, avx2),
`lc-intel-test` (avx512), `lc-zen5`, `lc-arm` and the Mac (scalar, arm). The frontend's
`test_factorials.cpp` runs the kernel through `Factorials` and through `detail::factorial_tables`
for every n <= 4099 at P = 4099.

## Measurements

`bench.cpp`, P = 998244353, buffers allocated once (warm), median ns per position over 11 rounds
(each position is two words, one per table). GCC 15.2, `tools/kernel.py` flags, 2026-10-11.

| lg | Zen 3 `lc-bench` (avx2) | Emerald Rapids `lc-intel` (avx512) | Zen 5 `lc-zen5` | Neoverse V2 `lc-arm` |
|---:|---:|---:|---:|---:|
| 6 | 7.70 (loop 8.24) | 6.51 (loop) | 6.63 (loop) | 8.09 (loop 9.55) |
| 8 | 2.43 | 2.85 | 2.21 | 2.78 |
| 10 | 1.30 (loop 7.02) | 0.975 (loop 5.79) | 0.751 (loop 5.88) | 1.59 (loop 8.13) |
| 12 | 0.776 | 0.626 | 0.480 | 1.07 |
| 14 | 0.658 (loop 6.93) | 0.484 (loop 5.74) | 0.320 (loop 5.83) | 0.935 (loop 8.04) |
| 16 | 0.654 | 0.449 | 0.281 | 0.901 |
| 18 | 0.663 (loop 6.94) | 0.457 (loop 5.76) | 0.385 (loop 5.83) | 0.895 (loop 8.04) |
| 20 | 0.666 | 0.575 (ymm) | 0.379 | 0.894 |
| 22 | 0.727 (loop 6.95) | 0.582 (ymm, loop 5.77) | 0.464 (loop 5.83) | 0.918 (loop 8.04) |

Against the loop (the frontend's code before this kernel), lg 14 / 18 / 22: Zen 3 10.5x / 10.5x /
9.6x, Emerald Rapids 11.9x / 12.6x / 9.9x, Zen 5 18.2x / 15.1x / 12.6x, Neoverse V2 8.6x / 9.0x /
8.8x. The issue's estimate from lc-opt's notes was about 0.6 ns per position on Zen 3; this is
0.65-0.73.

Through the frontend (`frontend/bench.cpp`, `factorials/<lg>`: fresh zero-filled vectors each
call), new/base against main (27905ef) on `lc-zen5` and `lc-arm`, 11 rounds: lg 10 0.155 and 0.203,
lg 14 0.061 and 0.122, lg 18 0.211 and 0.252, lg 22 0.250 and 0.293. `factorials/<lg>` against
`factorials@kernel/<lg>` (the same two vectors and the kernel call): 0.998-1.008 on Zen 5,
0.985-1.001 on Neoverse V2. At lg 22 the allocation (zero-fill and page faults of 32 MiB) is
1.3 ns of Zen 5's 1.79 ns per position; the kernel itself 0.46.

Passes at n = 2^18 (`explore/phases.cpp`, best of 7, ns per position; final code):

| Machine (S, K) | lane products | inversion | scan | all |
|---|---:|---:|---:|---:|
| Zen 3 (avx2, 2) | 0.120 | 215 ns | 0.524 | 0.648 |
| Emerald Rapids (zmm, 2) | 0.092 | 215 ns | 0.364 | 0.457 |
| Zen 5 (zmm, 2) | 0.071 | 167 ns | 0.314 | 0.389 |
| Neoverse V2 (neon, 4) | 0.172 | 236 ns | 0.720 | 0.894 |

Floors, multiplies only: two Montgomery products per position in the scan, 6 `vpmuludq` per 8
lanes on Zen 3's 2 multiply pipes = 0.75 cycles = 0.22 ns per position; Neoverse V2 4 multiplies
per 4 lanes on 2 pipes = 1 cycle = 0.34 ns. Counting every vector op of the scan (Montgomery 13,
multiplier 1, transpose 3, reduce 2 per vector and chain) Zen 3 needs 9.5 cycles per 8 positions
on 4 pipes, 0.35 ns per position; measured 0.52.

## Log

2026-10-11, claude (issue #33, round 1). New kernel; GCC 15.2, `tools/kernel.py` flags, main
27905ef. Timings are best of 7 (explore programs) or medians of 11 rounds (bench), ns per position.
Exploration files (the `explore/*.cpp` programs named below, `run.sh` and their outputs):
`~/Documents/hpc_library-explore/factorials/r1/`.
- Moved lc-opt's chains onto `simd` and `modarith`, with per-type transposes (4 x 4 Sse and Neon,
  8 x 8 Avx2 and Avx512Vl as lc-opt, 16 x 16 Avx512) and the lane layout above. First build:
  1.03-1.09 per position on Zen 3 (avx2, K = 4): GCC kept the scan's lambdas as calls and the
  transposes as loops. `[[gnu::flatten]]` and `#pragma GCC unroll`: scan 0.86 -> 0.65.
- Scan order: lc-opt's (next block's chains interleaved with this block's stores) against a block
  of chains then its stores (`scan2`), 2^18: Zen 3 avx2 K = 2 0.534 / 0.512, Emerald Rapids zmm
  K = 2 0.371 / 0.445, Zen 5 zmm K = 2 0.323 / 0.291, Neoverse V2 K = 4 0.720 / 0.735. Kept lc-opt's:
  never worse by more than 11%, 17% better on Emerald Rapids.
- Vectors per chain, scan at 2^18: K = 2 against 4: Zen 3 0.534 / 0.650, Emerald Rapids zmm 0.371 /
  0.496, Zen 5 zmm 0.323 / 0.380; Neoverse V2 K = 4 against 2: 0.720 / 0.770.
- Where the scan's time goes (split order, 2^18, Zen 3 avx2 K = 2): full 0.518, without transposes
  0.418, stores to one L1 block 0.409, chains only 0.311. Emerald Rapids zmm K = 2: 0.418, 0.343,
  0.332, 0.214; Zen 5 zmm K = 2: 0.278, 0.197, 0.204, 0.149. The chains alone run within 1.1-1.5x of
  the Montgomery stream rate (`modarith` notes); the transposes and stores add 65-95% on top.
- Lane products, two factors per step (pair products by finite differences) against one (lc-opt):
  Zen 3 K = 2 0.256 -> 0.120, Emerald Rapids zmm 0.184 -> 0.092, Zen 5 zmm 0.139 -> 0.071,
  Neoverse V2 K = 4 0.333 -> 0.172. Half the products and half the dependent steps.
- Fixed costs: the inversion by right-to-left powers (two overlapping chains) instead of
  left-to-right: 245 -> 215 ns on Zen 3; the scalar prefix with one dependent product per lane
  instead of two. Then the threshold against the loop (`explore/threshold.cpp`): chains / loop at
  n = 64: Zen 3 0.93, Neoverse V2 0.85, Zen 5 1.28 (0.88 at 96), Emerald Rapids 1.71 (1.17 at 96,
  0.90 at 128).
- lc-opt's x86 Montgomery with the multiplier's odd lanes from a precomputed odd base (an add
  instead of a shift): scan new/old Zen 3 0.97-1.03, Emerald Rapids 0.91-1.10, Zen 5 0.98-1.31.
  No gain; not kept (`explore/oddbase.cpp`).
- Neon: Montgomery with q = mul_lo(x, m') and m' = m P^-1 kept per lane like m (one multiply less
  per product): scan 0.972-0.988 on Neoverse V2. Too little for a kernel-local product; not kept
  (`explore/premul.cpp`).
- Large n (`explore/large.cpp`, `cross.cpp`, `place.cpp`). 32 lanes (64 write streams) degrade at
  large n where 16 do not: Zen 3 avx2 K = 4 1.35 at 2^22 against K = 2 0.73. On Emerald Rapids zmm
  K = 2 (32 lanes) is bimodal from 2^18 on, 0.44-0.49 or 0.71-0.83 by the tables' placement: in one
  2 MiB-aligned buffer, both tables 64-byte aligned 0.44-0.61 (by their distance), fact 16 bytes
  off (where glibc puts a large std::vector) 0.79-0.82 at 22 of 24 (n, distance) pairs. ymm K = 2 (16 lanes): 0.55-0.58 at any
  placement. Zen 5 zmm loses 5-12% when misaligned and, in std::vectors, falls behind ymm from
  5 2^20 on (0.545 against 0.499). Selected: ymm K = 2 from 2^19 on `avx512`, from 5 2^20 on
  `zen5`. Emerald Rapids lg 20 and 22: 0.80 -> 0.58.
- Frontend: `detail::factorial_tables` calls `factorials::tables` for 2^12 < P < 2^30 and the
  model otherwise; numbers in Measurements.
- Open: (1) 64-byte aligned tables would let zmm run at 0.44 instead of 0.58 on Emerald Rapids at
  large n; the frontend's tables are std::vectors. Aligning the lanes to the pointer needs both
  tables at the same offset, or separate up and down lane grids. (2) The frontend's allocation
  costs more than the kernel from 2^18 on (zero-fill and page faults). (3) Non-temporal stores for
  tables past the last-level cache, not tried. (4) Neoverse V2's scan (0.72) is 2x its multiply
  floor; the transposes and stores cost 0.3 of it.

## Sources

- lc-opt `lib/poly/factorials.hpp` and its notes (Factorials): the product chains, the scan with
  in-register 8 x 8 transposes, `scan_chunk`. Moved and rewritten here.
- Montgomery multiplication: modarith (P. Montgomery, Math. Comp. 44, 1985).
- Finite differences for the pair products g_j: the standard forward-difference table of a
  polynomial; no source used.
