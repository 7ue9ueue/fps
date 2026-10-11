# backend/factorials

Tables fact[i] = i! and inverse[i] = 1 / i! modulo a prime P (2^12 < P < 2^30), i < n <= P,
written into the caller's spans (`factorials.hpp`, API at the top). `select.hpp` picks the variant
per target and size (tiers: simd struct, vectors per chain, chain product); `model.hpp` is the
scalar loop (the frontend's before this kernel), run below the first tier, and the tests'
reference; `transpose.hpp` holds the in-register lane moves per vector type (transposes, lane
shifts). The frontend's `Factorials` calls `tables` (issue #33).

## Design

`Chains<S, K, P, step>`: W = K S::lanes lanes, lane s covers positions [a_s, a_s + C). From
lc-opt's product chains (`lib/poly/factorials.hpp`: `Chain`, `scan`, `scan_chunk`), rewritten on
`simd` and `modarith` so one source runs on every target.

- Lanes inside the tables: a_s = min(h + s C, n - C). The lanes past n - C all cover the last C
  positions (the same values, stored twice). So no lane reads a multiplier past n or writes past
  n: no scratch buffer and copy (the prototype's `factorials()` wrote 64 C words to scratch and
  copied n of each table), and no n + 1024 <= P limit (lc-opt's). Any n >= 4 S::lanes up to P.
- Lanes aligned to fact (round 2): h < S::lanes is the words from fact to its next vector
  boundary, so every store of the lanes s < k = (n - h) / C into fact is aligned, and into
  inverse when it has fact's offset (two std::vectors of the same size do). Positions below h
  come from constant tables of i! and 1 / i!, i < 16; h! starts lane 0.
- C = chunk(n): n / W rounded up to the vector length; from 64 on, a multiple of 16, and one more
  line (16 words) when m = C / 16 is a multiple of 512 / W (round 3). The W lanes start m lines
  apart, so in a 64-set L1 they fall into 64 / gcd(m, 64) sets; the bump keeps it to at most 4
  streams per set. lc-opt's `scan_chunk` bumped every even m, which at powers of two costs
  16 W positions of duplicate work: 25% at n = 2^10 with 16 lanes. Never bumping is far worse
  where m is a large power of two (Log).
- Pass 1, lane products (a_s, a_s + C - 1], four factors per step (round 2): the first
  r = (C - 1) % 4 factors one by one, then x <- montgomery(x, g) with g = w (w + 1)(w + 2)(w + 3)
  in Montgomery form, w = a + r + 1 + 4j. g moves by its forward differences d1, d2, d3 (four
  modular adds per step; d4 = 4! 4^4 is constant). The start values come from vector products:
  G = z (z + 2) with z = w (w + 3), z(w + 4) = z(w) + 8w + 28, so G at w, w + 4, w + 8, w + 12 is
  5 Montgomery products per vector. Round 1 took two factors per step, lc-opt one. Lane k - 1
  also gives the product up to n - C (whole steps, then up to 3 scalar factors for that lane
  only): the start of the lanes past it. From 16 steps on and for K <= 2, two accumulators take
  the steps in turn, halving the product chain (round 3). The lane starts and the multipliers'
  bases are vector ops (round 3; before, two scalar `% P` per lane: 31-67 ns of setup).
- Pass 2, `starts` (round 3): first[s] = (a_s)! and 1 / last[s] = 1 / (a_s + C - 1)! in vector
  registers. first is an exclusive prefix product over the lanes of g_s = (a_s + 1) ... (a_(s+1))
  = product[s] (a_s + C) (h! shifted in at lane 0; the factor up to n - C at lane k - 1; 1 past
  it), and 1 / last[s] = 1 / last[k - 1] times an exclusive suffix product of r_t = a_t
  product[t] (lane t's own positions). Both in Montgomery form, log2(S::lanes) shifted products
  within a vector (valignd, palignr, ext; Avx2 vpermd and a blend) and log2(K) across vectors.
  One scalar inversion of last[k - 1] last[k] (the lanes past n - C all end at n - 1), by a
  branch-free binary extended Euclid (38-51 ns); two products split it. Before: k / 4 + 4
  dependent scalar `% P` products, ~4 k in all, then x^(P - 2) for all W lanes in vectors
  (150-230 ns: 30 steps of 4 contended vector products).
- Pass 3, the scan: an up chain from (a_s)! times a_s + 1, a_s + 2, ... and a down chain from
  1 / (a_s + C - 1)! times a_s + C - 1, a_s + C - 2, .... x < 2P. Two chain products (`Step`):
  - montgomery: the multiplier in Montgomery form, m = base + u, u = j 2^32 mod P shared by all
    lanes (the down chain uses base + P - u). x holds the plain product.
  - shoup (round 3): x m - floor(x q / 2^32) P with m plain, m = base + j, and
    q = floor(base 2^32 / P) + floor(j 2^32 / P) (down: base - j, the difference of the two minus
    1). The sum of two floors is floor(m 2^32 / P) or one less; with x < 2^31 the result is still
    in [0, 2P). floor(j 2^32 / P) = -(j 2^32 mod P) P^-1 mod 2^32: one mul_lo per step for all
    lanes. Per chain vector and step: mul_lo, mulhi, mul_sub and 2 adds; on Neon with SVE2 one
    umulh, mul, mls (Montgomery: 2 mul, 2 umulh), on x86 4 multiplies instead of 6.
  Every S::lanes steps the chains' records form a block; an S::lanes x S::lanes transpose turns
  it into vectors of consecutive positions, reduced below P and stored. The chains of the next
  block are interleaved with the transposes and stores of the current one (lc-opt's order); the
  last block is stored without running the chains past the end.
- Code: `[[gnu::flatten]]` on each pass and `unrolled<N>` (index sequences) keep every lambda
  and modarith call inline and every block index constant. Without them GCC 15 kept the scan's
  lambdas as calls and the transposes as loops over stack arrays: scan 0.83-0.86 ns per position
  against 0.53-0.65 (`lc-bench`, avx2, K = 2 and 4). Lane moves are per-ISA intrinsics, not
  `S::permute`: GCC built Neon's `tbl` index tables at run time on every call (round 3, Log).
- Selection (`select.hpp`, round 3): tiers per target, from n:

  | Target | Tiers (struct.K.step from n) |
  |---|---|
  | `avx2` (Zen 3) | Sse.1.shoup 16, Avx2.1.shoup 48, Avx2.2.shoup 2048 |
  | `avx512` (Emerald Rapids) | Sse.1.shoup 16, Avx512Vl.1.mont 48, Avx512Vl.2.mont 256, Avx512.2.mont 1024, Avx512Vl.2.mont 2^19 |
  | `zen5` | Sse.1.shoup 16, Avx512Vl.1.shoup 48, Avx512Vl.2.shoup 256, Avx512.2.mont 2048, Avx512Vl.2.shoup 5 2^20 |
  | `arm` (Neoverse V2) | Neon.1.shoup 16, Neon.2.shoup 64, Neon.4.shoup 1280 |
  | `sse`; `scalar` (not timed) | Sse.1.shoup 16, Sse.2.shoup 256; Scalar.4.mont 64 |

  Fewer lanes win at small n: less duplicate work (C is at least the vector length, so W C >= W
  S::lanes positions) and fewer lanes to start. zmm (32 lanes, 64 write streams) stalls on memory
  from 2^19 on Emerald Rapids and 5 2^20 on Zen 5 (rounds 1, 2). Shoup wins on Zen 3, Neoverse V2
  and Zen 5's ymm; Montgomery on Emerald Rapids (vpmulld is 2 uops) and Zen 5's zmm (Log).

## Tests

`test.cpp`: the model against the definition (fact[i] = fact[i - 1] i, fact[i] inverse[i] = 1);
then `Chains` on every simd struct the machine can run, K = 1 (round 3), 2, 4, 8, both chain
products, and the selected `tables`, against the model: every n from 4 S::lanes up to 1100,
2^k - 1, 2^k, 2^k + 1 up to 2^20, both sides of every change of C up to 2^14 (so every count of
collapsed lanes), 20 random n, n = P - 1 and P for P = 4099 and 65537 (where the last multiplier is
P = 0 mod P), and P = 1073741789 (the largest prime below 2^30). Guard words after n must stay
untouched. Mutations caught: a lane start one lower, the wrong factor for the collapsed lanes, a
chain stopped a block early, the down chain's base off by one, the down chain's Shoup quotient
without its - 1 (round 3). Placements (round 2): on fewer sizes (every n up to 300, both sides of
each change of C up to 2^12, powers of two and neighbours from 2^9 to 2^17, 8 random n), K = 2,
tables at word offsets (1, 1), (4, 4), (12, 12), (15, 15), (0, 8), (3, 13), (9, 2) from a 64-byte
boundary, so every h and both equal and different offsets; words before the tables must stay
untouched too. The selected `tables` also on both sides of every tier's threshold and at
5 2^20 + 17 (Zen 5's last tier). Passes at -O2 and ASan/UBSan on `lc-amd` (scalar, sse, avx2),
`lc-intel-test` (avx512), `lc-zen5-test`, `lc-arm` and the Mac (scalar, arm). The frontend's `test_factorials.cpp` runs the kernel through
`Factorials` and through `detail::factorial_tables` for every n <= 4099 at P = 4099.

## Measurements

`bench.cpp`, P = 998244353, buffers allocated once (warm), median ns per position over 11 rounds
(each position is two words, one per table). GCC 15.2, `tools/kernel.py` flags, 2026-10-11,
round 3 (this branch on main 0d2355e, same run); main's numbers in brackets.

| lg | Zen 3 `lc-bench` (avx2) | Emerald Rapids `lc-intel` (avx512) | Zen 5 `lc-zen5` | Neoverse V2 `lc-arm` |
|---:|---:|---:|---:|---:|
| 6 | 2.73 (6.84) | 2.57 (6.50, loop) | 2.05 (6.63, loop) | 3.06 (7.73) |
| 8 | 1.30 (2.27) | 1.24 (2.38) | 0.882 (1.82) | 1.40 (2.63) |
| 10 | 0.722 (1.18) | 0.588 (0.818) | 0.495 (0.747) | 0.924 (1.49) |
| 12 | 0.574 (0.706) | 0.434 (0.543) | 0.329 (0.418) | 0.783 (0.999) |
| 14 | 0.539 (0.604) | 0.410 (0.424) | 0.252 (0.275) | 0.752 (0.875) |
| 16 | 0.554 (0.605) | 0.389 (0.393) | 0.224 (0.238) | 0.732 (0.844) |
| 18 | 0.549 (0.603) | 0.388 (0.391) | 0.346 (0.353) | 0.728 (0.838) |
| 20 | 0.548 (0.598) | 0.458 (0.482; ymm) | 0.338 (0.342) | 0.748 (0.839) |
| 22 | 0.556 (0.614) | 0.458 (0.482; ymm) | 0.411 (0.414) | 0.887 (0.913) |
| geomean new/main | 0.743 | 0.775 | 0.736 | 0.724 |

The tables in the bench are two std::vectors, 16 bytes off a 64-byte boundary from lg 15 on.
Zen 5 from lg 18 and Neoverse V2 at lg 22 vary by run (IQR up to 0.91-1.02).
Small-n times move by up to 13% between builds that differ only in code not run at that size
(Zen 3 lg 8: 292 ns in one build, 332 in the next; Log).

Through the frontend (`frontend/bench.cpp`, `factorials/<lg>`, two unwritten tables per call),
new/base against main 0d2355e, 11 rounds, lg 10 / 14 / 18 / 22: Zen 3 0.618 / 0.901 / 0.995 /
0.993, Emerald Rapids 0.729 / 0.965 / 1.002 / 0.990, Zen 5 0.752 / 0.927 / 0.991 / 1.004,
Neoverse V2 0.632 / 0.860 / 0.952 / 0.965. From lg 18 on the page faults of the fresh tables
cost more than the kernel (2-4 ns per position against 0.3-0.9).

Parts at n = 64 (`explore/phases4.cpp`, cumulative stages, median of 15, ns per call; the
selected tier, one lane-product accumulator):

| Machine (struct.K.step) | setup | lane products | starts (inverse) | scan | all |
|---|---:|---:|---:|---:|---:|
| Zen 3 (Avx2.1.shoup) | 4 | 39 | 87 (50) | 46 | 176 |
| Emerald Rapids (Avx512Vl.1.mont) | 3 | 32 | 83 (46) | 49 | 166 |
| Zen 5 (Avx2.1.shoup; the tier now runs Avx512Vl.1.shoup, 131 against 130 ns in `sweep.cpp`) | 2 | 28 | 77 (39) | 32 | 139 |
| Neoverse V2 (Neon.2.shoup) | 4 | 56 | 77 (52) | 56 | 193 |

Main's at n = 64 (`explore/phases3.cpp`): setup 54-67, scalar prefix 53-65, vector inversion
151-230, 416-498 ns in all (the loop on Emerald Rapids and Zen 5).

Floors, multiplies only: two products per position in the scan. Zen 3, Shoup: 4 multiplies per 8
lanes on 2 pipes = 0.5 cycles = 0.14 ns per position (Montgomery 6: 0.22); Neoverse V2, Shoup:
3 per 4 lanes on 2 pipes = 0.75 cycles = 0.25 ns (Montgomery 4: 0.34). Emerald Rapids is bound by
its vector ports: zmm Montgomery at 2^16 runs 2.76 uops per position on p0 and p5 at 1.555
cycles per position, p0 84% and p5 94% busy (Log); measured 0.39 ns per position.

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
  tables at the same offset, or separate up and down lane grids. (Round 2: done for fact's grid;
  zmm still loses at large n, see below.) (2) The frontend's allocation
  costs more than the kernel from 2^18 on (zero-fill and page faults). (3) Non-temporal stores for
  tables past the last-level cache, not tried. (4) Neoverse V2's scan (0.72) is 2x its multiply
  floor; the transposes and stores cost 0.3 of it.

2026-10-11, claude (issue #33, round 2). GCC 15.2, `tools/kernel.py` flags; base main e754d90,
then 626819e after #51. Exploration files (`explore/*.cpp`, `variants.hpp`, `mainf/` = main's
kernel renamed for side-by-side runs, `run.sh`, outputs in `out/`):
`~/Documents/hpc_library-explore/factorials/r2/`. Timings: medians of 7-11 interleaved rounds,
ns per position, warm buffers unless named.
- Lanes aligned to fact (`align.cpp`, `variants.hpp` kShift): tables 64-byte aligned ("a64"),
  both 16 bytes off ("off16", std::vector's place), fact aligned and inverse 32 bytes off
  ("mixed"), both 4 bytes off. Zen 3 avx2 off16: lg 20 / 22 / 23 0.72 / 0.89 / 0.91 -> 0.65 /
  0.65 / 0.66, a64 unchanged (0.65). Emerald Rapids zmm off16 lg 16-18 0.46-0.47 -> 0.45; Zen 5
  lg 15-16 1-2%. Neoverse V2 within 1% except the 4-byte offset at lg 21-23 (0.92-1.12 ->
  0.90-1.05). Kept, with the head from constant tables (a scalar chain for the head cost Zen 5
  2-15% at n = 256-1024, `small.cpp`).
- Non-temporal zmm stores (`_mm512_stream_si512`), the lanes past n - C also aligned and a
  scalar tail of < 16 positions: warm, Emerald Rapids 0.46-0.49 at lg 18-23 for every placement
  (plain zmm 0.45 to lg 19, then 0.60-2.8; ymm 0.55-0.74); Zen 5 0.30-0.31 from lg 17 (plain
  0.36-0.55; at lg 16 0.315 against 0.276). With the lanes past n - C misaligned and stored plain
  among streamed lanes, Emerald Rapids took 0.59-1.43. ymm and xmm stream stores write half
  lines: Zen 3 2.8-3.3, Emerald Rapids ymm 14-17. But on fresh tables (`fresh.cpp`,
  `fresh2.cpp`: allocated per call, as `Factorials` does) streaming loses: Emerald Rapids lg 20
  unwritten 3.61 against main 2.21 and plain zmm 2.27, zero-filled 3.33 against 2.64; Zen 5 lg 18
  unwritten 1.81 against plain 1.37 (page faults zero each page into the cache, then every
  streamed line evicts it). The frontend's `factorials/22` on Emerald Rapids, then zero-filled:
  1.58x main. Not kept: the warm gain is a bench artifact; callers write fresh tables.
- zmm with aligned lanes against ymm at large n on Emerald Rapids, plain stores (`fresh2.cpp`):
  lg 20 zero-filled 2.85 against 2.64, unwritten 2.27 against 2.21, warm 0.63 against 0.58; lg 22
  warm 0.80 against 0.59. ymm kept from 2^19. Zen 5's switch (5 2^20) is above the gated sizes;
  left as is.
- Four factors per lane-product step (finite differences of a quartic) against two:
  `phases.cpp` at 2^18 above (lane products halved on every machine). Paired, kernel bench,
  four against two (both with aligned lanes): Zen 3 0.88-0.93 at every gated size (geomean
  0.911), Neoverse V2 0.92-0.96 (0.940). At n = 1024 too (Zen 3 0.911, V2 0.945): the 5-product
  setup costs less than the halved chain.
- Prefix over the lanes in groups of 4: in the same commit; not timed alone. Expected from the
  scalar product's latency (about 12 cycles): 31 -> 11 dependent products on zmm, 15 -> 7 on
  Neoverse V2.
- Neon scan variants (`neon.cpp`, `lc-arm`, K = 4, 2^18, main 0.896): chain products reduced to
  [0, P) inside the product (min(r, r + P)) instead of at the store 0.876; plus st4 one-lane stores
  instead of transposes 1.54; plus ld4 from a stack block instead of transposes 0.915; plus
  premultiplied q = x m' (round 1's idea) 0.903; K = 6 0.880; K = 8 0.944; the block index hidden
  from the compiler (no induction variable per lane pointer) 0.877. `perf stat` (`lc-arm`, sudo):
  10.5 instructions and 2.7 cycles per position, IPC 3.9, 46% of the 8 dispatch slots used, 35%
  stalled on the back end; the loop has 116 loads and 108 stores per 64 positions (block records
  and spilled lane pointers). The 2% of the first variant alone was not worth a kernel-local
  product; not kept.
- Gate cases, this branch against main 626819e, 11 rounds, geomean of `tables/<lg>`: Zen 3
  0.891, Emerald Rapids 0.870, Zen 5 0.880 (lg 10 0.996), Neoverse V2 0.939. Frontend
  `factorials/<lg>`: Zen 3 0.949, Emerald Rapids 0.915, Zen 5 0.928, Neoverse V2 0.958. Tests
  pass at -O2 and ASan/UBSan on every target's test machines and the Mac (rebased on df74bf7).
- Open: (1) Neoverse V2's scan is unchanged (0.72); its loop spills block records and lane
  pointers (above). (2) The inversion (167-236 ns, 30 dependent vector products) and the scalar
  setup are most of the time below n = 2^10. (3) Zen 5 at n = 1024 gains nothing (0.996).

2026-10-11, claude (issue #33, round 3). GCC 15.2, `tools/kernel.py` flags, base main 0d2355e.
Exploration files (`explore/*.cpp`, `mainf/` = main's kernel renamed, `one_acc.hpp`, `run.sh`,
`go.sh`, outputs in `out/`): `~/Documents/hpc_library-explore/factorials/r3/`. Timings: medians
of 11 interleaved rounds, warm tables in two std::vectors, ns per call or per position.
- Where main's time went at small n (`phases3.cpp`, cumulative stages, ns per call): setup (two
  scalar `% P` per lane) 54-67, scalar prefix 53-99, vector inversion 145-232, scan 80-216 at
  n = 64: Zen 3 435 in all, Emerald Rapids 597, Zen 5 449, Neoverse V2 498 (the loop: 527, 416,
  423, 611). zmm's C >= 16 makes 32 lanes compute 512 positions for any n <= 512 (scan 213 ns flat
  on Emerald Rapids); ymm K = 2 at n = 128: 404 against 600.
- Lane starts and multiplier bases as vector ops: setup 31-67 -> 2-8 ns.
- `starts` (Design) instead of the scalar prefix and the vector inversion. The first version
  used `S::permute` with `S::index`: GCC built Neon's `tbl` byte tables at run time on every call
  and kept the cross-vector loop over a stack array; Neoverse V2 K = 4 lg 6 1.36x main
  (`cmp.cpp` run a). Per-ISA lane moves (`transpose.hpp`) and unrolled vector indices: 0.62.
- The one inversion (`inv.cpp`, dependent calls, ns): x^(P - 2) in scalar Montgomery form 86 /
  112 / 100 / 87 (Neoverse V2 / Zen 3 / Emerald Rapids / Zen 5), with `% P` 133 / 114 / 102 / 90;
  binary extended Euclid with the halvings deferred to the other coefficient: 75-79 as written
  with selects (GCC made them a branch that mispredicts), 51 / 50 / 43 / 38 with masks (no
  branch but the exit; Apple M4: 31 against 88). Kept the last. Whole kernel at n = 64
  (`phases4.cpp`): 239 -> 193, 241 -> 176, 215 -> 166, 188 -> 139.
- Shoup chains in the scan (`cmp.cpp`, ratio to main at lg 14 / 16 / 18 / 20, same lanes,
  Montgomery in brackets): Zen 3 avx2 K = 2 0.907 / 0.889 / 0.934 / 0.932 (0.981 / 0.987 / 0.990
  / 0.999); Neoverse V2 K = 4 0.862 / 0.868 / 0.869 / 0.886 (0.991 / 1.001 / 1.002 / 1.002);
  Emerald Rapids zmm 1.050 / 1.062 / 1.043 (0.973 / 0.990 / 0.995), ymm at lg 20 1.048 (0.994);
  Zen 5 zmm 1.027 / 1.094 / 1.013 (0.960 / 0.985 / 0.966), ymm Shoup 7-11% ahead of ymm
  Montgomery at lg 12-20. Emerald Rapids: `vpmulld` is 2 uops, so Shoup has as many multiply
  uops as Montgomery. Zen 5 zmm: Shoup's scan has 572 multiplies against 792 but 392 shifts
  against 264 and 873 stack accesses against 782 (asm of the whole scan, `asmstat.py`).
- Chunk rule (`cmp.cpp` with `-DFPS_EXPLORE_ALIAS`, ratio to main): bump every even m (lc-opt),
  m % 8 == 0, never. Never bumping: Zen 3 lg 14 (C = 1024) 1.44, Emerald Rapids zmm lg 16-20
  1.41-1.90, Zen 5 ymm lg 14 2.26. Not bumping small m: Neoverse V2 K = 4 lg 10 (C = 64) 0.793 ->
  0.669, lg 12 (C = 256) 0.840 -> 0.794; Zen 3 lg 10 0.830 -> 0.737, lg 12 0.868 -> 0.835;
  Emerald Rapids zmm lg 12 (C = 128) 0.917 -> 0.830 but lg 14 (C = 512) 0.974 -> 1.071; Zen 5 zmm
  lg 12 0.894 -> 0.827, lg 14 0.956 -> 0.937, lg 16 0.987 -> 1.027. The rule kept: bump when m is a
  multiple of 512 / W (W gcd(m, 64) > 256: more than 4 streams per L1 set).
- Store addresses as one base per table plus shared lane offsets instead of 2 W lane pointers
  (fewer pointer spills on Neon): Neoverse V2 0.994 of the pointers, Zen 3 0.99-1.04, Emerald
  Rapids and Zen 5 within 1% but Zen 5 lg 10 1.045. Not kept.
- Tiers (`sweep.cpp`, ns per call, every variant at 16 sizes from 16 to 4096): fewer lanes win at
  small n. At n = 64: loop 530 / 416 / 424 / 611 (Zen 3 / Emerald Rapids / Zen 5 / Neoverse V2),
  Avx2 K = 1 175, Avx512Vl K = 1 154, Avx2 K = 1 130, Neon K = 1 187. Crossovers (K = 1 -> 2 ->
  zmm): Zen 3 2048; Emerald Rapids 256, 1024 (zmm wins at multiples of 512 only: 768 positions
  cost it 1024); Zen 5 256, 2048; Neoverse V2 K = 2 -> 4 at ~1280. Sse K = 1 below 48 on x86
  (Zen 5 at 16: 111 against the loop's 146). On Zen 5 one build of Avx2 K = 2 Shoup ran 2.4x
  slower at 32-512 (293-660 ns against 177-232 for Avx512Vl), while the run before had them
  equal: Zen 5's ymm tiers use Avx512Vl. Large tiers checked: Emerald Rapids lg 19 zmm 0.395
  against ymm 0.474 in this run (2^19 kept, round 1's placement study); Zen 5 lg 23 zmm 0.472,
  ymm Shoup 0.425-0.435.
- Lane products on two accumulators (`acc.cpp`, against one, same run): Zen 5 zmm 0.944-0.965 at
  2^14-2^18, ymm 0.94-0.96; Emerald Rapids ymm 0.955-0.967 from 2^12, zmm 0.99-1.01 (port-bound);
  Zen 3 Avx2 K = 1 0.93-0.97 from 2^10; Neoverse V2 K = 2 0.95-0.98 from 2^10. But Neon K = 4
  1.004-1.011 (registers) and at small C 1.00-1.07 (the extra final product); Zen 3's Avx2 K = 2
  at n = 256 1.08-1.19 whether the second accumulator ran or not (code layout, a guess). Kept from
  16 four-factor steps on, for K <= 2.
- Emerald Rapids at large n (`loop3.cpp`, `perf stat`, zmm Montgomery, 2^16): 1.555 cycles per
  position, p0 busy 84% and p5 (+ p11) 94%: 2.76 vector uops per position, of which the scan
  2.25 (per chain vector and step: Montgomery 12, transpose 4, reduce 2) and the lane products
  0.36. Port-bound; Shoup has the same uop count there.
- Gate cases, this branch (a5350ea) against main 0d2355e, 11 rounds, geomean of `tables/<lg>`:
  Zen 3 0.743, Emerald Rapids 0.775, Zen 5 0.736, Neoverse V2 0.724 (Measurements). Frontend
  `factorials/<lg>`: 0.861, 0.914, 0.912, 0.840. Earlier builds: before the GCD inverse, Sse
  tiers and accumulators 0.792 / 0.828 / 0.705 / 0.756 (main's Zen 5 lg 10 took 1456 ns in that
  run, 765 in the others); before the accumulators 0.735 / 0.782 / 0.745 / 0.722 (Zen 3 lg 8
  292 ns there against 332 here, with the same code run at that size). Tests pass at -O2 and
  ASan/UBSan on every target's test machines and the Mac; the test takes 42-86 s per target
  (`lc-amd`, `lc-arm`).
- Open: (1) Emerald Rapids from 2^14 is at its p0/p5 bound with this algorithm (0.97-1.0 of
  main); fewer uops per position needs fewer products, not other ones. (2) Zen 5's zmm Shoup
  loses to Montgomery by register pressure and shifts (above); a Shoup step with fewer live
  vectors might win there. (3) At n = 64 the inversion is still 25-30% of the time (40-50 ns,
  a latency chain); the up chains could run while it completes.

## Sources

- lc-opt `lib/poly/factorials.hpp` and its notes (Factorials): the product chains, the scan with
  in-register 8 x 8 transposes, `scan_chunk`. Moved and rewritten here.
- Montgomery multiplication: modarith (P. Montgomery, Math. Comp. 44, 1985).
- Finite differences for the pair (round 1) and quad (round 2) products: the standard
  forward-difference table of a polynomial; no source used.
- Shoup multiplication (round 3, the scan's chains): modarith (V. Shoup, NTL; D. Harvey, J. Symb.
  Comp. 60, 2014). The stepped quotient (a sum of two floors, one short at most) is ours.
- Binary extended Euclid (round 3, the inversion): J. Stein's binary GCD; the halvings deferred
  to the other coefficient as doublings: the idea of T. Pornin, "Optimized Binary GCD for Modular
  Inversion" (IACR ePrint 2020/972); no code used.
