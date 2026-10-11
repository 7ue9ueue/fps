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
  only): the start of the lanes past it. The lane starts and the multipliers' bases are vector
  ops (round 3; before, two scalar `% P` per lane: 31-65 ns of setup).
- Pass 2, `starts` (round 3): first[s] = (a_s)! and 1 / last[s] = 1 / (a_s + C - 1)! in vector
  registers. first is an exclusive prefix product over the lanes of g_s = (a_s + 1) ... (a_(s+1))
  = product[s] (a_s + C) (h! shifted in at lane 0; the factor up to n - C at lane k - 1; 1 past
  it), and 1 / last[s] = 1 / last[k - 1] times an exclusive suffix product of r_t = a_t
  product[t] (lane t's own positions). Both in Montgomery form, log2(S::lanes) shifted products
  within a vector (valignd, palignr, ext; Avx2 vpermd and a blend) and log2(K) across vectors.
  One scalar exponentiation inverts last[k - 1] last[k] (the lanes past n - C all end at n - 1),
  in scalar Montgomery form; two products split it. Before: k / 4 + 4 dependent scalar `% P`
  products, ~4 k in all, then x^(P - 2) for all W lanes in vectors (150-230 ns: 30 steps of 4
  contended vector products).
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
  | `avx2` (Zen 3) | Avx2.1.shoup 32, Avx2.2.shoup 2048 |
  | `avx512` (Emerald Rapids) | Avx512Vl.1.mont 32, Avx512Vl.2.mont 256, Avx512.2.mont 1024, Avx512Vl.2.mont 2^19 |
  | `zen5` | Avx2.1.shoup 32, Avx2.2.shoup 256, Avx512.2.mont 2048, Avx2.2.shoup 5 2^20 |
  | `arm` (Neoverse V2) | Neon.2.shoup 32, Neon.4.shoup 1280 |
  | `sse`, `scalar` | Sse.2.shoup 32; Scalar.4.mont 64 (not timed) |

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
untouched too. The selected `tables` also at 5 2^20 + 17 (Zen 5's last tier). Passes at -O2 and
ASan/UBSan on `lc-amd` (scalar, sse, avx2), `lc-intel-test` (avx512), `lc-zen5-test`, `lc-arm`
and the Mac (scalar, arm). The frontend's `test_factorials.cpp` runs the kernel through
`Factorials` and through `detail::factorial_tables` for every n <= 4099 at P = 4099.

## Measurements

`bench.cpp`, P = 998244353, buffers allocated once (warm), median ns per position over 11 rounds
(each position is two words, one per table). GCC 15.2, `tools/kernel.py` flags, 2026-10-11,
round 2 (this branch on main 626819e); round 1's numbers in brackets.

| lg | Zen 3 `lc-bench` (avx2) | Emerald Rapids `lc-intel` (avx512) | Zen 5 `lc-zen5` | Neoverse V2 `lc-arm` |
|---:|---:|---:|---:|---:|
| 6 | 6.85 (7.70; loop 8.24) | 6.66 (loop) | 6.63 (loop) | 7.73 (8.09; loop 9.55) |
| 8 | 2.27 (2.43) | 2.44 (2.85) | 1.83 (2.21) | 2.63 (2.78) |
| 10 | 1.18 (1.30; loop 7.02) | 0.837 (0.975; loop 5.79) | 0.746 (0.751; loop 5.88) | 1.49 (1.59; loop 8.13) |
| 12 | 0.704 (0.776) | 0.553 (0.626) | 0.419 (0.480) | 0.999 (1.07) |
| 14 | 0.605 (0.658) | 0.433 (0.484) | 0.275 (0.320) | 0.875 (0.935) |
| 16 | 0.583 (0.654) | 0.403 (0.449) | 0.238 (0.281) | 0.844 (0.901) |
| 18 | 0.597 (0.663) | 0.395 (0.457) | 0.361 (0.385) | 0.838 (0.895) |
| 20 | 0.595 (0.666) | 0.493 (ymm; 0.575) | 0.352 (0.379) | 0.840 (0.894) |
| 22 | 0.611 (0.727) | 0.493 (ymm; 0.582) | 0.380 (0.464) | 0.995 (0.918) |

The tables in the bench are two std::vectors, 16 bytes off a 64-byte boundary from lg 15 on.
Neoverse V2 at lg 22 varies by run: main's own time was 1.05 in the same run (round 1: 0.918).

Through the frontend (`frontend/bench.cpp`, `factorials/<lg>`: since #51 two unwritten tables
per call, not zero-filled vectors), new/base against main 626819e, 11 rounds, lg 10 / 14 / 18 /
22: Zen 3 0.917 / 0.911 / 0.986 / 0.984, Emerald Rapids 0.848 / 0.885 / 0.970 / 0.963, Zen 5
0.898 / 0.871 / 0.971 / 0.978, Neoverse V2 0.945 / 0.936 / 0.975 / 0.979. From lg 18 on the page faults of the
fresh tables cost more than the kernel (2-4 ns per position against 0.4-0.8).

Passes at n = 2^18 (`explore/phases.cpp`, best of 7, ns per position; round 2's lane products,
round 1 in brackets):

| Machine (S, K) | lane products | inversion | scan | all |
|---|---:|---:|---:|---:|
| Zen 3 (avx2, 2) | 0.066 (0.120) | 216 ns | 0.545 | 0.595 |
| Emerald Rapids (zmm, 2) | 0.046 (0.092) | 216 ns | 0.360 | 0.389 |
| Zen 5 (zmm, 2) | 0.036 (0.071) | 168 ns | 0.303 | 0.351 |
| Neoverse V2 (neon, 4) | 0.114 (0.172) | 236 ns | 0.720 | 0.837 |

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

## Sources

- lc-opt `lib/poly/factorials.hpp` and its notes (Factorials): the product chains, the scan with
  in-register 8 x 8 transposes, `scan_chunk`. Moved and rewritten here.
- Montgomery multiplication: modarith (P. Montgomery, Math. Comp. 44, 1985).
- Finite differences for the pair (round 1) and quad (round 2) products: the standard
  forward-difference table of a polynomial; no source used.
