# backend/small

Products of short polynomials modulo P, 2^12 < P < 2^30. API: `small.hpp` (`multiply`,
`beats_ntt`). Variant: `schoolbook.hpp` (`Schoolbook<S, R, P>`, templates on any `backend/simd`
struct). Model: `model.hpp` (O(n m), one reduction per term). `select.hpp`: the struct, R and the
cost model per target. Tests: `test.cpp`; bench: `bench.cpp`.

## Design

- Long factor a (n words), short factor b (m <= n). Outputs in blocks of R vectors (R W words, W
  the struct's lanes). For each j, every vector of the block adds a[k - j] b'[j] for its words k:
  one unaligned load of a at k0 + r W - j, and one broadcast of b'[j] shared by the R vectors.
- Sums in 64-bit lanes:
  - x86 (`mul_even`): two vectors per output vector, the even and odd words. The odd words of the
    window at k0 + r W - j are the even words of the window one word on, which the step before
    loaded, so a step is 1 load, 2 `mul_even` and 2 `add64` per output vector (`EvenOdd`).
  - Neon: words 0-1 and 2-3, `umlal` and `umlal2` (`Halves`).
  - Scalar: one `uint64` per word (`Single`).
- b' = b 2^32 mod P (Montgomery form, by Shoup's product), so one Montgomery step (s + m P) / 2^32
  at the end gives the sum of a b. Bounds (`Sums<P>`): terms below (P - 1)^2; a fold
  s = h 2^32 + l -> h (2^32 mod P) + l keeps s below 2^32 (2^32 mod P + 1) and below 2^64 for
  `kFold` more terms; the last step needs s < 3P 2^32 (then it is below 4P, and two min-reductions
  make it canonical). For 998244353: a fold every 17 terms, at most 12 terms before the last step
  from zero and 11 after a fold.
- Interior blocks (k0 from m - 1 to n - R W - 1) run every j < m on every vector and load inside
  a. Edge blocks: each vector has its own range of j ([s + 1 - n, s + W) within [0, m), s its first
  word); both ends grow with the vector, so the j loop runs in phases with a fixed set of active
  vectors [first, last), each phase an unrolled loop picked by a chain of compares. Loads that
  would leave [0, n) read a zero-padded copy of the block's window of a (on the stack). The last
  block's tail goes through a stack buffer.
- b in panels of at most 1024 words, so the stack scratch is fixed (about 9 KiB); panels after the
  first add into out. Any n, m work.
- `beats_ntt(n, m)`: small's time modelled as term n m + word (n + m) + call + square min(n, m)^2
  ns, fitted per target on shapes within 0.4x-2.5x of the NTT; the NTT's time per transform
  length 2^lg, measured with the product in the caller's memory (construct, copy the factors in,
  multiply). Past 2^25 the NTT counts double per step, so huge balanced products still go to the
  NTT (which throws) and a short factor of any length goes to small.

## Measurements

GCC 15.2, `tools/kernel.py` flags, `lc-bench` (Zen 3), `lc-intel` (Emerald Rapids), `lc-zen5`
(Zen 5), `lc-arm` (Neoverse V2), 2026-10-11, this branch on main e754d90 (the NTT of issue #38,
round 1). `bench.cpp`, 3 rounds, median ns per call (ns per term), the selected R; `ntt`: the
`multiply@ntt` case (construct, copy the factors in, multiply into the same output).

| Shape | Zen 3 | ntt | Emerald Rapids | ntt | Zen 5 | ntt | Neoverse V2 | ntt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 46 (0.72) | 116 | 43 (0.67) | 149 | 34 (0.53) | 94 | 50 (0.78) | 234 |
| 16x16 | 73 (0.29) | 113 | 61 (0.24) | 144 | 45 (0.18) | 94 | 112 (0.44) | 234 |
| 32x32 | 167 (0.16) | 111 | 121 (0.12) | 145 | 68 (0.066) | 92 | 274 (0.27) | 232 |
| 64x64 | 430 (0.105) | 240 | 316 (0.077) | 228 | 173 (0.042) | 167 | 782 (0.19) | 478 |
| 128x128 | 1296 (0.079) | 497 | 953 (0.058) | 503 | 482 (0.029) | 338 | 2435 (0.149) | 1030 |
| 512x512 | 15753 (0.060) | 2188 | 11948 (0.046) | 2229 | 5785 (0.022) | 1295 | 32548 (0.124) | 4764 |
| 4096x4 | 1698 (0.104) | 22005 | 1235 (0.075) | 22769 | 653 (0.040) | 12304 | 3433 (0.21) | 46696 |
| 4096x16 | 4537 (0.069) | 21999 | 3282 (0.050) | 22758 | 1643 (0.025) | 12338 | 9044 (0.138) | 46688 |
| 4096x128 | 27792 (0.053) | 21995 | 21890 (0.042) | 22526 | 10564 (0.020) | 12310 | 58551 (0.112) | 46711 |
| 131072x64 | 463805 (0.055) | 990530 | 360091 (0.043) | 1057486 | 173627 (0.021) | 552531 | 967401 (0.115) | 2071638 |

The frontend's old loop (`detail::schoolbook`, scalar, 64-bit sums reduced by `% P` every 16
terms) took 0.24 ns per term on Zen 3 (`frontend/notes.md`).

Long products, cycles per term against the floor (multiplies over multiply pipes):
- Zen 3, ymm: 0.053 ns = 0.18 cycles at 3.45 GHz, 5.4 terms per cycle. Floor 8 (2 `vpmuludq`
  per 8 terms on 2 pipes); with the measured cost of a `vpmuludq` next to a 4-pipe op (0.633
  cycles per pair, `backend/ntt/notes.md`) 6.3. The loop (GCC): per j and 4 vectors, 1
  `vpbroadcastd`, 4 loads, 8 `vpmuludq`, 8 `vpaddq`, 3 scalar ops.
- Emerald Rapids, zmm: 0.042 ns, about 6.3 terms per cycle at 3.8 GHz (`backend/simd/notes.md`:
  zmm streams run at 3.7-4.0 GHz). Floor 8: `vpmuludq` and `vpaddq` share p0 and p5.
- Zen 5, zmm: 0.020 ns = 0.082 cycles at 4.12 GHz, 12 terms per cycle; floor 16.
- Neoverse V2: 0.112 ns = 0.33 cycles at 2.97 GHz, 3 terms per cycle; floor 4 (`umlal` and
  `umlal2` on 2 pipes). GCC keeps windows in registers across j (a window of vector r at step j
  is the window of vector r + 1 at step j + 4) and adds 9 vector moves per 6 vectors.

R: geomean over the bench shapes against the best R per machine: Zen 3 R = 2, 4, 6, 8: 1.12,
1.00, 1.33, 1.40 (3 vectors per output vector on 16 registers: 6 and 8 spill); Emerald Rapids
1.10, 1.01, 1.00, 1.02; Zen 5 1.22, 1.03, 1.00, 1.01; Neoverse V2 1.32, 1.04, 1.00, 1.08.
Avx512Vl (ymm, 32 registers) on AVX-512 machines: 1.25 (Emerald Rapids), 1.80 (Zen 5) of zmm.

### Cutoff against the NTT

`sweep.cpp` (`~/Documents/hpc_library-explore/small/r1/`): for each length 2^lg, lg = 6..17,
shapes with n + m - 1 = 2^lg and = 2^(lg-1) + 1, m = 1 .. 1024 (186 shapes); small::multiply and
the Convolution with the product in the caller's memory, interleaved, median of 5. The fit
(select.hpp) on the shapes within 0.4x-2.5x of the NTT, then beats_ntt against every swept shape:
wrong on 2 of 186 shapes on Zen 3 (loss at most 4%), 3 on Emerald Rapids (2%), 1 on Zen 5 (1%),
4 on Neoverse V2 (12% at 97x32, the others 1-4%). Fits, ns: term, word, call, min^2:

| Machine | term | word | call | min^2 |
|---|---:|---:|---:|---:|
| Zen 3 | 0.0501 | 0.265 | 60.1 | 0.027 |
| Emerald Rapids | 0.0386 | 0.209 | 43.3 | 0.014 |
| Zen 5 | 0.0186 | 0.148 | 36.2 | 0.0068 |
| Neoverse V2 | 0.107 | 0.515 | 58.7 | 0.043 |
| sse (lg <= 14 on `lc-bench`, not a timed target) | 0.106 | 0.498 | 83.7 | 0.059 |
| scalar (lg <= 14 on `lc-bench`, not a timed target) | 0.311 | 2.30 | 149.3 | 0.157 |

The NTT table past the sweep: the ntt bench's `multiply/<lg>` from the gate of PR #48 (raw times
on the Mac, `gate/pr48/<target>-ntt.json`), scaled by the sweep's time at 2^17 over the bench's
(1.03-1.06: the construct and copies); for sse and scalar 2.1x per lg (a guess).

Where beats_ntt picks small (`cutoffs.py` there):

| Machine | n x n | 2^10 x m | 2^14 x m | 2^20 x m | (2^10 - m) x m | (2^14 - m) x m | (2^20 - m) x m |
|---|---|---|---|---|---|---|---|
| Zen 3 | 1-25, 33-50, 65-75 | 1-81 | 1-119 | 1-174 | 1-38 | 1-53 | 1-79 |
| Emerald Rapids | 1-56, 65-90, 129-135 | 1-117 | 1-160 | 1-266 | 1-55 | 1-73 | 1-126 |
| Zen 5 | 1-104, 129-157 | 1-129 | 1-175 | 1-248 | 1-61 | 1-81 | 1-116 |
| Neoverse V2 | 1-31, 33-51, 65-78 | 1-87 | 1-118 | 1-171 | 1-39 | 1-53 | 1-79 |

The NTT's length doubles when n + m - 1 passes a power of two, so the cutoff jumps there: n = 2^k
with any m > 1 pays for 2^(k+1).

## Log

2026-10-11, claude (issue #37, round 1): new kernel; the frontend's `f * g` calls it.
- Task: move the frontend's `detail::schoolbook` here, fast per target on `simd` and `modarith`,
  Karatsuba where it wins, one call for the frontend, the cutoff against `ntt::Convolution` per
  target. The NTT on main when measured: e754d90 (issue #38 round 1: memory kept per thread; a
  32 x 32 Convolution about 110 ns on Zen 3, was 56 us).
- First version: a block's terms j in [j0, j1) for all R vectors, the window copied by a scalar
  loop with a bounds check per word. Zen 3, R = 4: 4096x128 0.056 ns per term, 32x32 185 ns,
  64x64 538 ns, 128x128 1600 ns.
- Clang (the Mac) did not inline the generic lambda in an `unrolled<R>` helper (index sequence),
  so the sums went through memory: 4096x32 0.43 ns per term on the M4 against 0.085 with
  `#pragma GCC unroll` loops.
- Exact term ranges per vector (phases) and the interior blocks apart: Zen 3 32x32 168 ns,
  64x64 429, 128x128 1299 (were 185, 538, 1600): the first version spent up to 2x the terms of a
  block on zeros at the ends of a balanced product. A window copy by `fill` / `copy` instead of
  the per-word loop. Long products unchanged.
- Karatsuba over the schoolbook (`karatsuba.cpp` there: one level per call down to a threshold
  T, sums and differences by modarith `add` / `sub`), balanced n x n, ns:

  | n | Zen 3 school / best K / ntt | Emerald Rapids | Zen 5 | Neoverse V2 |
  |---:|---|---|---|---|
  | 64 | 433 / 541 / 295 | 315 / 403 / 222 | 180 / 244 / 191 | 770 / 893 / 482 |
  | 128 | 1301 / 1339 / 489 | 932 / 969 / 496 | 502 / 572 / 341 | 2422 / 2406 / 1037 |
  | 256 | 4365 / 4081 / 1078 | 3200 / 3009 / 1053 | 1634 / 1736 / 696 | 8542 / 7367 / 2205 |
  | 512 | 15681 / 12348 / 2217 | 11966 / 8945 / 2218 | 5790 / 5402 / 1303 | 32023 / 22429 / 4785 |

  It beats the schoolbook only from 128-384 on (by at most 30%), where the NTT is 2.2-5x faster
  than either. Not kept. Unbalanced shapes would need balanced sub-products, which run slower per
  term than the long ones (0.08-0.10 against 0.053 ns on Zen 3), against an NTT that wins there
  too.
- R per target and the cost model: Measurements. The cutoff replaces the frontend's
  `min(n, m) <= 32 || n m < 3 2^16`, set when a Convolution cost 51 us.
- Tests: all P - 1 does not give the largest sums here, since b enters as b 2^32 mod P; the
  tests give b as x 2^-32 for x = P - 1 or near it. Mutations, each caught: the last-step bounds
  from zero and from a fold (+1), the fold interval (+1), no last fold, an edge range end (-1)
  and start (+1), the panel add dropped, the 2P reduction dropped, and the interior loop one block
  longer (x86 only: a load of one word past a, caught by ASan on `lc-amd`).

## Sources

- The even/odd 64-bit sums with one Montgomery step at the end, and the fold of the high word by
  2^32 mod P: lc-opt's `multiply_lanes` (`lib/poly/product_tree.hpp`, ours), rewritten for one
  product across the lanes instead of one product per lane.
- Montgomery: P. Montgomery, Math. Comp. 44 (1985). Shoup: V. Shoup, NTL; D. Harvey, J. Symb.
  Comp. 60 (2014). Karatsuba: A. Karatsuba, Yu. Ofman, Doklady Akad. Nauk SSSR 145 (1962).
