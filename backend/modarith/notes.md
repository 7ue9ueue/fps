# backend/modarith

Vector arithmetic modulo a prime 2^12 < P < 2^30 known at compile time, as templates on a simd
struct: `modarith::Ops<S, P>` (`modarith.hpp`, API and contracts at the top). `select.hpp` picks
the variant per target; `model.hpp` is the scalar model the tests run every variant against.

## Design

- Lanes are uint32. Each operation states its input ranges and its output range; lazy values
  above P are brought down by `reduce<M>` (min(x, x - M)).
- Montgomery, `montgomery(a, b)` = a b 2^-32, a, b < 2^31, a b < 2^32 P, result in [0, 2P):
  - `wide`: x86, lc-opt's form: even and odd lanes as 64-bit products (`vpmuludq`),
    (t + m P) / 2^32 with m = -t P^-1 mod 2^32; 6 `vpmuludq`. Neon: `umull`, `umull2`, one `mul`
    for m, `umlal`, `umlal2`, `uzp2`.
  - `mulhi`: any struct: m = (a b) P^-1 mod 2^32 by two `mul_lo`, hi(a b) - hi(m P) in (-P, P), + P.
  - `doubling`: Neon. Signed m; `sqdmulh` gives the high words of 2 a b and 2 m P, which differ by
    exactly 2r; `shsub` halves the difference. r is in (-P/2, 3P/2), and min(r, r + P) maps it to
    [0, 3P/2). 2 `sqdmulh` + 2 `mul` + 3 simple ops.
- Shoup, `shoup(x, w, q)` = x w for a fixed w with q = floor(w 2^32 / P), x < 2^32, result in
  [0, 2P): `mulhi` (x w - mulhi(x, q) P, Neon fuses the last step into `mls`) or `wide` (x86, all
  64-bit products).
- Barrett, `barrett(a, b)` = a b for a, b < P: high = (a b) >> (bits - 1), q = mulhi(high, m) with
  m = floor(2^(bits+31) / P). The quotient is short by less than 1 + E,
  E = P^2 / 2^(bits+31) + 2^(bits-1) / P, so the result is below 2P when E < 1 (998244353,
  1000000007) and below 3P otherwise (e.g. 268435459, 536870923); `kBarrettBound` is computed
  per P.
- Variants with their own contracts: `shoup_doubling` (Neon, x < 2^31, q31 = floor(w 2^31 / P):
  `sqdmulh`, `mul`, `mls`), `montgomery52` and `shoup52` (IFMA, 64-bit lanes: words split into
  even and odd halves and joined back; montgomery52 gives a b 2^-52; shoup52 needs no mask, as the
  low 32 bits of the 52-bit sum are the result).
- Selection (`select.hpp`): Montgomery `wide` on x86, `doubling` on Neon, `mulhi` where the
  struct lacks the variant (Scalar). Shoup `mulhi` everywhere.

## Tests

`test.cpp`, for every struct the machine can run and 8 primes (4099, 65537, 268435459,
469762049, 536870923, 998244353, 1000000007, 1073741789): every operation and variant, 300 random
vectors per input bound, inputs drawn below each bound with bound - 1, 0 and 1 often. Bounds
include each contract's limits: Montgomery (P, P), (2P, 2P), (4P, P), (2^31, the largest B with
(2^31 - 1)(B - 1) < 2^32 P), (4P, its partner); Shoup x < P, 4P, 2^31, 2^32; reduce at 2M - 1. The
model checks the bounds against the contract (a test drawing outside it fails) and every lane must
be below the model's bound and congruent to its value. Mutation check: removing the min in
`doubling` fails at (2^31, 2P + 1) for P = 4099. Passes at -O2 and ASan/UBSan on every target's
test machines and the Mac.

## Measurements

`bench.cpp`, P = 998244353, out[i] = op(a[i], b[i]) (Shoup: constant w) over 2048 words, 11
rounds, median ns per word and cycles per vector (16 lanes on AVX-512, 8 on Zen 3, 4 on Neon).
GCC 15.2, `tools/kernel.py` flags, 2026-10-10, commit f4baa1e. Clocks: `backend/simd/notes.md`.
IQR within 2.5% except where noted.

| Case (selected variant) | Zen 3 `lc-bench` | Emerald Rapids `lc-intel` | Zen 5 `lc-zen5` | Neoverse V2 `lc-arm` |
|---|---|---|---|---|
| add | 0.045 (1.3) | 0.038 (2.6) | 0.020 (1.3, IQR 11%) | 0.165 (2.0) |
| sub | 0.045 (1.3) | 0.038 (2.5) | 0.015 (1.0, IQR 10%) | 0.166 (2.0) |
| montgomery | 0.139 (3.8) | 0.097 (6.5) | 0.049 (3.2) | 0.251 (3.0) |
| shoup | 0.089 (2.5) | 0.077 (5.1) | 0.032 (2.1) | 0.253 (3.0) |
| barrett | 0.172 (4.8) | 0.134 (9.0) | 0.062 (4.1) | 0.289 (3.5) |

Floors (multiply uops over multiply pipes, cycles per vector): Montgomery `wide` 3 on Zen 3 and 5
(6 `vpmuludq` on 2 pipes; measured 3.8 and 3.2); 6 on Emerald Rapids zmm (12 uops, multiplies
included, all on p0/p5; measured 6.5); Neon `doubling` 2 (4 multiplies on V0/V2; 7 ops on 4 pipes
need 1.75; measured 3.0). Shoup 2 on Zen 3 and 5 (4 multiplies; measured 2.5 and 2.1), 5 on
Emerald Rapids (2 `vpmuludq` and 2 `vpmulld` are 6 uops, plus shifts, join and sub, all on p0/p5;
measured 5.1).

Variants, paired ratios in the same run (median, IQR):

| Ratio | Zen 3 | Emerald Rapids | Zen 5 | Neoverse V2 |
|---|---|---|---|---|
| montgomery wide / mulhi | 0.861 | 0.780 | 0.813 | 0.975 |
| montgomery doubling / wide | - | - | - | 0.834 (0.833-0.835) |
| shoup mulhi / wide | 0.641 | 0.740 | 0.597 | - |
| shoup_doubling / shoup mulhi | - | - | - | 0.669 |
| montgomery zmm / ymm | - | 0.747 | 0.509 | - |
| montgomery52 (IFMA) / montgomery wide | - | 1.452 | 1.611 | - |
| shoup52 (IFMA) / shoup mulhi | - | 1.097 | 1.461 | - |

Latency, 1024 dependent products on one vector (ns per product, cycles): Zen 3 wide 3.66 (12.7),
mulhi 2.68 (9.3), Shoup 2.63 (9.1); Emerald Rapids zmm wide 5.76 (24), Shoup 5.78 (24);
Zen 5 wide 4.37 (18), Shoup 2.91 (12); Neoverse V2 wide 4.35 (13.0), doubling 5.12 (15.3),
shoup_doubling 2.79 (8.3). Raw tables: `~/Documents/hpc_library-explore/simd/r1-measure/`.

- IFMA does not pay for 32-bit words: splitting into 64-bit lanes doubles the work. Its use is
  multimod with ~50-bit primes (not tried).
- On Neoverse V2 the add and sub streams take 2 cycles per vector for 3 simple ops (floor 0.75):
  the loop is 9 instructions (`perf`: IPC 4.4, 25% backend stall cycles). Guess: the dependent
  chain of 3 ops at latency 2 per vector fills the SIMD issue queues. Not fixed this round.

## Log

2026-10-10, claude (issue #2, round 1). New kernel, no base to gate against, so variants were
timed against each other in the same run (Measurements).
- Montgomery on Neon: `doubling` (`sqdmulh`, `shsub`) selected: 0.834 of `wide`, 0.81 of `mulhi`
  (`lc-arm`, 11 rounds). Its chain latency is 18% longer (15.3 against 13.0 cycles).
- Montgomery on x86: `wide` (lc-opt's) beats `mulhi` everywhere, most on Emerald Rapids where
  `vpmulld` is 2 uops (0.780).
- Shoup: `mulhi` beats `wide` on all x86 (0.60-0.74). Neon `shoup_doubling` is 0.669 of `mulhi`
  but needs x < 2^31, so it is not the default (`shoup` takes x < 2^32, e.g. lazy values < 4P for
  P > 2^29).
- AVX-512: Montgomery `wide` joins its 64-bit halves with `vpermt2d` (simd `join_high`): 0.984 on
  Emerald Rapids, 0.933 on Zen 5 (paired against commit 3ba9970, 11 rounds); details in
  `backend/simd/notes.md`.
- Barrett's error bound: the first estimate (error < 2.5 for all P) gave 3P; the tighter
  E = P^2 / 2^(bits+31) + 2^(bits-1) / P gives 2P for the common primes.
- Bug found by the tests: `shoup_quotient52` shifted w << 52 in 64 bits (overflow for w >= 2^12).

## Sources

- Shoup multiplication: V. Shoup, NTL; D. Harvey, "Faster arithmetic for number-theoretic
  transforms", J. Symb. Comp. 60 (2014) (lazy ranges).
- Montgomery: P. Montgomery, "Modular multiplication without trial division", Math. Comp. 44
  (1985). The `wide` x86 form is moved from lc-opt (`lib/poly/calculus.hpp`, `montgomery`).
- Barrett: P. Barrett, CRYPTO 1986.
- Signed Montgomery with `sqdmulh`/`shsub`: idea from H. Becker, V. Hwang, M. J. Kannwischer,
  B.-Y. Yang, S.-Y. Yang, "Neon NTT: Faster Dilithium, Kyber, and Saber on Cortex-A72 and Apple
  M1", TCHES 2022. Rewritten for unsigned 32-bit lanes and P < 2^30; no code used.
- IFMA: Intel's instruction set reference (`vpmadd52luq`, `vpmadd52huq`).
