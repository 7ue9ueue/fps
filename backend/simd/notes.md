# backend/simd

Vector wrappers, one struct per ISA, so kernels are templates written once (`simd.hpp`, API at the
top). `select.hpp` picks `simd::Native` per target and lists every struct the target can run
(`simd::All`). Also here: `probe.cpp` (instruction latency and throughput in cycles),
`machines.py` (machine models of the four timing machines), `stream.hpp` (bench helper).

## Design

- Structs `Scalar`, `Sse` (SSE4.1), `Avx2`, `Avx512Vl` (Avx2's ops on ymm with 32 registers, IFMA,
  `vpermt2d`), `Avx512` (zmm), `Neon`. Static inline functions on 32-bit lanes; no state.
- Thin wrappers are one instruction each. Composed ops say their instruction count in the header:
  `mulhi` (high word of a b), `mul_shr<k>`, `join_high` (high words of two 64-bit-lane vectors,
  interleaved), x86 `mul_sub`.
- x86 has `mul_even` (`vpmuludq`) and 64-bit lane ops; Neon has none of that and instead products
  of vector halves (`umull`, `umull2`, `umlal`), `uzp1/uzp2`, and the signed doubling ops
  (`sqdmulh`, `shsub`). Kernels test for an op with `requires` and fall back to `mulhi`/`mul_lo`.
- `permute` takes an `Index` built off the hot path, so it is one instruction everywhere (byte
  tables for `pshufb` and `tbl`, word indices for `vpermd`).
- `Native`: `Avx512` (zmm) on `avx512` and `zen5`, `Avx2`, `Sse`, `Neon`, `Scalar`. zmm against
  ymm (Avx512Vl), modarith streams: Emerald Rapids 1.34x (Montgomery) and 1.39x (Shoup) faster,
  Zen 5 1.96x and 2.04x.
- `mulhi` on AVX-512 joins its halves with one `vpermt2d` instead of `vpsrlq` + blend, except on
  Zen 4/5 (`__znver4__`, `__znver5__`), where Shoup got 5% slower with it (Log).
- SVE2: `Neon::mulhi` is one SVE2 `umulh` on the low 128 bits of the z registers when
  `__ARM_FEATURE_SVE2` is set (`arm` target; not the Mac's M4), through the NEON-SVE bridge
  (`svset_neonq`, `svget_neonq`), so it is right for any SVE vector length. Neoverse V2's SVE
  vectors are 128-bit, so the bridge costs nothing. `mul_shr` keeps `umull`/`shrn` (Log).
- `shr<k>`, `shl<k>`: 32-bit lane shifts on every struct (round 2, for modarith's Barrett `split`).
- `Avx512` halves from and to memory: `load_halves`, `store_half<h>`, `broadcast_halves` (ntt
  issue #5, for the zmm leaf products: two groups per vector, one per half).
- `store_first(p, a, k)`: the first k lanes, nothing past them (small issue #37, round 2: the last
  output vector of a product). AVX-512 and Avx512Vl: one masked store; Sse, Avx2, Neon: pieces of
  4, 2 and 1 words from the register. Neon `mls_wide_lo/hi` (`umlsl`, `umlsl2`), for small's fold.

## Tests

`test.cpp`: every struct the machine can run, 2000 random trials each (words drawn with edge
values 0, 1, 2^31 - 1, 2^31, 2^32 - 2, 2^32 - 1 one time in eight), every wrapper against lane-wise
scalar arithmetic: unaligned loads and stores at every offset, `mul_shr<k>` for k = 1, 7, 16, 29,
31, 32 with products at the 2^(32+k) limit, `shr`/`shl` for k = 1, 13, 31, blends with 6 masks, random permutes, IFMA against
128-bit products, `sqdmulh` saturation. Passes at -O2 and ASan/UBSan on every target's test
machines (`lc-amd` scalar/sse/avx2, `lc-intel-test` avx512, `lc-zen5`, `lc-arm`, Mac scalar/arm).

## Instruction costs

`probe.cpp`, GCC 15.2, cycles (latency / reciprocal throughput). Clock from a dependent chain of
register adds in the same run: Zen 3 3.46 GHz, Emerald Rapids 4.18, Zen 5 4.12, Neoverse V2 2.97.
`machines.py` holds these with pipe sets. 2026-10-10, commit f4baa1e.

| Instruction | Zen 3 (`lc-bench`) | Emerald Rapids (`lc-intel`) | Zen 5 (`lc-zen5`) |
|---|---|---|---|
| vpaddd ymm | 1 / 0.25 | 1.67 / 0.33 | 2 / 0.25 |
| vpminud ymm | 1 / 0.25 | 2 / 0.5 | 2 / 0.25 |
| vpmuludq ymm | 3 / 0.5 | 5 / 0.5 | 3 / 0.5 |
| vpmulld ymm | 3 / 0.5 | 10 / 1.03 | 3 / 0.5 |
| vpsrlq ymm | 1 / 0.5 | 1 / 0.5 | 2 / 0.5 |
| vpblendd ymm | 1 / 0.25 | 1.67 / 0.33 | 2 / 0.25 |
| vpermd ymm | 3 / 1.34 | 3 / 1.0 | 4 / 0.5 |
| vpermt2d ymm | - | 3 / 1.0 | 4 / 0.5 |
| vpmadd52luq ymm | - | 4 / 0.5 | 4 / 0.5 |
| vpaddd zmm | - | 2.5 / 0.5 | 2 / 0.25 |
| vpminud zmm | - | 3 / 1.0 | 2 / 0.25 |
| vpmuludq zmm | - | 5.8 / 0.53 | 3 / 0.5 |
| vpmulld zmm | - | 11.5 / 1.06 | 3 / 0.5 |
| vpsrlq zmm | - | 1 / 1.0 | 2 / 0.5 |
| vpblendmd zmm | - | 2.5 / 0.5 | 2 / 0.25 |
| vpermt2d zmm | - | 4 / 1.0 | 5 / 0.5 |
| vpmadd52luq zmm | - | 4.2 / 0.53 | 4 / 0.5 |

Mixed pairs (reciprocal throughput per instruction): `vpmuludq + vpmulld` 0.5 on Zen 3 and 5
(same pipes), 0.75 ymm / 0.79 zmm on Emerald Rapids (`vpmulld` is 2 uops); `vpmuludq + vpsrlq`
0.25 on Zen (disjoint pipes), 0.5 / 0.53 on Emerald Rapids; IFMA + `vpmuludq` 0.5 everywhere
(shared pipes).

Neoverse V2 (`lc-arm`), 128-bit: add, sub, umin, shsub, uzp1, uzp2, bsl, tbl 2 / 0.25; mul, umull,
umull2, sqdmulh 4 / 0.5; mla, mls, umlal, umlal2 1.15 / 0.5 (through the accumulator); ushr, shrn
2 / 0.5. Mixed: mul + umull and mul + sqdmulh 0.5 (shared pipes), mul + add and umull + uzp2 0.33
(not 0.25). SVE2 `umulh z.s` 4 / 0.5, on the multiply pipes (mul + umulh 0.5, umulh + add 0.33);
`usra` 1.15 / 0.5 through the accumulator (2026-10-11, round 2).

Per CPU, for the kernels:
- `vpmulld` costs as much as `vpmuludq` on Zen 3 and 5, twice (2 uops, latency 10-11.5) on Emerald
  Rapids.
- On Emerald Rapids, zmm shifts and `vpminud` run on one port (p0), half the ymm rate.
- Zen 5 has latency 2 for simple integer ops (Zen 3: 1) and four full 512-bit pipes.
- On Neoverse V2, accumulating multiplies chain at ~1 cycle through the accumulator.

## Composed products, streams

`bench.cpp`: out[i] = op(a[i], b[i]) over 2048 words, 11 rounds, median ns per word (cycles per
vector at the clock above). 2026-10-10, commit f4baa1e, GCC 15.2, `tools/kernel.py` flags.

| Case | Zen 3 | Emerald Rapids | Zen 5 | Neoverse V2 |
|---|---|---|---|---|
| mul_lo | 0.049 (1.3) | 0.032 (2.2) | 0.022 (1.5) | 0.088 (1.1) |
| mulhi | 0.075 (2.1) | 0.049 (3.3) | 0.032 (2.1) | 0.088 (1.05), SVE2 `umulh`; was 0.097 (1.2) |
| mul_shr<29> | 0.080 (2.2) | 0.063 (4.2) | 0.030 (2.0) | 0.117 (1.4) |

On Neoverse V2 every stream takes at least ~1.05 cycles per vector: 2 loads and 1 store, and the
loop's 7 instructions. mul_lo and mulhi now sit at that floor.

## Log

2026-10-10, claude (issue #2, round 1):
- Wrote the structs, tests, bench, probe and models. Tests pass on every target (Tests).
- Probe calibration: a chain of `add $1, reg` measured 24 "GHz" on `lc-intel`: Golden Cove-class
  cores fold add-immediate chains at rename. Register-register adds give 4.18 GHz.
- Bench arrays: first 4096 words with the output (48 KiB): out of Zen 3's 32 KiB L1, and
  `std::vector`'s 16-byte alignment split every zmm load. Now 2048 words, 64-byte aligned, n + 48
  words apart, read through local pointers (`stream.hpp`).
- `mulhi` join by `vpermt2d` (one uop on p5 / FP1-2) instead of `vpsrlq` + `vpblendd` (2 uops).
  Paired against the previous commit (3ba9970), same run, 11 rounds, new/base:
  `lc-intel` mulhi zmm 0.815, ymm 0.882; modarith shoup 0.961, barrett 0.951, montgomery 0.984;
  `lc-zen5` mulhi zmm 0.953, ymm 0.832, but shoup 1.051 (IQR 1.048-1.058) and every latency chain
  8-25% slower (`vpermt2d` latency 4-5 against 2 + 2). Kept on Intel; Zen 4/5 keep the blend in
  `mulhi` (montgomery's join still uses `vpermt2d` there: 0.933).
- Open: the cheapest streams (mul_lo, add, sub) are bimodal per process on some machines: Zen 5
  add 31 or 41 ns per call, Neoverse V2 mul_lo 181 or 224, Zen 3 mul_lo 80 or 102, the same binary
  in different runs. Not the data addresses: 2 MiB-aligned arrays (fixed low 21 address bits) kept
  it. Guess: code or physical-page placement. The gate's median of paired rounds should absorb a
  minority mode, but a 1-2% verdict on these cases is not reliable.

2026-10-11, claude (issue #2, round 2). All paired against main (8230b41) in the same run,
11 rounds, new/base, GCC 15.2, `tools/kernel.py` flags.
- SVE2 `umulh` for `Neon::mulhi` (1 instruction instead of `umull`, `umull2`, `uzp2`): `lc-arm`
  mulhi stream 0.907 (180 against 199 ns per 2048 words), now at the stream floor (Composed).
  modarith gains more (its notes): Shoup 0.671, Montgomery 0.800 (new selection), Barrett 0.662.
  x86 bench binaries are byte-identical.
- `mul_shr` from `umulh` and `mul` (`shl` + `usra` to join): 1.26x slower than `umull`, `umull2`,
  `shrn`, `shrn2` in the stream (302 against 240 ns): the join adds a 2-op chain. Not kept in
  simd; modarith's Barrett `split` uses the idea where `mul` is shared.
- Neoverse V2 add and sub streams at 1.8-2.0 cycles per vector (open item of round 1). Inline-asm
  loops over the same 2048 words (`neonchain.cpp` in `~/Documents/hpc_library-explore/simd/r2/`,
  cycles per vector at 2.97 GHz): the current loop 1.74-1.82; the same with two vectors per iteration 1.09; two dependent ops
  per vector (add, add) 1.05-1.30; three in a line (add, add, add) 1.74; a mul followed by the
  same add and umin 1.26-1.50; 4 ops, depth 2 after the loads (s = a + b, t = a + (b - P),
  min) 1.40; compare-and-subtract (add, cmhs, and, sub; depth 4) 1.44. Not the store: without it
  the loop still ran at the loop-carried accumulator's 2 cycles. So the core does not overlap
  enough iterations of a chain of three 2-cycle ops when they arrive one per iteration; the
  ops' count or depth alone does not predict it. Guess: issue-queue steering. A 4-op `add`
  would win this stream (1.40) but costs 33% more ALU ops in a kernel that interleaves several
  vectors (an NTT), where the stream's problem goes away (1.09). Not changed; kernels should
  interleave two or more independent vectors per loop iteration on this core.
- x86: odd words into the low halves for `mul_even` by `vpshufd` or `vmovshdup` (shuffle ports)
  instead of `vpsrlq` (zmm shifts run on p0 only on Emerald Rapids), `oddx86.cpp` there, 3
  runs: Emerald Rapids Montgomery 191 against 193 ns, Shoup 157 against 155, mulhi 105 against
  99; Zen 3 and Zen 5 within 2% both ways. No gain; not kept.

2026-10-11, claude (issue #2, round 3). No change kept; code at main 7b6d5f2 (simd and modarith
unchanged up to c4c56e9). GCC 15.2, `tools/kernel.py` flags unless noted. Files and raw output:
`~/Documents/hpc_library-explore/simd/r3/`. The modarith side of each attempt is in its notes.
- x86 odd words from an unaligned load one word further on (`load(p + 1)`, lc-opt's NTT trick)
  instead of `vpsrlq`, in streams (`oddload.cpp`, 2 runs, new/old): mulhi Zen 3 1.32, Emerald
  Rapids 1.23, Zen 5 2.06. A zmm load at 4 bytes past a 64-byte boundary always splits a cache
  line; a ymm one, every other time. Slower everywhere; not kept.
- Avx2 `join_high` by `vshufps` + `vpshufd` (two shuffles, no blend) instead of `vpsrlq` +
  `vpblendd` (`join.cpp`, then `kernel.py bench` against main, `lc-bench`, 21 rounds): simd mulhi
  1.0065 (IQR 1.0042-1.0127), modarith 0.993 (Montgomery 0.986, Shoup 0.974), but every latency
  chain 3-10% slower (Shoup chain 1.100: the shuffles add latency, guess: an int/FP bypass).
  `vpshufd` + `vpblendd` and `vpshufd` for the odd words changed nothing. With `-march=x86-64-v3`
  on the other machines: Emerald Rapids mulhi 0.92, Shoup 0.93; Zen 5 Barrett 1.21. On the
  `avx512` target the simd bench binary changed only in its `@avx2` cases, yet mulhi/2048 (same
  code) moved 1.0090 (IQR 1.0015-1.0136): binary layout. A 1-3% stream gain against a 3-10%
  latency loss; not kept.
- SVE2 `umulh` unpredicated (inline asm `umulh z.s, z.s, z.s`): GCC emits the predicated form,
  with a `movprfx` where the first source stays live (Montgomery, Shoup). Paired, `lc-arm`, 11
  rounds: Montgomery 1.0002, Shoup 0.9988, Barrett 1.0023, mulhi 1.0006. `movprfx` costs nothing
  on Neoverse V2; not kept.
- AVX-512 mul_shr and mulhi with one or both odd words by `vpshufd` (p5) instead of `vpsrlq` (p0;
  mul_shr has 4 shifts in 7 uops): Emerald Rapids mul_shr 130.1 / 130.7 / 130.5 ns (shift / both
  / one), mulhi 96.6 / 100.1 (shift / one); Zen 5 62.1 ns in every form. Not kept.
- `perf stat` on `lc-intel`, mulhi stream: 5.3 uops per vector on p0 + p5 (5 expected; GCC adds a
  `vmovdqa32` before the destructive `vpermt2d`, mostly eliminated at rename), 2.93 cycles per
  vector, p0 busy 88%. The core ran at 3.7-4.0 GHz in these zmm streams (cycles over reference
  cycles at 2.3 GHz), not the 4.18 GHz of the scalar probe clock: the Emerald Rapids cycle counts
  in Composed products are 5-10% high.

2026-10-11, claude (ntt issue #5, round 1): added `Avx512::load_halves` (ymm load, `vinserti64x4`
from memory), `store_half<h>` (ymm store, `vextracti64x4` to memory) and `broadcast_halves`
(`vpbroadcastd`, masked `vpbroadcastd` from memory). Tested against lane-wise copies in `test.cpp`.
On Emerald Rapids (`perf`, the ntt product bottom): `broadcast_halves` runs on p0 or p5 where the
`vpermd` it replaced ran on p5 only; `load_halves` still costs a p5 uop, like `vpermt2d`. The
simd and modarith bench binaries for avx512 are byte-identical to main's.

2026-10-11, claude (small issue #37, round 2): added `store_first` and Neon `mls_wide_lo/hi`
(Design). Tested against lane-wise copies and 64-bit products in `test.cpp` (every k from 0 to the
lanes, guard words around). A partial store through a stack buffer and piece copies cost the small
kernel a store-forwarding stall at 8x8 (about 18 cycles on Emerald Rapids); the masked store does
not (`backend/small/notes.md`, Log). The simd, modarith, ntt and factorials bench binaries are
byte-identical to main's (avx2, avx512).

## Sources

- Instruction costs: measured here. Pipe names and port layouts from AMD's Zen 3 and Zen 5, Intel's
  and Arm's Neoverse V2 optimization guides; uops.info for the idea of mixed-pair probes. No code
  used.
- lc-opt's NTT generator (`lib/ntt/gen_kernels.py`) holds the Zen 3 model this one extends.
- SVE2 `umulh` and the NEON-SVE bridge: Arm's A64 instruction set reference and ACLE
  (`arm_neon_sve_bridge.h`). No code used.
