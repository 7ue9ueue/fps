#!/usr/bin/env python3
"""Machine models of the four timing machines, for kernel generators and cost estimates.

Per machine: the clock, the vector pipes, loads and stores per cycle, and per instruction form
(mnemonic and register width) its latency in cycles, reciprocal throughput (cycles per
instruction when nothing else runs) and the pipes it issues to.

- latency and throughput: measured by probe.cpp on the timing machine (notes.md, 2026-10-11).
  Accumulating instructions (mla, umlal, ...) list the latency through the accumulator.
- pipes: the classes (which instructions share pipes) are measured by the probe's mixed pairs:
  two instructions that share pipes mix at the throughput of one, disjoint ones at half. The
  pipe names come from the vendors' guides (guesses where a guide was unclear). Ops whose class
  no mixed pair checked are listed in `guess`.
- loads and stores per cycle: from the guides, not measured (guesses).

Usage: import machines; machines.MACHINES['zen5'].ops[('vpmuludq', 512)].latency
       machines.py prints a summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Op:
    latency: float
    throughput: float  # reciprocal: cycles per instruction at full rate
    pipes: tuple[str, ...]


@dataclass(frozen=True)
class Machine:
    target: str  # tools/kernel.py target timed on this machine
    vm: str
    cpu: str
    clock_ghz: float  # from a dependent add chain, during the probe
    vector_bits: int  # widest native vector datapath
    vector_pipes: tuple[str, ...]
    loads_per_cycle: dict[int, int]  # by width in bits (guesses from the guides)
    stores_per_cycle: dict[int, int]
    ops: dict[tuple[str, int], Op]
    guess: tuple[tuple[str, int], ...] = field(default=())  # ops whose pipes are not checked
    notes: str = ''


def _ops(table: dict[tuple[str, int], tuple[float, float, str]]) -> dict[tuple[str, int], Op]:
    return {key: Op(lat, tp, tuple(pipes.split())) for key, (lat, tp, pipes) in table.items()}


# AMD Zen 3. lc-opt's NTT generator models the same: multiplies on 2 of the 4 pipes, shifts on 2.
ZEN3 = Machine(
    target='avx2', vm='lc-bench', cpu='AMD EPYC 7B13 (Zen 3)', clock_ghz=3.457, vector_bits=256,
    vector_pipes=('FP0', 'FP1', 'FP2', 'FP3'),
    loads_per_cycle={128: 3, 256: 2}, stores_per_cycle={128: 2, 256: 1},
    ops=_ops({
        ('vpaddd', 256): (1, 0.25, 'FP0 FP1 FP2 FP3'),
        ('vpsubd', 256): (1, 0.25, 'FP0 FP1 FP2 FP3'),
        ('vpaddq', 256): (1, 0.25, 'FP0 FP1 FP2 FP3'),
        ('vpminud', 256): (1, 0.25, 'FP0 FP1 FP2 FP3'),
        ('vpblendd', 256): (1, 0.25, 'FP0 FP1 FP2 FP3'),
        ('vpmuludq', 256): (3, 0.5, 'FP0 FP3'),
        ('vpmulld', 256): (3, 0.5, 'FP0 FP3'),
        ('vpsrlq', 256): (1, 0.5, 'FP1 FP2'),
        ('vpsllq', 256): (1, 0.5, 'FP1 FP2'),
        ('vpshufb', 256): (1, 0.5, 'FP1 FP2'),
        ('vpermd', 256): (3, 1.34, 'FP1 FP2'),
    }),
    guess=(('vpshufb', 256), ('vpermd', 256)),
    notes='vpermd ymm: 1.34 cycles per instruction measured (not 1 or 0.5).',
)

# Intel Emerald Rapids (Raptor Cove cores). 512-bit ops issue on p0 (p0 and p1 fused) and p5.
EMERALD_RAPIDS = Machine(
    target='avx512', vm='lc-intel', cpu='Intel Xeon Platinum 8581C (Emerald Rapids)', clock_ghz=4.18,
    vector_bits=512, vector_pipes=('p0', 'p1', 'p5'),
    loads_per_cycle={256: 3, 512: 2}, stores_per_cycle={256: 2, 512: 1},
    ops=_ops({
        ('vpaddd', 256): (1.67, 0.333, 'p0 p1 p5'),
        ('vpsubd', 256): (1.67, 0.333, 'p0 p1 p5'),
        ('vpaddq', 256): (1, 0.333, 'p0 p1 p5'),  # 1.67 in an earlier run
        ('vpblendd', 256): (1.67, 0.333, 'p0 p1 p5'),
        ('vpminud', 256): (2, 0.5, 'p0 p1'),
        ('vpmuludq', 256): (5, 0.5, 'p0 p1'),
        ('vpmulld', 256): (10, 1.0, 'p0 p1'),  # 2 uops
        ('vpsrlq', 256): (1, 0.5, 'p0 p1'),
        ('vpsllq', 256): (1, 0.5, 'p0 p1'),
        ('vpshufb', 256): (1, 0.5, 'p1 p5'),
        ('vpermd', 256): (3, 1.0, 'p5'),
        ('vpmadd52luq', 256): (4, 0.5, 'p0 p1'),
        ('vpmadd52huq', 256): (4, 0.5, 'p0 p1'),
        ('vpaddd', 512): (2.5, 0.5, 'p0 p5'),
        ('vpaddq', 512): (2.5, 0.5, 'p0 p5'),
        ('vpblendmd', 512): (2.5, 0.5, 'p0 p5'),
        ('vpminud', 512): (3, 1.0, 'p0'),
        ('vpmuludq', 512): (5.8, 0.53, 'p0 p5'),
        ('vpmulld', 512): (11.6, 1.06, 'p0 p5'),  # 2 uops
        ('vpsrlq', 512): (1, 1.0, 'p0'),
        ('vpsllq', 512): (1, 1.0, 'p0'),
        ('vpermd', 512): (4, 1.0, 'p5'),
        ('vpmadd52luq', 512): (4.2, 0.53, 'p0 p5'),
        ('vpmadd52huq', 512): (4.2, 0.53, 'p0 p5'),
        ('vpermt2d', 256): (3, 1.0, 'p5'),
        ('vpermt2d', 512): (4, 1.0, 'p5'),
    }),
    guess=(('vpshufb', 256), ('vpermd', 256), ('vpminud', 256), ('vpermd', 512), ('vpminud', 512),
           ('vpmadd52luq', 256), ('vpmadd52huq', 256), ('vpermt2d', 256), ('vpermt2d', 512)),
    notes=('Simple ops show latency 1.67 (ymm) and 2.5 (zmm) where the guides say 1, through a chain on '
           'one register; unexplained (vpaddq ymm: 1.67 in one run, 1 in the next). vpmulld is 2 uops.'),
)

# AMD Zen 5 (Turin): four 512-bit pipes; simple integer ops have latency 2.
ZEN5 = Machine(
    target='zen5', vm='lc-zen5', cpu='AMD EPYC 9B45 (Zen 5)', clock_ghz=4.12, vector_bits=512,
    vector_pipes=('FP0', 'FP1', 'FP2', 'FP3'),
    loads_per_cycle={256: 4, 512: 2}, stores_per_cycle={256: 2, 512: 1},
    ops=_ops({
        **{(m, w): (2, 0.25, 'FP0 FP1 FP2 FP3') for m in ('vpaddd', 'vpaddq', 'vpminud') for w in (256, 512)},
        ('vpsubd', 256): (2, 0.25, 'FP0 FP1 FP2 FP3'),
        ('vpblendd', 256): (2, 0.25, 'FP0 FP1 FP2 FP3'),
        ('vpblendmd', 512): (2, 0.25, 'FP0 FP1 FP2 FP3'),
        **{(m, w): (3, 0.5, 'FP0 FP3') for m in ('vpmuludq', 'vpmulld') for w in (256, 512)},
        **{(m, w): (4, 0.5, 'FP0 FP3') for m in ('vpmadd52luq', 'vpmadd52huq') for w in (256, 512)},
        **{(m, w): (2, 0.5, 'FP1 FP2') for m in ('vpsrlq', 'vpsllq') for w in (256, 512)},
        ('vpshufb', 256): (2, 0.5, 'FP1 FP2'),
        ('vpermd', 256): (4, 0.5, 'FP1 FP2'),
        ('vpermd', 512): (5, 0.5, 'FP1 FP2'),
        ('vpermt2d', 256): (4, 0.5, 'FP1 FP2'),
        ('vpermt2d', 512): (5, 0.5, 'FP1 FP2'),
    }),
    guess=(('vpshufb', 256), ('vpermd', 256), ('vpermd', 512), ('vpermt2d', 256), ('vpermt2d', 512)),
    notes='vpmulld costs the same as vpmuludq here (unlike Intel). IFMA shares the multiply pipes.',
)

# Arm Neoverse V2 (Google Axion): four 128-bit pipes V0-V3; SVE2 vectors are 128-bit too.
NEOVERSE_V2 = Machine(
    target='arm', vm='lc-arm', cpu='Google Axion (Neoverse V2)', clock_ghz=2.99, vector_bits=128,
    vector_pipes=('V0', 'V1', 'V2', 'V3'),
    loads_per_cycle={128: 3}, stores_per_cycle={128: 2},
    ops=_ops({
        **{(m, 128): (2, 0.25, 'V0 V1 V2 V3') for m in ('add', 'sub', 'umin', 'shsub', 'uzp1', 'uzp2', 'bsl', 'tbl')},
        **{(m, 128): (4, 0.5, 'V0 V2') for m in ('mul', 'umull', 'umull2', 'sqdmulh')},
        **{(m, 128): (1.15, 0.5, 'V0 V2') for m in ('mla', 'mls', 'umlal', 'umlal2')},
        ('ushr', 128): (2, 0.5, 'V1 V3'),
        ('shrn', 128): (2, 0.5, 'V1 V3'),
    }),
    guess=(('ushr', 128), ('shrn', 128), ('shsub', 128), ('bsl', 128), ('tbl', 128)),
    notes=('A multiply mixed 1:1 with add or uzp2 runs at 0.333 cycles per instruction, not 0.25: '
           'the ALU ops do not all move to V1/V3. mla/umlal from the multiplicands: not measured '
           '(guess: 4, as mul).'),
)

MACHINES = {m.target: m for m in (ZEN3, EMERALD_RAPIDS, ZEN5, NEOVERSE_V2)}


def main() -> None:
    for m in MACHINES.values():
        print(f'{m.target:7} {m.vm:9} {m.cpu}, {m.clock_ghz} GHz, {m.vector_bits}-bit, {len(m.ops)} ops')


if __name__ == '__main__':
    main()
