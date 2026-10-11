#!/usr/bin/env python3
"""Generate avx2_kernels.hpp: the AVX2 loops of the NTT as GCC inline assembly.

Each kernel is a dataflow graph of AVX2 instructions. The graph is list-scheduled on a Zen 3
model under a register budget, allocated onto ymm0-15 and emitted as one asm statement.
Arithmetic, value ranges and the scheduling knobs are documented in notes.md.

Usage: gen_avx2_kernels.py [--check]   (--check: exit 1 if avx2_kernels.hpp differs from the output)
"""
import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

OUT = Path(__file__).resolve().parent / 'avx2_kernels.hpp'

# Zen 3: multiplies on 2 of the 4 vector pipes, shifts on 2, everything else on any.
LATENCY = {'vpmuludq': 3, 'vpmulld': 3, 'vpsrlq': 1, 'vpaddd': 1, 'vpsubd': 1,
           'vpminud': 1, 'vpblendd': 1, 'vpaddq': 1}
PIPE = {'vpmuludq': 'mul', 'vpmulld': 'mul', 'vpsrlq': 'shift'}
COMMUTATIVE = {'vpaddd', 'vpaddq', 'vpminud', 'vpmuludq', 'vpmulld'}
LOADS = {'vmovdqa', 'vmovdqu', 'vbroadcastss'}
# Bases in the caller's memory, any 4-byte alignment: unaligned loads and stores.
UNALIGNED = {'s', 'o', 'o2'}


class ScheduleError(Exception):
    pass


class Value:
    """kind: 'reg' computed; 'mem' vector in memory; 'scalar' 32-bit word in memory, broadcast
    on use; 'creg' constant register; 'cmem' constant in memory (stack twiddles)."""

    def __init__(self, kind, mem=None, reg=None):
        self.kind, self.mem, self.reg = kind, mem, reg

    @property
    def in_memory(self):
        return self.kind in ('mem', 'cmem')


class Op:
    def __init__(self, mnemonic, dst, srcs, imm=None, store=None):
        self.mnemonic, self.dst, self.srcs, self.imm, self.store = mnemonic, dst, srcs, imm, store

    def unit(self):
        if self.store is not None:
            return 'store'
        if self.mnemonic in LOADS:
            return 'load'
        return PIPE.get(self.mnemonic, 'alu')

    def reg_srcs(self):
        return {s for s in self.srcs if s.kind == 'reg'}


class Graph:
    """Builds the ops of one butterfly or bottom stage, in source order. Memory operands fold into
    instructions where the encoding allows it; otherwise each is loaded once per graph."""

    def __init__(self, consts, disp=0, array='a', share_broadcasts=True, src=None, out=None):
        """Quarters are read from src and written to out, both array unless given."""
        self.ops, self.c, self.disp, self.array = [], consts, disp, array
        self.src, self.out = src or array, out or array
        self.share_broadcasts = share_broadcasts
        self.loaded = {}  # memory value (or broadcast address) -> register value

    def quarter(self, i, extra=0):
        """Vector i of the four the butterfly reads, extra bytes past it."""
        return Value('mem', mem=(self.src, i, self.disp + extra))

    def at(self, base, disp, fold=True):
        v = Value('mem', mem=(base, None, disp))
        return v if fold else self.materialize(v)

    def scalar(self, base, disp):
        return Value('scalar', mem=(base, None, disp))

    def materialize(self, v):
        if v.kind == 'scalar':
            if self.share_broadcasts and v.mem in self.loaded:
                return self.loaded[v.mem]
            r = Value('reg')
            self.ops.append(Op('vbroadcastss', r, [v]))
            if self.share_broadcasts:
                self.loaded[v.mem] = r
            return r
        if v.kind != 'mem':
            return v
        if v not in self.loaded:
            r = Value('reg')
            aligned = v.mem[2] % 32 == 0 and v.mem[0] not in UNALIGNED
            self.ops.append(Op('vmovdqa' if aligned else 'vmovdqu', r, [v]))
            self.loaded[v] = r
        return self.loaded[v]

    def op(self, mnemonic, *srcs, imm=None):
        srcs = [self.materialize(s) if s.kind == 'scalar' else self.loaded.get(s, s) for s in srcs]
        if mnemonic in ('vpsubd', 'vpblendd', 'vpsrlq'):  # first source in a register (VEX shifts take none from memory)
            srcs[0] = self.materialize(srcs[0])
            if srcs[0].kind == 'cmem':
                raise ScheduleError('constant minuend in memory')
        elif mnemonic in COMMUTATIVE and srcs[0].in_memory and srcs[1].in_memory:
            data = 0 if srcs[0].kind == 'mem' else 1
            srcs[data] = self.materialize(srcs[data])
        r = Value('reg')
        self.ops.append(Op(mnemonic, r, srcs, imm))
        return r

    def store(self, v, base, index, disp):
        self.ops.append(Op('vmovdqa', None, [v], store=(base, index, disp)))

    def store_quarter(self, v, i):
        self.store(v, self.out, i, self.disp)

    def add(self, x, y):
        return self.op('vpaddd', x, y)

    def low(self, x):  # [0, 4P) -> [0, 2P): min(x, x - 2P)
        return self.op('vpminud', x, self.op('vpsubd', x, self.c['2P']))

    def low_signed(self, t):  # t = x - y for x, y < 2P -> x - y mod 2P in [0, 2P)
        return self.op('vpminud', t, self.op('vpaddd', t, self.c['2P']))

    def diff(self, x, y):  # x - y + 2P
        return self.op('vpsubd', self.op('vpaddd', x, self.c['2P']), y)

    def canonical(self, x):  # [0, 4P) -> [0, P)
        y = self.low(x)
        return self.op('vpminud', y, self.op('vpsubd', y, self.c['P']))

    def shoup(self, x, w, wq, odd=None):
        """x * w mod P in [0, 2P) for any x < 2^32; wq = floor(w 2^32 / P). odd: a value whose
        even lanes hold the odd lanes of x."""
        if odd is None:
            odd = self.op('vpsrlq', x, imm=32)
        even_product = self.op('vpmuludq', x, wq)
        odd_product = self.op('vpmuludq', odd, wq)
        q = self.op('vpblendd', self.op('vpsrlq', even_product, imm=32), odd_product, imm=0xAA)
        return self.op('vpsubd', self.op('vpmulld', x, w), self.op('vpmulld', q, self.c['P']))


# Butterflies. Forward (Cooley-Tukey), inputs and outputs < 4P:
#   a = low(f0), b = low(f1), c = x f2, d = x f3, ac = low(a + c), amc = low(a - c),
#   out = ac + y(b + d), ac - y(b + d) + 2P, amc + z(b - d + 2P), amc - z(b - d + 2P) + 2P.
# Inverse (Gentleman-Sande), inputs and outputs < 2P:
#   ab = low(a + b), cd = low(c + d), amb = y(a - b + 2P), cmd = z(c - d + 2P),
#   out = low(ab + cd), low(amb + cmd), x(ab - cd + 2P), x(amb - cmd + 2P).
# Identity groups have x = y = 1. The odd lanes of f2 and f3 come from loads 4 bytes further on.

def forward_butterfly(g):
    c = g.c
    f0, f1, f2, f3 = (g.quarter(i) for i in range(4))
    a, b = g.low(f0), g.low(f1)
    cc = g.shoup(f2, c['WX'], c['QX'], odd=g.quarter(2, 4))
    dd = g.shoup(f3, c['WX'], c['QX'], odd=g.quarter(3, 4))
    ac = g.low(g.add(a, cc))
    amc = g.low_signed(g.op('vpsubd', a, cc))
    bd, bmd = g.add(b, dd), g.diff(b, dd)
    y = g.shoup(bd, c['WY'], c['QY'])
    z = g.shoup(bmd, c['WZ'], c['QZ'])
    g.store_quarter(g.add(ac, y), 0)
    g.store_quarter(g.diff(ac, y), 1)
    g.store_quarter(g.add(amc, z), 2)
    g.store_quarter(g.diff(amc, z), 3)


def inverse_butterfly(g):
    c = g.c
    f0, f1, f2, f3 = (g.quarter(i) for i in range(4))
    ab, cd = g.low(g.add(f0, f1)), g.low(g.add(f2, f3))
    amb = g.shoup(g.diff(f0, f1), c['WY'], c['QY'])
    cmd = g.shoup(g.diff(f2, f3), c['WZ'], c['QZ'])
    g.store_quarter(g.low(g.add(ab, cd)), 0)
    g.store_quarter(g.low(g.add(amb, cmd)), 1)
    g.store_quarter(g.shoup(g.diff(ab, cd), c['WX'], c['QX']), 2)
    g.store_quarter(g.shoup(g.diff(amb, cmd), c['WX'], c['QX']), 3)


def forward_identity(g):
    c = g.c
    a, b, cc, dd = (g.low(g.quarter(i)) for i in range(4))
    ac = g.low(g.add(a, cc))
    amc = g.low_signed(g.op('vpsubd', a, cc))
    bd, bmd = g.low(g.add(b, dd)), g.diff(b, dd)
    z = g.shoup(bmd, c['WZ'], c['QZ'])
    g.store_quarter(g.add(ac, bd), 0)
    g.store_quarter(g.diff(ac, bd), 1)
    g.store_quarter(g.add(amc, z), 2)
    g.store_quarter(g.diff(amc, z), 3)


def forward_top(g):
    """forward_identity on canonical inputs: no reduction before the outputs, which are < 4P."""
    c = g.c
    a, b, cc, dd = (g.quarter(i) for i in range(4))
    ac, bd = g.add(a, cc), g.add(b, dd)  # < 2P
    amc = g.op('vpsubd', g.op('vpaddd', a, c['P']), cc)  # < 2P
    z = g.shoup(g.op('vpsubd', g.op('vpaddd', b, c['P']), dd), c['WZ'], c['QZ'])
    g.store_quarter(g.add(ac, bd), 0)
    g.store_quarter(g.diff(ac, bd), 1)
    g.store_quarter(g.add(amc, z), 2)
    g.store_quarter(g.diff(amc, z), 3)


def forward_top3(g):
    """forward_top with the last quarter zero, not read: bd = b, z = shoup(b)."""
    c = g.c
    a, b, cc = (g.quarter(i) for i in range(3))
    z = g.shoup(b, c['WZ'], c['QZ'])  # first: it loads b, which the adds below then reuse
    ac = g.add(a, cc)  # < 2P
    amc = g.op('vpsubd', g.op('vpaddd', a, c['P']), cc)  # < 2P
    g.store_quarter(g.add(ac, b), 0)
    g.store_quarter(g.diff(ac, b), 1)
    g.store_quarter(g.add(amc, z), 2)
    g.store_quarter(g.diff(amc, z), 3)


def inverse_identity(g):
    c = g.c
    f0, f1, f2, f3 = (g.quarter(i) for i in range(4))
    ab, cd = g.low(g.add(f0, f1)), g.low(g.add(f2, f3))
    amb = g.low_signed(g.op('vpsubd', f0, f1))
    cmd = g.shoup(g.diff(f2, f3), c['WZ'], c['QZ'])
    g.store_quarter(g.low(g.add(ab, cd)), 0)
    g.store_quarter(g.low(g.add(amb, cmd)), 1)
    g.store_quarter(g.low_signed(g.op('vpsubd', ab, cd)), 2)
    g.store_quarter(g.low_signed(g.op('vpsubd', amb, cmd)), 3)


def scale_radix2(g):
    """f0, f1 < 2P -> canonical s(f0 + f1), s(f0 - f1 + 2P), s in WZ/QZ."""
    c = g.c
    x, y = g.quarter(0), g.quarter(1)
    for v, i in ((g.add(x, y), 0), (g.diff(x, y), 1)):
        r = g.shoup(v, c['WZ'], c['QZ'])
        g.store_quarter(g.op('vpminud', r, g.op('vpsubd', r, c['P'])), i)


# Fused bottom stage, one batch of four vectors (a radix-4 group with h = 1 and its leaves):
#   first: forward butterfly of a and b; leaf buffer rows [canonical(w_t A_t), canonical(A_t)]
#     and canonical(B_t); w_t from the weights pointer (four values, then four quotients).
#   last: leaf products A_t B_t mod (x^8 - w_t) in load-reuse form, Montgomery reduction,
#     inverse butterfly, four stores.
# The leaf buffer holds window[4][16] (bytes 64t) and coeff[4][8] (bytes 256 + 32t).

def forward_h1(g, array, x, y):
    tw = {'WX': g.scalar(x, 0), 'QX': g.scalar(x, 32), 'WY': g.scalar(y, 0), 'QY': g.scalar(y, 32),
          'WZ': g.scalar(y, 4), 'QZ': g.scalar(y, 36)}
    f = [g.at(array, 32 * t) for t in range(4)]
    a, b = g.low(f[0]), g.low(f[1])
    cc = g.shoup(f[2], tw['WX'], tw['QX'], odd=g.at(array, 68))
    dd = g.shoup(f[3], tw['WX'], tw['QX'], odd=g.at(array, 100))
    ac = g.low(g.add(a, cc))
    amc = g.low_signed(g.op('vpsubd', a, cc))
    bd, bmd = g.add(b, dd), g.diff(b, dd)
    y = g.shoup(bd, tw['WY'], tw['QY'])
    z = g.shoup(bmd, tw['WZ'], tw['QZ'])
    return [g.add(ac, y), g.diff(ac, y), g.add(amc, z), g.diff(amc, z)]


def bottom_first(g):
    """Each output is stored right after its butterfly, which keeps register pressure low."""
    for t, v in enumerate(forward_h1(g, 'a', 'x', 'y')):
        xa = g.canonical(v)
        g.store(xa, 'buf', None, 64 * t + 32)
        wa = g.shoup(xa, g.scalar('w', 4 * t), g.scalar('w', 16 + 4 * t))
        g.store(g.op('vpminud', wa, g.op('vpsubd', wa, g.c['P'])), 'buf', None, 64 * t)
    for t, v in enumerate(forward_h1(g, 'b', 'x', 'y')):
        g.store(g.canonical(v), 'buf', None, 256 + 32 * t)


def bottom_last(g):
    """Window vector X_k (k = 9..1) is loaded once and feeds e (with y_{8-k}) and o (y_{9-k})."""
    c = g.c
    leaves = []
    for t in range(4):
        e = o = None
        for k in range(9, 0, -1):
            x = g.at('cur', 64 * t + 4 * k, fold=False)
            if k <= 8:
                p = g.op('vpmuludq', x, g.scalar('cur', 256 + 32 * t + 4 * (8 - k)))
                e = p if e is None else g.op('vpaddq', e, p)
            if k >= 2:
                p = g.op('vpmuludq', x, g.scalar('cur', 256 + 32 * t + 4 * (9 - k)))
                o = p if o is None else g.op('vpaddq', o, p)
        e = g.op('vpaddq', e, g.op('vpmuludq', g.op('vpmuludq', e, c['NI']), c['P']))
        o = g.op('vpaddq', o, g.op('vpmuludq', g.op('vpmuludq', o, c['NI']), c['P']))
        leaves.append(g.low(g.op('vpblendd', g.op('vpsrlq', e, imm=32), o, imm=0xAA)))
    tw = {'WX': g.scalar('ix', 0), 'QX': g.scalar('ix', 32), 'WY': g.scalar('iy', 0), 'QY': g.scalar('iy', 32),
          'WZ': g.scalar('iy', 4), 'QZ': g.scalar('iy', 36)}
    r = leaves
    ab, cd = g.low(g.add(r[0], r[1])), g.low(g.add(r[2], r[3]))
    amb = g.shoup(g.diff(r[0], r[1]), tw['WY'], tw['QY'])
    cmd = g.shoup(g.diff(r[2], r[3]), tw['WZ'], tw['QZ'])
    g.store(g.low(g.add(ab, cd)), 'out', None, 0)
    g.store(g.low(g.add(amb, cmd)), 'out', None, 32)
    g.store(g.shoup(g.diff(ab, cd), tw['WX'], tw['QX']), 'out', None, 64)
    g.store(g.shoup(g.diff(amb, cmd), tw['WX'], tw['QX']), 'out', None, 96)


# Bottom level of a transform alone (avx2.hpp, Transform): two radix-4 groups with h = 1, k and
# k + 1 for an even k, at a. x = r + slot(k) and y = r + slot(2k) hold both groups' twiddles:
# x_m = x[m], y_m = y[2m], z_m = y[2m + 1], quotients 8 words on.

def leaf_twiddles(g, m):
    return {'WX': g.scalar('x', 4 * m), 'QX': g.scalar('x', 32 + 4 * m), 'WY': g.scalar('y', 8 * m),
            'QY': g.scalar('y', 32 + 8 * m), 'WZ': g.scalar('y', 8 * m + 4), 'QZ': g.scalar('y', 36 + 8 * m)}


def canonical_difference(g, x, y):  # x, y < 2P -> x - y mod P
    t = g.low_signed(g.op('vpsubd', x, y))
    return g.op('vpminud', t, g.op('vpsubd', t, g.c['P']))


def forward_leaves(g, groups=2):
    """Forward butterflies, values < 4P in, canonical leaves out. No odd-lane loads: the inputs
    were often just stored, and a load across two stores waits for them."""
    for m in range(groups):
        tw = leaf_twiddles(g, m)
        f = [g.at('a', 128 * m + 32 * t) for t in range(4)]
        a, b = g.low(f[0]), g.low(f[1])
        cc = g.shoup(f[2], tw['WX'], tw['QX'])
        dd = g.shoup(f[3], tw['WX'], tw['QX'])
        ac = g.low(g.add(a, cc))
        amc = g.low_signed(g.op('vpsubd', a, cc))
        y = g.shoup(g.add(b, dd), tw['WY'], tw['QY'])
        z = g.shoup(g.diff(b, dd), tw['WZ'], tw['QZ'])
        out = [g.canonical(g.add(ac, y)), canonical_difference(g, ac, y),
               g.canonical(g.add(amc, z)), canonical_difference(g, amc, z)]
        for t, v in enumerate(out):
            g.store(v, 'a', None, 128 * m + 32 * t)


def inverse_leaves(g, groups=2):
    """Inverse butterflies, canonical in (the leaf form), < 2P out; inverse twiddles."""
    c = g.c
    for m in range(groups):
        tw = leaf_twiddles(g, m)
        f = [g.at('a', 128 * m + 32 * t) for t in range(4)]
        ab, cd = g.add(f[0], f[1]), g.add(f[2], f[3])  # < 2P
        amb = g.shoup(g.op('vpsubd', g.op('vpaddd', f[0], c['P']), f[1]), tw['WY'], tw['QY'])
        cmd = g.shoup(g.op('vpsubd', g.op('vpaddd', f[2], c['P']), f[3]), tw['WZ'], tw['QZ'])
        g.store(g.low(g.add(ab, cd)), 'a', None, 128 * m)
        g.store(g.low(g.add(amb, cmd)), 'a', None, 128 * m + 32)
        g.store(g.shoup(g.diff(ab, cd), tw['WX'], tw['QX']), 'a', None, 128 * m + 64)
        g.store(g.shoup(g.diff(amb, cmd), tw['WX'], tw['QX']), 'a', None, 128 * m + 96)


LEAVES = {'forward_leaves': forward_leaves, 'inverse_leaves': inverse_leaves}


# ------------------------------------------------------------------------------- scheduling

@dataclass
class Knobs:
    seed: int = 0               # 0: no jitter
    margin: int = 0             # registers kept free when growing the live set is optional
    jitter: float = 0.0         # random priority noise, scaled
    load_latency: int = 8       # modeled cycles
    loads_per_cycle: int = 2    # an explicit load takes two of these slots, a folded one one
    window: Optional[int] = None  # issue only among the next `window` ops in source order


def schedule(ops, budget, knobs):
    """Cycle-driven list scheduling: 4 vector pipes (multiplies on 2, shifts on 2), loads, one
    store per cycle. Priority: longest latency path to a sink, plus jitter. An op that would raise
    the number of live registers above `budget` waits. Returns the ops in issue order."""
    rng = random.Random(knobs.seed) if knobs.seed else None
    jitter = {o: rng.random() * knobs.jitter if rng else 0.0 for o in ops}
    producer = {o.dst: o for o in ops if o.dst is not None}
    preds = {o: [producer[s] for s in o.srcs if s.kind == 'reg' and s in producer] for o in ops}
    succs = {o: [] for o in ops}
    for o in ops:
        for p in preds[o]:
            succs[p].append(o)

    def latency(o):
        unit = o.unit()
        return knobs.load_latency if unit == 'load' else 1 if unit == 'store' else LATENCY[o.mnemonic]

    priority = {}
    for o in reversed(ops):  # ops are in a topological order
        priority[o] = latency(o) + max((priority[s] for s in succs[o]), default=0)
    uses = {}
    for o in ops:
        for s in o.reg_srcs():
            uses[s] = uses.get(s, 0) + 1
    index = {o: i for i, o in enumerate(ops)}

    def delta(o):
        dying = sum(1 for s in o.reg_srcs() if uses[s] == 1)
        return dying, int(o.dst is not None)

    finish, order, remaining, live, cycle, stall = {}, [], list(ops), 0, 0, 0
    while remaining:
        free = {'mul': 2, 'shift': 2, 'vector': 4, 'load': knobs.loads_per_cycle, 'store': 1}
        candidates = remaining[:knobs.window] if knobs.window else remaining
        ready = [o for o in candidates if all(p in finish and finish[p] <= cycle for p in preds[o])]

        def key(o):
            dying, new = delta(o)
            grows = new - dying > 0 and live - dying + new > budget - knobs.margin
            return grows, -priority[o] - jitter[o], index[o]
        ready.sort(key=key)
        issued = False
        for o in ready:
            unit, folded = o.unit(), sum(1 for s in o.srcs if s.in_memory)
            if free['load'] < folded:
                continue
            if unit in ('mul', 'shift', 'alu'):
                if free['vector'] == 0 or (unit != 'alu' and free[unit] == 0):
                    continue
            elif free[unit] == 0:
                continue
            dying, new = delta(o)
            if live - dying + new > budget:
                continue
            if unit in ('mul', 'shift', 'alu'):
                free['vector'] -= 1
                if unit != 'alu':
                    free[unit] -= 1
            else:
                free[unit] -= 1
            free['load'] -= folded
            live += new - dying
            for s in o.reg_srcs():
                uses[s] -= 1
            finish[o] = cycle + latency(o)
            remaining.remove(o)
            order.append(o)
            issued = True
        stall = 0 if issued or any(f > cycle for f in finish.values()) else stall + 1
        if stall > 2:
            raise ScheduleError('register budget deadlock')
        cycle += 1
    return order


def prune(ops):
    """The ops some store needs, in their order: drops outputs a kernel does not store."""
    live, kept = set(), []
    for o in reversed(ops):
        if o.store is not None or o.dst in live:
            kept.append(o)
            live.update(s for s in o.srcs if s.kind == 'reg')
    return kept[::-1]


def allocate(seq, reserved):
    """Linear scan: a register frees at its value's last use and goes to the next new value."""
    last = {}
    for i, o in enumerate(seq):
        for s in o.srcs:
            if s.kind == 'reg':
                last[s] = i
    free = [r for r in reversed(range(16)) if r not in reserved]
    reg, live = {}, {}
    for i, o in enumerate(seq):
        for s in o.srcs:
            if s.kind == 'reg' and last.get(s) == i and live.get(reg[s]) is s:
                del live[reg[s]]
                free.append(reg[s])
        if o.dst is None:
            continue
        if o.dst not in last:
            raise ScheduleError('dead value')
        if not free:
            raise ScheduleError('out of registers')
        reg[o.dst] = free.pop()
        live[reg[o.dst]] = o.dst
    return reg


# ---------------------------------------------------------------------------------- emission

QUARTER = ['({b})', '({b},%[h])', '({b},%[h],2)', '({b},%[h3])']


def address(mem):
    base, index, disp = mem
    if index is None:
        return f'{disp}(%[{base}])' if disp else f'(%[{base}])'
    return (str(disp) if disp else '') + QUARTER[index].format(b=f'%[{base}]')


def operand(v, reg):
    if v.kind in ('mem', 'scalar'):
        return address(v.mem)
    if v.kind == 'cmem':
        return address(('w', None, v.mem))
    return f'%%ymm{v.reg if v.kind == "creg" else reg[v]}'


def emit(o, reg):
    if o.store is not None:
        move = 'vmovdqu' if o.store[0] in UNALIGNED else 'vmovdqa'
        return f'{move} {operand(o.srcs[0], reg)}, {address(o.store)}'
    dst = f'%%ymm{reg[o.dst]}'
    if o.mnemonic in LOADS:
        return f'{o.mnemonic} {operand(o.srcs[0], reg)}, {dst}'
    if o.mnemonic == 'vpsrlq':
        return f'vpsrlq ${o.imm}, {operand(o.srcs[0], reg)}, {dst}'
    x, y = o.srcs  # AT&T: dst = x OP y is written "op y, x, dst"; only y may be in memory
    if o.mnemonic in COMMUTATIVE and x.in_memory:
        x, y = y, x
    imm = f'${o.imm}, ' if o.imm is not None else ''
    return f'{o.mnemonic} {imm}{operand(y, reg)}, {operand(x, reg)}, {dst}'


# --------------------------------------------------------------------------------- kernels

@dataclass
class Loop:
    name: str
    graph: object
    twiddles: str          # 'xyz': x, y, z (values on the stack, quotients in ymm11-13); 'z': z only
    pair: bool = False     # one butterfly of a and one of b per iteration (else two of a)
    knobs: Knobs = field(default_factory=Knobs)
    src: bool = False      # reads the quarters from s (any alignment), writes them to a
    doc: str = ''


LOOPS = [
    Loop('forward', forward_butterfly, 'xyz', doc=(
        'Forward radix-4 butterflies j < h of one group on a[j + t h], t < 4; values < 4P.\n'
        'x, y: twiddles r[k], r[2k] in the block table; z = r[2k + 1] follows y. h even.')),
    Loop('forward_pair', forward_butterfly, 'xyz', pair=True,
         knobs=Knobs(seed=129, margin=1, loads_per_cycle=3), doc=(
             'forward() on a and b with the same twiddles, one butterfly of each per step.')),
    Loop('inverse', inverse_butterfly, 'xyz', knobs=Knobs(seed=138, load_latency=16), doc=(
        'Inverse radix-4 butterflies j < h of one group, values < 2P; inverse twiddles. h even.')),
    Loop('forward_identity', forward_identity, 'z', doc=(
        'forward() for group 0 (x = y = 1); z = y[1]: pass the table start. h even.')),
    Loop('forward_top', forward_top, 'z', doc=(
        'forward_identity() on canonical input: the first level of a transform. h even.')),
    Loop('forward_top_from', forward_top, 'z', src=True, doc=(
        'forward_top() from a source: columns j < columns of the quarters, the vectors at\n'
        's + 8 (j + t h) (any alignment; s may be a), into a[j + t h]. columns even.')),
    Loop('forward_top3_from', forward_top3, 'z', src=True, doc=(
        'forward_top_from() with the quarter at s + 24 h zero, not read. columns even.')),
    Loop('inverse_identity', inverse_identity, 'z', doc=(
        'inverse() for group 0; z = y[1]: pass the inverse table start. h even.')),
    Loop('scale_radix2', scale_radix2, 'z', doc=(
        'Last radix-2 layer and scale: a[j], a[j + h] < 2P become canonical s (a[j] + a[j + h])\n'
        'and s (a[j] - a[j + h]); s = y[1], its quotient y[9]. h even.')),
]

BOTTOM_ARGS = {'first': ['a', 'b', 'buf', 'x', 'y', 'w'], 'last': ['out', 'cur', 'ix', 'iy'],
               'both': ['a', 'b', 'buf', 'x', 'y', 'w', 'out', 'cur', 'ix', 'iy']}
# Knobs of all kernels: chosen by timing generated families on EPYC 7763 (QPoly exploration 009,
# via lc-opt lib/ntt).
BOTTOM_KNOBS = {'first': Knobs(window=12),
                'last': Knobs(seed=7, margin=2, jitter=3.0, window=12),
                'both': Knobs(seed=540, jitter=3.0, window=12)}
BOTTOM_SHIFT = -0.05  # 'both': the first stage's ops start this fraction earlier in source order
LEAVES_KNOBS = {'forward_leaves': Knobs(window=20), 'inverse_leaves': Knobs(window=20)}


def loop_body(loop):
    consts = {'P': Value('creg', reg=15), '2P': Value('creg', reg=14)}
    reserved = {14, 15}
    if loop.twiddles == 'z':
        consts.update(WZ=Value('creg', reg=12), QZ=Value('creg', reg=13))
        reserved |= {12, 13}
    else:
        for i, name in enumerate('XYZ'):
            consts['W' + name] = Value('cmem', mem=32 * i)
            consts['Q' + name] = Value('creg', reg=11 + i)
            reserved.add(11 + i)
    ops = []
    for k in range(2):
        g = Graph(consts, disp=0 if loop.pair else 32 * k, array='b' if loop.pair and k else 'a',
                  src='s' if loop.src else None)
        loop.graph(g)
        ops += g.ops
    seq = schedule(ops, 16 - len(reserved), loop.knobs)
    reg = allocate(seq, reserved)
    return [emit(o, reg) for o in seq]


def bottom_body(part):
    g = Graph({'P': Value('creg', reg=15), '2P': Value('creg', reg=14), 'NI': Value('creg', reg=13)},
              share_broadcasts=False)
    if part in ('last', 'both'):
        bottom_last(g)
    split = len(g.ops)
    if part in ('first', 'both'):
        bottom_first(g)
    ops = g.ops
    if part == 'both':  # interleave the two stages' source orders proportionally
        last, first = ops[:split], ops[split:]
        keyed = [(i / len(last), 0, i, o) for i, o in enumerate(last)]
        keyed += [(j / len(first) + BOTTOM_SHIFT, 1, j, o) for j, o in enumerate(first)]
        ops = [o for *_, o in sorted(keyed, key=lambda t: t[:3])]
    seq = schedule(ops, 13, BOTTOM_KNOBS[part])
    reg = allocate(seq, {13, 14, 15})
    return [emit(o, reg) for o in seq]


def asm_lines(lines, indent='        '):
    return [f'{indent}"{line}\\n\\t"' for line in lines]


def loop_asm(loop):
    step = 1 if loop.pair else 2
    head = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14']
    if loop.twiddles == 'z':
        head += ['vbroadcastss 4(%[y]), %%ymm12', 'vbroadcastss 36(%[y]), %%ymm13']
    else:
        for i, (src, disp) in enumerate([('x', 0), ('y', 0), ('y', 4)]):
            head += [f'vbroadcastss {address((src, None, disp))}, %%ymm0', f'vmovdqa %%ymm0, {address(("w", None, 32 * i))}',
                     f'vbroadcastss {address((src, None, disp + 32))}, %%ymm{11 + i}']
    tail = (['add $32, %[b]'] if loop.pair else []) + (['add $64, %[s]'] if loop.src else [])
    tail += [f'add ${32 * step}, %[a]', 'cmp %[end], %[a]', 'jne 1b']
    return head + ['.p2align 5', '1:'] + loop_body(loop) + tail


def bottom_asm(part):
    head = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14', 'vpbroadcastd %[NI], %%ymm13']
    return head + bottom_body(part)


def loop_function(loop):
    body = loop_asm(loop)
    args = "Vec* a, Vec* b, std::size_t h" if loop.pair else "Vec* a, std::size_t h"
    if loop.src:
        args = "const std::uint32_t* s, Vec* a, std::size_t columns, std::size_t h"
    twiddle_args = 'const std::uint32_t* x, const std::uint32_t* y' if loop.twiddles == 'xyz' else 'const std::uint32_t* y'
    outputs = '[a] "+r"(a)' + (', [b] "+r"(b)' if loop.pair else '') + (', [s] "+r"(s)' if loop.src else '')
    inputs = '[end] "r"(end), [h] "r"(32 * h), [h3] "r"(96 * h), [y] "r"(y)'
    if loop.twiddles == 'xyz':
        inputs += ', [x] "r"(x), [w] "r"(w)'
    lines = [f'// {line}' for line in loop.doc.split('\n')]
    lines += [f'inline void {loop.name}({args}, {twiddle_args}) {{']
    if loop.twiddles == 'xyz':
        lines += ['    alignas(32) Vec w[3];  // twiddle values x, y, z; their quotients stay in ymm11-13']
    lines += [f'    Vec* const end = a + {"columns" if loop.src else "h"};', '    asm volatile(']
    lines += asm_lines(body)
    lines += [f'        : {outputs}', f'        : {inputs}, [P] "m"(kP), [P2] "m"(k2P)',
              '        : ' + CLOBBERS + ', "cc");', '}']
    return lines


CLOBBERS = ', '.join(f'"xmm{i}"' for i in range(16)) + ', "memory"'

BOTTOM_DOC = {
    'first': 'Forward half of the bottom stage for the batch at a, b; fills buf.',
    'last': 'Leaf products from cur and the inverse butterfly; writes the batch at out.',
    'both': 'last() for the current batch interleaved with first() for the next.',
}


def bottom_function(part):
    args = BOTTOM_ARGS[part]
    params = ', '.join(f'{"void*" if a in ("buf", "out") else "const void*"} {a}' for a in args)
    lines = [f'// {BOTTOM_DOC[part]}', f'inline void bottom_{part}({params}) {{', '    asm volatile(']
    lines += asm_lines(bottom_asm(part))
    lines += ['        :', '        : ' + ', '.join(f'[{a}] "r"({a})' for a in args) + ',',
              '          [P] "m"(kP), [P2] "m"(k2P), [NI] "m"(kNI)',
              '        : ' + CLOBBERS + ');', '}']
    return lines


def leaves_function(name):
    g = Graph({'P': Value('creg', reg=15), '2P': Value('creg', reg=14)})
    LEAVES[name](g)
    seq = schedule(g.ops, 14, LEAVES_KNOBS[name])
    reg = allocate(seq, {14, 15})
    body = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14'] + [emit(o, reg) for o in seq]
    doc = LEAVES[name].__doc__.split('.')[0]
    lines = [f'// Groups k, k + 1 (k even) with h = 1 at a: {doc[0].lower() + doc[1:]}.', '// x = r + slot(k), y = r + slot(2k).',
             f'inline void {name}(Vec* a, const std::uint32_t* x, const std::uint32_t* y) {{', '    asm volatile(']
    lines += asm_lines(body)
    lines += ['        :', '        : [a] "r"(a), [x] "r"(x), [y] "r"(y), [P] "m"(kP), [P2] "m"(k2P)',
              '        : ' + CLOBBERS + ');', '}']
    return lines


HEADER = '''\
// Generated by backend/ntt/gen_avx2_kernels.py. Do not edit.
// AVX2 loops of the NTT as inline assembly, scheduled for Zen 3. Arithmetic and ranges: notes.md.
// Vectors hold 8 residues mod P = 998244353. A twiddle w < P comes with its Shoup quotient
// floor(w 2^32 / P) 8 words later (block table layout, see common.hpp). The forward kernels read
// 4 bytes past their last vector: the odd lanes of a multiplicand come from a load 4 bytes on.
#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace fps::ntt::avx2_kernels {

using Vec = __m256i;

inline constexpr std::uint32_t kP = 998244353, k2P = 2 * kP, kNI = 998244351;  // kNI = -1 / P mod 2^32
'''


def generate():
    lines = HEADER.split('\n')
    for loop in LOOPS:
        lines += loop_function(loop) + ['']
    lines += ['// Bottom stage: a batch of four vectors is a radix-4 group with h = 1; each output vector',
              '// v_t is a leaf, a polynomial mod x^8 - w_t. buf, cur: leaf buffers (384 bytes, 64-byte',
              '// aligned); w: the four leaf weights, then their quotients.', '']
    for part in ('first', 'last', 'both'):
        lines += bottom_function(part) + ['']
    lines += ['// Bottom level of a transform alone: leaves out (forward), leaves in (inverse).', '']
    for name in LEAVES:
        lines += leaves_function(name) + ['']
    lines += ['}  // namespace fps::ntt::avx2_kernels']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = generate()
    if args.check:
        if OUT.read_text() != text:
            print(f'{OUT} is stale: run {Path(__file__).name}')
            return 1
        return 0
    OUT.write_text(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
