#!/usr/bin/env python3
"""Generate neon_kernels.hpp: NTT butterfly kernels for Neoverse V2 (arm target) as GCC inline asm.

Each kernel is a dataflow graph of Neon and SVE2 instructions on 128-bit vectors (4 words). The
graph is put in a source order, list-scheduled on a Neoverse V2 model (below) under a register
budget, reordered so that its multiplies and other vector ops alternate (repattern), allocated
onto v0-v28 by linear scan and emitted as one asm statement. The arithmetic is intrinsics.hpp's
(Bottom<Neon>, Butterflies<Neon>): outputs equal the intrinsics' mod P, in the same ranges.

Kernels (neon.hpp calls them; design and measurements in notes.md, notes/151-r1.md, 151-r2.md):
  bottom_first, bottom_last, bottom_both: the bottom stage of a product, two groups k, k + 1 (k
    even) per statement. first: forward butterflies of a and b (h = 1), the leaf rows [w_t A_t,
    A_t] and B_t into a leaf buffer; last: the 8 x 8 leaf products from the buffer (umull/umlal
    with B_t by element, 64-bit sums < 8 P^2, Montgomery reduction) and the inverse butterflies
    into a; both: last for one pair interleaved with first for the next.
  bottom_rows_first, bottom_square_*, bottom_stored_*: the same for a square (B_t = A_t, from the
    rows) and for a product by stored leaves (B_t from memory; Transform's fused ops).
  bottom_forward_pair, bottom_inverse_pair: the bottom level of a transform, two groups.
  forward_group, inverse_group (and _identity_, group 0; forward_group2: inputs < 2P from other
    memory): one radix-4 group on quarters of q words, an asm loop of 4 columns per iteration.
  forward_top_group, inverse_top_group (and _identity_, _from, _to), inverse_scaled_group (and
    _identity_), inverse_radix2_level (and _scaled, _to): the top levels and the scaled top groups
    of the subtrees, loops as the group loops with their twiddles in a block (values, then
    quotients 8 words on).

Usage: gen_neon_kernels.py [--check] [--out FILE] [--knobs JSON]
  --check: exit 1 if neon_kernels.hpp differs from the output. --out, --knobs (exploration): write
  elsewhere, with knobs over the defaults: {"<kernel>": {"<knob>": value}, "merge": {"shift": s}}.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

OUT = Path(__file__).resolve().parent / 'neon_kernels.hpp'

# ------------------------------------------------------------------------------ machine model
# Neoverse V2 (lc-arm), backend/simd/probe.cpp and machines.py: four 128-bit pipes V0-V3. Classes:
#   M  multiplies (mul, mls, umull, umlal, SVE2 umulh; by element too), V0 and V2, latency 4; an
#      accumulating op reads its accumulator 1 cycle after a producing accumulating op;
#   A  add, sub, umin, uzp, dup: any pipe, latency 2;
#   S  stores: a data uop on V0 or V1 (str q + mul mix at the rate of two multiplies), 2 a cycle;
#   L  loads: 3 a cycle, 2 slots for a load across a 32-byte boundary, latency 6 (a guess).
PIPES = {'M': (0, 2), 'A': (3, 1, 2, 0), 'S': (1, 0)}
LATENCY = {'M': 4, 'A': 2, 'L': 6, 'S': 1}
ACC_LATENCY = 1
LOADS_PER_CYCLE, STORES_PER_CYCLE = 3, 2

# Registers: v0-v28 for values, v29 = P, v30 = 2P, v31 = -P^-1 mod 2^32 (broadcast).
VALUE_REGS = 29
CONST = {'P': 29, '2P': 30, 'NI': 31}


class ScheduleError(Exception):
    pass


class Value:
    __slots__ = ('reg',)

    def __init__(self, reg=None):
        self.reg = reg  # a fixed register (constants, twiddles), else allocated


class Op:
    __slots__ = ('cls', 'fmt', 'dst', 'srcs', 'acc', 'slots')

    def __init__(self, cls, fmt, dst, srcs, acc=False, slots=1):
        # fmt: the instruction with {d} (destination) and {s0}, {s1}, ... (sources); acc: srcs[0]
        # is an accumulator tied to dst; slots: load slots.
        self.cls, self.fmt, self.dst, self.srcs, self.acc, self.slots = cls, fmt, dst, srcs, acc, slots

    def reg_srcs(self):
        return [s for s in self.srcs if s.reg is None]


class Graph:
    def __init__(self):
        self.ops: list[Op] = []
        self.c = {name: Value(reg) for name, reg in CONST.items()}

    def emit(self, cls, fmt, srcs, acc=False, slots=1, dst=True):
        d = Value() if dst else None
        self.ops.append(Op(cls, fmt, d, list(srcs), acc, slots))
        return d

    # Memory. disp in bytes from the base operand %[base].
    def load(self, base, disp, form='q'):
        crosses = form == 'q' and disp % 32 > 16
        mnemonic = 'ldr' if disp % (16 if form == 'q' else 8) == 0 else 'ldur'
        return self.emit('L', f'{mnemonic} {form}{{d}}, [%[{base}], #{disp}]', [], slots=2 if crosses else 1)

    def store(self, v, base, disp):
        mnemonic = 'str' if disp % 16 == 0 else 'stur'
        self.emit('S', f'{mnemonic} q{{s0}}, [%[{base}], #{disp}]', [v], dst=False)

    # Lane-wise ops.
    def add(self, x, y):
        return self.emit('A', 'add v{d}.4s, v{s0}.4s, v{s1}.4s', [x, y])

    def sub(self, x, y):
        return self.emit('A', 'sub v{d}.4s, v{s0}.4s, v{s1}.4s', [x, y])

    def umin(self, x, y):
        return self.emit('A', 'umin v{d}.4s, v{s0}.4s, v{s1}.4s', [x, y])

    def uzp(self, half, x, y):
        return self.emit('A', f'uzp{half} v{{d}}.4s, v{{s0}}.4s, v{{s1}}.4s', [x, y])

    def dup(self, x, lane):
        return self.emit('A', f'dup v{{d}}.4s, v{{s0}}.s[{lane}]', [x])

    def mul(self, x, w, lane=None):
        e = '4s' if lane is None else f's[{lane}]'
        return self.emit('M', f'mul v{{d}}.4s, v{{s0}}.4s, v{{s1}}.{e}', [x, w])

    def umulh(self, x, q):  # SVE2, unpredicated; 128-bit vectors on Neoverse V2
        return self.emit('M', 'umulh z{d}.s, z{s0}.s, z{s1}.s', [x, q])

    def mls(self, acc, x, y):  # acc - x y
        return self.emit('M', 'mls v{d}.4s, v{s1}.4s, v{s2}.4s', [acc, x, y], acc=True)

    def umull(self, x, b, lane, high):  # 64-bit products of lanes 0, 1 (2, 3 if high) of x and b[lane]
        e = f's[{lane}]' if lane is not None else ('4s' if high else '2s')
        return self.emit('M', f'umull{"2" if high else ""} v{{d}}.2d, v{{s0}}.{"4s" if high else "2s"}, v{{s1}}.{e}',
                         [x, b])

    def umlal(self, acc, x, b, lane, high):
        e = f's[{lane}]' if lane is not None else ('4s' if high else '2s')
        return self.emit('M', f'umlal{"2" if high else ""} v{{d}}.2d, v{{s1}}.{"4s" if high else "2s"}, v{{s2}}.{e}',
                         [acc, x, b], acc=True)

    # Lazy arithmetic mod P (intrinsics.hpp, Arith).
    def low(self, x):  # [0, 4P) -> [0, 2P)
        return self.umin(x, self.sub(x, self.c['2P']))

    def canonical(self, x):  # [0, 2P) -> [0, P)
        return self.umin(x, self.sub(x, self.c['P']))

    def diff(self, x, y):  # x - y + 2P for y <= 2P
        return self.sub(self.add(x, self.c['2P']), y)

    def low_diff(self, x, y):  # x - y mod 2P in [0, 2P) for x, y < 2P
        t = self.sub(x, y)
        return self.umin(t, self.add(t, self.c['2P']))

    def shoup(self, x, w, q):
        """x w mod P in [0, 2P) for any x < 2^32. w = (register, lane): the twiddle by element;
        q: its Shoup quotient floor(w 2^32 / P), broadcast."""
        t = self.umulh(x, q)
        return self.mls(self.mul(x, *w), t, self.c['P'])


# ------------------------------------------------------------------------------- butterflies
# Forward (intrinsics.hpp, forward4): inputs < 4P (kin = 4) or < 2P (kin = 2: no input
# reductions), outputs < 4P.
#   a = low(f0), b = low(f1), c = x f2, d = x f3, ac = low(a + c), amc = low_diff(a, c),
#   out = ac + y(b + d), ac - y(b + d) + 2P, amc + z(b - d + 2P), amc - z(b - d + 2P) + 2P.
# Inverse (inverse4): inputs < 2P, outputs < 2P.
#   ab = low(a + b), cd = low(c + d), amb = y(a - b + 2P), cmd = z(c - d + 2P),
#   out = low(ab + cd), low(amb + cmd), x(ab - cd + 2P), x(amb - cmd + 2P).
# Identity groups (k = 0) have x = y = 1. tw: {'x': ((vector, lane), quotient), 'y': ..., 'z': ...}.

def forward4(g, f, tw, kin=4):
    a, b = (g.low(f[0]), g.low(f[1])) if kin == 4 else (f[0], f[1])
    c, d = g.shoup(f[2], *tw['x']), g.shoup(f[3], *tw['x'])
    ac, amc = g.low(g.add(a, c)), g.low_diff(a, c)
    y, z = g.shoup(g.add(b, d), *tw['y']), g.shoup(g.diff(b, d), *tw['z'])
    return [g.add(ac, y), g.diff(ac, y), g.add(amc, z), g.diff(amc, z)]


def inverse4(g, f, tw):
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb, cmd = g.shoup(g.diff(f[0], f[1]), *tw['y']), g.shoup(g.diff(f[2], f[3]), *tw['z'])
    return [g.low(g.add(ab, cd)), g.low(g.add(amb, cmd)), g.shoup(g.diff(ab, cd), *tw['x']),
            g.shoup(g.diff(amb, cmd), *tw['x'])]


def forward_identity4(g, f, tw, kin=4):
    a, b, c, d = (g.low(v) for v in f) if kin == 4 else f
    ac, amc, bd = g.low(g.add(a, c)), g.low_diff(a, c), g.low(g.add(b, d))
    z = g.shoup(g.diff(b, d), *tw['z'])
    return [g.add(ac, bd), g.diff(ac, bd), g.add(amc, z), g.diff(amc, z)]


def inverse_identity4(g, f, tw):
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb, cmd = g.low_diff(f[0], f[1]), g.shoup(g.diff(f[2], f[3]), *tw['z'])
    return [g.low(g.add(ab, cd)), g.low(g.add(amb, cmd)), g.low_diff(ab, cd), g.low_diff(amb, cmd)]


# Top and scaled groups (intrinsics.hpp: forward_identity_first<1>, inverse_group_scaled,
# inverse_radix4). Twiddle names: s (the scale), sx, sy, sz (s times x, y, z), x, z.
def forward_identity1(g, f, tw):
    """forward_identity4 on canonical inputs (a transform's top group): no reductions; out < 4P."""
    a, b, c, d = f
    ac, amc, bd = g.add(a, c), g.sub(g.add(a, g.c['P']), c), g.add(b, d)
    z = g.shoup(g.diff(b, d), *tw['z'])
    return [g.add(ac, bd), g.diff(ac, bd), g.add(amc, z), g.diff(amc, z)]


def inverse_scaled4(g, f, tw):
    """inverse4 with every output times s (group k > 0 under the top level); in and out < 2P."""
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb, cmd = g.shoup(g.diff(f[0], f[1]), *tw['sy']), g.shoup(g.diff(f[2], f[3]), *tw['sz'])
    return [g.shoup(g.add(ab, cd), *tw['s']), g.low(g.add(amb, cmd)), g.shoup(g.diff(ab, cd), *tw['sx']),
            g.shoup(g.diff(amb, cmd), *tw['x'])]


def inverse_scaled_identity4(g, f, tw):
    """inverse_scaled4 for k = 0 (sx = sy = s)."""
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb, cmd = g.shoup(g.diff(f[0], f[1]), *tw['s']), g.shoup(g.diff(f[2], f[3]), *tw['sz'])
    return [g.shoup(g.add(ab, cd), *tw['s']), g.low(g.add(amb, cmd)), g.shoup(g.diff(ab, cd), *tw['s']),
            g.low_diff(amb, cmd)]


def inverse_top4(g, f, tw):
    """The inverse top group times s (inverse_radix4, s != 0): in < 2P, canonical out."""
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb, cmd = g.shoup(g.diff(f[0], f[1]), *tw['s']), g.shoup(g.diff(f[2], f[3]), *tw['sz'])
    return [g.canonical(g.shoup(g.add(ab, cd), *tw['s'])), g.canonical(g.low(g.add(amb, cmd))),
            g.canonical(g.shoup(g.diff(ab, cd), *tw['s'])), g.canonical(g.low(g.diff(amb, cmd)))]


def inverse_top_identity4(g, f, tw):
    """The inverse top group without a scale (s = 0: folded into the groups below): canonical out."""
    return [g.canonical(v) for v in inverse_identity4(g, f, tw)]


# The inverse radix-2 level on halves x, y (intrinsics.hpp, inverse_radix2). The forward one (an add
# and a subtract per pair) ran as fast as the intrinsics on L1 data: not generated.
def inverse_radix2(g, f, tw):
    """In < 2P, canonical out; times s if tw has it."""
    x, y = f
    if 's' in tw:
        return [g.canonical(g.shoup(g.add(x, y), *tw['s'])), g.canonical(g.shoup(g.diff(x, y), *tw['s']))]
    return [g.canonical(g.low(g.add(x, y))), g.canonical(g.low(g.diff(x, y)))]


class Twiddles:
    """Twiddles of groups k + m (k even): x_m at x[m] (quotients 8 words on), y_m, z_m at yz[2m],
    yz[2m + 1]. Values are used by element; quotients broadcast (dup), as SVE2 umulh has no
    by-element form. Each group loads its own at its first use: loads are cheap, long-lived
    registers are not."""

    def __init__(self, x, yz):
        self.x, self.yz, self.cache = x, yz, {}

    def group(self, g, m):
        if m not in self.cache:
            xv, xq = g.load(self.x, 0, 'd'), g.load(self.x, 32, 'd')
            yzv, yzq = g.load(self.yz, 0, 'q'), g.load(self.yz, 32, 'q')
            self.cache[m] = {'x': ((xv, m), g.dup(xq, m)), 'y': ((yzv, 2 * m), g.dup(yzq, 2 * m)),
                             'z': ((yzv, 2 * m + 1), g.dup(yzq, 2 * m + 1))}
        return self.cache[m]


# --------------------------------------------------------------------------------- the bottom
# Group k at a: quarter t (words 8t .. 8t + 7) is leaf 4k + t after the forward butterflies, a
# polynomial mod x^8 - w_t with w_t = y, -y, z, -z (y = r(2k), z = r(2k + 1)). Columns u = 0, 1:
# words 4u .. 4u + 3 of each quarter. Leaf product of A and B mod x^8 - w: row [w A, A] (16 words,
# canonical; for w = -y: P - canonical(y A), in (0, P]), x^i A mod x^8 - w = its words 8 - i ..
# 15 - i, and word l of A B 2^-32 is the Montgomery reduction of sum_i b_i row[8 - i + l] < 8 P^2,
# which is < 2.9P; low() brings it below 2P for the inverse butterflies.

LEAVES = 384  # leaf buffer bytes per group: rows[4][16] words ([w_t A_t, A_t]), then B[4][8]


def forward_half(g, m, tw, kind):
    """Group m (at a + 128 m bytes, and b): forward butterflies, inputs < 4P; the canonical leaf
    rows of a, and for a product the leaves of b, into the leaf buffer of group m (buf + LEAVES m).
    kind: 'product' (a and b), 'square' or 'stored' (a alone), 'fill' (a holds canonical leaves:
    the rows alone, no butterflies)."""
    at = LEAVES * m
    for name in ('a', 'b') if kind == 'product' else ('a',):
        for u in range(2):
            f = [g.load(name, 128 * m + 32 * t + 16 * u) for t in range(4)]
            for t, v in enumerate(f if kind == 'fill' else forward4(g, f, tw)):
                leaf = v if kind == 'fill' else g.canonical(g.low(v))
                if name == 'b':
                    g.store(leaf, 'buf', at + 256 + 32 * t + 16 * u)
                    continue
                g.store(leaf, 'buf', at + 64 * t + 32 + 16 * u)
                wa = g.canonical(g.shoup(leaf, *(tw['y'] if t < 2 else tw['z'])))
                g.store(g.sub(g.c['P'], wa) if t % 2 else wa, 'buf', at + 64 * t + 16 * u)


def leaf_product(g, window, B, u):
    """Column u (words 4u .. 4u + 3) of the leaf product, < 2P. window(o): words o .. o + 3 of the
    row; B: two vectors (b_0..3, b_4..7), used by element."""
    lo = hi = None
    for i in range(8):
        x, b, lane = window(8 - i + 4 * u), B[i // 4], i % 4
        if i == 0:
            lo, hi = g.umull(x, b, lane, False), g.umull(x, b, lane, True)
        else:
            lo, hi = g.umlal(lo, x, b, lane, False), g.umlal(hi, x, b, lane, True)
    m = g.mul(g.uzp(1, lo, hi), g.c['NI'])
    lo, hi = g.umlal(lo, m, g.c['P'], None, False), g.umlal(hi, m, g.c['P'], None, True)
    return g.low(g.uzp(2, lo, hi))


def product_half(g, m, iw, kind):
    """Group m's leaf products from its leaf buffer at cur<m> (an operand: the windows' offsets must
    fit ldur's 9 bits), then the inverse butterflies into out + 128 m bytes, out < 2P. B_t: the
    buffer's (product), A_t from the row (square), or b's leaves at bs + 128 m bytes (stored).
    Column by column: each column's windows and B are loaded for it (loads are cheap, registers
    are not)."""
    cur = f'cur{m}'
    for u in range(2):
        c = []
        for t in range(4):
            base, at = {'product': (cur, 256 + 32 * t), 'square': (cur, 64 * t + 32),
                        'stored': ('bs', 128 * m + 32 * t)}[kind]
            B = [g.load(base, at + 16 * h) for h in range(2)]
            c.append(leaf_product(g, lambda o, t=t: g.load(cur, 64 * t + 4 * o), B, u))
        for t, v in enumerate(inverse4(g, c, iw.group(g, m))):
            g.store(v, 'out', 128 * m + 32 * t + 16 * u)


def interleave(first, second, shift=0.0):
    """Two op lists merged proportionally to their lengths, second starting `shift` earlier."""
    keyed = [(i / len(first), 0, i, o) for i, o in enumerate(first)]
    keyed += [(j / len(second) + shift, 1, j, o) for j, o in enumerate(second)]
    return [o for *_, o in sorted(keyed, key=lambda t: t[:3])]


def bottom(g, part, kind='product', groups=2, shift=0.0):
    """The bottom stage of `groups` groups k + m (k even) per statement, in two halves: first
    (forward_half of the batch at a, b into buf) and last (product_half of the batch in cur0,
    cur1 into out); both: last for the current batch interleaved with first for the next one.
    kind 'fill': rows from leaves, then the stored form's last half. Returns the source order."""
    ops = {}
    for half in ('last', 'first'):
        if part not in (half, 'both'):
            continue
        start = len(g.ops)
        if half == 'first':
            fw = Twiddles('x', 'yz')
            for m in range(groups):
                forward_half(g, m, fw.group(g, m), kind)
        else:
            iw = Twiddles('ix', 'iyz')
            for m in range(groups):
                product_half(g, m, iw, 'stored' if kind == 'fill' else kind)
        ops[half] = g.ops[start:]
    seq = interleave(ops['last'], ops['first'], shift) if part == 'both' else g.ops
    return prune(seq) if kind == 'fill' else seq  # fill: x's twiddle unused


def transform_bottom(g, direction, groups=2):
    """The bottom level of a transform (Bottom::forward, Bottom::inverse) for groups k + m, m <
    groups (k even), 32 words each at f + 128 m bytes, in place: forward in < 4P, canonical leaves
    out; inverse in and out < 2P."""
    tw = Twiddles('x', 'yz')
    for m in range(groups):
        t = tw.group(g, m)
        for u in range(2):
            f = [g.load('f', 128 * m + 32 * q + 16 * u) for q in range(4)]
            out = forward4(g, f, t) if direction == 'forward' else inverse4(g, f, t)
            for q, v in enumerate(out):
                g.store(g.canonical(g.low(v)) if direction == 'forward' else v, 'f', 128 * m + 32 * q + 16 * u)
    return g.ops


# ------------------------------------------------------------------------------ group loops
# A radix-4 level's group k on quarters of q words (intrinsics.hpp, forward_group, inverse_group):
# one asm loop over the columns, `columns` per iteration, the quarter pointers f0..f3 (and s0..s3
# for a source) advanced at its end. Twiddles stay in v24-v28 (x in v28 lane 0, y and z in v27 lanes
# 0 and 1, their quotients broadcast in v26, v25, v24); the body has v0-v23.
LOOP_REGS = 24
LOOP_TWIDDLES = {'x': ((Value(28), 0), Value(26)), 'y': ((Value(27), 0), Value(25)), 'z': ((Value(27), 1), Value(24))}
BUTTERFLY = {'forward': forward4, 'forward_identity': forward_identity4, 'inverse': inverse4,
             'inverse_identity': inverse_identity4,
             'forward2': lambda g, f, tw: forward4(g, f, tw, 2),
             'forward_identity2': lambda g, f, tw: forward_identity4(g, f, tw, 2)}


BUTTERFLY.update({'forward_identity1': forward_identity1, 'inverse_scaled': inverse_scaled4,
                  'inverse_scaled_identity': inverse_scaled_identity4, 'inverse_top': inverse_top4,
                  'inverse_top_identity': inverse_top_identity4, 'inverse_radix2': inverse_radix2,
                  'inverse_radix2_scaled': inverse_radix2})
# Twiddles of the loops with a twiddle block (TwLoop): values in v28 (and v27 past 4), quotients
# broadcast in the registers below; the body has the rest.
TW_NAMES = {'forward_identity1': ['z'], 'inverse_scaled': ['s', 'sx', 'sy', 'sz', 'x'],
            'inverse_scaled_identity': ['s', 'sz'], 'inverse_top': ['s', 'sz'], 'inverse_top_identity': ['z'],
            'inverse_radix2': [], 'inverse_radix2_scaled': ['s']}
WAYS = {'inverse_radix2': 2, 'inverse_radix2_scaled': 2}  # else 4 quarters


def tw_layout(names):
    """(twiddle map, the number of registers v0.. left for the body)."""
    vectors = -(-len(names) // 4)
    tw = {n: ((Value(28 - i // 4), i % 4), Value(28 - vectors - i)) for i, n in enumerate(names)}
    return tw, 28 - vectors - len(names) + 1


def prune(ops):
    """ops without the ones whose results nothing uses (stores are used)."""
    while True:
        used = {s for o in ops for s in o.srcs}
        kept = [o for o in ops if o.dst is None or o.dst in used]
        if len(kept) == len(ops):
            return ops
        ops = kept


def group_loop(g, kind, columns, source=False, tw=None, store=4, upper=False):
    """One iteration: columns c < columns at bytes 16 c of the quarters (read from s0..s3 if source,
    else from f0..f3). Outputs t < store to f0.. (upper: the upper half of the outputs)."""
    for c in range(columns):
        f = [g.load(f's{t}' if source else f'f{t}', 16 * c) for t in range(WAYS.get(kind, 4))]
        out = BUTTERFLY[kind](g, f, LOOP_TWIDDLES if tw is None else tw)
        for t, v in enumerate(out[len(out) // 2:] if upper else out):
            if t < store:
                g.store(v, f'f{t}', 16 * c)
    g.ops[:] = prune(g.ops)
    return g.ops


# ------------------------------------------------------------------------------- scheduling

@dataclass
class Knobs:
    window: Optional[int] = None  # issue only among the next `window` ops in source order (0: no scheduling)
    margin: int = 0             # below budget - margin, ops that grow the live set go last
    age: float = 0.0            # priority lost per op of source order (large: oldest ready op first)
    latency_scale: float = 1.0  # below 1: consumers closer to producers (the out-of-order core waits)
    seed: int = 0               # 0: no jitter
    jitter: float = 0.0         # random priority noise, in cycles of priority
    pattern: str = ''           # repattern() the schedule: vector op classes in this cycle ('MA')
    pattern_window: int = 16
    pattern_stores: str = 'S'   # 'A': stores take the place of an A in the pattern


class Liveness:
    """The values of ops (a topological order) live while some of them are taken out of order:
    produced and not yet read by every consumer. feasible(i): after taking op i, the ops not yet
    taken can still follow in the given order with at most `budget` values live, so a scheduler
    that only takes feasible ops cannot deadlock."""

    def __init__(self, ops, budget):
        vid = {o.dst: i for i, o in enumerate(o for o in ops if o.dst is not None)}
        self.n, self.budget = len(ops), budget
        self.srcs = [[vid[v] for v in o.reg_srcs()] for o in ops]
        self.dst = [vid[o.dst] if o.dst is not None else -1 for o in ops]
        producer = {self.dst[i]: i for i in range(self.n) if self.dst[i] >= 0}
        self.preds = [[producer[v] for v in s] for s in self.srcs]
        self.uses = [0] * len(vid)
        for s in self.srcs:
            for v in s:
                self.uses[v] += 1
        self.taken, self.live, self.head = [False] * self.n, 0, 0

    def growth(self, i, uses=None):
        u, s = uses or self.uses, self.srcs[i]
        return (self.dst[i] >= 0) - sum(1 for v in set(s) if u[v] == s.count(v))

    def feasible(self, i):
        gr = self.growth(i)
        if gr <= 0:
            return True
        if self.live + gr > self.budget:
            return False
        u = self.uses[:]
        cur = self.live + gr
        for v in self.srcs[i]:
            u[v] -= 1
        for j in range(self.head, self.n):
            if self.taken[j] or j == i:
                continue
            cur += self.growth(j, u)
            if cur > self.budget:
                return False
            for v in self.srcs[j]:
                u[v] -= 1
        return True

    def take(self, i):
        self.live += self.growth(i)
        for v in self.srcs[i]:
            self.uses[v] -= 1
        self.taken[i] = True
        while self.head < self.n and self.taken[self.head]:
            self.head += 1


def schedule(ops, budget, knobs):
    """Cycle-driven list scheduling on the model above, ops in a topological source order whose
    live set never exceeds `budget`. Priority: the longest latency path to a sink (plus jitter,
    minus age). Returns (order, modeled cycles)."""
    n, lv = len(ops), Liveness(ops, budget)
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in lv.preds[i]:
            succs[p].append(i)

    def edge(p, i):  # cycles from p's issue to i's
        if ops[i].acc and ops[p].acc and lv.srcs[i][0] == lv.dst[p]:
            return ACC_LATENCY * knobs.latency_scale
        return LATENCY[ops[p].cls] * knobs.latency_scale

    rng = random.Random(knobs.seed) if knobs.seed else None
    priority = [0.0] * n
    for i in reversed(range(n)):
        priority[i] = max((edge(i, s) + priority[s] for s in succs[i]), default=LATENCY[ops[i].cls])
        if rng:
            priority[i] += rng.random() * knobs.jitter
    priority = [priority[i] - knobs.age * i for i in range(n)]
    issued_at, order, cycle = [None] * n, [], 0
    while len(order) < n:
        pipes, loads, stores = [True] * 4, LOADS_PER_CYCLE, STORES_PER_CYCLE
        end = n if knobs.window is None else min(n, lv.head + knobs.window)
        ready = [i for i in range(lv.head, end) if not lv.taken[i]
                 and all(issued_at[p] is not None and issued_at[p] + edge(p, i) <= cycle for p in lv.preds[i])]
        ready.sort(key=lambda i: (lv.growth(i) > 0 and lv.live + lv.growth(i) > budget - knobs.margin, -priority[i], i))
        for i in ready:
            c = ops[i].cls
            if (c == 'L' and loads < ops[i].slots) or (c == 'S' and stores == 0):
                continue
            pipe = None if c == 'L' else next((p for p in PIPES[c] if pipes[p]), None)
            if (c != 'L' and pipe is None) or not lv.feasible(i):
                continue
            if c == 'L':
                loads -= ops[i].slots
            else:
                pipes[pipe] = False
                stores -= c == 'S'
            lv.take(i)
            issued_at[i] = cycle
            order.append(ops[i])
        cycle += 1
        if cycle > 100 * n:
            raise ScheduleError('no progress')
    return order, cycle


def repattern(seq, budget, pattern, window, stores='S'):
    """seq reordered so that its vector ops follow `pattern` (e.g. 'MA': multiply, other,
    multiply, ...) where the dependences allow: the next op of the wanted class among the first
    `window` ops of seq not yet taken whose producers are taken; else the first such op. Loads and
    stores go when they are the first takeable op (stores count as class `stores` in the pattern if
    it is 'A'). The live set stays within budget (Liveness). On Neoverse V2 the pattern gained 2-5%
    over the scheduled order (notes/151-r1.md): the core seems to pick pipes at dispatch."""
    n, lv = len(seq), Liveness(seq, budget)
    cls = [('A' if o.cls == 'S' and stores == 'A' else o.cls) for o in seq]
    out, slot = [], 0
    while lv.head < n:
        want = pattern[slot % len(pattern)]
        open_ = [i for i in range(lv.head, min(n, lv.head + window))
                 if not lv.taken[i] and all(lv.taken[p] for p in lv.preds[i])]
        pick = next((i for i in open_ if ((cls[i] in ('L', 'S') and i == open_[0]) or cls[i] == want) and lv.feasible(i)),
                    None)
        if pick is None:
            pick = next((i for i in open_ if lv.feasible(i)), None)
        if pick is None:
            raise ScheduleError('repattern: nothing takeable')
        slot += cls[pick] in ('M', 'A')
        lv.take(pick)
        out.append(seq[pick])
    return out


def order_ops(order, budget, knobs):
    """The kernel's program order from its source order: (ops, modeled cycles)."""
    seq, cycles = (order, 0) if knobs.window == 0 else schedule(order, budget, knobs)
    if knobs.pattern:
        seq = repattern(seq, budget, knobs.pattern, knobs.pattern_window, knobs.pattern_stores)
    return seq, cycles


def allocate(seq, nregs=VALUE_REGS):
    """Linear scan: a register frees at its value's last use; an accumulating op's destination
    takes its accumulator's register (the accumulator dies there)."""
    last = {}
    for i, o in enumerate(seq):
        for s in o.reg_srcs():
            last[s] = i
    free = list(reversed(range(nregs)))
    reg = {}
    for i, o in enumerate(seq):
        if o.acc:
            acc = o.srcs[0]
            if last[acc] != i or o.srcs[1:].count(acc):
                raise ScheduleError('accumulator used after its update')
        for s in dict.fromkeys(o.reg_srcs()):  # in order: the output must not depend on hashes
            if last[s] == i and not (o.acc and s is o.srcs[0]):
                free.append(reg[s])
        if o.dst is None:
            continue
        if o.dst not in last:
            raise ScheduleError('dead value')
        if o.acc:
            reg[o.dst] = reg[o.srcs[0]]
        else:
            if not free:
                raise ScheduleError('out of registers')
            reg[o.dst] = free.pop()
    return reg


def emit(o, reg):
    r = lambda v: v.reg if v.reg is not None else reg[v]
    fields = {f's{i}': r(s) for i, s in enumerate(o.srcs)}
    if o.dst is not None:
        fields['d'] = r(o.dst)
    return o.fmt.format(**fields)


# ---------------------------------------------------------------------------------- kernels

CLOBBERS = ', '.join(f'"v{i}"' for i in range(32)) + ', "memory"'


def asm_lines(lines, indent='        '):
    return [f'{indent}"{line}\\n\\t"' for line in lines]


def counts(ops):
    c = {k: sum(1 for o in ops if o.cls == k) for k in 'MASL'}
    return f'{c["M"]} multiplies, {c["A"]} other vector ops, {c["L"]} loads, {c["S"]} stores'


@dataclass
class Kernel:
    """One asm statement on pointer arguments (out, buf, f written; the others read)."""
    name: str
    build: object  # build(graph) -> source order
    args: list
    doc: str
    knobs: Knobs = field(default_factory=Knobs)


@dataclass
class Loop:
    """An asm loop over a group's columns (group_loop)."""
    name: str
    kind: str
    columns: int
    doc: str
    knobs: Knobs = field(default_factory=Knobs)
    source: bool = False  # quarters read from `from` (s0..s3), written to f


@dataclass
class TwLoop:
    """An asm loop over columns whose twiddles come in a block tw: values tw[0..8), their Shoup
    quotients tw[8..16), in the order of TW_NAMES[kind] (top groups, scaled groups)."""
    name: str
    kind: str
    doc: str
    source: bool = False  # quarters read from `from` (s0..s3), written to f
    store: int = 4        # outputs t < store written, to quarters t of f
    upper: bool = False   # the upper half of the outputs instead (2 + t, or 1 + t for radix 2)
    columns: int = 4
    knobs: Knobs = field(default_factory=lambda: Knobs(window=None, pattern='MA', pattern_window=32,
                                                       pattern_stores='A'))


FIRST_DOC = ('Forward half of the bottom stage for groups k, k + 1 (k even) of a{b}, inputs < 4P: buf gets\n'
             'their leaf rows{rows}. x = r + slot(k), yz = r + slot(2k).')
LAST_DOC = ('Leaf products from the leaf buffers at cur0, cur1 (groups k, k + 1){of} and the inverse\n'
            'butterflies into out, < 2P with the factor 2^-32. ix = ir + slot(k), iyz = ir + slot(2k).')
BOTH_DOC = '{last}() for groups k, k + 1 interleaved with {first}() for the next two groups.'
TRANSFORM_BOTTOM_DOC = {
    'forward': 'The bottom level of a forward transform for groups k, k + 1 (k even) at f, 32 words each, in\n'
               'place: in < 4P, canonical leaves out. x = r + slot(k), yz = r + slot(2k).',
    'inverse': 'The bottom level of an inverse transform for groups k, k + 1 (k even) at f, in place, in and\n'
               'out < 2P. x = ir + slot(k), yz = ir + slot(2k).',
}
# Timed on lc-arm, kernels on L1 data, 62 variants of the product bottom and 16 of the group loops
# (notes/151-r1.md): the M/A pattern gained 2-5% everywhere; window, margin, age, latency_scale
# and the merge's shift moved the time within about 4%; 4 columns per loop iteration beat 2 and 8.
PATTERN = dict(pattern='MA', pattern_window=32, pattern_stores='A')
BOTTOM_KNOBS = {'first': Knobs(window=200, age=0.05, **PATTERN), 'last': Knobs(age=0.2, **PATTERN),
                'both': Knobs(window=160, **PATTERN)}
BOTTOM_MERGE = {'shift': 0.0}  # interleave() of 'both'
LOOP_KNOBS = dict(window=None, **PATTERN)


def bottom_kernels():
    """Products (bottom_first, _last, _both), squares (bottom_rows_first, bottom_square_*),
    products by stored leaves (bottom_rows_first, bottom_stored_*), transform bottoms."""
    first = {'product': ['a', 'b', 'buf', 'x', 'yz'], 'rows': ['a', 'buf', 'x', 'yz']}
    last = {'product': ['out', 'cur0', 'cur1', 'ix', 'iyz'], 'square': ['out', 'cur0', 'cur1', 'ix', 'iyz'],
            'stored': ['out', 'cur0', 'cur1', 'bs', 'ix', 'iyz']}
    names = {'product': 'bottom_{}', 'square': 'bottom_square_{}', 'stored': 'bottom_stored_{}'}
    of = {'product': '', 'square': ', B = A', 'stored': ' and the leaves of b at bs'}
    out = []
    for kind in ('product', 'rows', 'square', 'stored'):
        parts = ('first',) if kind == 'rows' else ('last', 'both') if kind != 'product' else ('first', 'last', 'both')
        for part in parts:
            fk = 'product' if kind == 'product' else 'rows'
            name = 'bottom_rows_first' if kind == 'rows' else names[kind].format(part)
            build_kind = 'square' if kind == 'rows' else kind
            args = first[fk] if part == 'first' else last[build_kind] if part == 'last' else first[fk] + last[build_kind]
            doc = {'first': FIRST_DOC.format(b=' and b' if fk == 'product' else '',
                                             rows=' and B leaves' if fk == 'product' else ''),
                   'last': LAST_DOC.format(of=of[build_kind]),
                   'both': BOTH_DOC.format(last=names[build_kind].format('last'),
                                           first='bottom_first' if fk == 'product' else 'bottom_rows_first')}[part]
            out.append(Kernel(name, (lambda g, part=part, k=build_kind: bottom(g, part, k, **BOTTOM_MERGE)), args, doc,
                              copy.deepcopy(BOTTOM_KNOBS[part])))
    out += [Kernel(f'bottom_{d}_pair', (lambda g, d=d: transform_bottom(g, d)), ['f', 'x', 'yz'],
                   TRANSFORM_BOTTOM_DOC[d], copy.deepcopy(BOTTOM_KNOBS['first'])) for d in ('forward', 'inverse')]
    # Transform::inverse_product of one term a b: rows from a's leaves (no butterflies), the stored
    # form's products and inverse butterflies.
    out += [Kernel('bottom_fill_first', (lambda g: bottom(g, 'first', 'fill', **BOTTOM_MERGE)), first['rows'],
                   'bottom_rows_first() from the canonical leaves of groups k, k + 1 at a (no butterflies). x unused.',
                   copy.deepcopy(BOTTOM_KNOBS['first'])),
            Kernel('bottom_fill_both', (lambda g: bottom(g, 'both', 'fill', **BOTTOM_MERGE)), first['rows'] + last['stored'],
                   BOTH_DOC.format(last='bottom_stored_last', first='bottom_fill_first'), copy.deepcopy(BOTTOM_KNOBS['both']))]
    return out


KERNELS = bottom_kernels()
LOOPS = [
    Loop('forward_group', 'forward', 4, 'Forward radix-4 butterflies of group k > 0 on the quarters f + tq, t < 4, of q words:\n'
         'columns [0, words / 4), in place, in < 4P, out < 4P. x = r + slot(k), yz = r + slot(2k). words a\n'
         'positive multiple of 16.', Knobs(**LOOP_KNOBS)),
    Loop('forward_identity_group', 'forward_identity', 4, 'forward_group() for k = 0 (x = y = 1); yz = r + slot(0).',
         Knobs(**LOOP_KNOBS)),
    Loop('inverse_group', 'inverse', 4, 'Inverse radix-4 butterflies of group k > 0, in < 2P, out < 2P; inverse twiddles.',
         Knobs(**LOOP_KNOBS)),
    Loop('inverse_identity_group', 'inverse_identity', 4, 'inverse_group() for k = 0; yz = ir + slot(0).',
         Knobs(**LOOP_KNOBS)),
    Loop('forward_group2', 'forward2', 4, 'forward_group() on inputs < 2P (no input reductions), read from the quarters of\n'
         '`from` (may be f).', Knobs(**LOOP_KNOBS), True),
    Loop('forward_identity_group2', 'forward_identity2', 4, 'forward_identity_group() on inputs < 2P, read from `from`.',
         Knobs(**LOOP_KNOBS), True),
]
TW_LOOPS = [
    TwLoop('forward_top_group', 'forward_identity1', 'The top group of a forward transform (group 0, x = y = 1) on canonical inputs, in place:\n'
           'columns [0, words / 4) of the quarters f + tq, out < 4P. tw: z. words a positive multiple of 16.'),
    TwLoop('forward_top_group_from', 'forward_identity1', 'forward_top_group() from the quarters of `from` into f.', True),
    TwLoop('inverse_scaled_group', 'inverse_scaled', 'Inverse group k > 0 with every output times s, in place, in and out < 2P. tw: s, s x, s y,\n'
           's z, x (inverse twiddles of group k).'),
    TwLoop('inverse_scaled_identity_group', 'inverse_scaled_identity', 'inverse_scaled_group() for k = 0. tw: s, s z.'),
    TwLoop('inverse_top_group', 'inverse_top', 'The top group of an inverse transform times s, in place: in < 2P, canonical out. tw: s, s z.'),
    TwLoop('inverse_top_identity_group', 'inverse_top_identity', 'inverse_top_group() without the scale. tw: z.'),
    TwLoop('inverse_top_group_to', 'inverse_top', 'inverse_top_group() from the quarters of `from` into f.', True),
    TwLoop('inverse_top_identity_group_to', 'inverse_top_identity', 'inverse_top_identity_group() from `from` into f.', True),
    TwLoop('inverse_radix2_level', 'inverse_radix2', 'The radix-2 level of an inverse transform on the halves f, f + q, in place: in < 2P,\n'
           'canonical out; columns [0, words / 4). words a positive multiple of 16.'),
    TwLoop('inverse_radix2_level_scaled', 'inverse_radix2_scaled', 'inverse_radix2_level() times s. tw: s.'),
    TwLoop('inverse_radix2_level_to', 'inverse_radix2', 'inverse_radix2_level() from the halves of `from` into f.', True),
    TwLoop('inverse_radix2_level_scaled_to', 'inverse_radix2_scaled', 'inverse_radix2_level_scaled() from `from` into f.', True),
]
# Into part of a product's output (a cut product, an upper half): the outputs t < store, or 2 + t
# (radix-4) and 1 + t (radix-2) if upper, to quarters (halves) t of f; the rest is not computed.
for scaled in (True, False):
    for upper in (False, True):
        top = 'inverse_top_group' if scaled else 'inverse_top_identity_group'
        r2 = 'inverse_radix2_level_scaled' if scaled else 'inverse_radix2_level'
        TW_LOOPS.append(TwLoop(f'{top}_{"upper" if upper else "lower"}_to', 'inverse_top' if scaled else 'inverse_top_identity',
                               f'{top}_to(), outputs {"2 and 3" if upper else "0 and 1"} only (to quarters 0 and 1 of f).',
                               True, 2, upper))
        TW_LOOPS.append(TwLoop(f'{r2}_{"differences" if upper else "sums"}_to', 'inverse_radix2_scaled' if scaled else 'inverse_radix2',
                               f'{r2}_to(), the {"differences" if upper else "sums"} only (to f).', True, 1, upper))


def function(k):
    g = Graph()
    seq, cycles = order_ops(k.build(g), VALUE_REGS, k.knobs)
    reg = allocate(seq)
    body = ['dup v29.4s, %w[p]', 'dup v30.4s, %w[p2]', 'dup v31.4s, %w[ni]'] + [emit(o, reg) for o in seq]
    params = ', '.join(f'{"std::uint32_t*" if a in ("out", "buf", "f") else "const std::uint32_t*"} {a}' for a in k.args)
    lines = [f'// {line}' for line in k.doc.split('\n')]
    lines += [f'// {counts(seq)}; {cycles} cycles modeled.']
    lines += [f'[[gnu::always_inline]] inline void {k.name}({params}) {{', '    asm volatile(']
    lines += asm_lines(body)
    lines += ['        :', '        : ' + ', '.join(f'[{a}] "r"({a})' for a in k.args) + ',',
              '          [p] "r"(kP), [p2] "r"(2 * kP), [ni] "r"(kNegInverse)',
              '        : ' + CLOBBERS + ');', '}']
    return lines


def loop_function(lp):
    g = Graph()
    seq, cycles = order_ops(group_loop(g, lp.kind, lp.columns, lp.source), LOOP_REGS, lp.knobs)
    reg = allocate(seq, LOOP_REGS)
    identity = 'identity' in lp.kind
    head = ['dup v29.4s, %w[p]', 'dup v30.4s, %w[p2]', 'ldr d27, [%[yz]]', 'ldr d25, [%[yz], #32]',
            'dup v24.4s, v25.s[1]', 'dup v25.4s, v25.s[0]']
    if not identity:
        head += ['ldr s28, [%[x]]', 'ldr s26, [%[x], #32]', 'dup v26.4s, v26.s[0]']
    head += ['.p2align 5', '1:']
    pointers = [f'f{t}' for t in range(4)] + ([f's{t}' for t in range(4)] if lp.source else [])
    tail = [f'add %[{x}], %[{x}], #{16 * lp.columns}' for x in pointers] + ['cmp %[f0], %[end]', 'b.ne 1b']
    args = ('const std::uint32_t* from, ' if lp.source else '') + 'std::uint32_t* f, std::size_t q, std::size_t words, '
    args += 'const std::uint32_t* yz' if identity else 'const std::uint32_t* x, const std::uint32_t* yz'
    lines = [f'// {line}' for line in lp.doc.split('\n')]
    lines += [f'// Per iteration ({lp.columns} columns): {counts(g.ops)}; {cycles} cycles modeled.']
    lines += [f'[[gnu::always_inline]] inline void {lp.name}({args}) {{',
              '    std::uint32_t *f0 = f, *f1 = f + q, *f2 = f + 2 * q, *f3 = f + 3 * q;']
    if lp.source:
        lines += ['    const std::uint32_t *s0 = from, *s1 = from + q, *s2 = from + 2 * q, *s3 = from + 3 * q;']
    lines += ['    const std::uint32_t* const end = f + words;', '    asm volatile(']
    lines += asm_lines(head + [emit(o, reg) for o in seq] + tail)
    # Early clobber: end may equal f1's start, and the loop writes f1 before its last read of end.
    lines += ['        : ' + ', '.join(f'[{x}] "+&r"({x})' for x in pointers),
              '        : [end] "r"(end), ' + ('' if identity else '[x] "r"(x), ') + '[yz] "r"(yz), [p] "r"(kP), [p2] "r"(2 * kP)',
              '        : ' + CLOBBERS + ', "cc");', '}']
    return lines


def twloop_function(lp):
    names = TW_NAMES[lp.kind]
    tw, regs = tw_layout(names)
    g = Graph()
    seq, cycles = order_ops(group_loop(g, lp.kind, lp.columns, lp.source, tw, lp.store, lp.upper), regs, lp.knobs)
    reg = allocate(seq, regs)
    vectors = -(-len(names) // 4)
    # Twiddles from general registers (tw[i], tw[8 + i] read by the compiler): a block just built on
    # the stack and read by vector loads waited for its scalar stores (2^6 inverse 1.12x slower).
    head = ['dup v29.4s, %w[p]', 'dup v30.4s, %w[p2]']
    head += [f'mov v{28 - i // 4}.s[{i % 4}], %w[w{i}]' for i in range(len(names))]
    head += [f'dup v{28 - vectors - i}.4s, %w[q{i}]' for i in range(len(names))]
    head += ['.p2align 5', '1:']
    ways = WAYS.get(lp.kind, 4)
    dests = [f'f{t}' for t in range(min(lp.store, ways) if lp.source else ways)]
    pointers = dests + ([f's{t}' for t in range(ways)] if lp.source else [])
    tail = [f'add %[{x}], %[{x}], #{16 * lp.columns}' for x in pointers] + ['cmp %[f0], %[end]', 'b.ne 1b']
    args = ('const std::uint32_t* from, ' if lp.source else '') + 'std::uint32_t* f, std::size_t q, std::size_t words'
    args += ', const std::uint32_t* tw' if names else ''
    at = lambda base, t: base + (f' + {t} * q' if t > 1 else ' + q' if t else '')
    lines = [f'// {line}' for line in lp.doc.split('\n')]
    lines += [f'// Per iteration ({lp.columns} columns): {counts(g.ops)}; {cycles} cycles modeled.']
    lines += [f'[[gnu::always_inline]] inline void {lp.name}({args}) {{',
              '    std::uint32_t ' + ', '.join(f'*f{t} = {at("f", t)}' for t in range(len(dests))) + ';']
    if lp.source:
        lines += ['    const std::uint32_t ' + ', '.join(f'*s{t} = {at("from", t)}' for t in range(ways)) + ';']
    lines += ['    const std::uint32_t* const end = f + words;', '    asm volatile(']
    lines += asm_lines(head + [emit(o, reg) for o in seq] + tail)
    lines += ['        : ' + ', '.join(f'[{x}] "+&r"({x})' for x in pointers),
              '        : [end] "r"(end), ' + ''.join(f'[w{i}] "r"(tw[{i}]), [q{i}] "r"(tw[{8 + i}]), ' for i in range(len(names)))
              + '[p] "r"(kP), [p2] "r"(2 * kP)',
              '        : ' + CLOBBERS + ', "cc");', '}']
    return lines


HEADER = '''\
// Generated by backend/ntt/gen_neon_kernels.py. Do not edit.
// NTT kernels for Neoverse V2 as inline assembly (Neon and SVE2 umulh on 128-bit vectors),
// scheduled by the generator. Arithmetic and ranges: intrinsics.hpp and notes.md.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE2)
#define FPS_NTT_NEON_KERNELS 1

namespace fps::ntt::neon_kernels {

inline constexpr std::uint32_t kP = 998244353, kNegInverse = 998244351;  // -P^-1 mod 2^32
'''


def generate():
    lines = HEADER.split('\n')
    for k in KERNELS:
        lines += function(k) + ['']
    for lp in LOOPS:
        lines += loop_function(lp) + ['']
    for lp in TW_LOOPS:
        lines += twloop_function(lp) + ['']
    lines += ['}  // namespace fps::ntt::neon_kernels', '', '#endif']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--knobs', default='{}')
    args = parser.parse_args()
    overrides = json.loads(args.knobs)
    BOTTOM_MERGE.update(overrides.pop('merge', {}))
    for k in KERNELS + LOOPS:
        for name, value in overrides.get(k.name, {}).items():
            setattr(k.knobs, name, value)
    text = generate()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f'{OUT.name} is stale: run {Path(__file__).name}')
            return 1
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
