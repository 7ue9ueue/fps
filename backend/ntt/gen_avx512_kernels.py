#!/usr/bin/env python3
"""Generate avx512_kernels.hpp and zen5_kernels.hpp: the NTT's butterfly kernels on zmm as GCC
inline assembly, scheduled for Emerald Rapids (avx512 target) and Zen 5 (zen5 target; OUTPUTS).

The AVX-512 counterpart of gen_avx2_kernels.py: each kernel is a dataflow graph of AVX-512
instructions, list-scheduled on a machine model (backend/simd/machines.py: pipes, latencies and
uops per instruction, measured on the timing machine), allocated onto zmm0-31 by linear scan and
emitted as one asm statement. EVEX forms: any source may be a memory operand or an embedded
broadcast, shifts by an immediate read memory, vpermt2d and vpermi2d overwrite a source ("tied"),
broadcasts merge under a mask.

Arithmetic and lazy ranges are the intrinsics variant's (intrinsics.hpp, notes.md): Shoup
products by twiddles, forward values < 4P, inverse < 2P. A Shoup product x w - floor(x q / 2^32) P
takes the odd words' product from a load 4 bytes on where x is loaded from memory (no shift), and
joins the halves of the quotient with vpermt2d (p5 on Emerald Rapids, where shifts and vpminud take
p0).

Kernels (the headers have their contracts):
  forward, inverse, forward_identity, inverse_identity (and forms for inputs < 2P, < P, and with
    prefetch): radix-4 butterflies of one group on columns of 16 words, 2 to 8 columns per
    iteration (intrinsics.hpp's forward_group and inverse_group); forward_*from*: the same from
    other memory into the array; inverse_scaled*: inverse groups with every output times s
    (inverse_group_scaled).
  forward_groups4: groups of two columns (quarters of 32 words), four per statement.
  bottom_forward, bottom_inverse (1, 2 or 4 pairs of groups per statement), bottom_multiply,
    bottom_square: the bottom level of two groups (64 words) per pair (intrinsics.hpp's
    WideBottom), multiply and square with the leaf products and the inverse butterflies; the
    product bottoms in two stages (*_first, *_last, *_both).

Usage: gen_avx512_kernels.py [--check]   (--check: exit 1 if a header differs from the output)
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'simd'))
import machines  # noqa: E402

OUT = HERE / 'avx512_kernels.hpp'


# ------------------------------------------------------------------------------------ model

@dataclass(frozen=True)
class Cost:
    latency: int
    uops: int
    pipes: tuple[str, ...]


def model(machine: machines.Machine) -> dict[str, Cost]:
    """Per mnemonic: latency (cycles, rounded up), uops and the pipes they issue to, from the
    machine's measured 512-bit forms. Where the machine has no entry: vpsubd costs as vpaddd,
    vpermi2d as vpermt2d, a broadcast merged under a mask as vpblendmd (guess)."""
    ops = {m: machine.ops[(m, 512)] for m in
           ('vpaddd', 'vpaddq', 'vpminud', 'vpmuludq', 'vpmulld', 'vpsrlq', 'vpermt2d', 'vpblendmd')}
    ops['vpsubd'] = machine.ops.get(('vpsubd', 512), ops['vpaddd'])
    ops['vpermi2d'] = machine.ops.get(('vpermi2d', 512), ops['vpermt2d'])
    ops['vpsllq'] = ops['vpsrlq']
    ops['vpbroadcastd_mask'] = machine.ops.get(('vpbroadcastd{k}', 512), ops['vpblendmd'])
    ops['vpermd'] = machine.ops[('vpermd', 512)]
    cost = {}
    for m, o in ops.items():
        uops = max(1, round(o.throughput * len(o.pipes)))
        cost[m] = Cost(-int(-o.latency // 1), uops, o.pipes)
    return cost


def split_load_slots(machine: machines.Machine) -> int:
    """Load slots of a zmm load across a cache line (4 bytes past a 64-byte boundary): measured
    on Zen 5 (2); 1 where not measured."""
    split, whole = machine.ops.get(('vmovdqu64 load split', 512)), machine.ops.get(('vmovdqu64 load', 512))
    return round(split.throughput / whole.throughput) if split and whole else 1


# The machine the scheduler models (use()): its costs, the load slots of a load across a line,
# whether an op's uops go to pipes by a matching over the cycle's ops (Zen 5: classes that share
# pipes) or each to the first free pipe of its list, and whether it takes only ops that keep the
# source order feasible (Liveness: no register deadlock). Emerald Rapids keeps the scheduler #150
# tuned its kernels with: first free pipe, a load op counted as two load slots, no feasibility.
MACHINE = machines.MACHINES['avx512']
COST = model(MACHINE)
SPLIT_LOADS, MATCHING, FEASIBLE, LOAD_OP_EXTRA = 1, False, False, 1


def use(target):
    global MACHINE, COST, SPLIT_LOADS, MATCHING, FEASIBLE, LOAD_OP_EXTRA
    MACHINE = machines.MACHINES[target]
    COST = model(MACHINE)
    SPLIT_LOADS = split_load_slots(MACHINE)
    emr = target == 'avx512'
    MATCHING, FEASIBLE, LOAD_OP_EXTRA = not emr, not emr, int(emr)
LOADS = {'vmovdqu64', 'vpbroadcastd'}
COMMUTATIVE = {'vpaddd', 'vpaddq', 'vpminud', 'vpmuludq', 'vpmulld'}
TIED = {'vpermt2d', 'vpermi2d', 'vpbroadcastd_mask'}  # the result overwrites srcs[0]


class ScheduleError(Exception):
    pass


# ------------------------------------------------------------------------------------ graph

class Value:
    """kind: 'reg' computed; 'fixed' a register the statement does not allocate (constants);
    'mem' a vector in memory; 'bcst' a 32-bit word in memory broadcast to every lane on use."""

    def __init__(self, kind, mem=None, reg=None):
        self.kind, self.mem, self.reg = kind, mem, reg

    @property
    def in_memory(self):
        return self.kind in ('mem', 'bcst')


class Op:
    def __init__(self, mnemonic, dst, srcs, imm=None, store=None, mask=None):
        self.mnemonic, self.dst, self.srcs, self.imm, self.store, self.mask = mnemonic, dst, srcs, imm, store, mask

    @property
    def tied(self):
        return self.mnemonic in TIED

    def unit(self):
        if self.store is not None:
            return 'store'
        if self.mnemonic in LOADS:
            return 'load'
        return 'vector'

    def reg_srcs(self):
        return [s for s in self.srcs if s.kind == 'reg']

    def loads(self):
        """Load slots: a vector 4 bytes past a 64-byte boundary (the odd words' loads) takes
        SPLIT_LOADS (operands are 64-byte aligned plus their displacement). A load op counts
        LOAD_OP_EXTRA more (Emerald Rapids: 1, as #150 tuned its kernels)."""
        def slots(v):
            return SPLIT_LOADS if v.kind == 'mem' and v.mem[3] % 64 else 1
        return sum(slots(s) for s in self.srcs if s.in_memory) + (LOAD_OP_EXTRA if self.unit() == 'load' else 0)


def mem(base, disp=0, index=None, scale=1):
    return (base, index, scale, disp)


class Graph:
    """The ops of one kernel body in source order. Memory operands are symbolic until finish():
    one used once where its op can take it folds into the op, the others are loaded once."""

    def __init__(self, consts, join='perm'):
        self.ops, self.c, self.join, self.twiddles, self.ostore = [], consts, join, 'bcst', 'perm'
        self.memo = {}

    def at(self, base, disp=0, index=None, scale=1):
        key = ('mem', mem(base, disp, index, scale))
        if key not in self.memo:
            self.memo[key] = Value('mem', mem=key[1])
        return self.memo[key]

    def bcst(self, base, disp):
        key = ('bcst', mem(base, disp))
        if key not in self.memo:
            self.memo[key] = Value('bcst', mem=key[1])
        return self.memo[key]

    def op(self, mnemonic, *srcs, imm=None):
        r = Value('reg')
        self.ops.append(Op(mnemonic, r, list(srcs), imm))
        return r

    def load(self, v):
        """A fresh register holding memory value v (each call loads again)."""
        r = Value('reg')
        self.ops.append(Op('vpbroadcastd' if v.kind == 'bcst' else 'vmovdqu64', r, [v]))
        return r

    def store(self, v, address):
        self.ops.append(Op('vmovdqu64', None, [v], store=address))

    # Arithmetic mod P (intrinsics.hpp's Arith).
    def add(self, x, y):
        return self.op('vpaddd', x, y)

    def sub(self, x, y):
        return self.op('vpsubd', x, y)

    def low(self, x):  # [0, 4P) -> [0, 2P)
        return self.op('vpminud', x, self.sub(x, self.c['2P']))

    def low_diff(self, x, y):  # x - y mod 2P for x, y < 2P, in [0, 2P)
        t = self.sub(x, y)
        return self.op('vpminud', t, self.add(t, self.c['2P']))

    def diff(self, x, y):  # x - y + 2P, y <= 2P
        return self.sub(self.add(x, self.c['2P']), y)

    def canonical(self, x):  # [0, 2P) -> [0, P)
        return self.op('vpminud', x, self.sub(x, self.c['P']))

    def shoup(self, x, f, odd=None):
        """x w mod P in [0, 2P) for any x < 2^32; f = (w, q), q = floor(w 2^32 / P), values whose
        words 2i and 2i + 1 are equal. odd: a value whose even words are x's odd ones."""
        w, q = f
        if odd is None:
            odd = self.op('vpsrlq', x, imm=32)
        qx = self.join_high(self.op('vpmuludq', x, q), self.op('vpmuludq', odd, q))
        return self.sub(self.op('vpmulld', x, w), self.op('vpmulld', qx, self.c['P']))

    def join_high(self, e, o):
        """Words 2i + 1 of e and o (high halves of 64-bit lanes) into words 2i and 2i + 1: one
        vpermt2d (p5 on Emerald Rapids), or a shift (p0) and a blend under the mask odd."""
        if self.join == 'perm':
            return self.op('vpermt2d', e, self.c['J'], o)
        return self.op('vpblendmd', self.op('vpsrlq', e, imm=32), o)

    def halves(self, base, d0, d1):
        """Words d0 and d1 bytes past base, broadcast to the low and the high 8 lanes."""
        return self.op('vpbroadcastd_mask', self.load(self.bcst(base, d0)), self.bcst(base, d1))

    def finish(self):
        """Folds each memory value used once, as the source its op can take from memory; loads
        the others once, before their first use."""
        uses = {}
        for o in self.ops:
            for s in o.srcs:
                if s.in_memory:
                    uses[id(s)] = uses.get(id(s), 0) + 1
        loaded, out = {}, []
        for o in self.ops:
            if o.store is not None or o.mnemonic in LOADS:
                out.append(o)
                continue
            fold = None
            for i, s in enumerate(o.srcs):
                if s.in_memory and uses[id(s)] == 1 and id(s) not in loaded and foldable(o, i) and fold is None:
                    fold = i
            for i, s in enumerate(o.srcs):
                if s.in_memory and i != fold:
                    if id(s) not in loaded:
                        r = Value('reg')
                        out.append(Op('vpbroadcastd' if s.kind == 'bcst' else 'vmovdqu64', r, [s]))
                        loaded[id(s)] = r
                    o.srcs[i] = loaded[id(s)]
            if fold == 0 and o.mnemonic in COMMUTATIVE:
                o.srcs[0], o.srcs[1] = o.srcs[1], o.srcs[0]
            out.append(o)
        self.ops = out


def foldable(o, i):
    """Whether source i of o may be a memory operand: Intel's last source (either, for a
    commutative op); a broadcast only where the element is 32 bits."""
    m, s = o.mnemonic, o.srcs[i]
    if m in ('vpsrlq', 'vpsllq'):
        return s.kind == 'mem'
    if s.kind == 'bcst' and m in ('vpmuludq', 'vpaddq'):
        return False  # {1to8}: 64-bit elements
    if m in ('vpermt2d', 'vpermi2d', 'vpbroadcastd_mask'):
        return i == len(o.srcs) - 1
    if m == 'vpermd':
        return i == 0
    return m in COMMUTATIVE or i == len(o.srcs) - 1


# ------------------------------------------------------------------------------- scheduling

@dataclass
class Knobs:
    seed: int = 0               # 0: no jitter
    jitter: float = 0.0         # random priority noise
    margin: int = 0             # registers kept free when growing the live set is optional
    load_latency: int = 7       # modeled cycles
    loads_per_cycle: int = 2
    window: Optional[int] = None  # issue only among the next `window` ops in source order
    model: Optional[str] = None   # schedule on this target's model and scheduler (default: the output's)
    latency_scale: float = 1.0    # vector ops' modeled latencies times this
    width: Optional[int] = None   # at most this many vector uops per modeled cycle


def latency(o, knobs):
    if o.unit() == 'load':
        return knobs.load_latency
    if o.unit() == 'store':
        return 1
    lat = COST[o.mnemonic].latency
    if knobs.latency_scale != 1.0:
        lat = max(1, round(lat * knobs.latency_scale))
    return lat + (knobs.load_latency if any(s.in_memory for s in o.srcs) else 0)


def memory_accesses(o):
    """(reads, writes) of o: (base, index, scale, first byte, end byte)."""
    def span(m, size):
        return m[:3] + (m[3], m[3] + size)
    reads = [span(s.mem, 4 if s.kind == 'bcst' else 64) for s in o.srcs if s.in_memory]
    writes = [span(o.store, 64)] if o.store is not None else []
    if o.mask:  # half a vector: bytes [0, 32) or [32, 64) of the store
        b = writes[0][3] + (32 if o.mask == 'hi' else 0)
        writes = [writes[0][:3] + (b, b + 32)]
    return reads, writes


def overlap(xs, ys):
    """Accesses with the same base and index that share a byte. Other indices are quarters of a
    group, at least 128 bytes apart: a load 4 bytes on reads one word of the next quarter, whose
    lane (odd) no op uses."""
    return any(x[:3] == y[:3] and x[3] < y[4] and y[3] < x[4] for x in xs for y in ys)


class Liveness:
    """Values live while ops (in source order) are taken out of order. feasible(o): after taking
    o, the ops not yet taken can still follow in source order with at most `budget` values live, so
    a scheduler that takes only feasible ops cannot deadlock (gen_neon_kernels.py's rule, #151)."""

    def __init__(self, ops, budget):
        self.ops, self.budget, self.index = ops, budget, {o: i for i, o in enumerate(ops)}
        self.uses = {}
        for o in ops:
            for s in o.reg_srcs():
                self.uses[s] = self.uses.get(s, 0) + 1
        self.taken, self.live, self.head = [False] * len(ops), 0, 0

    def growth(self, o, uses):
        srcs = o.reg_srcs()
        return int(o.dst is not None) - sum(1 for s in set(srcs) if uses[s] == srcs.count(s))

    def feasible(self, o):
        grow = self.growth(o, self.uses)
        if grow <= 0:
            return True
        live, uses, i = self.live + grow, dict(self.uses), self.index[o]
        if live > self.budget:
            return False
        for s in o.reg_srcs():
            uses[s] -= 1
        for j in range(self.head, len(self.ops)):
            if self.taken[j] or j == i:
                continue
            live += self.growth(self.ops[j], uses)
            if live > self.budget:
                return False
            for s in self.ops[j].reg_srcs():
                uses[s] -= 1
        return True

    def take(self, o):
        self.live += self.growth(o, self.uses)
        for s in o.reg_srcs():
            self.uses[s] -= 1
        self.taken[self.index[o]] = True
        while self.head < len(self.ops) and self.taken[self.head]:
            self.head += 1


def peak_live(ops):
    """The most values live at once when ops run in this order."""
    uses = {}
    for o in ops:
        for s in o.reg_srcs():
            uses[s] = uses.get(s, 0) + 1
    live = top = 0
    for o in ops:
        srcs = o.reg_srcs()
        live += int(o.dst is not None) - sum(1 for s in set(srcs) if uses[s] == srcs.count(s))
        top = max(top, live)
        for s in srcs:
            uses[s] -= 1
    return top


def lean_order(ops, preds, after):
    """A topological order (data and ordering edges) that takes, at each step, the ready op that
    grows the live set least (then the earliest): for interleaved graphs (multiply_both) whose
    source order needs more registers than there are."""
    index = {o: i for i, o in enumerate(ops)}
    uses = {}
    for o in ops:
        for s in o.reg_srcs():
            uses[s] = uses.get(s, 0) + 1
    done, out, rest = set(), [], list(ops)
    while rest:
        ready = [o for o in rest if all(p in done for p in preds[o]) and all(u in done for u in after[o])]

        def growth(o):
            srcs = o.reg_srcs()
            return int(o.dst is not None) - sum(1 for s in set(srcs) if uses[s] == srcs.count(s))
        o = min(ready, key=lambda o: (growth(o), index[o]))
        for s in o.reg_srcs():
            uses[s] -= 1
        done.add(o)
        out.append(o)
        rest.remove(o)
    return out


def assignable(sets):
    """Whether uops with these pipe sets can each take a pipe of its set, one uop per pipe
    (augmenting paths)."""
    owner = {}

    def place(i, seen):
        for p in sets[i]:
            if p not in seen:
                seen.add(p)
                if p not in owner or place(owner[p], seen):
                    owner[p] = i
                    return True
        return False
    return all(place(i, set()) for i in range(len(sets)))


def schedule(ops, budget, knobs):
    """Cycle-driven list scheduling on the machine's vector pipes (each uop to a free pipe it may
    use), loads and one store per cycle. Priority: the longest latency path to a sink, plus
    jitter. An op that would raise the live registers above `budget` waits. A tied op waits for
    every other use of the source it overwrites."""
    rng = random.Random(knobs.seed) if knobs.seed else None
    jitter = {o: rng.random() * knobs.jitter if rng else 0.0 for o in ops}
    producer = {o.dst: o for o in ops if o.dst is not None}
    preds = {o: [producer[s] for s in o.reg_srcs() if s in producer] for o in ops}
    after = {o: [] for o in ops}  # ordering only: other readers of a tied source, memory order
    for o in ops:
        if o.tied and o.srcs[0].kind == 'reg':
            after[o] = [u for u in ops if u is not o and o.srcs[0] in u.srcs]
    accesses = [(o, memory_accesses(o)) for o in ops]
    for i, (o, (reads, writes)) in enumerate(accesses):
        for p, (p_reads, p_writes) in accesses[:i]:
            if overlap(p_writes, reads + writes) or overlap(p_reads, writes):
                after[o].append(p)
    succs = {o: [] for o in ops}
    for o in ops:
        for p in preds[o]:
            succs[p].append(o)
    priority = {}
    for o in reversed(ops):
        priority[o] = latency(o, knobs) + max((priority[s] for s in succs[o]), default=0)
    uses = {}
    for o in ops:
        for s in o.reg_srcs():
            uses[s] = uses.get(s, 0) + 1
    index = {o: i for i, o in enumerate(ops)}

    def delta(o):
        dying = len({s for s in o.reg_srcs() if uses[s] == o.reg_srcs().count(s)})
        return dying, int(o.dst is not None)

    pipes = MACHINE.vector_pipes
    issued_at, order, remaining, live, cycle, stall = {}, [], list(ops), 0, 0, 0
    finish = {}
    lv = None
    if FEASIBLE:  # the reference order: the source order if it fits the budget
        lv = Liveness(ops if peak_live(ops) <= budget else lean_order(ops, preds, after), budget)
    while remaining:
        free = {p: 1 for p in pipes}
        free['load'], free['store'] = knobs.loads_per_cycle, 1
        uops = []  # MATCHING: the pipe sets of the uops issued this cycle
        candidates = remaining[:knobs.window] if knobs.window else remaining
        if lv and knobs.window and lv.ops[lv.head] not in candidates:  # the reference order's next op
            candidates = candidates + [lv.ops[lv.head]]
        ready = [o for o in candidates if all(p in finish and finish[p] <= cycle for p in preds[o])
                 and all(u in issued_at for u in after[o])]

        def key(o):
            dying, new = delta(o)
            grows = new - dying > 0 and live - dying + new > budget - knobs.margin
            return grows, -priority[o] - jitter[o], index[o]
        ready.sort(key=key)
        issued = False
        for o in ready:
            if any(u not in issued_at for u in after[o]):
                continue
            loads = o.loads()
            if free['load'] < loads:
                continue
            taken = []
            if o.unit() == 'vector' and MATCHING:
                cost = COST[o.mnemonic]
                if knobs.width and len(uops) + cost.uops > knobs.width:
                    continue
                if not assignable(uops + [cost.pipes] * cost.uops):
                    continue
            elif o.unit() == 'vector':
                cost = COST[o.mnemonic]
                for _ in range(cost.uops):
                    p = next((p for p in cost.pipes if free.get(p, 0) > taken.count(p)), None)
                    if p is None:
                        break
                    taken.append(p)
                if len(taken) < cost.uops:
                    continue
            elif o.unit() == 'store' and free['store'] == 0:
                continue
            dying, new = delta(o)
            if live - dying + new > budget or (lv and not lv.feasible(o)):
                continue
            if lv:
                lv.take(o)
            for p in taken:
                free[p] -= 1
            if o.unit() == 'vector' and MATCHING:
                uops += [COST[o.mnemonic].pipes] * COST[o.mnemonic].uops
            if o.unit() == 'store':
                free['store'] -= 1
            free['load'] -= loads
            live += new - dying
            for s in set(o.reg_srcs()):
                uses[s] -= o.reg_srcs().count(s)
            finish[o] = cycle + latency(o, knobs)
            issued_at[o] = cycle
            remaining.remove(o)
            order.append(o)
            issued = True
        stall = 0 if issued or any(f > cycle for f in finish.values()) else stall + 1
        if stall > 2:
            raise ScheduleError('register budget deadlock')
        cycle += 1
    return order


def allocate(seq, free_regs):
    """Linear scan: a register frees at its value's last use and goes to the next new value; a
    tied op's result takes the register of the source it overwrites, which dies there."""
    last = {}
    for i, o in enumerate(seq):
        for s in o.reg_srcs():
            last[s] = i
    free = list(reversed(free_regs))
    reg, live = {}, {}
    for i, o in enumerate(seq):
        dying = [s for s in dict.fromkeys(o.reg_srcs()) if last.get(s) == i]
        if o.tied:
            src = o.srcs[0]
            if src.kind != 'reg' or src not in dying:
                raise ScheduleError('tied source still live')
            dying.remove(src)
            reg[o.dst] = reg[src]
            live[reg[src]] = o.dst
        for s in dying:
            if live.get(reg[s]) is s:
                del live[reg[s]]
                free.append(reg[s])
        if o.dst is None or o.tied:
            continue
        if o.dst not in last:
            raise ScheduleError('dead value')
        if not free:
            raise ScheduleError('out of registers')
        reg[o.dst] = free.pop()
        live[reg[o.dst]] = o.dst
    return reg


# --------------------------------------------------------------------------------- emission

def address(m):
    base, index, scale, disp = m
    d = str(disp) if disp else ''
    if index is None:
        return f'{d}(%[{base}])'
    return f'{d}(%[{base}],%[{index}]{"," + str(scale) if scale != 1 else ""})'


def operand(v, reg, bcst=True):
    if v.kind == 'mem':
        return address(v.mem)
    if v.kind == 'bcst':
        return address(v.mem) + ('%{1to16%}' if bcst else '')
    if v.kind == 'fixed':
        return f'%[{v.reg}]' if isinstance(v.reg, str) else f'%%zmm{v.reg}'
    return f'%%zmm{reg[v]}'


def emit(o, reg):
    if o.store is not None and o.mask:
        return f'vmovdqu32 {operand(o.srcs[0], reg)}, {address(o.store)}%{{%[{o.mask}]%}}'
    if o.store is not None:
        return f'vmovdqu64 {operand(o.srcs[0], reg)}, {address(o.store)}'
    dst = f'%%zmm{reg[o.dst]}'
    m = o.mnemonic
    if m in LOADS:
        return f'{m} {operand(o.srcs[0], reg, bcst=False)}, {dst}'
    if m in ('vpsrlq', 'vpsllq'):
        return f'{m} ${o.imm}, {operand(o.srcs[0], reg)}, {dst}'
    if m == 'vpermt2d':  # dst = srcs[0] (overwritten), index srcs[1], second table srcs[2]
        return f'vpermt2d {operand(o.srcs[2], reg)}, {operand(o.srcs[1], reg)}, {dst}'
    if m == 'vpermi2d':  # dst = srcs[0], the index (overwritten); tables srcs[1], srcs[2]
        return f'vpermi2d {operand(o.srcs[2], reg)}, {operand(o.srcs[1], reg)}, {dst}'
    if m == 'vpbroadcastd_mask':  # dst = srcs[0] with its high 8 words from srcs[1]
        return f'vpbroadcastd {operand(o.srcs[1], reg, bcst=False)}, {dst}%{{%[hi]%}}'
    if m == 'vpblendmd':  # srcs[0] with its odd words from srcs[1]
        return f'vpblendmd {operand(o.srcs[1], reg)}, {operand(o.srcs[0], reg)}, {dst}%{{%[odd]%}}'
    if m == 'vpermd':  # dst = table srcs[0] permuted by the index srcs[1]
        return f'vpermd {operand(o.srcs[0], reg)}, {operand(o.srcs[1], reg)}, {dst}'
    x, y = o.srcs  # AT&T: dst = x OP y is "op y, x, dst"; only y may be in memory
    if m in COMMUTATIVE and x.in_memory:
        x, y = y, x
    return f'{m} {operand(y, reg)}, {operand(x, reg)}, {dst}'


def asm_lines(lines, indent='        '):
    return [f'{indent}"{line}\\n\\t"' for line in lines]


def compile_graph(g, reserved, knobs):
    g.finish()
    regs = [r for r in range(32) if r not in reserved]
    target = MACHINE.target
    if knobs.model:
        use(knobs.model)
    try:
        seq = schedule(g.ops, len(regs), knobs)
    finally:
        use(target)
    reg = allocate(seq, regs)
    return [emit(o, reg) for o in seq]


# ------------------------------------------------------------------------------ butterflies
# Forward (Cooley-Tukey), inputs < In P, outputs < 4P:
#   a = low(f0), b = low(f1) (In = 4), c = x f2, d = x f3, ac = low(a + c), amc = low(a - c),
#   out = ac + y (b + d), ac - y (b + d) + 2P, amc + z (b - d + 2P), amc - z (b - d + 2P) + 2P.
# Inverse (Gentleman-Sande), inputs and outputs < 2P:
#   ab = low(a + b), cd = low(c + d), amb = y (a - b + 2P), cmd = z (c - d + 2P),
#   out = low(ab + cd), low(amb + cmd), x (ab - cd + 2P), x (amb - cmd + 2P).
# Identity groups have x = y = 1.

def forward4(g, f, tw, kin=4, odd=lambda t: None):
    a, b = (g.low(f[0]), g.low(f[1])) if kin > 2 else (f[0], f[1])
    c, d = g.shoup(f[2], tw['X'], odd(2)), g.shoup(f[3], tw['X'], odd(3))
    ac, amc = g.low(g.add(a, c)), g.low_diff(a, c)
    y, z = g.shoup(g.add(b, d), tw['Y']), g.shoup(g.diff(b, d), tw['Z'])
    return [g.add(ac, y), g.diff(ac, y), g.add(amc, z), g.diff(amc, z)]


def inverse4(g, f, tw):
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb, cmd = g.shoup(g.diff(f[0], f[1]), tw['Y']), g.shoup(g.diff(f[2], f[3]), tw['Z'])
    return [g.low(g.add(ab, cd)), g.low(g.add(amb, cmd)), g.shoup(g.diff(ab, cd), tw['X']), g.shoup(g.diff(amb, cmd), tw['X'])]


def forward_identity4(g, f, tw, kin=4):
    """intrinsics.hpp's forward_identity_first and _second: out < 4P."""
    a, b, c, d = (g.low(x) for x in f) if kin > 2 else f
    if kin == 1:  # canonical in: sums < 2P without reductions
        ac, amc, bd = g.add(a, c), g.sub(g.add(a, g.c['P']), c), g.add(b, d)
    else:
        ac, amc, bd = g.low(g.add(a, c)), g.low_diff(a, c), g.low(g.add(b, d))
    zd = g.shoup(g.diff(b, d), tw['Z'])
    return [g.add(ac, bd), g.diff(ac, bd), g.add(amc, zd), g.diff(amc, zd)]


def inverse_identity4(g, f, tw):
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb, cmd = g.low_diff(f[0], f[1]), g.shoup(g.diff(f[2], f[3]), tw['Z'])
    return [g.low(g.add(ab, cd)), g.low(g.add(amb, cmd)), g.low_diff(ab, cd), g.low_diff(amb, cmd)]


def inverse_scaled4(g, f, tw):
    """intrinsics.hpp's inverse_group_scaled, k > 0: inverse4 with every output times s; twiddles S
    = s, SX = s x, Y = s y, Z = s z and X = x."""
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb, cmd = g.shoup(g.diff(f[0], f[1]), tw['Y']), g.shoup(g.diff(f[2], f[3]), tw['Z'])
    return [g.shoup(g.add(ab, cd), tw['S']), g.low(g.add(amb, cmd)), g.shoup(g.diff(ab, cd), tw['SX']),
            g.shoup(g.diff(amb, cmd), tw['X'])]


def inverse_scaled_identity4(g, f, tw):
    """inverse_scaled4() for k = 0: s y = s, x = 1."""
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb, cmd = g.shoup(g.diff(f[0], f[1]), tw['S']), g.shoup(g.diff(f[2], f[3]), tw['Z'])
    return [g.shoup(g.add(ab, cd), tw['S']), g.low(g.add(amb, cmd)), g.shoup(g.diff(ab, cd), tw['S']),
            g.low_diff(amb, cmd)]


# --------------------------------------------------------------------------------- column loops
# One group's quarters q words apart at a: a column is vector t of each quarter, at a + disp +
# t q (h = 4q bytes, h3 = 12q). Twiddles broadcast once per call into fixed registers.

QUARTER = [(None, 1), ('h', 1), ('h', 2), ('h3', 1)]
REG_P, REG_2P, REG_J = 31, 30, 29
TWIDDLE_REGS = {'X': (23, 24), 'Y': (25, 26), 'Z': (27, 28)}
TWIDDLE_SOURCES = {'X': ('x', 0), 'Y': ('y', 0), 'Z': ('y', 4)}  # value; its quotient 32 bytes on
PREFETCH = 2048  # bytes ahead in each quarter (512 words, intrinsics.hpp's kPrefetch)


@dataclass
class Loop:
    name: str
    butterfly: object
    twiddles: str             # 'XYZ' (group k > 0) or 'Z' (identity)
    doc: str
    kin: Optional[int] = None
    columns: int = 2          # per iteration
    odd_loads: bool = True    # forward: odd words of quarters 2, 3 from loads 4 bytes on
    prefetch: bool = False
    knobs: Knobs = field(default_factory=Knobs)
    join: str = 'perm'
    sources: Optional[tuple] = None  # (name, argument, byte offset) per twiddle, else by `twiddles`
    src: bool = False         # reads the columns at src, writes them at a


def loop_twiddles(loop):
    """(name, (value register, quotient register), (argument, byte offset)) per twiddle."""
    if loop.sources is None:
        return [(n, TWIDDLE_REGS[n], TWIDDLE_SOURCES[n]) for n in loop.twiddles]
    return [(n, (28 - 2 * i, 27 - 2 * i), (arg, disp)) for i, (n, arg, disp) in enumerate(loop.sources)]


def loop_args(loop):
    """The twiddle pointers' names, in order."""
    if loop.sources is None:
        return ['x', 'y'] if loop.twiddles == 'XYZ' else ['y']
    return list(dict.fromkeys(arg for _, arg, _ in loop.sources))


def loop_consts():
    return {'P': Value('fixed', reg=REG_P), '2P': Value('fixed', reg=REG_2P), 'J': Value('fixed', reg=REG_J)}


def loop_body(loop):
    twiddles = loop_twiddles(loop)
    tw = {n: (Value('fixed', reg=regs[0]), Value('fixed', reg=regs[1])) for n, regs, _ in twiddles}
    reserved = {REG_P, REG_2P, REG_J} | {r for _, regs, _ in twiddles for r in regs}
    g = Graph(loop_consts(), loop.join)
    base = 'src' if loop.src else 'a'
    for k in range(loop.columns):
        quarter = lambda t, extra=0, k=k: g.at(base, 64 * k + extra, *QUARTER[t])
        f = [quarter(t) for t in range(4)]
        if loop.butterfly is forward4:
            out = forward4(g, f, tw, loop.kin, (lambda t: quarter(t, 4)) if loop.odd_loads else (lambda t: None))
        elif loop.kin is not None:
            out = loop.butterfly(g, f, tw, loop.kin)
        else:
            out = loop.butterfly(g, f, tw)
        for t, v in enumerate(out):
            g.store(v, mem('a', 64 * k, *QUARTER[t]))
    body = compile_graph(g, reserved, loop.knobs)
    head = ['vpbroadcastd %[P], %%zmm31', 'vpbroadcastd %[P2], %%zmm30', 'vmovdqu64 %[J], %%zmm29']
    for _, regs, (arg, disp) in twiddles:
        head += [f'vpbroadcastd {disp}(%[{arg}]), %%zmm{regs[0]}', f'vpbroadcastd {disp + 32}(%[{arg}]), %%zmm{regs[1]}']
    if loop.prefetch:  # the line PREFETCH bytes ahead in each quarter, per column
        pf = [f'prefetcht0 {address(mem("a", PREFETCH + 64 * k, *QUARTER[t]))}' for k in range(loop.columns) for t in range(4)]
        body = pf[:4] + body[:len(body) // 2] + pf[4:] + body[len(body) // 2:]
    tail = [f'add ${64 * loop.columns}, %[a]'] + ([f'add ${64 * loop.columns}, %[src]'] if loop.src else [])
    tail += ['cmp %[end], %[a]', 'jne 1b']
    return head + ['.p2align 5', '1:'] + body + tail


LOOP_CLOBBERS = ', '.join(f'"xmm{i}"' for i in range(32)) + ', "memory", "cc"'


def loop_function(loop):
    args = loop_args(loop)
    twiddle_args = ', '.join(f'const u32* {a}' for a in args)
    lines = [f'// {line}' for line in loop.doc.split('\n')]
    src = 'const u32* src, ' if loop.src else ''
    lines += [f'[[gnu::always_inline]] inline void {loop.name}({src}u32* a, std::size_t q, std::size_t columns, {twiddle_args}) {{',
              '    const u32* const end = a + 16 * columns;', '    asm volatile(']
    lines += asm_lines(loop_body(loop))
    if loop.sources is None:
        inputs = '[end] "r"(end), [h] "r"(4 * q), [h3] "r"(12 * q), [y] "r"(y)' + (', [x] "r"(x)' if loop.twiddles == 'XYZ' else '')
    else:
        inputs = '[end] "r"(end), [h] "r"(4 * q), [h3] "r"(12 * q), ' + ', '.join(f'[{a}] "r"({a})' for a in args)
    if loop.join != 'perm':
        inputs += ', [odd] "Yk"(__mmask16(0xaaaa))'
    outputs = '[a] "+r"(a)' + (', [src] "+r"(src)' if loop.src else '')
    lines += [f'        : {outputs}', f'        : {inputs},', '          [P] "m"(kP), [P2] "m"(k2P), [J] "m"(kJoin)',
              f'        : {LOOP_CLOBBERS});', '}']
    return lines


LOOP_DOC = {
    'forward': ('Forward radix-4 butterflies of group k > 0 on columns [0, columns) of the quarters q words\n'
                'apart at a, in place; in < 4P, out < 4P. x = r + slot(k), y = r + slot(2k). Reads 4 bytes\n'
                'past the column (the odd words of quarters 2, 3). columns a multiple of 4.'),
    'forward_in2': 'forward() for inputs < 2P (no input reductions).',
    'inverse': ('Inverse radix-4 butterflies of group k > 0, in place, in and out < 2P; x = ir + slot(k),\n'
                'y = ir + slot(2k).'),
    'forward_identity': 'forward() for group 0 (x = y = 1), y = r: z = r(1) at y + 4.',
    'forward_identity_in2': 'forward_identity() for inputs < 2P.',
    'forward_identity_in1': 'forward_identity() for canonical inputs.',
    'inverse_identity': 'inverse() for group 0, y = ir.',
}

# Knobs from timing families on lc-intel (Emerald Rapids), groups of 16 columns on L1 data
# (notes/150-r1.md): 4 columns per iteration, deterministic schedule, vpermt2d joins.
LOOPS = [
    Loop('forward', forward4, 'XYZ', LOOP_DOC['forward'], kin=4, columns=4),
    Loop('forward_in2', forward4, 'XYZ', LOOP_DOC['forward_in2'], kin=2, columns=4),
    Loop('inverse', inverse4, 'XYZ', LOOP_DOC['inverse']),
    Loop('forward_identity', forward_identity4, 'Z', LOOP_DOC['forward_identity'], kin=4),
    Loop('forward_identity_in2', forward_identity4, 'Z', LOOP_DOC['forward_identity_in2'], kin=2),
    Loop('forward_identity_in1', forward_identity4, 'Z', LOOP_DOC['forward_identity_in1'], kin=1),
    Loop('inverse_identity', inverse_identity4, 'Z', LOOP_DOC['inverse_identity']),
]
# What avx512_kernels.hpp holds: the loops that beat intrinsics.hpp's on Emerald Rapids (the
# others ran as fast: identity groups, inverse groups; notes/150-r1.md), with prefetching forms.
EMITTED_LOOPS = ('forward', 'forward_in2')
PREFETCHED = ('forward', 'forward_in2', 'inverse')


def with_prefetch(loop):
    return Loop(loop.name + '_prefetch', loop.butterfly, loop.twiddles,
                f'{loop.name}() with prefetcht0 {PREFETCH} bytes ahead in each quarter (quarters past L2).',
                loop.kin, loop.columns, loop.odd_loads, True, loop.knobs, loop.join)


# ---------------------------------------------------------------------------------- bottoms
# Two groups k, k + 1 (k even) of the bottom level, 64 words at a: vector t holds quarter t of
# both, [group k, group k + 1] (transposed in and out by vpermt2d), so the butterflies are
# lane-wise with twiddles [w_k x 8, w_(k+1) x 8]: x = r(k), r(k + 1) at x, x + 4; y = r(2k),
# r(2k + 2) at y, y + 8; z = r(2k + 1), r(2k + 3) at y + 4, y + 12; quotients 32 bytes on.
# Each statement loads its constants into fixed registers first (loads only, from kIndex): P, 2P,
# the join index J, the transposes' indices LO ([0, 8) of both tables) and HI ([8, 16)), and for
# twiddles by vpermd ('perm') the 6 indices of TWIDDLE_LANES; the masks hi (words 8-15) and odd.

BOTTOM_FIXED = {'P': 31, '2P': 30, 'J': 29, 'LO': 28, 'HI': 27}
# Twiddle pairs by vpermd of a table block: lanes of (value, quotient) for x, y, z.
TWIDDLE_LANES = {'XW': (0, 1), 'XQ': (8, 9), 'YW': (0, 2), 'YQ': (8, 10), 'ZW': (1, 3), 'ZQ': (9, 11)}
TWIDDLE_FIXED = {name: 26 - i for i, name in enumerate(TWIDDLE_LANES)}


def bottom_consts(twiddles):
    fixed = dict(BOTTOM_FIXED, **(TWIDDLE_FIXED if twiddles == 'perm' else {}))
    return {name: Value('fixed', reg=r) for name, r in fixed.items()}


def pair_twiddles(g, x, y, p=0):
    """The twiddles of a pair at x, y; p > 0: pair p of groups k + 2p, k + 2p + 1 with k a multiple
    of 2 * pairs (pairs <= 4), entries 2p and 4p on from x's and y's: in x's block, and in y's or
    the next one ('bcst' only)."""
    if g.twiddles == 'perm':
        xs, ys = g.at(x, 0), g.at(y, 0)
        perm = lambda table, lanes: g.op('vpermd', table, g.c[lanes])
        return {'X': (perm(xs, 'XW'), perm(xs, 'XQ')), 'Y': (perm(ys, 'YW'), perm(ys, 'YQ')),
                'Z': (perm(ys, 'ZW'), perm(ys, 'ZQ'))}
    dx, dy = 8 * p, 4 * (4 * p // 8 * 16 + 4 * p % 8)  # slot() offsets in bytes
    return {'X': (g.halves(x, dx, dx + 4), g.halves(x, dx + 32, dx + 36)),
            'Y': (g.halves(y, dy, dy + 8), g.halves(y, dy + 32, dy + 40)),
            'Z': (g.halves(y, dy + 4, dy + 12), g.halves(y, dy + 36, dy + 44))}


def load_pair(g, base, disp=0):
    """The quarter vectors of the 64 words disp bytes past base (two groups)."""
    z = [g.at(base, disp + 64 * i) for i in range(4)]
    lo, hi = g.c['LO'], g.c['HI']
    return [g.op('vpermt2d', g.load(z[0]), lo, z[2]), g.op('vpermt2d', g.load(z[0]), hi, z[2]),
            g.op('vpermt2d', g.load(z[1]), lo, z[3]), g.op('vpermt2d', g.load(z[1]), hi, z[3])]


def store_pair(g, base, q, disp=0):
    """load_pair undone: words [16i, 16i + 16) from quarters (q[0], q[1]) and (q[2], q[3]). By
    vpermi2d, or ('mask') by 8 stores of half vectors under masks (no shuffle uops)."""
    if g.ostore == 'mask':
        for t, v in enumerate(q):  # quarter t: words [8t, 8t + 8) of group k, [32 + 8t, ...) of k + 1
            g.ops.append(Op('vmovdqu32', None, [v], store=mem(base, disp + 32 * t), mask='lo'))
            g.ops.append(Op('vmovdqu32', None, [v], store=mem(base, disp + 96 + 32 * t), mask='hi'))
        return
    lo, hi = g.c['LO'], g.c['HI']
    for i, (u, v, idx) in enumerate([(q[0], q[1], lo), (q[2], q[3], lo), (q[0], q[1], hi), (q[2], q[3], hi)]):
        g.store(g.op('vpermi2d', g.load(INDEX[idx is hi]), u, v), mem(base, disp + 64 * i))


INDEX = {False: Value('mem', mem=mem('ix', 0)), True: Value('mem', mem=mem('ix', 64))}  # LO, HI in memory


def bottom_forward(g, pairs=1):
    """Forward butterflies of groups k, k + 1 at a, in < 4P; canonical leaves out. More pairs (k a
    multiple of 2 pairs): pair p (groups k + 2p, k + 2p + 1) at a + 64 p words."""
    for p in range(pairs):
        out = forward4(g, load_pair(g, 'a', 256 * p), pair_twiddles(g, 'x', 'y', p))
        store_pair(g, 'a', [g.canonical(g.low(v)) for v in out], 256 * p)


def bottom_inverse(g, pairs=1):
    """Inverse butterflies of groups k, k + 1 at a, in and out < 2P (inverse table). pairs: as
    bottom_forward."""
    for p in range(pairs):
        store_pair(g, 'a', inverse4(g, load_pair(g, 'a', 256 * p), pair_twiddles(g, 'x', 'y', p)), 256 * p)


def small_groups(g, inverse=False, identity_first=False, groups=4, odd_loads=False):
    """Radix-4 groups k .. k + groups - 1 (k a multiple of groups, groups <= 4) on quarters of 32
    words (two columns each), group i at a + 128 i words, in place: forward in < 4P, out < 4P;
    inverse in and out < 2P. Twiddles of group i: x + 4 i bytes (r(k + i)), y + 8 i (r(2k + 2i)),
    y + 8 i + 4 (r(2k + 2i + 1)), quotients 32 bytes on (x = t + slot(k), y = t + slot(2k)).
    identity_first: group k is group 0 (x = y = 1, z at y + 4). odd_loads: the odd words of
    quarters 2 and 3 from loads 4 bytes on (forward; 1.36x slower on Zen 5: loads across two
    recent stores do not forward)."""
    for i in range(groups):
        identity = identity_first and i == 0
        tw = {'X': (g.bcst('x', 4 * i), g.bcst('x', 4 * i + 32)), 'Y': (g.bcst('y', 8 * i), g.bcst('y', 8 * i + 32)),
              'Z': (g.bcst('y', 8 * i + 4), g.bcst('y', 8 * i + 36))}
        for c in range(2):
            def quarter(t, extra=0):
                return g.at('a', 512 * i + 64 * c + 128 * t + extra)
            f = [quarter(t) for t in range(4)]
            if inverse:
                out = inverse_identity4(g, f, tw) if identity else inverse4(g, f, tw)
            else:
                odd = (lambda t: quarter(t, 4)) if odd_loads else (lambda t: None)
                out = forward_identity4(g, f, tw, 4) if identity else forward4(g, f, tw, 4, odd)
            for t, v in enumerate(out):
                g.store(v, mem('a', 512 * i + 64 * c + 128 * t))


# Leaf products of a pair: leaf t of each group is a polynomial mod x^8 - w_t, w_t = y, -y, z, -z
# (the group's own twiddles y = r(2k), z = r(2k + 1)). Row t = [w_t A_t, A_t] per half (canonical;
# P - canonical(y A_t) for t odd), so x^(8-j) A_t mod x^8 - w_t is words j .. j + 7 of the row:
# window j, built by vpermi2d from the registers W = w_t A_t and A = A_t (index WINDOW[j]; the
# window 9 reads one don't-care word). Word l of the product: e (even l) and o (odd l) sum b_i
# times windows 8 - i and 9 - i over i < 8 (64-bit sums < 8 P^2), Montgomery reduction (factor
# 2^-32), join, low: < 2P. b_i per half by broadcasts from B_t stored at buf + 64 t.

NI = 998244351  # -1 / P mod 2^32


def window_index(j):
    idx = []
    for h in range(2):
        for l in range(8):
            s = j + l
            idx.append(8 * h + s if s < 8 else 16 + (8 * h + s - 8) % 16)
    return idx


WINDOWS = [j for j in range(1, 10) if j != 8]  # window 8 is A itself


def leaf_rows(g, a, tw):
    """Registers (W_t, A_t) per leaf t: W_t = canonical(w_t A_t), P - that for t odd."""
    rows = []
    for t, v in enumerate(a):
        w = g.canonical(g.shoup(v, tw['Y'] if t < 2 else tw['Z']))
        if t % 2:
            w = g.sub(g.c['P'], w)
        rows.append((w, v))
    return rows


def leaf_product(g, row, t):
    w, a = row
    win = {8: a}
    for j in WINDOWS:
        win[j] = g.op('vpermi2d', g.load(g.at('ix', 128 + 64 * WINDOWS.index(j))), w, a)
    e = o = None
    for i in range(8):
        b = g.halves('buf', 64 * t + 4 * i, 64 * t + 32 + 4 * i)
        pe, po = g.op('vpmuludq', win[8 - i], b), g.op('vpmuludq', win[9 - i], b)
        e = pe if e is None else g.op('vpaddq', e, pe)
        o = po if o is None else g.op('vpaddq', o, po)
    ni, p = g.c['NI'], g.c['P']
    e = g.op('vpaddq', e, g.op('vpmuludq', g.op('vpmuludq', e, ni), p))
    o = g.op('vpaddq', o, g.op('vpmuludq', g.op('vpmuludq', o, ni), p))
    return g.low(g.join_high(e, o))


def bottom_multiply(g, square=False):
    """Groups k, k + 1 of a (and b): forward butterflies (in < 4P), leaf products a b (a a if
    square) and inverse butterflies (twiddles at ix_, iy_) into a, out < 2P."""
    tw = pair_twiddles(g, 'x', 'y')
    a = [g.canonical(g.low(v)) for v in forward4(g, load_pair(g, 'a'), tw)]
    bs = a if square else [g.canonical(g.low(v)) for v in forward4(g, load_pair(g, 'b'), tw)]
    for t, v in enumerate(bs):
        g.store(v, mem('buf', 64 * t))
    rows = leaf_rows(g, a, tw)
    c = [leaf_product(g, rows[t], t) for t in range(4)]
    itw = pair_twiddles(g, 'ix_', 'iy_')
    store_pair(g, 'a', inverse4(g, c, itw))


# The bottom of a product in two stages, as gen_avx2_product_kernels.py's: first (forward
# butterflies into a leaf buffer: W_t at 128 t, A_t at 128 t + 64, B_t at 512 + 64 t) and last
# (leaf products from the buffer, inverse butterflies, stores to out); both: last of a pair
# interleaved with first of the next.
LEAF_BUF = 768


def multiply_first(g, square=False, pairs=1):
    """square (or 'stored'): the rows of a alone. pairs = 2: groups k .. k + 3 (k a multiple of 4),
    pair p at a + 64 p words (and b) into buf + LEAF_BUF p bytes."""
    for p in range(pairs):
        tw = pair_twiddles(g, 'x', 'y', p)
        da, dbuf = 256 * p, LEAF_BUF * p
        a = [g.canonical(g.low(v)) for v in forward4(g, load_pair(g, 'a', da), tw)]
        for t, (w, v) in enumerate(leaf_rows(g, a, tw)):
            g.store(w, mem('buf', dbuf + 128 * t))
            g.store(v, mem('buf', dbuf + 128 * t + 64))
        if not square:
            for t, v in enumerate(forward4(g, load_pair(g, 'b', da), tw)):
                g.store(g.canonical(g.low(v)), mem('buf', dbuf + 512 + 64 * t))


def multiply_last(g, square=False, pairs=1):
    """square: B_t is A_t; 'stored': B is a leaf form at bs (groups k, k + 1: 64 words). pairs = 2:
    pair p from cur + LEAF_BUF p bytes (and bs + 64 p words) to out + 64 p words."""
    for p in range(pairs):
        dcur, dout = LEAF_BUF * p, 256 * p
        c = []
        for t in range(4):
            w, a = g.at('cur', dcur + 128 * t), g.at('cur', dcur + 128 * t + 64)
            win = {8: a}
            for j in WINDOWS:
                win[j] = g.op('vpermi2d', g.load(g.at('ix', 128 + 64 * WINDOWS.index(j))), w, a)
            if square == 'stored':  # word i of B_t: bs + 32 t + 4 i (group k), 128 bytes on (group k + 1)
                base, b0, b1 = 'bs', dout + 32 * t, dout + 128 + 32 * t
            else:
                bt = dcur + (128 * t + 64 if square else 512 + 64 * t)
                base, b0, b1 = 'cur', bt, bt + 32
            e = o = None
            for i in range(8):
                b = g.halves(base, b0 + 4 * i, b1 + 4 * i)
                pe, po = g.op('vpmuludq', win[8 - i], b), g.op('vpmuludq', win[9 - i], b)
                e = pe if e is None else g.op('vpaddq', e, pe)
                o = po if o is None else g.op('vpaddq', o, po)
            ni, P = g.c['NI'], g.c['P']
            e = g.op('vpaddq', e, g.op('vpmuludq', g.op('vpmuludq', e, ni), P))
            o = g.op('vpaddq', o, g.op('vpmuludq', g.op('vpmuludq', o, ni), P))
            c.append(g.low(g.join_high(e, o)))
        store_pair(g, 'out', inverse4(g, c, pair_twiddles(g, 'ix_', 'iy_', p)), dout)


def multiply_both(g, square=False, shift=0.0, pairs=1):
    multiply_last(g, square, pairs)
    split = len(g.ops)
    multiply_first(g, square, pairs)
    older, newer = g.ops[:split], g.ops[split:]
    keyed = [(i / len(older), 0, i, o) for i, o in enumerate(older)]
    keyed += [(j / len(newer) + shift, 1, j, o) for j, o in enumerate(newer)]
    g.ops = [o for *_, o in sorted(keyed, key=lambda t: t[:3])]


@dataclass
class Bottom:
    name: str
    build: object
    args: list
    doc: str
    knobs: Knobs = field(default_factory=Knobs)
    join: str = 'perm'
    twiddles: str = 'bcst'  # pair twiddles by masked broadcasts ('bcst') or vpermd of a table block ('perm')
    ostore: str = 'perm'    # outputs transposed by vpermi2d ('perm') or stored in halves under masks ('mask')
    in_place: bool = False  # writes a (a u32* parameter)


ARG_TYPES = {'out': 'u32*', 'buf': 'u32*'}  # the others are read only

BOTTOMS = [
    Bottom('bottom_forward', bottom_forward, ['a', 'x', 'y'], (
        'Forward butterflies of the bottom groups k, k + 1 (k even) at a, 64 words, in < 4P: canonical\n'
        'leaves out. x = r + slot(k), y = r + slot(2k).')),
    Bottom('bottom_inverse', bottom_inverse, ['a', 'x', 'y'], (
        'Inverse butterflies of the bottom groups k, k + 1 at a, in and out < 2P. x = ir + slot(k),\n'
        'y = ir + slot(2k).')),
    Bottom('bottom_multiply', bottom_multiply, ['a', 'b', 'buf', 'x', 'y', 'ix_', 'iy_'], (
        'Groups k, k + 1 of a and b (in < 4P): forward butterflies, leaf products a b 2^-32 into a and\n'
        'inverse butterflies, out < 2P. x, y as bottom_forward; ix_, iy_ the same in the inverse table.\n'
        'buf: 256 bytes.'), Knobs(window=32)),
    Bottom('bottom_square', lambda g: bottom_multiply(g, square=True), ['a', 'buf', 'x', 'y', 'ix_', 'iy_'], (
        'bottom_multiply() of a and a (b not read).'), Knobs(window=32)),
]

# The product bottoms in stages (the fused ones above ran 0.97 of the intrinsics' time, these 0.93:
# notes/150-r1.md). Knobs from timing families on lc-intel: window 20, no jitter, no shift.
STAGED_KNOBS = Knobs(window=20)
STAGED = [
    ('multiply', False, ['a', 'b', 'buf', 'x', 'y'], ['out', 'cur', 'ix_', 'iy_'],
     'a times b: first() the forward butterflies of groups k, k + 1 of a and b'),
    ('square', True, ['a', 'buf', 'x', 'y'], ['out', 'cur', 'ix_', 'iy_'],
     'a times a: first() the forward butterflies of a'),
    ('stored', 'stored', ['a', 'buf', 'x', 'y'], ['out', 'cur', 'bs', 'ix_', 'iy_'],
     'a times the leaf form b (canonical, at bs: groups k, k + 1): first() the forward butterflies of a'),
]
for name, kind, first_args, last_args, what in STAGED:
    BOTTOMS += [
        Bottom(f'{name}_first', lambda g, kind=kind: multiply_first(g, kind), first_args, (
            f'The bottom stage of a product in two steps, two groups k, k + 1 (k even) at a time; {what}\n'
            f'into the leaf buffer buf ({LEAF_BUF} bytes, 64-byte aligned); x = r + slot(k), y = r + slot(2k).'),
            STAGED_KNOBS),
        Bottom(f'{name}_last', lambda g, kind=kind: multiply_last(g, kind), last_args, (
            'last(): the leaf products of the pair in the buffer cur (with the factor 2^-32) and the inverse\n'
            'butterflies into out, < 2P; ix_ = ir + slot(k), iy_ = ir + slot(2k).'), STAGED_KNOBS),
        Bottom(f'{name}_both', lambda g, kind=kind: multiply_both(g, kind), first_args + last_args, (
            'last() of the pair in cur interleaved with first() of the next pair (at a, into buf).'), STAGED_KNOBS),
    ]
EMITTED_BOTTOMS = tuple(f'{name}_{part}' for name, *_ in STAGED for part in ('first', 'last', 'both')
                        if not (name == 'stored' and part == 'first'))  # stored_first is square_first

# zen5_kernels.hpp (notes/152-r1.md): every loop, on the Zen 5 model, 4 columns per iteration (8
# for inverse identity groups), no window; forward groups of two columns 4 per statement; the
# bottom level in statements of 4 and 2 pairs; the product bottoms in stages, one pair each. Knobs
# from timing families on lc-zen5, kernels on L1 data, against intrinsics<Avx512>: forward groups
# 0.87, inverse 0.89, forward identity 0.86, inverse identity 0.91, two-column forward groups 0.77;
# bottom forward 0.80 (4 pairs), inverse 0.85; product bottom 0.91, square 0.82. The product
# bottom keeps a schedule on the Emerald Rapids model (model='avx512'): 0.91 against 0.93 at best
# on the Zen 5 model (28 knob settings). Round 2 (notes/152-r2.md) added the scaled inverse top
# groups (0.88-0.95) and first levels from other memory (0.87-0.98). Zen 5's op cache holds about
# 5.9K instructions and the hot code of a product is near that (past it, about 3 instructions per
# cycle), so a kernel's size counts as well as its speed: inverse groups of two columns (0.86-0.94
# on L1 data) made products at 2^10-2^11 3-4% slower and are not emitted.
ZEN5_LOOP_DOC = dict(LOOP_DOC, inverse=LOOP_DOC['inverse'] + ' columns a multiple of 4.',
                     forward_identity=LOOP_DOC['forward_identity'] + ' columns a multiple of 4.',
                     inverse_identity=LOOP_DOC['inverse_identity'] + ' columns a multiple of 8.')
ZEN5_LOOPS = [replace(l, columns=8 if l.name == 'inverse_identity' else 4, doc=ZEN5_LOOP_DOC[l.name]) for l in LOOPS]
# The inverse top groups with the scale (intrinsics.hpp's inverse_group_scaled, k < 4): s = the
# scaled block (s at s, quotient 32 bytes on), sx = s + k, sy = s + 2k (s y, then s z), x = ir +
# slot(k); for k = 0 s alone (s y = s, s z at s + 4).
ZEN5_LOOPS += [
    Loop('inverse_scaled', inverse_scaled4, '', (
        'inverse() of group k > 0 (k < 4) with every output times s: s = the scaled block, sx = s + k,\n'
        'sy = s + 2k, x = ir + slot(k). In and out < 2P. columns a multiple of 4.'), columns=4,
        sources=(('S', 's', 0), ('SX', 'sx', 0), ('Y', 'sy', 0), ('Z', 'sy', 4), ('X', 'x', 0))),
    Loop('inverse_scaled_identity', inverse_scaled_identity4, '', (
        'inverse_scaled() for group 0: s = the scaled block. columns a multiple of 4.'), columns=4,
        sources=(('S', 's', 0), ('Z', 's', 4))),
]
# The first level of a transform from other memory (a factor in the caller's memory): the columns
# at src (quarters q words apart, any alignment) into a; no reads past a column (odd words by
# shifts). Inputs < 2P (the subtrees' first levels under a radix-2 top). For canonical inputs (the
# radix-4 top at odd lg) the asm (0.94 on L1 data) made middle products at 2^9 3-8% slower: the
# intrinsics run there (notes/152-r2.md).
ZEN5_LOOPS += [
    Loop('forward_from_in2', forward4, 'XYZ', 'forward_in2() from src into a; reads nothing past the columns.',
         kin=2, columns=4, odd_loads=False, src=True),
    Loop('forward_identity_from_in2', forward_identity4, 'Z', 'forward_identity_in2() from src into a.', kin=2,
         columns=4, src=True),
]


# The bottom level in statements of 2 and 4 pairs, knobs per target from timing families on L1 data
# (issue #176, notes/176-r1.md), time against WideBottom<Avx512>. Zen 5: 4 pairs on its model with
# a window of 128 (forward 0.75, inverse 0.81; round 152's no window 0.80, 0.86), forward 2 pairs
# on the Emerald Rapids model (0.86; 0.89), inverse 2 pairs as round 152's (0.91: the Emerald
# Rapids model's 0.89 made inverse_values/9 10% slower, its loads close behind the join's stores).
# Emerald Rapids: forward 2 pairs on its model, window 128 (0.86-0.87), 4 pairs on the Zen 5 model,
# window 128 (0.85-0.88: its own scheduler deadlocks there); inverse 4 pairs on the Zen 5 model,
# window 64 (0.95-0.97), 2 pairs 1.00 and 1 pair 1.08-1.10 (not emitted: the intrinsics run).
PAIR_KNOBS = {
    'zen5': {('bottom_forward', 4): Knobs(window=128), ('bottom_inverse', 4): Knobs(window=128),
             ('bottom_forward', 2): Knobs(window=128, model='avx512'), ('bottom_inverse', 2): Knobs()},
    'avx512': {('bottom_forward', 4): Knobs(window=128, model='zen5'), ('bottom_forward', 2): Knobs(window=128),
               ('bottom_inverse', 4): Knobs(window=64, model='zen5')},
}


def pair_bottoms(target):
    named = {b.name: b for b in BOTTOMS}
    out = []
    for name, build in (('bottom_forward', bottom_forward), ('bottom_inverse', bottom_inverse)):
        base = named[name]
        for pairs in (2, 4):
            knobs = PAIR_KNOBS[target].get((name, pairs))
            if knobs is None:
                continue
            doc = (f'{base.doc}\n{name}{pairs}(): {pairs} such pairs, k a multiple of {2 * pairs}: pair p (groups k + 2p, k + 2p + 1)\n'
                   f'at a + 64 p words.')
            out.append(replace(base, name=f'{name}{pairs}', doc=doc, knobs=knobs, twiddles='bcst', ostore='perm',
                               build=lambda g, build=build, pairs=pairs: build(g, pairs), in_place=True))
    return out


def zen5_bottoms():
    out = pair_bottoms('zen5')
    for name, inverse, window in (('forward_groups4', False, 128),):
        doc = (f'{"Inverse" if inverse else "Forward"} radix-4 groups k .. k + 3 (k a multiple of 4) on quarters of 32 words, group\n'
               f'k + i at a + 128 i words, in place; {"in and out < 2P" if inverse else "in < 4P, out < 4P"}. x = t + slot(k), y = t + slot(2k)\n'
               f'({"ir" if inverse else "r"}). {name}_first(): k = 0 (group 0 the identity).')
        for suffix, first in (('_first', True), ('', False)):
            out.append(Bottom(name + suffix, lambda g, inverse=inverse, first=first: small_groups(g, inverse, first),
                              ['a', 'x', 'y'], doc if not first else f'{name}() for k = 0.', Knobs(window=window), in_place=True))
    # The product bottoms one pair per stage. Two pairs per stage (multiply_first(g, kind, 2) ...)
    # ran 0.88 of the intrinsics' time on L1 data against 0.91, but whole products 2-10% slower
    # (twice the code: notes/152-r1.md).
    for b in BOTTOMS:
        if b.name in EMITTED_BOTTOMS:
            knobs = Knobs(window=64, model='avx512') if b.name.startswith('multiply') else Knobs(window=64)
            out.append(replace(b, knobs=knobs))
    return out


ZEN5_BOTTOMS = zen5_bottoms()


def index_table():
    """kIndex rows: LO, HI, the windows 1-7 and 9, NI (broadcast), J, the twiddle lanes, P, 2P, and
    the masks (words 0 and 1 of the last row)."""
    lo = list(range(8)) + list(range(16, 24))
    hi = list(range(8, 16)) + list(range(24, 32))
    join = [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31]
    lanes = [[a] * 8 + [b] * 8 for a, b in TWIDDLE_LANES.values()]
    rows = [lo, hi] + [window_index(j) for j in WINDOWS] + [[NI] * 16, join] + lanes
    rows += [[998244353] * 16, [2 * 998244353] * 16, [0xff00, 0xaaaa] + [0] * 14]
    return rows


INDEX_ROWS = len(index_table())
ROW = {'LO': 0, 'HI': 1, 'NI': 2 + len(WINDOWS), 'J': 3 + len(WINDOWS), 'P': INDEX_ROWS - 3, '2P': INDEX_ROWS - 2,
       'MASKS': INDEX_ROWS - 1, **{name: 4 + len(WINDOWS) + i for i, name in enumerate(TWIDDLE_LANES)}}


def bottom_body(bottom):
    consts = bottom_consts(bottom.twiddles)
    consts['NI'] = Value('bcst', mem=mem('ix', 64 * ROW['NI']))
    g = Graph(consts, bottom.join)
    g.twiddles, g.ostore = bottom.twiddles, bottom.ostore
    bottom.build(g)
    reserved = {v.reg for v in consts.values() if v.kind == 'fixed'}
    head = [f'vmovdqu64 {64 * ROW[name]}(%[ix]), %%zmm{v.reg}' for name, v in consts.items() if v.kind == 'fixed']
    return head + compile_graph(g, reserved, bottom.knobs)


def bottom_function(bottom):
    """The masks are operands (a kmovw costs a p5 uop, measured: notes/150-r1.md), so a caller's
    loop keeps them in mask registers."""
    body = bottom_body(bottom)
    params = ', '.join(f'{"u32*" if bottom.in_place and a == "a" else ARG_TYPES.get(a, "const u32*")} {a}'
                       for a in bottom.args)
    lines = [f'// {line}' for line in bottom.doc.split('\n')]
    lines += [f'[[gnu::always_inline]] inline void {bottom.name}({params}, __mmask16 lo = 0x00ff, __mmask16 hi = 0xff00,',
              '                                            __mmask16 odd = 0xaaaa) {', '    asm volatile(']
    lines += asm_lines(body)
    inputs = ', '.join(f'[{a}] "r"({a})' for a in bottom.args)
    lines += ['        :', f'        : {inputs}, [ix] "r"(kIndex), [lo] "Yk"(lo), [hi] "Yk"(hi), [odd] "Yk"(odd)',
              f'        : {BOTTOM_CLOBBERS});', '}']
    return lines


BOTTOM_CLOBBERS = ', '.join(f'"xmm{i}"' for i in range(32)) + ', "memory"'


# ------------------------------------------------------------------------------------ output

HEADER = '''\
// Generated by backend/ntt/gen_avx512_kernels.py. Do not edit.
// The NTT's butterflies on zmm as inline assembly, scheduled for {cpu}.
// Arithmetic and lazy ranges as intrinsics.hpp's (notes.md). Vectors hold 16 residues mod
// P = 998244353. A twiddle w < P comes with its Shoup quotient floor(w 2^32 / P) 8 words later
// (common.hpp, slot()).
#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace fps::ntt::{namespace} {{

using u32 = std::uint32_t;

inline constexpr u32 kP = 998244353, k2P = 2 * kP;
inline constexpr std::size_t kLeafBuffer = {leaf_words};  // words of a leaf buffer of the product bottoms
// vpermt2d index of join_high: words 2i + 1 of the even and odd products into words 2i, 2i + 1.
alignas(64) inline constexpr u32 kJoin[16] = {{1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31}};
'''

@dataclass
class Output:
    """A header of kernels scheduled for one target's machine (use(target))."""
    target: str
    path: Path
    namespace: str
    loops: list           # Loop, emitted in this order
    prefetched: tuple     # names of loops emitted with a prefetching form too
    bottoms: list         # Bottom


def generate(out):
    use(out.target)
    lines = HEADER.format(cpu=MACHINE.cpu, leaf_words=LEAF_BUF // 4, namespace=out.namespace).split('\n')
    rows = index_table()
    lines += ['// The bottoms\' constants (index_table() in the generator): the vpermt2d and vpermi2d indices of',
              '// the transposes (LO, HI) and windows (1-7 and 9), -1 / P mod 2^32, join_high\'s index, the',
              '// twiddle lanes, P, 2P, and the masks 0xff00 and 0xaaaa.',
              f'alignas(64) inline constexpr u32 kIndex[{len(rows)}][16] = {{']
    lines += ['    {' + ', '.join(str(x) for x in r) + '},' for r in rows] + ['};', '']
    for loop in out.loops:
        lines += loop_function(loop) + ['']
        if loop.name in out.prefetched:
            lines += loop_function(with_prefetch(loop)) + ['']
    for bottom in out.bottoms:
        lines += bottom_function(bottom) + ['']
    lines += [f'}}  // namespace fps::ntt::{out.namespace}']
    return '\n'.join(lines) + '\n'


OUTPUTS = [
    Output('avx512', OUT, 'avx512_kernels', [l for l in LOOPS if l.name in EMITTED_LOOPS], PREFETCHED,
           pair_bottoms('avx512') + [b for b in BOTTOMS if b.name in EMITTED_BOTTOMS]),
    Output('zen5', HERE / 'zen5_kernels.hpp', 'zen5_kernels', ZEN5_LOOPS, (), ZEN5_BOTTOMS),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = 0
    for out in OUTPUTS:
        text = generate(out)
        if not args.check:
            out.path.write_text(text)
        elif not out.path.exists() or out.path.read_text() != text:
            print(f'{out.path.name} is stale: run {Path(__file__).name}')
            stale = 1
    return stale


if __name__ == '__main__':
    sys.exit(main())
