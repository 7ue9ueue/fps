#!/usr/bin/env python3
"""Generate avx2_product_kernels.hpp, the asm kernels of the AVX2 product path (avx2.hpp,
ProductBottom and product()): the NTT's bottom stage with two groups per asm statement, and the
inverse top level with the scale.

Built on gen_avx2_kernels.py (same dataflow graphs, scheduler and register allocator). The bottom
stage differs from avx2_kernels.hpp's in three ways, measured on Zen 3 (notes.md):
- Two batches (radix-4 groups with h = 1) per statement. The caller passes an even group index,
  so the second batch's twiddles are in the same 8-entry table block: x + 4 and y + 8 bytes.
- No leaf weight array: the leaf moduli x^8 - w_t of group k have w_t = y, -y, z, -z with y, z the
  group's own forward twiddles r[2k], r[2k + 1]. The windows of t = 1, 3 hold P - canonical(y A_t)
  (in (0, P]) instead of a product with P - y.
- Leaf products in broadcast-reuse form: each coefficient of B is broadcast once and multiplies
  two windows read as memory operands.
Its functions are always inlined, so the operands stay in registers between statements.
inverse_top fuses avx2_kernels.hpp's inverse_identity, inverse (group 1) and scale_radix2 into one
pass, with the scale folded into the twiddles: 5 more Shoup products per column instead of 8.
inverse_top_to and inverse_top4_to (products from spans) write the first T output blocks to o:
the ops of the other outputs are dropped (a product cut to fewer words than the transform).
bottom_square_*: the same bottom stage for a square, a times a: no forward butterflies of b, and
the leaf products broadcast the coefficients of A from the window rows' second halves.
bottom_stored_*: the same for b given as leaves (Transform's fused ops): no forward butterflies of
b, the leaf products broadcast its coefficients from memory (bs, the two batches' 8 leaves).
bottom_leaves_*: the same with a given as leaves too: the windows are filled from a's canonical
leaves, no forward butterflies (the inverse of a leaf product, Transform::inverse_product).
bottom_leaves2_*, bottom_leaves3_*: the sum of 2 or 3 such products (a, a1, a2 times bs, bs1,
bs2), each reduced, then added; the leaf buffers of term k from 2 k LEAVES bytes on.

Usage: gen_avx2_product_kernels.py [--check]   (--check: exit 1 if avx2_product_kernels.hpp differs)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_avx2_kernels as gk  # noqa: E402

OUT = HERE / 'avx2_product_kernels.hpp'
BATCHES = 2
LEAVES = 384  # leaf buffer bytes per batch: window[4][16], then coefficients[4][8]
KNOBS = {'first': gk.Knobs(window=12),
         'last': gk.Knobs(seed=7, margin=2, jitter=3.0, window=12),
         'both': gk.Knobs(window=20)}
SHIFT = -0.05  # 'both': the first stage's ops start this fraction earlier in source order
# Square 'both': the knobs above deadlock the scheduler (register budget); these did not, with
# window 12 and shift -0.05 or -0.1 (notes.md, issue #55).
SQUARE_KNOBS = dict(KNOBS, both=gk.Knobs(window=16))
SQUARE_SHIFT = 0.0
# Leaves 'both': window 14 and more deadlock (register budget); 12 does not.
LEAVES_KNOBS = dict(SQUARE_KNOBS, both=gk.Knobs(window=12))
# Sums 'both': windows from 8 deadlock (register budget); 6 (2 terms) and 4 (3 terms) with shift -0.05 do not.
TERMS_KNOBS = {'leaves2': dict(LEAVES_KNOBS, both=gk.Knobs(window=6)), 'leaves3': dict(LEAVES_KNOBS, both=gk.Knobs(window=4))}
TERMS_SHIFT = -0.05
ARGS = {'first': ['a', 'b', 'buf', 'x', 'y'], 'last': ['out', 'cur', 'ix', 'iy'],
        'both': ['a', 'b', 'buf', 'x', 'y', 'out', 'cur', 'ix', 'iy']}
SQUARE_ARGS = {part: [a for a in args if a != 'b'] for part, args in ARGS.items()}
STORED_ARGS = {'first': SQUARE_ARGS['first'], 'last': ['out', 'cur', 'bs', 'ix', 'iy'],
               'both': SQUARE_ARGS['first'] + ['out', 'cur', 'bs', 'ix', 'iy']}
TERMS = {'leaves2': 2, 'leaves3': 3}  # sums of leaf products


def term_args(terms):
    a = ['a'] + [f'a{k}' for k in range(1, terms)]
    bs = ['bs'] + [f'bs{k}' for k in range(1, terms)]
    first = a + ['buf', 'x', 'y']
    last = ['out', 'cur'] + bs + ['ix', 'iy']
    return {'first': first, 'last': last, 'both': first + last}


def term_name(base, k):
    return base if k == 0 else f'{base}{k}'
DOC = {
    'first': 'Forward half of the bottom stage for the two batches at a, b; fills buf.',
    'last': 'Leaf products from cur and the inverse butterflies; writes the two batches at out.',
    'both': 'bottom_last() for the current two batches interleaved with bottom_first() for the next two.',
}


class Batch(gk.Graph):
    """A graph whose addresses move by a fixed offset per base: batch m of a statement."""

    def __init__(self, consts):
        super().__init__(consts, share_broadcasts=False)
        self.offset = {}

    def at(self, base, disp, fold=True):
        return super().at(base, disp + self.offset.get(base, 0), fold)

    def scalar(self, base, disp):
        return super().scalar(base, disp + self.offset.get(base, 0))

    def store(self, v, base, index, disp):
        super().store(v, base, index, disp + self.offset.get(base, 0))


def first(g, square=False):
    """Forward butterflies of a and b; leaf buffer rows [w_t A_t, A_t] and B_t, all in [0, P].
    square: a alone; B_t is A_t, the rows' second halves."""
    if square in TERMS:
        for k in range(TERMS[square]):
            first_rows(g, [g.materialize(g.at(term_name('a', k), 32 * t)) for t in range(4)], 2 * LEAVES * k)
        return
    leaves = square == 'leaves'
    values = [g.materialize(g.at('a', 32 * t)) for t in range(4)] if leaves else gk.forward_h1(g, 'a', 'x', 'y')
    for t, v in enumerate(values):
        a = v if leaves else g.canonical(v)
        g.store(a, 'buf', None, 64 * t + 32)
        w, q = (0, 32) if t < 2 else (4, 36)  # y or z and its quotient
        wa = g.shoup(a, g.scalar('y', w), g.scalar('y', q))
        wa = g.op('vpminud', wa, g.op('vpsubd', wa, g.c['P']))
        if t % 2:
            wa = g.op('vpsubd', g.c['P'], wa)  # (P - w) A = P - w A
        g.store(wa, 'buf', None, 64 * t)
    if square:
        return
    for t, v in enumerate(gk.forward_h1(g, 'b', 'x', 'y')):
        g.store(g.canonical(v), 'buf', None, 256 + 32 * t)


def first_rows(g, values, at):
    """Rows [w_t A_t, A_t] of canonical leaves A_t into buf from byte at (first() for leaves)."""
    for t, a in enumerate(values):
        g.store(a, 'buf', None, at + 64 * t + 32)
        w, q = (0, 32) if t < 2 else (4, 36)
        wa = g.shoup(a, g.scalar('y', w), g.scalar('y', q))
        wa = g.op('vpminud', wa, g.op('vpsubd', wa, g.c['P']))
        if t % 2:
            wa = g.op('vpsubd', g.c['P'], wa)
        g.store(wa, 'buf', None, at + 64 * t)


def leaf_sum(g, t, bs, at):
    """Leaf t's product from the windows at cur + at and b_j from bs: reduced, < 2P."""
    c = g.c
    e = o = None
    for j in range(8):
        b = g.materialize(g.scalar(bs, 32 * t + 4 * j))
        pe = g.op('vpmuludq', g.at('cur', at + 64 * t + 4 * (8 - j)), b)
        po = g.op('vpmuludq', g.at('cur', at + 64 * t + 4 * (9 - j)), b)
        e = pe if e is None else g.op('vpaddq', e, pe)
        o = po if o is None else g.op('vpaddq', o, po)
    e = g.op('vpaddq', e, g.op('vpmuludq', g.op('vpmuludq', e, c['NI']), c['P']))
    o = g.op('vpaddq', o, g.op('vpmuludq', g.op('vpmuludq', o, c['NI']), c['P']))
    return g.low(g.op('vpblendd', g.op('vpsrlq', e, imm=32), o, imm=0xAA))


def last(g, square=False):  # square: True, or 'stored' (b_j from bs)
    """Leaf t: e (even lanes) and o (odd lanes) sum b_j times the window 8 - j and 9 - j words in;
    sums < 8 P^2, then Montgomery reduction and the inverse butterfly as gen_kernels.bottom_last.
    square: b_j from A_t, words 8 + j of window t."""
    c = g.c
    leaves = []
    for t in range(4 if square not in TERMS else 0):
        e = o = None
        for j in range(8):
            if square in ('stored', 'leaves'):
                b = g.materialize(g.scalar('bs', 32 * t + 4 * j))
            else:
                b = g.materialize(g.scalar('cur', 64 * t + 32 + 4 * j if square else 256 + 32 * t + 4 * j))
            pe = g.op('vpmuludq', g.at('cur', 64 * t + 4 * (8 - j)), b)
            po = g.op('vpmuludq', g.at('cur', 64 * t + 4 * (9 - j)), b)
            e = pe if e is None else g.op('vpaddq', e, pe)
            o = po if o is None else g.op('vpaddq', o, po)
        e = g.op('vpaddq', e, g.op('vpmuludq', g.op('vpmuludq', e, c['NI']), c['P']))
        o = g.op('vpaddq', o, g.op('vpmuludq', g.op('vpmuludq', o, c['NI']), c['P']))
        leaves.append(g.low(g.op('vpblendd', g.op('vpsrlq', e, imm=32), o, imm=0xAA)))
    if square in TERMS:
        for t in range(4):
            leaf = None
            for k in range(TERMS[square]):
                x = leaf_sum(g, t, term_name('bs', k), 2 * LEAVES * k)
                leaf = x if leaf is None else g.low(g.add(leaf, x))
            leaves.append(leaf)
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


def body(part, square=False):
    g = Batch({'P': gk.Value('creg', reg=15), '2P': gk.Value('creg', reg=14), 'NI': gk.Value('creg', reg=13)})
    if part in ('last', 'both'):
        for m in range(BATCHES):
            g.offset = {'cur': LEAVES * m, 'out': 128 * m, 'bs': 128 * m, 'bs1': 128 * m, 'bs2': 128 * m,
                        'ix': 4 * m, 'iy': 8 * m}
            last(g, square)
    split = len(g.ops)
    if part in ('first', 'both'):
        for m in range(BATCHES):
            g.offset = {'a': 128 * m, 'a1': 128 * m, 'a2': 128 * m, 'b': 128 * m, 'buf': LEAVES * m, 'x': 4 * m,
                        'y': 8 * m}
            first(g, square)
    ops = g.ops
    if part == 'both':  # interleave the two stages' source orders proportionally
        older, newer = ops[:split], ops[split:]
        keyed = [(i / len(older), 0, i, o) for i, o in enumerate(older)]
        shift = TERMS_SHIFT if square in TERMS else SQUARE_SHIFT if square else SHIFT
        keyed += [(j / len(newer) + shift, 1, j, o) for j, o in enumerate(newer)]
        ops = [o for *_, o in sorted(keyed, key=lambda t: t[:3])]
    knobs = TERMS_KNOBS[square] if square in TERMS else LEAVES_KNOBS if square == 'leaves' else SQUARE_KNOBS if square else KNOBS
    seq = gk.schedule(ops, 13, knobs[part])
    reg = gk.allocate(seq, {13, 14, 15})
    head = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14', 'vpbroadcastd %[NI], %%ymm13']
    return head + [gk.emit(o, reg) for o in seq]


def function(part, square=False):
    stored = square in ('stored', 'leaves') or square in TERMS
    args = (term_args(TERMS[square])[part] if square in TERMS else STORED_ARGS[part] if stored
            else SQUARE_ARGS[part] if square else ARGS[part])
    params = ', '.join(f'{"void*" if a in ("buf", "out") else "const void*"} {a}' for a in args)
    doc = DOC[part] + (f' Leaves: the sum of {TERMS[square]} products of leaves (a.. times bs..).' if square in TERMS else
                       ' Leaves: a and b as leaves (a at a, b at bs).' if square == 'leaves' else
                       ' Stored: b as leaves at bs.' if stored else ' Square: a times a.' if square else '')
    name = f'bottom_{square}_{part}' if stored else f'bottom_square_{part}' if square else f'bottom_{part}'
    lines = [f'// {doc}', f'[[gnu::always_inline]] inline void {name}({params}) {{', '    asm volatile(']
    lines += gk.asm_lines(body(part, square))
    lines += ['        :', '        : ' + ', '.join(f'[{a}] "r"({a})' for a in args) + ',',
              '          [P] "m"(kP), [P2] "m"(k2P), [NI] "m"(kNI)',
              '        : ' + gk.CLOBBERS + ');', '}']
    return lines


# Inverse top of a transform of 8q vectors: blocks t < 4 (a) are the identity group's quarters,
# blocks 4 + t (b = a + 4q) group 1's; a column is vector j of each block. With twiddles z0 (group 0),
# x1, y1, z1 (group 1) and the scale s, values < 2P in:
#   U = s * inverse_identity(f), V = s * inverse(g); out_t = canonical(U_t + V_t),
#   out_{4+t} = canonical(U_t - V_t + 2P),
# where s is folded into the twiddles: U uses s, s z0; V uses s, s y1, s z1, s x1 and x1.
TOP_TWIDDLES = ['S', 'SZ0', 'SY1', 'SZ1', 'SX1', 'X1']


def top_column(ga, gb, blocks, upper=False):
    c = ga.c
    shoup = lambda g, x, name: g.shoup(x, c['W' + name], c['Q' + name])
    f = [ga.quarter(i) for i in range(4)]
    sab, samb = shoup(ga, ga.add(f[0], f[1]), 'S'), shoup(ga, ga.diff(f[0], f[1]), 'S')
    scd, scmd = shoup(ga, ga.add(f[2], f[3]), 'S'), shoup(ga, ga.diff(f[2], f[3]), 'SZ0')
    u = [ga.low(ga.add(sab, scd)), ga.low(ga.add(samb, scmd)),
         ga.low_signed(ga.op('vpsubd', sab, scd)), ga.low_signed(ga.op('vpsubd', samb, scmd))]
    g = [gb.quarter(i) for i in range(4)]
    ab, cd = gb.low(gb.add(g[0], g[1])), gb.low(gb.add(g[2], g[3]))
    amb, cmd = shoup(gb, gb.diff(g[0], g[1]), 'SY1'), shoup(gb, gb.diff(g[2], g[3]), 'SZ1')
    v = [shoup(gb, gb.add(ab, cd), 'S'), gb.low(gb.add(amb, cmd)),
         shoup(gb, gb.diff(ab, cd), 'SX1'), shoup(gb, gb.diff(amb, cmd), 'X1')]
    for t in range(4):
        if upper:
            if t < blocks:
                ga.store_quarter(ga.canonical(ga.diff(u[t], v[t])), t)
            continue
        if t < blocks:
            ga.store_quarter(ga.canonical(ga.add(u[t], v[t])), t)
        if 4 + t < blocks:
            gb.store_quarter(gb.canonical(gb.diff(u[t], v[t])), t)


def top_body(blocks=8, to=False, upper=False):
    """to: the first `blocks` output blocks to o (o2 for blocks 4..7), else all 8 in place.
    upper: the first `blocks` of output blocks 4..7 to o."""
    consts = {'P': gk.Value('creg', reg=15), '2P': gk.Value('creg', reg=14)}
    for i, name in enumerate(TOP_TWIDDLES):  # w[2i] value, w[2i + 1] quotient, broadcast
        consts['W' + name] = gk.Value('cmem', mem=64 * i)
        consts['Q' + name] = gk.Value('cmem', mem=64 * i + 32)
    ga = gk.Graph(consts, array='a', out='o' if to else None)
    gb = gk.Graph(consts, array='b', out='o2' if to else None)
    gb.ops = ga.ops  # one column: two address bases, one op list
    top_column(ga, gb, blocks, upper)
    seq = gk.schedule(gk.prune(ga.ops), 14, gk.Knobs())
    reg = gk.allocate(seq, {14, 15})
    head = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14', '.p2align 5', '1:']
    tail = ['add $32, %[a]', 'add $32, %[b]']
    if to:
        tail += ['add $32, %[o]'] + (['add $32, %[o2]'] if blocks > 4 and not upper else [])
    tail += ['cmp %[end], %[a]', 'jne 1b']
    return head + [gk.emit(o, reg) for o in seq] + tail


TOP_DOC = '''\
// Inverse top level of a transform of 8q vectors (avx2_kernels.hpp's inverse_identity, inverse and
// scale_radix2 in one pass): blocks a[tq, (t+1)q), t < 4, hold the identity group, t >= 4 group 1,
// values < 2P. Writes the canonical scaled result. w: twiddle vectors s, s z0, s y1, s z1, s x1,
// x1, each followed by its Shoup quotient.
inline void inverse_top(Vec* a, std::size_t q, const Vec* w) {
    Vec* b = a + 4 * q;
    Vec* const end = a + q;
    asm volatile('''


def top_function():
    lines = TOP_DOC.split('\n')
    lines += gk.asm_lines(top_body())
    lines += ['        : [a] "+r"(a), [b] "+r"(b)',
              '        : [end] "r"(end), [h] "r"(32 * q), [h3] "r"(96 * q), [w] "r"(w), [P] "m"(kP), [P2] "m"(k2P)',
              '        : ' + gk.CLOBBERS + ', "cc");', '}']
    return lines


TOP_TO_DOC = '''\
// inverse_top() for a product from spans: columns j < columns of the result's blocks t < T to the
// vectors at o + 8 (tq + j), any alignment.
template <int T>
void inverse_top_to(const Vec* a, std::uint32_t* o, std::size_t columns, std::size_t q, const Vec* w);
'''


def top_to_function(blocks, upper=False):
    name = 'inverse_top_upper_to' if upper else 'inverse_top_to'
    lines = ['template <>',
             f'inline void {name}<{blocks}>(const Vec* a, std::uint32_t* o, std::size_t columns, std::size_t q,',
             '                                  const Vec* w) {',
             '    const Vec* b = a + 4 * q;']
    two = blocks > 4 and not upper
    if two:
        lines += ['    std::uint32_t* o2 = o + 32 * q;']
    lines += ['    const Vec* const end = a + columns;', '    asm volatile(']
    lines += gk.asm_lines(top_body(blocks, True, upper))
    lines += ['        : [a] "+r"(a), [b] "+r"(b), [o] "+r"(o)' + (', [o2] "+r"(o2)' if two else ''),
              '        : [end] "r"(end), [h] "r"(32 * q), [h3] "r"(96 * q), [w] "r"(w), [P] "m"(kP), [P2] "m"(k2P)',
              '        : ' + gk.CLOBBERS + ', "cc");', '}']
    return lines


# Inverse top of a transform of 4q vectors (odd lg): the identity group and the scale s, folded into
# its twiddles s and s z, z = r^-1(1). Values < 2P in, canonical out; two columns per iteration:
#   out = s (f0 + f1) + s (f2 + f3), s (f0 - f1) + s z (f2 - f3), and the two differences.
TOP4_DOC = '''\
// Inverse top level of a transform of 4q vectors (avx2_kernels.hpp's inverse_identity and the scale
// in one pass): quarters a[tq, (t+1)q), t < 4, values < 2P; writes the canonical scaled result.
// w: twiddle vectors s, s z, each followed by its Shoup quotient. q even.
inline void inverse_top4(Vec* a, std::size_t q, const Vec* w) {
    Vec* const end = a + q;
    asm volatile('''

TOP4_TO_DOC = '''\
// inverse_top4() for a product from spans: columns j < columns of the result's quarters t < T to
// the vectors at o + 8 (tq + j), any alignment. columns even.
template <int T>
void inverse_top4_to(const Vec* a, std::uint32_t* o, std::size_t columns, std::size_t q, const Vec* w);
'''


def top4_column(g, blocks, upper=False):
    c = g.c
    shoup = lambda x, name: g.shoup(x, c['W' + name], c['Q' + name])
    f = [g.quarter(i) for i in range(4)]
    sab, samb = shoup(g.add(f[0], f[1]), 'S'), shoup(g.diff(f[0], f[1]), 'S')
    scd, scmd = shoup(g.add(f[2], f[3]), 'S'), shoup(g.diff(f[2], f[3]), 'SZ')
    outputs = [lambda: g.canonical(g.add(sab, scd)), lambda: g.canonical(g.add(samb, scmd)),
               lambda: gk.canonical_difference(g, sab, scd), lambda: gk.canonical_difference(g, samb, scmd)]
    for t in range(blocks):
        g.store_quarter(outputs[2 + t if upper else t](), t)


def top4_body(blocks=4, to=False, upper=False):
    """to: the first `blocks` output quarters to o, else all 4 in place. upper: of quarters 2, 3."""
    consts = {'P': gk.Value('creg', reg=15), '2P': gk.Value('creg', reg=14)}
    for i, name in enumerate(['S', 'SZ']):  # w[2i] value, w[2i + 1] quotient, broadcast
        consts['W' + name] = gk.Value('cmem', mem=64 * i)
        consts['Q' + name] = gk.Value('cmem', mem=64 * i + 32)
    ops = []
    for k in range(2):
        g = gk.Graph(consts, disp=32 * k, out='o' if to else None)
        top4_column(g, blocks, upper)
        ops += g.ops
    seq = gk.schedule(gk.prune(ops), 14, TOP4_KNOBS)
    reg = gk.allocate(seq, {14, 15})
    body = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14', '.p2align 5', '1:']
    tail = ['add $64, %[a]'] + (['add $64, %[o]'] if to else []) + ['cmp %[end], %[a]', 'jne 1b']
    return body + [gk.emit(o, reg) for o in seq] + tail


def top4_function():
    lines = TOP4_DOC.split('\n') + gk.asm_lines(top4_body())
    lines += ['        : [a] "+r"(a)',
              '        : [end] "r"(end), [h] "r"(32 * q), [h3] "r"(96 * q), [w] "r"(w), [P] "m"(kP), [P2] "m"(k2P)',
              '        : ' + gk.CLOBBERS + ', "cc");', '}']
    return lines


def top4_to_function(blocks, upper=False):
    name = 'inverse_top4_upper_to' if upper else 'inverse_top4_to'
    lines = ['template <>',
             f'inline void {name}<{blocks}>(const Vec* a, std::uint32_t* o, std::size_t columns, std::size_t q,',
             '                                   const Vec* w) {',
             '    const Vec* const end = a + columns;', '    asm volatile(']
    body = top4_body(blocks, True, upper)
    lines += gk.asm_lines(body)
    lines += ['        : [a] "+r"(a), [o] "+r"(o)',
              '        : [end] "r"(end), [h] "r"(32 * q), [h3] "r"(96 * q), [w] "r"(w), [P] "m"(kP), [P2] "m"(k2P)',
              '        : ' + gk.CLOBBERS + ', "cc");', '}']
    return lines


TOP4_KNOBS = gk.Knobs()

TOP_UPPER_TO_DOC = '''\
// inverse_top_to() for the upper half of the result (Transform's fused ops): columns j < columns of
// its blocks 4 + t, t < T, to the vectors at o + 8 (tq + j), any alignment.
template <int T>
void inverse_top_upper_to(const Vec* a, std::uint32_t* o, std::size_t columns, std::size_t q, const Vec* w);
'''

TOP4_UPPER_TO_DOC = '''\
// inverse_top4_to() for the upper half of the result: columns j < columns of its quarters 2 + t,
// t < T, to the vectors at o + 8 (tq + j), any alignment. columns even.
template <int T>
void inverse_top4_upper_to(const Vec* a, std::uint32_t* o, std::size_t columns, std::size_t q, const Vec* w);
'''


HEADER = '''\
// Generated by backend/ntt/gen_avx2_product_kernels.py. Do not edit.
// Kernels of the AVX2 product path (avx2.hpp). The bottom stage runs two radix-4 groups with h = 1
// per statement (avx2_kernels.hpp runs one). A batch of four vectors is a group; each output
// vector v_t is a leaf, a polynomial mod x^8 - w_t with w_t = y, -y, z, -z from the group's
// twiddles y = r[2k], z = r[2k + 1]. buf, cur: leaf buffers, 384 bytes per batch, 64-byte aligned.
// x, y, ix, iy: twiddles of the first batch, whose group index is even.
#pragma once

#include <cstddef>
#include <cstdint>

#include "backend/ntt/avx2_kernels.hpp"

namespace fps::ntt::avx2_product_kernels {

using avx2_kernels::k2P;
using avx2_kernels::kNI;
using avx2_kernels::kP;
using avx2_kernels::Vec;
'''


def generate():
    lines = HEADER.split('\n')
    for square in (False, True, 'stored', 'leaves', 'leaves2', 'leaves3'):
        for part in ('first', 'last', 'both'):
            lines += function(part, square) + ['']
    lines += top_function() + [''] + top4_function() + ['']
    lines += TOP_TO_DOC.split('\n')
    for blocks in range(1, 9):
        lines += top_to_function(blocks) + ['']
    lines += TOP4_TO_DOC.split('\n')
    for blocks in range(1, 5):
        lines += top4_to_function(blocks) + ['']
    lines += TOP_UPPER_TO_DOC.split('\n')
    for blocks in range(1, 5):
        lines += top_to_function(blocks, True) + ['']
    lines += TOP4_UPPER_TO_DOC.split('\n')
    for blocks in range(1, 3):
        lines += top4_to_function(blocks, True) + ['']
    lines += ['}  // namespace fps::ntt::avx2_product_kernels']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = generate()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f'{OUT.name} stale: run {Path(__file__).name}')
            return 1
        return 0
    OUT.write_text(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
