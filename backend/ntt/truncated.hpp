// Truncated products (issue #155 round 2): c = f g of N words, L < N < 2L, L = 2^lg, from a
// truncated transform of length 2L: the variants' group products on the nodes of the leaf form's
// first rr >= N words, about N words of products where the plain one runs 2L. ntt.hpp runs them
// (truncated_into) for multiply and square when ragged.hpp's plan is truncated, and middle products
// of N words (truncated_middle_into, round 3: Middle products below); this header has the shape and
// the levels above the group products.
//
//   shape(N)                          the levels (the partial nodes, the root first), the full nodes
//   forward(f, n, F, s, r)            f mod each full node, into F (the leaf form of length 2L, cut)
//   inverse(F, out, count, s, r, ir)  F's group products -> out = c[0, count), count <= N
//   co_forward(v, F, s, r, ir)        a middle product's v (N words) extended to p: p mod each full node
//   pruned_inverse(F, out, lo, hi, s, r, ir)  F's group products, zero past them -> out = q[lo, hi)
//
// Node k of s words is x^s - r(k)^2 (r: the twiddles of common.hpp); its halves are nodes 2k and 2k
// + 1, its quarters 4k + t, the root x^(2L) - 1 node 0. A node whose s / 8 is a power of 4 is the
// variants' group k with s / 8 leaves; nodes split into quarters where those are groups, else into
// halves (as the variants' trees: a radix-2 level, then radix-4 ones), so only the root has radix
// 2. The leaf form's first rr words (rr: N rounded up, ragged::rounded) are the full nodes; as c has
// no words past rr, its residues there give c (van der Hoeven's truncated transform, with the roots
// of unity of the transform of 2L, so no twist pass).
//
// Levels: a partial node of s words, its first n words unknown (0 < n < s), has children of c = s /
// radix words: t = n / c full ones (group products), then a partial child with n mod c unknown words
// if n mod c > 0, and none past it. Its forward writes those children. Its inverse runs on columns
// j < c: word j of each chunk p_u = p[u c, (u + 1) c) and of each child's residue R_v = sum_u rho_v^u
// p_u (rho = y, -y, z, -z with y = r(2k), z = r(2k + 1); radix 2: x, -x with x = r(k)). Step 1,
// top-down, on the columns j >= n mod c: p_u is known for u >= t (the root's are zero; under a
// radix-2 root, x^L + 1's are x^L - 1's, read in place), so the full children give p_u for u < t,
// and R_t, the partial child's known word j. Step 2, bottom-up after the partial child, on j < n mod
// c: p_u for u <= t from R_v, v <= t. The p_u go in place of the children's words (the root's to
// out); known words are not read again.
//
// Scales: unnormalized, as the variants' inverses: a group product of nv leaves is nv 2^-32 times
// its residue, a node's p comes out times its leaves 2^-32 too, and the root's scale goes into its
// level's constants. A node's known words are mu times its scale (mu, a power of 2, folded into its
// constants): radix when its parent had full children, else the parent's mu times radix (1 for x^L
// + 1's under a radix-2 root).
//
// Sources, read for the ideas: J. van der Hoeven, "The truncated Fourier transform and
// applications", ISSAC 2004; D. Harvey, "A cache-friendly truncated FFT", Theor. Comp. Sci. 410
// (2009); D. Harvey, D. S. Roche, "An in-place truncated Fourier transform and applications to
// polynomial multiplication", ISSAC 2010. Notes: notes/155-r2.md.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "backend/ntt/common.hpp"
#include "backend/ntt/intrinsics.hpp"
#include "backend/ntt/ragged.hpp"

namespace fps::ntt::detail::truncated {

using u32 = std::uint32_t;
using ragged::Constant;
using ragged::Mod;
using ragged::V;
using ragged::Vec;
using ragged::kLanes;

// A node: k in the transform of 2L, `size` words, at word `at` of its leaf form.
struct Node {
    std::size_t k, size, at;
};
// A partial node: its first n words unknown, 0 < n < size; children of size / radix words; mu: the
// scale of its known words over its own (above). sibling: those are the words of the node before
// it, x^L - 1's half of a radix-2 root (c has no words from L + n on: they agree there).
struct Level {
    Node node;
    std::size_t n;
    int radix;
    u32 mu;
    bool sibling = false;
};

inline constexpr int kMaxLevels = 12, kMaxFulls = 3 * kMaxLevels;

struct Shape {
    int lg = 0;                 // L = 2^lg, the transform 2^(lg + 1)
    std::size_t N = 0, rr = 0;  // the product's words, and those of the leaf form computed: N <= rr < 2L
    int levels = 0, fulls = 0;
    Level level[kMaxLevels];  // the root first
    Node full[kMaxFulls];     // the group products
};

// The shape of a product of N words (ragged::truncates(N)).
constexpr Shape shape(std::size_t N) {
    Shape s;
    s.lg = int(std::bit_width(N - 1)) - 1;
    s.N = N;
    s.rr = ragged::rounded(std::size_t(2) << s.lg, N);
    Node node{0, std::size_t(2) << s.lg, 0};
    u32 mu = 1;
    for (std::size_t n = s.rr;;) {
        const int radix = ragged::radix(node.size);
        const std::size_t c = node.size / std::size_t(radix), t = n / c, rest = n % c;
        const bool sibling = s.levels == 1 && s.level[0].radix == 2;
        if (sibling) mu = 1;
        s.level[s.levels++] = {node, n, radix, mu, sibling};
        for (std::size_t u = 0; u < t; ++u) s.full[s.fulls++] = {radix * node.k + u, c, node.at + u * c};
        if (!rest) break;
        node = {radix * node.k + t, c, node.at + t * c};
        mu = t ? u32(radix) : mu * u32(radix);
        n = rest;
    }
    return s;
}

// a +- b for a, b < 2P: [0, 2P) (add2, sub2), or < 4P, a Shoup product's input (add4, sub4).
[[gnu::always_inline]] inline V add2(V a, V b) { return ragged::add2(a, b); }
[[gnu::always_inline]] inline V sub2(V a, V b) { return ragged::sub2(a, b); }
[[gnu::always_inline]] inline V add4(V a, V b) { return Vec::add(a, b); }
[[gnu::always_inline]] inline V sub4(V a, V b) { return Mod::sub_lazy<2 * kP>(a, b); }

// Software prefetch on chunks of at least kPrefetchMin words (radix 4) or kPrefetchHalfMin (radix
// 2), kPrefetch words ahead, as the variants' passes (intrinsics.hpp: Emerald Rapids only).
template <int kRadix>
inline bool prefetches(std::size_t c) {
    using namespace intrinsics;
    return kPrefetch > 0 && c >= (kRadix == 4 ? kPrefetchMin : kPrefetchHalfMin);
}
[[gnu::always_inline]] inline void ahead(const u32* p) { __builtin_prefetch(p + intrinsics::kPrefetch, 0, 3); }
[[gnu::always_inline]] inline void ahead_write(const u32* p) {
#if defined(__x86_64__)
    asm volatile("prefetchw %0" : : "m"(*reinterpret_cast<const char*>(p + intrinsics::kPrefetch)));
#else
    __builtin_prefetch(p + intrinsics::kPrefetch, 1, 3);
#endif
}

// ---- Forward

// A node's twiddles: x = r(k), y = r(2k), z = r(2k + 1) (radix 2: x).
struct Twiddles {
    Constant x, y, z;
    Twiddles(const u32* r, std::size_t k, int radix)
        : x(Constant::entry(r, k)), y(Constant::entry(r, radix == 4 ? 2 * k : k)),
          z(Constant::entry(r, radix == 4 ? 2 * k + 1 : k)) {}
};

// Children t <= kLast of a node at column j from chunks u < kLoad of src (the others zero, not
// read), c words apart; kEdge: each chunk read to len words. kRoot: k = 0 (x = y = 1). In and out
// < 2P, to dst + t c.
template <int kRadix, int kLoad, int kLast, bool kEdge, bool kRoot>
[[gnu::always_inline]] inline void forward_column(const u32* src, std::size_t c, std::size_t len, u32* dst,
                                                  std::size_t j, const Twiddles& w) {
    const V zero = Vec::broadcast(0);
    const auto load = [&]<int u>() -> V {
        if constexpr (u >= kLoad) return zero;
        else if constexpr (kEdge) return ragged::load_upto(src, std::size_t(u) * c + j, len);
        else return Vec::load(src + std::size_t(u) * c + j);
    };
    const auto store = [&](int t, V v) { Vec::store(dst + std::size_t(t) * c + j, v); };
    const auto times = [&](const Constant& k, V v) { return kRoot ? v : k.times(v); };  // by x or y
    if constexpr (kRadix == 2) {
        const V a = load.template operator()<0>();
        if constexpr (kLoad < 2) {
            store(0, a);
            if constexpr (kLast >= 1) store(1, a);
        } else {
            const V b = times(w.x, load.template operator()<1>());
            store(0, add2(a, b));
            if constexpr (kLast >= 1) store(1, sub2(a, b));
        }
    } else {
        const V q0 = load.template operator()<0>(), q1 = load.template operator()<1>();
        V e = q0, g = q0, f = q1, h = q1;  // E, G = q0 +- x q2; F, H = q1 +- x q3
        if constexpr (kLoad > 2) {
            const V xq2 = times(w.x, load.template operator()<2>());
            e = add2(q0, xq2), g = sub2(q0, xq2);
        }
        if constexpr (kLoad > 3) {
            const V xq3 = times(w.x, load.template operator()<3>());
            f = add2(q1, xq3), h = sub2(q1, xq3);
        }
        if constexpr (kLoad > 1) {
            const V yf = times(w.y, f);
            store(0, add2(e, yf));
            if constexpr (kLast >= 1) store(1, sub2(e, yf));
            if constexpr (kLast >= 2) {
                const V zh = w.z.times(h);
                store(2, add2(g, zh));
                if constexpr (kLast >= 3) store(3, sub2(g, zh));
            }
        } else {
            store(0, e);
            if constexpr (kLast >= 1) store(1, e);
            if constexpr (kLast >= 2) store(2, g);
            if constexpr (kLast >= 3) store(3, g);
        }
    }
}

template <int kRadix, int kLoad, int kLast, bool kEdge, bool kRoot>
[[gnu::noinline]] void forward_columns(const u32* src, std::size_t c, std::size_t len, u32* dst, std::size_t j0,
                                       std::size_t j1, const Twiddles& w) {
    if (!kEdge && prefetches<kRadix>(c)) {
        for (std::size_t j = j0; j < j1; j += kLanes) {
            if (j % 16 == 0) {  // a line per chunk
                for (int u = 0; u < kLoad; ++u) ahead(src + std::size_t(u) * c + j);
                for (int t = src == dst ? kLoad : 0; t <= kLast; ++t) ahead_write(dst + std::size_t(t) * c + j);
            }
            forward_column<kRadix, kLoad, kLast, kEdge, kRoot>(src, c, len, dst, j, w);
        }
        return;
    }
    for (std::size_t j = j0; j < j1; j += kLanes)
        forward_column<kRadix, kLoad, kLast, kEdge, kRoot>(src, c, len, dst, j, w);
}

// One level's forward: the node's residue from src (its first len <= size words, zero past them),
// children t <= last to dst + t c.
template <int kRadix, bool kRoot>
void forward_level(const u32* src, std::size_t len, std::size_t c, int last, u32* dst, const Twiddles& w) {
    const std::size_t whole = len / c, e = len % c, e0 = e / kLanes * kLanes, e1 = e0 == e ? e : e0 + kLanes;
    with_blocks<kRadix - 1>(last, [&](auto t) {
        constexpr int kLast = decltype(t)::value;
        // Columns [0, e0) read whole + 1 chunks, the one at e0 (if e is inside it) to len, the rest whole.
        if (e0) with_blocks<kRadix>(int(whole) + 1, [&](auto load) {
            forward_columns<kRadix, decltype(load)::value, kLast, false, kRoot>(src, c, len, dst, 0, e0, w);
        });
        if (e1 > e0) forward_columns<kRadix, kRadix, kLast, true, kRoot>(src, c, len, dst, e0, e1, w);
        with_blocks<kRadix>(int(whole), [&](auto load) {
            forward_columns<kRadix, decltype(load)::value, kLast, false, kRoot>(src, c, len, dst, e1, c, w);
        });
    });
}

// f (n < 2L words, canonical) mod each full node of s into F, the leaf form of length 2L (2L words
// kept). r: the forward table of 2L. Outputs < 2P (group products take < 4P).
inline void forward(const u32* f, std::size_t n, u32* F, const Shape& s, const u32* r) {
    // A radix-2 root and n <= L: f is its own residue mod x^L -+ 1, so the root writes x^L - 1's half
    // alone and the next level reads f: no copy of f written and read back.
    const bool direct = s.level[0].radix == 2 && s.levels > 1 && n <= s.level[1].node.size;
    for (int d = 0; d < s.levels; ++d) {
        const Level& l = s.level[d];
        const std::size_t c = l.node.size / std::size_t(l.radix), t = l.n / c;
        const int last = int(l.n % c ? t : t - 1);
        u32* const at = F + l.node.at;
        const Twiddles w(r, l.node.k, l.radix);
        if (d == 1 && direct) forward_level<4, false>(f, n, c, last, at, w);
        else if (d > 0) forward_level<4, false>(at, l.node.size, c, last, at, w);
        else if (l.radix == 4) forward_level<4, true>(f, n, c, last, at, w);
        else forward_level<2, true>(f, n, c, direct ? 0 : last, at, w);
    }
}

// ---- Inverse

// A level's constants (names: what they multiply by). The root's: sc = 1 / its scale, z = r(1).
struct Constants {
    Constant x, y, z, x_mu, y_mu, two_y, half_z, inv_x, inv_y, inv_z, sc, sc_z;
    Constants() = default;  // unset
    Constants(const u32* r, const u32* ir, const Level& l, int lg2) {
        const std::size_t k = l.node.k;
        constexpr u32 half = (kP + 1) / 2;
        u32 inv_mu = 1;  // a power of 2
        for (u32 m = l.mu; m > 1; m /= 2) inv_mu = multiply_mod(inv_mu, half);
        const bool root = k == 0;
        if (l.radix == 4) {
            z = Constant::entry(r, 2 * k + 1), half_z = Constant(multiply_mod(half, r[slot(2 * k + 1)]));
            if (root) {
                sc_z = Constant(multiply_mod(product_scale(lg2), ir[slot(2 * k + 1)]));
            } else {
                x = Constant::entry(r, k), y = Constant::entry(r, 2 * k);
                x_mu = Constant(multiply_mod(r[slot(k)], inv_mu));
                y_mu = Constant(multiply_mod(r[slot(2 * k)], inv_mu));
                two_y = Constant(multiply_mod(2, ir[slot(2 * k)]));
                inv_x = Constant::entry(ir, k), inv_y = Constant::entry(ir, 2 * k);
                inv_z = Constant::entry(ir, 2 * k + 1);
            }
        }
        if (root) sc = Constant(product_scale(lg2));
    }
};

// The root's words: c[at, at + kLanes), canonical; kCut: cut to count words (count <= N).
struct Out {
    u32* c;
    std::size_t count;
    template <bool kCut>
    [[gnu::always_inline]] void put(std::size_t at, V v) const {
        v = Mod::reduce<kP>(v);
        if (!kCut || at + kLanes <= count) Vec::store(c + at, v);
        else if (at < count) Vec::store_first(c + at, v, int(count - at));
    }
};

// One column j of a level's inverse, kU of its children's residues known (R_v, v < kU, at X + v c +
// j; the last one the partial child's in step 2, at the same scale) and p_u for u >= kU (K_u, at
// known + u c + j; the root's zero): p_u for u < kU (P_u; the root's c[u c + j]), and with kNext R_kU
// (Q) to X + kU c + j, at the scale of the node's words (mu 1). Radix 4, true values: E, G = p0 +- x
// p2; F, H = p1 +- x p3; R0, R1 = E +- y F; R2, R3 = G +- z H. Radix 2: R0, R1 = p0 +- x p1. With o
// the node's scale, R_v = o / radix times its true value and K_u = mu o times p_u. kCut: the root's
// words past out.count are none (0), in chunk kU - 1 (1) or anywhere (2).
template <int kRadix, int kU, bool kNext, bool kRoot, int kCut = 0>
    requires(kRoot || kRadix == 4)
[[gnu::always_inline]] inline void inverse_column(u32* X, const u32* known, std::size_t c, std::size_t j,
                                                  const Constants& k, const Out& out) {
    const auto R = [&](int v) { return Vec::load(X + std::size_t(v) * c + j); };
    const auto K = [&](int u) { return Vec::load(known + std::size_t(u) * c + j); };
    const auto P = [&](int u, V v) {
        if constexpr (kRoot) {
            if (kCut == 2 || (kCut == 1 && u == kU - 1)) out.put<true>(std::size_t(u) * c + j, v);
            else out.put<false>(std::size_t(u) * c + j, v);
        } else {
            Vec::store(X + std::size_t(u) * c + j, v);
        }
    };
    const auto Q = [&](V v) { Vec::store(X + std::size_t(kU) * c + j, v); };
    const auto twice = [](V v) { return add2(v, v); };
    if constexpr (kRoot) {  // k = 0: x = y = 1, z = r(1); K = 0; outputs times sc
        if constexpr (kRadix == 2) {
            if constexpr (kU == 1) {  // no Q: x^L + 1's known words are x^L - 1's (Level::sibling)
                P(0, k.sc.times(twice(R(0))));
            } else {
                const V r0 = R(0), r1 = R(1);
                P(0, k.sc.times(add4(r0, r1))), P(1, k.sc.times(sub4(r0, r1)));
            }
        } else if constexpr (kU == 2) {
            const V r0 = R(0), r1 = R(1), e = twice(add2(r0, r1)), f = twice(sub2(r0, r1));
            P(0, k.sc.times(e)), P(1, k.sc.times(f));
            if constexpr (kNext) Q(add2(e, k.z.times(f)));
        } else if constexpr (kU == 3) {
            const V r0 = R(0), r1 = R(1), r2 = twice(R(2)), s01 = add2(r0, r1), f = twice(sub2(r0, r1));
            const V zh = k.half_z.times(f);
            P(0, k.sc.times(sub4(add2(s01, r2), zh))), P(1, k.sc.times(f)), P(2, k.sc.times(add4(sub2(s01, r2), zh)));
            if constexpr (kNext) Q(sub2(twice(r2), twice(twice(zh))));
        } else {
            const V r0 = R(0), r1 = R(1), r2 = R(2), r3 = R(3), s01 = add2(r0, r1), s23 = add2(r2, r3);
            const V a = k.sc.times(sub4(r0, r1)), b = k.sc_z.times(sub4(r2, r3));
            P(0, k.sc.times(add4(s01, s23))), P(2, k.sc.times(sub4(s01, s23)));
            P(1, add2(a, b)), P(3, sub2(a, b));
        }
    } else if constexpr (kU == 0) {  // below the root, radix 4 (shape)
        Q(add2(add2(K(0), k.x.times(K(2))), k.y.times(add2(K(1), k.x.times(K(3))))));
    } else if constexpr (kU == 1) {
        const V yf = k.y_mu.times(add2(K(1), k.x.times(K(3)))), r0 = twice(twice(R(0)));
        P(0, sub2(sub2(r0, yf), k.x_mu.times(K(2))));
        if constexpr (kNext) Q(sub2(r0, twice(yf)));
    } else if constexpr (kU == 2) {
        const V r0 = R(0), r1 = R(1), e = twice(add2(r0, r1)), f = k.two_y.times(sub4(r0, r1));
        const V xk2 = k.x_mu.times(K(2)), xk3 = k.x_mu.times(K(3)), p0 = sub2(e, xk2), p1 = sub2(f, xk3);
        P(0, p0), P(1, p1);
        if constexpr (kNext) Q(add2(sub2(p0, xk2), k.z.times(sub2(p1, xk3))));
    } else if constexpr (kU == 3) {
        const V r0 = R(0), r1 = R(1), r2 = twice(R(2)), s01 = add2(r0, r1), f = k.two_y.times(sub4(r0, r1));
        const V xk3 = k.x_mu.times(K(3)), p1 = sub2(f, xk3), zh = k.half_z.times(sub2(p1, xk3));
        P(0, sub2(add2(s01, r2), zh)), P(1, p1), P(2, k.inv_x.times(add4(sub2(s01, r2), zh)));
        if constexpr (kNext) Q(sub2(twice(r2), twice(twice(zh))));
    } else {
        const V r0 = R(0), r1 = R(1), r2 = R(2), r3 = R(3), s01 = add2(r0, r1), s23 = add2(r2, r3);
        const V a = k.inv_y.times(sub4(r0, r1)), b = k.inv_z.times(sub4(r2, r3));
        P(0, add2(s01, s23)), P(1, add2(a, b)), P(2, k.inv_x.times(sub4(s01, s23))), P(3, k.inv_x.times(sub4(a, b)));
    }
}

template <int kRadix, int kU, bool kNext, bool kRoot, int kCut = 0>
    requires(kRoot || kRadix == 4)
[[gnu::noinline]] void inverse_columns(u32* X, const u32* known, std::size_t c, std::size_t j0, std::size_t j1,
                                       const Constants& k, const Out& out) {
    if (prefetches<kRadix>(c)) {
        for (std::size_t j = j0; j < j1; j += kLanes) {
            if (j % 16 == 0) {  // a line per chunk read, and per chunk of out written
                for (int u = 0; u < kRadix; ++u) ahead(X + std::size_t(u) * c + j);
                if constexpr (kRoot)
                    for (int u = 0; u < kU; ++u) ahead_write(out.c + std::size_t(u) * c + j);
            }
            inverse_column<kRadix, kU, kNext, kRoot, kCut>(X, known, c, j, k, out);
        }
        return;
    }
    for (std::size_t j = j0; j < j1; j += kLanes)
        inverse_column<kRadix, kU, kNext, kRoot, kCut>(X, known, c, j, k, out);
}

// Step 1 (top-down) or step 2 (bottom-up, second) of a level; N: the product's words.
template <int kRadix, bool kRoot>
void inverse_step(const Level& l, bool second, u32* X, const u32* known, std::size_t N, const Constants& k,
                  const Out& out) {
    const std::size_t c = l.node.size / std::size_t(kRadix), t = l.n / c, rest = l.n % c;
    with_blocks<kRadix - 1>(int(t), [&](auto u) {
        constexpr int kT = decltype(u)::value;
        // The root's last chunk written, t in step 2 or t - 1 with no partial child, may pass N:
        // columns from N - (its chunk) c on are cut; N > t c if rest > 0, else N > (t - 1) c. A cut
        // product (out.count < N) checks every word.
        const auto run = [&]<int kU, bool kNext>(std::size_t j0, std::size_t j1) {
            if constexpr (kRoot) {
                if (out.count < N) return inverse_columns<kRadix, kU, kNext, kRoot, 2>(X, known, c, j0, j1, k, out);
                const std::size_t cut = kNext ? j1 : std::clamp((N - (kU - 1) * c) / kLanes * kLanes, j0, j1);
                inverse_columns<kRadix, kU, kNext, kRoot>(X, known, c, j0, cut, k, out);
                inverse_columns<kRadix, kU, kNext, kRoot, 1>(X, known, c, cut, j1, k, out);
            } else {
                inverse_columns<kRadix, kU, kNext, kRoot>(X, known, c, j0, j1, k, out);
            }
        };
        if constexpr (!kRoot || 2 * kT >= kRadix) {  // the root has t >= radix / 2 (N > L)
            if (second) run.template operator()<kT + 1, false>(0, rest);
            else if (rest) run.template operator()<kT, true>(rest, c);
            else if constexpr (kT > 0) run.template operator()<kT, false>(0, c);
        }
    });
}

// out = c[0, count), count <= N, from F, the leaf form after the group products of s's full nodes
// (< 2P, times their scale): the levels' steps 1, then their steps 2. F is overwritten. r, ir: the
// tables of 2L.
inline void inverse(u32* F, u32* out, std::size_t count, const Shape& s, const u32* r, const u32* ir) {
    const Out o{out, count};
    Constants constants[kMaxLevels];
    const auto step = [&](int d, bool second) {
        const Level& l = s.level[d];
        if (!second) constants[d] = Constants(r, ir, l, s.lg + 1);
        const Constants& k = constants[d];
        u32* const X = F + l.node.at;
        const u32* known = l.sibling ? X - l.node.size : X;
        if (d > 0) inverse_step<4, false>(l, second, X, known, s.N, k, o);
        else if (l.radix == 4) inverse_step<4, true>(l, second, X, X, s.N, k, o);
        else inverse_step<2, true>(l, second, X, X, s.N, k, o);
    };
    for (int d = 0; d < s.levels; ++d) step(d, false);
    for (int d = s.levels - 1; d >= 0; --d)
        if (s.level[d].n % (s.level[d].node.size / std::size_t(s.level[d].radix))) step(d, true);
}

// ---- Middle products (round 3)
//
// out[i] = (v g)[m - 1 + i] for i < count, v of N = m - 1 + count words (ntt::middle_product). These
// words of the cyclic product p g mod (x^(2L) - 1) read only p[0, N), for any p (no term wraps). So
// take p = v on [0, rr) (zero from N on) and its words [rr, 2L) such that p = 0 mod each node past
// the leaf form's first rr words (S': at each level the children past the partial one; 2L - rr words
// in all, as many conditions as free words: p = v + x^rr h, h = -v x^-rr mod their product). Then q
// = p g mod (x^(2L) - 1) is 0 mod those nodes too, so it follows from its residues at the full nodes
// alone: co_forward gives p's there, the group products (with g's, from forward) q's, and
// pruned_inverse q's words [m - 1, N), an inverse whose inputs past the full nodes are zero. This
// costs about what the truncated product of N words does, with the variants' group products as they
// are (the transposed truncated product, Tellegen's form, would need transposed ones).
//
// co_forward per level: a partial node of s words, chunks p_u of c words, its first n words known
// (the root's: v), t = n / c full children, a partial child t with rest = n mod c known words if
// rest > 0, the children past it zero. Column j: P(X) = sum_u p_u[j] X^u, R_v = P(lambda_v) (lambda
// = y, -y, z, -z as Level's comment). Step 1, top-down, on j < rest (all j, with t - 1, if rest = 0:
// child t - 1 is full): p_u known for u <= t and P = 0 at lambda_v for v > t give p_u for u > t (the
// node's free words) and R_v for v <= t (R_t: the partial child's known word j). Step 2, bottom-up,
// after the partial child, on j >= rest: p_u known for u < t, R_t (the child's free word j) and the
// zeros give p_u for u >= t and R_v for v < t. Free words go in place of the zero children's and of
// the partial child's own free words, where the parent's step 2 reads them; the root's are not
// written. True scale, words in [0, 2P).
//
// pruned_inverse, bottom-up: a node's words from its children's, zero past the partial child, by the
// variants' unnormalized inverse: a node's words come out times its leaves 2^-32, the scale of a
// group product of its size, so the levels need no rescaling; the root's carry 1 / its scale.

// A level's constants (names: what they multiply by): node k's x = r(k), y = r(2k), z = r(2k + 1),
// their inverses and products; the root's sc (1 / its scale) and sc z^-1. lg2: the transform's log.
struct MiddleConstants {
    Constant x, y, z, inv_x, inv_y, inv_z, inv_xy, half, inv_2y, quarter, quarter_x, quarter_y, quarter_xy, sc, sc_z;
    MiddleConstants() = default;  // unset
    MiddleConstants(const u32* r, const u32* ir, std::size_t k, int lg2) {
        constexpr u32 h = (kP + 1) / 2, q = u32(std::uint64_t(h) * h % kP);
        const u32 ix = ir[slot(k)], iy = ir[slot(2 * k)], ixy = multiply_mod(ix, iy);
        x = Constant::entry(r, k), y = Constant::entry(r, 2 * k), z = Constant::entry(r, 2 * k + 1);
        inv_x = Constant::entry(ir, k), inv_y = Constant::entry(ir, 2 * k), inv_z = Constant::entry(ir, 2 * k + 1);
        inv_xy = Constant(ixy), half = Constant(h), inv_2y = Constant(multiply_mod(h, iy)), quarter = Constant(q);
        quarter_x = Constant(multiply_mod(q, ix)), quarter_y = Constant(multiply_mod(q, iy));
        quarter_xy = Constant(multiply_mod(q, ixy));
        sc = Constant(product_scale(lg2)), sc_z = Constant(multiply_mod(product_scale(lg2), ir[slot(1)]));
    }
};

// A product by x, y or their inverses, which are 1 at the root: there a is returned, < 2P. The sums
// that feed them: lazy (< 4P) below the root, reduced (< 2P) at it.
template <bool kRoot>
[[gnu::always_inline]] inline V by(const Constant& w, V a) {
    if constexpr (kRoot) return a;
    else return w.times(a);
}
template <bool kRoot>
[[gnu::always_inline]] inline V lazy_add(V a, V b) {
    if constexpr (kRoot) return add2(a, b);
    else return add4(a, b);
}
template <bool kRoot>
[[gnu::always_inline]] inline V lazy_sub(V a, V b) {
    if constexpr (kRoot) return sub2(a, b);
    else return sub4(a, b);
}
[[gnu::always_inline]] inline V twice(V a) { return add2(a, a); }

// Column j of a radix-4 level's co_forward (above): p(u) is p_u's word j (u <= kT in step 1, u < kT
// in step 2), X + v c + j gets R_v (v <= kT in step 1, v < kT in step 2; R_kT is read there in step
// 2) and, below the root, the free p_u (u > kT in step 1, u >= kT in step 2). kRoot: node 0 (x = y =
// 1, z = r(1)), kT >= 2 there (rr > L: t = 2 with rest > 0, or 3). In [0, 2P) from [0, 2P).
template <bool kSecond, int kT, bool kRoot, class Load>
[[gnu::always_inline]] inline void co_column4(Load p, u32* X, std::size_t c, std::size_t j, const MiddleConstants& k) {
    const auto R = [&](int v) { return Vec::load(X + std::size_t(v) * c + j); };
    const auto put = [&](int u, V v) { Vec::store(X + std::size_t(u) * c + j, v); };
    if constexpr (!kSecond && kT == 3) {  // the plain forward
        const V q0 = p(0), q1 = p(1), xq2 = by<kRoot>(k.x, p(2)), xq3 = by<kRoot>(k.x, p(3));
        const V e = add2(q0, xq2), g = sub2(q0, xq2), f = by<kRoot>(k.y, lazy_add<kRoot>(q1, xq3)), h = k.z.times(sub4(q1, xq3));
        put(0, add2(e, f)), put(1, sub2(e, f)), put(2, add2(g, h)), put(3, sub2(g, h));
    } else if constexpr (!kSecond && kT == 2) {  // R3 = 0: z H = G, so R2 = 2G and F = 2 p1 - G / z
        const V q0 = p(0), q1 = p(1), xq2 = by<kRoot>(k.x, p(2));
        const V e = add2(q0, xq2), g = sub2(q0, xq2), gz = k.inv_z.times(g);
        const V f = by<kRoot>(k.y, lazy_sub<kRoot>(twice(q1), gz));
        put(0, add2(e, f)), put(1, sub2(e, f)), put(2, twice(g));
        if constexpr (!kRoot) put(3, k.inv_x.times(sub4(q1, gz)));
    } else if constexpr (!kSecond && kT == 1) {  // R2 = R3 = 0: p2 = p0 / x, p3 = p1 / x
        static_assert(!kRoot);
        const V q0 = p(0), q1 = p(1), yq1 = k.y.times(q1);
        put(0, twice(add2(q0, yq1))), put(1, twice(sub2(q0, yq1))), put(2, k.inv_x.times(q0)), put(3, k.inv_x.times(q1));
    } else if constexpr (!kSecond) {  // kT = 0: P = p0 (X + y)(X^2 + x) / (x y)
        static_assert(!kRoot);
        const V q0 = p(0);
        put(0, twice(twice(q0))), put(1, k.inv_y.times(q0)), put(2, k.inv_x.times(q0)), put(3, k.inv_xy.times(q0));
    } else if constexpr (kT == 0) {  // P = R0 (X + y)(X^2 + x) / (4 x y)
        static_assert(!kRoot);
        const V r0 = R(0);
        put(0, k.quarter.times(r0)), put(1, k.quarter_y.times(r0)), put(2, k.quarter_x.times(r0));
        put(3, k.quarter_xy.times(r0));
    } else if constexpr (kT == 1) {  // R2 = R3 = 0: R0 + R1 = 4 p0, y p1 = p0 - R1 / 2
        static_assert(!kRoot);
        const V q0 = p(0), r1 = R(1), q1 = sub2(k.inv_y.times(q0), k.inv_2y.times(r1));
        put(0, sub2(twice(twice(q0)), r1)), put(1, q1), put(2, k.inv_x.times(q0)), put(3, k.inv_x.times(q1));
    } else if constexpr (kT == 2) {  // R3 = 0: G = R2 / 2, H = G / z
        const V q0 = p(0), q1 = p(1), g = k.half.times(R(2)), h = k.inv_z.times(g);
        const V e = sub2(twice(q0), g), f = by<kRoot>(k.y, lazy_sub<kRoot>(twice(q1), h));
        put(0, add2(e, f)), put(1, sub2(e, f));
        if constexpr (!kRoot) put(2, k.inv_x.times(sub4(q0, g))), put(3, k.inv_x.times(sub4(q1, h)));
    } else {  // kT = 3: H = (G - R3) / z
        const V q0 = p(0), q1 = p(1), xq2 = by<kRoot>(k.x, p(2)), r3 = R(3);
        const V e = add2(q0, xq2), g = sub2(q0, xq2), d = k.inv_z.times(sub4(r3, g));
        const V f = by<kRoot>(k.y, lazy_add<kRoot>(twice(q1), d));
        put(0, add2(e, f)), put(1, sub2(e, f)), put(2, sub2(twice(g), r3));
        if constexpr (!kRoot) put(3, k.inv_x.times(add4(q1, d)));
    }
}

// The same for a radix-2 root (x^L - 1, x^L + 1; t = 1): R0, R1 = p0 +- p1; step 2: R0 = 2 p0 - R1.
template <bool kSecond, class Load>
[[gnu::always_inline]] inline void co_column2(Load p, u32* X, std::size_t c, std::size_t j) {
    if constexpr (!kSecond) {
        const V q0 = p(0), q1 = p(1);
        Vec::store(X + j, add2(q0, q1)), Vec::store(X + c + j, sub2(q0, q1));
    } else {
        Vec::store(X + j, sub2(twice(p(0)), Vec::load(X + c + j)));
    }
}

// Columns [j0, j1) of a level below the root, in place (X: its words).
template <bool kSecond, int kT>
[[gnu::noinline]] void co_columns(u32* X, std::size_t c, std::size_t j0, std::size_t j1, const MiddleConstants& k) {
    const auto column = [&] [[gnu::always_inline]] (std::size_t j) {
        const auto p = [&] [[gnu::always_inline]] (int u) { return Vec::load(X + std::size_t(u) * c + j); };
        co_column4<kSecond, kT, false>(p, X, c, j, k);
    };
    if (prefetches<4>(c)) {
        for (std::size_t j = j0; j < j1; j += kLanes) {
            if (j % 16 == 0)
                for (int u = 0; u < 4; ++u) ahead_write(X + std::size_t(u) * c + j);
            column(j);
        }
        return;
    }
    for (std::size_t j = j0; j < j1; j += kLanes) column(j);
}

// Root columns [j0, j1): p_u from v (N words, zero past them), chunks u < kLoad read whole, or each
// to N (kEdge); the children to X.
template <int kRadix, bool kSecond, int kT, int kLoad, bool kEdge>
[[gnu::noinline]] void co_root_columns(const u32* v, std::size_t N, u32* X, std::size_t c, std::size_t j0, std::size_t j1,
                                       const MiddleConstants& k) {
    const auto column = [&] [[gnu::always_inline]] (std::size_t j) {
        const auto p = [&] [[gnu::always_inline]] (int u) {
            const std::size_t at = std::size_t(u) * c + j;
            if constexpr (kEdge) return ragged::load_upto(v, at, N);
            else return u < kLoad ? Vec::load(v + at) : Vec::broadcast(0);
        };
        if constexpr (kRadix == 2) co_column2<kSecond>(p, X, c, j);
        else co_column4<kSecond, kT, true>(p, X, c, j, k);
    };
    if (!kEdge && prefetches<kRadix>(c)) {
        for (std::size_t j = j0; j < j1; j += kLanes) {
            if (j % 16 == 0) {  // a line per chunk read and written
                for (int u = 0; u < kLoad; ++u) ahead(v + std::size_t(u) * c + j);
                for (int u = 0; u < kRadix; ++u) ahead_write(X + std::size_t(u) * c + j);
            }
            column(j);
        }
        return;
    }
    for (std::size_t j = j0; j < j1; j += kLanes) column(j);
}

// Root columns [j0, j1): N = whole c + e words of v; columns below e read whole + 1 chunks, the one
// holding e reads each to N, the rest whole.
template <int kRadix, bool kSecond, int kT>
void co_root(const u32* v, std::size_t N, u32* X, std::size_t c, std::size_t j0, std::size_t j1, const MiddleConstants& k) {
    const std::size_t whole = N / c, e = N % c, e0 = e / kLanes * kLanes, e1 = e0 == e ? e : e0 + kLanes;
    const auto part = [&](std::size_t a, std::size_t b, int load) {
        a = std::max(a, j0), b = std::min(b, j1);
        if (a >= b) return;
        if (load < 0) return co_root_columns<kRadix, kSecond, kT, kRadix, true>(v, N, X, c, a, b, k);
        with_blocks<kRadix>(load, [&](auto blocks) {
            co_root_columns<kRadix, kSecond, kT, decltype(blocks)::value, false>(v, N, X, c, a, b, k);
        });
    };
    part(0, e0, int(whole) + 1);
    part(e0, e1, -1);
    part(e1, c, int(whole));
}

// p's residues at s's full nodes into F (the leaf form of length 2L, cut; 2L words kept), from v of
// s.N words (canonical): above. r, ir: the tables of 2L. Outputs < 2P (group products take < 4P).
inline void co_forward(const u32* v, u32* F, const Shape& s, const u32* r, const u32* ir) {
    MiddleConstants constants[kMaxLevels];
    const auto step = [&](int d, bool second) {
        const Level& l = s.level[d];
        const std::size_t c = l.node.size / std::size_t(l.radix), t = l.n / c, rest = l.n % c;
        const std::size_t j0 = second ? rest : 0, j1 = second || !rest ? c : rest;
        const int kt = int(rest || second ? t : t - 1);
        u32* const X = F + l.node.at;
        const MiddleConstants& k = constants[d];
        if (d == 0 && l.radix == 2) {  // t = 1, rest > 0
            if (second) co_root<2, true, 1>(v, s.N, X, c, j0, j1, k);
            else co_root<2, false, 1>(v, s.N, X, c, j0, j1, k);
            return;
        }
        with_blocks<3>(kt, [&](auto u) {
            constexpr int kT = decltype(u)::value;
            if (d > 0) {
                if (second) co_columns<true, kT>(X, c, j0, j1, k);
                else co_columns<false, kT>(X, c, j0, j1, k);
            } else if constexpr (kT >= 2) {  // a radix-4 root: kt >= 2 (rr > L)
                if (second) co_root<4, true, kT>(v, s.N, X, c, j0, j1, k);
                else co_root<4, false, kT>(v, s.N, X, c, j0, j1, k);
            }
        });
    };
    for (int d = 0; d < s.levels; ++d) {
        constants[d] = MiddleConstants(r, ir, s.level[d].node.k, s.lg + 1);
        step(d, false);
    }
    for (int d = s.levels - 1; d >= 0; --d)
        if (s.level[d].n % (s.level[d].node.size / std::size_t(s.level[d].radix))) step(d, true);
}

// Column j of a radix-4 level's pruned inverse below the root: R_v (v <= kT; zero past) at X + v c +
// j -> the node's words p_u at X + u c + j, unnormalized. [0, 2P) from [0, 2P).
template <int kT>
[[gnu::always_inline]] inline void pi_column4(u32* X, std::size_t c, std::size_t j, const MiddleConstants& k) {
    const auto R = [&](int v) { return Vec::load(X + std::size_t(v) * c + j); };
    const auto put = [&](int u, V v) { Vec::store(X + std::size_t(u) * c + j, v); };
    if constexpr (kT == 0) {  // p0 = R0 in place
        const V r0 = R(0);
        put(1, k.inv_y.times(r0)), put(2, k.inv_x.times(r0)), put(3, k.inv_xy.times(r0));
    } else {
        const V r0 = R(0), r1 = R(1), e = add2(r0, r1), f = k.inv_y.times(sub4(r0, r1));
        if constexpr (kT == 1) {
            put(0, e), put(1, f), put(2, k.inv_x.times(e)), put(3, k.inv_x.times(f));
        } else {
            V g = R(2), h;
            if constexpr (kT == 3) {
                const V r2 = g, r3 = R(3);
                g = add2(r2, r3), h = k.inv_z.times(sub4(r2, r3));
            } else {
                h = k.inv_z.times(g);
            }
            put(0, add2(e, g)), put(1, add2(f, h)), put(2, k.inv_x.times(sub4(e, g))), put(3, k.inv_x.times(sub4(f, h)));
        }
    }
}

template <int kT>
[[gnu::noinline]] void pi_columns(u32* X, std::size_t c, const MiddleConstants& k) {
    if (prefetches<4>(c)) {
        for (std::size_t j = 0; j < c; j += kLanes) {
            if (j % 16 == 0)
                for (int u = 0; u < 4; ++u) ahead_write(X + std::size_t(u) * c + j);
            pi_column4<kT>(X, c, j, k);
        }
        return;
    }
    for (std::size_t j = 0; j < c; j += kLanes) pi_column4<kT>(X, c, j, k);
}

// The root's words at column j (true scale, canonical) to sink(u, word): radix 2 from R0, R1;
// radix 4 from R_v, v <= kT (zero past; kT >= 2: rr > L).
template <int kRadix, int kT, class Sink>
[[gnu::always_inline]] inline void pi_root_column(const u32* X, std::size_t c, std::size_t j, const MiddleConstants& k,
                                                  Sink sink) {
    const auto R = [&](int v) { return Vec::load(X + std::size_t(v) * c + j); };
    const auto out = [&](int u, V v) { sink(u, Mod::reduce<kP>(v)); };
    const V r0 = R(0), r1 = R(1);
    if constexpr (kRadix == 2) {
        out(0, k.sc.times(add4(r0, r1))), out(1, k.sc.times(sub4(r0, r1)));
    } else {
        const V s01 = add2(r0, r1), a = k.sc.times(sub4(r0, r1));
        V s23 = R(2), b;
        if constexpr (kT == 3) {
            const V r2 = s23, r3 = R(3);
            s23 = add2(r2, r3), b = k.sc_z.times(sub4(r2, r3));
        } else {
            b = k.sc_z.times(s23);
        }
        out(0, k.sc.times(add4(s01, s23))), out(1, add2(a, b)), out(2, k.sc.times(sub4(s01, s23))), out(3, sub2(a, b));
    }
}

// Root columns [j0, j1) whose chunks u with bit u of kMask lie in [lo, hi) (the others outside):
// those to out[u c + j - lo].
template <int kRadix, int kT, int kMask>
[[gnu::noinline]] void pi_root_columns(const u32* X, std::size_t c, std::size_t j0, std::size_t j1, u32* out, std::size_t lo,
                                       const MiddleConstants& k) {
    for (std::size_t j = j0; j < j1; j += kLanes) {
        if (prefetches<kRadix>(c) && j % 16 == 0)
            for (int u = 0; u < kRadix; ++u) {
                ahead(X + std::size_t(u) * c + j);
                if (kMask >> u & 1) ahead_write(out + (std::size_t(u) * c + j - lo));
            }
        pi_root_column<kRadix, kT>(X, c, j, k, [&] [[gnu::always_inline]] (int u, V v) {
            if (kMask >> u & 1) Vec::store(out + (std::size_t(u) * c + j - lo), v);
        });
    }
}

// The root's words [lo, hi) to out (0 <= lo < hi <= 2L): runs of columns with the same chunks inside,
// and the vectors across an edge one by one.
template <int kRadix, int kT>
void pi_root(const u32* X, std::size_t c, u32* out, std::size_t lo, std::size_t hi, const MiddleConstants& k) {
    std::size_t cut[2 * kRadix], a[kRadix], b[kRadix];  // chunk u's columns inside: [a_u, b_u)
    for (int u = 0; u < kRadix; ++u) {
        const std::size_t first = std::size_t(u) * c;
        a[u] = std::clamp(lo, first, first + c) - first, b[u] = std::clamp(hi, first, first + c) - first;
        cut[2 * u] = a[u], cut[2 * u + 1] = b[u];
    }
    for (std::size_t j = 0; j < c;) {
        std::size_t next = c;  // the first cut past j
        bool edge = false;
        for (const std::size_t p : cut) {
            if (p > j) next = std::min(next, p);
            edge |= p > j && p < j + kLanes;
        }
        if (edge) {  // a cut inside the vector at j: each word apart
            pi_root_column<kRadix, kT>(X, c, j, k, [&] [[gnu::always_inline]] (int u, V v) {
                alignas(64) u32 lanes[kLanes];
                Vec::store(lanes, v);
                for (std::size_t l = 0; l < kLanes; ++l)
                    if (j + l >= a[u] && j + l < b[u]) out[std::size_t(u) * c + j + l - lo] = lanes[l];
            });
            j += kLanes;
            continue;
        }
        const std::size_t end = next / kLanes * kLanes;
        int mask = 0;
        for (int u = 0; u < kRadix; ++u) mask |= int(a[u] <= j && j < b[u]) << u;
        if (mask)
            with_blocks<(1 << kRadix) - 1>(mask, [&](auto m) {
                if constexpr (decltype(m)::value > 0) pi_root_columns<kRadix, kT, decltype(m)::value>(X, c, j, end, out, lo, k);
            });
        j = end;
    }
}

// out = q[lo, hi) (lo < hi <= s.N) from F after the group products (q's residues at s's full nodes
// times their scale, < 2P; zero past them): the levels bottom-up. F is overwritten.
inline void pruned_inverse(u32* F, u32* out, std::size_t lo, std::size_t hi, const Shape& s, const u32* r, const u32* ir) {
    for (int d = s.levels - 1; d >= 0; --d) {
        const Level& l = s.level[d];
        const std::size_t c = l.node.size / std::size_t(l.radix), t = l.n / c, rest = l.n % c;
        const MiddleConstants k(r, ir, l.node.k, s.lg + 1);
        u32* const X = F + l.node.at;
        if (d == 0 && l.radix == 2) {
            pi_root<2, 1>(X, c, out, lo, hi, k);
            continue;
        }
        with_blocks<3>(int(rest ? t : t - 1), [&](auto u) {
            constexpr int kT = decltype(u)::value;
            if (d > 0) pi_columns<kT>(X, c, k);
            else if constexpr (kT >= 2) pi_root<4, kT>(X, c, out, lo, hi, k);  // kt >= 2 at a radix-4 root (rr > L)
        });
    }
}

}  // namespace fps::ntt::detail::truncated
