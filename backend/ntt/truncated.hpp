// Truncated products (issue #155 round 2): c = f g of N words, L < N < 2L, L = 2^lg, from a
// truncated transform of length 2L: the variants' group products on the nodes of the leaf form's
// first rr >= N words, about N words of products where the plain one runs 2L. ntt.hpp runs them
// (truncated_into) for multiply and square when ragged.hpp's plan is truncated; this header has the
// shape and the levels above the group products.
//
//   shape(N)                  the levels (the partial nodes, the root first) and the full nodes
//   forward(f, n, F, s, r)    f mod each full node, into F (the leaf form of length 2L, cut)
//   inverse(F, out, s, r, ir) F's group products -> c = out[0, N)
//
// Node k of s words is x^s - r(k)^2 (r: the twiddles of common.hpp); its halves are nodes 2k and 2k
// + 1, its quarters 4k + t, the root x^(2L) - 1 node 0. A node whose s / 8 is a power of 4 is the
// variants' group k with s / 8 leaves; nodes split into quarters where those are groups, else into
// halves (as the variants' trees: a radix-2 level, then radix-4 ones), so only the root has radix 2. The leaf form's first rr
// words (rr: N rounded up, ragged::rounded) are the full nodes; as c has no words past rr, its
// residues there give c (van der Hoeven's truncated transform, with the roots of unity of the
// transform of 2L, so no twist pass).
//
// Levels: a partial node of s words, its first n words unknown (0 < n < s), has children of c = s /
// radix words: t = n / c full ones (group products), then a partial child with n mod c unknown words
// if n mod c > 0, and none past it. Its forward writes those children. Its inverse runs on columns
// j < c: word j of each chunk p_u = p[u c, (u + 1) c) and of each child's residue R_v = sum_u rho_v^u
// p_u (rho = y, -y, z, -z with y = r(2k), z = r(2k + 1); radix 2: x, -x with x = r(k)). Step 1,
// top-down, on the columns j >= n mod c: p_u is known for u >= t (the root's are zero), so the full
// children give p_u for u < t, and R_t, the partial child's known word j. Step 2, bottom-up after the
// partial child, on j < n mod c: p_u for u <= t from R_v, v <= t. The p_u go in place of the
// children's words (the root's to out); known words are not read again.
//
// Scales: unnormalized, as the variants' inverses: a group product of nv leaves is nv 2^-32 times
// its residue, a node's p comes out times its leaves 2^-32 too, and the root's scale goes into its
// level's constants. A node's known words are mu times its scale (mu, a power of 2, folded into its
// constants): radix when its parent had full children, else the parent's mu times radix.
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
// scale of its known words over its own (above).
struct Level {
    Node node;
    std::size_t n;
    int radix;
    u32 mu;
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
        s.level[s.levels++] = {node, n, radix, mu};
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
        : x(Constant::entry(r, k)), y(Constant::entry(r, radix == 4 ? 2 * k : k)), z(Constant::entry(r, radix == 4 ? 2 * k + 1 : k)) {}
};

// Children t <= kLast of a node at column j from chunks u < kLoad of src (the others zero, not
// read), c words apart; kEdge: each chunk read to len words. kRoot: k = 0 (x = y = 1). In and out
// < 2P, to dst + t c.
template <int kRadix, int kLoad, int kLast, bool kEdge, bool kRoot>
[[gnu::always_inline]] inline void forward_column(const u32* src, std::size_t c, std::size_t len, u32* dst, std::size_t j,
                                                  const Twiddles& w) {
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
    for (std::size_t j = j0; j < j1; j += kLanes) forward_column<kRadix, kLoad, kLast, kEdge, kRoot>(src, c, len, dst, j, w);
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
    for (int d = 0; d < s.levels; ++d) {
        const Level& l = s.level[d];
        const std::size_t c = l.node.size / std::size_t(l.radix), t = l.n / c;
        const int last = int(l.n % c ? t : t - 1);
        u32* const at = F + l.node.at;
        const Twiddles w(r, l.node.k, l.radix);
        if (d > 0) forward_level<4, false>(at, l.node.size, c, last, at, w);
        else if (l.radix == 4) forward_level<4, true>(f, n, c, last, at, w);
        else forward_level<2, true>(f, n, c, last, at, w);
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
                x_mu = Constant(multiply_mod(r[slot(k)], inv_mu)), y_mu = Constant(multiply_mod(r[slot(2 * k)], inv_mu));
                two_y = Constant(multiply_mod(2, ir[slot(2 * k)]));
                inv_x = Constant::entry(ir, k), inv_y = Constant::entry(ir, 2 * k), inv_z = Constant::entry(ir, 2 * k + 1);
            }
        }
        if (root) sc = Constant(product_scale(lg2));
    }
};

// The root's words: c[at, at + kLanes), canonical; kCut: cut to N words.
struct Out {
    u32* c;
    std::size_t N;
    template <bool kCut>
    [[gnu::always_inline]] void put(std::size_t at, V v) const {
        v = Mod::reduce<kP>(v);
        if (!kCut || at + kLanes <= N) Vec::store(c + at, v);
        else if (at < N) Vec::store_first(c + at, v, int(N - at));
    }
};

// One column j of a level's inverse, kU of its children's residues known (R_v, v < kU, at X + v c +
// j; the last one the partial child's in step 2, at the same scale) and p_u for u >= kU (K_u, at X +
// u c + j; the root's zero): p_u for u < kU (P_u; the root's c[u c + j]), and with kNext R_kU (Q) to
// X + kU c + j, at the scale of the node's words (mu 1). Radix 4, true values: E, G = p0 +- x p2; F,
// H = p1 +- x p3; R0, R1 = E +- y F; R2, R3 = G +- z H. Radix 2: R0, R1 = p0 +- x p1. With o the
// node's scale, R_v = o / radix times its true value and K_u = mu o times p_u.
template <int kRadix, int kU, bool kNext, bool kRoot, bool kCut = false>
    requires(kRoot || kRadix == 4)
[[gnu::always_inline]] inline void inverse_column(u32* X, std::size_t c, std::size_t j, const Constants& k, const Out& out) {
    const auto R = [&](int v) { return Vec::load(X + std::size_t(v) * c + j); };
    const auto K = [&](int u) { return Vec::load(X + std::size_t(u) * c + j); };
    const auto P = [&](int u, V v) {  // the root's chunk kU - 1 may pass N (kCut)
        if constexpr (kRoot) {
            if (kCut && u == kU - 1) out.put<true>(std::size_t(u) * c + j, v);
            else out.put<false>(std::size_t(u) * c + j, v);
        } else {
            Vec::store(X + std::size_t(u) * c + j, v);
        }
    };
    const auto Q = [&](V v) { Vec::store(X + std::size_t(kU) * c + j, v); };
    const auto twice = [](V v) { return add2(v, v); };
    if constexpr (kRoot) {  // k = 0: x = y = 1, z = r(1); K = 0; outputs times sc
        if constexpr (kRadix == 2) {
            if constexpr (kU == 1) {
                const V p0 = twice(R(0));
                P(0, k.sc.times(p0));
                if constexpr (kNext) Q(p0);
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

template <int kRadix, int kU, bool kNext, bool kRoot, bool kCut = false>
    requires(kRoot || kRadix == 4)
[[gnu::noinline]] void inverse_columns(u32* X, std::size_t c, std::size_t j0, std::size_t j1, const Constants& k,
                                       const Out& out) {
    if (prefetches<kRadix>(c)) {
        for (std::size_t j = j0; j < j1; j += kLanes) {
            if (j % 16 == 0) {  // a line per chunk read, and per chunk of out written
                for (int u = 0; u < kRadix; ++u) ahead(X + std::size_t(u) * c + j);
                if constexpr (kRoot)
                    for (int u = 0; u < kU; ++u) ahead_write(out.c + std::size_t(u) * c + j);
            }
            inverse_column<kRadix, kU, kNext, kRoot, kCut>(X, c, j, k, out);
        }
        return;
    }
    for (std::size_t j = j0; j < j1; j += kLanes) inverse_column<kRadix, kU, kNext, kRoot, kCut>(X, c, j, k, out);
}

// Step 1 (top-down) or step 2 (bottom-up, second) of a level.
template <int kRadix, bool kRoot>
void inverse_step(const Level& l, bool second, u32* X, const Constants& k, const Out& out) {
    const std::size_t c = l.node.size / std::size_t(kRadix), t = l.n / c, rest = l.n % c;
    with_blocks<kRadix - 1>(int(t), [&](auto u) {
        constexpr int kT = decltype(u)::value;
        // The root has t >= radix / 2 (N > L).
        // The root's last chunk written, t in step 2 or t - 1 with no partial child, may pass N:
        // columns from N - (its chunk) c on are cut; N > t c if rest > 0, else N > (t - 1) c.
        const auto run = [&]<int kU, bool kNext>(std::size_t j0, std::size_t j1) {
            if constexpr (kRoot) {
                const std::size_t cut = std::clamp((out.N - (kU - 1) * c) / kLanes * kLanes, j0, j1);
                inverse_columns<kRadix, kU, kNext, kRoot>(X, c, j0, cut, k, out);
                inverse_columns<kRadix, kU, kNext, kRoot, true>(X, c, cut, j1, k, out);
            } else {
                inverse_columns<kRadix, kU, kNext, kRoot>(X, c, j0, j1, k, out);
            }
        };
        if constexpr (!kRoot || 2 * kT >= kRadix) {
            if (second) run.template operator()<kT + 1, false>(0, rest);
            else if (rest) inverse_columns<kRadix, kT, true, kRoot>(X, c, rest, c, k, out);
            else if constexpr (kT > 0) run.template operator()<kT, false>(0, c);
        }
    });
}

// c = out[0, N) from F, the leaf form after the group products of s's full nodes (< 2P, times
// their scale): the levels' steps 1, then their steps 2. F is overwritten. r, ir: the tables of 2L.
inline void inverse(u32* F, u32* out, const Shape& s, const u32* r, const u32* ir) {
    const Out o{out, s.N};
    Constants constants[kMaxLevels];
    const auto step = [&](int d, bool second) {
        const Level& l = s.level[d];
        if (!second) constants[d] = Constants(r, ir, l, s.lg + 1);
        const Constants& k = constants[d];
        u32* const X = F + l.node.at;
        if (d > 0) inverse_step<4, false>(l, second, X, k, o);
        else if (l.radix == 4) inverse_step<4, true>(l, second, X, k, o);
        else inverse_step<2, true>(l, second, X, k, o);
    };
    for (int d = 0; d < s.levels; ++d) step(d, false);
    for (int d = s.levels - 1; d >= 0; --d)
        if (s.level[d].n % (s.level[d].node.size / std::size_t(s.level[d].radix))) step(d, true);
}

}  // namespace fps::ntt::detail::truncated
