// Variant intrinsics<S>: the NTT of avx2.hpp (same splitting, same lazy ranges, same leaf
// products) written once as templates on a simd struct S (backend/simd) and modarith, so every
// target runs it. Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/common.hpp"
#include "backend/simd/simd.hpp"

namespace fps::ntt::detail::intrinsics {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// The struct for work inside 8-word leaves: S if it has at most 8 lanes, else its ymm form.
template <class S>
struct Narrow {
    using type = S;
};
#if defined(__AVX2__) && defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512BW__)
template <>
struct Narrow<simd::Avx512> {
    using type = simd::Avx512Vl;
};
#endif

// Lazy arithmetic mod P on S: values < 4P in forward transforms, < 2P in inverse ones.
template <class S>
struct Arith {
    using V = typename S::V;
    using M = modarith::Ops<S, kP>;

    struct Factor {  // a constant w < P and its Shoup quotient
        V w, q;
    };
    static Factor factor(u32 w, u32 q) { return {S::broadcast(w), S::broadcast(q)}; }
    static Factor factor(u32 w) { return factor(w, quotient(w)); }
    static Factor entry(const u32* table, std::size_t k) { return factor(table[slot(k)], table[slot(k) + 8]); }

    static V add(V x, V y) { return S::add(x, y); }
    static V diff(V x, V y) { return S::sub(S::add(x, S::broadcast(2 * kP)), y); }  // x - y + 2P, y <= 2P
    static V low(V x) { return M::template reduce<2 * kP>(x); }                       // [0, 4P) -> [0, 2P)
    static V canonical(V x) { return M::template reduce<kP>(x); }                     // [0, 2P) -> [0, P)
    // x - y mod 2P for x, y < 2P, in [0, 2P): x - y wraps above 2P exactly when it is negative.
    static V low_diff(V x, V y) {
        const V t = S::sub(x, y);
        return S::min(t, S::add(t, S::broadcast(2 * kP)));
    }
    static V mul(V x, const Factor& f) { return whole(M::shoup(x, f.w, f.q)); }  // any x -> [0, 2P)

    // x86: x, unchanged, through an empty asm statement. Without it GCC distributes a product
    // x w - q P over its uses (a + c -> (a - q P) + x w): 24 adds and subtracts per forward
    // radix-4 column instead of 19 (notes.md, issue #5). Not on Neon, where GCC fuses those adds
    // into mla/mls (products 1-2% slower with the statement).
    static V whole(V x) {
#if defined(__x86_64__)
        if constexpr (S::lanes > 1) asm("" : "+v"(x));
#endif
        return x;
    }
};

// Radix-4 butterflies on 4 quarters of q words, q a multiple of S::lanes. Group k at quarter q is
// a polynomial mod X^4 - r(k)^2, X = x^q; x = r(k), y = r(2k), z = r(2k + 1) (scalar.hpp).
template <class S>
struct Butterflies : Arith<S> {
    using A = Arith<S>;
    using V = typename S::V;
    using F = typename A::Factor;
    using A::add, A::diff, A::low, A::low_diff, A::mul;

    // In < kIn P (kIn = 1, 2 or 4), out < 4P. Inputs below 2P need no reduction (the first level).
    template <int kIn = 4>
    [[gnu::always_inline]] static void forward4(V& f0, V& f1, V& f2, V& f3, const F& x, const F& y, const F& z) {
        const V a = below2(f0, kIn), b = below2(f1, kIn), c = mul(f2, x), d = mul(f3, x);
        const V ac = low(add(a, c)), amc = low_diff(a, c);
        const V yb = mul(add(b, d), y), zd = mul(diff(b, d), z);
        f0 = add(ac, yb), f1 = diff(ac, yb), f2 = add(amc, zd), f3 = diff(amc, zd);
    }
    // x = y = 1. In < kIn P, out < 4P.
    template <int kIn = 4>
    [[gnu::always_inline]] static void forward4_identity(V& f0, V& f1, V& f2, V& f3, const F& z) {
        const V a = below2(f0, kIn), b = below2(f1, kIn), c = below2(f2, kIn), d = below2(f3, kIn);
        if constexpr (kIn == 1) {  // sums < 2P; a - c + P in (0, 2P)
            const V ac = add(a, c), amc = S::sub(S::add(a, S::broadcast(kP)), c);
            const V bd = add(b, d), zd = mul(diff(b, d), z);
            f0 = add(ac, bd), f1 = diff(ac, bd), f2 = add(amc, zd), f3 = diff(amc, zd);
            return;
        }
        const V ac = low(add(a, c)), amc = low_diff(a, c);
        const V bd = low(add(b, d)), zd = mul(diff(b, d), z);
        f0 = add(ac, bd), f1 = diff(ac, bd), f2 = add(amc, zd), f3 = diff(amc, zd);
    }
    // x < bound P -> < 2P.
    [[gnu::always_inline]] static V below2(V x, int bound) { return bound > 2 ? low(x) : x; }
    // In < 2P, out < 2P.
    [[gnu::always_inline]] static void inverse4(V& f0, V& f1, V& f2, V& f3, const F& x, const F& y, const F& z) {
        const V ab = low(add(f0, f1)), cd = low(add(f2, f3));
        const V amb = mul(diff(f0, f1), y), cmd = mul(diff(f2, f3), z);
        f0 = low(add(ab, cd)), f1 = low(add(amb, cmd)), f2 = mul(diff(ab, cd), x), f3 = mul(diff(amb, cmd), x);
    }
    [[gnu::always_inline]] static void inverse4_identity(V& f0, V& f1, V& f2, V& f3, const F& z) {
        const V ab = low(add(f0, f1)), cd = low(add(f2, f3));
        const V amb = low_diff(f0, f1), cmd = mul(diff(f2, f3), z);
        f0 = low(add(ab, cd)), f1 = low(add(amb, cmd)), f2 = low_diff(ab, cd), f3 = low_diff(amb, cmd);
    }

    // Applies op to the 4 quarters of `from`, S::lanes words at a time, into those of f (may be from).
    template <class Op>
    [[gnu::always_inline]] static void each(const u32* from, u32* f, std::size_t q, Op op) {
        for (std::size_t j = 0; j < q; j += S::lanes) {
            const u32* s = from + j;
            V f0 = S::load(s), f1 = S::load(s + q), f2 = S::load(s + 2 * q), f3 = S::load(s + 3 * q);
            op(f0, f1, f2, f3);
            u32* g = f + j;
            S::store(g, f0), S::store(g + q, f1), S::store(g + 2 * q, f2), S::store(g + 3 * q, f3);
        }
    }
    template <class Op>
    [[gnu::always_inline]] static void each(u32* f, std::size_t q, Op op) { each(f, f, q, op); }

    // Group k of `from` into f (may be from). In < kIn P, out < 4P.
    template <int kIn = 4>
    static void forward_group(const u32* from, u32* f, std::size_t q, std::size_t k, const u32* r) {
        if (k == 0) {
            const F z = A::entry(r, 1);
            return each(from, f, q, [&] [[gnu::always_inline]] (V& a, V& b, V& c, V& d) {
                forward4_identity<kIn>(a, b, c, d, z);
            });
        }
        const F x = A::entry(r, k), y = A::entry(r, 2 * k), z = A::entry(r, 2 * k + 1);
        each(from, f, q, [&] [[gnu::always_inline]] (V& a, V& b, V& c, V& d) { forward4<kIn>(a, b, c, d, x, y, z); });
    }
    template <int kIn = 4>
    static void forward_group(u32* f, std::size_t q, std::size_t k, const u32* r) { forward_group<kIn>(f, f, q, k, r); }

    static void inverse_group(u32* f, std::size_t q, std::size_t k, const u32* ir) {
        if (k == 0) {
            const F z = A::entry(ir, 1);
            return each(f, q, [&] [[gnu::always_inline]] (V& a, V& b, V& c, V& d) { inverse4_identity(a, b, c, d, z); });
        }
        const F x = A::entry(ir, k), y = A::entry(ir, 2 * k), z = A::entry(ir, 2 * k + 1);
        each(f, q, [&] [[gnu::always_inline]] (V& a, V& b, V& c, V& d) { inverse4(a, b, c, d, x, y, z); });
    }
};

// The bottom level (quarters of 8 words, one leaf each) and the leaf products, on L (<= 8 lanes).
template <class L>
struct Bottom : Butterflies<L> {
    static_assert(8 % L::lanes == 0);
    using B = Butterflies<L>;
    using V = typename L::V;
    using F = typename B::Factor;
    using M = typename B::M;
    using B::canonical, B::low;

    static constexpr u32 kNegInverse = u32(0) - M::kPinv;  // -P^-1 mod 2^32

    // Groups [first, first + count), 32 words each, in < 4P; canonical leaves out.
    static void forward(u32* f, std::size_t count, std::size_t first, const u32* r) {
        for (std::size_t k = first; k < first + count; ++k, f += 32) {
            const F x = B::entry(r, k), y = B::entry(r, 2 * k), z = B::entry(r, 2 * k + 1);
            B::each(f, 8, [&] [[gnu::always_inline]] (V& a, V& b, V& c, V& d) {
                B::forward4(a, b, c, d, x, y, z);
                a = canonical(low(a)), b = canonical(low(b)), c = canonical(low(c)), d = canonical(low(d));
            });
        }
    }

    // In and out < 2P.
    static void inverse(u32* f, std::size_t count, std::size_t first, const u32* ir) {
        for (std::size_t k = first; k < first + count; ++k, f += 32) {
            const F x = B::entry(ir, k), y = B::entry(ir, 2 * k), z = B::entry(ir, 2 * k + 1);
            B::each(f, 8, [&] [[gnu::always_inline]] (V& a, V& b, V& c, V& d) { B::inverse4(a, b, c, d, x, y, z); });
        }
    }

    // One group's leaves for the products. Row t of window is [w_t A_t, A_t], canonical: x^i A_t
    // mod x^8 - w_t is the 8 words from 8 - i. The leaf products read 1 word past a row.
    struct alignas(64) Leaves {
        u32 window[4][16];
        u32 b[4][8];
    };

    // Forward butterflies of group k of a and b into leaves. The moduli x^8 - w_t of leaves
    // 4k + t: w_t = y, -y, z, -z (quotient(P - w) = ~quotient(w)).
    [[gnu::always_inline]] static void forward_leaves(const u32* a, const u32* b, Leaves& out, std::size_t k, const u32* r) {
        const F x = B::entry(r, k), y = B::entry(r, 2 * k), z = B::entry(r, 2 * k + 1);
        const u32* yz = r + slot(2 * k);
        const F w[4] = {B::factor(yz[0], yz[8]), B::factor(kP - yz[0], ~yz[8]), B::factor(yz[1], yz[9]),
                        B::factor(kP - yz[1], ~yz[9])};
        #pragma GCC unroll 8
        for (std::size_t j = 0; j < 8; j += L::lanes) {
            V f[4] = {L::load(a + j), L::load(a + 8 + j), L::load(a + 16 + j), L::load(a + 24 + j)};
            B::forward4(f[0], f[1], f[2], f[3], x, y, z);
            #pragma GCC unroll 8
            for (int t = 0; t < 4; ++t) {
                const V v = canonical(low(f[t]));
                L::store(out.window[t] + 8 + j, v);
                L::store(out.window[t] + j, canonical(B::mul(v, w[t])));
            }
            V g[4] = {L::load(b + j), L::load(b + 8 + j), L::load(b + 16 + j), L::load(b + 24 + j)};
            B::forward4(g[0], g[1], g[2], g[3], x, y, z);
            #pragma GCC unroll 8
            for (int t = 0; t < 4; ++t) L::store(out.b[t] + j, canonical(low(g[t])));
        }
    }

    // Words [j, j + lanes) of A B 2^-32 mod x^8 - w, < 2P: word l is the sum over i of
    // b_i window[8 - i + l] (< 8 P^2 < 2^63), then a Montgomery reduction (< 2^64 before the shift).
    [[gnu::always_inline]] static V leaf_product(const u32* window, const u32* b, std::size_t j) {
        const V p = L::broadcast(kP), ninv = L::broadcast(kNegInverse);
        if constexpr (requires(V v) { L::mul_even(v, v); }) {
            // 64-bit lanes: e gets the even words, o the odd ones (from loads one word on).
            V e = L::mul_even(L::load(window + 8 + j), L::broadcast(b[0]));
            V o = L::mul_even(L::load(window + 9 + j), L::broadcast(b[0]));
            #pragma GCC unroll 8
            for (int i = 1; i < 8; ++i) {
                const V bi = L::broadcast(b[i]);
                e = L::add64(e, L::mul_even(L::load(window + 8 - i + j), bi));
                o = L::add64(o, L::mul_even(L::load(window + 9 - i + j), bi));
            }
            e = L::add64(e, L::mul_even(L::mul_even(e, ninv), p));
            o = L::add64(o, L::mul_even(L::mul_even(o, ninv), p));
            return low(L::join_high(e, o));
        } else if constexpr (requires(V v) { L::mul_wide_lo(v, v); }) {
            const V x0 = L::load(window + 8 + j), b0 = L::broadcast(b[0]);
            auto lo = L::mul_wide_lo(x0, b0), hi = L::mul_wide_hi(x0, b0);
            #pragma GCC unroll 8
            for (int i = 1; i < 8; ++i) {
                const V x = L::load(window + 8 - i + j), bi = L::broadcast(b[i]);
                lo = L::mla_wide_lo(lo, x, bi), hi = L::mla_wide_hi(hi, x, bi);
            }
            const V m = L::mul_lo(L::lo_halves(lo, hi), ninv);
            lo = L::mla_wide_lo(lo, m, p), hi = L::mla_wide_hi(hi, m, p);
            return low(L::hi_halves(lo, hi));
        } else {
            static_assert(L::lanes == 1);
            u64 s = 0;
            #pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) s += u64(window[8 - i + j]) * b[i];
            const u32 m = u32(s) * kNegInverse;
            return low(V((s + u64(m) * kP) >> 32));
        }
    }

    // Leaf products of one group into a, then its inverse butterflies; out < 2P.
    [[gnu::always_inline]] static void multiply_leaves(u32* a, const Leaves& in, std::size_t k, const u32* ir) {
        const F x = B::entry(ir, k), y = B::entry(ir, 2 * k), z = B::entry(ir, 2 * k + 1);
        #pragma GCC unroll 8
        for (std::size_t j = 0; j < 8; j += L::lanes) {
            V c[4];
            #pragma GCC unroll 8
            for (int t = 0; t < 4; ++t) c[t] = leaf_product(in.window[t], in.b[t], j);
            B::inverse4(c[0], c[1], c[2], c[3], x, y, z);
            #pragma GCC unroll 8
            for (int t = 0; t < 4; ++t) L::store(a + 8 * t + j, c[t]);
        }
    }

    // Groups [first, first + count) of a and b, in < 4P: forward butterflies, leaf products into
    // a, inverse butterflies; out < 2P. Group g + 1's forward half runs before group g's
    // products, so the leaf rows are not loaded right after their stores.
    static void multiply(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        Leaves leaves[2];
        forward_leaves(a, b, leaves[0], first, r);
        for (std::size_t g = 0; g < count; ++g) {
            if (g + 1 < count) forward_leaves(a + 32 * (g + 1), b + 32 * (g + 1), leaves[(g + 1) % 2], first + g + 1, r);
            multiply_leaves(a + 32 * g, leaves[g % 2], first + g, ir);
        }
    }
};

// The bottom level on S with 16 lanes (zmm), two groups k, k + 1 (k even) at a time: vector t
// holds quarter t of both, [group k, group k + 1], so the butterflies are lane-wise; twiddles are
// [w_k x 8, w_(k+1) x 8].
template <class S>
struct WideBottom : Butterflies<S> {
    static_assert(S::lanes == 16);
    using B = Butterflies<S>;
    using V = typename S::V;
    using F = typename B::Factor;
    using B::canonical, B::low;

    static typename S::Index index(int a, int b) {
        std::array<int, 16> lanes;
        for (int i = 0; i < 16; ++i) lanes[i] = i < 8 ? a : b;
        return S::index(lanes);
    }

    // Lanes of a table block: x pair (0, 1), y pair (0, 2), z pair (1, 3); quotients 8 on.
    struct Indices {
        typename S::Index xw = index(0, 1), xq = index(8, 9), yw = index(0, 2), yq = index(8, 10),
                          zw = index(1, 3), zq = index(9, 11), low, high;
        Indices() {
            alignas(64) u32 l[16], h[16];
            for (int i = 0; i < 16; ++i) l[i] = i < 8 ? i : 8 + i, h[i] = l[i] + 8;
            low = S::load(l), high = S::load(h);
        }
    };

    struct Pair {
        F x, y, z;
    };
    // Twiddles of groups k, k + 1; the table blocks read are 16 words from entry k and 2k.
    [[gnu::always_inline]] static Pair twiddles(const u32* table, std::size_t k, const Indices& n) {
        const V xs = S::load(table + slot(k)), yz = S::load(table + slot(2 * k));
        return {{S::permute(xs, n.xw), S::permute(xs, n.xq)}, {S::permute(yz, n.yw), S::permute(yz, n.yq)},
                {S::permute(yz, n.zw), S::permute(yz, n.zq)}};
    }
    // The same by broadcast_halves: off Intel's p5, which the leaf products load (notes.md).
    [[gnu::always_inline]] static Pair twiddles(const u32* table, std::size_t k) {
        const u32 *x = table + slot(k), *y = table + slot(2 * k);
        const auto pair = [](const u32* p, int a, int b) { return S::broadcast_halves(p + a, p + b); };
        return {{pair(x, 0, 1), pair(x, 8, 9)}, {pair(y, 0, 2), pair(y, 8, 10)}, {pair(y, 1, 3), pair(y, 9, 11)}};
    }

    // 64 words of groups k, k + 1 -> the quarter vectors, and back.
    [[gnu::always_inline]] static void load(const u32* f, V q[4], const Indices& n) {
        const V z0 = S::load(f), z1 = S::load(f + 16), z2 = S::load(f + 32), z3 = S::load(f + 48);
        q[0] = S::permute2(z0, n.low, z2), q[1] = S::permute2(z0, n.high, z2);
        q[2] = S::permute2(z1, n.low, z3), q[3] = S::permute2(z1, n.high, z3);
    }
    [[gnu::always_inline]] static void store(u32* f, const V q[4], const Indices& n) {
        S::store(f, S::permute2(q[0], n.low, q[1])), S::store(f + 16, S::permute2(q[2], n.low, q[3]));
        S::store(f + 32, S::permute2(q[0], n.high, q[1])), S::store(f + 48, S::permute2(q[2], n.high, q[3]));
    }

    // Leaf::forward and Leaf::inverse on groups [first, first + count), first and count even.
    static void forward(u32* f, std::size_t count, std::size_t first, const u32* r) {
        const Indices n;
        for (std::size_t k = first; k < first + count; k += 2, f += 64) {
            const Pair w = twiddles(r, k, n);
            V q[4];
            load(f, q, n);
            B::forward4(q[0], q[1], q[2], q[3], w.x, w.y, w.z);
            for (V& v : q) v = canonical(low(v));
            store(f, q, n);
        }
    }

    static void inverse(u32* f, std::size_t count, std::size_t first, const u32* ir) {
        const Indices n;
        for (std::size_t k = first; k < first + count; k += 2, f += 64) {
            const Pair w = twiddles(ir, k, n);
            V q[4];
            load(f, q, n);
            B::inverse4(q[0], q[1], q[2], q[3], w.x, w.y, w.z);
            store(f, q, n);
        }
    }

    // The canonical leaves of groups k, k + 1 in memory, for leaf_products. rows[t][h] = [w a, a]
    // for quarter t of group k + h: x^(8-j) A mod x^8 - w is words j .. j + 7 of a row (j = 1 .. 9;
    // j = 9 reads 1 word past the row, a don't-care lane). a[t], b[t] = [quarter t of k, of k + 1].
    struct alignas(64) Leaves {
        u32 rows[4][2][16];
        u32 a[4][16], b[4][16];
    };

    // [A_k B_k, A_(k+1) B_(k+1)] 2^-32 mod x^8 - w per half, < 2P. Windows are load_halves of the
    // rows, b_i per half broadcast_halves of b: those run on p0 or p5 on Emerald Rapids, where the
    // vpermd of b ran on p5 only, as all shuffles do (notes.md).
    [[gnu::always_inline]] static V leaf_products(const Leaves& in, int t) {
        using M = typename B::M;
        const V p = S::broadcast(kP), ninv = S::broadcast(u32(0) - M::kPinv);
        const auto x = [&](int j) { return j == 8 ? S::load(in.a[t]) : S::load_halves(in.rows[t][0] + j, in.rows[t][1] + j); };
        V e = {}, o = {};
        #pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) {
            const V bi = S::broadcast_halves(in.b[t] + i, in.b[t] + 8 + i);
            const V pe = S::mul_even(x(8 - i), bi), po = S::mul_even(x(9 - i), bi);
            e = i == 0 ? pe : S::add64(e, pe), o = i == 0 ? po : S::add64(o, po);
        }
        e = S::add64(e, S::mul_even(S::mul_even(e, ninv), p));
        o = S::add64(o, S::mul_even(S::mul_even(o, ninv), p));
        return low(S::join_high(e, o));
    }

    [[gnu::always_inline]] static void forward_leaves(const u32* a, const u32* b, std::size_t k, const u32* r,
                                                      const Indices& n, Leaves& out) {
        const V p = S::broadcast(kP), ones = S::broadcast(~u32(0));
        const Pair w = twiddles(r, k);
        const F minus_y = {S::sub(p, w.y.w), S::sub(ones, w.y.q)}, minus_z = {S::sub(p, w.z.w), S::sub(ones, w.z.q)};
        const F* weight[4] = {&w.y, &minus_y, &w.z, &minus_z};
        V f[4], g[4];
        load(a, f, n);
        load(b, g, n);
        B::forward4(f[0], f[1], f[2], f[3], w.x, w.y, w.z);
        B::forward4(g[0], g[1], g[2], g[3], w.x, w.y, w.z);
        #pragma GCC unroll 4
        for (int t = 0; t < 4; ++t) {
            const V at = canonical(low(f[t])), wat = canonical(B::mul(at, *weight[t]));
            S::template store_half<0>(out.rows[t][0], wat), S::template store_half<0>(out.rows[t][0] + 8, at);
            S::template store_half<1>(out.rows[t][1], wat), S::template store_half<1>(out.rows[t][1] + 8, at);
            S::store(out.a[t], at), S::store(out.b[t], canonical(low(g[t])));
        }
    }

    [[gnu::always_inline]] static void multiply_leaves(u32* a, std::size_t k, const u32* ir, const Indices& n,
                                                       const Leaves& in) {
        V f[4];
        #pragma GCC unroll 4
        for (int t = 0; t < 4; ++t) f[t] = leaf_products(in, t);
        const Pair iw = twiddles(ir, k);
        B::inverse4(f[0], f[1], f[2], f[3], iw.x, iw.y, iw.z);
        store(a, f, n);
    }

    // Leaf::multiply on S; count and first even. Per chunk of kChunk pairs, one loop of forward
    // halves into a buffer, then one of leaf products and inverse halves: shorter dependency chains
    // per loop iteration than both in one loop (notes.md).
    static constexpr std::size_t kChunk = 4;
    static void multiply(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        const Indices n;
        Leaves leaves[kChunk];
        for (std::size_t c = 0; c < count / 2; c += kChunk, a += 64 * kChunk, b += 64 * kChunk) {
            const std::size_t m = std::min(kChunk, count / 2 - c), k = first + 2 * c;
            for (std::size_t g = 0; g < m; ++g) forward_leaves(a + 64 * g, b + 64 * g, k + 2 * g, r, n, leaves[g]);
            for (std::size_t g = 0; g < m; ++g) multiply_leaves(a + 64 * g, k + 2 * g, ir, n, leaves[g]);
        }
    }
};

// Lengths below 2^kWideMinLog run on Narrow<S> (notes.md, 2026-10-11): at 2^7 zmm was slower
// than ymm on Emerald Rapids (inverse 1.36x the asm against 1.05x), not on Zen 5.
#if defined(__znver4__) || defined(__znver5__)
inline constexpr int kWideMinLog = 7;
#else
inline constexpr int kWideMinLog = 8;
#endif

template <class S>
struct Ntt : Butterflies<S> {
    using N = typename Narrow<S>::type;
    using Leaf = Bottom<N>;
    using B = Butterflies<S>;
    using V = typename S::V;
    using F = typename B::Factor;
    using B::add, B::diff, B::canonical, B::forward_group, B::inverse_group;

    // Subtrees of at most this many leaves run level by level (8 KiB per array).
    static constexpr std::size_t kTile = 256;

    // variant::scalar::build_table, N::lanes entries at a time past the static prefix.
    static void build_table(u32* t, std::size_t count, bool inverse) {
        static_assert(kStaticCount >= 8);
        using A = Arith<N>;
        const Roots& roots = kRoots[inverse];
        for (std::size_t h = copy_static_table(t, count, inverse); h < count; h *= 2) {
            // Quotient of v: floor(v 2^32 / P) = -(v 2^32 mod P) P^-1 mod 2^32.
            const auto by = A::factor(roots[std::countr_zero(h)]), to_r = A::factor(kR);
            for (std::size_t j = 0; j < h; j += N::lanes) {
                const auto v = A::canonical(A::mul(N::load(t + slot(j)), by));
                const auto vr = A::canonical(A::mul(v, to_r));
                N::store(t + slot(h + j), v);
                N::store(t + slot(h + j) + 8, N::mul_lo(vr, N::broadcast(Leaf::kNegInverse)));
            }
        }
    }

    // The bottom level of groups [first, first + count): on S two groups at a time if S has 16
    // lanes and count is even, else on N.
    static constexpr bool kWideBottom = S::lanes == 16;
    static void bottom_forward(u32* f, std::size_t count, std::size_t first, const u32* r) {
        if constexpr (kWideBottom)
            if (count % 2 == 0) return WideBottom<S>::forward(f, count, first, r);
        Leaf::forward(f, count, first, r);
    }
    static void bottom_inverse(u32* f, std::size_t count, std::size_t first, const u32* ir) {
        if constexpr (kWideBottom)
            if (count % 2 == 0) return WideBottom<S>::inverse(f, count, first, ir);
        Leaf::inverse(f, count, first, ir);
    }
    static void bottom_multiply(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r,
                                const u32* ir) {
        if constexpr (kWideBottom)
            if (count % 2 == 0) return WideBottom<S>::multiply(a, b, count, first, r, ir);
        Leaf::multiply(a, b, count, first, r, ir);
    }

    // Subtree of group k with nv leaves (a power of 4), depth first; its inputs < kIn P.
    template <int kIn = 4>
    static void forward_tree(u32* f, std::size_t nv, std::size_t k, const u32* r) {
        std::size_t h = nv / 4;
        if (nv > kTile) {
            B::template forward_group<kIn>(f, 8 * h, k, r);
            for (std::size_t t = 0; t < 4; ++t) forward_tree(f + 8 * h * t, h, 4 * k + t, r);
            return;
        }
        if (h >= 4) B::template forward_group<kIn>(f, 8 * h, k, r), h /= 4;  // the first level: group k alone
        for (; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < nv; j += 4 * h, ++g) forward_group(f + 8 * j, 8 * h, g, r);
        bottom_forward(f, nv / 4, k * (nv / 4), r);
    }

    static void inverse_tree(u32* f, std::size_t nv, std::size_t k, const u32* ir) {
        if (nv > kTile) {
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) inverse_tree(f + 8 * h * t, h, 4 * k + t, ir);
            inverse_group(f, 8 * h, k, ir);
            return;
        }
        bottom_inverse(f, nv / 4, k * (nv / 4), ir);
        for (std::size_t h = 4; h < nv; h *= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < nv; j += 4 * h, ++g) inverse_group(f + 8 * j, 8 * h, g, ir);
    }

    // Both transforms, the leaf products and the inverse transform of a subtree; inputs < kIn P.
    // The first level reads a_from and b_from (a and b, or other arrays if nv >= 16).
    template <int kIn = 4>
    static void multiply_tree(const u32* a_from, const u32* b_from, u32* a, u32* b, std::size_t nv, std::size_t k,
                              const u32* r, const u32* ir) {
        std::size_t h = nv / 4;
        if (nv > kTile) {
            B::template forward_group<kIn>(a_from, a, 8 * h, k, r);
            B::template forward_group<kIn>(b_from, b, 8 * h, k, r);
            for (std::size_t t = 0; t < 4; ++t) multiply_tree(a + 8 * h * t, b + 8 * h * t, h, 4 * k + t, r, ir);
            inverse_group(a, 8 * h, k, ir);
            return;
        }
        if (h >= 4) {  // the first level: group k alone
            B::template forward_group<kIn>(a_from, a, 8 * h, k, r);
            B::template forward_group<kIn>(b_from, b, 8 * h, k, r);
            h /= 4;
        }
        for (; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < nv; j += 4 * h, ++g) {
                forward_group(a + 8 * j, 8 * h, g, r);
                forward_group(b + 8 * j, 8 * h, g, r);
            }
        bottom_multiply(a, b, nv / 4, k * (nv / 4), r, ir);
        for (std::size_t h = 4; h < nv; h *= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < nv; j += 4 * h, ++g) inverse_group(a + 8 * j, 8 * h, g, ir);
    }
    template <int kIn = 4>
    static void multiply_tree(u32* a, u32* b, std::size_t nv, std::size_t k, const u32* r, const u32* ir) {
        multiply_tree<kIn>(a, b, a, b, nv, k, r, ir);
    }

    // Top levels. Inputs canonical. If the upper half of f is zero ("sparse"), it is not read.

    // The radix-2 level on halves of `half` words: out < 2P.
    static void forward_radix2(u32* f, std::size_t half, bool sparse) {
        if (sparse) {
            std::memcpy(f + half, f, half * sizeof(u32));
            return;
        }
        const V p = S::broadcast(kP);
        for (std::size_t i = 0; i < half; i += S::lanes) {
            const V x = S::load(f + i), y = S::load(f + i + half);
            S::store(f + i, add(x, y)), S::store(f + i + half, S::sub(S::add(x, p), y));
        }
    }

    // Group 0 on quarters of q words: out < 4P.
    static void forward_radix4(u32* f, std::size_t q, bool sparse, const u32* r) {
        if (!sparse) return B::template forward_group<1>(f, q, 0, r);
        const F z = B::entry(r, 1);
        for (std::size_t j = 0; j < q; j += S::lanes) {
            const V a = S::load(f + j), b = S::load(f + j + q), zb = B::mul(b, z);
            S::store(f + j, add(a, b)), S::store(f + j + q, diff(a, b));
            S::store(f + j + 2 * q, add(a, zb)), S::store(f + j + 3 * q, diff(a, zb));
        }
    }

    // The inverse radix-2 level and the scale s: in < 2P, canonical out.
    static void inverse_radix2(u32* f, std::size_t half, u32 s) {
        const F scale = B::factor(s);
        for (std::size_t i = 0; i < half; i += S::lanes) {
            const V x = S::load(f + i), y = S::load(f + i + half);
            S::store(f + i, canonical(B::mul(add(x, y), scale)));
            S::store(f + i + half, canonical(B::mul(diff(x, y), scale)));
        }
    }

    // Inverse group 0 on quarters of q words and the scale s: in < 2P, canonical out. 4 products per
    // column, not 5: s z (c - d) and s (a - b) serve two outputs each.
    static void inverse_radix4(u32* f, std::size_t q, const u32* ir, u32 s) {
        const F scale = B::factor(s), sz = B::factor(multiply_mod(s, ir[slot(1)]));
        B::each(f, q, [&] [[gnu::always_inline]] (V& a, V& b, V& c, V& d) {
            const V ab = B::low(add(a, b)), cd = B::low(add(c, d));
            const V amb = B::mul(diff(a, b), scale), cmd = B::mul(diff(c, d), sz);
            a = canonical(B::mul(add(ab, cd), scale)), c = canonical(B::mul(diff(ab, cd), scale));
            b = canonical(B::low(add(amb, cmd))), d = canonical(B::low(diff(amb, cmd)));
        });
    }

    static constexpr bool kNarrow = !std::is_same_v<S, N>;

    // Top levels by lg: nv = 4^j leaves start with group 0; nv = 2 * 4^j with the radix-2 level.
    static void forward(u32* f, int lg, const u32* r) {
        if constexpr (kNarrow)
            if (lg < kWideMinLog) return Ntt<N>::forward(f, lg, r);
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        if (!has_radix2(lg)) return forward_tree<1>(f, nv, 0, r);
        forward_radix2(f, len / 2, false);
        forward_tree<2>(f, nv / 2, 0, r);
        forward_tree<2>(f + len / 2, nv / 2, 1, r);
    }

    static void inverse(u32* f, int lg, const u32* ir, u32 s) {
        if constexpr (kNarrow)
            if (lg < kWideMinLog) return Ntt<N>::inverse(f, lg, ir, s);
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        if (!has_radix2(lg)) {
            for (std::size_t t = 0; t < 4; ++t) inverse_tree(f + t * len / 4, nv / 4, t, ir);
            return inverse_radix4(f, len / 4, ir, s);
        }
        inverse_tree(f, nv / 2, 0, ir);
        inverse_tree(f + len / 2, nv / 2, 1, ir);
        inverse_radix2(f, len / 2, s);
    }

    static void multiply(u32* a, u32* b, std::size_t n, std::size_t m, int lg, const u32* r, const u32* ir) {
        if constexpr (kNarrow)
            if (lg < kWideMinLog) return Ntt<N>::multiply(a, b, n, m, lg, r, ir);
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        const bool sparse_a = n <= len / 2, sparse_b = m <= len / 2;
        const u32 s = product_scale(lg);
        if (!has_radix2(lg)) {
            const std::size_t q = len / 4;
            forward_radix4(a, q, sparse_a, r);
            forward_radix4(b, q, sparse_b, r);
            for (std::size_t t = 0; t < 4; ++t) multiply_tree(a + t * q, b + t * q, nv / 4, t, r, ir);
            return inverse_radix4(a, q, ir, s);
        }
        // After the radix-2 level a sparse factor's upper half equals its lower half, so the upper
        // subtree's first level reads the lower half instead of a copy, before the lower subtree runs
        // in place. Not for nv / 2 = 4, whose first level is the bottom (in place only).
        const std::size_t half = len / 2;
        const bool share_a = sparse_a && nv / 2 >= 16, share_b = sparse_b && nv / 2 >= 16;
        if (!share_a) forward_radix2(a, half, sparse_a);
        if (!share_b) forward_radix2(b, half, sparse_b);
        multiply_tree<2>(share_a ? a : a + half, share_b ? b : b + half, a + half, b + half, nv / 2, 1, r, ir);
        multiply_tree<2>(a, b, nv / 2, 0, r, ir);
        inverse_radix2(a, half, s);
    }
};

}  // namespace fps::ntt::detail::intrinsics

namespace fps::ntt::variant {

// S: a struct of backend/simd/simd.hpp this target has.
template <class S>
struct intrinsics {
    using Ntt = detail::intrinsics::Ntt<S>;

    // Entries [0, count) of the forward or inverse table.
    static void build_table(std::uint32_t* t, std::size_t count, bool inverse) { Ntt::build_table(t, count, inverse); }

    static void forward(std::uint32_t* f, int lg, const std::uint32_t* roots) { Ntt::forward(f, lg, roots); }

    static void inverse(std::uint32_t* f, int lg, const std::uint32_t* inverse_roots) {
        Ntt::inverse(f, lg, inverse_roots, detail::leaf_scale(lg));
    }

    // Does not read the upper half of a factor of at most half the transform length.
    static void multiply(std::uint32_t* a, std::uint32_t* b, std::size_t n, std::size_t m, int lg,
                         const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        Ntt::multiply(a, b, n, m, lg, roots, inverse_roots);
    }
};

}  // namespace fps::ntt::variant
