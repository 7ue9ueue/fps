// Composition of power series modulo P = 998244353 (backend kernel): Kinoshita and Li's power
// projection and its transpose. From lc-opt's lib/poly/{composition,projection,
// compositional_inverse}.hpp, rewritten on ntt::Transform's leaf form and backend/simd, so every
// target runs it. Design and measurements: notes.md.
//
//   fps::composition::compose(f, g, out);                // out = f(g) mod x^n, n = out.size()
//   fps::composition::power_projection(g, out);          // out[k] = [x^(n-1)] g^k, k < n
//   fps::composition::compositional_inverse(f, out);     // out = g with f(g) = x mod x^n
//
// Spans of words < P. Only f and g mod x^n are read (shorter ones have zeros past their end);
// g[0] and f[0] are not read and count as 0. compositional_inverse needs f[1] != 0 and n <= P
// (not checked: the frontend checks them). out must not overlap f or g. n past kMaxSize throws
// std::length_error before out is written.
//
// Kinoshita and Li (arXiv 2404.05177), with lc-opt's layout. m = 2^T >= n, Q_0 = 1 - y g(x),
// Q_(s+1)(x^2, y) = Q_s(x, y) Q_s(-x, y) mod x^L, L = m / 2^s; Q_s has degree Y = 2^s in y and
// Q_s(x, 0) = Q_s(0, y) = 1. Level s is stored at stride 2L (x = z, y = z^(2L), x below L), and
// its products are transforms of length 4m: no carry from x into y, and y wraps mod y^(2Y).
// - Down (Graeffe): the leaf form of Q_s at 4m. Leaves 2p, 2p + 1 have the moduli z^8 -+ s,
//   s = r(p), and Q(-z) mod (z^8 - w) is the leaf with its odd coefficients negated, so
//   V = Q(z) Q(-z) mod (u^4 -+ s), u = z^2, is c(z) c(-z) per leaf (20 products, lc-opt's leaf
//   sums: 64-bit sums of products, one Montgomery reduction each, on x86 and Scalar; Neon: one
//   Montgomery product per term), and their CRT is leaf p of V's leaf form at 2m (s^2 = w_p). The
//   inverse at 2m gives V at stride L; row 0 = 1 + row 2Y (the wrap), x below L/2 kept: Q_(s+1).
// - Power projection: P_0 = x^(m-n), P_(s+1)(x^2) x = the odd part of P_s(x) Q_s(-x) mod x^L,
//   per leaf the odd coefficients of a(z) c(-z) (32 products), the same CRT; a[k] = P_T's row k.
// - compose is its transpose. With P^_(s+1) = P_(s+1) reversed in x and y (stride L in u = z^2),
//   R = P^(z^2) Q_s(-z) mod (z^4m - 1) holds P^_s in rows Y .. 2Y - 1, x below L: the next
//   input as it stands. P^_T = f reversed; h[k] = row 1 of level 0's R. Per level: the leaf form
//   of P^ at 2m, its leaves spread to those of P^(z^2) at 4m (lo +- s hi), one inverse_product
//   with the stored leaf form of Q_s(-z), upper half out.
// - Level T - 1 is one-dimensional: Q_(T-1) = 1 + x q(y), so compose's step is one cyclic product
//   of length m (P^_T q) and power projection's a = p1 - p0 q one product.
// Per level 12m transformed words and one pass of leaf sums; lc-opt's pruned transforms, fused
// bottoms and its other one-dimensional levels (0, 1, T - 3, T - 2) are not here (notes.md).
// Zen 3, n = 2^17: compose 42.4 ms against lc-opt's 29.5 and the frontend's plain version 176.
//
// compositional_inverse: Lagrange, (n-1) [x^(n-1)] f^k = k [x^(n-1-k)] (x / g)^(n-1), so with
// a = power_projection(f), h[j] = a[N-j] / (N-j) = [x^j] (x / g)^N / N for N = n - 1, and
// g / x = exp(-log(h / h[0]) / N) / f[1] mod x^N (series::log, series::exp).
//
// Memory, kept per thread across calls (the largest so far): compose about 16 m (T + 2) bytes
// (the leaf forms of Q_0 .. Q_(T-2), 4m words each: 35 MiB at m = 2^17, as lc-opt; 1.5 GiB at
// m = 2^22), power projection about 64 m bytes. Twiddle tables: one ntt::Transform per length,
// per thread.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/intrinsics.hpp"
#include "backend/ntt/memory.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/series/series.hpp"
#include "backend/simd/select.hpp"

namespace fps::composition {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 kP = ntt::kModulus;
inline constexpr std::size_t kMaxSize = std::size_t(1) << (ntt::kMaxLog - 2);  // 4m <= 2^kMaxLog

namespace detail {

using Span = std::span<const u32>;
using Out = std::span<u32>;

constexpr u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kP); }
constexpr u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
constexpr u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + kP - b; }
constexpr u32 kR = u32((u64(1) << 32) % kP);  // 2^32 mod P
constexpr u32 kR2 = mul(kR, kR);
constexpr u32 kHalf = (kP + 1) / 2;
inline u32 at(Span f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// Up to this many coefficients, the direct methods.
inline constexpr std::size_t kComposeBase = 16;
inline constexpr std::size_t kProjectionBase = 16;
// m = 2^T >= n, at least 64: transforms of m >= 2^kMinLog words, m / 4 leaf pairs (a multiple of 8).
inline std::size_t length_for(std::size_t n) { return std::max<std::size_t>(std::bit_ceil(n), 64); }

// ---- Memory -----------------------------------------------------------------------------------

// Words after each array: ntt::kPadding read by the transforms, and the rest so that arrays of a
// power-of-two length sit 320 bytes apart modulo 4 KiB (4K aliasing, backend/ntt/notes.md).
inline constexpr std::size_t kGap = 80;
constexpr std::size_t footprint(std::size_t words) { return (words + 15) / 16 * 16 + kGap; }

// This thread's memory, kept across calls: the largest region asked for so far.
struct Kept {
    ntt::detail::Region region;
    ~Kept() { ntt::detail::unmap_region(region); }
};
inline u32* thread_words(std::size_t words) {
    thread_local Kept kept;
    const std::size_t bytes = words * sizeof(u32);
    if (kept.region.bytes < bytes) {
        ntt::detail::unmap_region(std::exchange(kept.region, {}));
        kept.region = ntt::detail::map_region(bytes);
    }
    return reinterpret_cast<u32*>(kept.region.start);
}

// 64-byte aligned arrays carved from one span, in order.
class Arena {
public:
    explicit Arena(u32* start) : next_(start) {}
    u32* take(std::size_t words) {
        u32* p = next_;
        next_ += footprint(words);
        return p;
    }

private:
    u32* next_;
};

// This thread's transform of length 2^lg, kept across calls.
inline const ntt::Transform& transform(int lg) {
    thread_local std::array<std::unique_ptr<ntt::Transform>, ntt::kMaxLog + 1> cache;
    if (!cache[lg]) cache[lg] = std::make_unique<ntt::Transform>(lg);
    return *cache[lg];
}

// Per leaf pair p of a transform of 4m words (moduli z^8 -+ s, s = r(p)), for p < m / 4: s 2^32
// (Montgomery's form of s) and 2^64 / (2 s) (the CRT's factor, a Montgomery product taking its
// 2^-32 out). r(p) does not depend on the length: one table per thread, grown to the longest.
struct PairTables {
    std::vector<u32> s, crt;
};
inline const PairTables& pair_tables(std::size_t pairs) {
    thread_local PairTables t;
    if (t.s.size() < pairs) {
        const std::size_t old = t.s.size(), size = std::max(pairs, 2 * old);
        std::vector<u32> r(size), ri(size);
        r[0] = ri[0] = 1;
        for (std::size_t p = 1; p < size; ++p) {
            const int j = std::countr_zero(p);
            r[p] = mul(r[p & (p - 1)], ntt::detail::kRoots[0][j]);
            ri[p] = mul(ri[p & (p - 1)], ntt::detail::kRoots[1][j]);
        }
        t.s.resize(size), t.crt.resize(size);
        for (std::size_t p = 0; p < size; ++p) t.s[p] = mul(r[p], kR), t.crt[p] = mul(mul(ri[p], kHalf), kR2);
    }
    return t;
}

// ---- Leaf passes (lanes = leaf pairs) ---------------------------------------------------------

template <class S>
using Narrow = typename ntt::detail::intrinsics::Narrow<S>::type;

// L: a simd struct of 1, 4 or 8 lanes; lane l of a set is leaf pair p + l.
template <class L>
struct Pairs {
    using V = typename L::V;
    using M = modarith::Ops<L, kP>;
    static constexpr int W = L::lanes;
    using Set = V[8];

    // x[k] lane l = p[stride l + k] for k < 8, and back.
    [[gnu::always_inline]] static void load(const u32* p, std::size_t stride, Set& x) {
        if constexpr (W == 1) {
#pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) x[k] = p[k];
        } else {
#pragma GCC unroll 8
            for (int h = 0; h < 8; h += W) {
                V r[W];
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) r[u] = L::load(p + stride * std::size_t(u) + std::size_t(h));
                ntt::detail::intrinsics::Values<L>::transpose(r);
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) x[h + u] = r[u];
            }
        }
    }
    [[gnu::always_inline]] static void store(u32* p, std::size_t stride, const Set& x) {
        if constexpr (W == 1) {
#pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) p[k] = x[k];
        } else {
#pragma GCC unroll 8
            for (int h = 0; h < 8; h += W) {
                V r[W];
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) r[u] = x[h + u];
                ntt::detail::intrinsics::Values<L>::transpose(r);
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) L::store(p + stride * std::size_t(u) + std::size_t(h), r[u]);
            }
        }
    }
    // load(), and the leaves stored back with their odd coefficients negated (Q(-z)'s leaves).
    [[gnu::always_inline]] static void load_negating(u32* p, std::size_t stride, Set& x) {
        if constexpr (W == 1) {
#pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) {
                x[k] = p[k];
                if (k % 2) p[k] = x[k] ? kP - x[k] : 0;
            }
        } else {
            const V zero = L::broadcast(0);
#pragma GCC unroll 8
            for (int h = 0; h < 8; h += W) {
                V r[W];
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) {
                    u32* q = p + stride * std::size_t(u) + std::size_t(h);
                    r[u] = L::load(q);
                    L::store(q, L::template blend<0xaaaa>(r[u], M::sub(zero, r[u])));
                }
                ntt::detail::intrinsics::Values<L>::transpose(r);
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) x[h + u] = r[u];
            }
        }
    }

    [[gnu::always_inline]] static V red(V x) { return M::template reduce<kP>(x); }           // [0, 2P) -> [0, P)
    [[gnu::always_inline]] static V red4(V x) { return red(M::template reduce<2 * kP>(x)); }  // [0, 4P) -> [0, P)
    [[gnu::always_inline]] static V mont(V a, V b) { return red(M::montgomery(a, b)); }      // a b 2^-32, [0, P)

    // ---- Leaf sums in 64 bits (lc-opt's): x86 per 64-bit lane (mul_even, the even words, then the
    // odd words shifted down), and Scalar. Each output is one sum of products of canonical words,
    // the negative terms offset by 4 P^2, below 8 P^2, and one Montgomery reduction.
    static constexpr bool kX86 = requires(V v) { L::mul_even(v, v); };
    static constexpr bool kWide = kX86 || W == 1;
    using U = std::conditional_t<W == 1, u64, V>;  // a 64-bit sum (per 64-bit lane on x86)
    static constexpr u64 kOffset = 4 * u64(kP) * kP;
    static constexpr u32 kNegInverse = u32(0) - u32(modarith::inverse_mod_power_of_two(kP, 32));

    [[gnu::always_inline]] static U product(V a, V b) {
        if constexpr (W == 1) return u64(a) * b;
        else return L::mul_even(a, b);
    }
    [[gnu::always_inline]] static U plus(U a, U b) {
        if constexpr (W == 1) return a + b;
        else return L::add64(a, b);
    }
    [[gnu::always_inline]] static U minus(U a, U b) {
        if constexpr (W == 1) return a - b;
        else return L::sub64(a, b);
    }
    [[gnu::always_inline]] static U twice(U a) { return plus(a, a); }
    [[gnu::always_inline]] static U offset() {
        if constexpr (W == 1) return kOffset;
        else return L::broadcast64(kOffset);
    }
    // x 2^-32 mod P in [0, x / 2^32 + P) per 64-bit lane (Montgomery), from the sums of the even
    // words and of the odd words (x86); canonical.
    [[gnu::always_inline]] static V reduce_sums(U even, U odd) {
        if constexpr (W == 1) {
            return red4(V((even + u64(u32(even) * kNegInverse) * kP) >> 32));
        } else {
            const V ninv = L::broadcast(kNegInverse), p = L::broadcast(kP);
            even = L::add64(even, L::mul_even(L::mul_even(even, ninv), p));
            odd = L::add64(odd, L::mul_even(L::mul_even(odd, ninv), p));
            return red4(L::join_high(even, odd));
        }
    }
    // The odd words of each lane pair moved to the even ones (x86).
    [[gnu::always_inline]] static void odd_words(const Set& x, Set& out) {
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) out[i] = L::template shr64<32>(x[i]);
    }
    // tc[j] = t c[j] for j >= first (canonical), tR = t 2^32; the rest unset.
    [[gnu::always_inline]] static void scaled(const Set& c, V tR, int first, Set& tc) {
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) tc[j] = j < first ? L::broadcast(0) : mont(c[j], tR);
    }
    // Coefficient 2k of c(z) c(-z) mod (z^8 - t): sum over i + j = 2k of (-1)^j c_i c_j, and over
    // i + j = 2k + 8 of (-1)^j c_i tc_j (tc for j >= 4). 20 products.
    [[gnu::always_inline]] static void graeffe_sums(const Set& c, const Set& tc, U (&sum)[4]) {
        const auto x = [&c](int i, int j) { return product(c[i], c[j]); };
        const auto t = [&c, &tc](int i, int j) { return product(c[i], tc[j]); };
        sum[0] = minus(minus(plus(plus(x(0, 0), twice(t(2, 6))), plus(t(4, 4), offset())), twice(t(1, 7))), twice(t(3, 5)));
        sum[1] = minus(minus(plus(plus(twice(x(0, 2)), twice(t(4, 6))), offset()), x(1, 1)), plus(twice(t(3, 7)), t(5, 5)));
        sum[2] = minus(minus(plus(plus(twice(x(0, 4)), x(2, 2)), plus(t(6, 6), offset())), twice(x(1, 3))), twice(t(5, 7)));
        sum[3] = minus(minus(plus(plus(twice(x(0, 6)), twice(x(2, 4))), offset()), twice(x(1, 5))), plus(x(3, 3), t(7, 7)));
    }
    // Coefficient 2k + 1 of a(z) c(-z) mod (z^8 - t): sum over i + j = 2k + 1 of (-1)^j a_i c_j and
    // over i + j = 2k + 9 of (-1)^j a_i tc_j (tc for j >= 2). 32 products.
    [[gnu::always_inline]] static void odd_sums(const Set& a, const Set& c, const Set& tc, U (&sum)[4]) {
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            U s = offset();
#pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) {
                const int j = 2 * k + 1 - i, l = j + 8;
                if (j >= 0) s = j % 2 ? minus(s, product(a[i], c[j])) : plus(s, product(a[i], c[j]));
                else if (l < 8) s = l % 2 ? minus(s, product(a[i], tc[l])) : plus(s, product(a[i], tc[l]));
            }
            sum[k] = s;
        }
    }
    [[gnu::always_inline]] static void graeffe_wide(const Set& c, const Set& tc, V (&A)[4]) {
        U even[4], odd[4] = {};
        graeffe_sums(c, tc, even);
        if constexpr (kX86) {
            Set c2, tc2;
            odd_words(c, c2), odd_words(tc, tc2);
            graeffe_sums(c2, tc2, odd);
        }
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) A[k] = reduce_sums(even[k], odd[k]);
    }
    [[gnu::always_inline]] static void odd_wide(const Set& a, const Set& c, const Set& tc, V (&B)[4]) {
        U even[4], odd[4] = {};
        odd_sums(a, c, tc, even);
        if constexpr (kX86) {
            Set a2, c2, tc2;
            odd_words(a, a2), odd_words(c, c2), odd_words(tc, tc2);
            odd_sums(a2, c2, tc2, odd);
        }
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) B[k] = reduce_sums(even[k], odd[k]);
    }

    // A = c(z) c(-z) mod (z^8 - t) as a series in u = z^2 mod (u^4 - t), times 2^-32, canonical:
    // e^2 - u o^2 for c = e(u) + z o(u); tR = t 2^32 per lane.
    [[gnu::always_inline]] static void graeffe(const Set& c, V tR, V (&A)[4]) {
        if constexpr (kWide) {
            Set tc;
            scaled(c, tR, 4, tc);
            return graeffe_wide(c, tc, A);
        }
        const V e0 = c[0], e1 = c[2], e2 = c[4], e3 = c[6], o0 = c[1], o1 = c[3], o2 = c[5], o3 = c[7];
        const V te2 = mont(e2, tR), te3 = mont(e3, tR), to2 = mont(o2, tR), to3 = mont(o3, tR);
        const auto twice = [](V x) { return L::add(x, x); };
        const V E0 = red4(L::add(L::add(mont(e0, e0), twice(mont(e1, te3))), mont(e2, te2)));
        const V E1 = red4(twice(L::add(mont(e0, e1), mont(e2, te3))));
        const V E2 = red4(L::add(L::add(twice(mont(e0, e2)), mont(e1, e1)), mont(e3, te3)));
        const V E3 = red4(twice(L::add(mont(e0, e3), mont(e1, e2))));
        const V O0 = red4(L::add(L::add(mont(o0, o0), twice(mont(o1, to3))), mont(o2, to2)));
        const V O1 = red4(twice(L::add(mont(o0, o1), mont(o2, to3))));
        const V O2 = red4(L::add(L::add(twice(mont(o0, o2)), mont(o1, o1)), mont(o3, to3)));
        const V tO3 = red4(twice(L::add(mont(o0, to3), mont(o1, to2))));
        A[0] = M::sub(E0, tO3), A[1] = M::sub(E1, O0), A[2] = M::sub(E2, O1), A[3] = M::sub(E3, O2);
    }

    // The odd part of a(z) c(-z) mod (z^8 - t), divided by z, as a series in u mod (u^4 - t), times
    // 2^-32, canonical: po ce - pe co for a = pe(u) + z po(u), c = ce(u) + z co(u).
    [[gnu::always_inline]] static void odd_product(const Set& a, const Set& c, V tR, V (&B)[4]) {
        if constexpr (kWide) {
            Set tc;
            scaled(c, tR, 2, tc);
            return odd_wide(a, c, tc, B);
        }
        V ce[4], co[4], tce[4], tco[4];
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) ce[i] = c[2 * i], co[i] = c[2 * i + 1];
#pragma GCC unroll 3
        for (int i = 1; i < 4; ++i) tce[i] = mont(ce[i], tR), tco[i] = mont(co[i], tR);
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            V plus = L::broadcast(0), minus = L::broadcast(0);
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i) {
                const int j = k - i;
                const V po = a[2 * i + 1], pe = a[2 * i];
                plus = L::add(plus, j >= 0 ? mont(po, ce[j]) : mont(po, tce[j + 4]));
                minus = L::add(minus, j >= 0 ? mont(pe, co[j]) : mont(pe, tco[j + 4]));
            }
            B[k] = M::sub(red4(plus), red4(minus));
        }
    }

    // Leaf p of the transform of length 2m from A mod (u^4 - s) and B mod (u^4 + s), both times
    // 2^-32: (A + B) / 2 + u^4 (A - B) / (2 s), canonical.
    [[gnu::always_inline]] static void crt(const V (&A)[4], const V (&B)[4], V c, Set& out) {
        const V half = L::broadcast(mul(kHalf, kR2)), p = L::broadcast(kP);
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            out[k] = red(M::montgomery(L::add(A[k], B[k]), half));
            out[k + 4] = red(M::montgomery(L::add(L::sub(A[k], B[k]), p), c));
        }
    }

    // Graeffe: from F (4m words, the leaf form of Q) to G (2m words, the leaf form of V), pairs
    // = m / 4. negate: F's leaves are stored back as Q(-z)'s.
    static void graeffe_pass(u32* F, u32* G, std::size_t pairs, const PairTables& t, bool negate) {
        const V p = L::broadcast(kP);
        for (std::size_t q = 0; q < pairs; q += W) {
            Set a, b, out;
            if (negate) load_negating(F + 16 * q, 16, a), load_negating(F + 16 * q + 8, 16, b);
            else load(F + 16 * q, 16, a), load(F + 16 * q + 8, 16, b);
            const V s = L::load(t.s.data() + q);
            V A[4], B[4];
            graeffe(a, s, A);
            graeffe(b, L::sub(p, s), B);
            crt(A, B, L::load(t.crt.data() + q), out);
            store(G + 8 * q, 8, out);
        }
    }

    // Power projection: from FQ and FP (4m words, the leaf forms of Q and P) to GV and GW (2m
    // words, the leaf forms of V and of W = the odd part of P(x) Q(-x) over x). GV null: W only.
    static void projection_pass(const u32* FQ, const u32* FP, u32* GV, u32* GW, std::size_t pairs, const PairTables& t) {
        const V p = L::broadcast(kP);
        for (std::size_t q = 0; q < pairs; q += W) {
            Set ca, cb, pa, pb, out;
            load(FQ + 16 * q, 16, ca), load(FQ + 16 * q + 8, 16, cb);
            load(FP + 16 * q, 16, pa), load(FP + 16 * q + 8, 16, pb);
            const V s = L::load(t.s.data() + q), ns = L::sub(p, s), c = L::load(t.crt.data() + q);
            V A[4], B[4];
            if (GV) {
                graeffe(ca, s, A);
                graeffe(cb, ns, B);
                crt(A, B, c, out);
                store(GV + 8 * q, 8, out);
            }
            odd_product(pa, ca, s, A);
            odd_product(pb, cb, ns, B);
            crt(A, B, c, out);
            store(GW + 8 * q, 8, out);
        }
    }

    // compose's backward step: leaf p of FP (2m words, the leaf form of P(u) mod (u^8 - s^2),
    // lo + u^4 hi) to leaves 2p, 2p + 1 of A (4m words), P(z^2) mod (z^8 -+ s) = lo(u) +- s hi(u)
    // at the even coefficients, zeros at the odd ones.
    static void spread_pass(const u32* FP, u32* A, std::size_t pairs, const PairTables& t) {
        const V zero = L::broadcast(0);
        for (std::size_t q = 0; q < pairs; q += W) {
            Set x, a, b;
            load(FP + 8 * q, 8, x);
            const V s = L::load(t.s.data() + q);
#pragma GCC unroll 4
            for (int k = 0; k < 4; ++k) {
                const V hi = mont(x[k + 4], s);
                a[2 * k] = M::add(x[k], hi), b[2 * k] = M::sub(x[k], hi);
                a[2 * k + 1] = b[2 * k + 1] = zero;
            }
            store(A + 16 * q, 16, a);
            store(A + 16 * q + 8, 16, b);
        }
    }
};

using Native = Pairs<Narrow<simd::Native>>;

// The odd coefficients of each leaf of F (words) negated.
inline void negate_odd(u32* F, std::size_t words) {
    for (std::size_t i = 1; i < words; i += 2) F[i] = F[i] ? kP - F[i] : 0;
}

// From V at stride L (rows 0 .. 2Y - 1 in q[0, 2m), row 0 = the true row 0 = 1 plus row 2Y), the
// next level: rows 0 .. 2Y at stride L, x below L / 2 (the rest of rows 0 .. 2Y - 1 zero).
inline void next_level(u32* q, std::size_t m, std::size_t L) {
    const std::size_t half = L / 2, rows = 2 * m / L;
    std::copy_n(q, half, q + 2 * m);
    q[2 * m] = sub(q[2 * m], 1);
    q[0] = 1;
    std::fill_n(q + 1, half - 1, 0u);
    for (std::size_t j = 0; j < rows; ++j) std::fill_n(q + j * L + half, half, 0u);
}

// Q_0 = 1 - y g(x) at stride 2m, x below n: q[0, 2m + n).
inline std::size_t first_level(Span g, std::size_t n, std::size_t m, u32* q) {
    std::fill_n(q, 2 * m, 0u);
    q[0] = 1;
    q[2 * m] = 0;
    for (std::size_t i = 1; i < n; ++i) q[2 * m + i] = at(g, i) ? kP - at(g, i) : 0;
    return 2 * m + n;
}

// ---- Few coefficients --------------------------------------------------------------------------

// h = f(g) mod x^n by Horner's rule, n = h.size(): h = h g + f_i from the top coefficient down,
// so h[k] reads only h[0, k) (g[0] = 0).
inline void compose_direct(Span f, Span g, Out h) {
    const std::size_t n = h.size();
    std::fill(h.begin(), h.end(), 0u);
    for (std::size_t i = std::min(f.size(), n); i-- > 0;) {
        for (std::size_t k = n; k-- > 1;) {
            u64 sum = 0;
            for (std::size_t j = 1; j <= k && j < g.size(); ++j) sum += u64(g[j]) * h[k - j] % kP;
            h[k] = u32(sum % kP);
        }
        h[0] = f[i];
    }
}

// a[k] = [x^(n-1)] g^k for k < n = a.size(), from the powers of g.
inline void projection_direct(Span g, Out a) {
    const std::size_t n = a.size();
    std::vector<u32> pw(n, 0), next(n);
    pw[0] = 1;
    for (std::size_t k = 0; k < n; ++k) {
        a[k] = pw[n - 1];
        for (std::size_t i = 0; i < n; ++i) {
            u64 c = 0;
            for (std::size_t j = 1; j <= i && j < g.size(); ++j) c += u64(g[j]) * pw[i - j] % kP;
            next[i] = u32(c % kP);
        }
        std::swap(pw, next);
    }
}

// ---- Kinoshita and Li ---------------------------------------------------------------------------

// Q_(T-1) = 1 + x q(y) at stride 4 in q (x below 2, rows 0 .. m/2): q(y) into col (m words).
inline void last_column(const u32* q, std::size_t m, u32* col) {
    for (std::size_t b = 0; b <= m / 2; ++b) col[b] = q[4 * b + 1];
    std::fill_n(col + m / 2 + 1, m / 2 - 1, 0u);
}

inline void compose_levels(Span f, Span g, Out h) {
    const std::size_t n = h.size(), m = length_for(n), pairs = m / 4;
    const int T = std::countr_zero(m);
    const ntt::Transform &t4 = transform(T + 2), &t2 = transform(T + 1), &t1 = transform(T);
    const PairTables& tables = pair_tables(pairs);
    const std::size_t words = std::size_t(T - 1) * footprint(4 * m) + 3 * footprint(4 * m) + 2 * footprint(2 * m) +
                              footprint(3 * m) + 16;
    Arena arena(thread_words(words));
    std::vector<u32*> levels(static_cast<std::size_t>(T - 1));
    for (auto& F : levels) F = arena.take(4 * m);
    u32 *const q = arena.take(3 * m), *const G = arena.take(2 * m), *const p = arena.take(2 * m);
    u32 *const FP = arena.take(4 * m), *const A = arena.take(4 * m), *const work = arena.take(4 * m);

    // Down: the leaf forms of Q_0 .. Q_(T-2) at 4m, stored as Q_s(-z)'s; Q_(T-1) = 1 + x q(y)
    // one-dimensional: the leaf form of q at m, in FP.
    std::size_t size = first_level(g, n, m, q);
    for (int s = 0; s + 1 < T; ++s) {
        const std::size_t L = m >> s;
        t4.forward(Span(q, size), levels[std::size_t(s)]);
        Native::graeffe_pass(levels[std::size_t(s)], G, pairs, tables, true);
        t2.inverse(G, Out(q, 2 * m));
        next_level(q, m, L);
        size = 2 * m + L / 2;
    }
    last_column(q, m, G);
    t1.forward(Span(G, m / 2 + 1), FP);
    // Up. Level T - 1: P^_T = f reversed (m words in y), R = P^ (1 - x q): P^_(T-1) has columns
    // f reversed and -(P^ q mod (y^m - 1)), rows m/2 .. m - 1, at stride 4 in p.
    for (std::size_t b = 0; b < m; ++b) A[b] = at(f, m - 1 - b);
    t1.cyclic_product(Span(A, m), FP, work, Out(G, m / 2), {.offset = m / 2, .negate = true});
    std::fill_n(p, 2 * m, 0u);
    for (std::size_t b = 0; b < m / 2; ++b) p[4 * b] = A[m / 2 + b], p[4 * b + 1] = G[b];
    // Levels T - 2 .. 0: R = P^(z^2) Q_s(-z), rows Y .. 2Y - 1, x below L.
    for (int s = T - 2; s >= 0; --s) {
        const std::size_t L = m >> s, Y = std::size_t(1) << s;
        t2.forward(Span(p, 2 * m), FP);
        Native::spread_pass(FP, A, pairs, tables);
        const ntt::Term terms[1] = {{A, levels[std::size_t(s)]}};
        if (s == 0) {
            t4.inverse_product(terms, work, h, {.offset = 2 * m});
            break;
        }
        t4.inverse_product(terms, work, Out(p, 2 * m), {.offset = 2 * m});
        for (std::size_t j = 0; j < Y; ++j) std::fill_n(p + 2 * L * j + L, L, 0u);
    }
}

inline void projection_levels(Span g, Out a) {
    const std::size_t n = a.size(), m = length_for(n), pairs = m / 4;
    const int T = std::countr_zero(m);
    const ntt::Transform &t4 = transform(T + 2), &t2 = transform(T + 1);
    const PairTables& tables = pair_tables(pairs);
    const std::size_t words = 2 * footprint(4 * m) + 2 * footprint(2 * m) + footprint(3 * m) + footprint(2 * m) + 16;
    Arena arena(thread_words(words));
    u32 *const FQ = arena.take(4 * m), *const FP = arena.take(4 * m), *const GV = arena.take(2 * m);
    u32 *const GW = arena.take(2 * m), *const q = arena.take(3 * m), *const p = arena.take(2 * m);

    std::size_t size = first_level(g, n, m, q);
    std::fill_n(p, 2 * m, 0u);
    p[m - n] = 1;  // P_0 = x^(m - n)
    for (int s = 0; s + 1 < T; ++s) {
        const std::size_t L = m >> s, Y = std::size_t(1) << s;
        t4.forward(Span(q, size), FQ);
        t4.forward(Span(p, 2 * m), FP);
        Native::projection_pass(FQ, FP, GV, GW, pairs, tables);
        t2.inverse(GW, Out(p, 2 * m));
        for (std::size_t j = 0; j < 2 * Y; ++j) std::fill_n(p + L * j + L / 2, L / 2, 0u);
        t2.inverse(GV, Out(q, 2 * m));
        next_level(q, m, L);
        size = 2 * m + L / 2;
    }
    // Level T - 1, one-dimensional: P = p0(y) + x p1(y) (stride 4, rows below m/2), Q = 1 + x q(y);
    // a = p1 - p0 q (degree below m).
    u32 *const col = GV, *const p0 = FQ, *const prod = FP;
    last_column(q, m, col);
    for (std::size_t b = 0; b < m / 2; ++b) p0[b] = p[4 * b];
    ntt::multiply(Span(p0, m / 2), Span(col, m / 2 + 1), Out(prod, n));
    for (std::size_t k = 0; k < n; ++k) a[k] = sub(k < m / 2 ? p[4 * k + 1] : 0, prod[k]);
}

inline void check_length(std::size_t n, const char* what) {
    if (n > kMaxSize) throw std::length_error(what);
}

}  // namespace detail

// out = f(g) mod x^n, n = out.size(); g[0] counts as 0.
inline void compose(std::span<const u32> f, std::span<const u32> g, std::span<u32> out) {
    const std::size_t n = out.size();
    detail::check_length(n, "fps::composition::compose: n past kMaxSize");
    if (n == 0) return;
    f = f.first(std::min(f.size(), n)), g = g.first(std::min(g.size(), n));
    if (n <= detail::kComposeBase) return detail::compose_direct(f, g, out);
    detail::compose_levels(f, g, out);
}

// out[k] = [x^(n-1)] g^k for k < n = out.size(); g[0] counts as 0.
inline void power_projection(std::span<const u32> g, std::span<u32> out) {
    const std::size_t n = out.size();
    detail::check_length(n, "fps::composition::power_projection: n past kMaxSize");
    if (n == 0) return;
    g = g.first(std::min(g.size(), n));
    if (n <= detail::kProjectionBase) return detail::projection_direct(g, out);
    detail::projection_levels(g, out);
}

// out = g with f(g) = g(f) = x mod x^n, n = out.size(); f[0] counts as 0, f[1] != 0, n <= P.
inline void compositional_inverse(std::span<const u32> f, std::span<u32> out) {
    using namespace detail;
    const std::size_t n = out.size();
    check_length(n, "fps::composition::compositional_inverse: n past kMaxSize");
    if (n == 0) return;
    out[0] = 0;
    if (n == 1) return;
    const u32 inverse1 = power(at(f, 1), kP - 2);
    if (n == 2) {
        out[1] = inverse1;
        return;
    }
    // h[j] = a[N - j] / (N - j) for j < N, a = power_projection(f); then g / x = exp(-log(h /
    // h[0]) / N) / f[1] mod x^N.
    const std::size_t N = n - 1;
    std::vector<u32> a(n), inv(n), h(N);
    power_projection(f.first(std::min(f.size(), n)), a);
    inv[1] = 1;
    for (std::size_t i = 2; i < n; ++i) inv[i] = mul(kP - kP / u32(i), inv[kP % i]);
    for (std::size_t j = 0; j < N; ++j) h[j] = mul(a[N - j], inv[N - j]);
    const u32 c = power(h[0], kP - 2), e = kP - inv[N];
    for (auto& x : h) x = mul(x, c);
    series::log(h, h);
    for (auto& x : h) x = mul(x, e);
    series::exp(h, out.subspan(1));
    for (std::size_t j = 1; j < n; ++j) out[j] = mul(out[j], inverse1);
}

}  // namespace fps::composition
