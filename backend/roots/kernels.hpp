// The root finder's loops, as templates on a backend/simd struct L for rows of 8 words (one vector
// of 8 lanes, or 8 / L::lanes vectors: product_tree::Lanes per target): the tangent Graeffe step
// on pairs of leaves, the folds of the coset evaluation, zero masks and powers of 3. 64-bit sums of
// products with one Montgomery step (product_tree::detail::Sums: mul_even on x86, umull/umlal on
// Neon, u64 on Scalar). Words below P; "Montgomery form" is x 2^32 mod P. Design: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/product_tree/kernels.hpp"
#include "backend/roots/select.hpp"

namespace fps::root_finding::detail {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using product_tree::detail::canonical4;
using product_tree::detail::kP;
using product_tree::detail::kR;
using product_tree::detail::montgomery;
using product_tree::detail::montgomery_canonical;
using product_tree::detail::multiply_mod;
using product_tree::detail::Ops;
using product_tree::detail::power;
using product_tree::detail::Sums;

constexpr u32 add_mod(u32 x, u32 y) { return x >= kP - y ? x - (kP - y) : x + y; }
constexpr u32 to_montgomery(u32 x) { return u32((u64(x) << 32) % kP); }
constexpr u32 inverse_mod(u32 x) { return power(x, kP - 2); }

// ---- Graeffe step ---------------------------------------------------------------------------

// x[j], j < 8: lane t holds word j of the 8 words at p + stride t, t < L::lanes.
template <class L>
[[gnu::always_inline]] inline void load_columns(const u32* p, std::size_t stride, typename L::V (&x)[8]) {
    constexpr int W = L::lanes;
#pragma GCC unroll 8
    for (int u = 0; u < 8 / W; ++u) {
        typename L::V r[W];
#pragma GCC unroll 8
        for (int t = 0; t < W; ++t) r[t] = L::load(p + stride * t + u * W);
        product_tree::detail::transpose<L>(r);
#pragma GCC unroll 8
        for (int i = 0; i < W; ++i) x[u * W + i] = r[i];
    }
}

// Lanes: leaves a and b mod (x^8 - w), lane t of a[j] and b[j] holding coefficient j of one leaf.
// With a = E(y) + x O(y) and b = F(y) + x G(y), y = x^2, the even parts mod (y^4 - w):
//   E^2 - y O^2 = u + w v,  E F - y O G = s + w t   (4 coefficients each),
// out = u, v, s, t, canonical, times 2^-32 (lc-opt's even_parts).
template <class L>
[[gnu::always_inline]] inline void even_parts(const typename L::V (&a)[8], const typename L::V (&b)[8], typename L::V (&out)[16]) {
    using S = Sums<L>;
    using V = typename L::V;
    const V p = L::broadcast(kP);
    const auto twice = [](V x) { return S::of(L::add(x, x)); };    // < 2P
    const auto minus = [p](V x) { return S::of(L::sub(p, x)); };  // in (0, P]
    const auto e0 = S::of(a[0]), e1 = S::of(a[2]), e2 = S::of(a[4]), e3 = S::of(a[6]);
    const auto o1 = S::of(a[3]), o2 = S::of(a[5]), o3 = S::of(a[7]), o0 = S::of(a[1]);
    const auto n0 = minus(a[1]), n1 = minus(a[3]), n2 = minus(a[5]), n3 = minus(a[7]);
    const auto d1 = twice(a[2]), d2 = twice(a[4]), d3 = twice(a[6]);
    const auto q1 = twice(a[3]), q2 = twice(a[5]), q3 = twice(a[7]);
    const auto sum = [](auto first, auto... rest) {
        auto s = S::mul(first.first, first.second);
        (S::mac(s, rest.first, rest.second), ...);
        return S::reduce(s);
    };
    using Pair = std::pair<typename S::Op, typename S::Op>;
    // E^2 - y O^2: products below 2 P^2, at most 4 per sum (< 9 P^2, Sums' bound).
    out[0] = sum(Pair{e0, e0});
    out[1] = sum(Pair{e0, d1}, Pair{n0, o0});
    out[2] = sum(Pair{e0, d2}, Pair{e1, e1}, Pair{n0, q1});
    out[3] = sum(Pair{e0, d3}, Pair{e1, d2}, Pair{n0, q2}, Pair{n1, o1});
    out[4] = sum(Pair{e1, d3}, Pair{e2, e2}, Pair{n0, q3}, Pair{n1, q2});
    out[5] = sum(Pair{e2, d3}, Pair{n1, q3}, Pair{n2, o2});
    out[6] = sum(Pair{e3, e3}, Pair{n2, q3});
    out[7] = sum(Pair{n3, o3});
    // E F - y O G: products at most P^2, at most 7 per sum.
    const auto f0 = S::of(b[0]), f1 = S::of(b[2]), f2 = S::of(b[4]), f3 = S::of(b[6]);
    const auto g0 = S::of(b[1]), g1 = S::of(b[3]), g2 = S::of(b[5]), g3 = S::of(b[7]);
    out[8] = sum(Pair{e0, f0});
    out[9] = sum(Pair{e0, f1}, Pair{e1, f0}, Pair{n0, g0});
    out[10] = sum(Pair{e0, f2}, Pair{e1, f1}, Pair{e2, f0}, Pair{n0, g1}, Pair{n1, g0});
    out[11] = sum(Pair{e0, f3}, Pair{e1, f2}, Pair{e2, f1}, Pair{e3, f0}, Pair{n0, g2}, Pair{n1, g1}, Pair{n2, g0});
    out[12] = sum(Pair{e1, f3}, Pair{e2, f2}, Pair{e3, f1}, Pair{n0, g3}, Pair{n1, g2}, Pair{n2, g1}, Pair{n3, g0});
    out[13] = sum(Pair{e2, f3}, Pair{e3, f2}, Pair{n1, g3}, Pair{n2, g2}, Pair{n3, g1});
    out[14] = sum(Pair{e3, f3}, Pair{n2, g3}, Pair{n3, g2});
    out[15] = sum(Pair{n3, g3});
}

// One tangent Graeffe step on the leaf forms a and b of length 2n (n / 4 leaves each): leaves 2q
// and 2q + 1 (weights r = r(q) and -r) give leaf q (weight r^2) of the leaf form of length n of
// the result, written over a[0, n) and b[0, n). With C = u + r v on leaf 2q and C' = u' - r v' on
// leaf 2q + 1, the result mod (y^8 - r^2) is (C + C') + y^4 (C - C') / r, without the factor 1/2.
// weights[q] = r(q) 2^32 and inverse_weights[q] = r(q)^-1 2^32 mod P for q < n / 8; n / 8 a
// multiple of L::lanes.
// even_parts for x86 (Sums' EvenOdd), from columns in memory: column j (a word per pair) at cols
// + 16 j, 8 words and 8 more that the odd words' loads read (one word on). Each lane parity runs
// its own pass of 64-bit sums, so an operand is one register or a memory operand, not two: GCC
// spilled 678 times per 8 pairs with all of even_parts' operands live (#122). Columns: 0-7 a,
// 8-15 b, 16-19 P - a odd (n), 20-22 2 a even (d1-d3), 23-25 2 a odd (q1-q3).
template <class L>
[[gnu::always_inline]] inline void even_parts_columns(const u32* cols, typename L::V (&out)[16]) {
    using V = typename L::V;
    constexpr int W = L::lanes;
    static_assert(W == 8);
    enum : int { e0 = 0, o0 = 1, e1 = 2, o1 = 3, e2 = 4, o2 = 5, e3 = 6, o3 = 7 };
    enum : int { f0 = 8, g0 = 9, f1 = 10, g1 = 11, f2 = 12, g2 = 13, f3 = 14, g3 = 15 };
    enum : int { n0 = 16, n1 = 17, n2 = 18, n3 = 19, d1 = 20, d2 = 21, d3 = 22, q1 = 23, q2 = 24, q3 = 25 };
    static constexpr int kTerms[16][7][2] = {
        {{e0, e0}},
        {{e0, d1}, {n0, o0}},
        {{e0, d2}, {e1, e1}, {n0, q1}},
        {{e0, d3}, {e1, d2}, {n0, q2}, {n1, o1}},
        {{e1, d3}, {e2, e2}, {n0, q3}, {n1, q2}},
        {{e2, d3}, {n1, q3}, {n2, o2}},
        {{e3, e3}, {n2, q3}},
        {{n3, o3}},
        {{e0, f0}},
        {{e0, f1}, {e1, f0}, {n0, g0}},
        {{e0, f2}, {e1, f1}, {e2, f0}, {n0, g1}, {n1, g0}},
        {{e0, f3}, {e1, f2}, {e2, f1}, {e3, f0}, {n0, g2}, {n1, g1}, {n2, g0}},
        {{e1, f3}, {e2, f2}, {e3, f1}, {n0, g3}, {n1, g2}, {n2, g1}, {n3, g0}},
        {{e2, f3}, {e3, f2}, {n1, g3}, {n2, g2}, {n3, g1}},
        {{e3, f3}, {n2, g3}, {n3, g2}},
        {{n3, g3}},
    };
    static constexpr int kCount[16] = {1, 2, 3, 4, 4, 3, 2, 1, 1, 3, 5, 7, 7, 5, 3, 1};
    const V ninv = L::broadcast(product_tree::detail::kNegInverse), p = L::broadcast(kP);
    alignas(64) u32 half[2][16][8];  // the parities' reduced sums: results in the high words
#pragma GCC unroll 2
    for (int parity = 0; parity < 2; ++parity)
#pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            const auto column = [&](int j) { return L::load(cols + 16 * j + parity); };
            V s = L::mul_even(column(kTerms[i][0][0]), column(kTerms[i][0][1]));
#pragma GCC unroll 7
            for (int t = 1; t < kCount[i]; ++t) s = L::add64(s, L::mul_even(column(kTerms[i][t][0]), column(kTerms[i][t][1])));
            L::store(half[parity][i], L::add64(s, L::mul_even(L::mul_even(s, ninv), p)));
        }
#pragma GCC unroll 16
    for (int i = 0; i < 16; ++i) out[i] = canonical4<L>(L::join_high(L::load(half[0][i]), L::load(half[1][i])));
}

template <class L>
void pair_step(u32* a, u32* b, std::size_t n, const u32* weights, const u32* inverse_weights) {
    using V = typename L::V;
    using M = Ops<L>;
    constexpr int W = L::lanes;
    constexpr bool kColumns = kPairColumns && std::is_same_v<Sums<L>, product_tree::detail::EvenOdd<L>> && W == 8;
    const V p = L::broadcast(kP);
    for (std::size_t q = 0; q < n / 8; q += W) {
        alignas(64) u32 c[2][16][W];  // even_parts of the even and the odd leaves, through memory
        if constexpr (kColumns) {
            alignas(64) u32 cols[26 * 16];
#pragma GCC unroll 2
            for (int s = 0; s < 2; ++s) {
                V x[8], y[8];  // lane t: pair q + t
                load_columns<L>(a + 16 * q + 8 * s, 16, x);
                load_columns<L>(b + 16 * q + 8 * s, 16, y);
#pragma GCC unroll 8
                for (int j = 0; j < 8; ++j) L::store(cols + 16 * j, x[j]), L::store(cols + 16 * (8 + j), y[j]);
#pragma GCC unroll 4
                for (int j = 0; j < 4; ++j) L::store(cols + 16 * (16 + j), L::sub(p, x[2 * j + 1]));
#pragma GCC unroll 3
                for (int j = 1; j < 4; ++j) {
                    L::store(cols + 16 * (19 + j), L::add(x[2 * j], x[2 * j]));
                    L::store(cols + 16 * (22 + j), L::add(x[2 * j + 1], x[2 * j + 1]));
                }
                V e[16];
                even_parts_columns<L>(cols, e);
#pragma GCC unroll 16
                for (int i = 0; i < 16; ++i) L::store(c[s][i], e[i]);
            }
        } else {
#pragma GCC unroll 2
            for (int s = 0; s < 2; ++s) {
                V x[8], y[8], e[16];  // lane t: pair q + t
                load_columns<L>(a + 16 * q + 8 * s, 16, x);
                load_columns<L>(b + 16 * q + 8 * s, 16, y);
                even_parts<L>(x, y, e);
#pragma GCC unroll 16
                for (int i = 0; i < 16; ++i) L::store(c[s][i], e[i]);
            }
        }
        const V r = L::load(weights + q), r_inverse = L::load(inverse_weights + q);
        V out[2][8];
#pragma GCC unroll 2
        for (int f = 0; f < 2; ++f)
#pragma GCC unroll 4
            for (int k = 0; k < 4; ++k) {
                const V u0 = L::load(c[0][8 * f + k]), v0 = L::load(c[0][8 * f + 4 + k]);
                const V u1 = L::load(c[1][8 * f + k]), v1 = L::load(c[1][8 * f + 4 + k]);
                const V lo = L::add(M::add(u0, u1), M::montgomery(L::add(L::sub(v0, v1), p), r));
                const V hi = L::add(M::montgomery(L::add(L::sub(u0, u1), p), r_inverse), M::add(v0, v1));
                out[f][k] = canonical4<L>(lo), out[f][4 + k] = canonical4<L>(hi);  // < 3P
            }
        product_tree::detail::store_columns<L>(a + 8 * q, out[0]);
        product_tree::detail::store_columns<L>(b + 8 * q, out[1]);
    }
}

// ---- Wide sums --------------------------------------------------------------------------------

// Sums<L>::Acc: 64-bit sums of the 8 lanes' products (one per lane), added and folded. folded(s):
// 2^32 h + l -> h (2^32 mod P) + l, the same residue below 2^60.2. Sums::reduce maps a sum below
// 3 P 2^32 (2^63.4) to its residue times 2^-32, canonical.
template <class L>
struct Wide {
    using S = Sums<L>;
    using Acc = typename S::Acc;

    [[gnu::always_inline]] static Acc add(const Acc& x, const Acc& y) {
        if constexpr (std::is_same_v<S, product_tree::detail::EvenOdd<L>>) {
            return {L::add64(x.even, y.even), L::add64(x.odd, y.odd)};
#if defined(__ARM_NEON) && defined(__aarch64__)
        } else if constexpr (std::is_same_v<S, product_tree::detail::Halves<L>>) {
            return {vaddq_u64(x.lo, y.lo), vaddq_u64(x.hi, y.hi)};
#endif
        } else {
            return x + y;
        }
    }

    [[gnu::always_inline]] static Acc folded(const Acc& x) {
        if constexpr (std::is_same_v<S, product_tree::detail::EvenOdd<L>>) {
            using V = typename L::V;
            constexpr unsigned kOddWords = 0xAAAAu & ((1u << L::lanes) - 1);
            const V r = L::broadcast(kR), zero = L::broadcast(0);
            const auto fold = [&](V s) {
                return L::add64(L::mul_even(L::template shr64<32>(s), r), L::template blend<kOddWords>(s, zero));
            };
            return {fold(x.even), fold(x.odd)};
#if defined(__ARM_NEON) && defined(__aarch64__)
        } else if constexpr (std::is_same_v<S, product_tree::detail::Halves<L>>) {
            const auto fold = [](uint64x2_t s) {
                return vmlal_u32(vandq_u64(s, vdupq_n_u64(0xffffffffu)), vshrn_n_u64(s, 32), vdup_n_u32(kR));
            };
            return {fold(x.lo), fold(x.hi)};
#endif
        } else {
            return (x >> 32) * kR + u32(x);
        }
    }
};

// ---- Zeros and points -----------------------------------------------------------------------

// Bit l set where word l of p[0, 8) is 0.
inline unsigned zero_mask(const u32* p) {
#if defined(__AVX2__)
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(x, _mm256_setzero_si256()))));
#elif defined(__SSE4_1__)
    const __m128i zero = _mm_setzero_si128();
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 4));
    return unsigned(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(lo, zero)))) |
           unsigned(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(hi, zero)))) << 4;
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const uint32x4_t lo = vld1q_u32(p), hi = vld1q_u32(p + 4);
    if (vminvq_u32(vminq_u32(lo, hi)) != 0) return 0;
    const uint32x4_t bits = {1, 2, 4, 8};
    return vaddvq_u32(vandq_u32(vceqzq_u32(lo), bits)) | vaddvq_u32(vandq_u32(vceqzq_u32(hi), bits)) << 4;
#else
    unsigned m = 0;
    for (int l = 0; l < 8; ++l) m |= unsigned(p[l] == 0) << l;
    return m;
#endif
}

// Sorts p[0, n), words below 2^30, increasing: LSD radix sort of 3 passes of 10 bits through tmp
// (n words); std::sort below 256 words. At 4000 roots std::sort took about 1/6 of the root finding.
inline void sort_words(u32* p, std::size_t n, u32* tmp) {
    if (n < 256) return std::sort(p, p + n);
    std::size_t count[3][1024] = {};
    for (std::size_t i = 0; i < n; ++i) {
        const u32 x = p[i];
        ++count[0][x & 1023], ++count[1][x >> 10 & 1023], ++count[2][x >> 20 & 1023];
    }
    for (auto& c : count)
        for (std::size_t v = 0, sum = 0; v < 1024; ++v) {
            const std::size_t k = c[v];
            c[v] = sum, sum += k;
        }
    for (std::size_t i = 0; i < n; ++i) tmp[count[0][p[i] & 1023]++] = p[i];
    for (std::size_t i = 0; i < n; ++i) p[count[1][tmp[i] >> 10 & 1023]++] = tmp[i];
    for (std::size_t i = 0; i < n; ++i) tmp[count[2][p[i] >> 20 & 1023]++] = p[i];
    std::copy_n(tmp, n, p);
}

// 3^(i 2^(10 k)) 2^32 mod P for i < 1024, k < 3: 3^e for e < 2^30 is a product of three entries.
struct Power3Tables {
    u32 t[3][1024];
};
inline constexpr Power3Tables kPower3 = [] {
    Power3Tables s{};
    u32 base = 3;
    for (int k = 0; k < 3; ++k) {
        u32 x = 1;
        for (int i = 0; i < 1024; ++i, x = multiply_mod(x, base)) s.t[k][i] = to_montgomery(x);
        base = x;  // 3^(2^(10 (k + 1)))
    }
    return s;
}();

// 3^e 2^32 mod P (Montgomery form), e < 2^30.
inline u32 power3(u32 e) {
    return montgomery(montgomery(kPower3.t[0][e & 1023], kPower3.t[1][e >> 10 & 1023]), kPower3.t[2][e >> 20]);
}

}  // namespace fps::root_finding::detail
