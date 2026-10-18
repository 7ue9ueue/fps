// The product tree's kernels, as templates on backend/simd structs: 64-bit sums of products with
// one Montgomery step, the lanes' schoolbook, the standard layout's leaf products, pointwise
// products and conversions. Words below P; "Montgomery form" is x 2^32 mod P.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/common.hpp"
#include "backend/product_tree/select.hpp"

namespace fps::product_tree::detail {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 kP = ntt::kModulus;
inline constexpr u32 kNegInverse = u32(0) - u32(modarith::inverse_mod_power_of_two(kP, 32));  // -P^-1 mod 2^32

constexpr u32 multiply_mod(u32 x, u32 y) { return u32(u64(x) * y % kP); }

constexpr u32 power(u32 x, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, x = multiply_mod(x, x))
        if (e & 1) r = multiply_mod(r, x);
    return r;
}

inline constexpr u32 kR = ntt::detail::kR;              // 2^32 mod P: 1 in Montgomery form
inline constexpr u32 kR2 = multiply_mod(kR, kR);        // montgomery(x, kR2) = x 2^32
inline constexpr u32 kRInverse = power(kR, kP - 2);

// x y 2^-32 mod P, canonical, for x y < 2^32 P.
constexpr u32 montgomery(u32 x, u32 y) {
    const u64 s = u64(x) * y;
    const u32 r = u32((s + u64(u32(s) * kNegInverse) * kP) >> 32);  // < 2P
    return r >= kP ? r - kP : r;
}

template <class S>
using Ops = modarith::Ops<S, kP>;

// x < 4P -> [0, P).
template <class S>
[[gnu::always_inline]] inline typename S::V canonical4(typename S::V x) {
    return Ops<S>::template reduce<kP>(Ops<S>::template reduce<2 * kP>(x));
}

// x y 2^-32, canonical; x, y < P.
template <class S>
[[gnu::always_inline]] inline typename S::V montgomery_canonical(typename S::V x, typename S::V y) {
    return Ops<S>::template reduce<kP>(Ops<S>::montgomery(x, y));
}

// Sums of at most kMaxTerms products of words < P (< 9 P^2 < 2^63), then one Montgomery step:
// (s + m P) / 2^32 < 9 P^2 / 2^32 + P < 3.1 P, then canonical. An Op is one operand: on x86 its
// words and its odd words moved to the even ones (mul_even); Neon: umull / umlal on halves.
inline constexpr int kMaxTerms = 9;

template <class L>
struct EvenOdd {
    using V = typename L::V;
    struct Op {
        V even, odd;
    };
    struct Acc {
        V even, odd;
    };
    [[gnu::always_inline]] static Op of(V x) { return {x, L::template shr64<32>(x)}; }
    // Reads p[lanes]: the odd words from the load one word on.
    [[gnu::always_inline]] static Op load(const u32* p) { return {L::load(p), L::load(p + 1)}; }
    [[gnu::always_inline]] static Op broadcast(u32 x) {
        const V v = L::broadcast(x);
        return {v, v};
    }
    [[gnu::always_inline]] static Acc mul(const Op& a, const Op& b) {
        return {L::mul_even(a.even, b.even), L::mul_even(a.odd, b.odd)};
    }
    [[gnu::always_inline]] static void mac(Acc& s, const Op& a, const Op& b) {
        s.even = L::add64(s.even, L::mul_even(a.even, b.even));
        s.odd = L::add64(s.odd, L::mul_even(a.odd, b.odd));
    }
    [[gnu::always_inline]] static V reduce(const Acc& s) {
        const V ninv = L::broadcast(kNegInverse), p = L::broadcast(kP);
        const V e = L::add64(s.even, L::mul_even(L::mul_even(s.even, ninv), p));
        const V o = L::add64(s.odd, L::mul_even(L::mul_even(s.odd, ninv), p));
        return canonical4<L>(L::join_high(e, o));
    }
};

template <class L>
struct Halves {
    using V = typename L::V;
    using W = typename L::W;
    using Op = V;
    struct Acc {
        W lo, hi;
    };
    [[gnu::always_inline]] static Op of(V x) { return x; }
    [[gnu::always_inline]] static Op load(const u32* p) { return L::load(p); }
    [[gnu::always_inline]] static Op broadcast(u32 x) { return L::broadcast(x); }
    [[gnu::always_inline]] static Acc mul(Op a, Op b) { return {L::mul_wide_lo(a, b), L::mul_wide_hi(a, b)}; }
    [[gnu::always_inline]] static void mac(Acc& s, Op a, Op b) {
        s.lo = L::mla_wide_lo(s.lo, a, b);
        s.hi = L::mla_wide_hi(s.hi, a, b);
    }
    [[gnu::always_inline]] static V reduce(const Acc& s) {
        const V m = L::mul_lo(L::lo_halves(s.lo, s.hi), L::broadcast(kNegInverse)), p = L::broadcast(kP);
        return canonical4<L>(L::hi_halves(L::mla_wide_lo(s.lo, m, p), L::mla_wide_hi(s.hi, m, p)));
    }
};

template <class L>
struct Single {
    using V = u32;
    using Op = u32;
    using Acc = u64;
    [[gnu::always_inline]] static Op of(V x) { return x; }
    [[gnu::always_inline]] static Op load(const u32* p) { return *p; }
    [[gnu::always_inline]] static Op broadcast(u32 x) { return x; }
    [[gnu::always_inline]] static Acc mul(Op a, Op b) { return u64(a) * b; }
    [[gnu::always_inline]] static void mac(Acc& s, Op a, Op b) { s += u64(a) * b; }
    [[gnu::always_inline]] static V reduce(Acc s) {
        const u32 r = u32((s + u64(u32(s) * kNegInverse) * kP) >> 32);
        return canonical4<L>(r);
    }
};

template <class L>
using Sums = std::conditional_t<requires(typename L::V v) { L::mul_even(v, v); }, EvenOdd<L>,
                                std::conditional_t<requires(typename L::V v) { L::mul_wide_lo(v, v); }, Halves<L>, Single<L>>>;

// Lanes: 8 polynomials side by side, word 8 j + l = coefficient j of polynomial l.

// c = a b, lane by lane, for a of degree m and b of degree k (m + 1 and k + 1 coefficients of 8
// words), min(m, k) < kMaxTerms, m + k <= 32; a, b in Montgomery form, c too (canonical).
template <class L>
[[gnu::always_inline]] inline void multiply_lanes_inline(const u32* a, std::size_t m, const u32* b, std::size_t k, u32* c) {
    using S = Sums<L>;
    constexpr int kVectors = 8 / L::lanes;
#pragma GCC unroll 8
    for (int u = 0; u < kVectors; ++u) {
        typename S::Op x[33], y[33];
#pragma GCC unroll 33
        for (std::size_t i = 0; i <= m; ++i) x[i] = S::of(L::load(a + 8 * i + u * L::lanes));
#pragma GCC unroll 33
        for (std::size_t j = 0; j <= k; ++j) y[j] = S::of(L::load(b + 8 * j + u * L::lanes));
#pragma GCC unroll 65
        for (std::size_t t = 0; t <= m + k; ++t) {
            const std::size_t first = t > k ? t - k : 0, last = std::min(t, m);
            typename S::Acc s = S::mul(x[first], y[t - first]);
#pragma GCC unroll 9
            for (std::size_t i = first + 1; i <= last; ++i) S::mac(s, x[i], y[t - i]);
            L::store(c + 8 * t + u * L::lanes, S::reduce(s));
        }
    }
}

// multiply_lanes_inline, unrolled for m = k = 1, 2, 4, 8.
template <class L>
void multiply_lanes(const u32* a, std::size_t m, const u32* b, std::size_t k, u32* c) {
    if (m == k) {
        switch (m) {
            case 1: return multiply_lanes_inline<L>(a, 1, b, 1, c);
            case 2: return multiply_lanes_inline<L>(a, 2, b, 2, c);
            case 4: return multiply_lanes_inline<L>(a, 4, b, 4, c);
            case 8: return multiply_lanes_inline<L>(a, 8, b, 8, c);
        }
    }
    multiply_lanes_inline<L>(a, m, b, k, c);
}

// multiply_lanes for a[0] = b[0] = kR (1 in Montgomery form; products of factors 1 + a x, such as
// backend/evaluation's): the terms a[0] b[t] and a[t] b[0] are b[t] and a[t] themselves, added
// after the Montgomery step; c[0] = kR. Montgomery form in and out (canonical); 1 <= m, k <= 8.
template <class L>
[[gnu::always_inline]] inline void multiply_units_inline(const u32* a, std::size_t m, const u32* b, std::size_t k, u32* c) {
    using S = Sums<L>;
    using M = Ops<L>;
    using V = typename L::V;
    constexpr int kVectors = 8 / L::lanes;
#pragma GCC unroll 8
    for (int u = 0; u < kVectors; ++u) {
        typename S::Op x[9], y[9];
#pragma GCC unroll 9
        for (std::size_t i = 1; i <= m; ++i) x[i] = S::of(L::load(a + 8 * i + u * L::lanes));
#pragma GCC unroll 9
        for (std::size_t j = 1; j <= k; ++j) y[j] = S::of(L::load(b + 8 * j + u * L::lanes));
        L::store(c + u * L::lanes, L::broadcast(kR));
#pragma GCC unroll 17
        for (std::size_t t = 1; t <= m + k; ++t) {
            V r = L::broadcast(0);
            if (t <= m) r = L::load(a + 8 * t + u * L::lanes);
            if (t <= k) r = M::add(r, L::load(b + 8 * t + u * L::lanes));
            // a[i] b[t - i] for i in [max(1, t - k), min(t - 1, m)]
            const std::size_t first = t > k ? t - k : 1, last = std::min(t - 1, m);
            if (first <= last) {
                typename S::Acc s = S::mul(x[first], y[t - first]);
#pragma GCC unroll 8
                for (std::size_t i = first + 1; i <= last; ++i) S::mac(s, x[i], y[t - i]);
                r = M::add(r, S::reduce(s));
            }
            L::store(c + 8 * t + u * L::lanes, r);
        }
    }
}

// multiply_units_inline, unrolled for m = k = 1, 2, 4, 8.
template <class L>
void multiply_units(const u32* a, std::size_t m, const u32* b, std::size_t k, u32* c) {
    if (m == k) {
        switch (m) {
            case 1: return multiply_units_inline<L>(a, 1, b, 1, c);
            case 2: return multiply_units_inline<L>(a, 2, b, 2, c);
            case 4: return multiply_units_inline<L>(a, 4, b, 4, c);
            case 8: return multiply_units_inline<L>(a, 8, b, 8, c);
        }
    }
    multiply_units_inline<L>(a, m, b, k, c);
}

// out[i] = f(in[i]) for i < n, f on vectors of S and on single words.
template <class S, class F, class G>
[[gnu::always_inline]] inline void each(std::size_t n, F f, G g) {
    std::size_t i = 0;
    for (; i + S::lanes <= n; i += S::lanes) f(i);
    for (; i < n; ++i) g(i);
}

#if defined(__ARM_NEON) && defined(__aarch64__)
// Neon leaf products, terms i .. 7: sums lo, hi (words 0-1, 2-3 of each 4-word half u) of leaf k's
// product j += x^i A_k mod (x^8 - w) (row k, words 8 - i ..) times b_jk[i], an element of b_jk's
// half i / 4 (umlal by lane: no broadcasts).
template <int i, int kLeaves, int K>
[[gnu::always_inline]] inline void neon_leaf_terms(const u32* const (&row)[kLeaves], const uint32x4_t (&b)[kLeaves][K][2],
                                                   uint64x2_t (&lo)[kLeaves][K][2], uint64x2_t (&hi)[kLeaves][K][2]) {
#pragma GCC unroll 4
    for (int k = 0; k < kLeaves; ++k)
#pragma GCC unroll 2
        for (int u = 0; u < 2; ++u) {
            const uint32x4_t x = vld1q_u32(row[k] + 8 - i + 4 * u);
#pragma GCC unroll 2
            for (int j = 0; j < K; ++j) {
                const uint32x4_t bi = b[k][j][i / 4];
                if constexpr (i == 0) {
                    lo[k][j][u] = vmull_laneq_u32(vget_low_u32(x), bi, 0), hi[k][j][u] = vmull_high_laneq_u32(x, bi, 0);
                } else {
                    lo[k][j][u] = vmlal_laneq_u32(lo[k][j][u], vget_low_u32(x), bi, i % 4);
                    hi[k][j][u] = vmlal_high_laneq_u32(hi[k][j][u], x, bi, i % 4);
                }
            }
        }
    if constexpr (i < 7) neon_leaf_terms<i + 1, kLeaves, K>(row, b, lo, hi);
}
#endif

// Standard layout: out[j] leaf v = a b[j] 2^-32 mod (x^8 - w_v) for j < K and leaves v < count (8
// words each), with weights[v] = w_v 2^32 mod P; inputs canonical, out canonical; for K = 1 the
// same into copy if not null. out[j] and copy may be a or b[j]. One window of each leaf of a serves
// its K products (backend/evaluation's descent multiplies one state by two kept transforms).
//
// Rows [w A, A] of 16 leaves, one 64-byte line each (x^i A mod (x^8 - w) is the 8 words from 8 - i:
// in the line; on x86 the odd words' load one word on crosses into the next row for i = 0 only),
// are written before their products read them. x86: one leaf per 8 lanes, b_i broadcast, 64-bit
// sums of even and odd words; Zen 4/5 (kWideLeaves): two leaves per zmm, their rows' windows by
// load_halves and b_i by broadcast_halves (ntt's WideBottom); Neon: two leaves per step for K = 1
// (8 sums), one for K = 2, b_i an element of b's vectors (umlal by lane). Measurements:
// notes/136-r3.md.
template <class L, int K>
void leaf_products_k(const u32* a, const u32* const (&b)[K], u32* const (&out)[K], std::size_t count, const u32* weights,
                     u32* copy = nullptr) {
    using S = Sums<L>;
    using V = typename L::V;
    constexpr int kVectors = 8 / L::lanes, kBatch = 16;
    // Row kBatch (and row n of a short batch) only for the x86 loads one word past a row.
    alignas(64) u32 window[kBatch + 1][16];
    std::fill_n(window[kBatch], 16, 0u);
    const auto store = [&](int j, std::size_t v, int word, V r) {
        L::store(out[j] + 8 * v + word, r);
        if (K == 1 && copy) L::store(copy + 8 * v + word, r);
    };
    for (std::size_t v0 = 0; v0 < count; v0 += kBatch) {
        const std::size_t n = std::min<std::size_t>(kBatch, count - v0);
        if (n < kBatch) std::fill_n(window[n], 16, 0u);
        for (std::size_t t = 0; t < n; ++t) {
            const V w = L::broadcast(weights[v0 + t]);
#pragma GCC unroll 8
            for (int u = 0; u < kVectors; ++u) {
                const V x = L::load(a + 8 * (v0 + t) + u * L::lanes);
                L::store(window[t] + 8 + u * L::lanes, x);
                L::store(window[t] + u * L::lanes, montgomery_canonical<L>(x, w));
            }
        }
        std::size_t t = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
        if constexpr (std::is_same_v<L, simd::Neon>) {
            constexpr int kLeaves = K == 1 ? 2 : 1;
            const V p = L::broadcast(kP), ninv = L::broadcast(kNegInverse);
            for (; t + kLeaves <= n; t += kLeaves) {
                const u32* row[kLeaves];
                V bv[kLeaves][K][2];
                uint64x2_t lo[kLeaves][K][2], hi[kLeaves][K][2];
#pragma GCC unroll 2
                for (int k = 0; k < kLeaves; ++k) {
                    row[k] = window[t + k];
#pragma GCC unroll 2
                    for (int j = 0; j < K; ++j) {
                        const u32* bt = b[j] + 8 * (v0 + t + k);
                        bv[k][j][0] = L::load(bt), bv[k][j][1] = L::load(bt + 4);
                    }
                }
                neon_leaf_terms<0, kLeaves, K>(row, bv, lo, hi);
#pragma GCC unroll 2
                for (int k = 0; k < kLeaves; ++k)
#pragma GCC unroll 2
                    for (int j = 0; j < K; ++j)
#pragma GCC unroll 2
                        for (int u = 0; u < 2; ++u) {
                            const V m = L::mul_lo(L::lo_halves(lo[k][j][u], hi[k][j][u]), ninv);
                            const uint64x2_t l = L::mla_wide_lo(lo[k][j][u], m, p), h = L::mla_wide_hi(hi[k][j][u], m, p);
                            store(j, v0 + t + k, 4 * u, canonical4<L>(L::hi_halves(l, h)));
                        }
            }
        }
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        if constexpr (kWideLeaves && std::is_same_v<L, simd::Avx512Vl>) {
            using Z = simd::Avx512;
            using E = EvenOdd<Z>;
            for (; t + 2 <= n; t += 2) {
                typename E::Acc s[K];
#pragma GCC unroll 8
                for (int i = 0; i < 8; ++i) {
                    const typename E::Op x = {Z::load_halves(window[t] + 8 - i, window[t + 1] + 8 - i),
                                              Z::load_halves(window[t] + 9 - i, window[t + 1] + 9 - i)};
#pragma GCC unroll 2
                    for (int j = 0; j < K; ++j) {
                        const u32* bt = b[j] + 8 * (v0 + t) + i;
                        const typename Z::V bi = Z::broadcast_halves(bt, bt + 8);
                        if (i == 0) s[j] = E::mul(x, {bi, bi});
                        else E::mac(s[j], x, {bi, bi});
                    }
                }
#pragma GCC unroll 2
                for (int j = 0; j < K; ++j) {
                    const typename Z::V r = E::reduce(s[j]);
                    Z::store(out[j] + 8 * (v0 + t), r);
                    if (K == 1 && copy) Z::store(copy + 8 * (v0 + t), r);
                }
            }
        }
#endif
        for (; t < n; ++t) {
            typename S::Acc s[K][kVectors];
#pragma GCC unroll 8
            for (int u = 0; u < kVectors; ++u) {
                const auto x = S::load(window[t] + 8 + u * L::lanes);
#pragma GCC unroll 2
                for (int j = 0; j < K; ++j) s[j][u] = S::mul(x, S::broadcast(b[j][8 * (v0 + t)]));
#pragma GCC unroll 8
                for (int i = 1; i < 8; ++i) {
                    const auto y = S::load(window[t] + 8 - i + u * L::lanes);
#pragma GCC unroll 2
                    for (int j = 0; j < K; ++j) S::mac(s[j][u], y, S::broadcast(b[j][8 * (v0 + t) + i]));
                }
            }
#pragma GCC unroll 2
            for (int j = 0; j < K; ++j)
#pragma GCC unroll 8
                for (int u = 0; u < kVectors; ++u) store(j, v0 + t, u * L::lanes, S::reduce(s[j][u]));
        }
    }
}

// leaf_products_k with one product: out = a b, the same into copy if not null.
template <class L>
void leaf_products(const u32* a, const u32* b, u32* out, std::size_t count, const u32* weights, u32* copy = nullptr) {
    leaf_products_k<L, 1>(a, {b}, {out}, count, weights, copy);
}

// leaf_products_k with two: out0 = a b0, out1 = a b1.
template <class L>
void leaf_products_twice(const u32* a, const u32* b0, const u32* b1, u32* out0, u32* out1, std::size_t count, const u32* weights) {
    leaf_products_k<L, 2>(a, {b0, b1}, {out0, out1}, count, weights);
}

// x = [x0 .. x(W-1)], y alike -> even = their even words, odd = their odd words, in one order of
// the word pairs (x_2i, x_2i+1) and (y_2i, y_2i+1), the same in both: AVX2 [x0 x2 y0 y2 x4 x6 y4
// y6], SSE and Neon [x0 x2 y0 y2].
template <class L>
[[gnu::always_inline]] inline void deinterleave(typename L::V x, typename L::V y, typename L::V& even, typename L::V& odd) {
    using V = typename L::V;
    if constexpr (false) {
#if defined(__AVX2__)
    } else if constexpr (L::lanes == 8 && sizeof(V) == 32) {
        const __m256 a = _mm256_castsi256_ps(x), b = _mm256_castsi256_ps(y);
        even = _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88)), odd = _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xdd));
#endif
#if defined(__SSE4_1__)
    } else if constexpr (L::lanes == 4 && sizeof(V) == 16) {
        const __m128 a = _mm_castsi128_ps(x), b = _mm_castsi128_ps(y);
        even = _mm_castps_si128(_mm_shuffle_ps(a, b, 0x88)), odd = _mm_castps_si128(_mm_shuffle_ps(a, b, 0xdd));
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    } else if constexpr (L::lanes == 4) {
        even = vuzp1q_u32(x, y), odd = vuzp2q_u32(x, y);
#endif
    } else {
        static_assert(L::lanes == 0, "product_tree: no deinterleave for this struct");
    }
}

// The untwist factors for standard leaves u < count: words [8u, 8u + 8) = rho_u^-k / 8 in
// Montgomery form (k < 8), rho_u = r(4u) (ntt's values form: an 8th root of leaf u's w_u). Prefix
// compatible across lengths; this thread's, kept for the most leaves yet. count <= 2^20 (rho_u's
// order up to 2^23; past ntt's kRoots, to 2^22, the roots are computed here).
inline const u32* untwist_factors(std::size_t count) {
    thread_local std::vector<u32> table;
    if (table.size() < 8 * count) {
        const std::size_t n = std::bit_ceil(count);
        std::vector<u32> rho(n);  // rho_u^-1
        rho[0] = 1;
        for (std::size_t bit = 1, j = 2; bit < n; bit *= 2, ++j) {
            // the inverse of a root of order 2^(j + 2), as kRoots[1][j]
            const u32 root = j < std::size_t(ntt::detail::kRootCount) ? ntt::detail::kRoots[1][j]
                                                                       : power(power(3, (kP - 1) >> (j + 2)), kP - 2);
            for (std::size_t i = 0; i < bit; ++i) rho[bit + i] = multiply_mod(rho[i], root);
        }
        table.resize(8 * n);
        const u32 eighth = multiply_mod(power(8, kP - 2), kR);
        for (std::size_t u = 0; u < n; ++u)
            for (u32 k = 0, f = eighth; k < 8; ++k, f = multiply_mod(f, rho[u])) table[8 * u + k] = f;
    }
    return table.data();
}

// Rows r[0 .. L::lanes): word t of r[i] <-> word i of r[t].
template <class L>
[[gnu::always_inline]] inline void transpose(typename L::V* r) {
    using V = typename L::V;
    if constexpr (L::lanes == 1) {
        return;
#if defined(__AVX2__)
    } else if constexpr (L::lanes == 8 && sizeof(V) == 32) {
        V t[8], u[8];
#pragma GCC unroll 8
        for (int i = 0; i < 8; i += 2) t[i] = _mm256_unpacklo_epi32(r[i], r[i + 1]), t[i + 1] = _mm256_unpackhi_epi32(r[i], r[i + 1]);
#pragma GCC unroll 8
        for (int i = 0; i < 8; i += 4) {
            u[i] = _mm256_unpacklo_epi64(t[i], t[i + 2]), u[i + 1] = _mm256_unpackhi_epi64(t[i], t[i + 2]);
            u[i + 2] = _mm256_unpacklo_epi64(t[i + 1], t[i + 3]), u[i + 3] = _mm256_unpackhi_epi64(t[i + 1], t[i + 3]);
        }
#pragma GCC unroll 8
        for (int i = 0; i < 4; ++i) {
            r[i] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20);
            r[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31);
        }
#endif
#if defined(__SSE4_1__)
    } else if constexpr (L::lanes == 4 && sizeof(V) == 16) {
        const V t0 = _mm_unpacklo_epi32(r[0], r[1]), t1 = _mm_unpackhi_epi32(r[0], r[1]);
        const V t2 = _mm_unpacklo_epi32(r[2], r[3]), t3 = _mm_unpackhi_epi32(r[2], r[3]);
        r[0] = _mm_unpacklo_epi64(t0, t2), r[1] = _mm_unpackhi_epi64(t0, t2);
        r[2] = _mm_unpacklo_epi64(t1, t3), r[3] = _mm_unpackhi_epi64(t1, t3);
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    } else if constexpr (L::lanes == 4) {
        const V t0 = vtrn1q_u32(r[0], r[1]), t1 = vtrn2q_u32(r[0], r[1]);
        const V t2 = vtrn1q_u32(r[2], r[3]), t3 = vtrn2q_u32(r[2], r[3]);
        const auto wide = [](V v) { return vreinterpretq_u64_u32(v); };
        r[0] = vreinterpretq_u32_u64(vtrn1q_u64(wide(t0), wide(t2)));
        r[1] = vreinterpretq_u32_u64(vtrn1q_u64(wide(t1), wide(t3)));
        r[2] = vreinterpretq_u32_u64(vtrn2q_u64(wide(t0), wide(t2)));
        r[3] = vreinterpretq_u32_u64(vtrn2q_u64(wide(t1), wide(t3)));
#endif
    } else {
        static_assert(L::lanes == 1, "product_tree: no transpose for this struct");
    }
}

// The 8 coefficients of the L::lanes leaves at p (8 words each), as vectors: x[j] word t is
// coefficient j of leaf t. store_columns undoes it.
template <class L>
[[gnu::always_inline]] inline void load_columns(const u32* p, typename L::V (&x)[8]) {
    constexpr int W = L::lanes;
#pragma GCC unroll 8
    for (int u = 0; u < 8 / W; ++u) {
        typename L::V r[W];
#pragma GCC unroll 8
        for (int t = 0; t < W; ++t) r[t] = L::load(p + 8 * t + u * W);
        transpose<L>(r);
#pragma GCC unroll 8
        for (int i = 0; i < W; ++i) x[u * W + i] = r[i];
    }
}

template <class L>
[[gnu::always_inline]] inline void store_columns(u32* p, const typename L::V (&x)[8]) {
    constexpr int W = L::lanes;
#pragma GCC unroll 8
    for (int u = 0; u < 8 / W; ++u) {
        typename L::V r[W];
#pragma GCC unroll 8
        for (int i = 0; i < W; ++i) r[i] = x[u * W + i];
        transpose<L>(r);
#pragma GCC unroll 8
        for (int t = 0; t < W; ++t) L::store(p + 8 * t + u * W, r[t]);
    }
}

// Lanes to the standard layout, for the standard tree's nodes over two lane products: lanes is a
// lanes transform of 8 T words (leaf v holds the 8 lanes at w_v, v < T); out[j] (j < 4; null: not
// written) gets the standard leaf form of length T of q_j = p_2j p_(2j+1) 2^-32 (T / 8 leaves of
// 8 words). The 8 roots of standard leaf u (x^8 - w_u) are lanes leaves 8u + i, at rho_u
// z^bitrev3(i) (ntt's values form): pointwise products, an 8-point inverse transform, then c_k
// rho_u^-k / 8 (untwist_factors). T a power of two >= 64; canonical in and out. copy (u32* const*,
// or none: std::nullptr_t, known at compile time): copy[j] gets out[j]'s words too.
//
// lane_pairs_by: the same with the pointwise step as pair(a, b), the vector (word on Scalar) of
// lanes 2j, 2j + 1 combined (canonical): a and b are word offsets in the lanes transform of two
// vectors whose even and odd words, by deinterleave(), are lanes 2j and 2j + 1 (Scalar: of the
// words of lanes 2j and 2j + 1). LanePairProducts: lane_pairs' products of `lanes` (inline: as a
// functor GCC scheduled the loop 5% slower on Zen 3); backend/evaluation also sums.
struct LanePairProducts {};

template <class L, class Pair = LanePairProducts, class Copy = std::nullptr_t>
void lane_pairs_by(const u32* lanes, std::size_t T, u32* const out[4], const u32* untwist, Pair pair = {}, Copy copy = nullptr) {
    using V = typename L::V;
    using M = Ops<L>;
    constexpr int W = L::lanes;
    const auto factor = [](u32 x) { return L::broadcast(multiply_mod(x, kR)); };
    const V r1 = factor(ntt::detail::kRoots[1][0]), z1 = factor(ntt::detail::kRoots[1][1]);
    const V r3 = factor(multiply_mod(ntt::detail::kRoots[1][0], ntt::detail::kRoots[1][1]));
    const auto mul = [](V x, V y) { return montgomery_canonical<L>(x, y); };
    // The 8-point inverse of y (values at s_i, s_2m+1 = -s_2m: 1, r(1), z, r(3)), times 8.
    const auto inverse8 = [&](V (&y)[8]) {
        const V u0 = M::add(y[0], y[1]), u1 = M::sub(y[0], y[1]), u2 = M::add(y[2], y[3]), u3 = mul(M::sub(y[2], y[3]), r1);
        const V u4 = M::add(y[4], y[5]), u5 = mul(M::sub(y[4], y[5]), z1), u6 = M::add(y[6], y[7]), u7 = mul(M::sub(y[6], y[7]), r3);
        const V v0 = M::add(u0, u2), v1 = M::add(u1, u3), v2 = M::sub(u0, u2), v3 = M::sub(u1, u3);
        const V v4 = M::add(u4, u6), v5 = M::add(u5, u7), v6 = mul(M::sub(u4, u6), r1), v7 = mul(M::sub(u5, u7), r1);
        y[0] = M::add(v0, v4), y[1] = M::add(v1, v5), y[2] = M::add(v2, v6), y[3] = M::add(v3, v7);
        y[4] = M::sub(v0, v4), y[5] = M::sub(v1, v5), y[6] = M::sub(v2, v6), y[7] = M::sub(v3, v7);
    };
    const auto put = [&](std::size_t j, std::size_t u, int half, V c) {
        if (!out[j]) return;
        L::store(out[j] + 8 * u + half, c);
        if constexpr (!std::is_same_v<Copy, std::nullptr_t>) L::store(copy[j] + 8 * u + half, c);
    };
    if constexpr (W == 8) {
        // Two leaves u, u + 1 per step; lane 4h + 2b + a of a vector: pair 2h + a of leaf u + b.
        for (std::size_t u = 0; u < T / 8; u += 2) {
            V y[8];
#pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) {
                if constexpr (std::is_same_v<Pair, LanePairProducts>) {
                    V even, odd;
                    deinterleave<L>(L::load(lanes + 8 * (8 * u + i)), L::load(lanes + 8 * (8 * u + 8 + i)), even, odd);
                    y[i] = mul(even, odd);
                } else {
                    y[i] = pair(8 * (8 * u + i), 8 * (8 * u + 8 + i));
                }
            }
            inverse8(y);
            transpose<L>(y);  // y[r]: lane r's 8 coefficients
#pragma GCC unroll 8
            for (int r = 0; r < 8; ++r) {
                const std::size_t j = 2 * (r / 4) + r % 2, at = u + (r / 2) % 2;
                put(j, at, 0, mul(y[r], L::load(untwist + 8 * at)));
            }
        }
    } else if constexpr (W == 4) {
        // One leaf per step; lane j: pair j.
        for (std::size_t u = 0; u < T / 8; ++u) {
            V y[8];
#pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) {
                if constexpr (std::is_same_v<Pair, LanePairProducts>) {
                    V even, odd;
                    deinterleave<L>(L::load(lanes + 8 * (8 * u + i)), L::load(lanes + 8 * (8 * u + i) + 4), even, odd);
                    y[i] = mul(even, odd);
                } else {
                    y[i] = pair(8 * (8 * u + i), 8 * (8 * u + i) + 4);
                }
            }
            inverse8(y);
            V lo[4] = {y[0], y[1], y[2], y[3]}, hi[4] = {y[4], y[5], y[6], y[7]};
            transpose<L>(lo), transpose<L>(hi);
            const V f0 = L::load(untwist + 8 * u), f1 = L::load(untwist + 8 * u + 4);
#pragma GCC unroll 4
            for (int r = 0; r < 4; ++r) {
                put(r, u, 0, mul(lo[r], f0)), put(r, u, 4, mul(hi[r], f1));
            }
        }
    } else {
        static_assert(W == 1);
        for (std::size_t u = 0; u < T / 8; ++u)
            for (std::size_t j = 0; j < 4; ++j) {
                V y[8];
                for (int i = 0; i < 8; ++i) {
                    const std::size_t a = 8 * (8 * u + i) + 2 * j;
                    if constexpr (std::is_same_v<Pair, LanePairProducts>) y[i] = mul(lanes[a], lanes[a + 1]);
                    else y[i] = pair(a, a + 1);
                }
                inverse8(y);
                for (int k = 0; k < 8; ++k) put(j, u, k, mul(y[k], untwist[8 * u + k]));
            }
    }
}

template <class L, class Copy = std::nullptr_t>
void lane_pairs(const u32* lanes, std::size_t T, u32* const out[4], const u32* untwist, Copy copy = nullptr) {
    lane_pairs_by<L>(lanes, T, out, untwist, LanePairProducts{}, copy);
}

// leaf_products, L::lanes leaves at a time in columns (word t of a vector: leaf t): coefficient
// j of the product is sum_{i <= j} a_i b_(j-i) + sum_{i > j} (w a_i) b_(j-i+8), lane by lane, so
// no window, broadcasts or shifted loads; count a multiple of L::lanes.
template <class L>
void leaf_products_columns(const u32* a, const u32* b, u32* out, std::size_t count, const u32* weights, u32* copy = nullptr) {
    using S = Sums<L>;
    using V = typename L::V;
    constexpr int W = L::lanes;
    for (std::size_t v = 0; v < count; v += W) {
        V x[8], y[8], wx[8];
        load_columns<L>(a + 8 * v, x);
        load_columns<L>(b + 8 * v, y);
        const V w = L::load(weights + v);
#pragma GCC unroll 8
        for (int i = 1; i < 8; ++i) wx[i] = montgomery_canonical<L>(x[i], w);
        typename S::Op xo[8], wo[8], yo[8];
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) xo[i] = S::of(x[i]), yo[i] = S::of(y[i]), wo[i] = S::of(i ? wx[i] : x[0]);
        V z[8];
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) {
            typename S::Acc s = S::mul(xo[0], yo[j]);
#pragma GCC unroll 8
            for (int i = 1; i < 8; ++i) S::mac(s, i <= j ? xo[i] : wo[i], yo[(j - i) & 7]);
            z[j] = S::reduce(s);
        }
        store_columns<L>(out + 8 * v, z);
        if (copy) store_columns<L>(copy + 8 * v, z);
    }
}

// out = a b 2^-32 word by word (canonical); a, b < P. n a multiple of 8. out may be a or b.
template <class S>
void pointwise(const u32* a, const u32* b, u32* out, std::size_t n) {
    for (std::size_t i = 0; i < n; i += S::lanes) S::store(out + i, montgomery_canonical<S>(S::load(a + i), S::load(b + i)));
}

// pointwise into out and copy.
template <class S>
void pointwise2(const u32* a, const u32* b, u32* out, u32* copy, std::size_t n) {
    for (std::size_t i = 0; i < n; i += S::lanes) {
        const typename S::V x = montgomery_canonical<S>(S::load(a + i), S::load(b + i));
        S::store(out + i, x), S::store(copy + i, x);
    }
}

// Lanes: 8 linear factors, two words at each p[l], into rows c[0, 8) and c[8, 16) in Montgomery
// form. AVX2 and Neon: the factors' words loaded in pairs and transposed in registers; a vector
// load of words just stored one by one (the generic path) waits for the stores (notes.md).
template <class L>
void load_linear(const u32* const* p, u32* c) {
    using V = typename L::V;
    const V r2 = L::broadcast(kR2);
#if defined(__AVX2__)
    if constexpr (L::lanes == 8 && sizeof(V) == 32) {
        u64 w[8];
        for (int l = 0; l < 8; ++l) std::memcpy(&w[l], p[l], 8);
        const V idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);  // pairs -> [c0 of 4 | c1 of 4]
        const V x = _mm256_permutevar8x32_epi32(_mm256_setr_epi64x(w[0], w[1], w[2], w[3]), idx);
        const V y = _mm256_permutevar8x32_epi32(_mm256_setr_epi64x(w[4], w[5], w[6], w[7]), idx);
        L::store(c, montgomery_canonical<L>(_mm256_permute2x128_si256(x, y, 0x20), r2));
        L::store(c + 8, montgomery_canonical<L>(_mm256_permute2x128_si256(x, y, 0x31), r2));
        return;
    }
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    if constexpr (L::lanes == 4 && std::is_same_v<V, uint32x4_t>) {
        for (int h = 0; h < 8; h += 4) {
            const V x = vcombine_u32(vld1_u32(p[h]), vld1_u32(p[h + 1])), y = vcombine_u32(vld1_u32(p[h + 2]), vld1_u32(p[h + 3]));
            L::store(c + h, montgomery_canonical<L>(vuzp1q_u32(x, y), r2));
            L::store(c + 8 + h, montgomery_canonical<L>(vuzp2q_u32(x, y), r2));
        }
        return;
    }
#endif
    for (int l = 0; l < 8; ++l) c[l] = p[l][0], c[8 + l] = p[l][1];
    for (int i = 0; i < 16; i += L::lanes) L::store(c + i, montgomery_canonical<L>(L::load(c + i), r2));
}

// Lanes: 8 factors of degree d >= 1, words p[l][0 .. d], into rows c[8j + l] (j <= d) in
// Montgomery form. AVX2 and Neon: each factor's words as vectors (the last one by a masked load,
// vpmaskmovd or vmovdqu32 {k}, or by lane loads on Neon: no word past a factor is read), then
// transposed in registers. Elsewhere word by word, then one pass to Montgomery form.
template <class L>
void load_full(const u32* const* p, u32 d, u32* c) {
    using V = typename L::V;
    const V r2 = L::broadcast(kR2);
    const std::size_t n = d + 1;
#if defined(__AVX2__)
    if constexpr (L::lanes == 8 && sizeof(V) == 32) {
        for (std::size_t j = 0; j < n; j += 8) {
            const std::size_t m = std::min<std::size_t>(8, n - j);
            V r[8];
            if (m == 8) {
#pragma GCC unroll 8
                for (int l = 0; l < 8; ++l) r[l] = L::load(p[l] + j);
            } else if constexpr (requires { L::load_mask(p[0], 0u); }) {
#pragma GCC unroll 8
                for (int l = 0; l < 8; ++l) r[l] = L::load_mask(p[l] + j, (1u << m) - 1);
            } else {
                const V mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(int(m)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
#pragma GCC unroll 8
                for (int l = 0; l < 8; ++l) r[l] = _mm256_maskload_epi32(reinterpret_cast<const int*>(p[l] + j), mask);
            }
            transpose<L>(r);
            for (std::size_t t = 0; t < m; ++t) L::store(c + 8 * (j + t), montgomery_canonical<L>(r[t], r2));
        }
        return;
    }
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    if constexpr (L::lanes == 4 && std::is_same_v<V, uint32x4_t>) {
        const auto part = [](const u32* q, std::size_t m) -> V {  // q[0 .. m), m < 4, zeros past
            const uint32x2_t zero = vdup_n_u32(0);
            if (m == 1) return vcombine_u32(vld1_lane_u32(q, zero, 0), zero);
            if (m == 2) return vcombine_u32(vld1_u32(q), zero);
            return vcombine_u32(vld1_u32(q), vld1_lane_u32(q + 2, zero, 0));
        };
        for (std::size_t j = 0; j < n; j += 4) {
            const std::size_t m = std::min<std::size_t>(4, n - j);
#pragma GCC unroll 2
            for (int h = 0; h < 8; h += 4) {
                V r[4];
#pragma GCC unroll 4
                for (int t = 0; t < 4; ++t) r[t] = m == 4 ? L::load(p[h + t] + j) : part(p[h + t] + j, m);
                transpose<L>(r);
                for (std::size_t i = 0; i < m; ++i) L::store(c + 8 * (j + i) + h, montgomery_canonical<L>(r[i], r2));
            }
        }
        return;
    }
#endif
    for (std::size_t j = 0; j < n; ++j)
        for (int l = 0; l < 8; ++l) c[8 * j + l] = p[l][j];
    for (std::size_t i = 0; i < 8 * n; i += L::lanes) L::store(c + i, montgomery_canonical<L>(L::load(c + i), r2));
}

// out = copy = in, n words, n a multiple of S::lanes.
template <class S>
void copy2(const u32* in, u32* out, u32* copy, std::size_t n) {
    for (std::size_t i = 0; i < n; i += S::lanes) {
        const typename S::V x = S::load(in + i);
        S::store(out + i, x), S::store(copy + i, x);
    }
}

// out[i] = in[i] y 2^-32 (canonical) for i < n; in[i] < P, y < P. out may be in.
template <class S>
void scale(const u32* in, u32 y, u32* out, std::size_t n) {
    const typename S::V v = S::broadcast(y);
    each<S>(n, [&](std::size_t i) { S::store(out + i, montgomery_canonical<S>(S::load(in + i), v)); },
            [&](std::size_t i) { out[i] = montgomery(in[i], y); });
}

}  // namespace fps::product_tree::detail
