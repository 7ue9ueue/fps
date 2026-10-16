// The product tree's kernels, as templates on backend/simd structs: 64-bit sums of products with
// one Montgomery step, the lanes' schoolbook, the standard layout's leaf products, pointwise
// products and conversions. Words below P; "Montgomery form" is x 2^32 mod P.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

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

// Standard layout: out leaf v = a b 2^-32 mod (x^8 - w_v) for leaves v < count (8 words each), with
// weights[v] = w_v 2^32 mod P; inputs canonical, out canonical; the same into copy if not null.
// out and copy may be a or b.
template <class L>
void leaf_products(const u32* a, const u32* b, u32* out, std::size_t count, const u32* weights, u32* copy = nullptr) {
    using S = Sums<L>;
    using V = typename L::V;
    constexpr int kVectors = 8 / L::lanes, kBatch = 16;
    // Row t: [w A, A, 0 ...] for leaf A: x^i A mod (x^8 - w) is the 8 words from 8 - i. Loads read
    // one word past the 16 (EvenOdd::load).
    alignas(64) u32 window[kBatch][24] = {};
    for (std::size_t v0 = 0; v0 < count; v0 += kBatch) {
        const std::size_t n = std::min<std::size_t>(kBatch, count - v0);
        for (std::size_t t = 0; t < n; ++t) {
            const V w = L::broadcast(weights[v0 + t]);
#pragma GCC unroll 8
            for (int u = 0; u < kVectors; ++u) {
                const V x = L::load(a + 8 * (v0 + t) + u * L::lanes);
                L::store(window[t] + 8 + u * L::lanes, x);
                L::store(window[t] + u * L::lanes, montgomery_canonical<L>(x, w));
            }
        }
        for (std::size_t t = 0; t < n; ++t) {
            const u32* bt = b + 8 * (v0 + t);
            V r[kVectors];
#pragma GCC unroll 8
            for (int u = 0; u < kVectors; ++u) {
                typename S::Acc s = S::mul(S::load(window[t] + 8 + u * L::lanes), S::broadcast(bt[0]));
#pragma GCC unroll 8
                for (int i = 1; i < 8; ++i) S::mac(s, S::load(window[t] + 8 - i + u * L::lanes), S::broadcast(bt[i]));
                r[u] = S::reduce(s);
            }
#pragma GCC unroll 8
            for (int u = 0; u < kVectors; ++u) L::store(out + 8 * (v0 + t) + u * L::lanes, r[u]);
            if (copy) {
#pragma GCC unroll 8
                for (int u = 0; u < kVectors; ++u) L::store(copy + 8 * (v0 + t) + u * L::lanes, r[u]);
            }
        }
    }
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

// out[i] = in[i] y 2^-32 (canonical) for i < n; in[i] < P, y < P. out may be in.
template <class S>
void scale(const u32* in, u32 y, u32* out, std::size_t n) {
    const typename S::V v = S::broadcast(y);
    each<S>(n, [&](std::size_t i) { S::store(out + i, montgomery_canonical<S>(S::load(in + i), v)); },
            [&](std::size_t i) { out[i] = montgomery(in[i], y); });
}

}  // namespace fps::product_tree::detail
