// The product tree's kernels, as templates on backend/simd structs: 64-bit sums of products with
// one Montgomery step, the lanes' schoolbook, the standard layout's leaf products, pointwise
// products, twists and conversions. Words below P; "Montgomery form" is x 2^32 mod P.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
    constexpr int kVectors = 8 / L::lanes, kBatch = 8;
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

// Standard layout: out[i] = c[i] t[i] 2^-32 for i < count, zero for count <= i < m (t: twiddles
// in Montgomery form).
template <class S>
void twist_words(const u32* c, std::size_t count, const u32* t, u32* out, std::size_t m) {
    each<S>(count, [&](std::size_t i) { S::store(out + i, montgomery_canonical<S>(S::load(c + i), S::load(t + i))); },
            [&](std::size_t i) { out[i] = montgomery(c[i], t[i]); });
    std::fill(out + count, out + m, 0u);
}

// Standard layout: word j of each leaf of f (n words) times z[j] 2^-32 (z: 8 words, Montgomery form).
template <class L>
void scale_leaves(u32* f, std::size_t n, const u32* z) {
    constexpr int kVectors = 8 / L::lanes;
    typename L::V w[kVectors];
    for (int u = 0; u < kVectors; ++u) w[u] = L::load(z + u * L::lanes);
    for (std::size_t i = 0; i < n; i += 8) {
#pragma GCC unroll 8
        for (int u = 0; u < kVectors; ++u)
            L::store(f + i + u * L::lanes, montgomery_canonical<L>(L::load(f + i + u * L::lanes), w[u]));
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

// Lanes: coefficient j of out (8 words) = coefficient j of c times t[j] 2^-32 for j < count, zero
// for count <= j < m. t: twiddles in Montgomery form.
template <class L>
void twist(const u32* c, std::size_t count, const u32* t, u32* out, std::size_t m) {
    constexpr int kVectors = 8 / L::lanes;
    for (std::size_t j = 0; j < count; ++j) {
        const typename L::V w = L::broadcast(t[j]);
#pragma GCC unroll 8
        for (int u = 0; u < kVectors; ++u)
            L::store(out + 8 * j + u * L::lanes, montgomery_canonical<L>(L::load(c + 8 * j + u * L::lanes), w));
    }
    std::fill(out + 8 * count, out + 8 * m, 0u);
}

// out[i] = in[i] y 2^-32 (canonical) for i < n; in[i] < P, y < P. out may be in.
template <class S>
void scale(const u32* in, u32 y, u32* out, std::size_t n) {
    const typename S::V v = S::broadcast(y);
    each<S>(n, [&](std::size_t i) { S::store(out + i, montgomery_canonical<S>(S::load(in + i), v)); },
            [&](std::size_t i) { out[i] = montgomery(in[i], y); });
}

}  // namespace fps::product_tree::detail
