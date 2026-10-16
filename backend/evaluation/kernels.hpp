// The evaluation kernel's loops, as templates on backend/simd structs: middle products and sums of
// cross products in lanes (the bases of the descents), Horner's rule, batch inversion and passes
// over transforms. Words below P; "Montgomery form" is x 2^32 mod P. Lanes: 8 polynomials side
// by side, word 8 j + l = coefficient j of polynomial l (backend/product_tree/tree.hpp).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#include "backend/product_tree/kernels.hpp"

namespace fps::evaluation::detail {

using product_tree::Lanes;
using product_tree::Wide;
using product_tree::detail::kMaxTerms;
using product_tree::detail::kP;
using product_tree::detail::kR;
using product_tree::detail::kR2;
using product_tree::detail::montgomery;
using product_tree::detail::montgomery_canonical;
using product_tree::detail::multiply_mod;
using product_tree::detail::Ops;
using product_tree::detail::power;
using product_tree::detail::Sums;
using product_tree::detail::u32;
using product_tree::detail::u64;

constexpr u32 add_mod(u32 x, u32 y) { return x >= kP - y ? x - (kP - y) : x + y; }
constexpr u32 sub_mod(u32 x, u32 y) { return x >= y ? x - y : x + kP - y; }
constexpr u32 inverse_mod(u32 x) { return power(x, kP - 2); }
constexpr u32 half(u32 x) { return multiply_mod(x, (kP + 1) / 2); }

// Lanes: out[t] = sum_(j <= k) w[k + t - j] q[j] 2^-32 for t < count, the middle product of w (k +
// count coefficients) and q (k + 1); canonical in and out, q in Montgomery form for w's form out.
// k < kMaxTerms, k + count <= 32. kUnit: q[0] = kR (1 in Montgomery form, as in the products of
// factors 1 + a x), so the term j = 0 is w[k + t] itself, added after the Montgomery step.
template <class L, bool kUnit = false>
[[gnu::always_inline]] inline void middle_lanes_inline(const u32* w, std::size_t k, const u32* q, std::size_t count, u32* out) {
    using S = Sums<L>;
    constexpr int kVectors = 8 / L::lanes;
    constexpr std::size_t j0 = kUnit ? 1 : 0;
#pragma GCC unroll 8
    for (int u = 0; u < kVectors; ++u) {
        typename S::Op x[32], y[kMaxTerms];
#pragma GCC unroll 32
        for (std::size_t i = 0; i < k + count; ++i) x[i] = S::of(L::load(w + 8 * i + u * L::lanes));
#pragma GCC unroll 9
        for (std::size_t j = j0; j <= k; ++j) y[j] = S::of(L::load(q + 8 * j + u * L::lanes));
#pragma GCC unroll 16
        for (std::size_t t = 0; t < count; ++t) {
            typename S::Acc s = S::mul(x[k + t - j0], y[j0]);
#pragma GCC unroll 9
            for (std::size_t j = j0 + 1; j <= k; ++j) S::mac(s, x[k + t - j], y[j]);
            typename L::V r = S::reduce(s);
            if constexpr (kUnit) r = Ops<L>::add(r, L::load(w + 8 * (k + t) + u * L::lanes));
            L::store(out + 8 * t + u * L::lanes, r);
        }
    }
}

// middle_lanes_inline, unrolled for k = count = 1, 2, 4, 8.
template <class L, bool kUnit = false>
void middle_lanes(const u32* w, std::size_t k, const u32* q, std::size_t count, u32* out) {
    if (k == count) {
        switch (k) {
            case 1: return middle_lanes_inline<L, kUnit>(w, 1, q, 1, out);
            case 2: return middle_lanes_inline<L, kUnit>(w, 2, q, 2, out);
            case 4: return middle_lanes_inline<L, kUnit>(w, 4, q, 4, out);
            case 8: return middle_lanes_inline<L, kUnit>(w, 8, q, 8, out);
        }
    }
    middle_lanes_inline<L, kUnit>(w, k, q, count, out);
}

// Lanes: out[t] = (sum_i a[i] d[t - i] + sum_i b[i] c[t - i]) 2^-32 for t < n + k, a of n
// coefficients, d of k + 1, b of k, c of n + 1 (R_l Q_r + R_r Q_l: a = R_l, b = R_r, c = Q_l, d =
// Q_r); canonical in and out. 1 <= n, k < kMaxTerms: each sum has at most kMaxTerms terms.
template <class L>
[[gnu::always_inline]] inline void cross_lanes_inline(const u32* a, std::size_t n, const u32* b, std::size_t k, const u32* c,
                                                      const u32* d, u32* out) {
    using S = Sums<L>;
    using M = Ops<L>;
    constexpr int kVectors = 8 / L::lanes;
#pragma GCC unroll 8
    for (int u = 0; u < kVectors; ++u) {
        typename S::Op xa[kMaxTerms], xb[kMaxTerms], yc[kMaxTerms + 1], yd[kMaxTerms + 1];
#pragma GCC unroll 9
        for (std::size_t i = 0; i < n; ++i) xa[i] = S::of(L::load(a + 8 * i + u * L::lanes));
#pragma GCC unroll 9
        for (std::size_t i = 0; i < k; ++i) xb[i] = S::of(L::load(b + 8 * i + u * L::lanes));
#pragma GCC unroll 10
        for (std::size_t j = 0; j <= n; ++j) yc[j] = S::of(L::load(c + 8 * j + u * L::lanes));
#pragma GCC unroll 10
        for (std::size_t j = 0; j <= k; ++j) yd[j] = S::of(L::load(d + 8 * j + u * L::lanes));
#pragma GCC unroll 16
        for (std::size_t t = 0; t < n + k; ++t) {
            const std::size_t first = t > k ? t - k : 0, last = std::min(t, n - 1);
            typename S::Acc s = S::mul(xa[first], yd[t - first]);
#pragma GCC unroll 9
            for (std::size_t i = first + 1; i <= last; ++i) S::mac(s, xa[i], yd[t - i]);
            const std::size_t first2 = t > n ? t - n : 0, last2 = std::min(t, k - 1);
            typename S::Acc s2 = S::mul(xb[first2], yc[t - first2]);
#pragma GCC unroll 9
            for (std::size_t i = first2 + 1; i <= last2; ++i) S::mac(s2, xb[i], yc[t - i]);
            L::store(out + 8 * t + u * L::lanes, M::add(S::reduce(s), S::reduce(s2)));
        }
    }
}

// cross_lanes_inline, unrolled for n = k = 1, 2, 4, 8.
template <class L>
void cross_lanes(const u32* a, std::size_t n, const u32* b, std::size_t k, const u32* c, const u32* d, u32* out) {
    if (n == k) {
        switch (n) {
            case 1: return cross_lanes_inline<L>(a, 1, b, 1, c, d, out);
            case 2: return cross_lanes_inline<L>(a, 2, b, 2, c, d, out);
            case 4: return cross_lanes_inline<L>(a, 4, b, 4, c, d, out);
            case 8: return cross_lanes_inline<L>(a, 8, b, 8, c, d, out);
        }
    }
    cross_lanes_inline<L>(a, n, b, k, c, d, out);
}

// cross_lanes for c[0] = d[0] = kR (the products of factors 1 + a x): the terms a[t] d[0] and b[t]
// c[0] are a[t] and b[t] themselves, added after the Montgomery step, and the two sums share one
// step when they have at most kMaxTerms products together. Same arguments.
template <class L>
[[gnu::always_inline]] inline void cross_units_inline(const u32* a, std::size_t n, const u32* b, std::size_t k, const u32* c,
                                                      const u32* d, u32* out) {
    using S = Sums<L>;
    using M = Ops<L>;
    using V = typename L::V;
    constexpr int kVectors = 8 / L::lanes;
#pragma GCC unroll 8
    for (int u = 0; u < kVectors; ++u) {
        typename S::Op xa[kMaxTerms], xb[kMaxTerms], yc[kMaxTerms + 1], yd[kMaxTerms + 1];
#pragma GCC unroll 9
        for (std::size_t i = 0; i < n; ++i) xa[i] = S::of(L::load(a + 8 * i + u * L::lanes));
#pragma GCC unroll 9
        for (std::size_t i = 0; i < k; ++i) xb[i] = S::of(L::load(b + 8 * i + u * L::lanes));
#pragma GCC unroll 10
        for (std::size_t j = 1; j <= n; ++j) yc[j] = S::of(L::load(c + 8 * j + u * L::lanes));
#pragma GCC unroll 10
        for (std::size_t j = 1; j <= k; ++j) yd[j] = S::of(L::load(d + 8 * j + u * L::lanes));
#pragma GCC unroll 16
        for (std::size_t t = 0; t < n + k; ++t) {
            // Products a[i] d[t - i] for i in [first, end), b[i] c[t - i] for i in [first2, end2).
            const std::size_t first = t > k ? t - k : 0, end = std::min(t, n);
            const std::size_t first2 = t > n ? t - n : 0, end2 = std::min(t, k);
            const std::size_t terms = (end > first ? end - first : 0) + (end2 > first2 ? end2 - first2 : 0);
            V r = t < n ? L::load(a + 8 * t + u * L::lanes) : L::broadcast(0);
            if (t < k) r = M::add(r, L::load(b + 8 * t + u * L::lanes));
            typename S::Acc s{}, s2{};
            bool started = false, started2 = false;
            const bool merged = terms <= std::size_t(kMaxTerms);
#pragma GCC unroll 9
            for (std::size_t i = first; i < end; ++i) {
                if (started) S::mac(s, xa[i], yd[t - i]);
                else s = S::mul(xa[i], yd[t - i]), started = true;
            }
#pragma GCC unroll 9
            for (std::size_t i = first2; i < end2; ++i) {
                if (merged && started) S::mac(s, xb[i], yc[t - i]);
                else if (merged) s = S::mul(xb[i], yc[t - i]), started = true;
                else if (started2) S::mac(s2, xb[i], yc[t - i]);
                else s2 = S::mul(xb[i], yc[t - i]), started2 = true;
            }
            if (started) r = M::add(r, S::reduce(s));
            if (started2) r = M::add(r, S::reduce(s2));
            L::store(out + 8 * t + u * L::lanes, r);
        }
    }
}

// cross_units_inline, unrolled for n = k = 1, 2, 4, 8.
template <class L>
void cross_units(const u32* a, std::size_t n, const u32* b, std::size_t k, const u32* c, const u32* d, u32* out) {
    if (n == k) {
        switch (n) {
            case 1: return cross_units_inline<L>(a, 1, b, 1, c, d, out);
            case 2: return cross_units_inline<L>(a, 2, b, 2, c, d, out);
            case 4: return cross_units_inline<L>(a, 4, b, 4, c, d, out);
            case 8: return cross_units_inline<L>(a, 8, b, 8, c, d, out);
        }
    }
    cross_units_inline<L>(a, n, b, k, c, d, out);
}

// c = a b, lane by lane, for a of degree m and b of degree k with a[0] = b[0] = kR (the products
// of factors 1 + a x): the terms a[0] b[t] and a[t] b[0] are b[t] and a[t] themselves, added
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

// multiply_units_inline, unrolled for m = k = 1, 2, 4.
template <class L>
void multiply_units(const u32* a, std::size_t m, const u32* b, std::size_t k, u32* c) {
    if (m == k) {
        switch (m) {
            case 1: return multiply_units_inline<L>(a, 1, b, 1, c);
            case 2: return multiply_units_inline<L>(a, 2, b, 2, c);
            case 4: return multiply_units_inline<L>(a, 4, b, 4, c);
        }
    }
    multiply_units_inline<L>(a, m, b, k, c);
}

// lanes[8 j + l] = rows[l stride + j] for l < 8, j < count (a multiple of 8): 8 rows to lanes, by
// transposes of L::lanes x L::lanes words. lanes_to_rows undoes it.
template <class L>
void rows_to_lanes(const u32* rows, std::size_t stride, std::size_t count, u32* lanes) {
    constexpr int W = L::lanes;
    for (std::size_t j = 0; j < count; j += W)
        for (int l = 0; l < 8; l += W) {
            typename L::V r[W];
            for (int t = 0; t < W; ++t) r[t] = L::load(rows + (l + t) * stride + j);
            product_tree::detail::transpose<L>(r);
            for (int i = 0; i < W; ++i) L::store(lanes + 8 * (j + i) + l, r[i]);
        }
}
template <class L>
void lanes_to_rows(const u32* lanes, std::size_t count, u32* rows, std::size_t stride) {
    constexpr int W = L::lanes;
    for (std::size_t j = 0; j < count; j += W)
        for (int l = 0; l < 8; l += W) {
            typename L::V r[W];
            for (int i = 0; i < W; ++i) r[i] = L::load(lanes + 8 * (j + i) + l);
            product_tree::detail::transpose<L>(r);
            for (int t = 0; t < W; ++t) L::store(rows + (l + t) * stride + j, r[t]);
        }
}

// f[i] = f[i] - f[h + i] (sign < 0) or f[i] + f[h + i], i < h; canonical, h a multiple of S::lanes.
template <class S>
void combine_halves(u32* f, std::size_t h, int sign) {
    using M = Ops<S>;
    if (sign < 0) {
        for (std::size_t i = 0; i < h; i += S::lanes) S::store(f + i, M::sub(S::load(f + i), S::load(f + h + i)));
    } else {
        for (std::size_t i = 0; i < h; i += S::lanes) S::store(f + i, M::add(S::load(f + i), S::load(f + h + i)));
    }
}

// out = a b + c d (pointwise, each product times 2^-32), canonical; the same into copy if not
// null. n a multiple of S::lanes; out and copy may be a or c.
template <class S>
void pointwise_sums(const u32* a, const u32* b, const u32* c, const u32* d, u32* out, u32* copy, std::size_t n) {
    using M = Ops<S>;
    for (std::size_t i = 0; i < n; i += S::lanes) {
        const typename S::V x = M::add(montgomery_canonical<S>(S::load(a + i), S::load(b + i)),
                                       montgomery_canonical<S>(S::load(c + i), S::load(d + i)));
        S::store(out + i, x);
        if (copy) S::store(copy + i, x);
    }
}

// out[i] = a[i] + b[i] mod P, canonical; out may be a or b. n a multiple of S::lanes.
template <class S>
void add_words(const u32* a, const u32* b, u32* out, std::size_t n) {
    using M = Ops<S>;
    for (std::size_t i = 0; i < n; i += S::lanes) S::store(out + i, M::add(S::load(a + i), S::load(b + i)));
}

// out[i] = a[i] - b[i] y 2^-32 mod P, canonical; a, b, y < P. out may be a.
template <class S>
void subtract_scaled(const u32* a, const u32* b, u32 y, u32* out, std::size_t n) {
    using M = Ops<S>;
    const typename S::V v = S::broadcast(y);
    product_tree::detail::each<S>(
        n, [&](std::size_t i) { S::store(out + i, M::sub(S::load(a + i), montgomery_canonical<S>(S::load(b + i), v))); },
        [&](std::size_t i) { out[i] = sub_mod(a[i], montgomery(b[i], y)); });
}

// x^e 2^32 for x in Montgomery form (x 2^32), e >= 0, lane by lane.
template <class S>
typename S::V power_montgomery(typename S::V x, std::size_t e) {
    typename S::V r = S::broadcast(kR);
    for (; e; e >>= 1, x = montgomery_canonical<S>(x, x))
        if (e & 1) r = montgomery_canonical<S>(r, x);
    return r;
}

// One step of Horner's rule, acc x + c: acc < 2P (lazy), x in Montgomery form, c < P.
template <class S>
[[gnu::always_inline]] inline typename S::V horner_step(typename S::V acc, typename S::V x, typename S::V c) {
    using M = Ops<S>;
    return M::template reduce<2 * kP>(S::add(M::montgomery(acc, x), c));
}

// values[i] = f(x[i]) for i < m by Horner's rule: blocks of kChains vectors of points, each a
// chain (the steps' latency: 8 chains where there are 32 registers). A short block of v vectors
// runs f in kChains / v parts of b coefficients (chains too), joined by powers x^b. f of n >= 1
// words, all canonical.
template <class S, std::size_t kChains = S::registers >= 32 ? 8 : 4>
void horner(const u32* f, std::size_t n, const u32* x, std::size_t m, u32* values) {
    using V = typename S::V;
    using M = Ops<S>;
    constexpr std::size_t W = S::lanes;
    const V r2 = S::broadcast(kR2);
    for (std::size_t i0 = 0; i0 < m; i0 += kChains * W) {
        const std::size_t here = std::min(kChains * W, m - i0), vectors = (here + W - 1) / W;
        alignas(64) u32 points[kChains * W] = {}, out[kChains * W];
        std::copy_n(x + i0, here, points);
        V xm[kChains], acc[kChains];
        for (std::size_t c = 0; c < kChains; ++c) xm[c] = montgomery_canonical<S>(S::load(points + W * (c % vectors)), r2);
        for (std::size_t c = 0; c < kChains; ++c) acc[c] = S::broadcast(0);
        std::size_t parts = 1, b = n;
        if (vectors == kChains) {
            for (std::size_t s = n; s-- > 0;) {
                const V c = S::broadcast(f[s]);
#pragma GCC unroll 8
                for (std::size_t t = 0; t < kChains; ++t) acc[t] = horner_step<S>(acc[t], xm[t], c);
            }
        } else {
            // Chain c: vector c % vectors, part c / vectors, coefficients [part b, part b + b).
            parts = n >= 64 * kChains ? kChains / vectors : 1, b = (n + parts - 1) / parts;
            const std::size_t chains = vectors * parts;
            std::size_t first[kChains];
            for (std::size_t c = 0; c < chains; ++c) first[c] = c / vectors * b;
            for (std::size_t s = b; s-- > 0;)
                for (std::size_t c = 0; c < chains; ++c) {
                    const std::size_t i = first[c] + s;
                    acc[c] = horner_step<S>(acc[c], xm[c], S::broadcast(i < n ? f[i] : 0));
                }
        }
        for (std::size_t v = 0; v < vectors; ++v) {
            V y = acc[v + (parts - 1) * vectors];
            if (parts > 1) {
                const V xb = power_montgomery<S>(xm[v], b);
                for (std::size_t p = parts - 1; p-- > 0;) y = horner_step<S>(y, xb, M::template reduce<kP>(acc[v + p * vectors]));
            }
            S::store(out + W * v, M::template reduce<kP>(y));
        }
        std::copy_n(out, here, values + i0);
    }
}

// v[i] = y[i] / v[i] for i < count (a multiple of L::lanes, at most 8 kLaneBase), but 0 where
// first + i >= m (v[i] read as 1 there): prefix products over the vectors, one vector power for
// their lanes' inverses (Montgomery's trick); canonical in and out. Returns false, v unspecified,
// if some v[i] with first + i < m is 0.
template <class L>
bool divide_slots(u32* v, const u32* y, std::size_t first, std::size_t m, std::size_t count) {
    using V = typename L::V;
    constexpr std::size_t W = L::lanes;
    if (first + count > m) std::fill(v + (m > first ? m - first : 0), v + count, 1u);
    const V r2 = L::broadcast(kR2);
    V prefix[8 * product_tree::kLaneBase / W];
    V all = L::broadcast(kR);
    for (std::size_t k = 0; k < count / W; ++k) prefix[k] = all = montgomery_canonical<L>(all, montgomery_canonical<L>(L::load(v + W * k), r2));
    alignas(64) u32 lanes[W];
    L::store(lanes, all);
    for (std::size_t l = 0; l < W; ++l)
        if (lanes[l] == 0) return false;
    V e = power_montgomery<L>(all, kP - 2);  // (product 2^32)^(P-2) 2^32 = product^-1 2^32
    for (std::size_t k = count / W; k-- > 0;) {
        const V x = montgomery_canonical<L>(L::load(v + W * k), r2);
        const V reciprocal = k ? montgomery_canonical<L>(e, prefix[k - 1]) : e;
        e = montgomery_canonical<L>(e, x);
        L::store(v + W * k, montgomery_canonical<L>(L::load(y + W * k), reciprocal));
    }
    return true;
}

}  // namespace fps::evaluation::detail
