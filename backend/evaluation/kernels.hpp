// The evaluation kernel's loops, as templates on backend/simd structs: middle products and sums of
// cross products in lanes (the bases of the descents), Horner's rule, batch inversion and passes
// over transforms. Words below P; "Montgomery form" is x 2^32 mod P. Lanes: 8 polynomials side
// by side, word 8 j + l = coefficient j of polynomial l (backend/product_tree/tree.hpp).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

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
using product_tree::detail::multiply_units;
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

// middle_lanes_inline for k = count = K, in blocks of B outputs: per term, w's and q's vectors
// loaded (their odd words by shifts), the sums in registers, no arrays of operands to spill. x86
// (Sums EvenOdd, 8 lanes per vector): a base's descent of 16 slots in 0.84 of the time on Zen 3
// (notes/136-r1.md).
template <class L, std::size_t K, std::size_t B, bool kUnit>
[[gnu::always_inline]] inline void middle_blocks(const u32* w, const u32* q, u32* out) {
    using S = Sums<L>;
    constexpr std::size_t j0 = kUnit ? 1 : 0;
#pragma GCC unroll 8
    for (std::size_t t0 = 0; t0 < K; t0 += B) {
        typename S::Acc s[B];
#pragma GCC unroll 9
        for (std::size_t j = j0; j <= K; ++j) {
            const typename S::Op y = S::of(L::load(q + 8 * j));
#pragma GCC unroll 8
            for (std::size_t i = 0; i < B; ++i) {
                const typename S::Op x = S::of(L::load(w + 8 * (K + t0 + i - j)));
                if (j == j0) s[i] = S::mul(x, y);
                else S::mac(s[i], x, y);
            }
        }
#pragma GCC unroll 8
        for (std::size_t i = 0; i < B; ++i) {
            typename L::V r = S::reduce(s[i]);
            if constexpr (kUnit) r = Ops<L>::add(r, L::load(w + 8 * (K + t0 + i)));
            L::store(out + 8 * (t0 + i), r);
        }
    }
}

// middle_lanes_inline, unrolled for k = count = 1, 2, 4, 8 (x86: middle_blocks).
template <class L, bool kUnit = false>
void middle_lanes(const u32* w, std::size_t k, const u32* q, std::size_t count, u32* out) {
    if (k == count) {
        if constexpr (std::is_same_v<Sums<L>, product_tree::detail::EvenOdd<L>> && L::lanes == 8) {
            switch (k) {
                case 1: return middle_blocks<L, 1, 1, kUnit>(w, q, out);
                case 2: return middle_blocks<L, 2, 2, kUnit>(w, q, out);
                case 4: return middle_blocks<L, 4, 2, kUnit>(w, q, out);
                case 8: return middle_blocks<L, 8, 2, kUnit>(w, q, out);
            }
        } else {
            switch (k) {
                case 1: return middle_lanes_inline<L, kUnit>(w, 1, q, 1, out);
                case 2: return middle_lanes_inline<L, kUnit>(w, 2, q, 2, out);
                case 4: return middle_lanes_inline<L, kUnit>(w, 4, q, 4, out);
                case 8: return middle_lanes_inline<L, kUnit>(w, 8, q, 8, out);
            }
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

// cross_units_inline for n = k = K, one output at a time with every operand loaded per term (its
// odd words by shifts) and the sums in registers, as middle_blocks (x86, 8 lanes per vector).
template <class L, std::size_t K>
[[gnu::always_inline]] inline void cross_loads(const u32* a, const u32* b, const u32* c, const u32* d, u32* out) {
    using S = Sums<L>;
    using M = Ops<L>;
    using V = typename L::V;
    const auto op = [](const u32* p) { return S::of(L::load(p)); };
#pragma GCC unroll 16
    for (std::size_t t = 0; t < 2 * K; ++t) {
        const std::size_t first = t > K ? t - K : 0, end = std::min(t, K);
        const std::size_t terms = 2 * (end > first ? end - first : 0);
        V r = t < K ? M::add(L::load(a + 8 * t), L::load(b + 8 * t)) : L::broadcast(0);
        if (terms == 0) {
            L::store(out + 8 * t, r);
            continue;
        }
        const bool merged = terms <= std::size_t(kMaxTerms);
        typename S::Acc s = S::mul(op(a + 8 * first), op(d + 8 * (t - first))), s2{};
#pragma GCC unroll 8
        for (std::size_t i = first + 1; i < end; ++i) S::mac(s, op(a + 8 * i), op(d + 8 * (t - i)));
        if (merged) {
#pragma GCC unroll 8
            for (std::size_t i = first; i < end; ++i) S::mac(s, op(b + 8 * i), op(c + 8 * (t - i)));
            r = M::add(r, S::reduce(s));
        } else {
            s2 = S::mul(op(b + 8 * first), op(c + 8 * (t - first)));
#pragma GCC unroll 8
            for (std::size_t i = first + 1; i < end; ++i) S::mac(s2, op(b + 8 * i), op(c + 8 * (t - i)));
            r = M::add(M::add(r, S::reduce(s)), S::reduce(s2));
        }
        L::store(out + 8 * t, r);
    }
}

// cross_units_inline, unrolled for n = k = 1, 2, 4, 8 (x86: cross_loads).
template <class L>
void cross_units(const u32* a, std::size_t n, const u32* b, std::size_t k, const u32* c, const u32* d, u32* out) {
    if (n == k) {
        if constexpr (std::is_same_v<Sums<L>, product_tree::detail::EvenOdd<L>> && L::lanes == 8) {
            switch (n) {
                case 1: return cross_loads<L, 1>(a, b, c, d, out);
                case 2: return cross_loads<L, 2>(a, b, c, d, out);
                case 4: return cross_loads<L, 4>(a, b, c, d, out);
                case 8: return cross_loads<L, 8>(a, b, c, d, out);
            }
        } else {
            switch (n) {
                case 1: return cross_units_inline<L>(a, 1, b, 1, c, d, out);
                case 2: return cross_units_inline<L>(a, 2, b, 2, c, d, out);
                case 4: return cross_units_inline<L>(a, 4, b, 4, c, d, out);
                case 8: return cross_units_inline<L>(a, 8, b, 8, c, d, out);
            }
        }
    }
    cross_units_inline<L>(a, n, b, k, c, d, out);
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

// M = prod (x - a_i) over m points: m + 1 words, canonical (M[m] = 1). In vectors over the
// coefficients, M = M x - a_i M from the top down.
template <class S>
void product_of_linear(const u32* a, std::size_t m, u32* M) {
    using O = Ops<S>;
    constexpr std::size_t W = S::lanes;
    const typename S::V r2 = S::broadcast(kR2);
    std::fill(M, M + m + 1, 0u);
    M[0] = 1;
    for (std::size_t i = 0; i < m; ++i) {  // degree i -> i + 1: M[j] = M[j - 1] - a M[j], j = i + 1 down to 0
        const u32 am = montgomery(a[i], kR2);
        const typename S::V av = O::template reduce<kP>(O::montgomery(S::broadcast(a[i]), r2));
        std::size_t j = i + 1;
        for (; j >= W; j -= W) {  // words [j - W + 1, j]
            const typename S::V high = S::load(M + j - W + 1), low = S::load(M + j - W);
            S::store(M + j - W + 1, O::sub(low, montgomery_canonical<S>(av, high)));
        }
        for (; j > 0; --j) M[j] = sub_mod(M[j - 1], montgomery(am, M[j]));
        M[0] = sub_mod(0, montgomery(am, M[0]));
    }
}

// The direct interpolation: M'(a_i) by Horner's rule (horner()), w_i = y_i / M'(a_i), then f[j] =
// sum over the points of w q[j] for q = M / (x - a) by synthetic division (q[j - 1] = M[j] + a
// q[j], from the top), on blocks of S::lanes points, kChains blocks at a time (a vector of sums
// per coefficient, its lanes summed at the end). M: m + 1 words (prod (x - a_i), M[m] = 1), a and
// y: m words, all canonical; f: m words; scratch: (S::lanes + 2) m words. Steps lazy (< 2P), as
// horner_step. Returns false, f unspecified, if some M'(a_i) is 0.
template <class S, std::size_t kChains = 4>
bool interpolate_small(const u32* M, std::size_t m, const u32* a, const u32* y, u32* f, u32* scratch) {
    using V = typename S::V;
    using O = Ops<S>;
    constexpr std::size_t W = S::lanes, B = kChains * W;
    u32* const acc = scratch;
    u32* const derivative = acc + W * m;
    u32* const values = derivative + m;
    for (std::size_t j = 0; j < m; ++j) derivative[j] = multiply_mod(u32(j + 1), M[j + 1]);
    horner<S>(derivative, m, a, m, values);
    const V r2 = S::broadcast(kR2);
    std::fill(acc, acc + W * m, 0u);
    for (std::size_t i0 = 0; i0 < m; i0 += B) {
        const std::size_t here = std::min(B, m - i0);
        alignas(64) u32 points[B] = {}, ys[B] = {}, check[B];
        std::copy_n(a + i0, here, points);
        std::copy_n(y + i0, here, ys);
        std::copy_n(values + i0, here, check);
        for (std::size_t l = 0; l < B; ++l) {
            if (l >= here) check[l] = 1;  // padding: no point
            else if (check[l] == 0) return false;
        }
        V am[kChains], q[kChains], w[kChains];
        for (std::size_t c = 0; c < kChains; ++c) {
            am[c] = montgomery_canonical<S>(S::load(points + W * c), r2);
            // 1 / M'(a) times 2^32, times y: w 2^32 (Montgomery form, for the products below)
            const V inverse = power_montgomery<S>(montgomery_canonical<S>(S::load(check + W * c), r2), kP - 2);
            w[c] = montgomery_canonical<S>(montgomery_canonical<S>(S::load(ys + W * c), inverse), r2);
            q[c] = S::broadcast(0);
        }
        for (std::size_t j = m; j > 0; --j) {  // acc[j - 1] += sum of w q[j - 1]
            const V c = S::broadcast(M[j]);
            V sum = S::load(acc + W * (j - 1));
#pragma GCC unroll 8
            for (std::size_t t = 0; t < kChains; ++t) {
                q[t] = horner_step<S>(q[t], am[t], c);
                sum = O::add(sum, O::template reduce<kP>(O::montgomery(w[t], O::template reduce<kP>(q[t]))));
            }
            S::store(acc + W * (j - 1), sum);
        }
    }
    for (std::size_t j = 0; j < m; ++j) {
        u32 s = 0;
        for (std::size_t l = 0; l < W; ++l) s = add_mod(s, acc[W * j + l]);
        f[j] = s;
    }
    return true;
}

// The Newton basis by synthetic divisions, B = K S::lanes of them at once in a systolic array of K
// vectors: c (n words of f in, f's Newton coefficients out) and points p (n - 1 words), all
// canonical; e: n + 2 B words of scratch. Division k by x - p_k (c[k..n) = c[k] + (x - p_k)
// c[k+1..n), from the top: c[j] += p_k c[j + 1]) runs in lane k - k0 of a block of B divisions;
// lane l works on coefficient j + l while lane 0 works on j, taking lane l - 1's last result (a
// lane shift), so a block is one pass over the coefficients, from the top: one step's latency
// per coefficient, for B divisions. Above f, zeros (lanes not started compute zeros there; lanes
// past the last division divide by x, which keeps c). Each step stores every lane at its own
// coefficient: lane B - 1's values (the block's last division) are the last stored at each
// coefficient, and the next block reads them.
template <class S, std::size_t K = 4>
void newton_small(u32* c, std::size_t n, const u32* p, u32* e) {
    using V = typename S::V;
    constexpr std::size_t W = S::lanes, B = K * W;
    if (n < 2) return;
    std::array<int, W> from{}, last{};
    for (std::size_t l = 0; l < W; ++l) from[l] = int(l ? l - 1 : 0), last[l] = int(W - 1);
    const typename S::Index up = S::index(from), top = S::index(last);
    const V r2 = S::broadcast(kR2);
    std::copy_n(c, n, e);
    std::fill(e + n, e + n + 2 * B, 0u);
    for (std::size_t k0 = 0; k0 + 1 < n; k0 += B) {
        alignas(64) u32 points[B] = {};
        for (std::size_t l = 0; l < B && k0 + l + 1 < n; ++l) points[l] = p[k0 + l];
        V pm[K], r[K];
        for (std::size_t v = 0; v < K; ++v) pm[v] = montgomery_canonical<S>(S::load(points + W * v), r2), r[v] = S::broadcast(0);
        for (std::size_t j = n; j-- > k0;) {  // lane 0 on coefficient j
            V x[K];
#pragma GCC unroll 4
            for (std::size_t v = 0; v < K; ++v)
                x[v] = S::template blend<1>(S::permute(r[v], up), v ? S::permute(r[v - 1], top) : S::broadcast(e[j]));
#pragma GCC unroll 4
            for (std::size_t v = 0; v < K; ++v) {
                r[v] = Ops<S>::add(x[v], montgomery_canonical<S>(pm[v], r[v]));
                S::store(e + j + W * v, r[v]);
            }
        }
    }
    std::copy_n(e, n, c);
}

// x^e 2^32 for x in Montgomery form, lazy: x < 2P in, < 2P out (montgomery's bounds hold, (2P)^2 <
// 2^32 P), the squarings one chain.
template <class S>
typename S::V power_lazy(typename S::V x, std::size_t e) {
    using O = Ops<S>;
    typename S::V r = S::broadcast(kR);
    for (; e; e >>= 1, x = O::montgomery(x, x))
        if (e & 1) r = O::montgomery(r, x);
    return r;
}

// v[i] = y[i] / v[i] for i < count (a multiple of L::lanes, at most 8 kLaneBase), but 0 where
// first + i >= m (v[i] read as 1 there): Montgomery's trick in kChains interleaved chains of prefix
// products over the vectors (vector k in chain k % kChains), one vector power for the lanes'
// inverses of all their products, then each chain back; lazy (< 2P) but the outputs, which are
// canonical. The steps' latency is the cost (a base's 16 vectors); kChains chains cut the prefix
// and back passes to a quarter. Returns false, v unspecified, if some v[i] with first + i < m is 0.
template <class L, std::size_t kChains = 4>
bool divide_slots(u32* v, const u32* y, std::size_t first, std::size_t m, std::size_t count) {
    using V = typename L::V;
    using O = Ops<L>;
    constexpr std::size_t W = L::lanes, C = kChains;
    const std::size_t vectors = count / W;
    if (first + count > m) std::fill(v + (m > first ? m - first : 0), v + count, 1u);
    const V r2 = L::broadcast(kR2);
    V prefix[8 * product_tree::kLaneBase / W], x[8 * product_tree::kLaneBase / W], total[C];
    for (std::size_t c = 0; c < C; ++c) total[c] = L::broadcast(kR);
    for (std::size_t k = 0; k < vectors; ++k) {
        x[k] = montgomery_canonical<L>(L::load(v + W * k), r2);
        prefix[k] = total[k % C] = O::montgomery(total[k % C], x[k]);
    }
    // The products of all but chain c's, and of all.
    V before[C], after[C];
    before[0] = after[C - 1] = L::broadcast(kR);
    for (std::size_t c = 1; c < C; ++c) before[c] = O::montgomery(before[c - 1], total[c - 1]);
    for (std::size_t c = C - 1; c-- > 0;) after[c] = O::montgomery(after[c + 1], total[c + 1]);
    const V all = O::template reduce<kP>(O::montgomery(before[C - 1], total[C - 1]));
    alignas(64) u32 lanes[W];
    L::store(lanes, all);
    for (std::size_t l = 0; l < W; ++l)
        if (lanes[l] == 0) return false;
    const V e = power_lazy<L>(all, kP - 2);  // (product 2^32)^(P-2) 2^32 = product^-1 2^32
    V inverse[C];  // 1 / chain c's product, times 2^32
    for (std::size_t c = 0; c < C; ++c) inverse[c] = O::montgomery(e, O::montgomery(before[c], after[c]));
    for (std::size_t k = vectors; k-- > 0;) {
        const std::size_t c = k % C;
        const V reciprocal = k >= C ? O::montgomery(inverse[c], prefix[k - C]) : inverse[c];
        inverse[c] = O::montgomery(inverse[c], x[k]);
        L::store(v + W * k, montgomery_canonical<L>(L::load(y + W * k), reciprocal));
    }
    return true;
}

}  // namespace fps::evaluation::detail
