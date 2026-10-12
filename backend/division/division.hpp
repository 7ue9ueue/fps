// Division with remainder of polynomials modulo P = 998244353 (backend kernel): f = q g + r with
// deg r < deg g, into caller memory. From lc-opt's lib/poly/division.hpp (its issue #81), rewritten
// on ntt::Transform's fused ops and series::inverse, so every target runs it. Design and
// measurements: notes.md.
//
//   fps::division::divide(f, g, q, r);    // f: n words, g: m >= 1 words with g[m - 1] != 0;
//                                         // q: quotient_size(n, m) words, r: m - 1 words, or none
//   fps::division::divide(f, g, q, r, scratch);   // in the caller's scratch_words(n, m, r) words
//
// f, g: words < P; q and r get canonical words. With d = m - 1: q has max(n - d, 0) words (its last
// one f's over g's leading coefficient when n > d); r has d words, f's then zeros when n <= d. An
// empty r skips the remainder (f / g). q, r and scratch apart from f, g and each other. Throws
// std::invalid_argument for other sizes, an empty g or g[m - 1] = 0, and std::length_error past
// the transforms' 2^25 words (n past about 2^25).
//
// With k = n - d, F = rev(f) = x^(n-1) f(1/x) and G = rev(g) (G[0] = g[d] != 0): rev(q) = Q =
// F / G mod x^k, and r = f - q g mod x^d. A plan picks the cheapest way by a cost model of the ops
// on this target (select.hpp, tuned per target):
// - Long division (k d small): k steps of rest -= c x^i g, vectors of Shoup products.
// - Blocks of s words (lc-opt's): h = 1 / G mod x^s (series::inverse), H = T_2s(h); block j of Q
//   is h R_j mod x^s for the residual R_j = (F - G Q_(<js))[js, js + s), by one cyclic product of
//   length 2s. The last block may be short (c words): its product at a shorter length, with its
//   own transform of h mod x^c, or by small::multiply. Residuals: if d < s ("tails"), (G Q)[js +
//   i] = (G T)[d + i] for T = Q[js - d, js), one cyclic product of length Lt >= 2d with the stored
//   transform of x^(Lt/2 - d) G (the upper half out, negated). Else ("windows") the sum over t <=
//   min(j, d / s + 1) of (W_t Q_(j-t))[s, 2s), W_t = G[(t-1)s, (t+1)s), from stored transforms
//   of W_t and of the last blocks, by one inverse_product (sums past 4 terms: leaf products in
//   memory first).
// - Remainder: (wrap) q g mod (x^L - 1) for L = bit_ceil(d): its word j < d is (q g)[j] + f[j + L]
//   + f[j + 2L] + ..., so r[j] = (f[j] + f[j + L] + ...) - (q g mod (x^L - 1))[j]. (split) for L'
//   = L/2 < d and k + d - L' <= L' + 1: q g mod (x^L' - 1), and (q g)[L', d) as words [k - 1, k - 1
//   + d - L') of q g[L' - k + 1, d) mod (x^L' - 1), both with one T_L'(q). (tail) with tails, rev(r)
//   = (F - G Q)[k, k + d) is one more tail product. (direct) small::multiply(q, g mod x^d).
//
// Memory: one scratch span (this thread's NTT region, ntt::detail::take_region, kept across calls,
// or the caller's): G, h and T(h), then series::inverse's scratch overlapping the blocks' arrays;
// the remainder's arrays from the start again. Transforms: series' per-thread ntt::Transform per
// length (series::detail::transform).
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include "backend/division/select.hpp"
#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/series/series.hpp"
#include "backend/simd/select.hpp"
#include "backend/small/small.hpp"

namespace fps::division {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 kP = ntt::kModulus;

// Words of the quotient of f (n words) by g (m >= 1 words).
inline std::size_t quotient_size(std::size_t n, std::size_t m) { return m && n >= m ? n - m + 1 : 0; }

namespace detail {

using Span = std::span<const u32>;
using Out = std::span<u32>;
using S = simd::Native;
using M = modarith::Ops<S, kP>;
using V = S::V;
constexpr std::size_t kLanes = S::lanes;
using series::detail::Arena;
using series::detail::footprint;
using series::detail::transform;

constexpr u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kP); }
constexpr u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
constexpr u32 kR = u32((u64(1) << 32) % kP);  // 2^32 mod P
constexpr u32 add1(u32 a, u32 b) { return a + b >= kP ? a + b - kP : a + b; }
constexpr u32 sub1(u32 a, u32 b) { return a >= b ? a - b : a + kP - b; }

inline constexpr std::size_t kLongDivision = std::size_t(1) << 12;  // long division when k d <= this
inline constexpr std::size_t kMaxWindows = 8;                       // stored window transforms
inline constexpr std::size_t kMaxDirect = 128;                      // the direct remainder: k <= this
inline constexpr int kMinLg = ntt::kMinLog, kMaxLg = ntt::kMaxLog;

constexpr int lg_of(std::size_t words) { return std::max(kMinLg, int(std::bit_width(words - 1))); }  // words >= 1

// ---- Element passes (simd::Native) ------------------------------------------------------------

inline S::Index reverse_index() {
    std::array<int, kLanes> lanes{};
    for (std::size_t l = 0; l < kLanes; ++l) lanes[l] = int(kLanes - 1 - l);
    return S::index(lanes);
}

// out[i] = top[-i] (with add: out[i] + top[-i] mod P) for i < count; top[-i] canonical.
template <bool add>
void reversed(const u32* top, u32* out, std::size_t count) {
    const S::Index rev = reverse_index();
    std::size_t i = 0;
    for (; i + kLanes <= count; i += kLanes) {
        const V x = S::permute(S::load(top - i - (kLanes - 1)), rev);
        S::store(out + i, add ? M::add(S::load(out + i), x) : x);
    }
    for (; i < count; ++i) out[i] = add ? add1(out[i], *(top - i)) : *(top - i);
}

// out[j] = sum over t of a[j + t out.size()] mod P.
inline void fold(Span a, Out out) {
    const std::size_t len = out.size(), first = std::min(len, a.size());
    std::copy_n(a.data(), first, out.data());
    std::fill(out.begin() + std::ptrdiff_t(first), out.end(), 0u);
    for (std::size_t from = len; from < a.size(); from += len) {
        const std::size_t count = std::min(len, a.size() - from);
        const u32* x = a.data() + from;
        std::size_t j = 0;
        for (; j + kLanes <= count; j += kLanes) S::store(out.data() + j, M::add(S::load(out.data() + j), S::load(x + j)));
        for (; j < count; ++j) out[j] = add1(out[j], x[j]);
    }
}

// r[j] = (f[j] + f[j + L] + f[j + 2L] + ...) - r[j] - (j < e ? r[L + j] : 0) for j < count <= L.
inline void fold_remainder(Span f, std::size_t len, Out r, std::size_t count, std::size_t e) {
    const std::size_t n = f.size();
    std::size_t j = 0;
    for (; j + kLanes <= count; j += kLanes) {
        V acc = S::broadcast(0);
        std::size_t at = j;
        for (; at + kLanes <= n; at += len) acc = M::add(acc, S::load(f.data() + at));
        if (at < n) {  // the last lanes of f
            alignas(64) u32 t[kLanes] = {};
            std::copy_n(f.data() + at, n - at, t);
            acc = M::add(acc, S::load(t));
        }
        V x = M::sub(acc, S::load(r.data() + j));
        if (j < e) {
            alignas(64) u32 t[kLanes] = {};
            std::copy_n(r.data() + len + j, std::min(kLanes, e - j), t);
            x = M::sub(x, S::load(t));
        }
        S::store(r.data() + j, x);
    }
    for (; j < count; ++j) {
        u32 acc = 0;
        for (std::size_t at = j; at < n; at += len) acc = add1(acc, f[at]);
        acc = sub1(acc, r[j]);
        r[j] = j < e ? sub1(acc, r[len + j]) : acc;
    }
}

// sum over i < count of a[i] b[i] mod P, canonical words.
inline u32 dot(const u32* a, const u32* b, std::size_t count) {
    V acc0 = S::broadcast(0), acc1 = acc0;
    std::size_t i = 0;
    for (; i + 2 * kLanes <= count; i += 2 * kLanes) {  // Montgomery products: 2^-32 a b
        acc0 = M::add(acc0, M::template reduce<kP>(M::montgomery(S::load(a + i), S::load(b + i))));
        acc1 = M::add(acc1, M::template reduce<kP>(M::montgomery(S::load(a + i + kLanes), S::load(b + i + kLanes))));
    }
    alignas(64) u32 lanes[kLanes];
    S::store(lanes, M::add(acc0, acc1));
    u64 sum = 0;
    for (const u32 x : lanes) sum += x;
    sum = mul(u32(sum % kP), kR);
    for (; i < count; ++i) sum = (sum + u64(a[i]) * b[i]) % kP;
    return u32(sum);
}

// out[i] = a[i] - b[i] mod P for i < count.
inline void subtract(const u32* a, const u32* b, u32* out, std::size_t count) {
    std::size_t i = 0;
    for (; i + kLanes <= count; i += kLanes) S::store(out + i, M::sub(S::load(a + i), S::load(b + i)));
    for (; i < count; ++i) out[i] = sub1(a[i], b[i]);
}

// ---- Long division ------------------------------------------------------------------------------


// x[t] = x[t] - c g[t] mod P for t < d, given cr = c 2^32 mod P (Montgomery products: no division).
inline void subtract_multiple(u32* x, const u32* g, u32 c, u32 cr, std::size_t d) {
    std::size_t t = 0;
    if (d >= kLanes) {
        const V w = S::broadcast(cr);
        for (; t + kLanes <= d; t += kLanes) {
            const V p = M::template reduce<kP>(M::montgomery(S::load(g + t), w));
            S::store(x + t, M::sub(S::load(x + t), p));
        }
    }
    for (; t < d; ++t) x[t] = sub1(x[t], mul(c, g[t]));
}

// q, r from f and g (d = g.size() - 1, k = q.size() = n - d >= 1); r of d words or empty. rest: n words.
inline void long_division(Span f, Span g, Out q, Out r, u32* rest) {
    const std::size_t d = g.size() - 1;
    std::copy(f.begin(), f.end(), rest);
    const u32 lead = power(g[d], kP - 2), lead_r = mul(lead, kR);
    for (std::size_t i = q.size(); i-- > 0;) {
        const u32 c = mul(rest[i + d], lead);
        q[i] = c;
        subtract_multiple(rest + i, g.data(), c, mul(rest[i + d], lead_r), d);
    }
    std::copy_n(rest, r.size(), r.data());
}

// ---- The plan -----------------------------------------------------------------------------------

// How the last block (c words) runs: its residual as the other blocks' (or F's, for one block),
// then h R mod x^c at length 2s with T_2s(h) (full; for one block at the least length >= 2c - 1
// with T(h)), at a shorter length with a transform of h mod x^c (short), or by small::multiply
// (small); or its residual by c dot products of g with Q, then small::multiply (dot).
enum class Last : std::uint8_t { full, short_length, small, dot };
enum class Remainder : std::uint8_t { none, direct, wrap, split, tail };

struct Plan {
    std::size_t n = 0, m = 0, k = 0, d = 0;
    bool long_division = false;
    std::size_t s = 0, blocks = 0, last = 0;  // blocks of s words, the last of `last` words
    bool tails = false;
    std::size_t reach = 0, windows = 0;  // windows: nonzero windows; stored ones
    Last last_method = Last::full;
    Remainder remainder = Remainder::none;
    int lg_tail = 0, lg_last = 0, lg_remainder = 0;
    double ns = std::numeric_limits<double>::infinity();

    // Coefficients of h.
    std::size_t precision() const { return blocks == 1 ? k : s; }
    // Blocks 1 .. residuals() take their residual from a transform (tails or windows).
    std::size_t residuals() const { return blocks - 1 - (last_method == Last::dot); }
    // Whether the tail products' transform of G is built.
    bool tail_transform() const { return tails && (residuals() > 0 || remainder == Remainder::tail); }
    // The length of T(h): 2s, the one block's product length, or none.
    std::size_t h_length() const {
        if (blocks > 1) return 2 * s;
        return last_method == Last::small ? 0 : std::size_t(1) << lg_last;
    }
    // Whether the last block's product is small::multiply.
    bool small_last() const { return last_method == Last::small || last_method == Last::dot; }
};

// The quotient by blocks of s with the last block's method; ns infinite if it cannot run.
inline Plan block_plan(const Cost& cost, std::size_t n, std::size_t m, std::size_t s, Last last) {
    Plan p;
    p.n = n, p.m = m, p.d = m - 1, p.k = n - p.d, p.s = s, p.last_method = last;
    const std::size_t k = p.k, d = p.d;
    p.blocks = (k + s - 1) / s;
    p.last = k - (p.blocks - 1) * s;
    const std::size_t B = p.blocks, c = p.last;
    const int lg2 = lg_of(2 * s);
    if ((B > 1 && lg2 > kMaxLg) || (B == 1 && (last == Last::short_length || last == Last::dot))) return p;
    p.tails = d < s;
    if (p.tails) p.lg_tail = lg_of(2 * d);
    const std::size_t R = p.residuals();
    double ns = cost.inverse(p.precision()) + cost.pass(3 * k);
    if (B > 1) ns += cost.forward(lg2) + double(B - 1) * cost.cyclic(lg2);
    if (R > 0 && p.tails) {
        ns += cost.forward(p.lg_tail) + double(R) * cost.cyclic(p.lg_tail);
    } else if (R > 0) {
        p.reach = d / s + 1;
        p.windows = std::min(p.reach, R);
        if (p.windows > kMaxWindows) return p;
        ns += double(p.windows + R) * cost.forward(lg2);  // T(W_t), and T(Q_j) for j < R
        for (std::size_t j = 1; j <= std::min(R, p.reach - 1); ++j) ns += cost.product(lg2, j);
        if (R >= p.reach) ns += double(R - p.reach + 1) * cost.product(lg2, p.reach);
    }
    // The last block's product (the only block's if B = 1).
    if (last == Last::dot) ns += cost.dot(c * std::min(d, k - c));
    if (p.small_last()) {
        ns += small::model_ns(small::kModel, c, c) + cost.pass(2 * c);
    } else if (B > 1 && last == Last::full) {
        p.lg_last = lg2;
        ns += cost.cyclic(lg2);
    } else {
        p.lg_last = lg_of(2 * c - 1);
        if (p.lg_last > kMaxLg || (B > 1 && p.lg_last >= lg2)) return p;
        ns += cost.forward(p.lg_last) + cost.cyclic(p.lg_last);
    }
    p.ns = ns;
    return p;
}

// The remainder's method added to a quotient plan (ns infinite if it cannot run).
inline Plan with_remainder(const Cost& cost, Plan p, Remainder how) {
    const std::size_t n = p.n, k = p.k, d = p.d;
    p.remainder = how;
    if (how == Remainder::none) return p;
    const int lg = lg_of(d);
    switch (how) {
        case Remainder::none: break;
        case Remainder::direct:
            if (k > kMaxDirect) p.ns = std::numeric_limits<double>::infinity();
            else p.ns += small::model_ns(small::kModel, k, d) + cost.pass(2 * d);
            break;
        case Remainder::wrap:
            p.lg_remainder = lg;
            p.ns += cost.forward(lg) + cost.cyclic(lg) + cost.pass(n + d + (k > (std::size_t(1) << lg) ? k : 0));
            break;
        case Remainder::split: {
            const std::size_t half = std::size_t(1) << (lg - 1), e = d - std::min(d, half);
            p.lg_remainder = lg - 1;
            if (lg - 1 < kMinLg || e == 0 || k + e > half + 1) p.ns = std::numeric_limits<double>::infinity();
            else p.ns += cost.forward(lg - 1) + 2 * cost.cyclic(lg - 1) + cost.pass(n + d + half);
            break;
        }
        case Remainder::tail:
            if (p.long_division || !(p.tails || p.blocks == 1) || k < d) {
                p.ns = std::numeric_limits<double>::infinity();
                break;
            }
            if (!p.tails) p.tails = true, p.lg_tail = lg_of(2 * d);  // one block, d >= s
            if (p.lg_tail > kMaxLg) {
                p.ns = std::numeric_limits<double>::infinity();
                break;
            }
            if (p.blocks == 1) p.ns += cost.forward(p.lg_tail);  // T(G) for the remainder alone
            p.ns += cost.cyclic(p.lg_tail) + cost.pass(d);
            break;
    }
    return p;
}

// The cheapest plan for f of n and g of m words, n >= m >= 1, the remainder if `remainder`.
inline Plan plan(std::size_t n, std::size_t m, bool remainder) {
    const Cost cost(costs.get());
    Plan best;
    best.n = n, best.m = m, best.d = m - 1, best.k = n - best.d, best.long_division = true;
    const std::size_t k = best.k, d = best.d;
    best.ns = cost.long_division(k, d);
    if (k * d <= kLongDivision) return best;
    constexpr Remainder kMethods[] = {Remainder::direct, Remainder::wrap, Remainder::split, Remainder::tail};
    for (std::size_t s = 32;; s *= 2) {
        for (const Last last : {Last::full, Last::short_length, Last::small, Last::dot}) {
            const Plan q = block_plan(cost, n, m, s, last);
            if (!(q.ns < best.ns)) continue;
            if (!remainder || d == 0) {
                best = q;
                continue;
            }
            for (const Remainder how : kMethods) {
                const Plan p = with_remainder(cost, q, how);
                if (p.ns < best.ns) best = p;
            }
        }
        if (s >= k || 2 * s > (std::size_t(1) << kMaxLg)) break;
    }
    return best;
}

// ---- Memory -------------------------------------------------------------------------------------

// Words of G: the inverse's precision, the windows (W_t for t <= windows).
inline std::size_t divisor_words(const Plan& p) {
    return std::min(p.m, std::max(p.precision(), p.windows ? (p.windows + 1) * p.s : 0));
}

// Words of the work array of the fused ops in the blocks.
inline std::size_t block_work(const Plan& p) {
    std::size_t w = p.h_length();
    if (p.tail_transform()) w = std::max(w, std::size_t(1) << p.lg_tail);
    if (p.last_method == Last::short_length) w = std::max(w, std::size_t(1) << p.lg_last);
    return w;
}

// Arena words of the quotient (blocks) and of the remainder after it, in the order taken.
inline std::size_t quotient_words(const Plan& p) {
    const std::size_t len = 2 * p.s;
    std::size_t later = footprint(p.k) + footprint(block_work(p));
    if (p.tail_transform()) later += footprint(std::size_t(1) << p.lg_tail);
    if (p.remainder == Remainder::tail) later += footprint(p.d);
    if (p.windows) later += 2 * p.windows * footprint(len) + (p.windows > 4 ? footprint(len) : 0);
    if (p.last_method == Last::short_length && p.blocks > 1) later += footprint(std::size_t(1) << p.lg_last);
    if (p.small_last()) later += footprint(2 * p.last);
    const std::size_t h = p.h_length() ? footprint(p.h_length()) : 0;
    return footprint(divisor_words(p)) + footprint(p.precision()) + h +
           std::max(footprint(series::inverse_scratch(p.precision())), later);
}

inline std::size_t remainder_words(const Plan& p) {
    const std::size_t len = std::size_t(1) << p.lg_remainder;
    switch (p.remainder) {
        case Remainder::direct: return footprint(p.k + p.d - 1);
        case Remainder::wrap: return 4 * footprint(len);
        case Remainder::split: return 4 * footprint(len);
        default: return 0;
    }
}

inline std::size_t words(const Plan& p) {
    if (p.long_division) return footprint(p.n);
    return std::max(quotient_words(p), remainder_words(p));
}

// ---- Blocks -------------------------------------------------------------------------------------

// Q = rev(q) block by block into a buffer of k words, each block written reversed into q once
// done; with the tail remainder also r.
inline void divide_blocks(const Plan& p, Span f, Span g, Out q, Out r, Arena& arena) {
    const std::size_t n = p.n, k = p.k, d = p.d, s = p.s, B = p.blocks, len = 2 * s;
    const std::size_t precision = p.precision();
    const Out G(arena.take(divisor_words(p)), divisor_words(p));
    std::reverse_copy(g.end() - std::ptrdiff_t(G.size()), g.end(), G.begin());
    const Out h(arena.take(precision), precision);
    u32* const H = p.h_length() ? arena.take(p.h_length()) : nullptr;
    {
        Arena rest = arena;  // the inverse's scratch, under the arrays taken next
        const std::size_t words = series::inverse_scratch(precision);
        series::inverse(Span(G.data(), std::min(G.size(), precision)), h, Out(rest.take(words), words));
    }
    if (H) transform(std::countr_zero(p.h_length())).forward(Span(h), H);

    u32* const qs = arena.take(k);
    u32* const work = arena.take(block_work(p));
    // F[i] = f[n - 1 - i]: words [at, at + count) of F into out, or added to it.
    const auto take_f = [&](std::size_t at, u32* out, std::size_t count, bool add) {
        if (add) reversed<true>(f.data() + (n - 1 - at), out, count);
        else reversed<false>(f.data() + (n - 1 - at), out, count);
    };
    // Tails: Gt = T_Lt(x^(Lt/2 - d) G), so (G T)[d, 2d) is words [Lt/2, Lt/2 + d) of T x^(Lt/2 - d) G
    // (degree < Lt: no wrap).
    const ntt::Transform* tt = nullptr;
    u32* Gt = nullptr;
    std::size_t tail_half = 0;
    if (p.tail_transform()) {
        tt = &transform(p.lg_tail);
        tail_half = tt->length() / 2;
        Gt = arena.take(tt->length());
        u32* const shifted = work;  // tail_half + 1 words, free until the first product
        std::fill_n(shifted, tail_half - d, 0u);
        std::reverse_copy(g.begin(), g.end(), shifted + (tail_half - d));
        tt->forward(Span(shifted, tail_half + 1), Gt);
    }
    // out = -(G Q)[at, at + count) for count <= d, from T = Q[at - d, at).
    const auto tail_product = [&](std::size_t at, u32* out, std::size_t count) {
        tt->cyclic_product(Span(qs + at - d, d), Gt, work, Out(out, count), {.offset = tail_half, .negate = true});
    };
    u32* const tail_out = p.remainder == Remainder::tail ? arena.take(d) : nullptr;
    // Windows: W[i] = T_2s(W_(i+1)), W_t = G[(t-1)s, (t+1)s); ring[j % windows] = T_2s(Q_j).
    const ntt::Transform* t2 = B > 1 ? &transform(std::countr_zero(len)) : nullptr;
    std::array<u32*, kMaxWindows> W{}, ring{};
    for (std::size_t i = 0; i < p.windows; ++i) {
        W[i] = arena.take(len), ring[i] = arena.take(len);
        const std::size_t from = i * s;
        t2->forward(Span(G.data() + from, std::min(len, G.size() - from)), W[i]);
    }
    u32* const sum = p.windows > 4 ? arena.take(len) : nullptr;  // leaf products past 4 terms
    u32* Hc = nullptr;
    if (p.last_method == Last::short_length && B > 1) {
        Hc = arena.take(std::size_t(1) << p.lg_last);
        transform(p.lg_last).forward(Span(h.data(), p.last), Hc);
    }
    u32* const product = p.small_last() ? arena.take(2 * p.last) : nullptr;
    const std::size_t residuals = p.residuals();

    for (std::size_t j = 0; j < B; ++j) {
        const std::size_t at = j * s, count = j + 1 < B ? s : p.last;
        u32* const block = qs + at;
        const Out R(block, count);
        // R = (F - G Q)[at, at + count).
        if (j == 0) {
            take_f(0, block, count, false);
        } else if (j > residuals) {  // the last block by dot products: words of Q below at only
            take_f(at, block, count, false);
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t first = d > at + i ? d - at - i : 0;  // g[b] Q[at + i - d + b], b in [first, d - i)
                if (first < d - std::min(d, i)) block[i] = sub1(block[i], dot(g.data() + first, qs + at + i - d + first, d - i - first));
            }
        } else if (p.tails) {
            const std::size_t head = std::min(count, d);
            tail_product(at, block, head);
            take_f(at, block, head, true);
            take_f(at + head, block + head, count - head, false);
        } else {
            const std::size_t terms = std::min(j, p.reach);
            const auto stored = [&](std::size_t t) { return ring[(j - 1 - t) % p.windows]; };  // T(Q_(j-1-t))
            std::array<ntt::Term, 4> sums{};
            std::size_t used = 0, t = 0;
            if (terms > 4) {
                t2->multiply(W[0], stored(0), sum);
                for (t = 1; t + 3 < terms; ++t) t2->multiply_add(sum, W[t], stored(t));
                sums[used++] = {sum, nullptr};
            }
            for (; t < terms; ++t) sums[used++] = {W[t], stored(t)};
            t2->inverse_product(std::span<const ntt::Term>(sums.data(), used), work, R, {.offset = s, .negate = true});
            take_f(at, block, count, true);
        }
        // Q_j = h R mod x^count.
        if (j + 1 < B) {
            t2->cyclic_product(R, H, work, R);
            if (p.windows && j < residuals) t2->forward(Span(block, s), ring[j % p.windows]);
        } else if (p.small_last()) {
            small::multiply<kP>(Span(h.data(), count), Span(R), Out(product, 2 * count - 1));
            std::copy_n(product, count, block);
        } else if (B > 1 && p.last_method == Last::full) {
            t2->cyclic_product(R, H, work, R);
        } else {
            transform(p.lg_last).cyclic_product(R, B > 1 ? Hc : H, work, R);
        }
        reversed<false>(block + count - 1, q.data() + (k - at - count), count);  // q[k - 1 - x] = Q[x]
    }
    if (tail_out) {  // r[d - 1 - i] = f[d - 1 - i] - (G Q)[k + i]
        tail_product(k, tail_out, d);
        std::copy_n(f.data(), d, r.data());
        reversed<true>(tail_out + d - 1, r.data(), d);
    }
}

// ---- Remainder ------------------------------------------------------------------------------------

// r = f - q g mod x^d from q g mod (x^L - 1), L = 2^lg >= d.
inline void remainder_wrap(const Plan& p, Span f, Span g, Span q, Out r, Arena& arena) {
    const ntt::Transform& t = transform(p.lg_remainder);
    const std::size_t len = t.length();
    u32* const A = arena.take(len);
    u32* const work = arena.take(len);
    if (g.size() > len) {  // g.size() = len + 1
        u32* const folded = arena.take(len);
        fold(g, Out(folded, len));
        g = Span(folded, len);
    }
    t.forward(g, A);
    if (q.size() > len) {
        u32* const folded = arena.take(len);
        fold(q, Out(folded, len));
        q = Span(folded, len);
    }
    t.cyclic_product(q, A, work, r);
    fold_remainder(f, len, r, r.size(), 0);
}

// The same for d = L + e, 0 < e, k + e <= L + 1, L = 2^lg: with c = q g mod (x^L - 1), for j < L,
// r[j] = (f[j] + f[j + L] + ...) - c[j] - (j < e ? r[L + j] : 0), and r[L + i] = f[L + i] - (q g)[L
// + i], words [k - 1, k - 1 + e) of q g[L - k + 1, d) (no wrap reaches them).
inline void remainder_split(const Plan& p, Span f, Span g, Span q, Out r, Arena& arena) {
    const ntt::Transform& t = transform(p.lg_remainder);
    const std::size_t len = t.length(), k = q.size(), d = r.size(), e = d - len;
    u32* const Tq = arena.take(len);
    u32* const work = arena.take(len);
    u32* const folded = arena.take(len);
    u32* const middle = arena.take(k - 1 + e);
    t.forward(q, Tq);
    t.cyclic_product(g.subspan(len + 1 - k, k + e - 1), Tq, work, Out(middle, k - 1 + e));
    subtract(f.data() + len, middle + (k - 1), r.data() + len, e);
    fold(g, Out(folded, len));
    t.cyclic_product(Span(folded, len), Tq, work, r.first(len));
    fold_remainder(f, len, r, len, e);
}

// r = f - (q g mod x^d) by small::multiply.
inline void remainder_direct(Span f, Span g, Span q, Out r, Arena& arena) {
    const std::size_t d = r.size(), k = q.size();
    u32* const product = arena.take(k + d - 1);
    small::multiply<kP>(q, g.first(d), Out(product, k + d - 1));
    subtract(f.data(), product, r.data(), d);
}

inline void run(const Plan& p, Span f, Span g, Out q, Out r, u32* memory, std::size_t words) {
    if (p.long_division) {
        Arena arena(memory, words);
        return long_division(f, g, q, r, arena.take(p.n));
    }
    {
        Arena arena(memory, words);
        divide_blocks(p, f, g, q, r, arena);
    }
    Arena arena(memory, words);
    switch (p.remainder) {
        case Remainder::none:
        case Remainder::tail: return;
        case Remainder::direct: return remainder_direct(f, g, q, r, arena);
        case Remainder::wrap: return remainder_wrap(p, f, g, q, r, arena);
        case Remainder::split: return remainder_split(p, f, g, q, r, arena);
    }
}

inline void check(std::size_t n, std::size_t m, Span g, std::size_t q, std::size_t r) {
    if (m == 0 || g[m - 1] == 0) throw std::invalid_argument("fps::division::divide: g empty or g[m - 1] = 0");
    if (q != quotient_size(n, m) || (r != 0 && r != m - 1))
        throw std::invalid_argument("fps::division::divide: q needs quotient_size(n, m) words, r m - 1 or none");
}

inline constexpr std::size_t kAlignSlack = 16;

}  // namespace detail

// Words of scratch divide() takes for f of n and g of m >= 1 words (with the remainder or not).
inline std::size_t scratch_words(std::size_t n, std::size_t m, bool remainder = true) {
    if (quotient_size(n, m) == 0) return 0;
    return detail::kAlignSlack + detail::words(detail::plan(n, m, remainder));
}

// q, r with f = q g + r, deg r < m - 1 (above). scratch: none (this thread's region) or
// scratch_words(n, m, !r.empty()) words, any alignment.
inline void divide(std::span<const u32> f, std::span<const u32> g, std::span<u32> q, std::span<u32> r,
                   std::span<u32> scratch = {}) {
    const std::size_t n = f.size(), m = g.size();
    detail::check(n, m, g, q.size(), r.size());
    if (q.empty()) {  // n <= d: r = f, then zeros
        if (r.empty()) return;
        std::copy(f.begin(), f.end(), r.begin());
        std::fill(r.begin() + std::ptrdiff_t(std::min(n, r.size())), r.end(), 0u);
        return;
    }
    const detail::Plan p = detail::plan(n, m, !r.empty());
    if (!(p.ns < std::numeric_limits<double>::infinity()))
        throw std::length_error("fps::division::divide: longer than the transforms' 2^25 words");
    const std::size_t words = detail::words(p) + detail::kAlignSlack;
    if (!scratch.empty()) {
        if (scratch.size() < words) throw std::invalid_argument("fps::division::divide: scratch too small");
        return detail::run(p, f, g, q, r, scratch.data(), scratch.size());
    }
    const series::detail::RegionGuard guard{ntt::detail::take_region(words * sizeof(u32))};
    detail::run(p, f, g, q, r, reinterpret_cast<u32*>(guard.region.start), guard.region.bytes / sizeof(u32));
}

}  // namespace fps::division

