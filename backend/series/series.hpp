// Power series modulo P = 998244353: inverse, log and exp into caller memory (backend kernel).
// From lc-opt's lib/poly/{inverse,log,exp}.hpp, rewritten on ntt::Transform's leaf form, so every
// target runs it. Design and measurements: notes.md.
//
//   fps::series::inverse(f, out);    // out = 1 / f mod x^n, n = out.size(); f[0] != 0; out apart from f
//   fps::series::log(f, out);        // out = log f mod x^n; f[0] = 1, n <= P; out may overlap f
//   fps::series::exp(f, out);        // out = exp f mod x^n; f[0] = 0, n <= P; out may overlap f
//
//   // The same in the caller's scratch, of inverse_scratch(n) (log_scratch, exp_scratch) words,
//   // any alignment, instead of this thread's memory (a shorter one: std::invalid_argument):
//   fps::series::inverse(f, out, scratch);
//
// f: words < P, of which f mod x^n is read (a shorter f has zeros past its end). The preconditions
// on f[0] and n are not checked (the frontend checks them); n past kMaxSize throws
// std::length_error before out is written. If out overlaps f, log copies f first, and exp reads
// all of f before it writes out.
//
// Newton iteration on ntt::Transform's fused ops (ntt.hpp: sources and outputs in caller memory, a
// cyclic product or an inverse of leaf products in one depth-first pass, half outputs), in units
// of T(n) (a transform of n words) and LP(n) (leaf products of n words), as lc-opt's:
// - inverse: k -> 2k with transforms of 2k: G = T(g mod x^k) (a half-zero source), e = (f g)[k,
//   k + c) by a cyclic product (upper half out), then g[k, k + c) = -(e g mod x^c) by another (in
//   place, negated): 5 transforms and 2 leaf products of 2k a step, 10 T(n) + 4 LP(n). A last step
//   to n <= 3k/2 runs at length k: 6 T(k) + 3 LP(k). Below kInverseBase coefficients, the recurrence.
// - log f = integral of f' / f mod x^(n-1) in B <= 4 blocks of k coefficients (lc-opt's blocked
//   division): h = 1 / f mod x^k, then q_j = h (f' - f Q)[jk, (j+1)k) mod x^k, Q = q mod x^(jk),
//   (f Q)[jk, (j+1)k) = sum over i < j of (W_(j-i) q_i)[k, 2k), W_t = f[(t-1)k, (t+1)k), as one
//   inverse of a sum of leaf products of kept transforms of length 2k. About 11.5 T(n) + 6 LP(n).
// - exp: m -> 2m keeping h = 1 / g mod x^(m/2) and H = T_m(h), q = x f' (lc-opt's step, notes.md);
//   8 transforms and 3.5 leaf products of 2m a step. The last step, to n <= 2m, runs at length m
//   in two blocks of m/2 (lc-opt's exp_last_step): 14 transforms and 7 leaf products of m.
// - Divisions by indices (log's integral, exp's s): Montgomery's batch inversion (lc-opt's
//   divide_by_index), divisors below 2^13 from a table.
// Against lc-opt in the same run (Zen 3, n = 2^14 .. 2^21), and against main: notes/112-r1.md.
//
// Memory: this thread's region of the NTT (ntt::detail::take_region, kept across calls), or the
// caller's scratch: with N = bit_ceil(n), 2 N words for inverse, about n + 2 B (n / 2 + 80) for
// log (B <= 4 blocks) and n + 5.5 N for exp. The transforms' tables: one ntt::Transform per
// length, kept per thread, 2^lg bytes each.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"

namespace fps::series {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 kP = ntt::kModulus;
inline constexpr std::size_t kMaxSize = std::size_t(1) << ntt::kMaxLog;  // the longest n

namespace detail {

using Span = std::span<const u32>;
using Out = std::span<u32>;
using S = simd::Native;
using M = modarith::Ops<S, kP>;
using V = S::V;
constexpr std::size_t kLanes = S::lanes;

inline constexpr std::size_t kInverseBase = 32;  // inverse: the recurrence up to this many terms
inline constexpr std::size_t kLogBase = 64;      // log and exp: the recurrences up to this many
inline constexpr std::size_t kExpBase = 64;
static_assert(kExpBase <= kLogBase);
inline constexpr std::size_t kLogBlocks = 4;

constexpr u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kP); }
constexpr u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
constexpr u32 kR = u32((u64(1) << 32) % kP);  // 2^32 mod P

// ---- Memory ---------------------------------------------------------------------------------

// Words after each array: ntt::kPadding read by the transforms, and the rest so that arrays of a
// power-of-two length sit 320 bytes apart modulo 4 KiB (4K aliasing, backend/ntt/notes.md).
inline constexpr std::size_t kGap = 80;
constexpr std::size_t footprint(std::size_t words) { return (words + 15) / 16 * 16 + kGap; }

// 64-byte aligned arrays carved from one span, in order.
class Arena {
public:
    Arena(u32* start, std::size_t words) : next_(start), end_(start + words) {
        const std::size_t skip = (64 - reinterpret_cast<std::uintptr_t>(start) % 64) % 64 / sizeof(u32);
        next_ += skip;
    }
    u32* take(std::size_t words) {
        u32* p = next_;
        next_ += footprint(words);
        if (next_ > end_) throw std::logic_error("fps::series: scratch too small");
        return p;
    }

private:
    u32 *next_, *end_;
};
inline constexpr std::size_t kAlignSlack = 16;

// This thread's transform of length 2^lg, kept across calls.
inline const ntt::Transform& transform(int lg) {
    thread_local std::array<std::unique_ptr<ntt::Transform>, ntt::kMaxLog + 1> cache;
    if (!cache[lg]) cache[lg] = std::make_unique<ntt::Transform>(lg);
    return *cache[lg];
}

// Gives back the NTT's region of this thread on any exit.
struct RegionGuard {
    ntt::detail::Region region;
    ~RegionGuard() { ntt::detail::give_region(region); }
};

// ---- Element-wise passes (simd::Native) -----------------------------------------------------

inline void zero(u32* p, std::size_t n) { std::fill_n(p, n, 0u); }

// dst[i] = (first + i) src[i] mod P; dst may be src.
inline void multiply_by_index(const u32* src, std::size_t first, u32* dst, std::size_t n) {
    std::size_t i = 0;
    if (n >= kLanes) {
        alignas(64) u32 start[kLanes];
        for (std::size_t l = 0; l < kLanes; ++l) start[l] = mul(u32((first + l) % kP), kR);
        V index = S::load(start);
        const V step = S::broadcast(mul(u32(kLanes), kR));
        for (; i + kLanes <= n; i += kLanes, index = M::add(index, step))
            S::store(dst + i, M::template reduce<kP>(M::montgomery(S::load(src + i), index)));
    }
    for (; i < n; ++i) dst[i] = mul(u32((first + i) % kP), src[i]);
}

// The words [at, at + lanes) of p (n words), zero past n.
inline V load_part(const u32* p, std::size_t at, std::size_t n) {
    if (at + kLanes <= n) return S::load(p + at);
    alignas(64) u32 t[kLanes] = {};
    for (std::size_t l = 0; at + l < n; ++l) t[l] = p[at + l];
    return S::load(t);
}

// out[i] = a(i) / (first + i) mod P, canonical, for i < n; a(at): the vector of a(at .. at + lanes)
// in [0, 2P) (lanes past n are dropped; a must not read out). first >= 1, first + n <= P. lc-opt's
// divide_by_index: Montgomery's batch inversion in 4 interleaved chains of lanes, out holding the
// prefix products of blocks of 4 lanes integers (each step times 2^-32; the backward pass cancels
// it): 3 Montgomery products per word and one by a(i), no table of reciprocals. About 0.8 us a
// call besides (the lanes' totals inverted one by one): divide_by_index below takes divisors under
// kSmallDivisors from a table.
template <class Load>
void divide_by_batch(std::size_t first, u32* out, std::size_t n, Load a) {
    constexpr std::size_t kChains = 4, kBlock = kChains * kLanes;
    if (n == 0) return;
    alignas(64) u32 lanes[kLanes];
    for (std::size_t l = 0; l < kLanes; ++l) lanes[l] = u32(l);
    const V step = S::broadcast(u32(kBlock)), lane = S::load(lanes);
    // Lanes past n count as 1, in both passes.
    const auto masked = [n](V x, std::size_t at) {
        if (at + kLanes <= n) return x;
        alignas(64) u32 t[kLanes];
        S::store(t, x);
        for (std::size_t l = 0; l < kLanes; ++l)
            if (at + l >= n) t[l] = 1;
        return S::load(t);
    };
    V x[kChains], prefix[kChains];
    for (std::size_t v = 0; v < kChains; ++v) {
        x[v] = S::add(S::broadcast(u32(first + kLanes * v)), lane);
        prefix[v] = masked(x[v], kLanes * v);
    }
    const std::size_t last = (n - 1) / kBlock * kBlock;  // the last block
    for (std::size_t b = kBlock; b <= last; b += kBlock) {
        for (std::size_t v = 0; v < kChains; ++v) {
            S::store(out + b - kBlock + kLanes * v, prefix[v]);
            x[v] = S::add(x[v], step);
            prefix[v] = M::montgomery(prefix[v], masked(x[v], b + kLanes * v));
        }
    }
    // inverse[l] = 2^32 / (the prefix product of lane l), by one more batch inversion.
    alignas(64) u32 total[kBlock], inverse[kBlock];
    for (std::size_t v = 0; v < kChains; ++v) S::store(total + kLanes * v, M::template reduce<kP>(prefix[v]));
    u32 running = 1;
    for (std::size_t l = 0; l < kBlock; ++l) inverse[l] = running, running = mul(running, total[l]);
    running = mul(power(running, kP - 2), kR);
    for (std::size_t l = kBlock; l-- > 0;) inverse[l] = mul(inverse[l], running), running = mul(running, total[l]);
    // Backward: 1 / x = prefix(b - 1) inverse(b) 2^-32 (times 2^32, which the product with a
    // removes), inverse(b - 1) = inverse(b) x 2^-32.
    V inv[kChains];
    for (std::size_t v = 0; v < kChains; ++v) inv[v] = S::load(inverse + kLanes * v);
    const auto put = [&](std::size_t at, V reciprocal) {
        const V q = M::template reduce<kP>(M::montgomery(reciprocal, a(at)));
        if (at + kLanes <= n) return S::store(out + at, q);
        S::store_first(out + at, q, int(n - at));
    };
    for (std::size_t b = last; b > 0; b -= kBlock) {
        for (std::size_t v = 0; v < kChains; ++v) {
            const std::size_t at = b + kLanes * v;
            if (at < n) put(at, M::montgomery(S::load(out + at - kBlock), inv[v]));
            inv[v] = M::montgomery(inv[v], masked(x[v], at));
            x[v] = S::sub(x[v], step);
        }
    }
    for (std::size_t v = 0; v < kChains; ++v)
        if (kLanes * v < n) put(kLanes * v, inv[v]);
}

// out[c] = (first + 1 + c) f[first + 1 + c] mod P (the derivative's coefficient first + c), minus
// s[c] if s is not null, for c < count; f is zero past its end. out may be s.
inline void derivative_block(Span f, std::size_t first, std::size_t count, const u32* s, u32* out) {
    const std::size_t avail = f.size() > first + 1 ? std::min(count, f.size() - first - 1) : 0;
    const u32* from = f.data() + first + 1;
    std::size_t c = 0;
    if (count >= kLanes) {
        alignas(64) u32 start[kLanes];
        for (std::size_t l = 0; l < kLanes; ++l) start[l] = mul(u32((first + 1 + l) % kP), kR);
        V index = S::load(start);
        const V step = S::broadcast(mul(u32(kLanes), kR));
        for (; c + kLanes <= count; c += kLanes, index = M::add(index, step)) {
            const V d = M::template reduce<kP>(M::montgomery(load_part(from, c, avail), index));
            S::store(out + c, s ? M::sub(d, S::load(s + c)) : d);
        }
    }
    for (; c < count; ++c) {
        const u32 d = c < avail ? mul(u32((first + 1 + c) % kP), from[c]) : 0;
        out[c] = s ? (d >= s[c] ? d - s[c] : d + kP - s[c]) : d;
    }
}

// 2^32 / i mod P for 0 < i < kSmallDivisors (entry 0 unused): one table, built on first use.
inline constexpr std::size_t kSmallDivisors = std::size_t(1) << 13;
inline const u32* small_reciprocals() {
    static const std::array<u32, kSmallDivisors> table = [] {
        std::array<u32, kSmallDivisors> inverse{}, t{};
        inverse[1] = 1;
        for (std::size_t i = 2; i < kSmallDivisors; ++i) inverse[i] = mul(kP - kP / u32(i), inverse[kP % i]);
        for (std::size_t i = 1; i < kSmallDivisors; ++i) t[i] = mul(inverse[i], kR);
        return t;
    }();
    return table.data();
}

// divide_by_batch, with the divisors below kSmallDivisors from small_reciprocals(): one Montgomery
// product per word, no cost per call (log and exp of a few thousand words call it per block).
template <class Load>
void divide_by_index(std::size_t first, u32* out, std::size_t n, Load a) {
    std::size_t i = 0;
    if (first < kSmallDivisors) {
        const u32* r = small_reciprocals() + first;
        const std::size_t m = std::min(n, kSmallDivisors - first);
        for (; i < m; i += kLanes) {
            const V q = M::template reduce<kP>(M::montgomery(a(i), load_part(r, i, m)));
            if (i + kLanes <= m) S::store(out + i, q);
            else S::store_first(out + i, q, int(m - i));
        }
        i = m;
    }
    if (i < n) divide_by_batch(first + i, out + i, n - i, [&](std::size_t at) { return a(i + at); });
}

// ---- The recurrences for few coefficients -----------------------------------------------------

// 1 / i mod P for i < kLogBase (entry 0 unused).
inline constexpr auto kSmallInverses = [] {
    std::array<u32, kLogBase> r{};
    for (std::size_t i = 1; i < kLogBase; ++i) r[i] = power(u32(i), kP - 2);
    return r;
}();

// sum over lo <= j < hi of a[j] b[i - j] mod P: 64-bit sums of 16 products < P^2 (< 2^64), each
// reduced.
inline u32 dot(const u32* a, const u32* b, std::size_t i, std::size_t lo, std::size_t hi) {
    u64 total = 0;
    for (std::size_t j0 = lo; j0 < hi; j0 += 16) {
        u64 sum = 0;
        const std::size_t end = std::min(hi, j0 + 16);
        for (std::size_t j = j0; j < end; ++j) sum += u64(a[j]) * b[i - j];
        total += sum % kP;
    }
    return u32(total % kP);
}

// g = 1 / f mod x^n, n = g.size() <= kInverseBase: f0 g_i = -sum_(0<j<=i) f_j g_(i-j).
inline void inverse_direct(Span f, Out g) {
    u32 a[kInverseBase] = {};
    std::copy_n(f.begin(), std::min(f.size(), g.size()), a);
    const u32 c = power(a[0], kP - 2);
    g[0] = c;
    for (std::size_t i = 1; i < g.size(); ++i) {
        const u32 s = dot(a, g.data(), i, 1, i + 1);
        g[i] = mul(s ? kP - s : 0, c);
    }
}

// g = log f mod x^n, n = g.size() <= kLogBase, f[0] = 1, from G_i = i g_i:
// G_i = i f_i - sum_(0<k<i) G_k f_(i-k).
inline void log_direct(Span f, Out g) {
    const std::size_t n = g.size();
    u32 a[kLogBase] = {}, G[kLogBase] = {};
    std::copy_n(f.begin(), std::min(f.size(), n), a);
    g[0] = 0;
    for (std::size_t i = 1; i < n; ++i) {
        const u32 s = dot(G, a, i, 1, i), d = mul(u32(i), a[i]);
        G[i] = d >= s ? d - s : d + kP - s;
        g[i] = mul(G[i], kSmallInverses[i]);
    }
}

// g = exp f mod x^n, n = g.size() <= kExpBase, from q = x f': i g_i = sum_(0<k<=i) q_k g_(i-k).
inline void exp_direct(const u32* q, Out g) {
    g[0] = 1;
    for (std::size_t i = 1; i < g.size(); ++i) g[i] = mul(dot(q, g.data(), i, 1, i + 1), kSmallInverses[i]);
}

// ---- inverse ----------------------------------------------------------------------------------

// Arrays of inverse_newton for n coefficients: 2 of bit_ceil(n) words.
inline std::size_t inverse_words(std::size_t n) { return n <= kInverseBase ? 0 : 2 * footprint(std::bit_ceil(n)); }

// The last step from k to k + c, c <= k/2, k >= 64, at length k: as g f = 1 mod x^k, the cyclic
// product (g f_lo) mod (x^k - 1) is 1 + (g f_lo)[k, 2k); with f_hi = f[k, k + c) and g_c = g mod
// x^c, e = (f g)[k, k + c) = (g f_lo)[k, k + c) + (g_c f_hi) mod x^c, both from one inverse of
// T(f_lo) T(g) + T(f_hi) T(g_c); then g[k, k + c) = -(g_c e mod x^c). 6 transforms and 3 leaf
// products of length k, against 5 and 2 of length 2k. G, W: arrays of 2k + kGap words: T(g) at G,
// T(g_c) at G + k (G's padding is read, never written), T(f_lo) T(g) at W, T(f_hi) at W + k + 64
// (256 bytes from T(g_c)'s page offset, as W from G's); the inverse works in G, the last product in W.
inline void inverse_half_step(Span f, Out g, std::size_t k, std::size_t c, u32* G, u32* W) {
    const ntt::Transform& t = transform(std::countr_zero(k));
    u32 *const Gc = G + k, *const F = W + k + 64;
    const std::size_t lo = std::min(f.size(), k), hi = std::min(f.size(), k + c) - lo;
    const Out e = g.subspan(k, c);
    t.forward(Span(g.data(), k), G);
    t.forward_product(f.first(lo), G, W);
    t.forward(Span(g.data(), c), Gc);
    t.forward(f.subspan(lo, hi), F);
    const ntt::Term terms[] = {{W, nullptr}, {F, Gc}};
    t.inverse_product(terms, G, e);  // e[i] + [i = 0]
    e[0] = e[0] ? e[0] - 1 : kP - 1;
    t.cyclic_product(e, Gc, W, e, {.negate = true});
}

// g = 1 / f mod x^n, n = g.size() >= 1, f mod x^n given (f[0] != 0); g apart from f. G, W: arrays
// of bit_ceil(n) + kGap words. A step k -> 2k at length 2k: G = T(g mod x^k) (half zero), e =
// (f g)[k, k + c) by one cyclic product into g[k, k + c), then g[k, k + c) = -(e g mod x^c) by
// another, in place: one forward and two cyclic products (5 transforms, 2 leaf products) a step.
inline void inverse_newton(Span f, Out g, u32* G, u32* W) {
    const std::size_t n = g.size();
    inverse_direct(f, g.first(std::min(n, kInverseBase)));
    for (std::size_t k = kInverseBase; k < n; k *= 2) {
        const std::size_t c = std::min(k, n - k);
        if (2 * c <= k && k >= (std::size_t(1) << ntt::kMinLog)) return inverse_half_step(f, g, k, c, G, W);
        const ntt::Transform& t = transform(std::countr_zero(2 * k));
        const Out e = g.subspan(k, c);
        t.forward(Span(g.data(), k), G);
        t.cyclic_product(f.first(std::min(f.size(), k + c)), G, W, e, {.offset = k});
        t.cyclic_product(e, G, W, e, {.negate = true});
    }
}

// ---- log --------------------------------------------------------------------------------------

struct LogShape {
    std::size_t k, blocks;  // blocks of k coefficients of q = f' / f mod x^(n-1), k a power of two >= 32
    explicit LogShape(std::size_t n)
        : k(std::max<std::size_t>(32, std::bit_ceil((n - 1 + kLogBlocks - 1) / kLogBlocks))), blocks((n - 2) / k + 1) {}
};

inline std::size_t log_words(std::size_t n) {
    if (n <= kLogBase) return 0;
    const LogShape s(n);
    return footprint(n) + 2 * s.blocks * footprint(2 * s.k) + footprint(s.k);
}

// g = log f mod x^n, n = g.size() > kLogBase, f mod x^n given, f[0] = 1. f is read first.
inline void log_newton(Span f, Out g, Arena arena) {
    const std::size_t n = g.size();
    const LogShape shape(n);
    const std::size_t k = shape.k, len = 2 * k, blocks = shape.blocks;  // blocks >= 2 for n > kLogBase
    // The blocks read f' from f; a copy of f if f overlaps g (written block by block).
    const auto at = [](const u32* p) { return reinterpret_cast<std::uintptr_t>(p); };
    if (!f.empty() && at(f.data()) < at(g.data() + n) && at(g.data()) < at(f.data() + f.size())) {
        u32* const copy = arena.take(f.size());
        std::copy_n(f.data(), f.size(), copy);
        f = Span(copy, f.size());
    }
    u32* const H = arena.take(len);
    std::array<u32*, kLogBlocks> W{}, Q{};
    for (std::size_t t = 1; t < blocks; ++t) W[t] = arena.take(len);
    for (std::size_t t = 0; t + 1 < blocks; ++t) Q[t] = arena.take(len);
    u32* const R = arena.take(k);
    u32* const work = arena.take(len);

    const ntt::Transform& t = transform(std::countr_zero(len));
    inverse_newton(f.first(std::min(f.size(), k)), Out(R, k), W[1], Q[0]);  // h = 1 / f mod x^k
    t.forward(Span(R, k), H);
    for (std::size_t j = 1; j < blocks; ++j) {  // W_j = T(f[(j-1)k, (j+1)k))
        const std::size_t lo = std::min(f.size(), (j - 1) * k), hi = std::min(f.size(), (j + 1) * k);
        t.forward(f.subspan(lo, hi - lo), W[j]);
    }
    // Block j reads f[jk + 1, (j+1)k + 1) and writes g[jk + 1, (j+1)k + 1) (apart from f or f's copy).
    for (std::size_t j = 0; j < blocks; ++j) {
        const std::size_t first = j * k, count = std::min(k, n - 1 - first);
        const Out r(R, count);
        if (j == 0) {
            derivative_block(f, 0, count, nullptr, R);
            t.cyclic_product(r, H, work, r);  // q_0 = h f' mod x^k
        } else {  // R = f'[jk, jk + count) - sum_(i<j) (W_(j-i) q_i)[k, 2k), then q_j = h R mod x^k
            std::array<ntt::Term, kLogBlocks> terms{};
            for (std::size_t i = 0; i < j; ++i) terms[i] = {W[j - i], Q[i]};
            t.inverse_product(std::span<const ntt::Term>(terms.data(), j), work, r, {.offset = k});
            derivative_block(f, first, count, R, R);
            t.cyclic_product(r, H, work, r);
        }
        if (j + 1 < blocks) t.forward(Span(R, k), Q[j]);
        divide_by_index(first + 1, g.data() + first + 1, count, [&](std::size_t at) { return load_part(R, at, count); });
    }
    g[0] = 0;
}

// ---- exp --------------------------------------------------------------------------------------

inline std::size_t exp_words(std::size_t n) {
    if (n <= kExpBase) return 0;
    const std::size_t len = std::bit_ceil(n);
    return footprint(n) + 3 * footprint(len / 2) + 4 * footprint(len);
}

// The state of exp's Newton steps at precision m: g = exp f mod x^m in out, h = 1 / g mod x^(m/2),
// H = T_m(h), Gt = T_m(g mod x^(m/2)) (after a full step), q = x f' (n words); X, W (the
// products' work array), R, s: scratch.
struct ExpState {
    Out g;
    const u32* q;
    u32 *h, *Gt, *H, *X, *W, *R, *s;
};

// e.s[i] = (q[m + i] + t[i] - q[i]) / (m + i), i < c: (f - log g)[m, m + c). t apart from e.s.
inline void exp_residual(const ExpState& e, std::size_t m, const u32* t, std::size_t c) {
    divide_by_index(m, e.s, c, [&](std::size_t at) {
        return M::sub(M::add(load_part(e.q + m, at, c), load_part(t, at, c)), load_part(e.q, at, c));
    });
}

// g[m, 2m) (c = min(m, n - m) words of it), h to precision m, H = T_2m(h), Gt = T_2m(g mod x^m).
// 16 transforms of length m (8 of 2m) and 3.5 leaf products of 2m.
inline void exp_full_step(ExpState& e, std::size_t m, std::size_t c) {
    const int lg = std::countr_zero(m);
    const ntt::Transform &tm = transform(lg), &t2 = transform(lg + 1);
    const std::size_t half = m / 2;
    u32 *const Gt = e.Gt, *const H = e.H, *const X = e.X, *const W = e.W, *const R = e.R;
    const Out h_hi(e.h + half, half);
    // G_lo = T_m(g); e = (g h)[m/2, m) into h[m/2, m), then h[m/2, m) = -(e h mod x^(m/2))
    tm.forward(Span(e.g.data(), m), Gt);
    const ntt::Term gh[] = {{Gt, H}};
    tm.inverse_product(gh, W, h_hi, {.offset = half});
    tm.cyclic_product(h_hi, H, W, h_hi, {.negate = true});
    // T_m(r) = T_m(q mod x^m) G_lo at X[0, m); r = q g mod (x^m - 1) = x g' + (q g)[m, 2m) at R
    tm.forward_product(Span(e.q, m), Gt, X);
    const ntt::Term r[] = {{X, nullptr}};
    tm.inverse_product(r, W, Out(R, m));
    // H = T_2m(h); X = T_2m(r) = [T_m(r), its upper half]; t = h r mod x^m at R[0, c)
    t2.forward(Span(e.h, m), H);
    t2.forward_upper(Span(R, m), X + m);
    const ntt::Term hr[] = {{X, H}};
    t2.inverse_product(hr, W, Out(R, c));
    // s = (f - log g)[m, m + c); Gt = T_2m(g mod x^m); g[m, m + c) = s g mod x^c
    exp_residual(e, m, R, c);
    t2.forward_upper(Span(e.g.data(), m), Gt + m);
    t2.cyclic_product(Span(e.s, c), Gt, W, e.g.subspan(m, c));
}

// The last step, from m to m + c, c <= m, at length m only (lc-opt's exp_last_step), in blocks of
// B = m/2: with h0 = h mod x^B (H = T_m(h0)), g0 = g mod x^B (Gt = T_m(g0)) and G = g mod x^(kB),
// g_k = g[kB, kB + B) = g0 (f - log G)[kB, kB + B) mod x^B, k = 2, 3, where x (f - log G)' from kB
// on is ((q mod x^(kB)) G) h0 mod x^B + q_k (q_k = q[kB, kB + B)).
// - Block 2: r = q g mod (x^m - 1) = x G' + (q G)[m, 2m), t = h0 r mod x^B, s = (q[m + i] + t_i -
//   q_i) / (m + i) (x G' h0 = q mod x^B), g_2 = g0 s mod x^c.
// - Block 3 (c > B): (q G)[3B, 4B) for G = g mod x^(3B) is r[B, 2B) - x G'[B, 2B) plus the upper
//   half of (T(Q) T(g_2) + T(q_2) T(g mod x^m)) mod (x^m - 1), Q = q mod x^m; then s_3 = (t_3 +
//   q[3B + i]) / (3B + i) with t_3 = that times h0 mod x^B, and g_3 = g0 s_3.
// 14 transforms and 7 leaf products of length m for c > B (a full step: 16 and 7); 7 and 3 else.
// Arrays: G_lo = T_m(g mod x^m) at X, T(Q) at X + m + 64 (X has 2m + kGap words; 64 words apart
// against 4K aliasing), T(g_2) at Gt + m + 64, T(q_2) at H + m + 64; t and t_3 in h (no longer read).
inline void exp_last_step(ExpState& e, std::size_t m, std::size_t c, bool first) {
    const ntt::Transform& tm = transform(std::countr_zero(m));
    const std::size_t half = m / 2, c2 = std::min(c, half);
    u32 *const Gt = e.Gt, *const H = e.H, *const X = e.X, *const W = e.W, *const R = e.R, *const t = e.h;
    u32 *const TQ = X + m + 64, *const G2 = Gt + m + 64, *const Q2 = H + m + 64;
    if (first) tm.forward(Span(e.g.data(), half), Gt);  // no full step before: Gt = T_m(g0)
    tm.forward(Span(e.g.data(), m), X);
    if (c > half) {  // r at R[0, m), T(Q) kept
        tm.forward(Span(e.q, m), TQ);
        const ntt::Term qg[] = {{TQ, X}};
        tm.inverse_product(qg, W, Out(R, m));
    } else {  // r mod x^B at R
        tm.cyclic_product(Span(e.q, m), X, W, Out(R, half));
    }
    // Block 2: t = h0 (r mod x^B) mod x^c2, s, g_2 = g0 s mod x^c2
    tm.cyclic_product(Span(R, half), H, W, Out(t, c2));
    exp_residual(e, m, t, c2);
    tm.cyclic_product(Span(e.s, c2), Gt, W, e.g.subspan(m, c2));
    if (c <= half) return;
    // Block 3: S = (q mod x^(3B) g mod x^(3B))[3B, 4B) at t[0, B)
    const std::size_t c3 = c - half;
    tm.forward(Span(e.g.data() + m, half), G2);
    tm.forward(Span(e.q + m, half), Q2);
    const ntt::Term sum[] = {{TQ, G2}, {Q2, X}};
    tm.inverse_product(sum, W, Out(t, half), {.offset = half});
    std::size_t i = 0;  // t += r[B, 2B) - (B + i) g[B + i]
    const u32* g1 = e.g.data() + half;
    if (half >= kLanes) {
        alignas(64) u32 start[kLanes];
        for (std::size_t l = 0; l < kLanes; ++l) start[l] = mul(u32(half + l), kR);
        V index = S::load(start);
        const V step = S::broadcast(mul(u32(kLanes), kR));
        for (; i + kLanes <= half; i += kLanes, index = M::add(index, step)) {
            const V xg = M::template reduce<kP>(M::montgomery(S::load(g1 + i), index));
            S::store(t + i, M::sub(M::add(S::load(t + i), S::load(R + half + i)), xg));
        }
    }
    for (; i < half; ++i) t[i] = u32((u64(t[i]) + R[half + i] + kP - mul(u32(half + i), g1[i])) % kP);
    // t_3 = S h0 mod x^c3, s_3 = (t_3 + q[3B + i]) / (3B + i), g_3 = g0 s_3 mod x^c3
    tm.cyclic_product(Span(t, c3), H, W, Out(t, c3));
    divide_by_index(m + half, e.s, c3, [&](std::size_t at) {
        return M::add(load_part(t, at, c3), load_part(e.q + m + half, at, c3));
    });
    tm.cyclic_product(Span(e.s, c3), Gt, W, e.g.subspan(m + half, c3));
}

// g = exp f mod x^n, n = g.size() > kExpBase, f mod x^n given, f[0] = 0. f is read first.
inline void exp_newton(Span f, Out g, Arena arena) {
    const std::size_t n = g.size(), len = std::bit_ceil(n);
    u32* const q = arena.take(n);
    const std::size_t fl = std::min(f.size(), n);
    multiply_by_index(f.data(), 0, q, fl);  // q = x f'
    zero(q + fl, n - fl);
    // f is not read from here on.
    ExpState e{g, q, arena.take(len / 2), arena.take(len), arena.take(len), arena.take(len), arena.take(len),
               arena.take(len / 2), arena.take(len / 2)};
    exp_direct(q, g.first(kExpBase));
    inverse_direct(Span(g.data(), kExpBase / 2), Out(e.h, kExpBase / 2));
    transform(std::countr_zero(kExpBase)).forward(Span(e.h, kExpBase / 2), e.H);
    for (std::size_t m = kExpBase; m < n; m *= 2) {
        const std::size_t c = std::min(m, n - m);
        if (n <= 2 * m) exp_last_step(e, m, c, m == kExpBase);
        else exp_full_step(e, m, c);
    }
}

}  // namespace detail

// Words of scratch the forms with a scratch span take for n coefficients.
inline std::size_t inverse_scratch(std::size_t n) { return detail::kAlignSlack + detail::inverse_words(n); }
inline std::size_t log_scratch(std::size_t n) { return detail::kAlignSlack + detail::log_words(n); }
inline std::size_t exp_scratch(std::size_t n) { return detail::kAlignSlack + detail::exp_words(n); }

namespace detail {

inline void check_length(std::size_t n, const char* what) {
    if (n > kMaxSize) throw std::length_error(what);
}

// Runs body(arena) in the caller's scratch, or in this thread's region when scratch is empty.
template <class Body>
void with_memory(std::size_t words, std::span<u32> scratch, const Body& body) {
    if (words == 0) return body(Arena(nullptr, 0));
    if (!scratch.empty()) {
        if (scratch.size() < words + kAlignSlack) throw std::invalid_argument("fps::series: scratch too small");
        return body(Arena(scratch.data(), scratch.size()));
    }
    RegionGuard guard{ntt::detail::take_region((words + kAlignSlack) * sizeof(u32))};
    body(Arena(reinterpret_cast<u32*>(guard.region.start), guard.region.bytes / sizeof(u32)));
}

}  // namespace detail

// out = 1 / f mod x^n, n = out.size(); f[0] != 0; out apart from f.
inline void inverse(std::span<const u32> f, std::span<u32> out, std::span<u32> scratch = {}) {
    const std::size_t n = out.size();
    if (n == 0) return;
    detail::check_length(n, "fps::series::inverse: n > 2^25");
    f = f.first(std::min(f.size(), n));
    if (n <= detail::kInverseBase) return detail::inverse_direct(f, out);
    const std::size_t len = std::bit_ceil(n);
    detail::with_memory(detail::inverse_words(n), scratch, [&](detail::Arena a) {
        u32* G = a.take(len);
        u32* W = a.take(len);
        detail::inverse_newton(f, out, G, W);
    });
}

// out = log f mod x^n, n = out.size(); f[0] = 1, n <= P; out may overlap f.
inline void log(std::span<const u32> f, std::span<u32> out, std::span<u32> scratch = {}) {
    const std::size_t n = out.size();
    if (n == 0) return;
    detail::check_length(n, "fps::series::log: n > 2^25");
    f = f.first(std::min(f.size(), n));
    if (n <= detail::kLogBase) return detail::log_direct(f, out);
    detail::with_memory(detail::log_words(n), scratch, [&](detail::Arena a) { detail::log_newton(f, out, a); });
}

// out = exp f mod x^n, n = out.size(); f[0] = 0, n <= P; out may overlap f.
inline void exp(std::span<const u32> f, std::span<u32> out, std::span<u32> scratch = {}) {
    const std::size_t n = out.size();
    if (n == 0) return;
    detail::check_length(n, "fps::series::exp: n > 2^25");
    f = f.first(std::min(f.size(), n));
    if (n <= detail::kExpBase) {
        u32 q[detail::kExpBase] = {};
        detail::multiply_by_index(f.data(), 0, q, f.size());
        return detail::exp_direct(q, out);
    }
    detail::with_memory(detail::exp_words(n), scratch, [&](detail::Arena a) { detail::exp_newton(f, out, a); });
}

}  // namespace fps::series
