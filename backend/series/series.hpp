// Power series modulo P = 998244353: inverse, log, exp, pow and sqrt into caller memory (backend
// kernel). From lc-opt's lib/poly/{inverse,log,exp,pow,sqrt}.hpp, rewritten on ntt::Transform's
// leaf form, so every target runs it. Design and measurements: notes.md.
//
//   fps::series::inverse(f, out);    // out = 1 / f mod x^n, n = out.size(); f[0] != 0; out apart from f
//   fps::series::log(f, out);        // out = log f mod x^n; f[0] = 1, n <= P; out may overlap f
//   fps::series::exp(f, out);        // out = exp f mod x^n; f[0] = 0, n <= P; out may overlap f
//   fps::series::pow(f, e, c, out);  // out = c (f / f[0])^e mod x^n; f[0] != 0, e, c < P, n <= P;
//                                    // out may overlap f. f^M for an integer M >= 0 (n <= P):
//                                    // pow(f, M mod P, f[0]^M, out)
//   fps::series::sqrt(f, c, out);    // out = g with g^2 = f mod x^n, g[0] = c; c^2 = f[0] != 0;
//                                    // out may overlap f
//
//   // The same in the caller's scratch, of inverse_scratch(n) (log_scratch, exp_scratch,
//   // pow_scratch, sqrt_scratch) words, any alignment, instead of this thread's memory (a shorter
//   // one: std::invalid_argument):
//   fps::series::inverse(f, out, scratch);
//
// f: words < P, of which f mod x^n is read (a shorter f has zeros past its end). The preconditions
// on f[0], c and n are not checked (the frontend checks them); n past kMaxSize throws
// std::length_error before out is written. If out overlaps f, log and sqrt copy f first, and exp
// and pow read all of f before they write out.
//
// Newton iteration on ntt::Transform's fused ops (ntt.hpp: sources and outputs in caller memory, a
// cyclic product or an inverse of leaf products in one depth-first pass, half outputs), in units
// of T(n) (a transform of n words) and LP(n) (leaf products of n words), as lc-opt's:
// - inverse: k -> 2k with transforms of 2k: G = T(g mod x^k) (a half-zero source), e = (f g)[k,
//   k + c) by a cyclic product (upper half out), then g[k, k + c) = -(e g mod x^c) by another (in
//   place, negated): 5 transforms and 2 leaf products of 2k a step, 10 T(n) + 4 LP(n). A last step
//   to n <= 3k/2 runs at length k: 6 T(k) + 3 LP(k). Below kInverseBase coefficients, the recurrence.
// - log f = integral of f' / f mod x^(n-1) in B <= 4 blocks of k coefficients (lc-opt's blocked
//   division, divide()): h = 1 / f mod x^k, then q_j = h (f' - f Q)[jk, (j+1)k) mod x^k, Q = q mod
//   x^(jk), (f Q)[jk, (j+1)k) = sum over i < j of (W_(j-i) q_i)[k, 2k), W_t = f[(t-1)k, (t+1)k),
//   as one inverse of a sum of leaf products of kept transforms of length 2k; for 4 blocks, blocks
//   2 and 3 share lc-opt's Toeplitz product (ntt::Transform::toeplitz_product: 4 leaf products, not
//   5). About 11.5 T(n) + 5.5 LP(n). h and log's q_j are transformed in place.
// - exp: m -> 2m keeping h = 1 / g mod x^(m/2) and H = T_m(h), q = x f' (lc-opt's step, notes.md);
//   8 transforms and 3.5 leaf products of 2m a step. The last step, to n <= 2m, runs at length m
//   in two blocks of m/2 (lc-opt's exp_last_step): 14 transforms and 7 leaf products of m.
// - pow: g = c (f / f[0])^e solves x g' = q g, q = e x f' / f (lc-opt's power): q by the division
//   of N = e x f' by f (over n coefficients, so its blocks are q's and its kept transforms give T(q
//   mod x^m) and T(q[m, 3m/2)) of exp's last step), then exp's steps from q with g[0] = c. No
//   integral, derivative or passes for f / f[0], e and c: about log + exp - 1 T(n).
// - sqrt: m -> 2m keeping h = -1 / g mod x^(m/2) and H = T_m(h) (lc-opt's step): 11 transforms and
//   5 leaf products of m a step; the last step, to n <= 2m, at length m (Karp and Markstein's
//   split): 8 transforms and 4 leaf products of m. About 9.5 T(n) + 4.5 LP(n).
// - Divisions by indices (log's integral, exp's s): Montgomery's batch inversion (lc-opt's
//   divide_by_index), divisors below 2^13 from a table.
// Ragged n (issue #214), n = L + r, L = bit_floor(n): about the time of n coefficients, not of
// 2L (ns per coefficient at 2^20 + 1, 1.25 2^20 and 6e5 within 0.91-1.16 of 2^21's on Zen 3,
// Emerald Rapids, Zen 5 and Neoverse V2, against 1.12-1.70 before; 1.07-1.15 at 1.75 2^20:
// notes/214-r1.md):
// - inverse, r <= 9L/32: Newton steps whose lengths halve down from n, so they land on it; a step
//   to n' = L' + r' runs at length L' with two products of r' words (the corners) besides
//   (inverse_ragged_step).
// - exp, sqrt, the division's last block (log, pow): a last step or block of c coefficients runs
//   its products of c words as low products (about the cost of 2c words, NTT products of ragged
//   length) to 9m/32, its windows by sums of products (window()) to kDirectWindow, and the step
//   before a light one as a last step too; the division takes 5 blocks of k/2 where 3 of k would
//   leave a short one.
// Against lc-opt in the same run (Zen 3, n = 2^14 .. 2^21): notes/112-r1.md (inverse, log, exp),
// notes/107-r1.md (pow, sqrt).
//
// Memory: this thread's region of the NTT (ntt::detail::take_region, kept across calls), or the
// caller's scratch: with N = bit_ceil(n), 2 N words for inverse (the chain: 3 L words and its
// corners'), about n + 2 B (n / 2 + 80) for log (B <= 5 blocks), n + 5.5 N for exp, n + 7 N for
// pow and n + 2.25 N for sqrt. The transforms' tables: one ntt::Transform per length, kept per
// thread, 2^lg bytes each.
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

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"
#include "backend/small/small.hpp"

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
inline constexpr std::size_t kLogBlocks = 4;  // blocks of the division, 5 at some ragged counts (DivisionShape)
inline constexpr std::size_t kMaxBlocks = 5;

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

// out[c] = a (from + c) f[from + c] mod P, minus s[c] if s is not null, for c < count; f is zero
// past its end, a < P. out may be s. The derivative's coefficients first .. first + count - 1:
// from = first + 1, a = 1 (log); those of e x f': from = first, a = e (pow).
inline void index_block(Span f, std::size_t from, u32 a, std::size_t count, const u32* s, u32* out) {
    const std::size_t avail = f.size() > from ? std::min(count, f.size() - from) : 0;
    const u32* src = avail ? f.data() + from : f.data();
    std::size_t c = 0;
    const u32 ar = a == 1 ? kR : mul(a, kR);  // a 2^32: Montgomery products by (i a 2^32) give i a x
    if (count >= kLanes) {
        alignas(64) u32 start[kLanes];
        for (std::size_t l = 0; l < kLanes; ++l) start[l] = mul(u32((from + l) % kP), ar);
        V index = S::load(start);
        const V step = S::broadcast(mul(u32(kLanes), ar));
        for (; c + kLanes <= count; c += kLanes, index = M::add(index, step)) {
            const V d = M::template reduce<kP>(M::montgomery(load_part(src, c, avail), index));
            S::store(out + c, s ? M::sub(d, S::load(s + c)) : d);
        }
    }
    for (; c < count; ++c) {
        u32 d = c < avail ? mul(u32((from + c) % kP), src[c]) : 0;
        if (a != 1) d = mul(d, a);
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
inline void exp_direct(const u32* q, Out g, u32 g0 = 1) {
    g[0] = g0;
    for (std::size_t i = 1; i < g.size(); ++i) g[i] = mul(dot(q, g.data(), i, 1, i + 1), kSmallInverses[i]);
}

// g = c (f / f[0])^e mod x^n, n = g.size() <= kExpBase, f[0] != 0. f g' = e f' g gives, with a = f
// / f[0] and b_j = (e + 1) j a_j: g_i = (sum_(0<j<=i) b_j g_(i-j)) / i - sum_(0<j<=i) a_j g_(i-j).
// f is read before g is written.
inline void pow_direct(Span f, u32 e, u32 c, Out g) {
    const std::size_t n = g.size();
    u32 a[kExpBase] = {}, b[kExpBase] = {};
    std::copy_n(f.begin(), std::min(f.size(), n), a);
    const u32 inverse0 = power(a[0], kP - 2), e1 = e + 1 == kP ? 0 : e + 1;
    for (std::size_t j = 1; j < n; ++j) a[j] = mul(a[j], inverse0), b[j] = mul(mul(e1, u32(j)), a[j]);
    g[0] = c;
    for (std::size_t i = 1; i < n; ++i) {
        const u32 x = mul(dot(b, g.data(), i, 1, i + 1), kSmallInverses[i]), y = dot(a, g.data(), i, 1, i + 1);
        g[i] = x >= y ? x - y : x + kP - y;
    }
}

// g = sqrt f mod x^n with g[0] = c, c^2 = f[0] != 0, n = g.size() <= kSqrtBase: 2 c g_i = f_i -
// sum_(0<j<i) g_j g_(i-j). f is read before g is written.
inline constexpr std::size_t kSqrtBase = 64;
inline void sqrt_direct(Span f, u32 c, Out g) {
    const std::size_t n = g.size();
    u32 a[kSqrtBase] = {};
    std::copy_n(f.begin(), std::min(f.size(), n), a);
    const u32 inverse = power(mul(2, c), kP - 2);
    g[0] = c;
    for (std::size_t i = 1; i < n; ++i) {
        const u32 s = dot(g.data(), g.data(), i, 1, i);
        g[i] = mul(a[i] >= s ? a[i] - s : a[i] + kP - s, inverse);
    }
}

// ---- Short products (the corners of steps at ragged lengths, issue #214) ----------------------

// out[i] = (a b)[pos + i] for i < count, a of na and b of nb words: sums of Montgomery products of
// a and of b read backwards (a lane reversal), 4 positions a pass over the words (so each pass
// reads a and b once), lazy in [0, 2P). For a last step's few coefficients (kDirectWindow): about
// 0.1 ns a product, against a transform of the step's length.
inline constexpr std::size_t kDirectWindow = 16;
inline void window(const u32* a, std::size_t na, const u32* b, std::size_t nb, std::size_t pos, std::size_t count,
                   u32* out) {
    constexpr std::size_t kGroup = 4;
    const auto reversal = [] {
        std::array<int, kLanes> lanes{};
        for (std::size_t l = 0; l < kLanes; ++l) lanes[l] = int(kLanes - 1 - l);
        return S::index(lanes);
    }();
    const auto lo_of = [nb](std::size_t p) { return p + 1 > nb ? p + 1 - nb : 0; };
    const auto hi_of = [na](std::size_t p) { return std::min(na, p + 1); };
    for (std::size_t g = 0; g < count; g += kGroup) {
        const std::size_t G = std::min(kGroup, count - g), first = pos + g;
        const std::size_t lo = lo_of(first + G - 1), hi = hi_of(first);  // the words every position reads
        const std::size_t end = lo < hi ? lo + (hi - lo) / kLanes * kLanes : lo;
        V acc[kGroup];
        for (std::size_t t = 0; t < kGroup; ++t) acc[t] = S::broadcast(0);
        for (std::size_t j = lo; j < end; j += kLanes) {
            const V x = S::load(a + j);
            for (std::size_t t = 0; t < G; ++t) {
                const V y = S::permute(S::load(b + (first + t - j - (kLanes - 1))), reversal);
                acc[t] = M::template reduce<2 * kP>(S::add(acc[t], M::montgomery(x, y)));
            }
        }
        for (std::size_t t = 0; t < G; ++t) {
            alignas(64) u32 lanes[kLanes];
            S::store(lanes, acc[t]);
            u64 vector_sum = 0;
            for (u32 v : lanes) vector_sum += v;
            const std::size_t p = first + t;
            u64 sum = 0;  // the words outside [lo, end), at most 2 kLanes + 6 of them
            for (std::size_t j = lo_of(p); j < std::min(lo, hi_of(p)); ++j) sum = (sum + u64(a[j]) * b[p - j]) % kP;
            for (std::size_t j = std::max(end, lo_of(p)); j < hi_of(p); ++j) sum = (sum + u64(a[j]) * b[p - j]) % kP;
            out[g + t] = u32((u64(mul(u32(vector_sum % kP), kR)) + sum) % kP);
        }
    }
}

// out = (a b) mod x^c, a and b mod x^c given (shorter spans have zeros past them), out apart from
// both: sums of products to kDirectProduct coefficients, small::multiply where it beats the NTT,
// else the NTT's product of ragged length (about the cost of its 2c - 1 words, backend/ntt/
// ragged.hpp): a plain plan cut to c words into out, a split one (ntt::detail::ragged_into) into
// work, then copied. work: low_product_words(c) words, 64-byte aligned (the thread's region is the
// caller's: ntt::multiply would map another).
inline constexpr std::size_t kDirectProduct = 32;

// Words of work low_product needs: for N = 2c - 1 words, a plain plan's two arrays of 2^lg <=
// bit_ceil(N) words and 4 KiB to place them, or the product and ragged::split_words of the main
// length (below N).
inline std::size_t low_product_words(std::size_t c) {
    if (c <= kDirectProduct) return 0;
    const std::size_t N = 2 * c - 1;
    const std::size_t plain = 2 * (std::bit_ceil(N) + ntt::kPadding) + 1024;
    const std::size_t split = footprint(N) + ntt::detail::ragged::split_words(std::bit_floor(N - 1));
    return std::max(plain, split) + 16;
}

inline void low_product(Span a, Span b, std::size_t c, u32* out, u32* work) {
    const std::size_t na = std::min(a.size(), c), nb = std::min(b.size(), c);
    if (na == 0 || nb == 0) return zero(out, c);
    if (c <= kDirectProduct) {
        for (std::size_t i = 0; i < c; ++i) {
            const std::size_t lo = i + 1 > nb ? i + 1 - nb : 0, hi = std::min(i + 1, na);
            out[i] = lo < hi ? dot(a.data(), b.data(), i, lo, hi) : 0;
        }
        return;
    }
    const std::size_t N = na + nb - 1, count = std::min(N, c);
    namespace nd = ntt::detail;
    if (small::beats_ntt(na, nb)) {
        small::multiply<kP>(a.first(na), b.first(nb), std::span<u32>(work, N));
        std::copy_n(work, count, out);
    } else {
        using Variant = ntt::SelectedMultiply;
        const nd::ragged::Plan plan = nd::ragged::choose(N);
        const auto [roots, inverse_roots] = nd::tables<Variant>(plan.lg);
        if (!plan.split) {
            const std::size_t span = (std::size_t(1) << plan.lg) + ntt::kPadding;
            const std::size_t at = nd::place(a.data(), b.data(), out, span * sizeof(u32), nd::place_period(plan.lg));
            u32* const A = work + at / sizeof(u32);
            const nd::Product product{a.data(), na, b.data(), nb, A, A + span, out, count, plan.lg};
            nd::run_product<Variant, false>(product, roots, inverse_roots);
        } else {
            u32* const rest = work + footprint(N);
            nd::ragged_into<Variant, false>(a.data(), na, b.data(), nb, work, rest, plan, roots, inverse_roots);
            std::copy_n(work, count, out);
        }
    }
    zero(out + count, c - count);
}

// Whether a last step's product of c coefficients mod x^c runs as low_product (two factors of c
// words, about the cost of 2c) rather than as a cyclic product of length m with a kept transform:
// c <= 9m/32 (notes/214-r1.md), and its work fits in `words`.
inline bool short_product(std::size_t c, std::size_t m, std::size_t words) {
    return 32 * c <= 9 * m && low_product_words(c) <= words;
}

// dst[i] = a[i] + b[i] (kSub: a[i] - b[i]) mod P for i < n, canonical words; dst may be a or b.
template <bool kSub>
void add_words(u32* dst, const u32* a, const u32* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        const V x = S::load(a + i), y = S::load(b + i);
        S::store(dst + i, kSub ? M::sub(x, y) : M::add(x, y));
    }
    for (; i < n; ++i) dst[i] = kSub ? (a[i] >= b[i] ? a[i] - b[i] : a[i] + kP - b[i]) : u32((u64(a[i]) + b[i]) % kP);
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

// The ragged chain (issue #214): Newton steps whose lengths halve down from n, ceil(n / 2^j), so
// they land on n; each step from k = ceil(n' / 2) to n' runs at L = bit_floor(n' - 1) < n' when
// n' = L + r with r <= L/2 (inverse_ragged_step), else at bit_ceil(n') (inverse_plain_step). Arrays:
// G, W, E of the longest length, C (r words) and work (low_product_words(r)) for the corners.
struct ChainArrays {
    u32 *G, *W, *E, *C, *work;
};

// Whether inverse runs the chain at n: n = L + r, L = bit_floor(n) >= 64, 0 < r <= 9L/32 (beyond
// it, the corners cost more than the steps to L and inverse_half_step: notes/214-r1.md).
inline bool use_chain(std::size_t n) {
    const std::size_t L = std::bit_floor(n);
    return L >= 64 && n > L && 32 * (n - L) <= 9 * L;
}

inline std::size_t chain_words(std::size_t n) {
    const std::size_t L = std::bit_floor(n - 1), r = n - L;
    return 3 * footprint(std::max<std::size_t>(64, L)) + footprint(r) + footprint(low_product_words(r));
}

// The step from k to n' at length 2^lg >= n' (2^lg >= 64, k <= 2^lg / 2): G = T(g mod x^k) (half
// zero), e = (f g)[k, n') from one cyclic product (the upper half when k = 2^(lg-1), else the first
// n' words into E), then g[k, n') = -(e g mod x^(n'-k)) by another, in place.
inline void inverse_plain_step(Span f, Out g, std::size_t k, std::size_t np, u32* G, u32* W, u32* E) {
    const std::size_t len = std::max<std::size_t>(64, std::bit_ceil(np)), c = np - k;
    const ntt::Transform& t = transform(std::countr_zero(len));
    const Out e = g.subspan(k, c);
    const Span fs = f.first(std::min(f.size(), np));
    t.forward(Span(g.data(), k), G);
    if (2 * k == len) {
        t.cyclic_product(fs, G, W, e, {.offset = k});
    } else {
        t.cyclic_product(fs, G, W, Out(E, np));
        std::copy_n(E + k, c, e.data());
    }
    t.cyclic_product(e, G, W, e, {.negate = true});
}

// The step from k = ceil(n' / 2) to n' = L + r, 1 <= r <= L/2, at length L (L/2 < k < L, c = n' -
// k). With f_lo = f mod x^L, f_hi = f[L, n') and g_r = g mod x^r:
// - e = (f g)[k, n'): f_lo g has L + k - 1 words, so (f_lo g) mod (x^L - 1) is exact at [k, L) and
//   at [0, r) holds [i = 0] + (f_lo g)[L + i] (g f = 1 mod x^k); (f g)[L + i] adds (f_hi g_r)[i].
// - g[k, n') = -(g e mod x^c): with e1 = e[0, L - k) and e2 = e[L - k, c) (r words), g e1 has L - 1
//   words (a cyclic product of length L, exact), and x^(L - k) g e2 adds (g_r e2) mod x^r.
// The power-of-two step's 5 transforms and 2 leaf products of length L, on n' instead of L words,
// and two short products of r words (the corners, low_product).
// Both corners share g_r: where the NTT's plan for their 2r - 1 words is a plain product of length
// 2^lg (r past kDirectProduct), they run as cyclic products of length 2^lg with one kept T(g_r) (in
// work), a transform fewer than two low_products.
inline void inverse_ragged_step(Span f, Out g, std::size_t k, std::size_t np, const ChainArrays& a) {
    const std::size_t L = std::bit_floor(np - 1), r = np - L, c = np - k;
    const ntt::Transform& t = transform(std::countr_zero(L));
    u32* const x = g.data();
    const Span fhi = f.size() > L ? f.subspan(L, std::min(f.size(), np) - L) : Span();
    const ntt::detail::ragged::Plan plan = ntt::detail::ragged::choose(2 * r - 1);
    const bool kept = r > kDirectProduct && !plan.split && !small::beats_ntt(r, r);
    const ntt::Transform* tc = kept ? &transform(plan.lg) : nullptr;
    if (kept) {
        tc->forward(Span(x, r), a.work);
        tc->cyclic_product(fhi, a.work, a.W, Out(a.C, r));
    } else {
        low_product(fhi, Span(x, r), r, a.C, a.work);
    }
    t.forward(Span(x, k), a.G);
    t.cyclic_product(f.first(std::min(f.size(), L)), a.G, a.W, Out(a.E, L));
    std::copy(a.E + k, a.E + L, x + k);
    a.E[0] = a.E[0] ? a.E[0] - 1 : kP - 1;
    add_words<false>(x + L, a.E, a.C, r);  // e2
    if (kept) tc->cyclic_product(Span(x + L, r), a.work, a.W, Out(a.C, r));
    else low_product(Span(x, r), Span(x + L, r), r, a.C, a.work);
    t.cyclic_product(Span(x + k, L - k), a.G, a.W, g.subspan(k, c), {.negate = true});
    add_words<true>(x + L, x + L, a.C, r);
}

// g = 1 / f mod x^n by the ragged chain, n = g.size() > 64, f mod x^n given (f[0] != 0).
inline void inverse_chain(Span f, Out g, const ChainArrays& a) {
    std::array<std::size_t, 64> lengths;
    int steps = 0;
    for (std::size_t x = g.size(); x > kInverseBase; x = (x + 1) / 2) lengths[steps++] = x;
    std::size_t k = (lengths[steps - 1] + 1) / 2;
    inverse_direct(f, g.first(k));
    for (int j = steps - 1; j >= 0; --j) {
        const std::size_t np = lengths[j], L = std::bit_floor(np - 1);
        if (L >= 64 && 2 * (np - L) <= L) inverse_ragged_step(f, g, k, np, a);
        else inverse_plain_step(f, g, k, np, a.G, a.W, a.E);
        k = np;
    }
}

// ---- The blocked division (log, pow) ----------------------------------------------------------

// q = N / f mod x^count in blocks of k coefficients, k a power of two >= 32: blocks = 2 .. 4 for
// count >= 64 (lc-opt's), or 5 blocks of half that k where they cover count (issue #214: count in
// (4k, 5k], e.g. 2^j + 1 or 1.25 2^j, where 3 blocks of 2k would leave a short one).
struct DivisionShape {
    std::size_t k, blocks;
    explicit DivisionShape(std::size_t count) {
        k = std::max<std::size_t>(32, std::bit_ceil((count + kLogBlocks - 1) / kLogBlocks));
        blocks = (count - 1) / k + 1;
        if (blocks == 3 && k >= 64 && count <= 5 * (k / 2)) k /= 2, blocks = 5;
    }
};

// The division's arrays: H = T(h), W_1 .. W_(blocks-1), the blocks' transforms Q_0 .. Q_(blocks-2)
// (2k words each), R (k words: log's last block), work (2k), hc (h mod x^c for a last block of c
// words that is not full, below). W_1 is followed by another array of 2k words (W_2 or Q_0).
struct DivisionArrays {
    u32* H;
    std::array<u32*, kMaxBlocks> W, Q;
    u32 *R, *work, *hc;
};

// The last block of a division, c = count - (blocks - 1) k coefficients (issue #214): direct to
// kDirectWindow (its residual by window() from q's coefficients and f: no transforms for it, so
// no W_(B-1), T(q_(B-2)) or Toeplitz product), low where short_product holds at 2k (q_j = h_c times
// the residual by low_product, in W_1 and the array after it; h_c copied before T(h)), else full.
enum class LastBlock { full, low, direct };
inline LastBlock last_block(const DivisionShape& s, std::size_t count) {
    const std::size_t c = count - (s.blocks - 1) * s.k;
    if (c <= kDirectWindow) return LastBlock::direct;
    return short_product(c, 2 * s.k, 2 * footprint(2 * s.k)) ? LastBlock::low : LastBlock::full;
}

// Words of the division's arrays but Q's.
inline std::size_t division_words(const DivisionShape& s) {
    return (s.blocks + 1) * footprint(2 * s.k) + footprint(s.k);
}

// lc-opt's blocked division (its log.hpp): q = N / f mod x^count, f[0] != 0, f mod x^count given,
// in blocks q_j of k coefficients from h = 1 / f mod x^k, with transforms of length 2k:
//   q_j = h (N - f Q)[jk, (j+1)k) mod x^k,  Q = q mod x^(jk),
//   (f Q)[jk, (j+1)k) = sum over i < j of (W_(j-i) q_i)[k, 2k),  W_t = f[(t-1)k, (t+1)k),
// the sum as one inverse of leaf products of kept transforms (its upper half is exact).
// numerator(first, c, s, out): out[i] = N[first + i] - s[i] (s null: N alone), i < c, out may
// be s. place(j): where block j's residual and q_j go (k words; a.Q[j]: its transform then runs in
// place). done(j, first, c): q_j is at place(j); after it returns, its transform goes to a.Q[j] if
// j + 1 < blocks. prefix(first): q mod x^first, for a direct last block. h and its transform in H
// (in place). For 4 blocks, lc-opt's Toeplitz sharing: block 2's residual and W_3 = W_3 Q_0 + W_2
// Q_1 by one toeplitz_product (3 leaf products), block 3's from W_3 and W_1 Q_2. (3 + 5B)
// transforms and 2 + B + B(B - 1)/2 leaf products of 2k for B blocks, one fewer for B = 4; a short
// last block as last_block() says.
template <class Numerator, class Place, class Done, class Prefix>
void divide(Span f, std::size_t count, const DivisionShape& shape, const DivisionArrays& a, const Numerator& numerator,
            const Place& place, const Done& done, const Prefix& prefix) {
    const std::size_t k = shape.k, len = 2 * k, blocks = shape.blocks, c_last = count - (blocks - 1) * k;
    const LastBlock last = last_block(shape, count);
    const bool direct = last == LastBlock::direct, toeplitz = blocks - direct == 4;  // 4 blocks by transforms
    const ntt::Transform& t = transform(std::countr_zero(len));
    inverse_newton(f.first(std::min(f.size(), k)), Out(a.H, k), a.W[1], a.Q[0]);  // h = 1 / f mod x^k
    if (last != LastBlock::full) std::copy_n(a.H, c_last, a.hc);
    t.forward(Span(a.H, k), a.H);
    for (std::size_t j = 1; j < (direct ? blocks - 1 : blocks); ++j) {  // W_j = T(f[(j-1)k, (j+1)k))
        const std::size_t lo = std::min(f.size(), (j - 1) * k), hi = std::min(f.size(), (j + 1) * k);
        t.forward(f.subspan(lo, hi - lo), a.W[j]);
    }
    for (std::size_t j = 0; j < blocks; ++j) {
        const std::size_t first = j * k, c = std::min(k, count - first);
        const bool is_last = j + 1 == blocks;
        u32* const p = place(j);
        // The residual at p, or for a last block that is not full at H (T(h) is not read for it)
        u32* const res = is_last && last != LastBlock::full ? a.H : p;
        const Out r(res, c);
        if (j == 0) {
            numerator(0, c, nullptr, res);
        } else {  // N[jk, jk + c) - sum_(i<j) (W_(j-i) q_i)[k, 2k)
            if (is_last && direct) {  // (f Q)[jk + i] from the coefficients
                window(prefix(first), first, f.data(), f.size(), first, c, res);
            } else if (toeplitz && j >= 2) {  // lc-opt's Toeplitz sharing: W_3 = W_3 Q_0 + W_2 Q_1 in block 2
                const ntt::Term terms[] = {{a.W[1], a.Q[2]}, {a.W[3], nullptr}};
                if (j == 2) t.toeplitz_product(a.W[1], a.W[2], a.W[3], a.Q[0], a.Q[1], a.work, r, {.offset = k});
                else t.inverse_product(terms, a.work, r, {.offset = k});
            } else {
                std::array<ntt::Term, kMaxBlocks - 1> terms{};
                for (std::size_t i = 0; i < j; ++i) terms[i] = {a.W[j - i], a.Q[i]};
                t.inverse_product(std::span<const ntt::Term>(terms.data(), j), a.work, r, {.offset = k});
            }
            numerator(first, c, res, res);
        }
        if (res != p) low_product(Span(a.hc, c), Span(res, c), c, p, a.W[1]);  // q_j = h_c (residual) mod x^c
        else t.cyclic_product(r, a.H, a.work, r);                              // q_j = h (residual) mod x^k
        done(j, first, c);
        if (j + 1 < blocks && !(direct && j + 2 == blocks)) t.forward(Span(p, k), a.Q[j]);
    }
}

// ---- log --------------------------------------------------------------------------------------

inline std::size_t log_words(std::size_t n) {
    if (n <= kLogBase) return 0;
    const DivisionShape s(n - 1);
    const std::size_t hc = last_block(s, n - 1) == LastBlock::full ? 0 : footprint(n - 1 - (s.blocks - 1) * s.k);
    return footprint(n) + (s.blocks - 1) * footprint(2 * s.k) + division_words(s) + hc;
}

// g = log f mod x^n, n = g.size() > kLogBase, f mod x^n given, f[0] = 1: the integral of q = f' /
// f mod x^(n-1) from divide(). f is read first.
inline void log_newton(Span f, Out g, Arena arena) {
    const std::size_t n = g.size();
    const DivisionShape shape(n - 1);
    const std::size_t k = shape.k, len = 2 * k, blocks = shape.blocks;  // blocks >= 2 for n > kLogBase
    // The blocks read f' from f; a copy of f if f overlaps g (written block by block).
    const auto at = [](const u32* p) { return reinterpret_cast<std::uintptr_t>(p); };
    if (!f.empty() && at(f.data()) < at(g.data() + n) && at(g.data()) < at(f.data() + f.size())) {
        u32* const copy = arena.take(f.size());
        std::copy_n(f.data(), f.size(), copy);
        f = Span(copy, f.size());
    }
    DivisionArrays a{};
    a.H = arena.take(len);
    for (std::size_t t = 1; t < blocks; ++t) a.W[t] = arena.take(len);
    for (std::size_t t = 0; t + 1 < blocks; ++t) a.Q[t] = arena.take(len);
    a.R = arena.take(k);
    a.work = arena.take(len);
    if (last_block(shape, n - 1) != LastBlock::full) a.hc = arena.take(n - 1 - (blocks - 1) * k);
    // Block j reads f[jk + 1, (j+1)k + 1) and writes g[jk + 1, (j+1)k + 1) (apart from f or f's copy).
    // q mod x^first for a direct last block: i g_i from g, in W_1 and the arrays after it.
    divide(
        f, n - 1, shape, a,
        [f](std::size_t first, std::size_t c, const u32* s, u32* out) { index_block(f, first + 1, 1, c, s, out); },
        [&a, blocks](std::size_t j) { return j + 1 < blocks ? a.Q[j] : a.R; },
        [&](std::size_t j, std::size_t first, std::size_t c) {
            const u32* q = j + 1 < blocks ? a.Q[j] : a.R;
            divide_by_index(first + 1, g.data() + first + 1, c, [&](std::size_t i) { return load_part(q, i, c); });
        },
        [&](std::size_t first) {
            multiply_by_index(g.data() + 1, 1, a.W[1], first);
            return static_cast<const u32*>(a.W[1]);
        });
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

// Transforms of length m of the last step (n <= 2m) that a caller already has (pow: from the
// division): low = T_m(q mod x^m), block2 = T_m(q[m, 3m/2)) or null. exp has none
// (NoQTransforms): its steps compile as they did before pow.
struct QTransforms {
    const u32 *low, *block2;
};
struct NoQTransforms {};

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
// 14 transforms and 7 leaf products of length m for c > B (a full step: 16 and 7); 7 and 3 else;
// 2 and 1 transforms fewer with given's (pow). Arrays: G_lo = T_m(g mod x^m) at X, T(Q) at X + m
// + 64 (X has 2m + kGap words; 64 words apart against 4K aliasing), T(g_2) at Gt + m + 64, T(q_2)
// at H + m + 64; t and t_3 in h (no longer read).
// A block of few coefficients (issue #214): to kDirectWindow, its window of q G by sums of products
// (window(): no transforms for that block); to 9m/32 (short_product), the products by h0 and g0 mod
// x^c by low_product, its work in X and W, t and S at h + B (h0 is read). So a step past m costs about
// its c coefficients, not a block of m/2. keep_h: t and S at h + B too, for a light step after this
// one (exp_steps).
template <class Given>
void exp_last_step(ExpState& e, std::size_t m, std::size_t c, bool first, const Given& given, bool keep_h = false) {
    const ntt::Transform& tm = transform(std::countr_zero(m));
    const std::size_t half = m / 2, c2 = std::min(c, half), c3 = c > half ? c - half : 0;
    u32 *const Gt = e.Gt, *const H = e.H, *const X = e.X, *const W = e.W, *const R = e.R;
    u32* const g = e.g.data();
    u32* const G2 = Gt + m + 64;
    const std::size_t work_words = 2 * footprint(2 * m);  // X and W, one after the other
    const bool direct3 = c3 && c3 <= kDirectWindow, direct2 = !c3 && c2 <= kDirectWindow;
    const bool short2 = short_product(c2, m, work_words), short3 = c3 && short_product(c3, m, work_words);
    const u32 *TQ = nullptr, *Q2 = nullptr;
    if constexpr (std::is_same_v<Given, QTransforms>) TQ = given.low, Q2 = given.block2;
    if (first && (!short2 || (c3 && !short3))) tm.forward(Span(g, half), Gt);  // no full step before: Gt = T_m(g0)
    // r at R[0, m) for block 3 by transforms, else r mod x^c2
    if (direct2) {
        window(e.q, m, g, m, m, c2, R);  // (q G)[m, m + c2), plus x G'
        for (std::size_t i = 0; i < c2; ++i) R[i] = u32((u64(R[i]) + mul(u32(i), g[i])) % kP);
    } else {
        tm.forward(Span(g, m), X);
        if ((c3 && !direct3) || TQ) {  // T(Q) kept
            if (!TQ) {
                tm.forward(Span(e.q, m), X + m + 64);
                TQ = X + m + 64;
            }
            const ntt::Term qg[] = {{TQ, X}};
            tm.inverse_product(qg, W, Out(R, c3 && !direct3 ? m : c2));
        } else {
            tm.cyclic_product(Span(e.q, m), X, W, Out(R, c2));
        }
    }
    // Block 2: t = h0 (r mod x^c2) mod x^c2, s, g_2 = g0 s mod x^c2
    u32* const t = short2 || short3 || keep_h ? e.h + half : e.h;  // h0 is read by the low products
    if (short2) low_product(Span(e.h, c2), Span(R, c2), c2, t, X);
    else tm.cyclic_product(Span(R, c2), H, W, Out(t, c2));
    exp_residual(e, m, t, c2);
    if (short2) low_product(Span(g, c2), Span(e.s, c2), c2, g + m, X);
    else tm.cyclic_product(Span(e.s, c2), Gt, W, e.g.subspan(m, c2));
    if (!c3) return;
    // Block 3: S = (q mod x^(3B) g mod x^(3B))[3B, 3B + c3) at S
    u32* const S3 = short3 || keep_h ? e.h + half : e.h;
    if (direct3) {
        window(e.q, m + half, g, m + half, m + half, c3, S3);
    } else {
        tm.forward(Span(g + m, half), G2);
        if (!Q2) {
            tm.forward(Span(e.q + m, half), H + m + 64);
            Q2 = H + m + 64;
        }
        const ntt::Term sum[] = {{TQ, G2}, {Q2, X}};
        tm.inverse_product(sum, W, Out(S3, c3), {.offset = half});
        std::size_t i = 0;  // S += r[B, 2B) - (B + i) g[B + i]
        const u32* g1 = g + half;
        if (c3 >= kLanes) {
            alignas(64) u32 start[kLanes];
            for (std::size_t l = 0; l < kLanes; ++l) start[l] = mul(u32(half + l), kR);
            V index = S::load(start);
            const V step = S::broadcast(mul(u32(kLanes), kR));
            for (; i + kLanes <= c3; i += kLanes, index = M::add(index, step)) {
                const V xg = M::template reduce<kP>(M::montgomery(S::load(g1 + i), index));
                S::store(S3 + i, M::sub(M::add(S::load(S3 + i), S::load(R + half + i)), xg));
            }
        }
        for (; i < c3; ++i) S3[i] = u32((u64(S3[i]) + R[half + i] + kP - mul(u32(half + i), g1[i])) % kP);
    }
    // t_3 = S h0 mod x^c3, s_3 = (t_3 + q[3B + i]) / (3B + i), g_3 = g0 s_3 mod x^c3
    u32* const t3 = short3 ? R : S3;
    if (short3) low_product(Span(e.h, c3), Span(S3, c3), c3, t3, X);
    else tm.cyclic_product(Span(S3, c3), H, W, Out(t3, c3));
    divide_by_index(m + half, e.s, c3, [&](std::size_t at) {
        return M::add(load_part(t3, at, c3), load_part(e.q + m + half, at, c3));
    });
    if (short3) low_product(Span(g, c3), Span(e.s, c3), c3, g + m + half, X);
    else tm.cyclic_product(Span(e.s, c3), Gt, W, e.g.subspan(m + half, c3));
}

// Whether the last step at m, of c <= m/4 coefficients, reads neither H nor Gt (window() and
// low_product only, with h mod x^c): then the step before it runs as a last step too (exp_last_step
// to 2m keeps h mod x^(m/2) and costs 14 transforms against a full step's 16), issue #214.
inline bool exp_light(std::size_t m, std::size_t c) {
    return 4 * c <= m && (c <= kDirectWindow || short_product(c, m, 2 * footprint(2 * m)));
}

// Words of exp_steps' arrays for n coefficients.
inline std::size_t exp_step_words(std::size_t n) {
    const std::size_t len = std::bit_ceil(n);
    return 3 * footprint(len / 2) + 4 * footprint(len);
}

// g mod x^n, n = g.size() > kExpBase, for x g' = q g with g[0] = g0 != 0 (q: n words, q[0] = 0):
// exp's Newton steps, which do not depend on the scale of g. given: for the last step.
template <class Given>
void exp_steps(const u32* q, u32 g0, Out g, Arena arena, const Given& given) {
    const std::size_t n = g.size(), len = std::bit_ceil(n);
    ExpState e{g, q, arena.take(len / 2), arena.take(len), arena.take(len), arena.take(len), arena.take(len),
               arena.take(len / 2), arena.take(len / 2)};
    exp_direct(q, g.first(kExpBase), g0);
    inverse_direct(Span(g.data(), kExpBase / 2), Out(e.h, kExpBase / 2));
    transform(std::countr_zero(kExpBase)).forward(Span(e.h, kExpBase / 2), e.H);
    for (std::size_t m = kExpBase; m < n; m *= 2) {
        const std::size_t c = std::min(m, n - m);
        if (n <= 2 * m) exp_last_step(e, m, c, m == kExpBase, given);
        else if (n <= 4 * m && exp_light(2 * m, n - 2 * m))
            exp_last_step(e, m, m, m == kExpBase, NoQTransforms{}, true);
        else exp_full_step(e, m, c);
    }
}

// g = exp f mod x^n, n = g.size() > kExpBase, f mod x^n given, f[0] = 0. f is read first.
inline void exp_newton(Span f, Out g, Arena arena) {
    const std::size_t n = g.size();
    u32* const q = arena.take(n);
    const std::size_t fl = std::min(f.size(), n);
    multiply_by_index(f.data(), 0, q, fl);  // q = x f'
    zero(q + fl, n - fl);
    // f is not read from here on.
    exp_steps(q, 1, g, arena, NoQTransforms{});
}

// ---- pow --------------------------------------------------------------------------------------

// g = c (f / f[0])^e solves x g' = q g with q = e x f' / f and g[0] = c. lc-opt's power: q from
// log's blocked division, then exp's steps, which skips log's integral, exp's derivative and the
// passes for f / f[0], e and c. Here the division runs on N = e x f' (not f'), over n
// coefficients, so its blocks are q's own (no shift, no product by e: lc-opt's sink pass and
// times_x) and its kept transforms give the last exp step's: with k = m/2, m = bit_ceil(n) / 2
// (the division's k for every n > kExpBase), T_m(q mod x^m) = T(q_0) + s T(q_1), s = x^k = 1 on
// the lower half of the leaves and -1 on the upper half, and T_m(q[m, 3m/2)) = T(q_2) (when exp's
// block 3 runs, the division has 4 blocks).

inline std::size_t pow_words(std::size_t n) {
    if (n <= kExpBase) return 0;
    const DivisionShape s(n);
    return footprint(n) + (s.blocks - 1) * footprint(2 * s.k) + std::max(division_words(s), exp_step_words(n));
}

// a[i] = a[i] + b[i] on the lower half of len words (leaves), a[i] - b[i] on the upper half.
inline void add_lower_sub_upper(u32* a, const u32* b, std::size_t len) {
    const std::size_t half = len / 2;
    for (std::size_t i = 0; i < half; i += kLanes) S::store(a + i, M::add(S::load(a + i), S::load(b + i)));
    for (std::size_t i = half; i < len; i += kLanes) S::store(a + i, M::sub(S::load(a + i), S::load(b + i)));
}

// g = c (f / f[0])^e mod x^n, n = g.size() > kExpBase, f mod x^n given, f[0] != 0. f is read first.
inline void pow_newton(Span f, u32 e, u32 c, Out g, Arena arena) {
    const std::size_t n = g.size();
    const DivisionShape shape(n);
    const std::size_t k = shape.k, len = 2 * k, blocks = shape.blocks;  // 3 or 4 blocks with 2k = bit_ceil(n) / 2, or 5
    u32* const q = arena.take(n);
    DivisionArrays a{};
    for (std::size_t t = 0; t + 1 < blocks; ++t) a.Q[t] = arena.take(len);
    Arena rest = arena;  // the division's other arrays, then exp's
    a.H = rest.take(len);
    for (std::size_t t = 1; t < blocks; ++t) a.W[t] = rest.take(len);
    a.R = rest.take(k);
    a.work = rest.take(len);
    a.hc = a.R;  // R is not a place here
    divide(
        f, n, shape, a,
        [f, e](std::size_t first, std::size_t count, const u32* s, u32* out) { index_block(f, first, e, count, s, out); },
        [q, k](std::size_t j) { return q + j * k; }, [](std::size_t, std::size_t, std::size_t) {},
        [q](std::size_t) { return static_cast<const u32*>(q); });
    // f is not read from here on. The kept transforms are exp's last step's when 2k = m (not for 5
    // blocks); a direct last block has no transform of the block before it: T(q_1) for 3 blocks
    // (exp's last step is then direct too, c <= kDirectWindow), T(q_2) for 4.
    const bool direct = last_block(shape, n) == LastBlock::direct, given = 2 * len == std::bit_ceil(n);
    const bool low = given && !(direct && blocks == 3), block2 = given && blocks == 4 && !direct;
    if (low) add_lower_sub_upper(a.Q[0], a.Q[1], len);  // T(q mod x^2k)
    exp_steps(q, c, g, arena, QTransforms{low ? a.Q[0] : nullptr, block2 ? a.Q[2] : nullptr});
}

// ---- sqrt -------------------------------------------------------------------------------------

// out[i] = (s[i] - f[first + i] - f[m + first + i]) / 2 (plus add[i] if add is not null), i < c:
// for s = g^2 mod (x^m - 1) and g^2 = f mod x^m, s - f mod x^m = (g^2)[m, 2m), so out = (g^2 -
// f)[m + first, m + first + c) / 2. f is zero past its end. out may be s or add.
template <bool kAdd>
void half_residual(Span f, std::size_t m, std::size_t first, const u32* s, const u32* add, u32* out, std::size_t c) {
    // x / 2 for x < P: x >> 1, plus (P + 1) / 2 if x is odd (< P).
    const V half = S::broadcast((kP + 1) / 2);
    const auto step = [&](V lo, V hi, V si, V ai) {
        const V x = M::sub(M::sub(si, lo), hi), y = S::template shr<1>(x);
        const V r = S::add(y, S::mul_lo(S::sub(x, S::add(y, y)), half));
        return kAdd ? M::add(r, ai) : r;
    };
    std::size_t i = 0;
    if (m + first + c <= f.size()) {  // f read in place
        const u32 *lo = f.data() + first, *hi = f.data() + m + first;
        for (; i + kLanes <= c; i += kLanes)
            S::store(out + i, step(S::load(lo + i), S::load(hi + i), S::load(s + i), kAdd ? S::load(add + i) : V{}));
    }
    for (; i < c; i += kLanes) {
        const V x = step(load_part(f.data(), first + i, f.size()), load_part(f.data(), m + first + i, f.size()),
                         load_part(s, i, c), kAdd ? load_part(add, i, c) : V{});
        if (i + kLanes <= c) S::store(out + i, x);
        else S::store_first(out + i, x, int(c - i));
    }
}

inline std::size_t sqrt_words(std::size_t n) {
    if (n <= kSqrtBase) return 0;
    const std::size_t len = std::bit_ceil(n) / 2;
    return footprint(n) + 4 * footprint(len) + footprint(len / 2);
}

// sqrt's last step, from m to m + rest, rest <= m, at length m (see sqrt_newton), with h = -1 / g
// mod x^(m/2) and H = T_m(h); h is kept. A few coefficients (issue #214) as exp's: (g^2)[m, m +
// rest) by window() to kDirectWindow, the products by h mod x^rest as low_product to 9m/32.
inline void sqrt_last_step(Span f, Out g, std::size_t m, u32* G, u32* H, u32* R, u32* h, u32* W) {
    const std::size_t n = g.size();  // m + rest
    const ntt::Transform& tm = transform(std::countr_zero(m));
    const std::size_t half = m / 2, rest = n - m, work_words = 2 * footprint(m);  // G and H, one after the other
    const ntt::Term gg[] = {{G, G}};
    if (rest <= half) {
        if (rest <= kDirectWindow) {  // s = f mod x^m + (g^2)[m, m + rest)
            window(g.data(), m, g.data(), m, m, rest, R);
            for (std::size_t i = 0; i < rest; ++i) R[i] = u32((u64(R[i]) + (i < f.size() ? f[i] : 0)) % kP);
        } else {
            tm.forward(Span(g.data(), m), G);
            tm.inverse_product(gg, W, Out(R, rest));
        }
        half_residual<false>(f, m, 0, R, nullptr, R, rest);
        if (short_product(rest, m, work_words)) low_product(Span(h, rest), Span(R, rest), rest, g.data() + m, G);
        else tm.cyclic_product(Span(R, rest), H, W, g.subspan(m, rest));
        return;
    }
    const std::size_t c1 = rest - half;
    const bool short1 = short_product(c1, m, work_words);
    tm.forward(Span(g.data(), m), G);
    tm.inverse_product(gg, W, Out(R, rest));
    half_residual<false>(f, m, 0, R, nullptr, R, half);
    tm.cyclic_product(Span(R, half), H, W, g.subspan(m, half));  // d0
    // r1 = r[m/2, rest) + (g d0)[m/2, m/2 + c1) at R + m/2, (g d0) at A = g[m + m/2, rest) (d1 goes
    // there; h is kept for a light step after this one)
    u32* const A = g.data() + m + half;
    if (c1 <= kDirectWindow) window(g.data(), m, g.data() + m, half, half, c1, A);
    else tm.cyclic_product(Span(g.data() + m, half), G, W, Out(A, c1), {.offset = half});
    half_residual<true>(f, m, half, R + half, A, R + half, c1);
    if (short1) low_product(Span(h, c1), Span(R + half, c1), c1, g.data() + m + half, G);
    else tm.cyclic_product(Span(R + half, c1), H, W, g.subspan(m + half, c1));  // d1
}

// Whether sqrt's last step at m, of rest <= m/4 coefficients, reads neither H nor h past x^rest:
// then the step before it runs as a last step too (8 transforms against a full step's 11).
inline bool sqrt_light(std::size_t m, std::size_t rest) {
    return 4 * rest <= m && (rest <= kDirectWindow || short_product(rest, m, 2 * footprint(m)));
}

// g = sqrt f mod x^n with g[0] = c, c^2 = f[0] != 0, n = g.size() > kSqrtBase, f mod x^n given
// (lc-opt's sqrt). Newton m -> 2m keeping h = -1 / g mod x^(m/2) and H = T_m(h) (the sign saves
// negations):
//   G = T_m(g); e = (g h)[m/2, m) (an inverse_product); h[m/2, m) = e h mod x^(m/2) (cyclic);
//   r = (g^2 - f)[m, 2m) / 2 from s = g^2 mod (x^m - 1) (an inverse_product of G G) and f;
//   H = T_2m(h) (also the next step's); g[m, 2m) = h r mod x^m (cyclic, length 2m).
// 11 transforms and 5 leaf products of m a step. The last step, to n <= 2m, at length m: d = h r
// mod x^(n - m) for n - m <= m/2 (4 transforms); else Karp and Markstein's split, d0 = h r0 mod
// x^(m/2) (r0 = r mod x^(m/2)), d1 = h (r + g d0)[m/2, n - m) mod x^(n - 3m/2), (g d0)[m/2, m) the
// upper half of a cyclic product: 8 transforms and 4 leaf products. f is copied first if it
// overlaps g. Arrays G, H, W, R of bit_ceil(n) / 2 words, h of half that.
inline void sqrt_newton(Span f, u32 c, Out g, Arena arena) {
    const std::size_t n = g.size(), len = std::bit_ceil(n) / 2;
    const auto at = [](const u32* p) { return reinterpret_cast<std::uintptr_t>(p); };
    if (!f.empty() && at(f.data()) < at(g.data() + n) && at(g.data()) < at(f.data() + f.size())) {
        u32* const copy = arena.take(f.size());
        std::copy_n(f.data(), f.size(), copy);
        f = Span(copy, f.size());
    }
    u32 *const G = arena.take(len), *const H = arena.take(len), *const R = arena.take(len), *const h = arena.take(len / 2);
    u32* const W = arena.take(len);
    sqrt_direct(f, c, g.first(kSqrtBase));
    std::size_t m = kSqrtBase;
    inverse_direct(Span(g.data(), m / 2), Out(h, m / 2));  // h = -1 / g mod x^(m/2)
    for (u32& x : Out(h, m / 2)) x = x ? kP - x : 0;
    transform(std::countr_zero(m)).forward(Span(h, m / 2), H);
    for (; 2 * m < n; m *= 2) {
        if (n <= 4 * m && sqrt_light(2 * m, n - 2 * m)) {  // to 2m as a last step, h kept
            sqrt_last_step(f, g.first(2 * m), m, G, H, R, h, W);
            continue;
        }
        const int lg = std::countr_zero(m);
        const ntt::Transform &tm = transform(lg), &t2 = transform(lg + 1);
        const std::size_t half = m / 2;
        const Out h_hi(h + half, half);
        tm.forward(Span(g.data(), m), G);
        const ntt::Term gh[] = {{G, H}}, gg[] = {{G, G}};
        tm.inverse_product(gh, W, h_hi, {.offset = half});
        tm.cyclic_product(h_hi, H, W, h_hi);
        tm.inverse_product(gg, W, Out(R, m));
        half_residual<false>(f, m, 0, R, nullptr, R, m);
        t2.forward(Span(h, m), H);
        t2.cyclic_product(Span(R, m), H, W, g.subspan(m, m));
    }
    sqrt_last_step(f, g, m, G, H, R, h, W);
}

}  // namespace detail

// Words of scratch the forms with a scratch span take for n coefficients.
inline std::size_t inverse_scratch(std::size_t n) {
    const bool chain = n > detail::kInverseBase && detail::use_chain(n);
    return detail::kAlignSlack + (chain ? detail::chain_words(n) : detail::inverse_words(n));
}
inline std::size_t log_scratch(std::size_t n) { return detail::kAlignSlack + detail::log_words(n); }
inline std::size_t exp_scratch(std::size_t n) { return detail::kAlignSlack + detail::exp_words(n); }
inline std::size_t pow_scratch(std::size_t n) { return detail::kAlignSlack + detail::pow_words(n); }
inline std::size_t sqrt_scratch(std::size_t n) { return detail::kAlignSlack + detail::sqrt_words(n); }

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
    if (detail::use_chain(n)) {
        const std::size_t L = std::max<std::size_t>(64, std::bit_floor(n - 1)), r = n - std::bit_floor(n - 1);
        return detail::with_memory(detail::chain_words(n), scratch, [&](detail::Arena a) {
            const detail::ChainArrays arrays{a.take(L), a.take(L), a.take(L), a.take(r),
                                             a.take(detail::low_product_words(r))};
            detail::inverse_chain(f, out, arrays);
        });
    }
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

// out = c (f / f[0])^e mod x^n, n = out.size(); f[0] != 0, e < P, c < P, n <= P; out may overlap f.
inline void pow(std::span<const u32> f, u32 e, u32 c, std::span<u32> out, std::span<u32> scratch = {}) {
    const std::size_t n = out.size();
    if (n == 0) return;
    detail::check_length(n, "fps::series::pow: n > 2^25");
    f = f.first(std::min(f.size(), n));
    if (n <= detail::kExpBase) return detail::pow_direct(f, e, c, out);
    detail::with_memory(detail::pow_words(n), scratch, [&](detail::Arena a) { detail::pow_newton(f, e, c, out, a); });
}

// out = the square root g of f mod x^n with g[0] = c, n = out.size(); c^2 = f[0] != 0, c < P; out
// may overlap f.
inline void sqrt(std::span<const u32> f, u32 c, std::span<u32> out, std::span<u32> scratch = {}) {
    const std::size_t n = out.size();
    if (n == 0) return;
    detail::check_length(n, "fps::series::sqrt: n > 2^25");
    f = f.first(std::min(f.size(), n));
    if (n <= detail::kSqrtBase) return detail::sqrt_direct(f, c, out);
    detail::with_memory(detail::sqrt_words(n), scratch, [&](detail::Arena a) { detail::sqrt_newton(f, c, out, a); });
}

}  // namespace fps::series
