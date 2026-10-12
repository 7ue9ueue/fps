// Power series modulo P = 998244353: inverse, log and exp into caller memory (backend kernel).
// From lc-opt's lib/poly/{inverse,log,exp}.hpp, rewritten on ntt::Transform's leaf form, so every
// target runs it. Design and measurements: notes.md.
//
//   fps::series::inverse(f, out);    // out = 1 / f mod x^n, n = out.size(); f[0] != 0; out apart from f
//   fps::series::log(f, out);        // out = log f mod x^n; f[0] = 1, n <= P; out may overlap f
//   fps::series::exp(f, out);        // out = exp f mod x^n; f[0] = 0, n <= P; out may overlap f
//
//   // The same in the caller's scratch, of inverse_scratch(n) (log_scratch, exp_scratch) words,
//   // any alignment, instead of this thread's memory:
//   fps::series::inverse(f, out, scratch);
//
// f: words < P, of which f mod x^n is read (a shorter f has zeros past its end). The preconditions
// on f[0] and n are not checked (the frontend checks them); n past kMaxSize throws
// std::length_error before out is written. log and exp read all of f before they write out.
//
// Newton iteration on transforms of the leaf form (ntt.hpp), in units of T(n) (a transform of n
// words) and LP(n) (leaf products of n words); a product n x n -> n is about 6 T(n) + 2 LP(n):
// - inverse: k -> 2k with transforms of 2k: G = T(g mod x^k), E = T(-f mod x^2k), e = E G
//   inverse (its words [k, 2k) are -(f g)[k, 2k)), then g[k, 2k) from T(x^k e[k, 2k)) G.
//   5 transforms and 2 leaf products of 2k a step: 10 T(n) + 4 LP(n). Below kInverseBase
//   coefficients, the recurrence.
// - log f = integral of f' / f mod x^(n-1) in B <= 4 blocks of k coefficients (lc-opt's blocked
//   division): h = 1 / f mod x^k, then q_j = h (f' - f Q)[jk, (j+1)k) mod x^k, Q = q mod x^(jk),
//   (f Q)[jk, (j+1)k) = sum over i < j of (W_(j-i) q_i)[k, 2k), W_t = f[(t-1)k, (t+1)k), as one
//   inverse of leaf products of kept transforms of length 2k. About 11.5 T(n) + 6 LP(n).
// - exp: m -> 2m keeping h = 1 / g mod x^(m/2) and H = T_m(h), q = x f' (lc-opt's step, notes.md);
//   8 transforms and 3.5 leaf products of 2m a step. The upper half of T_2m(a), a of m words, is
//   T_m of a twisted copy (detail::twist). A last step to n <= 3m/2 runs at length m: 7 T(m).
//
// Memory: this thread's region of the NTT (ntt::detail::take_region, kept across calls); about
// 2 n words for inverse, 6.5 n for log and 8 n for exp. The transforms' tables: one ntt::Transform
// per length, kept per thread, 2^lg bytes each.
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
#include <vector>

#include "backend/factorials/factorials.hpp"
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
constexpr u32 kR3 = mul(mul(kR, kR), kR);     // 2^96 mod P
constexpr u32 kRinv = power(kR, kP - 2);      // 2^-32 mod P

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

// dst = -src mod P; dst may be src.
inline void negate(const u32* src, u32* dst, std::size_t n) {
    std::size_t i = 0;
    for (const V z = S::broadcast(0); i + kLanes <= n; i += kLanes) S::store(dst + i, M::sub(z, S::load(src + i)));
    for (; i < n; ++i) dst[i] = src[i] ? kP - src[i] : 0;
}

// dst[i] = a[i] b[i] 2^-32 mod P: b in Montgomery form (times 2^32). dst may be a or b.
inline void multiply_montgomery(const u32* a, const u32* b, u32* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes)
        S::store(dst + i, M::template reduce<kP>(M::montgomery(S::load(a + i), S::load(b + i))));
    for (; i < n; ++i) dst[i] = mul(mul(a[i], b[i]), kRinv);
}

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

// r[i] = 2^32 / i mod P (Montgomery form of 1 / i) for 0 < i < n = r.size() <= P, r[0] = 0; tmp: n
// words. From the factorial tables: 1 / i = (i - 1)! / i!.
inline void reciprocals(Out r, u32* tmp) {
    const std::size_t n = r.size();
    if (n == 0) return;
    factorials::tables<kP>(r, Out(tmp, n));
    // r[i] = r[i - 1] tmp[i] 2^32, from the top down: a block reads r below where it writes.
    std::size_t i = n;
    for (const V c = S::broadcast(kR3); i >= kLanes + 1; i -= kLanes) {
        const std::size_t at = i - kLanes;
        const V x = M::montgomery(S::load(r.data() + at - 1), S::load(tmp + at));
        S::store(r.data() + at, M::template reduce<kP>(M::montgomery(x, c)));
    }
    for (; i-- > 1;) r[i] = mul(mul(r[i - 1], tmp[i]), kR);
    r[0] = 0;
}

// The twists of the upper half of a transform of 16 g words: tw[g + j] = rho^j 2^32 mod P for
// j < g, rho = r(g / 2) of order 2g (g a power of two, 8 <= g <= the largest yet). This thread's.
// The leaf form puts leaf g + u of a transform of 16 g words at w = rho w_u (w_v = (-1)^v r(v / 2),
// r multiplicative over disjoint bits), and word t of a leaf is C_t(w) for a = sum_t x^t C_t(x^8);
// so for a of 8 g words the upper half of T_16g(a) is T_8g(a') with a'[8j + t] = a[8j + t] rho^j.
inline const u32* twists(std::size_t g) {
    thread_local std::vector<u32> t;
    if (t.size() < 2 * g) {
        t.assign(2 * g, 0);
        const u32 rho = ntt::detail::kRoots[0][std::countr_zero(g) - 1];
        t[g] = kR;
        for (std::size_t j = 1; j < g; ++j) t[g + j] = mul(t[g + j - 1], rho);
        for (std::size_t h = g / 2; h >= 1; h /= 2)  // rho_h = rho_2h^2
            for (std::size_t j = 0; j < h; ++j) t[h + j] = t[2 * h + 2 * j];
    }
    return t.data();
}

// dst = the twisted copy of src (twists above), 8 g words, g >= 8; dst apart from src.
template <class T = S>
void twist(const u32* src, u32* dst, std::size_t g) {
    using Mt = modarith::Ops<T, kP>;
    const u32* tw = twists(g) + g;
    if constexpr (T::lanes == 16) {
        for (std::size_t j = 0; j < g; j += 2) {
            const auto w = T::broadcast_halves(tw + j, tw + j + 1);
            T::store(dst + 8 * j, Mt::template reduce<kP>(Mt::montgomery(T::load(src + 8 * j), w)));
        }
    } else {
        static_assert(8 % T::lanes == 0);
        for (std::size_t j = 0; j < g; ++j) {
            const auto w = T::broadcast(tw[j]);
            for (std::size_t v = 0; v < 8; v += T::lanes)
                T::store(dst + 8 * j + v, Mt::template reduce<kP>(Mt::montgomery(T::load(src + 8 * j + v), w)));
        }
    }
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
// product (g f_lo) mod (x^k - 1) is 1 + (g f_lo)[k, 2k), f_lo = f mod x^k; with f_hi = f[k, k + c)
// and g_c = g mod x^c, e = (f g)[k, k + c) = (g f_lo)[k, k + c) + (g_c f_hi) mod x^c, both read
// from one inverse; then g[k, k + c) = -(g_c e mod x^c). 7 transforms and 3 leaf products of
// length k, against 5 and 2 of length 2k. G, E as for inverse_newton (2k + kGap words each): G_c
// at G + k (G's padding is read, never written), f_hi's transform at E + k + 64 (256 bytes from
// G_c's page offset, as E from G's).
inline void inverse_half_step(Span f, Out g, std::size_t k, std::size_t c, u32* G, u32* E) {
    const ntt::Transform& t = transform(std::countr_zero(k));
    u32 *const Gc = G + k, *const F = E + k + 64;
    const std::size_t lo = std::min(f.size(), k), hi = std::min(f.size(), k + c) - lo;
    std::copy_n(g.data(), k, G);
    t.forward(G);
    std::copy_n(g.data(), c, Gc);
    zero(Gc + c, k - c + ntt::kPadding);
    t.forward(Gc);
    negate(f.data(), E, lo);
    zero(E + lo, k - lo);
    t.forward(E);
    t.multiply(E, G, E);
    negate(f.data() + lo, F, hi);
    zero(F + hi, k - hi + ntt::kPadding);
    t.forward(F);
    t.multiply_add(E, F, Gc);
    t.inverse(E);  // E[i] = -e[i] - [i = 0], i < c
    E[0] = E[0] == kP - 1 ? 0 : E[0] + 1;
    zero(E + c, k - c);
    t.forward(E);
    t.multiply(E, Gc, E);
    t.inverse(E);
    std::copy_n(E, c, g.data() + k);
}

// g = 1 / f mod x^n, n = g.size() >= 1, f mod x^n given (f[0] != 0); g apart from f. G, E: arrays
// of bit_ceil(n) + kGap words.
inline void inverse_newton(Span f, Out g, u32* G, u32* E) {
    const std::size_t n = g.size();
    inverse_direct(f, g.first(std::min(n, kInverseBase)));
    for (std::size_t k = kInverseBase; k < n; k *= 2) {
        const std::size_t len = 2 * k, c = std::min(k, n - k), fl = std::min(f.size(), k + c);
        if (2 * c <= k && k >= (std::size_t(1) << ntt::kMinLog)) return inverse_half_step(f, g, k, c, G, E);
        const ntt::Transform& t = transform(std::countr_zero(len));
        std::copy_n(g.data(), k, G);
        zero(G + k, k + ntt::kPadding);
        t.forward(G);
        negate(f.data(), E, fl);
        zero(E + fl, len - fl + ntt::kPadding);
        t.forward(E);
        t.multiply(E, G, E);
        t.inverse(E);  // E[k, k + c) = -(f g)[k, k + c)
        zero(E, k);
        t.forward(E);
        t.multiply(E, G, E);
        t.inverse(E);
        std::copy_n(E + k, c, g.data() + k);
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
    return 2 * footprint(n) + 2 * s.blocks * footprint(2 * s.k) + 2 * footprint(s.k);
}

// g = log f mod x^n, n = g.size() > kLogBase, f mod x^n given, f[0] = 1. f is read first.
inline void log_newton(Span f, Out g, Arena arena) {
    const std::size_t n = g.size();
    const LogShape shape(n);
    const std::size_t k = shape.k, len = 2 * k, blocks = shape.blocks;
    u32* const recip = arena.take(n);
    u32* const d = arena.take(n);
    reciprocals(Out(recip, n), d);
    // d = f' mod x^(n-1)
    const std::size_t fd = f.size() > 1 ? std::min(f.size() - 1, n - 1) : 0;
    if (fd) multiply_by_index(f.data() + 1, 1, d, fd);
    zero(d + fd, n - 1 - fd);
    u32* const H = arena.take(len);
    std::array<u32*, kLogBlocks> W{}, Q{};
    for (std::size_t t = 1; t < blocks; ++t) W[t] = arena.take(len);
    u32* const R = arena.take(len);
    for (std::size_t t = 0; t + 1 < blocks; ++t) Q[t] = arena.take(len);
    u32* const G2 = arena.take(k);
    u32* const E2 = arena.take(k);

    const ntt::Transform& t = transform(std::countr_zero(len));
    inverse_newton(f.first(std::min(f.size(), k)), Out(H, k), G2, E2);  // h = 1 / f mod x^k
    zero(H + k, k + ntt::kPadding);
    t.forward(H);
    for (std::size_t j = 1; j < blocks; ++j) {  // W_j = T(f[(j-1)k, (j+1)k))
        const std::size_t lo = std::min(f.size(), (j - 1) * k), hi = std::min(f.size(), (j + 1) * k);
        std::copy(f.data() + lo, f.data() + hi, W[j]);
        zero(W[j] + (hi - lo), len - (hi - lo) + ntt::kPadding);
        t.forward(W[j]);
    }
    // f is not read from here on.
    for (std::size_t j = 0; j < blocks; ++j) {
        const std::size_t first = j * k, count = std::min(k, n - 1 - first);
        if (j == 0) {
            std::copy_n(d, count, R);
        } else {  // R = d[jk, jk + count) - sum_(i<j) (W_(j-i) q_i)[k, 2k)
            t.multiply(W[j], Q[0], R);
            for (std::size_t i = 1; i < j; ++i) t.multiply_add(R, W[j - i], Q[i]);
            t.inverse(R);
            std::size_t c = 0;
            for (; c + kLanes <= count; c += kLanes)
                S::store(R + c, M::sub(S::load(d + first + c), S::load(R + k + c)));
            for (; c < count; ++c) R[c] = d[first + c] >= R[k + c] ? d[first + c] - R[k + c] : d[first + c] + kP - R[k + c];
        }
        zero(R + count, len - count + ntt::kPadding);
        t.forward(R);
        t.multiply(R, H, R);
        t.inverse(R);  // q_j = R[0, count)
        if (j + 1 < blocks) {
            std::copy_n(R, k, Q[j]);
            zero(Q[j] + k, k + ntt::kPadding);
            t.forward(Q[j]);
        }
        multiply_montgomery(R, recip + first + 1, g.data() + first + 1, count);
    }
    g[0] = 0;
}

// ---- exp --------------------------------------------------------------------------------------

inline std::size_t exp_words(std::size_t n) {
    if (n <= kExpBase) return 0;
    const std::size_t len = std::bit_ceil(n);
    return 2 * footprint(n) + footprint(len / 2) + 4 * footprint(len);
}

// The state of exp's Newton steps at precision m: g = exp f mod x^m in out, h = 1 / g mod x^(m/2),
// H = T_m(h), Gt = T_m(g mod x^(m/2)) (after a full step), q = x f' (n words), recip (n words).
struct ExpState {
    Out g;
    const u32 *q, *recip;
    u32 *h, *Gt, *H, *W, *X;
};

// s[i] = (q[m + i] + t[i] - q[i]) / (m + i) into dst, i < c: (f - log g)[m, m + c).
inline void exp_residual(const ExpState& e, std::size_t m, const u32* t, u32* dst, std::size_t c) {
    std::size_t i = 0;
    for (; i + kLanes <= c; i += kLanes) {
        const V x = M::sub(M::add(S::load(e.q + m + i), S::load(t + i)), S::load(e.q + i));
        S::store(dst + i, M::template reduce<kP>(M::montgomery(x, S::load(e.recip + m + i))));
    }
    for (; i < c; ++i) {
        const u32 a = u32((u64(e.q[m + i]) + t[i] + kP - e.q[i]) % kP);
        dst[i] = mul(mul(a, e.recip[m + i]), kRinv);
    }
}

// g[m, 2m) (c = min(m, n - m) words of it), h to precision m, H = T_2m(h), Gt = T_2m(g mod x^m).
inline void exp_full_step(ExpState& e, std::size_t m, std::size_t c) {
    const int lg = std::countr_zero(m);
    const ntt::Transform &tm = transform(lg), &t2 = transform(lg + 1);
    const std::size_t half = m / 2, len = 2 * m;
    u32 *const Gt = e.Gt, *const H = e.H, *const W = e.W, *const X = e.X;
    // G_lo = T_m(g); e = (g h)[m/2, m) at W[m/2, m)
    std::copy_n(e.g.data(), m, Gt);
    zero(Gt + m, ntt::kPadding);
    tm.forward(Gt);
    tm.multiply(Gt, H, W);
    tm.inverse(W);
    // h[m/2, m) = -(x^(m/2) e h mod (x^m - 1))[m/2, m)
    zero(W, half);
    negate(W + half, W + half, half);
    zero(W + m, ntt::kPadding);
    tm.forward(W);
    tm.multiply(W, H, W);
    tm.inverse(W);
    std::copy_n(W + half, half, e.h + half);
    // T_m(r) = T_m(q mod x^m) G_lo at X[0, m); r = q g mod (x^m - 1) = x g' + (q g)[m, 2m) at W[0, m)
    std::copy_n(e.q, m, X);
    zero(X + m, ntt::kPadding);
    tm.forward(X);
    tm.multiply(X, Gt, X);
    std::copy_n(X, m, W);
    tm.inverse(W);
    // H = T_2m(h); X = T_2m(r) = [T_m(r), T_m(twisted r)]; t = h r mod x^m at X[0, m)
    std::copy_n(e.h, m, H);
    zero(H + m, m + ntt::kPadding);
    t2.forward(H);
    twist(W, X + m, m / 8);
    zero(X + len, ntt::kPadding);
    tm.forward(X + m);
    t2.multiply(X, H, X);
    t2.inverse(X);
    // W = x^m s, s = (f - log g)[m, m + c)
    exp_residual(e, m, X, W + m, c);
    zero(W, m);
    zero(W + m + c, m - c + ntt::kPadding);
    // Gt = T_2m(g mod x^m) = [G_lo, T_m(twisted g)]; g[m, m + c) = (x^m s g mod (x^2m - 1))[m, m + c)
    twist(e.g.data(), Gt + m, m / 8);
    zero(Gt + len, ntt::kPadding);
    tm.forward(Gt + m);
    t2.forward(W);
    t2.multiply(W, Gt, W);
    t2.inverse(W);
    std::copy_n(W + m, c, e.g.data() + m);
}

// The last step, from m to m + c, c <= m/2, at length m: t = h r mod x^(m/2) with h and r mod
// x^(m/2), g[m, m + c) = (g mod x^(m/2)) s mod x^c, by Gt = T_m(g mod x^(m/2)).
inline void exp_half_step(ExpState& e, std::size_t m, std::size_t c, bool first) {
    const ntt::Transform& tm = transform(std::countr_zero(m));
    const std::size_t half = m / 2;
    u32 *const Gt = e.Gt, *const H = e.H, *const W = e.W, *const X = e.X;
    if (first) {  // no full step before: Gt = T_m(g mod x^(m/2))
        std::copy_n(e.g.data(), half, Gt);
        zero(Gt + half, half + ntt::kPadding);
        tm.forward(Gt);
    }
    // r = q g mod (x^m - 1) at W[0, m)
    std::copy_n(e.g.data(), m, X);
    zero(X + m, ntt::kPadding);
    tm.forward(X);
    std::copy_n(e.q, m, W);
    zero(W + m, ntt::kPadding);
    tm.forward(W);
    tm.multiply(W, X, W);
    tm.inverse(W);
    // t = h (r mod x^(m/2)) mod x^(m/2) at W[0, c)
    zero(W + half, half);
    tm.forward(W);
    tm.multiply(W, H, W);
    tm.inverse(W);
    // s at X[0, c); g[m, m + c) = (g mod x^(m/2)) s mod x^c
    exp_residual(e, m, W, X, c);
    zero(X + c, m - c);
    tm.forward(X);
    tm.multiply(X, Gt, X);
    tm.inverse(X);
    std::copy_n(X, c, e.g.data() + m);
}

// g = exp f mod x^n, n = g.size() > kExpBase, f mod x^n given, f[0] = 0. f is read first.
inline void exp_newton(Span f, Out g, Arena arena) {
    const std::size_t n = g.size(), len = std::bit_ceil(n);
    u32* const recip = arena.take(n);
    u32* const q = arena.take(n);
    reciprocals(Out(recip, n), q);
    const std::size_t fl = std::min(f.size(), n);
    multiply_by_index(f.data(), 0, q, fl);  // q = x f'
    zero(q + fl, n - fl);
    // f is not read from here on.
    ExpState e{g, q, recip, arena.take(len / 2), arena.take(len), arena.take(len), arena.take(len), arena.take(len)};
    exp_direct(q, g.first(kExpBase));
    inverse_direct(Span(g.data(), kExpBase / 2), Out(e.h, kExpBase / 2));
    std::copy_n(e.h, kExpBase / 2, e.H);
    zero(e.H + kExpBase / 2, kExpBase / 2 + ntt::kPadding);
    transform(std::countr_zero(kExpBase)).forward(e.H);
    for (std::size_t m = kExpBase; m < n; m *= 2) {
        const std::size_t c = std::min(m, n - m);
        if (2 * c <= m) exp_half_step(e, m, c, m == kExpBase);
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
        u32* E = a.take(len);
        detail::inverse_newton(f, out, G, E);
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
