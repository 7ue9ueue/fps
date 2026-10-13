// Composition of power series modulo P = 998244353 (backend kernel): Kinoshita and Li's power
// projection and its transpose. From lc-opt's lib/poly/{composition,projection,
// compositional_inverse}.hpp, rewritten on ntt::Transform's leaf form and backend/simd, so every
// target runs it. Design and measurements: notes.md.
//
//   fps::composition::compose(f, g, out);                // out = f(g) mod x^n, n = out.size()
//   fps::composition::power_projection(g, out);          // out[k] = [x^(n-1)] g^k, k < n
//   fps::composition::compositional_inverse(f, out);     // out = g with f(g) = x mod x^n
//
// Spans of words < P. Only f and g mod x^n are read (shorter ones have zeros past their end);
// g[0] and f[0] are not read and count as 0. compositional_inverse needs f[1] != 0 and n <= P
// (not checked: the frontend checks them). out must not overlap f or g. n past kMaxSize throws
// std::length_error before out is written.
//
// Kinoshita and Li (arXiv 2404.05177), with lc-opt's layout. m = 2^T >= n, Q_0 = 1 - y g(x),
// Q_(s+1)(x^2, y) = Q_s(x, y) Q_s(-x, y) mod x^L, L = m / 2^s; Q_s has degree Y = 2^s in y and
// Q_s(x, 0) = Q_s(0, y) = 1. Level s is stored at stride 2L (x = z, y = z^(2L), x below L), and
// its products are transforms of length 4m: no carry from x into y, and y wraps mod y^(2Y).
// - Down (Graeffe), a generic level: the leaf form of Q_s at 4m. Leaves 2p, 2p + 1 have the
//   moduli z^8 -+ s, s = r(p), and Q(-z) mod (z^8 - w) is the leaf with its odd coefficients
//   negated, so V = Q(z) Q(-z) mod (u^4 -+ s), u = z^2, is c(z) c(-z) per leaf (20 products in
//   64-bit sums, one Montgomery reduction per output: lc-opt's leaf sums), and their CRT is leaf p
//   of V's leaf form at 2m (s^2 = w_p). The Graeffe runs in the forward's bottom (a hook of
//   ntt::Transform::forward_hooked, in cache); V's inverse at 2m in place gives V at stride L;
//   row 0 = 1 + row 2Y (the wrap), x below L/2 kept: Q_(s+1).
// - Power projection: P_0 = x^(m-n), P_(s+1)(x^2) x = the odd part of P_s(x) Q_s(-x) mod x^L,
//   per leaf the odd coefficients of a(z) c(-z) (32 products) in Q's forward hook, the same CRT.
// - compose is its transpose. With P^_(s+1) = P_(s+1) reversed in x and y (stride L in u = z^2),
//   R = P^(z^2) Q_s(-z) mod (z^4m - 1) holds P^_s in rows Y .. 2Y - 1, x below L: the next
//   input as it stands. P^_T = f reversed; h[k] = row 1 of level 0's R. Per level: the leaf form
//   of P^ at 2m, then R's inverse at 4m (upper half) whose hook computes R's leaves from P^'s leaf
//   p (P^(z^2) mod (z^8 -+ s) = lo +- s hi) and Q_s's stored leaves: 32 products per leaf.
// - Pruned transforms (ntt's column bits): each row holds x below half its stride, so the levels
//   above that bit run on half the columns: forward of Q_s (bit L), inverse of V (L/2), forward
//   of P^ (L/2), inverse of R (L).
// - One-dimensional levels (lc-opt's): level 0 (Q_0 = 1 - y g: a Graeffe of g at 2m, compose's
//   step a product of 2m), level 1 (Q_1 = 1 + y q1 + y^2 q2: transforms of q1, q2 at m, products
//   G(q1), G(q2), E(q1, q2) in one hook, inverses at m/2), and from m = 256 levels T - 3, T - 2
//   as polynomials in y per column (transforms of m/4, m/2), and T - 1 (Q = 1 + x q(y)).
// Zen 3, n = 2^17: compose 31.7 ms against lc-opt's 29.2 (was 42.4), power projection 30.3
// against 25.8 (notes.md).
//
// compositional_inverse: Lagrange, (n-1) [x^(n-1)] f^k = k [x^(n-1-k)] (x / g)^(n-1), so with
// a = power_projection(f), h[j] = a[N-j] / (N-j) = [x^j] (x / g)^N / N for N = n - 1, and
// g / x = exp(-log(h / h[0]) / N) / f[1] mod x^N (series::log, series::exp).
//
// Memory, kept per thread across calls (the largest so far): compose about 16 m (T + 3) bytes
// (the leaf forms of Q_0 .. Q_(T-4), 4m words each, and the column levels' arrays: 40 MiB at m =
// 2^17; 1.6 GiB at m = 2^22), power projection about 150 m bytes. Twiddle tables: one
// ntt::Transform per length, per thread.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/intrinsics.hpp"
#include "backend/ntt/memory.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/series/series.hpp"
#include "backend/simd/select.hpp"

namespace fps::composition {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 kP = ntt::kModulus;
inline constexpr std::size_t kMaxSize = std::size_t(1) << (ntt::kMaxLog - 2);  // 4m <= 2^kMaxLog

namespace detail {

using Span = std::span<const u32>;
using Out = std::span<u32>;

constexpr u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kP); }
constexpr u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
constexpr u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + kP - b; }
constexpr u32 kR = u32((u64(1) << 32) % kP);  // 2^32 mod P
constexpr u32 kHalf = (kP + 1) / 2;
inline u32 at(Span f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// Up to this many coefficients, the direct methods.
inline constexpr std::size_t kComposeBase = 16;
inline constexpr std::size_t kProjectionBase = 16;
// m = 2^T >= n, at least 64: transforms of m >= 2^kMinLog words, m / 4 leaf pairs (a multiple of 8).
inline std::size_t length_for(std::size_t n) { return std::max<std::size_t>(std::bit_ceil(n), 64); }

// ---- Memory -----------------------------------------------------------------------------------

// Words after each array: ntt::kPadding read by the transforms, and the rest so that arrays of a
// power-of-two length sit 320 bytes apart modulo 4 KiB (4K aliasing, backend/ntt/notes.md).
inline constexpr std::size_t kGap = 80;
constexpr std::size_t footprint(std::size_t words) { return (words + 15) / 16 * 16 + kGap; }

// This thread's memory, kept across calls: the largest region asked for so far.
struct Kept {
    ntt::detail::Region region;
    ~Kept() { ntt::detail::unmap_region(region); }
};
inline u32* thread_words(std::size_t words) {
    thread_local Kept kept;
    const std::size_t bytes = words * sizeof(u32);
    if (kept.region.bytes < bytes) {
        ntt::detail::unmap_region(std::exchange(kept.region, {}));
        kept.region = ntt::detail::map_region(bytes);
    }
    return reinterpret_cast<u32*>(kept.region.start);
}

// 64-byte aligned arrays carved from words words at start, in order; past them std::logic_error.
class Arena {
public:
    Arena(u32* start, std::size_t words) : next_(start), end_(start + words) {}
    u32* take(std::size_t words) {
        u32* p = next_;
        next_ += footprint(words);
        if (next_ > end_) throw std::logic_error("fps::composition: arena too small");
        return p;
    }

private:
    u32 *next_, *end_;
};

// This thread's transform of length 2^lg, kept across calls.
inline const ntt::Transform& transform(int lg) {
    thread_local std::array<std::unique_ptr<ntt::Transform>, ntt::kMaxLog + 1> cache;
    if (!cache[lg]) cache[lg] = std::make_unique<ntt::Transform>(lg);
    return *cache[lg];
}

// Per leaf pair p of a transform of 4m words (moduli z^8 -+ s, s = r(p)), for p < m / 4: s 2^32
// (Montgomery's form of s) and 2^32 / (2 s) (the CRT's factor: its Montgomery product keeps the
// sums' 2^-32 for the hooked inverse). r(p) does not depend on the length: one table per thread, grown to the longest.
struct PairTables {
    std::vector<u32> s, crt;
};
inline const PairTables& pair_tables(std::size_t pairs) {
    thread_local PairTables t;
    if (t.s.size() < pairs) {
        const std::size_t old = t.s.size(), size = std::max(pairs, 2 * old);
        std::vector<u32> r(size), ri(size);
        r[0] = ri[0] = 1;
        for (std::size_t p = 1; p < size; ++p) {
            const int j = std::countr_zero(p);
            r[p] = mul(r[p & (p - 1)], ntt::detail::kRoots[0][j]);
            ri[p] = mul(ri[p & (p - 1)], ntt::detail::kRoots[1][j]);
        }
        t.s.resize(size), t.crt.resize(size);
        for (std::size_t p = 0; p < size; ++p) t.s[p] = mul(r[p], kR), t.crt[p] = mul(mul(ri[p], kHalf), kR);
    }
    return t;
}

// ---- Leaf passes (lanes = leaf pairs) ---------------------------------------------------------

template <class S>
using Narrow = typename ntt::detail::intrinsics::Narrow<S>::type;

// L: a simd struct of 1, 4 or 8 lanes; lane l of a set is leaf pair p + l.
template <class L>
struct Pairs {
    using Lanes = L;
    using V = typename L::V;
    using M = modarith::Ops<L, kP>;
    static constexpr int W = L::lanes;
    using Set = V[8];

    // x[k] lane l = p[stride l + k] for k < 8, and back.
    [[gnu::always_inline]] static void load(const u32* p, std::size_t stride, Set& x) {
        if constexpr (W == 1) {
#pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) x[k] = p[k];
        } else {
#pragma GCC unroll 8
            for (int h = 0; h < 8; h += W) {
                V r[W];
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) r[u] = L::load(p + stride * std::size_t(u) + std::size_t(h));
                ntt::detail::intrinsics::Values<L>::transpose(r);
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) x[h + u] = r[u];
            }
        }
    }
    [[gnu::always_inline]] static void store(u32* p, std::size_t stride, const Set& x) {
        if constexpr (W == 1) {
#pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) p[k] = x[k];
        } else {
#pragma GCC unroll 8
            for (int h = 0; h < 8; h += W) {
                V r[W];
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) r[u] = x[h + u];
                ntt::detail::intrinsics::Values<L>::transpose(r);
#pragma GCC unroll 8
                for (int u = 0; u < W; ++u) L::store(p + stride * std::size_t(u) + std::size_t(h), r[u]);
            }
        }
    }
    [[gnu::always_inline]] static V red(V x) { return M::template reduce<kP>(x); }           // [0, 2P) -> [0, P)
    [[gnu::always_inline]] static V red4(V x) { return red(M::template reduce<2 * kP>(x)); }  // [0, 4P) -> [0, P)
    [[gnu::always_inline]] static V mont(V a, V b) { return red(M::montgomery(a, b)); }      // a b 2^-32, [0, P)

    // ---- Leaf sums in 64 bits (lc-opt's): x86 per 64-bit lane (mul_even, the even words, then the
    // odd words shifted down), Neon per lane (umull and umull2 into two 64-bit halves; the adds
    // fold into umlal), and Scalar. Each output is one sum of products of canonical words, the
    // negative terms offset by multiples of 4 P^2 (below 2^64), and one Montgomery reduction.
    static constexpr bool kX86 = requires(V v) { L::mul_even(v, v); };
    static constexpr bool kNeon = requires(V v) { L::mul_wide_hi(v, v); };
    static_assert(kX86 || kNeon || W == 1, "composition: no 64-bit products on this simd struct");
#if defined(__ARM_NEON)
    struct NeonSum {
        uint64x2_t lo, hi;  // lanes 0, 1 and 2, 3
    };
    using U = std::conditional_t<W == 1, u64, std::conditional_t<kNeon, NeonSum, V>>;
#else
    using U = std::conditional_t<W == 1, u64, V>;  // a 64-bit sum (per 64-bit lane on x86)
#endif
    static constexpr u64 kOffset = 4 * u64(kP) * kP;
    static constexpr u32 kNegInverse = u32(0) - u32(modarith::inverse_mod_power_of_two(kP, 32));

    [[gnu::always_inline]] static U product(V a, V b) {
        if constexpr (W == 1) return u64(a) * b;
#if defined(__ARM_NEON)
        else if constexpr (kNeon) return U{L::mul_wide_lo(a, b), L::mul_wide_hi(a, b)};
#endif
        else return L::mul_even(a, b);
    }
    [[gnu::always_inline]] static U plus(U a, U b) {
        if constexpr (W == 1) return a + b;
#if defined(__ARM_NEON)
        else if constexpr (kNeon) return U{vaddq_u64(a.lo, b.lo), vaddq_u64(a.hi, b.hi)};
#endif
        else return L::add64(a, b);
    }
    [[gnu::always_inline]] static U minus(U a, U b) {
        if constexpr (W == 1) return a - b;
#if defined(__ARM_NEON)
        else if constexpr (kNeon) return U{vsubq_u64(a.lo, b.lo), vsubq_u64(a.hi, b.hi)};
#endif
        else return L::sub64(a, b);
    }
    [[gnu::always_inline]] static U twice(U a) { return plus(a, a); }
    [[gnu::always_inline]] static U offset() {
        if constexpr (W == 1) return kOffset;
#if defined(__ARM_NEON)
        else if constexpr (kNeon) return U{vdupq_n_u64(kOffset), vdupq_n_u64(kOffset)};
#endif
        else return L::broadcast64(kOffset);
    }
    // x 2^-32 mod P in [0, x / 2^32 + P) per 64-bit sum (Montgomery), from the sums of the even
    // words and of the odd words (x86; Neon and Scalar: even only); canonical.
    [[gnu::always_inline]] static V reduce_sums(U even, U odd) {
        if constexpr (W == 1) {
            return red4(V((even + u64(u32(even) * kNegInverse) * kP) >> 32));
        }
#if defined(__ARM_NEON)
        else if constexpr (kNeon) {
            const V q = L::mul_lo(L::lo_halves(even.lo, even.hi), L::broadcast(kNegInverse)), p = L::broadcast(kP);
            return red4(L::hi_halves(L::mla_wide_lo(even.lo, q, p), L::mla_wide_hi(even.hi, q, p)));
        }
#endif
        else {
            const V ninv = L::broadcast(kNegInverse), p = L::broadcast(kP);
            even = L::add64(even, L::mul_even(L::mul_even(even, ninv), p));
            odd = L::add64(odd, L::mul_even(L::mul_even(odd, ninv), p));
            return red4(L::join_high(even, odd));
        }
    }
    // The odd words of each lane pair moved to the even ones (x86).
    [[gnu::always_inline]] static void odd_words(const Set& x, Set& out) {
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) out[i] = L::template shr64<32>(x[i]);
    }
    // tc[j] = t c[j] for j >= first (canonical), tR = t 2^32; the rest unset.
    [[gnu::always_inline]] static void scaled(const Set& c, V tR, int first, Set& tc) {
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) tc[j] = j < first ? L::broadcast(0) : mont(c[j], tR);
    }
    // Coefficient 2k of c(z) c(-z) mod (z^8 - t): sum over i + j = 2k of (-1)^j c_i c_j, and over
    // i + j = 2k + 8 of (-1)^j c_i tc_j (tc for j >= 4). 20 products.
    [[gnu::always_inline]] static void graeffe_sums(const Set& c, const Set& tc, U (&sum)[4]) {
        const auto x = [&c](int i, int j) { return product(c[i], c[j]); };
        const auto t = [&c, &tc](int i, int j) { return product(c[i], tc[j]); };
        sum[0] = minus(minus(plus(plus(x(0, 0), twice(t(2, 6))), plus(t(4, 4), offset())), twice(t(1, 7))), twice(t(3, 5)));
        sum[1] = minus(minus(plus(plus(twice(x(0, 2)), twice(t(4, 6))), offset()), x(1, 1)), plus(twice(t(3, 7)), t(5, 5)));
        sum[2] = minus(minus(plus(plus(twice(x(0, 4)), x(2, 2)), plus(t(6, 6), offset())), twice(x(1, 3))), twice(t(5, 7)));
        sum[3] = minus(minus(plus(plus(twice(x(0, 6)), twice(x(2, 4))), offset()), twice(x(1, 5))), plus(x(3, 3), t(7, 7)));
    }
    // Coefficient 2k + 1 of a(z) c(-z) mod (z^8 - t): sum over i + j = 2k + 1 of (-1)^j a_i c_j and
    // over i + j = 2k + 9 of (-1)^j a_i tc_j (tc for j >= 2). 32 products.
    [[gnu::always_inline]] static void odd_sums(const Set& a, const Set& c, const Set& tc, U (&sum)[4]) {
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            U s = offset();
#pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) {
                const int j = 2 * k + 1 - i, l = j + 8;
                if (j >= 0) s = j % 2 ? minus(s, product(a[i], c[j])) : plus(s, product(a[i], c[j]));
                else if (l < 8) s = l % 2 ? minus(s, product(a[i], tc[l])) : plus(s, product(a[i], tc[l]));
            }
            sum[k] = s;
        }
    }
    // Coefficient 2k of a(z) c(-z) mod (z^8 - t): sum over i + j = 2k of (-1)^j a_i c_j and over
    // i + j = 2k + 8 of (-1)^j a_i tc_j (tc for j >= 1). 32 products.
    [[gnu::always_inline]] static void even_sums(const Set& a, const Set& c, const Set& tc, U (&sum)[4]) {
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            U s = offset();
#pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) {
                const int j = 2 * k - i, l = j + 8;
                if (j >= 0) s = j % 2 ? minus(s, product(a[i], c[j])) : plus(s, product(a[i], c[j]));
                else s = l % 2 ? minus(s, product(a[i], tc[l])) : plus(s, product(a[i], tc[l]));
            }
            sum[k] = s;
        }
    }
    [[gnu::always_inline]] static void even_wide(const Set& a, const Set& c, const Set& tc, V (&B)[4]) {
        U even[4], odd[4] = {};
        even_sums(a, c, tc, even);
        if constexpr (kX86) {
            Set a2, c2, tc2;
            odd_words(a, a2), odd_words(c, c2), odd_words(tc, tc2);
            even_sums(a2, c2, tc2, odd);
        }
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) B[k] = reduce_sums(even[k], odd[k]);
    }
    [[gnu::always_inline]] static void graeffe_wide(const Set& c, const Set& tc, V (&A)[4]) {
        U even[4], odd[4] = {};
        graeffe_sums(c, tc, even);
        if constexpr (kX86) {
            Set c2, tc2;
            odd_words(c, c2), odd_words(tc, tc2);
            graeffe_sums(c2, tc2, odd);
        }
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) A[k] = reduce_sums(even[k], odd[k]);
    }
    [[gnu::always_inline]] static void odd_wide(const Set& a, const Set& c, const Set& tc, V (&B)[4]) {
        U even[4], odd[4] = {};
        odd_sums(a, c, tc, even);
        if constexpr (kX86) {
            Set a2, c2, tc2;
            odd_words(a, a2), odd_words(c, c2), odd_words(tc, tc2);
            odd_sums(a2, c2, tc2, odd);
        }
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) B[k] = reduce_sums(even[k], odd[k]);
    }

    // A = c(z) c(-z) mod (z^8 - t) as a series in u = z^2 mod (u^4 - t), times 2^-32, canonical:
    // e^2 - u o^2 for c = e(u) + z o(u); tR = t 2^32 per lane.
    [[gnu::always_inline]] static void graeffe(const Set& c, V tR, V (&A)[4]) {
        Set tc;
        scaled(c, tR, 4, tc);
        graeffe_wide(c, tc, A);
    }

    // The odd part of a(z) c(-z) mod (z^8 - t), divided by z, as a series in u mod (u^4 - t), times
    // 2^-32, canonical: po ce - pe co for a = pe(u) + z po(u), c = ce(u) + z co(u).
    [[gnu::always_inline]] static void odd_product(const Set& a, const Set& c, V tR, V (&B)[4]) {
        Set tc;
        scaled(c, tR, 2, tc);
        odd_wide(a, c, tc, B);
    }

    // graeffe(c, tR, A) and odd_product(a, c, tR, B) with c's t c_j and odd words shared.
    [[gnu::always_inline]] static void graeffe_odd(const Set& c, const Set& a, V tR, V (&A)[4], V (&B)[4]) {
        Set tc;
        scaled(c, tR, 2, tc);
        U ge[4], go[4] = {}, oe[4], oo[4] = {};
        graeffe_sums(c, tc, ge);
        odd_sums(a, c, tc, oe);
        if constexpr (kX86) {
            Set c2, tc2, a2;
            odd_words(c, c2), odd_words(tc, tc2), odd_words(a, a2);
            graeffe_sums(c2, tc2, go);
            odd_sums(a2, c2, tc2, oo);
        }
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) A[k] = reduce_sums(ge[k], go[k]), B[k] = reduce_sums(oe[k], oo[k]);
    }

    // The even part of a(z) c(-z) mod (z^8 - t) as a series in u mod (u^4 - t), times 2^-32,
    // canonical: ae ce - u ao co.
    [[gnu::always_inline]] static void even_product(const Set& a, const Set& c, V tR, V (&B)[4]) {
        Set tc;
        scaled(c, tR, 1, tc);
        even_wide(a, c, tc, B);
    }

    // Leaf p of the transform of length 2m from A mod (u^4 - s) and B mod (u^4 + s), both times
    // 2^-32: ((A + B) / 2 + u^4 (A - B) / (2 s)) 2^-32, canonical (the hooked inverse takes the 2^-32).
    [[gnu::always_inline]] static void crt(const V (&A)[4], const V (&B)[4], V c, Set& out) {
        const V half = L::broadcast(mul(kHalf, kR)), p = L::broadcast(kP);
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            out[k] = red(M::montgomery(L::add(A[k], B[k]), half));
            out[k + 4] = red(M::montgomery(L::add(L::sub(A[k], B[k]), p), c));
        }
    }

    // The runs below take the leaf pairs [p0, p0 + pairs) (pairs a multiple of W); their pointers
    // are at those pairs: leaf 2 p0 of a leaf form of 4m words, leaf p0 of one of 2m, entry p0 of
    // the tables.

    // Graeffe: F (Q's leaf form at 4m) to G (V's leaf form at 2m, times 2^-32).
    static void graeffe_run(const u32* F, u32* G, std::size_t p0, std::size_t pairs, const PairTables& t) {
        const V p = L::broadcast(kP);
        for (std::size_t q = 0; q < pairs; q += W) {
            Set a, b, out;
            load(F + 16 * q, 16, a), load(F + 16 * q + 8, 16, b);
            const V s = L::load(t.s.data() + p0 + q);
            V A[4], B[4];
            graeffe(a, s, A);
            graeffe(b, L::sub(p, s), B);
            crt(A, B, L::load(t.crt.data() + p0 + q), out);
            store(G + 8 * q, 8, out);
        }
    }

    // Power projection: from FQ and FP (4m, the leaf forms of Q and P) to GV and GW (2m, the leaf
    // forms of V and of W = the odd part of P(x) Q(-x) over x, times 2^-32).
    static void projection_run(const u32* FQ, const u32* FP, u32* GV, u32* GW, std::size_t p0, std::size_t pairs,
                               const PairTables& t) {
        const V p = L::broadcast(kP);
        for (std::size_t q = 0; q < pairs; q += W) {
            Set ca, cb, pa, pb, out;
            load(FQ + 16 * q, 16, ca), load(FQ + 16 * q + 8, 16, cb);
            load(FP + 16 * q, 16, pa), load(FP + 16 * q + 8, 16, pb);
            const V s = L::load(t.s.data() + p0 + q), ns = L::sub(p, s), c = L::load(t.crt.data() + p0 + q);
            V A[4], B[4], OA[4], OB[4];
            graeffe_odd(ca, pa, s, A, OA);
            graeffe_odd(cb, pb, ns, B, OB);
            crt(A, B, c, out);
            store(GV + 8 * q, 8, out);
            crt(OA, OB, c, out);
            store(GW + 8 * q, 8, out);
        }
    }

    // Level 1, one-dimensional: from the leaf forms at m of q1 (F1), q2 (F2) and (power projection;
    // else null) p1 (FP), per leaf pair the leaf forms at m/2, times 2^-32, of G(q1), G(q2), E(q1,
    // q2) and O(p1, q1), O(p1, q2) into out[0 .. 4]: G(a) = a(x) a(-x), E and O the even and the odd
    // part (over x) of a(x) b(-x).
    static void level1_run(const u32* F1, const u32* F2, const u32* FP, u32* const* out, std::size_t p0, std::size_t pairs,
                           const PairTables& t) {
        const V p = L::broadcast(kP);
        for (std::size_t q = 0; q < pairs; q += W) {
            Set ca, cb, da, db, r;
            load(F1 + 16 * q, 16, ca), load(F1 + 16 * q + 8, 16, cb);
            load(F2 + 16 * q, 16, da), load(F2 + 16 * q + 8, 16, db);
            const V s = L::load(t.s.data() + p0 + q), ns = L::sub(p, s), c = L::load(t.crt.data() + p0 + q);
            V A[4], B[4];
            graeffe(ca, s, A), graeffe(cb, ns, B), crt(A, B, c, r), store(out[0] + 8 * q, 8, r);
            graeffe(da, s, A), graeffe(db, ns, B), crt(A, B, c, r), store(out[1] + 8 * q, 8, r);
            even_product(ca, da, s, A), even_product(cb, db, ns, B), crt(A, B, c, r), store(out[2] + 8 * q, 8, r);
            if (!FP) continue;
            Set pa, pb;
            load(FP + 16 * q, 16, pa), load(FP + 16 * q + 8, 16, pb);
            odd_product(pa, ca, s, A), odd_product(pb, cb, ns, B), crt(A, B, c, r), store(out[3] + 8 * q, 8, r);
            odd_product(pa, da, s, A), odd_product(pb, db, ns, B), crt(A, B, c, r), store(out[4] + 8 * q, 8, r);
        }
    }

    // out = sum over K terms of a(z^2) c(-z) mod (z^8 - t), times 2^-32, canonical, for a of 4
    // coefficients (in u = z^2): coefficient k is (-1)^k (a(z^2) c(z))_k, a sum of 4 products (t a_i
    // where 2i > k) per term.
    template <int K>
    [[gnu::always_inline]] static void spread_product(const V (&a)[K][4], const Set* const (&c)[K], V tR, Set& out) {
        V ta[K][4];
#pragma GCC unroll 2
        for (int e = 0; e < K; ++e) {
            ta[e][0] = a[e][0];
#pragma GCC unroll 3
            for (int i = 1; i < 4; ++i) ta[e][i] = mont(a[e][i], tR);
        }
        {
            const auto sums = [](const V (&x)[K][4], const V (&tx)[K][4], const Set* const (&y)[K], U (&sum)[8]) {
#pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) {
                    U s = product(x[0][0], (*y[0])[k]);
#pragma GCC unroll 2
                    for (int e = 0; e < K; ++e)
#pragma GCC unroll 4
                        for (int i = e ? 0 : 1; i < 4; ++i)
                            s = plus(s, product(2 * i <= k ? x[e][i] : tx[e][i], (*y[e])[(k - 2 * i) & 7]));
                    if (k % 2) {  // K 4 P^2 > s, = 0 mod P
                        U o = offset();
                        if constexpr (K == 2) o = plus(o, o);
                        s = minus(o, s);
                    }
                    sum[k] = s;
                }
            };
            U even[8], odd[8] = {};
            sums(a, ta, c, even);
            if constexpr (kX86) {
                V a2[K][4], ta2[K][4];
                Set c2[K];
                const Set* c2p[K];
#pragma GCC unroll 2
                for (int e = 0; e < K; ++e) {
#pragma GCC unroll 4
                    for (int i = 0; i < 4; ++i)
                        a2[e][i] = L::template shr64<32>(a[e][i]), ta2[e][i] = L::template shr64<32>(ta[e][i]);
                    odd_words(*c[e], c2[e]);
                    c2p[e] = &c2[e];
                }
                sums(a2, ta2, c2p, odd);
            }
#pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) out[k] = reduce_sums(even[k], odd[k]);
        }
    }

    // compose's backward step: R = sum over K terms of P^(z^2) Q(-z) at 4m (leaves 2p, 2p + 1, times
    // 2^-32) from FP (P^(u)'s leaf form at 2m, leaf p = lo(u) + u^4 hi(u) mod u^8 - s^2) and FQ (Q's
    // at 4m): P^(z^2) mod (z^8 -+ s) = lo(u) +- s hi(u), 32 products per leaf and term.
    static void product_run(const u32* FP, const u32* FQ, u32* R, std::size_t p0, std::size_t pairs, const PairTables& t) {
        const u32* const fp[1] = {FP};
        const u32* const fq[1] = {FQ};
        product_run<1>(fp, fq, R, p0, pairs, t);
    }
    template <int K>
    static void product_run(const u32* const (&FP)[K], const u32* const (&FQ)[K], u32* R, std::size_t p0, std::size_t pairs,
                            const PairTables& t) {
        const V p = L::broadcast(kP);
        for (std::size_t q = 0; q < pairs; q += W) {
            Set ca[K], cb[K], out;
            V a[K][4], b[K][4];
            const V s = L::load(t.s.data() + p0 + q);
#pragma GCC unroll 2
            for (int e = 0; e < K; ++e) {
                Set x;
                load(FP[e] + 8 * q, 8, x);
                load(FQ[e] + 16 * q, 16, ca[e]), load(FQ[e] + 16 * q + 8, 16, cb[e]);
#pragma GCC unroll 4
                for (int k = 0; k < 4; ++k) {
                    const V hi = mont(x[k + 4], s);
                    a[e][k] = M::add(x[k], hi), b[e][k] = M::sub(x[k], hi);
                }
            }
            const Set* pa[K];
            const Set* pb[K];
#pragma GCC unroll 2
            for (int e = 0; e < K; ++e) pa[e] = &ca[e], pb[e] = &cb[e];
            spread_product<K>(a, pa, s, out);
            store(R + 16 * q, 16, out);
            spread_product<K>(b, pb, L::sub(p, s), out);
            store(R + 16 * q + 8, 16, out);
        }
    }
};

using Native = Pairs<Narrow<simd::Native>>;

// From V at stride L (rows 0 .. 2Y - 1 in q[0, 2m), row 0 = the true row 0 = 1 plus row 2Y), the
// next level: rows 0 .. 2Y at stride L, x below L / 2 (the rest of rows 0 .. 2Y - 1 zero).
inline void next_level(u32* q, std::size_t m, std::size_t L) {
    const std::size_t half = L / 2, rows = 2 * m / L;
    std::copy_n(q, half, q + 2 * m);
    q[2 * m] = sub(q[2 * m], 1);
    q[0] = 1;
    std::fill_n(q + 1, half - 1, 0u);
    for (std::size_t j = 0; j < rows; ++j) std::fill_n(q + j * L + half, half, 0u);
}

// ---- Few coefficients --------------------------------------------------------------------------

// h = f(g) mod x^n by Horner's rule, n = h.size(): h = h g + f_i from the top coefficient down,
// so h[k] reads only h[0, k) (g[0] = 0).
inline void compose_direct(Span f, Span g, Out h) {
    const std::size_t n = h.size();
    std::fill(h.begin(), h.end(), 0u);
    for (std::size_t i = std::min(f.size(), n); i-- > 0;) {
        for (std::size_t k = n; k-- > 1;) {
            u64 sum = 0;
            for (std::size_t j = 1; j <= k && j < g.size(); ++j) sum += u64(g[j]) * h[k - j] % kP;
            h[k] = u32(sum % kP);
        }
        h[0] = f[i];
    }
}

// a[k] = [x^(n-1)] g^k for k < n = a.size(), from the powers of g.
inline void projection_direct(Span g, Out a) {
    const std::size_t n = a.size();
    std::vector<u32> pw(n, 0), next(n);
    pw[0] = 1;
    for (std::size_t k = 0; k < n; ++k) {
        a[k] = pw[n - 1];
        for (std::size_t i = 0; i < n; ++i) {
            u64 c = 0;
            for (std::size_t j = 1; j <= i && j < g.size(); ++j) c += u64(g[j]) * pw[i - j] % kP;
            next[i] = u32(c % kP);
        }
        std::swap(pw, next);
    }
}

// ---- Kinoshita and Li ---------------------------------------------------------------------------

// ---- Column levels ------------------------------------------------------------------------------
// Levels T - 3, T - 2 (L = 8, 4 columns; from m = kColumnsMin on) and T - 1 (L = 2), lc-opt's
// one-dimensional levels: Q_s = sum_(k<L) x^k q_k(y) as the polynomials c_k = (-1)^k q_k in y (the
// columns of Q_s(-x); c_0 = 1, c_k(0) = 0, degree at most Y), with products in y of N = 2Y = 2m/L
// words: cyclic, the one wrapped coefficient (y^N) lands on y^0, which is 0.
// - Down: Q_(s+1)'s columns c'_j = (-1)^j sum_(a+b=2j) (-1)^a c_a c_b, N + 1 coefficients:
//   L = 8: c'_1 = c1^2 - 2 c2, c'_2 = c2^2 - 2 c1 c3 + 2 c4, c'_3 = c3^2 + 2 c1 c5 - 2 c2 c4 - 2 c6;
//   L = 4: c'_1 = c1^2 - 2 c2. The factors 2 sit in scaled leaf forms (D1 = 2 C1, ...).
// - compose's up step: R's columns r_c = sum_(2j+k=c) P_j c_k, rows Y .. 2Y - 1: the upper half
//   of a cyclic product of N, P^'s columns P_j (j < L/2, N rows) in leaf form.
// - Power projection: P_(s+1)'s columns p'_j = sum_(a+b=2j+1) p_a c_b (p_a: Y rows, so no wrap).
// Per level: L - 1 + L/2 forwards and L/2 - 1 + L - 1 inverses of N (compose), against the
// generic level's 12m transformed words.
inline constexpr std::size_t kColumnsMin = 256;

// Columns k = first .. 7 of rows y < rows of a layout at stride 16 into c[k][y], negated at odd k
// if negate: 8 x 8 transposes (Native's), then one row at a time.
inline void columns8_from(const u32* q, std::size_t rows, u32* const* c, int first, bool negate) {
    using N = Native;
    using L = N::Lanes;
    const typename N::V zero = L::broadcast(0);
    std::size_t y = 0;
    for (; y + N::W <= rows; y += N::W) {
        typename N::Set x;
        N::load(q + 16 * y, 16, x);
        for (int k = first; k < 8; ++k) L::store(c[k] + y, negate && k % 2 ? N::M::sub(zero, x[k]) : x[k]);
    }
    for (; y < rows; ++y)
        for (int k = first; k < 8; ++k) c[k][y] = negate && k % 2 && q[16 * y + k] ? kP - q[16 * y + k] : q[16 * y + k];
}

// The inverse: rows y < rows of p at stride 16, columns 0 .. 7 from r[c][y], columns 8 .. 15 zero.
inline void columns8_to(u32* const* r, std::size_t rows, u32* p) {
    using N = Native;
    using L = N::Lanes;
    typename N::Set zero;
    for (auto& v : zero) v = L::broadcast(0);
    std::size_t y = 0;
    for (; y + N::W <= rows; y += N::W) {
        typename N::Set x;
        for (int c = 0; c < 8; ++c) x[c] = L::load(r[c] + y);
        N::store(p + 16 * y, 16, x);
        N::store(p + 16 * y + 8, 16, zero);
    }
    for (; y < rows; ++y) {
        for (std::size_t c = 0; c < 8; ++c) p[16 * y + c] = r[c][y];
        std::fill_n(p + 16 * y + 8, 8, 0u);
    }
}

// 2 x mod P, x canonical.
constexpr u32 twice(u32 x) { return x >= kP - x ? x - (kP - x) : x + x; }

// to = 2 from, negated if negate (words, canonical).
inline void scale2(const u32* __restrict from, u32* __restrict to, std::size_t words, bool negate) {
    if (negate) {
        for (std::size_t i = 0; i < words; ++i) {
            const u32 x = twice(from[i]);
            to[i] = x ? kP - x : 0;
        }
        return;
    }
    for (std::size_t i = 0; i < words; ++i) to[i] = twice(from[i]);
}

// out (N + 1 words) = the sum of the terms' products in y, unwrapped (out(0) = 0), plus 2 c (minus,
// if negate) for c of N/2 + 1 words, if not null.
inline void column_down(const ntt::Transform& t, std::span<const ntt::Term> terms, u32* work, u32* out, const u32* c,
                        bool negate) {
    const std::size_t N = t.length();
    t.inverse_product(terms, work, Out(out, N));
    out[N] = out[0], out[0] = 0;
    if (!c) return;
    u32* __restrict o = out;
    const u32* __restrict d = c;
    if (negate) {
        for (std::size_t y = 0; y <= N / 2; ++y) o[y] = sub(o[y], twice(d[y]));
        return;
    }
    for (std::size_t y = 0; y <= N / 2; ++y) {
        const u32 x = o[y] + twice(d[y]);
        o[y] = x >= kP ? x - kP : x;
    }
}

// The column levels' arrays, carved from an Arena: per level (L = 8, 4) the columns c_k, their leaf
// forms C_k (kept for compose's up pass), and the scaled forms; c'_1 of Q_(T-1) (m/2 + 1 words).
struct Columns {
    Columns(Arena& arena, std::size_t m) {
        for (int k = 1; k < 8; ++k) c8[k] = arena.take(m / 8 + 1), C8[k] = arena.take(m / 4);
        for (int k = 1; k < 4; ++k) c4[k] = arena.take(m / 4 + 1), C4[k] = arena.take(m / 2);
        D1 = arena.take(m / 4), ND1 = arena.take(m / 4), ND2 = arena.take(m / 4);
        c2 = arena.take(m / 2 + 1);
        for (auto& r : r4) r = arena.take(m / 4);
        for (auto& r : r8) r = arena.take(m / 8);
    }
    static std::size_t words(std::size_t m) {
        return 7 * (footprint(m / 8 + 1) + footprint(m / 4)) + 3 * (footprint(m / 4 + 1) + footprint(m / 2)) +
               3 * footprint(m / 4) + footprint(m / 2 + 1) + 4 * footprint(m / 4) + 8 * footprint(m / 8);
    }
    u32 *c8[8] = {}, *C8[8] = {}, *c4[4] = {}, *C4[4] = {}, *D1, *ND1, *ND2, *c2;
    u32 *r4[4], *r8[8];  // compose's up pass: R's columns at levels T - 2 and T - 3
};

// Down from Q_(T-3) at stride 16 in q (rows 0 .. m/8) through levels T - 3 and T - 2: the leaf forms
// C8, C4 and Q_(T-1)'s column c'_1 = cols.c2 (m/2 + 1 words). work: m/2 + kPadding words.
inline void columns_down(const u32* q, std::size_t m, Columns& cols, u32* work) {
    const int T = std::countr_zero(m);
    const ntt::Transform &t8 = transform(T - 2), &t4 = transform(T - 1);
    columns8_from(q, m / 8 + 1, cols.c8, 1, true);
    for (int k = 1; k < 8; ++k) t8.forward(Span(cols.c8[k], m / 8 + 1), cols.C8[k]);
    u32* const* C = cols.C8;
    scale2(C[1], cols.D1, m / 4, false), scale2(C[1], cols.ND1, m / 4, true), scale2(C[2], cols.ND2, m / 4, true);
    const ntt::Term t1[1] = {{C[1], C[1]}}, t2[2] = {{C[2], C[2]}, {C[3], cols.ND1}},
                    t3[3] = {{C[3], C[3]}, {C[5], cols.D1}, {C[4], cols.ND2}};
    column_down(t8, t1, work, cols.c4[1], cols.c8[2], true);
    column_down(t8, t2, work, cols.c4[2], cols.c8[4], false);
    column_down(t8, t3, work, cols.c4[3], cols.c8[6], true);
    for (int k = 1; k < 4; ++k) t4.forward(Span(cols.c4[k], m / 4 + 1), cols.C4[k]);
    const ntt::Term s1[1] = {{cols.C4[1], cols.C4[1]}};
    column_down(t4, s1, work, cols.c2, cols.c4[2], true);
}

// Q_(T-1)'s column c_1 = -q_1 (m/2 + 1 words) from its layout at stride 4 in q (short transforms).
inline void last_column(const u32* q, std::size_t m, u32* c1) {
    for (std::size_t b = 0; b <= m / 2; ++b) c1[b] = q[4 * b + 1] ? kP - q[4 * b + 1] : 0;
}

// compose's up pass through levels T - 2 and T - 3 (L = 4, 8): from P^_(T-1)'s columns r[0], r[1]
// (m/2 words each) to P^_(T-3) at stride 16 in p (x < 8, rows below m/8; 2m words). work: m/2 +
// kPadding words; forms: 4 arrays of m/2 words (+ kPadding).
inline void columns_up(const Columns& cols, std::size_t m, u32* const* r, u32* const* forms, u32* work, u32* p) {
    u32* const* r2 = cols.r4;
    u32* const* r3 = cols.r8;
    const int T = std::countr_zero(m);
    const ntt::Transform &t8 = transform(T - 2), &t4 = transform(T - 1);
    // Level T - 2: P_0, P_1 at N = m/2; r2[c] = R's column c, rows m/4 .. m/2 - 1.
    {
        const std::size_t N = m / 2;
        u32 *const P0 = forms[0], *const P1 = forms[1];
        t4.forward(Span(r[0], N), P0), t4.forward(Span(r[1], N), P1);
        u32* const* C = cols.C4;
        const ntt::Output upper{.offset = N / 2};
        std::copy_n(r[0] + N / 2, N / 2, r2[0]);
        const ntt::Term a[1] = {{P0, C[1]}}, b[2] = {{P0, C[2]}, {P1, nullptr}}, c[2] = {{P1, C[1]}, {P0, C[3]}};
        t4.inverse_product(a, work, Out(r2[1], N / 2), upper);
        t4.inverse_product(b, work, Out(r2[2], N / 2), upper);
        t4.inverse_product(c, work, Out(r2[3], N / 2), upper);
    }
    // Level T - 3: P_0 .. P_3 at N = m/4.
    {
        const std::size_t N = m / 4;
        u32 *const P0 = forms[0], *const P1 = forms[1], *const P2 = forms[2], *const P3 = forms[3];
        t8.forward(Span(r2[0], N), P0), t8.forward(Span(r2[1], N), P1);
        t8.forward(Span(r2[2], N), P2), t8.forward(Span(r2[3], N), P3);
        u32* const* C = cols.C8;
        const ntt::Output upper{.offset = N / 2};
        std::copy_n(r2[0] + N / 2, N / 2, r3[0]);
        const ntt::Term t1[1] = {{P0, C[1]}}, t2[2] = {{P0, C[2]}, {P1, nullptr}}, t3[2] = {{P0, C[3]}, {P1, C[1]}},
                        t4s[3] = {{P0, C[4]}, {P1, C[2]}, {P2, nullptr}}, t5[3] = {{P0, C[5]}, {P1, C[3]}, {P2, C[1]}},
                        t6[4] = {{P0, C[6]}, {P1, C[4]}, {P2, C[2]}, {P3, nullptr}},
                        t7[4] = {{P0, C[7]}, {P1, C[5]}, {P2, C[3]}, {P3, C[1]}};
        t8.inverse_product(t1, work, Out(r3[1], N / 2), upper);
        t8.inverse_product(t2, work, Out(r3[2], N / 2), upper);
        t8.inverse_product(t3, work, Out(r3[3], N / 2), upper);
        t8.inverse_product(t4s, work, Out(r3[4], N / 2), upper);
        t8.inverse_product(t5, work, Out(r3[5], N / 2), upper);
        t8.inverse_product(t6, work, Out(r3[6], N / 2), upper);
        t8.inverse_product(t7, work, Out(r3[7], N / 2), upper);
        columns8_to(r3, N / 2, p);
    }
}

// Runs on Native's lanes, or one pair at a time for runs of fewer pairs (short transforms).
template <class Op>
[[gnu::always_inline]] inline void on_pairs(std::size_t pairs, Op op) {
    if (pairs % Native::W == 0) return op(std::type_identity<Native>{});
    op(std::type_identity<Pairs<simd::Scalar>>{});
}

// Hooks for ntt::Transform's hooked ops (leaves [first, first + count) of a transform of 4m).
inline constexpr auto kNoHook = [](u32*, std::size_t, std::size_t) {};

// Q's forward at 4m: the Graeffe of each run into G (V's leaf form at 2m).
struct GraeffeHook {
    u32* G;
    const PairTables& t;
    [[gnu::always_inline]] void operator()(u32* leaves, std::size_t first, std::size_t count) const {
        on_pairs(count / 2, [&]<class N>(std::type_identity<N>) { N::graeffe_run(leaves, G + 4 * first, first / 2, count / 2, t); });
    }
};

// Q's forward at 4m with P's leaf form FP stored: V and W of each run into GV and GW.
struct ProjectionHook {
    const u32* FP;
    u32 *GV, *GW;
    const PairTables& t;
    [[gnu::always_inline]] void operator()(u32* leaves, std::size_t first, std::size_t count) const {
        on_pairs(count / 2, [&]<class N>(std::type_identity<N>) {
            N::projection_run(leaves, FP + 8 * first, GV + 4 * first, GW + 4 * first, first / 2, count / 2, t);
        });
    }
};

// R's inverse at 4m: its leaves from P^'s leaf form FP at 2m and the stored leaf form FQ of Q_s.
struct ProductHook {
    const u32 *FP, *FQ;
    const PairTables& t;
    [[gnu::always_inline]] void operator()(u32* leaves, std::size_t first, std::size_t count) const {
        on_pairs(count / 2, [&]<class N>(std::type_identity<N>) {
            const u32* const fp[1] = {FP + 4 * first};
            const u32* const fq[1] = {FQ + 8 * first};
            N::template product_run<1>(fp, fq, leaves, first / 2, count / 2, t);
        });
    }
};

// The same for the sum of two products (level 1).
struct ProductHook2 {
    const u32 *FPa, *FQa, *FPb, *FQb;
    const PairTables& t;
    [[gnu::always_inline]] void operator()(u32* leaves, std::size_t first, std::size_t count) const {
        on_pairs(count / 2, [&]<class N>(std::type_identity<N>) {
            const u32* const fp[2] = {FPa + 4 * first, FPb + 4 * first};
            const u32* const fq[2] = {FQa + 8 * first, FQb + 8 * first};
            N::template product_run<2>(fp, fq, leaves, first / 2, count / 2, t);
        });
    }
};

// Level 1's products in the forward of q2 at m, from the stored leaf forms F1 (q1) and FP (p1, or
// null): five outputs at m/2 (level1_run).
struct Level1Hook {
    const u32 *F1, *FP;
    u32* const* out;
    const PairTables& t;
    [[gnu::always_inline]] void operator()(u32* leaves, std::size_t first, std::size_t count) const {
        on_pairs(count / 2, [&]<class N>(std::type_identity<N>) {
            u32* o[5];  // out[i] may be null (no p1): no offset on it
            for (int i = 0; i < 5; ++i) o[i] = out[i] ? out[i] + 4 * first : nullptr;
            N::level1_run(F1 + 8 * first, leaves, FP ? FP + 8 * first : nullptr, o, first / 2, count / 2, t);
        });
    }
};

// Level 0, one-dimensional (Q_0 = 1 - y g(x), lc-opt's): Q_1(x^2, y) = Q_0(x) Q_0(-x) = 1 - y (g(x) +
// g(-x)) + y^2 g(x) g(-x), the last a Graeffe of g at 2m (no wrap: deg g < m). g into q[0, n) with
// g[0] = 0, F0 = g's leaf form at 2m (kept: compose's up step), G (m words) = the Graeffe's leaf form
// at m; then Q_1 at stride m in q (rows 0 .. 2, x < m/2). Returns the words of q to transform.
inline std::size_t first_level(Span g, std::size_t n, std::size_t m, u32* q, u32* F0, u32* G, const PairTables& t) {
    const int T = std::countr_zero(m);
    const std::size_t k = std::min(n, g.size());
    q[0] = 0;
    if (k > 1) std::copy_n(g.data() + 1, k - 1, q + 1);
    std::fill(q + std::max<std::size_t>(k, 1), q + m, 0u);
    transform(T + 1).forward_hooked(Span(q, n), F0, GraeffeHook{G, t});
    transform(T).inverse_hooked(kNoHook, G, Out(q + 2 * m, m / 2));
    std::fill_n(q + 2 * m + m / 2, m / 2, 0u);
    // Row 1: -2 g_even, from q[0, m) (row 1 starts at m).
    const u32* __restrict even = q;
    u32* __restrict row = q + m;
    for (std::size_t i = 0; i < m / 2; ++i) {
        const u32 x = even[2 * i] + even[2 * i];
        const u32 y = x >= kP ? x - kP : x;
        row[i] = y ? kP - y : 0;
    }
    std::fill_n(q + m + m / 2, m / 2, 0u);
    q[0] = 1;
    std::fill_n(q + 1, m - 1, 0u);
    return 2 * m + m / 2;
}

// Level 1, one-dimensional (lc-opt's first level): Q_1 = 1 + y q1(x) + y^2 q2(x) at stride m in q
// (x < m/2), Q_2(x^2) = Q_1(x) Q_1(-x): rows 1 .. 4 are 2 q1e, 2 q2e + G(q1), 2 E(q1, q2), G(q2) (in u
// = x^2, u < m/4; ae the even part of a, G and E as level1_run), at stride m/2 in q. F1, F2: the
// leaf forms of q1, q2 at m (kept by compose); outs: 5 arrays of m/2 words; temp: 5 of m/4.
// Power projection (p not null): P_1 = p0 + y p1 at stride m in p, p0 = [e odd] x^d, d = (e - 1) /
// 2; P_2 (stride m/2, x < m/4) = the odd part over x of P_1(x) Q_1(-x): rows O(p0), O(p1) + O(p0
// q1(-x)), O(p1 q1(-x)) + O(p0 q2(-x)), O(p1 q2(-x)), where O(x^d q(-x))[i] = (-1)^j q_j, j = 2i + 1 -
// d. FP: p1's leaf form at m. Returns the words of q for level 2's forward.
inline std::size_t second_level(u32* q, u32* p, std::size_t m, std::size_t e, u32* F1, u32* F2, u32* FP, u32* const* outs,
                                u32* const* temp, const PairTables& t) {
    const int T = std::countr_zero(m);
    const ntt::Transform &t1 = transform(T), &th = transform(T - 1);
    const std::size_t h = m / 2, quarter = m / 4;
    t1.forward(Span(q + m, h), F1);
    if (p) t1.forward(Span(p + m, h), FP);
    t1.forward_hooked(Span(q + 2 * m, h), F2, Level1Hook{F1, p ? FP : nullptr, outs, t});
    for (int k = 0; k < (p ? 5 : 3); ++k) th.inverse_hooked(kNoHook, outs[k], Out(temp[k], quarter));
    const auto add = [](u32 x, u32 y) { return x + y >= kP ? x + y - kP : x + y; };
    if (p) {
        const u32 *p1 = p + m, *q1 = q + m, *q2 = q + 2 * m;
        const bool odd = e % 2 && (e - 1) / 2 % 2;  // O(p0) = 1 at i = (d - 1) / 2
        const std::size_t d = e % 2 ? (e - 1) / 2 : 0;
        // O(x^d q(-x))[i]: q_j (-1)^j for j = 2i + 1 - d in [0, m/2), if p0 = x^d.
        const auto shifted = [&](const u32* qk, std::size_t i) -> u32 {
            if (e % 2 == 0 || 2 * i + 1 < d || 2 * i + 1 - d >= h) return 0;
            const std::size_t j = 2 * i + 1 - d;
            return j % 2 && qk[j] ? kP - qk[j] : qk[j];
        };
        for (std::size_t i = 0; i < quarter; ++i) p[h + i] = add(p1[2 * i + 1], shifted(q1, i));
        for (std::size_t i = 0; i < quarter; ++i) p[m + i] = add(temp[3][i], shifted(q2, i));
        std::copy_n(temp[4], quarter, p + 3 * h);
        std::fill_n(p, quarter, 0u);
        if (odd) p[(d - 1) / 2] = 1;
        for (std::size_t r = 0; r < 4; ++r) std::fill_n(p + r * h + quarter, quarter, 0u);
    }
    for (std::size_t i = 0; i < quarter; ++i) q[h + i] = twice(q[m + 2 * i]);
    for (std::size_t i = 0; i < quarter; ++i) q[m + i] = add(twice(q[2 * m + 2 * i]), temp[0][i]);
    for (std::size_t i = 0; i < quarter; ++i) q[m + h + i] = twice(temp[2][i]);
    std::copy_n(temp[1], quarter, q + 2 * m);
    q[0] = 1;
    std::fill_n(q + 1, quarter - 1, 0u);
    for (std::size_t r = 0; r < 4; ++r) std::fill_n(q + r * h + quarter, quarter, 0u);
    return 2 * m + quarter;
}

inline void compose_levels(Span f, Span g, Out h) {
    const std::size_t n = h.size(), m = length_for(n), pairs = m / 4;
    const int T = std::countr_zero(m);
    const bool columns = m >= kColumnsMin;
    const int generic = columns ? T - 3 : T - 1;  // levels s < generic are generic
    const ntt::Transform &t4 = transform(T + 2), &t2 = transform(T + 1), &t1 = transform(T);
    const PairTables& tables = pair_tables(pairs);
    const std::size_t words = std::size_t(generic) * footprint(4 * m) + 2 * footprint(4 * m) + 2 * footprint(2 * m) +
                              footprint(3 * m) + (columns ? Columns::words(m) : 0) + 16;
    Arena arena(thread_words(words), words);
    std::vector<u32*> levels(static_cast<std::size_t>(generic));
    for (auto& F : levels) F = arena.take(4 * m);
    u32 *const q = arena.take(3 * m), *const G = arena.take(2 * m), *const p = arena.take(2 * m);
    u32 *const FP = arena.take(4 * m), *const work = arena.take(4 * m);

    // Down: level 0 one-dimensional (g's leaf form at 2m kept in levels[0]); the leaf forms of Q_1 ..
    // Q_(generic-1) at 4m (row stride 2L, x < L: column bit L zero), the Graeffe in their forwards;
    // V's inverse in place in G (x < L/2 used). Then the column levels, or (short transforms)
    // Q_(T-1) = 1 + x q(y) at stride 4. c1: Q_(T-1)'s column -q (m/2 + 1 words).
    std::size_t size = first_level(g, n, m, q, levels[0], G, tables);
    // Level 1 one-dimensional from m = 128 on (its products have m/2 >= 2^kMinLog words).
    const int first = m >= 128 ? 2 : 1;
    u32 *const F1 = levels[1], *const F2 = levels[1] + footprint(m);
    if (first == 2) {
        u32* const outs[5] = {FP, FP + footprint(m / 2), FP + 2 * footprint(m / 2), nullptr, nullptr};
        u32* const temp[5] = {G, G + footprint(m / 4), G + 2 * footprint(m / 4), nullptr, nullptr};
        size = second_level(q, nullptr, m, 0, F1, F2, nullptr, outs, temp, tables);
    }
    for (int s = first; s < generic; ++s) {
        const std::size_t L = m >> s;
        t4.forward_hooked(Span(q, size), levels[std::size_t(s)], GraeffeHook{G, tables}, L);
        t2.inverse_hooked(kNoHook, G, Out(q, 2 * m), {}, L / 2);
        next_level(q, m, L);
        size = 2 * m + L / 2;
    }
    std::optional<Columns> cols;
    const u32* c1 = G;
    if (columns) {
        cols.emplace(arena, m);
        columns_down(q, m, *cols, work);
        c1 = cols->c2;
    } else {
        last_column(q, m, G);
    }
    t1.forward(Span(c1, m / 2 + 1), FP);
    // Up. Level T - 1: P^_T = f reversed (m words in y), R = P^ (1 - x q): P^_(T-1)'s columns are
    // r0 = f reversed and r1 = P^ c1 mod (y^m - 1), rows m/2 .. m - 1.
    u32* const A = q;  // q is free from here
    for (std::size_t b = 0; b < m; ++b) A[b] = at(f, m - 1 - b);
    u32 *const r0 = A + m / 2, *const r1 = G;
    t1.cyclic_product(Span(A, m), FP, work, Out(r1, m / 2), {.offset = m / 2});
    if (columns) {
        // The leaf forms of the columns P_j in FP (4 footprint(m/2) <= 4m words).
        u32* const r[2] = {r0, r1};
        u32* const forms[4] = {FP, FP + footprint(m / 2), FP + 2 * footprint(m / 2), FP + 3 * footprint(m / 2)};
        columns_up(*cols, m, r, forms, work, p);
    } else {
        std::fill_n(p, 2 * m, 0u);
        for (std::size_t b = 0; b < m / 2; ++b) p[4 * b] = r0[b], p[4 * b + 1] = r1[b];
    }
    // Levels generic - 1 .. 2: R = P^(z^2) Q_s(-z), rows Y .. 2Y - 1, x below L (column bit L
    // unused); P^ at stride L, x < L/2 (column bit L/2 zero).
    for (int s = generic - 1; s >= first; --s) {
        const std::size_t L = m >> s, Y = std::size_t(1) << s;
        t2.forward_hooked(Span(p, 2 * m), FP, kNoHook, L / 2);
        t4.inverse_hooked(ProductHook{FP, levels[std::size_t(s)], tables}, work, Out(p, 2 * m), {.offset = 2 * m}, L);
        for (std::size_t j = 0; j < Y; ++j) std::fill_n(p + 2 * L * j + L, L, 0u);
    }
    // Level 1, one-dimensional: P^_2 = P_0 .. P_3 at stride m/2 (x < m/4); R's rows 2, 3 = P_2(x^2) +
    // P_1(x^2) q1(-x) + P_0(x^2) q2(-x) and P_3(x^2) + P_2(x^2) q1(-x) + P_1(x^2) q2(-x), products of
    // m (no wrap), x < m/2 kept: P^_1's rows p0, p1.
    u32 *const p0 = first == 2 ? G : p, *const p1 = first == 2 ? G + footprint(m / 2) : p + m;
    if (first == 2) {
        const std::size_t half = m / 2, quarter = m / 4;
        const ntt::Transform& th = transform(T - 1);
        u32* const P[3] = {FP, FP + footprint(half), FP + 2 * footprint(half)};
        for (int j = 0; j < 3; ++j) th.forward(Span(p + j * half, quarter), P[j]);
        t1.inverse_hooked(ProductHook2{P[1], F1, P[0], F2, tables}, work, Out(p0, half));
        t1.inverse_hooked(ProductHook2{P[2], F1, P[1], F2, tables}, work, Out(p1, half));
        for (std::size_t i = 0; i < quarter; ++i) {
            const u32 x = p0[2 * i] + p[2 * half + i], y = p1[2 * i] + p[3 * half + i];
            p0[2 * i] = x >= kP ? x - kP : x, p1[2 * i] = y >= kP ? y - kP : y;
        }
    }
    // Level 0, one-dimensional: P^_1 = p0(u) + y p1(u); h = R's row 1 = p1(x^2) - p0(x^2) g(-x), a
    // product of 2m (no wrap) with p0's leaf form at m spread.
    t1.forward(Span(p0, m / 2), FP);
    t2.inverse_hooked(ProductHook{FP, levels[0], tables}, work, h, {.negate = true});
    for (std::size_t i = 0; 2 * i < n; ++i) {
        const u32 x = h[2 * i] + p1[i];
        h[2 * i] = x >= kP ? x - kP : x;
    }
}

// Power projection's P columns at the column levels: p8 (P_(T-3), m/8 rows each) and P8 (their
// leaf forms at m/4), p4 (P_(T-2), m/4 rows) and P4 (at m/2), p2 (P_(T-1), m/2 rows); tmp: m/4 words.
struct ProjectionColumns {
    ProjectionColumns(Arena& arena, std::size_t m) {
        for (auto& x : p8) x = arena.take(m / 8);
        for (auto& x : P8) x = arena.take(m / 4);
        for (auto& x : p4) x = arena.take(m / 4);
        for (auto& x : P4) x = arena.take(m / 2);
        for (auto& x : p2) x = arena.take(m / 2);
        tmp = arena.take(m / 4);
    }
    static std::size_t words(std::size_t m) {
        return 8 * footprint(m / 8) + 7 * footprint(m / 4) + 4 * footprint(m / 4) + 3 * footprint(m / 2) +
               2 * footprint(m / 2) + footprint(m / 4);
    }
    u32 *p8[8], *P8[7], *p4[4], *P4[3], *p2[2], *tmp;
};

// out (N words) = the inverse of the sum of the terms' products (up to 8; past 4 a second inverse
// into tmp, added) plus c (N/2 words).
inline void column_sum(const ntt::Transform& t, std::span<const ntt::Term> terms, u32* work, u32* out, u32* tmp,
                       const u32* c) {
    const std::size_t N = t.length();
    t.inverse_product(terms.first(std::min<std::size_t>(terms.size(), 4)), work, Out(out, N));
    if (terms.size() > 4) {
        t.inverse_product(terms.subspan(4), work, Out(tmp, N));
        ntt::detail::add_words(out, tmp, N);
    }
    ntt::detail::add_words(out, c, N / 2);
}

// Power projection through the column levels: from P_(T-3) and Q_(T-3) at stride 16 in p and q,
// a[k] (k < a.size()). work: m/2 + kPadding words; prod: a.size() words.
inline void projection_columns(const u32* p, const u32* q, std::size_t m, Columns& cols, ProjectionColumns& pc,
                               u32* work, u32* prod, Out a) {
    const int T = std::countr_zero(m);
    const ntt::Transform &t8 = transform(T - 2), &t4 = transform(T - 1);
    columns_down(q, m, cols, work);
    // Level T - 3 (N = m/4): P_(T-2)'s columns p'_j = sum_(a+b=2j+1) p_a c_b.
    columns8_from(p, m / 8, pc.p8, 0, false);
    for (int c = 0; c < 7; ++c) t8.forward(Span(pc.p8[c], m / 8), pc.P8[c]);
    {
        u32* const* P = pc.P8;
        u32* const* C = cols.C8;
        const ntt::Term j0[1] = {{P[0], C[1]}}, j1[3] = {{P[2], C[1]}, {P[1], C[2]}, {P[0], C[3]}},
                        j2[5] = {{P[4], C[1]}, {P[3], C[2]}, {P[2], C[3]}, {P[1], C[4]}, {P[0], C[5]}},
                        j3[7] = {{P[6], C[1]}, {P[5], C[2]}, {P[4], C[3]}, {P[3], C[4]}, {P[2], C[5]}, {P[1], C[6]},
                                 {P[0], C[7]}};
        column_sum(t8, j0, work, pc.p4[0], pc.tmp, pc.p8[1]);
        column_sum(t8, j1, work, pc.p4[1], pc.tmp, pc.p8[3]);
        column_sum(t8, j2, work, pc.p4[2], pc.tmp, pc.p8[5]);
        column_sum(t8, j3, work, pc.p4[3], pc.tmp, pc.p8[7]);
    }
    // Level T - 2 (N = m/2).
    for (int c = 0; c < 3; ++c) t4.forward(Span(pc.p4[c], m / 4), pc.P4[c]);
    {
        u32* const* P = pc.P4;
        u32* const* C = cols.C4;
        const ntt::Term j0[1] = {{P[0], C[1]}}, j1[3] = {{P[2], C[1]}, {P[1], C[2]}, {P[0], C[3]}};
        column_sum(t4, j0, work, pc.p2[0], pc.tmp, pc.p4[1]);
        column_sum(t4, j1, work, pc.p2[1], pc.tmp, pc.p4[3]);
    }
    // Level T - 1: a = p1 + p0 c1.
    const std::size_t n = a.size();
    ntt::multiply(Span(pc.p2[0], m / 2), Span(cols.c2, m / 2 + 1), Out(prod, n));
    for (std::size_t k = 0; k < n; ++k) {
        const u32 x = (k < m / 2 ? pc.p2[1][k] : 0) + prod[k];
        a[k] = x >= kP ? x - kP : x;
    }
}

inline void projection_levels(Span g, Out a) {
    const std::size_t n = a.size(), m = length_for(n), pairs = m / 4;
    const int T = std::countr_zero(m);
    const bool columns = m >= kColumnsMin;
    const int generic = columns ? T - 3 : T - 1;  // levels s < generic are generic
    const ntt::Transform &t4 = transform(T + 2), &t2 = transform(T + 1);
    const PairTables& tables = pair_tables(pairs);
    const std::size_t level1 = 3 * footprint(m) + 5 * footprint(m / 2) + 5 * footprint(m / 4);
    const std::size_t words = 2 * footprint(4 * m) + 2 * footprint(2 * m) + footprint(3 * m) + footprint(2 * m) + level1 +
                              (columns ? Columns::words(m) + ProjectionColumns::words(m) : 0) + 16;
    Arena arena(thread_words(words), words);
    u32 *const FQ = arena.take(4 * m), *const FP = arena.take(4 * m), *const GV = arena.take(2 * m);
    u32 *const GW = arena.take(2 * m), *const q = arena.take(3 * m), *const p = arena.take(2 * m);

    // Level 0, one-dimensional: Q_1 as compose's; P_0 = x^e, e = m - n, so P_1 (stride m, rows 0,
    // 1, x < m/2) = [e odd] u^((e-1)/2) - y W, W[i] = (-1)^j g_j for j = 2i + 1 - e.
    std::size_t size = first_level(g, n, m, q, FQ, GV, tables);
    std::fill_n(p, 2 * m, 0u);
    const std::size_t e = m - n;
    if (e % 2) p[(e - 1) / 2] = 1;
    for (std::size_t j = 1; j < n; ++j)
        if ((e + j) % 2) {
            const u32 x = at(g, j);
            p[m + (e + j - 1) / 2] = j % 2 || !x ? x : kP - x;  // -(-1)^j g_j
        }
    const int first = m >= 128 ? 2 : 1;  // level 1 one-dimensional from m = 128 on, as compose's
    if (first == 2) {
        u32 *const F1 = arena.take(m), *const F2 = arena.take(m), *const FP1 = arena.take(m);
        u32 *outs[5], *temp[5];
        for (auto& x : outs) x = arena.take(m / 2);
        for (auto& x : temp) x = arena.take(m / 4);
        size = second_level(q, p, m, e, F1, F2, FP1, outs, temp, tables);
    }
    for (int s = first; s < generic; ++s) {
        const std::size_t L = m >> s, Y = std::size_t(1) << s;
        t4.forward_hooked(Span(p, 2 * m), FP, kNoHook, L);
        t4.forward_hooked(Span(q, size), FQ, ProjectionHook{FP, GV, GW, tables}, L);
        t2.inverse_hooked(kNoHook, GW, Out(p, 2 * m), {}, L / 2);
        for (std::size_t j = 0; j < 2 * Y; ++j) std::fill_n(p + L * j + L / 2, L / 2, 0u);
        t2.inverse_hooked(kNoHook, GV, Out(q, 2 * m), {}, L / 2);
        next_level(q, m, L);
        size = 2 * m + L / 2;
    }
    if (columns) {
        Columns cols(arena, m);
        ProjectionColumns pc(arena, m);
        return projection_columns(p, q, m, cols, pc, FQ, FP, a);
    }
    // Level T - 1, one-dimensional: P = p0(y) + x p1(y) (stride 4, rows below m/2), Q = 1 + x q(y);
    // a = p1 + p0 c1, c1 = -q (degree below m).
    u32 *const c1 = GV, *const p0 = FQ, *const prod = FP;
    last_column(q, m, c1);
    for (std::size_t b = 0; b < m / 2; ++b) p0[b] = p[4 * b];
    ntt::multiply(Span(p0, m / 2), Span(c1, m / 2 + 1), Out(prod, n));
    for (std::size_t k = 0; k < n; ++k) {
        const u32 x = (k < m / 2 ? p[4 * k + 1] : 0) + prod[k];
        a[k] = x >= kP ? x - kP : x;
    }
}

inline void check_length(std::size_t n, const char* what) {
    if (n > kMaxSize) throw std::length_error(what);
}

}  // namespace detail

// out = f(g) mod x^n, n = out.size(); g[0] counts as 0.
inline void compose(std::span<const u32> f, std::span<const u32> g, std::span<u32> out) {
    const std::size_t n = out.size();
    detail::check_length(n, "fps::composition::compose: n past kMaxSize");
    if (n == 0) return;
    f = f.first(std::min(f.size(), n)), g = g.first(std::min(g.size(), n));
    if (n <= detail::kComposeBase) return detail::compose_direct(f, g, out);
    detail::compose_levels(f, g, out);
}

// out[k] = [x^(n-1)] g^k for k < n = out.size(); g[0] counts as 0.
inline void power_projection(std::span<const u32> g, std::span<u32> out) {
    const std::size_t n = out.size();
    detail::check_length(n, "fps::composition::power_projection: n past kMaxSize");
    if (n == 0) return;
    g = g.first(std::min(g.size(), n));
    if (n <= detail::kProjectionBase) return detail::projection_direct(g, out);
    detail::projection_levels(g, out);
}

// out = g with f(g) = g(f) = x mod x^n, n = out.size(); f[0] counts as 0, f[1] != 0, n <= P.
inline void compositional_inverse(std::span<const u32> f, std::span<u32> out) {
    using namespace detail;
    const std::size_t n = out.size();
    check_length(n, "fps::composition::compositional_inverse: n past kMaxSize");
    if (n == 0) return;
    out[0] = 0;
    if (n == 1) return;
    const u32 inverse1 = power(at(f, 1), kP - 2);
    if (n == 2) {
        out[1] = inverse1;
        return;
    }
    // h[j] = a[N - j] / (N - j) for j < N, a = power_projection(f); then g / x = exp(-log(h /
    // h[0]) / N) / f[1] mod x^N.
    const std::size_t N = n - 1;
    std::vector<u32> a(n), inv(n), h(N);
    power_projection(f.first(std::min(f.size(), n)), a);
    inv[1] = 1;
    for (std::size_t i = 2; i < n; ++i) inv[i] = mul(kP - kP / u32(i), inv[kP % i]);
    for (std::size_t j = 0; j < N; ++j) h[j] = mul(a[N - j], inv[N - j]);
    const u32 c = power(h[0], kP - 2), e = kP - inv[N];
    for (auto& x : h) x = mul(x, c);
    series::log(h, h);
    for (auto& x : h) x = mul(x, e);
    series::exp(h, out.subspan(1));
    for (std::size_t j = 1; j < n; ++j) out[j] = mul(out[j], inverse1);
}

}  // namespace fps::composition
