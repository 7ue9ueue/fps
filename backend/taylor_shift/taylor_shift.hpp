// Taylor shift modulo P = 998244353 (backend kernel): the coefficients of f(x + c). From lc-opt's
// problems/polynomial/polynomial_taylor_shift (issue #124), rewritten on backend/simd, ntt's
// in-place product and backend/factorials' lane grid, so every target runs it.
//
//   fps::taylor_shift::shift(f, c, out);   // out[k] = [x^k] f(x + c), k < out.size() <= f.size()
//
// f: n words < P; c: any word (taken mod P). out: t <= n words (else std::invalid_argument), any
// memory; it may overlap f (f is read before out is written; an overlap costs one more store per
// coefficient). n > kMaxSize = 2^24 throws std::length_error before out is written.
//
// With A_i = f_i i! and E_j = c^j / j!, k! out[k] = sum_(j >= 0) A_(k+j) E_j = A_k + h_k, where
// h_k is coefficient L/2 + k of the cyclic product of A (words [0, n)) and E' (E_j at word L/2 - j,
// 1 <= j < n), L = 2^lg >= 2n: both factors in the lower half, the result the upper half in
// natural order (lc-opt's layout). E_0 = 1 would sit at L/2, outside the half; it comes back as
// A_k / k! = f_k.
//   1. weights: A and E' by product chains in W = K S::lanes lanes (select.hpp) on
//      backend/factorials' lane grid (Chains::lanes: lane s on positions [a_s, a_s + C), with a_s!
//      and 1 / (a_s + C - 1)!): i! up from a_s, c^j / j! down from a_s + C - 1, each step one
//      modular product by a multiplier in arithmetic progression (one add). Every S::lanes steps
//      an in-register transpose turns the lanes' terms into vectors of consecutive positions.
//   2. h = the upper half of A E' mod (x^L - 1): ntt's product in place (ntt::Convolution's path
//      and lc-opt's: both first levels in place on the half-zero factors, the subtrees of both
//      factors transformed, multiplied and inverted in cache), A and E' written into its arrays.
//   3. output: out[k] = h_k / k! + f_k, 1 / k! by a chain down each lane; if out overlaps f,
//      (h_k + A_k) / k! with A from a copy the weights also store.
// n <= kSmall (select.hpp): Horner's rule, O(n^2), in vectors. c = 0: a copy.
//
// Same run against lc-opt on Zen 3: 0.99-1.01 of its time at n = 2^15, 2^17, 2^19 and between,
// 0.52 at 2^16 and 2^18 (lc-opt runs even lg only), 0.31-0.93 below 2^14. Memory: one region per
// thread (ntt's, kept across calls), 2 L + n words and padding; the product's twiddle tables are
// ntt's (memory.hpp). Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

#include "backend/factorials/factorials.hpp"
#include "backend/factorials/transpose.hpp"
#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"
#include "backend/taylor_shift/select.hpp"

namespace fps::taylor_shift {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 kP = ntt::kModulus;
inline constexpr std::size_t kMaxSize = std::size_t(1) << (ntt::kMaxLog - 1);  // L = 2^lg >= 2n

namespace detail {

inline constexpr u32 kR = u32((u64(1) << 32) % kP);  // 2^32 mod P
inline constexpr u32 kR2 = u32(u64(kR) * kR % kP);   // 2^64 mod P

inline u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kP); }
inline u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

// out[k] = [x^k] f(x + c) for k < t <= n <= kMax, by Horner's rule from the top: g <- g (x + c)
// + f_i, that is g_k <- g_(k-1) + c g_k with g_(-1) = f_i, one pass over g from its top vector down
// (each vector's loads come before its store, and the vector below reads only words below it).
template <class S, std::size_t kMax = kSmall>
void horner(const u32* f, std::size_t n, u32 c, u32* out, std::size_t t) {
    using M = modarith::Ops<S, kP>;
    using V = typename S::V;
    constexpr int kL = S::lanes;
    alignas(64) u32 buffer[kMax + 2 * kL + 1] = {};  // g_(-1) at buffer[kL - 1], g_k at g[k]
    u32* const g = buffer + kL;
    const V w = S::broadcast(c), q = S::broadcast(M::shoup_quotient(c));
    for (std::size_t i = n; i-- > 0;) {
        g[-1] = f[i];
        for (std::size_t k = (n - 1 - i) / kL * kL + kL; k > 0;) {  // g has n - 1 - i words before the step
            k -= kL;
            const V x = S::load(g + k), below = S::load(g + k - 1);
            S::store(g + k, M::add(below, M::template reduce<kP>(M::shoup(x, w, q))));
        }
    }
    std::copy_n(g, t, out);
}

// The scans: product chains in W = K S::lanes lanes on backend/factorials' lane grid. Lane s covers
// positions [a_s, a_s + C), a_s = min(h + s C, n - C): the lanes past n - C all cover the last C
// positions and store the same words twice, so no lane reads or writes past n. Every kL steps the
// lanes' records form a block; an in-register transpose turns it into vectors of kL consecutive
// positions. The chains of block b + 1 are interleaved with the stores of block b (lc-opt's order).
// The up chain of i! and the down chain of 1 / k! (multipliers i + 1 and k, positions < 2^24) step
// as kStep says (factorials::Step): montgomery, x <- montgomery(x, m) with m = base + u in [0, 2P),
// u = j 2^32 mod P shared by every lane (one modular add per step); or shoup, x <- x m -
// mulhi(x, q) P with m = base +- j plain and q = floor(base 2^32 / P) +- floor(j 2^32 / P) (down:
// minus 1 more), at most one below floor(m 2^32 / P), shared floor(j 2^32 / P) = -(j 2^32 mod P)
// P^-1 mod 2^32 (one mul_lo per step); with x < 2^31 the result stays in [0, 2P). The chain of
// c^j / j! (multipliers j c^-1 mod P, which wrap) is always montgomery. x in [0, 2P) throughout.
template <class S, int K, factorials::Step kStep = factorials::Step::montgomery>
struct Scan {
    using Chains = factorials::Chains<S, K, kP>;
    using M = modarith::Ops<S, kP>;
    using V = typename S::V;
    using Lanes = typename Chains::Lanes;
    static constexpr int kL = S::lanes, kW = K * kL;
    static constexpr std::size_t kMinSize = Chains::kMinSize;
    // The weights' lanes start at position 1 (lc-opt's E lanes run over (s C, (s + 1) C]): E_j at
    // L/2 - j is then stored aligned. One-lane structs start at 0.
    static constexpr std::size_t kHead = kL > 1 ? 1 : 0;

    struct Grid {
        std::size_t c, h;
        Lanes start, first, down;  // a_s, (a_s)!, 1 / (a_s + c - 1)!
    };
    static Grid grid(std::size_t n, std::size_t h) {
        Grid g;
        g.h = h;
        g.c = Chains::lanes(n, h, g.start, g.first, g.down);
        return g;
    }

    template <class F>
    [[gnu::always_inline]] static void unrolled(F&& f) {
        factorials::detail::unrolled<K>(f);
    }
    // x m - mulhi(x, q) P for x < 2^31, m < P, q at most one below floor(m 2^32 / P): [0, 2P).
    static V shoup(V x, V m, V q) { return S::mul_sub(S::mul_lo(x, m), S::mulhi(x, q), S::broadcast(kP)); }

    // a[i] = f[i] i! for g.h <= i < n, and E[-j] = c^j / j! for g.h <= j < n (E: word L/2 of E'),
    // c != 0. The up chain x = i! 2^32 (multipliers (i + 1) 2^32 or i + 1), a = montgomery(f, x);
    // the down chain y = c^j / j! (multipliers j c^-1 2^32), from c^e / e! at the lane's last
    // position e. a: the product's array (overwritten by it); kCopy: A[i] = a[i] too (the copy the
    // output reads when out overlaps f).
    template <bool kCopy>
    [[gnu::flatten]] static void weights(const u32* f, std::size_t n, u32 c, const Grid& g, u32* A, u32* a, u32* E) {
        const std::size_t C = g.c;
        alignas(64) Lanes power_e;  // c^(a_s + C - 1): along the grid by c^C, then c^(n - 1)
        {
            const u32 step = power(c, C), last = power(c, n - 1);
            u32 x = power(c, g.h + C - 1);
            for (int s = 0; s < kW; ++s, x = mul(x, step)) power_e[s] = g.start[s] == g.h + s * C ? x : last;
        }
        const u32 inverse_c = Chains::invert(c);
        // montgomery(j, c^-1 2^64) = j c^-1 2^32
        const V r2 = S::broadcast(kR2), ic = S::broadcast(mul(inverse_c, kR2)), p = S::broadcast(kP);
        V up[K], down[K], up_base[K], up_q[K], down_base[K];
        unrolled([&](auto v) {
            const V a = S::load(g.start.data() + v * kL);
            up[v] = M::template reduce<kP>(M::montgomery(S::load(g.first.data() + v * kL), r2));
            const V next = S::add(a, S::broadcast(1)), next_r = M::template reduce<kP>(M::montgomery(next, r2));
            if constexpr (kStep == factorials::Step::shoup) up_base[v] = next, up_q[v] = S::mul_lo(next_r, S::broadcast(0u - M::kPinv));
            else up_base[v] = next_r;
            const V e = S::add(a, S::broadcast(u32(C - 1)));
            const V y = M::montgomery(S::load(power_e.data() + v * kL), S::load(g.down.data() + v * kL));
            down[v] = M::template reduce<kP>(M::montgomery(y, r2));
            down_base[v] = S::add(M::template reduce<kP>(M::montgomery(e, ic)), p);  // m = base - u in [1, 2P)
        });
        const u32* in[kW];
        u32* a_out[kW];
        u32* e_out[kW];
        u32* p_out[kW];
        for (int s = 0; s < kW; ++s) {
            in[s] = f + g.start[s], a_out[s] = kCopy ? A + g.start[s] : nullptr, p_out[s] = a + g.start[s];
            e_out[s] = E - (g.start[s] + C - 1);
        }
        V up_block[2][K][kL], down_block[2][K][kL];
        V u = S::broadcast(0), w = S::broadcast(0);  // u = j 2^32, w = j c^-1 2^32 after j steps
        V j = S::broadcast(0), qj = S::broadcast(0);  // shoup: j and floor(j 2^32 / P)
        const V r = S::broadcast(kR), d = S::broadcast(mul(inverse_c, kR)), neg_pinv = S::broadcast(0u - M::kPinv);
        const auto advance = [&](int b, auto t) {
            unrolled([&](auto v) {
                up_block[b][v][t] = up[v];
                down_block[b][v][t] = down[v];
                if constexpr (kStep == factorials::Step::shoup) up[v] = shoup(up[v], S::add(up_base[v], j), S::add(up_q[v], qj));
                else up[v] = M::montgomery(up[v], S::add(up_base[v], u));
                down[v] = M::montgomery(down[v], S::sub(down_base[v], w));
            });
            u = M::add(u, r), w = M::add(w, d);
            if constexpr (kStep == factorials::Step::shoup) j = S::add(j, S::broadcast(1)), qj = S::mul_lo(u, neg_pinv);
        };
        // Step t of a block stores lanes t K .. t K + K - 1; a vector is transposed at its first lane.
        // The down chain's block is in falling positions: stored at E - j, rising addresses.
        const auto emit = [&](int b, std::size_t i, auto t) {
            unrolled([&](auto l) {
                constexpr int e = decltype(t)::value * K + decltype(l)::value, v = e / kL, lane = e % kL;
                if constexpr (lane == 0) factorials::transpose(up_block[b][v]), factorials::transpose(down_block[b][v]);
                const V x = M::template reduce<kP>(M::montgomery(S::load(in[e] + i), up_block[b][v][lane]));
                if constexpr (kCopy) S::store(a_out[e] + i, x);
                S::store(p_out[e] + i, x);
                S::store(e_out[e] + i, M::template reduce<kP>(down_block[b][v][lane]));
            });
        };
        factorials::detail::unrolled<kL>([&](auto t) { advance(0, t); });
        std::size_t i = 0;
        int cur = 0;
        for (; i + kL < C; i += kL, cur ^= 1)
            factorials::detail::unrolled<kL>([&](auto t) { advance(cur ^ 1, t), emit(cur, i, t); });
        factorials::detail::unrolled<kL>([&](auto t) { emit(cur, i, t); });
    }

    // out[k] = (h[k] + A[k]) / k! for g.h <= k < t (the grid's n is t), or with kF (A = f, out apart
    // from f) h[k] / k! + f[k]. The chain z = 2^32 / k! runs down from the lane's last position
    // (multipliers k 2^32 or k); the output is montgomery(h + A, z), or montgomery(h, z) + f.
    template <bool kF>
    [[gnu::flatten]] static void output(const u32* h, const u32* A, const Grid& g, u32* out) {
        const std::size_t C = g.c;
        const V r2 = S::broadcast(kR2), p = S::broadcast(kP), neg_pinv = S::broadcast(0u - M::kPinv);
        V z[K], base[K], q[K];
        unrolled([&](auto v) {
            const V e = S::add(S::load(g.start.data() + v * kL), S::broadcast(u32(C - 1)));
            const V e_r = M::template reduce<kP>(M::montgomery(e, r2));
            z[v] = M::template reduce<kP>(M::montgomery(S::load(g.down.data() + v * kL), r2));
            if constexpr (kStep == factorials::Step::shoup) base[v] = e, q[v] = S::sub(S::mul_lo(e_r, neg_pinv), S::broadcast(1));
            else base[v] = S::add(e_r, p);
        });
        std::size_t top[kW];  // the first position of the lane's last vector
        for (int s = 0; s < kW; ++s) top[s] = g.start[s] + C - kL;
        V block[2][K][kL];
        V u = S::broadcast(0), j = S::broadcast(0), qj = S::broadcast(0);
        const V r = S::broadcast(kR);
        const auto advance = [&](int b, auto t) {
            unrolled([&](auto v) {
                block[b][v][kL - 1 - decltype(t)::value] = z[v];
                if constexpr (kStep == factorials::Step::shoup) z[v] = shoup(z[v], S::sub(base[v], j), S::sub(q[v], qj));
                else z[v] = M::montgomery(z[v], S::sub(base[v], u));
            });
            u = M::add(u, r);
            if constexpr (kStep == factorials::Step::shoup) j = S::add(j, S::broadcast(1)), qj = S::mul_lo(u, neg_pinv);
        };
        const auto emit = [&](int b, std::size_t i, auto t) {
            unrolled([&](auto l) {
                constexpr int e = decltype(t)::value * K + decltype(l)::value, v = e / kL, lane = e % kL;
                if constexpr (lane == 0) factorials::transpose(block[b][v]);
                const std::size_t at = top[e] - i;
                if constexpr (kF) {
                    const V x = M::template reduce<kP>(M::montgomery(S::load(h + at), block[b][v][lane]));
                    S::store(out + at, M::add(x, S::load(A + at)));
                } else {
                    const V x = S::add(S::load(h + at), S::load(A + at));
                    S::store(out + at, M::template reduce<kP>(M::montgomery(x, block[b][v][lane])));
                }
            });
        };
        factorials::detail::unrolled<kL>([&](auto t) { advance(0, t); });
        std::size_t i = 0;
        int cur = 0;
        for (; i + kL < C; i += kL, cur ^= 1)
            factorials::detail::unrolled<kL>([&](auto t) { advance(cur ^ 1, t), emit(cur, i, t); });
        factorials::detail::unrolled<kL>([&](auto t) { emit(cur, i, t); });
    }
};

// Words after each array: ntt::kPadding read by the transforms, and the rest so that arrays of a
// power-of-two length sit 320 bytes apart modulo 4 KiB (as backend/series).
inline constexpr std::size_t kGap = 80;
constexpr std::size_t footprint(std::size_t words) { return (words + 15) / 16 * 16 + kGap; }

// The offset (bytes, a multiple of 64 below 4096) of a region's first array A such that A and the
// array `span` bytes after it are each 256 bytes or more from f's and out's offsets modulo 4 KiB
// (as ntt::detail::place: a pass between arrays 0 to 192 bytes apart modulo 4 KiB ran up to 1.7x
// slower, 4K aliasing).
inline std::size_t place(const void* f, const void* out, std::size_t span) {
    const auto offset = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 4096; };
    const auto apart = [](std::size_t x, std::size_t y) { return (x - y + 256) % 4096 >= 512; };
    for (std::size_t a = 0; a < 4096; a += 64)
        if (apart(a, offset(f)) && apart(a, offset(out)) && apart((a + span) % 4096, offset(f)) &&
            apart((a + span) % 4096, offset(out)))
            return a;
    return 0;  // not reached: the four conditions exclude at most 4 x 512 of the 4096 bytes
}

// The shift for kSmall < n <= kMaxSize, c != 0, 1 <= t <= n, on Scan<S, K, kStep>.
template <class S, int K, factorials::Step kStep = kScanStep>
void shift_lanes(const u32* f, std::size_t n, u32 c, u32* out, std::size_t t) {
    using Sc = Scan<S, K, kStep>;
    using Product = ntt::SelectedMultiply;
    static_assert(Sc::kMinSize <= kSmall + 1, "shift() runs the lanes from kSmall + 1 on");
    constexpr std::size_t kL = S::lanes;
    const int lg = std::max(ntt::kMinLog, int(std::bit_width(2 * n - 1)));
    const std::size_t len = std::size_t(1) << lg, half = len / 2;
    // A (the output's copy) at a 64-byte boundary minus (kL - 1) words: the lanes from position 1
    // store aligned. The product's arrays a (A, then a E' in place) and b (E', E_j at half - j:
    // aligned for the lanes' last positions a_s + C - 1 = (s + 1) C): 64-byte aligned, L + kPadding
    // words each, b right after a (ntt::Convolution's layout).
    const std::size_t a_words = footprint(n + kL), p_words = len + ntt::kPadding;
    const ntt::detail::Region region = ntt::detail::take_region((a_words + 2 * p_words) * sizeof(u32) + 4096);
    struct Guard {
        ntt::detail::Region r;
        ~Guard() { ntt::detail::give_region(r); }
    } guard{region};
    auto* base = reinterpret_cast<u32*>(region.start + place(f, out, a_words * sizeof(u32)));
    u32* const A = base + (Sc::kHead ? kL - Sc::kHead : 0);
    u32* const a = base + a_words;
    u32* const b = a + p_words;
    const typename Sc::Grid g = Sc::grid(n, Sc::kHead);
    // out apart from f: the output adds f_k = A_k / k! itself; else it reads A from the copy (the
    // lanes past t - C read positions another lane has written).
    const auto word = [](const u32* p) { return reinterpret_cast<std::uintptr_t>(p); };
    const bool apart = word(out + t) <= word(f) || word(f + n) <= word(out);
    if (Sc::kHead) A[0] = a[0] = f[0];
    std::fill(a + n, a + half, 0u);         // the product reads the lower halves only (n, half <= L/2)
    std::fill(b, b + (half - n + 1), 0u);  // E_j for n <= j <= half: zero
    if (apart) Sc::template weights<false>(f, n, c, g, A, a, b + half);
    else Sc::template weights<true>(f, n, c, g, A, a, b + half);
    const auto [roots, inverse_roots] = ntt::detail::tables<Product>(lg);
    Product::multiply(a, b, n, half, lg, roots, inverse_roots);  // a = A E' mod (x^L - 1)
    const u32* const h = a + half;
    // out[k] = (h_k + A_k) / k!: on the weights' grid if t = n, else on a grid of t positions.
    const auto run = [&](const typename Sc::Grid& o) {
        if (apart) Sc::template output<true>(h, f, o, out);
        else Sc::template output<false>(h, A, o, out);
    };
    if (t == n) {
        if (Sc::kHead) out[0] = (h[0] + A[0]) % kP;
        run(g);
    } else if (t >= Sc::kMinSize) {
        run(Sc::grid(t, 0));
    } else {
        u32 inverse = 1;  // (t - 1)!, then 1 / k! from the top
        for (std::size_t k = 2; k < t; ++k) inverse = mul(inverse, u32(k));
        inverse = Sc::Chains::invert(inverse);
        for (std::size_t k = t; k-- > 0;) {
            out[k] = apart ? (mul(h[k], inverse) + f[k]) % kP : mul((h[k] + A[k]) % kP, inverse);
            if (k) inverse = mul(inverse, u32(k));
        }
    }
}

}  // namespace detail

// out[k] = [x^k] f(x + c) for k < out.size() <= f.size() (above).
inline void shift(std::span<const u32> f, u32 c, std::span<u32> out) {
    const std::size_t n = f.size(), t = out.size();
    if (t > n) throw std::invalid_argument("fps::taylor_shift::shift: out longer than f");
    if (n > kMaxSize) throw std::length_error("fps::taylor_shift::shift: f longer than 2^24");
    if (t == 0) return;
    c %= kP;
    if (c == 0) {
        if (out.data() != f.data()) std::memmove(out.data(), f.data(), t * sizeof(u32));
        return;
    }
    if (n <= kSmall) return detail::horner<simd::Native>(f.data(), n, c, out.data(), t);
    if constexpr (kNarrowBelow > 0)
        if (n < kNarrowBelow) return detail::shift_lanes<ScanIsa, 1>(f.data(), n, c, out.data(), t);
    detail::shift_lanes<ScanIsa, kScanVectors>(f.data(), n, c, out.data(), t);
}

}  // namespace fps::taylor_shift
