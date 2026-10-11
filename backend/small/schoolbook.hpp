// Schoolbook products on a simd struct S, modulo P, 2^12 < P < 2^30 (variant of the small kernel).
//
//   Schoolbook<S, R, P>::run(a, n, b, m, out);  // out = a b: n + m - 1 words; n >= m >= 1
//
// Inputs below P, output in [0, P). out must not overlap a or b. No allocation: scratch is on the
// stack (about 9 KiB).
//
// Outputs go in blocks of R vectors (R S::lanes words), their sums held in registers. For each j,
// every vector r of the block adds a[k - j] b'[j] for its words k: one load of a at k0 + r W - j
// and a broadcast of b'[j], shared by the R vectors. b' = b 2^32 mod P (Montgomery form), so one
// Montgomery step at the end gives the canonical sum.
//   x86 (mul_even): two 64-bit-lane sums per vector, even and odd words. The odd words of the
//     window at k0 + r W - j are the even words of the window one word on, which the step before
//     loaded: 2 mul_even, 2 add64 and 1 load per vector and j.
//   Neon: two 64-bit-lane sums per vector, words 0-1 and 2-3 (umlal, umlal2).
//   Scalar: one 64-bit sum per word.
// Sums: products < (P - 1)^2; every Sums<P>::kFold products a fold keeps them below 2^64; the
// last step needs them below 3P 2^32 (Sums). Blocks whose loads of a would leave [0, n) read a
// zero-padded copy instead. b is taken in panels of at most kPanel words, so the scratch is fixed;
// panels after the first add into out.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "backend/modarith/modarith.hpp"
#include "backend/simd/simd.hpp"

namespace fps::small {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

// Bounds of 64-bit sums of products x y, x, y < P. fold(s) = h R1 + l for s = h 2^32 + l, with
// R1 = 2^32 mod P: congruent to s and below kFolded. Montgomery's last step (s + m P) / 2^32 is
// below 4P when s < 3P 2^32.
template <u32 P>
struct Sums {
    static_assert(P > (1u << 12) && P < (1u << 30));
    static constexpr u128 kProduct = u128(P - 1) * (P - 1);
    static constexpr u32 kR1 = u32((u64(1) << 32) % P);
    static constexpr u128 kFolded = (u128(1) << 32) * (kR1 + 1);
    static constexpr u128 kLast = u128(3) * P << 32;
    static constexpr u128 kCap = u128(1) << 20;  // small P allow more; no j range is that long
    // Products allowed after a fold or from zero, before the sum could pass 2^64.
    static constexpr std::size_t kFold = std::size_t(std::min(((u128(1) << 64) - kFolded) / kProduct, kCap));
    // Products allowed from zero, or after a fold, before the last step.
    static constexpr std::size_t kLastFromZero = std::size_t(std::min((kLast - 1) / kProduct, kCap));
    static constexpr std::size_t kLastFromFold = std::size_t(std::min((kLast - kFolded) / kProduct, kCap));
    static_assert(kLastFromFold >= 1 && kFold >= kLastFromZero);
};

namespace detail {

// The sums of one output vector, per family of simd struct. start(x, w): before the first step,
// whose window starts at w. step(x, w, b): add the window at w times b. fold(x): keep the sums
// small. finish(x): the sums times 2^-32, in [0, P).
template <class S, u32 P>
struct EvenOdd {  // x86: 64-bit lanes by mul_even
    using V = typename S::V;
    using M = modarith::Ops<S, P>;
    struct Acc {
        V even, odd, next;  // next: the window one word on; its even words are this one's odd words
    };

    [[gnu::always_inline]] static void start(Acc& x, const u32* w) {
        x.even = x.odd = V{};
        x.next = S::load(w + 1);
    }
    [[gnu::always_inline]] static void step(Acc& x, const u32* w, V b) {
        const V v = S::load(w);
        x.odd = S::add64(x.odd, S::mul_even(x.next, b));
        x.even = S::add64(x.even, S::mul_even(v, b));
        x.next = v;
    }
    [[gnu::always_inline]] static V fold(V s) {
        const V low = S::template blend<0xaaaa>(s, V{});
        return S::add64(S::mul_even(S::template shr64<32>(s), S::broadcast(Sums<P>::kR1)), low);
    }
    [[gnu::always_inline]] static void fold(Acc& x) { x.even = fold(x.even), x.odd = fold(x.odd); }
    [[gnu::always_inline]] static V finish(const Acc& x) {
        const V ninv = S::broadcast(0u - M::kPinv), p = S::broadcast(P);
        const V even = S::add64(x.even, S::mul_even(S::mul_even(x.even, ninv), p));
        const V odd = S::add64(x.odd, S::mul_even(S::mul_even(x.odd, ninv), p));
        return M::template reduce<P>(M::template reduce<2 * P>(S::join_high(even, odd)));
    }
};

template <class S, u32 P>
struct Halves {  // Neon: words 0-1 and 2-3 in 64-bit lanes
    using V = typename S::V;
    using W = typename S::W;
    using M = modarith::Ops<S, P>;
    struct Acc {
        W lo, hi;
    };

    [[gnu::always_inline]] static void start(Acc& x, const u32*) { x.lo = x.hi = W{}; }
    [[gnu::always_inline]] static void step(Acc& x, const u32* w, V b) {
        const V v = S::load(w);
        x.lo = S::mla_wide_lo(x.lo, v, b);
        x.hi = S::mla_wide_hi(x.hi, v, b);
    }
    [[gnu::always_inline]] static void fold(Acc& x) {
        const V low = S::lo_halves(x.lo, x.hi), high = S::hi_halves(x.lo, x.hi);
        const V one = S::broadcast(1), r1 = S::broadcast(Sums<P>::kR1);
        x.lo = S::mla_wide_lo(S::mul_wide_lo(low, one), high, r1);
        x.hi = S::mla_wide_hi(S::mul_wide_hi(low, one), high, r1);
    }
    [[gnu::always_inline]] static V finish(const Acc& x) {
        const V m = S::mul_lo(S::lo_halves(x.lo, x.hi), S::broadcast(0u - M::kPinv)), p = S::broadcast(P);
        const V r = S::hi_halves(S::mla_wide_lo(x.lo, m, p), S::mla_wide_hi(x.hi, m, p));
        return M::template reduce<P>(M::template reduce<2 * P>(r));
    }
};

template <class S, u32 P>
struct Single {  // one word per vector
    using V = typename S::V;
    using M = modarith::Ops<S, P>;
    struct Acc {
        u64 s;
    };

    static void start(Acc& x, const u32*) { x.s = 0; }
    static void step(Acc& x, const u32* w, V b) { x.s += u64(*w) * b; }
    static void fold(Acc& x) { x.s = (x.s >> 32) * Sums<P>::kR1 + u32(x.s); }
    static V finish(const Acc& x) {
        const u32 m = u32(x.s) * (0u - M::kPinv);
        const u32 r = u32((x.s + u64(m) * P) >> 32);
        return M::template reduce<P>(M::template reduce<2 * P>(r));
    }
};

template <class S>
concept HasMulEven = requires(typename S::V a) { S::mul_even(a, a); };
template <class S>
concept HasMulWide = requires(typename S::V a) { S::mul_wide_lo(a, a); };

template <class S, u32 P>
using Lanes = std::conditional_t<HasMulEven<S>, EvenOdd<S, P>,
                                 std::conditional_t<HasMulWide<S>, Halves<S, P>, Single<S, P>>>;

}  // namespace detail

template <class S, int R, u32 P>
class Schoolbook {
public:
    static constexpr std::size_t kWidth = S::lanes, kBlock = R * kWidth;
    static constexpr std::size_t kPanel = 1024;  // words of b per panel

    static void run(const u32* a, std::size_t n, const u32* b, std::size_t m, u32* out) {
        alignas(64) u32 scaled[kPanel];
        if (m > kPanel) std::fill(out + n + kPanel - 1, out + n + m - 1, 0u);
        for (std::size_t first = 0; first < m; first += kPanel) {
            const std::size_t count = std::min(kPanel, m - first);
            to_montgomery(b + first, count, scaled);
            panel(a, n, scaled, count, out + first, first > 0);
        }
    }

private:
    using L = detail::Lanes<S, P>;
    using M = modarith::Ops<S, P>;
    using V = typename S::V;
    using Acc = typename L::Acc;

    // The terms of one output vector: j in [lo, hi).
    struct Range {
        std::size_t lo, hi;
    };

    // out[k] (+)= sum over j of a[k - j] b[j], k < n + m - 1, m <= kPanel. Blocks from k0 = m - 1
    // to n - kBlock - 1 have all terms j < m in every vector and load inside a; the others
    // (edges) go through edge().
    static void panel(const u32* a, std::size_t n, const u32* b, std::size_t m, u32* out, bool add) {
        const std::size_t total = n + m - 1;
        std::size_t k0 = 0;
        for (; k0 < total && k0 + 1 < m; k0 += kBlock) edge(a, n, b, m, k0, out, add);
        for (; k0 + kBlock < n; k0 += kBlock) full(a, k0, b, m, out + k0, add);
        for (; k0 < total; k0 += kBlock) edge(a, n, b, m, k0, out, add);
    }

    [[gnu::always_inline]] static void full(const u32* a, std::size_t k0, const u32* b, std::size_t m, u32* dst,
                                            bool add) {
        Acc acc[R];
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r) L::start(acc[r], a + k0 + r * kWidth);
        for (std::size_t j = 0;;) {
            const std::size_t stop = std::min(m, j + Sums<P>::kFold);
            terms<0, R>(acc, a, k0, b, j, stop);
            if ((j = stop) == m) break;
            fold(acc, R);
        }
        finish(acc, R, m, dst, add);
    }

    // The block of outputs [k0, k0 + kBlock) near either end of the product. Vector r holds outputs
    // [s, s + W), s = k0 + r W: terms with j in [s + 1 - n, s + W) within [0, m). Both ends of
    // that range grow with r. The vector loads a in [s + 1 - hi, s + W - lo]; out of [0, n), the
    // loads read a zero-padded copy. The j loop runs in phases with a fixed set of active vectors
    // [first, last), cut every kFold terms for a fold.
    static void edge(const u32* a, std::size_t n, const u32* b, std::size_t m, std::size_t k0, u32* out, bool add) {
        alignas(64) u32 window[kBlock + kPanel], partial[kBlock];
        const std::size_t total = n + m - 1;
        const int count = int(std::min<std::size_t>(R, (total - k0 + kWidth - 1) / kWidth));
        Range range[R];
        for (int r = 0; r < count; ++r) {
            const std::size_t s = k0 + r * kWidth;
            range[r] = {s + 1 > n ? s + 1 - n : 0, std::min(m, s + kWidth)};
        }
        const std::ptrdiff_t lo = std::ptrdiff_t(k0 + 1) - std::ptrdiff_t(range[0].hi);
        const std::ptrdiff_t hi = std::ptrdiff_t(k0 + count * kWidth - range[count - 1].lo);
        const bool whole = k0 + kBlock <= total;
        u32* const dst = whole ? out + k0 : partial;
        if (!whole && add) std::fill(std::copy(out + k0, out + total, partial), partial + kBlock, 0u);
        const u32* src = a;
        std::size_t k = k0;
        if (lo < 0 || hi >= std::ptrdiff_t(n)) {  // window[x] = a[lo + x], 0 outside a
            const std::ptrdiff_t from = std::max<std::ptrdiff_t>(lo, 0), to = std::min<std::ptrdiff_t>(hi + 1, n);
            std::fill(window, window + (from - lo), 0u);
            std::copy(a + from, a + to, window + (from - lo));
            std::fill(window + (to - lo), window + (hi + 1 - lo), 0u);
            src = window, k = std::size_t(std::ptrdiff_t(k0) - lo);
        }
        Acc acc[R];
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r)
            if (r < count) L::start(acc[r], src + k + r * kWidth - range[r].lo);
        const std::size_t begin = range[0].lo, end = range[count - 1].hi;
        int first = 0, last = 0;
        for (std::size_t j = begin;;) {
            const std::size_t segment = std::min(end, j + Sums<P>::kFold);
            while (j < segment) {
                while (range[first].hi <= j) ++first;
                while (last < count && range[last].lo <= j) ++last;
                std::size_t stop = std::min(segment, range[first].hi);
                if (last < count) stop = std::min(stop, range[last].lo);
                phase(first, last, acc, src, k, b, j, stop);
                j = stop;
            }
            if (j == end) break;
            fold(acc, count);
        }
        finish(acc, count, end - begin, dst, add);
        if (!whole) std::copy(partial, partial + (total - k0), out + k0);
    }

    // After `terms` steps with a fold every kFold: a last fold if needed, the last step, and the
    // stores of the first count vectors.
    [[gnu::always_inline]] static void finish(Acc (&acc)[R], int count, std::size_t terms, u32* dst, bool add) {
        const std::size_t since = (terms - 1) % Sums<P>::kFold + 1;
        if (since > (terms > Sums<P>::kFold ? Sums<P>::kLastFromFold : Sums<P>::kLastFromZero)) fold(acc, count);
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r) {
            if (r == count) break;
            V v = L::finish(acc[r]);
            if (add) v = M::add(v, S::load(dst + r * kWidth));
            S::store(dst + r * kWidth, v);
        }
    }

    // Terms j in [j, stop) of the vectors [First, Last).
    template <int First, int Last>
    [[gnu::always_inline]] static void terms(Acc (&acc)[R], const u32* src, std::size_t k, const u32* b,
                                             std::size_t j, std::size_t stop) {
        for (; j < stop; ++j) {
            const V bj = S::broadcast(b[j]);
#pragma GCC unroll 16
            for (int r = First; r < Last; ++r) L::step(acc[r], src + k + r * kWidth - j, bj);
        }
    }

    // terms<first, last> for runtime first < last <= R.
    template <int First = 0, int Last = R>
    [[gnu::always_inline]] static void phase(int first, int last, Acc (&acc)[R], const u32* src, std::size_t k,
                                             const u32* b, std::size_t j, std::size_t stop) {
        if (first == First && last == Last) return terms<First, Last>(acc, src, k, b, j, stop);
        if constexpr (Last > First + 1) phase<First, Last - 1>(first, last, acc, src, k, b, j, stop);
        else if constexpr (First + 1 < R) phase<First + 1, R>(first, last, acc, src, k, b, j, stop);
    }

    [[gnu::always_inline]] static void fold(Acc (&acc)[R], int count) {
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r)
            if (r < count) L::fold(acc[r]);
    }

    // out[j] = b[j] 2^32 mod P, by Shoup's product with w = 2^32 mod P.
    static void to_montgomery(const u32* b, std::size_t m, u32* out) {
        constexpr u32 w = Sums<P>::kR1, q = modarith::Ops<simd::Scalar, P>::shoup_quotient(w);
        std::size_t j = 0;
        for (; j + kWidth <= m; j += kWidth)
            S::store(out + j, M::template reduce<P>(M::shoup(S::load(b + j), S::broadcast(w), S::broadcast(q))));
        using N = modarith::Ops<simd::Scalar, P>;
        for (; j < m; ++j) out[j] = N::template reduce<P>(N::shoup(b[j], w, q));
    }
};

}  // namespace fps::small
