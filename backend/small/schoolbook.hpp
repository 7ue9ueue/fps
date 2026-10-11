// Schoolbook products on a simd struct S, modulo P, 2^12 < P < 2^30 (variant of the small kernel).
//
//   Schoolbook<S, R, P, U, Solo>::run(a, n, b, m, out);  // out = a b: n + m - 1 words; n >= m >= 1
//
// Inputs below P, output in [0, P). out must not overlap a or b. No allocation: scratch is on the
// stack (about 9 KiB). R: vectors per block; U: unrolling of the loop with all R vectors; Solo:
// the blocks at the ends of the product run one vector at a time, without phases.
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
// zero-padded copy of a's start or end instead, made once per panel; the last vector's words past
// the product are not stored (store_first). b is taken in panels of at most kPanel words, so the
// scratch is fixed; panels after the first add into out.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/modarith/modarith.hpp"
#include "backend/simd/simd.hpp"

namespace fps::small {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

// Bounds of 64-bit sums of products x y, x, y < P. fold(s) = s - h kQ = h R1 + l for
// s = h 2^32 + l, with R1 = 2^32 mod P and kQ = 2^32 - R1, a multiple of P: congruent to s and
// below kFolded. Montgomery's last step (s + m P) / 2^32 is below 4P when s < 3P 2^32.
template <u32 P>
struct Sums {
    static_assert(P > (1u << 12) && P < (1u << 30));
    static constexpr u128 kProduct = u128(P - 1) * (P - 1);
    static constexpr u32 kR1 = u32((u64(1) << 32) % P);
    static constexpr u32 kQ = u32((u64(1) << 32) - kR1);
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
        return S::sub64(s, S::mul_even(S::template shr64<32>(s), S::broadcast(Sums<P>::kQ)));
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
        const V high = S::hi_halves(x.lo, x.hi), q = S::broadcast(Sums<P>::kQ);
        x.lo = S::mls_wide_lo(x.lo, high, q);
        x.hi = S::mls_wide_hi(x.hi, high, q);
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
    static void fold(Acc& x) { x.s -= (x.s >> 32) * Sums<P>::kQ; }
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

template <class S, int R, u32 P, int U = 1, bool Solo = false>
class Schoolbook {
public:
    static constexpr std::size_t kWidth = S::lanes, kBlock = R * kWidth;
    static constexpr std::size_t kPanel = 1024;  // words of b per panel

    // One call for b of at most one panel: callers inline it once per candidate (small.hpp).
    static void run(const u32* a, std::size_t n, const u32* b, std::size_t m, u32* out) {
        if (m <= kPanel) return panel(a, n, b, m, out, false);
        panels(a, n, b, m, out);
    }

private:
    using L = detail::Lanes<S, P>;
    using M = modarith::Ops<S, P>;
    using V = typename S::V;
    using Acc = typename L::Acc;

    // b of more than one panel: panels after the first add into out.
    [[gnu::noinline]] static void panels(const u32* a, std::size_t n, const u32* b, std::size_t m, u32* out) {
        std::fill(out + n + kPanel - 1, out + n + m - 1, 0u);
        for (std::size_t first = 0; first < m; first += kPanel)
            panel(a, n, b + first, std::min(kPanel, m - first), out + first, first > 0);
    }

    // The terms of one output vector: j in [lo, hi).
    struct Range {
        std::size_t lo, hi;
    };

    // Words of a that panel() copies at most: its start up to the first vector whose windows stay in
    // a, its end from the last interior block on, or all of a when no interior block fits.
    static constexpr std::size_t kCopy = kPanel + kWidth + kBlock;

    // out[k] (+)= sum over j of a[k - j] b[j], k < n + m - 1, m <= kPanel. Vectors at s >= m - 1 with
    // s + kWidth < n load inside a: full blocks of R from `begin` on. The others (edges, and the
    // interior vectors left over before the end) read a zero-padded copy of the stretch of a they
    // load, made once for the start and once for the end. The first copy comes before b' is made:
    // a load that spans several of its stores waits until they reach the cache.
    static void panel(const u32* a, std::size_t n, const u32* b, std::size_t m, u32* out, bool add) {
        alignas(64) u32 scaled[kPanel], copy[kWidth + kCopy + 2 * kWidth];
        const std::size_t total = n + m - 1, begin = (m + kWidth - 2) / kWidth * kWidth;
        if (begin + kBlock >= n) {  // no interior block: n < kCopy
            pad(a, n, copy);
            to_montgomery(b, m, scaled);
            for (std::size_t k0 = 0; k0 < total; k0 += kBlock) edge(copy + kWidth, k0, n, scaled, m, k0, out, add);
            return;
        }
        if (begin > 0) pad(a, begin + 1, copy);  // windows reach a[-(W - 1)] .. a[begin]
        to_montgomery(b, m, scaled);
        for (std::size_t k0 = 0; k0 < begin; k0 += kBlock)
            edge(copy + kWidth, k0, n, scaled, m, k0, out, add, int(std::min<std::size_t>(R, (begin - k0) / kWidth)));
        std::size_t k0 = begin;
        for (; k0 + kBlock < n; k0 += kBlock) full(a, k0, scaled, m, out + k0, add);
        const std::size_t q = k0 - (m - 1);  // windows reach a[q] .. a[n - 1 + W]
        pad(a + q, n - q, copy);
        for (; k0 < total; k0 += kBlock) edge(copy + kWidth, k0 - q, n, scaled, m, k0, out, add);
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
        finish(acc, R, m, dst, kBlock, add);
    }

    // The block of outputs [k0, k0 + count W) near either end of the product (count = R, or fewer
    // up to the end). Vector r holds outputs [s, s + W), s = k0 + r W: terms with j in
    // [s + 1 - n, s + W) within [0, m). Both ends of that range grow with r. Its window at step j
    // is src + k + r W - j: a[s - j .. s - j + W] in a zero-padded copy. The j loop runs in phases
    // with a fixed set of active vectors [first, last), cut every kFold terms for a fold; with Solo,
    // one vector after the other instead.
    static void edge(const u32* src, std::size_t k, std::size_t n, const u32* b, std::size_t m, std::size_t k0,
                     u32* out, bool add, int count = R) {
        const std::size_t total = n + m - 1;
        count = int(std::min<std::size_t>(count, (total - k0 + kWidth - 1) / kWidth));
        if constexpr (Solo) {
            for (int r = 0; r < count; ++r) {
                const std::size_t s = k0 + r * kWidth;
                one(src + k + r * kWidth, b, s + 1 > n ? s + 1 - n : 0, std::min(m, s + kWidth), out + s, total - s,
                    add);
            }
        } else {
            phases(src, k, n, b, m, k0, out, add, count);
        }
    }

    // edge() without Solo.
    [[gnu::always_inline]] static void phases(const u32* src, std::size_t k, std::size_t n, const u32* b,
                                              std::size_t m, std::size_t k0, u32* out, bool add, int count) {
        const std::size_t total = n + m - 1;
        Range range[R];
        for (int r = 0; r < count; ++r) {
            const std::size_t s = k0 + r * kWidth;
            range[r] = {s + 1 > n ? s + 1 - n : 0, std::min(m, s + kWidth)};
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
        finish(acc, count, end - begin, out + k0, total - k0, add);
    }

    // One output vector at dst: terms j in [lo, hi), window w - j; only the words below `valid` exist.
    static void one(const u32* w, const u32* b, std::size_t lo, std::size_t hi, u32* dst, std::size_t valid, bool add) {
        Acc acc[1];
        L::start(acc[0], w - lo);
        for (std::size_t j = lo;;) {
            const std::size_t stop = std::min(hi, j + Sums<P>::kFold);
            for (; j < stop; ++j) L::step(acc[0], w - j, S::broadcast(b[j]));
            if (j == hi) break;
            L::fold(acc[0]);
        }
        finish<1>(acc, 1, hi - lo, dst, valid, add);
    }

    // After `terms` steps with a fold every kFold: a last fold if needed, the last step, and the
    // stores of the first count vectors, of which only the words below `valid` exist.
    template <int N = R>
    [[gnu::always_inline]] static void finish(Acc (&acc)[N], int count, std::size_t terms, u32* dst, std::size_t valid,
                                              bool add) {
        const std::size_t since = (terms - 1) % Sums<P>::kFold + 1;
        if (since > (terms > Sums<P>::kFold ? Sums<P>::kLastFromFold : Sums<P>::kLastFromZero)) fold(acc, count);
#pragma GCC unroll 16
        for (int r = 0; r < N; ++r) {
            if (r == count) break;
            V v = L::finish(acc[r]);
            u32* const p = dst + r * kWidth;
            if (r + 1 == count && valid < std::size_t(count) * kWidth)
                return store_short(p, v, valid - r * kWidth, add);
            if (add) v = M::add(v, S::load(p));
            S::store(p, v);
        }
    }

    // p[0, c) (+)= the first c < W words of v.
    static void store_short(u32* p, V v, std::size_t c, bool add) {
        if (add) {
            alignas(64) u32 t[kWidth] = {};
            copy_short(t, p, c);
            v = M::add(v, S::load(t));
        }
        S::store_first(p, v, int(c));
    }

    // copy[0, W) = 0, copy[W, W + c) = a[0, c), copy[W + c, W + c + 2W) = 0.
    static void pad(const u32* a, std::size_t c, u32* copy) {
        u32* const dst = copy + kWidth;
        S::store(copy, V{});
        std::size_t x = 0;
        for (; x + kWidth <= c; x += kWidth) S::store(dst + x, S::load(a + x));
        if (c >= kWidth) S::store(dst + c - kWidth, S::load(a + c - kWidth));
        else copy_short(dst, a, c);
        S::store(dst + c, V{});
        S::store(dst + c + kWidth, V{});
    }

    // dst[0, c) = src[0, c), c < W: pieces of 8, 4, 2 and 1 words (no call to memcpy).
    static void copy_short(u32* dst, const u32* src, std::size_t c) {
        std::size_t x = 0;
        piece<8>(dst, src, c, x);
        piece<4>(dst, src, c, x);
        piece<2>(dst, src, c, x);
        piece<1>(dst, src, c, x);
    }

    template <std::size_t kWords>
    [[gnu::always_inline]] static void piece(u32* dst, const u32* src, std::size_t c, std::size_t& x) {
        if constexpr (kWords < kWidth)
            if (c & kWords) {
                std::memcpy(dst + x, src + x, kWords * sizeof(u32));
                x += kWords;
            }
    }

    // Terms j in [j, stop) of the vectors [First, Last); the loop with all R vectors unrolled U times.
    template <int First, int Last>
    [[gnu::always_inline]] static void terms(Acc (&acc)[R], const u32* src, std::size_t k, const u32* b,
                                             std::size_t j, std::size_t stop) {
        if constexpr (First == 0 && Last == R && U > 1) {
#pragma GCC unroll U
            for (; j < stop; ++j) step<First, Last>(acc, src, k, b, j);
        } else {
            for (; j < stop; ++j) step<First, Last>(acc, src, k, b, j);
        }
    }

    template <int First, int Last>
    [[gnu::always_inline]] static void step(Acc (&acc)[R], const u32* src, std::size_t k, const u32* b, std::size_t j) {
        const V bj = S::broadcast(b[j]);
        const u32* base = src + k - j;
        // Neon: GCC keeps windows in registers from one j to the next (vector r's window at step j is
        // vector r + 1's at step j + W) and pays a move for each; an opaque base stops it (notes.md).
        if constexpr (detail::HasMulWide<S>) asm("" : "+r"(base));
#pragma GCC unroll 16
        for (int r = First; r < Last; ++r) L::step(acc[r], base + r * kWidth, bj);
    }

    // terms<first, last> for runtime first < last <= R: one jump through a table (a chain of compares
    // cost 3-9% on products of 32-128 words a side, notes.md).
    [[gnu::always_inline]] static void phase(int first, int last, Acc (&acc)[R], const u32* src, std::size_t k,
                                             const u32* b, std::size_t j, std::size_t stop) {
        static_assert(R <= 8);
#define FPS_SMALL_PHASE(F, L)                                              \
    case F * 16 + L:                                                       \
        if constexpr (L <= R) return terms<F, L>(acc, src, k, b, j, stop); \
        break;
        switch (first * 16 + last) {
            FPS_SMALL_PHASE(0, 1) FPS_SMALL_PHASE(0, 2) FPS_SMALL_PHASE(0, 3) FPS_SMALL_PHASE(0, 4)
            FPS_SMALL_PHASE(0, 5) FPS_SMALL_PHASE(0, 6) FPS_SMALL_PHASE(0, 7) FPS_SMALL_PHASE(0, 8)
            FPS_SMALL_PHASE(1, 2) FPS_SMALL_PHASE(1, 3) FPS_SMALL_PHASE(1, 4) FPS_SMALL_PHASE(1, 5)
            FPS_SMALL_PHASE(1, 6) FPS_SMALL_PHASE(1, 7) FPS_SMALL_PHASE(1, 8)
            FPS_SMALL_PHASE(2, 3) FPS_SMALL_PHASE(2, 4) FPS_SMALL_PHASE(2, 5) FPS_SMALL_PHASE(2, 6)
            FPS_SMALL_PHASE(2, 7) FPS_SMALL_PHASE(2, 8)
            FPS_SMALL_PHASE(3, 4) FPS_SMALL_PHASE(3, 5) FPS_SMALL_PHASE(3, 6) FPS_SMALL_PHASE(3, 7)
            FPS_SMALL_PHASE(3, 8)
            FPS_SMALL_PHASE(4, 5) FPS_SMALL_PHASE(4, 6) FPS_SMALL_PHASE(4, 7) FPS_SMALL_PHASE(4, 8)
            FPS_SMALL_PHASE(5, 6) FPS_SMALL_PHASE(5, 7) FPS_SMALL_PHASE(5, 8)
            FPS_SMALL_PHASE(6, 7) FPS_SMALL_PHASE(6, 8)
            FPS_SMALL_PHASE(7, 8)
        }
#undef FPS_SMALL_PHASE
        __builtin_unreachable();
    }

    template <int N>
    [[gnu::always_inline]] static void fold(Acc (&acc)[N], int count) {
#pragma GCC unroll 16
        for (int r = 0; r < N; ++r)
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
