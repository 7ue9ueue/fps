// Tables of factorials and inverse factorials modulo a prime P, 2^12 < P < 2^30 (backend kernel).
//
//   fps::factorials::tables<P>(fact, inverse);  // fact[i] = i!, inverse[i] = 1 / i!, i < n
//
// fact and inverse are spans of n = fact.size() = inverse.size() words, n <= P (P! = 0 has no
// inverse); values in [0, P). Any alignment; no checks. select.hpp picks the variant per target
// and n:
//   model::tables<P>(fact, inverse, n)              the scalar loop (model.hpp), any n <= P
//   Chains<S, K, P>::run(fact, inverse, n, stream)  n >= Chains::kMinSize; W = K S::lanes lanes;
//                                                   stream: non-temporal stores (Avx512 only)
// Both are public, as the fast path for callers that pick their own variant.
//
// Chains, from lc-opt's product chains (lib/poly/factorials.hpp). Lane s of W covers positions
// [a_s, a_s + C), C = chunk(n) a multiple of the vector length, a_s = h + s C with h < S::lanes
// the words to fact's next vector boundary, so every store into fact is aligned (and into
// inverse when it has fact's offset). Lanes past the end all cover the last C positions: every
// lane stays inside the tables. Positions below h come from constant tables. Three passes:
//   1. lane products (a_s, a_s + C - 1], two factors per chain step, and from them (a_s)! and
//      (a_s + C - 1)! by a scalar prefix product over the lanes;
//   2. 1 / (a_s + C - 1)! by one exponentiation per lane, all lanes at once;
//   3. two chains: up from (a_s)! (times a_s + 1, a_s + 2, ...) and down from 1 / (a_s + C - 1)!
//      (times a_s + C - 1, a_s + C - 2, ...). Every vector length of steps, an in-register
//      transpose turns the lanes' terms into vectors of consecutive positions for the stores.
// A chain step is x <- montgomery(x, m) with m = i 2^32 mod P (Montgomery form), so x holds the
// plain product; x and m stay below 2P, m = base + u with u = j 2^32 mod P shared by all lanes.
// With stream, the lanes past the end start at an aligned position too and a scalar tail of
// fewer than S::lanes positions ends the tables; zmm stores then fill whole cache lines, which
// non-temporal stores need. Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

#include "backend/factorials/model.hpp"
#include "backend/factorials/select.hpp"
#include "backend/factorials/transpose.hpp"
#include "backend/modarith/modarith.hpp"

namespace fps::factorials {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

namespace detail {

// f(std::integral_constant<int, i>{}) for i = 0 .. N - 1, unrolled.
template <int N, class F>
[[gnu::always_inline]] inline void unrolled(F&& f) {
    [&]<int... i>(std::integer_sequence<int, i...>) {
        (f(std::integral_constant<int, i>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

// Non-temporal stores: Avx512 only, where one aligned store fills a cache line. Narrower ones
// leave lines partly written in the write-combining buffers (Zen 3, ymm: 4.5x slower).
#if defined(__AVX512F__) && defined(__AVX512BW__)
template <class S>
inline constexpr bool streams = std::is_same_v<S, simd::Avx512>;
inline void stream(u32* p, __m512i x) { _mm512_stream_si512(reinterpret_cast<__m512i*>(p), x); }
inline void fence() { _mm_sfence(); }
#else
template <class S>
inline constexpr bool streams = false;
inline void stream(u32*, auto) {}
inline void fence() {}
#endif

}  // namespace detail

template <class S, int K, u32 P>
class Chains {
public:
    using M = modarith::Ops<S, P>;
    using V = typename S::V;
    static constexpr int kLanes = S::lanes, kWidth = K * kLanes;
    static constexpr std::size_t kMinSize = 64;  // chunk(n) <= n - 15, so a_0 = h <= n - C
    static_assert(kWidth >= 2);

    // C: n / W rounded up to a multiple of the vector length; from 64 on, to a multiple of 16
    // with C / 16 odd, so that the W streams fall into different cache sets (lc-opt's
    // scan_chunk). C <= n / 2 + 32 <= n for n >= 64.
    static std::size_t chunk(std::size_t n) {
        std::size_t c = ((n + kWidth - 1) / kWidth + kLanes - 1) / kLanes * kLanes;
        if (c >= 64) {
            c = (c + 15) / 16 * 16;
            if (c / 16 % 2 == 0) c += 16;
        }
        return c;
    }

    static void run(u32* fact, u32* inverse, std::size_t n, bool stream = false) {
        stream = stream && detail::streams<S>;
        const std::size_t c = chunk(n);
        const std::size_t h = (kLanes - reinterpret_cast<std::uintptr_t>(fact) / 4 % kLanes) % kLanes;
        const std::size_t k = (n - h) / c;  // lanes s < k start at h + s c, lanes s >= k at a_last
        std::size_t a_last = n - c;
        if (stream && k < kWidth) a_last -= (n - c - h) % kLanes;  // aligned; [a_last + c, n) the tail
        const std::size_t tail = k < kWidth ? a_last + c : n;
        Lanes start{}, up_base, down_base, product, partial, range, first, last;
        for (int s = 0; s < kWidth; ++s) {
            const std::size_t a = std::min(h + s * c, a_last);
            start[s] = u32(a);
            up_base[s] = montgomery_form(a + 1);
            down_base[s] = montgomery_form(a + c - 1) + P;  // m = down_base - u is in [1, 2P)
        }
        // Lane k - 1 also gives a_last! / (a_(k-1))!, the factor that starts the lanes s >= k.
        lane_products(start, c - 1, k < kWidth ? a_last - start[k - 1] : 0, product, partial);
        // first[s] = (a_s)!, last[s] = (a_s + c - 1)!. For 0 < s < k, a_s = a_(s-1) + c, so
        // first[s] = last[s - 1] a_s; one dependent product per lane.
        for (std::size_t s = 1; s < k; ++s) range[s] = model::multiply<P>(start[s], product[s]);
        first[0] = kSmall[0][h], last[0] = model::multiply<P>(kSmall[0][h], product[0]);
        for (std::size_t s = 1; s < k; ++s)
            first[s] = model::multiply<P>(last[s - 1], start[s]), last[s] = model::multiply<P>(last[s - 1], range[s]);
        for (std::size_t s = k; s < kWidth; ++s)
            first[s] = model::multiply<P>(first[k - 1], partial[k - 1]), last[s] = model::multiply<P>(first[s], product[s]);
        // The tail: fact up to (n - 1)!, which the lanes s >= k invert; then inverse down from it.
        u32 f = last[kWidth - 1];
        for (std::size_t i = tail; i < n; ++i) fact[i] = f = model::multiply<P>(f, u32(i));
        for (std::size_t s = k; s < kWidth; ++s) last[s] = f;
        invert(last);
        u32 g = last[kWidth - 1];
        for (std::size_t i = n; i > tail; --i) inverse[i - 1] = g, g = model::multiply<P>(g, u32(i - 1));
        for (std::size_t s = k; s < kWidth; ++s) last[s] = g;
        std::copy_n(kSmall[0].data(), h, fact), std::copy_n(kSmall[1].data(), h, inverse);
        const auto chains = [&]<bool up, bool down> {
            scan<up, down>(fact, inverse, c, start, first, last, up_base, down_base);
        };
        if constexpr (detail::streams<S>) {
            if (stream) {  // inverse's stores are aligned when it has fact's offset
                const auto at = [](u32* p) { return reinterpret_cast<std::uintptr_t>(p); };
                if ((at(inverse) - at(fact)) % (4 * kLanes) == 0) chains.template operator()<true, true>();
                else chains.template operator()<true, false>();
                return detail::fence();
            }
        }
        chains.template operator()<false, false>();
    }

private:
    using Lanes = std::array<u32, kWidth>;
    static constexpr u32 kR = u32((u64(1) << 32) % P);  // 1 in Montgomery form
    static constexpr u32 kR2 = u32(u64(kR) * kR % P);
    // i! and 1 / i! for i < 16 (h < S::lanes <= 16).
    static constexpr auto kSmall = [] {
        std::array<std::array<u32, 16>, 2> t{};
        for (u32 i = 0, f = 1; i < 16; f = model::multiply<P>(f, ++i)) t[0][i] = f, t[1][i] = model::inverse<P>(f);
        return t;
    }();

    static u32 montgomery_form(std::size_t i) { return u32((u64(i % P) << 32) % P); }

    static void load(V (&x)[K], const Lanes& lanes) {
        detail::unrolled<K>([&](auto v) { x[v] = S::load(lanes.data() + v * kLanes); });
    }
    static void store(Lanes& lanes, const V (&x)[K]) {
        detail::unrolled<K>([&](auto v) { S::store(lanes.data() + v * kLanes, M::template reduce<P>(x[v])); });
    }

    // product[s] = (a + 1) (a + 2) ... (a + factors) for a = start[s]; partial[s] the same with
    // `capture` <= factors factors. Two factors per step: x <- montgomery(x, g) with g = f 2^32
    // mod P for f = (a + 2j + 1)(a + 2j + 2); g steps by d = (8j + 4a + 10) 2^32, d by 8 2^32.
    [[gnu::flatten]] static void lane_products(const Lanes& start, std::size_t factors, std::size_t capture,
                                               Lanes& product, Lanes& partial) {
        Lanes g0, d0;
        for (int s = 0; s < kWidth; ++s) {
            const u64 a = start[s];
            g0[s] = montgomery_form((a + 1) % P * ((a + 2) % P));
            d0[s] = montgomery_form(4 * a + 10);
        }
        V x[K], g[K], d[K];
        load(g, g0), load(d, d0);
        for (V& v : x) v = S::broadcast(1);
        const V eight = S::broadcast(montgomery_form(8));
        const auto steps = [&](std::size_t pairs) {
            for (std::size_t j = 0; j < pairs; ++j)
                detail::unrolled<K>([&](auto v) {
                    x[v] = M::montgomery(x[v], g[v]);
                    g[v] = M::add(g[v], d[v]), d[v] = M::add(d[v], eight);
                });
        };
        steps(capture / 2);
        store(partial, x);
        steps(factors / 2 - capture / 2);
        store(product, x);
        for (int s = 0; s < kWidth; ++s) {  // odd counts: the last factor
            if (capture % 2) partial[s] = model::multiply<P>(partial[s], u32((start[s] + capture) % P));
            if (factors % 2) product[s] = model::multiply<P>(product[s], u32((start[s] + factors) % P));
        }
    }

    // x[s] <- x[s]^(P - 2) = 1 / x[s], for x[s] in [1, P); in Montgomery form throughout.
    [[gnu::flatten]] static void invert(Lanes& lanes) {
        V x[K], y[K];
        load(x, lanes);
        detail::unrolled<K>([&](auto v) { x[v] = M::montgomery(x[v], S::broadcast(kR2)), y[v] = S::broadcast(kR); });
        for (u32 e = P - 2; e; e >>= 1) {  // y = x^(e's low bits so far); the two chains overlap
            if (e & 1) detail::unrolled<K>([&](auto v) { y[v] = M::montgomery(y[v], x[v]); });
            detail::unrolled<K>([&](auto v) { x[v] = M::montgomery(x[v], x[v]); });
        }
        detail::unrolled<K>([&](auto v) { y[v] = M::montgomery(y[v], S::broadcast(1)); });
        store(lanes, y);
    }

    // The up and down chains over c steps, kLanes steps per block. Block b of a chain holds the
    // terms of its next kLanes steps; while the chains fill one block, the other is transposed
    // and stored. The down chain records each block in reverse, so it is stored in position order.
    // up, down: non-temporal stores into fact, inverse (aligned addresses only).
    template <bool up_streams, bool down_streams>
    [[gnu::flatten]] static void scan(u32* fact, u32* inverse, std::size_t c, const Lanes& start, const Lanes& first,
                     const Lanes& last, const Lanes& up_base, const Lanes& down_base) {
        V up[K], down[K], ub[K], db[K];
        load(up, first), load(down, last), load(ub, up_base), load(db, down_base);
        u32* up_out[kWidth];
        u32* down_out[kWidth];
        for (int s = 0; s < kWidth; ++s) up_out[s] = fact + start[s], down_out[s] = inverse + start[s] + c - kLanes;
        V up_block[2][K][kLanes], down_block[2][K][kLanes];
        V u = S::broadcast(0);
        const V r = S::broadcast(kR);
        const auto put = [](auto streams, u32* p, V x) {
            if constexpr (streams) detail::stream(p, x);
            else S::store(p, x);
        };
        const auto advance = [&](int b, auto t) {
            detail::unrolled<K>([&](auto v) {
                up_block[b][v][t] = up[v];
                down_block[b][v][kLanes - 1 - t] = down[v];
                up[v] = M::montgomery(up[v], S::add(ub[v], u));
                down[v] = M::montgomery(down[v], S::sub(db[v], u));
            });
            u = M::add(u, r);
        };
        // Step t of a block stores lanes t K .. t K + K - 1; a vector is transposed at its first lane.
        const auto emit = [&](int b, std::size_t j, auto t) {
            detail::unrolled<K>([&](auto i) {
                constexpr int e = decltype(t)::value * K + decltype(i)::value, v = e / kLanes, l = e % kLanes;
                if constexpr (l == 0) transpose(up_block[b][v]), transpose(down_block[b][v]);
                put(std::bool_constant<up_streams>{}, up_out[e] + j, M::template reduce<P>(up_block[b][v][l]));
                put(std::bool_constant<down_streams>{}, down_out[e] - j, M::template reduce<P>(down_block[b][v][l]));
            });
        };
        detail::unrolled<kLanes>([&](auto t) { advance(0, t); });
        std::size_t j = 0;
        int cur = 0;
        for (; j + kLanes < c; j += kLanes, cur ^= 1)
            detail::unrolled<kLanes>([&](auto t) { advance(cur ^ 1, t), emit(cur, j, t); });
        detail::unrolled<kLanes>([&](auto t) { emit(cur, j, t); });
    }
};

// fact[i] = i!, inverse[i] = 1 / i! for i < n = fact.size() = inverse.size() <= P.
template <u32 P>
void tables(std::span<u32> fact, std::span<u32> inverse) {
    const std::size_t n = fact.size();
    if (n < std::max(kMinChains, Chains<Isa, kVectors, P>::kMinSize)) model::tables<P>(fact.data(), inverse.data(), n);
    else Chains<Isa, kVectors, P>::run(fact.data(), inverse.data(), n, n >= kMinStream);
}

}  // namespace fps::factorials
