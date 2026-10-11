// Tables of factorials and inverse factorials modulo a prime P, 2^12 < P < 2^30 (backend kernel).
//
//   fps::factorials::tables<P>(fact, inverse);  // fact[i] = i!, inverse[i] = 1 / i!, i < n
//
// fact and inverse are spans of n = fact.size() = inverse.size() words, n <= P (P! = 0 has no
// inverse); values in [0, P). Any alignment; no checks. select.hpp picks the variant per target
// and n:
//   model::tables<P>(fact, inverse, n)          the scalar loop (model.hpp), any n <= P
//   Chains<S, K, P>::run(fact, inverse, n)      n >= Chains::kMinSize; W = K S::lanes lanes
// Both are public, as the fast path for callers that pick their own variant.
//
// Chains, from lc-opt's product chains (lib/poly/factorials.hpp). Lane s of W covers positions
// [a_s, a_s + C), C = chunk(n) a multiple of the vector length, a_s = min(h + s C, n - C) with
// h < S::lanes the words to fact's next vector boundary: the stores into fact are aligned (and
// into inverse when it has fact's offset) except for the lanes past n - C, which all cover the
// last C positions, so every lane stays inside the tables. Positions below h come from constant
// tables. Three passes:
//   1. lane products (a_s, a_s + C - 1], four factors per chain step, and from them (a_s)! and
//      (a_s + C - 1)! by a scalar prefix product over the lanes;
//   2. 1 / (a_s + C - 1)! by one exponentiation per lane, all lanes at once;
//   3. two chains: up from (a_s)! (times a_s + 1, a_s + 2, ...) and down from 1 / (a_s + C - 1)!
//      (times a_s + C - 1, a_s + C - 2, ...). Every vector length of steps, an in-register
//      transpose turns the lanes' terms into vectors of consecutive positions for the stores.
// A chain step is x <- montgomery(x, m) with m = i 2^32 mod P (Montgomery form), so x holds the
// plain product; x and m stay below 2P, m = base + u with u = j 2^32 mod P shared by all lanes.
// Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
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

    static void run(u32* fact, u32* inverse, std::size_t n) {
        const std::size_t c = chunk(n);
        const std::size_t h = (kLanes - reinterpret_cast<std::uintptr_t>(fact) / 4 % kLanes) % kLanes;
        const std::size_t k = (n - h) / c;  // lanes s < k start at h + s c, lanes s >= k at n - c
        Lanes start{}, up_base, down_base, product, range, first, last;
        for (int s = 0; s < kWidth; ++s) {
            const std::size_t a = std::min(h + s * c, n - c);
            start[s] = u32(a);
            up_base[s] = montgomery_form(a + 1);
            down_base[s] = montgomery_form(a + c - 1) + P;  // m = down_base - u is in [1, 2P)
        }
        // Lane k - 1 also gives (n - c)! / (a_(k-1))!, the factor that starts the lanes s >= k.
        const u32 partial = lane_products(start, up_base, c - 1, k < kWidth ? n - c - start[k - 1] : 0, k - 1, product);
        // first[s] = (a_s)!, last[s] = (a_s + c - 1)! = range[0] ... range[s] for s < k, range[s]
        // = a_s product[s] (range[0] = h! product[0]); first[s] = last[s - 1] a_s. The prefix
        // products in groups of 4 lanes (independent chains), then across the groups: k / 4 + 4
        // dependent products, not k.
        range[0] = model::multiply<P>(kSmall[0][h], product[0]);
        for (std::size_t s = 1; s < k; ++s) range[s] = model::multiply<P>(start[s], product[s]);
        for (std::size_t s = 0; s < k; ++s) last[s] = s % 4 ? model::multiply<P>(last[s - 1], range[s]) : range[s];
        for (std::size_t g = 4; g < k; g += 4)
            for (std::size_t s = g; s < std::min(g + 4, k); ++s) last[s] = model::multiply<P>(last[g - 1], last[s]);
        first[0] = kSmall[0][h];
        for (std::size_t s = 1; s < k; ++s) first[s] = model::multiply<P>(last[s - 1], start[s]);
        for (std::size_t s = k; s < kWidth; ++s)
            first[s] = model::multiply<P>(first[k - 1], partial), last[s] = model::multiply<P>(first[s], product[s]);
        invert(last);
        for (std::size_t i = 0; i < h; ++i) fact[i] = kSmall[0][i], inverse[i] = kSmall[1][i];
        scan(fact, inverse, c, start, first, last, up_base, down_base);
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

    // product[s] = (a + 1) (a + 2) ... (a + factors) for a = start[s], from up_base[s] = the
    // Montgomery form of a + 1; returns the product of the first `capture` <= factors factors for
    // lane `lane`. The first r = factors % 4 factors one per step, then four per step:
    // x <- montgomery(x, g) with g the Montgomery form of G = w (w + 1)(w + 2)(w + 3) for
    // w = a + r + 1 + 4j, stepped by its forward differences d1, d2, d3 (d4 = 4! 4^4). The start
    // values from G = z (z + 2), z = w (w + 3), and z(w + 4) = z(w) + 8w + 28.
    [[gnu::flatten]] static u32 lane_products(const Lanes& start, const Lanes& up_base, std::size_t factors,
                                              std::size_t capture, std::size_t lane, Lanes& product) {
        const std::size_t r = factors % 4;
        const auto c = [](u32 i) { return S::broadcast(montgomery_form(i)); };
        const auto times = [](V a, V b) { return M::template reduce<P>(M::montgomery(a, b)); };
        V x[K], w[K], g[K], d1[K], d2[K], d3[K];
        load(w, up_base);
        for (V& v : x) v = S::broadcast(1);
        for (std::size_t i = 0; i < r; ++i)
            detail::unrolled<K>([&](auto v) { x[v] = M::montgomery(x[v], w[v]), w[v] = M::add(w[v], c(1)); });
        detail::unrolled<K>([&](auto v) {
            V z = times(w[v], M::add(w[v], c(3)));
            V step = M::add(w[v], w[v]);
            step = M::add(step, step), step = M::add(M::add(step, step), c(28));  // 8w + 28
            V G[4];
            for (int m = 0; m < 4; ++m) {
                G[m] = times(z, M::add(z, c(2)));
                z = M::add(z, step), step = M::add(step, c(32));
            }
            g[v] = G[0], d1[v] = M::sub(G[1], G[0]);
            d2[v] = M::sub(M::sub(G[2], G[1]), d1[v]);
            d3[v] = M::sub(M::sub(M::sub(G[3], G[2]), M::sub(G[2], G[1])), d2[v]);
        });
        const V d4 = c(6144);
        const auto steps = [&](std::size_t count) {
            for (std::size_t j = 0; j < count; ++j)
                detail::unrolled<K>([&](auto v) {
                    x[v] = M::montgomery(x[v], g[v]);
                    g[v] = M::add(g[v], d1[v]), d1[v] = M::add(d1[v], d2[v]);
                    d2[v] = M::add(d2[v], d3[v]), d3[v] = M::add(d3[v], d4);
                });
        };
        // The capture: whole steps, then up to 3 factors for the one lane.
        const std::size_t whole = capture > r ? (capture - r) / 4 : 0;
        steps(whole);
        store(product, x);
        u32 partial = capture > r ? product[lane] : 1;
        for (std::size_t i = capture > r ? r + 4 * whole : 0; i < capture; ++i)
            partial = model::multiply<P>(partial, u32((start[lane] + i + 1) % P));
        steps((factors - r) / 4 - whole);
        store(product, x);
        return partial;
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
                S::store(up_out[e] + j, M::template reduce<P>(up_block[b][v][l]));
                S::store(down_out[e] - j, M::template reduce<P>(down_block[b][v][l]));
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
    else if (n < kMinLarge) Chains<Isa, kVectors, P>::run(fact.data(), inverse.data(), n);
    else Chains<IsaLarge, kVectorsLarge, P>::run(fact.data(), inverse.data(), n);
}

}  // namespace fps::factorials
