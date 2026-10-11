// Tables of factorials and inverse factorials modulo a prime P, 2^12 < P < 2^30 (backend kernel).
//
//   fps::factorials::tables<P>(fact, inverse);  // fact[i] = i!, inverse[i] = 1 / i!, i < n
//
// fact and inverse are spans of n = fact.size() = inverse.size() words, n <= P (P! = 0 has no
// inverse); values in [0, P). Any alignment; no checks. select.hpp picks the variant per target
// and n:
//   model::tables<P>(fact, inverse, n)          the scalar loop (model.hpp), any n <= P
//   Chains<S, K, P, step>::run(fact, inverse, n)
//                                               n >= Chains::kMinSize; W = K S::lanes lanes;
//                                               step: the scan's chain product (Step)
// Both are public, as the fast path for callers that pick their own variant.
//
// Chains, from lc-opt's product chains (lib/poly/factorials.hpp). Lane s of W covers positions
// [a_s, a_s + C), C = chunk(n) a multiple of the vector length, a_s = min(h + s C, n - C) with
// h < S::lanes the words to fact's next vector boundary: the stores into fact are aligned (and
// into inverse when it has fact's offset) except for the lanes past n - C, which all cover the
// last C positions, so every lane stays inside the tables. Positions below h come from constant
// tables. Three passes:
//   1. lane products (a_s, a_s + C - 1], four factors per chain step;
//   2. from them (a_s)! and 1 / (a_s + C - 1)! for every lane: a prefix and a suffix product
//      over the lanes in vector registers, and one scalar exponentiation (starts);
//   3. two chains: up from (a_s)! (times a_s + 1, a_s + 2, ...) and down from 1 / (a_s + C - 1)!
//      (times a_s + C - 1, a_s + C - 2, ...). Every vector length of steps, an in-register
//      transpose turns the lanes' terms into vectors of consecutive positions for the stores.
// Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
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

inline constexpr auto kIota = [] {
    std::array<u32, 128> t{};
    for (u32 i = 0; i < 128; ++i) t[i] = i;
    return t;
}();

}  // namespace detail

template <class S, int K, u32 P, Step kStep = Step::shoup>
class Chains {
public:
    using M = modarith::Ops<S, P>;
    using V = typename S::V;
    static constexpr int kLanes = S::lanes, kWidth = K * kLanes;
    // chunk(n) + h <= n: C <= n / 2 + 2 kLanes - 1 and h < kLanes (C >= 64 only from n >= 98).
    static constexpr std::size_t kMinSize = 4 * kLanes;
    static_assert(kWidth >= 2 && kWidth <= 128);

    // C: n / W rounded up to a multiple of the vector length; from 64 on, to a multiple of 16,
    // m = C / 16 lines, plus one line where the lanes' streams would share few cache sets. The
    // W lanes start m lines apart, so in a 64-set L1 they use 64 / gcd(m, 64) sets; m is bumped
    // when it is a multiple of 512 / W, where that leaves more than 4 streams per set (lc-opt's
    // scan_chunk bumped every even m, which costs up to 25% duplicate work at n = 2^10).
    static std::size_t chunk(std::size_t n) {
        std::size_t c = ((n + kWidth - 1) / kWidth + kLanes - 1) / kLanes * kLanes;
        if (c >= 64) {
            c = (c + 15) / 16 * 16;
            if (c / 16 % (512 / kWidth) == 0) c += 16;
        }
        return c;
    }

    static void run(u32* fact, u32* inverse, std::size_t n) {
        const std::size_t c = chunk(n);
        const std::size_t h = (kLanes - reinterpret_cast<std::uintptr_t>(fact) / 4 % kLanes) % kLanes;
        const std::size_t k = (n - h) / c;  // lanes s < k start at h + s c, lanes s >= k at n - c
        Lanes start, up_base, product, first, down;
        detail::unrolled<K>([&](auto v) {
            const V iota = S::load(detail::kIota.data() + v * kLanes);
            const V a = S::min(S::add(S::broadcast(u32(h)), S::mul_lo(iota, S::broadcast(u32(c)))),
                               S::broadcast(u32(n - c)));
            S::store(start.data() + v * kLanes, a);
            S::store(up_base.data() + v * kLanes, to_montgomery(S::add(a, S::broadcast(1))));
        });
        // Lane k - 1 also gives (n - c)! / (a_(k-1))!, the factor that starts the lanes s >= k.
        const u32 partial = lane_products(start, up_base, c - 1, k < kWidth ? n - c - start[k - 1] : 0, k - 1, product);
        starts(start, product, partial, c, h, k, first, down);
        for (std::size_t i = 0; i < h; ++i) fact[i] = kSmall[0][i], inverse[i] = kSmall[1][i];
        scan(fact, inverse, c, start, first, down);
    }

private:
    using Lanes = std::array<u32, kWidth>;
    static constexpr u32 kR = u32((u64(1) << 32) % P);  // 1 in Montgomery form
    static constexpr u32 kR2 = u32(u64(kR) * kR % P);
    static constexpr u32 kR3 = u32(u64(kR2) * kR % P);
    static constexpr auto kHalves = [] {  // 2^-t mod P, t < 60
        std::array<u32, 60> h{1};
        for (int t = 1; t < 60; ++t) h[t] = model::multiply<P>(h[t - 1], (P + 1) / 2);
        return h;
    }();
    // i! and 1 / i! for i < 16 (h < S::lanes <= 16).
    static constexpr auto kSmall = [] {
        std::array<std::array<u32, 16>, 2> t{};
        for (u32 i = 0, f = 1; i < 16; f = model::multiply<P>(f, ++i)) t[0][i] = f, t[1][i] = model::inverse<P>(f);
        return t;
    }();

    static u32 montgomery_form(std::size_t i) { return u32((u64(i % P) << 32) % P); }
    // x 2^32 mod P in [0, P), for x <= P.
    static V to_montgomery(V x) { return M::template reduce<P>(M::montgomery(x, S::broadcast(kR2))); }

    static void load(V (&x)[K], const u32* lanes) {
        detail::unrolled<K>([&](auto v) { x[v] = S::load(lanes + v * kLanes); });
    }
    static void store(Lanes& lanes, const V (&x)[K]) {
        detail::unrolled<K>([&](auto v) { S::store(lanes.data() + v * kLanes, M::template reduce<P>(x[v])); });
    }

    // product[s] = (a + 1) (a + 2) ... (a + factors) for a = start[s], from up_base[s] = the
    // Montgomery form of a + 1; returns the product of the first `capture` <= factors factors for
    // lane `lane`. The first r = factors % 4 factors one per step, then four per step:
    // x <- montgomery(x, g) with g the Montgomery form of G = w (w + 1)(w + 2)(w + 3) for
    // w = a + r + 1 + 4j, stepped by its forward differences d1, d2, d3 (d4 = 4! 4^4). The start
    // values from G = z (z + 2), z = w (w + 3), and z(w + 4) = z(w) + 8w + 28. From 16 four-factor
    // steps on and for K <= 2, two accumulators take them in turn (x plain, y in Montgomery form):
    // the product chain is half as long. Fewer steps lose the final product's latency, K = 4
    // runs out of registers (Log).
    [[gnu::flatten]] static u32 lane_products(const Lanes& start, const Lanes& up_base, std::size_t factors,
                                              std::size_t capture, std::size_t lane, Lanes& product) {
        const std::size_t r = factors % 4;
        const auto c = [](u32 i) { return S::broadcast(montgomery_form(i)); };
        const auto times = [](V a, V b) { return M::template reduce<P>(M::montgomery(a, b)); };
        V x[K], y[K], w[K], g[K], d1[K], d2[K], d3[K];
        load(w, up_base.data());
        for (V& v : x) v = S::broadcast(1);
        for (V& v : y) v = S::broadcast(kR);
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
        const auto step = [&](V (&acc)[K]) {
            detail::unrolled<K>([&](auto v) {
                acc[v] = M::montgomery(acc[v], g[v]);
                g[v] = M::add(g[v], d1[v]), d1[v] = M::add(d1[v], d2[v]);
                d2[v] = M::add(d2[v], d3[v]), d3[v] = M::add(d3[v], d4);
            });
        };
        const bool two = K <= 2 && (factors - r) / 4 >= 16;
        const auto steps = [&](std::size_t count) {
            if (two) {
                for (std::size_t j = 0; j + 1 < count; j += 2) step(x), step(y);
                if (count % 2) step(x);
            } else {
                for (std::size_t j = 0; j < count; ++j) step(x);
            }
        };
        const auto store_product = [&] {
            if (two) detail::unrolled<K>([&](auto v) { x[v] = M::montgomery(x[v], y[v]), y[v] = S::broadcast(kR); });
            store(product, x);
        };
        // The capture: whole steps, then up to 3 factors for the one lane.
        const std::size_t whole = capture > r ? (capture - r) / 4 : 0;
        steps(whole);
        store_product();
        u32 partial = capture > r ? product[lane] : 1;
        for (std::size_t i = capture > r ? r + 4 * whole : 0; i < capture; ++i)
            partial = model::multiply<P>(partial, u32((start[lane] + i + 1) % P));
        steps((factors - r) / 4 - whole);
        store_product();
        return partial;
    }

    // Inclusive prefix (suffix) products over the W lanes, lane s = lane s % kLanes of x[s / kLanes],
    // Montgomery forms in [0, 2P): within each vector by log2(kLanes) shifted products, then
    // across the vectors (Hillis and Steele).
    static void prefix(V (&x)[K]) {
        const V one = S::broadcast(kR);
        detail::unrolled<std::bit_width(unsigned(kLanes)) - 1>([&](auto i) {
            detail::unrolled<K>([&](auto v) {
                x[v] = M::montgomery(x[v], shift_up<S, (1 << decltype(i)::value)>(x[v], one));
            });
        });
        detail::unrolled<std::bit_width(unsigned(K)) - 1>([&](auto i) {
            constexpr int span = 1 << decltype(i)::value;
            detail::unrolled<K - span>([&](auto u) {  // v from K - 1 down to span: reads x[v - span] before it changes
                constexpr int v = K - 1 - decltype(u)::value;
                x[v] = M::montgomery(x[v], broadcast_lane<S, kLanes - 1>(x[v - span]));
            });
        });
    }
    static void suffix(V (&x)[K]) {
        const V one = S::broadcast(kR);
        detail::unrolled<std::bit_width(unsigned(kLanes)) - 1>([&](auto i) {
            detail::unrolled<K>([&](auto v) {
                x[v] = M::montgomery(x[v], shift_down<S, (1 << decltype(i)::value)>(x[v], one));
            });
        });
        detail::unrolled<std::bit_width(unsigned(K)) - 1>([&](auto i) {
            constexpr int span = 1 << decltype(i)::value;
            detail::unrolled<K - span>([&](auto v) { x[v] = M::montgomery(x[v], broadcast_lane<S, 0>(x[v + span])); });
        });
    }

    // 1 / x for x in [1, P): binary extended Euclid, without branches but the exit. Invariants:
    // a, b odd, a 2^t = u x and b 2^t = v x (mod P), |u|, |v| <= 2^t, a b 2^t <= x P < 2^60. A step
    // keeps min(a, b) and replaces the other by |a - b| / 2^s (s >= 1 trailing zeros); the halving
    // goes to the other coefficient as a doubling (Pornin's deferral), so u, v stay integers. At
    // a = b = 1, 1 / x = u 2^-t. 40-50 ns against 85-110 for x^(P - 2) (Log).
    static u32 inverse(u32 x) {
        using i64 = std::int64_t;
        int t = std::countr_zero(x);
        u64 a = x >> t, b = P;
        i64 u = 1, v = 0;
        for (;;) {
            const i64 d = i64(a) - i64(b);
            if (d == 0) break;
            const int s = std::countr_zero(u64(d));
            const i64 less = d >> 63;  // all ones if a < b
            const i64 low = v ^ ((u ^ v) & less);  // min(a, b)'s coefficient
            b = std::min(a, b);
            a = u64((d ^ less) - less) >> s;
            u = ((u - v) ^ less) - less;
            v = low * (i64(1) << s);
            t += s;
        }
        const i64 r = u % i64(P);
        return model::multiply<P>(u32(r < 0 ? r + P : r), kHalves[t]);
    }

    // first[s] = (a_s)! and down[s] = 1 / (a_s + c - 1)!, from product[s] = (a_s + 1) ... (a_s + c - 1)
    // and partial = (a_(k-1) + 1) ... (n - c). In Montgomery form: first[s] = h! g_0 ... g_(s-1),
    // g_s = (a_s + 1) ... (a_(s+1)) = product[s] (a_s + c) for s < k - 1, partial for s = k - 1 and
    // 1 beyond. last[s] = first[s] product[s]; for s < k, 1 / last[s] = 1 / last[k-1] times
    // r_(s+1) ... r_(k-1), r_t = a_t product[t]. The lanes s >= k all end at n - 1: 1 / last[k].
    // One scalar exponentiation inverts last[k-1] last[k].
    [[gnu::flatten]] static void starts(const Lanes& start, const Lanes& product, u32 partial, std::size_t c,
                                        std::size_t h, std::size_t k, Lanes& first, Lanes& down) {
        std::array<u32, kWidth + 1> g, r;  // g[s + 1] = g_s, g[0] = h!; r[s] = r_s, r[kWidth] = 1
        g[0] = montgomery_form(kSmall[0][h]);
        r[kWidth] = kR;
        detail::unrolled<K>([&](auto v) {
            const V a = S::load(start.data() + v * kLanes), p = S::load(product.data() + v * kLanes);
            const V r3 = S::broadcast(kR3);  // montgomery(x, r3) = x 2^64 mod P
            S::store(g.data() + 1 + v * kLanes, M::montgomery(p, M::montgomery(S::add(a, S::broadcast(u32(c))), r3)));
            S::store(r.data() + v * kLanes, M::montgomery(p, M::montgomery(a, r3)));
        });
        if (k < kWidth) {
            g[k] = montgomery_form(partial);
            for (std::size_t s = k + 1; s < kWidth; ++s) g[s] = kR;
            for (std::size_t s = k; s < kWidth; ++s) r[s] = kR;
        }
        V f[K], e[K], last[K];
        load(f, g.data());
        load(e, r.data() + 1);
        prefix(f);
        suffix(e);
        detail::unrolled<K>([&](auto v) {
            const V p = S::load(product.data() + v * kLanes);
            last[v] = M::montgomery(f[v], p);
            f[v] = M::montgomery(f[v], S::broadcast(1));
        });
        store(first, f);
        Lanes ends;
        store(ends, last);
        // 1 / last[k-1] and 1 / last[k] from 1 / (last[k-1] last[k]).
        const u32 below = ends[k - 1], above = k < kWidth ? ends[k] : 1;
        const u32 product_inverse = inverse(model::multiply<P>(below, above));
        const u32 below_inverse = model::multiply<P>(product_inverse, above);
        detail::unrolled<K>([&](auto v) { e[v] = M::montgomery(e[v], S::broadcast(below_inverse)); });
        store(down, e);
        if (k < kWidth) {
            const u32 above_inverse = model::multiply<P>(product_inverse, below);
            for (std::size_t s = k; s < kWidth; ++s) down[s] = above_inverse;
        }
    }

    // The up and down chains over c steps, kLanes steps per block. Block b of a chain holds the
    // terms of its next kLanes steps; while the chains fill one block, the other is transposed
    // and stored. The down chain records each block in reverse, so it is stored in position order.
    // A chain step is x <- x m, x < 2P. Step::montgomery: m in Montgomery form, base + u with
    // u = j 2^32 mod P shared by all lanes. Step::shoup: x m - floor(x q / 2^32) P with m plain and
    // q = floor(m_0 2^32 / P) + floor(j 2^32 / P) (down: minus, and 1 less), one below
    // floor(m 2^32 / P) at most; with x < 2^31 the result is still in [0, 2P).
    [[gnu::flatten]] static void scan(u32* fact, u32* inverse, std::size_t c, const Lanes& start, const Lanes& first,
                                      const Lanes& down) {
        V up[K], dn[K], ub[K], db[K], uq[K], dq[K];
        load(up, first.data()), load(dn, down.data());
        const V neg_pinv = S::broadcast(u32(0) - M::kPinv);
        detail::unrolled<K>([&](auto v) {
            const V a = S::load(start.data() + v * kLanes);
            const V u = S::add(a, S::broadcast(1)), d = S::add(a, S::broadcast(u32(c - 1)));
            if constexpr (kStep == Step::montgomery) {
                ub[v] = to_montgomery(u);
                db[v] = S::add(to_montgomery(d), S::broadcast(P));  // m = db - u is in [1, 2P)
            } else {
                ub[v] = u, db[v] = d;
                // floor(m 2^32 / P) = (m 2^32 - (m 2^32 mod P)) / P = -(m 2^32 mod P) P^-1 mod 2^32
                uq[v] = S::mul_lo(to_montgomery(u), neg_pinv);
                dq[v] = S::sub(S::mul_lo(to_montgomery(d), neg_pinv), S::broadcast(1));
            }
        });
        u32* up_out[kWidth];
        u32* down_out[kWidth];
        for (int s = 0; s < kWidth; ++s) up_out[s] = fact + start[s], down_out[s] = inverse + start[s] + c - kLanes;
        V up_block[2][K][kLanes], down_block[2][K][kLanes];
        V u = S::broadcast(0), j = S::broadcast(0), qj = S::broadcast(0);  // u = j 2^32 mod P
        const V r = S::broadcast(kR), p = S::broadcast(P);
        const auto shoup = [p](V x, V m, V q) { return S::mul_sub(S::mul_lo(x, m), S::mulhi(x, q), p); };
        const auto advance = [&](int b, auto t) {
            detail::unrolled<K>([&](auto v) {
                up_block[b][v][t] = up[v];
                down_block[b][v][kLanes - 1 - t] = dn[v];
                if constexpr (kStep == Step::montgomery) {
                    up[v] = M::montgomery(up[v], S::add(ub[v], u));
                    dn[v] = M::montgomery(dn[v], S::sub(db[v], u));
                } else {
                    up[v] = shoup(up[v], S::add(ub[v], j), S::add(uq[v], qj));
                    dn[v] = shoup(dn[v], S::sub(db[v], j), S::sub(dq[v], qj));
                }
            });
            u = M::add(u, r);
            if constexpr (kStep == Step::shoup) j = S::add(j, S::broadcast(1)), qj = S::mul_lo(u, neg_pinv);
        };
        // Step t of a block stores lanes t K .. t K + K - 1; a vector is transposed at its first lane.
        const auto emit = [&](int b, std::size_t i, auto t) {
            detail::unrolled<K>([&](auto l) {
                constexpr int e = decltype(t)::value * K + decltype(l)::value, v = e / kLanes, lane = e % kLanes;
                if constexpr (lane == 0) transpose(up_block[b][v]), transpose(down_block[b][v]);
                S::store(up_out[e] + i, M::template reduce<P>(up_block[b][v][lane]));
                S::store(down_out[e] - i, M::template reduce<P>(down_block[b][v][lane]));
            });
        };
        detail::unrolled<kLanes>([&](auto t) { advance(0, t); });
        std::size_t i = 0;
        int cur = 0;
        for (; i + kLanes < c; i += kLanes, cur ^= 1)
            detail::unrolled<kLanes>([&](auto t) { advance(cur ^ 1, t), emit(cur, i, t); });
        detail::unrolled<kLanes>([&](auto t) { emit(cur, i, t); });
    }
};

// fact[i] = i!, inverse[i] = 1 / i! for i < n = fact.size() = inverse.size() <= P.
template <u32 P>
void tables(std::span<u32> fact, std::span<u32> inverse) {
    const std::size_t n = fact.size();
    [&]<class... T>(Tiers<T...>) {
        static_assert(((T::min_size >= Chains<typename T::Isa, T::vectors, P, T::scan>::kMinSize) && ...));
        const int tier = ((n >= T::min_size) + ...) - 1;  // the last tier with min_size <= n
        if (tier < 0) return model::tables<P>(fact.data(), inverse.data(), n);
        int t = 0;
        ((t++ == tier ? Chains<typename T::Isa, T::vectors, P, T::scan>::run(fact.data(), inverse.data(), n) : void()),
         ...);
    }(Selected{});
}

}  // namespace fps::factorials
