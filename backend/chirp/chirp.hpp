// Chirps modulo P = 998244353 (backend kernel): x_k = c s^k q^t(k) with t(k) = k (k - 1) / 2,
// the pointwise factors of the chirp z-transform. Evaluation and interpolation on geometric
// sequences and the shift of sampling points are built on them (geometric.hpp, shift.hpp).
//
//   fps::chirp::chirp(c, s, q, out);              // out[k] = x_k for k < out.size()
//   fps::chirp::multiply_chirp(c, s, q, f, out);  // out[k] = f[k] x_k; f.size() == out.size(), out may be f
//   fps::chirp::multiply_chirp(c, s, q, f);       // f[k] = f[k] x_k
//
// c, s, q and f's words in [0, P); out canonical. Spans of any size and alignment. 0^0 = 1, so
// s = 0 gives x_0 = c and x_k = 0 after it.
//
// Terms<S, K> (the fast path for fused passes): x_k, W = K S::lanes at a time, in K vectors of S
// (select.hpp picks S = simd::Native, and K by size): after j calls of next(), terms[v] holds
// x_(W j + v S::lanes + l) in lane l, in [0, 2P). x_(k+W) = x_k g_k with g_k = s^W q^(W k + t(W))
// and g_(k+W) = g_k q^(W^2); g is kept times 2^32, so a step is one Montgomery product (x) and
// one Shoup product (g) per vector. Pass c 2^32 mod P as c for factors of montgomery(f, x_k) = f x
// (multiply_chirp does). From lc-opt's lib/poly/chirp.hpp (AVX2, K = 4), on backend/simd.
// Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

#include "backend/chirp/select.hpp"
#include "backend/modarith/modarith.hpp"

namespace fps::chirp {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 kP = 998244353;

namespace detail {

constexpr u32 mul(u32 a, u32 b, u32 p = kP) { return u32(u64(a) * b % p); }

// x^e mod p for any e >= 0 (0^0 = 1).
constexpr u32 power(u32 x, u64 e, u32 p = kP) {
    u32 r = 1;
    for (; e; e >>= 1, x = mul(x, x, p))
        if (e & 1) r = mul(r, x, p);
    return r;
}

// x^e mod p for x != 0 and any e: exponents reduce mod p - 1.
constexpr u32 power_of(u32 x, u64 e, u32 p = kP) { return power(x, e % (p - 1), p); }

constexpr u32 inverse(u32 x, u32 p = kP) { return power(x, p - 2, p); }

// t(k) = k (k - 1) / 2 mod (p - 1), the exponent of q in x_k, for k < 2^63.
constexpr u64 triangle(u64 k, u32 p = kP) {
    const u64 a = k % 2 ? k : k / 2, b = k % 2 ? (k - 1) / 2 : k - 1;
    return mul(u32(a % (p - 1)), u32(b % (p - 1)), p - 1);
}

// f(std::integral_constant<int, i>{}) for i = 0 .. N - 1, unrolled.
template <int N, class F>
[[gnu::always_inline]] inline void unrolled(F&& f) {
    [&]<int... i>(std::integer_sequence<int, i...>) {
        (f(std::integral_constant<int, i>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

}  // namespace detail

template <class S, int K, u32 P = kP>
class Terms {
public:
    using M = modarith::Ops<S, P>;
    using V = typename S::V;
    static constexpr int kVectors = K, kLanes = S::lanes, kWidth = K * S::lanes;
    static_assert(kWidth <= 256);

    Terms(u32 c, u32 s, u32 q) {
        using detail::mul;
        using detail::power;
        constexpr u32 R = u32((u64(1) << 32) % P);
        constexpr u64 W = kWidth;
        const u32 qw = power(q, W, P);
        u32 term = c, ratio = s;  // ratio = x_(k+1) / x_k = s q^k
        u32 g_k = mul(mul(power(s, W, P), power(q, W * (W - 1) / 2, P), P), R, P);
        for (int k = 0; k < kWidth; ++k) {
            x_[k] = term, g_[k] = g_k;
            term = mul(term, ratio, P), ratio = mul(ratio, q, P), g_k = mul(g_k, qw, P);
        }
        step_ = power(qw, W, P);
    }

    V operator[](int v) const { return S::load(x_ + v * kLanes); }

    void next() {
        State t = state();
        t.next();
        for (int v = 0; v < K; ++v) S::store(x_ + v * kLanes, t.x[v]), S::store(g_ + v * kLanes, t.g[v]);
    }

    // The state as locals of a pass: arrays of an object are reloaded after every store of a
    // vector (stores may alias them), locals are not. The object holds words, not vectors: GCC's
    // ASan stack frames misalign 64-byte vectors (avx512, notes.md).
    struct State {
        V x[K], g[K], step, step_quotient;

        [[gnu::always_inline]] void next() {
            detail::unrolled<K>([&](auto v) {
                x[v] = M::montgomery(x[v], g[v]);
                g[v] = M::shoup(g[v], step, step_quotient);
            });
        }
    };

    [[gnu::always_inline]] State state() const {
        State t;
        for (int v = 0; v < K; ++v) t.x[v] = S::load(x_ + v * kLanes), t.g[v] = S::load(g_ + v * kLanes);
        t.step = S::broadcast(step_), t.step_quotient = S::broadcast(M::shoup_quotient(step_));
        return t;
    }

private:
    u32 x_[kWidth], g_[kWidth], step_;
};

namespace detail {

// out[k] = put(x_k, in[k]) for k < n, a vector at a time; with kRead false in is not read (put
// gets zeros). in may be out. put returns canonical words.
template <bool kRead, class S, int K, u32 P, class Put>
[[gnu::always_inline]] inline void pass(const Terms<S, K, P>& terms, const u32* in, u32* out, std::size_t n, Put put) {
    using T = Terms<S, K, P>;
    using V = typename S::V;
    constexpr int L = T::kLanes;
    constexpr std::size_t W = T::kWidth;
    auto t = terms.state();
    std::size_t k = 0;
    for (; k + W <= n; k += W) {
        unrolled<K>([&](auto v) {
            V a{};
            if constexpr (kRead) a = S::load(in + k + v * L);
            S::store(out + k + v * L, put(t.x[v], a));
        });
        t.next();
    }
    for (int v = 0; v < K && k < n; ++v, k += L) {
        const int count = int(std::min<std::size_t>(L, n - k));
        u32 rest[L] = {};
        if constexpr (kRead) std::copy(in + k, in + k + count, rest);
        S::store_first(out + k, put(t.x[v], S::load(rest)), count);
    }
}

}  // namespace detail

// out[k] = c s^k q^t(k), with Terms<S, K>; K = 0: kSmallVectors, or kLargeVectors from
// kLargeSize terms on (select.hpp).
template <class S = Wide, int K = 0>
void chirp(u32 c, u32 s, u32 q, std::span<u32> out) {
    if constexpr (K == 0) {
        if (out.size() >= kLargeSize) return chirp<S, kLargeVectors>(c, s, q, out);
        return chirp<S, kSmallVectors>(c, s, q, out);
    } else {
        using M = modarith::Ops<S, kP>;
        detail::pass<false>(Terms<S, K>(c, s, q), nullptr, out.data(), out.size(),
                            [](typename S::V x, typename S::V) { return M::template reduce<kP>(x); });
    }
}

// out[k] = f[k] c s^k q^t(k); out may be f. K as for chirp.
template <class S = Wide, int K = 0>
void multiply_chirp(u32 c, u32 s, u32 q, std::span<const u32> f, std::span<u32> out) {
    if (f.size() != out.size()) throw std::invalid_argument("fps::chirp::multiply_chirp: f and out differ in size");
    if constexpr (K == 0) {
        if (out.size() >= kLargeSize) return multiply_chirp<S, kLargeVectors>(c, s, q, f, out);
        return multiply_chirp<S, kSmallVectors>(c, s, q, f, out);
    } else {
        using M = modarith::Ops<S, kP>;
        constexpr u32 R = u32((u64(1) << 32) % kP);
        detail::pass<true>(Terms<S, K>(detail::mul(c, R), s, q), f.data(), out.data(), out.size(),
                           [](typename S::V x, typename S::V a) {
                               return M::template reduce<kP>(M::montgomery(a, x));
                           });
    }
}

template <class S = Wide, int K = 0>
void multiply_chirp(u32 c, u32 s, u32 q, std::span<u32> f) {
    multiply_chirp<S, K>(c, s, q, std::span<const u32>(f), f);
}

}  // namespace fps::chirp
