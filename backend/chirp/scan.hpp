// Helpers of geometric.hpp and shift.hpp: prefix products and their inverses, pointwise
// passes, and per-thread scratch memory.
//
//   detail::products(u, n, prefix, sp, inverse, si);  // prefix[k] = sp u(0) .. u(k - 1),
//                                                       // inverse[k] = si / (u(0) .. u(k - 1)), k <= n
//   detail::multiply(f, g, out, n, c);                 // out[k] = f[k] g[k] c
//   detail::Scratch::get(words)                        // this thread's scratch, grown as needed
//
// Factors: Arithmetic (u(s) = u0 + s) and Geometric (u(s) = a + b z^s). products() runs the
// lanes (products_lanes) from kLanesMinSize words on, else interleaved scalar chains
// (products_scalar); both are public for tests and benches.
//
// products_lanes<S, K>: W = K S::lanes lanes, lane l walks block [l C, (l + 1) C) (C a multiple of
// S::lanes, W C <= n); every S::lanes steps an in-register transpose (backend/factorials) turns
// the lanes' values into vectors of consecutive positions for the stores. Then each block's
// prefixes get the product of the blocks before it (a vector pass), the positions [W C, n) run
// in a scalar chain, and the inverses run down the same lanes from the blocks' tops, which come
// from one inverse and the blocks' products. The factors are kept times 2^32 (Montgomery form).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "backend/chirp/chirp.hpp"
#include "backend/factorials/transpose.hpp"
#include "backend/ntt/memory.hpp"

namespace fps::chirp::detail {

inline constexpr u32 kR = u32((u64(1) << 32) % kP);  // 2^32 mod P

// Factors u(s) = u0 + s mod P.
struct Arithmetic {
    u32 u0;

    u32 at(std::size_t s) const { return u32((u0 + s) % kP); }

    struct Cursor {
        u32 v;
        u32 up() {  // u(s), then s + 1
            const u32 r = v;
            v = v + 1 == kP ? 0 : v + 1;
            return r;
        }
        u32 down() {  // s - 1, then u(s)
            v = v == 0 ? kP - 1 : v - 1;
            return v;
        }
    };
    Cursor cursor(std::size_t s) const { return {at(s)}; }

    // Lane i at position starts[i]; value() is u(s) 2^32 mod P, canonical.
    template <class S>
    struct Lanes {
        using M = modarith::Ops<S, kP>;
        typename S::V u;
        Lanes(const Arithmetic& g, const std::size_t* starts) {
            u32 w[S::lanes];
            for (int i = 0; i < S::lanes; ++i) w[i] = mul(g.at(starts[i]), kR);
            u = S::load(w);
        }
        typename S::V value() const { return u; }
        void up() { u = M::add(u, S::broadcast(kR)); }
        void down() { u = M::sub(u, S::broadcast(kR)); }
    };
};

// Factors u(s) = a + b z^s mod P, z != 0.
struct Geometric {
    u32 a, b, z, z_inverse;
    Geometric(u32 a, u32 b, u32 z) : a(a), b(b), z(z), z_inverse(inverse(z)) {}

    u32 at(std::size_t s) const {
        const u32 w = mul(b, power_of(z, s));
        return a + w >= kP ? a + w - kP : a + w;
    }

    struct Cursor {
        u32 a, w, z, z_inverse;  // w = b z^s
        u32 up() {
            const u32 r = a + w >= kP ? a + w - kP : a + w;
            w = mul(w, z);
            return r;
        }
        u32 down() {
            w = mul(w, z_inverse);
            return a + w >= kP ? a + w - kP : a + w;
        }
    };
    Cursor cursor(std::size_t s) const { return {a, mul(b, power_of(z, s)), z, z_inverse}; }

    template <class S>
    struct Lanes {
        using M = modarith::Ops<S, kP>;
        using V = typename S::V;
        V a, w, z, zq, zi, ziq;  // w = b z^s 2^32, canonical
        Lanes(const Geometric& g, const std::size_t* starts)
            : a(S::broadcast(mul(g.a, kR))), z(S::broadcast(g.z)), zq(S::broadcast(M::shoup_quotient(g.z))),
              zi(S::broadcast(g.z_inverse)), ziq(S::broadcast(M::shoup_quotient(g.z_inverse))) {
            u32 t[S::lanes];
            for (int i = 0; i < S::lanes; ++i) t[i] = mul(mul(g.b, power_of(g.z, starts[i])), kR);
            w = S::load(t);
        }
        V value() const { return M::add(a, w); }
        void up() { w = M::template reduce<kP>(M::shoup(w, z, zq)); }
        void down() { w = M::template reduce<kP>(M::shoup(w, zi, ziq)); }
    };
};

// f[k] = f[k] c for k < n.
inline void multiply_by(u32* f, std::size_t n, u32 c) {
    using S = Wide;
    using M = modarith::Ops<S, kP>;
    const typename S::V factor = S::broadcast(mul(c, kR));
    std::size_t k = 0;
    for (; k + S::lanes <= n; k += S::lanes)
        S::store(f + k, M::template reduce<kP>(M::montgomery(S::load(f + k), factor)));
    for (; k < n; ++k) f[k] = mul(f[k], c);
}

// prefix[k] = sp u(0) u(1) .. u(k - 1) and, if inverse_prefix is not null, inverse_prefix[k] =
// si / (u(0) .. u(k - 1)) for k <= n, canonical; the inverses need every u(s) != 0, s < n (else
// they are garbage). kChains blocks run interleaved, so their products overlap; each block's
// prefixes then get the product of the blocks before it (a vector pass).
template <class U>
void products_scalar(const U& u, std::size_t n, u32* prefix, u32 sp, u32* inverse_prefix, u32 si) {
    constexpr int kChains = 8;
    const int chains = n >= 64 * kChains ? kChains : 1;
    const std::size_t chunk = n / chains;
    std::array<std::size_t, kChains> start{};
    for (int b = 0; b < chains; ++b) start[b] = b * chunk;
    // Block b: [start[b], start[b] + chunk), the last block to n.
    std::array<typename U::Cursor, kChains> cursor;
    std::array<u32, kChains> local;
    for (int b = 0; b < chains; ++b) cursor[b] = u.cursor(start[b]), local[b] = b == 0 ? sp : 1;
    for (std::size_t j = 0; j < chunk; ++j)
        for (int b = 0; b < chains; ++b) {
            prefix[start[b] + j] = local[b];
            local[b] = mul(local[b], cursor[b].up());
        }
    for (std::size_t k = start[chains - 1] + chunk; k < n; ++k) {
        prefix[k] = local[chains - 1];
        local[chains - 1] = mul(local[chains - 1], cursor[chains - 1].up());
    }
    // local[b]: block b's product (times sp for b = 0). Carry them into the later blocks.
    u32 carry = local[0];
    for (int b = 1; b < chains; ++b) {
        const std::size_t end = b + 1 < chains ? start[b] + chunk : n;
        multiply_by(prefix + start[b], end - start[b], carry);
        carry = mul(carry, local[b]);
    }
    prefix[n] = carry;
    if (!inverse_prefix) return;
    // inverse_prefix at the end of block b = si / (u(0) .. u(end - 1)): from si sp / prefix[n] times
    // the products of the blocks after b. Then down each block: inverse[k] = inverse[k + 1] u(k).
    std::array<u32, kChains> top{};
    u32 suffix = mul(mul(si, sp), inverse(carry));
    for (int b = chains; b-- > 0;) {
        top[b] = suffix;
        suffix = mul(suffix, local[b]);  // local[0] carries sp; the last product is unused
    }
    inverse_prefix[n] = top[chains - 1];
    for (int b = 0; b < chains; ++b) cursor[b] = u.cursor(b + 1 < chains ? start[b] + chunk : n), local[b] = top[b];
    for (std::size_t k = n; k-- > start[chains - 1] + chunk;) {
        local[chains - 1] = mul(local[chains - 1], cursor[chains - 1].down());
        inverse_prefix[k] = local[chains - 1];
    }
    for (std::size_t j = chunk; j-- > 0;)
        for (int b = 0; b < chains; ++b) {
            local[b] = mul(local[b], cursor[b].down());
            inverse_prefix[start[b] + j] = local[b];
        }
}

// products_scalar's contract, in W = K S::lanes lanes (above). n >= W S::lanes.
template <class S, int K, class U>
void products_lanes(const U& u, std::size_t n, u32* prefix, u32 sp, u32* inverse_prefix, u32 si) {
    using M = modarith::Ops<S, kP>;
    using V = typename S::V;
    using G = typename U::template Lanes<S>;
    constexpr int L = S::lanes, W = K * L;
    const std::size_t c = n / W / L * L, covered = W * c;  // c >= L
    std::size_t starts[W];
    for (int l = 0; l < W; ++l) starts[l] = l * c;
    // Up the lanes: x = sp (lane 0) or 1 times u(l c) .. u(l c + j - 1); stores of x before each step.
    const auto lanes = [&u]<int... v>(const std::size_t* at, std::integer_sequence<int, v...>) {
        return std::array<G, K>{G(u, at + v * L)...};
    };
    V x[K];
    auto gen = lanes(starts, std::make_integer_sequence<int, K>{});
    u32 ones[W];
    for (int l = 0; l < W; ++l) ones[l] = l == 0 ? sp : 1;
    detail::unrolled<K>([&](auto v) { x[v] = S::load(ones + v * L); });
    for (std::size_t j = 0; j < c; j += L) {
        V tile[K][L];
        for (int t = 0; t < L; ++t)
            detail::unrolled<K>([&](auto v) {
                tile[v][t] = x[v];
                x[v] = M::montgomery(x[v], gen[v].value());
                gen[v].up();
            });
        detail::unrolled<K>([&](auto v) {
            factorials::transpose(tile[v]);
            for (int i = 0; i < L; ++i) S::store(prefix + starts[v * L + i] + j, M::template reduce<kP>(tile[v][i]));
        });
    }
    u32 total[W];  // block l's product, times sp for l = 0
    detail::unrolled<K>([&](auto v) { S::store(total + v * L, M::template reduce<kP>(x[v])); });
    u32 carry = total[0];
    for (int l = 1; l < W; ++l) {
        multiply_by(prefix + starts[l], c, carry);
        carry = mul(carry, total[l]);
    }
    auto tail = u.cursor(covered);
    for (std::size_t k = covered; k < n; ++k) prefix[k] = carry, carry = mul(carry, tail.up());
    prefix[n] = carry;
    if (!inverse_prefix) return;
    // Down: the tail, then the lanes from their blocks' tops.
    u32 top = mul(mul(si, sp), inverse(carry));
    inverse_prefix[n] = top;
    for (std::size_t k = n; k-- > covered;) top = mul(top, tail.down()), inverse_prefix[k] = top;
    u32 tops[W];
    for (int l = W; l-- > 0;) tops[l] = top, top = mul(top, total[l]);
    std::size_t ends[W];
    for (int l = 0; l < W; ++l) ends[l] = starts[l] + c;
    gen = lanes(ends, std::make_integer_sequence<int, K>{});
    detail::unrolled<K>([&](auto v) { x[v] = S::load(tops + v * L); });
    for (std::size_t j = c; j > 0; j -= L) {
        V tile[K][L];
        for (int t = L; t-- > 0;)
            detail::unrolled<K>([&](auto v) {
                gen[v].down();
                x[v] = M::montgomery(x[v], gen[v].value());
                tile[v][t] = x[v];
            });
        detail::unrolled<K>([&](auto v) {
            factorials::transpose(tile[v]);
            for (int i = 0; i < L; ++i)
                S::store(inverse_prefix + starts[v * L + i] + j - L, M::template reduce<kP>(tile[v][i]));
        });
    }
}

template <class U>
void products(const U& u, std::size_t n, u32* prefix, u32 sp, u32* inverse_prefix, u32 si) {
    constexpr std::size_t kMin = std::size_t(kLanesVectors) * Wide::lanes * Wide::lanes;
    if (n >= std::max(kMin, kLanesMinSize))
        return products_lanes<Wide, kLanesVectors>(u, n, prefix, sp, inverse_prefix, si);
    products_scalar(u, n, prefix, sp, inverse_prefix, si);
}

// out[k] = f[k] g[k] c for k < n; out may be f or g.
inline void multiply(const u32* f, const u32* g, u32* out, std::size_t n, u32 c = 1) {
    using S = Wide;
    using M = modarith::Ops<S, kP>;
    const typename S::V factor = S::broadcast(mul(mul(c, kR), kR));
    std::size_t k = 0;
    for (; k + S::lanes <= n; k += S::lanes)
        S::store(out + k, M::template reduce<kP>(M::montgomery(M::montgomery(S::load(f + k), factor), S::load(g + k))));
    for (; k < n; ++k) out[k] = mul(mul(f[k], g[k]), c);
}

// This thread's scratch: one region, grown as needed, kept until the thread exits.
class Scratch {
public:
    static u32* get(std::size_t words) {
        thread_local Scratch scratch;
        const std::size_t bytes = words * sizeof(u32);
        if (scratch.region_.bytes < bytes) {
            ntt::detail::unmap_region(std::exchange(scratch.region_, {}));
            scratch.region_ = ntt::detail::map_region(bytes);
        }
        return reinterpret_cast<u32*>(scratch.region_.start);
    }

    ~Scratch() { ntt::detail::unmap_region(region_); }

private:
    ntt::detail::Region region_;
};

}  // namespace fps::chirp::detail
