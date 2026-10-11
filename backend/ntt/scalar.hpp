// Variant scalar: the NTT in plain C++, every value canonical. It runs on every target and is the
// model the tests hold the other variants to: same splitting, same leaves (ntt.hpp, "Leaf form").
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "backend/ntt/common.hpp"

namespace fps::ntt::detail::scalar {

using u32 = std::uint32_t;

inline u32 add(u32 x, u32 y) {
    const u32 s = x + y;  // < 2P < 2^31
    return s >= kP ? s - kP : s;
}
inline u32 sub(u32 x, u32 y) { return x >= y ? x - y : x + kP - y; }
inline u32 mul(u32 x, u32 y) { return multiply_mod(x, y); }

// Words per quarter of the groups on the first radix-4 level.
inline std::size_t top_quarter(int lg) { return std::size_t(1) << (lg - (has_radix2(lg) ? 3 : 2)); }

// Group k of a level with quarters of q words is f[4qk, 4q(k + 1)): a polynomial mod X^4 - r(k)^2,
// X = x^q, quarter t the coefficient of X^t. Its children 4k + t are the residues mod X - r(2k),
// X + r(2k), X - r(2k + 1), X + r(2k + 1); quarters of 8 words are the leaves.
inline void forward(u32* f, int lg, const u32* r) {
    const std::size_t len = std::size_t(1) << lg;
    if (has_radix2(lg)) {
        for (std::size_t i = 0; i < len / 2; ++i) {
            const u32 x = f[i], y = f[i + len / 2];
            f[i] = add(x, y), f[i + len / 2] = sub(x, y);
        }
    }
    for (std::size_t q = top_quarter(lg); q >= 8; q /= 4) {
        for (std::size_t k = 0; k < len / (4 * q); ++k) {
            const u32 x = r[slot(k)], y = r[slot(2 * k)], z = r[slot(2 * k + 1)];
            u32* g = f + 4 * q * k;
            for (std::size_t j = 0; j < q; ++j) {
                const u32 xc = mul(x, g[j + 2 * q]), xd = mul(x, g[j + 3 * q]);
                const u32 a = add(g[j], xc), c = sub(g[j], xc);
                const u32 yb = mul(y, add(g[j + q], xd)), zd = mul(z, sub(g[j + q], xd));
                g[j] = add(a, yb), g[j + q] = sub(a, yb), g[j + 2 * q] = add(c, zd), g[j + 3 * q] = sub(c, zd);
            }
        }
    }
}

// forward() undone, times nv = len / 8 (the scale is left to the caller).
inline void inverse_unscaled(u32* f, int lg, const u32* ir) {
    const std::size_t len = std::size_t(1) << lg;
    for (std::size_t q = 8; q <= top_quarter(lg); q *= 4) {
        for (std::size_t k = 0; k < len / (4 * q); ++k) {
            const u32 x = ir[slot(k)], y = ir[slot(2 * k)], z = ir[slot(2 * k + 1)];
            u32* g = f + 4 * q * k;
            for (std::size_t j = 0; j < q; ++j) {
                const u32 ab = add(g[j], g[j + q]), cd = add(g[j + 2 * q], g[j + 3 * q]);
                const u32 amb = mul(y, sub(g[j], g[j + q])), cmd = mul(z, sub(g[j + 2 * q], g[j + 3 * q]));
                g[j] = add(ab, cd), g[j + q] = add(amb, cmd);
                g[j + 2 * q] = mul(x, sub(ab, cd)), g[j + 3 * q] = mul(x, sub(amb, cmd));
            }
        }
    }
    if (has_radix2(lg)) {
        for (std::size_t i = 0; i < len / 2; ++i) {
            const u32 x = f[i], y = f[i + len / 2];
            f[i] = add(x, y), f[i + len / 2] = sub(x, y);
        }
    }
}

inline void scale(u32* f, std::size_t count, u32 s) {
    for (std::size_t i = 0; i < count; ++i) f[i] = mul(f[i], s);
}

// a = a b mod x^8 - w_v in every leaf v, w_v = (-1)^v r(v / 2).
inline void multiply_leaves(u32* a, const u32* b, int lg, const u32* r) {
    for (std::size_t v = 0; v < (std::size_t(1) << lg) / 8; ++v) {
        const u32 w = v % 2 ? sub(0, r[slot(v / 2)]) : r[slot(v / 2)];
        u32* x = a + 8 * v;
        const u32* y = b + 8 * v;
        u32 c[8];
        for (int l = 0; l < 8; ++l) {
            std::uint64_t low = 0, high = 0;  // each < 8 P^2 < 2^63
            for (int i = 0; i <= l; ++i) low += std::uint64_t(x[i]) * y[l - i];
            for (int i = l + 1; i < 8; ++i) high += std::uint64_t(x[i]) * y[l + 8 - i];
            c[l] = add(u32(low % kP), mul(w, u32(high % kP)));
        }
        std::copy(c, c + 8, x);
    }
}

}  // namespace fps::ntt::detail::scalar

namespace fps::ntt::variant {

struct scalar {
    // Entries [0, count) of the forward or inverse table.
    static void build_table(std::uint32_t* t, std::size_t count, bool inverse) {
        for (std::size_t c = 2 * detail::copy_static_table(t, count, inverse); c <= count; c *= 2)
            detail::extend_table(t, c, detail::kRoots[inverse]);
    }

    static void forward(std::uint32_t* f, int lg, const std::uint32_t* roots) { detail::scalar::forward(f, lg, roots); }

    static void inverse(std::uint32_t* f, int lg, const std::uint32_t* inverse_roots) {
        detail::scalar::inverse_unscaled(f, lg, inverse_roots);
        detail::scalar::scale(f, std::size_t(1) << lg, detail::leaf_scale(lg));
    }

    // Words [n, L) of a are zero, but the upper half of a factor of at most L / 2 coefficients may
    // hold anything: it is zeroed here, as the other variants do not read it. Same for b.
    static void multiply(std::uint32_t* a, std::uint32_t* b, std::size_t n, std::size_t m, int lg,
                         const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        const std::size_t len = std::size_t(1) << lg;
        if (2 * n <= len) std::fill(a + len / 2, a + len, 0u);
        if (2 * m <= len) std::fill(b + len / 2, b + len, 0u);
        forward(a, lg, roots);
        forward(b, lg, roots);
        detail::scalar::multiply_leaves(a, b, lg, roots);
        inverse(a, lg, inverse_roots);
    }
};

}  // namespace fps::ntt::variant
