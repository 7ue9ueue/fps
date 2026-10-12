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
inline void forward_group(u32* g, std::size_t q, std::size_t k, const u32* r) {
    const u32 x = r[slot(k)], y = r[slot(2 * k)], z = r[slot(2 * k + 1)];
    for (std::size_t j = 0; j < q; ++j) {
        const u32 xc = mul(x, g[j + 2 * q]), xd = mul(x, g[j + 3 * q]);
        const u32 a = add(g[j], xc), c = sub(g[j], xc);
        const u32 yb = mul(y, add(g[j + q], xd)), zd = mul(z, sub(g[j + q], xd));
        g[j] = add(a, yb), g[j + q] = sub(a, yb), g[j + 2 * q] = add(c, zd), g[j + 3 * q] = sub(c, zd);
    }
}

// The levels with quarters q = top .. 8 on f, which starts at group `first` of the level with
// quarters of `top` words (and spans `count` of them).
inline void forward_levels(u32* f, std::size_t top, std::size_t first, std::size_t count, const u32* r) {
    for (std::size_t q = top, n = count, k0 = first; q >= 8; q /= 4, n *= 4, k0 *= 4)
        for (std::size_t k = 0; k < n; ++k) forward_group(f + 4 * q * k, q, k0 + k, r);
}

inline void forward(u32* f, int lg, const u32* r) {
    const std::size_t len = std::size_t(1) << lg;
    if (has_radix2(lg)) {
        for (std::size_t i = 0; i < len / 2; ++i) {
            const u32 x = f[i], y = f[i + len / 2];
            f[i] = add(x, y), f[i + len / 2] = sub(x, y);
        }
    }
    forward_levels(f, top_quarter(lg), 0, len / (4 * top_quarter(lg)), r);
}

// The upper half of the leaf form of length 2L, L = 2^lg, from f = p mod (x^L + 1) (canonical);
// r: the table of length 2L. Below a radix-2 level the upper half is group 1; below group 0 (radix
// 4) groups 2 and 3, the residues mod x^(L/2) -+ r(1).
inline void forward_upper(u32* f, int lg, const u32* r) {
    const std::size_t len = std::size_t(1) << lg;
    if (has_radix2(lg + 1)) return forward_levels(f, len / 4, 1, 1, r);
    const u32 i1 = r[slot(1)];
    for (std::size_t j = 0; j < len / 2; ++j) {
        const u32 x = f[j], y = mul(i1, f[j + len / 2]);
        f[j] = add(x, y), f[j + len / 2] = sub(x, y);
    }
    forward_levels(f, len / 8, 2, 2, r);
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

// out = a b mod x^8 - w_v in every leaf v, w_v = (-1)^v r(v / 2); out += a b if accumulate. out
// may be a or b.
inline void multiply_leaves(const u32* a, const u32* b, u32* out, int lg, const u32* r, bool accumulate = false) {
    for (std::size_t v = 0; v < (std::size_t(1) << lg) / 8; ++v) {
        const u32 w = v % 2 ? sub(0, r[slot(v / 2)]) : r[slot(v / 2)];
        const u32 *x = a + 8 * v, *y = b + 8 * v;
        u32 c[8];
        for (int l = 0; l < 8; ++l) {
            std::uint64_t low = 0, high = 0;  // each < 8 P^2 < 2^63
            for (int i = 0; i <= l; ++i) low += std::uint64_t(x[i]) * y[l - i];
            for (int i = l + 1; i < 8; ++i) high += std::uint64_t(x[i]) * y[l + 8 - i];
            c[l] = add(u32(low % kP), mul(w, u32(high % kP)));
        }
        for (int l = 0; l < 8; ++l) out[8 * v + l] = accumulate ? add(out[8 * v + l], c[l]) : c[l];
    }
}
inline void multiply_leaves(u32* a, const u32* b, int lg, const u32* r) { multiply_leaves(a, b, a, lg, r); }

// The values form (common.hpp) of the leaf form at `from`, into `to` (may be from); rows:
// build_value_rows(lg).
inline void split(const u32* from, u32* to, int lg, const u32* rows) {
    const u32* z = kStaticTables.words[0];  // z^k = r(eighth_root(k)), k < 4
    for (std::size_t g = 0; g < (std::size_t(1) << lg) / 64; ++g) {
        u32 c[64];
        std::copy_n(from + 64 * g, 64, c);
        const u32* a = rows + 32 * (g / 8);
        const auto& b = kValueLanes.forward[g % 8];
        for (int t = 0; t < 8; ++t) {
            u32 x[8];
            for (int k = 0; k < 8; ++k) x[k] = mul(mul(c[8 * t + k], b[k][t]), a[k]);
            for (int half = 4; half >= 1; half /= 2)
                for (int s = 0; s < 8; s += 2 * half)
                    for (int k = 0; k < half; ++k) {
                        const u32 u = x[s + k], v = x[s + k + half];
                        x[s + k] = add(u, v), x[s + k + half] = mul(sub(u, v), z[slot(eighth_root(k * 4 / half))]);
                    }
            for (int i = 0; i < 8; ++i) to[64 * g + 8 * i + t] = x[i];
        }
    }
}

// split() undone.
inline void join(const u32* from, u32* to, int lg, const u32* rows) {
    const u32* z = kStaticTables.words[1];  // z^-k
    for (std::size_t g = 0; g < (std::size_t(1) << lg) / 64; ++g) {
        u32 c[64];
        std::copy_n(from + 64 * g, 64, c);
        const u32* a = rows + 32 * (g / 8) + 16;
        const auto& b = kValueLanes.inverse[g % 8];
        for (int t = 0; t < 8; ++t) {
            u32 x[8];
            for (int i = 0; i < 8; ++i) x[i] = c[8 * i + t];
            for (int half = 1; half <= 4; half *= 2)
                for (int s = 0; s < 8; s += 2 * half)
                    for (int k = 0; k < half; ++k) {
                        const u32 u = x[s + k], v = mul(x[s + k + half], z[slot(eighth_root(k * 4 / half))]);
                        x[s + k] = add(u, v), x[s + k + half] = sub(u, v);
                    }
            for (int k = 0; k < 8; ++k) to[64 * g + 8 * t + k] = mul(mul(x[k], b[k][t]), a[k]);
        }
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

    static void forward_upper(std::uint32_t* f, int lg, const std::uint32_t* roots) {
        detail::scalar::forward_upper(f, lg, roots);
    }

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

    // detail::Product: the factors copied into a and b, zero past them, and the product copied out.
    static void multiply(const detail::Product& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        const std::size_t len = std::size_t(1) << p.lg;
        std::fill(std::copy_n(p.f, p.n, p.a), p.a + len, 0u);
        if (p.g != p.b) std::copy_n(p.g, p.m, p.b);  // g may be b (in place)
        std::fill(p.b + p.m, p.b + len, 0u);
        multiply(p.a, p.b, len, len, p.lg, roots, inverse_roots);
        std::copy_n(p.a, p.count, p.out);
    }

    // The square of p.f (g, m and b unused): one forward transform.
    static void square(const detail::Product& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        const std::size_t len = std::size_t(1) << p.lg;
        std::fill(std::copy_n(p.f, p.n, p.a), p.a + len, 0u);
        forward(p.a, p.lg, roots);
        detail::scalar::multiply_leaves(p.a, p.a, p.lg, roots);
        inverse(p.a, p.lg, inverse_roots);
        std::copy_n(p.a, p.count, p.out);
    }

    // Leaf-form operations (ntt.hpp, Transform): their contracts are the Transform calls'.
    static void leaf_products(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out, int lg,
                              const std::uint32_t* roots, bool accumulate) {
        detail::scalar::multiply_leaves(a, b, out, lg, roots, accumulate);
    }

    static void forward_values(std::uint32_t* f, int lg, const std::uint32_t* roots, const std::uint32_t* rows) {
        forward(f, lg, roots);
        detail::scalar::split(f, f, lg, rows);
    }

    // The leaf form (rows null) or values form at `from` -> its coefficients [0, count) to out,
    // through f (64-byte aligned, 2^lg + kPadding words; may be from, out may be f).
    static void inverse(const std::uint32_t* from, std::uint32_t* f, std::uint32_t* out, std::size_t count, int lg,
                        const std::uint32_t* inverse_roots, const std::uint32_t* rows) {
        if (rows) detail::scalar::join(from, f, lg, rows);
        else if (from != f) std::copy_n(from, std::size_t(1) << lg, f);
        inverse(f, lg, inverse_roots);
        if (out != f) std::copy_n(f, count, out);
    }
};

}  // namespace fps::ntt::variant
