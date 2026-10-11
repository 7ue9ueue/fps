// Variant avx2: lc-opt's AVX2 NTT, with kernels scheduled for Zen 3 (avx2_kernels.hpp,
// avx2_product_kernels.hpp). Algorithm, value ranges and measurements: notes.md.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "backend/ntt/avx2_kernels.hpp"
#include "backend/ntt/avx2_product_kernels.hpp"
#include "backend/ntt/common.hpp"

namespace fps::ntt::detail::avx2 {

namespace kernels = avx2_kernels;
namespace product_kernels = avx2_product_kernels;
using kernels::Vec;
using u32 = std::uint32_t;

inline Vec broadcast(u32 x) { return _mm256_set1_epi32(int(x)); }
inline Vec add(Vec x, Vec y) { return _mm256_add_epi32(x, y); }
inline Vec diff(Vec x, Vec y) { return _mm256_sub_epi32(_mm256_add_epi32(x, broadcast(2 * kP)), y); }  // x - y + 2P
inline Vec diff_canonical(Vec x, Vec y) { return _mm256_sub_epi32(_mm256_add_epi32(x, broadcast(kP)), y); }  // y < P

// x < 2 bound -> x mod bound.
inline Vec reduce(Vec x, u32 bound) { return _mm256_min_epu32(x, _mm256_sub_epi32(x, broadcast(bound))); }
inline Vec canonical(Vec x) { return reduce(reduce(x, 2 * kP), kP); }  // x < 4P

// A constant factor w < P and its Shoup quotient.
struct Factor {
    Vec w, q;
    explicit Factor(u32 value) : w(broadcast(value)), q(broadcast(quotient(value))) {}
    Factor(u32 value, u32 value_quotient) : w(broadcast(value)), q(broadcast(value_quotient)) {}
};

// Entry k of a table, with its quotient.
inline Factor entry(const u32* table, std::size_t k) { return Factor(table[slot(k)], table[slot(k) + 8]); }

// x w mod P in [0, 2P), for any x < 2^32.
inline Vec multiply(Vec x, const Factor& f) {
    const Vec even = _mm256_srli_epi64(_mm256_mul_epu32(x, f.q), 32);
    const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), f.q);
    const Vec q = _mm256_blend_epi32(even, odd, 0xAA);
    return _mm256_sub_epi32(_mm256_mullo_epi32(x, f.w), _mm256_mullo_epi32(q, broadcast(kP)));
}


inline Vec load(const u32* p) { return _mm256_load_si256(reinterpret_cast<const Vec*>(p)); }
inline void store(u32* p, Vec x) { _mm256_store_si256(reinterpret_cast<Vec*>(p), x); }

// variant::scalar::build_table, 8 entries at a time past the static prefix.
inline void build_table(u32* t, std::size_t count, bool inverse) {
    static_assert(kStaticCount >= 8);
    const Roots& roots = kRoots[inverse];
    for (std::size_t h = copy_static_table(t, count, inverse); h < count; h *= 2) {
        // Quotient of v: (v 2^32 mod P) / -P mod 2^32.
        const Factor by(roots[std::countr_zero(h)]), to_r(kR);
        for (std::size_t j = 0; j < h; j += 8) {
            const Vec v = reduce(multiply(load(t + slot(j)), by), kP);
            const Vec vr = reduce(multiply(v, to_r), kP);
            store(t + slot(h + j), v);
            store(t + slot(h + j) + 8, _mm256_mullo_epi32(vr, broadcast(kernels::kNI)));
        }
    }
}

// Radix-4 groups. Group k at stride h holds 4h vectors, a polynomial mod X^4 - r(k)^2 with
// X = x^(8h); its outputs are the children 4k + t. A vector is a leaf: a polynomial mod x^8 - w.
// Forward values stay < 4P, inverse < 2P.

inline void forward_group(Vec* f, std::size_t h, std::size_t k, const u32* r) {
    if (k == 0) return kernels::forward_identity(f, h, r);
    kernels::forward(f, h, r + slot(k), r + slot(2 * k));
}

inline void forward_groups(Vec* a, Vec* b, std::size_t h, std::size_t k, const u32* r) {
    if (k == 0) {
        kernels::forward_identity(a, h, r);
        kernels::forward_identity(b, h, r);
        return;
    }
    const u32 *x = r + slot(k), *y = r + slot(2 * k);
    if (h == 4) return kernels::forward_pair(a, b, h, x, y);
    kernels::forward(a, h, x, y);
    kernels::forward(b, h, x, y);
}

inline void inverse_group(Vec* f, std::size_t h, std::size_t k, const u32* ir) {
    if (k == 0) return kernels::inverse_identity(f, h, ir);
    kernels::inverse(f, h, ir + slot(k), ir + slot(2 * k));
}

struct alignas(64) Leaves {
    u32 window[4][16];  // [w A_t, A_t]: x^i A_t mod x^8 - w is a sliding window
    u32 coefficients[4][8];
};

// Bottom stage of a product: the groups [first, first + nv / 4) with h = 1, their leaf products
// and inverse groups. Batch j + 1's forward half overlaps batch j's products and inverse.

// One group per kernel; the leaf moduli come in a weight array.
struct GroupBottom {
    static constexpr std::size_t kMinVectors = 4;

    // Moduli x^8 - w of the four leaves under group k: w = y, -y, z, -z, then their quotients.
    static void weights(const u32* r, std::size_t k, u32* w) {
        const u32* y = r + slot(2 * k);
        w[0] = y[0], w[1] = kP - y[0], w[2] = y[1], w[3] = kP - y[1];
        w[4] = y[8], w[5] = ~y[8], w[6] = y[9], w[7] = ~y[9];  // quotient(P - w) = ~quotient(w)
    }

    static void run(Vec* a, Vec* b, std::size_t nv, std::size_t first, const u32* r, const u32* ir) {
        Leaves leaves[2];
        alignas(32) u32 w[2][8];
        weights(r, first, w[0]);
        kernels::bottom_first(a, b, &leaves[0], r + slot(first), r + slot(2 * first), w[0]);
        for (std::size_t j = 0, k = first;; j += 4, ++k) {
            const std::size_t cur = j / 4 % 2, next = cur ^ 1;
            const u32 *ix = ir + slot(k), *iy = ir + slot(2 * k);
            if (j + 4 == nv) return kernels::bottom_last(a + j, &leaves[cur], ix, iy);
            weights(r, k + 1, w[next]);
            kernels::bottom_both(a + j + 4, b + j + 4, &leaves[next], r + slot(k + 1), r + slot(2 * k + 2), w[next],
                                 a + j, &leaves[cur], ix, iy);
        }
    }
};

// Two groups per always-inlined kernel; the leaf moduli are the groups' own twiddles. first even.
struct PairBottom {
    static constexpr std::size_t kMinVectors = 16;

    static void run(Vec* a, Vec* b, std::size_t nv, std::size_t first, const u32* r, const u32* ir) {
        Leaves leaves[2][2];
        product_kernels::bottom_first(a, b, leaves[0], r + slot(first), r + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const u32 *ix = ir + slot(k), *iy = ir + slot(2 * k);
            if (j + 8 == nv) return product_kernels::bottom_last(a + j, leaves[cur], ix, iy);
            product_kernels::bottom_both(a + j + 8, b + j + 8, leaves[next], r + slot(k + 2), r + slot(2 * k + 4),
                                         a + j, leaves[cur], ix, iy);
        }
    }
};

inline constexpr std::size_t kTile = 256;  // vectors

// Depth-first recursion of a product: forward transforms of a and b, leaf products into a,
// inverse transform. Runs of sibling subtrees of at most kTile vectors in all (8 KiB per array)
// go level by level as one tile, so one bottom stage pipelines across them.
template <class Bottom>
class Tree {
public:
    Tree(const u32* roots, const u32* inverse_roots) : r_(roots), ir_(inverse_roots) {}

    // The count subtrees of groups first, first + 1, ... with nv vectors each, one after another
    // from a and b: nv a power of 4, at least Bottom::kMinVectors.
    void visit(Vec* a, Vec* b, std::size_t nv, std::size_t count, std::size_t first) const {
        switch (nv) {
        case 4:
            return tiles<4>(a, b, count, first);
        case 16: return tiles<16>(a, b, count, first);
        case 64: return tiles<64>(a, b, count, first);
        case 256: return tiles<256>(a, b, count, first);
        }
        const std::size_t h = nv / 4;
        for (std::size_t k = first; k < first + count; ++k, a += nv, b += nv) {
            forward_groups(a, b, h, k, r_);
            visit(a, b, h, 4, 4 * k);
            inverse_group(a, h, k, ir_);
        }
    }

private:
    template <std::size_t NV>
    void tiles(Vec* a, Vec* b, std::size_t count, std::size_t first) const {
        const std::size_t run = std::min(count, kTile / NV);
        for (std::size_t c = 0; c < count; c += run) tile<NV>(a + c * NV, b + c * NV, run * NV, first + c);
    }

    // Subtrees of NV vectors each, total vectors from a and b; the first is group k of its level,
    // so the groups with stride h are numbered from k NV / 4h.
    template <std::size_t NV>
    [[gnu::noinline]] void tile(Vec* a, Vec* b, std::size_t total, std::size_t k) const {
        for (std::size_t h = NV / 4; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (NV / (4 * h)); j < total; j += 4 * h, ++g) forward_groups(a + j, b + j, h, g, r_);
        Bottom::run(a, b, total, k * (NV / 4), r_, ir_);
        for (std::size_t h = 4; h < NV; h *= 4)
            for (std::size_t j = 0, g = k * (NV / (4 * h)); j < total; j += 4 * h, ++g) inverse_group(a + j, h, g, ir_);
    }

    const u32 *r_, *ir_;
};

// Top levels. Inputs of the first level are canonical. If an input's upper half is zero ("sparse"),
// the first radix-2 level is a copy and the first radix-4 level reads only the lower half.

// f[i] and f[i + h] for i < h -> their sum and difference, < 2P: inputs canonical, no reduction.
inline void forward_radix2(Vec* f, std::size_t h, bool sparse) {
    if (sparse) {
        std::memcpy(f + h, f, h * sizeof(Vec));
        return;
    }
    for (std::size_t i = 0; i < h; ++i) {
        const Vec x = f[i], y = f[i + h];
        f[i] = add(x, y), f[i + h] = diff_canonical(x, y);
    }
}

// The first radix-4 level of a transform, on canonical input; z = r(1).
inline void forward_radix4(Vec* f, std::size_t h, bool sparse, const u32* roots) {
    if (!sparse) return kernels::forward_top(f, h, roots);
    const Factor z = entry(roots, 1);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec a = f[j], b = f[j + h], zb = multiply(b, z);  // < 4P
        f[j] = add(a, b), f[j + h] = diff(a, b), f[j + 2 * h] = add(a, zb), f[j + 3 * h] = diff(a, zb);
    }
}

// inverse_identity() and the scale s: canonical output; z = r^-1(1).
inline void inverse_radix4(Vec* f, std::size_t h, const u32* inverse_roots, u32 s) {
    const u32 sz = multiply_mod(s, inverse_roots[slot(1)]);
    alignas(32) const Vec w[4] = {broadcast(s), broadcast(quotient(s)), broadcast(sz), broadcast(quotient(sz))};
    product_kernels::inverse_top4(f, h, w);
}

// kernels::scale_radix2 with the scale s.
inline void scale_radix2(Vec* f, std::size_t h, u32 s) {
    alignas(32) u32 table[16] = {};  // table layout: s at entry 1
    table[1] = s;
    table[9] = quotient(s);
    kernels::scale_radix2(f, h, table);
}

// First level of an nv = 2 * 4^j transform, nv = 8q, in one pass: the radix-2 level and the first
// radix-4 group of each half (group 0 to f[0, 4q), group 1 to f[4q, 8q)). Inputs canonical;
// outputs < 4P. Sparse: the upper half f[4q, 8q) is zero and not read; modulo x^(n/2) - 1 and
// x^(n/2) + 1 the factor is then unchanged.
inline void forward_radix8_sparse(Vec* f, std::size_t q, const u32* roots) {
    const Factor i = entry(roots, 1), y = entry(roots, 2), z = entry(roots, 3);
    for (std::size_t j = 0; j < q; ++j) {
        const Vec f0 = f[j], f1 = f[j + q], f2 = f[j + 2 * q], f3 = f[j + 3 * q];
        // Group 0 (twiddles 1, 1, i): every term < 2P.
        const Vec g0 = add(f0, f2), g1 = add(f1, f3);
        const Vec h0 = diff_canonical(f0, f2), ih1 = multiply(diff_canonical(f1, f3), i);
        f[j] = add(g0, g1), f[j + q] = diff(g0, g1);
        f[j + 2 * q] = add(h0, ih1), f[j + 3 * q] = diff(h0, ih1);
        // Group 1 (twiddles i, y, z).
        const Vec if2 = multiply(f2, i), if3 = multiply(f3, i);
        const Vec u0 = reduce(add(f0, if2), 2 * kP), v0 = reduce(diff(f0, if2), 2 * kP);
        const Vec yu1 = multiply(add(f1, if3), y), zv1 = multiply(diff(f1, if3), z);
        f[j + 4 * q] = add(u0, yu1), f[j + 5 * q] = diff(u0, yu1);
        f[j + 6 * q] = add(v0, zv1), f[j + 7 * q] = diff(v0, zv1);
    }
}

inline void forward_radix8_dense(Vec* f, std::size_t q, const u32* roots) {
    const Factor i = entry(roots, 1), y = entry(roots, 2), z = entry(roots, 3);
    for (std::size_t j = 0; j < q; ++j) {
        Vec u[4], v[4];  // the halves mod x^(n/2) - 1 and x^(n/2) + 1, < 2P
        for (std::size_t t = 0; t < 4; ++t) {
            const Vec low = f[j + t * q], high = f[j + (t + 4) * q];
            u[t] = add(low, high), v[t] = diff_canonical(low, high);
        }
        // Group 0 (twiddles 1, 1, i).
        const Vec g0 = reduce(add(u[0], u[2]), 2 * kP), g1 = reduce(add(u[1], u[3]), 2 * kP);
        const Vec h0 = reduce(diff(u[0], u[2]), 2 * kP), ih1 = multiply(diff(u[1], u[3]), i);
        f[j] = add(g0, g1), f[j + q] = diff(g0, g1);
        f[j + 2 * q] = add(h0, ih1), f[j + 3 * q] = diff(h0, ih1);
        // Group 1 (twiddles i, y, z).
        const Vec iv2 = multiply(v[2], i), iv3 = multiply(v[3], i);
        const Vec a0 = reduce(add(v[0], iv2), 2 * kP), c0 = reduce(diff(v[0], iv2), 2 * kP);
        const Vec yb = multiply(add(v[1], iv3), y), zd = multiply(diff(v[1], iv3), z);
        f[j + 4 * q] = add(a0, yb), f[j + 5 * q] = diff(a0, yb);
        f[j + 6 * q] = add(c0, zd), f[j + 7 * q] = diff(c0, zd);
    }
}

inline void forward_radix8(Vec* f, std::size_t q, bool sparse, const u32* roots) {
    if (sparse) return forward_radix8_sparse(f, q, roots);
    forward_radix8_dense(f, q, roots);
}

// The last inverse level of each half, the radix-2 level and the scale s in one pass over the 8q
// vectors after the subtrees: canonical output.
inline void inverse_radix8(Vec* f, std::size_t q, const u32* ir, u32 s) {
    const u32 z0 = ir[slot(1)], y1 = ir[slot(2)], z1 = ir[slot(3)], x1 = ir[slot(1)];
    const u32 twiddles[6] = {s, multiply_mod(s, z0), multiply_mod(s, y1), multiply_mod(s, z1), multiply_mod(s, x1), x1};
    alignas(32) Vec w[12];
    for (int i = 0; i < 6; ++i) w[2 * i] = broadcast(twiddles[i]), w[2 * i + 1] = broadcast(quotient(twiddles[i]));
    product_kernels::inverse_top(f, q, w);
}

// a b mod x^len - 1 into a; the scale is fused into the last inverse level. A sparse factor (upper
// half zero) has its upper half neither read nor needed: its first levels read the lower half only.
template <class Bottom>
inline void product(u32* a_words, u32* b_words, bool sparse_a, bool sparse_b, int lg, const u32* r, const u32* ir) {
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    auto* a = reinterpret_cast<Vec*>(a_words);
    auto* b = reinterpret_cast<Vec*>(b_words);
    const Tree<Bottom> tree(r, ir);
    if (!has_radix2(lg)) {
        const std::size_t h = nv / 4;
        forward_radix4(a, h, sparse_a, r);
        forward_radix4(b, h, sparse_b, r);
        tree.visit(a, b, h, 4, 0);
        return inverse_radix4(a, h, ir, product_scale(lg));
    }
    if (nv == 8) {  // lg = 6: subtrees of 4 vectors
        forward_radix2(a, 4, sparse_a);
        forward_radix2(b, 4, sparse_b);
        tree.visit(a, b, 4, 2, 0);
        return scale_radix2(a, 4, product_scale(lg));
    }
    const std::size_t q = nv / 8;
    forward_radix8(a, q, sparse_a, r);
    forward_radix8(b, q, sparse_b, r);
    tree.visit(a, b, q, 8, 0);
    inverse_radix8(a, q, ir, product_scale(lg));
}

// Transforms alone. The bottom level (h = 1) has a group per 4 vectors, each with its own twiddles.

// Forward groups [first, first + count) with h = 1, values < 4P in; canonical leaves out. first
// and count even.
inline void forward_bottom(Vec* f, std::size_t count, std::size_t first, const u32* r) {
    for (std::size_t k = first; k < first + count; k += 2, f += 8) kernels::forward_leaves(f, r + slot(k), r + slot(2 * k));
}

// Inverse groups [first, first + count) with h = 1; canonical in, < 2P out. first and count even.
inline void inverse_bottom(Vec* f, std::size_t count, std::size_t first, const u32* ir) {
    for (std::size_t k = first; k < first + count; k += 2, f += 8) kernels::inverse_leaves(f, ir + slot(k), ir + slot(2 * k));
}

// The count subtrees of groups first, first + 1, ... with nv vectors each (a power of 4), one after
// another from f; depth first, runs of siblings of at most kTile vectors level by level, as Tree.

inline void forward_trees(Vec* f, std::size_t nv, std::size_t count, std::size_t first, const u32* r) {
    if (nv > kTile) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) {
            forward_group(f, nv / 4, k, r);
            forward_trees(f, nv / 4, 4, 4 * k, r);
        }
        return;
    }
    const std::size_t run = std::min(count, kTile / nv), total = run * nv;
    for (std::size_t k = first; k < first + count; k += run, f += total) {
        for (std::size_t h = nv / 4; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < total; j += 4 * h, ++g) forward_group(f + j, h, g, r);
        forward_bottom(f, total / 4, k * (nv / 4), r);
    }
}

inline void inverse_trees(Vec* f, std::size_t nv, std::size_t count, std::size_t first, const u32* ir) {
    if (nv > kTile) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) {
            inverse_trees(f, nv / 4, 4, 4 * k, ir);
            inverse_group(f, nv / 4, k, ir);
        }
        return;
    }
    const std::size_t run = std::min(count, kTile / nv), total = run * nv;
    for (std::size_t k = first; k < first + count; k += run, f += total) {
        inverse_bottom(f, total / 4, k * (nv / 4), ir);
        for (std::size_t h = 4; h < nv; h *= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < total; j += 4 * h, ++g) inverse_group(f + j, h, g, ir);
    }
}

// Below this length the radix-2 pass and the asm forward_identity beat forward_radix8_dense (it
// spills); above, the saved pass over memory wins (notes.md, 2026-10-11).
inline constexpr int kDenseRadix8MinLog = 16;

inline void forward(u32* words, int lg, const u32* r) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    if (!has_radix2(lg)) {
        kernels::forward_top(f, nv / 4, r);
        return forward_trees(f, nv / 4, 4, 0, r);
    }
    if (lg < kDenseRadix8MinLog) {
        forward_radix2(f, nv / 2, false);
        return forward_trees(f, nv / 2, 2, 0, r);
    }
    const std::size_t q = nv / 8;
    forward_radix8_dense(f, q, r);
    forward_trees(f, q, 8, 0, r);
}

inline void inverse(u32* words, int lg, const u32* ir) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    if (!has_radix2(lg)) {
        inverse_trees(f, nv / 4, 4, 0, ir);
        return inverse_radix4(f, nv / 4, ir, leaf_scale(lg));
    }
    if (nv == 8) {
        inverse_trees(f, 4, 2, 0, ir);
        return scale_radix2(f, 4, leaf_scale(lg));
    }
    inverse_trees(f, nv / 8, 8, 0, ir);
    inverse_radix8(f, nv / 8, ir, leaf_scale(lg));
}

}  // namespace fps::ntt::detail::avx2

namespace fps::ntt::variant {

struct avx2 {
    static void build_table(std::uint32_t* t, std::size_t count, bool inverse) {
        detail::avx2::build_table(t, count, inverse);
    }

    static void forward(std::uint32_t* f, int lg, const std::uint32_t* roots) { detail::avx2::forward(f, lg, roots); }

    static void inverse(std::uint32_t* f, int lg, const std::uint32_t* inverse_roots) {
        detail::avx2::inverse(f, lg, inverse_roots);
    }

    // Two groups per bottom kernel, but one at 2^7 and 2^8, where that was faster (notes.md).
    static void multiply(std::uint32_t* a, std::uint32_t* b, std::size_t n, std::size_t m, int lg,
                         const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        const std::size_t len = std::size_t(1) << lg;
        const bool sparse_a = 2 * n <= len, sparse_b = 2 * m <= len;
        if (lg == 7 || lg == 8)
            return detail::avx2::product<detail::avx2::GroupBottom>(a, b, sparse_a, sparse_b, lg, roots, inverse_roots);
        detail::avx2::product<detail::avx2::PairBottom>(a, b, sparse_a, sparse_b, lg, roots, inverse_roots);
    }
};

}  // namespace fps::ntt::variant
