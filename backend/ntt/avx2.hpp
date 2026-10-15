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
#include "backend/ntt/intrinsics.hpp"

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
    static constexpr bool kSquare = false;

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

// Two groups per always-inlined kernel; the leaf moduli are the groups' own twiddles. first even,
// nv a multiple of 8.
struct PairBottom {
    static constexpr bool kSquare = false;

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

// The bottom stage of a square, a times a (Tree skips b): PairBottom's kernels without b's forward
// half; the leaf products read A's coefficients from the windows. At every length: at 2^8 it took
// 0.93 of the time of GroupBottom with b = a (a's forward half twice), at 2^7 0.96, inside the
// spread of code layout there (notes.md, issue #55).
struct PairSquareBottom {
    static constexpr bool kSquare = true;

    static void run(Vec* a, Vec*, std::size_t nv, std::size_t first, const u32* r, const u32* ir) {
        Leaves leaves[2][2];
        product_kernels::bottom_square_first(a, leaves[0], r + slot(first), r + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const u32 *ix = ir + slot(k), *iy = ir + slot(2 * k);
            if (j + 8 == nv) return product_kernels::bottom_square_last(a + j, leaves[cur], ix, iy);
            product_kernels::bottom_square_both(a + j + 8, leaves[next], r + slot(k + 2), r + slot(2 * k + 4), a + j,
                                                leaves[cur], ix, iy);
        }
    }
};

inline constexpr std::size_t kTile = 256;  // vectors

// forward_groups(), of a alone for a square.
template <bool kSquare>
inline void forward_factors(Vec* a, Vec* b, std::size_t h, std::size_t k, const u32* r) {
    if constexpr (kSquare) forward_group(a, h, k, r);
    else forward_groups(a, b, h, k, r);
}

// Depth-first recursion of a product: forward transforms of a and b (of a alone if
// Bottom::kSquare), leaf products into a, inverse transform. Runs of sibling subtrees of at most kTile vectors in all (8 KiB per array)
// go level by level as one tile, so one bottom stage pipelines across them.
template <class Bottom>
class Tree {
public:
    Tree(const u32* roots, const u32* inverse_roots) : r_(roots), ir_(inverse_roots) {}

    // The count subtrees of groups first, first + 1, ... with nv vectors each, one after another
    // from a and b: nv a power of 4, count nv >= 8, first even.
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
            forward_factors<Bottom::kSquare>(a, b, h, k, r_);
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
            for (std::size_t j = 0, g = k * (NV / (4 * h)); j < total; j += 4 * h, ++g)
                forward_factors<Bottom::kSquare>(a + j, b + j, h, g, r_);
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

// Products from spans (detail::Product, ntt::multiply). Their first level reads a factor from a
// source of n words in caller memory, zero past them (detail::first_level), and their last level
// writes the product's first count words to caller memory (detail::last_level), any alignment.
// The *_part<T> functions run columns j < columns of a level from the vectors at s + 8 (j + t h),
// to f[j + t h] (forward: blocks t < T read), or from f to o + 8 (j + t h) (inverse: t < T
// written). Products in place (Convolution) keep the levels above: through these, 2^6-2^7
// products ran 2-4% slower on Emerald Rapids and Zen 5 (notes.md, issue #38 round 2).

inline Vec loadu(const u32* p) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(p)); }
inline void storeu(u32* p, Vec x) { _mm256_storeu_si256(reinterpret_cast<Vec*>(p), x); }

// The radix-2 level on halves of h vectors, halves t < kLoad of s read (the rest zero): the sum
// and difference, < 2P: inputs canonical, no reduction.
template <int kLoad>
[[gnu::noinline]] void forward_radix2_part(const u32* s, Vec* f, std::size_t columns, std::size_t h) {
    const bool write = s != reinterpret_cast<const u32*>(f);  // else f[j] holds x already
    for (std::size_t j = 0; j < columns; ++j) {
        const Vec x = kLoad > 0 ? loadu(s + 8 * j) : _mm256_setzero_si256();
        if constexpr (kLoad < 2) {
            if (write || kLoad == 0) f[j] = x;
            f[j + h] = x;
        } else {
            const Vec y = loadu(s + 8 * (j + h));
            f[j] = add(x, y), f[j + h] = diff_canonical(x, y);
        }
    }
}

inline void forward_radix2_from(const u32* f, std::size_t n, u32* to, std::size_t h) {
    first_level<2>(f, n, to, 8 * h, 8, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        forward_radix2_part<decltype(blocks)::value>(s, reinterpret_cast<Vec*>(d), words / 8, h);
    });
}

// The first radix-4 level of a transform, group 0 (z = r(1)), on quarters of h vectors, quarters
// t < kLoad of s read: the asm kernels forward_top_from and forward_top3_from for four and three.
template <int kLoad>
[[gnu::noinline]] void forward_radix4_part(const u32* s, Vec* f, std::size_t columns, std::size_t h, const u32* roots) {
    if constexpr (kLoad == 4) {
        kernels::forward_top_from(s, f, columns, h, roots);
    } else if constexpr (kLoad == 3) {
        kernels::forward_top3_from(s, f, columns, h, roots);
    } else {
        const Factor z = entry(roots, 1);
        for (std::size_t j = 0; j < columns; ++j) {
            const Vec a = kLoad > 0 ? loadu(s + 8 * j) : _mm256_setzero_si256();
            if constexpr (kLoad < 2) {
                f[j] = a, f[j + h] = a, f[j + 2 * h] = a, f[j + 3 * h] = a;
            } else {
                const Vec b = loadu(s + 8 * (j + h)), zb = multiply(b, z);  // < 4P
                f[j] = add(a, b), f[j + h] = diff(a, b), f[j + 2 * h] = add(a, zb), f[j + 3 * h] = diff(a, zb);
            }
        }
    }
}

inline void forward_radix4_from(const u32* f, std::size_t n, u32* to, std::size_t h, const u32* roots) {
    first_level<4>(f, n, to, 8 * h, 16, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        forward_radix4_part<decltype(blocks)::value>(s, reinterpret_cast<Vec*>(d), words / 8, h, roots);
    });
}

// inverse_identity() and the scale on quarters of h vectors (z = r^-1(1)): canonical output,
// count words of it to out. w: as product_kernels::inverse_top4's.
template <int T>
[[gnu::noinline]] void inverse_radix4_part(const Vec* f, u32* o, std::size_t columns, std::size_t h, const Vec* w) {
    product_kernels::inverse_top4_to<T>(f, o, columns, h, w);
}

inline void inverse_radix4_out(u32* f, std::size_t h, u32* out, std::size_t count, const u32* inverse_roots,
                               u32 scale) {
    const u32 sz = multiply_mod(scale, inverse_roots[slot(1)]);
    alignas(32) const Vec w[4] = {broadcast(scale), broadcast(quotient(scale)), broadcast(sz), broadcast(quotient(sz))};
    last_level<4>(f, 8 * h, out, count, 16, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        constexpr int T = decltype(blocks)::value;
        if constexpr (T > 0) inverse_radix4_part<T>(reinterpret_cast<const Vec*>(s), d, words / 8, h, w);
    });
}

// scale_radix2() on halves of h vectors, halves t < T of the output to o (products at lg = 6).
template <int T>
[[gnu::noinline]] void scale_radix2_part(const Vec* f, u32* o, std::size_t columns, std::size_t h, u32 s) {
    const Factor scale(s);
    for (std::size_t j = 0; j < columns; ++j) {
        const Vec x = f[j], y = f[j + h];
        storeu(o + 8 * j, reduce(multiply(add(x, y), scale), kP));
        if constexpr (T > 1) storeu(o + 8 * (j + h), reduce(multiply(diff(x, y), scale), kP));
    }
}

inline void scale_radix2_out(u32* f, std::size_t h, u32* out, std::size_t count, u32 scale) {
    last_level<2>(f, 8 * h, out, count, 8, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        constexpr int T = decltype(blocks)::value;
        if constexpr (T > 0) scale_radix2_part<T>(reinterpret_cast<const Vec*>(s), d, words / 8, h, scale);
    });
}

// First level of an nv = 2 * 4^j transform, nv = 8q, in one pass: the radix-2 level and the first
// radix-4 group of each half (group 0 to f[0, 4q), group 1 to f[4q, 8q)). Blocks t < kLoad of s
// read, the rest zero; outputs < 4P. With kLoad <= 4 the upper half is zero: modulo x^(n/2) - 1
// and x^(n/2) + 1 the factor is then its lower half, and the radix-2 level adds nothing.
template <int kLoad>
[[gnu::noinline]] void forward_radix8_part(const u32* s, Vec* f, std::size_t columns, std::size_t q, const u32* roots) {
    const Factor i = entry(roots, 1), y = entry(roots, 2), z = entry(roots, 3);
    const Vec zero = _mm256_setzero_si256();
    for (std::size_t j = 0; j < columns; ++j) {
        const auto in = [&] [[gnu::always_inline]] (std::size_t t) { return t < kLoad ? loadu(s + 8 * (j + t * q)) : zero; };
        if constexpr (kLoad <= 4) {
            const Vec f0 = in(0), f1 = in(1), f2 = in(2), f3 = in(3);
            // Group 0 (twiddles 1, 1, i): every term < 2P.
            const Vec g0 = add(f0, f2), g1 = add(f1, f3);
            const Vec h0 = diff_canonical(f0, f2), ih1 = kLoad > 1 ? multiply(diff_canonical(f1, f3), i) : zero;
            f[j] = add(g0, g1), f[j + q] = diff(g0, g1);
            f[j + 2 * q] = add(h0, ih1), f[j + 3 * q] = diff(h0, ih1);
            // Group 1 (twiddles i, y, z).
            const Vec if2 = kLoad > 2 ? multiply(f2, i) : zero, if3 = kLoad > 3 ? multiply(f3, i) : zero;
            const Vec u0 = reduce(add(f0, if2), 2 * kP), v0 = reduce(diff(f0, if2), 2 * kP);
            const Vec yu1 = kLoad > 1 ? multiply(add(f1, if3), y) : zero, zv1 = kLoad > 1 ? multiply(diff(f1, if3), z) : zero;
            f[j + 4 * q] = add(u0, yu1), f[j + 5 * q] = diff(u0, yu1);
            f[j + 6 * q] = add(v0, zv1), f[j + 7 * q] = diff(v0, zv1);
        } else {
            // The halves mod x^(n/2) - 1 and x^(n/2) + 1, < 2P.
            const auto halves = [&] [[gnu::always_inline]] (std::size_t t, Vec& u, Vec& v) {
                const Vec low = in(t);
                if (t + 4 < kLoad) {
                    const Vec high = in(t + 4);
                    u = add(low, high), v = diff_canonical(low, high);
                } else {
                    u = v = low;
                }
            };
            Vec u0, u1, u2, u3, v0, v1, v2, v3;
            halves(0, u0, v0), halves(1, u1, v1), halves(2, u2, v2), halves(3, u3, v3);
            // Group 0 (twiddles 1, 1, i).
            const Vec g0 = reduce(add(u0, u2), 2 * kP), g1 = reduce(add(u1, u3), 2 * kP);
            const Vec h0 = reduce(diff(u0, u2), 2 * kP), ih1 = multiply(diff(u1, u3), i);
            f[j] = add(g0, g1), f[j + q] = diff(g0, g1);
            f[j + 2 * q] = add(h0, ih1), f[j + 3 * q] = diff(h0, ih1);
            // Group 1 (twiddles i, y, z).
            const Vec iv2 = multiply(v2, i), iv3 = multiply(v3, i);
            const Vec a0 = reduce(add(v0, iv2), 2 * kP), c0 = reduce(diff(v0, iv2), 2 * kP);
            const Vec yb = multiply(add(v1, iv3), y), zd = multiply(diff(v1, iv3), z);
            f[j + 4 * q] = add(a0, yb), f[j + 5 * q] = diff(a0, yb);
            f[j + 6 * q] = add(c0, zd), f[j + 7 * q] = diff(c0, zd);
        }
    }
}

inline void forward_radix8_from(const u32* f, std::size_t n, u32* to, std::size_t q, const u32* roots) {
    first_level<8>(f, n, to, 8 * q, 8, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        forward_radix8_part<decltype(blocks)::value>(s, reinterpret_cast<Vec*>(d), words / 8, q, roots);
    });
}

// The last inverse level of each half, the radix-2 level and the scale in one pass over the 8q
// vectors after the subtrees: canonical output, count words of it to out. w: as
// product_kernels::inverse_top's.
template <int T>
[[gnu::noinline]] void inverse_radix8_part(const Vec* f, u32* o, std::size_t columns, std::size_t q, const Vec* w) {
    product_kernels::inverse_top_to<T>(f, o, columns, q, w);
}

inline void inverse_radix8_out(u32* f, std::size_t q, u32* out, std::size_t count, const u32* ir, u32 scale) {
    const u32 z0 = ir[slot(1)], y1 = ir[slot(2)], z1 = ir[slot(3)], x1 = ir[slot(1)];
    const u32 twiddles[6] = {scale,
                             multiply_mod(scale, z0),
                             multiply_mod(scale, y1),
                             multiply_mod(scale, z1),
                             multiply_mod(scale, x1),
                             x1};
    alignas(32) Vec w[12];
    for (int i = 0; i < 6; ++i) w[2 * i] = broadcast(twiddles[i]), w[2 * i + 1] = broadcast(quotient(twiddles[i]));
    last_level<8>(f, 8 * q, out, count, 8, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        constexpr int T = decltype(blocks)::value;
        if constexpr (T > 0) inverse_radix8_part<T>(reinterpret_cast<const Vec*>(s), d, words / 8, q, w);
    });
}

// The product p (detail::Product); the scale is fused into the last inverse level. A factor's
// words past its n are neither read nor needed.
template <class Bottom>
inline void product(const Product& p, const u32* r, const u32* ir) {
    const std::size_t nv = (std::size_t(1) << p.lg) / 8;
    auto* a = reinterpret_cast<Vec*>(p.a);
    auto* b = reinterpret_cast<Vec*>(p.b);
    const Tree<Bottom> tree(r, ir);
    if (!has_radix2(p.lg)) {
        const std::size_t h = nv / 4;
        forward_radix4_from(p.f, p.n, p.a, h, r);
        forward_radix4_from(p.g, p.m, p.b, h, r);
        tree.visit(a, b, h, 4, 0);
        return inverse_radix4_out(p.a, h, p.out, p.count, ir, product_scale(p.lg));
    }
    if (nv == 8) {  // lg = 6: subtrees of 4 vectors
        forward_radix2_from(p.f, p.n, p.a, 4);
        forward_radix2_from(p.g, p.m, p.b, 4);
        tree.visit(a, b, 4, 2, 0);
        return scale_radix2_out(p.a, 4, p.out, p.count, product_scale(p.lg));
    }
    const std::size_t q = nv / 8;
    forward_radix8_from(p.f, p.n, p.a, q, r);
    forward_radix8_from(p.g, p.m, p.b, q, r);
    tree.visit(a, b, q, 8, 0);
    inverse_radix8_out(p.a, q, p.out, p.count, ir, product_scale(p.lg));
}

// The square of p.f (detail::Product; g, m and b unused): product() with one forward transform.
template <class Bottom>
inline void square(const Product& p, const u32* r, const u32* ir) {
    static_assert(Bottom::kSquare);
    const std::size_t nv = (std::size_t(1) << p.lg) / 8;
    auto* a = reinterpret_cast<Vec*>(p.a);
    const Tree<Bottom> tree(r, ir);
    if (!has_radix2(p.lg)) {
        const std::size_t h = nv / 4;
        forward_radix4_from(p.f, p.n, p.a, h, r);
        tree.visit(a, a, h, 4, 0);
        return inverse_radix4_out(p.a, h, p.out, p.count, ir, product_scale(p.lg));
    }
    if (nv == 8) {
        forward_radix2_from(p.f, p.n, p.a, 4);
        tree.visit(a, a, 4, 2, 0);
        return scale_radix2_out(p.a, 4, p.out, p.count, product_scale(p.lg));
    }
    const std::size_t q = nv / 8;
    forward_radix8_from(p.f, p.n, p.a, q, r);
    tree.visit(a, a, q, 8, 0);
    inverse_radix8_out(p.a, q, p.out, p.count, ir, product_scale(p.lg));
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

// A tile hook: tile(f, total) runs on the total vectors (leaves) at f of a run of subtrees, after
// their forward bottom level or before their inverse one (the values form, inverses from other
// memory). A run has at least 8 leaves. NoTile: none.
struct NoTile {
    [[gnu::always_inline]] void operator()(u32*, std::size_t) const {}
};

// forward_trees and inverse_trees with a tile hook. Copies, not one template with NoTile: with the
// in-place transforms on the template, bench.cpp's short products moved by code placement alone
// (Zen 3 product/6 1.29 against main; notes.md, issue #49).
template <class Tile>
inline void forward_tiles(Vec* f, std::size_t nv, std::size_t count, std::size_t first, const u32* r, Tile tile) {
    if (nv > kTile) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) {
            forward_group(f, nv / 4, k, r);
            forward_tiles(f, nv / 4, 4, 4 * k, r, tile);
        }
        return;
    }
    const std::size_t run = std::min(count, kTile / nv), total = run * nv;
    for (std::size_t k = first; k < first + count; k += run, f += total) {
        for (std::size_t h = nv / 4; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < total; j += 4 * h, ++g) forward_group(f + j, h, g, r);
        forward_bottom(f, total / 4, k * (nv / 4), r);
        tile(reinterpret_cast<u32*>(f), total);
    }
}

template <class Tile>
inline void inverse_tiles(Vec* f, std::size_t nv, std::size_t count, std::size_t first, const u32* ir, Tile tile) {
    if (nv > kTile) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) {
            inverse_tiles(f, nv / 4, 4, 4 * k, ir, tile);
            inverse_group(f, nv / 4, k, ir);
        }
        return;
    }
    const std::size_t run = std::min(count, kTile / nv), total = run * nv;
    for (std::size_t k = first; k < first + count; k += run, f += total) {
        tile(reinterpret_cast<u32*>(f), total);
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

// The upper half of the leaf form of length 2L, L = 2^lg, from words = p mod (x^L + 1), canonical;
// r: the table of length 2L. Below a radix-2 level that is group 1's subtree; below group 0
// (radix 4) those of groups 2 and 3, from the residues mod x^(L/2) -+ r(1) (out < 3P).
inline void forward_upper(u32* words, int lg, const u32* r) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8, h = nv / 2;
    if (has_radix2(lg + 1)) return forward_trees(f, nv, 1, 1, r);
    const Factor z = entry(r, 1);
    for (std::size_t i = 0; i < h; ++i) {
        const Vec x = f[i], y = multiply(f[i + h], z);
        f[i] = add(x, y), f[i + h] = diff(x, y);
    }
    forward_trees(f, h, 2, 2, r);
}

// forward_upper undone: the upper half of the leaf form of length 2L, L = 2^lg, to p mod (x^L + 1),
// canonical; ir: the table of length 2L.
inline void inverse_upper(u32* words, int lg, const u32* ir) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8, h = nv / 2;
    const Factor scale(leaf_scale(lg));
    if (has_radix2(lg + 1)) {
        inverse_trees(f, nv, 1, 1, ir);
        for (std::size_t i = 0; i < nv; ++i) f[i] = reduce(multiply(f[i], scale), kP);
        return;
    }
    inverse_trees(f, h, 2, 2, ir);
    const Factor sz(multiply_mod(leaf_scale(lg), ir[slot(1)]));
    for (std::size_t i = 0; i < h; ++i) {
        const Vec x = f[i], y = f[i + h];
        f[i] = reduce(multiply(add(x, y), scale), kP), f[i + h] = reduce(multiply(diff(x, y), sz), kP);
    }
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

// Leaf-form operations (ntt.hpp, Transform), with the intrinsics on Avx2 for the leaf products and
// the values form (intrinsics.hpp). lc-opt's asm leaf product, one leaf per step, ran 1.29x slower
// than the intrinsics' 4 leaves per step on Zen 3 (notes.md).
using Leaf = intrinsics::Bottom<simd::Avx2>;
using Values = intrinsics::Values<simd::Avx2>;

// The values form: the split runs on each run of subtrees after its bottom level, in cache.
inline void forward_values(u32* words, int lg, const u32* r, const u32* rows) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    const auto tile = [words, rows] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
        Values::split(t, t, leaves / 8, std::size_t(t - words) / 64, rows);
    };
    if (!has_radix2(lg)) {
        kernels::forward_top(f, nv / 4, r);
        return forward_tiles(f, nv / 4, 4, 0, r, tile);
    }
    if (lg < kDenseRadix8MinLog) {
        forward_radix2(f, nv / 2, false);
        return forward_tiles(f, nv / 2, 2, 0, r, tile);
    }
    const std::size_t q = nv / 8;
    forward_radix8_dense(f, q, r);
    forward_tiles(f, q, 8, 0, r, tile);
}

// inverse() through `words` with tile hooks, the top level writing count coefficients to out
// (out = words: in place, count = 2^lg).
template <class Tile>
inline void inverse_into(u32* words, u32* out, std::size_t count, int lg, const u32* ir, Tile tile) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    const u32 s = leaf_scale(lg);
    if (!has_radix2(lg)) {
        inverse_tiles(f, nv / 4, 4, 0, ir, tile);
        if (out == words) return inverse_radix4(f, nv / 4, ir, s);
        return inverse_radix4_out(words, nv / 4, out, count, ir, s);
    }
    if (nv == 8) {
        inverse_tiles(f, 4, 2, 0, ir, tile);
        if (out == words) return scale_radix2(f, 4, s);
        return scale_radix2_out(words, 4, out, count, s);
    }
    inverse_tiles(f, nv / 8, 8, 0, ir, tile);
    if (out == words) return inverse_radix8(f, nv / 8, ir, s);
    inverse_radix8_out(words, nv / 8, out, count, ir, s);
}

// The leaf form (rows null) or values form at `from` -> coefficients [0, count) to out, through
// words (may be from). The runs of subtrees read their leaves from `from` (copied, or joined, into
// words before their bottom level).
inline void inverse(const u32* from, u32* words, u32* out, std::size_t count, int lg, const u32* ir, const u32* rows) {
    if (rows) {
        return inverse_into(words, out, count, lg, ir, [from, words, rows] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
            const std::size_t at = std::size_t(t - words);
            Values::join(from + at, t, leaves / 8, at / 64, rows);
        });
    }
    if (from != words) {
        return inverse_into(words, out, count, lg, ir, [from, words] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
            std::memcpy(t, from + (t - words), 8 * leaves * sizeof(u32));
        });
    }
    inverse_into(words, out, count, lg, ir, NoTile{});
}

// Transform's fused ops (ntt.hpp): a forward transform from in (n <= 2^lg words of caller memory,
// zero past them) into words; for a cyclic product, each run of subtrees' bottom stage with the
// leaf products (StoredBottom), then the inverse levels, the top level writing count words of the
// lower or upper half to out; for an inverse of leaf products, the bottom stage from the stored
// leaves (LeavesBottom, SumBottom). The leaf products of forward_product: a tile hook in cache.

// forward_tiles() and inverse_tiles() in one recursion: on each run of subtrees, the forward levels,
// the bottom stage stage(f, total) (forward bottom level, leaf products, inverse bottom level) and
// the inverse levels.
template <class Stage>
inline void cycle_trees(Vec* f, std::size_t nv, std::size_t count, std::size_t first, const u32* r, const u32* ir,
                        Stage stage) {
    if (nv > kTile) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) {
            forward_group(f, nv / 4, k, r);
            cycle_trees(f, nv / 4, 4, 4 * k, r, ir, stage);
            inverse_group(f, nv / 4, k, ir);
        }
        return;
    }
    const std::size_t run = std::min(count, kTile / nv), total = run * nv;
    for (std::size_t k = first; k < first + count; k += run, f += total) {
        for (std::size_t h = nv / 4; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < total; j += 4 * h, ++g) forward_group(f + j, h, g, r);
        stage(reinterpret_cast<u32*>(f), total);
        for (std::size_t h = 4; h < nv; h *= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < total; j += 4 * h, ++g) inverse_group(f + j, h, g, ir);
    }
}

// inverse_trees() with the bottom level replaced by stage(f, total), which writes the leaves and
// runs the inverse bottom level.
template <class Stage>
inline void inverse_trees_with(Vec* f, std::size_t nv, std::size_t count, std::size_t first, const u32* ir, Stage stage) {
    if (nv > kTile) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) {
            inverse_trees_with(f, nv / 4, 4, 4 * k, ir, stage);
            inverse_group(f, nv / 4, k, ir);
        }
        return;
    }
    const std::size_t run = std::min(count, kTile / nv), total = run * nv;
    for (std::size_t k = first; k < first + count; k += run, f += total) {
        stage(reinterpret_cast<u32*>(f), total);
        for (std::size_t h = 4; h < nv; h *= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < total; j += 4 * h, ++g) inverse_group(f + j, h, g, ir);
    }
}

// The forward top level from in, then trees(f, nv, count): the count subtrees of nv vectors below.
// A radix-2 level, not product()'s radix 8: from a source apart from words, its 16 streams 2^lg / 8
// words apart ran forward(in, F) 1.48x slower than in place at 2^18 on Zen 3 (a guess: 16 streams
// on the same sets of its 8-way L2); with the radix-2 level 1.04x, within 1% elsewhere (notes).
template <class Trees>
[[gnu::always_inline]] inline void from_top(const u32* in, std::size_t n, u32* words, int lg, const u32* r, Trees trees) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    if (!has_radix2(lg)) {
        forward_radix4_from(in, n, words, nv / 4, r);
        return trees(f, nv / 4, 4);
    }
    forward_radix2_from(in, n, words, nv / 2);
    trees(f, nv / 2, 2);
}

template <class Tile>
inline void forward_from(const u32* in, std::size_t n, u32* words, int lg, const u32* r, Tile tile) {
    from_top(in, n, words, lg, r, [&] [[gnu::always_inline]] (Vec* f, std::size_t nv, std::size_t count) {
        forward_tiles(f, nv, count, 0, r, tile);
    });
}

inline void forward_from(const u32* in, std::size_t n, u32* words, int lg, const u32* r) {
    from_top(in, n, words, lg, r, [&] [[gnu::always_inline]] (Vec* f, std::size_t nv, std::size_t count) {
        forward_trees(f, nv, count, 0, r);
    });
}

inline void forward_product(const u32* in, std::size_t n, const u32* b, u32* words, int lg, const u32* r) {
    forward_from(in, n, words, lg, r, [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
        const std::size_t at = std::size_t(t - words);
        intrinsics::leaf_products_at<simd::Avx2>(t, b + at, leaves / 4, at / 32, r);
    });
}

// forward_upper() of in (n <= 2^lg words) into words: in is copied in first under a radix-2 level
// (lc-opt's forward_upper does the same), read by the first level under a radix-4 one.
inline void forward_upper_from(const u32* in, std::size_t n, u32* words, int lg, const u32* r) {
    const std::size_t len = std::size_t(1) << lg;
    if (has_radix2(lg + 1) || n < len) {
        if (in != words) std::memmove(words, in, n * sizeof(u32));
        std::fill(words + n, words + len, 0u);
        return forward_upper(words, lg, r);
    }
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t h = len / 16;
    const Factor z = entry(r, 1);
    for (std::size_t i = 0; i < h; ++i) {
        const Vec x = loadu(in + 8 * i), y = multiply(loadu(in + 8 * (i + h)), z);
        f[i] = add(x, y), f[i + h] = diff(x, y);
    }
    forward_trees(f, h, 2, 2, r);
}

// The bottom stages of the fused products, two groups per asm kernel as PairBottom: b given as
// leaves (bottom_stored_*: forward butterflies of a, leaf products, inverse butterflies), or a and
// b both (bottom_leaves_*: the windows filled from a's leaves). Leaf products carry 2^-32, which
// the top level's scale undoes. first even, nv a multiple of 8.
struct StoredBottom {
    static void run(Vec* a, const u32* b, std::size_t nv, std::size_t first, const u32* r, const u32* ir) {
        Leaves leaves[2][2];
        product_kernels::bottom_stored_first(a, leaves[0], r + slot(first), r + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const u32 *ix = ir + slot(k), *iy = ir + slot(2 * k);
            if (j + 8 == nv) return product_kernels::bottom_stored_last(a + j, leaves[cur], b + 8 * j, ix, iy);
            product_kernels::bottom_stored_both(a + j + 8, leaves[next], r + slot(k + 2), r + slot(2 * k + 4), a + j,
                                                leaves[cur], b + 8 * j, ix, iy);
        }
    }
};

// out = the inverse butterflies of the leaf products of a and b (leaves at the same groups).
struct LeavesBottom {
    static void run(Vec* out, const u32* a, const u32* b, std::size_t nv, std::size_t first, const u32* r, const u32* ir) {
        Leaves leaves[2][2];
        product_kernels::bottom_leaves_first(a, leaves[0], r + slot(first), r + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const u32 *ix = ir + slot(k), *iy = ir + slot(2 * k);
            if (j + 8 == nv) return product_kernels::bottom_leaves_last(out + j, leaves[cur], b + 8 * j, ix, iy);
            product_kernels::bottom_leaves_both(a + 8 * (j + 8), leaves[next], r + slot(k + 2), r + slot(2 * k + 4), out + j,
                                                leaves[cur], b + 8 * j, ix, iy);
        }
    }
};

// The same for a sum of K = 2 or 3 leaf products (bottom_leaves2_*, bottom_leaves3_*: each product
// reduced, then added). Term k's leaf buffers at leaves[.][2k].
template <int K>
struct SumBottom {
    static void run(Vec* out, const LeafTerm* terms, std::size_t at, std::size_t nv, std::size_t first, const u32* r,
                    const u32* ir) {
        static_assert(K == 2 || K == 3);
        Leaves leaves[2][2 * K];
        const u32 *a0 = terms[0].a + at, *a1 = terms[1].a + at, *a2 = terms[K - 1].a + at;
        const u32 *b0 = terms[0].b + at, *b1 = terms[1].b + at, *b2 = terms[K - 1].b + at;
        const auto fill = [&](std::size_t j, std::size_t k, Leaves* buf) {
            if constexpr (K == 2) product_kernels::bottom_leaves2_first(a0 + 8 * j, a1 + 8 * j, buf, r + slot(k), r + slot(2 * k));
            else product_kernels::bottom_leaves3_first(a0 + 8 * j, a1 + 8 * j, a2 + 8 * j, buf, r + slot(k), r + slot(2 * k));
        };
        fill(0, first, leaves[0]);
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const u32 *ix = ir + slot(k), *iy = ir + slot(2 * k);
            if (j + 8 == nv) {
                if constexpr (K == 2) return product_kernels::bottom_leaves2_last(out + j, leaves[cur], b0 + 8 * j, b1 + 8 * j, ix, iy);
                else return product_kernels::bottom_leaves3_last(out + j, leaves[cur], b0 + 8 * j, b1 + 8 * j, b2 + 8 * j, ix, iy);
            }
            const std::size_t n8 = 8 * (j + 8);
            if constexpr (K == 2)
                product_kernels::bottom_leaves2_both(a0 + n8, a1 + n8, leaves[next], r + slot(k + 2), r + slot(2 * k + 4), out + j,
                                                     leaves[cur], b0 + 8 * j, b1 + 8 * j, ix, iy);
            else
                product_kernels::bottom_leaves3_both(a0 + n8, a1 + n8, a2 + n8, leaves[next], r + slot(k + 2), r + slot(2 * k + 4),
                                                     out + j, leaves[cur], b0 + 8 * j, b1 + 8 * j, b2 + 8 * j, ix, iy);
        }
    }
};

// Sums with a term alone: the intrinsics' bottom on Avx2 (Bottom::inverse_terms).
using TermsLeaf = intrinsics::Bottom<simd::Avx2>;

// Upper halves of the inverse top levels (Transform's fused ops): count words of the upper half of
// the result to out. Radix 4, radix 8: the generated kernels; radix 2: the differences.
inline void inverse_radix4_upper_out(u32* f, std::size_t h, u32* out, std::size_t count, const u32* inverse_roots,
                                     u32 scale) {
    const u32 sz = multiply_mod(scale, inverse_roots[slot(1)]);
    alignas(32) const Vec w[4] = {broadcast(scale), broadcast(quotient(scale)), broadcast(sz), broadcast(quotient(sz))};
    last_level<2>(f, 8 * h, out, count, 16, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        constexpr int T = decltype(blocks)::value;
        if constexpr (T > 0) product_kernels::inverse_top4_upper_to<T>(reinterpret_cast<const Vec*>(s), d, words / 8, h, w);
    });
}

[[gnu::noinline]] inline void scale_radix2_upper_part(const Vec* f, u32* o, std::size_t columns, std::size_t h, u32 s) {
    const Factor scale(s);
    for (std::size_t j = 0; j < columns; ++j) storeu(o + 8 * j, reduce(multiply(diff(f[j], f[j + h]), scale), kP));
}

inline void scale_radix2_upper_out(u32* f, std::size_t h, u32* out, std::size_t count, u32 scale) {
    last_level<1>(f, 8 * h, out, count, 8, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        if constexpr (decltype(blocks)::value > 0) scale_radix2_upper_part(reinterpret_cast<const Vec*>(s), d, words / 8, h, scale);
    });
}

inline void inverse_radix8_upper_out(u32* f, std::size_t q, u32* out, std::size_t count, const u32* ir, u32 scale) {
    const u32 z0 = ir[slot(1)], y1 = ir[slot(2)], z1 = ir[slot(3)], x1 = ir[slot(1)];
    const u32 twiddles[6] = {scale,
                             multiply_mod(scale, z0),
                             multiply_mod(scale, y1),
                             multiply_mod(scale, z1),
                             multiply_mod(scale, x1),
                             x1};
    alignas(32) Vec w[12];
    for (int i = 0; i < 6; ++i) w[2 * i] = broadcast(twiddles[i]), w[2 * i + 1] = broadcast(quotient(twiddles[i]));
    last_level<4>(f, 8 * q, out, count, 8, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        constexpr int T = decltype(blocks)::value;
        if constexpr (T > 0) product_kernels::inverse_top_upper_to<T>(reinterpret_cast<const Vec*>(s), d, words / 8, q, w);
    });
}

// The scale of a fused product's top level: product_scale, negated for kNegated.
inline u32 output_scale(int lg, unsigned output) {
    const u32 s = product_scale(lg);
    return output & kNegated ? kP - s : s;
}

inline void cyclic_product(const u32* in, std::size_t n, const u32* b, u32* words, u32* out, std::size_t count, int lg,
                           const u32* r, const u32* ir, unsigned output) {
    from_top(in, n, words, lg, r, [&] [[gnu::always_inline]] (Vec* f, std::size_t nv, std::size_t subtrees) {
        cycle_trees(f, nv, subtrees, 0, r, ir, [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
            const std::size_t at = std::size_t(t - words);
            StoredBottom::run(reinterpret_cast<Vec*>(t), b + at, leaves, at / 32, r, ir);
        });
    });
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    const u32 s = output_scale(lg, output);
    const bool upper = output & kUpperHalf;
    if (!has_radix2(lg)) {
        if (upper) return inverse_radix4_upper_out(words, nv / 4, out, count, ir, s);
        return inverse_radix4_out(words, nv / 4, out, count, ir, s);
    }
    if (upper) return scale_radix2_upper_out(words, nv / 2, out, count, s);
    scale_radix2_out(words, nv / 2, out, count, s);
}

inline void inverse_product(const LeafTerm* terms, int n, u32* words, u32* out, std::size_t count, int lg, const u32* r,
                            const u32* ir, unsigned output) {
    if (n == 1 && !terms[0].b && output == 0) return inverse(terms[0].a, words, out, count, lg, ir, nullptr);
    // Products only, 1 to 3 of them: the asm bottoms; a term alone, or 4 terms: the intrinsics'.
    const int products = std::all_of(terms, terms + n, [](const LeafTerm& t) { return t.b != nullptr; }) && n <= 3 ? n : 0;
    const auto stage = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
        const std::size_t at = std::size_t(t - words);
        auto* v = reinterpret_cast<Vec*>(t);
        if (products == 1) LeavesBottom::run(v, terms[0].a + at, terms[0].b + at, leaves, at / 32, r, ir);
        else if (products == 2) SumBottom<2>::run(v, terms, at, leaves, at / 32, r, ir);
        else if (products == 3) SumBottom<3>::run(v, terms, at, leaves, at / 32, r, ir);
        else TermsLeaf::inverse_terms(t, terms, n, leaves / 4, at / 32, r, ir);
    };
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    const u32 s = output_scale(lg, output);
    const bool upper = output & kUpperHalf;
    if (!has_radix2(lg)) {
        inverse_trees_with(f, nv / 4, 4, 0, ir, stage);
        if (upper) return inverse_radix4_upper_out(words, nv / 4, out, count, ir, s);
        return inverse_radix4_out(words, nv / 4, out, count, ir, s);
    }
    if (nv == 8) {
        inverse_trees_with(f, 4, 2, 0, ir, stage);
        if (upper) return scale_radix2_upper_out(words, 4, out, count, s);
        return scale_radix2_out(words, 4, out, count, s);
    }
    inverse_trees_with(f, nv / 8, 8, 0, ir, stage);
    if (upper) return inverse_radix8_upper_out(words, nv / 8, out, count, ir, s);
    inverse_radix8_out(words, nv / 8, out, count, ir, s);
}

// Hooked ops (Transform, issue #149): a forward from in whose runs of leaves go to a hook after
// their bottom level, an inverse whose runs of leaves a hook writes before theirs; both pruned for
// Kronecker layouts. width (vectors, a power of two; 0: none): vector bit log2(width) of the
// index is zero in the forward's input, or unused in the inverse's output, so the groups above it
// run only on the columns j with (j & width) == 0 (kernels::*_columns, *_even_columns), and the
// group holding it on half its inputs or outputs. As in lc-opt, groups with quarters below
// kPruneForward (kPruneInverse) vectors run whole: narrow columns gain little. The top level is
// pruned where its step allows (detail::PrunedLevel), and writes zeros to the skipped columns.
inline constexpr std::size_t kPruneForward = 32, kPruneInverse = 8;

// The group whose quarter index holds the bit (issue #149 round 2; lc-opt's forward_half and
// inverse_half): width = h (quarters 1 and 3 zero in the forward, unused in the inverse) or 2h
// (quarters 2 and 3). The forward reads half its inputs, the inverse writes half its outputs.
// With f2 = f3 = 0 the outputs are f0 + y f1, f0 - y f1, f0 + z f1, f0 - z f1; with f1 = f3 = 0
// they are A, A, C, C for A, C = f0 +- x f2. In < 4P, out < 4P.
inline void forward_group_half(Vec* f, std::size_t h, std::size_t k, const u32* r, bool high) {
    const Factor x = entry(r, k), y = entry(r, 2 * k), z = entry(r, 2 * k + 1);
    if (high) {
        for (std::size_t j = 0; j < h; ++j) {
            const Vec u = reduce(f[j], 2 * kP), b = f[j + h], yb = multiply(b, y), zb = multiply(b, z);
            f[j] = add(u, yb), f[j + h] = diff(u, yb), f[j + 2 * h] = add(u, zb), f[j + 3 * h] = diff(u, zb);
        }
        return;
    }
    for (std::size_t j = 0; j < h; ++j) {
        const Vec u = reduce(f[j], 2 * kP), xc = multiply(f[j + 2 * h], x), s = add(u, xc), d = diff(u, xc);
        f[j] = s, f[j + h] = s, f[j + 2 * h] = d, f[j + 3 * h] = d;
    }
}

// Inverse: outputs 0 and h (high) or 0 and 2h of group k, in and out < 2P.
inline void inverse_group_half(Vec* f, std::size_t h, std::size_t k, const u32* ir, bool high) {
    const Factor x = entry(ir, k), y = entry(ir, 2 * k), z = entry(ir, 2 * k + 1);
    if (high) {
        for (std::size_t j = 0; j < h; ++j) {
            const Vec f0 = f[j], f1 = f[j + h], f2 = f[j + 2 * h], f3 = f[j + 3 * h];
            const Vec ab = reduce(add(f0, f1), 2 * kP), cd = reduce(add(f2, f3), 2 * kP);
            f[j] = reduce(add(ab, cd), 2 * kP);
            f[j + h] = reduce(add(multiply(diff(f0, f1), y), multiply(diff(f2, f3), z)), 2 * kP);
        }
        return;
    }
    for (std::size_t j = 0; j < h; ++j) {
        const Vec ab = reduce(add(f[j], f[j + h]), 2 * kP), cd = reduce(add(f[j + 2 * h], f[j + 3 * h]), 2 * kP);
        f[j] = reduce(add(ab, cd), 2 * kP), f[j + 2 * h] = multiply(diff(ab, cd), x);
    }
}

// Width 1 (vector bit 0, issue #149 round 2): the even columns (kernels::*_even_columns).
inline void forward_group_pruned(Vec* f, std::size_t h, std::size_t k, const u32* r, std::size_t width) {
    if (width >= 1 && h >= kPruneForward) {
        if (width == 1) return kernels::forward_even_columns(f, h, r + slot(k), r + slot(2 * k));
        if (h >= 2 * width) return kernels::forward_columns(f, h, width, r + slot(k), r + slot(2 * k));
        if (h == width || 2 * h == width) return forward_group_half(f, h, k, r, 2 * h == width);
    }
    forward_group(f, h, k, r);
}

inline void inverse_group_pruned(Vec* f, std::size_t h, std::size_t k, const u32* ir, std::size_t width) {
    if (width >= 1 && h >= kPruneInverse) {
        if (width == 1) return kernels::inverse_even_columns(f, h, ir + slot(k), ir + slot(2 * k));
        if (h >= 2 * width) return kernels::inverse_columns(f, h, width, ir + slot(k), ir + slot(2 * k));
        if (h == width || 2 * h == width) return inverse_group_half(f, h, k, ir, 2 * h == width);
    }
    inverse_group(f, h, k, ir);
}

// forward_tiles() with pruned groups.
template <class Tile>
inline void forward_pruned_tiles(Vec* f, std::size_t nv, std::size_t count, std::size_t first, const u32* r,
                                 std::size_t width, Tile tile) {
    if (nv > kTile) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) {
            forward_group_pruned(f, nv / 4, k, r, width);
            forward_pruned_tiles(f, nv / 4, 4, 4 * k, r, width, tile);
        }
        return;
    }
    const std::size_t run = std::min(count, kTile / nv), total = run * nv;
    for (std::size_t k = first; k < first + count; k += run, f += total) {
        for (std::size_t h = nv / 4; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < total; j += 4 * h, ++g) forward_group_pruned(f + j, h, g, r, width);
        forward_bottom(f, total / 4, k * (nv / 4), r);
        tile(reinterpret_cast<u32*>(f), total);
    }
}

// inverse_trees_with() with pruned groups.
template <class Stage>
inline void inverse_pruned_with(Vec* f, std::size_t nv, std::size_t count, std::size_t first, const u32* ir,
                                std::size_t width, Stage stage) {
    if (nv > kTile) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) {
            inverse_pruned_with(f, nv / 4, 4, 4 * k, ir, width, stage);
            inverse_group_pruned(f, nv / 4, k, ir, width);
        }
        return;
    }
    const std::size_t run = std::min(count, kTile / nv), total = run * nv;
    for (std::size_t k = first; k < first + count; k += run, f += total) {
        stage(reinterpret_cast<u32*>(f), total);
        for (std::size_t h = 4; h < nv; h *= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < total; j += 4 * h, ++g) inverse_group_pruned(f + j, h, g, ir, width);
    }
}

// The top levels of the hooked ops, pruned (detail::PrunedLevel; width in vectors): the forward's
// writes zeros to the skipped columns, the inverse's leaves them unwritten. As forward_radix4_from,
// forward_radix2_from and the *_out functions above.
inline void forward_radix4_from_pruned(const u32* f, std::size_t n, u32* to, std::size_t h, const u32* roots,
                                       std::size_t width) {
    first_level_pruned<4>(f, n, to, 8 * h, 16, 8 * width, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        forward_radix4_part<decltype(blocks)::value>(s, reinterpret_cast<Vec*>(d), words / 8, h, roots);
    });
}
inline void forward_radix2_from_pruned(const u32* f, std::size_t n, u32* to, std::size_t h, std::size_t width) {
    first_level_pruned<2>(f, n, to, 8 * h, 8, 8 * width, [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
        forward_radix2_part<decltype(blocks)::value>(s, reinterpret_cast<Vec*>(d), words / 8, h);
    });
}

// The hooked inverse's top level: count words of the lower or upper half (upper) to out, scaled by
// s; columns with (j & width) set unwritten.
inline void inverse_top_pruned(u32* words, std::size_t nv, int lg, u32* out, std::size_t count, const u32* ir, u32 s,
                               bool upper, std::size_t width) {
    const std::size_t w = 8 * width;
    if (!has_radix2(lg)) {
        const std::size_t h = nv / 4;
        const u32 sz = multiply_mod(s, ir[slot(1)]);
        alignas(32) const Vec weights[4] = {broadcast(s), broadcast(quotient(s)), broadcast(sz), broadcast(quotient(sz))};
        if (upper) {
            return last_level_pruned<2>(words, 8 * h, out, count, 16, w, [&] [[gnu::always_inline]] (auto blocks, const u32* f, u32* d, std::size_t n) {
                constexpr int T = decltype(blocks)::value;
                if constexpr (T > 0) product_kernels::inverse_top4_upper_to<T>(reinterpret_cast<const Vec*>(f), d, n / 8, h, weights);
            });
        }
        return last_level_pruned<4>(words, 8 * h, out, count, 16, w, [&] [[gnu::always_inline]] (auto blocks, const u32* f, u32* d, std::size_t n) {
            constexpr int T = decltype(blocks)::value;
            if constexpr (T > 0) inverse_radix4_part<T>(reinterpret_cast<const Vec*>(f), d, n / 8, h, weights);
        });
    }
    if (nv == 8) {
        if (upper) return scale_radix2_upper_out(words, 4, out, count, s);
        return scale_radix2_out(words, 4, out, count, s);
    }
    const std::size_t q = nv / 8;
    const u32 z0 = ir[slot(1)], y1 = ir[slot(2)], z1 = ir[slot(3)], x1 = ir[slot(1)];
    const u32 twiddles[6] = {s, multiply_mod(s, z0), multiply_mod(s, y1), multiply_mod(s, z1), multiply_mod(s, x1), x1};
    alignas(32) Vec weights[12];
    for (int i = 0; i < 6; ++i) weights[2 * i] = broadcast(twiddles[i]), weights[2 * i + 1] = broadcast(quotient(twiddles[i]));
    if (upper) {
        return last_level_pruned<4>(words, 8 * q, out, count, 8, w, [&] [[gnu::always_inline]] (auto blocks, const u32* f, u32* d, std::size_t n) {
            constexpr int T = decltype(blocks)::value;
            if constexpr (T > 0) product_kernels::inverse_top_upper_to<T>(reinterpret_cast<const Vec*>(f), d, n / 8, q, weights);
        });
    }
    last_level_pruned<8>(words, 8 * q, out, count, 8, w, [&] [[gnu::always_inline]] (auto blocks, const u32* f, u32* d, std::size_t n) {
        constexpr int T = decltype(blocks)::value;
        if constexpr (T > 0) inverse_radix8_part<T>(reinterpret_cast<const Vec*>(f), d, n / 8, q, weights);
    });
}

// hook(leaves, first, count) on each run of leaves of the leaf form of in after its bottom level.
template <class Hook>
inline void forward_hooked(const u32* in, std::size_t n, u32* words, int lg, const u32* r, std::size_t width, Hook hook) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    const auto tile = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) { hook(t, std::size_t(t - words) / 8, leaves); };
    if (!has_radix2(lg)) {
        forward_radix4_from_pruned(in, n, words, nv / 4, r, width);
        return forward_pruned_tiles(f, nv / 4, 4, 0, r, width, tile);
    }
    forward_radix2_from_pruned(in, n, words, nv / 2, width);
    forward_pruned_tiles(f, nv / 2, 2, 0, r, width, tile);
}

// hook(leaves, first, count) writes each run of leaves (canonical, times 2^-32) before its bottom
// level; the top level writes count words of the half `output` picks to out.
template <class Hook>
inline void inverse_hooked(Hook hook, u32* words, u32* out, std::size_t count, int lg, const u32* ir, unsigned output,
                           std::size_t width) {
    const auto stage = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
        const std::size_t at = std::size_t(t - words);
        hook(t, at / 8, leaves);
        inverse_bottom(reinterpret_cast<Vec*>(t), leaves / 4, at / 32, ir);
    };
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8;
    if (!has_radix2(lg)) inverse_pruned_with(f, nv / 4, 4, 0, ir, width, stage);
    else if (nv == 8) inverse_pruned_with(f, 4, 2, 0, ir, width, stage);
    else inverse_pruned_with(f, nv / 8, 8, 0, ir, width, stage);
    inverse_top_pruned(words, nv, lg, out, count, ir, output_scale(lg, output), output & kUpperHalf, width);
}

// Doubling (Transform::truncate_blocks, join_blocks, forward_upper_hooked; issue #149 round 2), as
// intrinsics.hpp's: a block of nb = block / 8 vectors (nb >= 4) is group j's subtree if nb is a
// power of 4, else those of groups 2j, 2j + 1 (nb / 2 vectors) under a radix-2 level with r(j).
// Blocks go in runs of a tile.
template <class Hook>
inline void truncate_blocks(u32* words, std::size_t block, u32* copy, int lg, const u32* r, const u32* ir, Hook hook) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t blocks = (std::size_t(1) << lg) / block, nb = block / 8, hv = nb / 2;
    const std::size_t run = std::max<std::size_t>(1, kTile / nb);
    const bool aligned = std::countr_zero(nb) % 2 == 0;
    const auto tile = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) { hook(t, std::size_t(t - words) / 8, leaves); };
    for (std::size_t j0 = 0; j0 < blocks; j0 += run) {
        const std::size_t n = std::min(run, blocks - j0);
        Vec* const g = f + j0 * nb;
        // The levels within the blocks undone: the lower halves, < 2P, times nb.
        if (aligned && nb == 4) {
            inverse_bottom(g, n, j0, ir);
        } else if (aligned) {
            inverse_trees(g, nb / 4, 4 * n, 4 * j0, ir);
            for (std::size_t j = 0; j < n; ++j) inverse_group_half(g + j * nb, nb / 4, j0 + j, ir, true);
        } else {  // the radix-2 level's lower output: u + v
            inverse_trees(g, nb / 2, 2 * n, 2 * j0, ir);
            for (Vec *b = g, *e = g + n * nb; b < e; b += nb)
                for (std::size_t i = 0; i < hv; ++i) b[i] = reduce(add(b[i], b[i + hv]), 2 * kP);
        }
        // The lower halves to copy (canonical); the upper halves zero (aligned) or a copy of the
        // lower ones (a radix-2 level with a zero upper input), then the levels within the blocks.
        for (std::size_t j = 0; j < n; ++j) {
            Vec* b = g + j * nb;
            u32* c = copy + (j0 + j) * block;
            for (std::size_t i = 0; i < hv; ++i) {
                store(c + 8 * i, reduce(b[i], kP));
                if (!aligned) b[i + hv] = b[i];
            }
        }
        if (aligned && nb == 4) {
            for (std::size_t j = 0; j < n; ++j) g[4 * j + 2] = g[4 * j + 3] = _mm256_setzero_si256();
            forward_bottom(g, n, j0, r);
            tile(reinterpret_cast<u32*>(g), 4 * n);
        } else if (aligned) {
            for (std::size_t j = 0; j < n; ++j) forward_group_half(g + j * nb, nb / 4, j0 + j, r, true);
            forward_tiles(g, nb / 4, 4 * n, 4 * j0, r, tile);
        } else {
            forward_tiles(g, nb / 2, 2 * n, 2 * j0, r, tile);
        }
    }
}

// The first stage of inverse group k on quarters of h vectors, quarters 0 and 2 only: a + c, x (a - c).
inline void join_radix2(Vec* f, std::size_t h, std::size_t k, const u32* ir) {
    const Factor x = entry(ir, k);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec a = f[j], c = f[j + 2 * h];
        f[j] = reduce(add(a, c), 2 * kP), f[j + 2 * h] = multiply(diff(a, c), x);
    }
}

// Subtrees [first, first + count) of nv vectors at f: their inverse levels above the blocks of nb
// vectors, columns with (j & width) == 0 (inverse_group_pruned).
inline void join_trees(Vec* f, std::size_t nv, std::size_t count, std::size_t first, std::size_t nb, const u32* ir,
                       std::size_t width) {
    if (nv == nb) return;
    if (nv == 2 * nb) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) join_radix2(f, nv / 4, k, ir);
        return;
    }
    if (nv > kTile) {
        for (std::size_t k = first; k < first + count; ++k, f += nv) {
            join_trees(f, nv / 4, 4, 4 * k, nb, ir, width);
            inverse_group_pruned(f, nv / 4, k, ir, width);
        }
        return;
    }
    const std::size_t run = std::min(count, kTile / nv), total = run * nv;
    const bool aligned = std::countr_zero(nb) % 2 == 0;
    for (std::size_t k = first; k < first + count; k += run, f += total) {
        std::size_t h = nb;  // vectors per quarter of the lowest level above the blocks
        if (!aligned) {
            for (std::size_t j = 0, g = k * (nv / (2 * nb)); j < total; j += 2 * nb, ++g) join_radix2(f + j, nb / 2, g, ir);
            h = 2 * nb;
        }
        for (; h < nv; h *= 4)
            for (std::size_t j = 0, g = k * (nv / (4 * h)); j < total; j += 4 * h, ++g) inverse_group_pruned(f + j, h, g, ir, width);
    }
}

inline void join_blocks(u32* words, std::size_t block, u32 scale, int lg, const u32* ir) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t len = std::size_t(1) << lg, nv = len / 8, nb = block / 8, width = nb / 2;
    if (!has_radix2(lg)) join_trees(f, nv / 4, 4, 0, nb, ir, width);
    else join_trees(f, nv / 8, 8, 0, nb, ir, width);  // the top level holds the radix-2 level and both halves' last
    // The levels above the blocks multiply by nv / nb.
    inverse_top_pruned(words, nv, lg, words, len, ir, multiply_mod(scale, power(u32(nv / nb), kP - 2)), false, width);
}

// forward_upper with pruned groups and a hook (leaf indices from 2^lg / 8; words is the upper half
// of a leaf form of 2^(lg + 1) words). Under a radix-2 top the first level is group 1's, which
// leaves the skipped columns as they are: below 64 vectors a whole group at the column bit reads
// them, so they are zeroed first. Under group 0 the radix-2 level with r(1) writes zeros there.
template <class Hook>
inline void forward_upper_hooked(u32* words, int lg, const u32* r, std::size_t width, Hook hook) {
    auto* f = reinterpret_cast<Vec*>(words);
    const std::size_t nv = (std::size_t(1) << lg) / 8, h = nv / 2;
    const u32* const base = words - (std::size_t(1) << lg);
    const auto tile = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) { hook(t, std::size_t(t - base) / 8, leaves); };
    if (has_radix2(lg + 1)) {
        if (width && width < 64)
            for (std::size_t j = width; j < nv; j += 2 * width) std::fill_n(f + j, width, _mm256_setzero_si256());
        return forward_pruned_tiles(f, nv, 1, 1, r, width, tile);
    }
    const Factor z = entry(r, 1);
    const bool pruned = width >= 1 && width < h;
    for (std::size_t c = 0; c < h; c += pruned ? 2 * width : h) {
        const std::size_t end = pruned ? c + width : h;
        for (std::size_t i = c; i < end; ++i) {
            const Vec x = f[i], y = multiply(f[i + h], z);
            f[i] = add(x, y), f[i + h] = diff(x, y);
        }
        if (pruned)
            for (std::size_t i = end; i < end + width; ++i) f[i] = f[i + h] = _mm256_setzero_si256();
    }
    forward_pruned_tiles(f, h, 2, 2, r, width, tile);
}

}  // namespace fps::ntt::detail::avx2

namespace fps::ntt::variant {

struct avx2 {
    static void build_table(std::uint32_t* t, std::size_t count, bool inverse) {
        detail::avx2::build_table(t, count, inverse);
    }

    static void forward(std::uint32_t* f, int lg, const std::uint32_t* roots) { detail::avx2::forward(f, lg, roots); }

    static void forward_upper(std::uint32_t* f, int lg, const std::uint32_t* roots) {
        detail::avx2::forward_upper(f, lg, roots);
    }

    static void inverse(std::uint32_t* f, int lg, const std::uint32_t* inverse_roots) {
        detail::avx2::inverse(f, lg, inverse_roots);
    }

    static void inverse_upper(std::uint32_t* f, int lg, const std::uint32_t* inverse_roots) {
        detail::avx2::inverse_upper(f, lg, inverse_roots);
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

    static void multiply(const detail::Product& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        if (p.lg == 7 || p.lg == 8) return detail::avx2::product<detail::avx2::GroupBottom>(p, roots, inverse_roots);
        detail::avx2::product<detail::avx2::PairBottom>(p, roots, inverse_roots);
    }

    static void square(const detail::Product& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        detail::avx2::square<detail::avx2::PairSquareBottom>(p, roots, inverse_roots);
    }

    // Leaf-form operations (ntt.hpp, Transform): their contracts are the Transform calls'.
    static void leaf_products(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out, int lg,
                              const std::uint32_t* roots, bool accumulate) {
        const std::size_t groups = (std::size_t(1) << lg) / 32;
        if (accumulate) return detail::avx2::Leaf::products<true>(a, b, out, groups, 0, roots);
        detail::avx2::Leaf::products<false>(a, b, out, groups, 0, roots);
    }

    static void forward_values(std::uint32_t* f, int lg, const std::uint32_t* roots, const std::uint32_t* rows) {
        detail::avx2::forward_values(f, lg, roots, rows);
    }

    static void inverse(const std::uint32_t* from, std::uint32_t* f, std::uint32_t* out, std::size_t count, int lg,
                        const std::uint32_t* inverse_roots, const std::uint32_t* rows) {
        detail::avx2::inverse(from, f, out, count, lg, inverse_roots, rows);
    }

    static void forward_from(const std::uint32_t* in, std::size_t n, std::uint32_t* f, int lg, const std::uint32_t* roots) {
        detail::avx2::forward_from(in, n, f, lg, roots);
    }

    static void forward_product(const std::uint32_t* in, std::size_t n, const std::uint32_t* b, std::uint32_t* f, int lg,
                                const std::uint32_t* roots) {
        detail::avx2::forward_product(in, n, b, f, lg, roots);
    }

    static void forward_upper_from(const std::uint32_t* in, std::size_t n, std::uint32_t* f, int lg,
                                   const std::uint32_t* roots) {
        detail::avx2::forward_upper_from(in, n, f, lg, roots);
    }

    static void cyclic_product(const std::uint32_t* in, std::size_t n, const std::uint32_t* b, std::uint32_t* f,
                               std::uint32_t* out, std::size_t count, int lg, const std::uint32_t* roots,
                               const std::uint32_t* inverse_roots, unsigned output) {
        detail::avx2::cyclic_product(in, n, b, f, out, count, lg, roots, inverse_roots, output);
    }

    static void inverse_product(const detail::LeafTerm* terms, int n, std::uint32_t* f, std::uint32_t* out,
                                std::size_t count, int lg, const std::uint32_t* roots, const std::uint32_t* inverse_roots,
                                unsigned output) {
        detail::avx2::inverse_product(terms, n, f, out, count, lg, roots, inverse_roots, output);
    }

    // Hooked ops; width in words (Transform::forward(in, F, hook, ...)).
    template <class Hook>
    static void forward_hooked(const std::uint32_t* in, std::size_t n, std::uint32_t* f, int lg, const std::uint32_t* roots,
                               std::size_t width, Hook hook) {
        detail::avx2::forward_hooked(in, n, f, lg, roots, width / 8, hook);
    }

    template <class Hook>
    static void inverse_hooked(Hook hook, std::uint32_t* f, std::uint32_t* out, std::size_t count, int lg,
                               const std::uint32_t*, const std::uint32_t* inverse_roots, unsigned output, std::size_t width) {
        detail::avx2::inverse_hooked(hook, f, out, count, lg, inverse_roots, output, width / 8);
    }

    // Doubling ops (Transform::truncate_blocks, join_blocks, forward_upper_hooked); width in words.
    template <class Hook>
    static void truncate_blocks(std::uint32_t* f, std::size_t block, std::uint32_t* copy, int lg, const std::uint32_t* roots,
                                const std::uint32_t* inverse_roots, Hook hook) {
        detail::avx2::truncate_blocks(f, block, copy, lg, roots, inverse_roots, hook);
    }

    static void join_blocks(std::uint32_t* f, std::size_t block, std::uint32_t scale, int lg,
                            const std::uint32_t* inverse_roots) {
        detail::avx2::join_blocks(f, block, scale, lg, inverse_roots);
    }

    template <class Hook>
    static void forward_upper_hooked(std::uint32_t* f, int lg, const std::uint32_t* roots, std::size_t width, Hook hook) {
        detail::avx2::forward_upper_hooked(f, lg, roots, width / 8, hook);
    }
};

}  // namespace fps::ntt::variant
