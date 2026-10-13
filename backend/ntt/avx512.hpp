// Variant avx512: the intrinsics variant on zmm (intrinsics.hpp, Ntt<S>) with generated asm in
// place of some kernels, scheduled for the target's machine by gen_avx512_kernels.py:
// avx512_kernels.hpp for Emerald Rapids (avx512 target), zen5_kernels.hpp for Zen 5 (zen5 target,
// -march=znver5). On both: the forward radix-4 groups k > 0 of 4 columns and more, in place, and
// the bottom stage of products (forward butterflies, leaf products, inverse butterflies; also
// squares and Transform's cyclic products). On Zen 5 also the identity and inverse groups, forward
// groups of two columns and the transforms' bottom level. Same splitting, tables, lazy ranges and
// outputs as every variant. The other kernels ran as fast in asm as in intrinsics (or, on Zen 5,
// slowed products) and stay intrinsics. Design and measurements: notes.md (avx512),
// notes/150-r1.md, notes/152-r1.md.
#pragma once

#include "backend/ntt/intrinsics.hpp"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#if defined(__znver4__) || defined(__znver5__)
#include "backend/ntt/zen5_kernels.hpp"
#else
#include "backend/ntt/avx512_kernels.hpp"
#endif

namespace fps::ntt::detail::avx512 {

// simd::Avx512 under a name of its own: intrinsics.hpp's Generated<Zmm> swaps in the asm.
struct Zmm : simd::Avx512 {
    static constexpr const char* name = "avx512";
};

#if defined(__znver4__) || defined(__znver5__)
namespace kernels = zen5_kernels;
#else
namespace kernels = avx512_kernels;
#endif

}  // namespace fps::ntt::detail::avx512

namespace fps::ntt::detail::intrinsics {

template <>
struct Narrow<avx512::Zmm> {
    using type = simd::Avx512Vl;
};

template <>
struct Generated<avx512::Zmm> {
    static constexpr bool kOn = true;

    // Group k > 0 on quarters of q words, columns [0, words / 16) in place, if their count is a
    // multiple of 4; in < kIn P, out < 4P. False if not run (group 0, or fewer columns: the
    // intrinsics ran as fast there). Prefetching where intrinsics.hpp's groups prefetch.
    template <int kIn>
    [[gnu::always_inline]] static bool forward(u32* f, std::size_t q, std::size_t words, std::size_t k, const u32* r) {
        namespace K = avx512::kernels;
#if !defined(__znver4__) && !defined(__znver5__)
        if (k == 0 || words % 64 != 0) return false;
        const std::size_t columns = words / 16;
        const u32 *x = r + slot(k), *y = r + slot(2 * k);
        const bool pf = kPrefetch > 0 && q >= kPrefetchMin;
        if constexpr (kIn <= 2) {
            if (pf) K::forward_in2_prefetch(f, q, columns, x, y);
            else K::forward_in2(f, q, columns, x, y);
        } else {
            if (pf) K::forward_prefetch(f, q, columns, x, y);
            else K::forward(f, q, columns, x, y);
        }
#else  // Zen 5: group 0 too (identity groups); no prefetch (kPrefetch)
        if (words % 64 != 0) return false;
        const std::size_t columns = words / 16;
        if (k == 0) {
            if constexpr (kIn == 1) K::forward_identity_in1(f, q, columns, r);
            else if constexpr (kIn == 2) K::forward_identity_in2(f, q, columns, r);
            else K::forward_identity(f, q, columns, r);
            return true;
        }
        const u32 *x = r + slot(k), *y = r + slot(2 * k);
        if constexpr (kIn <= 2) K::forward_in2(f, q, columns, x, y);
        else K::forward(f, q, columns, x, y);
#endif
        return true;
    }

#if defined(__znver4__) || defined(__znver5__)
    // Zen 5 (notes/152-r1.md): inverse groups of 4 columns and more (identity groups of 8 and
    // more), in place, in and out < 2P; false if not run (groups of 2 columns).
    [[gnu::always_inline]] static bool inverse(u32* f, std::size_t q, std::size_t k, const u32* ir) {
        namespace K = avx512::kernels;
        if (k == 0) {
            if (q % 128 != 0) return false;
            K::inverse_identity(f, q, q / 16, ir);
            return true;
        }
        if (q % 64 != 0) return false;
        K::inverse(f, q, q / 16, ir + slot(k), ir + slot(2 * k));
        return true;
    }

    // Forward groups [k, k + count) of quarters of 32 words (two columns each), 4 per statement,
    // if k and count are multiples of 4; false if not run. In < 4P, out < 4P. (The same for inverse
    // groups made the inverse transform 1-3% faster at 2^10-2^12 but products 6-8% slower at 2^10:
    // notes/152-r1.md.)
    static bool forward_groups(u32* f, std::size_t q, std::size_t k, std::size_t count, const u32* r) {
        namespace K = avx512::kernels;
        if (q != 32 || k % 4 != 0 || count % 4 != 0) return false;
        for (const std::size_t end = k + count; k < end; k += 4, f += 512) {
            if (k == 0) K::forward_groups4_first(f, r, r);
            else K::forward_groups4(f, r + slot(k), r + slot(2 * k));
        }
        return true;
    }

    // The bottom level of groups [first, first + count) (both even): 4 pairs of groups per
    // statement from a multiple of 8, then 2 from a multiple of 4; other pairs by the intrinsics
    // (one pair per statement ran as fast). Forward: in < 4P, canonical leaves out; inverse: < 2P.
    static void bottom_forward(u32* f, std::size_t count, std::size_t first, const u32* r) {
        namespace K = avx512::kernels;
        const std::size_t end = first + count;
        for (std::size_t k = first; k < end;) {
            if (k % 8 == 0 && k + 8 <= end) K::bottom_forward4(f, r + slot(k), r + slot(2 * k)), k += 8, f += 256;
            else if (k % 4 == 0 && k + 4 <= end) K::bottom_forward2(f, r + slot(k), r + slot(2 * k)), k += 4, f += 128;
            else WideBottom<avx512::Zmm>::forward(f, 2, k, r), k += 2, f += 64;
        }
    }
    static void bottom_inverse(u32* f, std::size_t count, std::size_t first, const u32* ir) {
        namespace K = avx512::kernels;
        const std::size_t end = first + count;
        for (std::size_t k = first; k < end;) {
            if (k % 8 == 0 && k + 8 <= end) K::bottom_inverse4(f, ir + slot(k), ir + slot(2 * k)), k += 8, f += 256;
            else if (k % 4 == 0 && k + 4 <= end) K::bottom_inverse2(f, ir + slot(k), ir + slot(2 * k)), k += 4, f += 128;
            else WideBottom<avx512::Zmm>::inverse(f, 2, k, ir), k += 2, f += 64;
        }
    }
#endif

    // The bottom stage of a product, groups [first, first + count) (both even), as WideBottom's
    // multiply: pair j's leaf products and inverse butterflies interleaved with pair j + 1's
    // forward butterflies (the *_both kernels), through two leaf buffers.
    template <bool kSquare>
    static void multiply(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        namespace K = avx512::kernels;
        alignas(64) u32 buf[2][K::kLeafBuffer];
        const std::size_t end = first + count;
        if constexpr (kSquare) K::square_first(a, buf[0], r + slot(first), r + slot(2 * first));
        else K::multiply_first(a, b, buf[0], r + slot(first), r + slot(2 * first));
        for (std::size_t k = first, cur = 0; k < end; k += 2, a += 64, b += 64, cur ^= 1) {
            const u32 *ix = ir + slot(k), *iy = ir + slot(2 * k);
            if (k + 2 == end) {
                if constexpr (kSquare) K::square_last(a, buf[cur], ix, iy);
                else K::multiply_last(a, buf[cur], ix, iy);
                break;
            }
            const u32 *x = r + slot(k + 2), *y = r + slot(2 * k + 4);
            if constexpr (kSquare) K::square_both(a + 64, buf[cur ^ 1], x, y, a, buf[cur], ix, iy);
            else K::multiply_both(a + 64, b + 64, buf[cur ^ 1], x, y, a, buf[cur], ix, iy);
        }
    }

    // WideBottom::multiply_stored: b the leaf form of the groups (canonical), at b + 32 (k - first).
    static void multiply_stored(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r,
                                const u32* ir) {
        namespace K = avx512::kernels;
        alignas(64) u32 buf[2][K::kLeafBuffer];
        const std::size_t end = first + count;
        K::square_first(a, buf[0], r + slot(first), r + slot(2 * first));
        for (std::size_t k = first, cur = 0; k < end; k += 2, a += 64, b += 64, cur ^= 1) {
            const u32 *ix = ir + slot(k), *iy = ir + slot(2 * k);
            if (k + 2 == end) {
                K::stored_last(a, buf[cur], b, ix, iy);
                break;
            }
            K::stored_both(a + 64, buf[cur ^ 1], r + slot(k + 2), r + slot(2 * k + 4), a, buf[cur], b, ix, iy);
        }
    }
};

}  // namespace fps::ntt::detail::intrinsics

namespace fps::ntt::variant {

// The candidate "avx512" (select.hpp).
struct avx512 : intrinsics<detail::avx512::Zmm> {};

}  // namespace fps::ntt::variant

#endif
