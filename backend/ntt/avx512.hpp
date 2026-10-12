// Variant avx512: the intrinsics variant on zmm (intrinsics.hpp, Ntt<S>) with generated asm in
// place of two kernels (avx512_kernels.hpp, gen_avx512_kernels.py, scheduled for Emerald Rapids):
// the forward radix-4 groups k > 0 of 4 columns and more, in place, and the bottom stage of
// products (forward butterflies, leaf products, inverse butterflies; also squares and Transform's
// cyclic products). Same splitting, tables, lazy ranges and outputs as every variant. The other
// kernels ran as fast in asm as in intrinsics on Emerald Rapids and stay intrinsics. Design and
// measurements: notes.md (avx512), notes/150-r1.md.
#pragma once

#include "backend/ntt/intrinsics.hpp"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#include "backend/ntt/avx512_kernels.hpp"

namespace fps::ntt::detail::avx512 {

// simd::Avx512 under a name of its own: intrinsics.hpp's Generated<Zmm> swaps in the asm.
struct Zmm : simd::Avx512 {
    static constexpr const char* name = "avx512";
};

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
        namespace K = avx512_kernels;
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
        return true;
    }

    // The bottom stage of a product, groups [first, first + count) (both even), as WideBottom's
    // multiply: pair j's leaf products and inverse butterflies interleaved with pair j + 1's
    // forward butterflies (the *_both kernels), through two leaf buffers.
    template <bool kSquare>
    static void multiply(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        namespace K = avx512_kernels;
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
        namespace K = avx512_kernels;
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
