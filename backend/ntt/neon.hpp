// Variant neon (arm): intrinsics<Neon> with asm kernels generated and scheduled for Neoverse V2
// (gen_neon_kernels.py, neon_kernels.hpp) in place of the product bottoms (products, squares,
// Transform's products by stored leaves), the group levels and the bottom level of transforms.
// The kernels need SVE2 (umulh); without it (the Mac's M4) neon is intrinsics<Neon> under its own
// name, so every arm build has the same candidate list (select.hpp). Same splitting, tables, lazy
// ranges and outputs as every variant. Design and measurements: notes.md (neon), notes/151-r1.md.
#pragma once

#include <cstddef>
#include <cstdint>

#include "backend/ntt/intrinsics.hpp"
#include "backend/ntt/neon_kernels.hpp"

#if defined(__ARM_NEON) && defined(__aarch64__)

namespace fps::ntt::detail::neon {

// simd::Neon under a name of its own: intrinsics.hpp's Generated<Vec> swaps in the asm.
struct Vec : simd::Neon {
    static constexpr const char* name = "neon";
};

}  // namespace fps::ntt::detail::neon

#ifdef FPS_NTT_NEON_KERNELS
namespace fps::ntt::detail::intrinsics {

template <>
struct Generated<neon::Vec> {
    static constexpr bool kOn = true;

    // Group k on quarters of q words, columns [0, words / 4), words a multiple of 16: in place with
    // inputs < kIn P (forward), or from other memory with inputs < 2P (forward_from). False if not
    // run (canonical inputs, kIn = 1: the intrinsics skip more reductions there).
    template <int kIn>
    [[gnu::always_inline]] static bool forward(u32* f, std::size_t q, std::size_t words, std::size_t k, const u32* r) {
        if constexpr (kIn == 2) return forward_from<2>(f, f, q, words, k, r);
        if (kIn != 4 || words % 16 != 0) return false;
        if (k == 0) neon_kernels::forward_identity_group(f, q, words, r);
        else neon_kernels::forward_group(f, q, words, r + slot(k), r + slot(2 * k));
        return true;
    }
    template <int kIn>
    [[gnu::always_inline]] static bool forward_from(const u32* from, u32* f, std::size_t q, std::size_t words, std::size_t k,
                                                    const u32* r) {
        if (kIn != 2 || words % 16 != 0) return false;
        if (k == 0) neon_kernels::forward_identity_group2(from, f, q, words, r);
        else neon_kernels::forward_group2(from, f, q, words, r + slot(k), r + slot(2 * k));
        return true;
    }
    // Inverse group k in place, in and out < 2P.
    [[gnu::always_inline]] static bool inverse(u32* f, std::size_t q, std::size_t k, const u32* ir) {
        if (q % 16 != 0) return false;
        if (k == 0) neon_kernels::inverse_identity_group(f, q, q, ir);
        else neon_kernels::inverse_group(f, q, q, ir + slot(k), ir + slot(2 * k));
        return true;
    }

    // The bottom level of a transform (Bottom::forward, inverse), count even: two groups per kernel.
    static void bottom_forward(u32* f, std::size_t count, std::size_t first, const u32* r) {
        for (std::size_t g = 0, k = first; g < count; g += 2, k += 2)
            neon_kernels::bottom_forward_pair(f + 32 * g, r + slot(k), r + slot(2 * k));
    }
    static void bottom_inverse(u32* f, std::size_t count, std::size_t first, const u32* ir) {
        for (std::size_t g = 0, k = first; g < count; g += 2, k += 2)
            neon_kernels::bottom_inverse_pair(f + 32 * g, ir + slot(k), ir + slot(2 * k));
    }

    // The bottom stage of a product (Bottom::multiply), count even (first is then even): pairs of
    // groups, the forward half of pair j + 1 (into one leaf buffer) in the same asm statement as
    // the products and inverse of pair j (from the other).
    template <bool kSquare>
    static void multiply(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        alignas(64) u32 leaves[2][2 * 96];  // per group: rows [w_t A_t, A_t] (4 x 16 words), B_t (4 x 8)
        if constexpr (kSquare) neon_kernels::bottom_rows_first(a, leaves[0], r + slot(first), r + slot(2 * first));
        else neon_kernels::bottom_first(a, b, leaves[0], r + slot(first), r + slot(2 * first));
        for (std::size_t g = 0, k = first;; g += 2, k += 2) {
            u32 *cur = leaves[g / 2 % 2], *next = leaves[(g / 2 + 1) % 2];
            const u32 *ix = ir + slot(k), *iyz = ir + slot(2 * k), *x = r + slot(k + 2), *yz = r + slot(2 * k + 4);
            if constexpr (kSquare) {
                if (g + 2 == count) return neon_kernels::bottom_square_last(a + 32 * g, cur, cur + 96, ix, iyz);
                neon_kernels::bottom_square_both(a + 32 * (g + 2), next, x, yz, a + 32 * g, cur, cur + 96, ix, iyz);
            } else {
                if (g + 2 == count) return neon_kernels::bottom_last(a + 32 * g, cur, cur + 96, ix, iyz);
                neon_kernels::bottom_both(a + 32 * (g + 2), b + 32 * (g + 2), next, x, yz, a + 32 * g, cur, cur + 96, ix, iyz);
            }
        }
    }

    // Bottom::multiply_stored: b holds the leaves of group first and on (canonical).
    static void multiply_stored(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        alignas(64) u32 leaves[2][2 * 96];
        neon_kernels::bottom_rows_first(a, leaves[0], r + slot(first), r + slot(2 * first));
        for (std::size_t g = 0, k = first;; g += 2, k += 2) {
            u32 *cur = leaves[g / 2 % 2], *next = leaves[(g / 2 + 1) % 2];
            const u32 *ix = ir + slot(k), *iyz = ir + slot(2 * k);
            if (g + 2 == count) return neon_kernels::bottom_stored_last(a + 32 * g, cur, cur + 96, b + 32 * g, ix, iyz);
            neon_kernels::bottom_stored_both(a + 32 * (g + 2), next, r + slot(k + 2), r + slot(2 * k + 4), a + 32 * g, cur,
                                             cur + 96, b + 32 * g, ix, iyz);
        }
    }
};

}  // namespace fps::ntt::detail::intrinsics
#endif

namespace fps::ntt::variant {

// The candidate "neon" (select.hpp).
struct neon : intrinsics<detail::neon::Vec> {};

}  // namespace fps::ntt::variant

#endif
