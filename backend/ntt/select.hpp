// The NTT variant per op and target. -DFPS_NTT_FORWARD=<variant> (and _INVERSE, _MULTIPLY)
// forces one, e.g. -DFPS_NTT_FORWARD='intrinsics<fps::simd::Avx512>'. Variants: scalar (every
// target), intrinsics<S> (every simd struct S the target has), avx2 (__AVX2__), and
// by_length<lg, Small, Large> (Small below 2^lg, Large from there).
#pragma once

#include <cstddef>
#include <cstdint>

#include "backend/ntt/intrinsics.hpp"
#include "backend/ntt/scalar.hpp"
#include "backend/simd/select.hpp"
#ifdef __AVX2__
#include "backend/ntt/avx2.hpp"
#endif

namespace fps::ntt {

namespace variant {

// Small for lengths below 2^kFromLog, Large from there. Every variant builds the same tables.
template <int kFromLog, class Small, class Large>
struct by_length {
    static void build_table(std::uint32_t* t, std::size_t count, bool inverse) { Large::build_table(t, count, inverse); }

    static void forward(std::uint32_t* f, int lg, const std::uint32_t* roots) {
        if (lg < kFromLog) return Small::forward(f, lg, roots);
        Large::forward(f, lg, roots);
    }

    static void inverse(std::uint32_t* f, int lg, const std::uint32_t* inverse_roots) {
        if (lg < kFromLog) return Small::inverse(f, lg, inverse_roots);
        Large::inverse(f, lg, inverse_roots);
    }

    static void multiply(std::uint32_t* a, std::uint32_t* b, std::size_t n, std::size_t m, int lg,
                         const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        if (lg < kFromLog) return Small::multiply(a, b, n, m, lg, roots, inverse_roots);
        Large::multiply(a, b, n, m, lg, roots, inverse_roots);
    }
};

}  // namespace variant

// Native: the variant that builds the tables of ntt::Transform; Native<Op>: the fastest per op.
// avx2 (asm) on AVX2 without AVX-512: intrinsics<Avx2> is 1.06-1.22x slower on Zen 3. With
// AVX-512, intrinsics<Avx512> (zmm) from the lengths where it beats the asm (notes.md), the asm
// below. Elsewhere intrinsics on the target's widest struct.
#if defined(__AVX2__) && !(defined(__AVX512F__) && defined(__AVX512BW__))
using Native = variant::avx2;
using NativeForward = Native;
using NativeInverse = Native;
using NativeMultiply = Native;
#elif defined(__AVX2__)
using Native = variant::intrinsics<fps::simd::Native>;
#if defined(__znver4__) || defined(__znver5__)
using NativeForward = variant::by_length<8, variant::avx2, Native>;  // Zen 5
using NativeInverse = variant::by_length<8, variant::avx2, Native>;
using NativeMultiply = variant::by_length<8, variant::avx2, Native>;
#else
using NativeForward = variant::by_length<9, variant::avx2, Native>;  // Emerald Rapids
using NativeInverse = Native;
using NativeMultiply = variant::by_length<10, variant::avx2, Native>;
#endif
#else
using Native = variant::intrinsics<fps::simd::Native>;
using NativeForward = Native;
using NativeInverse = Native;
using NativeMultiply = Native;
#endif

#ifdef FPS_NTT_FORWARD
using SelectedForward = variant::FPS_NTT_FORWARD;
#else
using SelectedForward = NativeForward;
#endif

#ifdef FPS_NTT_INVERSE
using SelectedInverse = variant::FPS_NTT_INVERSE;
#else
using SelectedInverse = NativeInverse;
#endif

#ifdef FPS_NTT_MULTIPLY
using SelectedMultiply = variant::FPS_NTT_MULTIPLY;
#else
using SelectedMultiply = NativeMultiply;
#endif

}  // namespace fps::ntt
