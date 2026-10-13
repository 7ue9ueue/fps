// Per target: the simd structs of bostan_mori.hpp's loops and the size up to which coefficient()
// runs schoolbook steps (measured: notes.md). -DFPS_BOSTAN_MORI_SCHOOLBOOK_MAX=<words> forces it.
#pragma once

#include <cstddef>

#include "backend/simd/select.hpp"
#include "backend/tune/tune.hpp"

namespace fps::bostan_mori {

// The halving pass works on rows of 8 words: 8-lane structs (ymm on AVX-512), 4 or 1.
#if defined(__AVX2__) && defined(__AVX512VL__)
using PassIsa = simd::Avx512Vl;
#elif defined(__AVX2__)
using PassIsa = simd::Avx2;
#elif defined(__SSE4_1__)
using PassIsa = simd::Sse;
#elif defined(__ARM_NEON) && defined(__aarch64__)
using PassIsa = simd::Neon;
#else
using PassIsa = simd::Scalar;
#endif
using SchoolbookIsa = PassIsa;

// Words of p and q the schoolbook steps take at most (their stack arrays).
inline constexpr std::size_t kSchoolbookCapacity = 256;

// coefficient(): schoolbook while p and q have at most this many words, the values form beyond.
#if defined(FPS_BOSTAN_MORI_SCHOOLBOOK_MAX)
inline constexpr std::size_t kSchoolbookMax = FPS_BOSTAN_MORI_SCHOOLBOOK_MAX;
#else
inline constexpr std::size_t kSchoolbookMax = [] {
    switch (tune::kTarget) {
        case tune::Target::avx2: return std::size_t(96);    // Zen 3
        case tune::Target::avx512: return std::size_t(96);  // Emerald Rapids
        case tune::Target::zen5: return std::size_t(88);
        case tune::Target::arm: return std::size_t(76);     // Neoverse V2
        default: return std::size_t(64);                    // sse, scalar: not timed
    }
}();
#endif
static_assert(kSchoolbookMax <= kSchoolbookCapacity);

}  // namespace fps::bostan_mori
