// The Taylor shift's choices per target: the simd struct, vectors per chain and chain step of its
// scans (K = 1 below kNarrowBelow), and the largest n for Horner's rule.
//   -DFPS_TAYLOR_SHIFT_ISA=<struct>     force the scans' struct (e.g. Avx512Vl)
//   -DFPS_TAYLOR_SHIFT_VECTORS=<K>      force K vectors per chain (W = K S::lanes lanes)
//   -DFPS_TAYLOR_SHIFT_STEP=<step>      force the chain step: montgomery or shoup
//   -DFPS_TAYLOR_SHIFT_SMALL=<n>        force the largest n for Horner's rule
// Measurements: notes.md.
#pragma once

#include <cstddef>

#include "backend/factorials/select.hpp"
#include "backend/simd/select.hpp"

namespace fps::taylor_shift {

#if defined(FPS_TAYLOR_SHIFT_ISA)
using ScanIsa = simd::FPS_TAYLOR_SHIFT_ISA;
#elif defined(__AVX512F__) && defined(__AVX512BW__)
using ScanIsa = simd::Avx512;
#elif defined(__AVX2__)
using ScanIsa = simd::Avx2;
#elif defined(__SSE4_1__)
using ScanIsa = simd::Sse;
#elif defined(__ARM_NEON) && defined(__aarch64__)
using ScanIsa = simd::Neon;
#else
using ScanIsa = simd::Scalar;
#endif

// K and the chain step per target: the whole shift with each variant, same run, n = 2^8 .. 2^21
// (notes/124-r1.md). Zen 3: K = 1 Shoup (0.92-0.98 of K = 2 Montgomery from 2^19, within 1%
// below; lc-opt has 32 lanes). Emerald Rapids: zmm K = 2 Montgomery, K = 1 below 2^10 (0.85 at
// 2^8). Zen 5: zmm K = 1 Shoup (0.75 at 2^8, 0.92 from 2^20). Neoverse V2: K = 2 Shoup (SVE2's
// umulh; 0.94-1.00 of K = 4 Montgomery).
#if defined(FPS_TAYLOR_SHIFT_VECTORS)
inline constexpr int kScanVectors = FPS_TAYLOR_SHIFT_VECTORS;
#elif defined(__znver5__) || (defined(__AVX2__) && !defined(__AVX512F__))
inline constexpr int kScanVectors = 1;
#elif defined(__AVX512F__) || defined(__ARM_NEON) || defined(__SSE4_1__)
inline constexpr int kScanVectors = 2;
#else
inline constexpr int kScanVectors = 8;  // Scalar: 8 lanes
#endif
// Below this n the scans run K = 1 (fewer lanes repeat fewer positions at small n).
#if defined(__AVX512F__) && !defined(__znver5__)
inline constexpr std::size_t kNarrowBelow = 1024;
#else
inline constexpr std::size_t kNarrowBelow = 0;
#endif

// The scans' chain step for i! and 1 / k! (taylor_shift.hpp, Scan).
#if defined(FPS_TAYLOR_SHIFT_STEP)
inline constexpr factorials::Step kScanStep = factorials::Step::FPS_TAYLOR_SHIFT_STEP;
#elif defined(__znver5__) || (defined(__AVX2__) && !defined(__AVX512F__)) || defined(__ARM_FEATURE_SVE2)
inline constexpr factorials::Step kScanStep = factorials::Step::shoup;
#else
inline constexpr factorials::Step kScanStep = factorials::Step::montgomery;
#endif

// Horner's rule up to n = kSmall: against the lanes (K = 1), same run, Horner took 0.77-0.86 of
// their time at n = 80 on Zen 3, Emerald Rapids and Zen 5 (1.00 on Neoverse V2), and 1.01-1.33
// from 96 on (the lanes run a transform of 256 words for 65 <= n <= 128).
#if defined(FPS_TAYLOR_SHIFT_SMALL)
inline constexpr std::size_t kSmall = FPS_TAYLOR_SHIFT_SMALL;
#else
inline constexpr std::size_t kSmall = 88;
#endif

}  // namespace fps::taylor_shift
