// The factorials variant per target and size n:
//   n < kMinChains                 the scalar loop (model.hpp)
//   kMinChains <= n                Chains<Isa, kVectors, P>, with non-temporal stores from
//                                  n = kMinStream on (Avx512 only)
// -DFPS_FACTORIALS_ISA=<struct>, -DFPS_FACTORIALS_VECTORS=<K>, -DFPS_FACTORIALS_MIN_CHAINS=<n> and
// -DFPS_FACTORIALS_MIN_STREAM=<n> force them. Measurements: notes.md.
#pragma once

#include <cstddef>
#include <limits>

#include "backend/simd/select.hpp"

namespace fps::factorials {

#if defined(FPS_FACTORIALS_ISA)
using Isa = simd::FPS_FACTORIALS_ISA;
#else
using Isa = simd::Native;
#endif

#if defined(FPS_FACTORIALS_VECTORS)
inline constexpr int kVectors = FPS_FACTORIALS_VECTORS;
#else
// x86: 2 (4 was 10-30% slower in the scan pass); Neon (4 lanes, 32 registers) and Scalar: 4.
inline constexpr int kVectors = Isa::lanes == 1 || (Isa::lanes == 4 && Isa::registers == 32) ? 4 : 2;
#endif

// chains / loop time at the threshold and below it.
#if defined(FPS_FACTORIALS_MIN_CHAINS)
inline constexpr std::size_t kMinChains = FPS_FACTORIALS_MIN_CHAINS;
#elif defined(__znver5__)
inline constexpr std::size_t kMinChains = 96;  // Zen 5: 0.88 at 96, 1.28 at 64
#elif defined(__AVX512F__)
inline constexpr std::size_t kMinChains = 128;  // Emerald Rapids: 0.90 at 128, 1.17 at 96
#else
inline constexpr std::size_t kMinChains = 64;  // Zen 3: 0.93, Neoverse V2: 0.85 at 64
#endif

// Non-temporal stores, from the size where the tables (8 n bytes) no longer stay in L2, ns per
// position against plain stores: Emerald Rapids 0.47 against 0.45-0.59 at 2^19, 0.60-0.73 at
// 2^20; Zen 5 0.31 against 0.36 at 2^17, 0.28 against 0.32 at 2^16.
#if defined(FPS_FACTORIALS_MIN_STREAM)
inline constexpr std::size_t kMinStream = FPS_FACTORIALS_MIN_STREAM;
#elif defined(__znver5__)
inline constexpr std::size_t kMinStream = std::size_t(1) << 17;
#elif defined(__AVX512F__)
inline constexpr std::size_t kMinStream = std::size_t(1) << 19;
#else
inline constexpr std::size_t kMinStream = std::numeric_limits<std::size_t>::max();
#endif

}  // namespace fps::factorials
