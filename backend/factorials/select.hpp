// The factorials variant per target and size n:
//   n < kMinChains                 the scalar loop (model.hpp)
//   kMinChains <= n < kMinLarge    Chains<Isa, kVectors, P>
//   kMinLarge <= n                 Chains<IsaLarge, kVectorsLarge, P>: fewer lanes, so fewer
//                                  write streams, where many streams stall on memory
// -DFPS_FACTORIALS_ISA=<struct>, -DFPS_FACTORIALS_VECTORS=<K> and -DFPS_FACTORIALS_MIN_CHAINS=<n>
// force them (a forced struct runs at every size). Measurements: notes.md.
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

// zmm with 32 lanes (64 streams) against ymm with 16 lanes, ns per position: Emerald Rapids
// 0.45-0.49 or 0.71-0.83 (by the tables' placement) from 2^18 on, against 0.54-0.62; Zen 5
// 0.37-0.47 against 0.50 up to 2^22, 0.55-0.72 against 0.50-0.54 from 5 2^20 on.
#if !defined(FPS_FACTORIALS_ISA) && defined(__znver5__)
using IsaLarge = simd::Avx512Vl;
inline constexpr int kVectorsLarge = 2;
inline constexpr std::size_t kMinLarge = std::size_t(5) << 20;
#elif !defined(FPS_FACTORIALS_ISA) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
using IsaLarge = simd::Avx512Vl;
inline constexpr int kVectorsLarge = 2;
inline constexpr std::size_t kMinLarge = std::size_t(1) << 19;
#else
using IsaLarge = Isa;
inline constexpr int kVectorsLarge = kVectors;
inline constexpr std::size_t kMinLarge = std::numeric_limits<std::size_t>::max();
#endif

}  // namespace fps::factorials
