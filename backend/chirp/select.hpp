// The chirp kernel's choices per target: the struct of its passes and the vectors per step of
// Terms (K), and of products_lanes (scan.hpp). -DFPS_CHIRP_VECTORS=<K> and
// -DFPS_CHIRP_LANES_VECTORS=<K> force them. Measurements behind the choices: notes.md.
#pragma once

#include <cstddef>

#include "backend/simd/select.hpp"

namespace fps::chirp {

using Wide = simd::Native;

// Terms per step for chirp and multiply_chirp: kSmallVectors below kLargeSize terms, else
// kLargeVectors. x86 timing machines: K = 4 from 2^12 on, K = 2 at 2^10 (Terms' scalar setup
// grows with K). Neoverse V2: K = 4 from 2^10 on. notes/106-r1.md.
#if defined(FPS_CHIRP_VECTORS)
inline constexpr int kSmallVectors = FPS_CHIRP_VECTORS, kLargeVectors = FPS_CHIRP_VECTORS;
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr int kSmallVectors = 4, kLargeVectors = 4;
#else
inline constexpr int kSmallVectors = 2, kLargeVectors = 4;
#endif
inline constexpr std::size_t kLargeSize = 1 << 11;

// products_lanes: vectors per step, and the size from which products() runs it. K = 2 on the
// x86 timing machines (2^20: 1.0-1.2x faster than K = 1 or 4), K = 4 on Neoverse V2 from 2^12 on
// (2^20: 1.22x faster than K = 2).
#if defined(FPS_CHIRP_LANES_VECTORS)
inline constexpr int kLanesVectors = FPS_CHIRP_LANES_VECTORS;
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr int kLanesVectors = 4;
#else
inline constexpr int kLanesVectors = 2;
#endif
inline constexpr std::size_t kLanesMinSize = 1024;

}  // namespace fps::chirp
