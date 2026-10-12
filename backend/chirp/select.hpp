// The chirp kernel's choices per target: the struct of its passes and the vectors per step of
// Terms (K), and of products_lanes (scan.hpp). -DFPS_CHIRP_VECTORS=<K> and
// -DFPS_CHIRP_LANES_VECTORS=<K> force them. Measurements behind the choices: notes.md.
#pragma once

#include <cstddef>

#include "backend/simd/select.hpp"

namespace fps::chirp {

using Wide = simd::Native;

#if defined(FPS_CHIRP_VECTORS)
inline constexpr int kVectors = FPS_CHIRP_VECTORS;
#else
inline constexpr int kVectors = 4;
#endif

// products_lanes: vectors per step, and the size from which products() runs it.
#if defined(FPS_CHIRP_LANES_VECTORS)
inline constexpr int kLanesVectors = FPS_CHIRP_LANES_VECTORS;
#else
inline constexpr int kLanesVectors = 2;
#endif
inline constexpr std::size_t kLanesMinSize = 1024;

}  // namespace fps::chirp
