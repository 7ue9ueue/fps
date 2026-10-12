// The chirp kernel's choices per target: the struct of its passes and the vectors per step of
// Terms (K), and of products_lanes (scan.hpp). -DFPS_CHIRP_VECTORS=<K> and
// -DFPS_CHIRP_LANES_VECTORS=<K> force them. Measurements behind the choices: notes.md.
#pragma once

#include <cstddef>

#include "backend/simd/select.hpp"

namespace fps::chirp {

using Wide = simd::Native;

// Terms per step for chirp and multiply_chirp: kSmallVectors below kLargeSize terms, else
// kLargeVectors. K = 8 is the fastest from 2^16 on every x86 timing machine, K = 2 at 2^10
// (Terms' scalar setup grows with K); notes/106-r1.md.
#if defined(FPS_CHIRP_VECTORS)
inline constexpr int kSmallVectors = FPS_CHIRP_VECTORS, kLargeVectors = FPS_CHIRP_VECTORS;
#else
inline constexpr int kSmallVectors = 2, kLargeVectors = 8;
#endif
inline constexpr std::size_t kLargeSize = 1 << 13;

// products_lanes: vectors per step, and the size from which products() runs it.
#if defined(FPS_CHIRP_LANES_VECTORS)
inline constexpr int kLanesVectors = FPS_CHIRP_LANES_VECTORS;
#else
inline constexpr int kLanesVectors = 2;
#endif
inline constexpr std::size_t kLanesMinSize = 1024;

// interpolate_geometric's two products: ntt::multiply each (6 transforms), or ntt::Transform in the
// values form with B's transform shared (5). -DFPS_CHIRP_INTERPOLATION=<variant> forces one.
enum class Products { multiply, transform };
#if defined(FPS_CHIRP_INTERPOLATION)
inline constexpr Products kInterpolationProducts = Products::FPS_CHIRP_INTERPOLATION;
#else
inline constexpr Products kInterpolationProducts = Products::multiply;
#endif

}  // namespace fps::chirp
